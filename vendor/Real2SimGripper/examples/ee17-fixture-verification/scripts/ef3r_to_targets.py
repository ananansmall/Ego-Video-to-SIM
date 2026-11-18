#!/usr/bin/env python3
"""ef3r_to_targets.py — EgoFound3R 输出 → Real2SimGripper 大范围后端的 targets npz。

EgoFound3R (mano_ws/EgoFound3R, arXiv 2610.01210) 对 ego 视频输出世界系米制
MANO 21 关节 + 相机 c2w + pick/place 状态标签。与 HaWoR→hawor_to_traj 链相比:
  - 场景相对运动 (人走动分解进相机轨迹, 不再混入手部轨迹) —— pick-place 需要;
  - 自带米制尺度 (hand_depth_scale 标定), 无 SLAM 单目尺度漂移 (456483158 实测
    HaWoR+SLAM 手腕世界幅度 1.97m vs EF3R 0.33m, SLAM 尺度可疑偏大 ~3x);
  - 自带状态分段 (PICK/LIFT/HOLD/PLACE/RELEASED), 可替代管线的启发式抓取帧检测。

坐标链:
  EF3R 世界 (y-up, 原点=anchor 帧相机中心, 实测 LIFT 段 +y 上抬)
    → M=(x, -z, y) 轴变换 → sim 世界 (z-up, x右/y前/z上, 与 chairbottle 链一致)
  旋转矩阵同步 R_sim = M @ R_ef @ M.T; 相机 c2w 同变换 (OpenCV 相机系两边一致)。

夹爪位姿: 复用 cb_svd_retarget.svd_gripper_pose (MANO j0/j4/j8 三点 SVD,
关节序与 EF3R meta.json 文档化的完全一致)。

输出: <out>/ef3r_right_targets.npz
  pos/rot/grip/valid (同 cb_right_targets) + R_c2w/t_c2w (已转到 sim 世界系)
"""
import argparse
import os
import sys

import numpy as np

# svd_gripper_pose 与 SLAM 段对齐复用 chairbottle 链的实现 (_scratch/cb_svd_retarget.py)
sys.path.insert(0, "/path/to/workspace/_scratch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from cb_svd_retarget import svd_gripper_pose
except ImportError:
    raise SystemExit("需要 _scratch/cb_svd_retarget.py (svd_gripper_pose) 在 python 路径上")            # noqa: E402
from large_range_track import realign_slam_segments     # noqa: E402  通用内核复用
import gravity_align as ga                              # noqa: E402  重力对齐 M'

# EF3R 世界 (x右/y上/z后, 右手) -> sim 世界 (x右/y前/z上): 正交右手系, det=+1
_M = np.array([[1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0],
               [0.0, 1.0, 0.0]], np.float64)


def _to_sim_R(R):
    return _M @ R @ _M.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef3r-dir", required=True, help="EgoFound3R outputs/<clip> 目录")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--out-dir", default=None, help="默认 <ef3r-dir>")
    args = ap.parse_args()
    out_dir = args.out_dir or args.ef3r_dir
    h = 1 if args.side == "right" else 0

    J = np.load(os.path.join(args.ef3r_dir, "hands", "joint_xyz_world.npy"))  # (N,2,21,3) 米
    val = np.load(os.path.join(args.ef3r_dir, "hands", "valid.npy"))          # (N,2)
    tc = np.load(os.path.join(args.ef3r_dir, "camera", "traj_c2w.npy"))       # (N,4,4) c2w
    N = J.shape[0]
    print(f"EF3R {os.path.basename(args.ef3r_dir)} {args.side}手: {N} 帧, "
          f"有效 {int(val[:, h].sum())}/{N}")

    M_sim = ga.sim_rotation(args.ef3r_dir)
    if M_sim is not None:
        print(f"  EF3R→sim: 重力对齐 M_sim (gravity_align)")
    else:
        M_sim = _M
        print("  EF3R→sim: 旧 _M (无 gravity 数据)")
    jw = J[:, h]                                   # (N,21,3) EF3R 世界系
    jw = np.einsum("ij,nkj->nki", M_sim, jw)       # → sim 世界 (重力对齐 z-up)
    valid = val[:, h].copy()

    # EF3R 是端到端模型, 无"相机系关节全零"问题; 但保留 NaN 帧防护 (与 cb 链一致)
    bad = ~np.isfinite(jw).all(axis=(1, 2))
    if bad.any():
        print(f"  ⚠ 非有限帧 {int(bad.sum())} 个, 用最近有效帧补齐")
        good = np.where(~bad)[0]
        for k in np.where(bad)[0]:
            jw[k] = jw[good[np.argmin(np.abs(good - k))]]
            valid[k] = valid[good[np.argmin(np.abs(good - k))]]

    pos = np.zeros((N, 3)); rot = np.zeros((N, 3, 3)); grip = np.zeros((N, 2))
    for k in range(N):
        p, R, jv = svd_gripper_pose(jw[k, 0], jw[k, 4], jw[k, 8], args.side)
        pos[k], rot[k], grip[k] = p, R, [jv, jv]

    # 相机 c2w → sim 世界 (FPV 用; 当前 follow 后端的机载 FPV 实际不读它, 保留以备用)
    Rc_sim = np.einsum("ij,njk->nik", M_sim, tc[:, :3, :3])
    tc_sim = (np.einsum("ij,nj->ni", M_sim, tc[:, :3, 3]))
    tc_sim, _jumps = realign_slam_segments(tc_sim)
    print(f"  SLAM 段对齐: 跳变帧 {_jumps}")

    def _geo(a, b):
        c = (np.trace(a.T @ b) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    dd = np.array([_geo(rot[k - 1], rot[k]) for k in range(1, N) if valid[k] and valid[k - 1]])
    print(f"  姿态帧间跳变: max {dd.max():.1f}°, >30° 帧 {int((dd > 30).sum())}")
    for axis, nm in enumerate("xyz"):
        v = pos[valid, axis]
        print(f"  夹爪 pos {nm}: {v.min():.2f}~{v.max():.2f} m (幅度 {np.ptp(v):.2f})")
    print(f"  grip 开合: {grip[valid,0].min():.4f}~{grip[valid,0].max():.4f} m")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "ef3r_right_targets.npz" if args.side == "right"
                       else "ef3r_left_targets.npz")
    extra = {}
    # 状态标签 (EgoFound3R state_labels.json, 如 PICK/LIFT/HOLD/PLACE/RELEASED):
    # 转成逐帧整数标签存入 npz, 供 cb_large_range_sim 的 grip_mode="states"
    # 离散夹爪控制使用 (闭合@LIFT 起, 张开@RELEASED 起)。
    sl = os.path.join(args.ef3r_dir, "state_labels.json")
    if os.path.exists(sl):
        import json
        s = json.load(open(sl))["states"]
        labels = np.zeros(N, np.int64)
        names = ["APPROACH"]
        for i, (nm, (a, b)) in enumerate(s.items()):
            labels[a:min(b, N)] = i + 1
            names.append(nm)
        extra = {"grip_states": labels, "grip_state_names": np.array(names)}
        print(f"  状态标签: {names[1:]} → 逐帧标签 (APPROACH {int((labels==0).sum())} 帧)")
    np.savez(out, pos=pos, rot=rot, grip=grip, valid=valid, R_c2w=Rc_sim, t_c2w=tc_sim,
             **extra)
    print(f"  ✓ 已保存 {out}")


if __name__ == "__main__":
    main()
