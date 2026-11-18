#!/usr/bin/env python3
"""hawor_to_traj.py — 把一份 HaWoR 输出目录转成 Real2SimGripper 要的轨迹 npz。

适用目录结构 (HaWoR demov2 / demo-vggt-omega 产出):
    tracks_*/model_joints.npy   (2,N,21,3)  相机系 21 关节 (未除 SLAM 尺度)
    SLAM/*.npz                  traj(N,7)=[tx,ty,tz,qx,qy,qz,qw] 相机 c2w + scale
    world_space_res.pth         joblib 存的 [trans,rot,hand_pose,betas,valid]
    cam_space/<hand_id>/...     MANO 参数 (相机系)
    extracted_images/           原始抽帧

坐标链 (与 chairbottle 一致, 但这里显式处理尺度与竖直轴):
    p_cam(未缩放) --R_c2w,t_c2w--> SLAM 世界(未缩放) --/scale--> 米
    --> R_zu (首帧相机系: x右 / y下 / z前 -> x右 / y前 / z上) --> sim 世界

输出 npz (供 cb_large_range_sim.py 等任务脚本直接吃):
    pos (N,3) / rot (N,3,3) / grip (N,2) / valid (N,)
    R_c2w (N,3,3) / t_c2w (N,3)   已转到 sim 世界系, 供第一视角使用
      (约定同 OpenCV: R_c2w@[0,0,1]=光轴, R_c2w@[0,-1,0]=画面上方, 但都在 sim 系下)
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import large_range_track as lrt   # noqa: E402

# 首帧相机系 (x右 / y下 / z前) -> sim 世界 (x右 / y前 / z上)
R_ZU = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0],
                 [0.0, -1.0, 0.0]])


def _quat_to_R(q):
    """[qx,qy,qz,qw] -> 旋转矩阵 (HaWoR traj 的四元数顺序)。"""
    x, y, z, w = np.asarray(q, float) / max(np.linalg.norm(q), 1e-9)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _geo(a, b):
    c = (np.trace(a.T @ b) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def repair_cam_rot(Rs, thr_deg=15.0):
    """修复 SLAM 相机旋转的单帧狂跳 (实测本序列 F111 单帧 34.9°)。

    单帧 >15°/帧 (30fps 下相当于 450°/s) 不可能来自真实头动, 是跟踪丢帧。
    做法: 对跳变帧用前后帧 SLERP 插值替换 (只改这一帧, 不污染后续段)。
    """
    R = np.array(Rs, float, copy=True)
    bad = [k for k in range(1, len(R)) if _geo(R[k - 1], R[k]) > thr_deg]
    for k in bad:
        if 0 < k < len(R) - 1:
            R[k] = lrt._slerp(R[k - 1], R[k + 1], 0.5)
    return R, bad


def load(seq_dir, hand="right"):
    """读 joints / c2w / scale / valid。返回 (joints_cam(N,21,3), Rs, ts, scale, valid)。"""
    mj = sorted(glob.glob(os.path.join(seq_dir, "tracks_*", "model_joints.npy")))
    if not mj:
        raise FileNotFoundError(f"{seq_dir} 下没有 tracks_*/model_joints.npy")
    J = np.load(mj[0])
    idx = 1 if hand == "right" else 0
    joints_cam = J[idx].astype(float)

    sl = sorted(glob.glob(os.path.join(seq_dir, "SLAM", "*.npz")))
    if not sl:
        raise FileNotFoundError(f"{seq_dir} 下没有 SLAM/*.npz")
    z = np.load(sl[0])
    traj = np.asarray(z["traj"], float)
    scale = float(z["scale"])
    Rs = np.array([_quat_to_R(traj[k, 3:7]) for k in range(len(traj))])
    ts = traj[:, :3].copy()

    valid = np.ones(len(joints_cam), bool)
    ws = os.path.join(seq_dir, "world_space_res.pth")
    if os.path.exists(ws):
        try:
            import joblib
            obj = joblib.load(ws)
            v = np.asarray(obj[4][idx], bool)
            n = min(len(v), len(valid))
            valid[:n] = v[:n]
        except Exception as e:                       # 不是 joblib 格式就当没有
            print(f"  [warn] world_space_res.pth 读取失败 ({e}), 只用关节零值判有效帧")

    n = min(len(joints_cam), len(Rs), len(valid))
    return joints_cam[:n], Rs[:n], ts[:n], scale, valid[:n]


def main():
    ap = argparse.ArgumentParser(description="HaWoR 输出目录 -> 夹爪轨迹 npz")
    ap.add_argument("--dir", required=True, help="HaWoR 输出目录 (含 tracks_*/SLAM/)")
    ap.add_argument("--hand", choices=["left", "right"], default="right")
    ap.add_argument("--out", required=True, help="输出 npz 路径")
    ap.add_argument("--no-zup", action="store_true",
                    help="不转到 z-up sim 系 (保持首帧相机系 x右/y下/z前)")
    a = ap.parse_args()

    joints_cam, Rs, ts, scale, valid = load(a.dir, a.hand)
    N = len(joints_cam)
    print(f"HaWoR 序列 {os.path.basename(a.dir.rstrip('/'))}: {N} 帧, {a.hand}, "
          f"SLAM scale={scale:.3f}")

    # ---- 有效性: pred_valid 之外, 关节全零的帧 (HaWoR 未重建) 也算无效, 先插值补齐 ----
    zero = np.all(np.abs(joints_cam.reshape(N, -1)) < 1e-12, axis=1)
    bad_valid = zero | ~valid
    good = np.where(~bad_valid)[0]
    print(f"  无效帧 (valid=False 或 关节全零): {int(bad_valid.sum())}")
    if len(good) == 0:
        raise RuntimeError("整段没有有效帧")
    if bad_valid.any():
        for j in range(21):
            for ax in range(3):
                joints_cam[:, j, ax] = np.interp(np.arange(N), good,
                                                 joints_cam[good, j, ax])

    # ---- 先换米, 再做 SLAM 修复 (realign 的阈值是米制, 用未缩放单位会误判) ----
    ts_m = ts / scale
    joints_m = joints_cam / scale
    ts_m, t_jumps = lrt.realign_slam_segments(ts_m)
    Rs, r_jumps = repair_cam_rot(Rs)
    print(f"  SLAM 修复: 平移跳变帧 {t_jumps} | 旋转跳变帧 {r_jumps}")

    # ---- 相机系 -> SLAM 世界 ----
    jw = np.einsum("nij,nkj->nki", Rs, joints_m) + ts_m[:, None, :]
    tc = ts_m

    if not a.no_zup:
        # 世界系换轴: p_sim = R_ZU @ p_slam -> 相机外参也必须左乘同一个 R_ZU
        # (右乘会变成"相机自身轴换轴", 首帧 R=I 时看起来没变, 但 ego 相机朝向全错)
        jw = np.einsum("ij,nkj->nki", R_ZU, jw)
        tc = tc @ R_ZU.T
        Rs = np.einsum("ij,njk->nik", R_ZU, Rs)

    # ---- 重定向: 腕 / 拇指尖 / 食指尖 -> 夹爪位姿 ----
    # off_scale=scale: 世界链路把相机系坐标整体 /scale (为了把 SLAM 单位换成米),
    # 但 MANO 网格的**局部几何本身已经是米制** (实测腕->食指根 88mm、腕->食指尖 147mm,
    # 与 MANO forward 逐关节一致)。不还原的话指间距 51mm 会被压到 3.8mm < 指基距
    # 26.9mm -> 指关节行程恒被 clamp 到 0 (夹爪全程闭合), gripper_link 还会偏到腕后。
    pos = np.zeros((N, 3)); rot = np.zeros((N, 3, 3)); grip = np.zeros((N, 2))
    for k in range(N):
        p, R, jv = lrt.mano_gripper_pose(jw[k, 0], jw[k, 4], jw[k, 8], a.hand,
                                         off_scale=scale)
        pos[k], rot[k], grip[k] = p, R, [jv, jv]

    # ---- 质检 ----
    dd = np.array([_geo(rot[k - 1], rot[k]) for k in range(1, N)])
    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    print(f"  夹爪 pos: " + "  ".join(
        f"{nm} {pos[:, i].min():.2f}~{pos[:, i].max():.2f}m" for i, nm in enumerate("xyz")))
    print(f"  帧间: 位置中位 {np.median(dp)*1000:.1f}mm max {dp.max()*1000:.1f}mm | "
          f"姿态中位 {np.median(dd):.1f}° max {dd.max():.1f}°")
    print(f"  夹爪开合: {grip[:, 0].min():.4f}~{grip[:, 0].max():.4f} m")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, pos=pos, rot=rot, grip=grip, valid=~bad_valid,
             R_c2w=Rs, t_c2w=tc)
    print(f"  ✓ {a.out}")


if __name__ == "__main__":
    main()
