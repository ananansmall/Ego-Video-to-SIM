#!/usr/bin/env python3
"""agile_to_targets.py — AGILE 物理优化手-物交互 → Real2SimGripper targets npz。

链路定位 (456483158 三段式):
  ① EgoFound3R  手+物体 mesh/点云、交互物体判断、世界系相机轨迹  (已完成)
  ② AGILE       单手交互物理优化 (视频→ACE 手部估计→MoGe 深度→SAM3D 物体网格
                →FoundationPose 位姿→逐帧接触/遮挡优化), 修复 EF3R 手-物相对
                几何不物理的问题 (EF3R HOLD 段手-物 159±98mm, AGILE 约束到接触)
  ③ 本脚本      AGILE MANO 轨迹 (相机系, OpenCV, 米) → EF3R 世界系 → sim z-up
                targets npz, 供 cb_large_range_sim 跟随抓取 (重定向+跟随)

坐标链 (与 ef3r_to_targets.py 完全同构):
  AGILE 相机系关节 (OpenCV, 米, 与 EF3R 同一视频同一相机)
    → EF3R c2w[video_frame] 组合 → EF3R 世界 (y-up, 原点=anchor 帧相机中心)
    → M=(x,-z,y) 轴变换 → sim 世界 (z-up, x右/y前/z上)
  旋转 R_sim = M @ R_ef @ M.T。EF3R 世界系与 AGILE 的 MoGe 米制尺度都在同一
  视频上自标定, 残差 (~10% 内) 由 follow 后端的 fit_height / 底盘跟随吸收。

时间轴: AGILE 链以 --subsample (默认 3) 从 30fps 视频抽帧 (10fps, 100 帧)。
输出两份: 原生 F 帧 npz + 上采样回视频 300 帧的 npz (线性 pos / SLERP rot /
线性 grip), 与 EF3R 链 (ef3r_right_targets.npz, 300 帧) 同时间轴可逐帧对比。

夹爪位姿: 复用 cb_svd_retarget.svd_gripper_pose (MANO j0/j4/j8 三点 SVD,
与 EF3R 链同一重定向算子 → 差异只来自上游手部轨迹来源)。

状态标签: 复用 EF3R 的 state_labels.json (PICK/LIFT/HOLD/PLACE/RELEASED,
300 帧时间轴) —— 两条链共用同一任务分段, grip_mode="states" 可直接用。

输出: <agile-dir>/agile_right_targets.npz        (F 帧, 原生)
      <agile-dir>/agile_right_targets_300.npz    (300 帧, 视频时间轴)
  字段与 ef3r_right_targets.npz 一致: pos/rot/grip/valid/R_c2w/t_c2w
  + grip_states/grip_state_names。
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/path/to/workspace/_scratch")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from cb_svd_retarget import svd_gripper_pose
except ImportError:
    raise SystemExit("需要 _scratch/cb_svd_retarget.py (svd_gripper_pose) 在 python 路径上")  # noqa: E402
from large_range_track import realign_slam_segments  # noqa: E402

# EF3R 世界 (x右/y上/z后, 右手) -> sim 世界 (x右/y前/z上): 正交右手系, det=+1
_M = np.array([[1.0, 0.0, 0.0],
               [0.0, 0.0, -1.0],
               [0.0, 1.0, 0.0]], np.float64)


def _to_sim_R(R):
    return _M @ R @ _M.T


def quat_slerp(q0, q1, t):
    """标量首插值 (等价于旋转矩阵逐帧球面插值)。q 为 (4,) wxyz。"""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 1.0 - 1e-9:
        q = q0 + t * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / s


def rotmat_slerp_seq(Rs, xq):
    """(N,3,3) 旋转序列 → 在锚点坐标 xq (锚点索引单位, 可非整数) 处分段 slerp 采样。"""
    from scipy.spatial.transform import Rotation as Rot
    qs = Rot.from_matrix(Rs).as_quat()  # (N,4) xyzw
    qs = qs[:, [3, 0, 1, 2]]            # → wxyz
    N = len(qs)
    out = np.zeros((len(xq), 3, 3))
    for i, x in enumerate(np.asarray(xq, np.float64)):
        k = min(int(np.floor(x)), N - 2)
        t = float(x - k)
        q = quat_slerp(qs[k], qs[k + 1], t)
        out[i] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agile-dir", required=True,
                    help="AGILE output/<seq> 目录 (含 runs/opt/aligned)")
    ap.add_argument("--ef3r-dir", required=True,
                    help="EgoFound3R outputs/<clip> 目录 (同一视频; 提供 c2w 与 state_labels)")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--subsample", type=int, default=3,
                    help="AGILE 预处理相对源视频的抽帧步长")
    ap.add_argument("--out-num", type=int, default=0,
                    help="上采样输出帧数 (默认 0=不输出上采样版; 456483158 用 300)")
    ap.add_argument("--video-fps", type=float, default=30.0)
    args = ap.parse_args()

    aligned = os.path.join(args.agile_dir, "runs", "opt", "aligned")
    Jc = np.load(os.path.join(aligned, "optimized_mano_joints.npz"))["mano_joints"]  # (F,1,21,3)
    Jc = Jc[:, 0].astype(np.float64)
    F = Jc.shape[0]
    cd = np.load(os.path.join(aligned, "camera_data.npz"))
    pred_ids = np.asarray(cd["prediction_frame_ids"], np.int64)
    vid = pred_ids * args.subsample          # 源视频帧号 (30fps 时间轴)
    obj_pose = np.load(os.path.join(aligned, "object_poses.npz"))
    T_co = obj_pose["object_poses_in_cam"].astype(np.float64)  # (F,4,4) 相机→物体
    obj_scale = float(np.asarray(obj_pose["object_scale"]).ravel()[0])
    print(f"AGILE {os.path.basename(args.agile_dir.rstrip('/'))} {args.side}手: "
          f"{F} 帧 (视频帧 {vid[0]}..{vid[-1]}, 抽帧 {args.subsample}x), 物体 scale {obj_scale:.4f}")

    # ---- EF3R 相机轨迹 (300 帧, EF3R 世界 y-up 米制) ----
    c2w = np.load(os.path.join(args.ef3r_dir, "camera", "traj_c2w.npy")).astype(np.float64)
    Nv = c2w.shape[0]
    assert vid.max() < Nv, f"AGILE 帧号 {vid.max()} 超出 EF3R 轨迹 {Nv} 帧"

    # ---- 相机系尺度锚定: 重力约束相似对齐 (替代自由 Umeyama) ----
    # 自由 Umeyama 把两链的坐标系差拟合成非物理旋转 (实测 R=18°, 顺带扭曲重力)。
    # 正确做法 (借鉴 Ego-Video-to-SIM 001 的重力对齐场景思想): 旋转的"竖直分量"由
    # 重力对齐决定 (n_cam→g_ef 最小旋转), 只留 1-DOF 偏航 + 尺度 + 平移交给手部
    # 对应点。自由 Umeyama 仍计算一份作残差对比诊断。
    import gravity_align as ga
    ginfo = ga.load_or_build(args.ef3r_dir) or ga.build_gravity_align(args.ef3r_dir)
    g_ef = np.asarray(ginfo["g_ef"], np.float64)
    M_sim = np.asarray(ginfo["M_sim"], np.float64)
    h = 1 if args.side == "right" else 0
    J_ef_cam = np.load(os.path.join(args.ef3r_dir, "hands", "joint_xyz_cam.npy"))[:, h].astype(np.float64)
    Pa = Jc.reshape(-1, 3)                    # (F*21, 3) AGILE 相机系点云 (源)
    Pb = J_ef_cam[vid].reshape(-1, 3)         # (F*21, 3) EF3R 相机系点云 (目标)

    def _fit_fixed_R(R):
        """固定旋转的闭式 (s,t) + 残差。"""
        Ra = Pa @ R.T
        ca, cb = Ra.mean(0), Pb.mean(0)
        num = float(((Ra - ca) * (Pb - cb)).sum())
        den = float(((Ra - ca) ** 2).sum())
        s = num / den if den > 1e-12 else 1.0
        t = cb - s * ca
        res = np.linalg.norm(s * (Ra) + t - Pb, axis=1)
        return s, t, res

    # 自由 Umeyama (诊断对照)
    ca0, cb0 = Pa.mean(0), Pb.mean(0)
    H0 = (Pb - cb0).T @ (Pa - ca0) / len(Pa)
    U0, S0, Vt0 = np.linalg.svd(H0)
    d0 = np.sign(np.linalg.det(U0 @ Vt0))
    R_free = U0 @ np.diag([1.0, 1.0, d0]) @ Vt0
    s_free = float((np.asarray(S0) * np.array([1.0, 1.0, d0])).sum()
                   / max(float(((Pa - ca0) ** 2).sum() / len(Pa)), 1e-12))
    t_free = cb0 - s_free * (R_free @ ca0)
    res_free = np.linalg.norm((s_free * (Jc @ R_free.T) + t_free) - J_ef_cam[vid], axis=2)

    # 重力约束: R_base = n_cam → g_ef 最小旋转; 偏航 = 绕 g_ef 的 1-DOF 搜索
    # n_cam: 世界系重力经 EF3R c2w 拉回相机系 (方向级, 多帧平均)
    n_cam = np.mean([c2w[k, :3, :3].T @ g_ef for k in vid], 0)
    n_cam /= np.linalg.norm(n_cam)
    R_base = ga._rodrigues_toward(n_cam, g_ef)
    up = g_ef / np.linalg.norm(g_ef)

    def _axis_rot(axis, th):
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

    best = None
    for th in np.radians(np.arange(-180, 180, 5.0)):
        s_, t_, r_ = _fit_fixed_R(_axis_rot(up, th) @ R_base)
        if best is None or r_.mean() < best[0]:
            best = (r_.mean(), th)
    th0 = best[1]
    for th in np.arange(th0 - np.radians(5), th0 + np.radians(5), np.radians(0.5)):
        s_, t_, r_ = _fit_fixed_R(_axis_rot(up, th) @ R_base)
        if r_.mean() < best[0]:
            best = (r_.mean(), th)
    th = best[1]
    R_al = _axis_rot(up, th) @ R_base
    s_al, t_al, res = _fit_fixed_R(R_al)
    res = res.reshape(F, 21)
    ang = np.degrees(np.arccos(np.clip((np.trace(R_al) - 1) / 2, -1, 1)))
    tilt_free = float(np.degrees(np.arccos(np.clip((R_free @ n_cam) @ g_ef, -1, 1))))
    print(f"  尺度锚定 (重力约束): n_cam {np.round(n_cam,3)} → g_ef {np.round(g_ef,3)} | "
          f"偏航 {np.degrees(th):+.1f}° s={s_al:.3f} | 残差 mean {res.mean()*1000:.1f}mm "
          f"p95 {np.percentile(res,95)*1000:.1f}mm")
    print(f"    (对照 自由 Umeyama: s={s_free:.3f} 残差 {res_free.mean()*1000:.1f}mm | "
          f"其旋转使重力偏 {tilt_free:.1f}° —— 重力约束消除了这一非物理分量)")
    Jc = s_al * (Jc @ R_al.T) + t_al          # → EF3R 相机系度量
    D = np.eye(4); D[:3, :3] = s_al * R_al; D[:3, 3] = t_al
    T_co = np.einsum("ij,fjk->fik", D, T_co)  # 物体位姿同变换 (obj→cam 左乘相似)
    np.savez(os.path.join(args.agile_dir, "cam_similarity_to_ef3r.npz"),
             s=s_al, R=R_al, t=t_al, yaw_deg=np.degrees(th),
             g_ef=g_ef, n_cam=n_cam,
             resid_mean_mm=float(res.mean() * 1000),
             resid_free_umeyama_mm=float(res_free.mean() * 1000))

    # ---- 相机系 → EF3R 世界 → sim z-up (M_sim = M @ R_g, 重力对齐) ----
    Rw = c2w[vid, :3, :3]                       # (F,3,3)
    tw = c2w[vid, :3, 3]                        # (F,3)
    jw_ef = np.einsum("fij,fkj->fki", Rw, Jc) + tw[:, None, :]   # EF3R 世界
    jw = np.einsum("ij,fnj->fni", M_sim, jw_ef)                  # sim 世界 (重力对齐 z-up)

    bad = ~np.isfinite(jw).all(axis=(1, 2))
    valid = np.ones(F, bool)
    if bad.any():
        print(f"  ⚠ 非有限帧 {int(bad.sum())} 个, 用最近有效帧补齐")
        good = np.where(~bad)[0]
        for k in np.where(bad)[0]:
            jw[k] = jw[good[np.argmin(np.abs(good - k))]]
    jw = np.clip(jw, -50, 50)  # 数值防护

    # ---- SVD 重定向 (与 EF3R 链同一算子) ----
    pos = np.zeros((F, 3)); rot = np.zeros((F, 3, 3)); grip = np.zeros((F, 2))
    for k in range(F):
        p, R, jv = svd_gripper_pose(jw[k, 0], jw[k, 4], jw[k, 8], args.side)
        pos[k], rot[k], grip[k] = p, R, [jv, jv]

    # ---- 相机 c2w → sim (与 ef3r_to_targets 同链, FPV 备用) ----
    Rc_sim = np.einsum("ij,fjk->fik", M_sim, Rw)
    tc_sim = np.einsum("ij,fj->fi", M_sim, tw)
    tc_sim, _jumps = realign_slam_segments(tc_sim)
    print(f"  SLAM 段对齐: 跳变帧 {_jumps}")

    # ---- 物理一致性诊断: HOLD 段 手掌质心-物体中心 距离 (相机系, 不受世界系影响) ----
    sl_p = os.path.join(args.ef3r_dir, "state_labels.json")
    states = {}
    if os.path.exists(sl_p):
        states = json.load(open(sl_p))["states"]
    obj_c = T_co[:, :3, 3]
    palm = Jc[:, [0, 5, 9, 13, 17]].mean(1)
    d_ho = np.linalg.norm(obj_c - palm, axis=1)
    print("  手掌-物体距离 (相机系, AGILE 优化后应稳定在接触量级):")
    for nm in ("PICK", "LIFT", "HOLD", "PLACE", "RELEASED"):
        if nm in states:
            a, b = states[nm]
            sel = (vid >= a) & (vid < min(b, Nv))
            if sel.any():
                print(f"    {nm:9s}: mean {d_ho[sel].mean()*1000:6.1f}mm  "
                      f"std {d_ho[sel].std()*1000:6.1f}mm  ({int(sel.sum())} 帧)")
    if "HOLD" in states:
        a, b = states["HOLD"]
        sel = (vid >= a) & (vid < min(b, Nv))
        if sel.any():
            print(f"    (EF3R 直连 HOLD 段参考: 159±98mm —— AGILE 接触优化后应显著更小更稳)")

    # ---- 姿态/范围诊断 (与 ef3r_to_targets 相同口径) ----
    def _geo(a, b):
        c = (np.trace(a.T @ b) - 1.0) / 2.0
        return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    dd = np.array([_geo(rot[k - 1], rot[k]) for k in range(1, F)])
    print(f"  姿态帧间跳变: max {dd.max():.1f}°, >30° 帧 {int((dd > 30).sum())}")
    for axis, nm in enumerate("xyz"):
        v = pos[:, axis]
        print(f"  夹爪 pos {nm}: {v.min():.2f}~{v.max():.2f} m (幅度 {np.ptp(v):.2f})")
    print(f"  grip 开合: {grip[:,0].min():.4f}~{grip[:,0].max():.4f} m")

    # ---- 状态标签 (EF3R 300 帧时间轴 → AGILE 帧采样) ----
    def labels_at(frames):
        labels = np.zeros(len(frames), np.int64)
        names = ["APPROACH"]
        for i, (nm, (a, b)) in enumerate(states.items()):
            labels[(frames >= a) & (frames < min(b, Nv))] = i + 1
            names.append(nm)
        return labels, names

    labels_native, names = labels_at(vid)

    def save(path, pos_, rot_, grip_, valid_, labels_, n_):
        np.savez(path, pos=pos_, rot=rot_, grip=grip_, valid=valid_,
                 R_c2w=Rc_sim if len(pos_) == F else None, t_c2w=tc_sim if len(pos_) == F else None,
                 grip_states=labels_, grip_state_names=np.array(names[:len(names)]),
                 src_frames=vid, video_fps=args.video_fps)
        print(f"  ✓ 已保存 {path} ({n_} 帧)")

    out_native = os.path.join(args.agile_dir, f"agile_{args.side}_targets.npz")
    save(out_native, pos, rot, grip, valid, labels_native, F)

    # ---- 上采样到视频时间轴 (out_num=300 同 EF3R 链; 900=rt3 3x 重定时) ----
    # 锚点只覆盖到 vid[-1] (AGILE prep 100 帧×3 = 视频 0..285): 超出部分**截短**
    # 而不是钳位 —— 钳位会让目标轨迹在末尾冻结 0.47s (实测 F285-299 全零速度,
    # GIF 里目标和机器人集体静止 15 帧)。
    if args.out_num > 0:
        # 重定时语义: out_num 帧覆盖**同一运动区间** [0, vid[-1]] —— out_num=300
        # ≈1:1, out_num=900 = 3× 慢放 (duck time_scale 同款, 给 PD 力爬坡留时间)。
        # 尾部不钳位: f_vid 最大 = vid[-1] (锚点覆盖到哪, 轨迹到哪)。
        f_out = np.arange(args.out_num)
        f_vid = f_out * (float(vid[-1]) / max(args.out_num - 1, 1))
        pos_u = np.stack([np.interp(f_vid, vid, pos[:, a]) for a in range(3)], 1)
        _sp = float(np.median(np.diff(vid)))     # 锚点间隔 (视频帧)
        rot_u = rotmat_slerp_seq(rot, f_vid / _sp)
        grip_u = np.stack([np.interp(f_vid, vid, grip[:, a]) for a in range(2)], 1)
        labels_u, _ = labels_at(f_vid.astype(np.int64))
        Rw_u = np.stack([c2w[min(f, Nv - 1)][:3, :3] for f in f_vid.astype(np.int64)])
        tw_u = np.stack([c2w[min(f, Nv - 1)][:3, 3] for f in f_vid.astype(np.int64)])
        Rc_u = np.einsum("ij,fjk->fik", M_sim, Rw_u)
        tc_u = np.einsum("ij,fj->fi", M_sim, tw_u)
        tc_u, _j2 = realign_slam_segments(tc_u)
        out_u = os.path.join(args.agile_dir, f"agile_{args.side}_targets_{args.out_num}.npz")
        np.savez(out_u, pos=pos_u, rot=rot_u, grip=grip_u, valid=np.ones(args.out_num, bool),
                 R_c2w=Rc_u, t_c2w=tc_u, grip_states=labels_u,
                 grip_state_names=np.array(names), video_fps=args.video_fps)
        print(f"  ✓ 已保存 {out_u} ({args.out_num} 帧, 上采样)")


if __name__ == "__main__":
    main()
