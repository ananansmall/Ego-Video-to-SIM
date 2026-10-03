#!/usr/bin/env python3
"""
check_slam_traj2.py — 修正四元数解析后重新分析 SLAM vs 重建 vs RAS
"""

import numpy as np
import os
import glob

EXAMPLES = {
    '121': {
        'slam_npz': '/home/an/data/hawor/121_C5_CellPhone_161deg/SLAM/hawor_slam_w_scale_0_600.npz',
        'recon_npz': '/home/an/data/hawor/121_C5_CellPhone_161deg/reconstruction/hawor_results_0_600.npz',
        'ras_dir': '/home/an/data/ras/121_C5_CellPhone_161deg_vggt_omega',
    },
    '7': {
        'slam_npz': '/home/an/data/hawor/7/SLAM/hawor_slam_w_scale_0_113.npz',
        'recon_npz': '/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz',
        'ras_dir': '/home/an/data/ras/7_vggt_omega',
    },
}

Rx_hand = np.diag([1, -1, -1])


def load_ras_extrinsics(ras_dir):
    ext_dir = os.path.join(ras_dir, 'extrinsics')
    files = sorted(glob.glob(os.path.join(ext_dir, '*.txt')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
    R_list, t_list = [], []
    for f in files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        R_c2w = R_w2c.T
        t_list.append(-R_c2w @ t_w2c)
        R_list.append(R_c2w)
    return np.array(R_list), np.array(t_list)


def quat_to_rotmat(q):
    """四元数 [qx, qy, qz, qw] → 旋转矩阵 (3,3)"""
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


def pca_angle(pts_a, pts_b):
    a_c = pts_a - pts_a.mean(axis=0)
    b_c = pts_b - pts_b.mean(axis=0)
    _, _, Vt_a = np.linalg.svd(a_c, full_matrices=False)
    _, _, Vt_b = np.linalg.svd(b_c, full_matrices=False)
    dot = np.clip(np.dot(Vt_a[0], Vt_b[0]), -1, 1)
    if dot < 0:
        dot = -dot
    return np.degrees(np.arccos(dot))


def umeyama_full(src, dst):
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    sigma_src = np.sqrt(np.mean(np.sum(src_c ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_c ** 2, axis=1)))
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    s = sigma_dst / sigma_src if sigma_src > 1e-10 else 1.0
    t = dst_mean - s * R @ src_mean
    return R, s, t


for name, paths in EXAMPLES.items():
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # 加载 SLAM: traj = [x, y, z, qx, qy, qz, qw]
    slam = dict(np.load(paths['slam_npz'], allow_pickle=True))
    traj = slam['traj']
    t_slam = traj[:, :3]
    q_slam = traj[:, 3:7]  # [qx, qy, qz, qw]
    R_slam = np.array([quat_to_rotmat(q_slam[i]) for i in range(len(q_slam))])
    print(f"  SLAM: {len(t_slam)} 帧, 第一帧 R_c2w[0]:\n{R_slam[0]}")
    print(f"  det(R_slam[0]): {np.linalg.det(R_slam[0]):.6f}")

    # 检查: 第一帧 R_c2w ≈ I ?
    eye_diff = np.linalg.norm(R_slam[0] - np.eye(3))
    print(f"  R_slam[0] 与 I 的差异: {eye_diff:.6f}")

    # 加载重建
    recon = dict(np.load(paths['recon_npz'], allow_pickle=True))
    R_recon = recon['R_c2w']
    t_recon = recon['t_c2w']
    print(f"  重建: {len(t_recon)} 帧, 第一帧 R_c2w[0]:\n{R_recon[0]}")
    print(f"  det(R_recon[0]): {np.linalg.det(R_recon[0]):.6f}")
    recon_eye_diff = np.linalg.norm(R_recon[0] - np.eye(3))
    print(f"  R_recon[0] 与 I 的差异: {recon_eye_diff:.6f}")

    # 验证: 重建 R_c2w = diag([1,-1,-1]) @ R_slam_w2c ?
    R_pred = Rx_hand @ R_slam[0].T  # Rx_hand @ R_slam_w2c[0]
    diff = np.linalg.norm(R_pred - R_recon[0])
    print(f"\n  Rx_hand @ R_slam_w2c[0] vs R_recon[0]: 差异={diff:.6f}")
    if diff < 0.01:
        print(f"  ✓ 确认: 重建 R_c2w = diag([1,-1,-1]) @ R_slam_w2c")

    # 或: 重建 R_c2w = R_slam_c2w @ Rx_hand ?
    R_pred2 = R_slam[0] @ Rx_hand
    diff2 = np.linalg.norm(R_pred2 - R_recon[0])
    print(f"  R_slam_c2w @ Rx_hand vs R_recon[0]: 差异={diff2:.6f}")
    if diff2 < 0.01:
        print(f"  ✓ 确认: 重建 R_c2w = R_slam_c2w @ Rx_hand")

    # 或: 重建 R_c2w = Rx_hand @ R_slam_c2w ?
    R_pred3 = Rx_hand @ R_slam[0]
    diff3 = np.linalg.norm(R_pred3 - R_recon[0])
    print(f"  Rx_hand @ R_slam_c2w vs R_recon[0]: 差异={diff3:.6f}")
    if diff3 < 0.01:
        print(f"  ✓ 确认: 重建 R_c2w = Rx_hand @ R_slam_c2w")

    # 或: 重建 R_c2w ≈ R_slam_c2w ?
    diff4 = np.linalg.norm(R_slam[0] - R_recon[0])
    print(f"  R_slam_c2w vs R_recon[0]: 差异={diff4:.6f}")
    if diff4 < 0.01:
        print(f"  ✓ 确认: 重建 R_c2w ≈ R_slam_c2w")

    # 加载 RAS
    R_r, t_r = load_ras_extrinsics(paths['ras_dir'])
    print(f"\n  RAS: {len(t_r)} 帧")
    n_ras = len(t_r)
    n_ha = len(t_recon)
    indices = np.linspace(0, n_ha - 1, n_ras, dtype=int)

    # ===== cam0 空间分析 =====
    R_r0 = R_r[0]
    R_h0_recon = R_recon[0]  # 重建第一帧
    R_h0_slam = R_slam[0]    # SLAM 第一帧 (≈ I)

    ras_cam0 = np.array([R_r0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])

    # 重建 cam0: 用 R_c2w_hawor0.T
    recon_cam0 = np.array([R_h0_recon.T @ (t_recon[i] - t_recon[0]) for i in range(n_ha)])
    recon_cam0_s = recon_cam0[indices]

    # SLAM cam0: 用 R_slam0.T (≈ I)
    slam_cam0 = np.array([R_h0_slam.T @ (t_slam[i] - t_slam[0]) for i in range(n_ha)])
    slam_cam0_s = slam_cam0[indices]

    # SLAM cam0 用 Rx_hand 转换: 模拟重建的 Y/Z 翻转
    slam_cam0_rx = np.array([Rx_hand @ (R_h0_slam.T @ (t_slam[i] - t_slam[0])) for i in range(n_ha)])
    slam_cam0_rx_s = slam_cam0_rx[indices]

    # 轨迹范围
    print(f"\n  [轨迹范围]")
    print(f"    RAS cam0: X=[{ras_cam0[:,0].min():.4f},{ras_cam0[:,0].max():.4f}] "
          f"Y=[{ras_cam0[:,1].min():.4f},{ras_cam0[:,1].max():.4f}] "
          f"Z=[{ras_cam0[:,2].min():.4f},{ras_cam0[:,2].max():.4f}]")
    print(f"    SLAM cam0: X=[{slam_cam0_s[:,0].min():.4f},{slam_cam0_s[:,0].max():.4f}] "
          f"Y=[{slam_cam0_s[:,1].min():.4f},{slam_cam0_s[:,1].max():.4f}] "
          f"Z=[{slam_cam0_s[:,2].min():.4f},{slam_cam0_s[:,2].max():.4f}]")
    print(f"    SLAM@Rx cam0: X=[{slam_cam0_rx_s[:,0].min():.4f},{slam_cam0_rx_s[:,0].max():.4f}] "
          f"Y=[{slam_cam0_rx_s[:,1].min():.4f},{slam_cam0_rx_s[:,1].max():.4f}] "
          f"Z=[{slam_cam0_rx_s[:,2].min():.4f},{slam_cam0_rx_s[:,2].max():.4f}]")

    # PCA 对比
    print(f"\n  [cam0 空间 PCA 对比]")
    print(f"    RAS vs 重建: {pca_angle(recon_cam0_s, ras_cam0):.1f}°")
    print(f"    RAS vs SLAM: {pca_angle(slam_cam0_s, ras_cam0):.1f}°")
    print(f"    RAS vs SLAM@Rx: {pca_angle(slam_cam0_rx_s, ras_cam0):.1f}°")

    # Kabsch 最优对齐
    R_opt_recon, s_opt_recon, _ = umeyama_full(recon_cam0_s, ras_cam0)
    R_opt_slam, s_opt_slam, _ = umeyama_full(slam_cam0_s, ras_cam0)
    R_opt_slam_rx, s_opt_slam_rx, _ = umeyama_full(slam_cam0_rx_s, ras_cam0)

    def rot_to_axis_angle(R):
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-10:
            return np.array([1, 0, 0]), 0.0
        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        return axis, np.degrees(angle)

    print(f"\n  [Kabsch 最优旋转后 PCA]")
    for label, ha_cam0, R_opt, s_opt in [
        ('重建', recon_cam0_s, R_opt_recon, s_opt_recon),
        ('SLAM', slam_cam0_s, R_opt_slam, s_opt_slam),
        ('SLAM@Rx', slam_cam0_rx_s, R_opt_slam_rx, s_opt_slam_rx)
    ]:
        ha_aligned = s_opt * (R_opt @ ha_cam0.T).T
        pca_after = pca_angle(ha_aligned, ras_cam0)
        ax, ang = rot_to_axis_angle(R_opt)
        print(f"    {label}: PCA {pca_after:.1f}°, 旋转 {ang:.1f}° 绕 {ax}, 尺度 {s_opt:.4f}")

    # 直接比较: 重建的 t 和 SLAM 的 t 是什么关系?
    print(f"\n  [t_recon vs t_slam 关系]")
    # 重建的 t_recon 是 SLAM 的 t_slam 经过某种转换
    # 可能: t_recon = Rx_hand @ t_slam (因为 OpenGL→OpenCV 转换)
    t_pred = Rx_hand @ t_slam.T
    t_pred = t_pred.T  # 恢复 (N, 3)
    t_diff = np.linalg.norm(t_pred - t_recon, axis=1).mean()
    print(f"    t_recon 与 Rx_hand @ t_slam 的平均差异: {t_diff:.6f}")

    # 可能: t_recon = Rx_hand @ (t_slam - t_slam[0]) + t_recon[0] (带平移)
    t_pred2 = Rx_hand @ (t_slam - t_slam[0]).T
    t_pred2 = t_pred2.T + t_recon[0]
    t_diff2 = np.linalg.norm(t_pred2 - t_recon, axis=1).mean()
    print(f"    t_recon 与 Rx_hand@(t_slam-t_slam[0])+t_recon[0] 的平均差异: {t_diff2:.6f}")

    # 也许重建的 t 和 SLAM 的 t 没有直接关系?
    # 检查重建的 t 是否来自 depth alignment 或其他优化
    print(f"\n  [SLAM vs 重建轨迹形状]")
    print(f"    SLAM vs 重建 (cam0) PCA: {pca_angle(slam_cam0_s, recon_cam0_s):.1f}°")
    print(f"    SLAM@Rx vs 重建 (cam0) PCA: {pca_angle(slam_cam0_rx_s, recon_cam0_s):.1f}°")

    # 对 7: 哪个轨迹与 RAS 更匹配?
    print(f"\n  [结论]")
    pca_r = pca_angle(recon_cam0_s, ras_cam0)
    pca_s = pca_angle(slam_cam0_s, ras_cam0)
    pca_sr = pca_angle(slam_cam0_rx_s, ras_cam0)
    if pca_r < pca_s and pca_r < pca_sr:
        print(f"    重建轨迹与 RAS 最匹配 (PCA {pca_r:.1f}°)")
    elif pca_s < pca_r and pca_s < pca_sr:
        print(f"    SLAM 原始轨迹与 RAS 最匹配 (PCA {pca_s:.1f}°)")
    elif pca_sr < pca_r and pca_sr < pca_s:
        print(f"    SLAM@Rx 轨迹与 RAS 最匹配 (PCA {pca_sr:.1f}°)")
    else:
        print(f"    所有轨迹都不匹配: 重建={pca_r:.1f}°, SLAM={pca_s:.1f}°, SLAM@Rx={pca_sr:.1f}°")

    # 关键: 重建 t 和 SLAM t 是否在同一个坐标系?
    print(f"\n  [坐标系检查]")
    # 对 121: 检查重建和 SLAM 的 t 是否在右手系
    print(f"    SLAM t[0] = {t_slam[0]}")
    print(f"    重建 t[0] = {t_recon[0]}")
    print(f"    Rx_hand @ t_slam[0] = {Rx_hand @ t_slam[0]}")
    print(f"    SLAM t 范围: [{t_slam.min():.4f}, {t_slam.max():.4f}]")
    print(f"    重建 t 范围: [{t_recon.min():.4f}, {t_recon.max():.4f}]")