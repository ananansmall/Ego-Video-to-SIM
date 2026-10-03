#!/usr/bin/env python3
"""
check_slam_traj.py — 检查 SLAM 原始轨迹与重建轨迹的关系
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
    """四元数 [qw, qx, qy, qz] → 旋转矩阵 (3,3)"""
    qw, qx, qy, qz = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


for name, paths in EXAMPLES.items():
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # 加载 SLAM
    slam = dict(np.load(paths['slam_npz'], allow_pickle=True))
    traj = slam['traj']  # (N, 7) = [x, y, z, qw, qx, qy, qz]
    t_slam = traj[:, :3]  # (N, 3)
    q_slam = traj[:, 3:]  # (N, 4) = [qw, qx, qy, qz]

    # 加载重建
    recon = dict(np.load(paths['recon_npz'], allow_pickle=True))
    R_recon = recon['R_c2w']
    t_recon = recon['t_c2w']

    print(f"  SLAM traj: {traj.shape}")
    print(f"  重建 R_c2w: {R_recon.shape}")
    print(f"  重建 t_c2w: {t_recon.shape}")

    # 将 SLAM 四元数转为旋转矩阵
    R_slam = np.array([quat_to_rotmat(q_slam[i]) for i in range(len(q_slam))])

    # SLAM 第一帧
    print(f"\n  SLAM 第一帧:")
    print(f"    t: {t_slam[0]}")
    print(f"    q: {q_slam[0]}")
    print(f"    R_c2w (from quat):\n{R_slam[0]}")
    print(f"    det(R_slam[0]): {np.linalg.det(R_slam[0]):.6f}")

    # 重建第一帧
    print(f"\n  重建第一帧:")
    print(f"    t: {t_recon[0]}")
    print(f"    R_c2w:\n{R_recon[0]}")
    print(f"    det(R_recon[0]): {np.linalg.det(R_recon[0]):.6f}")

    # 检查: R_recon[0] 与 Rx_hand @ R_slam_w2c[0] 的关系
    # R_slam 存储的是 c2w (从四元数推导)
    # 但公式说: R_recon_c2w = diag([1,-1,-1]) @ R_slam_w2c
    # R_slam_w2c = R_slam_c2w.T
    R_slam_w2c_0 = R_slam[0].T
    R_pred = Rx_hand @ R_slam_w2c_0
    diff = np.linalg.norm(R_pred - R_recon[0])
    print(f"\n  Rx_hand @ R_slam_w2c[0] vs R_recon[0]: 差异={diff:.6f}")
    if diff < 0.01:
        print(f"  → 确认: 重建 R_c2w = diag([1,-1,-1]) @ R_slam_w2c ✓")
    else:
        print(f"  → 不匹配! ✗")

    # 试试不同的组合
    # 可能是 R_recon_c2w = R_slam_c2w (直接用 SLAM 的 c2w)
    diff2 = np.linalg.norm(R_slam[0] - R_recon[0])
    print(f"  R_slam_c2w[0] vs R_recon[0]: 差异={diff2:.6f}")

    # 可能是 R_recon_c2w = R_slam_c2w @ Rx_hand
    diff3 = np.linalg.norm(R_slam[0] @ Rx_hand - R_recon[0])
    print(f"  R_slam_c2w @ Rx_hand vs R_recon[0]: 差异={diff3:.6f}")

    # 可能是 R_recon_c2w = Rx_hand @ R_slam_c2w
    diff4 = np.linalg.norm(Rx_hand @ R_slam[0] - R_recon[0])
    print(f"  Rx_hand @ R_slam_c2w vs R_recon[0]: 差异={diff4:.6f}")

    # SLAM 轨迹 vs 重建轨迹 (t)
    print(f"\n  SLAM t 范围:")
    print(f"    X: [{t_slam[:,0].min():.4f}, {t_slam[:,0].max():.4f}]")
    print(f"    Y: [{t_slam[:,1].min():.4f}, {t_slam[:,1].max():.4f}]")
    print(f"    Z: [{t_slam[:,2].min():.4f}, {t_slam[:,2].max():.4f}]")

    print(f"\n  重建 t 范围:")
    print(f"    X: [{t_recon[:,0].min():.4f}, {t_recon[:,0].max():.4f}]")
    print(f"    Y: [{t_recon[:,1].min():.4f}, {t_recon[:,1].max():.4f}]")
    print(f"    Z: [{t_recon[:,2].min():.4f}, {t_recon[:,2].max():.4f}]")

    # 检查 t_recon 和 t_slam 的关系
    # t_recon 可能 = Rx_hand @ t_slam (因为 OpenGL→OpenCV 转换)
    t_pred = Rx_hand @ t_slam[0]
    print(f"\n  Rx_hand @ t_slam[0] = {t_pred}")
    print(f"  t_recon[0] = {t_recon[0]}")
    print(f"  差异: {np.linalg.norm(t_pred - t_recon[0]):.6f}")

    # 在 cam0 空间比较 SLAM 轨迹 vs RAS
    R_r, t_r = load_ras_extrinsics(paths['ras_dir'])
    n_ras = len(t_r)
    n_ha = len(t_recon)

    R_c2w_r0 = R_r[0]
    R_c2w_h0 = R_recon[0]  # 重建的 R_c2w[0]
    R_slam0 = R_slam[0]    # SLAM 的 R_c2w[0]

    ras_cam0 = np.array([R_c2w_r0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])

    # 重建 cam0
    ha_cam0_recon = np.array([R_c2w_h0.T @ (t_recon[i] - t_recon[0]) for i in range(n_ha)])
    indices = np.linspace(0, n_ha - 1, n_ras, dtype=int)
    ha_cam0_recon_sampled = ha_cam0_recon[indices]

    # SLAM cam0 (用 SLAM 的 R_c2w[0])
    ha_cam0_slam = np.array([R_slam0.T @ (t_slam[i] - t_slam[0]) for i in range(len(t_slam))])
    ha_cam0_slam_sampled = ha_cam0_slam[indices]

    # PCA 对比
    def pca_angle(pts_a, pts_b):
        a_c = pts_a - pts_a.mean(axis=0)
        b_c = pts_b - pts_b.mean(axis=0)
        _, _, Vt_a = np.linalg.svd(a_c, full_matrices=False)
        _, _, Vt_b = np.linalg.svd(b_c, full_matrices=False)
        dot = np.clip(np.dot(Vt_a[0], Vt_b[0]), -1, 1)
        if dot < 0:
            dot = -dot
        return np.degrees(np.arccos(dot))

    pca_recon = pca_angle(ha_cam0_recon_sampled, ras_cam0)
    pca_slam = pca_angle(ha_cam0_slam_sampled, ras_cam0)

    print(f"\n  cam0 空间 PCA:")
    print(f"    重建 vs RAS: {pca_recon:.1f}°")
    print(f"    SLAM vs RAS: {pca_slam:.1f}°")

    # 用 Kabsch 在 SLAM cam0 空间找最优旋转
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

    R_opt_recon, s_opt_recon, _ = umeyama_full(ha_cam0_recon_sampled, ras_cam0)
    R_opt_slam, s_opt_slam, _ = umeyama_full(ha_cam0_slam_sampled, ras_cam0)

    def rot_to_axis_angle(R):
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-10:
            return np.array([1, 0, 0]), 0.0
        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]
        axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
        return axis, np.degrees(angle)

    axis_recon, ang_recon = rot_to_axis_angle(R_opt_recon)
    axis_slam, ang_slam = rot_to_axis_angle(R_opt_slam)

    print(f"\n  Kabsch 最优旋转 (cam0 空间):")
    print(f"    重建: 轴={axis_recon}, 角度={ang_recon:.1f}°, 尺度={s_opt_recon:.4f}")
    print(f"    SLAM: 轴={axis_slam}, 角度={ang_slam:.1f}°, 尺度={s_opt_slam:.4f}")

    # 应用最优旋转后 PCA
    ha_cam0_recon_aligned = s_opt_recon * (R_opt_recon @ ha_cam0_recon_sampled.T).T
    ha_cam0_slam_aligned = s_opt_slam * (R_opt_slam @ ha_cam0_slam_sampled.T).T
    pca_recon_aligned = pca_angle(ha_cam0_recon_aligned, ras_cam0)
    pca_slam_aligned = pca_angle(ha_cam0_slam_aligned, ras_cam0)

    print(f"    重建 PCA 改善: {pca_recon:.1f}° → {pca_recon_aligned:.1f}°")
    print(f"    SLAM PCA 改善: {pca_slam:.1f}° → {pca_slam_aligned:.1f}°")

    # 检查 SLAM 轨迹是否 vs 重建轨迹形状一致
    pca_slam_vs_recon = pca_angle(ha_cam0_slam_sampled, ha_cam0_recon_sampled)
    print(f"\n  SLAM vs 重建 cam0 轨迹 PCA: {pca_slam_vs_recon:.1f}°")
    print(f"  SLAM 与重建 t 差异: {np.linalg.norm(t_slam - t_recon, axis=1).mean():.6f}")