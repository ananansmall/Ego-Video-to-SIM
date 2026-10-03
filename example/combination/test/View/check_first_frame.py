#!/usr/bin/env python3
"""
check_first_frame.py — 检查 HaWoR SLAM 原始数据 vs 重建数据的第一帧
"""

import numpy as np
import os

EXAMPLES = {
    '121': {
        'slam_npz': '/home/an/data/hawor/121_C5_CellPhone_161deg/SLAM/hawor_slam_w_scale_0_600.npz',
        'recon_npz': '/home/an/data/hawor/121_C5_CellPhone_161deg/reconstruction/hawor_results_0_600.npz',
        'ras_dir': '/home/an/data/ras/121_C5_CellPhone_161deg_vggt_omega',
    },
    'hoi4d': {
        'slam_npz': None,  # 可能没有
        'recon_npz': '/home/an/data/hawor/hoi4d/reconstruction/hawor_results_0_600.npz',
        'ras_dir': '/home/an/data/ras/hoi4d1_vggt_omega',
    },
    '7': {
        'slam_npz': '/home/an/data/hawor/7/SLAM/hawor_slam_w_scale_0_113.npz',
        'recon_npz': '/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz',
        'ras_dir': '/home/an/data/ras/7_vggt_omega',
    },
}

Rx_hand = np.diag([1, -1, -1])  # OpenGL → OpenCV


def load_ras_extrinsics(ras_dir):
    import glob
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


for name, paths in EXAMPLES.items():
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    # RAS 第一帧
    R_r, t_r = load_ras_extrinsics(paths['ras_dir'])
    print(f"\n  RAS 第一帧 R_c2w[0]:")
    print(f"    {R_r[0]}")
    print(f"    Forward (Z列): {R_r[0, :, 2]}")
    print(f"    Up (Y列): {R_r[0, :, 1]}")
    print(f"    Right (X列): {R_r[0, :, 0]}")
    print(f"  t_c2w[0]: {t_r[0]}")
    print(f"  det(R): {np.linalg.det(R_r[0]):.6f}")

    # 重建数据第一帧
    recon = dict(np.load(paths['recon_npz'], allow_pickle=True))
    R_recon = recon['R_c2w']
    t_recon = recon['t_c2w']
    print(f"\n  HaWoR 重建 R_c2w[0]:")
    print(f"    {R_recon[0]}")
    print(f"    Forward (Z列): {R_recon[0, :, 2]}")
    print(f"    Up (Y列): {R_recon[0, :, 1]}")
    print(f"    Right (X列): {R_recon[0, :, 0]}")
    print(f"  t_c2w[0]: {t_recon[0]}")
    print(f"  det(R): {np.linalg.det(R_recon[0]):.6f}")

    # 重建数据中的其他字段
    print(f"\n  重建 npz 包含的字段: {list(recon.keys())}")
    for k in recon.keys():
        v = recon[k]
        if isinstance(v, np.ndarray):
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

    # SLAM 数据 (如果有)
    slam_path = paths['slam_npz']
    if slam_path and os.path.exists(slam_path):
        slam = dict(np.load(slam_path, allow_pickle=True))
        print(f"\n  SLAM npz 包含的字段: {list(slam.keys())}")
        for k in slam.keys():
            v = slam[k]
            if isinstance(v, np.ndarray):
                print(f"    {k}: shape={v.shape}, dtype={v.dtype}")

        # 尝试找到 SLAM 的位姿
        if 'R_c2w' in slam:
            R_slam = slam['R_c2w']
            print(f"\n  SLAM R_c2w[0]:")
            print(f"    {R_slam[0]}")
            print(f"  det(R): {np.linalg.det(R_slam[0]):.6f}")
        elif 'R_w2c' in slam:
            R_slam_w2c = slam['R_w2c']
            print(f"\n  SLAM R_w2c[0]:")
            print(f"    {R_slam_w2c[0]}")
            R_slam_c2w = R_slam_w2c[0].T
            print(f"  SLAM R_c2w[0] (= R_w2c[0].T):")
            print(f"    {R_slam_c2w}")
            print(f"  det(R): {np.linalg.det(R_slam_c2w):.6f}")

            # 重建的 R_c2w 应该 = diag([1,-1,-1]) @ R_w2c
            R_pred = Rx_hand @ R_slam_w2c[0]  # diag([1,-1,-1]) @ R_slam_w2c
            print(f"\n  Rx_hand @ R_slam_w2c[0] (预测重建 R_c2w):")
            print(f"    {R_pred}")
            diff = np.linalg.norm(R_pred - R_recon[0])
            print(f"  与重建 R_c2w[0] 差异: {diff:.6f}")
            if diff < 0.01:
                print(f"  → 确认: 重建 R_c2w = diag([1,-1,-1]) @ R_slam_w2c ✓")
            else:
                print(f"  → 不匹配! 重建 R_c2w != diag([1,-1,-1]) @ R_slam_w2c ✗")

        # 检查 SLAM 的 t
        if 't_c2w' in slam:
            t_slam = slam['t_c2w']
            print(f"\n  SLAM t_c2w[0]: {t_slam[0]}")
        elif 't_w2c' in slam:
            t_slam_w2c = slam['t_w2c']
            print(f"\n  SLAM t_w2c[0]: {t_slam_w2c[0]}")
            # c2w 位置 = -R_c2w @ t_w2c
            R_slam_c2w = R_slam_w2c[0].T
            t_slam_c2w = -R_slam_c2w @ t_slam_w2c[0]
            print(f"  SLAM t_c2w[0] (= -R_c2w @ t_w2c): {t_slam_c2w}")

    # 比较 R_hand_to_glb 的两种方式
    print(f"\n  当前 R_hand_to_glb = R_c2w_ras0 @ R_X @ R_c2w_hawor0.T")
    R_current = R_r[0] @ np.eye(3) @ R_recon[0].T
    print(f"  = R_r[0] @ R_recon[0].T")
    print(f"  {R_current}")
    angle = np.degrees(np.arccos(np.clip((np.trace(R_current)-1)/2, -1, 1)))
    print(f"  旋转角度: {angle:.2f}°")
    print(f"  det: {np.linalg.det(R_current):.6f}")

    # 尝试不同 R_X
    for name_rx, Rx in [('I', np.eye(3)), ('Rx_hand=diag(1,-1,-1)', Rx_hand)]:
        R_try = R_r[0] @ Rx @ R_recon[0].T
        angle_try = np.degrees(np.arccos(np.clip((np.trace(R_try)-1)/2, -1, 1)))
        print(f"    R_X={name_rx}: 旋转角度={angle_try:.2f}°, det={np.linalg.det(R_try):.6f}")

    # 重建数据的轨迹范围
    print(f"\n  HaWoR 重建轨迹范围:")
    print(f"    X: [{t_recon[:,0].min():.4f}, {t_recon[:,0].max():.4f}]")
    print(f"    Y: [{t_recon[:,1].min():.4f}, {t_recon[:,1].max():.4f}]")
    print(f"    Z: [{t_recon[:,2].min():.4f}, {t_recon[:,2].max():.4f}]")
    print(f"    运动幅度: {np.linalg.norm(t_recon.max(axis=0) - t_recon.min(axis=0)):.4f}")

    # RAS 轨迹范围
    print(f"\n  RAS 轨迹范围:")
    print(f"    X: [{t_r[:,0].min():.4f}, {t_r[:,0].max():.4f}]")
    print(f"    Y: [{t_r[:,1].min():.4f}, {t_r[:,1].max():.4f}]")
    print(f"    Z: [{t_r[:,2].min():.4f}, {t_r[:,2].max():.4f}]")
    print(f"    运动幅度: {np.linalg.norm(t_r.max(axis=0) - t_r.min(axis=0)):.4f}")

    # 在 cam0 空间对比 PCA
    R_c2w_r0 = R_r[0]
    R_c2w_h0 = R_recon[0]
    n_ras = len(t_r)
    n_ha = len(t_recon)
    ras_cam0 = np.array([R_c2w_r0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])
    ha_cam0 = np.array([R_c2w_h0.T @ (t_recon[i] - t_recon[0]) for i in range(n_ha)])
    # 重采样
    indices = np.linspace(0, n_ha - 1, n_ras, dtype=int)
    ha_cam0_sampled = ha_cam0[indices]

    # PCA
    _, _, Vt_r = np.linalg.svd(ras_cam0 - ras_cam0.mean(axis=0), full_matrices=False)
    _, _, Vt_h = np.linalg.svd(ha_cam0_sampled - ha_cam0_sampled.mean(axis=0), full_matrices=False)
    dot = np.clip(np.dot(Vt_r[0], Vt_h[0]), -1, 1)
    if dot < 0:
        dot = -dot
    pca_cam0 = np.degrees(np.arccos(dot))
    print(f"\n  cam0 空间 PCA 主方向夹角: {pca_cam0:.1f}°")
    print(f"  RAS cam0 PCA 主方向: {Vt_r[0]}")
    print(f"  HaWoR cam0 PCA 主方向: {Vt_h[0]}")