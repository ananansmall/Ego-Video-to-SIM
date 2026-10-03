#!/usr/bin/env python3
"""
test_two_frame_align.py — 测试用前N帧对齐(首帧+第二帧)找最优旋转
"""

import numpy as np
import os
import glob

EXAMPLES = {
    '121': {
        'ras_dir': '/home/an/data/ras/121_C5_CellPhone_161deg_vggt_omega',
        'hawor_npz': '/home/an/data/hawor/121_C5_CellPhone_161deg/reconstruction/hawor_results_0_600.npz',
        'params': '/home/an/data/hawor/121_C5_CellPhone_161deg/output/alignment/transform_params.npz',
    },
    'hoi4d': {
        'ras_dir': '/home/an/data/ras/hoi4d1_vggt_omega',
        'hawor_npz': '/home/an/data/hawor/hoi4d/reconstruction/hawor_results_0_600.npz',
        'params': '/home/an/data/hawor/hoi4d/output/alignment/transform_params.npz',
    },
    '7': {
        'ras_dir': '/home/an/data/ras/7_vggt_omega',
        'hawor_npz': '/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz',
        'params': '/home/an/data/hawor/7/output/alignment/transform_params.npz',
    },
}

R_X = np.eye(3)


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


def resample(points, n_target):
    n_src = len(points)
    indices = np.linspace(0, n_src - 1, n_target, dtype=int)
    return points[indices]


def pca_angle(a, b):
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    _, _, Vt_a = np.linalg.svd(a_c, full_matrices=False)
    _, _, Vt_b = np.linalg.svd(b_c, full_matrices=False)
    dot = np.clip(np.dot(Vt_a[0], Vt_b[0]), -1, 1)
    if dot < 0:
        dot = -dot
    return np.degrees(np.arccos(dot))


def kabsch_rotation(src, dst):
    """Kabsch SVD: 找最优旋转 R (src→dst), 无尺度"""
    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def rot_to_axis_angle(R):
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        return np.array([1, 0, 0]), 0.0
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
    return axis, np.degrees(angle)


for name, paths in EXAMPLES.items():
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    R_r, t_r = load_ras_extrinsics(paths['ras_dir'])
    h = dict(np.load(paths['hawor_npz'], allow_pickle=True))
    R_h = h['R_c2w']
    t_h = h['t_c2w']
    p = dict(np.load(paths['params']))
    R_hand_to_glb_current = p['R_hand_to_glb']
    t_hand_to_glb_current = p['t_hand_to_glb']
    s_current = float(p['scale_ratio'])

    n_ras = len(t_r)
    n_ha = len(t_h)

    # ========== 当前方法 (只用第一帧) ==========
    t_h_glb_current = s_current * (R_hand_to_glb_current @ t_h.T).T + t_hand_to_glb_current
    t_h_glb_current_s = resample(t_h_glb_current, n_ras)
    pca_current = pca_angle(t_h_glb_current_s, t_r)

    # ========== 方法1: 用前 N 帧在 cam0 空间找最优旋转 ==========
    print(f"\n  [方法1: 前N帧 cam0 空间 Kabsch]")

    R_r0 = R_r[0]
    R_h0 = R_h[0]

    # cam0 空间
    ras_cam0 = np.array([R_r0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])
    ha_cam0 = np.array([R_h0.T @ (t_h[i] - t_h[0]) for i in range(n_ha)])

    # 帧索引映射
    ha_indices = np.linspace(0, n_ha - 1, n_ras, dtype=int)
    ha_cam0_s = ha_cam0[ha_indices]

    for n_frames in [2, 3, 5, 10]:
        if n_frames > n_ras:
            continue
        # 取前 n_frames 帧
        src = ha_cam0_s[:n_frames]
        dst = ras_cam0[:n_frames]

        # Kabsch 找最优旋转 (cam0 空间)
        R_cam_align = kabsch_rotation(src, dst)
        _, ang = rot_to_axis_angle(R_cam_align)

        # 构建新的 R_hand_to_glb
        R_new = R_r0 @ R_cam_align @ R_X @ R_h0.T
        s_new = s_current  # 保持尺度不变
        t_new = t_r[0] - s_new * R_new @ t_h[0]

        # 应用
        t_h_new = s_new * (R_new @ t_h.T).T + t_new
        t_h_new_s = resample(t_h_new, n_ras)
        pca_new = pca_angle(t_h_new_s, t_r)
        mean_dist = np.linalg.norm(t_h_new_s - t_r, axis=1).mean()

        imp = pca_current - pca_new
        marker = "✓" if imp > 10 else ("⚠" if imp > 3 else "✗")
        print(f"    {n_frames}帧: PCA {pca_current:.1f}°→{pca_new:.1f}° ({imp:+.1f}°) "
              f"旋转 {ang:.1f}° 平均距离 {mean_dist:.4f}m {marker}")

    # ========== 方法2: 用前 N 帧在原始世界空间找最优旋转 ==========
    print(f"\n  [方法2: 前N帧原始世界空间 Kabsch]")

    t_h_sampled = resample(t_h, n_ras)

    for n_frames in [2, 3, 5, 10]:
        if n_frames > n_ras:
            continue
        src = t_h_sampled[:n_frames]
        dst = t_r[:n_frames]

        # Kabsch 找最优旋转+尺度 (原始世界空间)
        src_c = src - src.mean(axis=0)
        dst_c = dst - dst.mean(axis=0)
        H = src_c.T @ dst_c
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        sigma_src = np.sqrt(np.mean(np.sum(src_c ** 2, axis=1)))
        sigma_dst = np.sqrt(np.mean(np.sum(dst_c ** 2, axis=1)))
        s = sigma_dst / sigma_src if sigma_src > 1e-10 else 1.0
        t_mean = dst.mean(axis=0) - s * R @ src.mean(axis=0)

        # 不利用第一帧信息, 直接用 Kabsch 结果
        t_h_kabsch = s * (R @ t_h.T).T + t_mean
        t_h_kabsch_s = resample(t_h_kabsch, n_ras)
        pca_kabsch = pca_angle(t_h_kabsch_s, t_r)
        mean_dist = np.linalg.norm(t_h_kabsch_s - t_r, axis=1).mean()
        _, ang = rot_to_axis_angle(R)

        imp = pca_current - pca_kabsch
        marker = "✓" if imp > 10 else ("⚠" if imp > 3 else "✗")
        print(f"    {n_frames}帧: PCA {pca_current:.1f}°→{pca_kabsch:.1f}° ({imp:+.1f}°) "
              f"旋转 {ang:.1f}° 尺度 {s:.4f} 平均距离 {mean_dist:.4f}m {marker}")

    # ========== 方法3: 用前N帧的 ORIENTATION 找旋转 ==========
    print(f"\n  [方法3: 前N帧 Orientation 平均]")

    for n_frames in [2, 3, 5, 10]:
        if n_frames > n_ras:
            continue
        # 对每帧计算 R_i = R_ras_i @ R_ha_i.T, 然后平均
        R_avg = np.zeros((3, 3))
        for i in range(min(n_frames, n_ras)):
            hi = ha_indices[i]
            R_i = R_r[i] @ R_h[hi].T
            R_avg += R_i
        # 投影到 SO(3)
        U, _, Vt = np.linalg.svd(R_avg)
        R_orient = U @ Vt

        s_new = s_current
        t_new = t_r[0] - s_new * R_orient @ t_h[0]
        t_h_new = s_new * (R_orient @ t_h.T).T + t_new
        t_h_new_s = resample(t_h_new, n_ras)
        pca_new = pca_angle(t_h_new_s, t_r)
        mean_dist = np.linalg.norm(t_h_new_s - t_r, axis=1).mean()
        _, ang = rot_to_axis_angle(R_orient)

        imp = pca_current - pca_new
        marker = "✓" if imp > 10 else ("⚠" if imp > 3 else "✗")
        print(f"    {n_frames}帧: PCA {pca_current:.1f}°→{pca_new:.1f}° ({imp:+.1f}°) "
              f"旋转 {ang:.1f}° 平均距离 {mean_dist:.4f}m {marker}")

    # ========== 方法4: 用前N帧的位移向量找旋转 ==========
    print(f"\n  [方法4: 前N帧位移向量 Kabsch]")

    for n_frames in [3, 5, 10]:
        if n_frames > n_ras:
            continue
        # 用 cam0 空间中的位移向量
        # 位移 = cam0[i] - cam0[0] (i=1,2,...,n_frames-1)
        src_disps = np.array([ha_cam0_s[i] - ha_cam0_s[0] for i in range(1, n_frames)])
        dst_disps = np.array([ras_cam0[i] - ras_cam0[0] for i in range(1, n_frames)])

        R_disp = kabsch_rotation(src_disps, dst_disps)
        _, ang = rot_to_axis_angle(R_disp)

        R_new = R_r0 @ R_disp @ R_X @ R_h0.T
        s_new = s_current
        t_new = t_r[0] - s_new * R_new @ t_h[0]
        t_h_new = s_new * (R_new @ t_h.T).T + t_new
        t_h_new_s = resample(t_h_new, n_ras)
        pca_new = pca_angle(t_h_new_s, t_r)
        mean_dist = np.linalg.norm(t_h_new_s - t_r, axis=1).mean()

        imp = pca_current - pca_new
        marker = "✓" if imp > 10 else ("⚠" if imp > 3 else "✗")
        print(f"    {n_frames-1}个位移: PCA {pca_current:.1f}°→{pca_new:.1f}° ({imp:+.1f}°) "
              f"旋转 {ang:.1f}° 平均距离 {mean_dist:.4f}m {marker}")

    # ========== 方法5: 用全部帧 Kabsch (上限) ==========
    print(f"\n  [方法5: 全部帧 Kabsch (上限)]")
    R_full = kabsch_rotation(ha_cam0_s, ras_cam0)
    _, ang_full = rot_to_axis_angle(R_full)
    R_new_full = R_r0 @ R_full @ R_X @ R_h0.T
    t_new_full = t_r[0] - s_current * R_new_full @ t_h[0]
    t_h_full = s_current * (R_new_full @ t_h.T).T + t_new_full
    t_h_full_s = resample(t_h_full, n_ras)
    pca_full = pca_angle(t_h_full_s, t_r)
    mean_dist_full = np.linalg.norm(t_h_full_s - t_r, axis=1).mean()
    print(f"    全部帧: PCA {pca_current:.1f}°→{pca_full:.1f}° ({pca_current-pca_full:+.1f}°) "
          f"旋转 {ang_full:.1f}° 平均距离 {mean_dist_full:.4f}m")

    # ========== 总结 ==========
    print(f"\n  [总结]")
    print(f"    当前(首帧): PCA={pca_current:.1f}°")
    print(f"    全部帧上限: PCA={pca_full:.1f}°")
    print(f"    差距: {pca_current - pca_full:.1f}° 需要旋转 {ang_full:.1f}°")