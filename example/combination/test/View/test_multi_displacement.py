#!/usr/bin/env python3
"""
test_multi_displacement.py — 用前N帧的位移向量找完整旋转
  用户方案: 第一帧对齐朝向, 位移向量对齐坐标系
  
  位移向量 = 每帧相对于第0帧的位移 (在cam0空间)
  使用前K帧的K-1个位移向量 → Kabsch确定完整3D旋转
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
    R_hg_cur = p['R_hand_to_glb']
    t_hg_cur = p['t_hand_to_glb']
    s_cur = float(p['scale_ratio'])

    n_ras = len(t_r)
    n_ha = len(t_h)
    ha_indices = np.linspace(0, n_ha - 1, n_ras, dtype=int)

    # 当前方法
    t_h_cur = s_cur * (R_hg_cur @ t_h.T).T + t_hg_cur
    t_h_cur_s = resample(t_h_cur, n_ras)
    pca_cur = pca_angle(t_h_cur_s, t_r)

    # cam0 空间
    R_r0 = R_r[0]
    R_h0 = R_h[0]
    ras_cam0 = np.array([R_r0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])
    ha_cam0_raw = np.array([R_h0.T @ (t_h[i] - t_h[0]) for i in range(n_ha)])
    ha_cam0 = ha_cam0_raw[ha_indices]

    # ========== 核心测试: 用前K帧的位移向量 ==========
    # 位移向量 = cam0[i] - cam0[0] (i=1, 2, ..., K-1)
    # 这些向量方向不同, 共同约束完整的3D旋转
    print(f"\n  [用前K帧位移向量找旋转]")
    print(f"  {'K帧':<6} {'位移数':<6} {'PCA改善':<20} {'旋转角':<10} {'平均距离':<12} {'结果':<10}")
    print(f"  {'-'*64}")

    best_pca = 999
    best_K = 0
    best_R = None

    for K in [2, 3, 4, 5, 6, 8, 10, 15, 20, 30]:
        if K > n_ras:
            continue
        # 位移向量 = cam0[i] - cam0[0], i=1..K-1
        src_disps = np.array([ha_cam0[i] - ha_cam0[0] for i in range(1, K)])
        dst_disps = np.array([ras_cam0[i] - ras_cam0[0] for i in range(1, K)])

        # 排除零位移 (第一帧几乎没动的情况)
        valid = np.linalg.norm(src_disps, axis=1) > 1e-6
        if valid.sum() < 2:
            continue
        src_v = src_disps[valid]
        dst_v = dst_disps[valid]

        R_cam = kabsch_rotation(src_v, dst_v)
        _, ang = rot_to_axis_angle(R_cam)

        # 应用到 R_hand_to_glb
        R_new = R_r0 @ R_cam @ R_h0.T
        # 保持尺度不变, 平移来自第一帧对齐
        t_new = t_r[0] - s_cur * R_new @ t_h[0]
        t_h_new = s_cur * (R_new @ t_h.T).T + t_new
        t_h_new_s = resample(t_h_new, n_ras)
        pca_new = pca_angle(t_h_new_s, t_r)
        dist = np.linalg.norm(t_h_new_s - t_r, axis=1).mean()

        imp = pca_cur - pca_new
        if imp > 10:
            marker = "✓✓"
        elif imp > 3:
            marker = "✓"
        elif imp > -3:
            marker = "~"
        else:
            marker = "✗"

        print(f"  {K:<6} {valid.sum():<6} {pca_cur:.1f}°→{pca_new:.1f}° ({imp:+.1f}°) {ang:<8.1f}° {dist:<10.4f}m {marker}")

        if pca_new < best_pca:
            best_pca = pca_new
            best_K = K
            best_R = R_cam

    # ========== 全部帧 (上限) ==========
    R_full = kabsch_rotation(ha_cam0, ras_cam0)
    _, ang_full = rot_to_axis_angle(R_full)
    R_hg_full = R_r0 @ R_full @ R_h0.T
    t_full = t_r[0] - s_cur * R_hg_full @ t_h[0]
    t_h_full = s_cur * (R_hg_full @ t_h.T).T + t_full
    t_h_full_s = resample(t_h_full, n_ras)
    pca_full = pca_angle(t_h_full_s, t_r)
    dist_full = np.linalg.norm(t_h_full_s - t_r, axis=1).mean()
    print(f"  {'全部':<6} {'-':<6} {pca_cur:.1f}°→{pca_full:.1f}° ({pca_cur-pca_full:+.1f}°) {ang_full:<8.1f}° {dist_full:<10.4f}m")

    # ========== 最佳K帧的结果 ==========
    print(f"\n  [最佳: K={best_K}帧, PCA={pca_cur:.1f}°→{best_pca:.1f}°]")
    R_cam_best = best_R
    R_hg_best = R_r0 @ R_cam_best @ R_h0.T
    
    # 最优旋转与当前R_hand_to_glb的差异
    R_diff = R_hg_best @ R_hg_cur.T
    _, ang_diff = rot_to_axis_angle(R_diff)
    print(f"    R_cam_align (cam0空间):\n{R_cam_best}")
    print(f"    旋转: {rot_to_axis_angle(R_cam_best)[1]:.2f}°")
    print(f"    R_hand_to_glb 差异: {ang_diff:.2f}°")
    
    # 建议: 在 001_align_scene.py 中, 用 K 帧位移向量做 Kabsch
    print(f"\n  [建议]")
    print(f"    在 001_align_scene.py 中:")
    print(f"    1. 计算 cam0 空间位移向量 (前K帧)")
    print(f"    2. Kabsch 找 R_cam_align")
    print(f"    3. R_hand_to_glb = R_c2w_ras0 @ R_cam_align @ R_c2w_hawor0.T")
    print(f"    4. t_hand_to_glb = t_r0 - s * R_hand_to_glb @ t_h0")