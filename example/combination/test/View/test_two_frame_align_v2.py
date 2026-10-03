#!/usr/bin/env python3
"""
test_two_frame_align_v2.py — 精确实现用户方案:
  - 第一帧: 对齐相机朝向 (R_hand_to_glb_base)
  - 前两帧位移向量: 对齐世界坐标系 (R_cam_align)

核心思想: 相机朝向对了，但世界坐标系没对齐。
位移向量(0→1帧)在不同坐标系下方向不同，对齐位移向量就固定了世界坐标系旋转。
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
    R_hand_to_glb_cur = p['R_hand_to_glb']
    t_hand_to_glb_cur = p['t_hand_to_glb']
    s_cur = float(p['scale_ratio'])

    n_ras = len(t_r)
    n_ha = len(t_h)
    ha_indices = np.linspace(0, n_ha - 1, n_ras, dtype=int)

    # ========== 当前方法 (首帧) ==========
    t_h_cur = s_cur * (R_hand_to_glb_cur @ t_h.T).T + t_hand_to_glb_cur
    t_h_cur_s = resample(t_h_cur, n_ras)
    pca_cur = pca_angle(t_h_cur_s, t_r)

    # ========== 用户方案 ==========
    # 1. 第一帧: 对齐相机朝向
    R_r0 = R_r[0]
    R_h0 = R_h[0]
    t_r0 = t_r[0]
    t_h0 = t_h[0]

    # 2. 前两帧位移向量 (在 cam0 空间)
    # RAS 位移 (cam0 空间)
    delta_r = R_r0.T @ (t_r[1] - t_r[0])
    # HaWoR 位移 (cam0 空间)
    delta_h = R_h0.T @ (t_h[1] - t_h[0])

    print(f"\n  [前两帧位移向量 (cam0 空间)]")
    print(f"    RAS 位移:   {delta_r}")
    print(f"    HaWoR 位移: {delta_h}")

    # 归一化方向
    dir_r = delta_r / np.linalg.norm(delta_r)
    dir_h = delta_h / np.linalg.norm(delta_h)
    print(f"    RAS 方向:   {dir_r}")
    print(f"    HaWoR 方向: {dir_h}")
    dir_dot = np.clip(np.dot(dir_r, dir_h), -1, 1)
    print(f"    方向夹角:   {np.degrees(np.arccos(dir_dot)):.1f}°")

    # 位移大小
    print(f"    RAS 位移大小: {np.linalg.norm(delta_r):.4f}m")
    print(f"    HaWoR 位移大小: {np.linalg.norm(delta_h):.4f}m")
    print(f"    大小比例: {np.linalg.norm(delta_r) / np.linalg.norm(delta_h):.4f}")

    # 3. 用前两帧 (位置 + 姿态) 在 cam0 空间找最优旋转
    # 方法A: 只用位移向量 + 相机朝向 (2个约束)
    # 构造点集: 帧0位置 + 帧0三个轴 + 帧1位置 + 帧1三个轴
    # 帧0在cam0空间: 位置(0,0,0), 轴为单位向量
    R_r1 = R_r[0].T @ R_r[1]  # 帧1的姿态在cam0空间
    R_h1 = R_h0.T @ R_h[ha_indices[1]]  # 帧1的姿态在cam0空间

    # 构造8个点: 帧0位置 + 帧0X轴 + 帧0Y轴 + 帧0Z轴 + 帧1位置 + 帧1X轴 + 帧1Y轴 + 帧1Z轴
    eye = np.eye(3)
    ras_points = np.vstack([
        [0, 0, 0],  # 帧0位置
        eye[0], eye[1], eye[2],  # 帧0三轴
        delta_r,  # 帧1位置
        delta_r + R_r1[0], delta_r + R_r1[1], delta_r + R_r1[2],  # 帧1三轴
    ])
    ha_points = np.vstack([
        [0, 0, 0],
        eye[0], eye[1], eye[2],
        delta_h,
        delta_h + R_h1[0], delta_h + R_h1[1], delta_h + R_h1[2],
    ])

    # Kabsch 在 cam0 空间
    R_cam_align = kabsch_rotation(ha_points, ras_points)
    axis_cam, ang_cam = rot_to_axis_angle(R_cam_align)

    print(f"\n  [cam0 空间最优旋转 (8点=位置+姿态)]")
    print(f"    旋转矩阵:\n{R_cam_align}")
    print(f"    轴: {axis_cam}, 角度: {ang_cam:.2f}°")

    # 应用: R_hand_to_glb_new = R_r0 @ R_cam_align @ R_h0.T
    R_hand_to_glb_new = R_r0 @ R_cam_align @ R_h0.T
    t_new = t_r[0] - s_cur * R_hand_to_glb_new @ t_h[0]

    t_h_new = s_cur * (R_hand_to_glb_new @ t_h.T).T + t_new
    t_h_new_s = resample(t_h_new, n_ras)
    pca_new = pca_angle(t_h_new_s, t_r)
    mean_dist = np.linalg.norm(t_h_new_s - t_r, axis=1).mean()

    imp = pca_cur - pca_new
    print(f"    结果: PCA {pca_cur:.1f}° → {pca_new:.1f}° ({imp:+.1f}°)")
    print(f"    平均距离: {mean_dist:.4f}m")

    # 对比: 只用位移 (2个点)
    R_cam_2pt = kabsch_rotation(
        np.array([[0,0,0], delta_h]),
        np.array([[0,0,0], delta_r])
    )
    _, ang_2pt = rot_to_axis_angle(R_cam_2pt)
    R_hg_2pt = R_r0 @ R_cam_2pt @ R_h0.T
    t_2pt = t_r[0] - s_cur * R_hg_2pt @ t_h[0]
    t_h_2pt = s_cur * (R_hg_2pt @ t_h.T).T + t_2pt
    t_h_2pt_s = resample(t_h_2pt, n_ras)
    pca_2pt = pca_angle(t_h_2pt_s, t_r)
    dist_2pt = np.linalg.norm(t_h_2pt_s - t_r, axis=1).mean()
    print(f"  [只用位移2点] 旋转 {ang_2pt:.1f}°  PCA {pca_cur:.1f}°→{pca_2pt:.1f}° 距离 {dist_2pt:.4f}m")

    # 对比: 全部帧 (上限)
    R_cam_full = kabsch_rotation(
        resample(np.array([R_h0.T @ (t_h[i] - t_h[0]) for i in range(n_ha)]), n_ras),
        np.array([R_r0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])
    )
    _, ang_full = rot_to_axis_angle(R_cam_full)
    R_hg_full = R_r0 @ R_cam_full @ R_h0.T
    t_full = t_r[0] - s_cur * R_hg_full @ t_h[0]
    t_h_full = s_cur * (R_hg_full @ t_h.T).T + t_full
    t_h_full_s = resample(t_h_full, n_ras)
    pca_full = pca_angle(t_h_full_s, t_r)
    dist_full = np.linalg.norm(t_h_full_s - t_r, axis=1).mean()
    print(f"  [全部帧上限] 旋转 {ang_full:.1f}°  PCA {pca_cur:.1f}°→{pca_full:.1f}° 距离 {dist_full:.4f}m")

    # 你的方案 vs 当前方法 R_hand_to_glb 的差异
    R_diff = R_hand_to_glb_new @ R_hand_to_glb_cur.T
    _, ang_diff = rot_to_axis_angle(R_diff)
    print(f"\n  [R_hand_to_glb 差异]")
    print(f"    当前: R_hand_to_glb (首帧朝向)")
    print(f"    新方案: R_hand_to_glb (首帧朝向 + 2帧位移坐标系对齐)")
    print(f"    差异矩阵:\n{R_diff}")
    print(f"    差异旋转: {ang_diff:.2f}°")

    # 解释: 为什么这个旋转很重要
    print(f"\n  [解释]")
    print(f"    第一帧相机朝向: R_r0 = R_hand_to_glb @ R_h0")
    print(f"    前两帧位移: delta_r = R_r0 @ R_cam_align @ R_h0.T @ delta_h")
    print(f"               = R_cam_align_in_world @ delta_h")
    print(f"    R_cam_align 就是对齐世界坐标系的旋转!")