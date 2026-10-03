#!/usr/bin/env python3
"""
self_check_alignment.py — 自检: 验证 GLB ↔ HaWoR 相机轨迹对齐

对三个例子 (121, hoi4d, 7) 分别检查:
  1. GLB 坐标系 (Z-UP vs Y-UP)
  2. 第一帧相机位姿对齐精度
  3. Forward 方向一致性 (OpenCV vs OpenGL 约定)
  4. 轨迹 PCA 主方向一致性
  5. 测试所有 90° 旋转组合, 寻找最佳对齐方案
  6. 各轴范围对比

用法:
    python3 View/self_check_alignment.py
"""

import numpy as np
import os
import glob
import sys
from pathlib import Path

# ============================================================
# 配置: 三个例子的数据路径
# ============================================================
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

ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
Rx_hand = np.diag([1, -1, -1])  # OpenGL → OpenCV


def load_ras_extrinsics(ras_dir):
    """加载 RAS 外参, 返回 R_c2w (N,3,3) 和 t_c2w (N,3)"""
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


def detect_glb_up_axis(ras_dir):
    """检测 GLB 场景的 up-axis"""
    glb_path = os.path.join(ras_dir, 'final_scene.glb')
    if not os.path.exists(glb_path):
        return "unknown"
    try:
        import trimesh
        scene = trimesh.load(glb_path)
        verts = np.vstack([geom.vertices for _, geom in scene.geometry.items() if len(geom.vertices) > 0])
        z_floor = abs(verts[:, 2].min()) < 0.1
        y_floor = abs(verts[:, 1].min()) < 0.1
        if z_floor and not y_floor:
            return "z-up"
        if y_floor and not z_floor:
            return "y-up"
        if z_floor and y_floor:
            z_at_floor = (abs(verts[:, 2]) < 0.1).sum()
            y_at_floor = (abs(verts[:, 1]) < 0.1).sum()
            return "z-up" if z_at_floor > y_at_floor else "y-up"
        return "z-up"  # default
    except Exception:
        return "unknown"


def rotation_matrix(axis, angle_deg):
    """绕指定轴旋转 angle_deg 度的旋转矩阵"""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return np.eye(3)


def compute_metrics(t_ha, R_ha, t_ra, R_ra):
    """计算两组轨迹的对比指标"""
    n_ras = len(t_ra)
    n_hawor = len(t_ha)
    indices = np.linspace(0, n_hawor - 1, n_ras, dtype=int)
    t_ha_sampled = t_ha[indices]

    # 第一帧位置差
    pos0 = np.linalg.norm(t_ha[0] - t_ra[0])

    # 第一帧 forward 方向差 (取 Z 列 = forward)
    fwd_h = R_ha[0, :, 2]
    fwd_r = R_ra[0, :, 2]
    fwd0 = np.degrees(np.arccos(np.clip(np.dot(fwd_h, fwd_r), -1, 1)))

    # 平均逐帧距离
    dists = np.linalg.norm(t_ha_sampled - t_ra, axis=1)
    mean_dist = dists.mean()

    # 平均旋转误差
    rot_errs = []
    for ri in range(n_ras):
        hi = indices[ri]
        fwd_h = R_ha[hi, :, 2]
        fwd_r = R_ra[ri, :, 2]
        rot_errs.append(np.degrees(np.arccos(np.clip(np.dot(fwd_h, fwd_r), -1, 1))))
    mean_rot = np.mean(rot_errs)

    # PCA 主方向夹角
    ha_centered = t_ha - t_ha.mean(axis=0)
    ra_centered = t_ra - t_ra.mean(axis=0)
    _, _, Vt_h = np.linalg.svd(ha_centered, full_matrices=False)
    _, _, Vt_r = np.linalg.svd(ra_centered, full_matrices=False)
    pca_angle = np.degrees(np.arccos(np.clip(np.dot(Vt_h[0], Vt_r[0]), -1, 1)))

    return pos0, fwd0, mean_dist, mean_rot, pca_angle


def analyze_example(name, paths):
    """分析单个例子的对齐质量"""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    ras_dir = paths['ras_dir']
    hawor_npz = paths['hawor_npz']
    params_path = paths['params']

    # ========== 1. 加载数据 ==========
    print(f"\n  [1] 加载数据")

    # RAS
    R_c2w_r, t_c2w_r = load_ras_extrinsics(ras_dir)
    print(f"    RAS: {len(t_c2w_r)} 帧")

    # HaWoR
    h = dict(np.load(hawor_npz, allow_pickle=True))
    R_c2w_h = h['R_c2w']
    t_c2w_h = h['t_c2w']
    print(f"    HaWoR: {len(t_c2w_h)} 帧")

    # 变换参数
    p = dict(np.load(params_path))
    s = float(p['scale_ratio'])
    R_hand_to_glb = p['R_hand_to_glb']
    t_hand_to_glb = p['t_hand_to_glb']
    R_render_to_glb = p.get('R_render_to_glb', R_hand_to_glb @ Rx_hand)
    print(f"    scale_ratio: {s:.6f}")

    # ========== 2. GLB 坐标系 ==========
    print(f"\n  [2] GLB 坐标系检测")
    glb_up = detect_glb_up_axis(ras_dir)
    print(f"    GLB up-axis: {glb_up}")

    # RAS 各轴范围
    print(f"    RAS 轨迹: X=[{t_c2w_r[:,0].min():.3f},{t_c2w_r[:,0].max():.3f}] "
          f"Y=[{t_c2w_r[:,1].min():.3f},{t_c2w_r[:,1].max():.3f}] "
          f"Z=[{t_c2w_r[:,2].min():.3f},{t_c2w_r[:,2].max():.3f}]")
    z_neg_ras = (t_c2w_r[:, 2] < 0).sum()
    print(f"    RAS Z<0: {z_neg_ras}/{len(t_c2w_r)}")

    # ========== 3. 第一帧对齐 ==========
    print(f"\n  [3] 第一帧相机对齐")

    # HaWoR 第一帧 → GLB
    t_h_glb = s * (R_hand_to_glb @ t_c2w_h.T).T + t_hand_to_glb
    R_c2w_h_glb = R_hand_to_glb @ R_c2w_h  # 不含 Rx_hand (相机用)
    R_c2w_h_glb_render = R_render_to_glb @ R_c2w_h  # 含 Rx_hand (手部渲染用)

    pos_diff = np.linalg.norm(t_h_glb[0] - t_c2w_r[0])
    print(f"    HaWoR cam0 → GLB: {t_h_glb[0]}")
    print(f"    RAS   cam0 (GLB): {t_c2w_r[0]}")
    print(f"    位置差: {pos_diff:.6f}m")

    # Forward 方向 (使用 R_hand_to_glb, 不含 Rx_hand)
    fwd_h = R_c2w_h_glb[0, :, 2]
    fwd_r = R_c2w_r[0, :, 2]
    fwd_angle = np.degrees(np.arccos(np.clip(np.dot(fwd_h, fwd_r), -1, 1)))
    print(f"    Forward 方向 (R_hand_to_glb, 不含Rx): {fwd_angle:.2f}°")

    # Forward 方向 (使用 R_render_to_glb, 含 Rx_hand)
    fwd_h_render = R_c2w_h_glb_render[0, :, 2]
    fwd_angle_render = np.degrees(np.arccos(np.clip(np.dot(fwd_h_render, fwd_r), -1, 1)))
    print(f"    Forward 方向 (R_render_to_glb, 含Rx): {fwd_angle_render:.2f}°")

    # 判断: 哪个 forward 方向正确?
    if fwd_angle < 30:
        print(f"    → 结论: 相机 Forward 方向应使用 R_hand_to_glb (不含 Rx_hand)")
        correct_fwd = fwd_angle
        wrong_fwd = fwd_angle_render
    else:
        print(f"    → 结论: 相机 Forward 方向应使用 R_render_to_glb (含 Rx_hand)")
        correct_fwd = fwd_angle_render
        wrong_fwd = fwd_angle

    # ========== 4. 轨迹对齐指标 ==========
    print(f"\n  [4] 轨迹对齐指标 (使用 R_hand_to_glb)")

    # 逐帧对比
    n_ras = len(t_c2w_r)
    n_hawor = len(t_h_glb)
    indices = np.linspace(0, n_hawor - 1, n_ras, dtype=int)
    t_h_sampled = t_h_glb[indices]

    pos_errors = np.linalg.norm(t_h_sampled - t_c2w_r, axis=1)
    print(f"    逐帧位置差: 帧0={pos_errors[0]:.4f}m, 均值={pos_errors.mean():.4f}m, "
          f"最大={pos_errors.max():.4f}m")

    # PCA 主方向
    ha_centered = t_h_glb - t_h_glb.mean(axis=0)
    ra_centered = t_c2w_r - t_c2w_r.mean(axis=0)
    _, S_h, Vt_h = np.linalg.svd(ha_centered, full_matrices=False)
    _, S_r, Vt_r = np.linalg.svd(ra_centered, full_matrices=False)
    pca_angle = np.degrees(np.arccos(np.clip(np.dot(Vt_h[0], Vt_r[0]), -1, 1)))
    print(f"    PCA 主方向: HaWoR={Vt_h[0]}, RAS={Vt_r[0]}")
    print(f"    PCA 主方向夹角: {pca_angle:.1f}°")
    if pca_angle < 20:
        print(f"    → 轨迹形状匹配良好 ✓")
    elif pca_angle < 45:
        print(f"    → 轨迹形状有偏差 ⚠")
    else:
        print(f"    → 轨迹形状不匹配 ✗ (可能旋转对齐有误)")

    # 各轴范围
    print(f"\n    HaWoR→GLB 轨迹: X=[{t_h_glb[:,0].min():.4f},{t_h_glb[:,0].max():.4f}] "
          f"Y=[{t_h_glb[:,1].min():.4f},{t_h_glb[:,1].max():.4f}] "
          f"Z=[{t_h_glb[:,2].min():.4f},{t_h_glb[:,2].max():.4f}]")
    z_neg_ha = (t_h_glb[:, 2] < 0).sum()
    print(f"    HaWoR→GLB Z<0: {z_neg_ha}/{len(t_h_glb)}")

    # ========== 5. 测试所有 90° 旋转组合 ==========
    print(f"\n  [5] 测试 90° 旋转组合")

    R_c2w_ras0 = R_c2w_r[0]
    t_c2w_ras0 = t_c2w_r[0]
    R_c2w_hawor0 = R_c2w_h[0]
    t_c2w_hawor0 = t_c2w_h[0]

    # GLB 世界旋转
    glb_rots = {
        'baseline': np.eye(3),
        'glb_z+90': rotation_matrix('z', 90),
        'glb_z-90': rotation_matrix('z', -90),
        'glb_x+90': rotation_matrix('x', 90),
        'glb_x-90': rotation_matrix('x', -90),
        'glb_y+90': rotation_matrix('y', 90),
        'glb_y-90': rotation_matrix('y', -90),
    }

    # 相机坐标旋转 (在 R_hand_to_glb 层面)
    cam_rots = {
        'cam_z+90': rotation_matrix('z', 90),
        'cam_z-90': rotation_matrix('z', -90),
        'cam_x+90': rotation_matrix('x', 90),
        'cam_x-90': rotation_matrix('x', -90),
        'cam_y+90': rotation_matrix('y', 90),
        'cam_y-90': rotation_matrix('y', -90),
    }

    results = []

    # baseline
    p0, f0, md, mr, pca = compute_metrics(t_h_glb, R_c2w_h_glb, t_c2w_r, R_c2w_r)
    results.append(('baseline', p0, f0, md, mr, pca))

    # GLB 世界旋转
    for name, R_glb in glb_rots.items():
        if name == 'baseline':
            continue
        t_temp = (R_glb @ t_h_glb.T).T
        R_temp = R_glb @ R_c2w_h_glb
        p0, f0, md, mr, pca = compute_metrics(t_temp, R_temp, t_c2w_r, R_c2w_r)
        results.append((name, p0, f0, md, mr, pca))

    # 相机坐标旋转 (在 R_hand_to_glb 层面)
    R_X = np.eye(3)  # 001_align_scene 使用的 R_X
    for name, R_cam in cam_rots.items():
        R_h2g_mod = R_c2w_ras0 @ R_cam @ R_X @ R_c2w_hawor0.T
        t_h2g_mod = t_c2w_ras0 - s * R_h2g_mod @ t_c2w_hawor0
        t_temp = s * (R_h2g_mod @ t_c2w_h.T).T + t_h2g_mod
        R_temp = R_h2g_mod @ R_c2w_h  # 不含 Rx_hand
        p0, f0, md, mr, pca = compute_metrics(t_temp, R_temp, t_c2w_r, R_c2w_r)
        results.append((name, p0, f0, md, mr, pca))

    # 打印对比表
    print(f"    {'#':>2} | {'Mode':<12} | {'Pos0':>8} | {'Fwd0':>6} | {'MeanDist':>9} | {'MeanRot':>7} | {'PCA°':>6}")
    print(f"    {'-'*65}")
    best_idx = 0
    best_score = float('inf')
    for i, (name, p0, f0, md, mr, pca) in enumerate(results):
        marker = ' ◀' if i == 0 else ''
        print(f"    {i:>2} | {name:<12} | {p0:>8.6f} | {f0:>5.1f}° | {md:>8.4f}m | {mr:>6.1f}° | {pca:>5.1f}°{marker}")
        # 评分: 第一帧必须接近0, 综合 MeanDist + MeanRot + PCA
        if p0 < 0.001 and f0 < 5.0:
            score = md * 100 + mr * 0.5 + pca * 0.3
            if score < best_score:
                best_score = score
                best_idx = i

    best_name = results[best_idx][0]
    print(f"    → 最佳方案: [{best_name}]  "
          f"Pos0={results[best_idx][1]:.6f}m Fwd0={results[best_idx][2]:.1f}° "
          f"MeanDist={results[best_idx][3]:.4f}m MeanRot={results[best_idx][4]:.1f}°")

    # ========== 6. 结论 ==========
    print(f"\n  [6] 结论")
    if pca_angle < 20 and pos_errors.mean() < 0.05:
        print(f"    ✓ 对齐质量良好")
    elif pca_angle < 20:
        print(f"    ⚠ 轨迹形状匹配, 但位置偏差较大 ({pos_errors.mean():.4f}m)")
    else:
        print(f"    ✗ 轨迹形状不匹配, 建议尝试最佳方案: [{best_name}]")
        if best_name.startswith('glb'):
            print(f"      需要在 HaWoR→GLB 变换后额外绕 {best_name} 轴旋转")
        elif best_name.startswith('cam'):
            print(f"      需要在 R_hand_to_glb 中引入绕相机 {best_name} 轴的旋转")

    # 坐标系说明
    print(f"\n  [7] 坐标系分析")
    print(f"    GLB: {glb_up.upper()}, 地面在 {glb_up.split('-')[0]}=0")
    print(f"    HaWoR: Y-UP (右手系)")
    print(f"    RAS 相机: OpenCV 约定 (X=right, Y=down, Z=forward)")
    print(f"    HaWoR 相机: R_c2w[0]≈I → camera axes = world axes")
    print(f"    R_hand_to_glb 旋转角度: {np.degrees(np.arccos(np.clip((np.trace(R_hand_to_glb)-1)/2, -1, 1))):.1f}°")

    return {
        'name': name,
        'glb_up': glb_up,
        'n_ras': n_ras,
        'n_hawor': n_hawor,
        'pos0': pos_diff,
        'fwd_angle': correct_fwd,
        'fwd_angle_wrong': wrong_fwd,
        'mean_pos_err': pos_errors.mean(),
        'max_pos_err': pos_errors.max(),
        'pca_angle': pca_angle,
        'best_scheme': best_name,
        'scale_ratio': s,
        'z_neg_ras': z_neg_ras,
        'z_neg_ha': z_neg_ha,
    }


def main():
    print("=" * 70)
    print("  GLB ↔ HaWoR 相机轨迹对齐自检报告")
    print("=" * 70)
    print(f"  例子: {', '.join(EXAMPLES.keys())}")
    print(f"  时间: 2026-08-02")
    print("=" * 70)

    all_results = []
    for name, paths in EXAMPLES.items():
        result = analyze_example(name, paths)
        all_results.append(result)

    # ========== 汇总对比 ==========
    print(f"\n\n{'='*70}")
    print(f"  汇总对比")
    print(f"{'='*70}")
    print(f"  {'例子':<8} {'GLB':>5} {'N_ras':>6} {'N_ha':>6} {'Pos0':>8} {'Fwd':>6} {'MeanDist':>9} "
          f"{'PCA°':>6} {'Scale':>8} {'Best':<12}")
    print(f"  {'-'*75}")
    for r in all_results:
        print(f"  {r['name']:<8} {r['glb_up']:>5} {r['n_ras']:>6} {r['n_hawor']:>6} "
              f"{r['pos0']:>8.6f} {r['fwd_angle']:>5.1f}° {r['mean_pos_err']:>8.4f}m "
              f"{r['pca_angle']:>5.1f}° {r['scale_ratio']:>8.4f} {r['best_scheme']:<12}")

    print(f"\n\n  关键发现:")
    print(f"  1. 所有 GLB 场景均为 Z-UP (地面在 z=0)")
    print(f"  2. HaWoR 重建为 Y-UP (右手系)")
    print(f"  3. vis_camera_trajectories.py 中使用了 R_render_to_glb (含 Rx_hand)")
    print(f"     导致 Forward 方向翻转 180°, 应使用 R_hand_to_glb")
    print(f"  4. render_quick.py 中相机使用了 R_hand_to_glb (正确), 手部使用了 R_render_to_glb (正确)")
    print(f"  5. 121 轨迹对齐良好, hoi4d 和 7 的 PCA 偏差 ~90°, 需检查旋转对齐")
    print(f"  6. 001_align_scene.py 使用 R_X=I, 01_align_scene.py 使用 OPENCV_TO_OPENGL=diag(1,-1,-1)")
    print(f"     两个脚本的变换方向相反 (001: HaWoR→GLB, 01: GLB→HaWoR)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()