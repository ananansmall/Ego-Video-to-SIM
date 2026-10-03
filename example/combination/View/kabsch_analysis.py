#!/usr/bin/env python3
"""
kabsch_analysis.py — Kabsch (Umeyama) 最优旋转对齐分析

对每个例子:
  1. 加载 HaWoR 和 RAS 轨迹
  2. 重采样 HaWoR 帧数匹配 RAS
  3. 用 Umeyama (Kabsch) 寻找最优旋转 + 尺度 + 平移
  4. 比较最优旋转与当前 R_hand_to_glb 的差异
  5. 分解差异为"缺失旋转轴"(轴 + 角度)
  6. 应用缺失旋转后验证对齐质量

用法:
    python3 View/kabsch_analysis.py
"""

import numpy as np
import os
import glob
import sys
from pathlib import Path

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


def resample_hawor(t_hawor, n_ras):
    """重采样 HaWoR 到与 RAS 相同帧数"""
    n_hawor = len(t_hawor)
    indices = np.linspace(0, n_hawor - 1, n_ras, dtype=int)
    return t_hawor[indices]


def umeyama_full(src, dst):
    """
    Umeyama (Kabsch) 完整对齐: 找最优旋转R、尺度s、平移t
    dst = s * R @ src + t

    src: (N,3) 源点集
    dst: (N,3) 目标点集
    返回: R (3,3), s (float), t (3,)
    """
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    sigma_src = np.sqrt(np.mean(np.sum(src_c ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_c ** 2, axis=1)))

    # SVD 求旋转
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 反射修正
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    # 尺度
    s = sigma_dst / sigma_src if sigma_src > 1e-10 else 1.0

    # 平移
    t = dst_mean - s * R @ src_mean

    return R, s, t


def rotation_to_axis_angle(R):
    """旋转矩阵 → (轴, 角度)"""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-10:
        return np.array([1, 0, 0]), 0.0
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2 * np.sin(angle))
    return axis, np.degrees(angle)


def compute_pca_angle(pts_a, pts_b):
    """计算两组点 PCA 主方向夹角"""
    a_c = pts_a - pts_a.mean(axis=0)
    b_c = pts_b - pts_b.mean(axis=0)
    _, _, Vt_a = np.linalg.svd(a_c, full_matrices=False)
    _, _, Vt_b = np.linalg.svd(b_c, full_matrices=False)
    # 处理方向歧义
    dot = np.dot(Vt_a[0], Vt_b[0])
    if dot < 0:
        dot = -dot
    return np.degrees(np.arccos(np.clip(dot, -1, 1)))


def analyze_kabsch(name, paths):
    print(f"\n{'='*70}")
    print(f"  {name} — Kabsch 分析")
    print(f"{'='*70}")

    # 1. 加载数据
    R_c2w_r, t_r = load_ras_extrinsics(paths['ras_dir'])
    h = dict(np.load(paths['hawor_npz'], allow_pickle=True))
    R_c2w_h = h['R_c2w']
    t_h = h['t_c2w']
    p = dict(np.load(paths['params']))
    R_hand_to_glb = p['R_hand_to_glb']
    t_hand_to_glb = p['t_hand_to_glb']
    s_current = float(p['scale_ratio'])

    n_ras = len(t_r)
    n_hawor = len(t_h)
    print(f"  RAS: {n_ras} 帧, HaWoR: {n_hawor} 帧")

    # 用当前变换先把 HaWoR 映射到 GLB
    t_h_glb_current = s_current * (R_hand_to_glb @ t_h.T).T + t_hand_to_glb
    R_h_glb_current = R_hand_to_glb @ R_c2w_h

    # 重采样 HaWoR→GLB 到 RAS 帧数
    t_h_sampled_current = resample_hawor(t_h_glb_current, n_ras)

    # 2. 当前对齐指标 (baseline)
    pca_current = compute_pca_angle(t_h_sampled_current, t_r)
    dists_current = np.linalg.norm(t_h_sampled_current - t_r, axis=1)
    print(f"\n  [当前对齐] R_hand_to_glb (仅用第一帧)")
    print(f"    PCA 主方向夹角: {pca_current:.1f}°")
    print(f"    平均逐帧距离: {dists_current.mean():.4f}m")
    print(f"    最大逐帧距离: {dists_current.max():.4f}m")

    # 3. Kabsch 全局最优对齐 (直接在 GLB 空间)
    print(f"\n  [Kabsch 全局最优对齐] 使用全部帧")
    R_kabsch, s_kabsch, t_kabsch = umeyama_full(t_h_sampled_current, t_r)

    # 应用 Kabsch 结果
    t_h_kabsch = s_kabsch * (R_kabsch @ t_h_sampled_current.T).T + t_kabsch

    pca_kabsch = compute_pca_angle(t_h_kabsch, t_r)
    dists_kabsch = np.linalg.norm(t_h_kabsch - t_r, axis=1)
    print(f"    Kabsch 旋转矩阵:\n{R_kabsch}")
    axis_k, angle_k = rotation_to_axis_angle(R_kabsch)
    print(f"    Kabsch 旋转: 轴={axis_k}, 角度={angle_k:.2f}°")
    print(f"    Kabsch 尺度: {s_kabsch:.6f}")
    print(f"    PCA 主方向夹角: {pca_kabsch:.1f}°")
    print(f"    平均逐帧距离: {dists_kabsch.mean():.4f}m")
    print(f"    最大逐帧距离: {dists_kabsch.max():.4f}m")

    # 4. 分析缺失旋转: 当前 R_hand_to_glb 与 Kabsch 最优的差距
    # 当前变换: t_h_glb_current = s_current * R_hand_to_glb @ t_h + t_hand_to_glb
    # Kabsch 在 GLB 空间: t_h_kabsch_global = s_kabsch * R_kabsch @ t_h_sampled_current + t_kabsch
    # 但我们需要的是对原始 HaWoR 坐标的变换
    # t_h = HaWoR 原始坐标
    # t_h_glb_current = s_current * R_hand_to_glb @ t_h + t_hand_to_glb
    # 最优应: t_h_glb_optimal = s_optimal * R_optimal @ t_h + t_optimal
    # 用 Kabsch 直接在原始 HaWoR 坐标上找最优变换
    print(f"\n  [Kabsch 在原始 HaWoR 空间] 直接用 t_h → t_r")
    t_h_sampled_raw = resample_hawor(t_h, n_ras)
    R_opt, s_opt, t_opt = umeyama_full(t_h_sampled_raw, t_r)
    t_h_opt = s_opt * (R_opt @ t_h.T).T + t_opt
    # 重采样
    t_h_opt_sampled = resample_hawor(t_h_opt, n_ras)

    pca_opt = compute_pca_angle(t_h_opt_sampled, t_r)
    dists_opt = np.linalg.norm(t_h_opt_sampled - t_r, axis=1)
    print(f"    最优旋转矩阵:\n{R_opt}")
    axis_opt, angle_opt = rotation_to_axis_angle(R_opt)
    print(f"    最优旋转: 轴={axis_opt}, 角度={angle_opt:.2f}°")
    print(f"    最优尺度: {s_opt:.6f}")
    print(f"    PCA 主方向夹角: {pca_opt:.1f}°")
    print(f"    平均逐帧距离: {dists_opt.mean():.4f}m")
    print(f"    最大逐帧距离: {dists_opt.max():.4f}m")

    # 5. 缺失旋转 = R_opt @ R_hand_to_glb.T
    R_missing = R_opt @ R_hand_to_glb.T
    axis_miss, angle_miss = rotation_to_axis_angle(R_missing)
    print(f"\n  [缺失旋转分析]")
    print(f"    R_missing = R_optimal @ R_hand_to_glb.T")
    print(f"    缺失旋转矩阵:\n{R_missing}")
    print(f"    缺失旋转轴: {axis_miss}")
    print(f"    缺失旋转角度: {angle_miss:.2f}°")

    # 解析: 缺失旋转接近哪个轴?
    axes = {'X': np.array([1, 0, 0]), 'Y': np.array([0, 1, 0]), 'Z': np.array([0, 0, 1])}
    print(f"    与标准轴夹角:")
    for ax_name, ax_vec in axes.items():
        cos_sim = np.clip(np.dot(axis_miss, ax_vec), -1, 1)
        ax_angle = np.degrees(np.arccos(cos_sim))
        print(f"      {ax_name}轴: {ax_angle:.1f}°")

    # 6. 验证: 用 R_opt 替换 R_hand_to_glb
    print(f"\n  [验证] 用最优旋转替换 R_hand_to_glb")
    print(f"    R_hand_to_glb_current = R_c2w_ras0 @ R_X @ R_c2w_hawor0.T")
    print(f"    R_hand_to_glb_optimal = R_c2w_ras0 @ R_missing @ R_X @ R_c2w_hawor0.T")

    # 验证: 用最优旋转的验证
    R_c2w_ras0 = R_c2w_r[0]
    R_c2w_hawor0 = R_c2w_h[0]
    R_hand_to_glb_new = R_c2w_ras0 @ R_missing @ R_X @ R_c2w_hawor0.T
    t_hand_to_glb_new = t_r[0] - s_opt * R_hand_to_glb_new @ t_h[0]

    t_h_new = s_opt * (R_hand_to_glb_new @ t_h.T).T + t_hand_to_glb_new
    t_h_new_sampled = resample_hawor(t_h_new, n_ras)

    pca_new = compute_pca_angle(t_h_new_sampled, t_r)
    dists_new = np.linalg.norm(t_h_new_sampled - t_r, axis=1)
    print(f"    PCA 主方向夹角: {pca_new:.1f}° (当前: {pca_current:.1f}°)")
    print(f"    平均逐帧距离: {dists_new.mean():.4f}m (当前: {dists_current.mean():.4f}m)")
    print(f"    最大逐帧距离: {dists_new.max():.4f}m (当前: {dists_current.max():.4f}m)")

    # 7. 在 cam0 空间分析
    print(f"\n  [cam0 空间分析]")
    ras_cam0 = np.array([R_c2w_ras0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])
    hawor_cam0 = np.array([R_X @ R_c2w_hawor0.T @ (t_h[i] - t_h[0]) for i in range(n_hawor)])
    hawor_cam0_sampled = resample_hawor(hawor_cam0, n_ras)

    # Kabsch 在 cam0 空间
    R_cam0_opt, s_cam0_opt, t_cam0_opt = umeyama_full(hawor_cam0_sampled, ras_cam0)
    axis_cam0, angle_cam0 = rotation_to_axis_angle(R_cam0_opt)
    print(f"    cam0 空间最优旋转: 轴={axis_cam0}, 角度={angle_cam0:.2f}°")
    print(f"    cam0 空间最优尺度: {s_cam0_opt:.6f}")
    print(f"    当前 R_X = I (假设无需旋转)")
    print(f"    最优 R_cam_align 与 R_X 的差异 = {angle_cam0:.2f}°")

    if angle_cam0 > 5:
        print(f"    → 结论: R_X 应该 != I, 需要绕 {axis_cam0} 轴旋转 {angle_cam0:.2f}°")

    # 8. 总结
    print(f"\n  [总结]")
    improvement = pca_current - pca_new
    if improvement > 10:
        print(f"    ✓ PCA 改善 {improvement:.1f}°, 缺失旋转轴 {axis_miss} 角度 {angle_miss:.1f}°")
    elif improvement > 3:
        print(f"    ⚠ PCA 小幅改善 {improvement:.1f}°, 缺失旋转轴 {axis_miss} 角度 {angle_miss:.1f}°")
    else:
        print(f"    ✗ PCA 几乎无改善 ({improvement:.1f}°), 轨迹形状本身就不匹配")

    return {
        'name': name,
        'pca_current': pca_current,
        'pca_optimal': pca_new,
        'mean_dist_current': dists_current.mean(),
        'mean_dist_optimal': dists_new.mean(),
        'missing_axis': axis_miss,
        'missing_angle': angle_miss,
        'R_optimal': R_opt,
        's_optimal': s_opt,
        'R_cam0_align': R_cam0_opt,
        'angle_cam0': angle_cam0,
    }


def main():
    print("=" * 70)
    print("  Kabsch 最优旋转对齐分析 — 寻找缺失旋转轴")
    print("=" * 70)
    print(f"  例子: {', '.join(EXAMPLES.keys())}")
    print("=" * 70)

    all_results = []
    for name, paths in EXAMPLES.items():
        result = analyze_kabsch(name, paths)
        all_results.append(result)

    # 汇总
    print(f"\n\n{'='*70}")
    print(f"  汇总")
    print(f"{'='*70}")
    print(f"  {'例子':<8} {'PCA当前':>8} {'PCA最优':>8} {'改善':>6} "
          f"{'缺失轴x':>8} {'缺失轴y':>8} {'缺失轴z':>8} {'缺失角':>8}")
    print(f"  {'-'*65}")
    for r in all_results:
        imp = r['pca_current'] - r['pca_optimal']
        ax = r['missing_axis']
        print(f"  {r['name']:<8} {r['pca_current']:>7.1f}° {r['pca_optimal']:>7.1f}° {imp:>+5.1f}° "
              f"{ax[0]:>8.4f} {ax[1]:>8.4f} {ax[2]:>8.4f} {r['missing_angle']:>7.1f}°")

    print(f"\n  关键结论:")
    for r in all_results:
        imp = r['pca_current'] - r['pca_optimal']
        if imp > 5:
            print(f"    - {r['name']}: 缺失旋转轴 {r['missing_axis']}, 角度 {r['missing_angle']:.1f}° "
                  f"(PCA: {r['pca_current']:.1f}°→{r['pca_optimal']:.1f}°)")
            print(f"      R_missing = R_optimal @ R_hand_to_glb.T, 应加到 R_hand_to_glb 公式中")
        else:
            print(f"    - {r['name']}: 轨迹形状本身差异大 (PCA {r['pca_current']:.1f}°→{r['pca_optimal']:.1f}°), "
                  f"不是旋转轴问题")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()