#!/usr/bin/env python3
"""
vis_camera_trajectories.py — 在 GLB 空间下对比 HaWoR 和 RAS 相机轨迹

使用 001 变换 (transform_params.npz) 把 HaWoR 相机转到 GLB 世界，
然后和 RAS 相机轨迹直接在 GLB 空间（Z-UP）下对比。

用法:
    python vis_camera_trajectories.py \\
        --hawor_dir /path/to/hawor/data \\
        --ras_dir /path/to/ras/data \\
        [--params_path /path/to/transform_params.npz]

    如果不传 --params_path, 默认用 {hawor_dir}/output/alignment/transform_params.npz

输出说明:
  - X/Y/Z 范围: 相机中心在 GLB 世界坐标系中各轴的运动范围 (米)
  - Z<0: 统计有多少帧的 Z 值为负 (GLB 空间中桌面在 Z≈0, 相机在桌面上方 Z>0, 正常应为 0)
  - 逐帧距离: 将 HaWoR 重采样到与 RAS 相同帧数后, 逐帧计算位置差 (米)
"""
import argparse
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R_X = np.eye(3)  # 经验验证: HaWoR 和 RAS 相机坐标系约定一致 (OpenCV), 无需额外转换


def main():
    parser = argparse.ArgumentParser(
        description="在 GLB 空间下对比 HaWoR 和 RAS 相机轨迹"
    )
    parser.add_argument("--hawor_dir", type=str, required=True,
                        help="HaWoR 数据目录 (内含 reconstruction/ 子目录)")
    parser.add_argument("--ras_dir", type=str, required=True,
                        help="RAS 数据目录 (内含 extrinsics/ 子目录)")
    parser.add_argument("--params_path", type=str, default=None,
                        help="transform_params.npz 路径 (默认: {hawor_dir}/output/alignment/transform_params.npz)")
    parser.add_argument("--z90", action="store_true",
                         help="对 HaWoR 轨迹绕 GLB Z 轴逆时针转 90° (测试)")
    parser.add_argument("--cam90", action="store_true",
                         help="绕相机 forward 轴 (Z) 转 90°: 保持 forward 不变, 修正轨迹方向")
    parser.add_argument("--test_all", action="store_true",
                         help="测试所有 90° 旋转组合并打印对比表")
    args = parser.parse_args()

    hawor_dir = args.hawor_dir
    ras_dir = args.ras_dir
    params_path = args.params_path or f'{hawor_dir}/output/alignment/transform_params.npz'

    # 绕 Z 轴逆时针 90° 旋转矩阵 (GLB 世界坐标)
    R_z90 = np.array([[0, -1, 0],
                      [1,  0, 0],
                      [0,  0, 1]])

    # 绕相机 forward 轴 (Z) 旋转 90° (相机坐标)
    # 保持 forward 方向不变, 交换 right/up 方向
    R_cam_z90 = np.array([[0, -1, 0],
                          [1,  0, 0],
                          [0,  0, 1]])  # 逆时针
    R_cam_z_neg90 = R_cam_z90.T  # 顺时针 (反向)

    # ========== 1. 加载 HaWoR 相机位姿 ==========
    # HaWoR 坐标系: Y-UP 右手系, camera-to-world (OpenGL 风格)
    rec_dir = f'{hawor_dir}/reconstruction'
    # 优先使用 depth_aligned 版本, 自动匹配任意帧数的文件名
    aligned_matches = sorted(glob.glob(f'{rec_dir}/hawor_results_*_depth_aligned.npz'))
    raw_matches = sorted(glob.glob(f'{rec_dir}/hawor_results_*.npz'))
    if aligned_matches:
        npz_path = aligned_matches[-1]
    elif raw_matches:
        npz_path = raw_matches[-1]
    else:
        raise FileNotFoundError(f'找不到 HaWoR 重建文件: {rec_dir}')
    print(f'Using: {os.path.basename(npz_path)}')
    h = dict(np.load(npz_path, allow_pickle=True))
    R_c2w_h = h['R_c2w']  # (N,3,3) 相机→HaWoR世界 的旋转
    t_c2w_h = h['t_c2w']  # (N,3)   相机在 HaWoR 世界的位置

    # ========== 2. 加载 001 变换参数 (HaWoR → GLB) ==========
    # 这个 npz 由 001_align_scene.py 生成, 包含:
    #   scale_ratio:     GLB 与 HaWoR 的尺度比
    #   R_hand_to_glb:   将 HaWoR 世界中的点旋转到 GLB 世界
    #   t_hand_to_glb:   将 HaWoR 世界中的点平移到 GLB 世界
    # 变换公式: p_glb = scale_ratio * R_hand_to_glb @ p_hawor + t_hand_to_glb
    params = dict(np.load(params_path))
    s = float(params['scale_ratio'])
    R_hand_to_glb = params['R_hand_to_glb']
    t_hand_to_glb = params['t_hand_to_glb']

    # ========== 3. 加载 RAS 相机位姿 ==========
    # RAS 外参存储为 world-to-camera (w2c) 矩阵, 需要转成 c2w
    # 坐标系: Z-UP, 已经在 GLB 空间下, 无需额外变换
    ext_files = sorted(glob.glob(f'{ras_dir}/extrinsics/*.txt'),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    R_all, t_all = [], []
    for f in ext_files:
        ext = np.loadtxt(f)            # (3,4) 或 (4,4) w2c 矩阵
        R_w2c = ext[:3,:3]
        t_w2c = ext[:3,3]
        R_all.append(R_w2c.T)          # R_c2w = R_w2c.T
        t_all.append(-R_w2c.T @ t_w2c) # t_c2w = -R_w2c.T @ t_w2c
    R_c2w_r = np.array(R_all)  # (N,3,3) 相机→GLB世界 的旋转
    t_c2w_r = np.array(t_all)  # (N,3)   相机在 GLB 世界的位置

    # ========== 4. 将 HaWoR 相机位姿变换到 GLB 世界 ==========
    # 对位置: p_glb = s * R_hand_to_glb @ p_hawor + t_hand_to_glb
    t_hawor_glb = s * (R_hand_to_glb @ t_c2w_h.T).T + t_hand_to_glb
    # 对朝向: R_c2w_glb = R_hand_to_glb @ R_c2w_hawor
    # (旋转矩阵只关心方向, 不受尺度影响)
    R_c2w_h_glb = R_hand_to_glb @ R_c2w_h  # (N,3,3)

    # 可选的 Z-90° 校正 (测试坐标系偏差)
    if args.z90:
        print('\n⚠️  应用绕 GLB Z 轴逆时针 90° 校正到 HaWoR 轨迹')
        t_hawor_glb = (R_z90 @ t_hawor_glb.T).T
        R_c2w_h_glb = R_z90 @ R_c2w_h_glb

    # 可选的相机 forward 轴 90° 校正
    # 保持 forward 方向不变, 修正 right/up 方向 (即修正轨迹形状)
    if args.cam90:
        # 在 R_hand_to_glb 层面加入相机 Z 轴 90° 旋转
        # R_hand_to_glb_mod = R_c2w_ras0 @ R_cam_z90 @ R_X @ R_c2w_hawor0.T
        # 这样 t_hand_to_glb 也会自动重算, 保证第一帧位置对齐
        R_c2w_ras0 = R_c2w_r[0]
        t_c2w_ras0 = t_c2w_r[0]
        R_c2w_hawor0 = R_c2w_h[0]
        t_c2w_hawor0 = t_c2w_h[0]
        R_hand_to_glb = R_c2w_ras0 @ R_cam_z90 @ R_X @ R_c2w_hawor0.T
        t_hand_to_glb = t_c2w_ras0 - s * R_hand_to_glb @ t_c2w_hawor0
        # 用新的变换重新计算 HaWoR 轨迹
        t_hawor_glb = s * (R_hand_to_glb @ t_c2w_h.T).T + t_hand_to_glb
        R_c2w_h_glb = R_hand_to_glb @ R_c2w_h
        print(f'\n⚠️  在 R_hand_to_glb 中加入绕相机 forward 轴 90° 旋转')
        print(f'  R_hand_to_glb (修正后):\n{R_hand_to_glb}')

    # ========== 5. 第一帧对齐检查 ==========
    # 检查第一帧相机是否对齐, 这是整个对齐方法的基础假设
    print('\n=== 第一帧对齐检查 ===')
    print(f'HaWoR 第一帧位置 (GLB): {t_hawor_glb[0]}')
    print(f'RAS   第一帧位置 (GLB): {t_c2w_r[0]}')
    pos_diff = np.linalg.norm(t_hawor_glb[0] - t_c2w_r[0])
    print(f'位置偏差: {pos_diff:.4f}m')

    # 两个系统都是 OpenCV 约定, forward = +Z列
    fwd_hawor = R_c2w_h_glb[0, :, 2]
    fwd_ras = R_c2w_r[0, :, 2]
    fwd_dot = np.dot(fwd_hawor, fwd_ras)
    fwd_angle = np.degrees(np.arccos(np.clip(fwd_dot, -1, 1)))
    print(f'朝向偏差 (forward 方向夹角): {fwd_angle:.2f}°')
    print(f'\nHaWoR forward (GLB): {fwd_hawor}')
    print(f'RAS   forward (GLB): {fwd_ras}')
    print(f'Forward 方向夹角: {fwd_angle:.2f}°')

    # ========== 6. 逐帧差异分析 ==========
    # 分析每一帧的差异, 看是随机漂移还是系统性旋转偏差
    print('\n=== 逐帧差异分析 (HaWoR vs RAS, 按帧对应) ===')

    # 对 RAS 的每一帧, 找到对应的 HaWoR 帧
    n_ras = len(t_c2w_r)
    n_hawor = len(t_hawor_glb)
    frame_map = np.linspace(0, n_hawor - 1, n_ras, dtype=int)  # RAS ri → HaWoR hi

    # 收集每一帧的差异数据
    pos_errors = []        # 位置差 (米)
    rot_angles = []        # 旋转差 (度)
    rel_rot_angles = []    # 相对于第一帧的旋转增益: 第i帧相对于第0帧的旋转差 - 第0帧的旋转差
    diffs = []             # 变换矩阵 (4x4) 的差异

    for ri in range(n_ras):
        hi = frame_map[ri]

        # 位置差异
        pos_err = np.linalg.norm(t_hawor_glb[hi] - t_c2w_r[ri])
        pos_errors.append(pos_err)

        # 朝向差异: 比较 forward 方向 (两个系统都是 OpenCV 约定, forward=+Z列)
        fwd_h = R_c2w_h_glb[hi, :, 2]
        fwd_r = R_c2w_r[ri, :, 2]
        fwd_dot = np.dot(fwd_h, fwd_r)
        fwd_err = np.degrees(np.arccos(np.clip(fwd_dot, -1, 1)))
        rot_angles.append(fwd_err)

        # 相对旋转增益: 排除第一帧已知的偏差 (虽然目前是 0)
        # R_rel = R_diff @ R_first_diff.T  其中 R_first_diff ≈ I
        # 直接用 rot_err 即可, 因为第一帧偏差为 0

        # 4x4 差异矩阵 (位置差, 朝向用 forward 方向近似)
        T_diff = np.eye(4)
        T_diff[:3, 3] = t_c2w_r[ri] - t_hawor_glb[hi]
        diffs.append(T_diff)

    pos_errors = np.array(pos_errors)
    rot_angles = np.array(rot_angles)

    # 输出汇总
    print(f'位置差异:  帧0={pos_errors[0]:.4f}m, 均值={pos_errors.mean():.4f}m, '
          f'最大={pos_errors.max():.4f}m, 末帧={pos_errors[-1]:.4f}m')
    print(f'旋转差异:  帧0={rot_angles[0]:.1f}°, 均值={rot_angles.mean():.1f}°, '
          f'最大={rot_angles.max():.1f}°, 末帧={rot_angles[-1]:.1f}°')

    # 分析趋势: 前1/3, 中1/3, 后1/3
    n3 = n_ras // 3
    print(f'\n分段分析:')
    print(f'  前 {n3} 帧: 位置均值={pos_errors[:n3].mean():.4f}m, 旋转均值={rot_angles[:n3].mean():.1f}°')
    print(f'  中 {n3} 帧: 位置均值={pos_errors[n3:2*n3].mean():.4f}m, 旋转均值={rot_angles[n3:2*n3].mean():.1f}°')
    print(f'  后 {n3} 帧: 位置均值={pos_errors[-n3:].mean():.4f}m, 旋转均值={rot_angles[-n3:].mean():.1f}°')

    # 检测末帧累积旋转: 与第一帧相比, 最后一个对应帧的旋转差变化量
    # 如果旋转变大, 说明 HaWoR 和 RAS 的旋转估计有系统性差异
    rot_drift = rot_angles[-1] - rot_angles[0]
    print(f'\n旋转漂移 (末帧-首帧): {rot_drift:.1f}°')
    if abs(rot_drift) > 10:
        print('  ⚠️ 旋转漂移较大, 说明两个系统的旋转估计有系统性偏差!')
    else:
        print('  ✓ 旋转漂移较小, 两个系统的旋转估计基本一致')

    # 轨迹主方向分析 (PCA)
    print('\n=== 轨迹主方向分析 ===')
    hawor_centered = t_hawor_glb - t_hawor_glb.mean(axis=0)
    U_h, S_h, Vt_h = np.linalg.svd(hawor_centered, full_matrices=False)
    hawor_main_dir = Vt_h[0]
    hawor_explained = S_h[0]**2 / (S_h**2).sum()
    print(f'HaWoR 主运动方向: [{hawor_main_dir[0]:.4f}, {hawor_main_dir[1]:.4f}, {hawor_main_dir[2]:.4f}]  '
          f'解释方差={hawor_explained:.1%}')

    ras_centered = t_c2w_r - t_c2w_r.mean(axis=0)
    U_r, S_r, Vt_r = np.linalg.svd(ras_centered, full_matrices=False)
    ras_main_dir = Vt_r[0]
    ras_explained = S_r[0]**2 / (S_r**2).sum()
    print(f'RAS  主运动方向: [{ras_main_dir[0]:.4f}, {ras_main_dir[1]:.4f}, {ras_main_dir[2]:.4f}]  '
          f'解释方差={ras_explained:.1%}')

    dir_angle = np.degrees(np.arccos(np.clip(np.dot(hawor_main_dir, ras_main_dir), -1, 1)))
    print(f'主方向夹角: {dir_angle:.1f}°')

    # R_hand_to_glb 的信息
    R_angle = np.degrees(np.arccos(np.clip((np.trace(R_hand_to_glb) - 1) / 2, -1, 1)))
    print(f'\nR_hand_to_glb 旋转角度: {R_angle:.1f}° (HaWoR世界→GLB世界的转换)')

    # ========== 7. 🔬 --test_all: 测试所有 90° 旋转组合 ==========
    if args.test_all:
        print('\n' + '='*70)
        print('🔬 测试所有 90° 旋转组合')
        print('='*70)

        # 定义所有旋转矩阵
        # GLB 世界坐标轴旋转 (应用到轨迹上)
        glb_rots = {
            'baseline': np.eye(3),
            'glb_x+90': np.array([[1,0,0],[0,0,-1],[0,1,0]]),
            'glb_x-90': np.array([[1,0,0],[0,0,1],[0,-1,0]]),
            'glb_y+90': np.array([[0,0,1],[0,1,0],[-1,0,0]]),
            'glb_y-90': np.array([[0,0,-1],[0,1,0],[1,0,0]]),
            'glb_z+90': np.array([[0,-1,0],[1,0,0],[0,0,1]]),
            'glb_z-90': np.array([[0,1,0],[-1,0,0],[0,0,1]]),
        }

        # 相机坐标轴旋转 (在 R_hand_to_glb 层面)
        cam_rots = {
            'cam_x+90': np.array([[1,0,0],[0,0,-1],[0,1,0]]),
            'cam_x-90': np.array([[1,0,0],[0,0,1],[0,-1,0]]),
            'cam_y+90': np.array([[0,0,1],[0,1,0],[-1,0,0]]),
            'cam_y-90': np.array([[0,0,-1],[0,1,0],[1,0,0]]),
            'cam_z+90': np.array([[0,-1,0],[1,0,0],[0,0,1]]),
            'cam_z-90': np.array([[0,1,0],[-1,0,0],[0,0,1]]),
        }

        def _compute_metrics(t_ha, R_ha, t_ra, R_ra):
            """计算一组轨迹的对比指标"""
            n_ras = len(t_ra)
            n_hawor = len(t_ha)
            indices = np.linspace(0, n_hawor-1, n_ras, dtype=int)
            t_ha_sampled = t_ha[indices]
            pos0 = np.linalg.norm(t_ha[0] - t_ra[0])
            fwd_h = R_ha[0, :, 2]
            fwd_r = R_ra[0, :, 2]
            fwd0 = np.degrees(np.arccos(np.clip(np.dot(fwd_h, fwd_r), -1, 1)))
            dists = np.linalg.norm(t_ha_sampled - t_ra, axis=1)
            mean_dist = dists.mean()
            # PCA
            ha_centered = t_ha - t_ha.mean(axis=0)
            _, S_h, Vt_h = np.linalg.svd(ha_centered, full_matrices=False)
            ra_centered = t_ra - t_ra.mean(axis=0)
            _, S_r, Vt_r = np.linalg.svd(ra_centered, full_matrices=False)
            pca_angle = np.degrees(np.arccos(np.clip(np.dot(Vt_h[0], Vt_r[0]), -1, 1)))
            # 旋转误差
            rot_errs = []
            for ri in range(n_ras):
                hi = indices[ri]
                fwd_h = -R_ha[hi, :, 2]
                fwd_r = R_ra[ri, :, 2]
                rot_errs.append(np.degrees(np.arccos(np.clip(np.dot(fwd_h, fwd_r), -1, 1))))
            mean_rot = np.mean(rot_errs)
            return pos0, fwd0, mean_dist, mean_rot, pca_angle

        results = []
        # 1. baseline
        pos0, fwd0, md, mr, pca = _compute_metrics(t_hawor_glb, R_c2w_h_glb, t_c2w_r, R_c2w_r)
        results.append(('baseline', pos0, fwd0, md, mr, pca))

        # 2. GLB 世界旋转
        for name, R_glb in glb_rots.items():
            if name == 'baseline':
                continue
            t_temp = (R_glb @ t_hawor_glb.T).T
            R_temp = R_glb @ R_c2w_h_glb
            pos0, fwd0, md, mr, pca = _compute_metrics(t_temp, R_temp, t_c2w_r, R_c2w_r)
            results.append((name, pos0, fwd0, md, mr, pca))

        # 3. 相机坐标旋转 (在 R_hand_to_glb 层面)
        R_c2w_ras0 = R_c2w_r[0]
        t_c2w_ras0 = t_c2w_r[0]
        R_c2w_hawor0 = R_c2w_h[0]
        t_c2w_hawor0 = t_c2w_h[0]
        for name, R_cam in cam_rots.items():
            R_h2g_mod = R_c2w_ras0 @ R_cam @ R_X @ R_c2w_hawor0.T
            t_h2g_mod = t_c2w_ras0 - s * R_h2g_mod @ t_c2w_hawor0
            t_temp = s * (R_h2g_mod @ t_c2w_h.T).T + t_h2g_mod
            R_temp = R_h2g_mod @ R_c2w_h
            pos0, fwd0, md, mr, pca = _compute_metrics(t_temp, R_temp, t_c2w_r, R_c2w_r)
            results.append((name, pos0, fwd0, md, mr, pca))

        # 打印对比表
        print(f'\n{"#":>2} | {"Mode":<12} | {"Pos0":>6} | {"Fwd0":>6} | {"MeanDist":>9} | {"MeanRot":>7} | {"PCA°":>6}')
        print('-'*70)
        best_idx = 0
        best_score = float('inf')
        for i, (name, p0, f0, md, mr, pca) in enumerate(results):
            marker = ' ◀' if i == 0 else ''
            print(f'{i:>2} | {name:<12} | {p0:>6.4f} | {f0:>5.1f}° | {md:>8.4f}m | {mr:>6.1f}° | {pca:>5.1f}°{marker}')
            # 评分: 综合第一帧(必须接近0) + 平均距离 + 旋转误差 + PCA
            if p0 < 0.001 and f0 < 1.0:
                score = md * 100 + mr * 0.5 + pca * 0.3
                if score < best_score:
                    best_score = score
                    best_idx = i

        print(f'\n🏆 最佳: [{results[best_idx][0]}]  '
              f'Pos0={results[best_idx][1]:.4f}m Fwd0={results[best_idx][2]:.1f}° '
              f'MeanDist={results[best_idx][3]:.4f}m MeanRot={results[best_idx][4]:.1f}°')

        # 使用最佳方案更新轨迹
        best_name = results[best_idx][0]
        if best_name in glb_rots:
            R_best = glb_rots[best_name]
            t_hawor_glb = (R_best @ t_hawor_glb.T).T
            R_c2w_h_glb = R_best @ R_c2w_h_glb
            print(f'\n✅ 应用最佳方案: {best_name}')
        elif best_name in cam_rots:
            R_cam_best = cam_rots[best_name]
            R_h2g_best = R_c2w_ras0 @ R_cam_best @ R_X @ R_c2w_hawor0.T
            t_h2g_best = t_c2w_ras0 - s * R_h2g_best @ t_c2w_hawor0
            t_hawor_glb = s * (R_h2g_best @ t_c2w_h.T).T + t_h2g_best
            R_c2w_h_glb = R_h2g_best @ R_c2w_h
            print(f'\n✅ 应用最佳方案: {best_name}')
        print('='*70 + '\n')

    # ========== 8. 统计轨迹差异 ==========
    print('\n=== GLB 世界空间 (Z-UP) ===')
    print('各轴范围表示相机中心在 X/Y/Z 方向上的运动幅度:')
    print(f'HaWoR 相机: X=[{t_hawor_glb[:,0].min():.4f},{t_hawor_glb[:,0].max():.4f}]  '
          f'Y=[{t_hawor_glb[:,1].min():.4f},{t_hawor_glb[:,1].max():.4f}]  '
          f'Z=[{t_hawor_glb[:,2].min():.4f},{t_hawor_glb[:,2].max():.4f}]  '
          f'Z<0: {(t_hawor_glb[:,2]<0).sum()}/{len(t_hawor_glb)}')
    print(f'RAS 相机:   X=[{t_c2w_r[:,0].min():.4f},{t_c2w_r[:,0].max():.4f}]  '
          f'Y=[{t_c2w_r[:,1].min():.4f},{t_c2w_r[:,1].max():.4f}]  '
          f'Z=[{t_c2w_r[:,2].min():.4f},{t_c2w_r[:,2].max():.4f}]  '
          f'Z<0: {(t_c2w_r[:,2]<0).sum()}/{len(t_c2w_r)}')

    # 逐帧距离 (重采样后)
    indices = np.linspace(0, n_hawor - 1, n_ras, dtype=int)
    t_hawor_sampled = t_hawor_glb[indices]
    distances = np.linalg.norm(t_hawor_sampled - t_c2w_r, axis=1)
    print(f'\n逐帧距离: min={distances.min():.4f}m  max={distances.max():.4f}m  '
          f'mean={distances.mean():.4f}m')

    # ========== 8. 3D 可视化 ==========
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 自动计算坐标轴范围 (在轨迹范围基础上加 padding)
    padding = 0.1
    for coord in range(3):
        vmin = min(t_hawor_glb[:, coord].min(), t_c2w_r[:, coord].min()) - padding
        vmax = max(t_hawor_glb[:, coord].max(), t_c2w_r[:, coord].max()) + padding
        if coord == 0: ax.set_xlim(vmin, vmax)
        elif coord == 1: ax.set_ylim(vmin, vmax)
        else: ax.set_zlim(vmin, vmax)

    # HaWoR 轨迹 (蓝色): 实线 + 起点(圆点) + 终点(叉号)
    ax.plot(t_hawor_glb[:,0], t_hawor_glb[:,1], t_hawor_glb[:,2], 'b-', alpha=0.7,
             linewidth=2, label=f'HaWoR ({n_hawor}f)')
    ax.scatter(*t_hawor_glb[0], c='b', s=100, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
    ax.scatter(*t_hawor_glb[-1], c='b', s=60, marker='x', edgecolors='k', linewidths=0.5, zorder=5)

    # RAS 轨迹 (红色): 实线 + 起点(方块) + 终点(菱形)
    ax.plot(t_c2w_r[:,0], t_c2w_r[:,1], t_c2w_r[:,2], 'r-', alpha=0.9,
             linewidth=3, label=f'RAS ({n_ras}f)')
    ax.scatter(*t_c2w_r[0], c='r', s=100, marker='s', edgecolors='k', linewidths=0.5, zorder=5)
    ax.scatter(*t_c2w_r[-1], c='r', s=60, marker='D', edgecolors='k', linewidths=0.5, zorder=5)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    title = 'Camera Trajectories in GLB World (Z-UP)\nblue=HaWoR red=RAS'
    if args.z90:
        title += '\n[HaWoR + GLB Z-90°]'
    if args.cam90:
        title += '\n[HaWoR + Camera Z-90°]'
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)

    # 第一帧朝向箭头: 显示相机 forward 方向 (HaWoR=蓝, RAS=红)
    arrow_len = 0.05
    for origin, fwd, color in [(t_hawor_glb[0], fwd_hawor, 'blue'),
                                (t_c2w_r[0], fwd_ras, 'red')]:
        ax.quiver(origin[0], origin[1], origin[2],
                   fwd[0], fwd[1], fwd[2],
                   length=arrow_len, color=color, linewidth=2, alpha=0.8)

    plt.tight_layout()
    plt.draw()

    # ========== 8. 逐帧误差图 ==========
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    frames = np.arange(n_ras)

    ax1.plot(frames, pos_errors, 'b-', linewidth=1.5)
    ax1.axhline(pos_errors.mean(), color='r', ls='--', alpha=0.5, label=f'mean={pos_errors.mean():.3f}m')
    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Per-Frame Position & Rotation Difference (HaWoR → RAS)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(frames, rot_angles, 'b-', linewidth=1.5)
    ax2.axhline(rot_angles.mean(), color='r', ls='--', alpha=0.5, label=f'mean={rot_angles.mean():.1f}°')
    ax2.axhline(rot_angles[-1], color='g', ls='--', alpha=0.5, label=f'final={rot_angles[-1]:.1f}°')
    ax2.set_xlabel('Frame (RAS index)')
    ax2.set_ylabel('Rotation Error (deg)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()