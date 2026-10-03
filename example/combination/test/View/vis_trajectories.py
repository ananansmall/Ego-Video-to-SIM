#!/usr/bin/env python3
"""
vis_trajectories.py — 可视化 cam0 空间轨迹对比
"""

import numpy as np
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EXAMPLES = {
    '121': {
        'ras_dir': '/home/an/data/ras/121_C5_CellPhone_161deg_vggt_omega',
        'hawor_npz': '/home/an/data/hawor/121_C5_CellPhone_161deg/reconstruction/hawor_results_0_600.npz',
        'slam_npz': '/home/an/data/hawor/121_C5_CellPhone_161deg/SLAM/hawor_slam_w_scale_0_600.npz',
    },
    'hoi4d': {
        'ras_dir': '/home/an/data/ras/hoi4d1_vggt_omega',
        'hawor_npz': '/home/an/data/hawor/hoi4d/reconstruction/hawor_results_0_600.npz',
    },
    '7': {
        'ras_dir': '/home/an/data/ras/7_vggt_omega',
        'hawor_npz': '/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz',
        'slam_npz': '/home/an/data/hawor/7/SLAM/hawor_slam_w_scale_0_113.npz',
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
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


def resample(points, n_target):
    n_src = len(points)
    indices = np.linspace(0, n_src - 1, n_target, dtype=int)
    return points[indices]


output_dir = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/View/traj_plots'
os.makedirs(output_dir, exist_ok=True)

for name, paths in EXAMPLES.items():
    print(f"\n处理 {name}...")

    # 加载 RAS
    R_r, t_r = load_ras_extrinsics(paths['ras_dir'])
    R_r0 = R_r[0]
    t_r0 = t_r[0]
    n_ras = len(t_r)

    # 加载重建
    recon = dict(np.load(paths['hawor_npz'], allow_pickle=True))
    R_h = recon['R_c2w']
    t_h = recon['t_c2w']
    R_h0 = R_h[0]
    t_h0 = t_h[0]
    n_ha = len(t_h)

    # RAS cam0
    ras_cam0 = np.array([R_r0.T @ (t_r[i] - t_r0) for i in range(n_ras)])

    # 重建 cam0
    ha_cam0 = np.array([R_h0.T @ (t_h[i] - t_h0) for i in range(n_ha)])
    ha_cam0_s = resample(ha_cam0, n_ras)

    # SLAM (如果有)
    slam_cam0_s = None
    slam_cam0_rx_s = None
    if 'slam_npz' in paths and paths['slam_npz'] and os.path.exists(paths['slam_npz']):
        slam = dict(np.load(paths['slam_npz'], allow_pickle=True))
        traj = slam['traj']
        t_slam = traj[:, :3]
        q_slam = traj[:, 3:7]
        R_slam = np.array([quat_to_rotmat(q_slam[i]) for i in range(len(q_slam))])
        R_s0 = R_slam[0]
        t_s0 = t_slam[0]

        slam_cam0 = np.array([R_s0.T @ (t_slam[i] - t_s0) for i in range(len(t_slam))])
        slam_cam0_s = resample(slam_cam0, n_ras)
        slam_cam0_rx = np.array([Rx_hand @ (R_s0.T @ (t_slam[i] - t_s0)) for i in range(len(t_slam))])
        slam_cam0_rx_s = resample(slam_cam0_rx, n_ras)

    # 计算 PCA 对比
    def pca(pts):
        c = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        return Vt[0]

    def pca_angle(a, b):
        a_c = a - a.mean(axis=0)
        b_c = b - b.mean(axis=0)
        _, _, Vt_a = np.linalg.svd(a_c, full_matrices=False)
        _, _, Vt_b = np.linalg.svd(b_c, full_matrices=False)
        dot = np.clip(np.dot(Vt_a[0], Vt_b[0]), -1, 1)
        if dot < 0:
            dot = -dot
        return np.degrees(np.arccos(dot))

    # ===== 3D 可视化 =====
    fig = plt.figure(figsize=(18, 6))

    # 子图1: 3D 轨迹
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(ras_cam0[:, 0], ras_cam0[:, 1], ras_cam0[:, 2], 'b-', linewidth=2, label=f'RAS ({n_ras}帧)')
    ax1.plot(ha_cam0_s[:, 0], ha_cam0_s[:, 1], ha_cam0_s[:, 2], 'r--', linewidth=2, label=f'HaWoR重建 ({n_ha}帧)')
    if slam_cam0_s is not None:
        ax1.plot(slam_cam0_s[:, 0], slam_cam0_s[:, 1], slam_cam0_s[:, 2], 'g:', linewidth=2, label=f'SLAM原始')
    if slam_cam0_rx_s is not None:
        ax1.plot(slam_cam0_rx_s[:, 0], slam_cam0_rx_s[:, 1], slam_cam0_rx_s[:, 2], 'm-.', linewidth=2, label=f'SLAM@Rx')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{name}: cam0 空间轨迹')
    ax1.legend()

    # 子图2: XY 投影
    ax2 = fig.add_subplot(132)
    ax2.plot(ras_cam0[:, 0], ras_cam0[:, 1], 'b-', linewidth=2, label='RAS')
    ax2.plot(ha_cam0_s[:, 0], ha_cam0_s[:, 1], 'r--', linewidth=2, label='HaWoR重建')
    if slam_cam0_s is not None:
        ax2.plot(slam_cam0_s[:, 0], slam_cam0_s[:, 1], 'g:', linewidth=2, label='SLAM原始')
    if slam_cam0_rx_s is not None:
        ax2.plot(slam_cam0_rx_s[:, 0], slam_cam0_rx_s[:, 1], 'm-.', linewidth=2, label='SLAM@Rx')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'{name}: XY 投影')
    ax2.axis('equal')
    ax2.legend()
    ax2.grid(True)

    # 子图3: 各轴分量随时间变化
    ax3 = fig.add_subplot(133)
    t_axis = np.arange(n_ras)
    ax3.plot(t_axis, ras_cam0[:, 0], 'b-', label='RAS X')
    ax3.plot(t_axis, ras_cam0[:, 1], 'b--', label='RAS Y')
    ax3.plot(t_axis, ras_cam0[:, 2], 'b:', label='RAS Z')
    ax3.plot(t_axis, ha_cam0_s[:, 0], 'r-', label='HaWoR X')
    ax3.plot(t_axis, ha_cam0_s[:, 1], 'r--', label='HaWoR Y')
    ax3.plot(t_axis, ha_cam0_s[:, 2], 'r:', label='HaWoR Z')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position (m)')
    ax3.set_title(f'{name}: 各轴分量 (PCA={pca_angle(ha_cam0_s, ras_cam0):.1f}°)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{name}_trajectories.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  保存到 {out_path}")

    # ===== PCA 主方向对比图 =====
    fig2 = plt.figure(figsize=(12, 5))

    # 显示 PCA 主方向
    ax_pca = fig2.add_subplot(121, projection='3d')
    pca_r = pca(ras_cam0)
    pca_h = pca(ha_cam0_s)

    # 从原点画 PCA 方向
    origin = np.zeros(3)
    for arr, label, color, scale in [(pca_r, 'RAS', 'b', 0.1), (pca_h, 'HaWoR', 'r', 0.1)]:
        ax_pca.quiver(*origin, *(-arr * scale), color=color, arrow_length_ratio=0.1, linewidth=3)
        ax_pca.quiver(*origin, *(arr * scale), color=color, arrow_length_ratio=0.1, linewidth=3, label=label)

    ax_pca.set_xlim([-0.15, 0.15])
    ax_pca.set_ylim([-0.15, 0.15])
    ax_pca.set_zlim([-0.15, 0.15])
    ax_pca.set_xlabel('X')
    ax_pca.set_ylabel('Y')
    ax_pca.set_zlabel('Z')
    ax_pca.set_title(f'{name}: PCA 主方向 (夹角={pca_angle(ha_cam0_s, ras_cam0):.1f}°)')
    ax_pca.legend()

    # 各轴范围对比
    ax_range = fig2.add_subplot(122)
    labels = ['X', 'Y', 'Z']
    ras_ranges = [ras_cam0[:, i].max() - ras_cam0[:, i].min() for i in range(3)]
    ha_ranges = [ha_cam0_s[:, i].max() - ha_cam0_s[:, i].min() for i in range(3)]

    x = np.arange(3)
    width = 0.35
    ax_range.bar(x - width/2, ras_ranges, width, label='RAS', color='b', alpha=0.7)
    ax_range.bar(x + width/2, ha_ranges, width, label='HaWoR', color='r', alpha=0.7)
    ax_range.set_xticks(x)
    ax_range.set_xticklabels(labels)
    ax_range.set_ylabel('Range (m)')
    ax_range.set_title(f'{name}: 各轴运动范围')
    ax_range.legend()
    ax_range.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, f'{name}_pca.png')
    plt.savefig(out_path2, dpi=150)
    plt.close()
    print(f"  保存到 {out_path2}")

print(f"\n所有可视化保存到 {output_dir}/")
print("文件列表:")
for f in sorted(os.listdir(output_dir)):
    print(f"  {f}")