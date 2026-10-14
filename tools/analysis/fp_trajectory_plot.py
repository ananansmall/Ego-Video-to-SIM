"""用 FoundationPose 位姿绘制 3D 轨迹图（世界系, Z-up）。

复用 trajectory_video_overlay 的三联图布局, 数据源换成
foundationpose_out/ob_in_world/*.txt。输出 fp_trajectory/<obj>_trajectory.png
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    ap.add_argument('--obj', default='toy_inst1')
    ap.add_argument('--out_dir', default=None)
    args = ap.parse_args()
    sd = args.scene_dir
    fp_dir = os.path.join(sd, 'foundationpose_out', 'ob_in_world')
    out_dir = args.out_dir or os.path.join(sd, 'fp_trajectory')
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(os.listdir(fp_dir))
    poses = np.stack([np.loadtxt(os.path.join(fp_dir, f)) for f in files])
    frames = np.array([int(os.path.splitext(f)[0]) for f in files])
    centers = poses[:, :3, 3]
    rotmat = poses[:, :3, :3]

    # 相机轨迹做空间参照（extrinsics 的逆给出 cam2world）
    ext_dir = os.path.join(sd, 'extrinsics')
    cam_files = sorted(os.listdir(ext_dir),
                       key=lambda x: int(os.path.splitext(x)[0]))
    cam_pos = np.stack([np.linalg.inv(np.loadtxt(os.path.join(ext_dir, f)))[:3, 3]
                        for f in cam_files])

    from scipy.spatial.transform import Rotation
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 姿态曲线不能用欧拉角/旋转向量: 物体总转角 167°, rotvec 在 ±π 边界会跳变
    # (实测逐帧真实旋转仅 4.8°/帧, 数据健康, 纯属表示退化)。
    # 改用「相对首帧」的旋转矩阵元素, 连续且无奇异。
    R0 = rotmat[0]
    rel_mats = np.einsum('ij,njk->nik', R0.T, rotmat)   # R0^T @ R_t
    rel_xyz = rel_mats.reshape(len(rotmat), 9)[:, [0, 4, 8]]  # R00, R11, R22
    step_ang = np.degrees([np.arccos(np.clip((np.trace(rotmat[i - 1].T @ rotmat[i]) - 1) / 2, -1, 1))
                           for i in range(1, len(rotmat))])
    tsec = frames / 2.0
    n = len(centers)

    fig = plt.figure(figsize=(15, 5.4))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # ---- (1) 3D 轨迹 ----
    cmap = plt.get_cmap('turbo')
    ax.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
            color='gray', alpha=0.35, linewidth=1.1, label='camera path')
    for i in range(n - 1):
        ax.plot(centers[i:i + 2, 0], centers[i:i + 2, 1], centers[i:i + 2, 2],
                color=cmap(i / max(n - 1, 1)), linewidth=2.6, zorder=3)
    ax.scatter(*centers[0], s=170, c='#00c853', edgecolors='k',
               linewidths=1.5, label=f'start f{frames[0]}', depthshade=False, zorder=6)
    ax.scatter(*centers[-1], s=180, c='#ff6d00', edgecolors='k',
               linewidths=1.5, label=f'end f{frames[-1]}', depthshade=False, zorder=6)
    for frac in (0.25, 0.5, 0.75):
        i = int(frac * (n - 1))
        ax.scatter(*centers[i], s=48, c=cmap(i / max(n - 1, 1)),
                   edgecolors='k', linewidths=0.9, depthshade=False, zorder=5)
        ax.text(centers[i, 0], centers[i, 1], centers[i, 2] + 0.02,
                f'f{frames[i]}', fontsize=8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('FoundationPose 6DoF trajectory (world, Z-up)', fontsize=11)
    ax.legend(fontsize=8.5, loc='best')
    ax.set_box_aspect((max(np.ptp(centers[:, 0]), 0.1),
                       max(np.ptp(centers[:, 1]), 0.1),
                       max(np.ptp(centers[:, 2]), 0.1)))

    # ---- (2) 高度曲线 ----
    ax2.plot(tsec, centers[:, 2], linewidth=2.2, color='#d62728')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Height (lift phase visible)', fontsize=11)
    ax2.grid(alpha=0.3)
    z0, z1 = centers[:, 2].min(), centers[:, 2].max()
    ax2.axhspan(z0, z0 + 0.02, color='#00c853', alpha=0.12)
    ax2.annotate(f'lift {z1 - z0:.3f}m', xy=(tsec[int(np.argmax(centers[:, 2]))],
                                             centers[:, 2].max()),
                 xytext=(10, -18), textcoords='offset points', fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='k'))

    # ---- (3) 相对首帧的旋转矩阵对角元素(连续表示) ----
    ax3.plot(tsec, rel_xyz[:, 0], label='R00', linewidth=1.8)
    ax3.plot(tsec, rel_xyz[:, 1], label='R11', linewidth=1.8)
    ax3.plot(tsec, rel_xyz[:, 2], label='R22', linewidth=1.8)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('rotation matrix element')
    ax3.set_ylim(-1.05, 1.05)
    tot_ang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ rotmat[-1]) - 1) / 2, -1, 1)))
    ax3.set_title(f'Orientation (rel. to frame 0)  total {tot_ang:.0f}°, '
                  f'step {step_ang.mean():.1f}°/frame (max {step_ang.max():.0f}°)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax4 = ax3.twinx()
    ax4.plot(tsec[1:], step_ang, color='#888', linewidth=0.9, alpha=0.7)
    ax4.set_ylabel('per-frame Δ (deg)', fontsize=8, color='#888')

    disp = float(np.linalg.norm(centers[-1] - centers[0]))
    fig.suptitle(f'{args.obj}  |  net displacement {disp:.4f} m  |  '
                 f'{n} frames  |  FoundationPose', fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, f'{args.obj}_trajectory.png')
    fig.savefig(out, dpi=130)
    print(f'saved {out}')
    print(f'net displacement = {disp:.4f} m, '
          f'Z range = {centers[:,2].min():.4f}..{centers[:,2].max():.4f} m')


if __name__ == '__main__':
    main()