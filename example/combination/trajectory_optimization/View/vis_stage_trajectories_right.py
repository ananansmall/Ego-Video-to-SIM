#!/usr/bin/env python3
"""
Stage 0/1/2/3 轨迹对比图 (gripper_right)
包含: 原始 MANO 轨迹, Stage0 (纯 retargeting), Stage1 (CEM 优化), Stage2 (轨迹重建), Stage3 (完整管线)
所有坐标均为 GLB 坐标系
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

OUTPUT_DIR = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_right'
SAVE_DIR = OUTPUT_DIR

# ============================================================
# 1. 加载数据
# ============================================================

# --- Stage0: 纯 MANO retargeting ---
s0_qpos = np.load(f'{OUTPUT_DIR}/stage0/grasp_gripper_only_right_qpos.npy')
s0_gripper_pos = s0_qpos[:, :3]  # qpos 前3维是位置
s0_obj_z = np.load(f'{OUTPUT_DIR}/stage0/run_obj_z_traj.npy')
s0_obj_xy = np.load(f'{OUTPUT_DIR}/stage0/run_obj_xy_traj.npy')
s0_obj_pos = np.column_stack([s0_obj_xy, s0_obj_z])
s0_frames = np.arange(len(s0_qpos))

# --- Stage1: CEM 抓取姿态优化 ---
s1_data = np.load(f'{OUTPUT_DIR}/stage1/stage1/best_grasp.npz', allow_pickle=True)
s1_pos = s1_data['pos']  # 单点抓取姿态
s1_obj_lift = float(s1_data['obj_lift'])
s1_obj_drift = float(s1_data['obj_xy_drift'])

# --- Stage2: 轨迹重建 ---
s2_data = np.load(f'{OUTPUT_DIR}/stage2/stage2/stage2_recon.npz', allow_pickle=True)
s2_pos = s2_data['pos']  # (25, 3)
s2_frames = s2_data['frames']

# --- Stage3: 完整管线 ---
s3_data = np.load(f'{OUTPUT_DIR}/stage3/stage3/stage3_result.npz', allow_pickle=True)
s3_result = s3_data['result'].item()
s3_mano_pos = np.array(s3_result['mano_pos'])  # (112, 3)
s3_opt_pos = np.array(s3_result['opt_pos'])    # (112, 3)
s3_obj_pos = np.array(s3_result['obj_pos_traj'])  # (112, 3)
s3_frames = np.arange(len(s3_mano_pos))
s3_obj_lift = float(s3_result['obj_lift'])
s3_xy_drift = float(s3_result['xy_drift'])

# --- 原始 MANO 轨迹 (从 stage3 的 mano_pos 获取) ---
mano_pos = s3_mano_pos
mano_frames = s3_frames

print(f"Stage0: {len(s0_gripper_pos)} frames")
print(f"Stage1: single grasp pose at {s1_pos}")
print(f"Stage2: {len(s2_pos)} frames (frames {s2_frames[0]}-{s2_frames[-1]})")
print(f"Stage3: {len(s3_opt_pos)} frames")
print(f"MANO: {len(mano_pos)} frames")

# ============================================================
# 2. 绘制 3D 轨迹对比图
# ============================================================
fig = plt.figure(figsize=(20, 16))

# --- 子图1: 3D 轨迹对比 ---
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# 原始 MANO 轨迹
ax1.plot(mano_pos[:, 0], mano_pos[:, 1], mano_pos[:, 2], 'k--', alpha=0.4, linewidth=1, label='Original MANO')
# Stage0 gripper
ax1.plot(s0_gripper_pos[:, 0], s0_gripper_pos[:, 1], s0_gripper_pos[:, 2], 'b-', linewidth=1.5, label='Stage0 (retargeting)')
# Stage1 grasp pose (single point)
ax1.scatter(*s1_pos, c='green', s=100, marker='*', zorder=5, label=f'Stage1 (CEM, lift={s1_obj_lift:.1f}cm)')
# Stage2 trajectory
ax1.plot(s2_pos[:, 0], s2_pos[:, 1], s2_pos[:, 2], 'r-', linewidth=2, label='Stage2 (reconstruct)')
# Stage3 opt trajectory
ax1.plot(s3_opt_pos[:, 0], s3_opt_pos[:, 1], s3_opt_pos[:, 2], 'm-', linewidth=2, label=f'Stage3 (full, lift={s3_obj_lift:.1f}cm)')
# Stage3 MANO reference
ax1.plot(s3_mano_pos[:, 0], s3_mano_pos[:, 1], s3_mano_pos[:, 2], 'm--', alpha=0.3, linewidth=1)

# 物体轨迹 (stage3)
ax1.plot(s3_obj_pos[:, 0], s3_obj_pos[:, 1], s3_obj_pos[:, 2], 'c-', linewidth=1.5, label='Object (Stage3)')

ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Trajectory Comparison (GLB coords)')
ax1.legend(fontsize=8, loc='upper left')
ax1.view_init(elev=25, azim=45)

# --- 子图2: XZ 平面投影 ---
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(mano_pos[:, 0], mano_pos[:, 2], 'k--', alpha=0.4, linewidth=1, label='Original MANO')
ax2.plot(s0_gripper_pos[:, 0], s0_gripper_pos[:, 2], 'b-', linewidth=1.5, label='Stage0')
ax2.scatter(*s1_pos[[0, 2]], c='green', s=100, marker='*', zorder=5, label=f'Stage1 (lift={s1_obj_lift:.1f}cm)')
ax2.plot(s2_pos[:, 0], s2_pos[:, 2], 'r-', linewidth=2, label='Stage2')
ax2.plot(s3_opt_pos[:, 0], s3_opt_pos[:, 2], 'm-', linewidth=2, label=f'Stage3 (lift={s3_obj_lift:.1f}cm)')
ax2.plot(s3_obj_pos[:, 0], s3_obj_pos[:, 2], 'c-', linewidth=1.5, label='Object (Stage3)')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
ax2.set_title('XZ Plane Projection')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- 子图3: Z 轴随帧数变化 ---
ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(mano_frames, mano_pos[:, 2], 'k--', alpha=0.4, linewidth=1, label='Original MANO')
ax3.plot(s0_frames, s0_gripper_pos[:, 2], 'b-', linewidth=1.5, label='Stage0')
ax3.scatter([0], [s1_pos[2]], c='green', s=100, marker='*', zorder=5, label=f'Stage1 (z={s1_pos[2]:.3f})')
ax3.plot(s2_frames, s2_pos[:, 2], 'r-', linewidth=2, label='Stage2')
ax3.plot(s3_frames, s3_opt_pos[:, 2], 'm-', linewidth=2, label=f'Stage3 (lift={s3_obj_lift:.1f}cm)')
ax3.plot(s3_frames, s3_obj_pos[:, 2], 'c-', linewidth=1.5, label='Object (Stage3)')
ax3.set_xlabel('Frame')
ax3.set_ylabel('Z (m)')
ax3.set_title('Gripper Z vs Frame')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# --- 子图4: XY 平面投影 ---
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(mano_pos[:, 0], mano_pos[:, 1], 'k--', alpha=0.4, linewidth=1, label='Original MANO')
ax4.plot(s0_gripper_pos[:, 0], s0_gripper_pos[:, 1], 'b-', linewidth=1.5, label='Stage0')
ax4.scatter(*s1_pos[[0, 1]], c='green', s=100, marker='*', zorder=5, label=f'Stage1')
ax4.plot(s2_pos[:, 0], s2_pos[:, 1], 'r-', linewidth=2, label='Stage2')
ax4.plot(s3_opt_pos[:, 0], s3_opt_pos[:, 1], 'm-', linewidth=2, label=f'Stage3 (drift={s3_xy_drift:.1f}cm)')
ax4.plot(s3_obj_pos[:, 0], s3_obj_pos[:, 1], 'c-', linewidth=1.5, label='Object (Stage3)')
ax4.set_xlabel('X (m)')
ax4.set_ylabel('Y (m)')
ax4.set_title('XY Plane Projection')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.suptitle('Gripper Right - Stage 0/1/2/3 Trajectory Comparison (GLB)', fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = f'{SAVE_DIR}/trajectory_comparison_gripper_right.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.close()

# ============================================================
# 3. 打印关键数据对比
# ============================================================
print("\n" + "="*60)
print("TRAJECTORY COMPARISON SUMMARY (GLB coords)")
print("="*60)
print(f"\n{'Metric':<30} {'Stage0':<15} {'Stage1':<15} {'Stage2':<15} {'Stage3':<15}")
print("-"*90)

# Gripper start/end Z
print(f"{'Gripper start Z (m)':<30} {s0_gripper_pos[0,2]:<15.4f} {'-':<15} {s2_pos[0,2]:<15.4f} {s3_opt_pos[0,2]:<15.4f}")
print(f"{'Gripper end Z (m)':<30} {s0_gripper_pos[-1,2]:<15.4f} {'-':<15} {s2_pos[-1,2]:<15.4f} {s3_opt_pos[-1,2]:<15.4f}")
print(f"{'Gripper Z range (m)':<30} {s0_gripper_pos[:,2].max()-s0_gripper_pos[:,2].min():<15.4f} {'-':<15} {s2_pos[:,2].max()-s2_pos[:,2].min():<15.4f} {s3_opt_pos[:,2].max()-s3_opt_pos[:,2].min():<15.4f}")

# Object lift
print(f"{'Object lift (cm)':<30} {'-':<15} {s1_obj_lift:<15.1f} {'-':<15} {s3_obj_lift:<15.1f}")
print(f"{'Object XY drift (cm)':<30} {'-':<15} {s1_obj_drift:<15.1f} {'-':<15} {s3_xy_drift:<15.1f}")

# Frame counts
print(f"{'Frames':<30} {len(s0_gripper_pos):<15} {'1 (grasp pose)':<15} {len(s2_pos):<15} {len(s3_opt_pos):<15}")

# Gripper position at frame 50 (approximate grasp point)
f50_idx = min(49, len(s0_gripper_pos)-1)
print(f"\nGripper pos at frame ~50:")
print(f"  Stage0: {s0_gripper_pos[f50_idx]}")
print(f"  Stage3 opt: {s3_opt_pos[min(49, len(s3_opt_pos)-1)]}")
print(f"  Stage3 mano: {s3_mano_pos[min(49, len(s3_mano_pos)-1)]}")