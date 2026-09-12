#!/usr/bin/env python3
"""
Stage0 误差分析 + 平滑处理 (参考 do-as-i-do 的 tracking_error 和 moving_average_filter)
对比: 原始 MANO vs Stage0 retargeting vs 平滑后 Stage0
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

OUTPUT_DIR = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_right'

# ============================================================
# 1. 加载数据
# ============================================================

# Stage0: 纯 retargeting 的 gripper qpos
s0_qpos = np.load(f'{OUTPUT_DIR}/stage0/grasp_gripper_only_right_qpos.npy')
s0_gripper_pos = s0_qpos[:, :3]  # (113, 3)
s0_gripper_q = s0_qpos[:, 3]     # (113,) gripper open/close

# Stage3: 包含原始 MANO 轨迹 (作为参考)
s3_data = np.load(f'{OUTPUT_DIR}/stage3/stage3/stage3_result.npz', allow_pickle=True)
s3_result = s3_data['result'].item()
s3_mano_pos = np.array(s3_result['mano_pos'])  # (112, 3) 原始 MANO
s3_opt_pos = np.array(s3_result['opt_pos'])    # (112, 3) 优化后
s3_obj_pos = np.array(s3_result['obj_pos_traj'])  # (112, 3) 物体

# Stage1
s1_data = np.load(f'{OUTPUT_DIR}/stage1/stage1/best_grasp.npz', allow_pickle=True)
s1_pos = s1_data['pos']

# Stage2
s2_data = np.load(f'{OUTPUT_DIR}/stage2/stage2/stage2_recon.npz', allow_pickle=True)
s2_pos = s2_data['pos']
s2_frames = s2_data['frames']

print("="*70)
print("STAGE0 误差分析 (参考 do-as-i-do tracking_error)")
print("="*70)

# ============================================================
# 2. 对齐帧数: Stage0 (113帧) vs Stage3 MANO (112帧)
# ============================================================
# Stage0 有 113 帧, Stage3 有 112 帧, 取前 112 帧对齐
n_align = min(len(s0_gripper_pos), len(s3_mano_pos))
s0_aligned = s0_gripper_pos[:n_align]
mano_aligned = s3_mano_pos[:n_align]
opt_aligned = s3_opt_pos[:n_align]
obj_aligned = s3_obj_pos[:n_align]

print(f"\n对齐帧数: {n_align}")
print(f"  Stage0: {len(s0_gripper_pos)} -> {n_align}")
print(f"  MANO:   {len(s3_mano_pos)} -> {n_align}")

# ============================================================
# 3. 计算位置误差 (参考 do-as-i-do compute_object_tracking_error)
# ============================================================
# Stage0 vs MANO
pos_err_s0_mano = np.linalg.norm(s0_aligned - mano_aligned, axis=1)
# Stage3 opt vs MANO
pos_err_opt_mano = np.linalg.norm(opt_aligned - mano_aligned, axis=1)
# Stage3 opt vs Stage0
pos_err_opt_s0 = np.linalg.norm(opt_aligned - s0_aligned, axis=1)

print(f"\n--- 位置误差 (m) ---")
print(f"  Stage0 vs MANO:  mean={pos_err_s0_mano.mean():.4f}, max={pos_err_s0_mano.max():.4f}, std={pos_err_s0_mano.std():.4f}")
print(f"  Stage3 opt vs MANO: mean={pos_err_opt_mano.mean():.4f}, max={pos_err_opt_mano.max():.4f}, std={pos_err_opt_mano.std():.4f}")
print(f"  Stage3 opt vs Stage0: mean={pos_err_opt_s0.mean():.4f}, max={pos_err_opt_s0.max():.4f}, std={pos_err_opt_s0.std():.4f}")

# 分轴误差
for axis, name in enumerate(['X', 'Y', 'Z']):
    err_s0 = np.abs(s0_aligned[:, axis] - mano_aligned[:, axis])
    err_opt = np.abs(opt_aligned[:, axis] - mano_aligned[:, axis])
    print(f"  {name}-axis: Stage0 mean={err_s0.mean():.4f}m, Stage3 opt mean={err_opt.mean():.4f}m")

# ============================================================
# 4. 平滑处理 (参考 do-as-i-do moving_average_filter, window=5)
# ============================================================
def moving_average_filter(signal_data, window_size=5):
    """do-as-i-do 风格的移动平均滤波"""
    return np.convolve(signal_data, np.ones(window_size) / window_size, mode='valid')

# 对 Stage0 gripper 位置做平滑
s0_smoothed = np.zeros((len(s0_aligned) - 4, 3))
for i in range(3):
    s0_smoothed[:, i] = moving_average_filter(s0_aligned[:, i], window_size=5)

# 对 MANO 也做平滑
mano_smoothed = np.zeros((len(mano_aligned) - 4, 3))
for i in range(3):
    mano_smoothed[:, i] = moving_average_filter(mano_aligned[:, i], window_size=5)

# 平滑后误差
pos_err_smooth = np.linalg.norm(s0_smoothed - mano_smoothed, axis=1)
print(f"\n--- 平滑后 (window=5) ---")
print(f"  Stage0_smoothed vs MANO_smoothed: mean={pos_err_smooth.mean():.4f}, max={pos_err_smooth.max():.4f}")
print(f"  平滑前: mean={pos_err_s0_mano.mean():.4f}")
print(f"  平滑后: mean={pos_err_smooth.mean():.4f}")
print(f"  改善: {(1 - pos_err_smooth.mean()/pos_err_s0_mano.mean())*100:.1f}%")

# ============================================================
# 5. 速度/加速度分析
# ============================================================
# 计算速度
vel_s0 = np.diff(s0_aligned, axis=0) * 30  # 30Hz
vel_mano = np.diff(mano_aligned, axis=0) * 30
vel_opt = np.diff(opt_aligned, axis=0) * 30

print(f"\n--- 速度 (m/s) ---")
print(f"  Stage0: mean={np.linalg.norm(vel_s0, axis=1).mean():.4f}, max={np.linalg.norm(vel_s0, axis=1).max():.4f}")
print(f"  MANO:   mean={np.linalg.norm(vel_mano, axis=1).mean():.4f}, max={np.linalg.norm(vel_mano, axis=1).max():.4f}")
print(f"  Stage3: mean={np.linalg.norm(vel_opt, axis=1).mean():.4f}, max={np.linalg.norm(vel_opt, axis=1).max():.4f}")

# 计算加速度
acc_s0 = np.diff(vel_s0, axis=0) * 30
acc_mano = np.diff(vel_mano, axis=0) * 30
acc_opt = np.diff(vel_opt, axis=0) * 30

print(f"\n--- 加速度 (m/s²) ---")
print(f"  Stage0: mean={np.linalg.norm(acc_s0, axis=1).mean():.4f}, max={np.linalg.norm(acc_s0, axis=1).max():.4f}")
print(f"  MANO:   mean={np.linalg.norm(acc_mano, axis=1).mean():.4f}, max={np.linalg.norm(acc_mano, axis=1).max():.4f}")
print(f"  Stage3: mean={np.linalg.norm(acc_opt, axis=1).mean():.4f}, max={np.linalg.norm(acc_opt, axis=1).max():.4f}")

# ============================================================
# 6. 绘制对比图
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))

# --- 子图1: 位置误差随帧数 ---
ax = axes[0, 0]
ax.plot(range(n_align), pos_err_s0_mano * 100, 'b-', linewidth=1.5, label='Stage0 vs MANO')
ax.plot(range(n_align), pos_err_opt_mano * 100, 'm-', linewidth=1.5, label='Stage3 opt vs MANO')
ax.plot(range(n_align), pos_err_opt_s0 * 100, 'r--', linewidth=1, label='Stage3 opt vs Stage0')
ax.axhline(y=pos_err_s0_mano.mean()*100, color='b', linestyle=':', alpha=0.5)
ax.axhline(y=pos_err_opt_mano.mean()*100, color='m', linestyle=':', alpha=0.5)
ax.set_xlabel('Frame')
ax.set_ylabel('Position Error (cm)')
ax.set_title('Position Tracking Error vs MANO')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 子图2: Z 轴轨迹对比 ---
ax = axes[0, 1]
ax.plot(range(n_align), s0_aligned[:, 2], 'b-', linewidth=1.5, label='Stage0')
ax.plot(range(n_align), mano_aligned[:, 2], 'k--', alpha=0.5, linewidth=1, label='Original MANO')
ax.plot(range(n_align), opt_aligned[:, 2], 'm-', linewidth=1.5, label='Stage3 opt')
ax.scatter([0], [s1_pos[2]], c='green', s=100, marker='*', zorder=5, label=f'Stage1 (z={s1_pos[2]:.3f})')
ax.plot(s2_frames, s2_pos[:, 2], 'r-', linewidth=2, label='Stage2 (25 frames)')
ax.set_xlabel('Frame')
ax.set_ylabel('Z (m)')
ax.set_title('Gripper Z Trajectory')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 子图3: 平滑前后对比 ---
ax = axes[1, 0]
ax.plot(range(n_align), s0_aligned[:, 2], 'b-', alpha=0.5, linewidth=1, label='Stage0 (raw)')
ax.plot(range(len(s0_smoothed)), s0_smoothed[:, 2], 'b-', linewidth=2, label='Stage0 (smoothed, w=5)')
ax.plot(range(n_align), mano_aligned[:, 2], 'k--', alpha=0.5, linewidth=1, label='MANO (raw)')
ax.plot(range(len(mano_smoothed)), mano_smoothed[:, 2], 'k-', linewidth=2, label='MANO (smoothed, w=5)')
ax.set_xlabel('Frame')
ax.set_ylabel('Z (m)')
ax.set_title('Smoothing Effect (Z-axis, window=5)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 子图4: 速度对比 ---
ax = axes[1, 1]
ax.plot(range(len(vel_s0)), np.linalg.norm(vel_s0, axis=1), 'b-', linewidth=1.5, label='Stage0')
ax.plot(range(len(vel_mano)), np.linalg.norm(vel_mano, axis=1), 'k--', alpha=0.5, linewidth=1, label='MANO')
ax.plot(range(len(vel_opt)), np.linalg.norm(vel_opt, axis=1), 'm-', linewidth=1.5, label='Stage3 opt')
ax.set_xlabel('Frame')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('Gripper Velocity')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 子图5: 加速度对比 ---
ax = axes[2, 0]
ax.plot(range(len(acc_s0)), np.linalg.norm(acc_s0, axis=1), 'b-', linewidth=1.5, label='Stage0')
ax.plot(range(len(acc_mano)), np.linalg.norm(acc_mano, axis=1), 'k--', alpha=0.5, linewidth=1, label='MANO')
ax.plot(range(len(acc_opt)), np.linalg.norm(acc_opt, axis=1), 'm-', linewidth=1.5, label='Stage3 opt')
ax.set_xlabel('Frame')
ax.set_ylabel('Acceleration (m/s²)')
ax.set_title('Gripper Acceleration')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- 子图6: 3D 轨迹对比 (含平滑) ---
ax = fig.add_subplot(3, 2, 6, projection='3d')
ax.plot(mano_aligned[:, 0], mano_aligned[:, 1], mano_aligned[:, 2], 'k--', alpha=0.3, linewidth=1, label='MANO (raw)')
ax.plot(s0_aligned[:, 0], s0_aligned[:, 1], s0_aligned[:, 2], 'b-', alpha=0.5, linewidth=1, label='Stage0 (raw)')
ax.plot(s0_smoothed[:, 0], s0_smoothed[:, 1], s0_smoothed[:, 2], 'b-', linewidth=2, label='Stage0 (smoothed)')
ax.plot(opt_aligned[:, 0], opt_aligned[:, 1], opt_aligned[:, 2], 'm-', linewidth=2, label='Stage3 opt')
ax.scatter(*s1_pos, c='green', s=100, marker='*', zorder=5, label='Stage1')
ax.plot(s2_pos[:, 0], s2_pos[:, 1], s2_pos[:, 2], 'r-', linewidth=2, label='Stage2')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory (with smoothing)')
ax.legend(fontsize=7, loc='upper left')
ax.view_init(elev=25, azim=45)

plt.suptitle('Gripper Right - Stage0 Error Analysis & Smoothing (GLB)', fontsize=14, fontweight='bold')
plt.tight_layout()

save_path = f'{OUTPUT_DIR}/stage0_error_analysis.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {save_path}")
plt.close()

# ============================================================
# 7. 保存平滑后的数据
# ============================================================
np.savez(f'{OUTPUT_DIR}/stage0/stage0_smoothed.npz',
         pos_raw=s0_aligned,
         pos_smoothed=s0_smoothed,
         mano_raw=mano_aligned,
         mano_smoothed=mano_smoothed,
         pos_err_s0_mano=pos_err_s0_mano,
         pos_err_smooth=pos_err_smooth,
         allow_pickle=True)
print(f"Saved smoothed data: {OUTPUT_DIR}/stage0/stage0_smoothed.npz")

# ============================================================
# 8. Stage2 帧数说明
# ============================================================
print("\n" + "="*70)
print("STAGE2 帧数说明")
print("="*70)
print(f"  reconstruct_trajectory 生成: {len(s2_pos)} 帧 (frame {s2_frames[0]}-{s2_frames[-1]})")
print(f"  rollout_follow_trajectory 录制: 120 帧 (完整仿真)")
print(f"  重建轨迹只覆盖 frame 38-62 (25帧), 但视频录制了完整 120 帧仿真")
print(f"  视频包含: 接近 -> 抓取 -> 抬升 -> 携带 全过程")