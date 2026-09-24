#!/usr/bin/env python3
"""
验证 7a 场景的 Stage 3 (large_plane) 能否从 3D 点云找到 floor 并产生坐标轴变化。

不依赖原始视频，直接用已有的 depth + extrinsics + intrinsic 重建 world_points，
然后调用改过的 align_via_large_plane 函数（不传 floor_masks，让它从点云几何找平面）。

注意：align_via_large_plane 的逻辑是"如果有 floor_masks 就用 mask 裁剪点云"，
但 7a 没有 SAM3 floor mask，所以我们需要直接测试 _build_R_t_from_floor + PCA 的逻辑。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry_utils import (
    get_plane_info,
    _orient_floor_normal,
    _build_R_t_from_floor,
)


SCENE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_v2/7a_vggt_omega")


def load_extrinsics():
    ext_dir = os.path.join(SCENE_DIR, "extrinsics")
    frames = sorted([f for f in os.listdir(ext_dir) if f.endswith('.txt')])
    extrinsics = []
    for fname in frames:
        arr = np.loadtxt(os.path.join(ext_dir, fname)).reshape(4, 4)
        extrinsics.append(arr[:3, :4])
    return np.array(extrinsics)


def load_intrinsics():
    return np.loadtxt(os.path.join(SCENE_DIR, "intrinsic.txt")).reshape(3, 3)


def reconstruct_world_points(extrinsics, intrinsics, num_frames=20):
    """从 depth + extrinsics + intrinsic 重建 world_points"""
    H, W = 384, 688
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    all_world_points = []
    for i in range(min(num_frames, len(extrinsics))):
        depth_path = os.path.join(SCENE_DIR, "depth", f"{i}.png")
        if not os.path.exists(depth_path):
            continue

        depth_img = __import__('cv2').imread(depth_path, __import__('cv2').IMREAD_UNCHANGED)
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]
        depth = (depth_img.astype(np.float32) / 1000.0)

        # 过滤无效深度
        valid = (depth > 0.01) & (depth < 10.0)

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x_cam = (u - cx) / fx * depth
        y_cam = (v - cy) / fy * depth
        z_cam = depth

        cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)

        R_w2c = extrinsics[i, :3, :3]
        t_w2c = extrinsics[i, :3, 3]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        cam_flat = cam_points.reshape(-1, 3)
        world_flat = (R_c2w @ cam_flat.T).T + t_c2w
        all_world_points.append(world_flat.reshape(H, W, 3))

    return np.array(all_world_points)


def main():
    print("=" * 70)
    print("7a 场景 Stage 3 验证：从 3D 点云找 floor 平面")
    print("=" * 70)

    # 加载数据
    extrinsics = load_extrinsics()
    intrinsics = load_intrinsics()
    print(f"\n  Extrinsics: {extrinsics.shape}")
    print(f"  Intrinsics: {intrinsics.shape}")

    # 重建 world_points
    print("\n  重建 world_points...")
    world_points = reconstruct_world_points(extrinsics, intrinsics, num_frames=20)
    print(f"  world_points: {world_points.shape}")

    # 统计有效点数
    valid_count = np.sum(world_points[:, :, :, 2] > 0.01)
    total_count = world_points.size // 3
    print(f"  有效点: {valid_count}/{total_count} ({valid_count/total_count:.1%})")

    # 将所有帧的点拼在一起做 PCA
    all_points = world_points.reshape(-1, 3)
    print(f"\n  总点数: {len(all_points)}")
    print(f"  点云范围:")
    print(f"    x: [{all_points[:, 0].min():.3f}, {all_points[:, 0].max():.3f}]")
    print(f"    y: [{all_points[:, 1].min():.3f}, {all_points[:, 1].max():.3f}]")
    print(f"    z: [{all_points[:, 2].min():.3f}, {all_points[:, 2].max():.3f}]")

    # PCA 找最大平面
    centroid = np.mean(all_points, axis=0)
    centered = all_points - centroid
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    print(f"\n  PCA 特征值:")
    print(f"    λ1 = {eigenvalues[0]:.1f} (最大方向)")
    print(f"    λ2 = {eigenvalues[1]:.1f}")
    print(f"    λ3 = {eigenvalues[2]:.1f} (最小方向 = 法线)")

    # 最小特征值方向 = 平面法线
    min_idx = np.argmin(eigenvalues)
    floor_normal = eigenvectors[:, min_idx]
    floor_normal = floor_normal / np.linalg.norm(floor_normal)

    # 确保法线指向上方
    floor_normal = _orient_floor_normal(floor_normal, centroid, all_points)

    print(f"\n  Floor normal: {floor_normal}")
    angle_from_z = np.degrees(np.arccos(np.clip(abs(floor_normal[2]), -1, 1)))
    print(f"  与 z 轴夹角: {angle_from_z:.1f}°")

    # 估算 floor centroid (用 z 方向的中位数)
    floor_z = np.median(all_points[:, 2])
    floor_centroid = np.array([centroid[0], centroid[1], floor_z])
    print(f"  Floor centroid: {floor_centroid}")

    # 构建对齐变换
    R, t = _build_R_t_from_floor(world_points, floor_normal, floor_centroid)

    print(f"\n  对齐矩阵 R:")
    print(f"    {R[0]}")
    print(f"    {R[1]}")
    print(f"    {R[2]}")
    print(f"  平移向量 t: {t}")

    # 关键验证
    r_diff = np.linalg.norm(R - np.eye(3))
    print(f"\n  R 与单位矩阵的差异 (Frobenius norm): {r_diff:.4f}")
    print(f"  R[2,2] = {R[2,2]:.4f}")
    print(f"  det(R) = {np.linalg.det(R):.6f}")

    # 验证 R 是合法正交矩阵
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "R 必须是正交矩阵"
    assert abs(np.linalg.det(R) - 1.0) < 1e-5, "det(R) 应 ≈ 1"

    # 判断是否通过
    passes = R[2, 2] > 0.3  # 新阈值
    has_rotation = r_diff > 0.1

    print(f"\n  {'✅' if passes else '❌'} R[2,2] = {R[2,2]:.4f} {'>' if passes else '<'} 0.3 (新阈值)")
    print(f"  {'✅' if has_rotation else '❌'} 坐标轴有显著旋转 (R ≠ I, diff={r_diff:.4f})")

    # 应用变换到 extrinsics，验证相机姿态变化
    n_frames = len(extrinsics)
    R_applied = np.array([R @ extrinsics[i, :3, :3] for i in range(n_frames)])
    t_applied = np.array([R @ extrinsics[i, :3, 3] + t for i in range(n_frames)])

    orig_positions = np.array([-extrinsics[i, :3, :3].T @ extrinsics[i, :3, 3] for i in range(n_frames)])
    new_positions = np.array([-R_applied[i].T @ t_applied[i] for i in range(n_frames)])

    print(f"\n  相机位置变化:")
    print(f"    变换前 z 范围: [{orig_positions[:, 2].min():.3f}, {orig_positions[:, 2].max():.3f}]")
    print(f"    变换后 z 范围: [{new_positions[:, 2].min():.3f}, {new_positions[:, 2].max():.3f}]")
    print(f"    z 轴翻转: {abs(new_positions[:, 2].mean() - orig_positions[:, 2].mean()):.3f}")

    # 总结
    print("\n" + "=" * 70)
    if passes and has_rotation:
        print("✅ 7a 场景能通过 Stage 3 产生有效的坐标轴变化!")
        print("   - R[2,2] > 0.3 阈值")
        print("   - 坐标轴发生显著旋转")
        print("   - 变换后 z 轴朝上")
    elif passes:
        print("⚠️  R[2,2] > 0.3 但旋转幅度较小")
        print("   - 场景本身可能已经接近正确坐标系")
    else:
        print("❌ 7a 场景无法通过 Stage 3 对齐")
        print("   - R[2,2] < 0.3 阈值")
        print("   - 点云中没有明显的水平面")
    print("=" * 70)


if __name__ == "__main__":
    main()
