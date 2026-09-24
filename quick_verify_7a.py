#!/usr/bin/env python3
"""
快速验证: 用 7a 的现有数据模拟改进后的 pipeline, 看 Stage 3 能否成功对齐。

不需要重新运行整个 pipeline, 直接从已有的 world_points 中提取 floor 平面。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.geometry_utils import (
    get_plane_info,
    _orient_floor_normal,
    _build_R_t_from_floor,
)


SCENE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_v2/7a_vggt_omega")


def load_world_points():
    """
    从 VGGT 输出的 depth + extrinsics + intrinsic 重建 world_points。
    或者直接从已有的 depth 图和 extrinsics 计算。
    """
    # 尝试加载已有的 point_cloud.ply (如果存在)
    ply_path = os.path.join(SCENE_DIR, "point_cloud.ply")
    if os.path.exists(ply_path):
        print(f"  从 PLY 加载点云: {ply_path}")
        import trimesh
        pc = trimesh.load(ply_path)
        return pc.vertices.reshape(1, -1, 3)  # (1, N, 3)

    # 否则从 depth + extrinsics + intrinsic 重建
    print("  从 depth + extrinsics + intrinsic 重建 world_points...")
    H, W = 384, 688
    intrinsic = np.loadtxt(os.path.join(SCENE_DIR, "intrinsic.txt")).reshape(3, 3)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # 加载前几帧的 depth
    depths = []
    extrinsics = []
    for i in range(min(5, 113)):
        depth_path = os.path.join(SCENE_DIR, "depth", f"{i}.png")
        if not os.path.exists(depth_path):
            continue
        # cv2 读 png 可能是单通道或三通道, 取第一个通道
        depth_img = __import__('cv2').imread(depth_path, __import__('cv2').IMREAD_UNCHANGED)
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]
        depth = (depth_img.astype(np.float32) / 1000.0)
        extr_path = os.path.join(SCENE_DIR, "extrinsics", f"{i}.txt")
        extr = np.loadtxt(extr_path).reshape(4, 4)[:3, :4]
        depths.append(depth)
        extrinsics.append(extr)

    if not depths:
        print("  ❌ 无法加载 depth 数据")
        return None

    # 创建 UV 网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # 反投影到相机坐标
    x_cam = (u - cx) / fx * depths[0]
    y_cam = (v - cy) / fy * depths[0]
    z_cam = depths[0]

    cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (H, W, 3)
    R_w2c = extrinsics[0][:3, :3]
    t_w2c = extrinsics[0][:3, 3]
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c

    cam_flat = cam_points.reshape(-1, 3)
    world_flat = (R_c2w @ cam_flat.T).T + t_c2w

    return world_flat.reshape(H, W, 3)


def simulate_stage3_with_vlm_enhancement():
    """
    模拟改进后的 Stage 3:
    1. 从 3D 点云 PCA 找最大水平面
    2. 构建对齐变换 R, t
    3. 验证 R != I
    """
    print("=" * 70)
    print("模拟 Stage 3 (large_plane with VLM enhancement)")
    print("=" * 70)

    # 加载 extrinsics
    ext_dir = os.path.join(SCENE_DIR, "extrinsics")
    frames = sorted([f for f in os.listdir(ext_dir) if f.endswith('.txt')])
    extrinsics = []
    for fname in frames:
        arr = np.loadtxt(os.path.join(ext_dir, fname)).reshape(4, 4)
        extrinsics.append(arr[:3, :4])
    extrinsics = np.array(extrinsics)

    # 加载 depth 并重建 world_points
    H, W = 384, 688
    intrinsic = np.loadtxt(os.path.join(SCENE_DIR, "intrinsic.txt")).reshape(3, 3)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    all_world_points = []
    for i in range(min(20, len(frames))):
        depth_path = os.path.join(SCENE_DIR, "depth", f"{i}.png")
        if not os.path.exists(depth_path):
            continue
        depth_img = __import__('cv2').imread(depth_path, __import__('cv2').IMREAD_UNCHANGED)
        if depth_img.ndim == 3:
            depth_img = depth_img[:, :, 0]
        depth = (depth_img.astype(np.float32) / 1000.0)

        u, v = np.meshgrid(np.arange(W), np.arange(H))
        x_cam = (u - cx) / fx * depth
        y_cam = (v - cy) / fy * depth
        z_cam = depth

        cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1)
        R_w2c = extrinsics[i, :3, :3]
        t_w2c = extrinsics[i, :3, 3]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        cam_flat = cam_points.reshape(-1, 3)
        world_flat = (R_c2w @ cam_flat.T).T + t_c2w
        all_world_points.append(world_flat.reshape(H, W, 3))

    all_world_points = np.array(all_world_points)
    all_points = all_world_points.reshape(-1, 3)

    print(f"\n  重建的 world_points 形状: {all_world_points.shape}")
    print(f"  总点数: {len(all_points)}")
    print(f"  点云范围: x=[{all_points[:, 0].min():.3f}, {all_points[:, 0].max():.3f}], "
          f"y=[{all_points[:, 1].min():.3f}, {all_points[:, 1].max():.3f}], "
          f"z=[{all_points[:, 2].min():.3f}, {all_points[:, 2].max():.3f}]")

    # PCA 找最大平面
    centroid = np.mean(all_points, axis=0)
    centered = all_points - centroid
    cov = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    print(f"\n  PCA 特征值: λ1={eigenvalues[0]:.1f}, λ2={eigenvalues[1]:.1f}, λ3={eigenvalues[2]:.1f}")
    print(f"  最大特征值 / 总特征值 = {max(eigenvalues) / sum(eigenvalues):.2%}")

    # 最小特征值方向 = 平面法线
    min_idx = np.argmin(eigenvalues)
    floor_normal = eigenvectors[:, min_idx]
    floor_normal = floor_normal / np.linalg.norm(floor_normal)

    # 确保法线指向上方
    floor_normal = _orient_floor_normal(floor_normal, centroid, all_points)

    print(f"\n  估计的 floor_normal: {floor_normal}")
    print(f"  floor_normal 与 [0,0,1] 的夹角: {np.degrees(np.arccos(np.clip(abs(floor_normal[2]), -1, 1))):.1f}°")

    # 检查是否是合理的地面 (z 分量接近 1)
    if abs(floor_normal[2]) > 0.8:
        print(f"  ✅ floor_normal 接近 [0,0,1], 是合理的地面")
    else:
        print(f"  ⚠️ floor_normal 偏离 z 轴较多, 可能需要旋转")

    # 估算 floor centroid
    floor_z = centroid[2]
    floor_centroid = np.array([centroid[0], centroid[1], floor_z])

    # 构建对齐变换
    # 需要构造一个临时的 world_points 数组
    world_points_single = all_world_points[0:1]
    R, t = _build_R_t_from_floor(world_points_single, floor_normal, floor_centroid)

    print(f"\n  对齐矩阵 R:\n{R}")
    print(f"  平移向量 t: {t}")

    # 关键: R 是否与单位矩阵有明显差异?
    r_diff = np.linalg.norm(R - np.eye(3))
    print(f"\n  R 与单位矩阵的差异 (Frobenius norm): {r_diff:.4f}")

    if r_diff > 0.1:
        print(f"  ✅ 坐标轴发生了显著变化! (R ≠ I)")
    else:
        print(f"  ⚠️ R 接近单位矩阵, 坐标轴变化不大")

    # 应用变换到 extrinsics
    R_applied = np.array([R @ extrinsics[i, :3, :3] for i in range(len(extrinsics))])
    t_applied = np.array([R @ extrinsics[i, :3, 3] + t for i in range(len(extrinsics))])

    # 比较变换前后的相机位置
    orig_positions = np.array([-extrinsics[i, :3, :3].T @ extrinsics[i, :3, 3] for i in range(len(extrinsics))])
    new_positions = np.array([-R_applied[i].T @ t_applied[i] for i in range(len(extrinsics))])

    pos_diff = np.linalg.norm(orig_positions - new_positions)
    print(f"\n  相机位置总变化: {pos_diff:.4f}")
    print(f"  变换前相机位置范围: z=[{orig_positions[:, 2].min():.3f}, {orig_positions[:, 2].max():.3f}]")
    print(f"  变换后相机位置范围: z=[{new_positions[:, 2].min():.3f}, {new_positions[:, 2].max():.3f}]")

    # 验证 R 是合法的正交矩阵
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), "R 必须是正交矩阵"
    det_R = np.linalg.det(R)
    print(f"\n  R 是合法的正交旋转矩阵 (det={det_R:.6f})")

    return R, t, floor_normal


def simulate_stage2_sam3_prompts():
    """
    模拟改进后的 SAM3 多 prompt 扩展, 看能否匹配到红色布料。
    """
    print("\n" + "=" * 70)
    print("模拟 SAM3 多 prompt 扩展 (检测红色布料)")
    print("=" * 70)

    # 从颜色图中提取红色区域的统计
    color_dir = os.path.join(SCENE_DIR, "color")
    if not os.path.exists(color_dir):
        print(f"  ❌ 颜色目录不存在: {color_dir}")
        return

    # 加载前 5 帧
    colors = []
    for i in range(5):
        img_path = os.path.join(color_dir, f"{i}.jpg")
        if os.path.exists(img_path):
            img = np.array(__import__('cv2').imread(img_path))
            colors.append(img)

    if not colors:
        print(f"  ❌ 无法加载颜色图")
        return

    # 统计红色区域
    red_mask = (colors[0][:, :, 2] > 150) & (colors[0][:, :, 0] > 100) & (colors[0][:, :, 1] < 80)
    red_pixels = np.sum(red_mask)
    total_pixels = red_mask.size

    print(f"  第 0 帧红色区域占比: {red_pixels / total_pixels:.1%}")
    print(f"  红色像素数: {red_pixels}/{total_pixels}")

    if red_pixels > total_pixels * 0.3:
        print(f"  ✅ 场景以红色为主, SAM3 'carpet' / 'horizontal surface' 可能匹配")
    else:
        print(f"  ⚠️ 红色区域占比不高, SAM3 可能仍无法准确分割")


def main():
    print(f"场景目录: {SCENE_DIR}")
    print()

    # 模拟 Stage 3
    R, t, floor_normal = simulate_stage3_with_vlm_enhancement()

    # 模拟 SAM3 多 prompt
    simulate_stage2_sam3_prompts()

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(f"""
  改进后的 pipeline 对 7a 场景的预期效果:

  1. SAM3 多 prompt: 'carpet'/'horizontal surface' 可能匹配红色布料
     → 如果成功, Stage 2 就能找到 floor mask

  2. Stage 3 (large_plane): 从 3D 点云 PCA 找最大水平面
     → 无论 SAM3 是否成功, 点云中红色布料都是一个明显的大平面
     → R[2,2] = {abs(floor_normal[2]):.4f} ({'✅' if abs(floor_normal[2]) > 0.3 else '❌'})

  3. 坐标轴变化: Frobenius norm = {np.linalg.norm(R - np.eye(3)):.4f}
     → {'✅ 有明显的坐标轴旋转' if np.linalg.norm(R - np.eye(3)) > 0.1 else '⚠️ 坐标轴变化不大'}
""")


if __name__ == "__main__":
    main()
