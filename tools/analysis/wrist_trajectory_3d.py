"""
将 HaWoR 的手腕 mask 投影到 VGGT-Omega 的 3D 世界坐标系, 输出手腕轨迹 GLB.

流程:
  1. 加载 HaWoR model_masks.npy (N_hawor, H_hawor, W_hawor) bool
  2. 加载 VGGT-Omega depth/extrinsics/intrinsic
  3. 帧对齐: HaWoR 帧数 N_hawor → VGGT-Omega 帧数 N_vggt (均匀采样)
  4. 将 mask resize 到 depth 分辨率
  5. 对每帧: mask 像素 + depth + intrinsic + extrinsic → 3D 世界坐标
  6. 取质心 → 手腕轨迹
  7. 输出 GLB (轨迹管道 + 质心球)
"""

import numpy as np
import cv2
import trimesh
import os
import argparse


def load_hawor_masks(hawor_dir):
    """加载 HaWoR model_masks.npy"""
    mask_path = os.path.join(hawor_dir, 'model_masks.npy')
    masks = np.load(mask_path)
    print(f"HaWoR masks: shape={masks.shape}, dtype={masks.dtype}")
    return masks


def align_frames(n_hawor, n_vggt):
    """
    计算 HaWoR 帧索引 → VGGT-Omega 帧索引的映射.
    HaWoR 每帧都有, VGGT-Omega 可能是均匀采样.
    返回: vggt_indices[i] = VGGT-Omega 帧索引对应 HaWoR 的第 i 帧
    """
    if n_hawor == n_vggt:
        return np.arange(n_vggt)
    # 少的那个是均匀采样的: VGGT-Omega 的第 j 帧对应 HaWoR 的第 floor(j * n_hawor / n_vggt) 帧
    # 反过来: HaWoR 的第 i 帧对应 VGGT-Omega 的第 round(i * (n_vggt-1) / (n_hawor-1)) 帧
    hawor_to_vggt = np.round(np.arange(n_hawor) * (n_vggt - 1) / (n_hawor - 1)).astype(int)
    return hawor_to_vggt


def resize_mask_to_depth(mask, depth_h, depth_w):
    """将 mask resize 到 depth 分辨率 (nearest-neighbor, 保持 bool)"""
    return cv2.resize(mask.astype(np.uint8), (depth_w, depth_h),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def unproject_mask_to_world(mask, depth, R_w2c, t_w2c, fx, fy, cx, cy):
    """
    将 mask 区域的像素用 depth + intrinsic + extrinsic 反投影到世界坐标.
    extrinsic 约定: w2c, p_cam = R @ p_world + t → p_world = R^T @ (p_cam - t)

    Args:
        mask: (H, W) bool
        depth: (H, W) float, meters
        R_w2c: (3, 3)
        t_w2c: (3,)
        fx, fy, cx, cy: intrinsic params

    Returns:
        points_world: (N, 3) 世界坐标 (N = mask 中有效像素数)
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 3))

    d = depth[ys, xs]
    valid = d > 1e-6
    if not valid.any():
        return np.zeros((0, 3))

    ys, xs, d = ys[valid], xs[valid], d[valid]

    # 针孔模型反投影: cam coords
    x_cam = (xs - cx) * d / fx
    y_cam = (ys - cy) * d / fy
    z_cam = d
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # (N, 3)

    # w2c 逆变换: p_world = R^T @ (p_cam - t)
    pts_world = (R_w2c.T @ (pts_cam - t_w2c).T).T  # (N, 3)

    return pts_world


def build_trajectory_glb(trajectory):
    """
    将轨迹点列表构建为简单 GLB 点云.

    Args:
        trajectory: list of (3,) numpy arrays, 每帧的手腕质心

    Returns:
        trimesh.PointCloud
    """
    valid_pts = [pt for pt in trajectory if np.all(np.isfinite(pt))]
    if not valid_pts:
        return trimesh.PointCloud(vertices=np.zeros((0, 3)))
    return trimesh.PointCloud(vertices=np.array(valid_pts))


def main():
    parser = argparse.ArgumentParser(description="HaWoR wrist mask → 3D trajectory via VGGT-Omega depth")
    parser.add_argument("--hawor_dir", type=str,
                        default="/mnt/data_8THDD/lza/workspace/robot_world_ws/src/Ego-Video-to-SIM/HaWoR/output/7/tracks_0_113",
                        help="HaWoR tracks 输出目录 (含 model_masks.npy)")
    parser.add_argument("--vggt_dir", type=str,
                        default="/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega",
                        help="VGGT-Omega 输出目录 (含 depth/, extrinsics/, intrinsic.txt)")
    parser.add_argument("--output", type=str, default=None, help="输出 GLB 路径 (默认: vggt_dir/wrist_trajectory.glb)")
    parser.add_argument("--conf_percentile", type=float, default=10.0,
                        help="深度置信度过滤: 去掉最浅 conf_percentile% 的点 (默认 10)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.vggt_dir, "wrist_trajectory.glb")

    # 1. 加载 HaWoR mask
    masks_hawor = load_hawor_masks(args.hawor_dir)  # (N_hawor, H_hawor, W_hawor)
    n_hawor = masks_hawor.shape[0]

    # 2. 加载 VGGT-Omega 数据
    intrinsic = np.loadtxt(os.path.join(args.vggt_dir, 'intrinsic.txt'))
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    print(f"Intrinsic: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    depth_dir = os.path.join(args.vggt_dir, 'depth')
    ext_dir = os.path.join(args.vggt_dir, 'extrinsics')
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')],
                         key=lambda x: int(x.replace('.png', '')))
    n_vggt = len(depth_files)
    print(f"VGGT-Omega frames: {n_vggt}, HaWoR frames: {n_hawor}")

    # 3. 帧对齐
    hawor_to_vggt = align_frames(n_hawor, n_vggt)
    print(f"Frame alignment: HaWoR[0]→VGGT[{hawor_to_vggt[0]}], HaWoR[-1]→VGGT[{hawor_to_vggt[-1]}]")

    # 4. 逐帧投影
    trajectory = []
    for vggt_idx in range(n_vggt):
        # 找到映射到这个 vggt_idx 的所有 hawor 帧 (取最近的)
        hawor_frames = np.where(hawor_to_vggt == vggt_idx)[0]
        if len(hawor_frames) == 0:
            trajectory.append(np.array([np.nan, np.nan, np.nan]))
            continue

        # 加载 depth 和 extrinsic
        depth_raw = cv2.imread(os.path.join(depth_dir, f'{vggt_idx}.png'), cv2.IMREAD_UNCHANGED)
        depth = depth_raw.astype(np.float32) / 1000.0  # mm → meters
        depth_h, depth_w = depth.shape

        ext = np.loadtxt(os.path.join(ext_dir, f'{vggt_idx}.txt'))
        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]

        # 取所有映射帧的 mask 并集 (多个 hawor 帧可能映射到同一 vggt 帧)
        combined_mask = np.zeros((depth_h, depth_w), dtype=bool)
        for hf in hawor_frames:
            mask_resized = resize_mask_to_depth(masks_hawor[hf], depth_h, depth_w)
            combined_mask |= mask_resized

        # 反投影
        pts_world = unproject_mask_to_world(combined_mask, depth, R_w2c, t_w2c, fx, fy, cx, cy)

        if len(pts_world) == 0:
            trajectory.append(np.array([np.nan, np.nan, np.nan]))
            continue

        # 过滤异常点: 去掉深度最近/最远的 conf_percentile%
        depths_mask = depth[combined_mask]
        depths_mask = depths_mask[depths_mask > 1e-6]
        if len(depths_mask) > 10:
            lo = np.percentile(depths_mask, args.conf_percentile)
            hi = np.percentile(depths_mask, 100 - args.conf_percentile)
            pt_depths = np.linalg.norm(pts_world - (-R_w2c.T @ t_w2c), axis=1)  # 到相机距离
            valid = (pt_depths >= lo) & (pt_depths <= hi)
            pts_world = pts_world[valid]

        if len(pts_world) == 0:
            trajectory.append(np.array([np.nan, np.nan, np.nan]))
            continue

        # 质心
        centroid = np.median(pts_world, axis=0)
        trajectory.append(centroid)

        if vggt_idx % 20 == 0:
            n_mask_pixels = combined_mask.sum()
            print(f"  Frame {vggt_idx}: mask_pixels={n_mask_pixels}, 3D_points={len(pts_world)}, "
                  f"centroid=({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")

    # 5. 统计
    valid_traj = [p for p in trajectory if np.all(np.isfinite(p))]
    print(f"\n有效帧: {len(valid_traj)} / {len(trajectory)}")
    if valid_traj:
        arr = np.array(valid_traj)
        print(f"轨迹范围: X=[{arr[:,0].min():.3f}, {arr[:,0].max():.3f}], "
              f"Y=[{arr[:,1].min():.3f}, {arr[:,1].max():.3f}], "
              f"Z=[{arr[:,2].min():.3f}, {arr[:,2].max():.3f}]")
        total_length = sum(np.linalg.norm(np.array(valid_traj[i+1]) - np.array(valid_traj[i]))
                          for i in range(len(valid_traj)-1))
        print(f"轨迹总长度: {total_length:.3f}m")

    # 6. 保存 GLB
    scene = build_trajectory_glb(trajectory)
    scene.export(output_path)
    print(f"\nGLB 已保存: {output_path}")

    # 同时保存 numpy 格式的轨迹
    npy_path = output_path.replace('.glb', '.npy')
    np.save(npy_path, np.array(trajectory))
    print(f"NumPy 轨迹已保存: {npy_path}")


if __name__ == '__main__':
    main()
