"""
将 Replica 数据集转换为 Stage 4 可用的格式

Replica 格式:
    room0/
    ├── results/
    │   ├── frame000000.jpg   (RGB)
    │   └── depth000000.png   (uint16, scale=6553.5 → meters)
    ├── traj.txt              (c2w, OpenGL, y-up, 每行一个4x4矩阵)
    └── ..
    cam_params.json           (fx, fy, cx, cy, scale)
    room0_mesh.ply            (y-up, 单个mesh)

Stage 4 格式:
    <output>/
    ├── final_scene_basic.glb       (y-up, 多个geometry)
    ├── intrinsic.txt         (3x3)
    ├── color/                (0.jpg, 1.jpg, ...)
    ├── depth/                (0.png, uint16 mm)
    └── extrinsics/           (0.txt, 4x4, w2c, OpenCV, z-up)

坐标系转换:
    Replica: c2w OpenGL (y-up, x-right, y-up, z-backward)
    Stage4:  w2c OpenCV (z-up, x-right, y-down, z-forward)

    c2w_opengl → w2c_opencv:  inv(c2w)
    w2c_opengl → w2c_opencv_zup:
        flip = [[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]  # OpenGL→OpenCV flip
        y2z = [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]    # y-up→z-up
        w2c_opencv_zup = y2z @ w2c_opengl

Usage:
    python scripts/prepare_replica_for_stage4.py \
        --replica_dir /mnt/data_8THDD/lza/dataset/Replica \
        --scene room0 \
        --output_dir /tmp/replica_room0_stage4 \
        --max_frames 100
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import cv2
import trimesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_replica_trajectory(traj_path):
    traj = []
    with open(traj_path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) == 16:
                traj.append(np.array(vals).reshape(4, 4))
    return np.array(traj)


def c2w_opengl_to_w2c_opencv_zup(c2w_opengl):
    """Convert Replica c2w (OpenGL, y-up) to w2c (OpenCV, z-up).

    Replica depth was rendered with camera looking along +z (Habitat convention),
    so we need to flip the camera z-axis before converting.
    """
    y2z = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    gl2cv = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    flip_z = np.diag([1.0, 1.0, -1.0, 1.0])
    c2w_corrected = c2w_opengl @ flip_z
    w2c_gl = np.linalg.inv(c2w_corrected)
    w2c_cv = gl2cv @ w2c_gl
    w2c_cv_zup = y2z @ w2c_cv @ np.linalg.inv(y2z)
    return w2c_cv_zup


def split_mesh_into_objects(mesh_ply_path, output_dir):
    mesh = trimesh.load(mesh_ply_path)
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, 'geometry'):
            geometries = list(mesh.geometry.values())
        else:
            geometries = [mesh]
    else:
        geometries = [mesh]

    if len(geometries) == 1:
        print("  单个mesh，按连通分量拆分...")
        big_mesh = geometries[0]
        split = big_mesh.split(only_watertight=False)
        if len(split) <= 1:
            print("  无法拆分连通分量，按体素聚类拆分...")
            split = _split_by_spatial_clustering(big_mesh)

        print(f"  拆分为 {len(split)} 个物体")
        geometries = split

    scene = trimesh.Scene()
    for i, geom in enumerate(geometries):
        if len(geom.vertices) < 10 or len(geom.faces) < 10:
            continue
        geom_name = f"object_{i:03d}"
        scene.add_geometry(geom, node_name=geom_name, geom_name=geom_name)

    glb_path = os.path.join(output_dir, "final_scene_basic.glb")
    scene.export(glb_path)
    print(f"  GLB已保存: {glb_path} ({len(scene.geometry)} 个物体)")
    return glb_path


def _split_by_spatial_clustering(mesh, grid_size=1.0):
    from scipy.ndimage import label as ndlabel

    grid = np.round(mesh.vertices / grid_size).astype(int)
    occupied = set(map(tuple, grid))
    if not occupied:
        return [mesh]

    min_coords = np.min(list(occupied), axis=0)
    shifted = {tuple(np.array(p) - min_coords) for p in occupied}
    max_coords = np.max(list(shifted), axis=0) + 1

    volume = np.zeros(max_coords, dtype=bool)
    for p in shifted:
        volume[p] = True

    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num_features = ndlabel(volume, structure=structure)

    vertex_labels = np.full(len(mesh.vertices), -1, dtype=int)
    for i, v in enumerate(grid):
        shifted_v = tuple(v - min_coords)
        if shifted_v in shifted:
            vertex_labels[i] = labeled[shifted_v]

    valid_labels = set(vertex_labels[vertex_labels >= 0])
    if not valid_labels:
        return [mesh]

    label_sizes = {l: (vertex_labels == l).sum() for l in valid_labels}
    sorted_labels = sorted(label_sizes.keys(), key=lambda l: label_sizes[l], reverse=True)

    result = []
    for lbl in sorted_labels[:50]:
        mask = vertex_labels == lbl
        if mask.sum() < 100:
            continue
        submesh = mesh.vertex_mask(mask)
        if len(submesh.vertices) > 50 and len(submesh.faces) > 10:
            result.append(submesh)

    if not result:
        result = [mesh]

    return result


def prepare_replica(replica_dir, scene_name, output_dir, max_frames=100):
    scene_dir = os.path.join(replica_dir, scene_name)
    results_dir = os.path.join(scene_dir, "results")
    traj_path = os.path.join(scene_dir, "traj.txt")
    cam_params_path = os.path.join(replica_dir, "cam_params.json")
    mesh_path = os.path.join(replica_dir, f"{scene_name}_mesh.ply")

    for p in [scene_dir, results_dir, traj_path, cam_params_path, mesh_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}")

    with open(cam_params_path) as f:
        cam_params = json.load(f)["camera"]

    fx, fy = cam_params["fx"], cam_params["fy"]
    cx, cy = cam_params["cx"], cam_params["cy"]
    depth_scale = cam_params["scale"]

    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    os.makedirs(output_dir, exist_ok=True)
    color_dir = os.path.join(output_dir, "color")
    depth_dir = os.path.join(output_dir, "depth")
    ext_dir = os.path.join(output_dir, "extrinsics")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(ext_dir, exist_ok=True)

    frame_files = sorted([f for f in os.listdir(results_dir) if f.startswith("frame") and f.endswith(".jpg")])
    total_frames = len(frame_files)
    print(f"Replica {scene_name}: {total_frames} 帧, 分辨率 {cam_params['w']}x{cam_params['h']}")

    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        print(f"采样 {max_frames} 帧 (从 {total_frames} 帧中等间隔采样)")
    else:
        indices = np.arange(total_frames)
        max_frames = total_frames

    traj = load_replica_trajectory(traj_path)
    print(f"轨迹: {len(traj)} 帧")

    for new_idx, orig_idx in enumerate(indices):
        frame_src = os.path.join(results_dir, f"frame{orig_idx:06d}.jpg")
        frame_dst = os.path.join(color_dir, f"{new_idx}.jpg")
        shutil.copy2(frame_src, frame_dst)

        depth_src = os.path.join(results_dir, f"depth{orig_idx:06d}.png")
        depth_raw = cv2.imread(depth_src, cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) / depth_scale
        depth_mm = (depth_m * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_dir, f"{new_idx}.png"), depth_mm)

        if orig_idx < len(traj):
            w2c = c2w_opengl_to_w2c_opencv_zup(traj[orig_idx])
        else:
            w2c = c2w_opengl_to_w2c_opencv_zup(traj[-1])
        np.savetxt(os.path.join(ext_dir, f"{new_idx}.txt"), w2c, fmt='%.15e')

    np.savetxt(os.path.join(output_dir, "intrinsic.txt"), intrinsic, fmt='%.6f')

    print("拆分mesh为物体...")
    split_mesh_into_objects(mesh_path, output_dir)

    print(f"\n✅ 转换完成!")
    print(f"输出目录: {output_dir}")
    print(f"帧数: {max_frames}")
    print(f"内参: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"\n运行 Stage 4:")
    print(f"  cd {REPO_ROOT}")
    print(f"  conda run -n ReplicateAnyScene python stage4/run_alignment.py \\")
    print(f"    --input_path {output_dir} \\")
    print(f"    --output_dir {output_dir}/aligned \\")
    print(f"    --num_iterations 8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Replica dataset for Stage 4 testing")
    parser.add_argument("--replica_dir", type=str,
                        default="/mnt/data_8THDD/lza/dataset/Replica",
                        help="Replica数据集根目录")
    parser.add_argument("--scene", type=str, default="room0",
                        help="场景名称 (room0, office0, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: /tmp/replica_{scene}_stage4)")
    parser.add_argument("--max_frames", type=int, default=100,
                        help="最大帧数 (默认100, 从2000帧中等间隔采样)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/tmp/replica_{args.scene}_stage4"

    prepare_replica(args.replica_dir, args.scene, args.output_dir, args.max_frames)
