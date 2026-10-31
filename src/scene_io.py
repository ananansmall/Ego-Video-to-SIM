"""场景 I/O 工具: 读取管线产物、重建世界点云。

这些原本在旧 Stage 4 的 run_alignment.py 里, 但它们是**与对齐无关的基础工具**
(读 color/depth/extrinsics、按 mask 反投影出世界点云), 多个工具脚本都要用。
Stage 4 已永久废弃并归档到 archive/stage4, 这里保留最小实现, 避免任何模块
再去 import archive.stage4.*。
"""
import os

import cv2
import numpy as np


def load_scene_instances(input_path):
    """Load the final_scene_basic.glb and extract per-object meshes."""
    glb_path = os.path.join(input_path, 'final_scene_basic.glb')
    if not os.path.exists(glb_path):
        raise FileNotFoundError(f"final_scene_basic.glb not found in {input_path}")

    scene = trimesh.load(glb_path)

    instances = {}
    if isinstance(scene, trimesh.Scene):
        node_to_geom = {}
        for node in scene.graph.nodes:
            T, geom_name = scene.graph.get(node)
            if geom_name is not None and geom_name in scene.geometry:
                node_to_geom[geom_name] = (node, T)

        for geom_name in scene.geometry:
            mesh = scene.geometry[geom_name]

            # Skip non-mesh geometries (e.g., Path3D for floor grids)
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"  ⏭️ Skipping non-mesh geometry: {geom_name} (type={type(mesh).__name__})")
                continue

            if geom_name in node_to_geom:
                node_name, T = node_to_geom[geom_name]
            else:
                T = np.eye(4)
                node_name = geom_name

            mesh.apply_transform(T)

            yup_to_zup = np.array([[1, 0, 0, 0],
                                   [0, 0, -1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]], dtype=np.float64)
            mesh.apply_transform(yup_to_zup)

            category = node_name.rsplit('_', 1)[0] if '_' in node_name else node_name
            if category not in instances:
                instances[category] = []
            instances[category].append({
                'original_mesh': mesh,
                'T': np.eye(4),
                'node_name': node_name,
            })
            print(f"  {node_name} ({geom_name}): {len(mesh.vertices)} verts, bounds={mesh.bounds[0].round(3)}~{mesh.bounds[1].round(3)}")

    return instances


def load_vggt_results(input_path):
    """Load all VGGT prediction results from the input directory."""
    color_dir = os.path.join(input_path, 'color')
    depth_dir = os.path.join(input_path, 'depth')
    ext_dir = os.path.join(input_path, 'extrinsics')
    intrinsic_path = os.path.join(input_path, 'intrinsic.txt')

    for required in [color_dir, depth_dir, ext_dir, intrinsic_path]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required path not found: {required}")

    color_files = sorted(
        [f for f in os.listdir(color_dir) if f.endswith(('.jpg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    num_frames = len(color_files)
    print(f"  {num_frames} frames, ", end='')

    colors = []
    for fname in color_files:
        img = cv2.imread(os.path.join(color_dir, fname))
        colors.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    colors = np.array(colors)

    depths = []
    for i in range(num_frames):
        d = cv2.imread(os.path.join(depth_dir, f'{i}.png'), cv2.IMREAD_UNCHANGED)
        depths.append(d.astype(np.float32) / 1000.0)
    depths = np.array(depths)

    extrinsics = []
    for i in range(num_frames):
        ext = np.loadtxt(os.path.join(ext_dir, f'{i}.txt'))
        if ext.shape == (3, 4):
            full = np.eye(4)
            full[:3, :] = ext
            ext = full
        extrinsics.append(ext)
    extrinsics = np.array(extrinsics)

    intrinsic = np.loadtxt(intrinsic_path)

    print(f"image {colors.shape[1]}x{colors.shape[2]}, intrinsic loaded")
    return {
        'colors': colors,
        'depths': depths,
        'extrinsics': extrinsics,
        'intrinsic': intrinsic,
    }


def reconstruct_world_points(depths, extrinsics, intrinsic):
    """Reconstruct world_points from depth maps + camera params (VGGT convention)."""
    from archive.stage4.projection_alignment import unproject_depth_to_world

    H, W = depths.shape[1], depths.shape[2]
    world_points = np.zeros((len(depths), H, W, 3), dtype=np.float32)

    for i, (d, ext) in enumerate(zip(depths, extrinsics)):
        world_points[i] = unproject_depth_to_world(d, ext, intrinsic)

    return world_points

