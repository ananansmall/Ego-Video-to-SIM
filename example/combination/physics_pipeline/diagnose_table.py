"""诊断 GLB 物体 Z 值分布和桌子位置问题

检查:
1. 每个几何体的 Z 范围、体积、flatness、max_extent
2. is_static 分类结果
3. dynamic 物体的 min_z (用于桌子高度)
4. 相机位置 vs 桌子位置
"""
import sys
import numpy as np
import trimesh
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from pybullet_pipeline import RXWORLD_TO_SAPIEN

GLB_PATH = Path(__file__).parent.parent / "output" / "alignment" / "scene_in_sapien.glb"
TRANSFORM_PARAMS = Path(__file__).parent.parent / "output" / "alignment" / "transform_params.npz"

print(f"GLB: {GLB_PATH}")
print(f"Transform: {TRANSFORM_PARAMS}")
print(f"GLB exists: {GLB_PATH.exists()}")
print(f"Transform exists: {TRANSFORM_PARAMS.exists()}")

params = np.load(str(TRANSFORM_PARAMS))
s_inv = float(params['s_inv'])
R_inv = params['R_inv']
t_inv = params['t_inv']
print(f"\nTransform params: s_inv={s_inv:.4f}, t_inv={t_inv}")

scene = trimesh.load(str(GLB_PATH))
print(f"\nGeometries: {len(scene.geometry)}")

all_verts_sapien = []
geom_info = []

for geom_name, geom in scene.geometry.items():
    if len(geom.vertices) == 0:
        continue
    vertices = geom.vertices.copy()
    vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
    vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
    all_verts_sapien.append(vertices_sapien)

    bbox_size = vertices_sapien.max(axis=0) - vertices_sapien.min(axis=0)
    volume = abs(bbox_size[0] * bbox_size[1] * bbox_size[2])
    max_extent = max(bbox_size)
    flatness = bbox_size[2] / max(max(bbox_size[0], bbox_size[1]), 1e-6)
    is_static = (volume > 0.01 and flatness < 0.3) or max_extent > 0.8

    z_min = vertices_sapien[:, 2].min()
    z_max = vertices_sapien[:, 2].max()
    centroid = vertices_sapien.mean(axis=0)

    geom_info.append({
        'name': geom_name,
        'z_min': z_min,
        'z_max': z_max,
        'height': z_max - z_min,
        'volume': volume,
        'flatness': flatness,
        'max_extent': max_extent,
        'is_static': is_static,
        'centroid': centroid,
        'bbox': bbox_size,
    })

    print(f"\n  {geom_name}:")
    print(f"    Z range: [{z_min:.4f}, {z_max:.4f}] (height={z_max-z_min:.4f}m)")
    print(f"    XY centroid: ({centroid[0]:.4f}, {centroid[1]:.4f})")
    print(f"    BBox: x={bbox_size[0]:.4f} y={bbox_size[1]:.4f} z={bbox_size[2]:.4f}")
    print(f"    Volume: {volume:.6f} m3, flatness: {flatness:.4f}, max_extent: {max_extent:.4f}")
    print(f"    is_static: {is_static}")

# 桌子高度计算 (当前逻辑)
dynamic_verts_z = [g['z_min'] for g in geom_info if not g['is_static']]
static_verts_z = [g['z_min'] for g in geom_info if g['is_static']]

print(f"\n{'='*60}")
print(f"Static objects: {len(static_verts_z)}")
if static_verts_z:
    print(f"  Static min Z: {min(static_verts_z):.4f}m")
    print(f"  Static max Z: {max(static_verts_z):.4f}m")

print(f"Dynamic objects: {len(dynamic_verts_z)}")
if dynamic_verts_z:
    print(f"  Dynamic min Z: {min(dynamic_verts_z):.4f}m")
    print(f"  Dynamic max Z: {max([g['z_max'] for g in geom_info if not g['is_static']]):.4f}m")

if dynamic_verts_z:
    min_z = min(dynamic_verts_z)
    print(f"\n  -> Table top (current logic): {min_z - 0.002:.4f}m")
    print(f"     (based on dynamic min_z = {min_z:.4f}m)")

# 所有物体的 Z 范围
all_z_min = min(g['z_min'] for g in geom_info)
all_z_max = max(g['z_max'] for g in geom_info)
print(f"\nAll objects Z range: [{all_z_min:.4f}, {all_z_max:.4f}]m")
print(f"Z span: {all_z_max - all_z_min:.4f}m")

# 检查相机位置 (第一帧)
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("mod02", Path(__file__).parent.parent / "02_render_scene.py")
    mod02 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod02)
    hawor_dir = Path(__file__).parent.parent / "output" / "7_my_7mp4_result"
    # 尝试找到 hawor 目录
    possible_hawor = [
        Path(__file__).parent.parent / "output" / "7_my_7mp4_result",
        Path(__file__).parent.parent / "hand_track" / "output" / "7",
    ]
    for hd in possible_hawor:
        if hd.exists():
            print(f"\nHaWoR dir: {hd}")
            try:
                R_c2w_all, t_c2w_all = mod02.load_hawor_c2w(hd)
                if R_c2w_all is not None and len(t_c2w_all) > 0:
                    cam_pos = RXWORLD_TO_SAPIEN @ t_c2w_all[0]
                    print(f"  Camera[0] pos (SAPIEN): {cam_pos}")
                    print(f"  Camera Z: {cam_pos[2]:.4f}m")
                    break
            except Exception as e:
                print(f"  load_hawor_c2w failed: {e}")
except Exception as e:
    print(f"\nCamera check skipped: {e}")

print(f"\n{'='*60}")
print("诊断完成")
