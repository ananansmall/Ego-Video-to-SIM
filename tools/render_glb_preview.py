#!/usr/bin/env python3
"""Render a GLB scene to a multi-view preview PNG (headless EGL, no display needed).

Usage:
    python tools/render_glb_preview.py output_v2/xxx/final_scene.glb
    python tools/render_glb_preview.py a.glb b.glb --output preview.png --views front,iso

Outputs a 2x2 grid (front / side / top / iso) with vertex colors or textures,
written next to the GLB as <stem>_preview.png by default.
"""

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import numpy as np
import trimesh
import pyrender
import cv2


def build_pyrender_scene(glb_path):
    """Load GLB, convert mesh geometries to a pyrender scene."""
    loaded = trimesh.load(glb_path, force="scene")
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.35, 0.35, 0.35])

    geometries = loaded.geometry.values() if isinstance(loaded, trimesh.Scene) else [loaded]
    n_added = 0
    for geom in geometries:
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue  # 跳过 grid 线条等非 mesh 几何
        prm = None
        try:
            if geom.visual.kind == "texture" and getattr(geom.visual.material, "baseColorTexture", None) is not None:
                mat = pyrender.MetallicRoughnessMaterial(
                    baseColorTexture=geom.visual.material.baseColorTexture,
                    metallicFactor=0.0, roughnessFactor=0.9,
                    alphaMode="BLEND" if geom.visual.material.alphaMode == "BLEND" else "OPAQUE",
                )
                prm = pyrender.Mesh.from_trimesh(geom, material=mat, smooth=False)
        except Exception:
            prm = None
        if prm is None:
            prm = pyrender.Mesh.from_trimesh(geom, smooth=False)
        if len(prm.primitives) > 0:
            scene.add(prm)
            n_added += 1
    return scene, n_added


def look_at(eye, target, up=np.array([0.0, 0.0, 1.0])):
    """Camera pose (c2w) looking at target. trimesh 加载 GLB 后坐标为 z-up (见 w2c_to_pyrender_pose 说明)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:  # 视线接近 up 方向时换参考轴
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)
    pose[:3, 3] = eye
    return pose


def render_views(scene, bounds, views, size):
    center = bounds.mean(axis=0)
    radius = float(np.linalg.norm(bounds[1] - bounds[0])) / 2
    dist = radius * 2.6
    dirs = {  # trimesh 加载后为 z-up 坐标: z 是高度方向
        "front": np.array([0.0, -1.0, 0.5]),    # -y 方向看, 略微俯视
        "side":  np.array([1.0, 0.0, 0.5]),     # +x 方向看
        "top":   np.array([0.0, 0.0, 1.0]),     # 从正上方俯视
        "iso":   np.array([0.8, -0.8, 0.7]),
    }
    cam = pyrender.IntrinsicsCamera(fy=size, fx=size, cy=size / 2, cx=size / 2)  # fov≈53°
    light = pyrender.DirectionalLight(intensity=3.0)
    renderer = pyrender.OffscreenRenderer(size, size)

    tiles = []
    try:
        first_dir = dirs[views[0]] / np.linalg.norm(dirs[views[0]])
        first_pose = look_at(center + first_dir * dist, center)
        cam_node = scene.add(cam, pose=first_pose, name="preview_cam")
        light_node = scene.add(light, pose=first_pose, name="preview_light")

        for name in views:
            d = dirs[name] / np.linalg.norm(dirs[name])
            pose = look_at(center + d * dist, center)
            scene.set_pose(cam_node, pose)
            scene.set_pose(light_node, pose)
            img, _ = renderer.render(scene)
            img = img.copy()
            cv2.putText(img, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(img)
    finally:
        renderer.delete()
    return tiles


def grid2x2(tiles):
    while len(tiles) < 4:
        tiles.append(np.full_like(tiles[0], 255))
    top = np.hstack([tiles[0], tiles[1]])
    bot = np.hstack([tiles[2], tiles[3]])
    return np.vstack([top, bot])


def grid_rows(tiles, cols=2):
    """任意数量 tile → 按行拼接 (不足补白)."""
    while len(tiles) % cols != 0:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


# ── 真实相机视角渲染 ─────────────────────────────────────────
# 坐标系说明: trimesh 加载 GLB 时会把 GLB 的 y-up 约定自动转回 trimesh 内部 z-up，
# 对本管线导出的 GLB，净效果 = 恢复管线原始 z-up 世界坐标 (与 extrinsics/world_points 同一坐标系)。
# 因此相机位姿无需再做 y-up 翻转，只需 OpenCV→OpenGL 相机约定转换 (与 stage4/renderer.py 渲染几何部分一致)。


def load_camera_rig(scene_dir):
    """从场景输出目录读取相机数据.

    Returns:
        (intrinsic 3x3, frame_size (W, H), {frame_id: w2c 4x4}) 或 None
    """
    intrinsic_path = os.path.join(scene_dir, "intrinsic.txt")
    extrinsics_dir = os.path.join(scene_dir, "extrinsics")
    if not (os.path.isfile(intrinsic_path) and os.path.isdir(extrinsics_dir)):
        return None
    intrinsic = np.loadtxt(intrinsic_path).reshape(3, 3)
    color0 = os.path.join(scene_dir, "color", "0.jpg")
    if os.path.isfile(color0):
        import cv2 as _cv2
        h, w = _cv2.imread(color0).shape[:2]
    else:
        h, w = int(round(2 * intrinsic[1, 2])), int(round(2 * intrinsic[0, 2]))
    extrinsics = {}
    for fname in os.listdir(extrinsics_dir):
        if fname.endswith(".txt"):
            fid = int(os.path.splitext(fname)[0])
            extrinsics[fid] = np.loadtxt(os.path.join(extrinsics_dir, fname)).reshape(4, 4)
    if not extrinsics:
        return None
    return intrinsic, (w, h), extrinsics


def w2c_to_pyrender_pose(w2c):
    """w2c (z-up 世界, OpenCV 相机 +z 前向 +y 向下) → pyrender c2w (OpenGL 相机 -z 前向 +y 向上).

    世界保持 z-up (与 trimesh 加载的 GLB 几何同坐标系), 仅转换相机内约定。
    """
    c2w = np.linalg.inv(w2c)
    return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])


def get_optimal_frame_ids(scene_dir, cap=6):
    """从 optimal_frames/ 文件名解析各实例的'最大帧' (最大表面积视角帧)."""
    opt_dir = os.path.join(scene_dir, "optimal_frames")
    if not os.path.isdir(opt_dir):
        return []
    import re
    fids = set()
    for fname in os.listdir(opt_dir):
        m = re.search(r"frame(\d+)\.jpg$", fname)
        if m:
            fids.add(int(m.group(1)))
    return sorted(fids)[:cap]


def render_camera_views(glb_path, scene_dir, frame_ids, max_width=1024):
    """用输出目录里的真实相机外参渲染 GLB, 返回 (tiles, (W,H)).

    每帧渲染分辨率与 color/{fid}.jpg 一致 (超过 max_width 时等比缩小内参).
    """
    import cv2 as _cv2
    rig = load_camera_rig(scene_dir)
    if rig is None:
        raise FileNotFoundError(f"{scene_dir} 下未找到 intrinsic.txt / extrinsics/")
    intrinsic, (W, H), extrinsics = rig

    scale = min(1.0, max_width / W)
    fw, fh = int(W * scale), int(H * scale)
    fx, fy = intrinsic[0, 0] * scale, intrinsic[1, 1] * scale
    cx, cy = intrinsic[0, 2] * scale, intrinsic[1, 2] * scale

    scene, n_meshes = build_pyrender_scene(glb_path)
    if n_meshes == 0:
        raise ValueError(f"{glb_path} 中没有 mesh 几何")
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=200.0)
    light = pyrender.DirectionalLight(intensity=3.0)
    renderer = pyrender.OffscreenRenderer(fw, fh)

    tiles = []
    try:
        first_pose = w2c_to_pyrender_pose(extrinsics[frame_ids[0]])
        cam_node = scene.add(cam, pose=first_pose, name="camview_cam")
        light_node = scene.add(light, pose=first_pose, name="camview_light")
        for fid in frame_ids:
            if fid not in extrinsics:
                print(f"⚠️ 缺少 extrinsics/{fid}.txt, 跳过")
                continue
            pose = w2c_to_pyrender_pose(extrinsics[fid])
            scene.set_pose(cam_node, pose)
            scene.set_pose(light_node, pose)
            img, _ = renderer.render(scene)
            img = img.copy()
            _cv2.putText(img, f"cam{fid}", (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, _cv2.LINE_AA)
            tiles.append(img)
    finally:
        renderer.delete()
    return tiles, (fw, fh)


def render_camview_auto(glb_path, cap=6):
    """管线集成入口: 自动探测场景目录的相机数据, 渲染真实相机视角 (含各实例'最大帧').

    Returns: 输出 PNG 路径, 无相机数据时返回 None
    """
    import cv2 as _cv2
    scene_dir = os.path.dirname(os.path.abspath(glb_path))
    if load_camera_rig(scene_dir) is None:
        return None
    frame_ids = get_optimal_frame_ids(scene_dir, cap=cap) or [0]
    tiles, _ = render_camera_views(glb_path, scene_dir, frame_ids)
    out = os.path.splitext(glb_path)[0] + "_camview.png"
    _cv2.imwrite(out, _cv2.cvtColor(grid_rows(tiles), _cv2.COLOR_RGB2BGR))
    print(f"💾 {out}  (camera views: {','.join(str(f) for f in frame_ids)})")
    return out


def render_preview_png(glb_path, views=("front", "side", "top", "iso"), size=640):
    """渲染 GLB 多视角合成预览图 (默认四视图 front/side/top/iso)

    保存为 <glb 同目录>/<stem>_preview.png, 返回输出路径; 失败返回 None。
    供管线在 Stage5 保存最终 GLB 后自动调用。
    """
    try:
        import cv2
        scene, n_meshes = build_pyrender_scene(glb_path)
        if n_meshes == 0:
            print(f"⚠️ {glb_path}: 没有 mesh 几何, 跳过预览", flush=True)
            return None
        bounds = scene.bounds
        tiles = render_views(scene, bounds, list(views), size)
        grid = grid2x2(tiles) if len(tiles) > 1 else tiles[0]
        out = os.path.join(
            os.path.dirname(os.path.abspath(glb_path)),
            f"{os.path.splitext(os.path.basename(glb_path))[0]}_preview.png")
        cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"   🖼️ 四视图预览已生成: {out}", flush=True)
        return out
    except Exception as e:
        print(f"   ⚠️ 四视图预览渲染失败: {e}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="GLB → 多视角预览 PNG (无头 EGL 渲染)")
    parser.add_argument("glb", nargs="+", help="GLB 文件路径 (可多个)")
    parser.add_argument("--output", default=None, help="输出 PNG (仅单个 GLB 时有效; 默认 <stem>_preview.png)")
    parser.add_argument("--views", default="front,side,top,iso",
                        help="逗号分隔合成视角: front,side,top,iso | none(只渲染相机视角)")
    parser.add_argument("--camera_frame", default=None,
                        help="真实相机视角: optimal(各实例最大帧) | 逗号分隔帧号 (如 0,6,85) | 单个帧号")
    parser.add_argument("--size", type=int, default=640, help="合成视角单帧分辨率 (像素)")
    parser.add_argument("--cam_max_width", type=int, default=1024, help="相机视角渲染最大宽度")
    args = parser.parse_args()

    views = [v.strip() for v in args.views.split(",") if v.strip() and v.strip() != "none"]
    outputs = []
    for glb_path in args.glb:
        out = args.output
        if out is None or len(args.glb) > 1:
            stem = os.path.splitext(os.path.basename(glb_path))[0]
            out = os.path.join(os.path.dirname(os.path.abspath(glb_path)), f"{stem}_preview.png")

        if views:
            scene, n_meshes = build_pyrender_scene(glb_path)
            if n_meshes == 0:
                print(f"⚠️ {glb_path}: 没有 mesh 几何, 跳过")
                continue
            bounds = scene.bounds  # (2,3): [min-corner, max-corner] 已聚合全部节点
            tiles = render_views(scene, bounds, views, args.size)
            grid = grid2x2(tiles) if len(tiles) > 1 else tiles[0]
            cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"💾 {out}  ({n_meshes} meshes, views: {','.join(views)})")
            outputs.append(out)

        if args.camera_frame:
            scene_dir = os.path.dirname(os.path.abspath(glb_path))
            if args.camera_frame == "optimal":
                frame_ids = get_optimal_frame_ids(scene_dir) or [0]
            else:
                frame_ids = [int(x) for x in args.camera_frame.split(",") if x.strip()]
            cam_tiles, _ = render_camera_views(glb_path, scene_dir, frame_ids, args.cam_max_width)
            cam_stem = os.path.splitext(glb_path)[0]
            cam_out = f"{cam_stem}_camview.png"
            cv2.imwrite(cam_out, cv2.cvtColor(grid_rows(cam_tiles), cv2.COLOR_RGB2BGR))
            print(f"💾 {cam_out}  (camera views: {','.join(str(f) for f in frame_ids)})")
            outputs.append(cam_out)

    if not outputs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
