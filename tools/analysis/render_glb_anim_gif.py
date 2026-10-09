"""
把 dynamic_glb/*.glb 的动画渲染成 GIF (真实 mesh 沿 6DoF 轨迹在 3D 场景中移动)。

与 trajectory_gif.py 的区别:
  - 那个直接调 stage61_gif.render_frames(), 位姿从 stage61 的 URDF 解析来
  - 这个从已导出的 GLB 读回动画关键帧, 保证渲染的就是 GLB 里那份数据
    (可用 --glb 换成自己的 GLB 预览)

渲染后端: pyrender 0.1.45 离屏(EGL), 与 tools/stage61_gif.py 一致。
注意 pyrender 每帧必须重建 Scene —— 相机位姿只在 add(cam, pose=...) 时生效,
后续改 node.matrix 无效(见 CHANGE_LOG.md:1021)。

坐标系: GLB 是 Y-up(导出时已做 Z-up→Y-up), 场景 mesh 也是 Y-up, 直接用。

用法:
    python render_glb_anim_gif.py
    python render_glb_anim_gif.py --glb xxx.glb --scene_glb final_scene_stage5.glb
"""

import argparse
import json
import os
import struct

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

from gltf_builder import T_ZUP_TO_YUP


def load_glb(path):
    """读 GLB → (meshes: {name: trimesh}, anim: {name: (T,4,4)}, fps)

    只支持本仓库 gltf_builder.py 导出的那类文件: 单个 animation、
    全部是 MAT4 transform 的 LINEAR sampler。
    """
    raw = open(path, 'rb').read()
    magic, version, total = struct.unpack('<III', raw[:12])
    if magic != 1179937895 or version != 2:
        raise ValueError(f'not a glb v2: {path}')

    json_len, json_type = struct.unpack('<II', raw[12:20])
    if json_type != 1313821514:
        raise ValueError('bad JSON chunk')
    gltf = json.loads(raw[20:20 + json_len].decode('utf-8'))
    bin_off = 20 + json_len
    bin_len, bin_type = struct.unpack('<II', raw[bin_off:bin_off + 8])
    if bin_type != 5130562:
        raise ValueError('bad BIN chunk')
    blob = raw[bin_off + 8:bin_off + 8 + bin_len]

    def read_acc(idx):
        a = gltf['accessors'][idx]
        bv = gltf['bufferViews'][a['bufferView']]
        dt = {5126: '<f4', 5125: '<u4', 5123: '<u2'}[a['componentType']]
        ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}[a['type']]
        start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        return np.frombuffer(blob, dtype=dt, count=a['count'] * ncomp,
                             offset=start).reshape(a['count'], ncomp)

    meshes = {}
    for i, node in enumerate(gltf['nodes']):
        if 'mesh' not in node:
            continue
        me = gltf['meshes'][node['mesh']]
        prim = me['primitives'][0]
        v = read_acc(prim['attributes']['POSITION'])
        f = read_acc(prim['indices']).reshape(-1, 3)
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        if 'COLOR_0' in prim['attributes']:
            c = read_acc(prim['attributes']['COLOR_0'])
            rgba = np.clip(c * 255, 0, 255).astype(np.uint8)
            m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=rgba)
        meshes[node.get('name', str(i))] = m

    anim = {}
    if 'animations' in gltf:
        a = gltf['animations'][0]
        smp = a['samplers'][0]
        mats = read_acc(smp['output']).reshape(-1, 4, 4)
        times = read_acc(smp['input']).ravel()
        for ch in a['channels']:
            if ch['target']['path'] != 'matrix':
                continue
            name = gltf['nodes'][ch['target']['node']].get('name')
            anim[name] = (mats, times)

    fps = 30.0
    if anim:
        t0 = next(iter(anim.values()))[1]
        if len(t0) > 2:
            fps = 1.0 / float(t0[1] - t0[0])
    print(f'  meshes: {sorted(meshes)}')
    for k, (m, t) in anim.items():
        print(f"  anim '{k}': {len(m)} frames, t=[{t[0]:.3f}..{t[-1]:.3f}]s -> "
              f'{fps:.1f} fps')
    return meshes, anim, fps


def look_at(eye, target, up=(0, 1, 0)):
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    n = np.linalg.norm(fwd)
    if n < 1e-09:
        fwd = np.array([1.0, 0, 0])
    fwd /= n
    up = np.asarray(up, float)
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0.0, 1.0, 0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = true_up
    m[:3, 2] = -fwd
    m[:3, 3] = eye
    return m


def scene_bbox(meshes, anim, n_sample=8):
    """采样若干帧求整体包围盒(动画物体位置会变)。"""
    pts = []
    for name, mesh in meshes.items():
        v = mesh.vertices
        if len(v) > 2000:
            v = v[np.linspace(0, len(v) - 1, 2000).astype(int)]
        if name in anim:
            mats = anim[name][0]
            for i in np.linspace(0, len(mats) - 1, n_sample).astype(int):
                pts.append(v @ mats[i][:3, :3].T + mats[i][:3, 3])
        pts.append(v)
    allp = np.concatenate(pts)
    return allp.min(axis=0), allp.max(axis=0)


def to_pyrender_mesh(geom, radius=0.0025):
    """trimesh 几何 → pyrender.Mesh。PointCloud 没有面, 扩成小八面体实心化。"""
    if not isinstance(geom, trimesh.PointCloud):
        return pyrender.Mesh.from_trimesh(geom, smooth=False)

    v = np.asarray(geom.vertices)
    if len(v) == 0:
        return None

    offsets = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1], [1, 1, 1], [1, -1, -1],
    ], dtype=np.float64)
    verts = (v[:, None, None] + offsets[None] * radius).reshape(-1, 3)

    faces = []
    for i in range(len(v)):
        b = i * 8
        local = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                 [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
                 [0, 2, 6], [2, 1, 7], [1, 3, 6], [3, 0, 7],
                 [0, 1, 6], [2, 3, 7], [1, 0, 7], [3, 2, 6]]
        faces += [[b + a, b + c, b + d] for a, c, d in local]

    solid = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    if geom.visual is not None and geom.visual.vertex_colors is not None:
        colors = np.asarray(geom.visual.vertex_colors)
        solid.visual = trimesh.visual.ColorVisuals(
            mesh=solid,
            vertex_colors=np.tile(colors[:, :1], (len(solid.vertices), 1)),
        )
    return pyrender.Mesh.from_trimesh(solid, smooth=False)


def render_gif(glb_path, out_path, scene_glb=None, n_frames=60, res=512, elev=35.0, fps=None,
               orbit=False, trail_radius=0.004, max_scene_points=120000,
               ground_z=0.0):
    meshes, anim, gltf_fps = load_glb(glb_path)
    if not anim:
        raise SystemExit('GLB has no transform animation')

    name = next(iter(anim))
    mats = anim[name][0]
    sel = np.unique(np.linspace(0, len(mats) - 1,
                                min(n_frames, len(mats))).astype(int))

    out_fps = fps if fps else max(gltf_fps * len(sel) / len(mats), 1.0)

    static = {k: v for k, v in meshes.items() if k not in anim}
    moving = {k: v for k, v in meshes.items() if k in anim}

    if len(mats) > 1:
        pts = mats[sel][:, :3, 3]
        segs = []
        for i in range(len(pts) - 1):
            v = pts[i + 1] - pts[i]
            ln = np.linalg.norm(v)
            if ln < 1e-09:
                continue
            cyl = trimesh.creation.cylinder(radius=trail_radius, height=ln, sections=6)
            cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], v / ln))
            cyl.apply_translation((pts[i] + pts[i + 1]) / 2)
            t = i / max(len(pts) - 2, 1)
            cyl.visual = trimesh.visual.ColorVisuals(
                mesh=cyl,
                vertex_colors=np.tile(
                    np.array([[255, int(200 * (1 - t)), 0]], dtype=np.uint8),
                    (len(cyl.vertices), 1)),
            )
            segs.append(cyl)
        if segs:
            static['trajectory_trail'] = trimesh.util.concatenate(segs)

    if scene_glb and os.path.isfile(scene_glb):
        sm = trimesh.load(scene_glb, force='mesh')
        if isinstance(sm, trimesh.Scene):
            sm = trimesh.util.concatenate(tuple(sm.geometry.values()))
        if len(sm.vertices) > max_scene_points:
            idx = np.random.choice(len(sm.vertices), max_scene_points, replace=False)
            sm = trimesh.PointCloud(sm.vertices[idx])
        sm.vertices = np.asarray(sm.vertices, np.float64) @ T_ZUP_TO_YUP[:3, :3].T \
            + T_ZUP_TO_YUP[:3, 3]
        if isinstance(sm, trimesh.PointCloud):
            sm.visual = trimesh.visual.ColorVisuals(
                mesh=sm,
                vertex_colors=np.tile(np.array([(170, 170, 170)], np.uint8),
                                      (len(sm.vertices), 1)),
            )
        static['static_scene'] = sm
        print(f'  + static scene {len(sm.vertices)} pts bbox '
              f'{np.round(sm.vertices.min(0), 2)}..'
              f'{np.round(sm.vertices.max(0), 2)}')

    lo, hi = scene_bbox(meshes, anim)

    center = (lo + hi) / 2.0
    radius = max(np.linalg.norm(hi - lo) / 2.0, 0.08) * 2.6
    if 'static_scene' in static:
        sv = np.asarray(static['static_scene'].vertices)
        print(f'  object bbox {np.round(lo, 3)} .. {np.round(hi, 3)} '
              f'radius={radius:.3f}')
        print(f'  scene bbox {np.round(sv.min(0), 2)} .. '
              f'{np.round(sv.max(0), 2)}')
    else:
        print(f'  bbox {np.round(lo, 3)} .. {np.round(hi, 3)}  '
              f'radius={radius:.3f}')

    floor_y = float(np.floor(lo[1] * 100) / 100) if ground_z is None else ground_z
    print(f'  floor_y={floor_y:.3f}')

    pmesh = {}
    for k, v in static.items():
        pm = to_pyrender_mesh(v)
        if pm is not None:
            pmesh[k] = pm
    for k, v in moving.items():
        pmesh[k] = pyrender.Mesh.from_trimesh(v, smooth=False)

    # 不画地面: 物体的Y 范围跨 0(桌面高度), 地面会横穿画面把物体挡住。
    ground = None

    light_rot = Rotation.from_euler('x', 55, degrees=True).as_matrix()

    lights = [([2.0, 5.0, 3.0], [1, 1, 1], 2.4),
              ([-3.0, 3.0, -2.0], [0.85, 0.9, 1.0], 1.5)]

    fov = 50.0
    renderer = pyrender.OffscreenRenderer(res, res)
    frames = []
    el = np.radians(elev)

    for fi, si in enumerate(sel):
        sc = pyrender.Scene(ambient_light=np.array([0.38, 0.38, 0.42, 1.0]),
                            bg_color=np.array([0.96, 0.97, 1.0, 1.0]))
        for lp, col, inten in lights:
            m = np.eye(4)
            m[:3, :3] = light_rot
            m[:3, 3] = lp
            sc.add(pyrender.DirectionalLight(color=np.array(col), intensity=inten),
                   pose=m)
        if ground is not None:
            sc.add(ground, pose=np.eye(4))
        for k in static:
            sc.add(pmesh[k])
        for k in moving:
            sc.add(pmesh[k], pose=mats[si])

        theta = np.radians(120) + (2 * np.pi * fi / len(sel) if orbit else 0.0)
        eye = center + radius * np.array([
            np.cos(el) * np.cos(theta),
            np.sin(el),
            np.cos(el) * np.sin(theta),
        ])
        if ground is not None and eye[1] < floor_y + 0.02:
            eye[1] = floor_y + 0.02
        cam = pyrender.PerspectiveCamera(yfov=np.radians(fov))
        sc.add(cam, pose=look_at(eye, center))

        color, _ = renderer.render(sc)
        frames.append(color)

    import imageio.v2 as imageio
    imageio.mimsave(out_path, frames, duration=1.0 / out_fps, loop=0)
    print(f'  GIF: {out_path} ({os.path.getsize(out_path) / 1000000.0:.1f} MB, '
          f'{len(frames)} frames @ {out_fps:.1f} fps)')
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='把 GLB 动画渲染成 GIF (真实 mesh 沿轨迹移动)')
    parser.add_argument('--glb', type=str, default=None,
                        help='要渲染的 GLB (默认 <scene_dir>/dynamic_glb 下全部)')
    parser.add_argument('--scene_dir', type=str,
                        default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')

    parser.add_argument('--glb_dir', type=str, default=None,
                        help='GLB 目录 (默认 <scene_dir>/dynamic_glb)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录 (默认 <scene_dir>/dynamic_glb/gif)')
    parser.add_argument('--scene_glb', type=str, default=None,
                        help='背景场景 GLB (默认 final_scene_stage5.glb)')
    parser.add_argument('--no_scene', action='store_true', help='不加载背景场景')
    parser.add_argument('--n_frames', type=int, default=60)
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--elev', type=float, default=35.0)
    parser.add_argument('--fps', type=float, default=None)
    parser.add_argument('--orbit', action='store_true', help='相机环绕')
    parser.add_argument('--max_scene_points', type=int, default=6000,
                        help='背景场景点云采样上限')
    args = parser.parse_args()

    glb_dir = args.glb_dir or os.path.join(args.scene_dir, 'dynamic_glb')
    out_dir = args.out_dir or os.path.join(glb_dir, 'gif')
    os.makedirs(out_dir, exist_ok=True)

    if args.glb:
        glbs = [args.glb]
    else:
        glbs = sorted(os.path.join(glb_dir, f)
                      for f in os.listdir(glb_dir) if f.endswith('.glb'))
    if not glbs:
        raise SystemExit(f'no GLB in {glb_dir}')
    print(f'GLB: {glbs}')

    scene_glb = None
    if not args.no_scene:
        scene_glb = args.scene_glb or os.path.join(args.scene_dir,
                                                   'final_scene_stage5.glb')

    outputs = []
    for g in glbs:
        name = os.path.splitext(os.path.basename(g))[0]
        print(f'\n[{name}]')
        out = os.path.join(out_dir, f'{name}.gif')
        render_gif(g, out, scene_glb=scene_glb, n_frames=args.n_frames,
                   res=args.res, elev=args.elev, fps=args.fps,
                   orbit=args.orbit, max_scene_points=args.max_scene_points)
        outputs.append(out)

    print(f'\nDone. {len(outputs)} GIF(s) in {out_dir}')


if __name__ == '__main__':
    main()
