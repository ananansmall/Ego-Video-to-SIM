"""
用**真实视频相机视角**在世界坐标系里渲染 FP 位姿动画 GIF。

与 render_glb_anim_gif.py 的区别:
  - 那个是环绕轨道相机(orbit), 与原视频视角不一致;
  - 这个直接用 extrinsics 的逆(= cam→world)得到每帧真实相机位姿,
    渲染画面与原视频对齐, 便于逐帧核对位姿是否正确。

GLB 本身是 Y-up(导出时做了 Z-up→Y-up), 所以先把 Y-up 相机位姿转回 Z-up
世界系再算 eye/target, 避免坐标系混用。

输出 <scene_dir>/fp_glb/gif_camview/<obj>_camview.gif
"""
import argparse
import os
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import pyrender
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from gltf_builder import T_ZUP_TO_YUP  # noqa: E402
from render_glb_anim_gif import load_glb, to_pyrender_mesh  # noqa: E402


def look_at_up(eye, target, up):
    """同render_glb_anim_gif.look_at, 但 up 可指定(原版硬编码 (0,1,0))。"""
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    n = np.linalg.norm(fwd)
    fwd = np.array([1.0, 0, 0]) if n < 1e-9 else fwd / n
    up = np.asarray(up, float)
    if abs(np.dot(fwd, up)) > 0.999:
        up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0] = right
    m[:3, 1] = true_up
    m[:3, 2] = -fwd
    m[:3, 3] = eye
    return m


def world_cam_positions(scene_dir, n):
    """真实相机在世界系(Z-up)的位姿 = inv(extrinsics)。"""
    ext_dir = os.path.join(scene_dir, 'extrinsics')
    files = sorted(os.listdir(ext_dir),
                   key=lambda x: int(os.path.splitext(x)[0]))[:n]
    return [np.linalg.inv(np.loadtxt(os.path.join(ext_dir, f))) for f in files]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    ap.add_argument('--glb', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--n_frames', type=int, default=60)
    ap.add_argument('--res', type=int, default=640)
    ap.add_argument('--trail_radius', type=float, default=0.004)
    ap.add_argument('--max_scene_points', type=int, default=60000)
    ap.add_argument('--scene_glb', default=None)
    ap.add_argument('--no_scene', action='store_true')
    ap.add_argument('--fov', type=float, default=45.0)
    ap.add_argument('--lead', type=float, default=0.45,
                    help='视线沿物体方向前移比例, 越大物体越靠画面下缘(给运动留出上方空间)')
    ap.add_argument('--back_off', type=float, default=0.0,
                    help='相机沿视线后移倍数(物体局部尺度很小, 需拉远看全场景)')
    args = ap.parse_args()

    sd = args.scene_dir
    glb = args.glb or os.path.join(sd, 'fp_glb', 'toy_inst1.glb')
    out_dir = args.out_dir or os.path.join(sd, 'fp_glb', 'gif_camview')
    os.makedirs(out_dir, exist_ok=True)

    meshes, anim, gltf_fps = load_glb(glb)
    if not anim:
        raise SystemExit('GLB has no transform animation')
    name = next(iter(anim))
    mats = anim[name][0]
    n_glb = len(mats)
    sel = np.unique(np.linspace(0, n_glb - 1, min(args.n_frames, n_glb)).astype(int))

    moving = {k: v for k, v in meshes.items() if k in anim}
    static = {}

    # 轨迹尾迹(已在 GLB 的 Y-up 世界系)
    pts = mats[sel][:, :3, 3]
    segs = []
    for i in range(len(pts) - 1):
        v = pts[i + 1] - pts[i]
        ln = np.linalg.norm(v)
        if ln < 1e-09:
            continue
        cyl = trimesh.creation.cylinder(radius=args.trail_radius, height=ln, sections=6)
        cyl.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], v / ln))
        cyl.apply_translation((pts[i] + pts[i + 1]) / 2)
        t = i / max(len(pts) - 2, 1)
        cyl.visual = trimesh.visual.ColorVisuals(
            mesh=cyl,
            vertex_colors=np.tile(np.array([[255, int(190 * (1 - t)), 0]], np.uint8),
                                 (len(cyl.vertices), 1)))
        segs.append(cyl)
    if segs:
        static['trail'] = trimesh.util.concatenate(segs)

    # 背景场景点云(Z-up world → Y-up)
    scene_glb = args.scene_glb or os.path.join(sd, 'final_scene_stage5.glb')
    if not args.no_scene and os.path.isfile(scene_glb):
        sm = trimesh.load(scene_glb, force='mesh')
        if isinstance(sm, trimesh.Scene):
            sm = trimesh.util.concatenate(tuple(sm.geometry.values()))
        if len(sm.vertices) > args.max_scene_points:
            idx = np.random.choice(len(sm.vertices), args.max_scene_points, replace=False)
            sm = trimesh.PointCloud(sm.vertices[idx])
        sm.vertices = np.asarray(sm.vertices, np.float64) @ T_ZUP_TO_YUP[:3, :3].T \
            + T_ZUP_TO_YUP[:3, 3]
        if isinstance(sm, trimesh.PointCloud):
            sm.visual = trimesh.visual.ColorVisuals(
                mesh=sm,
                vertex_colors=np.tile(np.array([(175, 175, 178)], np.uint8),
                                      (len(sm.vertices), 1)))
        static['scene'] = sm
        print(f'  background: {len(sm.vertices)} pts')

    cam_poses = world_cam_positions(sd, n_glb)
    print(f'  cameras: {len(cam_poses)} poses from extrinsics')

    pmesh = {}
    for k, v in static.items():
        pm = to_pyrender_mesh(v)
        if pm is not None:
            pmesh[k] = pm
    for k, v in moving.items():
        pmesh[k] = pyrender.Mesh.from_trimesh(v, smooth=False)

    light_rot = Rotation.from_euler('x', 55, degrees=True).as_matrix()
    lights = [([2.0, 5.0, 3.0], [1, 1, 1], 2.2),
              ([-3.0, 3.0, -2.0], [0.85, 0.9, 1.0], 1.4)]
    renderer = pyrender.OffscreenRenderer(args.res, args.res)
    frames = []

    for si in sel:
        C = cam_poses[si]                        # world (Z-up) camera pose
        # 注意: mats 来自 GLB, 已经是 Y-up 世界系(导出时已转过), 不能再乘 T_ZUP_TO_YUP;
        # 而 cam_poses 来自 extrinsics, 是 Z-up 世界系, 必须转。
        cam_pos_y = T_ZUP_TO_YUP[:3, :3] @ C[:3, 3] + T_ZUP_TO_YUP[:3, 3]
        obj_pos_y = mats[si][:3, 3]
        # 不假设相机前向轴约定, 直接由「相机位置 → 物体位置」定视线方向。
        # 物体在画面里偏左/右交给 up 向量自然处理。
        view = obj_pos_y - cam_pos_y
        dist = np.linalg.norm(view)
        if dist < 1e-6:
            view = np.array([1.0, 0, 0])
        # up 不能与视线平行, 否则 look_at 退化导致俯视/翻转。
        # 用世界 Y-up 投影到垂直于视线的平面上作为 up 参考。
        up_ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(up_ref, view) / dist) > 0.999:
            up_ref = np.array([0.0, 0.0, 1.0])
        cam_up = up_ref - view * (np.dot(up_ref, view) / (dist * dist))
        cam_up /= np.linalg.norm(cam_up)
        # 保持真实相机位置与朝向, 只把视线中心对准物体:
        # 沿「相机→物体」方向取一点做target, eye 即真实相机位置。
        eye = cam_pos_y
        target = obj_pos_y + view * (args.lead)
        # 固定 fov: 真实摄像机视角在整段视频里不变, 画面大小不应随距离抖动。
        fov = args.fov

        sc = pyrender.Scene(ambient_light=np.array([0.40, 0.40, 0.44, 1.0]),
                            bg_color=np.array([0.97, 0.98, 1.0, 1.0]))
        for lp, col, inten in lights:
            m = np.eye(4)
            m[:3, :3] = light_rot
            m[:3, 3] = lp
            sc.add(pyrender.DirectionalLight(color=np.array(col), intensity=inten), pose=m)
        for k in pmesh:
            if k in moving:
                sc.add(pmesh[k], pose=mats[si])
            else:
                sc.add(pmesh[k])
        cam = pyrender.PerspectiveCamera(yfov=np.radians(fov))
        sc.add(cam, pose=look_at_up(eye, target, cam_up))
        color, _ = renderer.render(sc)
        frames.append(color)

    import imageio.v2 as imageio
    out = os.path.join(out_dir,
                       os.path.basename(glb).replace('.glb', '_camview.gif'))
    imageio.mimsave(out, frames,
                    duration=1.0 / max(gltf_fps * len(sel) / n_glb, 1.0), loop=0)
    print(f'GIF: {out} ({os.path.getsize(out)/1e6:.1f} MB, {len(frames)} frames)')


if __name__ == '__main__':
    main()