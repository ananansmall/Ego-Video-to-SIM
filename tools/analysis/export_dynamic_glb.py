"""
导出带内嵌轨迹动画的 GLB —— 只处理 stage3 判定为动态的物体。

与 trajectory_gif.py 的区别:
  - 那个用 pyrender 离屏渲染成GIF(需要 EGL/GPU, 得到的是像素)
  - 这个导出 GLB 内嵌 animation, 物体 mesh 本身沿3D 轨迹运动,
    可以直接在 Blender / MeshLab / trimesh 里播放, 也能被仿真器读入

坐标系: VGGT-Omega world 系 (Z-up, 地板 z=0), 与 extrinsics / centroids 一致。

物体位姿:
  物体的局部 mesh(从 all_instances.pkl 取 original_mesh, 已裁剪到物体自身)
  与轨迹的参考点(首帧 mask 点云质心)并不重合 —— 实测重定位后反投影命中率仅
  73%,而 mask 点云是 100%。所以这里把 mesh 平移到"mask 质心相对 mesh 质心"
  的偏移量, 使 mesh 自身质心对齐轨迹原点, 之后直接施加 6DoF 位姿。

输出 (默认 <scene_dir>/dynamic_glb/):
  <obj_key>.glb            物体 mesh + 内嵌动画 + 轨迹折线参考
  <obj_key>_trajectory.npz 该物体用到的位姿数据(便于下游核对)
"""

import argparse
import json
import os
import pickle

import numpy as np
import trimesh

from gltf_builder import export_glb_with_animation


def parse_urdf_visual_meshes(urdf_path):
    """从 URDF 的 link/visual filename 解析 {link: mesh 路径}。

    不靠文件名猜 link —— stage61_urdf_generator.py 生成的 visual filename
    是相对路径, 其 basename 形如 `<link>_<link>.obj`, 但 link 名本身含下划线,
    按文件名切分不可靠。直接读 URDF 里的 link 名最稳。
    """
    import xml.etree.ElementTree as ET

    file_dir = os.path.dirname(urdf_path)
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall('.//link'):      # 只认 <link> 元素名, 不猜文件名
        name = link.get('name')
        if name == 'base' or name.startswith(('floor', 'ground')):
            continue
        visual = link.find('visual')
        if visual is None:
            continue
        mesh_el = visual.find('geometry/mesh')    # 路径写在 @filename 属性上
        if mesh_el is None or not mesh_el.get('filename'):
            continue
        path = os.path.normpath(os.path.join(file_dir, mesh_el.get('filename')))
        if os.path.isfile(path) and not path.endswith('_collision.obj'):
            out[name] = path
    return out


def load_dynamic_info(scene_dir, info_rel):
    path = os.path.join(scene_dir, info_rel)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        info = json.load(f)
    return {k: v for k, v in info.items()
            if isinstance(v, dict) and v.get('is_dynamic')}


def load_mask_instances(scene_dir, pkl_name='_pre_dedup_masks.pkl'):
    """读 mainv2 落盘的逐帧实例 mask + world_points。"""
    with open(os.path.join(scene_dir, pkl_name), 'rb') as f:
        pkl = pickle.load(f)
    return np.asarray(pkl['world_points'], dtype=np.float64), pkl['all_masks']


def match_trajectory_instance(inst_masks, ref_centroid0, n_ref_frames,
                              world_points, max_dist=0.1):
    """把 stage3 的物体实例匹配到 _pre_dedup 的轨迹实例编号。"""
    if len(inst_masks) != n_ref_frames:      # 帧数对不上 -> 不是同一个实例
        return None
    m0 = np.asarray(inst_masks[0]['mask']) > 0
    if m0.sum() < 20:                        # 首帧 mask 太小时质心不可靠
        return None
    c0 = world_points[inst_masks[0]['frame_id']][m0].mean(0)
    d = np.linalg.norm(ref_centroid0 - c0)  # 首帧质心距离, 单位 m
    return (c0, d) if d < max_dist else None


def export_animated_glb(out_path, static_geoms, moving_geoms, anim_transforms, fps):
    """导出带 transform animation 的 GLB(经gltf_builder)。

    static_geoms: {name: mesh}静止物体(含轨迹折线/场景)
    moving_geoms: {name: mesh}   运动物体
    anim_transforms: {name: (N,4,4)} 每个运动物体的位姿序列(世界系 Z-up)
    """

    meshes = dict(static_geoms)             # 静止几何先进字典
    meshes.update(moving_geoms)             # 再并入运动几何(同名会覆盖)
    gltf = export_glb_with_animation(out_path, meshes, anim_transforms, fps=fps)
    return gltf


def main():
    parser = argparse.ArgumentParser(
        description='导出 stage3 动态物体的带轨迹动画 GLB')
    parser.add_argument('--scene_dir', type=str,
                        default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')

    parser.add_argument('--dynamic_info', type=str,
                        default='dynamic_pointclouds/trajectory_info.json')
    parser.add_argument('--traj_dir', type=str, default=None,
                        help='轨迹目录 (默认 <scene_dir>/object_trajectories)')
    parser.add_argument('--glb', type=str, default=None,
                        help='背景场景 GLB (默认 final_scene_stage5.glb)')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='输出目录 (默认 <scene_dir>/dynamic_glb)')
    parser.add_argument('--stride', type=int, default=2,
                        help='位姿抽帧步长 (1=每帧都导出)')
    parser.add_argument('--frame_rate', type=int, default=30,
                        help='动画帧率')
    parser.add_argument('--no_trail', action='store_true',
                        help='不附带轨迹折线')
    parser.add_argument('--with_scene', action='store_true',
                        help='把 final_scene 的静态物体一起打进 GLB')
    parser.add_argument('--max_scene_points', type=int, default=150000)
    args = parser.parse_args()

    scene_dir = args.scene_dir
    traj_dir = args.traj_dir or os.path.join(scene_dir, 'object_trajectories')
    out_dir = args.out_dir or os.path.join(scene_dir, 'dynamic_glb')
    os.makedirs(out_dir, exist_ok=True)

    dyn_info = load_dynamic_info(scene_dir, args.dynamic_info)
    if not dyn_info:
        raise SystemExit(f'no dynamic object in {args.dynamic_info}')
    print(f'stage3 dynamic objects: {list(dyn_info.keys())}')

    world_points, all_masks = load_mask_instances(scene_dir)

    # 编号映射: stage3 的编号基于去重**后**的 all_instances, 轨迹用的是去重**前**
    key_map = {}
    for key, v in dyn_info.items():
        cat, inst = key.rsplit('_inst', 1)
        ref_c0 = np.array(v['centroids'])[0]
        for i, masks in enumerate(all_masks.get(cat, [])):
            m = match_trajectory_instance(masks, ref_c0,
                                          len(v['visible_frames']), world_points)
            if m is not None:
                mask_c0, d = m
                key_map[key] = (f'{cat}_inst{i}', mask_c0, d)
                print(f"  stage3 '{key}' -> 轨迹 '{cat}_inst{i}' (首帧质心差 "
                      f"{d * 1000:.1f} mm)")
                break
    if not key_map:
        raise SystemExit('failed to match any dynamic object to trajectory')

    # URDF link 名含下划线, 只能读 <link> 元素名, 不能靠 mesh 文件名切分
    urdf_path = os.path.join(scene_dir, 'urdf', 'scene.urdf')
    mesh_cache = {}
    if os.path.isfile(urdf_path):
        mesh_cache = parse_urdf_visual_meshes(urdf_path)
        print(f'URDF links with visual mesh: {sorted(mesh_cache)}')

    def load_scene_glb():
        if args.with_scene:
            p = args.glb or os.path.join(scene_dir, 'final_scene_stage5.glb')
            if os.path.isfile(p):
                m = trimesh.load(p, force='mesh')
                if isinstance(m, trimesh.Scene):
                    m = trimesh.util.concatenate(tuple(m.geometry.values()))
                if len(m.vertices) > args.max_scene_points:   # 静态背景抽稀成点云
                    idx = np.random.choice(len(m.vertices),
                                           args.max_scene_points, replace=False)
                    m = trimesh.PointCloud(m.vertices[idx])
                m.visual = trimesh.visual.ColorVisuals(
                    mesh=m,
                    vertex_colors=np.tile(np.array([[200, 200, 200]], dtype=np.uint8),
                                          (len(m.vertices), 1)))
                return m
            print(f'scene glb not found: {p}')
        return None

    scene_mesh = load_scene_glb()

    for key, (traj_key, mask_c0, match_d) in key_map.items():
        npz_path = os.path.join(traj_dir, f'{traj_key}_traj.npz')
        if not os.path.isfile(npz_path):
            print(f'\nskip {key}: no trajectory {npz_path}')
            continue
        d = np.load(npz_path, allow_pickle=True)
        traj = d['trajectory']
        centroids = d['centroids']
        valid = d['valid_frames']

        vidx = np.where(valid)[0]
        if len(vidx) < 5:
            print(f'\nskip {key}: only {len(vidx)} valid frames')
            continue

        # URDF link 名 = <category>_<去重后序号>, 名字本身含下划线, 只能这样拼
        cat, inst = key.rsplit('_inst', 1)
        link = f'{cat}_{int(inst)}'
        mesh_path = mesh_cache.get(link)
        if mesh_path is None or not os.path.isfile(mesh_path):
            print(f"\nskip {key}: no URDF mesh for link '{link}'")
            continue

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

        # mesh 自身质心对齐轨迹原点, 之后直接施加 6DoF 位姿(见模块 docstring)
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)

        # stride 抽帧: 步长越大动画越短
        sel = vidx[::args.stride]
        transforms = np.tile(np.eye(4), (len(sel), 1, 1))
        for i, f in enumerate(sel):
            transforms[i, :3, :3] = traj[f][:3, :3]
            transforms[i, :3, 3] = centroids[f]

        out_name = key
        glb_path = os.path.join(out_dir, f'{out_name}.glb')

        moving = {f'moving_{key}': mesh}
        anims = {f'moving_{key}': transforms}

        static = {}
        if scene_mesh is not None:
            static['static_scene'] = scene_mesh

        if not args.no_trail:
            trail = build_trail_mesh(centroids[sel])
            if trail is not None:
                static['trajectory_trail'] = trail

        export_animated_glb(glb_path, static, moving, anims, fps=args.frame_rate)

        np.savez(os.path.join(out_dir, f'{out_name}_trajectory.npz'),
                 trajectory=traj[sel], centroids=centroids[sel],
                 valid_frames=valid[sel], urdf_link=link,
                 source_trajectory=traj_key)

        print(f'\n[{key}] link={link} mesh={len(mesh.vertices)} verts frames='
              f'{len(sel)}')
        print(f'  displacement = '
              f'{np.linalg.norm(transforms[-1][:3, 3] - transforms[0][:3, 3]):.3f} m')
        print(f'  GLB: {glb_path} ({os.path.getsize(glb_path) / 1000000.0:.1f} MB)')


def build_trail_mesh(points, radius=0.003):
    """轨迹折线(红->蓝渐变) 作为静态参照。"""
    if len(points) < 2:
        return None
    geoms = []
    n = len(points)
    for i in range(n - 1):
        seg = trimesh.creation.cylinder(radius=radius,
                                        height=float(np.linalg.norm(points[i + 1] - points[i])),
                                        sections=6)
        vec = points[i + 1] - points[i]
        ln = np.linalg.norm(vec)
        if ln < 1e-9:                       # 跳过重合点, 否则方向无定义
            continue
        seg.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], vec / ln))
        seg.apply_translation((points[i] + points[i + 1]) / 2)   # 移到线段中点
        t = i / max(n - 2, 1)
        seg.visual = trimesh.visual.ColorVisuals(
            mesh=seg,
            vertex_colors=np.tile(np.array([[255 * (1 - t), 0, 255 * t]],
                                          dtype=np.uint8), (len(seg.vertices), 1)))
        geoms.append(seg)
    if not geoms:
        return None
    return trimesh.util.concatenate(geoms)


if __name__ == '__main__':
    main()
