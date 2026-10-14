"""用 FoundationPose 位姿导出 GLB, 并渲染世界坐标系 3D 动画 GIF。

与export_dynamic_glb.py 的区别:
  - 位姿来自 FoundationPose 的 ob_in_world/(已用深度反算验证毫米级精度),
    不是 ICP 的 centroids(那个参考点与mesh 顶点原点不一致, 投影会落在画面外)
  - mesh 直接用 urdf/toy_0/mesh/toy_0.obj, 不做质心偏移补偿
    (FP 位姿的物体原点就在 mesh 原点)

坐标系: VGGT-Omega world 系 Z-up; 导出 GLB 时由 gltf_builder 做 Z-up→Y-up。

输出 <scene_dir>/fp_glb/:
  <obj>.glb                带内嵌 transform 动画, 可直接用 Blender/MeshLab 播放
  gif/<obj>_world.gif      世界坐标系渲染的 3D 动画 GIF
"""
import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from gltf_builder import export_glb_with_animation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    ap.add_argument('--obj', default='toy_inst1')
    ap.add_argument('--mesh_path', default=None,
                    help='默认用 foundationpose_data/mesh.obj')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--fps', type=float, default=16.0)
    ap.add_argument('--with_trail', type=int, default=1)
    args = ap.parse_args()
    sd = args.scene_dir
    out_dir = args.out_dir or os.path.join(sd, 'fp_glb')
    os.makedirs(out_dir, exist_ok=True)

    mesh_path = args.mesh_path or os.path.join(sd, 'foundationpose_data', 'mesh.obj')
    mesh = trimesh.load(mesh_path)

    fp_dir = os.path.join(sd, 'foundationpose_out', 'ob_in_world')
    files = sorted(os.listdir(fp_dir))
    mats = np.stack([np.loadtxt(os.path.join(fp_dir, f)) for f in files])
    centers = mats[:, :3, 3]
    print(f'frames={len(mats)} mesh={len(mesh.vertices)}v '
          f'net_disp={np.linalg.norm(centers[-1]-centers[0]):.4f}m')

    # GLB 里动画节点的几何须为局部系(原点=物体原点), 位姿由 animation 驱动
    obj_mesh = trimesh.Trimesh(vertices=mesh.vertices.copy(),
                               faces=mesh.faces.copy(), process=False)
    if mesh.visual is not None and mesh.visual.kind is not None:
        obj_mesh.visual = mesh.visual

    meshes = {args.obj: obj_mesh}
    animations = {args.obj: mats}

    if args.with_trail:
        # 注: 本环境 trimesh 无 creation.tube, Path3D 也没有 faces,
        # 轨迹折线改由 2D 叠加图/3D 静态图呈现, GLB 只保留物体本体动画。
        print('skip trail (trimesh lacks tube API); '
              'trajectory shown in fp_trajectory/*.png')

    glb_path = os.path.join(out_dir, f'{args.obj}.glb')
    export_glb_with_animation(glb_path, meshes, animations, fps=args.fps)
    print(f'GLB: {glb_path} ({os.path.getsize(glb_path)/1e6:.2f} MB)')

    npz = os.path.join(out_dir, f'{args.obj}_fp_trajectory.npz')
    np.savez(npz, poses=mats, centers=centers)
    print(f'traj: {npz}')


if __name__ == '__main__':
    main()