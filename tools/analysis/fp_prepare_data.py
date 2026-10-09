"""
为 FoundationPose 准备输入数据 (model-based tracking)。

从场景目录提取:
  fp_data/rgb/    逐帧 RGB (000000.png ...)
  fp_data/depth/  逐帧深度 (16bit mm) — 由 VGGT world_points 投影到相机系取 z,
                  与重建同源, 避免 RGB-D 缺失
  fp_data/mask/   逐帧 SAM3 mask (帧间可能缺失, 缺帧不落盘)
  fp_data/cam_K/  逐帧内参 (recover_intrinsic, 与 vggt-omega 反投影同口径)
  fp_data/mesh.obj 物体 URDF mesh

用法:
  python fp_prepare_data.py [--obj toy_inst1]
"""
import argparse
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from trajectory_video_overlay import recover_intrinsic, TRAJ_DIR_SUFFIX  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str,
                        default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    parser.add_argument('--obj', type=str, default='toy_inst1')
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    sd = args.scene_dir
    cat, inst_s = args.obj.rsplit('_inst', 1)
    inst = int(inst_s)
    out_dir = args.out_dir or os.path.join(sd, 'foundationpose_data')

    print('loading _pre_dedup_masks.pkl ...')
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        pkl = pickle.load(f)
    world_points = np.asarray(pkl['world_points'], dtype=np.float64)
    all_masks = pkl['all_masks']
    frames = all_masks[cat][inst]
    n_frames = world_points.shape[0]
    print(f'world_points: {world_points.shape}, {args.obj}: {len(frames)} frames')

    for sub in ('rgb', 'depth', 'mask', 'cam_K'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # mesh — 注意 mask 实例编号 ≠ URDF link 编号 (stage3 去重后重编号),
    # 从 dynamic_glb/*_trajectory.npz 里读 source_trajectory → urdf_link 映射
    import glob
    mesh_src = None
    for npz_path in glob.glob(os.path.join(sd, 'dynamic_glb', '*_trajectory.npz')):
        g = np.load(npz_path, allow_pickle=True)
        if str(g['source_trajectory']) == args.obj:
            link = str(g['urdf_link'])
            mesh_src = os.path.join(sd, 'urdf', link, 'mesh', f'{link}.obj')
            break
    if mesh_src is None:
        mesh_src = os.path.join(sd, 'urdf', f'{cat}_{inst}', 'mesh', f'{cat}_{inst}.obj')
        print(f'WARNING: no dynamic_glb mapping for {args.obj}, fallback {mesh_src}')
    if not os.path.isfile(mesh_src):
        raise FileNotFoundError(mesh_src)
    import shutil
    shutil.copy(mesh_src, os.path.join(out_dir, 'mesh.obj'))
    print(f'mesh: {mesh_src}')

    frame_by_id = {fr['frame_id']: fr for fr in frames}

    for t in range(n_frames):
        # rgb
        rgb_src = os.path.join(sd, 'color', f'{t}.jpg')
        img = cv2.imread(rgb_src)
        if img is None:
            print(f'  skip frame {t}: no rgb')
            continue
        cv2.imwrite(os.path.join(out_dir, 'rgb', f'{t:06d}.png'), img)
        H, W = img.shape[:2]

        # 逐帧外参 (world-to-camera)
        ext = np.loadtxt(os.path.join(sd, 'extrinsics', f'{t}.txt'))
        R, tr = ext[:3, :3], ext[:3, 3]

        # 逐帧内参 (world_points + 像素网格反推)
        K = recover_intrinsic(world_points[t], R, tr,
                              np.array([[286.0, 0, W / 2], [0, 286.0, H / 2],
                                        [0, 0, 1.0]]))
        np.savetxt(os.path.join(out_dir, 'cam_K', f'{t:06d}.txt'), K)

        # 深度: world_points 投到相机系取 z (米 → 16bit mm)
        cam = world_points[t] @ R.T + tr
        z = cam[..., 2]
        z[~np.isfinite(z)] = 0
        z[z <= 1e-6] = 0
        cv2.imwrite(os.path.join(out_dir, 'depth', f'{t:06d}.png'),
                    (z * 1000).astype(np.uint16))

        # mask (可能缺帧)
        fr = frame_by_id.get(t)
        if fr is not None:
            m = (np.asarray(fr['mask']) > 0).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(out_dir, 'mask', f'{t:06d}.png'), m)

    print(f'done. rgb={len(os.listdir(os.path.join(out_dir, "rgb")))} '
          f'mask={len(os.listdir(os.path.join(out_dir, "mask")))} '
          f'depth={len(os.listdir(os.path.join(out_dir, "depth")))} '
          f'cam_K={len(os.listdir(os.path.join(out_dir, "cam_K")))}')
    print(f'out: {out_dir}')


if __name__ == '__main__':
    main()
