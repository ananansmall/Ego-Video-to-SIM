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


def load_instance(sd, obj):
    """读去重后的实例: 返回 (instance_info, 真实尺度 mesh, pre_masks, world_points)

    instance_info 来自 all_instances.pkl (mainv3 去重**后**的结果);
    mesh = original_mesh 已 apply_transform(T) —— original_mesh 是归一化尺度
    (约 1m 见方), 真实缩放在 T 的旋转块里(实测 toy_inst1 scale=0.1155,
    999mm -> 83.5mm), 不施加 T 会让 mesh 大 10 倍、IoU 崩到 0.01。
    pre_masks / world_points 来自 _pre_dedup_masks.pkl(去重**前**的逐帧像素)。
    """
    cat, inst_s = obj.rsplit('_inst', 1)
    inst = int(inst_s)
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        pkl = pickle.load(f)
    with open(os.path.join(sd, 'all_instances.pkl'), 'rb') as f:
        inst_pkl = pickle.load(f)
    inst_obj = inst_pkl['all_instances'][cat][inst]
    mesh = inst_obj['original_mesh'].copy()
    mesh.apply_transform(np.asarray(inst_obj['T']))
    return inst_obj, mesh, pkl['all_masks'], np.asarray(pkl['world_points'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str,
                        default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    parser.add_argument('--obj', type=str, default='toy_inst1')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--pre_inst', type=int, default=None,
                        help='显式指定 _pre_dedup mask 的实例号; '
                             '不给则读 fp_mask_select/<obj>.json 的自动选择结果')
    args = parser.parse_args()

    sd = args.scene_dir
    cat = args.obj.rsplit('_inst', 1)[0]
    out_dir = args.out_dir or os.path.join(sd, 'foundationpose_data')

    # 实例(去重后)+ mesh(真实尺度) + 去重前的 mask/world_points
    inst_obj, ref_mesh, pre_masks, world_points = load_instance(sd, args.obj)
    n_frames = world_points.shape[0]
    T_ref = np.asarray(inst_obj['T'])
    print(f'{args.obj}: mesh {len(ref_mesh.vertices)} verts, '
          f'extents={np.round(ref_mesh.extents * 1000, 1)}mm')

    # mask 在去重**前**的 _pre_dedup_masks.pkl 里(逐帧像素), 与去重后的编号
    # 完全不对应。几何启发式(质心距离/投影面积)都不可靠 —— T 是 ICP mesh 中心
    # 而非 mask 质心(实测差 20~150mm), 且物体朝向不同会让面积差 2~3 倍。
    # 唯一可靠判据是 register 后的投影 IoU, 由 fp_select_mask.py 给出。
    pre_inst = args.pre_inst
    if pre_inst is None:
        sel = os.path.join(sd, 'fp_mask_select', f'{args.obj}.json')
        if os.path.isfile(sel):
            import json
            with open(sel) as f:
                pre_inst = json.load(f)['best_pre_inst']
        else:
            raise SystemExit(
                f'{args.obj}: no mask selected. Run fp_select_mask.py first, '
                f'or pass --pre_inst explicitly.')
    frames = pre_masks[cat][pre_inst]
    if not frames:
        raise SystemExit(f'{args.obj}: _pre_dedup {cat}_inst{pre_inst} is empty')
    print(f'  using _pre_dedup mask: {cat}_inst{pre_inst} ({len(frames)} frames)')

    for sub in ('rgb', 'depth', 'mask', 'cam_K'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # mesh = ref_mesh(上面已 apply_transform(T) 到真实尺度), 直接落盘。
    # original_mesh 是**归一化尺度**(约 1m 见方), 真实缩放在 T 的旋转块里
    # (实测 toy_inst1 scale=0.1155, 999mm → 83.5mm); 不施加 T 会让 mesh
    # 比真实物体大 10 倍, IoU 直接崩到 0.01。
    mesh_path = os.path.join(out_dir, 'mesh.obj')
    ref_mesh.export(mesh_path)
    print(f'mesh: {mesh_path} ({len(ref_mesh.vertices)} verts, '
          f'extents={np.round(ref_mesh.extents * 1000, 1)}mm, scale='
          f'{np.round(np.linalg.norm(T_ref[:3, :3], axis=0), 4)})')

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
