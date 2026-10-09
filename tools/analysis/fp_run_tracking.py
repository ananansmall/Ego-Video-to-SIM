"""
FoundationPose model-based 6DoF tracking (相机姿态变化的 egocentric 视频)。

帧 0 用 RGB+depth+mask+mesh 注册 (render-and-compare 自行搜索初始位姿),
后续帧 track_one 逐帧跟踪, 每帧传各自的 VGGT 内参/深度。

输出 (<scene_dir>/foundationpose_out/):
  ob_in_cam/{t:06d}.txt   相机系物体位姿 4x4 (FoundationPose 原始输出)
  ob_in_world/{t:06d}.txt 世界系位姿 4x4 (经 extrinsics cam2world 转换, Z-up)
  tracking_report.json    逐帧验证: 投影 mesh vs SAM3 mask 的 IoU,
                          并与旧 ICP 轨迹 (object_trajectories) 对比

用法:
  python fp_run_tracking.py [--obj toy_inst1] [--gpu 0]
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

FP_ROOT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/FoundationPose'
sys.path.insert(0, FP_ROOT)
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

from trajectory_video_overlay import Projector, sample_object_surface_points  # noqa: E402


def mesh_to_mask(verts_cam, K, H, W, max_verts=20000):
    """投影 mesh 顶点 → 二值 mask (膨胀 3x3 模拟面片覆盖)。"""
    import trimesh
    idx = np.linspace(0, len(verts_cam) - 1, min(max_verts, len(verts_cam))).astype(int)
    v = verts_cam[idx]
    z = v[:, 2]
    ok = z > 1e-6
    uv = (v[ok, :2] / z[ok, None]) @ K[:2, :2].T + K[:2, 2]
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    m = np.zeros((H, W), np.uint8)
    m[uv[inb, 1].astype(int), uv[inb, 0].astype(int)] = 1
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return m > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str,
                        default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    parser.add_argument('--obj', type=str, default='toy_inst1')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sd = args.scene_dir
    data_dir = args.data_dir or os.path.join(sd, 'foundationpose_data')
    out_dir = args.out_dir or os.path.join(sd, 'foundationpose_out')
    for sub in ('ob_in_cam', 'ob_in_world', 'track_vis'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    n_frames = len(os.listdir(os.path.join(data_dir, 'rgb')))
    print(f'frames: {n_frames}')

    # ---- FoundationPose 初始化 ----
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

    mesh = trimesh.load(os.path.join(data_dir, 'mesh.obj'))
    print(f'mesh: {len(mesh.vertices)} verts')
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices.astype(np.float32),
                         model_normals=mesh.vertex_normals.astype(np.float32),
                         mesh=mesh, scorer=ScorePredictor(),
                         refiner=PoseRefinePredictor(),
                         debug_dir=out_dir, debug=0, glctx=glctx)
    print('estimator ready')

    # ---- 逐帧跟踪 ----
    poses_cam = {}
    for t in range(n_frames):
        rgb = cv2.imread(os.path.join(data_dir, 'rgb', f'{t:06d}.png'))[..., ::-1]
        depth = cv2.imread(os.path.join(data_dir, 'depth', f'{t:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e3
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        if t == 0:
            mask = cv2.imread(os.path.join(data_dir, 'mask', '000000.png'), 0)
            mask = (mask > 0).astype(bool)
            pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                iteration=args.est_refine_iter)
        else:
            pose = est.track_one(rgb=rgb, depth=depth, K=K,
                                 iteration=args.track_refine_iter)
        poses_cam[t] = np.asarray(pose)
        np.savetxt(os.path.join(out_dir, 'ob_in_cam', f'{t:06d}.txt'),
                   pose.reshape(4, 4))
        print(f'  [{t}] pose t={np.round(pose[:3, 3], 3)}')

    # ---- 相机系 → 世界系 (extrinsics 是 world-to-camera) ----
    for t in range(n_frames):
        ext = np.loadtxt(os.path.join(sd, 'extrinsics', f'{t}.txt'))
        T_c2w = np.linalg.inv(ext)
        np.savetxt(os.path.join(out_dir, 'ob_in_world', f'{t:06d}.txt'),
                   T_c2w @ poses_cam[t])

    # ---- 验证: 投影 mesh vs SAM3 mask IoU (对比旧 ICP 轨迹) ----
    print('validating IoU vs SAM3 masks ...')
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        pkl = pickle.load(f)
    all_masks = pkl['all_masks']
    cat, inst_s = args.obj.rsplit('_inst', 1)
    inst = int(inst_s)
    frames = all_masks[cat][inst]
    frame_by_id = {fr['frame_id']: fr for fr in frames}
    verts = np.asarray(mesh.vertices)

    report = {'obj': args.obj, 'frames': {}}
    for t in range(n_frames):
        fr = frame_by_id.get(t)
        if fr is None:
            continue
        m = np.asarray(fr['mask']) > 0
        H, W = m.shape
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        vc = (verts @ poses_cam[t][:3, :3].T + poses_cam[t][:3, 3])
        pm = mesh_to_mask(vc, K, H, W)
        inter = (pm & m).sum()
        union = (pm | m).sum()
        iou = float(inter / union) if union else 0.0
        report['frames'][str(t)] = {'iou_fp': iou}
    ious = [v['iou_fp'] for v in report['frames'].values()]
    report['iou_fp_mean'] = float(np.mean(ious))
    report['iou_fp_median'] = float(np.median(ious))
    report['iou_fp_min'] = float(np.min(ious))

    with open(os.path.join(out_dir, 'tracking_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"IoU (FoundationPose): mean={report['iou_fp_mean']:.3f} "
          f"median={report['iou_fp_median']:.3f} min={report['iou_fp_min']:.3f}")
    print(f'out: {out_dir}')


if __name__ == '__main__':
    main()
