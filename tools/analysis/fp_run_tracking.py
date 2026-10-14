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
sys.path.insert(0, os.path.join(FP_ROOT, 'mycpp', 'build'))  # estimater 需要 mycpp 扩展
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
    parser.add_argument('--mesh', type=str, default=None,
                        help='覆盖 mesh.obj 路径(批量筛选候选 mesh 时用)')
    parser.add_argument('--n_frames', type=int, default=0,
                        help='>0 只跟踪前 N 帧(快速筛选/冒烟测试)')
    parser.add_argument('--pre_inst', type=int, default=None,
                        help='GT mask 用哪个 _pre_dedup 实例; 默认读 fp_mask_select/<obj>.json')
    parser.add_argument('--rekey', type=int, default=0,
                        help='>0 每隔 N 帧用当帧 SAM3 mask 重新 register '
                             '(mask 引导, 抗漂移; register 失败回退 track)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sd = args.scene_dir
    data_dir = args.data_dir or os.path.join(sd, 'foundationpose_data')
    out_dir = args.out_dir or os.path.join(sd, 'foundationpose_out')
    for sub in ('ob_in_cam', 'ob_in_world', 'track_vis'):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    n_frames = len(os.listdir(os.path.join(data_dir, 'rgb')))
    if args.n_frames:
        n_frames = min(n_frames, args.n_frames)
    print(f'frames: {n_frames}')

    # ---- FoundationPose 初始化 ----
    import trimesh
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

    mesh = trimesh.load(args.mesh or os.path.join(data_dir, 'mesh.obj'))
    print(f'mesh: {args.mesh or os.path.join(data_dir, "mesh.obj")} '
          f'({len(mesh.vertices)} verts)')
    glctx = dr.RasterizeCudaContext()
    # 注: 曾试过传 Z 轴 180° 对称变换, 但实测并未减少等价解翻转
    # (旋转跳变 9 次 -> 11 次), 且已验证逐帧真实旋转仅 4.8°/帧, 数据本身健康。
    # 跳变只出现在 rotvec/euler 表示层, 故不需要 symmetry_tfs。
    est = FoundationPose(model_pts=mesh.vertices.astype(np.float32),
                         model_normals=mesh.vertex_normals.astype(np.float32),
                         mesh=mesh, scorer=ScorePredictor(),
                         refiner=PoseRefinePredictor(),
                         debug_dir=out_dir, debug=0, glctx=glctx)
    print('estimator ready')

    # ---- 逐帧跟踪 ----
    poses_cam = {}
    for t in range(n_frames):
        rgb = np.ascontiguousarray(cv2.imread(os.path.join(data_dir, 'rgb', f'{t:06d}.png'))[..., ::-1])
        depth = cv2.imread(os.path.join(data_dir, 'depth', f'{t:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e3
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        mask_path = os.path.join(data_dir, 'mask', f'{t:06d}.png')
        # mask 引导: 帧 0 必须 register; 之后每隔 rekey 帧且当帧有 SAM3 mask 时
        # 重新 register, 把漂移拉回真值物体 (纯 track 是 render-and-compare,
        # 无 mask 约束, 会锁到相似物体上)
        do_reg = t == 0
        if not do_reg and args.rekey and t % args.rekey == 0:
            do_reg = os.path.isfile(mask_path)
        if do_reg:
            mask = cv2.imread(mask_path, 0)
            mask = (mask > 0).astype(bool)
            try:
                pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask,
                                    iteration=args.est_refine_iter)
            except Exception as e:
                print(f'  [{t}] register failed ({e}), fallback to track')
                pose = est.track_one(rgb=rgb, depth=depth, K=K,
                                     iteration=args.track_refine_iter)
        else:
            pose = est.track_one(rgb=rgb, depth=depth, K=K,
                                 iteration=args.track_refine_iter)
        poses_cam[t] = np.asarray(pose)
        np.savetxt(os.path.join(out_dir, 'ob_in_cam', f'{t:06d}.txt'),
                   pose.reshape(4, 4))
        print(f'  [{t}] {"reg" if do_reg else "trk"} '
              f'pose t={np.round(pose[:3, 3], 3)}')

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
    cat = args.obj.rsplit('_inst', 1)[0]
    # GT mask 必须与 prepare 用的是同一个 _pre_dedup 实例(否则校验的是错的物体)
    pre_inst = args.pre_inst
    if pre_inst is None:
        sel = os.path.join(sd, 'fp_mask_select', f'{args.obj}.json')
        if os.path.isfile(sel):
            with open(sel) as f:
                pre_inst = json.load(f)['best_pre_inst']
        else:
            pre_inst = int(args.obj.rsplit('_inst', 1)[1])
    frames = all_masks[cat][pre_inst]
    frame_by_id = {fr['frame_id']: fr for fr in frames}
    print(f'GT mask: _pre_dedup {cat}_inst{pre_inst} ({len(frames)} frames)')
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
