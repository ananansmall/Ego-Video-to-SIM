"""为某个 mask 实例挑选正确的 mesh: 对每个候选 mesh 跑 FP register(帧0),
用 register 后的投影 IoU 选最高分的。

为什么需要: stage3 去重后 mask 实例号与 urdf link 号完全不同
(实测 7_vggt_omega 里 toy_inst0..6 对应的 urdf 编号是乱的),
按投影面积粗筛会选错 —— plate_inst0 唯一 mesh 也差7.5 倍误差。
register 后的实际投影 IoU 才是可靠判据。

用法:
    python fp_pick_mesh.py --scene_dir <sd> --obj toy_inst5
输出:
    <sd>/fp_mesh_pick/<obj>.json   {mesh_path, iou, candidates:[...]}
"""
import argparse
import glob
import json
import os
import pickle
import sys

os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from fp_iou_eval import rasterize_mesh  # noqa: E402

FP_ROOT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/FoundationPose'


def candidate_meshes(sd, cat):
    c = sorted(glob.glob(os.path.join(sd, 'urdf', f'{cat}_*', 'mesh', f'{cat}_*.obj')))
    return [p for p in c if not p.endswith('_collision.obj')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--obj', required=True)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--mesh', default=None,
                    help='只试这一个 mesh(批量脚本按候选逐个调用, 避免同进程 glctx 污染)')
    args = ap.parse_args()

    sd = args.scene_dir
    cat = args.obj.rsplit('_inst', 1)[0]
    data_dir = args.out_dir or os.path.join(sd, 'foundationpose_data')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        am = pickle.load(f)['all_masks']
    inst = int(args.obj.rsplit('_inst', 1)[1])
    fb = {fr['frame_id']: fr for fr in am[cat][inst]}
    t0 = min(fb)
    gt = np.asarray(fb[t0]['mask']) > 0
    H, W = gt.shape

    # 复用已准备好的帧0 rgb/depth/K(不依赖具体 mesh)
    rgb = np.ascontiguousarray(cv2.imread(os.path.join(data_dir, 'rgb', f'{t0:06d}.png'))[..., ::-1])
    depth = cv2.imread(os.path.join(data_dir, 'depth', f'{t0:06d}.txt'.replace('.txt', '.png')),
                       cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e3
    K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t0:06d}.txt'))
    mask = cv2.imread(os.path.join(data_dir, 'mask', f'{t0:06d}.png'), 0) > 0
    if mask.sum() < 20:
        raise SystemExit(f'{args.obj}: mask at frame {t0} empty ({mask.sum()}px) -> FP skip')

    sys.path.insert(0, FP_ROOT)
    sys.path.insert(0, os.path.join(FP_ROOT, 'mycpp', 'build'))
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

    glctx = dr.RasterizeCudaContext()
    cands = candidate_meshes(sd, cat)
    if args.mesh:
        cands = [args.mesh]
    if not cands:
        raise SystemExit(f'{args.obj}: no candidate mesh for category {cat}')

    # debug_dir 必须显式给: FoundationPose 默认用 os.path.expanduser('~'),
    # 该 conda 环境的 HOME (/home/bowen) 不可写 -> PermissionError
    dbg = os.path.join(sd, 'fp_mesh_pick', '_debug')
    os.makedirs(dbg, exist_ok=True)

    results = []
    for mp in cands:
        mesh = trimesh.load(mp)
        est = FoundationPose(model_pts=mesh.vertices.astype(np.float32),
                             model_normals=mesh.vertex_normals.astype(np.float32),
                             mesh=mesh, scorer=ScorePredictor(),
                             refiner=PoseRefinePredictor(),
                             debug_dir=dbg, debug=0, glctx=glctx)
        pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
        pose = np.asarray(pose)
        pm = rasterize_mesh(mesh, pose, K, H, W)
        inter = (pm & gt).sum()
        iou = float(inter / max((pm | gt).sum(), 1))
        iom = float(inter / max(gt.sum(), 1))
        results.append({'mesh': mp, 'iou': iou, 'iom': iom,
                        'proj_area': int(pm.sum()), 'mask_area': int(gt.sum())})
        print(f'  {os.path.basename(mp):16s} IoU={iou:.3f} IoM={iom:.3f} '
              f'proj={pm.sum():5d} mask={gt.sum():5d}')
        del est

    results.sort(key=lambda r: -r['iou'])
    out = {'obj': args.obj, 'register_frame': int(t0),
           'best_mesh': results[0]['mesh'], 'best_iou': results[0]['iou'],
           'candidates': results}
    od = os.path.join(sd, 'fp_mesh_pick')
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f'{args.obj}.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nbest: {os.path.basename(results[0]["mesh"])} IoU={results[0]["iou"]:.3f}')
    print(f'-> {p}')


if __name__ == '__main__':
    main()