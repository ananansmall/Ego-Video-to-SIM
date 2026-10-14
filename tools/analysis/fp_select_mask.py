"""为一个去重后的实例挑选正确的 _pre_dedup mask: 对每个候选 mask 跑一次
FoundationPose register, 用 register 后的投影 IoU 选最高分的组合。

为什么必须这么做(踩过的坑):
  - 去重后的 all_instances.pkl 与去重前的 _pre_dedup_masks.pkl **编号完全不对应**,
    实例号↔ urdf link 号更是毫无关系。
  - 几何启发式都不可靠: 首帧质心距离天然差 20~150mm(T 是 ICP mesh 中心而非
    mask 质心, 见 export_dynamic_glb 文档); 投影面积也会因物体朝向不同而差 2~3 倍
    (实测 block_inst0 与 block_inst1 的 xy 估计完全相同, 纯面积无法区分)。
  - 唯一可靠判据是 register 后的实际投影 IoU。

用法:
    python fp_select_mask.py --scene_dir <sd> --obj toy_inst1
输出:
    <sd>/fp_mask_select/<obj>.json   {best_pre_inst, best_iou, candidates:[...]}
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
from fp_prepare_data import load_instance  # noqa: E402

FP_ROOT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/FoundationPose'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--obj', required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--min_iou', type=float, default=0.35,
                    help='低于此 IoU 认为该实例无可用 mask(物体不存在/被遮挡)')
    args = ap.parse_args()

    sd = args.scene_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    inst_obj, mesh, _pre_masks, _wp = load_instance(sd, args.obj)
    cat = args.obj.rsplit('_inst', 1)[0]
    print(f'{args.obj}: mesh {len(mesh.vertices)} verts, '
          f'extents={np.round(mesh.extents * 1000, 1)}mm')

    # 候选 mask: 同类别下所有首帧面积 >=20px 的实例
    cands = []
    for j, fr in enumerate(_pre_masks[cat]):
        if not fr:
            continue
        fb = {x['frame_id']: x for x in fr}
        t0 = min(fb)
        gt = np.asarray(fb[t0]['mask']) > 0
        if gt.sum() < 20:
            continue
        cands.append((j, t0, gt))
    if not cands:
        raise SystemExit(f'{args.obj}: no candidate mask')
    print(f'  candidate masks: {[c[0] for c in cands]}')

    # 帧0 的 rgb/depth/K(与 mask 无关, 读一次)
    rgb0 = np.ascontiguousarray(
        cv2.imread(os.path.join(sd, 'color', '0.jpg'))[..., ::-1])
    ext0 = np.loadtxt(os.path.join(sd, 'extrinsics', '0.txt'))
    wpts = np.asarray(_wp[0])
    cam0 = wpts @ ext0[:3, :3].T + ext0[:3, 3]
    z0 = cam0[..., 2]
    z0[~np.isfinite(z0)] = 0
    z0[z0 <= 1e-6] = 0
    depth0 = (z0 * 1000).astype(np.uint16)
    K0 = np.loadtxt(os.path.join(sd, 'fp_data_shared', 'cam_K', '000000.txt')) \
        if os.path.isfile(os.path.join(sd, 'fp_data_shared', 'cam_K', '000000.txt')) \
        else _recover_K(sd, wpts, ext0, rgb0.shape[:2])

    sys.path.insert(0, FP_ROOT)
    sys.path.insert(0, os.path.join(FP_ROOT, 'mycpp', 'build'))
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

    dbg = os.path.join(sd, 'fp_mask_select', '_debug')
    os.makedirs(dbg, exist_ok=True)
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices.astype(np.float32),
                         model_normals=mesh.vertex_normals.astype(np.float32),
                         mesh=mesh, scorer=ScorePredictor(),
                         refiner=PoseRefinePredictor(),
                         debug_dir=dbg, debug=0, glctx=glctx)

    results = []
    for j, t0, gt in cands:
        H, W = gt.shape
        ob_mask = gt
        pose = np.asarray(est.register(K=K0, rgb=rgb0,
                                       depth=depth0.astype(np.float32) / 1e3,
                                       ob_mask=ob_mask, iteration=5))
        pm = rasterize_mesh(mesh, pose, K0, H, W)
        iou = float((pm & gt).sum() / max((pm | gt).sum(), 1))
        results.append({'pre_inst': j, 'iou': iou,
                        'proj_area': int(pm.sum()), 'mask_area': int(gt.sum())})
        print(f'  pre_{cat}_inst{j}: register IoU={iou:.3f} '
              f'(proj={pm.sum()} mask={gt.sum()})')

    results.sort(key=lambda r: -r['iou'])
    out = {'obj': args.obj, 'best_pre_inst': results[0]['pre_inst'],
           'best_iou': results[0]['iou'], 'candidates': results}
    od = os.path.join(sd, 'fp_mask_select')
    os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f'{args.obj}.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    ok = results[0]['iou'] >= args.min_iou
    print(f'\nbest: pre_{cat}_inst{results[0]["pre_inst"]} '
          f'IoU={results[0]["iou"]:.3f} -> {"USABLE" if ok else "REJECT"}')
    print(f'-> {p}')
    return 0 if ok else 1


def _recover_K(sd, wpts, ext, shape):
    from trajectory_video_overlay import recover_intrinsic
    H, W = shape
    return recover_intrinsic(wpts, ext[:3, :3], ext[:3, 3],
                             np.array([[286.0, 0, W / 2], [0, 286.0, H / 2],
                                       [0, 0, 1.0]]))


if __name__ == '__main__':
    sys.exit(main())