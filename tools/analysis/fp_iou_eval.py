"""用面片光栅化(而非顶点投影)评估 FoundationPose 位姿与 SAM3 mask 的一致性。

顶点投影 + 3x3 膨胀会系统性高估投影面积, 把 IoU 压低; 面片光栅化才是准确口径。
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from fp_run_tracking import mesh_to_mask  # noqa: E402,F401  (顶点投影版, 供对照)


def rasterize_mesh(mesh, pose, K, H, W):
    """把 mesh 按 pose 投影, 用三角形填充得到面片级 mask。"""
    P = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
    m = np.zeros((H, W), np.uint8)
    uv = np.full((len(P), 2), np.nan)
    z = P[:, 2]
    ok = z > 1e-6
    uv[ok, 0] = P[ok, 0] / z[ok] * K[0, 0] + K[0, 2]
    uv[ok, 1] = P[ok, 1] / z[ok] * K[1, 1] + K[1, 2]
    ok_tri = ok[mesh.faces].all(axis=1)
    faces = mesh.faces[ok_tri]
    tri_uv = np.round(uv[faces]).astype(np.int32)          # (M,3,2)
    # 三角形只需有一个顶点落在画面内就参与绘制(部分出画的三角形仍会覆盖边界像素)
    inb = ((tri_uv[..., 0] >= 0) & (tri_uv[..., 0] < W) &
           (tri_uv[..., 1] >= 0) & (tri_uv[..., 1] < H)).any(axis=1)
    for tri in tri_uv[inb]:
        cv2.fillConvexPoly(m, np.ascontiguousarray(tri), 1)
    return m > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    ap.add_argument('--obj', default='toy_inst1')
    ap.add_argument('--n_check', type=int, default=0, help='0=全部帧')
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--pre_inst', type=int, default=None,
                    help='GT mask 用哪个 _pre_dedup 实例; 默认读 fp_mask_select/<obj>.json')
    args = ap.parse_args()
    sd = args.scene_dir
    data_dir = args.data_dir or os.path.join(sd, 'foundationpose_data')
    fp_out = args.out_dir or os.path.join(sd, 'foundationpose_out')

    mesh = trimesh.load(os.path.join(data_dir, 'mesh.obj'))
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        all_masks = pickle.load(f)['all_masks']
    cat = args.obj.rsplit('_inst', 1)[0]
    # GT mask 必须与 prepare 时用的是同一个 _pre_dedup 实例, 否则评测的是错的物体。
    # 默认从 fp_mask_select/<obj>.json 读; 没有就退回同名(仅当显式 --pre_inst 一致时正确)。
    pre_inst = args.pre_inst
    if pre_inst is None:
        sel = os.path.join(sd, 'fp_mask_select', f'{args.obj}.json')
        if os.path.isfile(sel):
            import json
            with open(sel) as f:
                pre_inst = json.load(f)['best_pre_inst']
        else:
            pre_inst = int(args.obj.rsplit('_inst', 1)[1])
    fb = {fr['frame_id']: fr for fr in all_masks[cat][pre_inst]}
    if not fb:
        raise SystemExit(f'{args.obj}: _pre_dedup {cat}_inst{pre_inst} is empty')
    print(f'GT mask: _pre_dedup {cat}_inst{pre_inst} ({len(fb)} frames)')
    ts = sorted(fb)
    if args.n_check:
        ts = ts[::max(1, len(ts) // args.n_check)]

    res = {}
    for t in ts:
        gt = np.asarray(fb[t]['mask']) > 0
        H, W = gt.shape
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        P = np.loadtxt(os.path.join(fp_out, 'ob_in_cam', f'{t:06d}.txt'))
        pm = rasterize_mesh(mesh, P, K, H, W)
        res[t] = float((pm & gt).sum() / max((pm | gt).sum(), 1))

    print(f'面片光栅 IoU (共 {len(res)} 帧): mean={np.mean(list(res.values())):.3f} '
          f'median={np.median(list(res.values())):.3f} min={np.min(list(res.values())):.3f}')

    # 分段: 静止/运动 的分界由**世界系**逐帧位移自动判定, 不能硬编码帧号——
    # 每个实例的运动时段不同(实测 toy_inst1 从帧 51 起动, 不是 60)。
    # 必须用 ob_in_world: ob_in_cam 里相机本身在动, 物体相对相机的位移
    # 每帧都>1cm, 会把整段视频都判成"运动中"(实测静止段被误判)。
    ts_sorted = sorted(res)
    n_pose = len(os.listdir(os.path.join(fp_out, 'ob_in_cam')))
    wdir = os.path.join(fp_out, 'ob_in_world')
    if os.path.isdir(wdir):
        wposes = np.stack([np.loadtxt(os.path.join(wdir, f'{t:06d}.txt'))
                           for t in range(len(os.listdir(wdir)))])
    else:
        wposes = np.stack([np.loadtxt(os.path.join(fp_out, 'ob_in_cam', f'{t:06d}.txt'))
                           for t in range(n_pose)])
    step = np.linalg.norm(np.diff(wposes[:, :3, 3], axis=0), axis=1)
    # 用**滑动窗口净位移**而非逐帧 step: FP 在静止段本身就有 ~4cm/帧 的抖动
    # (实测 toy_inst1 逐帧 step 中位数 40mm, 而物体实际静止在桌上),
    # 逐帧阈值会把整段视频都判成"运动中"。窗口净位移对抖动不敏感。
    win = 5
    if len(wposes) > win:
        net = np.linalg.norm(wposes[win:, :3, 3] - wposes[:-win, :3, 3], axis=1)
        t_move = int(np.argmax(net > 0.03)) + 1 if (net > 0.03).any() else len(wposes)
        print(f'  自动判定: 运动起始帧 = {t_move} (窗口{win} 净位移>3cm; '
              f'逐帧 step 中位数 {np.median(step) * 1000:.0f}mm 为抖动, 已剔除)')
    else:
        t_move = 0
    for label, a, b in (('静止段', 0, t_move - 1), ('运动段', t_move, len(wposes) - 1)):
        v = [res[t] for t in ts_sorted if a <= t <= b]
        if v:
            print(f'  {label}[{a}-{b}]: mean={np.mean(v):.3f} min={np.min(v):.3f}')

    print('逐帧(每10帧):', ' '.join(f'{t}:{res[t]:.2f}' for t in ts_sorted if t % 10 == 0))

    with open(os.path.join(fp_out, 'iou_face_report.json'), 'w') as f:
        json.dump({'per_frame': {str(k): v for k, v in res.items()},
                   'move_start_frame': t_move,
                   'mean': float(np.mean(list(res.values())))}, f, indent=2)


if __name__ == '__main__':
    main()