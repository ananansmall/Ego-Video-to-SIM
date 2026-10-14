"""离线验证「静止段锁定位姿」能否消除 FP track 抖动 (方案B)。

不重跑 FP, 直接在已有 ob_in_world 轨迹上模拟:
  1. 用 GT mask 世界系轨迹 (mask质心+depth反投影) 做运动判定:
     滑动窗口(win=5)净位移 > thr → 运动, 否则静止
  2. 静止段: 位姿钉在段内中值 (位置中值 + 四元数平均), 运动段: 保留原位姿
  3. 对比锁定前后的路径长度 / 净位移 / GT 真值

输出对比表, 验证方案B是否值得接入 mainv3。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from fp_gif_2d3d import gt_world_traj, load_pre_masks, dynamic_keys  # noqa: E402
from fp_iou_eval import rasterize_mesh  # noqa: E402
import cv2  # noqa: E402
import trimesh  # noqa: E402


def mask_centroids(data_dir, sd, n, sel):
    """每帧 GT mask 质心 (u,v) 或 NaN。"""
    cat = sel['key'].rsplit('_inst', 1)[0]
    am = load_pre_masks(sd)
    fb = {fr['frame_id']: fr for fr in am[cat][sel['best_pre_inst']]}
    uv = np.full((n, 2), np.nan)
    for t, fr in fb.items():
        m = np.asarray(fr['mask']) > 0
        ys, xs = m.nonzero()
        if len(xs) >= 5:
            uv[t] = (xs.mean(), ys.mean())
    return uv


def mask_correct(P, data_dir, sd, n, sel, mesh, max_shift=0.10):
    """mask 引导平移校正: 每帧把 mesh 投影质心对齐 GT mask 质心 (相机系 xy)。

    max_shift: 单帧最大校正量 (m), 防止坏 mask 拉飞。
    """
    ext_dir = os.path.join(sd, 'extrinsics')
    exts = sorted(os.listdir(ext_dir), key=lambda x: int(os.path.splitext(x)[0]))[:n]
    uv_gt = mask_centroids(data_dir, sd, n, sel)
    vs = np.asarray(mesh.vertices)
    if len(vs) > 800:
        vs = vs[np.random.choice(len(vs), 800, replace=False)]

    Q = P.copy()
    for t in range(n):
        if not np.isfinite(uv_gt[t]).all():
            continue
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        ext = np.loadtxt(os.path.join(ext_dir, exts[t]))
        oc = ext @ P[t]
        vc = (oc[:3, :3] @ vs.T).T + oc[:3, 3]
        z = float(np.median(vc[:, 2]))
        if z <= 0.05:
            continue
        uv = vc[:, :2] / vc[:, 2:3] @ np.diag([K[0, 0], K[1, 1]]) + \
            np.array([[K[0, 2], K[1, 2]]])
        d_uv = uv_gt[t] - uv.mean(0)
        dx = d_uv[0] * z / K[0, 0]
        dy = d_uv[1] * z / K[1, 1]
        d = np.array([dx, dy, 0.0])
        ln = np.linalg.norm(d)
        if ln > max_shift:
            d *= max_shift / ln
        Q[t][:3, 3] += oc[:3, :3].T @ d  # cam→world
    return Q


def mask_arr(m, shape):
    m = np.asarray(m) > 0
    if m.shape != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0])) > 0
    return m


def mean_iou(P, data_dir, sd, n, sel, mesh, stride=1):
    cat = sel['key'].rsplit('_inst', 1)[0]
    am = load_pre_masks(sd)
    fb = {fr['frame_id']: fr for fr in am[cat][sel['best_pre_inst']]}
    ext_dir = os.path.join(sd, 'extrinsics')
    exts = sorted(os.listdir(ext_dir), key=lambda x: int(os.path.splitext(x)[0]))[:n]
    ious = []
    for t, fr in fb.items():
        if t % stride:
            continue
        img = cv2.imread(os.path.join(data_dir, 'rgb', f'{t:06d}.png'))
        if img is None:
            continue
        H, W = img.shape[:2]
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        pm = rasterize_mesh(mesh, np.loadtxt(os.path.join(ext_dir, exts[t])) @ P[t],
                            K, H, W)
        gm = mask_arr(fr['mask'], (H, W))
        inter = np.logical_and(pm > 0, gm).sum()
        union = np.logical_or(pm > 0, gm).sum()
        if union:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def motion_mask(gts, win=5, thr=0.02):
    """GT 世界系轨迹滑动窗口净位移 → 布尔运动掩码。NaN 帧按静止处理。"""
    n = len(gts)
    moving = np.zeros(n, bool)
    for t in range(n):
        seg = gts[max(0, t - win):t + win + 1]
        seg = seg[np.isfinite(seg).all(1)]
        if len(seg) < 3:
            continue
        net = np.linalg.norm(seg[-1] - seg[0])
        if net > thr:
            moving[t] = True
    return moving


def avg_quat(Rs):
    """符号对齐的四元数平均。"""
    qs = Rotation.from_matrix(Rs).as_quat()
    ref = qs[0]
    qs *= np.sign(qs @ ref)[:, None]
    q = qs.mean(0)
    return Rotation.from_quat(q / np.linalg.norm(q)).as_matrix()


def lock_static(P, moving, anchor='first'):
    """静止段 → 锚点位姿。anchor=first 用段首(register 准), median 用段内中值。"""
    Q = P.copy()
    n = len(P)
    t = 0
    while t < n:
        if moving[t]:
            t += 1
            continue
        s = t
        while t < n and not moving[t]:
            t += 1
        if anchor == 'first':
            Q[s:t, :3, :3] = P[s, :3, :3]
            Q[s:t, :3, 3] = P[s, :3, 3]
        else:
            Q[s:t, :3, :3] = avg_quat(P[s:t, :3, :3])
            Q[s:t, :3, 3] = np.median(P[s:t, :3, 3], axis=0)
    return Q


def smooth_moving(P, moving, win=9, order=3):
    """运动段: 平移逐轴 savgol + 旋转四元数 savgol (离线零滞后, 段内)。

    抖动是高频随机游走, 窗口内多项式拟合可压平高频; 真实 pick-place 运动
    (低频大位移) 保留。段短于窗口时原样保留。
    """
    Q = P.copy()
    n = len(P)
    t = 0
    while t < n:
        if not moving[t]:
            t += 1
            continue
        s = t
        while t < n and moving[t]:
            t += 1
        seg = Q[s:t]
        if len(seg) < win:
            continue
        Q[s:t, :3, 3] = savgol_filter(seg[:, :3, 3], win, order, axis=0)
        qs = Rotation.from_matrix(seg[:, :3, :3]).as_quat()
        qs *= np.sign(qs @ qs[0])[:, None]
        qs = savgol_filter(qs, win, order, axis=0)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        Q[s:t, :3, :3] = Rotation.from_quat(qs).as_matrix()
    return Q


def path_len(C):
    return float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())


def net_disp(C):
    return float(np.linalg.norm(C[-1] - C[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--win', type=int, default=5)
    ap.add_argument('--thr', type=float, default=0.02)
    ap.add_argument('--no_correct', action='store_true',
                    help='关闭 mask 质心校正, 只做静止段锁定')
    ap.add_argument('--anchor', default='first', choices=['first', 'median'])
    ap.add_argument('--smooth_win', type=int, default=9,
                    help='运动段 savgol 窗口 (0=不平滑)')
    ap.add_argument('--smooth_order', type=int, default=3)
    ap.add_argument('--save_locked', action='store_true',
                    help='保存修复后位姿到 fp_out_<obj>/ob_in_world_locked/')
    ap.add_argument('--objs', default=None, help='逗号分隔, 覆盖动态过滤')
    ap.add_argument('--all', action='store_true',
                    help='不过滤, 全部实例 (默认只评 stage3 判定为动态的实例)')
    args = ap.parse_args()
    sd = args.scene_dir

    rep = json.load(open(os.path.join(sd, 'fp_batch_report.json')))
    keys = list(rep.keys())
    if args.objs:
        keys = [k for k in keys if k in set(args.objs.split(','))]
    elif not args.all:
        keys = dynamic_keys(sd, keys)
    print(f'instances: {keys}')
    print(f'{"instance":<14}{"FP原始 path/IoU":>20}{"校正+锁定 path/IoU":>22}'
          f'{"GT path/net":>16}{"运动帧":>8}')
    rows = {}
    for k in keys:
        out = os.path.join(sd, f'fp_out_{k}')
        wdir = os.path.join(out, 'ob_in_world')
        n = len(os.listdir(wdir))
        P = np.stack([np.loadtxt(os.path.join(wdir, f'{t:06d}.txt'))
                      for t in range(n)])
        data = os.path.join(sd, f'fp_data_{k}')
        sel = json.load(open(os.path.join(sd, 'fp_mask_select', f'{k}.json')))
        sel['key'] = k
        gts = gt_world_traj(data, sd, n, sel)
        mesh = trimesh.load(os.path.join(data, 'mesh.obj'))

        mv = motion_mask(gts, win=args.win, thr=args.thr)
        base = P if args.no_correct else mask_correct(P, data, sd, n, sel, mesh)
        Qc = lock_static(base, mv, anchor=args.anchor)
        if args.smooth_win:
            Qc = smooth_moving(Qc, mv, win=args.smooth_win, order=args.smooth_order)
        if args.save_locked:
            ldir = os.path.join(out, 'ob_in_world_locked')
            os.makedirs(ldir, exist_ok=True)
            for t in range(n):
                np.savetxt(os.path.join(ldir, f'{t:06d}.txt'), Qc[t], fmt='%.8f')

        C0, C1 = P[:, :3, 3], Qc[:, :3, 3]
        ok = gts[np.isfinite(gts).all(1)]
        gt_path = path_len(ok) if len(ok) > 1 else 0.0
        gt_net = net_disp(ok) if len(ok) > 1 else 0.0
        iou0 = json.load(open(os.path.join(out, 'iou_face_report.json')))['mean']
        iou1 = mean_iou(Qc, data, sd, n, sel, mesh, stride=2)
        rows[k] = {'n_moving': int(mv.sum()), 'n': n,
                   'fp_path': path_len(C0), 'fp_net': net_disp(C0), 'fp_iou': iou0,
                   'lock_path': path_len(C1), 'lock_net': net_disp(C1), 'lock_iou': iou1,
                   'gt_path': gt_path, 'gt_net': gt_net}
        print(f'{k:<14}{path_len(C0):>10.3f}/{iou0:.3f}'
              f'{path_len(C1):>13.3f}/{iou1:.3f}'
              f'{gt_path:>9.3f}/{gt_net:<6.3f}{int(mv.sum()):>5}/{n}')

    with open(os.path.join(sd, 'fp_static_lock_eval.json'), 'w') as f:
        json.dump(rows, f, indent=2)
    print('\nsaved fp_static_lock_eval.json')


if __name__ == '__main__':
    main()
