"""生成 2D+3D 合一 GIF (combo.gif)。

左半 = 2D 视频叠加: 绿轮廓=FP mesh 投影, 红轮廓=SAM3 GT mask, 橙点=历史尾迹
右半 = 3D 世界系(Z-up): mesh 点云按 ob_in_world 变换 + 相机轨迹(灰) + FP 轨迹(橙)
        + GT 轨迹(绿, mask质心+depth反投影), 视角/范围固定不抖动。

输出 <scene_dir>/fp_vis/<obj>/combo.gif, 并同步 test/fp_vis/<obj>/。
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
from fp_iou_eval import rasterize_mesh  # noqa: E402

TEST_FP_VIS = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/test/fp_vis'


def dynamic_keys(sd, keys):
    """按主流程 stage3 判定 (pick_place_frames_combined.json) 过滤出动态实例。"""
    p = os.path.join(sd, 'pick_place_frames_combined.json')
    if not os.path.isfile(p):
        return keys
    names = {it['instance_name'] for it in
             json.load(open(p)).get('interactions', [])}
    return [k for k in keys if k in names]


def contour(img, m, color, th=2, fill=0.0):
    """轮廓 + 可选半透明填充 (fill=alpha)。GT mask 小 (~500px), 纯细轮廓
    在视频里像背景图案, 填充后红/绿位置一眼可辨。"""
    m = m.astype(np.uint8)
    if m.sum() == 0:
        return
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if fill > 0:
        ov = img.copy()
        cv2.drawContours(ov, cnts, -1, color, -1)
        cv2.addWeighted(ov, fill, img, 1 - fill, 0, img)
    cv2.drawContours(img, cnts, -1, color, th)


def load_pre_masks(sd):
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        return pickle.load(f)['all_masks']


def gt_world_traj(data_dir, sd, n, sel):
    """GT mask 质心 + depth 反投影到世界系 (Z-up), 返回 (n,3) 含 NaN。"""
    K0 = np.loadtxt(os.path.join(data_dir, 'cam_K', '000000.txt'))
    ext_dir = os.path.join(sd, 'extrinsics')
    exts = sorted(os.listdir(ext_dir), key=lambda x: int(os.path.splitext(x)[0]))[:n]
    cat = sel['key'].rsplit('_inst', 1)[0]
    am = load_pre_masks(sd)
    fb = {fr['frame_id']: fr for fr in am[cat][sel['best_pre_inst']]}

    pts = np.full((n, 3), np.nan)
    for t in range(n):
        if t not in fb:
            continue
        depth = cv2.imread(os.path.join(data_dir, 'depth', f'{t:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        m = np.asarray(fb[t]['mask']) > 0
        if m.shape != depth.shape:
            continue
        ys, xs = m.nonzero()
        if len(xs) < 5:
            continue
        z = float(np.median(depth[ys, xs]))
        if z <= 0.05:
            continue
        u, v = xs.mean(), ys.mean()
        x = (u - K0[0, 2]) / K0[0, 0] * z
        y = (v - K0[1, 2]) / K0[1, 1] * z
        ext = np.loadtxt(os.path.join(ext_dir, exts[t]))
        pw = np.linalg.inv(ext) @ np.array([x, y, z, 1.0])
        pts[t] = pw[:3]
    return pts


def build_frames(sd, key, stride=2, mesh_pts=1200, three_h=400, two_h=None,
                 poses_dir=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    data = os.path.join(sd, f'fp_data_{key}')
    out = os.path.join(sd, f'fp_out_{key}')
    # 帧数来源: poses 同级的 ob_in_cam 优先 (output_v3 等无 fp_out_<obj> 的场景)
    base = out
    if poses_dir and os.path.isdir(os.path.join(os.path.dirname(poses_dir),
                                                'ob_in_cam')):
        base = os.path.dirname(poses_dir)
    n = len(os.listdir(os.path.join(base, 'ob_in_cam')))

    mesh = trimesh.load(os.path.join(data, 'mesh.obj'))
    vs = np.asarray(mesh.vertices)
    if len(vs) > mesh_pts:
        vs = vs[np.random.choice(len(vs), mesh_pts, replace=False)]
    mesh_r = float(np.linalg.norm(vs - vs.mean(0), axis=1).max())

    pdir = poses_dir or os.path.join(out, 'ob_in_world')
    P = np.stack([np.loadtxt(os.path.join(pdir, f'{t:06d}.txt'))
                  for t in range(n)])

    # 轨迹画 mesh 质心位置 (R@c+t), 而非平移分量 t (mesh原点可能离几何中心很远,
    # 否则 3D 面板里物体点云和轨迹会分离)
    c = np.asarray(mesh.vertices).mean(0)
    C = np.einsum('tij,j->ti', P[:, :3, :3], c) + P[:, :3, 3]

    ext_dir = os.path.join(sd, 'extrinsics')
    exts = sorted(os.listdir(ext_dir), key=lambda x: int(os.path.splitext(x)[0]))[:n]
    cams = np.stack([np.linalg.inv(np.loadtxt(os.path.join(ext_dir, fn)))[:3, 3]
                     for fn in exts])

    sel = json.load(open(os.path.join(sd, 'fp_mask_select', f'{key}.json')))
    sel['key'] = key
    gts = gt_world_traj(data, sd, n, sel)

    # 固定 3D 范围: 覆盖 FP/GT 轨迹 + 相机 + mesh 半径
    cand = [C, cams]
    ok = gts[np.isfinite(gts).all(1)]
    if len(ok):
        cand.append(ok)
    allp = np.vstack(cand)
    lo, hi = allp.min(0) - mesh_r - 0.1, allp.max(0) + mesh_r + 0.1
    c = (lo + hi) / 2
    half = float((hi - lo).max()) / 2
    lo, hi = c - half, c + half

    cat = key.rsplit('_inst', 1)[0]
    am = load_pre_masks(sd)
    fb = {fr['frame_id']: fr for fr in am[cat][sel['best_pre_inst']]}

    frames = []
    hist = []
    ts = list(range(0, n, stride))
    for t in ts:
        img = cv2.imread(os.path.join(data, 'rgb', f'{t:06d}.png'))
        if img is None:
            continue
        H, W = img.shape[:2]
        if two_h and H != two_h:
            s = two_h / H
            img = cv2.resize(img, (int(W * s), two_h))
            H, W = img.shape[:2]
            Km = np.loadtxt(os.path.join(data, 'cam_K', f'{t:06d}.txt')) * s
        else:
            Km = np.loadtxt(os.path.join(data, 'cam_K', f'{t:06d}.txt'))
        Pw = P[t]
        ob_in_cam = np.loadtxt(os.path.join(ext_dir, exts[t])) @ Pw
        pm = rasterize_mesh(mesh, ob_in_cam, Km, H, W)
        contour(img, pm, (0, 220, 0), th=2, fill=0.18)
        if t in fb:
            m = np.asarray(fb[t]['mask']) > 0
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H)) > 0
            contour(img, m, (0, 0, 255), th=3, fill=0.45)
        pc = pm.nonzero()
        if len(pc[0]):
            hist.append((int(pc[1].mean()), int(pc[0].mean())))
            for u, v in hist[max(0, len(hist) - 40):-1]:
                cv2.circle(img, (u, v), 1, (0, 140, 255), -1)
        cv2.putText(img, f'{key} t={t}', (8, 22), 0, 0.55, (255, 255, 255), 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---- 3D 面板 ----
        fig = plt.figure(figsize=(4.4, 4.4), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        mv = (Pw[:3, :3] @ vs.T).T + Pw[:3, 3]
        ax.scatter(mv[:, 0], mv[:, 1], mv[:, 2], s=1.2, c='#2aa8a8', alpha=.5, depthshade=False)
        ax.plot(cams[:, 0], cams[:, 1], cams[:, 2], '-', c='#aaa', lw=1)
        ax.scatter(*cams[t], c='r', s=35, marker='^')
        fp = C[:t + 1]
        ax.plot(fp[:, 0], fp[:, 1], fp[:, 2], '-', c='#ff8c00', lw=1.6)
        ax.scatter(*fp[-1], c='#ff8c00', s=30)
        g = gts[:t + 1]
        ok = np.isfinite(g).all(1)
        if ok.any():
            ax.plot(g[ok, 0], g[ok, 1], g[ok, 2], '-', c='#2fb344', lw=1.6)
            ax.scatter(*g[ok][-1], c='#2fb344', s=30)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.view_init(elev=22, azim=-58)
        ax.set_title(f'world (Z-up) t={t}', fontsize=9)
        ax.tick_params(labelsize=6)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        buf = cv2.resize(buf, (three_h, three_h), interpolation=cv2.INTER_AREA)

        h = max(img.shape[0], buf.shape[0])
        canvas = np.full((h, img.shape[1] + buf.shape[1] + 6, 3), 255, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        canvas[:buf.shape[0], img.shape[1] + 6:] = buf
        frames.append(canvas)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--objs', default=None)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--fps', type=int, default=14)
    ap.add_argument('--two_h', type=int, default=400, help='2D 面板高度像素')
    ap.add_argument('--poses', default='ob_in_world',
                    help='fp_out_<obj> 下的位姿子目录 (如 ob_in_world_locked)')
    ap.add_argument('--all', action='store_true',
                    help='不过滤, 全部实例 (默认只画 stage3 判定为动态的实例)')
    args = ap.parse_args()
    sd = args.scene_dir

    if args.objs:
        keys = args.objs.split(',')
    else:
        rep = json.load(open(os.path.join(sd, 'fp_batch_report.json')))
        keys = list(rep.keys())
        if not args.all:
            keys = dynamic_keys(sd, keys)

    import imageio.v2 as imageio
    for k in keys:
        try:
            pd = args.poses
            if not os.path.isabs(pd):
                pd = os.path.join(sd, f'fp_out_{k}', pd)
            frames = build_frames(sd, k, stride=args.stride, two_h=args.two_h,
                                  poses_dir=pd)
            vis = os.path.join(sd, 'fp_vis', k)
            os.makedirs(vis, exist_ok=True)
            p = os.path.join(vis, 'combo.gif')
            imageio.mimsave(p, frames, fps=args.fps, loop=0)
            print(f'  {k}: {p} ({os.path.getsize(p) / 1e6:.1f} MB, {len(frames)}f)')
            test_dir = os.path.join(TEST_FP_VIS, f'{os.path.basename(sd.rstrip("/"))}_{k}')
            if os.path.isdir(os.path.dirname(test_dir)):
                os.makedirs(test_dir, exist_ok=True)
                import shutil
                shutil.copy(p, os.path.join(test_dir, 'combo.gif'))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'  {k}: FAILED {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
