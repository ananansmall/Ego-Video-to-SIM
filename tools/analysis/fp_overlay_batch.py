"""为每个 FP 实例生成 2D 叠加 GIF + 轨迹图, 供HTML 索引页展示。

叠加内容:
  - 绿色轮廓 = mesh 面片光栅化投影(FP 位姿)
  - 红色轮廓 = SAM3 GT mask
  - 橙色尾迹 = 历史轨迹
两者重合得好说明跟得准。

输出 <scene_dir>/fp_vis/<obj>/{overlay.gif, traj.png}
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from fp_iou_eval import rasterize_mesh  # noqa: E402
from fp_prepare_data import load_instance  # noqa: E402


def contour(img, m, color):
    m = m.astype(np.uint8)
    if m.sum() == 0:
        return
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, color, 2)


def make_overlay(sd, key, stride=2, fps=16):
    import imageio.v2 as imageio
    data = os.path.join(sd, f'fp_data_{key}')
    out = os.path.join(sd, f'fp_out_{key}')
    vis = os.path.join(sd, 'fp_vis', key)
    os.makedirs(vis, exist_ok=True)

    mesh = trimesh.load(os.path.join(data, 'mesh.obj'))
    sel = json.load(open(os.path.join(sd, 'fp_mask_select', f'{key}.json')))
    cat = key.rsplit('_inst', 1)[0]
    am = pickle_masks(sd)
    fb = {fr['frame_id']: fr for fr in am[cat][sel['best_pre_inst']]}

    n = len(os.listdir(os.path.join(out, 'ob_in_cam')))
    writer = imageio.get_writer(os.path.join(vis, 'overlay.gif'), fps=fps, loop=0)
    hist = []
    for t in range(0, n, stride):
        f = os.path.join(data, 'rgb', f'{t:06d}.png')
        img = cv2.imread(f)
        if img is None:
            continue
        P = np.loadtxt(os.path.join(out, 'ob_in_cam', f'{t:06d}.txt'))
        K = np.loadtxt(os.path.join(data, 'cam_K', f'{t:06d}.txt'))
        H, W = img.shape[:2]
        pm = rasterize_mesh(mesh, P, K, H, W)
        contour(img, pm, (0, 220, 0))                  # 绿: FP 投影
        if t in fb:
            contour(img, np.asarray(fb[t]['mask']) > 0, (0, 0, 255))  # 红: GT
        pc = pm.nonzero()
        if len(pc[0]):
            hist.append((int(pc[1].mean()), int(pc[0].mean())))
            for u, v in hist[max(0, len(hist) - 40):-1]:
                cv2.circle(img, (u, v), 1, (0, 140, 255), -1)
        cv2.putText(img, f'{key}  t={t}', (8, 22), 0, 0.6, (255, 255, 255), 2)
        writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    writer.close()
    return os.path.join(vis, 'overlay.gif')


def pickle_masks(sd):
    import pickle
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        return pickle.load(f)['all_masks']


def make_traj(sd, key):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    wdir = os.path.join(sd, f'fp_out_{key}', 'ob_in_world')
    n = len(os.listdir(wdir))
    P = np.stack([np.loadtxt(os.path.join(wdir, f'{t:06d}.txt')) for t in range(n)])
    C = P[:, :3, 3]
    step = np.linalg.norm(np.diff(C, axis=0), axis=1)
    net = np.linalg.norm(C[-1] - C[0])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(C[:, 0], C[:, 2], '-o', ms=2.5, lw=1.2)
    ax[0].plot(C[0, 0], C[0, 2], 'go', ms=9, label='start')
    ax[0].plot(C[-1, 0], C[-1, 2], 'o', ms=9, mfc='#e8710a', label='end')
    ax[0].set_xlabel('X (m)')
    ax[0].set_ylabel('Z (m)')
    ax[0].set_title(f'{key}  net={net:.3f}m  path={step.sum():.3f}m')
    ax[0].legend()
    ax[0].grid(alpha=.3)

    for i, (lo, hi, nm) in enumerate([(0, 0, 'X'), (1, 1, 'Y'), (2, 2, 'Z')]):
        ax[1].plot(C[:, i], lw=1.2, label=nm)
    ax[1].set_xlabel('frame')
    ax[1].set_ylabel('world coord (m)')
    ax[1].set_title('per-axis')
    ax[1].legend()
    ax[1].grid(alpha=.3)

    ax[2].plot(np.arange(len(step)), step * 100, lw=.8, color='#888')
    ax[2].set_xlabel('frame')
    ax[2].set_ylabel('per-frame step (mm)')
    ax[2].set_title(f'step median={np.median(step) * 1000:.0f}mm')
    ax[2].grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(sd, 'fp_vis', key, 'traj.png')
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--objs', default=None)
    ap.add_argument('--stride', type=int, default=2)
    args = ap.parse_args()
    sd = args.scene_dir

    rep_path = os.path.join(sd, 'fp_batch_report.json')
    keys = list(json.load(open(rep_path)).keys()) if os.path.isfile(rep_path) else []
    if args.objs:
        keys = [k for k in keys if k in set(args.objs.split(','))]

    for k in keys:
        try:
            g = make_overlay(sd, k, stride=args.stride)
            t = make_traj(sd, k)
            print(f'  {k}: {g} ({os.path.getsize(g) / 1e6:.1f} MB), {t}')
        except Exception as e:
            print(f'  {k}: FAILED {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()