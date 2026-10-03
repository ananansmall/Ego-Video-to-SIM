#!/usr/bin/env python3
"""
vis_alignment_3d.py — Generic 3D visualization of hand alignment pipeline

Usage:
    python vis_alignment_3d.py --hawor-dir /path/to/hawor --ras-dir /path/to/ras --output /tmp/out

Features:
  - Both hands (left=cyan, right=magenta) with Z<0 filtered
  - GLB world origin axes + camera frame axes (scaled to plot range)
  - Camera trajectory as connected line
  - Object bounding boxes (cuboid only, no mesh)
  - 01c vs raw comparison (side-by-side if both exist)
"""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import trimesh
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

sys.path.insert(0, str(Path('/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')))
from mano_layer import MANOLayer
import torch

SKELETON = [(0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
            (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]
H_COL = {0: 'cyan', 1: 'magenta'}
H_LBL = {0: 'Left', 1: 'Right'}
H_SID = {0: 'left', 1: 'right'}

def draw_bbox(ax, bounds, color, alpha=0.25):
    p0, p1 = bounds
    c = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                  [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        ax.plot3D(*c[[e[0],e[1]]].T, color=color, alpha=alpha, lw=1.0)

def draw_origin_axes(ax, scale, lw=3):
    for i, (c, lbl) in enumerate([('red','X'),('green','Y'),('blue','Z')]):
        v = np.zeros(3); v[i] = scale
        ax.plot([0,v[0]],[0,v[1]],[0,v[2]], color=c, lw=lw, zorder=10)
        ax.scatter(v[0],v[1],v[2], c=c, s=120, marker='>', alpha=0.9, zorder=10)
        ax.text(v[0]*1.2,v[1]*1.2,v[2]*1.2, lbl, color=c, fontsize=16, fontweight='bold')

def draw_camera_frame(ax, t, R, scale, lw=2.5):
    o = np.array(t)
    for i, (c, lbl) in enumerate([('red','X'),('green','Y'),('blue','Z')]):
        v = R[:, i] * scale; tip = o + v
        ax.plot([o[0],tip[0]],[o[1],tip[1]],[o[2],tip[2]],
                color=c, lw=lw, alpha=0.8, linestyle='--', zorder=10)
        ax.scatter(tip[0],tip[1],tip[2], c=c, s=120, marker='>', alpha=0.8, zorder=10)
        ax.text(tip[0]+v[0]*0.2,tip[1]+v[1]*0.2,tip[2]+v[2]*0.2,
                lbl, color=c, fontsize=13, fontweight='bold', alpha=0.8)
    ax.scatter(*o, c='black', s=80, marker='d', zorder=5)
    ax.text(o[0],o[1],o[2]+0.15, 'cam0', fontsize=10, color='black', fontweight='bold')

def find_npz(hawor_dir):
    rec = Path(hawor_dir)/'reconstruction'
    files = sorted(rec.glob('hawor_results_*.npz'))
    if not files: raise FileNotFoundError(f"No npz in {rec}")
    return files

def find_params(hawor_dir):
    p = Path(hawor_dir)/'output/alignment/transform_params.npz'
    if not p.exists(): raise FileNotFoundError(f"No params: {p}")
    return str(p)

def compute_joints(ml, rot, hp, t):
    p = torch.from_numpy(np.concatenate([rot, hp]).astype(np.float32)).unsqueeze(0)
    tt = torch.from_numpy(t.astype(np.float32)).unsqueeze(0)
    _, j = ml(p, tt)
    return j.detach().cpu().numpy()[0]

def xform(pts, s, R, t):
    return s * (R @ pts.T).T + t

def load_hands(npz, s, R, t, idxs):
    d = dict(np.load(npz, allow_pickle=True))
    pv = d['pred_valid']
    ml = {}
    res = {}
    for hidx in [0,1]:
        if not pv[hidx].any(): continue
        if hidx not in ml:
            ml[hidx] = MANOLayer(H_SID[hidx], d['pred_betas'][hidx,0].astype(np.float32))
        jg, rz = [], []
        for fi in idxs:
            if not pv[hidx,fi]: continue
            j = compute_joints(ml[hidx], d['pred_rot'][hidx,fi], d['pred_hand_pose'][hidx,fi], d['pred_trans'][hidx,fi])
            if np.any(np.isnan(j)): continue
            rz.append(j[0,2])
            jg.append(xform(j, s, R, t))
        if jg:
            res[hidx] = {'glb': np.array(jg), 'raw_z': np.array(rz)}
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--output', default='/tmp/align')
    parser.add_argument('--num-samples', type=int, default=30)
    parser.add_argument('--axis-scale', type=float, default=2.0)
    parser.add_argument('--cam-axis-scale', type=float, default=1.5)
    args = parser.parse_args()

    hd, rd = Path(args.hawor_dir), Path(args.ras_dir)
    npz_list = find_npz(hd)
    aligned = next((f for f in npz_list if '_depth_aligned' in str(f)), None)
    raw_npz = next((f for f in npz_list if '_depth_aligned' not in str(f)), None)
    if not aligned: aligned = raw_npz
    params_p = find_params(hd)
    glb_p = rd/'final_scene.glb'

    print(f'  Aligned: {aligned}\n  Raw: {raw_npz}\n  Params: {params_p}\n  GLB: {glb_p}')

    par = dict(np.load(params_p))
    s, R, t = float(par['scale_ratio']), par['R_hand_to_glb'], par['t_hand_to_glb']
    da = dict(np.load(aligned, allow_pickle=True))

    # Camera trajectory
    cam_glb = xform(da['t_c2w'], s, R, t)
    nf = len(da['R_c2w'])
    R0 = da['R_c2w'][0]; t0 = xform(np.array([da['t_c2w'][0]]), s, R, t)[0]
    R_cam = R @ R0

    # Objects
    print('Loading GLB (bounds only)...')
    try:
        sc = trimesh.load(str(glb_p))
        geoms = [(n, g.vertices.mean(0), g.bounds) for n, g in sc.geometry.items()] if hasattr(sc,'geometry') else [('obj', sc.vertices.mean(0), sc.bounds)]
    except:
        geoms = []

    # Hands
    step = max(1, nf // args.num_samples)
    idxs = np.arange(0, nf, step)[:args.num_samples]
    h_a = load_hands(str(aligned), s, R, t, idxs)
    h_r = load_hands(str(raw_npz), s, R, t, idxs) if raw_npz and raw_npz != aligned else {}

    # Stats
    trace_n = np.trace(R)
    ang = np.arccos(min(1,max(-1,(trace_n-1)/2)))*180/np.pi
    print(f'Scale={s:.3f}× Rot={ang:.1f}° Frames={nf}')
    for label, hh in [('01c',h_a),('raw',h_r)]:
        for hidx,inf in hh.items():
            gz = inf['glb'][:,0,2]; rz = inf['raw_z']
            print(f'  [{label}] {H_LBL[hidx]}: N={len(gz)}  HaWoR_Z=[{rz.min():.3f},{rz.max():.3f}]  GLB_Z=[{gz.min():.3f},{gz.max():.3f}]')

    # Compute axis scale from plot range
    all_pts = np.vstack([cam_glb] + [g[1] for g in geoms] + [inf['glb'][:,:3].reshape(-1,3) for hh in [h_a,h_r] for inf in hh.values()])
    plot_range = max(all_pts.ptp(0)) * 0.5
    ax_scale = plot_range * 0.35  # 35% of half-range
    cam_scale = plot_range * 0.25

    # ─── Plot ───
    ncols = 2 if h_r else 1
    fig, axes = plt.subplots(1, ncols, figsize=(14*ncols, 8), subplot_kw=dict(projection='3d'))
    if ncols == 1: axes = [axes]

    configs = [(h_a, '01c depth-aligned (HaWoR+GLB)', axes[0])]
    if h_r: configs.append((h_r, 'raw (no 01c)', axes[1]))

    labels_shown = {k: False for k in H_COL}
    for hands, title, ax in configs:
        draw_origin_axes(ax, ax_scale)
        draw_camera_frame(ax, t0, R_cam, cam_scale)

        # Objects
        for i, (nm, ct, bd) in enumerate(geoms):
            c = plt.cm.tab10(i)
            draw_bbox(ax, bd, c)
            sz = bd[1] - bd[0]
            ax.text(ct[0],ct[1],bd[1,2]+0.1, f'o{i} {sz[0]:.2f}x{sz[1]:.2f}x{sz[2]:.2f}m',
                    color=c, fontsize=7, fontweight='bold')

        # Hands
        for hidx, inf in hands.items():
            c = H_COL[hidx]
            for j in inf['glb']:
                v = j[:,2] >= 0
                if not v.any(): continue
                ax.scatter(j[v,0], j[v,1], j[v,2], s=2, color=c, alpha=0.6)
                for a,b in SKELETON:
                    if j[a,2]>=0 and j[b,2]>=0:
                        ax.plot([j[a,0],j[b,0]],[j[a,1],j[b,1]],[j[a,2],j[b,2]], color=c, alpha=0.4, lw=0.4)
            if not labels_shown[hidx]:
                ax.plot([],[],[], color=c, lw=2, alpha=0.8, label=f'{H_LBL[hidx]} hand')
                labels_shown[hidx] = True

        # Camera trajectory as connected line
        valid = cam_glb[:,2] >= 0
        seg_start = None
        for fi in range(len(cam_glb)):
            if valid[fi]:
                if seg_start is None: seg_start = fi
            elif seg_start is not None:
                ax.plot(cam_glb[seg_start:fi,0], cam_glb[seg_start:fi,1], cam_glb[seg_start:fi,2], 'b-', lw=1, alpha=0.5)
                seg_start = None
        if seg_start is not None:
            ax.plot(cam_glb[seg_start:,0], cam_glb[seg_start:,1], cam_glb[seg_start:,2], 'b-', lw=1, alpha=0.5)
        # Camera markers
        for fi in range(0, nf, max(1, nf//15)):
            if valid[fi]:
                ax.scatter(cam_glb[fi,0], cam_glb[fi,1], cam_glb[fi,2], c='blue', s=8, marker='^', alpha=0.5)
        ax.plot([],[],[], 'b-', lw=1.5, alpha=0.5, label='Camera trajectory')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    out = args.output
    plt.savefig(f'{out}_3d.png', dpi=200, bbox_inches='tight')
    print(f'Saved: {out}_3d.png')
    print(f'\n[DONE]')

if __name__ == '__main__':
    main()