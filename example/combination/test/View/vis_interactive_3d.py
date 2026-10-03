#!/usr/bin/env python3
"""
vis_interactive_3d.py — Rewritten following vis_hoi4d_alignment.py's proven pattern.

Side-by-side 3D view:
  LEFT:  HaWoR Camera View (before alignment)
  RIGHT: GLB World View (after 001 alignment)

Features:
  - Large coordinate axes with arrow markers (X=red, Y=green, Z=blue)
  - Camera pose with dashed XYZ axes + orange Look arrow
  - Object bounding boxes + centroid (lightweight)
  - Full hand skeletons (MANO, sampled frames) + ALL-frames wrist trajectory
  - Camera trajectory (ALL frames)
  - Plotly HTML for interactive 3D

Usage:
    python3 View/vis_interactive_3d.py \\
        --hawor-dir /path/to/hawor \\
        --ras-dir /path/to/ras \\
        --output /tmp/align_out
"""
import argparse, os, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import plotly.graph_objects as go

import trimesh

sys.path.insert(0, str(Path('/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')))
from mano_layer import MANOLayer
import torch

SKELETON = [(0,1),(0,5),(0,9),(0,13),(0,17),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),
            (9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]
H_COL = {0: 'cyan', 1: 'magenta'}
H_LBL = {0: 'Left', 1: 'Right'}
H_SID = {0: 'left', 1: 'right'}

# ============================================================
#  Utility functions (following vis_hoi4d_alignment.py pattern)
# ============================================================

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
    """Transform Nx3 points: p' = s * R @ p + t"""
    return s * (R @ pts.T).T + t

def draw_axes_arrows(ax, origin, scale=5.0, lw=6):
    """Big colored X/Y/Z axes with arrow markers."""
    o = np.array(origin, dtype=float)
    # X
    ax.plot([o[0], o[0]+scale], [o[1], o[1]], [o[2], o[2]], color='red', lw=lw, alpha=0.9, zorder=10)
    ax.scatter([o[0]+scale], [o[1]], [o[2]], c='red', s=200, marker='>', alpha=0.9, zorder=10)
    ax.text(o[0]+scale*1.1, o[1], o[2], 'X', color='red', fontsize=18, fontweight='bold', zorder=10)
    # Y
    ax.plot([o[0], o[0]], [o[1], o[1]+scale], [o[2], o[2]], color='green', lw=lw, alpha=0.9, zorder=10)
    ax.scatter([o[0]], [o[1]+scale], [o[2]], c='green', s=200, marker='^', alpha=0.9, zorder=10)
    ax.text(o[0], o[1]+scale*1.1, o[2], 'Y', color='green', fontsize=18, fontweight='bold', zorder=10)
    # Z
    ax.plot([o[0], o[0]], [o[1], o[1]], [o[2], o[2]+scale], color='blue', lw=lw, alpha=0.9, zorder=10)
    ax.scatter([o[0]], [o[1]], [o[2]+scale], c='blue', s=200, marker='^', alpha=0.9, zorder=10)
    ax.text(o[0], o[1], o[2]+scale*1.1, 'Z', color='blue', fontsize=18, fontweight='bold', zorder=10)

def draw_camera_axes(ax, origin, R_cam, scale=1.5):
    """Camera coordinate frame: dashed XYZ + orange Look arrow."""
    o = np.array(origin, dtype=float)
    for i, (c, lbl) in enumerate([('red','X'),('green','Y'),('blue','Z')]):
        vec = R_cam[:,i] * scale
        tip = o + vec
        ax.plot([o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]],
                color=c, lw=2, alpha=0.7, linestyle='--', zorder=10)
        ax.scatter([tip[0]], [tip[1]], [tip[2]], c=c, s=100, marker='>', alpha=0.7, zorder=10)
        ax.text(tip[0]+vec[0]*0.1, tip[1]+vec[1]*0.1, tip[2]+vec[2]*0.1,
                f'cam{lbl}', color=c, fontsize=11, fontweight='bold', alpha=0.7, zorder=10)
    # Look direction (camera -Z in GLB)
    look = -R_cam[:,2] * scale * 0.8
    tip_look = o + look
    ax.plot([o[0], tip_look[0]], [o[1], tip_look[1]], [o[2], tip_look[2]],
            color='orange', lw=4, alpha=1.0, zorder=11)
    ax.scatter([tip_look[0]], [tip_look[1]], [tip_look[2]], c='orange', s=200, marker='>', alpha=1.0, zorder=11)
    ax.text(tip_look[0]+look[0]*0.15, tip_look[1]+look[1]*0.15, tip_look[2]+look[2]*0.15,
            'Look', color='darkorange', fontsize=12, fontweight='bold', zorder=11)
    ax.scatter(*o, c='black', s=80, marker='d', zorder=10)
    ax.text(o[0], o[1], o[2]+0.12, 'cam0', fontsize=10, color='black', fontweight='bold', zorder=10)

def draw_bbox(ax, bounds, color, alpha=0.2):
    p0, p1 = bounds
    corners = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                        [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for e in edges:
        ax.plot3D(*corners[[e[0],e[1]]].T, color=color, alpha=alpha, lw=1.0)

def load_all_wrists(npz_path, s, R, t):
    """Load wrist trajectory for ALL frames: no MANO, just pred_trans."""
    d = dict(np.load(npz_path, allow_pickle=True))
    R0, t0 = d['R_c2w'][0], d['t_c2w'][0]
    pv = d['pred_valid']
    res = {}
    for hidx in [0,1]:
        if not pv[hidx].any(): continue
        pts = d['pred_trans'][hidx]
        valid = pv[hidx]
        # Camera space
        cam_pts = (R0.T @ (pts[valid] - t0).T).T
        # GLB space
        glb_pts = xform(pts[valid], s, R, t)
        res[hidx] = {'cam': cam_pts, 'glb': glb_pts, 'n': valid.sum()}
    return res

def load_mano_skeletons(npz_path, s, R, t, n_frames_max):
    """Load full MANO skeletons for sampled frames."""
    d = dict(np.load(npz_path, allow_pickle=True))
    pv = d['pred_valid']
    nf = len(d['R_c2w'])
    step = max(1, nf // n_frames_max)
    idxs = np.arange(0, nf, step)[:n_frames_max]
    ml, res = {}, {}
    for hidx in [0,1]:
        if not pv[hidx].any(): continue
        if hidx not in ml:
            ml[hidx] = MANOLayer(H_SID[hidx], d['pred_betas'][hidx,0].astype(np.float32))
        joints_glb, joints_cam = [], []
        R0, t0 = d['R_c2w'][0], d['t_c2w'][0]
        for fi in idxs:
            if not pv[hidx, fi]: continue
            j = compute_joints(ml[hidx], d['pred_rot'][hidx, fi], d['pred_hand_pose'][hidx, fi], d['pred_trans'][hidx, fi])
            if np.any(np.isnan(j)): continue
            joints_glb.append(xform(j, s, R, t))
            joints_cam.append((R0.T @ (j - t0).T).T)
        if joints_glb:
            res[hidx] = {'glb': np.array(joints_glb), 'cam': np.array(joints_cam)}
    return res

# ============================================================
#  Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--output', default='/tmp/align_interactive')
    parser.add_argument('--num-samples', type=int, default=3, help='MANO skeletons per hand (default=3, just to show hand shape)')
    parser.add_argument('--axis-scale', type=float, default=5.0, help='World axis length (m)')
    parser.add_argument('--camera-scale', type=float, default=0.35, help='Camera axis length (m)')
    parser.add_argument('--use-depth', action='store_true', help='Use depth_aligned npz instead of raw')
    args = parser.parse_args()

    hawor_dir, ras_dir = Path(args.hawor_dir), Path(args.ras_dir)
    npz_list = find_npz(hawor_dir)
    if args.use_depth:
        selected = next((f for f in npz_list if '_depth_aligned' in str(f)), None)
        if not selected: selected = npz_list[0]
    else:
        selected = next((f for f in npz_list if '_depth_aligned' not in str(f)), None)
        if not selected: selected = npz_list[0]
    params_p = find_params(hawor_dir)
    glb_p = ras_dir/'final_scene.glb'
    out_prefix = args.output

    print(f'  Data: {selected}\n  Params: {params_p}\n  GLB: {glb_p}')
    print(f'  Axis scale: {args.axis_scale}m  Camera axes: {args.camera_scale}m  MANO: {args.num_samples} per hand')

    # Load params
    par = dict(np.load(params_p))
    s, R, t = float(par['scale_ratio']), par['R_hand_to_glb'], par['t_hand_to_glb']
    trace = np.trace(R)
    rot_angle = np.arccos(min(1, max(-1, (trace-1)/2))) * 180 / np.pi

    # Load HaWoR data
    da = dict(np.load(selected, allow_pickle=True))
    nf = len(da['R_c2w'])
    R0, t0_hawor = da['R_c2w'][0], da['t_c2w'][0]
    R_cam_glb = R @ R0
    t0_glb = xform(np.array([t0_hawor]), s, R, t)[0]

    # ── Camera trajectory in GLB ──
    cam_glb = xform(da['t_c2w'], s, R, t)
    cam_valid = cam_glb[:, 2] >= 0

    # ── GLB objects ──
    geoms = []
    try:
        scene = trimesh.load(str(glb_p))
        for name, geom in scene.geometry.items():
            geoms.append((name, geom.vertices.mean(0), geom.bounds))
    except Exception as e:
        print(f'  [WARN] GLB load failed: {e}')

    # ── Hands ──
    m = args.num_samples  # MANO skeletons per hand (just a few to show shape)
    hands_mano = load_mano_skeletons(str(selected), s, R, t, m)
    hands_all = load_all_wrists(str(selected), s, R, t)
    print(f'  Scale={s:.3f}× Rot={rot_angle:.1f}°')
    print(f'  Camera: {cam_valid.sum()}/{nf} Z>=0')
    for hidx in [0,1]:
        if hidx in hands_all:
            print(f'  {H_LBL[hidx]}: wrists={hands_all[hidx]["n"]}  MANO={len(hands_mano.get(hidx,{}).get("glb",[]))}')
    for nm, ct, bd in geoms:
        print(f'  Object: Z=[{bd[0,2]:.3f},{bd[1,2]:.3f}]')

    # ================================================================
    #  MATPLOTLIB: side-by-side view
    # ================================================================
    ax_s = args.axis_scale
    cam_s = args.camera_scale

    # ── LEFT: Camera View ──
    fig = plt.figure(figsize=(22, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('HaWoR Camera View (before alignment)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('X (m)'); ax1.set_ylabel('Y (m)'); ax1.set_zlabel('Z (m)')

    # Origin axes in camera space (camera at origin)
    ax1.scatter(0, 0, 0, c='black', s=120, marker='d', zorder=10)
    ax1.text(0.5, 0, 0, 'Cam origin', fontsize=10, color='gray', fontweight='bold')
    draw_axes_arrows(ax1, [0,0,0], scale=ax_s, lw=4)

    # Objects in camera space (rough approx: R0.T @ (obj_center - t0))
    for i, (nm, ct, bd) in enumerate(geoms):
        c = plt.cm.tab10(i)
        ct_cam = (R0.T @ (ct - t0_hawor))
        # Approximate bbox in camera space
        bd_cam = np.array([(R0.T @ (bd[0] - t0_hawor)), (R0.T @ (bd[1] - t0_hawor))])
        draw_bbox(ax1, np.sort(bd_cam, axis=0), c, alpha=0.12)
        ax1.scatter(*ct_cam, c=c, s=60, marker='o', zorder=5)
        ax1.text(ct_cam[0], ct_cam[1], ct_cam[2]+0.05, f'o{i}', color=c, fontsize=7, fontweight='bold')

    # Hands in camera space
    for hidx, inf in hands_mano.items():
        col = H_COL[hidx]
        for j in inf['cam']:
            v = j[:, 2] >= 0
            if not v.any(): continue
            ax1.scatter(j[v,0], j[v,1], j[v,2], s=5, color=col, alpha=0.5)
            for a, b in SKELETON:
                if j[a,2] >= 0 and j[b,2] >= 0:
                    ax1.plot([j[a,0],j[b,0]],[j[a,1],j[b,1]],[j[a,2],j[b,2]], color=col, alpha=0.3, lw=1.0)
    # Wrist trajectory (all frames)
    for hidx in [0,1]:
        if hidx not in hands_all: continue
        pts = hands_all[hidx]['cam']
        valid = pts[:, 2] >= 0
        if valid.any():
            ax1.plot(pts[valid,0], pts[valid,1], pts[valid,2], color=H_COL[hidx], lw=2, alpha=0.7,
                     label=f'{H_LBL[hidx]} wrist ({pts.shape[0]} frames)')

    # Camera trajectory in camera space = stationary at origin
    ax1.scatter([],[],[], c='blue', s=12, marker='^', alpha=0.5, label='Camera (origin)')
    ax1.legend(fontsize=9, loc='upper left')

    # Set limits for camera view
    all_cam_pts = []
    for hidx in [0,1]:
        if hidx in hands_all: all_cam_pts.append(hands_all[hidx]['cam'])
    if all_cam_pts:
        all_cam = np.concatenate(all_cam_pts, axis=0)
        lo, hi = all_cam.min(axis=0), all_cam.max(axis=0)
        rng = np.maximum(hi - lo, 0.5)
        pad = rng * 0.15
        ax1.set_xlim3d(lo[0]-pad[0], hi[0]+pad[0])
        ax1.set_ylim3d(lo[1]-pad[1], hi[1]+pad[1])
        ax1.set_zlim3d(min(lo[2]-pad[2], -0.1), hi[2]+pad[2])

    # ── RIGHT: GLB World View ──
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('GLB World View (after 001 alignment)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('X (m)'); ax2.set_ylabel('Y (m)'); ax2.set_zlabel('Z (m)')

    # World origin
    draw_axes_arrows(ax2, [0,0,0], scale=ax_s, lw=6)
    ax2.text(0.5, 0.1, 0.1, 'ORIGIN (0,0,0)', fontsize=10, color='gray', fontweight='bold', alpha=0.6)

    # Camera frame
    draw_camera_axes(ax2, t0_glb, R_cam_glb, scale=cam_s)

    # Objects
    for i, (nm, ct, bd) in enumerate(geoms):
        c = plt.cm.tab10(i)
        draw_bbox(ax2, bd, c, alpha=0.12)
        ax2.scatter(*ct, c=c, s=60, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
        ax2.text(ct[0], ct[1], ct[2]+0.05, f'o{i}', color=c, fontsize=7, fontweight='bold')

    # Camera trajectory (all frames)
    if cam_valid.any():
        ax2.plot(cam_glb[cam_valid,0], cam_glb[cam_valid,1], cam_glb[cam_valid,2], 'b-', lw=2, alpha=0.6, label='Camera trajectory')
        # Markers every 10th frame
        for fi in range(0, nf, max(1, nf//20)):
            if cam_valid[fi]:
                ax2.scatter(*cam_glb[fi], c='blue', s=15, marker='^', alpha=0.4)

    # Hands (MANO skeletons)
    for hidx, inf in hands_mano.items():
        col = H_COL[hidx]
        for j in inf['glb']:
            v = j[:, 2] >= 0
            if not v.any(): continue
            ax2.scatter(j[v,0], j[v,1], j[v,2], s=5, color=col, alpha=0.5)
            for a, b in SKELETON:
                if j[a,2] >= 0 and j[b,2] >= 0:
                    ax2.plot([j[a,0],j[b,0]],[j[a,1],j[b,1]],[j[a,2],j[b,2]], color=col, alpha=0.3, lw=1.0)
    # Wrist trajectory (all frames)
    for hidx in [0,1]:
        if hidx not in hands_all: continue
        pts = hands_all[hidx]['glb']
        valid = pts[:, 2] >= 0
        if valid.any():
            ax2.plot(pts[valid,0], pts[valid,1], pts[valid,2], color=H_COL[hidx], lw=2, alpha=0.7,
                     label=f'{H_LBL[hidx]} wrist ({pts.shape[0]} frames)')

    ax2.legend(fontsize=9, loc='upper left')

    # Set limits for GLB view
    glb_pts = []
    if geoms:
        for nm, ct, bd in geoms:
            glb_pts.append(bd.reshape(-1,3))
    if cam_valid.any():
        glb_pts.append(cam_glb[cam_valid])
    for hidx in [0,1]:
        if hidx in hands_all:
            glb_pts.append(hands_all[hidx]['glb'])
    if glb_pts:
        all_g = np.concatenate(glb_pts, axis=0)
        lo, hi = all_g.min(axis=0), all_g.max(axis=0)
        rng = np.maximum(hi - lo, 0.5)
        pad = rng * 0.1
        ax2.set_xlim3d(lo[0]-pad[0], hi[0]+pad[0])
        ax2.set_ylim3d(lo[1]-pad[1], hi[1]+pad[1])
        ax2.set_zlim3d(max(lo[2]-pad[2], -0.1), hi[2]+pad[2])

    plt.tight_layout()
    png_path = f'{out_prefix}_3d_compare.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {png_path}')
    print('Interactive window — drag to rotate, scroll to zoom.')
    plt.show()
    print('Window closed.')

    # ================================================================
    #  PLOTLY HTML (for interactive 3D)
    # ================================================================
    print('Saving Plotly HTML...')
    fig_p = go.Figure()

    # Origin axes
    for i, (col, lbl) in enumerate([(0, 'X'),(1, 'Y'),(2,'Z')]):
        v = np.zeros(3); v[i] = ax_s
        clr = ['red','green','blue'][i]
        fig_p.add_trace(go.Scatter3d(x=[0,v[0]],y=[0,v[1]],z=[0,v[2]],
            mode='lines+text', line=dict(color=clr,width=10), text=['',f' {lbl}'],
            textfont=dict(color=clr,size=24), showlegend=False))
        fig_p.add_trace(go.Scatter3d(x=[v[0]],y=[v[1]],z=[v[2]],
            mode='markers', marker=dict(color=clr,symbol='arrow',size=12), showlegend=False))

    # Camera axes
    for i, (clr, lbl) in enumerate([('red','X'),('green','Y'),('blue','Z')]):
        v = R_cam_glb[:,i] * cam_s
        tip = t0_glb + v
        fig_p.add_trace(go.Scatter3d(x=[t0_glb[0],tip[0]],y=[t0_glb[1],tip[1]],z=[t0_glb[2],tip[2]],
            mode='lines+text', line=dict(color=clr,width=4,dash='dash'), text=['',f'cam{lbl}'],
            textfont=dict(color=clr,size=16), showlegend=False))
    # Look direction
    look = -R_cam_glb[:,2] * cam_s * 0.8
    tip_look = t0_glb + look
    fig_p.add_trace(go.Scatter3d(x=[t0_glb[0],tip_look[0]],y=[t0_glb[1],tip_look[1]],z=[t0_glb[2],tip_look[2]],
        mode='lines+text', line=dict(color='orange',width=6), text=['','Look'],
        textfont=dict(color='darkorange',size=14), showlegend=False))
    fig_p.add_trace(go.Scatter3d(x=[t0_glb[0]],y=[t0_glb[1]],z=[t0_glb[2]],
        mode='text', text=['cam0'], textfont=dict(color='black',size=12), showlegend=False))

    # Objects
    for i, (nm, ct, bd) in enumerate(geoms):
        col = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'][i % 5]
        p0, p1 = bd
        corners = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                           [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
        for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            fig_p.add_trace(go.Scatter3d(x=corners[e,0],y=corners[e,1],z=corners[e,2],
                mode='lines', line=dict(color=col,width=2), showlegend=False))
        sz = bd[1] - bd[0]
        fig_p.add_trace(go.Scatter3d(x=[ct[0]],y=[ct[1]],z=[bd[1,2]+0.05],
            mode='text', text=[f'o{i} ({sz[0]:.2f}x{sz[1]:.2f}x{sz[2]:.2f}m)'],
            textfont=dict(color=col,size=10), showlegend=False))

    # Camera
    if cam_valid.any():
        fig_p.add_trace(go.Scatter3d(x=cam_glb[cam_valid,0],y=cam_glb[cam_valid,1],z=cam_glb[cam_valid,2],
            mode='lines+markers', marker=dict(size=3,color='blue'), line=dict(color='blue',width=2), name='Camera'))

    # Hands
    col_rgba = {'cyan':'rgba(0,255,255,0.6)','magenta':'rgba(255,0,255,0.6)'}
    for hidx in [0,1]:
        if hidx not in hands_all: continue
        pts = hands_all[hidx]['glb']
        valid = pts[:, 2] >= 0
        if valid.any():
            fig_p.add_trace(go.Scatter3d(x=pts[valid,0],y=pts[valid,1],z=pts[valid,2],
                mode='lines+markers', marker=dict(size=3,color=col_rgba[H_COL[hidx]]),
                line=dict(color=col_rgba[H_COL[hidx]],width=2),
                name=f'{H_LBL[hidx]} hand ({valid.sum()} frames)'))

    fig_p.update_layout(title=f'GLB Alignment | Scale={s:.3f}× Rot={rot_angle:.1f}°',
        scene=dict(xaxis_title='X',yaxis_title='Y',zaxis_title='Z', aspectmode='data'),
        width=1200, height=800)
    html_path = f'{out_prefix}_3d.html'
    fig_p.write_html(html_path)
    print(f'Saved: {html_path}')
    print('[DONE]')

if __name__ == '__main__':
    main()