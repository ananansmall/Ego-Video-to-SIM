#!/usr/bin/env python3
"""
Interactive 3D viewer for aligned hand + camera + GLB data (SAPIEN-compatible transforms).
Uses plotly for interactive 3D visualization in browser.
Transform chain matches render_quick.py / render_hand_mesh.py exactly.

Reference points (transformed to GLB world space with the same xform as the camera):
  * Camera trajectory      : HaWoR / RAS camera position  t_c2w  ->  xform(t_c2w, s, R_hand_to_glb, t)
  * Hand root (pred_trans) : default mode plots this as the hand trajectory
  * MANO mesh (--mesh)     : full MANO hand via FK, wrist = joint 0 (== pred_trans)
  * Midpoint (render_quick): (thumb_tip_j20 + index_tip_j16) / 2, the pinch/grasp point
GLB objects are drawn at their node world-transform (T @ vertices), not raw local geometry.

Two modes in one file:
  * Default  : lightweight view of camera trajectories + hand-root (pred_trans) trajectories
               + GLB object bounding boxes. No torch / MANO model needed.
  * --mesh   : additionally render the full MANO hand mesh every N frames (per chosen hand),
               optionally comparing Raw vs Depth-aligned inputs. Requires torch + mano_layer.

Usage:
  # Lightweight viewer (no MANO model load)
  python View/render_viewer.py \
      --hawor-dir /path/to/hawor --ras-dir /path/to/ras \
      --output View/videos/viewer.html

  # With MANO hand meshes (every 10 frames, right hand)
  python View/render_viewer.py \
      --hawor-dir /path/to/hawor --ras-dir /path/to/ras --mesh \
      --step 10 --hands right \
      --depth-npz /path/to/hawor/reconstruction/hawor_results_*_depth_aligned.npz \
      --output View/videos/viewer_mesh.html
"""
import argparse, warnings, sys
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import trimesh

_BASE = Path(__file__).resolve().parent.parent  # combination/
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE.parent / 'position_retargeting'))


def xform(pts, s, R, t):
    """Apply s*R@pts + t to an array of points (or a single point)."""
    if pts.ndim == 1:
        return s * (R @ pts) + t
    return s * (R @ pts.T).T + t


def load_ras_cameras(ras_dir):
    """Load RAS camera positions (already in GLB world) from extrinsics/*.txt."""
    ext_dir = Path(ras_dir) / 'extrinsics'
    files = sorted(ext_dir.glob('*.txt'), key=lambda x: int(x.stem))
    ras_pos = []
    for f in files:
        ext = np.loadtxt(str(f))
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_w2c, t_w2c = ext[:3, :3], ext[:3, 3]
        R_c2w = R_w2c.T
        ras_pos.append(-R_c2w @ t_w2c)
    return np.array(ras_pos) if ras_pos else np.empty((0, 3))


def get_glb_objects(ras_dir):
    glb_p = Path(ras_dir) / 'final_scene.glb'
    geoms = []
    if glb_p.exists():
        try:
            scene = trimesh.load(str(glb_p), force='scene')
            for node_name in scene.graph.nodes_geometry:
                T, geom_name = scene.graph.get(node_name)
                if geom_name is None or geom_name not in scene.geometry:
                    continue
                geom = scene.geometry[geom_name]
                if not (hasattr(geom, 'vertices') and len(geom.vertices) > 0):
                    continue
                # 应用 GLB 节点的世界变换, 否则物体会偏移到本地几何原点 (误差可达 1.5 m)
                v_world = geom.vertices @ T[:3, :3].T + T[:3, 3]
                geoms.append((node_name, v_world.mean(0), np.array([v_world.min(0), v_world.max(0)])))
        except Exception as e:
            print(f'Warning: could not load GLB: {e}')
    return geoms


def add_axes(fig, axis_scale):
    for i, (clr, lbl) in enumerate([('red', 'X'), ('green', 'Y'), ('blue', 'Z')]):
        v = np.zeros(3); v[i] = axis_scale
        fig.add_trace(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]], mode='lines+text',
            line=dict(color=clr, width=8), text=['', f' {lbl}'],
            textfont=dict(color=clr, size=16), showlegend=False))


def add_glb_bboxes(fig, geoms):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    for i, (nm, ct, bd) in enumerate(geoms):
        col = colors[i % len(colors)]
        p0, p1 = bd
        corners = np.array([[p0[0], p0[1], p0[2]], [p1[0], p0[1], p0[2]], [p1[0], p1[1], p0[2]], [p0[0], p1[1], p0[2]],
                            [p0[0], p0[1], p1[2]], [p1[0], p0[1], p1[2]], [p1[0], p1[1], p1[2]], [p0[0], p1[1], p1[2]]])
        for e in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            fig.add_trace(go.Scatter3d(x=corners[e, 0], y=corners[e, 1], z=corners[e, 2], mode='lines',
                line=dict(color=col, width=2), showlegend=False))


def add_camera_frustums(fig, t0_glb, R_cam_glb, cam_s):
    for i, (clr, lbl) in enumerate([('red', 'camX'), ('green', 'camY'), ('blue', 'camZ')]):
        v = R_cam_glb[:, i] * cam_s
        tip = t0_glb + v
        fig.add_trace(go.Scatter3d(x=[t0_glb[0], tip[0]], y=[t0_glb[1], tip[1]], z=[t0_glb[2], tip[2]],
            mode='lines+text', line=dict(color=clr, width=4, dash='dash'), text=['', lbl],
            textfont=dict(color=clr, size=14), showlegend=False))
    look = R_cam_glb[:, 2] * cam_s * 0.8
    tip = t0_glb + look
    fig.add_trace(go.Scatter3d(x=[t0_glb[0], tip[0]], y=[t0_glb[1], tip[1]], z=[t0_glb[2], tip[2]],
        mode='lines+text', line=dict(color='orange', width=6), text=['', 'Look'],
        textfont=dict(color='darkorange', size=14), showlegend=False))


def add_world_point(fig, pt, name, color, size=6, symbol='diamond'):
    fig.add_trace(go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[pt[2]], mode='markers',
        marker=dict(size=size, color=color, symbol=symbol), name=name, showlegend=bool(name)))


def safe_layout(fig, all_pts, title):
    all_pts = [p for p in all_pts if p is not None and len(p) > 0]
    if all_pts:
        ag = np.concatenate(all_pts, axis=0)
        ag = ag[np.isfinite(ag).all(axis=1)]
        if len(ag) > 0:
            lo, hi = ag.min(0), ag.max(0)
            rng = np.maximum(hi - lo, 0.5)
            pad = rng * 0.15
            fig.update_layout(scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data',
                xaxis_range=[lo[0] - pad[0], hi[0] + pad[0]],
                yaxis_range=[lo[1] - pad[1], hi[1] + pad[1]],
                zaxis_range=[max(lo[2] - pad[2], -0.1), hi[2] + pad[2]]))
    fig.update_layout(title=title, scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        width=1200, height=800, showlegend=True, legend=dict(font=dict(size=12)))


def render_hand_meshes(fig, d_src, s, R_HAND, t, hidx, step, rgb, label):
    """Render MANO hand meshes every `step` frames from data dict `d_src`."""
    from mano_layer import MANOLayer
    import torch

    hand_type = 'left' if hidx == 0 else 'right'
    if 'pred_betas' in d_src:
        betas_arr = d_src['pred_betas'][hidx]
        fv = next((fi for fi in range(len(betas_arr)) if not np.any(np.isnan(betas_arr[fi]))), 0)
        betas = betas_arr[fv].astype(np.float32)
    else:
        betas = np.zeros(10, dtype=np.float32)
    ml = MANOLayer(hand_type, betas)
    faces = ml.f.cpu().numpy()
    r0, g0, b0 = rgb
    mesh_col = f'rgba({r0},{g0},{b0},0.15)'
    line_col = f'rgba({r0},{g0},{b0},0.4)'
    tip_col = f'rgb({r0},{g0},{b0})'

    n = len(d_src['pred_trans'][hidx])
    frame_indices = list(range(0, n, step))
    valid = 0
    for fi in frame_indices:
        trans = np.array(d_src['pred_trans'][hidx][fi]).astype(np.float32)
        rot = np.array(d_src['pred_rot'][hidx][fi]).astype(np.float32)
        hp = np.array(d_src['pred_hand_pose'][hidx][fi]).astype(np.float32)
        # Skip invalid frames
        if np.any(np.isnan(trans)) or np.all(np.abs(trans) < 1e-6):
            continue
        if np.any(np.isnan(rot)) or np.any(np.isnan(hp)):
            continue

        p_t = torch.from_numpy(np.concatenate([rot, hp])).unsqueeze(0)
        t_t = torch.from_numpy(trans).unsqueeze(0)
        with torch.no_grad():
            v, j = ml(p_t, t_t)
        verts = v[0].numpy()      # (778, 3) meters
        joints = j[0].numpy()     # (21, 3) meters

        verts_glb = xform(verts, s, R_HAND, t)
        joints_glb = xform(joints, s, R_HAND, t)

        fig.add_trace(go.Mesh3d(
            x=verts_glb[:, 0], y=verts_glb[:, 1], z=verts_glb[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=mesh_col, opacity=0.15, name=f'{label} f{fi}',
            hoverinfo='name', showlegend=False))

        # Thumb tip highlighted
        tip = joints_glb[20]
        fig.add_trace(go.Scatter3d(x=[tip[0]], y=[tip[1]], z=[tip[2]], mode='markers',
            marker=dict(size=4, color=tip_col, line=dict(color='black', width=1)),
            name=f'{label} thumb {fi}', showlegend=False))

        # Joint skeleton
        wrist = joints_glb[0]
        for jidx in range(1, 21):
            jp = joints_glb[jidx]
            fig.add_trace(go.Scatter3d(x=[wrist[0], jp[0]], y=[wrist[1], jp[1]], z=[wrist[2], jp[2]],
                mode='lines', line=dict(color=line_col, width=1), showlegend=False))
        valid += 1
    print(f"  {label}: {valid} valid meshes out of {len(frame_indices)} frames")
    return valid


def main():
    parser = argparse.ArgumentParser(
        description='Interactive 3D viewer (merged): camera + hand roots [+ MANO mesh]')
    parser.add_argument('--hawor-dir', required=True, help='HaWoR data dir')
    parser.add_argument('--ras-dir', required=True, help='RAS reconstruction dir')
    parser.add_argument('--output', default=None,
                        help='Output HTML path (default: <View>/videos/viewer.html)')
    parser.add_argument('--params-path', default=None,
                        help='transform_params.npz (default: hawor_dir/output/alignment/transform_params.npz)')
    parser.add_argument('--axis-scale', type=float, default=0.5, help='Axis arrow length')
    parser.add_argument('--camera-scale', type=float, default=0.05, help='Camera cone size')
    # mesh options
    parser.add_argument('--mesh', action='store_true', help='Render MANO hand meshes every N frames')
    parser.add_argument('--step', type=int, default=10, help='Mesh mode: render every N frames')
    parser.add_argument('--hands', choices=['left', 'right', 'both'], default='right',
                        help='Mesh mode: which hand(s) to render')
    parser.add_argument('--depth-npz', default=None,
                        help='Depth-aligned HaWoR npz (mesh mode: compare Raw vs Depth)')
    args = parser.parse_args()

    hawor_dir = Path(args.hawor_dir)
    ras_dir = Path(args.ras_dir)
    out_html = Path(args.output) if args.output else (Path(__file__).resolve().parent / 'videos' / 'viewer.html')
    out_html.parent.mkdir(parents=True, exist_ok=True)

    # ── Load transform params ──
    params_p = Path(args.params_path) if args.params_path else (hawor_dir / 'output' / 'alignment' / 'transform_params.npz')
    par = dict(np.load(params_p))
    s = float(par['scale_ratio'])
    R = par['R_hand_to_glb']
    t = par['t_hand_to_glb']
    Rx_hand = par.get('Rx_hand', np.diag([1, -1, -1]))
    R_HAND = R @ Rx_hand
    print(f"scale={s:.4f}, R_h2g@R_x trace={np.trace(R_HAND):.3f}")

    # ── Load HaWoR data ──
    rec_dir = hawor_dir / 'reconstruction'
    files = sorted(rec_dir.glob('hawor_results_*.npz'))
    raw_npz = next((f for f in files if '_depth_aligned' not in str(f)), files[0])
    d = dict(np.load(str(raw_npz), allow_pickle=True))
    R_c2w_all = d['R_c2w']
    t_c2w_all = d['t_c2w']
    pred_valid = d.get('pred_valid', None)

    # ── Transform HaWoR camera trajectory to GLB space ──
    cam_glb = xform(t_c2w_all, s, R, t)
    cam_valid = np.isfinite(cam_glb).all(axis=1) & (cam_glb[:, 2] >= 0)

    # ── Load RAS camera positions (already in GLB space) ──
    ras_pos = load_ras_cameras(ras_dir)
    ras_valid = (np.isfinite(ras_pos).all(axis=1) & (ras_pos[:, 2] >= 0)) if len(ras_pos) else np.array([])

    # ── Load GLB objects ──
    geoms = get_glb_objects(ras_dir)
    print(f"GLB: {len(geoms)} objects loaded")

    # ── Build figure ──
    fig = go.Figure()
    add_axes(fig, args.axis_scale)
    add_glb_bboxes(fig, geoms)

    # Camera trajectories
    if cam_valid.any():
        c = cam_glb[cam_valid]
        fig.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode='lines',
            line=dict(color='blue', width=3), name='HaWoR cam trajectory'))
    if ras_valid.any():
        r = ras_pos[ras_valid]
        fig.add_trace(go.Scatter3d(x=r[:, 0], y=r[:, 1], z=r[:, 2], mode='lines',
            line=dict(color='orange', width=3), name='RAS cam trajectory'))

    # Hand-root trajectories (both hands)
    hand_labels = {0: 'L Hand', 1: 'R Hand'}
    hand_colors = {0: 'rgba(0,255,0,0.8)', 1: 'rgba(255,215,0,0.8)'}
    for hidx in [0, 1]:
        pts = np.array(d['pred_trans'][hidx])
        v = np.ones(pts.shape[0], dtype=bool)
        if pred_valid is not None:
            pv = pred_valid
            v &= (pv[hidx] if pv.ndim == 2 else pv)
        v &= ~np.any(np.isnan(pts), axis=1)
        pts_glb = xform(pts, s, R_HAND, t)
        ok = v & np.isfinite(pts_glb).all(axis=1)
        if ok.any():
            p = pts_glb[ok]
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='lines+markers',
                marker=dict(size=2, color=hand_colors[hidx]),
                line=dict(color=hand_colors[hidx], width=1.5), name=hand_labels[hidx]))

    # Camera frustums at first valid frame
    first_cam = int(np.where(cam_valid)[0][0]) if cam_valid.any() else 0
    R0 = R_c2w_all[first_cam]
    t0_glb = xform(t_c2w_all[first_cam], s, R, t)
    add_camera_frustums(fig, t0_glb, R @ R0, args.camera_scale)
    add_world_point(fig, t0_glb, 'Cam0 Pos', 'black', symbol='diamond')

    # ── Alignment stats ──
    if cam_valid.any() and ras_valid.any():
        h0 = cam_glb[cam_valid][:1]
        r0 = ras_pos[ras_valid][:1]
        d0 = np.linalg.norm(h0[0] - r0[0])
        print(f"\nFirst frame alignment: dist={d0:.6f}m")
        h_pca = np.linalg.svd(cam_glb[cam_valid] - cam_glb[cam_valid].mean(0), full_matrices=False)[2]
        r_pca = np.linalg.svd(ras_pos[ras_valid] - ras_pos[ras_valid].mean(0), full_matrices=False)[2]
        cos_a = np.clip(np.abs(h_pca[0] @ r_pca[0]), 0, 1)
        print(f"Trajectory PCA cos(angle)={cos_a:.4f}")

    all_pts = [bd.reshape(-1, 3) for _, _, bd in geoms]
    if cam_valid.any():
        all_pts.append(cam_glb[cam_valid])
    if ras_valid.any():
        all_pts.append(ras_pos[ras_valid])

    # ── MESH MODE ──
    mode_desc = 'camera+hand-roots'
    if args.mesh:
        mode_desc = 'camera+hand-roots+MANO-mesh'
        hand_idxs = ([0] if args.hands == 'left' else [1] if args.hands == 'right' else [0, 1])
        d_da = None
        if args.depth_npz and Path(args.depth_npz).exists():
            d_da = dict(np.load(args.depth_npz, allow_pickle=True))
            print(f"Depth-aligned npz loaded: {args.depth_npz}")
        print(f"Rendering MANO meshes (step={args.step}, hands={args.hands})")
        for hidx in hand_idxs:
            lbl = 'L Hand Raw' if hidx == 0 else 'R Hand Raw'
            render_hand_meshes(fig, d, s, R_HAND, t, hidx, args.step, (170, 170, 170), lbl)
            if d_da is not None:
                dlbl = 'L Hand Depth' if hidx == 0 else 'R Hand Depth'
                render_hand_meshes(fig, d_da, s, R_HAND, t, hidx, args.step, (255, 215, 0), dlbl)

    # ── Layout + write ──
    safe_layout(fig, all_pts,
                f'Aligned Data | Scale={s:.3f}x | {mode_desc}')
    fig.write_html(str(out_html))
    print(f"\n✓ Saved interactive viewer: {out_html}")
    print("  Open in browser to rotate/pan/zoom")


if __name__ == '__main__':
    main()
