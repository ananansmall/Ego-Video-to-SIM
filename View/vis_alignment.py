#!/usr/bin/env python3
"""
vis_alignment.py — GLB 坐标系下对齐可视化 (与 render_quick.py 一致)
"""
import argparse, sys, warnings, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FixedLocator
from pathlib import Path
warnings.filterwarnings('ignore')

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE.parent / 'position_retargeting'))
import trimesh
from mano_layer import MANOLayer

def xform(pts, s, R, t):
    return s * (R @ pts.T).T + t

H_SID = {0: 'left', 1: 'right'}
COLORS = plt.cm.tab10.colors

def compute_mano_joints(d, hand_idx):
    """只计算指定 hand_idx 的 MANO 关节"""
    import torch
    side = H_SID[hand_idx]
    n = len(d['pred_trans'][hand_idx])
    betas_arr = d['pred_betas'][hand_idx]
    fv = next((fi for fi in range(n) if not np.any(np.isnan(betas_arr[fi]))), None)
    if fv is None:
        return np.full((n, 21, 3), np.nan)
    ml = MANOLayer(side, betas_arr[fv].astype(np.float32))
    joints = np.full((n, 21, 3), np.nan, dtype=np.float64)
    for fi in range(n):
        trans = d['pred_trans'][hand_idx, fi].astype(np.float32)
        rot = d['pred_rot'][hand_idx, fi].astype(np.float32)
        hp = d['pred_hand_pose'][hand_idx, fi].astype(np.float32)
        if np.any(np.isnan(trans)) or np.all(np.abs(trans) < 1e-6): continue
        if np.any(np.isnan(rot)) or np.any(np.isnan(hp)): continue
        with torch.no_grad():
            _, j = ml(torch.from_numpy(np.concatenate([rot, hp])).unsqueeze(0),
                      torch.from_numpy(trans).unsqueeze(0))
        jv = j[0].numpy()
        if not np.any(np.isnan(jv)): joints[fi] = jv
    return joints

def load_ras_cameras(ras_dir):
    files = sorted((Path(ras_dir)/'extrinsics').glob('*.txt'), key=lambda x: int(x.stem))
    ras_pos = []
    for f in files:
        ext = np.loadtxt(str(f))
        if ext.shape == (3,4): ext = np.vstack([ext, [0,0,0,1]])
        R_c2w = ext[:3,:3].T
        ras_pos.append(-R_c2w @ ext[:3,3])
    return np.array(ras_pos)

def load_glb_objects(ras_dir):
    """与 render_quick.py 一致: 本地坐标, 不应用节点变换, 过滤大地面"""
    scene = trimesh.load(str(ras_dir/'final_scene.glb'), force='scene')
    geom_to_node = {}
    for node_name in scene.graph.nodes_geometry:
        result = scene.graph.get(node_name)
        if result and result[1]:
            geom_to_node[result[1]] = node_name
    objects = []
    for name, geom in scene.geometry.items():
        if len(geom.vertices) == 0: continue
        center = geom.vertices.mean(0)
        bounds = geom.bounds
        size = bounds[1] - bounds[0]
        if np.any(size > 0.5):  # 过滤大地面
            continue
        display_name = geom_to_node.get(name, name)
        objects.append({'name': display_name, 'center': center,
                        'bbox_min': bounds[0], 'bbox_max': bounds[1]})
    return objects

def draw_bbox(ax, p0, p1, color, alpha=0.2):
    c = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                  [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
    for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot3D(*c[[e[0],e[1]]].T, color=color, alpha=alpha, lw=1.0)

def draw_axes_arrows(ax, origin, scale=0.15, lw=5):
    o = np.array(origin)
    for c, lbl, v in [('red','X',[scale,0,0]),('green','Y',[0,scale,0]),('blue','Z',[0,0,scale])]:
        tip = o + v
        ax.plot([o[0],tip[0]],[o[1],tip[1]],[o[2],tip[2]], color=c, lw=lw, alpha=0.9, zorder=10)
        ax.text(tip[0]+v[0]*0.15,tip[1]+v[1]*0.15,tip[2]+v[2]*0.15, lbl, color=c, fontsize=16, fontweight='bold')

def draw_camera_axes(ax, origin, R_cam, scale=0.15):
    o = np.array(origin)
    for i,(c,lbl) in enumerate([('red','X'),('green','Y'),('blue','Z')]):
        v = R_cam[:,i]*scale; tip = o+v
        ax.plot([o[0],tip[0]],[o[1],tip[1]],[o[2],tip[2]], color=c, lw=2, ls='--', alpha=0.7, zorder=10)
        ax.text(tip[0]+v[0]*0.15,tip[1]+v[1]*0.15,tip[2]+v[2]*0.15, f'cam{lbl}', color=c, fontsize=8, fontweight='bold')
    look = R_cam[:,2]*scale*1.0; tip_l = o+look
    ax.plot([o[0],tip_l[0]],[o[1],tip_l[1]],[o[2],tip_l[2]], color='orange', lw=4, alpha=1.0, zorder=11)
    ax.text(tip_l[0]+look[0]*0.2,tip_l[1]+look[1]*0.2,tip_l[2]+look[2]*0.2, 'Look', color='darkorange', fontsize=9, fontweight='bold')
    ax.scatter(*o, c='black', s=80, marker='d', zorder=10)
    ax.text(o[0],o[1],o[2]+0.04, 'cam0', fontsize=10, color='black', fontweight='bold')

def plot_traj(ax, pts, color, label, lw=1.0, alpha=0.8):
    v = np.isfinite(pts).all(axis=1)
    if not v.any(): return
    pts = pts[v]
    ax.plot(pts[:,0], pts[:,1], pts[:,2], color=color, lw=lw, alpha=alpha, label=label)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=0.5, color=color, alpha=alpha*0.6, marker='.')
    mid = pts.mean(0)
    ax.text(mid[0], mid[1], mid[2]+0.03, label, color=color, fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.8))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hawor-dir', required=True)
    ap.add_argument('--ras-dir', required=True)
    ap.add_argument('--hand-idx', type=int, default=0)
    ap.add_argument('--output', default='/tmp/alignment_quick')
    args = ap.parse_args()

    hawor_dir, ras_dir, out_dir, hand_idx = Path(args.hawor_dir), Path(args.ras_dir), Path(args.output), args.hand_idx
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 对齐参数 ===
    params_p = hawor_dir / 'output' / 'alignment' / 'transform_params.npz'
    par = dict(np.load(params_p))
    s, R, t = float(par['scale_ratio']), par['R_hand_to_glb'], par['t_hand_to_glb']
    R_hand = R @ par.get('Rx_hand', np.diag([1,-1,-1]))

    # === 数据 ===
    rec_file = sorted((hawor_dir/'reconstruction').glob('hawor_results_*.npz'))[0]
    d = dict(np.load(str(rec_file), allow_pickle=True))
    geoms = load_glb_objects(ras_dir)
    ras_pos = load_ras_cameras(ras_dir)

    # === HaWoR 相机 (GLB) ===
    cam_glb = xform(d['t_c2w'], s, R, t)
    R_c2w_h0 = d['R_c2w'][0]
    t0_glb = cam_glb[0]

    # === MANO 关节 (SLAM → GLB), 只算 hand_idx ===
    joints = compute_mano_joints(d, hand_idx)  # (N, 21, 3) in SLAM
    thumb = xform(joints[:, 20, :], s, R_hand, t)
    wrist = xform(joints[:, 0, :], s, R_hand, t)
    thumb_ok = ~np.any(np.isnan(thumb), axis=1)
    wrist_ok = ~np.any(np.isnan(wrist), axis=1)
    thumb_pts = thumb[thumb_ok] if thumb_ok.any() else np.zeros((0,3))
    wrist_pts = wrist[wrist_ok] if wrist_ok.any() else np.zeros((0,3))
    side_label = H_SID[hand_idx].upper()
    print(f'  Hand {hand_idx} ({side_label}): thumb={thumb_ok.sum()} fr, wrist={wrist_ok.sum()} fr')

    # === 对齐验证 ===
    h0 = cam_glb[:1]
    r0 = ras_pos[:1]
    d0 = np.linalg.norm(h0[0] - r0[0])
    print(f'  Frame 0: HaWoR={h0[0]}, RAS={r0[0]}, dist={d0:.6f}m')

    # === 自动缩放 ===
    all_pts = [g['bbox_min'].reshape(1,3) for g in geoms] + [g['bbox_max'].reshape(1,3) for g in geoms]
    all_pts += [ras_pos]
    if len(thumb_pts) > 0: all_pts.append(thumb_pts)
    if len(wrist_pts) > 0: all_pts.append(wrist_pts)
    # 只包括 z >= 0 的数据
    if all_pts:
        ag = np.concatenate(all_pts, axis=0)
        ag = ag[np.isfinite(ag).all(axis=1)]
        if len(ag) == 0: ag = np.zeros((1,3))
        # 裁剪 z < 0 的数据
        mask = ag[:, 2] >= 0
        if mask.any():
            ag = ag[mask]
        lo, hi = ag.min(0), ag.max(0)
        data_half = max((hi - lo).max() / 2, 0.05)
        # 确保 z 下限不小于 0
        z_lo = max(lo[2], 0)

    # === 3D 可视化 ===
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ra = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1)))
    ax.set_title(f'GLB Alignment | Scale={s:.3f}x Rot={ra:.1f}deg | Hand={side_label}', fontweight='bold')

    # 坐标轴放在原点 (地面)
    draw_axes_arrows(ax, [0,0,0], scale=data_half*0.15, lw=4)
    draw_camera_axes(ax, t0_glb, R @ R_c2w_h0, scale=data_half*0.08)

    # 物体 bbox
    for i,(g) in enumerate(geoms):
        draw_bbox(ax, g['bbox_min'], g['bbox_max'], COLORS[i % 10], alpha=0.15)
        ax.scatter(*g['center'], c=COLORS[i % 10], s=60, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
        ax.text(g['center'][0], g['center'][1], g['center'][2]+0.02, g['name'],
                color=COLORS[i % 10], fontsize=7)

    # === 轨迹 ===
    plot_traj(ax, cam_glb, 'navy', 'HaWoR cam', lw=1.5, alpha=0.5)
    plot_traj(ax, ras_pos, 'limegreen', 'RAS cam', lw=1.5, alpha=0.8)
    plot_traj(ax, thumb_pts, 'darkorange' if hand_idx==1 else 'red', f'{side_label} ThumbTip')
    plot_traj(ax, wrist_pts, 'sienna' if hand_idx==1 else 'magenta', f'{side_label} Wrist')

    # === 缩放 (z >= 0) ===
    half = data_half * 1.5
    x_lo, x_hi = -half, half
    y_lo, y_hi = -half, half
    z_lo = max(0.0, lo[2] - half * 0.2)
    z_hi = hi[2] + half * 0.2
    ax.set_xlim3d(x_lo, x_hi); ax.set_ylim3d(y_lo, y_hi); ax.set_zlim3d(z_lo, z_hi)
    x_span = x_hi - x_lo; y_span = y_hi - y_lo; z_span = z_hi - z_lo
    try: ax.set_box_aspect([x_span, y_span, z_span])
    except: pass
    ticks_x = np.linspace(x_lo, x_hi, 13)
    ticks_y = np.linspace(y_lo, y_hi, 13)
    ticks_z = np.linspace(z_lo, z_hi, 13)
    ax.xaxis.set_major_locator(FixedLocator(ticks_x))
    ax.yaxis.set_major_locator(FixedLocator(ticks_y))
    ax.zaxis.set_major_locator(FixedLocator(ticks_z))

    ax.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
    plt.tight_layout()
    out_path = out_dir / 'alignment_quick.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f'\n  3D 可视化: {out_path}')

if __name__ == '__main__':
    main()
