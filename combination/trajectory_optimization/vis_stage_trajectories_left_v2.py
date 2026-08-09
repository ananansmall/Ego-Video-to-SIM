#!/usr/bin/env python3
"""
vis_stage_trajectories_left_v2.py — GLB space, matching render_quick.py exactly.
Shows: GLB objects + Original MANO + Stage 1/2/3 trajectories.
Everything in GLB world coordinates (Y-up).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import torch

sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')

# ── Paths ──
GLB = '/home/an/data/ras/7_vggt_omega/final_scene.glb'
TRANSFORM = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_only_left/alignment/transform_params.npz'
OUT_DIR = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_only_left'
OUT_PNG = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/View/stage_left_v2.png'

# ── Transform params (same as render_quick.py) ──
p = np.load(TRANSFORM)
SCALE = float(p['scale_ratio'])          # 3.168
R_H2G = p['R_hand_to_glb']               # 3x3
T_H2G = p['t_hand_to_glb'].ravel()       # 3
RX = np.diag([1.0, -1.0, -1.0])          # Rx_hand (same as render_quick.py)
R_AXIS = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float64)

def xform(pts, s, R, t):
    """Same as render_quick.py: s * (R @ pts.T).T + t"""
    return s * (R @ pts.T).T + t

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════

# ── GLB objects (same as render_quick.py: use geom.vertices directly, no node transform) ──
scene = trimesh.load(GLB, force='scene')
# Build geom_name -> node_name mapping (same as render_quick.py)
geom_to_node = {}
for node_name in scene.graph.nodes_geometry:
    result = scene.graph.get(node_name)
    if result:
        _, geometry_name = result
        geom_to_node[geometry_name] = node_name

geoms = []
for name, geom in scene.geometry.items():
    center = geom.vertices.mean(0)
    bounds = geom.bounds
    display_name = geom_to_node.get(name, name)
    geoms.append((display_name, center, bounds))
    print(f"  Object {display_name}: center={center.round(4)}, bounds={bounds.round(4)}")

# ── HaWoR MANO reference (same as render_quick.py: xform with s, R_hand, t) ──
d = np.load('/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz', allow_pickle=True)
v = d['pred_valid'][0] & ~np.isnan(d['pred_trans'][0]).any(1) & ~np.isnan(d['pred_rot'][0]).any(1)
trans_v = d['pred_trans'][0][v]
rot_v = d['pred_rot'][0][v]
hp_v = d['pred_hand_pose'][0][v]
betas = d['pred_betas'][0][v].mean(0) if v.any() else np.zeros(10)
print(f"HaWoR MANO: {v.sum()} valid frames")

from mano_layer import MANOLayer
ml = MANOLayer('left', betas=betas)
mano_palm = []
for i in range(len(trans_v)):
    r = torch.from_numpy(rot_v[i:i+1].astype(np.float32))
    hp = torch.from_numpy(hp_v[i:i+1].astype(np.float32))
    t = torch.from_numpy(trans_v[i:i+1].astype(np.float32))
    p = torch.cat([r, hp], dim=1)
    vo = ml(p, t)
    if isinstance(vo, (tuple, list)): vo = vo[0]
    vn = vo.detach().cpu().numpy()
    mano_palm.append(vn[:, 745, :] if vn.shape[1] >= 745 else vn[:, 0, :])
mano_palm = np.concatenate(mano_palm, axis=0)  # HaWoR SLAM

# Same transform as render_quick.py: R_hand = R_h2g @ Rx_hand
R_hand = R_H2G @ RX
mano_palm_glb = xform(mano_palm, SCALE, R_hand, T_H2G)  # (N, 3) in GLB world
print(f"  Original MANO palm GLB: first={mano_palm_glb[0].round(4)}, last={mano_palm_glb[-1].round(4)}")
print(f"  Y range: {mano_palm_glb[:,1].min():.4f} ~ {mano_palm_glb[:,1].max():.4f}")

# ── Stage 1/2/3 data (SAPIEN -> GLB world) ──
# 统一 SAPIEN 空间 (对齐 001/002 链):
#   002 data_loader: sapien = R_AXIS @ (s_inv * R_h2g.T @ (v - t_h2g))
#   002 _mano_to_sapien: sapien = R_AXIS @ (s * R_h2g @ Rx_hand @ slam + t_h2g)
# 两者都用 R_AXIS (001 链 R_x=I), 在 SAPIEN 中同帧.
#
s_inv = 1.0 / SCALE

def mano_sapien_to_glb(pts):
    """SAPIEN -> GLB: inverse of _mano_to_sapien (对齐 002 链)
    _mano_to_sapien: sapien = R_AXIS @ (s * R_H2G @ RX @ slam + T_H2G) = R_AXIS @ p_glb
    Inverse: p_glb = R_AXIS.T @ sapien  (R_AXIS 正交)
    """
    pts = np.asarray(pts)
    if pts.ndim == 1:
        return R_AXIS.T @ pts
    return (R_AXIS.T @ pts.T).T

def sapien_to_glb(pts):
    """SAPIEN -> GLB: inverse of data_loader 002 链
    002 forward: sapien = R_AXIS @ (s_inv * R_H2G.T @ (v - T_H2G))
    => R_AXIS.T @ sapien = s_inv * R_H2G.T @ (v - T_H2G)
    => R_H2G @ R_AXIS.T @ sapien = s_inv * (v - T_H2G)
    => v = T_H2G + SCALE * R_H2G @ R_AXIS.T @ sapien
    """
    pts = np.asarray(pts)
    if pts.ndim == 1:
        return T_H2G + SCALE * (R_H2G @ (R_AXIS.T @ pts))
    return T_H2G + SCALE * (R_H2G @ (R_AXIS.T @ pts.T)).T

# Stage 1 grasp (in data_loader space)
s1 = np.load(os.path.join(OUT_DIR, 'stage1', 'best_grasp.npz'), allow_pickle=True)
s1_pos = np.array(s1['pos'])
s1_glb = sapien_to_glb(s1_pos)
print(f"Stage1 grasp GLB: {s1_glb.round(4)}")

# Stage 2 lift (in data_loader space)
s2 = np.load(os.path.join(OUT_DIR, 'stage2', 'stage2_recon.npz'), allow_pickle=True)
s2_pos = s2['pos']
s2_frames = s2['frames']
s2_glb = sapien_to_glb(s2_pos)
print(f"Stage2 GLB: {s2_glb.shape}, F{s2_frames[0]}-F{s2_frames[-1]}")
print(f"  Y range: {s2_glb[:,1].min():.4f} ~ {s2_glb[:,1].max():.4f}")

# Stage 3 data
s3 = np.load(os.path.join(OUT_DIR, 'stage3', 'stage3_result.npz'), allow_pickle=True)
r3 = s3['result'].item()
s3_mano = np.array(r3['mano_pos'])  # Original MANO gripper in optimization (_mano_to_sapien space)
s3_opt = np.array(r3['opt_pos'])    # Optimized gripper (data_loader space, with offset)
s3_obj = np.array(r3['obj_pos_traj'])  # Object position (data_loader space)

# Use correct inverse for each space
s3_mano_glb = mano_sapien_to_glb(s3_mano)  # raw MANO -> _mano_to_sapien inverse
s3_opt_glb = sapien_to_glb(s3_opt)          # optimized -> data_loader inverse
s3_obj_glb = sapien_to_glb(s3_obj)          # object -> data_loader inverse

print(f"Stage3 mano GLB: {s3_mano_glb.shape}, Y range: {s3_mano_glb[:,1].min():.4f} ~ {s3_mano_glb[:,1].max():.4f}")
print(f"Stage3 opt GLB: {s3_opt_glb.shape}, Y range: {s3_opt_glb[:,1].min():.4f} ~ {s3_opt_glb[:,1].max():.4f}")
print(f"Stage3 object GLB: {s3_obj_glb.shape}, Y range: {s3_obj_glb[:,1].min():.4f} ~ {s3_obj_glb[:,1].max():.4f}")

# ── Debug: compare Original MANO vs Stage 3 mano in same space ──
print("\n  --- MANO comparison (should be close) ---")
print(f"  Original MANO palm GLB:  first={mano_palm_glb[0].round(4)}, Y=[{mano_palm_glb[:,1].min():.3f},{mano_palm_glb[:,1].max():.3f}]")
print(f"  Stage3 mano GLB:         first={s3_mano_glb[0].round(4)}, Y=[{s3_mano_glb[:,1].min():.3f},{s3_mano_glb[:,1].max():.3f}]")
# Also convert Original MANO to SAPIEN for comparison (对齐 002 链)
# mano_palm_glb = SCALE * R_H2G @ RX @ mano_palm + T_H2G  (GLB 空间)
# 002 _mano_to_sapien: sapien = R_AXIS @ (SCALE * R_H2G @ RX @ slam + T_H2G) = R_AXIS @ mano_palm_glb
mano_palm_sapien = (R_AXIS @ mano_palm_glb.T).T
print(f"  Original MANO palm SAPIEN: first={mano_palm_sapien[0].round(4)}, Z=[{mano_palm_sapien[:,2].min():.3f},{mano_palm_sapien[:,2].max():.3f}]")
print(f"  Stage3 mano SAPIEN (raw):  first={s3_mano[0].round(4)}, Z=[{s3_mano[:,2].min():.3f},{s3_mano[:,2].max():.3f}]")

# ═══════════════════════════════════════════════════════════════
# 2. PRINT SCALE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SCALE ANALYSIS (GLB world, Y-up)")
print("="*70)

all_pts = np.vstack([mano_palm_glb, s1_glb.reshape(1,3), s2_glb, s3_mano_glb, s3_opt_glb, s3_obj_glb])
print(f"  Overall range: X=[{all_pts[:,0].min():.4f},{all_pts[:,0].max():.4f}] "
      f"Y=[{all_pts[:,1].min():.4f},{all_pts[:,1].max():.4f}] "
      f"Z=[{all_pts[:,2].min():.4f},{all_pts[:,2].max():.4f}]")
print(f"  Overall extent: {all_pts.max(0)-all_pts.min(0)} (m)")

# ═══════════════════════════════════════════════════════════════
# 3. PLOT — Single 3D view, GLB space (Y-up), matching render_quick.py style
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')

# ── GLB objects (same as render_quick.py: draw_bbox with geom.bounds) ──
obj_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#17becf']
for i, (nm, ct, bd) in enumerate(geoms):
    if 'grid' in nm.lower():  # skip ground grid
        continue
    c = obj_colors[i % len(obj_colors)]
    p0, p1 = bd
    # Draw bbox (same as render_quick.py draw_bbox)
    corners = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                        [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
    for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot3D(*corners[[e[0],e[1]]].T, color=c, alpha=0.15, lw=1.0)
    ax.scatter(*ct, c=c, s=60, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
    ax.text(ct[0], ct[1], ct[2]+0.05, nm, color=c, fontsize=7)

# ── World axes (same as render_quick.py draw_axes_arrows) ──
AX_LEN = 0.5
origin = np.array([0,0,0])
for c, lbl, v in [('red','X',[AX_LEN,0,0]),('green','Y',[0,AX_LEN,0]),('blue','Z',[0,0,AX_LEN])]:
    tip = origin + v
    ax.plot([origin[0],tip[0]],[origin[1],tip[1]],[origin[2],tip[2]], color=c, lw=3, alpha=0.9, zorder=10)
    ax.text(tip[0]+v[0]*0.1, tip[1]+v[1]*0.1, tip[2]+v[2]*0.1, lbl, color=c, fontsize=12, fontweight='bold')

# ── Original MANO (HaWoR FK palm, same transform as render_quick.py) ──
ax.plot(mano_palm_glb[:,0], mano_palm_glb[:,1], mano_palm_glb[:,2], '-', color='magenta', lw=2.5, alpha=0.8, zorder=6,
        label=f'Original MANO (HaWoR FK, palm, n={len(mano_palm_glb)})')
ax.scatter(*mano_palm_glb[0], c='magenta', s=40, marker='o', zorder=7)
ax.scatter(*mano_palm_glb[-1], c='magenta', s=80, marker='*', zorder=7, edgecolors='k')
ax.text(mano_palm_glb[-1,0], mano_palm_glb[-1,1], mano_palm_glb[-1,2], 'MANO_end', fontsize=7, color='magenta')

# ── Stage 3 mano_pos (original MANO gripper in optimization, same SAPIEN->GLB) ──
ax.plot(s3_mano_glb[:,0], s3_mano_glb[:,1], s3_mano_glb[:,2], '--', color='deeppink', lw=1.5, alpha=0.7, zorder=5,
        label=f'Stage3 MANO gripper (orig, n={len(s3_mano_glb)})')

# ── Stage 1 grasp ──
ax.scatter(*s1_glb, c='green', s=200, marker='D', edgecolors='k', linewidths=1.5, zorder=10)
ax.text(s1_glb[0], s1_glb[1], s1_glb[2]+0.03, 'S1 Grasp', fontsize=9, color='darkgreen', fontweight='bold', ha='center')

# ── Stage 2 lift ──
ax.plot(s2_glb[:,0], s2_glb[:,1], s2_glb[:,2], '-', color='darkorange', lw=2.5, alpha=0.9, zorder=6,
        label=f'Stage 2 Lift (F{s2_frames[0]}-F{s2_frames[-1]})')
ax.scatter(*s2_glb[0], c='darkorange', s=80, marker='^', zorder=8, edgecolors='k')
ax.scatter(*s2_glb[-1], c='darkorange', s=80, marker='v', zorder=8, edgecolors='k')
ax.text(s2_glb[0,0], s2_glb[0,1], s2_glb[0,2]-0.03, 'S2_start', fontsize=7, color='darkorange', ha='center')
ax.text(s2_glb[-1,0], s2_glb[-1,1], s2_glb[-1,2]-0.03, 'S2_end', fontsize=7, color='darkorange', ha='center')

# ── Stage 3 Opt ──
ax.plot(s3_opt_glb[:,0], s3_opt_glb[:,1], s3_opt_glb[:,2], '-', color='royalblue', lw=2.5, alpha=0.9, zorder=6,
        label=f'Stage 3 Opt (F0-F{len(s3_opt_glb)-1})')
ax.scatter(*s3_opt_glb[0], c='royalblue', s=30, marker='o', zorder=7)
ax.scatter(*s3_opt_glb[-1], c='royalblue', s=100, marker='*', zorder=8, edgecolors='k')
ax.text(s3_opt_glb[-1,0], s3_opt_glb[-1,1], s3_opt_glb[-1,2]+0.03, 'S3_end', fontsize=7, color='royalblue', fontweight='bold', ha='center')

# ── Stage 3 object trajectory ──
ax.plot(s3_obj_glb[:,0], s3_obj_glb[:,1], s3_obj_glb[:,2], ':', color='brown', lw=2.0, alpha=0.8, zorder=5,
        label=f'Object (simulation)')

# ── Axis limits ──
all_plot = np.vstack([mano_palm_glb, s1_glb.reshape(1,3), s2_glb, s3_mano_glb, s3_opt_glb, s3_obj_glb])
pad = 0.15
ax.set_xlim(all_plot[:,0].min()-pad, all_plot[:,0].max()+pad)
ax.set_ylim(all_plot[:,1].min()-pad, all_plot[:,1].max()+pad)
ax.set_zlim(all_plot[:,2].min()-pad, all_plot[:,2].max()+pad)

ax.set_xlabel('GLB X (m)', fontsize=10)
ax.set_ylabel('GLB Y (m, up)', fontsize=10)
ax.set_zlabel('GLB Z (m)', fontsize=10)
ax.set_title(f'Left Hand: Original MANO vs Stage 1/2/3 (GLB space, Y-up)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', framealpha=0.9, ncol=2)
ax.view_init(elev=20, azim=45)
ax.set_box_aspect([1,1,0.8])

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {OUT_PNG}")