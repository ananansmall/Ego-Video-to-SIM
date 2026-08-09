#!/usr/bin/env python3
"""完整诊断: 追踪 MANO SLAM→SAPIEN→gripper base 的每一步"""
import sys, os
import numpy as np
import torch

sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')

TRANSFORM = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_only_left/alignment/transform_params.npz'
OUT_DIR = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_only_left'

# ── Transform params ──
p = np.load(TRANSFORM)
SCALE = float(p['scale_ratio'])
R_H2G = p['R_hand_to_glb']
T_H2G = p['t_hand_to_glb'].ravel()
RX = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=np.float64)
R_hand = R_H2G @ RX

print(f"SCALE = {SCALE}")
print(f"R_H2G =\n{R_H2G}")
print(f"T_H2G = {T_H2G}")
print(f"R_hand = R_H2G @ RX =\n{R_hand}")
print(f"R_AXIS =\n{R_AXIS}")

def xform(pts, s, R, t):
    return s * (R @ pts.T).T + t

def _mano_to_sapien(pts_slam):
    """MANO SLAM 坐标 → SAPIEN 坐标 (与 002_render_scene.py _render_to_sapien 一致)
    002 链: p_glb = s * R_h2g @ Rx_hand @ p_slam + t_h2g
           p_sapien = R_AXIS @ p_glb
    """
    pts_glb = SCALE * (R_hand @ pts_slam.T).T + T_H2G
    return (R_AXIS @ pts_glb.T).T

# ── Load HaWoR data ──
d = np.load('/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz', allow_pickle=True)
v = d['pred_valid'][0] & ~np.isnan(d['pred_trans'][0]).any(1) & ~np.isnan(d['pred_rot'][0]).any(1)
trans_v = d['pred_trans'][0][v]
rot_v = d['pred_rot'][0][v]
hp_v = d['pred_hand_pose'][0][v]
betas = d['pred_betas'][0][v].mean(0) if v.any() else np.zeros(10)
print(f"\nHaWoR MANO: {v.sum()} valid frames")

from mano_layer import MANOLayer
ml = MANOLayer('left', betas=betas)

# ── Compute MANO joints for frame 0 ──
i = 0
r = torch.from_numpy(rot_v[i:i+1].astype(np.float32))
hp = torch.from_numpy(hp_v[i:i+1].astype(np.float32))
t = torch.from_numpy(trans_v[i:i+1].astype(np.float32))
p = torch.cat([r, hp], dim=1)
vo = ml(p, t)
if isinstance(vo, (tuple, list)):
    vertices, joints = vo[0], vo[1]
else:
    vertices = vo
    joints = None

vn = vertices.detach().cpu().numpy()
jn = joints.detach().cpu().numpy() if joints is not None else None

print(f"\n{'='*70}")
print("STEP 1: MANO FK in SLAM space")
print(f"{'='*70}")
print(f"  J0 (wrist):      {jn[0,0].round(6)}")
print(f"  J4 (index TIP):  {jn[0,4].round(6)}")
print(f"  J8 (middle TIP): {jn[0,8].round(6)}")
print(f"  Vertex 745:      {vn[0,745].round(6)}")
print(f"  J4 == V745: {np.allclose(jn[0,4], vn[0,745])}")

# ── Transform to SAPIEN ──
j0_sapien = _mano_to_sapien(jn[0,0:1])[0]
j4_sapien = _mano_to_sapien(jn[0,4:5])[0]
j8_sapien = _mano_to_sapien(jn[0,8:9])[0]

print(f"\n{'='*70}")
print("STEP 2: _mano_to_sapien (SLAM → SAPIEN)")
print(f"{'='*70}")
print(f"  J0 SAPIEN:  {j0_sapien.round(6)}")
print(f"  J4 SAPIEN:  {j4_sapien.round(6)}")
print(f"  J8 SAPIEN:  {j8_sapien.round(6)}")

# ── Compute gripper base using data_loader.py's compute_analytical_gripper_pose ──
from data_loader import compute_analytical_gripper_pose

root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
    j0_sapien, j4_sapien, j8_sapien, prefix="left"
)

print(f"\n{'='*70}")
print("STEP 3: compute_analytical_gripper_pose (data_loader.py)")
print(f"{'='*70}")
print(f"  root_pos (gripper base): {root_pos.round(6)}")
print(f"  root_R:\n{root_R.round(6)}")
print(f"  j1={j1:.6f}, j2={j2:.6f}")

# ── Compare with Stage 3 mano_pos ──
s3 = np.load(os.path.join(OUT_DIR, 'stage3', 'stage3_result.npz'), allow_pickle=True)
r3 = s3['result'].item()
s3_mano = np.array(r3['mano_pos'])  # (113, 3) in SAPIEN

print(f"\n{'='*70}")
print("STEP 4: COMPARISON")
print(f"{'='*70}")
print(f"  Computed gripper base:  {root_pos.round(6)}")
print(f"  Stage 3 mano_pos[0]:    {s3_mano[0].round(6)}")
diff = s3_mano[0] - root_pos
print(f"  Difference:             {diff.round(6)}")
print(f"  Euclidean distance:     {np.linalg.norm(diff):.6f} m")

# ── Also compare J4 directly ──
print(f"\n  J4 SAPIEN:              {j4_sapien.round(6)}")
print(f"  Gripper base - J4:      {(root_pos - j4_sapien).round(6)}")
print(f"  Distance:               {np.linalg.norm(root_pos - j4_sapien):.6f} m")

# ── Check: what if we use the trajectory_loader.py version? ──
from trajectory_loader import compute_analytical_gripper_pose as compute_old
root_pos_old, root_R_old, j1_old, j2_old = compute_old(
    j0_sapien, j4_sapien, j8_sapien
)
print(f"\n{'='*70}")
print("COMPARISON: data_loader.py vs trajectory_loader.py compute")
print(f"{'='*70}")
print(f"  data_loader.py:       {root_pos.round(6)}")
print(f"  trajectory_loader.py: {root_pos_old.round(6)}")
print(f"  Difference:           {(root_pos_old - root_pos).round(6)}")
print(f"  Distance:             {np.linalg.norm(root_pos_old - root_pos):.6f} m")