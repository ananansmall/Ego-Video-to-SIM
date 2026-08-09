"""诊断: 对比 Original MANO palm vs Stage 3 mano_pos"""
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

def xform(pts, s, R, t):
    return s * (R @ pts.T).T + t

# ── Load HaWoR data ──
d = np.load('/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz', allow_pickle=True)
v = d['pred_valid'][0] & ~np.isnan(d['pred_trans'][0]).any(1) & ~np.isnan(d['pred_rot'][0]).any(1)
trans_v = d['pred_trans'][0][v]
rot_v = d['pred_rot'][0][v]
hp_v = d['pred_hand_pose'][0][v]
betas = d['pred_betas'][0][v].mean(0) if v.any() else np.zeros(10)
print(f"HaWoR MANO: {v.sum()} valid frames")

from mano_layer import MANOLayer
ml = MANOLayer('left', betas=betas)

# Compute both palm vertex 745 and MANO joints for frame 0
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
palm_745 = vn[0, 745, :]  # SLAM coordinates
print(f"\nFrame 0:")
print(f"  Palm vertex 745 (SLAM):  {palm_745.round(6)}")

if joints is not None:
    jn = joints.detach().cpu().numpy()
    print(f"  Joints shape: {jn.shape}")
    for j in range(min(21, jn.shape[1])):
        print(f"  Joint {j:2d}: {jn[0, j].round(6)}")

# Convert palm to GLB
palm_glb = xform(palm_745.reshape(1,3), SCALE, R_hand, T_H2G)[0]
print(f"\n  Palm vertex 745 GLB:     {palm_glb.round(6)}")

# Convert palm to SAPIEN (对齐 002 链)
# 002 _mano_to_sapien: sapien = R_AXIS @ (s * R_h2g @ Rx_hand @ slam + t_h2g) = R_AXIS @ p_glb
# p_glb = SCALE * R_hand @ palm_745 + T_H2G (已算作 palm_glb)
# 所以 palm_sapien = R_AXIS @ palm_glb
palm_sapien = (R_AXIS @ palm_glb).ravel()
print(f"  Palm vertex 745 SAPIEN:  {palm_sapien.round(6)}")

# ── Load Stage 3 mano_pos ──
s3 = np.load(os.path.join(OUT_DIR, 'stage3', 'stage3_result.npz'), allow_pickle=True)
r3 = s3['result'].item()
s3_mano = np.array(r3['mano_pos'])  # (113, 3) in SAPIEN
print(f"\nStage 3 mano_pos (SAPIEN):")
print(f"  Shape: {s3_mano.shape}")
print(f"  First: {s3_mano[0].round(6)}")
print(f"  Range: X=[{s3_mano[:,0].min():.4f},{s3_mano[:,0].max():.4f}] "
      f"Y=[{s3_mano[:,1].min():.4f},{s3_mano[:,1].max():.4f}] "
      f"Z=[{s3_mano[:,2].min():.4f},{s3_mano[:,2].max():.4f}]")

# Convert Stage 3 mano to GLB (对齐 002 链)
# 002 _mano_to_sapien: sapien = R_AXIS @ (s * R_h2g @ Rx_hand @ slam + t_h2g) = R_AXIS @ p_glb
# Inverse: p_glb = R_AXIS.T @ sapien
s3_mano_glb = (R_AXIS.T @ s3_mano.T).T
print(f"\nStage 3 mano GLB (对齐 002_render_scene.py):")
print(f"  First: {s3_mano_glb[0].round(6)}")
print(f"  Range: X=[{s3_mano_glb[:,0].min():.4f},{s3_mano_glb[:,0].max():.4f}] "
      f"Y=[{s3_mano_glb[:,1].min():.4f},{s3_mano_glb[:,1].max():.4f}] "
      f"Z=[{s3_mano_glb[:,2].min():.4f},{s3_mano_glb[:,2].max():.4f}]")

# Compare
print(f"\n{'='*60}")
print(f"COMPARISON (Frame 0):")
print(f"{'='*60}")
print(f"  Palm 745 GLB:        {palm_glb.round(6)}")
print(f"  Stage3 mano GLB:     {s3_mano_glb[0].round(6)}")
diff = s3_mano_glb[0] - palm_glb
print(f"  Difference:          {diff.round(6)}")
print(f"  Euclidean distance:  {np.linalg.norm(diff):.4f} m")

# Also check the wrist position
print(f"\n{'='*60}")
print(f"MANO joints vs palm 745 (SLAM space):")
if joints is not None:
    wrist = jn[0, 0]
    print(f"  Wrist (J0):          {wrist.round(6)}")
    print(f"  Palm 745:            {palm_745.round(6)}")
    print(f"  Wrist→Palm dist:     {np.linalg.norm(palm_745 - wrist):.4f} m")
    print(f"  J4 (index TIP):      {jn[0, 4].round(6)}")
    print(f"  J8 (middle TIP):     {jn[0, 8].round(6)}")
    print(f"  J4→J8 dist:          {np.linalg.norm(jn[0, 8] - jn[0, 4]):.4f} m")
    print(f"  Wrist→J4 dist:       {np.linalg.norm(jn[0, 4] - wrist):.4f} m")
    print(f"  Wrist→Palm745 dist:  {np.linalg.norm(palm_745 - wrist):.4f} m")