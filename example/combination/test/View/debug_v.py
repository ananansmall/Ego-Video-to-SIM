#!/usr/bin/env python3
"""Debug: verify gripper pointing directions are correct."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/combination')
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/combination/hand_track')
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')
from gripper_config import compute_analytical_gripper_pose, init_gripper_retargeting, FINGER_GEOM_ARRAYS
from mano_layer import MANOLayer

rec_dir = Path('/home/an/data/hawor/7/reconstruction')
files = sorted(rec_dir.glob('hawor_results_*_depth_aligned.npz'), reverse=True)
rec = np.load(str(files[0]), allow_pickle=True)
valid = np.where(rec['pred_valid'][1])[0]
fi = valid[min(50, len(valid)-1)]
ml = MANOLayer('right', rec['pred_betas'][1][fi])
p = torch.from_numpy(np.concatenate([rec['pred_rot'][1][fi], rec['pred_hand_pose'][1][fi]]).astype(np.float32)).unsqueeze(0)
t = torch.from_numpy(rec['pred_trans'][1][fi].astype(np.float32)).unsqueeze(0)
_, j = ml(p, t)
j = j[0].numpy()
wrist, thumb, index = j[0], j[4], j[8]
mid = (thumb + index) / 2
w2m = (mid - wrist) / np.linalg.norm(mid - wrist)

def mat_from_aa(aa):
    a = np.linalg.norm(aa)
    if a < 1e-8: return np.eye(3)
    axis = aa/a; c,s = np.cos(a),np.sin(a)
    cx,cy,cz = axis
    return np.array([[c+cx*cx*(1-c),cx*cy*(1-c)-cz*s,cx*cz*(1-c)+cy*s],
                     [cy*cx*(1-c)+cz*s,c+cy*cy*(1-c),cy*cz*(1-c)-cx*s],
                     [cz*cx*(1-c)-cy*s,cz*cy*(1-c)+cx*s,c+cz*cz*(1-c)]])

# --- #1 MANO ---
R_mano = mat_from_aa(rec['pred_rot'][1][fi])
print("=== #1 MANO hand pose ===")
print(f"  wrist->fingertips = {w2m}")
print(f"  R_mano[:,0]       = {R_mano[:,0]}")
print(f"  dot(R_mano X, w->m) = {np.dot(R_mano[:,0], w2m):.4f}")

# --- #2 Analytical gripper ---
g_pos, g_R, j1, j2 = compute_analytical_gripper_pose(wrist, thumb, index, "right")
# compute analytical fingertips
fg = FINGER_GEOM_ARRAYS["right"]
f1_loc = fg["finger1_origin"].copy(); f1_loc[0] = 0.03689; f1_loc = f1_loc + fg["finger1_axis"]*j1
f2_loc = fg["finger2_origin"].copy(); f2_loc[0] = 0.03689; f2_loc = f2_loc + fg["finger2_axis"]*j2
ana_f1 = g_pos + g_R @ f1_loc
ana_f2 = g_pos + g_R @ f2_loc
ana_mid = (ana_f1 + ana_f2) / 2
ana_true_dir = (ana_mid - g_pos) / np.linalg.norm(ana_mid - g_pos)
print(f"\n=== #2 Analytical gripper ===")
print(f"  gripper_pos     = {g_pos}")
print(f"  g_R[:,0]        = {g_R[:,0]}")
print(f"  ana_fingertips  = {ana_f1}, {ana_f2}")
print(f"  base->mid vector= {ana_true_dir}")
print(f"  dot(g_R X, base->mid) = {np.dot(g_R[:,0], ana_true_dir):.4f}")
print(f"  dot(g_R X, w->m)      = {np.dot(g_R[:,0], w2m):.4f}")
print(f"  dist(f1, thumb) = {np.linalg.norm(ana_f1 - thumb):.4f}")
print(f"  dist(f2, index) = {np.linalg.norm(ana_f2 - index):.4f}")

# --- #3 Retargeting gripper ---
ret, ri, _ = init_gripper_retargeting('right', 0.03689, '/home/an/robot_world_ws/src')
ref = j[ri,:3].astype(np.float32)
fq = np.zeros(len(ret.optimizer.idx_pin2fixed), dtype=np.float32)
qpos = ret.retarget(ref, fq)
robot = ret.optimizer.robot
robot.compute_forward_kinematics(qpos)
opt_pos, opt_R, opt_f1, opt_f2 = None, None, None, None
for i, name in enumerate(robot.link_names):
    pose = robot.get_link_pose(i)
    pos, R = pose[:3,3].copy(), pose[:3,:3].copy()
    if name == "right_gripper_link": opt_pos, opt_R = pos, R
    if "finger_link1" in name: opt_f1 = pos
    if "finger_link2" in name: opt_f2 = pos

opt_mid = (opt_f1 + opt_f2) / 2
opt_true_dir = (opt_mid - opt_pos) / np.linalg.norm(opt_mid - opt_pos)
print(f"\n=== #3 Retargeting gripper ===")
print(f"  gripper_pos      = {opt_pos}")
print(f"  gripper_link X   = {opt_R[:,0]}")
print(f"  opt_fingertips   = {opt_f1}, {opt_f2}")
print(f"  base->mid vector = {opt_true_dir}")
print(f"  dot(opt FK X, base->mid) = {np.dot(opt_R[:,0], opt_true_dir):.4f}")
print(f"  dot(opt true dir, w->m)  = {np.dot(opt_true_dir, w2m):.4f}")
print(f"  dot(ana g_R X, opt FK X) = {np.dot(g_R[:,0], opt_R[:,0]):.4f}")
print(f"  dist(opt_f1, thumb) = {np.linalg.norm(opt_f1 - thumb):.4f}")
print(f"  dist(opt_f2, index) = {np.linalg.norm(opt_f2 - index):.4f}")
