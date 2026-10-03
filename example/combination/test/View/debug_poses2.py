#!/usr/bin/env python3
"""Debug: print all vectors to verify correctness."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/combination")
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/combination/hand_track")
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting")

from gripper_config import compute_analytical_gripper_pose, init_gripper_retargeting, FINGER_GEOM_ARRAYS

def matrix_from_compact_axis_angle(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-8: return np.eye(3)
    axis = aa / angle; c, s = np.cos(angle), np.sin(angle)
    cx, cy, cz = axis
    return np.array([
        [c+cx*cx*(1-c), cx*cy*(1-c)-cz*s, cx*cz*(1-c)+cy*s],
        [cy*cx*(1-c)+cz*s, c+cy*cy*(1-c), cy*cz*(1-c)-cx*s],
        [cz*cx*(1-c)-cy*s, cz*cy*(1-c)+cx*s, c+cz*cz*(1-c)]
    ])

def compute_mano_joints(rot, hp, trans, betas):
    from mano_layer import MANOLayer
    import torch
    ml = MANOLayer("right", betas)
    p = torch.from_numpy(np.concatenate([rot, hp]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = ml(p, t)
    return j.detach().cpu().numpy()[0]

# Load frame 50
rec_dir = Path("/home/an/data/hawor/7/reconstruction")
files = sorted(rec_dir.glob("hawor_results_*_depth_aligned.npz"), reverse=True)
rec = np.load(str(files[0]), allow_pickle=True)
valid = np.where(rec['pred_valid'][1])[0]
fi = valid[min(50, len(valid)-1)]
rot = rec['pred_rot'][1][fi]; trans = rec['pred_trans'][1][fi]
hp = rec['pred_hand_pose'][1][fi]; betas = rec['pred_betas'][1][fi]
j = compute_mano_joints(rot, hp, trans, betas)
wrist, thumb, index = j[0], j[4], j[8]
midpoint = (thumb + index) / 2

print("="*70)
print("MANO JOINTS")
print("="*70)
print(f"  wrist (j0)  = {wrist}")
print(f"  thumb (j4)  = {thumb}")
print(f"  index (j8)  = {index}")
print(f"  midpoint    = {midpoint}")
print(f"  wrist->mid  = {midpoint - wrist}")
print()

# === #1 MANO hand pose ===
R_mano = matrix_from_compact_axis_angle(rot)
v_mid = (midpoint - wrist) / np.linalg.norm(midpoint - wrist)
print("="*70)
print("#1 MANO hand pose (pred_rot)")
print("="*70)
print(f"  R_mano[:,0] = {R_mano[:,0]}")
print(f"  dot(X, w->m)= {np.dot(R_mano[:,0], v_mid):.4f}")
print(f"  -> X points palm-forward, which is OPPOSITE to wrist->fingertips")
print()

# === #2 Analytical gripper ===
print("="*70)
print("#2 Analytical gripper (compute_analytical_gripper_pose)")
print("="*70)
g_pos, g_R, joint1, joint2 = compute_analytical_gripper_pose(wrist, thumb, index, "right")
print(f"  gripper_pos = {g_pos}")
print(f"  g_R[:,0] (X) = {g_R[:,0]}")
print(f"  g_R[:,1] (Y) = {g_R[:,1]}")
print(f"  g_R[:,2] (Z) = {g_R[:,2]}")
print(f"  joint1={joint1:.4f}, joint2={joint2:.4f}")

# Correct finger positions (using FINGER_GEOM_ARRAYS)
fg = FINGER_GEOM_ARRAYS["right"]
f1_local = fg["finger1_origin"].copy()
f1_local[0] = 0.03689
f1_local = f1_local + fg["finger1_axis"] * joint1
f2_local = fg["finger2_origin"].copy()
f2_local[0] = 0.03689
f2_local = f2_local + fg["finger2_axis"] * joint2
f1_w = g_pos + g_R @ f1_local
f2_w = g_pos + g_R @ f2_local
print(f"  finger1_world = {f1_w}")
print(f"  finger2_world = {f2_w}")
print(f"  dist(f1, thumb/j4) = {np.linalg.norm(f1_w - thumb):.4f}")
print(f"  dist(f2, index/j8) = {np.linalg.norm(f2_w - index):.4f}")
print(f"  finger separation direction = {(f1_w - f2_w) / np.linalg.norm(f1_w - f2_w)}")
print(f"  thumb-index direction        = {(thumb - index) / np.linalg.norm(thumb - index)}")
print()

# === #3 Retargeting gripper ===
print("="*70)
print("#3 Retargeting gripper (init_gripper_retargeting + retarget)")
print("="*70)
ret, ri, _ = init_gripper_retargeting("right", 0.03689, "/home/an/robot_world_ws/src")
print(f"  ref_indices = {ri}")
ref_value = j[ri, :3].astype(np.float32)
num_fixed = len(ret.optimizer.idx_pin2fixed)
fixed_qpos = np.zeros(num_fixed, dtype=np.float32)
qpos = ret.retarget(ref_value, fixed_qpos)
print(f"  retarget_qpos = {qpos}")

robot = ret.optimizer.robot
robot.compute_forward_kinematics(qpos)
opt_pos, opt_R = None, None
for i, name in enumerate(robot.link_names):
    pose44 = robot.get_link_pose(i)
    pos = pose44[:3, 3].copy()
    R = pose44[:3, :3].copy()
    if name == "right_gripper_link":
        opt_pos, opt_R = pos, R
        print(f"  gripper_link pos = {pos}")
        print(f"  gripper_link X   = {R[:,0]}")
        print(f"  gripper_link Y   = {R[:,1]}")
        print(f"  gripper_link Z   = {R[:,2]}")
    if "finger" in name:
        print(f"  {name} pos = {pos}")
        if "finger_link1" in name:
            r1_dist = np.linalg.norm(pos - thumb)
            print(f"    dist to thumb/j4 = {r1_dist:.4f}")
        if "finger_link2" in name:
            r2_dist = np.linalg.norm(pos - index)
            print(f"    dist to index/j8 = {r2_dist:.4f}")

print(f"\n  dist(opt_gripper_link, wrist) = {np.linalg.norm(opt_pos - wrist):.4f}")

print()
print("="*70)
print("DIRECTION COMPARISON")
print("="*70)
print(f"  #1 R_mano X . (wrist->mid) = {np.dot(R_mano[:,0], v_mid):.4f} (opposite = expected)")
print(f"  #2 ana X    . (wrist->mid) = {np.dot(g_R[:,0], v_mid):.4f} (aligned = correct)")
print(f"  #3 opt X    . (wrist->mid) = {np.dot(opt_R[:,0], v_mid):.4f}")
print(f"  #2 ana X . #3 opt X        = {np.dot(g_R[:,0], opt_R[:,0]):.4f}")
print(f"  gap (ana base - opt base)  = {np.linalg.norm(g_pos - opt_pos):.4f}")
