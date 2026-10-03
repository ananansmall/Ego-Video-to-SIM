#!/usr/bin/env python3
"""Debug script: dump all pose vectors for manual inspection."""
import sys
import numpy as np
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/combination")
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/combination/hand_track")
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting")

def matrix_from_compact_axis_angle(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    c, s = np.cos(angle), np.sin(angle)
    cx, cy, cz = axis
    return np.array([
        [c + cx*cx*(1-c), cx*cy*(1-c) - cz*s, cx*cz*(1-c) + cy*s],
        [cy*cx*(1-c) + cz*s, c + cy*cy*(1-c), cy*cz*(1-c) - cx*s],
        [cz*cx*(1-c) - cy*s, cz*cy*(1-c) + cx*s, c + cz*cz*(1-c)]
    ])

def compute_mano_joints(rot, hp, trans, betas):
    from mano_layer import MANOLayer
    import torch
    ml = MANOLayer("right", betas)
    p = torch.from_numpy(np.concatenate([rot, hp]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = ml(p, t)
    return j.detach().cpu().numpy()[0]

from gripper_config import compute_analytical_gripper_pose

# Load frame 50
import hawor_loader
data = hawor_loader.load_hawor_data("/home/an/data/hawor/7", hand_idx=1, frame_idx=50)

rot = data["pred_rot"]
trans = data["pred_trans"]
hp = data["pred_hand_pose"]
betas = data["pred_betas"]
j = compute_mano_joints(rot, hp, trans, betas)

wrist = j[0]
thumb = j[4]
index = j[8]

print("="*70)
print("RAW MANO VECTORS")
print("="*70)
print(f"  wrist   (j0) = {wrist}")
print(f"  thumb   (j4) = {thumb}")
print(f"  index   (j8) = {index}")
print(f"  wrist->thumb = {thumb - wrist}")
print(f"  wrist->index = {index - wrist}")
midpoint = (thumb + index) / 2
print(f"  midpoint(ti) = {midpoint}")
print(f"  wrist->midpoint = {midpoint - wrist}")
print(f"  index - thumb = {index - thumb}")

print("\n" + "="*70)
print("MANO HAND ROTATION (pred_rot)")
print("="*70)
R_mano = matrix_from_compact_axis_angle(rot)
print(f"  pred_rot (axis-angle) = {rot}")
print(f"  R_mano[:,0] (X, palm forward?) = {R_mano[:,0]}")
print(f"  R_mano[:,1] (Y)                      = {R_mano[:,1]}")
print(f"  R_mano[:,2] (Z)                      = {R_mano[:,2]}")

print("\n" + "="*70)
print("ANALYTICAL GRIPPER POSE (compute_analytical_gripper_pose)")
print("="*70)
g_pos, g_R, joint1, joint2 = compute_analytical_gripper_pose(wrist, thumb, index, "right")
print(f"  gripper_pos  = {g_pos}")
print(f"  g_R[:,0] (X, pointing) = {g_R[:,0]}")
print(f"  g_R[:,1] (Y, opening)  = {g_R[:,1]}")
print(f"  g_R[:,2] (Z)             = {g_R[:,2]}")
print(f"  joint1={joint1:.4f}, joint2={joint2:.4f}")
print(f"  det(R) = {np.linalg.det(g_R):.6f}")

# What the URDF fingers look like
f1_w = g_pos + g_R @ np.array([0.03689, -0.013453 - joint1, -0.00012053])
f2_w = g_pos + g_R @ np.array([0.03689, 0.013453 + joint2, 0.00012067])
print(f"  analytical finger1 tip (world) = {f1_w}")
print(f"  analytical finger2 tip (world) = {f2_w}")
print(f"  dist to thumb/j4 = {np.linalg.norm(f1_w - thumb):.6f}")
print(f"  dist to index/j8 = {np.linalg.norm(f2_w - index):.6f}")

print("\n" + "="*70)
print("RETARGETING GRIPPER POSE (optimizer FK)")
print("="*70)
from gripper_config import init_gripper_retargeting
retargeting, ref_indices, _ = init_gripper_retargeting("right", 0.03689, "/home/an/robot_world_ws/src")
print(f"  ref_indices = {ref_indices}")
ref_value = j[ref_indices, :3].astype(np.float32)
num_fixed = len(retargeting.optimizer.idx_pin2fixed)
fixed_qpos = np.zeros(num_fixed, dtype=np.float32)
retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

internal_robot = retargeting.optimizer.robot
internal_robot.compute_forward_kinematics(retarget_qpos)
for i, name in enumerate(internal_robot.link_names):
    if name == "right_gripper_link":
        pose44 = internal_robot.get_link_pose(i)
        opt_pos = pose44[:3, 3].copy()
        opt_R = pose44[:3, :3].copy()
        print(f"  gripper_pos (opt)     = {opt_pos}")
        print(f"  opt_R[:,0] (X) = {opt_R[:,0]}")
        print(f"  opt_R[:,1] (Y) = {opt_R[:,1]}")
        print(f"  opt_R[:,2] (Z) = {opt_R[:,2]}")
        print(f"  det(R) = {np.linalg.det(opt_R):.6f}")

        # Get finger link positions
        for j2, name2 in enumerate(internal_robot.link_names):
            if "finger_link1" in name2 or "finger_link2" in name2:
                fp = internal_robot.get_link_pose(j2)[:3,3]
                print(f"  {name2} world pos = {fp}")

print(f"\n  retarget_qpos = {retarget_qpos}")
print(f"  dist (ana base -> opt base) = {np.linalg.norm(g_pos - opt_pos):.4f}")
