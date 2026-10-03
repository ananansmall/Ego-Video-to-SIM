#!/usr/bin/env python3
"""Full debug: print FK chain of gripper retargeting."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/combination')
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/combination/hand_track')
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')

from gripper_config import init_gripper_retargeting

# Load joints
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

ret, ri, _ = init_gripper_retargeting('right', 0.03689, '/home/an/robot_world_ws/src')
print(f'dof_joint_names = {ret.optimizer.robot.dof_joint_names}')
print(f'ref_indices = {ri}')
print(f'idx_pin2target = {ret.optimizer.idx_pin2target}')
print(f'idx_pin2fixed  = {ret.optimizer.idx_pin2fixed}')

ref = j[ri,:3].astype(np.float32)
num_fixed = len(ret.optimizer.idx_pin2fixed)
fq = np.zeros(num_fixed, dtype=np.float32)
qpos = ret.retarget(ref, fq)

robot = ret.optimizer.robot
robot.compute_forward_kinematics(qpos)

print(f'\nqpos (all {len(qpos)} joints):')
for name, val in zip(robot.dof_joint_names, qpos):
    print(f'  {name:40s} = {val:.6f}')

print(f'\nTarget values from optimizer:')
target_vals = qpos[ret.optimizer.idx_pin2target]
print(f'  {target_vals}')
print(f'  Fixed values:')
fixed_vals = qpos[ret.optimizer.idx_pin2fixed]
print(f'  {fixed_vals}')

print(f'\n--- Full FK chain (all links) ---')
for i, name in enumerate(robot.link_names):
    pose = robot.get_link_pose(i)
    pos = pose[:3,3]
    R = pose[:3,:3]
    print(f'  [{i:2d}] {name:40s} pos={pos}  det(R)={np.linalg.det(R):.4f}')
    if 'gripper_link' == name.split('_')[0] or 'finger' in name or 'gripper_link' in name:
        print(f'       X={R[:,0]}  Y={R[:,1]}  Z={R[:,2]}')

print(f'\nTarget check:')
print(f'  j[4] (thumb) = {j[4]}')
print(f'  j[8] (index) = {j[8]}')
print(f'  j[0] (wrist) = {j[0]}')
