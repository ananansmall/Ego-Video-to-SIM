#!/usr/bin/env python3
import numpy as np
import trimesh
from pathlib import Path
import torch
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))

R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

params = np.load('output/alignment/transform_params.npz')
s_inv = float(params['s_inv'])
R_inv = params['R_inv']
t_inv = params['t_inv']

glb_path = '/home/an/data/ras/my_7mp4_result/final_scene.glb'
scene = trimesh.load(glb_path)
all_verts_sapien = []
for name, geom in scene.geometry.items():
    v = geom.vertices.copy()
    v_hawor = s_inv * (R_inv @ v.T).T + t_inv
    v_sapien = (RXWORLD_TO_SAPIEN @ v_hawor.T).T
    all_verts_sapien.append(v_sapien)
all_verts_sapien = np.vstack(all_verts_sapien)
glb_center = all_verts_sapien.mean(axis=0)
glb_min = all_verts_sapien.min(axis=0)
glb_max = all_verts_sapien.max(axis=0)

print('=== GLB 在 SAPIEN 坐标系 ===')
print(f'  中心: {glb_center}')
print(f'  边界: [{glb_min}] ~ [{glb_max}]')
print(f'  尺寸: {glb_max - glb_min}')

from mano_layer import MANOLayer

hawor_dir = Path('/home/an/data/hawor/7')
hawor_data = dict(np.load(hawor_dir / 'reconstruction' / 'hawor_results_0_113.npz', allow_pickle=True))

hand_idx = 0
pred_betas = hawor_data['pred_betas'][hand_idx]
betas_mean = pred_betas[0].astype(np.float32)
mano_layer = MANOLayer('right', betas_mean)

pred_trans = hawor_data['pred_trans'][hand_idx]
pred_rot = hawor_data['pred_rot'][hand_idx]
pred_hand_pose = hawor_data['pred_hand_pose'][hand_idx]
pred_valid = hawor_data['pred_valid'][hand_idx]

wrist_positions = []
fingertip_positions = []
for i in range(len(pred_trans)):
    if not pred_valid[i]:
        continue
    rot = pred_rot[i]
    hand_pose = pred_hand_pose[i]
    trans = pred_trans[i]
    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    _, j = mano_layer(p, t)
    j = j[0].numpy()
    j_sapien = (RXWORLD_TO_SAPIEN @ j.T).T
    wrist_positions.append(j_sapien[0, :3])
    fingertip_positions.append(j_sapien[[4, 8, 12, 16, 20], :3])

wrist_positions = np.array(wrist_positions)
all_fingertips = np.vstack(fingertip_positions)

print(f'\n=== 手部在 SAPIEN 坐标系 ===')
print(f'  手腕中心: {wrist_positions.mean(axis=0)}')
print(f'  手腕范围: [{wrist_positions.min(axis=0)}] ~ [{wrist_positions.max(axis=0)}]')
print(f'  指尖中心: {all_fingertips.mean(axis=0)}')

dist_wrist_glb_center = np.linalg.norm(wrist_positions.mean(axis=0) - glb_center)
print(f'\n=== 距离分析 ===')
print(f'  手腕→GLB中心: {dist_wrist_glb_center:.4f} m')

from scipy.spatial import cKDTree
tree = cKDTree(all_verts_sapien)
dists_wrist, _ = tree.query(wrist_positions)
dists_fingertip, _ = tree.query(all_fingertips)

print(f'  手腕→GLB最近顶点: min={dists_wrist.min():.4f}m, mean={dists_wrist.mean():.4f}m, max={dists_wrist.max():.4f}m')
print(f'  指尖→GLB最近顶点: min={dists_fingertip.min():.4f}m, mean={dists_fingertip.mean():.4f}m, max={dists_fingertip.max():.4f}m')

print(f'\n=== 逐帧手腕→GLB最近距离 (前20帧) ===')
for i in range(min(20, len(dists_wrist))):
    print(f'  帧{i}: 手腕距GLB {dists_wrist[i]:.4f}m')

R_c2w_all = hawor_data['pred_cam_R']
t_c2w_all = hawor_data['pred_cam_t']
cam_positions = []
for i in range(len(R_c2w_all)):
    R = R_c2w_all[i]
    t = t_c2w_all[i]
    cam_pos_hawor = -R.T @ t
    cam_pos_sapien = RXWORLD_TO_SAPIEN @ cam_pos_hawor
    cam_positions.append(cam_pos_sapien)
cam_positions = np.array(cam_positions)

dist_cam_glb = np.linalg.norm(cam_positions.mean(axis=0) - glb_center)
dist_cam_wrist = np.linalg.norm(cam_positions.mean(axis=0) - wrist_positions.mean(axis=0))
print(f'\n=== 相机位置分析 ===')
print(f'  相机中心: {cam_positions.mean(axis=0)}')
print(f'  相机→GLB中心: {dist_cam_glb:.4f}m')
print(f'  相机→手腕中心: {dist_cam_wrist:.4f}m')
print(f'  相机移动范围: X[{cam_positions[:,0].min():.3f},{cam_positions[:,0].max():.3f}] Y[{cam_positions[:,1].min():.3f},{cam_positions[:,1].max():.3f}] Z[{cam_positions[:,2].min():.3f},{cam_positions[:,2].max():.3f}]')
