#!/usr/bin/env python3
"""test/ee5_scene_mocap.py — 用 tracker 相机位姿把 ego 深度点云定位到 mocap 世界系。

EgoExplorer tracker[0] = 头部相机 (位置+朝向, mocap 世界系, y-up) —— 相机到
mocap 系的外参**数据集自带**, 无需配准。ego 深度 (EF3R depth_global 锚帧)
反投影到相机系 → tracker 位姿变换 → mocap 世界系点云 → 体素化 → cb 场景。

输出: <ef3r>/pointcloud/scene_cloud_mocap.npz (already_sim=1, cb 直通)
"""
import sys, os
import numpy as np

EP = '/path/to/project_data/dataset/EgoExplorer/dataset/lerobot/episode_meta/episode_000005'
EF = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee5_oven'

d = np.load(EP + '/smplx/smplx_episode.npz')
tr_p = d['tracker_position']    # (N,5,3)  mocap 世界系 y-up
tr_r = d['tracker_rotation']    # (N,5,3,3)?
print("tracker_rotation shape:", tr_r.shape)
cam_p = tr_p[:, 0, :]           # 头部相机位置
cam_R = tr_r[:, 0] if tr_r.ndim == 4 else None
print("cam pos 范围:", np.round(cam_p.min(0), 2), '->', np.round(cam_p.max(0), 2))

import torch
ef = torch.load(EF + '/inference_output.pt', map_location='cpu', weights_only=False)
dep = ef['depth'].numpy()                       # (A,288,512) 米制
Ka = ef['K'].numpy()                            # (A,3,3)
dfr = ef['depth_frame_indices'].numpy()         # 锚帧→视频帧
print("深度锚帧:", dfr.tolist())

pts_all, col_all = [], []
import imageio.v2 as imageio
r = imageio.get_reader('/path/to/workspace/ee5_oven.mp4')
frames = {}
for i, img in enumerate(r):
    if i in dfr.tolist():
        frames[i] = img
    if i > max(dfr) + 1:
        break
r.close()

for ai, fa in enumerate(dfr):
    z = dep[ai]
    vs, us = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    m = np.isfinite(z) & (z > 0.1) & (z < 6.0)
    zz = z[m]
    K = Ka[ai]
    pc = np.stack([(us[m] - K[0, 2]) / K[0, 0] * zz,
                   (vs[m] - K[1, 2]) / K[1, 1] * zz, zz], 1)   # 相机系
    # 相机系 → mocap 世界系: tracker 旋转 (mocap y-up) + 位置
    Rw = tr_r[fa][0] if tr_r.ndim == 4 else np.eye(3)
    pw = pc @ Rw.T + cam_p[fa]
    # 采样 (降密度) + 颜色
    img = frames.get(fa)
    if img is not None:
        Hh, Ww = img.shape[:2]
        uu = (us[m] * (Ww / z.shape[1])).round().astype(int).clip(0, Ww - 1)
        vv = (vs[m] * (Hh / z.shape[0])).round().astype(int).clip(0, Hh - 1)
        col = img[vv, uu] / 255.0
    else:
        col = np.full((len(pw), 3), 0.6)
    sel = np.random.choice(len(pw), min(len(pw), 30000), replace=False)
    pts_all.append(pw[sel]); col_all.append(col[sel])
    print(f"  锚帧 F{fa}: {len(pw)} 点")

P = np.concatenate(pts_all); C = np.concatenate(col_all)
# mocap y-up → sim z-up (与 ee_mocap_targets 同变换) + 地板对齐:
# 地板 = 踝 tracker 最低 y - 0.1m → sim z=0 (机器人基座所在平面)
ankle_y = float(np.min(d['tracker_position'][:, 3:5, 1]))
floor_y = ankle_y - 0.10
M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
P = P @ M.T
P[:, 2] -= (-floor_y)   # sim z = -y_mocap; 地板 y=floor_y → z=0 即减 -floor_y
print(f"  地板对齐: 踝最低 y {ankle_y:.2f} → 地板 y {floor_y:.2f} → sim z=0 "
      f"(场景 z 平移 {floor_y:+.2f})")
print(f"场景点云 (mocap→sim z-up): {len(P)} 点 | 范围 "
      f"{np.round(P.min(0),2).tolist()} -> {np.round(P.max(0),2).tolist()}")
os.makedirs(EF + '/pointcloud', exist_ok=True)
np.savez(EF + '/pointcloud/scene_cloud_mocap.npz', points=P.astype(np.float32),
         colors=C.astype(np.float32), already_sim=np.array([1]))
print(f"✓ {EF}/pointcloud/scene_cloud_mocap.npz (already_sim=1)")
