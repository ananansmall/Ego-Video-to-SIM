#!/usr/bin/env python3
"""test/collision_wbc.py — 碰撞感知 WBC 实验: 场景体素→排斥球→WBC 代价。

验证: 排斥代价下 (a) 跟随精度不退化, (b) pad/臂离障碍物距离增大。
对照: 无排斥 (w_obs=0) vs 有排斥 (w_obs=0.05)。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

EE_DIR = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee5_mocap'
EF_DIR = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee5_oven'

# 障碍球: 场景体素 → pad 轨迹附近 (1.3m) + 工作高度带 → 网格下采样 ≤80 球 r=0.12
sc = np.load(EF_DIR + '/pointcloud/scene_cloud_mocap.npz')
P = sc['points']
dm = np.load(EE_DIR + '/ef3r_right_targets.npz')
pad = dm['pos']
Pd, pad_s = P[::10], pad[::20]                 # 降采样后近邻 (防 OOM)
near = Pd[(np.min(np.linalg.norm(Pd[:, None, :2] - pad_s[None, :, :2], axis=2), axis=1) < 1.3)
          & (Pd[:, 2] > 0.05) & (Pd[:, 2] < 1.5)]
print("路径附近场景点:", len(near))
# 网格 0.15m 聚合 → 占据格中心
grid = np.floor(near[:, :3] / 0.15).astype(np.int64)
occ = np.unique(grid, axis=0)
centers = (occ + 0.5) * 0.15
if len(centers) > 80:
    sel = np.random.choice(len(centers), 80, replace=False)
    centers = centers[sel]
obs = np.hstack([centers, np.full((len(centers), 1), 0.12)])
print("障碍球:", len(obs), "r=0.12m")

import cb_large_range_sim as cb
for w_obs in (0.0, 0.05):
    nm = 'ee5_coll_%d' % int(w_obs * 1000)
    cb.main(targets=EE_DIR + '/ef3r_right_targets.npz', name=nm,
            object_npz=EF_DIR + '/object_instance.npz',
            scene_cloud=EF_DIR + '/pointcloud/scene_cloud_mocap.npz',
            object_physics=False, w_ori=0.0, torso_wbc=True, arm_beam=4,
            base_ref_source="body", obstacles=obs, w_obs=w_obs,
            legacy_physics=False, base_interp=True, phys_only=True)
    t = np.load(f'results/large_range/{nm}_exec_trace.npz')
    g = t['grip'][:, :3]
    dmin = np.min(np.linalg.norm(g[:, None, :] - obs[None, :, :3], axis=2), axis=1)
    print(f"  w_obs={w_obs}: pad-障碍最小距离 p50 "
          f"{np.percentile(dmin,50)*1000:.0f}mm (越远越好)")
