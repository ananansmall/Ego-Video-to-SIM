#!/usr/bin/env python3
"""test/env_collision.py — 环境碰撞化实验: 场景体素→静态碰撞盒 + WBC 排斥联调。

验证: (a) 机器人不再穿过实验台 (碰撞生效), (b) WBC 排斥让规划绕开
(跟踪不崩), (c) 抓取仍成功。
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

EE_DIR = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee5_mocap'
EF_DIR = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee5_oven'
SC = EF_DIR + '/pointcloud/scene_cloud_mocap.npz'

# ---- 障碍/碰撞盒构建: 场景点云 → pad 轨迹附近 → 体素聚合 → 盒 ----
sc = np.load(SC)
P = sc['points']
dm = np.load(EE_DIR + '/ef3r_right_targets.npz')
pad = dm['pos']
Pd, pad_s = P[::8], pad[::20]
near = Pd[(np.min(np.linalg.norm(Pd[:, None, :2] - pad_s[None, :, :2], axis=2), axis=1) < 1.4)
          & (Pd[:, 2] > 0.05) & (Pd[:, 2] < 1.55)]
print("路径附近场景点:", len(near))
# 排除 pad 轨迹直接邻域 (30cm 球) —— 物体/指爪工作区, 碰撞盒会卡臂
d_pad = np.min(np.linalg.norm(near[:, None, :2] - pad_s[None, :, :2], axis=2), axis=1)
near = near[d_pad > 0.30]
# 15cm 体素聚合
grid = np.floor(near[:, :3] / 0.15).astype(np.int64)
occ = np.unique(grid, axis=0)
centers = (occ + 0.5) * 0.15
if len(centers) > 120:
    sel = np.random.RandomState(0).choice(len(centers), 120, replace=False)
    centers = centers[sel]
env_boxes = np.hstack([centers, np.full((len(centers), 3), 0.075)])
print("环境碰撞盒:", len(env_boxes))

# WBC 排斥球 (同集合, r=0.12 + margin)
obstacles = np.hstack([centers, np.full((len(centers), 1), 0.12)])

import cb_large_range_sim as cb
cb.main(targets=EE_DIR + '/ef3r_right_targets.npz', name='ee5_envcol',
        object_npz=EF_DIR + '/object_instance.npz',
        scene_cloud=SC,
        object_physics=True, w_ori=0.0, torso_wbc=True, arm_beam=4,
        finger_force_limit=30.0, base_ref_source="body",
        torso_gain=6.0, wrist_gain=3.0,
        depenetration_vel=0.05, base_interp=True,
        legacy_physics=False, phys_only=True,
        obstacles=obstacles, w_obs=0.05, env_boxes=env_boxes)

t = np.load('results/large_range/ee5_envcol_exec_trace.npz')
g = t['grip'][:, :3]; o = t['obj'][:, :3]
e = np.asarray(t['phys_err'])
dmin = np.min(np.linalg.norm(g[:, None, :] - centers[None, :, :], axis=2), axis=1)
print(f"跟随 mean {np.mean(e):.1f}mm | 抓持-障碍最小距离 p50 "
      f"{np.percentile(dmin,50)*1000:.0f}mm (穿模判定: <75mm=盒内)")
