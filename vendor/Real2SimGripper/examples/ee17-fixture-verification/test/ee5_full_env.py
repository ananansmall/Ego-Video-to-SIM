#!/usr/bin/env python3
"""test/ee5_full_env.py — mocap 抓取-移动-放置 + 拟合环境 全渲染。

坐标系: mocap 目标 (raw mocap 世界系, z-up) + 场景 (tracker 相机位姿直接
定位的 ego 深度点云, already_sim 直通 + z_shift 跟随 fit_height)。
真值: 手轨迹/交互时序 = mocap; 拟合: 场景点云/支撑面/mesh。
"""
import sys, os
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

EE_DIR = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee5_mocap'
EF_DIR = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee5_oven'

subprocess.run([sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                             "scripts", "ee_mocap_targets.py"),
                "--episode", "5", "--t0", "8", "--t1", "40"], check=True,
               capture_output=True)

import cb_large_range_sim as cb
cb.main(targets=EE_DIR + '/ef3r_right_targets.npz', name='ee5_full_env',
        object_npz=EF_DIR + '/object_instance.npz',
        scene_cloud=EF_DIR + '/pointcloud/scene_cloud_mocap.npz',
        object_physics=True, w_ori=0.0, torso_wbc=True, arm_beam=4,
        finger_force_limit=30.0, base_ref_source="body",
        torso_gain=6.0, wrist_gain=3.0,
        depenetration_vel=0.05, base_interp=True,
        legacy_physics=False, phys_only=False)
