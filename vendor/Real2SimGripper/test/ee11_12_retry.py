#!/usr/bin/env python3
"""test/ee11_12_retry.py — ep11/12 速度骤降闭合时序重试 (细长器皿抓取)。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import cb_large_range_sim as cb
for ep, obj in ((11, '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee11_win'),
                (12, '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee12_win')):
    obj_npz = obj + '/object_instance.npz'
    if not os.path.exists(obj_npz):
        print(f"ep{ep}: 无物体契约, 跳过"); continue
    cb.main(targets=f'/path/to/project_data/mano_ws/EgoFound3R/outputs/ee{ep}_mocap/ef3r_right_targets.npz',
            name=f'ee{ep}_mocap',
            object_npz=obj_npz, scene_cloud=None,
            object_physics=True, w_ori=0.0, torso_wbc=True, arm_beam=4,
            finger_force_limit=30.0, base_ref_source="body",
            torso_gain=6.0, wrist_gain=3.0,
            depenetration_vel=0.05, base_interp=True,
            legacy_physics=False, phys_only=True)
