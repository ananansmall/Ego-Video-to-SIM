#!/usr/bin/env python3
"""test/follow_audit.py — 跟随准确性审计: 滞后集 (ep8/9/10/16) 无物体纯跟随
对照, 分离 '轨迹质量问题' vs '物体接触干扰'。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import cb_large_range_sim as cb
import numpy as np

EPS = {8: 'ee8_bookshelf_organization', 9: 'ee9_clothes_folding',
       10: 'ee10_sandwich_making', 16: 'ee16_column'}
for ep, name in EPS.items():
    tgt = f'/path/to/project_data/mano_ws/EgoFound3R/outputs/ee{ep}_mocap/ef3r_right_targets.npz'
    if not os.path.exists(tgt):
        print(f"ep{ep}: 无目标, 跳过"); continue
    cb.main(targets=tgt, name=f'{name}_followonly',
            object_npz=None, scene_cloud=None,
            object_physics=False, w_ori=0.0, torso_wbc=True, arm_beam=4,
            base_ref_source="body", legacy_physics=False, base_interp=True,
            phys_only=True)
    t = np.load(f'results/large_range/{name}_followonly_exec_trace.npz')
    e = np.asarray(t['phys_err'])
    print(f"  ep{ep} 纯跟随: mean {np.mean(e):.1f}mm p95 {np.percentile(e,95):.1f} "
          f"max {e.max():.1f}mm", flush=True)
