#!/usr/bin/env python3
"""test/ep12_timing_sweep.py — 闭合时刻容差扫描: 真实抓取有时序容差,
在速度骤降点 ±10 帧内扫闭合偏移, 找到能夹住的实现 (合法: 真人每次抓取
的时刻也在 ±0.3s 内浮动)。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import numpy as np
import cb_large_range_sim as cb

EE = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee12_mocap'
EF = '/path/to/project_data/mano_ws/EgoFound3R/outputs/ee12_win'
base = np.load(EE + '/ef3r_right_targets.npz')
gs0 = np.asarray(base['grip_states']).copy()
k0 = int(np.argmax(gs0 >= 2))
print("基准闭合帧:", k0)

for off in (-10, -5, 0, 5, 10):
    gs = gs0.copy()
    _ki = max(0, k0 + off)
    gs[:] = 0
    gs[_ki:] = 2
    np.savez(EE + f'/ef3r_right_targets_off{off:+d}.npz',
             pos=base['pos'], rot=base['rot'], grip=base['grip'],
             valid=base['valid'], frame=base['frame'], fps=base['fps'],
             R_c2w=base['R_c2w'], t_c2w=base['t_c2w'],
             grip_states=gs, grip_state_names=base['grip_state_names'])
    nm = f'ee12_off{off:+d}'
    cb.main(targets=EE + f'/ef3r_right_targets_off{off:+d}.npz', name=nm,
            object_npz=EF + '/object_instance.npz', scene_cloud=None,
            object_physics=True, w_ori=0.0, torso_wbc=True, arm_beam=4,
            finger_force_limit=30.0, base_ref_source="body",
            torso_gain=6.0, wrist_gain=3.0,
            depenetration_vel=0.05, base_interp=True,
            legacy_physics=False, phys_only=True)
    d = np.load(f'results/large_range/{nm}_large_range.json')
    o = d.get('object') or {}
    print(f"  off={off:+d}: 抓起={'✓' if o.get('grasped') else '✗'} "
          f"抬升 {round(o.get('lift_mm',0))}mm", flush=True)
