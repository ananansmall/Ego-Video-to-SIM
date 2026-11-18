#!/usr/bin/env python3
"""run_seq456.py —— 用大范围管线跑 456483158 这条新轨迹。

轨迹由 hawor_to_traj.py 从 HaWoR 产出转换而来:
    /tmp/traj_456483158_right.npz   (300 帧, 右手, 键与 cb_right_targets.npz 一致)
CB_HAWOR 只是 targets 里没有 R_c2w/t_c2w 时的回退, 本 npz 已含, 不会触发。
"""
import sys

sys.path.insert(0, '/path/to/Real2SimGripper/scripts')
import cb_large_range_sim as cb

TRAJ = "/tmp/traj_456483158_right.npz"
print("=" * 70, flush=True)
print("跑新轨迹 456483158 (300 帧, 右手, 大范围 WBC)", flush=True)
print("=" * 70, flush=True)

cb.main(targets=TRAJ, name="seq456483158")
print("\nRUN_SEQ456_DONE", flush=True)
