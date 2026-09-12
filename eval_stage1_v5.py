#!/usr/bin/env python
"""v5.0 统一动作模板单次验证: 加载 best_grasp.npz, 160 帧 rollout (竖直提起 20cm).

不跑 CEM, 只验证: 旧最优位姿在新模板 + 新 reward 下表现如何.
"""
import sys
import numpy as np

sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization")
from grasp_hawor import GraspSimulator  # noqa: E402

NPZ = "/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_right/stage1/best_grasp.npz"

sim = GraspSimulator(
    hawor_dir="/home/an/data/hawor/7",
    ras_dir="/home/an/data/ras/7_vggt_omega",
    mode="gripper_only",
    side="right",
    output_dir="/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/output/gripper_right",
    num_frames=-1,
    start_frame=0,
    views="none",
    grasp_mode="hybrid",
    viewer=False,
)
sim._test_stage1_only = True
sim.video_tag = "stage1"
sim.f50_tilt_deg = 0.0
sim._align_scene()
sim.run()
side = "right"
sim._pre_translate_mano_trajectory(side)
sim._compute_neutral_offsets()

d = np.load(NPZ, allow_pickle=True)
grasp_pose = {
    "pos": d["pos"], "R": d["R"], "euler": d["euler"],
    "gripper_qpos": float(d["gripper_qpos"]),
    "strategy": str(d["strategy"]),
}
print("\n=== v5.0 统一动作模板验证 (160 帧, 竖直提起 20cm) ===")
print(f"加载位姿: pos={grasp_pose['pos'].round(4)}, reward(旧v4.37)={float(d['reward']):.2f}")

res = sim.rollout_grasp_only(grasp_pose, side, strategy="pd_then_lock", n_frames=160)
r_new = sim.stage1_reward(res)

lift = res["obj_lift_final"]
peak = res.get("obj_lift", 0.0)
print("\n--- 结果 ---")
print(f"lift_final = {lift*100:.2f} cm  (目标 20cm, 达成率 {min(1,lift/0.20)*100:.0f}%)")
print(f"lift_peak  = {peak*100:.2f} cm")
print(f"both_contact = {res.get('both_contact_count',0)} 帧 / 160")
print(f"xy_drift   = {res['obj_xy_drift']*100:.2f} cm")
print(f"peak_force = {res.get('peak_grip_force',0):.1f} N")
print(f"dist_final = {res.get('fingertip_obj_dist',9.9)*100:.1f} cm")
print(f"\nreward(v5.0) = {r_new:.1f}   [旧 npz 记录 reward(v4.37) = {float(d['reward']):.2f}]")
print("解读: 主导项 = 40*min(1, lift/20cm); 若 lift≈1cm 则主导项仅 ~2 分 ->",
      "该位姿在'跟随提升 20cm'标准下不合格, CEM 将被迫搜索真正能带起物体的位姿" if lift < 0.05 else "合格")
