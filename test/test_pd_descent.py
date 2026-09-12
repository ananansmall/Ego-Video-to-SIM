#!/usr/bin/env python3
"""最小对照实验: 虚拟关节纯 PD 能否物理跟踪一次 20cm 下降?

背景: 统一物理后 Stage3 基座在下降段卡死 (实际 z 差指令 6-9cm, 25 帧仅收敛 0.2mm),
而 K=50000 在 6cm 误差下应有 ~3000N 驱动力 (整机仅 ~0.7kg) —— 理论上 1-2 帧内到位.
本脚本隔离验证: 无物体、无手指闭合, 纯 drive 走 下降→上升, 逐步记录 actual vs cmd.
"""
import sys
from pathlib import Path
import numpy as np

_TJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TJ))

from grasp_hawor import GraspSimulator
from physics_env import physics_step

HAWOR = "/home/an/data/hawor/7"
RAS = "/home/an/data/ras/7_vggt_omega"


def main():
    sim = GraspSimulator(
        hawor_dir=HAWOR, ras_dir=RAS, mode="gripper_only", side="left",
        output_dir="/tmp/pd_test", num_frames=113, start_frame=0,
        views="none", grasp_mode="hybrid", viewer=False,
    )
    sim._align_scene()
    sim.run(run_main_loop=False)
    robot = sim.robot_info["robot"]
    vid = sim.robot_info["virtual_idx"]
    gi1 = sim.robot_info["gripper_idx1"]
    gi2 = sim.robot_info["gripper_idx2"]
    aji = sim.robot_info.get("arm_joint_indices", [])
    active = robot.get_active_joints()

    # 驱动属性 introspection
    try:
        j = active[vid['vz']]
        print("vz joint drive-related attrs:",
              [a for a in dir(j) if 'drive' in a.lower() or 'limit' in a.lower()])
        for attr in ("drive_stiffness", "drive_damping", "drive_force_limit",
                     "drive_target", "get_drive_property"):
            if hasattr(j, attr):
                v = getattr(j, attr)
                v = v() if callable(v) else v
                print(f"vz {attr} = {v}")
    except Exception as e:
        print(f"drive introspection fail: {e}")

    z0 = float(robot.get_qpos()[vid['vz']])
    print(f"\n起始 vz={z0:.4f}")

    def run_to(z_cmd, n_frames, tag):
        print(f"--- {tag}: → z={z_cmd:.3f} / {n_frames} 帧 ---")
        for f in range(n_frames):
            zc = float(robot.get_qpos()[vid['vz']])
            step = np.clip(z_cmd - zc, -0.005, 0.005)
            active[vid['vz']].set_drive_target(float(zc + step))
            physics_step(robot, aji, gi1, gi2, np.array([]),
                         0.04, 0.04, sim.scene)
            if f % max(1, n_frames // 8) == 0 or f == n_frames - 1:
                z_act = float(robot.get_qpos()[vid['vz']])
                print(f"  F{f:3d}: cmd={z_cmd:.4f} act={z_act:.4f} err={(z_cmd-z_act)*1000:+7.1f}mm")

    # 阶段A: 升到 0.25 悬空 (验证上升跟踪)
    run_to(0.25, 60, "阶段A 上升(原始驱动状态)")

    # v6.0c: 显式重设 drive property 后重测 (排除 warmup 覆盖)
    active[vid['vz']].set_drive_property(stiffness=50000.0, damping=600.0)
    print("已显式重设 vz drive property (50000, 600)")
    run_to(0.25, 60, "阶段A2 上升(重设驱动后)")

    # v6.0c: x 轴对照 (排除 z 特有配置)
    print("--- 阶段D x轴 +10cm / 30帧 ---")
    x0 = float(robot.get_qpos()[vid['vx']])
    for f in range(30):
        xc = float(robot.get_qpos()[vid['vx']])
        active[vid['vx']].set_drive_target(float(x0 + 0.10))
        physics_step(robot, aji, gi1, gi2, np.array([]), 0.04, 0.04, sim.scene)
        if f % 5 == 0 or f == 29:
            x_act = float(robot.get_qpos()[vid['vx']])
            print(f"  F{f:3d}: cmd={x0+0.10:.4f} act={x_act:.4f} err={(x0+0.10-x_act)*1000:+7.1f}mm")

    # 阶段B: 半空下降 15cm (无接触, 验证纯 PD 下降跟踪)
    run_to(0.10, 60, "阶段B 悬空下降")
    # 阶段C: 半空阶跃 -8cm (瞬时指令, 看响应速度)
    print("--- 阶段C 阶跃 -8cm (60帧) ---")
    for f in range(60):
        active[vid['vz']].set_drive_target(0.02)
        physics_step(robot, aji, gi1, gi2, np.array([]), 0.04, 0.04, sim.scene)
        if f % 10 == 0 or f == 59:
            z_act = float(robot.get_qpos()[vid['vz']])
            print(f"  F{f:3d}: act={z_act:.4f}")


if __name__ == "__main__":
    main()
