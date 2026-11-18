"""探针: 实测 R1 夹爪两指几何 (含正确驱动设置)。

目的: 确定
  1. 两指 link 原点间距 与 q 的真实关系 (是否只有一指移动)
  2. set_drive_target 是否真的生效 (实际 q vs 目标)
  3. jmap 名字 -> get_dof 索引是否正确
"""
import sys
sys.path.insert(0, "/path/to/Real2SimGripper/scripts")
import numpy as np
import sapien.core as sapien
import robotwin_r1_duck_grasp as rg
import bp14_dual_arm_sim as bp

engine = sapien.Engine()
scene = rg.setup_scene(engine)
tmp = "/tmp/r1_probe.urdf"
rg.make_r1_urdf_abs(bp.R1_URDF, tmp)
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(tmp)
robot.set_root_pose(sapien.Pose(p=rg.R1_BASE_P, q=rg.R1_BASE_Q))

joints = robot.get_active_joints()
jmap = {j.get_name(): j for j in joints}
print("active joints:", [j.get_name() for j in joints])
for j in joints:
    j.set_drive_property(stiffness=1000.0, damping=200.0)

for nm in ["right_gripper_finger_joint1", "right_gripper_finger_joint2"]:
    j = jmap[nm]
    print(f"{nm}: dof={j.get_dof()} limits={j.get_limits()}")


def measure(label):
    l1 = robot.find_link_by_name("right_gripper_finger_link1").get_entity_pose().p
    l2 = robot.find_link_by_name("right_gripper_finger_link2").get_entity_pose().p
    qa = robot.get_qpos()
    q1 = qa[jmap["right_gripper_finger_joint1"].get_dof()]
    q2 = qa[jmap["right_gripper_finger_joint2"].get_dof()]
    print(f"  {label}: q1={q1:.4f} q2={q2:.4f} | 指原点间距 "
          f"{np.linalg.norm(l1 - l2) * 1000:.1f}mm")


for q in (0.0, 0.01, 0.02, 0.05):
    for nm in ["right_gripper_finger_joint1", "right_gripper_finger_joint2"]:
        jmap[nm].set_drive_target(q)
    for _ in range(400):
        scene.step()
    measure(f"target={q:.3f}")
