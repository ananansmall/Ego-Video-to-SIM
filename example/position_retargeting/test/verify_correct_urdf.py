"""
关键验证：用r1_full/r1_lite_robot_glb.urdf（关节定义与floating arm一致）
替换r1/robot_glb.urdf（关节定义不同），测试IK精度

根本原因：
  r1/robot_glb.urdf 的右臂关节定义与 floating arm URDF 完全不同
  例如 joint1: floating=[0,0,0.086],axis=[0,0,1] vs full=[-0.001,-0.045,0],axis=[0,-1,0]
  而 r1_full/r1_lite_robot_glb.urdf 的关节定义与 floating arm 一致
"""
import sys
import tempfile
import numpy as np
import xml.etree.ElementTree as ET
import sapien
import sapien.physx
from pytransform3d import rotations as pr
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
R1_FULL_URDF = Path("/home/an/robot_world_ws/src/dex-retargeting/assets/robots/r1_full/r1_lite_robot_glb.urdf")
FLOATING_ARM_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1_lite" / "configs" / "urdfs" / "r1lite_floating_right.urdf"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GALAXEA_SIM_PATH))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dex_retargeting import yourdfpy as urdf
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1_lite" / "configs" / "settings_left.yaml"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1_lite" / "configs" / "settings_right.yaml"

Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])


def strip_visual_from_urdf(urdf_path, output_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            link.remove(visual)
    tree.write(output_path, xml_declaration=True)


def floating_arm_fk(joint_angles):
    JOINT_ORIGINS = [
        np.array([0, 0, 0.08605]),
        np.array([0, 0.03075, 0.04925]),
        np.array([-0.3, 0.00025004, 0]),
        np.array([0.1747, 0.00049739, 0.075485]),
        np.array([0.08, -0.031498, 0.0405]),
        np.array([0.022503, 0, -0.0405]),
    ]
    JOINT_AXES = [
        np.array([0, 0, 1]),
        np.array([0, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 0, 0]),
    ]
    GRIPPER_ORIGIN = np.array([0.08165, 0, 0])

    def rot_axis_angle(axis, angle):
        c, s = np.cos(angle), np.sin(angle)
        t = 1 - c
        x, y, z = axis
        return np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
        ])

    R = np.eye(3)
    p = np.zeros(3)
    for i in range(6):
        p = p + R @ JOINT_ORIGINS[i]
        R_joint = rot_axis_angle(JOINT_AXES[i], joint_angles[i])
        R = R @ R_joint
    p = p + R @ GRIPPER_ORIGIN
    return p, R


def main():
    print("=" * 80)
    print("验证: 用r1_lite_robot_glb.urdf (关节定义与floating arm一致)")
    print("=" * 80)

    tolerances = [0.001, 0.001, 0.001, 10.0, 10.0, 10.0]
    ik_solver = RelaxedIKSolver(
        left_setting_file_path=str(R1_LEFT_SETTINGS),
        right_setting_file_path=str(R1_RIGHT_SETTINGS),
        tolerances=tolerances,
    )

    right_arm_starting = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]

    test_targets = [
        np.array([0.30, 0.0, -0.20]),
        np.array([0.20, -0.15, -0.15]),
        np.array([0.45, 0.15, -0.30]),
        np.array([0.35, 0.10, -0.10]),
        np.array([0.25, -0.10, -0.25]),
        np.array([0.40, 0.05, -0.15]),
        np.array([0.15, 0.05, -0.30]),
    ]

    # 加载r1_lite_robot_glb.urdf
    temp_dir = tempfile.mkdtemp(prefix="r1_lite_test-")

    robot_urdf = urdf.URDF.load(str(R1_FULL_URDF), add_dummy_free_joints=True, build_scene_graph=False)
    temp_path = f"{temp_dir}/r1_lite_robot_glb.urdf"
    robot_urdf.write_xml_file(temp_path)

    no_vis_path = f"{temp_dir}/no_vis_r1_lite_robot_glb.urdf"
    strip_visual_from_urdf(temp_path, no_vis_path)

    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    scene.set_timestep(1 / 240)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(no_vis_path)

    robot.set_root_pose(sapien.Pose([0, 0, 0], Q_180Z.tolist()))

    active_joints = robot.get_active_joints()
    joint_names = [j.get_name() for j in active_joints]
    right_arm_indices = [i for i, name in enumerate(joint_names) if "right_arm" in name]

    print(f"关节数: {len(active_joints)}")
    print(f"右臂关节索引: {right_arm_indices}")
    print(f"右臂关节名: {[joint_names[i] for i in right_arm_indices]}")

    for joint in active_joints:
        joint.set_drive_property(stiffness=100000.0, damping=10000.0)

    ee_link = None
    base_link = None
    for link in robot.get_links():
        if link.get_name() == "right_gripper_link":
            ee_link = link
        if link.get_name() == "right_arm_base_link":
            base_link = link

    initial_qpos = robot.get_qpos().copy()
    for j, idx in enumerate(right_arm_indices):
        if j < len(right_arm_starting):
            initial_qpos[idx] = right_arm_starting[j]
    robot.set_qpos(initial_qpos)
    for joint in active_joints:
        joint.set_drive_target(initial_qpos[joint_names.index(joint.get_name())])
    for _ in range(5):
        scene.step()

    base_pose = base_link.get_entity_pose()
    base_p = np.array(base_pose.p)
    base_q = np.array(base_pose.q)
    base_R = pr.matrix_from_quaternion(base_q)

    print(f"\nbase_link 位置: {base_p}")
    print(f"base_link R ≈ R_180Z: {np.allclose(base_R, np.diag([-1, -1, 1]), atol=0.01)}")

    # 零位测试
    print(f"\n{'='*60}")
    print("零位FK验证:")
    print(f"{'='*60}")

    qpos_zero = robot.get_qpos().copy()
    for idx in right_arm_indices:
        qpos_zero[idx] = 0.0
    robot.set_qpos(qpos_zero)
    for joint in active_joints:
        joint.set_drive_target(qpos_zero[joint_names.index(joint.get_name())])
    for _ in range(10):
        scene.step()

    ee_pos_zero = np.array(ee_link.get_entity_pose().p)
    ee_in_base_zero = base_R.T @ (ee_pos_zero - base_p)
    math_fk_zero, _ = floating_arm_fk(np.zeros(6))

    print(f"  SAPIEN EE (世界帧): {ee_pos_zero}")
    print(f"  SAPIEN EE (base帧): {ee_in_base_zero}")
    print(f"  数学FK (base帧):    {math_fk_zero}")
    print(f"  差异: {np.linalg.norm(ee_in_base_zero - math_fk_zero)*100:.2f}cm")

    # IK测试
    print(f"\n{'='*60}")
    print("IK精度测试 (直接用pos_arm作为IK输入):")
    print(f"{'='*60}")

    errors = []
    for target in test_targets:
        ik_solver.relaxed_ik_right.reset(right_arm_starting)
        for _ in range(300):
            joints = ik_solver.solve_position_right(target.tolist(), [1, 0, 0, 0])
        joints = np.array(joints)

        qpos = robot.get_qpos().copy()
        for j, idx in enumerate(right_arm_indices):
            if j < len(joints):
                qpos[idx] = joints[j]
        robot.set_qpos(qpos)
        for joint in active_joints:
            joint.set_drive_target(qpos[joint_names.index(joint.get_name())])
        for _ in range(5):
            scene.step()

        ee_pos = np.array(ee_link.get_entity_pose().p)
        target_world = base_R @ target + base_p
        err = np.linalg.norm(ee_pos - target_world) * 100
        errors.append(err)

        ee_in_base = base_R.T @ (ee_pos - base_p)
        math_pos, _ = floating_arm_fk(joints)
        err_vs_math = np.linalg.norm(ee_in_base - math_pos) * 100

        print(f"  目标: {target} → 误差: {err:.2f}cm (vs数学FK: {err_vs_math:.2f}cm)")

    errors = np.array(errors)
    print(f"\n  平均={np.mean(errors):.2f}cm, 中位={np.median(errors):.2f}cm, 最大={np.max(errors):.2f}cm")
    print(f"  <2cm: {np.sum(errors < 2)}/{len(errors)}, <5cm: {np.sum(errors < 5)}/{len(errors)}")

    # 也测试base_R_inv变换
    print(f"\n{'='*60}")
    print("IK精度测试 (用base_R_inv变换):")
    print(f"{'='*60}")

    errors2 = []
    for target in test_targets:
        target_world = base_R @ target + base_p
        pos_base = base_R.T @ (target_world - base_p)

        ik_solver.relaxed_ik_right.reset(right_arm_starting)
        for _ in range(300):
            joints = ik_solver.solve_position_right(pos_base.tolist(), [1, 0, 0, 0])
        joints = np.array(joints)

        qpos = robot.get_qpos().copy()
        for j, idx in enumerate(right_arm_indices):
            if j < len(joints):
                qpos[idx] = joints[j]
        robot.set_qpos(qpos)
        for joint in active_joints:
            joint.set_drive_target(qpos[joint_names.index(joint.get_name())])
        for _ in range(5):
            scene.step()

        ee_pos = np.array(ee_link.get_entity_pose().p)
        err = np.linalg.norm(ee_pos - target_world) * 100
        errors2.append(err)

    errors2 = np.array(errors2)
    print(f"  平均={np.mean(errors2):.2f}cm, 中位={np.median(errors2):.2f}cm, 最大={np.max(errors2):.2f}cm")
    print(f"  <2cm: {np.sum(errors2 < 2)}/{len(errors2)}, <5cm: {np.sum(errors2 < 5)}/{len(errors2)}")

    print(f"\n{'='*80}")
    if np.mean(errors) < 5 or np.mean(errors2) < 5:
        print("✓ r1_lite_robot_glb.urdf 的IK精度显著改善！")
        print("  根本原因确认: r1/robot_glb.urdf 的关节定义与floating arm不兼容")
        print("  解决方案: 使用r1_full/r1_lite_robot_glb.urdf")
    else:
        print("✗ IK精度仍然不够好，需要进一步分析")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
