"""
R1 原版人形机器人 IK 位置跟踪测试

使用原版 r1_v2_1_0 机器人 (GalaxeaManipSim) + 原版 RelaxedIK 配置

关键: SAPIEN加载的URDF关节定义必须与RelaxedIK的floating arm URDF一致
  - SAPIEN: r1/robot_glb.urdf (原版人形机器人)
  - RelaxedIK: r1/configs/urdfs/r1_v2_1_0_floating_right.urdf (原版floating arm)
  两者的右臂关节定义完全一致，FK链匹配

坐标系变换:
  世界帧 → arm_base帧: pos_base = base_link_R_inv @ (pos_world - base_link_p)

用法:
    python r1_ik_position_test.py --num-frames 60
    python r1_ik_position_test.py --trajectory circle
    python r1_ik_position_test.py --view behind
"""

import argparse
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import sapien
from pytransform3d import rotations as pr

from dex_retargeting.constants import HandType, RobotName, RetargetingType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hand_robot_viewer import RobotHandDatasetSAPIENViewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")

sys.path.insert(0, str(GALAXEA_SIM_PATH))
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"

Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])
ARM_MAX_REACH = 0.713
WARMUP_FRAMES = 10
IK_WARMUP_ITERS = 200

CAMERA_QUATS = {
    "behind": [0.0, 0.0, 1.0, 0.0],
    "front": [1.0, 0.0, 0.0, 0.0],
    "topdown": [0.7071, 0.0, 0.7071, 0.0],
}


def _set_qpos_with_drive(r1_robot, qpos, active_joints, joint_names):
    r1_robot.set_qpos(qpos)
    for joint in active_joints:
        joint.set_drive_target(qpos[joint_names.index(joint.get_name())])


def _warmup_ik_solver(ik_solver, target_pos_base, target_quat_base_wxyz, n_iters=IK_WARMUP_ITERS, logger=None):
    for i in range(n_iters):
        joints = ik_solver.solve_position_right(target_pos_base.tolist(), target_quat_base_wxyz.tolist())
        if i == 0 or (i + 1) % 50 == 0:
            if logger:
                logger.info(f"    IK预热 迭代{i+1}/{n_iters}: 关节角={np.array(joints[:3])}...")

    best_joints = np.array(joints)

    if logger:
        logger.info(f"  IK预热完成 ({n_iters}次迭代)")
        logger.info(f"    最终关节角: {best_joints}")

    return best_joints


def _project_point(point_3d, camera):
    try:
        ext = camera.get_extrinsic_matrix().astype(np.float64)
        int_mat = camera.get_intrinsic_matrix().astype(np.float64)

        R = ext[:3, :3]
        t = ext[:3, 3]

        p_cam = R @ point_3d + t

        if p_cam[2] <= 0.01:
            return None

        uv = int_mat @ p_cam

        u = int(uv[0] / uv[2])
        v = int(uv[1] / uv[2])

        w_px = camera.get_width()
        h_px = camera.get_height()

        if 0 <= u < w_px and 0 <= v < h_px:
            return (u, v)
        return None
    except Exception:
        return None


def _draw_axes(bgr, origin, R, length, camera, label, label_color, pos_text=""):
    origin_2d = _project_point(origin, camera)
    if origin_2d is None:
        return

    x_end = origin + R[:, 0] * length
    y_end = origin + R[:, 1] * length
    z_end = origin + R[:, 2] * length

    x_2d = _project_point(x_end, camera)
    y_2d = _project_point(y_end, camera)
    z_2d = _project_point(z_end, camera)

    lw = 2

    if x_2d is not None:
        cv2.line(bgr, origin_2d, x_2d, (0, 0, 255), lw)

    if y_2d is not None:
        cv2.line(bgr, origin_2d, y_2d, (0, 255, 0), lw)

    if z_2d is not None:
        cv2.line(bgr, origin_2d, z_2d, (255, 0, 0), lw)

    cv2.putText(bgr, label, (origin_2d[0] - 15, origin_2d[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

    if pos_text:
        cv2.putText(bgr, pos_text, (origin_2d[0] - 15, origin_2d[1] - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1)


def generate_trajectory(traj_type, num_frames, base_link_p, base_link_R, arm_max_reach=ARM_MAX_REACH):
    safe_reach = arm_max_reach * 0.8

    center_arm = np.array([0.30, 0.0, -0.20])

    dist_center = np.linalg.norm(center_arm)
    if dist_center > safe_reach:
        center_arm = center_arm / dist_center * safe_reach * 0.9

    positions_world = []
    positions_arm = []
    orientations_world = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)

        if traj_type == "line":
            start_arm = np.array([0.20, -0.15, -0.15])
            end_arm = np.array([0.45, 0.15, -0.30])

            if np.linalg.norm(start_arm) > safe_reach:
                start_arm = start_arm / np.linalg.norm(start_arm) * safe_reach * 0.85
            if np.linalg.norm(end_arm) > safe_reach:
                end_arm = end_arm / np.linalg.norm(end_arm) * safe_reach * 0.85

            pos_arm = start_arm * (1 - t) + end_arm * t

        elif traj_type == "circle":
            max_radius = safe_reach - dist_center
            radius = min(0.12, max_radius * 0.8)

            angle = 2 * np.pi * t
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
            pos_arm = center_arm + offset

        elif traj_type == "square":
            max_side = (safe_reach - dist_center) * 1.5
            side = min(0.20, max_side)

            phase = t * 4

            if phase < 1:
                offset = np.array([-side / 2 + side * phase, -side / 2, 0.0])
            elif phase < 2:
                offset = np.array([side / 2, -side / 2 + side * (phase - 1), 0.0])
            elif phase < 3:
                offset = np.array([side / 2 - side * (phase - 2), side / 2, 0.0])
            else:
                offset = np.array([-side / 2, side / 2 - side * (phase - 3), 0.0])

            pos_arm = center_arm + offset

        elif traj_type == "helix":
            max_radius = (safe_reach - dist_center) * 0.7
            radius = min(0.10, max_radius)
            max_z = safe_reach * 0.3

            angle = 2 * np.pi * t
            z_offset = -max_z * np.sin(2 * np.pi * t)
            offset = np.array([radius * np.cos(angle), radius * np.sin(angle), z_offset])
            pos_arm = center_arm + offset

        else:
            pos_arm = center_arm.copy()

        dist_from_base = np.linalg.norm(pos_arm)
        if dist_from_base > arm_max_reach * 0.95:
            direction = pos_arm / dist_from_base
            pos_arm = direction * arm_max_reach * 0.9

        pos_world = base_link_R @ pos_arm + base_link_p

        positions_arm.append(pos_arm)
        positions_world.append(pos_world)

        R_grasp_arm = pr.matrix_from_euler([-np.pi / 2, 0, 0], 0, 1, 2, True)
        R_grasp_world = base_link_R @ R_grasp_arm
        quat_world = pr.quaternion_from_matrix(R_grasp_world)
        orientations_world.append(quat_world)

    return positions_world, positions_arm, orientations_world


def main():
    parser = argparse.ArgumentParser(description="R1 原版人形机器人 IK 位置跟踪测试")
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--output-video", type=str, default="r1_ik_position_test.mp4")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--trajectory", type=str, default="circle",
                        choices=["line", "circle", "square", "helix"])
    parser.add_argument("--ik-warmup-iters", type=int, default=IK_WARMUP_ITERS)
    parser.add_argument("--view", type=str, default="behind",
                        choices=["behind", "front", "topdown"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("IKTest")

    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    viewer = RobotHandDatasetSAPIENViewer(
        robot_names=[RobotName.r1_full],
        hand_type=HandType.right,
        headless=True,
    )
    scene = viewer.scene
    r1_robot = viewer.robots[0]

    robot_root_pos = np.array([0.0, 0.0, 0.0])
    r1_robot.set_root_pose(sapien.Pose(robot_root_pos.tolist(), Q_180Z.tolist()))

    active_joints = r1_robot.get_active_joints()
    joint_names = [j.get_name() for j in active_joints]

    right_arm_indices = [i for i, name in enumerate(joint_names) if "right_arm" in name]

    gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
    gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

    right_ee_link = None
    for link in r1_robot.get_links():
        if link.get_name() == "right_gripper_link":
            right_ee_link = link
            break

    for joint in active_joints:
        joint.set_drive_property(stiffness=100000.0, damping=10000.0)

    initial_qpos = r1_robot.get_qpos().copy()

    left_arm_default = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
    left_arm_indices = [i for i, name in enumerate(joint_names) if "left_arm" in name]
    for j, idx in enumerate(left_arm_indices):
        if j < len(left_arm_default):
            initial_qpos[idx] = left_arm_default[j]

    right_arm_starting = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
    for j, idx in enumerate(right_arm_indices):
        if j < len(right_arm_starting):
            initial_qpos[idx] = right_arm_starting[j]

    initial_qpos[gripper_idx1] = 0.04
    initial_qpos[gripper_idx2] = -0.04

    _set_qpos_with_drive(r1_robot, initial_qpos, active_joints, joint_names)
    for _ in range(5):
        scene.step()
    scene.update_render()

    right_arm_base_link = None
    for link in r1_robot.get_links():
        if link.get_name() == "right_arm_base_link":
            right_arm_base_link = link
            break

    base_link_pose = right_arm_base_link.get_entity_pose()
    base_link_p = np.array(base_link_pose.p)
    base_link_q = np.array(base_link_pose.q)

    base_link_R = pr.matrix_from_quaternion(base_link_q)
    base_link_R_inv = base_link_R.T

    ee_pose_init = right_ee_link.get_pose()
    ee_p_init = np.array(ee_pose_init.p)
    ee_in_base_init = base_link_R_inv @ (ee_p_init - base_link_p)
    logger.info(f"\n初始EE位置(世界帧): {ee_p_init}")
    logger.info(f"初始EE位置(arm_base帧): {ee_in_base_init}")
    logger.info(f"初始EE到base距离: {np.linalg.norm(ee_in_base_init):.4f}m")

    logger.info("=" * 80)
    logger.info(f"R1 原版人形机器人 IK 位置跟踪测试 | 轨迹: {args.trajectory} | 帧数: {args.num_frames}")
    logger.info("=" * 80)
    logger.info(f"base_link 位置: {base_link_p}")
    logger.info(f"base_link R:\n{base_link_R}")
    logger.info(f"RelaxedIK设置: {R1_RIGHT_SETTINGS}")

    traj_positions, traj_positions_arm, traj_orientations = generate_trajectory(
        args.trajectory, args.num_frames, base_link_p, base_link_R
    )

    logger.info(f"\n轨迹运动学约束检查:")
    distances = [np.linalg.norm(pos - base_link_p) for pos in traj_positions]
    logger.info(f"  最小距离: {min(distances):.3f}m, 最大距离: {max(distances):.3f}m, 平均: {np.mean(distances):.3f}m")
    if max(distances) > ARM_MAX_REACH * 0.95:
        logger.warning(f"  ⚠️  部分轨迹点超出95%臂展！")

    tolerances = [0.001, 0.001, 0.001, 10.0, 10.0, 10.0]

    ik_solver = RelaxedIKSolver(
        left_setting_file_path=str(R1_LEFT_SETTINGS),
        right_setting_file_path=str(R1_RIGHT_SETTINGS),
        tolerances=tolerances,
    )

    ik_solver.relaxed_ik_right.reset(right_arm_starting)

    first_pos_arm = traj_positions_arm[0]
    first_quat_world = traj_orientations[0]
    R_ee_first = pr.matrix_from_quaternion(first_quat_world)
    R_ee_base_first = base_link_R_inv @ R_ee_first
    first_quat_base = pr.quaternion_from_matrix(R_ee_base_first)

    logger.info(f"\n── IK预热：对第一个目标点迭代 {args.ik_warmup_iters} 次 ──")
    logger.info(f"  第一个目标(arm帧): {first_pos_arm}")
    logger.info(f"  第一个朝向(base帧, wxyz): {first_quat_base}")

    warmup_joints = _warmup_ik_solver(
        ik_solver, first_pos_arm, first_quat_base,
        n_iters=args.ik_warmup_iters, logger=logger
    )

    def solve_ik(pos_world, quat_world):
        pos_base = base_link_R_inv @ (pos_world - base_link_p)
        R_ee = pr.matrix_from_quaternion(quat_world)
        R_ee_base = base_link_R_inv @ R_ee
        ee_quat_base = pr.quaternion_from_matrix(R_ee_base)
        return np.array(ik_solver.solve_position_right(pos_base.tolist(), ee_quat_base.tolist()))

    test_qpos = initial_qpos.copy()
    for j, idx in enumerate(right_arm_indices):
        test_qpos[idx] = warmup_joints[j]

    r1_robot.set_qpos(test_qpos)
    ee_kin = np.array(right_ee_link.get_pose().p)
    err_kin = np.linalg.norm(ee_kin - traj_positions[0]) * 100

    _set_qpos_with_drive(r1_robot, test_qpos, active_joints, joint_names)
    for _ in range(5):
        scene.step()
    scene.update_render()
    ee_dyn = np.array(right_ee_link.get_entity_pose().p)
    err_dyn = np.linalg.norm(ee_dyn - traj_positions[0]) * 100

    logger.info(f"\n预热后 FK验证:")
    logger.info(f"  运动学FK: EE={ee_kin}, 误差={err_kin:.2f}cm")
    logger.info(f"  动力学FK: EE={ee_dyn}, 误差={err_dyn:.2f}cm")
    logger.info(f"  目标位置: {traj_positions[0]}")

    _set_qpos_with_drive(r1_robot, initial_qpos, active_joints, joint_names)
    for _ in range(5):
        scene.step()
    scene.update_render()

    current_right_joints = np.array([initial_qpos[i] for i in right_arm_indices])
    qpos_list = []

    for w in range(WARMUP_FRAMES):
        t = (w + 1) / WARMUP_FRAMES
        t_smooth = t * t * (3 - 2 * t)
        interp = current_right_joints * (1 - t_smooth) + warmup_joints * t_smooth

        qpos = r1_robot.get_qpos().copy()
        for j, idx in enumerate(right_arm_indices):
            qpos[idx] = interp[j]
        qpos[gripper_idx1] = 0.04
        qpos[gripper_idx2] = -0.04
        qpos_list.append(qpos)

    for local_idx in range(args.num_frames):
        pos_world = traj_positions[local_idx]
        quat_world = traj_orientations[local_idx]

        try:
            joints = solve_ik(pos_world, quat_world)
        except Exception as exc:
            logger.warning(f"帧 {local_idx} IK失败: {exc}")
            joints = warmup_joints

        qpos = r1_robot.get_qpos().copy()
        for j, idx in enumerate(right_arm_indices):
            qpos[idx] = joints[j]
        qpos[gripper_idx1] = 0.04
        qpos[gripper_idx2] = -0.04
        qpos_list.append(qpos)

        if (local_idx + 1) % 10 == 0:
            logger.info(f"  已计算 {local_idx + 1}/{args.num_frames} 帧 ...")

    logger.info(f"预计算完成: {len(qpos_list)} 帧")

    output_file = args.output_video
    logger.info(f"\n渲染视频 → {output_file}")

    robot_root = np.array(r1_robot.get_root_pose().p)

    if args.view == "behind":
        camera_pos = robot_root + np.array([2.5, 0.0, 1.2])
    elif args.view == "front":
        camera_pos = robot_root + np.array([-2.5, 0.0, 1.2])
    elif args.view == "topdown":
        camera_pos = robot_root + np.array([0.0, 0.0, 4.0])
    else:
        camera_pos = robot_root + np.array([2.5, 0.0, 1.2])
    camera_quat = CAMERA_QUATS.get(args.view, CAMERA_QUATS["behind"])

    camera = scene.add_camera(
        name="ik_test",
        width=1920,
        height=1080,
        fovy=np.deg2rad(60),
        near=0.01,
        far=200.0,
    )
    camera.set_local_pose(sapien.Pose(camera_pos.tolist(), camera_quat))

    render_fps = args.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_file, fourcc, render_fps,
        (camera.get_width(), camera.get_height()),
    )

    ee_trajectory = []
    all_errors_kin = []
    all_errors_dyn = []
    color_ee = (0, 255, 0)

    for frame_idx, qpos in enumerate(qpos_list):
        is_warmup = frame_idx < WARMUP_FRAMES
        data_frame_idx = frame_idx - WARMUP_FRAMES

        r1_robot.set_qpos(qpos)

        ee_pos_kin = None
        if not is_warmup:
            ee_pos_kin = np.array(right_ee_link.get_pose().p)
            if 0 <= data_frame_idx < len(traj_positions):
                err_kin = np.linalg.norm(ee_pos_kin - traj_positions[data_frame_idx])
                all_errors_kin.append(err_kin)

        _set_qpos_with_drive(r1_robot, qpos, active_joints, joint_names)
        for _ in range(3):
            scene.step()
        scene.update_render()

        ee_pos = None
        if not is_warmup:
            ee_pos = np.array(right_ee_link.get_entity_pose().p)
            ee_trajectory.append(ee_pos.copy())
            target_idx = data_frame_idx
            if 0 <= target_idx < len(traj_positions):
                err_dyn = np.linalg.norm(ee_pos - traj_positions[target_idx])
                all_errors_dyn.append(err_dyn)

        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        h, w = bgr.shape[:2]

        if is_warmup:
            t = (frame_idx + 1) / WARMUP_FRAMES
            cv2.rectangle(bgr, (0, 0), (w, 70), (0, 0, 128), -1)
            cv2.putText(bgr, f"WARMUP {frame_idx + 1}/{WARMUP_FRAMES}  ({t * 100:.0f}%)",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            cv2.putText(bgr, f"R1 v2 IK Position Test (original robot)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(bgr, f"Frame {data_frame_idx + 1}/{args.num_frames}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if ee_pos is not None and 0 <= data_frame_idx < len(traj_positions):
                err_cm = np.linalg.norm(ee_pos - traj_positions[data_frame_idx]) * 100
                err_color = (0, 255, 0) if err_cm < 2 else (0, 255, 255) if err_cm < 5 else (0, 0, 255)
                cv2.putText(bgr, f"Error(dyn): {err_cm:.1f}cm",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, err_color, 2)
            if ee_pos_kin is not None and 0 <= data_frame_idx < len(traj_positions):
                err_kin_cm = np.linalg.norm(ee_pos_kin - traj_positions[data_frame_idx]) * 100
                kin_color = (0, 255, 0) if err_kin_cm < 2 else (0, 255, 255) if err_kin_cm < 5 else (0, 0, 255)
                cv2.putText(bgr, f"Error(kin): {err_kin_cm:.1f}cm",
                            (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kin_color, 2)

            view_label = {"behind": "Behind", "front": "Front", "topdown": "Top-down"}.get(args.view, args.view)
            cv2.putText(bgr, f"View: {view_label}",
                        (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        axis_len = 0.15
        coord_origin = robot_root + np.array([-0.8, 0.0, 0.0])
        _draw_axes(bgr, coord_origin, np.eye(3), axis_len, camera, "WORLD", (200, 200, 200))
        _draw_axes(bgr, base_link_p, base_link_R_inv.T, axis_len * 0.8, camera, "ARM_BASE", (0, 200, 200))

        if ee_pos is not None:
            ee_q = np.array(right_ee_link.get_entity_pose().q)
            ee_R = pr.matrix_from_quaternion(ee_q)
            _draw_axes(bgr, ee_pos, ee_R, axis_len * 0.6, camera, "EE", color_ee)

        if not is_warmup and 0 <= data_frame_idx < len(traj_positions):
            tgt = traj_positions[data_frame_idx]
            tgt_2d = _project_point(tgt, camera)
            if tgt_2d is not None:
                cv2.circle(bgr, tgt_2d, 10, (0, 255, 255), 2)
                cv2.circle(bgr, tgt_2d, 4, (0, 255, 255), -1)
            _draw_axes(bgr, tgt, np.eye(3), axis_len * 0.5, camera, "IK_TGT", (0, 255, 255))

        if not is_warmup:
            pts_2d = [_project_point(p, camera) for p in traj_positions]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], (255, 255, 0), 1)

        if len(ee_trajectory) >= 2:
            pts_2d = [_project_point(p, camera) for p in ee_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], color_ee, 2)

        if ee_trajectory and not is_warmup and 0 <= data_frame_idx < len(traj_positions):
            ee_2d = _project_point(ee_trajectory[-1], camera)
            tgt_2d = _project_point(traj_positions[data_frame_idx], camera)
            if ee_2d is not None and tgt_2d is not None:
                err_cm = np.linalg.norm(ee_trajectory[-1] - traj_positions[data_frame_idx]) * 100
                line_color = (0, 255, 0) if err_cm < 2 else (0, 255, 255) if err_cm < 5 else (0, 0, 255)
                cv2.line(bgr, ee_2d, tgt_2d, line_color, 2)

        if not is_warmup:
            panel_x = w - 420
            panel_y = 120
            cv2.rectangle(bgr, (panel_x, panel_y), (w, panel_y + 200), (0, 0, 0), -1)

            row = 1
            if ee_pos is not None:
                cv2.putText(bgr, f"EE: {ee_pos[0]:.3f} {ee_pos[1]:.3f} {ee_pos[2]:.3f}",
                            (panel_x + 5, panel_y + 18 + row * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_ee, 1)
                row += 1

            if 0 <= data_frame_idx < len(traj_positions):
                tgt = traj_positions[data_frame_idx]
                cv2.putText(bgr, f"IK: {tgt[0]:.3f} {tgt[1]:.3f} {tgt[2]:.3f}",
                            (panel_x + 5, panel_y + 18 + row * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                row += 1

                ik_input = base_link_R_inv @ (tgt - base_link_p)
                cv2.putText(bgr, f"IK_in: {ik_input[0]:.3f} {ik_input[1]:.3f} {ik_input[2]:.3f}",
                            (panel_x + 5, panel_y + 18 + row * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
                row += 1

            cv2.putText(bgr, f"BASE: {base_link_p[0]:.3f} {base_link_p[1]:.3f} {base_link_p[2]:.3f}",
                        (panel_x + 5, panel_y + 18 + row * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1)
            row += 1

            if ee_pos is not None and 0 <= data_frame_idx < len(traj_positions):
                err = np.linalg.norm(ee_pos - traj_positions[data_frame_idx])
                err_color = (0, 255, 0) if err < 0.02 else (0, 255, 255) if err < 0.05 else (0, 0, 255)
                cv2.putText(bgr, f"Error: {err * 100:.2f}cm",
                            (panel_x + 5, panel_y + 18 + row * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, err_color, 1)
                row += 1

        if not is_warmup:
            legend_y = h - 80
            cv2.rectangle(bgr, (10, legend_y - 5), (450, h - 5), (0, 0, 0), -1)
            cv2.circle(bgr, (25, legend_y + 10), 5, color_ee, -1)
            cv2.putText(bgr, "EE trajectory", (40, legend_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ee, 1)
            cv2.circle(bgr, (25, legend_y + 30), 5, (0, 255, 255), -1)
            cv2.putText(bgr, "IK target", (40, legend_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.line(bgr, (25, legend_y + 50), (55, legend_y + 50), (255, 255, 0), 1)
            cv2.putText(bgr, "Desired trajectory", (60, legend_y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        writer.write(bgr)

    writer.release()
    logger.info(f"✓ 视频已保存: {output_file}")

    if all_errors_kin:
        errors_kin = np.array(all_errors_kin)
        logger.info(f"\n运动学FK误差统计:")
        logger.info(f"  平均: {np.mean(errors_kin) * 100:.2f}cm")
        logger.info(f"  中位: {np.median(errors_kin) * 100:.2f}cm")
        logger.info(f"  最大: {np.max(errors_kin) * 100:.2f}cm")
        logger.info(f"  <2cm: {np.sum(errors_kin < 0.02)}/{len(errors_kin)}")
        logger.info(f"  <5cm: {np.sum(errors_kin < 0.05)}/{len(errors_kin)}")

    if all_errors_dyn:
        errors_dyn = np.array(all_errors_dyn)
        logger.info(f"\n动力学FK误差统计:")
        logger.info(f"  平均: {np.mean(errors_dyn) * 100:.2f}cm")
        logger.info(f"  中位: {np.median(errors_dyn) * 100:.2f}cm")
        logger.info(f"  最大: {np.max(errors_dyn) * 100:.2f}cm")
        logger.info(f"  <2cm: {np.sum(errors_dyn < 0.02)}/{len(errors_dyn)}")
        logger.info(f"  <5cm: {np.sum(errors_dyn < 0.05)}/{len(errors_dyn)}")

    logger.info("\n" + "=" * 80)
    logger.info("测试完成!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
