#!/usr/bin/env python3
"""
================================================================================
  R1 Hand Tracking Simulation - PyBullet + Dex Retargeting + RelaxedIK
  (No SAPIEN dependency, no GPU required)
================================================================================

Pipeline:
  DexYCB -> MANO -> Dex Retargeting (gripper) -> RelaxedIK (arm) -> PyBullet -> Video

Key difference from SAPIEN version:
  - Uses PyBullet FK (instead of SAPIEN FK) to get gripper orientation from retargeting
  - Uses PyBullet software renderer (ER_TINY_RENDERER) instead of SAPIEN GPU renderer
  - No SAPIEN dependency at all

Usage:
  conda activate dex
  python r1_hand_tracking_sim.py --dexycb-dir /home/an/data/DexYCB --data-id 0
  python r1_hand_tracking_sim.py --dexycb-dir /home/an/data/DexYCB --data-id 0 --view front
  python r1_hand_tracking_sim.py --dexycb-dir /home/an/data/DexYCB --data-id 0 --num-frames 100
================================================================================
"""

import argparse
import logging
import os
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pybullet as p
import pybullet_data
import torch
from pytransform3d import rotations as pr

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
POSITION_RETARGETING_DIR = PROJECT_ROOT / "example" / "position_retargeting"

sys.path.insert(0, str(GALAXEA_SIM_PATH))
sys.path.insert(0, str(POSITION_RETARGETING_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import DexYCBVideoDataset, YCB_CLASSES
from dex_retargeting.constants import HandType, RobotName, RetargetingType, get_default_config_path
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.retargeting_config import RetargetingConfig
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver
from mano_layer import MANOLayer

R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"

LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
SAFETY_DISTANCE = 0.075
WARMUP_FRAMES = 30
ARM_BASE_OFFSET_LOCAL = np.array([0.09193, -0.33649, 0.97171])
ARM_MAX_REACH = 0.713
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]

R1_RIGHT_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798], [0.0, 3.1416], [-3.3161, 0.0],
    [-1.5708, 1.5708], [-1.5708, 1.5708], [-2.8798, 2.8798],
])

R1_FULL_JOINT_NAMES = [
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "torso_joint1", "torso_joint2", "torso_joint3",
    "right_gripper_finger_joint1", "right_gripper_finger_joint2",
    "left_gripper_finger_joint1", "left_gripper_finger_joint2",
]


class TrajectorySmoother:
    def __init__(self, fps=30, max_velocity=1.5, max_acceleration=4.0,
                 max_jerk=20.0, lp_alpha=0.25, max_iterations=10, convergence_eps=1e-5):
        self.dt = 1.0 / fps
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.lp_alpha = lp_alpha
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps

    def smooth_trajectory(self, qpos_sequence, smooth_indices):
        n_frames = len(qpos_sequence)
        n_joints = len(smooth_indices)
        trajectory = np.zeros((n_frames, n_joints))
        valid_mask = np.zeros(n_frames, dtype=bool)
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                trajectory[i] = qpos[smooth_indices]
                valid_mask[i] = True
        self._fill_invalid_frames(trajectory, valid_mask)
        trajectory_raw = trajectory.copy()
        trajectory = self._bidirectional_lowpass(trajectory)
        trajectory = self._iterative_clamp(trajectory)
        smoothed_sequence = []
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                qpos_new = qpos.copy()
            else:
                for j in range(i, -1, -1):
                    if qpos_sequence[j] is not None:
                        qpos_new = qpos_sequence[j].copy()
                        break
                else:
                    continue
            qpos_new[smooth_indices] = trajectory[i]
            smoothed_sequence.append(qpos_new)
        metrics = self._compute_metrics(trajectory, trajectory_raw)
        return smoothed_sequence, metrics

    def _fill_invalid_frames(self, trajectory, valid_mask):
        n_frames = len(trajectory)
        last_valid = 0
        for i in range(n_frames):
            if valid_mask[i]:
                last_valid = i
            else:
                trajectory[i] = trajectory[last_valid]
        first_valid = np.argmax(valid_mask)
        for i in range(first_valid):
            trajectory[i] = trajectory[first_valid]

    def _bidirectional_lowpass(self, trajectory):
        alpha = self.lp_alpha
        n = len(trajectory)
        s1_fwd = np.zeros_like(trajectory)
        s2_fwd = np.zeros_like(trajectory)
        s1_fwd[0] = trajectory[0]
        s2_fwd[0] = trajectory[0]
        for i in range(1, n):
            s1_fwd[i] = s1_fwd[i - 1] + alpha * (trajectory[i] - s1_fwd[i - 1])
            s2_fwd[i] = s2_fwd[i - 1] + alpha * (s1_fwd[i] - s2_fwd[i - 1])
        s1_bwd = np.zeros_like(trajectory)
        s2_bwd = np.zeros_like(trajectory)
        s1_bwd[-1] = s2_fwd[-1]
        s2_bwd[-1] = s2_fwd[-1]
        for i in range(n - 2, -1, -1):
            s1_bwd[i] = s1_bwd[i + 1] + alpha * (s2_fwd[i] - s1_bwd[i + 1])
            s2_bwd[i] = s2_bwd[i + 1] + alpha * (s1_bwd[i] - s2_bwd[i + 1])
        return s2_bwd

    def _clamp_velocity(self, trajectory):
        max_delta = self.max_velocity * self.dt
        for i in range(1, len(trajectory)):
            delta = trajectory[i] - trajectory[i - 1]
            trajectory[i] = trajectory[i - 1] + np.clip(delta, -max_delta, max_delta)
        return trajectory

    def _clamp_acceleration(self, trajectory):
        max_delta_v = self.max_acceleration * self.dt
        for i in range(2, len(trajectory)):
            v_prev = trajectory[i - 1] - trajectory[i - 2]
            v_curr = trajectory[i] - trajectory[i - 1]
            dv = np.clip(v_curr - v_prev, -max_delta_v, max_delta_v)
            trajectory[i] = trajectory[i - 1] + v_prev + dv
        return trajectory

    def _clamp_jerk(self, trajectory):
        max_delta_a = self.max_jerk * self.dt
        for i in range(3, len(trajectory)):
            v_im2 = trajectory[i - 2] - trajectory[i - 3]
            v_im1 = trajectory[i - 1] - trajectory[i - 2]
            v_i = trajectory[i] - trajectory[i - 1]
            a_prev = v_im1 - v_im2
            a_curr = v_i - v_im1
            da = np.clip(a_curr - a_prev, -max_delta_a, max_delta_a)
            trajectory[i] = trajectory[i - 1] + v_im1 + a_prev + da
        return trajectory

    def _iterative_clamp(self, trajectory):
        for _ in range(self.max_iterations):
            traj_before = trajectory.copy()
            trajectory = self._clamp_velocity(trajectory)
            trajectory = self._clamp_acceleration(trajectory)
            trajectory = self._clamp_jerk(trajectory)
            if np.max(np.abs(trajectory - traj_before)) < self.convergence_eps:
                break
        return trajectory

    def _compute_metrics(self, trajectory_smooth, trajectory_raw):
        dt = self.dt
        vel_raw = np.diff(trajectory_raw, axis=0) / dt
        acc_raw = np.diff(vel_raw, axis=0) / dt
        jerk_raw = np.diff(acc_raw, axis=0) / dt
        vel_smooth = np.diff(trajectory_smooth, axis=0) / dt
        acc_smooth = np.diff(vel_smooth, axis=0) / dt
        jerk_smooth = np.diff(acc_smooth, axis=0) / dt
        raw_max_vel = float(np.max(np.abs(vel_raw)))
        raw_si = float(np.sum(jerk_raw ** 2) * dt) if len(jerk_raw) > 0 else 0.0
        smooth_max_vel = float(np.max(np.abs(vel_smooth)))
        smooth_max_acc = float(np.max(np.abs(acc_smooth))) if len(acc_smooth) > 0 else 0.0
        smooth_max_jerk = float(np.max(np.abs(jerk_smooth))) if len(jerk_smooth) > 0 else 0.0
        smooth_si = float(np.sum(jerk_smooth ** 2) * dt) if len(jerk_smooth) > 0 else 0.0
        si_improvement = 1.0 - smooth_si / max(raw_si, 1e-12)
        return {
            "smooth_max_velocity": smooth_max_vel,
            "smooth_max_acceleration": smooth_max_acc,
            "smooth_max_jerk": smooth_max_jerk,
            "si_improvement": si_improvement,
            "raw_max_velocity": raw_max_vel,
        }


def _joint_comfort_score(joint_values, joint_limits):
    mid = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
    half_range = (joint_limits[:, 1] - joint_limits[:, 0]) / 2
    return float(1.0 - np.mean(np.abs(joint_values - mid) / half_range))


def quat_xyzw_to_wxyz(q_xyzw):
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def compute_hand_geometry(mano_layer, hand_pose_frame, camera_extrinsic):
    if np.abs(hand_pose_frame).sum() < 1e-5:
        return None, None
    if hand_pose_frame.ndim == 1:
        hand_pose_frame = hand_pose_frame[np.newaxis, :]
    pose_params = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))
    trans_params = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))
    vertex, joint = mano_layer(pose_params, trans_params)
    vertex = vertex.cpu().numpy()[0]
    joint = joint.cpu().numpy()[0]
    camera_mat = np.linalg.inv(camera_extrinsic)
    vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
    joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
    return vertex, joint


class PyBulletRenderer:
    def __init__(self, robot_urdf_path, base_pos, base_quat_xyzw, width=1280, height=720):
        self.width = width
        self.height = height
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF(
            robot_urdf_path,
            basePosition=base_pos.tolist(),
            baseOrientation=base_quat_xyzw,
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )

        self.right_arm_indices = []
        self.left_arm_indices = []
        self.torso_indices = []
        self.right_gripper_indices = []
        self.left_gripper_indices = []
        self.right_ee_link_index = -1
        self.left_ee_link_index = -1
        self.right_base_link_index = -1

        self.joint_name_to_idx = {}
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode()
            link_name = info[12].decode()
            joint_type = info[2]
            self.joint_name_to_idx[joint_name] = i

            if joint_name.startswith("right_arm_joint") and joint_type == p.JOINT_REVOLUTE:
                num = ''.join(c for c in joint_name if c.isdigit())
                self.right_arm_indices.append((int(num), i))
            elif joint_name.startswith("left_arm_joint") and joint_type == p.JOINT_REVOLUTE:
                num = ''.join(c for c in joint_name if c.isdigit())
                self.left_arm_indices.append((int(num), i))
            elif joint_name.startswith("torso_joint") and joint_type == p.JOINT_REVOLUTE:
                self.torso_indices.append(i)
            elif joint_name.startswith("right_gripper_finger_joint"):
                self.right_gripper_indices.append(i)
            elif joint_name.startswith("left_gripper_finger_joint"):
                self.left_gripper_indices.append(i)

            if link_name == "right_gripper_link":
                self.right_ee_link_index = i
            elif link_name == "right_arm_base_link":
                self.right_base_link_index = i

        self.right_arm_indices.sort(key=lambda x: x[0])
        self.left_arm_indices.sort(key=lambda x: x[0])
        self.right_arm_indices = [idx for _, idx in self.right_arm_indices]
        self.left_arm_indices = [idx for _, idx in self.left_arm_indices]

        for idx in self.right_arm_indices + self.left_arm_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL, force=0)
        for idx in self.right_gripper_indices + self.left_gripper_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL, force=0)
        for idx in self.torso_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=1000)

        self._set_initial_qpos()
        for _ in range(60):
            p.stepSimulation()
            self._hold_all_joints()

    def _set_initial_qpos(self):
        for i, idx in enumerate(self.right_arm_indices):
            if i < len(RIGHT_ARM_STARTING):
                p.resetJointState(self.robot_id, idx, RIGHT_ARM_STARTING[i])
        for idx in self.torso_indices:
            p.resetJointState(self.robot_id, idx, 0.0)
        for idx in self.right_gripper_indices + self.left_gripper_indices:
            p.resetJointState(self.robot_id, idx, 0.0)
        self._hold_all_joints()

    def _hold_all_joints(self):
        for idx in self.right_arm_indices + self.left_arm_indices:
            pos = p.getJointState(self.robot_id, idx)[0]
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=pos, force=500)
        for idx in self.torso_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=1000)
        for idx in self.right_gripper_indices + self.left_gripper_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=100)

    def set_qpos_by_names(self, joint_names, qpos_values):
        for j_idx, j_name in enumerate(joint_names):
            if j_idx < len(qpos_values) and j_name in self.joint_name_to_idx:
                pb_idx = self.joint_name_to_idx[j_name]
                val = float(qpos_values[j_idx])
                p.resetJointState(self.robot_id, pb_idx, val)
                p.setJointMotorControl2(self.robot_id, pb_idx, p.POSITION_CONTROL,
                                       targetPosition=val, force=500)
        for _ in range(5):
            p.stepSimulation()
            self._hold_all_joints()

    def set_qpos_for_fk(self, joint_names, qpos_values):
        for j_idx, j_name in enumerate(joint_names):
            if j_idx < len(qpos_values) and j_name in self.joint_name_to_idx:
                pb_idx = self.joint_name_to_idx[j_name]
                val = float(qpos_values[j_idx])
                p.resetJointState(self.robot_id, pb_idx, val)

    def get_link_pose_world(self, link_index):
        if link_index < 0:
            return None, None
        state = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        pos = np.array(state[4])
        quat_wxyz = quat_xyzw_to_wxyz(np.array(state[5]))
        return pos, quat_wxyz

    def get_ee_pose_world(self):
        return self.get_link_pose_world(self.right_ee_link_index)

    def get_base_pose_world(self):
        return self.get_link_pose_world(self.right_base_link_index)

    def render_frame(self, camera_yaw, camera_pitch, camera_distance, camera_target):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            camera_target, camera_distance, camera_yaw, camera_pitch, 0, 2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            60, self.width / float(self.height), 0.01, 100.0
        )
        _, _, rgba, _, _ = p.getCameraImage(
            self.width, self.height, view_matrix, proj_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb = np.uint8(rgba)[..., :3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def project_point(self, point_3d, camera_yaw, camera_pitch, camera_distance, camera_target):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            camera_target, camera_distance, camera_yaw, camera_pitch, 0, 2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            60, self.width / float(self.height), 0.01, 100.0
        )
        view = np.array(view_matrix).reshape(4, 4)
        proj = np.array(proj_matrix).reshape(4, 4)
        p_hom = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
        p_cam = view @ p_hom
        if p_cam[2] >= -0.01:
            return None
        p_proj = proj @ p_cam
        if abs(p_proj[3]) < 1e-8:
            return None
        u = int((p_proj[0] / p_proj[3] + 1.0) * 0.5 * self.width)
        v = int((1.0 - p_proj[1] / p_proj[3]) * 0.5 * self.height)
        if 0 <= u < self.width and 0 <= v < self.height:
            return (u, v)
        return None

    def disconnect(self):
        if p.isConnected():
            p.disconnect()


def _make_pybullet_urdf(urdf_path_str):
    import trimesh
    urdf_path = Path(urdf_path_str)
    with open(urdf_path, 'r') as f:
        content = f.read()
    tmp_dir = tempfile.mkdtemp(prefix="r1_render_")
    mesh_dir = urdf_path.parent / "meshes"
    if mesh_dir.exists():
        os.symlink(str(mesh_dir), os.path.join(tmp_dir, "meshes"))
    obj_dir = os.path.join(tmp_dir, "meshes_obj")
    os.makedirs(obj_dir, exist_ok=True)
    converted = set()

    def replace_glb(match):
        filename = match.group(1)
        clean = re.sub(r'^\.\/', '', filename)
        if filename.endswith('.usd.sapien.glb') or filename.endswith('.glb'):
            base = re.sub(r'\.usd\.sapien\.glb$', '', clean)
            base = re.sub(r'\.glb$', '', base)
            mesh_name = os.path.basename(base) + ".obj"
            if mesh_name not in converted:
                converted.add(mesh_name)
                glb_full = mesh_dir / clean.replace('meshes/', '', 1) if clean.startswith('meshes/') else mesh_dir / clean
                obj_full = os.path.join(obj_dir, mesh_name)
                if glb_full.exists() and not os.path.exists(obj_full):
                    try:
                        scene_obj = trimesh.load(str(glb_full))
                        mesh = scene_obj.to_mesh() if isinstance(scene_obj, trimesh.Scene) else scene_obj
                        mesh.export(obj_full)
                    except Exception:
                        return match.group(0)
            return f'filename="meshes_obj/{mesh_name}"'
        return match.group(0)

    content = re.sub(r'filename="([^"]*\.(?:usd\.sapien\.)?glb)"', replace_glb, content)
    tmp_urdf = os.path.join(tmp_dir, urdf_path.name)
    with open(tmp_urdf, 'w') as f:
        f.write(content)
    return tmp_urdf, tmp_dir


def _compute_ee_orientation_from_mano(hand_pose_frame, camera_extrinsic, base_link_R_inv):
    if hand_pose_frame.ndim == 1:
        hand_pose_frame = hand_pose_frame[np.newaxis, :]
    wrist_axis_angle = hand_pose_frame[0, :3].astype(np.float64)
    if np.abs(wrist_axis_angle).sum() < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    R_wrist_cam = pr.matrix_from_compact_axis_angle(wrist_axis_angle)
    camera_mat = np.linalg.inv(camera_extrinsic)
    R_wrist_world = camera_mat[:3, :3] @ R_wrist_cam
    OPERATOR2MANO_RIGHT = np.array([
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)
    R_ee_world = R_wrist_world @ OPERATOR2MANO_RIGHT.T
    R_ee_base = base_link_R_inv @ R_ee_world
    ee_quat_base = pr.quaternion_from_matrix(R_ee_base)
    return ee_quat_base


class R1HandTrackingPipeline:
    def __init__(self, dexycb_dir, data_id=0, output_video="r1_tracking_sim.mp4",
                 fps=30, view="behind", logger=None):
        self.dexycb_dir = Path(dexycb_dir)
        self.data_id = data_id
        self.output_video = output_video
        self.fps = fps
        self.view = view
        self.logger = logger or logging.getLogger("R1TrackingSim")

    def run(self, start_frame=0, num_frames=50):
        self.logger.info("=" * 80)
        self.logger.info("DexYCB -> Dex Retargeting -> RelaxedIK -> PyBullet Video (No SAPIEN, No GPU)")
        self.logger.info("=" * 80)

        self.logger.info("\n[1/8] Loading DexYCB ...")
        dataset = DexYCBVideoDataset(self.dexycb_dir, hand_type="right")
        sampled_data = dataset[self.data_id]
        hand_pose = sampled_data["hand_pose"]
        hand_shape = sampled_data["hand_shape"]
        extrinsics = sampled_data["extrinsics"]
        total_frames = hand_pose.shape[0]
        self.logger.info(f"  Trajectory: {total_frames} frames, objects: {[YCB_CLASSES[yid] for yid in sampled_data['ycb_ids']]}")

        actual_frames = min(num_frames, total_frames - start_frame)
        if actual_frames <= 0:
            raise ValueError(f"Invalid frame range")

        self.logger.info("\n[2/8] Initializing MANO + Dex Retargeting ...")
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        mano_layer = MANOLayer("right", hand_shape.astype(np.float32))

        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, HandType.right)
        override = dict(add_dummy_free_joint=True)
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()

        retarget_joint_names = retargeting.joint_names
        self.logger.info(f"  Retargeting joints: {retarget_joint_names}")
        self.logger.info("  MANO + Dex Retargeting ready (no SAPIEN)")

        self.logger.info("\n[3/8] Analyzing hand trajectory ...")
        wrist_positions, hand_stats = self._analyze_hand_trajectory(
            hand_pose, mano_layer, extrinsics, start_frame, actual_frames
        )
        self.logger.info(f"  Valid frames: {len(wrist_positions)}")
        self.logger.info(f"  Wrist centroid: [{hand_stats['centroid'][0]:.4f}, {hand_stats['centroid'][1]:.4f}, {hand_stats['centroid'][2]:.4f}]")

        self.logger.info("\n[4/8] Computing robot placement ...")
        R_180Z_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        arm_base_offset_world = R_180Z_mat @ ARM_BASE_OFFSET_LOCAL
        centroid = hand_stats["centroid"]
        robot_root_y = centroid[1] - arm_base_offset_world[1] - 0.3
        robot_root_pos = np.array([0.0, robot_root_y, 0.0])
        robot_root_quat_xyzw = np.array([0.0, 0.0, 1.0, 0.0])
        self.logger.info(f"  Robot base: [{robot_root_pos[0]:.4f}, {robot_root_pos[1]:.4f}, {robot_root_pos[2]:.4f}]")

        base_link_p = robot_root_pos + R_180Z_mat @ ARM_BASE_OFFSET_LOCAL
        base_link_R = R_180Z_mat
        base_link_R_inv = base_link_R.T
        self.logger.info(f"  right_arm_base_link: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")

        self.logger.info("\n[5/8] Computing workspace mapping ...")
        mapping_info = self._compute_workspace_mapping(hand_stats, base_link_p, base_link_R)
        self.logger.info(f"  Mapping offset: [{mapping_info['mapping_offset'][0]:.4f}, {mapping_info['mapping_offset'][1]:.4f}, {mapping_info['mapping_offset'][2]:.4f}]")

        self.logger.info("\n[6/8] Initializing RelaxedIK + PyBullet FK renderer ...")
        ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=[0.001, 0.001, 0.001, 10.0, 10.0, 10.0],
        )
        ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

        r1_full_urdf = str(PROJECT_ROOT / "assets" / "robots" / "r1_full" / "r1_lite_robot_glb.urdf")
        tmp_urdf, tmp_dir = _make_pybullet_urdf(r1_full_urdf)
        renderer = PyBulletRenderer(
            tmp_urdf, robot_root_pos, robot_root_quat_xyzw,
            width=1280, height=720,
        )
        self.logger.info("  RelaxedIK + PyBullet FK renderer ready")

        self.logger.info(f"\n[7/8] Precomputing {actual_frames} frames ...")
        qpos_sequence, ik_targets_world, eval_pre = self._precompute(
            hand_pose, start_frame, actual_frames,
            mano_layer, extrinsics, retargeting,
            ik_solver, base_link_p, base_link_R_inv, mapping_info,
            renderer,
        )

        valid = sum(1 for x in qpos_sequence if x is not None)
        self.logger.info(f"  Precompute done: {valid}/{actual_frames + WARMUP_FRAMES} valid")

        right_arm_sapien_indices = [0, 1, 2, 3, 4, 5]
        gripper_sapien_indices = [15, 16]
        smooth_indices = np.array(right_arm_sapien_indices + gripper_sapien_indices)
        smoother = TrajectorySmoother(fps=self.fps, max_velocity=1.5, max_acceleration=4.0,
                                      max_jerk=20.0, lp_alpha=0.25)
        warmup_qpos = qpos_sequence[:WARMUP_FRAMES]
        data_qpos = qpos_sequence[WARMUP_FRAMES:]
        data_smoothed, smooth_metrics = smoother.smooth_trajectory(data_qpos, smooth_indices)
        qpos_sequence = warmup_qpos + data_smoothed
        self.logger.info(f"  Smoothing: vel={smooth_metrics['smooth_max_velocity']:.2f} rad/s, "
                         f"acc={smooth_metrics['smooth_max_acceleration']:.2f} rad/s^2")

        self.logger.info(f"\n[8/8] Rendering video ...")
        eval_render = self._render_video(
            qpos_sequence, ik_targets_world,
            robot_root_pos, robot_root_quat_xyzw,
            base_link_p, base_link_R_inv, mapping_info,
            right_arm_sapien_indices, gripper_sapien_indices,
            renderer,
        )

        renderer.disconnect()
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

        self._output_evaluation(eval_pre, eval_render, hand_stats, mapping_info, valid, actual_frames, smooth_metrics)
        self.logger.info(f"\nPipeline complete! Video: {self.output_video}")

    def _analyze_hand_trajectory(self, hand_pose, mano_layer, extrinsics, start_frame, num_frames):
        wrist_positions = []
        for i in range(num_frames):
            global_idx = start_frame + i
            hand_frame = hand_pose[global_idx]
            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            _, joints = compute_hand_geometry(mano_layer, hand_frame, extrinsics)
            if joints is not None:
                wrist_positions.append(joints[0, :3].copy())
        if not wrist_positions:
            raise RuntimeError("No valid wrist positions")
        positions = np.array(wrist_positions)
        return wrist_positions, {
            "centroid": np.mean(positions, axis=0),
            "range": np.ptp(positions, axis=0),
            "std": np.std(positions, axis=0),
            "num_valid": len(wrist_positions),
        }

    def _compute_workspace_mapping(self, hand_stats, base_link_p, base_link_R):
        centroid = hand_stats["centroid"]
        comfort_target_world = base_link_R @ COMFORT_TARGET_IN_BASE + base_link_p
        mapping_offset = comfort_target_world - centroid
        approach_dir = base_link_p - comfort_target_world
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        safety_offset = approach_dir * SAFETY_DISTANCE
        mapped_centroid = centroid + mapping_offset + safety_offset
        mapped_in_base = base_link_R.T @ (mapped_centroid - base_link_p)
        return {
            "mapping_offset": mapping_offset,
            "safety_offset": safety_offset,
            "mapped_dist_to_base": np.linalg.norm(mapped_in_base),
        }

    def _precompute(self, hand_pose, start_frame, num_frames,
                    mano_layer, extrinsics, retargeting,
                    ik_solver, base_link_p, base_link_R_inv, mapping_info):
        qpos_sequence = []
        ik_targets_world = []
        eval_data = {"ik_errors": [], "out_of_reach": 0}

        ref_indices = retargeting.optimizer.target_link_human_indices
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
        retarget_joint_names = retargeting.joint_names

        retarget2sapien = {}
        sapien_idx = 0
        for name in R1_FULL_JOINT_NAMES:
            if name in retarget_joint_names:
                retarget2sapien[retarget_joint_names.index(name)] = sapien_idx
            sapien_idx += 1

        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        init_sapien_qpos = np.zeros(len(R1_FULL_JOINT_NAMES))
        for i, idx in enumerate([0, 1, 2, 3, 4, 5]):
            init_sapien_qpos[idx] = RIGHT_ARM_STARTING[i]
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in retarget2sapien:
                fixed_qpos[i] = init_sapien_qpos[retarget2sapien[retarget_idx]]

        mapping_offset = mapping_info["mapping_offset"]
        safety_offset = mapping_info["safety_offset"]

        first_ik_joints = None
        first_ik_target_world = None
        first_ik_target_base = None
        first_ee_quat_base = None

        for probe_idx in range(num_frames):
            global_idx = start_frame + probe_idx
            hand_frame = hand_pose[global_idx]
            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            if np.abs(hand_frame).sum() < 1e-5:
                continue
            _, joints = compute_hand_geometry(mano_layer, hand_frame, extrinsics)
            if joints is None:
                continue

            wrist_pos_world = joints[0, :3].copy()
            ref_value = joints[ref_indices, :].astype(np.float32)
            retargeting.retarget(ref_value, fixed_qpos)

            ik_target_world_raw = wrist_pos_world + mapping_offset + safety_offset
            first_ik_target_world = ik_target_world_raw.copy()
            ik_target_base = base_link_R_inv @ (ik_target_world_raw - base_link_p)

            ee_quat_base = _compute_ee_orientation_from_mano(
                hand_frame, extrinsics, base_link_R_inv
            )

            try:
                first_ik_joints = np.array(
                    ik_solver.solve_position_right(ik_target_base.tolist(), ee_quat_base.tolist())
                )
                first_ik_target_base = ik_target_base.copy()
                first_ee_quat_base = ee_quat_base.copy()
                break
            except Exception:
                continue

        if first_ik_joints is None:
            raise RuntimeError("Cannot solve IK for any valid frame")

        for _ in range(200):
            first_ik_joints = np.array(
                ik_solver.solve_position_right(first_ik_target_base.tolist(), first_ee_quat_base.tolist())
            )
        self.logger.info(f"  IK warmup done, joints: {first_ik_joints}")

        current_right_joints = np.array(RIGHT_ARM_STARTING, dtype=np.float64)
        ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
        ee_pos_filter.next(first_ik_target_world)
        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        joint_filter.next(current_right_joints)

        for w in range(WARMUP_FRAMES):
            t = (w + 1) / WARMUP_FRAMES
            t_smooth = t * t * (3 - 2 * t)
            interp = current_right_joints * (1 - t_smooth) + first_ik_joints * t_smooth
            interp = joint_filter.next(interp)
            sapien_qpos = np.zeros(len(R1_FULL_JOINT_NAMES))
            for j in range(6):
                sapien_qpos[j] = interp[j]
            sapien_qpos[15] = 0.04
            sapien_qpos[16] = -0.04
            qpos_sequence.append(sapien_qpos)
            ik_targets_world.append(first_ik_target_world.copy())

        for local_idx in range(num_frames):
            global_idx = start_frame + local_idx
            hand_frame = hand_pose[global_idx]
            if hand_frame.ndim == 1:
                hand_frame = hand_frame[np.newaxis, :]
            if np.abs(hand_frame).sum() < 1e-5:
                qpos_sequence.append(None)
                ik_targets_world.append(None)
                continue

            _, joints = compute_hand_geometry(mano_layer, hand_frame, extrinsics)
            if joints is None:
                qpos_sequence.append(None)
                ik_targets_world.append(None)
                continue

            wrist_pos_world = joints[0, :3].copy()
            ref_value = joints[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

            gripper1 = 0.0
            gripper2 = 0.0
            for retarget_idx, sapien_idx in retarget2sapien.items():
                if retarget_idx < len(retarget_qpos):
                    val = float(retarget_qpos[retarget_idx])
                    if sapien_idx == 15:
                        gripper1 = val
                    elif sapien_idx == 16:
                        gripper2 = val

            ee_quat_base = _compute_ee_orientation_from_mano(
                hand_frame, extrinsics, base_link_R_inv
            )

            ik_target_world_raw = wrist_pos_world + mapping_offset + safety_offset
            ik_target_world_filtered = ee_pos_filter.next(ik_target_world_raw)
            ik_targets_world.append(ik_target_world_filtered.copy())

            ik_target_base = base_link_R_inv @ (ik_target_world_filtered - base_link_p)
            dist_to_base = np.linalg.norm(ik_target_base)
            if dist_to_base > ARM_MAX_REACH:
                eval_data["out_of_reach"] += 1

            try:
                right_joints = np.array(
                    ik_solver.solve_position_right(ik_target_base.tolist(), ee_quat_base.tolist())
                )
            except Exception as exc:
                self.logger.warning(f"  Frame {global_idx}: IK failed - {exc}")
                qpos_sequence.append(None)
                eval_data["ik_errors"].append(str(exc))
                continue

            right_joints = joint_filter.next(right_joints)

            sapien_qpos = np.zeros(len(R1_FULL_JOINT_NAMES))
            for j in range(min(6, len(right_joints))):
                sapien_qpos[j] = right_joints[j]
            sapien_qpos[15] = gripper1
            sapien_qpos[16] = gripper2
            qpos_sequence.append(sapien_qpos)

            if (local_idx + 1) % 10 == 0:
                self.logger.info(f"  Computed {local_idx + 1}/{num_frames} frames ...")

        return qpos_sequence, ik_targets_world, eval_data

    def _render_video(self, qpos_sequence, ik_targets_world,
                      robot_root_pos, robot_root_quat_xyzw,
                      base_link_p, base_link_R_inv, mapping_info,
                      right_arm_indices, gripper_indices,
                      renderer):
        self.logger.info(f"\nRendering video -> {self.output_video}  [view: {self.view}]")

        if self.view == "behind":
            camera_yaw, camera_pitch = -135, -25
        elif self.view == "front":
            camera_yaw, camera_pitch = 45, -25
        elif self.view == "topdown":
            camera_yaw, camera_pitch = 0, -80
        else:
            camera_yaw, camera_pitch = -135, -25

        camera_distance = 2.5
        camera_target = [robot_root_pos[0], robot_root_pos[1], robot_root_pos[2] + 0.8]

        render_fps = int(self.fps * 0.75)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output_video, fourcc, render_fps,
                                 (renderer.width, renderer.height))

        fk_errors = []
        comfort_scores = []
        ee_trajectory = []
        ik_target_trajectory = []

        for frame_idx, qpos in enumerate(qpos_sequence):
            is_warmup = frame_idx < WARMUP_FRAMES
            data_frame_idx = frame_idx - WARMUP_FRAMES

            if qpos is not None:
                renderer.set_qpos_by_names(R1_FULL_JOINT_NAMES, qpos)

            ee_pos = None
            ee_quat_wxyz = None

            if not is_warmup and qpos is not None:
                ee_pos, ee_quat_wxyz = renderer.get_ee_pose_world()
                if ee_pos is not None:
                    ee_trajectory.append(ee_pos.copy())
                if ik_targets_world[frame_idx] is not None:
                    ik_target_trajectory.append(ik_targets_world[frame_idx].copy())
                if ee_pos is not None and ik_targets_world[frame_idx] is not None:
                    fk_errors.append(np.linalg.norm(ee_pos - ik_targets_world[frame_idx]))
                right_joints = np.array([qpos[i] for i in right_arm_indices])
                comfort_scores.append(_joint_comfort_score(right_joints, R1_RIGHT_JOINT_LIMITS))

            bgr = renderer.render_frame(camera_yaw, camera_pitch, camera_distance, camera_target)
            bgr = self._draw_annotations(
                bgr, frame_idx, qpos_sequence, is_warmup,
                ee_trajectory, ik_target_trajectory,
                robot_root_pos, base_link_p, base_link_R_inv, mapping_info,
                renderer, camera_yaw, camera_pitch, camera_distance, camera_target,
                ee_pos, ee_quat_wxyz,
            )
            writer.write(bgr)

            if (frame_idx + 1) % 10 == 0:
                self.logger.info(f"  Rendered {frame_idx + 1}/{len(qpos_sequence)} frames ...")

        writer.release()
        self.logger.info(f"Video saved: {self.output_video}")
        return {"fk_errors": fk_errors, "comfort_scores": comfort_scores}

    def _draw_annotations(self, bgr, frame_idx, qpos_sequence, is_warmup,
                          ee_trajectory, ik_target_trajectory,
                          robot_root, base_link_p, base_link_R_inv, mapping_info,
                          renderer, cy, cp, cd, ct, ee_pos, ee_quat_wxyz):
        h, w = bgr.shape[:2]
        data_frame_idx = frame_idx - WARMUP_FRAMES

        if is_warmup:
            t = (frame_idx + 1) / WARMUP_FRAMES
            cv2.rectangle(bgr, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(bgr, f"Warmup {frame_idx+1}/{WARMUP_FRAMES} ({t*100:.0f}%)",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            cv2.rectangle(bgr, (0, 50), (int(w * t), 56), (0, 200, 255), -1)
        else:
            cv2.rectangle(bgr, (0, 0), (w, 80), (0, 0, 0), -1)
            if data_frame_idx == 0:
                cv2.rectangle(bgr, (0, 80), (w, 140), (0, 100, 255), -1)
                cv2.putText(bgr, ">>> MAPPING START <<<",
                            (w//2 - 250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(bgr, f"Frame {data_frame_idx+1}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if ee_trajectory and ik_target_trajectory:
                err_cm = np.linalg.norm(ee_trajectory[-1] - ik_target_trajectory[-1]) * 100
                err_color = (0, 255, 0) if err_cm < 2 else (0, 255, 255) if err_cm < 5 else (0, 0, 255)
                cv2.putText(bgr, f"EE-IK: {err_cm:.1f}cm",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, err_color, 2)
            view_label = {"behind": "Behind", "front": "Front", "topdown": "Top-down"}.get(self.view, self.view)
            cv2.putText(bgr, f"View: {view_label}  0.75x",
                        (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        self._draw_axes(bgr, np.array([0, 0, 0]), np.eye(3), 0.15, renderer, cy, cp, cd, ct, "WORLD", (200, 200, 200))
        robot_root_R = pr.matrix_from_quaternion([0.0, 0.0, 1.0, 0.0])
        self._draw_axes(bgr, robot_root, robot_root_R, 0.12, renderer, cy, cp, cd, ct, "ROBOT", (0, 200, 200))
        self._draw_axes(bgr, base_link_p, base_link_R_inv.T, 0.12, renderer, cy, cp, cd, ct, "ARM_BASE", (200, 200, 0))
        if ee_pos is not None and ee_quat_wxyz is not None:
            ee_R = pr.matrix_from_quaternion(ee_quat_wxyz)
            self._draw_axes(bgr, ee_pos, ee_R, 0.09, renderer, cy, cp, cd, ct, "EE", (0, 255, 0))

        if len(ee_trajectory) >= 2:
            pts_2d = [renderer.project_point(pt, cy, cp, cd, ct) for pt in ee_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                alpha_val = max(0.3, i / len(pts_2d))
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i],
                         (0, int(255 * alpha_val), int(255 * (1 - alpha_val))), 2)
        if len(ik_target_trajectory) >= 2:
            pts_2d = [renderer.project_point(pt, cy, cp, cd, ct) for pt in ik_target_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], (0, 255, 255), 1)

        if ee_trajectory:
            ee_2d = renderer.project_point(ee_trajectory[-1], cy, cp, cd, ct)
            if ee_2d is not None:
                cv2.circle(bgr, ee_2d, 6, (0, 255, 0), -1)
        if ik_target_trajectory:
            tgt_2d = renderer.project_point(ik_target_trajectory[-1], cy, cp, cd, ct)
            if tgt_2d is not None:
                cv2.circle(bgr, tgt_2d, 6, (0, 255, 255), -1)

        if not is_warmup:
            panel_x, panel_y = w - 380, 90
            cv2.rectangle(bgr, (panel_x, panel_y), (w, panel_y + 180), (0, 0, 0), -1)
            row = 0
            if ee_pos is not None:
                cv2.putText(bgr, f"EE: {ee_pos[0]:.3f} {ee_pos[1]:.3f} {ee_pos[2]:.3f}",
                            (panel_x + 5, panel_y + 20 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                row += 1
            if ik_target_trajectory:
                ik_p = ik_target_trajectory[-1]
                cv2.putText(bgr, f"IK: {ik_p[0]:.3f} {ik_p[1]:.3f} {ik_p[2]:.3f}",
                            (panel_x + 5, panel_y + 20 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                row += 1
            cv2.putText(bgr, f"BASE: {base_link_p[0]:.3f} {base_link_p[1]:.3f} {base_link_p[2]:.3f}",
                        (panel_x + 5, panel_y + 20 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
            row += 1
            if ee_pos is not None and ik_target_trajectory:
                err = np.linalg.norm(ee_pos - ik_target_trajectory[-1])
                err_color = (0, 255, 0) if err < 0.02 else (0, 255, 255) if err < 0.05 else (0, 0, 255)
                cv2.putText(bgr, f"EE-IK err: {err*100:.2f}cm",
                            (panel_x + 5, panel_y + 20 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, err_color, 1)

        if not is_warmup:
            ly = h - 60
            cv2.rectangle(bgr, (10, ly - 5), (300, h - 5), (0, 0, 0), -1)
            cv2.circle(bgr, (25, ly + 10), 5, (0, 255, 0), -1)
            cv2.putText(bgr, "Robot EE", (40, ly + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.circle(bgr, (25, ly + 30), 5, (0, 255, 255), -1)
            cv2.putText(bgr, "IK target", (40, ly + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return bgr

    def _draw_axes(self, bgr, origin, R, length, renderer, cy, cp, cd, ct, label, label_color):
        origin_2d = renderer.project_point(origin, cy, cp, cd, ct)
        if origin_2d is None:
            return
        for axis_idx, (color, letter) in enumerate(zip([(0,0,255),(0,255,0),(255,0,0)], ["X","Y","Z"])):
            end = origin + R[:, axis_idx] * length
            end_2d = renderer.project_point(end, cy, cp, cd, ct)
            if end_2d is not None:
                cv2.line(bgr, origin_2d, end_2d, color, 2)
                cv2.putText(bgr, letter, (end_2d[0]+3, end_2d[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(bgr, label, (origin_2d[0]-15, origin_2d[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

    def _output_evaluation(self, eval_pre, eval_render, hand_stats, mapping_info, valid, total, smooth_metrics):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Evaluation Report")
        self.logger.info("=" * 80)
        self.logger.info(f"\n-- 1. Hand Trajectory --")
        self.logger.info(f"  Valid frames: {hand_stats['num_valid']}, centroid: {hand_stats['centroid']}")
        self.logger.info(f"\n-- 2. Workspace Mapping --")
        self.logger.info(f"  Mapped dist to base: {mapping_info['mapped_dist_to_base']:.4f}m / {ARM_MAX_REACH:.3f}m")
        self.logger.info(f"\n-- 3. IK Statistics --")
        self.logger.info(f"  Valid: {valid}/{total + WARMUP_FRAMES}, Failed: {len(eval_pre['ik_errors'])}, Out of reach: {eval_pre['out_of_reach']}")
        fk_errors = eval_render["fk_errors"]
        if fk_errors:
            self.logger.info(f"\n-- 4. FK Verification --")
            self.logger.info(f"  Avg error: {np.mean(fk_errors)*100:.2f}cm, Max: {np.max(fk_errors)*100:.2f}cm, <2cm: {sum(1 for e in fk_errors if e<0.02)}/{len(fk_errors)}")
        comfort = eval_render["comfort_scores"]
        if comfort:
            self.logger.info(f"\n-- 5. Joint Comfort --")
            self.logger.info(f"  Avg: {np.mean(comfort):.4f}, Min: {np.min(comfort):.4f}")
        if smooth_metrics:
            self.logger.info(f"\n-- 6. Smoothness --")
            self.logger.info(f"  Vel: {smooth_metrics['smooth_max_velocity']:.2f} rad/s, Acc: {smooth_metrics['smooth_max_acceleration']:.2f} rad/s^2")
        self.logger.info("\n" + "=" * 80)


def _setup_logger(output_video):
    logger = logging.getLogger("R1TrackingSim")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(Path.cwd() / f"{Path(output_video).stem}.log", mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def main():
    parser = argparse.ArgumentParser(description="DexYCB -> Dex Retargeting -> RelaxedIK -> PyBullet Video (No SAPIEN)")
    parser.add_argument("--dexycb-dir", type=str, required=True)
    parser.add_argument("--data-id", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--output-video", type=str, default="r1_tracking_sim.mp4")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--view", type=str, default="behind", choices=["behind", "front", "topdown"])
    args = parser.parse_args()

    logger = _setup_logger(args.output_video)
    pipeline = R1HandTrackingPipeline(
        dexycb_dir=args.dexycb_dir, data_id=args.data_id,
        output_video=args.output_video, fps=args.fps, view=args.view, logger=logger,
    )
    pipeline.run(start_frame=args.start_frame, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
