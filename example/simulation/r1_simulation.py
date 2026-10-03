#!/usr/bin/env python3
"""
================================================================================
  R1 Robot Simulation Environment - PyBullet + RelaxedIK
================================================================================

Features:
  1. Load R1 robot URDF models (r1_gripper / r1_full / r1_galaxea), switchable
  2. RelaxedIK integration for inverse kinematics solving
  3. Visual demo: robot reaches target points with PyBullet GUI display
  4. IK reachability test and verification
  5. Line following and custom trajectory following
  6. Workspace mapping: map arbitrary 3D trajectory to robot reachable space
  7. Trajectory smoothing: Butterworth LPF + velocity/acceleration/jerk clamping
  8. Warmup transition: smoothstep interpolation to avoid initial jump
  9. No GPU required - PyBullet uses OpenGL software rendering

Usage:
  conda activate dex
  python r1_simulation.py --model r1_galaxea           # Visual demo
  python r1_simulation.py --model r1_galaxea --test-ik # IK test
  python r1_simulation.py --line-follow                 # Line following
  python r1_simulation.py --trajectory                  # Custom trajectory (circle/sine)
  python r1_simulation.py --arm left                    # Left arm
================================================================================
"""

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

R1_GRIPPER_URDF = str(PROJECT_ROOT / "assets" / "robots" / "hands" / "r1_gripper" / "r1_gripper_glb.urdf")
R1_FULL_URDF = str(PROJECT_ROOT / "assets" / "robots" / "r1_full" / "r1_lite_robot_glb.urdf")
R1_GALAXEA_URDF = str(GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "robot_glb.urdf")

R1_GALAXEA_LEFT_SETTINGS = str(GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml")
R1_GALAXEA_RIGHT_SETTINGS = str(GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml")
R1_LITE_LEFT_SETTINGS = str(GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1_lite" / "configs" / "settings_left.yaml")
R1_LITE_RIGHT_SETTINGS = str(GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1_lite" / "configs" / "settings_right.yaml")

MODEL_PATHS = {
    "r1_gripper": R1_GRIPPER_URDF,
    "r1_full": R1_FULL_URDF,
    "r1_galaxea": R1_GALAXEA_URDF,
}

MODEL_IK_SETTINGS = {
    "r1_gripper": (R1_LITE_LEFT_SETTINGS, R1_LITE_RIGHT_SETTINGS),
    "r1_full": (R1_LITE_LEFT_SETTINGS, R1_LITE_RIGHT_SETTINGS),
    "r1_galaxea": (R1_GALAXEA_LEFT_SETTINGS, R1_GALAXEA_RIGHT_SETTINGS),
}

RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985]

ARM_MAX_REACH = 0.713
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])
SAFETY_DISTANCE = 0.075
WARMUP_FRAMES = 30

GALAXEA_TARGETS_RIGHT = [
    {"pos": [0.06, -0.17, -0.47], "quat_wxyz": [1, 0, 0, 0], "label": "starting-config"},
    {"pos": [0.26, -0.13, -0.36], "quat_wxyz": [1, 0, 0, 0], "label": "forward-up"},
    {"pos": [-0.11, -0.17, -0.27], "quat_wxyz": [1, 0, 0, 0], "label": "backward"},
    {"pos": [0.31, -0.02, -0.51], "quat_wxyz": [1, 0, 0, 0], "label": "extended"},
    {"pos": [0.21, -0.31, -0.12], "quat_wxyz": [1, 0, 0, 0], "label": "side-down"},
]

GALAXEA_TARGETS_LEFT = [
    {"pos": [0.06, 0.17, -0.47], "quat_wxyz": [1, 0, 0, 0], "label": "starting-config"},
    {"pos": [0.26, 0.13, -0.36], "quat_wxyz": [1, 0, 0, 0], "label": "forward-up"},
    {"pos": [-0.11, 0.17, -0.27], "quat_wxyz": [1, 0, 0, 0], "label": "backward"},
    {"pos": [0.31, 0.02, -0.51], "quat_wxyz": [1, 0, 0, 0], "label": "extended"},
    {"pos": [0.21, 0.31, -0.12], "quat_wxyz": [1, 0, 0, 0], "label": "side-down"},
]

LITE_TARGETS_RIGHT = [
    {"pos": [0.05, -0.45, 0.25], "quat_wxyz": [1, 0, 0, 0], "label": "starting-config"},
    {"pos": [0.26, -0.36, 0.21], "quat_wxyz": [1, 0, 0, 0], "label": "forward-up"},
    {"pos": [-0.11, -0.26, 0.21], "quat_wxyz": [1, 0, 0, 0], "label": "backward"},
    {"pos": [0.18, -0.10, 0.32], "quat_wxyz": [1, 0, 0, 0], "label": "extended"},
    {"pos": [0.30, -0.48, 0.14], "quat_wxyz": [1, 0, 0, 0], "label": "side-down"},
]

LITE_TARGETS_LEFT = [
    {"pos": [0.05, 0.45, 0.25], "quat_wxyz": [1, 0, 0, 0], "label": "starting-config"},
    {"pos": [0.26, 0.36, 0.21], "quat_wxyz": [1, 0, 0, 0], "label": "forward-up"},
    {"pos": [-0.11, 0.26, 0.21], "quat_wxyz": [1, 0, 0, 0], "label": "backward"},
    {"pos": [0.18, 0.10, 0.32], "quat_wxyz": [1, 0, 0, 0], "label": "extended"},
    {"pos": [0.30, 0.48, 0.14], "quat_wxyz": [1, 0, 0, 0], "label": "side-down"},
]

MODEL_TEST_TARGETS = {
    "r1_gripper": (LITE_TARGETS_RIGHT, LITE_TARGETS_LEFT),
    "r1_full": (LITE_TARGETS_RIGHT, LITE_TARGETS_LEFT),
    "r1_galaxea": (GALAXEA_TARGETS_RIGHT, GALAXEA_TARGETS_LEFT),
}


class LPFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def next(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value.copy()


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

    def smooth(self, trajectory):
        n_frames = len(trajectory)
        if n_frames < 4:
            return trajectory

        traj = np.array(trajectory, dtype=np.float64)
        traj = self._bidirectional_lowpass(traj)
        traj = self._iterative_clamp(traj)
        return traj.tolist()

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
            clamped = np.clip(delta, -max_delta, max_delta)
            trajectory[i] = trajectory[i - 1] + clamped
        return trajectory

    def _clamp_acceleration(self, trajectory):
        max_delta_v = self.max_acceleration * self.dt
        for i in range(2, len(trajectory)):
            v_prev = trajectory[i - 1] - trajectory[i - 2]
            v_curr = trajectory[i] - trajectory[i - 1]
            delta_v = v_curr - v_prev
            clamped_dv = np.clip(delta_v, -max_delta_v, max_delta_v)
            v_curr_clamped = v_prev + clamped_dv
            trajectory[i] = trajectory[i - 1] + v_curr_clamped
        return trajectory

    def _clamp_jerk(self, trajectory):
        max_delta_a = self.max_jerk * self.dt
        for i in range(3, len(trajectory)):
            v_im2 = trajectory[i - 2] - trajectory[i - 3]
            v_im1 = trajectory[i - 1] - trajectory[i - 2]
            v_i = trajectory[i] - trajectory[i - 1]
            a_prev = v_im1 - v_im2
            a_curr = v_i - v_im1
            delta_a = a_curr - a_prev
            clamped_da = np.clip(delta_a, -max_delta_a, max_delta_a)
            a_curr_clamped = a_prev + clamped_da
            v_i_clamped = v_im1 + a_curr_clamped
            trajectory[i] = trajectory[i - 1] + v_i_clamped
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


def quat_xyzw_to_wxyz(q_xyzw):
    return [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def transform_pose_inverse(base_pos, base_quat_wxyz, target_pos, target_quat_wxyz):
    R = quat_to_rot(base_quat_wxyz)
    rel_pos = R.T @ (target_pos - base_pos)
    rel_quat = quat_multiply(quat_conjugate(base_quat_wxyz), target_quat_wxyz)
    return rel_pos, rel_quat


def transform_pose_forward(base_pos, base_quat_wxyz, rel_pos, rel_quat_wxyz):
    R = quat_to_rot(base_quat_wxyz)
    world_pos = R @ rel_pos + base_pos
    world_quat = quat_multiply(base_quat_wxyz, rel_quat_wxyz)
    return world_pos, world_quat


def _make_pybullet_compatible_urdf(urdf_path):
    import trimesh

    urdf_path = Path(urdf_path)
    with open(urdf_path, 'r') as f:
        content = f.read()

    tmp_dir = tempfile.mkdtemp(prefix="r1_sim_")
    mesh_dir = urdf_path.parent / "meshes"
    if mesh_dir.exists():
        os.symlink(str(mesh_dir), os.path.join(tmp_dir, "meshes"))

    obj_dir = os.path.join(tmp_dir, "meshes_obj")
    os.makedirs(obj_dir, exist_ok=True)

    converted = set()

    def replace_glb_with_obj(match):
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
                        scene = trimesh.load(str(glb_full))
                        mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
                        mesh.export(obj_full)
                    except Exception as e:
                        print(f"  [Warning] Cannot convert {filename}: {e}")
                        return match.group(0)

            return f'filename="meshes_obj/{mesh_name}"'
        return match.group(0)

    def replace_collision_stl(match):
        filename = match.group(1)
        clean = re.sub(r'^\.\/', '', filename)
        clean = clean.replace('meshes/', '', 1) if clean.startswith('meshes/') else clean
        if clean.endswith('_collision.STL'):
            base_stl = clean.replace('_collision.STL', '.STL')
            stl_full = mesh_dir / base_stl
            if stl_full.exists():
                return f'filename="meshes/{base_stl}"'
        return match.group(0)

    content = re.sub(
        r'filename="([^"]*\.(?:usd\.sapien\.)?glb)"',
        replace_glb_with_obj,
        content
    )
    content = re.sub(
        r'filename="([^"]*_collision\.STL)"',
        replace_collision_stl,
        content
    )

    tmp_urdf = os.path.join(tmp_dir, urdf_path.name)
    with open(tmp_urdf, 'w') as f:
        f.write(content)

    return tmp_urdf, tmp_dir


def compute_workspace_mapping(trajectory_world, base_pos, base_quat_wxyz):
    """
    Map a 3D trajectory from world coordinates to the robot's reachable workspace.

    This follows the same approach as r1_hand_tracking_video.py:
    1. Compute trajectory centroid
    2. Compute comfortable target in world frame
    3. Compute mapping offset = comfortable_target_world - centroid
    4. Compute safety offset along approach direction
    5. Each point: mapped = original + mapping_offset + safety_offset
    """
    base_R = quat_to_rot(base_quat_wxyz)
    centroid = np.mean(trajectory_world, axis=0)
    comfort_target_world = base_R @ COMFORT_TARGET_IN_BASE + base_pos
    mapping_offset = comfort_target_world - centroid
    approach_dir = base_pos - comfort_target_world
    approach_dir = approach_dir / np.linalg.norm(approach_dir)
    safety_offset = approach_dir * SAFETY_DISTANCE

    mapped = np.array([pt + mapping_offset + safety_offset for pt in trajectory_world])
    return mapped, mapping_offset, safety_offset


def generate_circle_trajectory(center, radius, num_points, plane="xy"):
    trajectory = []
    for i in range(num_points):
        t = 2 * np.pi * i / num_points
        if plane == "xy":
            pt = center + np.array([radius * np.cos(t), radius * np.sin(t), 0])
        elif plane == "xz":
            pt = center + np.array([radius * np.cos(t), 0, radius * np.sin(t)])
        elif plane == "yz":
            pt = center + np.array([0, radius * np.cos(t), radius * np.sin(t)])
        trajectory.append(pt)
    return trajectory


def generate_sine_trajectory(start, end, amplitude, num_points, axis="z"):
    trajectory = []
    for i in range(num_points):
        t = i / (num_points - 1)
        pt = start + t * (end - start)
        phase = np.sin(2 * np.pi * t * 2)
        if axis == "z":
            pt[2] += amplitude * phase
        elif axis == "y":
            pt[1] += amplitude * phase
        elif axis == "x":
            pt[0] += amplitude * phase
        trajectory.append(pt)
    return trajectory


class R1SimulationEnv:
    def __init__(self, model_name="r1_galaxea", headless=False):
        self.model_name = model_name
        self.headless = headless
        self.physics_client = None
        self.robot_id = None
        self.ik_solver = None
        self.tmp_dir = None

        self.right_arm_indices = []
        self.left_arm_indices = []
        self.torso_indices = []
        self.right_gripper_indices = []
        self.left_gripper_indices = []
        self.right_ee_link_index = -1
        self.left_ee_link_index = -1
        self.right_base_link_index = -1
        self.left_base_link_index = -1

        self._init_physics(headless)
        self._load_robot(model_name)
        self._init_ik_solver()
        self._set_initial_qpos()
        self._disable_default_motors()

    def _init_physics(self, headless):
        if headless:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / 240.0)
        p.loadURDF("plane.urdf")
        print("[Sim] PyBullet initialized" + (" (headless)" if headless else ""))

    def _load_robot(self, model_name):
        urdf_path = MODEL_PATHS.get(model_name)
        if urdf_path is None or not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        pb_urdf, tmp_dir = _make_pybullet_compatible_urdf(urdf_path)
        self.tmp_dir = tmp_dir

        self.robot_id = p.loadURDF(
            pb_urdf,
            basePosition=[0, 0, 0],
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
        self.left_base_link_index = -1

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode()
            link_name = info[12].decode()
            joint_type = info[2]

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
            elif link_name == "left_gripper_link":
                self.left_ee_link_index = i
            elif link_name == "right_arm_base_link":
                self.right_base_link_index = i
            elif link_name == "left_arm_base_link":
                self.left_base_link_index = i

        self.right_arm_indices.sort(key=lambda x: x[0])
        self.left_arm_indices.sort(key=lambda x: x[0])
        self.right_arm_indices = [idx for _, idx in self.right_arm_indices]
        self.left_arm_indices = [idx for _, idx in self.left_arm_indices]

        print(f"[Sim] Model '{model_name}' loaded ({num_joints} joints)")

    def _disable_default_motors(self):
        for idx in self.right_arm_indices + self.left_arm_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL, force=0)
        for idx in self.right_gripper_indices + self.left_gripper_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL, force=0)
        for idx in self.torso_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=1000)

    def _init_ik_solver(self):
        left_settings, right_settings = MODEL_IK_SETTINGS.get(
            self.model_name, (R1_GALAXEA_LEFT_SETTINGS, R1_GALAXEA_RIGHT_SETTINGS),
        )
        self.ik_solver = RelaxedIKSolver(
            left_setting_file_path=left_settings,
            right_setting_file_path=right_settings,
            tolerances=[0.001] * 6,
        )
        print(f"[Sim] RelaxedIK solver initialized (config: {self.model_name})")

    def _set_initial_qpos(self):
        for i, idx in enumerate(self.right_arm_indices):
            if i < len(RIGHT_ARM_STARTING):
                p.resetJointState(self.robot_id, idx, RIGHT_ARM_STARTING[i])
        for i, idx in enumerate(self.left_arm_indices):
            if i < len(LEFT_ARM_STARTING):
                p.resetJointState(self.robot_id, idx, LEFT_ARM_STARTING[i])
        for idx in self.torso_indices:
            p.resetJointState(self.robot_id, idx, 0.0)
        for idx in self.right_gripper_indices:
            p.resetJointState(self.robot_id, idx, 0.0)
        for idx in self.left_gripper_indices:
            p.resetJointState(self.robot_id, idx, 0.0)
        self._hold_all_joints()

    def _hold_all_joints(self):
        for idx in self.right_arm_indices:
            pos = p.getJointState(self.robot_id, idx)[0]
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=pos, force=500)
        for idx in self.left_arm_indices:
            pos = p.getJointState(self.robot_id, idx)[0]
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=pos, force=500)
        for idx in self.torso_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=1000)
        for idx in self.right_gripper_indices + self.left_gripper_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=100)

    def _hold_torso_and_other_arm(self, active_arm="right"):
        for idx in self.torso_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=1000)
        other = self.left_arm_indices if active_arm == "right" else self.right_arm_indices
        for idx in other:
            pos = p.getJointState(self.robot_id, idx)[0]
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=pos, force=500)
        for idx in self.right_gripper_indices + self.left_gripper_indices:
            p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=0.0, force=100)

    def reset_robot(self):
        self._set_initial_qpos()
        self._disable_default_motors()
        self.ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)
        self.ik_solver.relaxed_ik_left.reset(LEFT_ARM_STARTING)

    def switch_model(self, model_name):
        if model_name == self.model_name:
            return
        if model_name not in MODEL_PATHS:
            print(f"[Sim] Unknown model: {model_name}")
            return
        old_name = self.model_name
        p.removeBody(self.robot_id)
        self.model_name = model_name
        self._load_robot(model_name)
        self._set_initial_qpos()
        self._disable_default_motors()
        self.ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)
        self.ik_solver.relaxed_ik_left.reset(LEFT_ARM_STARTING)
        print(f"[Sim] Model switched: '{old_name}' -> '{model_name}'")

    def _get_arm_indices(self, arm):
        return self.right_arm_indices if arm == "right" else self.left_arm_indices

    def _get_ee_link_index(self, arm):
        return self.right_ee_link_index if arm == "right" else self.left_ee_link_index

    def _get_base_link_index(self, arm):
        return self.right_base_link_index if arm == "right" else self.left_base_link_index

    def get_link_pose_world(self, link_index):
        if link_index < 0:
            return None, None
        state = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        pos = np.array(state[4])
        quat_xyzw = np.array(state[5])
        quat_wxyz = quat_xyzw_to_wxyz(quat_xyzw)
        return pos, quat_wxyz

    def get_ee_pose_world(self, arm="right"):
        return self.get_link_pose_world(self._get_ee_link_index(arm))

    def get_base_pose_world(self, arm="right"):
        return self.get_link_pose_world(self._get_base_link_index(arm))

    def get_ee_pose_base(self, arm="right"):
        ee_pos, ee_quat = self.get_ee_pose_world(arm)
        if ee_pos is None:
            return None, None
        base_pos, base_quat = self.get_base_pose_world(arm)
        if base_pos is None:
            return None, None
        return transform_pose_inverse(base_pos, base_quat, ee_pos, ee_quat)

    def world_to_base(self, target_pos_world, target_quat_wxyz, arm):
        base_pos, base_quat = self.get_base_pose_world(arm)
        return transform_pose_inverse(base_pos, base_quat, target_pos_world, target_quat_wxyz)

    def base_to_world(self, target_pos_base, target_quat_wxyz, arm):
        base_pos, base_quat = self.get_base_pose_world(arm)
        return transform_pose_forward(base_pos, base_quat, target_pos_base, target_quat_wxyz)

    def get_arm_joint_positions(self, arm="right"):
        indices = self._get_arm_indices(arm)
        return [p.getJointState(self.robot_id, idx)[0] for idx in indices]

    def set_arm_joint_positions(self, joint_angles, arm="right"):
        indices = self._get_arm_indices(arm)
        for j, idx in enumerate(indices):
            if j < len(joint_angles):
                p.resetJointState(self.robot_id, idx, float(joint_angles[j]))

    def solve_ik(self, target_pos_base, target_quat_wxyz, arm="right",
                 max_iterations=100, tolerance_cm=2.0):
        if arm == "right":
            solve_fn = self.ik_solver.solve_position_right
            reset_fn = self.ik_solver.relaxed_ik_right.reset
        else:
            solve_fn = self.ik_solver.solve_position_left
            reset_fn = self.ik_solver.relaxed_ik_left.reset

        target_pos_base = np.array(target_pos_base, dtype=np.float64)
        target_quat_wxyz = np.array(target_quat_wxyz, dtype=np.float64)

        current_joints = self.get_arm_joint_positions(arm)
        reset_fn(current_joints)

        best_joints = None
        best_error = float('inf')

        for iteration in range(max_iterations):
            try:
                joint_angles = np.array(
                    solve_fn(target_pos_base.tolist(), target_quat_wxyz.tolist())
                )
            except Exception:
                continue

            arm_indices = self._get_arm_indices(arm)
            for j, idx in enumerate(arm_indices):
                if j < len(joint_angles):
                    p.resetJointState(self.robot_id, idx, float(joint_angles[j]))
                    p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL,
                                           targetPosition=float(joint_angles[j]), force=500)

            ee_pos_base, _ = self.get_ee_pose_base(arm)
            if ee_pos_base is not None:
                error_cm = np.linalg.norm(ee_pos_base - target_pos_base) * 100
                if error_cm < best_error:
                    best_error = error_cm
                    best_joints = joint_angles.copy()
                if error_cm < tolerance_cm:
                    return best_joints

        return best_joints

    def solve_ik_animated(self, target_pos_base, target_quat_wxyz, arm="right",
                          max_iterations=100, tolerance_cm=2.0):
        if arm == "right":
            solve_fn = self.ik_solver.solve_position_right
            reset_fn = self.ik_solver.relaxed_ik_right.reset
        else:
            solve_fn = self.ik_solver.solve_position_left
            reset_fn = self.ik_solver.relaxed_ik_left.reset

        target_pos_base = np.array(target_pos_base, dtype=np.float64)
        target_quat_wxyz = np.array(target_quat_wxyz, dtype=np.float64)

        current_joints = self.get_arm_joint_positions(arm)
        reset_fn(current_joints)

        best_joints = None
        best_error = float('inf')

        for iteration in range(max_iterations):
            try:
                joint_angles = np.array(
                    solve_fn(target_pos_base.tolist(), target_quat_wxyz.tolist())
                )
            except Exception:
                continue

            arm_indices = self._get_arm_indices(arm)
            for j, idx in enumerate(arm_indices):
                if j < len(joint_angles):
                    p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL,
                                           targetPosition=float(joint_angles[j]),
                                           force=500, maxVelocity=1.5)

            for _ in range(10):
                p.stepSimulation()
                self._hold_torso_and_other_arm(arm)
                if not self.headless:
                    time.sleep(1.0 / 480.0)

            ee_pos_base, _ = self.get_ee_pose_base(arm)
            if ee_pos_base is not None:
                error_cm = np.linalg.norm(ee_pos_base - target_pos_base) * 100
                if error_cm < best_error:
                    best_error = error_cm
                    best_joints = joint_angles.copy()
                if error_cm < tolerance_cm:
                    return best_joints

        return best_joints

    def add_target_marker(self, target_pos_world, color=[0, 1, 0, 1], size=0.03):
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=size, rgbaColor=color)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                                 basePosition=target_pos_world.tolist())

    def remove_marker(self, marker_id):
        if marker_id is not None:
            p.removeBody(marker_id)

    def disconnect(self):
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir, ignore_errors=True)


def run_trajectory_follow(env, trajectory_world, arm="right", fps=30,
                          use_workspace_mapping=True, smooth=True):
    """
    Follow a 3D trajectory in world coordinates.

    Pipeline (same as r1_hand_tracking_video.py):
    1. Optional workspace mapping: map trajectory to robot reachable space
    2. Transform each point to base_link frame
    3. Solve IK for each point (with LPFilter for smoothness)
    4. Optional trajectory smoothing (Butterworth + velocity/acceleration/jerk clamping)
    5. Warmup transition (smoothstep interpolation)
    6. Execute trajectory with visual display
    """
    trajectory_world = np.array(trajectory_world, dtype=np.float64)
    num_points = len(trajectory_world)
    target_quat = np.array([1, 0, 0, 0], dtype=np.float64)

    print(f"\n{'=' * 60}")
    print(f"  Trajectory Following - {arm} arm (model: {env.model_name})")
    print(f"  Points: {num_points}")
    print(f"  Workspace mapping: {use_workspace_mapping}")
    print(f"  Trajectory smoothing: {smooth}")
    print(f"{'=' * 60}")

    base_pos, base_quat = env.get_base_pose_world(arm)
    print(f"  Arm base world pos: [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}]")

    if use_workspace_mapping:
        mapped_traj, mapping_offset, safety_offset = compute_workspace_mapping(
            trajectory_world, base_pos, base_quat
        )
        centroid = np.mean(trajectory_world, axis=0)
        print(f"  Mapping offset: [{mapping_offset[0]:.4f}, {mapping_offset[1]:.4f}, {mapping_offset[2]:.4f}]")
        print(f"  Safety offset:  [{safety_offset[0]:.4f}, {safety_offset[1]:.4f}, {safety_offset[2]:.4f}]")
    else:
        mapped_traj = trajectory_world.copy()

    if not env.headless:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
                                     cameraTargetPosition=[0, 0, 0.8])

    env.reset_robot()
    for _ in range(60):
        p.stepSimulation()
        env._hold_all_joints()
        if not env.headless:
            time.sleep(1.0 / 240.0)

    if not env.headless:
        for pt in mapped_traj[::max(1, num_points // 20)]:
            env.add_target_marker(pt, color=[1, 1, 0, 0.5], size=0.008)

    ee_pos_filter = LPFilter(alpha=0.6)
    joint_filter = LPFilter(alpha=0.5)

    joint_sequence = []
    ik_targets_base = []

    for i, target_world in enumerate(mapped_traj):
        target_world_filtered = ee_pos_filter.next(target_world)
        target_base, _ = env.world_to_base(target_world_filtered, target_quat, arm)
        ik_targets_base.append(target_base.copy())

        if arm == "right":
            solve_fn = env.ik_solver.solve_position_right
            env.ik_solver.relaxed_ik_right.reset(env.get_arm_joint_positions(arm))
        else:
            solve_fn = env.ik_solver.solve_position_left
            env.ik_solver.relaxed_ik_left.reset(env.get_arm_joint_positions(arm))

        try:
            joint_angles = np.array(solve_fn(target_base.tolist(), target_quat.tolist()))
        except Exception:
            joint_sequence.append(None)
            continue

        joint_angles = joint_filter.next(joint_angles)
        joint_sequence.append(joint_angles.copy())

    valid_joints = [j for j in joint_sequence if j is not None]
    if not valid_joints:
        print("  [Error] No valid IK solutions found!")
        env.disconnect()
        return

    if smooth and len(valid_joints) > 3:
        smoother = TrajectorySmoother(fps=fps, max_velocity=1.5, max_acceleration=4.0,
                                      max_jerk=20.0, lp_alpha=0.25)
        smoothed_joints = smoother.smooth(valid_joints)
        smooth_idx = 0
        for i in range(len(joint_sequence)):
            if joint_sequence[i] is not None:
                joint_sequence[i] = np.array(smoothed_joints[smooth_idx])
                smooth_idx += 1

    first_valid = None
    for j in joint_sequence:
        if j is not None:
            first_valid = j
            break

    if first_valid is not None:
        current_joints = np.array(env.get_arm_joint_positions(arm))
        warmup_sequence = []
        for w in range(WARMUP_FRAMES):
            t = (w + 1) / WARMUP_FRAMES
            t_smooth = t * t * (3 - 2 * t)
            interp = current_joints * (1 - t_smooth) + first_valid * t_smooth
            warmup_sequence.append(interp.copy())
    else:
        warmup_sequence = []

    full_sequence = warmup_sequence + [j for j in joint_sequence if j is not None]

    errors = []
    trail_markers = []

    for seq_idx, joint_angles in enumerate(full_sequence):
        arm_indices = env._get_arm_indices(arm)
        for j, idx in enumerate(arm_indices):
            if j < len(joint_angles):
                p.resetJointState(env.robot_id, idx, float(joint_angles[j]))
                p.setJointMotorControl2(env.robot_id, idx, p.POSITION_CONTROL,
                                       targetPosition=float(joint_angles[j]), force=500)
        env._hold_torso_and_other_arm(arm)

        data_idx = seq_idx - len(warmup_sequence)
        if data_idx >= 0 and data_idx < len(ik_targets_base):
            ee_pos_base, _ = env.get_ee_pose_base(arm)
            if ee_pos_base is not None:
                error_cm = np.linalg.norm(ee_pos_base - ik_targets_base[data_idx]) * 100
                errors.append(error_cm)

                if not env.headless:
                    ee_world, _ = env.get_ee_pose_world(arm)
                    if ee_world is not None:
                        m = env.add_target_marker(ee_world, color=[0, 1, 0, 0.8], size=0.01)
                        trail_markers.append(m)

        for _ in range(5):
            p.stepSimulation()
            env._hold_all_joints()
            if not env.headless:
                time.sleep(1.0 / 240.0)

    if errors:
        print(f"\n  Trajectory Following Result:")
        print(f"    Avg error: {np.mean(errors):.2f} cm")
        print(f"    Max error: {np.max(errors):.2f} cm")
        print(f"    Min error: {np.min(errors):.2f} cm")
        print(f"    <2cm: {sum(1 for e in errors if e < 2)}/{len(errors)}")

    if not env.headless:
        print(f"\n  Demo complete! Close the PyBullet window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                env._hold_all_joints()
                time.sleep(1.0 / 60.0)
        except:
            pass

    env.disconnect()


def run_visual_demo(env, arm="right"):
    print(f"\n{'=' * 60}")
    print(f"  R1 Visual Demo - {arm} arm (model: {env.model_name})")
    print(f"{'=' * 60}")

    targets_right, targets_left = MODEL_TEST_TARGETS.get(
        env.model_name, (GALAXEA_TARGETS_RIGHT, GALAXEA_TARGETS_LEFT)
    )
    targets = targets_right if arm == "right" else targets_left

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30,
                                 cameraTargetPosition=[0, 0, 0.8])

    for i, target in enumerate(targets):
        target_pos_base = np.array(target["pos"])
        target_quat = np.array(target["quat_wxyz"])
        target_pos_world, _ = env.base_to_world(target_pos_base, target_quat, arm)

        print(f"\n  [{i+1}/{len(targets)}] {target['label']}")

        marker_id = env.add_target_marker(target_pos_world, color=[0, 1, 0, 1])

        env.reset_robot()
        for _ in range(120):
            p.stepSimulation()
            env._hold_all_joints()
            time.sleep(1.0 / 240.0)

        joint_angles = env.solve_ik_animated(target_pos_base, target_quat, arm)

        ee_pos_base, _ = env.get_ee_pose_base(arm)
        if ee_pos_base is not None:
            error_cm = np.linalg.norm(ee_pos_base - target_pos_base) * 100
            status = "PASS" if error_cm < 2.0 else "ACCEPTABLE" if error_cm < 5.0 else "LARGE_ERROR"
            print(f"    Error: {error_cm:.2f} cm  [{status}]")

        for _ in range(300):
            p.stepSimulation()
            env._hold_all_joints()
            time.sleep(1.0 / 240.0)

        env.remove_marker(marker_id)

    print(f"\n  Demo complete! Close the PyBullet window to exit.")
    try:
        while p.isConnected():
            p.stepSimulation()
            env._hold_all_joints()
            time.sleep(1.0 / 60.0)
    except:
        pass
    env.disconnect()


def run_ik_test(env, arm="right"):
    print(f"\n{'=' * 60}")
    print(f"  IK Reachability Test - {arm} arm (model: {env.model_name})")
    print(f"{'=' * 60}")

    targets_right, targets_left = MODEL_TEST_TARGETS.get(
        env.model_name, (GALAXEA_TARGETS_RIGHT, GALAXEA_TARGETS_LEFT)
    )
    targets = targets_right if arm == "right" else targets_left

    results = []
    for i, target in enumerate(targets):
        target_pos_base = np.array(target["pos"])
        target_quat = np.array(target["quat_wxyz"])

        print(f"\n  Test {i + 1}/{len(targets)}: {target['label']}")

        env.reset_robot()
        joint_angles = env.solve_ik(target_pos_base, target_quat, arm)

        if joint_angles is None:
            print(f"    Result: IK solve failed")
            results.append({"success": False, "error": float('inf')})
            continue

        ee_pos_base, _ = env.get_ee_pose_base(arm)
        if ee_pos_base is not None:
            error_base = np.linalg.norm(ee_pos_base - target_pos_base) * 100
            status = "PASS" if error_base < 2.0 else "ACCEPTABLE" if error_base < 5.0 else "LARGE_ERROR"
            print(f"    Error: {error_base:.2f} cm  [{status}]")
            results.append({"success": True, "error_cm": error_base, "status": status})
        else:
            results.append({"success": False, "error": float('inf')})

    success_count = sum(1 for r in results if r["success"])
    avg_error = np.mean([r["error_cm"] for r in results if r["success"]]) if success_count > 0 else float('inf')

    print(f"\n  Summary: {success_count}/{len(targets)} success, avg error: {avg_error:.2f} cm")
    env.disconnect()


def run_line_follow(env, start_pos, end_pos, arm="right", num_steps=50):
    start_pos = np.array(start_pos, dtype=np.float64)
    end_pos = np.array(end_pos, dtype=np.float64)
    trajectory = [start_pos + t * (end_pos - start_pos) for t in np.linspace(0, 1, num_steps + 1)]
    run_trajectory_follow(env, trajectory, arm=arm, use_workspace_mapping=False, smooth=True)


def main():
    parser = argparse.ArgumentParser(description="R1 Robot Simulation (PyBullet + RelaxedIK)")
    parser.add_argument("--model", type=str, default="r1_galaxea",
                        choices=["r1_gripper", "r1_full", "r1_galaxea"])
    parser.add_argument("--arm", type=str, default="right", choices=["right", "left"])
    parser.add_argument("--test-ik", action="store_true")
    parser.add_argument("--line-follow", action="store_true")
    parser.add_argument("--trajectory", type=str, default=None,
                        choices=["circle", "sine", "figure8"],
                        help="Predefined trajectory type")
    parser.add_argument("--start-pos", type=float, nargs=3, default=None)
    parser.add_argument("--end-pos", type=float, nargs=3, default=None)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-workspace-mapping", action="store_true",
                        help="Disable workspace mapping for trajectory")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable trajectory smoothing")
    parser.add_argument("--target-pos", type=float, nargs=3, default=None)
    parser.add_argument("--target-quat", type=float, nargs=4, default=None)
    parser.add_argument("--headless", action="store_true")

    args = parser.parse_args()
    headless = args.headless or args.test_ik
    env = R1SimulationEnv(model_name=args.model, headless=headless)

    try:
        if args.test_ik:
            run_ik_test(env, arm=args.arm)
        elif args.trajectory:
            base_pos, _ = env.get_base_pose_world(args.arm)
            if args.trajectory == "circle":
                center = base_pos + np.array([0.3, -0.15 if args.arm == "right" else 0.15, -0.2])
                traj = generate_circle_trajectory(center, radius=0.1, num_points=args.num_steps, plane="xz")
            elif args.trajectory == "sine":
                start = base_pos + np.array([0.1, -0.3 if args.arm == "right" else 0.3, -0.3])
                end = base_pos + np.array([0.4, -0.3 if args.arm == "right" else 0.3, -0.3])
                traj = generate_sine_trajectory(start, end, amplitude=0.1, num_points=args.num_steps, axis="z")
            elif args.trajectory == "figure8":
                center = base_pos + np.array([0.3, -0.15 if args.arm == "right" else 0.15, -0.25])
                traj = []
                for i in range(args.num_steps):
                    t = 2 * np.pi * i / args.num_steps
                    x = 0.15 * np.sin(t)
                    z = 0.08 * np.sin(2 * t)
                    traj.append(center + np.array([x, 0, z]))
            else:
                traj = []
            run_trajectory_follow(env, traj, arm=args.arm, fps=args.fps,
                                  use_workspace_mapping=not args.no_workspace_mapping,
                                  smooth=not args.no_smooth)
        elif args.line_follow:
            targets_right, _ = MODEL_TEST_TARGETS.get(
                env.model_name, (GALAXEA_TARGETS_RIGHT, GALAXEA_TARGETS_LEFT)
            )
            start_pos = args.start_pos if args.start_pos is not None else targets_right[0]["pos"]
            end_pos = args.end_pos if args.end_pos is not None else targets_right[1]["pos"]
            run_line_follow(env, start_pos, end_pos, arm=args.arm, num_steps=args.num_steps)
        elif args.target_pos is not None:
            target_quat = args.target_quat if args.target_quat is not None else [1, 0, 0, 0]
            target_pos_base, target_quat_base = env.world_to_base(
                np.array(args.target_pos), np.array(target_quat), args.arm)
            joint_angles = env.solve_ik(target_pos_base, target_quat_base, args.arm)
            if joint_angles is not None:
                ee_pos_base, _ = env.get_ee_pose_base(args.arm)
                if ee_pos_base is not None:
                    error_cm = np.linalg.norm(ee_pos_base - target_pos_base) * 100
                    print(f"  Error: {error_cm:.2f} cm")
            env.disconnect()
        else:
            run_visual_demo(env, arm=args.arm)
    except KeyboardInterrupt:
        print("\n[Sim] Interrupted")
    finally:
        if env.physics_client is not None and p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
