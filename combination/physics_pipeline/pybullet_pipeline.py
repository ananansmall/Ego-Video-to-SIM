#!/usr/bin/env python3
"""
PyBullet 物理仿真管线 — 对齐 02_render_scene.py 的 run_robot_tracking

设计原则:
  完全复用 02_render_scene.py 的数据流和坐标变换:
  - 相机: 跟随 HaWoR 相机轨迹 (R_c2w, t_c2w), 与 02 第一人称视角一致
  - 机器人基座: 基于手腕位置计算最优固定基座 (与 02 一致)
  - GLB 物体: 使用相同的变换链 (s_inv*R_inv@p + t_inv → RXWORLD_TO_SAPIEN@p)
  - 关节映射: 按 URDF 关节名匹配 (right_arm_joint1..6 + right_gripper_finger_joint1..2)

与 02 的唯一区别:
  - 02: SAPIEN 运动学渲染 (set_qpos, 无物理交互)
  - 本脚本: PyBullet 运动学控制 (resetJointState) + GLB 物体物理交互

控制策略:
  R1 URDF 的连杆惯性极小, PyBullet 的 POSITION_CONTROL 和 TORQUE_CONTROL 均无法稳定控制。
  采用运动学控制: 每个物理子步用 resetJointState 设置目标关节角,
  GLB 物体由物理引擎驱动 (重力、碰撞), 机械臂可推动/碰撞 GLB 物体。

调用方式:
  # 使用 02 保存的轨迹渲染视频 (推荐)
  python pybullet_pipeline.py --render-video \
      --hawor-dir /home/an/data/hawor/7 \
      --ras-dir /home/an/data/ras/my_7mp4_result \
      --transform-params ./output/7_my_7mp4_result/alignment/transform_params.npz \
      --trajectory ./output/7_my_7mp4_result/tracking/hand_object_robot_tracking.npy

  # GUI 模式
  python pybullet_pipeline.py --render-video --gui ...

  # 基础测试
  python pybullet_pipeline.py --test
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import trimesh
import cv2
from pytransform3d import rotations as pr

# ============ 路径常量 ============
SCRIPT_DIR = Path(__file__).parent.resolve()
COMBINATION_DIR = SCRIPT_DIR.parent.resolve()
PROJECT_ROOT = COMBINATION_DIR.parent.parent

# 将 02_render_scene.py 所在目录加入 sys.path, 复用其函数
sys.path.insert(0, str(COMBINATION_DIR))

R1_URDF = "/home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/configs/urdfs/r1_v2_1_0_floating_right.urdf"
R1_MESH_DIR = "/home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/meshes"

# 默认路径 (与 02/04 一致)
DEFAULT_HAWOR_DIR = "/home/an/data/hawor/7"
DEFAULT_RAS_DIR = "/home/an/data/ras/my_7mp4_result"
DEFAULT_TRANSFORM_PARAMS = str(COMBINATION_DIR / "output" / "7_my_7mp4_result" / "alignment" / "transform_params.npz")
# 02 保存的轨迹 (运动学渲染, 确定性)
DEFAULT_TRAJECTORY = str(COMBINATION_DIR / "output" / "7_my_7mp4_result" / "tracking" / "hand_object_robot_tracking.npy")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "output" / "pybullet_render.mp4")

# ============ 物理参数 ============
PHYSICS_TIMESTEP = 1 / 240.0
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))  # =8
GRAVITY = [0, 0, -9.81]

# PD 控制增益 (与 SAPIEN 04 的 stiffness/damping 一致)
# 用于 computed torque control: tau = kp*(q_target - q) - kd*q_dot + tau_gravity
PD_KP_ARM = 1000.0      # 机械臂位置增益 (= SAPIEN JOINT_STIFFNESS)
PD_KD_ARM = 200.0       # 机械臂速度增益 (= SAPIEN JOINT_DAMPING)
PD_KP_GRIPPER = 1000.0  # 夹爪位置增益 (= SAPIEN GRIPPER_STIFFNESS)
PD_KD_GRIPPER = 200.0   # 夹爪速度增益 (= SAPIEN GRIPPER_DAMPING)

# ============ 视频参数 ============
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = 30
HAWOR_FOCAL_DEFAULT = 600.0

# ============ 坐标变换 (与 02_render_scene.py 完全一致) ============
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# ============ 机器人参数 (与 02 一致) ============
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
COMFORTABLE_REACH = 0.35
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
BASE_TRACKING_RANGE = 0.04
ARM_MAX_REACH = 0.713

# ============ 单夹爪模式: 夹爪几何常数 (与 04/hand_track 一致) ============
_FINGER1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689, 0.013453, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])
_FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906
GRIPPER_INIT_OPEN = 0.04

# 夹爪 URDF 模板 (只有 gripper 部分, 无机械臂, 与 04/hand_track 一致)
_GRIPPER_ONLY_URDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<robot name="r1_gripper_{prefix}">
  <link name="{prefix}_gripper_base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>
  <joint name="{prefix}_gripper_base_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_base_link"/>
    <child link="{prefix}_gripper_link"/>
  </joint>
  <link name="{prefix}_gripper_link">
    <inertial>
      <origin xyz="-0.031107240301242 -1.38928815840433E-07 -1.43700425780935E-07" rpy="0 0 0"/>
      <mass value="0.604"/>
      <inertia ixx="0.000175880119550986" ixy="4.17894263577595E-10" ixz="-5.34925118595879E-10"
               iyy="9.86374067070897E-05" iyz="-8.18555544397352E-08" izz="0.000165120109045834"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/>
      </geometry>
      <material name="">
        <color rgba="0.823529411764706 0.823529411764706 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/>
      </geometry>
    </collision>
  </link>
  <joint name="{prefix}_gripper_finger_joint1" type="prismatic">
    <origin xyz="0.03689 -0.013453 -0.00012053" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link1"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.05" effort="100" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link1">
    <inertial>
      <origin xyz="-0.0195895587205407 0.0151136130965041 -0.00542255818128545" rpy="0 0 0"/>
      <mass value="0.027"/>
      <inertia ixx="2.40569063762433E-06" ixy="-3.99002073372071E-07" ixz="-5.12217975840564E-08"
               iyy="5.71082134562374E-06" iyz="6.19457183851545E-08" izz="6.4848556091919E-06"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/>
      </geometry>
      <material name="">
        <color rgba="0.823529411764706 0.823529411764706 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/>
      </geometry>
    </collision>
  </link>
  <joint name="{prefix}_gripper_finger_joint2" type="prismatic">
    <origin xyz="0.03689 0.013453 0.00012067" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.05" effort="100" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link2">
    <inertial>
      <origin xyz="-0.019589448977496 -0.0151137821219537 0.00542248304315596" rpy="0 0 0"/>
      <mass value="0.027"/>
      <inertia ixx="2.40568339234574E-06" ixy="3.98973340378568E-07" ixz="5.12055978237686E-08"
               iyy="5.71082803574443E-06" iyz="6.19476812784019E-08" izz="6.48485579679143E-06"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.STL"/>
      </geometry>
      <material name="">
        <color rgba="0.823529411764706 0.823529411764706 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.STL"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def _generate_gripper_only_urdf(prefix="right"):
    """生成只包含夹爪的 URDF 文件 (无机械臂, 与 04/hand_track 一致)"""
    xml = _GRIPPER_ONLY_URDF_TEMPLATE.format(prefix=prefix, mesh_dir=R1_MESH_DIR)
    temp_dir = tempfile.mkdtemp(prefix=f'r1_gripper_only_{prefix}-')
    temp_path = f'{temp_dir}/r1_gripper_only_{prefix}.urdf'
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def _compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2):
    """从 MANO 指尖向量解析计算夹爪 root 位姿和手指关节值 (与 04/hand_track 一致)"""
    v_finger = mano_finger2 - mano_finger1
    finger_dist = np.linalg.norm(v_finger)
    if finger_dist < 1e-6:
        y_axis = np.array([0, 1, 0], dtype=np.float64)
    else:
        y_axis = v_finger / finger_dist

    finger_mid = (mano_finger1 + mano_finger2) / 2
    v_wrist = finger_mid - mano_wrist
    wrist_dist = np.linalg.norm(v_wrist)
    if wrist_dist < 1e-6:
        x_axis = np.array([1, 0, 0], dtype=np.float64)
    else:
        x_axis = v_wrist / wrist_dist

    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        x_axis = np.array([1, 0, 0], dtype=np.float64)
        x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-6:
            x_axis = np.array([0, 0, 1], dtype=np.float64)
            x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
            x_norm = np.linalg.norm(x_axis)
    x_axis = x_axis / x_norm
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)

    root_R = np.column_stack([x_axis, y_axis, z_axis])

    required_open_sum = finger_dist - _FINGER_BASE_DIST
    joint1 = max(0.0, min(0.05, required_open_sum / 2))
    joint2 = max(0.0, min(0.05, required_open_sum / 2))

    finger1_offset = _FINGER1_ORIGIN + _FINGER1_AXIS * joint1
    root_pos = mano_finger1 - root_R @ finger1_offset

    return root_pos, root_R, joint1, joint2


def prepare_urdf(src_urdf_path):
    """准备 URDF: 替换 mesh 路径 + 将夹爪关节改为 prismatic

    与 02_render_scene.py 的 prepare_arm_urdf 一致:
    1. 将 package://r1_v2_1_0/meshes/ 替换为绝对路径
    2. 将 right_gripper_finger_joint1/2 从 fixed 改为 prismatic
    """
    import re
    xml = Path(src_urdf_path).read_text()
    xml = xml.replace('package://r1_v2_1_0/meshes/', str(R1_MESH_DIR) + '/')
    xml = re.sub(
        r'(<joint\s+name="right_gripper_finger_joint1"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    xml = re.sub(
        r'(<joint\s+name="right_gripper_finger_joint2"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    temp_dir = tempfile.mkdtemp(prefix='r1_pybullet-')
    temp_path = f'{temp_dir}/{Path(src_urdf_path).name}'
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def hawor_cam_to_sapien_pose(R_c2w, t_c2w):
    """将 HaWoR 相机位姿转换为 SAPIEN 坐标系下的相机位姿

    与 02_render_scene.py 的 hawor_cam_to_sapien_pose 完全一致:
    1. 位置/旋转从 HaWoR render world 变换到 SAPIEN 坐标系
    2. OpenGL 相机约定 (Z=后方, X=右, Y=上) → SAPIEN 相机约定 (X=前, Y=左, Z=上)

    返回 (cam_pos_sapien, cam_R_sapien), 其中 cam_R_sapien 是 3x3 旋转矩阵
    (列向量: 第0列=forward, 第1列=left, 第2列=up)。
    """
    cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w
    cam_R_gl = RXWORLD_TO_SAPIEN @ R_c2w  # OpenGL 约定: X=右, Y=上, Z=后

    # OpenGL → SAPIEN 相机约定
    forward = -cam_R_gl[:, 2]  # -Z(后) = 前
    left = -cam_R_gl[:, 0]     # -X(右) = 左
    up = cam_R_gl[:, 1]        #  Y(上) = 上

    cam_R_sapien = np.eye(3)
    cam_R_sapien[:, 0] = forward
    cam_R_sapien[:, 1] = left
    cam_R_sapien[:, 2] = up

    # 保证是合法旋转矩阵
    if np.linalg.det(cam_R_sapien) < 0:
        U, _, VH = np.linalg.svd(cam_R_sapien)
        cam_R_sapien = U @ VH
    return cam_pos_sapien, cam_R_sapien


def sapien_cam_to_pybullet_view(cam_pos, cam_R, width, height, fov_deg=60.0, near=0.01, far=100.0):
    """将 SAPIEN 相机位姿转换为 PyBullet view matrix + projection matrix

    SAPIEN 相机约定: X=forward, Y=left, Z=up
    PyBullet computeViewMatrix: eye, target, up

    Args:
        cam_pos: (3,) 相机位置 (SAPIEN 坐标系)
        cam_R: (3, 3) 相机旋转矩阵 (SAPIEN 约定: X=forward, Y=left, Z=up)
        width: 图像宽度
        height: 图像高度
        fov_deg: 垂直 FOV (度)
        near: 近裁剪面
        far: 远裁剪面

    Returns:
        tuple: (view_matrix, projection_matrix)
    """
    # SAPIEN 相机: X=forward, Y=left, Z=up
    forward = cam_R[:, 0]
    up = cam_R[:, 2]

    eye = np.array(cam_pos, dtype=np.float64)
    target = eye + forward

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target.tolist(),
        cameraUpVector=up.tolist(),
    )

    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov_deg,
        aspect=aspect,
        nearVal=near,
        farVal=far,
    )
    return view_matrix, projection_matrix


def compute_cam_fov(img_focal, cam_height, cam_width):
    """根据 HaWoR 焦距计算相机 FOV (与 02 一致)"""
    if img_focal is not None and img_focal > 0:
        focal_for_render = img_focal * cam_width / 1280.0
        fov = 2 * np.arctan(cam_height / 2.0 / focal_for_render)
    else:
        focal_for_render = HAWOR_FOCAL_DEFAULT * cam_width / 1280.0
        fov = 2 * np.arctan(cam_height / 2.0 / focal_for_render)
    return float(np.degrees(fov))


def compute_optimal_fixed_base(wrist_positions_sapien):
    """计算机器人基座的最优固定位置和朝向 (与 02 一致)

    策略:
    1. 计算所有有效帧手腕位置的质心
    2. 基座放在质心正上方 COMFORTABLE_REACH (0.35m) 处
    3. 朝向: 绕Z轴旋转180°
    """
    if len(wrist_positions_sapien) == 0:
        return np.array([0.0, 0.0, COMFORTABLE_REACH]), pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))

    wrist_arr = np.array(wrist_positions_sapien)
    centroid = wrist_arr.mean(axis=0)
    wrist_range = wrist_arr.max(axis=0) - wrist_arr.min(axis=0)

    arm_base_pos = centroid.copy()
    arm_base_pos[2] += COMFORTABLE_REACH

    if wrist_range[0] > 0.01:
        arm_base_pos[0] += wrist_range[0] * 0.1

    # 朝向: 绕Z轴旋转180°
    z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
    arm_base_q = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))

    return arm_base_pos, arm_base_q


def compute_tracking_base_pos(initial_base_pos, wrist_pos_sapien, arm_base_q):
    """计算跟踪模式下的基座位置 (与 02 一致)

    基座在初始位置基础上, 沿 XY 方向跟踪手腕 (±BASE_TRACKING_RANGE)
    """
    base_R = pr.matrix_from_quaternion(arm_base_q)
    wrist_in_base = base_R.T @ (wrist_pos_sapien - initial_base_pos)
    offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE
    clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)
    delta_world = base_R @ clamped_offset
    return initial_base_pos + delta_world


def _load_module_02():
    """加载 02_render_scene.py 模块 (文件名以数字开头, 不能直接 import)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "render_scene_02", str(COMBINATION_DIR / "02_render_scene.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames):
    """预计算所有帧的手腕位置 (SAPIEN 坐标系)

    复用 02_render_scene.py 的 compute_mano_joints 函数。
    """
    mod = _load_module_02()
    compute_mano_joints = mod.compute_mano_joints

    positions = []
    for i in range(num_frames):
        global_idx = start_frame + i
        if not hawor_data["pred_valid"][global_idx]:
            continue
        rot = hawor_data["pred_rot"][global_idx]
        trans = hawor_data["pred_trans"][global_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)):
            continue
        _, j = compute_mano_joints(mano_layer, rot,
                                   hawor_data["pred_hand_pose"][global_idx], trans)
        joints_sapien = (RXWORLD_TO_SAPIEN @ j.T).T
        positions.append(joints_sapien[0, :3].copy())
    return positions


class PyBulletPipeline:
    """PyBullet 物理仿真管线: 对齐 02_render_scene.py 的 run_robot_tracking

    与 02 的区别:
    - 02: SAPIEN 运动学渲染 (set_qpos)
    - 本类: PyBullet 运动学控制 (resetJointState) + GLB 物体物理交互
    """

    def __init__(self, gui=False, cam_width=VIDEO_WIDTH, cam_height=VIDEO_HEIGHT,
                 single_gripper=False):
        self.gui = gui
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_fov_deg = 2 * np.degrees(np.arctan(self.cam_height / 2.0 / HAWOR_FOCAL_DEFAULT))
        self.single_gripper = single_gripper

        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(PHYSICS_TIMESTEP)
        p.setGravity(*GRAVITY)

        # 加载地面 (z=0, 后面会根据 GLB 物体调整)
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = None  # 桌面 ID (load_glb_objects 时创建)

        if self.single_gripper:
            # 单夹爪模式: 只加载夹爪 URDF (无机械臂), 参考 hand_track/render_gripper_only.py
            gripper_urdf_path = _generate_gripper_only_urdf("right")
            self.robot_id = p.loadURDF(
                gripper_urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True,  # kinematic: 每帧用 resetBasePositionAndOrientation 设置位姿
                flags=p.URDF_USE_SELF_COLLISION
            )
            self._identify_joints()
            # 禁用默认 motor controller (运动学控制)
            for idx in self.all_active_joint_indices:
                p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL, force=0)
            # 初始夹爪开合
            init_qpos = np.zeros(len(self.all_active_joint_indices))
            for i, idx in enumerate(self.gripper_joint_indices):
                init_qpos[self.all_active_joint_indices.index(idx)] = GRIPPER_INIT_OPEN
            self._set_qpos(init_qpos)
            print(f"\n  单夹爪已加载 (无机械臂, PD控制+重力补偿)")
            print(f"  夹爪关节(2): {self.gripper_joint_names}")
            print(f"  PD增益: gripper kp={PD_KP_GRIPPER} kd={PD_KD_GRIPPER}")
        else:
            # 完整机械臂模式
            urdf_path = prepare_urdf(R1_URDF)
            self.robot_id = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION
            )
            self._identify_joints()
            # 禁用默认 motor controller (运动学控制不需要)
            for idx in self.all_active_joint_indices:
                p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL, force=0)
            # 设置初始关节位置
            init_qpos = np.zeros(len(self.all_active_joint_indices))
            for j, idx in enumerate(self.arm_joint_indices):
                if j < len(RIGHT_ARM_STARTING):
                    init_qpos[self.arm_joint_indices.index(idx)] = RIGHT_ARM_STARTING[j]
            # 夹爪初始位置
            init_qpos[self.arm_joint_indices.__len__()] = 0.04  # gripper1
            init_qpos[self.arm_joint_indices.__len__() + 1] = -0.04  # gripper2
            self._set_qpos(init_qpos)
            print(f"\n  机器人已加载 (固定基座, PD控制+重力补偿)")
            print(f"  臂关节({len(self.arm_joint_indices)}): {self.arm_joint_names}")
            print(f"  夹爪关节(2): {self.gripper_joint_names}")
            print(f"  PD增益: arm kp={PD_KP_ARM} kd={PD_KD_ARM}, gripper kp={PD_KP_GRIPPER} kd={PD_KD_GRIPPER}")

        # GLB 物体列表
        self.obj_ids = []
        self.obj_info = {}

        # 基座参数 (后续由 setup_from_hawor 设置)
        self.arm_base_pos = None
        self.arm_base_q = None

        print(f"  DOF={len(self.all_active_joint_indices)}, decimation={DECIMATION}")

    def _identify_joints(self):
        """按关节名识别臂关节和夹爪关节 (与 02 一致)"""
        self.all_active_joint_indices = []
        self.all_active_joint_names = []
        self.arm_joint_indices = []
        self.arm_joint_names = []
        self.gripper_joint_indices = []
        self.gripper_joint_names = []

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode()
            joint_type = info[2]

            # 只处理活动关节 (revolute 或 prismatic)
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.all_active_joint_indices.append(i)
                self.all_active_joint_names.append(joint_name)

                if "right_arm_joint" in joint_name:
                    self.arm_joint_indices.append(i)
                    self.arm_joint_names.append(joint_name)
                elif "right_gripper_finger_joint" in joint_name:
                    self.gripper_joint_indices.append(i)
                    self.gripper_joint_names.append(joint_name)

        # 确保顺序: arm_joint1..6, gripper1, gripper2
        # PyBullet 按 URDF 顺序加载, 通常已经是正确顺序
        print(f"  识别到 {len(self.arm_joint_indices)} 个臂关节, {len(self.gripper_joint_indices)} 个夹爪关节")

    def _set_qpos(self, qpos):
        """设置所有活动关节位置 (运动学: resetJointState)"""
        for i, idx in enumerate(self.all_active_joint_indices):
            target = qpos[i] if i < len(qpos) else 0.0
            p.resetJointState(self.robot_id, idx, target, targetVelocity=0.0)

    def _set_arm_qpos(self, arm_qpos):
        """仅设置臂关节位置"""
        for i, idx in enumerate(self.arm_joint_indices):
            target = arm_qpos[i] if i < len(arm_qpos) else 0.0
            p.resetJointState(self.robot_id, idx, target, targetVelocity=0.0)

    def get_qpos(self):
        """获取当前所有活动关节角"""
        qpos = []
        for idx in self.all_active_joint_indices:
            state = p.getJointState(self.robot_id, idx)
            qpos.append(state[0])
        return np.array(qpos)

    def get_qvel(self):
        """获取当前所有活动关节速度"""
        qvel = []
        for idx in self.all_active_joint_indices:
            state = p.getJointState(self.robot_id, idx)
            qvel.append(state[1])
        return np.array(qvel)

    def get_ee_pose(self):
        """获取末端执行器位姿 (gripper_link)"""
        num_joints = p.getNumJoints(self.robot_id)
        gripper_link = -1
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            if "right_gripper_link" in info[12].decode():
                gripper_link = i
                break
        if gripper_link >= 0:
            state = p.getLinkState(self.robot_id, gripper_link)
            return np.array(state[0]), np.array(state[1])
        return None, None

    def set_base_pose(self, base_pos, base_quat):
        """设置机器人基座位姿

        Args:
            base_pos: (3,) 基座位置 (SAPIEN 坐标系)
            base_quat: (4,) 基座朝向四元数 (w,x,y,z)
        """
        self.arm_base_pos = np.array(base_pos)
        self.arm_base_q = np.array(base_quat)
        # PyBullet 的四元数格式是 [x,y,z,w]
        pb_quat = [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
        p.resetBasePositionAndOrientation(
            self.robot_id, base_pos.tolist(), pb_quat
        )

    def step(self, target_qpos):
        """执行一个控制步: PD控制 + 重力补偿 + 物理仿真 (与 SAPIEN 04 一致)

        与 SAPIEN 04 的 _physics_step 等价:
          1. 读取当前关节状态 (q, q_dot)
          2. 计算重力补偿力矩: tau_gravity = calculateInverseDynamics(q, q_dot, 0)
          3. 计算 PD 力矩: tau_pd = kp*(q_target - q) - kd*q_dot
          4. 施加总力矩: tau = tau_pd + tau_gravity
          5. 物理引擎步进 (DECIMATION 次)

        与运动学控制 (resetJointState) 的区别:
          - 机器人有真实物理: 惯性、重力、碰撞响应
          - 关节不会完美跟踪目标 (有 PD 误差), 但有重力补偿减少下垂
          - 可被 GLB 物体推动 (碰撞交互)
        """
        num_joints = len(self.all_active_joint_indices)
        for _ in range(DECIMATION):
            # 读取当前关节状态
            current_qpos = self.get_qpos()
            current_qvel = self.get_qvel()

            # 计算重力补偿力矩 (逆动力学, 零加速度 → 只含重力+科氏力)
            tau_gravity = p.calculateInverseDynamics(
                self.robot_id,
                current_qpos.tolist(),
                current_qvel.tolist(),
                [0.0] * num_joints
            )

            # 计算 PD + 重力补偿力矩, 施加到每个关节
            for i, idx in enumerate(self.all_active_joint_indices):
                target = target_qpos[i] if i < len(target_qpos) else 0.0
                # 选择增益: 臂关节 vs 夹爪关节
                if idx in self.gripper_joint_indices:
                    kp, kd = PD_KP_GRIPPER, PD_KD_GRIPPER
                else:
                    kp, kd = PD_KP_ARM, PD_KD_ARM
                tau_pd = kp * (target - current_qpos[i]) - kd * current_qvel[i]
                tau = tau_pd + tau_gravity[i]
                p.setJointMotorControl2(self.robot_id, idx, p.TORQUE_CONTROL, force=float(tau))

            p.stepSimulation()

    def load_glb_objects(self, glb_path, transform_params_path):
        """加载 GLB 场景物体 (与 02 的变换链一致)

        变换链: GLB (RAS y-down) → HaWoR render world (y-up) → SAPIEN (z-up)
        物体分类:
          - 大型扁平几何体 (桌面/地板): mass=0 → static
          - 小物体: mass=volume*density → dynamic
        """
        params = np.load(transform_params_path)
        s_inv = float(params['s_inv'])
        R_inv = params['R_inv']
        t_inv = params['t_inv']

        scene = trimesh.load(glb_path)
        self.obj_ids = []
        self.obj_info = {}

        # 第一遍: 计算所有物体的最低 Z 坐标和质心
        all_centroids = []
        all_verts_z = []
        geom_data = []
        for geom_name, geometry in scene.geometry.items():
            vertices = np.array(geometry.vertices)
            if len(vertices) == 0:
                continue
            # 与 02 一致的变换链
            vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
            vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
            centroid = vertices_sapien.mean(axis=0)
            all_centroids.append(centroid)
            all_verts_z.append(vertices_sapien[:, 2].min())
            geom_data.append((geom_name, geometry, vertices_sapien, centroid))

        if not all_centroids:
            print("  No geometry found in GLB")
            return

        # 计算支撑面高度 (与 04 一致: 物体最低 Z 下方 2mm, 紧贴支撑)
        min_z = min(all_verts_z)
        ground_z = min_z - 0.002
        # 计算 GLB 物体 XY 范围和质心 (用于桌面尺寸)
        all_verts_xy = np.array([[v[0], v[1]] for _, _, v, _ in geom_data for v in [v.min(axis=0), v.max(axis=0)]])
        all_centroids_arr = np.array(all_centroids)
        center_xy = all_centroids_arr[:, :2].mean(axis=0)
        extent_xy = all_verts_xy.max(axis=0) - all_verts_xy.min(axis=0)
        print(f"  Object Z range: min={min_z:.4f}m, ground/table at Z={ground_z:.4f}m")
        print(f"  Object XY center: {center_xy}, extent: {extent_xy}")

        # 移除默认地面, 添加自定义高度地面
        p.removeBody(self.plane_id)
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, ground_z])

        # 添加可见桌面支撑 (与 04 的 support_table 一致): 木色 box, 紧贴物体最低点
        table_half_x = max(0.3, extent_xy[0] / 2 + 0.1)
        table_half_y = max(0.3, extent_xy[1] / 2 + 0.1)
        table_half_z = 0.005  # 桌面厚度 1cm
        table_col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[table_half_x, table_half_y, table_half_z])
        table_vis_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[table_half_x, table_half_y, table_half_z],
            rgbaColor=[0.55, 0.45, 0.35, 1.0]  # 木色
        )
        # 桌面顶部 = ground_z, 桌面中心 Z = ground_z - table_half_z
        self.table_id = p.createMultiBody(
            baseMass=0,  # kinematic (固定)
            baseCollisionShapeIndex=table_col_shape,
            baseVisualShapeIndex=table_vis_shape,
            basePosition=[float(center_xy[0]), float(center_xy[1]), ground_z - table_half_z]
        )
        p.changeDynamics(self.table_id, -1, lateralFriction=1.0, restitution=0.1)
        print(f"  桌面: center=({center_xy[0]:.3f}, {center_xy[1]:.3f}), half_size=({table_half_x:.3f}, {table_half_y:.3f}), top_z={ground_z:.4f}")

        # 第二遍: 创建物体
        for geom_name, geometry, vertices_sapien, centroid in geom_data:
            bbox_size = vertices_sapien.max(axis=0) - vertices_sapien.min(axis=0)
            volume = abs(bbox_size[0] * bbox_size[1] * bbox_size[2])
            max_extent = max(bbox_size)
            flatness = bbox_size[2] / max(max(bbox_size[0], bbox_size[1]), 1e-6)
            is_static = (volume > 0.01 and flatness < 0.3) or max_extent > 0.8

            # 导出 OBJ 给 PyBullet
            temp_dir = tempfile.mkdtemp(prefix='glb_obj-')
            obj_path = f"{temp_dir}/{geom_name}.obj"
            mesh = trimesh.Trimesh(vertices=vertices_sapien - centroid, faces=np.array(geometry.faces))
            mesh.export(obj_path)

            try:
                collision_shape = p.createCollisionShape(p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1])
                visual_shape = p.createVisualShape(
                    p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1],
                    rgbaColor=[0.8, 0.6, 0.4, 1.0] if is_static else [0.4, 0.7, 0.9, 1.0]
                )

                if is_static:
                    obj_id = p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=collision_shape,
                        baseVisualShapeIndex=visual_shape,
                        basePosition=centroid.tolist()
                    )
                    print(f"  -> {geom_name}: static (vol={volume:.4f}m3)")
                else:
                    mass = max(0.01, volume * 1000)
                    obj_id = p.createMultiBody(
                        baseMass=mass,
                        baseCollisionShapeIndex=collision_shape,
                        baseVisualShapeIndex=visual_shape,
                        basePosition=centroid.tolist()
                    )
                    p.changeDynamics(obj_id, -1, lateralFriction=1.0, restitution=0.1)
                    print(f"  -> {geom_name}: dynamic (mass={mass:.3f}kg)")

                self.obj_ids.append(obj_id)
                self.obj_info[obj_id] = {
                    'name': geom_name,
                    'is_static': is_static,
                    'init_pos': centroid.copy()
                }
            except Exception as e:
                print(f"  ! {geom_name}: load failed - {e}")

        # 稳定化
        print("  Stabilizing objects...")
        if self.single_gripper:
            # 单夹爪模式: 只设置夹爪开合, 无机械臂关节
            init_qpos = np.zeros(len(self.all_active_joint_indices))
            for i, idx in enumerate(self.gripper_joint_indices):
                init_qpos[self.all_active_joint_indices.index(idx)] = GRIPPER_INIT_OPEN
        else:
            init_qpos = np.zeros(len(self.all_active_joint_indices))
            for j, idx in enumerate(self.arm_joint_indices):
                if j < len(RIGHT_ARM_STARTING):
                    init_qpos[j] = RIGHT_ARM_STARTING[j]
            init_qpos[len(self.arm_joint_indices)] = 0.04
            init_qpos[len(self.arm_joint_indices) + 1] = -0.04

        for _ in range(500):
            for i, idx in enumerate(self.all_active_joint_indices):
                p.resetJointState(self.robot_id, idx, init_qpos[i], targetVelocity=0.0)
            p.stepSimulation()
        print(f"  {len(self.obj_ids)} objects loaded and stabilized")

    def render_frame(self, cam_pos, cam_R):
        """渲染当前场景的一帧

        Args:
            cam_pos: (3,) 相机位置 (SAPIEN 坐标系)
            cam_R: (3, 3) 相机旋转矩阵 (SAPIEN 约定: X=forward, Y=left, Z=up)

        Returns:
            np.ndarray: (H, W, 3) BGR 图像
        """
        view_matrix, proj_matrix = sapien_cam_to_pybullet_view(
            cam_pos, cam_R, self.cam_width, self.cam_height, fov_deg=self.cam_fov_deg
        )

        _, _, px, _, _ = p.getCameraImage(
            self.cam_width, self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if not self.gui else p.ER_TINY_RENDERER
        )

        # PyBullet 返回 RGB, 转BGR for OpenCV
        rgb = np.array(px, dtype=np.uint8).reshape(self.cam_height, self.cam_width, 4)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        return bgr

    def render_video(self, trajectory_path, output_path, R_c2w_all, t_c2w_all,
                     num_frames=-1, base_pos=None, base_q=None, view="fpv"):
        """渲染视频: 加载轨迹, 跟随 HaWoR 相机, 循环渲染, 保存 mp4

        Args:
            trajectory_path: npy 轨迹文件路径 (N, 8): [arm_joint1..6, gripper1, gripper2]
            output_path: 输出 mp4 路径
            R_c2w_all: (N, 3, 3) HaWoR 相机旋转矩阵
            t_c2w_all: (N, 3) HaWoR 相机平移向量
            num_frames: 渲染帧数 (-1=全部)
            base_pos: (3,) 机器人基座位置 (None=使用已设置的)
            base_q: (4,) 机器人基座朝向 (None=使用已设置的)
            view: 相机视角 (fpv=跟随HaWoR相机, topdown=俯视, behind=后方, front=前方)
        """
        print(f"\n{'='*60}")
        print("Rendering Video (对齐 02 run_robot_tracking)")
        print(f"{'='*60}")

        # 加载轨迹
        trajectory = np.load(trajectory_path)
        total_frames = len(trajectory)
        if num_frames < 0 or num_frames > total_frames:
            num_frames = total_frames
        print(f"  Trajectory: {trajectory_path}")
        print(f"  Frames: {num_frames}/{total_frames}")
        print(f"  Camera: HaWoR trajectory ({len(R_c2w_all)} frames)")
        print(f"  Output: {output_path}")

        # 设置基座
        if base_pos is not None and base_q is not None:
            self.set_base_pose(base_pos, base_q)
        if self.arm_base_pos is None:
            print("  ⚠ 基座未设置, 使用默认 [0,0,0]")
            self.set_base_pose([0, 0, 0], [1, 0, 0, 0])

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 设置初始位置 (轨迹第一帧)
        first_frame = trajectory[0]
        self._set_qpos(first_frame)

        # 稳定化
        for _ in range(100):
            for i, idx in enumerate(self.all_active_joint_indices):
                p.resetJointState(self.robot_id, idx, first_frame[i], targetVelocity=0.0)
            p.stepSimulation()

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (self.cam_width, self.cam_height))
        print(f"  Video: {self.cam_width}x{self.cam_height} @ {VIDEO_FPS}fps, FOV={self.cam_fov_deg:.1f}°, view={view}")

        # 计算固定视角的相机位置 (非 fpv 模式)
        fixed_cam_pos = None
        fixed_cam_R = None
        if view != "fpv":
            # 用基座位置作为场景中心
            scene_center = np.array(self.arm_base_pos) if self.arm_base_pos is not None else np.array([0, 0, 0.3])
            if view == "topdown":
                fixed_cam_pos = scene_center + np.array([0, 0, 1.2])
                # 俯视: 相机朝下
                fixed_cam_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
            elif view == "behind":
                fixed_cam_pos = scene_center + np.array([-0.4, -0.5, 0.3])
                forward = scene_center - fixed_cam_pos
                forward /= np.linalg.norm(forward)
                up = np.array([0, 0, 1.0])
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                cam_up = np.cross(right, forward)
                fixed_cam_R = np.column_stack([forward, -right, cam_up])
            elif view == "front":
                fixed_cam_pos = scene_center + np.array([0.5, 0.3, 0.3])
                forward = scene_center - fixed_cam_pos
                forward /= np.linalg.norm(forward)
                up = np.array([0, 0, 1.0])
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                cam_up = np.cross(right, forward)
                fixed_cam_R = np.column_stack([forward, -right, cam_up])
            print(f"  固定视角 ({view}): cam_pos={fixed_cam_pos}")

        # 渲染循环
        print(f"\n  渲染循环 (PD控制 + 重力补偿, 与 SAPIEN 04 一致) ...")
        for frame_idx in range(num_frames):
            # 获取当前帧的关节角 (02 的 IK 解, 作为 PD 控制目标)
            qpos = trajectory[frame_idx]

            # PD控制 + 重力补偿 + 物理仿真
            self.step(qpos)

            # 读取物理仿真后的实际位姿 (与 02 的 IK 对比)
            actual_qpos = self.get_qpos()
            tracking_err = np.mean(np.abs(actual_qpos - qpos[:len(actual_qpos)]))

            # 计算相机位姿
            if view == "fpv":
                # fpv: 跟随 HaWoR 相机轨迹 (与 02 一致)
                cam_pos, cam_R = hawor_cam_to_sapien_pose(R_c2w_all[frame_idx], t_c2w_all[frame_idx])
            else:
                # 固定视角
                cam_pos, cam_R = fixed_cam_pos, fixed_cam_R

            # 渲染帧
            bgr = self.render_frame(cam_pos, cam_R)

            # 添加帧信息 (含 PD 跟踪误差: 物理仿真 vs IK 目标)
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            label = f"Frame {frame_idx+1}/{num_frames} | Physics | IK err={tracking_err:.4f}"
            cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            writer.write(bgr)

            if (frame_idx + 1) % 30 == 0:
                ee_pos, _ = self.get_ee_pose()
                ee_str = f", EE={np.array2string(ee_pos, precision=3)}" if ee_pos is not None else ""
                print(f"  Frame {frame_idx+1}/{num_frames} | IK err={tracking_err:.4f}{ee_str}")

        writer.release()
        print(f"\n  Video saved: {output_path}")

        # 检查文件大小
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"  File size: {file_size:.1f} MB")
        return True

    def render_single_gripper_video(self, hawor_dir, transform_params_path, glb_path, output_path,
                                     R_c2w_all, t_c2w_all, hand_idx=0, num_frames=-1,
                                     img_focal=None, view="fpv"):
        """单夹爪模式渲染: 只加载夹爪URDF (无机械臂), 直接用MANO手腕位姿驱动夹爪

        与 render_video 的区别:
        - 无机械臂, 无 IK, 无轨迹文件
        - 夹爪 root 位姿直接从 MANO 手腕/指尖解析计算 (_compute_analytical_gripper_pose)
        - 夹爪手指关节从 MANO 指尖距离计算
        - 物理仿真: 夹爪可碰撞/推动 GLB 物体 (运动学控制 + 物理交互)

        参考: hand_track/render_gripper_only.py 的 --mode gripper

        Args:
            hawor_dir: HaWoR 输出目录
            transform_params_path: transform_params.npz 路径
            glb_path: final_scene.glb 路径
            output_path: 输出 mp4 路径
            R_c2w_all: (N, 3, 3) HaWoR 相机旋转矩阵
            t_c2w_all: (N, 3) HaWoR 相机平移向量
            hand_idx: 手部索引 (0=左手, 1=右手)
            num_frames: 渲染帧数 (-1=全部)
            img_focal: 焦距 (None=用默认值)
            view: 相机视角 (fpv/topdown/behind/front)
        """
        print(f"\n{'='*60}")
        print("单夹爪模式渲染 (无机械臂, MANO手腕直接驱动)")
        print(f"{'='*60}")

        # [1/6] 加载 HaWoR 数据
        print(f"\n[1/6] 加载 HaWoR 数据 ...")
        mod = _load_module_02()
        hawor_data = mod.load_hawor_data(hawor_dir, hand_idx=hand_idx)
        n_total = len(hawor_data["pred_trans"])
        if num_frames < 0 or num_frames > n_total:
            num_frames = n_total
        print(f"  帧数: {num_frames}/{n_total}")

        # 设置相机 FOV
        if img_focal is not None:
            self.cam_fov_deg = compute_cam_fov(img_focal, self.cam_height, self.cam_width)

        # MANO layer (用于计算关节位置)
        from mano_layer import MANOLayer
        betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
        mano_side = "left" if hand_idx == 0 else "right"
        mano_layer = MANOLayer(mano_side, betas_mean)
        compute_mano_joints = mod.compute_mano_joints

        # [2/6] 加载 GLB 物体 (带桌面支撑)
        print(f"\n[2/6] 加载 GLB 物体 ...")
        self.load_glb_objects(glb_path, transform_params_path)

        # [3/6] 预计算手腕位置 (用于相机中心)
        print(f"\n[3/6] 预计算手腕位置 ...")
        wrist_positions = compute_wrist_positions_sapien(hawor_data, mano_layer, 0, num_frames)
        scene_center = np.mean(wrist_positions, axis=0) if wrist_positions else np.array([0, 0, 0.3])

        # [4/6] 设置初始夹爪位姿 (第一帧)
        print(f"\n[4/6] 设置初始夹爪位姿 ...")
        # MANO 关节索引: 拇指尖=4, 食指尖=8, 手腕=0
        ref_indices = [4, 8]

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # [5/6] 设置相机
        print(f"\n[5/6] 设置相机 (view={view}) ...")
        fixed_cam_pos = None
        fixed_cam_R = None
        if view != "fpv":
            if view == "topdown":
                fixed_cam_pos = scene_center + np.array([0, 0, 1.2])
                fixed_cam_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
            elif view == "behind":
                fixed_cam_pos = scene_center + np.array([-0.4, -0.5, 0.3])
                forward = scene_center - fixed_cam_pos
                forward /= np.linalg.norm(forward)
                up = np.array([0, 0, 1.0])
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                cam_up = np.cross(right, forward)
                fixed_cam_R = np.column_stack([forward, -right, cam_up])
            elif view == "front":
                fixed_cam_pos = scene_center + np.array([0.5, 0.3, 0.3])
                forward = scene_center - fixed_cam_pos
                forward /= np.linalg.norm(forward)
                up = np.array([0, 0, 1.0])
                right = np.cross(forward, up)
                right /= np.linalg.norm(right)
                cam_up = np.cross(right, forward)
                fixed_cam_R = np.column_stack([forward, -right, cam_up])
            print(f"  固定视角 ({view}): cam_pos={fixed_cam_pos}")

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, VIDEO_FPS, (self.cam_width, self.cam_height))
        print(f"  Video: {self.cam_width}x{self.cam_height} @ {VIDEO_FPS}fps, FOV={self.cam_fov_deg:.1f}°, view={view}")

        # [6/6] 渲染循环
        print(f"\n[6/6] 单夹爪渲染循环 (PD控制 + 重力补偿) ...")

        # 保存上一帧的目标 (用于无效帧保持目标)
        finger_targets = [GRIPPER_INIT_OPEN, GRIPPER_INIT_OPEN]
        root_pos = np.array([0, 0, 0])
        pb_quat = [0, 0, 0, 1]

        for frame_idx in range(num_frames):
            if hawor_data["pred_valid"][frame_idx]:
                # 计算 MANO 关节
                _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][frame_idx],
                                            hawor_data["pred_hand_pose"][frame_idx],
                                            hawor_data["pred_trans"][frame_idx])
                joints_sapien = (RXWORLD_TO_SAPIEN @ j.T).T

                mano_wrist = joints_sapien[0, :3]
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]

                # 解析计算夹爪位姿和手指关节
                root_pos, root_R, joint1, joint2 = _compute_analytical_gripper_pose(
                    mano_wrist, mano_finger1, mano_finger2)

                # 设置夹爪 root 位姿 (运动学, 像 SAPIEN 的 set_root_pose)
                root_quat_wxyz = pr.quaternion_from_matrix(root_R)
                pb_quat = [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]
                p.resetBasePositionAndOrientation(
                    self.robot_id, root_pos.tolist(), pb_quat
                )

                # 更新目标手指关节值
                finger_targets = [float(joint1), float(joint2)]

            # 物理仿真步进 (PD控制 + 重力补偿, 让 GLB 物体与夹爪交互)
            num_joints = len(self.all_active_joint_indices)
            for _ in range(DECIMATION):
                # 每个子步重置 root 位姿 (运动学控制, 像 SAPIEN 的 set_root_pose)
                p.resetBasePositionAndOrientation(
                    self.robot_id, root_pos.tolist(), pb_quat
                )

                # PD控制 + 重力补偿 (手指关节, 与 SAPIEN 04 一致)
                current_qpos = self.get_qpos()
                current_qvel = self.get_qvel()
                tau_gravity = p.calculateInverseDynamics(
                    self.robot_id,
                    current_qpos.tolist(),
                    current_qvel.tolist(),
                    [0.0] * num_joints
                )
                for i, idx in enumerate(self.all_active_joint_indices):
                    target = finger_targets[i] if i < len(finger_targets) else 0.0
                    tau_pd = PD_KP_GRIPPER * (target - current_qpos[i]) - PD_KD_GRIPPER * current_qvel[i]
                    tau = tau_pd + tau_gravity[i]
                    p.setJointMotorControl2(self.robot_id, idx, p.TORQUE_CONTROL, force=float(tau))

                p.stepSimulation()

            # 读取物理仿真后的实际位姿 (与目标对比)
            actual_qpos = self.get_qpos()
            tracking_err = np.mean(np.abs(actual_qpos - np.array(finger_targets[:len(actual_qpos)])))

            # 计算相机位姿
            if view == "fpv":
                cam_pos, cam_R = hawor_cam_to_sapien_pose(R_c2w_all[frame_idx], t_c2w_all[frame_idx])
            else:
                cam_pos, cam_R = fixed_cam_pos, fixed_cam_R

            # 渲染帧
            bgr = self.render_frame(cam_pos, cam_R)

            # 添加帧信息 (含 PD 跟踪误差)
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            label = f"Frame {frame_idx+1}/{num_frames} | Single-Gripper Physics | err={tracking_err:.4f}"
            cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            writer.write(bgr)

            if (frame_idx + 1) % 30 == 0:
                print(f"  Frame {frame_idx+1}/{num_frames} | PD err={tracking_err:.4f}")

        writer.release()
        print(f"\n  Video saved: {output_path}")

        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"  File size: {file_size:.1f} MB")
        return True

    def test_hold_position(self, num_frames=100):
        """测试1: 保持初始位置"""
        print(f"\n{'='*60}")
        print("Test 1: Hold position (kinematic control)")
        print(f"{'='*60}")

        init_qpos = np.zeros(len(self.all_active_joint_indices))
        for j, idx in enumerate(self.arm_joint_indices):
            if j < len(RIGHT_ARM_STARTING):
                init_qpos[j] = RIGHT_ARM_STARTING[j]
        init_qpos[len(self.arm_joint_indices)] = 0.04
        init_qpos[len(self.arm_joint_indices) + 1] = -0.04
        self._set_qpos(init_qpos)

        init_ee_pos, _ = self.get_ee_pose()
        max_ee_drift = 0

        for frame in range(num_frames):
            self.step(init_qpos)

            ee_pos, _ = self.get_ee_pose()
            if init_ee_pos is not None and ee_pos is not None:
                ee_drift = np.linalg.norm(ee_pos - init_ee_pos) * 1000
                max_ee_drift = max(max_ee_drift, ee_drift)

            if frame % 30 == 0:
                current = self.get_qpos()
                arm_current = np.array([current[self.all_active_joint_indices.index(idx)] for idx in self.arm_joint_indices])
                arm_target = np.array([init_qpos[self.all_active_joint_indices.index(idx)] for idx in self.arm_joint_indices])
                drift = np.degrees(np.abs(arm_current - arm_target)).max()
                ee_str = f", EE drift={ee_drift:.1f}mm" if init_ee_pos is not None else ""
                print(f"  Frame {frame:3d}: joint drift={drift:.4f} deg{ee_str}")

        passed = max_ee_drift < 10.0
        status = "PASS" if passed else "FAIL"
        print(f"\nResult [{status}]: max EE drift={max_ee_drift:.1f}mm")
        return passed

    def cleanup(self):
        p.disconnect()


def main():
    parser = argparse.ArgumentParser(description="PyBullet Physics Pipeline (对齐 02 run_robot_tracking)")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--render-video", action="store_true", help="Render video from trajectory")
    parser.add_argument("--gui", action="store_true", help="Use GUI mode")
    parser.add_argument("--hawor-dir", type=str, default=DEFAULT_HAWOR_DIR,
                        help=f"HaWoR 输出目录 (默认: {DEFAULT_HAWOR_DIR})")
    parser.add_argument("--ras-dir", type=str, default=DEFAULT_RAS_DIR,
                        help=f"RAS 输出目录 (默认: {DEFAULT_RAS_DIR})")
    parser.add_argument("--transform-params", type=str, default=DEFAULT_TRANSFORM_PARAMS,
                        help=f"transform_params.npz 路径")
    parser.add_argument("--trajectory", type=str, default=DEFAULT_TRAJECTORY,
                        help=f"qpos 轨迹文件 (默认: 02 保存的 {DEFAULT_TRAJECTORY})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"输出视频路径")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="渲染帧数 (-1=全部)")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="手部索引: 0=左手, 1=右手, -1=自动检测")
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH, help="渲染宽度")
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT, help="渲染高度")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角: fpv=第一人称(跟随HaWoR相机轨迹), topdown=俯视, behind=后方, front=前方")
    parser.add_argument("--single-gripper", action="store_true",
                        help="单夹爪模式: 只加载夹爪(无机械臂), 直接用MANO手腕位姿驱动, 参考 hand_track/render_gripper_only.py")
    args = parser.parse_args()

    run_all = not (args.test or args.render_video)
    results = {}

    if run_all or args.test:
        pipeline = PyBulletPipeline(gui=args.gui, cam_width=args.width, cam_height=args.height)
        results["hold_position"] = pipeline.test_hold_position()
        pipeline.cleanup()

    if run_all or args.render_video:
        # 检查必要文件
        hawor_dir = Path(args.hawor_dir)
        ras_dir = Path(args.ras_dir)
        glb_path = ras_dir / "final_scene.glb"
        transform_params = Path(args.transform_params)
        trajectory_path = Path(args.trajectory)

        if not glb_path.exists():
            print(f"✗ GLB 文件不存在: {glb_path}")
            return False
        if not transform_params.exists():
            print(f"✗ 变换参数不存在: {transform_params}")
            return False
        # 单夹爪模式不需要轨迹文件 (直接用 MANO 手腕位姿驱动)
        if not args.single_gripper and not trajectory_path.exists():
            print(f"✗ 轨迹文件不存在: {trajectory_path}")
            return False

        # 加载 HaWoR 数据 (复用 02 的函数)
        print(f"\n[1/4] 加载 HaWoR 数据 ...")
        mod = _load_module_02()

        # 自动检测手部索引
        hand_idx = args.hand_idx
        if hand_idx < 0:
            detected = mod._detect_hand_idx(hawor_dir)
            hand_idx = detected if detected is not None else 0
            hand_label = "左手" if hand_idx == 0 else "右手"
            print(f"  自动检测: {hand_label} (idx={hand_idx})")

        hawor_data = mod.load_hawor_data(hawor_dir, hand_idx=hand_idx)
        R_c2w_all, t_c2w_all = mod.load_hawor_c2w(hawor_dir)
        img_focal = hawor_data.get("img_focal", None)
        print(f"  帧数: {len(hawor_data['pred_trans'])}")
        print(f"  相机轨迹: {R_c2w_all.shape[0]} 帧" if R_c2w_all is not None else "  相机轨迹: 无")
        print(f"  焦距: {img_focal}")

        if args.single_gripper:
            # 单夹爪模式: 无机械臂, 无 IK, 无轨迹文件, 直接用 MANO 手腕位姿驱动夹爪
            print(f"\n[2/3] 创建 PyBullet 单夹爪管线 + 加载 GLB ...")
            pipeline = PyBulletPipeline(gui=args.gui, cam_width=args.width, cam_height=args.height,
                                         single_gripper=True)
            # 设置相机 FOV (与 02 一致)
            pipeline.cam_fov_deg = compute_cam_fov(img_focal, args.height, args.width)

            # 加载 GLB 物体 (load_glb_objects 内部会创建桌面支撑)
            pipeline.load_glb_objects(str(glb_path), str(transform_params))

            # 渲染单夹爪视频
            print(f"\n[3/3] 渲染单夹爪视频 ...")
            results["render_video"] = pipeline.render_single_gripper_video(
                str(hawor_dir), str(transform_params), str(glb_path), args.output,
                R_c2w_all, t_c2w_all,
                hand_idx=hand_idx,
                num_frames=args.num_frames,
                img_focal=img_focal,
                view=args.view
            )
            pipeline.cleanup()
        else:
            # 完整机械臂模式: 计算最优基座 (与 02 一致)
            print(f"\n[2/4] 计算最优基座位置 ...")
            from mano_layer import MANOLayer
            import torch
            betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
            mano_side = "left" if hand_idx == 0 else "right"
            mano_layer = MANOLayer(mano_side, betas_mean)

            num_frames = args.num_frames if args.num_frames > 0 else len(hawor_data["pred_trans"])
            wrist_positions = compute_wrist_positions_sapien(hawor_data, mano_layer, 0, num_frames)
            arm_base_pos, arm_base_q = compute_optimal_fixed_base(wrist_positions)
            print(f"  手腕质心: {np.mean(wrist_positions, axis=0)}")
            print(f"  基座位置: {arm_base_pos}")
            print(f"  基座朝向: {arm_base_q}")

            # 创建 PyBullet 管线
            print(f"\n[3/4] 创建 PyBullet 管线 + 加载 GLB ...")
            pipeline = PyBulletPipeline(gui=args.gui, cam_width=args.width, cam_height=args.height)

            # 设置相机 FOV (与 02 一致)
            pipeline.cam_fov_deg = compute_cam_fov(img_focal, args.height, args.width)

            # 设置基座
            pipeline.set_base_pose(arm_base_pos, arm_base_q)

            # 加载 GLB 物体
            pipeline.load_glb_objects(str(glb_path), str(transform_params))

            # 渲染视频
            print(f"\n[4/4] 渲染视频 ...")
            results["render_video"] = pipeline.render_video(
                str(trajectory_path), args.output,
                R_c2w_all, t_c2w_all,
                num_frames=args.num_frames,
                base_pos=arm_base_pos,
                base_q=arm_base_q,
                view=args.view
            )
            pipeline.cleanup()

    if results:
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        for name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {name}: {status}")

    return all(results.values()) if results else True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
