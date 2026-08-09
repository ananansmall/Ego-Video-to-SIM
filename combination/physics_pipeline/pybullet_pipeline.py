#!/usr/bin/env python3
"""
PyBullet 物理仿真管线 — 对齐 04_physics_simulation.py

设计原则:
  与 04_physics_simulation.py 完全对齐 (同一套 IK + 坐标变换 + 基座高度):
  - IK 来源: 04 生成的轨迹 (基座 0.70m, DexRetargeting + RelaxedIK)
             可选 02 的轨迹 (--use-02-trajectory, 基座 0.35m)
  - 相机: 跟随 HaWoR 相机轨迹 (R_c2w, t_c2w), 与 04 第一人称视角一致
  - 机器人基座: 基于手腕位置计算最优固定基座 (与 04 一致, 0.70m)
  - GLB 物体: 使用相同的变换链 (s_inv*R_inv@p + t_inv → RXWORLD_TO_SAPIEN@p → Z翻转)
  - 坐标变换: FLIP_Z_FOR_PHYSICS=True (与 04 一致, Z 翻转)
  - 关节映射: 按 URDF 关节名匹配 (right_arm_joint1..6 + right_gripper_finger_joint1..2)

与 04 的唯一区别:
  - 04: SAPIEN PD驱动 + 重力补偿 (PhysX, 真实物理交互)
  - 本脚本: PyBullet 运动学控制 (对齐 GalaxeaManipSim 效果) + GLB 物体物理交互

控制策略 (对齐 GalaxeaManipSim 效果):
  GalaxeaManipSim 使用 PD驱动 + 重力补偿 (Kp=1000, Kd=200), 机械臂非常刚硬,
  近似运动学控制。在 PyBullet 中, R1 URDF 连杆惯性极小, computed torque control
  不稳定。因此采用运动学控制 (resetJointState) 直接设置目标关节角,
  达到与 GalaxeaManipSim 同样的效果: 机械臂精确跟踪目标, 仍能推动 GLB 物体。

坐标变换 (与 04 一致, Z 翻转):
  FLIP_Z_FOR_PHYSICS=True, 物体/手腕/相机坐标变换后翻转 Z, 适配物理重力。

调用方式:
  # 使用 04 生成的轨迹渲染视频 (推荐, 基座 0.70m, transform-params/trajectory 自动推导)
  # 先运行 04 生成轨迹:
  #   python 04_physics_simulation.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result --fast-collision
  # 再运行 PyBullet:
  python pybullet_pipeline.py --render-video \
      --hawor-dir /home/an/data/hawor/7 \
      --ras-dir /home/an/data/ras/my_7mp4_result

  # 强制使用 02 的轨迹 (基座 0.35m, 会自动调整基座高度)
  python pybullet_pipeline.py --render-video --use-02-trajectory \
      --hawor-dir /home/an/data/hawor/7 \
      --ras-dir /home/an/data/ras/my_7mp4_result

  # 手动指定路径 (覆盖自动推导)
  python pybullet_pipeline.py --render-video \
      --hawor-dir /home/an/data/hawor/7 \
      --ras-dir /home/an/data/ras/my_7mp4_result \
      --transform-params ./output/7_my_7mp4_result/alignment/transform_params.npz \
      --trajectory ./output/7_my_7mp4_result/tracking/physics_sim_physics_tracking.npy

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

# ============ 坐标变换 (与 04_physics_simulation.py 一致, Z 翻转) ============
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# 物理仿真 Z 翻转标志 (与 04_physics_simulation.py 一致)
# 原始 RXWORLD_TO_SAPIEN: SAPIEN_Z = HaWoR_Y, 当 HaWoR Y 负 (上方) 时 SAPIEN Z 负 (地下)
# 物理仿真中重力 -Z, 物体 Z 负会继续往下掉, 不符合物理
# FLIP_Z_FOR_PHYSICS=True 时, 变换后翻转 Z 坐标: SAPIEN_Z = -HaWoR_Y
# 根据当前 GLB 场景判断: 需要翻转 (对齐 04)
FLIP_Z_FOR_PHYSICS = True

# ============ 机器人参数 (与 04_physics_simulation.py 一致) ============
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
# 基座高度: 0.70m, 让机械臂垂直抓取, 不靠近桌面 (与 04 一致)
# 注意: 必须使用 04 生成的 IK 轨迹 (04 的 IK 基于 0.70m 基座计算)
# 如果使用 02 的轨迹 (基于 0.35m), 机械臂位置会偏移 0.35m
COMFORTABLE_REACH = 0.70  # 基座高度: 与 04_physics_simulation.py 一致
COMFORT_TARGET_IN_BASE = np.array([0.25, 0.0, -0.55])  # 与 04 一致: 更低更近, 机械臂垂直下垂
BASE_TRACKING_RANGE = 0.0  # 固定底座: 不跟踪手腕 (与 04 一致)
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
  <!-- 手腕基座: 虚拟 link, 作为运动学根 -->
  <link name="{prefix}_wrist_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>
  <!-- 手腕关节: revolute, 允许夹爪绕手腕旋转 (对齐 GalaxeaManipSim 的 arm_joint6) -->
  <joint name="{prefix}_wrist_joint" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="{prefix}_wrist_link"/>
    <child link="{prefix}_gripper_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2.0"/>
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


def _compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2, prefix="right"):
    """从 MANO 3 个特征点计算夹爪 gripper_link 位姿和手指关节值

    与 hand_track/render_gripper_only.py 的 _compute_analytical_gripper_pose 一致:
    方法: 加权 SVD (Procrustes) + 匹配指尖中点
      1. 从 MANO 指尖距离计算手指关节值
      2. 用加权 SVD 找最近正交旋转矩阵, Y 轴 (开合方向) 权重更高,
         优先保证开合方向精确 (因为开合方向直接影响指尖位置)
      3. 匹配两个指尖的中点确定 gripper_link 位置

    关键: MANO 的指向方向 (wrist→finger_mid) 和开合方向 (finger1→finger2)
    通常不正交。当它们非正交时, 标准 SVD 会均等折中, 导致两个方向都不精确。
    给 Y 轴更高权重可以优先保证开合方向精确, 从而最小化指尖位置误差。
    """
    W_Y = 5.0  # Y 轴 (开合方向) 权重, 越大越优先保证开合方向精确

    # 1. 计算手指关节值
    v_finger = mano_finger2 - mano_finger1
    finger_dist = np.linalg.norm(v_finger)
    required_open_sum = finger_dist - _FINGER_BASE_DIST
    joint1 = max(0.0, min(0.05, required_open_sum / 2))
    joint2 = max(0.0, min(0.05, required_open_sum / 2))

    # 2. 加权 SVD 最近正交旋转
    # MANO 方向向量
    finger_mid = (mano_finger1 + mano_finger2) / 2
    pointing = finger_mid - mano_wrist
    pointing = pointing / max(np.linalg.norm(pointing), 1e-6)

    y_sign = 1.0 if prefix == "right" else -1.0
    opening = y_sign * v_finger / max(finger_dist, 1e-6)

    # gripper_link 坐标系中的方向
    # X 轴指向指尖: finger origins 的 X 分量 = 0.037
    # Y 轴为开合方向: finger1 在 -Y, finger2 在 +Y (右手)
    gripper_x = np.array([1.0, 0.0, 0.0])
    gripper_y = np.array([0.0, 1.0, 0.0])

    # 加权 Procrustes: 找 R 使得 R @ [gripper_x, w_y*gripper_y] ≈ [pointing, w_y*opening]
    W = np.diag([1.0, W_Y])
    A = np.column_stack([gripper_x, gripper_y]) @ W  # (3, 2)
    B = np.column_stack([pointing, opening]) @ W      # (3, 2)
    H = A @ B.T  # (3, 3)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    root_R = Vt.T @ np.diag([1.0, 1.0, np.sign(d)]) @ U.T

    # 3. 匹配指尖中点确定 gripper_link 位置
    finger1_in_gripper = _FINGER1_ORIGIN + _FINGER1_AXIS * joint1
    finger2_in_gripper = _FINGER2_ORIGIN + _FINGER2_AXIS * joint2
    finger_mid_in_gripper = (finger1_in_gripper + finger2_in_gripper) / 2
    root_pos = finger_mid - root_R @ finger_mid_in_gripper

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

    与 04_physics_simulation.py 的 hawor_cam_to_sapien_pose 一致 (含 Z 翻转):
    1. 位置/旋转从 HaWoR render world 变换到 SAPIEN 坐标系
    2. OpenGL 相机约定 (Z=后方, X=右, Y=上) → SAPIEN 相机约定 (X=前, Y=左, Z=上)
    3. FLIP_Z_FOR_PHYSICS=True 时翻转 Z 坐标 (对齐 04)

    返回 (cam_pos_sapien, cam_R_sapien), 其中 cam_R_sapien 是 3x3 旋转矩阵
    (列向量: 第0列=forward, 第1列=left, 第2列=up)。
    """
    cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w
    cam_R_gl = RXWORLD_TO_SAPIEN @ R_c2w  # OpenGL 约定: X=右, Y=上, Z=后

    if FLIP_Z_FOR_PHYSICS:
        # Z 翻转: 翻转位置 Z 分量, 只翻转 forward 和 up 的 Z 分量
        cam_pos_sapien = cam_pos_sapien.copy()
        cam_pos_sapien[2] = -cam_pos_sapien[2]
        forward = -cam_R_gl[:, 2].copy()
        up = cam_R_gl[:, 1].copy()
        forward[2] = -forward[2]
        up[2] = -up[2]
        # 重新计算 left = up × forward (保证右手坐标系, det=+1)
        left = np.cross(up, forward)
        left = left / max(np.linalg.norm(left), 1e-8)
        up = np.cross(forward, left)
    else:
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
    """计算机器人基座的最优固定位置和朝向 (固定底座, 不跟踪手腕)

    策略:
    1. 计算所有有效帧手腕位置的质心
    2. 基座放在质心正上方 COMFORTABLE_REACH (0.70m) 处, 让机械臂垂直抓取
    3. 朝向: 绕Z轴旋转180° (面向手腕工作区域)
    """
    if len(wrist_positions_sapien) == 0:
        return np.array([0.0, 0.0, COMFORTABLE_REACH]), pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))

    wrist_arr = np.array(wrist_positions_sapien)
    centroid = wrist_arr.mean(axis=0)

    # 固定底座: XY 在手腕质心位置, Z 在质心上方 COMFORTABLE_REACH
    arm_base_pos = centroid.copy()
    arm_base_pos[2] += COMFORTABLE_REACH

    # 朝向: 绕Z轴旋转180° (面向手腕工作区域)
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


def _load_module_01():
    """加载 01_align_scene.py 模块 (文件名以数字开头, 不能直接 import)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "align_scene_01", str(COMBINATION_DIR / "01_align_scene.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_module_02():
    """加载 02_render_scene.py 模块 (文件名以数字开头, 不能直接 import)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "render_scene_02", str(COMBINATION_DIR / "02_render_scene.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_hawor_npz(hawor_dir):
    """查找 HaWoR reconstruction npz 文件路径"""
    rec_dir = Path(hawor_dir) / "reconstruction"
    if rec_dir.exists():
        for f in rec_dir.glob("hawor_results_*.npz"):
            return str(f)
    return None


def _fill_nan_frames(data):
    """填充数据中的 NaN 帧, 并用最近有效值替换, 同时把 NaN 帧标记为 invalid"""
    n_frames = data["pred_trans"].shape[0]
    float_keys = ["pred_trans", "pred_rot", "pred_hand_pose", "pred_betas"]
    nan_mask = np.zeros(n_frames, dtype=bool)
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind == "f":
            nan_mask |= np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim)))
    if not nan_mask.any():
        return
    data["pred_valid"][nan_mask] = False
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind != "f":
            continue
        last_valid = None
        for i in range(n_frames):
            if not nan_mask[i]:
                last_valid = arr[i].copy()
            elif last_valid is not None:
                arr[i] = last_valid
        first_valid = None
        for i in range(n_frames - 1, -1, -1):
            if not nan_mask[i]:
                first_valid = arr[i].copy()
            elif first_valid is not None:
                arr[i] = first_valid


def _load_hawor_data(hawor_dir, hand_idx=0):
    """不依赖 sapien 的 HaWoR 数据加载 (从 02_render_scene.py 解耦)

    支持 reconstruction/hawor_results_*.npz 和 world_space_res.pth 两种格式。
    """
    import joblib

    hawor_path = Path(hawor_dir)
    rec_file = _find_hawor_npz(hawor_dir)
    ws_file = hawor_path / "world_space_res.pth"

    img_focal = None

    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        pred_trans = rec["pred_trans"]
        pred_rot = rec["pred_rot"]
        pred_hand_pose = rec["pred_hand_pose"]
        pred_betas = rec["pred_betas"]
        pred_valid = rec["pred_valid"]
        if "img_focal" in rec:
            img_focal = float(rec["img_focal"])
    elif ws_file.exists():
        ws = joblib.load(str(ws_file))
        pred_trans = ws[0].numpy() if hasattr(ws[0], "numpy") else np.array(ws[0])
        pred_rot = ws[1].numpy() if hasattr(ws[1], "numpy") else np.array(ws[1])
        pred_hand_pose = ws[2].numpy() if hasattr(ws[2], "numpy") else np.array(ws[2])
        pred_betas = ws[3].numpy() if hasattr(ws[3], "numpy") else np.array(ws[3])
        pred_valid = ws[4] if isinstance(ws[4], np.ndarray) else np.array(ws[4])
    else:
        raise FileNotFoundError(
            f"未找到 hawor 数据文件: {hawor_path / 'reconstruction' / 'hawor_results_*.npz'} 或 {ws_file}"
        )

    est_focal_file = hawor_path / "est_focal.txt"
    if img_focal is None and est_focal_file.exists():
        try:
            img_focal = float(est_focal_file.read_text().strip())
        except Exception:
            pass

    result = {
        "pred_trans": pred_trans[hand_idx],
        "pred_rot": pred_rot[hand_idx],
        "pred_hand_pose": pred_hand_pose[hand_idx],
        "pred_betas": pred_betas[hand_idx],
        "pred_valid": pred_valid[hand_idx],
        "img_focal": img_focal,
    }
    _fill_nan_frames(result)
    return result


def _compute_mano_joints(mano_layer, rot, hand_pose, trans):
    """不依赖 sapien 的 MANO 正运动学 (从 02_render_scene.py 解耦)"""
    import torch

    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = mano_layer(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]


def compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames):
    """预计算所有帧的手腕位置 (SAPIEN 坐标系, 含 Z 翻转, 对齐 04)"""
    positions = []
    for i in range(num_frames):
        global_idx = start_frame + i
        if not hawor_data["pred_valid"][global_idx]:
            continue
        rot = hawor_data["pred_rot"][global_idx]
        trans = hawor_data["pred_trans"][global_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)):
            continue
        _, j = _compute_mano_joints(mano_layer, rot,
                                    hawor_data["pred_hand_pose"][global_idx], trans)
        joints_sapien = (RXWORLD_TO_SAPIEN @ j.T).T
        if FLIP_Z_FOR_PHYSICS:
            joints_sapien[:, 2] = -joints_sapien[:, 2]
        positions.append(joints_sapien[0, :3].copy())
    return positions


class PyBulletPipeline:
    """PyBullet 物理仿真管线: 对齐 04_physics_simulation.py

    与 04 的区别:
    - 04: SAPIEN PD驱动 + 重力补偿 (PhysX)
    - 本类: PyBullet 运动学控制 (对齐 GalaxeaManipSim 效果) + GLB 物体物理交互

    坐标变换: 与 04 一致 (FLIP_Z_FOR_PHYSICS=True, Z 翻转)
    IK 来源: 04 生成的轨迹 (基座 0.70m), 或 02 的轨迹 (基座 0.35m, 可选)
    """

    def __init__(self, gui=False, cam_width=VIDEO_WIDTH, cam_height=VIDEO_HEIGHT,
                 single_gripper=False):
        self.gui = gui
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.cam_fov_deg = 2 * np.degrees(np.arctan(self.cam_height / 2.0 / HAWOR_FOCAL_DEFAULT))
        self.single_gripper = single_gripper

        # 基座参数 (后续由 setup_from_hawor 设置)
        self.arm_base_pos = None
        self.arm_base_q = None

        # GLB 物体列表
        self.obj_ids = []
        self.obj_info = {}

        # 物理客户端 ID (初始化前为 None)
        self.physics_client = None

        # 初始化仿真环境
        self._init_simulation()

    def _init_simulation(self):
        """创建物理客户端、加载地面和机器人

        被 __init__ 和 reset_simulation() 调用, 确保每次渲染都从全新仿真环境开始。
        """
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(PHYSICS_TIMESTEP)
        p.setGravity(*GRAVITY)

        # 加载地面 (z=0, 后面会根据 GLB 物体调整)
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = None  # 桌面 ID (load_glb_objects 时创建)

        if self.single_gripper:
            # 单夹爪模式: 使用 hand_track 的 gripper_config 生成 URDF (与解析法/Dex 一致)
            # gripper_base_link → (fixed) → gripper_link → (prismatic) → finger_link1/2
            # root = gripper_base_link, 与 gripper_link 原点重合 (fixed joint origin=0)
            from hand_track import gripper_config as _gc
            gripper_urdf_path = _gc.generate_gripper_urdf("right")
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
            print(f"\n  单夹爪已加载 (无机械臂, 运动学控制)")
            print(f"  夹爪关节(2): {self.gripper_joint_names}")
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
            print(f"\n  机器人已加载 (固定基座, 运动学控制)")
            print(f"  臂关节({len(self.arm_joint_indices)}): {self.arm_joint_names}")
            print(f"  夹爪关节(2): {self.gripper_joint_names}")

        print(f"  DOF={len(self.all_active_joint_indices)}, decimation={DECIMATION}")

    def reset_simulation(self):
        """重置仿真环境: 断开当前连接, 重新创建全新环境

        每次渲染视频前调用, 确保不会复用上一轮的物理状态。
        """
        # 断开当前连接 (如果已连接)
        if self.physics_client is not None:
            try:
                p.disconnect()
            except Exception:
                pass
            self.physics_client = None

        # 重新创建仿真环境
        self._init_simulation()

        # 清空 GLB 物体列表 (需要重新 load_glb_objects)
        self.obj_ids = []
        self.obj_info = {}

        # 重置基座参数
        self.arm_base_pos = None
        self.arm_base_q = None

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
                elif "wrist_joint" in joint_name:
                    # 手腕关节 (单夹爪模式): 归类为夹爪关节的一部分
                    self.gripper_joint_indices.append(i)
                    self.gripper_joint_names.append(joint_name)
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
        """执行一个控制步: 运动学控制 + 物理仿真

        控制策略 (对齐 GalaxeaManipSim 的效果):
          GalaxeaManipSim 使用 PD驱动 + 重力补偿 (Kp=1000, Kd=200), 机械臂非常刚硬,
          近似运动学控制。在 PyBullet 中, R1 URDF 连杆惯性极小, computed torque control
          不稳定 (IK 误差可达 25°+)。因此采用运动学控制 (resetJointState) 直接设置
          目标关节角, 达到与 GalaxeaManipSim 同样的效果: 机械臂精确跟踪目标。

          运动学控制的优势:
          - 机械臂精确跟踪 IK 目标 (零跟踪误差)
          - 机械臂仍能通过碰撞推动 GLB 物体 (碰撞检测正常工作)
          - GLB 物体由物理引擎驱动 (重力、碰撞)

          与 GalaxeaManipSim 的等价关系:
          - GalaxeaManipSim: PD驱动+重力补偿 → 机械臂极刚硬 → 近似运动学控制
          - PyBullet: resetJointState → 精确运动学控制 → 同样效果
        """
        for i, idx in enumerate(self.all_active_joint_indices):
            target = target_qpos[i] if i < len(target_qpos) else 0.0
            p.resetJointState(self.robot_id, idx, target, targetVelocity=0.0)

        # 物理仿真步进 (让 GLB 物体受重力/碰撞影响)
        for _ in range(DECIMATION):
            p.stepSimulation()

    def load_glb_objects(self, glb_path, transform_params_path):
        """加载 GLB 场景物体 (与 02 的变换链一致)

        变换链: GLB (RAS y-down) → HaWoR render world (y-up) → SAPIEN (z-up)
        含 Z 翻转 (FLIP_Z_FOR_PHYSICS=True, 对齐 04)
        物体分类:
          - 大型扁平几何体 (桌面/地板): mass=0 → static
          - 小物体: mass=volume*density → dynamic
        桌面支撑: 基于 dynamic 物体的最低 Z (不是所有物体的最低 Z),
                  因为 GLB 场景已包含桌面/地板 (static), 它们的 Z 很低
        """
        params = np.load(transform_params_path)
        s_inv = float(params['s_inv'])
        R_inv = params['R_inv']
        t_inv = params['t_inv']

        scene = trimesh.load(glb_path)
        self.obj_ids = []
        self.obj_info = {}

        # 第一遍: 变换顶点 + 分类 static/dynamic
        all_centroids = []
        geom_data = []
        for geom_name, geometry in scene.geometry.items():
            vertices = np.array(geometry.vertices)
            if len(vertices) == 0:
                continue
            # 与 04 一致的变换链 (含 Z 翻转)
            vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
            vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
            if FLIP_Z_FOR_PHYSICS:
                vertices_sapien[:, 2] = -vertices_sapien[:, 2]
            centroid = vertices_sapien.mean(axis=0)
            all_centroids.append(centroid)

            # 分类: 大型扁平几何体 (桌面/地板) vs 小物体
            bbox_size = vertices_sapien.max(axis=0) - vertices_sapien.min(axis=0)
            volume = abs(bbox_size[0] * bbox_size[1] * bbox_size[2])
            max_extent = max(bbox_size)
            flatness = bbox_size[2] / max(max(bbox_size[0], bbox_size[1]), 1e-6)
            is_static = (volume > 0.01 and flatness < 0.3) or max_extent > 0.8

            geom_data.append((geom_name, geometry, vertices_sapien, centroid, is_static))

        if not all_centroids:
            print("  No geometry found in GLB")
            return

        # 桌面高度: 基于 dynamic (小) 物体的最低 Z, 而不是所有物体的最低 Z
        # 因为 GLB 场景已包含桌面/地板 (static), 它们的 Z 很低 (可能 < 0)
        # 如果用所有物体的 min_z, 桌面会设在地板高度, 小物体悬空 30cm+
        dynamic_verts_z = [v[:, 2].min() for _, _, v, _, is_static in geom_data if not is_static]
        if dynamic_verts_z:
            min_z = min(dynamic_verts_z)
            print(f"  Dynamic objects min Z: {min_z:.4f}m (基于 {len(dynamic_verts_z)} 个小物体)")
        else:
            # 没有动态物体, 回退到所有物体的最低 Z
            all_verts_z = [v[:, 2].min() for _, _, v, _, _ in geom_data]
            min_z = min(all_verts_z)
            print(f"  All objects min Z: {min_z:.4f}m (无动态物体, 回退)")

        ground_z = min_z - 0.002  # 桌面顶部 = 小物体最低点下方 2mm

        # 计算 GLB 物体 XY 范围和质心 (用于桌面尺寸)
        all_verts_xy = np.array([[v[0], v[1]] for _, _, v, _, _ in geom_data for v in [v.min(axis=0), v.max(axis=0)]])
        all_centroids_arr = np.array(all_centroids)
        center_xy = all_centroids_arr[:, :2].mean(axis=0)
        extent_xy = all_verts_xy.max(axis=0) - all_verts_xy.min(axis=0)
        print(f"  Object Z range: dynamic_min={min_z:.4f}m, ground/table at Z={ground_z:.4f}m")
        print(f"  Object XY center: {center_xy}, extent: {extent_xy}")

        # 移除默认地面, 添加自定义高度地面 (仅碰撞, 不可见)
        # 地面不可见: 相机可能在地面下方 (Z < ground_z), 可见地面会挡住视线
        p.removeBody(self.plane_id)
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0, 0, ground_z])
        # 设置地面完全透明 (alpha=0), 仅保留碰撞功能
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.5, 0.5, 0.5, 0.0])

        # 添加可见桌面支撑 (与 04 的 support_table 一致): 木色 box, 紧贴物体最低点
        # 桌面尺寸: 基于物体 XY 范围 + 边距, 防止物体从边缘掉落
        TABLE_HALF_THICKNESS = 0.015  # 桌面半厚度=1.5cm (总厚度3cm, 与04一致)
        TABLE_XY_MARGIN = 0.15  # XY边距=15cm (与04一致)
        table_half_x = max(0.15, extent_xy[0] / 2 + TABLE_XY_MARGIN)
        table_half_y = max(0.15, extent_xy[1] / 2 + TABLE_XY_MARGIN)
        table_col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[table_half_x, table_half_y, TABLE_HALF_THICKNESS])
        table_vis_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[table_half_x, table_half_y, TABLE_HALF_THICKNESS],
            rgbaColor=[0.55, 0.45, 0.35, 1.0]  # 木色
        )
        # 桌面顶部 = ground_z, 桌面中心 Z = ground_z - TABLE_HALF_THICKNESS
        self.table_id = p.createMultiBody(
            baseMass=0,  # kinematic (固定)
            baseCollisionShapeIndex=table_col_shape,
            baseVisualShapeIndex=table_vis_shape,
            basePosition=[float(center_xy[0]), float(center_xy[1]), ground_z - TABLE_HALF_THICKNESS]
        )
        p.changeDynamics(self.table_id, -1, lateralFriction=1.0, restitution=0.1)
        print(f"  桌面: center=({center_xy[0]:.3f}, {center_xy[1]:.3f}), half_size=({table_half_x:.3f}, {table_half_y:.3f}), top_z={ground_z:.4f}")

        # 第二遍: 创建物体 (使用第一遍预计算的 is_static)
        for geom_name, geometry, vertices_sapien, centroid, is_static in geom_data:
            bbox_size = vertices_sapien.max(axis=0) - vertices_sapien.min(axis=0)
            volume = abs(bbox_size[0] * bbox_size[1] * bbox_size[2])

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

        # 保存 view/proj 矩阵供 _draw_axes_overlay 使用
        self._last_view_matrix = view_matrix
        self._last_proj_matrix = proj_matrix

        # 光照设置 (与 02 SAPIEN setup_scene 一致):
        # 02 使用: 主光 [2.5,2.5,2.5] from [1,-1,-1], 补光 [1.2,1.2,1.2], 环境光 0.5
        # PyBullet: lightDirection 是从光源指向场景的方向 (与 SAPIEN 相反)
        # SAPIEN [1,-1,-1] (光来的方向) → PyBullet [-1,1,1] (光去的方向)
        _, _, px, _, _ = p.getCameraImage(
            self.cam_width, self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            lightDirection=[-0.5, 0.5, 0.7],
            lightColor=[0.9, 0.9, 0.9],
            lightDistance=2.0,
            lightAmbientCoeff=0.5,
            lightDiffuseCoeff=0.7,
            lightSpecularCoeff=0.3,
            shadow=1,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # PyBullet 返回 RGB, 转BGR for OpenCV
        rgb = np.array(px, dtype=np.uint8).reshape(self.cam_height, self.cam_width, 4)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        return bgr

    def _draw_axes_overlay(self, bgr, cam_pos, cam_R, origin=None, axis_len=0.05):
        """在渲染帧上叠加世界坐标轴和重力方向

        Args:
            bgr: (H, W, 3) BGR 图像
            cam_pos: (3,) 相机位置
            cam_R: (3, 3) 相机旋转矩阵 (SAPIEN 约定)
            origin: (3,) 坐标轴原点 (None=桌面中心)
            axis_len: 坐标轴长度 (米)
        """
        if origin is None:
            # 默认原点: 桌面中心上方 1cm
            if self.table_id is not None:
                table_pos, _ = p.getBasePositionAndOrientation(self.table_id)
                origin = np.array([table_pos[0], table_pos[1], table_pos[2] + 0.015])
            else:
                origin = np.array([0.0, 0.0, 0.01])

        # 投影函数: 3D 世界坐标 → 2D 图像坐标
        # 使用场景中心作为 target 计算 view matrix (比 eye+forward 更稳定)
        forward = cam_R[:, 0]
        up = cam_R[:, 2]
        eye = np.array(cam_pos, dtype=np.float64)
        # target: 沿 forward 方向远处的点 (近似无穷远投影)
        target = eye + forward * 10.0

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        aspect = self.cam_width / self.cam_height
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.cam_fov_deg,
            aspect=aspect,
            nearVal=0.01,
            farVal=100.0,
        )

        # PyBullet 返回行优先 OpenGL 矩阵, 转置为列优先
        vm = np.array(view_matrix).reshape(4, 4).T
        pm = np.array(proj_matrix).reshape(4, 4).T
        mvp = pm @ vm

        def project(pt3d):
            """3D 点投影到 2D 图像坐标"""
            p4 = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
            clip = mvp @ p4
            if abs(clip[3]) < 1e-8:
                return None
            ndc = clip[:3] / clip[3]
            # NDC [-1,1] → 像素坐标
            x = int((ndc[0] + 1) * 0.5 * self.cam_width)
            y = int((1 - ndc[1]) * 0.5 * self.cam_height)
            if 0 <= x < self.cam_width and 0 <= y < self.cam_height:
                return (x, y)
            return None

        # 画坐标轴: X=红, Y=绿, Z=蓝
        axes = [
            (np.array([axis_len, 0, 0]), (0, 0, 255), "X"),  # 红
            (np.array([0, axis_len, 0]), (0, 255, 0), "Y"),  # 绿
            (np.array([0, 0, axis_len]), (255, 0, 0), "Z"),  # 蓝
        ]
        origin_2d = project(origin)
        if origin_2d is not None:
            for axis_dir, color, label in axes:
                end_3d = origin + axis_dir
                end_2d = project(end_3d)
                if end_2d is not None:
                    cv2.arrowedLine(bgr, origin_2d, end_2d, color, 2, tipLength=0.15)
                    cv2.putText(bgr, label, (end_2d[0] + 5, end_2d[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # 画重力方向: 黄色向下箭头
            gravity_end = origin + np.array([0, 0, -axis_len])
            gravity_2d = project(gravity_end)
            if gravity_2d is not None:
                cv2.arrowedLine(bgr, origin_2d, gravity_2d, (0, 255, 255), 2, tipLength=0.15)
                cv2.putText(bgr, "g", (gravity_2d[0] + 5, gravity_2d[1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # 右上角信息: 桌面高度 + 相机位置
        info_y = 60
        if self.table_id is not None:
            table_pos, _ = p.getBasePositionAndOrientation(self.table_id)
            cv2.putText(bgr, f"Table Z={table_pos[2]:.4f}m", (15, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            info_y += 18
        cv2.putText(bgr, f"Cam Z={cam_pos[2]:.4f}m", (15, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        info_y += 18
        cv2.putText(bgr, f"Origin Z={origin[2]:.4f}m", (15, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return bgr

    def _draw_mano_points_overlay(self, bgr, cam_pos, cam_R, mano_points):
        """在渲染帧上叠加 MANO 3 个特征点 (用于验证夹爪跟随)

        Args:
            bgr: (H, W, 3) BGR 图像
            cam_pos: (3,) 相机位置
            cam_R: (3, 3) 相机旋转矩阵 (SAPIEN 约定)
            mano_points: dict with "wrist", "finger1", "finger2" (each (3,))
        """
        # 投影函数 (与 _draw_axes_overlay 相同)
        forward = cam_R[:, 0]
        up = cam_R[:, 2]
        eye = np.array(cam_pos, dtype=np.float64)
        target = eye + forward * 10.0

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        aspect = self.cam_width / self.cam_height
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.cam_fov_deg, aspect=aspect, nearVal=0.01, farVal=100.0,
        )
        vm = np.array(view_matrix).reshape(4, 4).T
        pm = np.array(proj_matrix).reshape(4, 4).T
        mvp = pm @ vm

        def project(pt3d):
            p4 = np.array([pt3d[0], pt3d[1], pt3d[2], 1.0])
            clip = mvp @ p4
            if abs(clip[3]) < 1e-8:
                return None
            ndc = clip[:3] / clip[3]
            x = int((ndc[0] + 1) * 0.5 * self.cam_width)
            y = int((1 - ndc[1]) * 0.5 * self.cam_height)
            if 0 <= x < self.cam_width and 0 <= y < self.cam_height:
                return (x, y)
            return None

        # 画 3 个特征点: 手腕=红, 拇指尖=绿, 食指尖=蓝
        points = [
            (mano_points["wrist"], (0, 0, 255), "W"),
            (mano_points["finger1"], (0, 255, 0), "F1"),
            (mano_points["finger2"], (255, 0, 0), "F2"),
        ]
        for pt3d, color, label in points:
            pt2d = project(pt3d)
            if pt2d is not None:
                cv2.circle(bgr, pt2d, 6, color, -1)
                cv2.circle(bgr, pt2d, 6, (255, 255, 255), 1)
                cv2.putText(bgr, label, (pt2d[0] + 8, pt2d[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 画手腕→指尖中点连线 (黄色虚线, 显示指向方向)
        mid = (mano_points["finger1"] + mano_points["finger2"]) / 2
        w2d = project(mano_points["wrist"])
        m2d = project(mid)
        if w2d is not None and m2d is not None:
            cv2.line(bgr, w2d, m2d, (0, 255, 255), 1)

        # 画两指尖连线 (白色, 显示开合方向)
        f1_2d = project(mano_points["finger1"])
        f2_2d = project(mano_points["finger2"])
        if f1_2d is not None and f2_2d is not None:
            cv2.line(bgr, f1_2d, f2_2d, (255, 255, 255), 1)

    def render_video(self, trajectory_path, output_path, R_c2w_all, t_c2w_all,
                     num_frames=-1, base_pos=None, base_q=None, view="fpv",
                     glb_path=None, transform_params_path=None):
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
            glb_path: GLB 场景文件路径 (reset_simulation 后重新加载物体)
            transform_params_path: transform_params.npz 路径 (与 glb_path 配合使用)
        """
        print(f"\n{'='*60}")
        print("Rendering Video (对齐 02 run_robot_tracking)")
        print(f"{'='*60}")

        # 重置仿真环境, 确保每次渲染从全新状态开始
        self.reset_simulation()

        # 重新加载 GLB 物体 (reset_simulation 会清空物体列表)
        if glb_path and transform_params_path:
            self.load_glb_objects(glb_path, transform_params_path)

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
                # 俯视: forward=朝下(-Z), up=Y+(或X+)
                # SAPIEN 相机约定: X=forward, Y=left, Z=up
                forward = np.array([0, 0, -1.0])
                up = np.array([0, 1.0, 0])
                left = np.cross(up, forward)
                left = left / np.linalg.norm(left)
                up = np.cross(forward, left)
                fixed_cam_R = np.column_stack([forward, left, up])
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
        print(f"\n  渲染循环 (运动学控制, 对齐 GalaxeaManipSim 效果) ...")
        rendered_frames = 0
        for frame_idx in range(num_frames):
            # 获取当前帧的关节角 (02 的 IK 解, 运动学控制直接设置)
            qpos = trajectory[frame_idx]

            # 运动学控制 + 物理仿真
            self.step(qpos)

            # 计算相机位姿
            if view == "fpv":
                # fpv: 跟随 HaWoR 相机轨迹 (与 02 一致)
                cam_pos, cam_R = hawor_cam_to_sapien_pose(R_c2w_all[frame_idx], t_c2w_all[frame_idx])
            else:
                # 固定视角
                cam_pos, cam_R = fixed_cam_pos, fixed_cam_R

            # 渲染帧
            bgr = self.render_frame(cam_pos, cam_R)

            # 叠加坐标轴和重力方向
            self._draw_axes_overlay(bgr, cam_pos, cam_R)

            # 添加帧信息
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            label = f"Frame {frame_idx+1}/{num_frames} | Physics (kinematic)"
            cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            writer.write(bgr)
            rendered_frames += 1

            if (frame_idx + 1) % 30 == 0:
                ee_pos, _ = self.get_ee_pose()
                ee_str = f", EE={np.array2string(ee_pos, precision=3)}" if ee_pos is not None else ""
                print(f"  Frame {frame_idx+1}/{num_frames}{ee_str}")

        writer.release()
        print(f"\n  Video saved: {output_path}")

        # 检查文件大小
        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"  File size: {file_size:.1f} MB")

        # 验证: 渲染帧数需 >= 目标帧数的 50%
        min_required = max(1, num_frames // 2)
        if rendered_frames < min_required:
            print(f"  ⚠ 渲染不完整: {rendered_frames}/{num_frames} 帧 ({rendered_frames/num_frames*100:.0f}%), 不足 50%")
            return False
        print(f"  ✓ 渲染完成: {rendered_frames}/{num_frames} 帧 ({rendered_frames/num_frames*100:.0f}%)")
        return True

    def render_single_gripper_video(self, hawor_dir, transform_params_path, glb_path, output_path,
                                     R_c2w_all, t_c2w_all, hand_idx=1, num_frames=-1,
                                     img_focal=None, view="fpv", use_dex_retarget=False):
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

        # 重置仿真环境, 确保每次渲染从全新状态开始
        self.reset_simulation()

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
        prefix = "right" if hand_idx == 1 else "left"
        mano_layer = MANOLayer(mano_side, betas_mean)

        # 导入 hand_track 夹爪配置 (仅用于 Dex Retargeting 初始化)
        from hand_track import gripper_config as _gc

        # URDF 使用固定 37mm (与 _init_simulation 加载的一致, 不动态重新加载)
        # 解析法 (_compute_analytical_gripper_pose) 也用 37mm, 匹配指尖中点 (不是手腕)
        FINGER_ORIGIN_X_FIXED = 0.03689

        # 可选: 初始化 Dex Retargeting (使用固定 37mm URDF, 3 点目标: 2 指尖+手腕方向)
        dex_retargeting = None
        dex_fixed_qpos = None
        ref_indices_dex = [4, 8, 0]  # 3 点: 2 指尖 (位置精确) + 手腕 (方向约束, 中轴线上)
        if use_dex_retarget:
            print(f"\n  初始化 Dex Retargeting (hand_track gripper-only 配置, 37mm, 3 点目标) ...")
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from dex_retargeting.constants import HandType

            dex_retargeting, ref_indices_dex, _ = _gc.init_gripper_retargeting(
                prefix, FINGER_ORIGIN_X_FIXED, PROJECT_ROOT,
            )
            dex_fixed_qpos = np.zeros(len(dex_retargeting.optimizer.idx_pin2fixed), dtype=np.float32)
            dex_robot = dex_retargeting.optimizer.robot
            dex_joint_names = dex_robot.dof_joint_names
            print(f"  ✓ Dex Retargeting 就绪 (8 DOF = 6 dummy + 2 finger, 3 点目标: 2 指尖+手腕)")

        # [2/6] 加载 GLB 物体 (带桌面支撑) — 如果尚未加载
        if not self.obj_ids:
            print(f"\n[2/6] 加载 GLB 物体 ...")
            self.load_glb_objects(glb_path, transform_params_path)
        else:
            print(f"\n[2/6] GLB 物体已加载 ({len(self.obj_ids)} 个), 跳过")

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
        print(f"\n[6/6] 单夹爪渲染循环 (运动学控制, 对齐 GalaxeaManipSim) ...")

        # 保存上一帧的目标 (用于无效帧保持目标)
        # hand_track URDF: 只有 2 个 prismatic finger joints (无 wrist_joint)
        joint_targets = [GRIPPER_INIT_OPEN, GRIPPER_INIT_OPEN]
        root_pos = np.array([0, 0, 0])
        pb_quat = [0, 0, 0, 1]
        rendered_frames = 0
        last_mano_points = None  # 用于绘制 MANO 3 个特征点

        for frame_idx in range(num_frames):
            if hawor_data["pred_valid"][frame_idx]:
                # 计算 MANO 关节
                _, j = _compute_mano_joints(mano_layer, hawor_data["pred_rot"][frame_idx],
                                            hawor_data["pred_hand_pose"][frame_idx],
                                            hawor_data["pred_trans"][frame_idx])
                joints_sapien = (RXWORLD_TO_SAPIEN @ j.T).T

                mano_wrist = joints_sapien[0, :3]
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]

                # 解析计算夹爪位姿和手指关节 (37mm URDF, 匹配指尖中点, 不对齐手腕)
                root_pos, root_R, joint1, joint2 = _compute_analytical_gripper_pose(
                    mano_wrist, mano_finger1, mano_finger2, prefix=prefix,
                )

                # 可选: Dex Retargeting 微调
                if dex_retargeting is not None:
                    g_quat = pr.quaternion_from_matrix(root_R)
                    dex_retargeting.warm_start(
                        root_pos, g_quat,
                        hand_type=HandType.right if prefix == "right" else HandType.left,
                        is_mano_convention=False,
                    )
                    # 用解析 finger joints 初始化
                    finger_j1_name = f"{prefix}_gripper_finger_joint1"
                    finger_j2_name = f"{prefix}_gripper_finger_joint2"
                    for num, jname in enumerate(dex_retargeting.optimizer.target_joint_names):
                        if jname == finger_j1_name:
                            dex_retargeting.last_qpos[num] = joint1
                        elif jname == finger_j2_name:
                            dex_retargeting.last_qpos[num] = joint2
                    ref_value = joints_sapien[ref_indices_dex, :].astype(np.float32)
                    sapien_qpos = dex_retargeting.retarget(ref_value, fixed_qpos=dex_fixed_qpos)

                    # 从 Dex qpos 提取 gripper_link 位姿
                    dex_robot.compute_forward_kinematics(sapien_qpos)
                    gripper_link_idx = dex_robot.get_link_index(f"{prefix}_gripper_link")
                    T_gripper = dex_robot.get_link_pose(gripper_link_idx)
                    root_pos = T_gripper[:3, 3]
                    root_R = T_gripper[:3, :3]

                    # 从 Dex qpos 提取手指关节值
                    qpos_dict = {name: float(sapien_qpos[i]) for i, name in enumerate(dex_joint_names)}
                    joint1 = qpos_dict[finger_j1_name]
                    joint2 = qpos_dict[finger_j2_name]

                # 设置夹爪 root 位姿 (gripper_base_link 运动学, 与 gripper_link 原点重合)
                root_quat_wxyz = pr.quaternion_from_matrix(root_R)
                pb_quat = [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]

                # 更新目标关节值: [finger1, finger2] (hand_track URDF 无 wrist_joint)
                joint_targets = [float(joint1), float(joint2)]

                # 保存 MANO 3 个特征点 (用于渲染叠加)
                last_mano_points = {
                    "wrist": mano_wrist.copy(),
                    "finger1": mano_finger1.copy(),
                    "finger2": mano_finger2.copy(),
                }

            # 物理仿真步进 (运动学控制: 重置 root 位姿 + 关节, GLB 物体物理交互)
            p.resetBasePositionAndOrientation(
                self.robot_id, root_pos.tolist(), pb_quat
            )
            for i, idx in enumerate(self.all_active_joint_indices):
                target = joint_targets[i] if i < len(joint_targets) else 0.0
                p.resetJointState(self.robot_id, idx, target, targetVelocity=0.0)

            for _ in range(DECIMATION):
                p.stepSimulation()

            # 运动学控制无跟踪误差
            tracking_err = 0.0

            # 计算相机位姿
            if view == "fpv":
                cam_pos, cam_R = hawor_cam_to_sapien_pose(R_c2w_all[frame_idx], t_c2w_all[frame_idx])
            else:
                cam_pos, cam_R = fixed_cam_pos, fixed_cam_R

            # 渲染帧
            bgr = self.render_frame(cam_pos, cam_R)

            # 叠加坐标轴和重力方向
            self._draw_axes_overlay(bgr, cam_pos, cam_R)

            # 叠加 MANO 3 个特征点 (绿色球: 手腕/拇指尖/食指尖)
            if last_mano_points is not None:
                self._draw_mano_points_overlay(bgr, cam_pos, cam_R, last_mano_points)

            # 添加帧信息 (含 PD 跟踪误差)
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            label = f"Frame {frame_idx+1}/{num_frames} | Single-Gripper Physics (kinematic)"
            cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            writer.write(bgr)
            rendered_frames += 1

            if (frame_idx + 1) % 30 == 0:
                print(f"  Frame {frame_idx+1}/{num_frames}")

        writer.release()
        print(f"\n  Video saved: {output_path}")

        file_size = Path(output_path).stat().st_size / 1024 / 1024
        print(f"  File size: {file_size:.1f} MB")

        # 验证: 渲染帧数需 >= 目标帧数的 50%
        min_required = max(1, num_frames // 2)
        if rendered_frames < min_required:
            print(f"  ⚠ 渲染不完整: {rendered_frames}/{num_frames} 帧 ({rendered_frames/num_frames*100:.0f}%), 不足 50%")
            return False
        print(f"  ✓ 渲染完成: {rendered_frames}/{num_frames} 帧 ({rendered_frames/num_frames*100:.0f}%)")
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


def _gripper_finger_tips_world(root_pos, root_R, joint1, joint2):
    """根据解析法结果计算夹爪指尖世界位置"""
    finger1_in_gripper = _FINGER1_ORIGIN + _FINGER1_AXIS * joint1
    finger2_in_gripper = _FINGER2_ORIGIN + _FINGER2_AXIS * joint2
    tip1_world = root_pos + root_R @ finger1_in_gripper
    tip2_world = root_pos + root_R @ finger2_in_gripper
    return tip1_world, tip2_world


def test_gripper_tracking(hawor_dir, hand_idx=-1, num_frames=-1, use_dex_retarget=False):
    """测试单夹爪是否能跟随 MANO 手腕/指尖位姿

    对比方法:
      1. 解析法 (_compute_analytical_gripper_pose): 加权 SVD + 匹配指尖中点
      2. (可选) Dex Retargeting 优化器: 与 02_render_scene.py 相同的配置

    输出指标:
      - wrist_pos_err: 解析法 gripper root 位置与 MANO 手腕位置误差 (mm)
      - finger_tip_err: 解析法指尖位置与 MANO 目标指尖位置误差 (mm)
      - finger_dist_err: 解析法手指间距与 MANO 指尖间距误差 (mm)
      - dex_finger_err: Dex Retargeting 指尖位置误差 (mm) (如果启用)

    Args:
        hawor_dir: HaWoR 输出目录
        hand_idx: 手部索引 (0=左, 1=右, -1=自动)
        num_frames: 测试帧数 (-1=全部)
        use_dex_retarget: 是否同时运行 Dex Retargeting 对比

    Returns:
        dict: 误差统计结果
    """
    print(f"\n{'='*60}")
    print("单夹爪跟踪精度测试")
    print(f"{'='*60}")

    # 加载 HaWoR 数据 (不依赖 sapien)
    print(f"\n[1/3] 加载 HaWoR 数据 ...")
    hawor_data = _load_hawor_data(hawor_dir, hand_idx=hand_idx)
    n_total = len(hawor_data["pred_trans"])
    if num_frames < 0 or num_frames > n_total:
        num_frames = n_total
    print(f"  帧数: {num_frames}/{n_total}")

    # MANO layer
    sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))
    from mano_layer import MANOLayer
    betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
    mano_side = "left" if hand_idx == 0 else "right"
    prefix = "right" if hand_idx == 1 else "left"
    mano_layer = MANOLayer(mano_side, betas_mean)

    # URDF 使用固定 37mm (不动态缩放), 解析法也用 37mm 匹配指尖中点
    FINGER_ORIGIN_X_FIXED = 0.03689

    # 导入 hand_track 夹爪配置 (gripper-only URDF + Dex retargeting)
    from hand_track import gripper_config as _gc

    # 可选: 初始化 Dex Retargeting (37mm, 3 点目标: 2 指尖+手腕方向)
    dex_retargeting = None
    fixed_qpos = None
    ref_indices_dex = [4, 8, 0]  # 3 点: 2 指尖 (位置精确) + 手腕 (方向约束, 中轴线上)
    if use_dex_retarget:
        print(f"\n[2/3] 初始化 Dex Retargeting (hand_track gripper-only 配置, 37mm, 3 点目标) ...")
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from dex_retargeting.constants import HandType

        dex_retargeting, ref_indices_dex, _ = _gc.init_gripper_retargeting(
            prefix, FINGER_ORIGIN_X_FIXED, PROJECT_ROOT,
        )
        fixed_qpos = np.zeros(len(dex_retargeting.optimizer.idx_pin2fixed), dtype=np.float32)
        print(f"  非目标关节数: {len(fixed_qpos)}, 目标关节数: {len(dex_retargeting.optimizer.idx_pin2target)}")
        print(f"  目标关节: {dex_retargeting.optimizer.target_joint_names}")
        print(f"  ✓ Dex Retargeting 就绪")
        dex_robot = dex_retargeting.optimizer.robot
        dex_joint_names = dex_robot.dof_joint_names

    # [3/3] 逐帧测试
    print(f"\n[3/3] 逐帧计算误差 ...")

    errs_wrist = []
    errs_finger_tip = []
    errs_finger_dist = []
    errs_dex_wrist = []
    errs_dex_tip = []

    ref_indices = [4, 8]  # 拇指尖, 食指尖 (解析法)

    for frame_idx in range(num_frames):
        if not hawor_data["pred_valid"][frame_idx]:
            continue

        # MANO FK
        _, j = _compute_mano_joints(mano_layer, hawor_data["pred_rot"][frame_idx],
                                     hawor_data["pred_hand_pose"][frame_idx],
                                     hawor_data["pred_trans"][frame_idx])
        joints_sapien = (RXWORLD_TO_SAPIEN @ j.T).T

        mano_wrist = joints_sapien[0, :3]
        mano_finger1 = joints_sapien[ref_indices[0], :3]
        mano_finger2 = joints_sapien[ref_indices[1], :3]

        # 解析法 (37mm URDF, 匹配指尖中点, 不对齐手腕)
        root_pos, root_R, joint1, joint2 = _compute_analytical_gripper_pose(
            mano_wrist, mano_finger1, mano_finger2, prefix=prefix,
        )

        # 解析法误差
        wrist_err = np.linalg.norm(root_pos - mano_wrist) * 1000  # mm (gripper_link 不在手腕处, 误差大是正常的)
        fg = _gc.FINGER_GEOM_ARRAYS[prefix]
        f1_origin = fg["finger1_origin"].copy()  # 37mm (与 URDF 一致)
        f2_origin = fg["finger2_origin"].copy()
        tip1_world = root_pos + root_R @ (f1_origin + fg["finger1_axis"] * joint1)
        tip2_world = root_pos + root_R @ (f2_origin + fg["finger2_axis"] * joint2)
        tip_err = (np.linalg.norm(tip1_world - mano_finger1) +
                   np.linalg.norm(tip2_world - mano_finger2)) / 2 * 1000  # mm
        mano_finger_dist = np.linalg.norm(mano_finger2 - mano_finger1)
        gripper_finger_dist = _gc.FINGER_BASE_DIST + joint1 + joint2
        dist_err = abs(gripper_finger_dist - mano_finger_dist) * 1000  # mm

        errs_wrist.append(wrist_err)
        errs_finger_tip.append(tip_err)
        errs_finger_dist.append(dist_err)

        # Dex Retargeting
        if dex_retargeting is not None:
            g_quat = pr.quaternion_from_matrix(root_R)
            dex_retargeting.warm_start(
                root_pos, g_quat,
                hand_type=HandType.right if prefix == "right" else HandType.left,
                is_mano_convention=False,
            )
            # 用解析 finger joints 初始化优化器 last_qpos
            finger_j1_name = f"{prefix}_gripper_finger_joint1"
            finger_j2_name = f"{prefix}_gripper_finger_joint2"
            for num, jname in enumerate(dex_retargeting.optimizer.target_joint_names):
                if jname == finger_j1_name:
                    dex_retargeting.last_qpos[num] = joint1
                elif jname == finger_j2_name:
                    dex_retargeting.last_qpos[num] = joint2

            ref_value = joints_sapien[ref_indices_dex, :].astype(np.float32)
            sapien_qpos = dex_retargeting.retarget(ref_value, fixed_qpos=fixed_qpos)

            # 用 Pinocchio FK 从 Dex qpos 得到 gripper_link 位姿
            dex_robot.compute_forward_kinematics(sapien_qpos)
            gripper_link_idx = dex_robot.get_link_index(f"{prefix}_gripper_link")
            T_gripper = dex_robot.get_link_pose(gripper_link_idx)
            dex_root_pos = T_gripper[:3, 3]
            dex_root_R = T_gripper[:3, :3]

            # 从 Dex qpos 中提取手指关节值
            qpos_dict = {name: float(sapien_qpos[i]) for i, name in enumerate(dex_joint_names)}
            dex_joint1 = qpos_dict[finger_j1_name]
            dex_joint2 = qpos_dict[finger_j2_name]

            # Dex 指尖位置
            dex_tip1 = dex_root_pos + dex_root_R @ (f1_origin + fg["finger1_axis"] * dex_joint1)
            dex_tip2 = dex_root_pos + dex_root_R @ (f2_origin + fg["finger2_axis"] * dex_joint2)
            dex_tip_err = (np.linalg.norm(dex_tip1 - mano_finger1) +
                           np.linalg.norm(dex_tip2 - mano_finger2)) / 2 * 1000
            dex_wrist_err = np.linalg.norm(dex_root_pos - mano_wrist) * 1000
            errs_dex_tip.append(dex_tip_err)
            errs_dex_wrist.append(dex_wrist_err)

        if (frame_idx + 1) % 30 == 0:
            print(f"  Frame {frame_idx+1}/{num_frames}: wrist_err={np.mean(errs_wrist[-30:]):.2f}mm, "
                  f"tip_err={np.mean(errs_finger_tip[-30:]):.2f}mm, dist_err={np.mean(errs_finger_dist[-30:]):.2f}mm")

    # 统计结果
    result = {
        "num_frames": len(errs_wrist),
        "wrist_pos_err_mm": {
            "mean": float(np.mean(errs_wrist)) if errs_wrist else 0.0,
            "max": float(np.max(errs_wrist)) if errs_wrist else 0.0,
            "std": float(np.std(errs_wrist)) if errs_wrist else 0.0,
        },
        "finger_tip_err_mm": {
            "mean": float(np.mean(errs_finger_tip)) if errs_finger_tip else 0.0,
            "max": float(np.max(errs_finger_tip)) if errs_finger_tip else 0.0,
            "std": float(np.std(errs_finger_tip)) if errs_finger_tip else 0.0,
        },
        "finger_dist_err_mm": {
            "mean": float(np.mean(errs_finger_dist)) if errs_finger_dist else 0.0,
            "max": float(np.max(errs_finger_dist)) if errs_finger_dist else 0.0,
            "std": float(np.std(errs_finger_dist)) if errs_finger_dist else 0.0,
        },
    }

    print(f"\n{'='*60}")
    print("解析法误差统计")
    print(f"{'='*60}")
    print(f"  有效帧数: {result['num_frames']}")
    print(f"  手腕位置误差: mean={result['wrist_pos_err_mm']['mean']:.2f}mm, "
          f"max={result['wrist_pos_err_mm']['max']:.2f}mm, std={result['wrist_pos_err_mm']['std']:.2f}mm "
          f"(注: 解析法优化指尖中点, 不直接跟踪手腕)")
    print(f"  指尖位置误差: mean={result['finger_tip_err_mm']['mean']:.2f}mm, "
          f"max={result['finger_tip_err_mm']['max']:.2f}mm, std={result['finger_tip_err_mm']['std']:.2f}mm")
    print(f"  手指间距误差: mean={result['finger_dist_err_mm']['mean']:.2f}mm, "
          f"max={result['finger_dist_err_mm']['max']:.2f}mm, std={result['finger_dist_err_mm']['std']:.2f}mm")

    if errs_dex_tip:
        result["dex_wrist_pos_err_mm"] = {
            "mean": float(np.mean(errs_dex_wrist)),
            "max": float(np.max(errs_dex_wrist)),
            "std": float(np.std(errs_dex_wrist)),
        }
        result["dex_finger_tip_err_mm"] = {
            "mean": float(np.mean(errs_dex_tip)),
            "max": float(np.max(errs_dex_tip)),
            "std": float(np.std(errs_dex_tip)),
        }
        print(f"\nDex Retargeting 误差统计")
        print(f"  手腕位置误差: mean={result['dex_wrist_pos_err_mm']['mean']:.2f}mm, "
              f"max={result['dex_wrist_pos_err_mm']['max']:.2f}mm, "
              f"std={result['dex_wrist_pos_err_mm']['std']:.2f}mm")
        print(f"  指尖位置误差: mean={result['dex_finger_tip_err_mm']['mean']:.2f}mm, "
              f"max={result['dex_finger_tip_err_mm']['max']:.2f}mm, "
              f"std={result['dex_finger_tip_err_mm']['std']:.2f}mm")

    return result


def main():
    parser = argparse.ArgumentParser(description="PyBullet Physics Pipeline (对齐 02 run_robot_tracking)")
    parser.add_argument("--test", action="store_true", help="Run basic tests")
    parser.add_argument("--render-video", action="store_true", help="Render video from trajectory")
    parser.add_argument("--gui", action="store_true", help="Use GUI mode")
    parser.add_argument("--hawor-dir", type=str, default=DEFAULT_HAWOR_DIR,
                        help=f"HaWoR 输出目录 (默认: {DEFAULT_HAWOR_DIR})")
    parser.add_argument("--ras-dir", type=str, default=DEFAULT_RAS_DIR,
                        help=f"RAS 输出目录 (默认: {DEFAULT_RAS_DIR})")
    parser.add_argument("--transform-params", type=str, default=None,
                        help=f"transform_params.npz 路径 (默认: 从 hawor-dir/ras-dir 自动推导)")
    parser.add_argument("--trajectory", type=str, default=None,
                        help=f"qpos 轨迹文件 (默认: 从 hawor-dir/ras-dir 自动推导)")
    parser.add_argument("--output", type=str, default=None,
                        help=f"输出视频路径 (默认: 根据参数自动生成)")
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
    parser.add_argument("--test-gripper-tracking", action="store_true",
                        help="测试单夹爪跟随 MANO 手腕/指尖位姿的精度 (解析法 vs Dex Retargeting)")
    parser.add_argument("--use-dex-retarget", action="store_true",
                        help="在单夹爪模式中使用 Dex Retargeting 优化器 (解析法 warm-start + Dex 微调, 适用于 test-gripper-tracking 和 --single-gripper 渲染)")
    parser.add_argument("--use-02-trajectory", action="store_true",
                        help="强制使用 02 的轨迹 (基座 0.35m), 会自动调整基座高度为 0.35m。默认优先使用 04 的轨迹 (基座 0.70m)")
    args = parser.parse_args()

    # 如果使用 02 轨迹, 自动调整基座高度为 0.35m (匹配 02 的 IK)
    if args.use_02_trajectory:
        global COMFORTABLE_REACH, COMFORT_TARGET_IN_BASE
        COMFORTABLE_REACH = 0.35
        COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
        print(f"  --use-02-trajectory: 基座高度调整为 {COMFORTABLE_REACH}m (匹配 02 轨迹)")

    # 自动生成输出文件名 (包含关键参数, 避免不同命令的输出互相覆盖)
    if args.output is None and not args.test_gripper_tracking:
        name_parts = ["pybullet"]
        if args.single_gripper:
            name_parts.append("gripper")
        if args.use_dex_retarget:
            name_parts.append("dex")
        if args.use_02_trajectory:
            name_parts.append("02traj")
        if args.view != "fpv":
            name_parts.append(args.view)
        if args.hand_idx >= 0:
            name_parts.append(f"h{args.hand_idx}")
        args.output = str(SCRIPT_DIR / "output" / ("_".join(name_parts) + ".mp4"))

    # 自动推导 transform-params 和 trajectory (从 hawor-dir / ras-dir)
    hawor_bn = Path(args.hawor_dir).name
    ras_bn = Path(args.ras_dir).name
    session_name = f"{hawor_bn}_{ras_bn}"
    output_dir = COMBINATION_DIR / "output" / session_name

    if args.transform_params is None:
        # 1. 先查找已有文件
        candidates = [
            output_dir / "alignment" / "transform_params.npz",
            COMBINATION_DIR / "output" / "alignment" / "transform_params.npz",
        ]
        for c in candidates:
            if c.exists():
                args.transform_params = str(c)
                print(f"  自动推导 transform-params: {args.transform_params}")
                break

        # 2. 找不到则调用 01_align_scene.py 的函数生成
        if args.transform_params is None:
            print(f"  未找到 transform_params.npz, 自动调用 01_align_scene.py 生成 ...")
            try:
                mod01 = _load_module_01()
                # 查找 hawor reconstruction npz
                hawor_rec_dir = Path(args.hawor_dir) / "reconstruction"
                hawor_npz = None
                if hawor_rec_dir.exists():
                    for f in hawor_rec_dir.glob("hawor_results_*.npz"):
                        hawor_npz = str(f)
                        break
                if hawor_npz is None:
                    print(f"✗ 未找到 HaWoR reconstruction npz: {hawor_rec_dir}/hawor_results_*.npz")
                    return False
                alignment_dir = str(output_dir / "alignment")
                args.transform_params = mod01.compute_and_save_transform_params(
                    ras_output=str(args.ras_dir),
                    hawor_reconstruction=hawor_npz,
                    output_dir=alignment_dir,
                )
                print(f"  ✓ 自动生成 transform-params: {args.transform_params}")
            except Exception as e:
                print(f"✗ 自动生成 transform-params 失败: {e}")
                print(f"  请用 --transform-params 手动指定, 或先运行 01_align_scene.py")
                return False

    if args.trajectory is None:
        # 1. 先查找 04 生成的轨迹 (基于 0.70m 基座, 与 PyBullet 当前基座高度匹配)
        candidates_04 = [
            output_dir / "tracking" / "physics_sim_physics_tracking.npy",
            COMBINATION_DIR / "output" / "tracking" / "physics_sim_physics_tracking.npy",
        ]
        for c in candidates_04:
            if c.exists():
                args.trajectory = str(c)
                print(f"  自动推导 trajectory (04 生成, 基座 0.70m): {args.trajectory}")
                break

        # 2. 找不到 04 轨迹, 查找 02 的轨迹 (基于 0.35m 基座, 需 --use-02-trajectory)
        if args.trajectory is None:
            candidates_02 = [
                output_dir / "tracking" / "hand_object_robot_tracking.npy",
            ]
            for c in candidates_02:
                if c.exists():
                    if args.use_02_trajectory:
                        args.trajectory = str(c)
                        print(f"  自动推导 trajectory (02 生成, 基座 0.35m, --use-02-trajectory): {args.trajectory}")
                        break
                    else:
                        print(f"  ⚠ 发现 02 轨迹 (基座 0.35m): {c}")
                        print(f"     当前基座高度 0.70m, 直接使用会导致机械臂位置偏移 0.35m")
                        print(f"     选项: 1) 先运行 04 生成 0.70m 轨迹  2) 加 --use-02-trajectory 强制使用 (基座会自动改为 0.35m)")

        # 3. 找不到则提示
        if args.trajectory is None:
            print(f"⚠ 未找到 trajectory 文件")
            print(f"  推荐: 先运行 04_physics_simulation.py 生成 0.70m 基座的轨迹")
            print(f"    python 04_physics_simulation.py --hawor-dir {args.hawor_dir} --ras-dir {args.ras_dir} --fast-collision")
            print(f"  或用 --trajectory 手动指定, 或使用 --single-gripper 模式")
            # 单夹爪模式不需要轨迹文件, 不退出
            if not args.single_gripper:
                return False

    run_all = not (args.test or args.render_video or args.test_gripper_tracking)
    results = {}

    if run_all or args.test:
        pipeline = PyBulletPipeline(gui=args.gui, cam_width=args.width, cam_height=args.height)
        results["hold_position"] = pipeline.test_hold_position()
        pipeline.cleanup()

    if args.test_gripper_tracking:
        results["gripper_tracking"] = test_gripper_tracking(
            args.hawor_dir, hand_idx=args.hand_idx, num_frames=args.num_frames,
            use_dex_retarget=args.use_dex_retarget
        )

    if run_all or args.render_video:
        # 检查必要文件
        hawor_dir = Path(args.hawor_dir)
        ras_dir = Path(args.ras_dir)
        glb_path = ras_dir / "final_scene.glb"
        transform_params = Path(args.transform_params)
        trajectory_path = Path(args.trajectory) if args.trajectory else None

        if not glb_path.exists():
            print(f"✗ GLB 文件不存在: {glb_path}")
            return False
        if not transform_params.exists():
            print(f"✗ 变换参数不存在: {transform_params}")
            return False
        # 单夹爪模式不需要轨迹文件 (直接用 MANO 手腕位姿驱动)
        if not args.single_gripper and (trajectory_path is None or not trajectory_path.exists()):
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
                view=args.view,
                use_dex_retarget=args.use_dex_retarget,
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
                view=args.view,
                glb_path=str(glb_path),
                transform_params_path=str(transform_params)
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
