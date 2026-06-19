#!/usr/bin/env python3
"""
render_gripper_only.py — 只渲染夹爪URDF (不加载手臂)

与 render_auto.py 类似, 但只加载夹爪部分的 URDF:
  - gripper_base_link (固定根, 无mesh)
  - gripper_link (夹爪本体 mesh)
  - gripper_finger_link1/2 (两个手指 mesh, prismatic joint)

夹爪位姿直接从 retargeting FK 获取, 不需要 IK, 不需要手臂底座。
排除手臂底座不确定性的干扰, 只看夹爪跟踪效果。

用法:
    python hand_track/render_gripper_only.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result
"""

import os
import subprocess
import sys
import time
import logging
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import sapien
import sapien.render
import torch
from pytransform3d import rotations as pr
from tqdm import trange

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
COMBINATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = COMBINATION_DIR.parent.parent  # dex-retargeting/
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))
sys.path.insert(0, str(SCRIPT_DIR))

from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from mano_layer import MANOLayer

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"

R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

HAWOR_FOCAL_DEFAULT = 600.0

# ── 平滑参数 (与 02_render_scene.py 一致) ──
WARMUP_FRAMES = 30          # smoothstep 过渡帧数
LP_ALPHA_POS = 0.6          # EMA 位置平滑系数 (优化器模式)
LP_ALPHA_ORI = 0.6          # EMA 朝向平滑系数 (优化器模式)
LP_ALPHA_ANALYTICAL = 0.9   # EMA 平滑系数 (解析模式, MANO 数据本身平滑, 只需轻微平滑)
GRIPPER_INIT_OPEN = 0.04    # 夹爪初始开合量 (两个手指都是正值, 在 [0, 0.05] 范围内)


class EmaTargetSmoother:
    """EMA 目标平滑器 (从 02_render_scene.py 复制)

    对位置和朝向(四元数)做指数移动平均, 减少抖动。
    alpha 越大越跟随, 越小越平滑。
    """

    def __init__(self, pos_alpha=LP_ALPHA_POS, ori_alpha=LP_ALPHA_ORI):
        self.pos_alpha = pos_alpha
        self.ori_alpha = ori_alpha
        self.pos = None
        self.ori_quat = None

    def smooth(self, pos, ori_quat):
        if self.pos is None:
            self.pos = pos.copy()
            self.ori_quat = ori_quat.copy()
            return self.pos.copy(), self.ori_quat.copy()
        self.pos = self.pos + self.pos_alpha * (pos - self.pos)
        self.ori_quat = self.ori_quat + self.ori_alpha * (ori_quat - self.ori_quat)
        norm = np.linalg.norm(self.ori_quat)
        if norm > 1e-8:
            self.ori_quat /= norm
        if self.ori_quat[0] < 0:
            self.ori_quat = -self.ori_quat
        return self.pos.copy(), self.ori_quat.copy()

    def reset(self):
        self.pos = None
        self.ori_quat = None


class PositionEmaSmoother:
    """多点位 EMA 平滑器 (用于解析模式: 平滑 MANO 输入位置)

    对多个 3D 点位同时做 EMA 平滑, 保持各点之间的几何关系一致。
    用于解析模式: 先平滑 MANO 指尖/手腕位置, 再计算解析位姿,
    这样 root pose 和手指关节都从同一组平滑后的输入导出, 保持一致性。
    """
    def __init__(self, alpha=LP_ALPHA_POS):
        self.alpha = alpha
        self.positions = None  # (N, 3) array

    def smooth(self, positions):
        """positions: (N, 3) array, 返回平滑后的 (N, 3) array"""
        if self.positions is None:
            self.positions = positions.copy()
        else:
            self.positions = self.positions + self.alpha * (positions - self.positions)
        return self.positions.copy()

    def reset(self):
        self.positions = None


# ── 机器人夹爪几何常数 (从 URDF 提取) ──
# 两个手指闭合时的距离 (joint1=joint2=0)
_FINGER_BASE_DIST = 0.026906  # = abs(0.013453 - (-0.013453))

# prefix 相关的手指几何 (numpy 数组, 用于解析计算)
# 与 _GRIPPER_JOINT_GEOM 一致, 左右手 joint1/joint2 互换
_FINGER_GEOM_ARRAYS = {
    "left": {
        "finger1_origin": np.array([0.03689, 0.013453, 0.00012067]),
        "finger1_axis": np.array([0.0, 1.0, 0.0]),
        "finger2_origin": np.array([0.03689, -0.013453, -0.00012053]),
        "finger2_axis": np.array([0.0, -1.0, 0.0]),
    },
    "right": {
        "finger1_origin": np.array([0.03689, -0.013453, -0.00012053]),
        "finger1_axis": np.array([0.0, -1.0, 0.0]),
        "finger2_origin": np.array([0.03689, 0.013453, 0.00012067]),
        "finger2_axis": np.array([0.0, 1.0, 0.0]),
    },
}


def _compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2, prefix="right"):
    """从 MANO 指尖向量解析计算夹爪 gripper_link 位姿和手指关节值

    机器人夹爪几何 (prefix 相关):
      finger1 = gripper_pos + R @ (finger1_origin + finger1_axis * joint1)
      finger2 = gripper_pos + R @ (finger2_origin + finger2_axis * joint2)

    约束:
      finger1 = mano_finger1  (精确匹配)
      finger2 = mano_finger2  (距离匹配, 方向来自朝向)

    解法:
      1. 从 MANO 指尖向量确定 gripper 朝向 R
         - Y轴: finger1→finger2 方向 (对应机器人 finger 分离方向)
         - X轴: wrist→finger_mid 方向 (对应机器人指尖前向)
         - Z轴: X × Y
      2. 从指尖距离确定 joint1+joint2
      3. gripper_pos = mano_finger1 - R @ (finger1_origin + finger1_axis * joint1)
         (确保 finger1 精确匹配)

    Returns:
        gripper_pos: (3,) gripper_link 位置
        gripper_R: (3,3) gripper_link 旋转矩阵
        joint1, joint2: 手指关节值
    """
    fg = _FINGER_GEOM_ARRAYS[prefix]

    # 1. 计算 gripper 朝向
    v_finger = mano_finger2 - mano_finger1
    finger_dist = np.linalg.norm(v_finger)
    if finger_dist < 1e-6:
        y_axis = np.array([0, 1, 0], dtype=np.float64)
    else:
        # 关键: y_axis 方向需要与机器人 finger2-finger1 的 Y 分量符号一致
        # 右手: finger2_origin - finger1_origin = (0, +0.026906, 0) → y_sign=+1
        # 左手: finger2_origin - finger1_origin = (0, -0.026906, 0) → y_sign=-1
        # 这样 R @ (finger2_origin - finger1_origin) 与 (mano_finger2 - mano_finger1) 同向
        finger_diff_robot = fg["finger2_origin"] - fg["finger1_origin"]
        y_sign = np.sign(finger_diff_robot[1]) if abs(finger_diff_robot[1]) > 1e-6 else 1.0
        y_axis = y_sign * v_finger / finger_dist

    finger_mid = (mano_finger1 + mano_finger2) / 2
    v_wrist = finger_mid - mano_wrist
    wrist_dist = np.linalg.norm(v_wrist)
    if wrist_dist < 1e-6:
        x_axis = np.array([1, 0, 0], dtype=np.float64)
    else:
        x_axis = v_wrist / wrist_dist

    # Gram-Schmidt 正交化
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

    gripper_R = np.column_stack([x_axis, y_axis, z_axis])

    # 2. 计算手指关节值
    # robot finger_dist = _FINGER_BASE_DIST + joint1 + joint2
    required_open_sum = finger_dist - _FINGER_BASE_DIST
    joint1 = max(0.0, min(0.05, required_open_sum / 2))
    joint2 = max(0.0, min(0.05, required_open_sum / 2))

    # 3. 计算 gripper_pos (匹配 fingertip1, 确保精确跟踪)
    gripper_pos = mano_finger1 - gripper_R @ (fg["finger1_origin"] + fg["finger1_axis"] * joint1)

    return gripper_pos, gripper_R, joint1, joint2


# 夹爪 URDF 模板 (只有 gripper 部分)
_GRIPPER_URDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
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
    <origin xyz="{joint1_origin}" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link1"/>
    <axis xyz="{joint1_axis}"/>
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
    <origin xyz="{joint2_origin}" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link2"/>
    <axis xyz="{joint2_axis}"/>
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


# 夹爪+手臂末端 URDF 模板 (gripper + arm_link4/5/6)
# arm_base_link 代表 arm_link3 的位置, 包含 arm_joint4/5/6 + gripper
# 比纯夹爪更生动, 同时排除手臂底座不确定性
_GRIPPER_WITH_ARM_URDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<robot name="r1_gripper_arm_{prefix}">
  <link name="{prefix}_arm_base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.01"/>
      <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/>
    </inertial>
  </link>
  <joint name="{prefix}_arm_joint4" type="revolute">
    <origin xyz="0.02735 -0.069767 0" rpy="0 0 0"/>
    <parent link="{prefix}_arm_base_link"/>
    <child link="{prefix}_arm_link4"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.8798" upper="2.8798" effort="7" velocity="25.133"/>
  </joint>
  <link name="{prefix}_arm_link4">
    <inertial>
      <origin xyz="0.24285 -0.0023763 -3.4603E-07" rpy="0 0 0"/>
      <mass value="0.694"/>
      <inertia ixx="8.45E-05" ixy="8.2612E-07" ixz="2.2124E-09" iyy="0.00010174" iyz="5.3644E-09" izz="9.7044E-05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_arm_link4.STL"/>
      </geometry>
      <material name="">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_arm_link4.STL"/>
      </geometry>
    </collision>
  </link>
  <joint name="{prefix}_arm_joint5" type="revolute">
    <origin xyz="0.2463 0.00050106 0" rpy="0 0 0"/>
    <parent link="{prefix}_arm_link4"/>
    <child link="{prefix}_arm_link5"/>
    <axis xyz="0 -1 0"/>
    <limit lower="-1.6581" upper="1.6581" effort="7" velocity="25.133"/>
  </joint>
  <link name="{prefix}_arm_link5">
    <inertial>
      <origin xyz="0.054309 -0.0041807 -3.8613E-06" rpy="0 0 0"/>
      <mass value="0.417"/>
      <inertia ixx="8.4E-05" ixy="-1.6234E-05" ixz="-7.4239E-08" iyy="9.8498E-05" iyz="-1.3874E-08" izz="0.00011333"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_arm_link5.STL"/>
      </geometry>
      <material name="">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_arm_link5.STL"/>
      </geometry>
    </collision>
  </link>
  <joint name="{prefix}_arm_joint6" type="revolute">
    <origin xyz="0.058249 -0.00049975 0" rpy="0 0 0"/>
    <parent link="{prefix}_arm_link5"/>
    <child link="{prefix}_arm_link6"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.8798" upper="2.8798" effort="7" velocity="25.133"/>
  </joint>
  <link name="{prefix}_arm_link6">
    <inertial>
      <origin xyz="0.028138 1.2134E-07 5.405E-08" rpy="0 0 0"/>
      <mass value="0.037"/>
      <inertia ixx="3.5662E-06" ixy="6.6514E-12" ixz="2.9628E-12" iyy="2.0238E-06" iyz="-4.0687E-12" izz="2.0238E-06"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_arm_link6.STL"/>
      </geometry>
      <material name="">
        <color rgba="0.823529411764706 0.823529411764706 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_dir}/{prefix}_arm_link6.STL"/>
      </geometry>
    </collision>
  </link>
  <joint name="{prefix}_gripper_joint" type="fixed">
    <origin xyz="0.1039 0 0" rpy="0 0 0"/>
    <parent link="{prefix}_arm_link6"/>
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
    <origin xyz="{joint1_origin}" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link1"/>
    <axis xyz="{joint1_axis}"/>
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
    <origin xyz="{joint2_origin}" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link2"/>
    <axis xyz="{joint2_axis}"/>
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


# 夹爪手指关节几何 (从 robot.urdf 提取, 左右手 joint1/joint2 互换)
_GRIPPER_JOINT_GEOM = {
    "left": {
        "joint1_origin": "0.03689 0.013453 0.00012067",
        "joint1_axis": "0 1 0",
        "joint2_origin": "0.03689 -0.013453 -0.00012053",
        "joint2_axis": "0 -1 0",
    },
    "right": {
        "joint1_origin": "0.03689 -0.013453 -0.00012053",
        "joint1_axis": "0 -1 0",
        "joint2_origin": "0.03689 0.013453 0.00012067",
        "joint2_axis": "0 1 0",
    },
}


def _generate_gripper_urdf(prefix="right"):
    """生成只包含夹爪的 URDF 文件"""
    xml = _GRIPPER_URDF_TEMPLATE.format(
        prefix=prefix,
        mesh_dir=str(R1_MESH_DIR),
        **_GRIPPER_JOINT_GEOM[prefix],
    )
    temp_dir = tempfile.mkdtemp(prefix=f"r1_gripper_{prefix}-")
    temp_path = f"{temp_dir}/r1_gripper_{prefix}.urdf"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def _generate_gripper_with_arm_urdf(prefix="right"):
    """生成包含夹爪+arm_link4/5/6的 URDF 文件"""
    xml = _GRIPPER_WITH_ARM_URDF_TEMPLATE.format(
        prefix=prefix,
        mesh_dir=str(R1_MESH_DIR),
        **_GRIPPER_JOINT_GEOM[prefix],
    )
    temp_dir = tempfile.mkdtemp(prefix=f"r1_gripper_arm_{prefix}-")
    temp_path = f"{temp_dir}/r1_gripper_arm_{prefix}.urdf"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def _compute_gripper_offset_in_root(robot, prefix):
    """计算 gripper_link 相对于 root link 的位置 offset (当所有 arm_joint=0 时)

    由于 fix_root_link=True, root link 的位姿为 identity。
    gripper_link 的位姿就是它相对于 root 的 offset。

    用于 gripper_arm 模式: 设置 root pose 时需要补偿这个 offset,
    使得 gripper_link 的实际位置等于 retargeting FK 给出的位置。

    Returns:
        offset_pos: (3,) gripper_link 相对于 root 的位置
        offset_R: (3,3) gripper_link 相对于 root 的旋转
    """
    target_name = f"{prefix}_gripper_link"
    for link in robot.get_links():
        if link.get_name() == target_name:
            pose = link.get_entity_pose()
            offset_pos = np.array(pose.p)
            offset_R = pr.matrix_from_quaternion(np.array(pose.q))
            return offset_pos, offset_R
    return np.zeros(3), np.eye(3)


# ─── 从 common.py 导入共享函数 ──────────────────────────────────────────────

from common import (
    detect_hands, load_hawor_data, load_hawor_c2w, setup_scene,
    load_glb_transformed, compute_mano_joints, _render_to_sapien,
    _render_keypoints, hawor_cam_to_sapien_pose, make_look_at_camera,
    _compute_wrist_positions_sapien, _get_gripper_pose_from_retargeting,
)

# 重新定义 _ensure_transform_params (从 render_auto.py 复制)
def _ensure_transform_params(ras_dir, hawor_dir, output_dir, logger):
    tp_path = os.path.join(output_dir, "transform_params.npz")
    if os.path.exists(tp_path):
        logger.info(f"  transform_params 已存在: {tp_path}")
        return tp_path
    logger.info(f"  transform_params 不存在, 自动运行 01_align_scene.py ...")
    hawor_recon_dir = os.path.join(hawor_dir, "reconstruction")
    if not os.path.isdir(hawor_recon_dir):
        logger.error(f"  hawor reconstruction 目录不存在: {hawor_recon_dir}")
        return None
    recon_npz = None
    for f in os.listdir(hawor_recon_dir):
        if f.startswith("hawor_results_") and f.endswith(".npz"):
            recon_npz = os.path.join(hawor_recon_dir, f)
            break
    if recon_npz is None:
        logger.error(f"  未找到 hawor_results_*.npz: {hawor_recon_dir}")
        return None
    align_script = str(COMBINATION_DIR / "01_align_scene.py")
    cmd = [sys.executable, align_script, "--ras_output", ras_dir,
           "--hawor_reconstruction", recon_npz, "--output_dir", output_dir]
    logger.info(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  01_align_scene.py 失败:\n{result.stderr[-500:]}")
        return None
    if os.path.exists(tp_path):
        logger.info(f"  ✓ transform_params 已生成: {tp_path}")
        return tp_path
    logger.error("  01_align_scene.py 运行完成但未生成 transform_params.npz")
    return None


def _combine_videos_side_by_side(left_video, right_video, output, fps, crf, logger):
    """将左右视频并排合成一个视频"""
    cap_l = cv2.VideoCapture(left_video)
    cap_r = cv2.VideoCapture(right_video)
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_out = max(h_l, h_r)
    w_out = w_l + w_r
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w_out, h_out))
    frame_idx = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l and not ret_r:
            break
        if not ret_l:
            frame_l = np.zeros((h_l, w_l, 3), dtype=np.uint8)
        if not ret_r:
            frame_r = np.zeros((h_r, w_r, 3), dtype=np.uint8)
        if frame_l.shape[0] < h_out:
            pad = np.zeros((h_out - frame_l.shape[0], frame_l.shape[1], 3), dtype=np.uint8)
            frame_l = np.vstack([frame_l, pad])
        if frame_r.shape[0] < h_out:
            pad = np.zeros((h_out - frame_r.shape[0], frame_r.shape[1], 3), dtype=np.uint8)
            frame_r = np.vstack([frame_r, pad])
        combined = np.hstack([frame_l, frame_r])
        cv2.rectangle(combined, (0, 0), (w_l, 40), (0, 0, 0), -1)
        cv2.rectangle(combined, (w_l, 0), (w_out, 40), (0, 0, 0), -1)
        cv2.putText(combined, "Left Gripper", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Right Gripper", (w_l + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(combined)
        frame_idx += 1
    cap_l.release()
    cap_r.release()
    writer.release()
    # ffmpeg 重编码
    tmp_path = str(output).replace(".mp4", "_tmp.mp4")
    if os.path.exists(str(output)):
        os.rename(str(output), tmp_path)
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
                   "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
                   "-movflags", "+faststart", str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
                os.remove(tmp_path)
            else:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, output)
        except Exception:
            if os.path.exists(tmp_path):
                os.rename(tmp_path, output)
    logger.info(f"  ✓ 双夹爪合成视频: {output} ({frame_idx} 帧)")


def render_gripper_only_video(hawor_dir, ras_dir, transform_params_path, output,
                              hand_idx=1, fps=30, cam_width=1920, cam_height=1080,
                              view="fpv", crf=18, start_frame=0, num_frames=-1,
                              with_arm=False, smooth=1, viewer=False, verify=False,
                              analytical=True, logger=None):
    """渲染只有夹爪URDF的视频 (不加载手臂)

    只加载 gripper_link + finger_link1/2 的 URDF, 夹爪位姿直接从
    retargeting FK 获取。不需要 IK, 不需要手臂底座。

    Args:
        与 render_robot_video 相同
        smooth: 0=不平滑, 1=EMA平滑 (位置+朝向)
        viewer: True=使用 SAPIEN Viewer 实时循环播放 (不保存视频)
        verify: True=计算并输出指尖位置/手腕位姿误差
        analytical: True=解析模式 (从MANO指尖向量直接计算root位姿, 指尖误差≈0)
                    False=优化器模式 (NLopt SLSQP, 可能有局部最优问题)
    """
    import logging
    if logger is None:
        logger = logging.getLogger("gripper_only")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    prefix = "left" if hand_idx == 0 else "right"
    mode_label = "左手" if hand_idx == 0 else "右手"
    logger.info(f"夹爪URDF渲染: {mode_label}")

    # ── 加载数据 ──
    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    total_frames = hawor_data["pred_trans"].shape[0]
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame
    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)

    focal = hawor_data.get("img_focal")
    if focal is None or focal <= 0:
        focal = HAWOR_FOCAL_DEFAULT
    focal_render = focal * cam_width / 1280.0
    cam_fov = 2 * np.arctan(cam_height / 2.0 / focal_render)

    betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
    mano_layer = MANOLayer(prefix, betas_mean)

    # ── Retargeting (获取 ref_indices 和 FK) ──
    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    hand_type = HandType.left if hand_idx == 0 else HandType.right
    config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
    # 用3个目标点 (与 02_render_scene.py 的 run_robot_tracking 一致):
    #   finger_link1 → MANO joint4 (指尖1)
    #   finger_link2 → MANO joint8 (指尖2)
    #   gripper_link  → MANO joint0 (手腕)
    # 3点约束 (9 > 8 DOF) 完全确定 root pose (6 DOF) + 手指关节 (2 DOF)
    override = dict(
        add_dummy_free_joint=True, normal_delta=1e-5, huber_delta=0.01,
        target_link_names=[f"{prefix}_gripper_finger_link1",
                           f"{prefix}_gripper_finger_link2",
                           f"{prefix}_gripper_link"],
        target_link_human_indices=np.array([4, 8, 0]),
        target_joint_names=[f"{prefix}_gripper_finger_joint1",
                            f"{prefix}_gripper_finger_joint2"],
    )
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    ref_indices = retargeting.optimizer.target_link_human_indices
    fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
    # fixed_qpos 将在 URDF 加载后设置 (需要 init_qpos)

    # ── 创建场景 ──
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")

    # ── 加载夹爪 URDF ──
    if with_arm:
        gripper_urdf_path = _generate_gripper_with_arm_urdf(prefix)
        logger.info(f"  模式: 夹爪+手臂末端 (arm_link4/5/6)")
    else:
        gripper_urdf_path = _generate_gripper_urdf(prefix)
        logger.info(f"  模式: 仅夹爪")
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(gripper_urdf_path)

    joint_names = [j.name for j in robot.get_active_joints()]
    logger.info(f"  夹爪关节: {joint_names}")

    gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
    gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")
    # arm 关节索引 (gripper_arm 模式下需要显式设为 0, 防止物理仿真漂移)
    arm_joint_indices = [i for i, n in enumerate(joint_names) if 'arm_joint' in n]

    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=100000.0, damping=10000.0)

    init_qpos = robot.get_qpos().copy()
    init_qpos[gripper_idx1] = GRIPPER_INIT_OPEN
    init_qpos[gripper_idx2] = GRIPPER_INIT_OPEN
    robot.set_qpos(init_qpos)

    # retargeting → sapien qpos 映射
    retarget2sapien = np.array(
        [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
    ).astype(int)
    # SAPIEN 关节名 → retargeting 关节索引 (用于直接从 retarget_qpos 取手指关节值)
    sapien_name_to_retarget_idx = {
        n: retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names
    }
    # retargeting 索引 → SAPIEN 索引 (用于设置 fixed_qpos)
    sapien2retarget = {}
    for sapien_i, retarget_i in enumerate(retarget2sapien):
        sapien2retarget[retarget_i] = sapien_i
    # 设置 fixed_qpos (与 02_render_scene.py 一致: 从 init_qpos 取值)
    fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
    for i, retarget_idx in enumerate(fixed_retarget_indices):
        if retarget_idx in sapien2retarget:
            fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

    scene.step()
    scene.update_render()

    # 计算 gripper_link 相对于 root 的 offset (用于 gripper_arm 模式补偿)
    # 注意: scene.step() 会导致 arm 关节漂移, 必须显式重置为 0 后再计算 offset
    if with_arm and arm_joint_indices:
        qpos_now = robot.get_qpos().copy()
        for ai in arm_joint_indices:
            qpos_now[ai] = 0.0
        robot.set_qpos(qpos_now)
        scene.update_render()
    gripper_offset_pos, gripper_offset_R = _compute_gripper_offset_in_root(robot, prefix)
    if with_arm:
        logger.info(f"  gripper_link 相对于 root 的 offset: pos={gripper_offset_pos}, R=...")
    else:
        # gripper 模式下 offset 应该为 0 (gripper_base_link 和 gripper_link 之间是 fixed joint, origin 0 0 0)
        gripper_offset_pos = np.zeros(3)
        gripper_offset_R = np.eye(3)

    # ── Warm start retargeting ──
    for probe_idx in range(num_frames):
        g_idx = start_frame + probe_idx
        if not hawor_data["pred_valid"][g_idx]:
            continue
        rot = hawor_data["pred_rot"][g_idx]
        trans = hawor_data["pred_trans"][g_idx]
        hand_pose = hawor_data["pred_hand_pose"][g_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
        wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
        wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
        retargeting.warm_start(
            joints_sapien[0, :3], wrist_quat,
            hand_type=hand_type, is_mano_convention=True,
        )
        logger.info(f"  ✓ Warm start 完成 (帧 {g_idx})")
        break

    # ── 探测首帧有效位姿 (用于 warmup smoothstep) ──
    first_valid_pos = None
    first_valid_quat = None
    for probe_idx in range(num_frames):
        g_idx = start_frame + probe_idx
        if not hawor_data["pred_valid"][g_idx]:
            continue
        rot = hawor_data["pred_rot"][g_idx]
        trans = hawor_data["pred_trans"][g_idx]
        hand_pose = hawor_data["pred_hand_pose"][g_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        if analytical:
            # 解析模式: 直接从 MANO 指尖向量计算 gripper_link 位姿
            mano_wrist = joints_sapien[0, :3]
            mano_finger1 = joints_sapien[ref_indices[0], :3]
            mano_finger2 = joints_sapien[ref_indices[1], :3]
            g_pos, g_R, _, _ = _compute_analytical_gripper_pose(
                mano_wrist, mano_finger1, mano_finger2, prefix)
            # 转换为 root 位姿
            root_R = g_R @ gripper_offset_R.T
            root_pos = g_pos - root_R @ gripper_offset_pos
            first_valid_pos = root_pos.copy()
            first_valid_quat = pr.quaternion_from_matrix(root_R)
        else:
            ref_value = joints_sapien[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            _, gripper_R_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)
            # Hybrid: FK 朝向 + 解析位置 (匹配 fingertip1)
            mano_finger1 = joints_sapien[ref_indices[0], :3]
            mano_finger2 = joints_sapien[ref_indices[1], :3]
            mano_finger_dist = float(np.linalg.norm(mano_finger1 - mano_finger2))
            required_open = max(0.0, mano_finger_dist - _FINGER_BASE_DIST)
            joint1 = min(0.05, required_open / 2)
            root_R = gripper_R_fk @ gripper_offset_R.T
            fg = _FINGER_GEOM_ARRAYS[prefix]
            finger1_in_root = gripper_offset_pos + gripper_offset_R @ (fg["finger1_origin"] + fg["finger1_axis"] * joint1)
            root_pos = mano_finger1 - root_R @ finger1_in_root
            first_valid_pos = root_pos.copy()
            first_valid_quat = pr.quaternion_from_matrix(root_R)
        break

    # ── Warmup smoothstep 过渡 (从初始位姿到首帧有效位姿) ──
    if first_valid_pos is not None:
        init_root_pos = np.zeros(3)
        init_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        for wi in range(WARMUP_FRAMES):
            t = (wi + 1) / WARMUP_FRAMES
            t = t * t * (3 - 2 * t)  # smoothstep
            interp_pos = init_root_pos * (1 - t) + first_valid_pos * t
            interp_quat = init_root_quat * (1 - t) + first_valid_quat * t
            norm = np.linalg.norm(interp_quat)
            if norm > 1e-8:
                interp_quat /= norm
            robot.set_root_pose(sapien.Pose(interp_pos.tolist(), interp_quat.tolist()))
            scene.step()
        logger.info(f"  Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep 过渡)")

    # ── EMA 目标平滑器 ──
    # 解析模式: 平滑 MANO 输入位置 (保持 root pose 和手指关节一致性)
    # 优化器模式: 平滑输出 root pose
    target_smoother = None
    mano_smoother = None
    if smooth == 1:
        if analytical:
            mano_smoother = PositionEmaSmoother(alpha=LP_ALPHA_ANALYTICAL)
            logger.info(f"  平滑模式: MANO 输入位置 EMA (alpha={LP_ALPHA_ANALYTICAL})")
        else:
            target_smoother = EmaTargetSmoother()
            logger.info(f"  平滑模式: root pose EMA (pos_alpha={LP_ALPHA_POS}, ori_alpha={LP_ALPHA_ORI})")
    else:
        logger.info(f"  平滑模式: 不平滑")

    # ── 设置相机/Viewer ──
    sapien_viewer = None
    if viewer:
        from sapien.utils import Viewer
        sapien_viewer = Viewer()
        sapien_viewer.set_scene(scene)
        sapien_viewer.control_window.show_origin_frame = True
        sapien_viewer.control_window.show_grid = False
        camera = None
        logger.info("  模式: SAPIEN Viewer 实时循环 (不保存视频)")
    else:
        camera = scene.add_camera("gripper", cam_width, cam_height, cam_fov, 0.01, 100.0)

    wrist_positions = _compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])
    elif wrist_positions:
        centroid = np.mean(wrist_positions, axis=0)
        if view == "behind":
            cam_pos = centroid + np.array([2.5, 0.0, 1.2])
            cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
        elif view == "front":
            cam_pos = centroid + np.array([-2.5, 0.0, 1.2])
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

    # ── 视频写入器 (仅非 viewer 模式) ──
    writer = None
    if not viewer:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    # ── 验证误差记录 ──
    verify_errors = [] if verify else None
    if verify:
        logger.info("  验证模式: 开启 (计算指尖位置 + 手腕位姿误差)")

    # ── 渲染循环 (viewer 模式循环播放) ──
    animation_loop = True
    while animation_loop:
        if not viewer:
            animation_loop = False  # 非 viewer 模式只跑一遍

        for local_idx in trange(num_frames, desc=f"夹爪URDF-{prefix}", disable=viewer):
            global_idx = start_frame + local_idx

            # 更新相机
            if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                if camera:
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                if sapien_viewer:
                    sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

            if not hawor_data["pred_valid"][global_idx]:
                for node in kp_nodes:
                    internal_scene.remove_node(node)
                kp_nodes.clear()
                scene.step()
                if viewer:
                    sapien_viewer.render()
                else:
                    scene.update_render()
                    camera.take_picture()
                    rgb = camera.get_picture("Color")[..., :3]
                    bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                    h, w = bgr.shape[:2]
                    cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper {prefix}  |  INVALID",
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    writer.write(bgr)
                continue

            rot = hawor_data["pred_rot"][global_idx]
            trans = hawor_data["pred_trans"][global_idx]
            hand_pose = hawor_data["pred_hand_pose"][global_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                for node in kp_nodes:
                    internal_scene.remove_node(node)
                kp_nodes.clear()
                scene.step()
                if viewer:
                    sapien_viewer.render()
                else:
                    scene.update_render()
                    camera.take_picture()
                    rgb = camera.get_picture("Color")[..., :3]
                    bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                    h, w = bgr.shape[:2]
                    cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper {prefix}  |  NaN",
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    writer.write(bgr)
                continue

            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)

            # 关键点标记
            kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012)

            if analytical:
                # 解析模式: 先平滑 MANO 输入位置, 再计算解析位姿
                # (保持 root pose 和手指关节一致性, 指尖误差仅来自输入平滑滞后)
                mano_wrist = joints_sapien[0, :3]
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                if mano_smoother is not None:
                    mano_pts = np.stack([mano_wrist, mano_finger1, mano_finger2])
                    mano_pts = mano_smoother.smooth(mano_pts)
                    mano_wrist, mano_finger1, mano_finger2 = mano_pts[0], mano_pts[1], mano_pts[2]
                g_pos, g_R, joint1, joint2 = _compute_analytical_gripper_pose(
                    mano_wrist, mano_finger1, mano_finger2, prefix)
                # 从 gripper_link 位姿转换为 root 位姿 (gripper_arm 模式需要减去 offset)
                root_R = g_R @ gripper_offset_R.T
                root_pos = g_pos - root_R @ gripper_offset_pos
                root_quat = pr.quaternion_from_matrix(root_R)
                robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))
                qpos = robot.get_qpos().copy()
                # 显式设置 arm 关节为 0 (防止物理仿真漂移)
                for arm_idx in arm_joint_indices:
                    qpos[arm_idx] = 0.0
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                robot.set_qpos(qpos)
            else:
                # Retargeting 优化器模式 (与 02_render_scene.py 的 run_robot_tracking 一致):
                # 1. 用 3 点 retargeting 优化器获取 root pose (6 DOF) + 手指关节 (2 DOF)
                # 2. 从内部机器人 FK 获取 gripper_link 朝向 (优化器的最佳朝向估计)
                # 3. 手指关节用解析值 (从 MANO 指尖距离计算, 确保夹爪开合)
                # 4. root 位置用解析值 (匹配 fingertip1, 确保指尖精确跟踪)
                #    (Hybrid 方案: FK 朝向 + 解析位置 + 解析手指关节)
                ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

                # 夹爪朝向: 用 retargeting FK (与 02_render_scene.py 一致)
                _, gripper_R_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)

                # 手指关节: 用解析值从 MANO 指尖距离计算 (确保夹爪开合)
                # 优化器给的手指关节值常接近 0 (几何不匹配导致), 无法体现开合
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                mano_finger_dist = float(np.linalg.norm(mano_finger1 - mano_finger2))
                required_open = max(0.0, mano_finger_dist - _FINGER_BASE_DIST)
                joint1 = min(0.05, required_open / 2)
                joint2 = min(0.05, required_open / 2)

                # root 朝向: 用 FK 朝向 (补偿 gripper_link 相对于 root 的 offset)
                root_R = gripper_R_fk @ gripper_offset_R.T
                # root 位置: 解析计算, 匹配 fingertip1
                # finger1_pos = root_pos + root_R @ (gripper_offset_pos + gripper_offset_R @ (finger1_origin + finger1_axis * joint1))
                # 令 finger1_pos = mano_finger1, 解出 root_pos
                fg = _FINGER_GEOM_ARRAYS[prefix]
                finger1_in_root = gripper_offset_pos + gripper_offset_R @ (fg["finger1_origin"] + fg["finger1_axis"] * joint1)
                root_pos = mano_finger1 - root_R @ finger1_in_root
                root_quat = pr.quaternion_from_matrix(root_R)

                # EMA 平滑 (对 root pose 做平滑, 减少抖动)
                if target_smoother is not None:
                    root_pos, root_quat = target_smoother.smooth(root_pos, root_quat)

                robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))

                qpos = robot.get_qpos().copy()
                for arm_idx in arm_joint_indices:
                    qpos[arm_idx] = 0.0
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                robot.set_qpos(qpos)

            # 验证: 计算指尖位置和手腕位姿误差
            # 注意: verify 模式下不调用 scene.step(), 避免物理仿真导致 arm 关节漂移
            if verify:
                scene.update_render()
                # 获取 SAPIEN 中 finger_link1/2 的实际位置
                finger1_pos = None
                finger2_pos = None
                gripper_link_pos = None
                gripper_link_R = None
                for link in robot.get_links():
                    lname = link.get_name()
                    if lname == f"{prefix}_gripper_finger_link1":
                        finger1_pos = np.array(link.get_entity_pose().p)
                    elif lname == f"{prefix}_gripper_finger_link2":
                        finger2_pos = np.array(link.get_entity_pose().p)
                    elif lname == f"{prefix}_gripper_link":
                        pose = link.get_entity_pose()
                        gripper_link_pos = np.array(pose.p)
                        gripper_link_R = pr.matrix_from_quaternion(np.array(pose.q))
                # MANO 指尖位置 (ref_indices=[4,8,0], 即 finger1→joint4, finger2→joint8, wrist→joint0)
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                # MANO 手腕位置和朝向
                mano_wrist_pos = joints_sapien[0, :3]
                wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
                wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T

                err = {}
                if finger1_pos is not None:
                    err['finger1_mm'] = float(np.linalg.norm(finger1_pos - mano_finger1) * 1000)
                if finger2_pos is not None:
                    err['finger2_mm'] = float(np.linalg.norm(finger2_pos - mano_finger2) * 1000)
                if gripper_link_pos is not None:
                    err['wrist_pos_mm'] = float(np.linalg.norm(gripper_link_pos - mano_wrist_pos) * 1000)
                    # 朝向误差 (角度, 度)
                    R_diff = gripper_link_R.T @ wrist_R_sapien
                    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                    err['wrist_ori_deg'] = float(np.degrees(angle_rad))
                verify_errors.append(err)
            else:
                scene.step()

            if viewer:
                sapien_viewer.render()
            else:
                scene.update_render()
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper URDF {prefix}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(bgr)

        if viewer:
            logger.info("  动画播放完成, 重新开始... (关闭窗口退出)")
            # 重置 qpos 和平滑器
            init_qpos = robot.get_qpos().copy()
            init_qpos[gripper_idx1] = GRIPPER_INIT_OPEN
            init_qpos[gripper_idx2] = GRIPPER_INIT_OPEN
            robot.set_qpos(init_qpos)
            if target_smoother:
                target_smoother.reset()
            if mano_smoother:
                mano_smoother.reset()

    if not viewer:
        writer.release()

        # ffmpeg 重编码
        final_path = output
        tmp_path = str(output).replace(".mp4", "_tmp.mp4")
        if os.path.exists(str(output)):
            os.rename(str(output), tmp_path)
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                import subprocess as sp
                cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
                       "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
                       "-movflags", "+faststart", str(final_path)]
                result = sp.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    os.remove(tmp_path)
                else:
                    if os.path.exists(tmp_path):
                        os.rename(tmp_path, final_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, final_path)

        logger.info(f"\n✓ 夹爪URDF视频已保存: {final_path}")
    else:
        final_path = None
        logger.info("\n✓ Viewer 模式结束")

    # ── 验证误差报告 ──
    if verify and verify_errors:
        logger.info("\n  === 验证误差报告 ===")
        for key in ['finger1_mm', 'finger2_mm', 'wrist_pos_mm', 'wrist_ori_deg']:
            vals = [e[key] for e in verify_errors if key in e]
            if vals:
                mean_v = np.mean(vals)
                max_v = np.max(vals)
                unit = 'mm' if 'mm' in key else 'deg'
                label = {
                    'finger1_mm': '指尖1位置误差', 'finger2_mm': '指尖2位置误差',
                    'wrist_pos_mm': '手腕位置误差', 'wrist_ori_deg': '手腕朝向误差'
                }[key]
                logger.info(f"  {label}: mean={mean_v:.2f}{unit}, max={max_v:.2f}{unit}")
        # 计算相对误差 (以手腕位置为基准)
        wrist_pos_errors = [e['wrist_pos_mm'] for e in verify_errors if 'wrist_pos_mm' in e]
        if wrist_pos_errors:
            mean_wrist = np.mean(wrist_pos_errors)
            # 手腕运动范围作为基准
            wrist_range = np.ptp([e.get('wrist_pos_mm', 0) for e in verify_errors]) if len(verify_errors) > 1 else 100.0
            rel_err = mean_wrist / max(wrist_range, 1.0) * 100
            logger.info(f"  手腕位置相对误差: {rel_err:.2f}% (基准: 手腕运动范围 {wrist_range:.1f}mm)")

    return final_path


def render_dual_gripper_video(hawor_dir, ras_dir, transform_params_path, output,
                               fps=30, cam_width=1920, cam_height=1080,
                               view="fpv", crf=18, start_frame=0, num_frames=-1,
                               with_arm=False, smooth=1, viewer=False, verify=False,
                               analytical=True, logger=None):
    """在同一场景中渲染左右夹爪URDF (双手, 一个视频)

    在同一个 SAPIEN 场景中加载左右两个夹爪 URDF,
    两个夹爪同时渲染到同一个视频, 各自跟踪对应的手。

    Args:
        smooth: 0=不平滑, 1=EMA平滑 (位置+朝向)
        viewer: True=使用 SAPIEN Viewer 实时循环播放 (不保存视频)
        verify: True=计算并输出指尖位置/手腕位姿误差
        analytical: True=解析模式 (从MANO指尖向量直接计算root位姿, 指尖误差≈0)
                    False=优化器模式 (NLopt SLSQP, 可能有局部最优问题)
    """
    import logging
    if logger is None:
        logger = logging.getLogger("dual_gripper")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    logger.info(f"双夹爪URDF渲染: 同一场景")

    # ── 加载数据 ──
    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)

    # 为左右手分别加载数据和初始化 retargeting
    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    gripper_states = []
    for hi in [0, 1]:
        prefix = "left" if hi == 0 else "right"
        hand_type = HandType.left if hi == 0 else HandType.right

        hawor_data = load_hawor_data(hawor_dir, hand_idx=hi)
        total_frames = hawor_data["pred_trans"].shape[0]

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_layer = MANOLayer(prefix, betas_mean)

        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
        # 用3个目标点 (与 02_render_scene.py 的 run_robot_tracking 一致):
        #   finger_link1 → MANO joint4 (指尖1)
        #   finger_link2 → MANO joint8 (指尖2)
        #   gripper_link  → MANO joint0 (手腕)
        # 3点约束 (9 > 8 DOF) 完全确定 root pose (6 DOF) + 手指关节 (2 DOF)
        override = dict(
            add_dummy_free_joint=True, normal_delta=1e-5, huber_delta=0.01,
            target_link_names=[f"{prefix}_gripper_finger_link1",
                               f"{prefix}_gripper_finger_link2",
                               f"{prefix}_gripper_link"],
            target_link_human_indices=np.array([4, 8, 0]),
            target_joint_names=[f"{prefix}_gripper_finger_joint1",
                                f"{prefix}_gripper_finger_joint2"],
        )
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        ref_indices = retargeting.optimizer.target_link_human_indices
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
        # fixed_qpos 将在 URDF 加载后设置 (需要 init_qpos)

        gripper_states.append({
            "prefix": prefix, "hand_idx": hi, "hand_type": hand_type,
            "hawor_data": hawor_data, "mano_layer": mano_layer,
            "retargeting": retargeting, "ref_indices": ref_indices,
            "fixed_retarget_indices": fixed_retarget_indices,
            "total_frames": total_frames,
        })

    # 统一帧数
    total_frames = min(gs["total_frames"] for gs in gripper_states)
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame

    focal = gripper_states[0]["hawor_data"].get("img_focal")
    if focal is None or focal <= 0:
        focal = HAWOR_FOCAL_DEFAULT
    focal_render = focal * cam_width / 1280.0
    cam_fov = 2 * np.arctan(cam_height / 2.0 / focal_render)

    # ── 创建场景 + GLB ──
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")

    # ── 加载左右夹爪 URDF ──
    for gs in gripper_states:
        prefix = gs["prefix"]
        if with_arm:
            gripper_urdf_path = _generate_gripper_with_arm_urdf(prefix)
        else:
            gripper_urdf_path = _generate_gripper_urdf(prefix)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        robot = loader.load(gripper_urdf_path)

        joint_names = [j.name for j in robot.get_active_joints()]
        gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
        gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

        for joint in robot.get_active_joints():
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        init_qpos = robot.get_qpos().copy()
        init_qpos[gripper_idx1] = GRIPPER_INIT_OPEN
        init_qpos[gripper_idx2] = GRIPPER_INIT_OPEN
        robot.set_qpos(init_qpos)

        retargeting = gs["retargeting"]
        retarget2sapien = np.array(
            [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
        ).astype(int)
        # SAPIEN 关节名 → retargeting 关节索引 (用于直接从 retarget_qpos 取手指关节值)
        sapien_name_to_retarget_idx = {
            n: retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names
        }
        # retargeting 索引 → SAPIEN 索引 (用于设置 fixed_qpos)
        sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(retarget2sapien):
            sapien2retarget[retarget_i] = sapien_i
        # 设置 fixed_qpos (与 02_render_scene.py 一致: 从 init_qpos 取值)
        fixed_retarget_indices = gs["fixed_retarget_indices"]
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in sapien2retarget:
                fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

        gs["robot"] = robot
        gs["gripper_idx1"] = gripper_idx1
        gs["gripper_idx2"] = gripper_idx2
        gs["retarget2sapien"] = retarget2sapien
        gs["sapien_name_to_retarget_idx"] = sapien_name_to_retarget_idx
        gs["fixed_qpos"] = fixed_qpos
        gs["joint_names"] = joint_names
        # arm 关节索引 (gripper_arm 模式下需要显式设为 0, 防止物理仿真漂移)
        gs["arm_joint_indices"] = [i for i, n in enumerate(joint_names) if 'arm_joint' in n]
        logger.info(f"  ✓ {prefix} 夹爪已加载: {joint_names}")

    scene.step()
    scene.update_render()

    # 计算 gripper_link 相对于 root 的 offset (用于 gripper_arm 模式补偿)
    # 注意: scene.step() 会导致 arm 关节漂移, 必须显式重置为 0 后再计算 offset
    for gs in gripper_states:
        prefix = gs["prefix"]
        robot = gs["robot"]
        if with_arm:
            arm_idx = gs.get("arm_joint_indices", [])
            if arm_idx:
                qpos_now = robot.get_qpos().copy()
                for ai in arm_idx:
                    qpos_now[ai] = 0.0
                robot.set_qpos(qpos_now)
                scene.update_render()
            offset_pos, offset_R = _compute_gripper_offset_in_root(robot, prefix)
        else:
            offset_pos = np.zeros(3)
            offset_R = np.eye(3)
        gs["gripper_offset_pos"] = offset_pos
        gs["gripper_offset_R"] = offset_R

    # ── Warm start ──
    for gs in gripper_states:
        hawor_data = gs["hawor_data"]
        mano_layer = gs["mano_layer"]
        retargeting = gs["retargeting"]
        hand_type = gs["hand_type"]
        for probe_idx in range(num_frames):
            g_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][g_idx]:
                continue
            rot = hawor_data["pred_rot"][g_idx]
            trans = hawor_data["pred_trans"][g_idx]
            hand_pose = hawor_data["pred_hand_pose"][g_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                continue
            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)
            wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
            wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
            wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
            retargeting.warm_start(
                joints_sapien[0, :3], wrist_quat,
                hand_type=hand_type, is_mano_convention=True,
            )
            # 手动设置手指关节初始值: 根据 MANO 指尖距离推算
            # robot finger_dist = 0.026906 + joint1 + joint2
            # mano finger_dist = ||joints_sapien[4] - joints_sapien[8]||
            prefix = gs["prefix"]
            mano_finger_dist = float(np.linalg.norm(
                joints_sapien[ref_indices[0], :3] - joints_sapien[ref_indices[1], :3]))
            required_open = max(0.0, min(0.05, (mano_finger_dist - 0.026906) / 2))
            finger_j1_name = f"{prefix}_gripper_finger_joint1"
            finger_j2_name = f"{prefix}_gripper_finger_joint2"
            for num, jname in enumerate(retargeting.optimizer.target_joint_names):
                if jname == finger_j1_name or jname == finger_j2_name:
                    retargeting.last_qpos[num] = required_open
            logger.info(f"  ✓ {gs['prefix']} Warm start 完成 (帧 {g_idx}), finger_init={required_open:.4f} (mano_dist={mano_finger_dist*1000:.1f}mm)")
            break

    # ── 探测首帧有效位姿 + Warmup smoothstep ──
    for gs in gripper_states:
        hawor_data = gs["hawor_data"]
        mano_layer = gs["mano_layer"]
        retargeting = gs["retargeting"]
        prefix = gs["prefix"]
        ref_indices = gs["ref_indices"]
        fixed_qpos = gs["fixed_qpos"]
        gripper_offset_pos = gs["gripper_offset_pos"]
        gripper_offset_R = gs["gripper_offset_R"]
        first_valid_pos = None
        first_valid_quat = None
        for probe_idx in range(num_frames):
            g_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][g_idx]:
                continue
            rot = hawor_data["pred_rot"][g_idx]
            trans = hawor_data["pred_trans"][g_idx]
            hand_pose = hawor_data["pred_hand_pose"][g_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                continue
            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)
            if analytical:
                # 解析模式: 直接从 MANO 指尖向量计算 gripper_link 位姿
                mano_wrist = joints_sapien[0, :3]
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                g_pos, g_R, _, _ = _compute_analytical_gripper_pose(
                    mano_wrist, mano_finger1, mano_finger2, prefix)
                # 转换为 root 位姿
                root_R = g_R @ gripper_offset_R.T
                root_pos = g_pos - root_R @ gripper_offset_pos
                first_valid_pos = root_pos.copy()
                first_valid_quat = pr.quaternion_from_matrix(root_R)
            else:
                ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
                _, gripper_R_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)
                # Hybrid: FK 朝向 + 解析位置 (匹配 fingertip1)
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                mano_finger_dist = float(np.linalg.norm(mano_finger1 - mano_finger2))
                required_open = max(0.0, mano_finger_dist - _FINGER_BASE_DIST)
                joint1 = min(0.05, required_open / 2)
                root_R = gripper_R_fk @ gripper_offset_R.T
                fg = _FINGER_GEOM_ARRAYS[prefix]
                finger1_in_root = gripper_offset_pos + gripper_offset_R @ (fg["finger1_origin"] + fg["finger1_axis"] * joint1)
                root_pos = mano_finger1 - root_R @ finger1_in_root
                first_valid_pos = root_pos.copy()
                first_valid_quat = pr.quaternion_from_matrix(root_R)
            break
        # Warmup smoothstep
        if first_valid_pos is not None:
            init_root_pos = np.zeros(3)
            init_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
            robot = gs["robot"]
            for wi in range(WARMUP_FRAMES):
                t = (wi + 1) / WARMUP_FRAMES
                t = t * t * (3 - 2 * t)
                interp_pos = init_root_pos * (1 - t) + first_valid_pos * t
                interp_quat = init_root_quat * (1 - t) + first_valid_quat * t
                norm = np.linalg.norm(interp_quat)
                if norm > 1e-8:
                    interp_quat /= norm
                robot.set_root_pose(sapien.Pose(interp_pos.tolist(), interp_quat.tolist()))
                scene.step()
            logger.info(f"  ✓ {prefix} Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep)")
        # 平滑器: 解析模式用 MANO 输入位置平滑, 优化器模式用 root pose 平滑
        if smooth == 1:
            if analytical:
                gs["mano_smoother"] = PositionEmaSmoother(alpha=LP_ALPHA_ANALYTICAL)
                gs["target_smoother"] = None
            else:
                gs["mano_smoother"] = None
                gs["target_smoother"] = EmaTargetSmoother()
        else:
            gs["mano_smoother"] = None
            gs["target_smoother"] = None

    if smooth == 1:
        if analytical:
            logger.info(f"  平滑模式: MANO 输入位置 EMA (alpha={LP_ALPHA_ANALYTICAL})")
        else:
            logger.info(f"  平滑模式: root pose EMA (pos_alpha={LP_ALPHA_POS}, ori_alpha={LP_ALPHA_ORI})")
    else:
        logger.info(f"  平滑模式: 不平滑")

    # ── 设置相机/Viewer ──
    sapien_viewer = None
    if viewer:
        from sapien.utils import Viewer
        sapien_viewer = Viewer()
        sapien_viewer.set_scene(scene)
        sapien_viewer.control_window.show_origin_frame = True
        sapien_viewer.control_window.show_grid = False
        camera = None
        logger.info("  模式: SAPIEN Viewer 实时循环 (不保存视频)")
    else:
        camera = scene.add_camera("dual_gripper", cam_width, cam_height, cam_fov, 0.01, 100.0)

    all_wrist_positions = []
    for gs in gripper_states:
        wp = _compute_wrist_positions_sapien(gs["hawor_data"], gs["mano_layer"], start_frame, num_frames)
        all_wrist_positions.extend(wp)

    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])
    elif all_wrist_positions:
        centroid = np.mean(all_wrist_positions, axis=0)
        if view == "behind":
            cam_pos = centroid + np.array([2.5, 0.0, 1.2])
            cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
        elif view == "front":
            cam_pos = centroid + np.array([-2.5, 0.0, 1.2])
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

    # ── 视频写入器 (仅非 viewer 模式) ──
    writer = None
    if not viewer:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    # ── 验证误差记录 ──
    verify_errors = [] if verify else None
    if verify:
        logger.info("  验证模式: 开启 (计算指尖位置 + 手腕位姿误差)")

    # ── 渲染循环 (viewer 模式循环播放) ──
    animation_loop = True
    while animation_loop:
        if not viewer:
            animation_loop = False

        for local_idx in trange(num_frames, desc="双夹爪URDF", disable=viewer):
            global_idx = start_frame + local_idx

            # 更新相机
            if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                if camera:
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                if sapien_viewer:
                    sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

            # 清除上一帧的关键点
            for node in kp_nodes:
                internal_scene.remove_node(node)
            kp_nodes.clear()

            left_valid = False
            right_valid = False

            for gs in gripper_states:
                hawor_data = gs["hawor_data"]
                prefix = gs["prefix"]
                robot = gs["robot"]
                retargeting = gs["retargeting"]
                mano_layer = gs["mano_layer"]
                ref_indices = gs["ref_indices"]
                fixed_qpos = gs["fixed_qpos"]
                retarget2sapien = gs["retarget2sapien"]
                target_smoother = gs.get("target_smoother")

                if not hawor_data["pred_valid"][global_idx]:
                    continue

                rot = hawor_data["pred_rot"][global_idx]
                trans = hawor_data["pred_trans"][global_idx]
                hand_pose = hawor_data["pred_hand_pose"][global_idx]
                if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                    continue

                _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
                joints_sapien = _render_to_sapien(j)

                # 关键点标记 (左手清除重建, 右手累加)
                clear_kp = (prefix == "left")
                kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012, clear_existing=clear_kp)

                if analytical:
                    # 解析模式: 先平滑 MANO 输入位置, 再计算解析位姿
                    mano_wrist = joints_sapien[0, :3]
                    mano_finger1 = joints_sapien[ref_indices[0], :3]
                    mano_finger2 = joints_sapien[ref_indices[1], :3]
                    mano_smoother = gs.get("mano_smoother")
                    if mano_smoother is not None:
                        mano_pts = np.stack([mano_wrist, mano_finger1, mano_finger2])
                        mano_pts = mano_smoother.smooth(mano_pts)
                        mano_wrist, mano_finger1, mano_finger2 = mano_pts[0], mano_pts[1], mano_pts[2]
                    g_pos, g_R, joint1, joint2 = _compute_analytical_gripper_pose(
                        mano_wrist, mano_finger1, mano_finger2, prefix)
                    # 从 gripper_link 位姿转换为 root 位姿 (gripper_arm 模式需要减去 offset)
                    gripper_offset_pos = gs["gripper_offset_pos"]
                    gripper_offset_R = gs["gripper_offset_R"]
                    root_R = g_R @ gripper_offset_R.T
                    root_pos = g_pos - root_R @ gripper_offset_pos
                    root_quat = pr.quaternion_from_matrix(root_R)
                    robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))
                    qpos = robot.get_qpos().copy()
                    # 显式设置 arm 关节为 0 (防止物理仿真漂移)
                    for arm_idx in gs.get("arm_joint_indices", []):
                        qpos[arm_idx] = 0.0
                    qpos[gs["gripper_idx1"]] = float(joint1)
                    qpos[gs["gripper_idx2"]] = float(joint2)
                    robot.set_qpos(qpos)
                else:
                    # Retargeting 优化器模式 (与 02_render_scene.py 的 run_robot_tracking 一致):
                    # 1. 用 3 点 retargeting 优化器获取 root pose (6 DOF)
                    # 2. 从内部机器人 FK 获取 gripper_link 朝向 (优化器的最佳朝向估计)
                    # 3. 手指关节用解析值 (从 MANO 指尖距离计算, 确保夹爪开合)
                    # 4. root 位置用解析值 (匹配 fingertip1, 确保指尖精确跟踪)
                    #    (Hybrid 方案: FK 朝向 + 解析位置 + 解析手指关节)
                    ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                    retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

                    # 夹爪朝向: 用 retargeting FK
                    _, gripper_R_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)

                    # 补偿 offset
                    gripper_offset_pos = gs["gripper_offset_pos"]
                    gripper_offset_R = gs["gripper_offset_R"]

                    # 手指关节: 用解析值从 MANO 指尖距离计算 (确保夹爪开合)
                    mano_f1 = joints_sapien[ref_indices[0], :3]
                    mano_f2 = joints_sapien[ref_indices[1], :3]
                    mano_finger_dist = float(np.linalg.norm(mano_f1 - mano_f2))
                    required_open = max(0.0, mano_finger_dist - _FINGER_BASE_DIST)
                    joint1 = min(0.05, required_open / 2)
                    joint2 = min(0.05, required_open / 2)

                    # root 朝向: 用 FK 朝向
                    root_R = gripper_R_fk @ gripper_offset_R.T
                    # root 位置: 解析计算, 匹配 fingertip1
                    fg = _FINGER_GEOM_ARRAYS[prefix]
                    finger1_in_root = gripper_offset_pos + gripper_offset_R @ (fg["finger1_origin"] + fg["finger1_axis"] * joint1)
                    root_pos = mano_f1 - root_R @ finger1_in_root
                    root_quat = pr.quaternion_from_matrix(root_R)

                    # EMA 平滑
                    if target_smoother is not None:
                        root_pos, root_quat = target_smoother.smooth(root_pos, root_quat)

                    robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))

                    qpos = robot.get_qpos().copy()
                    for arm_idx in gs.get("arm_joint_indices", []):
                        qpos[arm_idx] = 0.0
                    qpos[gs["gripper_idx1"]] = float(joint1)
                    qpos[gs["gripper_idx2"]] = float(joint2)
                    robot.set_qpos(qpos)

                # 验证误差
                # 注意: verify 模式下不调用 scene.step(), 避免物理仿真导致 arm 关节漂移
                if verify:
                    scene.update_render()
                    for link in robot.get_links():
                        lname = link.get_name()
                        if lname == f"{prefix}_gripper_finger_link1":
                            finger1_pos = np.array(link.get_entity_pose().p)
                        elif lname == f"{prefix}_gripper_finger_link2":
                            finger2_pos = np.array(link.get_entity_pose().p)
                        elif lname == f"{prefix}_gripper_link":
                            pose = link.get_entity_pose()
                            gripper_link_pos = np.array(pose.p)
                            gripper_link_R = pr.matrix_from_quaternion(np.array(pose.q))
                    mano_finger1 = joints_sapien[ref_indices[0], :3]
                    mano_finger2 = joints_sapien[ref_indices[1], :3]
                    mano_wrist_pos = joints_sapien[0, :3]
                    wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
                    wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
                    err = {'prefix': prefix}
                    err[f'{prefix}_finger1_mm'] = float(np.linalg.norm(finger1_pos - mano_finger1) * 1000)
                    err[f'{prefix}_finger2_mm'] = float(np.linalg.norm(finger2_pos - mano_finger2) * 1000)
                    err[f'{prefix}_wrist_pos_mm'] = float(np.linalg.norm(gripper_link_pos - mano_wrist_pos) * 1000)
                    R_diff = gripper_link_R.T @ wrist_R_sapien
                    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
                    err[f'{prefix}_wrist_ori_deg'] = float(np.degrees(angle_rad))
                    # 记录手指关节值 (确认夹爪开合)
                    qpos_now = robot.get_qpos()
                    err[f'{prefix}_joint1'] = float(qpos_now[gs["gripper_idx1"]])
                    err[f'{prefix}_joint2'] = float(qpos_now[gs["gripper_idx2"]])
                    verify_errors.append(err)
                else:
                    scene.step()

                if prefix == "left":
                    left_valid = True
                else:
                    right_valid = True

            if not verify:
                scene.step()

            if viewer:
                sapien_viewer.render()
            else:
                scene.update_render()
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                hand_info = f"L:{'Y' if left_valid else 'N'} R:{'Y' if right_valid else 'N'}"
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Dual Gripper  |  {hand_info}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(bgr)

        if viewer:
            logger.info("  动画播放完成, 重新开始... (关闭窗口退出)")
            for gs in gripper_states:
                robot = gs["robot"]
                init_qpos = robot.get_qpos().copy()
                init_qpos[gs["gripper_idx1"]] = GRIPPER_INIT_OPEN
                init_qpos[gs["gripper_idx2"]] = GRIPPER_INIT_OPEN
                robot.set_qpos(init_qpos)
                if gs.get("target_smoother"):
                    gs["target_smoother"].reset()
                if gs.get("mano_smoother"):
                    gs["mano_smoother"].reset()

    if not viewer:
        writer.release()

        # ffmpeg 重编码
        final_path = output
        tmp_path = str(output).replace(".mp4", "_tmp.mp4")
        if os.path.exists(str(output)):
            os.rename(str(output), tmp_path)
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                import subprocess as sp
                cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
                       "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
                       "-movflags", "+faststart", str(final_path)]
                result = sp.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    os.remove(tmp_path)
                else:
                    if os.path.exists(tmp_path):
                        os.rename(tmp_path, final_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, final_path)

        logger.info(f"\n✓ 双夹爪URDF视频已保存: {final_path}")
    else:
        final_path = None
        logger.info("\n✓ Viewer 模式结束")

    # ── 验证误差报告 ──
    if verify and verify_errors:
        logger.info("\n  === 验证误差报告 ===")
        for pfx in ['left', 'right']:
            for suffix, label, unit in [('_finger1_mm', '指尖1位置误差', 'mm'),
                                         ('_finger2_mm', '指尖2位置误差', 'mm'),
                                         ('_wrist_pos_mm', '手腕位置误差', 'mm'),
                                         ('_wrist_ori_deg', '手腕朝向误差', 'deg')]:
                key = f'{pfx}{suffix}'
                vals = [e[key] for e in verify_errors if key in e]
                if vals:
                    mean_v = np.mean(vals)
                    max_v = np.max(vals)
                    logger.info(f"  [{pfx}] {label}: mean={mean_v:.2f}{unit}, max={max_v:.2f}{unit}")
            # 手指关节值范围 (确认夹爪开合)
            j1_vals = [e[f'{pfx}_joint1'] for e in verify_errors if f'{pfx}_joint1' in e]
            j2_vals = [e[f'{pfx}_joint2'] for e in verify_errors if f'{pfx}_joint2' in e]
            if j1_vals:
                logger.info(f"  [{pfx}] 手指关节: joint1=[{min(j1_vals):.4f}, {max(j1_vals):.4f}], "
                            f"joint2=[{min(j2_vals):.4f}, {max(j2_vals):.4f}] "
                            f"(开合范围: {max(j1_vals)+max(j2_vals)-min(j1_vals)-min(j2_vals):.4f}m)")
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="只渲染夹爪URDF (不加载手臂)",
    )
    parser.add_argument("--hawor-dir", type=str, required=True, help="HaWoR 数据目录")
    parser.add_argument("--ras-dir", type=str, required=True, help="RAS 重建结果目录")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: hand_track/output/{hawor_name})")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--num-frames", type=int, default=-1, help="处理帧数 (-1=全部)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--width", type=int, default=1920, help="视频宽度")
    parser.add_argument("--height", type=int, default=1080, help="视频高度")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF 质量参数")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角 (默认 fpv)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["gripper", "gripper_arm", "both"],
                        help="渲染模式: gripper=仅夹爪, gripper_arm=夹爪+手臂末端, both=两者都渲染 (默认)")
    parser.add_argument("--smooth", type=int, default=1,
                        choices=[0, 1],
                        help="平滑模式: 0=不平滑, 1=EMA平滑 (默认 1)")
    parser.add_argument("--viewer", action="store_true",
                        help="使用 SAPIEN Viewer 实时循环播放 (不保存视频)")
    parser.add_argument("--verify", action="store_true",
                        help="计算并输出指尖位置/手腕位姿误差")
    parser.add_argument("--optimizer", action="store_true",
                        help="使用优化器模式 (默认: 解析模式, 从 MANO 指尖向量直接计算位姿, 精确跟踪 3 个特征点)")
    args = parser.parse_args()

    # Logger
    logger = logging.getLogger("GripperOnly")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)

    # 输出目录
    hawor_name = Path(args.hawor_dir).name
    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / "output" / hawor_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # [1] 自动检测手部
    hand_indices = detect_hands(args.hawor_dir)
    hand_count = len(hand_indices)
    hand_label = "双手" if hand_count == 2 else ("左手" if hand_indices[0] == 0 else "右手")
    logger.info(f"[1/3] 手部检测: {hand_label} (indices={hand_indices})")

    # [2] 确保 transform_params 存在
    logger.info(f"\n[2/3] 准备 GLB 变换参数 ...")
    tp_path = _ensure_transform_params(args.ras_dir, args.hawor_dir, args.output_dir, logger)

    # [3] 渲染 — 默认同时渲染 gripper 和 gripper_arm
    modes_to_render = []
    if args.mode in ("gripper", "both"):
        modes_to_render.append(("gripper", False, ""))
    if args.mode in ("gripper_arm", "both"):
        modes_to_render.append(("gripper_arm", True, "_arm"))

    analytical = not args.optimizer
    logger.info(f"\n[3/3] 渲染夹爪URDF视频 (mode={args.mode}, smooth={args.smooth}, viewer={args.viewer}, verify={args.verify}, analytical={analytical}) ...")
    start_time = time.time()

    for mode_name, with_arm, mode_suffix in modes_to_render:
        logger.info(f"\n--- 渲染模式: {mode_name} ---")

        if hand_count == 1:
            hi = hand_indices[0]
            side = "left" if hi == 0 else "right"
            output_video = os.path.join(args.output_dir, "videos", f"hawor_r1_{side}_gripper_urdf{mode_suffix}.mp4")
            os.makedirs(os.path.dirname(output_video), exist_ok=True)

            render_gripper_only_video(
                hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                transform_params_path=tp_path, output=output_video,
                hand_idx=hi, fps=args.fps, cam_width=args.width, cam_height=args.height,
                view=args.view, crf=args.crf, start_frame=args.start_frame,
                num_frames=args.num_frames, with_arm=with_arm,
                smooth=args.smooth, viewer=args.viewer, verify=args.verify,
                analytical=analytical, logger=logger,
            )
        else:
            # 双手: 同一个场景中渲染左右夹爪
            output_video = os.path.join(args.output_dir, "videos", f"hawor_r1_dual_gripper_urdf{mode_suffix}.mp4")
            os.makedirs(os.path.dirname(output_video), exist_ok=True)

            render_dual_gripper_video(
                hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                transform_params_path=tp_path, output=output_video,
                fps=args.fps, cam_width=args.width, cam_height=args.height,
                view=args.view, crf=args.crf, start_frame=args.start_frame,
                num_frames=args.num_frames, with_arm=with_arm,
                smooth=args.smooth, viewer=args.viewer, verify=args.verify,
                analytical=analytical, logger=logger,
            )

    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
