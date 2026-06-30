#!/usr/bin/env python3
"""
gripper_config.py — 夹爪 URDF 模板、几何常数、生成函数、平滑器、retargeting 初始化

从 render_gripper_only.py 拆分出来, 方便其他模块复用。
"""

import re
import tempfile
from pathlib import Path

import numpy as np

# ── 路径常量 ──
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"

# ── 坐标变换 ──
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# ── 平滑参数 ──
WARMUP_FRAMES = 30
LP_ALPHA_POS = 0.6
LP_ALPHA_ORI = 0.6
LP_ALPHA_ANALYTICAL = 0.9
GRIPPER_INIT_OPEN = 0.04


# ── 平滑器 ──

class EmaTargetSmoother:
    """EMA 目标平滑器 (位置 + 四元数朝向)"""

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
        self.ori_quat = self.ori_alpha * (ori_quat - self.ori_quat) + self.ori_quat
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
    """多点位 EMA 平滑器 (用于解析模式: 平滑 MANO 输入位置)"""

    def __init__(self, alpha=LP_ALPHA_POS):
        self.alpha = alpha
        self.positions = None

    def smooth(self, positions):
        if self.positions is None:
            self.positions = positions.copy()
        else:
            self.positions = self.positions + self.alpha * (positions - self.positions)
        return self.positions.copy()

    def reset(self):
        self.positions = None


# ── 夹爪几何常数 ──

# 两个手指闭合时的距离 (joint1=joint2=0)
FINGER_BASE_DIST = 0.026906  # = abs(0.013453 - (-0.013453))

# prefix 相关的手指几何 (numpy 数组, 用于解析计算)
FINGER_GEOM_ARRAYS = {
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

# 夹爪手指关节几何 (从 robot.urdf 提取, 左右手 joint1/joint2 互换)
GRIPPER_JOINT_GEOM = {
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


# ── URDF 模板 ──

# 夹爪 URDF 模板 (只有 gripper 部分)
GRIPPER_URDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
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
GRIPPER_WITH_ARM_URDF_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
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


# ── URDF 生成函数 ──

def generate_gripper_urdf(prefix="right", finger_origin_x=0.03689,
                          finger1_origin_x=None, finger2_origin_x=None):
    """生成只包含夹爪的 URDF 文件

    Args:
        finger_origin_x: finger joint origin 的 X 分量 (两个手指相同)
            默认 0.03689 (原始 URDF 值, 夹爪长 37mm)
            设为 MANO 腕→指尖距离 (~0.12-0.14m) 可使3点都近似对应
        finger1_origin_x: finger1 joint origin 的 X 分量 (覆盖 finger_origin_x)
        finger2_origin_x: finger2 joint origin 的 X 分量 (覆盖 finger_origin_x)
    Returns:
        urdf_path: 生成的临时 URDF 文件路径
    """
    f1_x = finger1_origin_x if finger1_origin_x is not None else finger_origin_x
    f2_x = finger2_origin_x if finger2_origin_x is not None else finger_origin_x
    geom = dict(GRIPPER_JOINT_GEOM[prefix])
    # finger1 origin
    parts1 = geom["joint1_origin"].split()
    parts1[0] = f"{f1_x:.6f}"
    geom["joint1_origin"] = " ".join(parts1)
    # finger2 origin
    parts2 = geom["joint2_origin"].split()
    parts2[0] = f"{f2_x:.6f}"
    geom["joint2_origin"] = " ".join(parts2)
    xml = GRIPPER_URDF_TEMPLATE.format(
        prefix=prefix,
        mesh_dir=str(R1_MESH_DIR),
        **geom,
    )
    temp_dir = tempfile.mkdtemp(prefix=f"r1_gripper_{prefix}-")
    temp_path = f"{temp_dir}/r1_gripper_{prefix}.urdf"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def generate_gripper_with_arm_urdf(prefix="right", finger_origin_x=0.03689,
                                   finger1_origin_x=None, finger2_origin_x=None):
    """生成包含夹爪+arm_link4/5/6的 URDF 文件"""
    f1_x = finger1_origin_x if finger1_origin_x is not None else finger_origin_x
    f2_x = finger2_origin_x if finger2_origin_x is not None else finger_origin_x
    geom = dict(GRIPPER_JOINT_GEOM[prefix])
    # finger1 origin
    parts1 = geom["joint1_origin"].split()
    parts1[0] = f"{f1_x:.6f}"
    geom["joint1_origin"] = " ".join(parts1)
    # finger2 origin
    parts2 = geom["joint2_origin"].split()
    parts2[0] = f"{f2_x:.6f}"
    geom["joint2_origin"] = " ".join(parts2)
    xml = GRIPPER_WITH_ARM_URDF_TEMPLATE.format(
        prefix=prefix,
        mesh_dir=str(R1_MESH_DIR),
        **geom,
    )
    temp_dir = tempfile.mkdtemp(prefix=f"r1_gripper_arm_{prefix}-")
    temp_path = f"{temp_dir}/r1_gripper_arm_{prefix}.urdf"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def prepare_full_arm_urdf(prefix="right", finger_origin_x=0.03689,
                          finger1_origin_x=None, finger2_origin_x=None):
    """生成完整的夹爪+手臂 URDF (使用 GalaxeaManipSim 的浮动 URDF, 不缩放)

    参考 02_render_scene.py 的 prepare_arm_urdf(), 使用原始几何 (finger origin=0.03689).
    与 generate_gripper_with_arm_urdf 不同, 这个函数使用完整的运动链:
      arm_base_link → arm_link1/2/3/4/5/6 → gripper_link → finger_link1/2

    步骤:
      1. 读取 GalaxeaManipSim 的 r1_v2_1_0_floating_{prefix}.urdf
      2. 替换 package://r1_v2_1_0/meshes/ 为绝对路径
      3. 将 gripper_finger_joint1/2 从 fixed 改为 prismatic

    注意: 不缩放 finger origin, 保持原始 37mm 几何, 与 02_render_scene.py 一致.
    gripper_link 位姿通过 retargeting + FK 获取, 不需要缩放.

    Args:
        prefix: "right" 或 "left"
        finger_origin_x: 未使用 (保留参数兼容性)
        finger1_origin_x: 未使用 (保留参数兼容性)
        finger2_origin_x: 未使用 (保留参数兼容性)

    Returns:
        str: 生成的临时 URDF 文件路径
    """
    src_urdf = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / f"r1_v2_1_0_floating_{prefix}.urdf"
    xml = src_urdf.read_text()

    # 1. 替换 mesh 路径
    xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")

    # 2. 将 finger joint 从 fixed 改为 prismatic (右手 URDF 需要, 左手已是 prismatic)
    xml = re.sub(
        rf'(<joint\s+name="{prefix}_gripper_finger_joint1"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    xml = re.sub(
        rf'(<joint\s+name="{prefix}_gripper_finger_joint2"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )

    temp_dir = tempfile.mkdtemp(prefix=f"r1_full_arm_{prefix}-")
    temp_path = f"{temp_dir}/r1_full_arm_{prefix}.urdf"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def prepare_half_arm_urdf(prefix="right", finger_origin_x=0.03689,
                          finger1_origin_x=None, finger2_origin_x=None):
    """生成半个手臂+夹爪 URDF (arm_link4/5/6 + gripper, 从完整浮动 URDF 提取, 不缩放)

    从 GalaxeaManipSim 的 r1_v2_1_0_floating_{prefix}.urdf 中提取:
      arm_base_link → arm_link4/5/6 → gripper_link → finger_link1/2

    与 generate_gripper_with_arm_urdf 不同, 这个函数从完整 URDF 提取,
    保证了 arm_link4 的 origin 是 link3→link4 的正确变换 (而不是 base_link→link4).

    步骤:
      1. 读取完整浮动 URDF
      2. 替换 package://r1_v2_1_0/meshes/ 为绝对路径
      3. 将 finger joint 从 fixed 改为 prismatic
      4. 移除 arm_link1/2/3, 将 arm_joint4 的 parent 改为 arm_base_link

    Args:
        prefix: "right" 或 "left"
        finger_origin_x: 未使用 (保留参数兼容性)
        finger1_origin_x: 未使用 (保留参数兼容性)
        finger2_origin_x: 未使用 (保留参数兼容性)

    Returns:
        str: 生成的临时 URDF 文件路径
    """
    src_urdf = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / f"r1_v2_1_0_floating_{prefix}.urdf"
    xml = src_urdf.read_text()

    # 1. 替换 mesh 路径
    xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")

    # 2. 将 finger joint 从 fixed 改为 prismatic
    xml = re.sub(
        rf'(<joint\s+name="{prefix}_gripper_finger_joint1"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    xml = re.sub(
        rf'(<joint\s+name="{prefix}_gripper_finger_joint2"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )

    # 3. 移除 arm_link1/2/3 相关的 link 和 joint
    # 移除 arm_joint1, arm_joint2, arm_joint3 及其 child link (arm_link1/2/3)
    for jn in [1, 2, 3]:
        # 移除 joint 块: <joint name="..._arm_jointN" ... </joint>
        xml = re.sub(
            rf'<joint\s+name="{prefix}_arm_joint{jn}"[\s\S]*?</joint>\s*',
            '', xml
        )
        # 移除 link 块: <link name="..._arm_linkN"> ... </link>
        xml = re.sub(
            rf'<link\s+name="{prefix}_arm_link{jn}">[\s\S]*?</link>\s*',
            '', xml
        )

    # 4. 将 arm_joint4 的 parent 从 arm_link3 改为 arm_base_link
    xml = re.sub(
        rf'(<joint\s+name="{prefix}_arm_joint4"[\s\S]*?<parent\s+link="){prefix}_arm_link3(")',
        rf'\g<1>{prefix}_arm_base_link\g<2>', xml
    )

    temp_dir = tempfile.mkdtemp(prefix=f"r1_half_arm_{prefix}-")
    temp_path = f"{temp_dir}/r1_half_arm_{prefix}.urdf"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


# ── 解析位姿计算 ──

def compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2, prefix="right",
                                     finger_origin_x=0.03689,
                                     finger1_origin_x=None, finger2_origin_x=None):
    """从 MANO 3 个特征点计算夹爪 gripper_link 位姿和手指关节值

    方法 (Gram-Schmidt + 指尖中点匹配):
      1. 指向方向 X = normalize(指尖中点 - 手腕)
      2. 开合方向 Y = normalize(finger2-finger1 投影到 X 的垂直面)
      3. Z = X × Y
      4. 旋转 R = [X, Y, Z] (位姿对齐)
      5. 手指关节: 从 MANO 指尖距离计算 (与 URDF 37mm 几何一致)
      6. gripper_link 位置 = 指尖中点 - R @ finger_mid_in_gripper
         (两个指尖对齐 MANO 指尖, 手腕不对齐)

    关键: gripper_link 不放在手腕位置, 而是放在指尖中点后方 finger_origin_x 处。
    这样两个指尖精确对齐 MANO 指尖, gripper_link 在指尖和手腕之间 (视觉正确)。

    Args:
        finger_origin_x: finger joint origin 的 X 分量 (默认 0.03689 = URDF 原始值)
        finger1_origin_x: finger1 的独立 x 偏移 (覆盖 finger_origin_x)
        finger2_origin_x: finger2 的独立 x 偏移 (覆盖 finger_origin_x)

    Returns:
        gripper_pos: (3,) gripper_link 位置 (指尖中点后方, 不在手腕处)
        gripper_R: (3,3) gripper_link 旋转矩阵
        joint1, joint2: 手指关节值
    """
    fg = FINGER_GEOM_ARRAYS[prefix]

    # finger origin (默认 37mm, 与 URDF 一致)
    f1_x = finger1_origin_x if finger1_origin_x is not None else finger_origin_x
    f2_x = finger2_origin_x if finger2_origin_x is not None else finger_origin_x
    finger1_origin = fg["finger1_origin"].copy()
    finger1_origin[0] = f1_x
    finger2_origin = fg["finger2_origin"].copy()
    finger2_origin[0] = f2_x

    # 1. 指向方向: 手腕 → 指尖中点
    midpoint = (mano_finger1 + mano_finger2) / 2
    v_pointing = midpoint - mano_wrist
    X = v_pointing / np.linalg.norm(v_pointing)

    # 2. 开合方向: finger2-finger1 投影到 X 的垂直面
    v_opening = mano_finger2 - mano_finger1
    v_opening_proj = v_opening - np.dot(v_opening, X) * X
    Y = v_opening_proj / np.linalg.norm(v_opening_proj)

    # 3. Z = X × Y
    Z = np.cross(X, Y)

    # 4. 检查 Y 轴方向是否与夹爪几何一致
    gripper_opening_local = finger2_origin - finger1_origin
    sign = np.sign(gripper_opening_local[1])
    if sign * np.dot(Y, v_opening) < 0:
        Y = -Y
        Z = -Z

    # 5. 旋转矩阵
    R = np.column_stack([X, Y, Z])
    if np.linalg.det(R) < 0:
        Z = -Z
        R = np.column_stack([X, Y, Z])

    # 6. 手指关节: 从 MANO 指尖距离计算 (与 URDF 几何一致)
    finger_dist = np.linalg.norm(mano_finger2 - mano_finger1)
    required_open_sum = finger_dist - FINGER_BASE_DIST
    joint1 = max(0.0, min(0.05, required_open_sum / 2))
    joint2 = max(0.0, min(0.05, required_open_sum / 2))

    # 7. gripper_link 位置: 匹配指尖中点 (不是手腕!)
    #    两个指尖对齐 MANO 指尖, gripper_link 在指尖中点后方 finger_origin_x 处
    finger1_in_gripper = finger1_origin + fg["finger1_axis"] * joint1
    finger2_in_gripper = finger2_origin + fg["finger2_axis"] * joint2
    finger_mid_in_gripper = (finger1_in_gripper + finger2_in_gripper) / 2
    gripper_pos = midpoint - R @ finger_mid_in_gripper

    return gripper_pos, R, joint1, joint2


# ── Retargeting 初始化 ──

def init_gripper_retargeting(prefix, finger_origin_x, project_root,
                              finger1_origin_x=None, finger2_origin_x=None):
    """初始化夹爪 retargeting 优化器 (使用 gripper-only URDF)

    使用 gripper-only URDF + add_dummy_free_joint, 优化器有 8 DOF:
      - 6 dummy joints (root pose: 3 translation + 3 rotation)
      - 2 finger joints (prismatic)
    3 个目标点 (9 约束 > 8 DOF) 完全确定解, 误差应接近 0。

    Args:
        prefix: "left" 或 "right"
        finger_origin_x: 缩放后的 finger_origin_x (MANO 腕→指尖距离, 两个手指相同)
        project_root: dex-retargeting 项目根目录
        finger1_origin_x: finger1 的独立 x 偏移 (覆盖 finger_origin_x)
        finger2_origin_x: finger2 的独立 x 偏移 (覆盖 finger_origin_x)
    Returns:
        retargeting: SeqRetargeting 对象
        ref_indices: target_link_human_indices
        gripper_urdf_path: 生成的 URDF 文件路径
    """
    from dex_retargeting.retargeting_config import RetargetingConfig

    # 生成缩放后的 gripper-only URDF
    gripper_urdf_path = generate_gripper_urdf(
        prefix, finger_origin_x,
        finger1_origin_x=finger1_origin_x,
        finger2_origin_x=finger2_origin_x,
    )

    # 加载 YAML 配置
    config_dir = Path(__file__).resolve().parent / "configs"
    config_path = config_dir / f"r1_gripper_{prefix}.yml"

    # override URDF 路径为动态生成的缩放 URDF
    override = dict(urdf_path=gripper_urdf_path)

    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    ref_indices = retargeting.optimizer.target_link_human_indices

    return retargeting, ref_indices, gripper_urdf_path


def compute_gripper_offset_in_root(robot, prefix):
    """计算 gripper_link 相对于 root link 的位置 offset (当所有 arm_joint=0 时)

    用于 gripper_arm 模式: 设置 root pose 时需要补偿这个 offset。
    """
    from pytransform3d import rotations as pr

    target_name = f"{prefix}_gripper_link"
    for link in robot.get_links():
        if link.get_name() == target_name:
            pose = link.get_entity_pose()
            offset_pos = np.array(pose.p)
            offset_R = pr.matrix_from_quaternion(np.array(pose.q))
            return offset_pos, offset_R
    return np.zeros(3), np.eye(3)
