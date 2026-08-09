"""physics_env.py — SAPIEN 物理环境: URDF 准备 + 场景 + 机器人 + 物理步进 + 接触检测

从 grasp_hawor.py 抽出, 共享物理仿真基础设施。
"""
import os
import re
import gc
import sys
import logging
import tempfile
import importlib.util
from pathlib import Path

import numpy as np
import sapien
import sapien.render
from pytransform3d import rotations as pr

try:
    import trimesh
except ImportError:
    trimesh = None

# SAPIEN Monkey-Patch (必须在任何 URDF 加载前执行)
from sapien.wrapper.urdf_loader import URDFLoader as _URDFLoader
from sapien.wrapper.articulation_builder import (
    LinkBuilder as _LinkBuilder,
    ArticulationBuilder as _ArticulationBuilder,
)

# ============================================================
# SAPIEN Monkey-Patch: 浮动根支持 (fix_root_link=False → 'free' joint)
# ============================================================
# 问题: SAPIEN URDF loader 在 fix_root_link=False 时, 把根关节类型设为 "undefined".
# 但 PhysX 把 "undefined" 当作 fixed (0 DOF) 处理, 导致根无法移动,
# set_root_linear_velocity 设置的速度无效, 摩擦力无法计算 → 物体无法被提起.
#
# 关键根因: ArticulationBuilder.build_entities() 末尾有硬编码 override:
#   if fix_root_link is not None:
#       entities[0].components[0].joint.type = "fixed" if fix_root_link else "undefined"
# 这个 override 在所有 link 构建之后运行, 会覆盖任何之前设置的 joint.type.
# 因此必须 patch build_entities, 在 override 之后把 "undefined" 改回 "free".

# 1. 放宽 _check: 允许 "free" 作为根关节类型 (原代码只允许 "fixed" / "undefined")
def _patched_check(self):
    if self.parent is None:
        assert self.joint_record.joint_type in ["fixed", "undefined", "free"], \
            f"Invalid root joint type: {self.joint_record.joint_type}"
    else:
        assert self.joint_record.joint_type in [
            "fixed", "revolute", "revolute_unwrapped", "prismatic", "continuous",
        ], f"Invalid joint type: {self.joint_record.joint_type}"
_LinkBuilder._check = _patched_check

# 2. 拦截 _parse_articulation: 把根 joint_record 类型从 'undefined' 改为 'free'
#    (这样 build_entities 内部的 L106 会先设 joint.type="free", 虽然后续被 override 覆盖)
_original_parse_articulation = _URDFLoader._parse_articulation


def _patched_parse_articulation(self, root, fix_base):
    builder = _original_parse_articulation(self, root, fix_base)
    if not fix_base and builder.link_builders:
        builder.link_builders[0].joint_record.joint_type = "free"
    return builder


_URDFLoader._parse_articulation = _patched_parse_articulation

# 3. 关键 patch: 拦截 build_entities, 跳过 SAPIEN 的 override.
#    SAPIEN build_entities 末尾有 override:
#        if fix_root_link is not None:
#            entities[0].components[0].joint.type = "fixed" if fix_root_link else "undefined"
#    这个 override 会把 L106 设置的 "free" (来自我们的 _parse_articulation patch) 覆盖回 "undefined".
#    而且 C++ 在 build_physx_component 时可能已锁定关节类型, post-hoc 修改 joint.type 无效.
#    解决: 当 fix_root_link=False 时, 传 None 给原函数, 跳过 override,
#    让 L106 的 joint.type="free" (来自 joint_record.joint_type="free") 保持不变.
_original_build_entities = _ArticulationBuilder.build_entities


def _patched_build_entities(self, fix_root_link=None):
    if fix_root_link is False:
        # 传 None 跳过 SAPIEN 的 override, 保留 L106 设置的 "free" 类型.
        # 注意: 不传 fix_root_link=False, 否则 override 会把它改回 "undefined".
        entities = _original_build_entities(self, fix_root_link=None)
    else:
        entities = _original_build_entities(self, fix_root_link=fix_root_link)
    return entities


_ArticulationBuilder.build_entities = _patched_build_entities

# ============ 路径 ============
# grasp_hawor.py 在 dex-retargeting/example/combination/tri_model_physics/
# parents[0]=tri_model_physics, [1]=combination, [2]=example, [3]=dex-retargeting
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # dex-retargeting/
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tri_model_physics/

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_ASSETS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1"
R1_MESH_DIR = R1_ASSETS / "meshes"
R1_URDF_DIR = R1_ASSETS / "configs" / "urdfs"
FULL_ROBOT_URDF = R1_URDF_DIR / "r1_v2_1_0.urdf"
R1_RIGHT_SETTINGS = R1_ASSETS / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = R1_ASSETS / "configs" / "settings_left.yaml"

# 01_align_scene.py 所在目录
COMBINATION_DIR = Path(__file__).resolve().parents[1]  # combination/
sys.path.insert(0, str(COMBINATION_DIR))

# 01_align_scene.py 文件名以数字开头, 无法用普通 import, 用 importlib.util 加载一次缓存
_ALIGN_SCRIPT = COMBINATION_DIR / "01_align_scene.py"
_ALIGN_SPEC = importlib.util.spec_from_file_location("align_scene", str(_ALIGN_SCRIPT))
_ALIGN_MOD = importlib.util.module_from_spec(_ALIGN_SPEC)
_ALIGN_SPEC.loader.exec_module(_ALIGN_MOD)
compute_and_save_transform_params = _ALIGN_MOD.compute_and_save_transform_params

# ============ 坐标变换 (与 02/04 一致) ============
R_x = np.eye(3)  # 001 链: R_X = I (经验验证 HaWoR 和 RAS 相机坐标系一致)
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# ============ 机器人参数 ============
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985]
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.35
# 臂基座沿机器人 forward 反方向后退距离 (让机器人退后, 不挡物体)
# 对齐 04_physics_simulation.py 的 BASE_OFFSET_Y=0.30 思路, 但用 forward 而非世界 Y
BASE_BACK_OFFSET = 0.20
# R1 机器人臂基座相对 ROOT 的偏移 (世界坐标系, 单位四元数朝向)
# 原偏移在 180° Z 旋转下测量为 [0.032, +0.097, 1.403]
# 180° Z 将 [x,y,z]→[-x,-y,z], 切换到单位四元数需翻转 x/y 符号
# 右臂: [-0.032, -0.097, 1.403], 左臂对称: [-0.032, +0.097, 1.403]
ARM_BASE_OFFSET_RIGHT = np.array([-0.032, -0.097, 1.403])
ARM_BASE_OFFSET_LEFT = np.array([-0.032, 0.097, 1.403])
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
RIGHT_ARM_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798], [0.0, 3.2289], [-3.3161, 0.0],
    [-2.8798, 2.8798], [-1.6581, 1.6581], [-2.8798, 2.8798],
], dtype=np.float64)

# ============ 物理参数 (与 GalaxeaManipSim / 04 一致) ============
JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 200.0
# v4.7: PD 参数对齐 grasp_demo.py / R1Robot 默认 (joint_stiffness=1000, joint_damping=200)
# 用户反馈: "force和比如密度最好参考一下 grasp_demo.py"
# v4.9: 用户反馈 "调强 PD 让 peak_force 更大" — v4.8 peak_force=0.47N 偏小, lift 仅 0.94cm
# v4.14 close 阶段物理调参 (用户: "物体和夹爪需要保持住这个相对关系, 可以更改仿真参数"):
# 5 帧 smoothstep 内 PD 必须充分收敛到 _GRIP_PD_HOLD
# v4.14a-c 测试: 增大 stiffness 无效, 因为 PD 力被 force_limit=100N 饱和截断
# v4.14d: force_limit 500N + damping 400 → F50 q=0.016, lift=0.5cm (vs v4.14c 0.2cm)
# v4.14e: force_limit 1000N + damping 200 → 突破饱和更深, F50 q 应 ≈ 0.008 (充分收敛)
GRIPPER_STIFFNESS = 8000.0   # 保持
GRIPPER_DAMPING = 200.0      # v4.14e: 400 → 200 (饱和状态下 damping 主导收敛速度, 减半加速)
PHYSICS_TIMESTEP = 1 / 240.0
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))  # =8
GROUND_HEIGHT = 0
OBJECT_DENSITY = 1000.0  # 基础惯性变量: 所有物体统一密度 (kg/m³), 对齐水密度, 用户: "基础的惯性变量"
OBJECT_MIN_MASS = 0.05   # v4.14e: 0.1 → 0.05 (轻物体更易被夹爪带动跟随 base 移动, close 阶段不脱离)
GRIPPER_FRICTION = 2.0   # v4.14f: 2.0 最佳 (3.0 物体被卡住推不开 f1, 2.5 与 2.0 无明显差别)
GRIPPER_INIT_OPEN = 0.04
GRIPPER_MAX_OPEN = 0.05
GRIPPER_CLOSE_BIAS = 0.005  # 闭合偏移 5mm, 确保手指紧贴物体
# v4.7: force_limit 对齐 URDF effort=100N (原 500 超过 URDF 定义)
# v4.14d: force_limit 100 → 500N (用户: "可以更改仿真参数" — 突破 PD 饱和, 让 5 帧 smoothstep 内能充分收敛)
# v4.14e: 500 → 1000N (进一步突破饱和, 配合低 damping 让 PD 在 5 帧内充分收敛到 target)
# v4.14f: 1000 → 2000N (F50 q=0.014 仍不够, 继续突破)
# v4.14h: 2000 → 4000N (让 PD 在 F50 时 qpos 充分收敛到 target=0.0079, 物体真正被 Y 方向夹紧)
# 物理意义: 夹爪电机瞬时过载能力 (servo motor peak torque 通常 5-10x rated torque)
GRIPPER_FORCE = 4000.0  # v4.14h: 2000 → 4000 (F50 qpos 充分收敛, 物体真正被夹紧)
_DESCEND_OPEN = 0.012   # 边下降边夹时手指中张 qpos: y_gap≈5.1cm > 圆柱物体
_GRIP_CLOSE = 0.0       # PD 闭合目标 qpos; 设为关节下限, 让 PD 在接触后仍保持位置误差, 维持正压力
_GRIP_OVERCLOSURE = 0.0005  # 目标 qpos 比几何接触点再收紧 0.5mm, 保证有正压力

# Stage 1 夹持力策略 (与 plan v4.4 对齐, 减少硬编码解析)
# v4.7: force_limit 上限由 GRIPPER_FORCE=100N 决定, 单策略简化
GRASP_STRATEGIES = {
    'pd_10N': {'force_limit': 10.0, 'use_lock': False},
    'pd_20N': {'force_limit': 20.0, 'use_lock': False},
    'pd_30N': {'force_limit': 30.0, 'use_lock': False},
    'pd_then_lock': {'force_limit': 100.0, 'use_lock': True},  # 100N: 对齐 URDF effort
}
FINGER_FORWARD_NEUTRAL = 0.037  # 手指在 EE 前方 3.7cm (沿 gripper_R 的 X 轴)
MAX_ROOT_STEP = 0.008  # 根速度限制: 每帧 ≤ 0.8cm (盘子翻转临界点 0.008-0.010, 仅 0.008 让盘子不翻; 根误差增大是物理稳定代价)

# 夹爪几何 (与 04/hand_track 一致)
_FINGER1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689, 0.013453, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])
FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906
# URDF 中手指碰撞盒 y 维度 0.015m, 但实际 mesh 内延使有效半间距更小。
# 根据 2.17cm 物体成功抓取时 qpos≈0.0116 反推: 有效半间距 ≈ obj_max_dim/2 - qpos ≈ -0.00075,
# 因此将 FINGER_EFFECTIVE_HALF_SPACING 从 0.005953 下调到 0.002, 使闭合目标更紧。
GRIPPER_FINGER_HALF_THICKNESS = 0.011453  # 有效厚度(含 mesh 内延)
FINGER_EFFECTIVE_HALF_SPACING = 0.013453 - GRIPPER_FINGER_HALF_THICKNESS  # 0.002000


def rotmat_to_zyx_euler(R):
    """旋转矩阵 → ZYX Euler 角 (yaw, pitch, roll)

    对应 URDF 链: Rz(yaw) → Ry(pitch) → Rx(roll)
    对齐 05_gripper_test.py 的 rotmat_to_zyx_euler
    """
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy)

    if abs(sy) < 0.99999:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0

    return yaw, pitch, roll  # rz, ry, rx


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("grasp_hawor")

# 渲染失败的 scene 对象 (CPU 降级时保持引用避免 __del__ 段错误)
_FAILED_SCENES = []

# ============================================================
# 1. URDF 准备 — 修复多行 joint 正则 (根因修复)
# ============================================================
def prepare_full_robot_urdf(src_urdf_path, side="right", strip_visuals=False):
    """准备整个机器人 URDF: 替换 mesh 路径 + 修改关节类型

    修复旧版 "0臂关节" 根因:
      r1_v2_1_0.urdf 中 joint 定义跨多行:
        <joint
            name="right_arm_joint1"
            type="fixed">
      旧正则 (<joint\\s+name="..."\\s+type=")fixed 要求 name/type 同行, 匹配失败.
      本函数用 re.DOTALL + [\\s\\S]+ 匹配跨行, 正确转换 fixed→revolute.

    Args:
        src_urdf_path: 原始 URDF 路径
        side: "right" / "left" (决定夹爪关节前缀)
        strip_visuals: True 时移除 <visual> 块 (CPU 降级模式, 避免 RenderMaterial 失败)

    Returns:
        str: 修改后的临时 URDF 路径
    """
    xml = Path(src_urdf_path).read_text()
    # 1. 替换 package:// 路径
    xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")

    # 2. 臂关节 fixed → revolute (用 re.DOTALL 匹配跨行 name/type)
    n_arm_converted = 0
    for prefix in ["right", "left"]:
        for jn in range(1, 7):
            # 匹配 <joint ... name="xxx_arm_jointN" ... type="fixed"> 跨行
            pattern = rf'(<joint\s+name="{prefix}_arm_joint{jn}"[\s\S]*?type=")fixed(")'
            new_xml, n = re.subn(pattern, r'\1revolute\2', xml)
            if n > 0:
                xml = new_xml
                n_arm_converted += n
    logger.info(f"  URDF: 臂关节 fixed→revolute 转换 {n_arm_converted} 个")

    # 3. 夹爪关节 fixed → prismatic (同样跨行匹配)
    # side="both" 时转换两侧夹爪关节
    gripper_sides = ["right", "left"] if side == "both" else [side]
    n_gripper_converted = 0
    for gs in gripper_sides:
        for jn in [1, 2]:
            pattern = rf'(<joint\s+name="{gs}_gripper_finger_joint{jn}"[\s\S]*?type=")fixed(")'
            new_xml, n = re.subn(pattern, r'\1prismatic\2', xml)
            if n > 0:
                xml = new_xml
                n_gripper_converted += n
    logger.info(f"  URDF: 夹爪关节 fixed→prismatic 转换 {n_gripper_converted} 个")

    # 4. CPU 降级模式: 移除 <visual> 块 (保留 <collision> 和 <inertial>)
    if strip_visuals:
        xml = re.sub(r'<visual>[\s\S]*?</visual>', '', xml)
        logger.info("  URDF: <visual> 块已移除 (CPU 模式, 仅保留碰撞/惯量)")

    tmp_dir = tempfile.mkdtemp(prefix="r1_full_robot_")
    tmp_path = os.path.join(tmp_dir, os.path.basename(src_urdf_path))
    with open(tmp_path, "w") as f:
        f.write(xml)
    return tmp_path


def prepare_gripper_only_urdf(side="right", strip_visuals=False):
    """生成纯夹爪 URDF (含虚拟 6-DOF 关节, fix_root_link=True)

    结构: world (固定根) → virtual_x/y/z (prismatic) → virtual_rz/ry/rx (revolute)
           → gripper_base_link → gripper_link → finger_link1/2 (prismatic)

    虚拟关节通过 PD 驱动移动夹爪, 产生真实动量和接触力, 实现物理抓取.
    对齐 05_gripper_test.py 的成功架构.
    """
    template = """<?xml version="1.0" encoding="utf-8"?>
<robot name="r1_gripper_{prefix}">
  <!-- 虚拟机械臂: 3 个 prismatic 关节 (X, Y, Z) -->
  <link name="world"/>

  <joint name="virtual_x" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="world"/>
    <child link="virtual_x_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_x_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_y" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_x_link"/>
    <child link="virtual_y_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_y_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_z" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_y_link"/>
    <child link="virtual_z_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="3" effort="5000" velocity="5"/>
  </joint>
  <link name="virtual_z_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <!-- 虚拟姿态关节: ZYX Euler (Rz→Ry→Rx) -->
  <joint name="virtual_rz" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_z_link"/>
    <child link="virtual_rz_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_rz_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_ry" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_rz_link"/>
    <child link="virtual_ry_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_ry_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <joint name="virtual_rx" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_ry_link"/>
    <child link="virtual_rx_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="500" velocity="10"/>
  </joint>
  <link name="virtual_rx_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="1E-06" ixy="0" ixz="0" iyy="1E-06" iyz="0" izz="1E-06"/></inertial>
  </link>

  <!-- 虚拟臂 → 夹爪基座 -->
  <joint name="virtual_to_gripper" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="virtual_rx_link"/>
    <child link="{prefix}_gripper_base_link"/>
  </joint>

  <!-- 原始夹爪 -->
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
    <!-- visual+collision: gripper_link 主夹爪体 (恢复STL视觉) -->
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
  </link>
  <joint name="{prefix}_gripper_finger_joint1" type="prismatic">
    <origin xyz="0.03689 -0.013453 -0.00012053" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link1"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.05" effort="100" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link1">
    <inertial><origin xyz="-0.0195895587205407 0.0151136130965041 -0.00542255818128545" rpy="0 0 0"/>
    <mass value="0.027"/>
    <inertia ixx="2.40569063762433E-06" ixy="-3.99002073372071E-07" ixz="-5.12217975840564E-08"
             iyy="5.71082134562374E-06" iyz="6.19457183851545E-08" izz="6.4848557091919E-06"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <collision><origin xyz="0 0.01 0" rpy="0 0 0"/>
      <geometry><box size="0.06 0.01 0.04"/></geometry></collision>
  </link>
  <joint name="{prefix}_gripper_finger_joint2" type="prismatic">
    <origin xyz="0.03689 0.013453 0.00012067" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_link"/>
    <child link="{prefix}_gripper_finger_link2"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.05" effort="100" velocity="0.25"/>
  </joint>
  <link name="{prefix}_gripper_finger_link2">
    <inertial><origin xyz="-0.019589448977496 -0.0151137821219537 0.00542248304315596" rpy="0 0 0"/>
    <mass value="0.027"/>
    <inertia ixx="2.40568339234574E-06" ixy="3.98973340378568E-07" ixz="5.12055978237686E-08"
             iyy="5.71082803574443E-06" iyz="6.19476812784019E-08" izz="6.48485579679143E-06"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <collision><origin xyz="0 -0.01 0" rpy="0 0 0"/>
      <geometry><box size="0.06 0.01 0.04"/></geometry></collision>
  </link>
</robot>
"""
    xml = template.format(prefix=side, mesh_dir=str(R1_MESH_DIR))
    if strip_visuals:
        xml = re.sub(r'<visual>[\s\S]*?</visual>', '', xml)
        logger.info("  URDF: <visual> 块已移除 (CPU 模式, 仅保留碰撞/惯量)")
    tmp_dir = tempfile.mkdtemp(prefix=f"r1_gripper_only_{side}_")
    tmp_path = os.path.join(tmp_dir, f"r1_gripper_only_{side}.urdf")
    with open(tmp_path, "w") as f:
        f.write(xml)
    return tmp_path


# ============================================================
# 2. SAPIEN 场景 + GLB 加载 (参考 04_physics_simulation.py)
# ============================================================
def setup_physics_scene(ground_height=GROUND_HEIGHT, force_cpu=False):
    """创建 SAPIEN 物理场景

    对齐 002_render_scene.py 的渲染设置 + GalaxeaManipSim 的物理地面
    地面高度根据 GLB 物体最低点动态调整 (02 没有地面, 这里补上)

    Args:
        ground_height: 地面高度
        force_cpu: 强制使用 CPU 场景 (优化模式, 避免渲染初始化段错误)
    """
    render_available = False
    scene = None
    if force_cpu:
        # 优化模式: 直接创建 CPU 场景, 不尝试渲染 (避免 sandbox 环境段错误)
        scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
    else:
        # 正常模式: 先尝试创建带渲染的 Scene, 失败则降级 CPU
        try:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        except Exception:
            pass
        try:
            scene = sapien.Scene()
            render_available = True
            logger.info(f"  [setup_physics_scene] 带渲染场景创建成功, render_available={render_available}")
        except Exception as e:
            logger.warning(f"  [setup_physics_scene] 带渲染场景创建失败: {e}, 降级 CPU")
            scene = None
        if scene is None:
            scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
            logger.info(f"  [setup_physics_scene] CPU 场景创建 (降级模式)")

    # 渲染失败时自动重试一次 (GPU 资源可能需要释放时间)
    if not render_available and not force_cpu:
        import time
        time.sleep(1)
        try:
            scene = None
            scene = sapien.Scene()
            render_available = True
            logger.info(f"  [setup_physics_scene] 重试成功, render_available={render_available}")
        except Exception as e:
            logger.warning(f"  [setup_physics_scene] 重试仍失败: {e}")
            scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])

    scene.set_timestep(PHYSICS_TIMESTEP)

    if render_available:
        try:
            from sapien.asset import create_dome_envmap
            # 对齐 002_render_scene.py 的环境光照
            scene.set_environment_map(
                create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35])
            )
            scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
            scene.add_directional_light([-1, -0.5, -1], [1.2, 1.2, 1.2], shadow=False)
            scene.add_directional_light([0, 1, -0.5], [0.8, 0.8, 0.8], shadow=False)
            scene.set_ambient_light([0.5, 0.5, 0.5])
        except Exception:
            pass

    # 物理地面 (对齐 GalaxeaManipSim, 高度根据 GLB 物体调整)
    # R1 机器人 ROOT 在地下 (z≈-1.0), 地面 (z≈0) 会遮挡机器人身体.
    # 保留物理碰撞 (支撑物体), 隐藏地面视觉 (透明), 让整个机器人可见.
    # 地面摩擦对齐 GalaxeaManipSim 桌面 (static=1.0, dynamic=1.0, restitution=0.0)
    ground_material = sapien.physx.PhysxMaterial(
        static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    )
    # render=False 在 CPU 降级模式下跳过 RenderMaterial 创建 (避免 RuntimeError)
    ground_actor = scene.add_ground(
        ground_height, render=render_available, material=ground_material
    )
    if render_available:
        # gripper_only 模式不需要隐藏地面 (没有机器人躯干在地下)
        # full_robot 模式下地面视觉保持启用, 避免遮挡
        logger.info("  地面物理+视觉已加载 (render=True)")
    else:
        logger.info("  地面物理已加载 (无视觉, CPU 模式)")
    scene._render_available = render_available
    return scene


def setup_robot(scene, mode, side, base_pos, base_quat):
    """加载 R1 机器人并配置 PD 驱动

    Args:
        scene: SAPIEN 场景
        mode: "full_robot" / "gripper_only"
        side: "right" / "left" / "both"
        base_pos: (3,) 基座位置
        base_quat: (4,) 基座朝向

    Returns:
        dict: robot, joint_names, arm_joint_indices, gripper_indices, ee_links, init_qpos
              (gripper_idx1/gripper_idx2/ee_link 保留用于单侧向后兼容)
    """
    sides = ["left", "right"] if side == "both" else [side]
    strip_visuals = not getattr(scene, "_render_available", True)

    if mode == "full_robot":
        urdf_path = prepare_full_robot_urdf(FULL_ROBOT_URDF, side, strip_visuals=strip_visuals)
    else:
        if side == "both":
            logger.warning("  gripper_only + both 未完全支持, 仅加载右侧夹爪")
            urdf_path = prepare_gripper_only_urdf("right", strip_visuals=strip_visuals)
        else:
            urdf_path = prepare_gripper_only_urdf(side, strip_visuals=strip_visuals)

    loader = scene.create_urdf_loader()
    # gripper_only: fix_root_link=True, 虚拟 6-DOF 关节提供移动能力 (对齐 05_gripper_test.py)
    # full_robot: fix_root_link=False, 浮动根 (free joint 6 DOF) + monkey-patch
    if mode == "gripper_only":
        loader.fix_root_link = True
    else:
        loader.fix_root_link = False
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(urdf_path)

    active_joints = robot.get_active_joints()
    joint_names = [j.name for j in active_joints]

    _joint_debug = [(j.name, j.get_type(), j.get_dof()) for j in active_joints]
    logger.info(f"  [debug] active_joints: {_joint_debug}")
    logger.info(f"  [debug] fix_root_link={loader.fix_root_link}, total qpos size={len(robot.get_qpos())}, dof={robot.get_dof()}")

    # 计算 qpos 索引映射
    qpos_starts = {}
    _qpos_offset = 0
    for _ji, _joint in enumerate(active_joints):
        qpos_starts[_ji] = _qpos_offset
        _jt = _joint.get_type()
        if _jt == 'free':
            _qpos_offset += 7  # free joint: 3 pos + 4 quat
        else:
            _qpos_offset += _joint.get_dof()  # revolute/prismatic: 1
    has_floating_root = any(j.get_type() == 'free' for j in active_joints)

    # gripper_only: 虚拟关节索引 + PD 驱动 (对齐 05_gripper_test.py)
    virtual_idx = {}
    if mode == "gripper_only":
        virtual_idx = {
            'vx': joint_names.index("virtual_x"),
            'vy': joint_names.index("virtual_y"),
            'vz': joint_names.index("virtual_z"),
            'rz': joint_names.index("virtual_rz"),
            'ry': joint_names.index("virtual_ry"),
            'rx': joint_names.index("virtual_rx"),
        }
        VIRTUAL_STIFFNESS = 5000.0  # v4.17: 对齐 test8 (K=5000/D=1000, 纯 PD 无漂移)
        VIRTUAL_DAMPING = 1000.0
        for vkey in ['vx', 'vy', 'vz', 'rz', 'ry', 'rx']:
            active_joints[virtual_idx[vkey]].set_drive_property(
                stiffness=VIRTUAL_STIFFNESS, damping=VIRTUAL_DAMPING)
        logger.info(f"  虚拟关节 PD: stiffness={VIRTUAL_STIFFNESS}, damping={VIRTUAL_DAMPING} (纯 PD, 无 vlock)")
        if has_floating_root:
            logger.info(f"  浮动根: free joint qpos 占 7 槽, 后续 joint qpos 索引偏移")
    else:
        if has_floating_root:
            logger.info(f"  浮动根: free joint qpos 占 7 槽, 后续 joint qpos 索引偏移")
        else:
            logger.warning(f"  [警告] 浮动根未启用! fix_root_link=False 但 active_joints 中无 free joint")
            logger.warning(f"  [警告] 可能 SAPIEN URDF loader 未添加 root free joint, 需检查 URDF 结构")

    # 选择所有侧的臂关节和夹爪关节
    arm_joint_indices = []
    gripper_indices = {}  # {"left": (idx1, idx2), "right": (idx1, idx2)}
    for s in sides:
        arm_joint_indices += [i for i, n in enumerate(joint_names) if f"{s}_arm_joint" in n]
        gi1 = gi2 = None
        for i, n in enumerate(joint_names):
            if f"{s}_gripper_finger_joint1" == n:
                gi1 = i
            elif f"{s}_gripper_finger_joint2" == n:
                gi2 = i
        gripper_indices[s] = (gi1, gi2)

    # 所有夹爪关节索引集合 (用于 PD 参数设置)
    all_gripper_idxs = set()
    for gi1, gi2 in gripper_indices.values():
        if gi1 is not None:
            all_gripper_idxs.add(gi1)
        if gi2 is not None:
            all_gripper_idxs.add(gi2)

    # PD 驱动参数
    for i, joint in enumerate(active_joints):
        if i in all_gripper_idxs:
            joint.set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING,
                                     force_limit=GRIPPER_FORCE)
        else:
            joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)

    # 初始关节角
    init_qpos = robot.get_qpos().copy()
    if mode == "gripper_only":
        # 虚拟关节初始位置 = base_pos, 姿态 = base_quat → ZYX Euler
        init_qpos[virtual_idx['vx']] = float(base_pos[0])
        init_qpos[virtual_idx['vy']] = float(base_pos[1])
        init_qpos[virtual_idx['vz']] = float(base_pos[2])
        base_R = pr.matrix_from_quaternion(base_quat)[:3, :3]
        rz, ry, rx = rotmat_to_zyx_euler(base_R)
        init_qpos[virtual_idx['rz']] = float(rz)
        init_qpos[virtual_idx['ry']] = float(ry)
        init_qpos[virtual_idx['rx']] = float(rx)
    for s in sides:
        s_arm_indices = [i for i, n in enumerate(joint_names) if f"{s}_arm_joint" in n]
        starting = RIGHT_ARM_STARTING if s == "right" else LEFT_ARM_STARTING
        for j, idx in enumerate(s_arm_indices):
            if j < len(starting):
                init_qpos[qpos_starts[idx]] = starting[j]
        gi1, gi2 = gripper_indices[s]
        if gi1 is not None:
            init_qpos[qpos_starts[gi1]] = GRIPPER_INIT_OPEN
        if gi2 is not None:
            # full_robot URDF: 两手指 axis 相同, joint2 用负值对称张开
            # gripper_only URDF: 两手指 axis 相反 (axis="0 1 0" vs "0 -1 0"),
            #   joint1/joint2 都是正值表示张开 (对齐 04_physics_simulation.py L2320-2321)
            if mode == "gripper_only":
                init_qpos[qpos_starts[gi2]] = GRIPPER_INIT_OPEN
            else:
                init_qpos[qpos_starts[gi2]] = -GRIPPER_INIT_OPEN
    robot.set_qpos(init_qpos)

    # 关键: set_drive_target 与 set_qpos 一致, 否则 PD 拉向零位
    # 浮动根的 free joint 跳过 (由 set_root_pose + set_root_linear_velocity 驱动)
    for i, joint in enumerate(active_joints):
        if joint.get_type() == 'free':
            continue
        joint.set_drive_target(init_qpos[qpos_starts[i]])

    if mode == "full_robot":
        # full_robot: 浮动根, 通过 set_root_pose + set_root_linear_velocity 驱动
        robot.set_root_pose(sapien.Pose(np.asarray(base_pos).tolist(), np.asarray(base_quat).tolist()))
        # 禁用所有 robot link 的重力, 使 set_root_linear_velocity 生效.
        for link in robot.get_links():
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    component.set_disable_gravity(True)
        logger.info(f"  已禁用所有 robot link 重力 (set_disable_gravity=True), "
                    f"根由 set_root_pose + set_root_linear_velocity 驱动")
    else:
        # gripper_only: 也禁用重力, 防止手指 PD 反作用力 + 重力推偏虚拟关节.
        # 虚拟关节 PD 刚度 5000 N/m 足以对抗 0.6kg 夹爪重力 (0.6×9.8=5.9N, 5.9/5000=1.2mm 偏差),
        # 但 8 个物理子步的累积漂移可能很大, 禁用重力更稳定.
        for link in robot.get_links():
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    component.set_disable_gravity(True)
        logger.info(f"  gripper_only: 已禁用所有 robot link 重力 (虚拟关节 PD 驱动, 无重力干扰)")

    # 夹爪摩擦对齐 GalaxeaManipSim R1 (static=1.0, dynamic=1.0, restitution=0.6)
    gripper_material = scene.create_physical_material(
        static_friction=GRIPPER_FRICTION, dynamic_friction=GRIPPER_FRICTION, restitution=0.6
    )
    touch_link_names = []
    for s in sides:
        touch_link_names += [f"{s}_gripper_finger_link1", f"{s}_gripper_finger_link2"]
    for link in robot.get_links():
        if link.get_name() in touch_link_names:
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    for cs in component.get_collision_shapes():
                        cs.set_physical_material(gripper_material)

    # 碰撞处理: gripper_only 和 full_robot 不同
    if mode == "gripper_only":
        # 对齐 05_gripper_test.py: 禁用夹爪内部碰撞 (手指-手指, 手指-夹爪本体)
        # 虚拟关节 link 碰撞也禁用 (避免虚拟臂与地面/物体碰撞)
        finger_ignore_bit = 1 << 0
        finger_ignore_id = 1
        finger_link_names = set()
        for s in sides:
            finger_link_names |= {
                f"{s}_gripper_finger_link1",
                f"{s}_gripper_finger_link2",
                f"{s}_gripper_link",
            }
        # 夹爪内部: 用碰撞组过滤 (手指和夹爪本体之间互不碰撞, 但仍与外部物体碰撞)
        for link in robot.get_links():
            if link.get_name() in finger_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            g = list(cs.get_collision_groups())
                            g[2] |= finger_ignore_bit
                            g[3] = finger_ignore_id
                            cs.set_collision_groups(g)
            else:
                # 虚拟关节 link: 完全禁用碰撞 (不与任何物体碰撞)
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            cs.set_collision_groups([0, 0, 0, 0])
        logger.info(f"  gripper_only: 夹爪内部碰撞已过滤, 虚拟 link 碰撞已禁用")
    else:
        # full_robot: 禁用非夹爪 link 的碰撞 (ROOT 可能在地下, 躯干与地面碰撞会导致关节爆炸)
        collision_link_names = set()
        for s in sides:
            collision_link_names |= {
                f"{s}_gripper_finger_link1",
                f"{s}_gripper_finger_link2",
            }
        n_disabled = 0
        for link in robot.get_links():
            if link.get_name() not in collision_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            cs.set_collision_groups([0, 0, 0, 0])
                        n_disabled += 1
        if n_disabled > 0:
            logger.info(f"  禁用 {n_disabled} 个非夹爪 link 的碰撞 (避免躯干-地面碰撞)")

    # EE links (每侧一个)
    ee_links = {}
    for s in sides:
        for link in robot.get_links():
            if f"{s}_gripper_link" == link.get_name():
                ee_links[s] = link
                break

    n_gripper_total = sum(1 for gi1, gi2 in gripper_indices.values() if gi1 is not None) * 2
    logger.info(f"  机器人已加载: mode={mode}, side={side}, {len(arm_joint_indices)} 臂关节 + "
                f"{n_gripper_total} 夹爪关节")
    logger.info(f"    PD: arm stiffness={JOINT_STIFFNESS}, damping={JOINT_DAMPING}")
    logger.info(f"    夹爪摩擦: static={GRIPPER_FRICTION}, dynamic={GRIPPER_FRICTION}, restitution=0.6")

    # 向后兼容: 单侧时 gripper_idx1/gripper_idx2/ee_link
    first_side = sides[0]
    return {
        "robot": robot,
        "joint_names": joint_names,
        "arm_joint_indices": arm_joint_indices,
        "gripper_indices": gripper_indices,  # {"left": (i1,i2), "right": (i1,i2)}
        "gripper_idx1": gripper_indices[first_side][0],  # 向后兼容
        "gripper_idx2": gripper_indices[first_side][1],
        "ee_links": ee_links,  # {"left": link, "right": link}
        "ee_link": ee_links.get(first_side),  # 向后兼容
        "init_qpos": init_qpos,
        "qpos_starts": qpos_starts,  # active_joint list index → qpos index 映射
        "has_floating_root": has_floating_root,  # True 时根为 free joint (6 DOF)
        "virtual_idx": virtual_idx,  # gripper_only: 虚拟关节索引 {vx,vy,vz,rz,ry,rx}
    }


# ============================================================
# 4. 物理步进 (参考 04 _physics_step — 纯 PD 驱动)
# ============================================================
def _is_floating_root(robot):
    """检测 articulation 是否为浮动根 (root free joint).

    fix_root_link=False 时, 第一个 active joint 是 free joint (6 DOF).
    用于在 physics_step 中屏蔽根的广义力, 避免与 set_root_pose 冲突.
    """
    active_joints = robot.get_active_joints()
    if not active_joints:
        return False
    return active_joints[0].get_type() == 'free'


def physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                 arm_target, gripper_target1, gripper_target2, scene,
                 extra_gripper_indices=None, extra_arm_indices=None, extra_arm_target=None,
                 lock_root_pose=None, decimation=None,
                 virtual_lock_targets=None):
    """纯 PD 驱动 + 重力补偿 + decimation (对齐 GalaxeaManipSim / 05_gripper_test.py)

    关键: 不调用 set_qpos! set_qpos + set_drive_target 双重控制会导致震荡.
    extra_gripper_indices: 双手模式第二侧夹爪 [(idx, target), ...]
    extra_arm_indices/extra_arm_target: 双手模式第二侧臂关节索引和目标
    lock_root_pose: 向后兼容参数 (gripper_only 虚拟关节模式不再使用, 传 None).
        full_robot 浮动根模式仍使用此参数进行根锁定.
    decimation: 物理子步数, None 时使用全局 DECIMATION (=8).
    virtual_lock_targets: dict {joint_idx: target_value} — 虚拟关节运动学锁定.
        每个 scene.step() 后强制 set_qpos + set_qvel=0, 防止 PD 漂移.
        gripper_only 模式下 virtual_z 等 3cm 漂移的根因修复.
    """
    if decimation is None:
        decimation = DECIMATION
    active_joints = robot.get_active_joints()
    for i, idx in enumerate(arm_joint_indices):
        if i < len(arm_target):
            active_joints[idx].set_drive_target(float(arm_target[i]))
    if gripper_idx1 is not None:
        active_joints[gripper_idx1].set_drive_target(float(gripper_target1))
    if gripper_idx2 is not None:
        active_joints[gripper_idx2].set_drive_target(float(gripper_target2))
    if extra_arm_indices is not None and extra_arm_target is not None:
        for i, idx in enumerate(extra_arm_indices):
            if i < len(extra_arm_target):
                active_joints[idx].set_drive_target(float(extra_arm_target[i]))
    if extra_gripper_indices is not None:
        for idx, target in extra_gripper_indices:
            if idx is not None:
                active_joints[idx].set_drive_target(float(target))

    for _ in range(decimation):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        # 浮动根: 屏蔽根 free joint 的广义力.
        # 原因: compute_passive_force 会为根 free joint 计算重力补偿,
        # 与 set_root_pose (teleport) 冲突, 导致根漂移.
        if _is_floating_root(robot):
            qf[:6] = 0

        # 弹簧力 + 阻尼力 (仅 full_robot 浮动根模式使用).
        # gripper_only 虚拟关节模式: lock_root_pose=None, 跳过此块.
        if lock_root_pose is not None:
            target_pos = np.array(lock_root_pose.p)
            current_pos = np.array(robot.get_root_pose().p)
            _spring_k = 200000.0   # 弹簧刚度 N/m, 足够大以抵抗手指 PD 反作用力 (≤20N)
            _damping_k = 5000.0    # 阻尼系数 N·s/m, 抑制振荡
            pos_err = current_pos - target_pos
            qvel = robot.get_qvel()
            spring_force = -_spring_k * pos_err
            damping_force = -_damping_k * qvel[:3]
            # 旋转弹簧: 用四元数乘积计算 angle-axis 误差
            target_quat = np.array(lock_root_pose.q)
            current_quat = np.array(robot.get_root_pose().q)
            _spring_k_rot = 100000.0
            # Quaternion product: q1 * q2 = (w1*w2-v1·v2, w1*v2+w2*v1+v1×v2)
            w1, v1 = target_quat[0], target_quat[1:4]
            w2, v2 = current_quat[0], current_quat[1:4]
            quat_err_v = w1 * v2 + w2 * v1 + np.cross(v1, v2)
            angle_axis = 2.0 * quat_err_v
            spring_torque = -_spring_k_rot * angle_axis
            damping_torque = -_damping_k * qvel[3:6]

            # 对浮动根: 写入 qf (会覆盖 compute_passive_force 的重力补偿)
            if _is_floating_root(robot):
                qf[0:3] = spring_force + damping_force
                qf[3:6] = spring_torque + damping_torque

        robot.set_qf(qf)

        # PRE-step teleport: 仅 full_robot 浮动根模式.
        # gripper_only 虚拟关节: lock_root_pose=None, 跳过.
        if lock_root_pose is not None:
            robot.set_root_pose(lock_root_pose)
            robot.set_root_linear_velocity([0.0, 0.0, 0.0])
            robot.set_root_angular_velocity([0.0, 0.0, 0.0])

        scene.step()

        # POST-step teleport: 仅 full_robot 浮动根模式.
        if lock_root_pose is not None:
            robot.set_root_pose(lock_root_pose)
            robot.set_root_linear_velocity([0.0, 0.0, 0.0])
            robot.set_root_angular_velocity([0.0, 0.0, 0.0])
            _actual_pos = robot.get_root_pose().p
            _target_pos = lock_root_pose.p
            _drift = np.linalg.norm(np.array(_actual_pos) - np.array(_target_pos))
            if _drift > 0.001:
                logger.warning(f"  [WARNING] teleport 后仍漂移! drift={_drift:.6f}m, "
                               f"target={np.array(_target_pos).round(4)}, "
                               f"actual={np.array(_actual_pos).round(4)}")


# ============================================================
# 5. 接触检测 (参考 04 _fetch_contacts)
# ============================================================
def fetch_contacts(robot, obj_actors, side, scene):
    """检测夹爪与物体之间的接触

    Args:
        side: "left" / "right" / "both"

    Returns:
        tuple: (total_contacts, total_impulse, per_obj)
    """
    # side="both" 时检查两侧夹爪
    sides = ["left", "right"] if side == "both" else [side]
    # 所有含 side + gripper 的 link 都计入 (手指、掌心等)
    gripper_bodies = set()
    for link in robot.get_links():
        name = link.get_name()
        if any(s in name for s in sides) and "gripper" in name:
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    gripper_bodies.add(component)

    obj_bodies = {}
    for actor in obj_actors:
        for component in actor.components:
            if isinstance(component, (
                sapien.pysapien.physx.PhysxRigidDynamicComponent,
                sapien.pysapien.physx.PhysxRigidStaticComponent,
            )):
                obj_bodies[component] = actor.name
                break

    contacts = scene.get_contacts()
    total_contacts = 0
    total_impulse = 0.0
    per_obj = {}
    for c in contacts:
        b0, b1 = c.bodies[0], c.bodies[1]
        is_gripper_obj = (b0 in gripper_bodies and b1 in obj_bodies) or \
                         (b1 in gripper_bodies and b0 in obj_bodies)
        if not is_gripper_obj:
            continue
        total_contacts += 1
        impulse = np.array([p.impulse for p in c.points])
        imp_norm = 0.0
        if len(impulse) > 0:
            imp_norm = float(np.linalg.norm(np.sum(impulse, axis=0)))
        total_impulse += imp_norm
        obj_body = b1 if b0 in gripper_bodies else b0
        obj_name = obj_bodies.get(obj_body, "unknown")
        if obj_name not in per_obj:
            per_obj[obj_name] = {"n": 0, "impulse": 0.0}
        per_obj[obj_name]["n"] += 1
        per_obj[obj_name]["impulse"] += imp_norm

    return total_contacts, total_impulse, per_obj


# ============================================================
# 7b. 辅助函数: 接触检测 + 力获取 + 物体在夹爪内判断
# ============================================================
def get_finger_contacts(robot, side, scene, obj_actors):
    """获取夹爪两手指与 GLB 物体的接触信息

    对齐 04_physics_simulation.py _fetch_contacts (L1941-1995) 的筛选方式:
    用 PhysxArticulationLinkComponent / PhysxRigidDynamicComponent 做 body 匹配

    Args:
        robot: SAPIEN robot
        side: "left" / "right"
        scene: SAPIEN scene
        obj_actors: GLB 物体 actor (Entity) 列表

    Returns:
        (finger1_contact: bool, finger2_contact: bool, contact_obj_names: list[str])
    """
    finger_names = {
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
        f"{side}_gripper_link",
    }
    # 收集夹爪 link 的 PhysxArticulationLinkComponent (04 L1958-1963)
    gripper_bodies = set()
    finger_body_map = {}  # component → link_name
    for link in robot.get_links():
        if link.get_name() in finger_names:
            for comp in link.entity.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    gripper_bodies.add(comp)
                    finger_body_map[comp] = link.get_name()

    # 收集 GLB 物体的 PhysxRigidDynamicComponent (04 L1965-1970)
    obj_bodies = {}
    for actor in obj_actors:
        for comp in actor.components:
            if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                obj_bodies[comp] = actor.name
                break

    f1_contact = False
    f2_contact = False
    contact_objs = set()

    for c in scene.get_contacts():
        b0, b1 = c.bodies[0], c.bodies[1]
        is_gripper_obj = (b0 in gripper_bodies and b1 in obj_bodies) or \
                         (b1 in gripper_bodies and b0 in obj_bodies)
        if not is_gripper_obj:
            continue
        # 确定哪个是夹爪 component
        gripper_comp = b0 if b0 in gripper_bodies else b1
        obj_comp = b1 if b0 in gripper_bodies else b0
        link_name = finger_body_map.get(gripper_comp, "")
        if "finger_link1" in link_name:
            f1_contact = True
        elif "finger_link2" in link_name:
            f2_contact = True
        elif "gripper_link" in link_name:
            f1_contact = True
            f2_contact = True
        obj_name = obj_bodies.get(obj_comp, "")
        if obj_name:
            contact_objs.add(obj_name)

    return f1_contact, f2_contact, list(contact_objs)


def get_grasp_force(side, scene, obj_actors, robot):
    """获取夹爪当前夹紧力 (N)

    对齐 04 的 body 匹配方式, 通过 impulse/dt 计算

    Args:
        side: "left" / "right"
        scene: SAPIEN scene
        obj_actors: GLB 物体 actor 列表
        robot: SAPIEN robot

    Returns:
        float: 两手指总夹紧力 (N)
    """
    finger_names = {
        f"{side}_gripper_finger_link1",
        f"{side}_gripper_finger_link2",
        f"{side}_gripper_link",
    }
    gripper_bodies = set()
    for link in robot.get_links():
        if link.get_name() in finger_names:
            for comp in link.entity.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    gripper_bodies.add(comp)

    obj_bodies = set()
    for actor in obj_actors:
        for comp in actor.components:
            if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                obj_bodies.add(comp)
                break

    dt = PHYSICS_TIMESTEP * DECIMATION
    total_force = 0.0
    for c in scene.get_contacts():
        b0, b1 = c.bodies[0], c.bodies[1]
        is_gripper_obj = (b0 in gripper_bodies and b1 in obj_bodies) or \
                         (b1 in gripper_bodies and b0 in obj_bodies)
        if is_gripper_obj:
            for pt in c.points:
                impulse_norm = float(np.linalg.norm(pt.impulse))
                total_force += impulse_norm / max(dt, 1e-8)
    return total_force


def is_obj_in_gripper_frame(gripper_pos, gripper_R, obj_pos,
                            forward_threshold=0.08, lateral_threshold=0.04):
    """判断物体是否在夹爪前方 (即将进入两指之间)

    Args:
        gripper_pos: 夹爪世界位置 (3,)
        gripper_R: 夹爪旋转矩阵 (3,3)
        obj_pos: 物体世界位置 (3,)
        forward_threshold: 前方距离阈值 (m)
        lateral_threshold: 横向距离阈值 (m)

    Returns:
        bool: 物体是否在夹爪即将抓取的位置
    """
    obj_in_gripper = gripper_R.T @ (obj_pos - gripper_pos)
    # 夹爪 URDF: +X 前方, ±Y 两指方向
    return (0 < obj_in_gripper[0] < forward_threshold and
            abs(obj_in_gripper[1]) < lateral_threshold)


__all__ = [
    # 路径常量
    "PROJECT_ROOT", "GALAXEA_SIM_PATH", "R1_ASSETS", "R1_MESH_DIR", "R1_URDF_DIR",
    "FULL_ROBOT_URDF", "R1_RIGHT_SETTINGS", "R1_LEFT_SETTINGS", "COMBINATION_DIR",
    "compute_and_save_transform_params",
    # 坐标变换
    "R_x", "R_AXIS", "RXWORLD_TO_SAPIEN",
    # 机器人参数
    "RIGHT_ARM_STARTING", "LEFT_ARM_STARTING", "ARM_MAX_REACH", "COMFORTABLE_REACH",
    "BASE_BACK_OFFSET", "ARM_BASE_OFFSET_RIGHT", "ARM_BASE_OFFSET_LEFT",
    "COMFORT_TARGET_IN_BASE", "RIGHT_ARM_JOINT_LIMITS",
    # 物理参数
    "JOINT_STIFFNESS", "JOINT_DAMPING", "GRIPPER_STIFFNESS", "GRIPPER_DAMPING",
    "PHYSICS_TIMESTEP", "CONTROL_FREQ", "DECIMATION", "GROUND_HEIGHT",
    "OBJECT_DENSITY", "OBJECT_MIN_MASS", "GRIPPER_FRICTION",
    "GRIPPER_INIT_OPEN", "GRIPPER_MAX_OPEN", "GRIPPER_CLOSE_BIAS", "GRIPPER_FORCE",
    "_DESCEND_OPEN", "_GRIP_CLOSE", "_GRIP_OVERCLOSURE",
    "GRASP_STRATEGIES", "FINGER_FORWARD_NEUTRAL", "MAX_ROOT_STEP",
    "_FINGER1_ORIGIN", "_FINGER1_AXIS", "_FINGER2_ORIGIN", "_FINGER2_AXIS",
    "FINGER_BASE_DIST", "GRIPPER_FINGER_HALF_THICKNESS", "FINGER_EFFECTIVE_HALF_SPACING",
    # 内部状态
    "_FAILED_SCENES",
    # 函数
    "rotmat_to_zyx_euler",
    "prepare_full_robot_urdf", "prepare_gripper_only_urdf",
    "setup_physics_scene", "setup_robot",
    "physics_step", "fetch_contacts",
    "get_finger_contacts", "get_grasp_force", "is_obj_in_gripper_frame",
    "_is_floating_root",
    # logger
    "logger",
]
