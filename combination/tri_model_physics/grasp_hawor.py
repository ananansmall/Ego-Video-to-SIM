"""grasp_hawor.py — SAPIEN 物理仿真: 用 R1 机器人真实抓取 GLB 物体

项目目标:
  给定 HaWoR 手部重建 + RAS 场景重建 (GLB), 用 R1 机器人 URDF 在 SAPIEN 中
  复刻抓取 GLB 物体的动作, 并通过参数级验证 (物体提升/接触检测).

两种 URDF 模式 (用户要求 "两种状态都要"):
  1. full_robot  : r1_v2_1_0.urdf (整个机器人, 臂关节 fixed→revolute)
                   轨迹: DexRetargeting(夹爪) + RelaxedIK(臂) + 纯PD驱动
  2. gripper_only: 纯夹爪 URDF (无机械臂, 解析映射)
                   轨迹: MANO 指尖向量 → 夹爪位姿 + 手指关节角

关键修复 (旧版 "没有机械臂" 的根因):
  r1_v2_1_0.urdf 中 <joint name="..." type="fixed"> 跨多行, 旧正则
  要求 name/type 同行导致匹配失败, 臂关节保持 fixed → "0臂关节".
  本脚本用 re.DOTALL + \s+ (匹配换行) 修复.

参考:
  - 04_physics_simulation.py : PhysicsSimulator (纯PD驱动, 接触检测)
  - 01_align_scene.py        : compute_and_save_transform_params (对齐)
  - GalaxeaManipSim          : SAPIEN 物理参数 (stiffness=1000, damping=200)

用法:
  python grasp_hawor.py --mode full_robot \\
      --hawor-dir /home/an/data/hawor/7 \\
      --ras-dir /home/an/data/ras/my_7mp4_result

  python grasp_hawor.py --mode gripper_only \\
      --hawor-dir /home/an/data/hawor/7 \\
      --ras-dir /home/an/data/ras/my_7mp4_result
"""

import os
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
_intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
if 'VK_ICD_FILENAMES' not in os.environ:
    if os.path.exists(_nvidia_icd):
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            os.environ['VK_ICD_FILENAMES'] = _nvidia_icd if r.returncode == 0 else _intel_icd
        except Exception:
            os.environ['VK_ICD_FILENAMES'] = _intel_icd
    else:
        os.environ['VK_ICD_FILENAMES'] = _intel_icd

import sys
import re
import gc
import json
import logging
import tempfile
import argparse
from pathlib import Path

import numpy as np
import cv2
import sapien
import sapien.render
import torch
from pytransform3d import rotations as pr
from tqdm import trange

try:
    import trimesh
except ImportError:
    trimesh = None

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
from sapien.wrapper.urdf_loader import URDFLoader as _URDFLoader
from sapien.wrapper.articulation_builder import (
    LinkBuilder as _LinkBuilder,
    ArticulationBuilder as _ArticulationBuilder,
)

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
import importlib.util
_ALIGN_SCRIPT = COMBINATION_DIR / "01_align_scene.py"
_ALIGN_SPEC = importlib.util.spec_from_file_location("align_scene", str(_ALIGN_SCRIPT))
_ALIGN_MOD = importlib.util.module_from_spec(_ALIGN_SPEC)
_ALIGN_SPEC.loader.exec_module(_ALIGN_MOD)
compute_and_save_transform_params = _ALIGN_MOD.compute_and_save_transform_params

# ============ 坐标变换 (与 02/04 一致) ============
R_x = np.diag([1.0, -1.0, -1.0])
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
GRIPPER_STIFFNESS = 5000.0  # 第九轮: K=20000 把物体推开 6mm (冲击太大); K=5000 PD 力 30N 温和夹持.
                              # 慢速 LIFT (3mm/帧) 让物体有时间通过摩擦力跟上 pad.
GRIPPER_DAMPING = 200.0
PHYSICS_TIMESTEP = 1 / 240.0
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))  # =8
GROUND_HEIGHT = 0
OBJECT_DENSITY = 1000.0  # 基础惯性变量: 所有物体统一密度 (kg/m³), 对齐水密度, 用户: "基础的惯性变量"
OBJECT_MIN_MASS = 0.15   # 基础惯性变量: 物体质量下限 (kg), 防止轻物被 kinematic 根弹飞
GRIPPER_FRICTION = 2.0  # 夹爪摩擦 (第五轮: 1.0→2.0, 增大摩擦防止物体在夹爪内打滑被甩飞)
GRIPPER_INIT_OPEN = 0.04
GRIPPER_MAX_OPEN = 0.05
MAX_ROOT_STEP = 0.008  # 根速度限制: 每帧 ≤ 0.8cm (盘子翻转临界点 0.008-0.010, 仅 0.008 让盘子不翻; 根误差增大是物理稳定代价)

# 夹爪几何 (与 04/hand_track 一致)
_FINGER1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689, 0.013453, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])
FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906

# IK / 平滑
IK_SOLVE_PER_FRAME = 20
IK_TOLERANCES = [0.1] * 6
LP_ALPHA_JOINT = 0.5
WARMUP_FRAMES = 30

# 渲染
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
HAWOR_FOCAL_DEFAULT = 600.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("grasp_hawor")


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
    """生成纯夹爪 URDF (无机械臂, 与 04_physics_simulation.py 一致)

    结构: gripper_base_link (固定根) → gripper_link → finger_link1/2 (prismatic)
    Args:
        side: "right" / "left"
        strip_visuals: True 时移除 <visual> 块 (CPU 降级模式)
    """
    template = """<?xml version="1.0" encoding="utf-8"?>
<robot name="r1_gripper_{prefix}">
  <link name="{prefix}_gripper_base_link">
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="0.01"/>
    <inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/></inertial>
  </link>
  <joint name="{prefix}_gripper_base_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="{prefix}_gripper_base_link"/>
    <child link="{prefix}_gripper_link"/>
  </joint>
  <link name="{prefix}_gripper_link">
    <inertial><origin xyz="-0.031107240301242 -1.38928815840433E-07 -1.43700425780935E-07" rpy="0 0 0"/>
    <mass value="0.604"/>
    <inertia ixx="0.000175880119550986" ixy="4.17894263577595E-10" ixz="-5.34925118595879E-10"
             iyy="9.86374067070897E-05" iyz="-8.18555544397352E-08" izz="0.000165120109045834"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/></geometry></collision>
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
             iyy="5.71082134562374E-06" iyz="6.19457183851545E-08" izz="6.4848556091919E-06"/></inertial>
    <visual><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/></geometry>
      <material name=""><color rgba="0.823529411764706 0.823529411764706 1 1"/></material></visual>
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.012 0.020 0.024"/></geometry></collision>
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
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.012 0.020 0.024"/></geometry></collision>
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
# 渲染失败的 scene 对象 (CPU 降级时保持引用避免 __del__ 段错误)
_FAILED_SCENES = []

def setup_physics_scene(ground_height=GROUND_HEIGHT):
    """创建 SAPIEN 物理场景

    对齐 02_render_scene.py 的渲染设置 + GalaxeaManipSim 的物理地面
    地面高度根据 GLB 物体最低点动态调整 (02 没有地面, 这里补上)

    CPU 降级修复: sapien.Scene() 渲染设备不可用时, C++ 渲染系统残留状态会导致
    后续 sapien.Scene(systems=[PhysxCpuSystem()]) 段错误 (SIGSEGV).
    用 nvidia-smi 预检测: GPU 不可用时直接用 CPU scene, 不创建失败的渲染 scene.
    """
    import subprocess, shutil
    # 预检测 GPU 是否可用 (避免 sapien.Scene() 失败后残留状态段错误)
    # 用 shutil.which 找完整路径 + 增大 timeout (sandbox 中 subprocess 可能 PATH 不同/慢)
    gpu_ok = False
    nvidia_smi = shutil.which('nvidia-smi') or '/usr/bin/nvidia-smi'
    try:
        r = subprocess.run([nvidia_smi], capture_output=True, timeout=15)
        # returncode=9 (ECC 错误) 时 GPU 仍可用, 用 stdout 非空判断 (含驱动信息)
        gpu_ok = (r.returncode == 0 or len(r.stdout) > 50)
    except Exception:
        gpu_ok = False

    render_available = False
    if gpu_ok:
        try:
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
            sapien.render.set_ray_tracing_samples_per_pixel(16)
            scene = sapien.Scene()
            render_available = True
        except RuntimeError as e:
            if "rendering device" in str(e).lower():
                logger.warning("  SAPIEN 渲染设备不可用, 降级为纯物理场景")
                scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
            else:
                raise
    else:
        logger.info("  NVIDIA GPU 不可用 (nvidia-smi 失败), 使用 CPU 物理场景")
        scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])

    scene.set_timestep(PHYSICS_TIMESTEP)

    if render_available:
        try:
            from sapien.asset import create_dome_envmap
            # 对齐 02_render_scene.py 的环境光照
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
    ground_material = sapien.physx.PhysxMaterial(
        static_friction=0.5, dynamic_friction=0.5, restitution=0.0
    )
    # render=False 在 CPU 降级模式下跳过 RenderMaterial 创建 (避免 RuntimeError)
    ground_actor = scene.add_ground(
        ground_height, render=render_available, material=ground_material
    )
    if render_available:
        try:
            # SAPIEN 新版 API: 通过 RenderBodyComponent 控制视觉
            n_hidden = 0
            for c in ground_actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    c.disable()
                    n_hidden += 1
            if n_hidden > 0:
                logger.info(f"  地面视觉已隐藏 ({n_hidden} 个 RenderBody 禁用, 保留物理碰撞), 避免遮挡地下机器人身体")
            else:
                logger.info("  地面无 RenderBodyComponent (可能已透明)")
        except Exception as e:
            logger.warning(f"  无法隐藏地面视觉: {e}, 机器人身体可能被遮挡")
    else:
        logger.info("  地面物理已加载 (无视觉, CPU 模式)")
    scene._render_available = render_available
    return scene


def hawor_cam_to_sapien_pose(R_c2w, t_c2w):
    """将 HaWoR 相机位姿转换为 SAPIEN 相机位姿 (对齐 02_render_scene.py L1046)

    变换链:
      1. p_sapien = RXWORLD_TO_SAPIEN @ t_c2w
      2. OpenGL 约定 (Z=后方) → SAPIEN 相机约定 (Z=上方)
    """
    cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w
    cam_R_sapien = RXWORLD_TO_SAPIEN @ R_c2w

    forward = -cam_R_sapien[:, 2]
    left = -cam_R_sapien[:, 0]
    up = cam_R_sapien[:, 1]

    sapien_cam_R = np.eye(3)
    sapien_cam_R[:, 0] = forward
    sapien_cam_R[:, 1] = left
    sapien_cam_R[:, 2] = up

    if np.linalg.det(sapien_cam_R) < 0:
        U, _, VH = np.linalg.svd(sapien_cam_R)
        sapien_cam_R = U @ VH
    cam_quat = pr.quaternion_from_matrix(sapien_cam_R)
    return cam_pos_sapien, cam_quat


def make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
    """计算 look-at 相机姿态四元数 (对齐 02_render_scene.py L1016)"""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    else:
        right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_R = np.eye(3)
    cam_R[:, 0] = forward
    cam_R[:, 1] = -right
    cam_R[:, 2] = cam_up
    cam_quat = pr.quaternion_from_matrix(cam_R)
    return cam_quat


def compute_glb_ground_z(glb_path, transform_params_path):
    """预扫描 GLB, 返回 SAPIEN 坐标系下最低点 z (用于设置地面高度)

    变换链与 load_glb_with_physics / 02_render_scene.py 严格一致:
      p_hawor = s_inv * (R_inv @ p_ras) + t_inv
      p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
    """
    if trimesh is None:
        return 0.0
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    trimesh_scene = trimesh.load(str(glb_path))
    all_min_z = []
    for geom_name, geom in trimesh_scene.geometry.items():
        vertices = geom.vertices.copy()
        if len(vertices) == 0:
            continue
        vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
        vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
        all_min_z.append(float(vertices_sapien[:, 2].min()))
    return min(all_min_z) if all_min_z else 0.0


def load_glb_with_physics(glb_path, transform_params_path, scene, fast_collision=True):
    """加载 GLB 场景并创建带碰撞体的物理物体

    对齐 02_render_scene.py 的加载方式:
      - 顶点变换后不居中, 直接导出 PLY (顶点已在世界坐标系)
      - 大型扁平几何体 → kinematic, 小物体 → dynamic

    关键修复 (用户: "像 grasp_demo.py 一样真正的抓取物体"):
      - 加载后把物体直接放在地面上 (z_min = ground_z), 避免 dynamic 物体掉落
      - 之前物体悬浮在地面上方 3-5cm, 物理仿真开始后掉落, 但 obj_bbox_centers 还是悬浮位置,
        导致 _compute_grasp_demo_target 用悬浮位置算 grasp_pos, EE 在错误高度!

    返回: (obj_actors, ground_z, obj_bbox_centers, obj_info)
          ground_z=GLB物体最低点, obj_bbox_centers=每个物体包围盒中心(已对齐到地面)
          obj_info=每个物体的颜色和几何信息 (用于颜色/语义识别粉色物体和碗)
    """
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']

    if trimesh is None:
        logger.error("  trimesh 未安装, 无法加载 GLB")
        return [], 0.0, {}, {}

    trimesh_scene = trimesh.load(str(glb_path))
    obj_actors = []
    obj_bbox_centers = {}
    obj_bbox_mins = {}  # 每个物体的 bbox_min (用于 set_pose 对齐到地面)
    obj_info = {}  # {actor_name: {color, bbox_size, bbox_min, bbox_max, volume, flatness, body_type}}
    temp_files = []
    all_min_z = []

    for geom_idx, (geom_name, geom) in enumerate(trimesh_scene.geometry.items()):
        vertices = geom.vertices.copy()
        faces = geom.faces.copy()
        if len(vertices) == 0 or len(faces) == 0:
            continue

        # 变换链 (对齐 02_render_scene.py L917-918):
        #   p_hawor = s_inv * (R_inv @ p_ras) + t_inv
        #   p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
        vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
        vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T

        avg_color = None
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            vcolors = geom.visual.vertex_colors
            if len(vcolors) > 0:
                avg_rgb = vcolors[:, :3].mean(axis=0)
                avg_color = [avg_rgb[0] / 255.0, avg_rgb[1] / 255.0, avg_rgb[2] / 255.0, 1.0]

        # 关键修复: 顶点居中到 bbox_center (local 坐标), actor.set_pose 设到 bbox_center (world)
        # 之前顶点保存在世界坐标, 但 SAPIEN 把 PLY 顶点当作 actor LOCAL 坐标,
        # 导致 actor.get_pose().p=[0,0,tz] 与 bbox_center 不一致, EE 去了 bbox_center
        # 但实际 mesh 在 [bbox_center_x, bbox_center_y, bbox_center_z + tz], 抓不到物体!
        # 修复后: actor pose = bbox_center (world), mesh 居中在 origin (local), 两者一致
        bbox_min = vertices_sapien.min(axis=0)
        bbox_max = vertices_sapien.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        vertices_local = vertices_sapien - bbox_center  # 居中到 origin (local 坐标)
        temp_ply = f'/tmp/grasp_glb_{os.getpid()}_{geom_idx}.ply'
        geom_transformed = trimesh.Trimesh(vertices=vertices_local, faces=faces, visual=geom.visual)
        geom_transformed.export(temp_ply)
        temp_files.append(temp_ply)

        # 分类, 决定物理材质
        bbox_size = bbox_max - bbox_min
        volume = abs(np.prod(bbox_size))
        max_extent = max(bbox_size)
        flatness = bbox_size[2] / max(max(bbox_size[0], bbox_size[1]), 1e-6)
        is_scene_structure = (volume > 0.01 and flatness < 0.3) or max_extent > 0.8
        all_min_z.append(bbox_min[2])

        if is_scene_structure:
            phys_material = scene.create_physical_material(
                static_friction=0.5, dynamic_friction=0.5, restitution=0.3
            )
            body_type = "kinematic"
        else:
            # 可抓取物体: 高摩擦 + 零弹性, 便于稳定夹持且碰撞不反弹 (用户: "碰一下把盘子弄翻")
            phys_material = scene.create_physical_material(
                static_friction=1.2, dynamic_friction=1.2, restitution=0.0
            )
            body_type = "dynamic"

        # 收集物体信息 (颜色 + 几何), 用于按颜色识别粉色物体和按几何识别碗
        # 用户: "我需要夹住的是那个粉色的东西，放到碗里面"
        # 范式: 不硬编码物体名, 在不同场景文件夹中通用
        obj_info[f"glb_{geom_idx}"] = {
            "color": avg_color,  # [r, g, b, a] in [0,1] or None
            "bbox_size": bbox_size.tolist(),
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "volume": float(volume),
            "flatness": float(flatness),
            "body_type": body_type,
        }

        builder = scene.create_actor_builder()

        # CPU 降级模式: 跳过视觉, 仅创建碰撞体 (避免 RenderMaterial 失败)
        render_ok = getattr(scene, "_render_available", True)
        if render_ok:
            if avg_color is not None:
                material = sapien.render.RenderMaterial(
                    base_color=avg_color, metallic=0.0, roughness=0.7, specular=0.3
                )
                builder.add_visual_from_file(filename=temp_ply, material=material)
            else:
                builder.add_visual_from_file(filename=temp_ply)

        # 碰撞体: dynamic 物体用盒形碰撞 (平整接触面), kinematic 用凸包 (精确形状)
        # 修复: 凸包的斜面导致 pad 挤压时产生侧向力, 把物体推出夹爪 (lift=0)
        # 盒形碰撞与 grasp_demo.py create_box 一致, 接触面平整, 挤压力沿 x 轴
        if body_type == "dynamic":
            half_size = (bbox_size / 2.0).tolist()
            builder.add_box_collision(half_size=half_size, material=phys_material)
        else:
            try:
                builder.add_convex_collision_from_file(filename=temp_ply, material=phys_material)
            except Exception as e:
                logger.warning(f"    {geom_name}: 凸包碰撞失败 ({e}), 尝试非凸")
                try:
                    builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                except Exception as e2:
                    logger.warning(f"    {geom_name}: 碰撞体生成失败 ({e2})")

        builder.set_physx_body_type(body_type)
        actor = builder.build(name=f"glb_{geom_idx}")
        # actor pose = bbox_center (world), mesh 居中在 origin (local)
        actor.set_pose(sapien.Pose(p=bbox_center, q=[1, 0, 0, 0]))
        # dynamic 物体显式设置质量 (统一基础惯性变量 OBJECT_DENSITY, 用户: "基础的惯性变量")
        # 盘子等扁平物体 bbox 体积小, 默认质量可能 <0.05kg, 一碰就飞
        obj_mass = None
        if body_type == "dynamic":
            obj_mass = max(volume * OBJECT_DENSITY, OBJECT_MIN_MASS)  # 统一密度 + 质量下限
            try:
                for comp in actor.components:
                    if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                        comp.mass = obj_mass
                        # angular_damping 防扁平物体被碰翻 (用户: "碰一下把盘子弄翻")
                        # 5.0 不足以抑制 kinematic 根高速冲击的翻转力矩, 提到 50.0
                        # linear_damping 抑制物体被甩飞后的飞行距离 (第五轮: glb_5 xy_drift=224cm 飞太远)
                        # 之前 1.0 导致 lift=-26cm; 用 0.5 (影响减半) + 摩擦2.0 + angular50 应能保持提升
                        comp.angular_damping = 50.0
                        comp.linear_damping = 0.5
                        break
            except Exception:
                pass
        if obj_mass is not None:
            logger.info(f"    物体{geom_idx} '{geom_name}': {body_type} "
                        f"(vol={volume:.4f}m³, flat={flatness:.2f}, mass={obj_mass:.3f}kg)")
        else:
            logger.info(f"    物体{geom_idx} '{geom_name}': {body_type} "
                        f"(vol={volume:.4f}m³, flat={flatness:.2f}, z=[{bbox_min[2]:.3f},{bbox_max[2]:.3f}])")

        obj_actors.append(actor)
        obj_bbox_centers[actor.name] = bbox_center.tolist()
        obj_bbox_mins[actor.name] = bbox_min  # 记录 bbox_min, 用于 set_pose 对齐到地面
        gc.collect()

    for f in temp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    # 地面高度 = GLB 物体最低点
    ground_z = min(all_min_z) if all_min_z else 0.0

    # 关键修复: 把每个物体放在地面上 (mesh z_min = ground_z), 避免 dynamic 物体掉落
    # 现在 mesh 居中在 origin (local), actor pose = bbox_center (world)
    # mesh local z_min = bbox_min[2] - bbox_center[2] = -half_height
    # mesh world z_min = actor_pose_z + local_z_min
    # 要让 mesh world z_min = ground_z: actor_pose_z = ground_z + half_height
    for actor in obj_actors:
        name = actor.name
        if name in obj_bbox_mins:
            bbox_min = obj_bbox_mins[name]
            old_center = np.array(obj_bbox_centers[name])
            half_height = old_center[2] - bbox_min[2]  # = (bbox_max[2] - bbox_min[2]) / 2
            new_pose_z = ground_z + half_height  # 让 mesh z_min = ground_z
            if abs(new_pose_z - old_center[2]) > 1e-6:
                new_pose = np.array([old_center[0], old_center[1], new_pose_z])
                actor.set_pose(sapien.Pose(p=new_pose, q=[1, 0, 0, 0]))
                obj_bbox_centers[name] = new_pose.tolist()
                logger.info(f"    {name}: 对齐到地面 (mesh z_min={ground_z:.3f}), "
                            f"actor pose z: {old_center[2]:.3f} → {new_pose_z:.3f}")

    logger.info(f"  GLB 加载完成: {len(obj_actors)} 个物体, 地面高度 z={ground_z:.4f}, "
                f"所有物体已对齐到地面")
    return obj_actors, ground_z, obj_bbox_centers, obj_info


# ============================================================
# 3. 机器人加载 (参考 04 _setup_robot)
# ============================================================
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
    # 关键: fix_root_link=False → 浮动根 (free joint 6 DOF)
    # 原因: 固定根的约束将 set_root_linear_velocity 设置的速度覆盖为零,
    # PhysX 接触求解器无法获得速度信息, 无法计算摩擦力 → 物体无法被提起.
    # 改为浮动根后, 速度生效, 摩擦力可正常计算 (μN >> 物体重力).
    # 根由 set_root_pose (每帧 teleport) + set_root_linear_velocity 驱动,
    # 类似 kinematic body 但带正确速度信息.
    loader.fix_root_link = False
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(urdf_path)

    active_joints = robot.get_active_joints()
    joint_names = [j.name for j in active_joints]

    # 调试: 打印所有 active joint 的 name 和 type, 验证浮动根 free joint 是否被添加
    _joint_debug = [(j.name, j.get_type(), j.get_dof()) for j in active_joints]
    logger.info(f"  [debug] active_joints: {_joint_debug}")
    logger.info(f"  [debug] fix_root_link=False, total qpos size={len(robot.get_qpos())}, dof={robot.get_dof()}")

    # 计算 qpos 索引映射: 浮动根的第一个 active joint 是 free joint,
    # 占 7 qpos (3 pos + 4 quat); 后续 joint 的 qpos 索引需偏移.
    # 用于 init_qpos manipulation 和 set_drive_target (避免直接用 list index 作 qpos index).
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
            joint.set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)
        else:
            joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)

    # 初始关节角 (浮动根时, qpos 前 7 元素是 root pose, 非根 joint 用 qpos_starts 映射)
    init_qpos = robot.get_qpos().copy()
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

    robot.set_root_pose(sapien.Pose(np.asarray(base_pos).tolist(), np.asarray(base_quat).tolist()))

    # 禁用所有 robot link 的重力, 使 set_root_linear_velocity 生效.
    # 关键: "undefined" 根关节 (0 DOF) 下, compute_passive_force 无法为根计算重力补偿,
    # PhysX 内部重力会抵消 velocity 命令 (0.13 m/s < 8 步重力减速 0.33 m/s → 根不动).
    # 禁用所有 link 重力后, 根 velocity 不被重力抵消, 手指 PD 仍维持位置.
    # 对齐 test_minimal_grasp_v3.py 的成功配置 (所有 link set_disable_gravity=True).
    for link in robot.get_links():
        for component in link.entity.components:
            if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                component.set_disable_gravity(True)
    logger.info(f"  已禁用所有 robot link 重力 (set_disable_gravity=True), "
                f"根由 set_root_pose + set_root_linear_velocity 驱动")

    # 夹爪摩擦 (friction 对齐 GalaxeaManipSim=1.0; restitution 降到 0.1 防止物体被弹飞)
    # GalaxeaManipSim 用 restitution=0.6 (full_robot PD 驱动, 冲击小); 本场景 kinematic 根冲击大, 必须降低
    gripper_material = scene.create_physical_material(
        static_friction=GRIPPER_FRICTION, dynamic_friction=GRIPPER_FRICTION, restitution=0.1
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

    # 禁用非夹爪 link 的碰撞 (ROOT 可能在地下, 躯干与地面碰撞会导致关节爆炸)
    # 只保留夹爪手指 + gripper_link 的碰撞, 用于抓取物体
    collision_link_names = set()
    for s in sides:
        collision_link_names |= {
            f"{s}_gripper_finger_link1",
            f"{s}_gripper_finger_link2",
            f"{s}_gripper_link",
        }
    n_disabled = 0
    for link in robot.get_links():
        if link.get_name() not in collision_link_names:
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    for cs in component.get_collision_shapes():
                        cs.set_collision_groups([0, 0, 0, 0])  # g0=contact=0, g1=affinity=0 → 不与任何物体碰撞
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
    logger.info(f"    夹爪摩擦: static={GRIPPER_FRICTION}, dynamic={GRIPPER_FRICTION}, restitution=0.1")

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
                 extra_gripper_indices=None, extra_arm_indices=None, extra_arm_target=None):
    """纯 PD 驱动 + 重力补偿 + decimation (对齐 GalaxeaManipSim)

    关键: 不调用 set_qpos! set_qpos + set_drive_target 双重控制会导致震荡.
    extra_gripper_indices: 双手模式第二侧夹爪 [(idx, target), ...]
    extra_arm_indices/extra_arm_target: 双手模式第二侧臂关节索引和目标
    """
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

    for _ in range(DECIMATION):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        # 浮动根: 屏蔽根 free joint 的广义力 (qf 前 6 元素: 3 linear + 3 angular).
        # 原因: compute_passive_force 会为根 free joint 计算重力补偿 (即使根已 set_disable_gravity),
        # 这会在 set_qf 时施加于根, 与每帧 set_root_pose (teleport) 冲突, 导致根漂移.
        # 屏蔽后, 根完全由 set_root_pose + set_root_linear_velocity 控制 (kinematic-like).
        if _is_floating_root(robot):
            qf[:6] = 0
        robot.set_qf(qf)
        scene.step()


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
# 6. HaWoR 数据 + MANO FK (复用 trajectory_loader)
# ============================================================
def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据"""
    from trajectory_loader import load_hawor_data as _load
    return _load(hawor_dir, hand_idx=hand_idx)


def load_hawor_c2w(hawor_dir):
    """加载 HaWoR 相机轨迹"""
    from trajectory_loader import load_hawor_c2w as _load
    return _load(hawor_dir)


def compute_mano_joints(mano_layer, pred_rot, pred_hand_pose, pred_trans):
    """MANO FK 计算手部关节"""
    from trajectory_loader import compute_mano_joints as _compute
    return _compute(mano_layer, pred_rot, pred_hand_pose, pred_trans)


def compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2, prefix="right"):
    """从 MANO 3 个特征点计算夹爪 gripper_link 位姿和手指关节值

    完全对齐 04_physics_simulation.py 的 _compute_analytical_gripper_pose (L313-362):
    方法: 加权 SVD (Procrustes) + 匹配指尖中点
      1. 从 MANO 指尖距离计算手指关节值
      2. 用加权 SVD 找最近正交旋转矩阵, Y 轴 (开合方向) 权重更高,
         优先保证开合方向精确 (因为开合方向直接影响指尖位置)
      3. 匹配两个指尖的中点确定 gripper_link 位置

    关键: MANO 的指向方向 (wrist→finger_mid) 和开合方向 (finger1→finger2)
    通常不正交。当它们非正交时, 标准 SVD 会均等折中, 导致两个方向都不精确。
    给 Y 轴更高权重可以优先保证开合方向精确, 从而最小化指尖位置误差。

    旧版用 Gram-Schmidt 正交化 + 匹配 finger1, 导致:
      - 左手 opening 方向反 (没 y_sign)
      - 非正交时两方向都不准 (没加权)
      - 位置偏移 (匹配 finger1 而非中点)
    """
    W_Y = 5.0  # Y 轴 (开合方向) 权重, 越大越优先保证开合方向精确

    # 1. 计算手指关节值
    v_finger = mano_finger2 - mano_finger1
    finger_dist = np.linalg.norm(v_finger)
    required_open_sum = finger_dist - FINGER_BASE_DIST
    joint1 = max(0.0, min(0.05, required_open_sum / 2))
    joint2 = max(0.0, min(0.05, required_open_sum / 2))

    # 2. 加权 SVD 最近正交旋转
    finger_mid = (mano_finger1 + mano_finger2) / 2
    pointing = finger_mid - mano_wrist
    pointing = pointing / max(np.linalg.norm(pointing), 1e-6)

    y_sign = 1.0 if prefix == "right" else -1.0
    opening = y_sign * v_finger / max(finger_dist, 1e-6)

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


# ============================================================
# 7. 平滑滤波
# ============================================================
class JointFilter:
    def __init__(self, alpha=LP_ALPHA_JOINT):
        self.alpha = alpha
        self.prev = None

    def next(self, x):
        if self.prev is None:
            self.prev = np.array(x, dtype=np.float64)
        else:
            self.prev = self.alpha * np.array(x, dtype=np.float64) + (1 - self.alpha) * self.prev
        return self.prev.copy()

    def reset(self):
        self.prev = None


# ============================================================
# 7b. 自适应抓取控制器 (B+C 混合: MANO意图 + 相位状态机)
# ============================================================
# MANO 手指卷曲度阈值 (0=张开, 1=完全卷曲)
GRASP_TRIGGER_CURL = 0.10   # 10% 卷曲即触发抓取 (用户: "移动个10%就可以抓上")
RELEASE_TRIGGER_CURL = 0.05  # 5% 以下释放 (跟随 MANO 手指张开)
GRASP_RESET_CURL = 0.02      # 2% 以下回到 APPROACH (允许再次抓取)
# 力控参数 (HybridGraspController)
TARGET_GRASP_FORCE = 6.0     # 目标夹紧力 (N), 由 MANO curl 动态调整 (5→6 增强夹持)
FORCE_CLOSE_STEP = 0.0015    # 力控阶段每帧闭合步长 (m), 1.5mm/帧 (1→1.5 更快达到目标力)
MAX_FORCE_MULTIPLIER = 2.0   # 最大力度倍率 (相对 TARGET_GRASP_FORCE)
# 接触后固定夹紧偏移 (关键: 防止持续闭合把物体挤出)
# 接触后只在 qpos_at_contact 基础上再闭合固定量 (由 MANO curl 决定, max 3mm)
# 旧版每帧闭合 1.5mm → 10 帧闭合 15mm → 物体被挤出飞出 (glb_6 xy_drift=389cm)
CLAMP_OFFSET_MAX = 0.002     # 最大额外闭合 2mm (curl=1.0 时), curl=0.5 时 1mm
CLAMP_CURL_FLOOR = 0.5       # LIFT/HOLD 阶段 curl 下限 (防止提升中 curl 下降导致夹紧力不足物体滑落)
# 力估计系数: kinematic 模式下用闭合程度估计力 (closure × 系数 = N)
# 闭合 5mm → 4N, 闭合 7.5mm → 6N (50→80 增强力反馈, 避免物体滑落)
FORCE_ESTIMATE_COEFF = 80.0


class AdaptiveGraspController:
    """自适应抓取控制器 — 根据 MANO 轨迹意图 + 物体距离判断夹爪开合

    策略 (B+C 结合):
      B. 通过 MANO 手指卷曲度判断抓取意图 (手指开始闭合 = 想抓)
      C. 相位状态机: APPROACH → GRASP → HOLD → RELEASE → APPROACH

    关键特性:
      - 提前抓取: MANO 手指卷曲 >10% 即触发闭合, 不等到完全卷曲
      - 释放跟随: MANO 手指张开时释放 (用户要求)
      - 物体感知: 记录最近物体距离用于调试/验证
      - 每侧独立: 双手模式各侧一个 controller 实例

    用法:
        controller = AdaptiveGraspController(obj_actors, side="right")
        target, phase, info = controller.update(gripper_pos, mano_gripper_val)
        # target: 0.0=闭合, GRIPPER_MAX_OPEN=张开
    """

    APPROACH = "APPROACH"
    GRASP = "GRASP"
    HOLD = "HOLD"
    RELEASE = "RELEASE"

    def __init__(self, obj_actors, side="right"):
        self.obj_actors = obj_actors
        self.side = side
        self.phase = self.APPROACH
        self.frame_idx = 0
        self.grasp_count = 0
        self.last_target = GRIPPER_INIT_OPEN
        # 记录抓取事件供验证
        self.events = []  # [{"frame": N, "phase": "...", "curl": ..., "obj": ..., "dist": ...}]

    def _find_nearest_object(self, gripper_pos):
        """找最近物体及其距离 (用 actor 当前位置, 非初始)"""
        if not self.obj_actors:
            return None, float('inf')
        min_dist = float('inf')
        nearest = None
        for actor in self.obj_actors:
            obj_pos = np.array(actor.get_pose().p)
            dist = float(np.linalg.norm(gripper_pos - obj_pos))
            if dist < min_dist:
                min_dist = dist
                nearest = actor.name
        return nearest, min_dist

    @staticmethod
    def _mano_curl(mano_gripper_val):
        """MANO 夹爪值 → 手指卷曲度 [0,1]
        gripper_val=0 (闭合) → curl=1 (完全卷曲)
        gripper_val=MAX (张开) → curl=0 (张开)
        """
        curl = 1.0 - (float(mano_gripper_val) / GRIPPER_MAX_OPEN)
        return float(np.clip(curl, 0.0, 1.0))

    def update(self, gripper_pos, mano_gripper_val):
        """根据 MANO 意图 + 物体距离决定夹爪目标

        Args:
            gripper_pos: 夹爪世界坐标 (np.array [3])
            mano_gripper_val: MANO retargeting 的夹爪值 (0=闭合, GRIPPER_MAX_OPEN=张开)

        Returns:
            target: 夹爪目标 (0.0=闭合, GRIPPER_MAX_OPEN=张开)
            phase: 当前相位
            info: 调试信息 dict
        """
        mano_curl = self._mano_curl(mano_gripper_val)
        nearest_obj, obj_dist = self._find_nearest_object(gripper_pos)
        prev_phase = self.phase

        # 相位状态机
        if self.phase == self.APPROACH:
            # 接近: MANO 手指开始卷曲 (>10%) → 触发抓取
            if mano_curl > GRASP_TRIGGER_CURL:
                self.phase = self.GRASP
                self.grasp_count += 1
                self._log_event(self.GRASP, mano_curl, nearest_obj, obj_dist)

        elif self.phase == self.GRASP:
            # 抓取: 立即进入保持 (闭合已下发)
            self.phase = self.HOLD

        elif self.phase == self.HOLD:
            # 保持: 维持闭合, MANO 手指张开 (<5%) → 释放
            if mano_curl < RELEASE_TRIGGER_CURL:
                self.phase = self.RELEASE
                self._log_event(self.RELEASE, mano_curl, nearest_obj, obj_dist)

        elif self.phase == self.RELEASE:
            # 释放: 张开, MANO 手指完全张开 (<2%) → 回到接近
            if mano_curl < GRASP_RESET_CURL:
                self.phase = self.APPROACH

        # 根据相位决定目标
        if self.phase in (self.GRASP, self.HOLD):
            target = 0.0  # 闭合
        else:  # APPROACH, RELEASE
            target = GRIPPER_MAX_OPEN  # 张开

        self.last_target = target
        self.frame_idx += 1

        info = {
            "phase": self.phase,
            "prev_phase": prev_phase,
            "mano_curl": mano_curl,
            "mano_raw": float(mano_gripper_val),
            "obj_dist": obj_dist,
            "nearest_obj": nearest_obj,
            "grasp_count": self.grasp_count,
        }
        return target, self.phase, info

    def _log_event(self, phase, curl, obj, dist):
        """记录相位转换事件 (供验证)"""
        event = {
            "frame": self.frame_idx,
            "phase": phase,
            "curl": round(curl, 3),
            "obj": obj,
            "dist": round(dist, 4),
        }
        self.events.append(event)
        logger.info(f"  [grasp][{self.side}] F{self.frame_idx}: {phase} "
                    f"(curl={curl:.2f}, obj={obj}@{dist:.3f}m)")

    def summary(self):
        """返回抓取统计 (供验证日志)"""
        return {
            "side": self.side,
            "grasp_count": self.grasp_count,
            "events": self.events,
            "final_phase": self.phase,
        }


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


def find_target_object_by_trajectory(trans_side_hawor, obj_actors_sapien_pos, distance_threshold=0.05):
    """预扫描手腕轨迹, 找出真正要抓的物体 (用户: "老在弄那个盘子, 没触碰到正确物体")

    根因: F0 最近物体 (glb_5) 只是掠过, 真正停留的是 glb_6 (F18-F61, 44 帧 < 5cm).
    之前代码用 F0 最近物体, 锁定错误. 这里改为统计每个物体被作为最近物体的帧数,
    取停留时间最长的物体作为抓取目标.

    Args:
        trans_side_hawor: (N, 3) 单手手腕轨迹 (HaWoR SLAM 坐标系, z-forward, y-down)
        obj_actors_sapien_pos: dict {actor_name: np.array([x,y,z])} 物体在 SAPIEN 坐标系的位置
        distance_threshold: 距离阈值 (m), 小于此值视为"停留"

    Returns:
        target_obj_name: str or None
    """
    if trans_side_hawor is None or len(trans_side_hawor) == 0 or not obj_actors_sapien_pos:
        return None
    # 物体从 SAPIEN 反变换到 HaWoR SLAM (RXWORLD_TO_SAPIEN 的逆 = R_x @ R_AXIS.T)
    # RXWORLD_TO_SAPIEN = R_AXIS @ R_x, 逆 = R_x^T @ R_AXIS^T = R_x @ R_AXIS.T (R_x 对角)
    R_x_inv = np.diag([1.0, -1.0, -1.0])  # R_x^T = R_x (对角)
    R_AXIS_inv = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)  # R_AXIS^T
    SAPIEN_TO_HAWOR = R_x_inv @ R_AXIS_inv
    obj_hawor_pos = {}
    for name, p_sapien in obj_actors_sapien_pos.items():
        obj_hawor_pos[name] = SAPIEN_TO_HAWOR @ np.asarray(p_sapien)
    # 统计每个物体被作为最近物体的帧数 (在 HaWoR SLAM 坐标系比较)
    obj_close_frames = {name: 0 for name in obj_hawor_pos}
    for f in range(len(trans_side_hawor)):
        wrist = trans_side_hawor[f]
        dists = [(name, float(np.linalg.norm(wrist - p))) for name, p in obj_hawor_pos.items()]
        dists.sort(key=lambda x: x[1])
        nearest_name, nearest_dist = dists[0]
        if nearest_dist < distance_threshold:
            obj_close_frames[nearest_name] += 1
    # 取停留帧数最多的物体
    target_name = max(obj_close_frames, key=obj_close_frames.get)
    if obj_close_frames[target_name] == 0:
        # 退回: 取全程平均距离最近的物体
        avg_dists = {}
        for name, p in obj_hawor_pos.items():
            dists = [float(np.linalg.norm(trans_side_hawor[f] - p)) for f in range(len(trans_side_hawor))]
            avg_dists[name] = float(np.mean(dists))
        target_name = min(avg_dists, key=avg_dists.get)
    return target_name


def find_pink_object(obj_info):
    """识别粉色物体 (用户: "我需要夹住的是那个粉色的东西")

    范式: 基于颜色, 不硬编码物体名, 在不同场景文件夹通用.

    粉色/品红特征 (在 [0,1] RGB 空间):
      - R 较高 (>0.4)
      - G 很低 (<0.35, 区别于橙/黄)
      - B 中等 (>G, 排除纯红; B<0.6 排除紫色)

    测试 (my_7mp4_result 场景):
      - glb_1 (0.58, 0.06, 0.33): ✓ 粉色 (R>G, B>>G, B<0.6)
      - glb_4 (0.71, 0.32, 0.05): ✗ 橙色 (B<G)
      - glb_5 (0.69, 0.29, 0.13): ✗ 橙红 (B<G)
      - glb_3 (0.18, 0.36, 0.48): ✗ 蓝灰 (R<G)

    Args:
        obj_info: dict {name: {color, bbox_size, volume, flatness, body_type, ...}}

    Returns:
        pink_obj_name: str or None (无粉色物体时)
    """
    if not obj_info:
        return None
    candidates = []
    for name, info in obj_info.items():
        if info.get("body_type") != "dynamic":
            continue
        color = info.get("color")
        if color is None:
            continue
        r, g, b = color[0], color[1], color[2]
        if r > 0.4 and g < 0.35 and b > g and 0.15 < b < 0.6:
            # 粉色度评分: R 越高、G 越低、(B-G) 越大越粉
            pinkness = r * (1.0 - g) * (b - g)
            candidates.append((name, pinkness, (r, g, b)))
    if not candidates:
        logger.warning(f"  [find_pink_object] 未找到粉色物体, 物体颜色: "
                       f"{[(n, i.get('color')) for n, i in obj_info.items()]}")
        return None
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0]
    logger.info(f"  [find_pink_object] 粉色物体候选: "
                f"{[(c[0], f'rgb={c[2]}', f'score={c[1]:.4f}') for c in candidates]}")
    logger.info(f"  [find_pink_object] 选中: {best[0]} (rgb={best[2]}, score={best[1]:.4f})")
    return best[0]


def find_bowl(obj_info, exclude_names=None):
    """识别碗 (用户: "放到碗里面")

    范式: 基于几何特征, 不硬编码物体名, 在不同场景文件夹通用.

    碗的几何特征:
      - 容器形: 体积相对较大 (volume > 1e-4 m³)
      - 扁平: flatness < 0.55 (z 厚度小于水平尺寸)
      - dynamic (可被识别为目标)
      - 排除已锁定为抓取目标的物体

    测试 (my_7mp4_result 场景):
      - glb_3 volume=0.0002, flatness=0.446 → ✓ bowlness=0.0002*0.554=1.1e-4
      - glb_0 volume≈0, flatness=0.842 → ✗ volume太小
      - 其他物体 volume≈0 → ✗

    Args:
        obj_info: dict {name: {color, bbox_size, volume, flatness, body_type, ...}}
        exclude_names: list of str, 已锁定为抓取目标的物体名 (排除)

    Returns:
        bowl_obj_name: str or None (无碗时)
    """
    if not obj_info:
        return None
    exclude = set(exclude_names or [])
    candidates = []
    for name, info in obj_info.items():
        if name in exclude:
            continue
        if info.get("body_type") != "dynamic":
            continue
        volume = info.get("volume", 0.0)
        flatness = info.get("flatness", 1.0)
        # 碗: 大体积 + 扁平
        if volume > 1e-4 and flatness < 0.55:
            bowlness = volume * (1.0 - flatness)
            candidates.append((name, bowlness, volume, flatness))
    if not candidates:
        info_str = [(n, round(i.get('volume', 0.0), 4), round(i.get('flatness', 1.0), 3))
                    for n, i in obj_info.items()]
        logger.warning(f"  [find_bowl] 未找到碗形物体 (volume>1e-4, flatness<0.55), "
                       f"物体信息 (name, vol, flat): {info_str}")
        return None
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0]
    logger.info(f"  [find_bowl] 碗候选: "
                f"{[(c[0], f'vol={c[2]:.4f}', f'flat={c[3]:.3f}', f'score={c[1]:.6f}') for c in candidates]}")
    logger.info(f"  [find_bowl] 选中: {best[0]} (vol={best[2]:.4f}, flat={best[3]:.3f})")
    return best[0]


class HybridGraspController:
    """混合抓取控制器: MANO 参数驱动 + 接触力控

    核心思路 (用户反馈: "状态判断提升都是 MANO 参数, 主要跟随 MANO, 根据参数和物体状态分析给出不同力"):
      - MANO curl 决定力度: curl 越大, 夹紧力越大 (不是固定力)
      - MANO 腕部运动决定提升: wrist_z 上升 → 提升, wrist_z 下降 → 可能放下
      - 接触感知辅助: 检测是否碰到物体 (没碰到就不加力)
      - 物体状态反馈: 物体跟随提升 → 夹住了; 物体掉落 → 没夹住

    相位 (MANO 参数驱动):
      APPROACH → CLOSE → FORCE_CONTROL → LIFT → HOLD → RELEASE → APPROACH

    vs AdaptiveGraspController:
      - 不再无脑闭合 (target=0.0), 而是 MANO curl 映射到目标夹紧力
      - 接触后继续施力 (力控), 不是碰到就停
      - 有物体状态反馈 (提升检测)
    """

    APPROACH = "APPROACH"
    CLOSE = "CLOSE"                    # 位置控制: 缓慢闭合到刚接触
    FORCE_CONTROL = "FORCE_CONTROL"    # 力控: 根据 MANO curl 施加夹紧力
    LIFT = "LIFT"                      # 提升: MANO 腕部上升 + 维持力控
    HOLD = "HOLD"                      # 保持: 维持力控, 等待 MANO 释放
    RELEASE = "RELEASE"                # 释放: MANO 手指张开

    def __init__(self, obj_actors, side="right", scene=None, robot=None, target_obj=None, obj_positions=None,
                 bowl_obj=None):
        self.obj_actors = obj_actors
        self.side = side
        self.scene = scene
        self.robot = robot
        self.phase = self.APPROACH
        self.frame_idx = 0
        self.grasp_count = 0
        self.current_close_target = GRIPPER_INIT_OPEN
        self.events = []
        # 锁定的目标物体 (用户: "老在弄那个盘子, 没触碰到正确物体")
        # 通过 find_target_object_by_trajectory 预扫描手腕轨迹确定, 避免状态机在 F0 误判
        self.target_obj = target_obj
        # 物体世界坐标位置 (用 obj_bbox_centers, 而非 actor.get_pose().p 后者为 [0,0,0])
        self.obj_positions = obj_positions or {}
        # 放置目标 (碗): pick-and-place 用 (用户: "放到碗里面")
        # 若未指定, _compute_grasp_demo_target 退化为原 4 阶段 (APPROACH→DESCEND→CLOSE→LIFT)
        self.bowl_obj = bowl_obj
        self.bowl_pos = None
        if bowl_obj is not None and bowl_obj in (obj_positions or {}):
            self.bowl_pos = np.array(obj_positions[bowl_obj], dtype=np.float64)
        # MANO 腕部 z 历史 (判断提升/下降趋势)
        self.wrist_z_history = []
        # 物体 z 历史 (判断物体跟随)
        self.obj_z_history = {}
        # 夹紧力历史 (调试)
        self.grasp_force_history = []
        # 力控目标 (由 MANO curl 动态计算)
        self.target_force = TARGET_GRASP_FORCE
        # 接触前的 qpos (力控阶段从此处开始闭合)
        self.qpos_at_contact = None
        # 被抓物体 (LIFT/HOLD 阶段跟踪, 避免物体被甩飞后 nearest_obj 变化导致检测失效)
        self.grasped_obj = None
        self.grasped_obj_z_history = []

    @staticmethod
    def _mano_curl(mano_gripper_val):
        """MANO 夹爪值 → 手指卷曲度 [0,1]"""
        curl = 1.0 - (float(mano_gripper_val) / GRIPPER_MAX_OPEN)
        return float(np.clip(curl, 0.0, 1.0))

    def _find_nearest_object(self, gripper_pos):
        """找最近物体及其距离

        如果锁定了 target_obj (预扫描确定), 只跟踪该物体, 不切换到其他物体.
        这避免了 F0 误判 (如 glb_5 盘子) 后状态机锁定错误物体的问题.
        用 obj_positions (bbox 中心), 而非 actor.get_pose().p (后者为 [0,0,0]).
        """
        if not self.obj_actors:
            return None, float('inf'), None
        # 锁定模式: 只返回 target_obj 的距离
        if self.target_obj is not None:
            if self.target_obj in self.obj_positions:
                obj_pos = np.array(self.obj_positions[self.target_obj])
                dist = float(np.linalg.norm(gripper_pos - obj_pos))
                return self.target_obj, dist, obj_pos
            # target_obj 找不到 (异常), 退回最近
        min_dist = float('inf')
        nearest = None
        nearest_pos = None
        for name, pos in self.obj_positions.items():
            obj_pos = np.array(pos)
            dist = float(np.linalg.norm(gripper_pos - obj_pos))
            if dist < min_dist:
                min_dist = dist
                nearest = name
                nearest_pos = obj_pos
        return nearest, min_dist, nearest_pos

    def _update_histories(self, wrist_pos_z, nearest_obj, nearest_obj_pos):
        """更新腕部 z 和物体 z 历史"""
        self.wrist_z_history.append(wrist_pos_z)
        if len(self.wrist_z_history) > 30:
            self.wrist_z_history = self.wrist_z_history[-30:]
        if nearest_obj is not None and nearest_obj_pos is not None:
            self.obj_z_history.setdefault(nearest_obj, []).append(nearest_obj_pos[2])
            if len(self.obj_z_history[nearest_obj]) > 30:
                self.obj_z_history[nearest_obj] = self.obj_z_history[nearest_obj][-30:]

    def _wrist_is_rising(self, window=5, threshold=0.003):
        """MANO 腕部是否在上升 (提升趋势)"""
        if len(self.wrist_z_history) < window:
            return False
        recent = self.wrist_z_history[-window:]
        return (recent[-1] - recent[0]) > threshold

    def _wrist_is_falling(self, window=5, threshold=0.003):
        """MANO 腕部是否在下降 (放下趋势)"""
        if len(self.wrist_z_history) < window:
            return False
        recent = self.wrist_z_history[-window:]
        return (recent[0] - recent[-1]) > threshold

    def _obj_is_lifting(self, obj_name, window=5, threshold=0.005):
        """物体是否在上升 (跟随夹爪提升)"""
        hist = self.obj_z_history.get(obj_name, [])
        if len(hist) < window:
            return False
        recent = hist[-window:]
        return (recent[-1] - recent[0]) > threshold

    def _obj_is_falling(self, obj_name, window=5, threshold=0.005):
        """物体是否在掉落"""
        hist = self.obj_z_history.get(obj_name, [])
        if len(hist) < window:
            return False
        recent = hist[-window:]
        return (recent[0] - recent[-1]) > threshold

    def _get_obj_pos(self, obj_name):
        """获取物体当前位置

        优先用 obj_positions (bbox 中心, 世界坐标);
        退回用 actor.get_pose().p (注意: actor pose 通常为 [0,0,0], 不可靠).
        动态物体位置会变, 但 bbox 中心是初始位置, 足够用于跟踪判断.
        """
        if obj_name is None:
            return None
        if obj_name in self.obj_positions:
            return np.array(self.obj_positions[obj_name])
        for actor in self.obj_actors:
            if actor.name == obj_name:
                try:
                    return np.array(actor.get_pose().p)
                except Exception:
                    return None
        return None

    def _grasped_is_lifting(self, window=5, threshold=0.005):
        """被抓物体是否在上升 (跟随夹爪提升) — 用 grasped_obj_z_history"""
        if len(self.grasped_obj_z_history) < window:
            return False
        recent = self.grasped_obj_z_history[-window:]
        return (recent[-1] - recent[0]) > threshold

    def _grasped_is_falling(self, window=5, threshold=0.005):
        """被抓物体是否在掉落 — 用 grasped_obj_z_history"""
        if len(self.grasped_obj_z_history) < window:
            return False
        recent = self.grasped_obj_z_history[-window:]
        return (recent[0] - recent[-1]) > threshold

    def _log_event(self, phase, mano_curl, obj, dist, force=0.0):
        """记录相位转换事件"""
        event = {
            "frame": self.frame_idx,
            "phase": phase,
            "curl": round(mano_curl, 3),
            "obj": obj,
            "dist": round(dist, 4),
            "force": round(force, 2),
        }
        self.events.append(event)
        logger.info(f"  [hybrid][{self.side}] F{self.frame_idx}: {phase} "
                    f"(curl={mano_curl:.2f}, force={force:.1f}N, obj={obj}@{dist:.3f}m)")

    def update(self, gripper_pos, gripper_R, mano_gripper_val,
               robot=None, scene=None, current_qpos=None):
        """主更新函数 — MANO 参数驱动 + 接触力控

        Args:
            gripper_pos: 夹爪世界位置 (3,)
            gripper_R: 夹爪旋转矩阵 (3,3)
            mano_gripper_val: MANO retargeting 的夹爪值 (0=闭合, MAX=张开)
            robot: SAPIEN robot (用于接触检测, 优先用传入值)
            scene: SAPIEN scene (优先用传入值)
            current_qpos: 当前手指 qpos (力控阶段用)

        Returns:
            (close_target, phase, info)
        """
        robot = robot or self.robot
        scene = scene or self.scene
        current_qpos = current_qpos if current_qpos is not None else np.array([GRIPPER_INIT_OPEN])

        # 1. MANO 参数分析
        mano_curl = self._mano_curl(mano_gripper_val)

        # 2. 接触检测 + 夹紧力
        f1_contact, f2_contact, contact_objs = False, False, []
        grasp_force = 0.0
        if scene is not None and robot is not None:
            f1_contact, f2_contact, contact_objs = get_finger_contacts(
                robot, self.side, scene, self.obj_actors
            )
            grasp_force = get_grasp_force(self.side, scene, self.obj_actors, robot)
        # kinematic 模式后备: set_qpos 瞬移位置可能不产生有效 impulse,
        # 但 get_finger_contacts 能检测到接触 (有 contact 点).
        # 此时用闭合程度估计力: 闭合 5mm → 4N, 闭合 7.5mm → 6N (FORCE_ESTIMATE_COEFF × 闭合量)
        any_contact = f1_contact or f2_contact
        if grasp_force < 0.1 and any_contact and current_qpos is not None:
            qpos_val = float(current_qpos[0]) if hasattr(current_qpos, '__len__') else float(current_qpos)
            closure = max(0.0, GRIPPER_MAX_OPEN - qpos_val)
            grasp_force = closure * FORCE_ESTIMATE_COEFF
        self.grasp_force_history.append(grasp_force)

        # 3. 最近物体
        nearest_obj, obj_dist, nearest_obj_pos = self._find_nearest_object(gripper_pos)

        # 4. 更新历史
        self._update_histories(gripper_pos[2], nearest_obj, nearest_obj_pos)

        # 5. MANO curl → 目标夹紧力 (关键: curl 越大力度越大)
        #    curl=0.1 → force=0.5N, curl=0.5 → force=2.5N, curl=1.0 → force=5.0N
        mano_target_force = TARGET_GRASP_FORCE * mano_curl

        # 6. 状态机 (MANO 参数驱动)
        prev_phase = self.phase

        if self.phase == self.APPROACH:
            # 跟随 MANO 轨迹, 夹爪跟随 MANO 开合 (不干预)
            self.current_close_target = mano_gripper_val

            # MANO curl >10% → 进入 CLOSE (主要跟随 MANO 参数)
            # 辅助条件: 物体距离 <30cm (防止空抓, 但放宽条件因为 gripper_only 位姿不准)
            # 如果 MANO curl 很高 (>30%), 说明 MANO 确实在抓取, 即使物体稍远也尝试
            curl_with_obj = mano_curl > GRASP_TRIGGER_CURL and (
                nearest_obj is not None and obj_dist < 0.30
            )
            curl_strong = mano_curl > 0.30  # MANO 高度卷曲, 肯定在抓
            if curl_with_obj or curl_strong:
                self.phase = self.CLOSE
                self.grasp_count += 1
                self._log_event(self.CLOSE, mano_curl, nearest_obj, obj_dist, grasp_force)

        elif self.phase == self.CLOSE:
            # 闭合阶段: 跟随 MANO 闭合速度 (主要跟 MANO 参数)
            # 不限 max_step, 因为 MANO 已经控制了闭合速度
            self.current_close_target = mano_gripper_val

            # 任一手指接触物体 → 切到 FORCE_CONTROL (继续施力, 不停)
            if any_contact:
                self.phase = self.FORCE_CONTROL
                self.qpos_at_contact = float(current_qpos)
                self._log_event(self.FORCE_CONTROL, mano_curl, nearest_obj, obj_dist, grasp_force)

        elif self.phase == self.FORCE_CONTROL:
            # 力控: 接触后在 qpos_at_contact 基础上施加固定夹紧偏移 (由 MANO curl 决定)
            # 关键改进: 不再每帧持续闭合 (旧版会把物体挤出飞出), 而是固定夹紧位置
            #   旧版: current_qpos - FORCE_CLOSE_STEP 每帧 → 10帧闭合15mm → 物体飞出
            #   新版: qpos_at_contact - CLAMP_OFFSET_MAX*curl → 固定2mm夹紧 → 稳定夹持
            self.target_force = mano_target_force
            clamping_offset = CLAMP_OFFSET_MAX * mano_curl  # curl=0.5→1mm, curl=1.0→2mm
            self.current_close_target = self.qpos_at_contact - clamping_offset

            # 进入 LIFT: 不再依赖 MANO 腕部上升 (用户: "要像 grasp_demo 一样真正抓取")
            # 改为: 力控持续 N 帧 (FORCE_CONTROL_LIFT_TRIGGER) 后自动进入 LIFT, 由 arm 主动提升
            # 这处理 MANO 轨迹没有上升动作的情况 (HaWoR 7: 手腕 z 全程下降)
            if not hasattr(self, '_force_control_frames'):
                self._force_control_frames = 0
            self._force_control_frames += 1
            FORCE_CONTROL_LIFT_TRIGGER = 5  # 力控 5 帧后自动进入 LIFT
            if self._force_control_frames >= FORCE_CONTROL_LIFT_TRIGGER:
                self.phase = self.LIFT
                self.grasped_obj = nearest_obj
                self.grasped_obj_z_history = list(self.obj_z_history.get(nearest_obj, []))
                self._force_control_frames = 0  # 重置, 下次重新计数
                self._log_event(self.LIFT, mano_curl, nearest_obj, obj_dist, grasp_force)

            # MANO 手指张开 → 释放 (没夹住就放弃了)
            if mano_curl < RELEASE_TRIGGER_CURL:
                self.phase = self.RELEASE
                self.current_close_target = GRIPPER_MAX_OPEN
                self._log_event(self.RELEASE, mano_curl, nearest_obj, obj_dist, grasp_force)

        elif self.phase == self.LIFT:
            # 提升: 跟随 MANO 腕部上升, 维持固定夹紧 (不再持续闭合, 防止挤出)
            # curl 下限 CLAMP_CURL_FLOOR: 防止提升中 curl 下降导致夹紧力不足物体滑落
            lift_curl = max(mano_curl, CLAMP_CURL_FLOOR)
            clamping_offset = CLAMP_OFFSET_MAX * lift_curl
            self.current_close_target = self.qpos_at_contact - clamping_offset

            # 跟踪被抓物体的 z (用 grasped_obj 而非 nearest_obj, 避免物体被甩飞后丢失跟踪)
            track_obj = self.grasped_obj if self.grasped_obj else nearest_obj
            if track_obj:
                track_pos = self._get_obj_pos(track_obj)
                if track_pos is not None:
                    self.grasped_obj_z_history.append(track_pos[2])
                    if len(self.grasped_obj_z_history) > 30:
                        self.grasped_obj_z_history = self.grasped_obj_z_history[-30:]

            # LIFT 持续若干帧后自动进 HOLD (主动提升已够, 不再依赖物体跟随检测)
            if not hasattr(self, '_lift_frames'):
                self._lift_frames = 0
            self._lift_frames += 1
            LIFT_HOLD_FRAMES = 30  # 提升 30 帧 (约 15cm) 后进 HOLD
            if self._lift_frames >= LIFT_HOLD_FRAMES:
                self.phase = self.HOLD
                self._lift_frames = 0
                self._log_event(self.HOLD, mano_curl, nearest_obj, obj_dist, grasp_force)

            # 物体掉落 → 没夹住, 回 APPROACH
            if track_obj and self._grasped_is_falling():
                self.phase = self.APPROACH
                self.current_close_target = GRIPPER_MAX_OPEN
                self.grasped_obj = None
                self._log_event("FALL_BACK", mano_curl, nearest_obj, obj_dist, grasp_force)

            # MANO 腕部不再上升 + 已持续数帧 → 可能稳住了, 进 HOLD
            if not self._wrist_is_rising() and not self._wrist_is_falling() and self.frame_idx > 5:
                self.phase = self.HOLD
                self._log_event(self.HOLD, mano_curl, nearest_obj, obj_dist, grasp_force)

            # MANO 手指张开 → 释放
            if mano_curl < RELEASE_TRIGGER_CURL:
                self.phase = self.RELEASE
                self.current_close_target = GRIPPER_MAX_OPEN
                self._log_event(self.RELEASE, mano_curl, nearest_obj, obj_dist, grasp_force)

        elif self.phase == self.HOLD:
            # 保持: 维持固定夹紧 (不再持续闭合), MANO 张开时释放
            self.target_force = mano_target_force
            hold_curl = max(mano_curl, CLAMP_CURL_FLOOR)
            clamping_offset = CLAMP_OFFSET_MAX * hold_curl
            self.current_close_target = self.qpos_at_contact - clamping_offset

            if mano_curl < RELEASE_TRIGGER_CURL:
                self.phase = self.RELEASE
                self.current_close_target = GRIPPER_MAX_OPEN
                self._log_event(self.RELEASE, mano_curl, nearest_obj, obj_dist, grasp_force)

        elif self.phase == self.RELEASE:
            # 释放: 夹爪张开, 跟随 MANO
            self.current_close_target = mano_gripper_val  # 跟随 MANO 张开
            # 清除被抓物体跟踪 (释放阶段不再追踪 grasped_obj)
            self.grasped_obj = None
            self.grasped_obj_z_history = []
            if mano_curl < GRASP_RESET_CURL:
                self.phase = self.APPROACH

        self.frame_idx += 1
        info = {
            "phase": self.phase,
            "prev_phase": prev_phase,
            "mano_curl": mano_curl,
            "mano_raw": float(mano_gripper_val),
            "obj_dist": obj_dist,
            "nearest_obj": nearest_obj,
            "f1_contact": f1_contact,
            "f2_contact": f2_contact,
            "contact_objs": contact_objs,
            "grasp_force": grasp_force,
            "target_force": self.target_force,
        }
        return self.current_close_target, self.phase, info

    def summary(self):
        """返回抓取统计"""
        return {
            "side": self.side,
            "grasp_count": self.grasp_count,
            "events": self.events,
            "final_phase": self.phase,
            "max_force": max(self.grasp_force_history) if self.grasp_force_history else 0,
            "mean_force": float(np.mean(self.grasp_force_history)) if self.grasp_force_history else 0,
        }


# ============================================================
# 8. 主仿真器
# ============================================================
class GraspSimulator:
    """SAPIEN 物理仿真: R1 机器人抓取 GLB 物体"""

    def __init__(self, hawor_dir, ras_dir, mode="full_robot", side="right",
                 output_dir=None, num_frames=-1, start_frame=0, views="both",
                 grasp_mode="adaptive"):
        self.hawor_dir = Path(hawor_dir)
        self.ras_dir = Path(ras_dir)
        self.mode = mode
        self.side = side  # "left" / "right" / "both"
        self.sides = ["left", "right"] if side == "both" else [side]
        self.views = views  # "cam" / "god" / "both" — 指定渲染哪些视角
        self.grasp_mode = grasp_mode  # "adaptive" (MANO意图+相位) / "mano" (纯重放)
        # 输出目录: 当前脚本下的 output/<mode>_<side>/
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = Path(__file__).resolve().parent
            self.output_dir = script_dir / "output" / f"{mode}_{side}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.hand_indices = {"left": 0, "right": 1}  # hand_idx=0=左手, 1=右手
        self.hand_idx = 0 if side == "left" else 1  # 向后兼容 (单侧)

        # 将日志写入输出目录, 方便用户查看
        self._setup_file_logger()

        self.scene = None
        self.robot_info = None
        self.obj_actors = []
        self.retargeting = None
        self.ik_solver = None
        self.joint_filters = {s: JointFilter() for s in self.sides}  # per-side 滤波器
        self.joint_filter = self.joint_filters[self.sides[0]]  # 向后兼容 (单侧)
        self.transform_params_path = None
        # 自适应抓取控制器 (在 run() 中加载物体后初始化)
        self.grasp_controllers = None  # {side: AdaptiveGraspController}

    def _setup_file_logger(self):
        """配置日志同时输出到终端和输出目录的 log 文件"""
        log_path = self.output_dir / "grasp.log"
        formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        fh = logging.FileHandler(str(log_path), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        # 避免重复添加 handler
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and str(h.baseFilename) == str(log_path):
                return
        logger.addHandler(fh)
        logger.info(f"  日志文件: {log_path}")

    def _find_reconstruction_file(self):
        rec_dir = self.hawor_dir / "reconstruction"
        if rec_dir.exists():
            for f in rec_dir.glob("hawor_results_*.npz"):
                return f
        for f in self.hawor_dir.glob("hawor_results_*.npz"):
            return f
        return None

    def _align_scene(self):
        """调用 01_align_scene.py 对齐 RAS GLB → HaWoR 坐标系

        01_align_scene.py 已在模块顶部加载, compute_and_save_transform_params 是模块级常量
        """
        rec_file = self._find_reconstruction_file()
        if rec_file is None:
            raise FileNotFoundError(f"未找到 HaWoR 重建文件: {self.hawor_dir}")

        align_dir = self.output_dir / "alignment"
        align_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Step 1: 对齐 RAS GLB → HaWoR 坐标系 (调用 01_align_scene.py)")
        logger.info("=" * 60)
        self.transform_params_path = compute_and_save_transform_params(
            ras_output=str(self.ras_dir),
            hawor_reconstruction=str(rec_file),
            output_dir=str(align_dir),
        )
        logger.info(f"  transform_params: {self.transform_params_path}")

    def _init_retargeting(self):
        """初始化 DexRetargeting (仅 full_robot, 支持双手)"""
        if self.mode != "full_robot":
            return

        from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
        from dex_retargeting.retargeting_config import RetargetingConfig

        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        # 每侧创建独立 retargeting 实例
        self.retargeting = {}  # {"left": instance, "right": instance}
        self._ref_indices = {}
        self._retarget2sapien = {}
        self._sapien2retarget = {}
        self._fixed_qpos = {}

        sapien_joint_names = self.robot_info["joint_names"]
        init_qpos = self.robot_info["init_qpos"]

        for s in self.sides:
            hand_type = HandType.right if s == "right" else HandType.left
            config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)

            override = dict(
                add_dummy_free_joint=True,
                normal_delta=1e-5,
                huber_delta=0.01,
                target_link_names=[
                    f"{s}_gripper_finger_link1",
                    f"{s}_gripper_finger_link2",
                    f"{s}_gripper_link",
                ],
                target_link_human_indices=np.array([4, 8, 0]),
            )
            config = RetargetingConfig.load_from_file(config_path, override=override)
            self.retargeting[s] = config.build()
            self._ref_indices[s] = self.retargeting[s].optimizer.target_link_human_indices

            # retargeting ↔ sapien 关节映射
            retarget_joint_names = self.retargeting[s].joint_names
            self._retarget2sapien[s] = np.array(
                [retarget_joint_names.index(n) for n in sapien_joint_names if n in retarget_joint_names]
            ).astype(int)
            self._sapien2retarget[s] = {r: i for i, r in enumerate(self._retarget2sapien[s])}

            fixed_retarget_indices = self.retargeting[s].optimizer.idx_pin2fixed
            self._fixed_qpos[s] = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
            for i, retarget_idx in enumerate(fixed_retarget_indices):
                if retarget_idx in self._sapien2retarget[s]:
                    sapien_idx = self._sapien2retarget[s][retarget_idx]
                    if sapien_idx < len(init_qpos):
                        self._fixed_qpos[s][i] = init_qpos[sapien_idx]

            logger.info(f"  DexRetargeting 已初始化 ({s}), ref_indices={self._ref_indices[s]}")

    def _init_ik(self):
        """初始化 RelaxedIK (仅 full_robot, 支持双手)"""
        if self.mode != "full_robot":
            return

        from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

        self.ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=IK_TOLERANCES,
        )
        for s in self.sides:
            arm_starting = RIGHT_ARM_STARTING if s == "right" else LEFT_ARM_STARTING
            if s == "right":
                self.ik_solver.relaxed_ik_right.reset(arm_starting)
            else:
                self.ik_solver.relaxed_ik_left.reset(arm_starting)
        logger.info(f"  RelaxedIK 已初始化 (sides={self.sides})")

    def _get_gripper_pose_from_retargeting(self, retarget_qpos, side=None):
        """从 retargeting FK 获取期望夹爪位姿

        Args:
            retarget_qpos: retargeting 输出的关节角
            side: 指定侧 ("left"/"right"); None 时用 self.sides[0]
        """
        s = side if side is not None else self.sides[0]
        internal_robot = self.retargeting[s].optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)
        target_name = f"{s}_gripper_link"
        for i, name in enumerate(internal_robot.link_names):
            if name == target_name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        raise RuntimeError(f"内部机器人中找不到 {target_name}")

    def _compute_optimal_base(self, wrist_positions_sapien, R_c2w_all=None):
        """计算最优固定基座 — 使臂基座 (非 ROOT) 在手腕上方 COMFORTABLE_REACH 处

        R1 机器人臂基座比 ROOT 高 ~1.4m (躯干), 必须减去此偏移,
        否则臂基座远离手腕, IK 目标超出臂展.

        朝向修复: root_quat 根据 R_c2w_all 第一帧相机 forward 计算 yaw,
        使机器人 +X 对齐相机水平 forward (修复 90 度偏转: 默认单位四元数面向 +X,
        但相机看向 -Z, 差 90 度).
        """
        # 双手模式用两侧偏移的平均 (y=0, 因为左右对称)
        if self.side == "both":
            arm_base_offset = (ARM_BASE_OFFSET_LEFT + ARM_BASE_OFFSET_RIGHT) / 2.0
        else:
            arm_base_offset = ARM_BASE_OFFSET_LEFT if self.side == "left" else ARM_BASE_OFFSET_RIGHT
        if len(wrist_positions_sapien) == 0:
            root_pos = np.array([0.0, 0.0, COMFORTABLE_REACH]) - arm_base_offset
            root_quat = self._compute_robot_yaw_quat(R_c2w_all)
            return root_pos, root_quat

        wrist_arr = np.array(wrist_positions_sapien)
        centroid = wrist_arr.mean(axis=0)

        # 计算机器人 forward (用于让臂基座后退)
        root_quat = self._compute_robot_yaw_quat(R_c2w_all)
        try:
            # 用四元数旋转 [1,0,0] (URDF 默认前方) 得到机器人当前前方
            # pytransform3d 四元数格式 [w, x, y, z], 用 matrix_from_quaternion 转矩阵再乘向量
            R_root = pr.matrix_from_quaternion(root_quat)
            forward_3d = R_root @ np.array([1.0, 0.0, 0.0])
            forward_2d = np.array([forward_3d[0], forward_3d[1], 0.0])
            norm = float(np.linalg.norm(forward_2d))
            forward_2d = forward_2d / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])
        except Exception:
            forward_2d = np.array([0.0, -1.0, 0.0])

        # 目标臂基座位置 = 手腕质心正上方 COMFORTABLE_REACH, 沿 forward 反方向后退 BASE_BACK_OFFSET
        # (让机器人退后一点, 不挡物体; 对齐 04 BASE_OFFSET_Y=0.30 思路)
        desired_arm_base = centroid.copy()
        desired_arm_base[:2] -= forward_2d[:2] * BASE_BACK_OFFSET
        desired_arm_base[2] = centroid[2] + COMFORTABLE_REACH

        # ROOT 位置 = 臂基座位置 - 臂基座偏移
        root_pos = desired_arm_base - arm_base_offset

        # 检查手腕到臂基座的距离是否在臂展内
        max_dist = np.max(np.linalg.norm(wrist_arr - desired_arm_base, axis=1))
        if max_dist > ARM_MAX_REACH * 0.9:
            scale = ARM_MAX_REACH * 0.9 / max_dist
            offset = desired_arm_base - centroid
            desired_arm_base[:2] = centroid[:2] + offset[:2] * scale
            desired_arm_base[2] = centroid[2] + offset[2] * scale
            root_pos = desired_arm_base - arm_base_offset

        logger.info(f"  最优 ROOT: {root_pos.round(3)}, 臂基座将在: {desired_arm_base.round(3)} "
                    f"(后退 {BASE_BACK_OFFSET}m, forward={forward_2d.round(3)})")
        logger.info(f"  最远手腕距离: {max_dist:.4f}m (臂展={ARM_MAX_REACH}m)")
        return root_pos, root_quat

    def _compute_robot_yaw_quat(self, R_c2w_all=None):
        """根据第一帧相机 R_c2w 计算机器人 yaw 四元数 (绕 Z 轴)

        机器人 URDF 默认 +X 为前方, 相机 (OpenGL) 看向 -Z.
        旧版用单位四元数 (面向 +X), 与相机朝向差 90 度 → 视频中机器人侧着身.
        现在用相机 forward 的水平分量算 yaw, 让机器人面向相机看的方向.
        """
        if R_c2w_all is None or len(R_c2w_all) == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])  # 默认: 面向 +X
        try:
            R_c2w = R_c2w_all[min(self.start_frame, len(R_c2w_all) - 1)]
            cam_R_sapien = RXWORLD_TO_SAPIEN @ R_c2w
            forward = -cam_R_sapien[:, 2]  # OpenGL -Z forward
            yaw = float(np.arctan2(forward[1], forward[0]))
            root_quat = pr.quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0, yaw]))
            logger.info(f"  机器人朝向: yaw={np.degrees(yaw):.1f}° (对齐相机 forward={forward.round(3)})")
            return root_quat
        except Exception as e:
            logger.warning(f"  计算相机朝向失败: {e}, 使用默认朝向 (+X)")
            return np.array([1.0, 0.0, 0.0, 0.0])

    def _compute_mano_neutral_target(self, local_idx, side):
        """MANO+offset 中和态 (用户: "mano参数为主体, 你可以平移轨迹, 但不能离开轨迹, 偏移轨迹那么多")

        与 _compute_grasp_demo_target 的区别:
          - demo: 完全预规划 smoothstep 轨迹, 绕过 MANO (用户: "你现在完全是脱离mano参数了吗")
          - neutral: EE = MANO 解析夹爪位置 + 常量偏移, 保持 MANO 运动形状

        EE 位置 = mano_root_pos[local_idx] + offset
          - offset 在 "抓取帧" (f_grasp) 处对齐 MANO 到目标物体 (offset 最小化)
          - 常量偏移保持 MANO 轨迹形状不变, 只做平移 (用户: "你可以平移轨迹")
        gripper_R = 固定 top-down (稳定抓取朝向)
        gripper_val: APPROACH/DESCEND 跟随 MANO, CLOSE/LIFT/TRANSPORT 强制闭合, RELEASE 强制打开

        关键改进: 阶段判定基于 f_grasp (MANO 最接近目标的帧), 不是固定帧占比.
          - f_grasp 之前: APPROACH → DESCEND (MANO 自然接近目标)
          - f_grasp ~ f_grasp+close_dur: CLOSE (在 MANO 最接近处抓取)
          - 之后: LIFT → TRANSPORT → RELEASE → RETREAT (跟随 MANO 离开)
        这样 CLOSE 发生在 MANO 实际到达目标高度时, 而非固定 25% 处 (之前 z=15cm 太高抓不到).

        Returns:
            (gripper_pos, gripper_R, gripper_val, phase) or None (无 MANO 轨迹/无目标时退回纯 MANO)
        """
        traj = getattr(self, '_mano_gripper_traj', {}).get(side)
        offset = getattr(self, '_mano_neutral_offset', {}).get(side)
        f_grasp = getattr(self, '_mano_grasp_frame', {}).get(side)
        if traj is None or offset is None or f_grasp is None or len(traj["pos"]) == 0:
            return None
        # 越界保护
        if local_idx >= len(traj["pos"]):
            local_idx = len(traj["pos"]) - 1

        # 中和态: MANO 位置 + 常量偏移 (保持 MANO 轨迹形状)
        mano_pos = traj["pos"][local_idx]
        mano_j1 = float(traj["j1"][local_idx])

        # 固定 top-down 朝向 (gripper X 朝下, 稳定抓取)
        gripper_R = getattr(self, '_gripper_R_fixed', None)
        if gripper_R is None:
            gripper_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float64)

        # 阶段判定: 基于 f_grasp (MANO 最接近目标的帧), 不是固定帧占比
        # 这样 CLOSE 发生在 MANO 实际到达目标高度时, 解决 "z=15cm 太高抓不到" 问题
        n = max(self.num_frames, 1)
        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        has_bowl = ctrl is not None and ctrl.bowl_pos is not None

        # CLOSE 持续时间: ~15% 总帧数 (够 PD 收敛)
        close_dur = max(3, int(n * 0.15))
        close_end = min(f_grasp + close_dur, n - 1)
        # APPROACH 占 f_grasp 前约 40%, DESCEND 占后 60%
        approach_end = max(1, int(f_grasp * 0.4))

        if has_bowl:
            # CLOSE 后: LIFT → TRANSPORT → RELEASE → RETREAT
            remaining = max(1, n - close_end)
            lift_end = close_end + max(1, int(remaining * 0.3))
            transport_end = close_end + max(1, int(remaining * 0.6))
            release_end = close_end + max(1, int(remaining * 0.8))

            if local_idx < approach_end:
                phase = "APPROACH"
            elif local_idx < f_grasp:
                phase = "DESCEND"
            elif local_idx < close_end:
                phase = "CLOSE"
            elif local_idx < lift_end:
                phase = "LIFT"
            elif local_idx < transport_end:
                phase = "TRANSPORT"
            elif local_idx < release_end:
                phase = "RELEASE"
            else:
                phase = "RETREAT"
        else:
            # 无碗: APPROACH → DESCEND → CLOSE → LIFT
            if local_idx < approach_end:
                phase = "APPROACH"
            elif local_idx < f_grasp:
                phase = "DESCEND"
            elif local_idx < close_end:
                phase = "CLOSE"
            else:
                phase = "LIFT"

        # 位置计算: 基于阶段决定 (核心改进)
        # - APPROACH/DESCEND/TRANSPORT/RELEASE/RETREAT: 跟随 MANO 轨迹 + offset
        # - CLOSE: 保持抓取位置 (MANO 在 f_grasp 后会上升, 不跟随以让夹爪闭合)
        # - LIFT: 从抓取位置平滑过渡到 MANO+offset (整个 LIFT 阶段渐进, 避免突变甩飞物体)
        grasp_pos = traj["pos"][f_grasp] + offset  # 抓取位置 (f_grasp 处)
        mano_target_pos = mano_pos + offset  # MANO+offset 当前帧

        if phase == "CLOSE":
            # 保持抓取位置, 不跟随 MANO 上升 (让夹爪在物体高度闭合)
            gripper_pos = grasp_pos
        elif phase == "LIFT":
            # 平滑过渡: 从抓取位置渐变到 MANO+offset, 贯穿整个 LIFT 阶段
            # (之前 5 帧过渡太快, 速度突变导致物体甩飞 xy_drift=55cm)
            lift_total = max(1, lift_end - close_end)
            lift_t = min(1.0, (local_idx - close_end) / float(lift_total))
            lift_t = lift_t * lift_t * (3 - 2 * lift_t)  # smoothstep
            gripper_pos = grasp_pos * (1 - lift_t) + mano_target_pos * lift_t
        else:
            # 跟随 MANO 轨迹 + offset (保持 MANO 运动形状)
            gripper_pos = mano_target_pos

        # Z-floor: TRANSPORT/RELEASE/RETREAT 阶段确保手指不撞碗
        # (MANO+offset 轨迹可能经过碗位置, 闭合夹爪撞碗导致碗飞 207cm;
        #  RETREAT 时 MANO 自然下降, 手指会再次进入碗区域, 需同样保护)
        # 手指在 EE 下方 3.7cm (FINGER_FORWARD_OFFSET), 需 EE_z > bowl_z + 15cm + 3.7cm
        if has_bowl and phase in ("TRANSPORT", "RELEASE", "RETREAT"):
            bowl = ctrl.bowl_pos
            bowl_safe_z = float(bowl[2]) + 0.15 + 0.037  # 碗上方 15cm + 手指偏移
            if gripper_pos[2] < bowl_safe_z:
                gripper_pos = gripper_pos.copy()
                gripper_pos[2] = bowl_safe_z

        # gripper_val: 阶段强制为主 (确保抓取成功), MANO 手指值为辅
        # - APPROACH: 跟随 MANO (接近阶段不强制)
        # - DESCEND: 强制打开 (到达物体前手指张开, 避免推开物体)
        # - CLOSE: 强制闭合 0 (夹住物体)
        # - LIFT/TRANSPORT: 强制闭合 0 (维持抓取, 防 MANO 抖动松开物体)
        # - RELEASE: 强制打开 (释放物体到碗)
        # - RETREAT: 跟随 MANO
        if phase == "DESCEND":
            gripper_val = GRIPPER_MAX_OPEN
        elif phase in ("CLOSE", "LIFT", "TRANSPORT"):
            gripper_val = 0.0
        elif phase == "RELEASE":
            gripper_val = GRIPPER_MAX_OPEN
        else:
            gripper_val = mano_j1

        return gripper_pos, gripper_R, gripper_val, phase

    def _compute_grasp_demo_target(self, local_idx, side):
        """grasp_demo 式预规划轨迹 (用户: "像 grasp_demo.py 一样真正的抓取物体")

        完全绕过 MANO arm 轨迹, 直接规划到目标物体位置:
          Phase 1 (APPROACH):  F0-F25%     → 从初始 EE 位置移动到 (grasp_pos + [0,0,8cm])
          Phase 2 (DESCEND):   F25%-F50%   → 下降到 grasp_pos, gripper 保持 MAX_OPEN
          Phase 3 (CLOSE):    F50%-F80%   → EE 保持 grasp_pos, gripper 闭合到 0 (PD 需充分时间收敛)
          Phase 4 (LIFT):      F80%-F100%  → 从 grasp_pos 提升到 (grasp_pos + [0,0,20cm])

        Pick-and-Place 模式 (用户: "夹住粉色的东西, 放到碗里面"):
          在 LIFT 之后新增:
          Phase 5 (TRANSPORT): F55%-F75%   → 水平移动到碗上方 (bowl_pos + [0,0,15cm]), gripper 闭合
          Phase 6 (RELEASE):   F75%-F85%   → 在碗上方打开夹爪, 物体掉入碗中
          Phase 7 (RETREAT):   F85%-F100%  → 后退到碗上方 30cm, gripper 全开

        关键: DESCEND 和 CLOSE 分开 (对齐 grasp_demo.py: 先到位再闭合).
        之前边下降边闭合会导致手指在物体上方就闭合, 把物体推开而非夹住.

        用 smoothstep 插值, 固定 top-down 抓取朝向 (gripper X 朝下).

        Returns:
            (gripper_pos, gripper_R, gripper_val, phase) or None (无目标时退回 MANO)
        """
        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        if target_name is None or target_name not in self.obj_bbox_centers:
            return None
        target_pos = np.array(self.obj_bbox_centers[target_name], dtype=np.float64)

        # 检查是否有放置目标 (碗) - pick-and-place 模式
        bowl_pos = ctrl.bowl_pos if ctrl else None
        has_bowl = bowl_pos is not None

        # 缓存初始 EE 位置 (第一帧从 robot 获取)
        if not hasattr(self, '_grasp_demo_initial_ee'):
            self._grasp_demo_initial_ee = {}
        if side not in self._grasp_demo_initial_ee:
            robot = self.robot_info.get("robot")
            ee_pos = None
            if robot is not None:
                for link in robot.get_links():
                    if link.get_name() == f"{side}_gripper_link":
                        ee_pos = np.array(link.get_entity_pose().p, dtype=np.float64)
                        break
            if ee_pos is None:
                ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
            self._grasp_demo_initial_ee[side] = ee_pos
            logger.info(f"  [grasp_demo][{side}] 初始 EE={ee_pos.round(3)}, "
                        f"目标物体={target_name}@{target_pos.round(3)}, "
                        f"碗={ctrl.bowl_obj if ctrl else None}@{bowl_pos if has_bowl else None}")

        initial_ee = self._grasp_demo_initial_ee[side]
        n = max(self.num_frames, 1)

        # R1 夹爪手指 origin 在 gripper X 方向偏移 FINGER_FORWARD_OFFSET=0.037m
        # 当 gripper X = [0,0,-1] (top-down 朝下) 时, 手指在 EE 下方 3.7cm
        # 直接把 EE 设到物体中心会让手指在物体下方, 无法夹住物体 (lift=0.23cm 的根因)
        # 修复: EE 位置 = target - R[:,0] * FINGER_FORWARD_OFFSET, 让手指到达物体中心
        gripper_R_fixed = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ], dtype=np.float64)
        FINGER_FORWARD_OFFSET = 0.037
        ee_offset = -gripper_R_fixed[:, 0] * FINGER_FORWARD_OFFSET  # = [0, 0, +0.037]
        # 让手指到达目标物体中心: grasp 时手指包住物体, approach 在物体上方, lift 在物体上方更高
        grasp_pos = target_pos + ee_offset  # EE 在物体上方 3.7cm, 手指恰在物体中心
        approach_pos = grasp_pos + np.array([0.0, 0.0, 0.08])  # approach 在 grasp 上方 8cm
        # LIFT 高度 12cm (>10cm 满足 grasp_demo 标准), 每帧 3mm (40帧), 物体有时间响应摩擦力
        lift_pos = grasp_pos + np.array([0.0, 0.0, 0.12])  # lift 在 grasp 上方 12cm

        # 阶段分配
        if has_bowl:
            # Pick-and-Place: 7 阶段 (用户: "夹住粉色的东西, 放到碗里面")
            # 关键: LIFT 缩短到 15% 帧 (12cm/16帧=7.5mm/帧, 仍能维持接触)
            # TRANSPORT 20% 帧, 水平移动到碗上方
            # RELEASE 10% 帧, 夹爪打开 (物体掉入碗)
            # RETREAT 15% 帧, 后退
            approach_end = max(1, int(n * 0.10))     # F0-F10%: 接近 (缩短, 留时间给放置)
            descend_end = max(approach_end + 1, int(n * 0.25))  # F10%-F25%: 下降
            close_end = max(descend_end + 1, int(n * 0.40))     # F25%-F40%: 闭合
            lift_end = max(close_end + 1, int(n * 0.55))        # F40%-F55%: 提升
            transport_end = max(lift_end + 1, int(n * 0.75))    # F55%-F75%: 运输
            release_end = max(transport_end + 1, int(n * 0.85)) # F75%-F85%: 释放
            # RETREAT: F85%-F100%
            # 碗上方释放位置: 在碗中心上方 15cm + ee_offset (EE 在物体上方, 手指恰在释放点)
            bowl_release_pos = bowl_pos + np.array([0.0, 0.0, 0.15]) + ee_offset
            # 后退位置: 碗上方 30cm (远离物体, 避免碰撞)
            retreat_pos = bowl_release_pos + np.array([0.0, 0.0, 0.15])
        else:
            # 原始 4 阶段 (无碗): APPROACH → DESCEND → CLOSE → LIFT
            approach_end = max(1, int(n * 0.25))   # F0-F25%: 接近
            descend_end = max(approach_end + 1, int(n * 0.50))  # F25%-F50%: 下降, gripper 全开
            close_end = max(descend_end + 1, int(n * 0.65))     # F50%-F65%: 闭合 (K=20000, 5τ=9帧, 15% 够)

        def smoothstep(t):
            t = max(0.0, min(1.0, t))
            return t * t * (3 - 2 * t)

        if local_idx < approach_end:
            # APPROACH: EE 从初始位置 → approach_pos, gripper 全开
            t = smoothstep(local_idx / approach_end)
            gripper_pos = initial_ee * (1 - t) + approach_pos * t
            gripper_val = GRIPPER_MAX_OPEN
            phase = "APPROACH"
        elif local_idx < descend_end:
            # DESCEND: EE 从 approach_pos → grasp_pos, gripper 全开 (对齐 grasp_demo: 先到位)
            t = smoothstep((local_idx - approach_end) / max(descend_end - approach_end, 1))
            gripper_pos = approach_pos * (1 - t) + grasp_pos * t
            gripper_val = GRIPPER_MAX_OPEN
            phase = "DESCEND"
        elif local_idx < close_end:
            # CLOSE: EE 保持 grasp_pos, gripper 立即设为 0 (grip_cmd=0), 让 PD 充分收敛.
            # 旧版用 (1-t) 线性渐变, PD 始终滞后 cmd, LIFT 开始时 q 仍有 0.006, pad 间隙 9mm 没夹住物体.
            # 直接 cmd=0, PD 用 ~6 帧 (5τ, τ=0.04s) 收敛到 q=0, 剩余 28 帧稳定夹持.
            gripper_pos = grasp_pos
            gripper_val = 0.0
            phase = "CLOSE"
        elif has_bowl and local_idx < lift_end:
            # LIFT (pick-and-place): EE 从 grasp_pos → lift_pos, gripper 闭合
            # 12cm/16帧=7.5mm/帧 (vs 原始 3mm/帧), 仍能维持接触摩擦
            t = smoothstep((local_idx - close_end) / max(lift_end - close_end, 1))
            gripper_pos = grasp_pos * (1 - t) + lift_pos * t
            gripper_val = 0.0
            phase = "LIFT"
        elif has_bowl and local_idx < transport_end:
            # TRANSPORT (用户: "放到碗里面"): 水平移动到碗上方, gripper 闭合
            # 关键: 与 LIFT 一样不能 teleport (会破坏接触), 用 velocity 控制
            t = smoothstep((local_idx - lift_end) / max(transport_end - lift_end, 1))
            gripper_pos = lift_pos * (1 - t) + bowl_release_pos * t
            gripper_val = 0.0
            phase = "TRANSPORT"
        elif has_bowl and local_idx < release_end:
            # RELEASE: 在碗上方打开夹爪, 物体掉入碗
            gripper_pos = bowl_release_pos
            gripper_val = GRIPPER_MAX_OPEN
            phase = "RELEASE"
        elif has_bowl:
            # RETREAT: 后退到碗上方 30cm, gripper 全开
            t = smoothstep((local_idx - release_end) / max(n - release_end, 1))
            gripper_pos = bowl_release_pos * (1 - t) + retreat_pos * t
            gripper_val = GRIPPER_MAX_OPEN
            phase = "RETREAT"
        else:
            # LIFT (原始 4 阶段, 无碗): EE 从 grasp_pos → lift_pos, gripper 保持闭合
            t = smoothstep((local_idx - close_end) / max(n - close_end, 1))
            gripper_pos = grasp_pos * (1 - t) + lift_pos * t
            gripper_val = 0.0
            phase = "LIFT"

        # 固定 top-down 朝向: gripper X(前进) 朝下, Y(手指方向) 沿世界 X
        # 复用上方 gripper_R_fixed (避免重复定义)
        gripper_R = gripper_R_fixed

        return gripper_pos, gripper_R, gripper_val, phase

    def _step_full_robot(self, joints_sapien):
        """full_robot: Retargeting + IK → qpos

        单侧: joints_sapien 为 (N,3) array, 返回 (ik_joints, gripper_val)
        双手: joints_sapien 为 dict {"left": array, "right": array},
              返回 ({"left":..., "right":...}, {"left":..., "right":...})
        """
        is_bimanual = isinstance(joints_sapien, dict)
        sides = self.sides if is_bimanual else [self.side]
        js_dict = joints_sapien if is_bimanual else {self.side: joints_sapien}

        arm_targets = {}
        gripper_vals = {}

        local_idx = getattr(self, '_current_local_idx', 0)
        for s in sides:
            # === hybrid 模式: MANO+offset 中和态 (MANO 为主体, 常量平移对齐目标) ===
            # 用户: "mano参数为主体, 你可以平移轨迹, 但不能离开轨迹"
            # 之前的问题 (demo): 完全绕过 MANO, 差距太大
            # 现在: EE = MANO 轨迹 + 常量偏移, 保持 MANO 运动形状
            if self.grasp_mode == "hybrid" and self.grasp_controllers is not None:
                neutral_target = self._compute_mano_neutral_target(local_idx, s)
                if neutral_target is not None:
                    gripper_pos_fk, R_ee_world_fk, gripper_val, phase = neutral_target
                    gripper_vals[s] = float(gripper_val)
                    if local_idx == 0 or (local_idx > 0 and local_idx % 30 == 0):
                        logger.info(f"  [neutral][{s}] F{local_idx}: phase={phase}, "
                                    f"pos={gripper_pos_fk.round(3)}, grip={gripper_val:.4f}")
                else:
                    # 退回 MANO (无目标物体时)
                    js = js_dict[s]
                    ref_value = js[self._ref_indices[s], :].astype(np.float32)
                    retarget_qpos = self.retargeting[s].retarget(ref_value, self._fixed_qpos[s])
                    gripper_val = 0.0
                    for i, name in enumerate(self.retargeting[s].joint_names):
                        if "gripper_finger_joint1" in name and i < len(retarget_qpos):
                            gripper_val = float(retarget_qpos[i])
                    gripper_val = float(np.clip(gripper_val, 0.0, GRIPPER_MAX_OPEN))
                    gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retarget_qpos, s)
                    gripper_vals[s] = gripper_val
                # 跳到 IK 求解 (绕过下面的 adaptive/mano 逻辑)
                # 臂基座坐标系 — 关键: base_quat 会旋转 ARM_BASE_OFFSET, 必须用旋转后的 offset
                # (旧版用未旋转 offset, 导致 base_link_p 错位 13cm, IK target 在错误坐标系)
                base_link_q = self._base_quat
                base_link_R = pr.matrix_from_quaternion(base_link_q)
                arm_base_offset_raw = ARM_BASE_OFFSET_LEFT if s == "left" else ARM_BASE_OFFSET_RIGHT
                arm_base_offset_rotated = base_link_R @ arm_base_offset_raw
                base_link_p = self._base_pos + arm_base_offset_rotated
                base_link_R_inv = base_link_R.T
                ik_target_b = base_link_R_inv @ (gripper_pos_fk - base_link_p)
                R_ee_for_ik = R_ee_world_fk if R_ee_world_fk is not None else np.eye(3)
                ee_R_base = base_link_R_inv @ R_ee_for_ik
                ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
                if local_idx == 0:
                    logger.info(f"  [IK debug][{s}] grasp_demo target: gripper_pos_fk={gripper_pos_fk.round(3)}, "
                                f"base_link_p={base_link_p.round(3)}, "
                                f"ik_target_b={ik_target_b.round(3)}, |ik_target_b|={np.linalg.norm(ik_target_b):.3f} "
                                f"(reach={ARM_MAX_REACH})")
                solve_fn = (self.ik_solver.solve_position_right if s == "right"
                            else self.ik_solver.solve_position_left)
                ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
                for _ in range(IK_SOLVE_PER_FRAME - 1):
                    ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
                ik_joints = self.joint_filters[s].next(ik_joints)
                if s == "right" and len(ik_joints) == 6:
                    ik_joints = np.clip(ik_joints, RIGHT_ARM_JOINT_LIMITS[:, 0], RIGHT_ARM_JOINT_LIMITS[:, 1])
                arm_targets[s] = ik_joints
                continue
            # === 非 hybrid: MANO retargeting (原有逻辑) ===
            js = js_dict[s]
            ref_value = js[self._ref_indices[s], :].astype(np.float32)
            retarget_qpos = self.retargeting[s].retarget(ref_value, self._fixed_qpos[s])

            gripper_val = 0.0
            for i, name in enumerate(self.retargeting[s].joint_names):
                if "gripper_finger_joint1" in name and i < len(retarget_qpos):
                    gripper_val = float(retarget_qpos[i])
            gripper_val = float(np.clip(gripper_val, 0.0, GRIPPER_MAX_OPEN))

            gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retarget_qpos, s)

            if self.grasp_mode == "adaptive" and self.grasp_controllers is not None:
                adapted_target, phase, grasp_info = self.grasp_controllers[s].update(
                    gripper_pos_fk, gripper_val
                )
                gripper_val = adapted_target
                # 首帧/相位变化时打印调试
                if not hasattr(self, '_grasp_debug_printed'):
                    self._grasp_debug_printed = True
                    logger.info(f"  [grasp][{s}] 首帧: phase={phase}, "
                                f"curl={grasp_info['mano_curl']:.2f}, "
                                f"obj={grasp_info['nearest_obj']}@{grasp_info['obj_dist']:.3f}m")
            elif self.grasp_mode == "mano":
                # mano 模式: 纯重放 + 接触维持夹紧 (与 _step_gripper_only 一致, 用户: "夹爪和整个机器人的任务是一样的")
                # 接触前: 跟随 MANO (纯重放, gripper_val 来自 retargeting)
                # 接触后: 维持固定夹紧 (gripper_val0 - CLAMP_OFFSET*curl), 防止物体滑落
                # MANO 张开 (curl < RELEASE_TRIGGER_CURL) 时释放, 回到纯重放
                if not hasattr(self, '_mano_state_fr'):
                    self._mano_state_fr = {}
                if s not in self._mano_state_fr:
                    self._mano_state_fr[s] = {'contact': False, 'gval0': None, 'obj': None, 'logged': False}
                st = self._mano_state_fr[s]
                try:
                    f1, f2, c_objs = get_finger_contacts(self.robot_info["robot"], s, self.scene, self.obj_actors)
                except Exception:
                    f1, f2, c_objs = False, False, []
                mano_curl = 1.0 - float(gripper_val) / GRIPPER_MAX_OPEN
                if not st['contact']:
                    if f1 and f2 and c_objs and mano_curl > GRASP_TRIGGER_CURL:
                        st['contact'] = True
                        st['gval0'] = float(gripper_val)
                        st['obj'] = c_objs[0]
                        if not st['logged']:
                            st['logged'] = True
                            logger.info(f"  [mano-fr][{s}] 接触维持: obj={st['obj']}, "
                                        f"curl={mano_curl:.2f}, gval0={st['gval0']:.4f}")
                else:
                    if mano_curl < RELEASE_TRIGGER_CURL:
                        st['contact'] = False
                        st['gval0'] = None
                        st['obj'] = None
                    else:
                        clamp = CLAMP_OFFSET_MAX * max(mano_curl, CLAMP_CURL_FLOOR)
                        clamped = st['gval0'] - clamp
                        gripper_val = float(min(clamped, gripper_val))
            gripper_vals[s] = gripper_val

            # 臂基座坐标系 — 关键: base_quat 会旋转 ARM_BASE_OFFSET, 必须用旋转后的 offset
            # (旧版用未旋转 offset, 导致 base_link_p 错位 13cm, IK target 在错误坐标系)
            base_link_q = self._base_quat
            base_link_R = pr.matrix_from_quaternion(base_link_q)
            arm_base_offset_raw = ARM_BASE_OFFSET_LEFT if s == "left" else ARM_BASE_OFFSET_RIGHT
            arm_base_offset_rotated = base_link_R @ arm_base_offset_raw
            base_link_p = self._base_pos + arm_base_offset_rotated
            base_link_R_inv = base_link_R.T

            ik_target_b = base_link_R_inv @ (gripper_pos_fk - base_link_p)
            ee_R_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

            # Debug: 首帧打印 IK 输入
            if not hasattr(self, '_ik_debug_printed'):
                self._ik_debug_printed = True
                logger.info(f"  [IK debug][{s}] gripper_pos_fk={gripper_pos_fk.round(3)}")
                # 打印实际 EE 位置 (从 robot FK, 对比 IK 目标)
                try:
                    actual_ee = self._get_gripper_pose_from_retargeting(
                        self.robot_info["robot"].get_qpos(), s)
                    logger.info(f"  [IK debug][{s}] actual_ee_pos={np.array(actual_ee[0]).round(3)}")
                except Exception as e:
                    logger.info(f"  [IK debug][{s}] actual_ee 获取失败: {e}")
                logger.info(f"  [IK debug][{s}] base_link_p={base_link_p.round(3)} (computed from ROOT+offset)")
                logger.info(f"  [IK debug][{s}] ik_target_b (arm base frame)={ik_target_b.round(3)}")
                logger.info(f"  [IK debug][{s}] |ik_target_b|={np.linalg.norm(ik_target_b):.3f} (arm reach={ARM_MAX_REACH})")

            # IK 求解 (分别调用, 避免 solve_position_both 的双重转换 bug)
            solve_fn = (self.ik_solver.solve_position_right if s == "right"
                        else self.ik_solver.solve_position_left)
            ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
            for _ in range(IK_SOLVE_PER_FRAME - 1):
                ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))

            # per-side 滤波
            ik_joints = self.joint_filters[s].next(ik_joints)

            if s == "right" and len(ik_joints) == 6:
                ik_joints = np.clip(ik_joints, RIGHT_ARM_JOINT_LIMITS[:, 0], RIGHT_ARM_JOINT_LIMITS[:, 1])

            arm_targets[s] = ik_joints

        if is_bimanual:
            return arm_targets, gripper_vals
        else:
            return arm_targets[self.side], gripper_vals[self.side]

    def _step_gripper_only(self, joints_sapien):
        """gripper_only: 解析映射 → 夹爪位姿 + 关节角

        关键修复 (对齐 04_physics_simulation.py L2427-2430):
          - set_root_pose: kinematic 移动夹爪根 (跟随手腕)
          - set_qpos: 立即设置手指关节角 (不依赖 PD 自然收敛)
          - 主循环仍调 physics_step 设置 set_drive_target (PD 保持)
        旧版只 set_root_pose + drive_target, DECIMATION=8 步内手指来不及闭合 → "夹爪不动"

        hybrid 模式 (MANO+offset 中和态): EE = MANO 轨迹 + 常量偏移, 保持 MANO 运动形状
          (用户: "mano参数为主体, 你可以平移轨迹, 但不能离开轨迹, 偏移轨迹那么多")
        """
        local_idx = getattr(self, '_current_local_idx', 0)

        # === hybrid 模式: MANO+offset 中和态 (MANO 为主体, 常量平移对齐目标) ===
        if self.grasp_mode == "hybrid" and self.grasp_controllers is not None:
            neutral_target = self._compute_mano_neutral_target(local_idx, self.side)
            if neutral_target is not None:
                gripper_pos, gripper_R, gripper_val, phase = neutral_target
                root_quat = pr.quaternion_from_matrix(gripper_R)
                robot = self.robot_info["robot"]
                # 关键修复: LIFT/TRANSPORT 阶段不调 set_root_pose (teleport 破坏接触连续性,
                # 导致物体被弹飞后失去接触). APPROACH/DESCEND/CLOSE/RELEASE/RETREAT 可 teleport
                # (无接触, 或位置不变). 测试验证: 纯速度控制可提升物体 101.8mm.
                # TRANSPORT 同样需要保持接触 (用户: "放到碗里面"), 故与 LIFT 一致用 velocity 控制.
                if phase not in ("LIFT", "TRANSPORT"):
                    robot.set_root_pose(sapien.Pose(gripper_pos.tolist(), root_quat.tolist()))
                # 速度控制: 让物理引擎平滑移动根, 维持接触摩擦力.
                # LIFT/TRANSPORT 阶段: feedforward + P 反馈. feedforward 跟踪目标轨迹速度,
                # P 反馈纠正因接触力导致的根位置漂移 (KP=CONTROL_FREQ/4≈7.5, 每帧纠正 25% 误差).
                # 中和态下 ff_vel 来自 MANO 帧间位移 (保持 MANO 运动速度形状).
                # 非 LIFT/TRANSPORT 阶段: 纯 feedforward (root 被 teleport, 速度仅用于摩擦力).
                prev_pos = getattr(self, '_prev_demo_root_pos', None)
                if phase in ("LIFT", "TRANSPORT"):
                    actual_pos = np.array(robot.get_root_pose().p, dtype=np.float64)
                    pos_error = gripper_pos - actual_pos
                    if prev_pos is not None:
                        ff_vel = (gripper_pos - prev_pos) * float(CONTROL_FREQ)
                    else:
                        ff_vel = np.zeros(3)
                    root_vel = ff_vel + pos_error * (float(CONTROL_FREQ) / 4.0)
                    robot.set_root_linear_velocity(root_vel.tolist())
                else:
                    if prev_pos is not None:
                        root_vel = (gripper_pos - prev_pos) * float(CONTROL_FREQ)
                        robot.set_root_linear_velocity(root_vel.tolist())
                    else:
                        robot.set_root_linear_velocity([0.0, 0.0, 0.0])
                robot.set_root_angular_velocity([0.0, 0.0, 0.0])
                self._prev_demo_root_pos = gripper_pos.copy()
                joint1 = float(gripper_val)
                joint2 = float(gripper_val)
                if local_idx == 0 or local_idx % 30 == 0:
                    logger.info(f"  [neutral][{self.side}] F{local_idx}: phase={phase}, "
                                f"pos={gripper_pos.round(3)}, grip_cmd={gripper_val:.4f}")
                return (), (joint1, joint2)

        # === 非 hybrid: MANO 解析映射 (原有逻辑) ===
        wrist_pos = joints_sapien[0, :3]
        finger1_pos = joints_sapien[4, :3]
        finger2_pos = joints_sapien[8, :3]

        # prefix: 左手用 "left", 右手/双手用 "right" (对齐 04 的 y_sign 逻辑)
        prefix = self.side if self.side in ("left", "right") else "right"
        root_pos, root_R, joint1, joint2 = compute_analytical_gripper_pose(
            wrist_pos, finger1_pos, finger2_pos, prefix=prefix
        )

        # 轨迹跟踪: 记录 MANO 期望根位置 (限幅前 = 真实 MANO 轨迹, 用于误差验证)
        expected_root = root_pos.copy()
        # 根速度限制: kinematic 根瞬移会让动态物体跟不上 (惯性 + 摩擦限制)
        # 限制每帧根位置变化 ≤ MAX_ROOT_STEP, 防止提升中物体滑落 (glb_5 follow=8→10+)
        prev = getattr(self, "_prev_root_pos", None)
        if prev is not None:
            delta = root_pos - prev
            step = float(np.linalg.norm(delta))
            if step > MAX_ROOT_STEP:
                root_pos = prev + delta * (MAX_ROOT_STEP / step)
        self._prev_root_pos = root_pos.copy()

        root_quat = pr.quaternion_from_matrix(root_R)
        robot = self.robot_info["robot"]
        robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))
        # 关键修复: set_root_pose 不设置速度, 显式设置根线速度使摩擦力生效
        prev_vel_ref = getattr(self, "_prev_root_pos_vel", None)
        if prev_vel_ref is not None:
            root_vel = (root_pos - prev_vel_ref) * float(CONTROL_FREQ)
            robot.set_root_linear_velocity(root_vel.tolist())
        else:
            robot.set_root_linear_velocity([0.0, 0.0, 0.0])
        robot.set_root_angular_velocity([0.0, 0.0, 0.0])
        self._prev_root_pos_vel = root_pos.copy()

        # 立即设置手指 qpos (关键: 04 的做法, 让手指瞬间到位产生接触)
        gripper_idx1 = self.robot_info["gripper_idx1"]
        gripper_idx2 = self.robot_info["gripper_idx2"]
        qpos = robot.get_qpos().copy()
        qpos[gripper_idx1] = float(joint1)
        qpos[gripper_idx2] = float(joint2)

        # 轨迹跟踪: 记录 MANO 期望手指 (抓取调整前 = 真实 MANO 解析, 用于误差验证)
        expected_j1 = float(joint1)
        expected_j2 = float(joint2)
        # 自适应抓取: 用 MANO 意图 + 相位状态机决定夹爪开合
        # joint1/joint2: 0=闭合, GRIPPER_MAX_OPEN=张开 (与 full_robot 的 gripper_val 同约定)
        if self.grasp_mode == "hybrid" and self.grasp_controllers is not None:
            s = self.side
            controller = self.grasp_controllers.get(s)
            if controller is not None:
                mano_gripper_val = float(joint1)  # MANO 解析映射的夹爪开合
                current_qpos = qpos[gripper_idx1]  # 当前手指 qpos
                adapted_target, phase, grasp_info = controller.update(
                    root_pos, root_R, mano_gripper_val,
                    robot=robot, scene=self.scene, current_qpos=current_qpos
                )
                joint1 = adapted_target
                joint2 = adapted_target
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                if not hasattr(self, '_grasp_debug_printed'):
                    self._grasp_debug_printed = True
                    logger.info(f"  [hybrid][{s}] 首帧: phase={phase}, "
                                f"curl={grasp_info['mano_curl']:.2f}, "
                                f"obj={grasp_info['nearest_obj']}@{grasp_info['obj_dist']:.3f}m, "
                                f"force={grasp_info['grasp_force']:.1f}N")
        elif self.grasp_mode == "adaptive" and self.grasp_controllers is not None:
            s = self.side
            controller = self.grasp_controllers.get(s)
            if controller is not None:
                mano_gripper_val = float(joint1)  # MANO 解析映射的夹爪开合
                adapted_target, phase, grasp_info = controller.update(root_pos, mano_gripper_val)
                joint1 = adapted_target
                joint2 = adapted_target
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                if not hasattr(self, '_grasp_debug_printed'):
                    self._grasp_debug_printed = True
                    logger.info(f"  [grasp][{s}] 首帧: phase={phase}, "
                                f"curl={grasp_info['mano_curl']:.2f}, "
                                f"obj={grasp_info['nearest_obj']}@{grasp_info['obj_dist']:.3f}m")
        elif self.grasp_mode == "mano":
            # mano 模式: 纯重放 + 接触维持夹紧 (真正夹住物体, 用户: "实现能够真正的和物体交互")
            # 接触前: 跟随 MANO (纯重放, 当前行为)
            # 接触后: 记录 qpos_at_contact, 维持固定夹紧 (qpos0 - CLAMP_OFFSET*curl), 防止物体滑落
            # MANO 张开 (curl < RELEASE_TRIGGER_CURL) 时释放, 回到纯重放
            # 只维持已接触物体的夹紧, 不主动抓所有物体 (用户: "目前只有一个物体进行交互是对的")
            if not hasattr(self, '_mano_state'):
                self._mano_state = {'contact': False, 'qpos0': None, 'obj': None, 'logged': False}
            st = self._mano_state
            try:
                f1, f2, c_objs = get_finger_contacts(robot, self.side, self.scene, self.obj_actors)
            except Exception:
                f1, f2, c_objs = False, False, []
            mano_curl = 1.0 - float(joint1) / GRIPPER_MAX_OPEN
            current_qpos = qpos[gripper_idx1]
            if not st['contact']:
                # 接触前: 纯重放 (不改 qpos); 检测到双指接触 + MANO 卷曲 → 进入维持
                if f1 and f2 and c_objs and mano_curl > GRASP_TRIGGER_CURL:
                    st['contact'] = True
                    st['qpos0'] = current_qpos
                    st['obj'] = c_objs[0]
                    if not st['logged']:
                        st['logged'] = True
                        logger.info(f"  [mano][{self.side}] 接触维持: obj={st['obj']}, "
                                    f"curl={mano_curl:.2f}, qpos0={current_qpos:.4f}")
            else:
                # 接触后: MANO 张开 → 释放; 否则维持固定夹紧 (只夹紧不松开, 防 MANO 抖动)
                if mano_curl < RELEASE_TRIGGER_CURL:
                    st['contact'] = False
                    st['qpos0'] = None
                    st['obj'] = None
                else:
                    clamp = CLAMP_OFFSET_MAX * max(mano_curl, CLAMP_CURL_FLOOR)
                    clamped = st['qpos0'] - clamp
                    # min: 取更闭合的值 (MANO 若更闭合则跟随 MANO, 否则维持 clamped)
                    qpos[gripper_idx1] = float(min(clamped, qpos[gripper_idx1]))
                    qpos[gripper_idx2] = float(min(clamped, qpos[gripper_idx2]))
                    joint1 = qpos[gripper_idx1]
                    joint2 = qpos[gripper_idx2]

        # 轨迹跟踪误差: 物理输出(actual) vs 真实 MANO 期望(expected)
        # 用户: "夹爪运动要物理和真实输出的误差来判断准不准确"
        actual_j1 = float(qpos[gripper_idx1])
        actual_j2 = float(qpos[gripper_idx2])
        self._last_track = {
            'root_err_mm': float(np.linalg.norm(expected_root - root_pos) * 1000.0),  # 根位置误差 mm
            'j1_err_mm': float(abs(expected_j1 - actual_j1) * 1000.0),  # 手指1误差 mm
            'j2_err_mm': float(abs(expected_j2 - actual_j2) * 1000.0),  # 手指2误差 mm
        }
        robot.set_qpos(qpos)
        return None, (joint1, joint2)

    def _init_mano_markers(self):
        """初始化 MANO 3 参考点渲染节点 (wrist + finger1 + finger2)
        对齐 04_physics_simulation.py _render_keypoints L2029 的方式, 但只渲染 3 个点
        颜色: wrist=红, finger1=绿, finger2=蓝 (便于区分)
        """
        if not getattr(self.scene, "_render_available", True):
            return None
        try:
            import sapien.render
            self._mano_context = sapien.render.SapienRenderer()._internal_context
            self._mano_internal_scene = self.scene.render_system._internal_scene
            self._mano_marker_nodes = []
            # 创建 3 个不同颜色球体 (wrist=红, finger1=绿, finger2=蓝)
            colors = [
                np.array([1.0, 0.0, 0.0, 1.0]),  # wrist=红
                np.array([0.0, 1.0, 0.0, 1.0]),  # finger1=绿
                np.array([0.0, 0.5, 1.0, 1.0]),  # finger2=蓝
            ]
            for color in colors:
                mat = self._mano_context.create_material(np.zeros(4), color, 0.0, 0.5, 0)
                sphere = self._mano_context.create_uvsphere_mesh(12, 6)
                model = self._mano_context.create_model([sphere], [mat])
                node = self._mano_internal_scene.add_node()
                node.set_position([0, 0, 0])
                node.set_scale([0.015, 0.015, 0.015])  # 半径 1.5cm 球体
                obj = self._mano_internal_scene.add_object(model, node)
                obj.shading_mode = 0
                obj.cast_shadow = False
                obj.transparency = 0
                self._mano_marker_nodes.append(node)
            logger.info(f"  MANO 参考点渲染已初始化 (3 个球体: wrist=红/finger1=绿/finger2=蓝)")
            return self._mano_marker_nodes
        except Exception as e:
            logger.warning(f"  MANO 参考点渲染初始化失败: {e}")
            self._mano_marker_nodes = None
            return None

    def _update_mano_markers(self, wrist_pos, finger1_pos, finger2_pos):
        """每帧更新 MANO 3 参考点位置
        Args:
            wrist_pos: (3,) MANO 手腕在 SAPIEN 坐标系下的位置
            finger1_pos: (3,) MANO 指尖1 (joints[4]) 位置
            finger2_pos: (3,) MANO 指尖2 (joints[8]) 位置
        """
        nodes = getattr(self, "_mano_marker_nodes", None)
        if nodes is None or len(nodes) != 3:
            return
        try:
            positions = [wrist_pos, finger1_pos, finger2_pos]
            for node, pos in zip(nodes, positions):
                node.set_position(np.asarray(pos).tolist())
        except Exception:
            pass

    def run(self):
        """主仿真循环"""
        # 1. 对齐
        self._align_scene()

        # 2. 加载 HaWoR 数据
        logger.info("=" * 60)
        logger.info("Step 2: 加载 HaWoR 数据 + MANO FK")
        logger.info("=" * 60)
        # 双手模式加载两侧 HaWoR 数据; 单侧只加载一个
        if self.side == "both":
            hawor_data = {}
            mano_layer = {}
            betas_mean = {}
            for s in self.sides:  # ["left", "right"]
                hi = self.hand_indices[s]
                hawor_data[s] = load_hawor_data(self.hawor_dir, hand_idx=hi)
                betas_mean[s] = hawor_data[s]["pred_betas"][self.start_frame].astype(np.float32)
                mano_side = "left" if hi == 0 else "right"
                from mano_layer import MANOLayer
                mano_layer[s] = MANOLayer(mano_side, betas_mean[s])
            n_total = len(hawor_data[self.sides[0]]["pred_trans"])
        else:
            hawor_data = load_hawor_data(self.hawor_dir, hand_idx=self.hand_idx)
            n_total = len(hawor_data["pred_trans"])
            betas_mean = hawor_data["pred_betas"][self.start_frame].astype(np.float32)
            mano_side = "left" if self.hand_idx == 0 else "right"
            from mano_layer import MANOLayer
            mano_layer = MANOLayer(mano_side, betas_mean)
        if self.num_frames < 0 or self.num_frames > n_total - self.start_frame:
            self.num_frames = n_total - self.start_frame
        logger.info(f"  总帧数: {n_total}, 渲染: {self.start_frame}~{self.start_frame + self.num_frames - 1}")

        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)

        # 3. 预扫描 GLB 最低点 → 创建场景
        logger.info("=" * 60)
        logger.info("Step 3: 创建 SAPIEN 场景 + 加载 GLB")
        logger.info("=" * 60)
        glb_path = self.ras_dir / "final_scene.glb"
        ground_z = 0.0
        if glb_path.exists() and self.transform_params_path:
            ground_z = compute_glb_ground_z(glb_path, self.transform_params_path)
            logger.info(f"  GLB 预扫描地面高度: z={ground_z:.4f}")
        else:
            logger.warning(f"  GLB 不存在: {glb_path}")
        self._ground_z = ground_z

        # 地面高度: GLB 最低点 (支撑动态物体)
        # full_robot 的 ROOT 可能在地下, 通过禁用躯干碰撞避免干扰 (在 setup_robot 中处理)
        self.scene = setup_physics_scene(ground_height=ground_z)
        render_available = getattr(self.scene, "_render_available", False)

        self.obj_bbox_centers = {}
        self.obj_info = {}
        if glb_path.exists() and self.transform_params_path:
            self.obj_actors, _, self.obj_bbox_centers, self.obj_info = load_glb_with_physics(
                glb_path, self.transform_params_path, self.scene, fast_collision=True  # 单凸包 (CoACD 太慢, 可按需切换 False)
            )
        else:
            self.obj_actors = []

        # 初始化抓取控制器 (延迟到 setup_robot 之后, 因为 hybrid 需要 robot 对象)
        self.grasp_controllers = None
        if self.grasp_mode == "hybrid":
            # hybrid 模式在 setup_robot 后初始化 (需要 robot 对象)
            pass
        elif self.grasp_mode == "adaptive":
            self.grasp_controllers = {
                s: AdaptiveGraspController(self.obj_actors, side=s) for s in self.sides
            }
            logger.info(f"  自适应抓取控制器已启用 (grasp_mode=adaptive), 阈值: "
                        f"触发卷曲>{GRASP_TRIGGER_CURL}, 释放卷曲<{RELEASE_TRIGGER_CURL}")
        else:
            logger.info(f"  纯 MANO 重放模式 (grasp_mode=mano), 夹爪直接跟随 MANO 手指")

        # 4. 计算手腕位置 → 最优基座
        logger.info("=" * 60)
        logger.info("Step 4: 计算最优基座位置")
        logger.info("=" * 60)
        wrist_positions = []
        # 同时预计算 MANO 解析夹爪轨迹 (用户: "mano参数为主体, 可以平移轨迹")
        # 每帧: root_pos, j1, j2 from compute_analytical_gripper_pose
        mano_gripper_traj = {}  # side -> {"pos": [], "j1": [], "j2": []}
        if self.side == "both":
            # 双手: 收集两侧手腕位置 + 夹爪轨迹
            for fi in range(self.start_frame, self.start_frame + self.num_frames):
                for s in self.sides:
                    hd = hawor_data[s]
                    if fi < len(hd["pred_valid"]) and not hd["pred_valid"][fi]:
                        continue
                    if fi >= len(hd["pred_trans"]):
                        continue
                    try:
                        _, joints = compute_mano_joints(
                            mano_layer[s],
                            hd["pred_rot"][fi],
                            hd["pred_hand_pose"][fi],
                            hd["pred_trans"][fi],
                        )
                        joints_sapien = (RXWORLD_TO_SAPIEN @ joints.T).T
                        wrist_positions.append(joints_sapien[0, :3])
                        # 预计算夹爪位姿 (用于 hybrid 中和态)
                        if s not in mano_gripper_traj:
                            mano_gripper_traj[s] = {"pos": [], "j1": [], "j2": []}
                        root_pos, _, j1, j2 = compute_analytical_gripper_pose(
                            joints_sapien[0, :3], joints_sapien[4, :3],
                            joints_sapien[8, :3], prefix=s,
                        )
                        mano_gripper_traj[s]["pos"].append(root_pos)
                        mano_gripper_traj[s]["j1"].append(j1)
                        mano_gripper_traj[s]["j2"].append(j2)
                    except Exception:
                        continue
        else:
            for fi in range(self.start_frame, self.start_frame + self.num_frames):
                if fi < len(hawor_data["pred_valid"]) and not hawor_data["pred_valid"][fi]:
                    continue
                if fi >= len(hawor_data["pred_trans"]):
                    continue
                try:
                    _, joints = compute_mano_joints(
                        mano_layer,
                        hawor_data["pred_rot"][fi],
                        hawor_data["pred_hand_pose"][fi],
                        hawor_data["pred_trans"][fi],
                    )
                    joints_sapien = (RXWORLD_TO_SAPIEN @ joints.T).T
                    wrist_positions.append(joints_sapien[0, :3])
                    # 预计算夹爪位姿 (用于 hybrid 中和态)
                    s = self.side
                    if s not in mano_gripper_traj:
                        mano_gripper_traj[s] = {"pos": [], "j1": [], "j2": []}
                    root_pos, _, j1, j2 = compute_analytical_gripper_pose(
                        joints_sapien[0, :3], joints_sapien[4, :3],
                        joints_sapien[8, :3], prefix=s,
                    )
                    mano_gripper_traj[s]["pos"].append(root_pos)
                    mano_gripper_traj[s]["j1"].append(j1)
                    mano_gripper_traj[s]["j2"].append(j2)
                except Exception:
                    continue
        # 转为 numpy 数组, 存储到实例 (供 hybrid 中和态使用)
        for key in mano_gripper_traj:
            mano_gripper_traj[key]["pos"] = np.array(mano_gripper_traj[key]["pos"], dtype=np.float64)
            mano_gripper_traj[key]["j1"] = np.array(mano_gripper_traj[key]["j1"], dtype=np.float64)
            mano_gripper_traj[key]["j2"] = np.array(mano_gripper_traj[key]["j2"], dtype=np.float64)
        self._mano_gripper_traj = mano_gripper_traj

        # Debug: 手腕 vs 物体位置
        if wrist_positions:
            warr = np.array(wrist_positions)
            logger.info(f"  [debug] 手腕位置范围: x=[{warr[:,0].min():.3f},{warr[:,0].max():.3f}] "
                        f"y=[{warr[:,1].min():.3f},{warr[:,1].max():.3f}] "
                        f"z=[{warr[:,2].min():.3f},{warr[:,2].max():.3f}]")
            logger.info(f"  [debug] 手腕质心: {warr.mean(axis=0).round(3)}")
        for name, center in self.obj_bbox_centers.items():
            logger.info(f"  [debug] 物体 {name} bbox中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

        if self.mode == "full_robot":
            self._base_pos, self._base_quat = self._compute_optimal_base(wrist_positions, R_c2w_all)
        else:
            # gripper_only: 基座跟随手腕, 初始位置用第一帧
            if wrist_positions:
                self._base_pos = wrist_positions[0]
                self._base_quat = pr.quaternion_from_axis_angle(np.array([0, 0, 1, 0]))
            else:
                self._base_pos = np.array([0.0, 0.0, 0.3])
                self._base_quat = pr.quaternion_from_axis_angle(np.array([0, 0, 1, 0]))

        # 5. 加载机器人
        logger.info("=" * 60)
        logger.info(f"Step 5: 加载机器人 (mode={self.mode})")
        logger.info("=" * 60)
        self.robot_info = setup_robot(
            self.scene, self.mode, self.side, self._base_pos, self._base_quat
        )

        # Debug: 验证臂基座位置
        if self.mode == "full_robot":
            robot = self.robot_info["robot"]
            arm_base_offset_expected = ARM_BASE_OFFSET_LEFT if self.side == "left" else ARM_BASE_OFFSET_RIGHT
            for link in robot.get_links():
                if link.get_name() == f"{self.side}_arm_base_link":
                    abp = np.array(link.get_entity_pose().p)
                    logger.info(f"  [verify] ROOT={self._base_pos.round(3)}, "
                                f"arm_base={abp.round(3)}, "
                                f"offset={(abp - self._base_pos).round(3)}, "
                                f"expected_offset={arm_base_offset_expected.round(3)}")
                    break

        # 6. 初始化 Retargeting + IK
        self._init_retargeting()
        self._init_ik()

        # 6b. 初始化 hybrid 抓取控制器 (需要在 setup_robot 之后, 因为需要 robot 对象)
        if self.grasp_mode == "hybrid" and self.grasp_controllers is None:
            # 目标选择范式 (用户: "我需要夹住的是那个粉色的东西，放到碗里面"; "形成一个范式")
            # 1. 优先按颜色识别粉色物体 (find_pink_object), 在不同场景文件夹通用
            # 2. 找不到粉色物体时, 退回轨迹距离选择 (find_target_object_by_trajectory)
            # 3. 同时识别碗 (find_bowl, 按几何特征), 作为 pick-and-place 放置目标
            obj_sapien_pos = {name: np.array(c) for name, c in self.obj_bbox_centers.items()}
            target_objs = {}
            bowl_objs = {}

            # 1. 按颜色识别粉色物体 (范式, 通用)
            pink_obj = find_pink_object(self.obj_info)
            # 2. 识别碗 (按几何, 排除粉色物体)
            bowl_obj = find_bowl(self.obj_info, exclude_names=[pink_obj] if pink_obj else None)
            logger.info(f"  [范式] 粉色物体={pink_obj}, 碗={bowl_obj}")

            for s in self.sides:
                tgt = pink_obj
                # 退回: 无粉色物体时用轨迹距离选择
                if tgt is None:
                    if self.side == "both" and isinstance(hawor_data, dict) and s in hawor_data:
                        trans_side = hawor_data[s]["pred_trans"]
                    else:
                        trans_side = hawor_data["pred_trans"]
                    tgt = find_target_object_by_trajectory(np.asarray(trans_side), obj_sapien_pos)
                target_objs[s] = tgt
                bowl_objs[s] = bowl_obj
                if tgt:
                    tgt_pos = obj_sapien_pos[tgt]
                    logger.info(f"  [{s}] 锁定抓取目标: {tgt} (sapien_pos={tgt_pos.round(3)})")
                if bowl_obj:
                    bowl_pos = obj_sapien_pos[bowl_obj]
                    logger.info(f"  [{s}] 锁定放置目标 (碗): {bowl_obj} (sapien_pos={bowl_pos.round(3)})")
            self.grasp_controllers = {
                s: HybridGraspController(
                    self.obj_actors, side=s, scene=self.scene,
                    robot=self.robot_info.get("robot"),
                    target_obj=target_objs.get(s),
                    obj_positions=self.obj_bbox_centers,
                    bowl_obj=bowl_objs.get(s)
                ) for s in self.sides
            }
            logger.info(f"  混合抓取控制器已启用 (grasp_mode=hybrid), "
                        f"MANO curl→力度, 接触力控, 目标力={TARGET_GRASP_FORCE}N, "
                        f"抓取目标: {target_objs}, 放置目标 (碗): {bowl_objs}")

            # 6c. 计算 MANO+offset 中和态偏移量 (用户: "mano参数为主体, 可以平移轨迹, 但不能离开轨迹")
            # 找到 MANO 夹爪最接近目标物体的帧, 计算常量偏移对齐到正确抓取位置
            # EE = mano_root_pos[f] + offset, 保持 MANO 轨迹形状, 偏移最小化
            self._gripper_R_fixed = np.array([
                [0, 1, 0],
                [0, 0, -1],
                [-1, 0, 0]
            ], dtype=np.float64)
            FINGER_FORWARD_OFFSET_NEUTRAL = 0.037
            ee_offset_neutral = -self._gripper_R_fixed[:, 0] * FINGER_FORWARD_OFFSET_NEUTRAL
            self._mano_neutral_offset = {}
            self._mano_grasp_frame = {}  # side -> f_grasp (MANO 最接近目标的帧, 用于阶段判定)
            for s in self.sides:
                tgt = target_objs.get(s)
                traj = self._mano_gripper_traj.get(s)
                if tgt is None or tgt not in self.obj_bbox_centers or traj is None or len(traj["pos"]) == 0:
                    self._mano_neutral_offset[s] = None
                    self._mano_grasp_frame[s] = None
                    continue
                target_pos = np.array(self.obj_bbox_centers[tgt], dtype=np.float64)
                target_grasp_pos = target_pos + ee_offset_neutral
                # 找 MANO 夹爪最接近目标物体的帧 (offset 最小化的对齐点)
                mano_positions = traj["pos"]
                dists = np.linalg.norm(mano_positions - target_pos, axis=1)
                f_grasp = int(np.argmin(dists))
                offset = target_grasp_pos - mano_positions[f_grasp]
                self._mano_neutral_offset[s] = offset
                self._mano_grasp_frame[s] = f_grasp
                logger.info(f"  [neutral][{s}] MANO 最接近 {tgt} @ F{f_grasp} "
                            f"(dist={dists[f_grasp]:.3f}m), offset={offset.round(3)} "
                            f"(MANO@F{f_grasp}={mano_positions[f_grasp].round(3)}, "
                            f"target_grasp={target_grasp_pos.round(3)})")

        # 7. 相机设置 (按 --views 指定渲染哪些视角, 对齐 02_render_scene.py)
        cam_view = None
        god_view = None
        focal = float(hawor_data.get("img_focal", HAWOR_FOCAL_DEFAULT))
        render_cam = render_available and self.views in ("cam", "both")
        render_god = render_available and self.views in ("god", "both")
        if render_cam:
            # 7a. 相机视角 (第一人称): 用 02 的 hawor_cam_to_sapien_pose (正确映射)
            cam_view = self.scene.add_camera(
                "cam_view", CAM_WIDTH, CAM_HEIGHT,
                1.0, 0.01, 100.0,
            )
            R_c2w = R_c2w_all[min(self.start_frame, len(R_c2w_all) - 1)]
            t_c2w = t_c2w_all[min(self.start_frame, len(t_c2w_all) - 1)]
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w, t_c2w)
            cam_view.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            cam_view.set_focal_lengths(focal, focal)
            logger.info(f"  相机视角 (第一人称): pos={cam_pos.round(3)}, focal={focal:.1f}")

        if render_god:
            # 7b. 上帝视角: 高空斜俯视, 能看到整个机器人 + 夹爪 + 物体
            # 场景中心 = 抓取区域 (臂基座 + 物体质心), 不是地下 ROOT
            god_view = self.scene.add_camera(
                "god_view", CAM_WIDTH, CAM_HEIGHT,
                1.0, 0.01, 100.0,
            )
            arm_base_offset = ((ARM_BASE_OFFSET_LEFT + ARM_BASE_OFFSET_RIGHT) / 2.0) if self.side == "both" \
                else (ARM_BASE_OFFSET_LEFT if self.side == "left" else ARM_BASE_OFFSET_RIGHT)
            arm_base_pos = self._base_pos + arm_base_offset  # 臂基座 (地上, ~z=0.35)
            if self.obj_bbox_centers:
                obj_centers = np.array(list(self.obj_bbox_centers.values()))
                obj_centroid = obj_centers.mean(axis=0)
                # 场景中心 = 臂基座和物体的中间 (抓取区域)
                scene_center = (arm_base_pos + obj_centroid) / 2.0
            else:
                scene_center = arm_base_pos.copy()
            # 上帝视角: 放在机器人前方高处, 俯视抓取区域
            # 用 base_quat (含 yaw) 旋转 [1,0,0] (URDF 默认前方) 得到机器人当前前方
            # 修复: 旧版用世界 -Y, gripper_only 机器人 yaw 旋转后前方不是 -Y → 视角反了
            try:
                base_q = self._base_quat if self._base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
                R_base = pr.matrix_from_quaternion(base_q)
                forward_3d = R_base @ np.array([1.0, 0.0, 0.0])
                forward_2d = np.array([forward_3d[0], forward_3d[1], 0.0])
                norm = float(np.linalg.norm(forward_2d))
                if norm > 1e-6:
                    forward_2d = forward_2d / norm
                else:
                    forward_2d = np.array([0.0, -1.0, 0.0])
            except Exception:
                forward_2d = np.array([0.0, -1.0, 0.0])
            # god_view 固定相机在正上方俯瞰整个场景 (用户: "固定的摄像头在上方, 不要有什么跟随的操作")
            # 高度足够覆盖整个抓取工作空间 (物体 + 夹爪活动范围)
            # 用户反馈: "god视角太高了, 得低一点" → 1.0m → 0.5m → 0.2m (用户: "god降低到0.2")
            # pick-and-place 需要看到物体被放入碗的全过程, 高度低更清晰
            if self.mode == "gripper_only":
                god_height = 0.20  # 上方 0.2m (用户: "god降低到0.2")
            else:
                god_height = 1.50   # 上方 1.5m (俯瞰全机器人 + 抓取区域)
            # 相机在场景中心正上方, 往下看; up 方向用 forward_2d
            god_pos = scene_center + np.array([0.0, 0.0, god_height])
            god_quat = make_look_at_camera(god_pos, scene_center, up=forward_2d)
            god_view.set_local_pose(sapien.Pose(god_pos.tolist(), god_quat.tolist()))
            god_view.set_focal_lengths(focal, focal)
            # 固定相机: 不跟随夹爪, 初始化后位置不变
            self._god_height = god_height
            self._god_follow = False
            logger.info(f"  上帝视角: pos={god_pos.round(3)}, 看向={scene_center.round(3)} "
                        f"(height={god_height}m, 固定正上方俯瞰, 不跟随)")

        # 8. 视频录制 (按 --views 仅创建需要的 writer)
        video_path_cam = str(self.output_dir / f"cam_view_{self.mode}_{self.side}.mp4")
        video_path_god = str(self.output_dir / f"god_view_{self.mode}_{self.side}.mp4")
        writer_cam = None
        writer_god = None
        if render_cam:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_cam = cv2.VideoWriter(video_path_cam, fourcc, 30, (CAM_WIDTH, CAM_HEIGHT))
        if render_god:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_god = cv2.VideoWriter(video_path_god, fourcc, 30, (CAM_WIDTH, CAM_HEIGHT))

        # 8b. 初始化 MANO 3 参考点渲染 (仅渲染模式下创建)
        if render_available:
            self._init_mano_markers()

        # 9. Warmup (smoothstep 过渡)
        logger.info("=" * 60)
        logger.info("Step 6: Warmup + 物理仿真")
        logger.info("=" * 60)
        robot = self.robot_info["robot"]
        arm_joint_indices = self.robot_info["arm_joint_indices"]
        gripper_idx1 = self.robot_info["gripper_idx1"]
        gripper_idx2 = self.robot_info["gripper_idx2"]
        joint_names = self.robot_info["joint_names"]

        # Warmup: 让物理稳定
        if self.side == "both":
            # 双手: 一次 physics_step 设置两侧臂 + 夹爪起始位置
            left_arm_idxs = [i for i, n in enumerate(joint_names) if "left_arm_joint" in n]
            right_arm_idxs = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
            gi_l = self.robot_info["gripper_indices"]["left"]
            gi_r = self.robot_info["gripper_indices"]["right"]
            for _ in range(WARMUP_FRAMES):
                physics_step(
                    robot, left_arm_idxs, gi_l[0], gi_l[1],
                    LEFT_ARM_STARTING, GRIPPER_INIT_OPEN, -GRIPPER_INIT_OPEN, self.scene,
                    extra_gripper_indices=[(gi_r[0], GRIPPER_INIT_OPEN), (gi_r[1], -GRIPPER_INIT_OPEN)],
                    extra_arm_indices=right_arm_idxs, extra_arm_target=RIGHT_ARM_STARTING,
                )
        else:
            for _ in range(WARMUP_FRAMES):
                physics_step(
                    robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                    RIGHT_ARM_STARTING if self.side == "right" else LEFT_ARM_STARTING,
                    GRIPPER_INIT_OPEN, -GRIPPER_INIT_OPEN, self.scene
                )

        # 10. 主循环
        qpos_log = []
        contact_log = []
        obj_pos_log = []  # 参数级验证: 物体位置
        gripper_pos_log = []  # 综合抓取判定: 每帧每侧夹爪位置 (跟随判定用)
        track_log = []  # 轨迹跟踪误差: 每帧 MANO 期望 vs SAPIEN 实际 (root + finger)
        grasp_states = []

        # 记录物体初始位置 (用于验证提升)
        obj_initial_pos = {}
        for actor in self.obj_actors:
            obj_initial_pos[actor.name] = np.array(actor.get_pose().p)

        # 无效帧保持上一帧位姿 (跟踪丢失时机器人保持不动, 不是自由落体)
        if self.side == "both":
            last_arm_target = {}  # {"left": array, "right": array}
            last_gripper_t1 = {}
            last_gripper_t2 = {}
        else:
            last_arm_target = None
            last_gripper_t1 = GRIPPER_INIT_OPEN
            last_gripper_t2 = -GRIPPER_INIT_OPEN

        # 双手模式预计算臂/夹爪索引 (避免每帧重复查找)
        if self.side == "both":
            left_arm_idxs = [i for i, n in enumerate(joint_names) if "left_arm_joint" in n]
            right_arm_idxs = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
            gi_l = self.robot_info["gripper_indices"]["left"]
            gi_r = self.robot_info["gripper_indices"]["right"]

        def _is_frame_invalid(hd, gi):
            """检查指定侧的帧是否无效"""
            if gi >= len(hd["pred_valid"]) or not hd["pred_valid"][gi]:
                return True
            if gi >= len(hd["pred_trans"]):
                return True
            if np.isnan(hd["pred_trans"][gi]).any():
                return True
            return False

        pbar = trange(self.num_frames, desc=f"grasp_{self.mode}_{self.side}")
        for local_idx in pbar:
            global_idx = self.start_frame + local_idx
            if global_idx >= n_total:
                break

            # 存储 _current_local_idx 供 _step_full_robot / _step_gripper_only 使用 (hybrid 模式)
            self._current_local_idx = local_idx

            if self.side == "both":
                # 双手: 任一侧无效则整帧无效
                is_invalid = any(_is_frame_invalid(hawor_data[s], global_idx) for s in self.sides)
            else:
                is_invalid = _is_frame_invalid(hawor_data, global_idx)

            if is_invalid:
                # 无效帧: 保持上一帧位姿 (跟踪丢失时机器人保持不动, 不是自由落体)
                if self.side == "both":
                    if last_arm_target:  # dict 非空
                        physics_step(
                            robot, left_arm_idxs, gi_l[0], gi_l[1],
                            last_arm_target["left"], last_gripper_t1["left"], last_gripper_t2["left"], self.scene,
                            extra_gripper_indices=[(gi_r[0], last_gripper_t1["right"]), (gi_r[1], last_gripper_t2["right"])],
                            extra_arm_indices=right_arm_idxs, extra_arm_target=last_arm_target["right"],
                        )
                    else:
                        _floating = _is_floating_root(robot)
                        for _ in range(DECIMATION):
                            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                            if _floating:
                                qf[:6] = 0
                            robot.set_qf(qf)
                            self.scene.step()
                else:
                    if last_arm_target is not None:
                        physics_step(
                            robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                            last_arm_target, last_gripper_t1, last_gripper_t2, self.scene
                        )
                    else:
                        _floating = _is_floating_root(robot)
                        for _ in range(DECIMATION):
                            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                            if _floating:
                                qf[:6] = 0
                            robot.set_qf(qf)
                            self.scene.step()
                qpos_log.append(robot.get_qpos().copy())
                n_contacts, total_impulse, per_obj = 0, 0.0, {}
                is_grasping = False
            else:
                # 有效帧: MANO FK → retargeting/IK → 物理步进
                if self.side == "both":
                    # 双手: 分别计算两侧 MANO FK
                    joints_dict = {}
                    for s in self.sides:
                        hd = hawor_data[s]
                        _, joints = compute_mano_joints(
                            mano_layer[s],
                            hd["pred_rot"][global_idx],
                            hd["pred_hand_pose"][global_idx],
                            hd["pred_trans"][global_idx],
                        )
                        joints_dict[s] = (RXWORLD_TO_SAPIEN @ joints.T).T

                    if self.mode == "full_robot":
                        arm_targets, gripper_vals = self._step_full_robot(joints_dict)
                        physics_step(
                            robot, left_arm_idxs, gi_l[0], gi_l[1],
                            arm_targets["left"], gripper_vals["left"], -gripper_vals["left"], self.scene,
                            extra_gripper_indices=[(gi_r[0], gripper_vals["right"]), (gi_r[1], -gripper_vals["right"])],
                            extra_arm_indices=right_arm_idxs, extra_arm_target=arm_targets["right"],
                        )
                        # 记录状态 (供无效帧保持位姿)
                        last_arm_target = arm_targets
                        last_gripper_t1 = {s: gripper_vals[s] for s in self.sides}
                        last_gripper_t2 = {s: -gripper_vals[s] for s in self.sides}
                    else:
                        # gripper_only + both: setup_robot 已 warn, 仅保持初始位姿
                        _floating = _is_floating_root(robot)
                        for _ in range(DECIMATION):
                            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                            if _floating:
                                qf[:6] = 0
                            robot.set_qf(qf)
                            self.scene.step()

                    # 更新 MANO 3 参考点位置 (用左侧 hand 渲染, 双手时只渲染左)
                    first_js = joints_dict.get(self.sides[0])
                    if first_js is not None and len(first_js) >= 9:
                        self._update_mano_markers(
                            first_js[0, :3], first_js[4, :3], first_js[8, :3]
                        )
                else:
                    # 单侧: MANO FK → retargeting/IK → 物理步进
                    _, joints = compute_mano_joints(
                        mano_layer,
                        hawor_data["pred_rot"][global_idx],
                        hawor_data["pred_hand_pose"][global_idx],
                        hawor_data["pred_trans"][global_idx],
                    )
                    joints_sapien = (RXWORLD_TO_SAPIEN @ joints.T).T

                    if self.mode == "full_robot":
                        ik_joints, gripper_val = self._step_full_robot(joints_sapien)
                        arm_target = ik_joints
                        gripper_t1 = gripper_val
                        gripper_t2 = -gripper_val
                    else:
                        _, (joint1, joint2) = self._step_gripper_only(joints_sapien)
                        arm_target = np.array([])
                        gripper_t1 = joint1
                        gripper_t2 = joint2

                    # 物理步进 (纯 PD 驱动, 不用 set_qpos — physics_step 注释说会震荡)
                    # 用户: "像 grasp_demo.py 一样真正的抓取物体" — grasp_demo.py 用 PD 控制器跟随,
                    # 不用 set_qpos kinematic 移动 (会破坏物理接触力传播).
                    # PD 滞后通过延长 DESCEND 阶段 + smoothstep 平滑插值缓解.
                    physics_step(
                        robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                        arm_target, gripper_t1, gripper_t2, self.scene
                    )

                    # 记录上一帧目标 (供无效帧保持位姿)
                    last_arm_target = arm_target
                    last_gripper_t1 = gripper_t1
                    last_gripper_t2 = gripper_t2

                    # 更新 MANO 3 参考点位置 (wrist + finger1 + finger2)
                    if len(joints_sapien) >= 9:
                        self._update_mano_markers(
                            joints_sapien[0, :3], joints_sapien[4, :3], joints_sapien[8, :3]
                        )

                qpos_log.append(robot.get_qpos().copy())

                # Debug: 每 10 帧打印 EE 位置 vs 最近物体 (遍历所有侧)
                if (local_idx + 1) % 10 == 0 or local_idx == 0:
                    for s in self.sides:
                        ee_pos = None
                        for link in robot.get_links():
                            if link.get_name() == f"{s}_gripper_link":
                                ee_pos = np.array(link.get_entity_pose().p)
                                break
                        if ee_pos is not None and self.obj_bbox_centers:
                            dists = []
                            for name, center in self.obj_bbox_centers.items():
                                d = float(np.linalg.norm(ee_pos - np.array(center)))
                                dists.append((name, d, center))
                            dists.sort(key=lambda x: x[1])
                            name, d, center = dists[0]
                            logger.info(f"  [debug] F{local_idx+1} [{s}]: ee={ee_pos.round(3)}, "
                                        f"nearest={name} dist={d:.3f} center={np.array(center).round(3)}")
                            # Debug finger positions + glb_6 contacts during CLOSE→LIFT (F55-F113)
                            if 55 <= local_idx + 1 <= 113:
                                finger_positions = []
                                for link in robot.get_links():
                                    lname = link.get_name()
                                    if "finger" in lname and s in lname:
                                        fp = np.array(link.get_entity_pose().p)
                                        finger_positions.append(f"{lname}={fp.round(4).tolist()}")
                                if finger_positions:
                                    logger.info(f"  [debug] F{local_idx+1} [{s}] fingers: {' | '.join(finger_positions)}")

                # 接触检测
                n_contacts, total_impulse, per_obj = fetch_contacts(
                    robot, self.obj_actors, self.side, self.scene
                )
                is_grasping = n_contacts >= 2

                # glb_6 接触详情 + z 位置 (诊断 pad 是否真的碰到目标物体)
                if 60 <= local_idx + 1 <= 113:
                    glb6_actor = None
                    for ac in self.obj_actors:
                        if ac.name == "glb_6":
                            glb6_actor = ac
                            break
                    glb6_pos = np.array(glb6_actor.get_pose().p).round(4).tolist() if glb6_actor else None
                    glb6_c = per_obj.get("glb_6", {"n": 0, "impulse": 0.0})
                    logger.info(f"  [debug] F{local_idx+1} glb_6: pos={glb6_pos} "
                                f"contact_n={glb6_c['n']} impulse={glb6_c['impulse']:.4f} "
                                f"total_contacts={n_contacts}")

            # 公共: 日志记录 + 渲染 (有效帧和无效帧都执行, 确保视频帧数 == num_frames)
            contact_log.append({
                "frame": local_idx,
                "contacts": n_contacts,
                "impulse": total_impulse,
                "per_obj": per_obj,
            })
            grasp_states.append(is_grasping)

            # 物体位置记录 (参数级验证)
            frame_obj_pos = {}
            for actor in self.obj_actors:
                frame_obj_pos[actor.name] = np.array(actor.get_pose().p).tolist()
            obj_pos_log.append(frame_obj_pos)

            # 夹爪位置记录 (综合抓取判定: 跟随判定用) + 位姿 (相机跟随用)
            frame_gripper_pos = {}
            frame_gripper_pose = {}  # {side: sapien.Pose} 含旋转, 用于相机跟随夹爪实际朝向
            for s in self.sides:
                for link in robot.get_links():
                    if link.get_name() == f"{s}_gripper_link":
                        pose = link.get_entity_pose()
                        frame_gripper_pos[s] = np.array(pose.p).tolist()
                        frame_gripper_pose[s] = pose
                        break
            gripper_pos_log.append(frame_gripper_pos)
            # 轨迹跟踪误差收集 (物理输出 vs 真实 MANO 期望)
            track_log.append(getattr(self, '_last_track', None))

            # 渲染 (按 --views 渲染选定视角)
            # GPU 可能在运行时丢失 (vk::DeviceLostError), 捕获后降级为纯物理模式
            if render_available and (writer_cam is not None or writer_god is not None):
                try:
                    self.scene.update_render()
                    if writer_cam is not None:
                        # 第一人称: 每帧按 HaWoR 相机轨迹更新 (对齐 02_render_scene.py L2588-2590)
                        # 之前只初始化一次, 导致相机不动
                        global_idx = self.start_frame + local_idx
                        if global_idx < len(R_c2w_all):
                            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                            cam_view.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                        # 相机视角 (第一人称)
                        cam_view.take_picture()
                        rgb_cam = cam_view.get_picture("Color")[..., :3]
                        bgr_cam = np.ascontiguousarray((np.clip(rgb_cam, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        h, w = bgr_cam.shape[:2]
                        cv2.rectangle(bgr_cam, (0, 0), (w, 40), (0, 0, 0), -1)
                        label = f"F{local_idx+1}/{self.num_frames} | {self.mode} | CamView | C:{n_contacts} | Grasp:{is_grasping}"
                        cv2.putText(bgr_cam, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if is_grasping else (255, 255, 255), 2)
                        writer_cam.write(bgr_cam)
                    if writer_god is not None:
                        # 上帝视角: 固定相机在正上方俯瞰 (用户: "固定的摄像头在上方, 不要有什么跟随的操作")
                        # 初始化时已 set_local_pose, 这里只 take_picture, 不再每帧更新位置
                        god_view.take_picture()
                        rgb_god = god_view.get_picture("Color")[..., :3]
                        bgr_god = np.ascontiguousarray((np.clip(rgb_god, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        h, w = bgr_god.shape[:2]
                        cv2.rectangle(bgr_god, (0, 0), (w, 40), (0, 0, 0), -1)
                        label = f"F{local_idx+1}/{self.num_frames} | {self.mode} | GodView | C:{n_contacts} | Grasp:{is_grasping}"
                        cv2.putText(bgr_god, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if is_grasping else (255, 255, 255), 2)
                        writer_god.write(bgr_god)
                except RuntimeError as e:
                    # vk::DeviceLostError 等 GPU 运行时错误: 永久禁用渲染, 继续物理仿真
                    logger.warning(f"  GPU 渲染失败 ({e}), 降级为纯物理模式 (后续帧不再渲染)")
                    render_available = False
                    for w_ in (writer_cam, writer_god):
                        if w_ is not None:
                            try:
                                w_.release()
                            except Exception:
                                pass
                    writer_cam = None
                    writer_god = None

            if (local_idx + 1) % 30 == 0:
                obj_detail = " | ".join(f"{k}:C={v['n']}" for k, v in per_obj.items())
                logger.info(f"  帧 {local_idx+1}/{self.num_frames}: contacts={n_contacts}, "
                            f"impulse={total_impulse:.2f}, grasp={is_grasping}, {obj_detail}")

        if writer_cam is not None:
            writer_cam.release()
            logger.info(f"  相机视角视频已保存: {video_path_cam}")
        if writer_god is not None:
            writer_god.release()
            logger.info(f"  上帝视角视频已保存: {video_path_god}")

        # 11. 参数级验证
        self._verify_results(qpos_log, contact_log, obj_pos_log, obj_initial_pos, grasp_states, gripper_pos_log, track_log)

        # 12. 保存结果
        qpos_path = str(self.output_dir / f"grasp_{self.mode}_{self.side}_qpos.npy")
        np.save(qpos_path, np.array(qpos_log))
        logger.info(f"  qpos 已保存: {qpos_path}")

        return {
            "video_cam": video_path_cam if render_available else None,
            "video_god": video_path_god if render_available else None,
            "qpos": qpos_path,
            "grasp_states": grasp_states,
        }

    def _verify_results(self, qpos_log, contact_log, obj_pos_log, obj_initial_pos, grasp_states, gripper_pos_log=None, track_log=None):
        """参数级验证: 物体提升 + 接触检测 + 综合抓取判定

        综合判定 (用户要求): 接触 + 跟随 + 提升 三项都满足才算真正夹住
          1. 接触: 连续 >= 10 帧有接触 (n_contacts >= 2)
          2. 跟随: 物体相对最近夹爪位置变化 < 5cm 持续 10 帧 (物体被夹住跟着夹爪走)
          3. 提升: 物体 z 提升量 > 5cm (相对初始位置)
        """
        logger.info("=" * 60)
        logger.info("Step 7: 参数级验证")
        logger.info("=" * 60)

        # 1. 机械臂关节数验证 (确认 "没有机械臂" 问题已修复)
        n_arm = len(self.robot_info["arm_joint_indices"])
        gripper_indices = self.robot_info["gripper_indices"]
        n_gripper = sum(2 for gi1, gi2 in gripper_indices.values() if gi1 is not None)
        expected_arm = 12 if self.side == "both" else 6
        logger.info(f"  机械臂关节数: {n_arm} (full_robot 单侧应为 6, 双手应为 12, gripper_only 应为 0)")
        logger.info(f"  夹爪关节数: {n_gripper}")
        if self.mode == "full_robot" and n_arm == 0:
            logger.error("  ✗ 严重错误: full_robot 模式下臂关节数为 0! URDF 转换失败!")
        elif self.mode == "full_robot" and n_arm == expected_arm:
            logger.info(f"  ✓ full_robot 臂关节正确加载 ({n_arm} 个 revolute)")
        elif self.mode == "gripper_only" and n_arm == 0:
            logger.info("  ✓ gripper_only 无臂关节 (正确)")

        # 2. qpos 范围验证
        if len(qpos_log) > 0:
            qpos_arr = np.array(qpos_log)
            logger.info(f"  qpos shape: {qpos_arr.shape}")
            if self.mode == "full_robot" and n_arm == expected_arm:
                arm_qpos = qpos_arr[:, self.robot_info["arm_joint_indices"]]
                logger.info(f"  臂关节 qpos 范围: min={arm_qpos.min(axis=0).round(3)}, "
                            f"max={arm_qpos.max(axis=0).round(3)}")
                # 验证臂关节有运动 (不是全固定)
                arm_range = arm_qpos.max(axis=0) - arm_qpos.min(axis=0)
                if arm_range.max() > 0.01:
                    logger.info(f"  ✓ 臂关节有运动 (最大范围={arm_range.max():.3f} rad)")
                else:
                    logger.warning(f"  ⚠ 臂关节几乎无运动 (最大范围={arm_range.max():.3f} rad)")

            # 夹爪 qpos 范围 (遍历所有侧, 验证夹爪手指开合运动可见)
            for s, (gidx1, gidx2) in gripper_indices.items():
                if gidx1 is not None and gidx2 is not None:
                    g1 = qpos_arr[:, gidx1]
                    g2 = qpos_arr[:, gidx2]
                    logger.info(f"  [{s}] 夹爪 qpos 范围: finger1=[{g1.min():.4f}, {g1.max():.4f}], "
                                f"finger2=[{g2.min():.4f}, {g2.max():.4f}]")
                    g1_range = float(g1.max() - g1.min())
                    g2_range = float(g2.max() - g2.min())
                    logger.info(f"  [{s}] 夹爪开合幅度: finger1={g1_range*1000:.2f}mm, finger2={g2_range*1000:.2f}mm")
                    if max(g1_range, g2_range) > 0.001:
                        logger.info(f"  ✓ [{s}] 夹爪手指有开合运动 (夹爪操作物体可见)")
                    else:
                        logger.warning(f"  ⚠ [{s}] 夹爪手指几乎无运动")

        # 3. 接触验证
        total_contacts = sum(c["contacts"] for c in contact_log)
        grasp_frames = sum(grasp_states)
        logger.info(f"  总接触帧数: {grasp_frames}/{len(grasp_states)}")
        logger.info(f"  总接触点数: {total_contacts}")
        if grasp_frames > 0:
            logger.info(f"  ✓ 检测到抓取 (接触≥2 的帧数={grasp_frames})")
        else:
            logger.warning("  ⚠ 未检测到稳定抓取 (接触<2)")

        # 4. 物体提升验证
        if len(obj_pos_log) > 0 and obj_initial_pos:
            logger.info("  物体位置变化 (参数级验证):")
            for actor_name in obj_initial_pos:
                init_pos = obj_initial_pos[actor_name]
                final_pos = np.array(obj_pos_log[-1][actor_name])
                lift = final_pos[2] - init_pos[2]  # Z 轴提升
                xy_drift = np.linalg.norm(final_pos[:2] - init_pos[:2])
                logger.info(f"    {actor_name}: lift={lift*100:.2f}cm, xy_drift={xy_drift*100:.2f}cm")
                if lift > 0.01:
                    logger.info(f"    ✓ {actor_name} 被提升 {lift*100:.2f}cm (抓取成功)")

        # 5. 综合抓取判定 (用户要求: 接触 + 跟随 + 提升 三项都满足)
        grasp_quality = self._evaluate_grasp_quality(
            contact_log, obj_pos_log, obj_initial_pos, gripper_pos_log
        )
        verify_log = {
            "mode": self.mode,
            "side": self.side,
            "n_arm_joints": n_arm,
            "n_gripper_joints": n_gripper,
            "total_frames": len(qpos_log),
            "grasp_frames": grasp_frames,
            "total_contacts": total_contacts,
            "obj_initial_pos": {k: v.tolist() for k, v in obj_initial_pos.items()},
            "obj_final_pos": obj_pos_log[-1] if obj_pos_log else {},
            "grasp_quality": grasp_quality,
        }

        # 6. 抓取控制器摘要 (adaptive / hybrid 模式)
        if self.grasp_mode in ("adaptive", "hybrid") and self.grasp_controllers is not None:
            mode_label = "混合" if self.grasp_mode == "hybrid" else "自适应"
            logger.info(f"  {mode_label}抓取控制器摘要:")
            grasp_summaries = {}
            for s, ctrl in self.grasp_controllers.items():
                summary = ctrl.summary()
                grasp_summaries[s] = summary
                logger.info(f"    [{s}] 抓取次数: {summary['grasp_count']}, "
                            f"最终相位: {summary['final_phase']}, 事件数: {len(summary['events'])}")
                if "max_force" in summary:
                    logger.info(f"    [{s}] 最大夹紧力: {summary['max_force']:.1f}N, "
                                f"平均: {summary['mean_force']:.1f}N")
                for ev in summary["events"]:
                    force_str = f", force={ev['force']}N" if "force" in ev else ""
                    logger.info(f"      F{ev['frame']}: {ev['phase']} (curl={ev['curl']}, "
                                f"obj={ev['obj']}@{ev['dist']}m{force_str})")
            verify_log["grasp_mode"] = self.grasp_mode
            verify_log["grasp_summaries"] = grasp_summaries

        # 轨迹跟踪误差统计 (物理输出 vs 真实 MANO 期望)
        # 用户: "夹爪运动要物理和真实输出的误差来判断准不准确, 而不是只有一个开合的判断"
        if track_log:
            valid = [t for t in track_log if t is not None]
            if valid:
                root_errs = [t['root_err_mm'] for t in valid]
                j1_errs = [t['j1_err_mm'] for t in valid]
                j2_errs = [t['j2_err_mm'] for t in valid]
                track_err = {
                    "n_frames": len(valid),
                    "root_err_mean_mm": float(np.mean(root_errs)),
                    "root_err_max_mm": float(np.max(root_errs)),
                    "finger1_err_mean_mm": float(np.mean(j1_errs)),
                    "finger1_err_max_mm": float(np.max(j1_errs)),
                    "finger2_err_mean_mm": float(np.mean(j2_errs)),
                    "finger2_err_max_mm": float(np.max(j2_errs)),
                }
                verify_log["track_error"] = track_err
                logger.info(f"  轨迹跟踪误差 (物理输出 vs MANO 期望, {len(valid)}帧):")
                logger.info(f"    根位置: mean={track_err['root_err_mean_mm']:.2f}mm, "
                            f"max={track_err['root_err_max_mm']:.2f}mm")
                logger.info(f"    手指1:  mean={track_err['finger1_err_mean_mm']:.2f}mm, "
                            f"max={track_err['finger1_err_max_mm']:.2f}mm")
                logger.info(f"    手指2:  mean={track_err['finger2_err_mean_mm']:.2f}mm, "
                            f"max={track_err['finger2_err_max_mm']:.2f}mm")
                # 误差越小越准确 (root>10mm 或 finger>5mm 说明跟踪有偏差)
                root_ok = track_err['root_err_mean_mm'] < 10.0
                finger_ok = max(track_err['finger1_err_mean_mm'], track_err['finger2_err_mean_mm']) < 5.0
                logger.info(f"    {'✓ 根跟踪准确' if root_ok else '⚠ 根跟踪有偏差'} "
                            f"(<10mm), {'✓ 手指跟踪准确' if finger_ok else '⚠ 手指跟踪有偏差'} (<5mm)")

        verify_path = str(self.output_dir / f"grasp_{self.mode}_{self.side}_verify.json")
        with open(verify_path, "w") as f:
            json.dump(verify_log, f, indent=2, default=str)
        logger.info(f"  验证日志: {verify_path}")

    def _evaluate_grasp_quality(self, contact_log, obj_pos_log, obj_initial_pos, gripper_pos_log):
        """综合抓取质量评估: 接触 + 跟随 + 提升 (三项都满足才算真正夹住)

        判定标准 (用户确认: 综合判定):
          1. 接触: 连续 >= 10 帧 n_contacts >= 2 (夹爪持续接触物体)
          2. 跟随: 物体相对最近夹爪位置变化 < 5cm 持续 10 帧 (物体跟着夹爪走)
          3. 提升: 物体 z 提升量 > 5cm (相对初始位置)

        Returns:
            dict: 每个物体的 {contact_pass, follow_pass, lift_pass, grasp_pass, details}
        """
        logger.info("=" * 60)
        logger.info("Step 7b: 综合抓取判定 (接触 + 跟随 + 提升)")
        logger.info("=" * 60)

        MIN_CONTACT_FRAMES = 10
        MIN_FOLLOW_FRAMES = 10
        FOLLOW_THRESHOLD = 0.05  # 5cm
        LIFT_THRESHOLD = 0.05     # 5cm

        n_frames = len(contact_log)
        quality = {}

        # 1. 接触: 找最长连续接触帧数
        contact_pass = False
        max_contact_streak = 0
        cur_streak = 0
        for c in contact_log:
            if c["contacts"] >= 2:
                cur_streak += 1
                max_contact_streak = max(max_contact_streak, cur_streak)
            else:
                cur_streak = 0
        if max_contact_streak >= MIN_CONTACT_FRAMES:
            contact_pass = True
        logger.info(f"  [接触] 最长连续帧数: {max_contact_streak}/{n_frames} "
                    f"(需 >= {MIN_CONTACT_FRAMES}) → {'✓ PASS' if contact_pass else '✗ FAIL'}")

        # 2 + 3. 跟随 + 提升 (每个物体独立判定)
        for actor_name in obj_initial_pos:
            init_pos = obj_initial_pos[actor_name]

            # 3. 提升
            lift_pass = False
            lift_max = 0.0
            for fo in obj_pos_log:
                if actor_name in fo:
                    cur_lift = float(np.array(fo[actor_name])[2] - init_pos[2])
                    lift_max = max(lift_max, cur_lift)
            if lift_max > LIFT_THRESHOLD:
                lift_pass = True
            logger.info(f"  [{actor_name} 提升] 最大提升: {lift_max*100:.2f}cm "
                        f"(需 > {LIFT_THRESHOLD*100:.0f}cm) → {'✓ PASS' if lift_pass else '✗ FAIL'}")

            # 2. 跟随: 物体相对最近夹爪的位置变化
            follow_pass = False
            max_follow_streak = 0
            cur_follow_streak = 0
            if gripper_pos_log and n_frames > 0:
                # 找最近夹爪侧 (用第一帧有夹爪位置的数据)
                nearest_side = None
                min_dist = float('inf')
                for s, gp in gripper_pos_log[0].items():
                    d = float(np.linalg.norm(np.array(gp) - init_pos))
                    if d < min_dist:
                        min_dist = d
                        nearest_side = s
                if nearest_side is not None:
                    # 计算每帧 (物体 - 夹爪) 的相对位置, 检查连续帧间变化
                    prev_rel = None
                    for fi in range(n_frames):
                        if (actor_name not in obj_pos_log[fi]
                                or nearest_side not in gripper_pos_log[fi]):
                            cur_follow_streak = 0
                            prev_rel = None
                            continue
                        obj_p = np.array(obj_pos_log[fi][actor_name])
                        grp_p = np.array(gripper_pos_log[fi][nearest_side])
                        rel_p = obj_p - grp_p
                        if prev_rel is not None:
                            delta = float(np.linalg.norm(rel_p - prev_rel))
                            if delta < FOLLOW_THRESHOLD:
                                cur_follow_streak += 1
                                max_follow_streak = max(max_follow_streak, cur_follow_streak)
                            else:
                                cur_follow_streak = 0
                        prev_rel = rel_p
            if max_follow_streak >= MIN_FOLLOW_FRAMES:
                follow_pass = True
            logger.info(f"  [{actor_name} 跟随] 最长连续帧数: {max_follow_streak}/{n_frames} "
                        f"(需 >= {MIN_FOLLOW_FRAMES}, 阈值 {FOLLOW_THRESHOLD*100:.0f}cm) → "
                        f"{'✓ PASS' if follow_pass else '✗ FAIL'}")

            grasp_pass = contact_pass and follow_pass and lift_pass
            quality[actor_name] = {
                "contact_pass": contact_pass,
                "follow_pass": follow_pass,
                "lift_pass": lift_pass,
                "grasp_pass": grasp_pass,
                "max_contact_streak": max_contact_streak,
                "max_follow_streak": max_follow_streak,
                "lift_max_cm": lift_max * 100,
            }
            tag = "✓ 真正夹住" if grasp_pass else "✗ 未真正夹住"
            logger.info(f"  [{actor_name} 综合] {tag} "
                        f"(接触={contact_pass}, 跟随={follow_pass}, 提升={lift_pass})")

        n_grasped = sum(1 for q in quality.values() if q["grasp_pass"])
        logger.info(f"  >>> 真正夹住物体数: {n_grasped}/{len(quality)}")
        return quality


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAPIEN 物理仿真: R1 机器人抓取 GLB 物体"
    )
    parser.add_argument("--mode", type=str, default="full_robot",
                        choices=["full_robot", "gripper_only"],
                        help="URDF 模式: full_robot(整个机器人) / gripper_only(纯夹爪)")
    parser.add_argument("--side", type=str, default="right", choices=["right", "left", "both"],
                        help="手侧: right(右手) / left(左手) / both(双手协同)")
    parser.add_argument("--hawor-dir", type=str, default="/home/an/data/hawor/7")
    parser.add_argument("--ras-dir", type=str, default="/home/an/data/ras/my_7mp4_result")
    parser.add_argument("--output-dir", type=str, default=None,
                            help="输出目录; 默认按输入文件自动命名: <hawor名>_<ras名>_grasp_<mode>_<side>/")
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--views", type=str, default="both",
                        choices=["cam", "god", "both"],
                        help="渲染视角: cam(第一人称相机视角) / god(上帝视角) / both(双视角, 默认)")
    parser.add_argument("--grasp-mode", type=str, default="hybrid",
                        choices=["adaptive", "mano", "hybrid"],
                        help="抓取模式: hybrid(MANO驱动+接触力控, 默认) / adaptive(旧版MANO意图) / mano(纯MANO重放)")
    args = parser.parse_args()

    # output_dir 传 None, 由 GraspSimulator 自动按输入文件命名
    sim = GraspSimulator(
        hawor_dir=args.hawor_dir,
        ras_dir=args.ras_dir,
        mode=args.mode,
        side=args.side,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        views=args.views,
        grasp_mode=args.grasp_mode,
    )
    sim.run()


if __name__ == "__main__":
    main()
