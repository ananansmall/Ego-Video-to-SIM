"""共享物理参数与工具函数

参数来源:
  - SAPIEN PD驱动: 与 GalaxeaManipSim / 04_physics_simulation.py 一致 (stiffness=1000, damping=200)
  - PyBullet PD增益: 与 pybullet_pipeline.py 一致 (Kp=1000, Kd=200)
  - 坐标变换: 与 02_render_scene.py 一致 (RXWORLD_TO_SAPIEN)
  - 夹爪几何: 与 04_physics_simulation.py / hand_track 一致
"""

from pathlib import Path
import numpy as np

# ============ 路径常量 ============
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # dex-retargeting/
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
R1_ASSETS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1"
R1_MESH_DIR = R1_ASSETS / "meshes"
R1_URDF_DIR = R1_ASSETS / "configs" / "urdfs"

# 三形式 URDF 路径
FULL_ROBOT_URDF = R1_URDF_DIR / "r1_v2_1_0.urdf"
FLOATING_ARM_RIGHT_URDF = R1_URDF_DIR / "r1_v2_1_0_floating_right.urdf"
FLOATING_ARM_LEFT_URDF = R1_URDF_DIR / "r1_v2_1_0_floating_left.urdf"

# RelaxedIK 配置
R1_RIGHT_SETTINGS = R1_ASSETS / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = R1_ASSETS / "configs" / "settings_left.yaml"

# ============ 坐标变换 (与 02_render_scene.py 一致) ============
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# ============ 机器人参数 (与 02/04 一致) ============
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985]
ARM_MAX_REACH = 0.713

# R1 右臂关节限位 (rad) — 用于 IK 输出裁剪, 防止发散
# joint1: [-2.8798, 2.8798], joint2: [0, 3.2289], joint3: [-3.3161, 0],
# joint4: [-2.8798, 2.8798], joint5: [-1.6581, 1.6581], joint6: [-2.8798, 2.8798]
RIGHT_ARM_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798],
    [0.0, 3.2289],
    [-3.3161, 0.0],
    [-2.8798, 2.8798],
    [-1.6581, 1.6581],
    [-2.8798, 2.8798],
], dtype=np.float64)
COMFORTABLE_REACH = 0.35
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
BASE_TRACKING_RANGE = 0.04
BASE_TRACKING_ALPHA = 0.15
SAFETY_DISTANCE = 0.05
WARMUP_FRAMES = 30

# ============ 物理参数 (与 GalaxeaManipSim / 04 一致) ============
# PD 驱动: 高刚度(100000)配合set_qpos会震荡, 纯PD驱动用合理刚度
JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 200.0
GRIPPER_STIFFNESS = 1000.0
GRIPPER_DAMPING = 200.0
PHYSICS_TIMESTEP = 1 / 240.0
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))  # =8
OBJECT_DENSITY = 1000.0
GROUND_HEIGHT = 0  # 对齐 GalaxeaManipSim: scene.add_ground(0)
GRAVITY = [0, 0, -9.81]

# PyBullet PD 增益 (匹配 SAPIEN)
PD_KP_ARM = 1000.0
PD_KD_ARM = 200.0
PD_KP_GRIPPER = 1000.0
PD_KD_GRIPPER = 200.0

# ============ 夹爪几何常数 (与 04/hand_track 一致) ============
_FINGER1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])
_FINGER2_ORIGIN = np.array([0.03689, 0.013453, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])
FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906
GRIPPER_INIT_OPEN = 0.04
GRIPPER_MAX_OPEN = 0.05
GRIPPER_FRICTION = 1.0  # 高摩擦材质实现摩擦力抓取

# ============ 平滑参数 ============
LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
EMA_POS_ALPHA = 0.6
EMA_ORI_ALPHA = 0.6
SMOOTH_MAX_VELOCITY = 1.5
SMOOTH_MAX_ACCELERATION = 4.0
SMOOTH_MAX_JERK = 20.0
SMOOTH_LP_ALPHA = 0.25

# ============ IK 参数 ============
IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 20

# ============ 抓取检测 ============
GRASP_CONTACT_THRESHOLD = 2  # 至少2个接触点
GRASP_STABLE_FRAMES = 5      # 持续接触N帧判定稳定抓取
GRASP_OBJECT_VEL_THRESHOLD = 0.01  # 物体相对夹爪速度阈值

# ============ GLB 物体分类 ============
# 所有 GLB 物体设为 kinematic (静态), 避免物理不稳定导致物体飞走.
# 注: 之前小物体设为 dynamic, 但凸包碰撞不稳定, 物体飞到 x=-6.66, y=14.81.
# 设为 kinematic 后物体固定, 机器人仍可靠近做抓取姿态.
KINEMATIC_VOLUME_THRESHOLD = -1  # -1 表示所有物体都视为大型场景结构 (kinematic)
DYNAMIC_MASS_MAX = 5.0  # kg


def is_large_scene_object(bbox_min, bbox_max):
    """判断GLB物体是否为大型场景结构(桌面/地板), 应设为kinematic

    Args:
        bbox_min, bbox_max: 包围盒角点 (3,)

    Returns:
        bool: True表示大型场景结构
    """
    size = np.array(bbox_max) - np.array(bbox_min)
    volume = float(np.prod(size))
    return volume > KINEMATIC_VOLUME_THRESHOLD


def compute_object_mass(bbox_min, bbox_max, density=OBJECT_DENSITY):
    """根据包围盒体积和质量密度计算物体质量

    Args:
        bbox_min, bbox_max: 包围盒角点 (3,)
        density: 密度 kg/m^3

    Returns:
        float: 质量 kg
    """
    size = np.array(bbox_max) - np.array(bbox_min)
    volume = float(np.prod(size))
    return min(volume * density, DYNAMIC_MASS_MAX)
