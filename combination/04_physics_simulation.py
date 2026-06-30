"""
04_physics_simulation.py — 物理仿真驱动：真实抓取与交互

与 02_render_scene.py 的区别:
  - 02: 运动学驱动 (set_qpos 直接设置关节角，无物理交互)
  - 04: 动力学驱动 (纯PD驱动 + 重力补偿 + decimation，与GalaxeaManipSim一致)

核心改动:
  1. 机器人关节使用纯 PD 驱动 (set_drive_target)，不使用 set_qpos
     - GalaxeaManipSim 从不在 step() 中调用 set_qpos，纯PD驱动避免震荡
     - PD参数: stiffness=1000, damping=200 (与GalaxeaManipSim一致)
  2. 每控制步执行 decimation 次物理子步 (compute_passive_force + set_qf + scene.step)
     - PD控制器需要多次物理步才能收敛到目标位置
  3. GLB 物体自动分类: 大型扁平几何体(桌面/地板)→kinematic, 小物体→dynamic
  4. 夹爪手指设置高摩擦材质 (friction=1.0)，实现摩擦力抓取
  5. 浮动底座：小范围跟踪手腕，减少关节限位问题
  6. 物理地面：防止物体无限下落
  7. 接触力检测：分析夹爪与物体之间的接触状态

用法:
  python 04_physics_simulation.py \\
      --hawor_dir ./output/hawor_reconstruction \\
      --ras_dir ./output/ras_output \\
      --mode physics_tracking \\
      --crf 18
"""

import os
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
_intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
# 如果用户已设置 VK_ICD_FILENAMES, 不覆盖 (允许命令行/环境变量优先)
if 'VK_ICD_FILENAMES' not in os.environ:
    if os.path.exists(_nvidia_icd):
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            if r.returncode == 0:
                os.environ['VK_ICD_FILENAMES'] = _nvidia_icd
            else:
                os.environ['VK_ICD_FILENAMES'] = _intel_icd
        except Exception:
            os.environ['VK_ICD_FILENAMES'] = _intel_icd
    else:
        os.environ['VK_ICD_FILENAMES'] = _intel_icd

import sys
import gc
import re
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
import sapien
import sapien.render
import torch
import joblib
from pytransform3d import rotations as pr
from tqdm import trange

try:
    import trimesh
except ImportError:
    trimesh = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))

from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path, OPERATOR2MANO_RIGHT
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting import yourdfpy as urdf
from mano_layer import MANOLayer

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
FLOATING_RIGHT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_right.urdf"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"

R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

FLIP_Z_FOR_PHYSICS = False

RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]

# GLB 坐标系转换: Z-UP (RAS 导出常见) → Y-UP (SAPIEN 标准)
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
WARMUP_FRAMES = 30
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.70  # 基座高度: 提高到 0.70m, 让机械臂垂直抓取, 不靠近桌面
COMFORT_TARGET_IN_BASE = np.array([0.25, 0.0, -0.55])  # 舒适目标点: 更低更近, 机械臂垂直下垂
BASE_TRACKING_RANGE = 0.0  # 固定底座: 不跟踪手腕
BASE_TRACKING_ALPHA = 0.15
# 分段固定基座参数
BASE_CLUSTER_N = 3  # 将轨迹分为 N 个固定基座
BASE_CLUSTER_TRANSITION_FRAMES = 10  # 基座间过渡帧数
SAFETY_DISTANCE = 0.05
LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
EMA_POS_ALPHA = 0.6
EMA_ORI_ALPHA = 0.6
SMOOTH_MAX_VELOCITY = 1.5
SMOOTH_MAX_ACCELERATION = 4.0
SMOOTH_MAX_JERK = 20.0
SMOOTH_LP_ALPHA = 0.25
SMOOTH_MAX_ITERATIONS = 10
SMOOTH_CONVERGENCE_EPS = 1e-5
TWO_PASS_SMOOTH = False  # 由smooth参数控制，smooth==2时启用
IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 20

R_GRIPPER_ALIGN = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=np.float64)


def _detect_glb_up_axis(all_vertices):
    """检测 GLB 坐标系是 Z-UP 还是 Y-UP (复制自 02_render_scene.py).

    RAS 导出的 GLB 可能是 Y-UP 或 Z-UP.
    检测启发式: Z-UP 场景中地板在 z=0, 物体在 z>0;
                Y-UP 场景中地板在 y=0, 物体在 y>0.
    """
    FLOOR_THRESHOLD = 0.1
    min_z = all_vertices[:, 2].min()
    min_y = all_vertices[:, 1].min()
    z_is_floor = abs(min_z) < FLOOR_THRESHOLD
    y_is_floor = abs(min_y) < FLOOR_THRESHOLD
    if z_is_floor and not y_is_floor:
        return "z-up"
    if y_is_floor and not z_is_floor:
        return "y-up"
    if z_is_floor and y_is_floor:
        z_at_floor = (abs(all_vertices[:, 2]) < FLOOR_THRESHOLD).sum()
        y_at_floor = (abs(all_vertices[:, 1]) < FLOOR_THRESHOLD).sum()
        return "z-up" if z_at_floor > y_at_floor else "y-up"
    return "y-up"

CAM_WIDTH = 1920
CAM_HEIGHT = 1080
HAWOR_FOCAL_DEFAULT = 600.0

MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

JOINT_TO_FINGER = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
FINGER_GROUP_COLORS = [
    np.array([0.9, 0.9, 0.9, 1.0]),
    np.array([1.0, 0.2, 0.2, 1.0]),
    np.array([0.2, 0.9, 0.2, 1.0]),
    np.array([0.3, 0.5, 1.0, 1.0]),
    np.array([1.0, 0.9, 0.2, 1.0]),
    np.array([1.0, 0.5, 0.0, 1.0]),
]

# PD 驱动参数: 与 GalaxeaManipSim 一致 (stiffness=1000, damping=200)
# 高刚度(100000)配合set_qpos会导致PD力与位置约束冲突，产生震荡
# 纯PD驱动模式下，合理的刚度让关节平滑跟踪目标，允许柔顺性
JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 200.0
GRIPPER_STIFFNESS = 1000.0
GRIPPER_DAMPING = 200.0
PHYSICS_TIMESTEP = 1 / 240.0
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))
OBJECT_DENSITY = 1000.0
GROUND_HEIGHT = -0.5

# ── 单夹爪模式: 夹爪几何常数 (从 URDF 提取, 与 hand_track/render_gripper_only.py 一致) ──
_FINGER1_ORIGIN = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS = np.array([0, -1, 0])  # prismatic axis
_FINGER2_ORIGIN = np.array([0.03689, 0.013453, 0.00012067])
_FINGER2_AXIS = np.array([0, 1, 0])  # prismatic axis
_FINGER_BASE_DIST = abs(_FINGER1_ORIGIN[1] - _FINGER2_ORIGIN[1])  # 0.026906
GRIPPER_INIT_OPEN = 0.04

# 夹爪 URDF 模板 (只有 gripper 部分, 无机械臂, 与 hand_track/render_gripper_only.py 一致)
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
    """生成只包含夹爪的 URDF 文件 (无机械臂, 与 hand_track/render_gripper_only.py 一致)

    结构: gripper_base_link (固定根) → gripper_link → finger_link1/2 (prismatic)
    """
    xml = _GRIPPER_ONLY_URDF_TEMPLATE.format(
        prefix=prefix,
        mesh_dir=str(R1_MESH_DIR),
    )
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


def make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
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


def reencode_with_ffmpeg(input_path, output_path, crf=18, fps=30, logger=None):
    """使用 ffmpeg 将视频重编码为 H.264 格式

    Args:
        input_path: 输入视频路径 (mp4v 编码)
        output_path: 输出视频路径 (H.264 编码)
        crf: 恒定质量因子 (0=无损, 18=高质量, 23=默认, 28=低质量)
        fps: 输出帧率
        logger: 日志记录器

    Returns:
        bool: 重编码是否成功 (成功时删除原始文件)
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        if logger:
            logger.warning("  imageio-ffmpeg 未安装，跳过 H.264 重编码")
        return False
    if not os.path.exists(input_path):
        return False

    # 如果输入和输出路径相同, 使用临时文件避免冲突
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        tmp_path = str(output_path) + ".tmp.mp4"
    else:
        tmp_path = str(output_path)

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-movflags", "+faststart",
        tmp_path,
    ]
    if logger:
        logger.info(f"  ffmpeg 重编码: CRF={crf}, {fps}fps, libx264")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            # 如果用了临时文件, 替换原文件
            if tmp_path != str(output_path):
                os.replace(tmp_path, str(output_path))
            # 删除原始输入文件 (如果与输出不同)
            if os.path.abspath(input_path) != os.path.abspath(output_path):
                os.remove(input_path)
            if logger:
                sz = os.path.getsize(output_path)
                logger.info(f"  ✓ 重编码完成: {output_path} ({sz / 1024 / 1024:.1f}MB)")
            return True
    if logger:
        # 显示 stderr 最后 300 字符 (实际错误信息在末尾, 不是开头的 build info)
        err_tail = result.stderr[-300:] if result.stderr else "无错误输出"
        logger.warning(f"  ffmpeg 重编码失败 (returncode={result.returncode}): {err_tail}")
    return False


def axis_angle_to_matrix(aa):
    """将轴角表示转换为3x3旋转矩阵

    Args:
        aa: 轴角向量, shape=(3,)

    Returns:
        np.ndarray: 3x3 旋转矩阵
    """
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    return pr.matrix_from_axis_angle(np.array([axis[0], axis[1], axis[2], angle]))


def _find_reconstruction_file(hawor_path):
    """在 HaWoR 目录中查找重建结果 npz 文件

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        Path: 找到的 npz 文件路径，或 None
    """
    rec_dir = hawor_path / "reconstruction"
    if not rec_dir.exists():
        npz_files = list(hawor_path.glob("*.npz"))
        if npz_files:
            return npz_files[0]
        return None
    for f in rec_dir.glob("hawor_results_*.npz"):
        return f
    return None


def _detect_hand_idx(hawor_path):
    """自动检测 HaWoR 数据中活跃的手 (改进: 返回 handedness 字符串, 支持双手)

    通过检查 cam_space/ 目录下的子目录来判断:
    - 只有 0/ → 左手活跃
    - 只有 1/ → 右手活跃
    - 两者都有 → 双手 (hand_idx 默认 0, 调用方按需选择)

    回退机制: cam_space 不存在时, 通过 reconstruction npz 中 pred_valid 判断.

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        tuple: (hand_idx, handedness_str)
            hand_idx: int (0=左手, 1=右手); 无法检测时返回 (0, "unknown")
            handedness_str: "left" / "right" / "both" / "unknown"
    """
    cam_dir = Path(hawor_path) / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if 0 in detected and 1 in detected:
            return 0, "both"
        if 1 in detected:
            return 1, "right"
        if 0 in detected:
            return 0, "left"

    # 回退: 通过 npz pred_valid 检测
    rec_file = _find_reconstruction_file(Path(hawor_path))
    if rec_file is not None:
        try:
            rec = np.load(str(rec_file), allow_pickle=True)
            if 'pred_valid' in rec:
                pred_valid = rec['pred_valid']
                if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                    left_active = bool(pred_valid[0].any())
                    right_active = bool(pred_valid[1].any())
                    if left_active and right_active:
                        return 0, "both"
                    if right_active:
                        return 1, "right"
                    if left_active:
                        return 0, "left"
        except Exception:
            pass
    return 0, "unknown"


def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据

    支持两种数据格式:
    1. reconstruction/hawor_results_*.npz (推荐, 含相机轨迹和焦距)
    2. world_space_res.pth (旧格式, 无相机轨迹)

    Args:
        hawor_dir: HaWoR 输出目录路径
        hand_idx: 手部索引 (0=左手, 1=右手)

    Returns:
        dict: 包含 pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, img_focal
    """
    hawor_path = Path(hawor_dir)
    rec_file = _find_reconstruction_file(hawor_path)
    ws_file = hawor_path / "world_space_res.pth"

    img_focal = None

    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        pred_trans = rec['pred_trans']
        pred_rot = rec['pred_rot']
        pred_hand_pose = rec['pred_hand_pose']
        pred_betas = rec['pred_betas']
        pred_valid = rec['pred_valid']
        if 'img_focal' in rec:
            img_focal = float(rec['img_focal'])
    elif ws_file.exists():
        ws = joblib.load(str(ws_file))
        pred_trans = ws[0].numpy() if hasattr(ws[0], 'numpy') else np.array(ws[0])
        pred_rot = ws[1].numpy() if hasattr(ws[1], 'numpy') else np.array(ws[1])
        pred_hand_pose = ws[2].numpy() if hasattr(ws[2], 'numpy') else np.array(ws[2])
        pred_betas = ws[3].numpy() if hasattr(ws[3], 'numpy') else np.array(ws[3])
        pred_valid = ws[4] if isinstance(ws[4], np.ndarray) else np.array(ws[4])
    else:
        raise FileNotFoundError(
            f"未找到 hawor 数据文件!\n"
            f"  查找路径: {hawor_path}\n"
            f"  期望: {hawor_path}/reconstruction/hawor_results_*.npz\n"
            f"     或: {hawor_path}/world_space_res.pth\n"
            f"  请确认 --hawor-dir 指向正确的 HaWoR 输出目录\n"
            f"  示例: --hawor-dir /home/an/data/hawor/7")

    est_focal_file = hawor_path / "est_focal.txt"
    if img_focal is None and est_focal_file.exists():
        try:
            img_focal = float(est_focal_file.read_text().strip())
        except Exception:
            pass

    return {
        "pred_trans": pred_trans[hand_idx],
        "pred_rot": pred_rot[hand_idx],
        "pred_hand_pose": pred_hand_pose[hand_idx],
        "pred_betas": pred_betas[hand_idx],
        "pred_valid": pred_valid[hand_idx],
        "img_focal": img_focal,
    }


def load_hawor_c2w(hawor_dir):
    """加载 HaWoR 相机轨迹 (camera-to-world 变换)

    Args:
        hawor_dir: HaWoR 输出目录路径

    Returns:
        tuple: (R_c2w, t_c2w) 或 (None, None)
    """
    rec_file = _find_reconstruction_file(Path(hawor_dir))
    if rec_file is None:
        return None, None
    rec = np.load(str(rec_file), allow_pickle=True)
    if 'R_c2w' not in rec or 't_c2w' not in rec:
        return None, None
    return rec['R_c2w'], rec['t_c2w']


def compute_mano_joints(mano_layer, rot, hand_pose, trans):
    """通过 MANO 正运动学计算手部顶点和关节

    Args:
        mano_layer: MANOLayer 实例
        rot: (3,) 手腕轴角旋转
        hand_pose: (45,) 手指 PCA 参数
        trans: (3,) 手腕平移

    Returns:
        tuple: (vertices, joints)
    """
    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = mano_layer(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]


def compute_smooth_shading_normal(vertices, faces):
    """计算平滑着色法线 (顶点法线)

    Args:
        vertices: (V, 3) 顶点坐标
        faces: (F, 3) 面索引

    Returns:
        np.ndarray: (V, 3) 归一化的顶点法线
    """
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)
    normal = np.zeros_like(vertices)
    normal[faces[:, 0]] += face_normal
    normal[faces[:, 1]] += face_normal
    normal[faces[:, 2]] += face_normal
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return normal


def make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
    """计算 look-at 相机姿态的四元数

    Args:
        eye: 相机位置, shape=(3,)
        target: 目标点, shape=(3,)
        up: 上方向, 默认 [0,0,1]

    Returns:
        np.ndarray: 相机朝向四元数 (w,x,y,z)
    """
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


def hawor_cam_to_sapien_pose(R_c2w, t_c2w):
    """将 HaWoR 相机位姿转换为 SAPIEN 相机位姿

    变换链:
    1. 将 HaWoR 相机位置/旋转转换到 SAPIEN 坐标系
    2. 将 OpenGL 相机约定 (Z=后方) 转换为 SAPIEN 相机约定 (Z=上方)

    Args:
        R_c2w: (3, 3) HaWoR 相机旋转矩阵
        t_c2w: (3,) HaWoR 相机平移向量

    Returns:
        tuple: (cam_pos, cam_quat)
    """
    # 对齐 02_render_scene.py: 相机数据是 OpenGL 帧 (R_x 已应用), 用 R_AXIS 变换
    # 手部/GLB 用 RXWORLD_TO_SAPIEN @ SLAM = R_AXIS @ R_x @ SLAM = R_AXIS @ OpenGL
    # 相机用 R_AXIS @ OpenGL → 两者同帧
    cam_pos_sapien = R_AXIS @ t_c2w
    cam_R_sapien = R_AXIS @ R_c2w

    if FLIP_Z_FOR_PHYSICS:
        cam_pos_sapien[2] = -cam_pos_sapien[2]
        forward = -cam_R_sapien[:, 2].copy()
        up = cam_R_sapien[:, 1].copy()
        forward[2] = -forward[2]
        up[2] = -up[2]
        left = np.cross(up, forward)
        left = left / max(np.linalg.norm(left), 1e-8)
        up = np.cross(forward, left)
    else:
        # 与 02_render_scene.py hawor_cam_to_sapien_pose 完全一致
        forward = cam_R_sapien[:, 2]
        left = -cam_R_sapien[:, 0]
        up = -cam_R_sapien[:, 1]

    sapien_cam_R = np.eye(3)
    sapien_cam_R[:, 0] = forward
    sapien_cam_R[:, 1] = left
    sapien_cam_R[:, 2] = up

    if np.linalg.det(sapien_cam_R) < 0:
        U, _, VH = np.linalg.svd(sapien_cam_R)
        sapien_cam_R = U @ VH
    cam_quat = pr.quaternion_from_matrix(sapien_cam_R)
    return cam_pos_sapien, cam_quat


def _prepare_arm_urdf(src_urdf_path, arm_prefix="right"):
    """准备 R1 浮动臂 URDF: 替换 mesh 路径 + 修改夹爪关节类型

    1. 将 package://r1_v2_1_0/meshes/ 替换为绝对路径
    2. 将 gripper_finger_joint1/2 从 fixed 改为 prismatic

    Args:
        src_urdf_path: 原始 URDF 文件路径
        arm_prefix: 臂前缀 ("right" 或 "left")

    Returns:
        str: 修改后的临时 URDF 文件路径
    """
    xml = Path(src_urdf_path).read_text()
    xml = xml.replace('package://r1_v2_1_0/meshes/', str(R1_MESH_DIR) + '/')
    xml = re.sub(
        rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint1"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    xml = re.sub(
        rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint2"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    temp_dir = tempfile.mkdtemp(prefix='r1_physics_urdf-')
    temp_path = f'{temp_dir}/{Path(src_urdf_path).name}'
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def _compute_object_support_plane(glb_path, transform_params_path):
    """计算 GLB 物体的支撑平面 (桌面高度 + 物体范围 + 桌面颜色)

    分析所有几何体变换后的顶点:
    1. 用 Z 高度分箱, 找到最大水平面 (桌面表面)
    2. 从该水平面提取平均顶点颜色
    3. 用 dynamic (小) 物体的最低 Z 确定桌面高度

    Args:
        glb_path: GLB 文件路径
        transform_params_path: transform_params.npz 路径

    Returns:
        dict: {
            'min_z': dynamic 物体最低Z,
            'max_z': 所有物体最高Z,
            'support_z': 支撑面高度 (min_z - 小偏移),
            'center_xy': 物体质心XY,
            'extent_xy': 物体XY范围,
            'table_color': 桌面颜色RGBA (从GLB提取),
            'table_surface_z': 检测到的桌面表面Z,
        } 或 None (加载失败)
    """
    if trimesh is None:
        return None
    try:
        params = np.load(str(transform_params_path))
        s_inv = float(params['s_inv'])
        R_inv = params['R_inv']
        t_inv = params['t_inv']
        trimesh_scene = trimesh.load(str(glb_path))
        all_verts_sapien = []
        dynamic_verts_sapien = []
        # 存储每个顶点及其原始颜色 (用于桌面颜色提取)
        verts_with_color = []  # list of (vertices_sapien, vertex_colors)
        for geom_name, geom in trimesh_scene.geometry.items():
            if len(geom.vertices) == 0:
                continue
            vertices = geom.vertices.copy()
            vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
            vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
            if FLIP_Z_FOR_PHYSICS:
                vertices_sapien[:, 2] = -vertices_sapien[:, 2]
            all_verts_sapien.append(vertices_sapien)

            # 提取顶点颜色
            vc = None
            if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
                vc = geom.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            verts_with_color.append((vertices_sapien, vc))

            # 分类: 大型扁平几何体 (桌面/地板) vs 小物体
            bbox_size = vertices_sapien.max(axis=0) - vertices_sapien.min(axis=0)
            volume = abs(bbox_size[0] * bbox_size[1] * bbox_size[2])
            max_extent = max(bbox_size)
            flatness = bbox_size[2] / max(max(bbox_size[0], bbox_size[1]), 1e-6)
            is_static = (volume > 0.01 and flatness < 0.3) or max_extent > 0.8
            if not is_static:
                dynamic_verts_sapien.append(vertices_sapien)

        if not all_verts_sapien:
            return None

        # 桌面高度: 优先用 dynamic (小) 物体的最低 Z
        if dynamic_verts_sapien:
            dyn_verts = np.vstack(dynamic_verts_sapien)
            min_z = float(dyn_verts[:, 2].min())
        else:
            all_verts = np.vstack(all_verts_sapien)
            min_z = float(all_verts[:, 2].min())

        all_verts = np.vstack(all_verts_sapien)
        max_z = float(all_verts[:, 2].max())
        center_xy = all_verts[:, :2].mean(axis=0)
        extent_xy = all_verts[:, :2].max(axis=0) - all_verts[:, :2].min(axis=0)

        # === Ray casting: Z 高度分箱找最大水平面 ===
        # 将 Z 坐标按 1mm 分箱, 找顶点最多且 XY 范围最大的 Z 层
        z_bins = np.round(all_verts[:, 2] / 0.001) * 0.001
        unique_z, inv, counts = np.unique(z_bins, return_inverse=True, return_counts=True)

        table_surface_z = None
        table_color = np.array([0.55, 0.45, 0.35, 1.0])  # 默认木色
        if len(unique_z) > 0:
            z_scores = []
            for zi, z_val in enumerate(unique_z):
                mask = inv == zi
                verts_at_z = all_verts[mask]
                if len(verts_at_z) < 10:
                    continue
                xy_extent = verts_at_z[:, :2].max(axis=0) - verts_at_z[:, :2].min(axis=0)
                area = xy_extent[0] * xy_extent[1]
                # 分数 = 顶点数 * 面积 (大的水平面得分高)
                z_scores.append((z_val, counts[zi], area, xy_extent))

            if z_scores:
                # 找面积最大的水平面 (桌面)
                z_scores.sort(key=lambda x: x[2], reverse=True)
                table_surface_z = z_scores[0][0]
                table_surface_area = z_scores[0][2]
                table_surface_extent = z_scores[0][3]

                # 提取桌面颜色: 在 table_surface_z 附近 ±1mm 的顶点颜色取平均
                surface_mask = np.abs(all_verts[:, 2] - table_surface_z) < 0.002
                surface_colors = []
                offset = 0
                for v_sapien, vc in verts_with_color:
                    if vc is not None:
                        n_verts = len(v_sapien)
                        local_mask = surface_mask[offset:offset + n_verts]
                        if local_mask.any():
                            surface_colors.append(vc[local_mask])
                    offset += len(v_sapien)
                if surface_colors:
                    all_surface_colors = np.vstack(surface_colors)
                    avg_color = all_surface_colors.mean(axis=0)
                    table_color = np.array([avg_color[0], avg_color[1], avg_color[2], 1.0])

        return {
            'min_z': min_z,
            'max_z': max_z,
            'support_z': min_z - 0.002,
            'center_xy': center_xy,
            'extent_xy': extent_xy,
            'table_color': table_color,
            'table_surface_z': table_surface_z,
        }
    except Exception:
        return None


def setup_physics_scene(ground_height=GROUND_HEIGHT):
    """创建物理仿真场景, 配置光照和物理地面

    与 setup_scene() (02_render_scene.py) 的区别:
    - 设置物理时间步长 (1/240s)
    - 添加物理地面 (高度由 ground_height 参数决定, 用于兜底防掉落)
    - 地面不可见 (render_half_size=0), 仅提供碰撞支撑

    Args:
        ground_height: 物理地面高度 (Z坐标), 默认 GROUND_HEIGHT=-0.5m

    Returns:
        sapien.Scene: 配置好的物理场景实例
    """
    from sapien.asset import create_dome_envmap
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_ray_tracing_samples_per_pixel(64)
    scene = sapien.Scene()
    scene.set_timestep(PHYSICS_TIMESTEP)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35]))
    scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
    scene.add_directional_light([-1, -0.5, -1], [1.2, 1.2, 1.2], shadow=False)
    scene.add_directional_light([0, 1, -0.5], [0.8, 0.8, 0.8], shadow=False)
    scene.set_ambient_light([0.5, 0.5, 0.5])

    ground_mat = sapien.render.RenderMaterial()
    ground_mat.set_base_color([0.35, 0.35, 0.35, 1])
    ground_mat.set_roughness(0.9)
    ground_mat.set_metallic(0.0)
    ground_mat.set_specular(0.04)
    scene.add_ground(ground_height, render_half_size=[0, 0])

    return scene


def load_glb_with_physics(glb_path, transform_params_path, scene, logger=None, fast_collision=False):
    """加载 GLB 场景并创建带碰撞体的物理物体

    与 load_glb_transformed() (02_render_scene.py) 的区别:
    - 物体为 dynamic (可被推动/抓取)
    - 添加碰撞体: CoACD 凸分解 (精确) 或凸包 (快速)
    - 设置物理材质: friction=0.5, restitution=0.3
    - 设置密度: OBJECT_DENSITY=1000 kg/m³
    - 支持碰撞体缓存 (避免重复计算 CoACD)

    Args:
        glb_path: GLB 文件路径
        transform_params_path: transform_params.npz 路径
        scene: SAPIEN 场景实例
        logger: 日志记录器
        fast_collision: 是否使用快速凸包碰撞体 (调试用)

    Returns:
        list: SAPIEN actor 列表 (带碰撞体的动态物体)
    """
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    saved_glb_up_axis = str(params.get('glb_up_axis', 'y-up')) if 'glb_up_axis' in params else None

    if trimesh is None:
        if logger:
            logger.error("  ✗ trimesh 未安装, 无法加载 GLB")
        return []

    if logger:
        size_mb = Path(glb_path).stat().st_size / 1024 / 1024
        logger.info(f"  GLB 文件: {glb_path} ({size_mb:.1f} MB)")

    glb_cache_dir = Path(glb_path).parent / "physics_cache"
    glb_cache_dir.mkdir(exist_ok=True)
    glb_hash = f"{Path(glb_path).stem}_{Path(transform_params_path).stem}"
    if fast_collision:
        glb_hash += "_fast"

    trimesh_scene = trimesh.load(str(glb_path))
    n_geom = len(trimesh_scene.geometry)
    if logger:
        logger.info(f"  GLB 内容: {n_geom} 个几何体 (fast_collision={fast_collision})")

    # 检测 GLB 坐标系 (Z-UP vs Y-UP), 与 02_render_scene.py 一致
    all_verts_list = []
    for _, geom in trimesh_scene.geometry.items():
        if len(geom.vertices) > 0:
            all_verts_list.append(geom.vertices)
    if saved_glb_up_axis is not None:
        glb_up_axis = saved_glb_up_axis
    elif all_verts_list:
        glb_up_axis = _detect_glb_up_axis(np.vstack(all_verts_list))
    else:
        glb_up_axis = "y-up"
    need_zup_to_yup = glb_up_axis == "z-up"
    if logger:
        logger.info(f"  GLB 坐标系: {glb_up_axis}{' (将转换到 Y-UP)' if need_zup_to_yup else ''}")

    obj_actors = []
    temp_files = []

    for geom_idx, (geom_name, geom) in enumerate(trimesh_scene.geometry.items()):
        vertices = geom.vertices.copy()
        faces = geom.faces.copy()
        if len(vertices) == 0 or len(faces) == 0:
            continue

        if need_zup_to_yup:
            vertices = (ZUP_TO_YUP @ vertices.T).T

        vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
        vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
        if FLIP_Z_FOR_PHYSICS:
            vertices_sapien[:, 2] = -vertices_sapien[:, 2]

        avg_color = None
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            vertex_colors = geom.visual.vertex_colors
            if len(vertex_colors) > 0:
                avg_rgb = vertex_colors[:, :3].mean(axis=0)
                avg_color = [avg_rgb[0]/255.0, avg_rgb[1]/255.0, avg_rgb[2]/255.0, 1.0]

        # 计算质心: dynamic物体需要质心在几何体中心，否则重力会导致旋转/漂移
        centroid = vertices_sapien.mean(axis=0)
        # 居中顶点: 形状在actor局部坐标系中以原点为中心
        vertices_centered = vertices_sapien - centroid

        # 保存居中后的PLY (视觉+碰撞体都基于居中顶点)
        temp_ply = f'/tmp/glb_physics_{os.getpid()}_{geom_name.replace(" ", "_")}.ply'
        geom_centered = trimesh.Trimesh(
            vertices=vertices_centered,
            faces=faces,
            visual=geom.visual
        )
        geom_centered.export(temp_ply)
        temp_files.append(temp_ply)

        builder = scene.create_actor_builder()
        builder.set_physx_body_type("kinematic")

        phys_material = scene.create_physical_material(
            static_friction=0.8,
            dynamic_friction=0.8,
            restitution=0.0,
        )

        if avg_color is not None:
            material = sapien.render.RenderMaterial(
                base_color=avg_color,
                metallic=0.0,
                roughness=0.7,
                specular=0.3
            )
            builder.add_visual_from_file(filename=temp_ply, material=material)
        else:
            builder.add_visual_from_file(filename=temp_ply)
        if logger:
            logger.info(f"    ✓ {geom_name}: 视觉体已添加")

        cache_file = glb_cache_dir / f"{glb_hash}_{geom_name.replace(' ', '_')}.npz"

        collision_ok = False
        if fast_collision:
            try:
                builder.add_convex_collision_from_file(filename=temp_ply, material=phys_material)
                collision_ok = True
                if logger:
                    logger.info(f"    ✓ {geom_name}: 凸包碰撞体 (fast mode)")
            except Exception as e:
                if logger:
                    logger.warning(f"    ✗ {geom_name}: 凸包碰撞失败 ({e}), 尝试非凸")
                try:
                    builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                    collision_ok = True
                except Exception as e2:
                    if logger:
                        logger.warning(f"    ✗ {geom_name}: 碰撞体生成失败 ({e2})")
        elif cache_file.exists():
            try:
                cache_data = np.load(str(cache_file), allow_pickle=True)
                convex_parts = cache_data['convex_parts'].item()
                for part_verts, part_faces in convex_parts:
                    # 居中碰撞体顶点(与视觉体一致)
                    part_verts_centered = part_verts - centroid
                    part_ply = f'/tmp/glb_physics_part_{os.getpid()}_{geom_idx}.ply'
                    part_mesh = trimesh.Trimesh(vertices=part_verts_centered, faces=part_faces)
                    part_mesh.export(part_ply)
                    temp_files.append(part_ply)
                    builder.add_convex_collision_from_file(filename=part_ply, material=phys_material)
                collision_ok = True
                if logger:
                    logger.info(f"    ✓ {geom_name}: 缓存碰撞体 ({len(convex_parts)} 凸部件)")
            except Exception as e:
                if logger:
                    logger.warning(f"    ✗ {geom_name}: 缓存加载失败 ({e}), 重新计算 CoACD")
                try:
                    builder.add_multiple_convex_collisions_from_file(
                        filename=temp_ply, decomposition="coacd", material=phys_material,
                    )
                    collision_ok = True
                    if logger:
                        logger.info(f"    ✓ {geom_name}: CoACD 碰撞体已生成")
                except Exception as e2:
                    if logger:
                        logger.warning(f"    ✗ {geom_name}: CoACD 失败 ({e2}), 尝试非凸碰撞")
                    try:
                        builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                        collision_ok = True
                    except Exception as e3:
                        if logger:
                            logger.warning(f"    ✗ {geom_name}: 碰撞体生成失败 ({e3})")
        else:
            try:
                builder.add_multiple_convex_collisions_from_file(
                    filename=temp_ply,
                    decomposition="coacd",
                    material=phys_material,
                )
                collision_ok = True
                if logger:
                    logger.info(f"    ✓ {geom_name}: CoACD 碰撞体已生成")
            except Exception as e:
                if logger:
                    logger.warning(f"    ✗ {geom_name}: CoACD 失败 ({e}), 尝试非凸碰撞")
                try:
                    builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                    collision_ok = True
                except Exception as e2:
                    if logger:
                        logger.warning(f"    ✗ {geom_name}: 碰撞体生成失败 ({e2})")

        if not collision_ok:
            if logger:
                logger.warning(f"    ⚠ {geom_name}: 无碰撞体, 仅添加视觉体")

        # 判断几何体类型: 大型扁平几何体(桌面/地板)设为kinematic, 小物体设为dynamic
        # 启发式: 体积>0.01m³ 或 最长边>0.5m 且 扁平度(Z范围/XY范围<0.3) → 场景固定结构
        bbox_min = vertices_sapien.min(axis=0)
        bbox_max = vertices_sapien.max(axis=0)
        bbox_size = bbox_max - bbox_min
        volume = abs(bbox_size[0] * bbox_size[1] * bbox_size[2])
        max_extent = max(bbox_size)
        flatness = bbox_size[2] / max(max(bbox_size[0], bbox_size[1]), 1e-6)

        is_scene_structure = (volume > 0.01 and flatness < 0.3) or max_extent > 0.8

        if is_scene_structure:
            # 场景固定结构(桌面/地板/墙壁): kinematic, 不受重力影响
            builder.set_physx_body_type("kinematic")
            actor = builder.build(name=f"glb_{geom_name}")
            actor.set_pose(sapien.Pose(p=centroid.tolist(), q=[1, 0, 0, 0]))
            if logger:
                logger.info(f"    → {geom_name}: kinematic (场景结构, vol={volume:.4f}m³, flat={flatness:.2f})")
        else:
            # 可交互物体(杯子/书本等): dynamic, 可被推动/抓取
            builder.set_physx_body_type("dynamic")
            actor = builder.build(name=f"glb_{geom_name}")
            actor.set_pose(sapien.Pose(p=centroid.tolist(), q=[1, 0, 0, 0]))
            # 应用质量 (参考 GalaxeaManipSim: 质量下限 0.1kg, 防轻物被碰飞)
            obj_mass = max(volume * OBJECT_DENSITY, 0.1)
            for comp in actor.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                    comp.mass = obj_mass
                    break
            if logger:
                logger.info(f"    → {geom_name}: dynamic (可交互, vol={volume:.4f}m³, flat={flatness:.2f}, mass={obj_mass:.3f}kg)")

        obj_actors.append(actor)

        gc.collect()

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except OSError:
            pass

    if logger:
        logger.info(f"  ✓ GLB 物理物体: {len(obj_actors)} 个 (带碰撞体, density={OBJECT_DENSITY})")
    return obj_actors


class EmaTargetSmoother:
    def __init__(self, pos_alpha=0.3, ori_alpha=0.3):
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


class TrajectorySmoother:
    """轨迹平滑器: 基于物理约束的迭代限幅 (速度→加速度→jerk)

    与02_render_scene.py完全一致，使用双向Butterworth + 迭代限幅
    """

    SMOOTHNESS_THRESHOLDS = {
        "max_velocity": 3.0,
        "max_acceleration": 8.0,
        "max_jerk": 80.0,
        "si_improvement_min": 0.5,
    }

    def __init__(self, fps=30, max_velocity=SMOOTH_MAX_VELOCITY, max_acceleration=SMOOTH_MAX_ACCELERATION,
                 max_jerk=SMOOTH_MAX_JERK, lp_alpha=SMOOTH_LP_ALPHA, max_iterations=SMOOTH_MAX_ITERATIONS,
                 convergence_eps=SMOOTH_CONVERGENCE_EPS):
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
        n_frames = len(trajectory)
        s1_fwd = np.zeros_like(trajectory)
        s2_fwd = np.zeros_like(trajectory)
        s1_fwd[0] = trajectory[0]
        s2_fwd[0] = trajectory[0]
        for i in range(1, n_frames):
            s1_fwd[i] = s1_fwd[i - 1] + alpha * (trajectory[i] - s1_fwd[i - 1])
            s2_fwd[i] = s2_fwd[i - 1] + alpha * (s1_fwd[i] - s2_fwd[i - 1])
        s1_bwd = np.zeros_like(trajectory)
        s2_bwd = np.zeros_like(trajectory)
        s1_bwd[-1] = s2_fwd[-1]
        s2_bwd[-1] = s2_fwd[-1]
        for i in range(n_frames - 2, -1, -1):
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
            max_change = np.max(np.abs(trajectory - traj_before))
            if max_change < self.convergence_eps:
                break
        return trajectory

    def _compute_metrics(self, trajectory_smooth, trajectory_raw):
        dt = self.dt
        vel_raw = np.diff(trajectory_raw, axis=0) / dt
        vel_smooth = np.diff(trajectory_smooth, axis=0) / dt
        acc_smooth = np.diff(vel_smooth, axis=0) / dt
        jerk_smooth = np.diff(acc_smooth, axis=0) / dt
        return {
            "smooth_max_velocity": float(np.max(np.abs(vel_smooth))),
            "smooth_max_acceleration": float(np.max(np.abs(acc_smooth))) if len(acc_smooth) > 0 else 0.0,
            "smooth_max_jerk": float(np.max(np.abs(jerk_smooth))) if len(jerk_smooth) > 0 else 0.0,
        }


class OnlineTrajectorySmoother:
    """在线版轨迹平滑器: 单向Butterworth + 迭代限幅

    与离线的 TrajectorySmoother 不同, 此版本只做前向滤波, 可用于实时控制循环。
    维护一个短历史窗口 (lookback 帧), 对当前帧应用迭代限幅。

    限幅原理 (基于物理约束):
    - 最大关节速度: 1.5 rad/s → 30fps下每帧最大变化 0.05 rad ≈ 2.86°
    - 最大关节加速度: 4.0 rad/s² → 30fps下相邻帧速度差最大 0.133 rad/s
    - 最大关节jerk: 20.0 rad/s³ → 30fps下相邻帧加速度差最大 0.667 rad/s²
    """

    def __init__(self, max_velocity=SMOOTH_MAX_VELOCITY, max_acceleration=SMOOTH_MAX_ACCELERATION,
                 max_jerk=SMOOTH_MAX_JERK, dt=1.0/30.0, lookback=8):
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.lookback = lookback
        self.history = []

    def reset(self):
        self.history = []

    def smooth_step(self, qpos):
        """对单帧qpos应用迭代限幅

        Args:
            qpos: (J,) 当前帧IK输出关节角

        Returns:
            (J,) 平滑且限幅后的关节角
        """
        self.history.append(np.asarray(qpos, dtype=np.float64))
        if len(self.history) > self.lookback:
            self.history.pop(0)

        if len(self.history) < 4:
            return qpos.copy()

        traj = np.array(self.history)
        for _ in range(3):
            traj = self._clamp_velocity(traj)
            traj = self._clamp_acceleration(traj)
            traj = self._clamp_jerk(traj)

        return traj[-1]

    def _clamp_velocity(self, traj):
        max_delta = self.max_velocity * self.dt
        for i in range(1, len(traj)):
            delta = traj[i] - traj[i - 1]
            clamped = np.clip(delta, -max_delta, max_delta)
            traj[i] = traj[i - 1] + clamped
        return traj

    def _clamp_acceleration(self, traj):
        max_delta_v = self.max_acceleration * self.dt
        for i in range(2, len(traj)):
            v_prev = traj[i - 1] - traj[i - 2]
            v_curr = traj[i] - traj[i - 1]
            delta_v = v_curr - v_prev
            clamped_dv = np.clip(delta_v, -max_delta_v, max_delta_v)
            v_curr_clamped = v_prev + clamped_dv
            traj[i] = traj[i - 1] + v_curr_clamped
        return traj

    def _clamp_jerk(self, traj):
        max_delta_a = self.max_jerk * self.dt
        for i in range(3, len(traj)):
            v_im2 = traj[i - 2] - traj[i - 3]
            v_im1 = traj[i - 1] - traj[i - 2]
            v_i = traj[i] - traj[i - 1]
            a_prev = v_im1 - v_im2
            a_curr = v_i - v_im1
            delta_a = a_curr - a_prev
            clamped_da = np.clip(delta_a, -max_delta_a, max_delta_a)
            a_curr_clamped = a_prev + clamped_da
            v_i_clamped = v_im1 + a_curr_clamped
            traj[i] = traj[i - 1] + v_i_clamped
        return traj


class PhysicsSimulator:
    """物理仿真器: 使用 PD 驱动 + 碰撞检测实现真实抓取

    与 02_render_scene.py 的运动学驱动不同:
    - 02: set_qpos 直接设置关节角 → 无物理交互
    - 04: set_drive_target + set_qpos + compute_passive_force + scene.step → 真实物理抓取

    核心特性:
    - PD 驱动: 关节通过 stiffness/damping 控制目标位置
    - 重力补偿: compute_passive_force + set_qf 补偿重力和科里奥利力
    - 碰撞体: CoACD 凸分解 + 高摩擦夹爪
    - 接触力检测: 分析夹爪与物体之间的接触状态
    """

    def __init__(self, hawor_dir, ras_dir, transform_params_path,
                 output="physics_sim.mp4", fps=30, hand_idx=0,
                 logger=None, viewer=False, crf=18, fast_collision=False,
                 hide_hand=False, speed=1.0,
                 cam_width=CAM_WIDTH, cam_height=CAM_HEIGHT, smooth=1,
                 two_pass=False, support_table=True,
                 view="fpv", single_gripper=False, base_cluster=False, fixed_base=False):
        """初始化物理仿真器

        Args:
            hawor_dir: HaWoR 输出目录路径
            ras_dir: RAS 输出目录路径 (含 GLB 场景文件)
            transform_params_path: 01_align_scene.py 输出的变换参数路径
            output: 输出视频路径
            fps: 视频帧率
            hand_idx: 手部索引 (0=左手, 1=右手)
            logger: 日志记录器
            viewer: 是否使用交互式 Viewer 模式
            crf: H.264 编码质量因子
            fast_collision: 是否使用快速凸包碰撞体
            hide_hand: 是否隐藏手部 mesh 和骨架
            speed: 播放速度倍率 (1.0=原始速度, 保证视频帧数=HaWoR帧数)
            cam_width: 渲染宽度 (像素)
            cam_height: 渲染高度 (像素)
            smooth: 平滑模式 (0=不平滑, 1=在线EMA, 2=后处理双向滤波)
            two_pass: 两趟渲染 (True=先运动学set_qpos, 再物理PD驱动; False=单趟物理)
            support_table: 是否添加可见桌面支撑 (True=自适应桌面支撑GLB物体, False=仅物理地面)
            view: 相机视角 (fpv=第一人称跟随HaWoR相机, topdown=俯视, behind=后方, front=前方)
            single_gripper: 单夹爪模式 (True=只加载夹爪URDF无机械臂, 直接用MANO手腕位姿驱动)
            base_cluster: 分段固定基座 (True=将轨迹聚成N段, smoothstep过渡, 替代浮动基座)
            fixed_base: 固定基座模式 (True=基座不跟随手腕移动, 始终保持初始位置)
        """
        self.hawor_dir = Path(hawor_dir)
        self.ras_dir = Path(ras_dir)
        self.transform_params_path = Path(transform_params_path)
        self.output = output
        self.fps = fps
        self.hand_idx = hand_idx
        self.viewer = viewer
        self.crf = crf
        self.fast_collision = fast_collision
        self.hide_hand = hide_hand
        self.speed = speed
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.smooth = smooth
        self.two_pass = two_pass
        self.support_table = support_table
        self.view = view
        self.single_gripper = single_gripper
        self.base_cluster = base_cluster
        self.fixed_base = fixed_base
        self.logger = logger or logging.getLogger("PhysicsSim")
        self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / HAWOR_FOCAL_DEFAULT)
        self.scene = None

    def _find_glb_path(self):
        """查找 GLB 场景文件路径

        查找顺序:
        1. ras_dir/final_scene.glb (RAS 原始输出)
        2. ras_dir/scene_in_sapien.glb (01_align_scene 输出)
        3. transform_params 同级目录下的 scene_in_sapien.glb

        Returns:
            Path 或 None
        """
        candidates = [
            self.ras_dir / "final_scene.glb",
            self.ras_dir / "scene_in_sapien.glb",
            self.transform_params_path.parent / "scene_in_sapien.glb",
        ]
        for c in candidates:
            if c.exists():
                self.logger.info(f"  GLB 文件: {c}")
                return c
        self.logger.warning(f"  未找到 GLB 文件 (搜索: {candidates})")
        return None

    def _update_cam_fov(self, hawor_data):
        """根据 HaWoR 数据中的焦距更新相机视场角

        Args:
            hawor_data: load_hawor_data() 返回的字典
        """
        img_focal = hawor_data.get("img_focal", None)
        if img_focal is not None and img_focal > 0:
            focal_for_render = img_focal * self.cam_width / 1280.0
            self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / focal_for_render)
            self.logger.info(f"  相机焦距: {img_focal:.1f}px → {focal_for_render:.1f}px, FOV={np.degrees(self.cam_fov):.1f}°")
        else:
            self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / HAWOR_FOCAL_DEFAULT)
            self.logger.info(f"  相机焦距: 使用默认 {HAWOR_FOCAL_DEFAULT}px, FOV={np.degrees(self.cam_fov):.1f}°")

    def _render_to_sapien(self, pts_render):
        """将 HaWoR render world 坐标系的点转换到 SAPIEN 坐标系"""
        result = (RXWORLD_TO_SAPIEN @ pts_render.T).T
        if FLIP_Z_FOR_PHYSICS:
            result[..., 2] = -result[..., 2]
        return result

    def _log_object_positions(self, obj_actors, label="物体位置"):
        if not self.logger or not obj_actors:
            return
        self.logger.info(f"  ── {label} ──")
        for actor in obj_actors:
            pose = actor.get_pose()
            p = np.array(pose.p)
            self.logger.info(f"    {actor.get_name()}: pos=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    def _log_coordinate_diagnosis(self, hawor_data, mano_layer, obj_actors, start_frame=0):
        if not self.logger:
            return
        self.logger.info("\n  ═══ 坐标对齐诊断 (与02_render_scene.py对比) ═══")

        try:
            params = np.load(str(self.transform_params_path))
            s_inv = float(params['s_inv'])
            R_inv = params['R_inv']
            t_inv = params['t_inv']
            self.logger.info(f"    s_inv={s_inv:.6f}, t_inv={t_inv}")
            self.logger.info(f"    R_inv欧拉={np.degrees(trimesh.transformations.euler_from_matrix(R_inv, axes='sxyz'))}")
        except Exception as e:
            self.logger.info(f"    ⚠ 无法读取对齐参数: {e}")

        if hawor_data is not None:
            try:
                pred_trans = np.array(hawor_data["pred_trans"], dtype=np.float64)
                pred_valid = np.atleast_1d(hawor_data["pred_valid"])
                valid_mask = pred_valid.astype(bool)
                valid_frames = np.where(valid_mask)[0]
                if len(valid_frames) > 0:
                    hand_positions = pred_trans[valid_frames]
                    if hand_positions.ndim == 1:
                        hand_positions = hand_positions.reshape(-1, 3)
                    hand_mean_hawor = hand_positions.mean(axis=0)
                    hand_sapien = self._render_to_sapien(hand_mean_hawor.reshape(1, 3)).flatten()
                    self.logger.info(f"    手部均值 SAPIEN: ({hand_sapien[0]:+.4f}, {hand_sapien[1]:+.4f}, {hand_sapien[2]:+.4f})")

                    if obj_actors:
                        for actor in obj_actors:
                            pose = actor.get_pose()
                            p = np.array(pose.p)
                            dist = np.linalg.norm(p - hand_sapien)
                            self.logger.info(f"      → {actor.get_name()}: 距手部 {dist:.4f}m")
            except Exception as e:
                self.logger.info(f"    ⚠ 手部位置计算失败: {e}")

            try:
                R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)
                if R_c2w_all is not None:
                    cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
                    self.logger.info(f"    相机位置 SAPIEN: ({cam_pos[0]:+.4f}, {cam_pos[1]:+.4f}, {cam_pos[2]:+.4f})")
            except Exception as e:
                self.logger.info(f"    ⚠ 相机位置计算失败: {e}")

        if trimesh is not None and obj_actors:
            try:
                glb_path = self._find_glb_path()
                if glb_path is not None and glb_path.exists():
                    ts = trimesh.load(str(glb_path))
                    self.logger.info(f"    GLB场景图: {len(ts.graph.nodes)} 节点, {len(ts.geometry)} 几何体")
                    has_non_identity = False
                    for nn in ts.graph.nodes:
                        tf, gn = ts.graph.get(nn)
                        if gn is not None and not np.allclose(tf, np.eye(4), atol=1e-6):
                            has_non_identity = True
                            break
                    self.logger.info(f"    场景图非单位变换: {'有 (01对齐基于原始顶点,不应应用)' if has_non_identity else '无'}")
            except Exception as e:
                self.logger.info(f"    ⚠ GLB场景图分析失败: {e}")

        self.logger.info("  ═══ 诊断结束 ═══")

    def _compute_optimal_fixed_base(self, wrist_positions_sapien):
        """计算机器人基座的最优固定位置和朝向

        策略:
        1. 计算所有有效帧手腕位置的质心
        2. 基座放在质心正上方 COMFORTABLE_REACH (0.55m) 处
        3. 朝向: 绕Z轴旋转180°
        4. 检查最远手腕距离是否超出臂展

        Args:
            wrist_positions_sapien: (N, 3) 有效帧的手腕位置

        Returns:
            tuple: (base_pos, base_quat)
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

        z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
        arm_base_q = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))

        self.logger.info(f"  手腕质心(SAPIEN): {centroid}")
        self.logger.info(f"  手腕运动范围: X={wrist_range[0]:.4f} Y={wrist_range[1]:.4f} Z={wrist_range[2]:.4f}")
        self.logger.info(f"  最优固定基座位置: {arm_base_pos}")

        max_dist = 0
        for wp in wrist_positions_sapien:
            d = np.linalg.norm(wp - arm_base_pos)
            if d > max_dist:
                max_dist = d
        self.logger.info(f"  基座到最远手腕距离: {max_dist:.4f}m (臂展={ARM_MAX_REACH:.3f}m)")
        if max_dist > ARM_MAX_REACH * 0.9:
            self.logger.warning(f"  ⚠ 最远手腕距离 {max_dist:.4f}m 接近臂展 {ARM_MAX_REACH:.3f}m, IK可能不稳定!")

        return arm_base_pos, arm_base_q

    @staticmethod
    def _compute_tracking_base_pos(initial_base_pos, wrist_pos_sapien, arm_base_q):
        """计算跟踪模式下的基座位置 (小范围跟随手腕)

        基座在初始位置基础上, 沿 XY 方向跟踪手腕 (±BASE_TRACKING_RANGE=0, 固定底座),
        Z 方向保持固定。

        Args:
            initial_base_pos: (3,) 初始基座位置
            wrist_pos_sapien: (3,) 当前帧手腕位置
            arm_base_q: (4,) 基座朝向四元数

        Returns:
            np.ndarray: (3,) 调整后的基座位置
        """
        base_R = pr.matrix_from_quaternion(arm_base_q)
        wrist_in_base = base_R.T @ (wrist_pos_sapien - initial_base_pos)
        offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE
        clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)
        delta_world = base_R @ clamped_offset
        return initial_base_pos + delta_world

    @staticmethod
    def _compute_fixed_base_clusters(wrist_positions, n_clusters=BASE_CLUSTER_N,
                                      transition_frames=BASE_CLUSTER_TRANSITION_FRAMES):
        """将手腕轨迹按 XY 空间聚类成 N 个固定基座

        策略:
        1. 在 XY 平面上对 wrist positions 做 KMeans/均匀分段聚类
        2. 每个聚类计算最优固定基座位置
        3. 基座间过渡用 smoothstep 插值

        Args:
            wrist_positions: (M, 3) 有效帧的手腕位置
            n_clusters: 分段数 (默认 3)
            transition_frames: 每段过渡帧数 (默认 10)

        Returns:
            list of dict: [{
                'base_pos': (3,), 'base_quat': (4,),
                'start_frame': int, 'end_frame': int,
            }]
        """
        M = len(wrist_positions)
        if M < n_clusters * 2:
            # 帧数太少, 退回单基座模式
            if M > 0:
                wrist_arr = np.array(wrist_positions)
                centroid = wrist_arr.mean(axis=0)
                base_pos = centroid.copy()
                base_pos[2] += COMFORTABLE_REACH
                z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
                base_quat = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))
            else:
                base_pos = np.zeros(3)
                base_quat = np.array([1, 0, 0, 0])
            return [{'base_pos': base_pos, 'base_quat': base_quat, 'start_frame': 0, 'end_frame': M}]

        wrist_arr = np.array(wrist_positions)
        xy = wrist_arr[:, :2]

        # 均匀分段: 将轨迹按帧数等分为 N 段
        segment_size = M // n_clusters
        clusters = []
        for i in range(n_clusters):
            seg_start = i * segment_size
            seg_end = min((i + 1) * segment_size, M)
            if seg_end <= seg_start:
                continue
            seg_wrists = wrist_arr[seg_start:seg_end]
            centroid = seg_wrists.mean(axis=0)
            wrist_range = seg_wrists.max(axis=0) - seg_wrists.min(axis=0)

            base_pos = centroid.copy()
            base_pos[2] += COMFORTABLE_REACH
            if wrist_range[0] > 0.01:
                base_pos[0] += wrist_range[0] * 0.1

            z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
            base_quat = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))

            # 检查臂展
            max_dist = max(np.linalg.norm(wp - base_pos) for wp in seg_wrists)
            if max_dist > ARM_MAX_REACH * 0.9:
                # 超出臂展: 用全局质心基座
                wrist_centroid = wrist_arr.mean(axis=0)
                base_pos = wrist_centroid.copy()
                base_pos[2] += COMFORTABLE_REACH
                base_quat = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))

            clusters.append({
                'base_pos': base_pos,
                'base_quat': base_quat,
                'start_frame': seg_start,
                'end_frame': seg_end,
            })

        return clusters

    @staticmethod
    def _compute_frame_base_positions(clusters, num_frames):
        """预计算每帧的基座位置 (含 smoothstep 过渡)

        Args:
            clusters: _compute_fixed_base_clusters 的输出
            num_frames: 总帧数

        Returns:
            list of (base_pos, base_quat) 每帧一个
        """
        frame_bases = []
        for frame_idx in range(num_frames):
            # 找到当前帧所属的聚类
            current_cluster = None
            next_cluster = None
            for i, cluster in enumerate(clusters):
                if cluster['start_frame'] <= frame_idx < cluster['end_frame']:
                    current_cluster = cluster
                    if i + 1 < len(clusters):
                        next_cluster = clusters[i + 1]
                    break

            if current_cluster is None:
                # 帧超出范围, 用最后一个聚类
                current_cluster = clusters[-1]
                next_cluster = None

            if next_cluster is not None:
                # 检查是否在过渡区间内
                dist_to_end = current_cluster['end_frame'] - frame_idx
                transition_frames = BASE_CLUSTER_TRANSITION_FRAMES
                if dist_to_end <= transition_frames:
                    # smoothstep 过渡
                    t = (transition_frames - dist_to_end) / transition_frames
                    t = t * t * (3 - 2 * t)  # smoothstep
                    base_pos = (1 - t) * current_cluster['base_pos'] + t * next_cluster['base_pos']
                    # 四元数 slerp 近似: 线性插值后归一化
                    base_quat = (1 - t) * current_cluster['base_quat'] + t * next_cluster['base_quat']
                    base_quat = base_quat / np.linalg.norm(base_quat)
                    frame_bases.append((base_pos, base_quat))
                    continue

            frame_bases.append((current_cluster['base_pos'].copy(),
                                current_cluster['base_quat'].copy()))

        return frame_bases

    def _setup_robot(self, scene, arm_base_pos, arm_base_q):
        """创建并配置 R1 臂机器人 (与02_render_scene.py一致)

        配置内容:
        - 加载 URDF (fix_root_link=True, 固定基座, 与02一致)
        - 臂关节: PD驱动 stiffness=100000, damping=10000 (与02一致)
        - 夹爪关节: PD驱动 stiffness=100000, damping=10000 (与02一致)
        - 初始关节角: RIGHT_ARM_STARTING + gripper=[0.04, 0.04]

        Args:
            scene: SAPIEN 场景
            arm_base_pos: (3,) 基座位置
            arm_base_q: (4,) 基座朝向

        Returns:
            tuple: (robot, joint_names, arm_joint_indices, gripper_idx1, gripper_idx2, ee_link)
        """
        arm_urdf_path = _prepare_arm_urdf(FLOATING_RIGHT_URDF, "right")
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        robot = loader.load(arm_urdf_path)

        active_joints = robot.get_active_joints()
        joint_names = [j.name for j in active_joints]
        arm_joint_indices = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
        gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
        gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

        for i, joint in enumerate(active_joints):
            if i in arm_joint_indices:
                joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)
            elif i in [gripper_idx1, gripper_idx2]:
                joint.set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)
            else:
                joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)

        init_qpos = robot.get_qpos().copy()
        for j, idx in enumerate(arm_joint_indices):
            if j < len(RIGHT_ARM_STARTING):
                init_qpos[idx] = RIGHT_ARM_STARTING[j]
        init_qpos[gripper_idx1] = 0.04
        init_qpos[gripper_idx2] = 0.04
        robot.set_qpos(init_qpos)

        # 关键: 设置PD drive_target与set_qpos一致，否则PD控制器会把机械臂拉向零位
        active_joints = robot.get_active_joints()
        for i, joint in enumerate(active_joints):
            joint.set_drive_target(init_qpos[i])

        robot.set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))

        touch_link_names = [
            "right_gripper_finger_link1",
            "right_gripper_finger_link2",
        ]
        for link in robot.get_links():
            if link.get_name() in touch_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            cs.set_physical_material(
                                scene.create_physical_material(
                                    static_friction=1.0,
                                    dynamic_friction=1.0,
                                    restitution=0.6,
                                )
                            )

        # 禁用手指之间的自碰撞 + realsense_link 碰撞 (修复夹爪不对称打开)
        # 原因: finger1-finger2 自碰撞会把 joint1 卡在 0, joint2 过冲; realsense_link 与手指碰撞
        # SRDF (robot.srdf) 已声明手指互不碰撞, 但 loader.load() 未传 SRDF, 声明未生效
        # 用 set_collision_groups API:
        #   - ignore group (g2) + 相同 id (g3): 两 shape 互相忽略碰撞
        #   - [0,0,0,0]: 完全不参与任何碰撞
        finger_ignore_bit = 1 << 0  # bit 0
        finger_ignore_id = 1
        finger_link_names = {"right_gripper_finger_link1", "right_gripper_finger_link2"}
        for link in robot.get_links():
            if link.get_name() in finger_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            g = list(cs.get_collision_groups())
                            g[2] |= finger_ignore_bit
                            g[3] = finger_ignore_id
                            cs.set_collision_groups(g)
            elif link.get_name() == "right_realsense_link":
                # realsense_link 是相机, 不需要物理碰撞, 完全禁用
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            cs.set_collision_groups([0, 0, 0, 0])

        ee_link = None
        for link in robot.get_links():
            if "right_gripper_link" in link.get_name():
                ee_link = link
                break

        self.logger.info(f"  ✓ 机器人已加载: {len(arm_joint_indices)} 臂关节 + 2 夹爪关节 (物理驱动)")
        self.logger.info(f"    关节驱动: arm stiffness={JOINT_STIFFNESS}, damping={JOINT_DAMPING}")
        self.logger.info(f"    夹爪驱动: gripper stiffness={GRIPPER_STIFFNESS}, damping={GRIPPER_DAMPING}")
        self.logger.info(f"    夹爪摩擦: static=1.0, dynamic=1.0, restitution=0.6")
        self.logger.info(f"    物理子步: decimation={DECIMATION}, timestep={PHYSICS_TIMESTEP:.5f}s")

        return robot, joint_names, arm_joint_indices, gripper_idx1, gripper_idx2, ee_link

    def _get_gripper_pose_from_retargeting(self, retargeting, retarget_qpos):
        """从 retargeting 优化器的正运动学获取夹爪位姿

        Args:
            retargeting: DexRetargeting 优化器实例
            retarget_qpos: retargeting 输出的关节角

        Returns:
            tuple: (gripper_pos, gripper_R) 或 (None, None)
        """
        internal_robot = retargeting.optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)
        for i, name in enumerate(internal_robot.link_names):
            if "right_gripper_link" in name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        return None, None

    def _compute_ee_orientation_from_wrist(self, wrist_R_sapien):
        """从手腕旋转矩阵计算末端执行器朝向

        Args:
            wrist_R_sapien: (3, 3) SAPIEN 坐标系下的手腕旋转矩阵

        Returns:
            np.ndarray: (3, 3) 末端执行器旋转矩阵
        """
        R_mano2world = wrist_R_sapien @ OPERATOR2MANO_RIGHT.T
        return R_mano2world

    def _physics_step(self, robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                      arm_target, gripper_target1, gripper_target2):
        """执行一个物理控制步: 纯PD驱动 + 重力补偿 + decimation

        与 GalaxeaManipSim 控制方式一致:
        1. set_drive_target 设置PD目标 (PD控制器跟踪目标)
        2. decimation 次物理子步, 每子步:
           compute_passive_force + set_qf (重力补偿)
           scene.step() (PhysX求解PD力+补偿力+接触力)

        关键: 不调用 set_qpos!
        原因: set_qpos + set_drive_target 双重控制会导致 PhysX 求解器中
        PD力与直接位置约束冲突，产生"拉回->惯性冲出->再拉回"震荡。
        GalaxeaManipSim 从不在 step() 中调用 set_qpos，纯PD驱动。
        """
        active_joints = robot.get_active_joints()
        for i, idx in enumerate(arm_joint_indices):
            active_joints[idx].set_drive_target(arm_target[i])
        active_joints[gripper_idx1].set_drive_target(gripper_target1)
        active_joints[gripper_idx2].set_drive_target(gripper_target2)

        # decimation 次物理子步: PD控制器需要多次物理步才能收敛
        for _ in range(DECIMATION):
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            self.scene.step()

    def _kinematic_step(self, robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                        arm_target, gripper_target1, gripper_target2):
        """执行一个运动学控制步: set_qpos 直接设置关节角 (与02_render_scene.py一致)

        与 _physics_step 的区别:
        - _physics_step: PD驱动, 关节通过力跟踪目标, 有物理交互但非确定性
        - _kinematic_step: set_qpos, 关节直接设置到目标, 确定性无物理交互

        用途: 两趟渲染的第一趟 (kinematic), 产生确定性参考轨迹
        """
        qpos = robot.get_qpos().copy()
        for j_idx, arm_idx in enumerate(arm_joint_indices):
            qpos[arm_idx] = arm_target[j_idx]
        qpos[gripper_idx1] = gripper_target1
        qpos[gripper_idx2] = gripper_target2
        robot.set_qpos(qpos)
        # 单次scene.step()更新link pose (与02一致, 不需要decimation)
        self.scene.step()

    def _fetch_contacts(self, robot, obj_actors):
        """检测夹爪与物体之间的接触力

        遍历场景中的所有接触对, 筛选出夹爪-物体接触,
        统计接触点数和冲量, 按物体分组。

        Args:
            robot: SAPIEN 机器人实例
            obj_actors: GLB 物理物体 actor 列表

        Returns:
            tuple: (total_contacts, total_impulse, per_obj)
                - total_contacts: 总接触点数
                - total_impulse: 总冲量
                - per_obj: dict {obj_name: {"n": 接触点数, "impulse": 冲量}}
        """
        gripper_link_names = {"right_gripper_finger_link1", "right_gripper_finger_link2", "right_gripper_link"}
        gripper_bodies = set()
        for link in robot.get_links():
            if link.get_name() in gripper_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        gripper_bodies.add(component)

        obj_bodies = {}
        for actor in obj_actors:
            for component in actor.components:
                if isinstance(component, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                    obj_bodies[component] = actor.name
                    break

        contacts = self.scene.get_contacts()
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
                imp_norm = np.linalg.norm(np.sum(impulse, axis=0))
            total_impulse += imp_norm
            obj_body = b1 if b0 in gripper_bodies else b0
            obj_name = obj_bodies.get(obj_body, "unknown")
            if obj_name not in per_obj:
                per_obj[obj_name] = {"n": 0, "impulse": 0.0}
            per_obj[obj_name]["n"] += 1
            per_obj[obj_name]["impulse"] += imp_norm

        return total_contacts, total_impulse, per_obj

    def _update_hand_mesh(self, vertex_sapien, mano_face, mat_hand, context, internal_scene, hand_nodes):
        """更新 MANO 手部网格的渲染节点

        Args:
            vertex_sapien: (778, 3) SAPIEN 坐标系下的手部顶点
            mano_face: (F, 3) MANO 面索引
            mat_hand: 手部材质
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景
            hand_nodes: 已有的手部渲染节点列表

        Returns:
            list: 更新后的手部渲染节点列表
        """
        for node in hand_nodes:
            internal_scene.remove_node(node)
        hand_nodes.clear()
        normal = compute_smooth_shading_normal(vertex_sapien, mano_face)
        mesh = context.create_mesh_from_array(np.ascontiguousarray(vertex_sapien), mano_face, normal)
        model = context.create_model([mesh], [mat_hand])
        node = internal_scene.add_node()
        node.set_position([0, 0, 0])
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        hand_nodes.append(node)
        return hand_nodes

    def _render_keypoints(self, joints_sapien, context, internal_scene, kp_nodes,
                          radius=0.005, ref_indices=None):
        """渲染手部关键点为球体 (只渲染 ref_indices 中的关节)

        Args:
            joints_sapien: (21, 3) SAPIEN 坐标系下的关节位置
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景
            kp_nodes: 已有的关键点渲染节点列表
            radius: 球体半径 (米)
            ref_indices: retargeting 参考关节的索引集合

        Returns:
            list: 新创建的关键点渲染节点列表
        """
        for node in kp_nodes:
            internal_scene.remove_node(node)
        kp_nodes.clear()

        mat_ref = context.create_material(
            np.zeros(4), np.array([1.0, 0.0, 1.0, 1.0]), 0.0, 0.5, 0
        )

        for i, joint_pos in enumerate(joints_sapien):
            if ref_indices is not None and i not in ref_indices:
                continue
            r = radius * 2.0

            sphere = context.create_uvsphere_mesh(12, 6)
            model = context.create_model([sphere], [mat_ref])
            node = internal_scene.add_node()
            node.set_position(joint_pos.tolist())
            node.set_scale([r, r, r])
            obj = internal_scene.add_object(model, node)
            obj.shading_mode = 0
            obj.cast_shadow = False
            obj.transparency = 0
            kp_nodes.append(node)

        return kp_nodes

    def _render_cylinder_between(self, p1, p2, radius, mat, context, internal_scene):
        """在两点之间渲染一个圆柱体 (用于手部骨架线)

        Args:
            p1: 起点, shape=(3,)
            p2: 终点, shape=(3,)
            radius: 圆柱半径 (米)
            mat: 材质
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景

        Returns:
            渲染节点, 或 None
        """
        mid = (p1 + p2) / 2.0
        length = np.linalg.norm(p2 - p1)
        if length < 1e-6:
            return None
        cylinder = context.create_capsule_mesh(radius, length / 2, 8, 4)
        model = context.create_model([cylinder], [mat])
        node = internal_scene.add_node()
        node.set_position(mid.tolist())
        direction = (p2 - p1) / length
        z_axis = np.array([0, 0, 1.0])
        rot_axis = np.cross(z_axis, direction)
        rot_axis_len = np.linalg.norm(rot_axis)
        if rot_axis_len > 1e-6:
            rot_axis_n = rot_axis / rot_axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rot_quat_wxyz = pr.quaternion_from_axis_angle(
                np.array([rot_axis_n[0], rot_axis_n[1], rot_axis_n[2], angle])
            )
            rot_quat_xyzw = pr.quaternion_xyzw_from_wxyz(rot_quat_wxyz)
            node.set_rotation(rot_quat_xyzw.tolist())
        elif np.dot(z_axis, direction) < 0:
            node.set_rotation([0, 1, 0, 0])
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 0
        return node

    def _create_axes_actors(self, context, internal_scene, origin, axis_len=0.05, radius=0.003):
        """在场景中添加坐标轴可视化 (X红、Y绿、Z蓝) + 重力方向 (黄色向下)

        使用 SAPIEN 内部渲染 API (capsule mesh + node), 与 _render_cylinder_between 一致。

        Args:
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景
            origin: 坐标轴原点 (3,) np.ndarray
            axis_len: 轴长度 (米), 默认 0.05
            radius: 圆柱半径 (米), 默认 0.003

        Returns:
            list: 渲染节点列表 (需要保持引用, 避免被垃圾回收)
        """
        origin = np.array(origin)
        axes = [
            (np.array([axis_len, 0, 0]), np.array([1, 0, 0, 1]), "X"),  # 红
            (np.array([0, axis_len, 0]), np.array([0, 1, 0, 1]), "Y"),  # 绿
            (np.array([0, 0, axis_len]), np.array([0, 0, 1, 1]), "Z"),  # 蓝
        ]
        nodes = []
        for direction, color, label in axes:
            mat = context.create_material(np.zeros(4), color, 0.0, 0.5, 0)
            p1 = origin
            p2 = origin + direction
            node = self._render_cylinder_between(p1, p2, radius, mat, context, internal_scene)
            if node is not None:
                nodes.append(node)

        # 重力方向: 黄色向下箭头
        gravity_mat = context.create_material(np.zeros(4), np.array([1, 1, 0, 1]), 0.0, 0.5, 0)
        p1 = origin
        p2 = origin + np.array([0, 0, -axis_len])
        node = self._render_cylinder_between(p1, p2, radius, gravity_mat, context, internal_scene)
        if node is not None:
            nodes.append(node)

        return nodes

    def _render_hand_skeleton(self, joints_sapien, context, internal_scene, skel_nodes,
                              radius=0.002):
        """渲染手部骨架线 (关节之间的圆柱体连接)

        Args:
            joints_sapien: (21, 3) SAPIEN 坐标系下的关节位置
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景
            skel_nodes: 已有的骨架渲染节点列表
            radius: 圆柱体半径 (米)

        Returns:
            list: 新创建的骨架渲染节点列表
        """
        for node in skel_nodes:
            internal_scene.remove_node(node)
        skel_nodes.clear()
        for j1, j2 in MANO_SKELETON:
            finger_idx = JOINT_TO_FINGER[j1] if j1 < len(JOINT_TO_FINGER) else 0
            color = FINGER_GROUP_COLORS[finger_idx].copy()
            color[3] = 0.8
            p1 = joints_sapien[j1, :3]
            p2 = joints_sapien[j2, :3]
            mat = context.create_material(np.zeros(4), color, 0.0, 0.5, 0)
            node = self._render_cylinder_between(p1, p2, radius, mat, context, internal_scene)
            if node is not None:
                skel_nodes.append(node)
        return skel_nodes

    def _compute_wrist_positions_sapien(self, hawor_data, mano_layer, start_frame, num_frames):
        """预计算所有帧的手腕位置 (SAPIEN 坐标系)

        Args:
            hawor_data: load_hawor_data() 返回的字典
            mano_layer: MANOLayer 实例
            start_frame: 起始帧索引
            num_frames: 帧数

        Returns:
            list: 有效帧的手腕位置列表
        """
        positions = []
        for i in range(num_frames):
            global_idx = start_frame + i
            if not hawor_data["pred_valid"][global_idx]:
                continue
            _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                       hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
            joints_sapien = self._render_to_sapien(j)
            positions.append(joints_sapien[0, :3].copy())
        return positions

    def run_single_gripper_tracking(self, start_frame=0, num_frames=-1):
        """单夹爪模式: 只加载夹爪URDF (无机械臂), 直接用MANO手腕位姿驱动夹爪

        与 run_physics_tracking 的区别:
        - 无机械臂, 无 IK, 无 Dex Retargeting
        - 夹爪 root 位姿直接从 MANO 手腕/指尖解析计算 (_compute_analytical_gripper_pose)
        - 夹爪手指关节从 MANO 指尖距离计算
        - 物理仿真: 夹爪可碰撞/推动 GLB 物体, 但夹爪本身用 set_root_pose + set_qpos 控制

        参考: hand_track/render_gripper_only.py 的 --mode gripper

        Args:
            start_frame: 起始帧索引
            num_frames: 渲染帧数 (-1 表示全部)
        """
        self.logger.info("=" * 80)
        self.logger.info("单夹爪模式: 只加载夹爪URDF (无机械臂), MANO手腕直接驱动")
        self.logger.info("=" * 80)

        # [1/6] 加载数据
        self.logger.info("\n[1/6] 加载数据 ...")
        hawor_data = load_hawor_data(self.hawor_dir, hand_idx=self.hand_idx)
        n_total = len(hawor_data["pred_trans"])
        if num_frames < 0 or num_frames > n_total - start_frame:
            num_frames = n_total - start_frame
        num_frames = min(num_frames, n_total - start_frame)
        self.logger.info(f"  总帧数: {n_total}, 渲染: {start_frame}~{start_frame + num_frames - 1}")

        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)
        self._update_cam_fov(hawor_data)

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_side = "left" if self.hand_idx == 0 else "right"
        mano_layer = MANOLayer(mano_side, betas_mean)

        # [2/6] 创建物理场景 + 加载 GLB
        self.logger.info("\n[2/6] 创建物理场景 + 加载 GLB (带碰撞体) ...")
        glb_path = self._find_glb_path()
        ground_height = GROUND_HEIGHT
        support_plane = None
        if glb_path is not None and glb_path.exists() and self.transform_params_path.exists() and trimesh is not None:
            try:
                support_plane = _compute_object_support_plane(glb_path, self.transform_params_path)
                if support_plane is not None:
                    ground_height = support_plane['min_z'] - 0.002
                    self.logger.info(f"  桌面/地面高度: {ground_height:.4f}m")
            except Exception as e:
                self.logger.info(f"  ⚠ 支撑面计算失败: {e}")
        self.scene = setup_physics_scene(ground_height=ground_height)
        internal_scene = self.scene.render_system._internal_scene
        context = sapien.render.SapienRenderer()._internal_context

        # 桌面支撑 (可选)
        if self.support_table:
            table_builder = self.scene.create_actor_builder()
            table_mat = sapien.render.RenderMaterial()
            table_color = np.array([0.55, 0.45, 0.35, 1.0])
            if support_plane is not None and 'table_color' in support_plane:
                table_color = support_plane['table_color']
            table_mat.set_base_color(table_color.tolist())
            table_mat.set_roughness(0.8)
            if support_plane is not None:
                extent = support_plane['extent_xy']
                center = support_plane['center_xy']
                table_half_x = max(0.15, extent[0] / 2 + 0.15)
                table_half_y = max(0.15, extent[1] / 2 + 0.15)
                table_center_xy = center
            else:
                table_half_x = 0.5
                table_half_y = 0.5
                table_center_xy = np.array([0.0, 0.0])
            table_half_size = [table_half_x, table_half_y, 0.025]
            table_builder.add_box_visual(half_size=table_half_size, material=table_mat)
            table_phys_mat = self.scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
            table_builder.add_box_collision(half_size=table_half_size, material=table_phys_mat)
            table_builder.set_physx_body_type("kinematic")
            table_actor = table_builder.build(name="support_table")
            table_pos = [float(table_center_xy[0]), float(table_center_xy[1]), ground_height - 0.025]
            table_actor.set_pose(sapien.Pose(p=table_pos, q=[1, 0, 0, 0]))

        # 加载 GLB 物体 (带碰撞体)
        glb_actors = []
        if glb_path is not None and glb_path.exists() and self.transform_params_path.exists() and trimesh is not None:
            glb_actors = load_glb_with_physics(glb_path, self.transform_params_path, self.scene,
                                                logger=self.logger, fast_collision=self.fast_collision)
            self.logger.info(f"  加载 {len(glb_actors)} 个 GLB 物体")
            if glb_actors:
                self._log_object_positions(glb_actors, "GLB物体初始位置")
                # 物理稳定: 让dynamic物体自然落下, 消除初始重叠/穿透
                dynamic_actors = [
                    a for a in glb_actors
                    if any(
                        isinstance(c, sapien.pysapien.physx.PhysxRigidDynamicComponent) and not c.kinematic
                        for c in a.components
                    )
                ]
                if dynamic_actors:
                    self.logger.info(f"  稳定化 {len(dynamic_actors)} 个dynamic物体 (kinematic物体无需稳定) ...")
                    for _ in range(500):
                        self.scene.step()
                    for _ in range(100):
                        self.scene.step()
                self._log_object_positions(glb_actors, "GLB物体稳定后位置")

        # 坐标轴 + 重力方向可视化
        if support_plane is not None:
            axes_origin = np.array([support_plane['center_xy'][0], support_plane['center_xy'][1], ground_height + 0.01])
        else:
            axes_origin = np.array([0.0, 0.0, ground_height + 0.01])
        self._axes_nodes = self._create_axes_actors(context, internal_scene, axes_origin)
        self.logger.info(f"  坐标轴可视化: origin={axes_origin}")

        # [3/6] 加载夹爪 URDF (只有夹爪, 无机械臂)
        self.logger.info("\n[3/6] 加载单夹爪 URDF (无机械臂) ...")
        prefix = "right"
        gripper_urdf_path = _generate_gripper_only_urdf(prefix)
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        robot = loader.load(gripper_urdf_path)

        active_joints = robot.get_active_joints()
        joint_names = [j.name for j in active_joints]
        gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
        gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")
        self.logger.info(f"  夹爪关节: {joint_names}")

        # 设置 PD 驱动属性 (高刚度跟踪目标)
        for joint in active_joints:
            joint.set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)

        # 初始夹爪开合
        init_qpos = robot.get_qpos().copy()
        init_qpos[gripper_idx1] = GRIPPER_INIT_OPEN
        init_qpos[gripper_idx2] = GRIPPER_INIT_OPEN
        robot.set_qpos(init_qpos)
        for i, joint in enumerate(active_joints):
            joint.set_drive_target(init_qpos[i])

        # 设置夹爪手指高摩擦 (用于抓取物体)
        touch_link_names = [f"{prefix}_gripper_finger_link1", f"{prefix}_gripper_finger_link2"]
        for link in robot.get_links():
            if link.get_name() in touch_link_names:
                for component in link.entity.components:
                    if hasattr(component, 'physx_material'):
                        component.physx_material = self.scene.create_physical_material(
                            static_friction=1.0, dynamic_friction=1.0, restitution=0.0)

        # 禁用手指之间的自碰撞 (修复夹爪不对称打开, 与全臂模式一致)
        finger_ignore_bit = 1 << 0
        finger_ignore_id = 1
        finger_link_names_single = {f"{prefix}_gripper_finger_link1", f"{prefix}_gripper_finger_link2"}
        for link in robot.get_links():
            if link.get_name() in finger_link_names_single:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        for cs in component.get_collision_shapes():
                            g = list(cs.get_collision_groups())
                            g[2] |= finger_ignore_bit
                            g[3] = finger_ignore_id
                            cs.set_collision_groups(g)

        # 物理稳定化: 让物体落稳
        for _ in range(200):
            self.scene.step()

        # [4/6] 预计算手腕位置 (用于相机中心)
        self.logger.info("\n[4/6] 预计算手腕位置 ...")
        wrist_positions = self._compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
        scene_center = np.mean(wrist_positions, axis=0) if wrist_positions else np.array([0, 0, 0.3])

        # [5/6] 设置相机
        self.logger.info("\n[5/6] 设置相机 ...")
        camera = self.scene.add_camera("main", self.cam_width, self.cam_height, self.cam_fov, 0.01, 100.0)

        if self.view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  fpv视角: 跟随HaWoR相机轨迹 ({R_c2w_all.shape[0]}帧)")
        elif self.view == "topdown":
            cam_pos = scene_center + np.array([0, 0, 1.2])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  俯视视角: pos={cam_pos}")
        elif self.view == "behind":
            cam_pos = scene_center + np.array([-0.4, -0.5, 0.3])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  后方视角: pos={cam_pos}")
        elif self.view == "front":
            cam_pos = scene_center + np.array([0.5, 0.3, 0.3])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  前方视角: pos={cam_pos}")
        else:
            cam_pos = scene_center + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  回退视角: pos={cam_pos}")

        # [6/6] 渲染循环
        self.logger.info("\n[6/6] 单夹爪渲染循环 ...")
        render_fps = self.fps
        frame_repeat = max(1, round(1.0 / self.speed))

        if self.viewer:
            from sapien.utils import Viewer
            viewer = Viewer()
            viewer.set_scene(self.scene)
            viewer.control_window.show_origin_frame = True
            viewer.control_window.show_grid = False
            viewer_cam_pos = scene_center + np.array([-0.3, -0.3, 0.3])
            viewer.set_camera_xyz(x=viewer_cam_pos[0], y=viewer_cam_pos[1], z=viewer_cam_pos[2])
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output, fourcc, render_fps,
                                     (camera.get_width(), camera.get_height()))
            self.logger.info(f"  视频帧率: {render_fps}fps (speed={self.speed}x, 每帧重复{frame_repeat}次)")

        # MANO 关节索引: 拇指尖=4, 食指尖=8, 手腕=0
        ref_indices = [4, 8]

        animation_loop = True
        while animation_loop:
            if not self.viewer:
                animation_loop = False

            for local_idx in trange(num_frames, desc="单夹爪渲染", disable=False):
                global_idx = start_frame + local_idx

                # 相机更新 (仅 fpv 模式)
                if self.view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                    cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

                if hawor_data["pred_valid"][global_idx]:
                    # 计算 MANO 关节
                    _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                               hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                    joints_sapien = self._render_to_sapien(j)

                    mano_wrist = joints_sapien[0, :3]
                    mano_finger1 = joints_sapien[ref_indices[0], :3]
                    mano_finger2 = joints_sapien[ref_indices[1], :3]

                    # 解析计算夹爪位姿和手指关节
                    root_pos, root_R, joint1, joint2 = _compute_analytical_gripper_pose(
                        mano_wrist, mano_finger1, mano_finger2, prefix=mano_side)
                    root_quat = pr.quaternion_from_matrix(root_R)

                    # 设置夹爪 root 位姿 (直接设置, 无需 IK)
                    robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))

                    # 设置夹爪手指关节
                    qpos = robot.get_qpos().copy()
                    qpos[gripper_idx1] = float(joint1)
                    qpos[gripper_idx2] = float(joint2)
                    robot.set_qpos(qpos)

                    # 更新 PD drive target
                    active_joints = robot.get_active_joints()
                    active_joints[gripper_idx1].set_drive_target(float(joint1))
                    active_joints[gripper_idx2].set_drive_target(float(joint2))

                # 物理仿真步进 (decimation 次, 让物体与夹爪交互)
                for _ in range(DECIMATION):
                    qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                    robot.set_qf(qf)
                    self.scene.step()

                # 渲染
                if self.viewer:
                    viewer.render()
                else:
                    self.scene.update_render()
                    camera.take_picture()
                    rgb = camera.get_picture("Color")[..., :3]
                    bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                    # 叠加坐标轴信息
                    cv2.putText(bgr, f"Table Z={ground_height:.4f}m", (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cam_pos = np.array(camera.get_local_pose().p)
                    cv2.putText(bgr, f"Cam Z={cam_pos[2]:.4f}m", (15, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.putText(bgr, f"Origin Z={axes_origin[2]:.4f}m", (15, 96),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    for _ in range(frame_repeat):
                        writer.write(bgr)

        if not self.viewer:
            writer.release()
            self.logger.info(f"\n✓ 单夹爪渲染完成: {self.output}")
            # ffmpeg 重编码
            reencode_with_ffmpeg(self.output, self.output, crf=self.crf, fps=render_fps, logger=self.logger)

    def run_physics_tracking(self, start_frame=0, num_frames=-1):
        """执行物理仿真驱动: 真实抓取与交互

        8个步骤:
        1. 加载 HaWoR 数据 + 相机轨迹
        2. 创建物理场景 + 加载 GLB (带碰撞体)
        3. 初始化 R1 单臂机器人 (物理驱动)
        4. 初始化 Dex Retargeting
        5. 初始化 RelaxedIK + 预计算 qpos
        6. 设置相机
        7. Warmup + 预计算
        8. 物理仿真渲染 (PD驱动 + 接触力检测)

        Args:
            start_frame: 起始帧索引
            num_frames: 渲染帧数 (-1 表示全部)
        """
        # 单夹爪模式: 走独立的渲染流程 (无机械臂, 直接用MANO手腕位姿驱动夹爪)
        if self.single_gripper:
            return self.run_single_gripper_tracking(start_frame=start_frame, num_frames=num_frames)

        self.logger.info("=" * 80)
        self.logger.info("模式4: 物理仿真驱动 — 真实抓取与交互")
        self.logger.info("=" * 80)

        self.logger.info("\n[1/8] 加载数据 ...")
        hawor_data = load_hawor_data(self.hawor_dir, hand_idx=self.hand_idx)
        n_total = len(hawor_data["pred_trans"])
        if num_frames < 0 or num_frames > n_total - start_frame:
            num_frames = n_total - start_frame
        num_frames = min(num_frames, n_total - start_frame)
        self.logger.info(f"  总帧数: {n_total}, 渲染: {start_frame}~{start_frame + num_frames - 1}")

        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_side = "left" if self.hand_idx == 0 else "right"
        mano_layer = MANOLayer(mano_side, betas_mean)
        mano_face = mano_layer.f.cpu().numpy()

        self._update_cam_fov(hawor_data)

        self.logger.info("\n[2/8] 创建物理场景 + 加载 GLB (带碰撞体) ...")
        glb_path = self._find_glb_path()
        ground_height = GROUND_HEIGHT
        support_plane = None
        if glb_path is not None and glb_path.exists() and self.transform_params_path.exists() and trimesh is not None:
            try:
                support_plane = _compute_object_support_plane(glb_path, self.transform_params_path)
                if support_plane is not None:
                    ground_height = support_plane['min_z'] - 0.002
                    self.logger.info(f"  物体Z范围: [{support_plane['min_z']:.4f}, {support_plane['max_z']:.4f}]m")
                    self.logger.info(f"  物体XY质心: {support_plane['center_xy']}, 范围: {support_plane['extent_xy']}")
                    self.logger.info(f"  桌面/地面高度: {ground_height:.4f}m (物体最低点下方2mm, 紧贴支撑)")
            except Exception as e:
                self.logger.info(f"  ⚠ 支撑面计算失败: {e}, 使用默认地面: {ground_height:.4f}m")
        else:
            self.logger.info(f"  使用默认地面: {ground_height:.4f}m")
        self.scene = setup_physics_scene(ground_height=ground_height)
        internal_scene = self.scene.render_system._internal_scene
        context = sapien.render.SapienRenderer()._internal_context
        mat_hand = context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1.0]), 0.0, 0.8, 0)

        # 添加可见桌面支撑 (可选): 自适应位置和大小, 用于支撑GLB物体
        # 桌面位置 = GLB物体XY质心, 桌面大小 = GLB物体XY范围 + 边距
        TABLE_HALF_THICKNESS = 0.025  # 桌面半厚度=2.5cm (总厚度5cm), 防止物体被推出桌面
        TABLE_XY_MARGIN = 0.15  # 桌面XY边距=15cm, 防止物体从边缘掉落
        if self.support_table:
            table_builder = self.scene.create_actor_builder()
            table_mat = sapien.render.RenderMaterial()
            table_color = np.array([0.55, 0.45, 0.35, 1.0])  # 默认木色
            if support_plane is not None and 'table_color' in support_plane:
                table_color = support_plane['table_color']
            table_mat.set_base_color(table_color.tolist())
            table_mat.set_roughness(0.8)
            table_mat.set_metallic(0.0)
            if support_plane is not None:
                # 自适应桌面: 基于GLB物体范围
                extent = support_plane['extent_xy']
                center = support_plane['center_xy']
                # 桌面半尺寸 = 物体范围/2 + 边距, 最小0.15m
                table_half_x = max(0.15, extent[0] / 2 + TABLE_XY_MARGIN)
                table_half_y = max(0.15, extent[1] / 2 + TABLE_XY_MARGIN)
                table_center_xy = center
                self.logger.info(f"  自适应桌面: 中心={table_center_xy}, 半尺寸=[{table_half_x:.3f}, {table_half_y:.3f}]")
                if support_plane.get('table_surface_z') is not None:
                    self.logger.info(f"  检测到GLB桌面表面: Z={support_plane['table_surface_z']:.4f}m, 颜色={table_color}")
            else:
                # 默认桌面: 1m x 1m 在原点
                table_half_x = 0.5
                table_half_y = 0.5
                table_center_xy = np.array([0.0, 0.0])
            table_half_size = [table_half_x, table_half_y, TABLE_HALF_THICKNESS]
            table_builder.add_box_visual(half_size=table_half_size, material=table_mat)
            table_phys_mat = self.scene.create_physical_material(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
            table_builder.add_box_collision(half_size=table_half_size, material=table_phys_mat)
            table_builder.set_physx_body_type("kinematic")
            table_actor = table_builder.build(name="support_table")
            # 桌面顶部 = ground_height, 所以桌面中心Z = ground_height - TABLE_HALF_THICKNESS
            table_pos = [float(table_center_xy[0]), float(table_center_xy[1]), ground_height - TABLE_HALF_THICKNESS]
            table_actor.set_pose(sapien.Pose(p=table_pos, q=[1, 0, 0, 0]))
            self.logger.info(f"  ✓ 可见桌面支撑: pos={table_pos} ({table_half_x*2:.2f}m x {table_half_y*2:.2f}m, 厚{TABLE_HALF_THICKNESS*2:.2f}m)")
        else:
            self.logger.info(f"  桌面支撑已禁用 (--no-support-table), 仅使用物理地面")

        # 坐标轴 + 重力方向可视化
        if support_plane is not None:
            axes_origin = np.array([support_plane['center_xy'][0], support_plane['center_xy'][1], ground_height + 0.01])
        else:
            axes_origin = np.array([0.0, 0.0, ground_height + 0.01])
        self._axes_nodes = self._create_axes_actors(context, internal_scene, axes_origin)
        self.logger.info(f"  坐标轴可视化: origin={axes_origin}")

        obj_actors = []
        settled_obj_poses = []  # 稳定后的物体位置，供第二趟重放
        if glb_path is not None and glb_path.exists() and self.transform_params_path.exists():
            obj_actors = load_glb_with_physics(glb_path, self.transform_params_path, self.scene, logger=self.logger, fast_collision=self.fast_collision)
            if obj_actors:
                self.logger.info(f"  ✓ GLB 物理物体: {len(obj_actors)} 个")
                self._log_object_positions(obj_actors, "GLB物体初始位置")
                # 物理稳定: 只对dynamic物体做稳定化, kinematic物体不需要
                dynamic_actors = [
                    a for a in obj_actors
                    if any(
                        isinstance(c, sapien.pysapien.physx.PhysxRigidDynamicComponent) and not c.kinematic
                        for c in a.components
                    )
                ]
                if dynamic_actors:
                    self.logger.info(f"  稳定化 {len(dynamic_actors)} 个dynamic物体 (kinematic物体无需稳定) ...")
                    for _ in range(500):
                        self.scene.step()
                    for _ in range(100):
                        self.scene.step()
                self._log_object_positions(obj_actors, "GLB物体稳定后位置")
                # 保存稳定后的物体位置，供第二趟重放使用
                settled_obj_poses = []
                for actor in obj_actors:
                    pose = actor.get_pose()
                    settled_obj_poses.append((np.array(pose.p), np.array(pose.q)))
            else:
                self.logger.error(f"  ✗ GLB 加载失败")
        else:
            if not glb_path.exists():
                self.logger.error(f"  ✗ GLB 文件不存在: {glb_path}")
            if not self.transform_params_path.exists():
                self.logger.error(f"  ✗ 变换参数不存在: {self.transform_params_path}")

        self.logger.info("\n[3/8] 初始化 R1 单臂机器人 (物理驱动) ...")
        wrist_positions = self._compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
        if not wrist_positions:
            raise RuntimeError("无法提取有效手腕位置")

        arm_base_pos, arm_base_q = self._compute_optimal_fixed_base(wrist_positions)

        # 分段固定基座: 预计算每帧的基座位置
        frame_base_positions = None
        if self.base_cluster:
            clusters = self._compute_fixed_base_clusters(wrist_positions, n_clusters=BASE_CLUSTER_N)
            frame_base_positions = self._compute_frame_base_positions(clusters, num_frames)
            self.logger.info(f"  分段固定基座: {len(clusters)} 段")
            for i, c in enumerate(clusters):
                self.logger.info(f"    段{i}: 帧[{c['start_frame']}-{c['end_frame']}), base={c['base_pos']}")
        else:
            self.logger.info(f"  浮动基座模式: XY跟踪范围±{BASE_TRACKING_RANGE}m")
        if self.fixed_base:
            self.logger.info(f"  固定基座模式: 基座位置固定在 {arm_base_pos}")

        robot, joint_names, arm_joint_indices, gripper_idx1, gripper_idx2, ee_link = \
            self._setup_robot(self.scene, arm_base_pos, arm_base_q)

        # 与02一致: set_root_pose + 单次scene.step() + update_render
        # 不做50步物理仿真 (PD target=0会把机械臂拉向零位)
        robot.set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        self.scene.step()
        self.scene.update_render()

        base_link_p, base_link_q = None, None
        for link in robot.get_links():
            if "right_arm_base_link" == link.get_name():
                pose = link.get_entity_pose()
                base_link_p = np.array(pose.p)
                base_link_q = np.array(pose.q)
                break
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T
        self.logger.info(f"  臂基座位置: {arm_base_pos} (跟踪范围±{BASE_TRACKING_RANGE}m)")
        self.logger.info(f"  base_link 位置: {base_link_p}")

        self._log_coordinate_diagnosis(hawor_data, mano_layer, obj_actors, start_frame)

        self.logger.info("\n[4/8] 初始化 Dex Retargeting ...")
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, HandType.right)
        override = dict(
            add_dummy_free_joint=True,
            normal_delta=1e-5,
            huber_delta=0.01,
            target_link_names=[
                "right_gripper_finger_link1",
                "right_gripper_finger_link2",
                "right_gripper_link",
            ],
            target_link_human_indices=np.array([4, 8, 0]),
        )
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        ref_indices = retargeting.optimizer.target_link_human_indices
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed

        retarget2sapien = np.array(
            [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
        ).astype(int)
        sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(retarget2sapien):
            sapien2retarget[retarget_i] = sapien_i
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        init_qpos_robot = robot.get_qpos().copy()
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in sapien2retarget:
                fixed_qpos[i] = init_qpos_robot[sapien2retarget[retarget_idx]]
        self.logger.info(f"  重定向索引: {ref_indices} (3约束点: 4=拇指尖, 8=食指尖, 0=手腕)")

        for probe_idx in range(num_frames):
            g_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][g_idx]:
                continue
            _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][g_idx],
                                       hawor_data["pred_hand_pose"][g_idx], hawor_data["pred_trans"][g_idx])
            joints_sapien = self._render_to_sapien(j)
            wrist_R_render = pr.matrix_from_compact_axis_angle(hawor_data["pred_rot"][g_idx])
            wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
            if FLIP_Z_FOR_PHYSICS:
                Z_FLIP_R = np.diag([1.0, 1.0, -1.0])
                wrist_R_sapien = Z_FLIP_R @ wrist_R_sapien @ Z_FLIP_R
            wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
            retargeting.warm_start(
                joints_sapien[0, :3], wrist_quat,
                hand_type=HandType.right, is_mano_convention=True,
            )
            self.logger.info(f"  ✓ Warm start 完成 (帧 {g_idx})")
            break

        self.logger.info("\n[5/8] 初始化 RelaxedIK + 预计算 ...")
        from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver
        ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=IK_TOLERANCES,
        )
        ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

        mapping_offset = np.zeros(3)
        safety_offset = np.zeros(3)

        first_ik_joints = None
        first_ik_target_base = None
        first_ee_quat_base = None
        first_ik_target_world = None
        for probe_idx in range(num_frames):
            global_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][global_idx]:
                continue
            _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                       hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
            joints_sapien = self._render_to_sapien(j)

            ref_value = joints_sapien[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

            gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retargeting, retarget_qpos)
            if gripper_pos_fk is None:
                continue

            if self.fixed_base:
                tracked_base = arm_base_pos.copy()
                tracked_base_q = arm_base_q.copy()
            elif self.base_cluster and frame_base_positions is not None:
                tracked_base = frame_base_positions[probe_idx][0]
                tracked_base_q = frame_base_positions[probe_idx][1]
            else:
                tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                tracked_base_q = arm_base_q
            robot.set_root_pose(sapien.Pose(tracked_base.tolist(), tracked_base_q.tolist()))
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            self.scene.step()

            for link in robot.get_links():
                if "right_arm_base_link" == link.get_name():
                    pose = link.get_entity_pose()
                    base_link_p = np.array(pose.p)
                    base_link_q = np.array(pose.q)
                    break
            base_link_R = pr.matrix_from_quaternion(base_link_q)
            base_link_R_inv = base_link_R.T

            ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
            ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
            ee_R_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

            try:
                first_ik_joints = np.array(ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist()))
                first_ik_target_base = ik_target_b.copy()
                first_ee_quat_base = ee_quat_b.copy()
                first_ik_target_world = ik_target_raw.copy()
                break
            except Exception:
                continue

        if first_ik_joints is None:
            raise RuntimeError("无法求解任何有效帧的IK")

        for _ in range(IK_SOLVE_PER_FRAME * 5 - 1):
            first_ik_joints = np.array(ik_solver.solve_position_right(first_ik_target_base.tolist(), first_ee_quat_base.tolist()))

        self.logger.info("\n[6/8] 设置相机 ...")
        camera = self.scene.add_camera("main", self.cam_width, self.cam_height, self.cam_fov, 0.01, 100.0)

        # 计算场景中心 (手腕质心或默认值), 用于固定视角
        scene_center = np.mean(wrist_positions, axis=0) if wrist_positions else np.array([0, 0, 0.3])

        if self.view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  使用 hawor 相机轨迹 ({R_c2w_all.shape[0]}帧, fpv视角)")
        elif self.view == "topdown":
            cam_pos = scene_center + np.array([0, 0, 1.2])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  俯视视角: pos={cam_pos}, 看向={scene_center}")
        elif self.view == "behind":
            cam_pos = scene_center + np.array([-0.4, -0.5, 0.3])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  后方视角: pos={cam_pos}, 看向={scene_center}")
        elif self.view == "front":
            cam_pos = scene_center + np.array([0.5, 0.3, 0.3])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  前方视角: pos={cam_pos}, 看向={scene_center}")
        else:
            # fpv 但无 HaWoR 相机轨迹, 回退到后方视角
            cam_pos = scene_center + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, scene_center)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  回退视角(无HaWoR轨迹): pos={cam_pos}, 看向={scene_center}")

        self.logger.info("\n[7/8] Warmup (与02一致) ...")
        start_joints = np.array(RIGHT_ARM_STARTING)
        for w in range(WARMUP_FRAMES):
            t = (w + 1) / WARMUP_FRAMES
            t_smooth = t * t * (3 - 2 * t)
            interp = start_joints * (1 - t_smooth) + first_ik_joints * t_smooth
            self._physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                               interp, 0.04, 0.04)  # 同号: URDF finger_joint axis 相反 + limit 非负, 同号才对称开合

        self.logger.info("\n[8/8] 实时IK渲染 (与02_render_scene.py一致) ...")
        self.logger.info(f"  平滑模式: {self.smooth} ({'不平滑' if self.smooth == 0 else '在线EMA' if self.smooth == 1 else '后处理双向滤波'})")

        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        current_joints = np.array([robot.get_qpos()[i] for i in arm_joint_indices])
        joint_filter.next(current_joints)

        target_smoother = EmaTargetSmoother(pos_alpha=EMA_POS_ALPHA, ori_alpha=EMA_ORI_ALPHA) if self.smooth == 1 else None

        render_fps = self.fps
        frame_repeat = max(1, round(1.0 / self.speed))

        wrist_centroid = np.mean(wrist_positions, axis=0) if wrist_positions else arm_base_pos.copy()
        if self.viewer:
            from sapien.utils import Viewer
            viewer = Viewer()
            viewer.set_scene(self.scene)
            viewer.control_window.show_origin_frame = True
            viewer.control_window.show_grid = False
            viewer_cam_pos = wrist_centroid + np.array([-0.3, -0.3, 0.3])
            viewer.set_camera_xyz(x=viewer_cam_pos[0], y=viewer_cam_pos[1], z=viewer_cam_pos[2])
            self.logger.info(f"  Viewer 相机: pos={viewer_cam_pos}, 目标={wrist_centroid}")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output, fourcc, render_fps,
                                     (camera.get_width(), camera.get_height()))
            self.logger.info(f"  视频帧率: {render_fps}fps ({self.speed}x速度, 每帧重复{frame_repeat}次)")

        kp_nodes = []
        wrist_pos_sapien = None
        qpos_log = []

        gripper_link = None
        for link in robot.get_links():
            if "right_gripper_link" in link.get_name():
                gripper_link = link
                break

        base_pose_log = []  # 记录每帧的base pose，供第二趟重放
        last_tracked_base = arm_base_pos.copy()  # 初始化，供无效帧使用

        animation_loop = True
        while animation_loop:
            if not self.viewer:
                animation_loop = False

            for local_idx in trange(num_frames, desc="实时IK渲染", disable=False):
                global_idx = start_frame + local_idx

                # 仅 fpv 视角跟随 HaWoR 相机轨迹, 其他视角保持固定
                if self.view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                    cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

                if hawor_data["pred_valid"][global_idx]:
                    _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                               hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                    joints_sapien = self._render_to_sapien(j)
                    kp_nodes = self._render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes,
                                                      radius=0.006, ref_indices=set(ref_indices))

                    ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                    retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
                    sapien_qpos = retarget_qpos[retarget2sapien]

                    # [DEBUG] 打印重定向输出和映射后的夹爪关节值
                    if local_idx < 5:
                        r_idx1 = retargeting.joint_names.index("right_gripper_finger_joint1")
                        r_idx2 = retargeting.joint_names.index("right_gripper_finger_joint2")
                        self.logger.info(f"  [DEBUG 夹爪 帧{local_idx}] retarget_qpos[joint1]={retarget_qpos[r_idx1]:.6f}, retarget_qpos[joint2]={retarget_qpos[r_idx2]:.6f}")
                        self.logger.info(f"  [DEBUG 夹爪 帧{local_idx}] sapien_qpos[idx1={gripper_idx1}]={sapien_qpos[gripper_idx1] if gripper_idx1 < len(sapien_qpos) else 'OOB':.6f}, sapien_qpos[idx2={gripper_idx2}]={sapien_qpos[gripper_idx2] if gripper_idx2 < len(sapien_qpos) else 'OOB':.6f}")
                        self.logger.info(f"  [DEBUG 夹爪 帧{local_idx}] len(sapien_qpos)={len(sapien_qpos)}, gripper_idx1={gripper_idx1}, gripper_idx2={gripper_idx2}")

                    gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(
                        retargeting, retarget_qpos)

                    if self.fixed_base:
                        tracked_base = arm_base_pos.copy()
                        tracked_base_q = arm_base_q.copy()
                    elif self.base_cluster and frame_base_positions is not None:
                        tracked_base = frame_base_positions[local_idx][0]
                        tracked_base_q = frame_base_positions[local_idx][1]
                    else:
                        tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                        tracked_base_q = arm_base_q
                    robot.set_root_pose(sapien.Pose(tracked_base.tolist(), tracked_base_q.tolist()))
                    # 关键: 必须调用scene.step()更新link pose, 否则base_link_p/q是旧值
                    # 与02_render_scene.py一致, 但04需要重力补偿防止机械臂下坠
                    qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                    robot.set_qf(qf)
                    self.scene.step()

                    # 记录base pose供第二趟重放
                    base_pose_log.append((tracked_base.copy(), tracked_base_q.copy()))
                    last_tracked_base = tracked_base.copy()

                    for link in robot.get_links():
                        if "right_arm_base_link" == link.get_name():
                            pose = link.get_entity_pose()
                            base_link_p = np.array(pose.p)
                            base_link_q = np.array(pose.q)
                            break
                    base_link_R = pr.matrix_from_quaternion(base_link_q)
                    base_link_R_inv = base_link_R.T

                    # 诊断日志: 前5帧打印关键中间值
                    if local_idx < 5:
                        self.logger.info(f"  [诊断 帧{local_idx}] gripper_pos_fk={np.array2string(gripper_pos_fk*1000, precision=1)}mm")
                        self.logger.info(f"  [诊断 帧{local_idx}] wrist_pos_sapien={np.array2string(joints_sapien[0,:3]*1000, precision=1)}mm")
                        self.logger.info(f"  [诊断 帧{local_idx}] gripper-wrist距离={np.linalg.norm(gripper_pos_fk - joints_sapien[0,:3])*1000:.1f}mm")
                        self.logger.info(f"  [诊断 帧{local_idx}] tracked_base={np.array2string(tracked_base*1000, precision=1)}mm, arm_base_q={arm_base_q}")
                        self.logger.info(f"  [诊断 帧{local_idx}] base_link_p={np.array2string(base_link_p*1000, precision=1)}mm, base_link_q={base_link_q}")
                        self.logger.info(f"  [诊断 帧{local_idx}] base_link_R diag={np.diag(base_link_R)}")
                        self.logger.info(f"  [诊断 帧{local_idx}] tracked_base==base_link_p? {np.allclose(tracked_base, base_link_p, atol=1e-4)}")
                        # 检查gripper_pos_fk是否在SAPIEN世界坐标系中
                        gripper_wrist_dist = np.linalg.norm(gripper_pos_fk - joints_sapien[0,:3])
                        if gripper_wrist_dist > 0.5:
                            self.logger.warning(f"  [诊断 帧{local_idx}] ⚠ gripper_pos_fk可能不在SAPIEN世界坐标系中! 距手腕{gripper_wrist_dist*1000:.1f}mm")

                    ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
                    ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
                    ee_R_base = base_link_R_inv @ R_ee_world_fk
                    ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

                    if local_idx < 5:
                        self.logger.info(f"  [诊断 帧{local_idx}] ik_target_raw={np.array2string(ik_target_raw*1000, precision=1)}mm")
                        self.logger.info(f"  [诊断 帧{local_idx}] ik_target_b={np.array2string(ik_target_b*1000, precision=1)}mm")

                    if target_smoother is not None:
                        ik_target_b, ee_quat_b = target_smoother.smooth(ik_target_b, ee_quat_b)

                    ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())
                    for _ in range(IK_SOLVE_PER_FRAME - 1):
                        ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())

                    if self.smooth == 0:
                        filtered_joints = np.array(ik_joints)
                    else:
                        filtered_joints = joint_filter.next(np.array(ik_joints))

                    gripper_target1 = float(sapien_qpos[gripper_idx1]) if gripper_idx1 < len(sapien_qpos) else 0.04
                    gripper_target1 = max(0.0, min(0.05, gripper_target1))  # clamp 到 URDF limit
                    gripper_target2 = gripper_target1  # 同号, axis 相反, 对称开合

                    # [DEBUG] 打印 gripper target 和物理仿真后实际 qpos
                    if local_idx < 5:
                        self.logger.info(f"  [DEBUG 夹爪 帧{local_idx}] gripper_target1={gripper_target1:.6f}, gripper_target2={gripper_target2:.6f}")

                    if self.viewer:
                        # viewer模式: 纯PD驱动 + 重力补偿 + decimation (与_physics_step一致)
                        active_joints = robot.get_active_joints()
                        for i, idx in enumerate(arm_joint_indices):
                            active_joints[idx].set_drive_target(filtered_joints[i])
                        active_joints[gripper_idx1].set_drive_target(gripper_target1)
                        active_joints[gripper_idx2].set_drive_target(gripper_target2)
                        for _ in range(DECIMATION):
                            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                            robot.set_qf(qf)
                            self.scene.step()
                        # 记录PD目标值（IK输出）
                        qpos_v = robot.get_qpos().copy()
                        qpos_log.append(qpos_v.copy())
                    elif self.two_pass:
                        # 两趟渲染第一趟: 运动学 (set_qpos, 确定性, 与02一致)
                        self._kinematic_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                             filtered_joints, gripper_target1, gripper_target2)
                        # 保存IK目标轨迹 (供第二趟PD驱动使用)
                        qpos_target = robot.get_qpos().copy()
                        for j_idx, arm_idx in enumerate(arm_joint_indices):
                            qpos_target[arm_idx] = filtered_joints[j_idx]
                        qpos_target[gripper_idx1] = gripper_target1
                        qpos_target[gripper_idx2] = gripper_target2
                        qpos_log.append(qpos_target)
                    else:
                        self._physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                           filtered_joints, gripper_target1, gripper_target2)
                        # 纯PD驱动: 记录实际qpos（物理仿真后的真实关节角）
                        qpos_log.append(robot.get_qpos().copy())

                    # [DEBUG] 打印物理仿真后实际夹爪 qpos
                    if local_idx < 5:
                        actual_qpos = robot.get_qpos()
                        self.logger.info(f"  [DEBUG 夹爪 帧{local_idx}] actual_qpos[idx1]={actual_qpos[gripper_idx1]:.6f}, actual_qpos[idx2]={actual_qpos[gripper_idx2]:.6f}, diff={abs(actual_qpos[gripper_idx1] - actual_qpos[gripper_idx2]):.6f}")

                        # [DEBUG] 检测夹爪手指的所有接触 (包括与桌面/GLB/自身)
                        finger_link_names = {"right_gripper_finger_link1", "right_gripper_finger_link2"}
                        finger_bodies = {}
                        for link in robot.get_links():
                            if link.get_name() in finger_link_names:
                                for component in link.entity.components:
                                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                                        finger_bodies[component] = link.get_name()
                        all_contacts = self.scene.get_contacts()
                        f1_contacts = 0
                        f2_contacts = 0
                        for c in all_contacts:
                            b0, b1 = c.bodies[0], c.bodies[1]
                            if b0 in finger_bodies or b1 in finger_bodies:
                                # 找到另一侧的 body
                                if b0 in finger_bodies:
                                    finger_name = finger_bodies[b0]
                                    other_body = b1
                                else:
                                    finger_name = finger_bodies[b1]
                                    other_body = b0
                                # 尝试获取另一侧的名称
                                other_name = "unknown"
                                for link2 in robot.get_links():
                                    for comp2 in link2.entity.components:
                                        if isinstance(comp2, sapien.pysapien.physx.PhysxArticulationLinkComponent) and comp2 == other_body:
                                            other_name = link2.get_name()
                                if "finger_link1" in finger_name:
                                    f1_contacts += 1
                                elif "finger_link2" in finger_name:
                                    f2_contacts += 1
                                if local_idx < 2:
                                    self.logger.info(f"  [DEBUG 接触 帧{local_idx}] {finger_name} <-> {other_name}")
                        self.logger.info(f"  [DEBUG 接触 帧{local_idx}] finger_link1 接触数={f1_contacts}, finger_link2 接触数={f2_contacts}")

                        # [DEBUG] 打印手指 link 世界位置
                        for link in robot.get_links():
                            if link.get_name() in finger_link_names:
                                pose = link.get_entity_pose()
                                self.logger.info(f"  [DEBUG 位置 帧{local_idx}] {link.get_name()}: pos=({pose.p[0]:.4f}, {pose.p[1]:.4f}, {pose.p[2]:.4f})")

                    # 诊断: 检查set_qpos后实际EE位置 vs IK目标
                    if local_idx < 5 and gripper_link is not None:
                        ee_actual = np.array(gripper_link.get_entity_pose().p)
                        ee_err_mm = np.linalg.norm(ee_actual - ik_target_raw) * 1000
                        self.logger.info(f"  [诊断 帧{local_idx}] ik_joints={np.array2string(np.degrees(filtered_joints), precision=1)}°")
                        self.logger.info(f"  [诊断 帧{local_idx}] ee_actual={np.array2string(ee_actual*1000, precision=1)}mm, ik_target_raw={np.array2string(ik_target_raw*1000, precision=1)}mm, err={ee_err_mm:.1f}mm")
                        qpos_after = robot.get_qpos()
                        arm_qpos_after = np.array([qpos_after[i] for i in arm_joint_indices])
                        self.logger.info(f"  [诊断 帧{local_idx}] qpos_set={np.array2string(np.degrees(filtered_joints), precision=1)}°, qpos_after={np.array2string(np.degrees(arm_qpos_after), precision=1)}°, diff={np.degrees(np.abs(filtered_joints - arm_qpos_after)).max():.3f}°")
                    wrist_pos_sapien = joints_sapien[0, :3].copy()
                else:
                    for node in kp_nodes:
                        internal_scene.remove_node(node)
                    kp_nodes.clear()

                    # 无效帧: 重力补偿 + decimation次scene.step()
                    for _ in range(DECIMATION):
                        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                        robot.set_qf(qf)
                        self.scene.step()
                    # 无效帧也记录base pose以保持索引对齐
                    base_pose_log.append((last_tracked_base.copy(), arm_base_q.copy()))

                self.scene.update_render()

                n_contacts, total_impulse, per_obj = self._fetch_contacts(robot, obj_actors)

                if not self.viewer:
                    camera.take_picture()
                    rgb = camera.get_picture("Color")[..., :3]
                    bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                    h, w = bgr.shape[:2]
                    cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                    ee_err_cm = None
                    if gripper_link is not None and wrist_pos_sapien is not None:
                        ee_pos = np.array(gripper_link.get_entity_pose().p)
                        ee_err_cm = np.linalg.norm(ee_pos - wrist_pos_sapien) * 100
                    label = f"Frame {local_idx+1}/{num_frames}  |  Physics Sim"
                    if ee_err_cm is not None:
                        err_color = (0, 255, 0) if ee_err_cm < 2 else (0, 255, 255) if ee_err_cm < 5 else (0, 0, 255)
                        label += f"  EE:{ee_err_cm:.1f}cm"
                    label += f"  C:{n_contacts}"
                    cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                err_color if ee_err_cm is not None else (255, 255, 255), 2)
                    # 坐标轴信息
                    cv2.putText(bgr, f"Table Z={ground_height:.4f}m", (15, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cam_pos_val = np.array(camera.get_local_pose().p)
                    cv2.putText(bgr, f"Cam Z={cam_pos_val[2]:.4f}m", (15, 78),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    cv2.putText(bgr, f"Origin Z={axes_origin[2]:.4f}m", (15, 96),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    for _ in range(frame_repeat):
                        writer.write(bgr)
                else:
                    viewer.render()

                if (local_idx + 1) % 30 == 0 and not self.viewer:
                    obj_detail = " | ".join(f"{k[:15]}:C={v['n']}" for k, v in per_obj.items())
                    self.logger.info(f"  帧 {local_idx+1}/{num_frames}: contacts={n_contacts}, impulse={total_impulse:.2f}, {obj_detail}")

            if self.viewer:
                self.logger.info("  动画播放完成, 重新开始... (关闭窗口退出)")
                init_qpos = robot.get_qpos().copy()
                for j, idx in enumerate(arm_joint_indices):
                    if j < len(RIGHT_ARM_STARTING):
                        init_qpos[idx] = RIGHT_ARM_STARTING[j]
                init_qpos[gripper_idx1] = 0.04
                init_qpos[gripper_idx2] = 0.04
                robot.set_qpos(init_qpos)

        if not self.viewer:
            writer.release()

        for node in kp_nodes:
            internal_scene.remove_node(node)

        # === 两趟渲染: 第二趟 (物理PD驱动, 使用第一趟的运动学轨迹) ===
        if not self.viewer and self.two_pass and qpos_log:
            self.logger.info("\n  === 两趟渲染: 第二趟 (物理PD驱动 + 重力补偿) ===")
            self.logger.info(f"  第一趟(运动学)轨迹: {len(qpos_log)} 帧")
            qpos_arr = np.array(qpos_log)
            self.logger.info(f"  轨迹帧间: 平均 {np.degrees(np.abs(np.diff(qpos_arr[:, :6], axis=0))).mean():.2f}°, "
                             f"最大 {np.degrees(np.abs(np.diff(qpos_arr[:, :6], axis=0))).max():.2f}°")

            # 重置机器人到初始状态
            self.logger.info("  重置机器人到 RIGHT_ARM_STARTING ...")
            reset_qpos = robot.get_qpos().copy()
            for j, idx in enumerate(arm_joint_indices):
                if j < len(RIGHT_ARM_STARTING):
                    reset_qpos[idx] = RIGHT_ARM_STARTING[j]
            reset_qpos[gripper_idx1] = 0.04
            reset_qpos[gripper_idx2] = 0.04
            robot.set_qpos(reset_qpos)
            # 设置PD target与set_qpos一致, 防止PD控制器拉向零位
            active_joints = robot.get_active_joints()
            for i, joint in enumerate(active_joints):
                joint.set_drive_target(reset_qpos[i])

            # 重置GLB物体到稳定后位置
            if obj_actors and settled_obj_poses:
                for actor, (p, q) in zip(obj_actors, settled_obj_poses):
                    actor.set_pose(sapien.Pose(p.tolist(), q.tolist()))

            # Warmup (与第一趟一致)
            self.logger.info(f"  Warmup ({WARMUP_FRAMES} 帧 smoothstep 过渡) ...")
            first_target = qpos_log[0]
            for w in range(WARMUP_FRAMES):
                t = (w + 1) / WARMUP_FRAMES
                t_smooth = t * t * (3 - 2 * t)
                interp_arm = np.array(RIGHT_ARM_STARTING) * (1 - t_smooth) + \
                             np.array([first_target[idx] for idx in arm_joint_indices]) * t_smooth
                self._physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                   interp_arm, 0.04, 0.04)
            self.scene.update_render()

            # 第二趟渲染: PD驱动跟踪第一趟轨迹
            self.logger.info("  物理PD驱动渲染 (跟踪第一趟运动学轨迹) ...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer2 = cv2.VideoWriter(self.output, fourcc, render_fps,
                                      (camera.get_width(), camera.get_height()))
            qpos_log2 = []
            kp_nodes2 = []

            for local_idx in trange(num_frames, desc="第二趟物理渲染", disable=False):
                global_idx = start_frame + local_idx

                # 相机跟随HaWoR轨迹 (与第一趟一致)
                if R_c2w_all is not None and t_c2w_all is not None:
                    cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

                # 重放base tracking (与第一趟一致)
                if local_idx < len(base_pose_log):
                    bp, bq = base_pose_log[local_idx]
                    robot.set_root_pose(sapien.Pose(bp.tolist(), bq.tolist()))

                # 渲染手部关键点 (与第一趟一致)
                if hawor_data["pred_valid"][global_idx]:
                    _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                               hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                    joints_sapien = self._render_to_sapien(j)
                    kp_nodes2 = self._render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes2,
                                                      radius=0.006, ref_indices=set(ref_indices))
                    wrist_pos_sapien = joints_sapien[0, :3].copy()

                # PD驱动: 使用第一趟保存的轨迹作为目标
                if local_idx < len(qpos_log):
                    target_qpos = qpos_log[local_idx]
                    arm_target = np.array([target_qpos[idx] for idx in arm_joint_indices])
                    gripper_t1 = float(target_qpos[gripper_idx1]) if gripper_idx1 < len(target_qpos) else 0.04
                    gripper_t1 = max(0.0, min(0.05, gripper_t1))  # clamp 到 URDF limit
                    gripper_t2 = gripper_t1  # 同号, axis 相反, 对称开合
                    self._physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                       arm_target, gripper_t1, gripper_t2)
                else:
                    # 无效帧: 重力补偿 + decimation
                    for _ in range(DECIMATION):
                        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                        robot.set_qf(qf)
                        self.scene.step()
                qpos_log2.append(robot.get_qpos().copy())

                self.scene.update_render()
                n_contacts, total_impulse, per_obj = self._fetch_contacts(robot, obj_actors)

                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                ee_err_cm = None
                if gripper_link is not None and wrist_pos_sapien is not None:
                    ee_pos = np.array(gripper_link.get_entity_pose().p)
                    ee_err_cm = np.linalg.norm(ee_pos - wrist_pos_sapien) * 100
                label = f"Frame {local_idx+1}/{num_frames}  |  Physics [Two-Pass]"
                if ee_err_cm is not None:
                    err_color = (0, 255, 0) if ee_err_cm < 2 else (0, 255, 255) if ee_err_cm < 5 else (0, 0, 255)
                    label += f"  EE:{ee_err_cm:.1f}cm"
                label += f"  C:{n_contacts}"
                cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            err_color if ee_err_cm is not None else (255, 255, 255), 2)
                for _ in range(frame_repeat):
                    writer2.write(bgr)

                if (local_idx + 1) % 30 == 0:
                    self.logger.info(f"  帧 {local_idx+1}/{num_frames}: contacts={n_contacts}, impulse={total_impulse:.2f}")

            writer2.release()
            for node in kp_nodes2:
                internal_scene.remove_node(node)

            self.logger.info(f"  ✓ 第二趟物理渲染完成, qpos_log2: {len(qpos_log2)} 帧")
            qpos_log = qpos_log2

        if not self.viewer and self.smooth == 2 and qpos_log:
            self.logger.info("\n  === 第二趟: TrajectorySmoother 后处理 + 重新渲染 ===")
            qpos_arr = np.array(qpos_log)
            self.logger.info(f"  原始qpos帧间: 平均 {np.degrees(np.abs(np.diff(qpos_arr[:, :6], axis=0))).mean():.2f}°, "
                             f"最大 {np.degrees(np.abs(np.diff(qpos_arr[:, :6], axis=0))).max():.2f}°")
            smoother = TrajectorySmoother(fps=render_fps)
            smooth_indices = list(range(len(arm_joint_indices)))
            smoothed_qpos_list, metrics = smoother.smooth_trajectory(qpos_log, smooth_indices)
            self.logger.info(f"  平滑后qpos帧间: 平均 {np.degrees(np.abs(np.diff(np.array(smoothed_qpos_list)[:, :6], axis=0))).mean():.2f}°, "
                             f"最大 {np.degrees(np.abs(np.diff(np.array(smoothed_qpos_list)[:, :6], axis=0))).max():.2f}°")
            self.logger.info(f"  物理指标: max_vel={metrics['smooth_max_velocity']:.3f} rad/s, "
                             f"max_acc={metrics['smooth_max_acceleration']:.3f} rad/s², "
                             f"max_jerk={metrics['smooth_max_jerk']:.3f} rad/s³")

            self.logger.info("  重新初始化物理状态 (Reset to RIGHT_ARM_STARTING) ...")
            reset_qpos = robot.get_qpos().copy()
            for j, idx in enumerate(arm_joint_indices):
                if j < len(RIGHT_ARM_STARTING):
                    reset_qpos[idx] = RIGHT_ARM_STARTING[j]
            reset_qpos[gripper_idx1] = 0.04
            reset_qpos[gripper_idx2] = 0.04
            robot.set_qpos(reset_qpos)

            initial_obj_poses = settled_obj_poses

            for w in range(WARMUP_FRAMES):
                t = (w + 1) / WARMUP_FRAMES
                t_smooth = t * t * (3 - 2 * t)
                if smoothed_qpos_list:
                    first_smoothed = smoothed_qpos_list[0]
                    if first_smoothed is not None:
                        interp = np.array(RIGHT_ARM_STARTING) * (1 - t_smooth) + \
                                 np.array([first_smoothed[idx] for idx in arm_joint_indices]) * t_smooth
                    else:
                        interp = np.array(RIGHT_ARM_STARTING)
                else:
                    interp = np.array(RIGHT_ARM_STARTING)
                self._physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                   interp, 0.04, 0.04)
            self.scene.update_render()

            self.logger.info("  用平滑后的qpos重新渲染 (kinematic模式, 与02_render_scene.py一致) ...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer2 = cv2.VideoWriter(self.output, fourcc, render_fps,
                                      (camera.get_width(), camera.get_height()))
            qpos_log2 = []
            kp_nodes2 = []
            for local_idx in trange(num_frames, desc="第二趟渲染", disable=False):
                global_idx = start_frame + local_idx

                if R_c2w_all is not None and t_c2w_all is not None:
                    cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

                if obj_actors and initial_obj_poses:
                    for actor, (p, q) in zip(obj_actors, initial_obj_poses):
                        actor.set_pose(sapien.Pose(p.tolist(), q.tolist()))

                smoothed_qpos = smoothed_qpos_list[local_idx] if local_idx < len(smoothed_qpos_list) else None
                if smoothed_qpos is not None:
                    # 重放base tracking: 与第一趟一致，必须set_root_pose + scene.step
                    if local_idx < len(base_pose_log):
                        bp, bq = base_pose_log[local_idx]
                        robot.set_root_pose(sapien.Pose(bp.tolist(), bq.tolist()))
                        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                        robot.set_qf(qf)
                        self.scene.step()

                    if hawor_data["pred_valid"][global_idx]:
                        _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                                   hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                        joints_sapien = self._render_to_sapien(j)
                        kp_nodes2 = self._render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes2,
                                                          radius=0.006, ref_indices=set(ref_indices))

                    # 纯PD驱动: 设置drive_target，不用set_qpos
                    if smoothed_qpos is not None:
                        active_joints = robot.get_active_joints()
                        for j_idx, arm_idx in enumerate(arm_joint_indices):
                            active_joints[arm_idx].set_drive_target(float(smoothed_qpos[arm_idx]))
                        if gripper_idx1 < len(smoothed_qpos):
                            active_joints[gripper_idx1].set_drive_target(float(smoothed_qpos[gripper_idx1]))
                        if gripper_idx2 < len(smoothed_qpos):
                            active_joints[gripper_idx2].set_drive_target(float(smoothed_qpos[gripper_idx2]))
                    qpos_log2.append(robot.get_qpos().copy())
                else:
                    qpos_log2.append(robot.get_qpos().copy())

                # decimation次物理子步
                for _ in range(DECIMATION):
                    qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                    robot.set_qf(qf)
                    self.scene.step()
                self.scene.update_render()
                camera.take_picture()
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Physics Sim [Smoothed]", (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for _ in range(frame_repeat):
                    writer2.write(bgr)

            writer2.release()
            for node in kp_nodes2:
                internal_scene.remove_node(node)

            self.logger.info(f"  ✓ 第二趟渲染完成, qpos_log2: {len(qpos_log2)} 帧")
            qpos_log = qpos_log2

        if not self.viewer:
            final_path = self.output
            tmp_path = str(self.output).replace(".mp4", "_tmp.mp4")
            if os.path.exists(str(self.output)):
                os.rename(str(self.output), tmp_path)
                if reencode_with_ffmpeg(tmp_path, final_path, crf=self.crf, fps=render_fps, logger=self.logger):
                    pass
                else:
                    if os.path.exists(tmp_path):
                        os.rename(tmp_path, final_path)

        qpos_path = str(Path(self.output).with_suffix(".npy")).replace("videos", "tracking")
        os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
        if qpos_log:
            np.save(qpos_path, np.array(qpos_log))
            self.logger.info(f"  ✓ qpos 已保存: {qpos_path} ({len(qpos_log)} 帧)")

        self.logger.info(f"\n✓ 物理仿真视频已保存: {self.output if self.viewer else final_path}")


def main():
    """命令行入口: 物理仿真驱动 — 真实抓取与交互

    用法示例:
      python 04_physics_simulation.py --hawor-dir /path/to/hawor --ras-dir /path/to/ras
      python 04_physics_simulation.py --hawor-dir /path/to/hawor --ras-dir /path/to/ras --fast-collision
      python 04_physics_simulation.py --hawor-dir /path/to/hawor --ras-dir /path/to/ras --hide-hand --speed 0.3
    """
    import argparse
    parser = argparse.ArgumentParser(description="物理仿真驱动：真实抓取与交互")
    parser.add_argument("--mode", type=str, default="physics_tracking",
                        choices=["physics_tracking"],
                        help="仿真模式")
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="HaWoR 重建输出目录 (包含 reconstruction/ 子目录或 world_space_res.pth)")
    parser.add_argument("--ras-dir", type=str, required=True,
                        help="RAS 场景重建输出目录 (包含 final_scene.glb)")
    parser.add_argument("--transform-params", type=str, default=None,
                        help="01_align_scene.py 输出的 transform_params.npz 路径 (默认: 自动推导或自动生成)")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="手的索引: 0=左手, 1=右手, -1=自动检测")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920, help="渲染宽度 (像素)")
    parser.add_argument("--height", type=int, default=1080, help="渲染高度 (像素)")
    parser.add_argument("--crf", type=int, default=14,
                        help="H.264 编码质量 (0=无损, 14=高质量(默认), 18=较好, 23=默认, 28=低质量)")
    parser.add_argument("--viewer", action="store_true", help="交互式Viewer渲染（不保存视频）")
    parser.add_argument("--fast-collision", action="store_true",
                        help="使用快速凸包碰撞体代替 CoACD (速度快但精度低, 推荐调试时使用)")
    parser.add_argument("--hide-hand", action="store_true",
                        help="隐藏手部mesh和骨架，只显示3个跟随点")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="播放速度倍率 (0.5=半速慢放, 1.0=原始速度, 默认1.0保证视频帧数=HaWoR帧数)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="平滑模式: 0=不平滑, 1=在线EMA(默认), 2=后处理双向滤波")
    parser.add_argument("--two-pass", action="store_true",
                        help="两趟渲染: 第一趟运动学(set_qpos, 确定性), 第二趟物理(PD驱动+重力补偿)")
    parser.add_argument("--no-support-table", action="store_true",
                        help="禁用可见桌面支撑 (仅使用物理地面)")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角: fpv=第一人称(跟随HaWoR相机轨迹), topdown=俯视, behind=后方, front=前方")
    parser.add_argument("--single-gripper", action="store_true",
                        help="单夹爪模式: 只加载夹爪URDF(无机械臂), 直接用MANO手腕位姿驱动夹爪, 参考 hand_track/render_gripper_only.py")
    parser.add_argument("--base-cluster", action="store_true",
                        help="分段固定基座模式: 将轨迹按XY空间聚成N段, 基座间smoothstep过渡 (替代浮动基座)")
    parser.add_argument("--fixed-base", action="store_true", default=True,
                        help="固定基座模式 (默认开启): 基座不跟随手腕移动")
    parser.add_argument("--no-fixed-base", action="store_true",
                        help="禁用固定基座, 使用浮动基座 (XY跟踪范围±4cm)")
    args = parser.parse_args()

    # --no-fixed-base 覆盖 --fixed-base
    if args.no_fixed_base:
        args.fixed_base = False

    if args.output is None:
        # 输出到 physics_pipeline/output 目录 (与 PyBullet 管线统一)
        # 文件名包含关键参数, 避免不同命令的输出互相覆盖
        name_parts = [f"physics_sim_{args.mode}"]
        if args.single_gripper:
            name_parts.append("gripper")
        if args.view != "fpv":
            name_parts.append(args.view)
        if args.hand_idx >= 0:
            name_parts.append(f"h{args.hand_idx}")
        if args.base_cluster:
            name_parts.append("cluster")
        elif args.no_fixed_base:
            name_parts.append("float")
        if args.two_pass:
            name_parts.append("2pass")
        physics_output = Path(__file__).parent / "physics_pipeline" / "output"
        args.output = str(physics_output / ("_".join(name_parts) + ".mp4"))
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.hand_idx < 0:
        detected_idx, handedness = _detect_hand_idx(Path(args.hawor_dir))
        args.hand_idx = detected_idx
        # handedness: "left"/"right"/"both"/"unknown"
        # 注意: 04 仅支持单臂渲染 (与 02 一致), 双手场景默认取 idx=0 (左手) 由调用方按需切换
        label_map = {"left": "左手", "right": "右手", "both": "双手(默认左手)", "unknown": "未知(默认左手)"}
        hand_label = label_map.get(handedness, "未知(默认左手)")
        print(f"自动检测到手: {hand_label} (idx={detected_idx}, handedness={handedness})")

    # 自动推导 transform-params (先查找已有文件, 找不到则调用 01 生成)
    if args.transform_params is None:
        hawor_bn = Path(args.hawor_dir).name
        ras_bn = Path(args.ras_dir).name
        session_name = f"{hawor_bn}_{ras_bn}"
        output_dir = Path(__file__).parent / "output" / session_name

        # 1. 先查找已有文件
        candidates = [
            output_dir / "alignment" / "transform_params.npz",
            Path(__file__).parent / "output" / "alignment" / "transform_params.npz",
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
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "align_scene_01", str(Path(__file__).parent / "01_align_scene.py")
                )
                mod01 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod01)
                # 查找 hawor reconstruction npz
                hawor_rec_dir = Path(args.hawor_dir) / "reconstruction"
                hawor_npz = None
                if hawor_rec_dir.exists():
                    for f in hawor_rec_dir.glob("hawor_results_*.npz"):
                        hawor_npz = str(f)
                        break
                if hawor_npz is None:
                    raise FileNotFoundError(f"未找到 HaWoR reconstruction npz: {hawor_rec_dir}/hawor_results_*.npz")
                alignment_dir = str(output_dir / "alignment")
                args.transform_params = mod01.compute_and_save_transform_params(
                    ras_output=str(args.ras_dir),
                    hawor_reconstruction=hawor_npz,
                    output_dir=alignment_dir,
                )
                print(f"  ✓ 自动生成 transform-params: {args.transform_params}")
            except Exception as e:
                raise FileNotFoundError(
                    f"自动生成 transform-params 失败: {e}\n"
                    f"请先运行: python 01_align_scene.py --ras_output ... --hawor_reconstruction ...\n"
                    f"或用 --transform-params 手动指定"
                )

    if not Path(args.transform_params).exists():
        raise FileNotFoundError(
            f"未找到变换参数文件: {args.transform_params}\n"
            f"请先运行: python 01_align_scene.py --ras_output ... --hawor_reconstruction ..."
        )

    logger = logging.getLogger("PhysicsSim")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

    sim = PhysicsSimulator(hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                           transform_params_path=args.transform_params,
                           output=args.output, fps=args.fps, hand_idx=args.hand_idx,
                           logger=logger, viewer=args.viewer, crf=args.crf,
                           fast_collision=args.fast_collision,
                           hide_hand=args.hide_hand, speed=args.speed,
                           cam_width=args.width, cam_height=args.height,
                           smooth=args.smooth,
                           two_pass=args.two_pass,
                           support_table=not args.no_support_table,
                           view=args.view,
                           single_gripper=args.single_gripper,
                           base_cluster=args.base_cluster,
                           fixed_base=args.fixed_base)
    sim.run_physics_tracking(start_frame=args.start_frame, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
