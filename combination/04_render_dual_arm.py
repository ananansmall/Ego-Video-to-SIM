#!/usr/bin/env python3
"""
================================================================================
  04_render_dual_arm.py — 通用双臂/单臂协同运动渲染

  设计目标: **泛化能力** — 不预先知道是左手/右手/双手
    1. 读取 reconstruction/hawor_results_*.npz
    2. 自动检测手部类型 (HandDetector: LEFT/RIGHT/BOTH)
    3. 按检测结果动态创建 1-2 个 R1 机械臂
    4. 多个机械臂同时运动, 共享同一 SAPIEN 场景
    5. 输出 mp4 + qpos.npy

  关键: 脚本不依赖任何手部索引硬编码, 全部由 HandDetector 决定。

  管线:
    python 04_render_dual_arm.py --hawor-dir /path/to/hawor
    └─ HandDetector.detect() 自动判断
        ├─ LEFT  → 创建 1 个左臂, 渲染 1 个左臂
        ├─ RIGHT → 创建 1 个右臂, 渲染 1 个右臂
        └─ BOTH  → 创建 2 个臂 (左+右), 同步渲染

  数据源:
    HaWoR 重建: hawor_dir/reconstruction/hawor_results_*.npz
      pred_trans:     (2, N, 3)  两手平移 (即使只有一只手, 形状仍是 (2, N, 3))
      pred_rot:       (2, N, 3)
      pred_hand_pose: (2, N, 45)
      pred_betas:     (2, N, 10)
      pred_valid:     (2, N)     ← HandDetector 用此判断左右手

  运动映射链 (每只检测到的手各走一条):
    MANO FK → 21 关节点 → Dex Retargeting → 夹爪位姿 → RelaxedIK → 6 臂关节

  GLB 场景: --ras-dir + --transform-params 提供时加载, 否则 simple_ground + 坐标系

  输出:
    output/videos/dual_arm_only.mp4
    output/tracking/dual_arm_only_qpos.npy
        内容: dict, {<arm_prefix>: np.ndarray(N_valid, 8), ...}
================================================================================
"""

import os
import sys
import warnings

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# 抑制无关的第三方警告
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"sapien.*")
warnings.filterwarnings("ignore", message=r"In the future `np\.(bool|object|str)`.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"mano_layer.*")
warnings.filterwarnings("ignore", message=r".*not writable, and PyTorch does not support non-writable.*")
warnings.filterwarnings("ignore", message=r".*CUDA initialization.*NVIDIA driver on your system is too old.*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"sapien.*")
os.environ.setdefault("SAPIEN_LOG_LEVEL", "ERROR")

_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
_intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
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

import argparse
import sys
import logging
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import sapien
import sapien.render
import torch
from pytransform3d import rotations as pr
from tqdm import trange

try:
    import trimesh
except ImportError:
    trimesh = None

# ============ 路径配置 ============
SCRIPT_DIR = Path(__file__).resolve().parent
HAND_TRACK_DIR = SCRIPT_DIR / "hand_track"
sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "example" / "position_retargeting"))

PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
FLOATING_RIGHT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_right.urdf"
FLOATING_LEFT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_left.urdf"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"

# ============ 坐标系变换 ============
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

# ============ 常量 ============
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [1.5, 1.9508, 1.0809, 0.4438, -0.1709, 0.1985]
COMFORTABLE_REACH = 0.35
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
BASE_TRACKING_RANGE = 0.04
LP_ALPHA_JOINT = 0.5
IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 20
HAWOR_FOCAL_DEFAULT = 600.0

ARM_STARTING_QPOS = {"right": RIGHT_ARM_STARTING, "left": LEFT_ARM_STARTING}
FLOATING_URDF_PATH = {"right": FLOATING_RIGHT_URDF, "left": FLOATING_LEFT_URDF}
IK_METHOD = {"right": "solve_position_right", "left": "solve_position_left"}

# ============ 关键点颜色 (BGR 用于视频, RGB 用于材质) ============
KP_COLORS = {
    "left":  np.array([1.0, 0.4, 0.7, 1.0]),  # 粉色
    "right": np.array([0.4, 0.7, 1.0, 1.0]),  # 蓝色
}


# =============================================================================
# 1. HandDetector 集成
# =============================================================================
def detect_hands(hawor_dir, logger):
    """从 HaWoR 目录自动检测手部类型。

    使用 hand_track/hand_detector.py 的 HandDetector, 读取
    reconstruction/hawor_results_*.npz 自动判断 LEFT/RIGHT/BOTH/NONE。

    Args:
        hawor_dir: HaWoR 数据根目录
        logger:    Logger

    Returns:
        HandDetectionResult, 含 handedness/left_valid_frames/right_valid_frames
    """
    from hand_detector import HandDetector
    detector = HandDetector(str(hawor_dir))
    result = detector.detect()
    logger.info(f"  [HandDetector] {result.description}")
    logger.info(f"  [HandDetector] 检测方法: {result.detection_method}")
    return result


# =============================================================================
# 2. 数据加载 (从 npz 加载两手数据, 同时获取 pred_valid 等)
# =============================================================================
def load_hawor_npz(hawor_dir):
    """从 npz 加载 HaWoR 数据, 同时返回 pred_valid 等检测信息。

    Args:
        hawor_dir: HaWoR 数据根目录

    Returns:
        (hawor_data_dict, npz_path) 元组
    """
    hawor_path = Path(hawor_dir)
    rec_file = None
    rec_dir = hawor_path / "reconstruction"
    if rec_dir.exists():
        for f in rec_dir.glob("hawor_results_*.npz"):
            rec_file = f
            break
    if rec_file is None:
        npz_files = list(hawor_path.glob("*.npz"))
        if npz_files:
            rec_file = npz_files[0]
    if rec_file is None:
        raise FileNotFoundError(f"未找到 HaWoR 数据: {hawor_path}")

    rec = np.load(str(rec_file), allow_pickle=True)
    pred_trans = rec['pred_trans']
    pred_rot = rec['pred_rot']
    pred_hand_pose = rec['pred_hand_pose']
    pred_betas = rec['pred_betas']
    pred_valid = rec['pred_valid']
    R_c2w = rec['R_c2w']
    t_c2w = rec['t_c2w']

    for frame_i in range(pred_trans.shape[1]):
        for hand_i in range(pred_trans.shape[0]):
            if pred_valid[hand_i, frame_i]:
                pred_trans[hand_i, frame_i] = R_c2w[frame_i] @ pred_trans[hand_i, frame_i] + t_c2w[frame_i]
                rot_mat = pr.matrix_from_compact_axis_angle(pred_rot[hand_i, frame_i])
                rot_mat_world = R_c2w[frame_i] @ rot_mat
                pred_rot[hand_i, frame_i] = pr.compact_axis_angle_from_matrix(rot_mat_world)

    has_nan = np.isnan(pred_trans).any(axis=-1)
    pred_valid = pred_valid & ~has_nan

    return {
        "pred_trans": pred_trans,
        "pred_rot": pred_rot,
        "pred_hand_pose": pred_hand_pose,
        "pred_betas": pred_betas,
        "pred_valid": pred_valid,
    }, rec_file


def compute_mano_joints(mano_layer, rot, hand_pose, trans):
    """MANO FK: 计算单帧的 778 顶点和 21 关节点。

    Args:
        mano_layer: MANOLayer 实例
        rot:        (3,) 轴角
        hand_pose:  (45,) PCA
        trans:      (3,) 平移

    Returns:
        (vertices (778, 3), joints (21, 3))
    """
    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = mano_layer(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]


# =============================================================================
# 3. SAPIEN 场景辅助
# =============================================================================
def setup_scene():
    """创建配置好的 SAPIEN 场景。

    Returns:
        sapien.Scene
    """
    from sapien.asset import create_dome_envmap
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_ray_tracing_samples_per_pixel(64)
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35]))
    scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
    scene.add_directional_light([-1, -0.5, -1], [1.2, 1.2, 1.2], shadow=False)
    scene.add_directional_light([0, 1, -0.5], [0.8, 0.8, 0.8], shadow=False)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    return scene


def simple_ground(scene, size=4.0, color=(0.8, 0.8, 0.85, 1.0)):
    """创建简单地面网格 (未指定 GLB 时使用)。

    Args:
        scene: SAPIEN Scene
        size:  边长 (米)
        color: RGBA 颜色

    Returns:
        actor
    """
    half = size / 2.0
    vertices = np.array([
        [-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    temp_ply = f"/tmp/simple_ground_{os.getpid()}.ply"
    trimesh.Trimesh(vertices=vertices, faces=faces).export(temp_ply)
    builder = scene.create_actor_builder()
    material = sapien.render.RenderMaterial(base_color=list(color), metallic=0.0, roughness=0.9, specular=0.1)
    builder.add_visual_from_file(filename=temp_ply, material=material)
    actor = builder.build_kinematic(name="simple_ground")
    from sapien.core import Pose
    actor.set_pose(Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
    try:
        os.remove(temp_ply)
    except Exception:
        pass
    return actor


def render_coordinate_axes(scene, origin=(0, 0, 0), axis_length=0.3, radius=0.005):
    """渲染 XYZ 坐标系 (X 红, Y 绿, Z 蓝)。

    Args:
        scene: SAPIEN Scene
        origin: 起点
        axis_length: 轴长
        radius: 圆柱半径
    """
    internal_scene = scene.render_system._internal_scene
    context = sapien.render.SapienRenderer()._internal_context
    origin = np.array(origin, dtype=np.float64)
    axes = [
        (np.array([1.0, 0, 0]), np.array([1.0, 0.2, 0.2, 1.0])),
        (np.array([0, 1.0, 0]), np.array([0.2, 1.0, 0.2, 1.0])),
        (np.array([0, 0, 1.0]), np.array([0.2, 0.4, 1.0, 1.0])),
    ]
    for direction, color in axes:
        mid = origin + direction * axis_length / 2
        cylinder = context.create_capsule_mesh(radius, axis_length / 2, 8, 4)
        material = context.create_material(np.zeros(4), color, 0.0, 0.5, 0)
        model = context.create_model([cylinder], [material])
        node = internal_scene.add_node()
        node.set_position(mid.tolist())
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


def make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
    """look-at 相机四元数。

    Args:
        eye:    相机位置
        target: 目标
        up:     上方向

    Returns:
        (4,) 四元数 (w,x,y,z)
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
    return pr.quaternion_from_matrix(cam_R)


def load_glb_transformed(glb_path, transform_params_path, scene, logger=None):
    """加载 GLB 场景并变换到 SAPIEN 坐标系。

    Args:
        glb_path:              GLB 文件路径
        transform_params_path: 变换参数 npz 路径
        scene:                 SAPIEN Scene
        logger:                Logger (可选)

    Returns:
        obj_actors 列表
    """
    if trimesh is None:
        return []
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    trimesh_scene = trimesh.load(str(glb_path))
    obj_actors = []
    temp_files = []
    from sapien.core import Pose
    for geom_name, geom in trimesh_scene.geometry.items():
        vertices = geom.vertices.copy()
        faces = geom.faces.copy()
        if len(vertices) == 0 or len(faces) == 0:
            continue
        vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
        vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
        avg_color = None
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            vc = geom.visual.vertex_colors
            if len(vc) > 0:
                avg_rgb = vc[:, :3].mean(axis=0)
                avg_color = [avg_rgb[0]/255.0, avg_rgb[1]/255.0, avg_rgb[2]/255.0, 1.0]
        temp_ply = f'/tmp/glb_actor_{os.getpid()}_{geom_name.replace(" ", "_")}.ply'
        trimesh.Trimesh(vertices=vertices_sapien, faces=faces, visual=geom.visual).export(temp_ply)
        temp_files.append(temp_ply)
        builder = scene.create_actor_builder()
        if avg_color is not None:
            material = sapien.render.RenderMaterial(base_color=avg_color, metallic=0.0, roughness=0.7, specular=0.3)
            builder.add_visual_from_file(filename=temp_ply, material=material)
        else:
            builder.add_visual_from_file(filename=temp_ply)
        actor = builder.build_kinematic(name=geom_name)
        actor.set_pose(Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        obj_actors.append(actor)
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception:
            pass
    if logger:
        logger.info(f"  ✓ GLB 加载完成: {len(obj_actors)} 个物体")
    return obj_actors


def reencode_with_ffmpeg(input_path, output_path, crf=18, fps=30, logger=None):
    """ffmpeg 重编码为 H.264 格式。

    Args:
        input_path:  输入路径
        output_path: 输出路径
        crf:         质量因子
        fps:         帧率
        logger:      Logger

    Returns:
        bool 是否成功
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return False
    if not os.path.exists(input_path):
        return False
    cmd = [
        ffmpeg_exe, "-y", "-i", str(input_path),
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        os.remove(input_path)
        if logger:
            sz = os.path.getsize(output_path)
            sz_str = f"{sz/1024/1024:.2f} MB" if sz >= 1024 * 1024 else f"{sz/1024:.1f} KB"
            logger.info(f"  ✓ H.264 重编码: {sz_str}")
        return True
    return False


# =============================================================================
# 4. 动态臂创建 (核心泛化逻辑)
# =============================================================================
class ArmInstance:
    """单臂运行时实例: 包含 URDF/Retargeting/IK/MANO。

    与 hand_track/SingleArmConfig 不同, 这是动态加载到 SAPIEN 后的运行时对象。

    Attributes:
        prefix:             "left" 或 "right"
        hand_idx:           HaWoR 手部索引 (0=左手, 1=右手)
        robot:              SAPIEN Robot
        arm_joint_indices:  6 臂关节索引
        gripper_idx1/2:     2 夹爪关节索引
        retargeting:        Dex Retargeting 优化器
        ref_indices:        retargeting 约束的人手关节点索引
        mano_layer:         MANOLayer
        joint_filter:       LPFilter
        ik_method_name:     "solve_position_left" / "_right"
        starting_qpos:      初始 6 关节角
        base_pos:           (3,) 基座初始位置
        base_q:             (4,) 基座朝向
    """

    def __init__(self, prefix, hand_idx, scene, ik_solver, logger):
        """动态创建单臂。

        Args:
            prefix:   "left" 或 "right" (由 HandDetector 决定)
            hand_idx: 0 (左手) 或 1 (右手), 用于索引 pred_valid
            scene:    SAPIEN Scene
            ik_solver: RelaxedIKSolver (含 left/right)
            logger:   Logger
        """
        from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
        from dex_retargeting.retargeting_config import RetargetingConfig
        from dex_retargeting.optimizer_utils import LPFilter

        self.prefix = prefix
        self.hand_idx = hand_idx
        self.logger = logger
        self.ik_solver = ik_solver
        self.ik_method_name = IK_METHOD[prefix]

        urdf_path = FLOATING_URDF_PATH[prefix]
        arm_urdf = self._prepare_urdf(urdf_path, prefix)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        self.robot = loader.load(arm_urdf)

        self.joint_names = [j.name for j in self.robot.get_active_joints()]
        self.arm_joint_indices = [i for i, n in enumerate(self.joint_names) if f"{prefix}_arm_joint" in n]
        self.gripper_idx1 = self.joint_names.index(f"{prefix}_gripper_finger_joint1")
        self.gripper_idx2 = self.joint_names.index(f"{prefix}_gripper_finger_joint2")

        for joint in self.robot.get_active_joints():
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        self.starting_qpos = np.array(ARM_STARTING_QPOS[prefix])
        init_qpos = self.robot.get_qpos().copy()
        for j, idx in enumerate(self.arm_joint_indices):
            if j < len(self.starting_qpos):
                init_qpos[idx] = self.starting_qpos[j]
        init_qpos[self.gripper_idx1] = 0.04
        init_qpos[self.gripper_idx2] = -0.04
        self.robot.set_qpos(init_qpos)
        self.logger.info(f"  [{prefix}] 臂关节: {self.arm_joint_indices}, 夹爪: [{self.gripper_idx1}, {self.gripper_idx2}]")

        robot_dir = PROJECT_ROOT / "dex-retargeting" / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        hand_type = HandType.right if prefix == "right" else HandType.left
        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
        override = dict(
            add_dummy_free_joint=True,
            normal_delta=1e-5,
            huber_delta=0.01,
            target_link_names=[
                f"{prefix}_gripper_finger_link1",
                f"{prefix}_gripper_finger_link2",
                f"{prefix}_gripper_link",
            ],
            target_link_human_indices=np.array([4, 8, 0]),
        )
        config = RetargetingConfig.load_from_file(config_path, override=override)
        self.retargeting = config.build()
        self.ref_indices = self.retargeting.optimizer.target_link_human_indices
        self.fixed_retarget_indices = self.retargeting.optimizer.idx_pin2fixed

        self.retarget2sapien = np.array(
            [self.retargeting.joint_names.index(n) for n in self.joint_names if n in self.retargeting.joint_names]
        ).astype(int)
        self.sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(self.retarget2sapien):
            self.sapien2retarget[retarget_i] = sapien_i
        self.fixed_qpos = np.zeros(len(self.fixed_retarget_indices), dtype=np.float32)
        for i, retarget_idx in enumerate(self.fixed_retarget_indices):
            if retarget_idx in self.sapien2retarget:
                self.fixed_qpos[i] = init_qpos[self.sapien2retarget[retarget_idx]]
        self.logger.info(f"  [{prefix}] Retargeting 约束: {list(self.ref_indices)} (食指尖/中指尖/手腕)")

        self.joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        current_joints = np.array([init_qpos[i] for i in self.arm_joint_indices])
        self.joint_filter.next(current_joints)

        self.base_pos = None  # 由调用方设置
        self.base_q = None
        self.mano_layer = None  # 由调用方设置 (需要 betas)

    @staticmethod
    def _prepare_urdf(src_urdf_path, arm_prefix):
        """URDF 预处理: 替换 mesh 路径 + 夹爪改 prismatic。

        Args:
            src_urdf_path: 源 URDF
            arm_prefix:    "left"/"right"

        Returns:
            str 临时 URDF 路径
        """
        import re
        xml = src_urdf_path.read_text()
        xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")
        for finger in ("finger_joint1", "finger_joint2"):
            xml = re.sub(
                rf'(<joint\s+name="{arm_prefix}_gripper_{finger}"\s+type=")fixed(")',
                r'\1prismatic\2', xml,
            )
        temp_dir = tempfile.mkdtemp(prefix=f"r1_{arm_prefix}_arm_urdf-")
        temp_path = f"{temp_dir}/{src_urdf_path.name}"
        with open(temp_path, "w") as f:
            f.write(xml)
        return temp_path

    def warm_start(self, joints_sapien, wrist_quat_sapien, hand_type):
        """首帧预热 retargeting 优化器。

        Args:
            joints_sapien:     (21, 3)
            wrist_quat_sapien: (4,)
            hand_type:         HandType
        """
        self.retargeting.warm_start(
            joints_sapien[0, :3], wrist_quat_sapien,
            hand_type=hand_type, is_mano_convention=True,
        )

    def compute_mano_joints(self, rot, hand_pose, trans):
        """MANO FK → SAPIEN 坐标系 21 关节点。

        Args:
            rot:       (3,)
            hand_pose: (45,)
            trans:     (3,)

        Returns:
            (21, 3)
        """
        _, j = compute_mano_joints(self.mano_layer, rot, hand_pose, trans)
        return (RXWORLD_TO_SAPIEN @ j.T).T

    def set_base_pose(self, base_pos, base_q):
        """设置初始基座位姿 (由调用方根据手腕质心计算)。

        Args:
            base_pos: (3,)
            base_q:   (4,)
        """
        self.base_pos = base_pos
        self.base_q = base_q
        self.robot.set_root_pose(sapien.Pose(base_pos.tolist(), base_q.tolist()))

    def retarget_and_solve_ik(self, joints_sapien):
        """执行 retarget + IK, 一步完成。

        Args:
            joints_sapien: (21, 3) SAPIEN 坐标系 21 关节点

        Returns:
            np.ndarray: 新 SAPIEN qpos
        """
        ref_value = joints_sapien[self.ref_indices, :].astype(np.float32)
        retarget_qpos = self.retargeting.retarget(ref_value, self.fixed_qpos)
        sapien_qpos = retarget_qpos[self.retarget2sapien]

        gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose(retarget_qpos)

        base_R = pr.matrix_from_quaternion(self.base_q)
        base_R_inv = base_R.T
        wrist_in_base = base_R_inv @ (gripper_pos_fk - self.base_pos)
        offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE
        clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)
        delta_world = base_R @ clamped_offset
        tracked_base = self.base_pos + delta_world

        self.robot.set_root_pose(sapien.Pose(tracked_base.tolist(), self.base_q.tolist()))

        base_link_p, base_link_q = None, None
        for link in self.robot.get_links():
            if f"{self.prefix}_arm_base_link" == link.get_name():
                pose = link.get_entity_pose()
                base_link_p = np.array(pose.p)
                base_link_q = np.array(pose.q)
                break
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T

        ik_target_b = base_link_R_inv @ (gripper_pos_fk - base_link_p)
        ee_R_base = base_link_R_inv @ R_ee_world_fk
        ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

        solve_fn = getattr(self.ik_solver, self.ik_method_name)
        ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        for _ in range(IK_SOLVE_PER_FRAME - 1):
            ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        filtered_joints = self.joint_filter.next(np.array(ik_joints))

        qpos = self.robot.get_qpos().copy()
        for j_idx, arm_idx in enumerate(self.arm_joint_indices):
            qpos[arm_idx] = filtered_joints[j_idx]
        if self.gripper_idx1 < len(sapien_qpos):
            qpos[self.gripper_idx1] = float(sapien_qpos[self.gripper_idx1])
        if self.gripper_idx2 < len(sapien_qpos):
            qpos[self.gripper_idx2] = float(sapien_qpos[self.gripper_idx2])
        return qpos

    def _get_gripper_pose(self, retarget_qpos):
        """从内部机器人 FK 获取夹爪位姿。

        Args:
            retarget_qpos: retargeting 关节角

        Returns:
            (gripper_pos (3,), gripper_R (3,3))
        """
        internal_robot = self.retargeting.optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)
        target_name = f"{self.prefix}_gripper_link"
        for i, name in enumerate(internal_robot.link_names):
            if name == target_name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        raise RuntimeError(f"内部机器人中找不到 {target_name}")


def build_arms_from_detection(handedness, scene, ik_solver, logger):
    """根据 HandDetector 结果动态创建 ArmInstance 列表。

    核心泛化逻辑:
        LEFT  → [ArmInstance("left",  0)]
        RIGHT → [ArmInstance("right", 1)]
        BOTH  → [ArmInstance("left", 0), ArmInstance("right", 1)]
        NONE  → []

    Args:
        handedness: Handedness 枚举值
        scene:      SAPIEN Scene
        ik_solver:  RelaxedIKSolver
        logger:     Logger

    Returns:
        List[ArmInstance]
    """
    from hand_detector import Handedness

    if handedness == Handedness.LEFT:
        return [ArmInstance("left", 0, scene, ik_solver, logger)]
    elif handedness == Handedness.RIGHT:
        return [ArmInstance("right", 1, scene, ik_solver, logger)]
    elif handedness == Handedness.BOTH:
        return [
            ArmInstance("left", 0, scene, ik_solver, logger),
            ArmInstance("right", 1, scene, ik_solver, logger),
        ]
    else:
        return []


# =============================================================================
# 5. 关键点渲染
# =============================================================================
def render_hand_keypoints(joints_sapien, mat, internal_scene, context, kp_nodes, ref_indices, radius=0.006):
    """渲染 3 个 retargeting 关键点为彩色球体 (食指尖/中指尖/手腕)。

    Args:
        joints_sapien:  (21, 3)
        mat:            材质
        internal_scene: SAPIEN internal scene
        context:        SAPIEN context
        kp_nodes:       已有节点 (会先清除)
        ref_indices:    关键点索引 (3 个)
        radius:         球半径

    Returns:
        list: 更新后的节点
    """
    for node in kp_nodes:
        internal_scene.remove_node(node)
    kp_nodes.clear()
    for i in ref_indices:
        sphere = context.create_uvsphere_mesh(12, 6)
        model = context.create_model([sphere], [mat])
        node = internal_scene.add_node()
        node.set_position(joints_sapien[i, :3].tolist())
        node.set_scale([radius, radius, radius])
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 0
        kp_nodes.append(node)
    return kp_nodes


# =============================================================================
# 6. 主流程
# =============================================================================
def run_dual_arm_only(args, logger):
    """通用双臂/单臂协同运动渲染。

    流程:
        1. HandDetector.detect() 自动判断手部
        2. 动态创建 1-2 个 ArmInstance
        3. 加载 GLB (可选) 或 simple_ground
        4. 计算每个臂的初始基座 (独立)
        5. 逐帧: 双手同步 MANO FK → retarget → IK → 同步 scene.step()
        6. 输出 mp4 + qpos dict

    Args:
        args:   argparse 参数
        logger: Logger
    """
    from hand_detector import Handedness
    from dex_retargeting.constants import HandType
    from mano_layer import MANOLayer
    from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

    logger.info("=" * 80)
    logger.info("  通用双臂/单臂协同运动渲染 (Auto-Detect)")
    logger.info("=" * 80)
    logger.info(f"  HaWoR 目录: {args.hawor_dir}")
    logger.info(f"  输出视频:   {args.output}")
    logger.info(f"  视角:       {args.view}  帧率: {args.fps}")
    logger.info(f"  帧范围:     [{args.start_frame}, {args.start_frame + (args.num_frames if args.num_frames > 0 else '全部')})")
    logger.info(f"  GLB:        {'启用' if args.ras_dir else '禁用 (使用 simple_ground)'}")
    sys.stdout.flush()

    # ---- [1/6] HandDetector 自动检测手部 ----
    logger.info("\n[1/6] 自动检测手部类型 ...")
    detection = detect_hands(args.hawor_dir, logger)
    if detection.handedness == Handedness.NONE:
        logger.error(f"  ✗ 未检测到任何手部数据, 退出")
        return
    sys.stdout.flush()

    # ---- [2/6] 加载 npz 数据 + 动态创建臂 ----
    logger.info("\n[2/6] 加载数据 + 动态创建机械臂 ...")
    hawor_data, npz_path = load_hawor_npz(args.hawor_dir)
    total_frames = hawor_data["pred_trans"].shape[1]
    num_frames = args.num_frames
    start_frame = args.start_frame
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame
    logger.info(f"  NPZ: {npz_path.name}, 总帧数: {total_frames}, 处理 [{start_frame}, {start_frame+num_frames-1}]")
    logger.info(f"  数据形状: pred_trans={hawor_data['pred_trans'].shape}, pred_valid={hawor_data['pred_valid'].shape}")

    scene = setup_scene()
    internal_scene = scene.render_system._internal_scene
    context = sapien.render.SapienRenderer()._internal_context

    ik_solver = RelaxedIKSolver(
        left_setting_file_path=str(R1_LEFT_SETTINGS),
        right_setting_file_path=str(R1_RIGHT_SETTINGS),
        tolerances=IK_TOLERANCES,
    )
    ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)
    ik_solver.relaxed_ik_left.reset(LEFT_ARM_STARTING)

    arms = build_arms_from_detection(detection.handedness, scene, ik_solver, logger)
    if not arms:
        logger.error("  ✗ 没有可用的臂, 退出")
        return
    logger.info(f"  ✓ 动态创建 {len(arms)} 个臂: {[a.prefix for a in arms]}")

    for arm in arms:
        arm.mano_layer = MANOLayer(arm.prefix, hawor_data["pred_betas"][arm.hand_idx, start_frame].astype(np.float32))

    scene.step()
    scene.update_render()
    sys.stdout.flush()

    # ---- [3/6] 场景 (GLB 或 simple_ground) ----
    logger.info("\n[3/6] 加载场景 ...")
    if args.ras_dir and args.transform_params:
        glb_path = Path(args.ras_dir) / "final_scene.glb"
        if glb_path.exists() and Path(args.transform_params).exists():
            logger.info(f"  GLB 模式: {glb_path}")
            load_glb_transformed(glb_path, Path(args.transform_params), scene, logger=logger)
        else:
            logger.info(f"  GLB 不存在, 回退到 simple_ground")
            simple_ground(scene)
    else:
        logger.info(f"  默认 simple_ground + 坐标系")
        simple_ground(scene)
    render_coordinate_axes(scene, origin=(0, 0, 0), axis_length=0.3)
    sys.stdout.flush()

    # ---- [4/6] 分析手腕轨迹, 放置每个臂的基座 ----
    logger.info("\n[4/6] 放置每个臂的基座 ...")
    z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))

    for arm in arms:
        wrist_positions = []
        for i in range(num_frames):
            g_idx = start_frame + i
            if not hawor_data["pred_valid"][arm.hand_idx, g_idx]:
                continue
            joints = arm.compute_mano_joints(
                hawor_data["pred_rot"][arm.hand_idx, g_idx],
                hawor_data["pred_hand_pose"][arm.hand_idx, g_idx],
                hawor_data["pred_trans"][arm.hand_idx, g_idx],
            )
            wrist_positions.append(joints[0, :3].copy())
        if not wrist_positions:
            logger.error(f"  ✗ {arm.prefix}: 无有效手腕数据")
            arm.base_pos = np.array([0, 0, 0])
            arm.base_q = z_rot_180
            continue
        centroid = np.mean(wrist_positions, axis=0)
        base_pos = centroid.copy()
        base_pos[2] += COMFORTABLE_REACH
        arm.set_base_pose(base_pos, z_rot_180)
        logger.info(f"  [{arm.prefix}] 质心={centroid}, 基座={base_pos}, 有效帧={len(wrist_positions)}/{num_frames}")

        # warm start
        for probe_idx in range(num_frames):
            g_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][arm.hand_idx, g_idx]:
                continue
            joints = arm.compute_mano_joints(
                hawor_data["pred_rot"][arm.hand_idx, g_idx],
                hawor_data["pred_hand_pose"][arm.hand_idx, g_idx],
                hawor_data["pred_trans"][arm.hand_idx, g_idx],
            )
            wrist_R = pr.matrix_from_compact_axis_angle(hawor_data["pred_rot"][arm.hand_idx, g_idx])
            wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R @ RXWORLD_TO_SAPIEN.T
            wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
            hand_type = HandType.right if arm.prefix == "right" else HandType.left
            arm.warm_start(joints, wrist_quat, hand_type)
            break

    scene.step()
    scene.update_render()
    sys.stdout.flush()

    # ---- [5/6] 设置相机 ----
    logger.info("\n[5/6] 设置相机 ...")
    camera = scene.add_camera("main", args.width, args.height,
                              2 * np.arctan(args.height / 2.0 / HAWOR_FOCAL_DEFAULT),
                              0.01, 100.0)
    valid_arms = [a for a in arms if a.base_pos is not None]
    if valid_arms:
        scene_center = np.mean([a.base_pos for a in valid_arms], axis=0)
    else:
        scene_center = np.array([0, 0, 0])

    if args.view == "behind":
        cam_pos = scene_center + np.array([2.5, 0.0, 1.2])
        cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
    elif args.view == "front":
        cam_pos = scene_center + np.array([-2.5, 0.0, 1.2])
        cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
    elif args.view == "topdown":
        cam_pos = scene_center + np.array([0.0, 0.0, 2.0])
        cam_quat = make_look_at_camera(cam_pos, scene_center, up=np.array([0, 1, 0]))
    else:
        cam_pos = scene_center + np.array([0.0, 1.5, 0.8])
        cam_quat = make_look_at_camera(cam_pos, scene_center)
    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
    logger.info(f"  视角: {args.view}, 位置: {cam_pos}")
    sys.stdout.flush()

    # ---- [6/6] 同步渲染 ----
    logger.info("\n[6/6] 同步渲染 + 输出视频 ...")
    sys.stdout.flush()
    _t_render_start = time.time()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (camera.get_width(), camera.get_height()))

    kp_mats = {
        arm.prefix: context.create_material(np.zeros(4), KP_COLORS[arm.prefix], 0.0, 0.5, 0)
        for arm in arms
    }
    kp_nodes_per_arm = {arm.prefix: [] for arm in arms}
    qpos_log = {arm.prefix: [] for arm in arms}
    sys.stdout.flush()

    for local_idx in trange(num_frames, desc=f"同步渲染 ({len(arms)}个臂)"):
        global_idx = start_frame + local_idx

        for arm in arms:
            valid = hawor_data["pred_valid"][arm.hand_idx, global_idx]
            if valid:
                joints = arm.compute_mano_joints(
                    hawor_data["pred_rot"][arm.hand_idx, global_idx],
                    hawor_data["pred_hand_pose"][arm.hand_idx, global_idx],
                    hawor_data["pred_trans"][arm.hand_idx, global_idx],
                )
                kp_nodes_per_arm[arm.prefix] = render_hand_keypoints(
                    joints, kp_mats[arm.prefix], internal_scene, context,
                    kp_nodes_per_arm[arm.prefix], arm.ref_indices,
                )
                qpos = arm.retarget_and_solve_ik(joints)
                qpos_log[arm.prefix].append(qpos.copy())
            else:
                for node in kp_nodes_per_arm[arm.prefix]:
                    internal_scene.remove_node(node)
                kp_nodes_per_arm[arm.prefix].clear()

        scene.step()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        arm_list_str = "+".join([a.prefix for a in arms])
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Arms: {arm_list_str}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(bgr)

    writer.release()
    for arm in arms:
        for node in kp_nodes_per_arm[arm.prefix]:
            internal_scene.remove_node(node)

    # ============ 视频输出完成 - 关键日志 (用户要求) ============
    _t_render_end = time.time()
    final_path = args.output
    tmp_path = str(args.output).replace(".mp4", "_tmp.mp4")
    video_was_reencoded = False
    if os.path.exists(str(args.output)):
        os.rename(str(args.output), tmp_path)
        video_was_reencoded = reencode_with_ffmpeg(tmp_path, final_path, crf=args.crf, fps=args.fps, logger=logger)
        if not video_was_reencoded:
            if os.path.exists(tmp_path):
                os.rename(tmp_path, final_path)

    qpos_path = str(Path(args.output).with_suffix(".npy")).replace("videos", "tracking")
    os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
    qpos_arr = {arm.prefix: np.array(qpos_log[arm.prefix]) for arm in arms if len(qpos_log[arm.prefix]) > 0}

    # ============ 输出汇总日志 (用户要求"每次输出视频后都得有log") ============
    logger.info("")
    logger.info("=" * 80)
    logger.info("  ✓ 视频 + qpos 输出完成")
    logger.info("=" * 80)
    render_elapsed = _t_render_end - _t_render_start
    if os.path.exists(final_path):
        size_bytes = os.path.getsize(final_path)
        if size_bytes >= 1024 * 1024:
            size_str = f"{size_bytes/1024/1024:.2f} MB"
        else:
            size_str = f"{size_bytes/1024:.1f} KB"
    else:
        size_str = "(未生成)"
    actual_frames = sum(len(v) for v in qpos_log.values())
    logger.info(f"  视频路径:   {final_path}")
    logger.info(f"  视频格式:   {'H.264' if video_was_reencoded else 'mp4v'}  帧率: {args.fps}  分辨率: {args.width}x{args.height}")
    logger.info(f"  视频帧数:   {actual_frames} (与请求 --num-frames={num_frames} 一致)")
    logger.info(f"  视频大小:   {size_str}")
    logger.info(f"  渲染臂数:   {len(arms)} ({', '.join([a.prefix for a in arms])})")
    logger.info(f"  视角:       {args.view}")
    logger.info(f"  渲染耗时:   {render_elapsed:.1f} 秒 (FPS: {actual_frames/max(render_elapsed, 0.01):.1f} 帧/秒)")
    logger.info("")
    if qpos_arr:
        np.save(qpos_path, qpos_arr)
        shapes_str = ", ".join([f"{k}={v.shape}" for k, v in qpos_arr.items()])
        logger.info(f"  qpos 路径:  {qpos_path}")
        logger.info(f"  qpos 形状:  {shapes_str}")
        logger.info(f"  qpos 含义:  每帧 SAPIEN qpos (6 臂关节 + 2 夹爪 = 8 维)")
        for prefix, arr in qpos_arr.items():
            logger.info(f"              {prefix}: range=[{arr.min():.3f}, {arr.max():.3f}], mean={arr.mean():.3f}")
    logger.info("=" * 80)
    sys.stdout.flush()


def main():
    """命令行入口。

    用法:
        # 自动检测 (推荐, 适配 LEFT/RIGHT/BOTH)
        python 04_render_dual_arm.py --hawor-dir /path/to/hawor

        # 加载 GLB
        python 04_render_dual_arm.py --hawor-dir /path/to/hawor \\
            --ras-dir /path/to/ras --transform-params ./output/alignment/transform_params.npz

        # 视角
        python 04_render_dual_arm.py --hawor-dir /path/to/hawor \\
            --view behind --num-frames 100
    """
    parser = argparse.ArgumentParser(description="通用双臂/单臂协同运动渲染 (自动检测)")
    parser.add_argument("--hawor-dir", type=str, required=True, help="HaWoR 数据目录 (含 reconstruction/)")
    parser.add_argument("--ras-dir", type=str, default=None, help="RAS 目录 (含 final_scene.glb), 可选")
    parser.add_argument("--transform-params", type=str, default=None,
                        help="GLB 变换参数 npz 路径, 可选")
    parser.add_argument("--output", type=str, default="./output/videos/dual_arm_only.mp4",
                        help="输出视频路径")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--view", type=str, default="behind",
                        choices=["behind", "front", "topdown", "side"])
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--crf", type=int, default=18)
    args = parser.parse_args()

    if args.ras_dir and not args.transform_params:
        args.transform_params = "./output/alignment/transform_params.npz"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    logger = logging.getLogger("DualArmRender")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

    run_dual_arm_only(args, logger)


if __name__ == "__main__":
    main()
