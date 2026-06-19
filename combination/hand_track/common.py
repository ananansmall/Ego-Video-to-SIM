"""
common.py — 从 02_render_scene.py 提取的核心渲染逻辑 (自适应单/双臂)

提供:
  - load_hawor_data: 加载 HaWoR 数据 (含 NaN 填充)
  - load_hawor_c2w: 加载相机位姿
  - setup_scene: 创建 SAPIEN 场景
  - load_glb_transformed: 加载 GLB 场景物体
  - prepare_arm_urdf: 预处理 URDF (替换包路径 + 修改夹爪关节)
  - render_robot_video: 核心渲染函数 (自适应单/双臂)

用法:
    from common import render_robot_video, detect_hands
    hand_indices = detect_hands(hawor_dir)
    render_robot_video(hawor_dir, ras_dir, transform_params_path, output, hand_indices, ...)
"""

import os
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
if os.path.exists(_nvidia_icd):
    os.environ['VK_ICD_FILENAMES'] = _nvidia_icd
else:
    _intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
    os.environ['VK_ICD_FILENAMES'] = _intel_icd

import re
import sys
import tempfile
import warnings
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

# 抑制无关警告
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("SAPIEN_LOG_LEVEL", "ERROR")
try:
    sapien.set_log_level("error")
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))

from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.optimizer_utils import LPFilter
from mano_layer import MANOLayer

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
FLOATING_RIGHT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_right.urdf"
FLOATING_LEFT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_left.urdf"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"

R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]
WARMUP_FRAMES = 30
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.35
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
BASE_TRACKING_RANGE = 0.04
BASE_TRACKING_ALPHA = 0.15
LP_ALPHA_JOINT = 0.5
IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 20
HAWOR_FOCAL_DEFAULT = 600.0


# ─── 手部检测 ────────────────────────────────────────────────────────────────

def detect_hands(hawor_dir):
    """自动检测 HaWoR 数据中的活跃手 (左手/右手/双手)

    综合考虑 pred_valid 和 pred_trans/pred_betas 中的 NaN:
    - pred_valid=True 但 pred_trans/pred_betas 含 NaN 的帧视为无效
    - 有效帧占比 >= 5% 才认为该手活跃

    Args:
        hawor_dir: HaWoR 数据目录

    Returns:
        手部索引列表, 如 [0](左手), [1](右手), [0,1](双手)
    """
    hawor_path = Path(hawor_dir)
    VALID_RATIO_THRESHOLD = 0.05

    rec_file = _find_reconstruction_file(hawor_path)
    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        if 'pred_valid' in rec:
            pred_valid = rec['pred_valid']
            if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                total = pred_valid.shape[1]
                hands = []
                for hi in range(min(pred_valid.shape[0], 2)):
                    valid = pred_valid[hi].copy()
                    # 排除 NaN 帧
                    if 'pred_trans' in rec:
                        has_nan = np.isnan(rec['pred_trans'][hi]).any(axis=-1)
                        valid = valid & ~has_nan
                    if 'pred_betas' in rec:
                        has_nan_b = np.isnan(rec['pred_betas'][hi]).any(axis=-1)
                        valid = valid & ~has_nan_b
                    if valid.sum() / max(total, 1) >= VALID_RATIO_THRESHOLD:
                        hands.append(hi)
                if hands:
                    return hands

    # fallback: cam_space 目录
    cam_dir = hawor_path / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if detected:
            return sorted(detected)

    return [1]  # 默认右手


# ─── 数据加载 ────────────────────────────────────────────────────────────────

def _find_reconstruction_file(hawor_path):
    rec_dir = hawor_path / "reconstruction"
    if not rec_dir.exists():
        return None
    for f in rec_dir.glob("hawor_results_*.npz"):
        return f
    return None


def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 数据 (含 NaN 填充)

    Args:
        hawor_dir: HaWoR 数据目录
        hand_idx: 手部索引 0(左) 或 1(右)

    Returns:
        字典, 含 pred_trans/pred_rot/pred_hand_pose/pred_betas/pred_valid/img_focal
    """
    hawor_path = Path(hawor_dir)
    rec_file = _find_reconstruction_file(hawor_path)
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
    else:
        ws_file = hawor_path / "world_space_res.pth"
        if ws_file.exists():
            ws = joblib.load(str(ws_file))
            pred_trans = ws[0].numpy() if hasattr(ws[0], 'numpy') else np.array(ws[0])
            pred_rot = ws[1].numpy() if hasattr(ws[1], 'numpy') else np.array(ws[1])
            pred_hand_pose = ws[2].numpy() if hasattr(ws[2], 'numpy') else np.array(ws[2])
            pred_betas = ws[3].numpy() if hasattr(ws[3], 'numpy') else np.array(ws[3])
            pred_valid = ws[4] if isinstance(ws[4], np.ndarray) else np.array(ws[4])
        else:
            raise FileNotFoundError(f"未找到 hawor 数据文件: {hawor_path}")

    result = {
        "pred_trans": pred_trans[hand_idx],
        "pred_rot": pred_rot[hand_idx],
        "pred_hand_pose": pred_hand_pose[hand_idx],
        "pred_betas": pred_betas[hand_idx],
        "pred_valid": pred_valid[hand_idx],
        "img_focal": img_focal,
    }

    # 填充 NaN 帧
    _fill_nan_frames(result)
    return result


def _fill_nan_frames(data):
    """填充数据中的 NaN 帧: 前向+后向填充, NaN 帧标记为 invalid"""
    n_frames = data["pred_trans"].shape[0]
    float_keys = ["pred_trans", "pred_rot", "pred_hand_pose", "pred_betas"]

    nan_mask = np.zeros(n_frames, dtype=bool)
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind == 'f':
            nan_mask |= np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim)))

    if not nan_mask.any():
        return

    data["pred_valid"][nan_mask] = False

    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind != 'f':
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


def load_hawor_c2w(hawor_dir):
    """加载相机到世界坐标的变换"""
    rec_file = _find_reconstruction_file(Path(hawor_dir))
    if rec_file is None:
        return None, None
    rec = np.load(str(rec_file), allow_pickle=True)
    return rec['R_c2w'], rec['t_c2w']


def compute_mano_joints(mano_layer, rot, hand_pose, trans):
    """MANO FK: 计算顶点和关节点"""
    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = mano_layer(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]


# ─── SAPIEN 场景 ─────────────────────────────────────────────────────────────

def setup_scene():
    """创建并配置 SAPIEN 渲染场景"""
    from sapien.asset import create_dome_envmap
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_ray_tracing_samples_per_pixel(64)
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)
    # 禁用重力: 场景仅用于渲染, 不需要物理仿真
    # 避免 gripper_arm 模式下 arm 关节因重力下垂
    physx_config = scene.get_physx_system().get_config()
    physx_config.gravity = np.array([0, 0, 0])
    scene.set_environment_map(create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35]))
    scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
    scene.add_directional_light([-1, -0.5, -1], [1.2, 1.2, 1.2], shadow=False)
    scene.add_directional_light([0, 1, -0.5], [0.8, 0.8, 0.8], shadow=False)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    return scene


def load_glb_transformed(glb_path, transform_params_path, scene, logger=None):
    """加载 GLB 场景, 应用坐标变换, 创建 SAPIEN 静态 actor"""
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    if trimesh is None:
        return []
    trimesh_scene = trimesh.load(str(glb_path))
    obj_actors = []
    temp_files = []
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
        geom_transformed = trimesh.Trimesh(vertices=vertices_sapien, faces=faces, visual=geom.visual)
        geom_transformed.export(temp_ply)
        temp_files.append(temp_ply)
        builder = scene.create_actor_builder()
        if avg_color is not None:
            material = sapien.render.RenderMaterial(base_color=avg_color, metallic=0.0, roughness=0.7, specular=0.3)
            builder.add_visual_from_file(filename=temp_ply, material=material)
        else:
            builder.add_visual_from_file(filename=temp_ply)
        actor = builder.build_kinematic(name=geom_name)
        actor.set_pose(sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        obj_actors.append(actor)
    for tf in temp_files:
        try:
            os.remove(tf)
        except Exception:
            pass
    return obj_actors


# ─── URDF 预处理 ─────────────────────────────────────────────────────────────

def prepare_arm_urdf(src_urdf_path, arm_prefix="right"):
    """预处理单臂 URDF: 替换包路径 + 修改夹爪关节类型"""
    xml = Path(src_urdf_path).read_text()
    xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")
    xml = re.sub(
        rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint1"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    xml = re.sub(
        rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint2"\s+type=")fixed(")',
        r'\1prismatic\2', xml
    )
    temp_dir = tempfile.mkdtemp(prefix=f"r1_{arm_prefix}_arm_urdf-")
    temp_path = f"{temp_dir}/{src_urdf_path.name}"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


# ─── 坐标变换 ────────────────────────────────────────────────────────────────

def _render_to_sapien(pts):
    """HaWoR render 坐标 → SAPIEN 世界坐标"""
    return (RXWORLD_TO_SAPIEN @ pts.T).T


def hawor_cam_to_sapien_pose(R_c2w, t_c2w):
    """HaWoR 相机位姿 → SAPIEN 相机位姿"""
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
    return cam_pos_sapien, pr.quaternion_from_matrix(sapien_cam_R)


def make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
    """构造 look-at 相机旋转"""
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


# ─── Retargeting 辅助 ───────────────────────────────────────────────────────

def _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix):
    """从 retargeting 优化器的正运动学获取夹爪位姿

    与 02_render_scene.py 的实现一致: 使用 retargeting 内部机器人的 FK
    """
    internal_robot = retargeting.optimizer.robot
    internal_robot.compute_forward_kinematics(retarget_qpos)
    target_name = f"{prefix}_gripper_link"
    for i, name in enumerate(internal_robot.link_names):
        if name == target_name:
            pose = internal_robot.get_link_pose(i)
            return pose[:3, 3].copy(), pose[:3, :3].copy()
    raise RuntimeError(f"内部机器人中找不到 {target_name}")


def _compute_optimal_fixed_base(wrist_positions):
    """计算机器人基座的最优固定位置和朝向 (与 02 一致)

    策略:
    1. 计算所有有效帧手腕位置的质心
    2. 基座放在质心正上方 COMFORTABLE_REACH (0.35m) 处
    3. 朝向: 绕Z轴旋转180° (让机器人面朝操作者)
    4. 如果手腕运动范围大, 沿X方向微调
    """
    if not wrist_positions:
        return np.array([0.0, 0.0, COMFORTABLE_REACH]), pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))

    wrist_arr = np.array(wrist_positions)
    centroid = wrist_arr.mean(axis=0)
    wrist_range = wrist_arr.max(axis=0) - wrist_arr.min(axis=0)

    arm_base_pos = centroid.copy()
    arm_base_pos[2] += COMFORTABLE_REACH

    if wrist_range[0] > 0.01:
        arm_base_pos[0] += wrist_range[0] * 0.1

    z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
    arm_base_q = z_rot_180

    return arm_base_pos, arm_base_q


def _compute_tracking_base_pos(initial_base_pos, wrist_pos_sapien, arm_base_q):
    """计算跟踪模式下的基座位置 (与 02 一致)

    基座在初始位置基础上, 沿 XY 方向跟踪手腕 (±BASE_TRACKING_RANGE),
    Z 方向保持固定。
    """
    base_R = pr.matrix_from_quaternion(arm_base_q)
    wrist_in_base = base_R.T @ (wrist_pos_sapien - initial_base_pos)
    offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE
    clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)
    delta_world = base_R @ clamped_offset
    return initial_base_pos + delta_world


def _compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames):
    """计算所有有效手腕的 SAPIEN 世界系位置"""
    positions = []
    for i in range(num_frames):
        gi = start_frame + i
        if not hawor_data["pred_valid"][gi]:
            continue
        trans = hawor_data["pred_trans"][gi]
        if np.isnan(trans).any():
            continue
        _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][gi],
                                   hawor_data["pred_hand_pose"][gi], trans)
        joints_sapien = _render_to_sapien(j)
        positions.append(joints_sapien[0, :3].copy())
    return positions


# ─── 关键点渲染 ──────────────────────────────────────────────────────────────

def _render_keypoints(joints_sapien, context, internal_scene, kp_nodes,
                      ref_indices, radius=0.008, clear_existing=True):
    """渲染手部关键点为球体 (与02一致)

    ref_indices 中的3个关节 (手腕/食指尖/中指尖) 用绿色大球标记

    Args:
        joints_sapien: (21, 3) SAPIEN 坐标系下的关节位置
        context: SAPIEN 渲染上下文
        internal_scene: SAPIEN 内部场景
        kp_nodes: 已有的关键点渲染节点列表
        ref_indices: retargeting 参考关节索引 (绿色标记)
        radius: 球体半径
        clear_existing: 是否清除已有节点 (双手模式下设为False以累加)

    Returns:
        list: 关键点渲染节点列表
    """
    if clear_existing:
        for node in kp_nodes:
            internal_scene.remove_node(node)
        kp_nodes.clear()

    mat_ref = context.create_material(
        np.zeros(4), np.array([0.0, 1.0, 0.0, 1.0]), 0.0, 0.5, 0
    )

    for i in ref_indices:
        if i >= len(joints_sapien):
            continue
        joint_pos = joints_sapien[i, :3]
        r = radius
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


# ─── 核心渲染函数 ────────────────────────────────────────────────────────────

def render_robot_video(hawor_dir, ras_dir, transform_params_path, output,
                       hand_idx=1, fps=30, cam_width=1920, cam_height=1080,
                       view="fpv", crf=18, start_frame=0, num_frames=-1,
                       logger=None):
    """渲染 R1 单臂机器人 + GLB 场景视频

    核心逻辑来自 02_render_scene.py 的 run_robot_tracking:
      - 单臂浮动基座 + retargeting + RelaxedIK
      - GLB 场景物体加载
      - NaN 帧安全处理

    Args:
        hawor_dir:             HaWoR 数据目录
        ras_dir:               RAS 重建结果目录 (含 final_scene.glb)
        transform_params_path: GLB 坐标变换 npz 路径 (None=不加载GLB)
        output:                输出 mp4 路径
        hand_idx:              手部索引 0(左) 或 1(右)
        fps:                   帧率
        cam_width/height:      分辨率
        view:                  相机视角
        crf:                   编码质量
        start_frame:           起始帧
        num_frames:            帧数 (-1=全部)
        logger:                Logger
    """
    import logging
    if logger is None:
        logger = logging.getLogger("render")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

    prefix = "left" if hand_idx == 0 else "right"
    mode_label = "左手" if hand_idx == 0 else "右手"
    logger.info(f"渲染模式: R1 {mode_label}臂 + GLB 物体")

    # ── [1/7] 加载数据 ──
    logger.info("\n[1/7] 加载数据 ...")
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
    logger.info(f"  焦距: {focal:.1f}px, FOV={np.degrees(cam_fov):.1f}°")

    # ── [2/7] 创建场景 + 加载 GLB ──
    logger.info("\n[2/7] 创建场景 + 加载 GLB ...")
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")
    else:
        logger.warning(f"  GLB 或变换参数不存在, 跳过 GLB 加载")

    # ── [3/7] 加载机器人 + 初始化 retargeting/IK ──
    logger.info(f"\n[3/7] 初始化 R1 {prefix} 臂 ...")
    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    urdf_path = FLOATING_LEFT_URDF if hand_idx == 0 else FLOATING_RIGHT_URDF
    arm_starting = LEFT_ARM_STARTING if hand_idx == 0 else RIGHT_ARM_STARTING

    arm_urdf_path = prepare_arm_urdf(urdf_path, arm_prefix=prefix)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(arm_urdf_path)

    joint_names = [j.name for j in robot.get_active_joints()]
    arm_joint_indices = [i for i, n in enumerate(joint_names) if f"{prefix}_arm_joint" in n]
    gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
    gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=100000.0, damping=10000.0)

    init_qpos = robot.get_qpos().copy()
    for j, idx in enumerate(arm_joint_indices):
        if j < len(arm_starting):
            init_qpos[idx] = arm_starting[j]
    init_qpos[gripper_idx1] = 0.04
    init_qpos[gripper_idx2] = -0.04
    robot.set_qpos(init_qpos)
    scene.step()
    scene.update_render()

    # Retargeting
    hand_type = HandType.left if hand_idx == 0 else HandType.right
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
    for i, retarget_idx in enumerate(fixed_retarget_indices):
        if retarget_idx in sapien2retarget:
            fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

    # IK solver
    ik_solver = RelaxedIKSolver(
        left_setting_file_path=str(R1_LEFT_SETTINGS),
        right_setting_file_path=str(R1_RIGHT_SETTINGS),
        tolerances=IK_TOLERANCES,
    )
    if hand_idx == 0:
        ik_solver.relaxed_ik_left.reset(LEFT_ARM_STARTING)
    else:
        ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

    # MANO layer
    betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
    mano_layer = MANOLayer(prefix, betas_mean)

    logger.info(f"  ✓ {prefix} 臂已加载: {len(arm_joint_indices)}个臂关节 + 2个夹爪关节")

    # ── [4/7] Warm start ──
    logger.info("\n[4/7] Warm start ...")
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

    # ── [5/7] 放置机器人 + Warmup ──
    logger.info("\n[5/7] 放置机器人 ...")
    wrist_positions = _compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
    if not wrist_positions:
        logger.error("无法提取有效手腕位置")
        return

    arm_base_pos, arm_base_q = _compute_optimal_fixed_base(wrist_positions)
    robot.set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
    scene.step()
    scene.update_render()
    logger.info(f"  初始基座: {arm_base_pos}")

    mapping_offset = np.zeros(3)
    safety_offset = np.zeros(3)

    # 初始化 joint_filter
    init_qpos_arm = robot.get_qpos().copy()
    current_joints = np.array([init_qpos_arm[i] for i in arm_joint_indices])
    joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
    joint_filter.next(current_joints)

    # Warmup
    first_valid_qpos = None
    for fi in range(start_frame, start_frame + num_frames):
        if not hawor_data["pred_valid"][fi]:
            continue
        rot = hawor_data["pred_rot"][fi]
        trans = hawor_data["pred_trans"][fi]
        hand_pose = hawor_data["pred_hand_pose"][fi]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        ref_value = joints_sapien[ref_indices, :].astype(np.float32)
        retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
        sapien_qpos = retarget_qpos[retarget2sapien]
        gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)
        tracked_base = _compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
        robot.set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
        scene.step()
        for link in robot.get_links():
            if f"{prefix}_arm_base_link" == link.get_name():
                pose = link.get_entity_pose()
                base_link_p_w = np.array(pose.p)
                base_link_q_w = np.array(pose.q)
                break
        base_link_R_w = pr.matrix_from_quaternion(base_link_q_w)
        base_link_R_inv_w = base_link_R_w.T
        ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
        ik_target_b = base_link_R_inv_w @ (ik_target_raw - base_link_p_w)
        ee_R_base = base_link_R_inv_w @ R_ee_world_fk
        ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
        solve_fn = ik_solver.solve_position_left if prefix == "left" else ik_solver.solve_position_right
        ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        for _ in range(IK_SOLVE_PER_FRAME * 5 - 1):
            ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        first_valid_qpos = np.array(ik_joints)
        break

    if first_valid_qpos is not None:
        init_qpos_arm_arr = np.array(arm_starting)
        for wi in range(WARMUP_FRAMES):
            t = (wi + 1) / WARMUP_FRAMES
            t = t * t * (3 - 2 * t)
            interp = init_qpos_arm_arr * (1 - t) + first_valid_qpos * t
            qpos = robot.get_qpos().copy()
            for j_idx, arm_idx in enumerate(arm_joint_indices):
                qpos[arm_idx] = interp[j_idx]
            robot.set_qpos(qpos)
            scene.step()
        logger.info(f"  Warmup 完成 ({WARMUP_FRAMES} 帧)")

    # ── [6/7] 设置相机 ──
    logger.info("\n[6/7] 设置相机 ...")
    camera = scene.add_camera("main", cam_width, cam_height, cam_fov, 0.01, 100.0)

    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        logger.info(f"  使用 hawor 相机轨迹 (第一视角)")
    else:
        centroid = np.mean(wrist_positions, axis=0)
        robot_root = arm_base_pos.copy()
        if view == "topdown":
            cam_target = centroid
            cam_pos = cam_target + np.array([0.0, 0.0, 1.2])
            cam_quat = make_look_at_camera(cam_pos, cam_target, up=np.array([0, 1, 0]))
        elif view == "behind":
            cam_pos = robot_root + np.array([2.5, 0.0, 1.2])
            cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
        elif view == "front":
            cam_pos = robot_root + np.array([-2.5, 0.0, 1.2])
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        logger.info(f"  视角: {view}, 相机位置: {cam_pos}")

    # ── [7/7] 渲染视频 ──
    logger.info("\n[7/7] 渲染视频 ...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    qpos_log = []
    solve_fn = ik_solver.solve_position_left if prefix == "left" else ik_solver.solve_position_right
    kp_nodes = []  # 关键点渲染节点
    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene

    for local_idx in trange(num_frames, desc=f"渲染{prefix}"):
        global_idx = start_frame + local_idx

        # 更新相机
        if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

        if not hawor_data["pred_valid"][global_idx]:
            # 无效帧: 重复上一帧
            scene.step()
            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  R1 {prefix} Arm  |  INVALID",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(bgr)
            continue

        rot = hawor_data["pred_rot"][global_idx]
        trans = hawor_data["pred_trans"][global_idx]
        hand_pose = hawor_data["pred_hand_pose"][global_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            scene.step()
            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  R1 {prefix} Arm  |  NaN",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(bgr)
            continue

        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)

        # 渲染3个关键点 (手腕/食指尖/中指尖)
        kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices)

        ref_value = joints_sapien[ref_indices, :].astype(np.float32)
        retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
        sapien_qpos = retarget_qpos[retarget2sapien]

        gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)

        tracked_base = _compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
        robot.set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
        scene.step()

        for link in robot.get_links():
            if f"{prefix}_arm_base_link" == link.get_name():
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

        ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        for _ in range(IK_SOLVE_PER_FRAME - 1):
            ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())

        filtered_joints = joint_filter.next(np.array(ik_joints))

        qpos = robot.get_qpos().copy()
        for j_idx, arm_idx in enumerate(arm_joint_indices):
            qpos[arm_idx] = filtered_joints[j_idx]
        if gripper_idx1 < len(sapien_qpos):
            qpos[gripper_idx1] = float(sapien_qpos[gripper_idx1])
        if gripper_idx2 < len(sapien_qpos):
            qpos[gripper_idx2] = float(sapien_qpos[gripper_idx2])
        robot.set_qpos(qpos)
        qpos_log.append(qpos.copy())

        scene.step()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  R1 {prefix} Arm + Objects",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(bgr)

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

    # qpos 保存
    qpos_path = str(Path(output).with_suffix(".npy")).replace("videos", "tracking")
    qpos_path = qpos_path.replace(".npy", f"_{prefix}.npy")
    os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
    if qpos_log:
        np.save(qpos_path, np.array(qpos_log))
        logger.info(f"  ✓ {prefix} qpos 已保存: {qpos_path} ({len(qpos_log)} 帧)")

    logger.info(f"\n✓ 视频已保存: {final_path}")
    return final_path


def render_gripper_video(hawor_dir, ras_dir, transform_params_path, output,
                         hand_idx=1, fps=30, cam_width=1920, cam_height=1080,
                         view="fpv", crf=18, start_frame=0, num_frames=-1,
                         logger=None):
    """渲染夹爪末端跟踪视频 (只有3个关键点球体, 没有手臂)

    只渲染手腕/食指尖/中指尖的3个绿色球体, 不加载手臂URDF。
    用于验证手部跟踪精度, 排除手臂底座不确定性的干扰。

    Args:
        与 render_robot_video 相同
    """
    import logging
    if logger is None:
        logger = logging.getLogger("render_gripper")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    prefix = "left" if hand_idx == 0 else "right"
    mode_label = "左手" if hand_idx == 0 else "右手"
    logger.info(f"夹爪渲染: {mode_label}")

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

    # retargeting ref_indices
    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    hand_type = HandType.left if hand_idx == 0 else HandType.right
    config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
    override = dict(
        add_dummy_free_joint=True, normal_delta=1e-5, huber_delta=0.01,
        target_link_names=[f"{prefix}_gripper_finger_link1",
                           f"{prefix}_gripper_finger_link2",
                           f"{prefix}_gripper_link"],
        target_link_human_indices=np.array([4, 8, 0]),
    )
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    ref_indices = retargeting.optimizer.target_link_human_indices

    # ── 创建场景 ──
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")

    # ── 设置相机 ──
    camera = scene.add_camera("gripper", cam_width, cam_height, cam_fov, 0.01, 100.0)

    # 计算手腕位置用于放置相机
    wrist_positions = _compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
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
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

    # ── 渲染 ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    for local_idx in trange(num_frames, desc=f"夹爪{prefix}"):
        global_idx = start_frame + local_idx

        if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

        if not hawor_data["pred_valid"][global_idx]:
            for node in kp_nodes:
                internal_scene.remove_node(node)
            kp_nodes.clear()
            scene.step()
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

        # 只渲染3个关键点球体
        kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012)

        scene.step()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper {prefix} Only",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(bgr)

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

    logger.info(f"\n✓ 夹爪视频已保存: {final_path}")
    return final_path
