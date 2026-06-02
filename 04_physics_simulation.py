"""
04_physics_simulation.py — 物理仿真驱动：真实抓取与交互

与 02_render_scene.py 的区别:
  - 02: 运动学驱动 (set_qpos 直接设置关节角，无物理交互)
  - 04: 动力学驱动 (set_drive_target + compute_passive_force + scene.step，真实物理抓取)

核心改动:
  1. 机器人关节使用 PD 驱动 (stiffness/damping)，通过 set_drive_target 控制目标位置
  2. GLB 物体添加碰撞体 (CoACD 凸分解) + 物理属性 (密度/质量)
  3. 夹爪手指设置高摩擦材质 (friction=1.0)，实现摩擦力抓取
  4. 每帧执行 decimation 次物理子步，每子步都补偿被动力 (gravity+coriolis)
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
if os.path.exists(_nvidia_icd):
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if r.returncode == 0:
            os.environ.setdefault('VK_ICD_FILENAMES', _nvidia_icd)
        else:
            os.environ.setdefault('VK_ICD_FILENAMES', _intel_icd)
    except Exception:
        os.environ.setdefault('VK_ICD_FILENAMES', _intel_icd)
else:
    os.environ.setdefault('VK_ICD_FILENAMES', _intel_icd)

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

RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
WARMUP_FRAMES = 30
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.35
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
BASE_TRACKING_RANGE = 0.08
BASE_TRACKING_ALPHA = 0.15
SAFETY_DISTANCE = 0.05
LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 20

R_GRIPPER_ALIGN = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=np.float64)

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

JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 200.0
GRIPPER_STIFFNESS = 500.0
GRIPPER_DAMPING = 50.0
PHYSICS_TIMESTEP = 1 / 240.0
CONTROL_FREQ = 30
DECIMATION = max(1, int((1.0 / CONTROL_FREQ) / PHYSICS_TIMESTEP))
OBJECT_DENSITY = 1000.0
GROUND_HEIGHT = -0.5


def reencode_with_ffmpeg(input_path, output_path, crf=18, fps=30, logger=None):
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
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-movflags", "+faststart",
        str(output_path),
    ]
    if logger:
        logger.info(f"  ffmpeg 重编码: CRF={crf}, {fps}fps, libx264")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            os.remove(input_path)
            if logger:
                sz = os.path.getsize(output_path)
                logger.info(f"  ✓ 重编码完成: {output_path} ({sz / 1024 / 1024:.1f}MB)")
            return True
    if logger:
        logger.warning(f"  ffmpeg 重编码失败: {result.stderr[:200]}")
    return False


def axis_angle_to_matrix(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    return pr.matrix_from_axis_angle(np.array([axis[0], axis[1], axis[2], angle]))


def _find_reconstruction_file(hawor_path):
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
    cam_dir = Path(hawor_path) / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if 0 in detected and 1 not in detected:
            return 0
        if 1 in detected and 0 not in detected:
            return 1
        if 0 in detected:
            return 0
        if 1 in detected:
            return 1
    return None


def load_hawor_data(hawor_dir, hand_idx=0):
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
    rec_file = _find_reconstruction_file(Path(hawor_dir))
    if rec_file is None:
        return None, None
    rec = np.load(str(rec_file), allow_pickle=True)
    if 'R_c2w' not in rec or 't_c2w' not in rec:
        return None, None
    return rec['R_c2w'], rec['t_c2w']


def compute_mano_joints(mano_layer, rot, hand_pose, trans):
    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = mano_layer(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]


def compute_smooth_shading_normal(vertices, faces):
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


def _prepare_arm_urdf(src_urdf_path, arm_prefix="right"):
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


def setup_physics_scene():
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
    scene.add_ground(GROUND_HEIGHT, render_material=ground_mat, render_half_size=[100, 100])

    return scene


def load_glb_with_physics(glb_path, transform_params_path, scene, logger=None, fast_collision=False):
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']

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

    obj_actors = []
    temp_files = []

    for geom_idx, (geom_name, geom) in enumerate(trimesh_scene.geometry.items()):
        vertices = geom.vertices.copy()
        faces = geom.faces.copy()
        if len(vertices) == 0 or len(faces) == 0:
            continue

        vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
        vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T

        avg_color = None
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            vertex_colors = geom.visual.vertex_colors
            if len(vertex_colors) > 0:
                avg_rgb = vertex_colors[:, :3].mean(axis=0)
                avg_color = [avg_rgb[0]/255.0, avg_rgb[1]/255.0, avg_rgb[2]/255.0, 1.0]

        temp_ply = f'/tmp/glb_physics_{os.getpid()}_{geom_name.replace(" ", "_")}.ply'
        geom_transformed = trimesh.Trimesh(
            vertices=vertices_sapien,
            faces=faces,
            visual=geom.visual
        )
        geom_transformed.export(temp_ply)
        temp_files.append(temp_ply)

        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.use_density = True

        phys_material = scene.create_physical_material(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.3,
        )

        cache_file = glb_cache_dir / f"{glb_hash}_{geom_name.replace(' ', '_')}.npz"

        if fast_collision:
            try:
                builder.add_convex_collision_from_file(filename=temp_ply, material=phys_material)
                if logger:
                    logger.info(f"    ✓ {geom_name}: 凸包碰撞体 (fast mode)")
            except Exception as e:
                if logger:
                    logger.warning(f"    ✗ {geom_name}: 凸包碰撞失败 ({e}), 尝试非凸")
                try:
                    builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                except Exception as e2:
                    if logger:
                        logger.warning(f"    ✗ {geom_name}: 碰撞体生成失败 ({e2})")
                    continue
        elif cache_file.exists():
            try:
                cache_data = np.load(str(cache_file), allow_pickle=True)
                convex_parts = cache_data['convex_parts'].item()
                for part_verts, part_faces in convex_parts:
                    part_ply = f'/tmp/glb_physics_part_{os.getpid()}_{geom_idx}.ply'
                    part_mesh = trimesh.Trimesh(vertices=part_verts, faces=part_faces)
                    part_mesh.export(part_ply)
                    temp_files.append(part_ply)
                    builder.add_convex_collision_from_file(filename=part_ply, material=phys_material)
                if logger:
                    logger.info(f"    ✓ {geom_name}: 缓存碰撞体 ({len(convex_parts)} 凸部件)")
            except Exception as e:
                if logger:
                    logger.warning(f"    ✗ {geom_name}: 缓存加载失败 ({e}), 重新计算 CoACD")
                try:
                    builder.add_multiple_convex_collisions_from_file(
                        filename=temp_ply, decomposition="coacd", material=phys_material,
                    )
                    if logger:
                        logger.info(f"    ✓ {geom_name}: CoACD 碰撞体已生成")
                except Exception as e2:
                    if logger:
                        logger.warning(f"    ✗ {geom_name}: CoACD 失败 ({e2}), 尝试非凸碰撞")
                    try:
                        builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                    except Exception as e3:
                        if logger:
                            logger.warning(f"    ✗ {geom_name}: 碰撞体生成失败 ({e3})")
                        continue
        else:
            try:
                builder.add_multiple_convex_collisions_from_file(
                    filename=temp_ply,
                    decomposition="coacd",
                    material=phys_material,
                )
                if logger:
                    logger.info(f"    ✓ {geom_name}: CoACD 碰撞体已生成")
            except Exception as e:
                if logger:
                    logger.warning(f"    ✗ {geom_name}: CoACD 失败 ({e}), 尝试非凸碰撞")
                try:
                    builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                except Exception as e2:
                    if logger:
                        logger.warning(f"    ✗ {geom_name}: 碰撞体生成失败 ({e2})")
                    continue

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

        actor = builder.build(name=f"glb_{geom_name}")
        actor.set_pose(sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))

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


class PhysicsSimulator:
    def __init__(self, hawor_dir, ras_dir, transform_params_path,
                 output="physics_sim.mp4", fps=30, hand_idx=0,
                 logger=None, viewer=False, crf=18, fast_collision=False):
        self.hawor_dir = Path(hawor_dir)
        self.ras_dir = Path(ras_dir)
        self.transform_params_path = Path(transform_params_path)
        self.output = output
        self.fps = fps
        self.hand_idx = hand_idx
        self.viewer = viewer
        self.crf = crf
        self.fast_collision = fast_collision
        self.logger = logger or logging.getLogger("PhysicsSim")
        self.cam_fov = 2 * np.arctan(CAM_HEIGHT / 2.0 / HAWOR_FOCAL_DEFAULT)
        self.scene = None

    def _update_cam_fov(self, hawor_data):
        img_focal = hawor_data.get("img_focal", None)
        if img_focal is not None and img_focal > 0:
            focal_for_render = img_focal * CAM_WIDTH / 1280.0
            self.cam_fov = 2 * np.arctan(CAM_HEIGHT / 2.0 / focal_for_render)
            self.logger.info(f"  相机焦距: {img_focal:.1f}px → {focal_for_render:.1f}px, FOV={np.degrees(self.cam_fov):.1f}°")
        else:
            self.cam_fov = 2 * np.arctan(CAM_HEIGHT / 2.0 / HAWOR_FOCAL_DEFAULT)
            self.logger.info(f"  相机焦距: 使用默认 {HAWOR_FOCAL_DEFAULT}px, FOV={np.degrees(self.cam_fov):.1f}°")

    def _render_to_sapien(self, pts_render):
        return (RXWORLD_TO_SAPIEN @ pts_render.T).T

    def _compute_optimal_fixed_base(self, wrist_positions_sapien):
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
        base_R = pr.matrix_from_quaternion(arm_base_q)
        wrist_in_base = base_R.T @ (wrist_pos_sapien - initial_base_pos)
        offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE
        clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)
        delta_world = base_R @ clamped_offset
        return initial_base_pos + delta_world

    def _setup_robot(self, scene, arm_base_pos, arm_base_q):
        arm_urdf_path = _prepare_arm_urdf(FLOATING_RIGHT_URDF, "right")
        loader = scene.create_urdf_loader()
        loader.fix_root_link = False
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
        internal_robot = retargeting.optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)
        for i, name in enumerate(internal_robot.link_names):
            if "right_gripper_link" in name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        return None, None

    def _compute_ee_orientation_from_wrist(self, wrist_R_sapien):
        R_mano2world = wrist_R_sapien @ OPERATOR2MANO_RIGHT.T
        return R_mano2world

    def _physics_step(self, robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                      arm_target, gripper_target1, gripper_target2):
        active_joints = robot.get_active_joints()
        for i, idx in enumerate(arm_joint_indices):
            active_joints[idx].set_drive_target(arm_target[i])
        active_joints[gripper_idx1].set_drive_target(gripper_target1)
        active_joints[gripper_idx2].set_drive_target(gripper_target2)

        for _ in range(DECIMATION):
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            self.scene.step()

    def _fetch_contacts(self, robot, obj_actors):
        gripper_link_names = {"right_gripper_finger_link1", "right_gripper_finger_link2", "right_gripper_link"}
        gripper_bodies = set()
        for link in robot.get_links():
            if link.get_name() in gripper_link_names:
                for component in link.entity.components:
                    if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                        gripper_bodies.add(component)

        obj_bodies = set()
        for actor in obj_actors:
            for component in actor.components:
                if isinstance(component, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                    obj_bodies.add(component)
                    break

        contacts = self.scene.get_contacts()
        gripper_obj_contacts = []
        total_impulse = 0.0
        for c in contacts:
            b0, b1 = c.bodies[0], c.bodies[1]
            is_gripper_obj = (b0 in gripper_bodies and b1 in obj_bodies) or \
                             (b1 in gripper_bodies and b0 in obj_bodies)
            if not is_gripper_obj:
                continue
            impulse = np.array([p.impulse for p in c.points])
            if len(impulse) > 0:
                total_impulse += np.linalg.norm(np.sum(impulse, axis=0))
            gripper_obj_contacts.append(c)

        return len(gripper_obj_contacts), total_impulse

    def _update_hand_mesh(self, vertex_sapien, mano_face, mat_hand, context, internal_scene, hand_nodes):
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

    def _render_hand_skeleton(self, joints_sapien, context, internal_scene, skel_nodes,
                              radius=0.002):
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

    def run_physics_tracking(self, start_frame=0, num_frames=-1):
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
        self.scene = setup_physics_scene()
        internal_scene = self.scene.render_system._internal_scene
        context = sapien.render.SapienRenderer()._internal_context
        mat_hand = context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1.0]), 0.0, 0.8, 0)

        glb_path = self.ras_dir / "final_scene.glb"
        obj_actors = []
        if glb_path.exists() and self.transform_params_path.exists():
            obj_actors = load_glb_with_physics(glb_path, self.transform_params_path, self.scene, logger=self.logger, fast_collision=self.fast_collision)
            if obj_actors:
                self.logger.info(f"  ✓ GLB 物理物体: {len(obj_actors)} 个")
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

        robot, joint_names, arm_joint_indices, gripper_idx1, gripper_idx2, ee_link = \
            self._setup_robot(self.scene, arm_base_pos, arm_base_q)

        for _ in range(50):
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

        ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        current_joints = np.array([init_qpos_robot[i] for i in arm_joint_indices])
        joint_filter.next(current_joints)

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

            tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
            robot.set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
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
            ee_quat_b_wxyz = np.array([ee_quat_b[3], ee_quat_b[0], ee_quat_b[1], ee_quat_b[2]])

            try:
                first_ik_joints = np.array(ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b_wxyz.tolist()))
                first_ik_target_base = ik_target_b.copy()
                first_ee_quat_base = ee_quat_b_wxyz.copy()
                first_ik_target_world = ik_target_raw.copy()
                break
            except Exception:
                continue

        if first_ik_joints is None:
            raise RuntimeError("无法求解任何有效帧的IK")

        for _ in range(200):
            first_ik_joints = np.array(ik_solver.solve_position_right(first_ik_target_base.tolist(), first_ee_quat_base.tolist()))

        ee_pos_filter.next(first_ik_target_world)

        self.logger.info("\n[6/8] 设置相机 ...")
        camera = self.scene.add_camera("main", CAM_WIDTH, CAM_HEIGHT, self.cam_fov, 0.01, 100.0)

        if R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  使用 hawor 相机轨迹 ({R_c2w_all.shape[0]}帧)")
        else:
            centroid = np.mean(wrist_positions, axis=0) if wrist_positions else np.array([0, 0, 0.3])
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  相机位置: {cam_pos}, 看向: {centroid}")

        self.logger.info("\n[7/8] Warmup + 预计算 ...")
        qpos_sequence = []
        for w in range(WARMUP_FRAMES):
            t = (w + 1) / WARMUP_FRAMES
            t_smooth = t * t * (3 - 2 * t)
            interp = current_joints * (1 - t_smooth) + first_ik_joints * t_smooth
            interp = joint_filter.next(interp)
            qpos = robot.get_qpos().copy()
            for j, idx in enumerate(arm_joint_indices):
                qpos[idx] = interp[j]
            qpos[gripper_idx1] = 0.04
            qpos[gripper_idx2] = 0.04
            qpos_sequence.append(qpos)

        for local_idx in trange(num_frames, desc="预计算"):
            global_idx = start_frame + local_idx
            if not hawor_data["pred_valid"][global_idx]:
                qpos_sequence.append(None)
                continue

            _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                       hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
            joints_sapien = self._render_to_sapien(j)

            ref_value = joints_sapien[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            sapien_qpos = retarget_qpos[retarget2sapien]
            gripper1 = float(sapien_qpos[gripper_idx1]) if gripper_idx1 < len(sapien_qpos) else 0.04
            gripper2 = float(sapien_qpos[gripper_idx2]) if gripper_idx2 < len(sapien_qpos) else -0.04

            gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retargeting, retarget_qpos)
            if gripper_pos_fk is None:
                qpos_sequence.append(None)
                continue

            tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
            robot.set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
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
            ik_target_w = ee_pos_filter.next(ik_target_raw)
            ik_target_b = base_link_R_inv @ (ik_target_w - base_link_p)
            ee_R_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
            ee_quat_b_wxyz = np.array([ee_quat_b[3], ee_quat_b[0], ee_quat_b[1], ee_quat_b[2]])

            try:
                arm_joints = np.array(ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b_wxyz.tolist()))
                for _ in range(IK_SOLVE_PER_FRAME - 1):
                    arm_joints = np.array(ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b_wxyz.tolist()))
                ik_solver.relaxed_ik_right.reset(list(arm_joints))
            except Exception:
                qpos_sequence.append(None)
                continue

            arm_joints = joint_filter.next(arm_joints)
            qpos = robot.get_qpos().copy()
            for j, idx in enumerate(arm_joint_indices):
                qpos[idx] = arm_joints[j]
            qpos[gripper_idx1] = gripper1
            qpos[gripper_idx2] = gripper2
            qpos_sequence.append(qpos)

        valid = sum(1 for x in qpos_sequence if x is not None)
        self.logger.info(f"  ✓ 预计算完成: {valid}/{len(qpos_sequence)} 帧有效")

        self.logger.info("\n[8/8] 物理仿真渲染 ...")

        if self.viewer:
            from sapien.utils import Viewer
            viewer = Viewer()
            viewer.set_scene(self.scene)
            viewer.control_window.show_origin_frame = True
            viewer.control_window.show_grid = False
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output, fourcc, self.fps,
                                     (camera.get_width(), camera.get_height()))

        hand_nodes = []
        kp_nodes = []
        skel_nodes = []
        wrist_pos_sapien = None

        gripper_link = None
        for link in robot.get_links():
            if "right_gripper_link" in link.get_name():
                gripper_link = link
                break

        for frame_idx in trange(len(qpos_sequence), desc="物理仿真渲染"):
            is_warmup = frame_idx < WARMUP_FRAMES
            data_frame_idx = frame_idx - WARMUP_FRAMES
            global_idx = start_frame + max(data_frame_idx, 0)

            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            frame_data = qpos_sequence[frame_idx]
            if frame_data is not None:
                arm_target = np.array([frame_data[i] for i in arm_joint_indices])
                gripper_target1 = float(frame_data[gripper_idx1])
                gripper_target2 = float(frame_data[gripper_idx2])
                self._physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                   arm_target, gripper_target1, gripper_target2)

            if hawor_data["pred_valid"][global_idx]:
                vertex_render, joints_render = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                                       hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                vertex_sapien = self._render_to_sapien(vertex_render)
                joints_sapien = self._render_to_sapien(joints_render)
                hand_nodes = self._update_hand_mesh(vertex_sapien, mano_face, mat_hand, context, internal_scene, hand_nodes)
                kp_nodes = self._render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes,
                                                  radius=0.004, ref_indices=set(ref_indices))
                skel_nodes = self._render_hand_skeleton(joints_sapien[:, :3], context, internal_scene, skel_nodes)
                wrist_pos_sapien = joints_sapien[0, :3].copy()

            self.scene.update_render()

            n_contacts, total_impulse = self._fetch_contacts(robot, obj_actors)

            if not self.viewer:
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                if is_warmup:
                    t = (frame_idx + 1) / WARMUP_FRAMES
                    cv2.putText(bgr, f"Warmup {frame_idx+1}/{WARMUP_FRAMES} ({t*100:.0f}%)  |  Physics Sim",
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                else:
                    ee_err_cm = None
                    if gripper_link is not None and wrist_pos_sapien is not None and frame_data is not None:
                        ee_pos = np.array(gripper_link.get_entity_pose().p)
                        ee_err_cm = np.linalg.norm(ee_pos - wrist_pos_sapien) * 100
                    label = f"Frame {data_frame_idx+1}  |  Physics Sim"
                    if ee_err_cm is not None:
                        err_color = (0, 255, 0) if ee_err_cm < 2 else (0, 255, 255) if ee_err_cm < 5 else (0, 0, 255)
                        label += f"  EE:{ee_err_cm:.1f}cm"
                    contact_color = (0, 255, 0) if n_contacts > 0 else (100, 100, 100)
                    label += f"  C:{n_contacts}"
                    cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, err_color if ee_err_cm is not None else (255, 255, 255), 2)
                    if n_contacts > 0:
                        cv2.putText(bgr, f"Impulse:{total_impulse:.2f}", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, contact_color, 1)
                writer.write(bgr)
            else:
                if R_c2w_all is not None and t_c2w_all is not None:
                    cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                    viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])
                viewer.render()

            if (frame_idx + 1) % 30 == 0:
                self.logger.info(f"  帧 {frame_idx+1}/{len(qpos_sequence)}: contacts={n_contacts}, impulse={total_impulse:.2f}")

        if not self.viewer:
            writer.release()

        for node in hand_nodes + kp_nodes + skel_nodes:
            internal_scene.remove_node(node)

        if not self.viewer:
            final_path = self.output
            tmp_path = str(self.output).replace(".mp4", "_tmp.mp4")
            if os.path.exists(str(self.output)):
                os.rename(str(self.output), tmp_path)
                if reencode_with_ffmpeg(tmp_path, final_path, crf=self.crf, fps=self.fps, logger=self.logger):
                    pass
                else:
                    if os.path.exists(tmp_path):
                        os.rename(tmp_path, final_path)

        qpos_path = str(Path(self.output).with_suffix(".npy")).replace("videos", "tracking")
        os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
        valid_qpos = [q for q in qpos_sequence if q is not None]
        if valid_qpos:
            np.save(qpos_path, np.array(valid_qpos))
            self.logger.info(f"  ✓ qpos 已保存: {qpos_path} ({len(valid_qpos)} 帧)")

        self.logger.info(f"\n✓ 物理仿真视频已保存: {self.output if self.viewer else final_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="物理仿真驱动：真实抓取与交互")
    parser.add_argument("--mode", type=str, default="physics_tracking",
                        choices=["physics_tracking"],
                        help="仿真模式")
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="HaWoR 重建输出目录 (包含 reconstruction/ 子目录或 world_space_res.pth)")
    parser.add_argument("--ras-dir", type=str, required=True,
                        help="RAS 场景重建输出目录 (包含 final_scene.glb)")
    parser.add_argument("--transform-params", type=str, default="./output/alignment/transform_params.npz",
                        help="01_align_scene.py 输出的 transform_params.npz 路径")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="手的索引: 0=左手, 1=右手, -1=自动检测")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 编码质量 (0=无损, 18=高质量, 23=默认, 28=低质量)")
    parser.add_argument("--viewer", action="store_true", help="交互式Viewer渲染（不保存视频）")
    parser.add_argument("--fast-collision", action="store_true",
                        help="使用快速凸包碰撞体代替 CoACD (速度快但精度低, 推荐调试时使用)")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"output/videos/physics_sim_{args.mode}.mp4"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.hand_idx < 0:
        detected = _detect_hand_idx(Path(args.hawor_dir))
        if detected is not None:
            args.hand_idx = detected
            hand_label = "左手" if detected == 0 else "右手"
            print(f"自动检测到手: {hand_label} (idx={detected})")
        else:
            args.hand_idx = 0
            print(f"无法自动检测手, 默认使用左手 (idx=0)")

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
                           fast_collision=args.fast_collision)
    sim.run_physics_tracking(start_frame=args.start_frame, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
