"""轨迹加载器 — 复用 02_render_scene.py 的 HaWoR+GLB 轨迹加载逻辑

提供:
  - load_hawor_data: 加载 HaWoR 手部重建数据
  - load_hawor_c2w: 加载相机轨迹
  - load_glb_transformed: 加载 GLB 场景+变换参数
  - compute_mano_joints: MANO FK 计算手部关节
  - compute_analytical_gripper_pose: 解析计算夹爪位姿 (gripper_only 形式)
"""

import sys
import logging
from pathlib import Path

import numpy as np
from pytransform3d import rotations as pr

from physics_utils import (
    PROJECT_ROOT, RXWORLD_TO_SAPIEN, R1_MESH_DIR,
    _FINGER1_ORIGIN, _FINGER1_AXIS, _FINGER2_ORIGIN, _FINGER2_AXIS,
    FINGER_BASE_DIST, GRIPPER_INIT_OPEN,
)

logger = logging.getLogger(__name__)


def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据

    支持两种数据格式:
      1. reconstruction/hawor_results_*.npz (推荐)
      2. world_space_res.pth (旧格式)

    Args:
        hawor_dir: HaWoR 输出目录路径
        hand_idx: 手部索引 (0=左手, 1=右手)

    Returns:
        dict: 包含 pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid, img_focal
    """
    import torch
    hawor_path = Path(hawor_dir)
    rec_dir = hawor_path / "reconstruction"

    # 查找 npz 文件
    npz_file = None
    if rec_dir.exists():
        for f in rec_dir.glob("hawor_results_*.npz"):
            npz_file = f
            break
    if npz_file is None:
        # 直接在目录下查找
        for f in hawor_path.glob("hawor_results_*.npz"):
            npz_file = f
            break

    if npz_file is not None:
        data = np.load(str(npz_file), allow_pickle=True)
        result = {}
        for key in data.files:
            result[key] = data[key]

        # 确保必要字段存在
        required = ['pred_trans', 'pred_rot', 'pred_hand_pose', 'pred_betas']
        for k in required:
            if k not in result:
                raise KeyError(f"HaWoR 数据缺少字段: {k}")

        # 按hand_idx索引: 原始数据shape=(2, N, ...) → (N, ...)
        # hand_idx=0 → 左手, hand_idx=1 → 右手
        for key in ['pred_trans', 'pred_rot', 'pred_hand_pose', 'pred_betas']:
            arr = result[key]
            if arr.ndim >= 1 and arr.shape[0] == 2:
                result[key] = arr[hand_idx]
                logger.info(f"    {key}: {arr.shape} → indexed by hand_idx={hand_idx} → {result[key].shape}")

        # pred_valid: (2, N) → (N,)
        if 'pred_valid' in result:
            arr = result['pred_valid']
            if arr.ndim == 2 and arr.shape[0] == 2:
                result['pred_valid'] = arr[hand_idx]

        # img_focal
        if 'img_focal' in result:
            result['img_focal'] = float(result['img_focal'])
        else:
            result['img_focal'] = 600.0

        logger.info(f"  HaWoR 数据已加载: {npz_file.name}, {len(result['pred_trans'])} 帧")
        return result

    # 尝试旧格式 .pth
    ws_file = hawor_path / "world_space_res.pth"
    if ws_file.exists():
        ws = torch.load(str(ws_file), map_location='cpu', weights_only=False)
        result = {
            'pred_trans': ws['pred_trans'].numpy() if hasattr(ws['pred_trans'], 'numpy') else ws['pred_trans'],
            'pred_rot': ws['pred_rot'].numpy() if hasattr(ws['pred_rot'], 'numpy') else ws['pred_rot'],
            'pred_hand_pose': ws['pred_hand_pose'].numpy() if hasattr(ws['pred_hand_pose'], 'numpy') else ws['pred_hand_pose'],
            'pred_betas': ws['pred_betas'].numpy() if hasattr(ws['pred_betas'], 'numpy') else ws['pred_betas'],
            'pred_valid': np.ones(len(ws['pred_trans']), dtype=bool),
            'img_focal': 600.0,
        }
        logger.info(f"  HaWoR 数据已加载 (pth): {ws_file.name}, {len(result['pred_trans'])} 帧")
        return result

    raise FileNotFoundError(f"未找到 HaWoR 数据: {hawor_dir}")


def load_hawor_c2w(hawor_dir):
    """加载 HaWoR 相机轨迹 (c2w)

    Args:
        hawor_dir: HaWoR 输出目录路径

    Returns:
        R_c2w_all: (N, 3, 3) 旋转矩阵
        t_c2w_all: (N, 3) 平移向量
    """
    hawor_path = Path(hawor_dir)
    rec_dir = hawor_path / "reconstruction"

    npz_file = None
    if rec_dir.exists():
        for f in rec_dir.glob("hawor_results_*.npz"):
            npz_file = f
            break
    if npz_file is None:
        for f in hawor_path.glob("hawor_results_*.npz"):
            npz_file = f
            break

    if npz_file is not None:
        data = np.load(str(npz_file), allow_pickle=True)
        if 'R_c2w' in data and 't_c2w' in data:
            return data['R_c2w'], data['t_c2w']

    # 默认: 单位矩阵 (无相机轨迹)
    logger.warning("  未找到相机轨迹, 使用单位矩阵")
    return np.eye(3)[None], np.zeros((1, 3))


def compute_mano_joints(mano_layer, pred_rot, pred_hand_pose, pred_trans):
    """MANO FK 计算手部关节

    Args:
        mano_layer: MANOLayer 实例
        pred_rot: (3,) 手腕旋转 (轴角)
        pred_hand_pose: (45,) 手指关节角
        pred_trans: (3,) 手腕平移

    Returns:
        vertices: (778, 3) 顶点
        joints: (21, 3) 关节
    """
    import torch
    # MANOLayer.forward(p, t): p=(B,48), t=(B,3)
    # p = concat(wrist_rot(3), hand_pose(45)) = 48
    p = torch.tensor(
        np.concatenate([pred_rot, pred_hand_pose]),
        dtype=torch.float32
    ).unsqueeze(0)  # (1, 48)
    t = torch.tensor(pred_trans, dtype=torch.float32).unsqueeze(0)  # (1, 3)

    vertices, joints = mano_layer(p, t)
    vertices = vertices[0].detach().cpu().numpy()
    joints = joints[0].detach().cpu().numpy()
    return vertices, joints


def compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2):
    """从 MANO 指尖向量解析计算夹爪 root 位姿和手指关节值

    与 04_physics_simulation.py / hand_track/render_gripper_only.py 一致

    Args:
        mano_wrist: (3,) 手腕位置
        mano_finger1: (3,) 手指1位置 (食指尖, MANO joint 4)
        mano_finger2: (3,) 手指2位置 (小指尖, MANO joint 8)

    Returns:
        root_pos: (3,) 夹爪根位置
        root_R: (3,3) 夹爪根旋转矩阵
        joint1, joint2: 手指关节值 [0, 0.05]
    """
    v_finger = mano_finger2 - mano_finger1
    finger_dist = np.linalg.norm(v_finger)
    if finger_dist < 1e-6:
        y_axis = np.array([0, 1, 0], dtype=np.float64)
    else:
        y_axis = v_finger / finger_dist

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

    root_R = np.column_stack([x_axis, y_axis, z_axis])

    required_open_sum = finger_dist - FINGER_BASE_DIST
    joint1 = max(0.0, min(0.05, required_open_sum / 2))
    joint2 = max(0.0, min(0.05, required_open_sum / 2))

    finger1_offset = _FINGER1_ORIGIN + _FINGER1_AXIS * joint1
    root_pos = mano_finger1 - root_R @ finger1_offset

    return root_pos, root_R, joint1, joint2


def load_glb_transformed(glb_path, transform_params_path, scene=None, backend="sapien", logger=None):
    """加载 GLB 场景并应用变换参数

    Args:
        glb_path: GLB 文件路径
        transform_params_path: transform_params.npz 路径
        scene: SAPIEN scene 或 PyBullet physics_client (取决于backend)
        backend: "sapien" 或 "pybullet"
        logger: 日志记录器

    Returns:
        list: 加载的物体 actor 列表
    """
    glb_path = Path(glb_path)
    transform_params_path = Path(transform_params_path)

    if not glb_path.exists():
        if logger:
            logger.error(f"GLB 文件不存在: {glb_path}")
        return []
    if not transform_params_path.exists():
        if logger:
            logger.error(f"变换参数不存在: {transform_params_path}")
        return []

    tp = np.load(str(transform_params_path))
    s = float(tp['scale'])
    R_inv = tp['R_inv']  # (3, 3)
    t_inv = tp['t_inv']  # (3,)

    if backend == "sapien":
        return _load_glb_sapien(glb_path, s, R_inv, t_inv, scene, logger)
    elif backend == "pybullet":
        return _load_glb_pybullet(glb_path, s, R_inv, t_inv, scene, logger)
    else:
        raise ValueError(f"未知后端: {backend}")


def _load_glb_sapien(glb_path, scale, R_inv, t_inv, scene, logger=None):
    """SAPIEN: 加载 GLB 物体 (对齐 02_render_scene.py)

    关键: 使用 s_inv = 1/scale (缩小), 而非 scale (放大).
    顶点变换后直接设 pose=[0,0,0] (顶点已在世界坐标系), 不再用 center 做位姿.
    """
    try:
        import trimesh
        import sapien
        import tempfile
        import os

        # s_inv = 1/scale (对齐 02_render_scene.py: 用逆缩放将 RAS 米制 → HaWoR 米制)
        s_inv = 1.0 / float(scale)

        glb_scene = trimesh.load(str(glb_path))
        actors = []
        tmp_dir = tempfile.mkdtemp(prefix="glb_sapien_")

        if isinstance(glb_scene, trimesh.Scene):
            mesh_dict = glb_scene.geometry
        else:
            mesh_dict = {"object": glb_scene}

        from physics_utils import is_large_scene_object, compute_object_mass, OBJECT_DENSITY

        for idx, (name, mesh) in enumerate(mesh_dict.items()):
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
            if len(vertices) == 0 or len(faces) == 0:
                continue

            # 变换链 (对齐 02_render_scene.py L917-918):
            #   p_hawor = s_inv * (R_inv @ p_ras) + t_inv
            #   p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
            vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
            vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T

            # 重建变换后的mesh (保留顶点颜色)
            mesh_transformed = trimesh.Trimesh(
                vertices=vertices_sapien, faces=faces, visual=mesh.visual
            )

            # 计算包围盒 (用于判断大小和碰撞)
            bbox_min = vertices_sapien.min(axis=0)
            bbox_max = vertices_sapien.max(axis=0)
            size = bbox_max - bbox_min

            # 导出为临时 PLY 文件 (保留顶点颜色, 对齐 02_render_scene.py)
            ply_path = os.path.join(tmp_dir, f"mesh_{idx}.ply")
            mesh_transformed.export(ply_path)

            # 计算平均顶点颜色 (对齐 02_render_scene.py)
            avg_color = None
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vcolors = mesh.visual.vertex_colors
                if len(vcolors) > 0:
                    avg_rgb = vcolors[:, :3].mean(axis=0)
                    avg_color = [avg_rgb[0]/255.0, avg_rgb[1]/255.0, avg_rgb[2]/255.0, 1.0]

            if is_large_scene_object(bbox_min, bbox_max):
                # 大型场景结构 → kinematic
                builder = scene.create_actor_builder()
                if avg_color is not None:
                    material = sapien.render.RenderMaterial(
                        base_color=avg_color, metallic=0.0, roughness=0.7, specular=0.3
                    )
                    builder.add_visual_from_file(filename=ply_path, material=material)
                else:
                    builder.add_visual_from_file(filename=ply_path)
                builder.add_nonconvex_collision_from_file(ply_path)
                actor = builder.build_kinematic(name=name)
            else:
                # 小物体 → dynamic
                mass = compute_object_mass(bbox_min, bbox_max)
                builder = scene.create_actor_builder()
                if avg_color is not None:
                    material = sapien.render.RenderMaterial(
                        base_color=avg_color, metallic=0.0, roughness=0.7, specular=0.3
                    )
                    builder.add_visual_from_file(filename=ply_path, material=material)
                else:
                    builder.add_visual_from_file(filename=ply_path)
                # 碰撞用凸包
                try:
                    convex_mesh = mesh_transformed.convex_hull
                    convex_obj = os.path.join(tmp_dir, f"mesh_{idx}_convex.obj")
                    convex_mesh.export(convex_obj)
                    builder.add_convex_collision_from_file(convex_obj)
                except Exception:
                    builder.add_box_collision(
                        half_size=(size / 2).tolist(),
                        density=OBJECT_DENSITY,
                    )
                actor = builder.build(name=name)

            # 关键: 顶点已在世界坐标系, pose 设为原点 (对齐 02_render_scene.py L950)
            actor.set_pose(sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
            actors.append(actor)

            if logger:
                logger.info(f"    物体{idx} '{name}': bbox=[{bbox_min[0]:.3f},{bbox_min[1]:.3f},{bbox_min[2]:.3f}] → "
                            f"[{bbox_max[0]:.3f},{bbox_max[1]:.3f},{bbox_max[2]:.3f}], size={size.tolist()}")

        if logger:
            logger.info(f"  SAPIEN: 加载 {len(actors)} 个GLB物体 (s_inv={s_inv:.4f})")
        return actors

    except Exception as e:
        if logger:
            logger.error(f"  SAPIEN GLB加载失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def _load_glb_pybullet(glb_path, scale, R_inv, t_inv, physics_client, logger=None):
    """PyBullet: 加载 GLB 物体 (对齐 02_render_scene.py, 用 s_inv=1/scale)"""
    try:
        import trimesh
        import pybullet as p
        import tempfile

        s_inv = 1.0 / float(scale)

        glb_scene = trimesh.load(str(glb_path))
        actors = []

        if isinstance(glb_scene, trimesh.Scene):
            mesh_dict = glb_scene.geometry
        else:
            mesh_dict = {"object": glb_scene}

        for name, mesh in mesh_dict.items():
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
            if len(vertices) == 0 or len(faces) == 0:
                continue

            # 变换链 (对齐 02_render_scene.py): s_inv * R_inv @ p + t_inv → RXWORLD_TO_SAPIEN @ p
            vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
            vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T

            # 计算包围盒
            bbox_min = vertices_sapien.min(axis=0)
            bbox_max = vertices_sapien.max(axis=0)
            size = bbox_max - bbox_min

            from physics_utils import is_large_scene_object, compute_object_mass, OBJECT_DENSITY

            # 保存为临时 OBJ
            mesh_transformed = trimesh.Trimesh(vertices=vertices_sapien, faces=faces)
            tmp_dir = tempfile.mkdtemp(prefix="glb_obj_")
            obj_path = f"{tmp_dir}/{name}.obj"
            mesh_transformed.export(obj_path)

            if is_large_scene_object(bbox_min, bbox_max):
                # 大型场景结构 → static, pose=[0,0,0] (顶点已变换)
                col_id = p.createCollisionShape(
                    p.GEOM_MESH, fileName=obj_path,
                    meshScale=[1, 1, 1],
                    physicsClientId=physics_client,
                )
                body_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_id,
                    basePosition=[0, 0, 0],
                    physicsClientId=physics_client,
                )
            else:
                # 小物体 → dynamic, pose=[0,0,0] (顶点已变换)
                mass = compute_object_mass(bbox_min, bbox_max)
                try:
                    convex_mesh = mesh_transformed.convex_hull
                    convex_obj = f"{tmp_dir}/{name}_convex.obj"
                    convex_mesh.export(convex_obj)
                    col_id = p.createCollisionShape(
                        p.GEOM_MESH, fileName=convex_obj,
                        meshScale=[1, 1, 1],
                        physicsClientId=physics_client,
                    )
                except Exception:
                    half_extents = (size / 2).tolist()
                    col_id = p.createCollisionShape(
                        p.GEOM_BOX, halfExtents=half_extents,
                        physicsClientId=physics_client,
                    )
                vis_id = p.createVisualShape(
                    p.GEOM_MESH, fileName=obj_path,
                    meshScale=[1, 1, 1],
                    physicsClientId=physics_client,
                )
                body_id = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[0, 0, 0],
                    physicsClientId=physics_client,
                )

            actors.append(body_id)

        if logger:
            logger.info(f"  PyBullet: 加载 {len(actors)} 个GLB物体 (s_inv={s_inv:.4f})")
        return actors

    except Exception as e:
        if logger:
            logger.error(f"  PyBullet GLB加载失败: {e}")
        return []
