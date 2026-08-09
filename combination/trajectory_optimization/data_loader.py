"""data_loader.py — 数据加载: GLB 场景 + MANO/HaWoR + 物体查找 + 相机 + 数学工具 + 控制器

从 grasp_hawor.py 抽出, 共享数据加载与坐标变换基础设施。
"""
import os
import sys
import gc
import logging
from pathlib import Path

import numpy as np
import cv2
import sapien
import torch
from pytransform3d import rotations as pr

try:
    import trimesh
except ImportError:
    trimesh = None

# 从 physics_env 导入共享常量 (避免重复定义, physics_env 不依赖 data_loader, 无循环引用)
from physics_env import (
    R_x, R_AXIS, RXWORLD_TO_SAPIEN,
    OBJECT_DENSITY, OBJECT_MIN_MASS,
    GRIPPER_INIT_OPEN, GRIPPER_MAX_OPEN,
    FINGER_BASE_DIST,
    _FINGER1_ORIGIN, _FINGER1_AXIS, _FINGER2_ORIGIN, _FINGER2_AXIS,
)

logger = logging.getLogger("grasp_hawor")


# ============================================================
# 数学工具
# ============================================================
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


def rotation_distance(R1, R2):
    """计算两个旋转矩阵之间的角度差 (rad), 范围 [0, pi]."""
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return float(angle)


# IK / 平滑
IK_SOLVE_PER_FRAME = 20
IK_TOLERANCES = [0.1] * 6
LP_ALPHA_JOINT = 0.5
WARMUP_FRAMES = 30

# 渲染
CAM_WIDTH = 1920
CAM_HEIGHT = 1080
HAWOR_FOCAL_DEFAULT = 600.0


# ============================================================
# 相机
# ============================================================
def hawor_cam_to_sapien_pose(R_c2w, t_c2w, R_h2g=None, t_h2g=None, s=1.0):
    """HaWoR 相机位姿 → SAPIEN 相机位姿 (与 render_quick.py / hand_track/common.py 一致)

    变换链:
      1. HaWoR SLAM → GLB: cam_glb = s * R_h2g @ t_c2w + t_h2g  (无 Rx_hand)
      2. GLB → SAPIEN: R_AXIS @ cam_glb
      3. OpenGL 约定 (Z=后方) → SAPIEN 相机约定 (Z=上方)

    当传入 R_h2g/t_h2g/s 时做 001 对齐; 不传时直接 RXWORLD_TO_SAPIEN (backward compat).
    """
    if R_h2g is not None and t_h2g is not None:
        cam_pos_glb = s * R_h2g @ t_c2w + t_h2g
        R_cam_glb = R_h2g @ R_c2w
    else:
        cam_pos_glb = t_c2w
        R_cam_glb = R_c2w
    cam_pos_sapien = R_AXIS @ cam_pos_glb
    cam_R_sapien = R_AXIS @ R_cam_glb

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


def make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
    """计算 look-at 相机姿态四元数 (对齐 002_render_scene.py L1016)"""
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


# ============================================================
# GLB 加载
# ============================================================
def compute_glb_ground_z(glb_path, transform_params_path):
    """预扫描 GLB, 返回 SAPIEN 坐标系下最低点 z (用于设置地面高度)

    变换链与 load_glb_with_physics 严格一致 (对齐 002_render_scene.py):
      p_hawor = s_inv * (R_inv @ p_ras) + t_inv   (GLB → HaWoR SLAM world)
      p_sapien = RXWORLD_TO_SAPIEN @ p_hawor       (SLAM → SAPIEN)
    """
    if trimesh is None:
        return 0.0
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_h2g = params['R_hand_to_glb']  # 002 链: 使用 R_h2g.T (转置)
    t_h2g = params['t_hand_to_glb']
    trimesh_scene = trimesh.load(str(glb_path))
    all_min_z = []
    # 建立 geom_key → 原始 node_name 映射 (避免使用 trimesh 自动生成的 geometry_N)
    geom_to_node = {}
    for _node_name in trimesh_scene.graph.nodes:
        if _node_name == "world":
            continue
        try:
            _data = trimesh_scene.graph[_node_name]
            if isinstance(_data, tuple) and len(_data) == 2:
                _, _geom_key = _data
                if _geom_key:
                    geom_to_node[_geom_key] = _node_name
        except Exception:
            pass

    for geom_name, geom in trimesh_scene.geometry.items():
        real_name = geom_to_node.get(geom_name, geom_name)
        # v4.8 修复: 与 load_glb_with_physics 一致, 跳过非三角网格 (如 Path3D/点云)
        # 之前未过滤导致 SAPIEN 地面被设到非物理 mesh 的 z (如 0.0942), 物体悬空掉落
        if not hasattr(geom, 'faces') or not hasattr(geom, 'vertices'):
            continue
        vertices = geom.vertices.copy()
        faces = geom.faces if hasattr(geom, 'faces') else None
        if len(vertices) == 0 or faces is None or len(faces) == 0:
            continue
        # 002 链: p_slam = s_inv * R_h2g.T @ (v - t_h2g)
        #         p_sapien = R_AXIS @ p_slam  (R_x = I, 所以 RXWORLD_TO_SAPIEN = R_AXIS)
        vertices_hawor = s_inv * (R_h2g.T @ (vertices - t_h2g).T).T
        vertices_sapien = (R_AXIS @ vertices_hawor.T).T
        all_min_z.append(float(vertices_sapien[:, 2].min()))
    return min(all_min_z) if all_min_z else 0.0


def load_glb_with_physics(glb_path, transform_params_path, scene, fast_collision=True):
    """加载 GLB 场景并创建带碰撞体的物理物体

    对齐 002_render_scene.py 的加载方式:
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
    R_h2g = params['R_hand_to_glb']  # 002 链: 使用 R_h2g.T (转置)
    t_h2g = params['t_hand_to_glb']

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

    # 建立 geom_key → 原始 node_name 映射 (避免使用 trimesh 自动生成的 geometry_N)
    geom_to_node = {}
    for _node_name in trimesh_scene.graph.nodes:
        if _node_name == "world":
            continue
        try:
            _data = trimesh_scene.graph[_node_name]
            if isinstance(_data, tuple) and len(_data) == 2:
                _, _geom_key = _data
                if _geom_key:
                    geom_to_node[_geom_key] = _node_name
        except Exception:
            pass

    for geom_idx, (geom_name, geom) in enumerate(trimesh_scene.geometry.items()):
        real_name = geom_to_node.get(geom_name, geom_name)
        if not hasattr(geom, 'faces') or not hasattr(geom, 'vertices'):
            continue  # 跳过非三角网格 (如 Path3D)
        vertices = geom.vertices.copy()
        faces = geom.faces.copy()
        if len(vertices) == 0 or len(faces) == 0:
            continue

        # 002 链: p_slam = s_inv * R_h2g.T @ (v - t_h2g)
        #         p_sapien = R_AXIS @ p_slam  (R_x = I, 所以 RXWORLD_TO_SAPIEN = R_AXIS)
        # 与 002_render_scene.py 的 load_glb_to_sapien 一致: GLB 顶点 → R_AXIS → SAPIEN
        vertices_hawor = s_inv * (R_h2g.T @ (vertices - t_h2g).T).T
        vertices_sapien = (R_AXIS @ vertices_hawor.T).T

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
            # v4.7: 可抓取物体摩擦对齐 grasp_demo.py create_box 默认 (0.5/0.5/0.6)
            # v4.12: 摩擦 0.5 → 1.0, 增大摩擦力让 close 阶段物体不脱离夹爪
            # 物体-夹爪有效摩擦 = min(物体摩擦, 夹爪摩擦) = min(1.0, 1.0) = 1.0 (之前 0.5)
            # v4.14: restitution 0.3 → 0.0 (碰撞不反弹, close 阶段物体更稳定不被弹开)
            # v4.14e: 摩擦 1.0 → 2.0 (与夹爪摩擦一致, close 阶段物体被夹更紧跟随 base 移动)
            # v4.14g: 摩擦 3.0 退步 (物体被卡住推不开 f1), 保持 2.0
            # v4.14m: 摩擦 2.5 与 2.0 无差别 (v4.14k base_z 锁定后物体 xy 跟随主要受摩擦力影响, 但 2.5 vs 2.0 几乎一致)
            phys_material = scene.create_physical_material(
                static_friction=2.0, dynamic_friction=2.0, restitution=0.0
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
                logger.warning(f"    {real_name}: 凸包碰撞失败 ({e}), 尝试非凸")
                try:
                    builder.add_nonconvex_collision_from_file(filename=temp_ply, material=phys_material)
                except Exception as e2:
                    logger.warning(f"    {real_name}: 碰撞体生成失败 ({e2})")

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
            logger.info(f"    物体{geom_idx} '{real_name}': {body_type} "
                        f"(vol={volume:.4f}m³, flat={flatness:.2f}, mass={obj_mass:.3f}kg)")
        else:
            logger.info(f"    物体{geom_idx} '{real_name}': {body_type} "
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
# HaWoR 加载
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
TARGET_GRASP_FORCE = 10.0    # 目标夹紧力 (N), 由 MANO curl 动态调整 (6→10 增强夹持)
FORCE_CLOSE_STEP = 0.0015    # 力控阶段每帧闭合步长 (m), 1.5mm/帧 (1→1.5 更快达到目标力)
MAX_FORCE_MULTIPLIER = 2.0   # 最大力度倍率 (相对 TARGET_GRASP_FORCE)
# 接触后固定夹紧偏移 (关键: 防止持续闭合把物体挤出)
# 接触后只在 qpos_at_contact 基础上再闭合固定量 (由 MANO curl 决定, max 3mm)
# 旧版每帧闭合 1.5mm → 10 帧闭合 15mm → 物体被挤出飞出 (glb_6 xy_drift=389cm)
CLAMP_OFFSET_MAX = 0.005     # 最大额外闭合 5mm (curl=1.0 时), curl=0.5 时 2.5mm
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
# 物体查找 (按颜色/几何识别)
# ============================================================
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
        # 碗: 大体积 + 扁平 (第二十七轮: 提高阈值, 避免小盘子/盖子被误判为碗, 默认只做抓取不做放置)
        if volume > 1e-3 and flatness < 0.35:
            bowlness = volume * (1.0 - flatness)
            candidates.append((name, bowlness, volume, flatness))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    best = candidates[0]
    logger.info(f"  [find_bowl] 碗候选: "
                f"{[(c[0], f'vol={c[2]:.4f}', f'flat={c[3]:.3f}', f'score={c[1]:.6f}') for c in candidates]}")
    logger.info(f"  [find_bowl] 选中: {best[0]} (vol={best[2]:.4f}, flat={best[3]:.3f})")
    return best[0]


# ============================================================
# 混合抓取控制器: MANO 参数驱动 + 接触力控
# ============================================================
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
        from physics_env import get_finger_contacts, get_grasp_force
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


__all__ = [
    # 数学工具
    "rotmat_to_zyx_euler",
    "rotation_distance",
    # IK / 平滑 / 渲染常量
    "IK_SOLVE_PER_FRAME",
    "IK_TOLERANCES",
    "LP_ALPHA_JOINT",
    "WARMUP_FRAMES",
    "CAM_WIDTH",
    "CAM_HEIGHT",
    "HAWOR_FOCAL_DEFAULT",
    # 相机
    "hawor_cam_to_sapien_pose",
    "make_look_at_camera",
    # GLB 加载
    "compute_glb_ground_z",
    "load_glb_with_physics",
    # HaWoR / MANO
    "load_hawor_data",
    "load_hawor_c2w",
    "compute_mano_joints",
    "compute_analytical_gripper_pose",
    # 平滑滤波
    "JointFilter",
    # 力控常量
    "GRASP_TRIGGER_CURL",
    "RELEASE_TRIGGER_CURL",
    "GRASP_RESET_CURL",
    "TARGET_GRASP_FORCE",
    "FORCE_CLOSE_STEP",
    "MAX_FORCE_MULTIPLIER",
    "CLAMP_OFFSET_MAX",
    "CLAMP_CURL_FLOOR",
    "FORCE_ESTIMATE_COEFF",
    # 控制器
    "AdaptiveGraspController",
    "HybridGraspController",
    # 物体查找
    "find_target_object_by_trajectory",
    "find_pink_object",
    "find_bowl",
]
