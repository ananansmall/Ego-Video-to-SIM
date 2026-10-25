"""
物理仿真场景构建模块

管线位置:
  合并管线输出 (场景GLB + 物体mesh + 手部轨迹)
      → scene_builder.py (本文件)
      → action_player.py
      → run_simulation.py

输入:
  - final_scene_basic.glb     静态场景 (墙壁/地面/桌面)
  - 物体 mesh + 初始6DoF  (来自 ReplicateAnyScene + object_tracker)
  - hawor_results.npz   手部轨迹 (来自 HaWoR)
  - action_sequence.json 夹爪时序 (来自 action_semantics)

输出:
  - SAPIEN Scene 对象, 包含静态场景 + 动态物体 + R1Lite 机器人
  - 物体 actor 列表 (用于后续读取位姿验证)

坐标系:
  房间坐标系 (z-up) → SAPIEN (y-up) 转换:
    T = [[1,0,0,0], [0,0,1,0], [0,-1,0,0], [0,0,0,1]]
"""

import numpy as np
import sapien
from scipy.spatial.transform import Rotation

GALAXEA_SIM_DIR = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/GalaxeaManipSim"

T_ROOM_TO_SAPIEN = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def room_pose_to_sapien(position_room, rotation_room=None):
    """房间坐标系位姿 → SAPIEN 位姿

    Args:
        position_room: (3,) 房间坐标系位置 (z-up, 米)
        rotation_room: (3,3) 旋转矩阵 或 (4,) 四元数 或 None

    Returns:
        sapien.Pose
    """
    p_homo = np.append(position_room, 1.0)
    p_sapien = (T_ROOM_TO_SAPIEN @ p_homo)[:3]

    if rotation_room is not None:
        if rotation_room.shape == (4,):
            R_room = Rotation.from_quat(rotation_room).as_matrix()
        else:
            R_room = rotation_room
        T_room = np.eye(4)
        T_room[:3, :3] = R_room
        T_sapien = T_ROOM_TO_SAPIEN @ T_room @ T_ROOM_TO_SAPIEN.T
        R_sapien = T_sapien[:3, :3]
        q_sapien = Rotation.from_matrix(R_sapien).as_quat()
        return sapien.Pose(p=p_sapien, q=q_sapien)
    else:
        return sapien.Pose(p=p_sapien)


def build_scene(
    scene_glb_path=None,
    objects=None,
    robot_type="r1_lite",
    robot_base_pose_room=None,
    timestep=1.0 / 240.0,
):
    """构建完整的 SAPIEN 仿真场景

    Args:
        scene_glb_path: 静态场景 GLB 文件路径 (墙壁/地面/桌面)
        objects: 物体列表, 每个元素:
            {
                "mesh_path": str,       # GLB/OBJ 文件路径
                "position": (3,),       # 初始位置 (房间坐标系)
                "rotation": (3,3) | None,  # 初始旋转
                "half_size": (3,) | None,  # 碰撞半尺寸 (简化碰撞)
                "mass": float,          # 质量 (kg)
                "name": str,            # 物体名称
                "is_static": bool,      # 是否静态
            }
        robot_type: 机器人型号 ("r1_lite")
        robot_base_pose_room: 机器人基座位姿 (房间坐标系), 默认原点
        timestep: 物理仿真时间步

    Returns:
        dict: {
            "scene": sapien.Scene,
            "robot": BimanualRobot,
            "object_actors": {name: sapien.Entity},
            "static_actors": [sapien.Entity],
        }
    """
    scene = sapien.Scene()
    scene.set_timestep(timestep)
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 0, -1], [1, 1, 1])

    static_actors = []
    object_actors = {}

    if scene_glb_path is not None:
        static_actor = _load_static_scene(scene, scene_glb_path)
        if static_actor is not None:
            static_actors.append(static_actor)

    if objects is not None:
        for obj in objects:
            actor = _create_dynamic_object(scene, obj)
            if actor is not None:
                object_actors[obj["name"]] = actor

    robot = _load_robot(scene, robot_type, robot_base_pose_room)

    return {
        "scene": scene,
        "robot": robot,
        "object_actors": object_actors,
        "static_actors": static_actors,
    }


def _load_static_scene(scene, glb_path):
    """加载静态场景 GLB 为静态碰撞体"""
    import os
    if not os.path.exists(glb_path):
        print(f"[WARN] 静态场景文件不存在: {glb_path}")
        return None

    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    try:
        builder.add_multiple_convex_collisions_from_file(
            filename=glb_path, decomposition="coacd"
        )
        builder.add_visual_from_file(filename=glb_path)
        actor = builder.build(name="static_scene")
        actor.set_pose(sapien.Pose([0, 0, 0]))
        return actor
    except Exception as e:
        print(f"[WARN] 加载静态场景失败: {e}")
        return None


def _create_dynamic_object(scene, obj_desc):
    """创建动态物体 (刚体)"""
    name = obj_desc.get("name", "unknown")
    position = obj_desc.get("position", [0, 0, 0])
    rotation = obj_desc.get("rotation", None)
    is_static = obj_desc.get("is_static", False)
    mass = obj_desc.get("mass", 0.5)
    mesh_path = obj_desc.get("mesh_path", None)
    half_size = obj_desc.get("half_size", None)

    pose = room_pose_to_sapien(np.array(position), rotation)

    if half_size is not None:
        return _create_box_object(scene, pose, half_size, name, is_static, mass)
    elif mesh_path is not None:
        return _create_mesh_object(scene, pose, mesh_path, name, is_static, mass)
    else:
        print(f"[WARN] 物体 {name} 缺少 mesh_path 或 half_size")
        return None


def _create_box_object(scene, pose, half_size, name, is_static, mass):
    """用简化碰撞盒创建物体"""
    entity = sapien.Entity()
    entity.set_name(name)
    entity.set_pose(pose)

    rigid = sapien.physx.PhysxRigidDynamicComponent()
    if is_static:
        rigid = sapien.physx.PhysxRigidStaticComponent()

    material = scene.create_physical_material(0.5, 0.5, 0.6)
    collision = sapien.physx.PhysxCollisionShapeBox(
        half_size=half_size, material=material
    )
    rigid.attach(collision)

    if not is_static:
        rigid.set_mass(mass)

    render = sapien.render.RenderBodyComponent()
    render_material = sapien.render.RenderMaterial(base_color=[0.8, 0.6, 0.3, 1.0])
    render_shape = sapien.render.RenderShapeBox(half_size, render_material)
    render.attach(render_shape)

    entity.add_component(rigid)
    entity.add_component(render)
    scene.add_entity(entity)
    return entity


def _create_mesh_object(scene, pose, mesh_path, name, is_static, mass):
    """用 GLB/OBJ mesh 创建物体"""
    import os
    if not os.path.exists(mesh_path):
        print(f"[WARN] 物体 mesh 不存在: {mesh_path}")
        return None

    builder = scene.create_actor_builder()
    if is_static:
        builder.set_physx_body_type("static")
    else:
        builder.set_physx_body_type("dynamic")

    try:
        builder.add_multiple_convex_collisions_from_file(
            filename=mesh_path, decomposition="coacd"
        )
        builder.add_visual_from_file(filename=mesh_path)
        actor = builder.build(name=name)
        actor.set_pose(pose)
        return actor
    except Exception as e:
        print(f"[WARN] 加载物体 mesh 失败: {e}")
        return None


def _load_robot(scene, robot_type, base_pose_room):
    """加载 R1Lite 机器人"""
    import sys
    sys.path.insert(0, GALAXEA_SIM_DIR)

    if robot_type == "r1_lite":
        from galaxea_sim.robots.r1_lite import R1LiteRobot

        if base_pose_room is None:
            base_pose_room = {"position": [0, 0, 0], "rotation": None}

        base_pose = room_pose_to_sapien(
            np.array(base_pose_room["position"]),
            base_pose_room.get("rotation"),
        )

        robot = R1LiteRobot(
            scene,
            robot_origin_xyz=base_pose.p,
            robot_origin_quat=base_pose.q,
        )
        return robot
    else:
        raise ValueError(f"不支持的机器人型号: {robot_type}")
