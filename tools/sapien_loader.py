#!/usr/bin/env python3
"""
SAPIEN URDF Loader - 加载 URDF 到 SAPIEN 场景
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation as R


def load_scene_urdf(
    scene: sapien.Scene,
    file_path: str,
    file_dir: str,
    name: str = "scene",
    scale: float = 1.0,
) -> list[sapien.Entity]:
    """Load a multi-link scene URDF — each non-base link becomes its own actor.

    Returns list of loaded actors (one per link with visual/collision mesh).
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    links = root.findall(".//link")
    joints = root.findall(".//joint")

    # Build joint chain: child_name -> parent_name, xyz, rpy
    joint_poses = {}
    for joint in joints:
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        origin = joint.find("origin")
        xyz = [0, 0, 0]
        rpy = [0, 0, 0]
        if origin is not None:
            xyz = [float(x) for x in origin.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in origin.get("rpy", "0 0 0").split()]
        joint_poses[child] = {"parent": parent, "xyz": xyz, "rpy": rpy}

    actors = []
    for link_elem in links:
        link_name = link_elem.get("name", name)
        if link_name == "base":
            continue

        visual = link_elem.find(".//visual/geometry/mesh")
        collision = link_elem.find(".//collision/geometry/mesh")
        if visual is None or collision is None:
            continue

        visual_file = visual.get("filename")
        collision_file = collision.get("filename")
        visual_scale = np.array([float(x) for x in visual.get("scale", "1 1 1").split()]) * scale
        collision_scale = np.array([float(x) for x in collision.get("scale", "1 1 1").split()]) * scale

        # Get cumulative pose (chain of joint transforms)
        pose = _get_link_world_pose(link_name, joint_poses)

        # Link origin transforms
        v_origin = link_elem.find(".//visual/origin")
        v_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        if v_origin is not None:
            xyz = [float(x) for x in v_origin.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in v_origin.get("rpy", "0 0 0").split()]
            qx, qy, qz, qw = R.from_euler("xyz", rpy).as_quat()
            v_pose = sapien.Pose(p=xyz, q=[qw, qx, qy, qz])

        c_origin = link_elem.find(".//collision/origin")
        c_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        if c_origin is not None:
            xyz = [float(x) for x in c_origin.get("xyz", "0 0 0").split()]
            rpy = [float(x) for x in c_origin.get("rpy", "0 0 0").split()]
            qx, qy, qz, qw = R.from_euler("xyz", rpy).as_quat()
            c_pose = sapien.Pose(p=xyz, q=[qw, qx, qy, qz])

        mu1 = 0.6
        mu2 = 0.5
        gazebo = link_elem.find(".//collision/gazebo")
        if gazebo is not None:
            m1 = gazebo.find("mu1")
            m2 = gazebo.find("mu2")
            if m1 is not None:
                mu1 = float(m1.text)
            if m2 is not None:
                mu2 = float(m2.text)

        mass = 1.0
        inertial = link_elem.find(".//inertial/mass")
        if inertial is not None:
            mass = float(inertial.get("value", 1.0))

        import sapien.physx as physx

        material = physx.PhysxMaterial(
            static_friction=max(0.1, min(0.7, mu1)),
            dynamic_friction=max(0.1, min(0.6, mu2)),
            restitution=0.05,
        )

        builder = scene.create_actor_builder()
        builder.set_name(link_name)
        builder.set_initial_pose(pose)

        builder.add_multiple_convex_collisions_from_file(
            os.path.join(file_dir, collision_file),
            material=material,
            scale=collision_scale,
            pose=c_pose,
        )

        builder.add_visual_from_file(
            os.path.join(file_dir, visual_file),
            scale=visual_scale,
            pose=v_pose,
        )

        actor = builder.build(name=link_name)

        # SAPIEN doesn't read vertex colors from OBJ by default.
        # Compute avg vertex color from OBJ and set it on the render material.
        visual_path = os.path.join(file_dir, visual_file)
        try:
            import trimesh
            vmesh = trimesh.load(visual_path)
            if vmesh.visual.defined and hasattr(vmesh.visual, 'vertex_colors'):
                avg_color = np.array(vmesh.visual.vertex_colors[:, :3]).mean(axis=0) / 255.0
                base_color = [float(avg_color[0]), float(avg_color[1]), float(avg_color[2]), 1.0]
            else:
                base_color = [0.8, 0.8, 0.8, 1.0]
        except Exception:
            base_color = [0.8, 0.8, 0.8, 1.0]

        for comp in actor.get_components():
            if "RenderBody" in type(comp).__name__:
                for shape in comp.render_shapes:
                    shape.material.base_color = base_color
                    shape.material.roughness = 0.5
                    shape.material.metallic = 0.0
        actors.append(actor)

    return actors


def _get_link_world_pose(link_name: str, joint_poses: dict) -> sapien.Pose:
    """Compute world pose of a link by chaining parent joint transforms."""
    total_translation = np.array([0.0, 0.0, 0.0])
    total_quat = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
    visited = set()

    while link_name in joint_poses and link_name not in visited:
        visited.add(link_name)
        jp = joint_poses[link_name]
        t = np.array(jp["xyz"], dtype=np.float64)
        r = jp["rpy"]
        qx, qy, qz, qw = R.from_euler("xyz", r).as_quat()
        joint_q = np.array([qw, qx, qy, qz], dtype=np.float64)  # w, x, y, z

        # Apply parent rotation to local translation, then add
        parent_R = R.from_quat([total_quat[1], total_quat[2], total_quat[3], total_quat[0]])
        total_translation = total_translation + parent_R.as_matrix() @ t
        total_quat = _quat_multiply(total_quat, joint_q)
        link_name = jp["parent"]

    return sapien.Pose(p=total_translation, q=total_quat)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
