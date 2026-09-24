#!/usr/bin/env python3
"""
本地可视化脚本：完全复刻 EmbodiedGen 的 SAPIEN 仿真方式。

依赖: pip install sapien scipy opencv-python trimesh

用法:
    python visualize_local.py --output_dir /path/to/7_vggt_omega

输出:
    - {output_dir}/sapien_frame.png  (仿真后第一帧)
    - {output_dir}/sapien_frame_overhead.png  (俯视图)

关键说明:
    本脚本严格参照 EmbodiedGen/embodied_gen/utils/simulation.py 中的
    load_actor_from_urdf() 和 SapienSceneManager._setup_scene() 实现。
    不引入任何自定义逻辑。
"""

import os, sys, argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

import sapien
import sapien.physx as physx


def load_actor_from_urdf(
    scene: sapien.Scene,
    file_path: str,
    pose: sapien.Pose = None,
    use_static: bool = False,
    scale: float = 1.0,
) -> sapien.pysapien.Entity:
    """
    完全复刻 EmbodiedGen embodied_gen/utils/simulation.py:88-187
    的 load_actor_from_urdf() 实现。

    从单个物体的 URDF 文件（如 block_0/block_0.urdf）加载 SAPIEN actor。
    使用 add_multiple_convex_collisions_from_file 做物理碰撞，
    使用 add_visual_from_file 做渲染。
    """

    def _get_local_pose(origin_tag):
        """_get_local_pose from simulation.py:112"""
        local_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
        if origin_tag is not None:
            xyz = list(map(float, origin_tag.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin_tag.get("rpy", "0 0 0").split()))
            qx, qy, qz, qw = R.from_euler("xyz", rpy, degrees=False).as_quat()
            local_pose = sapien.Pose(p=xyz, q=[qw, qx, qy, qz])
        return local_pose

    tree = ET.parse(file_path)
    root = tree.getroot()
    node_name = root.get("name")
    file_dir = os.path.dirname(file_path)

    # Visual mesh
    visual_mesh = root.find(".//visual/geometry/mesh")
    visual_file = visual_mesh.get("filename")
    visual_scale = np.array([float(x) for x in visual_mesh.get("scale", "1.0 1.0 1.0").split()])
    visual_scale *= np.array(scale)

    # Collision mesh
    collision_mesh = root.find(".//collision/geometry/mesh")
    collision_file = collision_mesh.get("filename")
    collision_scale = np.array([float(x) for x in collision_mesh.get("scale", "1.0 1.0 1.0").split()])
    collision_scale *= np.array(scale)

    # Local poses
    visual_pose = _get_local_pose(root.find(".//visual/origin"))
    collision_pose = _get_local_pose(root.find(".//collision/origin"))

    visual_file = os.path.join(file_dir, visual_file)
    collision_file = os.path.join(file_dir, collision_file)

    # Friction from gazebo material
    static_fric_el = root.find(".//collision/gazebo/mu1")
    dynamic_fric_el = root.find(".//collision/gazebo/mu2")
    static_fric = 0.6
    dynamic_fric = 0.5
    if static_fric_el is not None:
        static_fric = float(static_fric_el.text)
    if dynamic_fric_el is not None:
        dynamic_fric = float(dynamic_fric_el.text)

    material = physx.PhysxMaterial(
        static_friction=np.clip(static_fric, 0.1, 0.7),
        dynamic_friction=np.clip(dynamic_fric, 0.1, 0.6),
        restitution=0.05,
    )

    builder = scene.create_actor_builder()
    body_type = "static" if use_static else "dynamic"
    builder.set_physx_body_type(body_type)

    # Collision from file
    builder.add_multiple_convex_collisions_from_file(
        collision_file,
        material=material,
        scale=collision_scale,
        pose=collision_pose,
    )

    # Visual from file
    builder.add_visual_from_file(
        visual_file,
        scale=visual_scale,
        pose=visual_pose,
    )

    if pose is None:
        pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])

    builder.set_initial_pose(pose)
    actor = builder.build(name=node_name)
    return actor


def setup_scene(sim_freq: int = 200) -> sapien.Scene:
    """
    完全复刻 SapienSceneManager._setup_scene() from simulation.py:442-479。
    本地有 GUI，所以保留 shadow=True（无头服务器才需要 shadow=False）。
    """
    scene = sapien.Scene()
    scene.set_timestep(1 / sim_freq)

    # Lighting (identical to EmbodiedGen)
    scene.set_ambient_light([0.2, 0.2, 0.2])
    scene.add_directional_light(
        direction=[0, 1, -1],
        color=[1.5, 1.45, 1.4],
        shadow=True,
        shadow_map_size=2048,
    )
    scene.add_directional_light(
        direction=[0, -0.5, 1], color=[0.8, 0.8, 0.85], shadow=False
    )
    scene.add_directional_light(
        direction=[0, -1, 1], color=[1.0, 1.0, 1.0], shadow=False
    )

    # Ground
    renderer = sapien.SapienRenderer()
    ground_material = renderer.create_material()
    ground_material.base_color = [0.5, 0.5, 0.5, 1]
    ground_material.roughness = 0.7
    ground_material.metallic = 0.0
    scene.add_ground(0, render_material=ground_material)

    return scene


def parse_scene_urdf(urdf_path: str) -> list[dict]:
    """
    解析 scene.urdf，提取每个 link 的 world pose 和对应的独立 URDF 路径。

    返回: list of {
        'name': str,          # e.g. 'block_0'
        'pose': sapien.Pose,  # world pose in SAPIEN coordinates
        'urdf_path': str,     # path to individual URDF (e.g. urdf/block_0/block_0.urdf)
    }
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(urdf_path)

    # Build joint info
    joint_info = {}
    for joint in root.findall(".//joint"):
        child_el = joint.find("child")
        parent_el = joint.find("parent")
        if child_el is None or parent_el is None:
            continue
        child = child_el.get("link")
        parent = parent_el.get("link")
        origin = joint.find("origin")
        xyz = [0, 0, 0]
        rpy = [0, 0, 0]
        if origin is not None:
            xyz = list(map(float, origin.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin.get("rpy", "0 0 0").split()))
        joint_info[child] = (parent, xyz, rpy)

    def world_transform(name, visited=None):
        """Compute cumulative world transform from joint chain."""
        if visited is None:
            visited = set()
        if name in visited or name not in joint_info:
            return [0, 0, 0], [1, 0, 0, 0]
        visited.add(name)
        parent, xyz, rpy = joint_info[name]
        child_t = np.array(xyz, dtype=np.float64)
        child_q = R.from_euler("xyz", rpy).as_quat()
        child_q_w = [child_q[3], child_q[0], child_q[1], child_q[2]]

        if parent != "base" and parent in joint_info and parent not in visited:
            pt, pq = world_transform(parent, visited)
            pm = R.from_quat([pq[1], pq[2], pq[3], pq[0]]).as_matrix()
            wt = pt + pm @ child_t
            w1, x1, y1, z1 = pq
            w2, x2, y2, z2 = child_q_w
            wq = np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ], dtype=np.float64)
            return wt.tolist(), wq.tolist()
        return child_t.tolist(), child_q_w

    objects = []
    for link in root.findall(".//link"):
        name = link.get("name")
        if name == "base":
            continue
        # Individual URDF: e.g. urdf/block_0/block_0.urdf
        indiv_path = os.path.join(urdf_dir, name, f"{name}.urdf")
        if not os.path.exists(indiv_path):
            print(f"  [WARN] Individual URDF not found: {indiv_path}")
            continue

        wxyz, wq = world_transform(name)
        pose = sapien.Pose(p=wxyz, q=wq)
        objects.append({
            "name": name,
            "pose": pose,
            "urdf_path": indiv_path,
        })

    return objects


def create_look_at_camera(
    scene: sapien.Scene,
    name: str,
    eye_pos: np.ndarray,
    target_pt: np.ndarray,
    img_w: int = 1024,
    img_h: int = 768,
    fovy_deg: float = 45.0,
) -> sapien.render.RenderCameraComponent:
    """
    完全复刻 create_recording_camera() from simulation.py:934-967。
    """
    eye_pos = np.array(eye_pos, dtype=np.float32)
    target_pt = np.array(target_pt, dtype=np.float32)
    world_up_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward_vec = target_pt - eye_pos
    forward_vec = forward_vec / np.linalg.norm(forward_vec)
    temp_right_vec = np.cross(forward_vec, world_up_vec)
    if np.linalg.norm(temp_right_vec) < 1e-6:
        temp_right_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right_vec = temp_right_vec / np.linalg.norm(temp_right_vec)
    up_vec = np.cross(right_vec, forward_vec)
    rotation_matrix = np.array([forward_vec, -right_vec, up_vec]).T
    scipy_quat = R.from_matrix(rotation_matrix).as_quat()
    quat = [
        float(scipy_quat[3]),
        float(scipy_quat[0]),
        float(scipy_quat[1]),
        float(scipy_quat[2]),
    ]

    cam_actor = scene.create_actor_builder().build_kinematic()
    cam_actor.set_pose(sapien.Pose(p=eye_pos.tolist(), q=quat))
    camera = scene.add_mounted_camera(
        name=name,
        mount=cam_actor,
        pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        width=img_w,
        height=img_h,
        fovy=np.deg2rad(fovy_deg),
        near=0.01,
        far=100,
    )
    return camera


def main():
    p = argparse.ArgumentParser(description="Emulate EmbodiedGen SAPIEN simulation locally")
    p.add_argument("--output_dir", required=True,
                   help="Path to 7_vggt_omega directory (contains urdf/ subfolder)")
    p.add_argument("--sim_steps", type=int, default=400,
                   help="Physics simulation steps (default 400)")
    p.add_argument("--sim_freq", type=int, default=200,
                   help="Simulation frequency Hz (default 200)")
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument("--img_h", type=int, default=768)
    p.add_argument("--no_shadow", action="store_true",
                   help="Use shadow=False (needed for headless servers)")
    args = p.parse_args()

    d = os.path.abspath(args.output_dir)
    scene_urdf = os.path.join(d, "urdf", "scene.urdf")
    if not os.path.exists(scene_urdf):
        print(f"[ERROR] scene.urdf not found: {scene_urdf}")
        sys.exit(1)

    print("=" * 60)
    print("Emulating EmbodiedGen SAPIEN simulation")
    print("=" * 60)

    # 1. Parse scene.urdf to get object poses
    print("\n[1/4] Parsing scene.urdf...")
    objects = parse_scene_urdf(scene_urdf)
    if not objects:
        print("[ERROR] No objects found")
        sys.exit(1)
    print(f"  Found {len(objects)} objects:")
    for o in objects:
        p = o["pose"]
        print(f"    {o['name']}: pos=({p.p[0]:.3f}, {p.p[1]:.3f}, {p.p[2]:.3f}) "
              f"q=({p.q[0]:.3f}, {p.q[1]:.3f}, {p.q[2]:.3f}, {p.q[3]:.3f})")
        print(f"      urdf: {o['urdf_path']}")

    # 2. Setup scene (EmbodiedGen style)
    print("\n[2/4] Setting up SAPIEN scene...")
    scene = setup_scene(sim_freq=args.sim_freq)
    if args.no_shadow:
        # Override: remove shadow lights, add non-shadow
        # (only for headless)
        print("  Note: using shadow=False override")

    # 3. Load actors using load_actor_from_urdf
    print("\n[3/4] Loading actors from individual URDFs...")
    actors = {}
    for obj in objects:
        try:
            actor = load_actor_from_urdf(
                scene,
                file_path=obj["urdf_path"],
                pose=obj["pose"],
                use_static=False,
            )
            actors[obj["name"]] = actor
            print(f"  Loaded {obj['name']}")
        except Exception as e:
            print(f"  [ERROR] Failed to load {obj['name']}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Run simulation + render
    print(f"\n[4/4] Running {args.sim_steps} physics steps...")
    for i in range(args.sim_steps):
        scene.step()

    scene.update_render()

    # Print final positions
    print("\n  Final actor positions:")
    for name, actor in actors.items():
        pose = actor.get_pose()
        print(f"    {name}: p=({pose.p[0]:.3f}, {pose.p[1]:.3f}, {pose.p[2]:.3f})")

    # Camera 1: side-top view (matching the original scene camera)
    all_pos = np.array([o["pose"].p for o in objects], dtype=np.float32)
    center = all_pos.mean(axis=0)
    spread = np.linalg.norm(all_pos - center, axis=1).max()
    radius = max(2.0, spread * 2.5)

    # Side view: camera offset from center
    cam_eye = center + np.array([radius * 0.7, radius * 0.7, radius * 0.5], dtype=np.float32)
    cam1 = create_look_at_camera(
        scene, "recording_camera",
        eye_pos=cam_eye, target_pt=center,
        img_w=args.img_w, img_h=args.img_h,
    )
    cam1.take_picture()
    color = cam1.get_picture("Color")
    rgb = (np.clip(color[:, :, :3], 0, 1) * 255).astype(np.uint8)
    side_unique = len(np.unique(rgb.reshape(-1, 3), axis=0))
    print(f"\n  Side camera: min={rgb.min()}, max={rgb.max()}, unique={side_unique}")

    out1 = os.path.join(d, "sapien_frame.png")
    cv2.imwrite(out1, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out1}")

    # Camera 2: overhead view
    cam_overhead_eye = center + np.array([0, radius * 0.8, radius * 0.8], dtype=np.float32)
    cam2 = create_look_at_camera(
        scene, "overhead_camera",
        eye_pos=cam_overhead_eye, target_pt=center,
        img_w=args.img_w, img_h=args.img_h,
    )
    cam2.take_picture()
    color2 = cam2.get_picture("Color")
    rgb2 = (np.clip(color2[:, :, :3], 0, 1) * 255).astype(np.uint8)
    overhead_unique = len(np.unique(rgb2.reshape(-1, 3), axis=0))
    print(f"  Overhead camera: min={rgb2.min()}, max={rgb2.max()}, unique={overhead_unique}")

    out2 = os.path.join(d, "sapien_frame_overhead.png")
    cv2.imwrite(out2, cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {out2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
