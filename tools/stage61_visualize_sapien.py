#!/usr/bin/env python3
"""
Stage 6.1 Viewer — SAPIEN physics simulation + colored box rendering.
Uses SAPIEN primitives (boxes) as colored visual proxies since OBJ vertex
colors are not rendered by SAPIEN 3.0.0b1.

Physics still uses the real OBJ collision meshes from URDF.
"""

import argparse
import os
import sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R

import sapien
import sapien.physx as physx

# Distinct colors for objects
OBJ_COLORS = [
    [1.0, 0.2, 0.2, 1.0],
    [0.2, 0.8, 0.2, 1.0],
    [0.2, 0.2, 1.0, 1.0],
    [1.0, 0.8, 0.1, 1.0],
    [1.0, 0.4, 0.8, 1.0],
    [0.2, 1.0, 1.0, 1.0],
]


def parse_urdf(urdf_path):
    """Parse scene.urdf and return list of object dicts."""
    file_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joint_info = {}
    for joint in root.findall(".//joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            xyz = list(map(float, origin.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin.get("rpy", "0 0 0").split()))
        joint_info[child] = (parent, xyz, rpy)

    def world_transform(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited or name not in joint_info:
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        visited.add(name)
        parent, xyz, rpy = joint_info[name]
        child_t = np.array(xyz, dtype=np.float64)
        child_q = R.from_euler("xyz", rpy).as_quat()
        child_q_w = [child_q[3], child_q[0], child_q[1], child_q[2]]

        if parent != "base" and parent in joint_info and parent not in visited:
            pt, pqw = world_transform(parent, visited)
            pm = R.from_quat([pqw[1], pqw[2], pqw[3], pqw[0]]).as_matrix()
            wt = pt + pm @ child_t
            w1, x1, y1, z1 = pqw
            w2, x2, y2, z2 = child_q_w
            wq = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ], dtype=np.float64)
            return wt.tolist(), wq.tolist()
        return child_t.tolist(), child_q_w

    objects = []
    for link in root.findall(".//link"):
        name = link.get("name")
        if name == "base":
            continue
        visual = link.find(".//visual/geometry/mesh")
        if visual is None:
            continue
        mesh_file = visual.get("filename")
        if not mesh_file:
            continue
        mesh_path = os.path.join(file_dir, mesh_file)
        if not os.path.exists(mesh_path):
            continue

        scale = np.array([float(x) for x in visual.get("scale", "1 1 1").split()])
        wxyz, wq = world_transform(name)

        # Get collision mesh path
        coll = link.find(".//collision/geometry/mesh")
        coll_path = None
        if coll is not None:
            cf = coll.get("filename")
            if cf:
                coll_path = os.path.join(file_dir, cf)

        # Get friction
        static_fric, dynamic_fric = 0.6, 0.5
        gaz = link.find(".//collision/gazebo")
        if gaz is not None:
            try:
                m1 = gaz.find("mu1")
                m2 = gaz.find("mu2")
                if m1 is not None:
                    static_fric = float(m1.text)
                if m2 is not None:
                    dynamic_fric = float(m2.text)
            except:
                pass

        objects.append({
            "name": name,
            "mesh_path": mesh_path,
            "coll_path": coll_path,
            "scale": scale,
            "world_pos": wxyz,
            "world_quat": wq,
            "static_fric": static_fric,
            "dynamic_fric": dynamic_fric,
        })

    return objects


def setup_scene():
    scene = sapien.Scene()
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light(
        direction=[0, 1, -1], color=[2.0, 2.0, 2.0], shadow=True, shadow_map_size=2048
    )
    scene.add_directional_light(
        direction=[-1, -0.5, 1], color=[0.8, 0.8, 0.8], shadow=False
    )
    scene.add_directional_light(
        direction=[1, -0.5, 1], color=[0.8, 0.8, 0.8], shadow=False
    )

    # Ground
    renderer = sapien.SapienRenderer()
    ground_mat = renderer.create_material()
    ground_mat.base_color = [0.25, 0.25, 0.25, 1.0]
    ground_mat.roughness = 0.9
    ground_mat.metallic = 0.0
    scene.add_ground(0, render_material=ground_mat)

    return scene


def compute_half_size(mesh_path, scale):
    """Read OBJ vertices and compute bounding box half size."""
    verts = []
    with open(mesh_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if len(verts) == 0:
        return [0.3, 0.3, 0.3]
    verts = np.array(verts)
    half = ((verts.max(axis=0) - verts.min(axis=0)) * scale / 2).tolist()
    # Minimum size
    half = [max(h, 0.05) for h in half]
    return half


def load_objects(scene, objects, z_offset=0.0):
    """Load objects with colored box visuals + real collision meshes."""
    renderer = sapien.SapienRenderer()

    for idx, obj in enumerate(objects):
        color = OBJ_COLORS[idx % len(OBJ_COLORS)]

        half_size = compute_half_size(obj["mesh_path"], obj["scale"])
        pos = obj["world_pos"]
        quat = obj["world_quat"]

        # Lift z if needed
        pos[2] += z_offset

        # Color material
        color_mat = renderer.create_material()
        color_mat.base_color = color
        color_mat.roughness = 0.5
        color_mat.metallic = 0.0

        builder = scene.create_actor_builder()
        builder.set_initial_pose(sapien.Pose(p=pos, q=quat))

        # Visual: colored box (SAPIEN primitive, renders correctly)
        builder.add_box_visual(half_size=half_size, material=color_mat)

        # Collision: real OBJ mesh for physics
        if obj["coll_path"] and os.path.exists(obj["coll_path"]):
            mat = physx.PhysxMaterial(
                static_friction=np.clip(obj["static_fric"], 0.1, 0.7),
                dynamic_friction=np.clip(obj["dynamic_fric"], 0.1, 0.6),
                restitution=0.05,
            )
            builder.add_multiple_convex_collisions_from_file(
                obj["coll_path"], material=mat, scale=obj["scale"]
            )
        else:
            builder.add_box_collision(half_size=half_size)

        builder.set_physx_body_type("dynamic")
        actor = builder.build(name=obj["name"])

        print(f"  {obj['name']}: pos={pos}, half_size={half_size}, "
              f"color={color[:3]}")

    return len(objects)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sim_steps", type=int, default=400)
    parser.add_argument("--n_frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--img_w", type=int, default=512)
    parser.add_argument("--img_h", type=int, default=512)
    parser.add_argument("--fovy", type=float, default=50.0)
    parser.add_argument("--elev", type=float, default=55.0)
    parser.add_argument("--radius", type=float, default=4.0)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    urdf_dir = os.path.join(output_dir, "urdf")
    urdf_path = os.path.join(urdf_dir, "scene.urdf")

    if not os.path.exists(urdf_path):
        print(f"[ERROR] scene.urdf not found: {urdf_path}")
        sys.exit(1)

    # Parse URDF
    print(f"Parsing {urdf_path}...")
    objects = parse_urdf(urdf_path)
    if not objects:
        print("[ERROR] No objects found in URDF")
        sys.exit(1)

    # Compute z_offset to lift objects above ground
    all_z = [o["world_pos"][2] for o in objects]
    min_z = min(all_z)
    z_offset = -min_z + 0.1 if min_z < 0.1 else 0.0
    print(f"  z_offset: {z_offset:.3f}m")

    # Build scene
    print("\nSetting up scene...")
    scene = setup_scene()
    scene.set_timestep(1.0 / 200)
    n_loaded = load_objects(scene, objects, z_offset=z_offset)
    print(f"Loaded {n_loaded} actors")

    # Compute camera position
    all_pos = np.array([o["world_pos"] for o in objects], dtype=np.float64)
    center = all_pos.mean(axis=0)
    max_dist = np.linalg.norm(all_pos - center, axis=1).max()
    radius = max(args.radius, max_dist * 2)
    elev_rad = np.radians(args.elev)

    # Camera: fixed side-top view, Z-up in SAPIEN
    cam_pos = np.array([
        center[0] + radius * np.cos(elev_rad),
        center[1],
        center[2] + radius * np.sin(elev_rad),
    ], dtype=np.float64)

    print(f"\nCamera: pos={cam_pos}, center={center}, radius={radius:.2f}")

    cam = scene.add_camera(
        "cam", width=args.img_w, height=args.img_h,
        fovy=np.deg2rad(args.fovy), near=0.01, far=50
    )

    # Look-at quaternion
    eye = cam_pos
    target = center
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    rot_mat = np.column_stack([right, up, -forward])
    xyzw = R.from_matrix(rot_mat).as_quat()
    qw = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]

    cam.set_entity_pose(sapien.Pose(p=eye.tolist(), q=qw))
    print(f"  quat=[{qw[0]:.3f}, {qw[1]:.3f}, {qw[2]:.3f}, {qw[3]:.3f}]")

    # Run simulation
    print(f"\nSimulating {args.sim_steps} steps, rendering {args.n_frames} frames...")
    scene.update_render()

    step_per_frame = max(args.sim_steps // args.n_frames, 1)
    frames = []

    for i in range(args.n_frames):
        for _ in range(step_per_frame):
            scene.step()

        scene.update_render()
        cam.take_picture()
        color = cam.get_picture("Color")
        rgb = np.clip(color[..., :3], 0, 1)
        frames.append((rgb * 255).astype(np.uint8))

        if i % 20 == 0:
            print(f"  Frame {i}/{args.n_frames} mean={rgb.mean():.4f}")

    # Check if we got anything other than pure gray
    sample = frames[0]
    if sample.max() == sample.min():
        print(f"\n[WARN] All pixels identical (value={sample.max()}), "
              f"render may be broken")

    # Save video
    print("\nSaving...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    mp4_path = os.path.join(output_dir, "stage61_sapien_video.mp4")
    writer = cv2.VideoWriter(mp4_path, fourcc, args.fps, (args.img_w, args.img_h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"  Video: {mp4_path}")

    # Save first frame
    png_path = os.path.join(output_dir, "stage61_sapien_frame0.png")
    cv2.imwrite(png_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    print(f"  Frame 0: {png_path}")

    # Print actor final positions
    print("\nFinal actor positions:")
    for a in scene.get_all_actors():
        if a.name in [o["name"] for o in objects]:
            print(f"  {a.name}: {a.get_pose().p}")

    print("\nDone!")


if __name__ == "__main__":
    main()
