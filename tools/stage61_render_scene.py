#!/usr/bin/env python3
"""Render scene from URDF using pyrender (for OBJ vertex colors) + fixed camera.

Usage:
    python stage61_render_scene.py --urdf urdf/scene.urdf

Outputs: stage61_render.png (fixed overhead view of all objects)
"""
import os
import sys
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R


def parse_urdf_objects(urdf_path):
    """Parse scene.urdf and return list of (mesh_path, scale, origin_xyz, origin_rpy)."""
    file_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Parse joints
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

    def world_transform(link_name, visited=None):
        if visited is None:
            visited = set()
        if link_name in visited or link_name not in joint_info:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64), np.eye(4)
        visited.add(link_name)

        parent, xyz, rpy = joint_info[link_name]
        child_t = np.array(xyz, dtype=np.float64)
        child_q_xyzw = R.from_euler("xyz", rpy, degrees=False).as_quat()

        child_mat = np.eye(4)
        child_mat[:3, :3] = R.from_quat(child_q_xyzw).as_matrix()
        child_mat[:3, 3] = child_t

        if parent != "base" and parent in joint_info and parent not in visited:
            parent_t, parent_mat = world_transform(parent, visited)
            world_mat = parent_mat @ child_mat
            return parent_t + parent_mat[:3, :3] @ child_t, world_mat

        return child_t, child_mat

    objects = []
    for link in root.findall(".//link"):
        name = link.get("name")
        if name == "base":
            continue
        visual = link.find(".//visual/geometry/mesh")
        if visual is None:
            continue
        visual_file = visual.get("filename")
        if not visual_file:
            continue
        visual_path = os.path.join(file_dir, visual_file)
        if not os.path.exists(visual_path):
            continue

        scale_str = visual.get("scale", "1.0 1.0 1.0")
        scale = np.array([float(x) for x in scale_str.split()], dtype=np.float64)

        world_t, world_mat = world_transform(name)
        objects.append({
            "name": name,
            "mesh_path": visual_path,
            "scale": scale,
            "world_transform": world_mat,  # 4x4 matrix in SAPIEN frame (Z-up)
        })

    return objects


def create_transform_matrix(world_mat, scale, z_offset=0.0):
    """Convert SAPIEN transform (Z-up) to pyrender transform (Y-up)."""
    # SAPIEN: Z is up; pyrender: Y is up
    # Rotate from SAPIEN Z-up to pyrender Y-up: rotate around X by -90 degrees
    z_to_y = R.from_euler("x", -90, degrees=True).as_matrix()
    z_to_y_mat = np.eye(4)
    z_to_y_mat[:3, :3] = z_to_y

    # SAPIEN transform * scale
    mat = world_mat.copy()
    mat[:3, :3] = mat[:3, :3] * scale

    # Add z_offset (which is SAPIEN Z-up)
    mat[2, 3] += z_offset

    # Convert to Y-up for pyrender
    mat = z_to_y_mat @ mat
    return mat


def render_with_pyrender(objects, output_path, cam_pos=None, look_at=None):
    """Render objects using pyrender with OBJ vertex colors.

    Coordinates: URDF mesh vertices are Z-up (SAPIEN). pyrender is Y-up.
    Convert: apply SAPIEN world transform, then rotate Z→Y (-90° around X).
    """
    scene = pyrender.Scene(ambient_light=np.array([0.5, 0.5, 0.5, 1.0]))

    # Directional light
    light_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 8],
        [0, 0, 0, 1]
    ])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=light_pose)

    print(f"Loading {len(objects)} objects...")
    for obj in objects:
        name = obj["name"]
        mesh_path = obj["mesh_path"]
        scale = obj["scale"]
        world_mat = obj["world_transform"]

        # Load OBJ with trimesh (preserves vertex colors)
        mesh = trimesh.load(mesh_path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            print(f"  {name}: not a TriMesh, skipping")
            continue

        # Get vertex colors
        if mesh.visual.defined and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            colors = mesh.visual.vertex_colors[:, :3]
        else:
            print(f"  {name}: no vertex colors, using default gray")
            colors = np.ones((len(mesh.vertices), 3), dtype=np.uint8) * 180

        # Scale vertices
        verts = mesh.vertices * scale

        # Apply SAPIEN world transform
        hom = np.column_stack([verts, np.ones(len(verts))])
        verts_world = (world_mat[:3, :3] @ verts.T + world_mat[:3, 3:]).T
        colors_transformed = colors

        # Convert from SAPIEN Z-up to pyrender Y-up: rotate -90° around X
        rot = R.from_euler("x", -90, degrees=True).as_matrix()
        verts_pr = (rot @ verts_world.T).T

        # Build pyrender mesh with vertex colors
        pr_mesh = pyrender.Mesh(
            positions=verts_pr.astype(np.float32),
            indices=mesh.faces,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                metallicFactor=0.0,
                roughnessFactor=1.0,
                alphaMode='OPAQUE'
            ),
            smooth=False
        )
        # Set per-vertex colors
        pr_mesh.primitives[0].material.baseColorFactor = np.array([1, 1, 1, 1])
        pr_mesh.primitives[0].baseColorMap = None

        # Attach vertex colors as attribute
        attr_name = pr_mesh.primitives[0].add_attribute(
            "baseColor", colors_transformed.astype(np.float32) / 255.0
        )
        # Use a shader that reads vertex color
        pr_mesh.primitives[0].material.uniforms["baseColorTexture"] = None

        scene.add(pr_mesh, pose=np.eye(4))
        print(f"  {name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, avg_color={colors.mean(axis=0).astype(int)}")

    # Camera: look-at transform
    if cam_pos is None:
        cam_pos = np.array([3.0, 5.0, 3.0])
    if look_at is None:
        look_at = np.array([0.0, 0.0, 0.0])

    cam_pos = np.array(cam_pos, dtype=np.float64)
    look_at = np.array(look_at, dtype=np.float64)
    forward = look_at - cam_pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(np.array([0, 1, 0], dtype=np.float64), forward)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(forward, right)

    # pyrender camera looks along -Z, so: columns = [right, up, -forward, position]
    cam_mat = np.eye(4)
    cam_mat[:3, 0] = right
    cam_mat[:3, 1] = up
    cam_mat[:3, 2] = -forward
    cam_mat[:3, 3] = cam_pos

    camera = pyrender.IntrinsicsCamera(
        fx=512, fy=512,
        cx=256, cy=256,
        znear=0.01, zfar=100
    )

    # Render
    viewer = pyrender.OffscreenRenderer(512, 512, point_size=1.0)
    color, depth = viewer.render(scene, camera=camera)
    viewer.delete()

    # Remove alpha channel if needed
    if len(color.shape) == 3 and color.shape[2] == 4:
        # Blend with gray background
        bg = np.ones((512, 512, 3), dtype=np.uint8) * 180
        alpha = color[:, :, 3:4].astype(np.float32) / 255.0
        rgb = (color[:, :, :3].astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha))
        color = rgb.astype(np.uint8)

    cv2.imwrite(output_path, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    print(f"Saved: {output_path}")
    return color


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Path to scene.urdf")
    args = parser.parse_args()

    urdf_path = args.urdf
    out_dir = os.path.dirname(urdf_path)
    output_path = os.path.join(out_dir, "stage61_render.png")

    print(f"Parsing {urdf_path}...")
    objects = parse_urdf_objects(urdf_path)
    if not objects:
        print("ERROR: No objects found in URDF")
        sys.exit(1)

    # Print object positions
    print("\nObjects:")
    for obj in objects:
        t = obj["world_transform"][:3, 3]
        print(f"  {obj['name']}: pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}), scale={obj['scale']}")

    # Find bounding box for camera placement
    all_centers = np.array([o["world_transform"][:3, 3] for o in objects])
    center = all_centers.mean(axis=0)
    max_dist = np.linalg.norm(all_centers - center, axis=1).max()

    # Camera: fixed side-top view
    # In SAPIEN Z-up: camera at (3, 2, 3) looking at center
    cam_pos = center + np.array([max_dist * 2, max_dist * 2, max_dist * 2])

    print(f"\nCamera at {cam_pos}, looking at {center}")
    color = render_with_pyrender(objects, output_path, cam_pos=cam_pos, look_at=center)

    # Also try overhead view
    overhead_path = os.path.join(out_dir, "stage61_render_overhead.png")
    cam_overhead = center + np.array([0, max_dist * 4, 0])  # straight above in Z
    color_overhead = render_with_pyrender(objects, overhead_path,
                                           cam_pos=cam_overhead, look_at=center)
    print(f"Saved overhead: {overhead_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
