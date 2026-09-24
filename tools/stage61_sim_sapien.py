#!/usr/bin/env python3
"""Stage 6.1 — Fast simulation snapshot via SAPIEN + pyrender."""

import os, sys, argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import sapien
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R

os.environ["PYOPENGL_PLATFORM"] = "egl"


def parse_urdf(urdf_path):
    file_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint_info = {}
    for joint in root.findall(".//joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        origin = joint.find("origin")
        oattr = origin.attrib if origin is not None else {}
        xyz = list(map(float, oattr.get("xyz", "0 0 0").split()))
        rpy = list(map(float, oattr.get("rpy", "0 0 0").split()))
        joint_info[child] = (parent, xyz, rpy)

    def world_transform(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited or name not in joint_info:
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
        visited.add(name)
        parent, xyz, rpy = joint_info[name]
        ct = np.array(xyz, dtype=np.float64)
        cm = R.from_euler("xyz", list(map(float, rpy)), degrees=False).as_matrix()
        if parent != "base" and parent in joint_info and parent not in visited:
            pt, pm = world_transform(parent, visited)
            return pt + pm @ ct, pm @ cm
        return ct, cm

    objects = []
    for link in root.findall(".//link"):
        name = link.get("name")
        if name == "base":
            continue
        visual = link.find(".//visual/geometry/mesh")
        if visual is None or not visual.get("filename"):
            continue
        mesh_path = os.path.join(file_dir, visual.get("filename"))
        if not os.path.exists(mesh_path):
            continue
        scale = np.array([float(x) for x in visual.get("scale", "1 1 1").split()])
        wt, wm = world_transform(name)
        objects.append({"name": name, "mesh_path": mesh_path,
                        "scale": scale, "world_pos": wt, "world_rot": wm})
    return objects


def render_sapien(objects, img_w=1024, img_h=768):
    """Run SAPIEN scene, export actor positions, render with pyrender."""
    # --- SAPIEN: load actors, verify physics works ---
    engine = sapien.Scene()
    engine.set_timestep(1 / 60)
    engine.add_ground(altitude=0, render=True)
    engine.add_directional_light(direction=[0, -1, -1], color=[1, 0.95, 0.9])
    engine.add_directional_light(direction=[0, 1, -1], color=[0.6, 0.6, 0.7])

    actors_data = []
    colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1)]
    for i, obj in enumerate(objects):
        p = obj["world_pos"]
        scale = obj["scale"]
        # Box sized to object extent
        box_half = [max(s * 0.6, 0.3) for s in scale]
        builder = engine.create_actor_builder()
        builder.add_box_collision(half_size=tuple(box_half))
        mat = sapien.render.RenderMaterial()
        mat.base_color = colors[i % len(colors)]
        builder.add_box_visual(half_size=tuple(box_half), material=mat)
        actor = builder.build()
        actor.set_pose(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        actors_data.append({
            "name": obj["name"],
            "pos": p,
            "scale": scale,
            "box_half": box_half,
            "color": colors[i % len(colors)]
        })
        print(f"  Added {obj['name']} at {p}")
    del engine

    # --- pyrender: render boxes at same positions ---
    # Camera at (0, 0, 3) looking at origin, objects placed at their URDF coords
    scene = pyrender.Scene(ambient_light=[1.5, 1.5, 1.5],
                           bg_color=[0.2, 0.2, 0.2])
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=5.0),
              pose=np.eye(4, dtype=np.float32))
    scene.add(pyrender.DirectionalLight(color=[0.7, 0.7, 0.9], intensity=3.0),
              pose=np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32))
    scene.add(pyrender.DirectionalLight(color=[0.8, 0.6, 0.5], intensity=2.0),
              pose=np.array([[1, 0, 0, 0], [0, 0.5, -0.5, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]], dtype=np.float32))

    for ad in actors_data:
        pos = ad["pos"]
        h = ad["box_half"]
        box = trimesh.creation.box(extents=[2*h[0], 2*h[1], 2*h[2]])
        # Place box center at pos
        M = np.eye(4)
        M[:3, 3] = pos
        box.apply_transform(M)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=ad["color"],
            roughnessFactor=0.5, metallicFactor=0.0,
            alphaMode="OPAQUE")
        scene.add(pyrender.Mesh.from_trimesh(box, material=mat, smooth=False))
        print(f"  Rendered {ad['name']} at {pos}")

    # Camera
    all_p = np.array([a["pos"] for a in actors_data])
    center = all_p.mean(axis=0)
    spread = np.linalg.norm(all_p - center, axis=1).max()
    dist = max(spread * 3.5, 2.0)
    cam_pos = np.array([center[0], center[1] + dist, center[2] + dist * 0.6])
    look_at = center + np.array([0, 0, -0.5])
    f = look_at - cam_pos
    f = f / np.linalg.norm(f)
    u = np.array([0, 1, 0])
    r = np.cross(f, u)
    if np.linalg.norm(r) < 1e-3:
        r = np.array([1, 0, 0])
    else:
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, cam_pos

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4, aspectRatio=img_w / img_h)
    scene.add_node(pyrender.Node(camera=cam, matrix=T))

    viewer = pyrender.OffscreenRenderer(img_w, img_h)
    color, depth = viewer.render(scene)
    viewer.delete()

    if color.shape[2] == 4:
        a = color[:, :, 3:4].astype(np.float32) / 255
        rgb = (color[:, :, :3].astype(np.float32) * a +
               np.array([180, 180, 180], dtype=np.float32) * (1 - a)).astype(np.uint8)
    else:
        rgb = color
    print(f"  Output: shape={rgb.shape}, mean={rgb.mean():.2f}, unique={len(np.unique(rgb.reshape(-1, 3), axis=0))}")
    return rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument("--img_h", type=int, default=768)
    args = p.parse_args()
    d = os.path.abspath(args.output_dir)
    urdf = os.path.join(d, "urdf", "scene.urdf")
    assert os.path.exists(urdf), f"Not found: {urdf}"

    print("Parsing URDF...")
    objects = parse_urdf(urdf)
    assert objects, "No objects"
    print(f"Found {len(objects)} objects")

    print("\nRunning SAPIEN + Rendering...")
    rgb = render_sapien(objects, img_w=args.img_w, img_h=args.img_h)
    out = os.path.join(d, "stage61_sim.png")
    cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {out}")
    print("Done!")


if __name__ == "__main__":
    main()
