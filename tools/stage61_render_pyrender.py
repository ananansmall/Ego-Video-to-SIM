#!/usr/bin/env python3
"""Stage 6.1 — SAPIEN physics + pyrender OBJ mesh output (simple)."""

import os, sys, argparse
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import sapien
import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_DEVICE_ID"] = "0"


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


def simulate_physics(objects):
    """Run SAPIEN with static actors, record poses."""
    engine = sapien.Scene()
    engine.set_timestep(1 / 60)
    engine.add_ground(altitude=0)
    for obj in objects:
        p = obj["world_pos"]
        builder = engine.create_actor_builder()
        builder.add_box_collision(half_size=(0.5, 0.5, 0.5), density=100)
        actor = builder.build()
        actor.set_pose(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        obj["final_pos"] = np.array(actor.get_pose().p).astype(np.float64)
        obj["final_orn"] = np.array(actor.get_pose().q).astype(np.float64)
    del engine
    return objects


def make_look_at_matrix(eye, target, up):
    """Build a 4x4 camera MODEL matrix for pyrender Node.

    pyrender stores camera as MODEL matrix (local→world):
      col0=+right, col1=+up, col2=+local_Z (= camera backward, opposite of look dir)
      col3=eye position
    Camera looks down local -Z, so look direction = target - eye = -col2.
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    local_z = eye - target       # col2 = camera backward
    local_z = local_z / np.linalg.norm(local_z)
    right = np.cross(up, local_z)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1, 0, 0], dtype=np.float64)
    else:
        right = right / np.linalg.norm(right)
    up_vec = np.cross(local_z, right)

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = right
    T[:3, 1] = up_vec
    T[:3, 2] = local_z
    T[:3, 3] = eye
    return T


def make_light_matrix(light_eye, light_target):
    """DirectionalLight model matrix. Light shines along -Z (opposite of col2).
    col2 = light_eye - light_target, so light direction = -col2 = toward target.
    """
    light_eye = np.asarray(light_eye, dtype=np.float64)
    light_target = np.asarray(light_target, dtype=np.float64)

    light_dir = light_target - light_eye
    light_dir = light_dir / np.linalg.norm(light_dir)
    local_z = -light_dir  # col2 = +local_Z = opposite of light dir

    up = np.array([0, 1, 0], dtype=np.float64)
    right = np.cross(up, local_z)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1, 0, 0], dtype=np.float64)
    else:
        right = right / np.linalg.norm(right)
    up_vec = np.cross(local_z, right)

    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = right
    T[:3, 1] = up_vec
    T[:3, 2] = local_z
    return T


def render_objects(objects, img_w=1024, img_h=768):
    """Render OBJ meshes with pyrender — simple, bright, white background."""
    all_p = np.array([o.get("final_pos", o["world_pos"]) for o in objects])
    center = all_p.mean(axis=0)
    spread = np.linalg.norm(all_p - center, axis=1).max()
    dist = max(spread * 2.0, 1.0)

    # Camera: high angle, looking down at scene
    eye = center + np.array([dist * 0.6, dist * 0.7, dist * 0.5])
    target = center
    print(f"  Camera: eye={eye}, target={target}")

    cam_T = make_look_at_matrix(eye, target, np.array([0, 1, 0]))

    # Simple 3-point lighting from multiple directions
    scene = pyrender.Scene(ambient_light=[10.0, 10.0, 10.0], bg_color=[1.0, 1.0, 1.0])
    # Key light from camera
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=100.0),
              pose=make_light_matrix(eye, target))
    # Fill light from opposite
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=80.0),
              pose=make_light_matrix(target - (eye - target), target))
    # Rim light from top
    scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=60.0),
              pose=make_light_matrix(target + np.array([0, 3, 0]), target))

    # OBJ has Z-up; rotate -90 deg around X to make Y-up
    z_to_y = R.from_euler("x", -90, degrees=True).as_matrix()

    for obj in objects:
        fp = obj.get("final_pos", obj["world_pos"])
        mesh = trimesh.load(obj["mesh_path"], process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        verts = mesh.vertices.astype(np.float64) * obj["scale"]
        verts = (z_to_y @ verts.T).T
        if "final_orn" in obj:
            verts = (R.from_quat(obj["final_orn"]).as_matrix() @ verts.T).T
        verts += fp
        m = trimesh.Trimesh(verts.astype(np.float32), mesh.faces[:, ::-1])
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1, 1, 1, 1], roughnessFactor=0.8, metallicFactor=0.0,
            doubleSided=True)
        scene.add(pyrender.Mesh.from_trimesh(m, material=mat, smooth=False), pose=np.eye(4))
        print(f"  {obj['name']}: pos=({fp[0]:.2f},{fp[1]:.2f},{fp[2]:.2f})")

    cam = pyrender.PerspectiveCamera(yfov=np.pi / 4, aspectRatio=img_w / img_h)
    scene.add_node(pyrender.Node(camera=cam, matrix=cam_T))

    viewer = pyrender.OffscreenRenderer(img_w, img_h)
    color, depth = viewer.render(scene)
    viewer.delete()

    rgb = color[:, :, :3] if color.shape[2] == 4 else color
    print(f"  Output: mean={rgb.mean():.1f}, unique={len(np.unique(rgb.reshape(-1, 3), axis=0))}")
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
    for o in objects:
        print(f"  {o['name']}: {o['world_pos']}")
    print()

    print("Running SAPIEN physics...")
    objects = simulate_physics(objects)
    for o in objects:
        print(f"  {o['name']}: final={o['final_pos']}")
    print()

    print("Rendering OBJ meshes with pyrender...")
    rgb = render_objects(objects, img_w=args.img_w, img_h=args.img_h)
    out = os.path.join(d, "stage61_sim.png")
    cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"\nSaved: {out}")
    print("Done!")


if __name__ == "__main__":
    main()
