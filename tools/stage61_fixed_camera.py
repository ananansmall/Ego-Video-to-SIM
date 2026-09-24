import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import sapien
import sapien.physx as physx

# Match EmbodiedGen's setup_scene pattern (simulation.py)
def setup_scene():
    scene = sapien.Scene()
    scene.set_ambient_light([0.2, 0.2, 0.2])
    scene.add_directional_light(
        direction=[0, 1, -1], color=[1.5, 1.45, 1.4], shadow=True, shadow_map_size=2048,
    )
    scene.add_directional_light(direction=[0, -0.5, 1], color=[0.8, 0.8, 0.85], shadow=False)
    scene.add_directional_light(direction=[0, -1, 1], color=[1.0, 1.0, 1.0], shadow=False)
    renderer = sapien.SapienRenderer()
    ground_mat = renderer.create_material()
    ground_mat.base_color = [0.5, 0.5, 0.5, 1]
    ground_mat.roughness = 0.7
    ground_mat.metallic = 0.0
    scene.add_ground(0, render_material=ground_mat)
    return scene


def _get_local_pose(origin_tag):
    """Replicate EmbodiedGen's _get_local_pose (simulation.py:112)."""
    if origin_tag is None:
        return sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
    xyz = list(map(float, origin_tag.get("xyz", "0 0 0").split()))
    rpy = list(map(float, origin_tag.get("rpy", "0 0 0").split()))
    qx, qy, qz, qw = R.from_euler("xyz", rpy, degrees=False).as_quat()
    return sapien.Pose(p=xyz, q=[qw, qx, qy, qz])


def load_scene_urdf(scene, urdf_path):
    """Parse scene.urdf and add all links to SAPIEN scene."""
    file_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Collect joints: child link name -> (parent, origin_xyz, origin_rpy)
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

    # Compute cumulative world transform for each link
    def world_transform(link_name, visited=None):
        if visited is None:
            visited = set()
        if link_name in visited or link_name not in joint_info:
            return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        visited.add(link_name)

        parent, xyz, rpy = joint_info[link_name]
        child_t = np.array(xyz, dtype=np.float64)
        child_q = R.from_euler("xyz", rpy).as_quat()
        child_q_wxyz = [child_q[3], child_q[0], child_q[1], child_q[2]]

        if parent != "base" and parent in joint_info and parent not in visited:
            parent_t, parent_q_wxyz = world_transform(parent, visited)
            parent_mat = R.from_quat([
                parent_q_wxyz[1], parent_q_wxyz[2],
                parent_q_wxyz[3], parent_q_wxyz[0],
            ]).as_matrix()
            world_t = parent_t + parent_mat @ child_t

            w1, x1, y1, z1 = parent_q_wxyz
            w2, x2, y2, z2 = child_q_wxyz
            world_q = np.array([
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ], dtype=np.float64)
            return world_t.tolist(), world_q.tolist()

        return child_t.tolist(), child_q_wxyz

    # Find all links with visual mesh
    link_mesh = {}
    for link in root.findall(".//link"):
        name = link.get("name")
        if name == "base":
            continue
        visual = link.find(".//visual/geometry/mesh")
        if visual is not None:
            link_mesh[name] = visual

    # Load each link into the scene
    loaded = []
    for name, visual_mesh in link_mesh.items():
        world_xyz, world_q = world_transform(name)

        visual_file = visual_mesh.get("filename")
        if not visual_file:
            continue
        visual_path = os.path.join(file_dir, visual_file)
        if not os.path.exists(visual_path):
            continue

        scale_str = visual_mesh.get("scale", "1.0 1.0 1.0")
        scale = np.array([float(x) for x in scale_str.split()])
        visual_origin = link.find(".//visual/origin")
        local_pose = _get_local_pose(visual_origin)

        # Pose
        pose = sapien.Pose(p=[world_xyz[0], world_xyz[1], world_xyz[2]], q=[
            world_q[0], world_q[1], world_q[2], world_q[3]])

        builder = scene.create_actor_builder()
        builder.set_initial_pose(pose)

        # Visual mesh
        builder.add_visual_from_file(visual_path, scale=scale, pose=local_pose)

        # Collision mesh + material
        collision = link.find(".//collision/geometry/mesh")
        collision_origin = link.find(".//collision/origin")
        collision_scale_str = "1.0 1.0 1.0"
        collision_file = None
        if collision is not None:
            collision_file = collision.get("filename")
            if collision_file:
                collision_file = os.path.join(file_dir, collision_file)
            cs = collision.get("scale", "1.0 1.0 1.0")
            collision_scale_str = cs

        if collision_file and os.path.exists(collision_file):
            # Find gazebo material params
            static_fric = 0.6
            dynamic_fric = 0.5
            for gaz in link.findall(".//collision/gazebo"):
                try:
                    static_fric = float(gaz.find("mu1").text)
                    dynamic_fric = float(gaz.find("mu2").text)
                except:
                    pass

            material = physx.PhysxMaterial(
                static_friction=np.clip(static_fric, 0.1, 0.7),
                dynamic_friction=np.clip(dynamic_fric, 0.1, 0.6),
                restitution=0.05,
            )
            collision_scale = np.array([float(x) for x in collision_scale_str.split()])
            builder.add_multiple_convex_collisions_from_file(
                collision_file, material=material, scale=collision_scale,
                pose=_get_local_pose(collision_origin),
            )
            builder.set_physx_body_type("dynamic")
        else:
            builder.set_physx_body_type("static")

        actor = builder.build(name=name)
        print(f"  Loaded {name}: world_xyz={world_xyz}")
        loaded.append(name)

    return loaded


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True, help="Path to scene.urdf")
    args = parser.parse_args()

    urdf_path = args.urdf
    out_dir = os.path.dirname(urdf_path)

    scene = setup_scene()
    scene.set_timestep(1.0 / 200)

    loaded = load_scene_urdf(scene, urdf_path)
    print(f"Loaded {len(loaded)} actors")

    # Run physics
    print("Running 400 physics steps...")
    for i in range(400):
        scene.step()
        scene.update_render()

    # Check actor positions after physics
    print("\nActor positions after physics:")
    for a in scene.get_all_actors():
        print(f"  {a.name}: pose.p={a.get_pose().p}")

    # Create fixed overhead camera - SAPIEN Y is UP
    # Camera at high Y, looking down (-Y direction)
    cam = scene.add_camera("cam", width=512, height=512,
        fovy=np.deg2rad(45), near=0.01, far=20)
    # Camera at Y=5, looking straight down (q=[1,0,0,0] = identity, camera looks -Z in SAPIEN)
    # Actually in SAPIEN camera looks along -Z by default, with Y up.
    # For overhead, put camera at Y=5, rotate so it looks down (-Y)
    # Quaternion for "look down" = 90° rotation around X axis = [0.707, 0.707, 0, 0]
    cam.set_entity_pose(sapien.Pose(p=[0.0, 5.0, 0.0], q=[0.707, 0.707, 0.0, 0.0]))
    scene.update_render()
    cam.take_picture()
    color = cam.get_picture("Color")
    rgb = (np.clip(color[:,:,:3], 0, 1) * 255).astype(np.uint8)
    print(f"\nOverhead cam: min={rgb.min()}, max={rgb.max()}, mean={rgb.mean():.2f}")
    print(f"Unique colors: {len(np.unique(rgb.reshape(-1,3), axis=0))}")
    out_path = os.path.join(out_dir, "stage61_overhead.png")
    cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}")

    # Also take a side view
    cam2 = scene.add_camera("cam2", width=512, height=512,
        fovy=np.deg2rad(50), near=0.01, far=20)
    # Side view: camera at X=3, Y=2, looking toward origin
    # Need to look toward -X direction from there, with slight downward tilt
    cam2.set_entity_pose(sapien.Pose(p=[3.0, 2.0, 0.0], q=[0.707, 0.0, 0.0, 0.707]))
    scene.update_render()
    cam2.take_picture()
    color2 = cam2.get_picture("Color")
    rgb2 = (np.clip(color2[:,:,:3], 0, 1) * 255).astype(np.uint8)
    print(f"Side cam: min={rgb2.min()}, max={rgb2.max()}, mean={rgb2.mean():.2f}")
    out_path2 = os.path.join(out_dir, "stage61_side.png")
    cv2.imwrite(out_path2, cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()
