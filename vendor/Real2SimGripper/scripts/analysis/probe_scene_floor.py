"""Check a frozen floor or support collision mesh on its own in SAPIEN."""

import argparse
import json
from pathlib import Path

import numpy as np
import sapien.core as sapien
import trimesh


def drop(mesh_path, centers_xy, height, offset):
    scene = sapien.Scene()
    scene.set_timestep(1 / 250)
    builder = scene.create_actor_builder()
    builder.add_nonconvex_collision_from_file(str(mesh_path))
    support = builder.build_static(name="tested_support")
    support.set_pose(sapien.Pose(p=[0., 0., offset]))
    radius = .025
    probes = []
    for i, xy in enumerate(centers_xy):
        builder = scene.create_actor_builder()
        builder.add_sphere_collision(radius=radius)
        probe = builder.build(name=f"probe_{i}")
        probe.set_pose(sapien.Pose(p=[*xy, height + .32]))
        probes.append(probe)

    positions = [np.array([probe.get_pose().p for probe in probes])]
    supported = np.zeros((250, len(probes)), dtype=bool)
    indices = {probe.name: i for i, probe in enumerate(probes)}
    for step in range(250):
        scene.step()
        positions.append(np.array([probe.get_pose().p for probe in probes]))
        for contact in scene.get_contacts():
            a, b = (body.get_entity().get_name() for body in contact.bodies)
            if a == "tested_support" and b in indices:
                probe_name, sign = b, -1.
            elif b == "tested_support" and a in indices:
                probe_name, sign = a, 1.
            else:
                continue
            supported[step, indices[probe_name]] = any(
                sign * point.impulse[2] > 1e-8 for point in contact.points)

    positions = np.array(positions)
    acceleration = 2 * np.polyfit(np.arange(9) * scene.get_timestep(), positions[:9, 0], 2)[0]
    bottoms = positions[-1, :, 2] - radius
    tail_drift = np.linalg.norm(positions[-20:] - positions[-1], axis=2).max(axis=0)
    upward_count = supported[-20:].sum(axis=0)
    passed = (np.linalg.norm(acceleration - [0., 0., -9.81]) < .05
              and np.max(np.abs(bottoms - height)) <= .002
              and np.max(tail_drift) <= .001
              and np.all(upward_count == 20))
    return dict(passed=bool(passed), mesh_offset_z_m=offset, steps=250,
                free_fall_accel_m_s2=acceleration.tolist(),
                final_bottom_z_m=bottoms.tolist(), max_last_20_step_drift_m=tail_drift.tolist(),
                upward_support_steps_last_20=upward_count.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--height", type=float, default=0.)
    args = parser.parse_args()
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    candidates = np.flatnonzero(
        (mesh.face_normals[:, 2] >= .95) & (np.abs(mesh.triangles_center[:, 2] - args.height) < .03))
    if not len(candidates):
        raise ValueError("No upward-facing mesh triangle near the supplied height")
    largest = candidates[np.argsort(-mesh.area_faces[candidates])]
    centers = [mesh.triangles_center[largest[0], :2]]
    for i in largest[1:]:
        xy = mesh.triangles_center[i, :2]
        if np.linalg.norm(xy - centers[0]) > .1:
            centers.append(xy)
            break
    positive = drop(args.mesh, centers, args.height, offset=0.)
    elevated = drop(args.mesh, centers, args.height, offset=.12)
    expected_negative = (not elevated["passed"] and min(elevated["upward_support_steps_last_20"]) == 20
                         and min(elevated["final_bottom_z_m"]) - args.height > .10)
    result = dict(scope="static collision mesh, no fallback floor, sphere drop; no video-based plane verification",
                  mesh=str(args.mesh.resolve()), expected_top_z_m=args.height,
                  drop_points_xy_m=np.asarray(centers).tolist(),
                  positive=positive, raised_mesh_control=elevated, passed=positive["passed"] and expected_negative)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
