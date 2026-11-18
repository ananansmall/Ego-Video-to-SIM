"""EE17 physically actuated fixture replay with a fixed-base task-space plan."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import sapien.core as sapien
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import cb_large_range_sim as cb

rg, bp = cb.rg, cb.bp
DATA = Path("/path/to/Real2SimGripper/results/large_range/"
            "ee17_posture_release600_fixed_20261110_231736")
MOCAP = Path("/path/to/project_data/dataset/EgoExplorer/dataset/lerobot/episode_meta/"
             "episode_000017/smplx/smplx_episode.npz")


def scene_geometry(out):
    # Pixel estimates on the visible ring in source images 00016/00149/00178.
    pixels = {16: [(1032, 389), (1196, 377), (1116, 447)],
              149: [(1124, 331), (1275, 332), (1195, 386), (1199, 271)],
              178: [(1129, 238), (1291, 226), (1207, 285)]}
    tf = json.loads((DATA / "ee17_ras_transform.json").read_text())
    with np.load("/tmp/ee17-ras.DMqO0l/raw_map.npz") as mapping:
        points = mapping["points"]
        _, h, w, _ = points.shape
        rings = []
        for frame, marks in pixels.items():
            idx = np.flatnonzero(mapping["frame_10fps"] == frame).item()
            u, v = (np.asarray(marks) / [1920, 970] * [w, h]).astype(int).T
            q = tf["scale"] * (points[idx, v, u] @ np.asarray(tf["R"]).T) + tf["t"]
            # Midpoint of the two opposing rim marks estimates the ring center.
            rings.append([(q[0, 0] + q[1, 0]) / 2,
                          (q[0, 1] + q[1, 1]) / 2, np.median(q[:, 2]),
                          np.linalg.norm(q[0, :2] - q[1, :2]) / 2])
    rings = np.asarray(rings)
    center = np.median(rings[:, :3], axis=0)
    inner_radius = float(np.median(rings[:, 3]))
    with np.load(MOCAP) as mocap:
        axis = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        body = mocap["transl"][11340:11940] @ axis.T
        wrists = mocap["tracker_position"][11340:11940, 1:3] @ axis.T
        ground = float(mocap["ground_y_m"])
    forward = center[:2] - np.median(body[:, :2], axis=0)
    forward /= np.linalg.norm(forward)
    rotation = np.eye(3)
    rotation[:2, :2] = [[forward[1], -forward[0]], forward]
    shift = np.r_[-rotation[:2, :2] @ center[:2], -ground]
    with np.load(DATA / "ee17_ras_scene.npz") as cloud:
        points = cloud["points"] @ rotation.T + shift
        colors = cloud["colors"]
    ring_z = float(center[2] - ground)
    table_band = ((np.linalg.norm(points[:, :2], axis=1) < 0.50)
                  & (abs(points[:, 2] - (tf["table_z_ras"] - ground)) < 0.025))
    table_z = float(np.median(points[table_band, 2]))
    np.savez(out / "scene_geometry.npz", points=points, colors=colors, rotation=rotation,
             translation=shift, ring_observations=rings, ring_z=ring_z,
             ring_inner_radius=inner_radius, table_z=table_z, table_points=points[table_band],
             wrists=wrists @ rotation.T + shift, pelvis=body @ rotation.T + shift)
    return ring_z, inner_radius, table_z, points, colors


def ring_actor(scene, height, radius, material):
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("static")
    segments = 24
    # The polygon's inner tangent is the measured inner radius.
    thickness = 0.004
    r = radius + thickness / 2
    for theta in np.arange(segments) * (2 * np.pi / segments):
        pose = sapien.Pose(p=[r * np.cos(theta), r * np.sin(theta), height - thickness / 2],
                           q=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)])
        half = [thickness / 2, (r + thickness / 2) * np.tan(np.pi / segments), thickness / 2]
        builder.add_box_collision(pose, half_size=half, material=material)
        builder.add_box_visual(pose, half_size=half, material=(0.12, 0.13, 0.15))
    builder.build(name="ee17_ring")


def funnel_actor(scene, out, height, radius, material):
    # Upright axisymmetric proxy based on the visible stopcock, bulb and neck.
    # Its shoulder is wider than the ring while the stem passes through.
    profiles = [
        [(-.066, .006), (-.042, .007), (-.025, .012), (0., radius),
         (.024, radius * 1.30), (.046, radius * 1.20), (.064, .012)],
        [(.059, .012), (.089, .012)],
    ]
    angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    for i, profile in enumerate(profiles):
        vertices = np.concatenate([np.column_stack((r * np.cos(angle), r * np.sin(angle),
                                                     np.full_like(angle, z))) for z, r in profile])
        path = out / f"funnel_section_{i}.obj"
        trimesh.convex.convex_hull(vertices).export(path)
        builder.add_convex_collision_from_file(str(path), material=material)
        builder.add_visual_from_file(str(path), material=(0.78, 0.57, 0.22))
    actor = builder.build(name="ee17_funnel")
    actor.set_pose(sapien.Pose(p=[0, 0, height + 0.001]))
    for component in actor.get_components():
        if hasattr(component, "set_disable_gravity"):
            component.mass = 0.12
            component.linear_damping = 0.08
            component.angular_damping = 0.12
    return actor


def solve_arm(robot, targets, q_seed, selected):
    limits = bp._limits("right")
    lo, hi = limits[:, 0] + 0.02, limits[:, 1] - 0.02
    rotation = Rotation.from_matrix(np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]))
    rng = np.random.default_rng(17)
    solved = []
    for i, target in enumerate(targets):
        def residual(q):
            pos, matrix = bp._gripper_pad_fk(robot, "right", selected, q)
            error = (rotation.inv() * Rotation.from_matrix(matrix)).as_rotvec()
            return np.r_[pos - target, 0.14 * error, 0.002 * (q - q_seed)]
        # SAPIEN FK is float32; SciPy's default float64 perturbation is too small.
        def jacobian(q):
            step = 1e-4
            return np.column_stack([(residual(q + dx) - residual(q - dx)) / (2 * step)
                                    for dx in step * np.eye(len(q))])

        seeds = [q_seed] if i else [q_seed, *rng.uniform(lo, hi, (12, 6))]
        results = [least_squares(residual, np.clip(seed, lo, hi), bounds=(lo, hi),
                                 jac=jacobian, max_nfev=90, ftol=1e-6) for seed in seeds]
        q_seed = min(results, key=lambda x: x.cost).x
        solved.append(q_seed)
    return np.asarray(solved)


def replay(out, full, render, reach):
    out.mkdir(parents=True, exist_ok=True)
    ring_z, radius, table_z, points, colors = scene_geometry(out)
    scene, robot = cb._build_scene()
    robot.set_root_pose(sapien.Pose(p=[-.142, -reach, 0], q=rg.R1_BASE_Q))
    material = scene.create_physical_material(1.2, 1.0, 0.0)
    # Worktop is a local collision slab, and the ring is a separate support.
    rg.create_box(scene, sapien.Pose(p=[0, .22, table_z - .025]),
                  half_size=[.48, .34, .025], color=(.12, .13, .14), name="ee17_worktop",
                  is_static=True)
    ring_actor(scene, ring_z, radius, material)
    post = scene.create_actor_builder()
    post.set_physx_body_type("static")
    pole = sapien.Pose(p=[-.085, .060, (ring_z + table_z + .12) / 2])
    pole_size = [.006, .006, (ring_z - table_z + .12) / 2]
    post.add_box_collision(pole, half_size=pole_size, material=material)
    post.add_box_visual(pole, half_size=pole_size, material=(.24, .25, .26))
    post.build(name="ee17_stand_post")
    actor = funnel_actor(scene, out, ring_z, radius, material)

    joints = robot.get_active_joints()
    indices = {j.get_name(): i for i, j in enumerate(joints)}
    selected = [indices[f"right_arm_joint{i+1}"] for i in range(6)]
    fingers = [indices[f"right_gripper_finger_joint{i+1}"] for i in range(2)]
    q_start = robot.get_qpos()
    q_start[fingers] = .045
    q_start[[indices[f'left_gripper_finger_joint{i+1}'] for i in range(2)]] = .025
    q_start[[indices[f"left_arm_joint{i+1}"] for i in range(6)]] = [1., .6, -.4, .5, 0., 0.]
    robot.set_qpos(q_start)
    for i, joint in enumerate(joints):
        if i in fingers:
            joint.set_drive_property(stiffness=4000, damping=120, force_limit=35)
        else:
            joint.set_drive_property(stiffness=1800, damping=120)
    for name in ("right_gripper_finger_link1", "right_gripper_finger_link2"):
        link = robot.find_link_by_name(name)
        for comp in link.entity.components:
            for shape in getattr(comp, "get_collision_shapes", lambda: ())():
                shape.set_physical_material(material)

    grasp_z = ring_z + .077
    initial = np.array([0., -.115, grasp_z])
    grip = np.array([0., 0., grasp_z])
    high = grip + [0, 0, .12]
    carry = high + [.07, 0, 0]
    if full:
        frames = np.array([0, 28, 47, 53, 95, 120, 210, 300, 390, 448, 505, 534, 551, 574, 599])
    else:
        frames = np.array([0, 18, 40, 55, 75, 90, 102, 114, 126, 143, 165, 180, 194, 215, 239])
    poses = np.array([initial, initial, grip, grip, high, high, carry, carry, high,
                      high, grip, grip, grip, initial, initial])
    opens = np.array([.045, .045, .045, .0, .0, .0, .0, .0, .0, .0,
                      .0, .0, .045, .045, .045])
    length = int(frames[-1] + 1)
    trajectory = np.empty((length, 3))
    opening = np.empty(length)
    for k, (i, j) in enumerate(zip(frames[:-1], frames[1:])):
        u = np.linspace(0., 1., j - i + 1)
        ease = u*u*(3 - 2*u)
        trajectory[i:j+1] = poses[k] + ease[:, None] * (poses[k+1] - poses[k])
        opening[i:j+1] = opens[k] + ease * (opens[k+1] - opens[k])
    arm = solve_arm(robot, trajectory, np.array([-.8, 1.3, -1., 0, 0, 0]), selected)
    q_start[selected] = arm[0]
    robot.set_qpos(q_start)
    for joint, q in zip(joints, q_start):
        joint.set_drive_target(float(q))
    pose = robot.find_link_by_name("right_gripper_link").get_entity_pose().to_transformation_matrix()
    print("initial commanded q", q_start[selected].round(3), "fk",
          (pose[:3, 3] + rg.PAD_FWD * pose[:3, 0]).round(4), flush=True)
    rg.step_sim(scene, robot, 125)
    print("initial actual q", robot.get_qpos()[selected].round(3), flush=True)
    print("settle collisions", [(c.bodies[0].get_entity().get_name(),
                                 c.bodies[1].get_entity().get_name())
                               for c in scene.get_contacts() if any(
                                   np.linalg.norm(p.impulse) > 1e-5 for p in c.points)], flush=True)
    print("object mass", [c.mass for c in actor.get_components() if hasattr(c, "mass")], flush=True)
    obj0 = np.asarray(actor.get_pose().p).copy()
    print("geometry", ring_z, radius, table_z, "settled", obj0, flush=True)
    print("target/actual initial", initial, bp._gripper_pad_fk(robot, "right", selected, robot.get_qpos()[selected])[0], flush=True)
    traces, p_grip, contact, both, step_counts, ring_only_steps = [], [], [], [], [], []
    names = ("right_gripper_finger_link1", "right_gripper_finger_link2",
             "ee17_ring", "ground", "ee17_worktop", "other")
    contact_column = {name: i for i, name in enumerate(names)}
    gripper_link = robot.find_link_by_name("right_gripper_link")
    if render:
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static")
        # Keep the reconstructed worktop and distant context. Above the local
        # worktop, retain the physical ring and funnel actors as the visible geometry.
        distance = np.linalg.norm(points[:, :2], axis=1)
        context = (points[:, 2] < table_z + .025) | ((distance > .30) & (points[:, 1] > .07))
        keep = context & (points[:, 2] >= table_z - .035) & (distance < .90)
        for p, c in zip(points[keep][::2], colors[keep][::2]):
            builder.add_box_visual(sapien.Pose(p=p), half_size=[.006]*3,
                                   material=tuple(np.clip(c, 0, 1)))
        builder.build(name="ee17_scene_cloud")
        front = scene.add_camera("oblique", 480, 360, np.deg2rad(48), .02, 10)
        front.entity.set_pose(rg.look_at_pose([.77, -.80, 1.53], [-.04, -.08, ring_z]))
        ego = scene.add_camera("robot_fpv", 480, 360, np.deg2rad(60), .02, 10)
        clip, fpv = [], []
    for k in range(length):
        for i, q in zip(selected, arm[k]):
            joints[i].set_drive_target(float(q))
        for i in fingers:
            joints[i].set_drive_target(float(opening[k]))
        j_sum = np.zeros(len(names))
        n_both = n_ring_only = 0
        steps = ((k + 1) * 250) // 30 - (k * 250) // 30
        step_counts.append(steps)
        for _ in range(steps):
            rg.step_sim(scene, robot)
            impulse = np.zeros(len(names))
            for item in scene.get_contacts():
                a, b = (body.get_entity().get_name() for body in item.bodies)
                other = b if a == "ee17_funnel" else a if b == "ee17_funnel" else None
                if other is not None:
                    column = contact_column.get(other, contact_column["other"])
                    impulse[column] += sum(np.linalg.norm(p.impulse) for p in item.points)
            j_sum += impulse
            n_both += int(np.all(impulse[:2] > 1e-6))
            n_ring_only += int(impulse[2] > 1e-5 and np.all(impulse[[0, 1, 3, 4, 5]] <= 1e-6))
        ring_only_steps.append(n_ring_only)
        pose = actor.get_pose()
        traces.append(np.r_[pose.p, pose.q])
        T = gripper_link.get_entity_pose().to_transformation_matrix()
        p_grip.append(T[:3, 3] + rg.PAD_FWD * T[:3, 0])
        contact.append(j_sum)
        both.append(n_both)
        if k in frames:
            print(k, "target", trajectory[k].round(3), "pad", np.round(p_grip[-1], 3),
                  "object", np.asarray(pose.p).round(3), "impulses", j_sum.round(4),
                  "both", n_both, flush=True)
        if render:
            root = robot.get_root_pose().p
            head_z = robot.find_link_by_name("torso_link4").get_entity_pose().p[2] + .34
            ego.entity.set_pose(rg.look_at_pose([root[0], root[1] + .23, head_z],
                                               [0., -.015, ring_z + .065]))
            scene.update_render()
            imgs = []
            for cam in (front, ego):
                cam.take_picture()
                imgs.append(np.asarray(np.clip(cam.get_picture("Color")[:, :, :3], 0, 1) * 255,
                                       dtype=np.uint8))
            fpv.append(imgs[1])
            if k % 2 == 0:
                clip.append(np.concatenate(imgs, axis=1))
    obj, impulse, both = np.asarray(traces), np.asarray(contact), np.asarray(both)
    drift = np.linalg.norm(obj[-15:, :3] - obj[-1, :3], axis=1).max()
    q_delta = np.abs(obj[-15:, 3:] @ obj[-1, 3:])
    angle_drift = np.degrees(2 * np.arccos(np.clip(q_delta, 0, 1))).max()
    supported = (impulse[:, 2] > 1e-4) & np.all(impulse[:, [0, 1, 3, 4, 5]] <= 1e-5, axis=1)
    lifted = (obj[:, 2] - obj0[2] > .04) & (both > 0) & (impulse[:, 2] <= 1e-5)
    settled_before_open = bool(np.all(impulse[frames[11]-5:frames[11], 2] > 1e-4))
    verdict = dict(frames=length, physics_steps=sum(step_counts), physical_seconds=sum(step_counts)/250,
                   independent_ring_support_last_15_s=sum(ring_only_steps[-15:])/250,
                   ring_height_m=ring_z, ring_inner_radius_m=radius,
                   table_height_m=table_z, lift_mm=float(1000*(obj[:, 2].max()-obj0[2])),
                   bilateral_dynamic_lift_frames=int(lifted.sum()),
                   bilateral_dynamic_total_frames=int(np.count_nonzero(both)),
                   independent_ring_support_last_15=int(supported[-15:].sum()),
                   position_drift_mm=float(drift * 1000),
                   rotation_drift_deg=float(angle_drift),
                   ground_contact_frames=int(np.count_nonzero(impulse[:, 3] > 1e-5)),
                   seated_before_open=settled_before_open,
                   pad_p95_mm=float(np.percentile(np.linalg.norm(np.asarray(p_grip)-trajectory, axis=1), 95)*1000))
    verdict["passed"] = bool(lifted.any() and settled_before_open and supported[-15:].all()
                             and sum(ring_only_steps[-15:]) >= 125 and drift < .003 and angle_drift < 3.)
    np.savez(out / "execution.npz", q_plan=arm, object_pose=obj, pad_actual=p_grip, pad_target=trajectory,
             grip_target=opening, bilateral_substeps=both, contact_impulse=impulse,
             contact_names=names, object_initial_settled=obj0, phase_frames=frames,
             physics_steps=step_counts, ring_only_substeps=ring_only_steps,
             phys_time=np.cumsum(step_counts)/250, source_frame=np.arange(length))
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    if render:
        imageio.mimsave(out / "external_and_ego.gif", clip, duration=1000/15, loop=0)
        imageio.mimsave(out / "robot_fpv.mp4", fpv, fps=30, macro_block_size=8)
    print(json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reach", type=float, default=.46)
    args = parser.parse_args()
    replay(args.output, args.full, args.render, args.reach)

