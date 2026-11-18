#!/usr/bin/env python3
"""Generic continuous R1 replay for one tracked object and arbitrary supports.

Package contract:
  paired_targets.npz: src_frames, video_fps, grip_states, grip[N,2],
    T_world_pad[N,4,4], T_world_object[N,4,4], object_ids[N], support_ids[N].
    Optional co-timed left arm: T_world_pad_left[N,4,4], left_pad_valid[N],
    grip_left[N,2]. Object contact/release evidence refers to the primary right hand.
  geometry.json: objects[id]={mesh, mass_kg, ...}; supports[id]=
    {center, half_extents, yaw}; optional object collision_mesh/texture,
    support_height_m, release_support_fraction, base_reference tolerances.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import sapien.core as sapien
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from cb_large_range_sim import _build_scene, bp, lrt, rg
from cycle_gate import PairReleaseAdapter, ReleaseGate
from contact_replay_policy import contact_plan, lead_stations, protected_transport


def asset(package, name):
    path = Path(name)
    return path if path.is_absolute() else package / path


def add_object(scene, package, object_id, data, pose, material):
    mesh = asset(package, data["mesh"])
    collision = asset(package, data.get("collision_mesh", data["mesh"]))
    builder = scene.create_actor_builder()
    builder.set_physx_body_type("dynamic")
    builder.add_convex_collision_from_file(str(collision), material=material)
    builder.add_visual_from_file(str(mesh), material=rg._textured_render_material(str(asset(package, data['texture']))) if data.get('texture') else None)
    body = builder.build(name=str(object_id))
    body.set_pose(sapien.Pose(p=pose[:3, 3],
        q=Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]))
    for component in body.get_components():
        if hasattr(component, "mass"):
            component.mass = float(data["mass_kg"])
    return body


def resolved_supports(package, supports):
    return {name: {**data, **{field: str(asset(package, data[field]))
                             for field in ("collision_mesh", "visual_mesh") if field in data}}
            for name, data in supports.items()}


def add_supports(scene, supports, material, height):
    for name, data in supports.items():
        if "collision_mesh" in data:
            builder = scene.create_actor_builder()
            builder.add_nonconvex_collision_from_file(data["collision_mesh"], material=material)
            if "visual_mesh" in data:
                builder.add_visual_from_file(data["visual_mesh"])
            builder.build_static(name=str(name))
            continue
        center = np.asarray(data["center"], dtype=float)
        yaw = float(data.get("yaw", 0.))
        quat = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
        rg.create_box(scene, sapien.Pose(p=center-[0, 0, height/2], q=quat),
                      [*data["half_extents"], height/2], color=(.55, .58, .6),
                      name=str(name), is_static=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stable-support-frames", type=int, default=3)
    parser.add_argument("--support-fraction", type=float)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed relative to the source frame rate")
    parser.add_argument("--contact-policy", action="store_true",
                        help="Probe actual contacts, then replan the same pair with contact-aware transport")
    parser.add_argument("--progressive-motion", action="store_true",
                        help="Apply progressive base and torso costs after the unchanged contact probe")
    args = parser.parse_args()
    if args.progressive_motion and not args.contact_policy:
        parser.error("--progressive-motion requires --contact-policy")
    out = args.output.resolve()
    if args.contact_policy:
        probe_out = out/"contact_probe"
        replay(args, probe_out)
        with np.load(probe_out/"multicontact_trace.npz", allow_pickle=False) as probe:
            replay(args, out, probe)
    else:
        replay(args, out)


def replay(args, out, probe=None):
    package = args.package.resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = np.load(package/"paired_targets.npz")
    meta = json.loads((package/"geometry.json").read_text())
    supports, objects = resolved_supports(package, meta["supports"]), meta["objects"]
    required = ("src_frames", "video_fps", "grip_states", "grip",
                "T_world_pad", "T_world_object", "object_ids", "support_ids")
    missing = [k for k in required if k not in targets]
    if missing:
        raise ValueError("pair missing fields: " + ", ".join(missing))
    frames = np.asarray(targets["src_frames"], dtype=int)
    states = np.asarray(targets["grip_states"])
    grip = np.asarray(targets["grip"], dtype=float)
    pad = np.asarray(targets["T_world_pad"], dtype=float)
    reference = np.asarray(targets["T_world_object"], dtype=float)
    object_ids = np.asarray(targets["object_ids"]).astype(str)
    support_ids = np.asarray(targets["support_ids"]).astype(str)
    n = len(frames)
    if n < 2 or not np.array_equal(np.diff(frames), np.ones(n-1, dtype=int)):
        raise ValueError("paired trajectory must contain contiguous video frames")
    if any(len(x) != n for x in (states, grip, pad, reference, object_ids, support_ids)):
        raise ValueError("pair arrays must have equal frame counts")
    if grip.shape != (n, 2) or pad.shape != (n, 4, 4) or reference.shape != (n, 4, 4):
        raise ValueError("expected grip[N,2] and pose arrays[N,4,4]")
    if not np.isfinite(grip).all() or not np.isfinite(pad).all() or not np.isfinite(reference).all():
        raise ValueError("pair contains non-finite targets")
    pad_left = np.asarray(targets["T_world_pad_left"], float) if "T_world_pad_left" in targets else None
    if pad_left is None and ("left_pad_valid" in targets or "grip_left" in targets):
        raise ValueError("left_pad_valid and grip_left require T_world_pad_left")
    left_valid = np.zeros(n, dtype=bool)
    grip_left = None
    if pad_left is not None:
        if pad_left.shape != pad.shape:
            raise ValueError("T_world_pad_left must align with the right pad targets")
        finite = np.isfinite(pad_left).all(axis=(1, 2))
        left_valid = np.asarray(targets["left_pad_valid"], dtype=bool) if "left_pad_valid" in targets else finite.copy()
        if left_valid.shape != (n,) or np.any(left_valid & ~finite):
            raise ValueError("left_pad_valid must label finite left poses on the same timeline")
        if "grip_left" in targets:
            grip_left = np.asarray(targets["grip_left"], float)
            if grip_left.shape != grip.shape or not np.isfinite(grip_left[left_valid]).all():
                raise ValueError("grip_left must have two finite joint targets on each valid left row")
    source_pad = pad
    source_pad_left = pad_left
    plans = []
    priority = np.ones((n, 2))
    if probe is not None:
        observed = np.ones(n, bool)
        support_steps = probe["support_impulse_by_name"].sum(axis=0)
        if "ground_contact_substeps" in probe:
            support_steps = support_steps + (probe["ground_contact_substeps"] > 0)
        right_plan = contact_plan(frames, pad, reference, grip.max(1), observed,
                                  probe["actual_pad"], probe["actual_object"],
                                  probe["bilateral_substeps"], support_steps, probe["physics_substeps"])
        if "placement_refinement" in meta:
            with np.load(asset(package, meta["placement_refinement"]), allow_pickle=False) as refine:
                refined = np.asarray(refine["T_world_pad"], dtype=float)
                mask = np.asarray(refine["valid"], dtype=bool)
                if (not np.array_equal(refine["src_frames"], frames) or refined.shape != pad.shape
                        or mask.shape != (n,) or np.any(mask & ~right_plan.holder)
                        or not np.isfinite(refined[mask]).all()):
                    raise ValueError("placement refinement must align with the probed held rows")
                right_plan.pad[mask] = refined[mask]
                right_plan.command_anchor[mask] = np.linalg.solve(reference[mask], refined[mask])
        plans.append(right_plan)
        pad = right_plan.pad
        if grip_left is not None and "actual_pad_left" in probe:
            left_plan = contact_plan(frames, pad_left, reference, grip_left.max(1), left_valid,
                                     probe["actual_pad_left"], probe["actual_object"],
                                     probe["bilateral_substeps_left"], support_steps, probe["physics_substeps"])
            plans.append(left_plan)
            pad_left = left_plan.pad
            priority[left_plan.holder & ~right_plan.holder, 0] = .25
            priority[right_plan.holder & ~left_plan.holder & left_valid, 1] = .25
        else:
            priority[right_plan.holder & left_valid, 1] = .25

    object_names = set(object_ids)
    if len(object_names) != 1:
        raise ValueError("this runtime supports one tracked object across multiple supports")
    object_id = next(iter(object_names))
    if object_id not in objects:
        raise ValueError(f"no geometry.objects entry for {object_id}")
    active_supports = set(support_ids) - {""}
    if not active_supports or not active_supports <= set(supports):
        raise ValueError("support_ids must name geometry.supports entries")
    fps = float(np.asarray(targets["video_fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0 or fps > 250:
        raise ValueError("video_fps must be in (0,250]")
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        raise ValueError("speed must be in (0,1]")
    playback_fps = fps * args.speed
    mass = float(objects[object_id]["mass_kg"])
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("object mass_kg must be positive")
    support_fraction = args.support_fraction
    if support_fraction is None:
        support_fraction = float(meta.get("release_support_fraction", .5))
    if not 0 < support_fraction <= 1 or args.stable_support_frames < 1:
        raise ValueError("support fraction and stable frame count must be positive")

    scene, robot = _build_scene(planar_base=True)
    if not np.isclose(scene.get_timestep(), 1/250):
        raise ValueError("R1 scene must run at 250 Hz")
    material = scene.create_physical_material(1., .9, 0.)
    add_supports(scene, supports, material, float(meta.get("support_height_m", .045)))
    body = add_object(scene, package, object_id, objects[object_id], reference[0], material)

    joints = robot.get_active_joints()
    ji = {joint.get_name(): i for i, joint in enumerate(joints)}
    planar = [ji[name] for name in ("planar_joint_x", "planar_joint_y", "planar_joint_yaw")]
    arm = [ji[f"right_arm_joint{i}"] for i in range(1, 7)]
    torso = [ji[f"torso_joint{i}"] for i in range(1, 5)]
    arm += torso
    fingers = [ji[f"right_gripper_finger_joint{i}"] for i in (1, 2)]
    left = [ji[f"left_arm_joint{i}"] for i in range(1, 7)]
    q = robot.get_qpos()
    q[left] = rg.R1_LEFT_START
    robot.set_qpos(q)
    limits = np.vstack((bp._limits("right"), [joints[i].get_limits()[0] for i in torso]))
    secondary = None
    left_fingers = [ji[f"left_gripper_finger_joint{i}"] for i in (1, 2)]
    if np.any(left_valid):
        arm += left
        limits = np.vstack((limits, bp._limits("left")))
        secondary = [(P[:3, 3], P[:3, :3]) if valid else None for P, valid in zip(pad_left, left_valid)]
    pitch_limit = np.asarray(meta.get("torso_pitch_limit_deg", [12., 16., 18.]), dtype=float)
    tracking_margin = np.asarray(meta.get("torso_pitch_tracking_margin_deg", [3., 5., 5.]), dtype=float)
    if (pitch_limit.shape != (3,) or tracking_margin.shape != (3,) or not np.isfinite(pitch_limit).all()
            or not np.isfinite(tracking_margin).all() or np.any(tracking_margin < 0.)
            or np.any(pitch_limit <= tracking_margin) or np.any(pitch_limit > 90.)):
        raise ValueError("invalid torso pitch limits or tracking margin")
    w_base = float(meta.get("base_reference_weight", 3.))
    if not np.isfinite(w_base) or w_base < 0.:
        raise ValueError("base_reference_weight must be finite and nonnegative")
    w_ori = float(meta.get("ik_orientation_weight", 4.))
    if not np.isfinite(w_ori) or w_ori <= 0.:
        raise ValueError("ik_orientation_weight must be finite and positive")
    for i, joint in enumerate(joints):
        if i in fingers or (grip_left is not None and secondary is not None and i in left_fingers):
            joint.set_drive_property(stiffness=4000., damping=120., force_limit=28.)
        elif i in planar:
            joint.set_drive_property(stiffness=35000., damping=3000., force_limit=5000.)
        else:
            joint.set_drive_property(stiffness=2400., damping=150.)

    base_ref = np.asarray(targets["base_reference_se2"], dtype=float) if "base_reference_se2" in targets else None
    if base_ref is None:
        base_ref = np.zeros((n, 3))
        base_ref[:, :2] = pad[:, :2, 3]
        base_ref[:, 0] -= max(float(d["half_extents"][0]) for d in supports.values()) + .30
    else:
        base_ref = base_ref.copy()
    if base_ref.shape != (n, 3) or not np.isfinite(base_ref).all():
        raise ValueError("base_reference_se2 must have shape [N,3]")
    base_ref[:, :2] -= np.asarray(rg.R1_BASE_P)[:2]
    base_ref[:, 2] = np.unwrap(base_ref[:, 2])
    station_phase = np.arange(n, dtype=float)
    original_base_ref = base_ref
    if plans:
        width = grip.max(1)
        edges = np.flatnonzero(states[1:] != states[:-1])+1
        lookahead = max(1, round(.14*fps))
        cues = [frames[k] for k in edges
                if np.max(np.abs(width[k:min(n,k+lookahead)]-width[k-1])) > .002]
        base_ref, station_phase = lead_stations(base_ref, frames, plans, fps, cues)
    target_p, target_R = pad[:, :3, 3], pad[:, :3, :3]
    arm_seed = np.asarray(meta.get("ik_initial_arm_seed_rad", q[arm]), dtype=float)
    if arm_seed.shape != (len(arm),) or not np.isfinite(arm_seed).all():
        raise ValueError("ik_initial_arm_seed_rad must match the active arm and torso joints")
    solve_options = dict(
        ref_yaw=base_ref[:, 2], n_local=1, n_rand=0, max_nfev=70,
        base_range=float(meta.get('base_reference_xy_tolerance_m', .35)),
        yaw_max=np.radians(float(meta.get('base_reference_yaw_tolerance_deg', 30.))),
        base_step_max=.025, joint_step_max=10., yaw_step_max=3.,
        w_pos=42., w_ori=w_ori, w_base=w_base, w_yaw=.8, w_bcont=4., w_cont=1.5,
        w_limit=.5, w_posture=.15, acceleration_weight=.5,
        q_neutral=np.r_[limits[:6].mean(1), np.zeros(4), q[arm[10:]]],
        initial_seed=np.r_[0., 0., 0., arm_seed], torso_pitch=np.radians(pitch_limit-tracking_margin),
        optimize_continuity=True, secondary_tgt=secondary, hand_task_scales=priority)
    solve_inputs = (robot, "right", arm, limits, base_ref[:,0], base_ref[:,1],
                    list(zip(target_p,target_R)), target_p, bp.se2_pose, bp._gripper_pad_fk)
    base, q_plan, pos_error, ori_error = lrt.solve_wbc_track(*solve_inputs, **solve_options)
    optimized_motion_rows = 0
    if plans and args.progressive_motion and plans[0].source_windows:
        more_base, more_q, _, _ = lrt.solve_wbc_track(*solve_inputs, **solve_options,
                                                   progressive_body_motion=True)
        original = np.column_stack([base, q_plan])
        proposal = np.column_stack([more_base, more_q])
        steps = np.r_[.025/np.sqrt(2), .025/np.sqrt(2), np.radians(3), np.full(len(arm), np.radians(10))]
        joined = protected_transport(frames, plans, fps, original, proposal, steps)
        constrained = np.abs(lrt.torso_link_pitch(joined[:,9:12])) <= np.radians(pitch_limit-tracking_margin)+1e-8
        in_limits = np.all(joined[:,3:] >= limits[:,0]-1e-8) and np.all(joined[:,3:] <= limits[:,1]+1e-8)
        if np.all(constrained) and in_limits:
            q_new = joined[:,3:]
            base_new = joined[:,:3]
            rows = np.flatnonzero(np.any(joined != original, axis=1))
            for k in rows:
                robot.set_root_pose(bp.se2_pose(*base_new[k]))
                p, R = bp._gripper_pad_fk(robot,"right",arm,q_new[k])
                pos_error[k] = 1000*np.linalg.norm(p-target_p[k])
                ori_error[k] = np.rad2deg(Rotation.from_matrix(R@target_R[k].T).magnitude())
                if secondary is not None and secondary[k] is not None:
                    p2, R2 = bp._gripper_pad_fk(robot,"left",arm,q_new[k])
                    goal_p, goal_R = secondary[k]
                    pos_error[k] = max(pos_error[k], 1000*np.linalg.norm(p2-goal_p))
                    ori_error[k] = max(ori_error[k], np.rad2deg(Rotation.from_matrix(R2@goal_R.T).magnitude()))
            base, q_plan = base_new, q_new
            optimized_motion_rows = len(rows)
    if np.max(np.abs(np.diff(base[:, :2], axis=0))) > .025/np.sqrt(2.) + 1e-9:
        raise ValueError("base plan exceeds R1 per-frame travel limit")

    initial_xy = np.asarray(rg.R1_BASE_P)[:2]
    initial_yaw = Rotation.from_quat(np.asarray(rg.R1_BASE_Q)[[1, 2, 3, 0]]).as_euler("xyz")[2]
    base_offset = np.r_[initial_xy, initial_yaw]
    robot.set_root_pose(sapien.Pose())
    q[planar] = base[0] + base_offset
    q[arm], q[fingers] = q_plan[0], grip[0]
    if grip_left is not None and left_valid[0]:
        for j, value in zip(left_fingers, grip_left[0]):
            q[j] = np.clip(value, *joints[j].get_limits()[0])
    robot.set_qpos(q)
    for joint, value in zip(joints, q):
        joint.set_drive_target(float(value))

    steps = np.diff(np.floor(np.arange(n+1)*250/playback_fps).astype(int))
    gate = PairReleaseAdapter(targets, args.stable_support_frames)
    actual_object = np.empty((n, 4, 4))
    actual_pad = np.empty((n, 4, 4))
    actual_pad_left = np.empty_like(actual_pad) if secondary is not None else None
    left_link = robot.find_link_by_name("left_gripper_link") if secondary is not None else None
    bilateral_substeps = np.zeros(n, dtype=int)
    bilateral_left = np.zeros(n, dtype=int)
    ground_contact = np.zeros(n, dtype=int)
    measured_opening_left = np.zeros(n, bool)
    authorized_left = np.zeros(n, bool)
    support_only_substeps = np.zeros(n, dtype=int)
    held_support_impulse = np.zeros(n)
    support_collision_substeps = np.zeros(n, dtype=int)
    measured_opening = np.zeros(n, dtype=bool)
    support_impulse = {name: np.zeros(n) for name in supports}
    authorized = np.zeros(n, dtype=bool)
    withdrawal = np.zeros(n, dtype=bool)
    regrasp_contact = np.zeros(n, dtype=bool)
    executed_grip = np.empty((n, 2))
    q_actual = np.empty((n, len(joints)))
    commanded_width = float(np.max(grip[0]))
    finger_links = {f"right_gripper_finger_link{i}" for i in (1, 2)}
    left_finger_links = {f"left_gripper_finger_link{i}" for i in (1, 2)}
    left_gates = {}
    left_holding = False
    left_contact_steps = 0
    right_holding = False
    right_contact_steps = 0
    left_command = q[left_fingers].copy()
    left_open_reference = None
    robot_links = {link.entity.get_name() for link in robot.get_links()}
    withdrawal_seen = False
    open_reference = None
    pad_link = robot.find_link_by_name("right_gripper_link")
    issued_base = np.empty_like(base)
    issued_arm = np.empty_like(q_plan)
    issued_pad = pad.copy()
    seat_wait = np.zeros(n, bool)
    placement_request = np.full(n,-1,int)
    if plans:
        for window in plans[0].source_windows:
            event = window["release_intent"]
            if event is None:
                continue
            k = int(np.searchsorted(frames, event))
            ahead = max(1, round(.6*playback_fps))
            if support_ids[k] and not np.any(probe["withdrawal_clear"][k:min(n,k+2*ahead)]):
                placement_request[max(1,k-ahead):k+1] = k
    seat_hold_support = None
    used_seat_hold = False
    seat_descent = 0.
    seat_descent_mm = np.zeros(n)
    seat_kinematics = None
    for i, substeps in enumerate(steps):
        support = support_ids[i]
        finger_contacts = set()
        frame_load = {name: 0. for name in supports}
        previous_base = issued_base[i-1] if i else base[i]
        previous_arm = issued_arm[i-1] if i else q_plan[i]
        desired_width = float(plans[0].aperture[i] if plans else np.max(grip[i]))
        opening = desired_width > commanded_width
        current_auth = bool(i and authorized[i-1] and support == support_ids[i-1])
        if i and (support != seat_hold_support or desired_width < commanded_width):
            seat_hold_support = None
        if plans and i and right_holding and not current_auth and support and support == support_ids[i-1]:
            loaded = held_support_impulse[i-1] >= support_fraction*mass*9.81*steps[i-1]/250
            if loaded:
                k = placement_request[i]
                released_pose = reference[k] if k >= 0 else reference[i]
                angular = released_pose[:3,2] @ actual_object[i-1,:3,2]
                nearby = np.linalg.norm(released_pose[:2,3]-actual_object[i-1,:2,3]) <= .04
                samples = slice(max(0,i-2), i)
                stable = i >= 2 and np.all(support_ids[samples] == support) and np.all(
                    held_support_impulse[samples] >= support_fraction*mass*9.81*steps[samples]/250)
                supported_seat = (stable and k >= 0 and support == support_ids[k]
                                  and nearby and angular >= np.cos(np.deg2rad(8.)))
                would_leave = opening and (np.linalg.norm(pad[i,:3,3]-pad[i-1,:3,3]) > .004
                                          or np.linalg.norm(pad[i,:3,:3]-pad[i-1,:3,:3]) > .04)
                if supported_seat or would_leave:
                    seat_hold_support = support
        seat_wait[i] = seat_hold_support is not None
        if seat_wait[i]:
            issued_pad[i] = issued_pad[i-1]
        base_target = previous_base if seat_wait[i] else base[i]
        arm_target = previous_arm if seat_wait[i] else q_plan[i]
        if not seat_wait[i]:
            seat_descent = 0.
        if (seat_wait[i] and not current_auth and right_holding and seat_descent < .008
                and bilateral_substeps[i-1] >= np.ceil(.6*steps[i-1])
                and not support_collision_substeps[i-1]
                and held_support_impulse[i-1] < support_fraction*mass*9.81*steps[i-1]/250):
            if seat_kinematics is None:
                seat_kinematics = _build_scene()
                scratch_keys = {joint.get_name(): j for j,joint in enumerate(seat_kinematics[1].get_active_joints())}
                scratch_arm = [scratch_keys[joints[j].get_name()] for j in arm]
            scratch = seat_kinematics[1]
            scratch_q = scratch.get_qpos()
            scratch_q[scratch_arm] = previous_arm
            scratch.set_root_pose(bp.se2_pose(*base_target))
            scratch.set_qpos(scratch_q)
            count = 6 if left_holding else 10
            indices, seed = scratch_arm[:count], previous_arm[:count]
            old_p, old_r = bp._gripper_pad_fk(scratch, "right", indices, seed)
            dz = min(.0006, .008-seat_descent)
            target_p = old_p - [0., 0., dz]
            lo = np.maximum(limits[:count,0], seed-np.deg2rad(2.))
            hi = np.minimum(limits[:count,1], seed+np.deg2rad(2.))
            solved = bp.limit_aware_resolve(scratch, "right", indices, seed,
                                           lo, hi, w_pos=42., w_ori=10.,
                                           target=(target_p, old_r))
            new_p, new_r = bp._gripper_pad_fk(scratch, "right", indices, solved)
            within_pitch = count == 6 or np.all(np.abs(lrt.torso_link_pitch(solved[6:9]))
                                               <= np.radians(pitch_limit-.5*tracking_margin))
            if within_pitch and new_p[2] < old_p[2] and np.linalg.norm(new_p-target_p) < .0003:
                arm_target = previous_arm.copy()
                arm_target[:count] = solved
                issued_pad[i,:3,3] = new_p
                issued_pad[i,:3,:3] = new_r
                seat_descent += old_p[2]-new_p[2]
        seat_descent_mm[i] = seat_descent*1000
        delta = base_target-previous_base
        delta_arm = arm_target-previous_arm
        if used_seat_hold:
            # Rejoin from the last issued seat pose without skipping a joint
            # or base step when the source clock has already moved ahead.
            bound = np.r_[.025/np.sqrt(2), .025/np.sqrt(2), np.deg2rad(3.)]
            delta = np.clip(delta, -bound, bound)
            delta_arm = np.clip(delta_arm, -np.deg2rad(10.), np.deg2rad(10.))
        used_seat_hold |= seat_wait[i]
        issued_base[i] = previous_base + delta if used_seat_hold else base[i]
        issued_arm[i] = previous_arm + delta_arm if used_seat_hold else q_plan[i]
        if right_holding and opening and not current_auth:
            desired_width = commanded_width
        if current_auth and desired_width > commanded_width and open_reference is None:
            open_reference = q_actual[i-1, fingers].copy()
        left_load = 0.
        left_clear_steps = 0
        left_asked = float(np.max(grip_left[i])) if grip_left is not None and left_valid[i] else float(np.max(left_command))
        left_current_auth = bool(i and authorized_left[i-1] and support == support_ids[i-1])
        left_gate = left_gates.setdefault(support, ReleaseGate(args.stable_support_frames)) if support and grip_left is not None else None
        if grip_left is not None and left_valid[i]:
            desired_left = grip_left[i]
            if len(plans) > 1 and plans[1].holder[i]:
                desired_left = np.minimum(desired_left, plans[1].aperture[i])
                left_asked = float(np.max(desired_left))
            if left_holding and left_asked > max(left_command) and not left_current_auth:
                desired_left = left_command
            if left_current_auth and left_asked > max(left_command) and left_open_reference is None:
                left_open_reference = q_actual[i-1, left_fingers].copy()
            for j, value in zip(left_fingers, desired_left):
                bounds = joints[j].get_limits()[0]
                joints[j].set_drive_target(float(np.clip(value, *bounds)))
            left_command = np.asarray(desired_left).copy()
        for j in fingers:
            joints[j].set_drive_target(desired_width)
        executed_grip[i] = desired_width
        for sub in range(int(substeps)):
            u = (sub+1)/substeps
            for j, value in zip(arm, previous_arm+delta_arm*u):
                joints[j].set_drive_target(float(value))
            for j, value, velocity in zip(planar, previous_base+delta*u+base_offset, delta*250/substeps):
                joints[j].set_drive_target(float(value))
                joints[j].set_drive_velocity_target(float(velocity))
            rg.step_sim(scene, robot)
            sub_fingers = set()
            sub_left = set()
            sub_load = 0.
            robot_hit_support = False
            ground_hit = False
            has_other_object_contact = False
            for contact in scene.get_contacts():
                a, b = (body.get_entity().get_name() for body in contact.bodies)
                if sum(float(np.linalg.norm(point.impulse)) for point in contact.points) <= 1e-6:
                    continue
                if object_id not in (a, b):
                    robot_hit_support |= (a in robot_links and b in supports) or (b in robot_links and a in supports)
                    continue
                other = b if a == object_id else a
                ground_hit |= other == "ground"
                if other in supports:
                    vertical = sum(abs(float(point.impulse[2])) for point in contact.points
                                   if abs(float(point.normal[2])) > .8)
                    frame_load[other] += vertical
                    if other == support:
                        sub_load += vertical
                if other in finger_links:
                    sub_fingers.add(other)
                elif other in left_finger_links:
                    sub_left.add(other)
                elif other != support:
                    has_other_object_contact = True
            ground_contact[i] += int(ground_hit)
            finger_contacts.update(sub_fingers)
            if finger_links <= sub_fingers:
                bilateral_substeps[i] += 1
                if not robot_hit_support:
                    held_support_impulse[i] += sub_load
            if left_finger_links <= sub_left:
                bilateral_left[i] += 1
                if not robot_hit_support:
                    left_load += sub_load
            if sub_load > 1e-6 and not sub_left and not sub_fingers and not has_other_object_contact:
                left_clear_steps += 1
            if sub_load > 1e-6 and not sub_fingers and not sub_left and not has_other_object_contact:
                support_only_substeps[i] += 1
            support_collision_substeps[i] += int(robot_hit_support)
        bilateral = bilateral_substeps[i] > 0
        if bilateral_substeps[i] >= np.ceil(.6*substeps):
            right_contact_steps += substeps
            right_holding |= right_contact_steps >= 12
        else:
            right_contact_steps = 0
        for name in supports:
            support_impulse[name][i] = frame_load[name]
        actual_object[i] = body.get_pose().to_transformation_matrix()
        actual_pad[i] = pad_link.entity.get_pose().to_transformation_matrix()
        actual_pad[i, :3, 3] += actual_pad[i, :3, 0] * rg.PAD_FWD
        if actual_pad_left is not None:
            actual_pad_left[i] = left_link.entity.get_pose().to_transformation_matrix()
            actual_pad_left[i, :3, 3] += actual_pad_left[i, :3, 0] * rg.PAD_FWD
        q_actual[i] = robot.get_qpos()
        measured_opening[i] = open_reference is not None and np.all(q_actual[i,fingers] > open_reference + .001)
        key = (object_id, support)
        threshold = support_fraction*mass*9.81*int(substeps)/250
        withdrawal_clear = bool(current_auth and measured_opening[i] and not finger_contacts
                                and frame_load.get(support, 0.) >= threshold
                                and support_only_substeps[i] == substeps)
        withdrawal[i] = withdrawal_clear
        if left_gate is not None:
            if bilateral_left[i] >= np.ceil(.6*substeps):
                left_contact_steps += substeps
                left_holding |= left_contact_steps >= 12
            else:
                left_contact_steps = 0
            measured_opening_left[i] = (left_open_reference is not None
                                        and np.all(q_actual[i, left_fingers] > left_open_reference + .001))
            left_clear = bool(left_current_auth and measured_opening_left[i]
                              and left_clear_steps == substeps and frame_load[support] >= threshold)
            # A deliberate aperture increase after a verified left hold requests release.
            # The right and left gates consume independent finger evidence.
            left_open_intent = left_holding and left_asked > max(left_command) + .001
            authorized_left[i] = left_gate.update(i, left_open_intent, left_asked, support,
                                                  left_load, threshold, bilateral_left[i] > 0, left_clear)
            if left_clear:
                left_holding = False
                left_open_reference = None
        thresholds = {key: threshold}
        loads = {key: held_support_impulse[i] if support_collision_substeps[i] == 0 else 0.}
        bilateral_map = {key: bilateral and right_holding}
        auth = gate.update(i, loads, thresholds, bilateral_map, withdrawal_clear)
        authorized[i] = auth
        if withdrawal_clear:
            seat_hold_support = None
            right_holding = False
            withdrawal_seen = True
            open_reference = None
        elif withdrawal_seen and bilateral and not opening:
            regrasp_contact[i] = True
            withdrawal_seen = False
        commanded_width = desired_width

    events = np.flatnonzero(authorized & ~np.r_[False, authorized[:-1]])
    regrasp_cycles = []
    for event in events:
        withdrawn = np.flatnonzero(withdrawal[event+1:])
        if len(withdrawn):
            withdrawn_at = event + 1 + int(withdrawn[0])
            regrasped = np.flatnonzero(regrasp_contact[withdrawn_at+1:])
            if len(regrasped):
                regrasp_cycles.append([int(frames[event]), int(frames[withdrawn_at]),
                                       int(frames[withdrawn_at+1+regrasped[0]])])
    actual_pair = np.linalg.inv(actual_pad) @ actual_object
    target_pair = np.linalg.inv(pad) @ reference
    pair_error = np.linalg.norm(actual_pair[:, :3, 3]-target_pair[:, :3, 3], axis=1)
    auxiliary = {} if actual_pad_left is None else dict(
        actual_pad_left=actual_pad_left, target_pad_left=pad_left, left_pad_valid=left_valid,
        source_pad_left=source_pad_left)
    policy_fields = dict(source_pad=source_pad, commanded_pad=issued_pad, hand_task_scales=priority,
                         bilateral_substeps_left=bilateral_left, ground_contact_substeps=ground_contact,
                         authorized_release_left=authorized_left,
                         measured_opening_left=measured_opening_left, base_reference_station_phase=station_phase,
                         seat_hold_rows=seat_wait, seat_descent_mm=seat_descent_mm)
    for name, plan in zip(("right", "left"), plans):
        policy_fields["held_plan_"+name] = plan.holder
        policy_fields["measured_grasp_anchor_"+name] = plan.measured_anchor
        policy_fields["command_grasp_anchor_"+name] = plan.command_anchor
    np.savez_compressed(out/"multicontact_trace.npz",
        source_frames=frames, source_time_frames=frames.astype(float), physics_substeps=steps, source_fps=fps, playback_fps=playback_fps,
        sim_time_s=np.cumsum(steps)/250, actual_object=actual_object,
        actual_pad=actual_pad, bilateral=bilateral_substeps > 0, authorized_release=authorized,
        bilateral_substeps=bilateral_substeps, held_support_impulse=held_support_impulse,
        support_only_substeps=support_only_substeps, support_collision_substeps=support_collision_substeps,
        measured_opening=measured_opening,
        release_event_indices=events, support_ids=support_ids,
        withdrawal_clear=withdrawal, regrasp_contact=regrasp_contact,
        support_impulse_by_name=np.array([support_impulse[k] for k in supports]),
        support_names=np.array(list(supports)), executed_grip=executed_grip,
        q_actual=q_actual, target_pair=target_pair, pair_position_error=pair_error,
        base_se2=q_actual[:, planar]-base_offset, base_se2_target=issued_base,
        arm_joint_indices=np.array(arm), arm_joint_targets=issued_arm,
        plan_position_error_mm=pos_error, plan_orientation_error_deg=ori_error, **auxiliary, **policy_fields)
    result = dict(runtime="r1_multicontact", frames=n, physics_substeps=int(steps.sum()),
        source_fps=fps, playback_fps=playback_fps, simulation_seconds=float(steps.sum()/250),
        object_id=object_id, active_supports=sorted(active_supports),
        left_goal_frames=int(left_valid.sum()), base_reference_weight=w_base,
        contact_policy=probe is not None, progressive_body_motion=bool(probe is not None and args.progressive_motion),
        optimized_motion_rows=optimized_motion_rows, station_reference_advance_rows=float((station_phase-np.arange(n)).max()),
        station_reference_change_mm=float(np.linalg.norm(base_ref[:,:2]-original_base_ref[:,:2], axis=1).max()*1000),
        probe_contact_windows={name: plan.source_windows for name, plan in zip(("right", "left"), plans)},
        bilateral_contact_frames_left=int((bilateral_left > 0).sum()),
        object_ground_contact_frames=int((ground_contact > 0).sum()),
        seat_hold_frames=int(seat_wait.sum()), seat_descent_max_mm=float(seat_descent_mm.max()),
        release_event_source_frames_left=frames[np.flatnonzero(authorized_left & ~np.r_[False, authorized_left[:-1]])].tolist(),
        ik_orientation_weight=w_ori,
        executed_torso_pitch_max_deg=np.degrees(np.abs(lrt.torso_link_pitch(q_actual[:,torso[:3]])).max(axis=0)).tolist(),
        release_event_source_frames=frames[events].tolist(),
        regrasp_cycles=regrasp_cycles,
        bilateral_contact_frames=int((bilateral_substeps > 0).sum()),
        held_support_frames=int((held_support_impulse > 0).sum()),
        measured_open_source_frames=frames[np.flatnonzero(measured_opening & ~np.r_[False, measured_opening[:-1]])].tolist(),
        withdrawal_clear_source_frames=frames[np.flatnonzero(withdrawal)].tolist(),
        robot_support_collision_frames=int((support_collision_substeps > 0).sum()),
        max_pair_translation_error_m=float(pair_error.max()))
    (out/"verdict.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

