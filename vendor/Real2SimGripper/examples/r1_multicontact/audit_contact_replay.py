#!/usr/bin/env python3
"""Summarize actual R1 wrist readback, base motion, and load transfer."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from replay_multicontact import lrt, rg


def p95(v):
    return None if not len(v) else float(np.quantile(v, .95))


def pose_error(actual, target):
    return (np.linalg.norm(actual[:, :3, 3]-target[:, :3, 3], axis=1)*1000,
            np.rad2deg(Rotation.from_matrix(actual[:, :3, :3] @ target[:, :3, :3].transpose(0,2,1)).magnitude()))


def audit(package, out):
    pair = np.load(package/"paired_targets.npz", allow_pickle=False)
    data = np.load(out/"multicontact_trace.npz", allow_pickle=False)
    verdict = json.loads((out/"verdict.json").read_text())
    frames = data["source_frames"]
    assert np.array_equal(frames, pair["src_frames"])
    t = data["sim_time_s"]
    dt = np.diff(t)
    base = data["base_se2"]
    delta = np.diff(base, axis=0)
    velocity = delta/dt[:, None]
    midpoint_dt = .5*(dt[1:]+dt[:-1])
    acceleration = np.diff(velocity, axis=0)/midpoint_dt[:, None]
    torso_q = data["q_actual"][:, data["arm_joint_indices"][6:9]]
    torso_pitch = np.rad2deg(lrt.torso_link_pitch(torso_q))
    torso_speed = np.diff(torso_pitch,axis=0)/dt[:,None]
    torso_accel = np.diff(torso_speed,axis=0)/midpoint_dt[:,None]
    steps = data["physics_substeps"]
    n = len(frames)
    cols = dict(source_frame=frames, sim_time_s=t, base_x_m=base[:,0], base_y_m=base[:,1],
                base_yaw_deg=np.rad2deg(base[:,2]),
                base_speed_m_s=np.r_[np.nan, np.linalg.norm(velocity[:,:2],axis=1)],
                base_yaw_speed_deg_s=np.r_[np.nan,np.rad2deg(velocity[:,2])],
                torso_link1_pitch_deg=torso_pitch[:,0], torso_link2_pitch_deg=torso_pitch[:,1],
                torso_link3_pitch_deg=torso_pitch[:,2],
                right_bilateral_steps=data["bilateral_substeps"],
                left_bilateral_steps=data["bilateral_substeps_left"],
                held_support_impulse_ns=data["held_support_impulse"],
                support_only_steps=data["support_only_substeps"], physics_steps=steps,
                release_authorized=data["authorized_release"], measured_opening=data["measured_opening"],
                support_id=data["support_ids"], robot_support_collision_steps=data["support_collision_substeps"],
                left_release_authorized=data["authorized_release_left"],
                left_measured_opening=data["measured_opening_left"])
    all_wrists = {}
    for side in ("right", "left"):
        right = side == "right"
        actual_key = "actual_pad" if right else "actual_pad_left"
        if actual_key not in data:
            all_wrists[side] = dict(target_rows=0)
            continue
        reference = pair["T_world_pad" if right else "T_world_pad_left"]
        command = data["commanded_pad" if right else "target_pad_left"]
        valid = np.ones(n,bool) if right else data["left_pad_valid"]
        has_observation_mask = bool(right and "source_wrist_observed" in pair)
        observed = pair["source_wrist_observed"] & valid if has_observation_mask else valid
        all_wrists[side] = dict(target_rows=int(valid.sum()),
                                source_observed_rows=int(observed.sum()) if has_observation_mask else None,
                                pair_error_rows=int(observed.sum()))
        for label, target, mask in (("pair",reference,observed), ("command",command,valid)):
            p, a = pose_error(data[actual_key][mask], target[mask])
            all_wrists[side][f"actual_vs_{label}_p95_mm_deg"] = [p95(p), p95(a)]
            for what, v in (("position_mm",p), ("angle_deg",a)):
                column = np.full(n,np.nan)
                column[mask] = v
                cols[f"{side}_{label}_{what}"] = column
        plan_key = "held_plan_"+side
        if plan_key in data:
            planned = data[plan_key]
            bilateral_steps = cols[f"{side}_bilateral_steps"]
            verified = planned & (bilateral_steps >= np.ceil(.6*steps))
            x = np.linalg.inv(data["actual_object"][verified]) @ data[actual_key][verified]
            p, a = pose_error(x, data["measured_grasp_anchor_"+side][verified])
            all_wrists[side].update(locked_plan_rows=int(planned.sum()), locked_verified_rows=int(verified.sum()),
                                    probed_grasp_deviation_p95_mm_deg=[p95(p),p95(a)])
            # Compare repeated command-space pose and actual grasp separately.
            actual = np.full(n,np.nan)
            actual[verified] = p
            cols[f"{side}_true_grasp_slip_mm"] = actual

    placements = []
    object_pose = data["actual_object"]
    support_names = data["support_names"].tolist()
    loads = data["support_impulse_by_name"]
    lookup = {name: k for k,name in enumerate(support_names)}
    active_load = np.array([loads[lookup[name], i] if name in lookup else 0.
                            for i,name in enumerate(data["support_ids"])])
    cols["active_support_impulse_ns"] = active_load
    cols["ground_contact_steps"] = data["ground_contact_substeps"] if "ground_contact_substeps" in data else np.zeros(n,int)
    for event in data["release_event_indices"]:
        next_edges = data["release_event_indices"][data["release_event_indices"] > event]
        bound = int(next_edges[0]) if len(next_edges) else n
        clear = np.flatnonzero(data["withdrawal_clear"][event:bound])+event
        opened = np.flatnonzero(data["measured_opening"][event:bound])+event
        back = slice(max(0,int(event)-2),int(event)+1)
        this = dict(support=str(data["support_ids"][event]), authorize=int(frames[event]),
                    measurable_preopening_source_frames=frames[back].tolist(),
                    simultaneous_held_load_ns=data["held_support_impulse"][back].tolist(),
                    bilateral_steps_at_authorization=int(data["bilateral_substeps"][event]),
                    measured_open=None if not len(opened) else int(frames[opened[0]]), independent=None)
        if len(clear):
            start = clear[0]
            end = start
            while end < bound and data["support_only_substeps"][end] == steps[end] and data["support_ids"][end] == this["support"]:
                end += 1
            xy = object_pose[start:end,:2,3]
            if len(xy):
                requested = pair["T_world_object"][end-1]
                vertical_axis = int(np.argmax(np.abs(requested[2,:3])))
                pitch = abs(object_pose[end-1,2,vertical_axis])
                final_rotation_error = Rotation.from_matrix(object_pose[end-1,:3,:3]@requested[:3,:3].T)
                this.update(independent=int(frames[start]), independent_until=int(frames[end-1]),
                            independent_steps=int(steps[start:end].sum()),
                            independent_seconds=float(steps[start:end].sum()/250),
                            maximum_xy_motion_mm=float(np.linalg.norm(xy-xy[0],axis=1).max()*1000),
                            object_local_vertical_axis="xyz"[vertical_axis],
                            final_tilt_deg=float(np.rad2deg(np.arccos(np.clip(pitch,0,1)))),
                            final_orientation_error_to_pair_deg=float(np.rad2deg(final_rotation_error.magnitude())),
                            final_xy_to_pair_mm=float(np.linalg.norm(xy[-1]-requested[:2,3])*1000))
        placements.append(this)

    stations = []
    if "base_reference_se2" in pair:
        reference = pair["base_reference_se2"].copy()
        reference[:,:2] -= np.asarray(rg.R1_BASE_P)[:2]
        window = int(max(1, np.round(.45*float(data["source_fps"]))))
        for side, windows in verdict.get("probe_contact_windows", {}).items():
            for contact in windows:
                for field in ("acquired", "release_intent"):
                    f = contact[field]
                    if f is None:
                        continue
                    k = int(np.searchsorted(frames,f))
                    r = np.arange(max(0,k-window),k+1)
                    error = np.linalg.norm(base[r,:2]-reference[k,:2],axis=1)
                    yaw = np.rad2deg(np.abs(np.angle(np.exp(1j*(base[r,2]-reference[k,2])))))
                    inside = (error <= .04) & (yaw <= 8)
                    got = np.flatnonzero(~inside)
                    first = (0 if not len(got) else int(got[-1])+1) if inside[-1] else None
                    stations.append(dict(side=side, event=field, frame=int(f),
                        measured_lead_s=None if first is None else float(t[k]-t[r[first]]),
                        frame_error_mm=float(error[-1]*1000), yaw_error_deg=float(yaw[-1])))

    with (out/"contact_replay_frames.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(zip(*cols.values(), strict=True))
    result = dict(frames=n, verdict=verdict, wrist=all_wrists, station_checks=stations,
                  placements=placements, torso=dict(
                    pitch_abs_max_deg=np.abs(torso_pitch).max(axis=0).tolist(),
                    pitch_speed_abs_p95_deg_s=[p95(abs(v)) for v in torso_speed.T],
                    pitch_acceleration_abs_p95_deg_s2=[p95(abs(v)) for v in torso_accel.T]),
                  base=dict(net_travel_m=float(np.linalg.norm(base[-1,:2]-base[0,:2])),
                    path_m=float(np.linalg.norm(delta[:,:2],axis=1).sum()),
                    xy_speed_p95_m_s=p95(np.linalg.norm(velocity[:,:2],axis=1)),
                    yaw_rate_p95_deg_s=p95(np.rad2deg(abs(velocity[:,2]))),
                    xy_accel_p95_m_s2=p95(np.linalg.norm(acceleration[:,:2],axis=1)),
                    yaw_accel_p95_deg_s2=p95(np.rad2deg(abs(acceleration[:,2])))))
    (out/"contact_replay_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.package, args.output)
