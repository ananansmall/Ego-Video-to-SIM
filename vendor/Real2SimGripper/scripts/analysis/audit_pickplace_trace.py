
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def read_trace(run, expected_sha):
    path = run / "exec_trace.npz"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise ValueError(f"{path}: verdict and trace SHA-256 differ")
    with np.load(path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in archive.files}
    frames = trace["source_frames"]
    if frames.ndim != 1 or not len(frames) or not np.all(np.diff(frames) == 1):
        raise ValueError(f"{path}: source frame IDs must be consecutive and nonempty")
    steps = trace["physics_substeps"]
    if steps.shape != frames.shape or not np.all(steps > 0) or not float(trace["dt"]) > 0:
        raise ValueError(f"{path}: invalid physics clock")
    with (run / "physics_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not np.array_equal(frames, [int(row["source_frame"]) for row in rows]):
        raise ValueError(f"{path}: contact CSV and trace timelines differ")
    furniture = np.array([int(row["robot_furniture_substeps"]) for row in rows])
    return trace, furniture


def audit_event(trace, furniture, event, drift_limit, minimum_support, slip_limit):
    frames = trace["source_frames"]
    lift, request, end = (event[key] for key in ("lift", "request", "end"))
    authorized = event["authorized"]
    if not (frames[0] <= lift < request <= end <= frames[-1]
            and (authorized is None or request <= authorized <= end)):
        raise ValueError(f"Invalid lift/request/authorization/end source IDs: {event}")
    lift_i, request_i, end_i = (int(value - frames[0]) for value in (lift, request, end))
    if request_i < 3:
        raise ValueError("Release needs three preceding source frames")
    authorize_i = None if authorized is None else int(authorized - frames[0])
    hand = trace["q_actual"][:, trace["finger_joint_indices"]]
    if hand.shape != (len(frames), 2):
        raise ValueError("Expected actual qpos for the two fingers")

    def measure_open(start, stop, reference_i):
        baseline = np.median(hand[reference_i - 3:reference_i], axis=0)
        matches = np.flatnonzero(np.all(hand[start:stop] > baseline + .001, axis=1))
        return start + int(matches[0]) if len(matches) else None

    early = measure_open(request_i, (end_i + 1) if authorize_i is None else authorize_i, request_i)
    measured = None if authorize_i is None else measure_open(authorize_i, end_i + 1, authorize_i)
    prior = np.arange(0) if authorize_i is None else np.arange(authorize_i - 3, authorize_i)
    bilateral = trace["bilateral"]
    vertical = trace["chair_vertical_impulse"]
    threshold = trace["chair_load_threshold"]
    load_verified = len(prior) == 3 and bool(np.all(
        (vertical[prior] >= threshold[prior]) & (bilateral[prior] > 0) & (furniture[prior] == 0)))
    latest_carry = end_i if authorize_i is None else authorize_i - 1
    load_after_lift = np.flatnonzero(
        vertical[lift_i:latest_carry + 1] >= threshold[lift_i:latest_carry + 1])
    carry_end = lift_i + int(load_after_lift[0]) - 1 if len(load_after_lift) else latest_carry
    carry = slice(lift_i, max(lift_i, carry_end) + 1)
    actual_object = trace["actual_object"]
    relative = np.linalg.solve(trace["actual_pad"][carry], actual_object[carry])
    shift = 1000 * np.linalg.norm(relative[:, :3, 3] - relative[0, :3, 3], axis=1)
    turn = np.rad2deg(Rotation.from_matrix(relative[0, :3, :3].T @ relative[:, :3, :3]).magnitude())
    steps = trace["physics_substeps"]
    target_only = trace["chair_only"] == steps
    settle = slice(end_i + 1, end_i + 1) if measured is None else slice(measured, end_i + 1)
    failures = np.flatnonzero(~target_only[settle])
    trailing_start = settle.start + (int(failures[-1]) + 1 if len(failures) else 0)
    trailing_support_s = float(steps[trailing_start:end_i + 1].sum() * trace["dt"])
    initially_supported = target_only[settle]
    first_break = np.flatnonzero(~initially_supported)
    first_segment_end = settle.start + (int(first_break[0]) if len(first_break) else len(initially_supported))
    support_from_open_s = float(steps[settle.start:first_segment_end].sum() * trace["dt"])
    last = actual_object[max(settle.start, end_i - 14):end_i + 1, :3, 3]
    drift = None if len(last) != 15 else float(1000 * np.linalg.norm(last - last[-1], axis=1).max())
    columns = trace["contact_columns"].tolist()
    impulses = trace["contact_impulses"]
    ground = impulses[lift_i:end_i + 1, columns.index("ground")]
    fingers = [i for i, name in enumerate(columns) if "finger" in name]
    if len(fingers) != 2:
        raise ValueError("Expected two finger contact columns")
    finger_contact_after_open = np.any(impulses[settle][:, fingers] > 1e-6, axis=1)
    after_first_support = int(np.argmax(initially_supported)) if initially_supported.any() else len(initially_supported)
    recontact = frames[settle][after_first_support:][finger_contact_after_open[after_first_support:]]
    moved_after_open = actual_object[settle, :3, 3]
    open_to_end_displacement = (None if measured is None else
        float(1000 * np.linalg.norm(moved_after_open - moved_after_open[0], axis=1).max()))
    hold = bool(np.all(bilateral[carry] > 0))
    checks = {
        "carry_has_bilateral_contact_each_frame": hold,
        "carry_slip_under_limit": float(shift.max()) <= slip_limit,
        "no_open_before_authorization": early is None,
        "three_frames_of_load_before_authorization": load_verified,
        "measured_open_after_gate": measured is not None,
        "continuous_independent_support": trailing_support_s >= minimum_support,
        "no_support_loss_after_first_independent_support": bool(
            not initially_supported.any() or initially_supported[np.argmax(initially_supported):].all()),
        "no_finger_recontact_after_independent_support": not len(recontact),
        "last_15_frames_independent_support": drift is not None and trailing_start <= end_i - 14,
        "last_15_frames_drift_under_limit": drift is not None and drift <= drift_limit,
        "no_ground_or_furniture_contact": not np.any(ground > 1e-6) and not np.any(furniture[lift_i:end_i + 1]),
    }
    return {
        "source_events": event,
        "carry_frames": [int(frames[carry.start]), int(frames[carry.stop - 1])],
        "carry_contact_loss_frames": frames[carry][bilateral[carry] == 0].tolist(),
        "carry_slip_mm_max": float(shift.max()),
        "carry_turn_deg_max": float(turn.max()),
        "early_actual_open_frame": None if early is None else int(frames[early]),
        "measured_open_frame": None if measured is None else int(frames[measured]),
        "prior_load": [{"frame": int(frames[i]), "impulse_ns": float(vertical[i]),
                        "threshold_ns": float(threshold[i])} for i in prior],
        "trailing_independent_support_s": trailing_support_s,
        "independent_support_from_open_s": support_from_open_s,
        "finger_recontact_frames_after_independent_support": recontact.tolist(),
        "max_displacement_since_open_mm": open_to_end_displacement,
        "last_15_frame_drift_mm": drift,
        "checks": checks,
        "all_audited_checks_pass": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description="Audit carry, release, and settling directly from R1 trace data.")
    parser.add_argument("run", type=Path, nargs="+", help="Directories with exec_trace.npz, physics_frames.csv and verdict.json")
    parser.add_argument("--events", type=Path, help="JSON event list for a single run, with lift/request/authorized/end source frames")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--drift-mm", type=float, default=3.)
    parser.add_argument("--support-s", type=float, default=.5)
    parser.add_argument("--carry-slip-mm", type=float, default=10.)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.events is not None and len(args.run) != 1:
        parser.error("--events applies to one trace")
    limits = [args.drift_mm, args.support_s, args.carry_slip_mm]
    if not np.isfinite(limits).all() or min(limits) < 0 or args.support_s == 0:
        parser.error("Set finite nonnegative drift/slip limits and a positive support duration")
    results = []
    for run in args.run:
        verdict = json.loads((run / "verdict.json").read_text())
        trace, furniture = read_trace(run, verdict["trace_sha256"])
        frames = trace["source_frames"]
        if args.events is None:
            events = [{
                "lift": verdict["first_lift_with_bilateral_source_frame"],
                "request": verdict["reference_release_source_frame"],
                "authorized": verdict["release_authorized_source_frame"],
                "end": int(frames[-1]),
            }]
        else:
            events = json.loads(args.events.read_text())
        if not isinstance(events, list) or not events:
            raise ValueError("Expected a nonempty list of events")
        for previous, following in zip(events, events[1:]):
            if following["lift"] is not None and previous["end"] >= following["lift"]:
                raise ValueError("Events must be ordered and have non-overlapping carry/settling intervals")
        missing_lift = any(event["lift"] is None for event in events)
        audited = [audit_event(trace, furniture, event, args.drift_mm, args.support_s, args.carry_slip_mm)
                   for event in events if event["lift"] is not None]
        results.append({
            "run": str(run.resolve()), "trace_sha256": verdict["trace_sha256"],
            "source_frames": [int(frames[0]), int(frames[-1])],
            "physical_time_s": float(trace["physics_substeps"].sum() * trace["dt"]),
            "reported_training_data_pass": verdict["training_data_pass"],
            "audit_status": "missing_lift" if missing_lift else ("pass" if all(r["all_audited_checks_pass"] for r in audited) else "fail"),
            "events": audited,
        })
    report = {"scope": "Trace-only checks. Source scene, grasp geometry, controller limits and rendered media need their own evidence.",
              "drift_limit_mm": args.drift_mm, "minimum_independent_support_s": args.support_s,
              "carry_slip_limit_mm": args.carry_slip_mm, "runs": results}
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(f'{Path(row["run"]).parent.name}: {row["audit_status"]}' for row in results))
    if args.require_pass and any(row["audit_status"] != "pass" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
