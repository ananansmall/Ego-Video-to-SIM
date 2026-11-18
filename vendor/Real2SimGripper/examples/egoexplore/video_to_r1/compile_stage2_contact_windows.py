"""Compile approved object-relative targets on an exact source frame timeline."""
import argparse
import json
from pathlib import Path

import numpy as np

from merge_agile_contact_windows import digest, merge


def reviewed(entry):
    return entry.get("status") == "accepted" and entry.get("consumable") is True


def contact_patch(baseline, entry):
    frames = baseline["src_frames"]
    ids = np.asarray(entry["source_frames"])
    anchors = np.asarray(entry["contact_evidence_frames"])
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) < 3 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Need exact ordered source frames and room around the contact anchors")
    selected = (frames >= ids[0]) & (frames <= ids[-1])
    if not np.array_equal(ids, frames[selected]):
        raise ValueError("The window frame list differs from the frozen source timeline")
    if (anchors.ndim != 1 or anchors.dtype.kind not in "iu" or not len(anchors)
            or not np.isin(anchors, ids[1:-1]).all()):
        raise ValueError("Contact evidence needs exact source frames inside the window")
    for field in ("hand_observed", "object_observed"):
        if field in baseline and not baseline[field][np.searchsorted(frames, anchors)].all():
            raise ValueError(f"Contact anchors include missing observations ({field})")
    joint = np.asarray(entry["grip"], float)
    if (joint.shape not in ((2,), (len(ids), 2)) or not np.isfinite(joint).all()
            or np.any((joint < -1e-8) | (joint > .05 + 1e-8))):
        raise ValueError("R1 finger targets must be in [0, 0.05] metres")
    local = np.asarray(entry["T_object_pad"], float)
    if local.shape not in ((4, 4), (len(ids), 4, 4)):
        raise ValueError("Expected one object-frame grasp or one grasp for each window row")
    start, end = anchors.min(), anchors.max()
    ramp = np.minimum((ids - ids[0]) / (start - ids[0]), (ids[-1] - ids) / (ids[-1] - end))
    weight = .5 - .5 * np.cos(np.pi * np.clip(ramp, 0., 1.))
    world_object = baseline["T_world_object"][selected]
    spec = dict(window_id=entry["window_id"], object_id=str(entry["object_id"]),
                first_frame=int(ids[0]), last_frame=int(ids[-1]), contact_evidence_frames=anchors)
    return spec, dict(source_frames=ids, object_id=np.array(spec["object_id"]), weight=weight,
                      grip=np.broadcast_to(np.clip(joint, 0., .05), (len(ids), 2)),
                      T_world_object=world_object, T_world_pad=world_object @ local)


def apply(baseline, windows):
    frames = np.asarray(baseline["src_frames"])
    if frames.ndim != 1 or frames.dtype.kind not in "iu" or not len(frames) or np.any(frames[1:] <= frames[:-1]):
        raise ValueError("The reference must have strictly increasing integer source IDs")
    if not windows:
        raise ValueError("There are no contact windows to review")
    if len({w["window_id"] for w in windows}) != len(windows):
        raise ValueError("Duplicate contact-window IDs")
    edits = [w for w in windows if w.get("event_type") != "carry_continuation"]
    if not edits:
        return None, dict(status="no_contact_edit", windows_pending=[])
    approved = [w for w in edits if reviewed(w)]
    pending = [w["window_id"] for w in edits if not reviewed(w)]
    if not approved:
        return None, dict(status="contact_review_required", windows_pending=pending)
    patches = [contact_patch(baseline, entry) for entry in approved]
    result, changed = merge(baseline, patches, allow_sampled_source=True, freeze_object=True)
    return result, dict(status="contact_review_partial" if pending else "contact_targets_compiled",
                        windows=changed, windows_pending=pending, physical_replay_verified=False,
                        source_contiguous=bool(np.all(np.diff(frames) == 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    reference_sha = digest(args.baseline)
    edits = [entry for entry in plan["windows"] if entry.get("event_type") != "carry_continuation"]
    accepted = any(reviewed(entry) for entry in edits)
    if accepted and plan.get("baseline_sha256") != reference_sha:
        raise ValueError("Approved contact plan does not reference this exact frozen baseline")
    with np.load(args.baseline, allow_pickle=False) as frozen:
        baseline = dict(frozen)
    if accepted:
        if "object_ids" not in baseline:
            parser.error("The accepted plan needs a Stage 0 baseline exported with source-frame object IDs")
        for original, current in (("T_world_pad_retarget", "T_world_pad"), ("grip_retarget", "grip")):
            if original not in baseline:
                baseline[original] = baseline[current]
    output, report = apply(baseline, plan["windows"])
    args.out.mkdir(parents=True, exist_ok=False)
    report.update(baseline_sha256=reference_sha, plan_sha256=digest(args.plan))
    if output is not None:
        target = args.out / ("review_targets.npz" if report["windows_pending"] else "contact_targets.npz")
        np.savez_compressed(target, **output)
        report.update(target_file=target.name, target_sha256=digest(target))
    (args.out / "stage2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
