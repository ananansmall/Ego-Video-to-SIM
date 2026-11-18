#!/usr/bin/env python3
"""Exclude R1 contact targets below a conservative pad-height bound."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def pad_height_lower_bound(model, torso_caps_deg, pad_offset_m, base_height_m=0.0, side="right"):
    """Return a lower bound in metres for a level, vertically fixed base.

    Caps are absolute torso link pitches, not relative joint angles. This
    checks the R1 chain conventions used to derive the bound. A target above
    the bound still needs a whole-body IK and physics check.
    """
    caps = tuple(torso_caps_deg)
    if len(caps) != 3 or any(not 0 <= value <= 90 for value in caps):
        raise ValueError("Expected three absolute torso pitch limits in [0, 90] degrees")
    if not math.isfinite(base_height_m) or not math.isfinite(pad_offset_m) or pad_offset_m < 0:
        raise ValueError("Base height and nonnegative pad offset must be finite")
    if side not in ("left", "right"):
        raise ValueError("Expected left or right R1 arm")

    root = ET.parse(model).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    torso = [f"torso_joint{i}" for i in range(1, 5)]
    arm = [f"{side}_arm_joint{i}" for i in range(1, 7)]
    chain = [*torso, f"{side}_arm_base_joint", *arm, f"{side}_gripper_joint"]
    try:
        ordered = [joints[name] for name in chain]
    except KeyError as error:
        raise ValueError(f"Unsupported R1 chain: missing {error.args[0]}") from error
    if any(a.find("child").get("link") != b.find("parent").get("link")
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Unsupported R1 chain topology")

    def origin(name):
        tag = joints[name].find("origin")
        if tuple(map(float, tag.get("rpy", "0 0 0").split())) != (0., 0., 0.):
            raise ValueError(f"Unsupported origin rotation at {name}")
        return tuple(map(float, tag.get("xyz", "0 0 0").split()))

    def axis(name):
        return tuple(map(float, joints[name].find("axis").get("xyz").split()))

    expected_axes = [(0., 1., 0.), (0., 1., 0.), (0., -1., 0.), (0., 0., 1.)]
    if [axis(name) for name in torso] != expected_axes:
        raise ValueError("Unsupported R1 torso axes")
    if any(joints[name].get("type") != "fixed"
           for name in (f"{side}_arm_base_joint", f"{side}_gripper_joint")):
        raise ValueError("Unsupported R1 arm base or gripper joint")
    spans = [origin(name) for name in torso[1:]]
    if any(x or y or z < 0 for x, y, z in spans):
        raise ValueError("Unsupported torso translation offsets")

    # The arm-2 offset lies along arm-1's rotation axis and is invariant to
    # that rotation. Combine all offsets that share the last torso link tilt.
    a = axis(arm[0])
    x, y, z = origin(arm[1])
    if math.hypot(y * a[2] - z * a[1], z * a[0] - x * a[2], x * a[1] - y * a[0]) > 1e-12:
        raise ValueError("The arm-2 offset is not parallel to the arm-1 axis")
    combined = tuple(map(sum, zip(origin(f"{side}_arm_base_joint"), origin(arm[0]), (x, y, z))))
    if combined[2] < 0:
        raise ValueError("Unsupported arm origin height")

    pitch = [math.radians(value) for value in caps]
    torso_height = sum(span[2] * math.cos(angle) for span, angle in zip(spans, pitch))
    shoulder_height = combined[2] * math.cos(pitch[-1]) - math.hypot(*combined[:2]) * math.sin(pitch[-1])
    max_arm_descent = sum(math.hypot(*origin(name)) for name in chain[7:]) + pad_offset_m
    return base_height_m + origin(torso[0])[2] + torso_height + shoulder_height - max_arm_descent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--torso-caps-deg", type=float, nargs=3, required=True, metavar=("A", "B", "C"))
    parser.add_argument("--pad-offset-m", type=float, required=True)
    parser.add_argument("--base-height-m", type=float, default=0.)
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--targets", type=Path, required=True, help="JSON object mapping target IDs to pad heights in metres")
    args = parser.parse_args()

    bound = pad_height_lower_bound(args.urdf, args.torso_caps_deg, args.pad_offset_m,
                                   args.base_height_m, args.side)
    targets = json.loads(args.targets.read_text())
    if not isinstance(targets, dict) or not targets:
        parser.error("Expected a nonempty JSON object of target heights")
    report = {}
    for name, height in targets.items():
        if not isinstance(height, (int, float)) or isinstance(height, bool) or not math.isfinite(height):
            parser.error(f"Target {name} height must be a finite number")
        report[name] = {
            "pad_height_m": height,
            "minimum_vertical_shortfall_m": max(0., bound - height),
            "status": "excluded_by_height" if height < bound - 1e-9 else "requires_ik",
        }
    print(json.dumps({
        "urdf_sha256": hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
        "side": args.side,
        "base_height_m": args.base_height_m,
        "torso_caps_deg": args.torso_caps_deg,
        "pad_offset_m": args.pad_offset_m,
        "conservative_pad_height_lower_bound_m": bound,
        "targets": report,
    }, indent=2))


if __name__ == "__main__":
    main()
