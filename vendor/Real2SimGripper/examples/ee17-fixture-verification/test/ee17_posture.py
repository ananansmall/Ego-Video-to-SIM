#!/usr/bin/env python3
"""Run matched ee17 A/B with the shared measured-contact backend."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TARGETS = ROOT / "dataset/ee17_mocap/robot/targets.npz"


def check():
    from large_range_track import (lead_base_reference, torso_pitch_coordinates,
                                   torso_link_pitch, stable_grasp_width, supported_rest, ring_approach)

    # Asymmetric bounds also represent bounds narrowed by the previous frame.
    lo = np.radians([-5., 15., 22.])
    hi = np.radians([10., 25., 29.])
    pitch = np.radians([8., 30., 8.])
    for unit in np.random.default_rng(0).uniform(size=(128, 3)):
        q = torso_pitch_coordinates(unit, lo, hi, pitch)
        assert np.all(q >= lo - 1e-12) and np.all(q <= hi + 1e-12)
        assert np.all(np.abs(torso_link_pitch(q)) <= pitch + 1e-12)
        np.testing.assert_allclose(torso_pitch_coordinates(q, lo, hi, pitch, encode=True), unit)
    body = np.column_stack((np.linspace(0.0, 0.03, 16), np.zeros(16)))
    reach = np.tile([0.15, -0.08], (16, 1))
    states = np.r_[np.zeros(8, int), np.full(8, 2)]
    lead = lead_base_reference(body, reach, states, lead_frames=3)
    np.testing.assert_allclose(lead[[5, 8]], reach[[5, 8]])
    np.testing.assert_allclose(lead[[0, 2, 11, 15]], body[[0, 2, 11, 15]])
    assert np.all(np.diff(lead[2:6, 0]) > 0)
    gaps = np.array([0.0824, 0.0826, 0.0825])
    fingers = np.full((3, 2), 0.012)
    assert np.isclose(stable_grasp_width(gaps, fingers, 0.0824), 0.0825)
    assert stable_grasp_width(gaps, fingers * [1, 0], 0.0824) is None
    assert stable_grasp_width(gaps, fingers * [1, np.nan], 0.0824) is None
    assert stable_grasp_width([0.042, *gaps[1:]], fingers, 0.0824) is None
    rest = np.zeros((25, 3))
    live = np.ones(25, bool)
    contacts = np.zeros((25, 4))
    contacts[5:, 2] = 0.005
    assert supported_rest(rest, live, contacts, 5)
    assert not supported_rest(rest, live & False, contacts, 5)
    contacts[11, 3] = 0.005
    assert not supported_rest(rest, live, contacts, 5)
    pad = np.tile([0.07, 0., 0.78], (50, 1))
    route = ring_approach(pad, 5, 20, 36, [0, 0], 0.83, 0.031,
                          [0.064, 0., 0.05], carry_half_z=0.075)
    assert route[10, 2] >= 0.94 - 1e-8 and np.allclose(route[36], [0., 0., 0.877])
    assert np.all(np.diff(route[28:37, 2]) <= 1e-8)
    print("PASS: all torso pitch prefixes, base lead, width-aware bilateral handoff, dynamic rest and ring approach")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("before", "after", "check"))
    ap.add_argument("--targets", type=Path, default=TARGETS)
    ap.add_argument("--object", type=Path)
    ap.add_argument("--scene-cloud", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--pitch-deg", type=float, nargs="+", default=[25., 25., 25.])
    ap.add_argument("--pitch-guard-deg", type=float, default=3.0)
    ap.add_argument("--ring-place", action="store_true", help="Enable annular placement path in after")
    ap.add_argument("--physics-only", action="store_true")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--lead-s", type=float, default=0.6)
    ap.add_argument("--joint-reserve-deg", type=float, default=5.0)
    ap.add_argument("--require-release", action="store_true")
    ap.add_argument("--ik-only", action="store_true")
    a = ap.parse_args()
    if a.mode == "check":
        check()
        return
    if a.object is None:
        ap.error("--object is required for A/B. Verify source episode and geometry provenance.")
    if len(a.pitch_deg) not in (1, 3):
        ap.error("--pitch-deg takes one cap or three torso link caps")
    for path in (a.targets, a.object, a.scene_cloud):
        if path is not None and not path.is_file():
            ap.error(f"missing input: {path}")
    with np.load(a.targets) as targets:
        states = targets["grip_states"]
        release = np.flatnonzero(states >= 5)
        if a.require_release and (not release.size or len(states) - release[0] < 35):
            ap.error("place validation needs a release label and at least 35 frames from release")
        if a.require_release and not np.any(states[:release[0]:5] == 3):
            ap.error("place validation also needs HOLD frames for contact and drift measurement")
        print(f"Input frames: {len(states)}. First release: {release[0] if release.size else 'absent'}")

    import cb_large_range_sim as cb

    name = a.name or f"ee17_posture_{a.mode}"
    if not name.startswith("ee17_posture_") or not name.replace("_", "").isalnum():
        ap.error("output name must start with ee17_posture_ and contain only letters, digits or underscores")
    if a.mode == "before":
        control = dict(w_ori=0.0, arm_beam=4)
    else:
        # The hard bounds already protect posture. Keep the remaining posture
        # preference weak enough for the gripper to reach its target.
        control = dict(w_ori=10.0, posture_weight=0.25,
                       torso_pitch_deg=a.pitch_deg, base_lead_s=a.lead_s,
                       torso_pitch_guard_deg=a.pitch_guard_deg, ring_place=a.ring_place,
                       joint_reserve_deg=a.joint_reserve_deg, joint_step_max=4.0)
    cb.main(targets=str(a.targets), object_npz=str(a.object), name=name,
            scene_cloud=str(a.scene_cloud) if a.scene_cloud else None,
            object_physics=True, torso_wbc=True, ik_only=a.ik_only, stride=a.stride,
            phys_only=a.physics_only, clock_mode="source", handoff_freeze_frames=8,
            handoff_at_release=True,
            finger_force_limit=50.0, base_ref_source="body", torso_gain=6.0,
            wrist_gain=3.0, depenetration_vel=0.05, base_interp=True,
            legacy_physics=False, **control)


if __name__ == "__main__":
    main()
