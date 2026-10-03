"""Pure timeline contract for the explicit local-infeasibility fallback.

This does not claim that the fallback succeeds in physics.  It proves only
that the structural change requested by the plan is explicit, deterministic,
and does not silently alter the historical MANO mode.
"""

import numpy as np

from stage2_reconstruct_mixin import build_stage2_timeline


def _reconstruction():
    frames = list(range(38, 63))
    return {
        "frames": frames,
        "pos": {f: np.array([0.002 * f, 0.001 * (f - 50), 0.10 + 0.0005 * (f - 50)])
                for f in frames},
        "R": {f: np.eye(3) for f in frames},
        "gripper_qpos": {f: 0.05 if f < 50 else 0.01 for f in frames},
        "alpha_pose": {f: 1.0 for f in frames},
        "safety_offset": {f: np.zeros(3) for f in frames},
        "grasp_anchor": {
            "source_frame": 50,
            "pos": np.array([0.10, 0.0, 0.10]),
            "R": np.eye(3),
            "gripper_qpos": 0.01,
            "q_start": 0.05,
        },
    }


def test_historical_mano_mode_remains_the_default(monkeypatch):
    monkeypatch.delenv("_S2_CARRY_MODE", raising=False)
    bundle = build_stage2_timeline(_reconstruction(), carry_time_scale=2.0)
    meta = bundle["timeline_meta"]
    assert meta["carry_mode"] == "mano"
    assert meta["carry_structural_deviation"] is False


def test_vertical_planar_is_explicit_and_keeps_verified_orientation(monkeypatch):
    monkeypatch.setenv("_S2_CARRY_MODE", "vertical_planar")
    monkeypatch.setenv("_S2_VERTICAL_LIFT_CONTROLS", "4")
    monkeypatch.setenv("_S2_VERTICAL_LIFT_M", "0.020")
    bundle = build_stage2_timeline(_reconstruction(), carry_time_scale=2.0)
    meta = bundle["timeline_meta"]
    phase = np.asarray(bundle["phase"])
    carry = np.flatnonzero(phase == "CARRY")

    assert meta["carry_mode"] == "vertical_planar"
    assert meta["carry_structural_deviation"] is True
    assert meta["carry_orientation_locked"] is True
    assert meta["carry_vertical_lift_controls"] == 4
    assert meta["carry_vertical_lift_m"] == 0.020
    assert carry.size == 4 + 22
    assert np.all(np.diff(bundle["source_frame"]) >= 0.0)
    np.testing.assert_allclose(
        bundle["base_R"][carry],
        np.repeat(np.eye(3)[None, :, :], carry.size, axis=0),
        atol=0.0,
        rtol=0.0,
    )

    lift = carry[:4]
    np.testing.assert_allclose(bundle["base_pos"][lift[-1], 2], 0.12, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        bundle["base_pos"][lift, :2],
        np.repeat([[0.10, 0.0]], lift.size, axis=0),
        atol=1e-12,
        rtol=0.0,
    )
    # The first planar command starts from the lifted anchor and is eased in;
    # it must not reproduce the old one-tick F50 -> F51 jump.
    planar = carry[4:]
    assert np.isclose(bundle["base_pos"][planar[0], 2], 0.12)
    assert np.linalg.norm(bundle["base_pos"][planar[0], :2] - [0.10, 0.0]) < 1e-3
    assert np.isclose(bundle["base_pos"][planar[-1], 2], 0.12)


def test_vertical_planar_rejects_duplicate_transition_controls(monkeypatch):
    monkeypatch.setenv("_S2_CARRY_MODE", "vertical_planar")
    try:
        build_stage2_timeline(_reconstruction(), carry_verify_controls=1)
    except ValueError as exc:
        assert "duplicate moving prefix" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("vertical_planar must not silently duplicate verify controls")
