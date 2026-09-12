import numpy as np
import pytest

from stage1_recipe import (
    STAGE1_CLOSE_CONTROLS,
    STAGE1_SETTLE_CONTROLS,
    build_physics_config,
    build_scene_digest,
    build_stage1_grasp_recipe,
    stage1_close_q_sequence,
)
from stage2_reconstruct_mixin import (
    CARRY_VERIFY_CONTROLS,
    F50_CLOSE_CONTROLS,
    F50_HOLD_CONTROLS,
    OBJECTIVE_WEIGHT_NAMES,
    STAGE1_TARGET_ISOLATED_CONTACT_MASKS,
    Stage2Mixin,
    TARGET_ONLY_CONTACT_MASKS,
    build_stage2_timeline,
    contact_masks_allow_pair,
)


def _simple_reconstruction():
    frames = list(range(38, 63))
    return {
        "frames": frames,
        "pos": {frame: np.array([0.002 * frame, 0.0, 0.10]) for frame in frames},
        "R": {frame: np.eye(3) for frame in frames},
        "gripper_qpos": {frame: 0.05 if frame < 50 else 0.01 for frame in frames},
        "alpha_pose": {frame: 1.0 if frame >= 50 else 0.0 for frame in frames},
        "safety_offset": {frame: np.zeros(3) for frame in frames},
        "grasp_anchor": {
            "source_frame": 50,
            "pos": np.array([0.10, 0.0, 0.10]),
            "R": np.eye(3),
            "gripper_qpos": 0.01,
            "q_start": 0.05,
        },
    }


# --------------------------------------------------------------------------- #
# Shared Stage1 recipe identity
# --------------------------------------------------------------------------- #


def test_f50_has_twenty_stage1_close_and_ten_settle_controls():
    bundle = build_stage2_timeline(_simple_reconstruction(), F50_IDX=50)
    at_f50 = bundle["source_frame"] == 50.0
    phases = list(bundle["phase"][at_f50])

    assert int(at_f50.sum()) == 30
    assert phases[:20] == ["STAGE1_CLOSE"] * 20
    assert phases[20:] == ["STAGE1_SETTLE"] * 10
    assert int(np.sum(bundle["phase"] == "STAGE1_CLOSE")) == F50_CLOSE_CONTROLS
    assert int(np.sum(bundle["phase"] == "STAGE1_SETTLE")) == F50_HOLD_CONTROLS
    assert F50_CLOSE_CONTROLS == STAGE1_CLOSE_CONTROLS == 20
    assert F50_HOLD_CONTROLS == STAGE1_SETTLE_CONTROLS == 10


def test_f50_close_equals_the_shared_stage1_recipe_verbatim():
    recon = _simple_reconstruction()
    recipe = build_stage1_grasp_recipe(
        target_name="glb_4",
        q_open=0.05,
        q_grasp=0.01,
        physics_config=build_physics_config(),
        scene_digest={"target_object": "glb_4"},
    )
    bundle = build_stage2_timeline(recon, F50_IDX=50, recipe=recipe)
    meta = bundle["timeline_meta"]

    close_q = bundle["gripper_q"][meta["close_start"] : meta["close_end"] + 1, 0]
    # The recipe is stored as JSON with 9-decimal rounding, so compare at 1e-9.
    np.testing.assert_allclose(close_q, np.asarray(recipe["close_ctrl_q"]), atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        close_q, stage1_close_q_sequence(0.05, 0.01, STAGE1_CLOSE_CONTROLS), atol=0.0, rtol=0
    )
    np.testing.assert_allclose(bundle["gripper_q"][meta["settle_start"] :, 0], 0.01)
    assert meta["close_profile"] == "cubic_smoothstep"
    assert meta["close_controls"] == STAGE1_CLOSE_CONTROLS
    assert meta["settle_controls"] == STAGE1_SETTLE_CONTROLS
    assert meta["recipe_scene_digest_hash"] == recipe["scene_digest_hash"]


def test_stage2_rejects_a_recipe_whose_close_profile_is_not_reproducible():
    recon = _simple_reconstruction()
    recipe = build_stage1_grasp_recipe(
        target_name="glb_4",
        q_open=0.05,
        q_grasp=0.01,
        physics_config=build_physics_config(),
        scene_digest={"target_object": "glb_4"},
    )
    recipe["close_ctrl_q"] = list(reversed(recipe["close_ctrl_q"]))

    with pytest.raises(ValueError, match="does not match the Stage1 recipe"):
        build_stage2_timeline(recon, F50_IDX=50, recipe=recipe)


def test_f50_is_append_only_then_immediately_moves_to_f51():
    bundle = build_stage2_timeline(_simple_reconstruction(), F50_IDX=50)
    meta = bundle["timeline_meta"]

    assert np.all(np.diff(bundle["source_frame"]) >= 0.0)
    assert bundle["source_frame"][meta["pre_f50_end"]] < 50.0
    assert bundle["source_frame"][meta["verify_start"]] == 50.2
    assert bundle["source_frame"][meta["verify_end"]] == 51.0
    assert int(np.sum(bundle["phase"] == "CARRY_VERIFY")) == CARRY_VERIFY_CONTROLS
    assert np.linalg.norm(
        bundle["base_pos"][meta["verify_start"]]
        - bundle["base_pos"][meta["hold_end"]]
    ) > 0.0


def test_gripper_stays_at_stage1_hold_after_fast_close():
    recon = _simple_reconstruction()
    bundle = build_stage2_timeline(recon, F50_IDX=50)
    meta = bundle["timeline_meta"]

    np.testing.assert_allclose(bundle["gripper_q"][meta["close_end"]], [0.01, 0.01])
    np.testing.assert_allclose(bundle["gripper_q"][meta["settle_start"] :], 0.01)


def test_reconstruction_pins_f50_and_keeps_fixed_correction_after_grasp():
    mixin = Stage2Mixin()
    mano_pos = np.stack([np.array([0.01 * frame, 0.0, 0.10]) for frame in range(80)])
    mano_R = np.repeat(np.eye(3)[None, :, :], 80, axis=0)
    mixin._mano_gripper_traj = {
        "left": {"pos": mano_pos, "R": mano_R, "j1": np.zeros(80), "j2": np.zeros(80)}
    }
    mixin._mano_neutral_offset = {"left": np.zeros(3)}
    grasp = {
        "pos": np.array([0.55, 0.02, 0.12]),
        "R": np.eye(3),
        "gripper_qpos": 0.01,
        "enclose_z": 0.12,
    }

    recon = mixin.reconstruct_trajectory(
        grasp, "left", F50_IDX=50, f_start=38, f_end=62
    )
    delta = grasp["pos"] - mano_pos[50]

    np.testing.assert_allclose(recon["pos"][50], grasp["pos"])
    np.testing.assert_allclose(recon["R"][50], grasp["R"])
    np.testing.assert_allclose(recon["pos"][62], mano_pos[62] + delta)
    assert recon["alpha_pose"][62] == 1.0


def test_reconstruction_replays_f0_f38_and_uses_different_f50_objectives():
    mixin = Stage2Mixin()
    mano_pos = np.stack([np.array([0.002 * frame, 0.0, 0.10]) for frame in range(80)])
    mano_R = np.repeat(np.eye(3)[None, :, :], 80, axis=0)
    mixin._mano_gripper_traj = {
        "left": {"pos": mano_pos, "R": mano_R, "j1": np.zeros(80), "j2": np.zeros(80)}
    }
    mixin._mano_neutral_offset = {"left": np.zeros(3)}
    grasp = {
        "pos": np.array([0.10, 0.0, 0.10]),
        "R": np.eye(3),
        "gripper_qpos": 0.01,
        "enclose_z": 0.10,
    }

    recon = mixin.reconstruct_trajectory(
        grasp, "left", F50_IDX=50, f_start=38, f_end=62, approach_min_z=0.0
    )
    bundle = build_stage2_timeline(recon)
    meta = bundle["timeline_meta"]

    np.testing.assert_allclose(bundle["source_frame"][:39], np.arange(39))
    assert np.all(bundle["phase"][:39] == "MANO_FOLLOW")
    np.testing.assert_allclose(bundle["base_pos"][:39], mano_pos[:39])
    assert tuple(bundle["objective_weight_names"]) == OBJECTIVE_WEIGHT_NAMES
    avoid_target = OBJECTIVE_WEIGHT_NAMES.index("avoid_target_contact")
    bilateral = OBJECTIVE_WEIGHT_NAMES.index("maintain_bilateral_contact")
    assert bundle["objective_weights"][meta["pre_f50_end"], avoid_target] > 0.0
    assert bundle["objective_weights"][meta["pre_f50_end"], bilateral] == 0.0
    assert bundle["objective_weights"][meta["close_start"], avoid_target] == 0.0
    assert bundle["objective_weights"][meta["close_start"], bilateral] > 0.0
    assert bundle["objective_weights"][meta["settle_start"], bilateral] > 0.0


def test_approach_follows_mano_with_smooth_relative_correction_and_no_z_lift():
    mixin = Stage2Mixin()
    frames = np.arange(80, dtype=np.float64)
    mano_pos = np.stack([0.002 * frames, 0.001 * np.sin(frames), 0.02 + 0.0002 * frames], axis=1)
    mano_R = np.repeat(np.eye(3)[None, :, :], 80, axis=0)
    mixin._mano_gripper_traj = {
        "left": {"pos": mano_pos, "R": mano_R, "j1": np.zeros(80), "j2": np.zeros(80)}
    }
    mixin._mano_neutral_offset = {"left": np.zeros(3)}
    grasp = {
        "pos": mano_pos[50] + np.array([0.01, -0.004, 0.003]),
        "R": np.eye(3),
        "gripper_qpos": 0.01,
    }

    recon = mixin.reconstruct_trajectory(grasp, "left", approach_min_z=0.0)
    delta = grasp["pos"] - mano_pos[50]
    for frame in range(38, 50):
        progress = (frame - 38) / 12.0
        alpha = progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)
        np.testing.assert_allclose(recon["pos"][frame], mano_pos[frame] + alpha * delta)
        np.testing.assert_allclose(recon["safety_offset"][frame], 0.0)


def test_approach_only_clamps_z_when_corrected_mano_is_below_floor():
    mixin = Stage2Mixin()
    mano_pos = np.repeat(np.array([[0.0, 0.0, -0.01]]), 80, axis=0)
    mano_R = np.repeat(np.eye(3)[None, :, :], 80, axis=0)
    mixin._mano_gripper_traj = {
        "left": {"pos": mano_pos, "R": mano_R, "j1": np.zeros(80), "j2": np.zeros(80)}
    }
    mixin._mano_neutral_offset = {"left": np.zeros(3)}
    grasp = {"pos": np.array([0.0, 0.0, 0.02]), "R": np.eye(3), "gripper_qpos": 0.01}

    recon = mixin.reconstruct_trajectory(grasp, "left", approach_min_z=0.0)
    for frame in range(38, 50):
        assert recon["pos"][frame][2] >= 0.0
        assert recon["safety_offset"][frame][0] == 0.0
        assert recon["safety_offset"][frame][1] == 0.0


def test_bundle_hash_is_deterministic_and_changes_with_timing():
    recon = _simple_reconstruction()
    bundle_a = build_stage2_timeline(recon, carry_time_scale=2.0)
    bundle_b = build_stage2_timeline(recon, carry_time_scale=2.0)
    bundle_c = build_stage2_timeline(recon, carry_time_scale=3.0)

    assert bundle_a["bundle_hash"] == bundle_b["bundle_hash"]
    assert bundle_a["bundle_hash"] != bundle_c["bundle_hash"]


# --------------------------------------------------------------------------- #
# stage1_target_isolated collision matrix (plan §3.4)
# --------------------------------------------------------------------------- #


def test_stage1_target_isolated_collision_masks_match_the_plan():
    masks = STAGE1_TARGET_ISOLATED_CONTACT_MASKS
    assert masks is TARGET_ONLY_CONTACT_MASKS

    assert contact_masks_allow_pair(masks["gripper"], masks["target"])
    # gripper-ground is real Stage1 physics in this round.
    assert contact_masks_allow_pair(masks["gripper"], masks["default"])
    assert contact_masks_allow_pair(masks["target"], masks["default"])
    # non-target objects keep their own dynamics.
    assert contact_masks_allow_pair(masks["non_target"], masks["default"])
    assert contact_masks_allow_pair(masks["non_target"], masks["non_target"])
    # only these two pairs are isolated this round.
    assert not contact_masks_allow_pair(masks["gripper"], masks["non_target"])
    assert not contact_masks_allow_pair(masks["target"], masks["non_target"])


# --------------------------------------------------------------------------- #
# stage1_reuse_report / scale + position consistency
# --------------------------------------------------------------------------- #


class _Gravity:
    def get_gravity(self):
        return np.array([0.0, 0.0, -9.81])


class _Pose:
    def __init__(self, p, q):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


class _Component:
    def __init__(self, mass):
        self.mass = mass


class _Actor:
    def __init__(self, name, pos, scale=1.0):
        self._name = name
        self._pose = _Pose(pos, [1.0, 0.0, 0.0, 0.0])
        self._scale = scale
        self._components = [_Component(0.5)]

    def get_name(self):
        return self._name

    def get_pose(self):
        return self._pose

    def get_components(self):
        return self._components


def _stub_stage2(scale=1.0, pos=(0.10, 0.0, 0.05)):
    actor = _Actor("glb_4", pos, scale=scale)
    mixin = Stage2Mixin()
    mixin.obj_actors = [actor]
    mixin.obj_info = {
        "glb_4": {
            "bbox_size": [0.02 * scale, 0.03 * scale, 0.04 * scale],
            "bbox_min": [0.09, -0.015 * scale, -0.02 * scale],
            "bbox_max": [0.11, 0.015 * scale, 0.02 * scale],
            "body_type": "dynamic",
        }
    }
    mixin._obj_initial_poses = {"glb_4": actor.get_pose()}
    mixin._ground_z = 0.0
    mixin.scene = _Gravity()
    return mixin, actor


def _recon_with(recipe, q_grasp=0.01, q_start=0.05):
    recon = _simple_reconstruction()
    recon["stage1_recipe"] = recipe
    recon["grasp_anchor"] = dict(recon["grasp_anchor"])
    recon["grasp_anchor"]["gripper_qpos"] = q_grasp
    recon["grasp_anchor"]["q_start"] = q_start
    return recon


def _recipe_for_current_scene(mixin, target_name="glb_4"):
    digest = mixin._current_scene_digest(target_name)
    return build_stage1_grasp_recipe(
        target_name=target_name,
        q_open=0.05,
        q_grasp=0.01,
        physics_config=build_physics_config(),
        scene_digest=digest,
    )


def test_reuse_report_passes_when_scale_and_position_are_identical():
    mixin, _actor = _stub_stage2()
    recipe = _recipe_for_current_scene(mixin)
    report = mixin._stage1_reuse_report(_recon_with(recipe), "glb_4")

    assert report["passed"] is True
    assert report["mismatches"] == []


def test_reuse_report_rejects_a_scaled_object():
    mixin, _actor = _stub_stage2(scale=1.0)
    recipe = _recipe_for_current_scene(mixin)

    scaled, _ = _stub_stage2(scale=0.6)
    report = scaled._stage1_reuse_report(_recon_with(recipe), "glb_4")

    assert report["passed"] is False
    assert report["failure"] == "STAGE1_REUSE_MISMATCH"
    assert "bbox_size" in report["mismatches"]


def test_reuse_report_rejects_a_moved_object():
    mixin, _actor = _stub_stage2(pos=(0.10, 0.0, 0.05))
    recipe = _recipe_for_current_scene(mixin)

    # The digest pins the shared reset pose both stages restore before rollout.
    mixin._obj_initial_poses["glb_4"] = _Pose((0.12, 0.0, 0.05), [1.0, 0.0, 0.0, 0.0])
    report = mixin._stage1_reuse_report(_recon_with(recipe), "glb_4")

    assert report["passed"] is False
    assert report["failure"] == "STAGE1_REUSE_MISMATCH"
    assert "initial_pose_pos" in report["mismatches"]


def test_reuse_report_rejects_a_stale_artifact_without_a_recipe():
    mixin, _actor = _stub_stage2()
    report = mixin._stage1_reuse_report(_simple_reconstruction(), "glb_4")

    assert report["passed"] is False
    assert report["failure"] == "STALE_STAGE1_ARTIFACT"


def test_reuse_report_rejects_an_anchor_that_disagrees_with_the_recipe():
    mixin, _actor = _stub_stage2()
    recipe = _recipe_for_current_scene(mixin)
    report = mixin._stage1_reuse_report(_recon_with(recipe, q_grasp=0.02), "glb_4")

    assert report["passed"] is False
    assert report["failure"] == "STAGE1_REUSE_MISMATCH"
    assert "q_grasp" in report["mismatches"]


# --------------------------------------------------------------------------- #
# Save contract
# --------------------------------------------------------------------------- #


def test_only_passed_bundle_can_be_saved(tmp_path):
    bundle = build_stage2_timeline(_simple_reconstruction())
    output = tmp_path / "stage2_bundle.npz"

    assert not Stage2Mixin.save_stage2_bundle(
        output, {"bundle": bundle, "gate_result": {"passed": False, "failure": "CARRY"}}
    )
    assert not output.exists()

    result = {
        "bundle": bundle,
        "trace": {},
        "gate_result": {
            "passed": True,
            "failure": None,
            "metrics": {"numpy_bool": np.bool_(True), "numpy_float": np.float64(1.0)},
        },
    }
    assert Stage2Mixin.save_stage2_bundle(output, result)
    saved = np.load(output, allow_pickle=False)
    np.testing.assert_allclose(saved["source_frame"], bundle["source_frame"])
    assert str(saved["timeline_meta_json"]) != ""
