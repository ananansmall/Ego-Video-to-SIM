import numpy as np

from stage2_reconstruct_mixin import build_stage2_timeline
from stage3_optimize_mixin import (
    Stage3Mixin,
    build_stage3_timeline,
    combine_stage2_stage3,
)


def _stage2_bundle():
    frames = list(range(38, 63))
    recon = {
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
    return build_stage2_timeline(recon)


def _mano():
    pos = np.stack(
        [np.array([0.002 * frame, 0.001 * max(0, frame - 62), 0.10]) for frame in range(113)]
    )
    rot = np.repeat(np.eye(3)[None, :, :], len(pos), axis=0)
    return pos, rot


def test_stage3_is_append_only_and_keeps_stage2_endpoint_correction():
    parent = _stage2_bundle()
    mano_pos, mano_R = _mano()
    suffix = build_stage3_timeline(mano_pos, mano_R, parent)
    meta = suffix["timeline_meta"]

    assert suffix["source_frame"][0] > parent["source_frame"][-1]
    assert np.all(np.diff(suffix["source_frame"]) >= 0.0)
    expected_delta = parent["base_pos"][-1] - mano_pos[62]
    carry = slice(meta["carry_start"], meta["carry_end"] + 1)
    for source, command in zip(suffix["source_frame"][carry], suffix["base_pos"][carry]):
        lo = int(np.floor(source))
        hi = min(lo + 1, len(mano_pos) - 1)
        t = source - lo
        mano_interp = (1.0 - t) * mano_pos[lo] + t * mano_pos[hi]
        np.testing.assert_allclose(command, mano_interp + expected_delta)
    assert suffix["mano_fidelity"]["position_shape_max_mm"] < 1e-9
    assert suffix["mano_fidelity"]["rotation_shape_max_deg"] < 1e-9
    np.testing.assert_allclose(suffix["mano_fidelity"]["path_length_ratio"], 1.0)


def test_stage3_holds_grasp_until_explicit_five_control_release():
    parent = _stage2_bundle()
    mano_pos, mano_R = _mano()
    suffix = build_stage3_timeline(mano_pos, mano_R, parent)
    meta = suffix["timeline_meta"]

    held_q = suffix["gripper_q"][: meta["pre_release_end"] + 1]
    np.testing.assert_allclose(held_q, np.full_like(held_q, 0.01))
    assert int(np.sum(suffix["phase"] == "RELEASE")) == 5
    np.testing.assert_allclose(suffix["gripper_q"][meta["release_end"]], [0.05, 0.05])
    np.testing.assert_allclose(suffix["gripper_q"][meta["release_settle_start"] :], 0.05)


def test_stage3_uses_actual_stage2_endpoint_not_nominal_q_grasp():
    parent = _stage2_bundle()
    # Simulate measured close/settle: the actual command is valid, but the
    # recipe metadata still contains an older nominal value.
    parent["timeline_meta"]["q_grasp"] = 0.02
    mano_pos, mano_R = _mano()

    suffix = build_stage3_timeline(mano_pos, mano_R, parent)

    np.testing.assert_allclose(suffix["gripper_q"][0], [0.01, 0.01])
    assert suffix["timeline_meta"]["q_grasp"] == 0.01


def test_stage3_preserves_asymmetric_actual_endpoint_q():
    parent = _stage2_bundle()
    parent["gripper_q"][-1] = [0.01, 0.008]
    mano_pos, mano_R = _mano()

    suffix = build_stage3_timeline(mano_pos, mano_R, parent)

    np.testing.assert_allclose(suffix["gripper_q"][0], [0.01, 0.008])
    assert suffix["timeline_meta"]["q_grasp"] == [0.01, 0.008]
    assert suffix["timeline_meta"]["q_hold_actual"] == [0.01, 0.008]


def test_combined_bundle_preserves_every_parent_array_prefix():
    parent = _stage2_bundle()
    before = {key: np.asarray(value).copy() for key, value in parent.items() if isinstance(value, np.ndarray)}
    mano_pos, mano_R = _mano()
    suffix = build_stage3_timeline(mano_pos, mano_R, parent)
    combined = suffix["combined_bundle"]
    n_parent = len(parent["source_frame"])

    for key in (
        "source_frame",
        "phase",
        "base_pos",
        "base_R",
        "gripper_q",
        "control_dt",
        "w_pose",
        "safety_offset",
        "expected_contact_policy",
        "objective_weights",
    ):
        assert combined[key][:n_parent].dtype == parent[key].dtype
        assert combined[key][:n_parent].tobytes() == parent[key].tobytes()
    for key, value in before.items():
        assert np.asarray(parent[key]).tobytes() == value.tobytes()
    assert combined["parent_bundle_hash"] == parent["bundle_hash"]


def test_executed_combination_can_stop_before_release_without_mutation():
    parent = _stage2_bundle()
    mano_pos, mano_R = _mano()
    suffix = build_stage3_timeline(mano_pos, mano_R, parent)
    count = suffix["timeline_meta"]["pre_release_end"] + 1
    combined = combine_stage2_stage3(parent, suffix, suffix_count=count)

    assert len(combined["source_frame"]) == len(parent["source_frame"]) + count
    assert not np.any(combined["phase"] == "RELEASE")
    assert combined["timeline_meta"]["stage3_executed_control_count"] == count


def test_time_scale_changes_only_carry_sampling_and_hash():
    parent = _stage2_bundle()
    mano_pos, mano_R = _mano()
    fast = build_stage3_timeline(mano_pos, mano_R, parent, carry_time_scale=1.0)
    slow = build_stage3_timeline(mano_pos, mano_R, parent, carry_time_scale=3.0)

    assert fast["bundle_hash"] != slow["bundle_hash"]
    assert slow["timeline_meta"]["carry_end"] > fast["timeline_meta"]["carry_end"]
    np.testing.assert_allclose(
        fast["base_pos"][fast["timeline_meta"]["carry_end"]],
        slow["base_pos"][slow["timeline_meta"]["carry_end"]],
    )


def test_stage3_save_requires_stage3a_pass(tmp_path):
    parent = _stage2_bundle()
    mano_pos, mano_R = _mano()
    suffix = build_stage3_timeline(mano_pos, mano_R, parent)
    output = tmp_path / "combined_bundle.npz"

    failed = {"bundle": suffix["combined_bundle"], "stage3a_gate": {"passed": False}}
    assert not Stage3Mixin.save_stage3_bundle(output, failed)
    assert not output.exists()

    # ★★ 2026-09-28 【契约同步 —— md §07】★★
    #   md 原话 ✓: "`save` 使用**同一总门**, **不能 Stage3A 单独通过就保存为正式成功**" ✗
    #   ⇒ 旧断言 (Stage3A=True 且 Stage3B=False ⇒ 保存) 与 md **直接冲突** ✗
    #     (那不是"改断言让它变绿"✗, 而是**有 md 依据的契约变更** ✓: 总门 = task_pass ✓)
    #   ① 反例: Stage3A 通过但**释放失败**(启用释放) ⇒ 必须拒绝 ✗
    gate_a_only = {
        "bundle": suffix["combined_bundle"],
        "stage3a_gate": {"passed": True, "metrics": {}},
        "stage3b_gate": {"passed": False, "failure": "NO_SUPPORT", "metrics": {}},
        "enable_release": True,
        "trace": {},
        "stage2_replay": {"trace": {}},
    }
    assert not Stage3Mixin.save_stage3_bundle(output, gate_a_only), \
        "Stage3A 单独通过就保存 ⇒ 违反 md §07 ✗"
    assert not output.exists()

    #   ② 正例: **显式 task_pass=True** ⇒ 允许保存 ✓
    passed = dict(gate_a_only, task_pass=True)
    assert Stage3Mixin.save_stage3_bundle(output, passed)
    saved = np.load(output, allow_pickle=False)
    assert str(saved["parent_bundle_hash"]) == parent["bundle_hash"]

    #   ③ 正例: 未启用释放 ⇒ 回退口径 (Stage3A ∧ 无需释放) 通过 ⇒ 保存 ✓
    output.unlink()
    no_release = dict(gate_a_only, enable_release=False)
    no_release["task_pass"] = None
    assert Stage3Mixin.save_stage3_bundle(output, no_release)
    assert output.exists()
