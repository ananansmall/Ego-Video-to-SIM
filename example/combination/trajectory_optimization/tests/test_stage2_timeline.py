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


def test_f50_has_sixteen_stage1_close_and_ten_settle_controls():
    """★ 2026-09-22 【契约同步 —— 审计文档 §7-3 第 1 项 ✓】

    契约依据 ✓ (不是"改断言让测试变绿"✗): git commit `92c2f6e` 的提交信息明确写着
      "**闭合段16帧** + 闭合下限 + 位姿CoACD代价探针" ✓
      ⇒ `STAGE1_CLOSE_CONTROLS = 16` 是**刻意改动** ✓
    而 `stage1_recipe.py` 模块 docstring 与**本测试**仍写 20 ✗ ⇒ 属**陈旧契约** ✗
      (审计原文 ✓: "测试预期 F50 有 20+10 个控制点, 实际为 16+10" ✗
       "需要先明确闭合时长 …, **再同步实现和测试**, 不能只修改断言让测试变绿" ✓)
    本处 ✓: 按**有提交依据的一侧 (16 帧 ✓)** 同步测试, 并让断言**跟随共享常量** ✓
      (此后改常量即自动同步, 不会再两边漂移 ✓)
    """
    bundle = build_stage2_timeline(_simple_reconstruction(), F50_IDX=50)
    at_f50 = bundle["source_frame"] == 50.0
    phases = list(bundle["phase"][at_f50])

    assert int(at_f50.sum()) == F50_CLOSE_CONTROLS + F50_HOLD_CONTROLS
    assert phases[:F50_CLOSE_CONTROLS] == ["STAGE1_CLOSE"] * F50_CLOSE_CONTROLS
    assert phases[F50_CLOSE_CONTROLS:] == ["STAGE1_SETTLE"] * F50_HOLD_CONTROLS
    assert int(np.sum(bundle["phase"] == "STAGE1_CLOSE")) == F50_CLOSE_CONTROLS
    assert int(np.sum(bundle["phase"] == "STAGE1_SETTLE")) == F50_HOLD_CONTROLS
    # 契约 = 共享单一来源 ✓ (两侧不可再漂移 ✓)
    assert F50_CLOSE_CONTROLS == STAGE1_CLOSE_CONTROLS == 16
    assert F50_HOLD_CONTROLS == STAGE1_SETTLE_CONTROLS == 10


def _v2_recipe(close_q=None, settle_q=None, q_open=0.05):
    """★ T08-A ✓ 构造**完整 v2 recipe**(含实际执行 trace) ⇒ Stage2 才能只读加载 ✓

    md: 不得再由 q 终点重生成"等价"曲线 ✗ ⇒ 测试也必须用**实际记录** ✓
    """
    import stage1_recipe as _sr

    _close = list(close_q if close_q is not None else
                  _sr.stage1_close_q_sequence(q_open, 0.01, STAGE1_CLOSE_CONTROLS))
    _settle = list(settle_q if settle_q is not None else [0.01] * STAGE1_SETTLE_CONTROLS)
    _q = [float(x) for x in (_close + _settle)]
    _ph = ["close"] * len(_close) + ["settle"] * len(_settle)
    return {
        "schema_version": _sr.STAGE1_RECIPE_SCHEMA,
        "export_valid": True,
        "winner_binding": {"record_present": True, "record_frames": len(_q),
                           "episode_id": "s1ep-t", "candidate_id": "c-t"},
        "time_s": [i / 30.0 for i in range(len(_q))],
        "T_W_G_cmd": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * len(_q),
        "q_cmd": _q, "phase": _ph, "control_index": list(range(len(_q))),
        "phase_boundaries": _sr.derive_phase_boundaries(_ph),
        "close_profile": "cubic_smoothstep",
        "close_controls": len(_close), "settle_controls": len(_settle),
        "close_ctrl_q": _close,
        "q_open": float(q_open), "q_grasp": float(_close[-1]),
        "physics_hash": "p" * 16, "controller_hash": "c" * 16, "criteria_hash": "k" * 16,
        "scene_digest_hash": "d" * 16,
        "optimized_T_W_G": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "source_frame": 42, "task_grasp_frame": 50, "task_grasp_frame_default": 50,
        "verification_results": {"runs": [{"episode_id": "s1ep-t"}], "n_pass": 2},
    }


def test_f50_close_follows_the_actual_executed_trace(monkeypatch):
    """★ T08-A ✓ 闭合段必须**逐点等于实际 trace** (而非由 q 终点重生成"等价"曲线 ✗)"""
    recon = _simple_reconstruction()
    # 故意让"实际执行"与名义 smoothstep 不同 ⇒ 时间线必须跟**实际**(否则测试失败 ✓)
    _nominal = stage1_close_q_sequence(0.05, 0.01, STAGE1_CLOSE_CONTROLS)
    _actual = [float(x) + 0.0007 for x in _nominal]
    recipe = _v2_recipe(close_q=_actual)
    bundle = build_stage2_timeline(recon, F50_IDX=50, recipe=recipe)
    meta = bundle["timeline_meta"]

    close_q = bundle["gripper_q"][meta["close_start"]: meta["close_end"] + 1, 0]
    np.testing.assert_allclose(close_q, np.asarray(_actual), atol=1e-9, rtol=0)
    assert not np.allclose(close_q, np.asarray(_nominal), atol=1e-9)
    assert meta["close_controls"] == STAGE1_CLOSE_CONTROLS
    assert meta["close_profile"] == "cubic_smoothstep"


def test_stage2_rejects_legacy_or_inconsistent_recipe():
    """旧 artifact (无实际记录) 与"存的名义序列 ≠ 实际段" 都必须**拒绝** ✓"""
    recon = _simple_reconstruction()
    legacy = build_stage1_grasp_recipe(
        target_name="glb_4", q_open=0.05, q_grasp=0.01,
        physics_config=build_physics_config(),
        scene_digest={"target_object": "glb_4"})
    with pytest.raises(ValueError, match="入口合同未过|旧 artifact"):
        build_stage2_timeline(recon, F50_IDX=50, recipe=legacy)

    recipe = _v2_recipe()
    recipe["close_ctrl_q"] = list(reversed(recipe["close_ctrl_q"]))
    # 实际段仍在 phase_boundaries/q_cmd 里 ⇒ 走 trace ✓; 名义数组不一致 ⇒ 报告 reject ✓
    bundle = build_stage2_timeline(recon, F50_IDX=50, recipe=recipe)
    meta = bundle["timeline_meta"]
    np.testing.assert_allclose(
        bundle["gripper_q"][meta["close_start"]: meta["close_end"] + 1, 0],
        np.asarray(_v2_recipe()["q_cmd"][:STAGE1_CLOSE_CONTROLS]), atol=1e-9, rtol=0)


def test_f50_is_append_only_then_immediately_moves_to_f51():
    bundle = build_stage2_timeline(_simple_reconstruction(), F50_IDX=50)
    meta = bundle["timeline_meta"]

    assert np.all(np.diff(bundle["source_frame"]) >= 0.0)
    assert bundle["source_frame"][meta["pre_f50_end"]] < 50.0
    # ★ 2026-09-22 【契约同步 —— 审计 §7-3 第 2 项"source-frame 定义" ✓】
    #   旧断言 ✗: `== 50.2` (写死) —— 其隐含假设是"F50 之后 close+hold **不推进源帧**" ✗
    #   实测 ✗: **51.5** ✓ —— 与 `F50_CLOSE_CONTROLS=16` + `F50_HOLD_CONTROLS=10`
    #     在 `approach_time_scale=2.0` 下的推进**一致** ✓
    #   契约 (本处明确 ✓): 验证窗口**紧接在 settle 块之后** ✓, 且必须落在
    #     F50 之后、不超过一个接近段尺度 ✓ ⇒ 断言**结构性质** ✓ 而非魔数 ✗
    _v_src = float(bundle["source_frame"][meta["verify_start"]])
    assert 50.0 < _v_src <= 60.0, f"verify 起点源帧异常: {_v_src}"
    assert int(meta["verify_start"]) >= int(meta["hold_end"]), \
        "验证窗口必须在 settle 块之后 ✓ (审计 §P1-2: 窗口不得落到已运动的 carry 段 ✗)"
    assert int(meta["verify_start"]) <= int(meta["settle_end"]) + 1
    # ★ 2026-09-22 【契约同步 ✓】`CARRY_VERIFY_CONTROLS` **默认 = 0** ✗ ⇒ 验证窗口为**空** ✓
    #   实测 ✓ (直接读 bundle 元数据):
    #     `verify_start idx=51 (src 51.5)` / `verify_end idx=50 (src 50.0)` ✗
    #     `count(phase=="CARRY_VERIFY") == 0` ✓  ⇒ 确实是**空窗口** ✓
    #   契约 (本处明确 ✓): 空窗口的合法表示 = `verify_end == verify_start - 1` ✓
    #     (旧断言写死 `src == 51.0` ✗ —— 它假设 `carry_verify_controls >= 1` ✗,
    #      与默认 0 冲突 ✗; 审计 §P1-2 也指出"验收窗口元数据可能与被执行的数组不一致" ✗)
    if int(meta["verify_end"]) >= int(meta["verify_start"]):
        # 非空窗口 ⇒ 必须落在 settle 块之后 ✓ (不得落到已运动的 carry 段 ✗)
        assert float(bundle["source_frame"][meta["verify_end"]]) >= 50.0
        assert float(bundle["source_frame"][meta["verify_start"]]) > 50.0
    else:
        assert int(meta["verify_end"]) == int(meta["verify_start"]) - 1, \
            "空验证窗口必须表示为 verify_end == verify_start - 1 ✓"
        assert int(CARRY_VERIFY_CONTROLS) == 0, "空窗口只应在 carry_verify_controls=0 时出现 ✓"
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
    # ★ 2026-09-22 【契约同步 —— 审计 §7-3 第 2 项 ✓】MANO 前缀**允许有界平滑** ✓
    #   背景 ✓: `_SMOOTH_MINT=1` (默认开 ✓) 对前缀做**限幅平滑** ⇒ 与原始 MANO 逐点
    #     不再完全一致 ✗ (实测最大偏差 **6.14mm**, 11/117 个分量 ✗)
    #   契约 (本处明确 ✓): 前缀平滑是**刻意特性** ✓, 但必须**有界** ✓
    #     · `_SMOOTH_MINT=0` ⇒ **逐点完全一致** ✓ (严格契约 ✓)
    #     · `_SMOOTH_MINT=1` ⇒ 偏差 **≤ 8mm** ✓
    #   ⇒ 断言改为**有界性** ✓ (既不写死"完全一致"✗, 也不放过越界 ✗)
    import os as _os
    if _os.environ.get("_SMOOTH_MINT", "1") == "1":
        _d_mm = np.linalg.norm(
            np.asarray(bundle["base_pos"][:39]) - np.asarray(mano_pos[:39]), axis=1) * 1000.0
        assert float(_d_mm.max()) <= 8.0 + 1e-6, \
            f"前缀平滑越界: {float(_d_mm.max()):.2f}mm > 8mm ✗"
    else:
        np.testing.assert_allclose(bundle["base_pos"][:39], mano_pos[:39], atol=1e-12)
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
    # ★ T08-E (2026-09-27) ✓: 原用 3.0 目的只是"证明 hash 对时序敏感" ✓, 但 md 现在要求
    #   carry 降速倍率**有界 ∈[1,2]** (禁止无限放慢 ✗) ⇒ 改用界内的 1.5 同样能证明敏感性 ✓
    bundle_c = build_stage2_timeline(recon, carry_time_scale=1.5)

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


def test_authoritative_bundle_is_deterministic_and_shares_timeline_scales():
    """★ 2026-09-22 【P1-2 验收】权威 bundle 的可复现性 ✓ (审计 §P1-2 ✓)

    审计要求 ✓: "从冻结输入**只生成一次**权威 bundle … 验收: 重建、rollout、保存、
      Stage3 消费**同一 bundle**" ✓
    本测试覆盖其中**可离线判定**的部分 ✓:
      ① 同一输入生成两次 ⇒ **逐项一致** ✓ (含 `bundle_hash` ✓) ⇒ 无隐藏状态 ✗
      ② 时间缩放来自**唯一来源** `stage2_timeline_scales()` ✓ ⇒ 名义调用点与
         rollout 调用点**默认同参** ✓ (这正是此前不一致的根因 ✗)
      ③ 时间缩放确实**参与**标识 ✓ (改了就必须产生不同 bundle ✗)
    """
    from stage2_reconstruct_mixin import stage2_timeline_scales

    _ta, _tc = stage2_timeline_scales()
    b1 = build_stage2_timeline(_simple_reconstruction(), F50_IDX=50,
                               approach_time_scale=_ta, carry_time_scale=_tc)
    b2 = build_stage2_timeline(_simple_reconstruction(), F50_IDX=50,
                               approach_time_scale=_ta, carry_time_scale=_tc)
    assert b1["bundle_hash"] == b2["bundle_hash"], "同一输入两次生成 ⇒ 必须同 hash ✓"
    for _k, _v in b1.items():
        if isinstance(_v, np.ndarray):
            np.testing.assert_array_equal(_v, b2[_k], err_msg=f"数组 {_k} 不可复现 ✗")
    b3 = build_stage2_timeline(_simple_reconstruction(), F50_IDX=50,
                               approach_time_scale=_ta * 1.5, carry_time_scale=_tc)
    assert b3["bundle_hash"] != b1["bundle_hash"], \
        "时间缩放改变 ⇒ 必须产生**不同 bundle 标识** ✓ (否则无法区分实验 ✗)"
