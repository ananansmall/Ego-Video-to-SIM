"""T072 反例 (不跑物理 ✓)

① md T07.2 ✓: "**强制触发出口限速**时, trace 必须等于**下发 spy 记录**,
   **而不是 clip 前值**" ✗ ⇒ 用 compare_trace_with_spy 逐点核对
② md T07.2 ✓: "判据标签 v2 但**无对应评估记录**、profile 不符、n_pass 不足,
   **所有正常 Stage2 入口均拒绝**; 诊断 override **不能产生正式 PASS**" ✓
"""
import os

import physics_env as pe
from stage1_recipe import (STAGE1_RECIPE_SCHEMA, compare_trace_with_spy,
                           validate_stage2_entry)

# ── ① 下发 spy 机制 (只读钩子 ✓) ─────────────────────────────────────────────
def test_spy_reset_and_read_empty_by_default():
    pe.set_dispatch_spy(False)
    assert pe.get_dispatch_spy() == []
    pe.set_dispatch_spy(True)
    assert pe.get_dispatch_spy() == []          # 开启即清空 ✓


def test_spy_hook_sits_inside_physics_step():
    """spy 必须挂在**驱动入口**(physics_step) 内 ⇒ 记的是真正下发的值 ✓"""
    import inspect
    src = inspect.getsource(pe.physics_step)
    assert "_DISPATCH_SPY" in src and "gripper_target1" in src


def test_trace_must_equal_dispatch_not_preclip():
    """限速(clip)生效时: 下发=clip后; trace 若记 clip 前 ⇒ **必须被拒** ✗"""
    preclip = [0.0300, 0.0300, 0.0300, 0.0300]       # 限速前的"想要"值
    dispatch = [0.0300, 0.0295, 0.0290, 0.0285]      # 被 0.0005/帧 限速后**实际下发**
    ok, det = compare_trace_with_spy(dispatch, dispatch)
    assert ok, "trace==下发 ⇒ 通过 ✓ (%s)" % det
    ok2, det2 = compare_trace_with_spy(preclip, dispatch)
    assert not ok2 and "clip" in det2, "记了 clip 前值 ⇒ 必须判不一致 ✗ (%s)" % det2


def test_spy_length_and_empty_cases():
    assert not compare_trace_with_spy([0.01], [])[0]            # spy 空
    assert not compare_trace_with_spy([0.01] * 3, [0.01] * 2)[0]  # 长度不一致(错帧)


# ── ② Stage2 入口合同 ─────────────────────────────────────────────────────────
def _v2(**over):
    r = {
        "schema_version": STAGE1_RECIPE_SCHEMA,
        "export_valid": True,
        "winner_binding": {"record_present": True, "record_frames": 5,
                           "episode_id": "s1ep-1", "candidate_id": "c1"},
        "time_s": [0.0, 0.033, 0.066], "T_W_G_cmd": [[[1, 0, 0, 0], [0, 1, 0, 0],
                                                     [0, 0, 1, 0], [0, 0, 0, 1]]] * 3,
        "q_cmd": [0.030, 0.028, 0.026], "phase": ["close"] * 3,
        "control_index": [0, 1, 2], "physics_hash": "p" * 16,
        "verification_results": {"runs": [{"episode_id": "s1ep-1"}], "n_pass": 2},
    }
    r.update(over)
    return r


def test_v2_complete_recipe_passes_entry():
    os.environ.pop("_ALLOW_UNVERIFIED_STAGE1", None)
    ok, errs = validate_stage2_entry(_v2())
    assert ok, "完整 v2 recipe 应放行 ✓ (%s)" % errs


def test_v2_without_assessment_is_rejected():
    ok, errs = validate_stage2_entry(_v2(verification_results=None))
    assert not ok and any("评估记录" in e or "verification_results" in e for e in errs)


def test_v2_insufficient_n_pass_rejected():
    ok, errs = validate_stage2_entry(_v2(verification_results={"runs": [], "n_pass": 1}))
    assert not ok and any("n_pass" in e for e in errs)


def test_v2_export_invalid_rejected():
    ok, errs = validate_stage2_entry(_v2(export_valid=False))
    assert not ok and any("export_valid" in e for e in errs)


def test_v2_missing_record_rejected():
    ok, errs = validate_stage2_entry(_v2(winner_binding={"record_present": False}))
    assert not ok and any("record_present" in e for e in errs)


def test_old_artifact_rejected_and_not_strict_replay_evidence():
    old = {"q_open": 0.03, "q_grasp": 0.01,
           "verification_results": {"runs": [], "n_pass": 3}}
    ok, errs = validate_stage2_entry(old, mode="feedback")
    assert not ok and any("旧 artifact" in e or "schema" in e for e in errs)
    ok2, errs2 = validate_stage2_entry(old, mode="strict_replay")
    assert not ok2 and any("strict_replay" in e for e in errs2)


def test_diagnostic_override_only_passes_not_formal():
    """诊断 override: 放行但**不产生正式 PASS** ✓ (errors 仍返回 ✓)"""
    ok, errs = validate_stage2_entry({"q_open": 0.03}, allow_unverified=True)
    assert ok is True and errs, "override 只放行, 仍须报出问题清单 ✓"
