"""T08-A 反例: 模式分流 + 拒绝重生成 (不跑物理 ✓)

md ✓: "先过 T07 schema/identity/profile 门, 再选模式; feedback 消费 ExecutionSpec,
  strict_replay 消费**完整实际 trace**; 两者均**不得再由 q 终点调用
  `stage1_close_q_sequence` 生成另一条'等价'曲线**" ✗
验收: 37/17 反例 (close=37, settle=17) 与 close=3 时 数组/时标/phase 计数一致 ✓
"""
import numpy as np
import pytest

import stage1_recipe as sr
from stage1_recipe import (STAGE1_RECIPE_SCHEMA, select_close_mode,
                           split_trace_by_phase)


def _recipe(close_n=37, settle_n=17, travel_n=0):
    ph = ["travel"] * travel_n + ["close"] * close_n + ["settle"] * settle_n + ["lift"] * 2
    n = len(ph)
    q = [0.030 - 0.0004 * i for i in range(n)]
    return {
        "schema_version": STAGE1_RECIPE_SCHEMA,
        "export_valid": True,
        "winner_binding": {"record_present": True, "record_frames": n,
                           "episode_id": "s1ep-1", "candidate_id": "c1"},
        "time_s": [i / 30.0 for i in range(n)],
        "T_W_G_cmd": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * n,
        "q_cmd": q, "phase": ph, "control_index": list(range(n)),
        "phase_boundaries": sr.derive_phase_boundaries(ph),
        "physics_hash": "p" * 16, "controller_hash": "c" * 16, "criteria_hash": "k" * 16,
        "q_open": 0.030, "q_grasp": q[ph.index("close") + close_n - 1],
        "optimized_T_W_G": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "source_frame": 42, "task_grasp_frame": 50, "task_grasp_frame_default": 50,
        "close_profile": "smoothstep", "close_controls": close_n,
        "settle_controls": settle_n, "close_ctrl_q": q[ph.index("close"):ph.index("close") + close_n],
        "verification_results": {"runs": [], "n_pass": 2},
    }


def test_strict_replay_uses_actual_trace_slices():
    r = _recipe(37, 17)
    mode, pay, errs = select_close_mode(r, "strict_replay")
    assert not errs, errs
    assert pay["provenance"] == "actual_executed"
    assert len(pay["close_q"]) == 37 and len(pay["settle_q"]) == 17
    assert pay["close_idx"] == [0, 37] and pay["settle_idx"] == [37, 54]


def test_close3_counter_example_still_consistent():
    r = _recipe(3, 1)
    mode, pay, errs = select_close_mode(r, "strict_replay")
    assert not errs and len(pay["close_q"]) == 3 and pay["close_idx"] == [0, 3]
    tr = split_trace_by_phase(r)
    assert len(tr["close"]["q"]) == 3 and len(tr["settle"]["q"]) == 1
    assert not tr["_errors"]


def test_feedback_consumes_spec_not_regenerated_curve():
    r = _recipe(37, 17)
    mode, pay, errs = select_close_mode(r, "feedback")
    assert not errs, errs
    assert pay["provenance"] == "execution_spec" and pay["adaptive_close"] is True
    spec = pay["spec"]
    assert spec["load_target"] == pytest.approx(r["close_ctrl_q"][-1])   # 实际末点 ✓
    assert spec["close_entry_q"] == pytest.approx(r["close_ctrl_q"][0])  # 实际入口 ✓
    assert spec["anchor"]["source_frame"] == 42 and spec["anchor"]["task_grasp_frame"] == 50


def test_never_calls_stage1_close_q_sequence(monkeypatch):
    """把重生成函数改成**抛异常** ⇒ 模式分流仍应正常 ⇒ 证明它不再被调用 ✓"""
    def _boom(*a, **k):
        raise AssertionError("不得调用 stage1_close_q_sequence 生成'等价'曲线 ✗")

    monkeypatch.setattr(sr, "stage1_close_q_sequence", _boom, raising=True)
    r = _recipe(37, 17)
    assert select_close_mode(r, "strict_replay")[2] == []
    assert select_close_mode(r, "feedback")[2] == []


def test_legacy_or_incomplete_recipe_rejected():
    r = _recipe(37, 17)
    r["phase_boundaries"] = {}
    mode, pay, errs = select_close_mode(r, "strict_replay")
    assert pay is None and errs
    old = {"q_open": 0.03, "q_grasp": 0.01, "verification_results": {"n_pass": 3}}
    assert select_close_mode(old, "feedback")[1] is None
    assert select_close_mode(old, "strict_replay")[1] is None


def test_reuse_report_items_have_expected_observed_match():
    r = _recipe(37, 17)
    rep = sr.reuse_report_items(r, observed={"physics_hash": r["physics_hash"]},
                                mode="feedback")
    names = [i["field"] for i in rep["items"]]
    for need in ("schema_version", "physics_hash", "controller_hash", "criteria_hash",
                 "optimized_T_W_G", "source_frame", "task_grasp_frame", "q_grasp",
                 "close_controls_actual", "settle_controls_actual", "mode"):
        assert need in names, "缺字段 %s ✗" % need
    item = [i for i in rep["items"] if i["field"] == "physics_hash"][0]
    assert item["match"] is True
    assert [i for i in rep["items"] if i["field"] == "close_controls_actual"][0]["expected"] == 37
