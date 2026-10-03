"""任务04 测试 —— Stage1 终验与交付**绑定同一个 run** (零物理 ✓)

md 位置: §4 任务 04（前置 03）
md 怎么测 ✓: "test_t07b/c/d/e 及 test_t072; **补充'124被选但只有128/129/130验证通过'**
  '相同q终点不同候选' '37/17非默认时长' '非有限数组'…" ✓
md 通过标准 ✓: "原始通过 run 可追溯; 所有正常 Stage2 入口重新验证合同" ✓

本文件用**纯函数** `assert_verification_binding` ⇒ 零物理 ✓ (§7.2: 先让失败反例成立 ✓)
实测旧病 (s31 / s44 / success_rec 三处一致) 🔴:
  `selected_episode='s1ep-000124'` 而终验通过的是 **128/129/130** ⇒ 记录与验证**不绑定** ✗
"""

from __future__ import annotations

import pytest

from stage1_recipe import assert_verification_binding


def _vr(selected="s1ep-000128", n_pass=3, ids=("s1ep-000128", "s1ep-000129", "s1ep-000130"),
        passed=(True, True, True)):
    return {
        "n_pass": n_pass,
        "n_repeat": 3,
        "selected_episode": selected,
        "runs": [{"run_index": i, "episode_id": e, "passed": bool(p)}
                 for i, (e, p) in enumerate(zip(ids, passed))],
    }


# --------------------------------------------------------------------------- #
# 1) ★ md 点名的反例: "124 被选但只有 128/129/130 验证通过"
# --------------------------------------------------------------------------- #
def test_selected_must_be_in_passed_list_the_124_counterexample():
    """★ md 原话反例 ✓: 源 000124 被选 ⇒ 必须**拒绝** (它不在 passed 列表里) ✗"""
    vr = _vr(selected="s1ep-000124")          # 搜索最优, 但**没参与终验** ✗
    ok, errs = assert_verification_binding(vr)
    assert ok is False, "124 被选却判通过 ⇒ md §04 绑定义务失效 ✗"
    assert any("不在" in e and "passed" in e for e in errs), errs


def test_selected_from_passed_list_passes():
    ok, errs = assert_verification_binding(_vr(selected="s1ep-000129"))
    assert ok is True, errs


# --------------------------------------------------------------------------- #
# 2) n_pass 必须由 runs 列表计算
# --------------------------------------------------------------------------- #
def test_n_pass_must_match_runs_count():
    vr = _vr(n_pass=3, passed=(True, False, False))      # 实际只 1 次通过 ✗
    ok, errs = assert_verification_binding(vr)
    assert ok is False
    assert any("n_pass" in e and "统计" in e for e in errs), errs


def test_missing_n_pass_is_rejected():
    vr = _vr()
    vr.pop("n_pass")
    ok, errs = assert_verification_binding(vr)
    assert ok is False and any("n_pass" in e for e in errs)


# --------------------------------------------------------------------------- #
# 3) runs 明细必须可追溯 (episode_id + passed 都在)
# --------------------------------------------------------------------------- #
def test_runs_must_be_traceable():
    vr = _vr()
    vr["runs"][1].pop("passed")
    ok, errs = assert_verification_binding(vr)
    assert ok is False and any("追溯不完整" in e for e in errs), errs

    vr2 = _vr()
    vr2["runs"][2]["episode_id"] = ""
    ok2, errs2 = assert_verification_binding(vr2)
    assert ok2 is False and any("追溯不完整" in e for e in errs2), errs2


def test_empty_runs_is_rejected():
    ok, errs = assert_verification_binding({"n_pass": 3, "runs": []})
    assert ok is False and any("空" in e for e in errs)


def test_selected_missing_is_rejected():
    vr = _vr()
    vr["selected_episode"] = None
    ok, errs = assert_verification_binding(vr)
    assert ok is False and any("selected_episode 缺失" in e for e in errs)


def test_zero_pass_still_self_consistent():
    """★ 边界 ✓: 0/3 时 selected 无合法取值 ⇒ 应报"缺失/不在" (不得编造 ✗)"""
    vr = _vr(selected="s1ep-000124", n_pass=0, passed=(False, False, False))
    ok, errs = assert_verification_binding(vr)
    assert ok is False
    assert any(("不在" in e) or ("缺失" in e) for e in errs), errs


def test_tolerant_of_extra_fields():
    """真实产物还有 criteria/failure_reason 等字段 ⇒ 不得因多余字段判错 ✓"""
    vr = _vr()
    vr.update({"criteria": {"lift_cm": 3.0}, "failure_reason": None,
               "n_pass_by_runs": 3, "selected_episode": "s1ep-000130"})
    ok, errs = assert_verification_binding(vr)
    assert ok is True, errs
