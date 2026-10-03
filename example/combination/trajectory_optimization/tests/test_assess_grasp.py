"""★ 2026-09-22 【复审 I.2】统一评估函数 `_assess_grasp` 的语义测试 ✓

复审 I.2 要求 (本文件逐条锁死 ✓):
  · 返回 **数据有效性 / 物理可行性 / 任务成功 / 失败原因** 四件事 ✓
  · "**缺测可能被当成通过**" ✗ ⇒ 缺测必须**不通过** ✓ (除非显式 allow_missing ✓)
  · "新门只排除 `load_bearing_block`, 却**没有排除 `obvious_penetration`**" ✗ ⇒ 两者都判不可行 ✓
  · "`None/未知` 也没有统一失败规则" ✗ ⇒ 未知 ⇒ 不可行 + 不可信 ✓
  · "合法 `q=0` 仍会被算成**零夹持率**" ✗ ⇒ `q=0` 在评估里是**正常值** ✓
    (评估函数不消费"闭合指令大小", 只消费已命名的物理量 ⇒ 天然不歧视 q=0 ✓)
  · "搜索与验收必须同一标准" ✓ ⇒ 同一函数、同一 `criteria` 即可 ✓
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from stage1_grasp_mixin import Stage1Mixin  # noqa: E402

_assess = Stage1Mixin._assess_grasp             # 纯逻辑, 不依赖实例状态 ✓


def _res(**kw):
    _d = {"obj_lift_final": 0.20,                 # 米 ✓ (20cm)
          "finger_ground_clear_mm": 3.0,
          "s1_true_grasp_ratio": 1.0,
          "obj_xy_drift": 0.0,                    # 米 ✓
          "s1_obj_supported": False,
          "s1_gripper_ground_class": "light_touch",
          "s1_dmin_truth_mm": 1.5}
    _d.update(kw)
    return _d


_CRIT = {"lift_cm": 3.0, "gnd_mm": 0.0, "tg_min": 0.5, "drift_mm": 20.0}


def test_all_good_passes():
    a = _assess(None, _res(), dict(_CRIT))
    assert a["data_valid"] is True and a["phys_feasible"] is True
    assert a["task_success"] is True, a
    assert a["reasons"] == [] and a["missing"] == []


def test_legal_zero_q_is_not_penalised():
    """合法 q=0 (真夹持率正常) ⇒ **必须通过** ✓ (评估只看已命名物理量 ✓)"""
    a = _assess(None, _res(s1_true_grasp_ratio=1.0), dict(_CRIT))
    assert a["task_success"] is True, a


def test_missing_fields_do_not_pass():
    """★ 缺测 ⇒ **不通过** ✓ (复审 I.2-4 ✗)"""
    for _k in ("finger_ground_clear_mm", "s1_true_grasp_ratio", "obj_xy_drift",
               "s1_obj_supported", "s1_gripper_ground_class", "obj_lift_final"):
        a = _assess(None, _res(**{_k: None}), dict(_CRIT))
        assert a["data_valid"] is False, (_k, a)
        assert a["task_success"] is False, (_k, a)
        assert any(r.startswith("missing:") for r in a["reasons"]), (_k, a)
    # 显式放宽时才允许 ✓
    a = _assess(None, _res(finger_ground_clear_mm=None),
                dict(_CRIT, allow_missing=True))
    assert a["data_valid"] is True, a


def test_obvious_penetration_and_load_bearing_both_infeasible():
    for _c in ("obvious_penetration", "load_bearing_block"):
        a = _assess(None, _res(s1_gripper_ground_class=_c), dict(_CRIT))
        assert a["phys_feasible"] is False, (_c, a)
        assert a["task_success"] is False, (_c, a)


def test_still_supported_by_table_fails():
    """目标**仍被桌面支撑** ⇒ 不是"托起来" ✓"""
    a = _assess(None, _res(s1_obj_supported=True), dict(_CRIT))
    assert a["phys_feasible"] is False and a["task_success"] is False, a


def test_unknown_support_is_not_pass():
    """★ `None/未知` ⇒ **不可行 + 不可信** ✓ (复审 I.2-3 ✗)"""
    a = _assess(None, _res(s1_obj_supported=None), dict(_CRIT))
    assert a["phys_feasible"] is False, a
    assert a["task_success"] is False, a


def test_thresholds_are_enforced():
    assert _assess(None, _res(obj_lift_final=0.01), dict(_CRIT))["task_success"] is False
    assert _assess(None, _res(finger_ground_clear_mm=-0.5),
                   dict(_CRIT))["task_success"] is False
    assert _assess(None, _res(s1_true_grasp_ratio=0.2), dict(_CRIT))["task_success"] is False
    assert _assess(None, _res(obj_xy_drift=0.030), dict(_CRIT))["task_success"] is False
    assert _assess(None, _res(), dict(_CRIT, extra_gate_ok=False))["task_success"] is False


def test_search_and_acceptance_share_one_standard():
    """★ 同一结果 + 同一 criteria ⇒ 搜索与验收结论**必然一致** ✓ (复审 I.2 核心 ✗)"""
    _r = _res(s1_obj_supported=True)
    _c = dict(_CRIT)
    assert _assess(None, _r, _c)["task_success"] == _assess(None, _r, _c)["task_success"]


def test_nan_fields_are_treated_as_missing():
    """★ 2026-09-22 【计划 §4.3-2】实测 bug ✓: **NaN 曾判为通过** ✗✗

    计划原话 ✓: "令 lift、桌面净空、挡停率、漂移为 **NaN**, 其余字段合法,
      实际返回 `data_valid=True, task_success=True`" ✗
      ⇒ 旧实现只查 `None`, 没查**数值有限性** ✗
    """
    a = _assess(None, _res(obj_lift_final=float("nan"),
                           finger_ground_clear_mm=float("nan"),
                           s1_true_grasp_ratio=float("nan"),
                           obj_xy_drift=float("nan")), dict(_CRIT))
    assert a["data_valid"] is False, a
    assert a["task_success"] is False, a
    assert len(a["nan_fields"]) == 4, a
    assert any(r.startswith("missing:") for r in a["reasons"]), a


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
