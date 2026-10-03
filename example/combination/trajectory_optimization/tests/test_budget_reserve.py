"""★ md §09-2/3/4 预算保留的零物理测试 (2026-09-30) ✓

背景 ✗: `s102_deep`/`s102_deep2` 两次 **48/48 全被搜索吃掉 ⇒ 零 recipe**;
        补上"搜索为终验留位"后, 首版又因 **`BudgetExhausted(rule, detail)` 只传 1 个参数**
        抛 `TypeError` ⇒ `s102_deep3` 在 **45/48 崩溃** ✗ ⇒ 本文件把契约钉死 ✓
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline_budget import BudgetExhausted, EpisodeBudget  # noqa: E402


def test_budget_exhausted_contract_two_args():
    e = BudgetExhausted("RULE_X", "detail text")
    assert e.rule == "RULE_X" and e.detail == "detail text"
    assert "BUDGET_EXHAUSTED:RULE_X" in str(e)
    # 只传一个参数**必须**报错 ⇒ 任何"少传参数"的调用会在测试期暴露 ✓
    with pytest.raises(TypeError):
        BudgetExhausted("only-one-arg")


def test_reserve_refuses_when_exhausted(tmp_path):
    b = EpisodeBudget(tmp_path / "led.json", task="stage1", action="search",
                      physical_budget=2)
    b.reserve(purpose="p1")
    b.reserve(purpose="p2")
    assert b.remaining() == 0
    with pytest.raises(BudgetExhausted):
        b.reserve(purpose="p3")            # reset/step 之前拒绝 ✓


def test_remaining_is_used_for_verify_reserve_rule(tmp_path):
    """搜索留位判据: `remaining() <= keep` ⇒ 应停止搜索 ✓ (与实现同口径)"""
    b = EpisodeBudget(tmp_path / "led2.json", task="stage1", action="search",
                      physical_budget=48)
    keep = 3
    for _ in range(45):
        assert b.remaining() > keep, "未到 45 次前不该触发留位停搜 ✓"
        b.reserve(purpose="screen")
    assert b.remaining() == keep
    assert b.remaining() <= keep, "达到留位线 ⇒ 搜索应停止 ✓ (终验仍有 3 次)"
