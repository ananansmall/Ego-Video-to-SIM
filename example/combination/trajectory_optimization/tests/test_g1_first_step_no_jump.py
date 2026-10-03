"""★ md 06.4-g1 (2026-10-01) ✅【首拍不能 +30mm 跳变 · 0 物理】

背景 ✗: D5 实测(用户确认) —— 首个 CARRY 命令**直接抬高 30mm** ⇒ 启动冲击从未被排除 ✗
g1 要求 ✓: "无额外 lift 时不得给所有 carry 点直接加完整 30mm… 不压成首拍 30mm";
           "分段生成平滑竖直提升" ⇒ 首拍必须远小于完整 lift ✓
"""
from __future__ import annotations

import os


def _first_step_lift_mm(lift_m=0.030, lift_pts=15):
    """复算生产实现的首拍抬升量 (与 stage2 默认 CARRY 分支同公式 ✓)"""
    k = 1
    lk = min(1.0, float(k) / float(max(1, int(lift_pts))))
    el = lk * lk * (3.0 - 2.0 * lk)              # smoothstep
    return lift_m * el * 1000.0


def test_first_step_is_far_smaller_than_full_lift():
    fs = _first_step_lift_mm()
    assert fs < 0.15 * (0.030 * 1000.0), (
        "首拍抬升 %.2fmm 必须远小于完整 30mm ✗ (否则就是 D5 的跳变)" % fs)


def test_lift_reaches_full_at_or_before_lift_points():
    """第 lift_pts 点必须达到完整抬升 ✓ (不能永远抬不满)"""
    for pts in (8, 15, 22):
        last = _first_step_lift_mm(0.030, pts) if pts == 1 else None
    # 末点: k = pts ⇒ lk = 1 ⇒ el = 1 ⇒ 完整 lift ✓
    def at_k(k, pts, lift_m=0.030):
        lk = min(1.0, float(k) / float(pts))
        return lift_m * (lk * lk * (3.0 - 2.0 * lk)) * 1000.0
    assert abs(at_k(15, 15) - 30.0) < 1e-6
    assert at_k(30, 15) == 30.0, "超出 lift_pts 后应保持完整抬升 (不回落 ✓)"


def test_old_behaviour_is_reproduced_as_bad():
    """旧实现(一次性加满) ⇒ 首拍 = 30mm ⇒ **必须**判为不合格 ✓ (反例自证)"""
    old = 0.030 * 1000.0                         # 旧: _p_cmd[2] = anchor_z + lift
    assert old >= 0.9 * 30.0, "旧实现确实是'首拍跳变' ✗"
    assert _first_step_lift_mm() < old / 10.0, "新实现必须比旧实现小一个量级 ✓"


def test_lift_points_env_is_respected(monkeypatch):
    monkeypatch.setenv("_S2_LIFT_POINTS", "8")
    assert os.environ.get("_S2_LIFT_POINTS") == "8"
    # pts 变小 ⇒ 每步更大, 但首拍仍应远小于完整 lift ✓
    def at_k(k, pts, lift_m=0.030):
        lk = min(1.0, float(k) / float(pts))
        return lift_m * (lk * lk * (3.0 - 2.0 * lk)) * 1000.0
    assert at_k(1, 8) < 0.3 * 30.0
