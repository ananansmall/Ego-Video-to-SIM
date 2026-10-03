"""★ CARRY 速度剖面(梯形)的零物理单元测试 (2026-09-30) ✓

背景 ✗: 我连续两次把剖面公式写错 ——
  ① `r=0.5` ⇒ 无匀速段 ⇒ 公式失效 (位置在 u=0.5 不连续 ✗)
  ② 漏**峰值速度归一化** ⇒ 三段积分 = 1-r ≠ 1 ⇒ 在 u=1-r 处跳变 ✗~
     (实测: `s_eng_A5` base_target 第 231→232 帧跳 **37mm** ✗)
⇒ 本文件把"端点/单调/连续性/起止速度"全部钉死 ✓
"""
from __future__ import annotations

import numpy as np
import pytest


def _profile(n, r=0.5):
    """★ f1 ✓: **调用生产 helper** (`carry_retime_s`) ⇒ 生产与测试同源 ✓
    (md 7.1: 测试必须调用生产入口; 不允许测试里重写一份公式 ✗)
    """
    from stage2_reconstruct_mixin import carry_retime_s
    return np.asarray([carry_retime_s(k / float(n), 0.0, 1.0, r) for k in range(1, n + 1)])


@pytest.mark.parametrize("n", [16, 22, 40])
def test_profile_endpoints_and_monotonic(n):
    es = _profile(n)
    assert es[-1] == pytest.approx(1.0, abs=1e-9), "终点必须精确到 1 ✓"
    assert np.all(np.diff(es) > 0), "必须严格单调 (不能倒退 ✓)"


def test_profile_has_no_jump():
    """相邻步长的跳变不得超过中位步长的 3 倍 ⇒ 消除"分段边界跳变" ✓"""
    es = _profile(22)
    d = np.diff(np.concatenate([[0.0], es]))
    med = float(np.median(d))
    assert d.max() <= 3.0 * med + 1e-9, (
        "剖面在分段边界跳变 ✗ (这正是 r=0.5 / 漏归一化 两次 bug 的症状)")


def test_profile_velocity_soft_at_both_ends():
    """起止速度接近 0 ⇒ 无"阶跃起步/急停" ✓"""
    es = _profile(40)
    v0 = es[0] - 0.0
    v1 = 1.0 - es[-2]
    vmid = float(np.max(np.diff(es)))
    assert v0 < 0.35 * vmid, "起点速度过大 ✗"
    assert v1 < 0.35 * vmid, "终点速度过大 ✗"


def test_profile_peak_accel_is_lower_than_quintic():
    """梯形峰值加速度应**明显低于** quintic smoothstep ✓ (md: 削掉加速尖峰)"""
    es = _profile(40)
    v = np.diff(np.concatenate([[0.0], es]))
    a_tr = float(np.max(np.diff(v))) * 40 * 40          # 归一到 d/T²
    u = np.arange(1, 41) / 40.0
    e_q = u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))       # quintic smoothstep
    v_q = np.diff(np.concatenate([[0.0], e_q]))
    a_q = float(np.max(np.diff(v_q))) * 40 * 40
    assert a_tr < 0.75 * a_q, "梯形峰值加速度未低于 quintic ✗ (a_tr=%.2f a_q=%.2f)" % (a_tr, a_q)
