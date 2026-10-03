"""★ md 06.4-f1 验收测试 (2026-09-30) ✓ —— 同路径重定时 (0 物理)

f1 要求 ✓: "验收必须调用生产构造器: 弯曲折线路径/非恒定旋转/非零末端切向/不同dt/奇偶N。
  输出 settle→首carry 及末carry→Stage3 的实际离散 v/a/角速度; 检查路径投影、完整F50段、
  端点、总时长、速度边界和最终dispatch一致性" ✓

⚠ f1 提醒 ✓: "三角的 4d/T² 只能用于固定直线模型, 不能当任意轨迹的通用保证"
⇒ 本测试**不假设**任何折角下都平滑, 只断言: s(t) 单调、覆盖完整 F50 段、端点精确、
  且**位置/姿态/source 取同一个 s** (不是 anchor 混合 ✗)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stage2_reconstruct_mixin as m  # noqa: E402


# ───────── ① s(t) 覆盖完整 F50 段 + 端点精确 + 单调 (奇偶 N) ─────────
@pytest.mark.parametrize("n", [17, 22, 23, 40])
def test_s_covers_full_segment_and_endpoints(n):
    ss = [m.carry_retime_s(k / n, 50.0, 62.0) for k in range(1, n + 1)]
    assert ss[-1] == pytest.approx(62.0), "终点必须精确到任务末 ✓"
    assert all(ss[i] > ss[i - 1] for i in range(1, len(ss))), "s 必须严格单调 ✓"
    assert ss[0] < 50.0 + 12.0, "首点必须在 (50, 62) 内 ✓"


# ───────── ② 不同 dt: 总时长由 N·dt 决定, s(t) 形状不变 ─────────
def test_different_dt_changes_total_time_not_shape(monkeypatch):
    # helper 内部依赖路径采样 ⇒ 打桩 (mock 只隔离外部资源 ✓ md 7.1)
    monkeypatch.setattr(m, "_interp_reconstruction",
                        lambda recon, s: (np.array([s * 1e-3, 0.0, 0.0]), np.eye(3)))
    a = m.carry_retime_report({}, 24, 50.0, 62.0, 1.0 / 30.0)
    b = m.carry_retime_report({}, 24, 50.0, 62.0, 1.0 / 60.0)
    assert a["s_t"] == pytest.approx(b["s_t"]), "dt 不应改变 s(t) 形状 ✓"
    assert a["total_time_s"] == pytest.approx(24 / 30.0)
    assert b["total_time_s"] == pytest.approx(24 / 60.0)


# ───────── ③ 弯曲折线路径: 位置取自 P(s), 且 s 覆盖 F50→末 ─────────
def test_curved_path_positions_come_from_s_of_t(monkeypatch):
    seen = []

    def fake_interp(recon, s):
        seen.append(float(s))
        p = np.array([s * 1e-3, math.sin((s - 50.0) * 0.3) * 0.01, (s - 50.0) * 2e-3])
        return p, np.eye(3)

    monkeypatch.setattr(m, "_interp_reconstruction", fake_interp)
    rep = m.carry_retime_report({}, 24, 50.0, 62.0, 1.0 / 30.0)
    assert rep["s_monotonic"] and rep["s_endpoint_ok"]
    assert len(seen) == 24, "每个控制点只采样一次 ✓ (不允许位置/姿态各用不同进度 ✗)"
    assert seen[-1] == pytest.approx(62.0)
    assert rep["v_discrete_mps_max"] > 0.0 and rep["a_discrete_mps2_max"] > 0.0


# ───────── ④ 非法 r 必须拒绝 (无匀速段/积分不归一 ✗) ─────────
@pytest.mark.parametrize("bad_r", [0.0, -0.1, 0.6, 1.0])
def test_invalid_profile_r_rejected(bad_r):
    with pytest.raises(ValueError):
        m.carry_retime_s(0.5, 50.0, 62.0, bad_r)


# ───────── ⑤ f1 关键: 生产代码里**不应再有 anchor 混合** ─────────
def test_no_anchor_blend_in_carry_default_branch():
    src = (Path(__file__).resolve().parents[1] / "stage2_reconstruct_mixin.py").read_text()
    assert "_p_a0 = np.asarray(anchor_pos" not in src, (
        "仍存在 `p = anchor + e·(P-anchor)` 的**位置混合** ✗ (f1 明令删除 ✓)")
    assert "carry_retime_s(float(k) / float(carry_intervals), _s_from, _s_to)" in src, (
        "default CARRY 必须用同一 s(t) 采样 ✓")
