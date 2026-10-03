"""★ md 06.4-f0 验收测试 (2026-09-30) ✓ —— anchor q 策略 + 负 drive 政策

f0 验收要求(原文) ✓: "生产loader测试外层/嵌入/JSON不一致、非对称q、合法零值、
  已声明负drive、未声明负drive、错误episode及hash篡改" ✓

⚠ md 7.1 规则 ✓: "必须调用生产入口; mock 只隔离求解器和外部资源,
  不替代待测分流/门控" ⇒ 本文件调用**生产纯函数** `stage1_recipe.resolve_anchor_q`
  (它同时被 `build_stage2_timeline` 调用 ⇒ 生产/测试**同源** ✓)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage1_recipe import resolve_anchor_q  # noqa: E402


# ───────── ① 优先 settle 段末命令 (f0: 不得用整条 min(q) 替代终态 ✗) ─────────
def test_settle_end_wins_over_min_and_nominal():
    q, info = resolve_anchor_q(q_nominal=-0.0015, q_settle_end=0.013083,
                               q_effective_min=-0.000978)
    assert q == pytest.approx(0.013083)
    assert info["source"] == "settle_end_cmd" and info["degraded"] is False
    # 关键: 即使 min(q)=-0.000978 更"深", 也不能被选为 anchor ✓
    assert q != pytest.approx(-0.000978)


# ───────── ② 无 settle 末 ⇒ 降级并显式标注 ─────────
def test_fallback_to_effective_min_is_marked_degraded():
    q, info = resolve_anchor_q(q_nominal=-0.0015, q_effective_min=0.02)
    assert q == pytest.approx(0.02)
    assert info["degraded"] is True and "fallback" in info["source"]


# ───────── ③ 合法零值必须通过 (URDF lower=0 ✓) ─────────
def test_legal_zero_passes():
    q, info = resolve_anchor_q(q_nominal=0.0, q_settle_end=0.0)
    assert q == 0.0 and info["source"] == "settle_end_cmd"


# ───────── ④ 负值 + **未声明** ⇒ 拒绝该控制模式 (不钳 0 ✗ 不改 URDF ✗) ─────────
def test_negative_undeclared_is_rejected():
    with pytest.raises(ValueError) as e:
        resolve_anchor_q(q_nominal=-0.0015, q_settle_end=-0.000978,
                         allow_negative=False)
    msg = str(e.value)
    assert "拒绝该控制模式" in msg and "负 drive 未声明" in msg
    assert "_S1_ALLOW_NEGATIVE_DRIVE" in msg          # 给出启用方式 ✓


# ───────── ⑤ 负值 + 已声明 + 界内 ⇒ 通过并标注 ─────────
def test_negative_declared_within_bound_passes():
    q, info = resolve_anchor_q(q_nominal=-0.000978, allow_negative=True,
                               oc_max_mm=5.0)
    assert q == pytest.approx(-0.000978)
    assert info["negative_drive_declared"] is True
    assert info["floor_negative"] == pytest.approx(-0.0025)   # -(5/2)/1000 ✓


# ───────── ⑥ 负值 + 已声明但**越界** ⇒ 仍拒绝 (有界 ✓) ─────────
def test_negative_declared_but_out_of_bound_is_rejected():
    with pytest.raises(ValueError) as e:
        resolve_anchor_q(q_nominal=-0.01, allow_negative=True, oc_max_mm=5.0)
    assert "超出" in str(e.value) and "负值下限" in str(e.value)


# ───────── ⑦ 三来源全 None ⇒ 拒绝 (缺测不得当通过 ✓) ─────────
def test_all_none_is_rejected():
    with pytest.raises(ValueError) as e:
        resolve_anchor_q(q_nominal=None)
    assert "无法解析" in str(e.value)


# ───────── ⑧ 三处不一致检测: 用真实 deep4 现场数据核对 (只读 ✓) ─────────
def test_real_deep4_artifact_consistency_and_policy():
    """真实 artifact: 三处一致 + 负 drive 未声明 ⇒ 该控制模式应被拒绝 ✓"""
    import json

    import numpy as np
    root = Path(__file__).resolve().parents[1]
    pj = root / "output/s102_deep4/stage1/stage1_grasp_recipe.json"
    pn = root / "output/s102_deep4/stage1/best_grasp.npz"
    if not (pj.exists() and pn.exists()):
        pytest.skip("deep4 artifact 不存在")
    J = json.loads(pj.read_text())
    with np.load(pn, allow_pickle=True) as z:
        outer = float(z["gripper_qpos"])
        inner = float(json.loads(str(z["stage1_recipe_json"]))["q_grasp"])
    assert float(J["q_grasp"]) == outer == inner, "三处必须一致 ✓"
    aqc = np.asarray(J["actual_q_commands"], dtype=float).ravel()
    cc, sc = int(J["close_controls"]), int(J["settle_controls"])
    settle_end = float(aqc[cc + sc - 1])
    assert settle_end > 0.0, "settle 末命令应为正 ✓ (anchor 取此 ✓)"
    assert float(aqc.min()) < 0.0, "该 artifact 确实含负 q ⇒ 越界 ✓"
    with pytest.raises(ValueError):
        resolve_anchor_q(q_nominal=float(J["q_grasp"]), q_settle_end=None,
                         allow_negative=False)   # 未声明 ⇒ 必须拒绝 ✓
    q, _ = resolve_anchor_q(q_nominal=float(J["q_grasp"]), q_settle_end=settle_end)
    assert q == pytest.approx(settle_end), "有 settle 末时 anchor 必须取它 ✓"


# ───────── ⑨ f0 反例: **非对称 q** (超限 ⇒ 拒绝; 不得按平均静默处理 ✗) ─────────
def test_asymmetric_q_over_limit_rejected():
    with pytest.raises(ValueError) as e:
        resolve_anchor_q(q_nominal=0.01, q_settle_end=0.01,
                         q_per_finger=(0.013, 0.007), asym_max_mm=1.0)   # |Δ|=6mm ✗
    assert "非对称 q 超限" in str(e.value)


def test_symmetric_q_passes():
    q, info = resolve_anchor_q(q_nominal=0.01, q_settle_end=0.01,
                               q_per_finger=(0.01, 0.0105), asym_max_mm=1.0)
    assert q == pytest.approx(0.01)
    assert info["asym_mm"] == pytest.approx(0.5) and info["asym_max_mm"] == 1.0


# ───────── ⑩ f0 反例: **错误 episode** (绑定不在 passed 列表 ⇒ 拒绝) ─────────
def test_wrong_episode_binding_rejected():
    from stage1_recipe import assert_verification_binding
    vr = {"n_pass": 3, "n_repeat": 3,
          "runs": [{"episode_id": "s1ep-000128", "passed": True},
                   {"episode_id": "s1ep-000129", "passed": True},
                   {"episode_id": "s1ep-000130", "passed": True}],
          "selected_episode": "s1ep-000124"}      # ← 不在 passed 列表 ✗
    ok, errs = assert_verification_binding(vr)
    assert ok is False and errs, "错误 episode 必须被拒绝 ✓"
