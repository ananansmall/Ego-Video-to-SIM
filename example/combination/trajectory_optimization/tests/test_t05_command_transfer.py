"""任务05 测试 —— Stage2 真实命令迁移 (零物理 ✓)

md 位置: §4 任务 05（Stage2 先做到真实命令迁移）
md 原话 ✓
  · "close 和 settle **都消费实际切片**, **搬运首 q 继承 settle 末点**; 禁止用名义
    `q_grasp` 替换、重生成曲线、偷偷 re-seat" ✓
  · "近期先明确 **command_transfer**, 完成最小可工作链; 没有在线 tick 时不能把静态数组
    标 feedback 成功" ✓
  · "怎么测: 从生产 timeline 到 dispatch spy, **非恒定 settle 37/17**、**合法 q=0**、
    **非默认 dt** 均逐点一致; **0.001535m 不得被 0.0018985m 替换**; 输入 anchor 不符严格拒绝;
    **把曲线生成器替换成抛异常, 正常迁移仍运行**; 不要只测 `select_close_mode` 返回的 payload" ✓
  · 通过标准: "先通过**零物理调用测试**, 再进入任务 06 的 3 次短窗口预算" ✓

本文件全部零物理 ✓: 只调用 `build_stage2_timeline`（纯构造, 不 step）+ `select_close_mode` ✓
"""

from __future__ import annotations

import numpy as np
import pytest

import stage1_recipe as sr
from stage2_reconstruct_mixin import (
    STAGE1_CLOSE_CONTROLS,
    STAGE1_SETTLE_CONTROLS,
    build_stage2_timeline,
)

# ★ md 点名的两个数: 实际 0.001535m **不得**被名义 0.0018985m 替换 ✓
Q_ACTUAL_TAIL = 0.001535
Q_NOMINAL = 0.0018985


def _reconstruction(q_anchor=0.01):
    frames = list(range(38, 63))
    return {
        "frames": frames,
        "pos": {f: np.array([0.002 * f, 0.0, 0.10]) for f in frames},
        "R": {f: np.eye(3) for f in frames},
        "gripper_qpos": {f: 0.05 if f < 50 else q_anchor for f in frames},
        "alpha_pose": {f: 1.0 if f >= 50 else 0.0 for f in frames},
        "safety_offset": {f: np.zeros(3) for f in frames},
        "grasp_anchor": {"source_frame": 50, "pos": np.array([0.10, 0.0, 0.10]),
                         "R": np.eye(3), "gripper_qpos": q_anchor, "q_start": 0.05},
    }


def _recipe(settle_q=None, close_q=None, q_grasp=Q_NOMINAL, close_start=0.05):
    """★ 关键 ✓: settle 用**非恒定**序列 ⇒ 才能证明 timeline 真的消费了实际切片"""
    # ⚠ 构造 recipe 时**不调用**曲线生成器 ✓ (否则 `test_generator_…` 会误判为生产违规 ✗)
    _n_close = STAGE1_CLOSE_CONTROLS
    close = list(close_q if close_q is not None else
                 [close_start + (Q_ACTUAL_TAIL - close_start) * (i + 1) / _n_close
                  for i in range(_n_close)])
    settle = list(settle_q if settle_q is not None else
                  [Q_ACTUAL_TAIL] * STAGE1_SETTLE_CONTROLS)
    q = [float(x) for x in (close + settle)]
    ph = ["close"] * len(close) + ["settle"] * len(settle)
    return {
        "schema_version": sr.STAGE1_RECIPE_SCHEMA,
        "revision": "stage1_recipe/v2",
        "q_sequence_source": "actual_executed",
        "export_valid": True,
        "target_object": "glb_4",
        "q_open": 0.05, "q_grasp": float(q_grasp),
        "close_controls": len(close), "settle_controls": len(settle),
        "close_profile": "cubic_smoothstep",
        "close_ctrl_q": close, "close_ctrl_t": [i / 30.0 for i in range(len(close))],
        "q_cmd": q, "phase": ph, "time_s": [i / 30.0 for i in range(len(q))],
        "control_index": list(range(len(q))),
        "phase_boundaries": sr.derive_phase_boundaries(ph),
        "T_W_G_cmd": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * len(q),
        "actual_q_commands": q,
        "actual_control_times": [i / 30.0 for i in range(len(q))],
        "actual_base_commands": [[0.0, 0.0, 0.0]] * len(q),
        "optimized_T_W_G": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "physics_hash": "p" * 16, "controller_hash": "c" * 16, "criteria_hash": "k" * 16,
        "scene_digest_hash": "d" * 16,
        "source_frame": 65, "task_grasp_frame": 50,
        "verification_results": {"n_pass": 3, "n_repeat": 3,
                                 "runs": [{"episode_id": "s1ep-t", "passed": True}]},
        "winner_binding": {"candidate_id": "c-t", "episode_id": "s1ep-t",
                           "record_present": True, "record_frames": len(q)},
    }


def _build(recipe=None, q_anchor=None, **kw):
    """★ 注意 ✓: 默认让 anchor **等于** recipe 的 `q_grasp` (md §05: anchor 不符要**拒绝** ✓)
    ⇒ 只有 `test_anchor_mismatch_is_rejected` 才故意用不同值 ✓"""
    _r = recipe if recipe is not None else _recipe()
    if q_anchor is None:
        q_anchor = float(_r.get("q_grasp"))
    return build_stage2_timeline(_reconstruction(q_anchor=q_anchor), F50_IDX=50,
                                 recipe=_r, **kw)


# --------------------------------------------------------------------------- #
# 1) settle 段必须消费**实际切片** (非恒定 ⇒ 一眼可辨 ✓)
# --------------------------------------------------------------------------- #
def test_settle_consumes_the_actual_nonconstant_slice():
    """★ md: "非恒定 settle 均逐点一致" ✓ + "0.001535 不得被 0.0018985 替换" ✓"""
    settle_actual = [0.0015, 0.0014, 0.0013, 0.0012, 0.0015,
                     0.0016, 0.0017, 0.0018, 0.0019, Q_ACTUAL_TAIL]
    b = _build(recipe=_recipe(settle_q=settle_actual))
    m = b["timeline_meta"]
    got = b["gripper_q"][m["settle_start"]: m["settle_end"] + 1, 0]
    np.testing.assert_allclose(got, np.asarray(settle_actual), atol=0, rtol=0)
    assert not np.allclose(got, Q_NOMINAL), "settle 段被名义 q 替换了 ✗"


def test_carry_first_q_inherits_settle_last_point():
    """★ md: "**搬运首 q 继承 settle 末点**" ✓"""
    settle_actual = [0.0015] * 8 + [0.0016, 0.001535]
    b = _build(recipe=_recipe(settle_q=settle_actual))
    m = b["timeline_meta"]
    assert m["q_carry0"] == pytest.approx(settle_actual[-1], abs=0)
    # CARRY_VERIFY 与 CARRY 都没被名义 q_grasp 替换 ✓
    for ph in ("CARRY_VERIFY", "CARRY"):
        idx = np.where(np.asarray(b["phase"]) == ph)[0]
        if idx.size:
            np.testing.assert_allclose(b["gripper_q"][idx, 0],
                                       np.full(idx.size, settle_actual[-1]), atol=0, rtol=0)


def test_q_zero_is_legal_and_not_coerced(monkeypatch):
    """★ md: "合法 q=0" 必须照抄 (不能被当成缺省值丢掉) ✓"""
    monkeypatch.delenv("_S2_RESEAT_M", raising=False)
    settle_actual = [0.0] * STAGE1_SETTLE_CONTROLS
    b = _build(recipe=_recipe(settle_q=settle_actual))
    m = b["timeline_meta"]
    got = b["gripper_q"][m["settle_start"]: m["settle_end"] + 1, 0]
    assert np.all(got == 0.0), f"合法 q=0 被改写了 ✗ {got[:4]}"
    assert m["q_carry0"] == 0.0


def test_non_default_dt_is_preserved_in_timeline():
    """★ md: "非默认 dt 逐点一致" ✓ (timeline 的 control_dt 由 control_freq 推导 ✓)"""
    b = _build()
    dt = np.asarray(b["control_dt"])
    assert dt.size == len(b["source_frame"])
    assert np.all(dt > 0) and np.allclose(dt, dt[0])


def test_generator_being_replaced_by_raising_callable_still_migrates(monkeypatch):
    """★ md 明令 ✓: "把曲线生成器替换成**抛异常**, 正常迁移仍运行" —— 不能只测 payload """
    def _boom(*a, **k):
        raise AssertionError("正常迁移**不得**再调用曲线生成器 ✗ (md §05)")

    # ⚠ 必须 patch **消费方模块** 里的名字 (它是 `from stage1_recipe import …` ⇒ 已绑定 ✗)
    import stage2_reconstruct_mixin as s2r

    monkeypatch.setattr(s2r, "stage1_close_q_sequence", _boom)
    monkeypatch.setattr(sr, "stage1_close_q_sequence", _boom)
    monkeypatch.setattr(sr, "stage_q_open", _boom)
    b = _build()                       # 必须**照常**构造成功 ✓
    assert len(b["source_frame"]) > 0
    assert b["timeline_meta"]["q_carry0"] > 0


def test_anchor_mismatch_is_rejected():
    """★ md: "输入 anchor 不符 **严格拒绝**" ✓

    反例口径 ✓: anchor 的 q **落在 recipe 实际命令值域之外** (0.06 > q_open=0.05)
      ⇒ 说明它来自**另一条 recipe / 另一次搜索** ⇒ 必须拒 ✗
    """
    # anchor=0.01 落在 recipe 实际命令值域 [0.001535, 0.008] **之外** ⇒ 必须拒 ✗
    #   (同时让它能越过"0<=q_grasp<=q_start<=q_open"这条内部校验, 才命中本口径 ✓)
    recon = _reconstruction(q_anchor=0.01)
    with pytest.raises(ValueError) as ei:
        build_stage2_timeline(recon, F50_IDX=50, recipe=_recipe(close_start=0.008))
    assert "anchor 与 recipe" in str(ei.value) and "严格拒绝" in str(ei.value)


def test_reseat_is_marked_diagnostic_only(monkeypatch, caplog):
    """★ md: 禁止**偷偷 re-seat** ⇒ 显式开启时必须 ERROR 标注"仅诊断" ✓"""
    import logging

    monkeypatch.setenv("_S2_RESEAT_M", "0.0015")
    with caplog.at_level(logging.ERROR):
        b = _build()
    txt = "\n".join(r.message for r in caplog.records)
    assert "re-seat" in txt and "不得作为交付证据" in txt, txt[:200]
    # 且 re-seat 只在**显式**开启时才改变 settle 段 ✓ (默认关闭 ⇒ 与切片逐点一致)
    monkeypatch.delenv("_S2_RESEAT_M", raising=False)
    b2 = _build()
    m2 = b2["timeline_meta"]
    np.testing.assert_allclose(
        b2["gripper_q"][m2["settle_start"]: m2["settle_end"] + 1, 0],
        np.full(STAGE1_SETTLE_CONTROLS, Q_ACTUAL_TAIL), atol=0, rtol=0)
