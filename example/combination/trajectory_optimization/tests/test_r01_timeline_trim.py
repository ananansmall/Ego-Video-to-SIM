"""R01 回归 —— 时间线截断与数组完整性 (零物理 ✓)

md 位置: docs/2026-09-23-物理仿真控制与奖励联合设计-推荐实施方案.md §H.1 R01

md 原话 ✓:
  1. "正式搬运**禁止现有 q-only 尾裁剪路径**; 若保留诊断开关, 启用时必须执行合法的
     **相位感知**逻辑或**显式拒绝**, 不能静默删 CARRY" ✓
  2. "从统一控制记录构造全部数组; 在输出/执行前检查 `source_frame/phase/T/q/dt/
     contact_policy/objective_weights/w_pose/safety_offset` **同长 N**, 全等则放行" ✓
  3. "回归上述生产反例: **恒定夹紧 q、基座持续移动**, trim 开/关均保留完整 F62 终点和
     22 点 CARRY (同一测试配置)" ✓

本文件 = 新增测试 ✓ (不改任何原有代码/原有测试 ✓)
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from stage1_recipe import (
    STAGE1_CLOSE_CONTROLS,
    STAGE1_SETTLE_CONTROLS,
    stage1_close_q_sequence,
)
from stage2_reconstruct_mixin import build_stage2_timeline

# 与 stage2 bundle 一致的 per-control 数组 (md R01-2 的清单 ✓)
PER_CTRL_KEYS = (
    "source_frame", "phase", "base_pos", "base_R", "gripper_q", "control_dt",
    "expected_contact_policy", "objective_weights", "w_pose", "safety_offset",
)


def _reconstruction():
    """★ 生产反例的核心 ✓: 基座**持续移动** (pos 随源帧变) 而夹爪 q 在搬运段**恒定**"""
    frames = list(range(38, 63))
    return {
        "frames": frames,
        "pos": {f: np.array([0.002 * f, 0.0, 0.10]) for f in frames},   # 持续移动 ✓
        "R": {f: np.eye(3) for f in frames},
        "gripper_qpos": {f: 0.05 if f < 50 else 0.01 for f in frames},
        "alpha_pose": {f: 1.0 if f >= 50 else 0.0 for f in frames},
        "safety_offset": {f: np.zeros(3) for f in frames},
        "grasp_anchor": {
            "source_frame": 50,
            "pos": np.array([0.10, 0.0, 0.10]),
            "R": np.eye(3),
            "gripper_qpos": 0.01,
            "q_start": 0.05,
        },
    }


def _recipe(q_open=0.05, q_grasp=0.01):
    import stage1_recipe as sr

    close = list(sr.stage1_close_q_sequence(q_open, q_grasp, STAGE1_CLOSE_CONTROLS))
    settle = [float(q_grasp)] * STAGE1_SETTLE_CONTROLS
    q = [float(x) for x in (close + settle)]
    ph = ["close"] * len(close) + ["settle"] * len(settle)
    return {
        "schema_version": sr.STAGE1_RECIPE_SCHEMA,
        "export_valid": True,
        "winner_binding": {"record_present": True, "record_frames": len(q),
                           "episode_id": "s1ep-r01", "candidate_id": "c-r01"},
        "time_s": [i / 30.0 for i in range(len(q))],
        "T_W_G_cmd": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * len(q),
        "q_cmd": q, "phase": ph, "control_index": list(range(len(q))),
        "phase_boundaries": sr.derive_phase_boundaries(ph),
        "close_profile": "cubic_smoothstep",
        "close_controls": len(close), "settle_controls": len(settle),
        "close_ctrl_q": close,
        "q_open": float(q_open), "q_grasp": float(q_grasp),
        "physics_hash": "p" * 16, "controller_hash": "c" * 16, "criteria_hash": "k" * 16,
        "scene_digest_hash": "d" * 16,
        "optimized_T_W_G": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "source_frame": 42, "task_grasp_frame": 50, "task_grasp_frame_default": 50,
        "verification_results": {"runs": [{"episode_id": "s1ep-r01"}], "n_pass": 3},
    }


def _build(**kwargs):
    return build_stage2_timeline(_reconstruction(), F50_IDX=50, recipe=_recipe(), **kwargs)


# --------------------------------------------------------------------------- #
# R01-1: q-only 尾裁剪不得删真实搬运
# --------------------------------------------------------------------------- #
def test_r01_trim_never_deletes_carry(monkeypatch, caplog):
    """★ 生产反例 ✓: 恒定夹紧 q + 基座持续移动 ⇒ **trim 开/关结果必须一致**

    旧病 (md) ✗: 22 个 CARRY 点被按 q 相同裁成 1 个, 终点 F62 退到 F51.5 ✗
    本测 ✓: 打开裁剪开关后, 若尾点属 CARRY ⇒ 必须**显式拒绝** (长度/终点/CARRY 数不变 ✓)
    """
    monkeypatch.delenv("_S2_TRIM_EXTRA_TAIL", raising=False)
    off = _build()

    monkeypatch.setenv("_S2_TRIM_EXTRA_TAIL", "1")
    with caplog.at_level(logging.WARNING):
        on = _build()

    n_off, n_on = int(len(off["source_frame"])), int(len(on["source_frame"]))
    carry_off = int(np.sum(np.asarray(off["phase"]) == "CARRY"))
    carry_on = int(np.sum(np.asarray(on["phase"]) == "CARRY"))
    end_off = float(off["source_frame"][-1])
    end_on = float(on["source_frame"][-1])

    print("\n[R01] trim 关: N=%d CARRY=%d 终点源帧=%.3f" % (n_off, carry_off, end_off))
    print("[R01] trim 开: N=%d CARRY=%d 终点源帧=%.3f" % (n_on, carry_on, end_on))

    assert n_on == n_off, "★ trim=1 竟然改了长度 ⇒ q-only 裁剪真的删了搬运 ✗ (md 禁止)"
    assert carry_on == carry_off, "★ CARRY 点数被 trim 改动 ✗ (md 禁止)"
    assert end_on == end_off, "★ 时间线终点被 trim 改动 ✗ (md: 必须保留完整终点)"
    assert carry_on > 1, "构造的反例里 CARRY 点太少, 反例不成立"

    # 拒绝路径应有明确日志 (不静默 ✓) —— 只有当确实检测到尾重复点时才要求
    txt = "\n".join(r.message for r in caplog.records)
    if "末尾重复点=" in txt:
        assert ("显式拒绝" in txt) or ("允许裁剪" in txt), \
            "检测到尾重复点却没给出**显式**结论 (拒绝/允许) ⇒ 属于静默路径 ✗"


def test_r01_tail_phase_is_carry_not_approach():
    """反例有效性自检 ✓: 本构造的时间线尾部确实落在**搬运**相位 (不是接近等待)"""
    b = _build()
    tail = [str(x) for x in b["phase"][-8:]]
    assert all(t == "CARRY" for t in tail), f"尾部相位应为 CARRY, 实际={tail}"
    # 且搬运段夹爪 q **恒定** (正是旧逻辑会误判"重复点"的条件 ✓)
    assert float(np.ptp(b["gripper_q"][-8:, 0])) == 0.0


# --------------------------------------------------------------------------- #
# R01-2: 数组完整性 (同长 N 才放行)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("trim", ["0", "1"])
def test_r01_all_per_control_arrays_share_the_same_length(monkeypatch, trim):
    """★ md R01-2 ✓: 10 个 per-control 数组必须**同长 N** (trim 开/关都要成立 ✓)"""
    monkeypatch.setenv("_S2_TRIM_EXTRA_TAIL", trim)
    b = _build()
    n = int(len(b["source_frame"]))
    lengths = {k: int(len(b[k])) for k in PER_CTRL_KEYS if k in b}
    print(f"\n[R01-2] trim={trim} N={n} lengths={lengths}")
    assert lengths, "bundle 里一个 per-control 数组都没有 ⇒ 测试失效"
    bad = {k: v for k, v in lengths.items() if v != n}
    assert not bad, f"数组长度不一致: {bad} (md: 全等才放行 ✗)"
    for k in PER_CTRL_KEYS:
        assert k in b, f"缺少 md 要求的数组: {k}"


def test_r01_objective_weights_second_axis_is_kept(monkeypatch):
    """objective_weights 是 (N, len(WEIGHT_NAMES)) ⇒ 第一维必须等于 N ✓"""
    from stage2_reconstruct_mixin import OBJECTIVE_WEIGHT_NAMES

    monkeypatch.setenv("_S2_TRIM_EXTRA_TAIL", "1")
    b = _build()
    w = np.asarray(b["objective_weights"])
    assert w.ndim == 2
    assert w.shape[0] == len(b["source_frame"])
    assert w.shape[1] == len(OBJECTIVE_WEIGHT_NAMES)
