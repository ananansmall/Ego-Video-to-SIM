"""任务01 测试 —— 切断隐式搜索 + 硬预算 (零物理 ✓, spy 只隔离外部资源 ✓)

md 位置: §4 任务 01；§7.3 第一批应证明 ✓:
  · **新 output 目录不会触发 CEM** ✓
  · **缺 Stage1 输入会明确停** ✓
  · **两套 env 不会暗互相覆盖** ✓
  · **第 N+1 次物理在 step 前拒绝** ✓
  (UNKNOWN 通道那条属任务 02, 见另一处测试)

本文件全部零物理 ✓: 只调用生产模块 `pipeline_budget` 的分流/预算/配置解析,
物理入口用"被调用就失败"的 spy 代表 (md §7.2-1: mock 只隔离求解器和外部资源 ✓)。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline_budget import (
    ACTIONS,
    DEFAULT_ACTION,
    SEARCH_BUDGET_DEFAULT,
    BudgetExhausted,
    EnvConflictError,
    EpisodeBudget,
    Stage1InputError,
    format_effective_config,
    resolve_effective_config,
    resolve_stage1_action,
)

# --------------------------------------------------------------------------- #
# 构造临时 artifact (不碰任何现场目录 ✓)
# --------------------------------------------------------------------------- #
def _recipe(n_pass=3, close_controls=16, settle_controls=10):
    q = [0.03 - 0.001 * i for i in range(close_controls)] + [0.001] * settle_controls
    return {
        # ★ 生产入口合同 (`stage1_recipe` 的 loader 检查) 要求这些字段齐 ✓
        "schema_version": "stage1_recipe/v2",
        "revision": "stage1_recipe/v2",
        "q_sequence_source": "actual_executed",
        "target_object": "glb_4",
        "q_open": 0.03, "q_grasp": 0.001,
        "close_controls": close_controls, "settle_controls": settle_controls,
        "close_profile": "cubic_smoothstep",
        "close_ctrl_q": q[:close_controls],
        "close_ctrl_t": [i / 30.0 for i in range(close_controls)],
        "q_cmd": q, "phase": ["close"] * close_controls + ["settle"] * settle_controls,
        "time_s": [i / 30.0 for i in range(len(q))],
        "control_index": list(range(len(q))),
        # 生产合同还要求"完整记录"非空 ✓ (v2 必须有 T_W_G_cmd / 实际命令数组)
        "T_W_G_cmd": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]] * len(q),
        "actual_q_commands": q,
        "actual_control_times": [i / 30.0 for i in range(len(q))],
        "actual_base_commands": [[0.0, 0.0, 0.0]] * len(q),
        "optimized_T_W_G": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        "phase_boundaries": {"close": [[0, close_controls]],
                             "settle": [[close_controls, close_controls + settle_controls]],
                             "n": len(q)},
        "source_frame": 65, "task_grasp_frame": 50,
        "physics_hash": "p" * 32, "controller_hash": "c" * 32, "criteria_hash": "k" * 32,
        "scene_digest_hash": "d" * 32,
        "export_valid": True,
        "verification_results": {"n_pass": n_pass, "n_repeat": 3,
                                 "runs": [{"episode_id": "s1ep-t", "passed": n_pass >= 2}]},
        "winner_binding": {"candidate_id": "c-t", "episode_id": "s1ep-t",
                           "record_present": True, "record_frames": len(q)},
    }


def _make_artifact(path: Path, *, n_pass=3, with_recipe=True, close_controls=16):
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"pos": np.zeros(3), "R": np.eye(3), "gripper_qpos": np.float64(0.001),
          "verification": np.array({"n_pass": n_pass, "n_repeat": 3}, dtype=object)}
    if with_recipe:
        kw["stage1_recipe_json"] = np.array(
            json.dumps(_recipe(close_controls=close_controls), default=str))
    np.savez(str(path), **kw)
    return path


# --------------------------------------------------------------------------- #
# 1) 默认 reuse + 缺输入明确停 (不回退搜索)
# --------------------------------------------------------------------------- #
def test_default_action_is_reuse_not_search():
    """md §01-1 ✓: 默认必须 reuse (不是 search ✗)"""
    assert DEFAULT_ACTION == "reuse"
    assert set(ACTIONS) == {"reuse", "verify", "search", "legacy"}


def test_new_output_dir_without_artifact_stops_with_specific_error(tmp_path, monkeypatch):
    """★ md §7.3 ✓: '新 output 目录不会触发 CEM' + '缺 Stage1 输入会明确停'"""
    monkeypatch.delenv("_S1_ARTIFACT", raising=False)
    monkeypatch.delenv("_S1_ARTIFACT_DIR", raising=False)
    out = tmp_path / "brand_new_run"          # 全新目录, 没有任何历史产物

    with pytest.raises(Stage1InputError) as ei:
        resolve_stage1_action(None, None, out)          # action 缺省 ⇒ reuse ✓
    assert ei.value.code == "STAGE1_ARTIFACT_MISSING"
    assert "不回退搜索" in str(ei.value)
    # ★ 关键 ✓: 全程**没有**任何搜索调用被登记 (零物理 + 零搜索 ✓)
    bud = EpisodeBudget(out / "budget_ledger.json", task="stage1")
    assert bud.search_calls() == 0
    assert bud.budget == 0                              # 修复路线搜索预算 = 0 ✓


def test_reuse_accepts_a_verified_artifact_and_reports_identity(tmp_path):
    art = _make_artifact(tmp_path / "out/stage1/best_grasp.npz", n_pass=3)
    plan = resolve_stage1_action(None, str(art), tmp_path / "out")
    assert plan.action == "reuse"
    assert plan.n_pass == 3
    assert plan.candidate_id == "c-t"
    assert plan.source_frame == 65
    s = plan.summary()
    assert "effective action=reuse" in s and "n_pass=3" in s   # 启动即打印 ✓


def test_reuse_rejects_unverified_or_stale_artifact(tmp_path):
    bad1 = _make_artifact(tmp_path / "a/stage1/best_grasp.npz", n_pass=1)
    with pytest.raises(Stage1InputError) as e1:
        resolve_stage1_action(None, str(bad1), tmp_path / "a")
    # ★ 两道门任一生效都算"明确拒绝" ✓:
    #   · 生产 loader 的入口合同 (md: 磁盘 loader 默认拒绝 n_pass<2 ✓) ⇒ ARTIFACT_UNREADABLE
    #   · 本模块自身的 n_pass<2 检查 ⇒ NOT_VERIFIED
    #   **关键是不回退搜索** ✓ (下面断言)
    assert e1.value.code in ("STAGE1_NOT_VERIFIED", "STAGE1_ARTIFACT_UNREADABLE")
    assert "不回退搜索" in str(e1.value)

    bad2 = _make_artifact(tmp_path / "b/stage1/best_grasp.npz", with_recipe=False)
    with pytest.raises(Stage1InputError) as e2:
        resolve_stage1_action(None, str(bad2), tmp_path / "b")
    # 旧产物 (无内嵌 recipe): 生产 loader 直接判"合同非法" ⇒ UNREADABLE;
    # 若 loader 放宽到返回 None, 本模块会判 STALE_STAGE1_ARTIFACT ✓ 两者都是**明确拒绝** ✓
    assert e2.value.code in ("STALE_STAGE1_ARTIFACT", "STAGE1_ARTIFACT_UNREADABLE")
    assert "不回退搜索" in str(e2.value)


def test_artifact_path_is_independent_of_output_dir(tmp_path, monkeypatch):
    """md §01-1 ✓: 产物路径**独立于** output-dir ⇒ 新目录也能复用已冻结成功产物"""
    frozen = _make_artifact(tmp_path / "frozen_stage1/best_grasp.npz", n_pass=3)
    monkeypatch.setenv("_S1_ARTIFACT", str(frozen))
    new_out = tmp_path / "fresh_out"
    new_out.mkdir()
    plan = resolve_stage1_action(None, None, new_out)
    assert plan.artifact == frozen
    assert plan.n_pass == 3


# --------------------------------------------------------------------------- #
# 2) 只有显式 search 才允许搜索, 且受 48 次预算
# --------------------------------------------------------------------------- #
def test_only_explicit_search_is_allowed_and_it_is_capped(tmp_path):
    plan = resolve_stage1_action("search", None, tmp_path / "out")
    assert plan.action == "search"
    assert plan.criterion == "search(explicit)"
    bud = EpisodeBudget(tmp_path / "out/budget_ledger.json", task="stage1", action="search")
    assert bud.budget == SEARCH_BUDGET_DEFAULT == 48

    # legacy 是诊断路径 ⇒ **不获得独立预算** (仍是 0 ✗)
    plan2 = resolve_stage1_action("legacy", None, tmp_path / "out2")
    assert plan2.criterion == "legacy_diagnostic"
    bud2 = EpisodeBudget(tmp_path / "out2/budget_ledger.json", task="stage1", action="legacy")
    assert bud2.budget == 0


def test_bad_action_is_rejected():
    with pytest.raises(Stage1InputError) as ei:
        resolve_stage1_action("autosearch", None, "/tmp/none")
    assert ei.value.code == "BAD_STAGE1_ACTION"


# --------------------------------------------------------------------------- #
# 3) 预算: 第 N+1 次在 step 前拒绝; 异常也消耗; 重启恢复账本
# --------------------------------------------------------------------------- #
def test_budget_rejects_n_plus_one_before_step(tmp_path):
    """★ md §7.3 ✓: '第 N+1 次物理在 step 前拒绝' (预占发生在进入 with 之前 ✓)"""
    led = tmp_path / "budget_ledger.json"
    bud = EpisodeBudget(led, task="stage2", physical_budget=2)
    steps = []

    for i in range(2):
        with bud.episode(purpose=f"stage2_repeat#{i}"):
            steps.append(i)                       # 代表 reset/step 已发生
    assert steps == [0, 1]
    assert bud.used() == 2 and bud.remaining() == 0

    with pytest.raises(BudgetExhausted) as ei:
        with bud.episode(purpose="stage2_repeat#3"):
            steps.append(99)                      # ← 绝不能执行到这一步 ✗
    assert "BUDGET_EXHAUSTED" in str(ei.value)
    assert steps == [0, 1], "第 3 次竟然执行到了 step ✗ (md: 必须在 step 前拒绝)"
    assert bud.used() == 2


def test_exception_still_consumes_budget(tmp_path):
    """md §01-4 ✓: '异常也消耗已占次数'"""
    bud = EpisodeBudget(tmp_path / "l.json", task="stage2", physical_budget=3)
    with pytest.raises(RuntimeError):
        with bud.episode(purpose="diagnose"):
            raise RuntimeError("boom")
    assert bud.used() == 1
    assert bud._data["entries"][-1]["status"] == "failed"


def test_ledger_survives_restart_and_counts_search_calls(tmp_path):
    """md §01-4 ✓: 跨进程/重启继续同一任务 ⇒ 读**已用账本**, 不允许新目录清零 ✓"""
    led = tmp_path / "budget_ledger.json"
    b1 = EpisodeBudget(led, task="stage1", physical_budget=2)
    with b1.episode(purpose="stage1_search_iter1"):
        pass

    b2 = EpisodeBudget(led, task="stage1", physical_budget=2)   # 模拟"重启"
    assert b2.used() == 1 and b2.remaining() == 1
    assert b2.search_calls() == 1
    with b2.episode(purpose="stage1_search_iter2"):
        pass
    with pytest.raises(BudgetExhausted):
        with b2.episode(purpose="stage1_search_iter3"):
            pass
    # 第三次重启: 账本仍记得已用 2 ✓ ⇒ 不能通过换进程清零 ✗
    b3 = EpisodeBudget(led, task="stage1", physical_budget=2)
    assert b3.used() == 2 and b3.remaining() == 0


class _CemSpy:
    """'被调用就失败'的搜索 spy ✓ (md §7.2-1: 生产入口必须不触发它)"""

    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        raise AssertionError("reuse 路径**绝不能**调到 CEM/搜索 ✗ (md §01-3)")


def test_reuse_path_never_calls_the_search_spy(tmp_path):
    art = _make_artifact(tmp_path / "out/stage1/best_grasp.npz", n_pass=3)
    spy = _CemSpy()
    plan = resolve_stage1_action(None, str(art), tmp_path / "out")
    if plan.action == "search":                 # 只有显式 search 才允许
        spy()
    assert spy.calls == 0
    bud = EpisodeBudget(tmp_path / "out/budget_ledger.json", task="stage1", action=plan.action)
    assert bud.search_calls() == 0
    assert bud.budget == 0


# --------------------------------------------------------------------------- #
# 4) 参数只解析一次 + 两套 env 不暗互相覆盖
# --------------------------------------------------------------------------- #
def test_effective_config_cli_wins_and_conflicts_are_rejected(monkeypatch):
    """★ md §7.3 ✓: '两套 env 不会暗互相覆盖'"""
    monkeypatch.setenv("_S2_MODE", "strict_replay")
    cfg = {"_S2_MODE": "feedback"}
    # config 与 env 值不同, 且 CLI 没给 ⇒ **拒绝** (不能后读悄悄覆盖 ✗)
    with pytest.raises(EnvConflictError):
        resolve_effective_config({}, cfg, {"_S2_MODE": "feedback"},
                                 protected_env=("_S2_MODE",))
    # CLI 显式给出 ⇒ 允许覆盖 (CLI > 配置 > 默认 ✓)
    eff = resolve_effective_config({"_S2_MODE": "command_transfer"}, cfg,
                                   {"_S2_MODE": "feedback"},
                                   protected_env=("_S2_MODE",))
    assert eff["_S2_MODE"] == "command_transfer"
    assert "effective" in format_effective_config(eff)


def test_effective_config_defaults_when_nothing_set(monkeypatch):
    monkeypatch.delenv("_S2_MODE", raising=False)
    eff = resolve_effective_config({}, {}, {"_S2_MODE": "command_transfer"},
                                   protected_env=("_S2_MODE",))
    assert eff["_S2_MODE"] == "command_transfer"
