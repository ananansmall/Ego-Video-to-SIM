"""T08-B 反例: 同入口 spy + 场景通道合同 (不跑物理 ✓)

md ✓: "Stage1/2 调用同一 tick、每周期只下发一次、步进一次";
  "逐 body/shape 核对 目标—双指/目标—桌面/双指—桌面 通道, 其他物体隔离";
  "**不能用全局开关把 目标—桌面 也关掉**";
  "故意修改一个物理/目标字段必须**在 step 前阻断**" ✓
"""
import inspect

import pytest

import physics_env as pe


def setup_function(_f):
    pe.reset_step_entries()
    pe.clear_scene_contract()


# ── ① 同一入口 / 同签名 ──────────────────────────────────────────────────────
def test_same_runner_ok_when_signature_matches():
    pe.record_step_entry("stage1", 16, 1 / 480.0)
    pe.record_step_entry("stage2", 16, 1 / 480.0)
    ok, det = pe.verify_same_runner()
    assert ok, det


def test_same_runner_rejects_missing_tag_or_divergent_sig():
    pe.record_step_entry("stage1", 16, 1 / 480.0)
    ok, det = pe.verify_same_runner()
    assert not ok and "缺少调用方 tag" in det
    pe.reset_step_entries()
    pe.record_step_entry("stage1", 16, 1 / 480.0)
    pe.record_step_entry("stage2", 8, 1 / 240.0)      # 频率分叉 ✗
    ok2, det2 = pe.verify_same_runner()
    assert not ok2 and "签名不一致" in det2
    pe.reset_step_entries()
    assert pe.verify_same_runner()[0] is False        # 无记录 ✗


# ── ② 场景通道合同 ────────────────────────────────────────────────────────────
_REQ = {"target_fingers": True, "target_table": True, "fingers_table": True}
_ACT_OK = {"target_fingers": True, "target_table": True, "fingers_table": True,
           "others": False}


def test_channels_ok():
    ok, diffs = pe.verify_scene_channels(_REQ, _ACT_OK)
    assert ok, diffs


def test_target_table_must_not_be_disabled_by_global_switch():
    """★ md 点名: **不得**用全局开关把 目标—桌面 一起关掉 ✗"""
    bad = dict(_ACT_OK, target_table=False)
    ok, diffs = pe.verify_scene_channels(_REQ, bad)
    assert not ok and any("target_table" in d for d in diffs)


@pytest.mark.parametrize("ch", ["target_fingers", "fingers_table"])
def test_other_required_channels_must_be_on(ch):
    ok, diffs = pe.verify_scene_channels(_REQ, dict(_ACT_OK, **{ch: False}))
    assert not ok and any(ch in d for d in diffs)


def test_others_must_be_isolated_off():
    ok, diffs = pe.verify_scene_channels(_REQ, dict(_ACT_OK, others=True))
    assert not ok and any("others" in d for d in diffs)


def test_missing_actual_is_not_treated_as_pass():
    ok, diffs = pe.verify_scene_channels(_REQ, {"target_fingers": True})
    assert not ok, "缺实际值必须判**不可信 ⇒ 不通过** ✗"


# ── ③ step 前阻断 ────────────────────────────────────────────────────────────
def test_step_gate_blocks_before_stepping():
    pe.register_scene_contract(_REQ, tag="stage2")
    assert pe.assert_scene_contract(_ACT_OK) is True          # 合规 ⇒ 放行 ✓
    with pytest.raises(RuntimeError, match="SCENE_CONTRACT_VIOLATION"):
        pe.assert_scene_contract(dict(_ACT_OK, target_table=False))   # 关掉 ✗ ⇒ 阻断 ✓


def test_unregistered_contract_is_noop():
    assert pe.assert_scene_contract({"target_table": False}) is True


def test_gate_hook_sits_before_any_scene_step_in_physics_step():
    """★ 结构证明: `assert_scene_contract()` 必须在任何 `scene.step(` **之前** ✓"""
    src = inspect.getsource(pe.physics_step)
    _lines = src.splitlines()
    _gate = [i for i, l in enumerate(_lines) if "assert_scene_contract()" in l]
    _steps = [i for i, l in enumerate(_lines) if l.strip().startswith("scene.step(")]  # 只认代码行 ✓ (排除 docstring 里的字面量 ✗)
    assert _gate, "physics_step 内必须有场景合同门 ✗"
    assert _steps, "physics_step 内应有 scene.step ✗"
    assert _gate[0] < _steps[0], "合同门必须在 step **之前** (否则无法阻断 ✗)"


def test_records_entry_each_call():
    pe.reset_step_entries()
    pe.record_step_entry("stage1", 16, 1 / 480.0, extra={"q": 0.01})
    ent = pe.get_step_entries()
    assert len(ent) == 1 and ent[0]["tag"] == "stage1" and ent[0]["decimation"] == 16
