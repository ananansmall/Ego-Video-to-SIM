"""T07e 反例/正例: 走**真实导出函数**出 v2 合同 recipe (不跑物理 ✓)

背景 (实测 s20/s26): winner 记录虽有 350 帧, 但导出里
  ① 引用了未定义的 `side` ⇒ S1-D 块整体抛错
  ② 合同关键字段(别名/winner_binding/export_valid)与该 try 同块 ⇒ 被连带清零
  ⇒ 合同必拒 (`T_W_G_cmd=0 vs actual_q_commands=350` ✗)
本测试用 `object.__new__` 造轻量实例直调 `_export_stage1_recipe` ⇒ 秒级锁住该缺陷 ✓
"""
import numpy as np
import pytest

import stage1_recipe as sr
from stage1_grasp_mixin import Stage1Mixin

_N = 350


def _trace(n=_N):
    ph = (["travel"] * 30 + ["close"] * 130 + ["settle"] * 15
          + ["lift"] * (n - 175))
    q = [0.030 - 0.00006 * i for i in range(n)]
    return {
        "schema_version": "s1_exec_trace/v2",
        "episode_id": "s1ep-000042", "candidate_id": "s1cand-test",
        "control_index": list(range(n)), "t_s": [i / 30.0 for i in range(n)],
        "T_W_G": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] for _ in range(n)],
        "q_cmd": q, "phase": ph, "source_phase": [float(i) for i in range(n)],
        "q_act": [[q[0], q[0]] for _ in range(n)],
        "obj_z": [0.02 for _ in range(n)], "valid": [True] * n,
        "base_pos": [[0.0, 0.0, 0.15] for _ in range(n)],
        "close_controls": 130, "settle_controls": 15, "n_frames": n,
    }


def _instance():
    obj = object.__new__(Stage1Mixin)
    obj.obj_info = {"glb_4": {"bbox_size": [0.044, 0.055, 0.039]}}
    obj._s1_episode_records = {"s1ep-000042": {"record": _trace(),
                                               "candidate_id": "s1cand-test",
                                               "source_frame": 42}}
    obj.scene = None
    obj.compute_gripper_qpos = lambda _bs: 0.005
    obj._ground_z = 0.0
    obj.obj_actors = []
    obj._obj_initial_poses = {}
    return obj


def _best(**over):
    b = {"pos": [0.1, 0.2, 0.15],
         "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
         "gripper_qpos": 0.005,
         "episode_id": "s1ep-000042", "candidate_id": "s1cand-test",
         "source_frame": 42,
         "verification": {"n_pass": 3, "n_repeat": 3,
                          "runs": [{"episode_id": "s1ep-000042", "passed": True}]},
         "immutable_record": _trace()}
    b.update(over)
    return b


def test_export_produces_contract_passing_v2_recipe():
    recipe = _instance()._export_stage1_recipe("glb_4", _best())
    assert recipe is not None, "导出必须产出 recipe ✗"
    assert recipe.get("schema_version") == sr.STAGE1_RECIPE_SCHEMA
    assert recipe.get("export_valid") is True, (
        "合同必须通过, 实测 errors=%s" % (recipe.get("contract_errors"),))
    ok, errs = sr.validate_recipe_contract(recipe)
    assert ok, errs


def test_export_fills_identity_and_frames_not_matrix():
    recipe = _instance()._export_stage1_recipe("glb_4", _best())
    assert isinstance(recipe["source_frame"], int) and recipe["source_frame"] == 42
    assert recipe["task_grasp_frame"] == 50 and recipe["task_grasp_frame_default"] == 50
    assert recipe["winner_binding"]["record_present"] is True
    assert recipe["winner_binding"]["record_frames"] == _N
    assert recipe["winner_binding"]["episode_id"] == "s1ep-000042"
    assert recipe["q_sequence_source"] == "actual_executed"


def test_export_lengths_and_phase_boundaries():
    recipe = _instance()._export_stage1_recipe("glb_4", _best())
    for k in ("time_s", "T_W_G_cmd", "phase", "control_index"):
        assert len(recipe.get(k) or []) == _N, "%s 长度应为 %d ✗" % (k, _N)
    pb = recipe["phase_boundaries"]
    segs = sorted(sum((v for kk, v in pb.items() if kk != "n"), []))
    assert segs[0][0] == 0 and segs[-1][1] == _N
    assert all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))
    assert len(pb.get("close", [])) == 1 and pb["close"][0] == [30, 160]


def test_export_recovers_record_by_episode_id_from_registry():
    """best_output 上没有记录时, 必须能按**自己的 episode_id** 从登记表取回 ✓"""
    b = _best(immutable_record=None)
    recipe = _instance()._export_stage1_recipe("glb_4", b)
    assert recipe.get("export_valid") is True
    assert recipe["winner_binding"]["record_frames"] == _N


def test_export_without_any_record_is_invalid_not_fabricated():
    """拿不到记录 ⇒ export_valid=False, **不得**用最后缓存/名义曲线顶替 ✗"""
    b = _best(immutable_record=None, episode_id="s1ep-does-not-exist")
    recipe = _instance()._export_stage1_recipe("glb_4", b)
    assert recipe.get("export_valid") is False
    assert recipe.get("contract_errors")


def test_export_recovers_source_frame_from_registry():
    """★ T07e3: 强绑定通道没带 source_frame 时, 必须从**登记表**取回整数帧号 ✓"""
    b = _best(source_frame=None)          # 记录在, 但帧号缺
    recipe = _instance()._export_stage1_recipe("glb_4", b)
    assert isinstance(recipe.get("source_frame"), int), (
        "source_frame 必须兜底成整数 ✗ (实测 %r)" % (recipe.get("source_frame"),))
    assert recipe["source_frame"] == 42
    ok, errs = sr.validate_recipe_contract(recipe)
    assert ok, errs


def test_export_sets_md_q_cmd_alias_and_contract_passes():
    """★ T07e3: md 字段名 `q_cmd` 别名必须与 actual_q_commands 同源同长 ✓"""
    recipe = _instance()._export_stage1_recipe("glb_4", _best())
    assert len(recipe.get("q_cmd") or []) == _N
    assert recipe.get("q_cmd") == recipe.get("actual_q_commands")
    ok, errs = sr.validate_recipe_contract(recipe)
    assert ok, errs
    import stage1_recipe as _sr
    ok_e, errs_e = _sr.validate_stage2_entry(recipe, mode="strict_replay")
    assert ok_e, "Stage2 入口必须放行 ✓ (实测 %s)" % (errs_e,)


def test_export_surfaces_verification_runs():
    """★ T07e4: md 要求 verification_results 含**全部重复 run 的 ID/门值/结果/失败原因** ✓"""
    recipe = _instance()._export_stage1_recipe("glb_4", _best())
    vr = recipe.get("verification_results") or {}
    assert len(vr.get("runs") or []) >= 1, "verification_results.runs 不能为空 ✗"
    assert vr.get("n_pass") is not None, "n_pass 必须写入 ✗"
    assert vr.get("selected_episode") == "s1ep-000042", "必须标明被选成功 episode ✗"
    import stage1_recipe as _sr
    ok_e, errs_e = _sr.validate_stage2_entry(recipe, mode="strict_replay")
    assert ok_e, "含 runs/n_pass 的 recipe 必须过 Stage2 入口 ✓ (%s)" % (errs_e,)
