"""T07.2 反例: 数据合同 (不跑物理 ✓)

md T07.1 口径 ✓:
  · `source_frame` = **整数帧号**(如 42), `task_grasp_frame` = **默认 50**, 且**不随闭合帧数变化**
  · 同长 N>0; 时标严格递增; control_index 连续; 相位边界**半开区间 [start,end) 不重叠不遗漏**
  · 空数组/NaN/缺旋转/时标重复/跨 run 拼接 ⇒ **ARTIFACT_INVALID**(合同不过 ✓)
"""
import copy

from stage1_recipe import (STAGE1_RECIPE_SCHEMA, derive_phase_boundaries,
                           validate_recipe_contract)


def _phases(n_travel=2, n_close=3, n_settle=2, n_lift=2):
    return (["travel"] * n_travel + ["close"] * n_close
            + ["settle"] * n_settle + ["lift"] * n_lift)


def _recipe(phases=None, close_ctrl=16, settle_ctrl=10):
    ph = phases if phases is not None else _phases()
    n = len(ph)
    return {
        "schema_version": STAGE1_RECIPE_SCHEMA,
        "source_frame": 42,
        "task_grasp_frame": 50,
        "task_grasp_frame_default": 50,
        "optimized_T_W_G": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "actual_q_commands": [0.030 - 0.002 * i for i in range(n)],
        "actual_control_times": [i / 30.0 for i in range(n)],
        "T_W_G_cmd": [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]] for _ in range(n)],
        "phase": list(ph),
        "control_index": list(range(n)),
        "phase_boundaries": derive_phase_boundaries(ph),
        "close_controls": int(close_ctrl),
        "settle_controls": int(settle_ctrl),
        "physics_hash": "p" * 16,
        "controller_hash": "c" * 16,
        "criteria_hash": "k" * 16,
    }


def test_valid_recipe_passes():
    ok, errs = validate_recipe_contract(_recipe())
    assert ok, "合法 recipe 应当通过, 实测错误: %s" % errs


def test_frames_invariant_under_more_close_points():
    """md: 多插入几十个闭合控制点 ⇒ source_frame=42 / task_grasp_frame=50 **不变** ✓"""
    a = _recipe(phases=_phases(n_close=3), close_ctrl=16, settle_ctrl=10)
    b = _recipe(phases=_phases(n_close=37, n_settle=17), close_ctrl=37, settle_ctrl=17)
    ok_a, _ = validate_recipe_contract(a)
    ok_b, e_b = validate_recipe_contract(b)
    assert ok_a and ok_b, "两种时长都必须合法 (37/17 反例): %s" % e_b
    assert a["source_frame"] == b["source_frame"] == 42
    assert a["task_grasp_frame"] == b["task_grasp_frame"] == 50
    assert len(b["phase_boundaries"]["close"]) == 1
    assert b["phase_boundaries"]["close"][0] == [2, 2 + 37]


def test_task_grasp_frame_must_stay_50():
    r = _recipe()
    r["task_grasp_frame"] = 145          # 旧实现的错值 (随闭合帧数漂移 ✗)
    ok, errs = validate_recipe_contract(r)
    assert not ok and any("task_grasp_frame" in e for e in errs)


def test_source_frame_must_be_int_frame_not_matrix():
    r = _recipe()
    r["source_frame"] = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    ok, errs = validate_recipe_contract(r)
    assert not ok and any("source_frame" in e for e in errs)


def test_empty_nan_duplicate_time_cross_run_all_rejected():
    cases = {}
    r = _recipe(); r["actual_q_commands"] = []; r["T_W_G_cmd"] = []; r["phase"] = []
    r["control_index"] = []; r["actual_control_times"] = []; r["phase_boundaries"] = {}
    cases["空数组"] = r
    r = copy.deepcopy(_recipe()); r["actual_q_commands"][2] = float("nan")
    cases["NaN"] = r
    r = copy.deepcopy(_recipe()); r["actual_control_times"][2] = r["actual_control_times"][1]
    cases["时标重复"] = r
    r = copy.deepcopy(_recipe()); r["control_index"][3] = 99
    cases["跨run拼接(索引跳号)"] = r
    r = copy.deepcopy(_recipe()); r["T_W_G_cmd"][1] = [[1, 0, 0, 0]] * 4
    cases["缺旋转(非4x4)"] = r
    r = copy.deepcopy(_recipe()); r["T_W_G_cmd"][1][0][0] = 3.0    # det≠1
    cases["非法旋转"] = r
    r = copy.deepcopy(_recipe()); r["controller_hash"] = ""
    cases["缺 controller_hash"] = r
    for name, rr in cases.items():
        ok, errs = validate_recipe_contract(rr)
        assert not ok, "%s 必须被拒 (ARTIFACT_INVALID) ✗" % name
        assert errs, "%s 必须给出原因 ✗" % name


def test_phase_boundaries_must_cover_without_overlap():
    r = _recipe()
    r["phase_boundaries"] = {"travel": [[0, 2]], "close": [[3, 6]]}   # 断裂 ✗
    ok, errs = validate_recipe_contract(r)
    assert not ok and any("phase_boundaries" in e for e in errs)
    # derive 本身: 半开、连续、覆盖 [0,N) ✓
    pb = derive_phase_boundaries(_phases(2, 3, 2, 2))
    segs = sorted(sum((v for k, v in pb.items() if k != "n"), []))
    assert segs[0][0] == 0 and segs[-1][1] == pb["n"]
    assert all(segs[i][1] == segs[i + 1][0] for i in range(len(segs) - 1))
