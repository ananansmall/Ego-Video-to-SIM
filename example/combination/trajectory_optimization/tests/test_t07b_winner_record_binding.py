"""T07.2 反例: 获胜记录**强绑定** (不跑物理仿真 ✓)

验收口径 (md T07.1): 候选 A/B 的 **q 终点完全相同**、位姿和成功结果不同 ⇒
  导出 A **只能**取 A 的指定 episode；拿不到 ⇒ `WINNER_RECORD_MISSING` ✗
  **禁止**按 q 终点猜、**禁止**"最长轨迹"回退、**禁止**最后一次缓存 ✗
"""
from stage1_recipe import bind_winner_record

_Q_END = 0.0123          # A/B **完全相同**的 q 终点 ✓


def _mk(ep, cand, pos, q):
    return {
        "episode_id": ep,
        "candidate_id": cand,
        "immutable_record": {
            "schema_version": "s1_exec_trace/v1",
            "episode_id": ep,
            "candidate_id": cand,
            "t_s": [0.0, 1.0 / 30.0, 2.0 / 30.0][:len(q)],
            "base_pos": [list(pos)] * len(q),
            "q_cmd": list(q),
        },
        "assessment": {"obj_lift": 0.05},
    }


def test_export_A_takes_A_episode_not_B():
    A = _mk("s1ep-000001", "candA", [0.10, 0.20, 0.30], [0.030, 0.020, _Q_END])
    B = _mk("s1ep-000002", "candB", [0.90, 0.80, 0.70], [0.029, 0.021, _Q_END])
    assert A["immutable_record"]["q_cmd"][-1] == B["immutable_record"]["q_cmd"][-1]
    rec_a, err_a = bind_winner_record(A)
    assert err_a is None
    assert rec_a["episode_id"] == "s1ep-000001", "导出 A 只能取 A 的 episode ✗"
    assert rec_a["candidate_id"] == "candA"
    assert rec_a["base_pos"][0] == [0.10, 0.20, 0.30], "位姿必须属于 A ✗ (串到 B 了)"


def test_missing_record_reports_winner_record_missing():
    rec, err = bind_winner_record({"episode_id": "s1ep-9", "candidate_id": "candX"})
    assert rec is None and err == "WINNER_RECORD_MISSING"


def test_no_fallback_to_longest_or_last_cache():
    """即使传入"历史轨迹/最长轨迹/最后缓存", 缺失绑定也必须报 MISSING ✗ (禁止替代 ✓)"""
    fake = {
        "episode_id": "s1ep-9",
        "candidate_id": "candX",
        "immutable_record": None,
        "_s1_exec_q_by_rollout": [{"q": [0.03] * 500}],
        "_last_close_q_executed": [0.03] * 300,
    }
    rec, err = bind_winner_record(fake)
    assert rec is None and err == "WINNER_RECORD_MISSING"


def test_empty_or_bad_record_rejected():
    for bad in ({"immutable_record": {"q_cmd": []}},
                {"immutable_record": {"q_cmd": None}},
                {"immutable_record": "not-a-dict"},
                {}):
        rec, err = bind_winner_record(bad)
        assert rec is None and err == "WINNER_RECORD_MISSING"


def test_repeat_runs_share_candidate_id_but_not_episode():
    """同一候选的重复 run: candidate_id **相同**, episode_id **必须不同** ✓"""
    r1 = _mk("s1ep-000011", "candA", [0.1, 0.2, 0.3], [0.030, _Q_END])
    r2 = _mk("s1ep-000012", "candA", [0.1, 0.2, 0.3], [0.030, _Q_END])
    assert r1["candidate_id"] == r2["candidate_id"]
    assert r1["episode_id"] != r2["episode_id"]
    rec2, _ = bind_winner_record(r2)
    assert rec2["episode_id"] == "s1ep-000012"      # 选中的是 r2 ⇒ 不能拿 r1 ✗
