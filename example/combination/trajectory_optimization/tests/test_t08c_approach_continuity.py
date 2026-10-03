"""T08-C: 连续接近审计 (不跑物理 ✓)

md ✓: t 持续增长 / MANO 源相位 s 单调不减 / 控制索引 k 递增;
  去掉末尾重复点与"张开空等"(必要闭合/稳定**不计**空等);
  world 线/角速度边界连续 (轴向 entry **不先停再启**);
  ready 连续 2 周期达标 (1mm/1°/0.5mm/5mm·s⁻¹/5°·s⁻¹);
  入口偏移 ⇒ ENTRY_STATE_MISMATCH (2mm/2°), **不得 set_pose 纠正**
"""
import stage1_recipe as sr


# ── ① 三个量 ─────────────────────────────────────────────────────────────────
def test_phase_clocks_good_case():
    ok, p = sr.audit_phase_clocks([10, 10, 11, 12], [0, 1, 2, 3], [0.0, .1, .2, .3])
    assert ok and not p


def test_phase_clocks_detect_regressions():
    assert not sr.audit_phase_clocks([12, 11], [0, 1], [0, .1])[0]      # s 倒退
    assert not sr.audit_phase_clocks([10, 11], [0, 2], [0, .1])[0]      # k 跳号
    assert not sr.audit_phase_clocks([10, 11], [0, 1], [0.1, 0.1])[0]   # t 未增长


# ── ② 末尾重复点 ──────────────────────────────────────────────────────────────
def test_duplicate_tail_points():
    n, keep = sr.detect_duplicate_tail_points([0.30, 0.20, 0.10, 0.10, 0.10])
    assert n == 2 and keep == 3, "重复末尾点必须被识别并可裁剪 ✓"
    assert sr.detect_duplicate_tail_points([0.3, 0.2, 0.1])[0] == 0


# ── ③ 张开空等 vs 必要闭合 ────────────────────────────────────────────────────
def test_open_idle_separated_from_useful_close():
    ph = ["approach"] * 5 + ["close"] * 4 + ["settle"] * 2
    q = [0.030] * 5 + [0.020, 0.014, 0.010, 0.008] + [0.008, 0.008]
    rep = sr.classify_phase_idle(ph, q=q, q_open=0.030)
    assert rep["open_idle_total"] == 5, "接近段全开 = 张开空等 ⇒ 计入 ✗"
    assert "close" not in (rep["open_idle_by_phase"] or {}), "必要闭合**不得**计为空等 ✓"
    assert rep["useful"]["close"] == 4


# ── ④ 速度连续性 (不先停再启) ─────────────────────────────────────────────────
def test_velocity_continuity_ok_and_stop_then_restart():
    V = [[0.2, 0, 0], [0.1, 0, 0], [0.05, 0, 0]]
    W = [[0, 0, 0.1], [0, 0, 0.05], [0, 0, 0.02]]
    assert sr.audit_velocity_continuity([0, .1, .2], V, W)[0] is True
    V2 = [[0.2, 0, 0], [0.0, 0, 0], [0.2, 0, 0]]     # 中途停再启 ✗
    ok, p = sr.audit_velocity_continuity([0, .1, .2], V2, W)
    assert not ok and any("停顿再启动" in x for x in p), p


def test_velocity_continuity_allows_true_standstill():
    V = [[0.0, 0, 0], [0.0, 0, 0]]
    assert sr.audit_velocity_continuity([0, .1], V, [[0, 0, 0], [0, 0, 0]])[0] is True


# ── ⑤ ready 容差 ─────────────────────────────────────────────────────────────
def test_ready_requires_two_consecutive_cycles():
    ok, det = sr.check_ready_samples([0.0008, 0.0005], [0.9, 0.5], [0.0004, 0.0003],
                                     [0.004, 0.002], [4.0, 2.0])
    assert ok, det
    assert not sr.check_ready_samples([0.0008], [0.9], [0.0004], [0.004], [4.0])[0]
    assert not sr.check_ready_samples([0.0015, 0.0004], [0.5, 0.5], [0.0004, 0.0004],
                                      [0.004, 0.004], [4.0, 4.0])[0]     # 1.5mm > 1mm ✗
    assert not sr.check_ready_samples([0.0004, 0.0004], [0.5, 0.5], [0.0004, 0.0004],
                                      [0.004, 0.004], [6.0, 4.0])[0]     # 角速 6°/s ✗


# ── ⑥ 入口状态 ───────────────────────────────────────────────────────────────
def test_entry_state_mismatch_reports_and_forbids_set_pose():
    ok, det = sr.check_entry_state(0.0015, 1.5)
    assert ok, det
    ok2, det2 = sr.check_entry_state(0.0035, 0.5)
    assert not ok2 and "ENTRY_STATE_MISMATCH" in det2 and "set_pose" in det2
    ok3, det3 = sr.check_entry_state(0.0005, 2.5)
    assert not ok3 and "ENTRY_STATE_MISMATCH" in det3
    assert not sr.check_entry_state(None, None)[0]      # 无实测 ⇒ 不得当作匹配 ✗
