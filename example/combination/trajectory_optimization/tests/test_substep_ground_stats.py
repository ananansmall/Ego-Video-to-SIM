"""★ 2026-09-22 【复审 I.1 / K-1】桌面支撑统计的**校准测试** ✓

复审 I.1 原话 ✓:
  "验收用三个微型场景即可: **静置支撑、完全离桌、先离桌再落下**;
   再增加**重复 reset** 和**两个接触对同时存在**的计数测试。
   未通过前暂停新增物理参数扫描" ✓
复审 I.1-3 ✓: "强制检查 `0 ≤ n_contact_substeps ≤ substeps`; **无采样返回 UNKNOWN**,
  不是'不受支撑'" ✓

本文件用**伪接触**直接驱动 `_probe_substep_ground` / `object_ground_report` /
`reset_substep_ground_log` ✓ (确定性、秒级 ✓ ⇒ 专门锁住"记账"这件事 ✓,
不依赖 SAPIEN 求解器 ⇒ 不会因为物理随机性而 flaky ✓)

曾经的 bug (本测试要锁死 ✓):
  ① 跨子步永久 True 的标志 ⇒ 静置盒 20 子步只得 `n=1 / ratio=0.05 / supported=False` ✗
  ② reset 没清 `obj_ground` ⇒ 再次 reset 后 `n=1, substeps=0, ratio=1.0, supported=True` ✗
  ③ 按接触对累加 ⇒ 占比可 >100% (151% / 88794.10%) ✗
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import physics_env as pe  # noqa: E402


class _Pt:
    def __init__(self, sep, imp):
        self.separation = float(sep)
        self.impulse = np.array([float(imp), 0.0, 0.0])


class _Contact:
    def __init__(self, b0, b1, pts):
        self.bodies = (b0, b1)
        self.points = list(pts)


class _Body:
    def __init__(self, name):
        self.name = name


class _Scene:
    """最小伪场景 ✓: 只实现探针用到的两个接口 ✓"""

    def __init__(self):
        self.contacts = []

    def get_all_actors(self):
        return []                                   # ⇒ 名称缓存为空 ⇒ 回退 `body.name` ✓

    def get_contacts(self):
        return list(self.contacts)


_OBJ = _Body("glb_4")
_GND = _Body("ground")
_F1 = _Body("left_gripper_finger_link1")
_F2 = _Body("left_gripper_finger_link2")


def _step(scene, obj_ground=0, gripper_ground=0, sep=0.0, imp=1.0):
    """执行**一个子步** ✓ (obj_ground / gripper_ground = 该子步内的接触**对数** ✓)"""
    scene.contacts = []
    for k in range(obj_ground):
        scene.contacts.append(_Contact(_OBJ, _GND, [_Pt(sep, imp)]))
    for k in range(gripper_ground):
        scene.contacts.append(_Contact(_F1 if k % 2 == 0 else _F2, _GND, [_Pt(sep, imp)]))
    pe._probe_substep_ground(scene)


def _init(n_obj_pairs_per_step=1, enable=True, target="glb_4"):
    pe.reset_substep_ground_log(enable=enable, phase="test")
    pe.set_object_ground_target(target)
    pe._SUBSTEP_NAME_CACHE.clear()
    return _Scene(), n_obj_pairs_per_step


# ─────────────────────── 场景 1: 静置支撑 (盒子一直被桌面托着 ✓) ───────────────────────
def test_scene1_static_support_is_supported():
    scene, n_pairs = _init()
    for _ in range(20):
        _step(scene, obj_ground=n_pairs)
    rep = pe.object_ground_report()
    assert rep["valid"] is True, rep
    assert rep["substeps"] == 20, rep
    assert rep["n_contact_substeps"] == 20, rep          # ★ 旧实现只会得 1 ✗
    assert rep["contact_ratio"] == pytest.approx(1.0), rep
    assert rep["supported"] is True, rep                 # ★ 旧实现得 False ✗
    assert rep["state"] == "SUPPORTED", rep
    assert rep["overflow"] == 0, rep


# ─────────────────────── 场景 2: 完全离桌 (没有任何目标—地面接触 ✓) ───────────────────────
def test_scene2_full_off_table_is_not_supported():
    scene, _ = _init()
    for _ in range(20):
        _step(scene, obj_ground=0)
    rep = pe.object_ground_report()
    assert rep["valid"] is True, rep
    assert rep["n_contact_substeps"] == 0, rep
    assert rep["contact_ratio"] == pytest.approx(0.0), rep
    assert rep["supported"] is False, rep
    assert rep["state"] == "OFF_TABLE", rep


# ──────────────── 场景 3: 先离桌再落下 (保持窗内后段重新接触 ⇒ 占比被稀释 ✓) ────────────────
def test_scene3_lift_then_relocate_dilutes_ratio():
    scene, _ = _init()
    # 抬升期: 一直有接触 (旧口径若拿整段会判 SUPPORTED ✗)
    for _ in range(10):
        _step(scene, obj_ground=1)
    # ★ 复审 I.1-4 ✓: 保持窗**开头重新 reset** ✓ (只统计抬升结束后 ✓)
    pe.reset_substep_ground_log(phase="hold")
    for _ in range(6):
        _step(scene, obj_ground=1)          # 前 6 子步仍在桌面
    for _ in range(14):
        _step(scene, obj_ground=0)          # 之后离桌
    rep = pe.object_ground_report()
    assert rep["substeps"] == 20, rep
    assert rep["n_contact_substeps"] == 6, rep
    assert rep["contact_ratio"] == pytest.approx(0.3), rep
    assert rep["state"] == "OFF_TABLE", rep
    assert rep["phase"] == "hold", rep


# ─────────────────────── 重复 reset: 无采样 ⇒ UNKNOWN (不是 True/False ✓) ───────────────────────
def test_repeated_reset_clears_everything_and_reports_unknown():
    scene, _ = _init()
    for _ in range(5):
        _step(scene, obj_ground=1)
    assert pe.object_ground_report()["n_contact_substeps"] == 5

    pe.reset_substep_ground_log()                     # ★ 再 reset, 尚未跑任何子步 ✓
    rep = pe.object_ground_report()
    assert rep["substeps"] == 0, rep
    assert rep["n_contact_substeps"] == 0, rep        # ★ 旧实现残留 1 ✗
    assert rep["valid"] is False, rep                 # ★ 无采样 ⇒ 不可信 ✓
    assert rep["supported"] is None, rep              # ★ 不是 True 也不是 False ✓
    assert rep["state"] == "UNKNOWN", rep
    assert rep["reason"] == "no_substep_samples", rep
    assert rep["sum_impulse"] == 0.0, rep             # 冲量也必须清零 ✓
    assert rep["min_sep_mm"] is None, rep


# ──────────────── 同一子步内**两个接触对同时存在** ⇒ 两个通道各至多加 1 ✓ ────────────────
def test_two_contact_pairs_same_substep_count_once_per_channel():
    scene, _ = _init()
    for _ in range(3):
        _step(scene, obj_ground=2, gripper_ground=2)   # 同子步: 目标×2 + 夹爪×2 ✓
    rep = pe.object_ground_report()
    assert rep["substeps"] == 3, rep
    assert rep["n_contact_substeps"] == 3, rep        # ★ 不是 6 ✗
    assert rep["n_gripper_contact_substeps"] == 3, rep  # ★ 不是 6 ✗
    assert rep["contact_ratio"] == pytest.approx(1.0), rep
    assert rep["overflow"] == 0, rep


# ─────────────────────── 有界性: n 恒 ≤ substeps 且 overflow 恒为 0 ✓ ───────────────────────
def test_counts_are_bounded():
    scene, _ = _init()
    for _ in range(30):
        _step(scene, obj_ground=3, gripper_ground=3)
    rep = pe.object_ground_report()
    assert rep["substeps"] == 30, rep
    assert rep["n_contact_substeps"] <= rep["substeps"], rep
    assert rep["n_gripper_contact_substeps"] <= rep["substeps"], rep
    assert rep["overflow"] == 0, rep
    assert rep["valid"] is True, rep
    # 逐 link 也应有界 ✓
    for _ln, _e in pe._SUBSTEP_GROUND_LOG["by_link"].items():
        assert _e["n"] <= rep["substeps"], (_ln, _e)


# ─────────────── 最小净空/冲量统计仍然有效 (修记账不能丢信息 ✓) ───────────────
def test_min_sep_and_impulse_still_recorded():
    scene, _ = _init()
    _step(scene, obj_ground=1, sep=0.0, imp=2.0)              # separation 单位 = **米** ✓
    _step(scene, obj_ground=1, sep=-0.5e-3, imp=3.0)          # 负 separation = 穿透 ✓ (−0.5mm)
    _step(scene, obj_ground=1, sep=1.0e-3, imp=1.0)           # +1.0mm ✓
    rep = pe.object_ground_report()
    assert rep["min_sep_mm"] == pytest.approx(-0.5), rep
    assert rep["sum_impulse"] == pytest.approx(6.0), rep
    assert rep["max_impulse"] == pytest.approx(3.0), rep


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
