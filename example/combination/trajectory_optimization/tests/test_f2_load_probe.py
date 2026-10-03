"""★ md 06.4-f2 验收测试 (2026-09-30) ✓ —— 逐指载荷 / 法向·切向 / 分类 (0 物理)

f2 验收原文 ✓: "掌部有力但双指无力、单指有力、其他物体有力、法向/切向分量、不同子步数、
  C0相对变换、配置返回副本/写入失败" ✓

⚠ md 7.1 ✓: mock **只隔离求解器/外部资源**(机器人与场景), **不替代**待测的分类与门控逻辑
  ⇒ 本文件直接调用生产函数 `physics_env.get_grasp_force_detail` ✓
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sapien  # noqa: E402
import physics_env as pe  # noqa: E402


class _Pt:
    def __init__(self, impulse, normal):
        self.impulse = np.asarray(impulse, dtype=np.float64)
        self.normal = np.asarray(normal, dtype=np.float64)


class _Contact:
    def __init__(self, b0, b1, points):
        self.bodies = (b0, b1)
        self.points = list(points)


class _CompAL:
    """mock: PhysxArticulationLinkComponent (夹爪 link 组件)"""


class _CompRD:
    """mock: PhysxRigidDynamicComponent (物体组件)"""


class _Ent:
    def __init__(self, comps):
        self.components = list(comps)


class _Link:
    def __init__(self, name, comp):
        self._n = name
        self._c = comp
        self.entity = _Ent([comp])

    def get_name(self):
        return self._n


class _Robot:
    def __init__(self, links):
        self._l = list(links)

    def get_links(self):
        return self._l


class _Actor:
    def __init__(self, name, comp):
        self._n = name
        self.components = [comp]

    def get_name(self):
        return self._n


class _Scene:
    def __init__(self, contacts):
        self._c = list(contacts)

    def get_contacts(self):
        return self._c


def _build(monkeypatch, contacts, side="left", target="glb_4"):
    monkeypatch.setattr(sapien.pysapien.physx, "PhysxArticulationLinkComponent",
                        _CompAL, raising=False)
    monkeypatch.setattr(sapien.pysapien.physx, "PhysxRigidDynamicComponent",
                        _CompRD, raising=False)
    c1, c2, cp = _CompAL(), _CompAL(), _CompAL()
    rob = _Robot([_Link(f"{side}_gripper_finger_link1", c1),
                  _Link(f"{side}_gripper_finger_link2", c2),
                  _Link(f"{side}_gripper_link", cp)])
    ct, co = _CompRD(), _CompRD()
    acts = [_Actor(target, ct), _Actor("other_obj", co)]
    return rob, acts, {"f1": c1, "f2": c2, "palm": cp, "tgt": ct, "oth": co}


def test_normal_tangent_split(monkeypatch):
    """impulse=(3,4,0), normal=(1,0,0) ⇒ 法向 3/dt、切向 4/dt ✓"""
    rob, acts, comps = _build(monkeypatch, [])
    # dt = PHYSICS_TIMESTEP; impulse 以"冲量"计 ⇒ 力 = 冲量/dt
    pt = _Pt([3.0, 4.0, 0.0], [1.0, 0.0, 0.0])
    sc = _Scene([_Contact(comps["f1"], comps["tgt"], [pt])])
    out = pe.get_grasp_force_detail("left", sc, acts, rob, "glb_4")
    dt = pe.PHYSICS_TIMESTEP
    assert out["n_target_f1"] == pytest.approx(3.0 / dt), "法向 = |impulse·normal|/dt ✓"
    assert out["t_target_f1"] == pytest.approx(4.0 / dt), "切向 = |impulse - 法向分量|/dt ✓"
    assert out["f_target_f1"] == pytest.approx(5.0 / dt), "总模长 = 5/dt ✓"
    assert out["n_target_f2"] == 0.0, "另一指必须为零 ✓"


def test_palm_only_does_not_count_as_bilateral(monkeypatch):
    rob, acts, comps = _build(monkeypatch, [])
    sc = _Scene([_Contact(comps["palm"], comps["tgt"], [_Pt([1.0, 0, 0], [1, 0, 0])])])
    out = pe.get_grasp_force_detail("left", sc, acts, rob, "glb_4")
    assert out["f_palm"] > 0.0
    assert out["n_target_f1"] == 0.0 and out["n_target_f2"] == 0.0, (
        "掌部有力**不得**计为双指承载 ✓ (md: 掌部/桌面接触不能计入左右目标载荷 ✓)")


def test_single_finger_only_is_reported_asymmetric(monkeypatch):
    rob, acts, comps = _build(monkeypatch, [])
    sc = _Scene([_Contact(comps["f2"], comps["tgt"], [_Pt([2.0, 0, 0], [1, 0, 0])])])
    out = pe.get_grasp_force_detail("left", sc, acts, rob, "glb_4")
    assert out["n_target_f2"] > 0.0 and out["n_target_f1"] == 0.0, "单指有力必须如实非对称 ✓"


def test_other_object_load_not_counted(monkeypatch):
    rob, acts, comps = _build(monkeypatch, [])
    sc = _Scene([_Contact(comps["f1"], comps["oth"], [_Pt([1.0, 0, 0], [1, 0, 0])])])
    out = pe.get_grasp_force_detail("left", sc, acts, rob, "glb_4")
    assert out["f_other"] > 0.0
    assert out["n_target_f1"] == 0.0, "其他物体有力不得计入目标载荷 ✓"


def test_normal_probe_missing_is_marked_not_zero(monkeypatch):
    class _PtNoNormal:
        impulse = np.array([1.0, 0.0, 0.0])
        # 无 normal 属性 ⇒ getattr 抛异常
        def __getattr__(self, _n):
            raise RuntimeError("no normal")

    rob, acts, comps = _build(monkeypatch, [])
    sc = _Scene([_Contact(comps["f1"], comps["tgt"], [_PtNoNormal()])])
    out = pe.get_grasp_force_detail("left", sc, acts, rob, "glb_4")
    assert out["normal_probe_state"].startswith("MISSING"), (
        "normal 不可读必须**显式标注缺测**, 不能用 0 冒充 ✓")
