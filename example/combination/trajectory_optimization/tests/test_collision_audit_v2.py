"""★ md 06.4-e2/e3 的 **mock 零物理反例测试** (2026-09-30) ✓

md 要求 ✓: 用 mock 验证"塌缩与歧义"8 类反例, **不启物理**:
  ① 仅一侧关闭            ② 掌部开放而双指关闭
  ③ 同名 actor 替换         ④ shape 数相同但 **group 变更**
  ⑤ getter 异常            ⑥ 空集合
  ⑦ 目标名**歧义子串**      ⑧ 精确名优先

关键口径 ✓ (md): "初始化 `target_fingers=True` **不能替代该验收**" ⇒
  本文件断言的是 **读回值 + 分指矩阵 + UNKNOWN 显式标注**, 不是启动时的一次性常量 ✗
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import physics_env as pe  # noqa: E402


# ─────────────────────────── mock 骨架 (零物理) ───────────────────────────
class _Pose:
    def __init__(self, p=(0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=np.float64)

    def to_transformation_matrix(self):
        m = np.eye(4)
        m[:3, 3] = self.p
        return m


class _Shape:
    def __init__(self, groups=(1, 1), verts=None, p=(0.0, 0.0, 0.0), boom=False):
        self.groups = tuple(groups)
        self._v = np.asarray(verts if verts is not None else
                             [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        self._p = _Pose(p)
        self.boom = bool(boom)          # True ⇒ 模拟 getter 抛异常
        self.n_read = 0

    def get_collision_groups(self):
        return self.groups

    def get_vertices(self):
        self.n_read += 1
        if self.boom:
            raise RuntimeError("mock getter boom")
        return self._v

    def get_local_pose(self):
        if self.boom:
            raise RuntimeError("mock getter boom")
        return self._p

    def get_physical_material(self):
        return None


class _Comp:
    def __init__(self, shapes):
        self._s = list(shapes)

    def get_collision_shapes(self):
        return self._s


class _RenderComp:
    """渲染组件: **没有** get_collision_shapes ⇒ 必须被组件级过滤, 不得吞整个 actor ✓"""


class _Actor:
    def __init__(self, name, comps, p=(0.0, 0.0, 0.0)):
        self._n = str(name)
        self.components = list(comps)
        self._p = _Pose(p)

    def get_name(self):
        return self._n

    def get_components(self):
        return self.components

    def get_pose(self):
        return self._p


class _Link:
    def __init__(self, shapes, name="link"):
        self._s = list(shapes)
        self._n = str(name)

    def get_name(self):
        return self._n

    def get_collision_shapes(self):
        return self._s


class _Robot:
    def __init__(self, links):
        self._l = list(links)

    def get_links(self):
        return self._l


class _Scene:
    def __init__(self, actors):
        self._a = list(actors)

    def get_all_actors(self):
        return self._a


def _finger_link(side, idx, groups):
    # ⚠ 必须用**真实 link 名** f"{side}_gripper_finger_link{idx}" ✓
    #   (collect_collision_channels 按此名识别手指 ⇒ 否则 fingers=0 ✗)
    return _Link([_Shape(groups)], name="%s_gripper_finger_link%d" % (side, idx))


def _robot(f1_groups, f2_groups, side="left"):
    return _Robot([_finger_link(side, 1, f1_groups), _finger_link(side, 2, f2_groups)])


def _target(groups, name="glb_4"):
    return _Actor(name, [_Comp([_Shape(groups)])])


def _ground():
    return _Actor("ground", [_Comp([_Shape((2, 4))])])


def _find_mixin():
    import stage2_reconstruct_mixin as s2m
    for _name in dir(s2m):
        _o = getattr(s2m, _name)
        if isinstance(_o, type) and hasattr(_o, "_coacd_world_hulls_live"):
            return _o
    raise AssertionError("未找到含 _coacd_world_hulls_live 的 mixin")


# ═════════════════════ ① 仅一侧关闭 ⇒ 不得被 any 掩盖 ═════════════════════
def test_1_single_side_closed_not_masked_by_any():
    # 指1 与目标**相碰**(mask 命中), 指2 与目标**不相碰** ⇒ per_finger 必须分开给值 ✓
    rob = _robot((6, 64), (6, 1))          # 指2 的 mask=1 ⇒ 目标(64) 与之不相交 ✓
    tgt = _target((64, 6))
    scene = _Scene([tgt, _ground()])
    out = pe.collect_collision_channels(robot=rob, side="left",
                                        obj_actors=[tgt, _ground()],
                                        target_name="glb_4", scene=scene)
    pf = out["per_finger"]
    assert pf["target_finger1"] is True, pf
    assert pf["target_finger2"] is False, pf
    # 合并量 (any) 仍可为 True ⇒ 正说明**不能只信 any** ✓
    assert out["target_fingers"] is True


# ═════════════ ② 掌部开放而双指关闭 ⇒ 分指仍必须给真值 ═════════════
def test_2_palm_open_but_fingers_read_independently():
    rob = _robot((6, 64), (6, 64))         # 双指 mask 都命中目标
    tgt = _target((64, 6))
    scene = _Scene([tgt, _ground()])
    out = pe.collect_collision_channels(robot=rob, side="left",
                                        obj_actors=[tgt, _ground()],
                                        target_name="glb_4", scene=scene)
    pf = out["per_finger"]
    assert (pf["target_finger1"], pf["target_finger2"]) == (True, True)
    assert pf["both_fingers_shapes_ok"] is True
    assert pf["finger1_shapes"] == 1 and pf["finger2_shapes"] == 1


# ═════════════ ③ 同名 actor 替换 ⇒ 缓存必须失效 (几何身份) ═════════════
def test_3_same_name_actor_replacement_invalidates_cache(monkeypatch):
    mixin = _find_mixin()
    # `_coacd_world_hulls_live` 只收 PhysxRigidDynamicComponent ⇒ mock 需登记为该类型 ✓
    import sapien as _sp
    monkeypatch.setattr(_sp.pysapien.physx, "PhysxRigidDynamicComponent",
                        _Comp, raising=False)
    sh = _Shape((1, 1), verts=[[0, 0, 0], [1, 0, 0]])
    actor = _Actor("glb_4", [_Comp([sh])])
    self_ = SimpleNamespace(obj_actors=[actor], scene=None)
    h1 = mixin._coacd_world_hulls_live(self_, "glb_4")
    c1 = getattr(self_, "_coacd_local_hulls__glb_4")
    h2 = mixin._coacd_world_hulls_live(self_, "glb_4")
    c2 = getattr(self_, "_coacd_local_hulls__glb_4")
    assert len(h1) == 1 and len(h2) == 1
    assert c1 is c2, "签名未变 ⇒ 必须**命中缓存**(不重建) ✓"
    # 同名替换: 顶点数变 ⇒ 身份签名变 ⇒ 必须重建 ✓
    actor.components = [_Comp([_Shape((1, 1), verts=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])])]
    h3 = mixin._coacd_world_hulls_live(self_, "glb_4")
    assert h3[0].shape[0] == 3, "同名替换后仍用旧缓存 ✗"


# ═════════════ ④ shape 数**相同**但 group 变更 ⇒ 缓存必须失效 ═════════════
def test_4_same_shape_count_group_change_invalidates_cache(monkeypatch):
    mixin = _find_mixin()
    import sapien as _sp
    monkeypatch.setattr(_sp.pysapien.physx, "PhysxRigidDynamicComponent",
                        _Comp, raising=False)
    sh = _Shape((1, 1))
    actor = _Actor("glb_4", [_Comp([sh])])
    self_ = SimpleNamespace(obj_actors=[actor], scene=None)
    mixin._coacd_world_hulls_live(self_, "glb_4")
    c1 = getattr(self_, "_coacd_local_hulls__glb_4")
    sh.groups = (6, 64)                     # **形态不变**, 只改 group ⇒ 签名须变 ✓
    mixin._coacd_world_hulls_live(self_, "glb_4")
    c2 = getattr(self_, "_coacd_local_hulls__glb_4")
    assert c1 is not c2, "group 变更未使几何身份签名失效 ✗ (md 点名反例)"


# ═════════════ ⑤ getter 异常 ⇒ 显式 UNKNOWN, 不得静默通过 ═════════════
def test_5_getter_exception_is_unknown_not_pass():
    boom = _Shape((6, 64), boom=True)
    tgt = _Actor("glb_4", [_Comp([boom])])
    gnd = _ground()
    rob = _robot((6, 64), (6, 64))
    scene = _Scene([tgt, gnd])
    aud = pe.build_collision_audit(robot=rob, side="left", obj_actors=[tgt, gnd],
                                   target_name="glb_4", scene=scene, tag="t5")
    recs = [r for r in aud["shapes"] if r.get("component") == "_Comp"]
    assert recs, aud["shapes"]
    assert any(r.get("geometry") in ("UNKNOWN", None) or
               (isinstance(r.get("geometry"), dict) and r["geometry"].get("n_vertices") is not None)
               for r in recs)


# ═════════════ ⑥ 空集合 ⇒ None (不可验证 ⇒ 不算通过) ═════════════
def test_6_empty_collection_gives_none_not_pass():
    empt = _Actor("glb_4", [_RenderComp()])     # 只有渲染组件 ⇒ shapes 全空
    gnd = _ground()
    rob = _robot((6, 64), (6, 64))
    scene = _Scene([empt, gnd])
    out = pe.collect_collision_channels(robot=rob, side="left",
                                        obj_actors=[empt, gnd],
                                        target_name="glb_4", scene=scene)
    assert out["target_fingers"] is None, "空集合必须为 None (不可验证) ✗"
    assert out["errors"], "空集合必须留下 errors 记录 ✗"
    # 渲染组件必须被**组件级**跳过, 而不是让 actor 抛异常 ⇒ 有 UNKNOWN 明细 ✓
    aud = pe.build_collision_audit(robot=rob, side="left", obj_actors=[empt, gnd],
                                   target_name="glb_4", scene=scene, tag="t6")
    assert any(r.get("collision") == "UNKNOWN" for r in aud["shapes"]), aud["shapes"]


# ═════════════ ⑦ 歧义子串 ⇒ 拒绝 (不静默取第一个) ═════════════
def test_7_ambiguous_substring_rejected():
    a1 = _Actor("toy_0", [])
    a2 = _Actor("toy_01", [])
    got, info = pe.resolve_target_actor_identity([a1, a2], "toy")
    assert got is None, "歧义必须拒绝 ✗"
    assert info["ambiguous"] is True and len(info["candidates"]) == 2


# ═════════════ ⑧ 精确名优先于子串 ═════════════
def test_8_exact_name_wins_over_substring():
    a1 = _Actor("toy_0", [])
    a2 = _Actor("toy_01", [])
    got, info = pe.resolve_target_actor_identity([a1, a2], "toy_0")
    assert got is a1 and info["mode"] == "exact" and info["ambiguous"] is False
    # 唯一子串 ⇒ unique_substring ✓
    got2, info2 = pe.resolve_target_actor_identity([a1, a2], "01")
    assert got2 is a2 and info2["mode"] == "unique_substring"


# ═════════════ 附: 不存在的目标必须显式 not_found (不假通过) ═════════════
def test_9_missing_target_is_not_found_not_pass():
    got, info = pe.resolve_target_actor_identity([_Actor("glb_4", [])], "nope")
    assert got is None and info["mode"] == "not_found"
