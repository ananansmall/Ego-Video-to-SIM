"""任务02 测试 —— 场景/观测接口核对与"不得持续放行" (零物理 ✓)

md 位置: §4 任务 02（前置 01）；§6 "profile/观测/碰撞通道 UNKNOWN ⇒ 回 02, 零物理反例先修"

md 原话 ✓
  1. "根据当前安装版本核对 Scene、actor/entity/component 和 shape 的实际接口,
     **修 `get_actors` 等错误调用**; mock 必须模拟实际对象结构" ✓
  2. "**不再持续豁免 UNKNOWN**; 移除'只打一次日志、无限次放行'; 碰撞配置应在场景建好后可读" ✓
  3. "反例: **全 None 连续 4 次不得持续放行**; getter 错误报告**具体对象** ✓"

实测背景 (2026-09-28, SAPIEN 3.0.0b0) ✗
  · `hasattr(sapien.Scene, "get_actors") == False` (正式 API = `get_all_actors`/`get_entities`)
  · 旧代码用它 ⇒ AttributeError 被吞 ⇒ 地面通道空 ⇒ `target_table=None`
  · ⇒ `SCENE_CONTRACT_VIOLATION: 通道 target_table 缺实际值` ⇒ **Stage2 一步都迈不出** (实测)

本文件全部零物理 ✓: 只用 mock 场景对象 (按 md: mock 必须模拟**实际对象结构** ✓)
"""

from __future__ import annotations

import numpy as np
import pytest

import physics_env as pe


# --------------------------------------------------------------------------- #
# mock SAPIEN 3.0 场景对象 (模拟真实结构: component.get_collision_shapes() → shape)
# --------------------------------------------------------------------------- #
class _Shape:
    def __init__(self, groups):
        self._g = list(groups)

    def get_collision_groups(self):
        return list(self._g)

    def set_collision_groups(self, g):
        self._g = list(g)


class _Component:
    def __init__(self, shapes):
        self._shapes = list(shapes)

    def get_collision_shapes(self):
        return list(self._shapes)

    def get_name(self):
        return "comp"


class _Actor:
    def __init__(self, name, groups=(1, 1, 1, 0)):
        self._name = name
        self._comp = _Component([_Shape(groups)])

    def get_name(self):
        return self._name

    def get_components(self):
        return [self._comp]


class _SceneV3:
    """★ SAPIEN 3.0 真实结构 ✓: 只有 `get_all_actors`, **没有** `get_actors` ✗"""

    def __init__(self, actors):
        self._actors = list(actors)

    def get_all_actors(self):
        return list(self._actors)


class _SceneLegacyOnly:
    """旧版本结构 (有 `get_actors`) ⇒ 兜底路径要能用 ✓"""

    def __init__(self, actors):
        self._actors = list(actors)

    def get_actors(self):
        return list(self._actors)


class _SceneBroken:
    """getter 抛错 ⇒ 必须报**具体对象**且不静默 ✓"""

    def get_all_actors(self):
        raise RuntimeError("physx scene not ready")

    def get_actors(self):
        raise RuntimeError("legacy path broken too")


# --------------------------------------------------------------------------- #
# 1) `_scene_actors`: 版本兼容 (md §02-1)
# --------------------------------------------------------------------------- #
def test_scene_actors_prefers_v3_api_when_get_actors_absent():
    """★ 核心反例 ✓: SAPIEN 3.0 没有 `get_actors` ⇒ 仍必须读到 actor"""
    ground = _Actor("ground")
    sc = _SceneV3([ground, _Actor("table")])
    assert not hasattr(sc, "get_actors")            # 模拟真实 3.0 结构 ✓
    got = pe._scene_actors(sc)
    assert [a.get_name() for a in got] == ["ground", "table"]


def test_scene_actors_falls_back_to_legacy_name():
    sc = _SceneLegacyOnly([_Actor("ground")])
    assert [a.get_name() for a in pe._scene_actors(sc)] == ["ground"]


def test_scene_actors_returns_empty_on_broken_getter_without_raising():
    assert pe._scene_actors(_SceneBroken()) == []
    assert pe._scene_actors(None) == []


# --------------------------------------------------------------------------- #
# 2) `collect_collision_channels`: 地面通道必须能读到 + 读不到要报具体对象 (md §02-1/3)
# --------------------------------------------------------------------------- #
def test_ground_channel_is_readable_on_v3_scene():
    """★ 修复前 ✗: `scene.get_actors()` 抛错 ⇒ 地面为空 ⇒ `target_table=None`
    修复后 ✓: 走 `get_all_actors` ⇒ 地面 shape 有值 ⇒ `target_table` 可判 ✓"""
    target = _Actor("glb_4", groups=(1, 1, 1, 0))
    ground = _Actor("ground", groups=(1, 1, 1, 0))
    sc = _SceneV3([ground, target])
    ch = pe.collect_collision_channels(robot=None, side="left",
                                       obj_actors=[target], target_name="glb_4", scene=sc)
    assert ch["counts"]["ground"] >= 1, f"地面通道仍为空 ✗ (errors={ch['errors']})"
    assert ch["counts"]["target"] >= 1
    assert ch["target_table"] is not None, "target_table 必须可判 (修复点 ✓)"


def test_missing_ground_reports_specific_error():
    """md §02-3 ✓: getter/内容错误必须**具体**, 不能静默 ✗"""
    target = _Actor("glb_4")
    sc = _SceneV3([target])                     # 没有任何名字含 ground 的 actor
    ch = pe.collect_collision_channels(robot=None, side="left",
                                       obj_actors=[target], target_name="glb_4", scene=sc)
    assert ch["counts"]["ground"] == 0
    assert ch["target_table"] is None            # ⇒ 不可验证 ⇒ 门会阻断 ✓
    assert any("ground" in e for e in ch["errors"]), ch["errors"]


def test_broken_scene_getter_error_names_the_object():
    ch = pe.collect_collision_channels(robot=None, side="left",
                                       obj_actors=[], target_name="glb_4",
                                       scene=_SceneBroken())
    assert ch["errors"], "getter 失败必须留下 errors ✗"
    assert any(("_SceneBroken" in e) or ("对象=" in e) for e in ch["errors"]), ch["errors"]


# --------------------------------------------------------------------------- #
# 3) ★ md §02-2/3: 全 None **连续 4 次不得持续放行**
# --------------------------------------------------------------------------- #
def _register_contract():
    """登记"期望通道为真" + 清掉自举计数 ⇒ 每次测试都从干净状态开始 ✓"""
    pe.register_scene_contract({"target_fingers": True, "target_table": True,
                                "fingers_table": True}, tag="t02")
    pe._SCENE_CONTRACT.pop("_bootstrap_used", None)
    pe._SCENE_CONTRACT.pop("_bootstrap_logged", None)


def test_all_none_channels_cannot_be_excused_four_times():
    """md ✓: "全 None 连续 4 次不得持续放行" —— 自举放行**只允许一次** ✓

    旧病 ✗ (修复前): 全 None 时**每次都** return True ⇒ 门控永久失效 ✗
    """
    _register_contract()
    pe.register_scene_sampler(lambda: {"target_fingers": None, "target_table": None,
                                  "fingers_table": None, "others": None, "counts": {},
                                  "errors": []})
    outcomes = []
    for _i in range(4):
        try:
            pe.assert_scene_contract(actual=None)
            outcomes.append("allowed")
        except RuntimeError as exc:
            outcomes.append("blocked")
            assert "SCENE_CONTRACT_VIOLATION" in str(exc)
            assert "不得持续豁免" in str(exc) or "连续无实际值" in str(exc)
    assert outcomes[0] == "allowed", "真正的首个 step 前应放行一次 ✓"
    assert outcomes[1:] == ["blocked", "blocked", "blocked"], \
        f"第 2~4 次必须阻断 ✗, 实际={outcomes} (md §02: 不得持续豁免 UNKNOWN)"


def test_contact_masks_restore_the_three_required_channels():
    """★ 任务02 实证反例 ✓: 掩码表必须让**目标—桌面 / 指尖—桌面**成立

    实测 ✗ (output/stage2_t02): 读回 `target_table=False`, `fingers_table=False`
      ⇒ 根因: 旧表 `default`(地面) 的 mask 只有 bit0, 而 PhysX 真实语义是**双向命中**
        `(gA & mB) and (gB & mA)` ⇒ `target(4) & default.mask(1) = 0` ✗
    """
    from stage2_reconstruct_mixin import STAGE1_TARGET_ISOLATED_CONTACT_MASKS as M

    def pair_ok(a, b):
        # PhysX 语义 (与 physics_env._pair_collides 一致 ✓)
        return bool((M[a][0] & M[b][1]) and (M[b][0] & M[a][1]))

    assert pair_ok("target", "default") is True, "目标—桌面必须可碰 ✗ (md: 不得关掉)"
    assert pair_ok("gripper", "default") is True, "指尖—桌面必须可碰 ✗"
    assert pair_ok("target", "gripper") is True, "目标—双指必须可碰 ✗"
    # 其他物体仍被隔离 ✓ (md: 其他物体隔离)
    assert pair_ok("non_target", "target") is False
    assert pair_ok("non_target", "gripper") is False
    # 但非目标仍要能落地 ✓ (否则会掉穿)
    assert pair_ok("non_target", "default") is True


def test_any_channel_with_value_still_strictly_checked():
    """只要有一条通道有值 ⇒ 立刻回到严格核对 (不放松 ✓)"""
    _register_contract()
    pe.register_scene_sampler(lambda: {"target_fingers": True, "target_table": None,
                                  "fingers_table": None, "others": False, "counts": {},
                                  "errors": []})
    with pytest.raises(RuntimeError) as ei:
        pe.assert_scene_contract(actual=None)
    assert "SCENE_CONTRACT_VIOLATION" in str(ei.value)
