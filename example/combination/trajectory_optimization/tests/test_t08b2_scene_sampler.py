"""T08-B2: 三通道**从场景自动采样** (不跑物理 ✓)

md ✓: "逐 body/shape 核对 目标—双指 / 目标—桌面 / 双指—桌面 通道, 其他物体隔离。
  **只比摘要而不验证实际场景不算完成**"; 且"改一个字段必须在 step 前阻断" ✓
PhysX 语义: 两 shape 相碰 ⟺ (groupA & maskB) != 0 且 (groupB & maskA) != 0 ✓
"""
import pytest

import physics_env as pe

_T_FING = 0b0001
_T_OBJ = 0b0010
_T_GND = 0b0100
_M_ALL = 0b1111


class _S:
    def __init__(self, g, m):
        self._g = [int(g), int(m), 0, 0]

    def get_collision_groups(self):
        return list(self._g)


class _Link:
    def __init__(self, name, g, m):
        self._n, self._s = name, _S(g, m)

    def get_name(self):
        return self._n

    def get_collision_shapes(self):
        return [self._s]


class _Act:
    def __init__(self, name, g, m):
        self._n, self._c = name, [_Link(name, g, m)]

    def get_name(self):
        return self._n

    def get_components(self):
        return self._c


class _Scene:
    def __init__(self, actors):
        self._a = actors

    def get_actors(self):
        return self._a


def _robot(f_mask=_M_ALL):
    return type("R", (), {"get_links": lambda s: [
        _Link("left_gripper_finger_link1", _T_FING, f_mask),
        _Link("left_gripper_finger_link2", _T_FING, f_mask)]})()


def _objs(target_mask=_M_ALL, other_mask=_T_GND):
    return [_Act("glb_4", _T_OBJ, target_mask), _Act("other_1", _T_OBJ, other_mask)]


def _sample(scene_actors=None, robot=None, obj=None):
    return pe.collect_collision_channels(
        robot=robot or _robot(), side="left", obj_actors=obj or _objs(),
        target_name="glb_4",
        scene=_Scene(scene_actors if scene_actors is not None
                     else [_Act("ground", _T_GND, _M_ALL)]))


def test_all_required_channels_read_on_and_others_isolated():
    ch = _sample()
    assert ch["target_fingers"] is True and ch["target_table"] is True
    assert ch["fingers_table"] is True
    assert ch["others"] is False, "非目标须隔离 (OFF ✓)"
    assert ch["counts"]["fingers"] == 2 and ch["counts"]["target"] == 1


def test_target_fingers_channel_reads_off_when_mask_forbids():
    ch = _sample(obj=_objs(target_mask=_T_GND))     # 目标 mask 不含手指组
    assert ch["target_fingers"] is False, "读回必须反映**真实**通道 ✗"


def test_others_channel_reads_on_when_not_isolated():
    ch = _sample(obj=_objs(other_mask=_M_ALL))
    ch2 = pe.collect_collision_channels(
        robot=_robot(f_mask=_M_ALL), side="left",
        obj_actors=[_Act("glb_4", _T_OBJ, _M_ALL), _Act("other_1", _T_OBJ, _M_ALL)],
        target_name="glb_4", scene=_Scene([_Act("ground", _T_GND, _M_ALL)]))
    assert ch2["others"] is True, "手指↔非目标相碰 ⇒ others 必须 ON ⇒ 违规 ✓"


def test_missing_ground_leaves_channels_unknown():
    ch = _sample(scene_actors=[])
    assert ch["target_table"] is None and ch["fingers_table"] is None
    assert ch["target_fingers"] is True


def test_gate_uses_sampler_and_blocks_before_step(monkeypatch=None):
    """★ 关键: 登记采样器后, step 前门必须用**真实读回** ⇒ 违规即阻断 ✓"""
    pe.clear_scene_contract()
    pe.clear_scene_sampler()
    _REQ = {"target_fingers": True, "target_table": True, "fingers_table": True}
    # ① 真实读回违规 (目标—桌面 被关) ⇒ 阻断 ✓
    _bad = _sample(obj=_objs(target_mask=_T_FING))   # 目标 mask 不含 ground
    pe.register_scene_contract(_REQ, tag="stage2")
    pe.register_scene_sampler(lambda: _bad)
    with pytest.raises(RuntimeError, match="SCENE_CONTRACT_VIOLATION"):
        pe.assert_scene_contract()
    # ② 换成合规读回 ⇒ 放行 ✓
    pe.register_scene_sampler(lambda: _sample())
    assert pe.assert_scene_contract() is True
    pe.clear_scene_contract()
    pe.clear_scene_sampler()


def test_sampler_failure_also_blocks(monkeypatch=None):
    pe.clear_scene_contract()
    pe.register_scene_contract({"target_fingers": True, "target_table": True,
                                "fingers_table": True}, tag="stage2")

    def _boom():
        raise ValueError("读回失败")

    pe.register_scene_sampler(_boom)
    with pytest.raises(RuntimeError, match="SCENE_SAMPLER_FAILED"):
        pe.assert_scene_contract()
    pe.clear_scene_contract()
    pe.clear_scene_sampler()
