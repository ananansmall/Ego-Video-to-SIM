"""T07.2 反例: 接触统计**不得**重复累计 (不跑物理仿真 ✓)

复审实测缺陷: 同一子步有 2 个接触对(指1-物体 + 指2-物体)时,
  · `both_force_steps` / `both_force_steps_total` 被 **+2** ✗ (应为 +1)
  · `sum_imp` 把**子步累加器**重复加 N 次 ⇒ 力被高估 ✗
本测试用**伪造 scene** 直接调用生产探针 `_probe_substep_ground` ⇒ 不开物理场景、秒级 ✓
"""
import math

import physics_env as pe


class _P:
    def __init__(self, imp, sep):
        self.impulse = [imp, 0.0, 0.0]
        self.separation = sep
        self.normal = [1.0, 0.0, 0.0]   # ← 法向: 与 impulse 同向 ⇒ 法向冲量 = |impulse| ✓


class _B:
    def __init__(self, name):
        self.name = name          # ← `_substep_body_name` 的回退字段 ✓

    def get_name(self):
        return self.name


class _C:
    def __init__(self, b0, b1, pts):
        self.bodies = [b0, b1]
        self.points = pts


class _S:
    def __init__(self, contacts):
        self._c = contacts

    def get_contacts(self):
        return self._c


def _one_substep_two_pairs():
    """一个子步内: 指1-物体 + 指2-物体 两个接触对 (各自冲量均超阈值)."""
    pe.set_object_ground_target("obj")
    pe.reset_target_contact_log(True, "obj")
    b1 = _B("left_gripper_finger_link1")
    b2 = _B("left_gripper_finger_link2")
    ob = _B("obj")
    # 预填名字缓存 ⇒ 不需要真实 scene/actor ✓
    pe._SUBSTEP_NAME_CACHE.update({b1: b1.name, b2: b2.name, ob: ob.name})
    dt = float(pe.PHYSICS_TIMESTEP)
    imp = (0.05 * dt) * 20.0                       # 明显高于阈值 ✓
    c1 = _C(b1, ob, [_P(imp, 0.0)])
    c2 = _C(b2, ob, [_P(imp, 0.0)])
    pe._SUBSTEP_GROUND_LOG["substeps"] = 1
    pe._probe_substep_ground(_S([c1, c2]))
    return pe.get_target_contact_report(), imp


def test_bilateral_substep_counted_once_per_substep():
    rep, _ = _one_substep_two_pairs()
    assert int(rep["both_force_steps"]) == 1, (
        "一个子步只能记 1 个双侧受力子步, 实测 %s ✗" % rep["both_force_steps"])
    _tot = int(pe._TARGET_CONTACT_LOG.get("both_force_steps_total", -1))
    assert _tot == 1, (
        "单调累计也只能 +1, 实测 %s ✗" % _tot)


def test_sum_imp_not_double_counted():
    rep, imp = _one_substep_two_pairs()
    f1 = float((rep.get("finger1") or {}).get("sum_imp", -1.0))
    assert math.isclose(f1, imp, rel_tol=1e-6, abs_tol=1e-12), (
        "每子步 sum_imp 只应加一次(=%g), 实测 %g ✗ (重复累加 ⇒ 力被高估)" % (imp, f1))


def test_n_force_at_most_one_per_substep():
    rep, _ = _one_substep_two_pairs()
    for k in ("finger1", "finger2"):
        assert int((rep.get(k) or {}).get("n_force", -1)) == 1, (
            "%s 每子步 n_force 至多 +1 ✗" % k)


def test_bilateral_count_bounded_by_substeps():
    """有界自检: 双侧子步数 不得超过 子步数 (复审 I.1-3 ✓)."""
    rep, _ = _one_substep_two_pairs()
    assert int(rep["both_force_steps"]) <= int(rep.get("substeps", 1))
    assert int(pe._TARGET_CONTACT_LOG.get("overflow", 0)) == 0
