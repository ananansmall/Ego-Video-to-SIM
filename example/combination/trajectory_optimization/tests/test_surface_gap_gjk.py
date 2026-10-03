"""★ 2026-09-23 【方案 §11.2 反例 A / §11.6-A】**真实表面间隙**的校准测试 ✓

复审给出的合成反例 ✓ (原话):
    目标：x∈[-10,10]，y,z∈[-5,5]
    左指：x∈[-20,-10]，y,z∈[-20,20]
    右指：x∈[10,20]，  y,z∈[-20,20]
    两指内侧面分别与目标 x=-10、x=10 的面接触 ⇒ **真实表面间隙 = 0**
    **旧实现返回 dmin = 21.213203435596427** ✗✗ (最近点在面内, 顶点↔顶点算不到 ✗)

⇒ 本测试锁住: 已接触必须报 **≈0** ✓; 有已知间隙必须报该间隙 ✓
   （未通过 ⇒ 不得启动 CEM ✓ 方案 §11.6-A ✓）
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1_grasp_mixin import Stage1Mixin  # noqa: E402


def _box_verts(x0, x1, y0, y1, z0, z1):
    """轴对齐盒的 8 个顶点 (米 ✓)"""
    v = np.array([[x, y, z]
                  for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
                 dtype=np.float64)
    return v / 1000.0


def _measure(target_v, finger_vs):
    """用哑元 self 调 `_measure_grasp_symmetry` (只依赖两个顶点来源 ✓)"""
    class _D(object):
        pass

    d = _D()
    d._coacd_world_hulls = lambda _n: [target_v]
    d._finger_world_vertices = lambda _r=None: list(finger_vs)
    return Stage1Mixin._measure_grasp_symmetry(d, "test_obj")


def test_counterexample_contact_reports_zero_gap():
    """★ 复审反例 A: 面内接触 ⇒ 真实间隙 **0** (旧实现 21.21mm ✗)"""
    tgt = _box_verts(-10, 10, -5, 5, -5, 5)
    left = _box_verts(-20, -10, -20, 20, -20, 20)
    right = _box_verts(10, 20, -20, 20, -20, 20)
    asym, dmin = _measure(tgt, [left, right])
    assert dmin is not None, "dmin 不应为 None (合成输入可计算 ✓)"
    assert asym is not None
    assert abs(asym) < 1e-3, f"两侧应对称 ⇒ asym≈0, 实得 {asym}"
    assert dmin < 1e-3, (
        f"已接触 ⇒ 真实表面间隙应 ≈0 mm (旧实现会报 21.21mm ✗), 实得 {dmin:.4f} mm")


def test_known_gap_3mm_is_reported():
    """已知 3mm 间隙 ⇒ 必须报 ≈3mm (校验量值正确, 不只是 0 ✓)"""
    tgt = _box_verts(-10, 10, -5, 5, -5, 5)
    left = _box_verts(-23, -13, -20, 20, -20, 20)     # 内侧面在 x=-13 ⇒ 离目标 -10 差 3mm ✓
    right = _box_verts(13, 23, -20, 20, -20, 20)      # 内侧面在 x=13 ⇒ 差 3mm ✓
    asym, dmin = _measure(tgt, [left, right])
    assert dmin is not None
    assert abs(dmin - 3.0) < 0.05, f"应报 ≈3.00mm, 实得 {dmin:.4f} mm"
    assert abs(asym) < 1e-3, f"两侧间隙应相等 ⇒ asym≈0, 实得 {asym}"


def test_far_apart_reports_large_gap():
    """明显分离 ⇒ 必须报大间隙 (不是恒 0 ✓)"""
    tgt = _box_verts(-10, 10, -5, 5, -5, 5)
    left = _box_verts(-60, -50, -20, 20, -20, 20)     # 内侧面 x=-50 ⇒ 间隙 40mm ✓
    right = _box_verts(50, 60, -20, 20, -20, 20)
    asym, dmin = _measure(tgt, [left, right])
    # ⚠ 期望值**不是** 40: 目标的 y,z 只到 ±5, 而手指 y,z 到 ±20 ⇒
    #   最近的是**角点**间距 = √(40² + 15² + 15²) ≈ 45.28 mm ✓ (旧顶点↔顶点也会算这个值 ✓)
    #   ⇒ 这一条只校验"分离时不为恒 0 且量级正确" ✓
    assert dmin is not None
    # ★ W5 (2026-09-26) 复审反例 ✓: 真实间隙 **40mm** 曾被报成 45.28mm ✗
    #   (旧测试把错误值当成正确答案 ✗ ⇒ "4 passed" 不能证明量法可靠 ✓)
    #   ⇒ 这里改为**按复审给出的真值 40mm** 断言 ⇒ 若实现有误会立刻失败 ✓
    assert abs(dmin - 40.0) < 0.5, (
        f"应报 **≈40.0mm**(真实间隙 ✓, 复审给定), 实得 {dmin:.4f} mm")


def test_asymmetric_case_reports_difference():
    """一侧贴住、另一侧 5mm ⇒ asym 必须报 ≈5mm (对称性检验 ✓)"""
    tgt = _box_verts(-10, 10, -5, 5, -5, 5)
    left = _box_verts(-20, -10, -20, 20, -20, 20)     # 接触 ⇒ 0 ✓
    right = _box_verts(15, 25, -20, 20, -20, 20)      # 内侧面 x=15 ⇒ 5mm ✓
    asym, dmin = _measure(tgt, [left, right])
    assert dmin is not None
    assert abs(asym - 5.0) < 0.05, f"asym 应 ≈5.00mm, 实得 {asym:.4f} mm"
    assert dmin < 1e-3, f"最近侧应 ≈0, 实得 {dmin:.4f} mm"
