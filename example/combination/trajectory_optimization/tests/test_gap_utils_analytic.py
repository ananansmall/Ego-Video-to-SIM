"""距离实现的**解析真值**回归 (方案 §10.2) ✓

原则 ✓ (方案原话): "**校准不依赖同一错误函数生成期望值**" ✓
  ⇒ 期望值全部是**解析几何答案** ✓

⚠ 2026-09-26 更正 ✓: 第一版测试的**几何本身写错了** ✗
  (左盒 x 到 `-0.01`、右盒从 `0.0+g` 起 ⇒ 真实间隙 = `0.01+g` ⇒ 期望值全差 10mm ✗)
  ⇒ 本版: 右盒起点 = **左盒终点 + g** ✓ ⇒ 真实间隙**恰好 = g** ✓
"""

import numpy as np
import pytest

from gap_utils import convex_gap_min_mm, convex_gap_mm


def box(x0, x1, y0, y1, z0, z1):
    """长方体 8 顶点 (单位 m) ✓"""
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)],
                    dtype=float)


def rot_z(deg):
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ───────── ① 解析盒间隙: 0/3/5/10/20/40mm ✓ ─────────
@pytest.mark.parametrize("gap_mm", [0.0, 3.0, 5.0, 10.0, 20.0, 40.0])
def test_analytic_box_gap(gap_mm):
    g = gap_mm / 1000.0
    left = box(-0.06, -0.01, -0.02, 0.02, -0.02, 0.02)
    right = box(-0.01 + g, 0.04 + g, -0.02, 0.02, -0.02, 0.02)   # ★ 起点 = 左盒终点 + g ✓
    gap, hit = convex_gap_mm(left, right)
    assert abs(gap - gap_mm) < 0.05, f"解析真值 {gap_mm}mm, 实得 {gap:.4f}mm"
    assert hit is (gap_mm == 0.0)


# ───────── ② 非对称: 一侧 40mm / 另一侧 25mm ⇒ 各自算准 ✓ ─────────
def test_asymmetric_takes_min():
    tgt = box(-0.01, 0.01, -0.02, 0.02, -0.02, 0.02)
    far = box(-0.09, -0.05, -0.02, 0.02, -0.02, 0.02)     # 与目标 40mm ✓
    near = box(0.035, 0.075, -0.02, 0.02, -0.02, 0.02)    # 与目标 25mm ✓
    g1, _ = convex_gap_mm(tgt, far)
    g2, _ = convex_gap_mm(tgt, near)
    assert abs(g1 - 40.0) < 0.05, f"far 应 40mm, 实得 {g1:.4f}"
    assert abs(g2 - 25.0) < 0.05, f"near 应 25mm, 实得 {g2:.4f}"


# ───────── ③ 共同旋转 + 平移 ⇒ 距离不变 ✓ ─────────
def test_rigid_transform_invariance():
    left = box(-0.06, -0.01, -0.02, 0.02, -0.02, 0.02)
    right = box(0.03, 0.08, -0.02, 0.02, -0.02, 0.02)      # 间隙 40mm ✓
    base, _ = convex_gap_mm(left, right)
    R = rot_z(37.0)
    t = np.array([0.31, -0.17, 0.09])
    moved, _ = convex_gap_mm(left @ R.T + t, right @ R.T + t)
    assert abs(base - moved) < 0.05, f"刚体变换后距离应不变: {base:.4f} vs {moved:.4f}"


# ───────── ④ 分片空隙(凹槽)不得被填实 ✓ (方案 §10.2 反例B) ─────────
def test_concave_piece_gap_not_filled():
    h1 = box(-0.030, -0.020, -0.005, 0.005, -0.005, 0.005)
    h2 = box(0.020, 0.030, -0.005, 0.005, -0.005, 0.005)
    # 指头在凹槽(x∈[-0.005,0.005])内, 上表面 z=0.020 ⇒ 离分片顶面(z=0.005) **15mm** ✓
    finger = box(-0.005, 0.005, -0.002, 0.002, 0.020, 0.030)
    gap, hit, idx = convex_gap_min_mm([h1, h2], finger)
    # ⚠ 解析真值 ✓ (我第一版又写错了 ✗): 指头 x∈[-0.005,0.005] 与分片 x 区间**不重叠**
    #   ⇒ 最近点是**角点对**: 水平 15mm、垂直 15mm ⇒ 真实距离 = √(15²+15²) = **21.213mm** ✓
    _expect = float(np.sqrt(15.0 ** 2 + 15.0 ** 2))
    assert not hit, "不得判为相交 ✗ (旧实现把凹槽填实 ⇒ 0 并判相交 ✗)"
    assert abs(gap - _expect) < 0.05, f"应报 {_expect:.3f}mm(解析角点距), 实得 {gap:.4f}mm"
    assert idx in (0, 1)


# ───────── ⑤ 穿透 ⇒ gap=0 且 hit=True ✓ ─────────
def test_penetration_reports_zero_and_hit():
    a = box(-0.02, 0.02, -0.02, 0.02, -0.02, 0.02)
    b = box(0.01, 0.05, -0.02, 0.02, -0.02, 0.02)
    gap, hit = convex_gap_mm(a, b)
    assert hit is True and gap < 1e-6


# ───────── ⑥ 退化输入 ⇒ 明确报错 (不得当成 0 ✗) ─────────
def test_degenerate_raises():
    a = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    b = box(0.02, 0.03, -0.01, 0.01, -0.01, 0.01)
    with pytest.raises(ValueError):
        convex_gap_mm(a, b)
    with pytest.raises(ValueError):
        convex_gap_min_mm([], b)
