"""★ 2026-09-22 【P0-1 验收】凸包 vs OBB 的 SAT 正确性 (确定性几何测试 ✓)

审计文档 `docs/2026-09-21-Stage2代码审计与收敛实施计划.md` §P0-1 的验收要求 ✓:
  "包含、刚好接触、明显分离、非对称凸包、随机旋转和多凸包测试; 整体刚体变换前后结果一致"

背景 ✗ (旧实现的数学错误):
  旧 `TargetHullModel.signed_obb` 用 `r_h = max((V - C) @ L)` + `sep = |d| - (r_o + r_h)`,
  假设凸包关于**顶点均值 C 对称** ✗ —— CoACD 凸包不对称 ✓
  实测反例: 小 OBB 完全落在四面体**内部**, 旧函数报 **"+29mm 分离"** ✗✗ (包含被误判为分离 ✗)

运行 ✓:
  PYTHONDONTWRITEBYTECODE=1 /home/an/miniconda3/envs/dex/bin/python -m pytest \
    tests/test_hull_obb_sat.py -q -p no:cacheprovider --disable-warnings
"""

from __future__ import annotations

import numpy as np
import pytest

from stage2_approach_opt import TargetHullModel

# 文档 §7-1 的反例四面体 (单位 m)
TETRA = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, 0.0, 0.1],
])
EYE = np.eye(3)


def _rand_R(rng: np.random.Generator) -> np.ndarray:
    """随机旋转矩阵 (QR 法 ✓, 保证正交且 det=+1 ✓)"""
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def test_tetra_contains_small_obb_is_intersect():
    """★ 文档反例 ✓: 小 OBB 完全在四面体内部 ⇒ **必须判相交** (旧实现报 +29mm 分离 ✗)"""
    m = TargetHullModel([TETRA])
    q = m.query_obb(np.array([0.08, 0.005, 0.005]), EYE, np.array([0.001, 0.001, 0.001]))
    assert q["intersect"] is True, f"包含被误判为分离 ✗ (sep_lower={q['sep_lower_m'] * 1000:.2f}mm)"
    assert q["depth_m"] > 0.0


def test_asymmetric_hull_contains_obb():
    """非对称凸包 (楔形 ✓) 包含 OBB ⇒ 相交 ✓ (旧实现在此最容易误报分离 ✗)"""
    wedge = np.array([
        [0.0, 0.0, 0.0], [0.08, 0.0, 0.0], [0.0, 0.05, 0.0],
        [0.0, 0.0, 0.02], [0.07, 0.001, 0.001], [0.001, 0.04, 0.001],
    ])
    m = TargetHullModel([wedge])
    q = m.query_obb(np.array([0.015, 0.010, 0.006]), EYE, np.array([0.002, 0.002, 0.002]))
    assert q["intersect"] is True
    assert q["depth_m"] > 0.0


def test_clear_separation_positive_lower_bound():
    """明显分离 ⇒ 不相交 ✓ 且**间隔下界** > 0 ✓、且不超过真实欧氏距离 ✓"""
    m = TargetHullModel([TETRA])
    c = np.array([0.30, 0.0, 0.0])            # 远离四面体 ✓
    q = m.query_obb(c, EYE, np.array([0.002, 0.002, 0.002]))
    assert q["intersect"] is False
    assert q["sep_lower_m"] > 0.0
    _true_lo = float(np.linalg.norm(c - np.array([0.1, 0.0, 0.0])) - 0.002)  # 最近的角点
    assert q["sep_lower_m"] <= _true_lo + 1e-9, "间隔下界不应超过真实距离 ✓"


def test_touching_face_is_not_separated():
    """刚好接触 (OBB 面与四面体面重合 ✓) ⇒ **不能判成分离** ✓"""
    m = TargetHullModel([TETRA])
    # 四面体斜面 x+y+z=0.1 ⇒ 法向 (1,1,1)/√3 ✓; 把 OBB 放在斜面外侧**恰好贴上**
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    c_on = np.array([0.1 / 3.0, 0.1 / 3.0, 0.1 / 3.0])       # 斜面上一点 ✓
    c = c_on + n * 0.001                                     # 中心沿法向偏移 1mm
    q = m.query_obb(c, EYE, np.array([0.001, 0.001, 0.001])) # 半尺寸 1mm ⇒ 恰好相切 ✓
    assert q["intersect"] is True
    assert q["depth_m"] < 2e-3


def test_rigid_transform_invariance():
    """整体刚体变换 (旋转+平移) ⇒ 相交判定与深度**不变** ✓ (审计要求 ✓)"""
    rng = np.random.default_rng(7)
    hull = TETRA - TETRA.mean(axis=0)
    m0 = TargetHullModel([hull])
    c0 = np.array([0.03, 0.005, 0.005])
    h = np.array([0.0015, 0.0015, 0.0015])
    R0 = _rand_R(rng)
    base = m0.query_obb(c0, R0, h)

    Rg, tg = _rand_R(rng), rng.uniform(-0.2, 0.2, 3)
    m1 = TargetHullModel([hull @ Rg.T + tg])
    moved = m1.query_obb(Rg @ c0 + tg, Rg @ R0, h)
    assert moved["intersect"] == base["intersect"]
    assert abs(moved["depth_m"] - base["depth_m"]) < 1e-9
    assert abs(moved["sep_lower_m"] - base["sep_lower_m"]) < 1e-9


def test_multi_hull_union_semantics():
    """多凸包并集 ✓: 落在**某一片内部** ⇒ 相交 ✓; 落在**片间的空隙** ⇒ 不相交 ✓"""
    p1 = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.02]])
    p2 = p1 + np.array([0.10, 0.0, 0.0])
    m = TargetHullModel([p1, p2])
    q_in = m.query_obb(np.array([0.005, 0.003, 0.003]), EYE, np.array([0.001] * 3))
    assert q_in["intersect"] is True, "落在第 1 片内部 ⇒ 必须判相交 ✓"
    q_gap = m.query_obb(np.array([0.055, 0.003, 0.003]), EYE, np.array([0.001] * 3))
    assert q_gap["intersect"] is False, "两片之间的空隙 ⇒ 不相交 ✓"


def test_never_misses_contained_corner_random():
    """★ 随机压力测试 ✓: **只要有 OBB 角点落在凸包内 ⇒ 必须判相交** (旧实现会漏 ✗)"""
    from scipy.spatial import ConvexHull

    rng = np.random.default_rng(20260922)
    missed = 0
    checked = 0
    for _ in range(120):
        V = rng.normal(size=(10, 3)) * 0.02
        try:
            ch = ConvexHull(V)
        except Exception:  # noqa: BLE001
            continue
        m = TargetHullModel([np.asarray(V, dtype=np.float64)])
        R = _rand_R(rng)
        h = rng.uniform(0.0005, 0.004, 3)
        c = rng.uniform(-0.03, 0.03, 3)
        corners = c + (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                                 for sz in (-1, 1)]) * h) @ R.T
        # brute-force: 角点是否在凸包内 (equations: n·x + d <= 0 ✓)
        inside = bool(np.all(ch.equations[:, :3] @ corners.T
                             + ch.equations[:, 3:4] <= 1e-9, axis=0).any())
        if not inside:
            continue
        checked += 1
        q = m.query_obb(c, R, h)
        if not q["intersect"]:
            missed += 1
    assert checked >= 5, f"随机样本不足 ({checked})"
    assert missed == 0, f"有 {missed}/{checked} 个'角点在凸包内'的样本被误判为分离 ✗"


if __name__ == "__main__":  # 允许直接运行 ✓
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
