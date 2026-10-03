"""凸体距离工具 (方案 §10.2) —— 单一实现, 可离线单测 ✓

为什么单独成模块 ✓ (方案要求 "校准不依赖同一错误函数生成期望值" ✓):
  · 之前距离实现是 `stage1_grasp_mixin` 内的**嵌套函数** ⇒ 无法单测 ✗,
    只能靠运行整管线才发现错(而且我的 GJK 把 40mm 报成 45.28mm ✗)
  · 现在抽到模块级 ⇒ 用**解析真值**(盒间隙 0/3/5/10/20/40mm 等)直接回归 ✓

实现选择 ✓: 半空间表示 + 小规模 QP
     min |a − b|²  s.t.  a∈A, b∈B
  ⇒ 对任意朝向、任意凸分片都成立 ✓ (不再依赖我那份有缺陷的 GJK ✗)
  ⇒ 每个目标分片 / 每个手指 shape **分别求** 再取最小 ⇒ 凹槽不会被填实 ✓
"""

from __future__ import annotations

import numpy as np

__all__ = ["convex_gap_mm", "convex_gap_min_mm", "convex_gap_points_mm"]


def convex_gap_mm(points_a, points_b):
    """两个凸多面体(顶点给出)的**最小距离**(mm) + 是否接触/相交 ✓

    参数 ✓: `points_a`, `points_b` 形如 (N,3) 的顶点数组 (世界系 ✓, 单位 m)
    返回 ✓: `(gap_mm, hit)`
        · `gap_mm` = 最小表面间隙 (mm ✓; 接触/相交时 **0.0**)
        · `hit`    = True ⇒ 接触或相交 ✓
    取不到 / 退化 ⇒ 抛异常 ✓ (调用方应记 **INVALID**, **不得**当成 0 ✗ —— 方案 §10.2)
    """
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull

    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or len(a) < 4 or len(b) < 4:
        raise ValueError("点数量不足, 无法构成凸包")

    # ① 半空间表示: n·x + d ≤ 0 ✓ (凸包的 H-表示 ✓)
    eq_a = ConvexHull(a).equations
    eq_b = ConvexHull(b).equations

    # ② 目标: min |x_a − x_b|² ✓ ; 约束: 两侧都必须在各自半空间内 ✓
    def _c_a(x):
        return -(eq_a[:, :3] @ x[0:3] + eq_a[:, 3])

    def _c_b(x):
        return -(eq_b[:, :3] @ x[3:6] + eq_b[:, 3])

    def _obj(x):
        d = x[0:3] - x[3:6]
        return float(d @ d)

    x0 = np.concatenate([a.mean(axis=0), b.mean(axis=0)])
    res = minimize(_obj, x0, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": _c_a},
                                {"type": "ineq", "fun": _c_b}],
                   options={"maxiter": 400, "ftol": 1e-12})

    # ③ 收敛性检查 ✓ (方案 §10.2: **未收敛不得当真值** ✗ ⇒ 抛错让调用方记 INVALID ✓)
    if not getattr(res, "success", False) and float(np.sqrt(max(_obj(res.x), 0.0))) > 1e-6:
        raise RuntimeError(f"距离求解未收敛: {getattr(res, 'message', 'unknown')}")

    gap_m = float(np.sqrt(max(_obj(res.x), 0.0)))
    return gap_m * 1000.0, bool(gap_m < 1e-9)


def convex_gap_points_mm(points_a, points_b):
    """同 `convex_gap_mm`, 但**同时返回最近点对** ✓ (方案 §10.5-② 要求 ✓)

    返回 ✓: `(gap_mm, hit, pa_m, pb_m)` —— `pa/pb` 为各自表面上的最近点(世界系 m ✓)
    """
    from scipy.optimize import minimize
    from scipy.spatial import ConvexHull

    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or len(a) < 4 or len(b) < 4:
        raise ValueError("点数量不足, 无法构成凸包")
    eq_a = ConvexHull(a).equations
    eq_b = ConvexHull(b).equations

    def _c_a(x):
        return -(eq_a[:, :3] @ x[0:3] + eq_a[:, 3])

    def _c_b(x):
        return -(eq_b[:, :3] @ x[3:6] + eq_b[:, 3])

    def _obj(x):
        d = x[0:3] - x[3:6]
        return float(d @ d)

    x0 = np.concatenate([a.mean(axis=0), b.mean(axis=0)])
    res = minimize(_obj, x0, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": _c_a},
                                {"type": "ineq", "fun": _c_b}],
                   options={"maxiter": 400, "ftol": 1e-12})
    if not getattr(res, "success", False) and float(np.sqrt(max(_obj(res.x), 0.0))) > 1e-6:
        raise RuntimeError(f"距离求解未收敛: {getattr(res, 'message', 'unknown')}")
    pa, pb = res.x[0:3], res.x[3:6]
    gap_m = float(np.linalg.norm(pa - pb))
    return gap_m * 1000.0, bool(gap_m < 1e-9), pa, pb


def convex_gap_min_mm(hulls_a, points_b):
    """**逐分片**求最小 (方案 §10.2 反例B ✓): 目标多个凸分片 × 手指一个 shape ✓

    返回 ✓: `(gap_mm, hit, idx_min)`; `hulls_a` 为空 ⇒ 抛 `ValueError` ✓
    """
    if not hulls_a:
        raise ValueError("hulls_a 为空")
    best, best_i, hit_any = None, -1, False
    for i, h in enumerate(hulls_a):
        gap, hit = convex_gap_mm(h, points_b)
        if best is None or gap < best:
            best, best_i = float(gap), int(i)
        hit_any = hit_any or bool(hit)
    return float(best), bool(hit_any), int(best_i)
