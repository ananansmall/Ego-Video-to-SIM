"""N-5 接近段优化的**可行性阻断**验收测试 (2026-09-21).

对应用户静态审查的两个 P0 复现样例 ✓:

  A. "**只撞桌、不碰目标**" ✗
     审查原文: "完整候选检查: feasible=False, **桌面穿透 14.2mm**;
               优化器返回: 零偏移, reason='already clear'" ✗
     ⇒ 期望: 早退**不再**发生 (必须继续搜偏移 ✓)

  B. "**全部候选不可行**" ✗
     审查原文: "全部候选不可行时仍选一个最小代价结果, 调用方仍可能应用偏移" ✗
     ⇒ 期望: 返回 `feasible=False` + `reason='no_feasible_candidate'` ✓ (显式失败 ✓)

纯几何, **不启动仿真** ✓ (毫秒级 ✓).
用法: /home/an/miniconda3/envs/dex/bin/python tests/test_n5_infeasible.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stage2_approach_opt import evaluate_candidate, optimize_approach  # noqa: E402


def _approach(z: float = 0.10, q: float = 0.03, n: int = 6):
    """沿 +x 直线接近的控制点 (u 从 1/n → 1) ✓"""
    ctrl = []
    for i in range(n):
        u = (i + 1) / n
        ctrl.append({
            "p": np.array([0.30 - 0.15 * u, 0.0, z], dtype=np.float64),
            "R": np.eye(3, dtype=np.float64),
            "q": float(q),
            "u": float(u),
        })
    return ctrl


def case_A_table_only() -> None:
    """A: 目标物**不在场** (无目标穿透), 但零偏移会撞桌 ⇒ **不得早退** ✓"""
    obj_c = np.array([9.0, 9.0, 9.0])          # 远 ⇒ 目标侧零穿透 ✓
    obj_h = np.array([0.02, 0.02, 0.02])
    # 桌面: 上表面 z = 0.115 (手指盒半厚 0.0185, 夹爪中心 z=0.10 ⇒ 会穿透桌面 ✓)
    table = (np.array([0.30, 0.0, 0.115 - 0.05]), np.array([0.5, 0.5, 0.05]))
    ctrl = _approach(z=0.10)
    _c0, d0 = evaluate_candidate(0.0, 0.0, ctrl, obj_c, obj_h, None, obstacles=[table])
    print(f"[A] 零偏移完整检查: feasible={d0.get('feasible')} "
          f"obst_pen={float(d0.get('obst_pen_mm', -1)):.2f}mm "
          f"pen_real={float(d0.get('pen_real_mm', -1)):.2f}mm")
    assert d0.get("feasible") is False, "构造失败: 该样例应当是**不可行**的 ✗"
    (a_y, a_z, a_x), rep = optimize_approach(
        ctrl, obj_c, obj_h, None, obstacles=[table],
        a_y_range=(-0.03, 0.03), a_z_range=(-0.02, 0.02), n_y=7, n_z=5, refine_rounds=1)
    print(f"[A] 优化器返回: reason={rep.get('reason')!r} a=({a_y * 1e3:.1f},"
          f"{a_z * 1e3:.1f},{a_x * 1e3:.1f})mm feasible={rep.get('feasible')}")
    assert rep.get("reason") != "already clear", (
        "★ P0 未修: 目标无穿透就早退, **漏掉桌面碰撞** ✗ (审查复现样例 A)")
    print("[A] ✓ 不再早退 (已检查桌面)")

    # 若确实搜到可行解 ⇒ 必须标记可行 ✓; 否则必须显式失败 ✓ (两者都不许"静默不可行交付" ✗)
    if rep.get("feasible"):
        assert abs(a_z) > 1e-6 or abs(a_y) > 1e-6 or abs(a_x) > 1e-6, (
            "标记可行却给出零偏移 ⇒ 自相矛盾 ✗")
        print("[A] ✓ 找到可行偏移")
    else:
        assert rep.get("reason") == "no_feasible_candidate", "不可行时必须显式失败 ✗"
        print("[A] ✓ 显式失败 (无可行解)")


def case_B_all_infeasible() -> None:
    """B: 障碍物**包住整个搜索体** ⇒ 可行集必空 ⇒ 必须显式失败 ✓"""
    obj_c = np.array([9.0, 9.0, 9.0])
    obj_h = np.array([0.02, 0.02, 0.02])
    # 一个**巨块**把接近段全包住 ⇒ 任何 (a_y,a_z,a_x) 都必然穿透 ✓
    huge = (np.array([0.30, 0.0, 0.10]), np.array([5.0, 5.0, 5.0]))
    ctrl = _approach(z=0.10)
    _c0, d0 = evaluate_candidate(0.0, 0.0, ctrl, obj_c, obj_h, None, obstacles=[huge])
    assert d0.get("feasible") is False, "构造失败: 巨块样例应当不可行 ✗"
    (a_y, a_z, a_x), rep = optimize_approach(
        ctrl, obj_c, obj_h, None, obstacles=[huge],
        a_y_range=(-0.03, 0.03), a_z_range=(-0.02, 0.02), n_y=7, n_z=5, refine_rounds=1)
    print(f"[B] 优化器返回: reason={rep.get('reason')!r} feasible={rep.get('feasible')}")
    assert rep.get("feasible") is False, "★ P0 未修: 全部候选不可行却仍标记可行 ✗"
    assert rep.get("reason") == "no_feasible_candidate", (
        f"★ P0 未修: 不可行时应显式失败, 实际 reason={rep.get('reason')!r} ✗")
    assert (abs(a_y) + abs(a_z) + abs(a_x)) == 0.0, "失败时不应返回偏移 ✗"
    print("[B] ✓ 显式失败且零偏移 (调用方可据此拒绝交付 ✓)")


def main() -> int:
    print("=" * 74)
    print("N-5 可行性阻断验收 (审查复现样例 A/B)")
    print("=" * 74)
    print(f"容差: _N5_PEN_TOL_MM = {os.environ.get('_N5_PEN_TOL_MM', '0.5')} (默认 0.5mm ✓)")
    case_A_table_only()
    case_B_all_infeasible()
    print("\n全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
