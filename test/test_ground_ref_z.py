#!/usr/bin/env python3
"""ground_floating_objects 的 ref_z 修复回归测试。

背景: float_z_max(0.5) 的上界保护无法区分两种"高空悬停":
  (a) 坐标系未对齐 —— 物体从基础精修起就高
  (b) 被上游推高   —— 基础精修时贴地, 但 5.2 穿模修复把它推上去了
实测 basic_pick_place_17: toy_1 基础精修 z_min=0.0000 → Stage5 变 0.8806m,
toy_5 由 0 → 0.6379m, 二者都因超 0.5 被跳过, 渲染时飘在空中。

修复: 传入 ref_z (基础精修后的 z_min) 区分两种情况。

本测试锁住:
  1. 参考值≈0 + 当前高出 ref_tol → 判为被推高, 压回地面
  2. 参考值本身就高 → 判为坐标系未对齐, 保持跳过
  3. 不传 ref_z → 维持原行为 (跳过), 向后兼容
  4. ref_tol 生效: 高出幅度小于 ref_tol 时不算被推高
  5. 复刻 basic_pick_place_17 的真实数值 (toy_1 0.8806 / toy_5 0.6379)

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_ground_ref_z.py
"""
import os
import sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.sp_refinement import ground_floating_objects  # noqa: E402

_ok = True


def check(name, cond, detail=""):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok = False


def make(z_off):
    import trimesh
    m = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    m.apply_translation([0, 0, 0.05])           # 底面在 z=0
    T = np.eye(4)
    T[:3, 3] = [0.4, -0.1, z_off]
    return {"original_mesh": m, "T": T}


def zmin(inst):
    import trimesh
    m = inst["original_mesh"].copy()
    m.apply_transform(np.asarray(inst["T"], dtype=float))
    return float(m.bounds[0, 2])


print("\n[1] 被上游推高 → 压回地面 (复刻 17 的 toy_1: 0 → 0.8806)")
insts = {"toy": [make(0.8806)]}
out, n = ground_floating_objects(
    insts, {"toy_0": "supported by floor"}, ref_z={"toy_0": 0.0000}, verbose=False)
check("压地计数 = 1", n == 1, f"n={n}")
check("z_min 归零", abs(zmin(out["toy"][0])) < 1e-6, f"z_min={zmin(out['toy'][0]):+.6f}")

print("\n[2] 坐标系未对齐 (参考值本身也高) → 保持跳过")
insts = {"toy": [make(0.8806)]}
out, n = ground_floating_objects(
    insts, {"toy_0": "supported by floor"}, ref_z={"toy_0": 0.9000}, verbose=False)
check("计数 = 0", n == 0, f"n={n}")
check("保持原位", abs(out["toy"][0]["T"][2, 3] - 0.8806) < 1e-9,
      f"z={out['toy'][0]['T'][2,3]:+.4f}")

print("\n[3] 不传 ref_z → 维持原行为 (跳过), 向后兼容")
insts = {"toy": [make(0.8806)]}
out, n = ground_floating_objects(
    insts, {"toy_0": "supported by floor"}, ref_z=None, verbose=False)
check("计数 = 0", n == 0, f"n={n}")

print("\n[4] ref_tol 生效 (参考值≈0 时按高出幅度判断)")
# 刚超过 float_z_max 也能识别为"被推高" (差 0.51 > tol 0.10)
insts = {"toy": [make(0.51)]}
out, n = ground_floating_objects(
    insts, {"toy_0": "supported by floor"},
    ref_z={"toy_0": 0.0}, ref_tol=0.10, verbose=False)
check("z=0.51, ref=0 → 压回", n == 1, f"n={n}")

# 参考值本身不接近地面 → 视为坐标系问题, 即使高出很多也跳过
insts = {"toy": [make(0.88)]}
out, n = ground_floating_objects(
    insts, {"toy_0": "supported by floor"},
    ref_z={"toy_0": 0.48}, ref_tol=0.10, verbose=False)
check("ref=0.48 不≈0 → 跳过 (坐标系问题)", n == 0, f"n={n}")

print("\n[5] 复刻 17 真实数值 (toy_1 0.8806 / toy_5 0.6379, 参考都为 0)")
insts = {"toy": [make(0.8806), make(0.6379)]}
out, n = ground_floating_objects(
    insts, {"toy_0": "supported by floor", "toy_1": "supported by floor"},
    ref_z={"toy_0": 0.0, "toy_1": 0.0}, verbose=False)
check("两个都压回", n == 2, f"n={n}")
check("toy_0 归零", abs(zmin(out["toy"][0])) < 1e-6)
check("toy_1 归零", abs(zmin(out["toy"][1])) < 1e-6)

print("\n[6] 低空悬空 (未超 float_z_max) 不受 ref_z 影响")
insts = {"toy": [make(0.3)]}
out, n = ground_floating_objects(
    insts, {"toy_0": "supported by floor"}, ref_z={"toy_0": 0.0}, verbose=False)
check("照常压回", n == 1, f"n={n}")

print("\n[7] 陷地分支不受 ref_z 影响")
insts = {"card": [make(-0.049)]}
out, n = ground_floating_objects(
    insts, {"card_0": "supported by floor"}, ref_z={"card_0": 0.5}, verbose=False)
check("陷地照常压回", n == 1 and abs(zmin(out["card"][0])) < 1e-6, f"n={n}")

print("\n" + ("✅ 全部通过" if _ok else "❌ 有失败"))
sys.exit(0 if _ok else 1)
