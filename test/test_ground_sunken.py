#!/usr/bin/env python3
"""ground_floating_objects 陷地修正回归测试。

背景: basic_pick_place_46 的 card_0 被 5.1 从 'supported by other objects'
误判为 'supported by floor', 导致 Stage3 的 refine_supported_by_floor_object
没执行, 物体保持 VGGT 原始位置 z_min=-0.049m, audit 标 sunken。
原 ground_floating_objects 的 `if z_min <= float_eps: continue` 把陷地和
正常贴地一起跳过, 所以 card_0 没被修。

本测试锁住:
  1. 陷地 (z_min < -eps) 被压回 z=0
  2. 正常贴地 (|z_min| <= eps) 不动
  3. 悬空且超 float_z_max 仍跳过 (坐标系保护不被破坏)
  4. 悬空未超上界仍压到地面 (原功能不回退)
  5. 只改 T[2,3], 不动旋转/xy/mesh
  6. 非 floor 关系一律不碰

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_ground_sunken.py
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


def make_inst(z_off, tag):
    """造一个 10cm 立方体, T 把它平移到 (1, 2, z_off)。"""
    import trimesh
    m = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    m.apply_translation([0, 0, 0.05])          # 底面在 z=0
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, z_off]
    return {"original_mesh": m, "T": T, "tag": tag}


def zmin(inst):
    import trimesh
    m = inst["original_mesh"].copy()
    m.apply_transform(np.asarray(inst["T"], dtype=float))
    return float(m.bounds[0, 2])


print("\n[1] 陷地被压回地面 (对应 card_0 z_min=-0.049)")
insts = {"card": [make_inst(-0.049, "sunken")]}
rel = {"card_0": "supported by floor"}
out, n = ground_floating_objects(insts, rel, verbose=False)
check("压地计数 = 1", n == 1, f"n={n}")
check("z_min 归零", abs(zmin(out["card"][0])) < 1e-6, f"z_min={zmin(out['card'][0]):+.6f}")
check("xy 未动", abs(out["card"][0]["T"][0, 3] - 1.0) < 1e-9
      and abs(out["card"][0]["T"][1, 3] - 2.0) < 1e-9)
check("旋转未动", np.allclose(out["card"][0]["T"][:3, :3], np.eye(3)))

print("\n[2] 正常贴地不动 (|z_min| <= float_eps)")
insts = {"box": [make_inst(0.0, "on_floor")]}
out, n = ground_floating_objects(insts, {"box_0": "supported by floor"}, verbose=False)
check("计数 = 0", n == 0, f"n={n}")
check("位置未动", abs(out["box"][0]["T"][2, 3] - 0.0) < 1e-9)

print("\n[3] 悬空超 float_z_max 仍跳过 (坐标系保护)")
insts = {"toy": [make_inst(0.7, "high")]}
out, n = ground_floating_objects(insts, {"toy_0": "supported by floor"},
                                 float_z_max=0.5, verbose=False)
check("计数 = 0", n == 0, f"n={n}")
check("保持原位", abs(out["toy"][0]["T"][2, 3] - 0.7) < 1e-9)

print("\n[4] 悬空未超上界仍压地 (原功能不回退)")
insts = {"toy": [make_inst(0.3, "low_float")]}
out, n = ground_floating_objects(insts, {"toy_0": "supported by floor"},
                                 float_z_max=0.5, verbose=False)
check("计数 = 1", n == 1, f"n={n}")
check("z_min 归零", abs(zmin(out["toy"][0])) < 1e-6, f"z_min={zmin(out['toy'][0]):+.6f}")

print("\n[5] 非 floor 关系不碰")
for r in ["supported by other objects", "attached to wall", "held by hand",
          "supported by box_0"]:
    insts = {"x": [make_inst(-0.049, "buried")]}
    out, n = ground_floating_objects(insts, {"x_0": r}, verbose=False)
    check(f"'{r}' 不动", n == 0 and abs(out["x"][0]["T"][2, 3] + 0.049) < 1e-9,
          f"n={n} z={out['x'][0]['T'][2,3]:+.4f}")

print("\n[6] 关系走类别级回退 (refined_relations 只有 category)")
# 注意: 陷地量必须明显大于 float_eps (默认 0.02), 否则会被当成"已贴地"跳过
insts = {"card": [make_inst(-0.049, "cat_rel")]}
out, n = ground_floating_objects(insts, {"card": "supported by floor"}, verbose=False)
check("按类别级关系也修", n == 1, f"n={n}")

print("\n[7] 陷地量小于 float_eps 视为已贴地 (浮点噪声, 不动)")
insts = {"box": [make_inst(-0.01, "tiny")]}
out, n = ground_floating_objects(insts, {"box_0": "supported by floor"},
                                 float_eps=0.02, verbose=False)
check("计数 = 0", n == 0, f"n={n}")
check("位置未动", abs(out["box"][0]["T"][2, 3] + 0.01) < 1e-9)

print("\n" + ("✅ 全部通过" if _ok else "❌ 有失败"))
sys.exit(0 if _ok else 1)
