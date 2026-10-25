#!/usr/bin/env python3
"""separate_object_penetrations 回归测试。

背景: 三层都没管"物体之间"的穿模 ——
  Stage1 prompt 的 "If unsure, default to: supported by floor"
  (tools/generate_scene_json_stage1.py:225) 让俯视桌面场景所有物体都标成 floor;
  基础精修只管单物体贴地; 5.2 的 resolve_penetrations 带
  only_refine_other_objects=True 跳过 floor。
实测 basic_pick_place_111 lego_0/lego_1 重叠 85%、46 的 box_1/box_2 92% 与
card_0 埋进 box_1 91.6%、118 的 croissant 埋进 pillow 68%。

本测试锁住:
  1. 完全重叠的两物体被推开到阈值以下
  2. 无穿模时不乱动 (0 次)
  3. 只改 xy, z/旋转/尺寸不变
  4. 完全同心也能分开 (不死循环)
  5. 部分重叠 (6.8%) 也能分开
  6. 三物体连环穿模全部解决
  7. 真实数据 basic_pick_place_46 (92%/73%/24.8% → 全部 < 阈值)

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_separate_penetration.py
"""
import os
import sys
import copy
import numpy as np
import trimesh

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.sp_refinement import separate_object_penetrations  # noqa: E402

_ok = True


def check(name, cond, detail=""):
    global _ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok = False


def mk(xy, z=0.0, ext=(0.05, 0.05, 0.05)):
    m = trimesh.creation.box(extents=list(ext))
    m.apply_translation([0, 0, ext[2] / 2])
    T = np.eye(4)
    T[:3, 3] = [xy[0], xy[1], z]
    return {"original_mesh": m, "T": T}


def pen(a, b):
    ma = a["original_mesh"].copy()
    ma.apply_transform(np.asarray(a["T"], dtype=float))
    mb = b["original_mesh"].copy()
    mb.apply_transform(np.asarray(b["T"], dtype=float))
    pa, pb = np.asarray(ma.vertices), np.asarray(mb.vertices)
    lo, hi = mb.bounds[0], mb.bounds[1]
    ra = float(np.all((pa >= lo) & (pa <= hi), axis=-1).mean())
    lo, hi = ma.bounds[0], ma.bounds[1]
    rb = float(np.all((pb >= lo) & (pb <= hi), axis=-1).mean())
    return max(ra, rb)


print("\n[1] 完全重叠 → 推开到阈值以下 (复刻 111 的 lego 85%)")
insts = {"lego": [mk((0, 0)), mk((0, 0))]}
before = pen(insts["lego"][0], insts["lego"][1])
out, n = separate_object_penetrations(copy.deepcopy(insts), {"lego_0": "supported by floor"},
                                       verbose=False)
after = pen(out["lego"][0], out["lego"][1])
check("修复前重叠 >50%", before > 0.5, f"{before*100:.1f}%")
check("修复后 < 2%", after < 0.02, f"{after*100:.1f}%")
check("确有推开", n > 0, f"n={n}")

print("\n[2] 无穿模时不动")
insts = {"a": [mk((0, 0))], "b": [mk((1.0, 1.0))]}
out, n = separate_object_penetrations(copy.deepcopy(insts), {}, verbose=False)
check("0 次推开", n == 0, f"n={n}")
check("位置未动", np.allclose(out["a"][0]["T"][:3, 3], [0, 0, 0]))

print("\n[3] 只改 xy, z/旋转/尺寸不变")
insts = {"lego": [mk((0, 0)), mk((0, 0))]}
T0 = np.asarray(insts["lego"][0]["T"], dtype=float).copy()
out, n = separate_object_penetrations(copy.deepcopy(insts), {}, verbose=False)
T1 = np.asarray(out["lego"][0]["T"], dtype=float)
check("z 未变", abs(T1[2, 3] - T0[2, 3]) < 1e-12, f"{T0[2,3]} → {T1[2,3]}")
check("旋转未变", np.allclose(T1[:3, :3], T0[:3, :3]))
check("xy 变了", abs(T1[0, 3] - T0[0, 3]) + abs(T1[1, 3] - T0[1, 3]) > 1e-6)

print("\n[4] 完全同心也能分开 (不死循环)")
insts = {"a": [mk((0.5, 0.5))], "b": [mk((0.5, 0.5))]}
out, n = separate_object_penetrations(copy.deepcopy(insts), {}, verbose=False, max_iter=40)
check("分开了", pen(out["a"][0], out["b"][0]) < 0.02,
      f"{pen(out['a'][0], out['b'][0])*100:.1f}%")

print("\n[5] 部分重叠 (复刻 14 的 screwdriver/bowl 6.8%)")
insts = {"bowl": [mk((0, 0), ext=(0.1, 0.1, 0.15))],
         "screw": [mk((0.045, 0), ext=(0.02, 0.02, 0.045))]}
b = pen(insts["bowl"][0], insts["screw"][0])
out, n = separate_object_penetrations(copy.deepcopy(insts), {}, verbose=False)
a = pen(out["bowl"][0], out["screw"][0])
check("有重叠 (>2%)", b > 0.02, f"{b*100:.1f}%")
check("分开了", a < 0.02, f"{a*100:.1f}%")

print("\n[6] 三物体连环穿模")
insts = {"a": [mk((0, 0))], "b": [mk((0, 0))], "c": [mk((0, 0))]}
out, n = separate_object_penetrations(copy.deepcopy(insts), {}, verbose=False)
r_ab = pen(out["a"][0], out["b"][0])
r_ac = pen(out["a"][0], out["c"][0])
r_bc = pen(out["b"][0], out["c"][0])
check("ab 分开", r_ab < 0.02, f"{r_ab*100:.1f}%")
check("ac 分开", r_ac < 0.02, f"{r_ac*100:.1f}%")
check("bc 分开", r_bc < 0.02, f"{r_bc*100:.1f}%")

print("\n[7] 真实数据 basic_pick_place_46")
p = os.path.join(REPO, "output_v3", "basic_pick_place_46", "all_instances.pkl")
if os.path.isfile(p):
    import pickle
    d = pickle.load(open(p, "rb"))["all_instances"]
    out, n = separate_object_penetrations(copy.deepcopy(d), {}, verbose=False)
    items = [(c, i, out[c][i]) for c in out for i in range(len(out[c]))]
    worst = 0.0
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            worst = max(worst, pen(items[x][2], items[y][2]))
    check("最大重叠 < 2%", worst < 0.02, f"worst={worst*100:.1f}% (修复前 92.0%)")
    print(f"      (推开 {n} 次)")
else:
    print("    (pkl 不存在, 跳过)")

print("\n" + ("✅ 全部通过" if _ok else "❌ 有失败"))
sys.exit(0 if _ok else 1)
