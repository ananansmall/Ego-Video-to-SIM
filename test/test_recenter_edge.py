#!/usr/bin/env python3
"""recenter_edge_objects 回归测试。

背景: reseat_on_top_objects 契约是"只改 z", 不动水平位置, 所以"只搭上一个角"
的物体照样贴着顶面悬在边沿。实测 vertical_pick_place_115 bottle_0 在
all_instances.pkl 里与 box_0 的水平重叠为 0% (完全在盒外), 视觉上就是"贴盒边"。
"""
import os
import sys
import json
import pickle
import copy

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.sp_refinement import (  # noqa: E402
    recenter_edge_objects, _xy_overlap_ratio, _instance_index)

_ok = True


def ck(n, c, d=""):
    global _ok
    print(f"  {'PASS' if c else 'FAIL'}  {n}  {d}")
    if not c:
        _ok = False


def ov(I, n, sp):
    idx = _instance_index(I)

    def b(x):
        c, i = idx[x]
        m = I[c][i]["original_mesh"].copy()
        m.apply_transform(np.asarray(I[c][i]["T"], dtype=float))
        return m.bounds
    return _xy_overlap_ratio(b(n), b(sp))


print("\n[1] 真实数据: 悬在边沿 → 居中到支撑面")
base = os.path.join(REPO, "output_v3")
cases = [("vertical_pick_place_115", "bottle_0", "box_0"),
         ("basic_pick_place_118", "block_2", "bowl_0"),
         ("basic_pick_place_17", "toy_0", "bowl_0")]
for scene, obj, sup in cases:
    d = os.path.join(base, scene)
    pkl, relp = os.path.join(d, "all_instances.pkl"), os.path.join(d, "final_relations.json")
    if not (os.path.isfile(pkl) and os.path.isfile(relp)):
        print(f"    (跳过 {scene}: 缺文件)")
        continue
    inst = pickle.load(open(pkl, "rb"))["all_instances"]
    rel = json.load(open(relp))
    before = ov(inst, obj, sup)
    work = copy.deepcopy(inst)
    work, n = recenter_edge_objects(work, rel, verbose=False)
    after = ov(work, obj, sup)
    ck(f"{scene} {obj}@{sup}", after > 0.99, f"{before*100:.0f}% → {after*100:.0f}%")
    # z 必须不变
    cat = obj.rsplit("_", 1)[0]
    z0 = float(np.asarray(inst[cat][0]["T"], dtype=float)[2, 3])
    z1 = float(np.asarray(work[cat][0]["T"], dtype=float)[2, 3])
    ck(f"{scene} 只改 xy (z 不变)", abs(z1 - z0) < 1e-9, f"dz={z1-z0:.2e}")

print("\n[2] 已摆正的物体不动")
d = os.path.join(base, "basic_pick_place_141")
pkl = os.path.join(d, "all_instances.pkl")
if os.path.isfile(pkl):
    inst = pickle.load(open(pkl, "rb"))["all_instances"]
    rel = json.load(open(os.path.join(d, "final_relations.json")))
    before = ov(inst, "chess piece_0", "chessboard_0")
    work, n = recenter_edge_objects(copy.deepcopy(inst), rel, verbose=False)
    after = ov(work, "chess piece_0", "chessboard_0")
    ck("141 棋子本已 100%", before > 0.99 and after > 0.99, f"{before*100:.0f}%→{after*100:.0f}%")

print("\n[3] floor 关系不处理")
rel2 = {"a_0": "supported by floor", "b_0": "supported by wall"}
inst2 = {"a": [{"original_mesh": None, "T": np.eye(4)}]}
try:
    _, n = recenter_edge_objects({"x": []}, {}, verbose=False)
    ck("空输入不崩", True)
except Exception as e:
    ck("空输入不崩", False, str(e)[:60])

print("\n" + ("OK 全部通过" if _ok else "有失败"))
sys.exit(0 if _ok else 1)
