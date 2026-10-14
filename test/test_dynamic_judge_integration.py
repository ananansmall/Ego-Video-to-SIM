#!/usr/bin/env python3
"""Stage 3.7 (Pick/Place) 集成回归测试。

mainv3 已把动态判定从自带的三条件换成 Stage 3.7 的 pick/place 结果:
    _pp_key = f"{category}_inst{inst_idx}"
    is_dynamic_inst = _pp_key in _pp_dynamic

本测试锁住集成时最容易错的三个点:
  1. key 命名一致 —— pick_place 输出 `{cat}_inst{i}`, mainv3 的拼接必须一模一样,
     差一个字 `_pp_key in _pp_dynamic` 就恒为假, 表现是"全部漏检"而没有任何报错。
  2. 动态召回 —— 5 个有 GT 的场景必须 5/5 命中 (对比: 旧三条件是 0/5)。
  3. frame_ratio —— 从 coordinate_alignment.json 的 total_frames 推导; 缺该字段
     会退化成 1.0, pick/place 帧号随之差几倍 (实测 ratio 2.29~4.93)。

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_dynamic_judge_integration.py
"""
import os
import sys
import json
import pickle
import shutil
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.pick_place_detector import run_stage3_7_pick_place  # noqa: E402

BASE = os.path.join(REPO, "output_v2")
SCENES = {
    "2chair-bottle_vggt_omega":           (2.29, {"bottle_inst0"}),
    "7_vggt_omega":                       (1.00, {"toy_inst1"}),
    "3chair-bottle_vggt_omega":           (4.93, {"bottle_inst0"}),
    "121_C5_CellPhone_161deg_vggt_omega": (3.00, {"bottle_inst0"}),
    "116_C5_CellPhone_185deg_vggt_omega": (3.00, {"bottle_inst0"}),
}

_ok_all = True


def check(name, cond, detail=""):
    global _ok_all
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        _ok_all = False


print("\n[1] key 命名: pick_place 输出必须能被 mainv3 的拼接命中")
hit = tot = 0
for scene, (ratio, gt) in SCENES.items():
    p = os.path.join(BASE, scene, "_pre_dedup_masks.pkl")
    if not os.path.isfile(p):
        continue
    with open(p, "rb") as f:
        data = pickle.load(f)
    masks, wp = data["all_masks"], data["world_points"]
    tmp = tempfile.mkdtemp()
    try:
        res = run_stage3_7_pick_place(tmp, all_masks=masks, world_points=wp,
                                      frame_ratio=ratio)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    names = {i["instance_name"] for i in res.get("interactions", [])}
    tot += 1
    # 构造 mainv3 的 key 并回查 (模拟 mainv3 的判定)
    dyn_by_mainv3 = set()
    for cat, insts in masks.items():
        for idx in range(len(insts)):
            key = f"{cat}_inst{idx}"          # mainv3 的拼接方式
            if key in names:
                dyn_by_mainv3.add(key)
    if dyn_by_mainv3 == names and names:
        hit += 1
    print(f"    {scene:34s} pick_place={sorted(names)}  回查命中={sorted(dyn_by_mainv3)}")
check("5 个场景的 key 都能被 mainv3 拼接命中", hit == tot and tot > 0, f"{hit}/{tot}")

print("\n[2] 动态召回: 集成后应 5/5 (旧三条件为 0/5)")
recall_hit = recall_fn = 0
for scene, (ratio, gt) in SCENES.items():
    p = os.path.join(BASE, scene, "_pre_dedup_masks.pkl")
    if not os.path.isfile(p):
        continue
    with open(p, "rb") as f:
        data = pickle.load(f)
    masks, wp = data["all_masks"], data["world_points"]
    tmp = tempfile.mkdtemp()
    try:
        res = run_stage3_7_pick_place(tmp, all_masks=masks, world_points=wp,
                                      frame_ratio=ratio)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    got = {i["instance_name"] for i in res.get("interactions", [])}
    recall_hit += len(got & gt)
    recall_fn += len(gt - got)
    mark = "✅" if got == gt else f"❌ 多={sorted(got - gt)} 漏={sorted(gt - got)}"
    print(f"    {scene:34s} GT={sorted(gt)} 检出={sorted(got)} {mark}")
check("召回 5/5", recall_hit == 5 and recall_fn == 0, f"命中{recall_hit} 漏{recall_fn}")

print("\n[3] frame_ratio: 必须从 coordinate_alignment.json 的 total_frames 读出")
scene = "7_vggt_omega"
p = os.path.join(BASE, scene, "_pre_dedup_masks.pkl")
if os.path.isfile(p):
    with open(p, "rb") as f:
        data = pickle.load(f)
    masks, wp = data["all_masks"], data["world_points"]
    n_rgb = int(wp.shape[0])

    # 3a. 有 total_frames → 应等于 total/n_rgb
    tmp = tempfile.mkdtemp()
    try:
        with open(os.path.join(tmp, "coordinate_alignment.json"), "w") as f:
            json.dump({"total_frames": n_rgb * 3, "n_frames": n_rgb}, f)
        r = run_stage3_7_pick_place(tmp, all_masks=masks, world_points=wp)
        check("有 total_frames 时 frame_ratio 正确",
              abs(r["frame_ratio"] - 3.0) < 1e-6, f"frame_ratio={r['frame_ratio']}")

        # 3b. 缺 total_frames → 退化为 1.0 (这就是改动前的老行为)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "coordinate_alignment.json"), "w") as f:
            json.dump({"n_frames": n_rgb}, f)     # 故意不写 total_frames
        r2 = run_stage3_7_pick_place(tmp, all_masks=masks, world_points=wp)
        check("缺 total_frames 时退化为 1.0 (已知风险)",
              abs(r2["frame_ratio"] - 1.0) < 1e-6, f"frame_ratio={r2['frame_ratio']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 3c. mainv3 现在会写 total_frames —— 静态检查字段名一致
    src = open(os.path.join(REPO, "mainv3.py"), encoding="utf-8").read()
    check("mainv3 写入了 total_frames 字段", '"total_frames"' in src)
    check("mainv3 调用了 run_stage3_7_pick_place",
          "run_stage3_7_pick_place" in src)
    check("mainv3 传的是去重后 masks (不用去重前 pkl)",
          "all_masks=deduplicated_all_masks" in src,
          "避免 pick_place 与管线实例编号错位")
    check("mainv3 用 pick_place 结果做判定",
          "_pp_key in _pp_dynamic" in src)

print("\n" + ("✅ 全部通过" if _ok_all else "❌ 有失败"))
sys.exit(0 if _ok_all else 1)
