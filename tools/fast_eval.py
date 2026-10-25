#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5 快速评估: 直接复用已有 Stage1~3 产物, 秒级出指标, 不跑模型。

为什么需要它
------------
mainv3 是一条 Stage1→2→3→4→5 的串行流水线, 且**没有任何 resume/skip 能力**。
改一次 Stage 5 就必须重跑整条 (实测 46 分钟), 而 Stage 3 的 SAM3D 又会因为
机器显存被占而 OOM —— 结果就是"改完没法验证", 只能靠单测猜。

但 Stage 5 的输入就是 Stage 3 落盘的 all_instances.pkl + final_relations.json,
后面全是纯几何计算, 不需要 GPU。所以可以直接加载产物、跑同一批函数、出指标。

用法
----
  # 评估全部场景
  python tools/fast_eval.py
  # 评估指定场景
  python tools/fast_eval.py --scenes basic_pick_place_46 vertical_pick_place_115
  # 跳过 Stage5 补正, 只看原始 Stage3 产物的问题 (对照用)
  python tools/fast_eval.py --no-fix
  # 把评估后的位姿写成 GLB 以便肉眼检查
  python tools/fast_eval.py --scenes basic_pick_place_46 --dump-glb

指标口径
--------
  穿模对      两两 AABB 内顶点比例最大值 > 2%
  悬空/陷地   底面 z 与支撑物顶面/地面 的 gap 超出 ±3cm
  支撑不足    on_top 物体与支撑物的水平重叠 < 50%
"""
import argparse
import copy
import glob
import json
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PEN_THRESH = 0.02     # 单向顶点落入对方 AABB 的比例阈值
GAP_TOL = 0.03        # 贴合容差(米)
SUPPORT_MIN = 0.50    # 水平重叠低于此值视为"悬在边沿"


def _bounds(info):
    m = info["original_mesh"].copy()
    m.apply_transform(np.asarray(info["T"], dtype=float))
    return m.bounds


def _items(insts):
    out = []
    for cat, lst in insts.items():
        for i, info in enumerate(lst):
            out.append((f"{cat}_{i}", info))
    return out


def metrics(insts, relations):
    """返回客观指标字典。"""
    items = _items(insts)
    # 穿模对
    pen = []
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            na, a = items[x]
            nb, b = items[y]
            ma = a["original_mesh"].copy()
            ma.apply_transform(np.asarray(a["T"], dtype=float))
            mb = b["original_mesh"].copy()
            mb.apply_transform(np.asarray(b["T"], dtype=float))
            pa, pb = np.asarray(ma.vertices), np.asarray(mb.vertices)
            if len(pa) == 0 or len(pb) == 0:
                continue
            lo, hi = mb.bounds[0], mb.bounds[1]
            ra = float(np.all((pa >= lo) & (pa <= hi), axis=-1).mean())
            lo, hi = ma.bounds[0], ma.bounds[1]
            rb = float(np.all((pb >= lo) & (pb <= hi), axis=-1).mean())
            r = max(ra, rb)
            if r > PEN_THRESH:
                pen.append((na, nb, r))

    # 悬空/陷地 + 支撑不足
    byname = dict(items)
    floating = sunken = edge = 0
    for name, info in items:
        rel = str(relations.get(name, ""))
        b = _bounds(info)
        sup = rel[len("supported by "):].strip() if rel.startswith("supported by ") else ""
        if sup and sup.lower() in ("floor", "ground") or not sup:
            gap = float(b[0][2])
            if gap > GAP_TOL:
                floating += 1
            elif gap < -GAP_TOL:
                sunken += 1
        elif sup in byname:
            sb = _bounds(byname[sup])
            gap = float(b[0][2] - sb[1][2])
            if gap > GAP_TOL:
                floating += 1
            elif gap < -GAP_TOL:
                sunken += 1
            # 水平重叠
            ox = max(0.0, min(b[1][0], sb[1][0]) - max(b[0][0], sb[0][0]))
            oy = max(0.0, min(b[1][1], sb[1][1]) - max(b[0][1], sb[0][1]))
            area = max((b[1][0] - b[0][0]) * (b[1][1] - b[0][1]), 1e-9)
            if ox * oy / area < SUPPORT_MIN:
                edge += 1
    return {
        "n": len(items),
        "pen": len(pen),
        "pen_detail": pen,
        "floating": floating,
        "sunken": sunken,
        "edge": edge,
    }


def run_stage5_chain(insts, relations, do_fix=True, verbose=False):
    """跑与 mainv3.run_stage5 完全一致的 5.25 调用顺序与参数。"""
    if not do_fix:
        return insts, relations
    from src.sp_refinement import (ground_floating_objects,
                                   separate_object_penetrations,
                                   reseat_on_top_objects,
                                   recenter_edge_objects,
                                   fix_supporter_instance)
    w = copy.deepcopy(insts)
    # 5.15 支撑物实例纠偏 (与 mainv3 同序, 在 5.2/5.25 之前)
    relations, _ = fix_supporter_instance(w, dict(relations), verbose=verbose)
    try:
        w, _ = ground_floating_objects(w, relations, verbose=verbose, ref_z={})
    except Exception as e:
        print(f"      ⚠️ 落地补正异常: {type(e).__name__}: {str(e)[:60]}")
    w, _ = reseat_on_top_objects(w, relations, placement_strategies=None,
                                 verbose=verbose)
    w, _ = recenter_edge_objects(w, relations, verbose=verbose)
    # 收敛循环: 分离必须在最后 (放前面会被重贴/居中撤销), 且需迭代。
    # 循环内不调用 recenter —— 它把物体聚拢, 与分离震荡。
    for rnd in range(3):
        w, ns = separate_object_penetrations(w, relations, verbose=verbose)
        if ns == 0:
            break
        w, _ = reseat_on_top_objects(w, relations, placement_strategies=None,
                                     verbose=False)
    return w, relations


def main():
    ap = argparse.ArgumentParser(description="Stage5 快速评估(不跑模型)")
    ap.add_argument("--root", default=os.path.join(REPO, "output_v3"))
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--no-fix", action="store_true", help="不跑补正, 只看原始产物")
    ap.add_argument("--dump-glb", action="store_true", help="导出评估后位姿的 GLB")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.root, "*")))
    if args.scenes:
        dirs = [os.path.join(args.root, s) for s in args.scenes]

    print(f"{'场景':28s} {'实例':>4s} {'穿模对':>6s} {'悬空':>4s} {'陷地':>4s} {'边沿':>4s}")
    print("-" * 60)
    tot = {"pen": 0, "floating": 0, "sunken": 0, "edge": 0, "n": 0}
    for d in dirs:
        s = os.path.basename(d)
        pkl = os.path.join(d, "all_instances.pkl")
        relp = os.path.join(d, "final_relations.json")
        if not (os.path.isfile(pkl) and os.path.isfile(relp)):
            print(f"{s:28s} (缺产物, 跳过)")
            continue
        try:
            inst = pickle.load(open(pkl, "rb"))["all_instances"]
            rel = json.load(open(relp))
        except Exception as e:
            print(f"{s:28s} 读取失败: {type(e).__name__}")
            continue
        fixed, rel_eff = run_stage5_chain(inst, rel, do_fix=not args.no_fix,
                                          verbose=args.verbose)
        # 必须用"纠偏后的关系"算 gap —— 用原始关系会拿错基准比较,
        # 把已经改判为 floor 的物体仍按原来的错误支撑物量 z 差。
        m = metrics(fixed, rel_eff)
        print(f"{s:28s} {m['n']:>4d} {m['pen']:>6d} "
              f"{m['floating']:>4d} {m['sunken']:>4d} {m['edge']:>4d}")
        if m["pen"] and args.verbose:
            for na, nb, r in m["pen_detail"][:5]:
                print(f"       穿模: {na} ↔ {nb}  {r*100:.0f}%")
        for k in ("pen", "floating", "sunken", "edge", "n"):
            tot[k] += m[k]
        if args.dump_glb:
            try:
                from mainv3 import save_final_glb
                save_final_glb(fixed, d, filename="final_scene_fast_eval.glb")
                print(f"       💾 已导出 final_scene_fast_eval.glb")
            except Exception as e:
                print(f"       ⚠️ GLB 导出失败: {type(e).__name__}: {str(e)[:60]}")

    print("-" * 60)
    print(f"{'合计':28s} {tot['n']:>4d} {tot['pen']:>6d} "
          f"{tot['floating']:>4d} {tot['sunken']:>4d} {tot['edge']:>4d}")
    print(f"\n模式: {'原始产物(未补正)' if args.no_fix else '已跑 Stage5 补正'}")


if __name__ == "__main__":
    main()
