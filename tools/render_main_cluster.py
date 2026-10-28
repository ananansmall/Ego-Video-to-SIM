#!/usr/bin/env python3
"""按"主簇"取景渲染场景（方案 C：只调取景，不动数据）。

为什么要这个工具:
  某些场景会有个别实例被重建到离主体 2~5 米外（实测 3chair-v04: curtain_2/3/4/5/6
  和 cabinet_1 落在 y≈3.0 一带，而主体 14 个实例在 y≈0.7~1.3）。此时用整个场景的
  bounds 取景，主体会被压得很小，视觉上像"一堆白片飘在远处"，看不出实际效果。

  这个工具**不修改任何数据**（all_instances.pkl / GLB 都不动），只是把渲染相机的
  取景范围设成"最大连通簇"的包围盒，让主体填满画面。远处实例仍在，只是出画。

用法:
  python tools/render_main_cluster.py --scene output_v3/3chair-v04
  python tools/render_main_cluster.py --scene output_v3/3chair-v04 \
      --glb final_scene_orphan_fixed.glb --link_dist 1.0
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))


def largest_cluster(centers, link_dist):
    """最大连通簇（边 = 两实例中心距离 < link_dist）。返回索引列表。"""
    n = len(centers)
    if n == 0:
        return []
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(centers[i] - centers[j]) < link_dist:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return max(groups.values(), key=len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--glb", default=None,
                    help="默认自动选 final_scene_stage5.glb / final_scene.glb")
    ap.add_argument("--link_dist", type=float, default=1.0,
                    help="两实例中心距离小于该值视为同一簇（米）")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--out_suffix", default="maincluster")
    args = ap.parse_args()

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import trimesh
    from tools.render_glb_preview import (build_pyrender_scene,
                                          render_views, grid2x2)

    D = os.path.abspath(args.scene)
    glb = args.glb
    if not glb:
        for cand in ("final_scene_stage5.glb", "final_scene_orphan_fixed.glb",
                     "final_scene_fixed.glb", "final_scene.glb"):
            p = os.path.join(D, cand)
            if os.path.isfile(p):
                glb = p
                break
    if not glb or not os.path.isfile(glb):
        print(f"❌ 找不到 GLB: {args.scene}")
        return 1

    scene = trimesh.load(glb)
    # 用 dump(concatenate=False) 拿"已在世界坐标"的几何 —— 直接解 scene.graph[n]
    # 在不同 trimesh 版本上返回值形态不一致 (会 unpack 失败), dump 更稳。
    geos = [g for g in scene.dump(concatenate=False)
            if isinstance(g, trimesh.Trimesh)]
    centers, bounds_list, names = [], [], []
    for i, g in enumerate(geos):
        b = g.bounds
        if b is None or not np.all(np.isfinite(b)):
            continue
        centers.append((b[0] + b[1]) / 2.0)
        bounds_list.append(b)
        names.append(g.metadata.get("name", f"geo_{i}"))
    if not centers:
        print("❌ 场景里没有可用几何")
        return 1

    idx = largest_cluster(np.asarray(centers), args.link_dist)
    # 排除地板/大平面: 若某个几何的水平跨度超过整体的 50%, 它是地面而不是"主体",
    # 让它参与渲染但不参与取景(否则地面撑满画面, 物体被压成几十像素看不清)。
    _all_lo = np.min([b[0] for b in bounds_list], axis=0)
    _all_hi = np.max([b[1] for b in bounds_list], axis=0)
    _span = max(float(_all_hi[0] - _all_lo[0]), float(_all_hi[1] - _all_lo[1]), 1e-9)
    _frame_idx = []
    for _i in idx:
        _b = bounds_list[_i]
        _s = max(float(_b[1][0] - _b[0][0]), float(_b[1][1] - _b[0][1]))
        if _s / _span < 0.5:
            _frame_idx.append(_i)
    if not _frame_idx:
        _frame_idx = idx
    elif len(_frame_idx) < len(idx):
        print(f"  取景排除 {len(idx)-len(_frame_idx)} 个大面积平面(地面)")
    lo = np.min([bounds_list[i][0] for i in _frame_idx], axis=0)
    hi = np.max([bounds_list[i][1] for i in _frame_idx], axis=0)
    main_bounds = np.array([lo, hi])

    all_lo = np.min([b[0] for b in bounds_list], axis=0)
    all_hi = np.max([b[1] for b in bounds_list], axis=0)
    print(f"  场景 {len(centers)} 个节点")
    print(f"  主簇 {len(idx)} 个节点, 包围盒 {(hi - lo).round(2).tolist()} m")
    print(f"  全场包围盒 {(all_hi - all_lo).round(2).tolist()} m")
    out_of_view = [names[i] for i in range(len(names)) if i not in set(idx)]
    if out_of_view:
        print(f"  出画的 {len(out_of_view)} 个: {', '.join(out_of_view[:8])}"
              f"{' ...' if len(out_of_view) > 8 else ''}")
        print("  (数据未改动, 只是没进画面)")

    # 用管线同款渲染器: build_pyrender_scene 出 pyrender.Scene (带 .bounds),
    # render_views 支持传自定义 bounds → 相机只框主簇, 远处实例出画。
    pr_scene, n_meshes = build_pyrender_scene(glb)
    if n_meshes == 0:
        print("❌ pyrender 场景为空")
        return 1
    tiles = render_views(pr_scene, main_bounds,
                         ["front", "side", "top", "iso"], args.size)
    grid = grid2x2(tiles)
    base = os.path.splitext(os.path.basename(glb))[0]
    out = os.path.join(D, f"{base}_{args.out_suffix}_preview.png")
    import cv2
    cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  ✅ 已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
