#!/usr/bin/env python3
"""诊断 3chair-bottle 多检的小交互: 量化每个交互段的物理振幅
   1) 原始低重叠段: path/net/时长/深度
   2) 合并后交互级 (gap≤MERGE_GAP原始帧): path/net/时长 —— 过滤阈值用这个口径
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pick_place_detector import (
    _extract_instance_points, _overlap_features, MATCH_LOW_THRESH, MERGE_GAP,
)

BASE = str(os.path.join(os.path.dirname(__file__), '..', 'output_v2'))
SCENES = {
    '2chair-bottle_vggt_omega':            2.29,
    '3chair-bottle_vggt_omega':            4.93,
    '121_C5_CellPhone_161deg_vggt_omega':  3.00,
    '116_C5_CellPhone_185deg_vggt_omega':  3.00,
}

for scene, ratio in SCENES.items():
    with open(os.path.join(BASE, scene, '_pre_dedup_masks.pkl'), 'rb') as f:
        data = pickle.load(f)
    insts = _extract_instance_points(data['all_masks'], data['world_points'])
    print(f"\n{'='*74}\n{scene} (ratio={ratio})\n{'='*74}")
    for key, d in sorted(insts.items()):
        if d['avg_area'] < 1000:
            continue  # 只看大物体(overlap法)的交互
        feats = _overlap_features(d['pts_list'])
        wm, fids, cent = feats['win_match'], d['fids'], d['centroids']
        # 原始低重叠段(未后处理), (a,b) 闭区间
        segs, s = [], None
        for j, v in enumerate(wm + [1.0]):
            if v < MATCH_LOW_THRESH:
                s = j if s is None else s
            elif s is not None:
                segs.append((s, j - 1)); s = None
        if not segs:
            continue
        disps = np.linalg.norm(cent[1:] - cent[:-1], axis=1)  # disps[k]: fid[k]→fid[k+1]
        print(f"  {key}:")
        for (a, b) in segs:
            path = float(disps[a:b].sum()) if b > a else 0.0
            net = float(np.linalg.norm(cent[b] - cent[a]))
            dur = (fids[b] - fids[a] + 1) * ratio
            print(f"    原始段 RGB[{fids[a]:3d}→{fids[b]:3d}] min={min(wm[a:b+1]):.3f} "
                  f"时长={dur:5.0f}帧 path={path:.3f}m net={net:.3f}m")
        # 合并后交互级 (与 postprocess 同口径: gap_rgb ≤ MERGE_GAP/ratio)
        merge_gap_rgb = MERGE_GAP / ratio
        merged = []
        for (a, b) in segs:
            if merged and fids[a] - fids[merged[-1][1]] <= merge_gap_rgb:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for (a, b) in merged:
            path = float(disps[a:b].sum()) if b > a else 0.0
            net = float(np.linalg.norm(cent[b] - cent[a]))
            dur = (fids[b] - fids[a] + 1) * ratio
            nseg = sum(1 for (x, y) in segs if x >= a and y <= b)
            print(f"    ★合并交互 orig[{int(fids[a]*ratio)}→{int(fids[b]*ratio)}] "
                  f"含{nseg}个原始段 时长={dur:5.0f}帧 path={path:.3f}m net={net:.3f}m")
