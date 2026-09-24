#!/usr/bin/env python3
"""5 场景回归测试: 重建后的 pick_place_detector (大小物体分治 + frame_ratio)

预期 (v2 最终方案):
  2chair-bottle (2.29): bottle_inst0 动态, chair/table 静态
  7_vggt       (1.00): toy_inst1 动态
  3chair-bottle(4.93): bottle_inst0 动态; gap≤20原始帧的相邻交互应合并
  121_CellPhone(3.00): bottle_inst0 动态
  116_CellPhone(3.00): bottle_inst0 动态 (low_count 方案修复慢速移动漏检)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pick_place_detector import (
    compute_dynamic_info, _detect_interactions_small, _detect_interactions_large,
    postprocess_interactions,
)

BASE = str(os.path.join(os.path.dirname(__file__), '..', 'output_v2'))

SCENES = {
    '2chair-bottle_vggt_omega':            2.29,   # 229帧 / 100 RGB
    '7_vggt_omega':                        1.00,   # 113帧 / 113 RGB
    '3chair-bottle_vggt_omega':            4.93,   # 592帧 / 120 RGB
    '121_C5_CellPhone_161deg_vggt_omega':  3.00,   # 300帧 / 100 RGB
    '116_C5_CellPhone_185deg_vggt_omega':  3.00,   # 300帧 / 100 RGB
}

import pickle

ONLY = sys.argv[1:]  # 可选: 命令行传入场景名子集, 如 python test_combined.py 7_vggt_omega

for scene, ratio in SCENES.items():
    if ONLY and scene not in ONLY:
        continue
    scene_dir = os.path.join(BASE, scene)
    with open(os.path.join(scene_dir, '_pre_dedup_masks.pkl'), 'rb') as f:
        data = pickle.load(f)
    masks, wp = data['all_masks'], data['world_points']
    print(f"\n{'='*72}\n{scene}  (frame_ratio={ratio}, n_frames_rgb={wp.shape[0]})\n{'='*72}", flush=True)

    info = compute_dynamic_info(masks, wp, ratio)
    dyn = []
    for key in sorted(info):
        d = info[key]
        ft = d['features']
        if d['method'] == 'centroid':
            extra = (f"path={ft['path']:.3f} net={ft['net']:.3f} "
                     f"pnet={ft['pnet']:.1f} run={ft['max_run']}x{ratio}={ft['max_run']*ratio:.0f}")
        else:
            extra = (f"min={ft['min_match']:.3f} low={ft['low_count']}x{ratio}="
                     f"{ft['low_count']*ratio:.0f} (path={ft.get('path', 0):.2f} net={ft.get('net', 0):.2f})")
        tag = "[DYN] " if d['is_dynamic'] else "[stat]"
        print(f"  {tag} {key:<22} area={int(d['avg_area']):6d} [{d['method']:8s}] {extra}", flush=True)
        if d['is_dynamic']:
            dyn.append(key)

    inters = []
    for key in dyn:
        d = info[key]
        if d['method'] == 'centroid':
            inters += _detect_interactions_small(key, d['fids'], d['centroids'], d['features'])
        else:
            inters += _detect_interactions_large(key, d['fids'], d['features'])
    inters = postprocess_interactions(inters, ratio)

    print(f"\n  交互 ({len(inters)}):", flush=True)
    for it in inters:
        print(f"    {it['instance_name']}: "
              f"pick_win={it['pick_window']} pick={it['pick_frame']} | "
              f"place={it['place_frame']} place_win={it['place_window']} "
              f"conf={it['confidence']:.2f}", flush=True)

    out = os.path.join(scene_dir, 'pick_place_frames_combined.json')
    with open(out, 'w') as f:
        json.dump({'frame_ratio': ratio, 'n_frames_rgb': int(wp.shape[0]),
                   'interactions': inters}, f, indent=2)
    print(f"  → {out}", flush=True)
