#!/usr/bin/env python3
"""对比测试: 主管线动态判定 (mainv3.py:631-710 复现) vs pick_place 分治判定, 5场景 + GT

主管线逻辑复现 (无 world_points_conf 时退化为均匀权重, 其余逐步一致):
  置信度加权质心 → disps → MAD剔除(|z|<3.5) → p50 → 最长连续>p50帧数
  → is_dynamic = consec > 30, 可见帧<15 跳过
"""
import os, sys, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.pick_place_detector import compute_dynamic_info

BASE = str(os.path.join(os.path.dirname(__file__), '..', 'output_v2'))

SCENES = {
    '2chair-bottle_vggt_omega':            (2.29, {'bottle_inst0'}),
    '7_vggt_omega':                        (1.00, {'toy_inst1'}),
    '3chair-bottle_vggt_omega':            (4.93, {'bottle_inst0'}),
    '121_C5_CellPhone_161deg_vggt_omega':  (3.00, {'bottle_inst0'}),
    '116_C5_CellPhone_185deg_vggt_omega':  (3.00, {'bottle_inst0'}),
}


def pipeline_judge(instance_masks, world_points, conf=None):
    """忠实复现 mainv3.py:631-710 第二轮动态判定"""
    sm = sorted(instance_masks, key=lambda x: x['frame_id'])
    centroid_data = []
    for im in sm:
        fid, mask = im['frame_id'], im['mask']
        if not np.any(mask):
            continue
        pts = world_points[fid][mask > 0]
        finite = np.all(np.isfinite(pts), axis=-1)
        if not np.any(finite):
            continue
        if conf is not None:
            confs = conf[fid][mask > 0]
            thr = np.percentile(confs[finite], 10)
            cm = finite & (confs > thr)
            if not np.any(cm):
                cm = finite
            w, p = confs[cm], pts[cm]
        else:
            w, p = np.ones(int(finite.sum())), pts[finite]
        centroid_data.append((fid, np.average(p, axis=0, weights=w)))
    if len(centroid_data) < 15:          # mainv3.py:700-704
        return False, len(centroid_data), 0
    cs = np.array([c for _, c in centroid_data])
    disps = np.array([np.linalg.norm(cs[i] - cs[i-1]) for i in range(1, len(cs))])
    med, mad = np.median(disps), np.median(np.abs(disps - np.median(disps)))
    clean = disps[np.abs(0.6745 * (disps - med) / mad) < 3.5] if mad > 1e-6 else disps
    if len(clean) == 0:
        return False, len(centroid_data), 0
    p50 = np.percentile(clean, 50)
    best = cur = 0
    for d in clean:
        cur = cur + 1 if d > p50 else 0
        best = max(best, cur)
    return best > 30, len(centroid_data), best   # mainv3.py:710


total = {'gt': 0, 'pipe_hit': 0, 'mine_hit': 0, 'pipe_fp': 0, 'mine_fp': 0,
         'pipe_fn': 0, 'mine_fn': 0}
for scene, (ratio, gt) in SCENES.items():
    with open(os.path.join(BASE, scene, '_pre_dedup_masks.pkl'), 'rb') as f:
        data = pickle.load(f)
    masks, wp = data['all_masks'], data['world_points']
    conf = data.get('world_points_conf')
    mine = compute_dynamic_info(masks, wp, ratio)
    print(f"\n{'='*74}\n{scene} (ratio={ratio})  GT动态={sorted(gt)}  conf={'有' if conf is not None else '无'}\n{'='*74}")
    for cat, insts in masks.items():
        for i, im in enumerate(insts):
            key = f"{cat}_inst{i}"
            pipe_dyn, nfr, consec = pipeline_judge(im, wp, conf)
            mine_dyn = mine.get(key, {}).get('is_dynamic', False)
            is_gt = key in gt
            marks = []
            if pipe_dyn != is_gt:
                marks.append(f"管线{'误检' if pipe_dyn else '漏检'}")
            if mine_dyn != is_gt:
                marks.append(f"分治{'误检' if mine_dyn else '漏检'}")
            flag = ' ← ' + ','.join(marks) if marks else ''
            if pipe_dyn or mine_dyn or is_gt or marks:
                print(f"  {key:<24} GT={'动' if is_gt else '静'} 管线={'动' if pipe_dyn else '静'}"
                      f"(consec={consec:3d},frames={nfr:3d}) 分治={'动' if mine_dyn else '静'}{flag}")
    # 汇总
    pipe_set = {f"{c}_inst{i}" for c, il in masks.items() for i, im_ in enumerate(il)
                if pipeline_judge(im_, wp, conf)[0]}
    mine_set = {k for k, v in mine.items() if v['is_dynamic']}
    total['gt'] += len(gt)
    total['pipe_hit'] += len(pipe_set & gt); total['mine_hit'] += len(mine_set & gt)
    total['pipe_fp'] += len(pipe_set - gt); total['mine_fp'] += len(mine_set - gt)
    total['pipe_fn'] += len(gt - pipe_set); total['mine_fn'] += len(gt - mine_set)

print(f"\n{'='*74}\n总汇 (GT动态共{total['gt']}个):")
print(f"  主管线判定(consec>30): 命中{total['pipe_hit']} 误检{total['pipe_fp']} 漏检{total['pipe_fn']}")
print(f"  分治判定(pick_place) : 命中{total['mine_hit']} 误检{total['mine_fp']} 漏检{total['mine_fn']}")
