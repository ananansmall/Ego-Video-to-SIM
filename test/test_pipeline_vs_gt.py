#!/usr/bin/env python3
"""对比测试: 管线兜底动态判据 (三条件, mainv3 复现) vs pick_place 分治判定, 5场景 + GT

背景 (2026-10-10 更新):
  mainv3 的动态判定已改为以 Stage 3.7 的 pick/place 结果为准
  (`is_dynamic_inst = f"{category}_inst{inst_idx}" in _pp_dynamic`);
  原先自带的三条件仅在 Stage 3.7 不可用时作为兜底。

  本脚本保留三条件作对比基线, 用来量化两者差距。
  原实现复现的是更早的 `max_consecutive_high > 30` —— 那个变量在 mainv3 里
  只被计算、从未参与判定 (判定用的是三条件), 所以它既低估了旧判据的漏检,
  也不反映任何当前代码路径。现改为复现真正的兜底三条件。

运行: /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test/test_pipeline_vs_gt.py
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
    """复现 mainv3 的兜底动态判据 (三条件), 作为与 pick_place 分治判定的对比基线。

    三条件: (global_disp > 0.05) and (moving_ratio > 0.3) and (light_moving_ratio > 0.8)
    返回 (is_dynamic, n_frames, global_disp)
    """
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

    if len(centroid_data) < 15:              # mainv3: 可见帧 < 15 直接跳过
        return False, len(centroid_data), 0.0

    cs = np.array([c for _, c in centroid_data])
    disps = np.array([np.linalg.norm(cs[i] - cs[i - 1]) for i in range(1, len(cs))])
    med, mad = np.median(disps), np.median(np.abs(disps - np.median(disps)))
    clean = disps[np.abs(0.6745 * (disps - med) / mad) < 3.5] if mad > 1e-6 else disps
    if len(clean) == 0:
        return False, len(centroid_data), 0.0

    # ── 三条件 (mainv3 兜底分支) ──
    n_q = max(3, len(centroid_data) // 4)
    head = np.mean([c for _, c in centroid_data[:n_q]], axis=0)
    tail = np.mean([c for _, c in centroid_data[-n_q:]], axis=0)
    global_disp = float(np.linalg.norm(tail - head))
    moving_ratio = float(np.sum(clean > 0.05) / len(clean))
    light_moving_ratio = float(np.sum(clean > 0.02) / len(clean))
    is_dyn = ((global_disp > 0.05) and (moving_ratio > 0.3)
              and (light_moving_ratio > 0.8))
    return is_dyn, len(centroid_data), global_disp


total = {'gt': 0, 'pipe_hit': 0, 'mine_hit': 0, 'pipe_fp': 0, 'mine_fp': 0,
         'pipe_fn': 0, 'mine_fn': 0}
for scene, (ratio, gt) in SCENES.items():
    with open(os.path.join(BASE, scene, '_pre_dedup_masks.pkl'), 'rb') as f:
        data = pickle.load(f)
    masks, wp = data['all_masks'], data['world_points']
    conf = data.get('world_points_conf')
    mine = compute_dynamic_info(masks, wp, ratio)
    print(f"\n{'='*78}\n{scene} (ratio={ratio})  GT动态={sorted(gt)}  conf={'有' if conf is not None else '无'}\n{'='*78}")
    for cat, insts in masks.items():
        for i, im in enumerate(insts):
            key = f"{cat}_inst{i}"
            pipe_dyn, nfr, gdisp = pipeline_judge(im, wp, conf)
            mine_dyn = mine.get(key, {}).get('is_dynamic', False)
            is_gt = key in gt
            marks = []
            if pipe_dyn != is_gt:
                marks.append(f"三条件{'误检' if pipe_dyn else '漏检'}")
            if mine_dyn != is_gt:
                marks.append(f"分治{'误检' if mine_dyn else '漏检'}")
            flag = ' ← ' + ','.join(marks) if marks else ''
            if pipe_dyn or mine_dyn or is_gt or marks:
                print(f"  {key:<24} GT={'动' if is_gt else '静'} 三条件={'动' if pipe_dyn else '静'}"
                      f"(gd={gdisp:.3f},frames={nfr:3d}) 分治={'动' if mine_dyn else '静'}{flag}")
    # 汇总
    pipe_set = {f"{c}_inst{i}" for c, il in masks.items() for i, im_ in enumerate(il)
                if pipeline_judge(im_, wp, conf)[0]}
    mine_set = {k for k, v in mine.items() if v['is_dynamic']}
    total['gt'] += len(gt)
    total['pipe_hit'] += len(pipe_set & gt); total['mine_hit'] += len(mine_set & gt)
    total['pipe_fp'] += len(pipe_set - gt); total['mine_fp'] += len(mine_set - gt)
    total['pipe_fn'] += len(gt - pipe_set); total['mine_fn'] += len(gt - mine_set)

print(f"\n{'='*78}\n总汇 (GT动态共{total['gt']}个):")
print(f"  兜底三条件 (mainv3 备用)  : 命中{total['pipe_hit']} 误检{total['pipe_fp']} 漏检{total['pipe_fn']}")
print(f"  pick_place 分治 (当前生效): 命中{total['mine_hit']} 误检{total['mine_fp']} 漏检{total['mine_fn']}")
print("\n  说明: 当前 mainv3 用的是 pick_place; 三条件仅在 Stage 3.7 失败时兜底。")
print("        (本脚本旧版复现的 `consec>30` 与 mainv3 代码不符, 已修正)")
