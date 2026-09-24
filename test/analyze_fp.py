#!/usr/bin/env python3
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.sg_deduplication import get_overlap_ratio
BASE = os.path.join(os.path.dirname(__file__), '..', 'output_v2')
for scene in ['3chair-bottle_vggt_omega', '7_vggt_omega', '2chair-bottle_vggt_omega', '121_C5_CellPhone_161deg_vggt_omega']:
    d = os.path.join(BASE, scene)
    if not os.path.exists(os.path.join(d, '_pre_dedup_masks.pkl')): continue
    with open(os.path.join(d, '_pre_dedup_masks.pkl'), 'rb') as f: data = pickle.load(f)
    am = data.get('all_masks'); wp = data.get('world_points')
    print(f"\n{'='*60}\n{scene}\n{'='*60}")
    for cat, cm in am.items():
        for i in range(min(2, len(cm))):
            inst = cm[i]; key = f"{cat}_inst{i}"
            sm = sorted(inst, key=lambda x: x['frame_id'])
            pts, fids = [], []
            for im in sm:
                p = wp[im['frame_id']][im['mask']>0]
                p = p[np.all(np.isfinite(p), axis=-1)]
                if len(p)>=10: pts.append(p); fids.append(im['frame_id'])
            if len(fids)<10: continue
            wm = [1.0]*5
            for j in range(5, len(fids)):
                cp = pts[j]; rp = np.concatenate(pts[j-5:j], axis=0)
                wm.append(get_overlap_ratio(cp, rp) if len(cp)>=10 and len(rp)>=10 else 1.0)
            mn = min(wm); r50=0; c=0
            for v in wm:
                if v<0.50: c+=1; r50=max(r50,c)
                else: c=0
            print(f"  {key:25s} min={mn:.3f} r50={r50:3d} n={len(fids)}")
