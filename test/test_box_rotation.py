#!/usr/bin/env python3
"""100_C4_Box 开盒盖旋转交互诊断: 对每个实例输出
  1) 5帧窗口 overlap 曲线 (现有逻辑, 不改)
  2) 点云 bbox 高度 (z范围) 曲线 —— 旋转/翻盖的补充信号
  3) 3D 质心位移
"""
import os, sys, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.sg_deduplication import get_overlap_ratio

BASE = os.path.join(os.path.dirname(__file__), '..', 'output_v2')
scene = '100_C4_Box_081deg_vggt_omega'

with open(os.path.join(BASE, scene, '_pre_dedup_masks.pkl'), 'rb') as f:
    data = pickle.load(f)
am, wp = data['all_masks'], data['world_points']
print(f"n_frames_rgb={wp.shape[0]}  frame_ratio=300/{wp.shape[0]}={300/wp.shape[0]:.2f}")
print(f"类别: {list(am.keys())}")

for cat, cm in am.items():
    for i, inst in enumerate(cm):
        key = f"{cat}_inst{i}"
        sm = sorted(inst, key=lambda x: x['frame_id'])
        pts_list, fids = [], []
        for im in sm:
            p = wp[im['frame_id']][im['mask'] > 0]
            p = p[np.all(np.isfinite(p), axis=-1)]
            if len(p) >= 10:
                pts_list.append(p)
                fids.append(im['frame_id'])
        if len(fids) < 10:
            print(f"\n{key}: 有效帧不足({len(fids)}), 跳过")
            continue
        # 1) overlap 5帧窗口
        wm = [1.0] * 5
        for j in range(5, len(fids)):
            cp = pts_list[j]
            rp = np.concatenate(pts_list[j-5:j], axis=0)
            wm.append(get_overlap_ratio(cp, rp) if len(cp) >= 10 and len(rp) >= 10 else 1.0)
        # 2) bbox z高度 + 3) 质心
        cent = np.array([p.mean(axis=0) for p in pts_list])
        zext = np.array([p[:, 2].max() - p[:, 2].min() for p in pts_list])
        disps = np.linalg.norm(cent[1:] - cent[:-1], axis=1) if len(cent) > 1 else np.zeros(1)

        print(f"\n{key}  n={len(fids)}  面积均值={int(np.mean([np.count_nonzero(im['mask']) for im in sm]))}px")
        print(f"  fid范围: {fids[0]}..{fids[-1]}")
        print(f"  overlap: min={min(wm):.3f} low_count(<0.5)={sum(1 for v in wm if v<0.5)}")
        print(f"  overlap曲线: {''.join('.' if v>=0.5 else '#' for v in wm)}")
        print(f"  低重叠帧: {[(fids[j] if j>=5 else fids[j], round(wm[j],2)) for j,w in enumerate(wm) if w<0.5][:20]}")
        print(f"  bbox高度(z): min={zext.min():.3f} max={zext.max():.3f} 比值={zext.max()/(zext.min()+1e-6):.2f}")
        zr = zext.max() / (zext.min() + 1e-6)
        print(f"  z高度曲线(归一化): {''.join('#' if (v-zext.min())/(zr-zext.min()+1e-6)>0.5 else '.' for v in zext)}")
        print(f"  质心位移: path={disps.sum():.3f} net={np.linalg.norm(cent[-1]-cent[0]):.3f} max_disp={disps.max():.3f}")
