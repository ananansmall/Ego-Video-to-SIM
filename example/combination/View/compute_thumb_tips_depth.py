#!/usr/bin/env python3
"""
Compute thumb tip (joint 20) positions using depth-aligned HaWoR data.
Uses the depth-corrected pred_trans values.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE.parent / 'position_retargeting'))
from mano_layer import MANOLayer

H_SID = {0: 'left', 1: 'right'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--depth-npz', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    # Load raw data for betas, rot, hand_pose
    rec_dir = Path(args.hawor_dir) / 'reconstruction'
    files = sorted(rec_dir.glob('hawor_results_*.npz'))
    p_raw = next((f for f in files if '_depth_aligned' not in str(f)), files[0])
    print(f'Raw: {p_raw}')
    d_raw = dict(np.load(str(p_raw), allow_pickle=True))

    # Load depth-aligned for corrected trans
    print(f'Depth aligned: {args.depth_npz}')
    d_da = dict(np.load(str(args.depth_npz), allow_pickle=True))

    # Determine which hand was depth-aligned (check pred_valid)
    da_valid = d_da['pred_valid']
    n = len(d_raw['pred_trans'][0])

    results = {}
    for hidx in [0, 1]:
        side = H_SID[hidx]

        # Find first valid betas from raw
        betas_arr = d_raw['pred_betas'][hidx]
        first_valid_beta_idx = None
        for fi in range(n):
            if not np.any(np.isnan(betas_arr[fi])):
                first_valid_beta_idx = fi
                break
        if first_valid_beta_idx is None:
            print(f'  {side}: NO valid betas at all!')
            results[f'thumb_{side}'] = np.full((n, 3), np.nan, dtype=np.float64)
            continue

        betas = betas_arr[first_valid_beta_idx].astype(np.float32)
        ml = MANOLayer(side, betas)

        tips = np.full((n, 3), np.nan, dtype=np.float64)

        # Use depth-aligned trans if available, else fall back to raw
        da_trans = d_da['pred_trans'][hidx]
        da_valid_mask = d_da['pred_valid'][hidx]

        for fi in range(n):
            # Prefer depth-aligned trans if available
            if da_valid_mask[fi] and not np.any(np.isnan(da_trans[fi])):
                trans = da_trans[fi].astype(np.float32)
            else:
                trans = d_raw['pred_trans'][hidx, fi].astype(np.float32)

            # Use raw rot and hand_pose
            rot = d_raw['pred_rot'][hidx, fi].astype(np.float32)
            hp = d_raw['pred_hand_pose'][hidx, fi].astype(np.float32)

            # Skip invalid
            if np.any(np.isnan(trans)) or np.all(np.abs(trans) < 1e-6):
                continue
            if np.any(np.isnan(rot)) or np.any(np.isnan(hp)):
                continue

            p_t = torch.from_numpy(np.concatenate([rot, hp])).unsqueeze(0)
            t_t = torch.from_numpy(trans).unsqueeze(0)
            with torch.no_grad():
                _, j = ml(p_t, t_t)
            j_val = j[0, 20].numpy()
            if not np.any(np.isnan(j_val)):
                tips[fi] = j_val

        key = f'thumb_{side}'
        results[key] = tips
        nv = np.sum(~np.any(np.isnan(tips), axis=1))
        if nv > 0:
            print(f'  {key}: {nv}/{n} valid, Z=[{np.nanmin(tips[:,2]):.4f}, {np.nanmax(tips[:,2]):.4f}]')
        else:
            print(f'  {key}: {nv}/{n} valid (all NaN)')

    np.savez(args.output, **results)
    print(f'Saved: {args.output}')

if __name__ == '__main__':
    main()