#!/usr/bin/env python3
"""
tools/test_smooth.py

Validate the two MANO smoothers in smooth_mano.py against a real HaWoR npz.

For each hand we:
  1. load raw params (pred_trans/rot/hand_pose/betas)
  2. FK the raw params -> joint trajectories (wrist j0, midpoint)
  3. run Kalman + Savitzky-Golay smoothing on the params
  4. FK the smoothed params -> joint trajectories
  5. report per-frame jump stats (max jump, F50->F51 jump) and RMSE vs raw

Usage:
  python3 tools/test_smooth.py \
      --npz example/combination/hand_track/output/7/hawor_world_space/reconstruction/hawor_results_0_113.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# allow importing smooth_mano + MANOLayer
THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))
sys.path.insert(0, str(THIS.parent.parent / "position_retargeting"))  # mano_layer.py
from smooth_mano import smooth_mano_kalman, smooth_mano_sg  # noqa: E402

from mano_layer import MANOLayer  # noqa: E402

H_SID = {0: "left", 1: "right"}


def fk_joints(r, hand_idx):
    """FK MANO joints from a params dict (uses r, NOT a global).
    Returns (N,21,3) in the HaWoR INPUT CAMERA frame (before any world transform)."""
    side = H_SID[hand_idx]
    trans = r["pred_trans"]
    rot = r["pred_rot"]
    hp = r["pred_hand_pose"]
    betas = r["pred_betas"]
    if trans.ndim == 3:
        trans, rot, hp, betas = trans[hand_idx], rot[hand_idx], hp[hand_idx], betas[hand_idx]
    n = len(trans)
    first_valid = next((i for i in range(n) if not np.any(np.isnan(betas[i]))), 0)
    ml = MANOLayer(side, betas[first_valid].astype(np.float32))
    joints = np.full((n, 21, 3), np.nan, dtype=np.float64)
    for i in range(n):
        if np.any(np.isnan(trans[i])) or np.all(np.abs(trans[i]) < 1e-6):
            continue
        if np.any(np.isnan(rot[i])) or np.any(np.isnan(hp[i])):
            continue
        p_t = np.concatenate([rot[i], hp[i]]).astype(np.float32)[None]
        t_t = trans[i].astype(np.float32)[None]
        with __import__("torch").no_grad():
            _, j = ml(__import__("torch").from_numpy(p_t), __import__("torch").from_numpy(t_t))
        jv = j[0].numpy()
        if not np.any(np.isnan(jv)):
            joints[i] = jv
    return joints


def midpoint(j):
    """(N,21,3) -> (N,3) midpoint of thumb tip (j20) & index tip (j16)."""
    return (j[:, 20, :] + j[:, 16, :]) / 2.0


def frame_jumps(x):
    """(N,3) -> (N-1,) per-frame euclidean jumps."""
    return np.linalg.norm(np.diff(x, axis=0), axis=1)


def report(name, raw_j, sm_j):
    raw_mp = midpoint(raw_j)
    sm_mp = midpoint(sm_j)
    ok = ~np.any(np.isnan(raw_mp), axis=1) & ~np.any(np.isnan(sm_mp), axis=1)
    raw_jumps = frame_jumps(raw_mp[ok])
    sm_jumps = frame_jumps(sm_mp[ok])
    rmse = np.sqrt(np.mean(np.sum((raw_mp[ok] - sm_mp[ok]) ** 2, axis=1)))
    print(f"  [{name}]")
    print(f"    raw max jump      : {raw_jumps.max()*100:.2f} cm  (frame {raw_jumps.argmax()})")
    print(f"    sm  max jump      : {sm_jumps.max()*100:.2f} cm  (frame {sm_jumps.argmax()})")
    if 50 < len(raw_jumps) and 50 < len(sm_jumps):
        print(f"    F50->F51 jump raw : {raw_jumps[49]*100:.2f} cm")
        print(f"    F50->F51 jump sm  : {sm_jumps[49]*100:.2f} cm")
    print(f"    RMSE smoothed vs raw (midpoint): {rmse*100:.2f} cm")
    return raw_jumps, sm_jumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--hands", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--sg-window", type=int, default=9)
    ap.add_argument("--sg-poly", type=int, default=3)
    ap.add_argument("--inject-spike", type=float, default=0.0,
                    help="Inject a synthetic translation spike (meters) at frame 50 to "
                         "verify outlier rejection. 0 = no injection.")
    args = ap.parse_args()

    raw = dict(np.load(args.npz, allow_pickle=True))
    print(f"Loaded {args.npz}")
    print(f"  pred_trans shape {raw['pred_trans'].shape}, frames={raw['pred_trans'].shape[-2]}")

    if args.inject_spike:
        for h in args.hands:
            raw["pred_trans"][h, 50] += np.array([args.inject_spike, 0.0, 0.0])
        print(f"  [TEST] injected {args.inject_spike*100:.1f} cm spike into pred_trans[*,50]")

    for h in args.hands:
        print(f"\n=== Hand {h} ({H_SID[h]}) ===")
        raw_j = fk_joints(raw, h)
        k = smooth_mano_kalman(raw, hand_idx=h)
        s = smooth_mano_sg(raw, hand_idx=h, window=args.sg_window, poly=args.sg_poly)
        k_j = fk_joints(k, h)
        s_j = fk_joints(s, h)
        report("Kalman", raw_j, k_j)
        report("SG", raw_j, s_j)

    print("\nDone.")


if __name__ == "__main__":
    main()
