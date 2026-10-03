#!/usr/bin/env python3
"""
Shared utilities for MANO-based gripper retargeting visualization.

Provides common functions and constants used across:
    - viz_mano_retargeting_web.py (single-frame interactive)
    - viz_mano_multiframe.py     (multi-frame comparison)
"""

import numpy as np
from pathlib import Path

# ── MANO hand skeleton connectivity ──
MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# ── MANO fingertip joint indices ──
FINGER_TIP = [4, 8, 12, 16, 20]


def load_hawor_data(hawor_dir, hand_idx=1, frame_idx=50):
    """Load HaWoR reconstruction data for a given frame.

    Looks for depth-aligned NPZ files first, falls back to raw NPZ.
    Returns a dict with pred_rot, pred_trans, pred_hand_pose, pred_betas, frame_idx.
    """
    rec_dir = Path(str(hawor_dir)) / "reconstruction"
    files = sorted(rec_dir.glob("hawor_results_*_depth_aligned.npz"), reverse=True)
    if not files:
        files = sorted(rec_dir.glob("hawor_results_*.npz"), reverse=True)
    rec = np.load(str(files[0]), allow_pickle=True)
    valid = np.where(rec['pred_valid'][hand_idx] &
                     ~np.any(np.isnan(rec['pred_hand_pose'][hand_idx]), axis=1))[0]
    if len(valid) == 0:
        raise ValueError("No valid frames!")
    fi = valid[min(frame_idx, len(valid) - 1)]
    return {
        "pred_rot": rec['pred_rot'][hand_idx][fi],
        "pred_trans": rec['pred_trans'][hand_idx][fi],
        "pred_hand_pose": rec['pred_hand_pose'][hand_idx][fi],
        "pred_betas": rec['pred_betas'][hand_idx][fi],
        "frame_idx": fi,
    }


def matrix_from_compact_axis_angle(aa):
    """Convert a compact axis-angle vector to a 3x3 rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    c, s = np.cos(angle), np.sin(angle)
    cx, cy, cz = axis
    return np.array([
        [c + cx*cx*(1-c), cx*cy*(1-c) - cz*s, cx*cz*(1-c) + cy*s],
        [cy*cx*(1-c) + cz*s, c + cy*cy*(1-c), cy*cz*(1-c) - cx*s],
        [cz*cx*(1-c) - cy*s, cz*cy*(1-c) + cx*s, c + cz*cz*(1-c)],
    ])


def compute_mano_joints(rot, hp, trans, betas):
    """Forward MANO model to get vertices and 3D joint positions."""
    from mano_layer import MANOLayer
    import torch
    ml = MANOLayer("right", betas)
    p = torch.from_numpy(np.concatenate([rot, hp]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = ml(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]
