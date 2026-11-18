"""Anatomical wrist and opening priors for existing R1 paired targets."""

import numpy as np


def _positions(*values):
    arrays = [np.asarray(value, dtype=float) for value in values]
    if any(value.ndim != 2 or value.shape[1] != 3 or value.shape != arrays[0].shape
           for value in arrays):
        raise ValueError("hand landmarks must share shape [N,3] in the same metric frame")
    return arrays


def anatomical_wrist(wrist, middle_mcp, thumb_mcp):
    """Return world-from-anatomical rotations and rows with observable wrist axes."""
    wrist, middle_mcp, thumb_mcp = _positions(wrist, middle_mcp, thumb_mcp)
    front = middle_mcp - wrist
    thumb = thumb_mcp - wrist
    front_length = np.linalg.norm(front, axis=1)
    x = front / np.maximum(front_length[:, None], 1e-9)
    across = thumb - np.sum(x*thumb, axis=1, keepdims=True)*x
    across_length = np.linalg.norm(across, axis=1)
    y = across / np.maximum(across_length[:, None], 1e-9)
    z = np.cross(x, y)
    valid = (np.isfinite(x).all(1) & np.isfinite(y).all(1)
             & (front_length > 1e-6) & (across_length > 1e-6))
    axes = np.stack((x, y, z), axis=-1)
    axes[~valid] = np.nan
    return axes, valid


def _phase(mask, n):
    values = np.asarray(mask, dtype=bool)
    if values.shape != (n,):
        raise ValueError("phase mask must have one entry per source frame")
    return values


def apply_hand_prior(pad, grip, wrist, middle_mcp, thumb_mcp, thumb_tip, index_tip,
                     R_anatomical_from_pad, jaw_gap_at_zero, joint_limits,
                     use_orientation, use_opening):
    """Change an existing pair's rotation and approach commands on explicit masks.

    All five landmarks are positions in the pair's metric world frame. Missing
    observations fall back to the old pair. The caller chooses where the source
    hand is trusted and where existing object/contact geometry takes priority.
    The R1 pad's location and the event stream come from the existing retarget.
    """
    pad = np.array(pad, dtype=float, copy=True)
    grip = np.array(grip, dtype=float, copy=True)
    tip1, tip2 = _positions(thumb_tip, index_tip)
    n = len(tip1)
    if pad.shape != (n,4,4) or grip.shape != (n,2):
        raise ValueError("R1 pair and landmarks must share source-frame order")
    axes, valid = anatomical_wrist(wrist, middle_mcp, thumb_mcp)
    if len(valid) != n:
        raise ValueError("wrist landmarks and fingertip landmarks must share source-frame order")
    orientation_mask = _phase(use_orientation, n) & valid
    mount = np.asarray(R_anatomical_from_pad, dtype=float)
    if (mount.shape != (3,3) or not np.allclose(mount.T@mount, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(mount), 1.0, atol=1e-6)):
        raise ValueError("wrist-to-pad mount must be a proper fixed rotation")
    pad[orientation_mask, :3, :3] = axes[orientation_mask] @ mount

    limits = np.asarray(joint_limits, dtype=float)
    if (limits.shape != (2,2) or not np.isfinite(limits).all()
            or np.max(limits[:,0]) > np.min(limits[:,1]) or not np.isfinite(jaw_gap_at_zero)):
        raise ValueError("a finite shared interval is required for the two finger joints")
    gap = np.linalg.norm(tip1-tip2, axis=1)
    opening_mask = _phase(use_opening, n) & np.isfinite(gap)
    # For this R1, the two opposite prismatic joints increase the internal
    # gap by q_left + q_right. Convert the metric prior to symmetric commands.
    q = (gap-jaw_gap_at_zero) * .5
    lo, hi = np.max(limits[:,0]), np.min(limits[:,1])
    grip[opening_mask] = np.clip(q[opening_mask, None], lo, hi)
    return pad, grip, {
        "orientation_rows": orientation_mask, "opening_rows": opening_mask,
        "anatomical_valid": valid, "human_aperture_m": gap,
        "opening_clipped": opening_mask & ((q < lo) | (q > hi)),
    }
