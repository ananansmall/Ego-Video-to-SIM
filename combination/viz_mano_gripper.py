#!/usr/bin/env python3
"""
Visualize MANO joints + wrist pose + analytical gripper pose from HaWoR data.
Standalone — no sapien dependency.

Usage:
    python viz_mano_gripper.py ~/data/hawor/7

Output:
    Saves a 3D PNG showing:
    - MANO wrist position (green sphere) + orientation (3 axes)
    - MANO thumb tip (j4, cyan) and index tip (j8, magenta)
    - Analytical gripper base pose (orange square) + orientation (3 axes)
    - Analytical gripper finger tips (purple/brown triangles)
    - Connecting lines: wrist->fingertips, gripper_base->finger_tips
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

PROJECT_ROOT = "/home/an/robot_world_ws/src/dex-retargeting/example/combination"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/combination/hand_track")

from gripper_config import compute_analytical_gripper_pose, FINGER_BASE_DIST

# R1 gripper URDF constants (from r1_gripper_right.yml)
FINGER_ORIGIN_X = 0.03689      # X offset from gripper_link to finger joint
FINGER_ORIGIN_Y = 0.013453     # Y offset from gripper_link to finger joint


# ============================================================================
# Standalone HaWoR data loader (no sapien dependency)
# ============================================================================

def _find_reconstruction_file(hawor_path):
    rec_dir = hawor_path / "reconstruction"
    if not rec_dir.exists():
        return None
    for f in rec_dir.glob("hawor_results_*_depth_aligned.npz"):
        return f
    for f in rec_dir.glob("hawor_results_*.npz"):
        return f
    return None


def load_hawor_data_standalone(hawor_dir, hand_idx=1):
    hawor_path = Path(str(hawor_dir))
    rec_file = _find_reconstruction_file(hawor_path)
    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        return {
            "pred_trans": rec['pred_trans'][hand_idx],
            "pred_rot": rec['pred_rot'][hand_idx],
            "pred_hand_pose": rec['pred_hand_pose'][hand_idx],
            "pred_betas": rec['pred_betas'][hand_idx],
            "pred_valid": rec['pred_valid'][hand_idx],
        }
    raise FileNotFoundError(f"Cannot find reconstruction file in {hawor_path}")


# ============================================================================
# MANO layer wrapper + fallback
# ============================================================================

def matrix_from_compact_axis_angle(aa):
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    c, s = np.cos(angle), np.sin(angle)
    cx, cy, cz = axis
    return np.array([
        [c + cx*cx*(1-c), cx*cy*(1-c) - cz*s, cx*cz*(1-c) + cy*s],
        [cy*cx*(1-c) + cz*s, c + cy*cy*(1-c), cy*cz*(1-c) - cx*s],
        [cz*cx*(1-c) - cy*s, cz*cy*(1-c) + cx*s, c + cz*cz*(1-c)]
    ])


def compute_mano_joints(rot, hand_pose, trans, betas):
    try:
        from mano_layer import MANOLayer
        mano_layer = MANOLayer("right", betas)
        v, j = mano_layer(rot, hand_pose, trans)
        return v, j
    except Exception as e:
        print(f"  WARNING: Could not use MANO layer: {e}")
        print("  Falling back to numpy approximation...")
        j = np.zeros((21, 3))
        j[0, :] = trans
        R = matrix_from_compact_axis_angle(rot)
        j[4, :] = trans + R[:, 0] * 0.08 + R[:, 1] * 0.03
        j[8, :] = trans + R[:, 0] * 0.09 + R[:, 1] * (-0.03)
        return np.zeros((6890, 3)), j


# ============================================================================
# Main visualization
# ============================================================================

def main(hawor_dir):
    hawor_dir = str(hawor_dir)
    print(f"Loading HaWoR data from: {hawor_dir}")

    hawor_data = load_hawor_data_standalone(hawor_dir, hand_idx=1)
    n_frames = len(hawor_data["pred_rot"])
    print(f"  Loaded {n_frames} frames, hand_idx=1 (right)")

    # Find first valid frame
    valid_mask = np.ones(n_frames, dtype=bool)
    for i in range(3):
        valid_mask &= ~np.isnan(hawor_data["pred_trans"][:, i])
    valid_mask &= ~np.any(np.isnan(hawor_data["pred_hand_pose"]), axis=1)

    first_valid = np.where(valid_mask)[0]
    if len(first_valid) == 0:
        print("  No valid frames found!")
        return
    frame_idx = first_valid[0]
    print(f"  Using first valid frame: {frame_idx}")

    rot = hawor_data["pred_rot"][frame_idx]
    trans = hawor_data["pred_trans"][frame_idx]
    hand_pose = hawor_data["pred_hand_pose"][frame_idx]
    betas = hawor_data["pred_betas"][frame_idx]

    print(f"  pred_rot: {rot}")
    print(f"  pred_trans: {trans}")

    # Compute MANO joints
    v, j = compute_mano_joints(rot, hand_pose, trans, betas)
    print(f"  MANO joints shape: {j.shape}")

    # Extract key points
    mango_wrist = j[0, :3]
    mango_thumb = j[4, :3]
    mango_index = j[8, :3]
    fi_dist = np.linalg.norm(mango_index - mango_thumb)

    print(f"\n=== MANO Joints ===")
    print(f"  Wrist (j0): {mango_wrist}")
    print(f"  Thumb (j4): {mango_thumb}")
    print(f"  Index (j8): {mango_index}")
    print(f"  Thumb-Index dist: {fi_dist:.6f}")
    print(f"  Wrist-Thumb dist: {np.linalg.norm(mango_thumb - mango_wrist):.6f}")
    print(f"  Wrist-Index dist: {np.linalg.norm(mango_index - mango_wrist):.6f}")

    # Compute gripper pose (analytical method)
    g_pos, g_R, joint1, joint2 = compute_analytical_gripper_pose(
        mango_wrist, mango_thumb, mango_index, "right")

    print(f"\n=== Analytical Gripper Pose ===")
    print(f"  Gripper base pos: {g_pos}")
    print(f"  Joint 1 (prismatic): {joint1:.6f}")
    print(f"  Joint 2 (prismatic): {joint2:.6f}")
    print(f"  Finger opening (j1+j2): {joint1+joint2:.6f}")
    print(f"  Finger tip distance: {2*FINGER_ORIGIN_Y + 2*joint1:.6f}")
    print(f"  (expected = thumb-index dist: {fi_dist:.6f})")
    print(f"  Gripper offset from wrist: {np.linalg.norm(g_pos - mango_wrist):.6f}")

    # Compute gripper finger tip world positions
    finger1_gripper = np.array([FINGER_ORIGIN_X, FINGER_ORIGIN_Y + joint1, 0.00012059])
    finger2_gripper = np.array([FINGER_ORIGIN_X, -(FINGER_ORIGIN_Y + joint2), 0.00012059])
    finger1_world = g_pos + g_R @ finger1_gripper
    finger2_world = g_pos + g_R @ finger2_gripper

    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    SCALE = 0.02

    # --- MANO Wrist (green sphere) ---
    ax.scatter([mango_wrist[0]], [mango_wrist[1]], [mango_wrist[2]],
               c='green', s=200, marker='o', label='MANO Wrist (j0)',
               edgecolors='black', depthshade=True)

    # MANO wrist orientation
    R_mano = matrix_from_compact_axis_angle(rot)
    for i, (color, label) in enumerate([('red', 'Wrist X'), ('blue', 'Wrist Y'), ('green', 'Wrist Z')]):
        ax.quiver(mango_wrist[0], mango_wrist[1], mango_wrist[2],
                  R_mano[0, i] * SCALE, R_mano[1, i] * SCALE, R_mano[2, i] * SCALE,
                  color=color, arrow_length_ratio=0.3, linewidths=2, label=label)

    # --- MANO fingertips ---
    ax.scatter([mango_thumb[0]], [mango_thumb[1]], [mango_thumb[2]],
               c='cyan', s=150, marker='o', label='MANO Thumb (j4)',
               edgecolors='black', depthshade=True)
    ax.scatter([mango_index[0]], [mango_index[1]], [mango_index[2]],
               c='magenta', s=150, marker='o', label='MANO Index (j8)',
               edgecolors='black', depthshade=True)

    # --- Gripper base (orange square) ---
    ax.scatter([g_pos[0]], [g_pos[1]], [g_pos[2]],
               c='orange', s=200, marker='s', label='Gripper Base',
               edgecolors='black', depthshade=True)

    # Gripper orientation
    for i, (color, label) in enumerate([('darkred', 'Gripper X'), ('darkblue', 'Gripper Y'), ('darkgreen', 'Gripper Z')]):
        ax.quiver(g_pos[0], g_pos[1], g_pos[2],
                  g_R[0, i] * SCALE, g_R[1, i] * SCALE, g_R[2, i] * SCALE,
                  color=color, arrow_length_ratio=0.3, linewidths=2, linestyle='--', label=label)

    # --- Gripper finger tips ---
    ax.scatter([finger1_world[0]], [finger1_world[1]], [finger1_world[2]],
               c='purple', s=150, marker='^', label='Gripper F1',
               edgecolors='black', depthshade=True)
    ax.scatter([finger2_world[0]], [finger2_world[1]], [finger2_world[2]],
               c='brown', s=150, marker='^', label='Gripper F2',
               edgecolors='black', depthshade=True)

    # --- Connecting lines ---
    ax.plot([mango_wrist[0], mango_thumb[0]], [mango_wrist[1], mango_thumb[1]],
            [mango_wrist[2], mango_thumb[2]], 'c--', linewidth=1.5, alpha=0.7, label='Wrist->Thumb')
    ax.plot([mango_wrist[0], mango_index[0]], [mango_wrist[1], mango_index[1]],
            [mango_wrist[2], mango_index[2]], 'm--', linewidth=1.5, alpha=0.7, label='Wrist->Index')
    ax.plot([g_pos[0], finger1_world[0]], [g_pos[1], finger1_world[1]],
            [g_pos[2], finger1_world[2]], 'purple', linewidth=2, alpha=0.7, label='Gripper->F1')
    ax.plot([g_pos[0], finger2_world[0]], [g_pos[1], finger2_world[1]],
            [g_pos[2], finger2_world[2]], 'brown', linewidth=2, alpha=0.7, label='Gripper->F2')
    ax.plot([mango_wrist[0], g_pos[0]], [mango_wrist[1], g_pos[1]],
            [mango_wrist[2], g_pos[2]], 'orange', linewidth=2, alpha=0.5, linestyle=':', label='Wrist->Gripper')

    # --- Labels and title ---
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    g_offset = np.linalg.norm(g_pos - mango_wrist)
    g_open = joint1 + joint2
    ax.set_title(
        f'MANO -> Gripper Retargeting (Analytical)\n'
        f'Frame {frame_idx}  |  Thumb-Index: {fi_dist:.4f}m  |  '
        f'Gripper Opening: {g_open:.4f}m  |  '
        f'Gripper Offset from Wrist: {g_offset:.4f}m'
    )

    ax.legend(loc='upper right', fontsize=7, ncol=2)

    # Auto-scaled limits
    all_points = np.array([mango_wrist, mango_thumb, mango_index, g_pos, finger1_world, finger2_world])
    pad = 0.02
    for i in range(3):
        lo, hi = all_points[:, i].min() - pad, all_points[:, i].max() + pad
        if i == 0:
            ax.set_xlim([lo, hi])
        elif i == 1:
            ax.set_ylim([lo, hi])
        else:
            ax.set_zlim([lo, hi])

    ax.view_init(elev=20, azim=45)

    output = "/home/an/robot_world_ws/src/dex-retargeting/example/combination/viz_mano_gripper.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hawor_dir", help="Path to HaWoR data directory")
    args = parser.parse_args()
    main(args.hawor_dir)
