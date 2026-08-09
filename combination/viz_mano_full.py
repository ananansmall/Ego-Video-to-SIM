#!/usr/bin/env python3
"""
Visualize all 21 MANO joints + wrist pose + analytical gripper pose.
From HaWoR data (~/data/hawor/7).

Usage:
    python viz_mano_full.py ~/data/hawor/7

Output:
    Saves PNG showing:
    - All 21 MANO joints (numbered spheres + kinematic skeleton)
    - MANO wrist orientation (3 axes from pred_rot)
    - Analytical gripper base (orange square) + orientation + finger tips
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
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting")

from gripper_config import compute_analytical_gripper_pose, FINGER_BASE_DIST

FINGER_ORIGIN_X = 0.03689
FINGER_ORIGIN_Y = 0.013453

# ============================================================================
# MANO joint kinematic skeleton
# j0=Wrist, j1-4=Thumb, j5-8=Index, j9-12=Middle, j13-16=Ring, j17-20=Little
# ============================================================================
MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),   # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Little
]

MANO_JOINT_NAMES = [
    "0:Wrist", "1:T_CMC", "2:T_MCP", "3:T_IP", "4:T_TIP",
    "5:I_CMC", "6:I_MCP", "7:I_PIP", "8:I_DIP",
    "9:M_CMC", "10:M_MCP", "11:M_PIP", "12:M_DIP",
    "13:R_CMC", "14:R_MCP", "15:R_PIP", "16:R_DIP",
    "17:L_CMC", "18:L_MCP", "19:L_PIP", "20:L_DIP",
]

FINGER_TIP_INDICES = [4, 8, 12, 16, 20]  # T_TIP, I_DIP, M_DIP, R_DIP, L_DIP

# ============================================================================
# Standalone HaWoR loader
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
    raise FileNotFoundError(f"No reconstruction file in {hawor_path}")

# ============================================================================
# MANO FK
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
        import torch
        mano_layer = MANOLayer("right", betas)
        p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
        t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
        v, j = mano_layer(p, t)
        return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]
    except Exception as e:
        print(f"  WARNING: MANO layer failed: {e}")
        j = np.zeros((21, 3))
        j[0] = trans
        R = matrix_from_compact_axis_angle(rot)
        # Approximate finger tip positions (very rough)
        j[4] = trans + R[:, 0] * 0.08 + R[:, 1] * 0.03
        j[8] = trans + R[:, 0] * 0.09 + R[:, 1] * (-0.03)
        j[12] = trans + R[:, 0] * 0.10
        j[16] = trans + R[:, 0] * 0.09 + R[:, 1] * 0.02
        j[20] = trans + R[:, 0] * 0.07 + R[:, 1] * (-0.04)
        return np.zeros((6890, 3)), j

# ============================================================================
# Main
# ============================================================================

def main(hawor_dir):
    hawor_dir = str(hawor_dir)
    print(f"Loading HaWoR data from: {hawor_dir}")

    hawor_data = load_hawor_data_standalone(hawor_dir, hand_idx=1)
    n_frames = len(hawor_data["pred_rot"])
    print(f"  Loaded {n_frames} frames, hand_idx=1 (right)")

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
    mano_joints = j
    mango_wrist = j[0, :3]
    mango_thumb = j[4, :3]
    mango_index = j[8, :3]
    fi_dist = np.linalg.norm(mango_index - mango_thumb)

    # Print all 21 joints
    print(f"\n=== All 21 MANO Joints ===")
    for i in range(21):
        print(f"  {MANO_JOINT_NAMES[i]}: [{j[i, 0]:+.6f}, {j[i, 1]:+.6f}, {j[i, 2]:+.6f}]")

    # Print fingertip distances
    print(f"\n=== Fingertip Distances ===")
    for i, idx in enumerate(FINGER_TIP_INDICES):
        if i == 0:
            continue
        d = np.linalg.norm(j[idx, :3] - j[0, :3])
        print(f"  Wrist->FingerTip[{idx}]: {d:.6f}m")

    # Thumb-Index opening
    for idx in [4, 8, 12, 16, 20]:
        d = np.linalg.norm(j[idx, :3] - j[0, :3])
        print(f"  {MANO_JOINT_NAMES[idx]} from wrist: {d:.6f}m")

    print(f"\n=== Key Measurements ===")
    print(f"  Thumb(4)-Index(8) dist: {fi_dist:.6f}m")
    print(f"  Thumb(4)-Wrist(0) dist: {np.linalg.norm(j[4,:3] - j[0,:3]):.6f}m")
    print(f"  Index(8)-Wrist(0) dist: {np.linalg.norm(j[8,:3] - j[0,:3]):.6f}m")

    # Compute gripper pose (analytical method)
    g_pos, g_R, joint1, joint2 = compute_analytical_gripper_pose(
        mango_wrist, mango_thumb, mango_index, "right")

    print(f"\n=== Analytical Gripper Pose ===")
    print(f"  Gripper base pos: {g_pos}")
    print(f"  Joint 1: {joint1:.6f}, Joint 2: {joint2:.6f}")
    print(f"  Finger opening (j1+j2): {joint1+joint2:.6f}")
    print(f"  Finger tip distance: {2*FINGER_ORIGIN_Y + 2*joint1:.6f}")
    print(f"  Gripper offset from wrist: {np.linalg.norm(g_pos - mango_wrist):.6f}")
    print(f"  FINGER_ORIGIN_X: {FINGER_ORIGIN_X}")
    print(f"  FINGER_BASE_DIST: {FINGER_BASE_DIST}")
    print(f"  (finger_dist - FINGER_BASE_DIST)/2 = {(fi_dist - FINGER_BASE_DIST)/2:.6f} = joint")

    # Gripper finger tips
    finger1_gripper = np.array([FINGER_ORIGIN_X, FINGER_ORIGIN_Y + joint1, 0.00012059])
    finger2_gripper = np.array([FINGER_ORIGIN_X, -(FINGER_ORIGIN_Y + joint2), 0.00012059])
    finger1_world = g_pos + g_R @ finger1_gripper
    finger2_world = g_pos + g_R @ finger2_gripper

    # ========================================================================
    # Create 3D plot
    # ========================================================================
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    SCALE = 0.015  # axis arrow length

    # --- All 21 MANO joints (numbered spheres) ---
    for i in range(21):
        is_tip = i in FINGER_TIP_INDICES
        is_wrist = i == 0
        color = 'green' if is_wrist else ('gold' if is_tip else 'steelblue')
        size = 180 if is_wrist or is_tip else 80
        alpha = 1.0
        ax.scatter([j[i, 0]], [j[i, 1]], [j[i, 2]], c=color, s=size,
                   marker='o', edgecolors='black', depthshade=True, alpha=alpha)
        # Label each joint
        ax.text(j[i, 0], j[i, 1], j[i, 2]+0.003, str(i), fontsize=7,
                ha='center', va='bottom')

    # --- MANO kinematic skeleton ---
    for parent, child in MANO_SKELETON:
        ax.plot([j[parent, 0], j[child, 0]],
                [j[parent, 1], j[child, 1]],
                [j[parent, 2], j[child, 2]],
                'gray', linewidth=1.5, alpha=0.6)

    # --- MANO wrist orientation (from pred_rot) ---
    R_mano = matrix_from_compact_axis_angle(rot)
    for i, (color, label) in enumerate([('red', 'Wrist X'), ('blue', 'Wrist Y'), ('green', 'Wrist Z')]):
        ax.quiver(mango_wrist[0], mango_wrist[1], mango_wrist[2],
                  R_mano[0, i] * SCALE, R_mano[1, i] * SCALE, R_mano[2, i] * SCALE,
                  color=color, arrow_length_ratio=0.3, linewidths=2, label=label)

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
               c='purple', s=180, marker='^', label='Gripper F1',
               edgecolors='black', depthshade=True)
    ax.scatter([finger2_world[0]], [finger2_world[1]], [finger2_world[2]],
               c='brown', s=180, marker='^', label='Gripper F2',
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
            [mango_wrist[2], g_pos[2]], 'orange', linewidth=2.5, alpha=0.5, linestyle=':', label='Wrist->Gripper')

    # --- Annotations: distances ---
    mid_thum_index = (mango_thumb + mango_index) / 2
    ax.text(mid_thum_index[0], mid_thum_index[1], mid_thum_index[2] - 0.005,
            f'{fi_dist*1000:.1f}mm', fontsize=9, color='purple', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    mid_wrist_gripper = (mango_wrist + g_pos) / 2
    g_offset = np.linalg.norm(g_pos - mango_wrist)
    ax.text(mid_wrist_gripper[0], mid_wrist_gripper[1], mid_wrist_gripper[2] + 0.005,
            f'{g_offset*1000:.1f}mm', fontsize=9, color='darkorange', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # --- Labels and title ---
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    g_open = joint1 + joint2
    ax.set_title(
        f'MANO (21 joints) -> Gripper Retargeting\n'
        f'Frame {frame_idx} | Thumb-Index: {fi_dist*1000:.1f}mm | Gripper Open: {g_open*1000:.1f}mm | '
        f'Gripper Offset: {g_offset*1000:.1f}mm | J1=J2={joint1*1000:.1f}mm'
    )

    ax.legend(loc='upper left', fontsize=7, ncol=2)

    # Auto-scale limits
    all_pts = np.vstack([j, g_pos, finger1_world, finger2_world])
    pad = 0.01
    for i in range(3):
        lo, hi = all_pts[:, i].min() - pad, all_pts[:, i].max() + pad
        if i == 0:
            ax.set_xlim([lo, hi])
        elif i == 1:
            ax.set_ylim([lo, hi])
        else:
            ax.set_zlim([lo, hi])

    ax.view_init(elev=20, azim=45)

    output = "/home/an/robot_world_ws/src/dex-retargeting/example/combination/viz_mano_full.png"
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches='tight')
    print(f"\nSaved: {output}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hawor_dir", help="Path to HaWoR data directory")
    args = parser.parse_args()
    main(args.hawor_dir)
