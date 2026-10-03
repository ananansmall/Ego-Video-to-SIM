#!/usr/bin/env python3
"""
vis_depth_align.py — Visualize full alignment pipeline

Pipeline:
  01c (depth correction, HaWoR coords)
    → 001 (HaWoR → GLB coordinate alignment, scale + R + t)

Shows:
  1. Depth-corrected hand in HaWoR coords vs GLB objects (mismatch — different spaces)
  2. GLB-aligned hand in GLB coords vs GLB objects (matched — same space)
  3. Wrist depth (Z) per-frame comparison across both stages — line chart

Usage:
    python vis_depth_align.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result

"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh
import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ─── MANO skeleton connections ───
SKELETON = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]


def create_mano_layer(side, betas):
    """Create a MANO layer for the given hand side."""
    sys.path.insert(0, str(Path(__file__).parent / ".." / "position_retargeting"))
    from mano_layer import MANOLayer
    return MANOLayer(side, betas)


def compute_joints(mano_layer, rot, hand_pose, trans):
    """MANO FK: return (vertices, joints) as numpy arrays."""
    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = mano_layer(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]


# ─── Pipeline: 01c depth alignment ──────────────────────────────────────────

def run_depth_align(hawor_dir, ras_dir):
    """Run 01c_depth_align.py if result doesn't exist yet."""
    script = Path(__file__).parent / "01c_depth_align.py"
    rec_dir = Path(hawor_dir) / "reconstruction"
    npz_files = sorted(rec_dir.glob("hawor_results_*.npz"))
    orig_files = [f for f in npz_files if "_depth_aligned" not in str(f)]
    if not orig_files:
        print(f"[ERROR] No hawor_results_*.npz found in {rec_dir}")
        sys.exit(1)
    src = str(orig_files[0])
    dst = src.replace(".npz", "_depth_aligned.npz")

    if os.path.exists(dst):
        print(f"[OK] Depth-aligned result exists: {dst}")
        return dst

    print("=" * 60)
    print("Running 01c_depth_align.py ...")
    print("=" * 60)
    result = subprocess.run([
        sys.executable, str(script),
        "--hawor-dir", str(hawor_dir),
        "--ras-dir", str(ras_dir),
    ], cwd=str(script.parent))
    if result.returncode != 0:
        print("[WARN] Depth alignment failed, showing raw data only")
        return src
    return dst


# ─── Pipeline: 001 coordinate alignment ─────────────────────────────────────

def run_coord_align(hawor_dir, ras_dir):
    """Run 001_align_scene.py to get HaWoR → GLB coordinate transform."""
    script = Path(__file__).parent / "001_align_scene.py"
    out_dir = Path(str(hawor_dir)) / "output" / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer depth-aligned npz for coordinate alignment
    rec_dir = Path(hawor_dir) / "reconstruction"
    aligned_npz = sorted(rec_dir.glob("*_depth_aligned.npz"))
    if aligned_npz:
        src = str(aligned_npz[0])
    else:
        src = str(sorted(rec_dir.glob("hawor_results_*.npz"))[0])

    params_path = str(out_dir / "transform_params.npz")

    if os.path.exists(params_path):
        print(f"[OK] Transform params exist: {params_path}")
        return params_path

    print("=" * 60)
    print("Running 001_align_scene.py ...")
    print("=" * 60)
    result = subprocess.run([
        sys.executable, str(script),
        "--ras_output", str(ras_dir),
        "--hawor_reconstruction", src,
        "--output_dir", str(out_dir),
    ], cwd=str(script.parent))
    if result.returncode != 0:
        print("[WARN] Coordinate alignment failed")
        return None
    return params_path


def transform_to_glb(joints_hawor, scale_ratio, R_hand_to_glb, t_hand_to_glb):
    """Apply HaWoR → GLB transform: p_glb = s * R @ p_hawor + t

    Args:
        joints_hawor: (N_frames, 21, 3) joints in HaWoR world coords
        scale_ratio: GLB/HaWoR scale factor
        R_hand_to_glb: (3,3) rotation matrix
        t_hand_to_glb: (3,) translation vector

    Returns:
        (N_frames, 21, 3) joints in GLB coords
    """
    n = joints_hawor.shape[0]
    joints_glb = np.zeros_like(joints_hawor, dtype=np.float64)
    for i in range(n):
        pts = joints_hawor[i]  # (21, 3)
        joints_glb[i] = scale_ratio * (R_hand_to_glb @ pts.T).T + t_hand_to_glb
    return joints_glb


# ─── Data loading ───────────────────────────────────────────────────────────

def load_glb_centroids(glb_path):
    """Load GLB scene, return list of (name, centroid, bounds)."""
    scene = trimesh.load(glb_path)
    geoms = []
    if hasattr(scene, "geometry"):
        for name, geom in scene.geometry.items():
            centroid = geom.vertices.mean(axis=0)
            geoms.append((name, centroid, geom.bounds))
    else:
        centroid = scene.vertices.mean(axis=0)
        geoms.append(("scene", centroid, scene.bounds))
    return geoms


def load_data(npz_path, hand_idx, num_samples=50):
    """Load MANO params and compute hand joints for sampled valid frames.

    Returns:
        all_joints: (N, 21, 3) hand joints in world space (HaWoR or GLB)
        wrist_only: (N, 3) wrist positions
        hand_label: str
        sample_indices: frame indices used
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    pred_trans = data["pred_trans"]
    pred_rot = data["pred_rot"]
    pred_hand_pose = data["pred_hand_pose"]
    pred_betas = data["pred_betas"]
    pred_valid = data["pred_valid"]

    # Auto-detect hand
    if hand_idx < 0:
        hand_idx = 0 if pred_valid[0].any() else 1
        if pred_valid[0].any() and pred_valid[1].any():
            print("  Both hands valid, showing left (idx=0)")
            hand_idx = 0

    hand_label = "left" if hand_idx == 0 else "right"
    side = "left" if hand_idx == 0 else "right"

    valid_indices = np.where(pred_valid[hand_idx])[0]
    if len(valid_indices) == 0:
        print(f"[ERROR] No valid frames for hand {hand_label}")
        sys.exit(1)

    step = max(1, len(valid_indices) // num_samples)
    sample_indices = valid_indices[::step][:num_samples]

    betas_mean = pred_betas[hand_idx, valid_indices[0]].astype(np.float32)
    try:
        mano_layer = create_mano_layer(side, betas_mean)
    except Exception as e:
        print(f"[ERROR] Failed to create MANO layer: {e}")
        sys.exit(1)

    all_joints = []
    wrist_only = []
    for fi in sample_indices:
        rot = pred_rot[hand_idx, fi]
        trans = pred_trans[hand_idx, fi]
        hp = pred_hand_pose[hand_idx, fi]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)):
            continue
        _, joints = compute_joints(mano_layer, rot, hp, trans)
        all_joints.append(joints)
        wrist_only.append(joints[0])

    return np.array(all_joints), np.array(wrist_only), hand_label, list(sample_indices)


# ─── Plotting ────────────────────────────────────────────────────────────────

def plot_skeleton_3d(ax, joints, color, alpha=0.6, linewidth=1.0):
    """Draw hand skeleton on an existing 3D axis."""
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
               c=color, s=15, alpha=alpha, zorder=3)
    for (i, j) in SKELETON:
        ax.plot([joints[i, 0], joints[j, 0]],
                [joints[i, 1], joints[j, 1]],
                [joints[i, 2], joints[j, 2]],
                color=color, alpha=alpha, linewidth=linewidth)


def plot_bounding_boxes(ax, geoms, color="gray", alpha=0.3):
    """Draw GLB object bounding boxes."""
    n_objs = len(geoms)
    cmap = plt.cm.tab10(np.linspace(0, 1, max(n_objs, 1)))
    for i, (name, centroid, bounds) in enumerate(geoms):
        c = cmap[i]
        corners = np.array([
            [bounds[0, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[0, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[0, 2]],
            [bounds[0, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[0, 1], bounds[1, 2]],
            [bounds[1, 0], bounds[1, 1], bounds[1, 2]],
            [bounds[0, 0], bounds[1, 1], bounds[1, 2]],
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0),
                 (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        for e in edges:
            ax.plot3D(*corners[[e[0], e[1]]].T, color=c,
                      alpha=alpha, linewidth=0.5)
        ax.scatter(*centroid, color=c, s=120, marker="o",
                   edgecolors="k", linewidths=0.6, zorder=5)


def plot_coordinate_axis(ax, origin, scale=0.15, title="",
                         z_color="b", y_color="g", x_color="r", zdir=(0, 0, 1)):
    """Draw XYZ coordinate axis with arrows at given origin.

    X = red, Y = green, Z = blue
    """
    x_axis = np.array([scale, 0, 0])
    y_axis = np.array([0, scale, 0])
    z_axis = np.array([0, 0, scale])
    ax.quiver(*origin, *x_axis, color=x_color, arrow_length_ratio=0.3,
              linewidth=3, alpha=0.9, label="X")
    ax.quiver(*origin, *y_axis, color=y_color, arrow_length_ratio=0.3,
              linewidth=3, alpha=0.9, label="Y")
    ax.quiver(*origin, *z_axis, color=z_color, arrow_length_ratio=0.3,
              linewidth=3, alpha=0.9, label="Z")
    if title:
        ax.text(origin[0] + 0.01, origin[1] + 0.01, origin[2] + 0.01,
                title, fontsize=12, fontweight="bold",
                color=(0.2, 0.2, 0.2))


def plot_coord_system_comparison(hawor_joints, glb_joints, geoms, hand_label):
    """Side-by-side 3D comparison of HaWoR and GLB coordinate systems.

    Left: HaWoR space — hand + objects in HaWoR coords (mismatched)
    Right: GLB space — hand + objects in GLB coords (matched)
    Shows XYZ axes at origin with scale markers.
    """
    fig = plt.figure(figsize=(24, 10))

    # ─── Left: HaWoR coordinate system ───
    ax1 = fig.add_subplot(121, projection="3d")

    # Hand skeleton (subsampled)
    n_show = min(10, len(hawor_joints))
    step = max(1, n_show // 3)
    for i in range(0, n_show, step):
        alpha_s = 0.4 + 0.5 * (i / n_show)
        plot_skeleton_3d(ax1, hawor_joints[i], "lime",
                         alpha=alpha_s, linewidth=0.7)

    # Object bounding boxes (in HaWoR coords for comparison)
    plot_bounding_boxes(ax1, geoms, alpha=0.3)

    # Coordinate axis at origin
    plot_coordinate_axis(ax1, [0, 0, 0], scale=0.12,
                         title="HaWoR origin", z_color="b", y_color="g", x_color="r")

    # Origin marker
    ax1.scatter(0, 0, 0, c="black", s=50, marker="+", linewidths=2, zorder=10)

    ax1.set_xlabel("X (m)", fontsize=10)
    ax1.set_ylabel("Y (m)", fontsize=10)
    ax1.set_zlabel("Z (m)", fontsize=10)
    ax1.set_title(f"HaWoR Coordinate System — {hand_label} hand\n(green=hand, boxes=GLB objects in HaWoR space)",
                  fontsize=11, fontweight="bold", color="dimgray")
    ax1.legend(loc="upper left", fontsize=8)

    # Equal aspect
    pts_hawor = np.vstack([hawor_joints[:n_show, 0], np.array([g[1] for g in geoms])])
    mid = pts_hawor.mean(axis=0)
    max_range = max(pts_hawor.ptp(axis=0)) / 2 * 1.2
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # ─── Right: GLB coordinate system ───
    ax2 = fig.add_subplot(122, projection="3d")

    # Hand skeleton (GLB coords, subsampled)
    for i in range(0, n_show, step):
        alpha_s = 0.4 + 0.5 * (i / n_show)
        plot_skeleton_3d(ax2, glb_joints[i], "lime",
                         alpha=alpha_s, linewidth=0.7)

    # Object bounding boxes (in GLB coords)
    plot_bounding_boxes(ax2, geoms, alpha=0.3)

    # Coordinate axis at origin
    plot_coordinate_axis(ax2, [0, 0, 0], scale=0.12,
                         title="GLB origin", z_color="b", y_color="g", x_color="r")

    # Origin marker
    ax2.scatter(0, 0, 0, c="black", s=50, marker="+", linewidths=2, zorder=10)

    ax2.set_xlabel("X (m)", fontsize=10)
    ax2.set_ylabel("Y (m)", fontsize=10)
    ax2.set_zlabel("Z (m)", fontsize=10)
    ax2.set_title(f"GLB Coordinate System — {hand_label} hand\n(green=hand, boxes=GLB objects in GLB space)",
                  fontsize=11, fontweight="bold", color="dimgray")
    ax2.legend(loc="upper left", fontsize=8)

    # Equal aspect
    pts_glb = np.vstack([glb_joints[:n_show, 0], np.array([g[1] for g in geoms])])
    mid = pts_glb.mean(axis=0)
    max_range = max(pts_glb.ptp(axis=0)) / 2 * 1.2
    ax2.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax2.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax2.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    return fig


def plot_3d_hand_glb(aligned_joints_list, geoms, R_c2w, t_c2w,
                     hand_label, title="GLB-aligned hand + objects"):
    """3D: GLB-aligned hand skeleton + GLB objects (same coordinate space).

    aligned_joints_list: (N_frames, 21, 3) joints in GLB coords
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    n_show = min(15, len(aligned_joints_list))
    step = max(1, n_show // 5)

    # Hand skeletons (subsampled frames, fading)
    for i in range(0, n_show, step):
        alpha_s = 0.3 + 0.7 * (i / n_show)
        plot_skeleton_3d(ax, aligned_joints_list[i], "lime",
                         alpha=alpha_s * 0.7, linewidth=0.8)

    # Wrist trajectory
    wrists = aligned_joints_list[:n_show, 0]
    ax.scatter(wrists[:, 0], wrists[:, 1], wrists[:, 2],
               c="lime", s=6, alpha=0.4, label=f"wrist trajectory")

    # GLB bounding boxes + centroids
    plot_bounding_boxes(ax, geoms, alpha=0.35)

    # Camera trajectory
    if R_c2w is not None and t_c2w is not None:
        cam_step = max(1, len(t_c2w) // 25)
        for fi in range(0, len(t_c2w), cam_step):
            cam_pos = t_c2w[fi]
            ax.scatter(*cam_pos, color="blue", s=8, alpha=0.15, marker="^")
        ax.scatter([], [], [], color="blue", s=10, marker="^",
                   label="camera poses (RAS)")

    ax.plot([], [], "lime", alpha=0.7, label="GLB-aligned hand")
    # Legend placeholder — actual colors come from plot_bounding_boxes
    ax.scatter([], [], [], c="tab:blue", s=80, label="GLB object centroids")

    ax.set_xlabel("X (m)", fontsize=11)
    ax.set_ylabel("Y (m)", fontsize=11)
    ax.set_zlabel("Z (m)", fontsize=11)
    ax.set_title(f"{title} — {hand_label} hand\nGLB coordinate space (001 aligned)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.8, ncol=2)

    # Equal aspect
    pts_list = [wrists] + [g[1] for g in geoms]
    all_pts = np.vstack(pts_list)
    mid = all_pts.mean(axis=0)
    max_range = max(all_pts.ptp(axis=0)) / 2 * 1.2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    return fig


def plot_depth_comparison(orig_wrists, corr_wrists, glb_wrists):
    """Wrist Z-depth per frame: original → depth-corrected → GLB-aligned."""
    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(orig_wrists)
    frames = np.arange(n)

    ax.plot(frames, orig_wrists[:, 2], "r-", alpha=0.7, linewidth=0.8,
            label=f"original (HaWoR, mean={orig_wrists[:,2].mean():.3f}m)")
    ax.plot(frames, corr_wrists[:, 2], "g-", alpha=0.7, linewidth=0.8,
            label=f"depth-corrected (HaWoR, mean={corr_wrists[:,2].mean():.3f}m)")
    if glb_wrists is not None:
        ax.plot(frames, glb_wrists[:, 2], "b-", alpha=0.7, linewidth=0.8,
                label=f"GLB-aligned (mean={glb_wrists[:,2].mean():.3f}m)")
    ax.set_xlabel("sampled frame index")
    ax.set_ylabel("wrist Z (m)")
    ax.set_title("Wrist Z per stage: original → depth-corrected → GLB-aligned")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize full alignment pipeline")
    parser.add_argument("--hawor-dir", default="/home/an/data/hawor/7")
    parser.add_argument("--ras-dir", default="/home/an/data/ras/my_7mp4_result")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="hand index: 0=left, 1=right, -1=auto")
    parser.add_argument("--no-depth", action="store_true",
                        help="skip depth alignment (01c)")
    parser.add_argument("--no-coord", action="store_true",
                        help="skip coordinate alignment (001)")
    parser.add_argument("--save", type=str, default=None,
                        help="save path prefix (e.g. /tmp/depth_align)")
    parser.add_argument("--num-frames", type=int, default=50,
                        help="number of frames to sample (default 50)")
    args = parser.parse_args()

    hawor_dir = Path(args.hawor_dir)
    ras_dir = Path(args.ras_dir)
    glb_path = ras_dir / "final_scene.glb"

    # ── Step 1: depth alignment (01c) ──
    if args.no_depth:
        rec_dir = hawor_dir / "reconstruction"
        npz_files = [f for f in rec_dir.glob("hawor_results_*.npz")
                     if "_depth_aligned" not in str(f)]
        if not npz_files:
            print(f"[ERROR] No hawor_results_*.npz in {rec_dir}")
            sys.exit(1)
        orig_path = str(npz_files[0])
        corr_path = orig_path
        print("[WARN] --no-depth: using raw data only")
    else:
        corr_path = run_depth_align(hawor_dir, ras_dir)
        orig_dir = hawor_dir / "reconstruction"
        orig_files = [f for f in orig_dir.glob("hawor_results_*.npz")
                      if "_depth_aligned" not in str(f)]
        if not orig_files:
            print("[ERROR] No original hawor_results_*.npz found")
            sys.exit(1)
        orig_path = str(orig_files[0])

    # ── Step 2: coordinate alignment (001) ──
    params_path = None
    if not args.no_coord:
        params_path = run_coord_align(hawor_dir, ras_dir)

    # ── Step 3: compute MANO joints ──
    print("\n" + "=" * 60)
    print("Computing MANO hand joints ...")
    print("=" * 60)

    orig_joints, orig_wrists, hand_label, sample_idx = load_data(
        orig_path, args.hand_idx, args.num_frames)
    corr_joints, corr_wrists, _, _ = load_data(
        corr_path, args.hand_idx, args.num_frames)

    print(f"  Hand: {hand_label}")
    print(f"  Original frames: {len(orig_joints)}")
    print(f"  Depth-corrected frames: {len(corr_joints)}")

    # ── Step 4: apply coordinate transform if params available ──
    glb_joints = None
    glb_wrists = None
    if params_path and os.path.exists(params_path):
        params = dict(np.load(params_path, allow_pickle=True))
        s = float(params["scale_ratio"])
        R_h2g = params["R_hand_to_glb"]
        t_h2g = params["t_hand_to_glb"]
        print(f"\n  Applying HaWoR → GLB transform (scale_ratio={s:.4f})")
        glb_joints = transform_to_glb(corr_joints, s, R_h2g, t_h2g)
        glb_wrists = glb_joints[:, 0, :]
        print(f"  GLB-aligned frames: {len(glb_joints)}")
        print(f"  Wrist Z: orig={orig_wrists[:,2].mean():.4f}m → "
              f"depth={corr_wrists[:,2].mean():.4f}m → "
              f"GLB={glb_wrists[:,2].mean():.4f}m")
    else:
        print("\n  [WARN] No coordinate params, skipping GLB transform")

    # ── Step 5: load GLB objects ──
    if glb_path.exists():
        geoms = load_glb_centroids(str(glb_path))
        print(f"  GLB objects: {len(geoms)}")
        for i, (n, c, b) in enumerate(geoms):
            print(f"    obj{i}: centroid=({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
    else:
        print(f"[WARN] GLB not found: {glb_path}")
        geoms = []

    # ── Step 6: camera params ──
    raw_data = dict(np.load(orig_path, allow_pickle=True))
    R_c2w = raw_data.get("R_c2w", None)
    t_c2w = raw_data.get("t_c2w", None)

    # ── Step 7: correction factors ──
    factor_path = corr_path.replace(".npz", "_factors.npz")
    if os.path.exists(factor_path):
        fdata = dict(np.load(factor_path, allow_pickle=True))
        factors = fdata.get("factors", None)
        if factors is not None:
            print(f"  Correction factors: mean={factors.mean():.4f}, "
                  f"range=[{factors.min():.4f}, {factors.max():.4f}]")

    # ── Step 8: generate plots ──
    print("\n" + "=" * 60)
    print("Generating plots ...")
    print("=" * 60)

    save_prefix = args.save if args.save else "/tmp/depth_align"
    figs = []

    # Fig 1: depth-corrected hand vs GLB objects (HaWoR coords, mismatch)
    fig1 = plot_3d_hand_glb(
        corr_joints, geoms, R_c2w, t_c2w, hand_label,
        title="Depth-corrected hand (HaWoR) vs GLB objects — NOT aligned")
    figs.append((save_prefix + "_01_depth_hawor_3d", fig1))

    # Fig 1b: coordinate system comparison (HaWoR vs GLB, side by side)
    if glb_joints is not None:
        fig1b = plot_coord_system_comparison(
            corr_joints, glb_joints, geoms, hand_label)
        figs.append((save_prefix + "_01b_coord_systems", fig1b))

    # Fig 2: GLB-aligned hand vs GLB objects (GLB coords, matched)
    if glb_joints is not None:
        fig2 = plot_3d_hand_glb(
            glb_joints, geoms, R_c2w, t_c2w, hand_label,
            title="GLB-aligned hand vs GLB objects — same space")
        figs.append((save_prefix + "_02_glb_3d", fig2))

    # Fig 3: depth chart
    fig3 = plot_depth_comparison(orig_wrists, corr_wrists, glb_wrists)
    figs.append((save_prefix + "_depth", fig3))

    for path, fig in figs:
        plt.show()

    print("\n[DONE]")


if __name__ == "__main__":
    main()
