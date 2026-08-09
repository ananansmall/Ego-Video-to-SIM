#!/usr/bin/env python3
"""vis_stage2_trajectory.py — Visualize Stage 2 trajectory: F44~F56 MANO→F50→MANO transition

Shows:
  Panel 1 (3D): Stage 2 trajectory + MANO+offset trajectory + F50 point + object position
  Panel 2: Position X/Y/Z vs frame number for both Stage 2 and MANO+offset
  Panel 3: Gripper qpos vs frame number (边走边夹 + 保持夹持 pattern)

Usage:
    python vis_stage2_trajectory.py
"""
import sys
from pathlib import Path

import numpy as np

import matplotlib
try:
    import tkinter
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from trajectory_loader import compute_mano_joints, compute_analytical_gripper_pose
from physics_utils import RXWORLD_TO_SAPIEN

# ── Paths ──
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output" / "gripper_only_right"
STAGE2_PATH = OUTPUT_DIR / "stage2" / "stage2_recon.npz"
BEST_GRASP_PATH = OUTPUT_DIR / "stage1" / "best_grasp.npz"
TRANSFORM_PATH = OUTPUT_DIR / "alignment" / "transform_params.npz"
HAWOR_DIR = Path("/home/an/data/hawor/7")
RAS_DIR = Path("/home/an/data/ras/my_7mp4_result")

# MANO model paths
MANO_MODEL_CANDIDATES = [
    BASE_DIR / ".." / "position_retargeting" / "manopth" / "mano_v1_2" / "models",
    Path.home() / "robot_world_ws" / "src" / "dex-retargeting" / "example" / "position_retargeting" / "manopth" / "mano_v1_2" / "models",
]
MANO_PY_CANDIDATES = [
    BASE_DIR / ".." / "position_retargeting",
    Path.home() / "robot_world_ws" / "src" / "dex-retargeting" / "example" / "position_retargeting",
]


def find_mano_model():
    for c in MANO_MODEL_CANDIDATES:
        if list(c.glob("*.pkl")):
            return c
    return None


def create_mano_layer(side, betas):
    for mp in MANO_PY_CANDIDATES:
        sp = str(mp)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    from mano_layer import MANOLayer
    return MANOLayer(side, betas)


def find_closest_glb_object_sapien(glb_path, ref_pos, s_inv, R_inv, t_inv):
    """Load GLB, transform all objects to SAPIEN space, return the closest to ref_pos."""
    import trimesh
    scene = trimesh.load(str(glb_path))
    best_pos = None
    best_dist = float('inf')
    if hasattr(scene, "geometry"):
        for geom_key, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                c_glb = geom.vertices.mean(axis=0)
                # Same transform chain as trajectory_loader._load_glb_sapien
                p_hawor = s_inv * (R_inv @ c_glb) + t_inv
                p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
                dist = np.linalg.norm(p_sapien - ref_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_pos = p_sapien.copy()
    return best_pos, best_dist


def compute_mano_gripper_traj(hawor_data, hand_idx, start_frame, num_frames, side="right"):
    """Compute MANO gripper base trajectory in SAPIEN space."""
    betas = hawor_data["pred_betas"][hand_idx, start_frame].astype(np.float32)
    mano_layer = create_mano_layer(side, betas)

    traj = {"pos": [], "R": [], "j1": [], "j2": []}
    for fi in range(start_frame, start_frame + num_frames):
        if fi >= hawor_data["pred_trans"].shape[1]:
            break
        try:
            _, joints = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][hand_idx, fi],
                hawor_data["pred_hand_pose"][hand_idx, fi],
                hawor_data["pred_trans"][hand_idx, fi],
            )
            joints_sapien = (RXWORLD_TO_SAPIEN @ joints.T).T
            wrist = joints_sapien[0, :3]
            f1 = joints_sapien[4, :3]
            f2 = joints_sapien[8, :3]
            root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(wrist, f1, f2)
            traj["pos"].append(root_pos)
            traj["R"].append(root_R)
            traj["j1"].append(j1)
            traj["j2"].append(j2)
        except Exception as e:
            print(f"  [WARN] F{fi}: failed: {e}")
            continue

    traj["pos"] = np.array(traj["pos"], dtype=np.float64)
    traj["R"] = np.array(traj["R"], dtype=np.float64)
    return traj


def main():
    print("=" * 60)
    print("Stage 2 Trajectory Visualization: F44~F56 MANO→F50→MANO")
    print("=" * 60)

    # ── 1. Load Stage 2 reconstruction ──
    print("\n[1] Loading Stage 2 reconstruction...")
    stage2 = dict(np.load(str(STAGE2_PATH), allow_pickle=True))
    s2_frames = stage2["frames"]
    s2_pos = stage2["pos"]
    s2_qpos = stage2["gripper_qpos"]
    print(f"  Frames: {s2_frames[0]}~{s2_frames[-1]} ({len(s2_frames)} frames)")
    print(f"  Pos range: X=[{s2_pos[:,0].min():.4f},{s2_pos[:,0].max():.4f}]")
    print(f"  Qpos range: [{s2_qpos.min():.4f},{s2_qpos.max():.4f}]")

    # ── 2. Load Stage 1 best grasp ──
    print("\n[2] Loading Stage 1 best grasp...")
    best_grasp = dict(np.load(str(BEST_GRASP_PATH), allow_pickle=True))
    grasp_pos = best_grasp["pos"]
    grasp_qpos = float(best_grasp["gripper_qpos"])
    print(f"  Best grasp pos: {grasp_pos.round(4)}")
    print(f"  Best grasp qpos: {grasp_qpos:.4f}")

    # ── 3. Load transform params ──
    print("\n[3] Loading transform params...")
    tp = dict(np.load(str(TRANSFORM_PATH), allow_pickle=True))
    s_inv = float(tp["s_inv"])
    R_inv = tp["R_inv"]
    t_inv = tp["t_inv"]
    print(f"  s_inv={s_inv:.4f}")

    # ── 4. Compute MANO gripper trajectory for F44~F56 ──
    print("\n[4] Computing MANO gripper trajectory...")
    rec_dir = HAWOR_DIR / "reconstruction"
    npz_files = sorted(rec_dir.glob("hawor_results_*.npz"))
    if not npz_files:
        print(f"[ERROR] No npz files in {rec_dir}")
        sys.exit(1)
    hawor_data = dict(np.load(str(npz_files[0]), allow_pickle=True))
    print(f"  Loaded: {npz_files[0].name}")

    hand_idx = 1  # right hand
    side = "right"
    start_frame = int(s2_frames[0])
    num_frames = int(s2_frames[-1] - s2_frames[0] + 1)
    print(f"  Computing frames {start_frame}~{start_frame+num_frames-1}...")
    mano_traj = compute_mano_gripper_traj(hawor_data, hand_idx, start_frame, num_frames, side)
    n_mano = len(mano_traj["pos"])
    print(f"  Computed {n_mano} frames")
    print(f"  MANO pos range: X=[{mano_traj['pos'][:,0].min():.4f},{mano_traj['pos'][:,0].max():.4f}]")

    # ── 5. Compute _mano_neutral_offset ──
    print("\n[5] Computing _mano_neutral_offset...")

    # Get object position: find GLB object closest to best_grasp pos
    obj_pos = None
    glb_path = RAS_DIR / "final_scene.glb"
    if glb_path.exists():
        obj_pos, obj_dist = find_closest_glb_object_sapien(
            str(glb_path), grasp_pos, s_inv, R_inv, t_inv)
        if obj_pos is not None:
            print(f"  Object (GLB→SAPIEN, closest to grasp): {obj_pos.round(4)}, dist={obj_dist:.4f}")

    if obj_pos is None:
        obj_pos = grasp_pos.copy()
        print(f"  Object (from best_grasp): {obj_pos.round(4)}")

    # Compute offset as in grasp_hawor.py _compute_neutral_offsets:
    # offset = obj_center - (mano_pos[F50] + R[F50] @ finger_forward)
    # This makes the MANO finger midpoint at F50 align with the object center
    F50_local = 50 - start_frame  # F50 is at frame 50, local index in our window
    if F50_local < n_mano:
        _finger_forward = np.array([0.03689, 0.0, 0.0])
        mano_f50_R = mano_traj["R"][F50_local]
        finger_mid_mano = mano_traj["pos"][F50_local] + mano_f50_R @ _finger_forward
        mano_offset = obj_pos - finger_mid_mano
    else:
        mano_offset = s2_pos[0] - mano_traj["pos"][0]
    print(f"  _mano_neutral_offset: {mano_offset.round(4)}")

    # ── 6. Apply offset to MANO trajectory ──
    mano_pos_offset = mano_traj["pos"] + mano_offset
    print(f"  MANO+offset pos range: X=[{mano_pos_offset[:,0].min():.4f},{mano_pos_offset[:,0].max():.4f}]")

    # ── 7. Create 3-panel figure ──
    print("\n[7] Creating visualization...")
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Stage 2 Trajectory: F44~F56 MANO->F50->MANO Transition", fontsize=14, fontweight="bold")

    # Panel 1: 3D trajectory
    ax1 = fig.add_subplot(2, 3, (1, 2), projection="3d")

    # Stage 2 trajectory
    ax1.plot(s2_pos[:, 0], s2_pos[:, 1], s2_pos[:, 2],
             "r-o", markersize=5, linewidth=1.5, alpha=0.8, label="Stage 2 trajectory", zorder=3)
    # Mark F50 (index 6 = frame 50)
    f50_idx = 50 - s2_frames[0]
    if 0 <= f50_idx < len(s2_pos):
        ax1.scatter(s2_pos[f50_idx, 0], s2_pos[f50_idx, 1], s2_pos[f50_idx, 2],
                    c="red", s=150, marker="*", zorder=6, label=f"F50 (Stage 2)")
        ax1.text(s2_pos[f50_idx, 0], s2_pos[f50_idx, 1], s2_pos[f50_idx, 2],
                 "  F50", color="red", fontsize=10, fontweight="bold")

    # MANO+offset trajectory
    ax1.plot(mano_pos_offset[:, 0], mano_pos_offset[:, 1], mano_pos_offset[:, 2],
             "b-s", markersize=4, linewidth=1.0, alpha=0.6, label="MANO+offset")
    if 0 <= F50_local < n_mano:
        ax1.scatter(mano_pos_offset[F50_local, 0], mano_pos_offset[F50_local, 1], mano_pos_offset[F50_local, 2],
                    c="blue", s=120, marker="D", zorder=5, label="F50 (MANO+offset)")

    # Object position
    ax1.scatter(*obj_pos, c="green", s=300, marker="s", edgecolors="k", linewidths=1.5,
                zorder=7, label="Object", alpha=0.7)

    # Best grasp position
    ax1.scatter(*grasp_pos, c="orange", s=150, marker="^", edgecolors="k", linewidths=1.0,
                zorder=6, label="Best grasp (Stage 1)", alpha=0.8)

    # Start/End markers
    ax1.scatter(s2_pos[0, 0], s2_pos[0, 1], s2_pos[0, 2],
                c="black", s=80, marker="o", zorder=5)
    ax1.text(s2_pos[0, 0], s2_pos[0, 1], s2_pos[0, 2], "  start", fontsize=9)
    ax1.scatter(s2_pos[-1, 0], s2_pos[-1, 1], s2_pos[-1, 2],
                c="gray", s=80, marker="x", zorder=5)
    ax1.text(s2_pos[-1, 0], s2_pos[-1, 1], s2_pos[-1, 2], "  end", fontsize=9)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory")
    ax1.legend(fontsize=8, loc="upper left")

    # Equal aspect
    all_pts = np.vstack([s2_pos, mano_pos_offset, obj_pos[None], grasp_pos[None]])
    mid = all_pts.mean(axis=0)
    max_range = max(all_pts.ptp(axis=0)) / 2 * 1.3
    ax1.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax1.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax1.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Panel 2: Position X/Y/Z vs frame number
    ax2_x = fig.add_subplot(2, 3, 4)
    ax2_y = fig.add_subplot(2, 3, 5)
    ax2_z = fig.add_subplot(2, 3, 6)

    frame_labels = s2_frames
    mano_frame_labels = np.arange(start_frame, start_frame + n_mano)

    for ax_comp, comp_idx, comp_name in [(ax2_x, 0, "X"), (ax2_y, 1, "Y"), (ax2_z, 2, "Z")]:
        ax_comp.plot(frame_labels, s2_pos[:, comp_idx], "r-o", markersize=4, linewidth=1.5,
                     label="Stage 2", alpha=0.8)
        ax_comp.plot(mano_frame_labels, mano_pos_offset[:, comp_idx], "b-s", markersize=3,
                     linewidth=1.0, label="MANO+offset", alpha=0.6)
        # F50 marker
        if 0 <= f50_idx < len(s2_pos):
            ax_comp.axvline(50, color="red", linestyle="--", alpha=0.5, label="F50")
        ax_comp.set_xlabel("Frame")
        ax_comp.set_ylabel(f"{comp_name} (m)")
        ax_comp.set_title(f"Position {comp_name}")
        ax_comp.legend(fontsize=7)
        ax_comp.grid(True, alpha=0.3)

    # Panel 3: Gripper qpos vs frame number
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(frame_labels, s2_qpos, "r-o", markersize=5, linewidth=1.5, label="Stage 2 qpos")
    ax3.axhline(grasp_qpos, color="orange", linestyle="--", alpha=0.5,
                label=f"Locked qpos={grasp_qpos:.4f}")

    # Annotate phases
    # Find the frame where qpos reaches locked value
    locked_frames = np.where(np.abs(s2_qpos - grasp_qpos) < 1e-5)[0]
    if len(locked_frames) > 0:
        transition_frame = frame_labels[locked_frames[0]]
        ax3.axvline(transition_frame, color="purple", linestyle=":", alpha=0.7,
                    label=f"Locked at F{transition_frame}")
        # Phase annotations
        ax3.text((frame_labels[0] + transition_frame) / 2, ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 0.005,
                 "Approach+Close", fontsize=10, ha="center", color="purple", fontweight="bold")
        ax3.text((transition_frame + frame_labels[-1]) / 2, ax3.get_ylim()[1] * 0.9 if ax3.get_ylim()[1] > 0 else 0.005,
                 "Hold Grasp", fontsize=10, ha="center", color="darkred", fontweight="bold")
    else:
        ax3.text(frame_labels[len(frame_labels)//2], max(s2_qpos) * 0.9,
                 "Approach+Close", fontsize=10, ha="center", color="purple", fontweight="bold")

    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Gripper qpos")
    ax3.set_title("Gripper Qpos (Approach+Close + Hold Grasp)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    save_path = OUTPUT_DIR / "stage2" / "stage2_trajectory.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {save_path}")

    plt.close(fig)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
