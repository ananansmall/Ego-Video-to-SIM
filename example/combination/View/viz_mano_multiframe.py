#!/usr/bin/env python3
"""
Multi-frame comparison viewer for 4 gripper retargeting methods.
Shows a 3x3 grid of frames with frame number as subplot annotation.
Interactive Plotly: hover for details, drag to rotate each subplot.

Usage:
    python viz_mano_multiframe.py ~/data/hawor/7

Output:
    /home/an/robot_world_ws/src/dex-retargeting/example/combination/View/viz_mano_multiframe.html
"""

import sys, json, os
from pathlib import Path
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

COMBO_DIR = "/home/an/robot_world_ws/src/dex-retargeting/example/combination"
sys.path.insert(0, COMBO_DIR)
sys.path.insert(0, COMBO_DIR + "/hand_track")
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting")

from gripper_config import compute_analytical_gripper_pose, init_gripper_retargeting
from utils import load_hawor_data, matrix_from_compact_axis_angle, compute_mano_joints
from utils import MANO_SKELETON, FINGER_TIP

AL = 0.035
ARROW_WIDTH = 4
SKELETON_WIDTH = 2


def compute_mano_based_gripper_pose(j, finger_origin_x=0.03689):
    """
    1. anchor = (j2_thumb_pip + j5_index_mcp) / 2
    2. 5 pts (j2,j5,j6,j7,j8) SVD → Z (palm normal)
    3. X raw = j5→j6 (MCP→PIP)
    4. X ⊥ Z via Gram-Schmidt; Y = Z×X; Z = X×Y
    5. base = anchor - offset * X
    """
    thumb_pip, index_mcp = j[2], j[5]
    index_pip, index_dip, index_tip = j[6], j[7], j[8]

    anchor = (thumb_pip + index_mcp) / 2

    palm_pts = np.array([thumb_pip, index_mcp, index_pip, index_dip, index_tip])
    centered = palm_pts - palm_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=True)
    z_axis = Vt[2] / np.linalg.norm(Vt[2])

    x_raw = index_pip - index_mcp
    if np.linalg.norm(x_raw) < 1e-8:
        x_raw = index_tip - index_mcp
    x_raw = x_raw / np.linalg.norm(x_raw)

    x_orth = x_raw - np.dot(x_raw, z_axis) * z_axis
    if np.linalg.norm(x_orth) < 1e-8:
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        x_axis = x_axis / np.linalg.norm(x_axis)
    else:
        x_axis = x_orth / np.linalg.norm(x_orth)

    y_axis = np.cross(z_axis, x_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])

    base = anchor - finger_origin_x * x_axis

    thumb_tip = j[4]
    finger_dist = np.linalg.norm(index_tip - thumb_tip)
    joint = max(0.0, min(0.05, (finger_dist - 0.026906) / 2))

    return base, R, joint


def compute_frame_data(hawor_dir, frame_idx):
    """Return dict with all computed data for one frame."""
    data = load_hawor_data(hawor_dir, hand_idx=1, frame_idx=frame_idx)
    fi = data["frame_idx"]

    _, j = compute_mano_joints(data["pred_rot"], data["pred_hand_pose"],
                               data["pred_trans"], data["pred_betas"])

    wrist, thumb, index = j[0], j[4], j[8]
    wrist2tip = (thumb - wrist) / np.linalg.norm(thumb - wrist)

    # #1 MANO hand
    R_mano = matrix_from_compact_axis_angle(data["pred_rot"])
    mano_dir = -R_mano[:, 0]

    # #2 Analytical
    g_pos, g_R, _, _ = compute_analytical_gripper_pose(wrist, thumb, index, "right")
    ana_dir = g_R[:, 0]

    # #3 Retargeting
    retargeting, ref_indices, _ = init_gripper_retargeting(
        "right", 0.03689, Path("/home/an/robot_world_ws/src"),
    )
    ref_value = j[ref_indices, :3].astype(np.float32)
    num_fixed = len(retargeting.optimizer.idx_pin2fixed)
    fixed_qpos = np.zeros(num_fixed, dtype=np.float32)
    retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
    robot = retargeting.optimizer.robot

    opt_pos = opt_f1 = opt_f2 = None
    robot.compute_forward_kinematics(retarget_qpos)
    for i, name in enumerate(robot.link_names):
        pos = robot.get_link_pose(i)[:3, 3].copy()
        if name == "right_gripper_link": opt_pos = pos
        if "finger_link1" in name: opt_f1 = pos
        if "finger_link2" in name: opt_f2 = pos
    opt_tip_mid = (opt_f1 + opt_f2) / 2
    opt_dir = (opt_pos - opt_tip_mid) / np.linalg.norm(opt_pos - opt_tip_mid)

    # #4 SVD palm
    m_pos, m_R, m_joint = compute_mano_based_gripper_pose(j)
    m_dir = m_R[:, 0]

    return {
        "fi": fi, "j": j, "wrist": wrist,
        "wrist2tip": wrist2tip,
        "mano_dir": mano_dir,
        "ana_pos": g_pos, "ana_dir": ana_dir,
        "opt_pos": opt_pos, "opt_dir": opt_dir,
        "m_pos": m_pos, "m_dir": m_dir,
    }


def build_scene(fd, idx_label=""):
    """Build list of plotly traces for one frame."""
    j = fd["j"]; wrist = fd["wrist"]
    sc = []

    # MANO joints
    for i in range(21):
        c = "#00FF00" if i == 0 else ("#FFD700" if i in FINGER_TIP else "#4682B4")
        sc.append({
            "type": "scatter3d", "mode": "markers",
            "x": [float(j[i,0])], "y": [float(j[i,1])], "z": [float(j[i,2])],
            "marker": {"size": 5, "color": c, "symbol": "circle", "opacity": 0.85},
            "name": f"joints {idx_label}", "showlegend": False,
        })

    # MANO skeleton
    for p, c in MANO_SKELETON:
        sc.append({
            "type": "scatter3d", "mode": "lines",
            "x": [float(j[p,0]), float(j[c,0])],
            "y": [float(j[p,1]), float(j[c,1])],
            "z": [float(j[p,2]), float(j[c,2])],
            "line": {"color": "gray", "width": SKELETON_WIDTH},
            "name": "skeleton", "showlegend": False, "opacity": 0.4,
        })

    # #4 SVD fitting points highlight
    for idx in [2, 5, 6, 7, 8]:
        sc.append({
            "type": "scatter3d", "mode": "markers",
            "x": [float(j[idx,0])], "y": [float(j[idx,1])], "z": [float(j[idx,2])],
            "marker": {"size": 6, "color": "cyan", "symbol": "diamond", "opacity": 0.85},
            "name": "SVD-fits", "showlegend": False,
        })

    # Origin markers
    sc.append({
        "type": "scatter3d", "mode": "markers",
        "x": [float(wrist[0])], "y": [float(wrist[1])], "z": [float(wrist[2])],
        "marker": {"size": 8, "color": "red", "symbol": "cross"},
        "name": f"#1 wrist {idx_label}", "showlegend": False,
    })
    for pos, color, sym in [(fd["ana_pos"], "#FF00FF", "diamond"),
                             (fd["opt_pos"], "orange", "square"),
                             (fd["m_pos"], "cyan", "square")]:
        sc.append({
            "type": "scatter3d", "mode": "markers",
            "x": [float(pos[0])], "y": [float(pos[1])], "z": [float(pos[2])],
            "marker": {"size": 7, "color": color, "symbol": sym},
            "showlegend": False,
        })

    # Direction arrows
    def arrow(x0, y0, z0, dx, dy, dz, color, w=ARROW_WIDTH):
        sc.append({
            "type": "scatter3d", "mode": "lines",
            "x": [float(x0), float(x0 + dx*AL)],
            "y": [float(y0), float(y0 + dy*AL)],
            "z": [float(z0), float(z0 + dz*AL)],
            "line": {"color": color, "width": w},
            "showlegend": False,
        })

    arrow(wrist[0], wrist[1], wrist[2],
          fd["mano_dir"][0], fd["mano_dir"][1], fd["mano_dir"][2], "red")
    arrow(fd["ana_pos"][0], fd["ana_pos"][1], fd["ana_pos"][2],
          fd["ana_dir"][0], fd["ana_dir"][1], fd["ana_dir"][2], "#FF00FF")
    arrow(fd["opt_pos"][0], fd["opt_pos"][1], fd["opt_pos"][2],
          fd["opt_dir"][0], fd["opt_dir"][1], fd["opt_dir"][2], "orange")
    anchor_4 = (j[2] + j[5]) / 2
    arrow(anchor_4[0], anchor_4[1], anchor_4[2],
          fd["m_dir"][0], fd["m_dir"][1], fd["m_dir"][2], "cyan")

    # Center for camera
    all_pts = np.vstack([j, wrist, fd["ana_pos"], fd["opt_pos"], fd["m_pos"]])
    center = all_pts.mean(axis=0)

    return sc, center


def make_multiframe_html(hawor_dir, n_frames=9):
    """Generate HTML with n_frames arranged in a 3x3 grid."""
    rec_dir = Path(str(hawor_dir)) / "reconstruction"
    files = sorted(rec_dir.glob("hawor_results_*_depth_aligned.npz"), reverse=True)
    if not files:
        files = sorted(rec_dir.glob("hawor_results_*.npz"), reverse=True)
    rec = np.load(str(files[0]), allow_pickle=True)
    valid = np.where(rec['pred_valid'][1] &
                     ~np.any(np.isnan(rec['pred_hand_pose'][1]), axis=1))[0]
    if len(valid) < n_frames:
        n_frames = len(valid)

    # Pick evenly spaced frames
    step = max(1, len(valid) // n_frames)
    frame_indices = [valid[i] for i in range(0, len(valid), step)][:n_frames]
    print(f"Showing {len(frame_indices)} frames: {frame_indices}")

    # Compute all frames
    frame_data = [compute_frame_data(hawor_dir, int(f)) for f in frame_indices]

    # Build subplot grid (3x3)
    nc = 3
    nr = (len(frame_data) + nc - 1) // nc
    fig = make_subplots(
        rows=nr, cols=nc,
        specs=[[{"type": "scene"}] * nc for _ in range(nr)],
        subplot_titles=[f"Frame {fd['fi']}" for fd in frame_data],
    )

    for i, fd in enumerate(frame_data):
        r = i // nc + 1
        c = i % nc + 1
        sc, center = build_scene(fd, idx_label=f"#{i}")
        for trace in sc:
            fig.add_trace(trace, row=r, col=c)

    # Camera config for all subplots
    for r in range(1, nr+1):
        for c in range(1, nc+1):
            fig.update_xaxes(
                camera_eye=dict(x=1.5, y=1.5, z=1.0),
                camera_center=dict(x=0, y=0, z=0),
                row=r, col=c,
            )

    fig.update_layout(
        title=f"{len(frame_data)}-Frame Comparison  |  4 Methods",
        height=nr * 450,
        width=nc * 520,
        legend=dict(
            orientation="h",
            y=-0.05,
            x=0.5,
            xanchor="center",
            font=dict(size=11),
            itemsizing="constant",
        ),
        margin=dict(l=10, r=10, t=50, b=40),
    )

    # Add legend trace (hidden, used for legend only)
    legend_items = [
        ("red", "#1 MANO (pred_rot)", "red"),
        ("#FF00FF", "#2 Analytical (Gram-Schmidt)", "#FF00FF"),
        ("orange", "#3 Retarget [j4,j8,j0]", "orange"),
        ("cyan", "#4 SVD palm (5-pt fit)", "cyan"),
    ]
    for color, name, c in legend_items:
        fig.add_trace(go.Scatter3d(
            mode="lines",
            x=[0, 0.02], y=[0, 0], z=[0, 0],
            line=dict(color=c, width=6),
            name=name,
            showlegend=True,
        ), row=1, col=1)

    # Hide the extra legend from subplots 1,1 (it's just a legend trace)
    # Actually keep it visible

    out = "/home/an/robot_world_ws/src/dex-retargeting/example/combination/View/viz_mano_multiframe.html"
    fig.write_html(out, include_plotlyjs="bundle")
    print(f"\nSaved: {out}")
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hawor_dir", help="HaWoR dir")
    parser.add_argument("--n", type=int, default=9, help="Number of frames")
    args = parser.parse_args()
    make_multiframe_html(args.hawor_dir, args.n)
