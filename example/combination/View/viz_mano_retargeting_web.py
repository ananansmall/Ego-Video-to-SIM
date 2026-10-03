#!/usr/bin/env python3
"""
Interactive 3D visualization of 3 pose orientations using plotly.
Open viz_mano_retargeting.html in a browser to rotate/zoom interactively.

Usage:
    python viz_mano_retargeting_web.py ~/data/hawor/7 --frame 50
Output:
    /home/an/robot_world_ws/src/dex-retargeting/example/combination/View/viz_mano_retargeting.html
"""

import sys, json, os
from pathlib import Path
from datetime import datetime
import numpy as np

COMBO_DIR = "/home/an/robot_world_ws/src/dex-retargeting/example/combination"
sys.path.insert(0, COMBO_DIR)
sys.path.insert(0, COMBO_DIR + "/hand_track")
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting")

from gripper_config import compute_analytical_gripper_pose, init_gripper_retargeting
from utils import load_hawor_data, matrix_from_compact_axis_angle, compute_mano_joints
from utils import MANO_SKELETON, FINGER_TIP


def compute_mano_based_gripper_pose(j, prefix="right",
                                      finger_origin_x=0.03689,
                                      finger1_origin_x=None, finger2_origin_x=None):
    """从 MANO 5 个关节点解算夹爪位姿 (SVD 拟合 + Gram-Schmidt).

    1. 锚点 anchor = (j3_thumb_pip + j5_index_mcp) / 2
    2. 5 点 (j3,j5,j6,j7,j8) SVD 拟合手掌平面 → Z 轴 (法向，手背侧)
    3. X 轴候选 = j5→j6 (MCP→PIP, 食指延伸方向)
    4. X = 对 Z 做 Gram-Schmidt 正交化
    5. Y = Z × X, 重新 Z = X × Y → 三轴严格正交
    6. gripper base = anchor - offset * X
    """
    # MANO 关节索引 (右手法)
    thumb_pip = j[3]          # 拇指 PIP (近端指间关节)
    index_mcp = j[5]          # 食指 MCP (掌指关节)
    index_pip = j[6]          # 食指 PIP (近端指间)
    index_dip = j[7]          # 食指 DIP (远端指间)
    index_tip = j[8]          # 食指尖

    # ── Step 1: 锚点 ──
    anchor = (thumb_pip + index_mcp) / 2

    # ── Step 2: SVD 拟合手掌平面 → Z ──
    palm_pts = np.array([thumb_pip, index_mcp, index_pip, index_dip, index_tip])
    palm_center = palm_pts.mean(axis=0)
    centered = palm_pts - palm_center
    U, S, Vt = np.linalg.svd(centered, full_matrices=True)
    z_axis = Vt[2]            # 最小奇异值方向 = 平面法向
    z_axis = z_axis / np.linalg.norm(z_axis)

    # ── Step 3: X 轴候选 (MCP→PIP) ──
    x_raw = index_pip - index_mcp
    if np.linalg.norm(x_raw) < 1e-8:
        x_raw = (index_tip - index_mcp)
    x_raw = x_raw / np.linalg.norm(x_raw)

    # ── Step 4-5: Gram-Schmidt 正交化 ──
    x_orth = x_raw - np.dot(x_raw, z_axis) * z_axis
    x_norm = np.linalg.norm(x_orth)
    if x_norm < 1e-8:
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        x_axis = x_axis / np.linalg.norm(x_axis)
    else:
        x_axis = x_orth / x_norm

    y_axis = np.cross(z_axis, x_axis)
    z_axis = np.cross(x_axis, y_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])

    # ── Step 6: gripper base = anchor (j3+j5)/2 ──
    base = anchor

    # ── 手指关节值 ──
    # finger_dist 是总开合量, 每边移动 finger_dist/2
    thumb_tip = j[4]
    finger_dist = np.linalg.norm(index_tip - thumb_tip)
    joint = max(0.0, min(0.05, finger_dist / 2))

    return base, R, joint, anchor


def main(hawor_dir, frame=50):
    print(f"Loading: {hawor_dir}, target frame={frame}")
    data = load_hawor_data(hawor_dir, hand_idx=1, frame_idx=frame)
    fi = data["frame_idx"]
    print(f"Using frame {fi}")

    rot = data["pred_rot"]
    trans = data["pred_trans"]
    hp = data["pred_hand_pose"]
    betas = data["pred_betas"]
    _, j = compute_mano_joints(rot, hp, trans, betas)

    wrist, thumb, index = j[0], j[4], j[8]
    wrist2tip_dir = (thumb - wrist) / np.linalg.norm(thumb - wrist)
    fi_dist = np.linalg.norm(index - thumb)

    # Pose #1: MANO hand (anatomical palm-forward = -pred_rot[:,0] points to fingertips)
    R_mano = matrix_from_compact_axis_angle(rot)
    mano_dir = -R_mano[:, 0]

    # Pose #2: Analytical gripper
    g_pos, g_R, _, _ = compute_analytical_gripper_pose(wrist, thumb, index, "right")
    ana_dir = g_R[:, 0]

    # Pose #3a: Retargeting — 3 targets [j4, j8, j0] (default)
    retargeting, ref_indices, _ = init_gripper_retargeting(
        "right", 0.03689, Path("/home/an/robot_world_ws/src"),
    )
    ref_value = j[ref_indices, :3].astype(np.float32)
    num_fixed = len(retargeting.optimizer.idx_pin2fixed)
    fixed_qpos = np.zeros(num_fixed, dtype=np.float32)
    retarget_qpos_3 = retargeting.retarget(ref_value, fixed_qpos)
    robot = retargeting.optimizer.robot

    def get_gripper_pose(robot, retarget_qpos):
        robot.compute_forward_kinematics(retarget_qpos)
        opt_pos = opt_f1 = opt_f2 = None
        for i, name in enumerate(robot.link_names):
            pos = robot.get_link_pose(i)[:3, 3].copy()
            if name == "right_gripper_link": opt_pos = pos
            if "finger_link1" in name: opt_f1 = pos
            if "finger_link2" in name: opt_f2 = pos
        assert opt_pos is not None
        return opt_pos, opt_f1, opt_f2

    opt_pos_3, opt_f1_3, opt_f2_3 = get_gripper_pose(robot, retarget_qpos_3)
    opt_tip_mid_3 = (opt_f1_3 + opt_f2_3) / 2
    opt_dir_3 = (opt_pos_3 - opt_tip_mid_3) / np.linalg.norm(opt_pos_3 - opt_tip_mid_3)

    # Pose #4: MANO-based gripper (SVD palm plane + Gram-Schmidt)
    m_pos, m_R, m_joint, m_anchor = compute_mano_based_gripper_pose(j, "right")
    m_dir = m_R[:, 0]  # X axis = finger extension direction

    # Default display: 3-target version
    opt_pos, opt_dir = opt_pos_3, opt_dir_3

    print(f"\n  wrist->thumb dir = {wrist2tip_dir}")
    print(f"  #1 MANO X         = {mano_dir}  dot(wrist->tip)={np.dot(mano_dir, wrist2tip_dir):.4f}")
    print(f"  #2 Analytical X   = {ana_dir}  dot(wrist->tip)={np.dot(ana_dir, wrist2tip_dir):.4f}")
    print(f"  #3 Retarget 3pts  = {opt_dir_3}  dot(wrist->tip)={np.dot(opt_dir_3, wrist2tip_dir):.4f}  base={opt_pos_3}")
    print(f"  #4 SVD palm X     = {m_dir}  dot(wrist->tip)={np.dot(m_dir, wrist2tip_dir):.4f}  base={m_pos}")

    AL = 0.050

    scenes = []

    # MANO joints
    joint_markers = []
    for i in range(21):
        if i == 0:
            c, sz = '#00FF00', 10
        elif i in FINGER_TIP:
            c, sz = '#FFD700', 7
        else:
            c, sz = '#4682B4', 5
        joint_markers.append({
            "type": "scatter3d",
            "mode": "markers",
            "x": [float(j[i,0])], "y": [float(j[i,1])], "z": [float(j[i,2])],
            "marker": {"size": sz, "color": c, "symbol": "circle", "opacity": 0.9},
            "name": "MANO-joints",
            "showlegend": i == 0,
        })
    scenes.extend(joint_markers)

    # ── Joint number labels ──
    for i in range(21):
        scenes.append({
            "type": "scatter3d",
            "mode": "text",
            "x": [float(j[i,0])], "y": [float(j[i,1])], "z": [float(j[i,2]) + 0.008],
            "text": [f"j{i}"],
            "textfont": {"size": 10, "color": "black"},
            "name": "joint-labels",
            "showlegend": False,
        })

    # ── #4 SVD fitting 5 points: j3 thumb PIP, j5/j6/j7/j8 index MCP/PIP/DIP/tip ──
    svd_idx = [3, 5, 6, 7, 8]
    svd_labels = ["j3-thumbPIP", "j5-indexMCP", "j6-indexPIP", "j7-indexDIP", "j8-indexTip"]
    for idx, lbl in zip(svd_idx, svd_labels):
        scenes.append({
            "type": "scatter3d", "mode": "markers",
            "x": [float(j[idx,0])], "y": [float(j[idx,1])], "z": [float(j[idx,2])],
            "marker": {"size": 9, "color": "cyan", "symbol": "diamond", "opacity": 0.85},
            "name": "j3/j5/j6/j7/j8 (SVD-fitting)",
            "showlegend": idx == 3,
        })

    # ── #4 anchor (j3+j5)/2 is the gripper base itself, shown as the base marker ──

    # MANO skeleton lines
    for p, c in MANO_SKELETON:
        scenes.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": [float(j[p,0]), float(j[c,0])],
            "y": [float(j[p,1]), float(j[c,1])],
            "z": [float(j[p,2]), float(j[c,2])],
            "line": {"color": "gray", "width": 4},
            "name": "MANO-skeleton", "showlegend": False,
            "opacity": 0.5,
        })

    # ── Origin markers ──
    scenes.append({
        "type": "scatter3d", "mode": "markers",
        "x": [float(wrist[0])], "y": [float(wrist[1])], "z": [float(wrist[2])],
        "marker": {"size": 10, "color": "red", "symbol": "cross"},
        "name": "#1 MANO wrist (j0)", "showlegend": True,
    })
    scenes.append({
        "type": "scatter3d", "mode": "markers",
        "x": [float(g_pos[0])], "y": [float(g_pos[1])], "z": [float(g_pos[2])],
        "marker": {"size": 10, "color": "#FF00FF", "symbol": "diamond"},
        "name": "#2 Analytical base", "showlegend": True,
    })
    scenes.append({
        "type": "scatter3d", "mode": "markers",
        "x": [float(opt_pos[0])], "y": [float(opt_pos[1])], "z": [float(opt_pos[2])],
        "marker": {"size": 10, "color": "orange", "symbol": "square"},
        "name": "#3 Retarget [j4,j8,j0] base", "showlegend": True,
    })
    scenes.append({
        "type": "scatter3d", "mode": "markers",
        "x": [float(m_pos[0])], "y": [float(m_pos[1])], "z": [float(m_pos[2])],
        "marker": {"size": 10, "color": "cyan", "symbol": "square"},
        "name": "#4 SVD palm base (anchor)", "showlegend": True,
    })

    # ── Direction arrows ──
    def make_arrow(x0, y0, z0, dx, dy, dz, color, name, width=8):
        scenes.append({
            "type": "scatter3d",
            "mode": "lines",
            "x": [float(x0), float(x0 + dx*AL)],
            "y": [float(y0), float(y0 + dy*AL)],
            "z": [float(z0), float(z0 + dz*AL)],
            "line": {"color": color, "width": width},
            "name": name, "showlegend": True,
        })

    make_arrow(wrist[0], wrist[1], wrist[2],
               mano_dir[0], mano_dir[1], mano_dir[2],
               "red", "#1 MANO (pred_rot)")
    make_arrow(g_pos[0], g_pos[1], g_pos[2],
               ana_dir[0], ana_dir[1], ana_dir[2],
               "#FF00FF", "#2 Analytical (Gram-Schmidt)")
    make_arrow(opt_pos[0], opt_pos[1], opt_pos[2],
               opt_dir[0], opt_dir[1], opt_dir[2],
               "orange", "#3 Retarget [j4,j8,j0]")
    make_arrow(m_pos[0], m_pos[1], m_pos[2],
               m_dir[0], m_dir[1], m_dir[2],
               "cyan", "#4 SVD palm X (final)")
    # Show construction: Z axis (plane normal) and raw X (MCP→PIP)
    make_arrow(m_pos[0], m_pos[1], m_pos[2],
               m_R[0,2]*0.6, m_R[1,2]*0.6, m_R[2,2]*0.6,
               "#00FF00", "#4 Z (SVD plane normal)", width=5)
    idx_mcp, idx_pip = j[5], j[6]
    x_raw_dir = (idx_pip - idx_mcp) / np.linalg.norm(idx_pip - idx_mcp)
    make_arrow(m_pos[0], m_pos[1], m_pos[2],
               x_raw_dir[0]*0.6, x_raw_dir[1]*0.6, x_raw_dir[2]*0.6,
               "#AA00FF", "#4 X-raw (j5->j6)", width=5)

    # camera target
    all_pts = np.vstack([j, wrist, g_pos, opt_pos, m_pos])
    center = all_pts.mean(axis=0)

    layout = {
        "title": f"5 Pose Directions  |  Frame {fi}  |  Thumb-Index={fi_dist*1000:.1f}mm",
        "scene": {
            "xaxis": {"title": "X (m)"},
            "yaxis": {"title": "Y (m)"},
            "zaxis": {"title": "Z (m)"},
            "camera": {
                "eye": {"x": 1.5, "y": 1.5, "z": 1.0},
                "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
            },
        },
        "margin": {"l": 0, "r": 0, "t": 30, "b": 0},
        "legend": {"font": {"size": 11}},
    }

    data_json = json.dumps(scenes, ensure_ascii=False)
    layout_json = json.dumps(layout, ensure_ascii=False)

    import plotly.graph_objects as go
    fig = go.Figure(data=scenes, layout=layout)

    plotly_html = fig.to_html(include_plotlyjs='inline', full_html=False,
                              config={'displayModeBar': True, 'showLink': False})
    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>3 Pose Directions - Frame {fi}</title>
<style>body {{ margin:0;padding:0;font-family:sans-serif;overflow:hidden; }}
#graph {{ width:100vw;height:100vh; }}
</style></head><body>
<div id="graph">{plotly_html}</div>
</body></html>"""

    out = str(Path(__file__).parent / "viz_mano_retargeting.html")
    with open(out, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"\nSaved: {out}")
    print("Open in browser: file://" + out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("hawor_dir", help="HaWoR dir")
    parser.add_argument("--frame", type=int, default=50)
    args = parser.parse_args()
    main(args.hawor_dir, frame=args.frame)
