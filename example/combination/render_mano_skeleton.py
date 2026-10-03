#!/usr/bin/env python3
"""render_mano_skeleton.py — 用真实 HaWoR 数据渲染 MANO 21 关节骨架

标出夹爪解算用到的 6 个关节 (j3..j8) 和锚点, 帮助理解 test9 的映射.
输出 PNG 到 gripper_test_output/mano_skeleton.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))
POS_DIR = SCRIPT_DIR.parent / "position_retargeting"
sys.path.insert(0, str(POS_DIR))

from hand_track.common import load_hawor_data, compute_mano_joints
from mano_layer import MANOLayer

# MANO 21 关节标准运动学链 (parent -> child 骨头)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
]
FINGER_COLORS = {
    "wrist": "gray", "thumb": "green", "index": "blue",
    "middle": "orange", "ring": "purple", "pinky": "brown",
}
JOINT_FINGER = (
    ["wrist"] + ["thumb"] * 4 + ["index"] * 4 + ["middle"] * 4
    + ["ring"] * 4 + ["pinky"] * 4
)

# 夹爪解算用到的关节
SVD_PTS = [3, 5, 6, 7, 8]      # 手掌平面 SVD
OPEN_PTS = [4, 8]              # 开合指尖距 j4-j8
ANCHOR_PTS = [3, 5]            # 锚点 (j3+j5)/2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor-dir", type=str, default="/home/an/data/hawor/121_C5_CellPhone_161deg")
    ap.add_argument("--hand-idx", type=int, default=1)
    ap.add_argument("--frame", type=int, default=-1, help="指定帧; -1=选开合最大的帧")
    ap.add_argument("--out", type=str, default=str(SCRIPT_DIR / "gripper_test_output" / "mano_skeleton.png"))
    args = ap.parse_args()

    hd = load_hawor_data(args.hawor_dir, hand_idx=args.hand_idx)
    valid = np.where(hd["pred_valid"])[0]
    first = valid[0]
    mano_layer = MANOLayer("right", hd["pred_betas"][first].astype(np.float32))

    # 选帧: 默认选开合最大 (手指最张开) 的帧, 姿态最清楚
    sel = args.frame
    if sel < 0:
        best, best_d = None, -1
        for i in valid:
            rot, trans, hp = hd["pred_rot"][i], hd["pred_trans"][i], hd["pred_hand_pose"][i]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hp)):
                continue
            _, j = compute_mano_joints(mano_layer, rot, hp, trans)
            j = _render_to_sapien(j)
            d = np.linalg.norm(j[8] - j[4])
            if d > best_d:
                best_d, best = d, i
        sel = best
    print(f"渲染帧: {sel}  (开合指尖距={best_d*1000:.1f}mm)")

    rot, trans, hp = hd["pred_rot"][sel], hd["pred_trans"][sel], hd["pred_hand_pose"][sel]
    _, j_raw = compute_mano_joints(mano_layer, rot, hp, trans)
    j = _render_to_sapien(j_raw)   # SAPIEN 世界坐标

    anchor = (j[3] + j[5]) / 2.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # 画骨头
    for a, b in BONES:
        col = FINGER_COLORS[JOINT_FINGER[a]]
        ax.plot([j[a, 0], j[b, 0]], [j[a, 1], j[b, 1]], [j[a, 2], j[b, 2]],
                color=col, lw=2.5, alpha=0.85)

    # 普通关节点
    ax.scatter(j[:, 0], j[:, 1], j[:, 2], c="lightgray", s=20, zorder=3)

    # 高亮 SVD 5 点 (手掌平面)
    svd_colors = {3: "green", 5: "blue", 6: "cyan", 7: "magenta", 8: "red"}
    labels = {3: "j3 拇指IP", 5: "j5 食指MCP", 6: "j6 食指PIP",
              7: "j7 食指DIP", 8: "j8 食指尖"}
    for idx in SVD_PTS:
        ax.scatter(j[idx, 0], j[idx, 1], j[idx, 2], c=svd_colors[idx], s=110,
                   edgecolors="black", zorder=5)
        ax.text(j[idx, 0], j[idx, 1], j[idx, 2], f" {labels[idx]}", fontsize=9)

    # j4 拇指尖 (开合用)
    ax.scatter(j[4, 0], j[4, 1], j[4, 2], c="lime", s=110, edgecolors="black", zorder=5)
    ax.text(j[4, 0], j[4, 1], j[4, 2], " j4 拇指尖", fontsize=9)

    # 开合连线 j4-j8
    ax.plot([j[4, 0], j[8, 0]], [j[4, 1], j[8, 1]], [j[4, 2], j[8, 2]],
            color="red", ls="--", lw=2)
    ax.text((j[4, 0] + j[8, 0]) / 2, (j[4, 1] + j[8, 1]) / 2, (j[4, 2] + j[8, 2]) / 2,
            f"开合={best_d*1000:.0f}mm", color="red", fontsize=9)

    # 锚点 (j3+j5)/2
    ax.scatter(anchor[0], anchor[1], anchor[2], c="yellow", s=140,
               marker="*", edgecolors="black", zorder=6)
    ax.text(anchor[0], anchor[1], anchor[2], " 锚点(j3+j5)/2", fontsize=9, color="goldenrod")

    # 手腕
    ax.scatter(j[0, 0], j[0, 1], j[0, 2], c="black", s=60, zorder=5)
    ax.text(j[0, 0], j[0, 1], j[0, 2], " j0 手腕", fontsize=9)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"MANO 21 关节骨架 (HaWoR 世界, 米)\\n"
                 f"SVD掌心平面={SVD_PTS}  开合=j4-j8  锚点=(j3+j5)/2")
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"已保存: {args.out}")

    # 同时打印坐标, 方便对照
    print("\n关节坐标 (SAPIEN 世界, m):")
    names = ["手腕", "拇1", "拇2", "拇IP", "拇尖", "食MCP", "食PIP", "食DIP", "食尖",
             "中MCP", "中PIP", "中DIP", "中尖", "无名MCP", "无名PIP", "无名DIP", "无名尖",
             "小MCP", "小PIP", "小DIP", "小尖"]
    for i in range(21):
        mark = " <-- SVD" if i in SVD_PTS else (" <-- 开合" if i in OPEN_PTS else "")
        print(f"  j{i:2d} {names[i]:5s} ({j[i,0]:+.3f},{j[i,1]:+.3f},{j[i,2]:+.3f}){mark}")


def _render_to_sapien(pts):
    from hand_track.common import RXWORLD_TO_SAPIEN
    return (RXWORLD_TO_SAPIEN @ pts.T).T


if __name__ == "__main__":
    main()
