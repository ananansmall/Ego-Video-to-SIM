#!/usr/bin/env python3
"""render_mano_minimal.py — 极简渲染: 只画手腕 + 两指尖(拇指尖/食指尖)

标注:
  j0  手腕
  j4  拇指尖
  j8  食指尖
  j4—j8 连线 = 夹爪开合距离
输出 PNG 到 gripper_test_output/mano_minimal.png
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
sys.path.insert(0, str(SCRIPT_DIR.parent / "position_retargeting"))

from hand_track.common import load_hawor_data, compute_mano_joints, RXWORLD_TO_SAPIEN
from mano_layer import MANOLayer


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor-dir", type=str, default="/home/an/data/hawor/121_C5_CellPhone_161deg")
    ap.add_argument("--hand-idx", type=int, default=1)
    ap.add_argument("--frame", type=int, default=-1, help="指定帧; -1=开合最大帧")
    ap.add_argument("--out", type=str, default=str(SCRIPT_DIR / "gripper_test_output" / "mano_minimal.png"))
    args = ap.parse_args()

    hd = load_hawor_data(args.hawor_dir, hand_idx=args.hand_idx)
    valid = np.where(hd["pred_valid"])[0]
    first = valid[0]
    mano_layer = MANOLayer("right", hd["pred_betas"][first].astype(np.float32))

    sel = args.frame
    if sel < 0:
        best, best_d, = None, -1
        for i in valid:
            rot, trans, hp = hd["pred_rot"][i], hd["pred_trans"][i], hd["pred_hand_pose"][i]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hp)):
                continue
            _, j = compute_mano_joints(mano_layer, rot, hp, trans)
            j = (RXWORLD_TO_SAPIEN @ j.T).T
            d = np.linalg.norm(j[8] - j[4])
            if d > best_d:
                best_d, best = d, i
        sel = best
    print(f"渲染帧: {sel}  开合={best_d*1000:.1f}mm")

    rot, trans, hp = hd["pred_rot"][sel], hd["pred_trans"][sel], hd["pred_hand_pose"][sel]
    _, j = compute_mano_joints(mano_layer, rot, hp, trans)
    j = (RXWORLD_TO_SAPIEN @ j.T).T

    wrist, thumb, index = j[0], j[4], j[8]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    # 仅 3 个点 (同色, 不区分)
    pts = np.array([wrist, thumb, index])
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="tab:blue", s=130,
               edgecolors="black", zorder=5)

    # 两指尖连线 = 夹爪开合
    ax.plot([thumb[0], index[0]], [thumb[1], index[1]], [thumb[2], index[2]],
            color="red", lw=3, zorder=4)
    ax.text((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2,
            (thumb[2] + index[2]) / 2, f"开合 {best_d*1000:.0f}mm",
            color="red", fontsize=10)

    labels = ["j0 手腕", "j4 拇指尖", "j8 食指尖"]
    for p, lab in zip(pts, labels):
        ax.text(p[0], p[1], p[2], f" {lab}", fontsize=10)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("极简: 手腕 + 两指尖 (拇指尖↔食指尖=夹爪开合)")
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()
