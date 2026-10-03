#!/usr/bin/env python3
"""diagnose_mano_scale.py — 诊断 MANO→夹爪映射的尺度/位移/开合问题

不启动 SAPIEN 物理, 只跑 数据加载→MANO FK→夹爪位姿解算,
打印关键统计, 判断问题出在"数据本身移动小"还是"映射丢失尺度".
"""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))
# MANOLayer 在 position_retargeting 目录
POS_DIR = SCRIPT_DIR.parent / "position_retargeting"
sys.path.insert(0, str(POS_DIR))

from hand_track.common import load_hawor_data, compute_mano_joints, _render_to_sapien
from hand_track.gripper_config import compute_mano_based_gripper_pose, FINGER_BASE_DIST
from mano_layer import MANOLayer


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor-dir", type=str, default="/home/an/data/hawor/7")
    ap.add_argument("--hand-idx", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    print("=" * 70)
    print("MANO→夹爪 尺度诊断")
    print(f"  HaWoR目录: {args.hawor_dir}")
    print(f"  hand_idx={args.hand_idx}  scale={args.scale}")
    print("=" * 70)

    hd = load_hawor_data(args.hawor_dir, hand_idx=args.hand_idx)
    valid = hd["pred_valid"]
    valid_ids = np.where(valid)[0]
    n = len(valid_ids)
    print(f"  总帧数={len(valid)}  有效帧={n}")
    if n == 0:
        print("  无有效帧, 退出")
        return

    first = valid_ids[0]
    betas = hd["pred_betas"][first].astype(np.float32)
    mano_layer = MANOLayer("right", betas)
    print(f"  betas from frame {first}")

    trans_list, anchor_list, wrist_list, finger_dist_list, joint_list = [], [], [], [], []
    prev_trans = None
    trans_deltas = []
    anchor_deltas = []

    for i in valid_ids:
        rot = hd["pred_rot"][i]
        trans = hd["pred_trans"][i]
        hp = hd["pred_hand_pose"][i]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hp)):
            continue
        _, j = compute_mano_joints(mano_layer, rot, hp, trans)
        js = _render_to_sapien(j)  # (21,3) SAPIEN 坐标

        anchor = (js[3] + js[5]) / 2.0          # 拇指PIP + 食指MCP 中点
        wrist = js[0]
        base, R, joint, _ = compute_mano_based_gripper_pose(js, prefix="right", scale=args.scale)
        finger_dist = float(np.linalg.norm(js[8] - js[4]))

        trans_list.append(trans.copy())
        anchor_list.append(anchor.copy())
        wrist_list.append(wrist.copy())
        finger_dist_list.append(finger_dist)
        joint_list.append(joint)

        if prev_trans is not None:
            trans_deltas.append(np.linalg.norm(trans - prev_trans))
            anchor_deltas.append(np.linalg.norm(anchor - anchor_list[-2]))
        prev_trans = trans.copy()

    trans_list = np.array(trans_list)
    anchor_list = np.array(anchor_list)
    wrist_list = np.array(wrist_list)
    finger_dist_list = np.array(finger_dist_list)
    joint_list = np.array(joint_list)
    trans_deltas = np.array(trans_deltas)
    anchor_deltas = np.array(anchor_deltas)

    def stats(name, arr, unit=""):
        print(f"  {name:28s} min={arr.min():.4f}  mean={arr.mean():.4f}  "
              f"max={arr.max():.4f}  std={arr.std():.4f}  {unit}")

    print("\n── 绝对位置 (SAPIEN 坐标, 米) ──")
    stats("trans (手腕全局平移)", trans_list)
    stats("anchor=(j3+j5)/2", anchor_list)
    stats("wrist=j[0]", wrist_list)

    print("\n── 逐帧位移幅度 (米) ──")
    stats("Δtrans (MANO全局平移变化)", trans_deltas)
    stats("Δanchor (夹爪root目标变化)", anchor_deltas)

    print("\n── 开合相关 ──")
    stats("finger_dist ‖j8-j4‖", finger_dist_list)
    print(f"  FINGER_BASE_DIST = {FINGER_BASE_DIST:.6f} m")
    stats("joint 映射值", joint_list)
    print(f"  joint 占比 FINGER_BASE_DIST 参考: (finger_dist-FINGER_BASE_DIST)/2 范围 = "
          f"[{(finger_dist_list.min()-FINGER_BASE_DIST)/2:.4f}, "
          f"{(finger_dist_list.max()-FINGER_BASE_DIST)/2:.4f}]")

    print("\n── 结论提示 ──")
    total_trans = trans_list.max(axis=0) - trans_list.min(axis=0)
    total_anchor = anchor_list.max(axis=0) - anchor_list.min(axis=0)
    print(f"  trans 总位移范围 (x,y,z) = {total_trans}")
    print(f"  anchor 总位移范围 (x,y,z) = {total_anchor}")
    print(f"  Δanchor 累计 = {anchor_deltas.sum():.4f} m  (≈ {anchor_deltas.sum()*1000:.1f} mm)")
    print(f"  Δtrans 累计 = {trans_deltas.sum():.4f} m  (≈ {trans_deltas.sum()*1000:.1f} mm)")


if __name__ == "__main__":
    main()
