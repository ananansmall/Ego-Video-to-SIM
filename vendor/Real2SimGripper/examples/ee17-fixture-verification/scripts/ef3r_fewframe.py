#!/usr/bin/env python3
"""ef3r_fewframe.py — pick/place 少帧接触校准 (只用 EF3R 自带数据, 替代 AGILE 重优化)。

背景: AGILE 链用 4 模型预处理 + 逐帧优化 (1.5~6h/条) 换"手-物交互正确性"。
实测发现机器人侧真正消费的只有三样: 抓取帧时机、人手实际握宽、物体在手内
偏移 —— 这些 **EF3R 自己就有** (joint_contact_logits 接触概率 + 关节轨迹 +
物体分割点云), 取 pick/place 几帧即可, 秒级。

输出 <ef3r-dir>/pick_place_calib.npz:
  grasp_frame        接触概率最高且稳定的帧 (抓取时机, 比状态段边界更准)
  grasp_prob         该帧接触概率
  human_grip_m       人手该帧拇指-食指尖距离 (人手实际握宽, 米)
  obj_in_palm_ef3r   物体质心相对掌心的偏移 (EF3R 世界系, 米) —— 物体在手内的
                     真实携带位姿
  place_ref_ef3r     RELEASED 段物体质心 (放置参考)
  prob_profile       全序列接触概率 (诊断)

用法 (vggt-omega env, 需 torch):
    python ef3r_fewframe.py --ef3r-dir outputs/pickplace_456483158
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef3r-dir", required=True)
    a = ap.parse_args()
    E = a.ef3r_dir

    import torch

    d = torch.load(os.path.join(E, "inference_output.pt"), map_location="cpu",
                   weights_only=False)
    logits = d["joint_contact_logits"].float().numpy()           # (N,2,21) bfloat16→fp32
    prob = 1.0 / (1.0 + np.exp(-logits))                          # sigmoid
    side = 1                                                      # 右手
    # 指尖+指根 (5,8 指尖; 4,7 近端) 的接触概率均值 —— 抓取时刻指尖先触物
    tip = prob[:, side, [4, 5, 7, 8]].mean(1)
    # 平滑 (11 帧窗) 后取最高且稳定的帧: max(平滑值) —— 抓取是持续接触不是瞬触
    kernel = np.ones(11) / 11.0
    smooth = np.convolve(tip, kernel, mode="same")
    # 抓取帧 = 首次**持续接触起始** (阈值=0.6×峰值, 保持≥10帧): PICK/LIFT/HOLD
    # 全程都接触, 全局 max 会选到 PLACE 释放段 —— 起始才是抓取时机。
    thr = 0.6 * float(smooth.max())
    above = smooth > thr
    grasp_frame, run = None, 0
    for i, v in enumerate(above):
        run = run + 1 if v else 0
        if run >= 10:
            grasp_frame = i - 9          # 起始帧
            break
    if grasp_frame is None:
        grasp_frame = int(np.argmax(smooth))

    J = np.load(os.path.join(E, "hands", "joint_xyz_world.npy"))[:, side]  # (N,21,3)
    human_grip = float(np.linalg.norm(J[grasp_frame, 4] - J[grasp_frame, 8]))

    states = {}
    sl = os.path.join(E, "state_labels.json")
    if os.path.exists(sl):
        states = json.load(open(sl))["states"]

    obj_p = os.path.join(E, "object_instance.npz")
    od = np.load(obj_p) if os.path.exists(obj_p) else None
    obj_c = od["obj_points"].mean(0) if od is not None else None
    palm = J[:, [0, 5, 9, 13, 17]].mean(1)
    obj_in_palm = (obj_c - palm[grasp_frame]) if obj_c is not None else None

    place_ref = None
    if "RELEASED" in states:
        a_r, b_r = states["RELEASED"]
        place_ref = palm[min(a_r + 5, len(palm) - 1)]

    out = os.path.join(E, "pick_place_calib.npz")
    np.savez(out, grasp_frame=grasp_frame, grasp_prob=float(smooth[grasp_frame]),
             human_grip_m=human_grip,
             obj_in_palm_ef3r=obj_in_palm if obj_in_palm is not None else np.zeros(3),
             place_ref_ef3r=place_ref if place_ref is not None else np.zeros(3),
             prob_profile=tip.astype(np.float32))
    print(f"  抓取帧 F{grasp_frame} (接触概率 {smooth[grasp_frame]:.2f}) | "
          f"人手握宽 {human_grip*1000:.1f}mm | 物体-掌心偏移 "
          f"{np.round(obj_in_palm,3).tolist() if obj_in_palm is not None else '无'}")
    if states:
        for nm, (a_s, b_s) in states.items():
            mark = " <-- 抓取帧落在此段" if a_s <= grasp_frame < b_s else ""
            print(f"    {nm:9s} [{a_s},{b_s}){mark}")
    print(f"  ✓ {out}")


if __name__ == "__main__":
    main()
