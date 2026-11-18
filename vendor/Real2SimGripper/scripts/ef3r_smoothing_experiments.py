#!/usr/bin/env python3
"""ef3r_smoothing_experiments.py — EF3R 链精度提升三方向对比实验。

基线: ef3r456 stride=1 (IK mean 12.39mm / ori 0.12° / <5°帧 115/300, 全部 0 跳变越限)。
五个变体, 全部 stride=1 + phys_only + auto WBC 参数, 只改一个因素:

  E1a/E1b  目标轨迹零相位盒平滑 (pos + 四元数, 窗 5 / 9, 2 遍)
  E2a/E2b  WBC 姿态权重降为 2.5 / 1.0 (等效提升位置优先级; 原始残差里
           1° 姿态 ≈ 2.5mm 位置等效, 降 w_ori 让位置让步换姿态)
  E3       savgol (窗9 二阶) 上采样平滑 — 比盒平均更保峰, 模拟更优的重采样

输出: 每个变体的指标 JSON (results/large_range/<name>_large_range.json) + 对比表。
"""
import functools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

BASE_NPZ = "/path/to/project_data/mano_ws/EgoFound3R/outputs/pickplace_456483158/ef3r_right_targets.npz"
RUNS = []


def smooth_quat(R, win, passes=2, mode="box"):
    """零相位平滑旋转矩阵: R→四元数(符号对齐)→逐分量平滑→归一化→R。"""
    from scipy.spatial.transform import Rotation as Rot
    from scipy.signal import savgol_filter
    q = Rot.from_matrix(R).as_quat()                       # (N,4) xyzw
    for k in range(1, len(q)):
        if q[k] @ q[k - 1] < 0:
            q[k] = -q[k]
    if mode == "box":
        y = q.copy()
        pad = win // 2
        for _ in range(passes):
            yp = np.pad(y, ((pad, pad), (0, 0)), mode="reflect")
            y = np.stack([np.convolve(yp[:, c], np.ones(win) / win, mode="valid")
                          for c in range(4)], axis=1)
    else:                                                   # savgol
        y = savgol_filter(q, win, 2, axis=0)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return Rot.from_quat(y).as_matrix()


def smooth_pos(pos, win, passes=2, mode="box"):
    from scipy.signal import savgol_filter
    if mode == "box":
        y = pos.copy()
        pad = win // 2
        for _ in range(passes):
            yp = np.pad(y, ((pad, pad), (0, 0)), mode="reflect")
            y = np.stack([np.convolve(yp[:, c], np.ones(win) / win, mode="valid")
                          for c in range(3)], axis=1)
    else:
        y = savgol_filter(pos, win, 2, axis=0)
    return y


def make_variant(name, win=None, mode="box", w_ori=None):
    """生成变体 npz (如需) 并跑一次, 收集指标。"""
    import cb_large_range_sim as cb
    import large_range_track as lrt

    targets = BASE_NPZ
    if win is not None:
        d = dict(np.load(BASE_NPZ))
        d["pos"] = smooth_pos(d["pos"], win, mode=mode)
        d["rot"] = smooth_quat(d["rot"], win, mode=mode)
        targets = f"/tmp/ef3r_{name}.npz"
        np.savez(targets, **d)

    patch = None
    if w_ori is not None:
        patch = lrt.solve_wbc_track
        lrt.solve_wbc_track = functools.partial(patch, w_ori=w_ori)
    try:
        cb.main(targets=targets, name=name, stride=1, phys_only=True)
    finally:
        if patch is not None:
            lrt.solve_wbc_track = patch
    m = json.load(open(f"/path/to/Real2SimGripper/results/large_range/{name}_large_range.json"))
    RUNS.append((name, m))


def main():
    make_variant("E1a_win5", win=5)
    make_variant("E1b_win9", win=9)
    make_variant("E2a_wori2p5", w_ori=2.5)
    make_variant("E2b_wori1p0", w_ori=1.0)
    make_variant("E3_savgol9", win=9, mode="savgol")

    print("\n================ 对比 (基线 ef3r456: IK 12.39/25.25/55.87mm, ori 0.12°, <5°:115, 滞后24%) ================")
    hdr = "%-14s %8s %8s %8s | %7s %6s | %6s %6s | %8s %6s"
    print(hdr % ("变体", "ik_mean", "ik_p95", "ik_max", "ori_mean", "ori_max", "<5°帧", "跳变", "phys_mean", "phys_max"))
    for name, m in RUNS:
        p = m.get("phys_pos_err_mm", {}) or {}
        print(hdr % (name,
                     "%.2f" % m["ik_pos_err_mm"]["mean"], "%.2f" % m["ik_pos_err_mm"]["p95"],
                     "%.2f" % m["ik_pos_err_mm"]["max"],
                     "%.3f" % m["ik_ori_err_deg"]["mean"], "%.2f" % m["ik_ori_err_deg"]["max"],
                     m.get("tight_frames_lt5deg", "?"), m.get("jump_frames", "?"),
                     "%.1f" % p.get("mean", float("nan")) if p else "?",
                     "%.1f" % p.get("max", float("nan")) if p else "?"))


if __name__ == "__main__":
    main()
