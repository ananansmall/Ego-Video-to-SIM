#!/usr/bin/env python3
"""ee17_fpv.py — 从 4 宫格仿真 GIF 提取机器人机载 FPV (ego 象限) → MP4。

cb_large_range_sim 每帧渲染 [正面|侧面 / FPV|跟随] 四宫格; FPV 象限 = 右下
(ego 相机: 躯干顶端前上方 0.30m, 视线锁手部目标 —— lrt.fpv_cam_pose)。
交付物 (用户步骤③): FPV MP4 (全程 + 抓取/持物/入环三窗口) + 改前/改后 FPV
对比动图 (同窗并排)。从已落盘 GIF 裁剪 = 精确复现该 run 的记录, 无重放偏差。

用法 (robotwin_curobo env):
    python test/ee17_fpv.py after    # 全程 MP4 + 三窗口 MP4
    python test/ee17_fpv.py compare  # 改前/改后 FPV 对比动图 (需 before GIF)
"""
import os
import sys

import imageio.v2 as imageio
import numpy as np

ROOT = "/path/to/Real2SimGripper"
GIF = {mode: f"{ROOT}/demo_output/large_range/ee17_posture_agile_{mode}_large_range.gif"
       for mode in ("object", "before")}
OUT = f"{ROOT}/demo_output/large_range"
# 释放窗阶段帧 (targets 600 帧系, sim 1:1): PICK 47 / LIFT 53 / HOLD 106-448 /
# PLACE 448 / RELEASED 534。窗口取阶段核心段。
WINDOWS = {"grasp": (25, 80), "hold": (200, 330), "ring": (470, 600)}


def ego_frames(gif_path):
    r = imageio.get_reader(gif_path)
    out = []
    for img in r:
        out.append(np.ascontiguousarray(img[360:720, 480:960]))   # ego 象限
    r.close()
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "after"
    if mode == "compare":
        fa = ego_frames(GIF["object"])
        fb = ego_frames(GIF["before"])
        n = min(len(fa), len(fb))
        for nm, (a, b) in WINDOWS.items():
            fs = [np.concatenate([fb[k], fa[k]], axis=1) for k in range(a, min(b, n))]
            p = f"{OUT}/ee17_agile_fpv_compare_{nm}.gif"
            imageio.mimsave(p, fs, fps=15, loop=0)
            print(f"✓ {p} ({len(fs)} 帧, 左=改前 右=改后)")
        return
    gif = GIF["object"]
    egos = ego_frames(gif)
    print(f"FPV 象限: {len(egos)} 帧 @30fps")
    p_full = f"{OUT}/ee17_agile_fpv_full.mp4"
    w = imageio.get_writer(p_full, fps=30, quality=8)
    for f in egos:
        w.append_data(f)
    w.close()
    print(f"✓ {p_full}")
    for nm, (a, b) in WINDOWS.items():
        p = f"{OUT}/ee17_agile_fpv_{nm}.mp4"
        w = imageio.get_writer(p, fps=30, quality=8)
        for f in egos[a:min(b, len(egos))]:
            w.append_data(f)
        w.close()
        print(f"✓ {p} (帧 {a}-{b})")


if __name__ == "__main__":
    main()
