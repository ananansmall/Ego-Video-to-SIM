#!/usr/bin/env python3
"""ee17_fpv_extract.py — 从四宫格仿真 GIF 提取机器人机载 FPV (左下 ego 象限) → MP4。

cb_large_range_sim 每帧渲染 [[正面, 侧面], [ego, 跟随]] 四宫格; ego 相机 =
机器人机载第一视角 (躯干顶端前方 0.30m, 视线锁手部目标, lrt.fpv_cam_pose)。
**ego 在左下** (quad[1][0]) —— 旧 ee17_fpv.py 裁右下(跟随)是错的, 本脚本修正。

用法:
    python test/ee17_fpv_extract.py <四宫格GIF路径> <输出MP4路径> [--fps 30]
    python test/ee17_fpv_extract.py <GIF> <OUT> --windows "grasp,25,80;ring,470,600"
"""
import argparse
import os

import imageio.v2 as imageio
import numpy as np


def ego_frames(gif_path):
    r = imageio.get_reader(gif_path)
    first = r.get_data(0)
    H, W = first.shape[:2]
    # 四宫格: 上半=正面|侧面, 下半=ego|跟随 → ego = 左下 H/2, 0..W/2
    hh, ww = H // 2, W // 2
    out = []
    for img in r:
        if img.shape[0] != H or img.shape[1] != W:
            img = np.asarray(imageio.core.util.asarray(img))[:H, :W]
        out.append(np.ascontiguousarray(img[hh:, :ww]))   # 左下 = ego
    r.close()
    return out, (hh, ww)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gif")
    ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--windows", default=None,
                    help="'name,a,b;name2,c,d' 追加窗口 MP4")
    a = ap.parse_args()
    egos, (hh, ww) = ego_frames(a.gif)
    print(f"FPV ego 象限: {len(egos)} 帧 ({hh}x{ww}) @30fps")
    w = imageio.get_writer(a.out, fps=a.fps, quality=a.quality)
    for f in egos:
        w.append_data(f)
    w.close()
    print(f"✓ {a.out}")
    if a.windows:
        outdir = os.path.dirname(a.out)
        base = os.path.splitext(os.path.basename(a.out))[0]
        for spec in a.windows.split(";"):
            nm, p0, p1 = spec.split(",")[0], int(spec.split(",")[1]), int(spec.split(",")[2])
            p = os.path.join(outdir, f"{base}_{nm}.mp4")
            w = imageio.get_writer(p, fps=a.fps, quality=a.quality)
            for f in egos[p0:p1]:
                w.append_data(f)
            w.close()
            print(f"✓ {p} (帧 {p0}-{p1})")


if __name__ == "__main__":
    main()
