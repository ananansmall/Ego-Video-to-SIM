#!/usr/bin/env python3
"""分析 mp4: 统计 橙色(夹爪若橙色) / 皮肤色(MANO) / 夹爪候选(非皮肤非背景, 灰金属) 像素占比."""
import sys
import numpy as np
import imageio.v2 as imageio

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test9_out2/test9_mano_trajectory.mp4"
reader = imageio.get_reader(path, "ffmpeg")
orange_list, skin_list, grip_list = [], [], []
for frame in reader:
    rgb = frame[..., :3].astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # 背景: 接近白/冷色高亮
    bg = (r > 0.80) & (g > 0.80) & (b > 0.80) & (np.abs(r-g) < 0.08) & (np.abs(g-b) < 0.08)
    # 皮肤 MANO: 暖色 r>g>b
    skin = (r > 0.55) & (g > 0.40) & (b > 0.30) & (r >= g) & (g >= b) & (r - b > 0.06) & (~bg)
    # 橙色夹爪 (sRGB 伽马校正后亮度更高, 阈值需放宽)
    orange = (r > 0.75) & (g > 0.30) & (g < 0.80) & (b < 0.50) & (r - b > 0.35) & (~bg)
    # 夹爪候选: 非背景 且 非皮肤 且 非橙 (即灰金属等)
    grip = (~bg) & (~skin) & (~orange)
    orange_list.append(float(orange.mean() * 100))
    skin_list.append(float(skin.mean() * 100))
    grip_list.append(float(grip.mean() * 100))
reader.close()
orange_list = np.array(orange_list); skin_list = np.array(skin_list); grip_list = np.array(grip_list)
print(f"帧数={len(orange_list)}")
print(f"橙色夹爪: avg={orange_list.mean():.2f}% max={orange_list.max():.2f}%")
print(f"皮肤MANO: avg={skin_list.mean():.2f}% max={skin_list.max():.2f}%")
print(f"夹爪候选(灰金属等, 非皮肤非背景): avg={grip_list.mean():.2f}% max={grip_list.max():.2f}%")
vis = np.where(grip_list > 0.1)[0]
print(f"能看到夹爪候选(>0.1%)的帧: {len(vis)}/{len(grip_list)}")
