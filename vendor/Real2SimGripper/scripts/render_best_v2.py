#!/usr/bin/env python3
"""render_best_v2.py —— 用新 IK(16 seeds + 300 iters)渲染确认后的最优配置。

配置来自 scripts/sweep_base_dense.py 的 14 点密扫结论:
    base_xy=[0.30,-0.38] gt=0.0818 为该邻域最优 (0.01 步长内无更优点)

产物:
    demo_output/robotwin_r1_duck_plate_best.gif / .mp4   5 视角 2x2
    demo_output/robotwin_r1_duck_replay.mp4              仿真FPV | 真实视频 逐帧并排
    scripts/agentic_replay_frames/best_*.png             关键帧并排 + 俯视图
"""
import json
import os
import sys

import numpy as np
from PIL import Image
import imageio

sys.path.insert(0, '/path/to/Real2SimGripper/scripts')
import robotwin_r1_duck_grasp as G
from collision_config import enable_coacd
from task_config import load_task, apply_task
from agentic_replay import load_video, make_panels, make_replay_video

cfg = load_task('../configs/duck_to_plate.json')
apply_task(G, cfg, verbose=False)
enable_coacd(G, None, verbose=False)

BEST = {
    "plate_geom": "geometry_8",
    "base_xy": [0.30, -0.38],
    "time_scale": 3.0,
    "place_hover": 0.07,
    "place_t": 0.0,
    "grip_open_q": 0.05,
    "ik_seeds": 16,          # 验收用满种子
    "ik_max_iters": 300,     # 新迭代上限(16.9x 提速来源)
}
print("渲染配置:", json.dumps(BEST), flush=True)
print(f"IK: seeds={G.IK_SEEDS} max_iters={G.IK_MAX_ITERS} pos_tol={G.IK_POS_TOL}", flush=True)

ep = G.build_episode(on_step=None, collect_frames=True, params=BEST)
v, m = ep["verdict"], ep["verdict"]["metrics"]
PLATE = np.array([0.18396, 0.22788])
EXT = np.array([0.2504, 0.2513])
GT = np.array([-0.06, 0.06])
duck = np.asarray(ep["duck_trace"][-1])
off = (duck[:2] - PLATE) / EXT

print("\n=== FINAL (新 IK, base=[0.30,-0.38]) ===", flush=True)
print("verdict(raw) = %s %s" % ("PASS" if v["pass"] else "FAIL", v["reasons"]), flush=True)
print("follow_max   = %.1f mm" % (m.get("follow_max_m", float("nan")) * 1000), flush=True)
print("lift/slip/knock = %.2f / %.2f / %.2f cm"
      % (m["lift_m"] * 100, m["hold_slip_m"] * 100, m["knock_m"] * 100), flush=True)
print("duck_final   = %s   z=%.4f (盘面静置 0.8046 / 桌面静置 0.7793)"
      % (np.round(duck, 4).tolist(), duck[2]), flush=True)
print("归一化偏移   = [%+.3f, %+.3f]  人类真值=[%+.3f, %+.3f]  偏离=%.4f"
      % (off[0], off[1], GT[0], GT[1], np.linalg.norm(off - GT)), flush=True)
print("折算误差     = x %.1f cm   y %.1f cm"
      % (abs(off[0] - GT[0]) * EXT[0] * 100, abs(off[1] - GT[1]) * EXT[1] * 100), flush=True)

# ---------- 5 视角 GIF / MP4 ----------
out = os.path.join(G.OUT_DIR, "robotwin_r1_duck_plate_best")
frames = ep["frames"]
pal = [Image.fromarray(f).quantize(colors=64, method=Image.MEDIANCUT) for f in frames]
pal[0].save(out + ".gif", save_all=True, append_images=pal[1:], duration=80,
            loop=0, optimize=True, disposal=2)
imageio.mimsave(out + ".mp4", frames, fps=12, macro_block_size=1)
print("\n5视角 GIF/MP4 -> %s  (%d 帧)" % (out, len(frames)), flush=True)

# ---------- 与真实视频并排 ----------
video = load_video("/path/to/workspace/7.mp4")
panels = make_panels(ep, video, "best")
print("对比图 -> %s" % panels, flush=True)

rv = os.path.join(G.OUT_DIR, "robotwin_r1_duck_replay.mp4")
make_replay_video(ep, video, rv)
print("重放视频 -> %s" % rv, flush=True)

json.dump({
    "params": BEST,
    "metrics": {k: (float(x) if isinstance(x, (int, float, np.floating)) else x)
                for k, x in m.items()},
    "gt_dist": float(np.linalg.norm(off - GT)),
    "gt_off": [float(off[0]), float(off[1])],
    "duck_final": [float(x) for x in duck],
    "verdict_raw": v["reasons"],
}, open(os.path.join(G.OUT_DIR, "best_v2_report.json"), "w"),
    ensure_ascii=False, indent=1, default=float)
print("RENDER_DONE", flush=True)
