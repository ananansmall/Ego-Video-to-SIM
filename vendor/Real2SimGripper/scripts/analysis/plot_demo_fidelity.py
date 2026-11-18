#!/usr/bin/env python3
"""plot_demo_fidelity.py — 创新点 1 的核心证据图。

把「人类示范真值」（从真实视频用颜色分割提取的 鸭-盘 归一化相对位置）
和「仿真 replay 的实际落点」画在同一坐标系里，回答一个问题：

    物理 verifier 说 PASS，但仿真真的复现了人类的放置吗？

真值来源（2026-10-16 实测，见 docs/05_示范真值与创新点.md）：
    7.mp4 末段 F108-112，严格颜色阈值 → GT = [-0.042, +0.042]，std [0.005, 0.003]
    多组阈值/帧范围稳健性检验：x ∈ [-0.11, -0.04]，y ∈ [+0.02, +0.11]  → 即「放在盘心」
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))          # scripts/
_RESULTS = _os.path.join(_os.path.dirname(_os.path.dirname(_HERE)), "results")
_CACHE = _os.path.join(_RESULTS, "agentic_optimize_cache.json")
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.dirname(_HERE))          # scripts/
_RESULTS = _os.path.join(_os.path.dirname(_os.path.dirname(_HERE)), "results")
_CACHE = _os.path.join(_RESULTS, "agentic_optimize_cache.json")
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = _CACHE
OUT = os.path.join(HERE, "agentic_replay_frames", "demo_fidelity.png")

PLATE = np.array([0.18396, 0.22788])      # geometry_8 盘心（仿真系）
EXT = np.array([0.2504, 0.2513])          # 盘包围盒尺寸
GT = np.array([-0.06, 0.06])              # 人类真值（多组阈值合并）
GT_SD = np.array([0.03, 0.03])


def load():
    c = json.load(open(CACHE))
    seen = {}
    for v in c.values():
        m, p = v["metrics"], v["params"]
        df = m.get("duck_final")
        if df is None or m.get("plate_rb") is None:
            continue
        key = (p.get("place_t"), p.get("time_scale"), p.get("place_hover"),
               tuple(p.get("base_xy", [])), p.get("grip_open_q"))
        if key in seen:
            continue
        pe = m.get("pass_eff")
        if pe is None:
            pe = m.get("pass", False)
        seen[key] = ((np.asarray(df, float)[:2] - PLATE) / EXT, bool(pe), m.get("follow_max_mm"))
    return seen


def main():
    seen = load()
    fig, ax = plt.subplots(figsize=(8.4, 8.2))
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, color="tab:blue", alpha=.13, ec="k", lw=2))
    ax.plot([-0.5, 0.5], [0, 0], "k--", lw=.6)
    ax.plot([0, 0], [-0.5, 0.5], "k--", lw=.6)
    ax.errorbar(GT[0], GT[1], xerr=GT_SD[0], yerr=GT_SD[1], fmt="*", ms=28,
                color="crimson", ecolor="crimson", elinewidth=2, capsize=5, zorder=6,
                label="HUMAN ground truth (extracted from real video)")
    lab = set()
    for key, (off, ok, fm) in sorted(seen.items(),
                                              key=lambda x: -np.linalg.norm(x[1][0] - GT)):
        col = "seagreen" if ok else "firebrick"
        tag = "sim replay — verifier PASS" if ok else "sim replay — verifier FAIL"
        ax.plot(off[0], off[1], "o" if ok else "X", ms=11, mew=2.4, color=col,
                label=tag if tag not in lab else None)
        lab.add(tag)
        pt, bx = key[0], key[3]
        ax.annotate("t=%.2f bx=%.2f\n%.0fmm" % (pt, bx[0] if bx else -9, fm or -1),
                    off + np.array([0.02, 0.02]), fontsize=6.5, color="dimgray")
    ax.set_xlim(-0.75, 0.55)
    ax.set_ylim(-1.05, 0.35)
    ax.set_aspect("equal")
    ax.set_xlabel("normalised x offset   (duck - plate_centre) / plate_size")
    ax.set_ylabel("normalised y offset")
    ax.set_title("Demonstration-grounded fidelity: where does the duck actually land?\n"
                 "(blue square = the plate. Human puts it at the centre; "
                 "no sim config gets close,\nand the physics verifier says PASS anyway)",
                 fontsize=10.5)
    ax.grid(alpha=.25)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print("saved ->", OUT, " 配置数 =", len(seen))
    # 顺便打印量化结论
    ok_cfgs = [(k, v) for k, v in seen.items() if v[1]]
    if ok_cfgs:
        k, v = min(ok_cfgs, key=lambda x: np.linalg.norm(x[1][0] - GT))
        off = v[0]
        print("最接近真值且 PASS: place_t=%.2f ts=%.1f hover=%.2f base_xy=%s"
              % (k[0], k[1], k[2], k[3]))
        print("  归一化偏移 = [%+.3f, %+.3f]  vs 真值 [%+.3f, %+.3f]"
              % (off[0], off[1], GT[0], GT[1]))
        print("  折算物理误差: x=%.1fcm  y=%.1fcm"
              % (abs(off[0] - GT[0]) * EXT[0] * 100, abs(off[1] - GT[1]) * EXT[1] * 100))


if __name__ == "__main__":
    main()
