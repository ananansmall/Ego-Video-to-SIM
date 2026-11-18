#!/usr/bin/env python3
"""scan_base.py — 纯左右平移基座扫描 (无 VLM / 无随机搜索)。

问题: 默认基座 [0,-0.50] 下, 人手搬运高点超出 R1 左臂可达空间, 后段 IK 顶限位冻结
(follow_max ≈ 230mm)。本脚本验证「只把基座沿 +x 左右平移」即可让左臂工作空间覆盖
搬运高点、消除冻结, 找到最简基座位置。

做法: 固定前后 y (靠桌安全位), 在左右 x 上网格扫描, 各自跑一次 build_episode
(不渲染), 记录 follow_max / 抬升 / 滑移 / pass。

实测 (y=-0.42): x<=0.0 冻结(follow_max≈230~350mm); x=0.1~0.2 完整跟随(74/62mm);
x=0.3 又变差(128mm)。故左右甜区 x≈0.1~0.2, 默认取 0.20。

用法:
  CUDA_VISIBLE_DEVICES=0 python scripts/scan_base.py
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import robotwin_r1_duck_grasp as G  # noqa: E402

X_GRID = [-0.20, -0.10, 0.00, 0.10, 0.20, 0.30]   # 左右 (机器人在桌面系 +x 侧为右)
Y_FIXED = -0.42                                    # 前后 (靠桌安全位)
OUT_JSON = os.path.join(_HERE, "scan_base_best.json")


def main():
    records = []
    for x in X_GRID:
        params = dict(G.PARAM_DEFAULTS)
        params["base_xy"] = [float(x), Y_FIXED]
        try:
            ep = G.build_episode(on_step=None, collect_frames=False, params=params)
            v, m = ep["verdict"], ep["verdict"]["metrics"]
            rec = {"x": float(x), "y": Y_FIXED, "pass": v["pass"],
                   "follow_max_mm": round(m.get("follow_max_m", -1) * 1000, 1),
                   "lift_cm": round(m["lift_m"] * 100, 2),
                   "knock_cm": round(m["knock_m"] * 100, 2),
                   "slip_cm": round(m["hold_slip_m"] * 100, 2)}
        except Exception as e:
            rec = {"x": float(x), "y": Y_FIXED, "pass": False, "error": str(e)[:80]}
        records.append(rec)
        print(f"[扫描] base_x={x:+.2f} -> {rec}", flush=True)

    records.sort(key=lambda r: r.get("follow_max_mm", 1e9))
    best = records[0]
    json.dump({"best": best, "records": records, "y_fixed": Y_FIXED},
              open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"\n最优简单平移基座: x={best['x']:+.2f}, y={Y_FIXED} "
          f"(follow_max={best.get('follow_max_mm')}mm, pass={best['pass']})")


if __name__ == "__main__":
    main()
