#!/usr/bin/env python3
"""v2_collision_report.py — 碰撞升级批量聚合报告 (旧 dataset vs 新 results)。

对每个 ee{N}_mocap(_wK) 窗口: 旧 acceptance (dataset/, 无碰撞基线) vs
新 run (results/, 带碰撞+场景云): 抓起 / 抬升 / 跟随 / 碰撞盒数 → 回归清单。

用法: python test/v2_collision_report.py [--json]
"""
import glob
import json
import os
import sys

ROOT = "/path/to/Real2SimGripper"
LR = f"{ROOT}/results/large_range"
DS = f"{ROOT}/dataset"


def load_acc(p):
    try:
        d = json.load(open(p))
    except Exception:
        return None
    o = d.get("object") or {}
    acc = o.get("acceptance") or {}
    crit = acc.get("criteria") or {}
    return {
        "grasped": o.get("grasped"),
        "lift_mm": round(o.get("lift_mm") or 0, 1),
        "follow_mm": round(d.get("phys_pos_err_mm", {}).get("mean") or 0, 1),
        "accept": f"{sum(1 for v in crit.values() if v)}/{len(crit)}",
        "collision": d.get("collision"),
        "travel_m": round(d.get("base", {}).get("travel_m") or 0, 1),
    }


def main():
    import time
    batch_start = time.time() - 3600 * 36   # 近 36h 内的 run 视为本轮重跑
    rows = []
    for p in sorted(glob.glob(f"{LR}/ee*_mocap*_large_range.json")):
        name = os.path.basename(p).replace("_large_range.json", "")
        if "_fix_" in name or "_env" in name or "_coll" in name or name.startswith("ee17_posture"):
            continue                      # 实验/诊断 run, 非数据集窗口
        new = load_acc(p)
        old = load_acc(f"{DS}/{name}/acceptance.json")
        if new is None or old is None:
            continue
        fresh = os.path.getmtime(p) > batch_start
        env_n = (new.get("collision") or {}).get("env_boxes_n")
        if not fresh and env_n is None:
            continue                      # 未重跑 (旧结果), 不进对比
        verdict = "?"
        if new["grasped"] and not old["grasped"]:
            verdict = "改进"
        elif old["grasped"] and not new["grasped"]:
            verdict = "回归"
        elif new["grasped"] and old["grasped"]:
            verdict = "持平" if abs(new["lift_mm"] - old["lift_mm"]) < 40 else (
                "改进" if new["lift_mm"] > old["lift_mm"] else "持平")
        else:
            verdict = "双失败"
        rows.append((name, old, new, env_n, verdict))

    n_reg = sum(1 for r in rows if r[4] == "回归")
    n_imp = sum(1 for r in rows if r[4] == "改进")
    print(f"{'窗口':22s} {'旧抓起':6s} {'旧抬升':8s} {'新抓起':6s} {'新抬升':8s} "
          f"{'盒':4s} {'判定':4s}")
    for name, old, new, env_n, v in rows:
        print(f"{name:22s} {str(old['grasped']):6s} {old['lift_mm']:>6.0f}mm "
              f"{str(new['grasped']):6s} {new['lift_mm']:>6.0f}mm "
              f"{str(env_n):>4s} {v}")
    print(f"\n窗口 {len(rows)}: 改进 {n_imp} | 回归 {n_reg} | 其余持平/双失败")
    if "--json" in sys.argv:
        out = f"{LR}/v2_collision_report.json"
        json.dump({"rows": [{"name": n, "old": o, "new": w, "env_boxes": e,
                             "verdict": v} for n, o, w, e, v in rows],
                   "n_regression": n_reg, "n_improve": n_imp},
                  open(out, "w"), ensure_ascii=False, indent=1, default=str)
        print(f"✓ {out}")


if __name__ == "__main__":
    main()
