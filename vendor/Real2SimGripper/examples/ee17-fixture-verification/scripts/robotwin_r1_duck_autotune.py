#!/usr/bin/env python3
"""robotwin_r1_duck_autotune.py — 仿真闭环 critic: 抓取失败时自动重搜参数。

思路 (借鉴 Agentic Real2Sim 的 simulator-in-the-loop 优化 / DexAgent 的 verifier):
  1) 用默认参数跑一次 build_episode -> verify_grasp 得到 PASS/FAIL + 失败原因
  2) 若 FAIL, 在一个有界参数空间内做随机搜索 (抬升/滑移/抓前位移 打分),
     每次 trial 重建场景并执行；保留 score 最优配置
  3) 输出最优参数 + 对比报告 (JSON), 供 grasp/dataset 复用

不用渲染 (collect_frames=False) 以加速; 成功配置可再单独跑渲染。

用法:
  CUDA_VISIBLE_DEVICES=0 python _scratch/robotwin_r1_duck_autotune.py [--trials 8] [--seed 0]
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import robotwin_r1_duck_grasp as G  # noqa: E402

OUT_JSON = os.path.join(_HERE, "autotune_best_params.json")


def score(v):
    """把 verdict 变成可比较的标量 (越大越好): 通过优先, 其次抬升大/滑移小/未撞飞。"""
    m = v["metrics"]
    s = 0.0
    s += 1000.0 if v["pass"] else 0.0
    s += m["lift_m"] * 100.0
    s -= m["hold_slip_m"] * 100.0
    s -= m["knock_m"] * 200.0
    s -= m.get("follow_max_m", 0.0) * 150.0   # 跟随段最大误差越小越好 (基座搜索主目标)
    return s


def sample_params(rng, base):
    """在关键调参项上采样一个候选 (围绕默认值的有界扰动)。"""
    p = dict(base)
    # 基座位置: 左右 x ±0.30, 前后 y 靠桌但留安全边距 (桌沿 y=-0.35)
    p["base_xy"] = [float(np.clip(base["base_xy"][0] + rng.uniform(-0.30, 0.30), -0.35, 0.35)),
                    float(np.clip(base["base_xy"][1] + rng.uniform(-0.15, 0.10), -0.70, -0.42))]
    p["duck_rb_xy"] = (G.DUCK_RB_XY + rng.uniform(-0.03, 0.03, 2)).astype(float)
    p["d_hover"] = float(np.clip(base["d_hover"] + rng.uniform(-0.03, 0.05), 0.08, 0.20))
    p["grip_hold_q"] = float(np.clip(base["grip_hold_q"] + rng.uniform(-0.008, 0.004), 0.022, 0.034))
    p["close_f"] = int(np.clip(base["close_f"] + rng.integers(-4, 5), 40, 60))
    p["open_f"] = int(np.clip(base["open_f"] + rng.integers(-6, 7), 84, 100))
    return p


def run_trial(params, idx):
    print(f"\n{'='*66}\n[trial {idx}] params: base={np.round(params['base_xy'],3).tolist()} "
          f"duck_xy={np.round(params['duck_rb_xy'],3).tolist()} "
          f"d_hover={params['d_hover']:.3f} hold_q={params['grip_hold_q']:.4f} "
          f"close_f={params['close_f']} open_f={params['open_f']}")
    try:
        ep = G.build_episode(on_step=None, collect_frames=False, params=params)
        return ep["verdict"], ep
    except Exception as e:  # 场景/物理异常视作该 trial 失败
        print(f"  trial 异常: {e}")
        return {"pass": False, "checks": {}, "metrics": {"lift_m": 0, "knock_m": 1e9,
                "hold_slip_m": 1e9, "landed_z": 0}, "reasons": ["exception"]}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8, help="随机搜索的额外 trial 数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = dict(G.PARAM_DEFAULTS)
    history = []

    # trial 0: 默认参数
    v0, ep0 = run_trial(base, 0)
    history.append({"params": _jsonify(base), "verdict": v0, "score": score(v0)})
    best = history[0]

    if not v0["pass"]:
        for i in range(1, args.trials + 1):
            cand = sample_params(rng, base)
            v, _ = run_trial(cand, i)
            rec = {"params": _jsonify(cand), "verdict": v, "score": score(v)}
            history.append(rec)
            if rec["score"] > best["score"]:
                best = rec
                print(f"  -> 新最优 (score={best['score']:.2f}, pass={v['pass']})")
            if best["verdict"]["pass"]:
                print(f"  -> 找到通过配置, 提前停止 (trial {i})")
                break
    else:
        print("\n默认参数已通过 verifier, 无需重搜。")

    # 输出
    report = {
        "best_params": best["params"],
        "best_verdict": best["verdict"],
        "passed": best["verdict"]["pass"],
        "n_trials": len(history),
        "history": history,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*66}")
    print(f"最优参数 (pass={report['passed']}): {best['params']}")
    print(f"报告 -> {OUT_JSON}")


def _jsonify(p):
    """把 numpy 值转成可 JSON 序列化。"""
    out = {}
    for k, v in p.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
