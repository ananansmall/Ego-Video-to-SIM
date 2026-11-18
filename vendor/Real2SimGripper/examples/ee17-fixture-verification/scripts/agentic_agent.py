#!/usr/bin/env python3
"""agentic_agent.py —— 真·agent 循环：**VLM 自己选工具 + 参数**。

与 `agentic_optimize.py` 的区别（A/B 的唯一变量）：

    A  agentic_optimize.pattern_search
       决策者 = 写死的坐标下降:  for param in SPACE: 试 ±step
       工具   = 只有 evaluate

    B  agentic_agent.agent_loop（本文件）
       决策者 = VLM: 每轮选一个工具（evaluate / propose_neighbors /
                query_history / score_of）并给参数
       工具   = 评测 + 邻域候选 + 历史查询 + 评分
       记忆   = 缓存 + 动作级记录（recent_actions），解决 VLM 重复自己的问题

相同点：起点、预算（评测次数）、评分函数、缓存文件 —— 全一致，可比。

用法:
    python agentic_agent.py --task configs/duck_to_plate.json --budget 4
    python agentic_agent.py --task configs/duck_to_plate.json --budget 4 --no-vlm  # 退化: 贪心邻域
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(os.path.dirname(_HERE), "results")
os.makedirs(_RESULTS, exist_ok=True)
sys.path.insert(0, _HERE)

import agentic_optimize as AO
from agent_tools import AgentTools, tool_manifest


# ---------------------------------------------------------------- VLM 决策
def prompt_view(value, include_gt):
    """Redact explicit GT fields from every prompt channel for the controlled arm."""
    if isinstance(value, dict):
        return {k: prompt_view(v, include_gt) for k, v in value.items()
                if include_gt or not str(k).startswith("gt_")}
    if isinstance(value, list):
        return [prompt_view(v, include_gt) for v in value]
    return value


def build_prompt(best, tools, budget_left, include_gt=True):
    hist = prompt_view(tools.query_history(limit=6), include_gt)
    acts = prompt_view(tools.recent_actions(limit=5), include_gt)
    met = best["metrics"]
    visible = {k: met[k] for k in ("follow_max_mm", "plate_margin_mm", "plate_offset_mm", "plate_rb",
               "on_plate", "pass_eff", "gt_dist", "gt_x_cm", "gt_y_cm") if k in met}
    visible = prompt_view(visible, include_gt)
    objective = ("gt_dist is normalised human-demo displacement (lower is better); "
                 "gt_x_cm / gt_y_cm are absolute per-axis errors in centimetres. "
                 "Improve fidelity while keeping physical success and follow_max_mm < 100."
                 if include_gt else "Improve the score while keeping physical success and follow_max_mm < 100.")
    return f"""You are driving a real-to-sim parameter search for a robot grasping replay.
Each expensive evaluation runs a physics episode, so budget is scarce.
{objective}
Parameter bounds = {json.dumps(tools.space)}
base_xy y bounds = {json.dumps(AO.BASE_XY_Y)}
base_xy is the robot base position in metres; plate_rb is the plate centre in metres.
place_t=0 aims at the plate centre; place_t=1 aims at its edge toward the robot.

{tool_manifest()}

=== CURRENT BEST ===
params = {json.dumps(best['params'], ensure_ascii=False)}
score  = {AO.score(best['metrics']):.1f}
metrics = {json.dumps(visible, ensure_ascii=False)}

=== HISTORY (best 6, higher score is better) ===
{json.dumps(hist, ensure_ascii=False, indent=1)}

=== YOUR RECENT ACTIONS (do NOT repeat something that already failed) ===
{json.dumps(acts, ensure_ascii=False, indent=1) if acts else '(none yet)'}

=== BUDGET ===
{budget_left} expensive evaluation(s) left.

Decide the SINGLE next tool call. Prefer `propose_neighbors` + `evaluate` when you have a
promising direction; use `query_history` if you are unsure what has been tried.
Reply JSON only:
{{"tool": "evaluate"|"propose_neighbors"|"query_history"|"score_of",
  "params": {{...optional partial param override...}},
  "keys": [..optional, for propose_neighbors..],
  "limit": <optional int, for query_history>,
  "reason": "<short>"}}
"""


def vlm_decide(prompt):
    from agentic_replay import vlm_call, parse_json
    return parse_json(vlm_call(prompt, []))


# ---------------------------------------------------------------- agent 循环
def merge(base, override):
    p = dict(base)
    if override:
        for k, v in override.items():
            if k in AO.SPACE:
                p[k] = AO.clamp(k, v)
    return AO._jsonify(p)


def agent_loop(seed, tools, budget, use_vlm=True, verbose=True,
               max_free=2, include_gt=True, strict_vlm=False):
    """max_free: 连续"不花预算"的动作上限。超了就强制 evaluate。

    实测 bug: query_history / propose_neighbors 不耗预算, VLM 会在两者之间
    无限打转(一次运行 10 个动作、0 次真评测), 永远跑不完。
    """
    if budget < 1:
        raise ValueError("budget must include at least the seed evaluation")
    best = {"params": AO._jsonify(seed), "metrics": tools.evaluate(seed)}
    print(f"  [seed] score={AO.score(best['metrics']):.1f} follow="
          f"{best['metrics'].get('follow_max_mm', -1):.1f}mm", flush=True)

    free_streak = 0       # 连续免费动作计数
    actions = 0
    while tools.n_eval < budget and actions < (budget + 1) * (max_free + 1):
        actions += 1
        left = budget - tools.n_eval

        if use_vlm:
            try:
                act = vlm_decide(build_prompt(best, tools, left, include_gt=include_gt))
                if not isinstance(act, dict):
                    raise ValueError("VLM action must be an object")
            except Exception as e:
                if strict_vlm:
                    raise
                print(f"  [VLM 失败] {e} → 退回贪心邻域", flush=True)
                act = {"tool": "propose_neighbors", "reason": "vlm-failed"}
        else:
            act = {"tool": "propose_neighbors", "reason": "greedy"}

        t = act.get("tool")
        if t not in ("evaluate", "propose_neighbors", "query_history", "score_of"):
            tools.log_action(act, "unsupported tool; using an untried evaluation")
            act = {"tool": "evaluate", "reason": "unsupported-tool"}
            t = "evaluate"
        print(f"  [agent] {t}  {act.get('reason', '')[:80]}", flush=True)

        if t in ("query_history", "score_of"):
            free_streak += 1
            result = (tools.query_history(limit=6) if t == "query_history"
                      else tools.score_of(best["metrics"]))
            tools.log_action(act, result)
            if free_streak >= max_free:
                print(f"  [强制] 连续 {free_streak} 次免费动作, 转入 evaluate", flush=True)
                t = "evaluate"
            else:
                continue

        if t == "propose_neighbors":
            free_streak += 1
            cands = tools.propose_neighbors(merge(best["params"], act.get("params")),
                                            keys=act.get("keys"))
            tools.log_action(act, cands)
            if not cands:
                print("  [无候选] 收敛或空间已穷尽", flush=True)
                break
            # 候选已应用 VLM 的中心覆盖，直接取未试邻域，避免覆盖回已试点。
            pick = cands[0]["params"]
            if AO.key_of(pick) in tools.tried_configs():
                continue
            met = tools.evaluate(pick)
            free_streak = 0
            tools.log_action({"tool": "evaluate", "params": pick}, f"score={AO.score(met):.1f}")
            print(f"       -> score={AO.score(met):.1f} follow={met.get('follow_max_mm', -1):.1f}mm",
                  flush=True)
            if AO.score(met) > AO.score(best["metrics"]):
                best = {"params": pick, "metrics": met}
                print("       ** 改进", flush=True)
            continue

        if t == "evaluate":
            p = merge(best["params"], act.get("params"))
            if AO.key_of(p) in tools.tried_configs():
                # 重复评测不耗预算 -> VLM 会在这里无限打转
                # (实测连续 6 次 duplicate-skip, 它自己都说
                #  "stuck in a duplicate-skip loop")。
                # 修法: 自动改取"尚未试过的邻域候选", 保证必然推进或终止。
                print("       -> 该配置已评测过, 自动改取未试候选", flush=True)
                tools.log_action(act, "duplicate-skip->neighbors")
                cands = tools.propose_neighbors(best["params"], keys=act.get("keys"))
                if not cands:
                    print("  [无未试候选] 收敛退出", flush=True)
                    break
                p = cands[0]["params"]
                print("       -> 改用 %s" % cands[0]["moved"], flush=True)
            met = tools.evaluate(p)
            free_streak = 0
            tools.log_action(act, f"score={AO.score(met):.1f}")
            print(f"       -> score={AO.score(met):.1f} follow={met.get('follow_max_mm', -1):.1f}mm",
                  flush=True)
            if AO.score(met) > AO.score(best["metrics"]):
                best = {"params": p, "metrics": met}
                print("       ** 改进", flush=True)
            continue

    return best


def main():
    ap = argparse.ArgumentParser(description="B: VLM 选工具的 agent 循环")
    ap.add_argument("--task", default="../configs/duck_to_plate.json")
    ap.add_argument("--budget", type=int, default=4, help="昂贵评测次数上限")
    ap.add_argument("--seed-t", type=float, default=0.0)
    ap.add_argument("--seed-ts", type=float, default=3.0)
    ap.add_argument("--ik-seeds", type=int, choices=(4, 8, 16), default=4,
                    help="搜索默认 fast=4；balanced=8；验收 accurate=16")
    ap.add_argument("--without-gt", action="store_true", help="prompt 消融：隐藏显式 GT 指标")
    ap.add_argument("--no-vlm", action="store_true", help="不调 VLM, 退化为贪心邻域")
    a = ap.parse_args()

    import pipeline as PIPE
    from task_config import load_task
    cfg = load_task(a.task)
    cfg.setdefault("tuning", {})["ik_seeds"] = a.ik_seeds
    PIPE.apply_adapters(cfg, verbose=True)

    AO.TASK_CFG = cfg
    # ⚠ 必须显式设这两个 tag, 否则缓存 key 用默认的 hull1/r1 算,
    #   与 coacd 条目永远不匹配 —— 实测导致每个配置都白跑一次 20 分钟。
    #   (agentic_optimize.main() 有这段, 这里漏了)
    from collision_config import state as _cs
    AO.COLLISION_TAG = "coacd" if _cs().get("enabled") else "hull1"
    AO.ROBOT_NAME = cfg["robot"].get("name", "r1")
    cache = AO.load_cache()
    AO.rescore(cache)

    from vlm_lifecycle import ensure_vlm, release_vlm
    use_vlm = False
    if not a.no_vlm:
        use_vlm = ensure_vlm()

    try:
        seed = dict(cfg.get("tuning", {}))
        seed["base_xy"] = list(cfg["robot"]["base_xy"])
        seed["plate_geom"] = (cfg.get("target") or {}).get("geom")
        seed["place_t"] = a.seed_t
        seed["time_scale"] = a.seed_ts
        seed = AO._jsonify(seed)

        tools = AgentTools(cache, AO.SPACE, cfg)
        print(f"[agent] 起点 base={seed['base_xy']} place_t={seed['place_t']} "
              f"ts={seed['time_scale']}  VLM={'开' if use_vlm else '关'}", flush=True)
        best = agent_loop(seed, tools, a.budget, use_vlm=use_vlm, include_gt=not a.without_gt)

        AO.save_cache(cache)
        out = {"best_params": best["params"], "best_score": AO.score(best["metrics"]),
               "metadata": best["metrics"], "n_eval": tools.n_eval,
               "ik_seeds": a.ik_seeds, "include_gt": not a.without_gt,
               "actions": tools.log, "mode": "agent" if use_vlm else "greedy"}
        p = os.path.join(_RESULTS, "agentic_agent_best.json")
        json.dump(out, open(p, "w"), ensure_ascii=False, indent=1, default=float)
        print(f"\n最优 score={out['best_score']:.1f}  params={best['params']}")
        print(f"报告 -> {p}")
    finally:
        release_vlm()


if __name__ == "__main__":
    main()
