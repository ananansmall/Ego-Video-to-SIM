#!/usr/bin/env python3
"""agent_tools.py —— agent 可调用的**工具层**。

把"脚本里写死的调用顺序"变成"agent 可以自己选工具"。

工具分两类：

  免费（不跑仿真，可随便调）:
    query_history(limit, only_pass)   查历史 —— **记忆**
    propose_neighbors(params, keys)   数值邻域候选 —— 数值能力
    tried_configs()                   已评测过的配置指纹（避免重复）
    score_of(metrics)                 给指标打分 —— 判据

  昂贵（一次 ~20 分钟，有缓存则不重复算）:
    evaluate(params)                  跑一次仿真 → metrics
    render(params)                    出对比图（内部会 evaluate 一次，缓存命中则只渲图）

设计要点：
  - 每个工具返回**可 JSON 化**的结果，便于直接塞进 VLM 的 prompt
  - 工具不知道 VLM 存在；VLM 也不知道工具内部实现 —— 两侧解耦
  - `evaluate` 复用 `agentic_optimize.evaluate`，因此与 A（坐标下降）共享同一缓存
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import agentic_optimize as AO            # 复用它的 evaluate / score / key_of / SPACE


class AgentTools:
    """agent 能用的全部动作。所有决策都由调用方(agent)做，这里只提供能力。"""

    def __init__(self, cache, space=None, cfg=None, log=None):
        self.cache = cache
        self.space = dict(space or AO.SPACE)
        self.cfg = cfg
        self.log = log or []
        self.n_eval = 0

    # ------------------------------------------------ 免费工具
    def score_of(self, met):
        """判据：给一组指标打分（越大越好）。"""
        return AO.score(met)

    def query_history(self, limit=8, only_pass=False):
        """**记忆查询**：返回历史评测记录（按分数降序），供 agent 参考"试过什么"。"""
        rows = []
        for k, v in self.cache.items():
            if not AO.cache_matches(k, v):
                continue
            s = v.get("score")
            if s is None or s < -1e8:
                continue
            met = v.get("metrics") or {}
            if only_pass and not met.get("pass_eff", met.get("pass")):
                continue
            rows.append({
                "params": {k: v["params"].get(k) for k in self.space if k in v["params"]},
                "score": round(float(s), 1),
                "gt_dist": round(float(met["gt_dist"]), 3) if met.get("gt_dist") is not None else None,
                "gt_x_cm": met.get("gt_x_cm"),
                "gt_y_cm": met.get("gt_y_cm"),
                "follow_mm": round(float(met.get("follow_max_mm") or -1), 1),
                "pass": bool(met.get("pass_eff", met.get("pass"))),
                "tag": v.get("tag"),
            })
        rows.sort(key=lambda r: -r["score"])
        return rows[:limit]

    def tried_configs(self):
        """已评测过的配置指纹集合（用于避免重复评测）。"""
        out = set()
        for k, v in self.cache.items():
            if AO.cache_matches(k, v):
                out.add(k)
        return out

    def propose_neighbors(self, params, keys=None, step_scale=1.0):
        """**数值能力**：给出该点周围尚未评测的候选（±step）。agent 可以直接挑一个。"""
        tried = self.tried_configs()
        keys = keys or list(self.space)
        cands = []
        for k in keys:
            if k not in self.space:
                continue
            for sgn in (+1, -1):
                try:
                    p = AO.perturb(params, k, sgn, step_scale, self.space)
                except Exception:
                    continue
                p = AO._jsonify(p)
                if AO.key_of(p) in tried:
                    continue
                cands.append({"params": p, "moved": f"{k}{'+' if sgn>0 else '-'}", "cost": 1})
        return cands

    def log_action(self, act, result):
        """记录一次动作与结果 —— 这就是**动作级记忆**，解决 VLM 重复自己的问题。"""
        self.log.append({"action": act, "result": result})

    def recent_actions(self, limit=6):
        return self.log[-limit:]

    # ------------------------------------------------ 昂贵工具
    def evaluate(self, params):
        """跑一次仿真。**先查缓存** —— 命中则秒回且不计入预算。返回 metrics。"""
        if self.cfg is None:
            raise RuntimeError("AgentTools 需要 cfg 才能 evaluate")
        k = AO.key_of(params)
        hit = self.cache.get(k)
        if hit and (hit.get("metrics") or {}).get("duck_final") is not None:
            print("       [缓存命中] 不计预算", flush=True)
            return hit["metrics"]
        ep, met = AO._run(params, collect_frames=False)
        self.n_eval += 1
        k = AO.key_of(params)
        self.cache[k] = {
            "params": AO._jsonify(params),
            "metrics": met,
            "score": AO.score(met),
            "tag": "agent",
            "robot": AO.ROBOT_NAME,
            "collision": AO.COLLISION_TAG,
        }
        return met

    def render(self, params):
        """出对比图（内部 evaluate 一次；缓存命中时不会重跑物理）。"""
        if self.cfg is None:
            raise RuntimeError("AgentTools 需要 cfg 才能 render")
        ep, met = AO._run(params, collect_frames=True)
        from agentic_replay import load_video, make_panels
        video = load_video(self.cfg["camera"]["ego_video"])
        return make_panels(ep, video, "agent")


def tool_manifest():
    """给 VLM 看的工具清单（紧凑文本，避免 prompt 过长）。"""
    return """可用工具（每次只能选一个）:
  evaluate         params={...}                    跑一次仿真，返回 metrics（昂贵，耗时取决于 IK 种子数）
  propose_neighbors params={...}, keys=[...]       返回该点周围尚未试过的候选（免费）
  query_history    limit=N, only_pass=false        查历史最优 N 条（免费）
  score_of         metrics={...}                   给指标打分（免费）

注意:
  - 预算有限，昂贵的工具要慎用
  - 用 query_history 先看"试过什么"，避免重复（这一点非常关键）
"""
