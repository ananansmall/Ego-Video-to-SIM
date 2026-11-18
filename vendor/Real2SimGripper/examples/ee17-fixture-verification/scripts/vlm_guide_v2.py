#!/usr/bin/env python3
"""vlm_guide_v2.py —— 聚焦版 VLM 指导（A/B 对照 v1 的失败 prompt）

v1 的问题（实测 2026-10-18）：给了 9 个耦合参数 + 10 个无优先级指标 + 一条写错的
手工规则（"贴边→place_t"，而 place_t 0.6→0.2 会让 follow 从 81 爆到 105mm）。
结果 VLM 两次给出一模一样的建议、且建议让 score 从 1235 掉到 261。

v2 把 `agentic_place.py`（**同一个 VLM 模型，已证明能做好空间推理**）的成功要素搬过来：
  1. 只让它决策 2 个变量（base_xy / place_t），而不是 9 个
  2. 明确告诉它 score 和**主目标**（示范保真度 gt_dist），而不是一堆平级数字
  3. 给它**可达性先验**和**因果知识**（place_t 与 follow 的 trade-off；
     base_x 对齐目标 x 能同时改善两者 —— 这是实测结论）
  4. 不给手写启发式规则，改为给证据
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def build_prompt(params, met, plate_xy):
    """构造聚焦版 prompt。plate_xy 用于告诉它"目标在哪、基座该对齐到哪"。"""
    gt = met.get("gt_dist")
    gt_off = met.get("gt_off") or [None, None]
    fm = met.get("follow_max_mm")
    off = met.get("plate_offset_mm")
    user = f"""You are tuning a real-to-sim replay. A robot gripper picks a pink duck and should place it on a blue plate, reproducing what the HUMAN did in the real video.

=== PRIMARY OBJECTIVE ===
Demonstration fidelity: gt_dist = {gt:.3f}  (normalised; 0.0 = exactly matches the human)
  - The human placed the duck essentially at the plate CENTRE.
  - Our replay is off by {met.get('gt_y_cm', -1):.1f} cm in y and {met.get('gt_x_cm', -1):.1f} cm in x.
  - MINIMISING this is the main goal.

=== HARD CONSTRAINT ===
follow_max = {fm:.1f} mm  (MUST stay < 100 mm; it is the arm's reachability limit)
  - If follow_max exceeds 100 mm the IK freezes and the replay FAILS.

=== YOU MAY CHANGE EXACTLY ONE PARAMETER ===

(1) "base_xy" = [x, y]  — where the robot stands, in metres.
    MEASURED EVIDENCE: aligning base_x with the target's x improves BOTH fidelity and
    follow at the same time. In our sweep, base_x 0.12 -> 0.30 cut the y error from
    10.3 cm to 3.4 cm AND improved follow from 90 to 84 mm.
    The target (plate centre) is at x = {plate_xy[0]:.3f}, y = {plate_xy[1]:.3f}.
    Constraints: x in [-0.35, 0.45]; y in [-0.72, -0.36] (robot must stay off the table).
    Current base_xy = {list(params.get('base_xy', [0.12, -0.40]))}

(2) "place_t" in [0, 1] — 0 = aim at plate centre, 1 = aim at plate edge toward the robot.
    TRADE-OFF (measured): decreasing place_t improves centring but INCREASES follow_max,
    because the plate centre is 0.63 m from the base — beyond the arm's reach.
    Current place_t = {params.get('place_t')}

Images: LEFT = simulator first-person render, RIGHT = real ego video (same control step):
1 approach, 2 grasp, 3 carry-high, 4 place, 5 end; then a TOP-VIEW plot.

Reply JSON only:
{{"param": "base_xy" | "place_t", "value": <float or [x,y]>, "reason": "<short>"}}
"""
    return user


def vlm_guide_v2(params, met, space, plate_xy=(0.18396, 0.22788)):
    """聚焦版指导：只允许 base_xy / place_t，并注入可达性先验与实测因果知识。"""
    from agentic_replay import vlm_call, parse_json
    prompt = build_prompt(params, met, plate_xy)
    try:
        act = parse_json(vlm_call(prompt, []))
    except Exception as e:
        print(f"  [VLM v2 指导失败] {e}", flush=True)
        return None
    key = act.get("param")
    if key not in ("base_xy", "place_t"):
        print(f"  [VLM v2] 建议的 {key!r} 不在允许集合 {{base_xy, place_t}} → 忽略", flush=True)
        return None
    val = act.get("value")
    if val is None:
        return None
    cand = dict(params)
    if key == "base_xy":
        b = np.asarray(val, dtype=float).reshape(-1)
        if b.size != 2:
            print("  [VLM v2] base_xy 需要两个数 → 忽略")
            return None
        cand["base_xy"] = [float(np.clip(b[0], -0.35, 0.45)),
                           float(np.clip(b[1], -0.72, -0.36))]
    else:
        cand["place_t"] = float(np.clip(float(val), 0.0, 1.0))
    print(f"  [VLM v2 指导] {key}: {params.get(key)} → {cand[key]}\n"
          f"              理由: {act.get('reason', '')}", flush=True)
    return cand
