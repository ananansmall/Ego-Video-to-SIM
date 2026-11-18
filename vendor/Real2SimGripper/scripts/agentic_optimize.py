#!/usr/bin/env python3
"""agentic_optimize.py — agent 驱动的未知参数优化（simulator-in-the-loop 搜索）。

设计要点
--------
1. **不修改任何已有脚本**: 只 import `robotwin_r1_duck_grasp` 并调用 `build_episode(params=...)`
   —— 与并行编辑者零冲突（本文件是新文件，不会覆盖别人的改动）。
2. **自动探测可用参数**: 搜索空间按 `G.PARAM_DEFAULTS` 实际存在的键过滤；
   万一某个参数被删（如 place_t），自动跳过而不是崩溃。
3. **结果缓存**: 每个参数组合的评测结果落盘缓存，中断后重跑不重复计算
   （一次评测 ~10-15 分钟，缓存很关键）。
4. **搜索策略**: 坐标下降 / 模式搜索（pattern search）——从种子出发，逐参数 ±step 扰动，
   有改进就移动并放大，无改进则收缩步长。比纯随机搜索样本效率高得多。
5. **VLM 可选**: 默认关闭（省 GPU）。加 `--vlm` 时把最优配置的对比图交给 VLM 复核并给建议。
6. **搜索期不渲染**: `collect_frames=False`，只在最后对最优配置渲染一次。

用法
----
  CUDA_VISIBLE_DEVICES=0 python scripts/agentic_optimize.py --budget 6
  CUDA_VISIBLE_DEVICES=0 python scripts/agentic_optimize.py --budget 6 --vlm     # 带 VLM 复核
  python scripts/agentic_optimize.py --dry-run                                    # 只看计划
"""
import argparse
import atexit
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(os.path.dirname(_HERE), "results")
os.makedirs(_RESULTS, exist_ok=True)
sys.path.insert(0, _HERE)
import robotwin_r1_duck_grasp as G  # noqa: E402

# 注意：**不**从 agentic_replay import metrics_of/score —— 那个文件正在被别人改写
# （曾经删掉 score 导致本脚本 ImportError）。这里自带一份，保证自给自足。
# 渲染/VLM 相关函数改成惰性 import，agentic_replay 坏了也不影响纯数值搜索。

OUT_JSON = os.path.join(_RESULTS, "agentic_optimize_best.json")
CACHE_JSON = os.path.join(_RESULTS, "agentic_optimize_cache.json")

BASE = {"plate_geom": "geometry_8", "base_xy": [0.12, -0.40],
        "time_scale": 3.0, "place_hover": 0.07, "place_t": 0.60,
        "grip_open_q": 0.05}      # COACD 下必须用满手指行程(0.05), 否则放不下

# 人类示范真值(归一化): 从真实视频提取的「鸭-盘」相对位置末段均值。
# 提取方法见 docs/05_示范真值与创新点.md §2(颜色分割 + 最大连通域 + 盘尺寸归一化)。
# 这是创新点 1 的核心: 物理 verifier 对"复现人类多少"没有分辨力(消融见 §8),
# 而这个量是整条管线上唯一免费的真值。
GT_NORM = np.array([-0.06, 0.06])

# ---------------- VLM 服务生命周期 ----------------
# 起停/显存释放统一由 vlm_lifecycle 管理（finally + atexit 双保险）
from vlm_lifecycle import VlmServer, gpu_mem_summary, ensure_vlm, release_vlm  # noqa: E402

_VLM = VlmServer()


# 搜索空间: (最小值, 最大值, 初始步长)。step 会被自适应缩放。
SPACE = {
    "place_t":     (0.0, 1.0, 0.10),
    "place_hover": (0.06, 0.18, 0.03),
    "time_scale":  (1.0, 3.0, 0.5),
    "open_f":      (84, 100, 4),
    "close_f":     (40, 60, 4),
    "d_hover":     (0.08, 0.20, 0.03),
    "grip_hold_q": (0.022, 0.034, 0.004),
    "grip_open_q": (0.045, 0.050, 0.005),   # prismatic 手指限位 [0, 0.05]
    # 基座位姿 —— **数值搜索原先没有这一项**, 而实测把 base_x 从 0.12 对齐到目标 x
    # 正是示范偏离 0.404→0.077 的主要来源。也就是说: 只有能建议 base_xy 的 agent
    # (VLM) 才发现得了这一步, 纯数值模式搜索结构上到不了。故必须放进空间。
    # 约束: 桌子占 x∈[-0.6,0.6], 桌腿在 y=±0.30±0.025 → 基座须 |y|>0.325 留在桌外
    "base_xy":     (-0.35, 0.45, 0.02),     # x 范围; y 范围见 BASE_XY_Y
}
BASE_XY_Y = (-0.72, -0.36)                  # 基座 y: 桌外且不撞桌腿
INTS = {"open_f", "close_f"}

# 已实测过的配置(来自 10-14 实跑), 预置进缓存避免重复计算 (省 ~45min)
SEED_RESULTS = [
    ({"place_t": 0.9, "time_scale": 2.0},
     {"pass": True, "reasons": [], "follow_max_mm": 76.8, "lift_mm": 116.5,
      "hold_slip_mm": 7.2, "knock_mm": 11.7,
      "plate_rb": [0.18396, 0.22788, 0.75264], "duck_final": [0.22173, 0.08151, 0.79279],
      "plate_offset_mm": 151.2, "plate_half_extent_mm": 177.36,
      "in_plate_box": False, "in_plate_circle": True, "plate_margin_mm": -21.4,
      "on_plate": False}),
    ({"place_t": 0.6, "time_scale": 2.0},
     {"pass": False, "reasons": ["follow_integrity"], "follow_max_mm": 105.4,
      "lift_mm": 123.0, "hold_slip_mm": 7.2, "knock_mm": 11.7,
      "plate_rb": [0.18396, 0.22788, 0.75264], "duck_final": [0.184, 0.1196, 0.7928],
      "plate_offset_mm": 88.5, "plate_half_extent_mm": 177.36,
      "in_plate_box": True, "in_plate_circle": True, "plate_margin_mm": 5.5,
      "on_plate": True}),
    ({"place_t": 0.6, "time_scale": 3.0},
     {"pass": False, "reasons": ["follow_integrity"], "follow_max_mm": 101.4,
      "lift_mm": 118.9, "hold_slip_mm": 11.5, "knock_mm": 11.7,
      "plate_rb": [0.18396, 0.22788, 0.75264], "duck_final": [0.184, 0.1322, 0.7928],
      "plate_offset_mm": 76.5, "plate_half_extent_mm": 177.36,
      "in_plate_box": True, "in_plate_circle": True, "plate_margin_mm": 17.5,
      "on_plate": True}),
]


# ---------- 参数规范化与缓存 ----------
# 参与缓存 key 的规范化键集合。**只放会影响仿真结果的项**。
# 修复动机(2026-10-20): 之前 canon 把 params 里出现的所有键都算进哈希, 于是
# "显式写 ctrl_win=5(等于默认)" 与 "省略 ctrl_win" 哈希不同 → 缓存永不命中,
# 实测导致 agent 每种一个已知配置都要白跑一次 20 分钟。
KEY_PARAMS = ("base_xy", "duck_rb_xy", "d_hover", "grasp_orient", "grasp_tilt_max",
              "grasp_fin", "grasp_carry_blend", "grip_open_q", "grip_hold_q",
              "close_f", "open_f", "ctrl_win", "ctrl_passes", "joint_win",
              "joint_passes", "max_qvel", "max_qacc", "max_qjerk",
              "time_scale", "plate_geom", "place_hover", "place_t", "ik_seeds")


def canon(params, keys=None):
    """把参数规范成可比较/可哈希的形式。

    **以"有效参数"为准**：缺失的键用核心脚本默认补齐，因此
    「显式传默认值」与「省略」得到同一个 hash。
    """
    src = {**getattr(G, "PARAM_DEFAULTS", {}),
           **((TASK_CFG or {}).get("tuning", {})), **(params or {})}
    out = {}
    for k in (KEY_PARAMS if keys is None else keys):
        if k not in src:
            continue
        v = src[k]
        if isinstance(v, (list, tuple, np.ndarray)):
            v = [round(float(x), 6) for x in v]
        elif isinstance(v, np.floating):
            v = round(float(v), 6)
        elif isinstance(v, float):
            v = round(v, 6)
        elif isinstance(v, (int, np.integer)):
            v = int(v)
        out[k] = v
    return out


# 当前机器人名 —— **必须进缓存 key**, 否则换机器人后会误命中别的机器人的旧结果
ROBOT_NAME = "r1"
# 碰撞设置标签 —— 同理必须进 key, 否则 "单凸壳" 与 "COACD" 的结果会互相污染
COLLISION_TAG = "hull1"
# 核心脚本代码指纹 —— 核心脚本被重构后参数→行为的映射会变, 旧缓存不再可比。
# 求解器、管线与三个 adapter 的内容共同决定缓存身份。
def _code_fingerprint():
    digest = hashlib.sha256()
    for name in ("robotwin_r1_duck_grasp.py", "pipeline.py", "task_config.py",
                 "robot_config.py", "collision_config.py"):
        digest.update(name.encode())
        with open(os.path.join(_HERE, name), "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:16]


CODE_TAG = _code_fingerprint()
SEED_CODE_TAG = None


def key_of(params):
    from collision_config import state
    from task_config import _strip_comments
    c = canon(params, keys=G.PARAM_DEFAULTS)
    c["__robot__"] = ROBOT_NAME
    c["__collision__"] = COLLISION_TAG
    # Include task/GT context, excluding comments and the JSON file's location.
    task = _strip_comments(TASK_CFG or {})
    col = task.get("sim", {}).get("collision", {})
    if col.get("mode") == "coacd":
        col["threshold"] = 0.05 if col.get("threshold") is None else float(col["threshold"])
        col.setdefault("max_depenetration_velocity", 1.0)
    c["__task__"] = task
    if COLLISION_TAG == "coacd":
        actual = state()
        c["__collision_options__"] = {
            "threshold": 0.05 if actual.get("threshold") is None else actual["threshold"],
            "max_depenetration_velocity": actual.get("max_depenetration_velocity", 1.0),
        }
    c["__code__"] = CODE_TAG
    s = json.dumps(c, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def cache_matches(key, record):
    """History/best selection must use both current code and current IK budget."""
    params = record["params"]
    return (canon(params)["ik_seeds"] == canon({})["ik_seeds"]
            and key == key_of(params))


def clamp(key, val):
    """把参数夹到合法范围。base_xy 是二维(x,y), 需分别夹。"""
    if key == "base_xy":
        b = np.asarray(val, dtype=float).reshape(-1)
        if b.size != 2:
            raise ValueError("base_xy 需要两个数")
        xlo, xhi, _ = SPACE["base_xy"]
        ylo, yhi = BASE_XY_Y
        return [float(np.clip(b[0], xlo, xhi)), float(np.clip(b[1], ylo, yhi))]
    lo, hi, _ = SPACE[key]
    if key in INTS:
        return int(np.clip(round(float(val)), lo, hi))
    return float(np.clip(float(val), lo, hi))


# ---------- 指标与评分（自带，不依赖 agentic_replay） ----------
# ---------------------------------------------------------------- 统一入口
# --task 给定时, 本 agent 不再自己 setup adapter, 而是全部走 pipeline.run_episode;
# 未给定时回退旧路径(向后兼容)。
TASK_CFG = None

def _run(params, collect_frames=False):
    """**所有 episode 都从这里跑**。返回 (ep, met)。"""
    if TASK_CFG is not None:
        import pipeline as PIPE
        return PIPE.run_episode(TASK_CFG, params=params,
                                collect_frames=collect_frames, setup=False)
    ep = G.build_episode(on_step=None, collect_frames=collect_frames, params=params)
    return ep, metrics_of(ep)


def metrics_of(ep):
    """从一次 episode 抽调参要用的指标。"""
    v = ep["verdict"]
    m = dict(v["metrics"])
    duck = ep["duck_trace"]
    out = {
        "pass": bool(v["pass"]), "reasons": list(v["reasons"]),
        "follow_max_mm": float(m.get("follow_max_m", float("nan"))) * 1000.0,
        "lift_mm": float(m["lift_m"]) * 1000.0,
        "hold_slip_mm": float(m["hold_slip_m"]) * 1000.0,
        "knock_mm": float(m["knock_m"]) * 1000.0,
    }
    if ep.get("plate_rb") is not None:
        pr = np.asarray(ep["plate_rb"], dtype=float)
        ext = np.asarray(ep["plate_ext"], dtype=float)
        d = duck[-1, :2] - pr[:2]
        out["plate_rb"] = pr.tolist()
        out["duck_final"] = duck[-1].tolist()
        out["plate_offset_mm"] = float(np.linalg.norm(d)) * 1000.0
        out["plate_half_extent_mm"] = float(np.linalg.norm(ext[:2]) / 2) * 1000.0
        # 逐轴包围盒判据(严格) —— 对角距离会把偏出盘边的情况误判成在盘上
        out["in_plate_box"] = bool(abs(d[0]) <= ext[0] / 2 and abs(d[1]) <= ext[1] / 2)
        out["in_plate_circle"] = bool(np.linalg.norm(d) <= np.linalg.norm(ext[:2]) / 2)
        out["on_plate"] = out["in_plate_box"]
        out["plate_margin_mm"] = float(min(ext[0] / 2 - abs(d[0]),
                                           ext[1] / 2 - abs(d[1]))) * 1000.0
        # ---- 示范保真度(创新点 1 的核心度量) ----
        # 人类真值: 从真实视频 7.mp4 颜色分割提取的「鸭-盘」归一化相对位置末段均值,
        # 多组阈值/帧范围稳健(x∈[-0.11,-0.04], y∈[+0.02,+0.11]), 详见 docs/05。
        # 该量无量纲、与场景无关(白桌/红床/道具/相机位姿都不影响), 故可跨 真实视频↔仿真 比较,
        # 也顺带绕开了上游 3D 重建在物体末态上 ~13cm 的漂移。
        off = d / np.maximum(ext[:2], 1e-9)
        out["gt_dist"] = float(np.linalg.norm(off - GT_NORM))
        out["gt_off"] = [float(off[0]), float(off[1])]
        out["gt_y_cm"] = float(abs(off[1] - GT_NORM[1]) * ext[1] * 100.0)
        out["gt_x_cm"] = float(abs(off[0] - GT_NORM[0]) * ext[0] * 100.0)
    # released_on_table 对"放到盘子上"是假阴性(盘面比桌面高), 修正出 pass_eff
    reasons_eff = [r for r in v["reasons"] if r != "released_on_table"]
    out["reasons_eff"] = reasons_eff
    out["pass_eff"] = bool(v["pass"]) or (len(reasons_eff) == 0 and out.get("on_plate") is True)
    return out


def score(met):
    """越大越好: 通过/落盘优先, 其次**示范保真度**, 再其次跟得准/不滑脱/没打飞。

    示范偏离(gt_dist)是主排序项: 物理 verifier 对它没有分辨力(ρ=-0.335、分布重叠,
    见 docs/05 §8 消融), 而它是整条管线上唯一免费的真值。权重 800 使 0.4 的偏离
    ≈320 分, 与其它项尺度可比, 不会把"通过"压掉。
    """
    s = 0.0
    s += 1000.0 if met.get("pass_eff", met.get("pass")) else 0.0
    s += 800.0 if met.get("on_plate") else 0.0
    s -= met.get("gt_dist", 0.0) * 800.0          # ← 示范保真度(主)
    s -= met.get("plate_offset_mm", 0.0) * 1.0    # 落点离盘心(次)
    # 贴边(余量 < 1cm)要罚, 否则会贴着盘边停
    if met.get("plate_margin_mm") is not None and met["plate_margin_mm"] < 10.0:
        s -= (10.0 - met["plate_margin_mm"]) * 5.0
    s -= met.get("follow_max_mm", 0.0) * 1.0
    s -= met.get("hold_slip_mm", 0.0) * 1.0
    s -= met.get("knock_mm", 0.0) * 2.0
    return s


def gt_dist_of(met):
    """从缓存条目指标反算示范偏离(兼容没有 gt_dist 字段的老条目)。"""
    if met.get("gt_dist") is not None:
        return float(met["gt_dist"])
    df, pr = met.get("duck_final"), met.get("plate_rb")
    if df is None or pr is None:
        return None
    off = (np.asarray(df, float)[:2] - np.asarray(pr, float)[:2]) / np.array([0.2504, 0.2513])
    return float(np.linalg.norm(off - GT_NORM))


def _jsonify(p):
    out = {}
    for k, v in p.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def supported_space():
    """只保留核心脚本实际支持的参数(防止别人删了某个 key 导致崩溃)。"""
    avail = set(G.PARAM_DEFAULTS.keys())
    return {k: v for k, v in SPACE.items() if k in avail}


# ---------- 评测 ----------
def _f(x, nd=1):
    """安全格式化：None/缺失返回 'n/a'（异常条目的指标可能缺失）。"""
    try:
        if x is None:
            return "n/a"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "n/a"


def _is_exc(rec):
    """该缓存条目是否为"评测异常"记录（没有真实指标，不能当有效结果）。"""
    if not rec:
        return True
    s = rec.get("score")
    if s is None or s <= -1e8:
        return True
    m = rec.get("metrics") or {}
    return "exception" in (m.get("reasons") or []) or not m.get("follow_max_mm")


def evaluate(params, cache, tag):
    k = key_of(params)
    if k in cache and not _is_exc(cache[k]):
        rec = cache[k]
        print(f"  [缓存命中] score={_f(rec.get('score'))} "
              f"follow={_f(rec['metrics'].get('follow_max_mm'))} "
              f"offset={_f(rec['metrics'].get('plate_offset_mm'))} "
              f"pass={rec['metrics'].get('pass')}")
        return rec
    if k in cache:
        print("  [缓存是异常记录] 重新评测")
    print(f"  [评测] {json.dumps(_jsonify(params), ensure_ascii=False)}", flush=True)
    t0 = time.time()
    try:
        ep, met = _run(params, collect_frames=False)
    except Exception as e:
        print(f"    !! 评测异常: {e}", flush=True)
        rec = {"params": _jsonify(params), "metrics": {"pass": False, "reasons": ["exception"]},
               "score": -1e9, "error": str(e), "robot": ROBOT_NAME,
               "collision": COLLISION_TAG}
        cache[k] = rec
        save_cache(cache)
        return rec
    sc = score(met)
    rec = {"params": _jsonify(params), "metrics": met, "score": sc,
           "secs": round(time.time() - t0), "tag": tag, "robot": ROBOT_NAME,
           "collision": COLLISION_TAG}
    cache[k] = rec
    save_cache(cache)
    print(f"    -> score={_f(sc)} pass={met['pass']} follow={_f(met.get('follow_max_mm'))}mm "
          f"offset={_f(met.get('plate_offset_mm'))}mm on_plate={met.get('on_plate')} "
          f"margin={_f(met.get('plate_margin_mm'))} ({rec['secs']}s)", flush=True)
    return rec


def save_cache(cache):
    with open(CACHE_JSON, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1, default=float)


def pass_eff_of(met):
    """兼容老缓存条目: 它们没有 pass_eff 字段, 按同样规则现算。"""
    if "pass_eff" in met:
        return bool(met["pass_eff"])
    reasons = [r for r in met.get("reasons", []) if r != "released_on_table"]
    return bool(met.get("pass")) or (len(reasons) == 0 and met.get("on_plate") is True)


def rescore(cache):
    """评分函数改版后, 用新函数重算所有历史条目的分数(无需重跑仿真)。
    同时**回填 robot 字段**: 早期条目没有该字段, 会被 best_cached 的机器人过滤
    误跳过(曾导致 hover=0.07 的 1414 分最优被漏掉, 搜索从 434 分的差点出发)。"""
    n = 0
    for v in cache.values():
        if "robot" not in v:
            v["robot"] = "r1"        # 早期条目全是 r1 跑的
            n += 1
        if "collision" not in v:
            v["collision"] = "hull1"  # 早期条目都是默认 decomposition="none"
            n += 1
        met = dict(v.get("metrics") or {})
        if not met:
            continue
        met["pass_eff"] = pass_eff_of(met)
        # 补算示范偏离(老条目没有该字段; 新 score 以它为主排序项)
        if met.get("gt_dist") is None:
            gd = gt_dist_of(met)
            if gd is not None:
                met["gt_dist"] = gd
                n += 1
        new = score(met)
        if abs(new - (v.get("score") or 0)) > 1e-6:
            v["score"] = new
            n += 1
        v["metrics"] = met
    if n:
        save_cache(cache)
    return n


def load_cache():
    if os.path.exists(CACHE_JSON):
        try:
            return json.load(open(CACHE_JSON))
        except Exception as e:
            print(f"[warn] 缓存损坏({e}), 重建")
    return {}


# ---------- 搜索 ----------
def perturb(base, key, sign, step_scale, space):
    """对单个参数做 ±step 扰动。base_xy 是二维, x/y 用不同步长与范围。"""
    p = dict(base)
    if key == "base_xy":
        lo, hi, st = space[key]
        cur = np.asarray(base.get(key, [0.12, -0.40]), dtype=float)
        # x 用 st, y 用更大步长(它的可动范围更窄, 且对可达性影响更直接)
        nxt = cur + sign * np.array([st, st * 1.5])
        p[key] = clamp("base_xy", nxt)
        return p
    lo, hi, st = space[key]
    step = st * step_scale
    p[key] = clamp(key, float(p[key]) + sign * step)
    return p


def best_cached(cache):
    """缓存里**当前机器人**已知最优的配置(用于决定搜索起点)。
    必须按机器人过滤 —— 否则会把 r1 的结果当 r1_lite 的起点。"""
    best = None
    for k, v in cache.items():
        if not cache_matches(k, v):
            continue
        if v.get("robot") != ROBOT_NAME:
            continue
        if v.get("collision", "hull1") != COLLISION_TAG:
            continue
        if _is_exc(v):
            continue
        s = v.get("score")
        if s is None:
            continue
        if best is None or s > best["score"]:
            best = v
    return best


def vlm_guide(params, met, space):
    """**VLM 进入搜索循环**：渲染「sim FPV | 原视频」对比图 + 数值，让 VLM 决定
    下一步该动哪个参数、往哪动。返回一个候选 params（或 None）。"""
    from agentic_replay import load_video, make_panels, vlm_call, parse_json
    video = load_video(os.environ.get("REAL_VIDEO", "/path/to/workspace/7.mp4"))
    ep, _met_g = _run(params, collect_frames=True)
    panels = make_panels(ep, video, "guide")
    met = metrics_of(ep)
    bounds = {k: list(v[:2]) for k, v in space.items()}
    user = (
        "This is a real-to-sim replay: a robot gripper must pick a pink duck and PLACE it on "
        "a BLUE plate. You are the search driver.\n"
        f"Current parameters: {json.dumps(_jsonify(params), ensure_ascii=False)}\n"
        f"Metrics: {json.dumps(met, ensure_ascii=False)}\n"
        f"Parameter bounds: {json.dumps(bounds, ensure_ascii=False)}\n\n"
        "Images: LEFT = simulator first-person render, RIGHT = real ego video (same control "
        "step): 1 approach, 2 grasp, 3 carry-high, 4 place, 5 end; then a TOP-VIEW plot.\n\n"
        "Decide the SINGLE most useful parameter change to make the replay match reality:\n"
        "- if the duck is off the plate or at its edge → change place_t (0=plate centre, "
        "1=plate edge toward robot) or place_hover\n"
        "- if the arm lags / does not reach → change time_scale (slower = better tracking) "
        "or base_xy\n"
        "- if it releases too early/late → change open_f\n"
        "Reply JSON only: "
        '{"param": "<key>", "value": <number or [x,y]>, "reason": "<short>"}'
    )
    try:
        act = parse_json(vlm_call(user, panels))
    except Exception as e:
        print(f"  [VLM 指导失败] {e}", flush=True)
        return None
    key = act.get("param")
    if key not in space:
        print(f"  [VLM] 建议的 {key!r} 不在搜索空间 {list(space)} → 忽略", flush=True)
        return None
    val = act.get("value")
    if val is None:
        return None
    cand = dict(params)
    if key == "base_xy":
        b = np.asarray(val, dtype=float).reshape(-1)
        if b.size != 2:
            print("  [VLM] base_xy 需要两个数 → 忽略")
            return None
        cand["base_xy"] = clamp("base_xy", b)      # x/y 各有各的范围(见 BASE_XY_Y)
    else:
        cand[key] = clamp(key, float(val))
    if abs(float(cand[key]) - float(params.get(key, cand[key]))) < 1e-9:
        print(f"  [VLM] 建议与当前相同（{key}={cand[key]}）→ 忽略", flush=True)
        return None
    print(f"  [VLM 指导] {key}: {params.get(key)} → {cand[key]}  "
          f"理由: {act.get('reason', '')}", flush=True)
    return cand


def pattern_search(seed_params, cache, space, budget, guide=None, guide_every=2,
                   from_seed=False):
    """坐标下降/模式搜索: 逐参数 ±step 扰动, 有改进就移动并放大, 无改进收缩步长。
    guide: 可选回调 guide(params, met, space) -> 候选 params 或 None（VLM 进循环）"""
    best = evaluate(seed_params, cache, "seed")
    # 若请求的种子已被评测过且不如缓存里的最优配置, 改从已知最优出发 ——
    # 否则会从一个坏点开始瞎走(实测 place_t=0.7/ts=3.0 就是个坏点)。
    bc = None if from_seed else best_cached(cache)
    if bc is not None and bc["score"] > best["score"] + 1e-9:
        print(f"  [改起点] 缓存里有更优配置 score={_f(bc.get('score'))} > 种子 {best['score']:.1f}, "
              f"改从它出发", flush=True)
        best = evaluate(bc["params"], cache, "cache-best")
    history = [best]
    scale = 1.0
    n_eval = 0
    last_guide = -1          # 上次 VLM 指导时的 n_eval（-1 表示还没指导过）

    while n_eval < budget:
        improved = False
        for key in space:
            if n_eval >= budget:
                break
            for sign in (+1, -1):
                if n_eval >= budget:
                    break
                # ---- VLM 进循环: 每次评测前检查间隔 ----
                # 必须放在**最内层**。放在 while 顶部/末尾都不行: 内层一轮参数扫描
                # (7 参数 × 2 方向 = 14 个扰动) 会把 budget 一次用光, 指导永远轮不到。
                if guide is not None and n_eval - last_guide >= max(1, guide_every):
                    last_guide = n_eval
                    gc = guide(best["params"], best["metrics"], space)
                    if gc is not None:
                        if key_of(gc) in cache:
                            print("  [VLM] 建议的配置已缓存 → 跳过")
                        else:
                            grec = evaluate(gc, cache, "vlm")
                            n_eval += 1
                            history.append(grec)
                            if grec["score"] > best["score"] + 1e-9:
                                print(f"    *** VLM 建议带来改进 "
                                      f"(score {_f(grec['score'])})", flush=True)
                                best = grec
                    if n_eval >= budget:
                        break
                cand = perturb(best["params"], key, sign, scale, space)
                ck = key_of(cand)
                if ck == key_of(best["params"]):
                    continue          # 被 clamp 到原地, 跳过
                rec = evaluate(cand, cache, f"{key}{'+' if sign>0 else '-'}")
                n_eval += 1
                history.append(rec)
                if rec["score"] > best["score"] + 1e-9:
                    print(f"    *** 改进: {key} {best['params'].get(key)} -> {cand[key]} "
                          f"(score {best['score']:.1f} -> {rec['score']:.1f})", flush=True)
                    best = rec
                    improved = True
                    scale = min(scale * 1.3, 2.0)   # 有改进 -> 放大步长
                    break                             # 该参数方向已找到, 换下一个
        if not improved:
            scale *= 0.5
            print(f"  [收缩步长] scale={scale:.3f}", flush=True)
            if scale < 0.12:
                print("  [收敛] 步长过小, 停止", flush=True)
                break
    return best, history


# ---------- VLM 复核(可选) ----------
def vlm_review(params, met):
    from agentic_replay import load_video, make_panels, vlm_audit
    video = load_video(os.environ.get("REAL_VIDEO", "/path/to/workspace/7.mp4"))
    ep, _met_g = _run(params, collect_frames=True)
    panels = make_panels(ep, video, "opt")
    return vlm_audit(metrics_of(ep), panels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=6, help="额外评测次数上限")
    ap.add_argument("--seed-t", type=float, default=0.70)
    ap.add_argument("--seed-ts", type=float, default=3.0)
    ap.add_argument("--base-xy", type=float, nargs=2, default=[0.12, -0.40],
                    help="机器人基座 xy（换机器人后需要重扫）")
    ap.add_argument("--task", default=None,
                    help="任务 JSON 路径(如 configs/duck_to_plate.json)。给定时所有 "
                         "episode 走 pipeline.run_episode 统一入口, 忽略 --coacd/--base-xy 等旧开关。")
    ap.add_argument("--vlm", action="store_true",
                    help="启用 VLM：既在搜索循环内指导调参，也在结束后复核")
    ap.add_argument("--guide-every", type=int, default=2,
                    help="每评测 N 次就让 VLM 指导一次（越小越像 agent，但越慢）")
    ap.add_argument("--robot", default="r1",
                    help="机器人（r1 / r1_lite；r1_pro 是 7-DoF 本仓不支持）")
    ap.add_argument("--coacd", action="store_true",
                    help="启用 COACD 凸分解（碰撞体由 1 个凸壳 → 多个凸包）")
    ap.add_argument("--coacd-threshold", type=float, default=0.05,
                    help="COACD 阈值：越小凸包越多(0.05→约8~10个, 0.02→约45个)")
    ap.add_argument("--vlm-v2", action="store_true",
                    help="用聚焦版 VLM 指导(只动 base_xy/place_t, 注入可达性先验与因果知识)。"
                         "v1 给了 9 个参数+10 个无优先级指标+一条错规则, 实测失败。")
    ap.add_argument("--from-seed", action="store_true",
                    help="强制从 --seed-* 指定的种子出发(不自动跳缓存最优)。"
                         "用于 A/B 对照: 想看 agent 从某个已知差点能走到哪。")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划, 不跑")
    args = ap.parse_args()

    # ---- 碰撞几何：COACD（猴子补丁，不改核心脚本）----
    global COLLISION_TAG
    if args.coacd:
        try:
            from collision_config import enable_coacd
            enable_coacd(G, args.coacd_threshold)
            # threshold 已证无效(见 collision_config docstring), 故 tag 不带它
            COLLISION_TAG = "coacd"
        except Exception as e:
            print(f"[collision_config] 启用 COACD 失败: {e}")

    # ---- 换机器人(猴子补丁, 不改核心脚本) ----
    try:
        from robot_config import apply_robot, inspect_robot
        rep = inspect_robot(args.robot)
        print(f"机器人 {args.robot}: 左臂 {rep.get('arm_dof')}-DoF, "
              f"命名符合={rep.get('naming_ok')}, 可运行={rep.get('runnable')}")
        if not rep.get("runnable"):
            print(f"  ⚠️ {rep.get('reasons')} → 仍尝试运行（若崩了请换 r1/r1_lite）")
        apply_robot(G, args.robot, overrides={"base_xy": list(args.base_xy)})
        global ROBOT_NAME
        ROBOT_NAME = args.robot
    except Exception as e:
        print(f"[robot_config] 跳过机器人切换: {e}")

    global TASK_CFG
    if args.task:
        import pipeline as PIPE
        from task_config import load_task
        TASK_CFG = load_task(args.task)
        PIPE.apply_adapters(TASK_CFG, verbose=True)
        # **必须在打补丁之后**再读碰撞状态 —— 否则 tag 仍是默认 "hull1",
        # best_cached 会把 coacd 的历史最优全部过滤掉(实测导致漏掉 1651 分的配置)。
        from collision_config import state as _cs
        COLLISION_TAG = "coacd" if _cs().get("enabled") else "hull1"
        BASE["plate_geom"] = (TASK_CFG.get("target") or {}).get("geom")
        BASE["base_xy"] = list(TASK_CFG["robot"]["base_xy"])
        print(f"[task] 统一入口模式: {args.task}  (base_xy={BASE['base_xy']}, "
              f"plate={BASE['plate_geom']})")
    space = supported_space()
    missing = set(SPACE) - set(space)
    print("=" * 74)
    print("可用搜索空间 (核心脚本实际支持的):")
    for k, (lo, hi, st) in space.items():
        print(f"  {k:<14} [{lo}, {hi}] step={st}")
    if missing:
        print(f"  跳过(核心脚本不支持): {sorted(missing)}")

    cache = load_cache()
    # 预置已实测结果, 避免重复计算
    # 预置结果是旧代码 + 单凸壳下测的; 仅当代码指纹/碰撞设置都匹配当前时才注入
    if CODE_TAG == SEED_CODE_TAG and COLLISION_TAG == "hull1":
      for extra, met in SEED_RESULTS:
        p = dict(BASE); p.update(extra)
        k = key_of(p)
        if k not in cache:
            cache[k] = {"params": _jsonify(p), "metrics": met, "score": score(met),
                        "tag": "seeded", "secs": None, "robot": ROBOT_NAME,
                        "collision": COLLISION_TAG}
    n_rs = rescore(cache)
    if n_rs:
        print(f"重打分条目 = {n_rs} (评分函数改版)")
    print(f"缓存条目 = {len(cache)} (含预置已实测结果)")
    bc = best_cached(cache)
    if bc:
        print(f"已知最优: place_t={bc['params'].get('place_t')} ts={bc['params'].get('time_scale')} "
              f"hover={bc['params'].get('place_hover')} score={_f(bc.get('score'))} "
              f"follow={_f(bc['metrics'].get('follow_max_mm'))} "
              f"offset={_f(bc['metrics'].get('plate_offset_mm'))} "
              f"pass_eff={bc['metrics'].get('pass_eff')}")

    seed = dict(BASE); seed["place_t"] = args.seed_t; seed["time_scale"] = args.seed_ts
    print(f"\n种子配置: {json.dumps(_jsonify(seed), ensure_ascii=False)}")
    if args.dry_run:
        print("(dry-run, 不评测)"); return

    # ---- VLM: agent 自己起、**finally 里一定停** ----
    guide = None
    if args.vlm:
        ok = ensure_vlm()
        if ok:
            if getattr(args, "vlm_v2", False):
                from vlm_guide_v2 import vlm_guide_v2
                guide = lambda pm, mt, sp: vlm_guide_v2(pm, mt, sp)
                print("[VLM] 使用**聚焦版 v2** 指导（只动 base_xy/place_t，含可达性先验）")
            else:
                guide = vlm_guide
            print(f"[VLM] 已进入搜索循环，每 {args.guide_every} 次评测指导一次")
        else:
            print("[VLM] 服务起不来 → 退化为纯数值搜索")

    t0 = time.time()
    vlm = None
    try:
        best, history = pattern_search(seed, cache, space, args.budget,
                                       guide=guide, guide_every=args.guide_every,
                                       from_seed=args.from_seed)
        # 最终复核要在停 VLM **之前**做
        if args.vlm and _VLM.is_up():
            try:
                vlm = vlm_review(best["params"], best["metrics"])
            except Exception as e:
                print(f"  [VLM 失败] {e}")
    finally:
        # 无论搜索成功/异常/被中断, 都释放显存
        if args.vlm:
            print("\n[VLM] 结束，释放显存 …", flush=True)
            release_vlm()
    print(f"\n{'='*74}\n搜索完成, 用时 {(time.time()-t0)/60:.1f} min, 评测 {len(history)} 次")

    report = {"best_params": best["params"], "best_metrics": best["metrics"],
              "best_score": best["score"], "budget": args.budget,
              "n_evals": len(history), "vlm_review": vlm,
              "history": [{k2: h[k2] for k2 in ("params", "score", "metrics", "tag")} for h in history]}
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)

    print(f"\n最优: {json.dumps(best['params'], ensure_ascii=False)}")
    m = best["metrics"]
    print(f"  score={best['score']:.1f}  pass={m.get('pass')}  "
          f"follow={m.get('follow_max_mm'):.1f}mm  offset={m.get('plate_offset_mm'):.1f}mm  "
          f"on_plate={m.get('on_plate')}  margin={m.get('plate_margin_mm')}")
    print(f"报告 -> {OUT_JSON}\n缓存 -> {CACHE_JSON}")


if __name__ == "__main__":
    main()
