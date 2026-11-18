#!/usr/bin/env python3
"""agentic_traj.py — 基于视频参考的轨迹平滑优化 (agent 的第二阶段用途)。

目标: 在不破坏抓取成功(verifier PASS)与跟随保真(follow_max 不退化)的前提下, 优化轨迹
参数让机械臂运动**更平滑**。核心手段:
  - time_scale 慢放/重定时: 把参考轨迹重采样到更密帧 → 逐控制步增量更小、更平滑
    (等价于"视频慢放"; 见 grasp 脚本 §慢放重定时)。
  - 平滑窗口 (ctrl_win/passes, joint_win/passes) 与运动学限幅 (max_qvel/qacc/qjerk)。

流程 (复现 Agentic Real2Sim 的 render→reason→adjust→re-render 闭环):
  1) 跑默认参数基线, 记录平滑度指标 + verdict。
  2) 对 time_scale 扫描 (慢放倍率), 再对平滑窗口/限幅做局部精修; 以**归一化平滑度代价**
     择优, 约束 = PASS 且 follow_max <= 基线 * 1.15。
  3) 生成「基线 vs 最优」关节速度/末端步长对比图。
  4) 交给本地 VLM (Qwen3.8-27B) 对照图像+数值复核: 运动是否更平滑、是否还需更大慢放;
     记录其理由 (可解释)。
  5) 输出最优参数 JSON, 回灌 grasp/dataset。

用法:
  # 先起本地 VLM 服务 (可选; 无则跳过第 4 步):
  #   CUDA_VISIBLE_DEVICES=2,5,6,7 ~/miniconda3/envs/opentouch/bin/python qwen_test/openai_server.py --port 18000 &
  CUDA_VISIBLE_DEVICES=0 python scripts/agentic_traj.py [--no-vlm]
"""
import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(os.path.dirname(_HERE), "results")
os.makedirs(_RESULTS, exist_ok=True)
sys.path.insert(0, _HERE)

import robotwin_r1_duck_grasp as G  # noqa: E402

# ---- 统一入口: 给定 --task 时所有 episode 走 pipeline.run_episode ----
TASK_CFG = None


def _run_episode(params, collect_frames=False):
    """所有 episode 的唯一通道。TASK_CFG 给定 -> pipeline 统一入口; 否则回退旧路径。"""
    if TASK_CFG is not None:
        import pipeline as PIPE
        ep, _ = PIPE.run_episode(TASK_CFG, params=params,
                                 collect_frames=collect_frames, setup=False)
        return ep
    return G.build_episode(on_step=None, collect_frames=collect_frames, params=params)


OUT_JSON = os.path.join(_RESULTS, "agentic_traj_best.json")
FIG_DIR = os.path.join(_HERE, "agentic_traj_frames")
os.makedirs(FIG_DIR, exist_ok=True)

ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:18000/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "Qwen3.8-27B")

# time_scale 候选 (慢放倍率) 与平滑精修候选
TS_CANDIDATES = [1.0, 1.5, 2.0, 3.0]
REFINE_CANDIDATES = [
    {},                                                        # 仅慢放
    {"joint_win": 9, "max_qjerk": 0.008},                      # 更大平滑窗 + 更紧 jerk
    {"max_qvel": 0.08, "max_qacc": 0.03, "max_qjerk": 0.006},  # 更紧速度/加速度/jerk
    {"ctrl_win": 9, "ctrl_passes": 3},                         # 更强控制点平滑
]


def _norm_cost(sm, ref):
    """归一化平滑度代价 (越小越平滑): 关节 jerk / 关节速度 / 末端步长 各占权重。"""
    return (sm["rms_jerk"] / max(ref["rms_jerk"], 1e-9)
            + sm["max_dq_deg"] / max(ref["max_dq_deg"], 1e-9)
            + 0.3 * sm["ee_max_step_mm"] / max(ref["ee_max_step_mm"], 1e-9))


def run(params, tag):
    t0 = time.time()
    try:
        ep = _run_episode(params, collect_frames=False)
    except Exception as e:
        print(f"[{tag}] 异常: {e}", flush=True)
        return None
    sm = ep["smooth"]
    v = ep["verdict"]
    follow = v["metrics"].get("follow_max_m", float("nan")) * 1000
    print(f"[{tag}] PASS={v['pass']} follow_max={follow:.1f}mm "
          f"max|Δq|={sm['max_dq_deg']:.3f}deg rms|jerk|={sm['rms_jerk']:.4f} "
          f"ee_step={sm['ee_max_step_mm']:.2f}mm 帧数={sm['n_frames']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return {"params": params, "smooth": sm, "verdict": v, "follow_max_mm": follow,
            "ts": float(params.get("time_scale", 1.0)),
            "xs": ep["xs"],
            "ee_step": np.linalg.norm(np.diff(ep["pad_trace"], axis=0), axis=1)}


def make_fig(base, best):
    """基线 vs 最优: 关节逐帧 max|Δq| 与 末端逐帧步长 对比图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # 关节速度 (用 smooth 指标无法还原曲线, 这里直接用 n_frames 占位需从 ep 取; 见主流程传 xs)
    for name, rec, col in (("baseline", base, "tab:red"), ("optimized", best, "tab:blue")):
        xs = rec.get("xs")
        if xs is None:
            continue
        dq = np.degrees(np.abs(np.diff(xs, axis=0))).max(axis=1)
        ax[0].plot(dq, color=col, label=f"{name} (max {dq.max():.2f}°)")
    ax[0].set_title("joint max |Δq| per control step"); ax[0].set_xlabel("k"); ax[0].legend()
    for name, rec, col in (("baseline", base, "tab:red"), ("optimized", best, "tab:blue")):
        ee = rec.get("ee_step")
        if ee is None:
            continue
        ax[1].plot(ee * 1000, color=col, label=f"{name} (max {ee.max()*1000:.1f}mm)")
    ax[1].set_title("EE displacement per control step (mm)"); ax[1].set_xlabel("k"); ax[1].legend()
    p = os.path.join(FIG_DIR, "smooth_compare.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    return p


def encode_img(path, max_side=900):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def vlm_review(fig_path, base, best):
    sys_p = ("You are a robotics trajectory-quality reviewer. You are shown joint-velocity and "
             "end-effector step-size profiles for two executions of the SAME grasp trajectory: "
             "red = baseline, blue = optimized (slower/retimed). Judge whether the blue motion is "
             "smoother and whether more slow-motion is needed. Reply JSON only.")
    user = (f"Metrics baseline: {json.dumps(base, ensure_ascii=False)}\n"
            f"Metrics optimized: {json.dumps(best, ensure_ascii=False)}\n\n"
            "IMAGE 1: left = joint max|Δq| per step, right = EE step-size per step.\n"
            "Answer JSON: {\"smoother\": true|false, \"need_more_slow\": true|false, "
            "\"suggested_time_scale\": <float>, \"reason\": \"<short>\"}")
    body = {"model": MODEL, "messages": [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": [{"type": "text", "text": user},
                                     {"type": "image_url", "image_url": {"url": encode_img(fig_path)}}]}],
        "max_tokens": 500, "temperature": 0.0}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    print(f"  [VLM] {text.strip()[:300]}", flush=True)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-vlm", action="store_true", help="跳过 VLM 复核")
    ap.add_argument("--task", default=None,
                    help="任务 JSON(如 configs/duck_to_plate.json)。给定时所有 episode 走 "
                         "pipeline.run_episode 统一入口。")
    args = ap.parse_args()

    global TASK_CFG
    if args.task:
        import pipeline as PIPE
        from task_config import load_task
        TASK_CFG = load_task(args.task)
        PIPE.apply_adapters(TASK_CFG, verbose=True)
        print(f"[task] 统一入口模式: {args.task}")

    base_params = dict(G.PARAM_DEFAULTS)
    base_params["time_scale"] = 1.0
    recs = []

    # 阶段 1: 基线
    base = run(base_params, "baseline")
    if base is None:
        print("基线运行失败, 终止"); return
    ref_sm = base["smooth"]
    recs.append(base)
    follow_ref = base["follow_max_mm"]

    def feasible(r):
        return (r is not None and r["verdict"]["pass"]
                and r["follow_max_mm"] <= follow_ref * 1.15)

    # 阶段 2a: time_scale 扫描 (仅慢放)
    ts_best = base
    for ts in TS_CANDIDATES:
        if abs(ts - 1.0) < 1e-6:
            continue
        p = dict(base_params); p["time_scale"] = float(ts)
        r = run(p, f"ts={ts}")
        if r is None:
            continue
        r["cost"] = _norm_cost(r["smooth"], ref_sm)
        recs.append(r)
        if feasible(r) and (not feasible(ts_best) or r["cost"] < ts_best.get("cost", 1e9)):
            ts_best = r
    print(f"\n[阶段2a] 最优慢放 time_scale={ts_best['ts']} "
          f"(PASS={ts_best['verdict']['pass']}, follow={ts_best['follow_max_mm']:.1f}mm)")

    # 阶段 2b: 在最优 time_scale 上精修平滑窗口/限幅
    best = ts_best
    for i, extra in enumerate(REFINE_CANDIDATES):
        if not extra:
            continue
        p = dict(base_params); p["time_scale"] = float(ts_best["ts"]); p.update(extra)
        r = run(p, f"refine{i}")
        if r is None:
            continue
        r["cost"] = _norm_cost(r["smooth"], ref_sm)
        recs.append(r)
        if feasible(r) and r["cost"] < best.get("cost", 1e9):
            best = r
            print(f"  -> 新最优 (cost={best['cost']:.3f}, {extra})")

    if "cost" not in best:
        best["cost"] = _norm_cost(best["smooth"], ref_sm)

    print(f"\n{'='*70}\n最优轨迹参数: time_scale={best['ts']} "
          f"cost={best['cost']:.3f} (基线 cost=1.0 相对单位)")
    print(f"  {json.dumps({k: best['params'].get(k) for k in ('time_scale','ctrl_win','joint_win','max_qvel','max_qacc','max_qjerk')}, ensure_ascii=False)}")
    print(f"  PASS={best['verdict']['pass']} follow_max={best['follow_max_mm']:.1f}mm "
          f"max|Δq|={best['smooth']['max_dq_deg']:.3f}deg rms|jerk|={best['smooth']['rms_jerk']:.4f} "
          f"ee_step={best['smooth']['ee_max_step_mm']:.2f}mm")

    # 阶段 3: 对比图 (直接复用 run() 缓存的 xs / ee_step 曲线, 无需重跑)
    fig_path = None
    try:
        fig_path = make_fig(base, best)
        print(f"对比图 -> {fig_path}")
    except Exception as e:
        print(f"对比图生成失败 (忽略): {e}")

    # 阶段 4: VLM 复核 (可解释)
    vlm_reply = None
    if not args.no_vlm and fig_path:
        try:
            bj = {"rms_jerk": base["smooth"]["rms_jerk"], "max_dq_deg": base["smooth"]["max_dq_deg"],
                  "ee_max_step_mm": base["smooth"]["ee_max_step_mm"], "time_scale": base["ts"]}
            oj = {"rms_jerk": best["smooth"]["rms_jerk"], "max_dq_deg": best["smooth"]["max_dq_deg"],
                  "ee_max_step_mm": best["smooth"]["ee_max_step_mm"], "time_scale": best["ts"]}
            vlm_reply = vlm_review(fig_path, bj, oj)
        except Exception as e:
            print(f"  [VLM 失败] {e} (忽略)")

    report = {
        "best_params": _jsonify(best["params"]),
        "best_metrics": best["smooth"],
        "best_follow_max_mm": best["follow_max_mm"],
        "baseline_metrics": base["smooth"],
        "baseline_follow_max_mm": follow_ref,
        "vlm_review": vlm_reply,
        "history": [{"params": _jsonify(r["params"]), "smooth": r["smooth"],
                     "pass": r["verdict"]["pass"], "follow_max_mm": r["follow_max_mm"],
                     "cost": r.get("cost")} for r in recs],
    }
    json.dump(report, open(OUT_JSON, "w"), ensure_ascii=False, indent=2, default=float)
    print(f"\n报告 -> {OUT_JSON}")


def _jsonify(p):
    """numpy -> 可 JSON 序列化 (数组转 list, 标量转 float/int)。"""
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


if __name__ == "__main__":
    main()
