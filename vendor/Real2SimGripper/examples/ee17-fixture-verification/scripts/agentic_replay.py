#!/usr/bin/env python3
"""agentic_replay.py — VLM 驱动的「原视频对照审计」(纯审计, 不调参)。

思路: 管线此前的 agent 闭环都只看**仿真自己的**渲染/数值; 现在我们手上有原始 ego
视频 (7.mp4, 113 帧), 于是把闭环升级成:
    仿真 FPV 帧  ↔  原始视频同帧  逐帧并排 → 连同物理指标一起交给本地 VLM
    VLM 判「场景对不对 / 抓得像不像 / 跟得准不准 / 落点对不对」 → 输出保真度判定

**纯对照审计**: 只做视觉保真度判定, 不做任何参数调整 (放盘调参线已移除)。

产物:
  - 并排对比图 (sim FPV | 原视频) @ 关键帧 (接近 / 抓取 / 搬运高点 / 放置 / 结束)
  - 俯视数值图 (鸭子轨迹 vs 盘子 vs pad 轨迹)
  - 逐帧并排重放视频 (sim FPV | 原视频, replay_sbs.mp4)
  - report JSON (物理指标 + VLM 判定)

用法:
  # 先起 VLM (OpenAI 兼容端点 :18000), 再跑:
  CUDA_VISIBLE_DEVICES=0 python scripts/agentic_replay.py
  # 无 VLM 时只出对照图 + 重放视频(不做视觉判定):
  CUDA_VISIBLE_DEVICES=0 python scripts/agentic_replay.py --no-vlm
  # 默认纯统一跟随; 要带放盘修正则显式指定盘子:
  CUDA_VISIBLE_DEVICES=0 python scripts/agentic_replay.py --plate geometry_8
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


def _run_episode(params, collect_frames=True):
    """所有 episode 的唯一通道。TASK_CFG 给定 -> pipeline 统一入口; 否则回退旧路径。"""
    if TASK_CFG is not None:
        import pipeline as PIPE
        ep, _ = PIPE.run_episode(TASK_CFG, params=params,
                                 collect_frames=collect_frames, setup=False)
        return ep
    return G.build_episode(on_step=None, collect_frames=collect_frames, params=params)


REAL_VIDEO = os.environ.get("REAL_VIDEO", "/path/to/workspace/7.mp4")
ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:18000/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "Qwen3.8-27B")
OUT_DIR = os.path.join(_HERE, "agentic_replay_frames")
OUT_JSON = os.path.join(_RESULTS, "agentic_replay_best.json")
os.makedirs(OUT_DIR, exist_ok=True)

SYS_PROMPT = (
    "You are a real-to-sim fidelity auditor for a Galaxea R1 dual-arm robot replaying a "
    "human egocentric grasp demo in a physics simulator. You are shown, for the SAME control "
    "step: (IMAGE 1) the simulator's first-person render reproducing the real ego camera, and "
    "(IMAGE 2) the corresponding frame from the real egocentric video. Judge whether the "
    "replay matches reality: is the object grasped, lifted, and finally placed ON the plate? "
    "Also read the numeric metrics. Reply with ONE JSON object only."
)


# ---------- 原视频读取 ----------
def load_video(path):
    import av
    c = av.open(path)
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    return np.stack(fr)


def encode_img(path, max_side=900):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def vlm_call(user_text, image_paths, max_tokens=700):
    content = [{"type": "text", "text": user_text}]
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": encode_img(p)}})
    body = {"model": MODEL, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": SYS_PROMPT},
                         {"role": "user", "content": content}]}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    print(f"  [VLM] {round(time.time()-t0,1)}s -> {text.strip()[:260]}", flush=True)
    return text


def parse_json(text):
    dec = json.JSONDecoder()
    s = text.strip()
    i = s.find("```json")
    if i != -1:
        j = s.find("```", i + 7)
        if j != -1:
            return json.loads(s[i + 7:j].strip())
    try:
        return json.loads(s.strip("`").strip())
    except Exception:
        pass
    for i, ch in enumerate(s):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(s[i:])
                return obj
            except Exception:
                continue
    raise ValueError(f"无法解析 VLM 输出: {text[:300]}")


# ---------- 指标 ----------
def metrics_of(ep):
    """从一次 episode 抽回放闭环要用的指标。"""
    m = dict(ep["verdict"]["metrics"])
    v = ep["verdict"]
    duck = ep["duck_trace"]
    out = {
        "pass": bool(v["pass"]), "reasons": v["reasons"],
        "follow_max_mm": float(v["metrics"].get("follow_max_m", float("nan"))) * 1000.0,
        "lift_mm": float(m["lift_m"]) * 1000.0,
        "hold_slip_mm": float(m["hold_slip_m"]) * 1000.0,
        "knock_mm": float(m["knock_m"]) * 1000.0,
    }
    if ep.get("plate_rb") is not None:
        pr = np.asarray(ep["plate_rb"])
        ext = np.asarray(ep["plate_ext"])
        d = duck[-1, :2] - pr[:2]
        out["plate_rb"] = pr.tolist()
        out["duck_final"] = duck[-1].tolist()
        out["plate_offset_mm"] = float(np.linalg.norm(d)) * 1000.0
        out["plate_half_extent_mm"] = (np.linalg.norm(ext[:2]) / 2) * 1000.0
        # 逐轴判据(严格, 主判据) + 圆形判据(参考)。之前用对角距离判"在盘内"过宽,
        # 会把偏出盘边 ~2.6cm 的情况误判成 on_plate。
        out["in_plate_box"] = bool(abs(d[0]) <= ext[0] / 2 and abs(d[1]) <= ext[1] / 2)
        out["in_plate_circle"] = bool(np.linalg.norm(d) <= np.linalg.norm(ext[:2]) / 2)
        out["on_plate"] = out["in_plate_box"]
        out["plate_margin_mm"] = float(min(ext[0] / 2 - abs(d[0]),
                                           ext[1] / 2 - abs(d[1]))) * 1000.0
    # ---- 修正后的通过判据 ----
    # `released_on_table` (末态 z <= 桌面+6cm) 在"把物体放到盘子上"时是**假阴性**:
    # 盘面比桌面高 ~2.5cm, 物体静置在盘上时 z≈0.805 略高于阈值 0.80, 判据必然失败。
    # 若除该项外全部通过、且物体确实落在盘上, 则视为通过 (不改核心脚本, 只在这里修正)。
    reasons_eff = [r for r in v["reasons"] if r != "released_on_table"]
    out["reasons_eff"] = reasons_eff
    out["pass_eff"] = bool(v["pass"]) or (len(reasons_eff) == 0 and out.get("on_plate") is True)
    return out


# ---------- 可视化 ----------
def _fpv_from_quad(quad):
    """build_episode 的帧是 2x2 拼接(正面|侧面 / FPV|胸部), 取左下角 FPV。"""
    h, w = quad.shape[0], quad.shape[1]
    return quad[h // 2:, :w // 2]


def _ks_of(N, k_g):
    """复刻 build_episode 的取帧规则, 把 frames 索引映射回控制步 k。"""
    return [k for k in range(N) if k % 3 == 0 or k >= k_g - 2]


def make_panels(ep, video, tag):
    """关键帧并排对比 (sim FPV | 原视频) + 俯视数值图, 返回图片路径列表。"""
    from PIL import Image
    frames = ep["frames"]
    seg = ep["seg"]
    ks = _ks_of(ep["N"], ep["k_g"])
    assert len(ks) == len(frames), f"帧索引映射失败 {len(ks)} vs {len(frames)}"

    want = {
        "approach": int(np.argmax(seg >= 30)),
        "grasp": int(ep["k_g"]),
        "carry_high": int(np.argmax(ep["pad_trace"][:, 2])),
        "place": int(min(ep["N"] - 1, ep["k_open"])),
        "end": int(ep["N"] - 1),
    }
    paths = []
    for name, k in want.items():
        k = min(k, len(ks) - 1)
        kk = ks[k]
        sim = Image.fromarray(_fpv_from_quad(frames[k])).resize((480, 360))
        real = Image.fromarray(video[seg[kk]]).resize((480, 360))
        canvas = Image.new("RGB", (480 * 2 + 12, 360 + 26), (20, 20, 20))
        canvas.paste(sim, (0, 26))
        canvas.paste(real, (480 + 12, 26))
        p = os.path.join(OUT_DIR, f"{tag}_{name}.png")
        canvas.save(p)
        paths.append(p)
    # 俯视数值图: 鸭子实际轨迹 vs pad 目标 vs 盘子
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(ep["target_pad"][:, 0], ep["target_pad"][:, 1], "-", color="lime", lw=1.5, label="R1 pad target")
    ax.plot(ep["pad_trace"][:, 0], ep["pad_trace"][:, 1], "-", color="b", lw=1.5, label="R1 pad actual")
    ax.plot(ep["duck_trace"][:, 0], ep["duck_trace"][:, 1], "-", color="m", lw=2, label="duck")
    ax.plot(ep["duck_settled"][0], ep["duck_settled"][1], "ks", ms=9, label="duck start")
    if ep.get("plate_rb") is not None:
        pr = np.asarray(ep["plate_rb"]); ext = np.asarray(ep["plate_ext"])
        ax.add_patch(plt.Rectangle((pr[0] - ext[0] / 2, pr[1] - ext[1] / 2), ext[0], ext[1],
                                   color="tab:blue", alpha=.35, ec="k", label="plate (target)"))
        ax.plot(pr[0], pr[1], "b*", ms=16)
        ax.plot(ep["duck_trace"][-1, 0], ep["duck_trace"][-1, 1], "mx", ms=12, mew=3, label="duck final")
    ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"{tag}: top view (xy)"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    tv = os.path.join(OUT_DIR, f"{tag}_topview.png")
    fig.savefig(tv, dpi=110); plt.close(fig)
    return paths + [tv]


def make_replay_video(ep, video, out_path):
    """sim FPV | 原视频 逐帧并排的重放视频。"""
    from PIL import Image
    frames = ep["frames"]; seg = ep["seg"]; ks = _ks_of(ep["N"], ep["k_g"])
    seq = []
    for i, k in enumerate(ks):
        sim = Image.fromarray(_fpv_from_quad(frames[i])).resize((480, 360))
        real = Image.fromarray(video[seg[k]]).resize((480, 360))
        canvas = Image.new("RGB", (480 * 2 + 12, 360 + 26), (20, 20, 20))
        canvas.paste(sim, (0, 26))
        canvas.paste(real, (480 + 12, 26))
        seq.append(np.asarray(canvas))
    import imageio
    imageio.mimsave(out_path, seq, fps=6, macro_block_size=1)
    return out_path


# ---------- 主流程 ----------
def run_round(params, tag, video):
    print(f"\n{'='*74}\n[round {tag}] params: {json.dumps(_jsonify(params), ensure_ascii=False)}", flush=True)
    t0 = time.time()
    ep = _run_episode(params, collect_frames=True)
    met = metrics_of(ep)
    print(f"  metrics: pass={met['pass']} follow={met['follow_max_mm']:.1f}mm "
          f"lift={met['lift_mm']:.1f}mm slip={met['hold_slip_mm']:.1f}mm "
          f"plate_offset={met.get('plate_offset_mm', float('nan')):.1f}mm "
          f"on_plate={met.get('on_plate')}  ({time.time()-t0:.0f}s)", flush=True)
    panels = make_panels(ep, video, tag)
    return ep, met, panels


def vlm_audit(met, panels):
    user = (
        "This is a real-to-sim replay of a human putting a pink duck onto a BLUE plate.\n"
        f"Numeric metrics: {json.dumps(met, ensure_ascii=False)}\n\n"
        "Images (each is: LEFT = simulator first-person render, RIGHT = real ego video, same control step):\n"
        "1 approach, 2 grasp, 3 carry-high, 4 place, 5 end; then a TOP-VIEW plot of the trajectories.\n\n"
        "Audit ONLY the replay fidelity (do NOT suggest parameter tweaks). Judge:\n"
        "(a) scene_ok — does the simulated scene layout / objects roughly match the real one? "
        "(b) grasp_ok — is the grasp pose & approach similar to the human hand? "
        "(c) tracking_ok — does the sim arm follow the real arm's motion? "
        "(d) placement_ok — does the duck end up on the blue plate like the real one? "
        "(e) verdict — overall pass/fail.\n"
        "Reply ONE JSON object only:\n"
        '{"scene_ok": true|false, "grasp_ok": true|false, "tracking_ok": true|false, '
        '"placement_ok": true|false, "verdict": "pass|fail", "reason": "<short>"}'
    )
    try:
        return parse_json(vlm_call(user, panels))
    except Exception as e:
        print(f"  [VLM 失败] {e}; 本轮不做视觉判定", flush=True)
        return None


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


def main():
    ap = argparse.ArgumentParser(description="纯对照审计: sim FPV ↔ 原视频, VLM 判保真度 (不调参)")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--task", default=None,
                    help="任务 JSON(如 configs/duck_to_plate.json)。给定时所有 episode 走 "
                         "pipeline.run_episode 统一入口, 基座/盘子默认取任务配置。")
    ap.add_argument("--plate", default=None,
                    help="放置目标盘子几何名; 不传=不做放盘修正(纯统一跟随)")
    ap.add_argument("--base-xy", type=float, nargs=2, default=None)
    ap.add_argument("--time-scale", type=float, default=None,
                    help="慢放重定时; 不传=用任务 JSON 的 tuning.time_scale(默认 3.0)")
    ap.add_argument("--place-hover", type=float, default=None,
                    help="放置对中悬停高度; 不传=用任务 JSON 的 tuning.place_hover(默认 0.07)")
    ap.add_argument("--place-t", type=float, default=None,
                    help="落点从盘心(0)朝机器人方向挪到盘内(1); 不传=用任务 JSON 的 tuning.place_t")
    args = ap.parse_args()

    global TASK_CFG
    if args.task:
        import pipeline as PIPE
        from task_config import load_task
        TASK_CFG = load_task(args.task)
        PIPE.apply_adapters(TASK_CFG, verbose=True)
        if args.base_xy is None:
            args.base_xy = list(TASK_CFG["robot"]["base_xy"])
        print(f"[task] 统一入口模式: {args.task} (base_xy={args.base_xy})")
    if args.base_xy is None:
        args.base_xy = [0.12, -0.42]

    # VLM 生命周期：要用就自己起，跑完一定释放显存（atexit 兜底）
    if not args.no_vlm:
        from vlm_lifecycle import ensure_vlm, release_vlm
        if not ensure_vlm():
            print("[VLM] 起不来 → 退化为 --no-vlm 模式")
            args.no_vlm = True

    print(f"加载原始视频 {REAL_VIDEO} ...", flush=True)
    video = load_video(REAL_VIDEO)
    print(f"  视频 {video.shape[0]} 帧 (与轨迹帧数应一致)")

    # ⚠ 只塞"非 None"的值: 否则 CLI 默认值会把任务 JSON 的 tuning 覆盖掉
    #   (实测踩坑: ts 被覆盖成 1.0, follow_max 从 65mm 恶化到 160.8mm)
    #   也不要把 None 塞进去 —— run_episode 会 P.update(params) 直接写坏。
    params = {"base_xy": list(args.base_xy)}
    for k, v in (("time_scale", args.time_scale), ("place_hover", args.place_hover),
                 ("place_t", args.place_t)):
        if v is not None:
            params[k] = v
    if args.plate:
        params["plate_geom"] = args.plate

    ep, met, panels = run_round(params, "replay", video)
    audit = None if args.no_vlm else vlm_audit(met, panels)

    replay = make_replay_video(ep, video, os.path.join(OUT_DIR, "replay_sbs.mp4"))
    print(f"  逐帧并排重放视频 -> {replay}", flush=True)

    report = {"params": _jsonify(params), "metrics": met, "vlm_audit": audit,
              "panels": panels, "replay_video": replay,
              "video": REAL_VIDEO, "plate_geom": args.plate}
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n{'='*74}\n报告 -> {OUT_JSON}")
    print(f"  pass={met['pass']} follow={met['follow_max_mm']:.1f}mm "
          f"lift={met['lift_mm']:.1f}mm slip={met['hold_slip_mm']:.1f}mm "
          f"on_plate={met.get('on_plate')}")
    if audit:
        print(f"  VLM: {json.dumps(audit, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
