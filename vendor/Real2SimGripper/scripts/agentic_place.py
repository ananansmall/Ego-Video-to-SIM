#!/usr/bin/env python3
"""agentic_place.py — VLM 驱动的机器人基座定位（agentic real2sim 闭环）。

模拟 Agentic Real2Sim 的 "渲染→视觉推理→调场景→再渲染" 循环，聚焦**机器人基座位置**：
  1) 用当前 base_xy 建场景并跑一次 episode (不渲染, collect_frames=False)
  2) verifier 给出 follow_integrity 是否通过 (IK 是否因不可达而冻结)
  3) 未通过则渲染失败帧的正视 + 俯视两张图, 连同数值(stats)交给本地 VLM
     (Qwen3.8-27B, OpenAI 兼容端点), VLM 看图+读数值后给出基座位移建议 {dx, dy, reason}
  4) 施加位移, 回到 1) 迭代, 直到 follow_integrity 通过或到达最大轮数
  5) 输出最优 base_xy (JSON), 供 grasp/dataset 回灌

用法:
  CUDA_VISIBLE_DEVICES=0 python _scratch/agentic_place.py --rounds 5
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

import sapien.core as sapien  # noqa: E402

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

ENDPOINT = os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:18000/v1/chat/completions")
MODEL = os.environ.get("VLM_MODEL", "Qwen3.8-27B")
OUT_JSON = os.path.join(_RESULTS, "agentic_place_best.json")
# 基座可搜索范围 (地板平面, 保持不穿桌; 桌沿 y=-0.35)
BASE_X_RANGE = (-0.35, 0.35)
BASE_Y_RANGE = (-0.72, -0.40)
OBJ_DIR = os.path.join(_HERE, "agentic_place_frames")
os.makedirs(OBJ_DIR, exist_ok=True)

SYS_PROMPT = (
    "You are an expert robotics scene-placement agent for a Galaxea R1 dual-arm robot in a "
    "physics simulator. You are shown rendered views of the tabletop scene at the frame where "
    "the robot's left arm fails to follow the reference trajectory. Your job: decide how to move "
    "the robot's base on the floor so that the left arm can reach the required grasp/transport "
    "trajectory. Reason from BOTH the images and the numeric metrics. Be conservative: small "
    "translations (< 0.15 m per step). Always reply with a single JSON object."
)


def encode_img(path, max_side=768):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def vlm_call(user_text, image_paths, max_tokens=800):
    content = [{"type": "text", "text": user_text}]
    for p in image_paths:
        content.append({"type": "image_url", "image_url": {"url": encode_img(p)}})
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS_PROMPT},
                     {"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    text = resp["choices"][0]["message"]["content"]
    print(f"  [VLM] {round(time.time()-t0,1)}s -> {text.strip()[:200]}", flush=True)
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


def render_failure(ep, k_fail, tag):
    """把左臂设到失败帧位形, 渲染左前侧 3/4 视角 + 俯视, 返回两张图路径。

    视角选择: 执行臂是**左臂**(位于机器人自身 -x 侧), 故相机放机器人**左前方**
    (world -x, -y 侧), 让工作臂成为画面主体; 俯视用 up=[0,-1,0] 避免垂直向下
    时 up 与视线平行导致 look_at 退化(空白图)。
    """
    scene, robot = ep["scene"], ep["robot"]
    jmap = {j.get_name(): j for j in robot.get_active_joints()}
    for jn, q in zip(G.ARM_JOINTS, ep["xs"][k_fail]):
        jmap[jn].set_drive_target(float(q))
    G.step_sim(scene, robot, 40)
    W, H = 480, 360
    # 左前侧 3/4 视角: 相机在机器人左前方(-x,-y), 俯视工作区, 工作臂(左)在近侧
    side = scene.add_camera(name=f"ag_side_{tag}", width=W, height=H,
                            fovy=np.deg2rad(55), near=0.05, far=100)
    side.entity.set_pose(G.look_at_pose([-1.35, -1.15, 1.45], [0.0, -0.25, 0.90]))
    # 正俯视: up 用水平方向避免退化
    top = scene.add_camera(name=f"ag_top_{tag}", width=W, height=H,
                           fovy=np.deg2rad(60), near=0.05, far=100)
    top.entity.set_pose(G.look_at_pose([0.0, -0.25, 2.05], [0.0, -0.25, 0.74],
                                       up=np.array([0.0, -1.0, 0.0])))
    scene.update_render()
    paths = []
    for c, nm in ((side, "side"), (top, "top")):
        c.take_picture()
        img = (np.clip(c.get_picture("Color")[:, :, :3], 0, 1) * 255).astype(np.uint8)
        from PIL import Image
        p = os.path.join(OBJ_DIR, f"{tag}_{nm}.png")
        Image.fromarray(img).save(p)
        paths.append(p)
    return paths


def build_params(base_xy):
    return {"base_xy": [float(base_xy[0]), float(base_xy[1])]}


def run_round(base_xy, rnd):
    print(f"\n{'='*70}\n[round {rnd}] base_xy = {np.round(base_xy,3).tolist()}", flush=True)
    ep = _run_episode(build_params(base_xy))
    v = ep["verdict"]
    m = v["metrics"]
    follow = m.get("follow_max_m", None)
    print(f"   verdict={'PASS' if v['pass'] else 'FAIL'} follow_max={follow} "
          f"lift={m['lift_m']:.3f} slip={m['hold_slip_m']:.3f} reasons={v['reasons']}", flush=True)
    return ep, v, follow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--start", type=float, nargs=2, default=[0.0, -0.50])
    ap.add_argument("--task", default=None,
                    help="任务 JSON(如 configs/duck_to_plate.json)。给定时所有 episode 走 "
                         "pipeline.run_episode 统一入口, 起始基座默认取任务配置。")
    args = ap.parse_args()

    global TASK_CFG
    if args.task:
        import pipeline as PIPE
        from task_config import load_task
        TASK_CFG = load_task(args.task)
        PIPE.apply_adapters(TASK_CFG, verbose=True)
        if args.start == [0.0, -0.50]:       # 未显式指定 -> 用任务里的基座
            args.start = list(TASK_CFG["robot"]["base_xy"])
        print(f"[task] 统一入口模式: {args.task} (start={args.start})")

    base_xy = np.array(args.start, dtype=float)
    history = []
    best = None
    for rnd in range(args.rounds):
        ep, v, follow = run_round(base_xy, rnd)
        rec = {"round": rnd, "base_xy": base_xy.tolist(), "pass": v["pass"],
               "follow_max_m": follow, "reasons": v["reasons"]}
        history.append(rec)
        score = (v["pass"], -(follow if follow is not None else 9.9))
        if best is None or score > best[0]:
            best = (score, base_xy.copy().tolist(), rec)
        if v["checks"].get("follow_integrity", False) and v["pass"]:
            print("  -> follow_integrity 通过, 提前结束", flush=True)
            break

        # 失败帧 = 抓取后跟随段误差最大处
        k_g = ep["k_g"]
        errs = np.linalg.norm(ep["pad_trace"][k_g:] - ep["target_pad"][k_g:], axis=1)
        k_fail = int(k_g + np.argmax(errs))
        print(f"  失败帧 k={k_fail} (err={errs.max()*1000:.0f}mm) "
              f"target={np.round(ep['target_pad'][k_fail],3).tolist()}", flush=True)
        imgs = render_failure(ep, k_fail, f"r{rnd}")

        stats = {
            "robot_base_xy_m": np.round(base_xy, 3).tolist(),
            "duck_xy_m": np.round(ep["duck_settled"][:2], 3).tolist(),
            "table_front_edge_y_m": -0.35,
            "failure_frame": k_fail,
            "failure_target_point_m": np.round(ep["target_pad"][k_fail], 3).tolist(),
            "current_target_point_m": np.round(ep["pad_trace"][k_fail], 3).tolist(),
            "max_tracking_error_mm": round(float(errs.max() * 1000), 1),
            "reachable_x_range_m": list(BASE_X_RANGE),
            "reachable_y_range_m": list(BASE_Y_RANGE),
            "note": "y increases toward the table; robot must stay off the table (y <= -0.42).",
        }
        user = (
            "The **LEFT** arm is the working arm and it cannot follow the reference grasp "
            "trajectory; the IK solver freezes because the target is out of reach.\n"
            f"Scene metrics: {json.dumps(stats, ensure_ascii=False)}\n\n"
            "Images: IMAGE 1 = left-front 3/4 view (the arm closest to the camera is the "
            "LEFT working arm), IMAGE 2 = top-down view of the failure frame.\n"
            "Decide how to translate the robot base (in the table coordinate frame) so the "
            "LEFT arm can reach the failing target. Reply JSON only:\n"
            '{"dx": <float meters, + = robot moves to +x>, "dy": <float meters, + = robot moves '
            'toward the table>, "reason": "<short>"}'
        )
        try:
            resp = vlm_call(user, imgs)
            act = parse_json(resp)
            dx = float(np.clip(act.get("dx", 0.0), -0.15, 0.15))
            dy = float(np.clip(act.get("dy", 0.0), -0.15, 0.15))
            print(f"  [VLM 决策] dx={dx:+.3f} dy={dy:+.3f} reason={act.get('reason','')}", flush=True)
        except Exception as e:
            print(f"  [VLM 失败] {e}; 回退到默认前移", flush=True)
            dx, dy = 0.0, 0.08
        # 零位移探索: VLM 若判定"不动", 仍给一个默认前移以跳出局部停滞
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            dx, dy = 0.0, 0.06
            print("  [探索] VLM 建议不动, 施加默认前移 dy=+0.06", flush=True)
        base_xy[0] = np.clip(base_xy[0] + dx, *BASE_X_RANGE)
        base_xy[1] = np.clip(base_xy[1] + dy, *BASE_Y_RANGE)

    result = {"best_base_xy": best[1], "best_record": best[2], "history": history}
    json.dump(result, open(OUT_JSON, "w"), ensure_ascii=False, indent=2)
    print(f"\n{'='*70}\n最优 base_xy = {best[1]} (pass={best[2]['pass']}, "
          f"follow_max={best[2]['follow_max_m']})\n报告 -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
