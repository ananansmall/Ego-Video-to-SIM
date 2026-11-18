#!/usr/bin/env python3
"""pipeline.py —— **唯一的仿真入口**。所有调用方（CLI / agent / 分析脚本）都必须走这里。

为什么要它
----------
在此之前，每一样东西都各自 setup：`agentic_optimize` 自己 import `collision_config` +
`robot_config` + 拼 params；`run_from_task_json` 自己做一遍；分析脚本又做一遍。
结果是同一个任务有三套 setup，改一处漏两处，且外部看不出整体逻辑。

现在统一成一层：

    run_episode(cfg, params=..., collect_frames=...)
        ├─ apply_robot(G, cfg.robot)          ← 换机器人
        ├─ apply_task(G, cfg)                 ← 换任务(JSON)
        ├─ enable_coacd(...)                  ← 换碰撞精度
        └─ G.build_episode(params=...)        ← 核心管线(唯一的物理执行)
             └─ metrics_of(ep, cfg)           ← 指标(含示范保真度 gt_dist)

所有 adapter 仍是**猴子补丁**（不改核心脚本），但现在只有一个地方按固定顺序调用它们。

用法
----
    from pipeline import run_episode, load_task
    cfg = load_task("configs/duck_to_plate.json")
    ep, met = run_episode(cfg)                      # 跑一次
    ep, met = run_episode(cfg, params={"place_t": 0.3})   # 覆盖某个参数
    ep, met = run_episode(cfg, collect_frames=True)       # 要渲染
"""
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import robotwin_r1_duck_grasp as G                                    # noqa: E402
from collision_config import enable_coacd                             # noqa: E402
from robot_config import apply_robot                                  # noqa: E402
from task_config import load_task, apply_task                         # noqa: E402


# ---------------------------------------------------------------- 指标
def metrics_of(ep, cfg=None):
    """从一次 episode 抽指标。**自带**，不依赖任何 agent 脚本。"""
    v = ep["verdict"]
    m = dict(v["metrics"])
    duck = np.asarray(ep["duck_trace"])
    out = {
        "pass": bool(v["pass"]),
        "reasons": list(v["reasons"]),
        "follow_max_mm": float(m.get("follow_max_m", float("nan"))) * 1000.0,
        "lift_mm": float(m["lift_m"]) * 1000.0,
        "hold_slip_mm": float(m["hold_slip_m"]) * 1000.0,
        "knock_mm": float(m["knock_m"]) * 1000.0,
    }
    # ---- 放置目标相关 ----
    if ep.get("plate_rb") is not None:
        pr = np.asarray(ep["plate_rb"], float)
        ext = np.asarray(ep["plate_ext"], float)
        d = duck[-1, :2] - pr[:2]
        out["plate_rb"] = pr.tolist()
        out["duck_final"] = duck[-1].tolist()
        out["plate_offset_mm"] = float(np.linalg.norm(d)) * 1000.0
        out["plate_half_extent_mm"] = float(np.linalg.norm(ext[:2]) / 2) * 1000.0
        out["in_plate_box"] = bool(abs(d[0]) <= ext[0] / 2 and abs(d[1]) <= ext[1] / 2)
        out["in_plate_circle"] = bool(np.linalg.norm(d) <= np.linalg.norm(ext[:2]) / 2)
        out["on_plate"] = out["in_plate_box"]
        out["plate_margin_mm"] = float(min(ext[0] / 2 - abs(d[0]),
                                           ext[1] / 2 - abs(d[1]))) * 1000.0
        # ---- 示范保真度(创新点 1 的核心度量) ----
        gt = (cfg or {}).get("demonstration_ground_truth", {}).get("normalised_offset")
        if gt is not None:
            gt = np.asarray(gt, float)
            off = d / np.maximum(ext[:2], 1e-9)
            out["gt_dist"] = float(np.linalg.norm(off - gt))
            out["gt_off"] = [float(off[0]), float(off[1])]
            out["gt_y_cm"] = float(abs(off[1] - gt[1]) * ext[1] * 100.0)
            out["gt_x_cm"] = float(abs(off[0] - gt[0]) * ext[0] * 100.0)
    # ---- 通过判据: 有放置目标时**信任核心 verifier**, 不再手工剔除 ----
    # 2026-10-19 起, 核心 `verify_grasp(target=...)` 已修正 `released_on_table` 的假阴性
    # (盘面比桌面高 ~2.5cm → 静置在盘上必然失败), 并新增 `on_target` 判据
    # (见 robotwin_r1_duck_grasp.py)。核心给出 on_target 时, 本层直接采用核心结论,
    # 避免"核心 PASS / 本层 FAIL"两套口径漂移。
    if "on_target" in v["metrics"]:
        out["reasons_eff"] = list(v["reasons"])
        out["pass_eff"] = bool(v["pass"])
    else:
        # 无放置目标(或旧核心)时保留历史兼容逻辑(仅供非放盘任务)
        reasons_eff = [r for r in v["reasons"] if r != "released_on_table"]
        out["reasons_eff"] = reasons_eff
        out["pass_eff"] = bool(v["pass"]) or (len(reasons_eff) == 0 and out.get("on_plate") is True)
    return out


# ---------------------------------------------------------------- 打补丁（进程内只需一次）
def apply_adapters(cfg, verbose=False):
    """把机器人/任务/碰撞三个 adapter 按固定顺序打到核心模块 G 上。

    每个进程只需调用一次；之后反复 run_episode(setup=False) 复用补丁。
    """
    rb = cfg.get("robot", {})
    if rb.get("name"):
        apply_robot(G, rb["name"],
                    overrides={"base_xy": list(rb.get("base_xy", [0.12, -0.40]))})
    apply_task(G, cfg, verbose=verbose)
    col = cfg.get("sim", {}).get("collision", {})
    if col.get("mode") == "coacd":
        enable_coacd(G, col.get("threshold"), verbose=verbose,
                     max_depenetration_velocity=float(col.get("max_depenetration_velocity", 1.0)))
    # ---- base_xy 有两个真相源, 必须对齐 ----
    # apply_robot 改 G.R1_BASE_P（初始位姿常量），apply_task 改 PARAM_DEFAULTS["base_xy"]，
    # 而 build_episode 实际用的是 **PARAM_DEFAULTS["base_xy"]**。
    # 实测踩坑: 先 apply_adapters(基座A) 再 apply_robot(基座B)，机器人仍站在 A —— 
    # 看起来改了、其实没改。这里强制对齐, 并以 cfg["robot"]["base_xy"] 为准。
    bx = list(cfg.get("robot", {}).get("base_xy") or G.PARAM_DEFAULTS.get("base_xy", [0.12, -0.40]))
    G.PARAM_DEFAULTS["base_xy"] = bx
    if hasattr(G, "R1_BASE_P"):
        G.R1_BASE_P = [float(bx[0]), float(bx[1]), 0.0]
    if verbose:
        print(f"[pipeline] base_xy 已对齐 = {bx} (PARAM_DEFAULTS 与 R1_BASE_P 同步)")
    return G


# ---------------------------------------------------------------- 唯一入口
def run_episode(cfg, params=None, collect_frames=False, on_step=None,
                setup=True, verbose=False):
    """跑一次 episode。返回 (ep, metrics)。

    params 覆盖 cfg["tuning"] 里的调参项（agent 搜索用）。
    setup=False 时跳过 adapter（同一进程内已打过补丁，避免每次评测重打）。
    """
    if setup:
        apply_adapters(cfg, verbose=verbose)
    P = dict(cfg.get("tuning", {}))
    if params:
        P.update(params)
    ep = G.build_episode(on_step=on_step, collect_frames=collect_frames, params=P)
    return ep, metrics_of(ep, cfg)


def run_large_range(cfg, **opts):
    """**底盘跟随（大范围）后端**。

    大范围任务需要"底盘跟随手部 + 限位感知 IK + 底盘/臂联合求解(WBC)"，
    与固定基座的 pick-place 是两套求解器（见 `large_range_track.py` 的
    comfort_frame / track_base / solve_wbc_track）。
    这里**不重写求解器**，而是调度到已跑通的 `cb_large_range_sim.main()`，
    让它复用本项目的配置 schema 与 results 目录。
    """
    import cb_large_range_sim as cb
    lr = cfg.get("large_range", {})
    # WBC 参数默认 None -> cb.main 内走 auto_wbc_params(稳健基准), 所有轨迹共用;
    # 若配置里显式给 w_base/w_bcont/w_ori 则覆盖 (按需).
    kw = dict(stride=int(lr.get("stride", 1)), ik_only=bool(lr.get("ik_only", False)),
              base_smooth=lr.get("base_smooth", None), wbc=bool(lr.get("wbc", True)),
              w_base=lr.get("w_base"), w_bcont=lr.get("w_bcont"),
              base_range=lr.get("base_range"), w_ori=lr.get("w_ori"),
              grip_mode=lr.get("grip_mode", "data"),
              use_yaw=bool(lr.get("use_yaw", True)), smooth=bool(lr.get("smooth", False)),
              # 被抓物体/场景点云/物理抓取开关 (ef3r_object / agile_object 产出)
              object_npz=lr.get("object_npz"), scene_cloud=lr.get("scene_cloud"),
              grip_grasp=lr.get("grip_grasp"), object_physics=bool(lr.get("object_physics", False)),
              phys_steps=int(lr.get("phys_steps", 40)))
    # 轨迹来源: 优先配置里的 traj_npz; 缺省则用 cb_large_range_sim 内置的 chairbottle。
    # traj_npz 需含 pos/rot/grip/valid(/R_c2w/t_c2w), 由 hawor_to_traj.py 产出。
    traj = lr.get("traj_npz") or lr.get("targets")
    if traj:
        kw["targets"] = traj
    name = lr.get("name") or (cfg.get("task_id") or "large_range")
    kw["name"] = name
    kw.update(opts)
    # cb.main() 返回 GIF 路径字符串(phys_only 时返回 None) —— 不是模块属性 OUT_GIF
    gif = cb.main(**kw)
    return {"mode": "follow", "ep": None, "metrics": None, "artifact": gif}


def run(cfg, params=None, collect_frames=False, on_step=None, verbose=False, **lr_opts):
    """**按 base_mode 自动选后端**。这是推荐给外部调用的统一入口。

      base_mode="fixed"  → run_episode()   （基座固定，pick-place，如粉鸭）
      base_mode="follow" → run_large_range()（底盘跟随，大范围移动，如 chairbottle）
    """
    mode = cfg.get("robot", {}).get("base_mode", "fixed")
    if mode == "follow":
        return run_large_range(cfg, **lr_opts)
    ep, met = run_episode(cfg, params=params, collect_frames=collect_frames,
                          on_step=on_step, verbose=verbose)
    return {"mode": "fixed", "ep": ep, "metrics": met, "artifact": None}


def describe(cfg, met):
    """一页人类可读的运行摘要。"""
    lines = [
        f"verdict(raw)  = {'PASS' if met['pass'] else 'FAIL'}  {met['reasons']}",
        f"verdict(修正) = {'PASS' if met['pass_eff'] else 'FAIL'}  {met['reasons_eff']}",
        f"follow_max    = {met['follow_max_mm']:.1f} mm",
        f"抬升/滑移/抓前 = {met['lift_mm']:.1f} / {met['hold_slip_mm']:.1f} / {met['knock_mm']:.1f} mm",
    ]
    if "gt_dist" in met:
        lines += [
            f"鸭子末态      = {np.round(met['duck_final'], 4).tolist()}",
            f"在目标上      = {met['on_plate']}  余量 {met['plate_margin_mm']:.1f} mm",
            f"示范偏离      = {met['gt_dist']:.3f}  (x {met['gt_x_cm']:.1f}cm / y {met['gt_y_cm']:.1f}cm)",
        ]
    elif "on_plate" in met:
        lines += [
            f"鸭子末态      = {np.round(met['duck_final'], 4).tolist()}",
            f"在目标上      = {met['on_plate']}  余量 {met['plate_margin_mm']:.1f} mm",
            f"(该任务无示范真值 GT, 跳过保真度)",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="统一仿真入口")
    ap.add_argument("task", nargs="?", default="../configs/duck_to_plate.json")
    ap.add_argument("--set", action="append", default=[],
                    help="覆盖调参项, 如 --set place_t=0.3 --set time_scale=2.0")
    ap.add_argument("--render", action="store_true")
    a = ap.parse_args()
    cfg = load_task(a.task)
    over = {}
    for kv in a.set:
        k, _, v = kv.partition("=")
        try:
            over[k] = json.loads(v)
        except json.JSONDecodeError:
            over[k] = v
    print(f"[pipeline] task = {cfg.get('task_id')}  "
          f"base_mode = {cfg.get('robot', {}).get('base_mode', 'fixed')}")
    # ⚠ 必须走 run() 调度器 —— 直接调 run_episode 会忽略 base_mode,
    #   把大范围任务(follow)错跑成固定基座的 pick-place 管线。
    res = run(cfg, params=over or None, collect_frames=a.render)
    if res["mode"] == "follow":
        print(f"[pipeline] 已走大范围后端(WBC); 产物见 "
              f"demo_output/large_range/ 与 results/large_range/")
    else:
        print(describe(cfg, res["metrics"]))
        if a.render:
            from agentic_replay import load_video, make_panels, make_replay_video
            video = load_video(cfg["camera"]["ego_video"])
            make_panels(res["ep"], video, "run")
            make_replay_video(res["ep"], video,
                              os.path.join(G.OUT_DIR, "replay_from_pipeline.mp4"))
            print("[pipeline] 已渲染对比图与重放视频")
