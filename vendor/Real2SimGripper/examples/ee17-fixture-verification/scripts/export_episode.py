#!/usr/bin/env python3
"""export_episode.py — 把一次完成的仿真运行打包成**数据集 episode**。

为什么需要: 管线的产物原先散落四处 (EF3R outputs/ 的 targets+物体契约,
results/large_range/ 的指标+执行轨迹, demo_output/ 的 GIF), 没有"数据集"
这个最终形态。本脚本把一条视频的完整记录收拢到统一目录:

    dataset/<name>/
        meta.json              episode 元数据 (来源/链路/机器人/验收结论)
        robot/exec_trace.npz   机器人实际执行序列 (base/q/obj/q_target/grip)
        robot/targets.npz      IK 目标序列 (pos/rot/grip/grip_states)
        object/object.npz      物体 mesh/PCA/支撑面/中心 (object_instance)
        object/calib.npz       抓取校准 (抓取帧/人手握宽/在手偏移)
        scene/state_labels.json  PICK/LIFT/HOLD/PLACE/RELEASED 分段
        scene/gravity_align.npz  重力对齐 (M_sim, 若有)
        render/episode.gif     仿真渲染 (若已渲染)
        render/follow_compare.png  目标/IK/执行三线对比图
        acceptance.json        验收指标 (含物理抓取 6 项判据)

用法:
    python scripts/export_episode.py pickplace_7
    python scripts/export_episode.py pickplace_7 --out /path/to/dataset
    python scripts/export_episode.py agile456_phys \
        --ef3r-dir .../outputs/pickplace_456483158   # AGILE 链的 EF3R 侧目录
"""
import argparse
import json
import os
import shutil

R2S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(R2S, "results", "large_range")
DEMO = os.path.join(R2S, "demo_output", "large_range")
EF3R_OUT = "/path/to/project_data/mano_ws/EgoFound3R/outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="运行名 (results/large_range/<name>_large_range.json)")
    ap.add_argument("--ef3r-dir", default=None,
                    help="EF3R 产物目录 (默认 ~/mano_ws/EgoFound3R/outputs/<name>)")
    ap.add_argument("--out", default=os.path.join(R2S, "dataset"),
                    help="数据集根目录 (默认 <repo>/dataset/)")
    a = ap.parse_args()

    acc_p = os.path.join(RESULTS, f"{a.name}_large_range.json")
    if not os.path.exists(acc_p):
        raise SystemExit(f"✗ 未找到运行指标: {acc_p} (先跑通仿真)")
    e3r = a.ef3r_dir or os.path.join(EF3R_OUT, a.name)
    ep = os.path.join(a.out, a.name)

    # ---- 组装 ----
    os.makedirs(os.path.join(ep, "robot"), exist_ok=True)
    os.makedirs(os.path.join(ep, "object"), exist_ok=True)
    os.makedirs(os.path.join(ep, "scene"), exist_ok=True)
    os.makedirs(os.path.join(ep, "render"), exist_ok=True)

    def cp(src, dst, required=True):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(ep, dst))
            return True
        if required:
            print(f"  ⚠ 缺 {dst} (源: {src})")
        return False

    _robot = {
        "exec": cp(os.path.join(RESULTS, f"{a.name}_exec_trace.npz"),
                   "robot/exec_trace.npz", required=True),
        "targets": cp(os.path.join(e3r, f"ef3r_right_targets.npz"),
                      "robot/targets.npz", required=False)
        or cp(os.path.join(e3r, "ef3r_left_targets.npz"), "robot/targets.npz",
              required=False)
        or cp(os.path.join(e3r, "agile_right_targets_286.npz"),
              "robot/targets.npz", required=False),
    }
    _object = {
        "instance": cp(os.path.join(e3r, "object_instance.npz"), "object/object.npz"),
        "calib": cp(os.path.join(e3r, "pick_place_calib.npz"), "object/calib.npz",
                    required=False),
    }
    _scene = {
        "labels": cp(os.path.join(e3r, "state_labels.json"), "scene/state_labels.json",
                     required=False),
        "gravity": cp(os.path.join(e3r, "gravity_align.npz"),
                      "scene/gravity_align.npz", required=False),
    }
    _render = {
        "gif": cp(os.path.join(DEMO, f"{a.name}_large_range.gif"), "render/episode.gif",
                  required=False),
        "compare": cp(os.path.join(RESULTS, f"{a.name}_follow_compare.png"),
                      "render/follow_compare.png", required=False),
    }

    acc = json.load(open(acc_p))
    _acc_obj = (acc.get("object") or {}).get("acceptance") or {}
    meta = {
        "episode": a.name,
        "chain": "ef3r_fastpath" if "ef3r" in str(_robot.get("targets")) or
                 os.path.exists(os.path.join(e3r, "inference_output.pt")) else "agile",
        "video_fps": 30.0,
        "robot": "R1 (6-DoF arm + 4-DoF torso lift + parallel gripper, mobile base)",
        "source_dirs": {"ef3r": e3r, "results": RESULTS},
        "n_frames": acc.get("config", {}).get("n_frames"),
        "acceptance_pass": _acc_obj.get("pass"),
        "acceptance_criteria": _acc_obj.get("criteria"),
        "artifacts": {"robot": _robot, "object": _object, "scene": _scene,
                      "render": _render},
    }
    with open(os.path.join(ep, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(ep, "acceptance.json"), "w") as f:
        json.dump(acc, f, ensure_ascii=False, indent=1, default=str)

    print(f"✓ episode -> {ep}")
    print(f"  验收: {'✅ 通过' if meta['acceptance_pass'] else '❌ 未通过'} "
          f"| 帧数 {meta['n_frames']} | 执行轨迹 "
          f"{'✓' if _robot['exec'] else '✗ (先重跑仿真以生成)'}")
    return ep


if __name__ == "__main__":
    main()
