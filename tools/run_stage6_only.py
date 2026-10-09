#!/usr/bin/env python3
"""给已跑完的场景单独补跑 Stage 6 (材质感知物理参数标注) —— 不必重跑整条管线。

为什么需要它:
  mainv3 里 Stage 6 默认关闭 (要 --enable_stage6), 早期产出的一批场景目录里没有
  `materials.json`, 于是 Stage 6.1 的物理仿真只能退回 EmbodiedGen 硬编码的
  mu1=0.60 / mu2=0.50 / restitution=0.02。
  本脚本只读场景目录里**已经存在**的东西:
      all_instances.pkl        -> {all_instances, categories_and_relations}
      optimal_frames/*.jpg     -> 每个实例的最佳裁剪图
  然后调用 tools/material_annotator.run_stage6() 补出 materials.json。
  不做裁剪图提取、不改 GLB、不做任何 Stage 1~5 的事。

前置: 必须有 GPU (跑 Qwen VLM)。7_vggt_omega 约 1~2 分钟。

Usage:
    python tools/run_stage6_only.py --scene 7_vggt_omega
    python tools/run_stage6_only.py --missing        # 补跑所有缺 materials.json 的场景
    python tools/run_stage6_only.py --scene 7_vggt_omega --vlm /mnt/data/lza/models/Qwen3.5-9B
"""

import argparse
import os
import pickle
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_VLM = "/mnt/data/lza/models/Qwen3.5-9B"
DATA_ROOT = os.path.join(_ROOT, "output_v2")


def load_inputs(scene_dir):
    """读出 Stage 6 需要的两个输入。失败返回 (None, reason)。"""
    pkl = os.path.join(scene_dir, "all_instances.pkl")
    if not os.path.exists(pkl):
        return None, f"缺 all_instances.pkl"
    frames = os.path.join(scene_dir, "optimal_frames")
    if not os.path.isdir(frames):
        return None, "缺 optimal_frames/"
    try:
        with open(pkl, "rb") as f:
            blob = pickle.load(f)
    except Exception as e:
        return None, f"all_instances.pkl 读取失败: {e}"
    ai = blob.get("all_instances")
    cr = blob.get("categories_and_relations")
    if not isinstance(ai, dict) or not isinstance(cr, dict):
        return None, "all_instances.pkl 结构不符 (需含 all_instances/categories_and_relations)"
    return (ai, cr), None


def run_one(scene, vlm):
    scene_dir = os.path.join(DATA_ROOT, scene)
    inputs, why = load_inputs(scene_dir)
    if inputs is None:
        print(f"[SKIP] {scene}: {why}")
        return None
    ai, cr = inputs
    from tools.material_annotator import run_stage6

    print(f"\n{'='*70}\n[RUN ] {scene}  ({sum(len(v) for v in ai.values())} 个实例)\n{'='*70}",
          flush=True)
    mats = run_stage6(scene_dir, ai, cr, vlm)
    known = sum(1 for v in mats.values() if v.get("material") not in (None, "default"))
    print(f"[DONE] {scene}: {known}/{len(mats)} 个实例识别到材质 "
          f"-> {os.path.join(scene_dir, 'materials.json')}", flush=True)
    return mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", nargs="*", default=None, help="场景目录名")
    ap.add_argument("--missing", action="store_true",
                    help="补跑 output_v2/ 下所有缺 materials.json 且有 urdf/scene.urdf 的场景")
    ap.add_argument("--vlm", default=DEFAULT_VLM)
    args = ap.parse_args()

    if args.missing:
        scenes = sorted(
            s for s in os.listdir(DATA_ROOT)
            if os.path.isdir(os.path.join(DATA_ROOT, s))
            and not os.path.exists(os.path.join(DATA_ROOT, s, "materials.json"))
            and os.path.exists(os.path.join(DATA_ROOT, s, "urdf", "scene.urdf"))
        )
        print(f"待补跑 {len(scenes)} 个场景: {scenes}")
    elif args.scene:
        scenes = args.scene
    else:
        ap.error("需要 --scene 或 --missing")
        return 2

    ok = 0
    for s in scenes:
        try:
            if run_one(s, args.vlm) is not None:
                ok += 1
        except Exception as e:
            print(f"[FAIL] {s}: {type(e).__name__}: {e}", flush=True)
    print(f"\n完成 {ok}/{len(scenes)} 个场景")
    return 0


if __name__ == "__main__":
    sys.exit(main())
