#!/usr/bin/env python3
"""驱动脚本: 只跑 mainv3 的 Stage1 (VLM物体发现) + Stage2 (VGGT+SAM3+去重)。

复用现有管线函数, 不修改任何原逻辑。跑完 Stage2 得到:
  color/ depth/ extrinsics/ _pre_dedup_masks.pkl / coordinate_alignment.json
用于旋转类交互 (开盒盖) 的 overlap 曲线分析。
"""
import os, sys, time, argparse

REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, REPO)
os.chdir(REPO)

from mainv3 import run_stage1, run_stage2, _resolve_vlm_checkpoint
from src.vlm_logger import set_vlm_log_dir

VIDEO = "/mnt/data_8THDD/lza/dataset/HOI4D_RGB/HOI4D_selected_200/100_C4_Box_081deg.mp4"
OUT = os.path.join(REPO, "output_v2", "100_C4_Box_081deg_vggt_omega")
MAX_FRAMES = 120

def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    set_vlm_log_dir(OUT)

    vlm_ckpt = _resolve_vlm_checkpoint(None)
    print(f"🤖 VLM: {vlm_ckpt}", flush=True)

    print("\n── Stage 1: VLM 物体发现 ──", flush=True)
    t1 = time.time()
    stage1_json, cats = run_stage1(VIDEO, OUT, vlm_ckpt,
                                   max_frames_stage1=10, vggt_max_frames=MAX_FRAMES)
    print(f"Stage1 完成 ({time.time()-t1:.0f}s): {list(cats.keys())}", flush=True)

    print("\n── Stage 2: VGGT + SAM3 + 去重 ──", flush=True)
    t2 = time.time()
    vggt_res, dedup_masks, wall_masks, floor_masks = run_stage2(
        VIDEO, OUT, cats, max_frames=MAX_FRAMES, vggt_model_type="vggt")
    print(f"Stage2 完成 ({time.time()-t2:.0f}s), 类别: {list(dedup_masks.keys())}", flush=True)
    print(f"\n✅ 全部完成 ({time.time()-t0:.0f}s) → {OUT}", flush=True)

if __name__ == '__main__':
    main()
