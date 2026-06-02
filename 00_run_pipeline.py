#!/usr/bin/env python3
"""
00_run_pipeline.py — 一键管线: 从原始数据文件夹生成仿真视频

管线:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  输入: hawor_dir/  +  ras_dir/                                         │
  │                                                                        │
  │  Step 1: 01_align_scene.py                                             │
  │    RAS GLB 场景 → 坐标对齐 → 变换参数                                    │
  │    输出: output/alignment/transform_params.npz                         │
  │                                                                        │
  │  Step 2: 02_render_scene.py --mode hand_only                           │
  │    MANO 手 + GLB 物体 → 第一人称视频                                    │
  │    输出: output/videos/hand_object_hand_only.mp4                       │
  │                                                                        │
  │  Step 3: 02_render_scene.py --mode robot_only                          │
  │    R1 机器人手部 + GLB 物体 → 机器人替代视频                            │
  │    输出: output/videos/hand_object_robot_only.mp4                      │
  │                                                                        │
  │  Step 4: 02_render_scene.py --mode robot_tracking                      │
  │    MANO 手 + R1 机器人 + GLB 物体 → 对比视频                           │
  │    输出: output/videos/hand_object_robot_tracking.mp4                  │
  │                                                                        │
  └─────────────────────────────────────────────────────────────────────────┘

用法:
python 00_run_pipeline.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result

    # 只运行部分步骤
    python 00_run_pipeline.py --hawor-dir ... --ras-dir ... --steps 1,2

    # 跳过对齐 (已有 transform_params.npz)
    python 00_run_pipeline.py --hawor-dir ... --ras-dir ... --skip-align

输入数据要求:
    hawor_dir/
    ├── reconstruction/
    │   └── hawor_results_*.npz    # HaWoR 手部重建结果
    └── cam_space/                  # 相机空间数据 (用于自动检测手索引)

    ras_dir/
    ├── final_scene.glb            # RAS 重建的 3D 场景
    └── cameras.txt                # RAS 相机外参

输出:
    output/
    ├── alignment/
    │   └── transform_params.npz   # 对齐参数
    └── videos/
        ├── hand_object_hand_only.mp4
        ├── hand_object_robot_only.mp4
        └── hand_object_robot_tracking.mp4
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(cmd, step_name):
    print(f"\n{'=' * 80}")
    print(f"  {step_name}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'=' * 80}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent),
                           stderr=subprocess.PIPE, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ {step_name} 失败!")
        print(f"  返回码: {result.returncode}")
        print(f"  耗时: {elapsed:.1f}s")
        if result.stderr:
            print(f"  ── 错误详情 ──")
            for line in result.stderr.strip().split('\n'):
                print(f"  {line}")
        return False
    print(f"\n✓ {step_name} 完成 (耗时 {elapsed:.1f}s)")
    return True


def find_reconstruction_file(hawor_dir):
    rec_dir = Path(hawor_dir) / "reconstruction"
    if not rec_dir.exists():
        return None
    for f in rec_dir.glob("hawor_results_*.npz"):
        return str(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="一键管线: 从 HaWoR + RAS 数据文件夹生成仿真视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="HaWoR 重建结果目录 (包含 reconstruction/ 子目录)")
    parser.add_argument("--ras-dir", type=str, required=True,
                        help="RAS 重建结果目录 (包含 final_scene.glb)")
    parser.add_argument("--steps", type=str, default="1,2,3,4",
                        help="运行的步骤, 逗号分隔 (1=对齐, 2=hand_only, 3=robot_only, 4=robot_tracking)")
    parser.add_argument("--skip-align", action="store_true",
                        help="跳过 Step 1 对齐 (使用已有的 transform_params.npz)")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="每步渲染的最大帧数 (-1=全部)")
    parser.add_argument("--fps", type=int, default=30,
                        help="视频帧率")
    parser.add_argument("--force-scale", type=float, default=None,
                        help="强制对齐尺度因子 (None=Umeyama自动计算)")
    parser.add_argument("--viewer", action="store_true",
                        help="交互式Viewer模式: 对齐后直接启动SAPIEN交互式渲染 (不生成视频)")
    args = parser.parse_args()

    steps = set(int(s.strip()) for s in args.steps.split(","))
    hawor_dir = Path(args.hawor_dir).resolve()
    ras_dir = Path(args.ras_dir).resolve()
    script_dir = Path(__file__).parent.resolve()

    print("=" * 80)
    print("  一键管线: HaWoR + RAS → 仿真视频")
    print("=" * 80)
    print(f"  HaWoR 目录: {hawor_dir}")
    print(f"  RAS   目录: {ras_dir}")
    print(f"  运行步骤:   {sorted(steps)}")
    print(f"  输出目录:   {script_dir / 'output'}")

    rec_file = find_reconstruction_file(hawor_dir)
    if rec_file is None:
        print(f"\n✗ 在 {hawor_dir}/reconstruction/ 中未找到 hawor_results_*.npz")
        sys.exit(1)
    print(f"  重建文件:   {rec_file}")

    transform_params = script_dir / "output" / "alignment" / "transform_params.npz"
    results = {}

    # ── Step 1: 对齐 ──────────────────────────────────────────────────
    if 1 in steps and not args.skip_align:
        cmd = [
            sys.executable, str(script_dir / "01_align_scene.py"),
            "--ras_output", str(ras_dir),
            "--hawor_reconstruction", str(rec_file),
            "--output_dir", str(script_dir / "output" / "alignment"),
        ]
        if args.force_scale is not None:
            cmd += ["--force_scale", str(args.force_scale)]
        results["Step 1 (对齐)"] = run_step(cmd, "Step 1: 对齐 RAS GLB → SAPIEN 坐标系")
    else:
        if transform_params.exists():
            print(f"\n  跳过 Step 1, 使用已有变换参数: {transform_params}")
            results["Step 1 (对齐)"] = True
        else:
            print(f"\n✗ 跳过对齐但未找到 {transform_params}")
            results["Step 1 (对齐)"] = False

    if not results.get("Step 1 (对齐)"):
        print("\n✗ Step 1 失败, 终止管线")
        sys.exit(1)

    # ── Step 2: hand_only / viewer ────────────────────────────────────
    if args.viewer:
        cmd = [
            sys.executable, str(script_dir / "02_render_scene.py"),
            "--mode", "hand_only",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--viewer",
        ]
        results["Step 2 (viewer)"] = run_step(cmd, "Step 2: 交互式 Viewer (手部+GLB+机器人)")
    elif 2 in steps:
        cmd = [
            sys.executable, str(script_dir / "02_render_scene.py"),
            "--mode", "hand_only",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--fps", str(args.fps),
        ]
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 2 (hand_only)"] = run_step(cmd, "Step 2: MANO 手 + GLB 物体")

    # ── Step 3: robot_only ────────────────────────────────────────────
    if 3 in steps:
        cmd = [
            sys.executable, str(script_dir / "02_render_scene.py"),
            "--mode", "robot_only",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--fps", str(args.fps),
        ]
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 3 (robot_only)"] = run_step(cmd, "Step 3: R1 机器人 + GLB 物体")

    # ── Step 4: robot_tracking ────────────────────────────────────────
    if 4 in steps:
        cmd = [
            sys.executable, str(script_dir / "02_render_scene.py"),
            "--mode", "robot_tracking",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--fps", str(args.fps),
        ]
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 4 (robot_tracking)"] = run_step(cmd, "Step 4: MANO 手 + R1 机器人 + GLB 物体")

    # ── 汇总 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  管线运行汇总")
    print("=" * 80)
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    success = all(results.values())
    if success:
        print(f"\n✓ 全部完成! 输出目录: {script_dir / 'output'}")
        for v in sorted((script_dir / "output" / "videos").glob("*.mp4")):
            size_mb = v.stat().st_size / 1024 / 1024
            print(f"    📹 {v.name} ({size_mb:.1f} MB)")
    else:
        print("\n✗ 管线部分步骤失败:")
        for name, ok in results.items():
            if not ok:
                print(f"    ✗ {name}")
        print("\n  请检查上方输出中的错误详情。")
        print("  可单独运行失败步骤以获取更多信息，例如:")
        for name, ok in results.items():
            if not ok:
                if "对齐" in name:
                    print(f"    python 01_align_scene.py --ras_output ... --hawor_reconstruction ...")
                elif "hand_only" in name:
                    print(f"    python 02_render_scene.py --mode hand_only --hawor-dir ... --ras-dir ... --transform-params ...")
                elif "robot_only" in name:
                    print(f"    python 02_render_scene.py --mode robot_only --hawor-dir ... --ras-dir ... --transform-params ...")
                elif "robot_tracking" in name:
                    print(f"    python 02_render_scene.py --mode robot_tracking --hawor-dir ... --ras-dir ... --transform-params ...")
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
