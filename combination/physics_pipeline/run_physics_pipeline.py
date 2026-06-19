#!/usr/bin/env python3
"""
run_physics_pipeline.py — 物理仿真独立管线

从原始数据 (HaWoR + RAS) 一键运行物理仿真, 自动处理依赖:

  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  输入: hawor_dir/  +  ras_dir/                                      │
  │                                                                     │
  │  Step 1: 坐标对齐 (01_align_scene.py)                               │
  │    RAS GLB 场景 → Umeyama 对齐 → transform_params.npz              │
  │    依赖: ras_dir/final_scene.glb + hawor reconstruction             │
  │    输出: <session>/alignment/transform_params.npz                   │
  │                                                                     │
  │  Step 2: 物理仿真 (04_physics_simulation.py)                        │
  │    PD控制 + 碰撞 + 抓取 → 仿真视频                                  │
  │    依赖: hawor_dir + ras_dir + transform_params.npz                 │
  │    输出: <session>/videos/physics_sim_physics_tracking.mp4          │
  │                                                                     │
  │  <session> = physics_pipeline/output/<hawor_name>_<ras_name>/       │
  │                                                                     │
  └──────────────────────────────────────────────────────────────────────┘

与 00_run_pipeline.py 的区别:
  - 只运行物理仿真所需的最少步骤 (对齐 + 物理仿真)
  - 独立输出目录, 不影响原有管线
  - 支持跳过对齐 (已有 transform_params.npz 时)

用法:
    # 完整运行 (对齐 + 物理仿真)
    python run_physics_pipeline.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result

    # 跳过对齐 (已有 transform_params.npz)
    python run_physics_pipeline.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result \\
        --skip-align \\
        --transform-params ./output/7_my_7mp4_result/alignment/transform_params.npz

    # 只运行物理仿真 (指定已有对齐参数)
    python run_physics_pipeline.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result \\
        --step physics-only \\
        --transform-params ./output/alignment/transform_params.npz

    # 调试模式 (快速碰撞, 显示手部)
    python run_physics_pipeline.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result \\
        --fast-collision --show-hand
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()
COMBINATION_DIR = SCRIPT_DIR.parent.resolve()


def run_step(cmd, step_name, log_file=None):
    print(f"\n{'=' * 80}")
    print(f"  {step_name}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'=' * 80}")
    t0 = time.time()
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"  {step_name}\n")
            f.write(f"  命令: {' '.join(cmd)}\n")
            f.write(f"  开始: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        with open(log_file, "a", encoding="utf-8") as f:
            result = subprocess.run(cmd, cwd=str(COMBINATION_DIR),
                                   stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd, cwd=str(COMBINATION_DIR),
                               capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ {step_name} 失败! (返回码: {result.returncode}, 耗时: {elapsed:.1f}s)")
        if log_file:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines[-30:]:
                    print(f"  {line.rstrip()}")
            except Exception:
                pass
        elif result.stderr:
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
        description="物理仿真独立管线: 对齐 + 物理仿真",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="HaWoR 重建结果目录")
    parser.add_argument("--ras-dir", type=str, required=True,
                        help="RAS 重建结果目录 (包含 final_scene.glb)")
    parser.add_argument("--step", type=str, default="full",
                        choices=["full", "align-only", "physics-only"],
                        help="运行步骤: full=对齐+物理, align-only=仅对齐, physics-only=仅物理")
    parser.add_argument("--skip-align", action="store_true",
                        help="跳过对齐 (使用已有的 transform_params.npz)")
    parser.add_argument("--transform-params", type=str, default=None,
                        help="已有的 transform_params.npz 路径 (跳过对齐时必需)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="自定义输出目录 (默认: physics_pipeline/output/<session>/)")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="最大渲染帧数 (-1=全部)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--fast-collision", action="store_true",
                        help="使用快速凸包碰撞体 (调试用, 速度更快)")
    parser.add_argument("--show-hand", action="store_true",
                        help="在物理仿真中显示 MANO 手部 (默认隐藏)")
    parser.add_argument("--viewer", action="store_true",
                        help="交互式 Viewer 模式")
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF (默认18)")
    parser.add_argument("--force-scale", type=float, default=None,
                        help="强制对齐尺度因子")
    args = parser.parse_args()

    hawor_dir = Path(args.hawor_dir).resolve()
    ras_dir = Path(args.ras_dir).resolve()

    hawor_name = hawor_dir.name
    ras_name = ras_dir.name
    session_name = f"{hawor_name}_{ras_name}"

    if args.output_dir:
        session_dir = Path(args.output_dir).resolve()
    else:
        session_dir = SCRIPT_DIR / "output" / session_name
    os.makedirs(session_dir, exist_ok=True)

    log_path = session_dir / "physics_pipeline.log"

    print("=" * 80)
    print("  物理仿真独立管线")
    print("=" * 80)
    print(f"  HaWoR 目录:  {hawor_dir}")
    print(f"  RAS   目录:  {ras_dir}")
    print(f"  运行步骤:    {args.step}")
    print(f"  会话名称:    {session_name}")
    print(f"  输出目录:    {session_dir}")
    print(f"  日志文件:    {log_path}")

    rec_file = find_reconstruction_file(hawor_dir)
    if rec_file is None:
        print(f"\n✗ 在 {hawor_dir}/reconstruction/ 中未找到 hawor_results_*.npz")
        sys.exit(1)
    print(f"  重建文件:    {rec_file}")

    transform_params = None
    results = {}

    if args.step == "align-only":
        cmd = [
            sys.executable, str(COMBINATION_DIR / "01_align_scene.py"),
            "--ras_output", str(ras_dir),
            "--hawor_reconstruction", str(rec_file),
            "--output_dir", str(session_dir / "alignment"),
        ]
        if args.force_scale is not None:
            cmd += ["--force_scale", str(args.force_scale)]
        ok = run_step(cmd, "Step 1: 坐标对齐 (RAS GLB → SAPIEN)", str(log_path))
        results["对齐"] = ok
        if ok:
            transform_params = session_dir / "alignment" / "transform_params.npz"
            print(f"\n  对齐参数保存至: {transform_params}")
        sys.exit(0 if ok else 1)

    if args.step in ("full",) and not args.skip_align:
        cmd = [
            sys.executable, str(COMBINATION_DIR / "01_align_scene.py"),
            "--ras_output", str(ras_dir),
            "--hawor_reconstruction", str(rec_file),
            "--output_dir", str(session_dir / "alignment"),
        ]
        if args.force_scale is not None:
            cmd += ["--force_scale", str(args.force_scale)]
        ok = run_step(cmd, "Step 1: 坐标对齐 (RAS GLB → SAPIEN)", str(log_path))
        results["对齐"] = ok
        if ok:
            transform_params = session_dir / "alignment" / "transform_params.npz"
        else:
            print("\n✗ 对齐失败, 终止管线")
            sys.exit(1)
    else:
        if args.transform_params:
            transform_params = Path(args.transform_params).resolve()
        elif (session_dir / "alignment" / "transform_params.npz").exists():
            transform_params = session_dir / "alignment" / "transform_params.npz"
        else:
            print(f"\n✗ 未指定 --transform-params 且未找到已有对齐参数")
            print(f"  请先运行: python {__file__} --hawor-dir ... --ras-dir ... --step align-only")
            sys.exit(1)

        if not transform_params.exists():
            print(f"\n✗ 对齐参数不存在: {transform_params}")
            sys.exit(1)

        print(f"\n  使用已有对齐参数: {transform_params}")
        results["对齐"] = True

    if args.step in ("full", "physics-only"):
        os.makedirs(session_dir / "videos", exist_ok=True)
        out_video = str(session_dir / "videos" / "physics_sim_physics_tracking.mp4")

        cmd = [
            sys.executable, str(COMBINATION_DIR / "04_physics_simulation.py"),
            "--mode", "physics_tracking",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--output", out_video,
            "--fps", str(args.fps),
            "--crf", str(args.crf),
        ]
        if args.fast_collision:
            cmd += ["--fast-collision"]
        if not args.show_hand:
            cmd += ["--hide-hand"]
        if args.viewer:
            cmd += ["--viewer"]
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]

        ok = run_step(cmd, "Step 2: 物理仿真 (PD控制 + 碰撞 + 抓取)", str(log_path))
        results["物理仿真"] = ok

    print("\n" + "=" * 80)
    print("  管线运行汇总")
    print("=" * 80)
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")

    success = all(results.values())
    if success:
        print(f"\n✓ 全部完成! 输出目录: {session_dir}")
        videos_dir = session_dir / "videos"
        if videos_dir.exists():
            for v in sorted(videos_dir.glob("*.mp4")):
                size_mb = v.stat().st_size / 1024 / 1024
                print(f"  📹 {v.name} ({size_mb:.1f} MB)")
        print(f"  📋 日志: {log_path}")
    else:
        print("\n✗ 管线部分步骤失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
