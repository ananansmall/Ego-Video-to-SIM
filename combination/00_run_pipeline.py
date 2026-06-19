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
  │    输出: <session>/alignment/transform_params.npz                      │
  │                                                                        │
  │  Step 2: 02_render_scene.py --mode hand_only                           │
  │    MANO 手 + GLB 物体 → 第一人称视频                                    │
  │    输出: <session>/videos/hand_object_hand_only.mp4                    │
  │                                                                        │
  │  Step 3: 02_render_scene.py --mode robot_only                          │
  │    R1 机器人手部 + GLB 物体 → 机器人替代视频                            │
  │    输出: <session>/videos/hand_object_robot_only.mp4                   │
  │                                                                        │
  │  Step 4: 02_render_scene.py --mode robot_tracking                      │
  │    MANO 手 + R1 机器人 + GLB 物体 → 对比视频                           │
  │    输出: <session>/videos/hand_object_robot_tracking.mp4               │
  │                                                                        │
  │  Step 5: 02_render_scene.py --mode robot_only --view topdown           │
  │    R1 机器人 + GLB 物体 → 顶部俯瞰视频                                 │
  │    输出: <session>/videos/hand_object_robot_only_topdown.mp4           │
  │                                                                        │
  │  Step 6: 04_physics_simulation.py                                      │
  │    物理仿真: PD控制 + 碰撞 + 抓取                                      │
  │    输出: <session>/videos/physics_sim_physics_tracking.mp4             │
  │                                                                        │
  │  <session> = output/<hawor_name>_<ras_name>/                           │
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
    output/<hawor_name>_<ras_name>/
    ├── alignment/
    │   └── transform_params.npz   # 对齐参数
    ├── videos/
    │   ├── hand_object_hand_only.mp4
    │   ├── hand_object_robot_only.mp4
    │   ├── hand_object_robot_tracking.mp4
    │   ├── hand_object_robot_only_topdown.mp4
    │   └── physics_sim_physics_tracking.mp4
    ├── tracking/
    │   └── *.npy                  # 关节角序列
    └── pipeline.log               # 运行日志
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(cmd, step_name, log_file=None):
    """执行一个子进程步骤，记录日志并返回是否成功

    Args:
        cmd: 命令列表，如 [sys.executable, "01_align_scene.py", "--ras_output", ...]
        step_name: 步骤名称，用于打印和日志
        log_file: 日志文件路径，如果提供则将子进程输出追加到该文件

    Returns:
        bool: 步骤是否成功 (returncode == 0)
    """
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
            result = subprocess.run(cmd, cwd=str(Path(__file__).parent),
                                   stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent),
                               capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n✗ {step_name} 失败!")
        print(f"  返回码: {result.returncode}")
        print(f"  耗时: {elapsed:.1f}s")
        if log_file:
            print(f"  ── 错误详情 (来自日志) ──")
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                tail_lines = lines[-30:] if len(lines) > 30 else lines
                for line in tail_lines:
                    print(f"  {line.rstrip()}")
            except Exception:
                print(f"  (无法读取日志文件 {log_file})")
        elif result.stderr:
            print(f"  ── 错误详情 ──")
            for line in result.stderr.strip().split('\n'):
                print(f"  {line}")
        return False
    print(f"\n✓ {step_name} 完成 (耗时 {elapsed:.1f}s)")
    return True


def find_reconstruction_file(hawor_dir):
    """在 HaWoR 目录中查找重建结果文件

    搜索 hawor_dir/reconstruction/ 下的 hawor_results_*.npz 文件

    Args:
        hawor_dir: HaWoR 输出目录路径

    Returns:
        str: 找到的 npz 文件路径，或 None
    """
    rec_dir = Path(hawor_dir) / "reconstruction"
    if not rec_dir.exists():
        return None
    for f in rec_dir.glob("hawor_results_*.npz"):
        return str(f)
    return None


def main():
    """一键管线入口: 依次执行对齐、渲染、物理仿真等步骤

    流程:
      Step 1: 01_align_scene.py — RAS GLB 场景坐标对齐
      Step 2: 02_render_scene.py --mode hand_only — MANO手+GLB物体视频
      Step 3: 02_render_scene.py --mode robot_only — R1机器人+GLB物体视频
      Step 4: 02_render_scene.py --mode robot_tracking — 手+机器人+GLB对比视频
      Step 5: 04_physics_simulation.py — 物理仿真视频

    输出目录结构:
      output/<hawor_name>_<ras_name>/
      ├── alignment/transform_params.npz
      ├── videos/*.mp4
      ├── tracking/*.npy
      └── pipeline.log
    """
    parser = argparse.ArgumentParser(
        description="一键管线: 从 HaWoR + RAS 数据文件夹生成仿真视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="HaWoR 重建结果目录 (包含 reconstruction/ 子目录)")
    parser.add_argument("--ras-dir", type=str, required=True,
                        help="RAS 重建结果目录 (包含 final_scene.glb)")
    parser.add_argument("--steps", type=str, default="1,2,3,4,5",
                        help="运行的步骤, 逗号分隔 (1=对齐, 2=hand_only, 3=robot_only, 4=robot_tracking, 5=robot_only_topdown, 6=physics, 7=hand_track自动检测)")
    parser.add_argument("--skip-align", action="store_true",
                        help="跳过 Step 1 对齐 (使用已有的 transform_params.npz)")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="每步渲染的最大帧数 (-1=全部)")
    parser.add_argument("--fps", type=int, default=60,
                        help="视频帧率")
    parser.add_argument("--width", type=int, default=1920,
                        help="渲染宽度 (像素)")
    parser.add_argument("--height", type=int, default=1080,
                        help="渲染高度 (像素)")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角: fpv=第一人称, topdown=顶部俯视, behind=后上方, front=正前方")
    parser.add_argument("--force-scale", type=float, default=None,
                        help="强制对齐尺度因子 (None=Umeyama自动计算)")
    parser.add_argument("--handedness", type=str, default="auto",
                        choices=["auto", "left", "right", "both"],
                        help="手部类型: auto=自动检测, left=左手, right=右手, both=双手 (仅 --use-auto 模式生效)")
    parser.add_argument("--use-auto", action="store_true",
                        help="使用 02_render_scene_auto.py (自动检测手部+映射机械臂), 默认使用 02_render_scene.py")
    parser.add_argument("--viewer", action="store_true",
                        help="交互式Viewer模式: 对齐后直接启动SAPIEN交互式渲染 (不生成视频)")
    parser.add_argument("--smooth", type=int, default=0,
                        choices=[0, 1, 2],
                        help="平滑模式: 0=不平滑(默认), 1=在线EMA, 2=后处理双向滤波")
    parser.add_argument("--crf", type=int, default=14,
                        help="视频质量 (0=无损, 51=最差, 默认18)")
    args = parser.parse_args()

    steps = set(int(s.strip()) for s in args.steps.split(","))
    hawor_dir = Path(args.hawor_dir).resolve()
    ras_dir = Path(args.ras_dir).resolve()
    script_dir = Path(__file__).parent.resolve()

    hawor_name = hawor_dir.name
    ras_name = ras_dir.name
    session_name = f"{hawor_name}_{ras_name}"
    session_dir = script_dir / "output" / session_name

    os.makedirs(session_dir, exist_ok=True)
    log_path = session_dir / "pipeline.log"
    log_str = str(log_path)

    with open(log_str, "a", encoding="utf-8") as f:
        f.write(f"\n\n{'=' * 80}\n")
        f.write(f"  管线启动: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  hawor={hawor_dir}, ras={ras_dir}, steps={sorted(steps)}\n")
        f.write(f"  会话目录: {session_dir}\n")

    print("=" * 80)
    print("  一键管线: HaWoR + RAS → 仿真视频")
    print("=" * 80)
    print(f"  HaWoR 目录:  {hawor_dir}")
    print(f"  RAS   目录:  {ras_dir}")
    print(f"  运行步骤:    {sorted(steps)}")
    print(f"  会话名称:    {session_name}")
    print(f"  输出目录:    {session_dir}")
    print(f"  日志文件:    {log_path}")

    rec_file = find_reconstruction_file(hawor_dir)
    if rec_file is None:
        print(f"\n✗ 在 {hawor_dir}/reconstruction/ 中未找到 hawor_results_*.npz")
        sys.exit(1)
    print(f"  重建文件:    {rec_file}")

    transform_params = session_dir / "alignment" / "transform_params.npz"
    results = {}

    # ── Step 1: 对齐 ──────────────────────────────────────────────────
    if 1 in steps and not args.skip_align:
        cmd = [
            sys.executable, str(script_dir / "01_align_scene.py"),
            "--ras_output", str(ras_dir),
            "--hawor_reconstruction", str(rec_file),
            "--output_dir", str(session_dir / "alignment"),
        ]
        if args.force_scale is not None:
            cmd += ["--force_scale", str(args.force_scale)]
        results["Step 1 (对齐)"] = run_step(cmd, "Step 1: 对齐 RAS GLB → SAPIEN 坐标系", log_str)
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
    if args.use_auto:
        RENDER_SCRIPT = str(script_dir / "02_render_scene_auto.py")
    else:
        RENDER_SCRIPT = str(script_dir / "02_render_scene.py")

    auto_extra = ["--handedness", args.handedness] if args.use_auto else []

    if args.viewer:
        cmd = [
            sys.executable, RENDER_SCRIPT,
            "--mode", "hand_only",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--smooth", str(args.smooth),
        ] + auto_extra + ["--viewer"]
        results["Step 2 (viewer)"] = run_step(cmd, "Step 2: 交互式 Viewer (手部+GLB+机器人)", log_str)
        print("\n✓ Viewer 已关闭, 跳过后续步骤")
        return
    elif 2 in steps:
        out_video = str(session_dir / "videos" / "hand_object_hand_only.mp4")
        cmd = [
            sys.executable, RENDER_SCRIPT,
            "--mode", "hand_only",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--output", out_video,
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--view", args.view,
            "--smooth", str(args.smooth),
            "--crf", str(args.crf),
        ] + auto_extra
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 2 (hand_only)"] = run_step(cmd, "Step 2: MANO 手 + GLB 物体", log_str)

    # ── Step 3: robot_only ────────────────────────────────────────────
    if 3 in steps:
        out_video = str(session_dir / "videos" / "hand_object_robot_only.mp4")
        cmd = [
            sys.executable, RENDER_SCRIPT,
            "--mode", "robot_only",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--output", out_video,
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--view", args.view,
            "--smooth", str(args.smooth),
            "--crf", str(args.crf),
        ] + auto_extra
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 3 (robot_only)"] = run_step(cmd, "Step 3: R1 机器人 + GLB 物体", log_str)

    # ── Step 4: robot_tracking ────────────────────────────────────────
    if 4 in steps:
        out_video = str(session_dir / "videos" / "hand_object_robot_tracking.mp4")
        cmd = [
            sys.executable, RENDER_SCRIPT,
            "--mode", "robot_tracking",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--output", out_video,
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--view", args.view,
            "--smooth", str(args.smooth),
            "--crf", str(args.crf),
        ] + auto_extra
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 4 (robot_tracking)"] = run_step(cmd, "Step 4: MANO 手 + R1 机器人 + GLB 物体", log_str)

    # ── Step 5: robot_only_topdown ──────────────────────────────────
    if 5 in steps:
        out_video = str(session_dir / "videos" / "hand_object_robot_only_topdown.mp4")
        cmd = [
            sys.executable, RENDER_SCRIPT,
            "--mode", "robot_only",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--output", out_video,
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--view", "topdown",
            "--smooth", str(args.smooth),
            "--crf", str(args.crf),
        ] + auto_extra
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 5 (robot_only_topdown)"] = run_step(cmd, "Step 5: R1 机器人 + GLB 物体 (顶部俯瞰)", log_str)

    # ── Step 6: physics ──────────────────────────────────────────────
    if 6 in steps:
        out_video = str(session_dir / "videos" / "physics_sim_physics_tracking.mp4")
        cmd = [
            sys.executable, str(script_dir / "04_physics_simulation.py"),
            "--mode", "physics_tracking",
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--transform-params", str(transform_params),
            "--output", out_video,
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--smooth", str(args.smooth),
            "--crf", str(args.crf),
            "--hide-hand",
            "--fast-collision",
        ]
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 6 (physics)"] = run_step(cmd, "Step 6: 物理仿真 (PD控制 + 碰撞 + 抓取)", log_str)

    # ── Step 7: hand_track 自动检测渲染 ──────────────────────────────
    if 7 in steps:
        cmd = [
            sys.executable, str(script_dir / "hand_track" / "render_auto.py"),
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--output-dir", str(session_dir),
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--view", args.view,
            "--crf", str(args.crf),
        ]
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        results["Step 7 (hand_track)"] = run_step(cmd, "Step 7: hand_track 自动检测手部 + 机械臂渲染", log_str)

    # ── 汇总 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  管线运行汇总")
    print("=" * 80)
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")

    success = all(results.values())
    with open(log_str, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"  管线汇总: success={success}\n")
        for name, ok in results.items():
            f.write(f"  {'✓' if ok else '✗'} {name}\n")

    if success:
        print(f"\n✓ 全部完成! 输出目录: {session_dir}")
        for v in sorted((session_dir / "videos").glob("*.mp4")):
            size_mb = v.stat().st_size / 1024 / 1024
            print(f"    📹 {v.name} ({size_mb:.1f} MB)")
        print(f"    📋 日志: {log_path}")
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
                    print(f"    python {os.path.basename(RENDER_SCRIPT)} --mode hand_only --hawor-dir ... --ras-dir ... --transform-params ...")
                elif "robot_only" in name:
                    print(f"    python {os.path.basename(RENDER_SCRIPT)} --mode robot_only --hawor-dir ... --ras-dir ... --transform-params ...")
                elif "robot_tracking" in name:
                    print(f"    python {os.path.basename(RENDER_SCRIPT)} --mode robot_tracking --hawor-dir ... --ras-dir ... --transform-params ...")
                break
        sys.exit(1)


if __name__ == "__main__":
    main()
