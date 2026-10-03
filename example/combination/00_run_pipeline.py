#!/usr/bin/env python3
"""
00_run_pipeline.py — 一键管线: 从原始数据文件夹生成仿真视频

管线:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                        │
  │  输入: hawor_dir/  +  ras_dir/                                         │
  │                                                                        │
  │  --mode full (默认):                                                   │
  │    Step 1: 01_align_scene.py                                           │
  │    Step 2: 02_render_scene.py --mode hand_only                         │
  │    Step 3: 02_render_scene.py --mode robot_only                        │
  │    Step 4: 02_render_scene.py --mode robot_tracking                    │
  │    Step 5: 02_render_scene.py --mode robot_only --view topdown         │
  │    Step 6: 04_physics_simulation.py                                    │
  │                                                                        │
  │  --mode full-depth (--depth-align):                                    │
  │    Step 0: 01c_depth_align.py  (深度校正)                              │
  │    Step 1: 01_align_scene.py                                           │
  │    Step 2-5: 同上 (02_render_scene.py 所有 mode)                       │
  │                                                                        │
  │  --mode align-render:                                                  │
  │    Step 1: 001_align_scene.py                                          │
  │    Step 2: 002_render_scene.py --mode robot_tracking                   │
  │                                                                        │
  │  --mode align-render-depth (--depth-align):                            │
  │    Step 0: 01c_depth_align.py  (深度校正)                              │
  │    Step 1: 001_align_scene.py                                          │
  │    Step 2: 002_render_scene.py --mode robot_tracking                   │
  │                                                                        │
  │  <session> = output/<hawor_name>_<ras_name>/                           │
  │                                                                        │
  └─────────────────────────────────────────────────────────────────────────┘

用法:
    # 旧坐标管线 + 深度校正: 01c → 01 → 02
    python 00_run_pipeline.py --mode full-depth --hawor-dir ... --ras-dir ...

    # 新坐标管线 + 深度校正: 01c → 001 → 002
    python 00_run_pipeline.py --mode align-render-depth --hawor-dir ... --ras-dir ...

    # 旧坐标管线 (无深度校正)
    python 00_run_pipeline.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result

    # handtrack 模式 (自动手部检测+双夹爪+GLB, 替代 02 的步骤 2-5)
    python 00_run_pipeline.py --hawor-dir ... --ras-dir ... --handtrack

    # dexterous 模式 (灵巧手渲染, allegro/inspire/shadow/ability/leap/svh)
    python 00_run_pipeline.py --hawor-dir ... --ras-dir ... --dexterous --robot-name allegro
    python 00_run_pipeline.py --hawor-dir ... --ras-dir ... --dexterous --robot-name inspire

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
import subprocess as sp
import sys
import time
from pathlib import Path

# ── 自动检测 dex 环境 Python ─────────────────────────────────────────────
_PYTHON = sys.executable
try:
    import sapien  # noqa: F401
except ImportError:
    # 当前 Python 没有 sapien, 尝试找 conda/dex 环境的 Python
    _SEARCH_PATHS = [
        Path.home() / "miniconda3" / "envs" / "dex" / "bin" / "python",
        Path.home() / "anaconda3" / "envs" / "dex" / "bin" / "python",
        Path("/opt/conda") / "envs" / "dex" / "bin" / "python",
        Path("/opt/miniconda3") / "envs" / "dex" / "bin" / "python",
    ]
    for p in _SEARCH_PATHS:
        if p.exists():
            try:
                r = sp.run([str(p), "-c", "import sapien"], capture_output=True, timeout=10)
                if r.returncode == 0:
                    _PYTHON = str(p)
                    print(f"  [INFO] 使用 dex 环境 Python: {_PYTHON}")
                    break
            except Exception:
                continue
    if _PYTHON == sys.executable:
        print("[WARN] 未找到带 sapien 的 Python 环境, 将继续使用当前 Python")
        print("       建议: conda activate dex && python 00_run_pipeline.py ...")

PY = _PYTHON


def run_step(cmd, step_name, log_file=None):
    """执行一个子进程步骤，记录日志并返回是否成功

    Args:
        cmd: 命令列表，如 [PY, "01_align_scene.py", "--ras_output", ...]
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
            result = sp.run(cmd, cwd=str(Path(__file__).parent),
                                   stdout=f, stderr=sp.STDOUT, text=True)
    else:
        result = sp.run(cmd, cwd=str(Path(__file__).parent),
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


def run_align_render(args):
    """001_align_scene.py → 002_render_scene.py (新坐标系管线)"""

    hawor_dir = Path(args.hawor_dir).resolve()
    ras_dir = Path(args.ras_dir).resolve()
    script_dir = Path(__file__).parent.resolve()

    glb_path = args.glb_path if args.glb_path else str(ras_dir / "final_scene.glb")
    transform_params = hawor_dir / "transform_params.npz"

    rec_file = find_reconstruction_file(hawor_dir)
    if rec_file is None:
        print(f"\n✗ 在 {hawor_dir}/reconstruction/ 中未找到 hawor_results_*.npz")
        sys.exit(1)

    # Step 0: 深度校正 (可选)
    if args.depth_align:
        aligned_rec = rec_file.replace('.npz', '_depth_aligned.npz')
        if Path(aligned_rec).exists():
            print(f"\n  深度校正结果已存在, 跳过: {aligned_rec}")
            rec_file = aligned_rec
        else:
            # 查找 model_masks.npy
            mask_file = None
            for d in sorted(Path(hawor_dir).glob("tracks_*")):
                mf = d / "model_masks.npy"
                if mf.exists():
                    mask_file = str(mf)
                    break
            if mask_file is None:
                print(f"\n⚠ 未找到 model_masks.npy, 跳过深度校正")
            else:
                cmd_0 = [
                    PY, str(script_dir / "01c_depth_align.py"),
                    "--hawor-reconstruction", str(rec_file),
                    "--hawor-masks", str(mask_file),
                    "--ras-dir", str(ras_dir),
                    "--output", str(aligned_rec),
                ]
                print(f"\n{'='*60}\n[Step 0] 01c_depth_align.py (深度校正)\n{'='*60}")
                result = sp.run(cmd_0, cwd=str(script_dir))
                if result.returncode != 0:
                    print(f"\n⚠ 深度校正失败, 使用原始数据")
                else:
                    rec_file = aligned_rec
                    print(f"  使用校正后数据: {rec_file}")

    # Step 1: 001_align_scene.py
    if transform_params.exists():
        print(f"\n  transform_params 已存在, 跳过 001_align_scene.py: {transform_params}")
    else:
        cmd_001 = [
            PY, str(script_dir / "001_align_scene.py"),
            "--hawor_reconstruction", str(rec_file),
            "--ras_output", str(ras_dir),
            "--output_dir", str(hawor_dir),
        ]
        print(f"\n{'='*60}\n[Step 1/2] 001_align_scene.py\n{'='*60}")
        result = sp.run(cmd_001, cwd=str(script_dir))
        if result.returncode != 0:
            print(f"\n✗ 001_align_scene.py 失败")
            sys.exit(1)

    if not transform_params.exists():
        print(f"错误: {transform_params} 未生成")
        sys.exit(1)

    # Step 2: 002_render_scene.py
    cmd_002 = [
        PY, str(script_dir / "002_render_scene.py"),
        "--hawor-dir", str(hawor_dir),
        "--ras-dir", str(ras_dir),
        "--glb-path", str(glb_path),
        "--transform-params", str(transform_params),
        "--hand-idx", str(args.hand_idx) if args.hand_idx >= 0 else "-1",
        "--mode", "robot_tracking",
        "--view", args.view or "fpv",
        "--fps", str(args.fps),
        "--crf", str(args.crf),
        "--num-frames", str(args.num_frames),
        "--smooth", str(args.smooth),
    ]
    if args.fixed_base or args.viewer:
        cmd_002.append("--fixed-base")
    if args.viewer:
        cmd_002.append("--viewer")

    print(f"\n{'='*60}\n[Step 2/2] 002_render_scene.py\n{'='*60}")
    result = sp.run(cmd_002, cwd=str(script_dir))
    if result.returncode != 0:
        print(f"\n✗ 002_render_scene.py 失败")
        sys.exit(1)

    print("\n✓ align-render 完成")


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
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "full-depth", "render", "track_only", "align-render", "align-render-depth"],
                        help="管线模式: full=全部步骤(旧坐标), full-depth=01c深度校正→01对齐→02渲染, render=仅渲染, track_only=仅跟踪, align-render=001对齐+002渲染(新坐标), align-render-depth=01c深度校正→001对齐+002渲染(新坐标)")
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="HaWoR 重建结果目录 (包含 reconstruction/ 子目录)")
    parser.add_argument("--ras-dir", type=str, required=True,
                        help="RAS 重建结果目录 (包含 final_scene.glb)")
    parser.add_argument("--steps", type=str, default=None,
                        help="运行的步骤, 逗号分隔 (1=对齐, 2=hand_only, 3=robot_only, 4=robot_tracking, 5=robot_only_topdown, 6=physics, 7=hand_track自动检测, 8=灵巧手渲染). 默认: --handtrack时1,7; --dexterous时1,8; 否则1,2,3,4,5")
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
                        help="手部类型: auto=自动检测, left=左手, right=右手, both=双手 (仅 --use-auto / --handtrack 模式生效)")
    parser.add_argument("--use-auto", action="store_true",
                        help="使用 02_render_scene_auto.py (自动检测手部+映射机械臂), 默认使用 02_render_scene.py")
    parser.add_argument("--handtrack", action="store_true",
                        help="使用 hand_track 管线 (自动手部检测+双夹爪+GLB), 替代 02 的步骤 2-5, 默认运行步骤 1,7")
    parser.add_argument("--dexterous", action="store_true",
                        help="使用灵巧手渲染管线 (allegro/inspire/shadow/...), 替代夹爪, 默认运行步骤 1,8")
    parser.add_argument("--robot-name", type=str, default="allegro",
                        choices=["allegro", "inspire", "shadow", "ability", "leap", "svh"],
                        help="灵巧手名称, 仅 --dexterous 时生效 (默认 allegro)")
    parser.add_argument("--optimizer", action="store_true",
                        help="hand_track 步骤使用 dex_retargeting PositionOptimizer (默认: 解析法)")
    parser.add_argument("--viewer", action="store_true",
                        help="交互式Viewer模式: 对齐后直接启动SAPIEN交互式渲染 (不生成视频)")
    parser.add_argument("--smooth", type=int, default=0,
                        choices=[0, 1, 2],
                        help="平滑模式: 0=不平滑(默认), 1=在线EMA, 2=后处理双向滤波")
    parser.add_argument("--crf", type=int, default=14,
                        help="视频质量 (0=无损, 51=最差, 默认18)")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="手部索引: -1=自动检测, 0=左手, 1=右手 (仅 align-render 模式生效)")
    parser.add_argument("--glb-path", type=str, default=None,
                        help="GLB 文件路径 (默认: <ras-dir>/final_scene.glb, 仅 align-render 模式生效)")
    parser.add_argument("--fixed-base", action="store_true",
                        help="固定基座模式 (仅 align-render 模式生效)")
    parser.add_argument("--depth-align", action="store_true",
                        help="在 align 之前运行深度校正 (01c_depth_align.py), 用 RAS 深度图校正 HaWoR 手部深度")
    args = parser.parse_args()

    # align-render / align-render-depth 模式: 直接走新坐标管线, 不走旧步骤
    if args.mode == "align-render":
        run_align_render(args)
        return
    if args.mode == "align-render-depth":
        # 新坐标管线 + 深度校正: 01c → 001 → 002
        args.depth_align = True
        run_align_render(args)
        return

    # full-depth 模式: 旧管线 + 深度校正: 01c → 01 → 02(所有mode)
    if args.mode == "full-depth":
        args.depth_align = True

    if args.steps is None:
        if args.dexterous:
            args.steps = "1,8"
        elif args.handtrack:
            args.steps = "1,7"
        else:
            args.steps = "1,2,3,4,5"
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

    # ── Step 0: 深度校正 (可选) ─────────────────────────────────────────
    if args.depth_align:
        aligned_rec = rec_file.replace('.npz', '_depth_aligned.npz')
        if Path(aligned_rec).exists():
            print(f"\n  深度校正结果已存在, 跳过: {aligned_rec}")
            rec_file = aligned_rec
        else:
            mask_file = None
            for d in sorted(hawor_dir.glob("tracks_*")):
                mf = d / "model_masks.npy"
                if mf.exists():
                    mask_file = str(mf)
                    break
            if mask_file is None:
                print(f"\n⚠ 未找到 model_masks.npy, 跳过深度校正")
            else:
                cmd_da = [
                    PY, str(script_dir / "01c_depth_align.py"),
                    "--hawor-reconstruction", str(rec_file),
                    "--hawor-masks", str(mask_file),
                    "--ras-dir", str(ras_dir),
                    "--output", str(aligned_rec),
                ]
                ok = run_step(cmd_da, "Step 0: 深度校正 (RAS 深度图 → HaWoR 手部深度)", log_str)
                if ok:
                    rec_file = aligned_rec
                    print(f"  使用校正后数据: {rec_file}")
                else:
                    print(f"  ⚠ 深度校正失败, 使用原始数据")

    transform_params = session_dir / "alignment" / "transform_params.npz"
    results = {}

    # ── Step 1: 对齐 ──────────────────────────────────────────────────
    if 1 in steps and not args.skip_align:
        cmd = [
            PY, str(script_dir / "01_align_scene.py"),
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
            PY, RENDER_SCRIPT,
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
            PY, RENDER_SCRIPT,
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
            PY, RENDER_SCRIPT,
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
            PY, RENDER_SCRIPT,
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
            PY, RENDER_SCRIPT,
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
            PY, str(script_dir / "04_physics_simulation.py"),
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
            PY, str(script_dir / "hand_track" / "render_auto.py"),
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
        # 手部选择: 从 --handedness 映射到 --hand-idx
        # both → -1 (自动检测, detect_hands 会返回 [0,1])
        handedness_to_idx = {"auto": "-1", "left": "0", "right": "1", "both": "-1"}
        cmd += ["--hand-idx", handedness_to_idx.get(args.handedness, "-1")]
        # --handtrack 模式: 渲染夹爪 + 带机械臂夹爪两种
        if args.handtrack:
            cmd += ["--mode", "both"]
        if args.optimizer:
            cmd += ["--optimizer"]
        results["Step 7 (hand_track)"] = run_step(cmd, "Step 7: hand_track 自动检测手部 + 机械臂渲染", log_str)

    # ── Step 8: 灵巧手渲染 (dexterous hand) ─────────────────────────
    if 8 in steps:
        cmd = [
            PY, str(script_dir / "hand_track" / "render_dexterous_only.py"),
            "--hawor-dir", str(hawor_dir),
            "--ras-dir", str(ras_dir),
            "--output-dir", str(session_dir),
            "--robot-name", args.robot_name,
            "--fps", str(args.fps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--view", args.view,
            "--crf", str(args.crf),
        ]
        if args.num_frames > 0:
            cmd += ["--num-frames", str(args.num_frames)]
        # 手部选择: 从 --handedness 映射到 --hand-idx
        handedness_to_idx = {"auto": "-1", "left": "0", "right": "1", "both": "-1"}
        cmd += ["--hand-idx", handedness_to_idx.get(args.handedness, "-1")]
        results["Step 8 (dexterous)"] = run_step(cmd, f"Step 8: 灵巧手渲染 ({args.robot_name})", log_str)

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
