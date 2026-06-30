#!/usr/bin/env python3
"""
render_auto.py — 自动检测手部 + 映射机械臂渲染

和 02_render_scene.py 对应, 但自动检测手部:
  - 自动检测左手/右手/双手
  - 单手: 直接渲染
  - 双手: 分别渲染左臂和右臂, 然后合成到一个视频 (左右并排)
  - 自动运行 01_align_scene.py 生成 transform_params (如果不存在)

用法:
    python hand_track/render_auto.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result
"""

import os
import subprocess
import sys
import time
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
COMBINATION_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from common import detect_hands, render_robot_video, render_gripper_video
from render_gripper_only import render_gripper_only_video, render_dual_gripper_video


def _ensure_transform_params(ras_dir, hawor_dir, output_dir, logger):
    """确保 transform_params.npz 存在, 不存在则自动运行 01_align_scene.py"""
    tp_path = os.path.join(output_dir, "transform_params.npz")
    if os.path.exists(tp_path):
        logger.info(f"  transform_params 已存在: {tp_path}")
        return tp_path

    logger.info(f"  transform_params 不存在, 自动运行 01_align_scene.py ...")
    hawor_recon_dir = os.path.join(hawor_dir, "reconstruction")
    if not os.path.isdir(hawor_recon_dir):
        logger.error(f"  hawor reconstruction 目录不存在: {hawor_recon_dir}")
        return None

    recon_npz = None
    for f in os.listdir(hawor_recon_dir):
        if f.startswith("hawor_results_") and f.endswith(".npz"):
            recon_npz = os.path.join(hawor_recon_dir, f)
            break
    if recon_npz is None:
        logger.error(f"  未找到 hawor_results_*.npz: {hawor_recon_dir}")
        return None

    align_script = str(COMBINATION_DIR / "01_align_scene.py")
    cmd = [
        sys.executable, align_script,
        "--ras_output", ras_dir,
        "--hawor_reconstruction", recon_npz,
        "--output_dir", output_dir,
    ]
    logger.info(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  01_align_scene.py 失败:\n{result.stderr[-500:]}")
        return None

    if os.path.exists(tp_path):
        logger.info(f"  ✓ transform_params 已生成: {tp_path}")
        return tp_path
    else:
        logger.error("  01_align_scene.py 运行完成但未生成 transform_params.npz")
        return None


def _combine_videos_side_by_side(left_video, right_video, output, fps, crf, logger):
    """将左右视频并排合成一个视频"""
    cap_l = cv2.VideoCapture(left_video)
    cap_r = cv2.VideoCapture(right_video)

    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))

    h_out = max(h_l, h_r)
    w_out = w_l + w_r

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w_out, h_out))

    frame_idx = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l and not ret_r:
            break

        # 如果一个视频比另一个短, 用黑帧填充
        if not ret_l:
            frame_l = np.zeros((h_l, w_l, 3), dtype=np.uint8)
        if not ret_r:
            frame_r = np.zeros((h_r, w_r, 3), dtype=np.uint8)

        # 调整高度一致
        if frame_l.shape[0] < h_out:
            pad = np.zeros((h_out - frame_l.shape[0], frame_l.shape[1], 3), dtype=np.uint8)
            frame_l = np.vstack([frame_l, pad])
        if frame_r.shape[0] < h_out:
            pad = np.zeros((h_out - frame_r.shape[0], frame_r.shape[1], 3), dtype=np.uint8)
            frame_r = np.vstack([frame_r, pad])

        combined = np.hstack([frame_l, frame_r])

        # 添加标签
        cv2.rectangle(combined, (0, 0), (w_l, 40), (0, 0, 0), -1)
        cv2.rectangle(combined, (w_l, 0), (w_out, 40), (0, 0, 0), -1)
        cv2.putText(combined, f"Left Arm", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Right Arm", (w_l + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, f"Frame {frame_idx+1}", (w_out // 2 - 50, h_out - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        writer.write(combined)
        frame_idx += 1

    cap_l.release()
    cap_r.release()
    writer.release()

    # ffmpeg 重编码
    tmp_path = str(output).replace(".mp4", "_tmp.mp4")
    if os.path.exists(str(output)):
        os.rename(str(output), tmp_path)
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
                   "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
                   "-movflags", "+faststart", str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
                os.remove(tmp_path)
            else:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, output)
        except Exception:
            if os.path.exists(tmp_path):
                os.rename(tmp_path, output)

    logger.info(f"  ✓ 双臂合成视频: {output} ({frame_idx} 帧)")


def main():
    parser = argparse.ArgumentParser(
        description="自动检测手部 + 映射机械臂渲染 (与02对应, 自动检测手部)",
    )
    parser.add_argument("--hawor-dir", type=str, required=True, help="HaWoR 数据目录")
    parser.add_argument("--ras-dir", type=str, required=True, help="RAS 重建结果目录 (含 final_scene.glb)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: hand_track/output/{hawor_name})")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--num-frames", type=int, default=-1, help="处理帧数 (-1=全部)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--width", type=int, default=1920, help="视频宽度")
    parser.add_argument("--height", type=int, default=1080, help="视频高度")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF 质量参数")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角 (默认 fpv 第一视角, 与02一致)")
    parser.add_argument("--mode", type=str, default="gripper",
                        choices=["gripper", "gripper_arm", "both"],
                        help="夹爪URDF渲染模式: gripper=仅夹爪, gripper_arm=夹爪+手臂末端, both=两者都渲染")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        choices=[-1, 0, 1],
                        help="手部索引: -1=自动检测, 0=强制左手, 1=强制右手 (默认 -1)")
    parser.add_argument("--optimizer", action="store_true",
                        help="使用 dex_retargeting PositionOptimizer (默认: 解析法 Gram-Schmidt)")
    args = parser.parse_args()

    # Logger
    logger = logging.getLogger("AutoRender")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)

    # 输出目录
    hawor_name = Path(args.hawor_dir).name
    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / "output" / hawor_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # [1] 自动检测手部 (或用户强制指定)
    if args.hand_idx >= 0:
        hand_indices = [args.hand_idx]
        hand_count = 1
        hand_label = "左手" if args.hand_idx == 0 else "右手"
        logger.info(f"[1/3] 手部指定: {hand_label} (index={args.hand_idx})")
    else:
        hand_indices = detect_hands(args.hawor_dir)
        hand_count = len(hand_indices)
        if hand_count == 0:
            logger.error("[1/3] 手部检测: 未检测到有效手部数据 (pred_valid 全为 False 或持续为 NaN), 停止生成")
            sys.exit(1)
        hand_label = "双手" if hand_count == 2 else ("左手" if hand_indices[0] == 0 else "右手")
        logger.info(f"[1/3] 手部检测: {hand_label} (indices={hand_indices})")

    # [2] 确保 transform_params 存在
    logger.info(f"\n[2/3] 准备 GLB 变换参数 ...")
    tp_path = _ensure_transform_params(args.ras_dir, args.hawor_dir, args.output_dir, logger)
    if tp_path is None:
        logger.warning("  无法获取 transform_params, 将不渲染 GLB 物体")

    # [3] 渲染
    logger.info(f"\n[3/3] 渲染视频 ...")
    start_time = time.time()

    # mode="both" 时渲染两轮: 先 gripper, 再 gripper_arm
    # tracking 和 keypoint 视频与 with_arm 无关, 只渲染一次; 夹爪URDF视频按 mode 循环
    render_modes = [False, True] if args.mode == "both" else [args.mode == "gripper_arm"]

    if hand_count == 1:
        # 单手: 直接渲染
        hi = hand_indices[0]
        side = "left" if hi == 0 else "right"
        video_name = f"hawor_r1_{side}_tracking.mp4"
        output_video = os.path.join(args.output_dir, "videos", video_name)
        os.makedirs(os.path.dirname(output_video), exist_ok=True)

        logger.info(f"  单手模式: {side}臂")
        logger.info(f"  output: {output_video}")

        render_robot_video(
            hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
            transform_params_path=tp_path, output=output_video,
            hand_idx=hi, fps=args.fps, cam_width=args.width, cam_height=args.height,
            view=args.view, crf=args.crf, start_frame=args.start_frame,
            num_frames=args.num_frames, logger=logger,
        )

        # 夹爪视频 (关键点球体)
        gripper_video = os.path.join(args.output_dir, "videos", f"hawor_r1_{side}_gripper.mp4")
        logger.info(f"\n  ── 渲染夹爪关键点 ──")
        render_gripper_video(
            hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
            transform_params_path=tp_path, output=gripper_video,
            hand_idx=hi, fps=args.fps, cam_width=args.width, cam_height=args.height,
            view=args.view, crf=args.crf, start_frame=args.start_frame,
            num_frames=args.num_frames, logger=logger,
        )

        # 夹爪URDF视频 (按 mode 循环渲染)
        for with_arm in render_modes:
            mode_suffix = "_arm" if with_arm else ""
            mode_label = "gripper_arm" if with_arm else "gripper"
            gripper_urdf_video = os.path.join(args.output_dir, "videos", f"hawor_r1_{side}_gripper_urdf{mode_suffix}.mp4")
            logger.info(f"\n  ── 渲染夹爪URDF (mode={mode_label}) ──")
            render_gripper_only_video(
                hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                transform_params_path=tp_path, output=gripper_urdf_video,
                hand_idx=hi, fps=args.fps, cam_width=args.width, cam_height=args.height,
                view=args.view, crf=args.crf, start_frame=args.start_frame,
                num_frames=args.num_frames, with_arm=with_arm, logger=logger,
                analytical=not args.optimizer,
            )
    else:
        # 双手: 分别渲染左臂和右臂, 然后合成
        videos_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)

        left_video = os.path.join(videos_dir, "hawor_r1_left_tracking.mp4")
        right_video = os.path.join(videos_dir, "hawor_r1_right_tracking.mp4")
        dual_video = os.path.join(videos_dir, "hawor_r1_dual_tracking.mp4")

        # 渲染左臂
        logger.info(f"\n  ── 渲染左臂 ──")
        render_robot_video(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            transform_params_path=tp_path,
            output=left_video,
            hand_idx=0,
            fps=args.fps,
            cam_width=args.width,
            cam_height=args.height,
            view=args.view,
            crf=args.crf,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            logger=logger,
        )

        # 渲染右臂
        logger.info(f"\n  ── 渲染右臂 ──")
        render_robot_video(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            transform_params_path=tp_path,
            output=right_video,
            hand_idx=1,
            fps=args.fps,
            cam_width=args.width,
            cam_height=args.height,
            view=args.view,
            crf=args.crf,
            start_frame=args.start_frame,
            num_frames=args.num_frames,
            logger=logger,
        )

        # 合成
        logger.info(f"\n  ── 合成双臂视频 ──")
        _combine_videos_side_by_side(left_video, right_video, dual_video, args.fps, args.crf, logger)
        # 删除单独的左/右视频 (用户不需要)
        for v in (left_video, right_video):
            if os.path.exists(v):
                os.remove(v)
        logger.info(f"  ✓ 已删除单独的左/右臂视频, 仅保留合成视频")

        # 夹爪URDF视频 (按 mode 循环渲染)
        for with_arm in render_modes:
            mode_suffix = "_arm" if with_arm else ""
            mode_label = "gripper_arm" if with_arm else "gripper"
            dual_gripper_urdf = os.path.join(videos_dir, f"hawor_r1_dual_gripper_urdf{mode_suffix}.mp4")

            logger.info(f"\n  ── 渲染双夹爪URDF (同场景, mode={mode_label}) ──")
            render_dual_gripper_video(
                hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                transform_params_path=tp_path, output=dual_gripper_urdf,
                fps=args.fps, cam_width=args.width, cam_height=args.height,
                view=args.view, crf=args.crf, start_frame=args.start_frame,
                num_frames=args.num_frames, with_arm=with_arm, logger=logger,
                hand_indices=hand_indices,
                analytical=not args.optimizer,
            )

    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
