"""
第一人称视频 -> 机器人手部重定向管道（Allegro 等独立手部）

完整流程：
  1. 从第一人称视频中提取手部姿态 (MediaPipe Hands)
  2. 将人手关节映射到机器人手部关节 (dex_retargeting, SeqRetargeting)
  3. 接触保持滤波 + 轨迹平滑
  4. 保存轨迹数据 (trajectory.pkl)
  5. 渲染机器人手部视频 (SAPIEN 3D / 软件回退)

注意：此管道仅适用于独立手部（如 Allegro），不包含手臂 IK 求解。
如需 R1 全身机器人仿真，请使用 r1_sapien_sim.py。

使用示例：
  conda run -n dex python run_pipeline.py \\
    --video-path /path/to/video.mp4 \\
    --output-dir ./output \\
    --robot allegro \\
    --hand right
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dex_retargeting.constants import RobotName, RetargetingType, HandType
from egocentric_hand_detector import EgocentricHandDetector
from arm_motion_analyzer import ArmMotionAnalyzer
from robot_hand_retargeter import RobotHandRetargeter
from contact_preservation import ContactPreservationFilter, TrajectorySmoother

# Allegro 手部手指连接关系（4指，每指4关节）
ALLEGRO_FINGER_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
]

# Allegro 手部各指颜色（BGR 格式：蓝、橙、绿、红）
ALLEGRO_FINGER_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0),
]

# MANO 手部手指连接关系（5指，含掌骨连接）
MANO_FINGER_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def render_software_trajectory(pkl_path, output_video, fps=30):
    """
    软件渲染机器人手部轨迹视频（SAPIEN 不可用时的回退方案）

    生成双面板视频：
      - 左面板：各手指关节角度柱状图
      - 右面板：手部状态可视化（抓取状态 + 手指骨架示意）

    Args:
        pkl_path: 轨迹数据文件路径 (trajectory.pkl)
        output_video: 输出视频文件路径
        fps: 帧率
    """
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    data = result["data"]
    wrist_trajectory = result.get("wrist_trajectory", [])
    meta_data = result["meta_data"]
    joint_names = meta_data["joint_names"]
    dof = meta_data["dof"]

    W, H = 1280, 720
    panel_w = W // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    valid_data = [d for d in data if d is not None]
    if len(valid_data) == 0:
        writer.release()
        return

    # 计算关节角度范围（用于归一化显示）
    all_qpos = np.array(valid_data)
    qpos_min = all_qpos.min(axis=0)
    qpos_max = all_qpos.max(axis=0)
    qpos_range = qpos_max - qpos_min
    qpos_range[qpos_range < 1e-6] = 1.0

    n_fingers = 4
    joints_per_finger = dof // n_fingers

    for frame_i in range(len(data)):
        canvas = np.full((H, W, 3), 30, dtype=np.uint8)

        qpos = data[frame_i]
        wt = wrist_trajectory[frame_i] if frame_i < len(wrist_trajectory) else None

        # 左右面板分割
        left_panel = canvas[:, :panel_w]
        right_panel = canvas[:, panel_w:]

        cv2.rectangle(left_panel, (0, 0), (panel_w, H), (20, 20, 30), -1)
        cv2.rectangle(right_panel, (0, 0), (panel_w, H), (20, 25, 20), -1)

        cv2.putText(left_panel, "Robot Joint Angles", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(right_panel, "Hand State", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if qpos is not None:
            margin_x, margin_y = 40, 60
            plot_w = panel_w - 2 * margin_x
            plot_h = (H - margin_y - 40) // n_fingers - 10

            # 左面板：绘制各手指关节角度柱状图
            for fi in range(n_fingers):
                y_off = margin_y + fi * (plot_h + 10)
                finger_names = ["Thumb", "Index", "Middle", "Ring"]
                color = ALLEGRO_FINGER_COLORS[fi]

                cv2.putText(left_panel, finger_names[fi], (margin_x, y_off - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                for ji in range(joints_per_finger):
                    idx = fi * joints_per_finger + ji
                    if idx >= dof:
                        break
                    val = qpos[idx]
                    norm_val = (val - qpos_min[idx]) / qpos_range[idx]
                    norm_val = np.clip(norm_val, 0, 1)

                    bar_w = plot_w // joints_per_finger - 4
                    bar_x = margin_x + ji * (plot_w // joints_per_finger)
                    bar_h = int(norm_val * (plot_h - 20))
                    bar_y = y_off + plot_h - bar_h

                    cv2.rectangle(left_panel,
                                  (bar_x, y_off + 15),
                                  (bar_x + bar_w, y_off + plot_h),
                                  (50, 50, 50), -1)
                    cv2.rectangle(left_panel,
                                  (bar_x, bar_y),
                                  (bar_x + bar_w, y_off + plot_h),
                                  color, -1)

                    cv2.putText(left_panel, f"J{ji}", (bar_x + 2, y_off + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

            # 右面板：手部状态可视化
            cx, cy = panel_w // 2, H // 2 + 30
            radius = min(panel_w, H) // 3

            if wt is not None:
                is_grasping = wt.get("is_grasping", False)
                finger_dist = wt.get("finger_distance", 0)
            else:
                is_grasping = False
                finger_dist = 0

            status = "GRASP" if is_grasping else "OPEN"
            status_color = (0, 255, 0) if is_grasping else (0, 100, 255)
            cv2.putText(right_panel, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(right_panel, f"Finger dist: {finger_dist:.4f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # 绘制手指骨架示意（基于关节角度的简化可视化）
            for fi in range(n_fingers):
                finger_joints = []
                for ji in range(joints_per_finger):
                    idx = fi * joints_per_finger + ji
                    if idx >= dof:
                        break
                    angle = qpos[idx]
                    norm_angle = (angle - qpos_min[idx]) / qpos_range[idx]
                    finger_joints.append(norm_angle)

                color = ALLEGRO_FINGER_COLORS[fi]
                base_angle = fi * (2 * np.pi / n_fingers) - np.pi / 2
                pts = [(cx, cy)]
                cur_angle = base_angle
                seg_len = radius / (len(finger_joints) + 1)

                for ja in finger_joints:
                    cur_angle += (ja - 0.5) * 1.2
                    px = int(cx + seg_len * np.cos(cur_angle))
                    py = int(cy + seg_len * np.sin(cur_angle))
                    pts.append((px, py))
                    seg_len *= 0.85

                for k in range(len(pts) - 1):
                    cv2.line(right_panel, pts[k], pts[k + 1], color, 3)
                for pt in pts:
                    cv2.circle(right_panel, pt, 4, color, -1)

            if wt is not None:
                wp = wt.get("wrist_pos", np.zeros(3))
                cv2.putText(right_panel,
                            f"Wrist: [{wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f}]",
                            (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (150, 150, 150), 1)
        else:
            cv2.putText(left_panel, "No detection", (panel_w // 4, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.putText(right_panel, "No detection", (panel_w // 4, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

        cv2.putText(canvas, f"Frame {frame_i + 1}/{len(data)}", (W // 2 - 60, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        writer.write(canvas)

    writer.release()


def run_full_pipeline(
    video_path: str,
    output_dir: str,
    robot_name: RobotName = RobotName.allegro,
    hand_type: HandType = HandType.right,
    retargeting_type: RetargetingType = RetargetingType.position,
    fps: int = 30,
    start_frame: int = 0,
    num_frames: int = -1,
    arm_offset: list = None,
    selfie: bool = True,
    grasp_threshold: float = None,
    release_threshold: float = None,
    visualize: bool = True,
    render: bool = True,
):
    """
    执行完整的第一人称视频 -> 机器人手部重定向管道

    三步流程：
      1. 手部检测 + 重定向 + 滤波 + 平滑
      2. 保存轨迹数据
      3. 渲染视频（SAPIEN 3D 或软件回退）

    Args:
        video_path: 输入视频文件路径
        output_dir: 输出目录路径
        robot_name: 机器人名称（如 RobotName.allegro）
        hand_type: 手别（HandType.right / HandType.left）
        retargeting_type: 重定向类型（position / vector / dexpilot）
        fps: 目标帧率
        start_frame: 起始帧索引
        num_frames: 处理帧数（-1 表示全部）
        arm_offset: 末端执行器偏移量
        selfie: 是否为自拍模式（第一人称视角）
        grasp_threshold: 抓取检测阈值
        release_threshold: 释放检测阈值
        visualize: 是否显示实时检测可视化
        render: 是否渲染机器人手部视频
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "trajectory.pkl"
    mp4_path = output_dir / "result.mp4"
    overlay_path = output_dir / "overlay.mp4"
    software_render_path = output_dir / "robot_hand.mp4"

    hand_str = "Right" if hand_type == HandType.right else "Left"
    print("=" * 70)
    print("  Egocentric Video -> Robot Hand Retargeting Pipeline")
    print("=" * 70)
    print(f"  Video:     {video_path}")
    print(f"  Robot:     {robot_name.name}")
    print(f"  Hand:      {hand_str}")
    print(f"  Retarget:  {retargeting_type.name}")
    print(f"  Selfie:    {selfie}")
    print(f"  Output:    {output_dir}")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 根据目标帧率计算跳帧数
    frame_skip = max(1, int(video_fps / fps))
    if num_frames < 0:
        num_frames = total_frames

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化各模块
    detector = EgocentricHandDetector(
        hand_type=hand_str,
        selfie=selfie,
    )
    arm_analyzer = ArmMotionAnalyzer(
        hand_type=hand_type,
        arm_offset=np.array(arm_offset) if arm_offset else np.zeros(3),
    )
    retargeter = RobotHandRetargeter(
        robot_name=robot_name,
        hand_type=hand_type,
        retargeting_type=retargeting_type,
    )
    contact_filter = ContactPreservationFilter(
        grasp_threshold=grasp_threshold if grasp_threshold else 0.04,
        release_threshold=release_threshold if release_threshold else 0.06,
    )
    trajectory_smoother = TrajectorySmoother(window_size=3, alpha=0.3)

    # 创建叠加视频写入器
    overlay_writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (vid_w, vid_h),
    )

    data = []
    wrist_trajectory = []
    meta_data = retargeter.get_meta_data()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0
    detected = 0
    finger_distances = []

    # 步骤1：手部检测 + 重定向
    print("[1/3] Detecting hands and retargeting...")
    with tqdm.tqdm(total=num_frames, desc="Processing") as pbar:
        while cap.isOpened() and processed < num_frames:
            frame_idx = start_frame + processed * frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            rgb = frame[..., ::-1]
            hand_data = detector.detect_single(rgb)

            bgr = frame.copy()

            if hand_data is None:
                data.append(None)
                wrist_trajectory.append(None)
                cv2.putText(bgr, "NO HAND", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                overlay_writer.write(bgr)
                processed += 1
                pbar.update(1)
                continue

            detected += 1
            joint_pos = hand_data["joint_pos"]
            wrist_pos = hand_data["wrist_pos_raw"]
            wrist_rot = hand_data["wrist_rot_mano"]

            from scipy.spatial.transform import Rotation as R
            wrist_quat = R.from_matrix(wrist_rot).as_quat()

            # 执行重定向优化
            qpos = retargeter.retarget(joint_pos, wrist_pos, wrist_quat)

            # 接触保持滤波 + 轨迹平滑
            finger_dist = ArmMotionAnalyzer.compute_finger_aperture(joint_pos)
            finger_distances.append(finger_dist)
            filtered_qpos, is_grasping = contact_filter.update(finger_dist, qpos)
            smoothed_qpos = trajectory_smoother.smooth(filtered_qpos)

            # 计算手腕6DoF位姿
            arm_info = arm_analyzer.compute_wrist_6dof(hand_data)

            data.append(smoothed_qpos)
            wrist_trajectory.append(
                {
                    "wrist_pos": arm_info["wrist_pos"],
                    "wrist_quat": arm_info["wrist_quat"],
                    "ee_pos": arm_info["ee_pos"],
                    "ee_quat": arm_info["ee_quat"],
                    "finger_distance": finger_dist,
                    "is_grasping": is_grasping,
                    "frame_idx": frame_idx,
                }
            )

            # 在叠加视频上绘制手部骨架和抓取状态
            detector.draw_skeleton_on_image(bgr, hand_data["keypoint_2d"])
            status = "GRASP" if is_grasping else "OPEN"
            color = (0, 255, 0) if is_grasping else (0, 0, 255)
            cv2.putText(
                bgr,
                f"{status} d={finger_dist:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            overlay_writer.write(bgr)

            if visualize:
                cv2.imshow("egocentric_retargeting", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            pbar.update(1)

    cap.release()
    overlay_writer.release()
    if visualize:
        cv2.destroyAllWindows()

    # 统计检测结果
    valid = sum(1 for d in data if d is not None)
    print(f"\n  Detection: {valid}/{processed} ({valid/max(processed,1)*100:.1f}%)")

    if finger_distances:
        fd = np.array(finger_distances)
        print(f"  Finger distance: mean={fd.mean():.4f}, std={fd.std():.4f}, "
              f"min={fd.min():.4f}, max={fd.max():.4f}")
        # 第一人称视角下手指距离通常较小，建议调整阈值
        if grasp_threshold is None and fd.max() < 0.04:
            print(f"  NOTE: Finger distances are small (egocentric view).")
            print(f"        Consider using --grasp-threshold {fd.mean()*0.6:.4f} "
                  f"--release-threshold {fd.mean()*0.9:.4f}")

    # 步骤2：保存轨迹数据
    print("[2/3] Saving trajectory...")
    with pkl_path.open("wb") as f:
        pickle.dump(
            {
                "data": data,
                "wrist_trajectory": wrist_trajectory,
                "meta_data": meta_data,
                "video_info": {
                    "video_path": str(video_path),
                    "start_frame": start_frame,
                    "num_frames": processed,
                    "fps": fps,
                    "frame_skip": frame_skip,
                },
            },
            f,
        )
    print(f"  Saved: {pkl_path}")
    print(f"  Saved: {overlay_path}")

    # 步骤3：渲染机器人手部视频
    if render:
        print("[3/3] Rendering robot hand video...")
        sapien_ok = False
        try:
            import subprocess
            # 检查 SAPIEN 是否可用（需要 GPU 和 Vulkan 支持）
            check_result = subprocess.run(
                [sys.executable, "-c", "import sapien; sapien.Scene()"],
                capture_output=True, timeout=15,
            )
            if check_result.returncode != 0:
                print(f"  SAPIEN not functional (no GPU/rendering device)")
                print(f"  Falling back to software rendering...")
            else:
                # 使用 SAPIEN 3D 渲染
                from render_trajectory import render_trajectory
                render_trajectory(str(pkl_path), str(mp4_path), headless=True, fps=fps)
                sapien_ok = True
                print(f"  Saved (SAPIEN): {mp4_path}")
        except Exception as e:
            print(f"  SAPIEN check failed: {e}")
            print(f"  Falling back to software rendering...")

        # SAPIEN 不可用时回退到软件渲染
        if not sapien_ok:
            render_software_trajectory(str(pkl_path), str(software_render_path), fps=fps)
            print(f"  Saved (software): {software_render_path}")
    else:
        print("[3/3] Skipping render (--no-render)")

    print()
    print("=" * 70)
    print("  Pipeline complete!")
    print(f"  Trajectory: {pkl_path}")
    print(f"  Overlay:    {overlay_path}")
    if render:
        if sapien_ok:
            print(f"  SAPIEN:     {mp4_path}")
        else:
            print(f"  Robot hand: {software_render_path}")
    print("=" * 70)

    # 输出重定向优化器的统计信息
    retargeter.retargeting.verbose()


def main():
    """命令行入口：解析参数并运行管道"""
    parser = argparse.ArgumentParser(
        description="Egocentric Video -> Robot Hand Retargeting Pipeline"
    )
    parser.add_argument("--video-path", type=str, required=True, help="Input MP4 video")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--robot",
        type=str,
        default="allegro",
        choices=[r.name for r in RobotName],
        help="Robot hand type",
    )
    parser.add_argument(
        "--hand",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Hand to track",
    )
    parser.add_argument(
        "--retargeting",
        type=str,
        default="position",
        choices=["position", "vector", "dexpilot"],
        help="Retargeting algorithm",
    )
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame")
    parser.add_argument("--num-frames", type=int, default=-1, help="Number of frames (-1=all)")
    parser.add_argument(
        "--no-selfie",
        action="store_true",
        help="Disable selfie mode (use for non-egocentric videos where image is not mirrored)",
    )
    parser.add_argument(
        "--grasp-threshold",
        type=float,
        default=None,
        help="Finger distance threshold for grasp detection (default: 0.04)",
    )
    parser.add_argument(
        "--release-threshold",
        type=float,
        default=None,
        help="Finger distance threshold for release detection (default: 0.06)",
    )
    parser.add_argument("--no-visualize", action="store_true", help="Disable detection visualization")
    parser.add_argument("--no-render", action="store_true", help="Skip SAPIEN rendering")

    args = parser.parse_args()

    retargeting_map = {
        "position": RetargetingType.position,
        "vector": RetargetingType.vector,
        "dexpilot": RetargetingType.dexpilot,
    }

    run_full_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        robot_name=RobotName[args.robot],
        hand_type=HandType[args.hand],
        retargeting_type=retargeting_map[args.retargeting],
        fps=args.fps,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        selfie=not args.no_selfie,
        grasp_threshold=args.grasp_threshold,
        release_threshold=args.release_threshold,
        visualize=not args.no_visualize,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
