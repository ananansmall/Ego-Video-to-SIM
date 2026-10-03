#!/usr/bin/env python3
"""
映射夹爪开合（MANO 拇指↔食指 → 夹爪开合）:
  输入: MANO 21 关节 (7a 数据集, preprocess_traj.npz)
        j4 = 拇指尖, j8 = 食指尖 (GLB = SAPIEN 仿真系)
  指距 = ‖j8 - j4‖                              # MANO 拇指尖↔食指尖距离 (GLB系, ~0.1379~0.1565m)
  真实指距 = 指距 / scale_ratio                 # scale_ratio=2.5458 (来自 transform_params.npz), 换算到仿真真实尺度
  j1 = j2 = clip((真实指距 - 0.01) / 2, 0, 0.05) # 减 URDF 初始间隙 1cm, 除2, 限幅 0~5cm
  gap(画面叠的开合距离) = 0.01 + 2*j1           # URDF 几何推导: 两手指内表面间距, q=0 时 gap=0.01m
  7a 实测: j1 ∈ [22.09, 25.74]mm (span 3.6mm), gap ∈ [54.18, 61.48]mm (span 7.3mm) — 变化极小
本脚本: 逐帧 set_drive_target(j1[i]) → physics_step (PD 物理控制) → render → 叠字 → mp4
"""
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parents[1]  # .../combination/
sys.path.insert(0, str(_BASE / "trajectory_optimization"))

import argparse
import shutil
import subprocess

import cv2
import numpy as np
import sapien
from scipy.spatial.transform import Rotation

from physics_env import (
    setup_physics_scene, setup_robot, physics_step,
    prepare_gripper_only_urdf, logger,
)
from physics_utils import (
    DECIMATION, PHYSICS_TIMESTEP,
    GRIPPER_INIT_OPEN, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
)
from data_loader import CAM_WIDTH, CAM_HEIGHT, make_look_at_camera

TRAJ_PATH = _BASE / "trajectory_optimization" / "output" / "hawor7a_left" / "preprocess_traj.npz"
TRANSFORM_PATH = _BASE / "trajectory_optimization" / "output" / "hawor7a_left" / "alignment" / "transform_params.npz"
SCALE_RATIO = 2.54577381576609
BASE_FRAME = 50  # F50 姿态, 手掌朝下, 最适合看开合
WARMUP_STEPS = 30
FINGER_BASE_X = 0.03689  # URDF: finger joint origin at base + X 0.03689
CAM_FOCAL = 900.0  # 特写放大
FPS = 30


def to_h264(path: Path) -> bool:
    """用 ffmpeg 转 H.264 (yuv420p + faststart), 兼容主流播放器."""
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart"]
        tmp = str(path).replace(".mp4", "_h264.mp4")
        cmd.append(tmp)
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            Path(path).unlink()
            Path(tmp).rename(path)
            return True
    # 兜底: 系统无 ffmpeg 时用 imageio_ffmpeg 自带的
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-y", "-i", str(path), "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart"]
        tmp = str(path).replace(".mp4", "_h264.mp4")
        cmd.append(tmp)
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            Path(path).unlink()
            Path(tmp).rename(path)
            return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_BASE / "View" / "gripper_open_close_7a.mp4"))
    parser.add_argument("--side", default="left")
    args = parser.parse_args()

    OUT_MP4 = Path(args.out)
    OUT_MP4.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # [1] 加载数据
    # ----------------------------------------------------------
    npz = np.load(TRAJ_PATH, allow_pickle=True)
    j1 = np.asarray(npz["j1"], dtype=np.float64)  # (113,)
    grip_pos = np.asarray(npz["grip_pos"])        # (113,3)
    grip_R = np.asarray(npz["grip_R"])            # (113,3,3)
    GAP = 0.01 + 2 * j1                            # 画面叠的开合距离

    # 自检: 用 transform_params.npz 的 scale_ratio 重算公式
    tp = np.load(TRANSFORM_PATH, allow_pickle=True)
    sr = float(tp["scale_ratio"])
    joints = np.asarray(npz["joints"])  # (113, 21, 3)
    j4 = joints[:, 4, :]  # 拇指尖
    j8 = joints[:, 8, :]  # 食指尖
    finger_dist = np.linalg.norm(j8 - j4, axis=1)  # GLB 系
    real_dist = finger_dist / sr
    j1_recalc = np.clip((real_dist - 0.01) / 2, 0, 0.05)
    logger.info(f"  自检: j1 max abs err = {np.max(np.abs(j1 - j1_recalc)):.8f}")

    print(f"j1: min={j1.min()*1e3:.2f}mm max={j1.max()*1e3:.2f}mm span={(j1.max()-j1.min())*1e3:.2f}mm")
    print(f"gap: min={GAP.min()*1e3:.2f}mm max={GAP.max()*1e3:.2f}mm span={(GAP.max()-GAP.min())*1e3:.2f}mm")
    print(f"scale_ratio={sr:.6f}, finger_dist(GLB)=[{finger_dist.min():.4f},{finger_dist.max():.4f}]m")

    # ----------------------------------------------------------
    # [2] 场景 + 夹爪 (固定 base, 选项A: 最纯净)
    # ----------------------------------------------------------
    scene = setup_physics_scene(ground_height=0.0, force_cpu=False)
    render_ok = getattr(scene, "_render_available", False)
    logger.info(f"  render_available={render_ok}")

    base_pos = np.array([0.0, 0.0, 0.10])
    # grip_R[50] → SAPIEN quat [w,x,y,z]
    r = Rotation.from_matrix(grip_R[BASE_FRAME])
    qv = r.as_quat()  # scipy 返回 [x,y,z,w]
    base_quat = np.array([qv[3], qv[0], qv[1], qv[2]])  # 转 [w,x,y,z]

    info = setup_robot(scene, "gripper_only", args.side, base_pos, base_quat)
    robot = info["robot"]
    aj = robot.get_active_joints()
    gi1, gi2 = info["gripper_indices"][args.side]
    qpos_starts = info["qpos_starts"]
    logger.info(f"  gripper indices: gi1={gi1}, gi2={gi2}, qpos_starts={qpos_starts}")

    # ----------------------------------------------------------
    # [3] 初始 q = j1[0] (关键坑: 不是 GRIPPER_INIT_OPEN=0.04)
    # ----------------------------------------------------------
    q0 = float(j1[0])
    # 用 init_qpos 做基底, 保留虚拟关节的 base_pos/base_quat 值
    qpos0 = np.asarray(info["init_qpos"], dtype=np.float64).copy()
    qpos0[qpos_starts[gi1]] = q0
    qpos0[qpos_starts[gi2]] = q0
    robot.set_qpos(qpos0)
    robot.set_qvel(np.zeros_like(qpos0))
    aj[gi1].set_drive_target(q0)
    aj[gi2].set_drive_target(q0)
    logger.info(f"  初始 q = j1[0] = {q0*1e3:.2f}mm (非 GRIPPER_INIT_OPEN={GRIPPER_INIT_OPEN*1e3:.0f}mm)")

    # ----------------------------------------------------------
    # [4] 相机 + VideoWriter
    # ----------------------------------------------------------
    finger_mid = base_pos + grip_R[BASE_FRAME] @ np.array([FINGER_BASE_X, 0, 0])
    cam_pos = finger_mid + np.array([0.14, -0.14, 0.10])  # 前侧俯视 ~0.21m
    cam = scene.add_camera("cam", CAM_WIDTH, CAM_HEIGHT, 1.0, 0.01, 100.0)
    q_cam = make_look_at_camera(cam_pos, finger_mid, np.array([0., 0., 1.]))
    cam.set_local_pose(sapien.Pose(cam_pos.tolist(), q_cam.tolist()))
    cam.set_focal_lengths(CAM_FOCAL, CAM_FOCAL)
    logger.info(f"  cam_pos={cam_pos.round(3)}, finger_mid={finger_mid.round(3)}, focal={CAM_FOCAL}")

    writer = cv2.VideoWriter(str(OUT_MP4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (CAM_WIDTH, CAM_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 无法打开 {OUT_MP4}")

    # ----------------------------------------------------------
    # [5] Warmup (让 PD 稳定到 j1[0])
    # ----------------------------------------------------------
    logger.info(f"  warmup: {WARMUP_STEPS} steps × DECIMATION={DECIMATION} sub-steps")
    for _ in range(WARMUP_STEPS):
        physics_step(robot, [], gi1, gi2, np.array([]), q0, q0, scene)

    # ----------------------------------------------------------
    # [6] 主循环 113 帧
    # ----------------------------------------------------------
    logs = []
    for i in range(len(j1)):
        t = float(j1[i])
        physics_step(robot, [], gi1, gi2, np.array([]), t, t, scene)
        q_act = float(np.array(robot.get_qpos())[qpos_starts[gi1]])

        if render_ok:
            try:
                scene.update_render()
                cam.take_picture()
                rgb = cam.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray(
                    (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]
                )
                w = bgr.shape[1]
                cv2.rectangle(bgr, (0, 0), (w, 90), (0, 0, 0), -1)
                cv2.putText(bgr, f"F{i}/{len(j1)}", (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(bgr, f"j1={t*1e3:.2f}mm  j2={t*1e3:.2f}mm", (15, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(bgr,
                            f"gap={GAP[i]*1e3:.2f}mm  q_act={q_act*1e3:.2f}mm  lag={(q_act-t)*1e3:+.2f}mm",
                            (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                writer.write(bgr)
            except RuntimeError as e:
                logger.warning(f"render lost, switch to no-render: {e}")
                render_ok = False
                # 降级后写黑帧 + 叠字, 保证 mp4 生成
                bgr = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
                cv2.rectangle(bgr, (0, 0), (CAM_WIDTH, 90), (0, 0, 0), -1)
                cv2.putText(bgr, f"F{i}/{len(j1)}  [no-render]", (15, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                writer.write(bgr)
                continue
        else:
            # 无渲染: 黑帧 + 叠字
            bgr = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            cv2.rectangle(bgr, (0, 0), (CAM_WIDTH, 90), (0, 0, 0), -1)
            cv2.putText(bgr, f"F{i}/{len(j1)}  [no-render]", (15, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(bgr, f"j1={t*1e3:.2f}mm  j2={t*1e3:.2f}mm", (15, 54),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(bgr,
                        f"gap={GAP[i]*1e3:.2f}mm  q_act={q_act*1e3:.2f}mm  lag={(q_act-t)*1e3:+.2f}mm",
                        (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            writer.write(bgr)

        logs.append((i, t, q_act))

    writer.release()

    # ----------------------------------------------------------
    # [7] H.264 转码
    # ----------------------------------------------------------
    if to_h264(OUT_MP4):
        logger.info(f"  H.264 转码完成")
    else:
        logger.warning(f"  H.264 转码失败, 保留 mp4v 编码")

    # ----------------------------------------------------------
    # [8] 汇总打印
    # ----------------------------------------------------------
    lags = [(q - t) * 1e3 for _, t, q in logs]
    print(f"PD lag: mean={np.mean(lags):.3f}mm max={np.max(np.abs(lags)):.3f}mm")
    print(f"frames: {len(logs)}, render_ok={render_ok}")
    sz = OUT_MP4.stat().st_size
    print(f"output: {OUT_MP4} ({sz/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
