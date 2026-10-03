#!/usr/bin/env python3
"""
第一人称视频 -> R1 机器人 SAPIEN 仿真管道

完整流程：
  0. 预扫描视频自动检测手别 (Auto Hand Detection)
  1. 从第一人称视频中提取手部姿态 (MediaPipe Hands)
  2. 将人手关节映射到 R1 夹爪 (dex_retargeting, SeqRetargeting + fixed_qpos + 物理限制钳位)
  3. 从手腕6DoF计算R1臂关节角度 (RelaxedIK 子进程)
  4. 在 SAPIEN 物理引擎中驱动 R1 全身机器人执行动作
  5. 离线渲染生成仿真视频（含桌面、环境光照、状态叠加、监测变量HUD）
  6. 记录所有关键参数到日志文件和 JSON 文件

关键设计：
  - 使用完整 R1 机器人 URDF（含底座、躯干、双臂、夹爪），而非仅夹爪 URDF
  - 子进程架构：SAPIEN 渲染和 IK 求解各自独立运行，避免 OpenGL/MediaPipe 上下文冲突
  - 自动手别检测：预扫描视频帧，统计 MediaPipe 检测到的左右手频次
  - 夹爪物理限制钳位：joint1 in [0, 0.05], joint2 in [-0.05, 0]
  - 非线性夹爪映射：使用幂律缩放使夹爪开合更自然
  - 检测间隙填充：短暂丢失时使用上一帧数据插值（最大 5 帧）
  - 相机动态定位：根据臂工作区域中心自动计算相机位置和朝向
  - 自适应工作空间映射：根据手腕轨迹范围调整舒适目标
  - FK 验证：渲染时计算末端执行器实际位置与 IK 目标的误差
  - 监测变量 HUD：视频叠加关节角度、末端位置、FK误差、帧率、手腕坐标等
  - Warmup 过渡：从初始姿态平滑过渡到第一帧IK结果
  - 步骤控制：--steps 参数可选择性执行 detect/ik/render 步骤
  - 自动输出目录：--output-dir 可选，默认根据视频文件名自动生成

视角模式 (--view)：
  - behind:  机器人背后，相机在+X侧看向臂工作区域
  - front:   机器人前方，相机在-X侧看向臂工作区域
  - topdown: 高空俯视，相机在+Z侧
  - side:    侧面视角，相机在+Y侧
  - closeup: 特写视角，聚焦末端执行器区域

使用示例：
  conda run -n dex python r1_sapien_sim.py \\
    --video-path /home/an/data/MP4/egodex_small/test/basic_pick_place/7.mp4 \\
    --view closeup

  # 仅执行检测步骤
  conda run -n dex python r1_sapien_sim.py \\
    --video-path video.mp4 --steps detect

  # 跳过渲染
  conda run -n dex python r1_sapien_sim.py \\
    --video-path video.mp4 --steps detect,ik
"""

import os
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "disable")
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
_vulkan_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
if os.path.exists(_vulkan_icd) and "VK_ICD_FILENAMES" not in os.environ:
    os.environ["VK_ICD_FILENAMES"] = _vulkan_icd

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if not (PROJECT_ROOT / "assets" / "robots" / "hands").exists():
    _alt = Path("/home/an/robot_world_ws/src/dex-retargeting")
    if (_alt / "assets" / "robots" / "hands").exists():
        PROJECT_ROOT = _alt
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(GALAXEA_SIM_PATH))

from dex_retargeting.constants import RobotName, RetargetingType, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from egocentric_hand_detector import EgocentricHandDetector
from arm_motion_analyzer import ArmMotionAnalyzer
from robot_hand_retargeter import RobotHandRetargeter
from contact_preservation import ContactPreservationFilter, TrajectorySmoother

R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"
R1_FULL_URDF_PATH = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "robot_glb.urdf"

LP_ALPHA_EE = 0.85
LP_ALPHA_JOINT = 0.5
Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])
ARM_BASE_OFFSET_RIGHT = np.array([-0.03207, -0.09724, 1.40347])
ARM_BASE_OFFSET_LEFT = np.array([-0.03207, 0.09724, 1.40347])
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.40
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])
SAFETY_DISTANCE = 0.075
WARMUP_FRAMES = 30

R1_RIGHT_JOINT_LIMITS = np.array([
    [-2.0944, 2.0944],
    [0.0, 3.1416],
    [-2.8798, 0.0],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-2.8798, 2.8798],
])

R1_LEFT_JOINT_LIMITS = np.array([
    [-2.0944, 2.0944],
    [0.0, 3.1416],
    [-2.8798, 0.0],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-2.8798, 2.8798],
])

R1_STARTING_CONFIG_RIGHT = np.array([-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985])
R1_STARTING_CONFIG_LEFT = np.array([1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985])

MAX_JOINT_VEL_DEG_PER_FRAME = 8.0

GRIPPER_LIMITS = {
    "right_gripper_finger_joint1": (0.0, 0.05),
    "right_gripper_finger_joint2": (-0.05, 0.0),
    "left_gripper_finger_joint1": (0.0, 0.05),
    "left_gripper_finger_joint2": (-0.05, 0.0),
}

MAX_GAP_FILL_FRAMES = 5
OTHER_ARM_DEFAULT = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]


def _joint_comfort_score(joint_values, joint_limits):
    mid = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
    half_range = (joint_limits[:, 1] - joint_limits[:, 0]) / 2
    normalized_dist = np.abs(joint_values - mid) / half_range
    return float(1.0 - np.mean(normalized_dist))


def clamp_arm_joints(joints, joint_limits):
    return np.clip(joints, joint_limits[:, 0], joint_limits[:, 1])


def limit_joint_velocity(joints, prev_joints, max_vel_deg=MAX_JOINT_VEL_DEG_PER_FRAME):
    max_vel_rad = np.radians(max_vel_deg)
    delta = joints - prev_joints
    clamped_delta = np.clip(delta, -max_vel_rad, max_vel_rad)
    return prev_joints + clamped_delta


def setup_logger(output_dir):
    log_path = Path(output_dir) / "pipeline_info.log"
    logger = logging.getLogger("r1_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def auto_detect_hand_type(video_path, selfie=True, sample_count=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  [WARN] Cannot open video for hand detection, defaulting to Right")
        return "Right"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, max(total_frames - 1, 0), min(sample_count, total_frames)).astype(int)
    import mediapipe as mp
    detector = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5,
    )
    right_count = 0
    left_count = 0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = frame[..., ::-1]
        results = detector.process(rgb)
        if results.multi_hand_landmarks:
            for i in range(len(results.multi_hand_landmarks)):
                label = results.multi_handedness[i].ListFields()[0][1][0].label
                if selfie:
                    if label == "Right":
                        left_count += 1
                    elif label == "Left":
                        right_count += 1
                else:
                    if label == "Right":
                        right_count += 1
                    elif label == "Left":
                        left_count += 1
    detector.close()
    cap.release()
    print(f"  Auto hand detection (selfie={selfie}): MediaPipe Right→actual Left, Left→actual Right")
    print(f"  Actual hand counts: Right={right_count}, Left={left_count}")
    if left_count > right_count * 1.5:
        detected = "Left"
    elif right_count > 0:
        detected = "Right"
    else:
        detected = "Right"
    print(f"  Detected hand type: {detected}")
    return detected


def clamp_gripper(qpos, hand_type="right"):
    joint1_name = f"{hand_type}_gripper_finger_joint1"
    joint2_name = f"{hand_type}_gripper_finger_joint2"
    if joint1_name in GRIPPER_LIMITS:
        lo, hi = GRIPPER_LIMITS[joint1_name]
        qpos[0] = np.clip(qpos[0], lo, hi)
    if joint2_name in GRIPPER_LIMITS:
        lo, hi = GRIPPER_LIMITS[joint2_name]
        qpos[1] = np.clip(qpos[1], lo, hi)
    return qpos


# ============================================================================
# SAPIEN 渲染子进程脚本
# 改进点：
#   1. 相机动态定位：基于臂工作区域中心，距离自适应手腕运动范围
#   2. 监测变量HUD增强：分区域显示（Gripper/Arm Joints/Kinematics）
#      包含手腕坐标、IK目标坐标、臂基座坐标、到达距离、映射偏移等
#   3. FK验证：计算末端执行器实际位置与IK目标的误差
#   4. Warmup过渡：从初始姿态平滑插值到第一帧IK结果
#   5. 工作空间映射（参照r1_hand_tracking_video.py）
#   6. 关节舒适度评估
#   7. closeup视角：聚焦末端执行器特写
# ============================================================================
_SAPIEN_RENDER_SCRIPT = '''#!/usr/bin/env python3
"""SAPIEN 离线渲染子进程 - 加载完整 R1 机器人并渲染仿真视频

改进点：
  - 相机聚焦臂基座+末端执行器区域，确保手部动作清晰可见
  - 监测变量HUD叠加在视频上（关节角度、末端位置、FK误差、帧率）
  - FK验证：计算末端执行器实际位置与IK目标的误差
  - Warmup过渡：从初始姿态平滑插值到第一帧IK结果
  - 关节舒适度评估
"""
import os
os.environ["MUJOCO_GL"] = "disable"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import sys
import pickle
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tqdm
import sapien

DEX_RETARGETING_ROOT = Path("/home/an/robot_world_ws/src/dex-retargeting")
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(DEX_RETARGETING_ROOT / "src"))
sys.path.insert(0, str(DEX_RETARGETING_ROOT / "example" / "position_retargeting"))
sys.path.insert(0, str(GALAXEA_SIM_PATH))

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.optimizer_utils import LPFilter

R1_FULL_URDF = os.path.join(GALAXEA_SIM_PATH, "galaxea_sim", "assets", "r1", "robot_glb.urdf")

GRIPPER_LIMITS = {
    "right_gripper_finger_joint1": (0.0, 0.05),
    "right_gripper_finger_joint2": (-0.05, 0.0),
    "left_gripper_finger_joint1": (0.0, 0.05),
    "left_gripper_finger_joint2": (-0.05, 0.0),
}

Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])
OTHER_ARM_DEFAULT = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
ARM_BASE_OFFSET_RIGHT = np.array([-0.03207, -0.09724, 1.40347])
ARM_BASE_OFFSET_LEFT = np.array([-0.03207, 0.09724, 1.40347])
ARM_MAX_REACH = 0.713
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])
SAFETY_DISTANCE = 0.075
WARMUP_FRAMES = 30
LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5

R1_JOINT_LIMITS = np.array([
    [-2.0944, 2.0944],
    [0.0, 3.1416],
    [-2.8798, 0.0],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-2.8798, 2.8798],
])

R1_STARTING_CONFIG_RIGHT = np.array([-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985])
R1_STARTING_CONFIG_LEFT = np.array([1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985])


def clamp_gripper_value(val, joint_name):
    if joint_name in GRIPPER_LIMITS:
        lo, hi = GRIPPER_LIMITS[joint_name]
        return np.clip(val, lo, hi)
    return val


def clamp_arm_joints(joints, joint_limits):
    return np.clip(joints, joint_limits[:, 0], joint_limits[:, 1])


def joint_comfort_score(joint_values, joint_limits):
    mid = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
    half_range = (joint_limits[:, 1] - joint_limits[:, 0]) / 2
    normalized_dist = np.abs(joint_values - mid) / half_range
    return float(1.0 - np.mean(normalized_dist))


def look_at_quaternion(camera_pos, target_pos):
    from pytransform3d import rotations as pr
    forward = np.array(target_pos) - np.array(camera_pos)
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = up
    R[:, 2] = -forward
    q_wxyz = pr.quaternion_from_matrix(R)
    return [float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])]


def draw_hud(bgr, frame_i, total_frames, qpos_data, wt_data, fk_error, comfort,
             ee_actual_pos, ee_target_pos, arm_joint_names, hand_type, fps_actual,
             wrist_world=None, ik_target_world=None, arm_base_pos=None,
             mapping_offset=None, reach_dist=None, arm_max_reach=0.713,
             elapsed_sec=0.0, is_warmup=False):
    H, W = bgr.shape[:2]

    phase = "WARMUP" if is_warmup else "DATA"
    cv2.putText(bgr, f"[{phase}] {frame_i+1}/{total_frames}  {fps_actual:.0f}fps  T:{elapsed_sec:.1f}s",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    if wt_data is not None:
        is_grasping = wt_data.get("is_grasping", False)
        finger_dist = wt_data.get("finger_distance", 0)
        status = "GRASP" if is_grasping else "OPEN"
        sc = (0, 255, 0) if is_grasping else (0, 100, 255)
        cv2.putText(bgr, f"{status} d={finger_dist:.3f}", (8, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, sc, 1)

    if fk_error is not None:
        err_cm = fk_error * 100
        ec = (0, 255, 0) if err_cm < 2 else (0, 165, 255) if err_cm < 5 else (0, 0, 255)
        cv2.putText(bgr, f"FK:{err_cm:.1f}cm", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ec, 1)

    if reach_dist is not None:
        reach_pct = reach_dist / arm_max_reach * 100
        rc = (0, 255, 0) if reach_pct < 70 else (0, 165, 255) if reach_pct < 90 else (0, 0, 255)
        cv2.putText(bgr, f"Reach:{reach_pct:.0f}%", (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.35, rc, 1)

    return bgr


MANO_FINGERS_RENDER = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]
MANO_PALM_RENDER = [(5, 9), (9, 13), (13, 17)]
FINGER_COLORS_RENDER = [
    (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)
]
FINGER_NAMES_RENDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
AXIS_COLORS_RENDER = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


def draw_mano_inset(bgr, joint_pos, wrist_rot_mano):
    H, W = bgr.shape[:2]
    inset_size = 200
    inset_x = W - inset_size - 10
    inset_y = H - inset_size - 10

    overlay_inset = bgr.copy()
    cv2.rectangle(overlay_inset, (inset_x, inset_y),
                  (inset_x + inset_size, inset_y + inset_size), (0, 0, 0), -1)
    cv2.addWeighted(overlay_inset, 0.75, bgr, 0.25, 0, bgr)

    if joint_pos is None:
        cv2.putText(bgr, "MANO (no data)", (inset_x + 10, inset_y + inset_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        return bgr

    joint_pos = np.array(joint_pos)
    wrist_rot_mano = np.array(wrist_rot_mano) if wrist_rot_mano is not None else np.eye(3)

    mano_cx = inset_x + inset_size // 2
    mano_cy = inset_y + inset_size // 2 + 15

    ref_dists = []
    for finger in MANO_FINGERS_RENDER:
        for j in range(len(finger) - 1):
            p1 = joint_pos[finger[j]]
            p2 = joint_pos[finger[j + 1]]
            d = np.linalg.norm(p2 - p1)
            ref_dists.append(d)
    scale_factor = np.median(ref_dists) if ref_dists else 0.05
    mano_scale = (inset_size * 0.35) / max(scale_factor, 0.001)

    mano_pts = []
    for i in range(21):
        dx = joint_pos[i][0] * mano_scale
        dy = joint_pos[i][1] * mano_scale
        mano_pts.append((int(mano_cx + dx), int(mano_cy + dy)))

    for fi, finger in enumerate(MANO_FINGERS_RENDER):
        for j in range(len(finger) - 1):
            p1 = mano_pts[finger[j]]
            p2 = mano_pts[finger[j + 1]]
            cv2.line(bgr, p1, p2, FINGER_COLORS_RENDER[fi], 2)
    for p1_idx, p2_idx in MANO_PALM_RENDER:
        cv2.line(bgr, mano_pts[p1_idx], mano_pts[p2_idx], (200, 200, 200), 1)

    for i in range(21):
        cv2.circle(bgr, mano_pts[i], 3, (255, 255, 255), -1)
        cv2.circle(bgr, mano_pts[i], 2, (0, 0, 0), -1)

    mano_axis_len = 35
    for i in range(3):
        dx = int(wrist_rot_mano[0, i] * mano_axis_len)
        dy = int(wrist_rot_mano[1, i] * mano_axis_len)
        ex, ey = mano_cx + dx, mano_cy + dy
        cv2.arrowedLine(bgr, (mano_cx, mano_cy), (ex, ey), AXIS_COLORS_RENDER[i], 2, tipLength=0.15)
        axis_name = ['Xm', 'Ym', 'Zm'][i]
        cv2.putText(bgr, axis_name, (ex + 3, ey - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, AXIS_COLORS_RENDER[i], 1)

    cv2.putText(bgr, "MANO", (inset_x + 5, inset_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    legend_y = inset_y + inset_size - 30
    for fi in range(5):
        lx = inset_x + 5 + fi * 38
        cv2.putText(bgr, FINGER_NAMES_RENDER[fi][:3], (lx, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, FINGER_COLORS_RENDER[fi], 1)

    return bgr


def draw_coord_transform_panel(bgr, wrist_pos_camera, wrist_pos_hand_local,
                                wrist_pos_base, ik_target_world, ik_target_base,
                                base_link_p, base_link_R, hand_type):
    H, W = bgr.shape[:2]
    R_cam2base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    R_cam2base_inv = R_cam2base.T
    base_link_R_inv = np.array(base_link_R).T if base_link_R is not None else np.eye(3)

    panel_w = 500
    panel_h = 320
    panel_x = W - panel_w - 10
    panel_y = 5
    overlay_img = bgr.copy()
    cv2.rectangle(overlay_img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_img, 0.8, bgr, 0.2, 0, bgr)

    y0 = panel_y + 16
    dy = 14
    fs = 0.30
    th = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(bgr, "=== COORD TRANSFORM CHAIN ===", (panel_x + 5, y0), font, 0.38, (0, 255, 255), th)
    y0 += dy + 3

    if wrist_pos_hand_local is not None:
        wp = np.array(wrist_pos_hand_local)
        cv2.putText(bgr, f"[0] Hand-local(world_landmarks):", (panel_x + 5, y0), font, fs, (120, 120, 180), th)
        y0 += dy
        cv2.putText(bgr, f"  X={wp[0]:+.4f} Y={wp[1]:+.4f} Z={wp[2]:+.4f}", (panel_x + 5, y0), font, fs, (120, 120, 180), th)
        y0 += dy
        cv2.putText(bgr, "  (hand-center local, NOT camera!)", (panel_x + 5, y0), font, fs * 0.85, (100, 100, 150), th)
        y0 += dy + 2

    if wrist_pos_camera is not None:
        wc = np.array(wrist_pos_camera)
        cv2.putText(bgr, f"[1] Camera(2D+depth):", (panel_x + 5, y0), font, fs, (180, 180, 255), th)
        y0 += dy
        cv2.putText(bgr, f"  X={wc[0]:+.4f} Y={wc[1]:+.4f} Z={wc[2]:+.4f}", (panel_x + 5, y0), font, fs, (180, 180, 255), th)
        y0 += dy
        cv2.putText(bgr, "  X(right) Y(down) Z(fwd)", (panel_x + 5, y0), font, fs * 0.85, (140, 140, 200), th)
        y0 += dy + 2

    if wrist_pos_base is not None:
        wb = np.array(wrist_pos_base)
        cv2.putText(bgr, f"[2] ArmBase(R_cam2base):", (panel_x + 5, y0), font, fs, (255, 150, 255), th)
        y0 += dy
        cv2.putText(bgr, f"  X={wb[0]:+.4f} Y={wb[1]:+.4f} Z={wb[2]:+.4f}", (panel_x + 5, y0), font, fs, (255, 150, 255), th)
        y0 += dy
        cv2.putText(bgr, "  +X(fwd) +Y(left) +Z(up)", (panel_x + 5, y0), font, fs * 0.85, (200, 120, 200), th)
        y0 += dy + 2

    if ik_target_base is not None:
        itb = np.array(ik_target_base)
        cv2.putText(bgr, f"[3] IK Target(arm base):", (panel_x + 5, y0), font, fs, (255, 200, 150), th)
        y0 += dy
        cv2.putText(bgr, f"  X={itb[0]:+.4f} Y={itb[1]:+.4f} Z={itb[2]:+.4f}", (panel_x + 5, y0), font, fs, (255, 200, 150), th)
        y0 += dy + 4

    cv2.putText(bgr, "=== AXIS LEGEND ===", (panel_x + 5, y0), font, 0.35, (0, 255, 255), th)
    y0 += dy + 2

    axis_ox = panel_x + 60
    axis_oy = y0 + 35
    axis_len = 30
    for i in range(3):
        dx = int([1, 0, 0.4][i] * axis_len)
        dy = int([0, 1, 0.4][i] * axis_len)
        ex, ey = axis_ox + dx, axis_oy + dy
        cv2.arrowedLine(bgr, (axis_ox, axis_oy), (ex, ey), AXIS_COLORS_RENDER[i], 2, tipLength=0.15)
    cv2.putText(bgr, "CAM", (axis_ox - 12, axis_oy - axis_len - 5), font, 0.28, (255, 255, 255), 1)

    axis2_ox = panel_x + 180
    for i in range(3):
        axis_cam = R_cam2base_inv[:, i]
        dx = int(axis_cam[0] * axis_len)
        dy = int(axis_cam[1] * axis_len)
        ex, ey = axis2_ox + dx, axis_oy + dy
        cv2.arrowedLine(bgr, (axis2_ox, axis_oy), (ex, ey), AXIS_COLORS_RENDER[i], 2, tipLength=0.15)
    cv2.putText(bgr, "ARM_BASE", (axis2_ox - 25, axis_oy - axis_len - 5), font, 0.28, (255, 150, 255), 1)

    axis3_ox = panel_x + 310
    for i in range(3):
        axis_cam = R_cam2base[:, i]
        dx = int(axis_cam[0] * axis_len)
        dy = int(axis_cam[1] * axis_len)
        ex, ey = axis3_ox + dx, axis_oy + dy
        cv2.arrowedLine(bgr, (axis3_ox, axis_oy), (ex, ey), AXIS_COLORS_RENDER[i], 2, tipLength=0.15)
    cv2.putText(bgr, "ARM_BASE", (axis3_ox - 25, axis_oy - axis_len - 5), font, 0.28, (255, 150, 255), 1)

    cv2.putText(bgr, "R=X G=Y B=Z", (panel_x + 5, axis_oy + axis_len + 15),
                font, 0.25, (180, 180, 180), 1)

    return bgr


def main():
    global mano_trail_global, mano_trail_rot_global
    mano_trail_global = []
    mano_trail_rot_global = []

    render_pkl_path = sys.argv[1]
    setup_pkl_path = sys.argv[2] if len(sys.argv) > 2 else None
    with open(render_pkl_path, "rb") as f:
        data = pickle.load(f)

    qpos_sequence = data["qpos_sequence"]
    wrist_traj = data["wrist_traj"]
    view = data["view"]
    fps = data["fps"]
    img_width = data["img_width"]
    img_height = data["img_height"]
    output_path = data["output_path"]
    hand_type = data.get("hand_type", "right")
    avg_wrist = np.array(data.get("avg_wrist", [0.3, 0.0, 0.5]))
    wrist_range = data.get("wrist_range", [0.1, 0.1, 0.1])
    wrist_std = data.get("wrist_std", [0.02, 0.02, 0.02])

    views_to_render = data.get("views", [view])
    output_paths = data.get("output_paths", [output_path])

    setup_data = None
    if setup_pkl_path is not None:
        with open(setup_pkl_path, "rb") as f:
            setup_data = pickle.load(f)

    # ========== 1. 创建 SAPIEN 场景 ==========
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_log_level("error")

    scene = sapien.Scene()
    scene.set_timestep(1 / 240)

    from sapien.asset import create_dome_envmap
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_directional_light(np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True)
    scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
    scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))

    ground_mat = sapien.render.RenderMaterial()
    ground_mat.set_base_color(np.array([0.5, 0.5, 0.5, 1]))
    ground_mat.set_roughness(0.7)
    ground_mat.set_metallic(1)
    ground_mat.set_specular(0.04)
    scene.add_ground(-1, render_material=ground_mat)

    table_mat = sapien.render.RenderMaterial()
    table_mat.set_base_color(np.array([0.8, 0.8, 0.8, 1]))
    table_mat.set_roughness(0.9)
    builder = scene.create_actor_builder()
    builder.add_box_collision(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02]))
    builder.add_box_visual(sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02]), material=table_mat)
    for dx, dy in [(0.4, 1.9), (-0.4, 1.9), (0.4, -1.9), (-0.4, -1.9)]:
        builder.add_box_visual(
            sapien.Pose([dx, dy, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=table_mat,
        )
    table = builder.build_static(name="table")
    table.set_pose(sapien.Pose([0.5, 0, 0]))

    # ========== 2. 加载完整 R1 机器人 ==========
    urdf_path = Path(R1_FULL_URDF)
    if not urdf_path.exists():
        print(f"  ERROR: R1 full URDF not found: {urdf_path}")
        sys.exit(1)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
    temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
    temp_path = f"{temp_dir}/{urdf_path.name}"
    robot_urdf.write_xml_file(temp_path)

    r1_robot = loader.load(temp_path)
    active_joints = r1_robot.get_active_joints()
    joint_names = [j.get_name() for j in active_joints]
    print(f"  R1 full robot loaded: {len(active_joints)} active joints")
    print(f"  Joint names: {joint_names}")

    # ========== 3. 设置机器人位置和关节 ==========
    arm_prefix = f"{hand_type}_arm"
    other_arm_prefix = "left_arm" if hand_type == "right" else "right_arm"
    gripper_j1_name = f"{hand_type}_gripper_finger_joint1"
    gripper_j2_name = f"{hand_type}_gripper_finger_joint2"

    arm_indices = [i for i, name in enumerate(joint_names) if arm_prefix in name]
    other_arm_indices = [i for i, name in enumerate(joint_names) if other_arm_prefix in name]
    arm_joint_names = [joint_names[i] for i in arm_indices]

    if gripper_j1_name not in joint_names or gripper_j2_name not in joint_names:
        print(f"  ERROR: Gripper joints not found. Available: {joint_names}")
        sys.exit(1)
    gripper_idx1 = joint_names.index(gripper_j1_name)
    gripper_idx2 = joint_names.index(gripper_j2_name)

    for joint in active_joints:
        joint.set_drive_property(stiffness=1000.0, damping=100.0)

    initial_qpos = r1_robot.get_qpos().copy()
    for j, idx in enumerate(other_arm_indices):
        if j < len(OTHER_ARM_DEFAULT):
            initial_qpos[idx] = OTHER_ARM_DEFAULT[j]

    # 设置目标臂为RelaxedIK的starting_config（而非零位）
    starting_config = R1_STARTING_CONFIG_RIGHT if hand_type == "right" else R1_STARTING_CONFIG_LEFT
    for j, idx in enumerate(arm_indices):
        if j < len(starting_config):
            initial_qpos[idx] = starting_config[j]

    # 设置夹爪为打开状态
    initial_qpos[gripper_idx1] = 0.04
    initial_qpos[gripper_idx2] = -0.04

    r1_robot.set_qpos(initial_qpos)

    base_link = None
    arm_base_link = None
    ee_link = None
    for link in r1_robot.get_links():
        link_name = link.get_name()
        if link_name == "base_link":
            base_link = link
        elif f"{arm_prefix}_base_link" in link_name:
            arm_base_link = link
        elif f"{hand_type}_gripper_link" in link_name:
            ee_link = link

    scene.step()
    scene.update_render()

    # ========== 工作空间映射（参照 r1_hand_tracking_video.py）==========
    R_180Z_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    arm_base_offset_local = ARM_BASE_OFFSET_LEFT if hand_type == "left" else ARM_BASE_OFFSET_RIGHT
    arm_base_offset_world = R_180Z_mat @ arm_base_offset_local

    if setup_data is not None and "robot_root_pos" in setup_data:
        robot_root_pos = np.array(setup_data["robot_root_pos"])
        print(f"  Using pre-render robot_root_pos: [{robot_root_pos[0]:.4f}, {robot_root_pos[1]:.4f}, {robot_root_pos[2]:.4f}]")
    else:
        robot_root_pos = np.array([0.0, 0.0, 0.0])

    r1_robot.set_root_pose(sapien.Pose(robot_root_pos.tolist(), Q_180Z.tolist()))
    print(f"  Robot root: [{robot_root_pos[0]:.3f}, {robot_root_pos[1]:.3f}, {robot_root_pos[2]:.3f}]")

    scene.step()
    scene.update_render()

    if arm_base_link is not None:
        base_link_pose = arm_base_link.get_entity_pose()
        base_link_p = np.array(base_link_pose.p)
        base_link_q = np.array(base_link_pose.q)
        from pytransform3d import rotations as pr
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T
    else:
        base_link_p = robot_root_pos + arm_base_offset_world
        base_link_R = R_180Z_mat
        base_link_R_inv = base_link_R.T

    print(f"  Arm base link: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")

    # ========== Warmup过渡 ==========
    # 找到第一个有效IK帧
    first_ik_joints = None
    first_gripper = [0.04, -0.04]
    first_ik_target_world = None
    for qi in range(len(qpos_sequence)):
        q = qpos_sequence[qi]
        if q is not None and arm_prefix in q:
            first_ik_joints = np.array(q[arm_prefix])
            first_gripper = q["gripper"]
            if "ee_target_base" in q:
                first_ik_target_base = np.array(q["ee_target_base"])
                first_ik_target_world = base_link_R @ first_ik_target_base + base_link_p
            break

    # 生成warmup帧（从初始姿态平滑插值到第一帧IK结果）
    warmup_qpos_list = []
    if first_ik_joints is not None:
        current_arm_joints = np.array([initial_qpos[i] for i in arm_indices])
        ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
        if first_ik_target_world is not None:
            ee_pos_filter.next(first_ik_target_world)
        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        joint_filter.next(current_arm_joints)

        print(f"  Warmup: {WARMUP_FRAMES} frames transition")
        print(f"    Current joints: [{', '.join(f'{np.degrees(j):.1f} deg' for j in current_arm_joints)}]")
        print(f"    Target joints:  [{', '.join(f'{np.degrees(j):.1f} deg' for j in first_ik_joints)}]")

        for w in range(WARMUP_FRAMES):
            t = (w + 1) / WARMUP_FRAMES
            t_smooth = t * t * (3 - 2 * t)
            interp_joints = current_arm_joints * (1 - t_smooth) + first_ik_joints * t_smooth
            interp_joints = joint_filter.next(interp_joints)

            r1_qpos = initial_qpos.copy()
            for j, idx in enumerate(arm_indices):
                r1_qpos[idx] = interp_joints[j]
            r1_qpos[gripper_idx1] = 0.04
            r1_qpos[gripper_idx2] = -0.04
            warmup_qpos_list.append(r1_qpos)

    # ========== 4. 设置相机（参照 r1_hand_tracking_video.py）==========
    robot_root = np.array(r1_robot.get_root_pose().p)

    def get_camera_pose_for_view(v):
        if v == "behind":
            return robot_root + np.array([2.0, 0.0, 1.0]), [0, 0, 0, 1]
        elif v == "front":
            return robot_root + np.array([-2.0, 0.0, 1.0]), [1, 0, 0, 0]
        elif v == "topdown":
            return robot_root + np.array([-0.5, 0.0, 5.3]), [0.7071, 0, 0.7071, 0]
        elif v == "side":
            return robot_root + np.array([0.0, 2.5, 1.5]), [0, 0, 0, 1]
        elif v == "egocentric":
            return robot_root + np.array([0.0, 0.0, 0.0]), [0, 0, 0, 1]
        else:
            return robot_root + np.array([2.0, 0.0, 1.5]), [0, 0, 0, 1]

    camera_pos, camera_quat = get_camera_pose_for_view(view)
    print(f"  Camera: pos=[{camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}], view={view}")
    print(f"  Robot root: [{robot_root[0]:.3f}, {robot_root[1]:.3f}, {robot_root[2]:.3f}]")
    print(f"  Views to render: {views_to_render} ({len(views_to_render)} total)")

    camera = scene.add_camera(
        name="main",
        width=img_width,
        height=img_height,
        fovy=np.deg2rad(60),
        near=0.01,
        far=200.0,
    )
    camera.set_local_pose(sapien.Pose(camera_pos.tolist(), camera_quat))

    # ========== MANO 3D可视化初始化 ==========
    from sapien import internal_renderer as R
    internal_scene = scene.render_system._internal_scene
    context = sapien.render.SapienRenderer()._internal_context

    mat_mano_hand = context.create_material(
        np.array([0.96, 0.75, 0.69, 1.0]),
        np.array([0.1, 0.1, 0.1, 1.0]), 0.5, 0.0, 0
    )
    mat_mano_finger = [
        context.create_material(np.array([1.0, 0.2, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
        context.create_material(np.array([1.0, 0.6, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
        context.create_material(np.array([0.2, 1.0, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
        context.create_material(np.array([0.2, 0.4, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0),
        context.create_material(np.array([0.8, 0.2, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0),
    ]
    mat_gripper = context.create_material(
        np.array([0.3, 0.8, 0.3, 0.9]), np.array([0.2, 0.2, 0.2, 1.0]), 0.3, 0.5, 0
    )
    mat_trail = context.create_material(
        np.array([1.0, 0.8, 0.0, 0.6]), np.zeros(4), 0.5, 0.0, 0
    )
    mat_axis = [
        context.create_material(np.array([1, 0, 0, 1]), np.zeros(4), 0.5, 0.0, 0),
        context.create_material(np.array([0, 1, 0, 1]), np.zeros(4), 0.5, 0.0, 0),
        context.create_material(np.array([0, 0, 1, 1]), np.zeros(4), 0.5, 0.0, 0),
    ]

    mano_3d_nodes = []
    trail_3d_nodes = []
    gripper_3d_nodes = []
    axis_3d_nodes = []
    robot_ee_trail = []
    MANO_FINGERS_3D = [
        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
    ]
    R_cam2base_mat = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
    OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=np.float64)
    OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)
    operator2mano = OPERATOR2MANO_RIGHT if hand_type == "right" else OPERATOR2MANO_LEFT
    mano_offset_world = np.array([0.0, -0.5, 0.0]) if hand_type == "left" else np.array([0.0, 0.5, 0.0])
    from pytransform3d import rotations as pr_rot_render

    def clear_3d_nodes(node_list):
        for _ in range(len(node_list)):
            node = node_list.pop()
            internal_scene.remove_node(node)

    def mat2quat(R_mat):
        trace = R_mat[0, 0] + R_mat[1, 1] + R_mat[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R_mat[2, 1] - R_mat[1, 2]) * s
            y = (R_mat[0, 2] - R_mat[2, 0]) * s
            z = (R_mat[1, 0] - R_mat[0, 1]) * s
        elif R_mat[0, 0] > R_mat[1, 1] and R_mat[0, 0] > R_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_mat[0, 0] - R_mat[1, 1] - R_mat[2, 2])
            w = (R_mat[2, 1] - R_mat[1, 2]) / s
            x = 0.25 * s
            y = (R_mat[0, 1] + R_mat[1, 0]) / s
            z = (R_mat[0, 2] + R_mat[2, 0]) / s
        elif R_mat[1, 1] > R_mat[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R_mat[1, 1] - R_mat[0, 0] - R_mat[2, 2])
            w = (R_mat[0, 2] - R_mat[2, 0]) / s
            x = (R_mat[0, 1] + R_mat[1, 0]) / s
            y = 0.25 * s
            z = (R_mat[1, 2] + R_mat[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R_mat[2, 2] - R_mat[0, 0] - R_mat[1, 1])
            w = (R_mat[1, 0] - R_mat[0, 1]) / s
            x = (R_mat[0, 2] + R_mat[2, 0]) / s
            y = (R_mat[1, 2] + R_mat[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])

    def rotation_from_z_to_dir(direction):
        z = np.array([0, 0, 1], dtype=np.float64)
        d = direction / np.linalg.norm(direction)
        v = np.cross(z, d)
        s = np.linalg.norm(v)
        c = np.dot(z, d)
        if s < 1e-6:
            if c > 0:
                return np.array([1, 0, 0, 0])
            else:
                return np.array([0, 0, 0, 1])
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_mat = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
        return mat2quat(R_mat)

    def add_capsule(p1, p2, radius, mat, node_list):
        mid = ((p1 + p2) / 2).astype(np.float32)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        quat = rotation_from_z_to_dir(direction).astype(np.float32)
        capsule_mesh = context.create_capsule_mesh(radius, length / 2, 8, 4)
        model = context.create_model([capsule_mesh], [mat])
        node = internal_scene.add_node()
        node.set_position(mid)
        node.set_rotation(quat)
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    def add_sphere(pos, radius, mat, node_list):
        sphere_mesh = context.create_uvsphere_mesh(12, 6)
        model = context.create_model([sphere_mesh], [mat])
        node = internal_scene.add_node()
        node.set_position(pos.astype(np.float32))
        node.set_scale(np.array([radius, radius, radius], dtype=np.float32))
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    # ========== 视角相关配置 ==========
    EGO_CAM_OFFSET = np.array([-0.30, 0.0, 0.45])
    EGO_LOOK_OFFSET = np.array([0.10, 0.0, -0.05])
    EGO_SMOOTH_ALPHA = 0.08
    EGO_MANO_OFFSET = {
        "left": np.array([0.0, -0.15, 0.0]),
        "right": np.array([0.0, 0.15, 0.0]),
    }

    def get_mano_offset(v):
        if v == "egocentric":
            return EGO_MANO_OFFSET.get(hand_type, np.array([0.0, 0.15, 0.0]))
        return mano_offset_world

    def compute_mano_scale(v, wrist_pos_camera_wt, ee_pos, ego_smooth_pos):
        if v == "egocentric" and ego_smooth_pos is not None:
            cam_pos_ego = ego_smooth_pos + EGO_CAM_OFFSET
            dist_to_hand = np.linalg.norm(ee_pos - cam_pos_ego)
            return 2.0 * (0.4 / max(dist_to_hand, 0.15))
        depth_val = wrist_pos_camera_wt[2] if wrist_pos_camera_wt is not None else 0.5
        return 2.0 * (0.4 / max(depth_val, 0.15))

    def update_egocentric_camera(ee_pos, ego_smooth_pos):
        if ego_smooth_pos is None:
            ego_smooth_pos = ee_pos.copy()
        ego_smooth_pos = (1 - EGO_SMOOTH_ALPHA) * ego_smooth_pos + EGO_SMOOTH_ALPHA * ee_pos
        cam_pos = ego_smooth_pos + EGO_CAM_OFFSET
        cam_target = ego_smooth_pos + EGO_LOOK_OFFSET
        forward = cam_target - cam_pos
        forward = forward / np.linalg.norm(forward)
        up_hint = np.array([0, 0, 1])
        right = np.cross(forward, up_hint)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        cam_R = np.eye(3)
        cam_R[:, 0] = right
        cam_R[:, 1] = -up
        cam_R[:, 2] = forward
        from scipy.spatial.transform import Rotation as R_scipy_ego
        q = R_scipy_ego.from_matrix(cam_R).as_quat()
        camera.set_local_pose(sapien.Pose(
            cam_pos.astype(np.float32),
            np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
        ))
        return ego_smooth_pos

    def render_mano_3d(joint_pos_arr, wrist_rot_arr, mano_origin, mano_scale, is_grasping):
        wrist_rot_in_base = R_cam2base_mat @ wrist_rot_arr @ operator2mano.T
        wrist_rot_in_world = base_link_R @ wrist_rot_in_base

        joints_world = []
        for i in range(21):
            j_in_base = R_cam2base_mat @ (wrist_rot_arr @ joint_pos_arr[i] * mano_scale)
            joints_world.append(base_link_R @ j_in_base + mano_origin)

        for fi, finger in enumerate(MANO_FINGERS_3D):
            for j in range(len(finger) - 1):
                add_capsule(joints_world[finger[j]], joints_world[finger[j + 1]],
                            0.015, mat_mano_finger[fi], mano_3d_nodes)
        for i in range(21):
            mat = mat_mano_hand if i == 0 else mat_mano_finger[min(i // 4, 4)]
            add_sphere(joints_world[i], 0.025, mat, mano_3d_nodes)

        for ai in range(3):
            axis_end = mano_origin + wrist_rot_in_world[:, ai] * 0.30
            add_capsule(mano_origin, axis_end, 0.014, mat_axis[ai], axis_3d_nodes)
        for mcp_idx in [1, 5, 9, 13, 17]:
            mcp_pos = joints_world[mcp_idx]
            for ai in range(3):
                add_capsule(mcp_pos, mcp_pos + wrist_rot_in_world[:, ai] * 0.06, 0.004, mat_axis[ai], axis_3d_nodes)
        for tip_idx in [4, 8, 12, 16, 20]:
            tip_pos = joints_world[tip_idx]
            for ai in range(3):
                add_capsule(tip_pos, tip_pos + wrist_rot_in_world[:, ai] * 0.05, 0.004, mat_axis[ai], axis_3d_nodes)

        gripper_dir = wrist_rot_in_world[:, 2]
        gripper_right = wrist_rot_in_world[:, 0]
        gripper_open = 0.04 if is_grasping else 0.08
        for sign in [1, -1]:
            finger_base = mano_origin + gripper_right * sign * gripper_open
            add_capsule(finger_base, finger_base + gripper_dir * 0.12, 0.01, mat_gripper, gripper_3d_nodes)

        palm_mesh = context.create_box_mesh()
        model = context.create_model([palm_mesh], [mat_gripper])
        node = internal_scene.add_node()
        node.set_position(mano_origin.astype(np.float32))
        node.set_rotation(mat2quat(wrist_rot_in_world).astype(np.float32))
        node.set_scale(np.array([0.04, 0.03, 0.015], dtype=np.float32))
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        gripper_3d_nodes.append(node)

        return joints_world, wrist_rot_in_world

    def render_robot_axes_and_trails(ee_pos, ee_R):
        for link in r1_robot.get_links():
            link_name = link.get_name()
            if arm_prefix in link_name and "base_link" not in link_name:
                link_pose = link.get_entity_pose()
                link_pos = np.array(link_pose.p).astype(np.float32)
                link_R = pr_rot_render.matrix_from_quaternion(np.array(link_pose.q))
                for ai in range(3):
                    add_capsule(np.array(link_pos), np.array(link_pos) + link_R[:, ai] * 0.10, 0.005, mat_axis[ai], axis_3d_nodes)

        if ee_R is not None:
            for ai in range(3):
                add_capsule(ee_pos, ee_pos + ee_R[:, ai] * 0.15, 0.008, mat_axis[ai], axis_3d_nodes)

        robot_ee_trail.append(ee_pos.copy())
        if len(robot_ee_trail) > 200:
            robot_ee_trail.pop(0)
        if len(robot_ee_trail) > 2:
            step = max(1, len(robot_ee_trail) // 80)
            for i in range(step, len(robot_ee_trail), step):
                alpha = i / len(robot_ee_trail)
                trail_mat = context.create_material(
                    np.array([0.0, 0.5 * alpha, 1.0 * alpha, 0.4 + 0.6 * alpha]),
                    np.zeros(4), 0.5, 0.0, 0)
                add_capsule(robot_ee_trail[i - step], robot_ee_trail[i], 0.008, trail_mat, trail_3d_nodes)
            add_sphere(robot_ee_trail[-1], 0.025,
                       context.create_material(np.array([0, 0.5, 1, 0.8]), np.zeros(4), 0.5, 0.0, 0),
                       trail_3d_nodes)

        mano_trail_global.append(ee_pos.copy())
        if len(mano_trail_global) > 200:
            mano_trail_global.pop(0)
        if len(mano_trail_global) > 2:
            step = max(1, len(mano_trail_global) // 80)
            for i in range(step, len(mano_trail_global), step):
                alpha = i / len(mano_trail_global)
                trail_mat = context.create_material(
                    np.array([1.0 * alpha, 0.8 * alpha, 0.0, 0.4 + 0.6 * alpha]),
                    np.zeros(4), 0.5, 0.0, 0)
                add_capsule(mano_trail_global[i - step], mano_trail_global[i], 0.008, trail_mat, trail_3d_nodes)
            add_sphere(mano_trail_global[-1], 0.025, mat_trail, trail_3d_nodes)

    def draw_mano_2d_annotations(bgr, wt, ee_pos, ee_R, mano_origin, mano_scale, ego_smooth_pos):
        joint_pos_wt_ann = wt.get("joint_pos")
        wrist_rot_mano_wt_ann = wt.get("wrist_rot_mano")
        if joint_pos_wt_ann is None:
            return
        try:
            K_ann = np.array(camera.get_intrinsic_matrix())
            fx_ann, fy_ann = K_ann[0, 0], K_ann[1, 1]
            cx_ann, cy_ann = K_ann[0, 2], K_ann[1, 2]
            cam_ext_ann = np.array(camera.get_extrinsic_matrix())
            R_c2w_ann = cam_ext_ann[:3, :3].T
            t_c2w_ann = -R_c2w_ann @ cam_ext_ann[:3, 3]

            jp_arr = np.array(joint_pos_wt_ann)
            wr_arr = np.array(wrist_rot_mano_wt_ann) if wrist_rot_mano_wt_ann is not None else np.eye(3)
            mano_origin_w = ee_pos + get_mano_offset(current_view)

            depth_val_ann = wt.get("wrist_pos_camera")
            if current_view == "egocentric" and ego_smooth_pos is not None:
                depth_ann = np.linalg.norm(ee_pos - (ego_smooth_pos + EGO_CAM_OFFSET))
            else:
                depth_ann = depth_val_ann[2] if depth_val_ann is not None else 0.5
            mano_scale_ann = np.clip(2.0 * (0.4 / max(depth_ann, 0.15)), 1.2, 4.0)

            JOINT_NAMES = ["W","T1","T2","T3","T4","I1","I2","I3","I4",
                           "M1","M2","M3","M4","R1","R2","R3","R4","P1","P2","P3","P4"]
            FINGER_COLORS = [(0,0,255)]*5 + [(0,165,255)]*5 + [(0,255,0)]*5 + [(255,0,0)]*5 + [(255,0,255)]*5
            FINGER_TIP_IDS = [4, 8, 12, 16, 20]
            MCP_IDS = [1, 5, 9, 13, 17]
            TIP_COLORS = [(0,0,255),(0,165,255),(0,255,0),(255,0,0),(255,0,255)]
            TIP_NAMES = ["Thumb","Index","Middle","Ring","Pinky"]

            mano_joints_2d = []
            for idx in range(21):
                j_in_base = R_cam2base_mat @ (wr_arr @ jp_arr[idx] * mano_scale_ann)
                j_in_world = base_link_R @ j_in_base + mano_origin_w
                p_cam = R_c2w_ann.T @ (j_in_world - t_c2w_ann)
                u, v = -1, -1
                if p_cam[2] > 0.01:
                    u = int(fx_ann * p_cam[0] / p_cam[2] + cx_ann)
                    v = int(fy_ann * p_cam[1] / p_cam[2] + cy_ann)
                mano_joints_2d.append((u, v, j_in_base, j_in_world))

            for idx in range(21):
                u, v, j_in_base, j_in_world = mano_joints_2d[idx]
                if not (0 <= u < bgr.shape[1] and 0 <= v < bgr.shape[0]):
                    continue
                color = FINGER_COLORS[idx]
                if idx == 0:
                    cv2.circle(bgr, (u, v), 7, (0,255,255), -1)
                    cv2.putText(bgr, "WRIST", (u+9, v-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                    cv2.putText(bgr, f"({j_in_base[0]:.3f},{j_in_base[1]:.3f},{j_in_base[2]:.3f})",
                                (u+9, v+8), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,255,255), 1)
                elif idx in FINGER_TIP_IDS:
                    fi = FINGER_TIP_IDS.index(idx)
                    cv2.circle(bgr, (u, v), 5, TIP_COLORS[fi], -1)
                    cv2.putText(bgr, JOINT_NAMES[idx], (u+7, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TIP_COLORS[fi], 1)
                    cv2.putText(bgr, f"({j_in_base[0]:.2f},{j_in_base[1]:.2f},{j_in_base[2]:.2f})",
                                (u+7, v+9), cv2.FONT_HERSHEY_SIMPLEX, 0.25, TIP_COLORS[fi], 1)
                elif idx in MCP_IDS:
                    cv2.circle(bgr, (u, v), 4, color, -1)
                    cv2.putText(bgr, JOINT_NAMES[idx], (u+6, v-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                else:
                    cv2.circle(bgr, (u, v), 3, color, -1)
                    cv2.putText(bgr, JOINT_NAMES[idx], (u+5, v-3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

            if ee_pos is not None:
                ee_cam = R_c2w_ann.T @ (ee_pos - t_c2w_ann)
                if ee_cam[2] > 0.01:
                    eu = int(fx_ann * ee_cam[0] / ee_cam[2] + cx_ann)
                    ev = int(fy_ann * ee_cam[1] / ee_cam[2] + cy_ann)
                    if 0 <= eu < bgr.shape[1] and 0 <= ev < bgr.shape[0]:
                        cv2.circle(bgr, (eu, ev), 7, (255, 200, 0), -1)
                        cv2.putText(bgr, "ROBOT_EE", (eu+9, ev-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

            panel_x, panel_y = 10, bgr.shape[0] - 210
            cv2.rectangle(bgr, (panel_x - 5, panel_y - 5), (panel_x + 380, panel_y + 205), (0, 0, 0), -1)
            cv2.rectangle(bgr, (panel_x - 5, panel_y - 5), (panel_x + 380, panel_y + 205), (100, 100, 100), 1)
            cv2.putText(bgr, "MANO 21 Keypoints vs Robot EE", (panel_x, panel_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            _, _, wrist_base, wrist_world = mano_joints_2d[0]
            wrist_ee_dist = np.linalg.norm(wrist_world - ee_pos) if ee_pos is not None else 0
            cv2.putText(bgr, f"Wrist-EE dist: {wrist_ee_dist:.3f}m", (panel_x, panel_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            for fi, (tip_id, name) in enumerate(zip(FINGER_TIP_IDS, TIP_NAMES)):
                _, _, tip_base, tip_world = mano_joints_2d[tip_id]
                tip_ee_dist = np.linalg.norm(tip_world - ee_pos) if ee_pos is not None else 0
                cv2.putText(bgr, f"{name}: ({tip_base[0]:+.3f},{tip_base[1]:+.3f},{tip_base[2]:+.3f}) d={tip_ee_dist:.3f}m",
                            (panel_x, panel_y + 48 + fi * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.28, TIP_COLORS[fi], 1)

            wrist_rot_world_ann = base_link_R @ R_cam2base_mat @ wr_arr @ operator2mano.T
            wrist_rot_euler = np.degrees(np.array([
                np.arctan2(wrist_rot_world_ann[2,1], wrist_rot_world_ann[2,2]),
                np.arctan2(-wrist_rot_world_ann[2,0], np.sqrt(wrist_rot_world_ann[2,1]**2 + wrist_rot_world_ann[2,2]**2)),
                np.arctan2(wrist_rot_world_ann[1,0], wrist_rot_world_ann[0,0])
            ]))
            cv2.putText(bgr, f"MANO Wrist rot (world): ({wrist_rot_euler[0]:+.1f},{wrist_rot_euler[1]:+.1f},{wrist_rot_euler[2]:+.1f}) deg",
                        (panel_x, panel_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 255), 1)

            if ee_R is not None:
                ee_rot_euler = np.degrees(np.array([
                    np.arctan2(ee_R[2,1], ee_R[2,2]),
                    np.arctan2(-ee_R[2,0], np.sqrt(ee_R[2,1]**2 + ee_R[2,2]**2)),
                    np.arctan2(ee_R[1,0], ee_R[0,0])
                ]))
                cv2.putText(bgr, f"Robot EE rot (RPD): ({ee_rot_euler[0]:+.1f},{ee_rot_euler[1]:+.1f},{ee_rot_euler[2]:+.1f}) deg",
                            (panel_x, panel_y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
                rot_diff = np.linalg.norm(wrist_rot_euler - ee_rot_euler)
                cv2.putText(bgr, f"Rot diff: {rot_diff:.1f} deg",
                            (panel_x, panel_y + 172), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

            wrist_cam_wt = wt.get("wrist_pos_camera")
            if wrist_cam_wt is not None:
                wc = np.array(wrist_cam_wt)
                cv2.putText(bgr, f"Depth: {wc[2]:.2f}m  CamXYZ: ({wc[0]:.2f},{wc[1]:.2f},{wc[2]:.2f})  Scale: {mano_scale_ann:.1f}x",
                            (panel_x, panel_y + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        except Exception:
            pass

    # ========== 5. 渲染视频（多视角） ==========
    total_render_frames = len(warmup_qpos_list) + len(qpos_sequence)
    fk_errors = []
    comfort_scores = []
    fk_details = []

    for view_idx, current_view in enumerate(views_to_render):
        current_output_path = output_paths[view_idx]
        print(f"\\n  === Rendering view [{view_idx+1}/{len(views_to_render)}]: {current_view} -> {current_output_path} ===")

        camera_pos_v, camera_quat_v = get_camera_pose_for_view(current_view)
        camera.set_local_pose(sapien.Pose(camera_pos_v.tolist(), camera_quat_v))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(current_output_path, fourcc, fps, (img_width, img_height))

        robot_ee_trail = []
        mano_trail_global = []
        mano_trail_rot_global = []
        frame_times = []
        ego_smooth_pos = None

        for frame_i in tqdm.tqdm(range(total_render_frames), desc=f"SAPIEN [{current_view}]"):
            t_start = time.time()

            is_warmup = frame_i < len(warmup_qpos_list)
            data_frame_idx = frame_i - len(warmup_qpos_list)

            if is_warmup:
                qpos = warmup_qpos_list[frame_i]
                wt = None
                qpos_data = None
            else:
                qpos_data = qpos_sequence[data_frame_idx] if data_frame_idx < len(qpos_sequence) else None
                wt = wrist_traj[data_frame_idx] if data_frame_idx < len(wrist_traj) else None

                if qpos_data is not None:
                    qpos = initial_qpos.copy()
                    arm_key = f"{hand_type}_arm"
                    if arm_key in qpos_data and len(qpos_data[arm_key]) == len(arm_indices):
                        arm_vals = np.array(qpos_data[arm_key])
                        arm_vals = clamp_arm_joints(arm_vals, R1_JOINT_LIMITS)
                        for j, idx in enumerate(arm_indices):
                            qpos[idx] = arm_vals[j]
                    qpos[gripper_idx1] = clamp_gripper_value(qpos_data["gripper"][0], gripper_j1_name)
                    qpos[gripper_idx2] = clamp_gripper_value(qpos_data["gripper"][1], gripper_j2_name)
                else:
                    qpos = None

            if qpos is not None:
                r1_robot.set_qpos(qpos)

            scene.step()
            scene.update_render()

            fk_error = None
            comfort = None
            ee_actual_pos = None
            ee_actual_quat = None
            ee_actual_R = None
            ee_target_pos = None
            ee_target_base = None
            wrist_world = None
            ik_target_world_pos = None
            reach_dist = None

            if not is_warmup and qpos_data is not None and ee_link is not None:
                ee_pose = ee_link.get_entity_pose()
                ee_actual_pos = np.array(ee_pose.p)
                ee_actual_quat = np.array(ee_pose.q)
                ee_actual_R = pr_rot_render.matrix_from_quaternion(ee_actual_quat)

                if "ee_target_base" in qpos_data:
                    ee_target_base = np.array(qpos_data["ee_target_base"])
                    ee_target_pos = base_link_R @ ee_target_base + base_link_p
                    fk_error = np.linalg.norm(ee_actual_pos - ee_target_pos)
                    if view_idx == 0:
                        fk_errors.append(fk_error)

                if "ee_target_world" in qpos_data:
                    ik_target_world_pos = np.array(qpos_data["ee_target_world"])

                if wt is not None and wt.get("wrist_pos_raw") is not None:
                    wrist_world = np.array(wt["wrist_pos_raw"])

                reach_dist = np.linalg.norm(ee_target_base) if ee_target_pos is not None else None

                current_qpos = r1_robot.get_qpos()
                arm_joints = np.array([current_qpos[i] for i in arm_indices])
                comfort = joint_comfort_score(arm_joints, R1_JOINT_LIMITS)
                if view_idx == 0:
                    comfort_scores.append(comfort)

                if view_idx == 0 and data_frame_idx < 5:
                    fk_details.append({
                        "frame": data_frame_idx,
                        "ee_actual": ee_actual_pos.copy().tolist(),
                        "ee_target": ee_target_pos.tolist() if ee_target_pos is not None else None,
                        "fk_error_cm": fk_error * 100 if fk_error is not None else None,
                        "arm_joints_deg": np.degrees(arm_joints).tolist(),
                        "comfort": comfort,
                    })
                    if data_frame_idx == 0:
                        print(f"  [FK DEBUG] Frame 0:")
                        print(f"    ee_actual_pos: [{ee_actual_pos[0]:.4f}, {ee_actual_pos[1]:.4f}, {ee_actual_pos[2]:.4f}]")
                        print(f"    ee_target_pos: [{ee_target_pos[0]:.4f}, {ee_target_pos[1]:.4f}, {ee_target_pos[2]:.4f}]")
                        print(f"    ee_target_base: [{ee_target_base[0]:.4f}, {ee_target_base[1]:.4f}, {ee_target_base[2]:.4f}]")
                        print(f"    base_link_p: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")
                        print(f"    base_link_R diag: [{base_link_R[0,0]:.4f}, {base_link_R[1,1]:.4f}, {base_link_R[2,2]:.4f}]")
                        print(f"    arm_joints_rad: [{arm_joints[0]:.4f}, {arm_joints[1]:.4f}, {arm_joints[2]:.4f}, {arm_joints[3]:.4f}, {arm_joints[4]:.4f}, {arm_joints[5]:.4f}]")
                        print(f"    arm_joints_deg: [{np.degrees(arm_joints[0]):.1f}, {np.degrees(arm_joints[1]):.1f}, {np.degrees(arm_joints[2]):.1f}, {np.degrees(arm_joints[3]):.1f}, {np.degrees(arm_joints[4]):.1f}, {np.degrees(arm_joints[5]):.1f}]")
                        print(f"    FK error: {fk_error*100:.2f} cm")

            if current_view == "egocentric" and not is_warmup and ee_actual_pos is not None:
                ego_smooth_pos = update_egocentric_camera(ee_actual_pos, ego_smooth_pos)

            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            bgr = rgb[..., ::-1].copy()

            frame_time = time.time() - t_start
            frame_times.append(frame_time)
            fps_actual = 1.0 / max(np.mean(frame_times[-30:]), 1e-6) if len(frame_times) > 0 else 0

            elapsed_sec = frame_i / max(fps, 1)
            bgr = draw_hud(bgr, frame_i, total_render_frames, qpos_data if not is_warmup else None,
                            wt, fk_error, comfort, ee_actual_pos, ee_target_pos,
                            arm_joint_names, hand_type, fps_actual,
                            wrist_world=wrist_world, ik_target_world=ik_target_world_pos,
                            arm_base_pos=base_link_p, mapping_offset=None,
                            reach_dist=reach_dist, arm_max_reach=ARM_MAX_REACH,
                            elapsed_sec=elapsed_sec, is_warmup=is_warmup)

            clear_3d_nodes(mano_3d_nodes)
            clear_3d_nodes(gripper_3d_nodes)
            clear_3d_nodes(axis_3d_nodes)
            clear_3d_nodes(trail_3d_nodes)

            if not is_warmup and wt is not None:
                joint_pos_wt = wt.get("joint_pos")
                wrist_rot_mano_wt = wt.get("wrist_rot_mano")
                is_grasping_wt = wt.get("is_grasping", False)
                wrist_pos_camera_wt = wt.get("wrist_pos_camera")

                if joint_pos_wt is not None and ee_actual_pos is not None:
                    joint_pos_arr = np.array(joint_pos_wt)
                    wrist_rot_arr = np.array(wrist_rot_mano_wt) if wrist_rot_mano_wt is not None else np.eye(3)

                    mano_origin = ee_actual_pos + get_mano_offset(current_view)
                    mano_scale = np.clip(compute_mano_scale(current_view, wrist_pos_camera_wt, ee_actual_pos, ego_smooth_pos), 1.2, 4.0)

                    render_mano_3d(joint_pos_arr, wrist_rot_arr, mano_origin, mano_scale, is_grasping_wt)

            if not is_warmup and ee_actual_pos is not None:
                render_robot_axes_and_trails(ee_actual_pos, ee_actual_R)

            cv2.putText(bgr, f"{'WARMUP' if is_warmup else 'DATA'} {frame_i+1}/{total_render_frames}",
                        (bgr.shape[1] - 250, bgr.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            if not is_warmup and wt is not None and ee_actual_pos is not None:
                draw_mano_2d_annotations(bgr, wt, ee_actual_pos, ee_actual_R, mano_origin if 'mano_origin' in dir() else ee_actual_pos, mano_scale if 'mano_scale' in dir() else 2.0, ego_smooth_pos)

            writer.write(bgr)

        writer.release()
        print(f"  View [{current_view}] saved: {current_output_path}")

    # ========== 6. 输出评估报告 ==========
    print(f"\\nSAPIEN rendering complete: {len(views_to_render)} views")
    print(f"  Total frames per view: {total_render_frames} (warmup={len(warmup_qpos_list)} + data={len(qpos_sequence)})")
    for v, op in zip(views_to_render, output_paths):
        print(f"  [{v:>11s}]: {op}")

    if fk_errors:
        print(f"\\n  FK Verification:")
        print(f"    Mean error:  {np.mean(fk_errors)*100:.2f} cm")
        print(f"    Max error:   {np.max(fk_errors)*100:.2f} cm")
        print(f"    Median error: {np.median(fk_errors)*100:.2f} cm")
        under_2cm = sum(1 for e in fk_errors if e < 0.02) / len(fk_errors) * 100
        print(f"    Error<2cm:   {under_2cm:.1f}%")

    if comfort_scores:
        print(f"\\n  Joint Comfort:")
        print(f"    Mean:  {np.mean(comfort_scores):.4f} (1.0=best, 0.0=worst)")
        print(f"    Min:   {np.min(comfort_scores):.4f}")

    if fk_details:
        print(f"\\n  Frame-by-frame (first 5):")
        for d in fk_details:
            print(f"    Frame {d['frame']}: FK error={d['fk_error_cm']:.2f}cm  "
                  f"comfort={d['comfort']:.4f}  "
                  f"joints=[{', '.join(f'{j:.1f}' for j in d['arm_joints_deg'])}]")

    eval_path = Path(output_paths[0]).parent / "render_evaluation.pkl"
    with open(eval_path, "wb") as f:
        pickle.dump({
            "fk_errors": fk_errors,
            "comfort_scores": comfort_scores,
            "fk_details": fk_details,
            "warmup_frames": len(warmup_qpos_list),
            "views_rendered": views_to_render,
        }, f)
    print(f"  Evaluation saved: {eval_path}")

if __name__ == "__main__":
    main()
'''

# ============================================================================
# 预渲染设置子进程脚本
# 功能：加载R1机器人，获取实际臂基座位置，计算工作空间映射
# 目的：确保IK求解器使用与渲染器一致的臂基座坐标
# ============================================================================
_PRE_RENDER_SETUP_SCRIPT = '''#!/usr/bin/env python3
"""预渲染设置 - 获取R1机器人实际臂基座位置和工作空间映射"""
import os
os.environ["MUJOCO_GL"] = "disable"
import sys
import pickle
import tempfile
import numpy as np

DEX_RETARGETING_ROOT = "/home/an/robot_world_ws/src/dex-retargeting"
GALAXEA_SIM_PATH = "/home/an/robot_world_ws/src/GalaxeaManipSim"
sys.path.insert(0, os.path.join(DEX_RETARGETING_ROOT, "src"))
sys.path.insert(0, os.path.join(DEX_RETARGETING_ROOT, "example", "position_retargeting"))
sys.path.insert(0, GALAXEA_SIM_PATH)

import sapien
from dex_retargeting import yourdfpy as urdf
from pytransform3d import rotations as pr

R1_FULL_URDF = os.path.join(GALAXEA_SIM_PATH, "galaxea_sim", "assets", "r1", "robot_glb.urdf")
Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])
ARM_BASE_OFFSET_RIGHT = np.array([-0.03207, -0.09724, 1.40347])
ARM_BASE_OFFSET_LEFT = np.array([-0.03207, 0.09724, 1.40347])
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])
SAFETY_DISTANCE = 0.075
ARM_MAX_REACH = 0.713
OTHER_ARM_DEFAULT = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]


def main():
    setup_pkl_path = sys.argv[1]
    with open(setup_pkl_path, "rb") as f:
        data = pickle.load(f)

    avg_wrist = np.array(data["avg_wrist"])
    hand_type = data.get("hand_type", "right")

    try:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        sapien.render.set_log_level("error")
        scene = sapien.Scene()
        scene.set_timestep(1 / 240)
    except Exception as e:
        R_180Z_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        arm_base_offset_local = ARM_BASE_OFFSET_LEFT if hand_type == "left" else ARM_BASE_OFFSET_RIGHT
        arm_base_offset_world = R_180Z_mat @ arm_base_offset_local
        robot_root_pos = np.array([0.0, 0.0, 0.0])
        base_link_p = robot_root_pos + arm_base_offset_world
        base_link_R = R_180Z_mat
        base_link_R_inv = base_link_R.T

        result = {
            "robot_root_pos": robot_root_pos.tolist(),
            "base_link_p": base_link_p.tolist(),
            "base_link_R": base_link_R.tolist(),
            "base_link_R_inv": base_link_R_inv.tolist(),
            "used_actual_fk": False,
        }
        with open(setup_pkl_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Fallback: using estimated arm base (SAPIEN unavailable)")
        print(f"  base_link_p: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")
        return

    # 加载R1机器人
    urdf_path = R1_FULL_URDF
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True

    robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
    temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
    urdf_stem = os.path.splitext(os.path.basename(urdf_path))[0]
    temp_path = os.path.join(temp_dir, f"{urdf_stem}.urdf")
    robot_urdf.write_xml_file(temp_path)

    r1_robot = loader.load(temp_path)
    active_joints = r1_robot.get_active_joints()
    joint_names = [j.get_name() for j in active_joints]

    # 设置另一臂默认姿态
    other_arm_prefix = "left_arm" if hand_type == "right" else "right_arm"
    other_arm_indices = [i for i, name in enumerate(joint_names) if other_arm_prefix in name]
    initial_qpos = r1_robot.get_qpos().copy()
    for j, idx in enumerate(other_arm_indices):
        if j < len(OTHER_ARM_DEFAULT):
            initial_qpos[idx] = OTHER_ARM_DEFAULT[j]
    r1_robot.set_qpos(initial_qpos)

    # 放置机器人
    arm_prefix = f"{hand_type}_arm"
    R_180Z_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
    arm_base_offset_local = ARM_BASE_OFFSET_LEFT if hand_type == "left" else ARM_BASE_OFFSET_RIGHT
    arm_base_offset_world = R_180Z_mat @ arm_base_offset_local
    robot_root_pos = np.array([0.0, 0.0, 0.0])
    r1_robot.set_root_pose(sapien.Pose(robot_root_pos.tolist(), Q_180Z.tolist()))

    scene.step()
    scene.update_render()

    # 获取实际臂基座位姿
    arm_base_link = None
    for link in r1_robot.get_links():
        if f"{arm_prefix}_base_link" in link.get_name():
            arm_base_link = link
            break

    if arm_base_link is not None:
        base_link_pose = arm_base_link.get_entity_pose()
        base_link_p = np.array(base_link_pose.p)
        base_link_q = np.array(base_link_pose.q)
        base_link_R = pr.matrix_from_quaternion(base_link_q)
    else:
        base_link_p = robot_root_pos + arm_base_offset_world
        base_link_R = R_180Z_mat

    base_link_R_inv = base_link_R.T

    print(f"  Robot root: [{robot_root_pos[0]:.4f}, {robot_root_pos[1]:.4f}, {robot_root_pos[2]:.4f}]")
    print(f"  Actual arm base: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")

    result = {
        "robot_root_pos": robot_root_pos.tolist(),
        "base_link_p": base_link_p.tolist(),
        "base_link_R": base_link_R.tolist(),
        "base_link_R_inv": base_link_R_inv.tolist(),
        "used_actual_fk": arm_base_link is not None,
    }

    with open(setup_pkl_path, "wb") as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    main()
'''
# 改进点：
#   1. 工作空间映射（参照r1_hand_tracking_video.py）
#   2. 关节角度低通滤波（alpha=0.5）
#   3. Warmup过渡支持
#   4. 超出臂展检测
# ============================================================================
_IK_SOLVER_SCRIPT = '''#!/usr/bin/env python3
"""RelaxedIK 逆运动学求解子进程

改进点：
  - 工作空间映射：将手腕轨迹映射到R1臂舒适工作空间
  - 关节角度低通滤波（alpha=0.5）
  - 超出臂展检测
  - 保存详细的IK调试信息
"""
import os
os.environ["MUJOCO_GL"] = "disable"
import sys
import pickle
import numpy as np
import tqdm

GALAXEA_SIM_PATH = "/home/an/robot_world_ws/src/GalaxeaManipSim"
sys.path.insert(0, GALAXEA_SIM_PATH)

from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver
from dex_retargeting.optimizer_utils import LPFilter

GRIPPER_LIMITS = {
    "right": {"joint1": (0.0, 0.05), "joint2": (-0.05, 0.0)},
    "left": {"joint1": (0.0, 0.05), "joint2": (-0.05, 0.0)},
}

GRIPPER_OPEN_VALUE = 0.04
GRIPPER_CLOSED_VALUE_J1 = 0.005
GRIPPER_CLOSED_VALUE_J2 = -0.005

R1_RIGHT_JOINT_LIMITS = np.array([
    [-2.0944, 2.0944],
    [0.0, 3.1416],
    [-2.8798, 0.0],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-2.8798, 2.8798],
])

R1_LEFT_JOINT_LIMITS = np.array([
    [-2.0944, 2.0944],
    [0.0, 3.1416],
    [-2.8798, 0.0],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-2.8798, 2.8798],
])

R1_STARTING_CONFIG_RIGHT = np.array([-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985])
R1_STARTING_CONFIG_LEFT = np.array([1.5, 1.9508, -1.0809, 0.4438, -0.1709, -0.1985])

MAX_JOINT_VEL_DEG_PER_FRAME = 8.0


def clamp_arm_joints(joints, joint_limits):
    return np.clip(joints, joint_limits[:, 0], joint_limits[:, 1])


def limit_joint_velocity(joints, prev_joints, max_vel_deg=MAX_JOINT_VEL_DEG_PER_FRAME):
    max_vel_rad = np.radians(max_vel_deg)
    delta = joints - prev_joints
    clamped_delta = np.clip(delta, -max_vel_rad, max_vel_rad)
    return prev_joints + clamped_delta


def nonlinear_gripper_scale(raw_val, lo, hi, power=0.7):
    if hi - lo < 1e-8:
        return lo
    norm = (raw_val - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    scaled = norm ** power
    return lo + scaled * (hi - lo)

ARM_BASE_OFFSET_RIGHT = np.array([-0.03207, -0.09724, 1.40347])
ARM_BASE_OFFSET_LEFT = np.array([-0.03207, 0.09724, 1.40347])
ARM_MAX_REACH = 0.713
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])
SAFETY_DISTANCE = 0.075
LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5


def main():
    ik_pkl_path = sys.argv[1]
    ik_result_path = sys.argv[2]
    setup_pkl_path = sys.argv[3] if len(sys.argv) > 3 else None

    with open(ik_pkl_path, "rb") as f:
        ik_input = pickle.load(f)

    wrist_traj = ik_input["wrist_traj"]
    left_settings = ik_input["left_settings"]
    right_settings = ik_input["right_settings"]
    hand_type = ik_input.get("hand_type", "right")
    avg_wrist = np.array(ik_input.get("avg_wrist", [0.3, 0.0, 0.5]))

    ik_solver = RelaxedIKSolver(
        left_setting_file_path=left_settings,
        right_setting_file_path=right_settings,
        tolerances=[0.1] * 6,
    )

    starting_config = R1_STARTING_CONFIG_RIGHT if hand_type == "right" else R1_STARTING_CONFIG_LEFT
    if hand_type == "right":
        ik_solver.relaxed_ik_right.reset(starting_config.tolist())
    else:
        ik_solver.relaxed_ik_left.reset(starting_config.tolist())
    print(f"  RelaxedIK reset to starting_config: [{', '.join(f'{np.degrees(j):.1f} deg' for j in starting_config)}]")

    valid_wrists = [np.array(w["wrist_pos"]) for w in wrist_traj if w["wrist_pos"] is not None]
    if len(valid_wrists) == 0:
        with open(ik_result_path, "wb") as f:
            pickle.dump([None] * len(wrist_traj), f)
        return

    avg_wrist = np.mean(valid_wrists, axis=0)
    wrist_range = np.ptp(valid_wrists, axis=0)
    motion_scale = 1.0
    min_range = 0.05
    for axis in range(3):
        if wrist_range[axis] > 0.001 and wrist_range[axis] < min_range:
            motion_scale = max(motion_scale, min_range / wrist_range[axis])
    motion_scale = min(motion_scale, 3.0)
    print(f"  Wrist range: [{wrist_range[0]:.4f}, {wrist_range[1]:.4f}, {wrist_range[2]:.4f}]")
    print(f"  Motion scale: {motion_scale:.2f}x (amplify small motions)")

    # 全部在臂基座坐标系中计算，无需世界坐标系
    # wrist_pos 已经是臂基座坐标（通过 R_cam2base 变换）
    # COMFORT_TARGET_IN_BASE 也是臂基座坐标
    mapping_offset_base = COMFORT_TARGET_IN_BASE - avg_wrist
    approach_dir_base = -COMFORT_TARGET_IN_BASE / max(np.linalg.norm(COMFORT_TARGET_IN_BASE), 1e-6)
    safety_offset_base = approach_dir_base * SAFETY_DISTANCE

    print(f"  IK workspace mapping (arm base coords):")
    print(f"    Avg wrist (base): [{avg_wrist[0]:.4f}, {avg_wrist[1]:.4f}, {avg_wrist[2]:.4f}]")
    print(f"    Mapping offset (base): [{mapping_offset_base[0]:.4f}, {mapping_offset_base[1]:.4f}, {mapping_offset_base[2]:.4f}]")
    print(f"    Safety offset (base): [{safety_offset_base[0]:.4f}, {safety_offset_base[1]:.4f}, {safety_offset_base[2]:.4f}]")

    if setup_pkl_path is not None:
        with open(setup_pkl_path, "rb") as f:
            setup_data = pickle.load(f)
        base_link_p = np.array(setup_data["base_link_p"])
        base_link_R = np.array(setup_data["base_link_R"])
        base_link_R_inv = np.array(setup_data["base_link_R_inv"])
        used_actual = setup_data.get("used_actual_fk", False)
        print(f"  Using pre-render setup (actual_fk={used_actual})")
    else:
        R_180Z_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        arm_base_offset_local = ARM_BASE_OFFSET_LEFT if hand_type == "left" else ARM_BASE_OFFSET_RIGHT
        arm_base_offset_world = R_180Z_mat @ arm_base_offset_local
        robot_root_pos = np.array([0.0, 0.0, 0.0])
        base_link_p = robot_root_pos + arm_base_offset_world
        base_link_R = R_180Z_mat
        base_link_R_inv = base_link_R.T
        print(f"  Using estimated arm base (no pre-render setup)")

    ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
    joint_filter = None
    first_ik_joints = None
    prev_arm_joints = None

    joint_limits = R1_RIGHT_JOINT_LIMITS if hand_type == "right" else R1_LEFT_JOINT_LIMITS
    starting_config = R1_STARTING_CONFIG_RIGHT if hand_type == "right" else R1_STARTING_CONFIG_LEFT

    qpos_sequence = []
    out_of_reach = 0
    ik_errors = 0

    for local_idx, wt in enumerate(tqdm.tqdm(wrist_traj, desc="IK Solving")):
        if wt["wrist_pos"] is None:
            qpos_sequence.append(None)
            continue

        wrist_pos_base = np.array(wt["wrist_pos"])

        wrist_delta = wrist_pos_base - avg_wrist
        wrist_delta_scaled = wrist_delta * motion_scale
        wrist_pos_scaled = avg_wrist + wrist_delta_scaled

        ik_target_base_raw = wrist_pos_scaled + mapping_offset_base + safety_offset_base
        ik_target_base = ee_pos_filter.next(ik_target_base_raw)

        dist_to_base = np.linalg.norm(ik_target_base)

        if dist_to_base > ARM_MAX_REACH:
            out_of_reach += 1

        ee_ik_quat = [1.0, 0.0, 0.0, 0.0]

        try:
            if hand_type == "right":
                arm_joints = np.array(
                    ik_solver.solve_position_right(ik_target_base.tolist(), ee_ik_quat)
                )
            else:
                arm_joints = np.array(
                    ik_solver.solve_position_left(ik_target_base.tolist(), ee_ik_quat)
                )
        except Exception:
            ik_errors += 1
            qpos_sequence.append(None)
            continue

        # 关节限位钳位（防止超出物理限制导致碰撞）
        arm_joints = clamp_arm_joints(arm_joints, joint_limits)

        if joint_filter is None:
            joint_filter = LPFilter(alpha=0.8)
            joint_filter.next(arm_joints.copy())
            prev_arm_joints = arm_joints.copy()
            first_ik_joints = arm_joints.copy()
        arm_joints = joint_filter.next(arm_joints)

        arm_joints = clamp_arm_joints(arm_joints, joint_limits)
        if prev_arm_joints is not None:
            arm_joints = limit_joint_velocity(arm_joints, prev_arm_joints, max_vel_deg=30.0)
        prev_arm_joints = arm_joints.copy()

        gripper_qpos = wt.get("gripper_qpos")
        if gripper_qpos is not None:
            gripper_qpos = np.array(gripper_qpos)
            g_limits = GRIPPER_LIMITS.get(hand_type, GRIPPER_LIMITS["right"])
            raw_g1 = float(gripper_qpos[0]) if len(gripper_qpos) >= 1 else 0.0
            raw_g2 = float(gripper_qpos[1]) if len(gripper_qpos) >= 2 else 0.0
            gripper1 = np.clip(nonlinear_gripper_scale(raw_g1, g_limits["joint1"][0], g_limits["joint1"][1]),
                               g_limits["joint1"][0], g_limits["joint1"][1])
            gripper2 = np.clip(nonlinear_gripper_scale(raw_g2, g_limits["joint2"][0], g_limits["joint2"][1]),
                               g_limits["joint2"][0], g_limits["joint2"][1])
        else:
            gripper1 = 0.0
            gripper2 = 0.0

        arm_key = f"{hand_type}_arm"
        ik_target_world = base_link_R @ ik_target_base + base_link_p
        qpos_sequence.append({
            arm_key: arm_joints.tolist(),
            "gripper": [gripper1, gripper2],
            "ee_target_base": ik_target_base.tolist(),
            "ee_target_world": ik_target_world.tolist(),
        })

        if local_idx == 0:
            print(f"  IK debug (frame 0):")
            print(f"    Wrist (base): [{wrist_pos_base[0]:.4f}, {wrist_pos_base[1]:.4f}, {wrist_pos_base[2]:.4f}]")
            print(f"    IK target (base): [{ik_target_base[0]:.4f}, {ik_target_base[1]:.4f}, {ik_target_base[2]:.4f}]")
            print(f"    IK target (world): [{ik_target_world[0]:.4f}, {ik_target_world[1]:.4f}, {ik_target_world[2]:.4f}]")
            print(f"    Dist to base: {dist_to_base:.4f}m {'OK' if dist_to_base < ARM_MAX_REACH else 'OUT OF REACH'}")

    print(f"  IK stats: {len(qpos_sequence)} total, {ik_errors} errors, {out_of_reach} out of reach")
    print(f"  First IK joints: [{', '.join(f'{np.degrees(j):.1f} deg' for j in first_ik_joints)}]" if first_ik_joints is not None else "")

    with open(ik_result_path, "wb") as f:
        pickle.dump({
            "qpos_sequence": qpos_sequence,
            "mapping_offset_base": mapping_offset_base.tolist(),
            "safety_offset_base": safety_offset_base.tolist(),
            "out_of_reach": out_of_reach,
            "ik_errors": ik_errors,
            "first_ik_joints": first_ik_joints.tolist() if first_ik_joints is not None else None,
            "base_link_p": base_link_p.tolist(),
        }, f)

if __name__ == "__main__":
    main()
'''


class R1SAPIENPipeline:
    """
    R1 机器人 SAPIEN 仿真管道主类

    改进点：
      - 工作空间映射（参照 r1_hand_tracking_video.py）
      - Warmup过渡（从初始姿态平滑插值到第一帧IK结果）
      - FK验证和关节舒适度评估
      - 监测变量HUD叠加在视频上
      - 子进程脚本按需生成（添加 --keep-scripts 控制）
      - 输出文件统一管理
    """

    def __init__(
        self,
        video_path: str,
        output_dir: str,
        fps: int = 30,
        start_frame: int = 0,
        num_frames: int = -1,
        selfie: bool = True,
        hand_type: str = "auto",
        grasp_threshold: float = None,
        release_threshold: float = None,
        view: str = "front",
        img_width: int = 1920,
        img_height: int = 1080,
        keep_scripts: bool = False,
        steps: str = "detect,ik,render",
    ):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.start_frame = start_frame
        self.num_frames = num_frames
        self.selfie = selfie
        self.hand_type_str = hand_type
        self.grasp_threshold = grasp_threshold
        self.release_threshold = release_threshold
        self.view = view
        self.img_width = img_width
        self.img_height = img_height
        self.keep_scripts = keep_scripts
        self.steps = [s.strip() for s in steps.split(",")]
        self.logger = None
        self.pipeline_info = {}
        self._temp_files = []
        self._prev_wrist_raw = None
        self._prev_wrist_cam = None
        self._wrist_raw_history = []

    def _draw_coord_debug(self, bgr, hand_data, arm_info, wrist_pos_raw):
        H, W = bgr.shape[:2]
        R_cam2base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

        wrist_pos_camera = hand_data.get("wrist_pos_camera", wrist_pos_raw)
        wrist_pos_base = R_cam2base @ wrist_pos_camera
        wrist_rot_mano = hand_data["wrist_rot_mano"]
        joint_pos = hand_data["joint_pos"]
        keypoint_2d = hand_data["keypoint_2d"]

        if not hasattr(self, '_avg_wrist_est'):
            self._avg_wrist_est = wrist_pos_base.copy()
        alpha_avg = 0.05
        self._avg_wrist_est = self._avg_wrist_est * (1 - alpha_avg) + wrist_pos_base * alpha_avg

        mapping_offset_base = COMFORT_TARGET_IN_BASE - self._avg_wrist_est
        approach_dir_base = -COMFORT_TARGET_IN_BASE / max(np.linalg.norm(COMFORT_TARGET_IN_BASE), 1e-6)
        safety_offset_base = approach_dir_base * SAFETY_DISTANCE

        ik_target_base = wrist_pos_base + mapping_offset_base + safety_offset_base

        wx = int(keypoint_2d.landmark[0].x * W)
        wy = int(keypoint_2d.landmark[0].y * H)

        if not hasattr(self, '_wrist_trail_2d'):
            self._wrist_trail_2d = []
        self._wrist_trail_2d.append((wx, wy))
        if len(self._wrist_trail_2d) > 60:
            self._wrist_trail_2d = self._wrist_trail_2d[-60:]

        AXIS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        AXIS_LABELS_CAM = ['X(right)', 'Y(down)', 'Z(fwd)']
        AXIS_LABELS_BASE = ['+X(fwd)', '+Y(left)', '+Z(up)']

        scale_2d = 60

        cam_ox, cam_oy = wx, wy
        for i in range(3):
            dx = int([1, 0, 0.4][i] * scale_2d)
            dy = int([0, 1, 0.4][i] * scale_2d)
            ex, ey = cam_ox + dx, cam_oy + dy
            cv2.arrowedLine(bgr, (cam_ox, cam_oy), (ex, ey), AXIS_COLORS[i], 3, tipLength=0.12)
            cv2.putText(bgr, AXIS_LABELS_CAM[i], (ex + 4, ey - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, AXIS_COLORS[i], 2)
        cv2.putText(bgr, "CAM", (cam_ox - 15, cam_oy - scale_2d - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        R_cam2base_inv = R_cam2base.T
        base_ox = wx + 200
        base_oy = wy
        for i in range(3):
            axis_cam = R_cam2base_inv[:, i]
            dx = int(axis_cam[0] * scale_2d)
            dy = int(axis_cam[1] * scale_2d)
            length = np.sqrt(dx * dx + dy * dy)
            if length < 5:
                cv2.circle(bgr, (base_ox, base_oy), 5, AXIS_COLORS[i], -1)
            else:
                ex, ey = base_ox + dx, base_oy + dy
                cv2.arrowedLine(bgr, (base_ox, base_oy), (ex, ey), AXIS_COLORS[i], 2, tipLength=0.15)
            cv2.putText(bgr, AXIS_LABELS_BASE[i], (base_ox + int(axis_cam[0] * scale_2d) + 4,
                        base_oy + int(axis_cam[1] * scale_2d) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, AXIS_COLORS[i], 1)
        cv2.putText(bgr, "ARM_BASE(R_cam2base)", (base_ox - 50, base_oy - scale_2d - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 255), 2)

        base2_ox = wx - 200
        base2_oy = wy
        for i in range(3):
            axis_cam = R_cam2base[:, i]
            dx = int(axis_cam[0] * scale_2d)
            dy = int(axis_cam[1] * scale_2d)
            length = np.sqrt(dx * dx + dy * dy)
            if length < 5:
                cv2.circle(bgr, (base2_ox, base2_oy), 5, AXIS_COLORS[i], -1)
            else:
                ex, ey = base2_ox + dx, base2_oy + dy
                cv2.arrowedLine(bgr, (base2_ox, base2_oy), (ex, ey), AXIS_COLORS[i], 2, tipLength=0.15)
            cv2.putText(bgr, AXIS_LABELS_BASE[i], (base2_ox + int(axis_cam[0] * scale_2d) + 4,
                        base2_oy + int(axis_cam[1] * scale_2d) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, AXIS_COLORS[i], 1)
        cv2.putText(bgr, "ARM_BASE(proj)", (base2_ox - 30, base2_oy - scale_2d - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 255), 2)

        fwd_cam_dir = np.array([0, 0, 1])
        fwd_in_base = R_cam2base @ fwd_cam_dir
        fwd_base_dx = int(fwd_in_base[0] * scale_2d * 1.5)
        fwd_base_dy = int(fwd_in_base[1] * scale_2d * 1.5)
        fwd_base_ex = base2_ox + fwd_base_dx
        fwd_base_ey = base2_oy + fwd_base_dy
        cv2.arrowedLine(bgr, (base2_ox, base2_oy), (fwd_base_ex, fwd_base_ey),
                        (0, 255, 255), 3, tipLength=0.15)
        is_fwd_correct = fwd_in_base[0] > 0 and abs(fwd_in_base[0]) > abs(fwd_in_base[1])
        fwd_check_color = (0, 255, 0) if is_fwd_correct else (0, 0, 255)
        fwd_check_text = "FWD->+X OK" if is_fwd_correct else "FWD->NOT+X!"
        cv2.putText(bgr, fwd_check_text, (fwd_base_ex + 5, fwd_base_ey - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fwd_check_color, 2)

        if len(self._wrist_trail_2d) > 2:
            for i in range(1, len(self._wrist_trail_2d)):
                alpha = i / len(self._wrist_trail_2d)
                color = (int(255 * alpha), int(100 * alpha), int(255 * (1 - alpha)))
                pt1 = self._wrist_trail_2d[i - 1]
                pt2 = self._wrist_trail_2d[i]
                cv2.line(bgr, pt1, pt2, color, 2)

        MANO_FINGERS = [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [0, 13, 14, 15, 16],
            [0, 17, 18, 19, 20],
        ]
        MANO_PALM = [(5, 9), (9, 13), (13, 17)]
        FINGER_COLORS = [
            (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)
        ]

        inset_size = 180
        inset_x = W - inset_size - 10
        inset_y = H - inset_size - 10
        overlay_inset = bgr.copy()
        cv2.rectangle(overlay_inset, (inset_x, inset_y),
                      (inset_x + inset_size, inset_y + inset_size), (0, 0, 0), -1)
        cv2.addWeighted(overlay_inset, 0.75, bgr, 0.25, 0, bgr)

        pts_2d = []
        for i in range(21):
            px = int(keypoint_2d.landmark[i].x * W)
            py = int(keypoint_2d.landmark[i].y * H)
            pts_2d.append((px, py))

        wrist_px = pts_2d[0]
        scale_factor = 0.0
        ref_dists = []
        for fi, finger in enumerate(MANO_FINGERS):
            for j in range(len(finger) - 1):
                p1 = pts_2d[finger[j]]
                p2 = pts_2d[finger[j + 1]]
                d = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                ref_dists.append(d)
        if ref_dists:
            scale_factor = np.median(ref_dists)

        mano_cx = inset_x + inset_size // 2
        mano_cy = inset_y + inset_size // 2 + 20
        mano_scale = (inset_size * 0.35) / max(scale_factor, 1)

        mano_pts = []
        for i in range(21):
            dx = (pts_2d[i][0] - wrist_px[0]) * mano_scale
            dy = (pts_2d[i][1] - wrist_px[1]) * mano_scale
            mano_pts.append((int(mano_cx + dx), int(mano_cy + dy)))

        for fi, finger in enumerate(MANO_FINGERS):
            for j in range(len(finger) - 1):
                p1 = mano_pts[finger[j]]
                p2 = mano_pts[finger[j + 1]]
                cv2.line(bgr, p1, p2, FINGER_COLORS[fi], 2)
        for p1_idx, p2_idx in MANO_PALM:
            cv2.line(bgr, mano_pts[p1_idx], mano_pts[p2_idx], (200, 200, 200), 1)

        for i in range(21):
            cv2.circle(bgr, mano_pts[i], 3, (255, 255, 255), -1)
            cv2.circle(bgr, mano_pts[i], 2, (0, 0, 0), -1)

        mano_axis_len = 40
        for i in range(3):
            dx = int(wrist_rot_mano[0, i] * mano_axis_len)
            dy = int(wrist_rot_mano[1, i] * mano_axis_len)
            ex, ey = mano_cx + dx, mano_cy + dy
            cv2.arrowedLine(bgr, (mano_cx, mano_cy), (ex, ey), AXIS_COLORS[i], 2, tipLength=0.15)
            axis_name = ['Xm', 'Ym', 'Zm'][i]
            cv2.putText(bgr, axis_name, (ex + 3, ey - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, AXIS_COLORS[i], 1)
        cv2.putText(bgr, "MANO", (inset_x + 5, inset_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        panel_w = 620
        panel_h = 440
        panel_x = W - panel_w - 5
        panel_y = 5
        overlay_img = bgr.copy()
        cv2.rectangle(overlay_img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img, 0.8, bgr, 0.2, 0, bgr)

        y0 = panel_y + 18
        dy = 15
        fs = 0.32
        th = 1
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(bgr, "=== COORD TRANSFORM (SIMPLIFIED) ===", (panel_x + 5, y0), font, 0.42, (0, 255, 255), th)
        y0 += dy + 4

        cv2.putText(bgr, "[0] Hand-local (world_landmarks):", (panel_x + 5, y0), font, fs, (120, 120, 180), th)
        y0 += dy
        cv2.putText(bgr, f"  X={wrist_pos_raw[0]:+.4f} Y={wrist_pos_raw[1]:+.4f} Z={wrist_pos_raw[2]:+.4f}",
                    (panel_x + 5, y0), font, fs, (120, 120, 180), th)
        y0 += dy
        cv2.putText(bgr, "  (hand-center local, NOT camera!)", (panel_x + 5, y0), font, fs * 0.85, (100, 100, 150), th)
        y0 += dy + 3

        cv2.putText(bgr, "[1] Camera (2D+depth estimate):", (panel_x + 5, y0), font, fs, (180, 180, 255), th)
        y0 += dy
        cv2.putText(bgr, f"  X={wrist_pos_camera[0]:+.4f} Y={wrist_pos_camera[1]:+.4f} Z={wrist_pos_camera[2]:+.4f}",
                    (panel_x + 5, y0), font, fs, (180, 180, 255), th)
        y0 += dy
        cv2.putText(bgr, "  X=right Y=down Z=forward(depth)", (panel_x + 5, y0), font, fs * 0.85, (140, 140, 200), th)
        y0 += dy + 3

        cv2.putText(bgr, "[2] R_cam2base -> ArmBase:", (panel_x + 5, y0), font, fs, (255, 150, 255), th)
        y0 += dy
        cv2.putText(bgr, f"  X={wrist_pos_base[0]:+.4f} Y={wrist_pos_base[1]:+.4f} Z={wrist_pos_base[2]:+.4f}",
                    (panel_x + 5, y0), font, fs, (255, 150, 255), th)
        y0 += dy
        cv2.putText(bgr, "  +X=fwd +Y=left +Z=up (arm base)", (panel_x + 5, y0), font, fs * 0.85, (200, 120, 200), th)
        y0 += dy
        cv2.putText(bgr, "  Cam+Z(fwd)->Base+X(fwd) DIRECT!", (panel_x + 5, y0), font, fs * 0.85, (0, 255, 255), th)
        y0 += dy + 3

        cv2.putText(bgr, "[3] IK Target (arm base):", (panel_x + 5, y0), font, fs, (255, 200, 150), th)
        y0 += dy
        cv2.putText(bgr, f"  X={ik_target_base[0]:+.4f} Y={ik_target_base[1]:+.4f} Z={ik_target_base[2]:+.4f}",
                    (panel_x + 5, y0), font, fs, (255, 200, 150), th)
        y0 += dy + 5

        cv2.putText(bgr, "=== FORWARD MAPPING TEST ===", (panel_x + 5, y0), font, 0.40, (0, 255, 255), th)
        y0 += dy + 2

        fwd_cam = np.array([0.0, 0.0, 0.1])
        fwd_base = R_cam2base @ fwd_cam
        cv2.putText(bgr, f"  Cam+Z(0.1m fwd)->Base[{fwd_base[0]:+.3f},{fwd_base[1]:+.3f},{fwd_base[2]:+.3f}]",
                    (panel_x + 5, y0), font, fs * 0.9, (255, 150, 255), th)
        y0 += dy

        is_fwd_correct = fwd_base[0] > 0 and abs(fwd_base[0]) > abs(fwd_base[1]) and abs(fwd_base[0]) > abs(fwd_base[2])
        check_color = (0, 255, 0) if is_fwd_correct else (0, 0, 255)
        check_text = "OK: cam fwd -> base +X(robot fwd)" if is_fwd_correct else "WRONG: cam fwd -> NOT base +X!"
        cv2.putText(bgr, f"  >> {check_text}", (panel_x + 5, y0), font, fs * 0.9, check_color, 2)
        y0 += dy + 5

        right_cam = np.array([0.1, 0.0, 0.0])
        right_base = R_cam2base @ right_cam
        cv2.putText(bgr, f"  Cam+X(0.1m right)->Base[{right_base[0]:+.3f},{right_base[1]:+.3f},{right_base[2]:+.3f}]",
                    (panel_x + 5, y0), font, fs * 0.85, (180, 180, 255), th)
        y0 += dy + 5

        cv2.putText(bgr, "=== ACTUAL MOTION ===", (panel_x + 5, y0), font, 0.40, (0, 255, 255), th)
        y0 += dy + 2

        if self._prev_wrist_raw is not None:
            delta_cam = wrist_pos_camera - self._prev_wrist_cam
            delta_base = R_cam2base @ delta_cam

            cv2.putText(bgr, f"  dCam: X={delta_cam[0]:+.5f} Y={delta_cam[1]:+.5f} Z={delta_cam[2]:+.5f}",
                        (panel_x + 5, y0), font, fs, (180, 180, 255), th)
            y0 += dy
            cv2.putText(bgr, f"  dBas: X={delta_base[0]:+.5f} Y={delta_base[1]:+.5f} Z={delta_base[2]:+.5f}",
                        (panel_x + 5, y0), font, fs, (255, 150, 255), th)
            y0 += dy

            dom_cam = np.argmax(np.abs(delta_cam))
            dom_base = np.argmax(np.abs(delta_base))
            cam_dir = ["RIGHT", "DOWN", "FWD"][dom_cam]
            base_dir = ["FWD(+X)", "LEFT(+Y)", "UP(+Z)"][dom_base]
            cam_sign = "+" if delta_cam[dom_cam] > 0 else "-"
            base_sign = "+" if delta_base[dom_base] > 0 else "-"

            cv2.putText(bgr, f"  Cam: {cam_sign}{cam_dir}  Base: {base_sign}{base_dir}",
                        (panel_x + 5, y0), font, fs * 0.9, (0, 255, 255), th)
            y0 += dy

            if dom_cam == 2 and abs(delta_cam[2]) > 0.001:
                if dom_base == 0 and delta_base[0] > 0:
                    cv2.putText(bgr, "  >> FWD->+FWD in base: CORRECT!", (panel_x + 5, y0),
                                font, fs, (0, 255, 0), 2)
                else:
                    cv2.putText(bgr, "  >> FWD->NOT+FWD in base: WRONG!", (panel_x + 5, y0),
                                font, fs, (0, 0, 255), 2)
                y0 += dy

        self._prev_wrist_raw = wrist_pos_raw.copy()
        self._prev_wrist_cam = wrist_pos_camera.copy()

        legend_y = panel_y + panel_h + 8
        legend_h = 60
        overlay_img2 = bgr.copy()
        cv2.rectangle(overlay_img2, (panel_x, legend_y), (panel_x + panel_w, legend_y + legend_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay_img2, 0.8, bgr, 0.2, 0, bgr)

        ly = legend_y + 16
        cv2.putText(bgr, "R=Red(X) G=Green(Y) B=Blue(Z)  Cyan=FWD test", (panel_x + 5, ly),
                    font, 0.28, (180, 180, 180), 1)
        ly += 16
        cv2.putText(bgr, "FIX: R_cam2base direct (no World intermediate)",
                    (panel_x + 5, ly), font, 0.28, (0, 255, 255), 1)
        ly += 16
        cv2.putText(bgr, "Cam+Z(fwd)->Base+X(fwd) egocentric=armbase",
                    (panel_x + 5, ly), font, 0.28, (0, 255, 0), 1)

    def run(self):
        """执行完整的仿真管道（步骤0-4）"""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.output_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
        except (OSError, PermissionError) as e:
            fallback = Path("/tmp/r1_sapien_output")
            print(f"  WARNING: Output dir {self.output_dir} is not writable ({e})")
            print(f"  Falling back to {fallback}")
            self.output_dir = fallback
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logger(self.output_dir)

        if self.hand_type_str == "auto":
            self.logger.info("[0/4] Auto-detecting hand type...")
            self.hand_type_str = auto_detect_hand_type(self.video_path, selfie=self.selfie)

        self.hand_type_enum = HandType.right if self.hand_type_str == "Right" else HandType.left
        self.arm_side = "right" if self.hand_type_str == "Right" else "left"

        self.logger.info("=" * 80)
        self.logger.info("  Egocentric Video -> R1 Robot (SAPIEN) Simulation Pipeline")
        self.logger.info("=" * 80)
        self.logger.info(f"  Video:          {self.video_path}")
        self.logger.info(f"  Output:         {self.output_dir}")
        self.logger.info(f"  View:           {self.view}")
        self.logger.info(f"  FPS:            {self.fps}")
        self.logger.info(f"  Hand:           {self.hand_type_str} (selfie={self.selfie})")
        self.logger.info(f"  Arm side:       {self.arm_side}")
        self.logger.info(f"  R1 URDF:        {R1_FULL_URDF_PATH}")
        self.logger.info(f"  Grasp thresh:   {self.grasp_threshold or 0.04}")
        self.logger.info(f"  Release thresh: {self.release_threshold or 0.06}")
        self.logger.info(f"  Keep scripts:   {self.keep_scripts}")
        self.logger.info(f"  Steps:          {self.steps}")
        self.logger.info("")

        self.pipeline_info["pipeline_config"] = {
            "video_path": str(self.video_path),
            "output_dir": str(self.output_dir),
            "fps": self.fps,
            "start_frame": self.start_frame,
            "num_frames": self.num_frames,
            "selfie": self.selfie,
            "hand_type": self.hand_type_str,
            "arm_side": self.arm_side,
            "view": self.view,
            "grasp_threshold": self.grasp_threshold or 0.04,
            "release_threshold": self.release_threshold or 0.06,
            "r1_urdf_path": str(R1_FULL_URDF_PATH),
            "r1_right_settings": str(R1_RIGHT_SETTINGS),
            "r1_left_settings": str(R1_LEFT_SETTINGS),
            "lp_alpha_ee": LP_ALPHA_EE,
            "lp_alpha_joint": LP_ALPHA_JOINT,
            "arm_base_offset_local": (ARM_BASE_OFFSET_LEFT if self.arm_side == "left" else ARM_BASE_OFFSET_RIGHT).tolist(),
            "arm_max_reach": ARM_MAX_REACH,
            "comfort_target_in_base": COMFORT_TARGET_IN_BASE.tolist(),
            "safety_distance": SAFETY_DISTANCE,
            "warmup_frames": WARMUP_FRAMES,
            "gripper_limits": {k: list(v) for k, v in GRIPPER_LIMITS.items()},
            "timestamp": datetime.now().isoformat(),
            "steps": self.steps,
        }

        try:
            hand_data_list, wrist_traj = self._detect_and_retarget()

            setup_data = None
            if "ik" in self.steps or "render" in self.steps:
                setup_data = self._pre_render_setup(wrist_traj)

            qpos_sequence = None
            if "ik" in self.steps:
                qpos_sequence = self._solve_arm_ik(hand_data_list, wrist_traj, setup_data=setup_data)
            else:
                self.logger.info("[2/4] SKIPPED (not in --steps)")
                qpos_sequence = [None] * len(wrist_traj)

            if "render" in self.steps:
                self._render_sapien_video(qpos_sequence, wrist_traj)
            else:
                self.logger.info("[3/4] SKIPPED (not in --steps)")

            self._save_info_file()
        finally:
            self._cleanup_temp_files()

    def _detect_and_retarget(self):
        """步骤1：手部检测 + R1 夹爪重定向"""
        self.logger.info("[1/4] Detecting hands and retargeting to R1 gripper (SeqRetargeting)...")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            import os
            path_info = []
            path_info.append(f"  Path: {self.video_path}")
            path_info.append(f"  Exists: {os.path.exists(self.video_path)}")
            if os.path.exists(self.video_path):
                path_info.append(f"  Size: {os.path.getsize(self.video_path)} bytes")
            else:
                parent = os.path.dirname(self.video_path)
                path_info.append(f"  Parent exists: {os.path.exists(parent)}")
                if os.path.exists(parent):
                    path_info.append(f"  Parent contents: {os.listdir(parent)[:5]}")
            raise ValueError(f"Cannot open video:\n" + "\n".join(path_info))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = max(1, int(video_fps / self.fps))
        if self.num_frames < 0:
            self.num_frames = total_frames

        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.pipeline_info["video_info"] = {
            "total_frames": total_frames,
            "video_fps": video_fps,
            "resolution": f"{vid_w}x{vid_h}",
            "frame_skip": frame_skip,
        }
        self.logger.info(f"  Video: {total_frames} frames, {video_fps:.1f} fps, {vid_w}x{vid_h}, skip={frame_skip}")

        detector = EgocentricHandDetector(hand_type=self.hand_type_str, selfie=self.selfie)
        arm_analyzer = ArmMotionAnalyzer(hand_type=self.hand_type_enum, arm_offset=np.zeros(3))

        # MediaPipe Pose检测器（用于肩-肘-腕链路估计）
        # 优先使用model_complexity=0（lite版本，通常已缓存）
        # 如果模型不可用则优雅降级
        import mediapipe as mp_pose_module
        pose_detector = None
        try:
            pose_detector = mp_pose_module.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        except Exception as e:
            self.logger.warning(f"  MediaPipe Pose unavailable ({e}), using hand-only tracking")
        is_left_label = (self.hand_type_str == "Right") if self.selfie else (self.hand_type_str == "Left")
        pose_shoulder_idx = 11 if is_left_label else 12
        pose_elbow_idx = 13 if is_left_label else 14
        pose_wrist_idx = 15 if is_left_label else 16

        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        retargeter = RobotHandRetargeter(
            robot_name=RobotName.r1,
            hand_type=self.hand_type_enum,
            retargeting_type=RetargetingType.position,
        )

        optimizer = retargeter.retargeting.optimizer
        fixed_retarget_indices = optimizer.idx_pin2fixed
        robot_joint_names = optimizer.robot.dof_joint_names
        n_dof = len(robot_joint_names)
        ref_indices = optimizer.target_link_human_indices

        init_qpos = np.zeros(n_dof, dtype=np.float32)
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx < n_dof:
                fixed_qpos[i] = init_qpos[retarget_idx]

        self.logger.info(f"  R1 gripper DOF: {n_dof}, Fixed joints: {len(fixed_retarget_indices)}, "
                         f"Target joints: {n_dof - len(fixed_retarget_indices)}")
        self.logger.info(f"  Ref indices (MANO->R1): {ref_indices.tolist()}")

        self.pipeline_info["retargeting_config"] = {
            "robot_name": "r1",
            "retargeting_type": "position",
            "n_dof": n_dof,
            "fixed_joints": len(fixed_retarget_indices),
            "target_joints": n_dof - len(fixed_retarget_indices),
            "ref_indices": ref_indices.tolist(),
            "robot_joint_names": robot_joint_names,
        }

        contact_filter = ContactPreservationFilter(
            grasp_threshold=self.grasp_threshold if self.grasp_threshold else 0.04,
            release_threshold=self.release_threshold if self.release_threshold else 0.06,
            smoothing_alpha=0.25,
            position_alpha=0.4,
        )
        trajectory_smoother = TrajectorySmoother(window_size=5, alpha=0.25)

        overlay_path = self.output_dir / "overlay.mp4"
        overlay_writer = cv2.VideoWriter(
            str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (vid_w, vid_h),
        )

        hand_data_list = []
        wrist_traj = []
        meta_data = retargeter.get_meta_data()
        detected = 0
        finger_distances = []
        last_valid_wt = None
        gap_count = 0
        coord_trace = []

        R_cam2base_trace = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        COMFORT_TARGET_IN_BASE_TRACE = np.array([0.30, 0.0, -0.25])
        SAFETY_DISTANCE_TRACE = 0.075
        avg_wrist_trace = COMFORT_TARGET_IN_BASE_TRACE.copy()
        R_180Z_trace = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        arm_base_off_local = ARM_BASE_OFFSET_LEFT if self.arm_side == "left" else ARM_BASE_OFFSET_RIGHT
        arm_base_off_world = R_180Z_trace @ arm_base_off_local
        base_p_trace = np.array([0.0, 0.0, 0.0]) + arm_base_off_world
        base_R_trace = R_180Z_trace
        base_R_inv_trace = base_R_trace.T

        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        processed = 0

        with tqdm.tqdm(total=self.num_frames, desc="Detecting") as pbar:
            while cap.isOpened() and processed < self.num_frames:
                frame_idx = self.start_frame + processed * frame_skip
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = frame[..., ::-1]
                hand_data = detector.detect_single(rgb)

                # MediaPipe Pose检测（肩-肘-腕链路）
                shoulder_pos = None
                elbow_pos = None
                elbow_angle = None
                arm_plane_normal = None
                if pose_detector is not None:
                    try:
                        pose_result = pose_detector.process(rgb)
                        if pose_result.pose_world_landmarks:
                            lm = pose_result.pose_world_landmarks.landmark
                            s_vis = lm[pose_shoulder_idx].visibility
                            e_vis = lm[pose_elbow_idx].visibility
                            if s_vis > 0.3 and e_vis > 0.3:
                                shoulder_pos = np.array([lm[pose_shoulder_idx].x, lm[pose_shoulder_idx].y, lm[pose_shoulder_idx].z])
                                elbow_pos = np.array([lm[pose_elbow_idx].x, lm[pose_elbow_idx].y, lm[pose_elbow_idx].z])
                                upper_arm = elbow_pos - shoulder_pos
                                ua_len = np.linalg.norm(upper_arm)
                                if ua_len > 0.01:
                                    upper_arm_norm = upper_arm / ua_len
                                    wrist_from_pose = np.array([lm[pose_wrist_idx].x, lm[pose_wrist_idx].y, lm[pose_wrist_idx].z])
                                    forearm = wrist_from_pose - elbow_pos
                                    fa_len = np.linalg.norm(forearm)
                                    if fa_len > 0.01:
                                        forearm_norm = forearm / fa_len
                                        cos_angle = np.clip(np.dot(upper_arm_norm, forearm_norm), -1.0, 1.0)
                                        elbow_angle = np.arccos(cos_angle)
                                        arm_plane_normal = np.cross(upper_arm_norm, forearm_norm)
                                        apn_len = np.linalg.norm(arm_plane_normal)
                                        if apn_len > 0.01:
                                            arm_plane_normal = arm_plane_normal / apn_len
                                        else:
                                            arm_plane_normal = None
                    except Exception:
                        pass

                bgr = frame.copy()

                if hand_data is None:
                    gap_count += 1
                    if gap_count <= MAX_GAP_FILL_FRAMES and last_valid_wt is not None:
                        hand_data_list.append(None)
                        wrist_traj.append(last_valid_wt)
                        cv2.putText(bgr, f"INTERP ({gap_count})", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
                    else:
                        hand_data_list.append(None)
                        wrist_traj.append(None)
                        cv2.putText(bgr, "NO HAND", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                    overlay_writer.write(bgr)
                    processed += 1
                    pbar.update(1)
                    continue

                gap_count = 0
                detected += 1
                joint_pos = hand_data["joint_pos"]
                wrist_pos = hand_data["wrist_pos_raw"]
                wrist_pos_cam = hand_data.get("wrist_pos_camera", wrist_pos)
                wrist_rot = hand_data["wrist_rot_mano"]

                wrist_pos_base_for_avg = R_cam2base_trace @ wrist_pos_cam
                alpha_avg_trace = 0.05
                avg_wrist_trace = avg_wrist_trace * (1 - alpha_avg_trace) + wrist_pos_base_for_avg * alpha_avg_trace

                from scipy.spatial.transform import Rotation as R_scipy
                wrist_quat = R_scipy.from_matrix(wrist_rot).as_quat()

                ref_value = joint_pos[ref_indices, :].astype(np.float32)
                retarget_qpos = retargeter.retargeting.retarget(ref_value, fixed_qpos)
                retarget_qpos = clamp_gripper(retarget_qpos, self.arm_side)

                finger_dist = ArmMotionAnalyzer.compute_finger_aperture(joint_pos)
                finger_distances.append(finger_dist)
                filtered_qpos, is_grasping = contact_filter.update(finger_dist, retarget_qpos)
                smoothed_qpos = trajectory_smoother.smooth(filtered_qpos)
                smoothed_qpos = clamp_gripper(smoothed_qpos, self.arm_side)

                arm_info = arm_analyzer.compute_wrist_6dof(hand_data)

                hand_data_list.append(hand_data)
                wt = {
                    "wrist_pos": arm_info["wrist_pos"],
                    "wrist_quat": arm_info["wrist_quat"],
                    "ee_pos": arm_info["ee_pos"],
                    "ee_quat": arm_info["ee_quat"],
                    "finger_distance": finger_dist,
                    "is_grasping": is_grasping,
                    "gripper_qpos": smoothed_qpos,
                    "frame_idx": frame_idx,
                    "shoulder_pos": shoulder_pos.tolist() if shoulder_pos is not None else None,
                    "elbow_pos": elbow_pos.tolist() if elbow_pos is not None else None,
                    "elbow_angle": float(elbow_angle) if elbow_angle is not None else None,
                    "arm_plane_normal": arm_plane_normal.tolist() if arm_plane_normal is not None else None,
                    "joint_pos": hand_data["joint_pos"],
                    "wrist_rot_mano": hand_data["wrist_rot_mano"],
                    "wrist_pos_camera": hand_data.get("wrist_pos_camera", wrist_pos),
                    "wrist_pos_hand_local": wrist_pos,
                }
                wrist_traj.append(wt)
                last_valid_wt = wt

                detector.draw_skeleton_on_image(bgr, hand_data["keypoint_2d"])
                status = "GRASP" if is_grasping else "OPEN"
                color = (0, 255, 0) if is_grasping else (0, 0, 255)
                cv2.putText(bgr, f"{status} d={finger_dist:.3f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                self._draw_coord_debug(bgr, hand_data, arm_info, wrist_pos)
                overlay_writer.write(bgr)

                wrist_pos_base_trace = R_cam2base_trace @ wrist_pos_cam
                mapping_off_base = COMFORT_TARGET_IN_BASE_TRACE - avg_wrist_trace
                approach_dir_base_trace = -COMFORT_TARGET_IN_BASE_TRACE / max(np.linalg.norm(COMFORT_TARGET_IN_BASE_TRACE), 1e-6)
                safety_off_base = approach_dir_base_trace * SAFETY_DISTANCE_TRACE
                ik_tgt_base_trace = wrist_pos_base_trace + mapping_off_base + safety_off_base
                trace_entry = {
                    "frame": frame_idx,
                    "wrist_hand_local": wrist_pos.tolist(),
                    "wrist_camera": wrist_pos_cam.tolist(),
                    "wrist_base": wrist_pos_base_trace.tolist(),
                    "ik_target_base": ik_tgt_base_trace.tolist(),
                    "arm_base_p": base_p_trace.tolist(),
                }
                coord_trace.append(trace_entry)
                if detected <= 5:
                    self.logger.info(f"  [CoordTrace] Frame {frame_idx}:")
                    self.logger.info(f"    HandLocal: [{wrist_pos[0]:+.4f}, {wrist_pos[1]:+.4f}, {wrist_pos[2]:+.4f}]")
                    self.logger.info(f"    Camera:    [{wrist_pos_cam[0]:+.4f}, {wrist_pos_cam[1]:+.4f}, {wrist_pos_cam[2]:+.4f}]")
                    self.logger.info(f"    Base:      [{wrist_pos_base_trace[0]:+.4f}, {wrist_pos_base_trace[1]:+.4f}, {wrist_pos_base_trace[2]:+.4f}]")
                    self.logger.info(f"    IKbase:    [{ik_tgt_base_trace[0]:+.4f}, {ik_tgt_base_trace[1]:+.4f}, {ik_tgt_base_trace[2]:+.4f}]")

                processed += 1
                pbar.update(1)

        cap.release()
        overlay_writer.release()

        if coord_trace:
            trace_path = self.output_dir / "coord_trace.json"
            with trace_path.open("w", encoding="utf-8") as f:
                json.dump(coord_trace, f, indent=2)
            self.logger.info(f"  Coord trace saved: {trace_path} ({len(coord_trace)} entries)")

            if len(coord_trace) >= 2:
                first = coord_trace[0]
                last = coord_trace[-1]
                cam_delta = np.array(last["wrist_camera"]) - np.array(first["wrist_camera"])
                base_delta = np.array(last["ik_target_base"]) - np.array(first["ik_target_base"])
                self.logger.info(f"  [CoordTrace] Cumulative motion (first->last):")
                self.logger.info(f"    Cam delta:   [{cam_delta[0]:+.4f}, {cam_delta[1]:+.4f}, {cam_delta[2]:+.4f}]")
                self.logger.info(f"    Base delta:  [{base_delta[0]:+.4f}, {base_delta[1]:+.4f}, {base_delta[2]:+.4f}]")
                cam_dom = np.argmax(np.abs(cam_delta))
                base_dom = np.argmax(np.abs(base_delta))
                cam_dir = ["X(right)", "Y(down)", "Z(fwd)"][cam_dom]
                base_dir = ["X(fwd)", "Y(left)", "Z(up)"][base_dom]
                self.logger.info(f"    Cam dominant: {cam_dir}  Base dominant: {base_dir}")
                if cam_dom == 2 and base_dom == 0:
                    self.logger.info(f"    *** CORRECT: Forward in cam -> Forward in arm base! ***")
                elif cam_dom == 2:
                    self.logger.info(f"    *** MISMATCH: Forward in cam but NOT forward in arm base! ***")

        valid = sum(1 for d in hand_data_list if d is not None)
        interp = sum(1 for i, d in enumerate(hand_data_list)
                     if d is None and i < len(wrist_traj) and wrist_traj[i] is not None)
        grasp_count = sum(1 for w in wrist_traj if w is not None and w.get("is_grasping"))
        open_count = sum(1 for w in wrist_traj if w is not None and not w.get("is_grasping"))

        self.logger.info(f"  Detection: {valid}/{processed} ({valid / max(processed, 1) * 100:.1f}%)")
        if interp > 0:
            self.logger.info(f"  Gap-filled: {interp} frames (max {MAX_GAP_FILL_FRAMES} frame gap)")
        self.logger.info(f"  Grasp frames: {grasp_count}, Open frames: {open_count}")

        # 计算手腕轨迹统计（用于工作空间映射）
        valid_wrist_pos = [w["wrist_pos"] for w in wrist_traj if w is not None]
        wrist_stats = {}
        if valid_wrist_pos:
            wp = np.array(valid_wrist_pos)
            wrist_stats = {
                "centroid": np.mean(wp, axis=0).tolist(),
                "range": np.ptp(wp, axis=0).tolist(),
                "std": np.std(wp, axis=0).tolist(),
                "min": np.min(wp, axis=0).tolist(),
                "max": np.max(wp, axis=0).tolist(),
            }
            self.logger.info(f"  Wrist centroid: [{wrist_stats['centroid'][0]:.4f}, "
                             f"{wrist_stats['centroid'][1]:.4f}, {wrist_stats['centroid'][2]:.4f}]")
            self.logger.info(f"  Wrist range: X={wrist_stats['range'][0]:.4f}, "
                             f"Y={wrist_stats['range'][1]:.4f}, Z={wrist_stats['range'][2]:.4f}")

        if finger_distances:
            fd = np.array(finger_distances)
            self.logger.info(f"  Finger distance: mean={fd.mean():.4f}, std={fd.std():.4f}, "
                             f"min={fd.min():.4f}, max={fd.max():.4f}")

        self.pipeline_info["detection"] = {
            "total_frames": processed,
            "detected_frames": valid,
            "detection_rate": f"{valid / max(processed, 1) * 100:.1f}%",
            "gap_filled_frames": interp,
            "grasp_frames": grasp_count,
            "open_frames": open_count,
            "finger_distance_mean": float(np.mean(finger_distances)) if finger_distances else 0,
            "finger_distance_std": float(np.std(finger_distances)) if finger_distances else 0,
            "finger_distance_min": float(np.min(finger_distances)) if finger_distances else 0,
            "finger_distance_max": float(np.max(finger_distances)) if finger_distances else 0,
            "grasp_threshold_used": self.grasp_threshold or 0.04,
            "release_threshold_used": self.release_threshold or 0.06,
            "wrist_stats": wrist_stats,
        }

        pkl_path = self.output_dir / "trajectory.pkl"
        with pkl_path.open("wb") as f:
            pickle.dump({
                "hand_data_list": hand_data_list,
                "wrist_trajectory": wrist_traj,
                "meta_data": meta_data,
                "video_info": {
                    "video_path": str(self.video_path),
                    "start_frame": self.start_frame,
                    "num_frames": processed,
                    "fps": self.fps,
                    "frame_skip": frame_skip,
                    "hand_type": self.hand_type_str,
                    "selfie": self.selfie,
                },
            }, f)
        self.logger.info(f"  Saved: {pkl_path}")
        self.logger.info(f"  Saved: {overlay_path}")

        return hand_data_list, wrist_traj

    def _pre_render_setup(self, wrist_traj):
        """步骤1.5：预渲染设置 - 获取实际臂基座位置和工作空间映射"""
        self.logger.info("\n[1.5/4] Pre-render setup: getting actual arm base position...")

        valid_wrists = [w["wrist_pos"] for w in wrist_traj if w is not None]
        avg_wrist = np.mean(valid_wrists, axis=0) if valid_wrists else np.array([0.3, 0.0, 0.5])

        setup_pkl_path = self.output_dir / "pre_render_setup.pkl"
        setup_input = {
            "avg_wrist": avg_wrist.tolist(),
            "hand_type": self.arm_side,
        }
        with setup_pkl_path.open("wb") as f:
            pickle.dump(setup_input, f)

        setup_script = self.output_dir / "_pre_render_setup.py"
        setup_script.write_text(_PRE_RENDER_SETUP_SCRIPT)
        self._temp_files.append(setup_script)

        import subprocess
        env = os.environ.copy()
        env["MEDIAPIPE_DISABLE_GPU"] = "1"

        result = subprocess.run(
            [sys.executable, str(setup_script), str(setup_pkl_path)],
            capture_output=True, text=True, timeout=120,
            env=env,
        )

        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                self.logger.info(f"  [Setup] {line}")

        if result.returncode != 0:
            self.logger.warning(f"  Pre-render setup failed, using estimated values:\n{result.stderr[-500:]}")
            return None

        with setup_pkl_path.open("rb") as f:
            setup_data = pickle.load(f)

        used_actual = setup_data.get("used_actual_fk", False)
        self.logger.info(f"  Arm base position: actual_fk={used_actual}")
        self.logger.info(f"  base_link_p: {setup_data.get('base_link_p')}")

        self.pipeline_info["pre_render_setup"] = {
            "used_actual_fk": used_actual,
            "base_link_p": setup_data.get("base_link_p"),
        }

        return setup_data

    def _solve_arm_ik(self, hand_data_list, wrist_traj, setup_data=None):
        """步骤2：通过 RelaxedIK 求解 R1 臂部逆运动学"""
        self.logger.info("\n[2/4] Solving R1 arm IK via RelaxedIK (subprocess)...")

        ik_pkl_path = self.output_dir / "ik_input.pkl"
        ik_result_path = self.output_dir / "ik_result.pkl"

        valid_wrists = [w["wrist_pos"] for w in wrist_traj if w is not None]
        avg_wrist = np.mean(valid_wrists, axis=0) if valid_wrists else np.array([0.3, 0.0, 0.5])

        # 计算手腕轨迹统计
        wrist_stats = self.pipeline_info.get("detection", {}).get("wrist_stats", {})
        wrist_range = wrist_stats.get("range", [0.1, 0.1, 0.1])
        wrist_std = wrist_stats.get("std", [0.02, 0.02, 0.02])

        self.pipeline_info["ik_config"] = {
            "avg_wrist_pos": avg_wrist.tolist(),
            "wrist_range": wrist_range,
            "wrist_std": wrist_std,
            "right_settings": str(R1_RIGHT_SETTINGS),
            "left_settings": str(R1_LEFT_SETTINGS),
            "hand_type": self.arm_side,
            "arm_base_offset_local": (ARM_BASE_OFFSET_LEFT if self.arm_side == "left" else ARM_BASE_OFFSET_RIGHT).tolist(),
            "arm_max_reach": ARM_MAX_REACH,
            "comfort_target_in_base": COMFORT_TARGET_IN_BASE.tolist(),
            "safety_distance": SAFETY_DISTANCE,
            "lp_alpha_ee": LP_ALPHA_EE,
            "lp_alpha_joint": LP_ALPHA_JOINT,
        }

        self.logger.info(f"  Avg wrist position: [{avg_wrist[0]:.4f}, {avg_wrist[1]:.4f}, {avg_wrist[2]:.4f}]")
        self.logger.info(f"  Wrist range: X={wrist_range[0]:.4f}, Y={wrist_range[1]:.4f}, Z={wrist_range[2]:.4f}")

        ik_input = {
            "wrist_traj": [
                {
                    "wrist_pos": wt["wrist_pos"].tolist() if wt is not None else None,
                    "gripper_qpos": wt["gripper_qpos"].tolist() if wt is not None and wt.get("gripper_qpos") is not None else None,
                    "elbow_angle": wt.get("elbow_angle"),
                    "arm_plane_normal": wt.get("arm_plane_normal"),
                }
                for wt in wrist_traj
            ],
            "left_settings": str(R1_LEFT_SETTINGS),
            "right_settings": str(R1_RIGHT_SETTINGS),
            "hand_type": self.arm_side,
            "avg_wrist": avg_wrist.tolist(),
            "wrist_range": wrist_range,
            "wrist_std": wrist_std,
        }
        with ik_pkl_path.open("wb") as f:
            pickle.dump(ik_input, f)

        ik_script = self.output_dir / "_ik_solver.py"
        ik_script.write_text(_IK_SOLVER_SCRIPT)
        self._temp_files.append(ik_script)

        import subprocess
        setup_pkl_path = self.output_dir / "pre_render_setup.pkl"
        ik_cmd = [sys.executable, str(ik_script), str(ik_pkl_path), str(ik_result_path)]
        if setup_data is not None and setup_pkl_path.exists():
            ik_cmd.append(str(setup_pkl_path))

        result = subprocess.run(
            ik_cmd,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            self.logger.error(f"  IK subprocess failed:\n{result.stderr[-2000:]}")
            return [None] * len(wrist_traj)

        # 打印IK子进程的输出（包含调试信息）
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                self.logger.info(f"  [IK] {line}")

        with ik_result_path.open("rb") as f:
            ik_result = pickle.load(f)

        # 兼容新旧格式
        if isinstance(ik_result, dict):
            qpos_sequence = ik_result["qpos_sequence"]
            self.pipeline_info["ik_result_extra"] = {
                "mapping_offset_base": ik_result.get("mapping_offset_base"),
                "safety_offset_base": ik_result.get("safety_offset_base"),
                "out_of_reach": ik_result.get("out_of_reach", 0),
                "ik_errors": ik_result.get("ik_errors", 0),
                "first_ik_joints": ik_result.get("first_ik_joints"),
                "base_link_p": ik_result.get("base_link_p"),
            }
        else:
            qpos_sequence = ik_result

        valid = sum(1 for q in qpos_sequence if q is not None)
        self.logger.info(f"  IK solved: {valid}/{len(qpos_sequence)} frames "
                         f"({valid / max(len(qpos_sequence), 1) * 100:.1f}%)")

        self.pipeline_info["ik_result"] = {
            "total_frames": len(qpos_sequence),
            "solved_frames": valid,
            "solve_rate": f"{valid / max(len(qpos_sequence), 1) * 100:.1f}%",
        }

        return qpos_sequence

    def _render_sapien_video(self, qpos_sequence, wrist_traj):
        """步骤3：SAPIEN 渲染仿真视频"""
        self.logger.info("\n[3/4] Rendering SAPIEN simulation video...")

        ALL_VIEWS = ["front", "behind", "side", "topdown", "closeup", "egocentric"]
        if self.view == "all":
            views_to_render = ALL_VIEWS
            self.logger.info(f"  Multi-view mode: rendering {len(views_to_render)} views in single subprocess")
        else:
            views_to_render = [self.view]

        render_result_paths = []
        for v in views_to_render:
            if v == "front":
                video_name = "r1_sapien_sim.mp4"
            else:
                video_name = f"r1_sapien_sim_{v}.mp4"
            render_result_paths.append(self.output_dir / video_name)

        valid_wrists = [w["wrist_pos"] for w in wrist_traj if w is not None]
        avg_wrist = np.mean(valid_wrists, axis=0) if valid_wrists else np.array([0.3, 0.0, 0.5])

        wrist_stats = self.pipeline_info.get("detection", {}).get("wrist_stats", {})
        wrist_range = wrist_stats.get("range", [0.1, 0.1, 0.1])
        wrist_std = wrist_stats.get("std", [0.02, 0.02, 0.02])

        render_input = {
            "qpos_sequence": qpos_sequence,
            "wrist_traj": [
                {
                    "is_grasping": wt["is_grasping"] if wt is not None else False,
                    "finger_distance": wt["finger_distance"] if wt is not None else 0.0,
                    "wrist_pos_raw": wt["wrist_pos"].tolist() if wt is not None and wt.get("wrist_pos") is not None else None,
                    "joint_pos": wt["joint_pos"].tolist() if wt is not None and wt.get("joint_pos") is not None else None,
                    "wrist_rot_mano": wt["wrist_rot_mano"].tolist() if wt is not None and wt.get("wrist_rot_mano") is not None else None,
                    "wrist_pos_camera": wt["wrist_pos_camera"].tolist() if wt is not None and wt.get("wrist_pos_camera") is not None else None,
                    "wrist_pos_hand_local": wt["wrist_pos_hand_local"].tolist() if wt is not None and wt.get("wrist_pos_hand_local") is not None else None,
                }
                for wt in wrist_traj
            ],
            "view": self.view,
            "views": views_to_render,
            "output_paths": [str(p) for p in render_result_paths],
            "fps": self.fps,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "output_path": str(render_result_paths[0]),
            "avg_wrist": avg_wrist.tolist(),
            "wrist_range": wrist_range,
            "wrist_std": wrist_std,
            "hand_type": self.arm_side,
        }
        render_pkl_path = self.output_dir / "render_input.pkl"
        with render_pkl_path.open("wb") as f:
            pickle.dump(render_input, f)

        sapien_ok = False
        try:
            sapien_ok = self._render_sapien_subprocess(render_pkl_path, render_result_paths[0])
        except Exception as e:
            self.logger.error(f"  SAPIEN render failed: {e}")

        if not sapien_ok:
            self.logger.info("  Falling back to software rendering...")
            for v, rp in zip(views_to_render, render_result_paths):
                self.view = v
                self._render_software_fallback(qpos_sequence, wrist_traj, rp)
            self.view = "all" if len(views_to_render) > 1 else views_to_render[0]

        self.pipeline_info["render"] = {
            "view": self.view,
            "views": views_to_render,
            "fps": self.fps,
            "resolution": f"{self.img_width}x{self.img_height}",
            "sapien_ok": sapien_ok,
            "r1_urdf": str(R1_FULL_URDF_PATH),
            "output_videos": [str(p) for p in render_result_paths],
            "output_video": str(render_result_paths[0]),
        }

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("  R1 SAPIEN Simulation Pipeline Complete!")
        self.logger.info(f"  Overlay:    {self.output_dir / 'overlay.mp4'}")
        self.logger.info(f"  Trajectory: {self.output_dir / 'trajectory.pkl'}")
        for v, rp in zip(views_to_render, render_result_paths):
            self.logger.info(f"  SAPIEN [{v:>11s}]: {rp}")
        self.logger.info(f"  Log:        {self.output_dir / 'pipeline_info.log'}")
        self.logger.info(f"  Info JSON:  {self.output_dir / 'pipeline_info.json'}")
        self.logger.info(f"  Desc:       {self.output_dir / 'output_description.txt'}")
        self.logger.info("=" * 80)

    def _render_sapien_subprocess(self, render_pkl_path, render_result_path):
        """在子进程中运行 SAPIEN 渲染"""
        import subprocess

        render_script = self.output_dir / "_sapien_renderer.py"
        render_script.write_text(_SAPIEN_RENDER_SCRIPT)
        self._temp_files.append(render_script)

        env = os.environ.copy()
        env["MEDIAPIPE_DISABLE_GPU"] = "1"

        setup_pkl_path = self.output_dir / "pre_render_setup.pkl"
        render_cmd = [sys.executable, str(render_script), str(render_pkl_path)]
        if setup_pkl_path.exists():
            render_cmd.append(str(setup_pkl_path))

        result = subprocess.run(
            render_cmd,
            capture_output=True, text=True, timeout=600,
            env=env,
        )

        # 打印渲染子进程的输出（包含FK验证和评估报告）
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                self.logger.info(f"  [Render] {line}")

        if result.returncode != 0:
            self.logger.error(f"  SAPIEN subprocess failed:\n{result.stderr[-2000:]}")
            return False

        if render_result_path.exists() and render_result_path.stat().st_size > 0:
            self.logger.info(f"  Video saved: {render_result_path}")
            return True
        else:
            self.logger.info("  SAPIEN produced no output")
            return False

    def _render_software_fallback(self, qpos_sequence, wrist_traj, output_path):
        """软件渲染回退方案"""
        self.logger.info("  Using software rendering fallback (R1-specific)...")

        W, H = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (W, H))

        valid_qpos = [q for q in qpos_sequence if q is not None]
        if len(valid_qpos) == 0:
            writer.release()
            return

        arm_key = f"{self.arm_side}_arm"
        all_arm = np.array([q[arm_key] for q in valid_qpos])
        arm_min = all_arm.min(axis=0)
        arm_max = all_arm.max(axis=0)
        arm_range = arm_max - arm_min
        arm_range[arm_range < 1e-6] = 1.0

        all_gripper = np.array([q["gripper"] for q in valid_qpos])
        gripper_min = all_gripper.min(axis=0)
        gripper_max = all_gripper.max(axis=0)
        gripper_range = gripper_max - gripper_min
        gripper_range[gripper_range < 1e-6] = 1.0

        ARM_JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
        FINGER_COLORS = [(0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0),
                         (255, 0, 255), (0, 255, 255)]

        for frame_i in range(len(qpos_sequence)):
            canvas = np.full((H, W, 3), 25, dtype=np.uint8)
            qpos = qpos_sequence[frame_i]
            wt = wrist_traj[frame_i] if frame_i < len(wrist_traj) else None

            cv2.putText(canvas, f"R1 Robot - {self.arm_side.title()} Arm (Software Render)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if qpos is not None:
                margin_x, margin_y = 40, 60
                plot_w = (W - 2 * margin_x) // 2 - 20
                plot_h = (H - margin_y - 40) // 3 - 10

                arm_vals = qpos[arm_key]
                for ji in range(min(6, len(arm_vals))):
                    row = ji // 2
                    col = ji % 2
                    x_off = margin_x + col * (plot_w + 40)
                    y_off = margin_y + row * (plot_h + 10)

                    val = arm_vals[ji]
                    norm_val = (val - arm_min[ji]) / arm_range[ji]
                    norm_val = np.clip(norm_val, 0, 1)

                    color = FINGER_COLORS[ji]
                    cv2.putText(canvas, f"Arm {ARM_JOINT_NAMES[ji]}: {np.degrees(val):.1f} deg",
                                (x_off, y_off - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    bar_h = int(norm_val * (plot_h - 20))
                    bar_y = y_off + plot_h - bar_h
                    cv2.rectangle(canvas, (x_off, y_off + 15), (x_off + plot_w - 5, y_off + plot_h),
                                  (50, 50, 50), -1)
                    cv2.rectangle(canvas, (x_off, bar_y), (x_off + plot_w - 5, y_off + plot_h),
                                  color, -1)

                gripper_x = margin_x
                gripper_y = margin_y + 3 * (plot_h + 10) + 10
                cv2.putText(canvas, "Gripper", (gripper_x, gripper_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                for gi in range(2):
                    val = qpos["gripper"][gi]
                    norm_val = (val - gripper_min[gi]) / gripper_range[gi]
                    norm_val = np.clip(norm_val, 0, 1)

                    bar_w = (W - 2 * margin_x) // 2 - 20
                    bar_h = int(norm_val * 40)
                    bar_x = gripper_x + gi * (bar_w + 40)
                    bar_y = gripper_y + 50 - bar_h

                    gcolor = (0, 200, 200) if gi == 0 else (200, 200, 0)
                    cv2.rectangle(canvas, (bar_x, gripper_y + 10), (bar_x + bar_w, gripper_y + 50),
                                  (50, 50, 50), -1)
                    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, gripper_y + 50),
                                  gcolor, -1)
                    cv2.putText(canvas, f"Finger{gi+1}: {val:.3f}", (bar_x, gripper_y + 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, gcolor, 1)

            if wt is not None:
                is_grasping = wt.get("is_grasping", False)
                finger_dist = wt.get("finger_distance", 0)
                status = "GRASP" if is_grasping else "OPEN"
                status_color = (0, 255, 0) if is_grasping else (0, 0, 255)
                cv2.putText(canvas, f"Status: {status}  d={finger_dist:.3f}", (10, H - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                joint_pos_wt = wt.get("joint_pos")
                wrist_rot_mano_wt = wt.get("wrist_rot_mano")
                wrist_pos_camera_wt = wt.get("wrist_pos_camera")
                wrist_pos_hand_local_wt = wt.get("wrist_pos_hand_local")

                if joint_pos_wt is not None:
                    joint_pos_arr = np.array(joint_pos_wt)
                    wrist_rot_arr = np.array(wrist_rot_mano_wt) if wrist_rot_mano_wt is not None else np.eye(3)

                    inset_size = 200
                    inset_x = W - inset_size - 10
                    inset_y = H - inset_size - 10

                    overlay_inset = canvas.copy()
                    cv2.rectangle(overlay_inset, (inset_x, inset_y),
                                  (inset_x + inset_size, inset_y + inset_size), (0, 0, 0), -1)
                    cv2.addWeighted(overlay_inset, 0.75, canvas, 0.25, 0, canvas)

                    MANO_FINGERS_SW = [
                        [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
                        [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
                    ]
                    MANO_PALM_SW = [(5, 9), (9, 13), (13, 17)]
                    FINGER_COLORS_SW = [
                        (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)
                    ]
                    FINGER_NAMES_SW = ["Thu", "Ind", "Mid", "Rng", "Pnk"]
                    AXIS_COLORS_SW = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]

                    mano_cx = inset_x + inset_size // 2
                    mano_cy = inset_y + inset_size // 2 + 15

                    ref_dists = []
                    for finger in MANO_FINGERS_SW:
                        for j in range(len(finger) - 1):
                            p1 = joint_pos_arr[finger[j]]
                            p2 = joint_pos_arr[finger[j + 1]]
                            d = np.linalg.norm(p2 - p1)
                            ref_dists.append(d)
                    scale_factor = np.median(ref_dists) if ref_dists else 0.05
                    mano_scale = (inset_size * 0.35) / max(scale_factor, 0.001)

                    mano_pts = []
                    for i in range(21):
                        dx = joint_pos_arr[i][0] * mano_scale
                        dy = joint_pos_arr[i][1] * mano_scale
                        mano_pts.append((int(mano_cx + dx), int(mano_cy + dy)))

                    for fi, finger in enumerate(MANO_FINGERS_SW):
                        for j in range(len(finger) - 1):
                            p1 = mano_pts[finger[j]]
                            p2 = mano_pts[finger[j + 1]]
                            cv2.line(canvas, p1, p2, FINGER_COLORS_SW[fi], 2)
                    for p1_idx, p2_idx in MANO_PALM_SW:
                        cv2.line(canvas, mano_pts[p1_idx], mano_pts[p2_idx], (200, 200, 200), 1)

                    for i in range(21):
                        cv2.circle(canvas, mano_pts[i], 3, (255, 255, 255), -1)
                        cv2.circle(canvas, mano_pts[i], 2, (0, 0, 0), -1)

                    mano_axis_len = 35
                    for i in range(3):
                        dx = int(wrist_rot_arr[0, i] * mano_axis_len)
                        dy = int(wrist_rot_arr[1, i] * mano_axis_len)
                        ex, ey = mano_cx + dx, mano_cy + dy
                        cv2.arrowedLine(canvas, (mano_cx, mano_cy), (ex, ey), AXIS_COLORS_SW[i], 2, tipLength=0.15)
                        axis_name = ['Xm', 'Ym', 'Zm'][i]
                        cv2.putText(canvas, axis_name, (ex + 3, ey - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, AXIS_COLORS_SW[i], 1)

                    cv2.putText(canvas, "MANO", (inset_x + 5, inset_y + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                    legend_y = inset_y + inset_size - 30
                    for fi in range(5):
                        lx = inset_x + 5 + fi * 38
                        cv2.putText(canvas, FINGER_NAMES_SW[fi], (lx, legend_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, FINGER_COLORS_SW[fi], 1)

                if wrist_pos_camera_wt is not None or wrist_pos_hand_local_wt is not None:
                    R_cam2base_sw = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
                    R_cam2base_sw_inv = R_cam2base_sw.T

                    panel_w = 480
                    panel_h = 220
                    panel_x = W - panel_w - 10
                    panel_y = 5
                    overlay_img = canvas.copy()
                    cv2.rectangle(overlay_img, (panel_x, panel_y),
                                  (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay_img, 0.8, canvas, 0.2, 0, canvas)

                    y0 = panel_y + 16
                    dy = 13
                    fs = 0.28
                    th = 1
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    cv2.putText(canvas, "=== COORD TRANSFORM CHAIN ===", (panel_x + 5, y0), font, 0.35, (0, 255, 255), th)
                    y0 += dy + 2

                    if wrist_pos_hand_local_wt is not None:
                        wp = np.array(wrist_pos_hand_local_wt)
                        cv2.putText(canvas, f"[0] Hand-local:", (panel_x + 5, y0), font, fs, (120, 120, 180), th)
                        y0 += dy
                        cv2.putText(canvas, f"  X={wp[0]:+.4f} Y={wp[1]:+.4f} Z={wp[2]:+.4f} (NOT camera!)",
                                    (panel_x + 5, y0), font, fs * 0.9, (120, 120, 180), th)
                        y0 += dy + 1

                    if wrist_pos_camera_wt is not None:
                        wc = np.array(wrist_pos_camera_wt)
                        cv2.putText(canvas, f"[1] Camera(2D+depth):", (panel_x + 5, y0), font, fs, (180, 180, 255), th)
                        y0 += dy
                        cv2.putText(canvas, f"  X={wc[0]:+.4f} Y={wc[1]:+.4f} Z={wc[2]:+.4f}",
                                    (panel_x + 5, y0), font, fs, (180, 180, 255), th)
                        y0 += dy
                        cv2.putText(canvas, "  X(right) Y(down) Z(fwd)", (panel_x + 5, y0), font, fs * 0.85, (140, 140, 200), th)
                        y0 += dy + 1

                        wb = R_cam2base_sw @ wc
                        cv2.putText(canvas, f"[2] ArmBase(R_cam2base):", (panel_x + 5, y0), font, fs, (255, 150, 255), th)
                        y0 += dy
                        cv2.putText(canvas, f"  X={wb[0]:+.4f} Y={wb[1]:+.4f} Z={wb[2]:+.4f}",
                                    (panel_x + 5, y0), font, fs, (255, 150, 255), th)
                        y0 += dy
                        cv2.putText(canvas, "  +X(fwd) +Y(left) +Z(up)", (panel_x + 5, y0), font, fs * 0.85, (200, 120, 200), th)
                        y0 += dy + 3

                    cv2.putText(canvas, "=== AXIS LEGEND ===", (panel_x + 5, y0), font, 0.32, (0, 255, 255), th)
                    y0 += dy + 2

                    axis_ox = panel_x + 60
                    axis_oy = y0 + 25
                    axis_len = 25
                    for i in range(3):
                        dx = int([1, 0, 0.4][i] * axis_len)
                        dy = int([0, 1, 0.4][i] * axis_len)
                        ex, ey = axis_ox + dx, axis_oy + dy
                        cv2.arrowedLine(canvas, (axis_ox, axis_oy), (ex, ey), AXIS_COLORS_SW[i], 2, tipLength=0.15)
                    cv2.putText(canvas, "CAM", (axis_ox - 12, axis_oy - axis_len - 5), font, 0.25, (255, 255, 255), 1)

                    axis2_ox = panel_x + 180
                    for i in range(3):
                        axis_cam = R_cam2base_sw_inv[:, i]
                        dx = int(axis_cam[0] * axis_len)
                        dy = int(axis_cam[1] * axis_len)
                        ex, ey = axis2_ox + dx, axis_oy + dy
                        cv2.arrowedLine(canvas, (axis2_ox, axis_oy), (ex, ey), AXIS_COLORS_SW[i], 2, tipLength=0.15)
                    cv2.putText(canvas, "ARM_BASE", (axis2_ox - 25, axis_oy - axis_len - 5), font, 0.25, (255, 150, 255), 1)

                    axis3_ox = panel_x + 310
                    for i in range(3):
                        axis_cam = R_cam2base_sw[:, i]
                        dx = int(axis_cam[0] * axis_len)
                        dy = int(axis_cam[1] * axis_len)
                        ex, ey = axis3_ox + dx, axis_oy + dy
                        cv2.arrowedLine(canvas, (axis3_ox, axis_oy), (ex, ey), AXIS_COLORS_SW[i], 2, tipLength=0.15)
                    cv2.putText(canvas, "BASE(proj)", (axis3_ox - 25, axis_oy - axis_len - 5), font, 0.25, (255, 150, 255), 1)

                    cv2.putText(canvas, "R=X G=Y B=Z", (panel_x + 5, axis_oy + axis_len + 12),
                                font, 0.22, (180, 180, 180), 1)

            if wt is not None:
                wrist_pos_camera_wt_sw = wt.get("wrist_pos_camera")
                wrist_rot_mano_wt_sw = wt.get("wrist_rot_mano")
                R_cam2base_sw_trail = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

                if wrist_pos_camera_wt_sw is not None:
                    wrist_base_sw = R_cam2base_sw_trail @ np.array(wrist_pos_camera_wt_sw)
                    if not hasattr(self, '_sw_trail'):
                        self._sw_trail = []
                        self._sw_trail_rot = []
                    self._sw_trail.append(wrist_base_sw.copy())
                    if wrist_rot_mano_wt_sw is not None:
                        self._sw_trail_rot.append(R_cam2base_sw_trail @ np.array(wrist_rot_mano_wt_sw))
                    else:
                        self._sw_trail_rot.append(np.eye(3))
                    if len(self._sw_trail) > 120:
                        self._sw_trail = self._sw_trail[-120:]
                        self._sw_trail_rot = self._sw_trail_rot[-120:]

                if len(self._sw_trail) > 2:
                    trail_panel_x = 10
                    trail_panel_y = H - 180
                    trail_panel_w = 300
                    trail_panel_h = 170
                    overlay_sw = canvas.copy()
                    cv2.rectangle(overlay_sw, (trail_panel_x, trail_panel_y),
                                  (trail_panel_x + trail_panel_w, trail_panel_y + trail_panel_h), (0, 0, 0), -1)
                    cv2.addWeighted(overlay_sw, 0.7, canvas, 0.3, 0, canvas)

                    trail_data = np.array(self._sw_trail)
                    trail_min = trail_data.min(axis=0)
                    trail_max = trail_data.max(axis=0)
                    trail_range = trail_max - trail_min
                    for ax in range(3):
                        if trail_range[ax] < 0.01:
                            trail_range[ax] = 0.01

                    pad = 15
                    plot_w = trail_panel_w - 2 * pad
                    plot_h = trail_panel_h - 35

                    cv2.putText(canvas, "MANO Wrist Trajectory (base)", (trail_panel_x + 5, trail_panel_y + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

                    AXIS_NAMES_TRAIL = ['X(fwd)', 'Y(left)', 'Z(up)']
                    AXIS_COLORS_TRAIL = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                    for ax in range(3):
                        pts_2d = []
                        for ti, tp in enumerate(self._sw_trail):
                            px = trail_panel_x + pad + int((ti / max(len(self._sw_trail) - 1, 1)) * plot_w)
                            norm_val = (tp[ax] - trail_min[ax]) / trail_range[ax]
                            py = trail_panel_y + 25 + int((1 - norm_val) * plot_h)
                            pts_2d.append((px, py))
                        if len(pts_2d) > 1:
                            cv2.polylines(canvas, [np.array(pts_2d)], False, AXIS_COLORS_TRAIL[ax], 1)
                        cv2.putText(canvas, AXIS_NAMES_TRAIL[ax], (trail_panel_x + trail_panel_w - 60, trail_panel_y + 25 + ax * 14),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.22, AXIS_COLORS_TRAIL[ax], 1)

                    if len(self._sw_trail_rot) > 0:
                        last_rot = self._sw_trail_rot[-1]
                        rot_ox = trail_panel_x + trail_panel_w - 40
                        rot_oy = trail_panel_y + trail_panel_h - 40
                        for ai in range(3):
                            dx = int(last_rot[0, ai] * 20)
                            dy = int(last_rot[1, ai] * 20)
                            cv2.arrowedLine(canvas, (rot_ox, rot_oy), (rot_ox + dx, rot_oy + dy),
                                            AXIS_COLORS_TRAIL[ai], 1, tipLength=0.2)

            cv2.putText(canvas, f"Frame {frame_i + 1}/{len(qpos_sequence)}", (W - 200, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

            writer.write(canvas)

        writer.release()
        self.logger.info(f"  Saved (software): {output_path}")

    def _cleanup_temp_files(self):
        """清理子进程脚本文件（除非 --keep-scripts）"""
        if self.keep_scripts:
            self.logger.info(f"  Scripts kept in output directory (--keep-scripts)")
            return

        for f in self._temp_files:
            if f.exists():
                f.unlink()
                self.logger.info(f"  Cleaned up: {f.name}")

        # 清理中间数据文件
        for name in ["ik_input.pkl", "render_input.pkl"]:
            p = self.output_dir / name
            if p.exists():
                p.unlink()
                self.logger.info(f"  Cleaned up: {name}")

    def _save_info_file(self):
        """步骤4：保存完整的参数信息文件和说明文档"""
        info_path = self.output_dir / "pipeline_info.json"
        with info_path.open("w", encoding="utf-8") as f:
            json.dump(self.pipeline_info, f, indent=2, ensure_ascii=False)
        self.logger.info(f"  Info saved: {info_path}")

        desc_path = self.output_dir / "output_description.txt"
        det = self.pipeline_info.get("detection", {})
        ik = self.pipeline_info.get("ik_result", {})
        ik_extra = self.pipeline_info.get("ik_result_extra", {})
        render = self.pipeline_info.get("render", {})
        cfg = self.pipeline_info.get("pipeline_config", {})
        wrist_stats = det.get("wrist_stats", {})

        desc_lines = [
            f"R1 SAPIEN Simulation Output - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "[输出文件说明]",
            "",
            "  1. r1_sapien_sim.mp4",
            "     R1 机器人仿真视频（SAPIEN 3D / 软件回退）",
            "     视频叠加监测变量HUD：关节角度、末端位置、FK误差、帧率",
            "     包含 Warmup 过渡帧（从初始姿态平滑插值到第一帧IK结果）",
            "",
            "  2. overlay.mp4",
            "     原始视频 + MediaPipe 手部骨架叠加 + 抓取状态标注",
            "",
            "  3. trajectory.pkl",
            "     完整轨迹数据（手部关键点、手腕6DoF位姿、夹爪关节值、抓取状态）",
            "",
            "  4. pipeline_info.log",
            "     详细参数日志（含时间戳），记录管道运行全过程",
            "",
            "  5. pipeline_info.json",
            "     结构化参数数据（机器可读），包含所有配置和统计信息",
            "",
            "  6. render_evaluation.pkl",
            "     渲染评估数据（FK误差、关节舒适度、逐帧详细验证）",
            "",
            "  7. output_description.txt",
            "     本文件，输出说明文档",
            "",
            "=" * 70,
            "",
            "[管道配置]",
            f"  输入视频:       {self.video_path}",
            f"  手别:           {self.hand_type_str} (selfie={self.selfie})",
            f"  工作臂:         {self.arm_side}",
            f"  相机视角:       {self.view}",
            f"  帧率:           {self.fps} fps",
            f"  分辨率:         {self.img_width}x{self.img_height}",
            f"  R1 URDF:        {cfg.get('r1_urdf_path', 'N/A')}",
            f"  抓取阈值:       {cfg.get('grasp_threshold', 'N/A')}",
            f"  释放阈值:       {cfg.get('release_threshold', 'N/A')}",
            f"  Warmup帧数:     {cfg.get('warmup_frames', WARMUP_FRAMES)}",
            f"  安全距离:       {cfg.get('safety_distance', SAFETY_DISTANCE)}m",
            "",
            "[手腕轨迹分析]",
            f"  质心:           {wrist_stats.get('centroid', 'N/A')}",
            f"  运动范围:       X={wrist_stats.get('range', ['N/A'])[0]}, "
            f"Y={wrist_stats.get('range', ['N/A', 'N/A'])[1]}, "
            f"Z={wrist_stats.get('range', ['N/A', 'N/A', 'N/A'])[2]}",
            "",
            "[检测结果]",
            f"  检测率:         {det.get('detection_rate', 'N/A')}",
            f"  有效帧:         {det.get('detected_frames', 'N/A')}/{det.get('total_frames', 'N/A')}",
            f"  间隙填充帧:     {det.get('gap_filled_frames', 'N/A')}",
            f"  抓取帧数:       {det.get('grasp_frames', 'N/A')}",
            f"  张开帧数:       {det.get('open_frames', 'N/A')}",
            "",
            "[IK 求解结果]",
            f"  求解率:         {ik.get('solve_rate', 'N/A')}",
            f"  超出臂展帧:     {ik_extra.get('out_of_reach', 'N/A')}",
            f"  IK失败帧:       {ik_extra.get('ik_errors', 'N/A')}",
            f"  工作空间映射偏移(base): {ik_extra.get('mapping_offset_base', 'N/A')}",
            f"  安全偏移(base):       {ik_extra.get('safety_offset_base', 'N/A')}",
            "",
            "[渲染信息]",
            f"  SAPIEN 渲染:    {'成功' if render.get('sapien_ok') else '失败（使用软件回退）'}",
            "",
            "[夹爪物理限制]",
            f"  finger_joint1:  [0, 0.05] m",
            f"  finger_joint2:  [-0.05, 0] m",
            "",
            "[监测变量说明（视频HUD）]",
            "  [Phase] Frame/FPS/T: 运行阶段/帧号/帧率/已用时间",
            "  --- Gripper ---",
            "  Status:         抓取状态（GRASP/OPEN）和手指距离",
            "  J1/J2:          夹爪两个手指的关节值（米）",
            "  --- Arm Joints ---",
            "  J1-J6:          臂部6关节角度（度）",
            "  --- Kinematics ---",
            "  Base:           臂基座世界坐标",
            "  Wrist:          手腕世界坐标（MediaPipe检测）",
            "  IK tgt:         IK目标世界坐标（映射后）",
            "  EE act:         末端执行器实际位置（FK计算）",
            "  EE tgt:         末端执行器IK目标位置",
            "  FK err:         FK位置误差（厘米），<2cm绿/<5cm橙/>5cm红",
            "  Reach:          到臂基座距离/臂展（%），<70%绿/<90%橙/>90%红",
            "  Comfort:        关节舒适度（1.0=最佳），>0.6绿/>0.3橙/<0.3红",
            "  Map off:        工作空间映射偏移量",
            "",
            "[重新生成命令]",
            f"  conda run -n dex python r1_sapien_sim.py \\",
            f"    --video-path {self.video_path} \\",
            f"    --output-dir {self.output_dir} \\",
            f"    --hand {self.hand_type_str} \\",
            f"    --view {self.view} \\",
            f"    --fps {self.fps}",
        ]
        desc_path.write_text("\n".join(desc_lines), encoding="utf-8")
        self.logger.info(f"  Description saved: {desc_path}")


def main():
    """命令行入口：解析参数并运行管道"""
    parser = argparse.ArgumentParser(
        description="Egocentric Video -> R1 Robot (SAPIEN) Simulation Pipeline"
    )
    parser.add_argument("--video-path", type=str, required=True,
                        help="输入 MP4 视频文件路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录路径（默认自动根据视频文件名生成）")
    parser.add_argument("--fps", type=int, default=30,
                        help="目标帧率（默认 30）")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="起始帧索引（默认 0）")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="处理帧数，-1 表示全部（默认 -1）")
    parser.add_argument("--no-selfie", action="store_true",
                        help="禁用自拍模式（用于非第一人称视角视频）")
    parser.add_argument("--hand", type=str, default="auto",
                        choices=["auto", "Right", "Left"],
                        help="手别：auto=自动检测, Right=右手, Left=左手")
    parser.add_argument("--grasp-threshold", type=float, default=None,
                        help="抓取检测的手指距离阈值（默认 0.04）")
    parser.add_argument("--release-threshold", type=float, default=None,
                        help="释放检测的手指距离阈值（默认 0.06）")
    parser.add_argument("--no-visualize", action="store_true",
                        help="禁用实时检测可视化")
    parser.add_argument("--view", type=str, default="front",
                        choices=["behind", "front", "topdown", "side", "closeup", "egocentric", "all"],
                        help="相机视角模式（默认 front，egocentric为第一人称跟随视角，all=生成所有视角）")
    parser.add_argument("--width", type=int, default=1920,
                        help="输出视频宽度（默认 1920）")
    parser.add_argument("--height", type=int, default=1080,
                        help="输出视频高度（默认 1080）")
    parser.add_argument("--keep-scripts", action="store_true",
                        help="保留子进程脚本文件（默认运行后自动清理）")
    parser.add_argument("--steps", type=str, default="detect,ik,render",
                        help="执行步骤，逗号分隔（detect,ik,render）。"
                        "可单独运行如 --steps detect 仅做检测")

    args = parser.parse_args()

    # 自动生成输出目录名（基于视频文件名）
    output_dir = args.output_dir
    if output_dir is None:
        video_stem = Path(args.video_path).stem
        output_dir = f"./output_{video_stem}_r1sapien"

    pipeline = R1SAPIENPipeline(
        video_path=args.video_path,
        output_dir=output_dir,
        fps=args.fps,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        selfie=not args.no_selfie,
        hand_type=args.hand,
        grasp_threshold=args.grasp_threshold,
        release_threshold=args.release_threshold,
        view=args.view,
        img_width=args.width,
        img_height=args.height,
        keep_scripts=args.keep_scripts,
        steps=args.steps,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
