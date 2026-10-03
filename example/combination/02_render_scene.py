#!/usr/bin/env python3
"""
================================================================================
  02_render_scene.py — Step 2: 渲染仿真场景 (手+物体+机器人)

  管线:
    00_run_pipeline.py  ← 一键入口
    01_align_scene.py   →  02_render_scene.py  →  03_track_robot.py
    (对齐场景)              (渲染仿真场景)         (独立机器人跟踪)

  功能:
    将 HaWoR 手部重建 + RAS 场景重建 + R1 机器人统一到 SAPIEN 仿真引擎中,
    生成第一人称视角的动态仿真视频。

    本脚本是管线的核心, 负责:
    - 加载 HaWoR 手部数据 (MANO 参数 → 21 关节点)
    - 加载 RAS GLB 场景 (按颜色分组渲染, 保留顶点颜色)
    - 驱动 SAPIEN 相机跟随 HaWoR 相机轨迹 (第一人称视角)
    - 将手部姿态映射到 R1 机器人 (Dex Retargeting + RelaxedIK)

  数据源:
    HaWoR 手部重建: hawor_dir/reconstruction/hawor_results_*.npz
      - pred_trans:     (2, N, 3)   世界坐标平移
      - pred_rot:       (2, N, 3)   世界坐标轴角旋转
      - pred_hand_pose: (2, N, 45)  手指 PCA 参数
      - pred_betas:     (2, N, 10)  MANO 形状参数
      - pred_valid:     (2, N)      有效帧标记
      - R_c2w:          (N, 3, 3)   相机旋转 (camera-to-world)
      - t_c2w:          (N, 3)      相机平移 (camera-to-world)

    RAS 场景重建: ras_dir/final_scene.glb
      - 单个几何体, 含 vertex colors (RGBA), 约 384K 顶点 / 769K 面
      - 坐标系: RAS y-down, 米制尺度

    对齐参数: output/alignment/transform_params.npz (由 01_align_scene.py 生成)
      - s_inv, R_inv, t_inv: GLB RAS y-down → HaWoR y-up 变换参数
      - 使用 load_glb_transformed() 直接加载原始 GLB 并变换

  坐标系变换链 (三者同帧 = R_AXIS @ OpenGL_world):
    ┌──────────────────────────────────────────────────────────────────────┐
    │  手部 (HaWoR SLAM world, z-forward, y-down, 米)                     │
    │      ↓ RXWORLD_TO_SAPIEN = R_AXIS @ R_x  (SLAM → SAPIEN)           │
    │  SAPIEN 世界 (帧: R_AXIS @ OpenGL)                                  │
    │                                                                      │
    │  相机 (R_c2w/t_c2w 已应用 R_x, 即 stored = R_x @ SLAM = OpenGL)     │
    │      ↓ hawor_cam_to_sapien_pose                                     │
    │        cam_pos = R_AXIS @ t_c2w  (注意: 用 R_AXIS, 非 RXWORLD!)     │
    │        cam_R   = R_AXIS @ R_c2w  (使相机与手部同帧)                  │
    │        sapien_cam_R: X=+col2(forward), Y=-col0(left), Z=-col1(up)  │
    │  SAPIEN 相机 (X=forward, Y=left, Z=up)                              │
    │                                                                      │
    │  GLB (RAS y-up, 米制)                                               │
    │      ↓ 01_align_scene 变换: p_hawor = s_inv*R_inv@p_ras + t_inv   │
    │  HaWoR SLAM world (z-forward, y-down, 米)                           │
    │      ↓ RXWORLD_TO_SAPIEN                                            │
    │  SAPIEN 世界 (帧: R_AXIS @ OpenGL, 与手部/相机一致)                 │
    └──────────────────────────────────────────────────────────────────────┘

  三种渲染模式:
    hand_only       — MANO 手 + GLB 物体 (第一人称视角, 验证手-物对齐)
    robot_only      — R1 机器人 + GLB 物体 (机器人替代人手操作)
    robot_tracking  — MANO 手 + R1 机器人 + GLB 物体 (对比手部与机器人)

  机器人映射链:
    MANO 21 关节点
        ↓ Dex Retargeting (position retargeting)
    夹爪关节角 (gripper qpos)
        ↓ MANO 手腕朝向 (pred_rot → axis_angle_to_matrix → RXWORLD_TO_SAPIEN)
    末端朝向 (SAPIEN 坐标系)
        ↓ + mapping_offset (映射到臂舒适工作空间)
    IK 目标位置 + 朝向 (base_link 坐标系)
        ↓ RelaxedIK (臂逆运动学, 位置容差1mm, 朝向容差0.1rad≈5.7°)
    臂关节角 (arm qpos)

  GLB 渲染方式:
    01_align_scene.py 计算 RAS Z-UP → HaWoR SLAM world 的变换参数并保存到 transform_params.npz,
    本脚本使用 load_glb_transformed() 以 pipeline_universal.py 风格加载:
      1. trimesh 加载 GLB → 遍历 geometry.items()
      2. 对每个几何体顶点应用变换: p_hawor = s_inv * R_inv @ p_ras + t_inv
      3. 转换到 SAPIEN 坐标系: p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
      4. 导出变换后顶点为临时 PLY, 计算平均顶点颜色
      5. 使用 scene.create_actor_builder().add_visual_from_file() + RenderMaterial 加载

  用法:
    # 单独运行 (需先运行 01_align_scene.py)
    python 02_render_scene.py --mode hand_only --hawor-dir ... --ras-dir ...
    python 02_render_scene.py --mode robot_only --hawor-dir ... --ras-dir ...
    python 02_render_scene.py --mode robot_tracking --hawor-dir ... --ras-dir ...

    # 一键运行 (推荐)
    python 00_run_pipeline.py --hawor-dir ... --ras-dir ...

  输出:
    output/videos/hand_object_{mode}.mp4
================================================================================
"""

import os
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
if os.path.exists(_nvidia_icd):
    os.environ['VK_ICD_FILENAMES'] = _nvidia_icd
else:
    _intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
    os.environ['VK_ICD_FILENAMES'] = _intel_icd

import argparse
import re
import sys
import logging
import tempfile
from pathlib import Path
from glob import glob
from natsort import natsorted

import cv2
import numpy as np
import sapien
import sapien.render
import torch
import joblib
from pytransform3d import rotations as pr
from tqdm import trange

try:
    import trimesh
except ImportError:
    trimesh = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))

# 全局: 是否优先加载 *_depth_aligned.npz（默认 True）
PREFER_DEPTH_ALIGNED = True

from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path, OPERATOR2MANO_RIGHT
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting import yourdfpy as urdf
from mano_layer import MANOLayer

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
FLOATING_RIGHT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_right.urdf"
FLOATING_LEFT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_left.urdf"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"

R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x

RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]
WARMUP_FRAMES = 30
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.35
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
BASE_TRACKING_RANGE = 0.04
BASE_TRACKING_ALPHA = 0.15
SAFETY_DISTANCE = 0.05
LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
IK_SOLVE_PER_FRAME = 20

R1_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798],
    [0.0, 3.2289],
    [-3.3161, 0.0],
    [-2.8798, 2.8798],
    [-1.6581, 1.6581],
    [-2.8798, 2.8798],
])

R_GRIPPER_ALIGN = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=np.float64)

CAM_WIDTH = 1920
CAM_HEIGHT = 1080
HAWOR_FOCAL_DEFAULT = 600.0

MANO_JOINT_NAMES = [
    "wrist",
    "index_MCP", "index_PIP", "index_DIP", "index_tip",
    "middle_MCP", "middle_PIP", "middle_DIP", "middle_tip",
    "pinky_MCP", "pinky_PIP", "pinky_DIP", "pinky_tip",
    "ring_MCP", "ring_PIP", "ring_DIP", "ring_tip",
    "thumb_CMC", "thumb_MCP", "thumb_IP", "thumb_tip",
]

FINGER_GROUP_COLORS = [
    np.array([0.9, 0.9, 0.9, 1.0]),
    np.array([1.0, 0.2, 0.2, 1.0]),
    np.array([0.2, 0.9, 0.2, 1.0]),
    np.array([0.3, 0.5, 1.0, 1.0]),
    np.array([1.0, 0.9, 0.2, 1.0]),
    np.array([1.0, 0.5, 0.0, 1.0]),
]

JOINT_TO_FINGER = [0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]

MANO_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


class TrajectorySmoother:
    SMOOTHNESS_THRESHOLDS = {
        "max_velocity": 3.0,
        "max_acceleration": 8.0,
        "max_jerk": 80.0,
        "si_improvement_min": 0.5,
    }

    def __init__(self, fps=30, max_velocity=1.5, max_acceleration=4.0,
                 max_jerk=20.0, lp_alpha=0.25, max_iterations=10, convergence_eps=1e-5):
        """初始化轨迹平滑器

        Args:
            fps: 帧率, 用于计算 dt
            max_velocity: 关节最大速度 (rad/s)
            max_acceleration: 关节最大加速度 (rad/s²)
            max_jerk: 关节最大加加速度 (rad/s³)
            lp_alpha: Butterworth 低通滤波器截止参数
            max_iterations: 迭代限幅最大次数
            convergence_eps: 收敛阈值
        """
        self.dt = 1.0 / fps
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.lp_alpha = lp_alpha
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps

    def smooth_trajectory(self, qpos_sequence, smooth_indices):
        """对 qpos 序列执行平滑处理

        流程:
        1. 提取需要平滑的关节角 (smooth_indices)
        2. 填充无效帧 (线性插值)
        3. 双向 Butterworth 低通滤波
        4. 迭代限幅 (速度→加速度→加加速度, 反复直到收敛)

        Args:
            qpos_sequence: qpos 列表, None 表示无效帧
            smooth_indices: 需要平滑的关节索引列表

        Returns:
            tuple: (smoothed_qpos_list, metrics_dict)
        """
        n_frames = len(qpos_sequence)
        n_joints = len(smooth_indices)
        trajectory = np.zeros((n_frames, n_joints))
        valid_mask = np.zeros(n_frames, dtype=bool)
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                trajectory[i] = qpos[smooth_indices]
                valid_mask[i] = True
        self._fill_invalid_frames(trajectory, valid_mask)
        trajectory_raw = trajectory.copy()
        trajectory = self._bidirectional_lowpass(trajectory)
        trajectory = self._iterative_clamp(trajectory)
        smoothed_sequence = []
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                qpos_new = qpos.copy()
            else:
                for j in range(i, -1, -1):
                    if qpos_sequence[j] is not None:
                        qpos_new = qpos_sequence[j].copy()
                        break
                else:
                    continue
            qpos_new[smooth_indices] = trajectory[i]
            smoothed_sequence.append(qpos_new)
        metrics = self._compute_metrics(trajectory, trajectory_raw)
        return smoothed_sequence, metrics

    def _fill_invalid_frames(self, trajectory, valid_mask):
        """用线性插值填充无效帧的关节数据

        对每个关节独立插值: 找到有效帧区间, 线性插值填充中间帧。
        首尾无效帧用最近的有效值填充。

        Args:
            trajectory: (N, J) 关节轨迹, 原地修改
            valid_mask: (N,) 有效帧标记
        """
        n_frames = len(trajectory)
        last_valid = 0
        for i in range(n_frames):
            if valid_mask[i]:
                last_valid = i
            else:
                trajectory[i] = trajectory[last_valid]
        first_valid = np.argmax(valid_mask)
        for i in range(first_valid):
            trajectory[i] = trajectory[first_valid]

    def _bidirectional_lowpass(self, trajectory):
        """双向 Butterworth 低通滤波

        先正向滤波, 再反向滤波, 实现零相位延迟平滑。
        使用 scipy.signal.butter 设计 2 阶低通滤波器。

        Args:
            trajectory: (N, J) 关节轨迹

        Returns:
            np.ndarray: (N, J) 平滑后的轨迹
        """
        alpha = self.lp_alpha
        n_frames = len(trajectory)
        s1_fwd = np.zeros_like(trajectory)
        s2_fwd = np.zeros_like(trajectory)
        s1_fwd[0] = trajectory[0]
        s2_fwd[0] = trajectory[0]
        for i in range(1, n_frames):
            s1_fwd[i] = s1_fwd[i - 1] + alpha * (trajectory[i] - s1_fwd[i - 1])
            s2_fwd[i] = s2_fwd[i - 1] + alpha * (s1_fwd[i] - s2_fwd[i - 1])
        s1_bwd = np.zeros_like(trajectory)
        s2_bwd = np.zeros_like(trajectory)
        s1_bwd[-1] = s2_fwd[-1]
        s2_bwd[-1] = s2_fwd[-1]
        for i in range(n_frames - 2, -1, -1):
            s1_bwd[i] = s1_bwd[i + 1] + alpha * (s2_fwd[i] - s1_bwd[i + 1])
            s2_bwd[i] = s2_bwd[i + 1] + alpha * (s1_bwd[i] - s2_bwd[i + 1])
        return s2_bwd

    def _clamp_velocity(self, trajectory):
        """限幅关节速度: 确保相邻帧之间的角速度不超过 max_velocity

        Args:
            trajectory: (N, J) 关节轨迹, 原地修改

        Returns:
            np.ndarray: 修改后的轨迹
        """
        max_delta = self.max_velocity * self.dt
        for i in range(1, len(trajectory)):
            delta = trajectory[i] - trajectory[i - 1]
            clamped = np.clip(delta, -max_delta, max_delta)
            trajectory[i] = trajectory[i - 1] + clamped
        return trajectory

    def _clamp_acceleration(self, trajectory):
        """限幅关节加速度: 确保角加速度不超过 max_acceleration

        Args:
            trajectory: (N, J) 关节轨迹, 原地修改

        Returns:
            np.ndarray: 修改后的轨迹
        """
        max_delta_v = self.max_acceleration * self.dt
        for i in range(2, len(trajectory)):
            v_prev = trajectory[i - 1] - trajectory[i - 2]
            v_curr = trajectory[i] - trajectory[i - 1]
            delta_v = v_curr - v_prev
            clamped_dv = np.clip(delta_v, -max_delta_v, max_delta_v)
            v_curr_clamped = v_prev + clamped_dv
            trajectory[i] = trajectory[i - 1] + v_curr_clamped
        return trajectory

    def _clamp_jerk(self, trajectory):
        """限幅关节加加速度: 确保 jerk 不超过 max_jerk

        Args:
            trajectory: (N, J) 关节轨迹, 原地修改

        Returns:
            np.ndarray: 修改后的轨迹
        """
        max_delta_a = self.max_jerk * self.dt
        for i in range(3, len(trajectory)):
            v_im2 = trajectory[i - 2] - trajectory[i - 3]
            v_im1 = trajectory[i - 1] - trajectory[i - 2]
            v_i = trajectory[i] - trajectory[i - 1]
            a_prev = v_im1 - v_im2
            a_curr = v_i - v_im1
            delta_a = a_curr - a_prev
            clamped_da = np.clip(delta_a, -max_delta_a, max_delta_a)
            a_curr_clamped = a_prev + clamped_da
            v_i_clamped = v_im1 + a_curr_clamped
            trajectory[i] = trajectory[i - 1] + v_i_clamped
        return trajectory

    def _iterative_clamp(self, trajectory):
        """迭代执行速度→加速度→加加速度限幅, 直到收敛

        每轮: clamp_velocity → clamp_acceleration → clamp_jerk
        重复 max_iterations 次或直到变化量 < convergence_eps

        Args:
            trajectory: (N, J) 关节轨迹, 原地修改

        Returns:
            np.ndarray: 修改后的轨迹
        """
        for _ in range(self.max_iterations):
            traj_before = trajectory.copy()
            trajectory = self._clamp_velocity(trajectory)
            trajectory = self._clamp_acceleration(trajectory)
            trajectory = self._clamp_jerk(trajectory)
            max_change = np.max(np.abs(trajectory - traj_before))
            if max_change < self.convergence_eps:
                break
        return trajectory

    def _compute_metrics(self, trajectory_smooth, trajectory_raw):
        """计算平滑前后的运动学指标

        指标: 最大速度, 最大加速度, 最大加加速度 (各关节平均)

        Args:
            trajectory_smooth: (N, J) 平滑后的轨迹
            trajectory_raw: (N, J) 原始轨迹

        Returns:
            dict: 包含 smooth 和 raw 两组指标
        """
        dt = self.dt
        vel_raw = np.diff(trajectory_raw, axis=0) / dt
        acc_raw = np.diff(vel_raw, axis=0) / dt
        jerk_raw = np.diff(acc_raw, axis=0) / dt
        vel_smooth = np.diff(trajectory_smooth, axis=0) / dt
        acc_smooth = np.diff(vel_smooth, axis=0) / dt
        jerk_smooth = np.diff(acc_smooth, axis=0) / dt
        raw_max_vel = float(np.max(np.abs(vel_raw)))
        raw_max_acc = float(np.max(np.abs(acc_raw))) if len(acc_raw) > 0 else 0.0
        raw_max_jerk = float(np.max(np.abs(jerk_raw))) if len(jerk_raw) > 0 else 0.0
        raw_si = float(np.sum(jerk_raw ** 2) * dt) if len(jerk_raw) > 0 else 0.0
        smooth_max_vel = float(np.max(np.abs(vel_smooth)))
        smooth_max_acc = float(np.max(np.abs(acc_smooth))) if len(acc_smooth) > 0 else 0.0
        smooth_max_jerk = float(np.max(np.abs(jerk_smooth))) if len(jerk_smooth) > 0 else 0.0
        smooth_si = float(np.sum(jerk_smooth ** 2) * dt) if len(jerk_smooth) > 0 else 0.0
        thresholds = self.SMOOTHNESS_THRESHOLDS
        si_improvement = 1.0 - smooth_si / max(raw_si, 1e-12)
        all_pass = (smooth_max_vel <= thresholds["max_velocity"] and
                    smooth_max_acc <= thresholds["max_acceleration"] and
                    smooth_max_jerk <= thresholds["max_jerk"] and
                    si_improvement >= thresholds["si_improvement_min"])
        return {
            "raw_max_velocity": raw_max_vel,
            "raw_max_acceleration": raw_max_acc,
            "raw_max_jerk": raw_max_jerk,
            "raw_smoothness_index": raw_si,
            "smooth_max_velocity": smooth_max_vel,
            "smooth_max_acceleration": smooth_max_acc,
            "smooth_max_jerk": smooth_max_jerk,
            "smooth_smoothness_index": smooth_si,
            "velocity_reduction": 1.0 - smooth_max_vel / max(raw_max_vel, 1e-6),
            "acceleration_reduction": 1.0 - smooth_max_acc / max(raw_max_acc, 1e-6),
            "jerk_reduction": 1.0 - smooth_si / max(raw_si, 1e-12),
            "pass_velocity": smooth_max_vel <= thresholds["max_velocity"],
            "pass_acceleration": smooth_max_acc <= thresholds["max_acceleration"],
            "pass_jerk": smooth_max_jerk <= thresholds["max_jerk"],
            "pass_si_improvement": si_improvement >= thresholds["si_improvement_min"],
            "all_pass": all_pass,
            "si_improvement": si_improvement,
        }


class EmaTargetSmoother:
    def __init__(self, pos_alpha=0.3, ori_alpha=0.3):
        self.pos_alpha = pos_alpha
        self.ori_alpha = ori_alpha
        self.pos = None
        self.ori_quat = None

    def smooth(self, pos, ori_quat):
        if self.pos is None:
            self.pos = pos.copy()
            self.ori_quat = ori_quat.copy()
            return self.pos.copy(), self.ori_quat.copy()
        self.pos = self.pos + self.pos_alpha * (pos - self.pos)
        self.ori_quat = self.ori_quat + self.ori_alpha * (ori_quat - self.ori_quat)
        norm = np.linalg.norm(self.ori_quat)
        if norm > 1e-8:
            self.ori_quat /= norm
        if self.ori_quat[0] < 0:
            self.ori_quat = -self.ori_quat
        return self.pos.copy(), self.ori_quat.copy()

    def reset(self):
        self.pos = None
        self.ori_quat = None


def reencode_with_ffmpeg(input_path, output_path, crf=18, fps=30, logger=None):
    """使用 ffmpeg 将视频重编码为 H.264 格式

    OpenCV 的 mp4v 编码器产生的视频兼容性差、体积大,
    用 ffmpeg libx264 重编码后体积更小、兼容性更好。

    Args:
        input_path: 输入视频路径 (mp4v 编码)
        output_path: 输出视频路径 (H.264 编码)
        crf: 恒定质量因子 (0=无损, 18=高质量, 23=默认, 28=低质量)
        fps: 输出帧率
        logger: 日志记录器

    Returns:
        bool: 重编码是否成功 (成功时删除原始文件)
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        if logger:
            logger.warning("  imageio-ffmpeg 未安装，跳过 H.264 重编码")
        return False
    if not os.path.exists(input_path):
        if logger:
            logger.warning(f"  ffmpeg 重编码失败: 输入文件不存在 {input_path}")
        return False

    # 检查输入文件大小 (空文件会导致 ffmpeg 立即失败, 错误信息只有版本号)
    input_size = os.path.getsize(input_path)
    if input_size == 0:
        if logger:
            logger.warning(f"  ffmpeg 重编码失败: 输入文件为空 (0 bytes) {input_path}")
        return False

    # 如果输入和输出路径相同, 使用临时文件避免冲突
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        tmp_path = str(output_path) + ".tmp.mp4"
    else:
        tmp_path = str(output_path)

    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-movflags", "+faststart",
        tmp_path,
    ]
    if logger:
        logger.info(f"  ffmpeg 重编码: CRF={crf}, {fps}fps, libx264")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
            # 如果用了临时文件, 替换原文件
            if tmp_path != str(output_path):
                os.replace(tmp_path, str(output_path))
            # 删除原始输入文件 (如果与输出不同)
            if os.path.abspath(input_path) != os.path.abspath(output_path):
                os.remove(input_path)
            if logger:
                old_size = os.path.getsize(output_path)
                logger.info(f"  ✓ 重编码完成: {output_path} ({old_size / 1024 / 1024:.1f}MB)")
            return True
    if logger:
        # 显示 stderr 最后 300 字符 (实际错误信息在末尾, 不是开头的 build info)
        err_tail = result.stderr[-300:] if result.stderr else "无错误输出"
        logger.warning(f"  ffmpeg 重编码失败 (returncode={result.returncode}): {err_tail}")
    return False


def axis_angle_to_matrix(aa):
    """将轴角表示转换为3x3旋转矩阵

    Args:
        aa: 轴角向量, shape=(3,), 方向=旋转轴, 模长=旋转角度(弧度)

    Returns:
        np.ndarray: 3x3 旋转矩阵
    """
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    return pr.matrix_from_axis_angle(np.array([axis[0], axis[1], axis[2], angle]))


def _find_reconstruction_file(hawor_path, prefer_depth_aligned=True):
    """在 HaWoR 目录中查找重建结果 npz 文件

    Args:
        hawor_path: HaWoR 输出目录路径
        prefer_depth_aligned: True=优先使用 *_depth_aligned.npz, False=只用原始文件

    Returns:
        Path: 找到的 npz 文件路径，或 None
    """
    rec_dir = hawor_path / "reconstruction"
    if not rec_dir.exists():
        return None
    if prefer_depth_aligned:
        for f in rec_dir.glob("hawor_results_*_depth_aligned.npz"):
            return f
    for f in rec_dir.glob("hawor_results_*.npz"):
        if "_depth_aligned" not in str(f):
            return f
    # 兜底: 如果上面没找到，用所有文件（包含 depth_aligned）
    for f in rec_dir.glob("hawor_results_*.npz"):
        return f
    return None


def _detect_hand_idx(hawor_path):
    """自动检测 HaWoR 数据中哪只手是活跃的

    通过检查 cam_space/ 目录下的子目录来判断:
    - 如果只有 0/ 目录 → 左手活跃
    - 如果只有 1/ 目录 → 右手活跃
    - 如果两者都有 → 默认左手 (idx=0)

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        int: 手部索引 (0=左手, 1=右手)，或 None(无法检测)
    """
    hands = _detect_hands(hawor_path)
    if not hands:
        return None
    if len(hands) == 2:
        return 0  # 双手时默认返回左手
    return hands[0]


def _detect_hands(hawor_path):
    """自动检测 HaWoR 数据中活跃的手

    通过检查 pred_valid 有效帧数量判断:
    - 只有左手有效帧 → [0]
    - 只有右手有效帧 → [1]
    - 两手都有有效帧 → [0, 1]
    - 都没有 → []

    Args:
        hawor_path: HaWoR 输出目录路径

    Returns:
        list: 活跃手索引列表, 如 [0], [1], [0, 1]
    """
    hawor_path = Path(hawor_path)

    # 方法1: 通过 cam_space 目录检测
    cam_dir = hawor_path / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if detected:
            return sorted(detected)

    # 方法2: 通过 reconstruction npz 的 pred_valid 检测
    rec_file = _find_reconstruction_file(hawor_path)
    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        if 'pred_valid' in rec:
            pred_valid = rec['pred_valid']
            if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                hands = []
                if pred_valid[0].any():
                    hands.append(0)
                if pred_valid[1].any():
                    hands.append(1)
                return hands

    # 方法3: 通过 world_space_res.pth 检测
    ws_file = hawor_path / "world_space_res.pth"
    if ws_file.exists():
        import torch
        data = torch.load(str(ws_file), map_location='cpu')
        if 'pred_valid' in data:
            pred_valid = data['pred_valid'].numpy()
            if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                hands = []
                if pred_valid[0].any():
                    hands.append(0)
                if pred_valid[1].any():
                    hands.append(1)
                return hands

    return []


def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据

    支持两种数据格式:
    1. reconstruction/hawor_results_*.npz (推荐, 含相机轨迹)
    2. world_space_res.pth (旧格式, 无相机轨迹)

    Args:
        hawor_dir: HaWoR 输出目录路径
        hand_idx: 手部索引 (0=左手, 1=右手)

    Returns:
        dict: 包含以下键:
            - pred_trans: (N, 3) 世界坐标平移
            - pred_rot: (N, 3) 世界坐标轴角旋转
            - pred_hand_pose: (N, 45) 手指 PCA 参数
            - pred_betas: (N, 10) MANO 形状参数
            - pred_valid: (N,) 有效帧标记
            - img_focal: float 相机焦距 (像素)
    """
    hawor_path = Path(hawor_dir)
    rec_file = _find_reconstruction_file(hawor_path, PREFER_DEPTH_ALIGNED)
    ws_file = hawor_path / "world_space_res.pth"

    img_focal = None

    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        pred_trans = rec['pred_trans']
        pred_rot = rec['pred_rot']
        pred_hand_pose = rec['pred_hand_pose']
        pred_betas = rec['pred_betas']
        pred_valid = rec['pred_valid']
        if 'img_focal' in rec:
            img_focal = float(rec['img_focal'])
    elif ws_file.exists():
        ws = joblib.load(str(ws_file))
        pred_trans = ws[0].numpy() if hasattr(ws[0], 'numpy') else np.array(ws[0])
        pred_rot = ws[1].numpy() if hasattr(ws[1], 'numpy') else np.array(ws[1])
        pred_hand_pose = ws[2].numpy() if hasattr(ws[2], 'numpy') else np.array(ws[2])
        pred_betas = ws[3].numpy() if hasattr(ws[3], 'numpy') else np.array(ws[3])
        pred_valid = ws[4] if isinstance(ws[4], np.ndarray) else np.array(ws[4])
    else:
        raise FileNotFoundError(f"未找到 hawor 数据文件: {hawor_path / 'reconstruction' / 'hawor_results_*.npz'} 或 {ws_file}")

    est_focal_file = hawor_path / "est_focal.txt"
    if img_focal is None and est_focal_file.exists():
        try:
            img_focal = float(est_focal_file.read_text().strip())
        except Exception:
            pass

    result = {
        "pred_trans": pred_trans[hand_idx],
        "pred_rot": pred_rot[hand_idx],
        "pred_hand_pose": pred_hand_pose[hand_idx],
        "pred_betas": pred_betas[hand_idx],
        "pred_valid": pred_valid[hand_idx],
        "img_focal": img_focal,
    }

    # 填充NaN帧: 用最近有效帧的值替换NaN, 并将NaN帧标记为invalid
    _fill_nan_frames(result)

    return result


def _fill_nan_frames(data):
    """填充数据中的NaN帧: 用最近有效帧的值替换NaN, 并将NaN帧标记为invalid

    处理 pred_trans, pred_rot, pred_hand_pose, pred_betas 中的NaN值。
    策略: 前向填充(用前一个有效帧的值), 首帧NaN则用后向填充。
    含NaN的帧同时标记为 pred_valid=False。

    Args:
        data: load_hawor_data() 返回的字典, 原地修改
    """
    n_frames = data["pred_trans"].shape[0]
    float_keys = ["pred_trans", "pred_rot", "pred_hand_pose", "pred_betas"]

    # 找出任何字段含NaN的帧
    nan_mask = np.zeros(n_frames, dtype=bool)
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind == 'f':
            nan_mask |= np.any(np.isnan(arr), axis=tuple(range(1, arr.ndim)))

    if not nan_mask.any():
        return

    nan_count = nan_mask.sum()
    # 将NaN帧标记为invalid
    data["pred_valid"][nan_mask] = False

    # 前向填充 + 后向填充
    for key in float_keys:
        arr = data[key]
        if arr.dtype.kind != 'f':
            continue
        # 前向填充
        last_valid = None
        for i in range(n_frames):
            if not nan_mask[i]:
                last_valid = arr[i].copy()
            elif last_valid is not None:
                arr[i] = last_valid
        # 后向填充 (处理开头NaN帧)
        first_valid = None
        for i in range(n_frames - 1, -1, -1):
            if not nan_mask[i]:
                first_valid = arr[i].copy()
            elif first_valid is not None:
                arr[i] = first_valid


def load_hawor_c2w(hawor_dir):
    """加载 HaWoR 相机轨迹 (camera-to-world 变换)

    Args:
        hawor_dir: HaWoR 输出目录路径

    Returns:
        tuple: (R_c2w, t_c2w)
            - R_c2w: (N, 3, 3) 相机旋转矩阵
            - t_c2w: (N, 3) 相机平移向量
            如果文件不存在则返回 (None, None)
    """
    rec_file = _find_reconstruction_file(Path(hawor_dir), PREFER_DEPTH_ALIGNED)
    if rec_file is None:
        return None, None
    rec = np.load(str(rec_file), allow_pickle=True)
    return rec['R_c2w'], rec['t_c2w']


def compute_mano_joints(mano_layer, rot, hand_pose, trans):
    """通过 MANO 正运动学计算手部顶点和关节

    Args:
        mano_layer: MANOLayer 实例
        rot: (3,) 手腕轴角旋转
        hand_pose: (45,) 手指 PCA 参数
        trans: (3,) 手腕平移

    Returns:
        tuple: (vertices, joints)
            - vertices: (778, 3) 手部网格顶点
            - joints: (21, 3) 21个手部关节3D坐标
    """
    p = torch.from_numpy(np.concatenate([rot, hand_pose]).astype(np.float32)).unsqueeze(0)
    t = torch.from_numpy(trans.astype(np.float32)).unsqueeze(0)
    v, j = mano_layer(p, t)
    return v.detach().cpu().numpy()[0], j.detach().cpu().numpy()[0]


def compute_smooth_shading_normal(vertices, faces):
    """计算平滑着色法线 (顶点法线)

    对每个面计算面法线, 然后将共享同一顶点的所有面法线累加并归一化,
    得到平滑的顶点法线, 用于 SAPIEN 渲染。

    Args:
        vertices: (V, 3) 顶点坐标
        faces: (F, 3) 面索引

    Returns:
        np.ndarray: (V, 3) 归一化的顶点法线
    """
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    face_normal = np.cross(v2 - v1, v3 - v1)
    normal = np.zeros_like(vertices)
    normal[faces[:, 0]] += face_normal
    normal[faces[:, 1]] += face_normal
    normal[faces[:, 2]] += face_normal
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return normal


def _detect_glb_up_axis(all_vertices):
    """检测 GLB 坐标系是 Z-UP 还是 Y-UP

    RAS 导出的 GLB 可能是 Y-UP (做了 z-up→y-up 转换) 或 Z-UP (未转换)。
    01_align_scene.py 的对齐变换 R_inv 假设 GLB 是 Y-UP。
    如果 GLB 实际是 Z-UP, 需要先用 ZUP_TO_YUP 转换顶点。

    检测启发式: Z-UP 场景中地板在 z=0, 物体在 z>0;
                Y-UP 场景中地板在 y=0, 物体在 y>0。

    Args:
        all_vertices: (N, 3) 所有 GLB 顶点

    Returns:
        str: "z-up" 或 "y-up"
    """
    FLOOR_THRESHOLD = 0.1
    min_z = all_vertices[:, 2].min()
    min_y = all_vertices[:, 1].min()
    z_is_floor = abs(min_z) < FLOOR_THRESHOLD
    y_is_floor = abs(min_y) < FLOOR_THRESHOLD
    if z_is_floor and not y_is_floor:
        return "z-up"
    if y_is_floor and not z_is_floor:
        return "y-up"
    if z_is_floor and y_is_floor:
        z_at_floor = (abs(all_vertices[:, 2]) < FLOOR_THRESHOLD).sum()
        y_at_floor = (abs(all_vertices[:, 1]) < FLOOR_THRESHOLD).sum()
        return "z-up" if z_at_floor > y_at_floor else "y-up"
    return "y-up"


ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def load_glb_transformed(glb_path, transform_params_path, scene, logger=None):
    """加载 GLB 场景并变换到 SAPIEN 坐标系

    变换链:
      GLB (RAS, 可能 z-up 或 y-up) → Y-UP (如需) → HaWoR SLAM world (z-forward) → SAPIEN (z-up)
    1. 读取 01_align_scene.py 生成的变换参数 (s_inv, R_inv, t_inv)
    2. trimesh 加载 GLB, 自动检测坐标系 (z-up / y-up)
    3. 若 z-up, 先用 ZUP_TO_YUP 转换顶点到 y-up
    4. 对顶点应用变换: p_hawor = s_inv * R_inv @ p_yup + t_inv
    5. 转换到 SAPIEN: p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
    6. 导出临时 PLY, 用 SAPIEN API 加载为 kinematic actor

    Args:
        glb_path: GLB 文件路径
        transform_params_path: transform_params.npz 路径 (由 01_align_scene.py 生成)
        scene: SAPIEN 场景实例
        logger: 日志记录器

    Returns:
        list: SAPIEN actor 列表 (每个 GLB geometry 对应一个 actor)
    """
    params = np.load(str(transform_params_path))
    s_inv = float(params['s_inv'])
    R_inv = params['R_inv']
    t_inv = params['t_inv']
    saved_glb_up_axis = str(params.get('glb_up_axis', 'y-up')) if 'glb_up_axis' in params else None

    if trimesh is None:
        if logger:
            logger.error("  ✗ trimesh 未安装, 无法加载 GLB")
        return []

    if logger:
        size_mb = Path(glb_path).stat().st_size / 1024 / 1024
        logger.info(f"  GLB 文件: {glb_path} ({size_mb:.1f} MB)")

    trimesh_scene = trimesh.load(str(glb_path))
    n_geom = len(trimesh_scene.geometry)
    if logger:
        logger.info(f"  GLB 内容: {n_geom} 个几何体")

    # 自动检测 GLB 坐标系 (优先使用 transform_params 中保存的值)
    all_verts_list = []
    for _, geom in trimesh_scene.geometry.items():
        if len(geom.vertices) > 0:
            all_verts_list.append(geom.vertices)
    if saved_glb_up_axis is not None:
        glb_up_axis = saved_glb_up_axis
    elif all_verts_list:
        glb_up_axis = _detect_glb_up_axis(np.vstack(all_verts_list))
    else:
        glb_up_axis = "y-up"
    need_zup_to_yup = glb_up_axis == "z-up"
    if logger:
        logger.info(f"  GLB 坐标系: {glb_up_axis}{' (将转换到 Y-UP)' if need_zup_to_yup else ''}")

    import gc
    from sapien.core import Pose

    obj_actors = []
    temp_files = []

    # 建立 geom_key → 原始 node_name 映射 (避免使用 trimesh 自动生成的 geometry_N)
    geom_to_node = {}
    for _node_name in trimesh_scene.graph.nodes:
        if _node_name == "world":
            continue
        try:
            _data = trimesh_scene.graph[_node_name]
            if isinstance(_data, tuple) and len(_data) == 2:
                _, _geom_key = _data
                if _geom_key:
                    geom_to_node[_geom_key] = _node_name
        except Exception:
            pass

    for geom_name, geom in trimesh_scene.geometry.items():
        real_name = geom_to_node.get(geom_name, geom_name)
        vertices = geom.vertices.copy()
        if not hasattr(geom, 'faces'):
            continue
        faces = geom.faces.copy()
        if len(vertices) == 0 or len(faces) == 0:
            continue

        # Z-UP → Y-UP 转换 (对齐变换 R_inv 假设 Y-UP 输入)
        if need_zup_to_yup:
            vertices = (ZUP_TO_YUP @ vertices.T).T

        vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
        vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T

        avg_color = None
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            vertex_colors = geom.visual.vertex_colors
            if len(vertex_colors) > 0:
                avg_rgb = vertex_colors[:, :3].mean(axis=0)
                avg_color = [avg_rgb[0]/255.0, avg_rgb[1]/255.0, avg_rgb[2]/255.0, 1.0]

        temp_ply = f'/tmp/glb_actor_{os.getpid()}_{real_name.replace(" ", "_")}.ply'
        geom_transformed = trimesh.Trimesh(
            vertices=vertices_sapien,
            faces=faces,
            visual=geom.visual
        )
        geom_transformed.export(temp_ply)
        temp_files.append(temp_ply)

        builder = scene.create_actor_builder()

        if avg_color is not None:
            material = sapien.render.RenderMaterial(
                base_color=avg_color,
                metallic=0.0,
                roughness=0.7,
                specular=0.3
            )
            builder.add_visual_from_file(filename=temp_ply, material=material)
        else:
            builder.add_visual_from_file(filename=temp_ply)

        actor = builder.build_kinematic(name=real_name)
        actor.set_pose(Pose(p=[0, 0, 0], q=[1, 0, 0, 0]))
        obj_actors.append(actor)

        gc.collect()

    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception:
            pass

    if logger:
        logger.info(f"  ✓ GLB 加载完成: {len(obj_actors)} 个物体 (SAPIEN 公开 API)")
    return obj_actors

def prepare_arm_urdf(src_urdf_path, arm_prefix="right"):
    """准备 R1 浮动臂 URDF: 替换 mesh 路径 + 修改夹爪关节类型

    1. 将 package://r1_v2_1_0/meshes/ 替换为绝对路径
    2. 将 gripper_finger_joint1/2 从 fixed 改为 prismatic (使夹爪可以开合)

    Args:
        src_urdf_path: 原始 URDF 文件路径
        arm_prefix: 臂前缀 ("right" 或 "left")

    Returns:
        str: 修改后的临时 URDF 文件路径
    """
    xml = src_urdf_path.read_text()
    xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")
    xml = re.sub(rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint1"\s+type=")fixed(")', r'\1prismatic\2', xml)
    xml = re.sub(rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint2"\s+type=")fixed(")', r'\1prismatic\2', xml)
    temp_dir = tempfile.mkdtemp(prefix="r1_arm_urdf-")
    temp_path = f"{temp_dir}/{src_urdf_path.name}"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


def setup_scene():
    """创建 SAPIEN 渲染场景, 配置光照和环境

    设置内容:
    - 着色器: default (光栅化)
    - 光线追踪采样: 64 spp
    - 环境贴图: 灰色圆顶 (sky=0.4, ground=0.35)
    - 3个方向光: 主光(带阴影), 补光x2(无阴影)
    - 环境光: 0.5

    Returns:
        sapien.Scene: 配置好的场景实例
    """
    from sapien.asset import create_dome_envmap
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_ray_tracing_samples_per_pixel(64)
    scene = sapien.Scene()
    scene.set_timestep(1 / 240)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35]))
    scene.add_directional_light([1, -1, -1], [2.5, 2.5, 2.5], shadow=True)
    scene.add_directional_light([-1, -0.5, -1], [1.2, 1.2, 1.2], shadow=False)
    scene.add_directional_light([0, 1, -0.5], [0.8, 0.8, 0.8], shadow=False)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    return scene


def make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
    """计算 look-at 相机姿态的四元数

    给定相机位置、目标点和上方向, 计算相机朝向的四元数。
    用于固定第三人称视角渲染。

    Args:
        eye: 相机位置, shape=(3,)
        target: 目标点, shape=(3,)
        up: 上方向, 默认 [0,0,1] (Z轴朝上)

    Returns:
        np.ndarray: 相机朝向四元数 (w,x,y,z)
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    else:
        right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_R = np.eye(3)
    cam_R[:, 0] = forward
    cam_R[:, 1] = -right
    cam_R[:, 2] = cam_up
    cam_quat = pr.quaternion_from_matrix(cam_R)
    return cam_quat


def hawor_cam_to_sapien_pose(R_c2w, t_c2w):
    """将 HaWoR 相机位姿转换为 SAPIEN 相机位姿

    R_c2w/t_c2w 已应用 R_x (SLAM→OpenGL 世界), 但相机约定仍为 OpenCV
    (col0=right, col1=down, col2=forward). SAPIEN 相机约定: col0=forward,
    col1=left, col2=up.

    关键: 相机 transform 必须用 R_AXIS (而非 RXWORLD_TO_SAPIEN), 才能与
    手部/GLB (用 RXWORLD_TO_SAPIEN @ SLAM_data = R_AXIS @ OpenGL_data) 同帧.
    手部 = R_AXIS @ R_x @ SLAM = R_AXIS @ OpenGL, 相机 = R_AXIS @ stored
    (stored = R_x @ SLAM = OpenGL), 两者均在 R_AXIS @ OpenGL 帧中.

    OpenCV 提取: forward = +col2, left = -col0, up = -col1

    Args:
        R_c2w: (3, 3) HaWoR 相机旋转矩阵 (camera-to-world)
        t_c2w: (3,) HaWoR 相机平移向量

    Returns:
        tuple: (cam_pos, cam_quat)
            - cam_pos: (3,) SAPIEN 坐标系下的相机位置
            - cam_quat: (4,) SAPIEN 相机朝向四元数
    """
    cam_pos_sapien = R_AXIS @ t_c2w
    cam_R_sapien = R_AXIS @ R_c2w

    forward = cam_R_sapien[:, 2]
    left = -cam_R_sapien[:, 0]
    up = -cam_R_sapien[:, 1]

    sapien_cam_R = np.eye(3)
    sapien_cam_R[:, 0] = forward
    sapien_cam_R[:, 1] = left
    sapien_cam_R[:, 2] = up

    if np.linalg.det(sapien_cam_R) < 0:
        U, _, VH = np.linalg.svd(sapien_cam_R)
        sapien_cam_R = U @ VH
    cam_quat = pr.quaternion_from_matrix(sapien_cam_R)
    return cam_pos_sapien, cam_quat


class HandObjectRenderer:
    def __init__(self, hawor_dir, ras_dir, transform_params_path, output="hand_object.mp4",
                 fps=30, hand_idx=0, logger=None, viewer=False, crf=18,
                 cam_width=CAM_WIDTH, cam_height=CAM_HEIGHT,
                 view="fpv", smooth=1, fixed_base=True):
        """初始化手部+物体渲染器

        Args:
            hawor_dir: HaWoR 输出目录路径
            ras_dir: RAS 输出目录路径 (含 GLB 场景文件)
            transform_params_path: 01_align_scene.py 输出的变换参数路径
            output: 输出视频路径
            fps: 视频帧率
            hand_idx: 手部索引 (0=左手, 1=右手)
            logger: 日志记录器
            viewer: 是否使用交互式 Viewer 模式 (不保存视频)
            crf: H.264 编码质量因子
            cam_width: 渲染宽度 (像素)
            cam_height: 渲染高度 (像素)
            view: 相机视角模式 (fpv=第一人称, topdown=顶部俯视, behind=后上方, front=正前方)
            smooth: 平滑模式 (0=不平滑, 1=在线EMA, 2=后处理双向滤波)
            fixed_base: 固定基座模式 (True=基座不跟随手腕移动, 始终保持初始位置)
        """
        self.hawor_dir = Path(hawor_dir)
        self.ras_dir = Path(ras_dir)
        self.transform_params_path = Path(transform_params_path)
        self.output = output
        self.fps = fps
        self.hand_idx = hand_idx
        self.viewer = viewer
        self.crf = crf
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.view = view
        self.smooth = smooth
        self.fixed_base = fixed_base
        self.logger = logger or logging.getLogger("HandObjectRender")
        self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / HAWOR_FOCAL_DEFAULT)

        # 自适应手数量: hand_indices 列表
        self.hand_indices = [self.hand_idx]  # 默认单手

    def _update_cam_fov(self, hawor_data):
        """根据 HaWoR 数据中的焦距更新相机视场角

        将 HaWoR 的焦距 (像素) 转换为 SAPIEN 的 FOV (弧度)。
        如果数据中没有焦距信息, 使用默认值。

        Args:
            hawor_data: load_hawor_data() 返回的字典
        """
        img_focal = hawor_data.get("img_focal", None)
        if img_focal is not None and img_focal > 0:
            focal_for_render = img_focal * self.cam_width / 1280.0
            self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / focal_for_render)
            self.logger.info(f"  相机焦距: {img_focal:.1f}px (原始), {focal_for_render:.1f}px (渲染), FOV={np.degrees(self.cam_fov):.1f}°")
        else:
            self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / HAWOR_FOCAL_DEFAULT)
            self.logger.info(f"  相机焦距: 使用默认 {HAWOR_FOCAL_DEFAULT}px, FOV={np.degrees(self.cam_fov):.1f}°")

    def _render_to_sapien(self, pts_render):
        """将 HaWoR SLAM 坐标系的点转换到 SAPIEN 坐标系

        pred_trans/顶点在 SLAM 世界 (z-forward, y-down), 用 RXWORLD_TO_SAPIEN
        (= R_AXIS @ R_x) 转换到 SAPIEN. 这与手部/GLB 一致, 三者同帧.

        Args:
            pts_render: (N, 3) 或 (3,) HaWoR SLAM 坐标

        Returns:
            np.ndarray: 同形状, SAPIEN 坐标系下的点
        """
        pts_sapien = (RXWORLD_TO_SAPIEN @ pts_render.T).T
        return pts_sapien

    def _compute_optimal_fixed_base(self, wrist_positions_sapien):
        """计算机器人基座的最优固定位置和朝向

        策略:
        1. 计算所有有效帧手腕位置的质心
        2. 基座放在质心正上方 COMFORTABLE_REACH (0.35m) 处
        3. 朝向: 绕Z轴旋转180° (让机器人面朝操作者)
        4. 如果手腕质心超出臂最大伸展范围, 沿水平方向拉近

        Args:
            wrist_positions_sapien: (N, 3) 有效帧的手腕位置 (SAPIEN 坐标系)

        Returns:
            tuple: (base_pos, base_quat)
                - base_pos: (3,) 基座位置
                - base_quat: (4,) 基座朝向四元数
        """
        if len(wrist_positions_sapien) == 0:
            return np.array([0.0, 0.0, COMFORTABLE_REACH]), pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))

        wrist_arr = np.array(wrist_positions_sapien)
        centroid = wrist_arr.mean(axis=0)
        wrist_range = wrist_arr.max(axis=0) - wrist_arr.min(axis=0)

        arm_base_pos = centroid.copy()
        arm_base_pos[2] += COMFORTABLE_REACH

        if wrist_range[0] > 0.01:
            arm_base_pos[0] += wrist_range[0] * 0.1

        z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
        arm_base_q = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))

        self.logger.info(f"  手腕质心(SAPIEN): {centroid}")
        self.logger.info(f"  手腕运动范围: X={wrist_range[0]:.4f} Y={wrist_range[1]:.4f} Z={wrist_range[2]:.4f}")
        self.logger.info(f"  最优固定基座位置: {arm_base_pos}")

        max_dist = 0
        for wp in wrist_positions_sapien:
            d = np.linalg.norm(wp - arm_base_pos)
            if d > max_dist:
                max_dist = d
        self.logger.info(f"  基座到最远手腕距离: {max_dist:.4f}m (臂展={ARM_MAX_REACH:.3f}m)")
        if max_dist > ARM_MAX_REACH * 0.9:
            self.logger.warning(f"  ⚠ 最远手腕距离 {max_dist:.4f}m 接近臂展 {ARM_MAX_REACH:.3f}m, IK可能不稳定!")

        return arm_base_pos, arm_base_q

    @staticmethod
    def _compute_tracking_base_pos(initial_base_pos, wrist_pos_sapien, arm_base_q):
        """计算跟踪模式下的基座位置 (小范围跟随手腕)

        基座在初始位置基础上, 沿 XY 方向跟踪手腕 (±4cm),
        Z 方向保持固定。这样机器人不需要 mapping_offset 也能
        跟上手部运动。

        Args:
            initial_base_pos: (3,) 初始基座位置
            wrist_pos_sapien: (3,) 当前帧手腕位置
            arm_base_q: (4,) 基座朝向四元数

        Returns:
            np.ndarray: (3,) 调整后的基座位置
        """
        base_R = pr.matrix_from_quaternion(arm_base_q)
        wrist_in_base = base_R.T @ (wrist_pos_sapien - initial_base_pos)
        offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE
        clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)
        delta_world = base_R @ clamped_offset
        return initial_base_pos + delta_world

    def _update_hand_mesh(self, vertex_sapien, mano_face, mat_hand, context, internal_scene, hand_nodes):
        """更新 MANO 手部网格的渲染节点

        将 MANO 顶点和面转换为 SAPIEN 内部渲染格式,
        更新已有的 hand_nodes 或创建新节点。

        Args:
            vertex_sapien: (778, 3) SAPIEN 坐标系下的手部顶点
            mano_face: (F, 3) MANO 面索引
            mat_hand: 手部材质 (红色半透明)
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景
            hand_nodes: 已有的手部渲染节点列表 (可能为空)

        Returns:
            list: 更新后的手部渲染节点列表
        """
        for node in hand_nodes:
            internal_scene.remove_node(node)
        hand_nodes.clear()
        normal = compute_smooth_shading_normal(vertex_sapien, mano_face)
        mesh = context.create_mesh_from_array(np.ascontiguousarray(vertex_sapien), mano_face, normal)
        model = context.create_model([mesh], [mat_hand])
        node = internal_scene.add_node()
        node.set_position([0, 0, 0])
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = True
        obj.transparency = 0
        hand_nodes.append(node)
        return hand_nodes

    def _render_keypoints(self, joints_sapien, context, internal_scene, kp_nodes,
                          radius=0.005, ref_indices=None):
        """渲染手部关键点为球体

        为每个关节创建一个小球体, ref_indices 中的关节用不同颜色。

        Args:
            joints_sapien: (21, 3) SAPIEN 坐标系下的关节位置
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景
            kp_nodes: 已有的关键点渲染节点列表 (先清除再重建)
            radius: 球体半径 (米)
            ref_indices: retargeting 参考关节的索引集合 (用绿色标记)

        Returns:
            list: 新创建的关键点渲染节点列表
        """
        for node in kp_nodes:
            internal_scene.remove_node(node)
        kp_nodes.clear()

        mat_ref = context.create_material(
            np.zeros(4), np.array([1.0, 0.0, 1.0, 1.0]), 0.0, 0.5, 0
        )

        for i, joint_pos in enumerate(joints_sapien):
            if ref_indices is not None and i not in ref_indices:
                continue
            r = radius * 2.0

            sphere = context.create_uvsphere_mesh(12, 6)
            model = context.create_model([sphere], [mat_ref])
            node = internal_scene.add_node()
            node.set_position(joint_pos.tolist())
            node.set_scale([r, r, r])
            obj = internal_scene.add_object(model, node)
            obj.shading_mode = 0
            obj.cast_shadow = False
            obj.transparency = 0
            kp_nodes.append(node)

        return kp_nodes

    def _render_cylinder_between(self, p1, p2, radius, mat, context, internal_scene):
        """在两点之间渲染一个圆柱体 (用于手部骨架线)

        Args:
            p1: 起点, shape=(3,)
            p2: 终点, shape=(3,)
            radius: 圆柱半径 (米)
            mat: 材质
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景

        Returns:
            渲染节点, 或 None (如果两点距离太近)
        """
        mid = (p1 + p2) / 2.0
        length = np.linalg.norm(p2 - p1)
        if length < 1e-6:
            return None
        cylinder = context.create_capsule_mesh(radius, length / 2, 8, 4)
        model = context.create_model([cylinder], [mat])
        node = internal_scene.add_node()
        node.set_position(mid.tolist())
        direction = (p2 - p1) / length
        z_axis = np.array([0, 0, 1.0])
        rot_axis = np.cross(z_axis, direction)
        rot_axis_len = np.linalg.norm(rot_axis)
        if rot_axis_len > 1e-6:
            rot_axis_n = rot_axis / rot_axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rot_quat_wxyz = pr.quaternion_from_axis_angle(
                np.array([rot_axis_n[0], rot_axis_n[1], rot_axis_n[2], angle])
            )
            rot_quat_xyzw = pr.quaternion_xyzw_from_wxyz(rot_quat_wxyz)
            node.set_rotation(rot_quat_xyzw.tolist())
        elif np.dot(z_axis, direction) < 0:
            node.set_rotation([0, 1, 0, 0])
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 0
        return node

    def _compute_ee_orientation_from_wrist(self, wrist_R_sapien):
        """从手腕旋转矩阵计算末端执行器朝向

        将 MANO 手腕朝向 (operator 坐标系) 转换为世界坐标系下的旋转矩阵,
        用于 IK 求解的目标朝向。

        Args:
            wrist_R_sapien: (3, 3) SAPIEN 坐标系下的手腕旋转矩阵

        Returns:
            np.ndarray: (3, 3) 末端执行器在世界坐标系下的旋转矩阵
        """
        R_mano2world = wrist_R_sapien @ OPERATOR2MANO_RIGHT.T
        return R_mano2world

    def _render_hand_skeleton(self, joints_sapien, context, internal_scene, skel_nodes,
                              radius=0.002):
        """渲染手部骨架线 (关节之间的圆柱体连接)

        连接关系: 手腕→每指根→每指节, 共 20 条线

        Args:
            joints_sapien: (21, 3) SAPIEN 坐标系下的关节位置
            context: SAPIEN 渲染上下文
            internal_scene: SAPIEN 内部场景
            skel_nodes: 已有的骨架渲染节点列表 (先清除再重建)
            radius: 圆柱体半径 (米)

        Returns:
            list: 新创建的骨架渲染节点列表
        """
        for node in skel_nodes:
            internal_scene.remove_node(node)
        skel_nodes.clear()
        for j1, j2 in MANO_SKELETON:
            finger_idx = JOINT_TO_FINGER[j1] if j1 < len(JOINT_TO_FINGER) else 0
            color = FINGER_GROUP_COLORS[finger_idx].copy()
            color[3] = 0.8
            p1 = joints_sapien[j1, :3]
            p2 = joints_sapien[j2, :3]
            mat = context.create_material(np.zeros(4), color, 0.0, 0.5, 0)
            node = self._render_cylinder_between(p1, p2, radius, mat, context, internal_scene)
            if node is not None:
                skel_nodes.append(node)
        return skel_nodes

    def _compute_wrist_positions_sapien(self, hawor_data, mano_layer, start_frame, num_frames):
        """预计算所有帧的手腕位置 (SAPIEN 坐标系)

        用于确定机器人基座的最优放置位置。

        Args:
            hawor_data: load_hawor_data() 返回的字典
            mano_layer: MANOLayer 实例
            start_frame: 起始帧索引
            num_frames: 帧数

        Returns:
            list: 有效帧的手腕位置列表, 每个元素为 (3,) ndarray
        """
        positions = []
        for i in range(num_frames):
            global_idx = start_frame + i
            if not hawor_data["pred_valid"][global_idx]:
                continue
            rot = hawor_data["pred_rot"][global_idx]
            trans = hawor_data["pred_trans"][global_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)):
                continue
            _, j = compute_mano_joints(mano_layer, rot,
                                       hawor_data["pred_hand_pose"][global_idx], trans)
            joints_sapien = self._render_to_sapien(j)
            positions.append(joints_sapien[0, :3].copy())
        return positions

    @staticmethod
    def _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, arm_prefix="right"):
        """从 retargeting 优化器的正运动学获取夹爪位姿

        在 retargeting 内部机器人上执行 FK, 获取夹爪连杆的世界位姿。
        这个位姿用于:
        1. 计算 IK 目标位置 (加上 mapping_offset + safety_offset)
        2. 计算 IK 目标朝向 (夹爪朝向)

        Args:
            retargeting: DexRetargeting 优化器实例
            retarget_qpos: retargeting 输出的关节角
            arm_prefix: 臂前缀 ("right" 或 "left")

        Returns:
            tuple: (gripper_pos, gripper_R)
                - gripper_pos: (3,) 夹爪世界位置
                - gripper_R: (3, 3) 夹爪世界旋转矩阵
        """
        internal_robot = retargeting.optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)
        target_name = f"{arm_prefix}_gripper_link"
        for i, name in enumerate(internal_robot.link_names):
            if name == target_name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        raise RuntimeError(f"内部机器人中找不到 {target_name}")

    def run_hand_only(self, start_frame=0, num_frames=-1):
        """模式1: 只渲染 MANO 手部 + GLB 场景物体 (验证对齐效果)

        流程:
        1. 加载 HaWoR 手部数据 + 相机轨迹
        2. 创建 SAPIEN 场景, 加载 GLB 物体
        3. 逐帧: MANO FK → 渲染手部网格/骨架/关键点 → 渲染
        4. 输出视频 (含 ffmpeg H.264 重编码)

        不涉及机器人, 仅用于验证手部与场景的对齐是否正确。

        Args:
            start_frame: 起始帧索引
            num_frames: 渲染帧数 (-1 表示全部)
        """
        self.logger.info("模式1: MANO 手部 + GLB 物体 (01 对齐)")
        self.logger.info("=" * 80)

        self.logger.info("\n[1/5] 加载数据 ...")
        hawor_data = load_hawor_data(self.hawor_dir, self.hand_idx)
        total_frames = hawor_data["pred_trans"].shape[0]
        if num_frames < 0 or num_frames > total_frames - start_frame:
            num_frames = total_frames - start_frame
        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_side = "left" if self.hand_idx == 0 else "right"
        mano_layer = MANOLayer(mano_side, betas_mean)
        mano_face = mano_layer.f.cpu().numpy()

        self._update_cam_fov(hawor_data)

        self.logger.info("\n[2/5] 创建 SAPIEN 场景 + 加载 GLB ...")
        scene = setup_scene()
        internal_scene = scene.render_system._internal_scene
        context = sapien.render.SapienRenderer()._internal_context
        mat_hand = context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1.0]), 0.0, 0.8, 0)

        glb_path = self.ras_dir / "final_scene.glb"
        obj_nodes = []
        if glb_path.exists() and self.transform_params_path.exists():
            obj_actors = load_glb_transformed(glb_path, self.transform_params_path, scene, logger=self.logger)
            if obj_actors:
                self.logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")
            else:
                self.logger.error(f"  ✗ GLB 加载失败")
        else:
            if not glb_path.exists():
                self.logger.error(f"  ✗ GLB 文件不存在: {glb_path}")
            if not self.transform_params_path.exists():
                self.logger.error(f"  ✗ 变换参数不存在: {self.transform_params_path}")
                self.logger.error(f"  请先运行: python 01_align_scene.py ...")

        self.logger.info("\n[3/5] 设置相机 ...")
        camera = scene.add_camera("main", self.cam_width, self.cam_height, self.cam_fov, 0.01, 100.0)

        if R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  使用 hawor 相机轨迹 ({R_c2w_all.shape[0]}帧)")
            self.logger.info(f"  cam[0] pos(SAPIEN): {cam_pos}")
        else:
            wrist_positions = self._compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
            if wrist_positions:
                centroid = np.mean(wrist_positions, axis=0)
                cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
                cam_quat = make_look_at_camera(cam_pos, centroid)
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                self.logger.info(f"  手腕质心: {centroid}")
                self.logger.info(f"  相机位置: {cam_pos}")
            else:
                camera.set_local_pose(sapien.Pose([0.3, -0.3, 0.2], [1, 0, 0, 0]))

        self.logger.info("\n[4/5] 验证对齐 ...")
        for i in range(min(3, num_frames)):
            global_idx = start_frame + i
            if not hawor_data["pred_valid"][global_idx]:
                continue
            _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                       hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
            joints_sapien = self._render_to_sapien(j)
            self.logger.info(f"  帧{global_idx} 手腕(SAPIEN): {joints_sapien[0, :3]}")

        if self.viewer:
            self.logger.info("\n[5/5] 启动交互式 Viewer (手部 + GLB + 机械臂) ...")
            from sapien.utils import Viewer
            from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

            viewer = Viewer()
            viewer.set_scene(scene)
            viewer.control_window.show_origin_frame = True
            viewer.control_window.show_grid = False

            arm_urdf_path = prepare_arm_urdf(FLOATING_RIGHT_URDF)
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True
            robot = loader.load(arm_urdf_path)

            joint_names = [j.name for j in robot.get_active_joints()]
            arm_joint_indices = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
            gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
            gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

            for joint in robot.get_active_joints():
                joint.set_drive_property(stiffness=100000.0, damping=10000.0)

            init_qpos = robot.get_qpos().copy()
            for j, idx in enumerate(arm_joint_indices):
                if j < len(RIGHT_ARM_STARTING):
                    init_qpos[idx] = RIGHT_ARM_STARTING[j]
            init_qpos[gripper_idx1] = 0.04
            init_qpos[gripper_idx2] = -0.04
            robot.set_qpos(init_qpos)

            robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
            RetargetingConfig.set_default_urdf_dir(str(robot_dir))
            config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, HandType.right)
            override = dict(
                add_dummy_free_joint=True,
                normal_delta=1e-5,
                huber_delta=0.01,
                target_link_names=[
                    "right_gripper_finger_link1",
                    "right_gripper_finger_link2",
                    "right_gripper_link",
                ],
                target_link_human_indices=np.array([4, 8, 0]),
            )
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            ref_indices = retargeting.optimizer.target_link_human_indices
            fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed

            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
            ).astype(int)
            sapien2retarget = {}
            for sapien_i, retarget_i in enumerate(retarget2sapien):
                sapien2retarget[retarget_i] = sapien_i
            fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
            for i, retarget_idx in enumerate(fixed_retarget_indices):
                if retarget_idx in sapien2retarget:
                    fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

            self.logger.info(f"  重定向索引: {ref_indices} (3约束点: 4=拇指尖, 8=食指尖, 0=手腕)")

            wrist_positions = self._compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
            arm_base_pos, arm_base_q = self._compute_optimal_fixed_base(wrist_positions)
            robot.set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
            self.logger.info(f"  机器人初始基座: {arm_base_pos} (跟踪范围±{BASE_TRACKING_RANGE}m)")

            scene.step()
            scene.update_render()

            base_link_p, base_link_q = None, None
            for link in robot.get_links():
                if "right_arm_base_link" == link.get_name():
                    pose = link.get_entity_pose()
                    base_link_p = np.array(pose.p)
                    base_link_q = np.array(pose.q)
                    break
            base_link_R = pr.matrix_from_quaternion(base_link_q)
            base_link_R_inv = base_link_R.T

            origin_sphere_radius = 0.015
            mat_ras_origin = context.create_material(np.zeros(4), np.array([1.0, 0.0, 0.0, 1.0]), 0.0, 0.5, 0)
            mat_hawor_origin = context.create_material(np.zeros(4), np.array([0.0, 0.0, 1.0, 1.0]), 0.0, 0.5, 0)
            mat_glb_center = context.create_material(np.zeros(4), np.array([0.0, 0.8, 0.0, 1.0]), 0.0, 0.5, 0)

            hawor_origin_sapien = np.array([0.0, 0.0, 0.0])
            hawor_origin_sphere = context.create_uvsphere_mesh(12, 6)
            hawor_origin_model = context.create_model([hawor_origin_sphere], [mat_hawor_origin])
            hawor_origin_node = internal_scene.add_node()
            hawor_origin_node.set_position(hawor_origin_sapien.tolist())
            hawor_origin_node.set_scale([origin_sphere_radius, origin_sphere_radius, origin_sphere_radius])
            hawor_origin_obj = internal_scene.add_object(hawor_origin_model, hawor_origin_node)
            hawor_origin_obj.shading_mode = 0

            ras_origin_ras = np.zeros(3)
            if self.transform_params_path.exists():
                tp = np.load(str(self.transform_params_path))
                ras_origin_sapien = RXWORLD_TO_SAPIEN @ (tp['s_inv'] * (tp['R_inv'] @ ras_origin_ras) + tp['t_inv'])
                ras_origin_sphere = context.create_uvsphere_mesh(12, 6)
                ras_origin_model = context.create_model([ras_origin_sphere], [mat_ras_origin])
                ras_origin_node = internal_scene.add_node()
                ras_origin_node.set_position(ras_origin_sapien.tolist())
                ras_origin_node.set_scale([origin_sphere_radius, origin_sphere_radius, origin_sphere_radius])
                ras_origin_obj = internal_scene.add_object(ras_origin_model, ras_origin_node)
                ras_origin_obj.shading_mode = 0
                self.logger.info(f"  RAS原点(红) SAPIEN: {ras_origin_sapien}")
                self.logger.info(f"  HaWoR原点(蓝) SAPIEN: {hawor_origin_sapien}")
                self.logger.info(f"  红色=RAS坐标系原点, 蓝色=HaWoR坐标系原点")

            ik_solver = RelaxedIKSolver(
                left_setting_file_path=str(R1_LEFT_SETTINGS),
                right_setting_file_path=str(R1_RIGHT_SETTINGS),
                tolerances=IK_TOLERANCES,
            )
            ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

            ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
            joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
            current_joints = np.array([init_qpos[i] for i in arm_joint_indices])
            joint_filter.next(current_joints)

            right_gripper_link = None
            for link in robot.get_links():
                if "right_gripper_link" in link.get_name():
                    right_gripper_link = link
                    break

            mapping_offset = np.zeros(3)
            safety_offset = np.zeros(3)

            for probe_idx in range(num_frames):
                gidx = start_frame + probe_idx
                if not hawor_data["pred_valid"][gidx]:
                    continue
                _, j_probe = compute_mano_joints(mano_layer, hawor_data["pred_rot"][gidx],
                                                  hawor_data["pred_hand_pose"][gidx], hawor_data["pred_trans"][gidx])
                joints_sapien_probe = self._render_to_sapien(j_probe)
                wrist_R_hawor = pr.matrix_from_compact_axis_angle(hawor_data["pred_rot"][gidx].flatten())
                wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_hawor @ RXWORLD_TO_SAPIEN.T
                wrist_quat_sapien = pr.quaternion_from_matrix(wrist_R_sapien)
                retargeting.warm_start(
                    joints_sapien_probe[0, :], wrist_quat_sapien,
                    hand_type=HandType.right, is_mano_convention=True,
                )
                self.logger.info(f"  warm_start: 用帧{gidx}的手腕位姿初始化优化器")
                break

            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
                viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

            self.logger.info("  关闭窗口退出 ...")
            hand_nodes = []
            kp_nodes = []
            skel_nodes = []
            local_idx = 0
            while not viewer.closed:
                global_idx = start_frame + (local_idx % num_frames)

                if R_c2w_all is not None and t_c2w_all is not None:
                    cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

                if hawor_data["pred_valid"][global_idx]:
                    vertex_render, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                                            hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                    vertex_sapien = self._render_to_sapien(vertex_render)
                    joints_sapien = self._render_to_sapien(j)
                    hand_nodes = self._update_hand_mesh(vertex_sapien, mano_face, mat_hand, context, internal_scene, hand_nodes)
                    kp_nodes = self._render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes,
                                                      radius=0.004, ref_indices=set(ref_indices))
                    skel_nodes = self._render_hand_skeleton(joints_sapien[:, :3], context, internal_scene, skel_nodes)

                    ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                    retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
                    sapien_qpos = retarget_qpos[retarget2sapien]

                    gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(
                        retargeting, retarget_qpos, "right")

                    if self.fixed_base:
                        tracked_base = arm_base_pos.copy()
                        tracked_base_q = arm_base_q.copy()
                    else:
                        tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                        tracked_base_q = arm_base_q
                    robot.set_root_pose(sapien.Pose(tracked_base.tolist(), tracked_base_q.tolist()))
                    scene.step()

                    for link in robot.get_links():
                        if "right_arm_base_link" == link.get_name():
                            pose = link.get_entity_pose()
                            base_link_p = np.array(pose.p)
                            base_link_q = np.array(pose.q)
                            break
                    base_link_R = pr.matrix_from_quaternion(base_link_q)
                    base_link_R_inv = base_link_R.T

                    ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
                    ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
                    ee_R_base = base_link_R_inv @ R_ee_world_fk
                    ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

                    ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())
                    for _ in range(IK_SOLVE_PER_FRAME - 1):
                        ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())

                    filtered_joints = joint_filter.next(np.array(ik_joints))

                    qpos = robot.get_qpos().copy()
                    for j_idx, arm_idx in enumerate(arm_joint_indices):
                        qpos[arm_idx] = filtered_joints[j_idx]
                    if gripper_idx1 < len(sapien_qpos):
                        qpos[gripper_idx1] = float(sapien_qpos[gripper_idx1])
                    if gripper_idx2 < len(sapien_qpos):
                        qpos[gripper_idx2] = float(sapien_qpos[gripper_idx2])
                    robot.set_qpos(qpos)

                scene.step()
                scene.update_render()
                viewer.render()
                local_idx += 1

            for node in hand_nodes + kp_nodes + skel_nodes:
                internal_scene.remove_node(node)
            return

        self.logger.info("\n[5/5] 渲染视频 ...")
        scene.step()
        scene.update_render()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output, fourcc, self.fps, (camera.get_width(), camera.get_height()))
        hand_nodes = []
        kp_nodes = []
        skel_nodes = []

        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, HandType.right)
        override = dict(
            add_dummy_free_joint=True,
            normal_delta=1e-5,
            huber_delta=0.01,
            target_link_names=[
                "right_gripper_finger_link1",
                "right_gripper_finger_link2",
                "right_gripper_link",
            ],
            target_link_human_indices=np.array([4, 8, 0]),
        )
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        ref_indices = retargeting.optimizer.target_link_human_indices
        self.logger.info(f"  重定向索引: {ref_indices} (3约束点: 4=拇指尖, 8=食指尖, 0=手腕)")

        for local_idx in range(num_frames):
            global_idx = start_frame + local_idx

            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            if not hawor_data["pred_valid"][global_idx]:
                scene.step()
                scene.update_render()
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                writer.write(bgr)
                continue

            vertex_render, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                                    hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
            vertex_sapien = self._render_to_sapien(vertex_render)
            joints_sapien = self._render_to_sapien(j)
            hand_nodes = self._update_hand_mesh(vertex_sapien, mano_face, mat_hand, context, internal_scene, hand_nodes)
            kp_nodes = self._render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes,
                                              radius=0.004, ref_indices=set(ref_indices))
            skel_nodes = self._render_hand_skeleton(joints_sapien[:, :3], context, internal_scene, skel_nodes)

            scene.step()
            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(bgr, f"Frame {local_idx + 1}/{num_frames}  |  Hand + Objects",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(bgr)
            if (local_idx + 1) % 10 == 0:
                self.logger.info(f"  已渲染 {local_idx + 1}/{num_frames} 帧 ...")

        writer.release()
        for node in hand_nodes + kp_nodes + skel_nodes:
            internal_scene.remove_node(node)

        final_path = self.output
        tmp_path = str(self.output).replace(".mp4", "_tmp.mp4")
        if os.path.exists(str(self.output)):
            os.rename(str(self.output), tmp_path)
            if reencode_with_ffmpeg(tmp_path, final_path, crf=self.crf, fps=self.fps, logger=self.logger):
                pass
            else:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, final_path)

        self.logger.info(f"\n✓ 视频已保存: {final_path}")

    def run_robot_tracking(self, start_frame=0, num_frames=-1):
        """模式2: 渲染手部 + R1 机器人 + GLB 场景 (完整对比视频)

        流程:
        1. 加载 HaWoR 手部数据 + 相机轨迹
        2. 创建 SAPIEN 场景, 加载 GLB 物体 + R1 浮动臂
        3. 预计算手腕位置 → 确定基座最优位置
        4. 初始化 DexRetargeting (手部关节→夹爪) + RelaxedIK (夹爪→臂关节)
        5. 逐帧实时渲染:
           a. MANO FK → 手部关节
           b. DexRetargeting → 夹爪位置+朝向
           c. 计算跟踪基座位置
           d. RelaxedIK → 臂关节角 (含 joint_filter 低通滤波)
           e. 设置夹爪值 (直接来自 retargeting, 不平滑)
           f. scene.step() + 渲染
        6. 输出视频 + qpos 日志

        Args:
            start_frame: 起始帧索引
            num_frames: 渲染帧数 (-1 表示全部)
        """
        self.logger.info("模式2: R1 机器人跟踪手部 (01 对齐)")
        self.logger.info("=" * 80)

        from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

        self.logger.info("\n[1/8] 加载数据 ...")
        hawor_data = load_hawor_data(self.hawor_dir, self.hand_idx)
        total_frames = hawor_data["pred_trans"].shape[0]
        if num_frames < 0 or num_frames > total_frames - start_frame:
            num_frames = total_frames - start_frame
        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_side = "left" if self.hand_idx == 0 else "right"
        mano_layer = MANOLayer(mano_side, betas_mean)
        mano_face = mano_layer.f.cpu().numpy()

        self._update_cam_fov(hawor_data)

        self.logger.info("\n[2/8] 创建 SAPIEN 场景 + 加载 GLB ...")
        scene = setup_scene()
        internal_scene = scene.render_system._internal_scene
        context = sapien.render.SapienRenderer()._internal_context
        mat_hand = context.create_material(np.zeros(4), np.array([0.96, 0.75, 0.69, 1.0]), 0.0, 0.8, 0)

        glb_path = self.ras_dir / "final_scene.glb"
        obj_actors = []
        if glb_path.exists() and self.transform_params_path.exists():
            obj_actors = load_glb_transformed(glb_path, self.transform_params_path, scene, logger=self.logger)
            if obj_actors:
                self.logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")
            else:
                self.logger.error(f"  ✗ GLB 加载失败")
        else:
            if not glb_path.exists():
                self.logger.error(f"  ✗ GLB 文件不存在: {glb_path}")
            if not self.transform_params_path.exists():
                self.logger.error(f"  ✗ 变换参数不存在: {self.transform_params_path}")

        self.logger.info("\n[3/8] 初始化 R1 单臂机器人 ...")
        arm_urdf_path = prepare_arm_urdf(FLOATING_RIGHT_URDF)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        robot = loader.load(arm_urdf_path)

        joint_names = [j.name for j in robot.get_active_joints()]
        arm_joint_indices = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
        gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
        gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

        for joint in robot.get_active_joints():
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        init_qpos = robot.get_qpos().copy()
        for j, idx in enumerate(arm_joint_indices):
            if j < len(RIGHT_ARM_STARTING):
                init_qpos[idx] = RIGHT_ARM_STARTING[j]
        init_qpos[gripper_idx1] = 0.04
        init_qpos[gripper_idx2] = -0.04
        robot.set_qpos(init_qpos)

        scene.step()
        scene.update_render()
        self.logger.info(f"  ✓ 单臂机器人已加载: {len(arm_joint_indices)}个臂关节 + 2个夹爪关节")

        self.logger.info("\n[4/8] 初始化 Dex Retargeting ...")
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, HandType.right)
        override = dict(
            add_dummy_free_joint=True,
            normal_delta=1e-5,
            huber_delta=0.01,
            target_link_names=[
                "right_gripper_finger_link1",
                "right_gripper_finger_link2",
                "right_gripper_link",
            ],
            target_link_human_indices=np.array([4, 8, 0]),
        )
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        ref_indices = retargeting.optimizer.target_link_human_indices
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed

        retarget2sapien = np.array(
            [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
        ).astype(int)
        sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(retarget2sapien):
            sapien2retarget[retarget_i] = sapien_i
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in sapien2retarget:
                fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]
        self.logger.info("  ✓ Dex Retargeting 就绪")

        for probe_idx in range(num_frames):
            g_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][g_idx]:
                continue
            _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][g_idx],
                                       hawor_data["pred_hand_pose"][g_idx], hawor_data["pred_trans"][g_idx])
            joints_sapien = self._render_to_sapien(j)
            wrist_R_render = pr.matrix_from_compact_axis_angle(hawor_data["pred_rot"][g_idx])
            wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
            wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
            retargeting.warm_start(
                joints_sapien[0, :3], wrist_quat,
                hand_type=HandType.right, is_mano_convention=True,
            )
            self.logger.info(f"  ✓ Warm start 完成 (帧 {g_idx})")
            break

        self.logger.info("\n[5/8] 分析手部轨迹 + 放置机器人 ...")
        wrist_positions = self._compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
        if not wrist_positions:
            raise RuntimeError("无法提取有效手腕位置")

        arm_base_pos, arm_base_q = self._compute_optimal_fixed_base(wrist_positions)
        robot.set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
        scene.step()
        scene.update_render()

        base_link_p, base_link_q = None, None
        for link in robot.get_links():
            if "right_arm_base_link" == link.get_name():
                pose = link.get_entity_pose()
                base_link_p = np.array(pose.p)
                base_link_q = np.array(pose.q)
                break
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T
        self.logger.info(f"  臂基座位置: {arm_base_pos} (跟踪范围±{BASE_TRACKING_RANGE}m)")
        self.logger.info(f"  base_link 位置: {base_link_p}")

        mapping_offset = np.zeros(3)
        safety_offset = np.zeros(3)

        self.logger.info("\n[6/8] 初始化 RelaxedIK ...")
        ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=IK_TOLERANCES,
        )
        ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        current_joints = np.array([init_qpos[i] for i in arm_joint_indices])
        joint_filter.next(current_joints)

        self.logger.info("\n[7/8] 设置相机 (hawor 相机轨迹) ...")
        camera = scene.add_camera("main", self.cam_width, self.cam_height, self.cam_fov, 0.01, 100.0)

        if R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  使用 hawor 相机轨迹 ({R_c2w_all.shape[0]}帧)")
        else:
            centroid = np.mean(wrist_positions, axis=0) if wrist_positions else np.array([0, 0, 0.3])
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  相机位置: {cam_pos}, 看向: {centroid}")

        self.logger.info("\n[8/8] 实时 IK 渲染视频 ...")
        self.logger.info(f"  平滑模式: {self.smooth} ({'不平滑' if self.smooth == 0 else '在线EMA' if self.smooth == 1 else '后处理双向滤波'})")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output, fourcc, self.fps, (camera.get_width(), camera.get_height()))
        hand_nodes = []
        kp_nodes = []
        skel_nodes = []
        wrist_pos_sapien = None
        qpos_log = []
        target_smoother = EmaTargetSmoother(pos_alpha=0.6, ori_alpha=0.6) if self.smooth == 1 else None

        gripper_link = None
        for link in robot.get_links():
            if "right_gripper_link" == link.get_name():
                gripper_link = link
                break

        first_valid_qpos = None
        for fi in range(start_frame, start_frame + num_frames):
            if hawor_data["pred_valid"][fi]:
                _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][fi],
                                           hawor_data["pred_hand_pose"][fi], hawor_data["pred_trans"][fi])
                joints_sapien = self._render_to_sapien(j)
                ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
                gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(
                    retargeting, retarget_qpos, "right")
                if self.fixed_base:
                    tracked_base = arm_base_pos.copy()
                    tracked_base_q = arm_base_q.copy()
                else:
                    tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                    tracked_base_q = arm_base_q
                robot.set_root_pose(sapien.Pose(tracked_base.tolist(), tracked_base_q.tolist()))
                scene.step()
                for link in robot.get_links():
                    if "right_arm_base_link" == link.get_name():
                        pose = link.get_entity_pose()
                        base_link_p_w = np.array(pose.p)
                        base_link_q_w = np.array(pose.q)
                        break
                base_link_R_w = pr.matrix_from_quaternion(base_link_q_w)
                base_link_R_inv_w = base_link_R_w.T
                ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
                ik_target_b = base_link_R_inv_w @ (ik_target_raw - base_link_p_w)
                ee_R_base = base_link_R_inv_w @ R_ee_world_fk
                ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
                ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())
                for _ in range(IK_SOLVE_PER_FRAME * 5 - 1):
                    ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())
                first_valid_qpos = np.array(ik_joints)
                break

        if first_valid_qpos is not None:
            init_qpos = np.array(RIGHT_ARM_STARTING)
            for wi in range(WARMUP_FRAMES):
                t = (wi + 1) / WARMUP_FRAMES
                t = t * t * (3 - 2 * t)
                interp = init_qpos * (1 - t) + first_valid_qpos * t
                qpos = robot.get_qpos().copy()
                for j_idx, arm_idx in enumerate(arm_joint_indices):
                    qpos[arm_idx] = interp[j_idx]
                robot.set_qpos(qpos)
                scene.step()
            self.logger.info(f"  Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep 过渡)")

        for local_idx in trange(num_frames, desc="实时IK渲染"):
            global_idx = start_frame + local_idx

            if self.view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            if hawor_data["pred_valid"][global_idx]:
                vertex_render, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                                        hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                vertex_sapien = self._render_to_sapien(vertex_render)
                joints_sapien = self._render_to_sapien(j)
                hand_nodes = self._update_hand_mesh(vertex_sapien, mano_face, mat_hand, context, internal_scene, hand_nodes)
                kp_nodes = self._render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes,
                                                  radius=0.004, ref_indices=set(ref_indices))
                skel_nodes = self._render_hand_skeleton(joints_sapien[:, :3], context, internal_scene, skel_nodes)
                wrist_pos_sapien = joints_sapien[0, :3].copy()

                ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
                sapien_qpos = retarget_qpos[retarget2sapien]

                gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(
                    retargeting, retarget_qpos, "right")

                if self.fixed_base:
                    tracked_base = arm_base_pos.copy()
                    tracked_base_q = arm_base_q.copy()
                else:
                    tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                    tracked_base_q = arm_base_q
                robot.set_root_pose(sapien.Pose(tracked_base.tolist(), tracked_base_q.tolist()))
                scene.step()

                for link in robot.get_links():
                    if "right_arm_base_link" == link.get_name():
                        pose = link.get_entity_pose()
                        base_link_p = np.array(pose.p)
                        base_link_q = np.array(pose.q)
                        break
                base_link_R = pr.matrix_from_quaternion(base_link_q)
                base_link_R_inv = base_link_R.T

                ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
                ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
                ee_R_base = base_link_R_inv @ R_ee_world_fk
                ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

                if target_smoother is not None:
                    ik_target_b, ee_quat_b = target_smoother.smooth(ik_target_b, ee_quat_b)

                ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())
                for _ in range(IK_SOLVE_PER_FRAME - 1):
                    ik_joints = ik_solver.solve_position_right(ik_target_b.tolist(), ee_quat_b.tolist())

                if self.smooth == 0:
                    filtered_joints = np.array(ik_joints)
                else:
                    filtered_joints = joint_filter.next(np.array(ik_joints))

                qpos = robot.get_qpos().copy()
                for j_idx, arm_idx in enumerate(arm_joint_indices):
                    qpos[arm_idx] = filtered_joints[j_idx]
                if gripper_idx1 < len(sapien_qpos):
                    qpos[gripper_idx1] = float(sapien_qpos[gripper_idx1])
                if gripper_idx2 < len(sapien_qpos):
                    qpos[gripper_idx2] = float(sapien_qpos[gripper_idx2])
                robot.set_qpos(qpos)
                qpos_log.append(qpos.copy())

            scene.step()
            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            ee_err_cm = None
            if gripper_link is not None and wrist_pos_sapien is not None:
                ee_pos = np.array(gripper_link.get_entity_pose().p)
                ee_err_cm = np.linalg.norm(ee_pos - wrist_pos_sapien) * 100
            label = f"Frame {local_idx+1}/{num_frames}  |  R1 + Hand + Objects"
            if ee_err_cm is not None:
                err_color = (0, 255, 0) if ee_err_cm < 2 else (0, 255, 255) if ee_err_cm < 5 else (0, 0, 255)
                label += f"  EE-Gap:{ee_err_cm:.1f}cm"
                cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, err_color, 2)
            else:
                cv2.putText(bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(bgr)

        writer.release()
        for node in hand_nodes + kp_nodes + skel_nodes:
            internal_scene.remove_node(node)

        final_path = self.output
        tmp_path = str(self.output).replace(".mp4", "_tmp.mp4")
        if os.path.exists(str(self.output)):
            os.rename(str(self.output), tmp_path)
            if reencode_with_ffmpeg(tmp_path, final_path, crf=self.crf, fps=self.fps, logger=self.logger):
                pass
            else:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, final_path)

        qpos_path = str(Path(self.output).with_suffix(".npy")).replace("videos", "tracking")
        os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
        if qpos_log:
            qpos_arr = np.array(qpos_log)
            if self.smooth == 2 and len(qpos_arr) > 3:
                self.logger.info("  后处理平滑 (双向Butterworth + 迭代限幅) ...")
                smoother = TrajectorySmoother(fps=self.fps)
                smooth_indices = list(range(len(arm_joint_indices)))
                smoothed, metrics = smoother.smooth_trajectory(qpos_log, smooth_indices)
                qpos_arr = np.array(smoothed)
                self.logger.info(f"    速度降低: {metrics['velocity_reduction']:.1%}, "
                                 f"加速度降低: {metrics['acceleration_reduction']:.1%}, "
                                 f"Jerk降低: {metrics['jerk_reduction']:.1%}")
            np.save(qpos_path, qpos_arr)
            self.logger.info(f"  ✓ qpos 已保存: {qpos_path} ({len(qpos_arr)} 帧)")

        self.logger.info(f"\n✓ 视频已保存: {final_path}")

    def run_robot_only(self, start_frame=0, num_frames=-1):
        """模式3: 只渲染 R1 机器人 + GLB 场景物体 (不渲染人手)

        支持自适应单臂/双臂渲染:
        - 单臂: 根据 self.hand_indices 渲染左手或右手
        - 双臂: 同时渲染左右两个 R1 臂

        与 run_robot_tracking 类似, 但不渲染 MANO 手部网格/骨架/关键点,
        只显示机器人跟踪手部运动。适合生成"机器人操作"视频。

        Args:
            start_frame: 起始帧索引
            num_frames: 渲染帧数 (-1 表示全部)
        """
        hand_count = len(self.hand_indices)
        mode_label = "双臂" if hand_count == 2 else "单臂"
        self.logger.info(f"模式3: R1 机器人手部替代 MANO 手 + GLB 物体 ({mode_label})")
        self.logger.info("=" * 80)

        from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

        self.logger.info("\n[1/7] 加载数据 ...")
        # 加载所有手的数据, 取最小帧数
        all_hawor_data = {}
        total_frames = None
        for hi in self.hand_indices:
            hd = load_hawor_data(self.hawor_dir, hand_idx=hi)
            all_hawor_data[hi] = hd
            n = hd["pred_trans"].shape[0]
            if total_frames is None or n < total_frames:
                total_frames = n
        if num_frames < 0 or num_frames > total_frames - start_frame:
            num_frames = total_frames - start_frame
        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)

        # 更新相机 FOV: 双手时取非 None 的焦距
        if hand_count == 2:
            focal_values = [all_hawor_data[hi].get("img_focal") for hi in self.hand_indices]
            focal = None
            for fv in focal_values:
                if fv is not None and fv > 0:
                    focal = fv
                    break
            if focal is not None:
                focal_for_render = focal * self.cam_width / 1280.0
                self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / focal_for_render)
                self.logger.info(f"  相机焦距: {focal:.1f}px (原始), {focal_for_render:.1f}px (渲染), FOV={np.degrees(self.cam_fov):.1f}°")
            else:
                self.cam_fov = 2 * np.arctan(self.cam_height / 2.0 / HAWOR_FOCAL_DEFAULT)
                self.logger.info(f"  相机焦距: 使用默认 {HAWOR_FOCAL_DEFAULT}px, FOV={np.degrees(self.cam_fov):.1f}°")
        else:
            self._update_cam_fov(all_hawor_data[self.hand_indices[0]])

        self.logger.info("\n[2/7] 创建 SAPIEN 场景 + 加载 GLB ...")
        scene = setup_scene()
        internal_scene = scene.render_system._internal_scene
        context = sapien.render.SapienRenderer()._internal_context

        glb_path = self.ras_dir / "final_scene.glb"
        obj_actors = []
        if glb_path.exists() and self.transform_params_path.exists():
            obj_actors = load_glb_transformed(glb_path, self.transform_params_path, scene, logger=self.logger)
            if obj_actors:
                self.logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")
            else:
                self.logger.error(f"  ✗ GLB 加载失败")
        else:
            if not glb_path.exists():
                self.logger.error(f"  ✗ GLB 文件不存在: {glb_path}")
            if not self.transform_params_path.exists():
                self.logger.error(f"  ✗ 变换参数不存在: {self.transform_params_path}")

        self.logger.info(f"\n[3/7] 初始化 R1 机器人 ({hand_count}个臂) ...")
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        arm_states = []
        for hi in self.hand_indices:
            prefix = "right"
            urdf_path = FLOATING_RIGHT_URDF
            arm_starting = RIGHT_ARM_STARTING

            arm_urdf_path = prepare_arm_urdf(urdf_path, arm_prefix=prefix)
            loader = scene.create_urdf_loader()
            loader.fix_root_link = True
            loader.load_multiple_collisions_from_file = True
            robot = loader.load(arm_urdf_path)

            joint_names = [j.name for j in robot.get_active_joints()]
            arm_joint_indices = [i for i, n in enumerate(joint_names) if f"{prefix}_arm_joint" in n]
            gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
            gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

            for joint in robot.get_active_joints():
                joint.set_drive_property(stiffness=100000.0, damping=10000.0)

            init_qpos = robot.get_qpos().copy()
            for j, idx in enumerate(arm_joint_indices):
                if j < len(arm_starting):
                    init_qpos[idx] = arm_starting[j]
            init_qpos[gripper_idx1] = 0.04
            init_qpos[gripper_idx2] = -0.04
            robot.set_qpos(init_qpos)
            scene.step()
            scene.update_render()

            # Retargeting
            hand_type = HandType.right
            config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
            override = dict(
                add_dummy_free_joint=True,
                normal_delta=1e-5,
                huber_delta=0.01,
                target_link_names=[
                    f"{prefix}_gripper_finger_link1",
                    f"{prefix}_gripper_finger_link2",
                    f"{prefix}_gripper_link",
                ],
                target_link_human_indices=np.array([4, 8, 0]),
            )
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()
            ref_indices = retargeting.optimizer.target_link_human_indices
            fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed

            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
            ).astype(int)
            sapien2retarget = {}
            for sapien_i, retarget_i in enumerate(retarget2sapien):
                sapien2retarget[retarget_i] = sapien_i
            fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
            for i, retarget_idx in enumerate(fixed_retarget_indices):
                if retarget_idx in sapien2retarget:
                    fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

            # IK solver
            ik_solver = RelaxedIKSolver(
                left_setting_file_path=str(R1_LEFT_SETTINGS),
                right_setting_file_path=str(R1_RIGHT_SETTINGS),
                tolerances=IK_TOLERANCES,
            )
            ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

            # MANO layer
            hawor_data = all_hawor_data[hi]
            betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
            mano_side = "left" if hi == 0 else "right"
            mano_layer = MANOLayer(mano_side, betas_mean)

            arm_states.append({
                "prefix": prefix,
                "hand_idx": hi,
                "robot": robot,
                "arm_joint_indices": arm_joint_indices,
                "gripper_idx1": gripper_idx1,
                "gripper_idx2": gripper_idx2,
                "retargeting": retargeting,
                "ref_indices": ref_indices,
                "fixed_qpos": fixed_qpos,
                "retarget2sapien": retarget2sapien,
                "sapien2retarget": sapien2retarget,
                "ik_solver": ik_solver,
                "joint_filter": LPFilter(alpha=LP_ALPHA_JOINT),
                "target_smoother": EmaTargetSmoother(pos_alpha=0.6, ori_alpha=0.6) if self.smooth == 1 else None,
                "first_valid_qpos": None,
                "arm_starting": arm_starting,
                "hawor_data": hawor_data,
                "mano_layer": mano_layer,
                "qpos_log": [],
            })
            self.logger.info(f"  ✓ {prefix} 臂已加载: {len(arm_joint_indices)}个臂关节 + 2个夹爪关节")

        self.logger.info("\n[4/7] 初始化 Dex Retargeting + RelaxedIK ...")
        # Warm start: 对每个臂分别执行
        for arm in arm_states:
            hawor_data = arm["hawor_data"]
            mano_layer = arm["mano_layer"]
            hand_type = HandType.right
            for probe_idx in range(num_frames):
                g_idx = start_frame + probe_idx
                if not hawor_data["pred_valid"][g_idx]:
                    continue
                # 跳过含NaN的帧
                rot = hawor_data["pred_rot"][g_idx]
                trans = hawor_data["pred_trans"][g_idx]
                hand_pose = hawor_data["pred_hand_pose"][g_idx]
                if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                    continue
                _, j = compute_mano_joints(mano_layer, rot,
                                           hawor_data["pred_hand_pose"][g_idx], trans)
                joints_sapien = self._render_to_sapien(j)
                wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
                wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
                wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
                arm["retargeting"].warm_start(
                    joints_sapien[0, :3], wrist_quat,
                    hand_type=hand_type, is_mano_convention=True,
                )
                self.logger.info(f"  ✓ {arm['prefix']} Warm start 完成 (帧 {g_idx})")
                break

        self.logger.info("\n[5/7] 放置机器人 ...")
        all_wrist_positions = []
        for arm in arm_states:
            wp = self._compute_wrist_positions_sapien(arm["hawor_data"], arm["mano_layer"], start_frame, num_frames)
            all_wrist_positions.extend(wp)

        if not all_wrist_positions:
            raise RuntimeError("无法提取有效手腕位置")

        arm_base_pos, arm_base_q = self._compute_optimal_fixed_base(all_wrist_positions)
        for arm in arm_states:
            arm["robot"].set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
            scene.step()
            scene.update_render()

        self.logger.info(f"  初始基座: {arm_base_pos} (跟踪范围±{BASE_TRACKING_RANGE}m)")

        mapping_offset = np.zeros(3)
        safety_offset = np.zeros(3)

        # 初始化 joint_filter
        for arm in arm_states:
            init_qpos_arm = arm["robot"].get_qpos().copy()
            current_joints = np.array([init_qpos_arm[i] for i in arm["arm_joint_indices"]])
            arm["joint_filter"].next(current_joints)

        # Warmup: 对每个臂分别做 smoothstep 过渡
        for arm in arm_states:
            first_valid_qpos = None
            for fi in range(start_frame, start_frame + num_frames):
                if not arm["hawor_data"]["pred_valid"][fi]:
                    continue
                rot = arm["hawor_data"]["pred_rot"][fi]
                trans = arm["hawor_data"]["pred_trans"][fi]
                hand_pose = arm["hawor_data"]["pred_hand_pose"][fi]
                if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                    continue
                _, j = compute_mano_joints(
                    arm["mano_layer"], arm["hawor_data"]["pred_rot"][fi],
                    arm["hawor_data"]["pred_hand_pose"][fi],
                    arm["hawor_data"]["pred_trans"][fi])
                joints_sapien = self._render_to_sapien(j)
                ref_value = joints_sapien[arm["ref_indices"], :].astype(np.float32)
                retarget_qpos = arm["retargeting"].retarget(ref_value, arm["fixed_qpos"])
                sapien_qpos = retarget_qpos[arm["retarget2sapien"]]
                gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(
                    arm["retargeting"], retarget_qpos, arm["prefix"])
                if self.fixed_base:
                    tracked_base = arm_base_pos.copy()
                    tracked_base_q = arm_base_q.copy()
                else:
                    tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                    tracked_base_q = arm_base_q
                arm["robot"].set_root_pose(sapien.Pose(tracked_base.tolist(), tracked_base_q.tolist()))
                scene.step()
                for link in arm["robot"].get_links():
                    if f"{arm['prefix']}_arm_base_link" == link.get_name():
                        pose = link.get_entity_pose()
                        base_link_p_w = np.array(pose.p)
                        base_link_q_w = np.array(pose.q)
                        break
                base_link_R_w = pr.matrix_from_quaternion(base_link_q_w)
                base_link_R_inv_w = base_link_R_w.T
                ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
                ik_target_b = base_link_R_inv_w @ (ik_target_raw - base_link_p_w)
                ee_R_base = base_link_R_inv_w @ R_ee_world_fk
                ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
                solve_fn = arm["ik_solver"].solve_position_left if arm["prefix"] == "left" else arm["ik_solver"].solve_position_right
                ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
                for _ in range(IK_SOLVE_PER_FRAME * 5 - 1):
                    ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
                first_valid_qpos = np.array(ik_joints)
                break

            arm["first_valid_qpos"] = first_valid_qpos
            if first_valid_qpos is not None:
                init_qpos_arm = np.array(arm["arm_starting"])
                for wi in range(WARMUP_FRAMES):
                    t = (wi + 1) / WARMUP_FRAMES
                    t = t * t * (3 - 2 * t)
                    interp = init_qpos_arm * (1 - t) + first_valid_qpos * t
                    qpos = arm["robot"].get_qpos().copy()
                    for j_idx, arm_idx in enumerate(arm["arm_joint_indices"]):
                        qpos[arm_idx] = interp[j_idx]
                    arm["robot"].set_qpos(qpos)
                    scene.step()
                self.logger.info(f"  {arm['prefix']} Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep 过渡)")

        self.logger.info("\n[6/7] 设置相机 ...")
        camera = scene.add_camera("main", self.cam_width, self.cam_height, self.cam_fov, 0.01, 100.0)

        if self.view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            self.logger.info(f"  使用 hawor 相机轨迹 (第一人称)")
        else:
            centroid = np.mean(all_wrist_positions, axis=0) if all_wrist_positions else np.array([0, 0, 0.3])
            robot_root = arm_base_pos.copy()
            if self.view == "topdown":
                cam_target = centroid if centroid is not None else robot_root
                cam_pos = cam_target + np.array([0.0, 0.0, 1.2])
                cam_quat = make_look_at_camera(cam_pos, cam_target, up=np.array([0, 1, 0]))
                self.logger.info(f"  顶部俯视视角 (高度1.2m, 目标={cam_target})")
            elif self.view == "behind":
                cam_pos = robot_root + np.array([2.5, 0.0, 1.2])
                cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
                self.logger.info(f"  后上方视角")
            elif self.view == "front":
                cam_pos = robot_root + np.array([-2.5, 0.0, 1.2])
                cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
                self.logger.info(f"  正前方视角")
            else:
                cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
                cam_quat = make_look_at_camera(cam_pos, centroid)
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

        self.logger.info("\n[7/7] 实时 IK 渲染视频 ...")
        self.logger.info(f"  平滑模式: {self.smooth} ({'不平滑' if self.smooth == 0 else '在线EMA' if self.smooth == 1 else '后处理双向滤波'})")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.output, fourcc, self.fps, (camera.get_width(), camera.get_height()))
        kp_nodes = []

        for local_idx in trange(num_frames, desc="实时IK渲染"):
            global_idx = start_frame + local_idx

            # 更新相机
            if self.view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            arm_valid_flags = []
            for arm in arm_states:
                hawor_data = arm["hawor_data"]
                rot = hawor_data["pred_rot"][global_idx]
                trans = hawor_data["pred_trans"][global_idx]
                hand_pose = hawor_data["pred_hand_pose"][global_idx]
                if hawor_data["pred_valid"][global_idx] and not (np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose))):
                    _, j = compute_mano_joints(arm["mano_layer"], hawor_data["pred_rot"][global_idx],
                                               hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                    joints_sapien = self._render_to_sapien(j)

                    ref_value = joints_sapien[arm["ref_indices"], :].astype(np.float32)
                    retarget_qpos = arm["retargeting"].retarget(ref_value, arm["fixed_qpos"])
                    sapien_qpos = retarget_qpos[arm["retarget2sapien"]]

                    gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(
                        arm["retargeting"], retarget_qpos, arm["prefix"])

                    if self.fixed_base:
                        tracked_base = arm_base_pos.copy()
                        tracked_base_q = arm_base_q.copy()
                    else:
                        tracked_base = self._compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                        tracked_base_q = arm_base_q
                    arm["robot"].set_root_pose(sapien.Pose(tracked_base.tolist(), tracked_base_q.tolist()))
                    scene.step()

                    for link in arm["robot"].get_links():
                        if f"{arm['prefix']}_arm_base_link" == link.get_name():
                            pose = link.get_entity_pose()
                            base_link_p = np.array(pose.p)
                            base_link_q = np.array(pose.q)
                            break
                    base_link_R = pr.matrix_from_quaternion(base_link_q)
                    base_link_R_inv = base_link_R.T

                    ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
                    ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
                    ee_R_base = base_link_R_inv @ R_ee_world_fk
                    ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

                    if arm["target_smoother"] is not None:
                        ik_target_b, ee_quat_b = arm["target_smoother"].smooth(ik_target_b, ee_quat_b)

                    solve_fn = arm["ik_solver"].solve_position_left if arm["prefix"] == "left" else arm["ik_solver"].solve_position_right
                    ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
                    for _ in range(IK_SOLVE_PER_FRAME - 1):
                        ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())

                    if self.smooth == 0:
                        filtered_joints = np.array(ik_joints)
                    else:
                        filtered_joints = arm["joint_filter"].next(np.array(ik_joints))

                    qpos = arm["robot"].get_qpos().copy()
                    for j_idx, arm_idx in enumerate(arm["arm_joint_indices"]):
                        qpos[arm_idx] = filtered_joints[j_idx]
                    if arm["gripper_idx1"] < len(sapien_qpos):
                        qpos[arm["gripper_idx1"]] = float(sapien_qpos[arm["gripper_idx1"]])
                    if arm["gripper_idx2"] < len(sapien_qpos):
                        qpos[arm["gripper_idx2"]] = float(sapien_qpos[arm["gripper_idx2"]])
                    arm["robot"].set_qpos(qpos)
                    arm["qpos_log"].append(qpos.copy())
                    arm_valid_flags.append(True)
                else:
                    arm_valid_flags.append(False)

            scene.step()
            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            if len(arm_states) == 2:
                hand_info = f"L:{'✓' if arm_valid_flags[0] else '✗'} R:{'✓' if arm_valid_flags[1] else '✗'}"
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Dual Arm  |  {hand_info}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                prefix = arm_states[0]["prefix"]
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  R1 {prefix} Arm + Objects",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(bgr)

        writer.release()
        for node in kp_nodes:
            internal_scene.remove_node(node)

        final_path = self.output
        tmp_path = str(self.output).replace(".mp4", "_tmp.mp4")
        if os.path.exists(str(self.output)):
            os.rename(str(self.output), tmp_path)
            if reencode_with_ffmpeg(tmp_path, final_path, crf=self.crf, fps=self.fps, logger=self.logger):
                pass
            else:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, final_path)

        # qpos 保存: 对每个臂分别保存
        for arm in arm_states:
            qpos_path = str(Path(self.output).with_suffix(".npy")).replace("videos", "tracking")
            qpos_path = qpos_path.replace(".npy", f"_{arm['prefix']}.npy")
            os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
            if arm["qpos_log"]:
                qpos_arr = np.array(arm["qpos_log"])
                if self.smooth == 2 and len(qpos_arr) > 3:
                    self.logger.info(f"  {arm['prefix']} 后处理平滑 (双向Butterworth + 迭代限幅) ...")
                    smoother = TrajectorySmoother(fps=self.fps)
                    smooth_indices = list(range(len(arm["arm_joint_indices"])))
                    smoothed, metrics = smoother.smooth_trajectory(arm["qpos_log"], smooth_indices)
                    qpos_arr = np.array(smoothed)
                    self.logger.info(f"    速度降低: {metrics['velocity_reduction']:.1%}, "
                                     f"加速度降低: {metrics['acceleration_reduction']:.1%}, "
                                     f"Jerk降低: {metrics['jerk_reduction']:.1%}")
                np.save(qpos_path, qpos_arr)
                self.logger.info(f"  ✓ {arm['prefix']} qpos 已保存: {qpos_path} ({len(qpos_arr)} 帧)")

        self.logger.info(f"\n✓ 视频已保存: {final_path}")


def main():
    """命令行入口: 渲染 HaWoR 手部 + RAS 场景 + R1 机器人视频

    支持三种模式:
      hand_only:      只渲染 MANO 手部 + GLB 场景物体 (验证对齐效果)
      robot_only:     只渲染 R1 机器人 + GLB 场景物体 (验证机器人跟踪)
      robot_tracking: 渲染手部 + 机器人 + GLB 场景 (完整对比视频)

    用法示例:
      python 02_render_scene.py --mode hand_only --hawor-dir /path/to/hawor --ras-dir /path/to/ras
      python 02_render_scene.py --mode robot_only --hawor-dir /path/to/hawor --ras-dir /path/to/ras --viewer
    """
    parser = argparse.ArgumentParser(description="Hawor 手部 + RAS 物体 → SAPIEN 渲染")
    parser.add_argument("--mode", type=str, default="hand_only", choices=["hand_only", "robot_tracking", "robot_only"])
    parser.add_argument("--hawor-dir", type=str, required=True)
    parser.add_argument("--ras-dir", type=str, required=True)
    parser.add_argument("--transform-params", type=str, default="./output/alignment/transform_params.npz",
                        help="01_align_scene.py 输出的 transform_params.npz 路径")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="手的索引: 0=左手, 1=右手, -1=自动检测, -2=强制双手")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920, help="渲染宽度 (像素)")
    parser.add_argument("--height", type=int, default=1080, help="渲染高度 (像素)")
    parser.add_argument("--crf", type=int, default=14,
                        help="H.264 编码质量 (0=无损, 14=高质量(默认), 18=较好, 23=默认, 28=低质量)")
    parser.add_argument("--viewer", action="store_true", help="交互式Viewer渲染（不保存视频）")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角: fpv=第一人称(默认), topdown=顶部俯视, behind=后上方, front=正前方")
    parser.add_argument("--smooth", type=int, default=1,
                        choices=[0, 1, 2],
                        help="平滑模式: 0=不平滑, 1=在线EMA平滑(默认), 2=后处理双向滤波+限幅")
    parser.add_argument("--fixed-base", action="store_true", default=True,
                        help="固定基座模式 (默认启用; 基座不跟随手腕移动)")
    parser.add_argument("--no-fixed-base", action="store_true",
                        help="禁用固定基座模式 (基座会小范围跟随手腕)")
    parser.add_argument("--no-depth-align", action="store_true",
                        help="不使用 *_depth_aligned.npz，改用原始 hawor_results_*.npz")

    args = parser.parse_args()
    # --no-fixed-base 覆盖 --fixed-base
    if args.no_fixed_base:
        args.fixed_base = False
    # --no-depth-align 禁用深度校正文件
    if args.no_depth_align:
        PREFER_DEPTH_ALIGNED = False
        print("  使用原始 HaWoR 数据（无深度校正）")
    else:
        PREFER_DEPTH_ALIGNED = True
        rec_dir = Path(args.hawor_dir) / "reconstruction"
        aligned = list(rec_dir.glob("*_depth_aligned.npz"))
        if aligned:
            print(f"  使用深度校正后的 HaWoR 数据: {aligned[0].name}")
        else:
            print("  未找到 *_depth_aligned.npz，使用原始数据")
    if args.output is None:
        args.output = f"output/videos/hand_object_{args.mode}.mp4"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.hand_idx < -1:
        # -2 = 强制双手 (仅对 robot_only 的旧行为兼容)
        detected_hands = _detect_hands(Path(args.hawor_dir))
        if len(detected_hands) == 2:
            args.hand_idx = 0
            print(f"双手检测: 左手+右手, 使用左手数据 (idx=0) 驱动右臂")
        else:
            args.hand_idx = detected_hands[0] if detected_hands else 0
            print(f"数据中未检测到双手, 退回单手模式 (idx={args.hand_idx})")
    elif args.hand_idx == -1:
        # 自动检测
        detected_hands = _detect_hands(Path(args.hawor_dir))
        if len(detected_hands) == 2:
            args.hand_idx = 0
            print(f"自动检测: 双手, 使用左手数据 (idx=0) 驱动右臂")
        elif len(detected_hands) == 1:
            args.hand_idx = detected_hands[0]
            hand_label = "左手" if detected_hands[0] == 0 else "右手"
            print(f"自动检测: {hand_label} (idx={detected_hands[0]})")
        else:
            args.hand_idx = 0
            print(f"无法自动检测手, 默认使用左手 (idx=0)")
    else:
        hand_label = "左手" if args.hand_idx == 0 else "右手"
        print(f"指定手: {hand_label} (idx={args.hand_idx})")

    if not Path(args.transform_params).exists():
        raise FileNotFoundError(
            f"未找到变换参数文件: {args.transform_params}\n"
            f"请先运行: python 01_align_scene.py --ras_output ... --hawor_reconstruction ..."
        )

    logger = logging.getLogger("HandObjectRender")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

    renderer = HandObjectRenderer(hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                                  transform_params_path=args.transform_params,
                                  output=args.output, fps=args.fps, hand_idx=args.hand_idx, logger=logger,
                                  viewer=args.viewer, crf=args.crf,
                                  cam_width=args.width, cam_height=args.height,
                                  view=args.view, smooth=args.smooth, fixed_base=args.fixed_base)

    # robot_only 固定单臂模式, 沿用 hand_idx (默认右臂)
    if args.mode == "robot_only":
        renderer.hand_indices = [args.hand_idx]

    if args.mode == "hand_only":
        renderer.run_hand_only(start_frame=args.start_frame, num_frames=args.num_frames)
    elif args.mode == "robot_tracking":
        renderer.run_robot_tracking(start_frame=args.start_frame, num_frames=args.num_frames)
    elif args.mode == "robot_only":
        renderer.run_robot_only(start_frame=args.start_frame, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
