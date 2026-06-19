#!/usr/bin/env python3
"""
================================================================================
  03_track_robot.py — 独立机器人跟踪 (无GLB物体, 快速验证)

  管线:
    00_run_pipeline.py  ← 一键入口
    01_align_scene.py   →  02_render_scene.py  →  03_track_robot.py
    (对齐场景)              (渲染仿真场景)         (独立机器人跟踪)

  功能:
    仅使用 HaWoR 手部数据驱动 R1 机器人, 不需要 RAS 场景重建。
    适合快速验证手部→机器人映射是否正确, 支持多种观察视角。

  数据流:
    HaWoR 手部重建 (.npz)
        → MANOLayer (MANO 参数 → 21 关节点)
        → Dex Retargeting (手部关节 → 夹爪关节角)
        → Retargeting 内部 FK (获取 gripper 位置/朝向)
        → mapping_offset (映射到臂舒适工作空间)
        → RelaxedIK (臂逆运动学) → qpos
        → SAPIEN 渲染 → 视频

  与 02_render_scene.py --mode robot_only 的区别:
    - 不需要 RAS GLB 场景, 不需要 01_align_scene.py
    - 不需要第一人称相机轨迹 (使用固定第三人称视角)
    - 支持多种视角: behind (后上方), front (正前方), topdown (俯视)
    - 独立可运行, 适合快速迭代验证手部→机器人映射

  用法:
    python 03_track_robot.py --hawor-dir /home/an/data/hawor/7
    python 03_track_robot.py --hawor-dir /home/an/data/hawor/7 --hand-idx 0 --num-frames 50
    python 03_track_robot.py --hawor-dir /home/an/data/hawor/7 --view front

  输出:
    output/videos/hawor_r1_tracking.mp4   — 渲染视频
    output/tracking/hawor_r1_tracking.npy — 关节角序列
================================================================================
"""

import argparse
import sys
import logging
from pathlib import Path

import cv2
import numpy as np
import sapien
import torch
import joblib
from pytransform3d import rotations as pr

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import (
    HandType,
    RobotName,
    OPERATOR2MANO_RIGHT,
    RetargetingType,
    get_default_config_path,
)
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from hand_robot_viewer import RobotHandDatasetSAPIENViewer
from mano_layer import MANOLayer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"

LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
Q_180Z = np.array([0.0, 0.0, 0.0, 1.0])
SAFETY_DISTANCE = 0.075
WARMUP_FRAMES = 30
ARM_BASE_OFFSET_LOCAL = np.array([0.09193, -0.33649, 0.97171])
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.40
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.25])

R1_RIGHT_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798],
    [0.0, 3.1416],
    [-3.3161, 0.0],
    [-1.5708, 1.5708],
    [-1.5708, 1.5708],
    [-2.8798, 2.8798],
])

CAMERA_QUATS = {
    "behind": [0.0, 0.0, 1.0, 0.0],
    "front": [1.0, 0.0, 0.0, 0.0],
    "topdown": [0.7071, 0.0, 0.7071, 0.0],
}

R_GRIPPER_ALIGN = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0],
], dtype=np.float64)

R_HAWOR2SAPIEN = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
], dtype=np.float64)


class TrajectorySmoother:
    """轨迹平滑器: 双向低通滤波 + 迭代速度/加速度/加加速度限幅

    用于预计算管线的后处理, 减少关节运动的抖动和突变。
    不适用于实时渲染 (02_render_scene.py 不使用此类)。
    """

    SMOOTHNESS_THRESHOLDS = {
        "max_velocity": 3.0,
        "max_acceleration": 8.0,
        "max_jerk": 80.0,
        "si_improvement_min": 0.5,
    }

    def __init__(self, fps=30, max_velocity=1.5, max_acceleration=4.0,
                 max_jerk=20.0, lp_alpha=0.25, butterworth_order=2,
                 max_iterations=10, convergence_eps=1e-5):
        """初始化轨迹平滑器

        Args:
            fps: 帧率, 用于计算 dt
            max_velocity: 关节最大速度 (rad/s)
            max_acceleration: 关节最大加速度 (rad/s²)
            max_jerk: 关节最大加加速度 (rad/s³)
            lp_alpha: 低通滤波器截止参数 (0~1, 越小越平滑)
            butterworth_order: 低通滤波器阶数 (1 或 2)
            max_iterations: 迭代限幅最大次数
            convergence_eps: 收敛阈值
        """
        self.dt = 1.0 / fps
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.lp_alpha = lp_alpha
        self.butterworth_order = butterworth_order
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps

    def smooth_trajectory(self, qpos_sequence, smooth_indices):
        """对 qpos 序列执行平滑处理

        流程:
        1. 提取需要平滑的关节角 (smooth_indices)
        2. 填充无效帧 (线性插值)
        3. 双向低通滤波 (1阶或2阶)
        4. 迭代限幅 (速度→加速度→加加速度, 反复直到收敛)
        5. 计算平滑前后指标

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
        """用最近有效值填充无效帧 (前向填充 + 首帧修正)

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
        """双向低通滤波: 根据阶数选择1阶或2阶

        Args:
            trajectory: (N, J) 关节轨迹

        Returns:
            np.ndarray: (N, J) 平滑后的轨迹
        """
        if self.butterworth_order == 1:
            return self._bidirectional_lpf_order1(trajectory)
        return self._bidirectional_lpf_order2(trajectory)

    def _bidirectional_lpf_order1(self, trajectory):
        """1阶双向低通滤波 (指数移动平均)

        先正向 EMA, 再反向 EMA, 实现零相位延迟。

        Args:
            trajectory: (N, J) 关节轨迹

        Returns:
            np.ndarray: (N, J) 平滑后的轨迹
        """
        alpha = self.lp_alpha
        forward = np.zeros_like(trajectory)
        forward[0] = trajectory[0]
        for i in range(1, len(trajectory)):
            forward[i] = forward[i - 1] + alpha * (trajectory[i] - forward[i - 1])
        backward = np.zeros_like(trajectory)
        backward[-1] = forward[-1]
        for i in range(len(trajectory) - 2, -1, -1):
            backward[i] = backward[i + 1] + alpha * (forward[i] - backward[i + 1])
        return backward

    def _bidirectional_lpf_order2(self, trajectory):
        """2阶双向低通滤波 (级联两个1阶EMA)

        先正向2级EMA, 再反向2级EMA, 比单阶更平滑。

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
        for iteration in range(self.max_iterations):
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

        指标: 最大速度, 最大加速度, 最大加加速度, 平滑度指数 (SI)
        SI = ∫jerk² dt, 越小越平滑

        Args:
            trajectory_smooth: (N, J) 平滑后的轨迹
            trajectory_raw: (N, J) 原始轨迹

        Returns:
            dict: 包含 smooth/raw 指标和通过/未通过判定
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


def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据

    支持两种数据格式:
    1. world_space_res.pth (旧格式, 世界坐标)
    2. reconstruction/hawor_results_*.npz (新格式, 相机坐标, 需转换到世界坐标)

    Args:
        hawor_dir: HaWoR 输出目录路径
        hand_idx: 手部索引 (0=左手, 1=右手)

    Returns:
        dict: 包含 pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid
    """
    hawor_path = Path(hawor_dir)
    ws_file = hawor_path / "world_space_res.pth"
    rec_file = hawor_path / "reconstruction" / "hawor_results_0_113.npz"

    if ws_file.exists():
        ws = joblib.load(str(ws_file))
        pred_trans = ws[0].numpy() if hasattr(ws[0], 'numpy') else np.array(ws[0])
        pred_rot = ws[1].numpy() if hasattr(ws[1], 'numpy') else np.array(ws[1])
        pred_hand_pose = ws[2].numpy() if hasattr(ws[2], 'numpy') else np.array(ws[2])
        pred_betas = ws[3].numpy() if hasattr(ws[3], 'numpy') else np.array(ws[3])
        pred_valid = ws[4] if isinstance(ws[4], np.ndarray) else np.array(ws[4])
    elif rec_file.exists():
        rec = np.load(str(rec_file), allow_pickle=True)
        pred_trans = rec['pred_trans']
        pred_rot = rec['pred_rot']
        pred_hand_pose = rec['pred_hand_pose']
        pred_betas = rec['pred_betas']
        pred_valid = rec['pred_valid']
        R_c2w = rec['R_c2w']
        t_c2w = rec['t_c2w']
        for frame_i in range(pred_trans.shape[1]):
            for hand_i in range(pred_trans.shape[0]):
                if pred_valid[hand_i, frame_i]:
                    pred_trans[hand_i, frame_i] = R_c2w[frame_i] @ pred_trans[hand_i, frame_i] + t_c2w[frame_i]
                    rot_mat = pr.matrix_from_compact_axis_angle(pred_rot[hand_i, frame_i])
                    rot_mat_world = R_c2w[frame_i] @ rot_mat
                    pred_rot[hand_i, frame_i] = pr.compact_axis_angle_from_matrix(rot_mat_world)
    else:
        raise FileNotFoundError(f"未找到 hawor 数据文件: {ws_file} 或 {rec_file}")

    data = {
        "pred_trans": pred_trans[hand_idx],
        "pred_rot": pred_rot[hand_idx],
        "pred_hand_pose": pred_hand_pose[hand_idx],
        "pred_betas": pred_betas[hand_idx],
        "pred_valid": pred_valid[hand_idx],
    }
    return data


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


def transform_hawor_to_sapien(points, R_transform):
    """使用旋转矩阵将点从一个坐标系变换到另一个

    Args:
        points: (N, 3) 或 (3,) 输入点
        R_transform: (3, 3) 旋转矩阵 (如 R_HAWOR2SAPIEN)

    Returns:
        np.ndarray: 同形状, 变换后的点
    """
    return (R_transform @ points.T).T


class HaworR1Pipeline:
    """HaWoR 手部 → R1 机器人映射管线

    完整流程:
    1. 加载 HaWoR 手部数据
    2. 确定坐标系变换 (自动检测相机坐标系/Z-up坐标系)
    3. 初始化 Dex Retargeting (手部关节→夹爪)
    4. 分析手部轨迹 (质心、范围)
    5. 放置 R1 机器人 (基座位置)
    6. 计算工作空间映射 (mapping_offset + safety_offset)
    7. 初始化 RelaxedIK (夹爪→臂关节)
    8. 预计算所有帧 + 轨迹平滑
    9. 渲染视频 (可选)
    """

    def __init__(
        self,
        hawor_dir: str,
        hand_idx: int = 0,
        output_video: str = "hawor_r1_tracking.mp4",
        fps: int = 30,
        view: str = "behind",
        coord_transform: str = "auto",
        no_render: bool = False,
        logger: logging.Logger = None,
    ):
        """初始化管线

        Args:
            hawor_dir: HaWoR 输出目录路径
            hand_idx: 手部索引 (0=左手, 1=右手)
            output_video: 输出视频路径
            fps: 视频帧率
            view: 视角模式 (behind/front/topdown)
            coord_transform: 坐标变换模式 (auto/hawor2sapien/none)
            no_render: 是否跳过视频渲染 (仅输出qpos)
            logger: 日志记录器
        """
        self.hawor_dir = Path(hawor_dir)
        self.hand_idx = hand_idx
        self.output_video = output_video
        self.fps = fps
        self.view = view
        self.coord_transform = coord_transform
        self.no_render = no_render
        self._galaxea_sim = GALAXEA_SIM_PATH
        self.logger = logger or logging.getLogger("HaworR1")

    def run(self, start_frame: int = 0, num_frames: int = -1):
        """执行完整的 HaWoR→R1 映射管线

        8个步骤:
        1. 加载 HaWoR 数据
        2. 确定坐标变换
        3. 初始化 Dex Retargeting
        4. 分析手部轨迹
        5. 放置 R1 机器人
        6. 计算工作空间映射
        7. 初始化 RelaxedIK
        8. 预计算 + 平滑 + 渲染

        Args:
            start_frame: 起始帧索引
            num_frames: 处理帧数 (-1 表示全部)
        """
        self.logger.info("=" * 80)
        self.logger.info("Hawor 手部姿态 → R1 机器人映射执行")
        self.logger.info("=" * 80)

        self.logger.info("\n[1/8] 加载 hawor 数据 ...")
        hawor_data = load_hawor_data(self.hawor_dir, self.hand_idx)
        total_frames = hawor_data["pred_trans"].shape[0]
        if num_frames < 0 or num_frames > total_frames - start_frame:
            num_frames = total_frames - start_frame
        self.logger.info(f"  总帧数: {total_frames}, 处理帧: {start_frame}~{start_frame + num_frames}")
        self.logger.info(f"  有效帧: {hawor_data['pred_valid'][start_frame:start_frame+num_frames].sum()}/{num_frames}")

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_layer = MANOLayer("right", betas_mean)
        self.logger.info(f"  MANOLayer 初始化完成 (betas from frame {start_frame})")

        self.logger.info("\n[2/8] 确定坐标变换 ...")
        R_coord = self._determine_coord_transform(hawor_data, mano_layer, start_frame, num_frames)
        self.logger.info(f"  坐标变换矩阵:\n{R_coord}")

        self.logger.info("\n[3/8] 初始化 Dex Retargeting ...")
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        viewer = RobotHandDatasetSAPIENViewer(
            robot_names=[RobotName.r1_full],
            hand_type=HandType.right,
            headless=True,
        )
        viewer.mano_layer = mano_layer
        viewer.mano_face = mano_layer.f.cpu().numpy()
        viewer.camera_pose = sapien.Pose(np.zeros(3), [1, 0, 0, 0])
        retargeting = viewer.retargetings[0]
        retarget2sapien = viewer.retarget2sapien[0]
        self.logger.info("  ✓ Dex Retargeting 就绪")

        self.logger.info("\n[4/8] 分析手部轨迹 ...")
        wrist_positions, hand_stats = self._analyze_hand_trajectory(
            hawor_data, mano_layer, R_coord, start_frame, num_frames
        )
        self.logger.info(f"  有效帧: {len(wrist_positions)}")
        self.logger.info(f"  手腕质心(SAPIEN): [{hand_stats['centroid'][0]:.4f}, {hand_stats['centroid'][1]:.4f}, {hand_stats['centroid'][2]:.4f}]")
        self.logger.info(f"  手腕范围: X[{hand_stats['range'][0]:.4f}], Y[{hand_stats['range'][1]:.4f}], Z[{hand_stats['range'][2]:.4f}]")

        self.logger.info("\n[5/8] 放置 R1 机器人 ...")
        scene = viewer.scene
        r1_robot = viewer.robots[0]

        R_180Z_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        arm_base_offset_world = R_180Z_mat @ ARM_BASE_OFFSET_LOCAL
        centroid = hand_stats["centroid"]
        robot_root_y = centroid[1] - arm_base_offset_world[1] - 0.3
        robot_root_pos = np.array([0.0, robot_root_y, 0.0])
        r1_robot.set_root_pose(sapien.Pose(robot_root_pos.tolist(), Q_180Z.tolist()))
        self.logger.info(f"  机器人基座位置: [{robot_root_pos[0]:.4f}, {robot_root_pos[1]:.4f}, {robot_root_pos[2]:.4f}]")

        active_joints = r1_robot.get_active_joints()
        joint_names = [j.get_name() for j in active_joints]
        right_arm_indices = [i for i, name in enumerate(joint_names) if "right_arm" in name]
        gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
        gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

        right_ee_link = None
        for link in r1_robot.get_links():
            if "right_gripper_link" in link.get_name():
                right_ee_link = link
                break
        if right_ee_link is None:
            raise RuntimeError("无法找到 R1 右末端连杆 'right_gripper_link'")

        for joint in active_joints:
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        left_arm_indices = [i for i, name in enumerate(joint_names) if "left_arm" in name]
        initial_qpos = r1_robot.get_qpos().copy()
        left_arm_default = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
        for j, idx in enumerate(left_arm_indices):
            if j < len(left_arm_default):
                initial_qpos[idx] = left_arm_default[j]

        right_arm_starting = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
        for j, idx in enumerate(right_arm_indices):
            if j < len(right_arm_starting):
                initial_qpos[idx] = right_arm_starting[j]

        r1_robot.set_qpos(initial_qpos)
        scene.step()
        scene.update_render()

        right_arm_base_link = None
        for link in r1_robot.get_links():
            if "right_arm_base_link" in link.get_name():
                right_arm_base_link = link
                break
        if right_arm_base_link is None:
            raise RuntimeError("无法找到 R1 右臂基座连杆 'right_arm_base_link'")

        base_link_pose = right_arm_base_link.get_entity_pose()
        base_link_p = np.array(base_link_pose.p)
        base_link_q = np.array(base_link_pose.q)
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T
        self.logger.info(f"  right_arm_base_link 位置: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")

        self.logger.info("\n[6/8] 计算工作空间映射 ...")
        mapping_info = self._compute_workspace_mapping(hand_stats, base_link_p, base_link_R)
        self.logger.info(f"  映射偏移: [{mapping_info['mapping_offset'][0]:.4f}, {mapping_info['mapping_offset'][1]:.4f}, {mapping_info['mapping_offset'][2]:.4f}]")
        self.logger.info(f"  安全偏移: [{mapping_info['safety_offset'][0]:.4f}, {mapping_info['safety_offset'][1]:.4f}, {mapping_info['safety_offset'][2]:.4f}]")
        self.logger.info(f"  映射后质心到base距离: {mapping_info['mapped_dist_to_base']:.4f}m / 臂展{ARM_MAX_REACH:.3f}m")

        self.logger.info("\n[7/8] 初始化 RelaxedIK ...")
        ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=[0.001, 0.001, 0.001, 10.0, 10.0, 10.0],
        )
        ik_solver.relaxed_ik_right.reset([-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985])
        self.logger.info("  ✓ RelaxedIK 就绪")

        self.logger.info(f"\n[8/8] 预计算 {num_frames} 帧 (含 {WARMUP_FRAMES} 帧warmup) ...")
        qpos_sequence, ik_targets_world, eval_pre = self._precompute(
            hawor_data=hawor_data,
            start_frame=start_frame,
            num_frames=num_frames,
            mano_layer=mano_layer,
            R_coord=R_coord,
            retargeting=retargeting,
            retarget2sapien=retarget2sapien,
            ik_solver=ik_solver,
            r1_robot=r1_robot,
            right_arm_indices=right_arm_indices,
            base_link_p=base_link_p,
            base_link_R_inv=base_link_R_inv,
            mapping_info=mapping_info,
        )

        valid = sum(1 for x in qpos_sequence if x is not None)
        self.logger.info(f"  ✓ 预计算完成: {valid}/{num_frames + WARMUP_FRAMES} 帧有效")

        smooth_indices = list(right_arm_indices) + [gripper_idx1, gripper_idx2]
        smoother = TrajectorySmoother(
            fps=self.fps, max_velocity=1.5, max_acceleration=4.0,
            max_jerk=20.0, lp_alpha=0.25, butterworth_order=2,
            max_iterations=10, convergence_eps=1e-5,
        )
        self.logger.info("\n  轨迹后处理平滑 ...")
        warmup_qpos = qpos_sequence[:WARMUP_FRAMES]
        data_qpos = qpos_sequence[WARMUP_FRAMES:]
        if data_qpos:
            data_smoothed, smooth_metrics = smoother.smooth_trajectory(data_qpos, smooth_indices)
            qpos_sequence = warmup_qpos + data_smoothed
            self.logger.info(f"  ✓ 平滑完成: 速度峰值 {smooth_metrics['smooth_max_velocity']:.2f} rad/s, "
                             f"加速度峰值 {smooth_metrics['smooth_max_acceleration']:.2f} rad/s²")
        else:
            qpos_sequence = warmup_qpos
            smooth_metrics = {}

        qpos_save_path = str(Path(self.output_video).with_suffix(".npy")).replace("videos", "tracking")
        os.makedirs(os.path.dirname(qpos_save_path), exist_ok=True)
        valid_qpos = [q for q in qpos_sequence if q is not None]
        if valid_qpos:
            np.save(qpos_save_path, np.array(valid_qpos))
            self.logger.info(f"  qpos 序列已保存: {qpos_save_path} (shape: {np.array(valid_qpos).shape})")

        if self.no_render:
            self.logger.info("\n  --no-render 模式: 跳过视频渲染")
            eval_render = {"fk_errors": [], "fk_details": [], "comfort_scores": []}
        else:
            eval_render = self._render_video(
                scene, r1_robot, qpos_sequence, ik_targets_world,
                viewer, start_frame, right_ee_link,
                base_link_p, base_link_R_inv, mapping_info, robot_root_pos,
            )

        self.logger.info("\n" + "=" * 80)
        self.logger.info("管线执行完成！")
        if not self.no_render:
            self.logger.info(f"视频: {self.output_video}")
        self.logger.info(f"qpos: {qpos_save_path}")
        self.logger.info("=" * 80)

    def _determine_coord_transform(self, hawor_data, mano_layer, start_frame, num_frames):
        """自动检测坐标系类型并返回变换矩阵

        检测策略:
        - 如果手腕Z坐标绝对值远大于X/Y → 相机坐标系 (Z=深度), 需要变换
        - 如果手腕Z运动范围远大于X/Y → 相机坐标系, 需要变换
        - 否则 → Z-up 坐标系, 不需要变换

        Args:
            hawor_data: load_hawor_data() 返回的字典
            mano_layer: MANOLayer 实例
            start_frame: 起始帧索引
            num_frames: 帧数

        Returns:
            np.ndarray: (3, 3) 坐标变换矩阵 (R_HAWOR2SAPIEN 或 np.eye(3))
        """
        if self.coord_transform == "none":
            return np.eye(3)
        elif self.coord_transform == "hawor2sapien":
            return R_HAWOR2SAPIEN

        sample_indices = range(start_frame, min(start_frame + num_frames, start_frame + 10))
        joints_list = []
        trans_list = []
        for i in sample_indices:
            if not hawor_data["pred_valid"][i]:
                continue
            _, j = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][i],
                hawor_data["pred_hand_pose"][i],
                hawor_data["pred_trans"][i],
            )
            joints_list.append(j)
            trans_list.append(hawor_data["pred_trans"][i])

        if not joints_list:
            return R_HAWOR2SAPIEN

        all_joints = np.vstack(joints_list)
        all_trans = np.array(trans_list)

        z_range = all_joints[:, 2].max() - all_joints[:, 2].min()
        y_range = all_joints[:, 1].max() - all_joints[:, 1].min()
        x_range = all_joints[:, 0].max() - all_joints[:, 0].min()

        mean_trans = np.mean(all_trans, axis=0)
        abs_z = abs(mean_trans[2])
        abs_y = abs(mean_trans[1])
        abs_x = abs(mean_trans[0])

        self.logger.info(f"  hawor 原始坐标范围: X={x_range:.4f}, Y={y_range:.4f}, Z={z_range:.4f}")
        self.logger.info(f"  hawor 手腕质心绝对值: |X|={abs_x:.4f}, |Y|={abs_y:.4f}, |Z|={abs_z:.4f}")

        if abs_z > max(abs_x, abs_y) * 1.5 and abs_z > 0.1:
            self.logger.info("  检测到相机坐标系 (Z=深度, 绝对值大), 应用 hawor→SAPIEN 变换")
            return R_HAWOR2SAPIEN
        elif z_range > max(x_range, y_range) * 1.5:
            self.logger.info("  检测到相机坐标系 (Z运动范围大), 应用 hawor→SAPIEN 变换")
            return R_HAWOR2SAPIEN
        else:
            self.logger.info("  检测到Z朝上坐标系, 无需坐标变换")
            return np.eye(3)

    def _analyze_hand_trajectory(self, hawor_data, mano_layer, R_coord, start_frame, num_frames):
        """分析手部轨迹, 计算手腕位置统计量

        遍历所有有效帧, 计算 MANO FK 得到手腕位置,
        然后统计质心、范围、标准差等。

        Args:
            hawor_data: load_hawor_data() 返回的字典
            mano_layer: MANOLayer 实例
            R_coord: 坐标变换矩阵
            start_frame: 起始帧索引
            num_frames: 帧数

        Returns:
            tuple: (wrist_positions, hand_stats)
                - wrist_positions: list of (3,) 手腕位置
                - hand_stats: dict 包含 centroid, range, min, max, std, num_valid
        """
        wrist_positions = []
        for i in range(num_frames):
            global_idx = start_frame + i
            if not hawor_data["pred_valid"][global_idx]:
                continue
            _, j = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][global_idx],
                hawor_data["pred_hand_pose"][global_idx],
                hawor_data["pred_trans"][global_idx],
            )
            wrist_pos = R_coord @ j[0, :3]
            wrist_positions.append(wrist_pos.copy())

        if not wrist_positions:
            raise RuntimeError("无法从 hawor 数据中提取有效手腕位置")

        positions = np.array(wrist_positions)
        return wrist_positions, {
            "centroid": np.mean(positions, axis=0),
            "range": np.ptp(positions, axis=0),
            "min": np.min(positions, axis=0),
            "max": np.max(positions, axis=0),
            "std": np.std(positions, axis=0),
            "num_valid": len(wrist_positions),
        }

    def _compute_workspace_mapping(self, hand_stats, base_link_p, base_link_R):
        """计算工作空间映射偏移量

        将手腕轨迹质心映射到臂舒适工作空间:
        1. 计算舒适目标点 (base_link_R @ COMFORT_TARGET_IN_BASE + base_link_p)
        2. mapping_offset = 舒适目标点 - 手腕质心 (把手腕拉到舒适区域)
        3. safety_offset = 沿 base→gripper 方向偏移 SAFETY_DISTANCE (避免重叠)

        Args:
            hand_stats: _analyze_hand_trajectory() 返回的统计量
            base_link_p: (3,) arm_base_link 位置
            base_link_R: (3, 3) arm_base_link 旋转矩阵

        Returns:
            dict: 包含 mapping_offset, safety_offset, comfort_target 等
        """
        centroid = hand_stats["centroid"]
        comfort_target_world = base_link_R @ COMFORT_TARGET_IN_BASE + base_link_p
        mapping_offset = comfort_target_world - centroid
        approach_dir = base_link_p - comfort_target_world
        approach_dir = approach_dir / np.linalg.norm(approach_dir)
        safety_offset = approach_dir * SAFETY_DISTANCE
        mapped_centroid = centroid + mapping_offset + safety_offset
        mapped_in_base = base_link_R.T @ (mapped_centroid - base_link_p)
        mapped_dist = np.linalg.norm(mapped_in_base)
        return {
            "mapping_offset": mapping_offset,
            "safety_offset": safety_offset,
            "comfort_target_base": COMFORT_TARGET_IN_BASE.copy(),
            "comfort_target_world": comfort_target_world,
            "mapped_centroid": mapped_centroid,
            "mapped_dist_to_base": mapped_dist,
            "approach_dir": approach_dir,
        }

    def _precompute(
        self,
        hawor_data,
        start_frame,
        num_frames,
        mano_layer,
        R_coord,
        retargeting,
        retarget2sapien,
        ik_solver,
        r1_robot,
        right_arm_indices,
        base_link_p,
        base_link_R_inv,
        mapping_info,
    ):
        """预计算所有帧的 qpos 序列

        流程:
        1. 找到第一个有效帧, 求解 IK 作为起始点
        2. IK 预热 (对第一个目标迭代200次)
        3. Warmup 阶段: smoothstep 插值从当前关节角过渡到第一帧
        4. 逐帧: MANO FK → Retargeting → IK → 低通滤波 → qpos

        Args:
            hawor_data: HaWoR 数据
            start_frame: 起始帧
            num_frames: 帧数
            mano_layer: MANOLayer
            R_coord: 坐标变换矩阵
            retargeting: DexRetargeting 实例
            retarget2sapien: retargeting→sapien 关节映射
            ik_solver: RelaxedIK 求解器
            r1_robot: SAPIEN R1 机器人实例
            right_arm_indices: 右臂关节索引列表
            base_link_p: arm_base_link 位置
            base_link_R_inv: arm_base_link 逆旋转矩阵
            mapping_info: _compute_workspace_mapping() 的输出

        Returns:
            tuple: (qpos_sequence, ik_targets_world, eval_data)
        """
        qpos_sequence = []
        ik_targets_world = []
        eval_data = {"ik_errors": [], "joint_values": [], "out_of_reach": 0}

        joint_names = [j.get_name() for j in r1_robot.get_active_joints()]
        gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
        gripper_idx2 = joint_names.index("right_gripper_finger_joint2")

        sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(retarget2sapien):
            sapien2retarget[retarget_i] = sapien_i
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        init_sapien_qpos = r1_robot.get_qpos().copy()
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in sapien2retarget:
                fixed_qpos[i] = init_sapien_qpos[sapien2retarget[retarget_idx]]

        ref_indices = retargeting.optimizer.target_link_human_indices
        mapping_offset = mapping_info["mapping_offset"]
        safety_offset = mapping_info["safety_offset"]

        first_valid_frame = None
        first_ik_joints = None
        first_gripper1 = 0.0
        first_gripper2 = 0.0
        first_ik_target_world = None
        first_ik_target_base = None
        first_ee_quat_base = None

        for probe_idx in range(num_frames):
            global_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][global_idx]:
                continue

            vertex, joints = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][global_idx],
                hawor_data["pred_hand_pose"][global_idx],
                hawor_data["pred_trans"][global_idx],
            )

            joints_sapien = (R_coord @ joints.T).T
            wrist_pos_world = joints_sapien[0, :3].copy()

            wrist_axis_angle = hawor_data["pred_rot"][global_idx].astype(np.float64)
            wrist_rot_mat_hawor = pr.matrix_from_compact_axis_angle(wrist_axis_angle)
            wrist_rot_mat_sapien = R_coord @ wrist_rot_mat_hawor
            wrist_quat_sapien = pr.quaternion_from_matrix(wrist_rot_mat_sapien)
            R_mano2world = wrist_rot_mat_sapien @ OPERATOR2MANO_RIGHT.T

            ref_value = joints_sapien[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            sapien_qpos = retarget_qpos[retarget2sapien]
            first_gripper1 = float(sapien_qpos[gripper_idx1])
            first_gripper2 = float(sapien_qpos[gripper_idx2])

            ik_target_world_raw = wrist_pos_world + mapping_offset + safety_offset
            first_ik_target_world = ik_target_world_raw.copy()

            ik_target_base = base_link_R_inv @ (ik_target_world_raw - base_link_p)
            R_ee_base = base_link_R_inv @ R_mano2world @ R_GRIPPER_ALIGN
            ee_quat_base = pr.quaternion_from_matrix(R_ee_base)
            try:
                first_ik_joints = np.array(
                    ik_solver.solve_position_right(
                        ik_target_base.tolist(), ee_quat_base.tolist()
                    )
                )
                first_ik_target_base = ik_target_base.copy()
                first_ee_quat_base = ee_quat_base.copy()
                first_valid_frame = probe_idx
                break
            except Exception:
                continue

        if first_ik_joints is None:
            raise RuntimeError("无法求解任何有效帧的IK, 请检查数据和工作空间映射")

        ik_warmup_iters = 200
        self.logger.info(f"  IK预热：对第一个目标点迭代 {ik_warmup_iters} 次 ...")
        for i in range(ik_warmup_iters):
            first_ik_joints = np.array(
                ik_solver.solve_position_right(
                    first_ik_target_base.tolist(), first_ee_quat_base.tolist()
                )
            )
        self.logger.info(f"  IK预热完成，关节角: {first_ik_joints}")

        current_right_joints = np.array([init_sapien_qpos[i] for i in right_arm_indices])
        self.logger.info(f"  Warmup: 从当前关节角过渡到第一帧IK结果 ({WARMUP_FRAMES}帧)")

        ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
        ee_pos_filter.next(first_ik_target_world)
        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        joint_filter.next(current_right_joints)

        for w in range(WARMUP_FRAMES):
            t = (w + 1) / WARMUP_FRAMES
            t_smooth = t * t * (3 - 2 * t)
            interp_joints = current_right_joints * (1 - t_smooth) + first_ik_joints * t_smooth
            interp_joints = joint_filter.next(interp_joints)
            r1_qpos = r1_robot.get_qpos().copy()
            for j, idx in enumerate(right_arm_indices):
                r1_qpos[idx] = interp_joints[j]
            r1_qpos[gripper_idx1] = 0.04
            r1_qpos[gripper_idx2] = -0.04
            qpos_sequence.append(r1_qpos)
            ik_targets_world.append(first_ik_target_world.copy())

        for local_idx in range(num_frames):
            global_idx = start_frame + local_idx
            if not hawor_data["pred_valid"][global_idx]:
                qpos_sequence.append(None)
                ik_targets_world.append(None)
                continue

            vertex, joints = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][global_idx],
                hawor_data["pred_hand_pose"][global_idx],
                hawor_data["pred_trans"][global_idx],
            )

            joints_sapien = (R_coord @ joints.T).T
            wrist_pos_world = joints_sapien[0, :3].copy()

            wrist_axis_angle = hawor_data["pred_rot"][global_idx].astype(np.float64)
            wrist_rot_mat_hawor = pr.matrix_from_compact_axis_angle(wrist_axis_angle)
            wrist_rot_mat_sapien = R_coord @ wrist_rot_mat_hawor
            R_mano2world = wrist_rot_mat_sapien @ OPERATOR2MANO_RIGHT.T

            ref_value = joints_sapien[ref_indices, :].astype(np.float32)
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            sapien_qpos = retarget_qpos[retarget2sapien]
            gripper1 = float(sapien_qpos[gripper_idx1])
            gripper2 = float(sapien_qpos[gripper_idx2])

            ik_target_world_raw = wrist_pos_world + mapping_offset + safety_offset
            ik_target_world = ee_pos_filter.next(ik_target_world_raw)
            ik_targets_world.append(ik_target_world.copy())

            ik_target_base = base_link_R_inv @ (ik_target_world - base_link_p)
            dist_to_base = np.linalg.norm(ik_target_base)

            if dist_to_base > ARM_MAX_REACH:
                eval_data["out_of_reach"] += 1

            R_ee_base = base_link_R_inv @ R_mano2world @ R_GRIPPER_ALIGN
            ee_quat_base = pr.quaternion_from_matrix(R_ee_base)
            try:
                right_joints = np.array(
                    ik_solver.solve_position_right(
                        ik_target_base.tolist(), ee_quat_base.tolist()
                    )
                )
            except Exception as exc:
                self.logger.warning(f"  帧 {global_idx}: IK 失败 - {exc}")
                qpos_sequence.append(None)
                eval_data["ik_errors"].append(str(exc))
                continue

            right_joints = joint_filter.next(right_joints)
            eval_data["joint_values"].append(right_joints.copy())

            r1_qpos = r1_robot.get_qpos().copy()
            if len(right_joints) == len(right_arm_indices):
                for j, idx in enumerate(right_arm_indices):
                    r1_qpos[idx] = right_joints[j]
            r1_qpos[gripper_idx1] = gripper1
            r1_qpos[gripper_idx2] = gripper2
            qpos_sequence.append(r1_qpos)

            if (local_idx + 1) % 10 == 0:
                self.logger.info(f"  已计算 {local_idx + 1}/{num_frames} 帧 ...")

        return qpos_sequence, ik_targets_world, eval_data

    def _render_video(
        self, scene, r1_robot, qpos_sequence, ik_targets_world,
        viewer, start_frame, right_ee_link,
        base_link_p, base_link_R_inv, mapping_info, robot_root_pos,
    ):
        """使用预计算的 qpos 序列渲染视频

        每帧: set_qpos → scene.step() → 渲染 → 叠加信息文字 → 写入视频
        叠加信息: 帧号、EE-IK 误差 (cm)

        Args:
            scene: SAPIEN 场景
            r1_robot: R1 机器人实例
            qpos_sequence: 预计算的 qpos 列表
            ik_targets_world: IK 目标位置列表
            viewer: SAPIEN Viewer
            start_frame: 起始帧
            right_ee_link: 右末端连杆
            base_link_p: arm_base_link 位置
            base_link_R_inv: arm_base_link 逆旋转矩阵
            mapping_info: 工作空间映射信息
            robot_root_pos: 机器人根位置

        Returns:
            dict: 渲染评估数据
        """
        self.logger.info(f"\n渲染视频 → {self.output_video}  [视角: {self.view}]")

        active_joints = r1_robot.get_active_joints()
        joint_names = [j.get_name() for j in active_joints]
        robot_root = np.array(r1_robot.get_root_pose().p)

        if self.view == "behind":
            camera_pos = robot_root + np.array([2.5, 0.0, 1.2])
        elif self.view == "front":
            camera_pos = robot_root + np.array([-2.5, 0.0, 1.2])
        elif self.view == "topdown":
            camera_pos = robot_root + np.array([0.0, 0.0, 4.0])
        else:
            camera_pos = robot_root + np.array([2.5, 0.0, 1.2])
        camera_quat = CAMERA_QUATS.get(self.view, CAMERA_QUATS["behind"])

        camera = scene.add_camera(
            name="main", width=1920, height=1080,
            fovy=np.deg2rad(60), near=0.01, far=200.0,
        )
        camera.set_local_pose(sapien.Pose(camera_pos.tolist(), camera_quat))

        render_fps = self.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            self.output_video, fourcc, render_fps,
            (camera.get_width(), camera.get_height()),
        )

        ee_trajectory = []
        ik_target_trajectory = []

        for frame_idx, qpos in enumerate(qpos_sequence):
            is_warmup = frame_idx < WARMUP_FRAMES
            data_frame_idx = frame_idx - WARMUP_FRAMES

            if qpos is not None:
                r1_robot.set_qpos(qpos)
                for joint in active_joints:
                    joint.set_drive_target(qpos[joint_names.index(joint.get_name())])

            scene.step()
            scene.update_render()

            if not is_warmup and qpos is not None:
                ee_pose = right_ee_link.get_entity_pose()
                ee_pos = np.array(ee_pose.p)
                ee_trajectory.append(ee_pos.copy())

                if ik_targets_world[frame_idx] is not None:
                    ik_target_trajectory.append(ik_targets_world[frame_idx].copy())

            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            bgr = np.ascontiguousarray(rgb[..., ::-1])

            h, w = bgr.shape[:2]
            if is_warmup:
                t = (frame_idx + 1) / WARMUP_FRAMES
                cv2.rectangle(bgr, (0, 0), (w, 50), (0, 0, 0), -1)
                cv2.putText(bgr, f"Warmup {frame_idx+1}/{WARMUP_FRAMES} ({t*100:.0f}%)",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            else:
                cv2.rectangle(bgr, (0, 0), (w, 50), (0, 0, 0), -1)
                cv2.putText(bgr, f"Frame {data_frame_idx+1}",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if ee_trajectory and ik_target_trajectory:
                    err_cm = np.linalg.norm(ee_trajectory[-1] - ik_target_trajectory[-1]) * 100
                    err_color = (0, 255, 0) if err_cm < 2 else (0, 255, 255) if err_cm < 5 else (0, 0, 255)
                    cv2.putText(bgr, f"EE-IK: {err_cm:.1f}cm",
                                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, err_color, 2)

            writer.write(bgr)

            if (frame_idx + 1) % 10 == 0:
                self.logger.info(f"  已渲染 {frame_idx + 1}/{len(qpos_sequence)} 帧 ...")

        writer.release()
        self.logger.info(f"✓ 视频已保存: {self.output_video}")

        return {"fk_errors": [], "fk_details": [], "comfort_scores": []}


def _setup_logger(output_video: str) -> logging.Logger:
    """配置日志记录器: 同时输出到控制台和日志文件

    Args:
        output_video: 输出视频路径 (用于生成日志文件名)

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger("HaworR1")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    video_name = Path(output_video).stem
    log_file = Path.cwd() / f"{video_name}.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main():
    """命令行入口: HaWoR 手部姿态 → R1 机器人映射

    用法示例:
      python 03_track_robot.py --hawor-dir /path/to/hawor
      python 03_track_robot.py --hawor-dir /path/to/hawor --view front --num-frames 100
      python 03_track_robot.py --hawor-dir /path/to/hawor --no-render  # 仅输出qpos
    """
    parser = argparse.ArgumentParser(
        description="Hawor 手部姿态 → R1 机器人映射执行"
    )
    parser.add_argument("--hawor-dir", type=str, required=True,
                        help="hawor 数据目录 (包含 world_space_res.pth)")
    parser.add_argument("--hand-idx", type=int, default=0,
                        help="手部索引 (0=右手, 1=左手, 默认0)")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="帧数 (-1=全部)")
    parser.add_argument("--output-video", type=str, default="output/videos/hawor_r1_tracking.mp4",
                        help="输出视频路径")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--view", type=str, default="behind",
                        choices=["behind", "front", "topdown"], help="视角模式")
    parser.add_argument("--coord-transform", type=str, default="auto",
                        choices=["auto", "hawor2sapien", "none"],
                        help="坐标变换: auto=自动检测, hawor2sapien=相机系→Z上, none=不变换")
    parser.add_argument("--no-render", action="store_true",
                        help="跳过视频渲染, 仅输出qpos序列 (无需GPU)")

    args = parser.parse_args()
    logger = _setup_logger(args.output_video)
    logger.info(f"日志文件: {Path.cwd() / f'{Path(args.output_video).stem}.log'}")

    pipeline = HaworR1Pipeline(
        hawor_dir=args.hawor_dir,
        hand_idx=args.hand_idx,
        output_video=args.output_video,
        fps=args.fps,
        view=args.view,
        coord_transform=args.coord_transform,
        no_render=args.no_render,
        logger=logger,
    )
    pipeline.run(start_frame=args.start_frame, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
