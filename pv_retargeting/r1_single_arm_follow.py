#!/usr/bin/env python3
"""
================================================================================
  R1 单臂手部跟随管线 (DexYCB)
  只加载单臂 (6关节+夹爪)，RelaxedIK 求解臂关节，Dex Retargeting 求解夹爪
================================================================================

管线总览
────────────────────────────────────────────────────────────────────────────────

  DexYCB数据集        MANO FK          Dex Retargeting         RelaxedIK
  ┌──────────┐    ┌───────────┐    ┌─────────────────────┐   ┌───────────┐
  │ hand_pose │──→│ 21个3D关节 │──→│ 优化器求解:          │   │ 输入:      │
  │ hand_shape│    │ +手腕朝向  │    │  ·6自由关节(底盘位姿)│   │  ·位置(手腕)│
  └──────────┘    └───────────┘    │  ·2夹爪关节(开合)    │   │  ·朝向(底盘)│
                                   └─────────────────────┘   │ 输出:      │
                                            │                │  6个臂关节角│
                                            │                └───────────┘
                                    ┌────────┴────────┐
                                    │                 │
                              夹爪开合值          底盘旋转→IK朝向目标
                           (gripper_joint1/2)   (Euler角→旋转矩阵)

核心映射逻辑
────────────────────────────────────────────────────────────────────────────────
  1. 位置映射: MANO手腕位置 → +mapping_offset → +safety_offset → IK目标位置
     - mapping_offset: 将手腕质心映射到臂舒适工作空间中心
     - safety_offset: 避免夹爪与人手重叠

  2. 朝向映射: Dex Retargeting的自由关节Euler角 → 底盘旋转 → IK目标朝向
     - retargeting优化器通过匹配人手指尖与R1夹爪指尖的3D位置来求解
     - 6个自由关节控制R1底盘的6DOF位姿(3平移+3旋转)
     - 底盘旋转 = Euler(rx, ry, rz)，编码了期望的夹爪朝向
     - IK目标朝向 = base_link_R_inv @ chassis_R
       (将底盘旋转从世界坐标系转换到臂基座坐标系)

  3. 夹爪映射: Dex Retargeting直接输出夹爪关节值
     - gripper_finger_joint1/2 控制夹爪开合
     - 参考关节: 人手食指尖(4)和中指尖(8)

  关键约束: retargeting的臂关节被固定为0，因此gripper相对于arm_base的
  朝向是恒定的(=单位矩阵)。IK朝向目标的有效信息全部来自底盘旋转。

阶段详解
────────────────────────────────────────────────────────────────────────────────

阶段 1: 数据加载 + 自动检测左右手
  - 读取 DexYCB 数据集 meta.yml 中的 mano_sides 字段
  - hand_pose: 前3维=手腕compact axis-angle, 中45维=手指PCA, 后3维=平移
  - hand_shape: 10维MANO形状参数
  - object_pose: 每帧每个YCB物体的7维位姿(3平移+4四元数)
  - extrinsics: 4x4相机外参矩阵

阶段 2: SAPIEN场景 + 机器人加载 + Dex Retargeting初始化
  - 创建SAPIEN场景, 设置光照和相机(与visualize_hand_object.py一致)
  - 加载R1机器人: 从RetargetingConfig获取URDF路径, 用yourdfpy加载_glb版本
    (带.obj/.mtl材质, 纹理正确显示), 写入临时文件后用SAPIEN加载
  - 创建SeqRetargeting (NLopt SLSQP位置优化器)
  - warm_start: 用MANO手腕四元数初始化自由关节(解析计算初始解)
  - retarget2sapien映射: retargeting的26个关节名 → SAPIEN的8个关节名

阶段 3: 手部轨迹分析
  - 遍历所有帧, 通过MANOLayer计算每帧的21个3D关节点
  - 提取手腕关节(joint[0])位置, 计算质心/范围/标准差
  - 用于后续机器人放置和工作空间映射

阶段 4: 单臂放置
  - 臂基座放在手腕质心上方COMFORTABLE_REACH处
  - 绕Z轴旋转180°(使臂朝向操作者)
  - 获取arm_base_link实际世界位姿(通过SAPIEN FK)

阶段 5: 工作空间映射
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 问题: R1臂基座高Z≈0.97m, 臂展0.71m, 最低可达Z≈0.26m              │
  │       DexYCB手腕高Z≈0.08m, 低于臂最低可达范围                      │
  │ 解决: 工作空间映射 — 将手部轨迹平移到臂可达空间内                   │
  │                                                                 │
  │ 1. 舒适目标(base帧): [0.30, 0.0, -0.30] (前方30cm, 下方30cm)     │
  │ 2. 舒适目标(世界帧): base_link_R @ COMFORT_TARGET + base_link_p   │
  │ 3. 映射偏移 = 舒适目标(世界帧) - 手腕质心                         │
  │ 4. 安全偏移 = normalize(base_link_p - 舒适目标) × SAFETY_DISTANCE │
  │ 5. 每帧: ik_target = wrist_pos + mapping_offset + safety_offset  │
  └─────────────────────────────────────────────────────────────────────┘

阶段 6: RelaxedIK 初始化
  - 6DOF臂IK求解器, 位置容差0.5mm, 朝向容差0.05rad(≈2.9°)
  - base_links=[arm_base_link], IK输入/输出都在base_link坐标系

阶段 7: 预计算 (含warmup过渡)
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 每帧处理流程:                                                       │
  │                                                                 │
  │ ① MANO正运动学: hand_pose → MANOLayer → vertex[778,3], joints[21,3]│
  │                                                                 │
  │ ② Dex Retargeting (手部→夹爪映射):                                │
  │    ref_value = joints[ref_indices]  (食指尖/中指尖等参考点)         │
  │    retarget_qpos = SeqRetargeting.retarget(ref_value, fixed_qpos) │
  │    → NLopt SLSQP 最小化位置误差 → 26维完整关节向量                 │
  │    → [0:3]=底盘平移, [3:6]=底盘Euler角, [6:24]=臂关节(固定为0),    │
  │      [24:26]=夹爪关节                                              │
  │    → sapien_qpos = retarget_qpos[retarget2sapien]  (8维)          │
  │    → gripper1, gripper2 = sapien_qpos[gripper_idx1/2]            │
  │                                                                 │
  │ ③ 夹爪朝向 (从retargeting内部FK获取):                              │
  │    internal_robot.compute_forward_kinematics(retarget_qpos)       │
  │    gripper_R = internal_robot.get_link_pose(gripper_link_id)[:3,:3]│
  │    → gripper_R = chassis_R (因为arm joints=0, gripper相对base恒定) │
  │                                                                 │
  │ ④ 工作空间映射 (IK目标计算):                                       │
  │    ik_target_world = wrist_pos + mapping_offset + safety_offset  │
  │    ik_target_world = LPFilter(α=0.6).next(ik_target_world)      │
  │    ik_target_base = base_link_R⁻¹ @ (ik_target_world - base_p)  │
  │    ee_R_base = base_link_R⁻¹ @ gripper_R                         │
  │    ee_quat_base = quat_from_matrix(ee_R_base)  # [w,x,y,z]       │
  │                                                                 │
  │ ⑤ RelaxedIK (臂IK求解):                                           │
  │    arm_joints = ik_solver.solve_position_right(                   │
  │        ik_target_base.tolist(),   # base_link帧坐标               │
  │        ee_quat_base.tolist()      # 末端朝向(wxyz)                │
  │    )                                                            │
  │    arm_joints = LPFilter(α=0.5).next(arm_joints)  # 关节平滑    │
  │                                                                 │
  │ ⑥ 组装qpos:                                                      │
  │    qpos[arm_joint_indices] = arm_joints                          │
  │    qpos[gripper_idx1] = gripper1                                 │
  │    qpos[gripper_idx2] = gripper2                                 │
  └─────────────────────────────────────────────────────────────────────┘

  Warmup过渡: smoothstep插值从初始姿态过渡到第一个IK目标, 避免跳变

阶段 8: 轨迹后处理平滑 (TrajectorySmoother)
  - 双向二阶Butterworth低通滤波 + 速度/加速度/加加速度迭代限幅

阶段 9: 渲染视频 (SAPIEN Viewer, 与visualize_hand_object.py一致)
  - 人手mesh + YCB物体(两份: 原始+偏移) + R1单臂 + 轨迹/坐标轴可视化
  - 不调用scene.step(), 仅scene.update_render() + viewer.render()

阶段 10: 评估报告

用法
────────────────────────────────────────────────────────────────────────────────
  python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb
  python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb --data-id 4
================================================================================
"""

import argparse
import logging
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import sapien
import torch
import yaml
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from tqdm import trange

from dataset import DexYCBVideoDataset, YCB_CLASSES
from dex_retargeting.constants import (
    HandType,
    RobotName,
    RetargetingType,
    get_default_config_path,
)
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting
from mano_layer import MANOLayer

np.bool = bool
np.int = int
np.float = float
np.str = str
np.complex = complex
np.object = object
try:
    np.unicode = np.unicode_
except AttributeError:
    np.unicode = np.str_

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
FLOATING_RIGHT_URDF = (
    GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_right.urdf"
)
FLOATING_LEFT_URDF = (
    GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_left.urdf"
)
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(GALAXEA_SIM_PATH))
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

LP_ALPHA_EE = 0.6
LP_ALPHA_JOINT = 0.5
WARMUP_FRAMES = 30
ARM_MAX_REACH = 0.713
COMFORTABLE_REACH = 0.30
COMFORT_TARGET_IN_BASE = np.array([0.30, 0.0, -0.30])
SAFETY_DISTANCE = 0.05

RIGHT_ARM_STARTING = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]
LEFT_ARM_STARTING = [1.5, 1.9508, 1.0809, -0.4438, -0.1709, 0.1985]

R1_JOINT_LIMITS = np.array([
    [-2.8798, 2.8798],
    [0.0, 3.2289],
    [-3.3161, 0.0],
    [-2.8798, 2.8798],
    [-1.6581, 1.6581],
    [-2.8798, 2.8798],
])


def _detect_hand_type_from_dataset(data_root: Path, data_id: int) -> str:
    """
    从 DexYCB 数据集的 meta.yml 自动检测手部类型

    DexYCB 每个 capture 的 meta.yml 包含 mano_sides 字段，
    记录该 capture 使用的是左手还是右手。
    本函数遍历数据集的 capture 目录，读取 meta.yml，
    返回指定 data_id 对应的 capture 的 mano_sides。

    Args:
        data_root: DexYCB 数据集根目录
        data_id: 数据索引

    Returns:
        "right" 或 "left"
    """
    _SUBJECTS = [
        "20200709-subject-01", "20200813-subject-02", "20200820-subject-03",
        "20200903-subject-04", "20200908-subject-05", "20200918-subject-06",
        "20200928-subject-07", "20201002-subject-08", "20201015-subject-09",
        "20201022-subject-10",
    ]
    captures = []
    for subject_dir in sorted(data_root.iterdir()):
        if subject_dir.stem not in _SUBJECTS:
            continue
        for capture_dir in sorted(subject_dir.iterdir()):
            meta_file = capture_dir / "meta.yml"
            if not meta_file.exists():
                continue
            captures.append(capture_dir)

    if data_id >= len(captures):
        print(f"  ⚠ data_id={data_id} 超出范围 (共{len(captures)}个capture)，默认使用右手")
        return "right"

    capture_dir = captures[data_id]
    meta_file = capture_dir / "meta.yml"
    with meta_file.open("r") as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)

    mano_sides = meta.get("mano_sides", [])
    if "right" in mano_sides:
        hand_type = "right"
    elif "left" in mano_sides:
        hand_type = "left"
    else:
        print(f"  ⚠ meta.yml 中未找到 mano_sides，默认使用右手")
        hand_type = "right"

    print(f"  自动检测手部类型: {hand_type} (来自 {capture_dir.name}/meta.yml, mano_sides={mano_sides})")
    return hand_type


def _prepare_arm_urdf(src_urdf_path: Path, arm_prefix: str) -> str:
    """
    读取浮动臂URDF，修改后写入临时文件：
    1. 将 package://r1_v2_1_0/meshes/ 替换为绝对路径
    2. 将 finger joints 从 fixed 改为 prismatic（使夹爪可以开合）
    """
    xml = src_urdf_path.read_text()
    mesh_dir_str = str(R1_MESH_DIR)
    xml = xml.replace("package://r1_v2_1_0/meshes/", mesh_dir_str + "/")
    xml = re.sub(
        rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint1"\s+type=")fixed(")',
        rf'\1prismatic\2',
        xml,
    )
    xml = re.sub(
        rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint2"\s+type=")fixed(")',
        rf'\1prismatic\2',
        xml,
    )
    temp_dir = tempfile.mkdtemp(prefix="r1_arm_urdf-")
    temp_path = f"{temp_dir}/{src_urdf_path.name}"
    with open(temp_path, "w") as f:
        f.write(xml)
    return temp_path


class TrajectorySmoother:
    """
    轨迹后处理平滑器 — 五阶段管线 + 迭代收敛

    A. 双向二阶Butterworth低通滤波 (零相位滞后)
    B. 速度限幅
    C. 加速度限幅
    D. 加加速度限幅
    E. 迭代收敛 (B→C→D循环)
    """

    SMOOTHNESS_THRESHOLDS = {
        "max_velocity": 3.0,
        "max_acceleration": 8.0,
        "max_jerk": 80.0,
        "si_improvement_min": 0.5,
    }

    def __init__(self, fps=30, max_velocity=1.5, max_acceleration=4.0,
                 max_jerk=20.0, lp_alpha=0.25, max_iterations=10, convergence_eps=1e-5):
        self.dt = 1.0 / fps
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.lp_alpha = lp_alpha
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps

    def smooth_trajectory(self, qpos_sequence, smooth_indices):
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
        max_delta = self.max_velocity * self.dt
        for i in range(1, len(trajectory)):
            delta = trajectory[i] - trajectory[i - 1]
            clamped = np.clip(delta, -max_delta, max_delta)
            trajectory[i] = trajectory[i - 1] + clamped
        return trajectory

    def _clamp_acceleration(self, trajectory):
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
        pass_vel = smooth_max_vel <= thresholds["max_velocity"]
        pass_acc = smooth_max_acc <= thresholds["max_acceleration"]
        pass_jerk = smooth_max_jerk <= thresholds["max_jerk"]
        si_improvement = 1.0 - smooth_si / max(raw_si, 1e-12)
        pass_si = si_improvement >= thresholds["si_improvement_min"]
        all_pass = pass_vel and pass_acc and pass_jerk and pass_si
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
            "pass_velocity": pass_vel,
            "pass_acceleration": pass_acc,
            "pass_jerk": pass_jerk,
            "pass_si_improvement": pass_si,
            "all_pass": all_pass,
            "si_improvement": si_improvement,
        }


def _joint_comfort_score(joint_values, joint_limits):
    mid = (joint_limits[:, 0] + joint_limits[:, 1]) / 2
    half_range = (joint_limits[:, 1] - joint_limits[:, 0]) / 2
    normalized_dist = np.abs(joint_values - mid) / half_range
    return float(1.0 - np.mean(normalized_dist))


class R1SingleArmFollower:
    """
    R1 单臂手部跟随管线

    只加载 R1 单臂 (6关节+夹爪)，通过 RelaxedIK 求解臂关节，
    通过 Dex Retargeting 求解夹爪关节。
    自动从 DexYCB 数据集检测左手/右手。
    """

    def __init__(
        self,
        dexycb_dir: str,
        data_id: int = 0,
        output_dir: str = "output_single_arm",
        fps: int = 30,
        headless: bool = True,
        loop: bool = False,
        logger: logging.Logger = None,
    ):
        self.dexycb_dir = Path(dexycb_dir)
        self.data_id = data_id
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.headless = headless
        self.loop = loop
        self.logger = logger or logging.getLogger("R1SingleArm")

    def run(self, start_frame: int = 0, num_frames: int = 50):
        """运行完整管线"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("=" * 80)
        self.logger.info("R1 单臂手部跟随管线 (DexYCB)")
        self.logger.info("=" * 80)

        # ── 阶段1: 数据加载 + 自动检测左右手 ──
        self.logger.info("\n[1/10] 加载 DexYCB 数据 + 自动检测手部类型 ...")
        hand_type_str = _detect_hand_type_from_dataset(self.dexycb_dir, self.data_id)
        self.is_right = hand_type_str == "right"
        self.arm_prefix = "right" if self.is_right else "left"
        self.hand_type = HandType.right if self.is_right else HandType.left

        dataset = DexYCBVideoDataset(self.dexycb_dir, hand_type=hand_type_str)
        sampled_data = dataset[self.data_id]
        hand_pose = sampled_data["hand_pose"]
        total_frames = hand_pose.shape[0]
        actual_frames = min(num_frames, total_frames - start_frame)
        if actual_frames <= 0:
            raise ValueError(f"帧范围无效: start={start_frame}, num={num_frames}, total={total_frames}")

        self.logger.info(f"  手部类型: {hand_type_str}, 轨迹: {total_frames} 帧")
        self.logger.info(f"  物体: {[YCB_CLASSES[yid] for yid in sampled_data['ycb_ids']]}")

        # ── 阶段2: 初始化 SAPIEN 场景 + 单臂机器人 ──
        self.logger.info("\n[2/10] 初始化 SAPIEN 场景 + 单臂机器人 ...")
        self._setup_scene()
        self._setup_arm_robot()
        self._setup_retargeting()
        self._setup_ik()

        # MANO 层
        mano_layer = MANOLayer(hand_type_str, sampled_data["hand_shape"].astype(np.float32))
        mano_face = mano_layer.f.cpu().numpy()

        # 相机外参矩阵
        extrinsic_mat = sampled_data["extrinsics"]
        pose_vec = pt.pq_from_transform(extrinsic_mat)
        camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()
        camera_mat = camera_pose.to_transformation_matrix()

        # YCB 物体
        ycb_ids = sampled_data["ycb_ids"]
        ycb_mesh_files = sampled_data["object_mesh_file"]
        objects = []
        objects_arm = []
        for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(ycb_mesh_file)
            actor = builder.build_static(name=YCB_CLASSES[ycb_id])
            objects.append(actor)
            builder2 = self.scene.create_actor_builder()
            builder2.add_visual_from_file(ycb_mesh_file)
            actor2 = builder2.build_static(name=f"{YCB_CLASSES[ycb_id]}_arm")
            objects_arm.append(actor2)

        object_pose = sampled_data["object_pose"]

        # ── 阶段3: 手部轨迹分析 ──
        self.logger.info("\n[3/10] 分析手部轨迹 ...")
        wrist_positions, hand_stats = self._analyze_hand_trajectory(
            hand_pose, mano_layer, camera_mat, start_frame, actual_frames
        )
        self.logger.info(f"  有效帧数: {len(wrist_positions)}")
        self.logger.info(f"  手腕质心: [{hand_stats['centroid'][0]:.4f}, {hand_stats['centroid'][1]:.4f}, {hand_stats['centroid'][2]:.4f}]")

        # ── 阶段4: 单臂放置 ──
        self.logger.info("\n[4/10] 放置单臂基座 ...")
        centroid = hand_stats["centroid"]
        arm_base_pos = centroid.copy()
        arm_base_pos[2] += COMFORTABLE_REACH
        z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
        arm_base_q = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))
        self.robot.set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
        self.scene.step()
        self.scene.update_render()

        base_link_p, base_link_q = self._get_arm_base_pose()
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T
        self.logger.info(f"  臂基座位置: [{arm_base_pos[0]:.4f}, {arm_base_pos[1]:.4f}, {arm_base_pos[2]:.4f}]")
        self.logger.info(f"  base_link 位置: [{base_link_p[0]:.4f}, {base_link_p[1]:.4f}, {base_link_p[2]:.4f}]")

        # ── 阶段5: 工作空间映射 ──
        self.logger.info("\n[5/10] 计算工作空间映射 ...")
        mapping_info = self._compute_workspace_mapping(hand_stats, base_link_p, base_link_R)
        self.logger.info(f"  映射偏移: [{mapping_info['mapping_offset'][0]:.4f}, {mapping_info['mapping_offset'][1]:.4f}, {mapping_info['mapping_offset'][2]:.4f}]")
        self.logger.info(f"  安全偏移: [{mapping_info['safety_offset'][0]:.4f}, {mapping_info['safety_offset'][1]:.4f}, {mapping_info['safety_offset'][2]:.4f}]")
        self.logger.info(f"  映射后质心到base距离: {mapping_info['mapped_dist_to_base']:.4f}m / 臂展{ARM_MAX_REACH:.3f}m")

        # ── 阶段6: RelaxedIK 初始化 (已在 _setup_ik 中完成) ──
        self.logger.info("\n[6/10] RelaxedIK 已就绪")

        # ── 阶段7: 预计算 ──
        self.logger.info(f"\n[7/10] 预计算 {actual_frames} 帧 (含 {WARMUP_FRAMES} 帧warmup) ...")
        qpos_sequence, ik_targets_world, eval_pre = self._precompute(
            hand_pose, start_frame, actual_frames,
            mano_layer, camera_mat, base_link_p, base_link_R_inv, mapping_info,
        )
        valid = sum(1 for x in qpos_sequence if x is not None)
        self.logger.info(f"  ✓ 预计算完成: {valid}/{actual_frames + WARMUP_FRAMES} 帧有效")

        # ── 阶段8: 轨迹后处理平滑 ──
        self.logger.info("\n[8/10] 轨迹后处理平滑 ...")
        smooth_indices = list(self.arm_joint_indices) + [self.gripper_idx1, self.gripper_idx2]
        smoother = TrajectorySmoother(
            fps=self.fps, max_velocity=1.5, max_acceleration=4.0,
            max_jerk=20.0, lp_alpha=0.25, max_iterations=10, convergence_eps=1e-5,
        )
        warmup_qpos = qpos_sequence[:WARMUP_FRAMES]
        data_qpos = qpos_sequence[WARMUP_FRAMES:]
        data_smoothed, smooth_metrics = smoother.smooth_trajectory(data_qpos, smooth_indices)
        qpos_sequence = warmup_qpos + data_smoothed
        self.logger.info(f"  ✓ 平滑完成: 速度峰值 {smooth_metrics['smooth_max_velocity']:.2f} rad/s, "
                         f"加速度峰值 {smooth_metrics['smooth_max_acceleration']:.2f} rad/s²")

        # ── 阶段9: 渲染视频 ──
        self.logger.info(f"\n[9/10] 渲染视频 ...")
        eval_render = self._render_video(
            qpos_sequence, ik_targets_world,
            hand_pose, object_pose, mano_layer, mano_face, camera_mat, camera_pose,
            objects, objects_arm, sampled_data, start_frame,
            base_link_p, base_link_R_inv, mapping_info,
        )

        # ── 阶段10: 评估报告 ──
        self.logger.info("\n[10/10] 评估报告 ...")
        self._output_evaluation(eval_pre, eval_render, hand_stats, mapping_info, valid, actual_frames, smooth_metrics)

        self.logger.info("\n" + "=" * 80)
        self.logger.info("管线执行完成！")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info("=" * 80)

    # ──────────────────────────────────────────────────────────────────
    # 场景和机器人初始化
    # ──────────────────────────────────────────────────────────────────

    def _setup_scene(self):
        """创建 SAPIEN 场景、光照、地面（与 visualize_hand_object.py 一致）"""
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")

        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)

        from sapien.asset import create_dome_envmap
        self.scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )
        self.scene.add_directional_light([1, -1, -1], [2, 2, 2], shadow=True)
        self.scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)
        self.scene.set_ambient_light([0.2, 0.2, 0.2])

        ground_mat = sapien.render.RenderMaterial()
        ground_mat.set_base_color([0.5, 0.5, 0.5, 1])
        ground_mat.set_roughness(0.7)
        ground_mat.set_metallic(1)
        ground_mat.set_specular(0.04)
        self.scene.add_ground(-1, render_material=ground_mat)

        table_mat = sapien.render.RenderMaterial()
        table_mat.set_base_color([0.8, 0.8, 0.8, 1])
        table_mat.set_roughness(0.9)
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(sapien.Pose([0, 0, -0.02]), half_size=[0.5, 2.0, 0.02])
        builder.add_box_visual(sapien.Pose([0, 0, -0.02]), half_size=[0.5, 2.0, 0.02], material=table_mat)
        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose([0.5, 0, 0]))

        if not self.headless:
            from sapien.utils import Viewer
            self.viewer = Viewer()
            self.viewer.set_scene(self.scene)
            self.viewer.set_camera_xyz(1.5, 0, 1)
            self.viewer.set_camera_rpy(0, -0.8, 3.14)
            self.viewer.control_window.toggle_origin_frame(False)
        else:
            self.camera = self.scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)
            self.camera.set_local_pose(
                sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061])
            )

        sapien.render.set_log_level("error")
        self.internal_scene = self.scene.render_system._internal_scene
        self.context = sapien.render.SapienRenderer()._internal_context
        self.mat_hand = self.context.create_material(
            np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0
        )
        self.mat_axis_x = self.context.create_material(
            np.array([1, 0, 0, 1]), np.zeros(4), 0.5, 0.5, 0
        )
        self.mat_axis_y = self.context.create_material(
            np.array([0, 1, 0, 1]), np.zeros(4), 0.5, 0.5, 0
        )
        self.mat_axis_z = self.context.create_material(
            np.array([0, 0, 1, 1]), np.zeros(4), 0.5, 0.5, 0
        )

    def _setup_arm_robot(self):
        """加载浮动臂 URDF 到 SAPIEN 场景"""
        src_urdf = FLOATING_RIGHT_URDF if self.is_right else FLOATING_LEFT_URDF
        arm_urdf_path = _prepare_arm_urdf(src_urdf, self.arm_prefix)

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        self.robot = loader.load(arm_urdf_path)

        active_joints = self.robot.get_active_joints()
        self.joint_names = [j.name for j in active_joints]
        self.arm_joint_indices = [i for i, n in enumerate(self.joint_names) if f"{self.arm_prefix}_arm_joint" in n]
        self.gripper_idx1 = self.joint_names.index(f"{self.arm_prefix}_gripper_finger_joint1")
        self.gripper_idx2 = self.joint_names.index(f"{self.arm_prefix}_gripper_finger_joint2")

        for joint in active_joints:
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        starting = RIGHT_ARM_STARTING if self.is_right else LEFT_ARM_STARTING
        init_qpos = self.robot.get_qpos().copy()
        for j, idx in enumerate(self.arm_joint_indices):
            if j < len(starting):
                init_qpos[idx] = starting[j]
        init_qpos[self.gripper_idx1] = 0.04
        init_qpos[self.gripper_idx2] = -0.04
        self.robot.set_qpos(init_qpos)

        self.ee_link = None
        for link in self.robot.get_links():
            if f"{self.arm_prefix}_gripper_link" in link.get_name():
                self.ee_link = link
                break

        self.scene.step()
        self.scene.update_render()
        self.logger.info(f"  ✓ 单臂机器人已加载: {len(self.arm_joint_indices)}个臂关节 + 2个夹爪关节")

    def _setup_retargeting(self):
        """初始化 Dex Retargeting 优化器"""
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, self.hand_type)
        gripper_link_name = f"{self.arm_prefix}_gripper_link"
        override = dict(
            add_dummy_free_joint=True,
            normal_delta=1e-5,
            huber_delta=0.01,
            target_link_names=[
                f"{self.arm_prefix}_gripper_finger_link1",
                f"{self.arm_prefix}_gripper_finger_link2",
                gripper_link_name,
            ],
            target_link_human_indices=np.array([4, 8, 0]),
        )
        self.retargeting_config = RetargetingConfig.load_from_file(config_path, override=override)
        self.retargeting: SeqRetargeting = self.retargeting_config.build()

        self.ref_indices = self.retargeting.optimizer.target_link_human_indices
        self.gripper_link_name = gripper_link_name
        self.fixed_retarget_indices = self.retargeting.optimizer.idx_pin2fixed

        retarget_joint_names = self.retargeting.joint_names
        sapien_joint_names = self.joint_names
        self.retarget2sapien = np.array(
            [retarget_joint_names.index(n) for n in sapien_joint_names if n in retarget_joint_names]
        ).astype(int)

        self._compute_fixed_qpos()
        self.logger.info(f"  ✓ Dex Retargeting 就绪 (参考关节索引: {self.ref_indices})")

    def _compute_fixed_qpos(self):
        """计算重定向优化器的固定关节默认值"""
        self.fixed_qpos = np.zeros(len(self.fixed_retarget_indices), dtype=np.float32)
        init_qpos = self.robot.get_qpos().copy()
        sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(self.retarget2sapien):
            sapien2retarget[retarget_i] = sapien_i
        for i, retarget_idx in enumerate(self.fixed_retarget_indices):
            if retarget_idx in sapien2retarget:
                self.fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

    def _setup_ik(self):
        """初始化 RelaxedIK 求解器"""
        self.ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        starting = RIGHT_ARM_STARTING if self.is_right else LEFT_ARM_STARTING
        if self.is_right:
            self.ik_solver.relaxed_ik_right.reset(starting)
        else:
            self.ik_solver.relaxed_ik_left.reset(starting)
        self.logger.info(f"  ✓ RelaxedIK 就绪 ({self.arm_prefix}臂)")

    def _get_arm_base_pose(self):
        for link in self.robot.get_links():
            if f"{self.arm_prefix}_arm_base_link" == link.get_name():
                pose = link.get_entity_pose()
                return np.array(pose.p), np.array(pose.q)
        raise RuntimeError(f"找不到 {self.arm_prefix}_arm_base_link")

    def _get_gripper_pose_from_retargeting(self, retarget_qpos_full):
        internal_robot = self.retargeting.optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos_full)
        for i, name in enumerate(internal_robot.link_names):
            if name == self.gripper_link_name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        raise RuntimeError(f"内部机器人中找不到 {self.gripper_link_name}")

    def _compute_gripper_orientation_from_hand(self, joints_world):
        wrist = joints_world[0, :3]
        thumb_tip = joints_world[4, :3]
        index_tip = joints_world[8, :3]
        index_mcp = joints_world[5, :3]
        middle_mcp = joints_world[9, :3]
        ring_mcp = joints_world[13, :3]

        y_axis = index_tip - thumb_tip
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-8:
            return None
        y_axis = y_axis / y_norm

        mcp_center = (index_mcp + middle_mcp + ring_mcp) / 3.0
        x_axis = mcp_center - wrist
        x_norm = np.linalg.norm(x_axis)
        if x_norm < 1e-8:
            return None
        x_axis = x_axis / x_norm

        z_axis = np.cross(x_axis, y_axis)
        z_norm = np.linalg.norm(z_axis)
        if z_norm < 1e-8:
            return None
        z_axis = z_axis / z_norm

        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        return np.column_stack([x_axis, y_axis, z_axis])

    def _add_capsule(self, p1, p2, radius, mat, node_list):
        mid = ((p1 + p2) / 2).astype(np.float32)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        quat = self._rotation_from_z_to_dir(direction).astype(np.float32)
        capsule_mesh = self.context.create_capsule_mesh(radius, length / 2, 8, 4)
        model = self.context.create_model([capsule_mesh], [mat])
        node = self.internal_scene.add_node()
        node.set_position(mid)
        node.set_rotation(quat)
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    def _add_sphere(self, pos, radius, mat, node_list):
        sphere_mesh = self.context.create_uvsphere_mesh(12, 6)
        model = self.context.create_model([sphere_mesh], [mat])
        node = self.internal_scene.add_node()
        node.set_position(pos.astype(np.float32))
        node.set_scale(np.array([radius, radius, radius], dtype=np.float32))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    def _rotation_from_z_to_dir(self, direction):
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
        return pr.quaternion_from_matrix(R_mat)

    def _draw_3d_axes(self, origin, R, length, node_list):
        self._add_capsule(origin, origin + R[:, 0] * length, 0.008, self.mat_axis_x, node_list)
        self._add_capsule(origin, origin + R[:, 1] * length, 0.008, self.mat_axis_y, node_list)
        self._add_capsule(origin, origin + R[:, 2] * length, 0.008, self.mat_axis_z, node_list)

    def _draw_3d_trail(self, positions, color, node_list, max_pts=200):
        if len(positions) < 2:
            return
        trail = positions[-max_pts:]
        step = max(1, len(trail) // 80)
        sampled = trail[::step]
        if len(sampled) < 2:
            return
        n = len(sampled)
        vertices = np.array(sampled, dtype=np.float32)
        colors = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            alpha = 0.3 + 0.7 * (i / max(n - 1, 1))
            colors[i] = [color[0] * alpha, color[1] * alpha, color[2] * alpha, alpha]
        line_set = self.context.create_line_set(vertices, colors)
        ls_node = self.internal_scene.add_line_set(line_set)
        ls_node.line_width = 3.0
        node_list.append(ls_node)
        self._add_sphere(sampled[-1], 0.012, self.context.create_material(
            np.array(color + [1.0]), np.zeros(4), 0.5, 0.5, 0
        ), node_list)

    # ──────────────────────────────────────────────────────────────────
    # 阶段3: 手部轨迹分析
    # ──────────────────────────────────────────────────────────────────

    def _analyze_hand_trajectory(self, hand_pose, mano_layer, camera_mat, start_frame, num_frames):
        """遍历所有帧，计算手腕位置统计"""
        wrist_positions = []
        for i in range(num_frames):
            global_idx = start_frame + i
            hp = hand_pose[global_idx]
            if hp.ndim == 1:
                hp = hp[np.newaxis, :]
            if np.abs(hp).sum() < 1e-5:
                continue
            p = torch.from_numpy(hp[:, :48].astype(np.float32))
            t = torch.from_numpy(hp[:, 48:51].astype(np.float32))
            _, joint = mano_layer(p, t)
            if joint is not None:
                jw = joint.cpu().numpy()[0] @ camera_mat[:3, :3].T + camera_mat[:3, 3]
                wrist_positions.append(jw[0, :3])

        if not wrist_positions:
            raise RuntimeError("无法从数据中提取有效手腕位置")

        positions = np.array(wrist_positions)
        return wrist_positions, {
            "centroid": np.mean(positions, axis=0),
            "range": np.ptp(positions, axis=0),
            "min": np.min(positions, axis=0),
            "max": np.max(positions, axis=0),
            "std": np.std(positions, axis=0),
            "num_valid": len(wrist_positions),
        }

    # ──────────────────────────────────────────────────────────────────
    # 阶段5: 工作空间映射
    # ──────────────────────────────────────────────────────────────────

    def _compute_workspace_mapping(self, hand_stats, base_link_p, base_link_R):
        """将手部轨迹平移到臂可达空间内"""
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
            "comfort_target_world": comfort_target_world,
            "mapped_centroid": mapped_centroid,
            "mapped_dist_to_base": mapped_dist,
            "approach_dir": approach_dir,
        }

    # ──────────────────────────────────────────────────────────────────
    # 阶段7: 预计算
    # ──────────────────────────────────────────────────────────────────

    def _precompute(
        self, hand_pose, start_frame, num_frames,
        mano_layer, camera_mat, base_link_p, base_link_R_inv, mapping_info,
    ):
        """预计算所有帧的关节角度"""
        qpos_sequence = []
        ik_targets_world = []
        eval_data = {"ik_errors": [], "joint_values": [], "out_of_reach": 0}

        mapping_offset = mapping_info["mapping_offset"]
        saved_qpos = self.robot.get_qpos().copy()

        hp_start = hand_pose[start_frame]
        if hp_start.ndim == 1:
            hp_start = hp_start[np.newaxis, :]
        p0 = torch.from_numpy(hp_start[:, :48].astype(np.float32))
        t0 = torch.from_numpy(hp_start[:, 48:51].astype(np.float32))
        _, joint0 = mano_layer(p0, t0)
        if joint0 is not None:
            j0 = joint0.cpu().numpy()[0] @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            wrist_R_cam = pr.matrix_from_compact_axis_angle(hp_start[0, :3])
            wrist_R_world = camera_mat[:3, :3] @ wrist_R_cam
            wrist_quat_world = pr.quaternion_from_matrix(wrist_R_world)
            self.retargeting.warm_start(
                j0[0, :], wrist_quat_world,
                hand_type=self.hand_type, is_mano_convention=True,
            )

        # ── 求解第一帧 ──
        first_ik_joints = None
        first_ik_target_world = None
        first_ik_target_base = None
        first_ee_quat_base = None
        first_gripper1 = 0.04
        first_gripper2 = -0.04

        for probe_idx in range(num_frames):
            global_idx = start_frame + probe_idx
            hp = hand_pose[global_idx]
            if hp.ndim == 1:
                hp = hp[np.newaxis, :]
            if np.abs(hp).sum() < 1e-5:
                continue
            p = torch.from_numpy(hp[:, :48].astype(np.float32))
            t = torch.from_numpy(hp[:, 48:51].astype(np.float32))
            _, joint = mano_layer(p, t)
            if joint is None:
                continue

            joint_world = joint.cpu().numpy()[0] @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint_world = np.ascontiguousarray(joint_world)

            ref_value = joint_world[self.ref_indices, :].astype(np.float32)
            retarget_qpos = self.retargeting.retarget(ref_value, self.fixed_qpos)
            sapien_qpos = retarget_qpos[self.retarget2sapien]
            first_gripper1 = float(sapien_qpos[self.gripper_idx1]) if self.gripper_idx1 < len(sapien_qpos) else 0.04
            first_gripper2 = float(sapien_qpos[self.gripper_idx2]) if self.gripper_idx2 < len(sapien_qpos) else -0.04

            gripper_pos_fk, gripper_R_fk = self._get_gripper_pose_from_retargeting(retarget_qpos)

            ik_target_raw = gripper_pos_fk + mapping_offset
            first_ik_target_world = ik_target_raw.copy()
            ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)

            ee_R_base = base_link_R_inv @ gripper_R_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

            try:
                solve_fn = self.ik_solver.solve_position_right if self.is_right else self.ik_solver.solve_position_left
                first_ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
                first_ik_target_base = ik_target_b.copy()
                first_ee_quat_base = ee_quat_b.copy()
                break
            except Exception:
                continue

        if first_ik_joints is None:
            raise RuntimeError("无法求解任何有效帧的IK")

        # IK预热
        self.logger.info(f"  IK预热: 迭代200次 ...")
        solve_fn = self.ik_solver.solve_position_right if self.is_right else self.ik_solver.solve_position_left
        for _ in range(200):
            first_ik_joints = np.array(solve_fn(first_ik_target_base.tolist(), first_ee_quat_base.tolist()))

        # ── Warmup过渡 ──
        current_joints = np.array([saved_qpos[i] for i in self.arm_joint_indices])
        ee_pos_filter = LPFilter(alpha=LP_ALPHA_EE)
        ee_pos_filter.next(first_ik_target_world)
        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        joint_filter.next(current_joints)

        self.logger.info(f"  Warmup: {WARMUP_FRAMES}帧过渡")
        for w in range(WARMUP_FRAMES):
            t_smooth = ((w + 1) / WARMUP_FRAMES) ** 2 * (3 - 2 * ((w + 1) / WARMUP_FRAMES))
            interp = current_joints * (1 - t_smooth) + first_ik_joints * t_smooth
            interp = joint_filter.next(interp)
            qpos = self.robot.get_qpos().copy()
            for j, idx in enumerate(self.arm_joint_indices):
                qpos[idx] = interp[j]
            qpos[self.gripper_idx1] = 0.04
            qpos[self.gripper_idx2] = -0.04
            qpos_sequence.append(qpos)
            ik_targets_world.append(first_ik_target_world.copy())

        # ── 正式预计算 ──
        for local_idx in trange(num_frames, desc="预计算"):
            global_idx = start_frame + local_idx
            hp = hand_pose[global_idx]
            if hp.ndim == 1:
                hp = hp[np.newaxis, :]
            if np.abs(hp).sum() < 1e-5:
                qpos_sequence.append(None)
                ik_targets_world.append(None)
                continue

            p = torch.from_numpy(hp[:, :48].astype(np.float32))
            t = torch.from_numpy(hp[:, 48:51].astype(np.float32))
            _, joint = mano_layer(p, t)
            if joint is None:
                qpos_sequence.append(None)
                ik_targets_world.append(None)
                continue

            joint_world = joint.cpu().numpy()[0] @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            joint_world = np.ascontiguousarray(joint_world)

            ref_value = joint_world[self.ref_indices, :].astype(np.float32)
            retarget_qpos = self.retargeting.retarget(ref_value, self.fixed_qpos)
            sapien_qpos = retarget_qpos[self.retarget2sapien]
            gripper1 = float(sapien_qpos[self.gripper_idx1]) if self.gripper_idx1 < len(sapien_qpos) else 0.04
            gripper2 = float(sapien_qpos[self.gripper_idx2]) if self.gripper_idx2 < len(sapien_qpos) else -0.04

            gripper_pos_fk, gripper_R_fk = self._get_gripper_pose_from_retargeting(retarget_qpos)

            ik_target_raw = gripper_pos_fk + mapping_offset
            ik_target_w = ee_pos_filter.next(ik_target_raw)
            ik_targets_world.append(ik_target_w.copy())
            ik_target_b = base_link_R_inv @ (ik_target_w - base_link_p)

            dist_to_base = np.linalg.norm(ik_target_b)
            if dist_to_base > ARM_MAX_REACH:
                eval_data["out_of_reach"] += 1

            ee_R_base = base_link_R_inv @ gripper_R_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

            try:
                arm_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
            except Exception as exc:
                self.logger.warning(f"  帧 {global_idx}: IK失败 - {exc}")
                qpos_sequence.append(None)
                eval_data["ik_errors"].append(str(exc))
                continue

            arm_joints = joint_filter.next(arm_joints)
            eval_data["joint_values"].append(arm_joints.copy())

            qpos = self.robot.get_qpos().copy()
            for j, idx in enumerate(self.arm_joint_indices):
                qpos[idx] = arm_joints[j]
            qpos[self.gripper_idx1] = gripper1
            qpos[self.gripper_idx2] = gripper2
            qpos_sequence.append(qpos)

        return qpos_sequence, ik_targets_world, eval_data

    # ──────────────────────────────────────────────────────────────────
    # 阶段9: 渲染视频
    # ──────────────────────────────────────────────────────────────────

    def _render_video(
        self, qpos_sequence, ik_targets_world,
        hand_pose, object_pose, mano_layer, mano_face, camera_mat, camera_pose,
        objects, objects_arm, sampled_data, start_frame,
        base_link_p, base_link_R_inv, mapping_info,
    ):
        video_path = self.output_dir / f"r1_{self.arm_prefix}_arm_follow.mp4"
        self.logger.info(f"  输出: {video_path}")

        if self.headless:
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                (self.camera.get_width(), self.camera.get_height()),
            )

        num_ycb = len(sampled_data["ycb_ids"])
        nodes = []
        viz_nodes = []
        ee_trajectory = []
        ik_target_trajectory = []
        wrist_trajectory = []
        comfort_scores = []
        fk_errors = []
        step_per_frame = int(60 / self.fps)

        arm_offset_pose = sapien.Pose(mapping_info["mapping_offset"].tolist())

        loop_count = 0
        while True:
            for frame_idx in trange(len(qpos_sequence), desc=f"渲染 (loop {loop_count+1})" if self.loop else "渲染"):
                is_warmup = frame_idx < WARMUP_FRAMES
                data_frame_idx = frame_idx - WARMUP_FRAMES
                global_idx = start_frame + max(data_frame_idx, 0)

                qpos = qpos_sequence[frame_idx]
                if qpos is not None:
                    self.robot.set_qpos(qpos)

                if not is_warmup and data_frame_idx >= 0 and global_idx < object_pose.shape[0]:
                    obj_frame = object_pose[global_idx]
                else:
                    obj_frame = object_pose[start_frame]
                for k in range(num_ycb):
                    pos_quat = obj_frame[k]
                    pose = camera_pose * sapien.Pose(
                        pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                    )
                    objects[k].set_pose(pose)
                    objects_arm[k].set_pose(arm_offset_pose * pose)

                for node in nodes:
                    self.internal_scene.remove_node(node)
                nodes.clear()

                for node in viz_nodes:
                    self.internal_scene.remove_node(node)
                viz_nodes.clear()

                hp = hand_pose[min(global_idx, hand_pose.shape[0] - 1)]
                if hp.ndim == 1:
                    hp = hp[np.newaxis, :]

                if np.abs(hp).sum() > 1e-5:
                    p = torch.from_numpy(hp[:, :48].astype(np.float32))
                    t = torch.from_numpy(hp[:, 48:51].astype(np.float32))
                    vertex, joint = mano_layer(p, t)
                    if vertex is not None:
                        vertex_np = vertex.cpu().numpy()[0]
                        vertex_world = vertex_np @ camera_mat[:3, :3].T + camera_mat[:3, 3]
                        vertex_world = np.ascontiguousarray(vertex_world)
                        normal = np.zeros_like(vertex_world)
                        v1 = vertex_world[mano_face[:, 0]]
                        v2 = vertex_world[mano_face[:, 1]]
                        v3 = vertex_world[mano_face[:, 2]]
                        face_normal = np.cross(v2 - v1, v3 - v1)
                        normal[mano_face[:, 0]] += face_normal
                        normal[mano_face[:, 1]] += face_normal
                        normal[mano_face[:, 2]] += face_normal
                        norm = np.linalg.norm(normal, axis=1, keepdims=True)
                        norm[norm < 1e-8] = 1
                        normal /= norm
                        mesh = self.context.create_mesh_from_array(vertex_world, mano_face, normal)
                        model = self.context.create_model([mesh], [self.mat_hand])
                        node = self.internal_scene.add_node()
                        node.set_position([0, 0, 0])
                        obj = self.internal_scene.add_object(model, node)
                        obj.shading_mode = 0
                        obj.cast_shadow = True
                        obj.transparency = 0
                        nodes.append(node)

                    if joint is not None and not is_warmup:
                        jw = joint.cpu().numpy()[0] @ camera_mat[:3, :3].T + camera_mat[:3, 3]
                        wrist_trajectory.append(jw[0, :3].copy())

                        wrist_aa = hp[0, :3]
                        wrist_R_cam = pr.matrix_from_compact_axis_angle(wrist_aa)
                        wrist_R_world = camera_mat[:3, :3] @ wrist_R_cam
                        self._draw_3d_axes(jw[0, :3], wrist_R_world, 0.06, viz_nodes)

                        mat_thumb = self.context.create_material(
                            np.array([1.0, 0.0, 0.0, 1.0]), np.zeros(4), 0.5, 0.5, 0)
                        mat_index = self.context.create_material(
                            np.array([0.0, 0.0, 1.0, 1.0]), np.zeros(4), 0.5, 0.5, 0)
                        self._add_sphere(jw[4, :3], 0.015, mat_thumb, viz_nodes)
                        self._add_sphere(jw[8, :3], 0.015, mat_index, viz_nodes)

                if not is_warmup and qpos is not None:
                    ee_pose = self.ee_link.get_entity_pose()
                    ee_pos = np.array(ee_pose.p)
                    ee_trajectory.append(ee_pos.copy())

                    if ik_targets_world[frame_idx] is not None:
                        ik_target_trajectory.append(ik_targets_world[frame_idx].copy())
                        error = np.linalg.norm(ee_pos - ik_targets_world[frame_idx])
                        fk_errors.append(error)

                    current_qpos = self.robot.get_qpos()
                    arm_joints = np.array([current_qpos[i] for i in self.arm_joint_indices])
                    comfort = _joint_comfort_score(arm_joints, R1_JOINT_LIMITS)
                    comfort_scores.append(comfort)

                    ee_R = pr.matrix_from_quaternion(np.array(ee_pose.q).astype(np.float64))
                    self._draw_3d_axes(ee_pos, ee_R, 0.08, viz_nodes)

                self._draw_3d_axes(base_link_p, base_link_R_inv.T, 0.12, viz_nodes)

                if len(wrist_trajectory) > 0:
                    self._draw_3d_trail(wrist_trajectory, [0, 0.65, 1], viz_nodes)
                if len(ee_trajectory) > 0:
                    self._draw_3d_trail(ee_trajectory, [0, 1, 0], viz_nodes)
                if len(ik_target_trajectory) > 0:
                    self._draw_3d_trail(ik_target_trajectory, [0, 1, 1], viz_nodes)

                self.scene.update_render()
                if self.headless:
                    self.camera.take_picture()
                    rgb = self.camera.get_picture("Color")[..., :3]
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    writer.write(rgb[..., ::-1])
                else:
                    for _ in range(step_per_frame):
                        self.viewer.render()

            if self.headless:
                writer.release()
                self.logger.info(f"  ✓ 视频已保存: {video_path}")
                break
            else:
                if not self.loop:
                    self.viewer.paused = True
                    self.viewer.render()
                    break
                ee_trajectory.clear()
                ik_target_trajectory.clear()
                wrist_trajectory.clear()
                loop_count += 1

        for node in nodes:
            self.internal_scene.remove_node(node)
        nodes.clear()
        for node in viz_nodes:
            self.internal_scene.remove_node(node)
        viz_nodes.clear()

        return {"fk_errors": fk_errors, "comfort_scores": comfort_scores}

    def _draw_annotations(
        self, bgr, frame_idx, qpos_sequence, is_warmup,
        ee_trajectory, ik_target_trajectory, wrist_trajectory,
        arm_base_pos, base_link_p, base_link_R_inv, mapping_info,
        camera, ee_pos, ee_quat_wxyz, wrist_pos,
    ):
        """在视频帧上绘制可视化标注"""
        h, w = bgr.shape[:2]
        data_frame_idx = frame_idx - WARMUP_FRAMES
        axis_len = 0.15

        # 顶部信息栏
        if is_warmup:
            t = (frame_idx + 1) / WARMUP_FRAMES
            cv2.rectangle(bgr, (0, 0), (w, 50), (0, 0, 0), -1)
            cv2.putText(bgr, f"Warmup {frame_idx+1}/{WARMUP_FRAMES} ({t*100:.0f}%)",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
            progress_w = int(w * t)
            cv2.rectangle(bgr, (0, 50), (progress_w, 56), (0, 200, 255), -1)
        else:
            cv2.rectangle(bgr, (0, 0), (w, 80), (0, 0, 0), -1)
            if data_frame_idx == 0:
                cv2.rectangle(bgr, (0, 80), (w, 140), (0, 100, 255), -1)
                cv2.putText(bgr, ">>> MAPPING START <<<",
                            (w // 2 - 250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            cv2.putText(bgr, f"Frame {data_frame_idx+1}  ({self.arm_prefix} arm)",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if ee_trajectory and ik_target_trajectory:
                err_cm = np.linalg.norm(ee_trajectory[-1] - ik_target_trajectory[-1]) * 100
                err_color = (0, 255, 0) if err_cm < 2 else (0, 255, 255) if err_cm < 5 else (0, 0, 255)
                cv2.putText(bgr, f"EE-IK: {err_cm:.1f}cm",
                            (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, err_color, 2)

            cv2.putText(bgr, f"View: Default",
                        (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 坐标系可视化
        self._draw_axes(bgr, arm_base_pos, np.eye(3), axis_len, camera, "BASE", (200, 200, 200))
        self._draw_axes(bgr, base_link_p, base_link_R_inv.T, axis_len * 0.8, camera, "ARM_BASE", (0, 200, 200))

        if ee_pos is not None and ee_quat_wxyz is not None:
            ee_R = pr.matrix_from_quaternion(ee_quat_wxyz)
            self._draw_axes(bgr, ee_pos, ee_R, axis_len * 0.6, camera, "EE", (0, 255, 0))

        if wrist_pos is not None:
            self._draw_axes(bgr, wrist_pos, np.eye(3), axis_len * 0.6, camera, "HAND", (0, 165, 255))

        # 轨迹线
        if len(ee_trajectory) >= 2:
            pts_2d = [self._project_point(p, camera) for p in ee_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                alpha_val = max(0.3, i / len(pts_2d))
                color = (0, int(255 * alpha_val), int(255 * (1 - alpha_val)))
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], color, 2)

        if len(ik_target_trajectory) >= 2:
            pts_2d = [self._project_point(p, camera) for p in ik_target_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], (255, 255, 0), 1)

        if len(wrist_trajectory) >= 2:
            pts_2d = [self._project_point(p, camera) for p in wrist_trajectory]
            pts_2d = [p for p in pts_2d if p is not None]
            for i in range(1, len(pts_2d)):
                cv2.line(bgr, pts_2d[i - 1], pts_2d[i], (0, 165, 255), 1)

        # 当前位置标记
        if ee_trajectory:
            ee_2d = self._project_point(ee_trajectory[-1], camera)
            if ee_2d is not None:
                cv2.circle(bgr, ee_2d, 6, (0, 255, 0), -1)
                cv2.circle(bgr, ee_2d, 6, (255, 255, 255), 1)

        if ik_target_trajectory:
            tgt_2d = self._project_point(ik_target_trajectory[-1], camera)
            if tgt_2d is not None:
                cv2.circle(bgr, tgt_2d, 6, (0, 255, 255), -1)

        if wrist_trajectory:
            w_2d = self._project_point(wrist_trajectory[-1], camera)
            if w_2d is not None:
                cv2.circle(bgr, w_2d, 5, (0, 165, 255), -1)

        # 右侧数值面板
        if not is_warmup:
            panel_x = w - 380
            panel_y = 90
            cv2.rectangle(bgr, (panel_x, panel_y), (w, panel_y + 220), (0, 0, 0), -1)
            cv2.putText(bgr, "Position Data (m):", (panel_x + 5, panel_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            row = 0
            if ee_pos is not None:
                cv2.putText(bgr, f"EE:", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
                cv2.putText(bgr, f"{ee_pos[0]:.3f} {ee_pos[1]:.3f} {ee_pos[2]:.3f}",
                            (panel_x + 40, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1
            if wrist_pos is not None:
                cv2.putText(bgr, f"HAND:", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 1)
                cv2.putText(bgr, f"{wrist_pos[0]:.3f} {wrist_pos[1]:.3f} {wrist_pos[2]:.3f}",
                            (panel_x + 55, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1
            if ik_target_trajectory:
                ik_p = ik_target_trajectory[-1]
                cv2.putText(bgr, f"IK:", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
                cv2.putText(bgr, f"{ik_p[0]:.3f} {ik_p[1]:.3f} {ik_p[2]:.3f}",
                            (panel_x + 40, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1

            cv2.putText(bgr, f"ARM_BASE:", (panel_x + 5, panel_y + 38 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 200), 1)
            cv2.putText(bgr, f"{base_link_p[0]:.3f} {base_link_p[1]:.3f} {base_link_p[2]:.3f}",
                        (panel_x + 85, panel_y + 38 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            row += 1

            if ee_pos is not None and ik_target_trajectory:
                err = np.linalg.norm(ee_pos - ik_target_trajectory[-1])
                err_color = (0, 255, 0) if err < 0.02 else (0, 255, 255) if err < 0.05 else (0, 0, 255)
                cv2.putText(bgr, f"EE-IK err: {err*100:.2f}cm", (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, err_color, 1)
                row += 1

            if ee_pos is not None and base_link_p is not None:
                reach = np.linalg.norm(ee_pos - base_link_p)
                cv2.putText(bgr, f"Reach: {reach:.3f}m / {ARM_MAX_REACH:.3f}m",
                            (panel_x + 5, panel_y + 38 + row * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
                row += 1

        # 图例
        if not is_warmup:
            legend_y = h - 80
            cv2.rectangle(bgr, (10, legend_y - 5), (340, h - 5), (0, 0, 0), -1)
            cv2.circle(bgr, (25, legend_y + 10), 5, (0, 255, 0), -1)
            cv2.putText(bgr, "Robot EE", (40, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.circle(bgr, (25, legend_y + 30), 5, (0, 255, 255), -1)
            cv2.putText(bgr, "IK target", (40, legend_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.circle(bgr, (25, legend_y + 50), 5, (0, 165, 255), -1)
            cv2.putText(bgr, "Hand wrist", (40, legend_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return bgr

    def _draw_axes(self, bgr, origin, R, length, camera, label, label_color):
        """绘制3D坐标轴"""
        origin_2d = self._project_point(origin, camera)
        if origin_2d is None:
            return
        x_end = origin + R[:, 0] * length
        y_end = origin + R[:, 1] * length
        z_end = origin + R[:, 2] * length
        x_2d = self._project_point(x_end, camera)
        y_2d = self._project_point(y_end, camera)
        z_2d = self._project_point(z_end, camera)
        lw = 2
        if x_2d is not None:
            cv2.line(bgr, origin_2d, x_2d, (0, 0, 255), lw)
        if y_2d is not None:
            cv2.line(bgr, origin_2d, y_2d, (0, 255, 0), lw)
        if z_2d is not None:
            cv2.line(bgr, origin_2d, z_2d, (255, 0, 0), lw)
        cv2.putText(bgr, label, (origin_2d[0] - 15, origin_2d[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

    def _project_point(self, point_3d, camera):
        """将3D点投影到2D图像平面"""
        try:
            ext = camera.get_extrinsic_matrix().astype(np.float64)
            int_mat = camera.get_intrinsic_matrix().astype(np.float64)
            R = ext[:3, :3]
            t = ext[:3, 3]
            p_cam = R @ point_3d + t
            if p_cam[2] <= 0.01:
                return None
            uv = int_mat @ p_cam
            u = int(uv[0] / uv[2])
            v = int(uv[1] / uv[2])
            if 0 <= u < camera.get_width() and 0 <= v < camera.get_height():
                return (u, v)
            return None
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────
    # 阶段10: 评估报告
    # ──────────────────────────────────────────────────────────────────

    def _output_evaluation(self, eval_pre, eval_render, hand_stats, mapping_info, valid, total, smooth_metrics):
        """输出评估报告"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("评估报告")
        self.logger.info("=" * 80)

        self.logger.info("\n── 1. 手部轨迹分析 ──")
        self.logger.info(f"  有效帧数:     {hand_stats['num_valid']}")
        self.logger.info(f"  手腕质心:     [{hand_stats['centroid'][0]:.4f}, {hand_stats['centroid'][1]:.4f}, {hand_stats['centroid'][2]:.4f}]")
        self.logger.info(f"  运动范围:     X={hand_stats['range'][0]:.4f}m, Y={hand_stats['range'][1]:.4f}m, Z={hand_stats['range'][2]:.4f}m")

        self.logger.info("\n── 2. 工作空间映射 ──")
        self.logger.info(f"  映射偏移:     [{mapping_info['mapping_offset'][0]:.4f}, {mapping_info['mapping_offset'][1]:.4f}, {mapping_info['mapping_offset'][2]:.4f}]")
        self.logger.info(f"  安全距离:     {SAFETY_DISTANCE:.3f}m")
        self.logger.info(f"  映射后质心到base: {mapping_info['mapped_dist_to_base']:.4f}m / 臂展{ARM_MAX_REACH:.3f}m")

        self.logger.info("\n── 3. IK 求解统计 ──")
        self.logger.info(f"  有效帧:       {valid}/{total + WARMUP_FRAMES} (含{WARMUP_FRAMES}帧warmup)")
        self.logger.info(f"  IK失败帧:     {len(eval_pre['ik_errors'])}")
        self.logger.info(f"  超出臂展帧:   {eval_pre['out_of_reach']}/{total}")

        fk_errors = eval_render["fk_errors"]
        if fk_errors:
            self.logger.info("\n── 4. FK 验证 ──")
            self.logger.info(f"  平均位置误差: {np.mean(fk_errors)*100:.2f} cm")
            self.logger.info(f"  最大位置误差: {np.max(fk_errors)*100:.2f} cm")

        comfort_scores = eval_render["comfort_scores"]
        if comfort_scores:
            self.logger.info("\n── 5. 关节舒适度 ──")
            self.logger.info(f"  平均舒适度:   {np.mean(comfort_scores):.4f} (1.0=最佳, 0.0=最差)")
            self.logger.info(f"  最低舒适度:   {np.min(comfort_scores):.4f}")

        if smooth_metrics:
            self.logger.info("\n── 6. 运动平滑度 ──")
            self.logger.info(f"  ┌─────────────────────┬──────────┬──────────┬──────┐")
            self.logger.info(f"  │ 指标                │ 平滑前   │ 平滑后   │ 判定 │")
            self.logger.info(f"  ├─────────────────────┼──────────┼──────────┼──────┤")
            vel_pass = "✓" if smooth_metrics["pass_velocity"] else "✗"
            acc_pass = "✓" if smooth_metrics["pass_acceleration"] else "✗"
            jerk_pass = "✓" if smooth_metrics["pass_jerk"] else "✗"
            si_pass = "✓" if smooth_metrics["pass_si_improvement"] else "✗"
            self.logger.info(f"  │ 最大角速度(rad/s)   │ {smooth_metrics['raw_max_velocity']:8.2f} │ {smooth_metrics['smooth_max_velocity']:8.2f} │  {vel_pass}   │")
            self.logger.info(f"  │ 最大角加速度(rad/s²)│ {smooth_metrics['raw_max_acceleration']:8.2f} │ {smooth_metrics['smooth_max_acceleration']:8.2f} │  {acc_pass}   │")
            self.logger.info(f"  │ 最大加加速度(rad/s³)│ {smooth_metrics['raw_max_jerk']:8.1f} │ {smooth_metrics['smooth_max_jerk']:8.1f} │  {jerk_pass}   │")
            self.logger.info(f"  │ 平滑度指数(SI)      │ {smooth_metrics['raw_smoothness_index']:8.1f} │ {smooth_metrics['smooth_smoothness_index']:8.1f} │  {si_pass}   │")
            self.logger.info(f"  └─────────────────────┴──────────┴──────────┴──────┘")
            overall = "✓ PASS" if smooth_metrics["all_pass"] else "✗ FAIL"
            self.logger.info(f"  综合判定: {overall}")

        self.logger.info("\n" + "=" * 80)


def _setup_logger(output_dir: Path) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger("R1SingleArm")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "r1_single_arm.log"
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def main():
    """
    主函数：命令行入口

    用法:
      python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb
      python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb --data-id 4 --num-frames 100
      python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb --headless --view front
    """
    parser = argparse.ArgumentParser(description="R1 单臂手部跟随 (DexYCB, 自动检测左右手)")
    parser.add_argument("--dexycb-dir", type=str, required=True, help="DexYCB 数据集根目录")
    parser.add_argument("--data-id", type=int, default=0, help="DexYCB 数据索引")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--num-frames", type=int, default=50, help="帧数")
    parser.add_argument("--output-dir", type=str, default="output_single_arm", help="输出目录")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--headless", action="store_true", help="无头模式渲染视频")
    parser.add_argument("--loop", action="store_true", help="循环播放 (交互模式)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    logger = _setup_logger(output_dir)

    follower = R1SingleArmFollower(
        dexycb_dir=args.dexycb_dir,
        data_id=args.data_id,
        output_dir=args.output_dir,
        fps=args.fps,
        headless=args.headless,
        loop=args.loop,
        logger=logger,
    )
    follower.run(start_frame=args.start_frame, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
