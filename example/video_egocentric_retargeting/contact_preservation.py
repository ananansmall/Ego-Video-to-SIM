"""
接触保持滤波器和轨迹平滑器

ContactPreservationFilter: 基于迟滞比较器的抓取/释放检测 + 关节变化限幅
TrajectorySmoother: 滑动窗口中值滤波 + 低通滤波组合

核心设计思想：
  1. 迟滞比较器：抓取阈值和释放阈值不同，避免在阈值附近频繁切换状态
  2. 抓取限幅：抓取状态下限制关节变化速率，防止夹爪意外松开
  3. 中值+低通组合：先中值滤波去除脉冲噪声，再低通滤波平滑整体轨迹
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
from dex_retargeting.optimizer_utils import LPFilter


class ContactPreservationFilter:
    """
    接触保持滤波器

    基于手指距离的迟滞比较器检测抓取/释放状态，
    并在抓取状态下限制关节变化速率以防止夹爪意外松开。

    迟滞比较器工作原理：
      - 当手指距离 < grasp_threshold 时，切换到抓取状态
      - 当手指距离 > release_threshold 时，切换到释放状态
      - grasp_threshold < release_threshold，形成迟滞区间
      - 在迟滞区间内保持当前状态不变，避免频繁切换

    抓取限幅工作原理：
      - 抓取状态下，每帧关节变化量不超过 max_change (0.05)
      - 超过限制的变化会被截断，防止夹爪因噪声意外松开

    Args:
        grasp_threshold: 抓取检测阈值，手指距离低于此值判定为抓取（默认 0.04m）
        release_threshold: 释放检测阈值，手指距离高于此值判定为释放（默认 0.06m）
        smoothing_alpha: 夹爪关节低通滤波系数（默认 0.3，0=完全平滑，1=无滤波）
        position_alpha: 位置低通滤波系数（默认 0.5）
    """

    def __init__(
        self,
        grasp_threshold=0.04,
        release_threshold=0.06,
        smoothing_alpha=0.3,
        position_alpha=0.5,
    ):
        self.grasp_threshold = grasp_threshold
        self.release_threshold = release_threshold
        self.smoothing_alpha = smoothing_alpha
        self.position_alpha = position_alpha

        self.is_grasping = False
        # 夹爪关节低通滤波器（逐关节独立滤波）
        self.gripper_filter = LPFilter(alpha=smoothing_alpha)
        # 位置低通滤波器列表（延迟初始化，根据关节数量创建）
        self.position_filter = None
        self.last_qpos = None

    def update(self, finger_distance, qpos):
        """
        更新滤波器状态

        步骤：
          1. 迟滞比较器判断抓取/释放状态
          2. 对关节值进行低通滤波平滑
          3. 抓取状态下限制关节变化速率

        Args:
            finger_distance: 当前帧的拇指-食指距离
            qpos: 当前帧的重定向关节值

        Returns:
            (filtered_qpos, is_grasping): 滤波后的关节值和抓取状态
        """
        # 迟滞比较器：根据手指距离切换抓取/释放状态
        if finger_distance < self.grasp_threshold and not self.is_grasping:
            self.is_grasping = True
        elif finger_distance > self.release_threshold and self.is_grasping:
            self.is_grasping = False

        # 延迟初始化位置滤波器（根据关节数量创建）
        if self.position_filter is None:
            self.position_filter = [LPFilter(alpha=self.position_alpha)] * len(qpos)

        # 对每个关节独立进行低通滤波
        filtered_qpos = np.array(
            [
                self.position_filter[i].next(qpos[i])
                for i in range(len(qpos))
            ]
        )

        # 抓取限幅：抓取状态下限制关节变化速率，防止夹爪意外松开
        if self.is_grasping and self.last_qpos is not None:
            finger_change = np.abs(filtered_qpos - self.last_qpos)
            max_change = 0.05  # 每帧最大允许变化量
            mask = finger_change > max_change
            # 超过限制的变化被截断到 max_change
            filtered_qpos[mask] = self.last_qpos[mask] + np.sign(
                filtered_qpos[mask] - self.last_qpos[mask]
            ) * max_change

        self.last_qpos = filtered_qpos.copy()
        return filtered_qpos, self.is_grasping


class TrajectorySmoother:
    """
    轨迹平滑器

    使用滑动窗口中值滤波 + 低通滤波的组合策略平滑关节轨迹。
    中值滤波去除脉冲噪声（如检测跳变），低通滤波平滑整体轨迹。

    工作流程：
      1. 将当前帧的关节值加入滑动窗口缓冲区
      2. 当缓冲区填满时，对窗口内数据取中值
      3. 对中值结果进行低通滤波，输出最终平滑值
      4. 缓冲区未填满时直接返回原始值

    Args:
        window_size: 滑动窗口大小（默认 5 帧）
        alpha: 低通滤波系数（默认 0.3，0=完全平滑，1=无滤波）
    """

    def __init__(self, window_size=5, alpha=0.3):
        self.window_size = window_size
        self.alpha = alpha
        self.buffer = []
        self.lp_filter = None

    def smooth(self, qpos):
        """
        平滑关节轨迹

        Args:
            qpos: 当前帧的关节值

        Returns:
            ndarray: 平滑后的关节值
        """
        # 将当前帧加入缓冲区
        self.buffer.append(qpos.copy())
        # 维持缓冲区大小不超过窗口大小
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

        # 缓冲区未填满时直接返回原始值
        if len(self.buffer) < self.window_size:
            return qpos

        # 对窗口内数据取中值（去除脉冲噪声）
        window_arr = np.array(self.buffer)
        median = np.median(window_arr, axis=0)

        # 延迟初始化低通滤波器
        if self.lp_filter is None:
            self.lp_filter = LPFilter(alpha=self.alpha)

        # 对中值结果进行低通滤波
        return self.lp_filter.next(median)

    def reset(self):
        """重置滤波器状态（清空缓冲区和低通滤波器）"""
        self.buffer = []
        self.lp_filter = None
