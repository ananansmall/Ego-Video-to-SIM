"""
人手到机器人手的重定向器封装

将 dex_retargeting 库的 SeqRetargeting 优化器封装为易用的接口，
支持位置重定向（POSITION）和向量重定向（VECTOR）两种模式。

核心功能：
  - 加载机器人 URDF 配置并构建重定向优化器
  - 计算人手关节到机器人关节的参考值
  - 支持热启动（warm_start）加速优化收敛
  - 支持固定非目标关节（fixed_qpos）用于仅重定向部分关节
  - 提供元数据查询接口

重定向类型说明：
  - POSITION: 直接匹配关节位置，适用于指尖到指尖的映射
  - VECTOR: 匹配关节间向量方向，适用于手指朝向的映射
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting


class RobotHandRetargeter:
    """
    人手到机器人手的重定向器

    封装 dex_retargeting 的 SeqRetargeting 优化器，将 MediaPipe 检测的
    人手 21 个关键点映射到机器人手部关节角度。

    Args:
        robot_name: 机器人名称枚举（如 RobotName.allegro, RobotName.r1）
        hand_type: 手别枚举（HandType.right / HandType.left）
        retargeting_type: 重定向类型（RetargetingType.position / RetargetingType.vector）
    """

    def __init__(
        self,
        robot_name=RobotName.allegro,
        hand_type=HandType.right,
        retargeting_type=RetargetingType.position,
    ):
        self.robot_name = robot_name
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type

        # 设置 URDF 搜索目录
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        # 获取默认重定向配置文件路径
        config_path = get_default_config_path(robot_name, retargeting_type, hand_type)

        # 位置重定向需要添加 6 自由度虚拟关节（使机器人手可在空间中自由移动）
        if retargeting_type == RetargetingType.position:
            override = {"add_dummy_free_joint": True}
        else:
            override = {}

        self.config_path = str(config_path)
        # 加载配置并构建 SeqRetargeting 优化器
        self.config = RetargetingConfig.load_from_file(config_path, override=override)
        self.retargeting: SeqRetargeting = self.config.build()
        self.warm_started = False

    def compute_ref_value(self, joint_pos):
        """
        根据重定向类型计算参考值

        POSITION 模式：直接提取目标关节的 3D 位置
        VECTOR 模式：计算关节间向量（task - origin）

        Args:
            joint_pos: 人手 21 个关键点的 3D 坐标 (21, 3)

        Returns:
            ndarray: 参考值数组，形状取决于重定向类型
        """
        retargeting_type = self.retargeting.optimizer.retargeting_type
        indices = self.retargeting.optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            # 位置重定向：直接提取目标关节位置
            ref_value = joint_pos[indices, :]
        else:
            # 向量重定向：计算关节间向量
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

        return ref_value.astype(np.float32)

    def warm_start(self, wrist_pos, wrist_quat):
        """
        热启动：使用手腕位姿初始化优化器

        提供初始手腕位姿可以显著加速 NLopt 优化器的收敛速度，
        避免从零开始搜索最优解。

        Args:
            wrist_pos: 手腕 3D 位置 (3,)
            wrist_quat: 手腕四元数 (4,)
        """
        self.retargeting.warm_start(
            wrist_pos,
            wrist_quat,
            hand_type=self.hand_type,
            is_mano_convention=True,
        )
        self.warm_started = True

    def retarget(self, joint_pos, wrist_pos=None, wrist_quat=None, fixed_qpos=None):
        """
        执行重定向优化

        将人手关节位置映射到机器人关节角度。使用 NLopt SLSQP 优化器
        求解最小化人手与机器人关节位置差异的最优解。

        Args:
            joint_pos: 人手 21 个关键点的 3D 坐标 (21, 3)
            wrist_pos: 手腕位置（用于热启动，可选）
            wrist_quat: 手腕四元数（用于热启动，可选）
            fixed_qpos: 固定关节的值数组（非目标关节保持固定，可选）
                例如 R1 机器人中，仅重定向夹爪关节时固定其他关节

        Returns:
            ndarray: 机器人关节角度
        """
        # 首次调用时自动热启动
        if not self.warm_started and wrist_pos is not None and wrist_quat is not None:
            self.warm_start(wrist_pos, wrist_quat)

        ref_value = self.compute_ref_value(joint_pos)
        # 传入 fixed_qpos 时，优化器会固定指定关节的值
        if fixed_qpos is not None:
            qpos = self.retargeting.retarget(ref_value, fixed_qpos)
        else:
            qpos = self.retargeting.retarget(ref_value)
        return qpos

    def get_meta_data(self):
        """
        获取重定向器的元数据

        Returns:
            dict: 包含配置路径、机器人名称、手别、自由度、关节名等信息
        """
        return {
            "config_path": self.config_path,
            "robot_name": self.robot_name.value,
            "hand_type": self.hand_type.value,
            "retargeting_type": self.retargeting_type.value
            if isinstance(self.retargeting_type, RetargetingType)
            else self.retargeting_type,
            "dof": len(self.retargeting.optimizer.robot.dof_joint_names),
            "joint_names": self.retargeting.optimizer.robot.dof_joint_names,
        }
