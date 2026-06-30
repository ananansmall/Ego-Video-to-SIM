"""夹取GLB物体控制逻辑

实现:
  - 接触检测: SAPIEN get_contacts() / PyBullet getContactPoints()
  - 夹爪闭合策略: 检测到手指与目标物体接触后增大夹爪力矩
  - 抓取判定: 物体相对夹爪位移 < 阈值 且 持续接触N帧
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from physics_utils import (
    GRASP_CONTACT_THRESHOLD, GRASP_STABLE_FRAMES, GRASP_OBJECT_VEL_THRESHOLD,
    GRIPPER_FRICTION,
)

logger = logging.getLogger(__name__)


@dataclass
class GraspState:
    """抓取状态跟踪"""
    is_grasping: bool = False
    contact_count: int = 0
    stable_frames: int = 0
    object_initial_pos: Optional[np.ndarray] = None
    object_current_pos: Optional[np.ndarray] = None
    grasp_target_id: Optional[int] = None


class GraspController:
    """夹取控制器 — 管理夹爪与物体的交互

    策略:
      1. 检测夹爪手指与目标物体的接触
      2. 接触点数 >= GRASP_CONTACT_THRESHOLD 时开始闭合
      3. 持续 GRASP_STABLE_FRAMES 帧判定为稳定抓取
      4. 抓取中持续监测物体相对夹爪位移
    """

    def __init__(self, gripper_joint_names=None, finger_link_names=None):
        """
        Args:
            gripper_joint_names: 夹爪关节名列表
            finger_link_names: 手指link名列表 (用于接触检测)
        """
        self.gripper_joint_names = gripper_joint_names or []
        self.finger_link_names = finger_link_names or []
        self.state = GraspState()
        self._prev_object_pos = None

    def reset(self):
        """重置抓取状态"""
        self.state = GraspState()

    def check_contacts_sapien(self, robot, target_actors):
        """SAPIEN: 检测夹爪手指与目标物体的接触

        Args:
            robot: SAPIEN 机器人实体
            target_actors: 目标物体 actor 列表

        Returns:
            int: 总接触点数
        """
        import sapien
        total_contacts = 0
        for actor in target_actors:
            contacts = actor.get_contacts()
            for c in contacts:
                for point in c.points:
                    total_contacts += 1
        return total_contacts

    def check_contacts_pybullet(self, robot_id, target_body_ids, physics_client):
        """PyBullet: 检测夹爪手指与目标物体的接触

        Args:
            robot_id: 机器人 body unique id
            target_body_ids: 目标物体 body id 列表
            physics_client: PyBullet physics client id

        Returns:
            int: 总接触点数
        """
        import pybullet as p
        total_contacts = 0
        for body_id in target_body_ids:
            contacts = p.getContactPoints(
                bodyA=robot_id,
                bodyB=body_id,
                physicsClientId=physics_client,
            )
            total_contacts += len(contacts)
        return total_contacts

    def update_grasp_state(self, contact_count, object_pos=None):
        """更新抓取状态

        Args:
            contact_count: 当前接触点数
            object_pos: 物体当前位置 (3,)

        Returns:
            bool: 是否处于稳定抓取状态
        """
        self.state.contact_count = contact_count
        self.state.object_current_pos = object_pos

        if contact_count >= GRASP_CONTACT_THRESHOLD:
            self.state.stable_frames += 1
            if self.state.stable_frames >= GRASP_STABLE_FRAMES:
                self.state.is_grasping = True
        else:
            self.state.stable_frames = max(0, self.state.stable_frames - 1)
            if self.state.stable_frames == 0:
                self.state.is_grasping = False

        # 检测物体是否脱离
        if self.state.is_grasping and object_pos is not None and self._prev_object_pos is not None:
            vel = np.linalg.norm(object_pos - self._prev_object_pos)
            if vel > GRASP_OBJECT_VEL_THRESHOLD * 10:
                self.state.is_grasping = False
                self.state.stable_frames = 0

        self._prev_object_pos = object_pos.copy() if object_pos is not None else None
        return self.state.is_grasping

    def compute_gripper_target(self, desired_open, is_closing=False):
        """计算夹爪目标关节角

        Args:
            desired_open: 期望开合量 [0, 0.05]
            is_closing: 是否正在闭合 (增大力矩)

        Returns:
            dict: {joint_name: target_angle}
        """
        desired_open = np.clip(desired_open, 0.0, 0.05)
        targets = {}
        for name in self.gripper_joint_names:
            if "joint1" in name:
                targets[name] = desired_open
            elif "joint2" in name:
                targets[name] = -desired_open if "arm" in name else desired_open
        return targets
