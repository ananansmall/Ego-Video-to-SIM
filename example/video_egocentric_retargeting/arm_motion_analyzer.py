"""
手臂运动分析器

从 MediaPipe 检测的手部关键点中计算手腕 6DoF 位姿（位置 + 四元数）、
手指开合距离、手掌法向量等运动学信息，为后续 IK 求解和抓取检测提供输入。

主要功能：
  - compute_wrist_6dof(): 计算手腕 6 自由度位姿（位置 + 旋转）
  - compute_finger_aperture(): 计算拇指尖与食指尖的距离（抓取状态指标）
  - compute_gripper_value(): 将手指距离映射为夹爪开合值
  - compute_palm_normal(): 计算手掌法向量
  - compute_hand_openness(): 计算手部张开程度

坐标系变换说明（R_cam2base）：
  第一人称视频的相机坐标系与机器人臂基座坐标系朝向相同，
  因为相机就是从机器人第一人称视角拍摄的。

  相机坐标系：+X右, +Y下, +Z前(远离相机)
  臂基座坐标系(左臂)：+X前, +Y左, +Z上

  直接映射关系：
    Camera +Z (前) → ArmBase +X (前)
    Camera +X (右) → ArmBase -Y (右, 因为+Y是左)
    Camera +Y (下) → ArmBase -Z (下, 因为+Z是上)

  R_cam2base = [[0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]]

  注意：之前使用 R_ego→R_180Z 两步变换（Camera→World→ArmBase），
  数学上等价但增加了不必要的中间步骤。由于第一人称视角下
  相机和臂基座朝向一致，直接变换更直观、更不容易出错。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from dex_retargeting.constants import OPERATOR2MANO, HandType


class ArmMotionAnalyzer:
    """
    手臂运动分析器

    从手部关键点数据中提取手腕位姿、手指距离等运动学信息。
    支持通过相机外参将手腕位姿从相机坐标系转换到世界坐标系。

    Args:
        hand_type: 手别枚举值（HandType.right / HandType.left）
        arm_offset: 末端执行器相对于手腕的偏移量（默认为零向量）
    """

    def __init__(self, hand_type=HandType.right, arm_offset=np.zeros(3)):
        self.hand_type = hand_type
        self.arm_offset = arm_offset
        # OPERATOR2MANO: 操作者坐标系到 MANO 坐标系的变换矩阵
        self.operator2mano = OPERATOR2MANO[hand_type]

    def compute_wrist_6dof(self, hand_data, camera_extrinsics=None):
        """
        计算手腕 6 自由度位姿（3D 位置 + 四元数旋转）

        从 MediaPipe 检测结果中提取手腕位置和旋转矩阵，
        使用相机坐标系下的手腕位置（wrist_pos_camera）直接变换到臂基座坐标系。

        关键设计：第一人称视角下，相机和臂基座朝向一致，
        直接用 R_cam2base 一步变换，无需经过世界坐标系。

        变换关系：
          Camera +Z(前) → ArmBase +X(前)
          Camera +X(右) → ArmBase -Y(右)
          Camera +Y(下) → ArmBase -Z(下)

        Args:
            hand_data: EgocentricHandDetector.detect_single() 的返回值
                - joint_pos: 21 个手部关键点的 3D 坐标 (21, 3)
                - wrist_pos_raw: 手腕位置(手心局部坐标) (3,)
                - wrist_pos_camera: 手腕位置(相机坐标) (3,)
                - wrist_rot_mano: 手腕旋转矩阵 (3, 3)，MANO 坐标系
            camera_extrinsics: 相机外参矩阵 (4, 4)，用于坐标变换
                如果为 None，则使用原始 MANO 坐标系

        Returns:
            dict: 包含以下键值：
                - wrist_pos: 手腕 3D 位置 (3,)，臂基座坐标系
                - wrist_rot: 手腕旋转矩阵 (3, 3)
                - wrist_quat: 手腕四元数 [x, y, z, w] (4,)
                - ee_pos: 末端执行器位置 = 手腕位置 + 偏移 (3,)
                - ee_quat: 末端执行器四元数 (4,)
                - wrist_pose_4x4: 手腕齐次变换矩阵 (4, 4)
        """
        joint_pos = hand_data["joint_pos"]
        wrist_pos_camera = hand_data.get("wrist_pos_camera", hand_data["wrist_pos_raw"])
        wrist_rot_mano = hand_data["wrist_rot_mano"]

        wrist_pos_base = wrist_pos_camera
        wrist_rot_base = wrist_rot_mano

        if camera_extrinsics is not None:
            R_cam2world = camera_extrinsics[:3, :3]
            t_cam2world = camera_extrinsics[:3, 3]
            wrist_pos_base = R_cam2world @ wrist_pos_camera + t_cam2world
            wrist_rot_base = R_cam2world @ wrist_rot_mano

        R_cam2base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)
        wrist_pos_base = R_cam2base @ wrist_pos_base
        wrist_rot_base = R_cam2base @ wrist_rot_base

        wrist_pose_4x4 = np.eye(4)
        wrist_pose_4x4[:3, :3] = wrist_rot_base
        wrist_pose_4x4[:3, 3] = wrist_pos_base

        ee_pos = wrist_pos_base + self.arm_offset
        ee_quat = R.from_matrix(wrist_rot_base).as_quat()

        return {
            "wrist_pos": wrist_pos_base,
            "wrist_rot": wrist_rot_base,
            "wrist_quat": ee_quat,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "wrist_pose_4x4": wrist_pose_4x4,
        }

    @staticmethod
    def compute_finger_aperture(joint_pos, thumb_idx=4, index_idx=8):
        """
        计算拇指尖与食指尖的欧几里得距离

        该距离是判断抓取状态的核心指标：
        - 距离小（< grasp_threshold）→ 抓取状态
        - 距离大（> release_threshold）→ 释放状态

        Args:
            joint_pos: 21 个手部关键点的 3D 坐标 (21, 3)
            thumb_idx: 拇指尖的索引（默认 4，MediaPipe 标准索引）
            index_idx: 食指尖的索引（默认 8，MediaPipe 标准索引）

        Returns:
            float: 拇指尖与食指尖的欧几里得距离
        """
        thumb_tip = joint_pos[thumb_idx]
        index_tip = joint_pos[index_idx]
        distance = np.linalg.norm(thumb_tip - index_tip)
        return distance

    @staticmethod
    def compute_gripper_value(
        finger_distance,
        close_dist=0.02,
        open_dist=0.08,
        max_open=0.05,
    ):
        """
        将手指距离映射为夹爪开合值

        线性映射：手指距离从 close_dist 到 open_dist 映射到夹爪值从 0 到 max_open。
        超出范围的值会被截断到 [0, max_open]。

        Args:
            finger_distance: 拇指尖与食指尖的距离
            close_dist: 完全闭合时对应的手指距离（默认 0.02m）
            open_dist: 完全张开时对应的手指距离（默认 0.08m）
            max_open: 夹爪最大开合值（默认 0.05m）

        Returns:
            float: 夹爪开合值，范围 [0, max_open]
        """
        normalized = np.clip(
            (finger_distance - close_dist) / (open_dist - close_dist), 0, 1
        )
        return normalized * max_open

    @staticmethod
    def compute_palm_normal(joint_pos):
        """
        计算手掌法向量

        使用手腕(0)、食指MCP(5)、中指MCP(9)三点构建两个向量，
        通过叉积计算法向量并归一化。

        Args:
            joint_pos: 21 个手部关键点的 3D 坐标 (21, 3)

        Returns:
            ndarray: 归一化的手掌法向量 (3,)
        """
        wrist = joint_pos[0]
        index_mcp = joint_pos[5]
        middle_mcp = joint_pos[9]
        v1 = index_mcp - wrist
        v2 = middle_mcp - wrist
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        return normal

    @staticmethod
    def compute_hand_openness(joint_pos):
        """
        计算手部张开程度

        定义为 5 个指尖到手腕的平均欧几里得距离。
        值越大表示手越张开，值越小表示手越握拢。

        Args:
            joint_pos: 21 个手部关键点的 3D 坐标 (21, 3)

        Returns:
            float: 5 指尖到手腕的平均距离
        """
        wrist = joint_pos[0]
        tips = [joint_pos[i] for i in [4, 8, 12, 16, 20]]
        distances = [np.linalg.norm(tip - wrist) for tip in tips]
        return np.mean(distances)
