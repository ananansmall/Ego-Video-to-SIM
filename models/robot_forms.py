"""三种机器人形式定义: full_robot / floating_arm / gripper_only

形式对应:
  - full_robot (人): r1_v2_1_0.urdf, 完整机器人含底座/躯干/双臂/双夹爪
  - floating_arm (机械臂): r1_v2_1_0_floating_right.urdf, 浮动单臂+夹爪
  - gripper_only (夹爪): 模板生成, 仅夹爪本体+两手指

每种形式提取结构信息: 关节名/索引/初始位姿/驱动参数
"""

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from physics_utils import (
    FULL_ROBOT_URDF, FLOATING_ARM_RIGHT_URDF, FLOATING_ARM_LEFT_URDF,
    R1_MESH_DIR, RIGHT_ARM_STARTING, LEFT_ARM_STARTING,
    JOINT_STIFFNESS, JOINT_DAMPING, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
    GRIPPER_INIT_OPEN,
)
from models.urdf_templates import generate_gripper_only_urdf


@dataclass
class RobotFormInfo:
    """机器人形式结构信息"""
    name: str  # full_robot / floating_arm / gripper_only
    urdf_path: str
    description: str
    arm_joint_names: List[str] = field(default_factory=list)
    gripper_joint_names: List[str] = field(default_factory=list)
    arm_joint_indices: List[int] = field(default_factory=list)
    gripper_joint_indices: List[int] = field(default_factory=list)
    init_qpos: np.ndarray = None
    has_arm: bool = True
    has_gripper: bool = True
    has_base: bool = True  # 是否有底座/躯干
    is_floating: bool = False


def _parse_urdf_joints(urdf_path):
    """解析URDF文件, 提取关节信息

    Returns:
        list of dict: 每个关节 {name, type, parent, child, axis, limit}
    """
    with open(urdf_path, 'r') as f:
        content = f.read()

    joints = []
    # 简单正则解析 (避免依赖 yourdfpy)
    joint_pattern = re.compile(
        r'<joint\s+name="([^"]+)"\s+type="([^"]+)">(.*?)</joint>',
        re.DOTALL
    )
    for m in joint_pattern.finditer(content):
        name, jtype, body = m.group(1), m.group(2), m.group(3)
        parent_m = re.search(r'<parent\s+link="([^"]+)"', body)
        child_m = re.search(r'<child\s+link="([^"]+)"', body)
        axis_m = re.search(r'<axis\s+xyz="([^"]+)"', body)
        limit_m = re.search(r'<limit\s+([^/]+)/>', body)

        joint = {
            'name': name,
            'type': jtype,
            'parent': parent_m.group(1) if parent_m else None,
            'child': child_m.group(1) if child_m else None,
            'axis': axis_m.group(1) if axis_m else None,
        }
        if limit_m:
            limit_str = limit_m.group(1)
            lower = re.search(r'lower="([^"]+)"', limit_str)
            upper = re.search(r'upper="([^"]+)"', limit_str)
            joint['lower'] = float(lower.group(1)) if lower else None
            joint['upper'] = float(upper.group(1)) if upper else None
        joints.append(joint)
    return joints


def prepare_full_robot_urdf():
    """准备完整机器人URDF (full_robot 形式)

    使用 r1_v2_1_0.urdf, 包含完整结构: base_link → torso → arms → grippers
    mesh路径为 package://r1_v2_1_0/meshes/..., SAPIEN/PyBullet需处理

    Returns:
        str: URDF文件路径
    """
    if not FULL_ROBOT_URDF.exists():
        raise FileNotFoundError(f"完整机器人URDF不存在: {FULL_ROBOT_URDF}")
    return str(FULL_ROBOT_URDF)


def prepare_floating_arm_urdf(side="right"):
    """准备浮动臂URDF (floating_arm 形式)

    使用 r1_v2_1_0_floating_right/left.urdf, 浮动底座+单臂+夹爪

    Args:
        side: "right" 或 "left"

    Returns:
        str: URDF文件路径
    """
    if side == "right":
        path = FLOATING_ARM_RIGHT_URDF
    else:
        path = FLOATING_ARM_LEFT_URDF
    if not path.exists():
        raise FileNotFoundError(f"浮动臂URDF不存在: {path}")
    return str(path)


def prepare_gripper_only_urdf(side="right"):
    """准备纯夹爪URDF (gripper_only 形式)

    使用模板生成, 仅夹爪本体+两手指, 无机械臂

    Args:
        side: "right" 或 "left"

    Returns:
        str: 生成的URDF文件路径
    """
    return generate_gripper_only_urdf(side)


def get_full_robot_info():
    """获取完整机器人形式信息

    Returns:
        RobotFormInfo: 完整机器人结构信息
    """
    urdf_path = prepare_full_robot_urdf()
    joints = _parse_urdf_joints(urdf_path)

    # 注意: 完整URDF中arm_joint可能是fixed类型(展示模型), 也匹配
    right_arm_joints = [j['name'] for j in joints
                        if j['name'].startswith('right_arm_joint')]
    left_arm_joints = [j['name'] for j in joints
                       if j['name'].startswith('left_arm_joint')]
    right_gripper_joints = [j['name'] for j in joints
                            if j['name'].startswith('right_gripper_finger_joint')]
    left_gripper_joints = [j['name'] for j in joints
                           if j['name'].startswith('left_gripper_finger_joint')]

    # 按关节编号排序
    def sort_key(name):
        nums = re.findall(r'\d+', name)
        return int(nums[-1]) if nums else 0

    right_arm_joints.sort(key=sort_key)
    right_gripper_joints.sort(key=sort_key)

    return RobotFormInfo(
        name="full_robot",
        urdf_path=urdf_path,
        description="完整R1机器人: 底座+躯干+双臂+双夹爪",
        arm_joint_names=right_arm_joints,
        gripper_joint_names=right_gripper_joints,
        init_qpos=None,  # 由后端加载后设置
        has_arm=True,
        has_gripper=True,
        has_base=True,
        is_floating=False,
    )


def get_floating_arm_info(side="right"):
    """获取浮动臂形式信息

    Returns:
        RobotFormInfo: 浮动臂结构信息
    """
    urdf_path = prepare_floating_arm_urdf(side)
    joints = _parse_urdf_joints(urdf_path)

    prefix = side
    arm_joints = [j['name'] for j in joints
                  if j['name'].startswith(f'{prefix}_arm_joint')]
    gripper_joints = [j['name'] for j in joints
                      if j['name'].startswith(f'{prefix}_gripper_finger_joint')]

    def sort_key(name):
        nums = re.findall(r'\d+', name)
        return int(nums[-1]) if nums else 0

    arm_joints.sort(key=sort_key)
    gripper_joints.sort(key=sort_key)

    return RobotFormInfo(
        name="floating_arm",
        urdf_path=urdf_path,
        description=f"浮动{side}臂: 浮动底座+单臂+夹爪",
        arm_joint_names=arm_joints,
        gripper_joint_names=gripper_joints,
        init_qpos=None,
        has_arm=True,
        has_gripper=True,
        has_base=False,
        is_floating=True,
    )


def get_gripper_only_info(side="right"):
    """获取纯夹爪形式信息

    Returns:
        RobotFormInfo: 纯夹爪结构信息
    """
    urdf_path = prepare_gripper_only_urdf(side)
    joints = _parse_urdf_joints(urdf_path)

    prefix = side
    gripper_joints = [j['name'] for j in joints
                      if j['name'].startswith(f'{prefix}_gripper_finger_joint')]

    def sort_key(name):
        nums = re.findall(r'\d+', name)
        return int(nums[-1]) if nums else 0

    gripper_joints.sort(key=sort_key)

    return RobotFormInfo(
        name="gripper_only",
        urdf_path=urdf_path,
        description=f"纯{side}夹爪: 仅夹爪本体+两手指(无机械臂)",
        arm_joint_names=[],
        gripper_joint_names=gripper_joints,
        init_qpos=None,
        has_arm=False,
        has_gripper=True,
        has_base=False,
        is_floating=False,
    )


def get_robot_form_info(form_name, side="right"):
    """根据形式名获取机器人形式信息

    Args:
        form_name: "full_robot" / "floating_arm" / "gripper_only"
        side: "right" 或 "left"

    Returns:
        RobotFormInfo
    """
    if form_name == "full_robot":
        return get_full_robot_info()
    elif form_name == "floating_arm":
        return get_floating_arm_info(side)
    elif form_name == "gripper_only":
        return get_gripper_only_info(side)
    else:
        raise ValueError(f"未知机器人形式: {form_name}, 可选: full_robot/floating_arm/gripper_only")


def get_init_qpos(form_name, side="right"):
    """获取初始关节角

    Args:
        form_name: 形式名
        side: "right" 或 "left"

    Returns:
        dict: {joint_name: angle}
    """
    qpos = {}
    if form_name in ("full_robot", "floating_arm"):
        starting = RIGHT_ARM_STARTING if side == "right" else LEFT_ARM_STARTING
        info = get_robot_form_info(form_name, side)
        for i, name in enumerate(info.arm_joint_names):
            if i < len(starting):
                qpos[name] = starting[i]
        # 夹爪初始开合
        for name in info.gripper_joint_names:
            if "joint1" in name:
                qpos[name] = GRIPPER_INIT_OPEN
            elif "joint2" in name:
                qpos[name] = -GRIPPER_INIT_OPEN
    elif form_name == "gripper_only":
        info = get_gripper_only_info(side)
        for name in info.gripper_joint_names:
            if "joint1" in name:
                qpos[name] = GRIPPER_INIT_OPEN
            elif "joint2" in name:
                qpos[name] = GRIPPER_INIT_OPEN
    return qpos


ALL_FORMS = ["full_robot", "floating_arm", "gripper_only"]
