#!/usr/bin/env python3
"""
align_strategy.py — 用户要求的对齐策略 (新文件)

策略 (用户原话):
  "关键先对齐夹爪两点, 最后第三个点, 是对齐位姿, 能够在同一条中轴线上即可"
  "MANO参数里面夹爪的中点和第三个手腕点的连线确定位姿"
  "最后把机械臂手腕的点放到位姿线上即可"
  "对齐的主要是夹爪末端, 次要是手腕, 但位姿一定要对"

实现:
  1. 主要: 对齐夹爪两点 (拇指尖[4], 食指尖[8]) 到 MANO 指尖
  2. 位姿: 夹爪中点 → 手腕[0] 连线确定 X 轴 (指向方向), 必须正确
  3. 次要: 机械臂手腕 (arm_link6) 放在位姿线上 (同一条中轴线)

与 gripper_config.compute_analytical_gripper_pose 的区别:
  - 增加夹爪开合缩放因子 (GRIPPER_OPEN_SCALE), 让开合更明显
  - 增加位姿线对齐验证 (verify_alignment)
  - 增加手腕到位姿线的投影计算 (project_wrist_to_pose_line)
"""

import numpy as np
from pytransform3d import rotations as pr

from gripper_config import (
    FINGER_BASE_DIST, FINGER_GEOM_ARRAYS,
    LP_ALPHA_ANALYTICAL, PositionEmaSmoother,
)

# 夹爪开合缩放因子
# 1.0 = 精确映射 (MANO 指尖距离 = 夹爪指尖距离)
# 3.0 = 放大 3 倍 (默认, 让开合更明显, 因为 MANO 拇指-食指距离通常较小)
GRIPPER_OPEN_SCALE = 3.0


def compute_gripper_pose_aligned(mano_wrist, mano_finger1, mano_finger2, prefix="right",
                                  open_scale=GRIPPER_OPEN_SCALE):
    """用户对齐策略: 先对齐夹爪两点, 再用中点-手腕连线确定位姿

    步骤:
      1. 指尖中点 midpoint = (finger1 + finger2) / 2
      2. X 轴 (指向方向) = normalize(midpoint - wrist)  ← 位姿核心
      3. Y 轴 (开合方向) = normalize(finger2-finger1 投影到 X 垂直面)
      4. Z = X × Y
      5. 手指关节 = (指尖距离 - 基准距离) * 缩放因子 / 2
      6. gripper_link 位置 = midpoint - R @ finger_mid_in_gripper  ← 对齐夹爪两点

    Args:
        mano_wrist: (3,) MANO 手腕位置 (joint 0)
        mano_finger1: (3,) MANO 拇指尖位置 (joint 4)
        mano_finger2: (3,) MANO 食指尖位置 (joint 8)
        prefix: "left" 或 "right"
        open_scale: 夹爪开合缩放因子 (默认 3.0, 放大开合效果)

    Returns:
        gripper_pos: (3,) gripper_link 位置 (指尖中点后方)
        gripper_R: (3,3) gripper_link 旋转矩阵
        joint1, joint2: 手指关节值
    """
    fg = FINGER_GEOM_ARRAYS[prefix]
    finger1_origin = fg["finger1_origin"]
    finger2_origin = fg["finger2_origin"]

    # 1. 指尖中点
    midpoint = (mano_finger1 + mano_finger2) / 2

    # 2. X 轴: 手腕 → 指尖中点 (指向方向, 位姿核心)
    v_pointing = midpoint - mano_wrist
    norm = np.linalg.norm(v_pointing)
    if norm < 1e-6:
        X = np.array([1.0, 0.0, 0.0])
    else:
        X = v_pointing / norm

    # 3. Y 轴: finger2-finger1 投影到 X 的垂直面 (开合方向)
    v_opening = mano_finger2 - mano_finger1
    v_opening_proj = v_opening - np.dot(v_opening, X) * X
    norm = np.linalg.norm(v_opening_proj)
    if norm < 1e-6:
        Y = np.array([0.0, 1.0, 0.0])
    else:
        Y = v_opening_proj / norm

    # 4. 确保 Y 方向与 URDF 几何一致 (夹爪 finger2 在 +Y 侧)
    gripper_opening_local = finger2_origin - finger1_origin
    sign = np.sign(gripper_opening_local[1])
    if sign * np.dot(Y, v_opening) < 0:
        Y = -Y

    # 5. Z = X × Y, 组装旋转矩阵
    Z = np.cross(X, Y)
    R = np.column_stack([X, Y, Z])
    if np.linalg.det(R) < 0:
        Z = -Z
        R = np.column_stack([X, Y, Z])

    # 6. 手指关节 (带缩放因子, 让开合更明显)
    #    使用 3D 指尖距离 (而非投影距离), 因为 MANO 拇指-食指开合主要沿指向方向,
    #    投影距离接近 0 会导致夹爪不开合。3D 距离能更好反映手的开合状态。
    #    注意: 这会引入 ~11mm 指尖位置误差 (1-DOF 夹爪无法匹配 3D 开合方向), 属于固有局限。
    finger_dist = np.linalg.norm(mano_finger2 - mano_finger1)
    required_open = max(0.0, finger_dist - FINGER_BASE_DIST)
    joint = min(0.05, required_open * open_scale / 2)
    joint1 = joint
    joint2 = joint

    # 7. gripper_link 位置: 指尖中点后方 (对齐夹爪两点)
    finger1_in_gripper = finger1_origin + fg["finger1_axis"] * joint1
    finger2_in_gripper = finger2_origin + fg["finger2_axis"] * joint2
    finger_mid_in_gripper = (finger1_in_gripper + finger2_in_gripper) / 2
    gripper_pos = midpoint - R @ finger_mid_in_gripper

    return gripper_pos, R, joint1, joint2


def compute_arm_root_pose(gripper_pos, gripper_R, gripper_offset_pos, gripper_offset_R):
    """计算机械臂 root 位姿, 使夹爪在 gripper_pos, 位姿为 gripper_R

    root_R = gripper_R @ gripper_offset_R.T
    root_pos = gripper_pos - root_R @ gripper_offset_pos

    Args:
        gripper_pos: (3,) 目标 gripper_link 位置
        gripper_R: (3,3) 目标 gripper_link 旋转
        gripper_offset_pos: (3,) gripper_link 相对 root 的位置偏移 (arm_joint=0 时)
        gripper_offset_R: (3,3) gripper_link 相对 root 的旋转偏移

    Returns:
        root_pos: (3,) root 位置
        root_quat: (4,) root 四元数 [w, x, y, z]
        root_R: (3,3) root 旋转矩阵
    """
    root_R = gripper_R @ gripper_offset_R.T
    root_pos = gripper_pos - root_R @ gripper_offset_pos
    root_quat = pr.quaternion_from_matrix(root_R)
    return root_pos, root_quat, root_R


def project_wrist_to_pose_line(wrist_pos, mano_wrist, mano_midpoint):
    """计算手腕到位姿线的投影距离 (验证手腕是否在位姿线上)

    位姿线 = mano_wrist → mano_midpoint 的连线
    手腕到位姿线的距离 = |(wrist - mano_wrist) - proj|

    Args:
        wrist_pos: (3,) 实际手腕位置 (arm_link6)
        mano_wrist: (3,) MANO 手腕位置
        mano_midpoint: (3,) MANO 指尖中点

    Returns:
        dist_mm: 手腕到位姿线的垂直距离 (mm)
        proj_pos: (3,) 手腕在位姿线上的投影点
    """
    direction = mano_midpoint - mano_wrist
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 0.0, mano_wrist.copy()
    direction = direction / norm
    vec = wrist_pos - mano_wrist
    proj_len = np.dot(vec, direction)
    proj_pos = mano_wrist + proj_len * direction
    perp = vec - proj_len * direction
    dist_mm = np.linalg.norm(perp) * 1000
    return dist_mm, proj_pos


def verify_alignment(robot, prefix, mano_wrist, mano_finger1, mano_finger2, scene):
    """验证对齐效果

    检查:
      1. 指尖位置误差 (主要)
      2. 位姿方向误差 (必须正确)
      3. 手腕到位姿线距离 (次要)

    Args:
        robot: SAPIEN Articulation
        prefix: "left" 或 "right"
        mano_wrist, mano_finger1, mano_finger2: MANO 特征点
        scene: SAPIEN Scene (用于 update_render)

    Returns:
        dict: 各项误差
    """
    scene.update_render()
    results = {}

    # 获取实际 link 位置
    for link in robot.get_links():
        name = link.get_name()
        pose = link.get_entity_pose()
        if name == f"{prefix}_gripper_finger_link1":
            results['finger1_actual'] = np.array(pose.p)
        elif name == f"{prefix}_gripper_finger_link2":
            results['finger2_actual'] = np.array(pose.p)
        elif name == f"{prefix}_gripper_link":
            results['gripper_actual'] = np.array(pose.p)
            results['gripper_R_actual'] = pr.matrix_from_quaternion(np.array(pose.q))
        elif name == f"{prefix}_arm_link6":
            results['wrist_actual'] = np.array(pose.p)

    midpoint = (mano_finger1 + mano_finger2) / 2

    # 1. 指尖误差 (主要)
    if 'finger1_actual' in results:
        results['finger1_err_mm'] = float(
            np.linalg.norm(results['finger1_actual'] - mano_finger1) * 1000)
    if 'finger2_actual' in results:
        results['finger2_err_mm'] = float(
            np.linalg.norm(results['finger2_actual'] - mano_finger2) * 1000)

    # 2. 位姿误差 (X 轴与 midpoint-wrist 方向的夹角, 必须正确)
    mano_pointing = midpoint - mano_wrist
    norm = np.linalg.norm(mano_pointing)
    if norm > 1e-6 and 'gripper_R_actual' in results:
        mano_pointing = mano_pointing / norm
        gripper_x = results['gripper_R_actual'][:, 0]
        cos_angle = np.clip(np.dot(gripper_x, mano_pointing), -1, 1)
        results['pose_err_deg'] = float(np.degrees(np.arccos(cos_angle)))

    # 3. 手腕到位姿线距离 (次要, 只需在同一条中轴线上)
    if 'wrist_actual' in results:
        dist_mm, proj_pos = project_wrist_to_pose_line(
            results['wrist_actual'], mano_wrist, midpoint)
        results['wrist_line_dist_mm'] = float(dist_mm)

    return results


def print_verification(results, logger=None):
    """打印验证结果"""
    import sys
    out = logger.info if logger else print

    def p(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    p("  === 对齐验证 ===")
    if 'finger1_err_mm' in results:
        p(f"  指尖1误差 (主要): {results['finger1_err_mm']:.2f} mm")
    if 'finger2_err_mm' in results:
        p(f"  指尖2误差 (主要): {results['finger2_err_mm']:.2f} mm")
    if 'pose_err_deg' in results:
        status = "✓" if results['pose_err_deg'] < 5.0 else "✗"
        p(f"  位姿误差 (必须): {results['pose_err_deg']:.2f} deg {status}")
    if 'wrist_line_dist_mm' in results:
        status = "✓" if results['wrist_line_dist_mm'] < 50.0 else "~"
        p(f"  手腕到位姿线 (次要): {results['wrist_line_dist_mm']:.2f} mm {status}")
