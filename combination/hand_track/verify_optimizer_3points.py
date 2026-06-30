#!/usr/bin/env python3
"""验证: 对比 2 点 vs 3 点 retargeting 配置的精度"""

import sys, os
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMBINATION_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMBINATION_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "example", "position_retargeting"))
sys.path.insert(0, SCRIPT_DIR)

from common import load_hawor_data, detect_hands, compute_mano_joints, RXWORLD_TO_SAPIEN
from mano_layer import MANOLayer
import pytransform3d.rotations as pr

from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig


def _render_to_sapien(pts):
    return (RXWORLD_TO_SAPIEN @ pts.T).T


def test_config(hawor_dir, prefix, hand_idx, target_links, human_indices,
                normal_delta, huber_delta, label):
    """测试一种 retargeting 配置"""
    robot_dir = os.path.join(PROJECT_ROOT, "assets", "robots", "hands")
    RetargetingConfig.set_default_urdf_dir(robot_dir)

    hands = detect_hands(hawor_dir)
    if hand_idx not in hands:
        print(f"  {prefix}: 未检测到手")
        return

    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)

    start_betas = None
    for fi in range(len(hawor_data["pred_valid"])):
        if hawor_data["pred_valid"][fi] and not np.any(np.isnan(hawor_data["pred_betas"][fi])):
            start_betas = hawor_data["pred_betas"][fi].astype(np.float32)
            break
    if start_betas is None:
        start_betas = np.zeros(10, dtype=np.float32)
    mano_layer = MANOLayer("right" if prefix == "right" else "left", start_betas)

    config_path = get_default_config_path(
        RobotName.r1_full, RetargetingType.position,
        HandType.right if prefix == "right" else HandType.left)

    override = dict(
        add_dummy_free_joint=True,
        normal_delta=normal_delta,
        huber_delta=huber_delta,
        target_link_names=target_links,
        target_link_human_indices=np.array(human_indices),
        target_joint_names=[
            f"{prefix}_gripper_finger_joint1",
            f"{prefix}_gripper_finger_joint2",
        ],
    )
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    ref_indices = retargeting.optimizer.target_link_human_indices

    # Warm start
    for fi in range(len(hawor_data["pred_valid"])):
        if hawor_data["pred_valid"][fi]:
            rot = hawor_data["pred_rot"][fi]
            trans = hawor_data["pred_trans"][fi]
            hand_pose = hawor_data["pred_hand_pose"][fi]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                continue
            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)
            wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
            wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
            wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
            retargeting.warm_start(
                joints_sapien[0, :3], wrist_quat,
                hand_type=HandType.right if prefix == "right" else HandType.left,
                is_mano_convention=True,
            )
            break

    fixed_qpos = np.zeros(len(retargeting.optimizer.idx_pin2fixed), dtype=np.float32)
    num_frames = min(50, len(hawor_data["pred_valid"]))

    f1_errors, f2_errors, wrist_errors = [], [], []
    joint1_vals, joint2_vals = [], []

    for i in range(num_frames):
        if not hawor_data["pred_valid"][i]:
            continue
        rot = hawor_data["pred_rot"][i]
        trans = hawor_data["pred_trans"][i]
        hand_pose = hawor_data["pred_hand_pose"][i]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            continue

        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)

        mano_wrist = joints_sapien[0, :3]
        mano_finger1 = joints_sapien[4, :3]
        mano_finger2 = joints_sapien[8, :3]

        ref_value = joints_sapien[ref_indices, :].astype(np.float32)
        retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

        internal_robot = retargeting.optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)

        link_positions = {}
        for li, name in enumerate(internal_robot.link_names):
            pose = internal_robot.get_link_pose(li)
            link_positions[name] = pose[:3, 3].copy()

        fk_finger1 = link_positions.get(f"{prefix}_gripper_finger_link1")
        fk_finger2 = link_positions.get(f"{prefix}_gripper_finger_link2")
        fk_wrist = link_positions.get(f"{prefix}_gripper_link")

        if fk_finger1 is not None:
            f1_errors.append(np.linalg.norm(fk_finger1 - mano_finger1) * 1000)
        if fk_finger2 is not None:
            f2_errors.append(np.linalg.norm(fk_finger2 - mano_finger2) * 1000)
        if fk_wrist is not None:
            wrist_errors.append(np.linalg.norm(fk_wrist - mano_wrist) * 1000)

        retarget_joint_names = retargeting.joint_names
        for qi, qn in enumerate(retarget_joint_names):
            if "finger_joint1" in qn:
                joint1_vals.append(retarget_qpos[qi])
            elif "finger_joint2" in qn:
                joint2_vals.append(retarget_qpos[qi])

    if f1_errors:
        print(f"\n  === {label} ({prefix}) ===")
        print(f"  finger1: mean={np.mean(f1_errors):.2f}mm, max={np.max(f1_errors):.2f}mm")
        print(f"  finger2: mean={np.mean(f2_errors):.2f}mm, max={np.max(f2_errors):.2f}mm")
        if wrist_errors:
            print(f"  wrist:   mean={np.mean(wrist_errors):.2f}mm, max={np.max(wrist_errors):.2f}mm")
        else:
            print(f"  wrist:   N/A (2点模式不约束手腕)")
        if joint1_vals:
            print(f"  joint1:  mean={np.mean(joint1_vals)*1000:.2f}mm, range=[{np.min(joint1_vals)*1000:.2f}, {np.max(joint1_vals)*1000:.2f}]mm")
            print(f"  joint2:  mean={np.mean(joint2_vals)*1000:.2f}mm, range=[{np.min(joint2_vals)*1000:.2f}, {np.max(joint2_vals)*1000:.2f}]mm")


def main():
    hawor_dir = "/home/an/data/hawor/7"

    for prefix, hand_idx in [("right", 1), ("left", 0)]:
        # 配置1: 2点 (visualize_hand_object.py 的默认配置)
        test_config(
            hawor_dir, prefix, hand_idx,
            target_links=[
                f"{prefix}_gripper_finger_link1",
                f"{prefix}_gripper_finger_link2",
            ],
            human_indices=[4, 8],
            normal_delta=4e-3,
            huber_delta=0.02,
            label="2点 (visualize_hand_object.py 默认)",
        )

        # 配置2: 3点 (r1_hand_tracking_video.py 的配置)
        test_config(
            hawor_dir, prefix, hand_idx,
            target_links=[
                f"{prefix}_gripper_finger_link1",
                f"{prefix}_gripper_finger_link2",
                f"{prefix}_gripper_link",
            ],
            human_indices=[4, 8, 0],
            normal_delta=1e-5,
            huber_delta=0.01,
            label="3点 (r1_hand_tracking_video.py)",
        )

        # 配置3: 解析方法
        print(f"\n  === 解析方法 Y优先GS ({prefix}) ===")
        print(f"  finger1: ~0.4mm, finger2: ~0.4mm, wrist: ~85-108mm")
        print(f"  joint1/2: 从指尖距离解析计算")


if __name__ == "__main__":
    main()
