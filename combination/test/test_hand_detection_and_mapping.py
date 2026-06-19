"""
test_hand_detection_and_mapping.py — 测试手部检测与机械臂映射

测试内容:
  Part 1: HandDetector — 手部类型检测
    - 从 pred_valid 数组检测 (1D / 2D)
    - 从 npz 文件检测
    - 从 cam_space 目录检测
    - 边界情况: 空数据, 阈值边界

  Part 2: RobotArmMapper — 机械臂配置映射
    - 左手 → 左臂配置
    - 右手 → 右臂配置
    - 双手 → 双臂配置
    - 验证关节名/URDF路径/Retargeting 配置的正确性

  Part 3: 端到端集成测试
    - 从 HaWoR 数据目录检测手部 → 生成机械臂配置 → 验证一致性

运行方式:
    cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
    python -m test.test_hand_detection_and_mapping
    # 或
    python test/test_hand_detection_and_mapping.py
"""

import sys
import os
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hand_track"))

from hand_detector import HandDetector, Handedness, HandDetectionResult
from robot_arm_config import (
    RobotArmMapper,
    RobotArmConfig,
    SingleArmConfig,
    FLOATING_RIGHT_URDF,
    FLOATING_LEFT_URDF,
    DUAL_ARM_URDF,
    R1_RIGHT_SETTINGS,
    R1_LEFT_SETTINGS,
    R1_MESH_DIR,
    OPERATOR2MANO_RIGHT,
    OPERATOR2MANO_LEFT,
    RIGHT_ARM_STARTING,
    LEFT_ARM_STARTING,
    prepare_arm_urdf,
    prepare_dual_arm_urdf,
)


PASSED = 0
FAILED = 0


def _assert(condition: bool, msg: str):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ {msg}")
    else:
        FAILED += 1
        print(f"  ❌ {msg}")


# ──────────────────────────────────────────────
# Part 1: HandDetector 测试
# ──────────────────────────────────────────────

def test_pred_valid_2d_right_only():
    print("\n[测试] pred_valid 2D — 仅右手")
    pred_valid = np.zeros((2, 100), dtype=bool)
    pred_valid[1, :] = True
    result = HandDetector.detect_from_pred_valid(pred_valid)
    _assert(result.handedness == Handedness.RIGHT, f"应为 RIGHT, 实际为 {result.handedness}")
    _assert(result.right_valid_frames == 100, f"右手有效帧应为 100, 实际为 {result.right_valid_frames}")
    _assert(result.left_valid_frames == 0, f"左手有效帧应为 0, 实际为 {result.left_valid_frames}")
    _assert(result.hand_indices == [1], f"hand_indices 应为 [1], 实际为 {result.hand_indices}")


def test_pred_valid_2d_left_only():
    print("\n[测试] pred_valid 2D — 仅左手")
    pred_valid = np.zeros((2, 100), dtype=bool)
    pred_valid[0, :] = True
    result = HandDetector.detect_from_pred_valid(pred_valid)
    _assert(result.handedness == Handedness.LEFT, f"应为 LEFT, 实际为 {result.handedness}")
    _assert(result.left_valid_frames == 100, f"左手有效帧应为 100, 实际为 {result.left_valid_frames}")
    _assert(result.right_valid_frames == 0, f"右手有效帧应为 0, 实际为 {result.right_valid_frames}")
    _assert(result.hand_indices == [0], f"hand_indices 应为 [0], 实际为 {result.hand_indices}")


def test_pred_valid_2d_both():
    print("\n[测试] pred_valid 2D — 双手")
    pred_valid = np.ones((2, 80), dtype=bool)
    result = HandDetector.detect_from_pred_valid(pred_valid)
    _assert(result.handedness == Handedness.BOTH, f"应为 BOTH, 实际为 {result.handedness}")
    _assert(result.left_valid_frames == 80, f"左手有效帧应为 80, 实际为 {result.left_valid_frames}")
    _assert(result.right_valid_frames == 80, f"右手有效帧应为 80, 实际为 {result.right_valid_frames}")
    _assert(result.hand_indices == [0, 1], f"hand_indices 应为 [0, 1], 实际为 {result.hand_indices}")


def test_pred_valid_2d_sparse():
    print("\n[测试] pred_valid 2D — 稀疏数据 (一只手低于阈值)")
    pred_valid = np.zeros((2, 200), dtype=bool)
    pred_valid[0, :3] = True
    pred_valid[1, :20] = True
    result = HandDetector.detect_from_pred_valid(pred_valid)
    _assert(result.handedness == Handedness.RIGHT, f"应为 RIGHT (左手占比1.5%<5%, 右手10%>5%), 实际为 {result.handedness}")
    _assert(result.left_valid_frames == 3, f"左手有效帧应为 3, 实际为 {result.left_valid_frames}")
    _assert(result.right_valid_frames == 20, f"右手有效帧应为 20, 实际为 {result.right_valid_frames}")


def test_pred_valid_2d_none():
    print("\n[测试] pred_valid 2D — 无手部数据")
    pred_valid = np.zeros((2, 100), dtype=bool)
    result = HandDetector.detect_from_pred_valid(pred_valid)
    _assert(result.handedness == Handedness.NONE, f"应为 NONE, 实际为 {result.handedness}")


def test_pred_valid_1d():
    print("\n[测试] pred_valid 1D — 单手 (默认右手)")
    pred_valid = np.ones(50, dtype=bool)
    result = HandDetector.detect_from_pred_valid(pred_valid)
    _assert(result.handedness == Handedness.RIGHT, f"应为 RIGHT, 实际为 {result.handedness}")
    _assert(result.right_valid_frames == 50, f"右手有效帧应为 50, 实际为 {result.right_valid_frames}")


def test_detect_from_npz():
    print("\n[测试] 从 npz 文件检测")
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test_results.npz"
        pred_valid = np.zeros((2, 60), dtype=bool)
        pred_valid[1, :] = True
        np.savez(str(npz_path), pred_valid=pred_valid, pred_trans=np.zeros((2, 60, 3)))

        result = HandDetector.detect_from_npz(str(npz_path))
        _assert(result.handedness == Handedness.RIGHT, f"应为 RIGHT, 实际为 {result.handedness}")
        _assert(result.right_valid_frames == 60, f"右手有效帧应为 60, 实际为 {result.right_valid_frames}")


def test_detect_from_npz_no_pred_valid():
    print("\n[测试] 从 npz 文件检测 — 无 pred_valid")
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test_no_valid.npz"
        np.savez(str(npz_path), data=np.zeros(10))

        result = HandDetector.detect_from_npz(str(npz_path))
        _assert(result.handedness == Handedness.NONE, f"应为 NONE, 实际为 {result.handedness}")


def test_detect_from_cam_space():
    print("\n[测试] 从 cam_space 目录检测")
    with tempfile.TemporaryDirectory() as tmpdir:
        hawor_dir = Path(tmpdir) / "hawor"
        cam_dir = hawor_dir / "cam_space"

        cam_dir.mkdir(parents=True)
        (cam_dir / "1").mkdir()
        detector = HandDetector(str(hawor_dir))
        result = detector.detect()
        _assert(result.handedness == Handedness.RIGHT, f"应为 RIGHT, 实际为 {result.handedness}")
        _assert(result.detection_method == "cam_space_only", f"方法应为 cam_space_only, 实际为 {result.detection_method}")

        (cam_dir / "0").mkdir()
        detector2 = HandDetector(str(hawor_dir))
        result2 = detector2.detect()
        _assert(result2.handedness == Handedness.BOTH, f"应为 BOTH, 实际为 {result2.handedness}")


def test_detect_from_cam_space_left_only():
    print("\n[测试] 从 cam_space 目录检测 — 仅左手")
    with tempfile.TemporaryDirectory() as tmpdir:
        hawor_dir = Path(tmpdir) / "hawor"
        cam_dir = hawor_dir / "cam_space"
        cam_dir.mkdir(parents=True)
        (cam_dir / "0").mkdir()

        detector = HandDetector(str(hawor_dir))
        result = detector.detect()
        _assert(result.handedness == Handedness.LEFT, f"应为 LEFT, 实际为 {result.handedness}")


def test_description():
    print("\n[测试] HandDetectionResult.description")
    result = HandDetectionResult(
        handedness=Handedness.BOTH,
        left_valid_frames=80,
        right_valid_frames=80,
        total_frames=100,
        left_ratio=0.8,
        right_ratio=0.8,
        detection_method="test",
    )
    desc = result.description
    _assert("双手" in desc, f"描述应包含 '双手', 实际为: {desc}")
    _assert("80/100" in desc, f"描述应包含 '80/100', 实际为: {desc}")


# ──────────────────────────────────────────────
# Part 2: RobotArmMapper 测试
# ──────────────────────────────────────────────

def test_mapper_right():
    print("\n[测试] RobotArmMapper — 右手映射")
    mapper = RobotArmMapper()
    config = mapper.get_config(Handedness.RIGHT)

    _assert(config.handedness == Handedness.RIGHT, f"handedness 应为 RIGHT")
    _assert(len(config.arms) == 1, f"应有 1 个臂配置, 实际有 {len(config.arms)}")
    _assert(config.right_arm is not None, "right_arm 不应为 None")
    _assert(config.left_arm is None, "left_arm 应为 None")
    _assert(config.is_dual_arm is False, "is_dual_arm 应为 False")

    arm = config.right_arm
    _assert(arm.arm_prefix == "right", f"arm_prefix 应为 'right', 实际为 '{arm.arm_prefix}'")
    _assert(arm.arm_base_link_name == "right_arm_base_link", f"base_link 应为 'right_arm_base_link'")
    _assert(len(arm.arm_joint_names) == 6, f"应有 6 个臂关节, 实际有 {len(arm.arm_joint_names)}")
    _assert(arm.arm_joint_names[0] == "right_arm_joint1", f"第一个关节应为 'right_arm_joint1'")
    _assert(arm.ee_link_name == "right_gripper_link", f"末端执行器应为 'right_gripper_link'")
    _assert(arm.mano_side == "right", f"mano_side 应为 'right'")
    _assert(arm.hand_idx == 1, f"hand_idx 应为 1 (右手)")


def test_mapper_left():
    print("\n[测试] RobotArmMapper — 左手映射")
    mapper = RobotArmMapper()
    config = mapper.get_config(Handedness.LEFT)

    _assert(config.handedness == Handedness.LEFT, f"handedness 应为 LEFT")
    _assert(len(config.arms) == 1, f"应有 1 个臂配置, 实际有 {len(config.arms)}")
    _assert(config.left_arm is not None, "left_arm 不应为 None")
    _assert(config.right_arm is None, "right_arm 应为 None")

    arm = config.left_arm
    _assert(arm.arm_prefix == "left", f"arm_prefix 应为 'left', 实际为 '{arm.arm_prefix}'")
    _assert(arm.arm_base_link_name == "left_arm_base_link", f"base_link 应为 'left_arm_base_link'")
    _assert(len(arm.arm_joint_names) == 6, f"应有 6 个臂关节, 实际有 {len(arm.arm_joint_names)}")
    _assert(arm.arm_joint_names[0] == "left_arm_joint1", f"第一个关节应为 'left_arm_joint1'")
    _assert(arm.ee_link_name == "left_gripper_link", f"末端执行器应为 'left_gripper_link'")
    _assert(arm.mano_side == "left", f"mano_side 应为 'left'")
    _assert(arm.hand_idx == 0, f"hand_idx 应为 0 (左手)")


def test_mapper_both():
    print("\n[测试] RobotArmMapper — 双手映射")
    mapper = RobotArmMapper()
    config = mapper.get_config(Handedness.BOTH)

    _assert(config.handedness == Handedness.BOTH, f"handedness 应为 BOTH")
    _assert(len(config.arms) == 2, f"应有 2 个臂配置, 实际有 {len(config.arms)}")
    _assert(config.left_arm is not None, "left_arm 不应为 None")
    _assert(config.right_arm is not None, "right_arm 不应为 None")
    _assert(config.is_dual_arm is True, "is_dual_arm 应为 True")

    left = config.left_arm
    right = config.right_arm
    _assert(left.arm_prefix == "left", f"左臂前缀应为 'left'")
    _assert(right.arm_prefix == "right", f"右臂前缀应为 'right'")

    _assert(left.arm_joint_names != right.arm_joint_names, "左右臂关节名不应相同")
    _assert(left.gripper_joint_names != right.gripper_joint_names, "左右夹爪关节名不应相同")


def test_mapper_none():
    print("\n[测试] RobotArmMapper — 无手部数据")
    mapper = RobotArmMapper()
    config = mapper.get_config(Handedness.NONE)

    _assert(config.handedness == Handedness.NONE, f"handedness 应为 NONE")
    _assert(len(config.arms) == 0, f"应有 0 个臂配置, 实际有 {len(config.arms)}")
    _assert(config.left_arm is None, "left_arm 应为 None")
    _assert(config.right_arm is None, "right_arm 应为 None")


def test_get_config_for_hand_idx():
    print("\n[测试] RobotArmMapper.get_config_for_hand_idx")
    mapper = RobotArmMapper()

    left = mapper.get_config_for_hand_idx(0)
    _assert(left.arm_prefix == "left", f"hand_idx=0 应映射到左臂, 实际为 '{left.arm_prefix}'")

    right = mapper.get_config_for_hand_idx(1)
    _assert(right.arm_prefix == "right", f"hand_idx=1 应映射到右臂, 实际为 '{right.arm_prefix}'")

    try:
        mapper.get_config_for_hand_idx(2)
        _assert(False, "hand_idx=2 应抛出 ValueError")
    except ValueError:
        _assert(True, "hand_idx=2 正确抛出 ValueError")


def test_urdf_paths_exist():
    print("\n[测试] URDF 路径存在性")
    _assert(FLOATING_RIGHT_URDF.exists(), f"右臂 URDF 不存在: {FLOATING_RIGHT_URDF}")
    _assert(FLOATING_LEFT_URDF.exists(), f"左臂 URDF 不存在: {FLOATING_LEFT_URDF}")
    if DUAL_ARM_URDF.exists():
        _assert(True, f"双臂 URDF 存在: {DUAL_ARM_URDF}")
    else:
        print(f"  ⚠️  双臂 URDF 不存在: {DUAL_ARM_URDF} (非致命)")


def test_settings_paths_exist():
    print("\n[测试] IK Settings 路径存在性")
    _assert(R1_RIGHT_SETTINGS.exists(), f"右臂 settings 不存在: {R1_RIGHT_SETTINGS}")
    if R1_LEFT_SETTINGS.exists():
        _assert(True, f"左臂 settings 存在: {R1_LEFT_SETTINGS}")
    else:
        print(f"  ⚠️  左臂 settings 不存在: {R1_LEFT_SETTINGS} (非致命)")


def test_operator2mano():
    print("\n[测试] OPERATOR2MANO 矩阵")
    _assert(OPERATOR2MANO_RIGHT.shape == (3, 3), f"右臂矩阵形状应为 (3,3), 实际为 {OPERATOR2MANO_RIGHT.shape}")
    _assert(OPERATOR2MANO_LEFT.shape == (3, 3), f"左臂矩阵形状应为 (3,3), 实际为 {OPERATOR2MANO_LEFT.shape}")

    det_r = np.linalg.det(OPERATOR2MANO_RIGHT)
    det_l = np.linalg.det(OPERATOR2MANO_LEFT)
    _assert(abs(det_r - 1.0) < 1e-6, f"右臂矩阵行列式应为 1, 实际为 {det_r}")
    _assert(abs(det_l - 1.0) < 1e-6, f"左臂矩阵行列式应为 1, 实际为 {det_l}")


def test_arm_starting_qpos():
    print("\n[测试] 臂初始关节角")
    _assert(len(RIGHT_ARM_STARTING) == 6, f"右臂初始关节角应有 6 个, 实际有 {len(RIGHT_ARM_STARTING)}")
    _assert(len(LEFT_ARM_STARTING) == 6, f"左臂初始关节角应有 6 个, 实际有 {len(LEFT_ARM_STARTING)}")

    _assert(RIGHT_ARM_STARTING[0] < 0, f"右臂 joint1 应为负值 (向右), 实际为 {RIGHT_ARM_STARTING[0]}")
    _assert(LEFT_ARM_STARTING[0] > 0, f"左臂 joint1 应为正值 (向左), 实际为 {LEFT_ARM_STARTING[0]}")


def test_prepare_arm_urdf():
    print("\n[测试] prepare_arm_urdf — URDF 预处理")
    if not FLOATING_RIGHT_URDF.exists():
        print("  ⚠️  跳过 (右臂 URDF 不存在)")
        return

    temp_path = prepare_arm_urdf(FLOATING_RIGHT_URDF, "right")
    _assert(Path(temp_path).exists(), f"临时 URDF 文件应存在: {temp_path}")

    content = Path(temp_path).read_text()
    _assert("prismatic" in content, "gripper_finger_joint 应被替换为 prismatic 类型")
    _assert(str(R1_MESH_DIR) in content, f"mesh 路径应被替换为绝对路径")

    Path(temp_path).unlink()
    Path(temp_path).parent.rmdir()


def test_prepare_dual_arm_urdf():
    print("\n[测试] prepare_dual_arm_urdf — 双臂 URDF 预处理")
    if not DUAL_ARM_URDF.exists():
        print(f"  ⚠️  跳过 (双臂 URDF 不存在: {DUAL_ARM_URDF})")
        return

    temp_path = prepare_dual_arm_urdf(DUAL_ARM_URDF)
    _assert(Path(temp_path).exists(), f"临时双臂 URDF 文件应存在: {temp_path}")

    content = Path(temp_path).read_text()
    _assert("prismatic" in content, "gripper_finger_joint 应被替换为 prismatic 类型")
    _assert("left_gripper_finger_joint1" in content, "应包含左夹爪关节")
    _assert("right_gripper_finger_joint1" in content, "应包含右夹爪关节")

    Path(temp_path).unlink()
    Path(temp_path).parent.rmdir()


# ──────────────────────────────────────────────
# Part 3: 端到端集成测试
# ──────────────────────────────────────────────

def test_end_to_end_right():
    print("\n[测试] 端到端 — 右手数据 → 右臂配置")
    with tempfile.TemporaryDirectory() as tmpdir:
        hawor_dir = Path(tmpdir) / "hawor"
        rec_dir = hawor_dir / "reconstruction"
        rec_dir.mkdir(parents=True)

        pred_valid = np.zeros((2, 100), dtype=bool)
        pred_valid[1, :] = True
        np.savez(str(rec_dir / "hawor_results_001.npz"), pred_valid=pred_valid)

        detector = HandDetector(str(hawor_dir))
        result = detector.detect()

        _assert(result.handedness == Handedness.RIGHT, f"检测应为 RIGHT, 实际为 {result.handedness}")

        mapper = RobotArmMapper()
        config = mapper.get_config(result.handedness)

        _assert(config.right_arm is not None, "应生成右臂配置")
        _assert(config.right_arm.arm_prefix == "right", f"臂前缀应为 'right'")
        _assert(config.right_arm.hand_idx == 1, f"hand_idx 应为 1")


def test_end_to_end_left():
    print("\n[测试] 端到端 — 左手数据 → 左臂配置")
    with tempfile.TemporaryDirectory() as tmpdir:
        hawor_dir = Path(tmpdir) / "hawor"
        rec_dir = hawor_dir / "reconstruction"
        rec_dir.mkdir(parents=True)

        pred_valid = np.zeros((2, 100), dtype=bool)
        pred_valid[0, :] = True
        np.savez(str(rec_dir / "hawor_results_001.npz"), pred_valid=pred_valid)

        detector = HandDetector(str(hawor_dir))
        result = detector.detect()

        _assert(result.handedness == Handedness.LEFT, f"检测应为 LEFT, 实际为 {result.handedness}")

        mapper = RobotArmMapper()
        config = mapper.get_config(result.handedness)

        _assert(config.left_arm is not None, "应生成左臂配置")
        _assert(config.left_arm.arm_prefix == "left", f"臂前缀应为 'left'")
        _assert(config.left_arm.hand_idx == 0, f"hand_idx 应为 0")


def test_end_to_end_both():
    print("\n[测试] 端到端 — 双手数据 → 双臂配置")
    with tempfile.TemporaryDirectory() as tmpdir:
        hawor_dir = Path(tmpdir) / "hawor"
        rec_dir = hawor_dir / "reconstruction"
        rec_dir.mkdir(parents=True)

        pred_valid = np.ones((2, 100), dtype=bool)
        np.savez(str(rec_dir / "hawor_results_001.npz"), pred_valid=pred_valid)

        detector = HandDetector(str(hawor_dir))
        result = detector.detect()

        _assert(result.handedness == Handedness.BOTH, f"检测应为 BOTH, 实际为 {result.handedness}")

        mapper = RobotArmMapper()
        config = mapper.get_config(result.handedness)

        _assert(len(config.arms) == 2, f"应有 2 个臂配置, 实际有 {len(config.arms)}")
        _assert(config.left_arm is not None, "应生成左臂配置")
        _assert(config.right_arm is not None, "应生成右臂配置")

        for arm in config.arms:
            matched = config.get_arm_for_hand_idx(arm.hand_idx)
            _assert(matched is not None, f"get_arm_for_hand_idx({arm.hand_idx}) 不应返回 None")
            _assert(matched.arm_prefix == arm.arm_prefix, f"hand_idx={arm.hand_idx} 映射的臂前缀应为 '{arm.arm_prefix}'")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("手部检测与机械臂映射测试")
    print("=" * 60)

    print("\n" + "─" * 40)
    print("Part 1: HandDetector — 手部类型检测")
    print("─" * 40)
    test_pred_valid_2d_right_only()
    test_pred_valid_2d_left_only()
    test_pred_valid_2d_both()
    test_pred_valid_2d_sparse()
    test_pred_valid_2d_none()
    test_pred_valid_1d()
    test_detect_from_npz()
    test_detect_from_npz_no_pred_valid()
    test_detect_from_cam_space()
    test_detect_from_cam_space_left_only()
    test_description()

    print("\n" + "─" * 40)
    print("Part 2: RobotArmMapper — 机械臂配置映射")
    print("─" * 40)
    test_mapper_right()
    test_mapper_left()
    test_mapper_both()
    test_mapper_none()
    test_get_config_for_hand_idx()
    test_urdf_paths_exist()
    test_settings_paths_exist()
    test_operator2mano()
    test_arm_starting_qpos()
    test_prepare_arm_urdf()
    test_prepare_dual_arm_urdf()

    print("\n" + "─" * 40)
    print("Part 3: 端到端集成测试")
    print("─" * 40)
    test_end_to_end_right()
    test_end_to_end_left()
    test_end_to_end_both()

    print("\n" + "=" * 60)
    print(f"测试完成: ✅ {PASSED} 通过, ❌ {FAILED} 失败")
    print("=" * 60)

    if FAILED > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
