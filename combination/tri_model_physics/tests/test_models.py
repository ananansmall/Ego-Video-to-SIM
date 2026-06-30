"""测试: 三种机器人形式 URDF 加载与结构验证"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 添加路径
SCRIPT_DIR = Path(__file__).resolve().parent
TRI_MODEL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRI_MODEL_DIR))
sys.path.insert(0, str(TRI_MODEL_DIR.parent.parent / "example" / "position_retargeting"))

from models.robot_forms import (
    get_robot_form_info, get_init_qpos, ALL_FORMS,
    prepare_full_robot_urdf, prepare_floating_arm_urdf, prepare_gripper_only_urdf,
    _parse_urdf_joints,
)
from physics_utils import (
    FULL_ROBOT_URDF, FLOATING_ARM_RIGHT_URDF, R1_MESH_DIR,
    RIGHT_ARM_STARTING, GRIPPER_INIT_OPEN,
)


class TestURDFPaths:
    """验证 URDF 文件路径存在"""

    def test_full_robot_urdf_exists(self):
        assert FULL_ROBOT_URDF.exists(), f"完整机器人URDF不存在: {FULL_ROBOT_URDF}"

    def test_floating_arm_right_urdf_exists(self):
        assert FLOATING_ARM_RIGHT_URDF.exists(), f"浮动臂URDF不存在: {FLOATING_ARM_RIGHT_URDF}"

    def test_mesh_dir_exists(self):
        assert R1_MESH_DIR.exists(), f"Mesh目录不存在: {R1_MESH_DIR}"


class TestRobotFormInfo:
    """验证三种形式的 RobotFormInfo 结构"""

    def test_full_robot_info(self):
        info = get_robot_form_info("full_robot", "right")
        assert info.name == "full_robot"
        assert info.has_arm is True
        assert info.has_gripper is True
        assert info.has_base is True
        assert info.is_floating is False
        assert len(info.arm_joint_names) > 0, "full_robot 应有臂关节"
        assert len(info.gripper_joint_names) > 0, "full_robot 应有夹爪关节"

    def test_floating_arm_info(self):
        info = get_robot_form_info("floating_arm", "right")
        assert info.name == "floating_arm"
        assert info.has_arm is True
        assert info.has_gripper is True
        assert info.is_floating is True
        assert len(info.arm_joint_names) == 6, f"浮动臂应有6个臂关节, 实际: {len(info.arm_joint_names)}"
        assert len(info.gripper_joint_names) == 2, f"浮动臂应有2个夹爪关节, 实际: {len(info.gripper_joint_names)}"

    def test_gripper_only_info(self):
        info = get_robot_form_info("gripper_only", "right")
        assert info.name == "gripper_only"
        assert info.has_arm is False
        assert info.has_gripper is True
        assert len(info.arm_joint_names) == 0, "纯夹爪不应有臂关节"
        assert len(info.gripper_joint_names) == 2, f"纯夹爪应有2个夹爪关节, 实际: {len(info.gripper_joint_names)}"

    def test_left_side(self):
        info = get_robot_form_info("floating_arm", "left")
        for name in info.arm_joint_names:
            assert "left" in name, f"左侧臂关节应包含'left': {name}"

    def test_unknown_form_raises(self):
        with pytest.raises(ValueError, match="未知机器人形式"):
            get_robot_form_info("unknown_form")


class TestInitQpos:
    """验证初始关节角"""

    def test_floating_arm_init_qpos(self):
        qpos = get_init_qpos("floating_arm", "right")
        assert len(qpos) > 0
        # 臂关节初始值应与 RIGHT_ARM_STARTING 一致
        arm_joints = [k for k in qpos if "arm_joint" in k]
        assert len(arm_joints) == 6

    def test_gripper_only_init_qpos(self):
        qpos = get_init_qpos("gripper_only", "right")
        assert len(qpos) == 2
        for name, val in qpos.items():
            assert 0 <= val <= 0.05, f"夹爪关节值应在[0, 0.05]: {name}={val}"


class TestURDFParsing:
    """验证 URDF 解析"""

    def test_parse_floating_arm_joints(self):
        joints = _parse_urdf_joints(str(FLOATING_ARM_RIGHT_URDF))
        assert len(joints) > 0
        joint_names = [j['name'] for j in joints]
        assert any("right_arm_joint" in n for n in joint_names)
        assert any("right_gripper_finger_joint" in n for n in joint_names)

    def test_parse_gripper_only_urdf(self):
        urdf_path = prepare_gripper_only_urdf("right")
        joints = _parse_urdf_joints(urdf_path)
        joint_names = [j['name'] for j in joints]
        assert any("gripper_finger_joint" in n for n in joint_names)
        # 纯夹爪不应有臂关节
        assert not any("arm_joint" in n for n in joint_names)


class TestGripperURDFGeneration:
    """验证纯夹爪 URDF 生成"""

    def test_generate_right_gripper(self):
        path = prepare_gripper_only_urdf("right")
        assert Path(path).exists()
        with open(path, 'r') as f:
            content = f.read()
        assert "right_gripper_link" in content
        assert "right_gripper_finger_joint1" in content

    def test_generate_left_gripper(self):
        path = prepare_gripper_only_urdf("left")
        assert Path(path).exists()
        with open(path, 'r') as f:
            content = f.read()
        assert "left_gripper_link" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
