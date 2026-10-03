"""测试: 夹取GLB物体"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
TRI_MODEL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRI_MODEL_DIR))

from grasp_controller import GraspController, GraspState
from physics_utils import GRASP_CONTACT_THRESHOLD, GRASP_STABLE_FRAMES


class TestGraspState:
    """测试抓取状态"""

    def test_initial_state(self):
        state = GraspState()
        assert state.is_grasping is False
        assert state.contact_count == 0
        assert state.stable_frames == 0


class TestGraspController:
    """测试夹取控制器"""

    def test_init(self):
        ctrl = GraspController(
            gripper_joint_names=["right_gripper_finger_joint1", "right_gripper_finger_joint2"],
            finger_link_names=["right_gripper_finger_link1", "right_gripper_finger_link2"],
        )
        assert len(ctrl.gripper_joint_names) == 2

    def test_no_contact_no_grasp(self):
        ctrl = GraspController()
        is_grasping = ctrl.update_grasp_state(contact_count=0)
        assert is_grasping is False

    def test_sustained_contact_triggers_grasp(self):
        ctrl = GraspController()
        # 持续接触 GRASP_STABLE_FRAMES 帧
        for _ in range(GRASP_STABLE_FRAMES):
            is_grasping = ctrl.update_grasp_state(contact_count=GRASP_CONTACT_THRESHOLD + 1)
        assert is_grasping is True

    def test_contact_lost_resets_grasp(self):
        ctrl = GraspController()
        # 先建立抓取
        for _ in range(GRASP_STABLE_FRAMES):
            ctrl.update_grasp_state(contact_count=GRASP_CONTACT_THRESHOLD + 1)
        assert ctrl.state.is_grasping is True

        # 失去接触
        for _ in range(GRASP_STABLE_FRAMES + 5):
            ctrl.update_grasp_state(contact_count=0)
        assert ctrl.state.is_grasping is False

    def test_compute_gripper_target(self):
        ctrl = GraspController(
            gripper_joint_names=["right_gripper_finger_joint1", "right_gripper_finger_joint2"],
        )
        targets = ctrl.compute_gripper_target(0.02)
        assert "right_gripper_finger_joint1" in targets
        assert "right_gripper_finger_joint2" in targets

    def test_gripper_target_clamped(self):
        ctrl = GraspController(
            gripper_joint_names=["right_gripper_finger_joint1", "right_gripper_finger_joint2"],
        )
        # 超出范围
        targets = ctrl.compute_gripper_target(1.0)
        for val in targets.values():
            assert val <= 0.05

    def test_reset(self):
        ctrl = GraspController()
        for _ in range(GRASP_STABLE_FRAMES):
            ctrl.update_grasp_state(contact_count=5)
        assert ctrl.state.is_grasping is True

        ctrl.reset()
        assert ctrl.state.is_grasping is False
        assert ctrl.state.stable_frames == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
