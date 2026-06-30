"""测试: 轨迹加载与解析映射"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
TRI_MODEL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRI_MODEL_DIR))
sys.path.insert(0, str(TRI_MODEL_DIR.parent.parent / "example" / "position_retargeting"))

from physics_utils import RXWORLD_TO_SAPIEN, _FINGER1_ORIGIN, _FINGER2_ORIGIN, FINGER_BASE_DIST
from trajectory_loader import compute_analytical_gripper_pose


class TestAnalyticalGripperPose:
    """测试解析夹爪位姿计算"""

    def test_basic_pose(self):
        """基本测试: 已知手指位置应产生合理的夹爪位姿"""
        wrist = np.array([0.0, 0.0, 0.0])
        finger1 = np.array([0.1, -0.02, 0.0])
        finger2 = np.array([0.1, 0.02, 0.0])

        root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(wrist, finger1, finger2)

        assert root_pos.shape == (3,)
        assert root_R.shape == (3, 3)
        # 旋转矩阵应正交
        assert np.allclose(root_R @ root_R.T, np.eye(3), atol=1e-6)
        # 手指关节值应在 [0, 0.05]
        assert 0 <= j1 <= 0.05
        assert 0 <= j2 <= 0.05

    def test_closed_gripper(self):
        """闭合夹爪: 手指距离接近 FINGER_BASE_DIST"""
        wrist = np.array([0.0, 0.0, 0.0])
        finger1 = np.array([0.1, -FINGER_BASE_DIST / 2, 0.0])
        finger2 = np.array([0.1, FINGER_BASE_DIST / 2, 0.0])

        root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(wrist, finger1, finger2)

        # 闭合时关节值应接近0
        assert j1 < 0.001
        assert j2 < 0.001

    def test_open_gripper(self):
        """张开夹爪: 手指距离远大于 FINGER_BASE_DIST"""
        wrist = np.array([0.0, 0.0, 0.0])
        finger1 = np.array([0.1, -0.05, 0.0])
        finger2 = np.array([0.1, 0.05, 0.0])

        root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(wrist, finger1, finger2)

        # 张开时关节值应 > 0
        assert j1 > 0
        assert j2 > 0

    def test_degenerate_case(self):
        """退化情况: 手指位置重合"""
        wrist = np.array([0.0, 0.0, 0.0])
        finger1 = np.array([0.1, 0.0, 0.0])
        finger2 = np.array([0.1, 0.0, 0.0])

        root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(wrist, finger1, finger2)

        # 不应崩溃
        assert root_pos is not None
        assert root_R.shape == (3, 3)


class TestCoordinateTransform:
    """测试坐标变换"""

    def test_rxworld_to_sapien_orthogonal(self):
        """RXWORLD_TO_SAPIEN 应为正交矩阵"""
        assert np.allclose(RXWORLD_TO_SAPIEN @ RXWORLD_TO_SAPIEN.T, np.eye(3), atol=1e-6)

    def test_transform_preserves_norm(self):
        """变换应保持向量长度"""
        v = np.array([1.0, 2.0, 3.0])
        v_transformed = RXWORLD_TO_SAPIEN @ v
        assert abs(np.linalg.norm(v_transformed) - np.linalg.norm(v)) < 1e-6


class TestHaWoRDataLoading:
    """测试 HaWoR 数据加载 (需要实际数据)"""

    def test_load_nonexistent_dir(self):
        """不存在的目录应抛出异常"""
        from trajectory_loader import load_hawor_data
        with pytest.raises(FileNotFoundError):
            load_hawor_data("/nonexistent/path")

    def test_load_existing_data(self):
        """如果默认数据存在, 测试加载"""
        from trajectory_loader import load_hawor_data
        default_hawor = Path("/home/an/data/hawor/7")
        if not default_hawor.exists():
            pytest.skip("默认 HaWoR 数据不存在")

        data = load_hawor_data(str(default_hawor))
        assert "pred_trans" in data
        assert "pred_rot" in data
        assert data["pred_trans"].shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
