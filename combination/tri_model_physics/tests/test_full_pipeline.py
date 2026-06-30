"""测试: 端到端集成测试 — 每种组合运行少量帧"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
TRI_MODEL_DIR = SCRIPT_DIR.parent
COMBINATION_DIR = TRI_MODEL_DIR.parent
sys.path.insert(0, str(TRI_MODEL_DIR))
sys.path.insert(0, str(COMBINATION_DIR.parent / "example" / "position_retargeting"))
sys.path.insert(0, str(Path("/home/an/robot_world_ws/src/GalaxeaManipSim")))


def find_test_data():
    """查找可用的测试数据"""
    default_hawor = Path("/home/an/data/hawor/7")
    default_ras = Path("/home/an/data/ras/my_7mp4_result")

    if not default_hawor.exists() or not default_ras.exists():
        return None, None, None

    # 查找 transform_params
    tp = default_ras / "alignment" / "transform_params.npz"
    if not tp.exists():
        # 在 combination/output 下查找
        combo_output = COMBINATION_DIR / "output"
        if combo_output.exists():
            for d in combo_output.iterdir():
                tp_candidate = d / "alignment" / "transform_params.npz"
                if tp_candidate.exists():
                    tp = tp_candidate
                    break

    if not tp.exists():
        return None, None, None

    return str(default_hawor), str(default_ras), str(tp)


class TestPyBulletIntegration:
    """PyBullet 端到端测试"""

    @pytest.fixture(scope="class")
    def test_data(self):
        hawor, ras, tp = find_test_data()
        if hawor is None:
            pytest.skip("测试数据不可用")
        return hawor, ras, tp

    def test_gripper_only_pybullet(self, test_data):
        """gripper_only × PyBullet: 运行10帧"""
        from pybullet_backend.pybullet_runner import PyBulletRunner
        hawor, ras, tp = test_data
        runner = PyBulletRunner("gripper_only", "right", headless=True)
        runner.build()
        result = runner.run_tracking(hawor, ras, tp, num_frames=10)
        assert len(result["qpos_sequence"]) == 10
        runner.disconnect()

    def test_floating_arm_pybullet(self, test_data):
        """floating_arm × PyBullet: 运行10帧"""
        from pybullet_backend.pybullet_runner import PyBulletRunner
        hawor, ras, tp = test_data
        runner = PyBulletRunner("floating_arm", "right", headless=True)
        runner.build()
        result = runner.run_tracking(hawor, ras, tp, num_frames=10)
        assert len(result["qpos_sequence"]) == 10
        runner.disconnect()


class TestSapienIntegration:
    """SAPIEN 端到端测试"""

    @pytest.fixture(scope="class")
    def test_data(self):
        hawor, ras, tp = find_test_data()
        if hawor is None:
            pytest.skip("测试数据不可用")
        return hawor, ras, tp

    @pytest.mark.skipif(
        not Path("/usr/share/vulkan/icd.d/nvidia_icd.json").exists(),
        reason="Vulkan/GPU 不可用"
    )
    def test_gripper_only_sapien(self, test_data):
        """gripper_only × SAPIEN: 运行10帧"""
        from sapien_backend.sapien_runner import SapienRunner
        hawor, ras, tp = test_data
        runner = SapienRunner("gripper_only", "right", headless=True)
        runner.build()
        result = runner.run_tracking(hawor, ras, tp, num_frames=10)
        assert len(result["qpos_sequence"]) == 10

    @pytest.mark.skipif(
        not Path("/usr/share/vulkan/icd.d/nvidia_icd.json").exists(),
        reason="Vulkan/GPU 不可用"
    )
    def test_floating_arm_sapien(self, test_data):
        """floating_arm × SAPIEN: 运行10帧"""
        from sapien_backend.sapien_runner import SapienRunner
        hawor, ras, tp = test_data
        runner = SapienRunner("floating_arm", "right", headless=True)
        runner.build()
        result = runner.run_tracking(hawor, ras, tp, num_frames=10)
        assert len(result["qpos_sequence"]) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
