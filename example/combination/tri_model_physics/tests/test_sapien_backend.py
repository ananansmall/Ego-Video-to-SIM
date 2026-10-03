"""测试: SAPIEN 后端"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
TRI_MODEL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRI_MODEL_DIR))
sys.path.insert(0, str(TRI_MODEL_DIR.parent.parent / "example" / "position_retargeting"))

try:
    import sapien
    _SAPIEN_AVAILABLE = True
except ImportError:
    _SAPIEN_AVAILABLE = False


@pytest.mark.skipif(not _SAPIEN_AVAILABLE, reason="SAPIEN 未安装")
class TestSapienEnv:
    """测试 SAPIEN 环境"""

    def test_create_scene(self):
        from sapien_backend.sapien_env import setup_sapien_scene
        scene = setup_sapien_scene()
        assert scene is not None

    def test_load_floating_arm(self):
        from sapien_backend.sapien_env import SapienEnv
        env = SapienEnv("floating_arm", "right", headless=True)
        env.build()
        assert env.robot is not None
        assert len(env.arm_joint_indices) == 6
        assert len(env.gripper_joint_indices) == 2

    def test_load_gripper_only(self):
        from sapien_backend.sapien_env import SapienEnv
        env = SapienEnv("gripper_only", "right", headless=True)
        env.build()
        assert env.robot is not None
        assert len(env.arm_joint_indices) == 0
        assert len(env.gripper_joint_indices) == 2

    def test_step_physics(self):
        from sapien_backend.sapien_env import SapienEnv
        env = SapienEnv("floating_arm", "right", headless=True)
        env.build()
        env.step_physics()
        qpos = env.get_qpos()
        assert qpos is not None

    def test_set_qpos(self):
        from sapien_backend.sapien_env import SapienEnv
        env = SapienEnv("floating_arm", "right", headless=True)
        env.build()
        target = env.get_qpos().copy()
        target[0] = 0.5
        env.set_qpos(target)
        result = env.get_qpos()
        assert abs(result[0] - 0.5) < 0.01


@pytest.mark.skipif(not _SAPIEN_AVAILABLE, reason="SAPIEN 未安装")
class TestSapienRunner:
    """测试 SAPIEN 执行器构建"""

    def test_runner_build(self):
        from sapien_backend.sapien_runner import SapienRunner
        runner = SapienRunner("floating_arm", "right", headless=True)
        runner.build()
        assert runner.env.robot is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
