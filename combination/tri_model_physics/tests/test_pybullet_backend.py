"""测试: PyBullet 后端"""

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
TRI_MODEL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(TRI_MODEL_DIR))
sys.path.insert(0, str(TRI_MODEL_DIR.parent.parent / "example" / "position_retargeting"))

try:
    import pybullet
    _PYBULLET_AVAILABLE = True
except ImportError:
    _PYBULLET_AVAILABLE = False


@pytest.mark.skipif(not _PYBULLET_AVAILABLE, reason="PyBullet 未安装")
class TestPyBulletEnv:
    """测试 PyBullet 环境"""

    def test_create_scene(self):
        from pybullet_backend.pybullet_env import PyBulletEnv
        env = PyBulletEnv("floating_arm", "right", headless=True)
        env.build()
        assert env.physics_client is not None
        assert env.robot_id is not None
        env.disconnect()

    def test_load_floating_arm(self):
        from pybullet_backend.pybullet_env import PyBulletEnv
        env = PyBulletEnv("floating_arm", "right", headless=True)
        env.build()
        assert len(env.arm_joint_indices) == 6
        assert len(env.gripper_joint_indices) == 2
        env.disconnect()

    def test_load_gripper_only(self):
        from pybullet_backend.pybullet_env import PyBulletEnv
        env = PyBulletEnv("gripper_only", "right", headless=True)
        env.build()
        assert len(env.arm_joint_indices) == 0
        assert len(env.gripper_joint_indices) == 2
        env.disconnect()

    def test_step_physics(self):
        from pybullet_backend.pybullet_env import PyBulletEnv
        env = PyBulletEnv("floating_arm", "right", headless=True)
        env.build()
        env.step_physics()
        qpos = env.get_qpos()
        assert qpos is not None
        env.disconnect()

    def test_pd_control(self):
        from pybullet_backend.pybullet_env import PyBulletEnv
        env = PyBulletEnv("floating_arm", "right", headless=True)
        env.build()
        target = {idx: 0.0 for idx in env.arm_joint_indices}
        env.step_physics(target)
        qpos = env.get_qpos()
        assert qpos is not None
        env.disconnect()


@pytest.mark.skipif(not _PYBULLET_AVAILABLE, reason="PyBullet 未安装")
class TestPyBulletRunner:
    """测试 PyBullet 执行器构建"""

    def test_runner_build(self):
        from pybullet_backend.pybullet_runner import PyBulletRunner
        runner = PyBulletRunner("floating_arm", "right", headless=True)
        runner.build()
        assert runner.env.robot_id is not None
        runner.disconnect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
