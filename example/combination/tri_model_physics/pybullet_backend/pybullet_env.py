"""PyBullet 物理仿真环境 — 三形式通用

核心功能:
  - 创建 PyBullet 场景 (含物理引擎)
  - 加载三种形式的 URDF 模型 (GLB→OBJ转换)
  - PD 驱动控制 (Kp=1000, Kd=200, 与 SAPIEN 一致)
  - GLB 物体加载 (大→static, 小→dynamic)
  - 物理步进
"""

import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from physics_utils import (
    PHYSICS_TIMESTEP, CONTROL_FREQ, DECIMATION, GRAVITY,
    PD_KP_ARM, PD_KD_ARM, PD_KP_GRIPPER, PD_KD_GRIPPER,
    GROUND_HEIGHT, OBJECT_DENSITY, GRIPPER_FRICTION, GRIPPER_INIT_OPEN,
    RIGHT_ARM_STARTING, LEFT_ARM_STARTING, GRIPPER_MAX_OPEN,
)
from models.robot_forms import get_robot_form_info, get_init_qpos, RobotFormInfo

logger = logging.getLogger(__name__)


def _make_pybullet_compatible_urdf(urdf_path):
    """将 URDF 转换为 PyBullet 兼容格式 (GLB→OBJ, package://→相对路径)

    复用 r1_simulation.py 的 _make_pybullet_compatible_urdf 逻辑

    Args:
        urdf_path: 原始 URDF 路径

    Returns:
        (str, str): (临时URDF路径, 临时目录路径)
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("需要 trimesh: pip install trimesh")

    urdf_path = Path(urdf_path)
    with open(urdf_path, 'r') as f:
        content = f.read()

    tmp_dir = tempfile.mkdtemp(prefix="r1_pb_")

    # 创建mesh目录链接 — 使用R1_MESH_DIR (URDF的parent/meshes可能不存在)
    from physics_utils import R1_MESH_DIR
    link_path = os.path.join(tmp_dir, "meshes")
    if not os.path.exists(link_path):
        os.symlink(str(R1_MESH_DIR), link_path)

    # 创建 OBJ 转换目录
    obj_dir = os.path.join(tmp_dir, "meshes_obj")
    os.makedirs(obj_dir, exist_ok=True)

    converted = set()

    def replace_glb_with_obj(match):
        filename = match.group(1)
        clean = re.sub(r'^\.\/', '', filename)
        if filename.endswith('.usd.sapien.glb') or filename.endswith('.glb'):
            base = re.sub(r'\.usd\.sapien\.glb$', '', clean)
            base = re.sub(r'\.glb$', '', base)
            mesh_name = os.path.basename(base) + ".obj"

            if mesh_name not in converted:
                converted.add(mesh_name)
                glb_full = R1_MESH_DIR / clean.replace('meshes/', '', 1) if clean.startswith('meshes/') else R1_MESH_DIR / clean
                obj_full = os.path.join(obj_dir, mesh_name)
                if glb_full.exists() and not os.path.exists(obj_full):
                    try:
                        scene = trimesh.load(str(glb_full))
                        mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
                        mesh.export(obj_full)
                    except Exception as e:
                        logger.warning(f"  GLB→OBJ转换失败 {filename}: {e}")
                        return match.group(0)

            return f'filename="meshes_obj/{mesh_name}"'
        return match.group(0)

    def replace_collision_stl(match):
        filename = match.group(1)
        clean = re.sub(r'^\.\/', '', filename)
        clean = clean.replace('meshes/', '', 1) if clean.startswith('meshes/') else clean
        if clean.endswith('_collision.STL'):
            base_stl = clean.replace('_collision.STL', '.STL')
            stl_full = R1_MESH_DIR / base_stl
            if stl_full.exists():
                return f'filename="meshes/{base_stl}"'
        return match.group(0)

    def replace_package(match):
        filename = match.group(1)
        # filename = "r1_v2_1_0/meshes/xxx.STL" (package://已被正则去掉)
        # → "meshes/xxx.STL"
        clean = re.sub(r'^[^/]+/', '', filename)
        return f'filename="{clean}"'

    content = re.sub(
        r'filename="([^"]*\.(?:usd\.sapien\.)?glb)"',
        replace_glb_with_obj, content
    )
    content = re.sub(
        r'filename="([^"]*_collision\.STL)"',
        replace_collision_stl, content
    )
    content = re.sub(
        r'filename="package://([^"]+)"',
        replace_package, content
    )

    tmp_urdf = os.path.join(tmp_dir, urdf_path.name)
    with open(tmp_urdf, 'w') as f:
        f.write(content)

    return tmp_urdf, tmp_dir


class PyBulletEnv:
    """PyBullet 物理仿真环境 — 支持三种机器人形式"""

    def __init__(self, form_name="floating_arm", side="right", headless=True):
        self.form_name = form_name
        self.side = side
        self.headless = headless
        self.physics_client = None
        self.robot_id = None
        self.form_info = None
        self.tmp_dir = None

        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        self.joint_names = []
        self.num_joints = 0
        # 非 fixed 关节索引 (用于 calculateInverseDynamics, 该函数要求数组长度=DOF数)
        self._dof_joint_indices = []

    def build(self):
        """构建场景和机器人"""
        import pybullet as p
        import pybullet_data

        if self.headless:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)

        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(*GRAVITY, physicsClientId=self.physics_client)
        p.setTimeStep(PHYSICS_TIMESTEP, physicsClientId=self.physics_client)

        # 地面
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        self.form_info = get_robot_form_info(self.form_name, self.side)
        self._load_robot()

        return self

    def _load_robot(self):
        """加载机器人 URDF"""
        import pybullet as p

        urdf_path = self.form_info.urdf_path

        # 对于 floating_arm/full_robot: 原始URDF中gripper_finger_joint是fixed类型
        # 需要替换为prismatic类型以实现夹爪驱动
        if self.form_name in ("floating_arm", "full_robot"):
            urdf_path = self._make_prismatic_gripper_urdf(urdf_path)

        pb_urdf, tmp_dir = _make_pybullet_compatible_urdf(urdf_path)
        self.tmp_dir = tmp_dir

        self.robot_id = p.loadURDF(
            pb_urdf,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.physics_client,
        )

        # 提取关节信息
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        self.joint_names = []
        self.arm_joint_indices = []
        self.gripper_joint_indices = []

        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            name = info[1].decode()
            self.joint_names.append(name)

            if name.startswith(f"{self.side}_arm_joint") and info[2] == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(i)
            elif name.startswith(f"{self.side}_gripper_finger_joint"):
                self.gripper_joint_indices.append(i)

            # 收集非 fixed 关节 (calculateInverseDynamics 需要)
            if info[2] != p.JOINT_FIXED:
                self._dof_joint_indices.append(i)

        # 排序
        def sort_key(idx):
            nums = re.findall(r'\d+', self.joint_names[idx])
            return int(nums[-1]) if nums else 0

        self.arm_joint_indices.sort(key=sort_key)
        self.gripper_joint_indices.sort(key=sort_key)

        # 设置初始关节角
        qpos_dict = get_init_qpos(self.form_name, self.side)
        for name, val in qpos_dict.items():
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                p.resetJointState(self.robot_id, idx, val, physicsClientId=self.physics_client)

        # 禁用默认电机
        for idx in self.arm_joint_indices + self.gripper_joint_indices:
            p.setJointMotorControl2(
                self.robot_id, idx, p.VELOCITY_CONTROL, force=0,
                physicsClientId=self.physics_client,
            )

        logger.info(f"  PyBullet: {self.form_name} 已加载 ({self.num_joints} joints, "
                     f"{len(self.arm_joint_indices)}臂 + {len(self.gripper_joint_indices)}夹爪)")

    def step_physics(self, target_qpos=None, kinematic_arm=False):
        """执行一个控制步 (decimation 次物理子步)

        PD 力矩控制 + 重力补偿 (对齐 MuJoCo/SAPIEN):
          τ = Kp×(q_target - q_current) - Kd×q̇ + τ_gravity
          其中 τ_gravity = p.calculateInverseDynamics(q, q̇, 0) (重力+科氏力补偿)

        Args:
            target_qpos: 目标关节角 dict {joint_index: angle}
            kinematic_arm: 若 True, 臂关节运动学设置 (resetJointState), 仅夹爪用PD驱动.
                           用于回放模式, 保证位置精确对应.
        """
        import pybullet as p

        if target_qpos is not None:
            if kinematic_arm and len(self.arm_joint_indices) > 0:
                # 回放模式: 臂关节运动学设置 (精确匹配目标)
                arm_targets = {idx: float(target_qpos[idx]) for idx in self.arm_joint_indices
                               if idx in target_qpos}
                for idx, val in arm_targets.items():
                    p.resetJointState(
                        self.robot_id, idx, val,
                        physicsClientId=self.physics_client,
                    )
                # 仅夹爪用 PD 力矩控制
                grip_targets = {idx: v for idx, v in target_qpos.items()
                                if idx in self.gripper_joint_indices}
            else:
                arm_targets = None
                grip_targets = target_qpos

            if grip_targets:
                # 收集当前关节状态 (用于 PD 与重力补偿计算)
                dof_indices = self._dof_joint_indices
                num_dof = len(dof_indices)
                cur_qpos_dof = [0.0] * num_dof
                cur_qvel_dof = [0.0] * num_dof
                cur_qpos = np.zeros(self.num_joints)
                cur_qvel = np.zeros(self.num_joints)
                for d, ji in enumerate(dof_indices):
                    state = p.getJointState(self.robot_id, ji, physicsClientId=self.physics_client)
                    cur_qpos_dof[d] = state[0]
                    cur_qvel_dof[d] = state[1]
                    cur_qpos[ji] = state[0]
                    cur_qvel[ji] = state[1]

                # 重力补偿
                zero_accel = [0.0] * num_dof
                gravity_comp_dof = p.calculateInverseDynamics(
                    self.robot_id, cur_qpos_dof, cur_qvel_dof,
                    zero_accel, physicsClientId=self.physics_client,
                )
                gravity_comp = np.zeros(self.num_joints)
                for d, ji in enumerate(dof_indices):
                    gravity_comp[ji] = gravity_comp_dof[d]

                # PD 力矩控制 + 重力补偿 (仅夹爪, 或全关节非回放模式)
                for idx, val in grip_targets.items():
                    if idx >= self.num_joints:
                        continue
                    if kinematic_arm and idx not in self.gripper_joint_indices:
                        continue
                    if idx in self.arm_joint_indices:
                        kp, kd = PD_KP_ARM, PD_KD_ARM
                    elif idx in self.gripper_joint_indices:
                        kp, kd = PD_KP_GRIPPER, PD_KD_GRIPPER
                    else:
                        continue
                    pd_torque = kp * (float(val) - cur_qpos[idx]) - kd * cur_qvel[idx]
                    total_torque = pd_torque + gravity_comp[idx]
                    p.setJointMotorControl2(
                        self.robot_id, idx, p.TORQUE_CONTROL,
                        force=float(total_torque),
                        physicsClientId=self.physics_client,
                    )

        for _ in range(DECIMATION):
            p.stepSimulation(physicsClientId=self.physics_client)
            # 回放模式: 每个子步后重置臂关节, 防止重力下垂
            if kinematic_arm and arm_targets:
                for idx, val in arm_targets.items():
                    p.resetJointState(
                        self.robot_id, idx, val,
                        physicsClientId=self.physics_client,
                    )
            # 夹爪硬限位
            for idx in self.gripper_joint_indices:
                state = p.getJointState(self.robot_id, idx, physicsClientId=self.physics_client)
                pos = state[0]
                if pos < 0.0 or pos > GRIPPER_MAX_OPEN:
                    clamped = max(0.0, min(GRIPPER_MAX_OPEN, pos))
                    p.resetJointState(
                        self.robot_id, idx, clamped,
                        physicsClientId=self.physics_client,
                    )

    def get_qpos(self):
        """获取当前关节角"""
        import pybullet as p
        qpos = []
        for i in range(self.num_joints):
            state = p.getJointState(self.robot_id, i, physicsClientId=self.physics_client)
            qpos.append(state[0])
        return np.array(qpos)

    def get_link_pose(self, link_name_or_index):
        """获取指定link的位姿

        Args:
            link_name_or_index: link名称(str)或关节索引(int)

        Returns:
            (pos, quat_wxyz) 或 (None, None)
        """
        import pybullet as p

        link_index = link_name_or_index
        if isinstance(link_name_or_index, str):
            # 通过名称查找link索引
            link_index = -1
            for i in range(self.num_joints):
                info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                name = info[12].decode() if info[12] else info[1].decode()
                if name == link_name_or_index:
                    link_index = i
                    break
            if link_index < 0:
                return None, None

        if link_index < 0 or link_index >= self.num_joints:
            return None, None
        state = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True,
                               physicsClientId=self.physics_client)
        pos = np.array(state[4])
        quat_xyzw = np.array(state[5])
        # 转换为 wxyz
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, quat_wxyz

    def set_base_position(self, pos):
        """设置机器人基座位置"""
        import pybullet as p
        # PyBullet 不支持直接移动 fixed base 机器人
        # 需要重置: 先移除再重新加载
        logger.warning("PyBullet fixed-base 机器人无法直接移动基座, 请在加载前设置位置")

    def set_root_pose(self, pos, quat_wxyz):
        """设置机器人根位姿 (用于 gripper_only 形式跟踪)

        PyBullet 四元数格式为 [x, y, z, w], 需从 pytransform3d 的 [w, x, y, z] 转换.
        对 fixed-base 机器人使用 resetBasePositionAndOrientation 重定位.

        Args:
            pos: 根位置 (3,)
            quat_wxyz: 根姿态四元数 [w, x, y, z]
        """
        import pybullet as p
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        # [w, x, y, z] → [x, y, z, w]
        quat_xyzw = [
            float(quat_wxyz[1]), float(quat_wxyz[2]),
            float(quat_wxyz[3]), float(quat_wxyz[0]),
        ]
        p.resetBasePositionAndOrientation(
            self.robot_id, pos.tolist(), quat_xyzw,
            physicsClientId=self.physics_client,
        )

    def get_contacts(self, body_id):
        """获取指定body的接触信息"""
        import pybullet as p
        return p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=body_id,
            physicsClientId=self.physics_client,
        )

    @staticmethod
    def _make_prismatic_gripper_urdf(urdf_path):
        """将URDF中gripper_finger_joint从fixed替换为prismatic, arm_joint从fixed替换为revolute

        full_robot URDF 中 right/left_arm_joint1-6 是 fixed 类型, 需改为 revolute 才能驱动.
        """
        import re
        with open(urdf_path, 'r') as f:
            content = f.read()

        # 1. 臂关节 fixed → revolute (full_robot URDF 中臂关节是 fixed, 已含 axis 和 limit)
        for prefix in ["right", "left"]:
            for jn in range(1, 7):
                content = re.sub(
                    rf'(<joint\s+name="{prefix}_arm_joint{jn}"\s+type=")fixed(")',
                    r'\1revolute\2', content
                )

        # 2. 夹爪关节 fixed → prismatic
        def replace_gripper_joint(match):
            joint_block = match.group(0)
            joint_name = re.search(r'name="([^"]+)"', joint_block)
            if joint_name and 'gripper_finger_joint' in joint_name.group(1):
                joint_block = joint_block.replace('type="fixed"', 'type="prismatic"')
                axis_xyz = '0 -1 0' if 'joint1' in joint_name.group(1) else '0 1 0'
                joint_block = joint_block.replace(
                    '</joint>',
                    f'  <axis xyz="{axis_xyz}"/>\n'
                    f'  <limit lower="0" upper="0.05" effort="100" velocity="0.25"/>\n'
                    f'</joint>'
                )
            return joint_block

        content = re.sub(
            r'<joint[^>]*name="[^"]*gripper_finger_joint[^"]*"[^>]*>.*?</joint>',
            replace_gripper_joint,
            content,
            flags=re.DOTALL,
        )

        tmp_dir = tempfile.mkdtemp(prefix="r1_prismatic_")
        tmp_path = os.path.join(tmp_dir, os.path.basename(urdf_path))
        with open(tmp_path, 'w') as f:
            f.write(content)
        return tmp_path

    def disconnect(self):
        """断开连接"""
        import pybullet as p
        if self.physics_client is not None:
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = None
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
