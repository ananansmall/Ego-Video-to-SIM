"""MuJoCo 物理仿真环境 — 三形式通用

核心功能:
  - URDF → MJCF 转换 (通过 mujoco 自带 URDF 加载器)
  - PD 驱动控制 (actuator)
  - GLB 物体加载 (转换为 MJCF mesh)
  - 物理步进
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

# 无头环境下 MuJoCo 渲染需要 OSMesa；如果用户未指定后端，默认使用 OSMesa。
if os.environ.get("MUJOCO_GL") is None:
    os.environ["MUJOCO_GL"] = "osmesa"

from physics_utils import (
    PHYSICS_TIMESTEP, CONTROL_FREQ, DECIMATION, GRAVITY,
    JOINT_STIFFNESS, JOINT_DAMPING, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
    GROUND_HEIGHT, OBJECT_DENSITY, GRIPPER_FRICTION, GRIPPER_INIT_OPEN,
    RIGHT_ARM_STARTING, RXWORLD_TO_SAPIEN, R1_MESH_DIR,
)
from models.robot_forms import get_robot_form_info, get_init_qpos

logger = logging.getLogger(__name__)


def _make_mjcf_from_urdf(urdf_path, side="right"):
    """将 URDF 转换为 MuJoCo 可加载的格式

    MuJoCo 加载 URDF 时, mesh 路径相对于 URDF 文件位置解析。
    策略: 将 URDF 复制到 mesh 目录中, 这样相对路径就能找到 mesh。
    """
    import re
    import shutil

    with open(urdf_path, 'r') as f:
        content = f.read()

    # 替换 package:// 路径为相对文件名
    content = re.sub(
        r'filename="package://[^/]+/meshes/([^"]+)"',
        r'filename="\1"',
        content,
    )

    # 将绝对路径转为相对文件名 (MuJoCo URDF 加载器对绝对路径支持不佳)
    content = re.sub(
        r'filename="[^"]*/([^/]+\.STL)"',
        r'filename="\1"',
        content,
    )
    content = re.sub(
        r'filename="[^"]*/([^/]+\.glb)"',
        r'filename="\1"',
        content,
    )

    # 替换 arm_joint fixed → revolute (full_robot URDF 中臂关节是 fixed, 已含 axis 和 limit)
    for prefix in ["right", "left"]:
        for jn in range(1, 7):
            content = re.sub(
                rf'(<joint\s+name="{prefix}_arm_joint{jn}"\s+type=")fixed(")',
                r'\1revolute\2', content
            )

    # 替换 gripper_finger_joint fixed → prismatic
    def replace_gripper_joint(match):
        joint_block = match.group(0)
        joint_name = re.search(r'name="([^"]+)"', joint_block)
        if joint_name and 'gripper_finger_joint' in joint_name.group(1):
            joint_block = joint_block.replace('type="fixed"', 'type="prismatic"')
            if '<axis' not in joint_block:
                axis_xyz = '0 -1 0' if 'joint1' in joint_name.group(1) else '0 1 0'
                joint_block = joint_block.replace(
                    '</joint>',
                    f'  <axis xyz="{axis_xyz}"/>\n'
                    f'  <limit lower="0" upper="0.05" effort="100" velocity="0.25"/>\n'
                    f'</joint>'
                )
            elif '<limit' not in joint_block:
                joint_block = joint_block.replace(
                    '</joint>',
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

    # 将 URDF 写入 mesh 目录中 (这样相对 mesh 路径能正确解析)
    tmp_dir = tempfile.mkdtemp(prefix="r1_mjcf_", dir=str(R1_MESH_DIR))
    tmp_urdf = os.path.join(tmp_dir, os.path.basename(urdf_path))
    with open(tmp_urdf, 'w') as f:
        f.write(content)

    # 创建到 mesh 文件的相对 symlinks (mesh 文件在上级目录)
    for mesh_file in R1_MESH_DIR.glob("*.STL"):
        link_path = os.path.join(tmp_dir, mesh_file.name)
        if not os.path.exists(link_path):
            os.symlink(f"../{mesh_file.name}", link_path)

    return tmp_urdf, tmp_dir


class MuJoCoEnv:
    """MuJoCo 物理仿真环境 — 支持三种机器人形式"""

    def __init__(self, form_name="floating_arm", side="right", headless=True):
        self.form_name = form_name
        self.side = side
        self.headless = headless
        self.model = None
        self.data = None
        self.form_info = None
        self.tmp_dir = None

        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        self.joint_names = []
        self.num_joints = 0

    def build(self):
        """构建场景和机器人"""
        import mujoco

        self.form_info = get_robot_form_info(self.form_name, self.side)
        urdf_path = self.form_info.urdf_path

        # 转换 URDF
        mjcf_urdf, tmp_dir = _make_mjcf_from_urdf(urdf_path, self.side)
        self.tmp_dir = tmp_dir

        try:
            # MuJoCo 直接加载 URDF
            model = mujoco.MjModel.from_xml_path(mjcf_urdf)
            data = mujoco.MjData(model)

            # 设置重力 — 与 physics_utils.GRAVITY 保持一致 (z-up 坐标系下为 -9.81)
            model.opt.gravity[:] = GRAVITY
            logger.info(f"  MuJoCo 重力已设置为: {model.opt.gravity.tolist()}")
            # 设置时间步
            model.opt.timestep = PHYSICS_TIMESTEP

            self.model = model
            self.data = data

            # 提取关节信息
            self.num_joints = model.njnt
            self.joint_names = []
            self.arm_joint_indices = []
            self.gripper_joint_indices = []

            for i in range(self.num_joints):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name is None:
                    continue
                self.joint_names.append(name)

                if name.startswith(f"{self.side}_arm_joint"):
                    self.arm_joint_indices.append(i)
                elif name.startswith(f"{self.side}_gripper_finger_joint"):
                    self.gripper_joint_indices.append(i)

            # 添加 PD 驱动器
            self._add_pd_actuators()

            # 设置初始关节角
            self._set_initial_qpos()

            logger.info(f"  MuJoCo: {self.form_name} 已加载 ({self.num_joints} joints, "
                        f"{len(self.arm_joint_indices)}臂 + {len(self.gripper_joint_indices)}夹爪)")

        except Exception as e:
            logger.error(f"  MuJoCo 加载失败: {e}")
            raise

        return self

    def _add_pd_actuators(self):
        """为关节添加 PD 驱动器

        MuJoCo 直接力矩控制 (qfrc_applied) 需要谨慎处理:
          - 力矩必须裁剪, 否则大位置误差会产生过大力矩导致仿真爆炸
          - 需要重力补偿 (qfrc_bias) 以克服重力
          - 增益需要适配 MuJoCo 的积分方式
        """
        import mujoco

        # MuJoCo 增益: 比 SAPIEN/PyBullet 小, 因为 MuJoCo 积分方式不同
        # 高增益(1000)在 MuJoCo 中容易导致 NaN, 使用更保守的值
        self._kp_arm = 100.0       # 臂关节刚度 (N·m/rad)
        self._kd_arm = 20.0        # 臂关节阻尼
        self._kp_gripper = 200.0   # 夹爪刚度 (更高以保持夹紧)
        self._kd_gripper = 40.0    # 夹爪阻尼

        # 力矩限制 (N·m) — 防止仿真爆炸
        self._torque_limit_arm = 50.0
        self._torque_limit_gripper = 20.0

        # 启用重力补偿: qfrc_bias 包含重力+科氏力+离心力
        self._use_gravity_comp = True

    def _set_initial_qpos(self):
        """设置初始关节角"""
        import mujoco
        qpos_dict = get_init_qpos(self.form_name, self.side)
        for name, val in qpos_dict.items():
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                self.data.qpos[idx] = val
        mujoco.mj_forward(self.model, self.data)

    def step_physics(self, target_qpos=None, kinematic_arm=False):
        """执行一个控制步 (decimation 次物理子步)

        PD 力矩控制 + 重力补偿:
          τ = Kp×(q_target - q_current) - Kd×q̇ + τ_gravity
          其中 τ_gravity = qfrc_bias (MuJoCo 自动计算的重力+科氏力补偿)

        Args:
            target_qpos: 目标关节角 dict {joint_index: angle}
            kinematic_arm: 若 True, 臂关节运动学设置 (data.qpos), 仅夹爪用PD驱动.
                           用于回放模式, 保证位置精确对应.
        """
        import mujoco

        if target_qpos is not None:
            if kinematic_arm and len(self.arm_joint_indices) > 0:
                # 回放模式: 臂关节运动学设置 (精确匹配目标)
                arm_targets = {idx: float(target_qpos[idx]) for idx in self.arm_joint_indices
                               if idx in target_qpos}
                for idx, val in arm_targets.items():
                    self.data.qpos[idx] = val
                    self.data.qvel[idx] = 0.0
                # 仅夹爪用 PD 力矩控制
                grip_targets = {idx: v for idx, v in target_qpos.items()
                                if idx in self.gripper_joint_indices}
            else:
                arm_targets = None
                grip_targets = target_qpos

            if grip_targets:
                # 计算重力补偿 (qfrc_bias 包含 C(q,q̇)q̇ + g(q))
                mujoco.mj_forward(self.model, self.data)
                gravity_comp = self.data.qfrc_bias.copy()

                # PD 力矩控制 + 重力补偿
                for idx, target_val in grip_targets.items():
                    if idx >= self.num_joints:
                        continue
                    if kinematic_arm and idx not in self.gripper_joint_indices:
                        continue
                    current_val = self.data.qpos[idx]
                    current_vel = self.data.qvel[idx]

                    if idx in self.arm_joint_indices:
                        kp, kd = self._kp_arm, self._kd_arm
                        torque_limit = self._torque_limit_arm
                    else:
                        kp, kd = self._kp_gripper, self._kd_gripper
                        torque_limit = self._torque_limit_gripper

                    pd_torque = kp * (target_val - current_val) - kd * current_vel
                    if self._use_gravity_comp:
                        total_torque = pd_torque + gravity_comp[idx]
                    else:
                        total_torque = pd_torque
                    total_torque = np.clip(total_torque, -torque_limit, torque_limit)
                    self.data.qfrc_applied[idx] = total_torque

        for _ in range(DECIMATION):
            mujoco.mj_step(self.model, self.data)
            # 回放模式: 每个子步后重置臂关节, 防止重力下垂
            if kinematic_arm and arm_targets:
                for idx, val in arm_targets.items():
                    self.data.qpos[idx] = val
                    self.data.qvel[idx] = 0.0

    def get_qpos(self):
        """获取当前关节角"""
        return np.array(self.data.qpos[:self.num_joints])

    def set_qpos(self, qpos):
        """直接设置关节角"""
        self.data.qpos[:len(qpos)] = qpos

    def get_link_pose(self, link_name):
        """获取指定link的位姿"""
        import mujoco
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id < 0:
            return None, None
        pos = self.data.xpos[body_id].copy()
        quat = self.data.xquat[body_id].copy()  # wxyz
        return pos, quat

    def set_root_pose(self, pos, quat):
        """设置机器人根位姿"""
        # MuJoCo 中根位姿通过 qpos[0:3] (位置) 和 qpos[3:7] (四元数) 设置
        # 但对于 fixed base 机器人, 需要修改 model
        if self.model.nq >= 7:
            self.data.qpos[0:3] = pos
            self.data.qpos[3:7] = quat

    def get_contacts(self):
        """获取接触信息"""
        import mujoco
        contacts = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            contacts.append(con)
        return contacts

    def disconnect(self):
        """清理"""
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            import shutil
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
