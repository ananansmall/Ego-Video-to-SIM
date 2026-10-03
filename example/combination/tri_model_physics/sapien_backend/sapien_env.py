"""SAPIEN 物理仿真环境 — 三形式通用

核心功能:
  - 创建 SAPIEN 场景 (含物理引擎)
  - 加载三种形式的 URDF 模型
  - PD 驱动控制 (stiffness=1000, damping=200, 与 GalaxeaManipSim 一致)
  - GLB 物体加载 (大→kinematic, 小→dynamic)
  - 物理步进 (decimation=8)
"""

import logging
import os
import tempfile
from pathlib import Path

import numpy as np

from physics_utils import (
    PHYSICS_TIMESTEP, CONTROL_FREQ, DECIMATION,
    JOINT_STIFFNESS, JOINT_DAMPING, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
    GROUND_HEIGHT, OBJECT_DENSITY, GRIPPER_FRICTION, GRIPPER_INIT_OPEN,
    RIGHT_ARM_STARTING, RXWORLD_TO_SAPIEN, R1_ASSETS, R1_MESH_DIR,
)
from models.robot_forms import (
    get_robot_form_info, get_init_qpos, RobotFormInfo,
    prepare_floating_arm_urdf, prepare_gripper_only_urdf,
)

logger = logging.getLogger(__name__)


def setup_sapien_scene(headless=True):
    """创建 SAPIEN 物理场景

    优先尝试带渲染的场景；若 Vulkan 渲染设备不可用（如无头环境缺少 DRI 权限），
    则自动降级为纯物理场景（无 RenderSystem），此时无法录制视频但物理跟踪正常。

    Args:
        headless: 是否无头模式

    Returns:
        sapien.Scene: 配置好的场景
    """
    import sapien

    render_available = False
    try:
        from sapien.asset import create_dome_envmap
        # 设置 shader (必须在创建 Scene 之前)
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        scene = sapien.Scene()  # 默认含 RenderSystem
        render_available = True
    except RuntimeError as e:
        if "rendering device" in str(e).lower():
            logger.warning("  SAPIEN 渲染设备不可用，降级为纯物理场景（无视频录制）")
            scene = sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
        else:
            raise

    scene.set_timestep(PHYSICS_TIMESTEP)

    # 渲染相关配置仅在渲染可用时设置
    if render_available:
        try:
            from sapien.asset import create_dome_envmap
            scene.set_environment_map(
                create_dome_envmap(sky_color=[0.4, 0.4, 0.45], ground_color=[0.35, 0.35, 0.35])
            )
            scene.add_directional_light([1, -1, -1], [0.8, 0.8, 0.8], shadow=True)
            scene.add_directional_light([-1, -0.5, -1], [0.4, 0.4, 0.4], shadow=False)
            scene.add_directional_light([0, 1, -0.5], [0.3, 0.3, 0.3], shadow=False)
            scene.set_ambient_light([0.2, 0.2, 0.2])
        except Exception:
            pass

    # 物理地面 — 对齐 GalaxeaManipSim base.py: scene.add_ground(0)
    # 注: 之前用 add_plane_collision(GROUND_HEIGHT=-0.5) 导致物体掉到 z=-6.3 (地面失效).
    # GalaxeaManipSim 用 scene.add_ground(0) 创建 z=0 的地面, 物体在 z=0.05 以上可正常放置.
    physical_material = sapien.physx.PhysxMaterial(
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=0.0,
    )
    ground = scene.add_ground(0, material=physical_material)

    # 标记渲染是否可用（供 runner 判断是否录制视频）
    scene._render_available = render_available
    return scene


class SapienEnv:
    """SAPIEN 物理仿真环境 — 支持三种机器人形式"""

    def __init__(self, form_name="floating_arm", side="right", headless=True):
        """
        Args:
            form_name: "full_robot" / "floating_arm" / "gripper_only"
            side: "right" 或 "left"
            headless: 是否无头模式 (不渲染)
        """
        self.form_name = form_name
        self.side = side
        self.headless = headless
        self.scene = None
        self.robot = None
        self.form_info = None
        self.arm_joint_indices = []
        self.gripper_joint_indices = []
        self.joint_names = []
        self._init_qpos = None

    def build(self):
        """构建场景和机器人"""
        self.scene = setup_sapien_scene()
        self.render_available = getattr(self.scene, "_render_available", False)
        self.form_info = get_robot_form_info(self.form_name, self.side)
        self._load_robot()
        return self

    @staticmethod
    def _prepare_sapien_urdf(urdf_path, arm_prefix="right"):
        """准备 URDF: 替换 mesh 路径 + 修改关节类型 (对齐 02_render_scene.py)

        1. 将 package://r1_v2_1_0/meshes/ 替换为绝对路径
        2. 将 right/left_arm_joint1-6 从 fixed 改为 revolute (full_robot URDF 中臂关节是 fixed)
        3. 将 gripper_finger_joint1/2 从 fixed 改为 prismatic (使夹爪可以开合)

        注: 原始 URDF 已含 axis 和 limit, 只需改 type.
        """
        import re
        xml = Path(urdf_path).read_text()
        # 1. 替换 package:// 路径 (对齐 02_render_scene.py)
        xml = xml.replace("package://r1_v2_1_0/meshes/", str(R1_MESH_DIR) + "/")

        # 2. 修改臂关节类型 fixed → revolute (full_robot URDF 中臂关节是 fixed)
        # 原始 URDF 已含 axis 和 limit, 只需改 type
        for prefix in ["right", "left"]:
            for jn in range(1, 7):
                xml = re.sub(
                    rf'(<joint\s+name="{prefix}_arm_joint{jn}"\s+type=")fixed(")',
                    r'\1revolute\2', xml
                )

        # 3. 修改夹爪关节类型 fixed → prismatic (对齐 02_render_scene.py)
        xml = re.sub(
            rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint1"\s+type=")fixed(")',
            r'\1prismatic\2', xml
        )
        xml = re.sub(
            rf'(<joint\s+name="{arm_prefix}_gripper_finger_joint2"\s+type=")fixed(")',
            r'\1prismatic\2', xml
        )

        tmp_dir = tempfile.mkdtemp(prefix="r1_sapien_")
        tmp_path = os.path.join(tmp_dir, os.path.basename(urdf_path))
        with open(tmp_path, 'w') as f:
            f.write(xml)
        return tmp_path

    def _load_robot(self):
        """根据形式加载机器人"""
        import sapien

        urdf_path = self.form_info.urdf_path

        # 统一 URDF 准备: 替换 mesh 路径 + 修改夹爪关节类型 (对齐 02_render_scene.py)
        # gripper_only 已由模板生成 prismatic 关节, 但仍需替换 mesh 路径
        # floating_arm/full_robot 需要将 fixed → prismatic
        urdf_path = self._prepare_sapien_urdf(urdf_path, arm_prefix=self.side)

        loader = self.scene.create_urdf_loader()

        if self.form_name == "floating_arm":
            loader.fix_root_link = True
        elif self.form_name == "gripper_only":
            loader.fix_root_link = True
        elif self.form_name == "full_robot":
            loader.fix_root_link = True

        loader.load_multiple_collisions_from_file = True

        self.robot = loader.load(urdf_path)

        # 提取关节索引
        self.joint_names = [j.name for j in self.robot.get_active_joints()]
        self.arm_joint_indices = [
            i for i, n in enumerate(self.joint_names)
            if f"{self.side}_arm_joint" in n
        ]
        self.gripper_joint_indices = [
            i for i, n in enumerate(self.joint_names)
            if f"{self.side}_gripper_finger_joint" in n
        ]

        # 设置驱动参数
        for joint in self.robot.get_active_joints():
            if "gripper_finger" in joint.name:
                joint.set_drive_property(stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING)
            else:
                joint.set_drive_property(stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING)

        # 设置初始关节角
        init_qpos = self.robot.get_qpos().copy()
        qpos_dict = get_init_qpos(self.form_name, self.side)
        for name, val in qpos_dict.items():
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                init_qpos[idx] = val
        self.robot.set_qpos(init_qpos)
        self._init_qpos = init_qpos.copy()

        # 设置夹爪摩擦
        gripper_material = sapien.physx.PhysxMaterial(
            static_friction=GRIPPER_FRICTION,
            dynamic_friction=GRIPPER_FRICTION,
            restitution=0.0,
        )
        for link in self.robot.get_links():
            if "gripper_finger" in link.get_name():
                for shape in link.get_collision_shapes():
                    shape.set_physical_material(gripper_material)

        # 使用 step_physics 而非 scene.step, 确保重力补偿 (对齐 GalaxeaManipSim)
        # 直接 scene.step() 无重力补偿会导致 stiffness=1000 的机器人在重力下坍塌
        self.step_physics(self._init_qpos)
        self.scene.update_render()
        logger.info(f"  SAPIEN: {self.form_name} 已加载 ({len(self.arm_joint_indices)}臂关节 + {len(self.gripper_joint_indices)}夹爪关节)")

    def step_physics(self, target_qpos=None, kinematic_arm=False):
        """执行一个控制步 (decimation 次物理子步)

        对齐 GalaxeaManipSim: 每个子步都计算重力补偿力 (compute_passive_force)
        并通过 set_qf 施加, 否则 stiffness=1000 的 PD 控制无法支撑机器人自重,
        机器人会在重力下坍塌, 无法跟随轨迹.

        Args:
            target_qpos: 目标关节角 (用于PD驱动), None则保持当前
            kinematic_arm: 若 True, 臂关节运动学设置 (set_qpos), 仅夹爪用PD驱动.
                           用于回放模式, 保证位置精确对应.
        """
        if target_qpos is not None:
            if kinematic_arm and len(self.arm_joint_indices) > 0:
                # 回放模式: 臂关节运动学设置 (精确匹配目标), 夹爪用PD驱动
                cur = self.robot.get_qpos().copy()
                for j, idx in enumerate(self.arm_joint_indices):
                    if j < len(target_qpos):
                        cur[idx] = float(target_qpos[idx])
                self.robot.set_qpos(cur)
                # 仅夹爪设置 drive_target
                for idx in self.gripper_joint_indices:
                    if idx < len(target_qpos):
                        self.robot.get_active_joints()[idx].set_drive_target(float(target_qpos[idx]))
            else:
                # 全PD驱动
                for i, val in enumerate(target_qpos):
                    if i < len(self.robot.get_active_joints()):
                        self.robot.get_active_joints()[i].set_drive_target(float(val))

        for _ in range(DECIMATION):
            # 重力补偿 (对齐 GalaxeaManipSim bimanual_manipulation.py L130-135)
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self.scene.step()

    def get_qpos(self):
        """获取当前关节角"""
        return np.array(self.robot.get_qpos())

    def set_qpos(self, qpos):
        """直接设置关节角 (运动学模式)"""
        self.robot.set_qpos(qpos)

    def get_link_pose(self, link_name):
        """获取指定link的位姿"""
        for link in self.robot.get_links():
            if link.get_name() == link_name:
                pose = link.get_entity_pose()
                return np.array(pose.p), np.array(pose.q)
        return None, None

    def set_root_pose(self, pos, quat):
        """设置机器人根位姿"""
        import sapien
        pos = np.asarray(pos, dtype=np.float64)
        quat = np.asarray(quat, dtype=np.float64)
        self.robot.set_root_pose(sapien.Pose(pos.tolist(), quat.tolist()))

    def get_contacts(self, actor=None):
        """获取接触信息

        SAPIEN 3.0 API: 通过 scene.get_contacts() 获取所有接触,
        然后过滤包含指定 actor 的接触

        Args:
            actor: 可选, 指定 actor 进行过滤。None 返回全部接触

        Returns:
            list: 接触列表
        """
        all_contacts = self.scene.get_contacts()
        if actor is None:
            return all_contacts
        # 过滤包含该 actor 的接触
        result = []
        for contact in all_contacts:
            try:
                if actor in contact.actors:
                    result.append(contact)
            except Exception:
                pass
        return result
