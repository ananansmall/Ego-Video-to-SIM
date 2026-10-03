"""
第一人称视频 -> 夹爪替换管道

完整流程：
  1. 从第一人称视频中提取手部姿态 (MediaPipe Hands)
  2. 将人手关节映射到夹爪关节 (dex_retargeting, SeqRetargeting)
  3. 接触保持滤波 + 轨迹平滑
  4. 在 SAPIEN 中以与视频相同的相机视角渲染夹爪
     - MANO 骨架可视化（球体+胶囊体，5指5色）
     - 坐标轴可视化（RGB=XYZ，手腕+末端执行器）
     - 轨迹拖尾（黄色=MANO，蓝色=夹爪末端）
  5. 将夹爪叠加到原始视频帧上，替换人手区域
  6. 生成多种可视化输出视频

输出视频说明：
  - gripper_replaced.mp4: 夹爪替换后的视频（人手被夹爪替换）
  - sapien_render.mp4: 纯 SAPIEN 3D 渲染视频（含 MANO 骨架、坐标轴、轨迹）
  - overlay.mp4: 原始视频 + 手部骨架标注 + 抓取状态
  - trajectory_viz.mp4: 关节角度/手腕位置/手指距离曲线可视化

使用示例：
  conda run -n dex python gripper_video_pipeline.py \\
    --video-path /path/to/video.mp4 \\
    --output-dir ./output_gripper \\
    --robot panda

  # 使用 R1 夹爪
  conda run -n dex python gripper_video_pipeline.py \\
    --video-path /path/to/video.mp4 \\
    --output-dir ./output_r1 \\
    --robot r1

  # 无 SAPIEN 时使用软件渲染回退
  conda run -n dex python gripper_video_pipeline.py \\
    --video-path /path/to/video.mp4 \\
    --output-dir ./output_soft \\
    --robot panda \\
    --software-render
"""

import os
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
_vulkan_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
if os.path.exists(_vulkan_icd) and "VK_ICD_FILENAMES" not in os.environ:
    os.environ["VK_ICD_FILENAMES"] = _vulkan_icd

import argparse
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if not (PROJECT_ROOT / "assets" / "robots" / "hands").exists():
    _alt = Path("/home/an/robot_world_ws/src/dex-retargeting")
    if (_alt / "assets" / "robots" / "hands").exists():
        PROJECT_ROOT = _alt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "example" / "vector_retargeting"))

from dex_retargeting.constants import RobotName, RetargetingType, HandType
from dex_retargeting.retargeting_config import RetargetingConfig
from egocentric_hand_detector import EgocentricHandDetector
from arm_motion_analyzer import ArmMotionAnalyzer
from robot_hand_retargeter import RobotHandRetargeter
from contact_preservation import ContactPreservationFilter, TrajectorySmoother


OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
OPERATOR2MANO_LEFT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])

R_CAM2BASE = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)

MANO_FINGERS = [
    [0, 1, 2, 3, 4],
    [0, 5, 6, 7, 8],
    [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16],
    [0, 17, 18, 19, 20],
]


def rotation_from_z_to_dir(direction):
    """
    计算从 Z 轴到目标方向的旋转四元数 (wxyz)

    用于将胶囊体（默认沿 Z 轴）旋转到任意方向，
    实现 MANO 骨架中骨段的可视化。

    Args:
        direction: 目标方向向量 (3,)

    Returns:
        ndarray: 四元数 [w, x, y, z] (4,)
    """
    direction = np.array(direction, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    d = direction / norm
    z = np.array([0.0, 0.0, 1.0])
    dot = np.dot(z, d)
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-8:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    axis_len = np.linalg.norm(axis)
    axis = axis / axis_len
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


class GripperSAPIENRenderer:
    """
    夹爪 SAPIEN 渲染器（含 MANO 骨架、坐标轴、轨迹可视化）

    参考 egocentric_sapien_render.py 的成熟实现，在 SAPIEN 3D 场景中：
      - 加载夹爪 URDF 并根据手腕位姿设置夹爪位置/朝向
      - 渲染 MANO 手部骨架（5指5色球体+胶囊体）
      - 渲染坐标轴（RGB=XYZ，手腕+末端执行器+各 link）
      - 渲染轨迹拖尾（黄色=MANO 轨迹，蓝色=夹爪末端轨迹）
      - 动态跟随相机保持第一人称视角

    坐标变换链（与 egocentric_sapien_render.py 完全一致）：
      MediaPipe 原始手腕旋转 mediapipe_wrist_rot
      → wrist_rot_mano = mediapipe_wrist_rot @ operator2mano
      → wrist_rot_world = R_CAM2BASE @ wrist_rot_mano @ operator2mano.T
      → 设置夹爪位姿

    Args:
        robot_name: 机器人名称枚举
        hand_type: 手别枚举
        retargeting_type: 重定向类型
        img_width: 渲染图像宽度
        img_height: 渲染图像高度
        mano_scale: MANO 骨架缩放系数（默认 2.0）
        show_axes: 是否显示坐标轴
        show_trail: 是否显示轨迹拖尾
        show_mano: 是否显示 MANO 骨架
    """

    def __init__(
        self,
        robot_name: RobotName = RobotName.panda,
        hand_type: HandType = HandType.right,
        retargeting_type: RetargetingType = RetargetingType.position,
        img_width: int = 1280,
        img_height: int = 720,
        mano_scale: float = 2.0,
        show_axes: bool = True,
        show_trail: bool = True,
        show_mano: bool = True,
    ):
        self.robot_name = robot_name
        self.hand_type = hand_type
        self.retargeting_type = retargeting_type
        self.img_width = img_width
        self.img_height = img_height
        self.mano_scale = mano_scale
        self.show_axes = show_axes
        self.show_trail = show_trail
        self.show_mano = show_mano
        self.is_right = hand_type == HandType.right
        self.operator2mano = OPERATOR2MANO_RIGHT if self.is_right else OPERATOR2MANO_LEFT

        self._smooth_cam_pos = None

        self.mano_trail = []
        self.robot_ee_trail = []
        self.mano_nodes = []
        self.axis_nodes = []
        self.trail_nodes = []

        self._setup_sapien_scene()
        self._setup_robot()
        self._setup_retargeting()
        self._setup_materials()

    def _setup_sapien_scene(self):
        """
        初始化 SAPIEN 场景、光照和相机

        场景配置与 egocentric_sapien_render.py 一致：
          - 灰色背景环境贴图
          - 地面平面
          - 环境光 + 方向光
          - 相机 FOV 使用固定值 0.9（与参考代码一致）
        """
        import sapien
        from sapien.asset import create_dome_envmap

        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")

        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 240)

        self.scene.set_environment_map(
            create_dome_envmap(sky_color=[0.3, 0.3, 0.3], ground_color=[0.15, 0.15, 0.15])
        )
        self.scene.add_directional_light(
            np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True
        )
        self.scene.add_directional_light([0, 0, -1], [1.5, 1.4, 1.4], shadow=False)
        self.scene.set_ambient_light(np.array([0.3, 0.3, 0.3]))

        ground_mat = sapien.render.RenderMaterial()
        ground_mat.set_base_color(np.array([0.4, 0.4, 0.4, 1]))
        ground_mat.set_roughness(0.8)
        self.scene.add_ground(-1, render_material=ground_mat)

        self.camera = self.scene.add_camera(
            "egocentric_cam", self.img_width, self.img_height, 0.9, 0.01, 100
        )

        sapien.render.set_log_level("error")
        self.internal_scene = self.scene.render_system._internal_scene
        self.context = sapien.render.SapienRenderer()._internal_context

    def _setup_robot(self):
        """
        加载夹爪 URDF 模型到 SAPIEN 场景

        与 egocentric_sapien_render.py 一致的加载流程：
          1. 设置 URDF 搜索目录
          2. 获取默认重定向配置文件路径
          3. 加载带 GLB 视觉模型的 URDF
          4. 使用 yourdfpy 添加 6 自由度虚拟关节
          5. 将 URDF 写入临时文件后加载到 SAPIEN
        """
        import sapien
        from dex_retargeting import yourdfpy as urdf
        from dex_retargeting.constants import get_default_config_path

        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        config_path = get_default_config_path(
            self.robot_name, self.retargeting_type, self.hand_type
        )
        override = {"add_dummy_free_joint": True}
        config = RetargetingConfig.load_from_file(config_path, override=override)

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True

        urdf_path = Path(config.urdf_path)
        if "glb" not in urdf_path.stem:
            urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        robot_urdf = urdf.URDF.load(
            str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
        )
        temp_dir = tempfile.mkdtemp(prefix="gripper_sapien-")
        temp_path = f"{temp_dir}/{urdf_path.name}"
        robot_urdf.write_xml_file(temp_path)

        self.robot = loader.load(temp_path)
        self.robot.set_pose(sapien.Pose([0, 0, 0]))

        self.retargeting_config = config
        self.robot_config_path = config_path

        self.robot_active_joints = self.robot.get_active_joints()
        self.robot_links = self.robot.get_links()

        self.dummy_joint_indices = []
        self.finger_joint_indices = []
        for i, j in enumerate(self.robot_active_joints):
            if "dummy" in j.name:
                self.dummy_joint_indices.append(i)
            else:
                self.finger_joint_indices.append(i)

        for idx in self.dummy_joint_indices:
            joint = self.robot_active_joints[idx]
            drive = joint.create_drive(
                stiffness=1e5,
                damping=1e4,
                force_limit=1e5,
            )
            drive.set_target_qpos(0.0)

        for idx in self.finger_joint_indices:
            joint = self.robot_active_joints[idx]
            drive = joint.create_drive(
                stiffness=1e4,
                damping=1e3,
                force_limit=1e4,
            )
            drive.set_target_qpos(0.0)

    def _setup_retargeting(self):
        """
        初始化重定向优化器和关节映射

        与 egocentric_sapien_render.py 一致：
          1. 从配置构建 SeqRetargeting 优化器
          2. 初始化固定关节值（含手臂关节默认值）
          3. 计算重定向关节到 SAPIEN 关节的索引映射
        """
        import sapien
        from dex_retargeting.seq_retarget import SeqRetargeting

        self.retargeting: SeqRetargeting = self.retargeting_config.build()

        opt = self.retargeting.optimizer
        num_fixed = len(opt.idx_pin2fixed)
        self.fixed_qpos = np.zeros(num_fixed, dtype=np.float32)

        fixed_names = opt.fixed_joint_names
        for i, name in enumerate(fixed_names):
            if "right_arm" in name:
                joint_idx = int(name.replace("right_arm_joint", "").replace("_joint", "")) - 1
                defaults = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
                if joint_idx < len(defaults):
                    self.fixed_qpos[i] = defaults[joint_idx]
            elif "left_arm" in name:
                joint_idx = int(name.replace("left_arm_joint", "").replace("_joint", "")) - 1
                defaults = [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]
                if joint_idx < len(defaults):
                    self.fixed_qpos[i] = defaults[joint_idx]

        sapien_joint_names = [j.name for j in self.robot_active_joints]
        self.retarget2sapien = np.array(
            [self.retargeting.joint_names.index(n) for n in sapien_joint_names]
        ).astype(int)

    def _setup_materials(self):
        """
        初始化 SAPIEN 渲染材质

        材质列表：
          - mat_mano_hand: 手腕球体（肤色）
          - mat_mano_finger: 5指5色（红/橙/绿/蓝/紫）
          - mat_axis: 坐标轴 RGB=XYZ
          - mat_trail: MANO 轨迹（黄色）
          - mat_robot_trail: 夹爪末端轨迹（蓝色）
        """
        self.mat_mano_hand = self.context.create_material(
            np.array([0.96, 0.75, 0.69, 1.0]), np.array([0.1, 0.1, 0.1, 1.0]), 0.5, 0.0, 0
        )
        self.mat_mano_finger = [
            self.context.create_material(np.array([1.0, 0.2, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([1.0, 0.6, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0.2, 1.0, 0.2, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0.2, 0.4, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0.8, 0.2, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0),
        ]
        self.mat_axis = [
            self.context.create_material(np.array([1, 0, 0, 1]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0, 1, 0, 1]), np.zeros(4), 0.5, 0.0, 0),
            self.context.create_material(np.array([0, 0, 1, 1]), np.zeros(4), 0.5, 0.0, 0),
        ]
        self.mat_trail = self.context.create_material(
            np.array([1.0, 0.8, 0.0, 0.8]), np.zeros(4), 0.5, 0.0, 0
        )
        self.mat_robot_trail = self.context.create_material(
            np.array([0.0, 0.5, 1.0, 0.8]), np.zeros(4), 0.5, 0.0, 0
        )

    def _clear_nodes(self, node_list):
        """清除 SAPIEN 场景中的可视化节点"""
        for _ in range(len(node_list)):
            node = node_list.pop()
            self.internal_scene.remove_node(node)

    def _add_sphere(self, pos, radius, mat, node_list):
        """在 SAPIEN 场景中添加球体节点（用于关节点可视化）"""
        sphere_mesh = self.context.create_uvsphere_mesh(12, 6)
        model = self.context.create_model([sphere_mesh], [mat])
        node = self.internal_scene.add_node()
        node.set_position(pos.astype(np.float32))
        node.set_scale(np.array([radius, radius, radius], dtype=np.float32))
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    def _add_capsule(self, p1, p2, radius, mat, node_list):
        """在 SAPIEN 场景中添加胶囊体节点（用于骨段/坐标轴可视化）"""
        mid = ((p1 + p2) / 2).astype(np.float32)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return
        quat = rotation_from_z_to_dir(direction).astype(np.float32)
        capsule_mesh = self.context.create_capsule_mesh(radius, length / 2, 8, 4)
        model = self.context.create_model([capsule_mesh], [mat])
        node = self.internal_scene.add_node()
        node.set_position(mid)
        node.set_rotation(quat)
        obj = self.internal_scene.add_object(model, node)
        obj.shading_mode = 0
        node_list.append(node)

    def _render_mano_skeleton(self, joint_pos, wrist_rot, origin_pos):
        """
        渲染 MANO 手部骨架

        在 SAPIEN 3D 场景中绘制 21 个关节点和 5 根手指的骨段：
          - 球体表示关节点（手腕=肤色，各指=对应颜色）
          - 胶囊体表示骨段（各指=对应颜色）

        Args:
            joint_pos: 21 个关节 3D 坐标 (21, 3)，MANO 坐标系
            wrist_rot: 手腕旋转矩阵 (3, 3)，世界坐标系
            origin_pos: 手腕位置 (3,)，世界坐标系
        """
        self._clear_nodes(self.mano_nodes)

        if joint_pos is None or wrist_rot is None:
            return

        scale = self.mano_scale
        offset = wrist_rot[:, 1] * 0.2 if wrist_rot is not None else np.array([0.0, 0.2, 0.0])
        mano_center = origin_pos + offset
        joints_world = []
        for i in range(21):
            j_world = mano_center + (wrist_rot @ (joint_pos[i] * scale)).astype(np.float64)
            joints_world.append(j_world)

        for fi, finger in enumerate(MANO_FINGERS):
            for j in range(len(finger) - 1):
                self._add_capsule(
                    joints_world[finger[j]], joints_world[finger[j + 1]],
                    0.008, self.mat_mano_finger[fi], self.mano_nodes
                )

        for i in range(21):
            mat = self.mat_mano_hand if i == 0 else self.mat_mano_finger[min(i // 4, 4)]
            self._add_sphere(joints_world[i], 0.012, mat, self.mano_nodes)

    def _render_axes(self, wrist_pos, wrist_rot, robot_ee_pos=None, robot_ee_rot=None):
        """
        渲染坐标轴

        在手腕位置和末端执行器位置绘制 RGB=XYZ 坐标轴

        Args:
            wrist_pos: 手腕位置 (3,)
            wrist_rot: 手腕旋转矩阵 (3, 3)
            robot_ee_pos: 末端执行器位置 (3,)，可选
            robot_ee_rot: 末端执行器旋转矩阵 (3, 3)，可选
        """
        from pytransform3d import rotations as pr_rot

        self._clear_nodes(self.axis_nodes)

        if not self.show_axes:
            return

        if wrist_pos is not None and wrist_rot is not None:
            axis_len = 0.25
            for ai in range(3):
                axis_end = wrist_pos + wrist_rot[:, ai] * axis_len
                self._add_capsule(wrist_pos, axis_end, 0.012, self.mat_axis[ai], self.axis_nodes)

        if robot_ee_pos is not None and robot_ee_rot is not None:
            axis_len = 0.12
            for ai in range(3):
                axis_end = robot_ee_pos + robot_ee_rot[:, ai] * axis_len
                self._add_capsule(robot_ee_pos, axis_end, 0.006, self.mat_axis[ai], self.axis_nodes)

        for link in self.robot_links:
            link_pose = link.get_entity_pose()
            link_pos = np.array(link_pose.p).astype(np.float64)
            link_quat = np.array(link_pose.q)
            link_R = pr_rot.matrix_from_quaternion(link_quat)
            for ai in range(3):
                axis_end = link_pos + link_R[:, ai] * 0.04
                self._add_capsule(link_pos, axis_end, 0.002, self.mat_axis[ai], self.axis_nodes)

    def _render_trails(self, mano_pos, robot_ee_pos):
        """
        渲染轨迹拖尾

        在 SAPIEN 场景中绘制手腕和夹爪末端的历史轨迹：
          - 黄色轨迹：MANO 手腕位置
          - 蓝色轨迹：夹爪末端执行器位置

        Args:
            mano_pos: 当前手腕位置 (3,)
            robot_ee_pos: 当前末端执行器位置 (3,)
        """
        self._clear_nodes(self.trail_nodes)

        if not self.show_trail:
            return

        if mano_pos is not None:
            self.mano_trail.append(mano_pos.copy())
            if len(self.mano_trail) > 200:
                self.mano_trail.pop(0)

        if robot_ee_pos is not None:
            self.robot_ee_trail.append(robot_ee_pos.copy())
            if len(self.robot_ee_trail) > 200:
                self.robot_ee_trail.pop(0)

        if len(self.mano_trail) > 2:
            step = max(1, len(self.mano_trail) // 80)
            for i in range(step, len(self.mano_trail), step):
                alpha = i / len(self.mano_trail)
                trail_mat = self.context.create_material(
                    np.array([1.0 * alpha, 0.8 * alpha, 0.0, 0.3 + 0.7 * alpha]),
                    np.zeros(4), 0.5, 0.0, 0
                )
                self._add_capsule(
                    self.mano_trail[i - step], self.mano_trail[i],
                    0.005, trail_mat, self.trail_nodes
                )
            self._add_sphere(self.mano_trail[-1], 0.018, self.mat_trail, self.trail_nodes)

        if len(self.robot_ee_trail) > 2:
            step = max(1, len(self.robot_ee_trail) // 80)
            for i in range(step, len(self.robot_ee_trail), step):
                alpha = i / len(self.robot_ee_trail)
                robot_trail_mat = self.context.create_material(
                    np.array([0.0, 0.5 * alpha, 1.0 * alpha, 0.3 + 0.7 * alpha]),
                    np.zeros(4), 0.5, 0.0, 0
                )
                self._add_capsule(
                    self.robot_ee_trail[i - step], self.robot_ee_trail[i],
                    0.005, robot_trail_mat, self.trail_nodes
                )
            self._add_sphere(self.robot_ee_trail[-1], 0.018, self.mat_robot_trail, self.trail_nodes)

    def update_gripper(self, wrist_pos_camera, mediapipe_wrist_rot, qpos, joint_pos=None):
        """
        更新夹爪位姿、关节角度和可视化元素

        坐标变换链（与 egocentric_sapien_render.py 完全一致）：
          1. wrist_pos_camera → R_CAM2BASE @ wrist_pos_camera → wrist_pos_base
          2. mediapipe_wrist_rot → mediapipe_wrist_rot @ operator2mano → wrist_rot_mano
          3. R_CAM2BASE @ wrist_rot_mano @ operator2mano.T → wrist_rot_world
          4. 设置夹爪位姿: Pose(wrist_pos_base, quat_from_matrix(wrist_rot_world))

        执行顺序（与 egocentric_sapien_render.py 一致）：
          1. set_qpos(robot_qpos) — 先设置关节角度
          2. set_pose(wrist_pos, wrist_quat) — 再设置基座位姿
          3. scene.step() — 物理步进
          4. set_pose(wrist_pos, wrist_quat) — 再次强制设置位姿（防止物理步进导致漂移）
          5. 更新驱动器目标值（防止重力导致下坠）

        Args:
            wrist_pos_camera: 手腕在相机坐标系下的 3D 位置 (3,)
            mediapipe_wrist_rot: MediaPipe 原始手腕旋转矩阵 (3, 3)
            qpos: 重定向后的关节角度数组
            joint_pos: 21 个关节 3D 坐标 (21, 3)，用于 MANO 骨架可视化

        Returns:
            dict: 包含手腕位置、旋转矩阵、末端执行器位姿等信息
        """
        import sapien
        from pytransform3d import rotations as pr_rot

        wrist_pos_base = R_CAM2BASE @ wrist_pos_camera

        wrist_rot_mano = mediapipe_wrist_rot @ self.operator2mano
        wrist_rot_world_mat = (R_CAM2BASE @ wrist_rot_mano @ self.operator2mano.T).astype(np.float64)

        base_quat_wxyz = pr_rot.quaternion_from_matrix(wrist_rot_world_mat)
        base_quat_wxyz = np.array([
            base_quat_wxyz[3], base_quat_wxyz[0],
            base_quat_wxyz[1], base_quat_wxyz[2]
        ])

        robot_qpos = qpos[self.retarget2sapien]

        self.robot.set_qpos(robot_qpos)

        target_pose = sapien.Pose(
            wrist_pos_base.astype(np.float32), base_quat_wxyz.astype(np.float32)
        )
        self.robot.set_pose(target_pose)

        for idx in self.dummy_joint_indices:
            self.robot_active_joints[idx].get_drive().set_target_qpos(robot_qpos[idx])
        for idx in self.finger_joint_indices:
            self.robot_active_joints[idx].get_drive().set_target_qpos(robot_qpos[idx])

        self.scene.step()

        self.robot.set_pose(target_pose)

        ee_link = self.robot_links[-1]
        ee_pose = ee_link.get_entity_pose()
        robot_ee_pos = np.array(ee_pose.p).astype(np.float64)
        ee_quat = np.array(ee_pose.q)
        robot_ee_rot = pr_rot.matrix_from_quaternion(ee_quat)

        if self.show_mano and joint_pos is not None:
            self._render_mano_skeleton(joint_pos, wrist_rot_world_mat, wrist_pos_base)
        else:
            self._clear_nodes(self.mano_nodes)

        self._render_axes(wrist_pos_base, wrist_rot_world_mat, robot_ee_pos, robot_ee_rot)
        self._render_trails(wrist_pos_base, robot_ee_pos)

        self._update_camera(wrist_pos_base)

        return {
            "wrist_pos_base": wrist_pos_base,
            "wrist_rot_world": wrist_rot_world_mat,
            "robot_ee_pos": robot_ee_pos,
            "robot_ee_rot": robot_ee_rot,
            "target_pose": target_pose,
        }

    def _update_camera(self, wrist_pos_base):
        """
        动态更新相机位姿，跟随手腕位置

        与 egocentric_sapien_render.py 完全一致的相机定位策略：
          1. 低通滤波平滑手腕位置（alpha=0.08）
          2. 相机放在手腕后方偏上位置
          3. 相机朝向手腕前方偏下位置
          4. look-at 方式计算相机旋转

        Args:
            wrist_pos_base: 手腕在 SAPIEN 世界坐标系中的位置 (3,)
        """
        import sapien

        if self._smooth_cam_pos is None:
            self._smooth_cam_pos = wrist_pos_base.copy()
        alpha_smooth = 0.08
        self._smooth_cam_pos = (1 - alpha_smooth) * self._smooth_cam_pos + alpha_smooth * wrist_pos_base

        smooth_target = self._smooth_cam_pos.copy()

        cam_pos = smooth_target + np.array([-0.30, 0.0, 0.45])
        cam_look_at = smooth_target + np.array([0.10, 0.0, -0.05])
        cam_up_hint = np.array([0, 0, 1])

        forward = cam_look_at - cam_pos
        forward_len = np.linalg.norm(forward)
        if forward_len > 1e-6:
            forward = forward / forward_len
        else:
            forward = np.array([1, 0, 0])

        right = np.cross(forward, cam_up_hint)
        right_len = np.linalg.norm(right)
        if right_len > 1e-6:
            right = right / right_len
        else:
            right = np.array([0, 1, 0])
        up = np.cross(right, forward)

        cam_R = np.eye(4)
        cam_R[:3, 0] = right
        cam_R[:3, 1] = -up
        cam_R[:3, 2] = forward
        cam_R[:3, 3] = cam_pos

        from scipy.spatial.transform import Rotation as R_scipy
        cam_quat_xyzw = R_scipy.from_matrix(cam_R[:3, :3]).as_quat()
        cam_quat_wxyz = np.array([cam_quat_xyzw[3], cam_quat_xyzw[0], cam_quat_xyzw[1], cam_quat_xyzw[2]])

        self.camera.set_local_pose(
            sapien.Pose(cam_pos.astype(np.float32), cam_quat_wxyz.astype(np.float32))
        )

    def render_frame(self, keypoint_2d=None, frame_idx=0, num_box=0, mano_origin=None):
        """
        渲染一帧 SAPIEN 图像

        Args:
            keypoint_2d: MediaPipe 2D 关键点，用于在 SAPIEN 渲染上绘制骨架
            frame_idx: 当前帧索引
            num_box: 检测到的手部数量
            mano_origin: 手腕位置，用于文字标注

        Returns:
            tuple: (sapien_bgr, gripper_bgr, gripper_mask)
        """
        self.scene.update_render()
        self.camera.take_picture()

        rgb = self.camera.get_picture("Color")[..., :3]
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        sapien_bgr = np.ascontiguousarray(rgb_uint8[..., ::-1])

        if keypoint_2d is not None and num_box > 0:
            EgocentricHandDetector.draw_skeleton_on_image(sapien_bgr, keypoint_2d, style="white")

        status = "DETECTED" if num_box > 0 else "LOST"
        status_color = (0, 255, 0) if num_box > 0 else (0, 0, 255)
        cv2.putText(sapien_bgr, f"Frame {frame_idx} | {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        if mano_origin is not None:
            cv2.putText(sapien_bgr, f"Wrist: ({mano_origin[0]:.2f}, {mano_origin[1]:.2f}, {mano_origin[2]:.2f})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        gripper_bgr = sapien_bgr.copy()
        bg_bgr = np.array([77, 77, 77], dtype=np.uint8)
        diff = np.abs(gripper_bgr.astype(np.int16) - bg_bgr.astype(np.int16))
        color_diff = np.sum(diff, axis=2)
        gripper_mask = np.where(color_diff > 25, 255, 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gripper_mask = cv2.morphologyEx(gripper_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        gripper_mask = cv2.GaussianBlur(gripper_mask, (7, 7), 0)

        return sapien_bgr, gripper_bgr, gripper_mask


def compute_hand_mask(keypoint_2d, img_h, img_w, dilation_pixels=30):
    """
    根据手部 2D 关键点生成手部区域掩码

    使用凸包算法找到手部关键点的外包络，然后膨胀掩码
    以覆盖手指之间的间隙和手腕区域。

    Args:
        keypoint_2d: MediaPipe 2D 关键点对象
        img_h: 图像高度
        img_w: 图像宽度
        dilation_pixels: 膨胀像素数

    Returns:
        ndarray: 二值掩码 (H, W)，uint8 格式
    """
    points = []
    for i in range(21):
        x = int(keypoint_2d.landmark[i].x * img_w)
        y = int(keypoint_2d.landmark[i].y * img_h)
        points.append([x, y])

    points = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(points)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilation_pixels + 1, 2 * dilation_pixels + 1)
    )
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (2 * dilation_pixels + 1, 2 * dilation_pixels + 1), 0)

    return mask


def composite_gripper_on_frame(original_frame, gripper_bgr, gripper_mask, hand_mask):
    """
    将夹爪渲染图像与原始视频帧进行合成

    合成策略：
      1. 夹爪掩码区域：完全用夹爪图像替换（alpha=1.0）
      2. 手部掩码区域（非夹爪）：用夹爪图像半透明覆盖（alpha=0.6）
      3. 非手部区域：保留原始图像

    Args:
        original_frame: 原始视频帧 BGR 图像 (H, W, 3)，uint8
        gripper_bgr: 夹爪渲染 BGR 图像 (H, W, 3)，uint8
        gripper_mask: 夹爪色键掩码 (H, W)，uint8，0-255 范围
        hand_mask: 手部区域掩码 (H, W)，uint8，0-255 范围

    Returns:
        ndarray: 合成后的 BGR 图像 (H, W, 3)，uint8
    """
    gripper_alpha = gripper_mask.astype(np.float32) / 255.0
    hand_alpha = hand_mask.astype(np.float32) / 255.0

    blend_alpha = np.maximum(gripper_alpha, hand_alpha * 0.6)
    blend_alpha = np.clip(blend_alpha, 0, 1)

    alpha_3ch = blend_alpha[..., np.newaxis]

    result = (
        original_frame.astype(np.float32) * (1 - alpha_3ch)
        + gripper_bgr.astype(np.float32) * alpha_3ch
    )

    return np.clip(result, 0, 255).astype(np.uint8)


def render_software_gripper_overlay(
    original_frame, qpos, hand_mask, wrist_pos, is_grasping, finger_distance
):
    """
    软件渲染夹爪叠加（SAPIEN 不可用时的回退方案）

    Args:
        original_frame: 原始视频帧 BGR 图像 (H, W, 3)
        qpos: 夹爪关节角度
        hand_mask: 手部区域掩码 (H, W)
        wrist_pos: 手腕 2D 位置 (x, y)，像素坐标
        is_grasping: 是否处于抓取状态
        finger_distance: 手指距离

    Returns:
        ndarray: 叠加后的 BGR 图像 (H, W, 3)
    """
    result = original_frame.copy()

    if hand_mask is None or wrist_pos is None:
        return result

    wx, wy = int(wrist_pos[0]), int(wrist_pos[1])

    gripper_width = 60
    gripper_height = 40
    finger_width = 12
    finger_height = 50

    if qpos is not None and len(qpos) > 0:
        gripper_open = float(np.clip(qpos[0] * 500, 0, 20))
    else:
        gripper_open = 10

    base_x = wx - gripper_width // 2
    base_y = wy - gripper_height // 2

    cv2.rectangle(
        result,
        (base_x, base_y),
        (base_x + gripper_width, base_y + gripper_height),
        (180, 180, 180), -1
    )
    cv2.rectangle(
        result,
        (base_x, base_y),
        (base_x + gripper_width, base_y + gripper_height),
        (100, 100, 100), 2
    )

    left_finger_x = base_x + gripper_width // 2 - finger_width - int(gripper_open)
    right_finger_x = base_x + gripper_width // 2 + int(gripper_open)

    finger_color = (0, 200, 0) if is_grasping else (0, 150, 255)
    cv2.rectangle(
        result,
        (left_finger_x, base_y - finger_height),
        (left_finger_x + finger_width, base_y),
        finger_color, -1
    )
    cv2.rectangle(
        result,
        (right_finger_x, base_y - finger_height),
        (right_finger_x + finger_width, base_y),
        finger_color, -1
    )

    status = "GRASP" if is_grasping else "OPEN"
    status_color = (0, 255, 0) if is_grasping else (0, 0, 255)
    cv2.putText(
        result, f"{status} d={finger_distance:.3f}",
        (wx - 40, base_y - finger_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
    )

    return result


def render_trajectory_video(pkl_path, output_video, fps=30):
    """
    从轨迹数据生成关节角度/手腕位置/手指距离曲线可视化视频

    Args:
        pkl_path: 轨迹数据文件路径 (trajectory.pkl)
        output_video: 输出视频文件路径
        fps: 帧率
    """
    with open(pkl_path, "rb") as f:
        result = pickle.load(f)

    data = result["data"]
    wrist_trajectory = result.get("wrist_trajectory", [])
    meta_data = result["meta_data"]
    joint_names = meta_data["joint_names"]
    dof = meta_data["dof"]

    W, H = 1280, 720
    panel_w = W // 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    valid_data = [d for d in data if d is not None]
    if len(valid_data) == 0:
        writer.release()
        return

    all_qpos = np.array(valid_data)
    qpos_min = all_qpos.min(axis=0)
    qpos_max = all_qpos.max(axis=0)
    qpos_range = qpos_max - qpos_min
    qpos_range[qpos_range < 1e-6] = 1.0

    wrist_positions = []
    finger_distances = []
    grasp_states = []
    for wt in wrist_trajectory:
        if wt is not None:
            wrist_positions.append(wt.get("wrist_pos", np.zeros(3)))
            finger_distances.append(wt.get("finger_distance", 0))
            grasp_states.append(wt.get("is_grasping", False))
        else:
            wrist_positions.append(np.zeros(3))
            finger_distances.append(0)
            grasp_states.append(False)

    wrist_positions = np.array(wrist_positions)
    finger_distances = np.array(finger_distances)

    n_fingers = max(1, min(4, dof))
    joints_per_finger = max(1, dof // n_fingers)

    for frame_i in range(len(data)):
        canvas = np.full((H, W, 3), 30, dtype=np.uint8)

        qpos = data[frame_i]
        wt = wrist_trajectory[frame_i] if frame_i < len(wrist_trajectory) else None

        left_panel = canvas[:, :panel_w]
        right_panel = canvas[:, panel_w:]

        cv2.rectangle(left_panel, (0, 0), (panel_w, H), (20, 20, 30), -1)
        cv2.rectangle(right_panel, (0, 0), (panel_w, H), (20, 25, 20), -1)

        cv2.putText(left_panel, "Robot Joint Angles", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(right_panel, "Wrist & Finger Trajectory", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if qpos is not None:
            margin_x, margin_y = 40, 60
            plot_w = panel_w - 2 * margin_x
            plot_h = (H - margin_y - 40) // n_fingers - 10

            finger_colors_bgr = [
                (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0),
            ]

            for fi in range(n_fingers):
                y_off = margin_y + fi * (plot_h + 10)
                finger_names = ["Thumb", "Index", "Middle", "Ring"]
                color = finger_colors_bgr[fi % len(finger_colors_bgr)]

                cv2.putText(left_panel, finger_names[fi % 4], (margin_x, y_off - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                for ji in range(joints_per_finger):
                    idx = fi * joints_per_finger + ji
                    if idx >= dof:
                        break
                    val = qpos[idx]
                    norm_val = (val - qpos_min[idx]) / qpos_range[idx]
                    norm_val = np.clip(norm_val, 0, 1)

                    bar_w = plot_w // joints_per_finger - 4
                    bar_x = margin_x + ji * (plot_w // joints_per_finger)
                    bar_h = int(norm_val * (plot_h - 20))
                    bar_y = y_off + plot_h - bar_h

                    cv2.rectangle(left_panel,
                                  (bar_x, y_off + 15),
                                  (bar_x + bar_w, y_off + plot_h),
                                  (50, 50, 50), -1)
                    cv2.rectangle(left_panel,
                                  (bar_x, bar_y),
                                  (bar_x + bar_w, y_off + plot_h),
                                  color, -1)

                    cv2.putText(left_panel, f"J{ji}", (bar_x + 2, y_off + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

            margin_x_r, margin_y_r = 40, 60
            plot_w_r = panel_w - 2 * margin_x_r

            if len(wrist_positions) > 0:
                wp = wrist_positions[frame_i] if frame_i < len(wrist_positions) else np.zeros(3)
                cv2.putText(right_panel, f"Wrist X: {wp[0]:.3f}", (margin_x_r, margin_y_r),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(right_panel, f"Wrist Y: {wp[1]:.3f}", (margin_x_r, margin_y_r + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(right_panel, f"Wrist Z: {wp[2]:.3f}", (margin_x_r, margin_y_r + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if len(finger_distances) > 0:
                fd = finger_distances[frame_i] if frame_i < len(finger_distances) else 0
                cv2.putText(right_panel, f"Finger dist: {fd:.4f}", (margin_x_r, margin_y_r + 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if wt is not None:
                is_grasping = wt.get("is_grasping", False)
                status = "GRASP" if is_grasping else "OPEN"
                status_color = (0, 255, 0) if is_grasping else (0, 100, 255)
                cv2.putText(right_panel, f"Status: {status}", (margin_x_r, margin_y_r + 115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            if frame_i > 0:
                trail_len = min(frame_i, 200)
                start = frame_i - trail_len
                for axis_idx, (color, label) in enumerate([
                    ((0, 0, 255), "X"), ((0, 255, 0), "Y"), ((255, 0, 0), "Z")
                ]):
                    vals = wrist_positions[start:frame_i + 1, axis_idx]
                    if len(vals) > 1:
                        v_min, v_max = vals.min(), vals.max()
                        v_range = max(v_max - v_min, 1e-6)
                        plot_y_base = margin_y_r + 160 + axis_idx * 80
                        plot_h_line = 60

                        cv2.rectangle(right_panel,
                                      (margin_x_r, plot_y_base),
                                      (margin_x_r + plot_w_r, plot_y_base + plot_h_line),
                                      (40, 40, 40), -1)
                        cv2.putText(right_panel, f"Wrist {label}", (margin_x_r, plot_y_base - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                        for k in range(1, len(vals)):
                            x1 = margin_x_r + int((k - 1) / trail_len * plot_w_r)
                            x2 = margin_x_r + int(k / trail_len * plot_w_r)
                            y1 = plot_y_base + plot_h_line - int((vals[k - 1] - v_min) / v_range * plot_h_line)
                            y2 = plot_y_base + plot_h_line - int((vals[k] - v_min) / v_range * plot_h_line)
                            cv2.line(right_panel, (x1, y1), (x2, y2), color, 1)

                if len(finger_distances) > 1:
                    fd_vals = finger_distances[start:frame_i + 1]
                    fd_min, fd_max = fd_vals.min(), fd_vals.max()
                    fd_range = max(fd_max - fd_min, 1e-6)
                    plot_y_base = margin_y_r + 160 + 3 * 80
                    plot_h_line = 60

                    cv2.rectangle(right_panel,
                                  (margin_x_r, plot_y_base),
                                  (margin_x_r + plot_w_r, plot_y_base + plot_h_line),
                                  (40, 40, 40), -1)
                    cv2.putText(right_panel, "Finger dist", (margin_x_r, plot_y_base - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

                    for k in range(1, len(fd_vals)):
                        x1 = margin_x_r + int((k - 1) / trail_len * plot_w_r)
                        x2 = margin_x_r + int(k / trail_len * plot_w_r)
                        y1 = plot_y_base + plot_h_line - int((fd_vals[k - 1] - fd_min) / fd_range * plot_h_line)
                        y2 = plot_y_base + plot_h_line - int((fd_vals[k] - fd_min) / fd_range * plot_h_line)
                        cv2.line(right_panel, (x1, y1), (x2, y2), (200, 200, 200), 1)
        else:
            cv2.putText(left_panel, "No detection", (panel_w // 4, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.putText(right_panel, "No detection", (panel_w // 4, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

        cv2.putText(canvas, f"Frame {frame_i + 1}/{len(data)}", (W // 2 - 60, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        writer.write(canvas)

    writer.release()


def run_gripper_replacement_pipeline(
    video_path: str,
    output_dir: str,
    robot_name: RobotName = RobotName.panda,
    hand_type: HandType = HandType.right,
    retargeting_type: RetargetingType = RetargetingType.position,
    fps: int = 30,
    start_frame: int = 0,
    num_frames: int = -1,
    selfie: bool = True,
    grasp_threshold: float = None,
    release_threshold: float = None,
    software_render: bool = False,
    visualize: bool = True,
    mask_dilation: int = 30,
    mano_scale: float = 2.0,
    show_axes: bool = True,
    show_trail: bool = True,
    show_mano: bool = True,
):
    """
    执行完整的第一人称视频 -> 夹爪替换管道

    输出视频：
      1. gripper_replaced.mp4: 夹爪替换后的视频
      2. sapien_render.mp4: 纯 SAPIEN 3D 渲染（含 MANO/坐标轴/轨迹）
      3. overlay.mp4: 原始视频 + 手部骨架标注
      4. trajectory_viz.mp4: 关节角度/手腕位置曲线可视化
      5. trajectory.pkl: 完整轨迹数据
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_video_path = output_dir / "gripper_replaced.mp4"
    sapien_video_path = output_dir / "sapien_render.mp4"
    overlay_path = output_dir / "overlay.mp4"
    traj_viz_path = output_dir / "trajectory_viz.mp4"
    pkl_path = output_dir / "trajectory.pkl"

    hand_str = "Right" if hand_type == HandType.right else "Left"
    print("=" * 70)
    print("  Egocentric Video -> Gripper Replacement Pipeline")
    print("=" * 70)
    print(f"  Video:     {video_path}")
    print(f"  Robot:     {robot_name.name}")
    print(f"  Hand:      {hand_str}")
    print(f"  Retarget:  {retargeting_type.name}")
    print(f"  Selfie:    {selfie}")
    print(f"  Render:    {'Software' if software_render else 'SAPIEN 3D'}")
    print(f"  MANO viz:  {show_mano}")
    print(f"  Axes viz:  {show_axes}")
    print(f"  Trail viz: {show_trail}")
    print(f"  Output:    {output_dir}")
    print()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_skip = max(1, int(video_fps / fps))
    if num_frames < 0:
        num_frames = total_frames

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Video info: {total_frames} frames, {video_fps:.1f} fps, {vid_w}x{vid_h}")
    print()

    detector = EgocentricHandDetector(
        hand_type=hand_str,
        selfie=selfie,
    )

    sapien_renderer = None
    if not software_render:
        try:
            import subprocess
            check_result = subprocess.run(
                [sys.executable, "-c", "import sapien; sapien.Scene()"],
                capture_output=True, timeout=15,
            )
            if check_result.returncode != 0:
                print("  SAPIEN not functional, falling back to software rendering...")
                software_render = True
            else:
                sapien_renderer = GripperSAPIENRenderer(
                    robot_name=robot_name,
                    hand_type=hand_type,
                    retargeting_type=retargeting_type,
                    img_width=vid_w,
                    img_height=vid_h,
                    mano_scale=mano_scale,
                    show_axes=show_axes,
                    show_trail=show_trail,
                    show_mano=show_mano,
                )
                print("  SAPIEN 3D renderer initialized successfully")
        except Exception as e:
            print(f"  SAPIEN check failed: {e}")
            print("  Falling back to software rendering...")
            software_render = True

    arm_analyzer = ArmMotionAnalyzer(
        hand_type=hand_type,
        arm_offset=np.zeros(3),
    )
    retargeter = RobotHandRetargeter(
        robot_name=robot_name,
        hand_type=hand_type,
        retargeting_type=retargeting_type,
    )
    contact_filter = ContactPreservationFilter(
        grasp_threshold=grasp_threshold if grasp_threshold else 0.04,
        release_threshold=release_threshold if release_threshold else 0.06,
    )
    trajectory_smoother = TrajectorySmoother(window_size=3, alpha=0.3)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    result_writer = cv2.VideoWriter(
        str(result_video_path), fourcc, fps, (vid_w, vid_h)
    )
    sapien_writer = None
    if sapien_renderer is not None:
        sapien_writer = cv2.VideoWriter(
            str(sapien_video_path), fourcc, fps, (vid_w, vid_h)
        )
    overlay_writer = cv2.VideoWriter(
        str(overlay_path), fourcc, fps, (vid_w, vid_h)
    )

    data = []
    wrist_trajectory = []
    meta_data = retargeter.get_meta_data()

    fov_h = np.radians(70)
    f_norm = 1.0 / (2 * np.tan(fov_h / 2))
    aspect = vid_w / max(vid_h, 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    processed = 0
    detected = 0
    finger_distances = []
    warmup_frames = 5
    prev_qpos = None
    prev_target_pose = None

    print("[1/4] Detecting hands, retargeting, and compositing...")
    with tqdm.tqdm(total=num_frames, desc="Processing") as pbar:
        while cap.isOpened() and processed < num_frames:
            frame_idx = start_frame + processed * frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            rgb = frame[..., ::-1]
            hand_data = detector.detect_single(rgb)

            bgr_overlay = frame.copy()
            result_frame = frame.copy()

            if hand_data is None:
                data.append(None)
                wrist_trajectory.append(None)
                cv2.putText(bgr_overlay, "NO HAND", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                overlay_writer.write(bgr_overlay)
                result_writer.write(result_frame)

                if sapien_renderer is not None and sapien_writer is not None:
                    if prev_qpos is not None:
                        robot_qpos = prev_qpos[sapien_renderer.retarget2sapien]
                        sapien_renderer.robot.set_qpos(robot_qpos)
                        for idx in sapien_renderer.dummy_joint_indices:
                            sapien_renderer.robot_active_joints[idx].get_drive().set_target_qpos(robot_qpos[idx])
                        for idx in sapien_renderer.finger_joint_indices:
                            sapien_renderer.robot_active_joints[idx].get_drive().set_target_qpos(robot_qpos[idx])
                    sapien_renderer._clear_nodes(sapien_renderer.mano_nodes)
                    sapien_renderer._clear_nodes(sapien_renderer.axis_nodes)
                    sapien_renderer.scene.step()
                    if prev_qpos is not None and prev_target_pose is not None:
                        sapien_renderer.robot.set_pose(prev_target_pose)
                    sapien_renderer.scene.update_render()
                    sapien_renderer.camera.take_picture()
                    rgb_s = sapien_renderer.camera.get_picture("Color")[..., :3]
                    bgr_s = (np.clip(rgb_s, 0, 1) * 255).astype(np.uint8)[..., ::-1]
                    cv2.putText(bgr_s, f"Frame {frame_idx} | LOST", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    sapien_writer.write(np.ascontiguousarray(bgr_s))

                processed += 1
                pbar.update(1)
                continue

            detected += 1
            joint_pos = hand_data["joint_pos"]
            wrist_pos_raw = hand_data["wrist_pos_raw"]
            mediapipe_wrist_rot = hand_data["wrist_rot_mano"]
            keypoint_2d = hand_data["keypoint_2d"]
            wrist_pos_camera = hand_data.get("wrist_pos_camera", wrist_pos_raw)

            if processed < 10:
                print(f"  Frame {frame_idx}: joint_pos range=[{joint_pos.min():.4f}, {joint_pos.max():.4f}], "
                      f"wrist_cam={wrist_pos_camera}, wrist_rot diag={np.diag(mediapipe_wrist_rot)}")

            retargeting = sapien_renderer.retargeting if sapien_renderer is not None else retargeter.retargeting
            indices = retargeting.optimizer.target_link_human_indices
            retargeting_type_str = retargeting.optimizer.retargeting_type

            if retargeting_type_str == "POSITION":
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]

            if sapien_renderer is not None:
                qpos = retargeting.retarget(ref_value.astype(np.float32), sapien_renderer.fixed_qpos)
            else:
                qpos = retargeter.retarget(joint_pos, wrist_pos_raw,
                                           R_scipy_quat=None)

            finger_dist = ArmMotionAnalyzer.compute_finger_aperture(joint_pos)
            finger_distances.append(finger_dist)
            filtered_qpos, is_grasping = contact_filter.update(finger_dist, qpos)
            smoothed_qpos = trajectory_smoother.smooth(filtered_qpos)

            arm_info = arm_analyzer.compute_wrist_6dof(hand_data)

            data.append(smoothed_qpos)
            wrist_trajectory.append(
                {
                    "wrist_pos": arm_info["wrist_pos"],
                    "wrist_quat": arm_info["wrist_quat"],
                    "ee_pos": arm_info["ee_pos"],
                    "ee_quat": arm_info["ee_quat"],
                    "finger_distance": finger_dist,
                    "is_grasping": is_grasping,
                    "frame_idx": frame_idx,
                    "wrist_pos_camera": wrist_pos_camera,
                }
            )

            detector.draw_skeleton_on_image(bgr_overlay, keypoint_2d)
            status = "GRASP" if is_grasping else "OPEN"
            color = (0, 255, 0) if is_grasping else (0, 0, 255)
            cv2.putText(
                bgr_overlay,
                f"{status} d={finger_dist:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            cv2.putText(
                bgr_overlay,
                f"Wrist(cam): ({wrist_pos_camera[0]:.2f}, {wrist_pos_camera[1]:.2f}, {wrist_pos_camera[2]:.2f})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )
            overlay_writer.write(bgr_overlay)

            wrist_x = keypoint_2d.landmark[0].x
            wrist_y = keypoint_2d.landmark[0].y
            wrist_2d_px = np.array([wrist_x * vid_w, wrist_y * vid_h])

            hand_mask = compute_hand_mask(keypoint_2d, vid_h, vid_w, mask_dilation)

            if not software_render and sapien_renderer is not None:
                gripper_info = sapien_renderer.update_gripper(
                    wrist_pos_camera, mediapipe_wrist_rot, smoothed_qpos, joint_pos=joint_pos
                )
                prev_qpos = smoothed_qpos
                prev_target_pose = gripper_info["target_pose"]

                if processed >= warmup_frames:
                    sapien_bgr, gripper_bgr, gripper_mask = sapien_renderer.render_frame(
                        keypoint_2d=keypoint_2d,
                        frame_idx=frame_idx,
                        num_box=1,
                        mano_origin=gripper_info["wrist_pos_base"],
                    )

                    result_frame = composite_gripper_on_frame(
                        result_frame, gripper_bgr, gripper_mask, hand_mask
                    )

                    if sapien_writer is not None:
                        sapien_writer.write(sapien_bgr)
                else:
                    sapien_renderer.scene.update_render()
                    sapien_renderer.camera.take_picture()
                    if sapien_writer is not None:
                        rgb_s = sapien_renderer.camera.get_picture("Color")[..., :3]
                        bgr_s = (np.clip(rgb_s, 0, 1) * 255).astype(np.uint8)[..., ::-1]
                        sapien_writer.write(np.ascontiguousarray(bgr_s))
            else:
                result_frame = render_software_gripper_overlay(
                    result_frame, smoothed_qpos, hand_mask,
                    wrist_2d_px, is_grasping, finger_dist
                )

            result_writer.write(result_frame)

            if visualize:
                display = np.hstack([frame, result_frame])
                display = cv2.resize(display, (vid_w, vid_h // 2))
                cv2.imshow("gripper_replacement", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            pbar.update(1)

    cap.release()
    result_writer.release()
    if sapien_writer is not None:
        sapien_writer.release()
    overlay_writer.release()
    if visualize:
        cv2.destroyAllWindows()

    valid = sum(1 for d in data if d is not None)
    print(f"\n  Detection: {valid}/{processed} ({valid/max(processed,1)*100:.1f}%)")

    if finger_distances:
        fd = np.array(finger_distances)
        print(f"  Finger distance: mean={fd.mean():.4f}, std={fd.std():.4f}, "
              f"min={fd.min():.4f}, max={fd.max():.4f}")
        if grasp_threshold is None and fd.max() < 0.04:
            print(f"  NOTE: Finger distances are small (egocentric view).")
            print(f"        Consider using --grasp-threshold {fd.mean()*0.6:.4f} "
                  f"--release-threshold {fd.mean()*0.9:.4f}")

    print("[2/4] Saving trajectory...")
    with pkl_path.open("wb") as f:
        pickle.dump(
            {
                "data": data,
                "wrist_trajectory": wrist_trajectory,
                "meta_data": meta_data,
                "video_info": {
                    "video_path": str(video_path),
                    "start_frame": start_frame,
                    "num_frames": processed,
                    "fps": fps,
                    "frame_skip": frame_skip,
                },
            },
            f,
        )
    print(f"  Saved: {pkl_path}")

    print("[3/4] Generating trajectory visualization video...")
    render_trajectory_video(str(pkl_path), str(traj_viz_path), fps=fps)
    print(f"  Saved: {traj_viz_path}")

    print("[4/4] Pipeline complete!")
    print()
    print("=" * 70)
    print("  Output files:")
    print(f"  Replaced video:  {result_video_path}")
    print(f"  SAPIEN render:   {sapien_video_path}")
    print(f"  Overlay video:   {overlay_path}")
    print(f"  Trajectory viz:  {traj_viz_path}")
    print(f"  Trajectory data: {pkl_path}")
    print("=" * 70)

    if sapien_renderer is not None:
        sapien_renderer.retargeting.verbose()
    else:
        retargeter.retargeting.verbose()


def main():
    """命令行入口：解析参数并运行夹爪替换管道"""
    parser = argparse.ArgumentParser(
        description="Egocentric Video -> Gripper Replacement Pipeline"
    )
    parser.add_argument("--video-path", type=str, required=True, help="输入 MP4 视频文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录路径")
    parser.add_argument(
        "--robot",
        type=str,
        default="panda",
        choices=[r.name for r in RobotName],
        help="夹爪类型（panda=Panda夹爪, r1=R1夹爪）",
    )
    parser.add_argument(
        "--hand",
        type=str,
        default="right",
        choices=["right", "left"],
        help="追踪的手别",
    )
    parser.add_argument(
        "--retargeting",
        type=str,
        default="position",
        choices=["position", "vector", "dexpilot"],
        help="重定向算法",
    )
    parser.add_argument("--fps", type=int, default=30, help="目标帧率")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧索引")
    parser.add_argument("--num-frames", type=int, default=-1, help="处理帧数（-1=全部）")
    parser.add_argument(
        "--no-selfie",
        action="store_true",
        help="禁用自拍模式（非第一人称视角视频使用）",
    )
    parser.add_argument(
        "--grasp-threshold",
        type=float,
        default=None,
        help="抓取检测阈值（默认 0.04）",
    )
    parser.add_argument(
        "--release-threshold",
        type=float,
        default=None,
        help="释放检测阈值（默认 0.06）",
    )
    parser.add_argument(
        "--software-render",
        action="store_true",
        help="使用软件渲染（不使用 SAPIEN 3D）",
    )
    parser.add_argument("--no-visualize", action="store_true", help="禁用实时可视化")
    parser.add_argument(
        "--mask-dilation",
        type=int,
        default=30,
        help="手部掩码膨胀像素数（默认 30）",
    )
    parser.add_argument(
        "--mano-scale",
        type=float,
        default=2.0,
        help="MANO 骨架缩放系数（默认 2.0）",
    )
    parser.add_argument("--no-axes", action="store_true", help="隐藏坐标轴")
    parser.add_argument("--no-trail", action="store_true", help="隐藏轨迹拖尾")
    parser.add_argument("--no-mano", action="store_true", help="隐藏 MANO 骨架")

    args = parser.parse_args()

    retargeting_map = {
        "position": RetargetingType.position,
        "vector": RetargetingType.vector,
        "dexpilot": RetargetingType.dexpilot,
    }

    run_gripper_replacement_pipeline(
        video_path=args.video_path,
        output_dir=args.output_dir,
        robot_name=RobotName[args.robot],
        hand_type=HandType[args.hand],
        retargeting_type=retargeting_map[args.retargeting],
        fps=args.fps,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        selfie=not args.no_selfie,
        grasp_threshold=args.grasp_threshold,
        release_threshold=args.release_threshold,
        software_render=args.software_render,
        visualize=not args.no_visualize,
        mask_dilation=args.mask_dilation,
        mano_scale=args.mano_scale,
        show_axes=not args.no_axes,
        show_trail=not args.no_trail,
        show_mano=not args.no_mano,
    )


if __name__ == "__main__":
    main()
