# 导入临时目录处理模块
import tempfile
# 导入路径处理模块
from pathlib import Path
# 导入类型提示模块
from typing import Dict, List

# 导入OpenCV计算机视觉库（用于视频录制）
import cv2
# 导入NumPy数值计算库
import numpy as np
# 导入SAPIEN物理仿真引擎
import sapien
# 从同级目录导入手部可视化器基类
from hand_viewer import HandDatasetSAPIENViewer
# 导入3D旋转计算库
from pytransform3d import rotations
# 导入进度条显示库
from tqdm import trange

# 从dex_retargeting包导入URDF解析工具
from dex_retargeting import yourdfpy as urdf
# 从constants模块导入常量定义（机器人名称、手部类型、重定向类型等）
from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
# 从重定向配置模块导入配置类
from dex_retargeting.retargeting_config import RetargetingConfig
# 从序列重定向模块导入序列重定向类
from dex_retargeting.seq_retarget import SeqRetargeting


class RobotHandDatasetSAPIENViewer(HandDatasetSAPIENViewer):
    """
    机器人手部数据集SAPIEN可视化器类
    继承自HandDatasetSAPIENViewer，添加了机器人手部的可视化和重定向功能
    """
    
    def __init__(
        self,
        robot_names: List[RobotName],
        hand_type: HandType,
        headless=False,
        use_ray_tracing=False,
        retargeting_overrides=None,
    ):
        super().__init__(headless=headless, use_ray_tracing=use_ray_tracing)

        self.robot_names = robot_names
        self.robots: List[sapien.Articulation] = []
        self.robot_file_names: List[str] = []
        self.retargetings: List[SeqRetargeting] = []
        self.retarget2sapien: List[np.ndarray] = []
        self.hand_type = hand_type

        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        for robot_name in robot_names:
            config_path = get_default_config_path(
                robot_name, RetargetingType.position, hand_type
            )

            override = dict(add_dummy_free_joint=True)
            if retargeting_overrides:
                override.update(retargeting_overrides)
            config = RetargetingConfig.load_from_file(config_path, override=override)
            retargeting = config.build()  # 构建重定向器对象
            robot_file_name = Path(config.urdf_path).stem  # 提取URDF文件名（不含扩展名）
            self.robot_file_names.append(robot_file_name)
            self.retargetings.append(retargeting)

            # Build robot - 构建机器人实体
            urdf_path = Path(config.urdf_path)
            if "glb" not in urdf_path.stem:
                urdf_path = urdf_path.with_stem(urdf_path.stem + "_glb")  # 使用带GLB视觉模型的URDF
            robot_urdf = urdf.URDF.load(
                str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False
            )  # 加载URDF，添加虚拟自由关节，不构建场景图
            urdf_name = urdf_path.name
            temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")  # 创建临时目录
            temp_path = f"{temp_dir}/{urdf_name}"  # 构造临时文件路径
            robot_urdf.write_xml_file(temp_path)  # 将URDF写入临时文件

            robot = loader.load(temp_path)  # 使用SAPIEN加载机器人
            self.robots.append(robot)
            sapien_joint_names = [joint.name for joint in robot.get_active_joints()]  # 获取SAPIEN关节名称
            retarget2sapien = np.array(
                [retargeting.joint_names.index(n) for n in sapien_joint_names]
            ).astype(int)  # 创建重定向关节到SAPIEN关节的索引映射
            self.retarget2sapien.append(retarget2sapien)

    def load_object_hand(self, data: Dict):
        """
        加载物体和手部数据（重写父类方法）
        为每个机器人都加载一份相同的YCB物体副本
        参数:
            data: 包含ycb_ids、object_mesh_file等的字典
        """
        super().load_object_hand(data)  # 先调用父类方法加载基础物体和手部模型
        ycb_ids = data["ycb_ids"]
        ycb_mesh_files = data["object_mesh_file"]

        # Load the same YCB objects for n times, n is the number of robots
        # So that for each robot, there will be an identical set of objects
        for _ in range(len(self.robots)):
            for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
                self._load_ycb_object(ycb_id, ycb_mesh_file)  # 为每个机器人加载YCB物体副本

    def render_dexycb_data(self, data: Dict, fps=5, y_offset=0.8):
        """
        渲染DexYCB数据集序列，同时显示人手和多个机器人手
        参数:
            data: 包含hand_pose和object_pose的字典
            fps: 播放帧率
            y_offset: 机器人在Y轴方向的间距
        """
        # Set table and viewer pose for better visual effect only
        global_y_offset = -y_offset * len(self.robots) / 2  # 计算全局Y轴偏移量（使机器人居中排列）
        self.table.set_pose(sapien.Pose([0.5, global_y_offset + 0.2, 0]))  # 设置桌子位姿
        if not self.headless:
            self.viewer.set_camera_xyz(1.5, global_y_offset, 1)  # 调整查看器相机位置
        else:
            local_pose = self.camera.get_local_pose()  # 获取离屏相机当前位姿
            local_pose.set_p(np.array([1.5, global_y_offset, 1]))  # 设置新的位置
            self.camera.set_local_pose(local_pose)  # 应用新的位姿

        hand_pose = data["hand_pose"]  # 手部姿态序列
        object_pose = data["object_pose"]  # 物体姿态序列
        num_frame = hand_pose.shape[0]  # 总帧数
        num_copy = len(self.robots) + 1  # 副本数量（1个人手 + N个机器人）
        num_ycb_objects = len(data["ycb_ids"])  # YCB物体数量
        pose_offsets = []  # 位姿偏移列表

        for i in range(len(self.robots) + 1):
            pose = sapien.Pose([0, -y_offset * i, 0])  # 每个副本在Y轴上间隔y_offset
            pose_offsets.append(pose)
            if i >= 1:
                self.robots[i - 1].set_pose(pose)  # 设置机器人初始位姿（从第1个开始）

        # Skip frames where human hand is not detected in DexYCB dataset
        start_frame = 0
        for i in range(0, num_frame):
            init_hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(init_hand_pose_frame)
            if vertex is not None:
                start_frame = i  # 找到第一个检测到手部的帧
                break

        if self.headless:
            robot_names = [robot.name for robot in self.robot_names]  # 提取机器人名称
            robot_names = "_".join(robot_names)  # 用下划线连接
            video_path = (
                Path(__file__).parent.resolve() / f"data/{robot_names}_video.mp4"
            )  # 视频输出路径（包含机器人名称）
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # MP4V编码器
                30.0,  # 30fps
                (self.camera.get_width(), self.camera.get_height()),  # 视频分辨率
            )  # 创建视频写入器

        # Warm start - 热启动：使用第一帧初始化重定向器
        hand_pose_start = hand_pose[start_frame]
        wrist_quat = rotations.quaternion_from_compact_axis_angle(
            hand_pose_start[0, 0:3]
        )  # 从前3维紧凑轴角转换为四元数（手腕旋转）
        vertex, joint = self._compute_hand_geometry(hand_pose_start)  # 计算起始帧手部几何形状
        for robot, retargeting, retarget2sapien in zip(
            self.robots, self.retargetings, self.retarget2sapien
        ):
            retargeting.warm_start(
                joint[0, :],
                wrist_quat,
                hand_type=self.hand_type,
                is_mano_convention=True,
            )  # 热启动重定向器（优化器需要初始解）

        # Loop rendering - 循环渲染每一帧
        step_per_frame = int(60 / fps)  # 每帧需要渲染的步数（基于60Hz刷新率）
        for i in trange(start_frame, num_frame):
            object_pose_frame = object_pose[i]
            hand_pose_frame = hand_pose[i]
            vertex, joint = self._compute_hand_geometry(hand_pose_frame)  # 计算手部几何形状

            # Update poses for YCB objects - 更新YCB物体位姿
            for k in range(num_ycb_objects):
                pos_quat = object_pose_frame[k]

                # Quaternion convention: xyzw -> wxyz
                pose = self.camera_pose * sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                )  # 组合相机位姿和物体位姿，注意四元数格式转换
                self.objects[k].set_pose(pose)  # 设置第1组物体位姿
                for copy_ind in range(num_copy):
                    self.objects[k + copy_ind * num_ycb_objects].set_pose(
                        pose_offsets[copy_ind] * pose
                    )  # 为每个副本设置带偏移的位姿

            # Update pose for human hand - 更新人手显示
            self._update_hand(vertex)

            # Update poses for robot hands - 更新机器人手部位姿
            for robot, retargeting, retarget2sapien in zip(
                self.robots, self.retargetings, self.retarget2sapien
            ):
                indices = retargeting.optimizer.target_link_human_indices  # 获取目标链接索引
                ref_value = joint[indices, :]  # 提取对应关节位置作为参考值
                qpos = retargeting.retarget(ref_value)[retarget2sapien]  # 执行重定向并重新映射关节顺序
                robot.set_qpos(qpos)  # 设置机器人关节位置

            self.scene.update_render()  # 更新场景渲染
            if self.headless:
                self.camera.take_picture()  # 相机拍摄一帧
                rgb = self.camera.get_picture("Color")[..., :3]  # 获取RGB通道
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)  # 转换为uint8
                writer.write(rgb[..., ::-1])  # 写入视频（BGR格式）
            else:
                for _ in range(step_per_frame):
                    self.viewer.render()  # 交互式渲染

        if not self.headless:
            self.viewer.paused = True  # 暂停查看器
            self.viewer.render()  # 最后渲染一次
        else:
            writer.release()  # 释放视频写入器
