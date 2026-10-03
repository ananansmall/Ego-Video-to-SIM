# 导入路径处理模块
from pathlib import Path
# 导入类型提示模块
from typing import Dict, List, Optional

# 导入OpenCV计算机视觉库（用于视频录制）
import cv2
# 导入进度条显示库
from tqdm import trange
# 导入NumPy数值计算库
import numpy as np
# 导入SAPIEN物理仿真引擎
import sapien
# 导入PyTorch深度学习框架（用于MANO模型）
import torch
# 导入3D变换库（用于位姿转换）
from pytransform3d import transformations as pt
# 导入SAPIEN内部渲染器
from sapien import internal_renderer as R
# 导入SAPIEN环境贴图创建工具
from sapien.asset import create_dome_envmap
# 导入SAPIEN可视化工具
from sapien.utils import Viewer

# 从本地数据集模块导入YCB物体类别定义
from dataset import YCB_CLASSES
# 从本地MANO层模块导入手部模型层
from mano_layer import MANOLayer


def compute_smooth_shading_normal_np(vertices, indices):
    """
    Compute the vertex normal from vertices and triangles with numpy
    Args:
        vertices: (n, 3) to represent vertices position
        indices: (m, 3) to represent the triangles, should be in counter-clockwise order to compute normal outwards
    Returns:
        (n, 3) vertex normal

    References:
        https://www.iquilezles.org/www/articles/normals/normals.htm
    """
    # 获取三角形的三个顶点坐标
    v1 = vertices[indices[:, 0]]
    v2 = vertices[indices[:, 1]]
    v3 = vertices[indices[:, 2]]
    # 计算每个三角形的面法线（未归一化），使用叉乘：(v2-v1) × (v3-v1)
    face_normal = np.cross(v2 - v1, v3 - v1)  # (n, 3) normal without normalization to 1

    # 初始化顶点法线数组，与顶点数组形状相同
    vertex_normal = np.zeros_like(vertices)
    # 累加每个三角形面对其三个顶点的法线贡献（平滑着色的关键）
    vertex_normal[indices[:, 0]] += face_normal
    vertex_normal[indices[:, 1]] += face_normal
    vertex_normal[indices[:, 2]] += face_normal
    # 对每个顶点的法线进行归一化，使其长度为1
    vertex_normal /= np.linalg.norm(vertex_normal, axis=1, keepdims=True)
    return vertex_normal


class HandDatasetSAPIENViewer:
    """手部数据集SAPIEN可视化器类"""
    
    def __init__(self, headless=False, use_ray_tracing=False):
        """
        初始化可视化器
        参数:
            headless: 是否无头模式（不显示GUI窗口）
            use_ray_tracing: 是否使用光线追踪渲染
        """
        # Setup - 设置渲染着色器
        if not use_ray_tracing:
            # 使用默认光栅化渲染器
            sapien.render.set_viewer_shader_dir("default")
            sapien.render.set_camera_shader_dir("default")
        else:
            # 使用光线追踪渲染器
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(64)  # 每像素64个采样
            sapien.render.set_ray_tracing_path_depth(8)  # 光线追踪路径深度为8
            sapien.render.set_ray_tracing_denoiser("oidn")  # 使用Intel OIDN去噪器

        # Scene - 创建场景
        scene = sapien.Scene()
        scene.set_timestep(1 / 240)  # 设置仿真时间步长为1/240秒（240Hz）

        # Lighting - 设置光照
        scene.set_environment_map(
            create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
        )  # 创建穹顶环境贴图，天空和地面都是深灰色
        scene.add_directional_light(
            np.array([1, -1, -1]), np.array([2, 2, 2]), shadow=True
        )  # 方向光1：从(1,-1,-1)方向照射，白色，启用阴影
        scene.add_directional_light([0, 0, -1], [1.8, 1.6, 1.6], shadow=False)  # 方向光2：从上往下，暖白色，无阴影
        scene.set_ambient_light(np.array([0.2, 0.2, 0.2]))  # 环境光为暗灰色

        # Add ground - 添加地面
        visual_material = sapien.render.RenderMaterial()
        visual_material.set_base_color(np.array([0.5, 0.5, 0.5, 1]))  # 中灰色，完全不透明
        visual_material.set_roughness(0.7)  # 粗糙度0.7
        visual_material.set_metallic(1)  # 金属度1（完全金属）
        visual_material.set_specular(0.04)  # 镜面反射强度0.04
        scene.add_ground(-1, render_material=visual_material)  # 在y=-1位置添加地面

        # Viewer - 创建查看器或相机
        if not headless:
            # 有头模式：创建交互式查看器
            viewer = Viewer()
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1.5, 0, 1)  # 相机位置(1.5, 0, 1)
            viewer.set_camera_rpy(0, -0.8, 3.14)  # 相机姿态（俯视角度）
            viewer.control_window.toggle_origin_frame(False)  # 隐藏原点坐标系
            self.viewer = viewer
        else:
            # 无头模式：创建离屏渲染相机
            self.camera = scene.add_camera("cam", 1920, 640, 0.9, 0.01, 100)  # 分辨率1920x640，FOV 0.9
            self.camera.set_local_pose(
                sapien.Pose([1.5, 0, 1], [0, 0.389418, 0, -0.921061])
            )  # 设置相机位姿

        self.headless = headless

        # Create table - 创建桌子
        white_diffuse = sapien.render.RenderMaterial()
        white_diffuse.set_base_color(np.array([0.8, 0.8, 0.8, 1]))  # 浅灰色
        white_diffuse.set_roughness(0.9)  # 非常粗糙
        builder = scene.create_actor_builder()
        builder.add_box_collision(
            sapien.Pose([0, 0, -0.02]), half_size=np.array([0.5, 2.0, 0.02])
        )  # 桌面碰撞体
        builder.add_box_visual(
            sapien.Pose([0, 0, -0.02]),
            half_size=np.array([0.5, 2.0, 0.02]),
            material=white_diffuse,
        )  # 桌面视觉模型
        builder.add_box_visual(
            sapien.Pose([0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )  # 桌腿1
        builder.add_box_visual(
            sapien.Pose([-0.4, 1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )  # 桌腿2
        builder.add_box_visual(
            sapien.Pose([0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )  # 桌腿3
        builder.add_box_visual(
            sapien.Pose([-0.4, -1.9, -0.51]),
            half_size=np.array([0.015, 0.015, 0.49]),
            material=white_diffuse,
        )  # 桌腿4
        self.table = builder.build_static(name="table")  # 构建静态桌子
        self.table.set_pose(sapien.Pose([0.5, 0, 0]))  # 设置桌子位姿

        # Caches - 缓存初始化
        sapien.render.set_log_level("error")  # 仅显示错误日志
        self.scene = scene
        self.internal_scene: R.Scene = scene.render_system._internal_scene  # 内部渲染场景
        self.context: R.Context = sapien.render.SapienRenderer()._internal_context  # 渲染上下文
        self.mat_hand = self.context.create_material(
            np.zeros(4), np.array([0.96, 0.75, 0.69, 1]), 0.0, 0.8, 0
        )  # 手部材质：肤色(0.96,0.75,0.69)，粗糙度0.0，金属度0.8

        self.mano_layer: Optional[MANOLayer] = None  # MANO手部模型层
        self.mano_face: Optional[np.ndarray] = None  # MANO手部网格面片索引
        self.camera_pose: Optional[sapien.Pose] = None  # 相机位姿
        self.objects: List[sapien.Entity] = []  # YCB物体列表
        self.nodes: List[R.Node] = []  # 内部渲染节点列表（用于手部网格）

    def clear_all(self):
        """清除场景中所有动态添加的物体和节点"""
        for actor in self.objects:
            self.scene.remove_actor(actor)  # 从场景中移除每个Actor
        for _ in range(len(self.objects)):
            actor = self.objects.pop()  # 弹出列表最后一个元素
            self.scene.remove_actor(actor)  # 从场景中移除
        self.clear_node()  # 清除所有渲染节点
        self.mano_layer = None  # 重置MANO层

    def clear_node(self):
        """清除所有内部渲染节点（用于手部网格显示）"""
        for _ in range(len(self.nodes)):
            node = self.nodes.pop()  # 弹出列表最后一个节点
            self.internal_scene.remove_node(node)  # 从内部渲染场景中移除节点

    def load_object_hand(self, data: Dict):
        """
        加载物体和手部数据
        参数:
            data: 包含ycb_ids、object_mesh_file、hand_shape、extrinsics的字典
        """
        ycb_ids = data["ycb_ids"]  # YCB物体ID列表
        ycb_mesh_files = data["object_mesh_file"]  # YCB物体网格文件路径列表
        hand_shape = data["hand_shape"]  # 手部形状参数（MANO模型的beta参数）
        extrinsic_mat = data["extrinsics"]  # 相机外参矩阵
        for ycb_id, ycb_mesh_file in zip(ycb_ids, ycb_mesh_files):
            self._load_ycb_object(ycb_id, ycb_mesh_file)  # 加载单个YCB物体

        self.mano_layer = MANOLayer("right", hand_shape.astype(np.float32))  # 创建右手MANO层
        self.mano_face = self.mano_layer.f.cpu().numpy()  # 获取手部网格面片索引
        pose_vec = pt.pq_from_transform(extrinsic_mat)  # 从变换矩阵提取位置+四元数
        self.camera_pose = sapien.Pose(pose_vec[0:3], pose_vec[3:7]).inv()  # 创建相机位姿并取逆

    def _load_ycb_object(self, ycb_id, ycb_mesh_file):
        """
        加载单个YCB物体
        参数:
            ycb_id: YCB物体ID
            ycb_mesh_file: 物体网格文件路径
        """
        builder = self.scene.create_actor_builder()  # 创建Actor构建器
        builder.add_visual_from_file(ycb_mesh_file)  # 从文件添加视觉模型
        actor = builder.build_static(name=YCB_CLASSES[ycb_id])  # 构建静态Actor
        self.objects.append(actor)  # 添加到物体列表

    def _compute_hand_geometry(self, hand_pose_frame, use_camera_frame=False):
        """
        计算手部几何形状（顶点和关节位置）
        参数:
            hand_pose_frame: 单帧手部姿态数据（48维姿态 + 3维平移）
            use_camera_frame: 是否使用相机坐标系
        返回:
            vertex: 手部网格顶点，joint: 手部关节位置
        """
        # pose parameters all zero, no hand is detected
        if np.abs(hand_pose_frame).sum() < 1e-5:
            return None, None  # 没有检测到手部
        p = torch.from_numpy(hand_pose_frame[:, :48].astype(np.float32))  # 前48维是姿态参数
        t = torch.from_numpy(hand_pose_frame[:, 48:51].astype(np.float32))  # 后3维是平移向量
        vertex, joint = self.mano_layer(p, t)  # MANO模型前向传播
        vertex = vertex.cpu().numpy()[0]  # 转换为NumPy数组
        joint = joint.cpu().numpy()[0]  # 转换为NumPy数组
        if not use_camera_frame:
            camera_mat = self.camera_pose.to_transformation_matrix()  # 获取相机位姿的4x4变换矩阵
            # 将顶点从相机坐标系变换到世界坐标系：vertex_world = vertex_cam @ R^T + t
            vertex = vertex @ camera_mat[:3, :3].T + camera_mat[:3, 3]
            vertex = np.ascontiguousarray(vertex)  # 确保内存连续
            joint = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]  # 同样变换关节
            joint = np.ascontiguousarray(joint)  # 确保内存连续

        return vertex, joint

    def _update_hand(self, vertex):
        """
        更新手部网格显示
        参数:
            vertex: 手部网格顶点数组
        """
        self.clear_node()  # 先清除之前的手部渲染节点
        normal = compute_smooth_shading_normal_np(vertex, self.mano_face)  # 计算顶点法线
        mesh = self.context.create_mesh_from_array(vertex, self.mano_face, normal)  # 创建渲染网格
        model = self.context.create_model([mesh], [self.mat_hand])  # 使用手部材质创建模型
        node = self.internal_scene.add_node()  # 添加渲染节点
        node.set_position(np.array([0, 0, 0]))  # 设置节点位置为原点
        obj = self.internal_scene.add_object(model, node)  # 将模型添加到节点
        obj.shading_mode = 0  # 平滑着色
        obj.cast_shadow = True  # 启用阴影投射
        obj.transparency = 0  # 完全不透明
        self.nodes.append(node)  # 保存节点引用

    def render_dexycb_data(self, data: Dict, fps=10):
        """
        渲染DexYCB数据集序列
        参数:
            data: 包含hand_pose和object_pose的字典
            fps: 播放帧率
        """
        hand_pose = data["hand_pose"]  # 手部姿态序列
        object_pose = data["object_pose"]  # 物体姿态序列
        frame_num = hand_pose.shape[0]  # 总帧数

        if self.headless:
            video_path = Path(__file__).parent.resolve() / "data/human_hand_video.mp4"  # 视频输出路径
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),  # MP4V编码器
                30.0,  # 30fps
                (self.camera.get_width(), self.camera.get_height()),  # 视频分辨率
            )  # 创建视频写入器

        step_per_frame = int(60 / fps)  # 每帧需要渲染的步数（基于60Hz刷新率）
        for i in trange(frame_num):
            object_pose_frame = object_pose[i]  # 当前帧物体姿态
            hand_pose_frame = hand_pose[i]  # 当前帧手部姿态
            vertex, _ = self._compute_hand_geometry(hand_pose_frame)  # 计算手部几何形状
            if vertex is not None:
                self._update_hand(vertex)  # 更新手部网格
            for k in range(len(self.objects)):
                pos_quat = object_pose_frame[k]  # 第k个物体的位置+四元数
                # 组合相机位姿和物体位姿，注意四元数从xyzw转换为wxyz
                pose = self.camera_pose * sapien.Pose(
                    pos_quat[4:], np.concatenate([pos_quat[3:4], pos_quat[:3]])
                )
                self.objects[k].set_pose(pose)  # 设置物体位姿
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
