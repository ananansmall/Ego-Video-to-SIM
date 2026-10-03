from pathlib import Path
from typing import Optional, List, Union, Dict

import cv2
import numpy as np
import sapien
import tqdm
import tyro
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.retargeting_config import RetargetingConfig

# Convert webp
# ffmpeg -i teaser.mp4 -vcodec libwebp -lossless 1 -loop 0 -preset default  -an -vsync 0 teaser.webp


def render_by_sapien(
    meta_data: Dict,
    data: List[Union[List[float], np.ndarray]],
    output_video_path: Optional[str] = None,
    headless: Optional[bool] = False,
):
    """
    使用SAPIEN引擎渲染机器人手部运动
    
    Args:
        meta_data: 元数据字典，包含配置路径、关节名称等信息
        data: 机器人关节角度列表，每帧一个关节角度数组
        output_video_path: 输出视频路径，如果为None则不录制视频
        headless: 是否无头模式（后台渲染），True为不显示窗口
    """
    # 根据headless参数选择渲染器：headless模式使用光线追踪，否则使用默认光栅化
    use_rt = headless
    if not use_rt:
        sapien.render.set_viewer_shader_dir("default")
        sapien.render.set_camera_shader_dir("default")
    else:
        sapien.render.set_viewer_shader_dir("rt")
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(16)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")

    # 加载配置文件以获取URDF路径和机器人名称
    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

    # 创建SAPIEN场景
    scene = sapien.Scene()

    # 添加地面并设置材质（颜色、金属度、粗糙度等）
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # 添加光源：方向光、点光源、环境光
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # 添加相机并设置位置和视角
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    # 创建Viewer窗口（非headless模式）
    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None
    record_video = output_video_path is not None

    # 加载URDF机器人模型
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    # 根据不同机器人类型设置缩放比例，使机器人在场景中大小合适
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    # 优先使用GLB格式（带纹理的模型）
    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)
    robot = loader.load(filepath)

    # 根据机器人类型设置初始位置，使其位于合适的高度
    if "ability" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.13]))
    elif "inspire" in robot_name:
        robot.set_pose(sapien.Pose([0, 0, -0.15]))

    # 初始化视频录制器
    if record_video:
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30.0,
            (cam.get_width(), cam.get_height()),
        )

    # 建立关节名称映射，处理不同系统的关节顺序差异
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    # 逐帧设置机器人关节角度并渲染
    for qpos in tqdm.tqdm(data):
        # 根据关节映射设置机器人关节角度
        robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        # 非headless模式：更新Viewer显示
        if not headless:
            for _ in range(2):
                viewer.render()
        # 录制视频模式：渲染场景并捕获图像
        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])  # RGB转BGR保存

    # 释放视频写入器
    if record_video:
        writer.release()

    scene = None


def main(
    pickle_path: str,
    output_video_path: Optional[str] = None,
    headless: bool = False,
):
    """
    加载保存的机器人姿态数据并渲染为屏幕显示或MP4视频
    
    Args:
        pickle_path: pickle文件路径，由detect_from_video.py生成
        output_video_path: 输出视频的.mp4文件路径，默认为None表示不保存视频
        headless: 是否无头模式，False会打开Viewer窗口进行可视化，True则后台渲染
    """
    # 设置URDF模型目录
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    # 加载pickle文件，提取元数据和关节角度数据
    pickle_data = np.load(pickle_path, allow_pickle=True)
    meta_data, data = pickle_data["meta_data"], pickle_data["data"]

    # 调用SAPIEN渲染函数
    render_by_sapien(meta_data, data, output_video_path, headless)


if __name__ == "__main__":
    tyro.cli(main)
