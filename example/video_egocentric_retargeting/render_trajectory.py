import pickle
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sapien
import tqdm
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from dex_retargeting.retargeting_config import RetargetingConfig


ROBOT_SCALE_MAP = {
    "ability": 1.5,
    "dclaw": 1.25,
    "allegro": 1.4,
    "shadow": 0.9,
    "bhand": 1.5,
    "leap": 1.4,
    "svh": 1.5,
    "inspire": 1.0,
}

ROBOT_POS_MAP = {
    "ability": [0, 0, -0.15],
    "shadow": [0, 0, -0.2],
    "dclaw": [0, 0, -0.15],
    "allegro": [0, 0, -0.05],
    "bhand": [0, 0, -0.2],
    "leap": [0, 0, -0.15],
    "svh": [0, 0, -0.13],
    "inspire": [0, 0, -0.15],
}


def render_trajectory(
    pickle_path: str,
    output_video: Optional[str] = None,
    headless: bool = False,
    fps: int = 30,
):
    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    with open(pickle_path, "rb") as f:
        result = pickle.load(f)

    meta_data = result["meta_data"]
    data = result["data"]
    wrist_trajectory = result.get("wrist_trajectory", [])

    config_path = meta_data["config_path"]
    config = RetargetingConfig.load_from_file(config_path)

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

    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )

    cam = scene.add_camera(name="main", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    if not headless:
        viewer = Viewer()
        viewer.set_scene(scene)
        viewer.control_window.show_origin_frame = False
        viewer.control_window.move_speed = 0.01
        viewer.control_window.toggle_camera_lines(False)
        viewer.set_camera_pose(cam.get_local_pose())
    else:
        viewer = None

    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True

    for key, scale in ROBOT_SCALE_MAP.items():
        if key in robot_name:
            loader.scale = scale
            break

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    robot = loader.load(filepath)

    for key, pos in ROBOT_POS_MAP.items():
        if key in robot_name:
            robot.set_pose(sapien.Pose(pos))
            break

    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = meta_data["joint_names"]
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    record_video = output_video is not None
    if record_video:
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (cam.get_width(), cam.get_height()),
        )

    valid_data = [d for d in data if d is not None]
    logger.info(f"Rendering {len(valid_data)} valid frames out of {len(data)} total")

    for qpos in tqdm.tqdm(data, desc="Rendering"):
        if qpos is not None:
            robot.set_qpos(np.array(qpos)[retargeting_to_sapien])

        if not headless:
            for _ in range(2):
                viewer.render()

        if record_video:
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            writer.write(rgb[..., ::-1])

    if record_video:
        writer.release()
        logger.info(f"Video saved: {output_video}")


def main(
    pickle_path: str,
    output_video: Optional[str] = None,
    headless: bool = False,
    fps: int = 30,
):
    render_trajectory(pickle_path, output_video, headless, fps)


if __name__ == "__main__":
    tyro.cli(main)
