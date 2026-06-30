#!/usr/bin/env python3
"""
render_dexterous_only.py — 只渲染灵巧手 URDF (使用 dex-retargeting 优化器)

与 render_gripper_only.py 对应, 但渲染多指灵巧手:
  - allegro / inspire / shadow / ability / leap / svh
  - 标准 dex-retargeting config + add_dummy_free_joint=True
  - dummy free joint 自动求解腕部位姿 (无需解析计算)
  - 仅渲染手部 (无机械臂)

灵巧手位姿完全由 dex-retargeting PositionOptimizer 求解:
  优化器同时求解 6D 腕部位姿 (dummy free joint) + N 个手指关节,
  使机器人指尖链接位置匹配 MANO 目标关节位置。

用法:
    python hand_track/render_dexterous_only.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result \\
        --robot-name allegro

    # 切换为 5 指类人手
    python hand_track/render_dexterous_only.py ... --robot-name inspire
    python hand_track/render_dexterous_only.py ... --robot-name shadow
"""

import os
import subprocess
import sys
import time
import logging
import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
import sapien
import sapien.render
import torch
from pytransform3d import rotations as pr
from tqdm import trange

os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
COMBINATION_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = COMBINATION_DIR.parent.parent  # dex-retargeting/
sys.path.insert(0, str(PROJECT_ROOT / "example" / "position_retargeting"))
sys.path.insert(0, str(SCRIPT_DIR))

from dex_retargeting import yourdfpy as urdf
from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from mano_layer import MANOLayer

from common import (
    RXWORLD_TO_SAPIEN,
    detect_hands, load_hawor_data, load_hawor_c2w, setup_scene,
    load_glb_transformed, compute_mano_joints, _render_to_sapien,
    _render_keypoints, hawor_cam_to_sapien_pose, make_look_at_camera,
    _compute_wrist_positions_sapien,
)

# ── 支持的灵巧手 ──
SUPPORTED_ROBOTS = ["allegro", "inspire", "shadow", "ability", "leap", "svh"]
ROBOT_DEFAULT = "allegro"
HAWOR_FOCAL_DEFAULT = 600.0
WARMUP_FRAMES = 30


def _robot_name_to_enum(name: str) -> RobotName:
    """字符串机器人名称 → RobotName 枚举"""
    name = name.lower()
    mapping = {
        "allegro": RobotName.allegro,
        "inspire": RobotName.inspire,
        "shadow": RobotName.shadow,
        "ability": RobotName.ability,
        "leap": RobotName.leap,
        "svh": RobotName.svh,
    }
    if name not in mapping:
        raise ValueError(f"不支持的机器人: {name}, 可选: {SUPPORTED_ROBOTS}")
    return mapping[name]


def _load_dexterous_robot(robot_name_str: str, hand_type: HandType, scene, logger):
    """加载灵巧手 URDF + 构建 retargeting (参考 hand_robot_viewer.py)

    Args:
        robot_name_str: 机器人名称字符串 (allegro/inspire/shadow/...)
        hand_type: HandType.left 或 HandType.right
        scene: SAPIEN 场景
        logger: 日志器

    Returns:
        robot: SAPIEN Articulation
        retargeting: SeqRetargeting
        retarget2sapien: np.ndarray, retargeting qpos → sapien qpos 索引映射
        target_link_human_indices: 优化器目标链接对应的 MANO 关节索引
        config: RetargetingConfig
    """
    robot_name = _robot_name_to_enum(robot_name_str)

    # 设置默认 URDF 目录
    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    # 加载 config (强制 add_dummy_free_joint=True, 让优化器求解腕部6D位姿)
    config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)
    override = dict(add_dummy_free_joint=True)
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()

    # 加载 URDF 到 SAPIEN (优先 _glb 版本以获得正确视觉模型, 不存在则回退)
    urdf_path = Path(config.urdf_path)
    if "glb" not in urdf_path.stem:
        glb_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        if glb_path.exists():
            urdf_path = glb_path
        else:
            logger.info(f"    未找到 _glb 版本, 使用原始 URDF: {urdf_path.name}")
    robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True, build_scene_graph=False)
    urdf_name = urdf_path.name
    temp_dir = tempfile.mkdtemp(prefix="dex_retargeting-")
    temp_path = f"{temp_dir}/{urdf_name}"
    robot_urdf.write_xml_file(temp_path)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(temp_path)

    # 构建 retargeting qpos → sapien qpos 索引映射
    sapien_joint_names = [j.name for j in robot.get_active_joints()]
    retarget2sapien = np.array(
        [retargeting.joint_names.index(n) for n in sapien_joint_names]
    ).astype(int)

    target_link_human_indices = retargeting.optimizer.target_link_human_indices

    logger.info(f"  ✓ 灵巧手加载: {robot_name_str} ({hand_type.name})")
    logger.info(f"    URDF: {urdf_path.name}")
    logger.info(f"    SAPIEN 关节数: {len(sapien_joint_names)}")
    logger.info(f"    目标链接数: {len(target_link_human_indices)}, MANO 索引: {target_link_human_indices.tolist()}")

    return robot, retargeting, retarget2sapien, target_link_human_indices, config


def _compute_wrist_quat_sapien(hand_pose_frame):
    """从 MANO hand_pose 计算 SAPIEN 空间下的腕部四元数

    hand_pose_frame[0:3] 是 MANO 紧凑轴角表示的腕部全局旋转 (SLAM 空间)
    转换到 SAPIEN 空间: R_sapien = RXWORLD_TO_SAPIEN @ R_mano
    """
    R_mano = pr.matrix_from_compact_axis_angle(hand_pose_frame[0:3])
    R_sapien = RXWORLD_TO_SAPIEN @ R_mano
    return pr.quaternion_from_matrix(R_sapien)


def render_dexterous_only_video(hawor_dir, ras_dir, transform_params_path, output,
                                robot_name="allegro", hand_idx=1,
                                fps=30, cam_width=1920, cam_height=1080,
                                view="fpv", crf=18, start_frame=0, num_frames=-1,
                                smooth=1, viewer=False, verify=False, logger=None):
    """渲染灵巧手 URDF 视频 (单手)

    Args:
        robot_name: 灵巧手名称 (allegro/inspire/shadow/ability/leap/svh)
        hand_idx: 0=左手, 1=右手
        smooth: 0=不平滑, 1=使用 retargeting 内置 LPFilter
        viewer: True=交互式 SAPIEN Viewer, False=保存视频
        verify: True=计算指尖位置误差
    """
    if logger is None:
        logger = logging.getLogger("dexterous")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    prefix = "left" if hand_idx == 0 else "right"
    hand_type = HandType.left if hand_idx == 0 else HandType.right
    mode_label = "左手" if hand_idx == 0 else "右手"
    logger.info(f"灵巧手渲染: {mode_label}, robot={robot_name}")

    # ── 加载 HaWoR 数据 ──
    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    total_frames = hawor_data["pred_trans"].shape[0]
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame

    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)

    focal = hawor_data.get("img_focal")
    if focal is None or focal <= 0:
        focal = HAWOR_FOCAL_DEFAULT
    focal_render = focal * cam_width / 1280.0
    cam_fov = 2 * np.arctan(cam_height / 2.0 / focal_render)

    betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
    mano_layer = MANOLayer(prefix, betas_mean)

    # ── 创建场景 + GLB ──
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")

    # ── 加载灵巧手 ──
    robot, retargeting, retarget2sapien, target_link_human_indices, _ = _load_dexterous_robot(
        robot_name, hand_type, scene, logger)

    # 设置关节驱动 (用于 scene.step 更新)
    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=100000.0, damping=10000.0)

    scene.step()
    scene.update_render()

    # ── Warm start (用首帧 MANO 腕部位姿初始化, 避免局部最优) ──
    warm_started = False
    for probe_idx in range(num_frames):
        g_idx = start_frame + probe_idx
        if not hawor_data["pred_valid"][g_idx]:
            continue
        rot = hawor_data["pred_rot"][g_idx]
        trans = hawor_data["pred_trans"][g_idx]
        hand_pose = hawor_data["pred_hand_pose"][g_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        wrist_pos = joints_sapien[0, :3].astype(np.float64)
        wrist_quat = _compute_wrist_quat_sapien(hand_pose)
        retargeting.warm_start(
            wrist_pos, wrist_quat,
            hand_type=hand_type, is_mano_convention=False,
        )
        logger.info(f"  ✓ Warm start 完成 (帧 {g_idx})")
        warm_started = True
        break

    if not warm_started:
        logger.warning("  ⚠ 未找到有效帧进行 warm start, 使用默认位姿")

    # ── 探测首帧有效位姿 + Warmup 平滑过渡 ──
    first_valid_pos = None
    first_valid_quat = None
    for probe_idx in range(num_frames):
        g_idx = start_frame + probe_idx
        if not hawor_data["pred_valid"][g_idx]:
            continue
        rot = hawor_data["pred_rot"][g_idx]
        trans = hawor_data["pred_trans"][g_idx]
        hand_pose = hawor_data["pred_hand_pose"][g_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        ref_value = joints_sapien[target_link_human_indices, :3].astype(np.float32)
        qpos = retargeting.retarget(ref_value)[retarget2sapien]
        robot.set_qpos(qpos)
        scene.update_render()
        # 获取 root link 位姿用于 warmup 插值
        root_pose = robot.get_root_pose()
        first_valid_pos = np.array(root_pose.p)
        first_valid_quat = np.array(root_pose.q)
        break

    if first_valid_pos is not None:
        init_root_pos = np.zeros(3)
        init_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
        for wi in range(WARMUP_FRAMES):
            t = (wi + 1) / WARMUP_FRAMES
            t = t * t * (3 - 2 * t)
            interp_pos = init_root_pos * (1 - t) + first_valid_pos * t
            interp_quat = init_root_quat * (1 - t) + first_valid_quat * t
            norm = np.linalg.norm(interp_quat)
            if norm > 1e-8:
                interp_quat /= norm
            robot.set_root_pose(sapien.Pose(interp_pos.tolist(), interp_quat.tolist()))
            scene.step()
        logger.info(f"  Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep 过渡)")

    # ── 设置相机/Viewer ──
    sapien_viewer = None
    if viewer:
        from sapien.utils import Viewer
        sapien_viewer = Viewer()
        sapien_viewer.set_scene(scene)
        sapien_viewer.control_window.show_origin_frame = True
        sapien_viewer.control_window.show_grid = False
        camera = None
        logger.info("  模式: SAPIEN Viewer 实时循环 (不保存视频)")
    else:
        camera = scene.add_camera("dexterous", cam_width, cam_height, cam_fov, 0.01, 100.0)

    wrist_positions = _compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames)
    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])
    elif wrist_positions:
        centroid = np.mean(wrist_positions, axis=0)
        if view == "behind":
            cam_pos = centroid + np.array([2.5, 0.0, 1.2])
            cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
        elif view == "front":
            cam_pos = centroid + np.array([-2.5, 0.0, 1.2])
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

    # ── 视频写入器 ──
    writer = None
    if not viewer:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    # ── 验证误差记录 ──
    verify_errors = [] if verify else None
    if verify:
        logger.info("  验证模式: 开启 (计算指尖位置误差)")

    # ── 渲染循环 ──
    animation_loop = True
    while animation_loop:
        if not viewer:
            animation_loop = False

        for local_idx in trange(num_frames, desc=f"灵巧手-{robot_name}-{prefix}", disable=viewer):
            global_idx = start_frame + local_idx

            # 更新相机 (与 render_gripper_only.py 一致)
            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                if camera:
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                if sapien_viewer:
                    sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

            if not hawor_data["pred_valid"][global_idx]:
                for node in kp_nodes:
                    internal_scene.remove_node(node)
                kp_nodes.clear()
                scene.step()
                if viewer:
                    sapien_viewer.render()
                else:
                    scene.update_render()
                    camera.take_picture()
                    rgb = camera.get_picture("Color")[..., :3]
                    bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                    h, w = bgr.shape[:2]
                    cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  {robot_name} {prefix}  |  INVALID",
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    writer.write(bgr)
                continue

            rot = hawor_data["pred_rot"][global_idx]
            trans = hawor_data["pred_trans"][global_idx]
            hand_pose = hawor_data["pred_hand_pose"][global_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                for node in kp_nodes:
                    internal_scene.remove_node(node)
                kp_nodes.clear()
                scene.step()
                if viewer:
                    sapien_viewer.render()
                else:
                    scene.update_render()
                    camera.take_picture()
                    rgb = camera.get_picture("Color")[..., :3]
                    bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                    h, w = bgr.shape[:2]
                    cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                    cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  {robot_name} {prefix}  |  NaN",
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    writer.write(bgr)
                continue

            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)

            # 关键点标记 (在 MANO 目标关节处画绿球)
            kp_nodes = _render_keypoints(
                joints_sapien[:, :3], context, internal_scene, kp_nodes,
                target_link_human_indices, radius=0.012)

            # retargeting: 优化器同时求解 6D 腕部 (dummy free joint) + 手指关节
            ref_value = joints_sapien[target_link_human_indices, :3].astype(np.float32)
            qpos = retargeting.retarget(ref_value)[retarget2sapien]
            robot.set_qpos(qpos)

            # 验证误差
            if verify:
                scene.update_render()
                # 计算机器人指尖与 MANO 目标点的距离
                robot_link_positions = {}
                for link in robot.get_links():
                    robot_link_positions[link.get_name()] = np.array(link.get_entity_pose().p)
                target_link_names = retargeting.optimizer.target_link_names
                err = {}
                for k, (link_name, mano_idx) in enumerate(zip(target_link_names, target_link_human_indices)):
                    if link_name in robot_link_positions:
                        robot_pos = robot_link_positions[link_name]
                        mano_pos = joints_sapien[mano_idx, :3]
                        err[f'link{k}_mm'] = float(np.linalg.norm(robot_pos - mano_pos) * 1000)
                verify_errors.append(err)
            else:
                scene.step()

            if viewer:
                sapien_viewer.render()
            else:
                scene.update_render()
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  {robot_name} {prefix}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(bgr)

        if viewer:
            logger.info("  动画播放完成, 重新开始... (关闭窗口退出)")
            init_qpos = robot.get_qpos().copy()
            robot.set_qpos(init_qpos)

    if not viewer:
        writer.release()

        # ffmpeg 重编码
        final_path = output
        tmp_path = str(output).replace(".mp4", "_tmp.mp4")
        if os.path.exists(str(output)):
            os.rename(str(output), tmp_path)
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                import subprocess as sp
                cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
                       "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
                       "-movflags", "+faststart", str(final_path)]
                result = sp.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    os.remove(tmp_path)
                else:
                    if os.path.exists(tmp_path):
                        os.rename(tmp_path, final_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, final_path)

        logger.info(f"\n✓ 灵巧手视频已保存: {final_path}")
    else:
        final_path = None
        logger.info("\n✓ Viewer 模式结束")

    # ── 验证误差报告 ──
    if verify and verify_errors:
        logger.info("\n  === 验证误差报告 (指尖位置) ===")
        for k in range(len(target_link_human_indices)):
            key = f'link{k}_mm'
            vals = [e[key] for e in verify_errors if key in e]
            if vals:
                mean_v = np.mean(vals)
                max_v = np.max(vals)
                logger.info(f"  目标链接 {k}: mean={mean_v:.2f}mm, max={max_v:.2f}mm")

    return final_path


def render_dual_dexterous_video(hawor_dir, ras_dir, transform_params_path, output,
                                robot_name="allegro",
                                fps=30, cam_width=1920, cam_height=1080,
                                view="fpv", crf=18, start_frame=0, num_frames=-1,
                                smooth=1, viewer=False, verify=False, logger=None,
                                hand_indices=None):
    """在同一场景中渲染左右灵巧手 URDF (双手, 一个视频)"""
    if logger is None:
        logger = logging.getLogger("dual_dexterous")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    logger.info(f"双灵巧手渲染: 同一场景, robot={robot_name}")

    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)

    if hand_indices is None:
        hand_indices = [0, 1]

    # ── 加载左右手数据 + 灵巧手 ──
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")

    hand_states = []
    for hi in hand_indices:
        prefix = "left" if hi == 0 else "right"
        hand_type = HandType.left if hi == 0 else HandType.right

        hawor_data = load_hawor_data(hawor_dir, hand_idx=hi)
        total_frames = hawor_data["pred_trans"].shape[0]
        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_layer = MANOLayer(prefix, betas_mean)

        robot, retargeting, retarget2sapien, target_link_human_indices, _ = _load_dexterous_robot(
            robot_name, hand_type, scene, logger)

        for joint in robot.get_active_joints():
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        hand_states.append({
            "prefix": prefix, "hand_idx": hi, "hand_type": hand_type,
            "hawor_data": hawor_data, "mano_layer": mano_layer,
            "total_frames": total_frames,
            "robot": robot, "retargeting": retargeting,
            "retarget2sapien": retarget2sapien,
            "target_link_human_indices": target_link_human_indices,
        })

    total_frames = min(hs["total_frames"] for hs in hand_states)
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame

    scene.step()
    scene.update_render()

    # ── Warm start (各自用首帧 MANO 腕部位姿初始化) ──
    for hs in hand_states:
        hawor_data = hs["hawor_data"]
        mano_layer = hs["mano_layer"]
        retargeting = hs["retargeting"]
        hand_type = hs["hand_type"]
        prefix = hs["prefix"]
        for probe_idx in range(num_frames):
            g_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][g_idx]:
                continue
            rot = hawor_data["pred_rot"][g_idx]
            trans = hawor_data["pred_trans"][g_idx]
            hand_pose = hawor_data["pred_hand_pose"][g_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                continue
            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)
            wrist_pos = joints_sapien[0, :3].astype(np.float64)
            wrist_quat = _compute_wrist_quat_sapien(hand_pose)
            retargeting.warm_start(
                wrist_pos, wrist_quat,
                hand_type=hand_type, is_mano_convention=False,
            )
            logger.info(f"  ✓ {prefix} Warm start 完成 (帧 {g_idx})")
            break

    # ── 设置相机/Viewer ──
    focal = hand_states[0]["hawor_data"].get("img_focal")
    if focal is None or focal <= 0:
        focal = HAWOR_FOCAL_DEFAULT
    focal_render = focal * cam_width / 1280.0
    cam_fov = 2 * np.arctan(cam_height / 2.0 / focal_render)

    sapien_viewer = None
    if viewer:
        from sapien.utils import Viewer
        sapien_viewer = Viewer()
        sapien_viewer.set_scene(scene)
        sapien_viewer.control_window.show_origin_frame = True
        sapien_viewer.control_window.show_grid = False
        camera = None
        logger.info("  模式: SAPIEN Viewer 实时循环 (不保存视频)")
    else:
        camera = scene.add_camera("dual_dexterous", cam_width, cam_height, cam_fov, 0.01, 100.0)

    all_wrist_positions = []
    for hs in hand_states:
        wp = _compute_wrist_positions_sapien(hs["hawor_data"], hs["mano_layer"], start_frame, num_frames)
        all_wrist_positions.extend(wp)

    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])
    elif all_wrist_positions:
        centroid = np.mean(all_wrist_positions, axis=0)
        if view == "behind":
            cam_pos = centroid + np.array([2.5, 0.0, 1.2])
            cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
        elif view == "front":
            cam_pos = centroid + np.array([-2.5, 0.0, 1.2])
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
        if camera:
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        if sapien_viewer:
            sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

    # ── 视频写入器 ──
    writer = None
    if not viewer:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    verify_errors = [] if verify else None
    if verify:
        logger.info("  验证模式: 开启 (计算指尖位置误差)")

    # ── 渲染循环 ──
    animation_loop = True
    while animation_loop:
        if not viewer:
            animation_loop = False

        for local_idx in trange(num_frames, desc=f"双灵巧手-{robot_name}", disable=viewer):
            global_idx = start_frame + local_idx

            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
                if camera:
                    camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                if sapien_viewer:
                    sapien_viewer.set_camera_xyz(x=cam_pos[0], y=cam_pos[1], z=cam_pos[2])

            for node in kp_nodes:
                internal_scene.remove_node(node)
            kp_nodes.clear()

            left_valid = False
            right_valid = False

            for hs in hand_states:
                hawor_data = hs["hawor_data"]
                prefix = hs["prefix"]
                robot = hs["robot"]
                retargeting = hs["retargeting"]
                mano_layer = hs["mano_layer"]
                retarget2sapien = hs["retarget2sapien"]
                target_link_human_indices = hs["target_link_human_indices"]

                if not hawor_data["pred_valid"][global_idx]:
                    continue

                rot = hawor_data["pred_rot"][global_idx]
                trans = hawor_data["pred_trans"][global_idx]
                hand_pose = hawor_data["pred_hand_pose"][global_idx]
                if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                    continue

                _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
                joints_sapien = _render_to_sapien(j)

                clear_kp = (prefix == "left")
                kp_nodes = _render_keypoints(
                    joints_sapien[:, :3], context, internal_scene, kp_nodes,
                    target_link_human_indices, radius=0.012, clear_existing=clear_kp)

                ref_value = joints_sapien[target_link_human_indices, :3].astype(np.float32)
                qpos = retargeting.retarget(ref_value)[retarget2sapien]
                robot.set_qpos(qpos)

                if verify:
                    scene.update_render()
                    robot_link_positions = {}
                    for link in robot.get_links():
                        robot_link_positions[link.get_name()] = np.array(link.get_entity_pose().p)
                    target_link_names = retargeting.optimizer.target_link_names
                    err = {'prefix': prefix}
                    for k, (link_name, mano_idx) in enumerate(zip(target_link_names, target_link_human_indices)):
                        if link_name in robot_link_positions:
                            robot_pos = robot_link_positions[link_name]
                            mano_pos = joints_sapien[mano_idx, :3]
                            err[f'{prefix}_link{k}_mm'] = float(np.linalg.norm(robot_pos - mano_pos) * 1000)
                    verify_errors.append(err)
                else:
                    scene.step()

                if prefix == "left":
                    left_valid = True
                else:
                    right_valid = True

            if not verify:
                scene.step()

            if viewer:
                sapien_viewer.render()
            else:
                scene.update_render()
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

                h, w = bgr.shape[:2]
                cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
                hand_info = f"L:{'Y' if left_valid else 'N'} R:{'Y' if right_valid else 'N'}"
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Dual {robot_name}  |  {hand_info}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(bgr)

        if viewer:
            logger.info("  动画播放完成, 重新开始... (关闭窗口退出)")
            for hs in hand_states:
                robot = hs["robot"]
                init_qpos = robot.get_qpos().copy()
                robot.set_qpos(init_qpos)

    if not viewer:
        writer.release()

        final_path = output
        tmp_path = str(output).replace(".mp4", "_tmp.mp4")
        if os.path.exists(str(output)):
            os.rename(str(output), tmp_path)
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                import subprocess as sp
                cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
                       "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
                       "-movflags", "+faststart", str(final_path)]
                result = sp.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0 and os.path.exists(final_path) and os.path.getsize(final_path) > 0:
                    os.remove(tmp_path)
                else:
                    if os.path.exists(tmp_path):
                        os.rename(tmp_path, final_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, final_path)

        logger.info(f"\n✓ 双灵巧手视频已保存: {final_path}")
    else:
        final_path = None
        logger.info("\n✓ Viewer 模式结束")

    # ── 验证误差报告 ──
    if verify and verify_errors:
        logger.info("\n  === 验证误差报告 (指尖位置) ===")
        for pfx in ['left', 'right']:
            for hs in hand_states:
                if hs['prefix'] != pfx:
                    continue
                num_links = len(hs['target_link_human_indices'])
                for k in range(num_links):
                    key = f'{pfx}_link{k}_mm'
                    vals = [e[key] for e in verify_errors if key in e]
                    if vals:
                        mean_v = np.mean(vals)
                        max_v = np.max(vals)
                        logger.info(f"  [{pfx}] 链接 {k}: mean={mean_v:.2f}mm, max={max_v:.2f}mm")
                break

    return final_path


def _ensure_transform_params(ras_dir, hawor_dir, output_dir, logger):
    """确保 transform_params.npz 存在, 不存在则自动运行 01_align_scene.py"""
    tp_path = os.path.join(output_dir, "transform_params.npz")
    if os.path.exists(tp_path):
        logger.info(f"  transform_params 已存在: {tp_path}")
        return tp_path

    logger.info(f"  transform_params 不存在, 自动运行 01_align_scene.py ...")
    hawor_recon_dir = os.path.join(hawor_dir, "reconstruction")
    if not os.path.isdir(hawor_recon_dir):
        logger.error(f"  hawor reconstruction 目录不存在: {hawor_recon_dir}")
        return None

    recon_npz = None
    for f in os.listdir(hawor_recon_dir):
        if f.startswith("hawor_results_") and f.endswith(".npz"):
            recon_npz = os.path.join(hawor_recon_dir, f)
            break
    if recon_npz is None:
        logger.error(f"  未找到 hawor_results_*.npz: {hawor_recon_dir}")
        return None

    align_script = str(COMBINATION_DIR / "01_align_scene.py")
    cmd = [sys.executable, align_script, "--ras_output", ras_dir,
           "--hawor_reconstruction", recon_npz, "--output_dir", output_dir]
    logger.info(f"  运行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  01_align_scene.py 失败:\n{result.stderr[-500:]}")
        return None

    if os.path.exists(tp_path):
        logger.info(f"  ✓ transform_params 已生成: {tp_path}")
        return tp_path
    logger.error("  01_align_scene.py 运行完成但未生成 transform_params.npz")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="只渲染灵巧手 URDF (使用 dex-retargeting 优化器, 无机械臂)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hawor-dir", type=str, required=True, help="HaWoR 数据目录")
    parser.add_argument("--ras-dir", type=str, required=True, help="RAS 重建结果目录")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: hand_track/output/{hawor_name})")
    parser.add_argument("--robot-name", type=str, default=ROBOT_DEFAULT,
                        choices=SUPPORTED_ROBOTS,
                        help=f"灵巧手名称 (默认 {ROBOT_DEFAULT})")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--num-frames", type=int, default=-1, help="处理帧数 (-1=全部)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--width", type=int, default=1920, help="视频宽度")
    parser.add_argument("--height", type=int, default=1080, help="视频高度")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF 质量参数")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角 (默认 fpv)")
    parser.add_argument("--smooth", type=int, default=1,
                        choices=[0, 1],
                        help="平滑模式: 0=不平滑, 1=retargeting 内置 LPFilter (默认 1)")
    parser.add_argument("--viewer", action="store_true",
                        help="使用 SAPIEN Viewer 实时循环播放 (不保存视频)")
    parser.add_argument("--verify", action="store_true",
                        help="计算并输出指尖位置误差")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        choices=[-1, 0, 1],
                        help="手部索引: -1=自动检测, 0=左手, 1=右手 (默认 -1)")
    args = parser.parse_args()

    # Logger
    logger = logging.getLogger("DexterousOnly")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)

    # 输出目录
    hawor_name = Path(args.hawor_dir).name
    if args.output_dir is None:
        args.output_dir = str(SCRIPT_DIR / "output" / hawor_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # [1] 检测手部
    if args.hand_idx >= 0:
        hand_indices = [args.hand_idx]
        hand_label = "左手" if args.hand_idx == 0 else "右手"
        logger.info(f"[1/3] 手部指定: {hand_label} (index={args.hand_idx})")
    else:
        hand_indices = detect_hands(args.hawor_dir)
        hand_count = len(hand_indices)
        if hand_count == 0:
            logger.error("[1/3] 手部检测: 未检测到有效手部数据, 停止生成")
            sys.exit(1)
        hand_label = "双手" if hand_count == 2 else ("左手" if hand_indices[0] == 0 else "右手")
        logger.info(f"[1/3] 手部检测: {hand_label} (indices={hand_indices})")

    # [2] 确保 transform_params 存在
    logger.info(f"\n[2/3] 准备 GLB 变换参数 ...")
    tp_path = _ensure_transform_params(args.ras_dir, args.hawor_dir, args.output_dir, logger)
    if tp_path is None:
        logger.warning("  无法获取 transform_params, 将不渲染 GLB 物体")

    # [3] 渲染
    logger.info(f"\n[3/3] 渲染灵巧手视频 (robot={args.robot_name}, smooth={args.smooth}, viewer={args.viewer}, verify={args.verify}) ...")
    start_time = time.time()

    if len(hand_indices) == 1:
        hi = hand_indices[0]
        side = "left" if hi == 0 else "right"
        output_video = os.path.join(args.output_dir, "videos", f"hawor_{args.robot_name}_{side}_urdf.mp4")
        os.makedirs(os.path.dirname(output_video), exist_ok=True)

        render_dexterous_only_video(
            hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
            transform_params_path=tp_path, output=output_video,
            robot_name=args.robot_name, hand_idx=hi,
            fps=args.fps, cam_width=args.width, cam_height=args.height,
            view=args.view, crf=args.crf, start_frame=args.start_frame,
            num_frames=args.num_frames, smooth=args.smooth,
            viewer=args.viewer, verify=args.verify, logger=logger,
        )
    else:
        output_video = os.path.join(args.output_dir, "videos", f"hawor_{args.robot_name}_dual_urdf.mp4")
        os.makedirs(os.path.dirname(output_video), exist_ok=True)

        render_dual_dexterous_video(
            hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
            transform_params_path=tp_path, output=output_video,
            robot_name=args.robot_name,
            fps=args.fps, cam_width=args.width, cam_height=args.height,
            view=args.view, crf=args.crf, start_frame=args.start_frame,
            num_frames=args.num_frames, smooth=args.smooth,
            viewer=args.viewer, verify=args.verify, logger=logger,
            hand_indices=hand_indices,
        )

    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
