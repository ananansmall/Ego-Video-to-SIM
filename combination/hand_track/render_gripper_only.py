#!/usr/bin/env python3
"""
render_gripper_only.py — 只渲染夹爪URDF (不加载手臂)

与 render_auto.py 类似, 但只加载夹爪部分的 URDF:
  - gripper_base_link (固定根, 无mesh)
  - gripper_link (夹爪本体 mesh)
  - gripper_finger_link1/2 (两个手指 mesh, prismatic joint)

夹爪位姿直接从 retargeting FK 获取, 不需要 IK, 不需要手臂底座。
排除手臂底座不确定性的干扰, 只看夹爪跟踪效果。

用法:
    python hand_track/render_gripper_only.py \\
        --hawor-dir /home/an/data/hawor/7 \\
        --ras-dir /home/an/data/ras/my_7mp4_result
"""

import os
import subprocess
import sys
import time
import logging
import argparse
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

from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from mano_layer import MANOLayer

# 从 gripper_config 导入夹爪配置
from gripper_config import (
    RXWORLD_TO_SAPIEN, R1_MESH_DIR,
    WARMUP_FRAMES, LP_ALPHA_POS, LP_ALPHA_ORI, LP_ALPHA_ANALYTICAL, GRIPPER_INIT_OPEN,
    FINGER_BASE_DIST, FINGER_GEOM_ARRAYS, GRIPPER_JOINT_GEOM,
    EmaTargetSmoother, PositionEmaSmoother,
    generate_gripper_urdf, prepare_full_arm_urdf, prepare_half_arm_urdf,
    compute_analytical_gripper_pose, compute_gripper_offset_in_root,
    init_gripper_retargeting,
)
# 新对齐策略 (用户要求: 先对齐夹爪两点, 再用中点-手腕连线确定位姿)
from align_strategy import (
    compute_gripper_pose_aligned, compute_arm_root_pose,
    verify_alignment, print_verification, GRIPPER_OPEN_SCALE,
)

HAWOR_FOCAL_DEFAULT = 600.0


def _compute_gripper_pose_by_strategy(strategy, mano_wrist, mano_finger1, mano_finger2,
                                       prefix, finger_origin_x, finger1_origin_x, finger2_origin_x,
                                       open_scale=GRIPPER_OPEN_SCALE):
    """根据对齐策略选择计算函数 (单/双手通用)

    strategy="aligned": 新策略 (先对齐夹爪两点 + 中点手腕连线确定位姿, 带开合缩放)
    strategy="analytical": 旧策略 (Gram-Schmidt)
    """
    if strategy == "aligned":
        return compute_gripper_pose_aligned(
            mano_wrist, mano_finger1, mano_finger2, prefix, open_scale=open_scale)
    else:
        return compute_analytical_gripper_pose(
            mano_wrist, mano_finger1, mano_finger2, prefix, finger_origin_x,
            finger1_origin_x=finger1_origin_x, finger2_origin_x=finger2_origin_x)

from common import (
    detect_hands, load_hawor_data, load_hawor_c2w, setup_scene,
    load_glb_transformed, compute_mano_joints, _render_to_sapien,
    _render_keypoints, hawor_cam_to_sapien_pose, make_look_at_camera,
    _compute_wrist_positions_sapien, _get_gripper_pose_from_retargeting,
)


def _ensure_transform_params(ras_dir, hawor_dir, output_dir, logger):
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


def _combine_videos_side_by_side(left_video, right_video, output, fps, crf, logger):
    """将左右视频并排合成一个视频"""
    cap_l = cv2.VideoCapture(left_video)
    cap_r = cv2.VideoCapture(right_video)
    w_l = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_l = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_r = int(cap_r.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_r = int(cap_r.get(cv2.CAP_PROP_FRAME_HEIGHT))
    h_out = max(h_l, h_r)
    w_out = w_l + w_r
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w_out, h_out))
    frame_idx = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l and not ret_r:
            break
        if not ret_l:
            frame_l = np.zeros((h_l, w_l, 3), dtype=np.uint8)
        if not ret_r:
            frame_r = np.zeros((h_r, w_r, 3), dtype=np.uint8)
        if frame_l.shape[0] < h_out:
            pad = np.zeros((h_out - frame_l.shape[0], frame_l.shape[1], 3), dtype=np.uint8)
            frame_l = np.vstack([frame_l, pad])
        if frame_r.shape[0] < h_out:
            pad = np.zeros((h_out - frame_r.shape[0], frame_r.shape[1], 3), dtype=np.uint8)
            frame_r = np.vstack([frame_r, pad])
        combined = np.hstack([frame_l, frame_r])
        cv2.rectangle(combined, (0, 0), (w_l, 40), (0, 0, 0), -1)
        cv2.rectangle(combined, (w_l, 0), (w_out, 40), (0, 0, 0), -1)
        cv2.putText(combined, "Left Gripper", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Right Gripper", (w_l + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(combined)
        frame_idx += 1
    cap_l.release()
    cap_r.release()
    writer.release()
    # ffmpeg 重编码
    tmp_path = str(output).replace(".mp4", "_tmp.mp4")
    if os.path.exists(str(output)):
        os.rename(str(output), tmp_path)
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
                   "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
                   "-movflags", "+faststart", str(output)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
                os.remove(tmp_path)
            else:
                if os.path.exists(tmp_path):
                    os.rename(tmp_path, output)
        except Exception:
            if os.path.exists(tmp_path):
                os.rename(tmp_path, output)
    logger.info(f"  ✓ 双夹爪合成视频: {output} ({frame_idx} 帧)")


def render_gripper_only_video(hawor_dir, ras_dir, transform_params_path, output,
                              hand_idx=1, fps=30, cam_width=1920, cam_height=1080,
                              view="fpv", crf=18, start_frame=0, num_frames=-1,
                              with_arm=False, smooth=1, viewer=False, verify=False,
                              analytical=True, arm_mode="half", logger=None,
                              strategy="aligned", open_scale=GRIPPER_OPEN_SCALE):
    """渲染只有夹爪URDF的视频 (不加载手臂)

    Args:
        analytical: True=解析模式 (Gram-Schmidt, 有投影误差)
                    False=优化器模式 (dex_retargeting PositionOptimizer, gripper-only URDF, 3点约束)
        arm_mode: gripper_arm 模式的手臂类型: "half"=link4-6, "full"=link1-6
        strategy: 对齐策略 "aligned"=新策略(先对齐夹爪两点+中点手腕连线), "analytical"=旧策略
        open_scale: 夹爪开合缩放因子 (仅 aligned 策略使用)
    """
    import logging
    if logger is None:
        logger = logging.getLogger("gripper_only")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    prefix = "left" if hand_idx == 0 else "right"
    mode_label = "左手" if hand_idx == 0 else "右手"
    logger.info(f"夹爪URDF渲染: {mode_label}")

    # ── 加载数据 ──
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

    # ── finger_origin_x: 始终使用 URDF 原始 37mm (与 02_render_scene.py 一致, 不缩放) ──
    # 02_render_scene.py 从不缩放 finger origin, 始终使用 r1_lite_robot_glb.urdf 的原始 0.03689
    # 我们也保持一致, 所有模式 (gripper / gripper_arm) 都使用 37mm
    finger_origin_x = 0.03689
    finger1_origin_x = 0.03689
    finger2_origin_x = 0.03689
    logger.info(f"  finger_origin_x = {finger_origin_x*1000:.1f}mm (URDF 原始值, 不缩放, 与 02_render_scene.py 一致)")

    # ── 初始化 Retargeting ──
    if analytical:
        # 解析模式: 仍需 ref_indices 用于关键点标记, 用 gripper-only 配置
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        retargeting, ref_indices, _ = init_gripper_retargeting(prefix, finger_origin_x, PROJECT_ROOT)
        fixed_qpos = np.zeros(0, dtype=np.float32)  # 解析模式不需要 fixed_qpos
        strat_label = "新对齐策略(先对齐夹爪两点+中点手腕连线)" if strategy == "aligned" else "旧策略(Gram-Schmidt)"
        logger.info(f"  模式: 解析 ({strat_label}, open_scale={open_scale})")
    else:
        # 优化器模式: 用 gripper-only URDF + 3 target links, 每个手指独立缩放
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))
        retargeting, ref_indices, _ = init_gripper_retargeting(
            prefix, finger_origin_x, PROJECT_ROOT,
            finger1_origin_x=finger1_origin_x,
            finger2_origin_x=finger2_origin_x,
        )
        # gripper-only URDF 没有 arm 关节, 不需要 fixed_qpos
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        logger.info(f"  模式: 优化器 (gripper-only URDF, 8 DOF = 6 dummy + 2 finger, 3 target points, 独立缩放)")

    # ── 对齐策略选择 ──
    def _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2):
        """根据 strategy 选择对齐策略计算夹爪位姿"""
        return _compute_gripper_pose_by_strategy(
            strategy, mano_wrist, mano_finger1, mano_finger2,
            prefix, finger_origin_x, finger1_origin_x, finger2_origin_x, open_scale)

    # ── 创建场景 ──
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")

    # ── 加载夹爪 URDF ──
    if with_arm:
        if arm_mode == "full":
            gripper_urdf_path = prepare_full_arm_urdf(prefix)
            logger.info(f"  渲染URDF: 完整手臂+夹爪 (arm_link1-6 + gripper, 不缩放, 参考 GalaxeaManipSim)")
        else:
            gripper_urdf_path = prepare_half_arm_urdf(prefix)
            logger.info(f"  渲染URDF: 半个手臂+夹爪 (arm_link4-6 + gripper, 不缩放, 参考 GalaxeaManipSim)")
    else:
        gripper_urdf_path = generate_gripper_urdf(
            prefix, finger_origin_x,
            finger1_origin_x=finger1_origin_x, finger2_origin_x=finger2_origin_x,
        )
        logger.info(f"  渲染URDF: 仅夹爪")
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(gripper_urdf_path)

    joint_names = [j.name for j in robot.get_active_joints()]
    logger.info(f"  夹爪关节: {joint_names}")

    gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
    gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")
    arm_joint_indices = [i for i, n in enumerate(joint_names) if 'arm_joint' in n]

    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=100000.0, damping=10000.0)

    init_qpos = robot.get_qpos().copy()
    init_qpos[gripper_idx1] = GRIPPER_INIT_OPEN
    init_qpos[gripper_idx2] = GRIPPER_INIT_OPEN
    robot.set_qpos(init_qpos)

    # retargeting → sapien qpos 映射
    sapien_name_to_retarget_idx = {
        n: retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names
    }

    scene.step()
    scene.update_render()

    # 计算 gripper_link 相对于 root 的 offset
    if with_arm and arm_joint_indices:
        qpos_now = robot.get_qpos().copy()
        for ai in arm_joint_indices:
            qpos_now[ai] = 0.0
        robot.set_qpos(qpos_now)
        scene.update_render()
    if with_arm:
        gripper_offset_pos, gripper_offset_R = compute_gripper_offset_in_root(robot, prefix)
    else:
        gripper_offset_pos = np.zeros(3)
        gripper_offset_R = np.eye(3)

    # ── Warm start retargeting (用解析位姿初始化, 避免局部最优) ──
    hand_type = HandType.left if hand_idx == 0 else HandType.right
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
        mano_wrist = joints_sapien[0, :3]
        mano_finger1 = joints_sapien[ref_indices[0], :3]
        mano_finger2 = joints_sapien[ref_indices[1], :3]
        g_pos, g_R, joint1, joint2 = _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2)
        g_quat = pr.quaternion_from_matrix(g_R)
        retargeting.warm_start(
            g_pos, g_quat,
            hand_type=hand_type, is_mano_convention=False,
        )
        # 用解析关节值初始化手指
        finger_j1_name = f"{prefix}_gripper_finger_joint1"
        finger_j2_name = f"{prefix}_gripper_finger_joint2"
        for num, jname in enumerate(retargeting.optimizer.target_joint_names):
            if jname == finger_j1_name:
                retargeting.last_qpos[num] = joint1
            elif jname == finger_j2_name:
                retargeting.last_qpos[num] = joint2
        logger.info(f"  ✓ Warm start 完成 (帧 {g_idx}), j1={joint1:.4f}, j2={joint2:.4f}")
        break

    # ── 探测首帧有效位姿 ──
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
        if analytical:
            mano_wrist = joints_sapien[0, :3]
            mano_finger1 = joints_sapien[ref_indices[0], :3]
            mano_finger2 = joints_sapien[ref_indices[1], :3]
            g_pos, g_R, _, _ = _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2)
            root_R = g_R @ gripper_offset_R.T
            root_pos = g_pos - root_R @ gripper_offset_pos
            first_valid_pos = root_pos.copy()
            first_valid_quat = pr.quaternion_from_matrix(root_R)
        else:
            ref_value = joints_sapien[ref_indices, :3]
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            g_pos_fk, g_R_fk = _get_gripper_pose_from_retargeting(
                retargeting, retarget_qpos, prefix)
            root_R = g_R_fk @ gripper_offset_R.T
            root_pos = g_pos_fk - root_R @ gripper_offset_pos
            first_valid_pos = root_pos.copy()
            first_valid_quat = pr.quaternion_from_matrix(root_R)
        break

    # ── Warmup smoothstep ──
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

    # ── EMA 平滑器 ──
    target_smoother = None
    mano_smoother = None
    if smooth == 1:
        if analytical:
            mano_smoother = PositionEmaSmoother(alpha=LP_ALPHA_ANALYTICAL)
            logger.info(f"  平滑模式: MANO 输入位置 EMA (alpha={LP_ALPHA_ANALYTICAL})")
        else:
            target_smoother = EmaTargetSmoother()
            logger.info(f"  平滑模式: root pose EMA (pos_alpha={LP_ALPHA_POS}, ori_alpha={LP_ALPHA_ORI})")
    else:
        logger.info(f"  平滑模式: 不平滑")

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
        camera = scene.add_camera("gripper", cam_width, cam_height, cam_fov, 0.01, 100.0)

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
        logger.info("  验证模式: 开启 (计算指尖位置 + 手腕位姿误差)")

    # ── 渲染循环 ──
    animation_loop = True
    while animation_loop:
        if not viewer:
            animation_loop = False

        for local_idx in trange(num_frames, desc=f"夹爪URDF-{prefix}", disable=viewer):
            global_idx = start_frame + local_idx

            # 更新相机 (与 02_render_scene.py 一致: 有相机轨迹时始终每帧更新)
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
                    cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper {prefix}  |  INVALID",
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
                    cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper {prefix}  |  NaN",
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    writer.write(bgr)
                continue

            _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)

            # 关键点标记
            kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012)

            if analytical:
                mano_wrist = joints_sapien[0, :3]
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                if mano_smoother is not None:
                    mano_pts = np.stack([mano_wrist, mano_finger1, mano_finger2])
                    mano_pts = mano_smoother.smooth(mano_pts)
                    mano_wrist, mano_finger1, mano_finger2 = mano_pts[0], mano_pts[1], mano_pts[2]
                g_pos, g_R, joint1, joint2 = _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2)
                root_R = g_R @ gripper_offset_R.T
                root_pos = g_pos - root_R @ gripper_offset_pos
                root_quat = pr.quaternion_from_matrix(root_R)
                robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))
                qpos = robot.get_qpos().copy()
                for arm_idx in arm_joint_indices:
                    qpos[arm_idx] = 0.0
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                robot.set_qpos(qpos)
            else:
                # 优化器模式: gripper-only URDF, 8 DOF, 3 target points
                ref_value = joints_sapien[ref_indices, :3]
                retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

                # 从优化器 FK 获取 gripper_link 位姿
                g_pos_fk, g_R_fk = _get_gripper_pose_from_retargeting(
                    retargeting, retarget_qpos, prefix)

                # 从优化器 qpos 提取手指关节值
                joint1 = retarget_qpos[sapien_name_to_retarget_idx[f"{prefix}_gripper_finger_joint1"]]
                joint2 = retarget_qpos[sapien_name_to_retarget_idx[f"{prefix}_gripper_finger_joint2"]]

                # 从 gripper_link 位姿转换为 root 位姿
                root_R = g_R_fk @ gripper_offset_R.T
                root_pos = g_pos_fk - root_R @ gripper_offset_pos
                root_quat = pr.quaternion_from_matrix(root_R)

                # EMA 平滑
                if target_smoother is not None:
                    root_pos, root_quat = target_smoother.smooth(root_pos, root_quat)

                robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))

                qpos = robot.get_qpos().copy()
                for arm_idx in arm_joint_indices:
                    qpos[arm_idx] = 0.0
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                robot.set_qpos(qpos)

            # 验证误差
            if verify:
                scene.update_render()
                finger1_pos = None
                finger2_pos = None
                gripper_link_pos = None
                gripper_link_R = None
                for link in robot.get_links():
                    lname = link.get_name()
                    if lname == f"{prefix}_gripper_finger_link1":
                        finger1_pos = np.array(link.get_entity_pose().p)
                    elif lname == f"{prefix}_gripper_finger_link2":
                        finger2_pos = np.array(link.get_entity_pose().p)
                    elif lname == f"{prefix}_gripper_link":
                        pose = link.get_entity_pose()
                        gripper_link_pos = np.array(pose.p)
                        gripper_link_R = pr.matrix_from_quaternion(np.array(pose.q))
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                mano_wrist_pos = joints_sapien[0, :3]

                err = {}
                if finger1_pos is not None:
                    err['finger1_mm'] = float(np.linalg.norm(finger1_pos - mano_finger1) * 1000)
                if finger2_pos is not None:
                    err['finger2_mm'] = float(np.linalg.norm(finger2_pos - mano_finger2) * 1000)
                if gripper_link_pos is not None:
                    err['wrist_pos_mm'] = float(np.linalg.norm(gripper_link_pos - mano_wrist_pos) * 1000)
                    mano_pointing = ((mano_finger1 + mano_finger2) / 2 - mano_wrist_pos)
                    mano_pointing_norm = np.linalg.norm(mano_pointing)
                    if mano_pointing_norm > 1e-6:
                        mano_pointing = mano_pointing / mano_pointing_norm
                        gripper_x = gripper_link_R[:, 0]
                        pointing_cos = np.clip(np.dot(gripper_x, mano_pointing), -1, 1)
                        err['pointing_deg'] = float(np.degrees(np.arccos(pointing_cos)))
                    mano_opening = mano_finger2 - mano_finger1
                    mano_opening_norm = np.linalg.norm(mano_opening)
                    if mano_opening_norm > 1e-6:
                        y_sign = 1.0 if prefix == "right" else -1.0
                        gripper_x = gripper_link_R[:, 0]
                        # 投影到指向方向垂直面，与夹爪y轴定义一致
                        mano_opening_proj = mano_opening - np.dot(mano_opening, gripper_x) * gripper_x
                        mano_opening_proj_norm = np.linalg.norm(mano_opening_proj)
                        if mano_opening_proj_norm > 1e-6:
                            mano_opening_dir = y_sign * mano_opening_proj / mano_opening_proj_norm
                            gripper_y = gripper_link_R[:, 1]
                            opening_cos = np.clip(np.dot(gripper_y, mano_opening_dir), -1, 1)
                            err['opening_deg'] = float(np.degrees(np.arccos(opening_cos)))
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
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper URDF {prefix}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(bgr)

        if viewer:
            logger.info("  动画播放完成, 重新开始... (关闭窗口退出)")
            init_qpos = robot.get_qpos().copy()
            init_qpos[gripper_idx1] = GRIPPER_INIT_OPEN
            init_qpos[gripper_idx2] = GRIPPER_INIT_OPEN
            robot.set_qpos(init_qpos)
            if target_smoother:
                target_smoother.reset()
            if mano_smoother:
                mano_smoother.reset()

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

        logger.info(f"\n✓ 夹爪URDF视频已保存: {final_path}")
    else:
        final_path = None
        logger.info("\n✓ Viewer 模式结束")

    # ── 验证误差报告 ──
    if verify and verify_errors:
        logger.info("\n  === 验证误差报告 ===")
        for key in ['finger1_mm', 'finger2_mm', 'wrist_pos_mm', 'pointing_deg', 'opening_deg']:
            vals = [e[key] for e in verify_errors if key in e]
            if vals:
                mean_v = np.mean(vals)
                max_v = np.max(vals)
                unit = 'mm' if 'mm' in key else 'deg'
                label = {
                    'finger1_mm': '指尖1位置误差', 'finger2_mm': '指尖2位置误差',
                    'wrist_pos_mm': '手腕位置误差',
                    'pointing_deg': '指向方向误差', 'opening_deg': '开合方向误差',
                }[key]
                logger.info(f"  {label}: mean={mean_v:.2f}{unit}, max={max_v:.2f}{unit}")

    return final_path


def render_dual_gripper_video(hawor_dir, ras_dir, transform_params_path, output,
                               fps=30, cam_width=1920, cam_height=1080,
                               view="fpv", crf=18, start_frame=0, num_frames=-1,
                               with_arm=False, smooth=1, viewer=False, verify=False,
                               analytical=True, arm_mode="half", logger=None, hand_indices=None,
                               strategy="aligned", open_scale=GRIPPER_OPEN_SCALE):
    """在同一场景中渲染左右夹爪URDF (双手, 一个视频)"""
    import logging
    if logger is None:
        logger = logging.getLogger("dual_gripper")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    logger.info(f"双夹爪URDF渲染: 同一场景")

    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)

    robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    gripper_states = []
    if hand_indices is None:
        hand_indices = [0, 1]
    for hi in hand_indices:
        prefix = "left" if hi == 0 else "right"
        hand_type = HandType.left if hi == 0 else HandType.right

        hawor_data = load_hawor_data(hawor_dir, hand_idx=hi)
        total_frames = hawor_data["pred_trans"].shape[0]

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_layer = MANOLayer(prefix, betas_mean)

        gripper_states.append({
            "prefix": prefix, "hand_idx": hi, "hand_type": hand_type,
            "hawor_data": hawor_data, "mano_layer": mano_layer,
            "total_frames": total_frames,
            "finger_origin_x": 0.03689,
        })

    total_frames = min(gs["total_frames"] for gs in gripper_states)
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame

    # finger_origin_x: 始终使用 URDF 原始 37mm (与 02_render_scene.py 一致, 不缩放)
    for gs in gripper_states:
        gs["finger1_origin_x"] = 0.03689
        gs["finger2_origin_x"] = 0.03689
        gs["finger_origin_x"] = 0.03689
        logger.info(f"  {gs['prefix']} finger_origin_x = 37mm (URDF 原始值, 不缩放, 与 02_render_scene.py 一致)")

    # 初始化 retargeting (gripper-only URDF, 独立缩放)
    for gs in gripper_states:
        prefix = gs["prefix"]
        fox = gs["finger_origin_x"]
        f1x = gs["finger1_origin_x"]
        f2x = gs["finger2_origin_x"]

        retargeting, ref_indices, _ = init_gripper_retargeting(
            prefix, fox, PROJECT_ROOT,
            finger1_origin_x=f1x, finger2_origin_x=f2x,
        )
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)

        gs["retargeting"] = retargeting
        gs["ref_indices"] = ref_indices
        gs["fixed_qpos"] = fixed_qpos
        logger.info(f"  {prefix} 优化器: gripper-only URDF, finger1={f1x*1000:.1f}mm, finger2={f2x*1000:.1f}mm")

    focal = gripper_states[0]["hawor_data"].get("img_focal")
    if focal is None or focal <= 0:
        focal = HAWOR_FOCAL_DEFAULT
    focal_render = focal * cam_width / 1280.0
    cam_fov = 2 * np.arctan(cam_height / 2.0 / focal_render)

    # ── 创建场景 + GLB ──
    scene = setup_scene()

    glb_path = Path(ras_dir) / "final_scene.glb"
    has_transform = transform_params_path is not None and Path(transform_params_path).exists()
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(glb_path, transform_params_path, scene, logger)
        if obj_actors:
            logger.info(f"  ✓ GLB 加载成功: {len(obj_actors)} 个物体")

    # ── 加载左右夹爪 URDF ──
    for gs in gripper_states:
        prefix = gs["prefix"]
        fox = gs["finger_origin_x"]
        f1x = gs["finger1_origin_x"]
        f2x = gs["finger2_origin_x"]
        if with_arm:
            if arm_mode == "full":
                gripper_urdf_path = prepare_full_arm_urdf(prefix)
                logger.info(f"  {prefix} 渲染URDF: 完整手臂+夹爪 (arm_link1-6 + gripper, 不缩放)")
            else:
                gripper_urdf_path = prepare_half_arm_urdf(prefix)
                logger.info(f"  {prefix} 渲染URDF: 半个手臂+夹爪 (arm_link4-6 + gripper, 不缩放)")
        else:
            gripper_urdf_path = generate_gripper_urdf(
                prefix, fox,
                finger1_origin_x=f1x, finger2_origin_x=f2x,
            )
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        robot = loader.load(gripper_urdf_path)

        joint_names = [j.name for j in robot.get_active_joints()]
        gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
        gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

        for joint in robot.get_active_joints():
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        init_qpos = robot.get_qpos().copy()
        init_qpos[gripper_idx1] = GRIPPER_INIT_OPEN
        init_qpos[gripper_idx2] = GRIPPER_INIT_OPEN
        robot.set_qpos(init_qpos)

        retargeting = gs["retargeting"]
        sapien_name_to_retarget_idx = {
            n: retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names
        }

        gs["robot"] = robot
        gs["gripper_idx1"] = gripper_idx1
        gs["gripper_idx2"] = gripper_idx2
        gs["sapien_name_to_retarget_idx"] = sapien_name_to_retarget_idx
        gs["joint_names"] = joint_names
        gs["arm_joint_indices"] = [i for i, n in enumerate(joint_names) if 'arm_joint' in n]
        logger.info(f"  ✓ {prefix} 夹爪已加载: {joint_names}")

    scene.step()
    scene.update_render()

    # 计算 gripper_link offset
    for gs in gripper_states:
        prefix = gs["prefix"]
        robot = gs["robot"]
        if with_arm:
            arm_idx = gs.get("arm_joint_indices", [])
            if arm_idx:
                qpos_now = robot.get_qpos().copy()
                for ai in arm_idx:
                    qpos_now[ai] = 0.0
                robot.set_qpos(qpos_now)
                scene.update_render()
            offset_pos, offset_R = compute_gripper_offset_in_root(robot, prefix)
        else:
            offset_pos = np.zeros(3)
            offset_R = np.eye(3)
        gs["gripper_offset_pos"] = offset_pos
        gs["gripper_offset_R"] = offset_R

    # ── Warm start (用解析位姿初始化, 避免局部最优) ──
    for gs in gripper_states:
        hawor_data = gs["hawor_data"]
        mano_layer = gs["mano_layer"]
        retargeting = gs["retargeting"]
        hand_type = gs["hand_type"]
        ref_indices = gs["ref_indices"]
        prefix = gs["prefix"]
        fox = gs["finger_origin_x"]
        f1x = gs["finger1_origin_x"]
        f2x = gs["finger2_origin_x"]
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
            mano_wrist = joints_sapien[0, :3]
            mano_finger1 = joints_sapien[ref_indices[0], :3]
            mano_finger2 = joints_sapien[ref_indices[1], :3]
            g_pos, g_R, joint1, joint2 = _compute_gripper_pose_by_strategy(
                strategy, mano_wrist, mano_finger1, mano_finger2,
                prefix, fox, f1x, f2x, open_scale)
            g_quat = pr.quaternion_from_matrix(g_R)
            retargeting.warm_start(
                g_pos, g_quat,
                hand_type=hand_type, is_mano_convention=False,
            )
            # 用解析关节值初始化手指
            finger_j1_name = f"{prefix}_gripper_finger_joint1"
            finger_j2_name = f"{prefix}_gripper_finger_joint2"
            for num, jname in enumerate(retargeting.optimizer.target_joint_names):
                if jname == finger_j1_name:
                    retargeting.last_qpos[num] = joint1
                elif jname == finger_j2_name:
                    retargeting.last_qpos[num] = joint2
            logger.info(f"  ✓ {prefix} Warm start 完成 (帧 {g_idx}), j1={joint1:.4f}, j2={joint2:.4f}")
            break

    # ── 探测首帧 + Warmup ──
    for gs in gripper_states:
        hawor_data = gs["hawor_data"]
        mano_layer = gs["mano_layer"]
        retargeting = gs["retargeting"]
        prefix = gs["prefix"]
        ref_indices = gs["ref_indices"]
        fixed_qpos = gs["fixed_qpos"]
        gripper_offset_pos = gs["gripper_offset_pos"]
        gripper_offset_R = gs["gripper_offset_R"]
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
            if analytical:
                mano_wrist = joints_sapien[0, :3]
                mano_finger1 = joints_sapien[ref_indices[0], :3]
                mano_finger2 = joints_sapien[ref_indices[1], :3]
                g_pos, g_R, _, _ = _compute_gripper_pose_by_strategy(
                    strategy, mano_wrist, mano_finger1, mano_finger2,
                    prefix, gs["finger_origin_x"], gs["finger1_origin_x"], gs["finger2_origin_x"], open_scale)
                root_R = g_R @ gripper_offset_R.T
                root_pos = g_pos - root_R @ gripper_offset_pos
                first_valid_pos = root_pos.copy()
                first_valid_quat = pr.quaternion_from_matrix(root_R)
            else:
                ref_value = joints_sapien[ref_indices, :3].astype(np.float32)
                retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
                g_pos_fk, g_R_fk = _get_gripper_pose_from_retargeting(
                    retargeting, retarget_qpos, prefix)
                root_R = g_R_fk @ gripper_offset_R.T
                root_pos = g_pos_fk - root_R @ gripper_offset_pos
                first_valid_pos = root_pos.copy()
                first_valid_quat = pr.quaternion_from_matrix(root_R)
            break
        if first_valid_pos is not None:
            init_root_pos = np.zeros(3)
            init_root_quat = np.array([1.0, 0.0, 0.0, 0.0])
            robot = gs["robot"]
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
            logger.info(f"  ✓ {prefix} Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep)")
        if smooth == 1:
            if analytical:
                gs["mano_smoother"] = PositionEmaSmoother(alpha=LP_ALPHA_ANALYTICAL)
                gs["target_smoother"] = None
            else:
                gs["mano_smoother"] = None
                gs["target_smoother"] = EmaTargetSmoother()
        else:
            gs["mano_smoother"] = None
            gs["target_smoother"] = None

    if smooth == 1:
        if analytical:
            logger.info(f"  平滑模式: MANO 输入位置 EMA (alpha={LP_ALPHA_ANALYTICAL})")
        else:
            logger.info(f"  平滑模式: root pose EMA (pos_alpha={LP_ALPHA_POS}, ori_alpha={LP_ALPHA_ORI})")
    else:
        logger.info(f"  平滑模式: 不平滑")

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
        camera = scene.add_camera("dual_gripper", cam_width, cam_height, cam_fov, 0.01, 100.0)

    all_wrist_positions = []
    for gs in gripper_states:
        wp = _compute_wrist_positions_sapien(gs["hawor_data"], gs["mano_layer"], start_frame, num_frames)
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
        logger.info("  验证模式: 开启 (计算指尖位置 + 手腕位姿误差)")

    # ── 渲染循环 ──
    animation_loop = True
    while animation_loop:
        if not viewer:
            animation_loop = False

        for local_idx in trange(num_frames, desc="双夹爪URDF", disable=viewer):
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

            for gs in gripper_states:
                hawor_data = gs["hawor_data"]
                prefix = gs["prefix"]
                robot = gs["robot"]
                retargeting = gs["retargeting"]
                mano_layer = gs["mano_layer"]
                ref_indices = gs["ref_indices"]
                fixed_qpos = gs["fixed_qpos"]
                target_smoother = gs.get("target_smoother")

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
                kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012, clear_existing=clear_kp)

                if analytical:
                    mano_wrist = joints_sapien[0, :3]
                    mano_finger1 = joints_sapien[ref_indices[0], :3]
                    mano_finger2 = joints_sapien[ref_indices[1], :3]
                    mano_smoother = gs.get("mano_smoother")
                    if mano_smoother is not None:
                        mano_pts = np.stack([mano_wrist, mano_finger1, mano_finger2])
                        mano_pts = mano_smoother.smooth(mano_pts)
                        mano_wrist, mano_finger1, mano_finger2 = mano_pts[0], mano_pts[1], mano_pts[2]
                    g_pos, g_R, joint1, joint2 = _compute_gripper_pose_by_strategy(
                        strategy, mano_wrist, mano_finger1, mano_finger2,
                        prefix, gs["finger_origin_x"], gs["finger1_origin_x"], gs["finger2_origin_x"], open_scale)
                    gripper_offset_pos = gs["gripper_offset_pos"]
                    gripper_offset_R = gs["gripper_offset_R"]
                    root_R = g_R @ gripper_offset_R.T
                    root_pos = g_pos - root_R @ gripper_offset_pos
                    root_quat = pr.quaternion_from_matrix(root_R)
                    robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))
                    qpos = robot.get_qpos().copy()
                    for arm_idx in gs.get("arm_joint_indices", []):
                        qpos[arm_idx] = 0.0
                    qpos[gs["gripper_idx1"]] = float(joint1)
                    qpos[gs["gripper_idx2"]] = float(joint2)
                    robot.set_qpos(qpos)
                else:
                    ref_value = joints_sapien[ref_indices, :3].astype(np.float32)
                    retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)

                    g_pos_fk, g_R_fk = _get_gripper_pose_from_retargeting(
                        retargeting, retarget_qpos, prefix)

                    sapien_name_to_retarget_idx = gs["sapien_name_to_retarget_idx"]
                    joint1 = retarget_qpos[sapien_name_to_retarget_idx[f"{prefix}_gripper_finger_joint1"]]
                    joint2 = retarget_qpos[sapien_name_to_retarget_idx[f"{prefix}_gripper_finger_joint2"]]

                    gripper_offset_pos = gs["gripper_offset_pos"]
                    gripper_offset_R = gs["gripper_offset_R"]
                    root_R = g_R_fk @ gripper_offset_R.T
                    root_pos = g_pos_fk - root_R @ gripper_offset_pos
                    root_quat = pr.quaternion_from_matrix(root_R)

                    if target_smoother is not None:
                        root_pos, root_quat = target_smoother.smooth(root_pos, root_quat)

                    robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))

                    qpos = robot.get_qpos().copy()
                    for arm_idx in gs.get("arm_joint_indices", []):
                        qpos[arm_idx] = 0.0
                    qpos[gs["gripper_idx1"]] = float(joint1)
                    qpos[gs["gripper_idx2"]] = float(joint2)
                    robot.set_qpos(qpos)

                if verify:
                    scene.update_render()
                    for link in robot.get_links():
                        lname = link.get_name()
                        if lname == f"{prefix}_gripper_finger_link1":
                            finger1_pos = np.array(link.get_entity_pose().p)
                        elif lname == f"{prefix}_gripper_finger_link2":
                            finger2_pos = np.array(link.get_entity_pose().p)
                        elif lname == f"{prefix}_gripper_link":
                            pose = link.get_entity_pose()
                            gripper_link_pos = np.array(pose.p)
                            gripper_link_R = pr.matrix_from_quaternion(np.array(pose.q))
                    mano_finger1 = joints_sapien[ref_indices[0], :3]
                    mano_finger2 = joints_sapien[ref_indices[1], :3]
                    mano_wrist_pos = joints_sapien[0, :3]
                    err = {'prefix': prefix}
                    err[f'{prefix}_finger1_mm'] = float(np.linalg.norm(finger1_pos - mano_finger1) * 1000)
                    err[f'{prefix}_finger2_mm'] = float(np.linalg.norm(finger2_pos - mano_finger2) * 1000)
                    err[f'{prefix}_wrist_pos_mm'] = float(np.linalg.norm(gripper_link_pos - mano_wrist_pos) * 1000)

                    mano_pointing = ((mano_finger1 + mano_finger2) / 2 - mano_wrist_pos)
                    mano_pointing_norm = np.linalg.norm(mano_pointing)
                    if mano_pointing_norm > 1e-6:
                        mano_pointing_dir = mano_pointing / mano_pointing_norm
                        gripper_x = gripper_link_R[:, 0]
                        pointing_cos = np.clip(np.dot(gripper_x, mano_pointing_dir), -1, 1)
                        err[f'{prefix}_pointing_deg'] = float(np.degrees(np.arccos(pointing_cos)))
                    mano_opening = mano_finger2 - mano_finger1
                    mano_opening_norm = np.linalg.norm(mano_opening)
                    if mano_opening_norm > 1e-6:
                        y_sign = 1.0 if prefix == "right" else -1.0
                        gripper_x = gripper_link_R[:, 0]
                        # 投影到指向方向垂直面，与夹爪y轴定义一致
                        mano_opening_proj = mano_opening - np.dot(mano_opening, gripper_x) * gripper_x
                        mano_opening_proj_norm = np.linalg.norm(mano_opening_proj)
                        if mano_opening_proj_norm > 1e-6:
                            mano_opening_dir = y_sign * mano_opening_proj / mano_opening_proj_norm
                            gripper_y = gripper_link_R[:, 1]
                            opening_cos = np.clip(np.dot(gripper_y, mano_opening_dir), -1, 1)
                            err[f'{prefix}_opening_deg'] = float(np.degrees(np.arccos(opening_cos)))
                    qpos_now = robot.get_qpos()
                    err[f'{prefix}_joint1'] = float(qpos_now[gs["gripper_idx1"]])
                    err[f'{prefix}_joint2'] = float(qpos_now[gs["gripper_idx2"]])
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
                cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Dual Gripper  |  {hand_info}",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                writer.write(bgr)

        if viewer:
            logger.info("  动画播放完成, 重新开始... (关闭窗口退出)")
            for gs in gripper_states:
                robot = gs["robot"]
                init_qpos = robot.get_qpos().copy()
                init_qpos[gs["gripper_idx1"]] = GRIPPER_INIT_OPEN
                init_qpos[gs["gripper_idx2"]] = GRIPPER_INIT_OPEN
                robot.set_qpos(init_qpos)
                if gs.get("target_smoother"):
                    gs["target_smoother"].reset()
                if gs.get("mano_smoother"):
                    gs["mano_smoother"].reset()

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

        logger.info(f"\n✓ 双夹爪URDF视频已保存: {final_path}")
    else:
        final_path = None
        logger.info("\n✓ Viewer 模式结束")

    # ── 验证误差报告 ──
    if verify and verify_errors:
        logger.info("\n  === 验证误差报告 ===")
        for pfx in ['left', 'right']:
            for suffix, label, unit in [('_finger1_mm', '指尖1位置误差', 'mm'),
                                         ('_finger2_mm', '指尖2位置误差', 'mm'),
                                         ('_wrist_pos_mm', '手腕位置误差', 'mm'),
                                         ('_pointing_deg', '指向方向误差', 'deg'),
                                         ('_opening_deg', '开合方向误差', 'deg')]:
                key = f'{pfx}{suffix}'
                vals = [e[key] for e in verify_errors if key in e]
                if vals:
                    mean_v = np.mean(vals)
                    max_v = np.max(vals)
                    logger.info(f"  [{pfx}] {label}: mean={mean_v:.2f}{unit}, max={max_v:.2f}{unit}")
            j1_vals = [e[f'{pfx}_joint1'] for e in verify_errors if f'{pfx}_joint1' in e]
            j2_vals = [e[f'{pfx}_joint2'] for e in verify_errors if f'{pfx}_joint2' in e]
            if j1_vals:
                logger.info(f"  [{pfx}] 手指关节: joint1=[{min(j1_vals):.4f}, {max(j1_vals):.4f}], "
                            f"joint2=[{min(j2_vals):.4f}, {max(j2_vals):.4f}]")
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="只渲染夹爪URDF (不加载手臂)",
    )
    parser.add_argument("--hawor-dir", type=str, required=True, help="HaWoR 数据目录")
    parser.add_argument("--ras-dir", type=str, required=True, help="RAS 重建结果目录")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: hand_track/output/{hawor_name})")
    parser.add_argument("--start-frame", type=int, default=0, help="起始帧")
    parser.add_argument("--num-frames", type=int, default=-1, help="处理帧数 (-1=全部)")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率")
    parser.add_argument("--width", type=int, default=1920, help="视频宽度")
    parser.add_argument("--height", type=int, default=1080, help="视频高度")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF 质量参数")
    parser.add_argument("--view", type=str, default="fpv",
                        choices=["fpv", "topdown", "behind", "front"],
                        help="相机视角 (默认 fpv)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["gripper", "gripper_arm", "both"],
                        help="渲染模式: gripper=仅夹爪, gripper_arm=夹爪+手臂, both=两者都渲染 (默认)")
    parser.add_argument("--arm-mode", type=str, default="half",
                        choices=["half", "full"],
                        help="gripper_arm 模式的手臂类型: half=半个手臂(link4-6), full=完整手臂(link1-6) (默认 half)")
    parser.add_argument("--smooth", type=int, default=1,
                        choices=[0, 1],
                        help="平滑模式: 0=不平滑, 1=EMA平滑 (默认 1)")
    parser.add_argument("--viewer", action="store_true",
                        help="使用 SAPIEN Viewer 实时循环播放 (不保存视频)")
    parser.add_argument("--verify", action="store_true",
                        help="计算并输出指尖位置/手腕位姿误差")
    parser.add_argument("--optimizer", action="store_true",
                        help="使用优化器模式 (默认: 解析模式; 加此参数使用 dex_retargeting PositionOptimizer + gripper-only URDF, 3点约束精确跟踪)")
    parser.add_argument("--strategy", type=str, default="aligned",
                        choices=["aligned", "analytical"],
                        help="对齐策略: aligned=新策略(先对齐夹爪两点+中点手腕连线确定位姿, 默认), analytical=旧策略(Gram-Schmidt)")
    parser.add_argument("--open-scale", type=float, default=GRIPPER_OPEN_SCALE,
                        help=f"夹爪开合缩放因子 (默认 {GRIPPER_OPEN_SCALE}, 放大开合效果; 1.0=精确映射)")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        choices=[-1, 0, 1],
                        help="手部索引: -1=自动检测, 0=左手, 1=右手 (默认 -1)")
    args = parser.parse_args()

    # Logger
    logger = logging.getLogger("GripperOnly")
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

    # [1] 检测手部 (自动检测或用户指定)
    if args.hand_idx >= 0:
        hand_indices = [args.hand_idx]
        hand_label = "左手" if args.hand_idx == 0 else "右手"
        logger.info(f"[1/3] 手部指定: {hand_label} (index={args.hand_idx})")
    else:
        hand_indices = detect_hands(args.hawor_dir)
        hand_count = len(hand_indices)
        if hand_count == 0:
            logger.error("[1/3] 手部检测: 未检测到有效手部数据 (pred_valid 全为 False 或持续为 NaN), 停止生成")
            sys.exit(1)
        hand_label = "双手" if hand_count == 2 else ("左手" if hand_indices[0] == 0 else "右手")
        logger.info(f"[1/3] 手部检测: {hand_label} (indices={hand_indices})")

    # [2] 确保 transform_params 存在
    logger.info(f"\n[2/3] 准备 GLB 变换参数 ...")
    tp_path = _ensure_transform_params(args.ras_dir, args.hawor_dir, args.output_dir, logger)

    # [3] 渲染
    modes_to_render = []
    if args.mode in ("gripper", "both"):
        modes_to_render.append(("gripper", False, ""))
    if args.mode in ("gripper_arm", "both"):
        modes_to_render.append(("gripper_arm", True, "_arm"))

    analytical = not args.optimizer
    logger.info(f"\n[3/3] 渲染夹爪URDF视频 (mode={args.mode}, strategy={args.strategy}, open_scale={args.open_scale}, smooth={args.smooth}, viewer={args.viewer}, verify={args.verify}, analytical={analytical}) ...")
    start_time = time.time()

    for mode_name, with_arm, mode_suffix in modes_to_render:
        logger.info(f"\n--- 渲染模式: {mode_name} ---")

        if len(hand_indices) == 1:
            hi = hand_indices[0]
            side = "left" if hi == 0 else "right"
            output_video = os.path.join(args.output_dir, "videos", f"hawor_r1_{side}_gripper_urdf{mode_suffix}.mp4")
            os.makedirs(os.path.dirname(output_video), exist_ok=True)

            render_gripper_only_video(
                hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                transform_params_path=tp_path, output=output_video,
                hand_idx=hi, fps=args.fps, cam_width=args.width, cam_height=args.height,
                view=args.view, crf=args.crf, start_frame=args.start_frame,
                num_frames=args.num_frames, with_arm=with_arm,
                smooth=args.smooth, viewer=args.viewer, verify=args.verify,
                analytical=analytical, arm_mode=args.arm_mode, logger=logger,
                strategy=args.strategy, open_scale=args.open_scale,
            )
        else:
            output_video = os.path.join(args.output_dir, "videos", f"hawor_r1_dual_gripper_urdf{mode_suffix}.mp4")
            os.makedirs(os.path.dirname(output_video), exist_ok=True)

            render_dual_gripper_video(
                hawor_dir=args.hawor_dir, ras_dir=args.ras_dir,
                transform_params_path=tp_path, output=output_video,
                fps=args.fps, cam_width=args.width, cam_height=args.height,
                view=args.view, crf=args.crf, start_frame=args.start_frame,
                num_frames=args.num_frames, with_arm=with_arm,
                smooth=args.smooth, viewer=args.viewer, verify=args.verify,
                analytical=analytical, arm_mode=args.arm_mode, logger=logger,
                hand_indices=hand_indices,
                strategy=args.strategy, open_scale=args.open_scale,
            )

    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
