#!/usr/bin/env python3
"""
002_render_scene.py — 新坐标系渲染脚本 (与 001_align_scene.py 配套)

继承 02_render_scene.py 全部功能, 集成自动手部检测 (单/双手),
使用新坐标系: 仿真坐标系 = GLB 原始 RAS 坐标系。

坐标哲学:
  旧 02: GLB → ZUP_TO_YUP → s_inv(R_inv@v + t_inv) → RXWORLD_TO_SAPIEN
  新 002: GLB → (1/s)*R_h2g.T@(v-t_h2g) → RXWORLD_TO_SAPIEN  (同样到 1× SAPIEN 空间)
         手部/相机: _render_to_sapien / hawor_cam_to_sapien_pose (与 02 完全一致)

用法:
    python 002_render_scene.py \\
        --hawor-dir data/hawor \\
        --ras-dir data/ras \\
        --glb-path data/ras/final_scene.glb \\
        --transform-params data/hawor/transform_params.npz \\
        --hand-idx 0 \\
        --mode robot_tracking \\
        --output /tmp/test.mp4
"""

import os
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
if os.path.exists(_nvidia_icd):
    os.environ['VK_ICD_FILENAMES'] = _nvidia_icd
else:
    _intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
    os.environ['VK_ICD_FILENAMES'] = _intel_icd

import sys
import logging
import argparse
import subprocess as sp
import time
from pathlib import Path

import cv2
import numpy as np
import sapien
import sapien.render
import torch
from pytransform3d import rotations as pr
from tqdm import trange
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch3d")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "dex-retargeting" / "example" / "position_retargeting"))

from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.optimizer_utils import LPFilter
from mano_layer import MANOLayer

GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "hand_track"))

from hand_track.common import (
    load_hawor_data, load_hawor_c2w,
    load_ras_cameras, ras_cam_to_sapien_pose,
    compute_mano_joints, detect_hands,
    setup_scene, prepare_arm_urdf,
    load_glb_to_sapien, hawor_cam_to_sapien_pose, _render_to_sapien,
    set_render_transform_params,
    RXWORLD_TO_SAPIEN, R_AXIS,
    _render_keypoints, _get_gripper_pose_from_retargeting,
    _compute_optimal_fixed_base, _compute_tracking_base_pos,
    IK_SOLVE_PER_FRAME, WARMUP_FRAMES, HAWOR_FOCAL_DEFAULT,
    make_look_at_camera,
    FLOATING_RIGHT_URDF, FLOATING_LEFT_URDF,
    R1_RIGHT_SETTINGS, R1_LEFT_SETTINGS,
    RIGHT_ARM_STARTING, LEFT_ARM_STARTING,
    ARM_MAX_REACH, COMFORTABLE_REACH, COMFORT_TARGET_IN_BASE,
    BASE_TRACKING_RANGE, LP_ALPHA_JOINT,
    R1_MESH_DIR,
)
from hand_track.gripper_config import (
    generate_gripper_urdf, prepare_full_arm_urdf, prepare_half_arm_urdf,
    LP_ALPHA_POS, LP_ALPHA_ORI, LP_ALPHA_ANALYTICAL, GRIPPER_INIT_OPEN,
    EmaTargetSmoother, PositionEmaSmoother,
    compute_analytical_gripper_pose, compute_gripper_offset_in_root,
    init_gripper_retargeting, compute_mano_based_gripper_pose,
)
from hand_track.align_strategy import (
    compute_gripper_pose_aligned, compute_arm_root_pose,
    verify_alignment, print_verification, GRIPPER_OPEN_SCALE,
)


# ─── TrajectorySmoother (from 02, for offline smoothing) ─────────────────────

class TrajectorySmoother:
    """离线后处理轨迹平滑器: 无效帧填充 + 双向低通滤波 + 速度/加速度/jerk 限幅"""

    SMOOTHNESS_THRESHOLDS = {
        "max_velocity": 3.0,
        "max_acceleration": 8.0,
        "max_jerk": 80.0,
        "si_improvement_min": 0.5,
    }

    def __init__(self, fps=30, max_velocity=1.5, max_acceleration=4.0,
                 max_jerk=20.0, lp_alpha=0.25, max_iterations=10, convergence_eps=1e-5):
        self.dt = 1.0 / fps
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.lp_alpha = lp_alpha
        self.max_iterations = max_iterations
        self.convergence_eps = convergence_eps

    def smooth_trajectory(self, qpos_sequence, smooth_indices):
        n_frames = len(qpos_sequence)
        n_joints = len(smooth_indices)
        trajectory = np.zeros((n_frames, n_joints))
        valid_mask = np.zeros(n_frames, dtype=bool)
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                trajectory[i] = qpos[smooth_indices]
                valid_mask[i] = True
        self._fill_invalid_frames(trajectory, valid_mask)
        trajectory_raw = trajectory.copy()
        trajectory = self._bidirectional_lowpass(trajectory)
        trajectory = self._iterative_clamp(trajectory)
        smoothed_sequence = []
        for i, qpos in enumerate(qpos_sequence):
            if qpos is not None:
                qpos_new = qpos.copy()
            else:
                for j in range(i, -1, -1):
                    if qpos_sequence[j] is not None:
                        qpos_new = qpos_sequence[j].copy()
                        break
                else:
                    continue
            qpos_new[smooth_indices] = trajectory[i]
            smoothed_sequence.append(qpos_new)
        metrics = self._compute_metrics(trajectory, trajectory_raw)
        return smoothed_sequence, metrics

    def _fill_invalid_frames(self, trajectory, valid_mask):
        n_frames = len(trajectory)
        last_valid = 0
        for i in range(n_frames):
            if valid_mask[i]:
                last_valid = i
            else:
                trajectory[i] = trajectory[last_valid]
        first_valid = np.argmax(valid_mask)
        for i in range(first_valid):
            trajectory[i] = trajectory[first_valid]

    def _bidirectional_lowpass(self, trajectory):
        alpha = self.lp_alpha
        n_frames = len(trajectory)
        s1_fwd = np.zeros_like(trajectory)
        s2_fwd = np.zeros_like(trajectory)
        s1_fwd[0] = trajectory[0]
        s2_fwd[0] = trajectory[0]
        for i in range(1, n_frames):
            s1_fwd[i] = s1_fwd[i - 1] + alpha * (trajectory[i] - s1_fwd[i - 1])
            s2_fwd[i] = s2_fwd[i - 1] + alpha * (s1_fwd[i] - s2_fwd[i - 1])
        s1_bwd = np.zeros_like(trajectory)
        s2_bwd = np.zeros_like(trajectory)
        s1_bwd[-1] = s2_fwd[-1]
        s2_bwd[-1] = s2_fwd[-1]
        for i in range(n_frames - 2, -1, -1):
            s1_bwd[i] = s1_bwd[i + 1] + alpha * (s2_fwd[i] - s1_bwd[i + 1])
            s2_bwd[i] = s2_bwd[i + 1] + alpha * (s1_bwd[i] - s2_bwd[i + 1])
        return s2_bwd

    def _clamp_velocity(self, trajectory):
        max_delta = self.max_velocity * self.dt
        for i in range(1, len(trajectory)):
            delta = trajectory[i] - trajectory[i - 1]
            clamped = np.clip(delta, -max_delta, max_delta)
            trajectory[i] = trajectory[i - 1] + clamped
        return trajectory

    def _clamp_acceleration(self, trajectory):
        max_delta_v = self.max_acceleration * self.dt
        for i in range(2, len(trajectory)):
            v_prev = trajectory[i - 1] - trajectory[i - 2]
            v_curr = trajectory[i] - trajectory[i - 1]
            delta_v = v_curr - v_prev
            clamped_dv = np.clip(delta_v, -max_delta_v, max_delta_v)
            v_curr_clamped = v_prev + clamped_dv
            trajectory[i] = trajectory[i - 1] + v_curr_clamped
        return trajectory

    def _clamp_jerk(self, trajectory):
        max_delta_a = self.max_jerk * self.dt
        for i in range(3, len(trajectory)):
            v_im2 = trajectory[i - 2] - trajectory[i - 3]
            v_im1 = trajectory[i - 1] - trajectory[i - 2]
            v_i = trajectory[i] - trajectory[i - 1]
            a_prev = v_im1 - v_im2
            a_curr = v_i - v_im1
            delta_a = a_curr - a_prev
            clamped_da = np.clip(delta_a, -max_delta_a, max_delta_a)
            a_curr_clamped = a_prev + clamped_da
            v_i_clamped = v_im1 + a_curr_clamped
            trajectory[i] = trajectory[i - 1] + v_i_clamped
        return trajectory

    def _iterative_clamp(self, trajectory):
        for _ in range(self.max_iterations):
            traj_before = trajectory.copy()
            trajectory = self._clamp_velocity(trajectory)
            trajectory = self._clamp_acceleration(trajectory)
            trajectory = self._clamp_jerk(trajectory)
            max_change = np.max(np.abs(trajectory - traj_before))
            if max_change < self.convergence_eps:
                break
        return trajectory

    def _compute_metrics(self, trajectory_smooth, trajectory_raw):
        dt = self.dt
        vel_raw = np.diff(trajectory_raw, axis=0) / dt
        acc_raw = np.diff(vel_raw, axis=0) / dt
        jerk_raw = np.diff(acc_raw, axis=0) / dt
        vel_smooth = np.diff(trajectory_smooth, axis=0) / dt
        acc_smooth = np.diff(vel_smooth, axis=0) / dt
        jerk_smooth = np.diff(acc_smooth, axis=0) / dt
        raw_max_vel = float(np.max(np.abs(vel_raw)))
        raw_max_acc = float(np.max(np.abs(acc_raw))) if len(acc_raw) > 0 else 0.0
        raw_max_jerk = float(np.max(np.abs(jerk_raw))) if len(jerk_raw) > 0 else 0.0
        raw_si = float(np.sum(jerk_raw ** 2) * dt) if len(jerk_raw) > 0 else 0.0
        smooth_max_vel = float(np.max(np.abs(vel_smooth)))
        smooth_max_acc = float(np.max(np.abs(acc_smooth))) if len(acc_smooth) > 0 else 0.0
        smooth_max_jerk = float(np.max(np.abs(jerk_smooth))) if len(jerk_smooth) > 0 else 0.0
        smooth_si = float(np.sum(jerk_smooth ** 2) * dt) if len(jerk_smooth) > 0 else 0.0
        thresholds = self.SMOOTHNESS_THRESHOLDS
        si_improvement = 1.0 - smooth_si / max(raw_si, 1e-12)
        all_pass = (smooth_max_vel <= thresholds["max_velocity"] and
                    smooth_max_acc <= thresholds["max_acceleration"] and
                    smooth_max_jerk <= thresholds["max_jerk"] and
                    si_improvement >= thresholds["si_improvement_min"])
        return {
            "raw_max_velocity": raw_max_vel,
            "raw_max_acceleration": raw_max_acc,
            "raw_max_jerk": raw_max_jerk,
            "raw_smoothness_index": raw_si,
            "smooth_max_velocity": smooth_max_vel,
            "smooth_max_acceleration": smooth_max_acc,
            "smooth_max_jerk": smooth_max_jerk,
            "smooth_smoothness_index": smooth_si,
            "velocity_reduction": 1.0 - smooth_max_vel / max(raw_max_vel, 1e-6),
            "acceleration_reduction": 1.0 - smooth_max_acc / max(raw_max_acc, 1e-6),
            "jerk_reduction": 1.0 - smooth_si / max(raw_si, 1e-12),
            "pass_velocity": smooth_max_vel <= thresholds["max_velocity"],
            "pass_acceleration": smooth_max_acc <= thresholds["max_acceleration"],
            "pass_jerk": smooth_max_jerk <= thresholds["max_jerk"],
            "pass_si_improvement": si_improvement >= thresholds["si_improvement_min"],
            "all_pass": all_pass,
            "si_improvement": si_improvement,
        }


# ─── 辅助函数 ────────────────────────────────────────────────────────────────

def _reencode_with_ffmpeg(input_path, crf, fps, logger):
    """ffmpeg H.264 重编码"""
    output = str(input_path)
    tmp_path = output.replace(".mp4", "_tmp.mp4")
    if not os.path.exists(output):
        return
    os.rename(output, tmp_path)
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264", "-crf", str(crf),
               "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(fps),
               "-movflags", "+faststart", output]
        result = sp.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0 and os.path.exists(output) and os.path.getsize(output) > 0:
            os.remove(tmp_path)
        else:
            if os.path.exists(tmp_path):
                os.rename(tmp_path, output)
    except Exception:
        if os.path.exists(tmp_path):
            os.rename(tmp_path, output)


def _compute_wrist_positions_sapien(hawor_data, mano_layer, start_frame, num_frames):
    """计算所有有效手腕的 SAPIEN 坐标系位置 (与 02 一致)"""
    positions = []
    for i in range(num_frames):
        gi = start_frame + i
        if not hawor_data["pred_valid"][gi]:
            continue
        trans = hawor_data["pred_trans"][gi]
        if np.isnan(trans).any():
            continue
        _, j = compute_mano_joints(mano_layer, hawor_data["pred_rot"][gi],
                                   hawor_data["pred_hand_pose"][gi], trans)
        joints_sapien = _render_to_sapien(j)
        positions.append(joints_sapien[0, :3].copy())
    return positions


def _render_single_sphere(pos, color, radius, context, internal_scene, old_node):
    """渲染/更新单个球体标记。复用旧节点避免每帧新建销毁。"""
    # 更新已有节点位置（不需要销毁重建）
    if old_node is not None:
        old_node.set_position(pos.tolist())
        return old_node

    mat = context.create_material(np.zeros(4), np.array(color), 0.0, 0.5, 0)
    sphere = context.create_uvsphere_mesh(12, 6)
    model = context.create_model([sphere], [mat])
    node = internal_scene.add_node()
    node.set_position(pos.tolist())
    node.set_scale([radius, radius, radius])
    obj = internal_scene.add_object(model, node)
    obj.shading_mode = 0
    obj.cast_shadow = False
    obj.transparency = 0
    return node


def _draw_camera_trajectory(positions, color, context, internal_scene,
                            sphere_radius=0.008, line_radius=0.002):
    """在 SAPIEN 场景中绘制相机轨迹 (球 + 连线)

    Args:
        positions: (N, 3) SAPIEN 坐标系下的相机位置序列
        color: [r, g, b] 颜色, 0~1
        context: SAPIEN 渲染上下文
        internal_scene: SAPIEN 内部场景
        sphere_radius: 球体半径
        line_radius: 连线半径
    """
    if len(positions) < 2:
        return
    color4 = np.array([color[0], color[1], color[2], 1.0])
    mat = context.create_material(np.zeros(4), color4, 0.0, 0.5, 0)

    # 采样: 轨迹点太多时每隔几帧取一个, 控制渲染对象数量
    max_points = 300
    step = max(1, len(positions) // max_points)
    sampled = positions[::step]

    # 画球 (轨迹点)
    sphere_mesh = context.create_uvsphere_mesh(8, 4)
    sphere_model = context.create_model([sphere_mesh], [mat])
    for pos in sampled:
        node = internal_scene.add_node()
        node.set_position(pos.tolist())
        node.set_scale([sphere_radius, sphere_radius, sphere_radius])
        obj = internal_scene.add_object(sphere_model, node)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 0

    # 画线 (胶囊体连接相邻点)
    for i in range(len(sampled) - 1):
        p1, p2 = sampled[i], sampled[i + 1]
        mid = (p1 + p2) / 2.0
        length = np.linalg.norm(p2 - p1)
        if length < 1e-6:
            continue
        cylinder = context.create_capsule_mesh(line_radius, length / 2, 6, 3)
        model = context.create_model([cylinder], [mat])
        node = internal_scene.add_node()
        node.set_position(mid.tolist())
        direction = (p2 - p1) / length
        z_axis = np.array([0, 0, 1.0])
        rot_axis = np.cross(z_axis, direction)
        rot_axis_len = np.linalg.norm(rot_axis)
        if rot_axis_len > 1e-6:
            rot_axis_n = rot_axis / rot_axis_len
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rot_quat_wxyz = pr.quaternion_from_axis_angle(
                np.array([rot_axis_n[0], rot_axis_n[1], rot_axis_n[2], angle])
            )
            rot_quat_xyzw = pr.quaternion_xyzw_from_wxyz(rot_quat_wxyz)
            node.set_rotation(rot_quat_xyzw.tolist())
        elif np.dot(z_axis, direction) < 0:
            node.set_rotation([0, 1, 0, 0])
        obj = internal_scene.add_object(model, node)
        obj.shading_mode = 0
        obj.cast_shadow = False
        obj.transparency = 0


# ─── 核心渲染函数 ────────────────────────────────────────────────────────────

def render_robot_video(
    hawor_dir, ras_dir, glb_path, transform_params,
    hand_idx=0, output="output.mp4",
    fps=30, cam_width=1920, cam_height=1080,
    view="fpv", crf=18, start_frame=0, num_frames=-1,
    fixed_base=False, viewer=False, logger=None,
):
    """渲染 R1 单臂机器人 + GLB 场景视频 (新坐标系)

    Args:
        hawor_dir: HaWoR 数据目录
        ras_dir: RAS 重建结果目录
        glb_path: GLB 文件路径
        transform_params: transform_params.npz 路径 (001_align_scene.py 输出)
        hand_idx: 手部索引 0(左) 或 1(右)
        output: 输出 mp4 路径
        fps: 帧率
        cam_width/height: 分辨率
        view: 相机视角
        crf: 编码质量
        start_frame: 起始帧
        num_frames: 帧数 (-1=全部)
        fixed_base: 固定基座模式
        logger: Logger
    """
    if logger is None:
        logger = logging.getLogger("002_render")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

    prefix = "left" if hand_idx == 0 else "right"
    mode_label = "左手" if hand_idx == 0 else "右手"

    # 加载变换参数
    params = np.load(str(transform_params))
    s = float(params["scale_ratio"])
    R_h2g = params["R_hand_to_glb"]
    t_h2g = params["t_hand_to_glb"]
    Rx_hand = params.get("Rx_hand", np.diag([1.0, -1.0, -1.0]))
    if Rx_hand is None:
        Rx_hand = np.diag([1.0, -1.0, -1.0])
    set_render_transform_params(R_h2g, t_h2g, Rx_hand, s)

    logger.info(f"渲染模式: R1 {mode_label}臂 + GLB 物体 (新坐标系)")
    logger.info(f"  scale_ratio={s:.6f}")

    # ── [1/7] 加载数据 ──
    logger.info("\n[1/7] 加载数据 ...")
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
    logger.info(f"  焦距: {focal:.1f}px, FOV={np.degrees(cam_fov):.1f}°")

    # ── [2/7] 创建场景 + 加载 GLB ──
    logger.info("\n[2/7] 创建场景 + 加载 GLB ...")
    scene = setup_scene()

    if glb_path:
        glb_path = Path(glb_path)
        if glb_path.exists():
            obj_actors = load_glb_to_sapien(glb_path, s, R_h2g, t_h2g, scene, logger)
    else:
        logger.warning(f"  GLB 文件不存在: {glb_path}")

    # ── [3/7] 加载机器人 + 初始化 retargeting/IK ──
    logger.info(f"\n[3/7] 初始化 R1 {prefix} 臂 ...")
    robot_dir = PROJECT_ROOT / "dex-retargeting" / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    urdf_path = FLOATING_LEFT_URDF if hand_idx == 0 else FLOATING_RIGHT_URDF
    arm_starting = LEFT_ARM_STARTING if hand_idx == 0 else RIGHT_ARM_STARTING

    arm_urdf_path = prepare_arm_urdf(urdf_path, arm_prefix=prefix)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = True
    robot = loader.load(arm_urdf_path)

    joint_names = [j.name for j in robot.get_active_joints()]
    arm_joint_indices = [i for i, n in enumerate(joint_names) if f"{prefix}_arm_joint" in n]
    gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
    gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

    for joint in robot.get_active_joints():
        joint.set_drive_property(stiffness=100000.0, damping=10000.0)

    init_qpos = robot.get_qpos().copy()
    for j, idx in enumerate(arm_joint_indices):
        if j < len(arm_starting):
            init_qpos[idx] = arm_starting[j]
    init_qpos[gripper_idx1] = 0.04
    init_qpos[gripper_idx2] = -0.04
    robot.set_qpos(init_qpos)
    scene.step()
    scene.update_render()

    # Retargeting
    hand_type = HandType.left if hand_idx == 0 else HandType.right
    config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
    override = dict(
        add_dummy_free_joint=True,
        normal_delta=1e-5,
        huber_delta=0.01,
        target_link_names=[
            f"{prefix}_gripper_finger_link1",
            f"{prefix}_gripper_finger_link2",
            f"{prefix}_gripper_link",
        ],
        target_link_human_indices=np.array([4, 8, 0]),
    )
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    ref_indices = retargeting.optimizer.target_link_human_indices
    fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed

    retarget2sapien = np.array(
        [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
    ).astype(int)
    sapien2retarget = {}
    for sapien_i, retarget_i in enumerate(retarget2sapien):
        sapien2retarget[retarget_i] = sapien_i
    fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
    for i, retarget_idx in enumerate(fixed_retarget_indices):
        if retarget_idx in sapien2retarget:
            fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

    # IK solver
    ik_solver = RelaxedIKSolver(
        left_setting_file_path=str(R1_LEFT_SETTINGS),
        right_setting_file_path=str(R1_RIGHT_SETTINGS),
        tolerances=[0.1] * 6,
    )
    if hand_idx == 0:
        ik_solver.relaxed_ik_left.reset(LEFT_ARM_STARTING)
    else:
        ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

    # MANO layer
    betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
    mano_layer = MANOLayer(prefix, betas_mean)

    logger.info(f"  ✓ {prefix} 臂已加载: {len(arm_joint_indices)}个臂关节 + 2个夹爪关节")

    # ── [4/7] Warm start ──
    logger.info("\n[4/7] Warm start ...")
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
        wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
        wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
        wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
        retargeting.warm_start(
            joints_sapien[0, :3], wrist_quat,
            hand_type=hand_type, is_mano_convention=True,
        )
        logger.info(f"  ✓ Warm start 完成 (帧 {g_idx})")
        break

    # ── [5/7] 放置机器人 + Warmup ──
    logger.info("\n[5/7] 放置机器人 ...")
    wrist_positions = _compute_wrist_positions_sapien(
        hawor_data, mano_layer, start_frame, num_frames)
    if not wrist_positions:
        logger.error("无法提取有效手腕位置")
        return None

    arm_base_pos, arm_base_q = _compute_optimal_fixed_base(wrist_positions)
    robot.set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
    scene.step()
    scene.update_render()
    logger.info(f"  初始基座: {arm_base_pos}")

    mapping_offset = np.zeros(3)
    safety_offset = np.zeros(3)

    init_qpos_arm = robot.get_qpos().copy()
    current_joints = np.array([init_qpos_arm[i] for i in arm_joint_indices])
    joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
    joint_filter.next(current_joints)

    # Warmup
    first_valid_qpos = None
    for fi in range(start_frame, start_frame + num_frames):
        if not hawor_data["pred_valid"][fi]:
            continue
        rot = hawor_data["pred_rot"][fi]
        trans = hawor_data["pred_trans"][fi]
        hand_pose = hawor_data["pred_hand_pose"][fi]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        ref_value = joints_sapien[ref_indices, :].astype(np.float32)
        retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
        sapien_qpos = retarget_qpos[retarget2sapien]
        gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)
        tracked_base = arm_base_pos if fixed_base else _compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
        robot.set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
        scene.step()
        for link in robot.get_links():
            if f"{prefix}_arm_base_link" == link.get_name():
                pose = link.get_entity_pose()
                base_link_p_w = np.array(pose.p)
                base_link_q_w = np.array(pose.q)
                break
        base_link_R_w = pr.matrix_from_quaternion(base_link_q_w)
        base_link_R_inv_w = base_link_R_w.T
        ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
        ik_target_b = base_link_R_inv_w @ (ik_target_raw - base_link_p_w)
        ee_R_base = base_link_R_inv_w @ R_ee_world_fk
        ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
        solve_fn = ik_solver.solve_position_left if prefix == "left" else ik_solver.solve_position_right
        ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        for _ in range(IK_SOLVE_PER_FRAME * 5 - 1):
            ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        first_valid_qpos = np.array(ik_joints)
        break

    if first_valid_qpos is not None:
        init_qpos_arm_arr = np.array(arm_starting)
        for wi in range(WARMUP_FRAMES):
            t = (wi + 1) / WARMUP_FRAMES
            t = t * t * (3 - 2 * t)
            interp = init_qpos_arm_arr * (1 - t) + first_valid_qpos * t
            qpos = robot.get_qpos().copy()
            for j_idx, arm_idx in enumerate(arm_joint_indices):
                qpos[arm_idx] = interp[j_idx]
            robot.set_qpos(qpos)
            scene.step()
        logger.info(f"  Warmup 完成 ({WARMUP_FRAMES} 帧)")

    # ── [6/7] 设置相机 ──
    logger.info("\n[6/7] 设置相机 ...")
    camera = scene.add_camera("main", cam_width, cam_height, cam_fov, 0.01, 100.0)

    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        logger.info(f"  使用 hawor 相机轨迹 (第一视角)")
    else:
        centroid = np.mean(wrist_positions, axis=0)
        robot_root = arm_base_pos.copy()
        if view == "topdown":
            cam_target = centroid
            cam_pos = cam_target + np.array([0.0, 0.0, 1.2])
            cam_quat = make_look_at_camera(cam_pos, cam_target, up=np.array([0, 1, 0]))
        elif view == "behind":
            cam_pos = robot_root + np.array([2.5, 0.0, 1.2])
            cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
        elif view == "front":
            cam_pos = robot_root + np.array([-2.5, 0.0, 1.2])
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        logger.info(f"  视角: {view}, 相机位置: {cam_pos}")

    # ── [7/7] 渲染视频 ──
    logger.info("\n[7/7] 渲染视频 ...")

    qpos_log = []
    solve_fn = ik_solver.solve_position_left if prefix == "left" else ik_solver.solve_position_right
    kp_nodes = []
    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene

    if viewer:
        # 绘制相机轨迹对比 (HaWoR=红, RAS=蓝)
        R_hawor_traj, t_hawor_traj = load_hawor_c2w(hawor_dir)
        if R_hawor_traj is not None and t_hawor_traj is not None:
            hawor_positions = []
            for i in range(len(t_hawor_traj)):
                p, _ = hawor_cam_to_sapien_pose(R_hawor_traj[i], t_hawor_traj[i])
                hawor_positions.append(p)
            hawor_positions = np.array(hawor_positions)
            _draw_camera_trajectory(hawor_positions, [1, 0, 0], context, internal_scene)
            logger.info(f"  HaWoR 相机轨迹已绘制 (红色, {len(hawor_positions)} 点)")
        if ras_dir:
            R_ras_traj, t_ras_traj = load_ras_cameras(ras_dir)
            ras_positions = []
            for i in range(len(t_ras_traj)):
                p, _ = ras_cam_to_sapien_pose(t_ras_traj[i], R_ras_traj[i])
                ras_positions.append(p)
            ras_positions = np.array(ras_positions)
            _draw_camera_trajectory(ras_positions, [0, 0, 1], context, internal_scene)
            logger.info(f"  RAS 相机轨迹已绘制 (蓝色, {len(ras_positions)} 点)")

        viewer_win = scene.create_viewer()
        viewer_win.set_camera_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        local_idx = 0
        logger.info("  按 ESC 或关闭窗口退出 Viewer (红=HaWoR相机, 蓝=RAS相机)")
        while not viewer_win.closed:
            global_idx = start_frame + (local_idx % num_frames)

            if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(
                    R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            if hawor_data["pred_valid"][global_idx]:
                rot = hawor_data["pred_rot"][global_idx]
                trans = hawor_data["pred_trans"][global_idx]
                hand_pose = hawor_data["pred_hand_pose"][global_idx]
                if not (np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose))):
                    _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
                    joints_sapien = _render_to_sapien(j)
                    kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices)
                    ref_value = joints_sapien[ref_indices, :].astype(np.float32)
                    retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
                    sapien_qpos = retarget_qpos[retarget2sapien]
                    gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)
                    tracked_base = arm_base_pos if fixed_base else _compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
                    robot.set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
                    scene.step()
                    for link in robot.get_links():
                        if f"{prefix}_arm_base_link" == link.get_name():
                            pose = link.get_entity_pose()
                            base_link_p = np.array(pose.p)
                            base_link_q = np.array(pose.q)
                            break
                    base_link_R = pr.matrix_from_quaternion(base_link_q)
                    base_link_R_inv = base_link_R.T
                    ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
                    ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
                    ee_R_base = base_link_R_inv @ R_ee_world_fk
                    ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
                    ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
                    for _ in range(IK_SOLVE_PER_FRAME - 1):
                        ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
                    filtered_joints = joint_filter.next(np.array(ik_joints))
                    qpos = robot.get_qpos().copy()
                    for j_idx, arm_idx in enumerate(arm_joint_indices):
                        qpos[arm_idx] = filtered_joints[j_idx]
                    if gripper_idx1 < len(sapien_qpos):
                        qpos[gripper_idx1] = float(sapien_qpos[gripper_idx1])
                    if gripper_idx2 < len(sapien_qpos):
                        qpos[gripper_idx2] = float(sapien_qpos[gripper_idx2])
                    robot.set_qpos(qpos)

            scene.step()
            scene.update_render()
            viewer_win.render()
            local_idx += 1
        logger.info("  Viewer 已关闭")
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    for local_idx in trange(num_frames, desc=f"渲染{prefix}"):
        global_idx = start_frame + local_idx

        # 更新相机
        if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(
                R_c2w_all[global_idx], t_c2w_all[global_idx])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

        if not hawor_data["pred_valid"][global_idx]:
            scene.step()
            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  R1 {prefix} Arm  |  INVALID",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(bgr)
            continue

        rot = hawor_data["pred_rot"][global_idx]
        trans = hawor_data["pred_trans"][global_idx]
        hand_pose = hawor_data["pred_hand_pose"][global_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            scene.step()
            scene.update_render()
            camera.take_picture()
            rgb = camera.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
            h, w = bgr.shape[:2]
            cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  R1 {prefix} Arm  |  NaN",
                        (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            writer.write(bgr)
            continue

        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)

        # 渲染3个关键点
        kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices)

        ref_value = joints_sapien[ref_indices, :].astype(np.float32)
        retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
        sapien_qpos = retarget_qpos[retarget2sapien]

        gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)

        tracked_base = arm_base_pos if fixed_base else _compute_tracking_base_pos(arm_base_pos, gripper_pos_fk, arm_base_q)
        robot.set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
        scene.step()

        for link in robot.get_links():
            if f"{prefix}_arm_base_link" == link.get_name():
                pose = link.get_entity_pose()
                base_link_p = np.array(pose.p)
                base_link_q = np.array(pose.q)
                break
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T

        ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
        ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
        ee_R_base = base_link_R_inv @ R_ee_world_fk
        ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

        ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
        for _ in range(IK_SOLVE_PER_FRAME - 1):
            ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())

        filtered_joints = joint_filter.next(np.array(ik_joints))

        qpos = robot.get_qpos().copy()
        for j_idx, arm_idx in enumerate(arm_joint_indices):
            qpos[arm_idx] = filtered_joints[j_idx]
        if gripper_idx1 < len(sapien_qpos):
            qpos[gripper_idx1] = float(sapien_qpos[gripper_idx1])
        if gripper_idx2 < len(sapien_qpos):
            qpos[gripper_idx2] = float(sapien_qpos[gripper_idx2])
        robot.set_qpos(qpos)
        qpos_log.append(qpos.copy())

        scene.step()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  R1 {prefix} Arm + Objects",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(bgr)

    writer.release()

    # ffmpeg 重编码
    _reencode_with_ffmpeg(output, crf, fps, logger)

    # qpos 保存
    qpos_path = str(Path(output).with_suffix(".npy")).replace("videos", "tracking")
    qpos_path = qpos_path.replace(".npy", f"_{prefix}.npy")
    os.makedirs(os.path.dirname(qpos_path), exist_ok=True)
    if qpos_log:
        np.save(qpos_path, np.array(qpos_log))
        logger.info(f"  ✓ {prefix} qpos 已保存: {qpos_path} ({len(qpos_log)} 帧)")

    logger.info(f"\n✓ 视频已保存: {output}")
    return output


def render_gripper_video(
    hawor_dir, glb_path, transform_params,
    hand_idx=0, output="gripper.mp4",
    fps=30, cam_width=1920, cam_height=1080,
    view="fpv", crf=18, start_frame=0, num_frames=-1,
    viewer=False, logger=None,
):
    """渲染夹爪末端跟踪视频 (3个关键点球体, 新坐标系)

    Args:
        hawor_dir: HaWoR 数据目录
        glb_path: GLB 文件路径
        transform_params: transform_params.npz 路径
        hand_idx: 手部索引
        output: 输出路径
        fps: 帧率
        cam_width/height: 分辨率
        view: 相机视角
        crf: 编码质量
        start_frame: 起始帧
        num_frames: 帧数
        logger: Logger
    """
    if logger is None:
        logger = logging.getLogger("002_gripper")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    prefix = "left" if hand_idx == 0 else "right"
    logger.info(f"夹爪渲染 (新坐标): {prefix}")

    params = np.load(str(transform_params))
    s = float(params["scale_ratio"])
    R_h2g = params["R_hand_to_glb"]
    t_h2g = params["t_hand_to_glb"]
    Rx_hand = params.get("Rx_hand", np.diag([1.0, -1.0, -1.0]))
    set_render_transform_params(R_h2g, t_h2g, Rx_hand, s)

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

    # retargeting ref_indices
    robot_dir = PROJECT_ROOT / "dex-retargeting" / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    hand_type = HandType.left if hand_idx == 0 else HandType.right
    config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
    override = dict(
        add_dummy_free_joint=True, normal_delta=1e-5, huber_delta=0.01,
        target_link_names=[f"{prefix}_gripper_finger_link1",
                           f"{prefix}_gripper_finger_link2",
                           f"{prefix}_gripper_link"],
        target_link_human_indices=np.array([4, 8, 0]),
    )
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    ref_indices = retargeting.optimizer.target_link_human_indices

    # ── 创建场景 ──
    scene = setup_scene()

    glb_path = Path(glb_path)
    if glb_path.exists():
        obj_actors = load_glb_to_sapien(glb_path, s, R_h2g, t_h2g, scene, logger)

    # ── 设置相机 ──
    camera = scene.add_camera("gripper", cam_width, cam_height, cam_fov, 0.01, 100.0)

    _cam_to_sapien = lambda R, t: hawor_cam_to_sapien_pose(R, t)
    wrist_positions = _compute_wrist_positions_sapien(
        hawor_data, mano_layer, start_frame, num_frames)
    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[0], t_c2w_all[0])
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
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
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

    # ── 渲染 ──
    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    if viewer:
        viewer_win = scene.create_viewer()
        viewer_win.set_camera_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        local_idx = 0
        logger.info("  按 ESC 或关闭窗口退出 Viewer")
        while not viewer_win.closed:
            # 安全计算 global_idx，防止越界
            if num_frames <= 0:
                effective_num = len(R_c2w_all) if R_c2w_all is not None else 1
            else:
                effective_num = num_frames
            raw_global_idx = start_frame + (local_idx % effective_num)
            # 夹爪在有效范围内
            if R_c2w_all is not None:
                global_idx = max(0, min(raw_global_idx, len(R_c2w_all) - 1))
            else:
                global_idx = raw_global_idx

            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            if hawor_data["pred_valid"][global_idx]:
                rot = hawor_data["pred_rot"][global_idx]
                trans = hawor_data["pred_trans"][global_idx]
                hand_pose = hawor_data["pred_hand_pose"][global_idx]
                if not (np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose))):
                    _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
                    joints_sapien = _render_to_sapien(j)
                    kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012)

            scene.step()
            scene.update_render()
            viewer_win.render()
            local_idx += 1
        logger.info("  Viewer 已关闭")
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    for local_idx in trange(num_frames, desc=f"夹爪{prefix}"):
        global_idx = start_frame + local_idx

        if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

        if not hawor_data["pred_valid"][global_idx]:
            for node in kp_nodes:
                internal_scene.remove_node(node)
            kp_nodes.clear()
            scene.step()
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

        kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012)

        scene.step()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper {prefix} Only",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(bgr)

    writer.release()
    _reencode_with_ffmpeg(output, crf, fps, logger)
    logger.info(f"\n✓ 夹爪视频已保存: {output}")
    return output


def _compute_gripper_pose_by_strategy(strategy, mano_wrist, mano_finger1, mano_finger2,
                                       prefix, finger_origin_x, finger1_origin_x, finger2_origin_x,
                                       open_scale=GRIPPER_OPEN_SCALE, mano_joints=None, scale=1.0):
    """根据对齐策略选择计算函数 (新坐标系, 输入已是 GLB 坐标)

    strategy:
      "aligned"     → compute_gripper_pose_aligned (3 点 Gram-Schmidt, 中点对齐)
      "analytical"  → compute_analytical_gripper_pose (3 点 Gram-Schmidt, 指尖中点)
      "svd_palm"    → compute_mano_based_gripper_pose (5 点 SVD + Gram-Schmidt, 需要 mano_joints)
    """
    if strategy == "svd_palm":
        if mano_joints is None:
            raise ValueError("strategy='svd_palm' 需要传入 mano_joints (joints_sapien 21-joint 数组)")
        return compute_mano_based_gripper_pose(mano_joints, prefix=prefix, scale=scale)
    elif strategy == "aligned":
        return compute_gripper_pose_aligned(
            mano_wrist, mano_finger1, mano_finger2, prefix, open_scale=open_scale)
    else:
        return compute_analytical_gripper_pose(
            mano_wrist, mano_finger1, mano_finger2, prefix, finger_origin_x,
            finger1_origin_x=finger1_origin_x, finger2_origin_x=finger2_origin_x)


def render_gripper_only_video(
    hawor_dir, glb_path, transform_params,
    hand_idx=1, output="gripper_urdf.mp4",
    fps=30, cam_width=1920, cam_height=1080,
    view="fpv", crf=18, start_frame=0, num_frames=-1,
    with_arm=False, smooth=1, verify=False,
    analytical=True, arm_mode="half", viewer=False, logger=None,
    strategy="aligned", open_scale=GRIPPER_OPEN_SCALE,
    ras_dir=None, use_ras_cam=False,
):
    """渲染只有夹爪 URDF 的视频 (新坐标系, 不加载手臂)

    从 render_gripper_only.py 照搬并改新坐标:
      _render_to_sapien(j) → _render_to_sapien(j)
      hawor_cam_to_sapien_pose(R, t) → hawor_cam_to_sapien_pose(R, t)
      load_glb_to_sapien(...) → load_glb_to_sapien(...)
      _compute_wrist_positions_sapien(...) → _compute_wrist_positions_sapien(...)

    Args:
        hawor_dir: HaWoR 数据目录
        glb_path: GLB 文件路径
        transform_params: transform_params.npz 路径
        hand_idx: 手部索引 0(左) 1(右)
        output: 输出 mp4 路径
        fps: 帧率
        cam_width/height: 分辨率
        view: 相机视角
        crf: 编码质量
        start_frame: 起始帧
        num_frames: 帧数 (-1=全部)
        with_arm: 是否附带手臂 URDF
        smooth: 0=不平滑, 1=EMA
        verify: 是否计算验证误差
        analytical: True=解析/对齐策略, False=优化器模式
        arm_mode: "half"=link4-6, "full"=link1-6
        logger: Logger
        strategy: "aligned"=新策略(先对齐夹爪两点), "analytical"=旧策略(Gram-Schmidt)
        open_scale: 夹爪开合缩放因子
    """
    if logger is None:
        logger = logging.getLogger("002_gripper_urdf")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    prefix = "left" if hand_idx == 0 else "right"
    mode_label = "左手" if hand_idx == 0 else "右手"
    logger.info(f"夹爪URDF渲染 (新坐标): {mode_label}")

    # 加载变换参数
    params = np.load(str(transform_params))
    s = float(params["scale_ratio"])
    R_h2g = params["R_hand_to_glb"]
    t_h2g = params["t_hand_to_glb"]
    Rx_hand = params.get("Rx_hand", np.diag([1.0, -1.0, -1.0]))
    set_render_transform_params(R_h2g, t_h2g, Rx_hand, s)

    # ── 加载数据 ──
    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    total_frames = hawor_data["pred_trans"].shape[0]
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame
    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)
    # ── 加载相机数据 (HaWoR 或 RAS) ──
    if use_ras_cam and ras_dir:
        R_c2w_all, t_c2w_all = load_ras_cameras(ras_dir)
        _cam_to_sapien = lambda R, t: ras_cam_to_sapien_pose(t, R)
        logger.info(f"  使用 RAS 相机轨迹 ({len(R_c2w_all)} 帧, 已在 GLB 空间)")
    else:
        R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)
        _cam_to_sapien = lambda R, t: hawor_cam_to_sapien_pose(R, t)
        if R_c2w_all is not None:
            logger.info(f"  使用 HaWoR 相机轨迹 ({len(R_c2w_all)} 帧)")

    focal = hawor_data.get("img_focal")
    if focal is None or focal <= 0:
        focal = HAWOR_FOCAL_DEFAULT
    focal_render = focal * cam_width / 1280.0
    cam_fov = 2 * np.arctan(cam_height / 2.0 / focal_render)

    betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
    mano_layer = MANOLayer(prefix, betas_mean)

    finger_origin_x = 0.03689
    finger1_origin_x = 0.03689
    finger2_origin_x = 0.03689
    logger.info(f"  finger_origin_x = {finger_origin_x*1000:.1f}mm (URDF 原始值, 不缩放)")

    # ── 初始化 Retargeting ──
    robot_dir = PROJECT_ROOT / "dex-retargeting" / "assets" / "robots" / "hands"
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    if analytical:
        retargeting, ref_indices, _ = init_gripper_retargeting(prefix, finger_origin_x, PROJECT_ROOT)
        fixed_qpos = np.zeros(0, dtype=np.float32)
        strat_map = {"aligned": "3点Gram-Schmidt(中点对齐)", "analytical": "3点Gram-Schmidt(指尖中点)", "svd_palm": "5点SVD+Gram-Schmidt"}
        strat_label = strat_map.get(strategy, "未知")
        logger.info(f"  模式: 解析 ({strat_label}, open_scale={open_scale})")
    else:
        retargeting, ref_indices, _ = init_gripper_retargeting(
            prefix, finger_origin_x, PROJECT_ROOT,
            finger1_origin_x=finger1_origin_x,
            finger2_origin_x=finger2_origin_x,
        )
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        logger.info(f"  模式: 优化器 (gripper-only URDF, 8 DOF, 3 target points, 独立缩放)")

    def _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2, mano_joints=None):
        return _compute_gripper_pose_by_strategy(
            strategy, mano_wrist, mano_finger1, mano_finger2,
            prefix, finger_origin_x, finger1_origin_x, finger2_origin_x,
            open_scale, mano_joints=mano_joints, scale=s)

    # ── 创建场景 ──
    scene = setup_scene()

    if glb_path:
        glb_path = Path(glb_path)
        if glb_path.exists():
            obj_actors = load_glb_to_sapien(glb_path, s, R_h2g, t_h2g, scene, logger)
        else:
            logger.warning(f"  GLB 文件不存在: {glb_path}")

    # ── 加载夹爪 URDF ──
    if with_arm:
        if arm_mode == "full":
            gripper_urdf_path = prepare_full_arm_urdf(prefix)
            logger.info(f"  渲染URDF: 完整手臂+夹爪 (arm_link1-6 + gripper)")
        else:
            gripper_urdf_path = prepare_half_arm_urdf(prefix)
            logger.info(f"  渲染URDF: 半个手臂+夹爪 (arm_link4-6 + gripper)")
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
    # 对 URDF 所有 link 应用 scale (匹配 MANO 的 s 缩放)
    robot_root_pose = robot.get_root_pose()
    root_pos = np.array(robot_root_pose.p)
    root_scale_pos = root_pos * s
    for link in robot.get_links():
        link.set_pose(sapien.Pose((np.array(link.get_entity_pose().p) * s).tolist(), link.get_entity_pose().q))
    logger.info(f"  URDF scale 应用: {s:.2f}")

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

    sapien_name_to_retarget_idx = {
        n: retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names
    }

    scene.step()
    scene.update_render()

    # 计算 gripper_link 相对于 root 的 offset
    arm_starting_values = []  # with_arm 时, arm 关节的起始值 (非 0, 让 arm 自然伸展)
    if with_arm and arm_joint_indices:
        qpos_now = robot.get_qpos().copy()
        arm_starting_full = RIGHT_ARM_STARTING if prefix == "right" else LEFT_ARM_STARTING
        if arm_mode == "half":
            arm_starting = arm_starting_full[-3:]  # 半臂: arm_link4-6
        else:
            arm_starting = arm_starting_full       # 全臂: arm_link1-6
        for i, ai in enumerate(arm_joint_indices):
            val = arm_starting[i] if i < len(arm_starting) else 0.0
            qpos_now[ai] = val
            arm_starting_values.append(val)
        robot.set_qpos(qpos_now)
        scene.update_render()
    if with_arm:
        gripper_offset_pos, gripper_offset_R = compute_gripper_offset_in_root(robot, prefix)
    else:
        gripper_offset_pos = np.zeros(3)
        gripper_offset_R = np.eye(3)

    # ── Warm start retargeting ──
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
        g_pos, g_R, joint1, joint2 = _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2, mano_joints=joints_sapien)
        g_quat = pr.quaternion_from_matrix(g_R)
        retargeting.warm_start(
            g_pos, g_quat,
            hand_type=hand_type, is_mano_convention=False,
        )
        finger_j1_name = f"{prefix}_gripper_finger_joint1"
        finger_j2_name = f"{prefix}_gripper_finger_joint2"
        for num, jname in enumerate(retargeting.optimizer.target_joint_names):
            if jname == finger_j1_name:
                retargeting.last_qpos[num] = joint1
            elif jname == finger_j2_name:
                retargeting.last_qpos[num] = joint2
        logger.info(f"  ✓ Warm start 完成 (帧 {g_idx}), j1={joint1:.4f}, j2={joint2:.4f}")
        break

    # ── 探测首帧有效位姿 + Warmup ──
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
            g_pos, g_R, _, _ = _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2, mano_joints=joints_sapien)
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
        logger.info(f"  Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep)")

    # ── 两阶段渲染: 收集 → 离线平滑 → 输出 ──
    # 平滑器: 离线后处理 (双向低通 + 速度/加速度/jerk 限幅)
    trajectory_smoother = TrajectorySmoother(fps=fps)
    logger.info(f"  平滑模式: 两阶段离线 (TrajectorySmoother, dt={1.0/fps:.4f}s)")

    # ── Phase 1: 收集所有帧的 raw pose + joints (不渲染) ──
    logger.info("\n[Phase 1/3] 收集所有帧的 raw pose + joints ...")
    raw_trajectory = []  # [root_pos(3), root_quat(4), joint1(1), joint2(1)] = 9 DOF
    raw_mano_finger1 = []
    raw_mano_finger2 = []
    for probe_idx in range(num_frames):
        g_idx = start_frame + probe_idx
        if not hawor_data["pred_valid"][g_idx]:
            raw_trajectory.append(None)
            raw_mano_finger1.append(None)
            raw_mano_finger2.append(None)
            continue
        rot = hawor_data["pred_rot"][g_idx]
        trans = hawor_data["pred_trans"][g_idx]
        hand_pose = hawor_data["pred_hand_pose"][g_idx]
        if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
            raw_trajectory.append(None)
            raw_mano_finger1.append(None)
            raw_mano_finger2.append(None)
            continue
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        mano_finger1 = joints_sapien[ref_indices[0], :3]
        mano_finger2 = joints_sapien[ref_indices[1], :3]
        raw_mano_finger1.append(mano_finger1.copy())
        raw_mano_finger2.append(mano_finger2.copy())
        if analytical:
            mano_wrist = joints_sapien[0, :3]
            g_pos, g_R, joint1, joint2 = _compute_gripper_pose(mano_wrist, mano_finger1, mano_finger2, mano_joints=joints_sapien)
            root_R = g_R @ gripper_offset_R.T
            root_pos = g_pos - root_R @ gripper_offset_pos
            root_quat = pr.quaternion_from_matrix(root_R)
            trajectory_frame = np.concatenate([root_pos, root_quat, [joint1, joint2]])  # 9 DOF
            raw_trajectory.append(trajectory_frame)
        else:
            ref_value = joints_sapien[ref_indices, :3]
            retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
            g_pos_fk, g_R_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)
            root_R = g_R_fk @ gripper_offset_R.T
            root_pos = g_pos_fk - root_R @ gripper_offset_pos
            root_quat = pr.quaternion_from_matrix(root_R)
            joint1 = retarget_qpos[sapien_name_to_retarget_idx[f"{prefix}_gripper_finger_joint1"]]
            joint2 = retarget_qpos[sapien_name_to_retarget_idx[f"{prefix}_gripper_finger_joint2"]]
            trajectory_frame = np.concatenate([root_pos, root_quat, [joint1, joint2]])
            raw_trajectory.append(trajectory_frame)
    logger.info(f"  收集完成: {sum(1 for t in raw_trajectory if t is not None)} 有效帧 / {num_frames}")

    # ── Phase 1b: 无效帧填充 (向前填充) ──
    for i in range(1, num_frames):
        if raw_trajectory[i] is None and raw_trajectory[i - 1] is not None:
            raw_trajectory[i] = raw_trajectory[i - 1].copy()
        if raw_mano_finger1[i] is None and raw_mano_finger1[i - 1] is not None:
            raw_mano_finger1[i] = raw_mano_finger1[i - 1].copy()
        if raw_mano_finger2[i] is None and raw_mano_finger2[i - 1] is not None:
            raw_mano_finger2[i] = raw_mano_finger2[i - 1].copy()
    first_valid = next((i for i, t in enumerate(raw_trajectory) if t is not None), 0)
    for i in range(first_valid):
        if raw_trajectory[i] is None:
            raw_trajectory[i] = raw_trajectory[first_valid].copy()
        if raw_mano_finger1[i] is None:
            raw_mano_finger1[i] = raw_mano_finger1[first_valid].copy()
        if raw_mano_finger2[i] is None:
            raw_mano_finger2[i] = raw_mano_finger2[first_valid].copy()

    # ── Phase 2: 离线平滑 (TrajectorySmoother) ──
    logger.info("\n[Phase 2/3] 离线平滑 ...")
    # frame = [root_pos(3), root_quat(4), joint1(1), joint2(1)] = 9 elements (indices 0-8)
    smooth_indices_pos = [0, 1, 2]   # root_pos
    smooth_indices_joint = [7, 8]    # joint1, joint2
    smooth_indices = smooth_indices_pos + smooth_indices_joint

    smoothed_traj, smooth_metrics = trajectory_smoother.smooth_trajectory(
        [t for t in raw_trajectory], smooth_indices)
    if smooth_metrics["all_pass"]:
        logger.info(f"  ✓ 平滑通过: max_vel={smooth_metrics['smooth_max_velocity']:.3f}, "
                    f"max_acc={smooth_metrics['smooth_max_acceleration']:.3f}, "
                    f"max_jerk={smooth_metrics['smooth_max_jerk']:.3f}")
    else:
        logger.warning(f"  ⚠ 平滑未完全通过: {smooth_metrics}")

    # 重建完整平滑轨迹
    smoothed_full = []
    smooth_idx = 0
    for i in range(num_frames):
        if raw_trajectory[i] is None:
            smoothed_full.append(None)
            continue
        if smoothed_traj[smooth_idx] is not None:
            frame = raw_trajectory[i].copy()
            frame[smooth_indices] = smoothed_traj[smooth_idx][smooth_indices]
            smoothed_full.append(frame)
        else:
            smoothed_full.append(raw_trajectory[i].copy())
        smooth_idx += 1

    # 四元数归一化
    for i, frame in enumerate(smoothed_full):
        if frame is None:
            continue
        quat = frame[3:7]
        norm = np.linalg.norm(quat)
        if norm > 1e-8:
            quat /= norm
            frame[3:7] = quat
        # 确保四元数一致性 (w >= 0)
        if frame[3] < 0:
            frame[3:7] = -frame[3:7]

    logger.info(f"  ✓ 平滑完成: {sum(1 for t in smoothed_full if t is not None)} 有效帧")

    # ── 设置相机 ──
    camera = scene.add_camera("gripper_urdf", cam_width, cam_height, cam_fov, 0.01, 100.0)

    wrist_positions = _compute_wrist_positions_sapien(
        hawor_data, mano_layer, start_frame, num_frames)
    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[0], t_c2w_all[0])
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
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
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

    # ── Viewer 交互模式 ──
    if viewer:
        # 绘制相机轨迹对比 (HaWoR=红, RAS=蓝)
        context_traj = sapien.render.SapienRenderer()._internal_context
        internal_scene_traj = scene.render_system._internal_scene
        # HaWoR 轨迹
        R_hawor, t_hawor = load_hawor_c2w(hawor_dir)
        if R_hawor is not None and t_hawor is not None:
            hawor_positions = []
            for i in range(len(t_hawor)):
                p, _ = hawor_cam_to_sapien_pose(R_hawor[i], t_hawor[i])
                hawor_positions.append(p)
            hawor_positions = np.array(hawor_positions)
            _draw_camera_trajectory(hawor_positions, [1, 0, 0], context_traj, internal_scene_traj)
            logger.info(f"  HaWoR 相机轨迹已绘制 (红色, {len(hawor_positions)} 点)")
        # RAS 轨迹
        if ras_dir:
            R_ras, t_ras = load_ras_cameras(ras_dir)
            ras_positions = []
            for i in range(len(t_ras)):
                p, _ = ras_cam_to_sapien_pose(t_ras[i], R_ras[i])
                ras_positions.append(p)
            ras_positions = np.array(ras_positions)
            _draw_camera_trajectory(ras_positions, [0, 0, 1], context_traj, internal_scene_traj)
            logger.info(f"  RAS 相机轨迹已绘制 (蓝色, {len(ras_positions)} 点)")

        viewer_win = scene.create_viewer()
        viewer_win.set_camera_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        local_idx = 0
        viewer_frames = len(R_c2w_all) if R_c2w_all is not None else num_frames

        # ── MANO 21 关节球 + j3/j5 标记球 ──
        viewer_ctx = sapien.render.SapienRenderer()._internal_context
        viewer_internal = scene.render_system._internal_scene
        _mano_nodes = [None] * 21   # 21 个 MANO 关节球 (红色)
        _mano_colors = [[1, 0, 0, 1]] * 21
        _mano_radius = 0.008         # 8mm 球体
        _j3_node = None
        _j5_node = None

        logger.info(f"  Viewer 启动: {len(smoothed_full)} 帧, {fps} fps, 黄=j3拇指PIP, 紫=j5食指MCP")
        frame_delay = 1.0 / fps
        while not viewer_win.closed:
            loop_frame = local_idx % num_frames
            global_idx = start_frame + (local_idx % viewer_frames)

            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            # 更新夹爪 URDF (URDF已放大s倍, joint值需除以s补偿)
            if loop_frame < len(smoothed_full):
                smoothed = smoothed_full[loop_frame]
                robot.set_root_pose(sapien.Pose(smoothed[:3], smoothed[3:7]))
                if len(smoothed) > 7:
                    qpos = robot.get_qpos().copy()
                    qpos[gripper_idx1] = float(smoothed[7]) / s
                    qpos[gripper_idx2] = float(smoothed[8]) / s
                    robot.set_qpos(qpos)
                scene.step()

            # 渲染 MANO 21 关节球 + j3/j5 高亮
            if hawor_data["pred_valid"][global_idx]:
                rot = hawor_data["pred_rot"][global_idx]
                trans = hawor_data["pred_trans"][global_idx]
                hand_pose = hawor_data["pred_hand_pose"][global_idx]
                if not (np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose))):
                    _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
                    joints_sapien = _render_to_sapien(j)
                    # 21 个 MANO 关节 (红色小球)
                    for kpi in range(21):
                        _mano_nodes[kpi] = _render_single_sphere(
                            joints_sapien[kpi, :3], [1, 0, 0, 1], _mano_radius,
                            viewer_ctx, viewer_internal, _mano_nodes[kpi])
                    # j3 拇指PIP (黄色大球) + j5 食指MCP (紫色大球)
                    _j3_node = _render_single_sphere(joints_sapien[3, :3], [1, 1, 0, 1], 0.020,
                                                      viewer_ctx, viewer_internal, _j3_node)
                    _j5_node = _render_single_sphere(joints_sapien[5, :3], [0.5, 0, 1, 1], 0.020,
                                                      viewer_ctx, viewer_internal, _j5_node)
                    # 打印 joint 值用于调试
                    if local_idx % 15 == 0:
                        sc_d = np.linalg.norm(joints_sapien[8] - joints_sapien[4])
                        _, _, jj1, jj2 = compute_mano_based_gripper_pose(joints_sapien, prefix)
                        logger.info(f"    [frame {global_idx}] j4->j8={sc_d*1000:.1f}mm joint={jj1*1000:.1f}mm qpos={jj1/s*1000:.1f}mm")
                else:
                    _mano_nodes = [None] * 21
                    _j3_node = None
                    _j5_node = None
            else:
                _mano_nodes = [None] * 21
                _j3_node = None
                _j5_node = None

            scene.update_render()
            viewer_win.render()
            local_idx += 1
            time.sleep(frame_delay)
        logger.info("  Viewer 已关闭")
        return None

    # ── Phase 3: 渲染视频 ──
    logger.info("\n[Phase 3/3] 渲染视频 ...")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    verify_errors = [] if verify else None
    if verify:
        logger.info("  验证模式: 开启 (计算指尖位置 + 手腕位姿误差)")

    for local_idx in trange(num_frames, desc=f"渲染夹爪URDF-{prefix}"):
        global_idx = start_frame + local_idx

        # 更新相机
        if R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[global_idx], t_c2w_all[global_idx])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

        # 使用平滑后的轨迹
        frame = smoothed_full[local_idx]

        # MANO 参考点 (用于关键点标记和验证)
        mano_finger1_ref = raw_mano_finger1[local_idx]
        mano_finger2_ref = raw_mano_finger2[local_idx]

        # 渲染 MANO 关键点
        if frame is None:
            for node in kp_nodes:
                internal_scene.remove_node(node)
            kp_nodes.clear()
        else:
            # 加载该帧 MANO 关节用于关键点标记
            g_idx = start_frame + local_idx
            if hawor_data["pred_valid"][g_idx]:
                rot = hawor_data["pred_rot"][g_idx]
                trans = hawor_data["pred_trans"][g_idx]
                hand_pose = hawor_data["pred_hand_pose"][g_idx]
                if not (np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose))):
                    _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
                    joints_sapien = _render_to_sapien(j)
                    kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, ref_indices, radius=0.012)
                else:
                    for node in kp_nodes:
                        internal_scene.remove_node(node)
                    kp_nodes.clear()
            else:
                for node in kp_nodes:
                    internal_scene.remove_node(node)
                kp_nodes.clear()

            # 设置机器人位姿 (从平滑轨迹)
            root_pos = frame[0:3]
            root_quat = frame[3:7]
            joint1 = float(frame[7])
            joint2 = float(frame[8])
            robot.set_root_pose(sapien.Pose(root_pos.tolist(), root_quat.tolist()))
            qpos = robot.get_qpos().copy()
            if with_arm and arm_joint_indices:
                for i, ai in enumerate(arm_joint_indices):
                    qpos[ai] = arm_starting_values[i] if i < len(arm_starting_values) else 0.0
            qpos[gripper_idx1] = joint1
            qpos[gripper_idx2] = joint2
            robot.set_qpos(qpos)

        # 验证误差 (与 MANO 参考点的差距)
        if verify and frame is not None:
            scene.update_render()
            finger1_actual = None
            finger2_actual = None
            gripper_link_pos = None
            gripper_link_R = None
            for link in robot.get_links():
                lname = link.get_name()
                if lname == f"{prefix}_gripper_finger_link1":
                    finger1_actual = np.array(link.get_entity_pose().p)
                elif lname == f"{prefix}_gripper_finger_link2":
                    finger2_actual = np.array(link.get_entity_pose().p)
                elif lname == f"{prefix}_gripper_link":
                    pose = link.get_entity_pose()
                    gripper_link_pos = np.array(pose.p)
                    gripper_link_R = pr.matrix_from_quaternion(np.array(pose.q))
            err = {}
            if finger1_actual is not None and mano_finger1_ref is not None:
                err['finger1_mm'] = float(np.linalg.norm(finger1_actual - mano_finger1_ref) * 1000)
            if finger2_actual is not None and mano_finger2_ref is not None:
                err['finger2_mm'] = float(np.linalg.norm(finger2_actual - mano_finger2_ref) * 1000)
            if gripper_link_pos is not None and mano_finger1_ref is not None and mano_finger2_ref is not None:
                midpoint_ref = (mano_finger1_ref + mano_finger2_ref) / 2
                err['wrist_pos_mm'] = float(np.linalg.norm(gripper_link_pos - midpoint_ref) * 1000)
                if gripper_link_R is not None:
                    pointing_cos = np.clip(np.dot(gripper_link_R[:, 0], np.array([1, 0, 0])), -1, 1)
                    err['pointing_deg'] = float(np.degrees(np.arccos(pointing_cos)))
            verify_errors.append(err)

        scene.step()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        status = "OK" if frame is not None else "SKIP"
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Gripper URDF {prefix}  |  {status}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(bgr)

    writer.release()
    _reencode_with_ffmpeg(output, crf, fps, logger)


    # 验证误差报告
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

    logger.info(f"\n✓ 夹爪URDF视频已保存: {output}")
    return output


def render_dual_gripper_video(
    hawor_dir, glb_path, transform_params,
    output="dual_gripper.mp4",
    fps=30, cam_width=1920, cam_height=1080,
    view="fpv", crf=18, start_frame=0, num_frames=-1,
    with_arm=False, smooth=1, verify=False,
    analytical=True, arm_mode="half", viewer=False, logger=None, hand_indices=None,
    strategy="aligned", open_scale=GRIPPER_OPEN_SCALE,
    ras_dir=None, use_ras_cam=False,
):
    """在同一场景中渲染左右夹爪 URDF (双手, 新坐标系)

    从 render_gripper_only.py 照搬并改新坐标。

    Args:
        hawor_dir: HaWoR 数据目录
        glb_path: GLB 文件路径
        transform_params: transform_params.npz 路径
        output: 输出 mp4 路径
        hand_indices: 手部索引列表, 默认 [0, 1]
        其余参数同 render_gripper_only_video
    """
    if logger is None:
        logger = logging.getLogger("002_dual_gripper")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(handler)

    logger.info(f"双夹爪URDF渲染 (新坐标): 同一场景")

    # 加载变换参数
    params = np.load(str(transform_params))
    s = float(params["scale_ratio"])
    R_h2g = params["R_hand_to_glb"]
    t_h2g = params["t_hand_to_glb"]
    Rx_hand = params.get("Rx_hand", np.diag([1.0, -1.0, -1.0]))
    set_render_transform_params(R_h2g, t_h2g, Rx_hand, s)

    # ── 加载相机数据 (HaWoR 或 RAS) ──
    if use_ras_cam and ras_dir:
        R_c2w_all, t_c2w_all = load_ras_cameras(ras_dir)
        _cam_to_sapien = lambda R, t: ras_cam_to_sapien_pose(t, R)
        logger.info(f"  使用 RAS 相机轨迹 ({len(R_c2w_all)} 帧, 已在 GLB 空间)")
    else:
        R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)
        _cam_to_sapien = lambda R, t: hawor_cam_to_sapien_pose(R, t)
        if R_c2w_all is not None:
            logger.info(f"  使用 HaWoR 相机轨迹 ({len(R_c2w_all)} 帧)")

    robot_dir = PROJECT_ROOT / "dex-retargeting" / "assets" / "robots" / "hands"
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
            "finger1_origin_x": 0.03689,
            "finger2_origin_x": 0.03689,
        })

    total_frames = min(gs["total_frames"] for gs in gripper_states)
    if num_frames < 0 or num_frames > total_frames - start_frame:
        num_frames = total_frames - start_frame

    # 初始化 retargeting
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

    glb_path = Path(glb_path)
    if glb_path.exists():
        obj_actors = load_glb_to_sapien(glb_path, s, R_h2g, t_h2g, scene, logger)

    # ── 加载左右夹爪 URDF ──
    for gs in gripper_states:
        prefix = gs["prefix"]
        fox = gs["finger_origin_x"]
        f1x = gs["finger1_origin_x"]
        f2x = gs["finger2_origin_x"]
        if with_arm:
            if arm_mode == "full":
                gripper_urdf_path = prepare_full_arm_urdf(prefix)
                logger.info(f"  {prefix} 渲染URDF: 完整手臂+夹爪")
            else:
                gripper_urdf_path = prepare_half_arm_urdf(prefix)
                logger.info(f"  {prefix} 渲染URDF: 半个手臂+夹爪")
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

    # ── Warm start ──
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
                prefix, fox, f1x, f2x, open_scale, mano_joints=joints_sapien, scale=s)
            g_quat = pr.quaternion_from_matrix(g_R)
            retargeting.warm_start(
                g_pos, g_quat,
                hand_type=hand_type, is_mano_convention=False,
            )
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
                    prefix, gs["finger_origin_x"], gs["finger1_origin_x"], gs["finger2_origin_x"],
                    open_scale, mano_joints=joints_sapien, scale=s)
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

    # ── 设置相机 ──
    camera = scene.add_camera("dual_gripper", cam_width, cam_height, cam_fov, 0.01, 100.0)

    all_wrist_positions = []
    for gs in gripper_states:
        wp = _compute_wrist_positions_sapien(
            gs["hawor_data"], gs["mano_layer"], start_frame, num_frames)
        all_wrist_positions.extend(wp)

    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[0], t_c2w_all[0])
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
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
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

    # ── 视频写入器 / Viewer ──
    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []

    verify_errors = [] if verify else None
    if verify:
        logger.info("  验证模式: 开启")

    if viewer:
        # 绘制相机轨迹对比 (HaWoR=红, RAS=蓝)
        # HaWoR 轨迹
        R_hawor, t_hawor = load_hawor_c2w(hawor_dir)
        if R_hawor is not None and t_hawor is not None:
            hawor_positions = []
            for i in range(len(t_hawor)):
                p, _ = hawor_cam_to_sapien_pose(R_hawor[i], t_hawor[i])
                hawor_positions.append(p)
            hawor_positions = np.array(hawor_positions)
            _draw_camera_trajectory(hawor_positions, [1, 0, 0], context, internal_scene)
            logger.info(f"  HaWoR 相机轨迹已绘制 (红色, {len(hawor_positions)} 点)")
        # RAS 轨迹
        if ras_dir:
            R_ras, t_ras = load_ras_cameras(ras_dir)
            ras_positions = []
            for i in range(len(t_ras)):
                p, _ = ras_cam_to_sapien_pose(t_ras[i], R_ras[i])
                ras_positions.append(p)
            ras_positions = np.array(ras_positions)
            _draw_camera_trajectory(ras_positions, [0, 0, 1], context, internal_scene)
            logger.info(f"  RAS 相机轨迹已绘制 (蓝色, {len(ras_positions)} 点)")

        viewer_win = scene.create_viewer()
        viewer_win.set_camera_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        local_idx = 0
        logger.info("  按 ESC 或关闭窗口退出 Viewer (双夹爪, 红=HaWoR相机, 蓝=RAS相机)")
        while not viewer_win.closed:
            # 安全计算 global_idx，避免越界
            if num_frames <= 0:
                effective_num = 1
            else:
                effective_num = num_frames
            raw_global_idx = start_frame + (local_idx % effective_num)
            if R_c2w_all is not None:
                global_idx = min(raw_global_idx, len(R_c2w_all) - 1)
            else:
                global_idx = raw_global_idx

            if R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            for node in kp_nodes:
                internal_scene.remove_node(node)
            kp_nodes.clear()

            for gs in gripper_states:
                hawor_data = gs["hawor_data"]
                prefix = gs["prefix"]
                robot = gs["robot"]
                retargeting = gs["retargeting"]
                mano_layer = gs["mano_layer"]
                ref_indices = gs["ref_indices"]
                fixed_qpos_gs = gs["fixed_qpos"]
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
                        prefix, gs["finger_origin_x"], gs["finger1_origin_x"], gs["finger2_origin_x"],
                        open_scale, mano_joints=joints_sapien, scale=s)
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
                    retarget_qpos = retargeting.retarget(ref_value, fixed_qpos_gs)
                    g_pos_fk, g_R_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, prefix)
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

            scene.step()
            scene.update_render()
            viewer_win.render()
            local_idx += 1
        logger.info("  Viewer 已关闭")
        return None

    # 当使用 RAS 相机时, 限制帧数不超过相机数据长度
    actual_frames = num_frames
    if R_c2w_all is not None and len(R_c2w_all) < actual_frames:
        actual_frames = len(R_c2w_all)
        logger.warning(f"  相机帧数({len(R_c2w_all)}) < HaWoR帧数({num_frames}), 限制为 {actual_frames} 帧")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    # ── 渲染循环 ──
    for local_idx in trange(actual_frames, desc="双夹爪URDF"):
        global_idx = start_frame + local_idx

        if R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = _cam_to_sapien(R_c2w_all[global_idx], t_c2w_all[global_idx])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

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
            fixed_qpos_gs = gs["fixed_qpos"]
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
                    prefix, gs["finger_origin_x"], gs["finger1_origin_x"], gs["finger2_origin_x"],
                    open_scale, mano_joints=joints_sapien, scale=s)
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
                retarget_qpos = retargeting.retarget(ref_value, fixed_qpos_gs)

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
                mano_finger1_v = joints_sapien[ref_indices[0], :3]
                mano_finger2_v = joints_sapien[ref_indices[1], :3]
                mano_wrist_pos = joints_sapien[0, :3]
                err = {'prefix': prefix}
                if finger1_pos is not None:
                    err[f'{prefix}_finger1_mm'] = float(np.linalg.norm(finger1_pos - mano_finger1_v) * 1000)
                if finger2_pos is not None:
                    err[f'{prefix}_finger2_mm'] = float(np.linalg.norm(finger2_pos - mano_finger2_v) * 1000)
                if gripper_link_pos is not None:
                    err[f'{prefix}_wrist_pos_mm'] = float(np.linalg.norm(gripper_link_pos - mano_wrist_pos) * 1000)
                verify_errors.append(err)
            else:
                scene.step()

            if prefix == "left":
                left_valid = True
            else:
                right_valid = True

        if not verify:
            scene.step()

        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        hand_info = f"L:{'Y' if left_valid else 'N'} R:{'Y' if right_valid else 'N'}"
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Dual Gripper URDF  |  {hand_info}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(bgr)

    writer.release()
    _reencode_with_ffmpeg(output, crf, fps, logger)

    # 验证误差报告
    if verify and verify_errors:
        logger.info("\n  === 验证误差报告 ===")
        for pfx in ['left', 'right']:
            for suffix, label, unit in [('_finger1_mm', '指尖1位置误差', 'mm'),
                                         ('_finger2_mm', '指尖2位置误差', 'mm'),
                                         ('_wrist_pos_mm', '手腕位置误差', 'mm')]:
                key = f'{pfx}{suffix}'
                vals = [e[key] for e in verify_errors if key in e]
                if vals:
                    mean_v = np.mean(vals)
                    max_v = np.max(vals)
                    logger.info(f"  [{pfx}] {label}: mean={mean_v:.2f}{unit}, max={max_v:.2f}{unit}")

    logger.info(f"\n✓ 双夹爪URDF视频已保存: {output}")
    return output


def render_dual_robot_video(
    hawor_dir, glb_path, transform_params,
    output="dual_robot.mp4",
    fps=30, cam_width=1920, cam_height=1080,
    view="fpv", crf=18, start_frame=0, num_frames=-1,
    smooth=1, viewer=False, logger=None, hand_indices=None,
):
    """在同一场景中渲染左右 R1 双臂 (IK 求解 + 共享相机)"""

    logger.info(f"\n{'='*60}")
    logger.info("双手臂渲染 (IK)")
    logger.info(f"{'='*60}")

    from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

    # ── [1/6] 场景 + 坐标 ──
    scene = setup_scene()

    params = np.load(str(transform_params))
    s = float(params["scale_ratio"])
    R_h2g = params["R_hand_to_glb"]
    t_h2g = params["t_hand_to_glb"]
    Rx_hand = params.get("Rx_hand", np.diag([1.0, -1.0, -1.0]))
    set_render_transform_params(R_h2g, t_h2g, Rx_hand, s)

    glb_path = Path(glb_path)
    if glb_path.exists():
        obj_actors = load_glb_to_sapien(glb_path, s, R_h2g, t_h2g, scene, logger)

    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)

    # ── [2/6] 准备双臂状态 ──
    arm_states = []
    for h_idx in hand_indices:
        prefix = "left" if h_idx == 0 else "right"
        hand_type = HandType.left if h_idx == 0 else HandType.right
        logger.info(f"\n--- 加载 {prefix} 臂 ---")

        hawor_data = load_hawor_data(hawor_dir, hand_idx=h_idx)
        if hawor_data is None:
            logger.error(f"未找到 {prefix} 手 HAWOR 数据")
            return None

        robot_dir = PROJECT_ROOT / "dex-retargeting" / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        urdf_path = FLOATING_LEFT_URDF if h_idx == 0 else FLOATING_RIGHT_URDF
        arm_starting = LEFT_ARM_STARTING if h_idx == 0 else RIGHT_ARM_STARTING

        arm_urdf_path = prepare_arm_urdf(urdf_path, arm_prefix=prefix)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.load_multiple_collisions_from_file = True
        robot = loader.load(arm_urdf_path)

        joint_names = [j.name for j in robot.get_active_joints()]
        arm_joint_indices = [i for i, n in enumerate(joint_names) if f"{prefix}_arm_joint" in n]
        gripper_idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")
        gripper_idx2 = joint_names.index(f"{prefix}_gripper_finger_joint2")

        for joint in robot.get_active_joints():
            joint.set_drive_property(stiffness=100000.0, damping=10000.0)

        init_qpos = robot.get_qpos().copy()
        for j, idx in enumerate(arm_joint_indices):
            if j < len(arm_starting):
                init_qpos[idx] = arm_starting[j]
        init_qpos[gripper_idx1] = 0.04
        init_qpos[gripper_idx2] = -0.04
        robot.set_qpos(init_qpos)
        scene.step()
        scene.update_render()
        arm_joint_names = [joint_names[i] for i in arm_joint_indices]

        # Retargeting
        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)
        override = dict(
            add_dummy_free_joint=True,
            normal_delta=1e-5,
            huber_delta=0.01,
            target_link_names=[
                f"{prefix}_gripper_finger_link1",
                f"{prefix}_gripper_finger_link2",
                f"{prefix}_gripper_link",
            ],
            target_link_human_indices=np.array([4, 8, 0]),
        )
        config = RetargetingConfig.load_from_file(config_path, override=override)
        retargeting = config.build()
        ref_indices = retargeting.optimizer.target_link_human_indices
        fixed_retarget_indices = retargeting.optimizer.idx_pin2fixed

        retarget2sapien = np.array(
            [retargeting.joint_names.index(n) for n in joint_names if n in retargeting.joint_names]
        ).astype(int)
        sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(retarget2sapien):
            sapien2retarget[retarget_i] = sapien_i
        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in sapien2retarget:
                fixed_qpos[i] = init_qpos[sapien2retarget[retarget_idx]]

        # IK solver
        ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=[0.1] * 6,
        )
        if h_idx == 0:
            ik_solver.relaxed_ik_left.reset(LEFT_ARM_STARTING)
        else:
            ik_solver.relaxed_ik_right.reset(RIGHT_ARM_STARTING)

        # MANO layer
        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_layer = MANOLayer(prefix, betas_mean)

        # Warm start
        logger.info(f"  Warm start {prefix} ...")
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
            wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
            wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
            wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
            retargeting.warm_start(
                joints_sapien[0, :3], wrist_quat,
                hand_type=hand_type, is_mano_convention=True,
            )
            logger.info(f"  ✓ {prefix} Warm start 完成 (帧 {g_idx})")
            break

        arm_states.append({
            "prefix": prefix,
            "hand_idx": h_idx,
            "hawor_data": hawor_data,
            "mano_layer": mano_layer,
            "retargeting": retargeting,
            "ref_indices": ref_indices,
            "fixed_qpos": fixed_qpos,
            "retarget2sapien": retarget2sapien,
            "robot": robot,
            "joint_names": joint_names,
            "arm_joint_indices": arm_joint_indices,
            "arm_joint_names": arm_joint_names,
            "arm_starting": arm_starting,
            "gripper_idx1": gripper_idx1,
            "gripper_idx2": gripper_idx2,
            "ik_solver": ik_solver,
        })

    # ── [3/6] 放置机器人 ──
    logger.info("\n放置机器人 ...")
    for s_i, st in enumerate(arm_states):
        prefix = st["prefix"]
        wrist_positions = _compute_wrist_positions_sapien(
            st["hawor_data"], st["mano_layer"], start_frame, num_frames)
        if not wrist_positions:
            logger.error(f"无法提取 {prefix} 手腕位置")
            return None
        arm_base_pos, arm_base_q = _compute_optimal_fixed_base(wrist_positions)
        st["arm_base_pos"] = arm_base_pos
        st["arm_base_q"] = arm_base_q
        st["robot"].set_root_pose(sapien.Pose(arm_base_pos.tolist(), arm_base_q.tolist()))
        scene.step()
        scene.update_render()
        logger.info(f"  {prefix} 基座: {arm_base_pos}")

        mapping_offset = np.zeros(3)
        safety_offset = np.zeros(3)
        st["mapping_offset"] = mapping_offset
        st["safety_offset"] = safety_offset

        init_qpos_arm = st["robot"].get_qpos().copy()
        current_joints = np.array([init_qpos_arm[i] for i in st["arm_joint_indices"]])
        joint_filter = LPFilter(alpha=LP_ALPHA_JOINT)
        joint_filter.next(current_joints)
        st["joint_filter"] = joint_filter

        # Warmup
        first_valid_qpos = None
        solve_fn = st["ik_solver"].solve_position_left if prefix == "left" else st["ik_solver"].solve_position_right
        for fi in range(start_frame, start_frame + num_frames):
            if not st["hawor_data"]["pred_valid"][fi]:
                continue
            rot = st["hawor_data"]["pred_rot"][fi]
            trans = st["hawor_data"]["pred_trans"][fi]
            hand_pose = st["hawor_data"]["pred_hand_pose"][fi]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                continue
            _, j = compute_mano_joints(st["mano_layer"], rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)
            ref_value = joints_sapien[st["ref_indices"], :].astype(np.float32)
            retarget_qpos = st["retargeting"].retarget(ref_value, st["fixed_qpos"])
            sapien_qpos = retarget_qpos[st["retarget2sapien"]]
            gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(st["retargeting"], retarget_qpos, prefix)
            tracked_base = arm_base_pos
            st["robot"].set_root_pose(sapien.Pose(tracked_base.tolist(), arm_base_q.tolist()))
            scene.step()
            for link in st["robot"].get_links():
                if f"{prefix}_arm_base_link" == link.get_name():
                    pose = link.get_entity_pose()
                    base_link_p_w = np.array(pose.p)
                    base_link_q_w = np.array(pose.q)
                    break
            base_link_R_w = pr.matrix_from_quaternion(base_link_q_w)
            base_link_R_inv_w = base_link_R_w.T
            ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
            ik_target_b = base_link_R_inv_w @ (ik_target_raw - base_link_p_w)
            ee_R_base = base_link_R_inv_w @ R_ee_world_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
            ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
            for _ in range(IK_SOLVE_PER_FRAME * 5 - 1):
                ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
            first_valid_qpos = np.array(ik_joints)
            break

        if first_valid_qpos is not None:
            init_qpos_arm_arr = np.array(st["arm_starting"])
            for wi in range(WARMUP_FRAMES):
                t = (wi + 1) / WARMUP_FRAMES
                t = t * t * (3 - 2 * t)
                interp = init_qpos_arm_arr * (1 - t) + first_valid_qpos * t
                qpos = st["robot"].get_qpos().copy()
                for j_idx, arm_idx in enumerate(st["arm_joint_indices"]):
                    qpos[arm_idx] = interp[j_idx]
                st["robot"].set_qpos(qpos)
                scene.step()
            logger.info(f"  {prefix} Warmup 完成 ({WARMUP_FRAMES} 帧)")

    # ── [4/6] 设置相机 ──
    logger.info("\n设置相机 ...")
    focal = None
    for st in arm_states:
        f = st["hawor_data"].get("img_focal")
        if f is not None and f > 0:
            focal = f
            break
    if focal is None or focal <= 0:
        focal = HAWOR_FOCAL_DEFAULT
    focal_render = focal * cam_width / 1280.0
    cam_fov = 2 * np.arctan(cam_height / 2.0 / focal_render)
    camera = scene.add_camera("dual_robot", cam_width, cam_height, cam_fov, 0.01, 100.0)

    all_wrist_positions = []
    for st in arm_states:
        wp = _compute_wrist_positions_sapien(
            st["hawor_data"], st["mano_layer"], start_frame, num_frames)
        all_wrist_positions.extend(wp)

    if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
    elif all_wrist_positions:
        centroid = np.mean(all_wrist_positions, axis=0)
        robot_root = np.mean([st["arm_base_pos"] for st in arm_states], axis=0)
        if view == "topdown":
            cam_target = centroid
            cam_pos = cam_target + np.array([0.0, 0.0, 1.2])
            cam_quat = make_look_at_camera(cam_pos, cam_target, up=np.array([0, 1, 0]))
        elif view == "behind":
            cam_pos = robot_root + np.array([2.5, 0.0, 1.2])
            cam_quat = np.array([0.0, 0.0, 1.0, 0.0])
        elif view == "front":
            cam_pos = robot_root + np.array([-2.5, 0.0, 1.2])
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            cam_pos = centroid + np.array([-0.15, -0.20, 0.10])
            cam_quat = make_look_at_camera(cam_pos, centroid)
        camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
    logger.info(f"  视角: {view}, 相机位置: {cam_pos}")

    # ── [5/6] 渲染 ──
    logger.info("\n渲染双机器人视频 ...")
    context = sapien.render.SapienRenderer()._internal_context
    internal_scene = scene.render_system._internal_scene
    kp_nodes = []
    qpos_logs = {st["prefix"]: [] for st in arm_states}

    if viewer:
        # 绘制相机轨迹 (HaWoR=红)
        R_hawor_traj, t_hawor_traj = load_hawor_c2w(hawor_dir)
        if R_hawor_traj is not None and t_hawor_traj is not None:
            hawor_positions = []
            for i in range(len(t_hawor_traj)):
                p, _ = hawor_cam_to_sapien_pose(R_hawor_traj[i], t_hawor_traj[i])
                hawor_positions.append(p)
            hawor_positions = np.array(hawor_positions)
            _draw_camera_trajectory(hawor_positions, [1, 0, 0], context, internal_scene)
            logger.info(f"  HaWoR 相机轨迹已绘制 (红色, {len(hawor_positions)} 点)")

        viewer_win = scene.create_viewer()
        viewer_win.set_camera_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        local_idx = 0
        logger.info("  按 ESC 或关闭窗口退出 Viewer (双臂, 红=HaWoR相机)")
        while not viewer_win.closed:
            global_idx = start_frame + (local_idx % num_frames)

            if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
                cam_pos, cam_quat = hawor_cam_to_sapien_pose(
                    R_c2w_all[global_idx], t_c2w_all[global_idx])
                camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

            for node in kp_nodes:
                internal_scene.remove_node(node)
            kp_nodes.clear()

            for st in arm_states:
                prefix = st["prefix"]
                hawor_data = st["hawor_data"]
                if not hawor_data["pred_valid"][global_idx]:
                    continue
                rot = hawor_data["pred_rot"][global_idx]
                trans = hawor_data["pred_trans"][global_idx]
                hand_pose = hawor_data["pred_hand_pose"][global_idx]
                if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                    continue
                _, j = compute_mano_joints(st["mano_layer"], rot, hand_pose, trans)
                joints_sapien = _render_to_sapien(j)
                kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, st["ref_indices"])
                ref_value = joints_sapien[st["ref_indices"], :].astype(np.float32)
                retarget_qpos = st["retargeting"].retarget(ref_value, st["fixed_qpos"])
                sapien_qpos = retarget_qpos[st["retarget2sapien"]]
                gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(st["retargeting"], retarget_qpos, prefix)
                tracked_base = st["arm_base_pos"]
                st["robot"].set_root_pose(sapien.Pose(tracked_base.tolist(), st["arm_base_q"].tolist()))
                scene.step()
                for link in st["robot"].get_links():
                    if f"{prefix}_arm_base_link" == link.get_name():
                        pose = link.get_entity_pose()
                        base_link_p = np.array(pose.p)
                        base_link_q = np.array(pose.q)
                        break
                base_link_R = pr.matrix_from_quaternion(base_link_q)
                base_link_R_inv = base_link_R.T
                ik_target_raw = gripper_pos_fk + st["mapping_offset"] + st["safety_offset"]
                ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
                ee_R_base = base_link_R_inv @ R_ee_world_fk
                ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
                solve_fn = st["ik_solver"].solve_position_left if prefix == "left" else st["ik_solver"].solve_position_right
                ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
                for _ in range(IK_SOLVE_PER_FRAME - 1):
                    ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
                filtered_joints = st["joint_filter"].next(np.array(ik_joints))
                qpos = st["robot"].get_qpos().copy()
                for j_idx, arm_idx in enumerate(st["arm_joint_indices"]):
                    qpos[arm_idx] = filtered_joints[j_idx]
                if st["gripper_idx1"] < len(sapien_qpos):
                    qpos[st["gripper_idx1"]] = float(sapien_qpos[st["gripper_idx1"]])
                if st["gripper_idx2"] < len(sapien_qpos):
                    qpos[st["gripper_idx2"]] = float(sapien_qpos[st["gripper_idx2"]])
                st["robot"].set_qpos(qpos)
                qpos_logs[st["prefix"]].append(qpos.copy())

            scene.step()
            scene.update_render()
            viewer_win.render()
            local_idx += 1
        logger.info("  Viewer 已关闭")
        return None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (camera.get_width(), camera.get_height()))

    for local_idx in trange(num_frames, desc="双R1臂"):
        global_idx = start_frame + local_idx

        if view == "fpv" and R_c2w_all is not None and t_c2w_all is not None:
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(
                R_c2w_all[global_idx], t_c2w_all[global_idx])
            camera.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))

        for node in kp_nodes:
            internal_scene.remove_node(node)
        kp_nodes.clear()

        left_valid = False
        right_valid = False

        for st in arm_states:
            prefix = st["prefix"]
            hawor_data = st["hawor_data"]
            if not hawor_data["pred_valid"][global_idx]:
                continue
            rot = hawor_data["pred_rot"][global_idx]
            trans = hawor_data["pred_trans"][global_idx]
            hand_pose = hawor_data["pred_hand_pose"][global_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)) or np.any(np.isnan(hand_pose)):
                continue

            _, j = compute_mano_joints(st["mano_layer"], rot, hand_pose, trans)
            joints_sapien = _render_to_sapien(j)
            kp_nodes = _render_keypoints(joints_sapien[:, :3], context, internal_scene, kp_nodes, st["ref_indices"])

            ref_value = joints_sapien[st["ref_indices"], :].astype(np.float32)
            retarget_qpos = st["retargeting"].retarget(ref_value, st["fixed_qpos"])
            sapien_qpos = retarget_qpos[st["retarget2sapien"]]
            gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(st["retargeting"], retarget_qpos, prefix)

            tracked_base = st["arm_base_pos"]
            st["robot"].set_root_pose(sapien.Pose(tracked_base.tolist(), st["arm_base_q"].tolist()))
            scene.step()
            for link in st["robot"].get_links():
                if f"{prefix}_arm_base_link" == link.get_name():
                    pose = link.get_entity_pose()
                    base_link_p = np.array(pose.p)
                    base_link_q = np.array(pose.q)
                    break
            base_link_R = pr.matrix_from_quaternion(base_link_q)
            base_link_R_inv = base_link_R.T
            ik_target_raw = gripper_pos_fk + st["mapping_offset"] + st["safety_offset"]
            ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
            ee_R_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
            solve_fn = st["ik_solver"].solve_position_left if prefix == "left" else st["ik_solver"].solve_position_right
            ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
            for _ in range(IK_SOLVE_PER_FRAME - 1):
                ik_joints = solve_fn(ik_target_b.tolist(), ee_quat_b.tolist())
            filtered_joints = st["joint_filter"].next(np.array(ik_joints))
            qpos = st["robot"].get_qpos().copy()
            for j_idx, arm_idx in enumerate(st["arm_joint_indices"]):
                qpos[arm_idx] = filtered_joints[j_idx]
            if st["gripper_idx1"] < len(sapien_qpos):
                qpos[st["gripper_idx1"]] = float(sapien_qpos[st["gripper_idx1"]])
            if st["gripper_idx2"] < len(sapien_qpos):
                qpos[st["gripper_idx2"]] = float(sapien_qpos[st["gripper_idx2"]])
            st["robot"].set_qpos(qpos)
            qpos_logs[st["prefix"]].append(qpos.copy())

            if prefix == "left":
                left_valid = True
            else:
                right_valid = True

        scene.step()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])

        h, w = bgr.shape[:2]
        cv2.rectangle(bgr, (0, 0), (w, 40), (0, 0, 0), -1)
        hand_info = f"L:{'Y' if left_valid else 'N'} R:{'Y' if right_valid else 'N'}"
        cv2.putText(bgr, f"Frame {local_idx+1}/{num_frames}  |  Dual R1  |  {hand_info}",
                    (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        writer.write(bgr)

    writer.release()
    _reencode_with_ffmpeg(output, crf, fps, logger)
    logger.info(f"\n✓ 双R1臂视频已保存: {output}")

    # 保存 qpos 日志
    output_path = Path(output)
    tracking_dir = output_path.parent.parent / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    for prefix, logs in qpos_logs.items():
        if logs:
            qpos_path = tracking_dir / f"{output_path.stem}_{prefix}.npy"
            np.save(str(qpos_path), np.array(logs))
            logger.info(f"  ✓ {prefix} qpos 已保存: {qpos_path} ({len(logs)} 帧)")

    return output


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _ensure_transform_params(ras_dir, hawor_dir, output_dir, logger):
    """确保 transform_params.npz 存在, 不存在则自动运行 001_align_scene.py"""
    tp_path = os.path.join(output_dir, "transform_params.npz")
    if os.path.exists(tp_path):
        logger.info(f"  transform_params 已存在: {tp_path}")
        return tp_path

    logger.info(f"  transform_params 不存在, 自动运行 001_align_scene.py ...")
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

    combination_dir = Path(__file__).resolve().parent
    align_script = str(combination_dir / "001_align_scene.py")
    cmd = [
        sys.executable, align_script,
        "--ras_output", ras_dir,
        "--hawor_reconstruction", recon_npz,
        "--output_dir", output_dir,
    ]
    logger.info(f"  运行: {' '.join(cmd)}")
    result = sp.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  001_align_scene.py 失败:\n{result.stderr[-500:]}")
        return None

    if os.path.exists(tp_path):
        logger.info(f"  ✓ transform_params 已生成: {tp_path}")
        return tp_path
    else:
        logger.error("  001_align_scene.py 运行完成但未生成 transform_params.npz")
        return None


def _validate_hands(hawor_dir, transform_params, hand_indices, start_frame, num_frames, logger):
    """预校验每只手是否能提取有效手腕位置, 剔除无效手
    如果指定范围内无效, 自动扫描到第一个有效帧.
    返回: (valid_indices, auto_start_frame)
    auto_start_frame: 实际使用的起始帧 (可能不同于 start_frame)"""
    params = np.load(str(transform_params))
    s = float(params["scale_ratio"])
    R_h2g = params["R_hand_to_glb"]
    t_h2g = params["t_hand_to_glb"]

    valid_indices = []
    auto_start = start_frame
    for hi in hand_indices:
        prefix = "left" if hi == 0 else "right"
        try:
            hawor_data = load_hawor_data(hawor_dir, hand_idx=hi)
            # hawor_data["pred_trans"] 已经是 (N, 3) 形状, 不需要再按 hi 索引
            total_frames = hawor_data["pred_trans"].shape[0]
            nf = num_frames
            if nf < 0 or nf > total_frames - start_frame:
                nf = total_frames - start_frame
            betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
            mano_layer = MANOLayer(prefix, betas_mean)
            wrist_positions = _compute_wrist_positions_sapien(
                hawor_data, mano_layer, start_frame, nf)
            if wrist_positions:
                valid_indices.append(hi)
                logger.info(f"  {prefix} 手预检通过: {len(wrist_positions)} 帧有效手腕")
            else:
                # 自动扫描找第一个有效帧 (注意: hawor_data 已按 hand_idx 切片)
                nan_mask = np.isnan(hawor_data["pred_trans"]).any(axis=1)
                valid_frames = np.where(~nan_mask)[0]
                if len(valid_frames) > 0:
                    candidate = valid_frames[0]
                    logger.info(f"  {prefix} 手前 {start_frame} 帧无效, 自动跳到帧 {candidate} 开始")
                    if candidate < total_frames:
                        betas_mean = hawor_data["pred_betas"][candidate].astype(np.float32)
                        mano_layer = MANOLayer(prefix, betas_mean)
                        auto_nf = min(num_frames if num_frames >= 0 else nf, total_frames - candidate)
                        wrist_positions = _compute_wrist_positions_sapien(
                            hawor_data, mano_layer, candidate, auto_nf)
                        if wrist_positions:
                            valid_indices.append(hi)
                            logger.info(f"  {prefix} 手自动扫描通过: {len(wrist_positions)} 帧有效手腕")
                            # 取所有手的最小有效帧作为统一 start_frame
                            auto_start = min(auto_start, candidate)
                            continue
                logger.warning(f"  {prefix} 手无法提取有效手腕位置, 已剔除")
        except Exception as e:
            logger.warning(f"  {prefix} 手预检异常, 已剔除: {e}")
    return valid_indices, auto_start


def _render_single_hand_set(
    hawor_dir, ras_dir, glb_path, transform_params,
    hand_idx, output_dir,
    fps, cam_width, cam_height, view, crf,
    start_frame, num_frames, fixed_base,
    gripper_mode, strategy, open_scale, smooth, viewer, logger,
    use_ras_cam=False,
):
    """渲染单手全套视频: tracking + gripper keypoint + gripper URDF"""
    prefix = "left" if hand_idx == 0 else "right"
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # viewer 模式直接渲染 gripper_only，跳过 tracking/keypoint
    if viewer:
        render_gripper_only_video(
            hawor_dir=hawor_dir, glb_path=glb_path,
            transform_params=transform_params,
            hand_idx=hand_idx, output="viewer_only.mp4",
            fps=fps, cam_width=cam_width, cam_height=cam_height,
            view=view, crf=crf,
            start_frame=start_frame, num_frames=num_frames,
            with_arm=(gripper_mode == "gripper_arm"),
            analytical=True,
            strategy=strategy, open_scale=open_scale,
            viewer=viewer, logger=logger,
            ras_dir=ras_dir, use_ras_cam=use_ras_cam,
        )
        return

    tracking_video = os.path.join(videos_dir, f"hawor_r1_{prefix}_tracking.mp4")
    logger.info(f"\n  ── 渲染 {prefix} 臂 tracking ──")
    render_robot_video(
        hawor_dir=hawor_dir, ras_dir=ras_dir, glb_path=glb_path,
        transform_params=transform_params,
        hand_idx=hand_idx, output=tracking_video,
        fps=fps, cam_width=cam_width, cam_height=cam_height,
        view=view, crf=crf,
        start_frame=start_frame, num_frames=num_frames,
        fixed_base=fixed_base, viewer=viewer, logger=logger,
    )

    gripper_video = os.path.join(videos_dir, f"hawor_r1_{prefix}_gripper.mp4")
    logger.info(f"\n  ── 渲染 {prefix} 夹爪关键点 ──")
    render_gripper_video(
        hawor_dir=hawor_dir, glb_path=glb_path,
        transform_params=transform_params,
        hand_idx=hand_idx, output=gripper_video,
        fps=fps, cam_width=cam_width, cam_height=cam_height,
        view=view, crf=crf,
        start_frame=start_frame, num_frames=num_frames,
        viewer=viewer, logger=logger,
    )

    modes_to_render = []
    if gripper_mode in ("gripper", "both"):
        modes_to_render.append(("gripper", False, ""))
    if gripper_mode in ("gripper_arm", "both"):
        modes_to_render.append(("gripper_arm", True, "_arm"))

    for mode_name, with_arm, mode_suffix in modes_to_render:
        urdf_video = os.path.join(videos_dir, f"hawor_r1_{prefix}_gripper_urdf{mode_suffix}.mp4")
        logger.info(f"\n  ── 渲染 {prefix} 夹爪URDF (mode={mode_name}) ──")
        render_gripper_only_video(
            hawor_dir=hawor_dir, glb_path=glb_path,
            transform_params=transform_params,
            hand_idx=hand_idx, output=urdf_video,
            fps=fps, cam_width=cam_width, cam_height=cam_height,
            view=view, crf=crf,
            start_frame=start_frame, num_frames=num_frames,
            with_arm=with_arm, smooth=smooth,
            analytical=True, arm_mode="half", viewer=viewer, logger=logger,
            strategy=strategy, open_scale=open_scale,
            ras_dir=ras_dir, use_ras_cam=use_ras_cam,
        )


def _render_dual_hand_set(
    hawor_dir, ras_dir, glb_path, transform_params,
    hand_indices, output_dir,
    fps, cam_width, cam_height, view, crf,
    start_frame, num_frames, fixed_base,
    gripper_mode, strategy, open_scale, smooth, viewer, logger,
    use_ras_cam=False,
):
    """渲染双手全套视频: 同场景 dual tracking + 合成 dual gripper keypoint + 同场景 dual gripper URDF"""
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    dual_tracking = os.path.join(videos_dir, "hawor_r1_dual_tracking.mp4")
    logger.info(f"\n  ── 渲染同场景双臂 tracking ──")
    render_dual_robot_video(
        hawor_dir=hawor_dir, glb_path=glb_path,
        transform_params=transform_params,
        output=dual_tracking,
        fps=fps, cam_width=cam_width, cam_height=cam_height,
        view=view, crf=crf,
        start_frame=start_frame, num_frames=num_frames,
        smooth=smooth, viewer=viewer, logger=logger,
        hand_indices=hand_indices,
    )

    modes_to_render = []
    if gripper_mode in ("gripper", "both"):
        modes_to_render.append(("gripper", False, ""))
    if gripper_mode in ("gripper_arm", "both"):
        modes_to_render.append(("gripper_arm", True, "_arm"))

    for mode_name, with_arm, mode_suffix in modes_to_render:
        dual_urdf = os.path.join(videos_dir, f"hawor_r1_dual_gripper_urdf{mode_suffix}.mp4")
        logger.info(f"\n  ── 渲染同场景双夹爪URDF (mode={mode_name}) ──")
        render_dual_gripper_video(
            hawor_dir=hawor_dir, glb_path=glb_path,
            transform_params=transform_params,
            output=dual_urdf,
            fps=fps, cam_width=cam_width, cam_height=cam_height,
            view=view, crf=crf,
            start_frame=start_frame, num_frames=num_frames,
            with_arm=with_arm, smooth=smooth,
            analytical=True, arm_mode="half", viewer=viewer, logger=logger,
            hand_indices=hand_indices,
            strategy=strategy, open_scale=open_scale,
            ras_dir=ras_dir, use_ras_cam=use_ras_cam,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="002_render_scene.py — 新坐标系统渲染")
    parser.add_argument("--mode", default="robot_tracking",
                        choices=["robot_tracking", "topdown", "gripper_only"],
                        help="渲染模式: robot_tracking=tracking+gripper+URDF, topdown=robot_tracking+俯视视角, gripper_only=仅gripper")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="-1=自动检测, 0=左手, 1=右手")
    parser.add_argument("--hawor-dir", required=True)
    parser.add_argument("--ras-dir", required=True)
    parser.add_argument("--glb-path", required=True)
    parser.add_argument("--transform-params", default=None,
                        help="001_align_scene.py 输出的 transform_params.npz 路径; 若未提供, 将在 --output-dir 自动查找或生成")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录 (默认: output/{hawor_dir_name})")
    parser.add_argument("--output", default=None,
                        help="兼容性参数, 等效于 --output-dir")
    parser.add_argument("--view", default="fpv", choices=["fpv", "behind", "front", "topdown"])
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--fixed-base", action="store_true", default=True,
                        help="固定基座模式 (默认开启, 与02一致; --no-fixed-base 关闭)")
    parser.add_argument("--no-fixed-base", dest="fixed_base", action="store_false",
                        help="关闭固定基座, 基座小范围跟踪手腕")
    parser.add_argument("--gripper-mode", default="both",
                        choices=["gripper", "gripper_arm", "both"],
                        help="夹爪URDF渲染方式: gripper=仅夹爪, gripper_arm=夹爪+手臂, both=两者都渲染")
    parser.add_argument("--strategy", default="aligned",
                        choices=["aligned", "analytical", "svd_palm"],
                        help="对齐策略: aligned=3点Gram-Schmidt, analytical=3点旧策略, svd_palm=5点SVD+Gram-Schmidt")
    parser.add_argument("--open-scale", type=float, default=GRIPPER_OPEN_SCALE,
                        help=f"夹爪开合缩放因子 (默认 {GRIPPER_OPEN_SCALE})")
    parser.add_argument("--smooth", type=int, default=1, choices=[0, 1],
                        help="平滑模式: 0=不平滑, 1=EMA (默认 1)")
    parser.add_argument("--viewer", action="store_true", help="交互式Viewer渲染（不保存视频）")
    parser.add_argument("--use-ras-cam", action="store_true",
                        help="使用 RAS 相机轨迹 (已在 GLB 空间) 替代 HaWoR 相机 (默认使用 HaWoR 相机)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Logger
    logger = logging.getLogger("002_render")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)

    # 输出目录
    hawor_name = Path(args.hawor_dir).name
    output_dir = args.output_dir or (args.output and os.path.dirname(args.output)) or f"output/{hawor_name}"
    # Bug-fix: if --output is a file path, output_dir should be its parent, not the filename
    if args.output and not args.output_dir:
        output_dir = os.path.dirname(os.path.abspath(args.output))
    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)

    # 视角映射: topdown 模式
    view = args.view
    if args.mode == "topdown":
        view = "topdown"

    # [1] 自动检测手部 (或用户强制指定)
    if args.hand_idx >= 0:
        hand_indices = [args.hand_idx]
        hand_label = "左手" if args.hand_idx == 0 else "右手"
        logger.info(f"[1/4] 手部指定: {hand_label} (index={args.hand_idx})")
    else:
        hand_indices = detect_hands(args.hawor_dir)
        hand_count = len(hand_indices)
        if hand_count == 0:
            logger.error("[1/4] 手部检测: 未检测到有效手部数据, 停止生成")
            sys.exit(1)
        hand_label = "双手" if hand_count == 2 else ("左手" if hand_indices[0] == 0 else "右手")
        logger.info(f"[1/4] 手部检测: {hand_label} (indices={hand_indices})")

    # [2] 确保 transform_params 存在
    logger.info(f"\n[2/4] 准备 GLB 变换参数 ...")
    if args.transform_params is not None and Path(args.transform_params).exists():
        tp_path = args.transform_params
        logger.info(f"  使用指定 transform_params: {tp_path}")
    else:
        tp_path = _ensure_transform_params(args.ras_dir, args.hawor_dir, output_dir, logger)
        if tp_path is None:
            logger.error("无法获取 transform_params, 停止生成")
            sys.exit(1)

    # [3] 预校验手部 (剔除无效手)
    logger.info(f"\n[3/4] 预校验手部 ...")
    hand_indices, auto_start_frame = _validate_hands(
        args.hawor_dir, tp_path, hand_indices,
        args.start_frame, args.num_frames, logger)
    if len(hand_indices) == 0:
        logger.error("预校验后没有有效手部, 停止生成")
        sys.exit(1)
    if len(hand_indices) == 1:
        logger.info(f"  预校验后按单手渲染: {'左手' if hand_indices[0] == 0 else '右手'}")
    else:
        logger.info(f"  预校验后按双手渲染")
    if auto_start_frame != args.start_frame:
        logger.info(f"  自动调整 start_frame: {args.start_frame} -> {auto_start_frame}")

    # [4] 渲染
    logger.info(f"\n[4/4] 渲染视频 ...")
    start_time = time.time()

    render_kwargs = dict(
        hawor_dir=args.hawor_dir,
        ras_dir=args.ras_dir,
        glb_path=args.glb_path,
        transform_params=tp_path,
        fps=args.fps,
        cam_width=args.width,
        cam_height=args.height,
        view=view,
        crf=args.crf,
        start_frame=auto_start_frame,
        num_frames=args.num_frames,
        fixed_base=args.fixed_base,
        gripper_mode=args.gripper_mode,
        strategy=args.strategy,
        open_scale=args.open_scale,
        smooth=args.smooth,
        viewer=args.viewer,
        logger=logger,
        use_ras_cam=args.use_ras_cam,
    )

    if args.mode == "gripper_only":
        if len(hand_indices) == 2:
            _render_dual_hand_set(hand_indices=hand_indices, output_dir=output_dir, **render_kwargs)
        else:
            _render_single_hand_set(hand_idx=hand_indices[0], output_dir=output_dir, **render_kwargs)
    else:
        # robot_tracking / topdown: 渲染 tracking + gripper keypoint + gripper URDF
        if len(hand_indices) == 2:
            _render_dual_hand_set(hand_indices=hand_indices, output_dir=output_dir, **render_kwargs)
        else:
            _render_single_hand_set(hand_idx=hand_indices[0], output_dir=output_dir, **render_kwargs)

    elapsed = time.time() - start_time
    logger.info(f"\n总耗时: {elapsed:.1f} 秒")


if __name__ == "__main__":
    main()
