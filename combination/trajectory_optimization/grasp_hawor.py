"""grasp_hawor.py — SAPIEN 物理仿真: 用 R1 机器人真实抓取 GLB 物体

项目目标:
  给定 HaWoR 手部重建 + RAS 场景重建 (GLB), 用 R1 机器人 URDF 在 SAPIEN 中
  复刻抓取 GLB 物体的动作, 并通过参数级验证 (物体提升/接触检测).

五阶段流水线 (v4):
  Stage 0: 候选抓取姿态生成 (无需仿真)
  Stage 1: 抓取姿态优化 (6DOF CMA-ES, ~80帧短仿真, "能不能夹住")
  Stage 2: 轨迹重建 (smoothstep插值, 确定性, F46-F54过渡)
  Stage 3: 全局轨迹优化 (CEM, 平滑+MANO跟踪+边界约束)
  Stage 4: 高精度验证与回放

两种 URDF 模式:
  1. full_robot  : r1_v2_1_0.urdf (整个机器人, 臂关节 fixed→revolute)
  2. gripper_only: 纯夹爪 URDF (虚拟6-DOF关节+PD驱动, 无机械臂)

项目文件结构:
  grasp_hawor.py    — 主程序: GraspSimulator 类 + Stage 1-3 逻辑 + main()
  physics_utils.py  — 物理参数常量 (摩擦/密度/PD参数/夹爪几何)
  traj_optimize.py  — 轨迹优化算法 (CEM/CMA-ES/MPPI)
  grasp_controller.py — 夹取控制器 (接触检测/稳定抓取)
  trajectory_loader.py — 轨迹加载 (HaWoR数据/MANO FK/GLB变换)
  grasp_demo.py     — GalaxeaManipSim 参考实现
  models/           — URDF 模板 (robot_forms.py, urdf_templates.py)
  tests/            — 测试和可视化脚本
  docs/             — 设计文档和问题记录
  output/           — 优化结果输出

用法:
  python grasp_hawor.py --test-stage1 --side right --output output/gripper_only_right
  python grasp_hawor.py --test-stage3 --side right --output output/gripper_only_right
  python grasp_hawor.py --mode gripper_only --side right \\
      --hawor-dir /home/an/data/hawor/7 \\
      --ras-dir /home/an/data/ras/my_7mp4_result
"""

import os
_nvidia_icd = '/usr/share/vulkan/icd.d/nvidia_icd.json'
_intel_icd = '/usr/share/vulkan/icd.d/intel_icd.x86_64.json'
if 'VK_ICD_FILENAMES' not in os.environ:
    if os.path.exists(_nvidia_icd):
        try:
            import subprocess
            r = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
            os.environ['VK_ICD_FILENAMES'] = _nvidia_icd if r.returncode == 0 else _intel_icd
        except Exception:
            os.environ['VK_ICD_FILENAMES'] = _intel_icd
    else:
        os.environ['VK_ICD_FILENAMES'] = _intel_icd

import sys
import re
import gc
import json
import logging
import tempfile
import argparse
from pathlib import Path

import numpy as np
import cv2
import sapien
import sapien.render
import torch
from pytransform3d import rotations as pr
from tqdm import trange

# ============================================================
# 从拆分模块导入 (physics_env 必须最先 import 以应用 SAPIEN patch)
# ============================================================
from physics_env import (
    # 路径
    PROJECT_ROOT, GALAXEA_SIM_PATH, R1_ASSETS, R1_MESH_DIR, R1_URDF_DIR,
    FULL_ROBOT_URDF, R1_RIGHT_SETTINGS, R1_LEFT_SETTINGS, COMBINATION_DIR,
    compute_and_save_transform_params,
    # 坐标变换
    R_x, R_AXIS, RXWORLD_TO_SAPIEN,
    # 机器人参数
    RIGHT_ARM_STARTING, LEFT_ARM_STARTING, ARM_MAX_REACH, COMFORTABLE_REACH,
    BASE_BACK_OFFSET, ARM_BASE_OFFSET_RIGHT, ARM_BASE_OFFSET_LEFT,
    COMFORT_TARGET_IN_BASE, RIGHT_ARM_JOINT_LIMITS,
    # 物理参数
    JOINT_STIFFNESS, JOINT_DAMPING, GRIPPER_STIFFNESS, GRIPPER_DAMPING,
    PHYSICS_TIMESTEP, CONTROL_FREQ, DECIMATION, GROUND_HEIGHT,
    OBJECT_DENSITY, OBJECT_MIN_MASS, GRIPPER_FRICTION,
    GRIPPER_INIT_OPEN, GRIPPER_MAX_OPEN, GRIPPER_CLOSE_BIAS, GRIPPER_FORCE,
    _DESCEND_OPEN, _GRIP_CLOSE, _GRIP_OVERCLOSURE,
    GRASP_STRATEGIES, FINGER_FORWARD_NEUTRAL, MAX_ROOT_STEP,
    _FINGER1_ORIGIN, _FINGER1_AXIS, _FINGER2_ORIGIN, _FINGER2_AXIS,
    FINGER_BASE_DIST, GRIPPER_FINGER_HALF_THICKNESS, FINGER_EFFECTIVE_HALF_SPACING,
    _FAILED_SCENES,
    # URDF + 物理 + 接触函数
    prepare_full_robot_urdf, prepare_gripper_only_urdf,
    setup_physics_scene, setup_robot, _is_floating_root,
    physics_step, fetch_contacts,
    get_finger_contacts, get_grasp_force, is_obj_in_gripper_frame,
    # logger
    logger,
)
from data_loader import (
    # 数学工具
    rotmat_to_zyx_euler, rotation_distance,
    # 相机
    hawor_cam_to_sapien_pose, make_look_at_camera,
    # GLB
    compute_glb_ground_z, load_glb_with_physics,
    # HaWoR
    load_hawor_data, load_hawor_c2w, compute_mano_joints,
    compute_analytical_gripper_pose,
    # 物体查找
    find_target_object_by_trajectory, find_pink_object, find_bowl,
    # 控制器
    JointFilter, AdaptiveGraspController, HybridGraspController,
    # 常量
    IK_SOLVE_PER_FRAME, IK_TOLERANCES, LP_ALPHA_JOINT, WARMUP_FRAMES,
    CAM_WIDTH, CAM_HEIGHT, HAWOR_FOCAL_DEFAULT,
    # 力控常量 (GraspSimulator._step_* 方法引用)
    GRASP_TRIGGER_CURL, RELEASE_TRIGGER_CURL, GRASP_RESET_CURL,
    TARGET_GRASP_FORCE, FORCE_CLOSE_STEP, MAX_FORCE_MULTIPLIER,
    CLAMP_OFFSET_MAX, CLAMP_CURL_FLOOR, FORCE_ESTIMATE_COEFF,
)
from stage1_grasp_mixin import Stage1Mixin
from stage2_reconstruct_mixin import Stage2Mixin
from stage3_optimize_mixin import Stage3Mixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("grasp_hawor")


# ============================================================
# 8. 主仿真器
# ============================================================
class GraspSimulator(Stage1Mixin, Stage2Mixin, Stage3Mixin):
    """R1 机器人 SAPIEN 仿真主类, 继承 Stage 1/2/3 Mixin"""

    def __init__(self, hawor_dir, ras_dir, mode="full_robot", side="right",
                 output_dir=None, num_frames=-1, start_frame=0, views="both",
                 grasp_mode="adaptive", viewer=False):
        self.hawor_dir = Path(hawor_dir)
        self.ras_dir = Path(ras_dir)
        self.mode = mode
        self.side = side  # "left" / "right" / "both"
        self.sides = ["left", "right"] if side == "both" else [side]
        self.views = views  # "cam" / "god" / "both" — 指定渲染哪些视角
        self.video_tag = ""  # v4.16: 视频文件名标签 (如 "stage1"/"stage2"/"stage3")
        self.grasp_mode = grasp_mode  # "adaptive" (MANO意图+相位) / "mano" (纯重放)
        self.viewer = viewer  # 是否启用交互式 Viewer
        # 输出目录: 当前脚本下的 output/<mode>_<side>/
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = Path(__file__).resolve().parent
            self.output_dir = script_dir / "output" / f"{mode}_{side}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.hand_indices = {"left": 0, "right": 1}  # hand_idx=0=左手, 1=右手
        self.hand_idx = 0 if side == "left" else 1  # 向后兼容 (单侧)

        # 将日志写入输出目录, 方便用户查看
        self._setup_file_logger()

        self.scene = None
        self.robot_info = None
        self.obj_actors = []
        self.retargeting = None
        self.ik_solver = None
        self.joint_filters = {s: JointFilter() for s in self.sides}  # per-side 滤波器
        self.joint_filter = self.joint_filters[self.sides[0]]  # 向后兼容 (单侧)
        self.transform_params_path = None
        # 自适应抓取控制器 (在 run() 中加载物体后初始化)
        self.grasp_controllers = None  # {side: AdaptiveGraspController}

    def _setup_file_logger(self):
        """配置日志同时输出到终端和输出目录的 log 文件"""
        log_path = self.output_dir / "grasp.log"
        formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        fh = logging.FileHandler(str(log_path), mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        # 避免重复添加 handler
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler) and str(h.baseFilename) == str(log_path):
                return
        logger.addHandler(fh)
        logger.info(f"  日志文件: {log_path}")

    def _find_reconstruction_file(self):
        rec_dir = self.hawor_dir / "reconstruction"
        if rec_dir.exists():
            for f in rec_dir.glob("hawor_results_*.npz"):
                return f
        for f in self.hawor_dir.glob("hawor_results_*.npz"):
            return f
        return None

    def _align_scene(self):
        """调用 01_align_scene.py 对齐 RAS GLB → HaWoR 坐标系

        01_align_scene.py 已在模块顶部加载, compute_and_save_transform_params 是模块级常量
        """
        rec_file = self._find_reconstruction_file()
        if rec_file is None:
            raise FileNotFoundError(f"未找到 HaWoR 重建文件: {self.hawor_dir}")

        align_dir = self.output_dir / "alignment"
        align_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Step 1: 对齐 RAS GLB → HaWoR 坐标系 (调用 01_align_scene.py)")
        logger.info("=" * 60)
        # 优先使用 hawor_dir/output/alignment 下已有的 transform_params (新版 01 输出, 含 R_hand_to_glb)
        # 不存在时才重新计算 (旧版 01 输出不含 R_hand_to_glb, 坐标变换可能不准)
        existing_params = Path(self.hawor_dir) / "output" / "alignment" / "transform_params.npz"
        if existing_params.exists():
            import shutil
            align_dir = self.output_dir / "alignment"
            align_dir.mkdir(parents=True, exist_ok=True)
            target_path = align_dir / "transform_params.npz"
            shutil.copy2(str(existing_params), str(target_path))
            self.transform_params_path = str(target_path)
            logger.info(f"  复用已有 transform_params (含 R_hand_to_glb): {existing_params}")
        else:
            self.transform_params_path = compute_and_save_transform_params(
                ras_output=str(self.ras_dir),
                hawor_reconstruction=str(rec_file),
                output_dir=str(align_dir),
            )
        logger.info(f"  transform_params: {self.transform_params_path}")

    def _init_retargeting(self):
        """初始化 DexRetargeting (仅 full_robot, 支持双手)"""
        if self.mode != "full_robot":
            return

        from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
        from dex_retargeting.retargeting_config import RetargetingConfig

        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        # 每侧创建独立 retargeting 实例
        self.retargeting = {}  # {"left": instance, "right": instance}
        self._ref_indices = {}
        self._retarget2sapien = {}
        self._sapien2retarget = {}
        self._fixed_qpos = {}

        sapien_joint_names = self.robot_info["joint_names"]
        init_qpos = self.robot_info["init_qpos"]

        for s in self.sides:
            hand_type = HandType.right if s == "right" else HandType.left
            config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)

            override = dict(
                add_dummy_free_joint=True,
                normal_delta=1e-5,
                huber_delta=0.01,
                target_link_names=[
                    f"{s}_gripper_finger_link1",
                    f"{s}_gripper_finger_link2",
                    f"{s}_gripper_link",
                ],
                target_link_human_indices=np.array([4, 8, 0]),
            )
            config = RetargetingConfig.load_from_file(config_path, override=override)
            self.retargeting[s] = config.build()
            self._ref_indices[s] = self.retargeting[s].optimizer.target_link_human_indices

            # retargeting ↔ sapien 关节映射
            retarget_joint_names = self.retargeting[s].joint_names
            self._retarget2sapien[s] = np.array(
                [retarget_joint_names.index(n) for n in sapien_joint_names if n in retarget_joint_names]
            ).astype(int)
            self._sapien2retarget[s] = {r: i for i, r in enumerate(self._retarget2sapien[s])}

            fixed_retarget_indices = self.retargeting[s].optimizer.idx_pin2fixed
            self._fixed_qpos[s] = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
            for i, retarget_idx in enumerate(fixed_retarget_indices):
                if retarget_idx in self._sapien2retarget[s]:
                    sapien_idx = self._sapien2retarget[s][retarget_idx]
                    if sapien_idx < len(init_qpos):
                        self._fixed_qpos[s][i] = init_qpos[sapien_idx]

            logger.info(f"  DexRetargeting 已初始化 ({s}), ref_indices={self._ref_indices[s]}")

    def _init_ik(self):
        """初始化 RelaxedIK (仅 full_robot, 支持双手)"""
        if self.mode != "full_robot":
            return

        from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

        self.ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=IK_TOLERANCES,
        )
        for s in self.sides:
            arm_starting = RIGHT_ARM_STARTING if s == "right" else LEFT_ARM_STARTING
            if s == "right":
                self.ik_solver.relaxed_ik_right.reset(arm_starting)
            else:
                self.ik_solver.relaxed_ik_left.reset(arm_starting)
        logger.info(f"  RelaxedIK 已初始化 (sides={self.sides})")

    def _get_gripper_pose_from_retargeting(self, retarget_qpos, side=None):
        """从 retargeting FK 获取期望夹爪位姿

        Args:
            retarget_qpos: retargeting 输出的关节角
            side: 指定侧 ("left"/"right"); None 时用 self.sides[0]
        """
        s = side if side is not None else self.sides[0]
        internal_robot = self.retargeting[s].optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)
        target_name = f"{s}_gripper_link"
        for i, name in enumerate(internal_robot.link_names):
            if name == target_name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        raise RuntimeError(f"内部机器人中找不到 {target_name}")

    def _compute_optimal_base(self, wrist_positions_sapien, R_c2w_all=None):
        """计算最优固定基座 — 使臂基座 (非 ROOT) 在手腕上方 COMFORTABLE_REACH 处

        R1 机器人臂基座比 ROOT 高 ~1.4m (躯干), 必须减去此偏移,
        否则臂基座远离手腕, IK 目标超出臂展.

        朝向修复: root_quat 根据 R_c2w_all 第一帧相机 forward 计算 yaw,
        使机器人 +X 对齐相机水平 forward (修复 90 度偏转: 默认单位四元数面向 +X,
        但相机看向 -Z, 差 90 度).
        """
        # 双手模式用两侧偏移的平均 (y=0, 因为左右对称)
        if self.side == "both":
            arm_base_offset = (ARM_BASE_OFFSET_LEFT + ARM_BASE_OFFSET_RIGHT) / 2.0
        else:
            arm_base_offset = ARM_BASE_OFFSET_LEFT if self.side == "left" else ARM_BASE_OFFSET_RIGHT
        if len(wrist_positions_sapien) == 0:
            root_pos = np.array([0.0, 0.0, COMFORTABLE_REACH]) - arm_base_offset
            root_quat = self._compute_robot_yaw_quat(R_c2w_all)
            return root_pos, root_quat

        wrist_arr = np.array(wrist_positions_sapien)
        centroid = wrist_arr.mean(axis=0)

        # 计算机器人 forward (用于让臂基座后退)
        root_quat = self._compute_robot_yaw_quat(R_c2w_all)
        try:
            # 用四元数旋转 [1,0,0] (URDF 默认前方) 得到机器人当前前方
            # pytransform3d 四元数格式 [w, x, y, z], 用 matrix_from_quaternion 转矩阵再乘向量
            R_root = pr.matrix_from_quaternion(root_quat)
            forward_3d = R_root @ np.array([1.0, 0.0, 0.0])
            forward_2d = np.array([forward_3d[0], forward_3d[1], 0.0])
            norm = float(np.linalg.norm(forward_2d))
            forward_2d = forward_2d / norm if norm > 1e-6 else np.array([0.0, -1.0, 0.0])
        except Exception:
            forward_2d = np.array([0.0, -1.0, 0.0])

        # 目标臂基座位置 = 手腕质心正上方 COMFORTABLE_REACH, 沿 forward 反方向后退 BASE_BACK_OFFSET
        # (让机器人退后一点, 不挡物体; 对齐 04 BASE_OFFSET_Y=0.30 思路)
        desired_arm_base = centroid.copy()
        desired_arm_base[:2] -= forward_2d[:2] * BASE_BACK_OFFSET
        desired_arm_base[2] = centroid[2] + COMFORTABLE_REACH

        # ROOT 位置 = 臂基座位置 - 臂基座偏移
        root_pos = desired_arm_base - arm_base_offset

        # 检查手腕到臂基座的距离是否在臂展内
        max_dist = np.max(np.linalg.norm(wrist_arr - desired_arm_base, axis=1))
        if max_dist > ARM_MAX_REACH * 0.9:
            scale = ARM_MAX_REACH * 0.9 / max_dist
            offset = desired_arm_base - centroid
            desired_arm_base[:2] = centroid[:2] + offset[:2] * scale
            desired_arm_base[2] = centroid[2] + offset[2] * scale
            root_pos = desired_arm_base - arm_base_offset

        logger.info(f"  最优 ROOT: {root_pos.round(3)}, 臂基座将在: {desired_arm_base.round(3)} "
                    f"(后退 {BASE_BACK_OFFSET}m, forward={forward_2d.round(3)})")
        logger.info(f"  最远手腕距离: {max_dist:.4f}m (臂展={ARM_MAX_REACH}m)")
        return root_pos, root_quat

    def _compute_robot_yaw_quat(self, R_c2w_all=None):
        """根据第一帧相机 R_c2w 计算机器人 yaw 四元数 (绕 Z 轴)

        机器人 URDF 默认 +X 为前方, 相机 (OpenGL) 看向 -Z.
        旧版用单位四元数 (面向 +X), 与相机朝向差 90 度 → 视频中机器人侧着身.
        现在用相机 forward 的水平分量算 yaw, 让机器人面向相机看的方向.
        """
        if R_c2w_all is None or len(R_c2w_all) == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])  # 默认: 面向 +X
        try:
            R_c2w = R_c2w_all[min(self.start_frame, len(R_c2w_all) - 1)]
            cam_R_sapien = RXWORLD_TO_SAPIEN @ R_c2w
            forward = -cam_R_sapien[:, 2]  # OpenGL -Z forward
            yaw = float(np.arctan2(forward[1], forward[0]))
            root_quat = pr.quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0, yaw]))
            logger.info(f"  机器人朝向: yaw={np.degrees(yaw):.1f}° (对齐相机 forward={forward.round(3)})")
            return root_quat
        except Exception as e:
            logger.warning(f"  计算相机朝向失败: {e}, 使用默认朝向 (+X)")
            return np.array([1.0, 0.0, 0.0, 0.0])

    def _make_horizontal_closing_R(self, R_mano):
        """调整 R 使手指闭合方向 (Y轴) 水平, 保留 R_X (手指前向).

        MANO 手势的 R_Y 常有大的 Z 分量 (垂直闭合), 无法夹取地面物体:
        一根手指在物体上方, 另一根被地面挡住. 旋转 R 绕 X 轴使 R_Y 水平,
        实现水平捏合 (最小姿态调整: 仅改闭合方向, 保留前向).

        用户: "不要调整姿态" — 这里仅在 CLOSE 窗口调整, 且只旋转 R_X 轴
        (手指前向不变), 是实现抓取的最小必要姿态偏离.
        """
        R_X = R_mano[:, 0].astype(np.float64)
        R_X = R_X / np.linalg.norm(R_X)
        world_z = np.array([0.0, 0.0, 1.0])
        R_Y = np.cross(world_z, R_X)
        norm_Y = float(np.linalg.norm(R_Y))
        if norm_Y < 1e-6:
            R_Y = np.array([0.0, 1.0, 0.0])
        else:
            R_Y = R_Y / norm_Y
        R_Z = np.cross(R_X, R_Y)
        R_Z = R_Z / np.linalg.norm(R_Z)
        return np.column_stack([R_X, R_Y, R_Z])

    @staticmethod
    def _normalize_R(R):
        """Re-orthogonalize rotation matrix (fix numerical drift for slerp)."""
        U, _, Vt = np.linalg.svd(R)
        R_ortho = U @ Vt
        if np.linalg.det(R_ortho) < 0:
            U[:, -1] *= -1
            R_ortho = U @ Vt
        return R_ortho

    def _get_actual_obj_pos(self, side):
        """从 obj_actors 获取实际物体位置 (物理仿真中的真实位置)"""
        _side_ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        if _side_ctrl and _side_ctrl.target_obj and hasattr(self, 'obj_actors'):
            for _ac in self.obj_actors:
                if _side_ctrl.target_obj in _ac.get_name():
                    return np.array(_ac.get_pose().p, dtype=np.float64)
        return None

    def _compute_mano_neutral_target(self, local_idx, side, opt_params=None):
        """三段控制: APPROACH/CLOSE/RELEASE, 夹取独立于位姿优化

        核心原则:
          - 位姿优化只管"夹爪去哪", 手指开合是固定策略
          - CLOSE 阶段: 夹爪跟随优化轨迹, 物体被夹住后自然跟随
          - 物体位置从 obj_actors 动态获取, 不硬编码
          - 全阶段使用 +90° Y 旋转 (手指朝下)
        """
        def _smoothstep(x):
            x = np.clip(x, 0, 1)
            return x * x * (3 - 2 * x)

        traj = getattr(self, '_opt_mano_gripper_traj', {}).get(side)
        if traj is None:
            traj = getattr(self, '_mano_gripper_traj', {}).get(side)
        if traj is None or len(traj["pos"]) == 0:
            return None

        N = len(traj["pos"])
        local_idx_safe = min(local_idx, N - 1)

        _fp = getattr(self, '_frame_params', None)
        _fo = getattr(self, '_fixed_offsets_654', None)

        if _fp is not None and _fo is not None and opt_params is not None:
            from traj_optimize import generate_trajectory_from_params

            traj_cache = getattr(self, '_traj_654_cache', None)
            if traj_cache is None or traj_cache.get('id') != id(opt_params):
                _all_fixed = sorted(set(_fp['fixed_frames']) | set(_fo.keys()))
                opt_pos, opt_R = generate_trajectory_from_params(
                    np.array(traj["pos"]), np.array(traj["R"]),
                    opt_params, fixed_offsets=_fo,
                    fixed_frames=_all_fixed,
                )
                traj_cache = {
                    'id': id(opt_params),
                    'opt_pos': opt_pos,
                    'opt_R': opt_R,
                }
                self._traj_654_cache = traj_cache

            _F50 = min(_fp['F50_IDX'], N - 1)
            _F95 = N  # v15f: CLOSE 持续到最后一帧, 不进入 RELEASE (避免物体掉回导致 lift=0)

            # v15e: 预计算 z 累积最大值 (从 F50 开始, 只升不降)
            # v15d 问题: opt_pos 的 z 波动剧烈 (F52=0.07, F53=0.02, F55=0.01)
            # 导致夹爪上下跳动把物体甩掉 (F53 抬起3.4cm, F60 掉回0.3cm)
            # 修复: z 只能上升不能下降, 物体被持续抬起
            if 'z_cummax' not in traj_cache:
                _z_full = traj_cache['opt_pos'][:, 2].copy()
                _z_full[_F50:] = np.maximum.accumulate(traj_cache['opt_pos'][_F50:, 2])
                traj_cache['z_cummax'] = _z_full

            # 获取实际物体位置
            _obj_pos = self._get_actual_obj_pos(side)
            if _obj_pos is None:
                _obj_pos = traj_cache['opt_pos'][_F50]
            # v15m: 保存初始物体位置 (CLOSE 阶段用, 防止物体被推高后 _enclose_z 跟着变高)
            if '_obj_init_z' not in traj_cache:
                traj_cache['_obj_init_z'] = float(_obj_pos[2])
            _obj_init_z = traj_cache['_obj_init_z']

            # v4.7: 根据目标物体尺寸计算闭合 qpos, 避免对 2cm 级物体过度闭合而推出
            _side_ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
            _target_name = _side_ctrl.target_obj if _side_ctrl else None
            if '_gripper_close_qpos' not in traj_cache:
                _bbox_size = self.obj_info.get(_target_name, {}).get('bbox_size', None)
                traj_cache['_gripper_close_qpos'] = float(self.compute_gripper_qpos(_bbox_size))
            gripper_close_qpos = traj_cache['_gripper_close_qpos']

            # +90° Y 旋转: 手指朝下
            from scipy.spatial.transform import Rotation as R
            _rot_close = R.from_euler("y", +90, degrees=True).as_matrix()

            # CLOSE 阶段夹爪 base 位置补偿 (借鉴 GalaxeaManipSim 的 -0.12m 后退补偿)
            # +90°Y 旋转后: 手指(局部+X 0.03689m) → 世界 -Z 方向 3.689cm
            # 要手指到物体高度: base_z = obj_z + 0.03689 (物体上方 3.7cm)
            # 抓取中心(两指中点) 旋转后世界偏移 = [0, 0, -0.03689]
            # v15m: 旋转后的 link offset — base→finger 在 _rot_close 下的真实偏移
            # URDF: finger origin xyz=[0.03689, ±0.013453, ±0.00012] → 手指在 base +X 方向 3.689cm
            # _rot_close (+90°Y) 旋转后: rot @ [0.03689, 0, 0] = [0, 0, -0.03689] → 手指朝下
            # 所以 base 应该在物体正上方 (xy 对齐, z = obj_z + 0.03689)
            _GRIPPER_LINK_OFFSET_LOCAL = np.array([0.03689, 0.0, 0.0])  # URDF 中 finger 在 base +X 方向
            _GRIPPER_LINK_OFFSET_ROTATED = _rot_close @ _GRIPPER_LINK_OFFSET_LOCAL  # 旋转后 = [0, 0, -0.03689]
            # v15m: 用初始物体位置计算 base_close (避免正反馈循环)
            # base_close = obj_pos - rotated_offset = obj_pos - [0, 0, -0.03689] = obj_pos + [0, 0, 0.03689]
            # 即 base 在物体正上方 3.689cm, xy 与物体对齐
            if '_gripper_base_close_cached' not in traj_cache:
                _init_obj = np.array([_obj_pos[0], _obj_pos[1], _obj_init_z])
                traj_cache['_gripper_base_close_cached'] = _init_obj - _GRIPPER_LINK_OFFSET_ROTATED
            _gripper_base_close = traj_cache['_gripper_base_close_cached']
            _approach_safe_z = 0.12
            # v15l: z 漂移补偿
            # v15k 问题: _DRIFT_COMP=0.02 时手指在物体下方2cm (vlock 消除了PD漂移, 不再需要补偿)
            #   v15j5: _DRIFT_COMP=0.01, 无 vlock, PD 漂移 3cm → 手指在上方 1cm
            #   v15k:  _DRIFT_COMP=0.02, 无 vlock, 仍漂移 3cm → 手指在上方 1cm (补偿不够)
            #   v15l:  _DRIFT_COMP=0.02, 有 vlock, PD 漂移被消除 → 手指在下方 2cm (过度补偿!)
            # 修复: vlock 消除了 PD 漂移, 不再需要 _DRIFT_COMP
            _DRIFT_COMP = 0.0  # v15l: 设为 0 (vlock 消除了 PD 漂移, 不再需要补偿)
            # v15l: 渐进闭合 — 更早预闭合 + 受控闭合 + 给物体空间
            #   用户: "闭合直接为0吗，那不是还要给物体空间"
            #   用户: "你得再F50之前就开始闭合夹爪"
            #   策略: F25 开始预闭合 → F50 到达 PRE_GRASP → F60 接触 → F70+ 维持
            #   关键: 手指 PD 从 GRIPPER_MAX_OPEN(5cm) 闭合到 4mm 需 ~40帧
            #         F25 开始预闭合, 到 F60 约 35 帧, PD 刚好收敛到目标
            # v15n: 两阶段闭合策略
            #   阶段1: 下降时手指中张 (qpos=0.012, y_gap≈5.1cm > 物体3.5cm, 不碰物体)
            #   阶段2: 到达物体中心后, 逐渐闭合 (每帧减1mm, 10帧从0.012→0.002)
            #   v15n3 问题: set_qpos+vlock 瞬间闭合 0.012→0.002, 物体被推出7.69cm
            #   修复: 渐进闭合, 每帧减小 qpos, 让物理引擎逐步处理重叠
            # v4.7 Stage2/3: 动态闭合 qpos + 推迟闭合, 避免 F50 跳变把物体推飞
            # APPROACH (F0-F50): 保持手指中张, F40 后下降到物体上方 4cm
            # CLOSE (F50-F95): F50-F60 边下沉到物体中心边闭合, F60-F75 保持夹持, F75-F95 抬升
            _descend_start = max(0, _F50 - 10)   # F40 开始下降
            _approach_end_z = _obj_init_z + 0.04  # 手指在物体中心上方约 3mm
            _enclose_z = _obj_init_z + 0.0369   # 手指正对物体中心

            if local_idx_safe < _F50:
                gripper_pos = _gripper_base_close.copy()
                if local_idx_safe < _descend_start:
                    gripper_pos[2] = _approach_safe_z
                else:
                    _t_desc = (local_idx_safe - _descend_start) / max(_F50 - _descend_start, 1)
                    _s = _smoothstep(_t_desc)
                    gripper_pos[2] = _approach_safe_z * (1 - _s) + _approach_end_z * _s
                # APPROACH 阶段手指保持中张, 到 F50 再开始闭合
                gripper_val = _DESCEND_OPEN
                gripper_R = _rot_close
                phase = "APPROACH"

            elif local_idx_safe < _F95:
                gripper_pos = np.zeros(3)
                gripper_R = _rot_close
                gripper_pos[0] = _gripper_base_close[0]
                gripper_pos[1] = _gripper_base_close[1]
                _enclose_end = _F50 + 10   # F60: 下沉+闭合完成
                _grip_end = _enclose_end + 15  # F75: 维持夹持
                _lift_end = min(_F50 + 45, _F95 - 1)  # F95: 抬升结束
                if local_idx_safe <= _enclose_end:
                    # F50→F60: 边下沉边闭合 (边走边夹)
                    _t = (local_idx_safe - _F50) / max(_enclose_end - _F50, 1)
                    _s = _smoothstep(_t)
                    gripper_pos[2] = _approach_end_z * (1 - _s) + _enclose_z * _s
                    gripper_val = _DESCEND_OPEN - (_DESCEND_OPEN - gripper_close_qpos) * _s
                elif local_idx_safe <= _grip_end:
                    # F60→F75: 停在物体中心, 手指闭合到物体尺寸
                    gripper_pos[2] = _enclose_z
                    gripper_val = gripper_close_qpos
                elif local_idx_safe <= _lift_end:
                    # F75→F95: 抬升
                    _z_cummax_val = traj_cache['z_cummax'][local_idx_safe]
                    _n_lift = local_idx_safe - _grip_end
                    _active_z = _enclose_z + 0.002 * _n_lift
                    gripper_pos[2] = max(_z_cummax_val, _active_z)
                    gripper_val = gripper_close_qpos
                else:
                    _z_cummax_val = traj_cache['z_cummax'][_lift_end]
                    _n_lift = _lift_end - _grip_end
                    _active_z = _enclose_z + 0.002 * _n_lift
                    gripper_pos[2] = max(_z_cummax_val, _active_z)
                    gripper_val = gripper_close_qpos
                phase = "CLOSE"

            else:
                # RELEASE (F95-F112): 跟随 MANO 轨迹, 手指张开释放
                gripper_pos = traj_cache['opt_pos'][local_idx_safe]
                gripper_R = traj_cache['opt_R'][local_idx_safe]
                gripper_val = GRIPPER_MAX_OPEN
                phase = "RELEASE"

            # 安全: z 不能低于地面
            _z_min = 0.01
            if gripper_pos[2] < _z_min:
                gripper_pos[2] = _z_min

            return gripper_pos, gripper_R, gripper_val, phase, 0.5

        # ============================================================
        # 旧路径 (回退, 无 _frame_params 时使用)
        # ============================================================
        f_grasp = getattr(self, '_mano_grasp_frame', {}).get(side)
        target_pos = getattr(self, '_mano_target_pos', {}).get(side)
        offset = getattr(self, '_mano_neutral_offset', {}).get(side)
        if offset is None or f_grasp is None or target_pos is None:
            return None

        mano_pos = traj["pos"][local_idx_safe]
        mano_j1 = float(traj["j1"][local_idx_safe])
        if "R" in traj and len(traj["R"]) > local_idx_safe:
            mano_R = traj["R"][local_idx_safe]
        else:
            mano_R = np.eye(3, dtype=np.float64)

        n = max(getattr(self, 'num_frames', N), 1)
        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        has_bowl = ctrl is not None and ctrl.bowl_pos is not None
        close_dur = max(20, int(n * 0.20))
        close_start = max(0, f_grasp - 30)
        close_end = min(close_start + close_dur, n - 1)

        if has_bowl:
            remaining = max(1, n - close_end)
            release_end = close_end + max(1, int(remaining * 0.8))
            if local_idx_safe < close_start:
                phase = "APPROACH"
            elif local_idx_safe < close_end:
                phase = "CLOSE"
            elif local_idx_safe < release_end:
                phase = "TRANSPORT"
            else:
                phase = "RELEASE"
        else:
            if local_idx_safe < close_start:
                phase = "APPROACH"
            elif local_idx_safe < close_end:
                phase = "CLOSE"
            else:
                phase = "HOLD"

        mano_target_pos = mano_pos + offset

        # 旧 CEM/CMA-ES 路径: 按 opt_params 维度分发
        if opt_params is not None and len(opt_params) == 3:
            from traj_optimize import apply_xyz_offset
            gripper_pos, gripper_R = apply_xyz_offset(mano_target_pos, mano_R, opt_params)
            # CEM 路径: HOLD 阶段保持手指闭合夹紧物体 (保持抓握), 而非张开
            gripper_val = mano_j1 if local_idx_safe < close_start else (0.0 if local_idx_safe < close_end else (0.0 if (has_bowl and local_idx_safe < release_end) else 0.0))
            return gripper_pos, gripper_R, gripper_val, phase, 0.5

        if opt_params is not None and len(opt_params) == 6:
            from traj_optimize import apply_xyz_offset
            from scipy.spatial.transform import Rotation as R
            gripper_pos, gripper_R = apply_xyz_offset(mano_target_pos, mano_R, opt_params[:3])
            if np.linalg.norm(opt_params[3:6]) > 1e-8:
                gripper_R = R.from_euler("xyz", opt_params[3:6]).as_matrix() @ mano_R
            # CEM-6D 路径: HOLD 阶段保持手指闭合夹紧物体 (保持抓握), 而非张开
            gripper_val = mano_j1 if local_idx_safe < close_start else (0.0 if local_idx_safe < close_end else (0.0 if (has_bowl and local_idx_safe < release_end) else 0.0))
            return gripper_pos, gripper_R, gripper_val, phase, 0.5

        if opt_params is not None and len(opt_params) == 42:
            from traj_optimize import interp_keyframes, apply_keyframe_offset
            kf_cache = getattr(self, '_kf_cache', None)
            if kf_cache is None or kf_cache.get('id') != id(opt_params):
                total = N
                kf_cache = {'id': id(opt_params), 'offsets': interp_keyframes(opt_params, total)}
                self._kf_cache = kf_cache
            offset_6d = kf_cache['offsets'][local_idx_safe]
            gripper_pos, gripper_R = apply_keyframe_offset(mano_target_pos, mano_R, offset_6d)
            gripper_val = mano_j1 if local_idx_safe < close_start else (0.0 if local_idx_safe < close_end else (0.0 if (has_bowl and local_idx_safe < release_end) else GRIPPER_MAX_OPEN))
            if has_bowl and phase in ("TRANSPORT", "RELEASE"):
                bowl = ctrl.bowl_pos
                finger_forward_z = abs(float(gripper_R[2, 0])) * 0.037
                bowl_safe_z = float(bowl[2]) + 0.15 + finger_forward_z
                if gripper_pos[2] < bowl_safe_z:
                    gripper_pos = gripper_pos.copy()
                    gripper_pos[2] = bowl_safe_z
            return gripper_pos, gripper_R, gripper_val, phase, 0.5

        # 第十八轮默认 9D 路径
        if opt_params is not None and len(opt_params) >= 9:
            grasp_pos_delta = opt_params[0:3]
            grasp_R_euler = opt_params[3:6]
            finger_close_target = float(opt_params[6])
            close_blend_ratio = float(opt_params[7])
            transport_vel_limit = float(opt_params[8])

            if phase == "CLOSE":
                gripper_R = self._make_horizontal_closing_R(mano_R)
                _close_tgt = target_pos
                _ctrl_close = self.grasp_controllers.get(side) if self.grasp_controllers else None
                if _ctrl_close and _ctrl_close.target_obj and hasattr(self, 'obj_actors'):
                    for _ac in self.obj_actors:
                        if _ctrl_close.target_obj in _ac.get_name():
                            _close_tgt = np.array(_ac.get_pose().p, dtype=np.float64)
                            break
                t = _smoothstep((local_idx_safe - close_start) / max(close_end - close_start, 1))
                grasp_pos = mano_target_pos + grasp_pos_delta
                gripper_pos = grasp_pos * (1 - t) + _close_tgt * t
                gripper_val = finger_close_target
            elif phase in ("TRANSPORT", "RELEASE"):
                t = _smoothstep((local_idx_safe - close_end) / max(n - close_end, 1))
                bowl_release_pos = target_pos.copy()
                bowl_release_pos[2] += 0.3
                retreat_pos = bowl_release_pos.copy()
                retreat_pos[2] += 0.1
                if has_bowl and ctrl and ctrl.bowl_pos is not None:
                    bowl_release_pos = np.array(ctrl.bowl_pos, dtype=np.float64)
                    bowl_release_pos[2] += 0.25
                    retreat_pos = bowl_release_pos.copy()
                    retreat_pos[2] += 0.1
                elif phase == "RELEASE":
                    gripper_pos = bowl_release_pos
                    gripper_val = GRIPPER_MAX_OPEN
                    phase = "RELEASE"
                    return gripper_pos, gripper_R, gripper_val, phase, transport_vel_limit
                else:
                    gripper_pos = bowl_release_pos * (1 - t) + retreat_pos * t
                    gripper_val = GRIPPER_MAX_OPEN
                    phase = "RETREAT"
            else:
                gripper_pos = mano_target_pos
                gripper_val = mano_j1

            gripper_R_fixed = self._make_horizontal_closing_R(
                traj["R"][local_idx_safe] if len(traj["R"]) > local_idx_safe else np.eye(3))
            if phase in ("APPROACH",):
                gripper_R = mano_R
            else:
                gripper_R = gripper_R_fixed
            return gripper_pos, gripper_R, gripper_val, phase, transport_vel_limit

        # 无 opt_params 回退
        gripper_pos = mano_pos + offset
        gripper_val = mano_j1 if local_idx_safe < close_start else 0.0
        return gripper_pos, mano_R, gripper_val, phase, 0.5

    def _compute_grasp_demo_target(self, local_idx, side):
        """grasp_demo 式预规划轨迹 (用户: "像 grasp_demo.py 一样真正的抓取物体")

        完全绕过 MANO arm 轨迹, 直接规划到目标物体位置:
          Phase 1 (APPROACH):  F0-F25%     → 从初始 EE 位置移动到 (grasp_pos + [0,0,8cm])
          Phase 2 (DESCEND):   F25%-F50%   → 下降到 grasp_pos, gripper 保持 MAX_OPEN
          Phase 3 (CLOSE):    F50%-F80%   → EE 保持 grasp_pos, gripper 闭合到 0 (PD 需充分时间收敛)
          Phase 4 (LIFT):      F80%-F100%  → 从 grasp_pos 提升到 (grasp_pos + [0,0,20cm])

        Pick-and-Place 模式 (用户: "夹住粉色的东西, 放到碗里面"):
          在 LIFT 之后新增:
          Phase 5 (TRANSPORT): F55%-F75%   → 水平移动到碗上方 (bowl_pos + [0,0,15cm]), gripper 闭合
          Phase 6 (RELEASE):   F75%-F85%   → 在碗上方打开夹爪, 物体掉入碗中
          Phase 7 (RETREAT):   F85%-F100%  → 后退到碗上方 30cm, gripper 全开

        关键: DESCEND 和 CLOSE 分开 (对齐 grasp_demo.py: 先到位再闭合).
        之前边下降边闭合会导致手指在物体上方就闭合, 把物体推开而非夹住.

        用 smoothstep 插值, 固定 top-down 抓取朝向 (gripper X 朝下).

        Returns:
            (gripper_pos, gripper_R, gripper_val, phase) or None (无目标时退回 MANO)
        """
        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        if target_name is None or target_name not in self.obj_bbox_centers:
            return None
        target_pos = np.array(self.obj_bbox_centers[target_name], dtype=np.float64)

        # 检查是否有放置目标 (碗) - pick-and-place 模式
        bowl_pos = ctrl.bowl_pos if ctrl else None
        has_bowl = bowl_pos is not None

        # 缓存初始 EE 位置 (第一帧从 robot 获取)
        if not hasattr(self, '_grasp_demo_initial_ee'):
            self._grasp_demo_initial_ee = {}
        if side not in self._grasp_demo_initial_ee:
            robot = self.robot_info.get("robot")
            ee_pos = None
            if robot is not None:
                for link in robot.get_links():
                    if link.get_name() == f"{side}_gripper_link":
                        ee_pos = np.array(link.get_entity_pose().p, dtype=np.float64)
                        break
            if ee_pos is None:
                ee_pos = target_pos + np.array([0.0, 0.0, 0.15])
            self._grasp_demo_initial_ee[side] = ee_pos
            logger.info(f"  [grasp_demo][{side}] 初始 EE={ee_pos.round(3)}, "
                        f"目标物体={target_name}@{target_pos.round(3)}, "
                        f"碗={ctrl.bowl_obj if ctrl else None}@{bowl_pos if has_bowl else None}")

        initial_ee = self._grasp_demo_initial_ee[side]
        n = max(self.num_frames, 1)

        # R1 夹爪手指 origin 在 gripper X 方向偏移 FINGER_FORWARD_OFFSET=0.037m
        # 当 gripper X = [0,0,-1] (top-down 朝下) 时, 手指在 EE 下方 3.7cm
        # 直接把 EE 设到物体中心会让手指在物体下方, 无法夹住物体 (lift=0.23cm 的根因)
        # 修复: EE 位置 = target - R[:,0] * FINGER_FORWARD_OFFSET, 让手指到达物体中心
        gripper_R_fixed = np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ], dtype=np.float64)
        FINGER_FORWARD_OFFSET = 0.037
        ee_offset = -gripper_R_fixed[:, 0] * FINGER_FORWARD_OFFSET  # = [0, 0, +0.037]
        # 让手指到达目标物体中心: grasp 时手指包住物体, approach 在物体上方, lift 在物体上方更高
        grasp_pos = target_pos + ee_offset  # EE 在物体上方 3.7cm, 手指恰在物体中心
        approach_pos = grasp_pos + np.array([0.0, 0.0, 0.08])  # approach 在 grasp 上方 8cm
        # LIFT 高度 12cm (>10cm 满足 grasp_demo 标准), 每帧 3mm (40帧), 物体有时间响应摩擦力
        lift_pos = grasp_pos + np.array([0.0, 0.0, 0.12])  # lift 在 grasp 上方 12cm

        # 阶段分配
        if has_bowl:
            # Pick-and-Place: 7 阶段 (用户: "夹住粉色的东西, 放到碗里面")
            # 关键: LIFT 缩短到 15% 帧 (12cm/16帧=7.5mm/帧, 仍能维持接触)
            # TRANSPORT 20% 帧, 水平移动到碗上方
            # RELEASE 10% 帧, 夹爪打开 (物体掉入碗)
            # RETREAT 15% 帧, 后退
            approach_end = max(1, int(n * 0.10))     # F0-F10%: 接近 (缩短, 留时间给放置)
            descend_end = max(approach_end + 1, int(n * 0.25))  # F10%-F25%: 下降
            close_end = max(descend_end + 1, int(n * 0.40))     # F25%-F40%: 闭合
            lift_end = max(close_end + 1, int(n * 0.55))        # F40%-F55%: 提升
            transport_end = max(lift_end + 1, int(n * 0.75))    # F55%-F75%: 运输
            release_end = max(transport_end + 1, int(n * 0.85)) # F75%-F85%: 释放
            # RETREAT: F85%-F100%
            # 碗上方释放位置: 在碗中心上方 15cm + ee_offset (EE 在物体上方, 手指恰在释放点)
            bowl_release_pos = bowl_pos + np.array([0.0, 0.0, 0.15]) + ee_offset
            # 后退位置: 碗上方 30cm (远离物体, 避免碰撞)
            retreat_pos = bowl_release_pos + np.array([0.0, 0.0, 0.15])
        else:
            # 原始 4 阶段 (无碗): APPROACH → DESCEND → CLOSE → LIFT
            approach_end = max(1, int(n * 0.25))   # F0-F25%: 接近
            descend_end = max(approach_end + 1, int(n * 0.50))  # F25%-F50%: 下降, gripper 全开
            close_end = max(descend_end + 1, int(n * 0.65))     # F50%-F65%: 闭合 (K=20000, 5τ=9帧, 15% 够)

        def smoothstep(t):
            t = max(0.0, min(1.0, t))
            return t * t * (3 - 2 * t)

        if local_idx < approach_end:
            # APPROACH: EE 从初始位置 → approach_pos, gripper 全开
            t = smoothstep(local_idx / approach_end)
            gripper_pos = initial_ee * (1 - t) + approach_pos * t
            gripper_val = GRIPPER_MAX_OPEN
            phase = "APPROACH"
        elif local_idx < descend_end:
            # DESCEND: EE 从 approach_pos → grasp_pos, gripper 全开 (对齐 grasp_demo: 先到位)
            t = smoothstep((local_idx - approach_end) / max(descend_end - approach_end, 1))
            gripper_pos = approach_pos * (1 - t) + grasp_pos * t
            gripper_val = GRIPPER_MAX_OPEN
            phase = "DESCEND"
        elif local_idx < close_end:
            # CLOSE: EE 保持 grasp_pos, gripper 立即设为 0 (grip_cmd=0), 让 PD 充分收敛.
            # 旧版用 (1-t) 线性渐变, PD 始终滞后 cmd, LIFT 开始时 q 仍有 0.006, pad 间隙 9mm 没夹住物体.
            # 直接 cmd=0, PD 用 ~6 帧 (5τ, τ=0.04s) 收敛到 q=0, 剩余 28 帧稳定夹持.
            gripper_pos = grasp_pos
            gripper_val = 0.0
            phase = "CLOSE"
        elif has_bowl and local_idx < lift_end:
            # LIFT (pick-and-place): EE 从 grasp_pos → lift_pos, gripper 闭合
            # 12cm/16帧=7.5mm/帧 (vs 原始 3mm/帧), 仍能维持接触摩擦
            t = smoothstep((local_idx - close_end) / max(lift_end - close_end, 1))
            gripper_pos = grasp_pos * (1 - t) + lift_pos * t
            gripper_val = 0.0
            phase = "LIFT"
        elif has_bowl and local_idx < transport_end:
            # TRANSPORT (用户: "放到碗里面"): 水平移动到碗上方, gripper 闭合
            # 关键: 与 LIFT 一样不能 teleport (会破坏接触), 用 velocity 控制
            t = smoothstep((local_idx - lift_end) / max(transport_end - lift_end, 1))
            gripper_pos = lift_pos * (1 - t) + bowl_release_pos * t
            gripper_val = 0.0
            phase = "TRANSPORT"
        elif has_bowl and local_idx < release_end:
            # RELEASE: 在碗上方打开夹爪, 物体掉入碗
            gripper_pos = bowl_release_pos
            gripper_val = GRIPPER_MAX_OPEN
            phase = "RELEASE"
        elif has_bowl:
            # RETREAT: 后退到碗上方 30cm, gripper 全开
            t = smoothstep((local_idx - release_end) / max(n - release_end, 1))
            gripper_pos = bowl_release_pos * (1 - t) + retreat_pos * t
            gripper_val = GRIPPER_MAX_OPEN
            phase = "RETREAT"
        else:
            # LIFT (原始 4 阶段, 无碗): EE 从 grasp_pos → lift_pos, gripper 保持闭合
            t = smoothstep((local_idx - close_end) / max(n - close_end, 1))
            gripper_pos = grasp_pos * (1 - t) + lift_pos * t
            gripper_val = 0.0
            phase = "LIFT"

        # 固定 top-down 朝向: gripper X(前进) 朝下, Y(手指方向) 沿世界 X
        # 复用上方 gripper_R_fixed (避免重复定义)
        gripper_R = gripper_R_fixed

        return gripper_pos, gripper_R, gripper_val, phase

    def _step_full_robot(self, joints_sapien):
        """full_robot: Retargeting + IK → qpos

        单侧: joints_sapien 为 (N,3) array, 返回 (ik_joints, gripper_val)
        双手: joints_sapien 为 dict {"left": array, "right": array},
              返回 ({"left":..., "right":...}, {"left":..., "right":...})
        """
        is_bimanual = isinstance(joints_sapien, dict)
        sides = self.sides if is_bimanual else [self.side]
        js_dict = joints_sapien if is_bimanual else {self.side: joints_sapien}

        arm_targets = {}
        gripper_vals = {}

        local_idx = getattr(self, '_current_local_idx', 0)
        for s in sides:
            # === hybrid 模式: 帧级偏移 (按 plan: F0=0, F50=grasp_offset, F95=0, F112=0) ===
            if self.grasp_mode == "hybrid" and self.grasp_controllers is not None:
                # 传递 opt_params (654维) 以使用 _fixed_offsets_654 帧级偏移路径
                _opt_p = getattr(self, '_opt_params', None)
                neutral_target = self._compute_mano_neutral_target(local_idx, s, _opt_p)
                if neutral_target is not None:
                    gripper_pos_fk, R_ee_world_fk, gripper_val, phase, _ = neutral_target
                    gripper_vals[s] = float(gripper_val)
                    self._last_phase = phase
                    self._current_gripper_pos = gripper_pos_fk.copy()
                    self._current_gripper_R = R_ee_world_fk.copy() if R_ee_world_fk is not None else np.eye(3)
                    if local_idx == 0 or (local_idx > 0 and local_idx % 30 == 0):
                        logger.info(f"  [neutral][{s}] F{local_idx}: phase={phase}, "
                                    f"pos={gripper_pos_fk.round(3)}, grip={gripper_val:.4f}")
                else:
                    # 退回 MANO (无目标物体时)
                    js = js_dict[s]
                    ref_value = js[self._ref_indices[s], :].astype(np.float32)
                    retarget_qpos = self.retargeting[s].retarget(ref_value, self._fixed_qpos[s])
                    gripper_val = 0.0
                    for i, name in enumerate(self.retargeting[s].joint_names):
                        if "gripper_finger_joint1" in name and i < len(retarget_qpos):
                            gripper_val = float(retarget_qpos[i])
                    gripper_val = float(np.clip(gripper_val, 0.0, GRIPPER_MAX_OPEN))
                    gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retarget_qpos, s)
                    gripper_vals[s] = gripper_val
                    self._last_phase = "APPROACH"
                    self._current_gripper_pos = gripper_pos_fk.copy()
                    self._current_gripper_R = R_ee_world_fk.copy() if R_ee_world_fk is not None else np.eye(3)
                # 跳到 IK 求解 (绕过下面的 adaptive/mano 逻辑)
                # 臂基座坐标系 — 关键: base_quat 会旋转 ARM_BASE_OFFSET, 必须用旋转后的 offset
                # (旧版用未旋转 offset, 导致 base_link_p 错位 13cm, IK target 在错误坐标系)
                base_link_q = self._base_quat
                base_link_R = pr.matrix_from_quaternion(base_link_q)
                arm_base_offset_raw = ARM_BASE_OFFSET_LEFT if s == "left" else ARM_BASE_OFFSET_RIGHT
                arm_base_offset_rotated = base_link_R @ arm_base_offset_raw
                base_link_p = self._base_pos + arm_base_offset_rotated
                base_link_R_inv = base_link_R.T
                ik_target_b = base_link_R_inv @ (gripper_pos_fk - base_link_p)
                R_ee_for_ik = R_ee_world_fk if R_ee_world_fk is not None else np.eye(3)
                ee_R_base = base_link_R_inv @ R_ee_for_ik
                ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
                if local_idx == 0:
                    logger.info(f"  [IK debug][{s}] grasp_demo target: gripper_pos_fk={gripper_pos_fk.round(3)}, "
                                f"base_link_p={base_link_p.round(3)}, "
                                f"ik_target_b={ik_target_b.round(3)}, |ik_target_b|={np.linalg.norm(ik_target_b):.3f} "
                                f"(reach={ARM_MAX_REACH})")
                solve_fn = (self.ik_solver.solve_position_right if s == "right"
                            else self.ik_solver.solve_position_left)
                ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
                for _ in range(IK_SOLVE_PER_FRAME - 1):
                    ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
                ik_joints = self.joint_filters[s].next(ik_joints)
                if s == "right" and len(ik_joints) == 6:
                    ik_joints = np.clip(ik_joints, RIGHT_ARM_JOINT_LIMITS[:, 0], RIGHT_ARM_JOINT_LIMITS[:, 1])
                arm_targets[s] = ik_joints
                continue
            # === 非 hybrid: MANO retargeting (原有逻辑) ===
            js = js_dict[s]
            ref_value = js[self._ref_indices[s], :].astype(np.float32)
            retarget_qpos = self.retargeting[s].retarget(ref_value, self._fixed_qpos[s])

            gripper_val = 0.0
            for i, name in enumerate(self.retargeting[s].joint_names):
                if "gripper_finger_joint1" in name and i < len(retarget_qpos):
                    gripper_val = float(retarget_qpos[i])
            gripper_val = float(np.clip(gripper_val, 0.0, GRIPPER_MAX_OPEN))

            gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retarget_qpos, s)
            self._current_gripper_pos = gripper_pos_fk.copy()
            self._current_gripper_R = R_ee_world_fk.copy() if R_ee_world_fk is not None else np.eye(3)

            if self.grasp_mode == "adaptive" and self.grasp_controllers is not None:
                adapted_target, phase, grasp_info = self.grasp_controllers[s].update(
                    gripper_pos_fk, gripper_val
                )
                gripper_val = adapted_target
                # 首帧/相位变化时打印调试
                if not hasattr(self, '_grasp_debug_printed'):
                    self._grasp_debug_printed = True
                    logger.info(f"  [grasp][{s}] 首帧: phase={phase}, "
                                f"curl={grasp_info['mano_curl']:.2f}, "
                                f"obj={grasp_info['nearest_obj']}@{grasp_info['obj_dist']:.3f}m")
            elif self.grasp_mode == "mano":
                # mano 模式: 纯重放 + 接触维持夹紧 (与 _step_gripper_only 一致, 用户: "夹爪和整个机器人的任务是一样的")
                # 接触前: 跟随 MANO (纯重放, gripper_val 来自 retargeting)
                # 接触后: 维持固定夹紧 (gripper_val0 - CLAMP_OFFSET*curl), 防止物体滑落
                # MANO 张开 (curl < RELEASE_TRIGGER_CURL) 时释放, 回到纯重放
                if not hasattr(self, '_mano_state_fr'):
                    self._mano_state_fr = {}
                if s not in self._mano_state_fr:
                    self._mano_state_fr[s] = {'contact': False, 'gval0': None, 'obj': None, 'logged': False}
                st = self._mano_state_fr[s]
                try:
                    f1, f2, c_objs = get_finger_contacts(self.robot_info["robot"], s, self.scene, self.obj_actors)
                except Exception:
                    f1, f2, c_objs = False, False, []
                mano_curl = 1.0 - float(gripper_val) / GRIPPER_MAX_OPEN
                if not st['contact']:
                    if f1 and f2 and c_objs and mano_curl > GRASP_TRIGGER_CURL:
                        st['contact'] = True
                        st['gval0'] = float(gripper_val)
                        st['obj'] = c_objs[0]
                        if not st['logged']:
                            st['logged'] = True
                            logger.info(f"  [mano-fr][{s}] 接触维持: obj={st['obj']}, "
                                        f"curl={mano_curl:.2f}, gval0={st['gval0']:.4f}")
                else:
                    if mano_curl < RELEASE_TRIGGER_CURL:
                        st['contact'] = False
                        st['gval0'] = None
                        st['obj'] = None
                    else:
                        clamp = CLAMP_OFFSET_MAX * max(mano_curl, CLAMP_CURL_FLOOR)
                        clamped = st['gval0'] - clamp
                        gripper_val = float(min(clamped, gripper_val))
            gripper_vals[s] = gripper_val

            # 臂基座坐标系 — 关键: base_quat 会旋转 ARM_BASE_OFFSET, 必须用旋转后的 offset
            # (旧版用未旋转 offset, 导致 base_link_p 错位 13cm, IK target 在错误坐标系)
            base_link_q = self._base_quat
            base_link_R = pr.matrix_from_quaternion(base_link_q)
            arm_base_offset_raw = ARM_BASE_OFFSET_LEFT if s == "left" else ARM_BASE_OFFSET_RIGHT
            arm_base_offset_rotated = base_link_R @ arm_base_offset_raw
            base_link_p = self._base_pos + arm_base_offset_rotated
            base_link_R_inv = base_link_R.T

            ik_target_b = base_link_R_inv @ (gripper_pos_fk - base_link_p)
            ee_R_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

            # Debug: 首帧打印 IK 输入
            if not hasattr(self, '_ik_debug_printed'):
                self._ik_debug_printed = True
                logger.info(f"  [IK debug][{s}] gripper_pos_fk={gripper_pos_fk.round(3)}")
                # 打印实际 EE 位置 (从 robot FK, 对比 IK 目标)
                try:
                    actual_ee = self._get_gripper_pose_from_retargeting(
                        self.robot_info["robot"].get_qpos(), s)
                    logger.info(f"  [IK debug][{s}] actual_ee_pos={np.array(actual_ee[0]).round(3)}")
                except Exception as e:
                    logger.info(f"  [IK debug][{s}] actual_ee 获取失败: {e}")
                logger.info(f"  [IK debug][{s}] base_link_p={base_link_p.round(3)} (computed from ROOT+offset)")
                logger.info(f"  [IK debug][{s}] ik_target_b (arm base frame)={ik_target_b.round(3)}")
                logger.info(f"  [IK debug][{s}] |ik_target_b|={np.linalg.norm(ik_target_b):.3f} (arm reach={ARM_MAX_REACH})")

            # IK 求解 (分别调用, 避免 solve_position_both 的双重转换 bug)
            solve_fn = (self.ik_solver.solve_position_right if s == "right"
                        else self.ik_solver.solve_position_left)
            ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
            for _ in range(IK_SOLVE_PER_FRAME - 1):
                ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))

            # per-side 滤波
            ik_joints = self.joint_filters[s].next(ik_joints)

            if s == "right" and len(ik_joints) == 6:
                ik_joints = np.clip(ik_joints, RIGHT_ARM_JOINT_LIMITS[:, 0], RIGHT_ARM_JOINT_LIMITS[:, 1])

            arm_targets[s] = ik_joints

        if is_bimanual:
            return arm_targets, gripper_vals
        else:
            return arm_targets[self.side], gripper_vals[self.side]

    def _step_gripper_only(self, joints_sapien):
        """gripper_only: 解析映射 → 夹爪位姿 + 关节角

        虚拟 6-DOF 关节架构 (对齐 05_gripper_test.py):
          - 虚拟关节 (vx/vy/vz/rz/ry/rx) 通过 PD 驱动移动夹爪, 产生真实动量和接触力
          - set_qpos + set_drive_target 设置虚拟关节目标位置 (加速收敛)
          - 手指关节: set_qpos + set_drive_target (保持原有行为)
          - 不再使用 set_root_pose / set_root_linear_velocity / lock_root_pose

        hybrid 模式 (MANO+offset 中和态): EE = MANO 轨迹 + 常量偏移, 保持 MANO 运动形状
          (用户: "mano参数为主体, 你可以平移轨迹, 但不能离开轨迹, 偏移轨迹那么多")
        """
        local_idx = getattr(self, '_current_local_idx', 0)
        virtual_idx = self.robot_info.get("virtual_idx", {})
        is_virtual = bool(virtual_idx)  # gripper_only + 虚拟关节模式

        # === hybrid 模式: MANO+offset 中和态 (第十八轮: CEM 优化) ===
        if self.grasp_mode == "hybrid" and self.grasp_controllers is not None:
            opt_params = getattr(self, '_opt_params', None)
            neutral_target = self._compute_mano_neutral_target(local_idx, self.side, opt_params)
            if neutral_target is not None:
                gripper_pos, gripper_R, gripper_val, phase, transport_vel_limit = neutral_target
                self._last_phase = phase  # 调试用: 主循环接触日志按阶段输出
                # 每帧重置手指 set_qpos 锁, 允许下一次 CLOSE 阶段执行 set_qpos 闭合
                if phase == "APPROACH":
                    self._close_finger_set = False
                robot = self.robot_info["robot"]
                gripper_idx1 = self.robot_info["gripper_idx1"]
                gripper_idx2 = self.robot_info["gripper_idx2"]
                # 虚拟关节: 不再需要 lock_root_pose, 虚拟关节 PD 驱动自动保持位置
                self._close_lock_pose = None
                joint1 = float(gripper_val)
                joint2 = float(gripper_val)
                # v15j: 不再强制 val=0.0, 使用 gripper_val (PRE_GRASP→CONTACT 渐变)
                #   旧版 CLOSE 全闭 → 物体被挤出; 新版留空间让物体保持在指间
                # v15n: 不再预闭合, APPROACH 全程大张
                if is_virtual:
                    # 虚拟关节控制策略:
                    # APPROACH/CLOSE: 每帧 set_qpos + set_drive_target
                    #   CLOSE阶段PD漂移3cm, 手指-物体反作用力超出PD维持能力
                    #   用set_qpos每帧固定位置, 避免漂移丢失接触
                    rz, ry, rx = rotmat_to_zyx_euler(gripper_R)
                    active_joints = robot.get_active_joints()
                    qpos = robot.get_qpos().copy()
                    qpos[virtual_idx['vx']] = float(gripper_pos[0])
                    qpos[virtual_idx['vy']] = float(gripper_pos[1])
                    qpos[virtual_idx['vz']] = float(gripper_pos[2])
                    qpos[virtual_idx['rz']] = float(rz)
                    qpos[virtual_idx['ry']] = float(ry)
                    qpos[virtual_idx['rx']] = float(rx)
                    # 边走边夹: APPROACH 且仍是中张时 set_qpos 防漂移;
                    # 开始闭合后交给 PD, 避免 set_qpos 推飞物体.
                    if phase == "APPROACH" and abs(float(joint1) - _DESCEND_OPEN) < 1e-6:
                        qpos[gripper_idx1] = float(joint1)
                        qpos[gripper_idx2] = float(joint2)
                    robot.set_qpos(qpos)
                    # 设置 PD 目标 (维持位置, 提供平滑过渡)
                    for vkey, vval in [('vx', gripper_pos[0]), ('vy', gripper_pos[1]),
                                       ('vz', gripper_pos[2]), ('rz', rz), ('ry', ry), ('rx', rx)]:
                        active_joints[virtual_idx[vkey]].set_drive_target(float(vval))
                # 手指: set_drive_target (PD自然闭合, 让物理引擎计算正确接触力)
                active_joints = robot.get_active_joints()
                active_joints[gripper_idx1].set_drive_target(float(joint1))
                active_joints[gripper_idx2].set_drive_target(float(joint2))
                # 关键修复: 不再用 set_qpos 传送手指到 0.0 (会导致物理爆炸推飞物体)
                # PD (stiffness=8000) 会自然闭合手指, 接触物体时停止
                # 第十九轮 v3: 保存当前 gripper 位姿供 rollout 计算接近奖励
                self._current_gripper_pos = gripper_pos.copy()
                self._current_gripper_R = gripper_R.copy()  # 供 physics_step 后纠正漂移
                if local_idx == 0 or local_idx % 10 == 0:
                    # 调试: 虚拟关节跟踪精度 + 手指状态
                    _actual_qpos = robot.get_qpos()
                    _vz_err = abs(_actual_qpos[virtual_idx['vz']] - gripper_pos[2]) if is_virtual else 0
                    _ee_pos = None
                    for link in robot.get_links():
                        if link.get_name() == f"{self.side}_gripper_link":
                            _ee_pos = np.array(link.get_entity_pose().p)
                            break
                    _ee_err = np.linalg.norm(gripper_pos - _ee_pos) if _ee_pos is not None else -1
                    _f1 = _actual_qpos[gripper_idx1] if gripper_idx1 is not None else -1
                    _f2 = _actual_qpos[gripper_idx2] if gripper_idx2 is not None else -1
                    logger.info(f"  [neutral][{self.side}] F{local_idx}: phase={phase}, "
                                f"pos={gripper_pos.round(3)}, grip_cmd={gripper_val:.4f}, "
                                f"vz_err={_vz_err:.4f}m, ee_err={_ee_err:.4f}m, "
                                f"f1={_f1:.4f}, f2={_f2:.4f}")
                return (), (joint1, joint2)

        # === 非 hybrid: MANO 解析映射 (原有逻辑) ===
        wrist_pos = joints_sapien[0, :3]
        finger1_pos = joints_sapien[4, :3]
        finger2_pos = joints_sapien[8, :3]

        # prefix: 左手用 "left", 右手/双手用 "right" (对齐 04 的 y_sign 逻辑)
        prefix = self.side if self.side in ("left", "right") else "right"
        root_pos, root_R, joint1, joint2 = compute_analytical_gripper_pose(
            wrist_pos, finger1_pos, finger2_pos, prefix=prefix
        )

        # 轨迹跟踪: 记录 MANO 期望根位置 (限幅前 = 真实 MANO 轨迹, 用于误差验证)
        expected_root = root_pos.copy()
        # 根速度限制: kinematic 根瞬移会让动态物体跟不上 (惯性 + 摩擦限制)
        # 限制每帧根位置变化 ≤ MAX_ROOT_STEP, 防止提升中物体滑落 (glb_5 follow=8→10+)
        prev = getattr(self, "_prev_root_pos", None)
        if prev is not None:
            delta = root_pos - prev
            step = float(np.linalg.norm(delta))
            if step > MAX_ROOT_STEP:
                root_pos = prev + delta * (MAX_ROOT_STEP / step)
        self._prev_root_pos = root_pos.copy()

        robot = self.robot_info["robot"]
        qpos = robot.get_qpos().copy()

        if is_virtual:
            # 虚拟关节: 设置 PD 目标 (位置 + ZYX Euler), 不再使用 set_root_pose
            rz, ry, rx = rotmat_to_zyx_euler(root_R)
            qpos[virtual_idx['vx']] = float(root_pos[0])
            qpos[virtual_idx['vy']] = float(root_pos[1])
            qpos[virtual_idx['vz']] = float(root_pos[2])
            qpos[virtual_idx['rz']] = float(rz)
            qpos[virtual_idx['ry']] = float(ry)
            qpos[virtual_idx['rx']] = float(rx)
            # set_drive_target 同时更新
            active_joints = robot.get_active_joints()
            for vkey, vval in [('vx', root_pos[0]), ('vy', root_pos[1]),
                               ('vz', root_pos[2]), ('rz', rz), ('ry', ry), ('rx', rx)]:
                active_joints[virtual_idx[vkey]].set_drive_target(float(vval))

        # 立即设置手指 qpos (关键: 04 的做法, 让手指瞬间到位产生接触)
        gripper_idx1 = self.robot_info["gripper_idx1"]
        gripper_idx2 = self.robot_info["gripper_idx2"]
        qpos[gripper_idx1] = float(joint1)
        qpos[gripper_idx2] = float(joint2)

        # 轨迹跟踪: 记录 MANO 期望手指 (抓取调整前 = 真实 MANO 解析, 用于误差验证)
        expected_j1 = float(joint1)
        expected_j2 = float(joint2)
        # 自适应抓取: 用 MANO 意图 + 相位状态机决定夹爪开合
        # joint1/joint2: 0=闭合, GRIPPER_MAX_OPEN=张开 (与 full_robot 的 gripper_val 同约定)
        if self.grasp_mode == "hybrid" and self.grasp_controllers is not None:
            s = self.side
            controller = self.grasp_controllers.get(s)
            if controller is not None:
                mano_gripper_val = float(joint1)  # MANO 解析映射的夹爪开合
                current_qpos = qpos[gripper_idx1]  # 当前手指 qpos
                adapted_target, phase, grasp_info = controller.update(
                    root_pos, root_R, mano_gripper_val,
                    robot=robot, scene=self.scene, current_qpos=current_qpos
                )
                joint1 = adapted_target
                joint2 = adapted_target
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                if not hasattr(self, '_grasp_debug_printed'):
                    self._grasp_debug_printed = True
                    logger.info(f"  [hybrid][{s}] 首帧: phase={phase}, "
                                f"curl={grasp_info['mano_curl']:.2f}, "
                                f"obj={grasp_info['nearest_obj']}@{grasp_info['obj_dist']:.3f}m, "
                                f"force={grasp_info['grasp_force']:.1f}N")
        elif self.grasp_mode == "adaptive" and self.grasp_controllers is not None:
            s = self.side
            controller = self.grasp_controllers.get(s)
            if controller is not None:
                mano_gripper_val = float(joint1)  # MANO 解析映射的夹爪开合
                adapted_target, phase, grasp_info = controller.update(root_pos, mano_gripper_val)
                joint1 = adapted_target
                joint2 = adapted_target
                qpos[gripper_idx1] = float(joint1)
                qpos[gripper_idx2] = float(joint2)
                if not hasattr(self, '_grasp_debug_printed'):
                    self._grasp_debug_printed = True
                    logger.info(f"  [grasp][{s}] 首帧: phase={phase}, "
                                f"curl={grasp_info['mano_curl']:.2f}, "
                                f"obj={grasp_info['nearest_obj']}@{grasp_info['obj_dist']:.3f}m")
        elif self.grasp_mode == "mano":
            # mano 模式: 纯重放 + 接触维持夹紧 (真正夹住物体, 用户: "实现能够真正的和物体交互")
            # 接触前: 跟随 MANO (纯重放, 当前行为)
            # 接触后: 记录 qpos_at_contact, 维持固定夹紧 (qpos0 - CLAMP_OFFSET*curl), 防止物体滑落
            # MANO 张开 (curl < RELEASE_TRIGGER_CURL) 时释放, 回到纯重放
            # 只维持已接触物体的夹紧, 不主动抓所有物体 (用户: "目前只有一个物体进行交互是对的")
            if not hasattr(self, '_mano_state'):
                self._mano_state = {'contact': False, 'qpos0': None, 'obj': None, 'logged': False}
            st = self._mano_state
            try:
                f1, f2, c_objs = get_finger_contacts(robot, self.side, self.scene, self.obj_actors)
            except Exception:
                f1, f2, c_objs = False, False, []
            mano_curl = 1.0 - float(joint1) / GRIPPER_MAX_OPEN
            current_qpos = qpos[gripper_idx1]
            if not st['contact']:
                # 接触前: 纯重放 (不改 qpos); 检测到双指接触 + MANO 卷曲 → 进入维持
                if f1 and f2 and c_objs and mano_curl > GRASP_TRIGGER_CURL:
                    st['contact'] = True
                    st['qpos0'] = current_qpos
                    st['obj'] = c_objs[0]
                    if not st['logged']:
                        st['logged'] = True
                        logger.info(f"  [mano][{self.side}] 接触维持: obj={st['obj']}, "
                                    f"curl={mano_curl:.2f}, qpos0={current_qpos:.4f}")
            else:
                # 接触后: MANO 张开 → 释放; 否则维持固定夹紧 (只夹紧不松开, 防 MANO 抖动)
                if mano_curl < RELEASE_TRIGGER_CURL:
                    st['contact'] = False
                    st['qpos0'] = None
                    st['obj'] = None
                else:
                    clamp = CLAMP_OFFSET_MAX * max(mano_curl, CLAMP_CURL_FLOOR)
                    clamped = st['qpos0'] - clamp
                    # min: 取更闭合的值 (MANO 若更闭合则跟随 MANO, 否则维持 clamped)
                    qpos[gripper_idx1] = float(min(clamped, qpos[gripper_idx1]))
                    qpos[gripper_idx2] = float(min(clamped, qpos[gripper_idx2]))
                    joint1 = qpos[gripper_idx1]
                    joint2 = qpos[gripper_idx2]

        # 轨迹跟踪误差: 物理输出(actual) vs 真实 MANO 期望(expected)
        # 用户: "夹爪运动要物理和真实输出的误差来判断准不准确"
        actual_j1 = float(qpos[gripper_idx1])
        actual_j2 = float(qpos[gripper_idx2])
        self._last_track = {
            'root_err_mm': float(np.linalg.norm(expected_root - root_pos) * 1000.0),  # 根位置误差 mm
            'j1_err_mm': float(abs(expected_j1 - actual_j1) * 1000.0),  # 手指1误差 mm
            'j2_err_mm': float(abs(expected_j2 - actual_j2) * 1000.0),  # 手指2误差 mm
        }
        robot.set_qpos(qpos)
        return None, (joint1, joint2)

    def _init_mano_markers(self):
        """初始化 MANO 3 参考点渲染节点 (wrist + finger1 + finger2)
        对齐 04_physics_simulation.py _render_keypoints L2029 的方式, 但只渲染 3 个点
        颜色: wrist=红, finger1=绿, finger2=蓝 (便于区分)
        """
        if not getattr(self.scene, "_render_available", True):
            return None
        try:
            import sapien.render
            self._mano_context = sapien.render.SapienRenderer()._internal_context
            self._mano_internal_scene = self.scene.render_system._internal_scene
            self._mano_marker_nodes = []
            # 创建 3 个不同颜色球体 (wrist=红, finger1=绿, finger2=蓝)
            colors = [
                np.array([1.0, 0.0, 0.0, 1.0]),  # wrist=红
                np.array([0.0, 1.0, 0.0, 1.0]),  # finger1=绿
                np.array([0.0, 0.5, 1.0, 1.0]),  # finger2=蓝
            ]
            for color in colors:
                mat = self._mano_context.create_material(np.zeros(4), color, 0.0, 0.5, 0)
                sphere = self._mano_context.create_uvsphere_mesh(12, 6)
                model = self._mano_context.create_model([sphere], [mat])
                node = self._mano_internal_scene.add_node()
                node.set_position([0, 0, 0])
                node.set_scale([0.015, 0.015, 0.015])  # 半径 1.5cm 球体
                obj = self._mano_internal_scene.add_object(model, node)
                obj.shading_mode = 0
                obj.cast_shadow = False
                obj.transparency = 0
                self._mano_marker_nodes.append(node)
            logger.info(f"  MANO 参考点渲染已初始化 (3 个球体: wrist=红/finger1=绿/finger2=蓝)")
            return self._mano_marker_nodes
        except Exception as e:
            logger.warning(f"  MANO 参考点渲染初始化失败: {e}")
            self._mano_marker_nodes = None
            return None

    def _update_mano_markers(self, wrist_pos, finger1_pos, finger2_pos):
        """每帧更新 MANO 3 参考点位置
        Args:
            wrist_pos: (3,) MANO 手腕在 SAPIEN 坐标系下的位置
            finger1_pos: (3,) MANO 指尖1 (joints[4]) 位置
            finger2_pos: (3,) MANO 指尖2 (joints[8]) 位置
        """
        nodes = getattr(self, "_mano_marker_nodes", None)
        if nodes is None or len(nodes) != 3:
            return
        try:
            positions = [wrist_pos, finger1_pos, finger2_pos]
            for node, pos in zip(nodes, positions):
                node.set_position(np.asarray(pos).tolist())
        except Exception:
            pass

    def _init_collision_visualization(self, robot):
        """初始化手指碰撞模型可视化 (第十三轮: 用户要求"把碰撞模型也展示到视频里面")

        创建半透明红色 RGBA[1,0,0,0.4] visual actor 覆盖在手指视觉模型外,
        每帧跟随手指 link 位姿, 视频中清晰看到碰撞体 vs 物体接触.
        不影响物理 (仅 visual, 不参与碰撞).
        """
        self._collision_visual_actors = {}  # link_name -> visual actor
        self._collision_link_map = {}  # link_name -> robot link (用于取位姿)
        if not getattr(self.scene, "_render_available", True):
            return
        try:
            import sapien.render
            sides = ["left", "right"] if self.side == "both" else [self.side]
            for s in sides:
                for link_name_suffix in ["finger_link1", "finger_link2"]:
                    full_link_name = f"{s}_gripper_{link_name_suffix}"
                    mesh_file = str(R1_MESH_DIR / f"{s}_gripper_{link_name_suffix}_collision.STL")
                    builder = self.scene.create_actor_builder()
                    mat = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 0.4])
                    builder.add_visual_from_file(mesh_file, material=mat)
                    actor = builder.build_kinematic(name=f"_collision_vis_{full_link_name}")
                    self._collision_visual_actors[full_link_name] = actor
            # 建立手指 link 映射 (用于每帧取位姿)
            for link in robot.get_links():
                lname = link.get_name()
                if lname in self._collision_visual_actors:
                    self._collision_link_map[lname] = link
            logger.info(f"  碰撞可视化已创建 ({len(self._collision_visual_actors)} 个半透明红色 actor, "
                        f"RGBA=[1,0,0,0.4], 跟随手指 link 位姿)")
        except Exception as e:
            logger.warning(f"  碰撞可视化初始化失败: {e}")
            self._collision_visual_actors = {}
            self._collision_link_map = {}

    def _update_collision_visualization(self):
        """每帧更新碰撞可视化位姿 (跟随手指 link)"""
        if not hasattr(self, '_collision_visual_actors') or not self._collision_visual_actors:
            return
        try:
            for link_name, vis_actor in self._collision_visual_actors.items():
                link = self._collision_link_map.get(link_name)
                if link is not None:
                    vis_actor.set_pose(link.get_pose())
        except Exception:
            pass

    def run(self, run_main_loop=True):
        """主仿真循环

        Args:
            run_main_loop: 为 False 时只执行初始化 (场景/机器人/控制器/MANO轨迹),
                           不进入 warmup 和主循环. 供 v4-pipeline 在 Stage 1/2/3 之前使用.
        """
        self._close_entered = False  # 重置 CLOSE 阶段标志
        # 1. 对齐
        self._align_scene()

        # 2. 加载 HaWoR 数据
        logger.info("=" * 60)
        logger.info("Step 2: 加载 HaWoR 数据 + MANO FK")
        logger.info("=" * 60)
        # 双手模式加载两侧 HaWoR 数据; 单侧只加载一个
        if self.side == "both":
            hawor_data = {}
            mano_layer = {}
            betas_mean = {}
            for s in self.sides:  # ["left", "right"]
                hi = self.hand_indices[s]
                hawor_data[s] = load_hawor_data(self.hawor_dir, hand_idx=hi)
                betas_mean[s] = hawor_data[s]["pred_betas"][self.start_frame].astype(np.float32)
                mano_side = "left" if hi == 0 else "right"
                from mano_layer import MANOLayer
                mano_layer[s] = MANOLayer(mano_side, betas_mean[s])
            n_total = len(hawor_data[self.sides[0]]["pred_trans"])
        else:
            hawor_data = load_hawor_data(self.hawor_dir, hand_idx=self.hand_idx)
            n_total = len(hawor_data["pred_trans"])
            betas_mean = hawor_data["pred_betas"][self.start_frame].astype(np.float32)
            mano_side = "left" if self.hand_idx == 0 else "right"
            from mano_layer import MANOLayer
            mano_layer = MANOLayer(mano_side, betas_mean)
        if self.num_frames < 0 or self.num_frames > n_total - self.start_frame:
            self.num_frames = n_total - self.start_frame
        logger.info(f"  总帧数: {n_total}, 渲染: {self.start_frame}~{self.start_frame + self.num_frames - 1}")

        # 检查是否已从优化器复制了 MANO 轨迹
        _has_opt_traj = getattr(self, '_opt_mano_gripper_traj', None) is not None
        if _has_opt_traj:
            _opt_n = len(self._opt_mano_gripper_traj[self.side]["pos"])
            logger.info(f"  优化器的 MANO 轨迹: {_opt_n} 帧 (原始: {self.num_frames} 帧)")
            # 不覆盖 num_frames — 优化后帧数不应改变. 若优化轨迹更短, 主循环自动取 min.
            if _opt_n < self.num_frames:
                logger.warning(f"  优化轨迹 {_opt_n} 帧 < 原始 {self.num_frames} 帧, "
                              f"将截断到 {_opt_n} 帧")

        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)

        # 3. 预扫描 GLB 最低点 → 创建场景
        logger.info("=" * 60)
        logger.info("Step 3: 创建 SAPIEN 场景 + 加载 GLB")
        logger.info("=" * 60)
        glb_path = self.ras_dir / "final_scene.glb"
        ground_z = 0.0
        if glb_path.exists() and self.transform_params_path:
            ground_z = compute_glb_ground_z(glb_path, self.transform_params_path)
            logger.info(f"  GLB 预扫描地面高度: z={ground_z:.4f}")
        else:
            logger.warning(f"  GLB 不存在: {glb_path}")
        self._ground_z = ground_z

        # 地面高度: GLB 最低点 (支撑动态物体)
        # full_robot 的 ROOT 可能在地下, 通过禁用躯干碰撞避免干扰 (在 setup_robot 中处理)
        # v15i: views=none 时 force_cpu, 避免 sandbox 渲染初始化 core dump
        _force_cpu = (self.views == "none" or not self.views)
        self.scene = setup_physics_scene(ground_height=ground_z, force_cpu=_force_cpu)
        render_available = getattr(self.scene, "_render_available", False)
        logger.info(f"  SAPIEN 场景创建: render_available={render_available}")

        self.obj_bbox_centers = {}
        self.obj_info = {}
        if glb_path.exists() and self.transform_params_path:
            logger.info(f"  开始加载 GLB: {glb_path}")
            self.obj_actors, _, self.obj_bbox_centers, self.obj_info = load_glb_with_physics(
                glb_path, self.transform_params_path, self.scene, fast_collision=True  # 单凸包 (CoACD 太慢, 可按需切换 False)
            )
            logger.info(f"  GLB 加载完成: {len(self.obj_actors)} 个物体")
        else:
            self.obj_actors = []

        # 初始化抓取控制器 (延迟到 setup_robot 之后, 因为 hybrid 需要 robot 对象)
        self.grasp_controllers = None
        if self.grasp_mode == "hybrid":
            # hybrid 模式在 setup_robot 后初始化 (需要 robot 对象)
            pass
        elif self.grasp_mode == "adaptive":
            self.grasp_controllers = {
                s: AdaptiveGraspController(self.obj_actors, side=s) for s in self.sides
            }
            logger.info(f"  自适应抓取控制器已启用 (grasp_mode=adaptive), 阈值: "
                        f"触发卷曲>{GRASP_TRIGGER_CURL}, 释放卷曲<{RELEASE_TRIGGER_CURL}")
        else:
            logger.info(f"  纯 MANO 重放模式 (grasp_mode=mano), 夹爪直接跟随 MANO 手指")

        # 4. 计算手腕位置 → 最优基座
        logger.info("=" * 60)
        logger.info("Step 4: 计算最优基座位置")
        logger.info("=" * 60)
        # 如果已从优化器复制了 MANO 轨迹, 直接复用, 跳过 FK 计算
        _has_opt_traj = getattr(self, '_opt_mano_gripper_traj', None) is not None
        if _has_opt_traj:
            self._mano_gripper_traj = self._opt_mano_gripper_traj
            _t = self._mano_gripper_traj[self.side]
            logger.info(f"  复用优化器的 MANO 轨迹: {len(_t['pos'])} 帧")
            wrist_positions = _t["pos"]  # root_pos 即是手腕位置
        # 加载 transform_params 用于 MANO 关节点的完整坐标变换
        # 变换链: SLAM → OpenGL (Rx_hand) → GLB (s * R_hand_to_glb @ p + t) → SAPIEN (R_AXIS)
        # 与 render_quick.py / common.py _render_to_sapien 完全一致
        _tp_s, _tp_R_h2g, _tp_t_h2g, _tp_Rx = 1.0, np.eye(3), np.zeros(3), np.diag([1.0, -1.0, -1.0])
        if self.transform_params_path:
            _tp = np.load(str(self.transform_params_path))
            _tp_s = float(_tp.get('scale_ratio', _tp.get('s_inv', 1.0)))
            # s_inv 是 1/scale, 如果没有 scale_ratio 就取倒数
            if 'scale_ratio' not in _tp and 's_inv' in _tp:
                _tp_s = 1.0 / float(_tp['s_inv'])
            _tp_R_h2g = _tp.get('R_hand_to_glb', _tp.get('R_inv', np.eye(3)))
            _tp_t_h2g = _tp.get('t_hand_to_glb', _tp.get('t_inv', np.zeros(3)))
            _tp_Rx = _tp.get('Rx_hand', np.diag([1.0, -1.0, -1.0]))
            _tp_R_hand = _tp_R_h2g @ _tp_Rx
            logger.info(f"  transform_params: s={_tp_s:.4f}, R_h2g shape={_tp_R_h2g.shape}, "
                        f"t_h2g={_tp_t_h2g.round(4)}, Rx_hand={np.diag(_tp_Rx).round(2)}")
        else:
            _tp_R_hand = _tp_R_h2g @ _tp_Rx
            logger.warning("  无 transform_params, MANO 轨迹使用 RXWORLD_TO_SAPIEN 变换 (可能不准)")

        # 存储到实例, 供 run_v4_pipeline / 相机变换使用
        self._mano_xform_s = _tp_s
        self._mano_xform_R_hand = _tp_R_hand
        self._mano_xform_t = _tp_t_h2g
        # 相机变换需要 raw R_h2g (不含 Rx_hand)
        self._cam_xform_R_h2g = _tp_R_h2g
        self._cam_xform_s = _tp_s
        self._cam_xform_t = _tp_t_h2g

        def _mano_to_sapien(pts_slam):
            """MANO SLAM 坐标 → SAPIEN 坐标 (与 002_render_scene.py _render_to_sapien 一致)
            
            002_render_scene.py 使用 _render_to_sapien 带 params:
              p_glb = s * R_h2g @ Rx_hand @ p_slam + t_h2g
              p_sapien = R_AXIS @ p_glb
            R_x = I (001 链), 所以 RXWORLD_TO_SAPIEN = R_AXIS.
            GLB 物体也使用相同的 R_AXIS (data_loader.py 已对齐 002 链), 三者同帧.
            """
            pts_glb = _tp_s * (_tp_R_h2g @ _tp_Rx @ pts_slam.T).T + _tp_t_h2g
            return (R_AXIS @ pts_glb.T).T

        if not _has_opt_traj:
            wrist_positions = []
            # 同时预计算 MANO 解析夹爪轨迹 (用户: "mano参数为主体, 可以平移轨迹")
            # 每帧: root_pos, j1, j2 from compute_analytical_gripper_pose
            mano_gripper_traj = {}  # side -> {"pos": [], "R": [], "j1": [], "j2": []}
            if self.side == "both":
                # 双手: 收集两侧手腕位置 + 夹爪轨迹
                for fi in range(self.start_frame, self.start_frame + self.num_frames):
                    for s in self.sides:
                        hd = hawor_data[s]
                        if fi < len(hd["pred_valid"]) and not hd["pred_valid"][fi]:
                            continue
                        if fi >= len(hd["pred_trans"]):
                            continue
                        try:
                            _, joints = compute_mano_joints(
                                mano_layer[s],
                                hd["pred_rot"][fi],
                                hd["pred_hand_pose"][fi],
                                hd["pred_trans"][fi],
                            )
                            joints_sapien = _mano_to_sapien(joints)
                            wrist_positions.append(joints_sapien[0, :3])
                            # 预计算夹爪位姿 (用于 hybrid 中和态, 第十三轮新增存储 R)
                            if s not in mano_gripper_traj:
                                mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": []}
                            root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
                                joints_sapien[0, :3], joints_sapien[4, :3],
                                joints_sapien[8, :3], prefix=s,
                            )
                            mano_gripper_traj[s]["pos"].append(root_pos)
                            mano_gripper_traj[s]["R"].append(root_R)
                            mano_gripper_traj[s]["j1"].append(j1)
                            mano_gripper_traj[s]["j2"].append(j2)
                        except Exception:
                            continue
            else:
                for fi in range(self.start_frame, self.start_frame + self.num_frames):
                    if fi < len(hawor_data["pred_valid"]) and not hawor_data["pred_valid"][fi]:
                        continue
                    if fi >= len(hawor_data["pred_trans"]):
                        continue
                    try:
                        _, joints = compute_mano_joints(
                            mano_layer,
                            hawor_data["pred_rot"][fi],
                            hawor_data["pred_hand_pose"][fi],
                            hawor_data["pred_trans"][fi],
                        )
                        joints_sapien = _mano_to_sapien(joints)
                        wrist_positions.append(joints_sapien[0, :3])
                        # 预计算夹爪位姿 (用于 hybrid 中和态, 第十三轮新增存储 R)
                        s = self.side
                        if s not in mano_gripper_traj:
                            mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": []}
                        root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
                            joints_sapien[0, :3], joints_sapien[4, :3],
                            joints_sapien[8, :3], prefix=s,
                        )
                        mano_gripper_traj[s]["pos"].append(root_pos)
                        mano_gripper_traj[s]["R"].append(root_R)
                        mano_gripper_traj[s]["j1"].append(j1)
                        mano_gripper_traj[s]["j2"].append(j2)
                    except Exception:
                        continue
            # 转为 numpy 数组, 存储到实例 (供 hybrid 中和态使用)
            for key in mano_gripper_traj:
                mano_gripper_traj[key]["pos"] = np.array(mano_gripper_traj[key]["pos"], dtype=np.float64)
                mano_gripper_traj[key]["R"] = np.array(mano_gripper_traj[key]["R"], dtype=np.float64)  # (N, 3, 3)
                mano_gripper_traj[key]["j1"] = np.array(mano_gripper_traj[key]["j1"], dtype=np.float64)
                mano_gripper_traj[key]["j2"] = np.array(mano_gripper_traj[key]["j2"], dtype=np.float64)
            self._mano_gripper_traj = mano_gripper_traj
            # Debug: 打印轨迹形状 (含 R)
            for key in mano_gripper_traj:
                t = mano_gripper_traj[key]
                logger.info(f"  [debug] mano_gripper_traj[{key}]: pos={t['pos'].shape}, "
                            f"R={t['R'].shape}, j1={t['j1'].shape}")

        # Debug: 手腕 vs 物体位置
        wrist_is_valid = hasattr(wrist_positions, '__len__') and len(wrist_positions) > 0
        if wrist_is_valid:
            warr = np.array(wrist_positions)
            logger.info(f"  [debug] 手腕位置范围: x=[{warr[:,0].min():.3f},{warr[:,0].max():.3f}] "
                        f"y=[{warr[:,1].min():.3f},{warr[:,1].max():.3f}] "
                        f"z=[{warr[:,2].min():.3f},{warr[:,2].max():.3f}]")
            logger.info(f"  [debug] 手腕质心: {warr.mean(axis=0).round(3)}")
        for name, center in self.obj_bbox_centers.items():
            logger.info(f"  [debug] 物体 {name} bbox中心: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")

        if self.mode == "full_robot":
            self._base_pos, self._base_quat = self._compute_optimal_base(wrist_positions, R_c2w_all)
        else:
            # gripper_only: 基座跟随手腕, 初始位置用第一帧
            if wrist_is_valid:
                self._base_pos = wrist_positions[0]
                self._base_quat = pr.quaternion_from_axis_angle(np.array([0, 0, 1, 0]))
            else:
                self._base_pos = np.array([0.0, 0.0, 0.3])
                self._base_quat = pr.quaternion_from_axis_angle(np.array([0, 0, 1, 0]))

        # 5. 加载机器人
        logger.info("=" * 60)
        logger.info(f"Step 5: 加载机器人 (mode={self.mode})")
        logger.info("=" * 60)
        self.robot_info = setup_robot(
            self.scene, self.mode, self.side, self._base_pos, self._base_quat
        )

        # Debug: 验证臂基座位置
        if self.mode == "full_robot":
            robot = self.robot_info["robot"]
            arm_base_offset_expected = ARM_BASE_OFFSET_LEFT if self.side == "left" else ARM_BASE_OFFSET_RIGHT
            for link in robot.get_links():
                if link.get_name() == f"{self.side}_arm_base_link":
                    abp = np.array(link.get_entity_pose().p)
                    logger.info(f"  [verify] ROOT={self._base_pos.round(3)}, "
                                f"arm_base={abp.round(3)}, "
                                f"offset={(abp - self._base_pos).round(3)}, "
                                f"expected_offset={arm_base_offset_expected.round(3)}")
                    break

        # 6. 初始化 Retargeting + IK
        self._init_retargeting()
        self._init_ik()

        # 6b. 初始化 hybrid 抓取控制器 (需要在 setup_robot 之后, 因为需要 robot 对象)
        if self.grasp_mode == "hybrid" and self.grasp_controllers is None:
            # 目标选择范式 (用户: "我需要夹住的是那个粉色的东西，放到碗里面"; "形成一个范式")
            # 1. 优先按颜色识别粉色物体 (find_pink_object), 在不同场景文件夹通用
            # 2. 找不到粉色物体时, 退回轨迹距离选择 (find_target_object_by_trajectory)
            # 3. 同时识别碗 (find_bowl, 按几何特征), 作为 pick-and-place 放置目标
            obj_sapien_pos = {name: np.array(c) for name, c in self.obj_bbox_centers.items()}
            target_objs = {}
            bowl_objs = {}

            # 1. 按颜色识别粉色物体 (范式, 通用)
            pink_obj = find_pink_object(self.obj_info)
            # 2. 识别碗 (按几何, 排除粉色物体)
            bowl_obj = find_bowl(self.obj_info, exclude_names=[pink_obj] if pink_obj else None)
            logger.info(f"  [范式] 粉色物体={pink_obj}, 碗={bowl_obj}")

            for s in self.sides:
                tgt = pink_obj
                # 退回: 无粉色物体时用轨迹距离选择
                if tgt is None:
                    if self.side == "both" and isinstance(hawor_data, dict) and s in hawor_data:
                        trans_side = hawor_data[s]["pred_trans"]
                    else:
                        trans_side = hawor_data["pred_trans"]
                    tgt = find_target_object_by_trajectory(np.asarray(trans_side), obj_sapien_pos)
                target_objs[s] = tgt
                bowl_objs[s] = bowl_obj
                if tgt:
                    tgt_pos = obj_sapien_pos[tgt]
                    logger.info(f"  [{s}] 锁定抓取目标: {tgt} (sapien_pos={tgt_pos.round(3)})")
                if bowl_obj:
                    bowl_pos = obj_sapien_pos[bowl_obj]
                    logger.info(f"  [{s}] 锁定放置目标 (碗): {bowl_obj} (sapien_pos={bowl_pos.round(3)})")
            self.grasp_controllers = {
                s: HybridGraspController(
                    self.obj_actors, side=s, scene=self.scene,
                    robot=self.robot_info.get("robot"),
                    target_obj=target_objs.get(s),
                    obj_positions=self.obj_bbox_centers,
                    bowl_obj=bowl_objs.get(s)
                ) for s in self.sides
            }
            logger.info(f"  混合抓取控制器已启用 (grasp_mode=hybrid), "
                        f"MANO curl→力度, 接触力控, 目标力={TARGET_GRASP_FORCE}N, "
                        f"抓取目标: {target_objs}, 放置目标 (碗): {bowl_objs}")

            # 6c. 计算帧级偏移参数 (按 plan: F0=0, F50=grasp_offset, F95=0, F112=0)
            # 如果已从优化器复制了 _frame_params 和 _fixed_offsets_654, 则跳过
            _has_frame_params = getattr(self, '_frame_params', None) is not None
            _has_fixed_offsets = getattr(self, '_fixed_offsets_654', None) is not None
            if _has_frame_params and _has_fixed_offsets:
                logger.info(f"  复用优化器的帧参数: FIXED={self._frame_params['fixed_frames']}")
            else:
                FINGER_FORWARD_NEUTRAL = 0.037  # 手指在 EE 前方 3.7cm (沿 gripper_R 的 X 轴)
                self._mano_neutral_offset = {}
                self._mano_grasp_frame = {}  # side -> f_grasp (MANO 最接近目标的帧, 用于阶段判定)
                self._mano_target_pos = {}  # side -> target_pos (CLOSE 阶段手指保持位置)
                for s in self.sides:
                    tgt = target_objs.get(s)
                    traj = self._mano_gripper_traj.get(s)
                    if (tgt is None or tgt not in self.obj_bbox_centers or traj is None
                            or len(traj["pos"]) == 0 or "R" not in traj or len(traj["R"]) == 0):
                        self._mano_neutral_offset[s] = None
                        self._mano_grasp_frame[s] = None
                        self._mano_target_pos[s] = None
                        continue
                    target_pos = np.array(self.obj_bbox_centers[tgt], dtype=np.float64)
                    # 找 MANO 夹爪最接近目标物体的帧 (offset 最小化的对齐点)
                    mano_positions = traj["pos"]
                    dists = np.linalg.norm(mano_positions - target_pos, axis=1)
                    f_grasp = int(np.argmin(dists))
                    # 用 f_grasp 处的 MANO root_R 计算手指前向偏移 (位姿不改变, 用真实朝向)
                    R_at_grasp = traj["R"][f_grasp]  # (3, 3)
                    finger_offset = R_at_grasp[:, 0] * FINGER_FORWARD_NEUTRAL  # 沿 gripper X 轴前向
                    # 最小 offset: EE + finger_offset 对齐 target_pos
                    offset = target_pos - mano_positions[f_grasp] - finger_offset
                    self._mano_neutral_offset[s] = offset
                    self._mano_grasp_frame[s] = f_grasp
                    self._mano_target_pos[s] = target_pos
                    logger.info(f"  [neutral][{s}] MANO 最接近 {tgt} @ F{f_grasp} "
                                f"(dist={dists[f_grasp]:.3f}m), minimal offset={offset.round(3)} "
                                f"(||offset||={np.linalg.norm(offset):.3f}m, "
                                f"MANO@F{f_grasp}={mano_positions[f_grasp].round(3)}, "
                                f"target={target_pos.round(3)}, "
                                f"R_X_axis={R_at_grasp[:, 0].round(3)}, "
                                f"R_Y_axis={R_at_grasp[:, 1].round(3)}, "
                                f"R_Z_axis={R_at_grasp[:, 2].round(3)})")

        # 7. 相机设置 (按 --views 指定渲染哪些视角, 对齐 002_render_scene.py)
        cam_view = None
        god_view = None
        # v4.15: 存储为实例属性, 供 Stage 3 rollout 录制视频
        self._cam_view = None
        self._god_view = None
        self._writer_cam = None
        self._writer_god = None
        focal = float(hawor_data.get("img_focal", HAWOR_FOCAL_DEFAULT))
        render_cam = render_available and self.views in ("cam", "both")
        render_god = render_available and self.views in ("god", "both")
        if render_cam:
            # 7a. 相机视角 (第一人称): 用 02 的 hawor_cam_to_sapien_pose (正确映射)
            cam_view = self.scene.add_camera(
                "cam_view", CAM_WIDTH, CAM_HEIGHT,
                1.0, 0.01, 100.0,
            )
            R_c2w = R_c2w_all[min(self.start_frame, len(R_c2w_all) - 1)]
            t_c2w = t_c2w_all[min(self.start_frame, len(t_c2w_all) - 1)]
            cam_pos, cam_quat = hawor_cam_to_sapien_pose(
                R_c2w, t_c2w,
                R_h2g=self._cam_xform_R_h2g, t_h2g=self._cam_xform_t, s=self._cam_xform_s)
            cam_view.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
            cam_view.set_focal_lengths(focal, focal)
            self._cam_view = cam_view  # v4.15: 存储供 Stage 3 rollout 录制视频
            logger.info(f"  相机视角 (第一人称): pos={cam_pos.round(3)}, focal={focal:.1f}")

        if render_god:
            # 7b. 上帝视角: 高空斜俯视, 能看到整个机器人 + 夹爪 + 物体
            # 场景中心 = 抓取区域 (臂基座 + 物体质心), 不是地下 ROOT
            god_view = self.scene.add_camera(
                "god_view", CAM_WIDTH, CAM_HEIGHT,
                1.0, 0.01, 100.0,
            )
            arm_base_offset = ((ARM_BASE_OFFSET_LEFT + ARM_BASE_OFFSET_RIGHT) / 2.0) if self.side == "both" \
                else (ARM_BASE_OFFSET_LEFT if self.side == "left" else ARM_BASE_OFFSET_RIGHT)
            arm_base_pos = self._base_pos + arm_base_offset  # 臂基座 (地上, ~z=0.35)
            if self.obj_bbox_centers:
                obj_centers = np.array(list(self.obj_bbox_centers.values()))
                obj_centroid = obj_centers.mean(axis=0)
                # 场景中心 = 臂基座和物体的中间 (抓取区域)
                scene_center = (arm_base_pos + obj_centroid) / 2.0
            else:
                scene_center = arm_base_pos.copy()
            # 上帝视角: 放在机器人前方高处, 俯视抓取区域
            # 用 base_quat (含 yaw) 旋转 [1,0,0] (URDF 默认前方) 得到机器人当前前方
            # 修复: 旧版用世界 -Y, gripper_only 机器人 yaw 旋转后前方不是 -Y → 视角反了
            try:
                base_q = self._base_quat if self._base_quat is not None else np.array([1.0, 0.0, 0.0, 0.0])
                R_base = pr.matrix_from_quaternion(base_q)
                forward_3d = R_base @ np.array([1.0, 0.0, 0.0])
                forward_2d = np.array([forward_3d[0], forward_3d[1], 0.0])
                norm = float(np.linalg.norm(forward_2d))
                if norm > 1e-6:
                    forward_2d = forward_2d / norm
                else:
                    forward_2d = np.array([0.0, -1.0, 0.0])
            except Exception:
                forward_2d = np.array([0.0, -1.0, 0.0])
            # god_view 固定相机在正上方俯瞰整个场景 (用户: "固定的摄像头在上方, 不要有什么跟随的操作")
            # 高度足够覆盖整个抓取工作空间 (物体 + 夹爪活动范围)
            # 用户反馈: "god视角太高了, 得低一点" → 1.0m → 0.5m → 0.2m (用户: "god降低到0.2")
            # pick-and-place 需要看到物体被放入碗的全过程, 高度低更清晰
            if self.mode == "gripper_only":
                god_height = 0.20  # 上方 0.2m (用户: "god降低到0.2")
            else:
                god_height = 1.50   # 上方 1.5m (俯瞰全机器人 + 抓取区域)
            # 相机在场景中心正上方, 往下看; up 方向用 forward_2d
            # 第十四轮修复: god_pos_z 用 self._ground_z (GLB 预扫描的真正地面高度)
            # 第十三轮用 obj_centers[:, 2].min() (物体中心最小Z), 物体在桌面上时偏高
            # 用户: "固定 0.2m还是很高，不知道为什么，是坐标系的问题吗？"
            ground_z = float(getattr(self, '_ground_z', 0.0))
            god_pos = np.array([scene_center[0], scene_center[1], ground_z + god_height])
            god_look_at = np.array([scene_center[0], scene_center[1], ground_z])
            god_quat = make_look_at_camera(god_pos, god_look_at, up=forward_2d)
            god_view.set_local_pose(sapien.Pose(god_pos.tolist(), god_quat.tolist()))
            god_view.set_focal_lengths(focal, focal)
            self._god_view = god_view  # v4.15: 存储供 Stage 3 rollout 录制视频
            # 固定相机: 不跟随夹爪, 初始化后位置不变
            self._god_height = god_height
            self._god_follow = False
            logger.info(f"  上帝视角: pos={god_pos.round(3)}, 看向={scene_center.round(3)} "
                        f"(height={god_height}m, 固定正上方俯瞰, 不跟随)")

        # 8. 视频录制 (按 --views 仅创建需要的 writer)
        _video_tag = f"_{self.video_tag}" if self.video_tag else ""
        video_path_cam = str(self.output_dir / f"cam_view{_video_tag}_{self.mode}_{self.side}.mp4")
        video_path_god = str(self.output_dir / f"god_view{_video_tag}_{self.mode}_{self.side}.mp4")
        writer_cam = None
        writer_god = None
        if render_cam:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_cam = cv2.VideoWriter(video_path_cam, fourcc, 30, (CAM_WIDTH, CAM_HEIGHT))
            self._writer_cam = writer_cam  # v4.15: 存储供 Stage 3 rollout 录制视频
        if render_god:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_god = cv2.VideoWriter(video_path_god, fourcc, 30, (CAM_WIDTH, CAM_HEIGHT))
            self._writer_god = writer_god  # v4.15: 存储供 Stage 3 rollout 录制视频

        # 8b. 初始化 MANO 3 参考点渲染 (仅渲染模式下创建)
        if render_available:
            self._init_mano_markers()
            # 8c. 初始化碰撞模型可视化 (第十三轮: 用户要求"把碰撞模型也展示到视频里面")
            # 半透明红色 RGBA[1,0,0,0.4] 覆盖在手指视觉模型外, 跟随手指 link 位姿
            self._init_collision_visualization(self.robot_info["robot"])

        # 8d. 交互式 Viewer (--viewer 模式)
        viewer_win = None
        if self.viewer and render_available:
            try:
                from sapien.utils import Viewer as SapienViewer
                viewer_win = SapienViewer()
                viewer_win.set_scene(self.scene)
                viewer_win.control_window.show_origin_frame = False
                viewer_win.control_window.show_grid = False
                # 将 Viewer 相机放在场景上方斜视角, 能看到全貌
                scene_center_view = np.array([0.0, 0.0, 0.0])
                if self.obj_bbox_centers:
                    centers = np.array(list(self.obj_bbox_centers.values()))
                    scene_center_view = centers.mean(axis=0)
                elif self._base_pos is not None:
                    scene_center_view = self._base_pos.copy()
                viewer_eye = scene_center_view + np.array([0.5, -0.5, 0.6])
                viewer_quat = make_look_at_camera(viewer_eye, scene_center_view)
                viewer_win.set_camera_pose(sapien.Pose(viewer_eye.tolist(), viewer_quat.tolist()))
                logger.info(f"  交互式 Viewer 已启用, 相机位置={viewer_eye.round(3)}, 看向={scene_center_view.round(3)}")
                logger.info("  (关闭窗口或按 ESC 结束仿真)")
            except Exception as e:
                logger.warning(f"  创建 Viewer 失败: {e}")
                viewer_win = None

        # v4.4: Stage 1 快速测试仅需要初始化, 不进入完整仿真循环
        if getattr(self, '_test_stage1_only', False):
            logger.info("  [test-stage1] 初始化完成, 跳过完整仿真循环")
            return

        # v4.4: v4-pipeline 在 Stage 1/2/3 优化前不进入主循环, 避免无优化轨迹把物体推飞
        if not run_main_loop:
            logger.info("  [v4-pipeline] 初始化完成, 跳过主循环 (等待 Stage 1/2/3 优化)")
            return

        # 9. Warmup (smoothstep 过渡)
        logger.info("=" * 60)
        logger.info("Step 6: Warmup + 物理仿真")
        logger.info("=" * 60)
        robot = self.robot_info["robot"]
        arm_joint_indices = self.robot_info["arm_joint_indices"]
        gripper_idx1 = self.robot_info["gripper_idx1"]
        gripper_idx2 = self.robot_info["gripper_idx2"]
        joint_names = self.robot_info["joint_names"]

        # 第二十六轮: 若已知优化时物体初始位姿, 重置物体以保证回放一致性
        if hasattr(self, '_obj_initial_poses') and self._obj_initial_poses:
            for actor in self.obj_actors:
                name = actor.get_name()
                if name in self._obj_initial_poses:
                    actor.set_pose(self._obj_initial_poses[name])
                    try:
                        for comp in actor.get_components():
                            if hasattr(comp, 'set_linear_velocity'):
                                comp.set_linear_velocity([0, 0, 0])
                                comp.set_angular_velocity([0, 0, 0])
                                break
                    except Exception:
                        pass

        # Warmup: 让物理稳定
        if self.side == "both":
            # 双手: 一次 physics_step 设置两侧臂 + 夹爪起始位置
            left_arm_idxs = [i for i, n in enumerate(joint_names) if "left_arm_joint" in n]
            right_arm_idxs = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
            gi_l = self.robot_info["gripper_indices"]["left"]
            gi_r = self.robot_info["gripper_indices"]["right"]
            for _ in range(WARMUP_FRAMES):
                physics_step(
                    robot, left_arm_idxs, gi_l[0], gi_l[1],
                    LEFT_ARM_STARTING, GRIPPER_INIT_OPEN, -GRIPPER_INIT_OPEN, self.scene,
                    extra_gripper_indices=[(gi_r[0], GRIPPER_INIT_OPEN), (gi_r[1], -GRIPPER_INIT_OPEN)],
                    extra_arm_indices=right_arm_idxs, extra_arm_target=RIGHT_ARM_STARTING,
                )
        else:
            for _ in range(WARMUP_FRAMES):
                physics_step(
                    robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                    RIGHT_ARM_STARTING if self.side == "right" else LEFT_ARM_STARTING,
                    GRIPPER_INIT_OPEN, -GRIPPER_INIT_OPEN, self.scene
                )

        # 10. 主循环
        qpos_log = []
        contact_log = []
        obj_pos_log = []  # 参数级验证: 物体位置
        gripper_pos_log = []  # 综合抓取判定: 每帧每侧夹爪位置 (跟随判定用)
        track_log = []  # 轨迹跟踪误差: 每帧 MANO 期望 vs SAPIEN 实际 (root + finger)
        grasp_states = []

        # 记录物体初始位置 (用于验证提升)
        obj_initial_pos = {}
        for actor in self.obj_actors:
            obj_initial_pos[actor.name] = np.array(actor.get_pose().p)

        # === DEBUG 插桩 (opt-sim-divergence): 记录 run() 的轨迹, 对比 rollout_single ===
        _dbg_run_obj_z_traj = []
        _dbg_run_obj_xy_traj = []
        _dbg_run_gripper_z_traj = []
        _dbg_run_first_frame_state = None
        _dbg_target_obj_name = None
        if self.grasp_controllers and self.side in self.grasp_controllers:
            _dbg_target_obj_name = self.grasp_controllers[self.side].target_obj
        # === DEBUG 插桩结束 ===

        # 无效帧保持上一帧位姿 (跟踪丢失时机器人保持不动, 不是自由落体)
        if self.side == "both":
            last_arm_target = {}  # {"left": array, "right": array}
            last_gripper_t1 = {}
            last_gripper_t2 = {}
        else:
            last_arm_target = None
            last_gripper_t1 = GRIPPER_INIT_OPEN
            last_gripper_t2 = -GRIPPER_INIT_OPEN

        # 双手模式预计算臂/夹爪索引 (避免每帧重复查找)
        if self.side == "both":
            left_arm_idxs = [i for i, n in enumerate(joint_names) if "left_arm_joint" in n]
            right_arm_idxs = [i for i, n in enumerate(joint_names) if "right_arm_joint" in n]
            gi_l = self.robot_info["gripper_indices"]["left"]
            gi_r = self.robot_info["gripper_indices"]["right"]

        # === 判定是否可用优化轨迹路径 (hybrid 模式 + 优化结果) ===
        # 优化路径: 所有帧有效, 不需要 MANO FK, 直接 replay 优化轨迹
        _use_opt_path = (getattr(self, '_opt_mano_gripper_traj', None) is not None
                         and self.grasp_mode == "hybrid"
                         and self.grasp_controllers is not None)
        _n_frames = self.num_frames  # 原始帧数不变, 优化轨迹若短则 _compute_mano_neutral_target 内部 clamp
        # v15l: 在 run() 作用域定义 _F50 (供 debug 日志和闭合逻辑使用)
        _run_fp = getattr(self, '_frame_params', None)
        _run_F50 = _run_fp['F50_IDX'] if _run_fp else 50
        _run_F95 = _n_frames  # 与 _compute_mano_neutral_target 对齐
        if _use_opt_path:
            _opt_n = len(self._opt_mano_gripper_traj[self.side]["pos"])
            if _opt_n < _n_frames:
                _n_frames = _opt_n
            logger.info(f"  使用优化轨迹直接 replay: {_n_frames} 帧, 跳过 MANO FK")
        else:
            logger.info(f"  使用 MANO FK 路径 (opt_traj={getattr(self, '_opt_mano_gripper_traj', None) is not None}, "
                         f"hybrid={self.grasp_mode == 'hybrid'}, ctrl={self.grasp_controllers is not None})")

        def _is_frame_invalid(hd, gi):
            if gi >= len(hd["pred_valid"]) or not hd["pred_valid"][gi]:
                return True
            if gi >= len(hd["pred_trans"]):
                return True
            if np.isnan(hd["pred_trans"][gi]).any():
                return True
            return False

        pbar = trange(_n_frames, desc=f"grasp_{self.mode}_{self.side}")
        for local_idx in pbar:
            self._current_local_idx = local_idx

            if _use_opt_path:
                # ============================================================
                # 优化轨迹路径: 直接 replay, 不计算 MANO FK, 所有帧有效
                # ============================================================
                if self.mode == "full_robot":
                    if self.side == "both":
                        _jd = {s: np.zeros((21, 3)) for s in self.sides}
                        arm_targets, gripper_vals = self._step_full_robot(_jd)
                        physics_step(robot, left_arm_idxs, gi_l[0], gi_l[1],
                                     arm_targets["left"], gripper_vals["left"], -gripper_vals["left"], self.scene,
                                     extra_gripper_indices=[(gi_r[0], gripper_vals["right"]), (gi_r[1], -gripper_vals["right"])],
                                     extra_arm_indices=right_arm_idxs, extra_arm_target=arm_targets["right"])
                        last_arm_target = arm_targets
                        last_gripper_t1 = {s: gripper_vals[s] for s in self.sides}
                        last_gripper_t2 = {s: -gripper_vals[s] for s in self.sides}
                    else:
                        ik_j, gv = self._step_full_robot(np.zeros((21, 3)))
                        physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                     ik_j, gv, -gv, self.scene,
                                     lock_root_pose=getattr(self, '_close_lock_pose', None))
                        last_arm_target = ik_j
                        last_gripper_t1 = gv
                        last_gripper_t2 = -gv
                else:
                    _, (j1, j2) = self._step_gripper_only(np.zeros((21, 3)))
                    # v15l: 构建虚拟关节锁定目标 — 每个子步后强制 set_qpos + qvel=0
                    # 根因: physics_step 8 个子步内 virtual_z PD 漂移 3cm
                    #   (target=0.034, 实际漂到 0.064, 手指仍在物体上方1cm)
                    # 修复: 传入 virtual_lock_targets, 在 decimation 循环内每步锁定
                    _virtual_idx_lock = self.robot_info.get("virtual_idx", {})
                    _gripper_pos_lock = getattr(self, '_current_gripper_pos', None)
                    _gripper_R_lock = getattr(self, '_current_gripper_R', None)
                    _vlock = None
                    if _virtual_idx_lock and _gripper_pos_lock is not None and _gripper_R_lock is not None:
                        from scipy.spatial.transform import Rotation as _R_lock
                        _rz_l, _ry_l, _rx_l = rotmat_to_zyx_euler(_gripper_R_lock)
                        # v15n4: vlock 不锁手指!
                        #   v15n3 问题: vlock+set_qpos 双重锁死手指, 重叠推飞物体
                        #   修复: 闭合阶段手指只用 PD 自然闭合, 碰到物体后接触力自然平衡
                        #   set_qpos 仍设定手指初始位置, 但 vlock 不强制 → 允许弹性变形
                        _vlock = {
                            _virtual_idx_lock['vx']: float(_gripper_pos_lock[0]),
                            _virtual_idx_lock['vy']: float(_gripper_pos_lock[1]),
                            _virtual_idx_lock['vz']: float(_gripper_pos_lock[2]),
                            _virtual_idx_lock['rz']: float(_rz_l),
                            _virtual_idx_lock['ry']: float(_ry_l),
                            _virtual_idx_lock['rx']: float(_rx_l),
                        }
                    # v15m debug: 验证 vlock 包含手指 + vz 漂移
                    if local_idx == _run_F50:
                        _vz_idx = _virtual_idx_lock.get('vz', -1)
                        if _vlock is not None:
                            logger.warning(f"  [v15m] F50 vlock OK: vz_target={_vlock.get(_vz_idx, 'N/A')}, "
                                            f"n_keys={len(_vlock)}")
                        else:
                            logger.warning(f"  [v15m] F50 vlock=None! vid={_virtual_idx_lock}, "
                                            f"pos={_gripper_pos_lock is not None}, R={_gripper_R_lock is not None}")
                    # v15m: 检查 physics_step 后的 vz + 手指漂移 (每10帧)
                    if local_idx > _run_F50 and local_idx % 10 == 0 and _vlock is not None:
                        _vz_idx = _virtual_idx_lock.get('vz', -1)
                        _qpos_after = robot.get_qpos()
                        if _vz_idx >= 0:
                            _vz_after = float(_qpos_after[_vz_idx])
                            _vz_target = _vlock.get(_vz_idx, 0)
                            _drift = abs(_vz_after - float(_vz_target))
                            _f1_after = float(_qpos_after[gripper_idx1])
                            _f1_target = _vlock.get(gripper_idx1, 0)
                            _f1_drift = abs(_f1_after - float(_f1_target))
                            logger.info(f"  [v15m] F{local_idx} vz: actual={_vz_after:.4f}, target={float(_vz_target):.4f}, drift={_drift:.4f}m | "
                                        f"f1: actual={_f1_after:.4f}, target={float(_f1_target):.4f}, drift={_f1_drift:.4f}")
                    physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                 np.array([]), j1, j2, self.scene,
                                 lock_root_pose=getattr(self, '_close_lock_pose', None),
                                 virtual_lock_targets=_vlock)
                    # v15l: post-step z 纠正已移除, 由 virtual_lock_targets (每子步锁定) 替代


                qpos_log.append(robot.get_qpos().copy())
                n_contacts, total_impulse, per_obj = fetch_contacts(
                    robot, self.obj_actors, self.side, self.scene)
                is_grasping = n_contacts >= 2

                # Debug: 每 10 帧打印 EE 位置 vs 最近物体
                if (local_idx + 1) % 10 == 0 or local_idx == 0:
                    for s in self.sides:
                        ee_pos = None
                        for link in robot.get_links():
                            if link.get_name() == f"{s}_gripper_link":
                                ee_pos = np.array(link.get_entity_pose().p)
                                break
                        if ee_pos is not None and self.obj_bbox_centers:
                            dists = [(n, float(np.linalg.norm(ee_pos - np.array(c))), c)
                                     for n, c in self.obj_bbox_centers.items()]
                            dists.sort(key=lambda x: x[1])
                            n, d, c = dists[0]
                            if local_idx == 0 or (local_idx + 1) % 10 == 0:
                                # 调试: 也打印手指位置
                                _f1_pos = _f2_pos = None
                                for link in robot.get_links():
                                    if link.get_name() == f"{s}_gripper_finger_link1":
                                        _f1_pos = np.array(link.get_entity_pose().p)
                                    elif link.get_name() == f"{s}_gripper_finger_link2":
                                        _f2_pos = np.array(link.get_entity_pose().p)
                                _finger_info = ""
                                if _f1_pos is not None:
                                    _finger_info = f", f1={_f1_pos.round(3)}, f2={_f2_pos.round(3)}" if _f2_pos is not None else f", f1={_f1_pos.round(3)}"
                                logger.info(f"  [debug] F{local_idx+1} [{s}]: ee={ee_pos.round(3)}, "
                                            f"nearest={n} dist={d:.3f} center={np.array(c).round(3)}{_finger_info}")
            else:
                # ============================================================
                # 原有路径: MANO FK → retargeting/IK → 物理步进
                # ============================================================
                global_idx = self.start_frame + local_idx
                if global_idx >= n_total:
                    break

                if self.side == "both":
                    is_invalid = any(_is_frame_invalid(hawor_data[s], global_idx) for s in self.sides)
                else:
                    is_invalid = _is_frame_invalid(hawor_data, global_idx)

                if is_invalid:
                    if self.side == "both":
                        if last_arm_target:
                            physics_step(robot, left_arm_idxs, gi_l[0], gi_l[1],
                                         last_arm_target["left"], last_gripper_t1["left"], last_gripper_t2["left"], self.scene,
                                         extra_gripper_indices=[(gi_r[0], last_gripper_t1["right"]), (gi_r[1], last_gripper_t2["right"])],
                                         extra_arm_indices=right_arm_idxs, extra_arm_target=last_arm_target["right"])
                        else:
                            _floating = _is_floating_root(robot)
                            for _ in range(DECIMATION):
                                qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                                if _floating:
                                    qf[:6] = 0
                                robot.set_qf(qf)
                                self.scene.step()
                    else:
                        if last_arm_target is not None:
                            physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                         last_arm_target, last_gripper_t1, last_gripper_t2, self.scene)
                        else:
                            _floating = _is_floating_root(robot)
                            for _ in range(DECIMATION):
                                qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                                if _floating:
                                    qf[:6] = 0
                                robot.set_qf(qf)
                                self.scene.step()
                    qpos_log.append(robot.get_qpos().copy())
                    n_contacts, total_impulse, per_obj = 0, 0.0, {}
                    is_grasping = False
                else:
                    if self.side == "both":
                        joints_dict = {}
                        for s in self.sides:
                            hd = hawor_data[s]
                            _, joints = compute_mano_joints(mano_layer[s], hd["pred_rot"][global_idx],
                                                            hd["pred_hand_pose"][global_idx], hd["pred_trans"][global_idx])
                            joints_dict[s] = _mano_to_sapien(joints)
                        if self.mode == "full_robot":
                            arm_targets, gripper_vals = self._step_full_robot(joints_dict)
                            physics_step(robot, left_arm_idxs, gi_l[0], gi_l[1],
                                         arm_targets["left"], gripper_vals["left"], -gripper_vals["left"], self.scene,
                                         extra_gripper_indices=[(gi_r[0], gripper_vals["right"]), (gi_r[1], -gripper_vals["right"])],
                                         extra_arm_indices=right_arm_idxs, extra_arm_target=arm_targets["right"])
                            last_arm_target = arm_targets
                            last_gripper_t1 = {s: gripper_vals[s] for s in self.sides}
                            last_gripper_t2 = {s: -gripper_vals[s] for s in self.sides}
                        else:
                            _floating = _is_floating_root(robot)
                            for _ in range(DECIMATION):
                                qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
                                if _floating:
                                    qf[:6] = 0
                                robot.set_qf(qf)
                                self.scene.step()
                        first_js = joints_dict.get(self.sides[0])
                        if first_js is not None and len(first_js) >= 9:
                            self._update_mano_markers(first_js[0, :3], first_js[4, :3], first_js[8, :3])
                    else:
                        _, joints = compute_mano_joints(mano_layer, hawor_data["pred_rot"][global_idx],
                                                        hawor_data["pred_hand_pose"][global_idx], hawor_data["pred_trans"][global_idx])
                        joints_sapien = _mano_to_sapien(joints)
                        if self.mode == "full_robot":
                            ik_joints, gripper_val = self._step_full_robot(joints_sapien)
                            arm_target, gripper_t1, gripper_t2 = ik_joints, gripper_val, -gripper_val
                        else:
                            _, (joint1, joint2) = self._step_gripper_only(joints_sapien)
                            arm_target, gripper_t1, gripper_t2 = np.array([]), joint1, joint2
                        # v15m: 非 opt_path 也加 virtual_lock_targets + 手指
                        _vlock_nonopt = None
                        _vid_nonopt = self.robot_info.get("virtual_idx", {})
                        _vpos_nonopt = getattr(self, '_current_gripper_pos', None)
                        _vR_nonopt = getattr(self, '_current_gripper_R', None)
                        if _vid_nonopt and _vpos_nonopt is not None and _vR_nonopt is not None:
                            from scipy.spatial.transform import Rotation as _R_nonopt
                            _rz_n, _ry_n, _rx_n = rotmat_to_zyx_euler(_vR_nonopt)
                            _vlock_nonopt = {
                                _vid_nonopt['vx']: float(_vpos_nonopt[0]),
                                _vid_nonopt['vy']: float(_vpos_nonopt[1]),
                                _vid_nonopt['vz']: float(_vpos_nonopt[2]),
                                _vid_nonopt['rz']: float(_rz_n),
                                _vid_nonopt['ry']: float(_ry_n),
                                _vid_nonopt['rx']: float(_rx_n),
                                # v15n4: 手指不加入 vlock, 让 PD 自然处理接触力
                            }
                        # v15m debug: non-opt_path F50 验证 vlock
                        if local_idx == _run_F50:
                            _vz_idx_n = _vid_nonopt.get('vz', -1)
                            if _vlock_nonopt is not None:
                                logger.warning(f"  [v15m] F50 non-opt vlock OK: vz_target={_vlock_nonopt.get(_vz_idx_n, 'N/A')}, "
                                                f"n_keys={len(_vlock_nonopt)}")
                            else:
                                logger.warning(f"  [v15m] F50 non-opt vlock=None! vid={_vid_nonopt}, "
                                                f"pos={_vpos_nonopt is not None}, R={_vR_nonopt is not None}")
                        physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                     arm_target, gripper_t1, gripper_t2, self.scene,
                                     lock_root_pose=getattr(self, '_close_lock_pose', None),
                                     virtual_lock_targets=_vlock_nonopt)
                        last_arm_target = arm_target
                        last_gripper_t1 = gripper_t1
                        last_gripper_t2 = gripper_t2
                        if len(joints_sapien) >= 9:
                            self._update_mano_markers(joints_sapien[0, :3], joints_sapien[4, :3], joints_sapien[8, :3])

                    qpos_log.append(robot.get_qpos().copy())

                # === DEBUG 插桩 (opt-sim-divergence): 记录 run() 每帧状态 ===
                if _dbg_target_obj_name is not None:
                    for actor in self.obj_actors:
                        if actor.name == _dbg_target_obj_name:
                            _o_pos = np.array(actor.get_pose().p)
                            _dbg_run_obj_z_traj.append(float(_o_pos[2]))
                            _dbg_run_obj_xy_traj.append(_o_pos[:2].tolist())
                            break
                if hasattr(self, '_current_gripper_pos'):
                    _dbg_run_gripper_z_traj.append(float(self._current_gripper_pos[2]))
                # 第一帧详细状态 (对比 rollout_single 的第一帧)
                if local_idx == 0 and _dbg_target_obj_name is not None:
                    robot_qpos = robot.get_qpos().copy()
                    robot_qvel = robot.get_qvel().copy()
                    for actor in self.obj_actors:
                        if actor.name == _dbg_target_obj_name:
                            _o_pos = np.array(actor.get_pose().p)
                            _dbg_run_first_frame_state = {
                                'robot_qpos': robot_qpos.tolist(),
                                'robot_qvel': robot_qvel.tolist(),
                                'gripper_idx1': gripper_idx1,
                                'gripper_idx2': gripper_idx2,
                                'obj_pos': _o_pos.tolist(),
                                'root_pose': robot.get_root_pose().p.tolist(),
                            }
                            break
                # === DEBUG 插桩结束 ===

                # Debug: 每 10 帧打印 EE 位置 vs 最近物体 (遍历所有侧)
                if (local_idx + 1) % 10 == 0 or local_idx == 0:
                    for s in self.sides:
                        ee_pos = None
                        for link in robot.get_links():
                            if link.get_name() == f"{s}_gripper_link":
                                ee_pos = np.array(link.get_entity_pose().p)
                                break
                        if ee_pos is not None and self.obj_bbox_centers:
                            dists = []
                            for name, center in self.obj_bbox_centers.items():
                                d = float(np.linalg.norm(ee_pos - np.array(center)))
                                dists.append((name, d, center))
                            dists.sort(key=lambda x: x[1])
                            name, d, center = dists[0]
                            logger.info(f"  [debug] F{local_idx+1} [{s}]: ee={ee_pos.round(3)}, "
                                        f"nearest={name} dist={d:.3f} center={np.array(center).round(3)}")

                # 接触检测
                n_contacts, total_impulse, per_obj = fetch_contacts(
                    robot, self.obj_actors, self.side, self.scene
                )
                is_grasping = n_contacts >= 2

                # 目标物体接触详情 (CLOSE/LIFT/TRANSPORT 阶段诊断: 手指是否真碰到目标)
                _phase = getattr(self, '_last_phase', '')
                _ctrl = self.grasp_controllers.get(self.side) if self.grasp_controllers else None
                _tgt_name = _ctrl.target_obj if _ctrl else None
                _f_grasp = getattr(self, '_mano_grasp_frame', {}).get(self.side, 0) or 0
                # CLOSE 阶段 + 前后 5 帧都记录 (看接触何时建立/丢失)
                _close_dur = max(3, int(max(self.num_frames, 1) * 0.15))
                _log_start = max(0, _f_grasp - 5)
                _log_end = _f_grasp + _close_dur + 15  # 覆盖 CLOSE+LIFT 早期
                if _tgt_name and _log_start <= local_idx <= _log_end:
                    _tgt_actor = None
                    for ac in self.obj_actors:
                        if ac.name == _tgt_name:
                            _tgt_actor = ac
                            break
                    _tgt_pos = np.array(_tgt_actor.get_pose().p).round(4).tolist() if _tgt_actor else None
                    _tgt_c = per_obj.get(_tgt_name, {"n": 0, "impulse": 0.0})
                    # 手指实际位置 (诊断夹爪是否包围物体)
                    _finger_positions = []
                    for link in robot.get_links():
                        _lname = link.get_name()
                        if "finger" in _lname and self.side in _lname:
                            _fp = np.array(link.get_entity_pose().p)
                            _finger_positions.append(f"{_lname.split('_')[-1]}={_fp.round(3).tolist()}")
                    logger.info(f"  [grasp] F{local_idx+1} phase={_phase} {_tgt_name}: "
                                f"pos={_tgt_pos} contact_n={_tgt_c['n']} "
                                f"impulse={_tgt_c['impulse']:.4f} "
                                f"fingers=[{' | '.join(_finger_positions)}]")

            # 公共: 日志记录 + 渲染 (有效帧和无效帧都执行, 确保视频帧数 == num_frames)
            contact_log.append({
                "frame": local_idx,
                "contacts": n_contacts,
                "impulse": total_impulse,
                "per_obj": per_obj,
            })
            grasp_states.append(is_grasping)

            # 物体位置记录 (参数级验证)
            frame_obj_pos = {}
            for actor in self.obj_actors:
                frame_obj_pos[actor.name] = np.array(actor.get_pose().p).tolist()
            obj_pos_log.append(frame_obj_pos)

            # v15i DIAG: 关键帧打印手指 qpos + 位置关系 (验证真正抓取)
            _diag_tgt = None
            if hasattr(self, 'grasp_controllers') and self.side in self.grasp_controllers:
                _diag_tgt = self.grasp_controllers[self.side].target_obj
            if local_idx in [50, 55, 60, 70, 80, 90, 100, 108] and _diag_tgt is not None:
                _tgt = _diag_tgt
                _obj_pos = np.array(frame_obj_pos.get(_tgt, [0, 0, 0]))
                _f1p, _f2p, _gp = None, None, None
                _aq = robot.get_qpos()
                _f1q = float(_aq[gripper_idx1]) if gripper_idx1 is not None else None
                _f2q = float(_aq[gripper_idx2]) if gripper_idx2 is not None else None
                for lnk in robot.get_links():
                    _ln = lnk.get_name()
                    if _ln == f"{self.side}_gripper_finger_link1":
                        _f1p = np.array(lnk.get_entity_pose().p)
                    elif _ln == f"{self.side}_gripper_finger_link2":
                        _f2p = np.array(lnk.get_entity_pose().p)
                    elif _ln == f"{self.side}_gripper_link":
                        _gp = np.array(lnk.get_entity_pose().p)
                _extra = ""
                if _f1p is not None and _f2p is not None:
                    _y_gap = abs(_f1p[1] - _f2p[1])
                    _f1z_o = _f1p[2] - _obj_pos[2]
                    _gz_o = _gp[2] - _obj_pos[2] if _gp is not None else 0.0
                    _extra = (f", y_gap={_y_gap:.4f}, f1-obj_z={_f1z_o:+.4f}"
                              f"({'上方' if _f1z_o > 0.005 else '下方' if _f1z_o < -0.005 else '夹住'})"
                              f", glink-obj_z={_gz_o:+.4f}")
                _qp = f", qpos=[{_f1q:.4f},{_f2q:.4f}]" if _f1q is not None else ""
                _cur_pos = getattr(self, '_current_gripper_pos', None)
                _base_str = (_cur_pos.round(4) if _cur_pos is not None else 'N/A')
                logger.warning(f"  [DIAG F{local_idx}] obj={_obj_pos.round(4)}, "
                                f"f1={(_f1p.round(4) if _f1p is not None else 'N/A')}, "
                                f"f2={(_f2p.round(4) if _f2p is not None else 'N/A')}, "
                                f"base={_base_str}{_extra}{_qp}")

            # 夹爪位置记录 (综合抓取判定: 跟随判定用) + 位姿 (相机跟随用)
            frame_gripper_pos = {}
            frame_gripper_pose = {}  # {side: sapien.Pose} 含旋转, 用于相机跟随夹爪实际朝向
            for s in self.sides:
                for link in robot.get_links():
                    if link.get_name() == f"{s}_gripper_link":
                        pose = link.get_entity_pose()
                        frame_gripper_pos[s] = np.array(pose.p).tolist()
                        frame_gripper_pose[s] = pose
                        break
            gripper_pos_log.append(frame_gripper_pos)
            # 轨迹跟踪误差收集 (物理输出 vs 真实 MANO 期望)
            track_log.append(getattr(self, '_last_track', None))

            # 渲染 (按 --views 渲染选定视角, 按 --viewer 显示交互窗口)
            # GPU 可能在运行时丢失 (vk::DeviceLostError), 捕获后降级为纯物理模式
            if render_available and (writer_cam is not None or writer_god is not None or viewer_win is not None):
                try:
                    # 第十三轮: 每帧更新碰撞可视化位姿 (跟随手指 link, 半透明红色)
                    self._update_collision_visualization()
                    self.scene.update_render()
                    # 交互式 Viewer
                    if viewer_win is not None:
                        viewer_win.render()
                        try:
                            viewer_closed = not viewer_win.window.is_running
                        except AttributeError:
                            try:
                                viewer_closed = not viewer_win.control_window.is_running
                            except AttributeError:
                                viewer_closed = False
                        if viewer_closed:
                            logger.info("  Viewer 窗口关闭, 提前退出仿真")
                            break
                    if writer_cam is not None:
                        # 第一人称: 每帧按 HaWoR 相机轨迹更新 (对齐 002_render_scene.py L2588-2590)
                        global_idx = self.start_frame + local_idx
                        if global_idx < len(R_c2w_all):
                            cam_pos, cam_quat = hawor_cam_to_sapien_pose(
                                R_c2w_all[global_idx], t_c2w_all[global_idx],
                                R_h2g=self._cam_xform_R_h2g, t_h2g=self._cam_xform_t, s=self._cam_xform_s)
                            cam_view.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                        # 相机视角 (第一人称)
                        cam_view.take_picture()
                        rgb_cam = cam_view.get_picture("Color")[..., :3]
                        bgr_cam = np.ascontiguousarray((np.clip(rgb_cam, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        h, w = bgr_cam.shape[:2]
                        cv2.rectangle(bgr_cam, (0, 0), (w, 40), (0, 0, 0), -1)
                        label = f"F{local_idx+1}/{self.num_frames} | {self.mode} | CamView | C:{n_contacts} | Grasp:{is_grasping}"
                        cv2.putText(bgr_cam, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if is_grasping else (255, 255, 255), 2)
                        writer_cam.write(bgr_cam)
                    if writer_god is not None:
                        # 上帝视角: 固定相机在正上方俯瞰 (用户: "固定的摄像头在上方, 不要有什么跟随的操作")
                        # 初始化时已 set_local_pose, 这里只 take_picture, 不再每帧更新位置
                        god_view.take_picture()
                        rgb_god = god_view.get_picture("Color")[..., :3]
                        bgr_god = np.ascontiguousarray((np.clip(rgb_god, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        h, w = bgr_god.shape[:2]
                        cv2.rectangle(bgr_god, (0, 0), (w, 40), (0, 0, 0), -1)
                        label = f"F{local_idx+1}/{self.num_frames} | {self.mode} | GodView | C:{n_contacts} | Grasp:{is_grasping}"
                        cv2.putText(bgr_god, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if is_grasping else (255, 255, 255), 2)
                        writer_god.write(bgr_god)
                except RuntimeError as e:
                    # vk::DeviceLostError 等 GPU 运行时错误: 永久禁用渲染, 继续物理仿真
                    logger.warning(f"  GPU 渲染失败 ({e}), 降级为纯物理模式 (后续帧不再渲染)")
                    render_available = False
                    for w_ in (writer_cam, writer_god):
                        if w_ is not None:
                            try:
                                w_.release()
                            except Exception:
                                pass
                    writer_cam = None
                    writer_god = None

            if (local_idx + 1) % 30 == 0:
                obj_detail = " | ".join(f"{k}:C={v['n']}" for k, v in per_obj.items())
                logger.info(f"  帧 {local_idx+1}/{self.num_frames}: contacts={n_contacts}, "
                            f"impulse={total_impulse:.2f}, grasp={is_grasping}, {obj_detail}")

        if writer_cam is not None:
            writer_cam.release()
            logger.info(f"  相机视角视频已保存: {video_path_cam}")
        if writer_god is not None:
            writer_god.release()
            logger.info(f"  上帝视角视频已保存: {video_path_god}")

        # 11. 参数级验证
        self._verify_results(qpos_log, contact_log, obj_pos_log, obj_initial_pos, grasp_states, gripper_pos_log, track_log)

        # 12. 保存结果
        qpos_path = str(self.output_dir / f"grasp_{self.mode}_{self.side}_qpos.npy")
        np.save(qpos_path, np.array(qpos_log))
        logger.info(f"  qpos 已保存: {qpos_path}")

        # === DEBUG 插桩 (opt-sim-divergence): 保存 run() 的轨迹, 对比 verify ===
        np.save(self.output_dir / "run_obj_z_traj.npy", np.array(_dbg_run_obj_z_traj))
        np.save(self.output_dir / "run_gripper_z_traj.npy", np.array(_dbg_run_gripper_z_traj))
        np.save(self.output_dir / "run_obj_xy_traj.npy", np.array(_dbg_run_obj_xy_traj))
        if _dbg_run_first_frame_state is not None:
            import json
            with open(self.output_dir / "run_first_frame.json", 'w') as _f:
                json.dump(_dbg_run_first_frame_state, _f, indent=2)
            gi1, gi2 = _dbg_run_first_frame_state['gripper_idx1'], _dbg_run_first_frame_state['gripper_idx2']
            rq, rv = _dbg_run_first_frame_state['robot_qpos'], _dbg_run_first_frame_state['robot_qvel']
            logger.info(f"  [DEBUG run] first_frame: gripper_qpos=[{rq[gi1]:.6f},{rq[gi2]:.6f}], "
                        f"gripper_qvel=[{rv[gi1]:.6f},{rv[gi2]:.6f}], "
                        f"obj_pos={_dbg_run_first_frame_state['obj_pos']}, root_pose={_dbg_run_first_frame_state['root_pose']}")
        # === DEBUG 插桩结束 ===

        return {
            "video_cam": video_path_cam if render_available else None,
            "video_god": video_path_god if render_available else None,
            "qpos": qpos_path,
            "grasp_states": grasp_states,
        }

    def _verify_results(self, qpos_log, contact_log, obj_pos_log, obj_initial_pos, grasp_states, gripper_pos_log=None, track_log=None):
        """参数级验证: 物体提升 + 接触检测 + 综合抓取判定

        综合判定 (用户要求): 接触 + 跟随 + 提升 三项都满足才算真正夹住
          1. 接触: 连续 >= 10 帧有接触 (n_contacts >= 2)
          2. 跟随: 物体相对最近夹爪位置变化 < 5cm 持续 10 帧 (物体被夹住跟着夹爪走)
          3. 提升: 物体 z 提升量 > 5cm (相对初始位置)
        """
        logger.info("=" * 60)
        logger.info("Step 7: 参数级验证")
        logger.info("=" * 60)

        # 1. 机械臂关节数验证 (确认 "没有机械臂" 问题已修复)
        n_arm = len(self.robot_info["arm_joint_indices"])
        gripper_indices = self.robot_info["gripper_indices"]
        n_gripper = sum(2 for gi1, gi2 in gripper_indices.values() if gi1 is not None)
        expected_arm = 12 if self.side == "both" else 6
        logger.info(f"  机械臂关节数: {n_arm} (full_robot 单侧应为 6, 双手应为 12, gripper_only 应为 0)")
        logger.info(f"  夹爪关节数: {n_gripper}")
        if self.mode == "full_robot" and n_arm == 0:
            logger.error("  ✗ 严重错误: full_robot 模式下臂关节数为 0! URDF 转换失败!")
        elif self.mode == "full_robot" and n_arm == expected_arm:
            logger.info(f"  ✓ full_robot 臂关节正确加载 ({n_arm} 个 revolute)")
        elif self.mode == "gripper_only" and n_arm == 0:
            logger.info("  ✓ gripper_only 无臂关节 (正确)")

        # 2. qpos 范围验证
        if len(qpos_log) > 0:
            qpos_arr = np.array(qpos_log)
            logger.info(f"  qpos shape: {qpos_arr.shape}")
            if self.mode == "full_robot" and n_arm == expected_arm:
                arm_qpos = qpos_arr[:, self.robot_info["arm_joint_indices"]]
                logger.info(f"  臂关节 qpos 范围: min={arm_qpos.min(axis=0).round(3)}, "
                            f"max={arm_qpos.max(axis=0).round(3)}")
                # 验证臂关节有运动 (不是全固定)
                arm_range = arm_qpos.max(axis=0) - arm_qpos.min(axis=0)
                if arm_range.max() > 0.01:
                    logger.info(f"  ✓ 臂关节有运动 (最大范围={arm_range.max():.3f} rad)")
                else:
                    logger.warning(f"  ⚠ 臂关节几乎无运动 (最大范围={arm_range.max():.3f} rad)")

            # 夹爪 qpos 范围 (遍历所有侧, 验证夹爪手指开合运动可见)
            for s, (gidx1, gidx2) in gripper_indices.items():
                if gidx1 is not None and gidx2 is not None:
                    g1 = qpos_arr[:, gidx1]
                    g2 = qpos_arr[:, gidx2]
                    logger.info(f"  [{s}] 夹爪 qpos 范围: finger1=[{g1.min():.4f}, {g1.max():.4f}], "
                                f"finger2=[{g2.min():.4f}, {g2.max():.4f}]")
                    g1_range = float(g1.max() - g1.min())
                    g2_range = float(g2.max() - g2.min())
                    logger.info(f"  [{s}] 夹爪开合幅度: finger1={g1_range*1000:.2f}mm, finger2={g2_range*1000:.2f}mm")
                    if max(g1_range, g2_range) > 0.001:
                        logger.info(f"  ✓ [{s}] 夹爪手指有开合运动 (夹爪操作物体可见)")
                    else:
                        logger.warning(f"  ⚠ [{s}] 夹爪手指几乎无运动")

        # 3. 接触验证
        total_contacts = sum(c["contacts"] for c in contact_log)
        grasp_frames = sum(grasp_states)
        logger.info(f"  总接触帧数: {grasp_frames}/{len(grasp_states)}")
        logger.info(f"  总接触点数: {total_contacts}")
        if grasp_frames > 0:
            logger.info(f"  ✓ 检测到抓取 (接触≥2 的帧数={grasp_frames})")
        else:
            logger.warning("  ⚠ 未检测到稳定抓取 (接触<2)")

        # 4. 物体提升验证
        if len(obj_pos_log) > 0 and obj_initial_pos:
            logger.info("  物体位置变化 (参数级验证):")
            for actor_name in obj_initial_pos:
                init_pos = obj_initial_pos[actor_name]
                final_pos = np.array(obj_pos_log[-1][actor_name])
                lift = final_pos[2] - init_pos[2]  # Z 轴提升
                xy_drift = np.linalg.norm(final_pos[:2] - init_pos[:2])
                logger.info(f"    {actor_name}: lift={lift*100:.2f}cm, xy_drift={xy_drift*100:.2f}cm")
                if lift > 0.01:
                    logger.info(f"    ✓ {actor_name} 被提升 {lift*100:.2f}cm (抓取成功)")

        # 5. 综合抓取判定 (用户要求: 接触 + 跟随 + 提升 三项都满足)
        grasp_quality = self._evaluate_grasp_quality(
            contact_log, obj_pos_log, obj_initial_pos, gripper_pos_log
        )
        verify_log = {
            "mode": self.mode,
            "side": self.side,
            "n_arm_joints": n_arm,
            "n_gripper_joints": n_gripper,
            "total_frames": len(qpos_log),
            "grasp_frames": grasp_frames,
            "total_contacts": total_contacts,
            "obj_initial_pos": {k: v.tolist() for k, v in obj_initial_pos.items()},
            "obj_final_pos": obj_pos_log[-1] if obj_pos_log else {},
            "grasp_quality": grasp_quality,
        }

        # 6. 抓取控制器摘要 (adaptive / hybrid 模式)
        if self.grasp_mode in ("adaptive", "hybrid") and self.grasp_controllers is not None:
            mode_label = "混合" if self.grasp_mode == "hybrid" else "自适应"
            logger.info(f"  {mode_label}抓取控制器摘要:")
            grasp_summaries = {}
            for s, ctrl in self.grasp_controllers.items():
                summary = ctrl.summary()
                grasp_summaries[s] = summary
                logger.info(f"    [{s}] 抓取次数: {summary['grasp_count']}, "
                            f"最终相位: {summary['final_phase']}, 事件数: {len(summary['events'])}")
                if "max_force" in summary:
                    logger.info(f"    [{s}] 最大夹紧力: {summary['max_force']:.1f}N, "
                                f"平均: {summary['mean_force']:.1f}N")
                for ev in summary["events"]:
                    force_str = f", force={ev['force']}N" if "force" in ev else ""
                    logger.info(f"      F{ev['frame']}: {ev['phase']} (curl={ev['curl']}, "
                                f"obj={ev['obj']}@{ev['dist']}m{force_str})")
            verify_log["grasp_mode"] = self.grasp_mode
            verify_log["grasp_summaries"] = grasp_summaries

        # 轨迹跟踪误差统计 (物理输出 vs 真实 MANO 期望)
        # 用户: "夹爪运动要物理和真实输出的误差来判断准不准确, 而不是只有一个开合的判断"
        if track_log:
            valid = [t for t in track_log if t is not None]
            if valid:
                root_errs = [t['root_err_mm'] for t in valid]
                j1_errs = [t['j1_err_mm'] for t in valid]
                j2_errs = [t['j2_err_mm'] for t in valid]
                track_err = {
                    "n_frames": len(valid),
                    "root_err_mean_mm": float(np.mean(root_errs)),
                    "root_err_max_mm": float(np.max(root_errs)),
                    "finger1_err_mean_mm": float(np.mean(j1_errs)),
                    "finger1_err_max_mm": float(np.max(j1_errs)),
                    "finger2_err_mean_mm": float(np.mean(j2_errs)),
                    "finger2_err_max_mm": float(np.max(j2_errs)),
                }
                verify_log["track_error"] = track_err
                logger.info(f"  轨迹跟踪误差 (物理输出 vs MANO 期望, {len(valid)}帧):")
                logger.info(f"    根位置: mean={track_err['root_err_mean_mm']:.2f}mm, "
                            f"max={track_err['root_err_max_mm']:.2f}mm")
                logger.info(f"    手指1:  mean={track_err['finger1_err_mean_mm']:.2f}mm, "
                            f"max={track_err['finger1_err_max_mm']:.2f}mm")
                logger.info(f"    手指2:  mean={track_err['finger2_err_mean_mm']:.2f}mm, "
                            f"max={track_err['finger2_err_max_mm']:.2f}mm")
                # 误差越小越准确 (root>10mm 或 finger>5mm 说明跟踪有偏差)
                root_ok = track_err['root_err_mean_mm'] < 10.0
                finger_ok = max(track_err['finger1_err_mean_mm'], track_err['finger2_err_mean_mm']) < 5.0
                logger.info(f"    {'✓ 根跟踪准确' if root_ok else '⚠ 根跟踪有偏差'} "
                            f"(<10mm), {'✓ 手指跟踪准确' if finger_ok else '⚠ 手指跟踪有偏差'} (<5mm)")

        verify_path = str(self.output_dir / f"grasp_{self.mode}_{self.side}_verify.json")
        with open(verify_path, "w") as f:
            json.dump(verify_log, f, indent=2, default=str)
        logger.info(f"  验证日志: {verify_path}")

    def _evaluate_grasp_quality(self, contact_log, obj_pos_log, obj_initial_pos, gripper_pos_log):
        """综合抓取质量评估 (第十三轮重设计: 修复 3 个误报 bug)

        旧版 bug:
          1. 接触: 全局 n_contacts>=2 (不区分物体), glb_1 的"61帧"可能来自其他物体
          2. 提升: 用 max(z)-init_z (托起又掉也算成功)
          3. 跟随: 不检查 z 稳定性 (娃娃机式托住也通过)

        新判据 (末段 10 帧, 全部满足才算真正夹住):
          1. 末段 per-object 接触 >= 2 (双指接触同一物体, 80% 帧)
          2. 末段 z 方差 < 1cm (稳定被夹, 不是掉落中)
          3. final_z - init_z > 3cm (最终被抬起, 不是 max(z) 然后掉回)
          4. 跟随: 末段 (物体-夹爪) 相对位置稳定 (保留旧逻辑)

        Returns:
            dict: 每个物体的 {contact_pass, z_stable_pass, lift_pass, follow_pass, grasp_pass, details}
        """
        logger.info("=" * 60)
        logger.info("Step 7b: 综合抓取判定 (末段 per-object 接触 + z 稳定 + final 提升 + 跟随)")
        logger.info("=" * 60)

        LAST_N = 10  # 末段 10 帧判定
        LIFT_THRESHOLD = 0.03       # final_z - init_z > 3cm
        Z_VAR_THRESHOLD = 1e-4      # 末段 z 方差 < 1cm (0.01^2)
        FOLLOW_THRESHOLD = 0.05     # 跟随: 相对位置变化 < 5cm
        MIN_FOLLOW_FRAMES = 8       # 跟随: 末段 10 帧中 >= 8 帧稳定

        n_frames = len(contact_log)
        quality = {}

        # 末段起始帧
        last_start = max(0, n_frames - LAST_N)

        # 2+3+4. 每个物体独立判定 (接触 per-object, z 稳定, final 提升, 跟随)
        for actor_name in obj_initial_pos:
            init_pos = obj_initial_pos[actor_name]

            # 1. 末段 per-object 接触: 末段 10 帧中 per_obj[name]["n"] >= 2 的比例 >= 80%
            contact_pass = False
            last_contact_count = 0
            for fi in range(last_start, n_frames):
                per_obj = contact_log[fi].get("per_obj", {})
                obj_contact = per_obj.get(actor_name, {"n": 0})
                if obj_contact["n"] >= 2:
                    last_contact_count += 1
            contact_pass = last_contact_count >= int(LAST_N * 0.8)

            # 2. 末段 z 方差 (排除"托起又掉")
            z_stable_pass = False
            last_zs = []
            for fi in range(last_start, n_frames):
                if actor_name in obj_pos_log[fi]:
                    last_zs.append(float(np.array(obj_pos_log[fi][actor_name])[2]))
            z_var = float(np.var(last_zs)) if len(last_zs) > 0 else float('inf')
            z_stable_pass = z_var < Z_VAR_THRESHOLD

            # 3. final 提升 (不是 max): final_z - init_z > 3cm
            lift_pass = False
            final_z = float(np.array(obj_pos_log[-1][actor_name])[2]) if actor_name in obj_pos_log[-1] else init_pos[2]
            final_lift = final_z - init_pos[2]
            lift_pass = final_lift > LIFT_THRESHOLD

            # 4. 跟随: 末段 (物体-夹爪) 相对位置稳定
            follow_pass = False
            max_follow_streak = 0
            cur_follow_streak = 0
            if gripper_pos_log and n_frames > 0:
                nearest_side = None
                min_dist = float('inf')
                for s, gp in gripper_pos_log[0].items():
                    d = float(np.linalg.norm(np.array(gp) - init_pos))
                    if d < min_dist:
                        min_dist = d
                        nearest_side = s
                if nearest_side is not None:
                    prev_rel = None
                    for fi in range(last_start, n_frames):
                        if (actor_name not in obj_pos_log[fi]
                                or nearest_side not in gripper_pos_log[fi]):
                            cur_follow_streak = 0
                            prev_rel = None
                            continue
                        obj_p = np.array(obj_pos_log[fi][actor_name])
                        grp_p = np.array(gripper_pos_log[fi][nearest_side])
                        rel_p = obj_p - grp_p
                        if prev_rel is not None:
                            delta = float(np.linalg.norm(rel_p - prev_rel))
                            if delta < FOLLOW_THRESHOLD:
                                cur_follow_streak += 1
                                max_follow_streak = max(max_follow_streak, cur_follow_streak)
                            else:
                                cur_follow_streak = 0
                        prev_rel = rel_p
            follow_pass = max_follow_streak >= MIN_FOLLOW_FRAMES

            grasp_pass = contact_pass and z_stable_pass and lift_pass and follow_pass
            quality[actor_name] = {
                "contact_pass": contact_pass,
                "z_stable_pass": z_stable_pass,
                "lift_pass": lift_pass,
                "follow_pass": follow_pass,
                "grasp_pass": grasp_pass,
                "last_contact_count": last_contact_count,
                "last_z_var_cm": z_var * 100,
                "final_lift_cm": final_lift * 100,
                "max_follow_streak": max_follow_streak,
            }
            tag = "✓ 真正夹住" if grasp_pass else "✗ 未真正夹住"
            logger.info(f"  [{actor_name}] {tag}")
            logger.info(f"    末段接触: {last_contact_count}/{LAST_N} 帧 per-object>=2 "
                        f"(需>={int(LAST_N*0.8)}) → {'✓' if contact_pass else '✗'}")
            logger.info(f"    末段z方差: {z_var*100:.3f}cm (需<{Z_VAR_THRESHOLD*100:.3f}cm) → "
                        f"{'✓' if z_stable_pass else '✗'}")
            logger.info(f"    final提升: {final_lift*100:.2f}cm (需>{LIFT_THRESHOLD*100:.0f}cm) → "
                        f"{'✓' if lift_pass else '✗'}  [init_z={init_pos[2]:.3f}, final_z={final_z:.3f}]")
            logger.info(f"    末段跟随: {max_follow_streak}/{LAST_N} (需>={MIN_FOLLOW_FRAMES}) → "
                        f"{'✓' if follow_pass else '✗'}")

        n_grasped = sum(1 for q in quality.values() if q["grasp_pass"])
        logger.info(f"  >>> 真正夹住物体数: {n_grasped}/{len(quality)}")
        return quality

    def run_v4_pipeline(self, side=None):
        """v4.4 完整流水线: Stage 1 → Stage 2 → Stage 3

        要求 scene/robot/MANO 轨迹/控制器已初始化 (由 run() 前置步骤完成).
        """
        from traj_optimize import compute_frame_params
        from scipy.spatial.transform import Rotation as R_scipy

        if self.side == "both":
            logger.error("[v4 pipeline] 暂不支持 both 模式, 请指定 --side right/left")
            return
        side = side or self.side

        traj = self._mano_gripper_traj.get(side)
        if traj is None or len(traj["pos"]) == 0:
            logger.error("[v4 pipeline] 无 MANO 轨迹")
            return

        # 确保物体初始位姿已记录
        if not hasattr(self, '_obj_initial_poses') or self._obj_initial_poses is None:
            self._obj_initial_poses = {actor.get_name(): actor.get_pose() for actor in self.obj_actors}

        N = len(traj["pos"])
        self._frame_params = compute_frame_params(N)
        fp = self._frame_params
        F50 = fp['F50_IDX']
        logger.info(f"[v4 pipeline] 总帧数={N}, F50={F50}")

        # Stage 1 需要 MANO F50 对齐到目标物体的世界坐标
        self._compute_neutral_offsets()

        # Stage 1: 6DOF 抓取姿态 + 夹持力策略优化
        # v4.5: Stage 1 的短仿真基于 gripper_only (虚拟关节 6DOF),
        # 与 full_robot 臂 IK 解耦, 确保"给定 base 位姿能否夹住"被正确评估.
        logger.info("=" * 60)
        logger.info("[v4 pipeline] Stage 1: 6DOF 抓取姿态优化 (gripper_only)")
        logger.info("=" * 60)
        stage1_sim = GraspSimulator(
            hawor_dir=self.hawor_dir,
            ras_dir=self.ras_dir,
            mode="gripper_only",
            side=side,
            output_dir=str(self.output_dir / "stage1"),
            num_frames=self.num_frames,
            start_frame=self.start_frame,
            views="none",  # Stage 1 子实例不需要渲染
            grasp_mode="hybrid",
            viewer=False,
        )
        stage1_sim.run(run_main_loop=False)
        stage1_sim._compute_neutral_offsets()
        best_grasp = stage1_sim.cem_grasp_pose_optimize(side, n_iterations=15, population_size=32)
        if best_grasp is None:
            logger.error("[v4 pipeline] Stage 1 失败")
            return
        # 把 Stage 1 结果同步到当前 full_robot simulator
        self._mano_neutral_offset = getattr(stage1_sim, '_mano_neutral_offset', self._mano_neutral_offset)
        self._mano_grasp_frame = getattr(stage1_sim, '_mano_grasp_frame', self._mano_grasp_frame)
        self._v4_best_grasp_stage1 = best_grasp
        logger.info(f"[v4 pipeline] Stage 1 最优 grasp: pos={best_grasp['pos'].round(4)}, "
                    f"euler={np.degrees(best_grasp['euler']).round(2)}, "
                    f"gripper_qpos={best_grasp['gripper_qpos']:.4f}")

        # Stage 2: 轨迹重建 + MANO x,y 位置参考
        logger.info("=" * 60)
        logger.info("[v4 pipeline] Stage 2: Minimum Jerk 轨迹重建 + MANO x,y 优化")
        logger.info("=" * 60)
        recon = self.reconstruct_trajectory(best_grasp, side, F50_IDX=F50, N_TRANS=4)
        if recon is None:
            logger.error("[v4 pipeline] Stage 2 失败")
            return

        # 将 Stage 2 结果转换为 _fixed_offsets_654 中的偏移
        pos_traj = np.asarray(traj["pos"])
        R_traj = np.asarray(traj["R"])
        fixed_offsets = {0: np.zeros(6)}
        for f in recon['frames']:
            off = np.zeros(6)
            off[:3] = recon['pos'][f] - pos_traj[f]
            R_corr = recon['R'][f] @ R_traj[f].T
            if np.linalg.norm(R_corr - np.eye(3)) > 1e-8:
                off[3:6] = R_scipy.from_matrix(R_corr).as_euler('xyz')
            fixed_offsets[f] = off
        # F90~F112 完全跟随 MANO (plan: F91~F112 跟随, F90 作为边界也固定)
        F90 = 90 if N > 90 else N - 1
        for f in range(F90, N):
            fixed_offsets[f] = np.zeros(6)

        # 计算 grasp_offset (用于 init_params, 不放入 fixed_offsets)
        f_start_s2 = min(recon['frames']) if recon['frames'] else 46
        f_end_s2 = max(recon['frames']) if recon['frames'] else 54
        _ABOVE_Z = 0.03  # 接近期额外 z 偏移
        _grasp_pos = np.asarray(best_grasp['pos'], dtype=np.float64)
        _grasp_offset = _grasp_pos - pos_traj[F50]
        logger.info(f"[v4 pipeline] grasp_offset={_grasp_offset.round(4)}")

        self._fixed_offsets_654 = fixed_offsets

        # 同步更新 _frame_params: 把 Stage 2 输出 + F90~F112 加入固定帧,
        # 这样 Stage 3 只优化 F1~F45 和 F55~F89
        new_fixed = sorted(set(fp['fixed_frames']) | set(recon['frames']) | set(range(F90, N)))
        fp['fixed_frames'] = new_fixed
        fp['n_optimized'] = N - len(new_fixed)
        fp['params_dim'] = fp['n_optimized'] * 6
        from traj_optimize import POS_RANGE, ROT_RANGE
        fp['param_range'] = np.tile(
            np.array([[-POS_RANGE, POS_RANGE],
                      [-POS_RANGE, POS_RANGE],
                      [-POS_RANGE, POS_RANGE],
                      [-ROT_RANGE, ROT_RANGE],
                      [-ROT_RANGE, ROT_RANGE],
                      [-ROT_RANGE, ROT_RANGE]]),
            (fp['n_optimized'], 1)
        )
        self._v4_stage2_recon = recon
        self._v4_best_grasp = best_grasp

        logger.info("[v4 pipeline] Stage 2 输出已固定:")
        for f in sorted(fixed_offsets.keys()):
            logger.info(f"  F{f}: xyz={fixed_offsets[f][:3].round(4)}, "
                        f"rpy={fixed_offsets[f][3:6].round(4)}")

        # Stage 3: 全局轨迹优化
        logger.info("=" * 60)
        logger.info("[v4 pipeline] Stage 3: 全局轨迹优化")
        logger.info("=" * 60)
        stage3_result = self.cem_stage3_optimize(side, n_iterations=10, population_size=12)
        if stage3_result is None:
            logger.warning("[v4 pipeline] Stage 3 跳过")
            best_stage3_params = np.zeros(0)
            best_stage3_reward = -float('inf')
        else:
            best_stage3_params, best_stage3_reward = stage3_result

        # 构造完整 654D 参数
        from traj_optimize import generate_trajectory_from_params
        full_params = np.zeros(fp['n_optimized'] * 6, dtype=np.float64)
        opt_idx = 0
        for f in range(N):
            if f in fixed_offsets:
                continue
            if opt_idx * 6 < len(best_stage3_params):
                full_params[opt_idx * 6:(opt_idx + 1) * 6] = best_stage3_params[opt_idx * 6:(opt_idx + 1) * 6]
            opt_idx += 1

        self._opt_params_best = full_params
        self._opt_mano_gripper_traj = self._mano_gripper_traj
        np.save(self.output_dir / "v4_opt_params.npy", full_params)
        np.save(self.output_dir / "v4_fixed_offsets.npy",
                {str(k): v for k, v in fixed_offsets.items()}, allow_pickle=True)
        np.save(self.output_dir / "frame_params.npy", fp)
        logger.info(f"[v4 pipeline] 完成. Stage3 reward={best_stage3_reward:.3f}, "
                    f"params shape={full_params.shape}")

        # 设置参数供 run() 主循环回放
        self.set_opt_params(full_params)
        logger.info("[v4 pipeline] 转入主仿真循环回放最优轨迹")

    # ============================================================
    # 第十八轮: CEM 优化支持
    # ============================================================
    def set_opt_params(self, opt_params):
        """设置优化参数 (第十八轮 CEM 优化 / 第二十六轮窗口优化)

        Args:
            opt_params: 9 维 / 42 维 / 窗口维 np.ndarray
        """
        import numpy as np
        if opt_params is not None:
            self._opt_params = opt_params
            logger.info(f"  设置优化参数: shape={opt_params.shape}, norm={np.linalg.norm(opt_params):.4f}")
            # 第二十六轮: 如果参数维度不是 9/42, 尝试加载窗口参数
            if len(opt_params) not in (9, 42):
                try:
                    wp_path = self.output_dir / "window_frames.npy"
                    if wp_path.exists():
                        from traj_optimize import build_window_params
                        grasp_window = np.load(wp_path)
                        n_total = self.num_frames if self.num_frames > 0 else len(grasp_window)
                        self._window_params = build_window_params(grasp_window, n_total)
                        logger.info(f"  加载窗口参数: {len(grasp_window)} frames [{grasp_window[0]}, {grasp_window[-1]}]")
                    else:
                        self._window_params = None
                except Exception as e:
                    logger.warning(f"  加载窗口参数失败: {e}")
                    self._window_params = None
        else:
            self._opt_params = None
            self._window_params = None

    def run_optimize(self):
        """离线 CEM 轨迹优化 (借鉴 do-as-i-do Stage 5)

        在 SAPIEN 中 rollout 多条候选参数, 评估抓取质量, CEM 迭代优化.
        最优参数保存到 output_dir/opt_params.npy, 存储在 self._opt_params_best.
        """
        from traj_optimize import cem_optimize, DEFAULT_PARAMS, PARAM_RANGE, REWARD_WEIGHTS

        # 1. 完整初始化 simulator (对齐 run() 的前 5 步)
        self._align_scene()
        # 加载数据
        if self.side == "both":
            raise ValueError("--optimize 仅支持单侧 (gripper_only)")
        hawor_data = load_hawor_data(self.hawor_dir, hand_idx=self.hand_idx)
        n_total = len(hawor_data["pred_trans"])
        if self.num_frames < 0 or self.num_frames > n_total - self.start_frame:
            self.num_frames = n_total - self.start_frame
        betas_mean = hawor_data["pred_betas"][self.start_frame].astype(np.float32)
        mano_side = "left" if self.hand_idx == 0 else "right"
        from mano_layer import MANOLayer
        mano_layer = MANOLayer(mano_side, betas_mean)
        R_c2w_all, t_c2w_all = load_hawor_c2w(self.hawor_dir)

        # 预扫描 GLB 地面高度
        glb_path = self.ras_dir / "final_scene.glb"
        ground_z = 0.0
        if glb_path.exists() and self.transform_params_path:
            ground_z = compute_glb_ground_z(glb_path, self.transform_params_path)
        # 第十八轮: 优化模式强制 CPU 场景, 避免渲染初始化段错误
        self.scene = setup_physics_scene(ground_height=ground_z, force_cpu=True)
        render_available = getattr(self.scene, "_render_available", False)

        # 加载 GLB 物体
        self.obj_actors, _, self.obj_bbox_centers, self.obj_info = load_glb_with_physics(
            glb_path, self.transform_params_path, self.scene, fast_collision=True
        )

        # 初始化 hybrid 控制器 (含 target_obj 和 bowl_obj)
        wrist_trans = np.asarray(hawor_data["pred_trans"])
        target_obj = find_target_object_by_trajectory(wrist_trans, self.obj_bbox_centers)
        pink_obj = find_pink_object(self.obj_info)
        bowl_obj = find_bowl(self.obj_info, exclude_names=[pink_obj] if pink_obj else None)
        self.grasp_controllers = {
            self.side: HybridGraspController(
                self.obj_actors, side=self.side, scene=self.scene,
                target_obj=pink_obj or target_obj,
                obj_positions=self.obj_bbox_centers,
                bowl_obj=bowl_obj,
            )
        }

        # 计算手腕位置 + MANO 夹爪轨迹
        # 使用正确的坐标变换 (与 002_render_scene.py _render_to_sapien 一致)
        def _mano_to_sapien_v4(pts_slam):
            """MANO SLAM → SAPIEN (与 002 链一致)
            p_glb = s * R_h2g @ Rx_hand @ p_slam + t_h2g
            p_sapien = R_AXIS @ p_glb
            """
            s = self._mano_xform_s
            R_hand = self._mano_xform_R_hand  # = R_h2g @ Rx_hand
            t = self._mano_xform_t
            pts_glb = s * (R_hand @ pts_slam.T).T + t
            return (R_AXIS @ pts_glb.T).T
        mano_gripper_traj = {}
        wrist_positions = []
        for fi in range(self.start_frame, self.start_frame + self.num_frames):
            if fi >= len(hawor_data["pred_trans"]):
                break
            if fi < len(hawor_data["pred_valid"]) and not hawor_data["pred_valid"][fi]:
                continue
            try:
                _, joints = compute_mano_joints(
                    mano_layer, hawor_data["pred_rot"][fi],
                    hawor_data["pred_hand_pose"][fi],
                    hawor_data["pred_trans"][fi],
                )
                joints_sapien = _mano_to_sapien_v4(joints)
                wrist_positions.append(joints_sapien[0, :3])
                s = self.side
                if s not in mano_gripper_traj:
                    mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": []}
                root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
                    joints_sapien[0, :3], joints_sapien[4, :3],
                    joints_sapien[8, :3], prefix=s,
                )
                mano_gripper_traj[s]["pos"].append(root_pos)
                mano_gripper_traj[s]["R"].append(root_R)
                mano_gripper_traj[s]["j1"].append(j1)
                mano_gripper_traj[s]["j2"].append(j2)
            except Exception:
                continue
        for key in mano_gripper_traj:
            mano_gripper_traj[key]["pos"] = np.array(mano_gripper_traj[key]["pos"], dtype=np.float64)
            mano_gripper_traj[key]["R"] = np.array(mano_gripper_traj[key]["R"], dtype=np.float64)
            mano_gripper_traj[key]["j1"] = np.array(mano_gripper_traj[key]["j1"], dtype=np.float64)
            mano_gripper_traj[key]["j2"] = np.array(mano_gripper_traj[key]["j2"], dtype=np.float64)
        self._mano_gripper_traj = mano_gripper_traj

        # 基座位置
        if wrist_positions:
            self._base_pos = wrist_positions[0]
            self._base_quat = pr.quaternion_from_axis_angle(np.array([0, 0, 1, 0]))
        else:
            self._base_pos = np.array([0.0, 0.0, 0.3])
            self._base_quat = pr.quaternion_from_axis_angle(np.array([0, 0, 1, 0]))

        # 加载机器人
        self.robot_info = setup_robot(self.scene, self.mode, self.side, self._base_pos, self._base_quat)

        # 计算 offset
        self._compute_neutral_offsets()

        # 第十九轮: 保存物体初始位姿 (用于每次 rollout 重置)
        self._obj_initial_poses = {}
        for actor in self.obj_actors:
            self._obj_initial_poses[actor.get_name()] = actor.get_pose()

        # 2. 定义 rollout 评估函数
        def rollout_single(opt_params, decimation=None):
            """单条 rollout: 重置场景状态 → 跑完整轨迹 → 评估

            Args:
                opt_params: 优化参数数组
                decimation: 物理子步数, None 时使用全局 DECIMATION (=8)
            """
            # 第十九轮: 设置 opt_params 供 _step_gripper_only → _compute_mano_neutral_target 使用
            self._opt_params = np.asarray(opt_params, dtype=np.float64)
            # 清除缓存 (新参数需要重新计算)
            for cache_attr in ('_kf_cache', '_traj_654_cache', '_window_params'):
                if hasattr(self, cache_attr):
                    delattr(self, cache_attr)
            # === CEM 路径修复: 将 CEM 3D/6D 参数扩展为 654 路径参数 ===
            # 根因: CEM rollout 中 _frame_params/_fixed_offsets_654 未设置,
            #        导致 _compute_mano_neutral_target 走旧路径 (全局偏移, 无三段划分).
            #        CEM 优化基于错误位置评分, 优化结果不准.
            # 修复: 将 CEM 小维度参数映射为 F50 的偏移, 构造 654 路径参数, 走三段划分.
            _cem_dim = len(self._opt_params)
            # 动态窗口: 仅当维度匹配 _grasp_window_meta['dim'] 时才走动态窗口路径
            # 修复: 之前 _is_grasp_window 判断太宽泛, 648D 全局参数也会匹配, 导致覆盖 _fixed_offsets_654
            _gw_meta_check = getattr(self, '_grasp_window_meta', None)
            _is_grasp_window = (
                _gw_meta_check is not None
                and _cem_dim == _gw_meta_check['dim']
                and _cem_dim % 6 == 0
            )
            if _cem_dim in (3, 6) or _is_grasp_window:
                from traj_optimize import compute_frame_params
                _n_frames = len(self._mano_gripper_traj.get(self.side, {}).get("pos", []))
                if _n_frames > 0:
                    _fp = compute_frame_params(_n_frames)
                    self._frame_params = _fp
                    _F50 = _fp['F50_IDX']
                    _neutral_off = getattr(self, '_mano_neutral_offset', {}).get(self.side)
                    # 动态窗口大小 (从 _grasp_window_meta 读取, 由 Stage 1 设置)
                    _gw_meta = getattr(self, '_grasp_window_meta', None)
                    if _gw_meta is not None:
                        _N_TRANS = _gw_meta['n_trans']
                        _GRASP_FRAMES = _gw_meta['frames']
                        _F50_LOCAL = _gw_meta['f50_local_idx']
                    else:
                        _N_TRANS = 2
                        _GRASP_FRAMES = list(range(_F50 - _N_TRANS, _F50 + _N_TRANS + 1))
                        _F50_LOCAL = _N_TRANS

                    if _is_grasp_window:
                        # 动态窗口: N_TRANS*2+1 帧 × 6DOF, 每帧独立偏移
                        # 修复双重叠加: CEM 参数已是完整偏移, 不再叠加 _neutral_off
                        _n_window = len(_GRASP_FRAMES)
                        _expected_dim = _n_window * 6
                        if _cem_dim != _expected_dim:
                            logger.warning(f"  [CEM路径] 维度不匹配: cem_dim={_cem_dim}, expected={_expected_dim}, "
                                          f"回退到固定 5 帧窗口")
                            _N_TRANS = 2
                            _GRASP_FRAMES = list(range(_F50 - _N_TRANS, _F50 + _N_TRANS + 1))
                            _F50_LOCAL = _N_TRANS
                            _n_window = len(_GRASP_FRAMES)
                        _grasp_frame_offsets = {}
                        for _gi in range(_n_window):
                            _frame_idx = _GRASP_FRAMES[_gi]
                            _off = self._opt_params[_gi*6:(_gi+1)*6].copy()
                            # 不再叠加 _neutral_off: CEM 参数直接是完整偏移 (从 MANO 出发)
                            _grasp_frame_offsets[_frame_idx] = _off
                        # w_track 动态惩罚: F50 不惩罚, 距离 F50 越近的帧允许偏离越大
                        # 系数从 F50 向外线性递增: 紧邻=0.3, 最远=1.0
                        # 关键: 姿态惩罚不在此处, 由 result 中的 track_pen 只算位置
                        _gw_meta = getattr(self, '_grasp_window_meta', None)
                        _track_pen = 0.0
                        if _gw_meta is not None and _gw_meta.get('n_trans', 0) > 0:
                            _nt = _gw_meta['n_trans']
                            # 从 F50 向外: 距离 1, 2, ..., _nt
                            # 系数: 0.3 + 0.7 * (dist / _nt), 范围 [0.3, 1.0]
                            _trans_pen_sum = 0.0
                            _trans_count = 0
                            for _fi, _off in _grasp_frame_offsets.items():
                                if _fi == _F50:
                                    continue
                                _dist_to_f50 = abs(_fi - _F50)
                                _weight = 0.3 + 0.7 * (_dist_to_f50 / max(_nt, 1))
                                _trans_pen_sum += _weight * float(np.linalg.norm(_off[:3]))
                                _trans_count += 1
                            _track_pen = _trans_pen_sum / max(_trans_count, 1)
                        else:
                            _trans_offsets = [
                                np.linalg.norm(_off[:3])
                                for _fi, _off in _grasp_frame_offsets.items()
                                if _fi != _F50
                            ]
                            _track_pen = float(np.mean(_trans_offsets)) if _trans_offsets else 0.0
                        self._current_track_pen = _track_pen
                    elif _cem_dim == 6:
                        # 6DOF: F50 单帧偏移
                        _grasp_offset = np.zeros(6)
                        if _neutral_off is not None:
                            _grasp_offset[:3] = _neutral_off[:3]
                        _grasp_offset[:_cem_dim] += self._opt_params[:_cem_dim]
                        _grasp_frame_offsets = {_F50: _grasp_offset}
                    else:
                        # 3DOF: F50 单帧位置偏移
                        _grasp_offset = np.zeros(6)
                        if _neutral_off is not None:
                            _grasp_offset[:3] = _neutral_off[:3]
                        _grasp_offset[:_cem_dim] += self._opt_params[:_cem_dim]
                        _grasp_frame_offsets = {_F50: _grasp_offset}

                    # 提升偏移: F95/F112 增加 z 提升
                    _LIFT_Z = 0.15
                    _f95_off = np.zeros(6); _f95_off[2] = _LIFT_Z
                    _f112_off = np.zeros(6); _f112_off[2] = _LIFT_Z

                    # 固定帧: F0=0, 抓取窗口帧, F95/F112=提升
                    _fixed_offsets = {0: np.zeros(6)}
                    _fixed_offsets.update(_grasp_frame_offsets)
                    _fixed_offsets[_fp['F95_IDX']] = _f95_off
                    _fixed_offsets[_fp['F112_IDX']] = _f112_off
                    self._fixed_offsets_654 = _fixed_offsets
                    # 构造 654 维 opt_params (固定帧之间线性插值)
                    _n_opt = _fp['n_optimized']
                    _full_params = np.zeros(_n_opt * 6, dtype=np.float64)
                    _ff_list = sorted(self._fixed_offsets_654.keys())
                    # 关键: opt_idx 计算必须用 _fixed_offsets_654 的所有帧 (不是 _fp['fixed_frames'])
                    # 否则 F55+ 的偏移会写到错误的位置
                    _all_fixed_sorted = _ff_list  # [0, 46, 47, ..., 54, 94, 111]
                    for _si in range(len(_ff_list) - 1):
                        _f_s = _ff_list[_si]
                        _f_e = _ff_list[_si + 1]
                        _off_s = self._fixed_offsets_654[_f_s]
                        _off_e = self._fixed_offsets_654[_f_e]
                        for _fi in range(_f_s + 1, _f_e):
                            if _fi in self._fixed_offsets_654:
                                continue
                            _t = (_fi - _f_s) / max(_f_e - _f_s, 1)
                            # opt_idx: 从所有固定帧中减去 (用 _all_fixed_sorted 而不是 _fp['fixed_frames'])
                            _oi = _fi - sum(1 for _f in _all_fixed_sorted if _f < _fi)
                            if 0 <= _oi < _n_opt:
                                _full_params[_oi * 6:(_oi + 1) * 6] = (1 - _t) * _off_s + _t * _off_e
                    self._opt_params = _full_params
                    # 清除缓存 (参数变了)
                    if hasattr(self, '_traj_654_cache'):
                        delattr(self, '_traj_654_cache')
            # 第十九轮修复: 重置帧间状态变量 (否则前一 rollout 的末位置会导致第一帧速度异常)
            for attr in ('_prev_demo_root_pos', '_prev_root_pos', '_prev_root_pos_vel', '_close_entered'):
                if hasattr(self, attr):
                    delattr(self, attr)
            # 不重建 scene (SAPIEN 多次创建 scene 会段错误)
            # 只需重置 robot 根位姿和关节位置
            robot = self.robot_info["robot"]
            # 第十九轮修复: 重置夹爪 qpos 到初始张开状态 (前一 rollout 闭合的手指会影响下一次)
            robot.set_qpos(self.robot_info["init_qpos"].copy())
            # === BUG 修复 (opt-sim-divergence): set_qpos 不重置 qvel ===
            # 根因: set_qpos 只设置位置, 不重置速度. 前一 rollout 结束时手指在快速运动 (闭合),
            # qvel 残留到下一次 rollout 第一帧, 导致 PD 力 = Kp*(target-qpos) - Kd*qvel 不同.
            # run() 有 30 帧 WARMUP 让 qvel 衰减, rollout_single 没有 → 优化 vs 最终仿真物理结果不一致.
            # 修复: 显式 set_qvel(零), 对齐 WARMUP 后的 qvel≈0 状态.
            robot.set_qvel(np.zeros_like(robot.get_qvel()))
            # === BUG 修复结束 ===
            # 虚拟关节模式下, root (world) 应在原点, 位置由虚拟关节编码.
            # set_root_pose(_base_pos) 会导致 EE 位置 = root偏移 + 虚拟关节值 = 双倍偏移.
            if self.robot_info.get("virtual_idx"):
                robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
            else:
                robot.set_root_pose(sapien.Pose(self._base_pos.tolist(), self._base_quat.tolist()))
            robot.set_root_linear_velocity([0, 0, 0])
            robot.set_root_angular_velocity([0, 0, 0])
            # 重置物体到初始位姿 (每次 rollout 公平评估, 必须在 warmup 之前)
            for actor in self.obj_actors:
                name = actor.get_name()
                if name in self._obj_initial_poses:
                    actor.set_pose(self._obj_initial_poses[name])
                    try:
                        for comp in actor.get_components():
                            if hasattr(comp, 'set_linear_velocity'):
                                comp.set_linear_velocity([0, 0, 0])
                                comp.set_angular_velocity([0, 0, 0])
                                break
                    except Exception:
                        pass
            # WARMUP: 5 帧让物理稳定 (物体已重置, 用当前 qpos 保持, 不夹紧)
            _gi1, _gi2 = self.robot_info["gripper_idx1"], self.robot_info["gripper_idx2"]
            _aji = self.robot_info.get("arm_joint_indices", [])
            for _ in range(5):
                _qpos = robot.get_qpos()
                physics_step(
                    robot, _aji, _gi1, _gi2,
                    np.array([]),
                    float(_qpos[_gi1]), float(_qpos[_gi2]),
                    self.scene,
                    lock_root_pose=None,
                )
            wrist_trans_opt = np.asarray(hawor_data["pred_trans"])
            target_obj_opt = find_target_object_by_trajectory(wrist_trans_opt, self.obj_bbox_centers)
            pink_obj_opt = find_pink_object(self.obj_info)
            bowl_obj_opt = find_bowl(self.obj_info, exclude_names=[pink_obj_opt] if pink_obj_opt else None)
            self.grasp_controllers = {self.side: HybridGraspController(
                self.obj_actors, side=self.side, scene=self.scene,
                target_obj=pink_obj_opt or target_obj_opt,
                obj_positions=self.obj_bbox_centers,
                bowl_obj=bowl_obj_opt,
            )}
            self._compute_neutral_offsets()

            # 跑完整轨迹
            # 第十九轮修复: 物体初始 z 必须在 rollout 开始前捕获 (原代码在结束后才赋值, 导致 lift 恒为 0)
            target_obj_for_init = self.grasp_controllers[self.side].target_obj
            obj_init_z = 0.0
            obj_init_pos = None
            obj_init_xy = [0.0, 0.0]
            if target_obj_for_init is not None:
                for actor in self.obj_actors:
                    if target_obj_for_init in actor.get_name():
                        obj_init_pos = np.array(actor.get_pose().p)
                        obj_init_z = float(obj_init_pos[2])
                        obj_init_xy = obj_init_pos[:2].tolist()
                        break
            # 第十九轮调试: 检查物体是否被正确重置 (优化 rollout 间物体重置失败会导致奖励虚高)
            if hasattr(self, '_opt_eval_count'):
                self._opt_eval_count += 1
            else:
                self._opt_eval_count = 1
            if self._opt_eval_count <= 5 or self._opt_eval_count % 32 == 0:
                expected_pos = self._obj_initial_poses.get(target_obj_for_init)
                if expected_pos is not None and obj_init_pos is not None:
                    expected_p = np.array(expected_pos.p)
                    drift = np.linalg.norm(obj_init_pos - expected_p)
                    logger.info(f"  [rollout {self._opt_eval_count}] {target_obj_for_init} init_pos={obj_init_pos.round(4)}, "
                                f"expected={expected_p.round(4)}, drift={drift:.4f}m")
            contact_frames_in_close = 0
            max_penetration = 0.0
            min_dist_in_close = 1.0  # 第十九轮 v3: CLOSE 阶段 gripper-物体最近距离 (诊断)
            # 第二十轮: CLOSE 阶段所有帧的 gripper-物体距离列表 (用于计算 avg_dist_in_close_last5)
            _close_dists = []
            close_start = -1
            close_end = -1
            # 第二十一轮: 全程 gripper-物体距离列表 (粗搜接近奖励用)
            _all_dists = []
            _gripper_z_min = float('inf')
            _obj_z_target = None
            # 第二十八轮: CLOSE 阶段接触标志序列 (用于计算末段接触)
            _close_contact_flags = []
            _close_offsets = []  # 物体跟随度量: 每帧相对偏移
            # 第二十六轮: 帧间平滑度
            smoothness_cost = 0.0
            wp = getattr(self, '_window_params', None)
            if wp is not None and len(opt_params) == wp['dim'] and wp['n_window'] > 1:
                M = wp['n_window']
                pos_diff = 0.0
                rot_diff = 0.0
                for wi in range(M - 1):
                    b1 = wi * 6
                    b2 = (wi + 1) * 6
                    pos_diff += np.sum((opt_params[b1:b1+3] - opt_params[b2:b2+3]) ** 2)
                    rot_diff += np.sum((opt_params[b1+3:b1+6] - opt_params[b2+3:b2+6]) ** 2)
                smoothness_cost = (pos_diff + rot_diff * 0.1) / (M - 1)

            # === DEBUG 插桩 (opt-sim-divergence): 记录每帧物体轨迹 ===
            _dbg_obj_z_traj = []
            _dbg_obj_xy_traj = []
            _dbg_contact_traj = []
            _dbg_gripper_z_traj = []
            _dbg_first_frame_state = None
            # === DEBUG 插桩结束 ===

            for local_idx in range(self.num_frames):
                fi = self.start_frame + local_idx
                if fi >= len(hawor_data["pred_trans"]):
                    break
                if fi < len(hawor_data["pred_valid"]) and not hawor_data["pred_valid"][fi]:
                    continue

                self._current_local_idx = local_idx
                self._last_phase = "APPROACH"
                # 第二十九轮: 跳过 MANO FK (预计算位存在 self._mano_gripper_traj 中)
                arm_target, (gripper_t1, gripper_t2) = self._step_gripper_only(np.zeros((21, 3)))

                # 三段力切换 (夹取控制独立于位姿)
                _fp = getattr(self, '_frame_params', None)
                if _fp is not None:
                    phase = getattr(self, '_last_phase', "APPROACH")
                    robot = self.robot_info["robot"]
                    gi1, gi2 = self.robot_info["gripper_idx1"], self.robot_info["gripper_idx2"]
                    active_joints = robot.get_active_joints()
                    if phase == "CLOSE" or (phase == "APPROACH" and gripper_t1 < _DESCEND_OPEN - 1e-6):
                        # 高力闭合 (边走边夹的闭合段也使用高力)
                        for j_idx in [gi1, gi2]:
                            active_joints[j_idx].set_drive_property(
                                stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING, force_limit=GRIPPER_FORCE)
                    elif phase == "RELEASE":
                        # 低力释放
                        for j_idx in [gi1, gi2]:
                            active_joints[j_idx].set_drive_property(
                                stiffness=50, damping=2, force_limit=3.0)
                    else:
                        # APPROACH 中张段: 低力跟随
                        for j_idx in [gi1, gi2]:
                            active_joints[j_idx].set_drive_property(
                                stiffness=100, damping=5, force_limit=5.0)

                # 物理步进 (第十九轮修复: 必须调 physics_step, 否则 set_drive_target 缺失, 手指永远不闭合)
                robot = self.robot_info["robot"]
                arm_joint_indices = self.robot_info.get("arm_joint_indices", [])
                gi1, gi2 = self.robot_info["gripper_idx1"], self.robot_info["gripper_idx2"]
                # v15k: 虚拟关节运动学锁定 (与 run() 方法一致)
                _vlock_rs = None
                _vid_rs = self.robot_info.get("virtual_idx", {})
                _vpos_rs = getattr(self, '_current_gripper_pos', None)
                _vR_rs = getattr(self, '_current_gripper_R', None)
                if _vid_rs and _vpos_rs is not None and _vR_rs is not None:
                    from scipy.spatial.transform import Rotation as _R_rs
                    _rz_rs, _ry_rs, _rx_rs = rotmat_to_zyx_euler(_vR_rs)
                    _vlock_rs = {
                        _vid_rs['vx']: float(_vpos_rs[0]),
                        _vid_rs['vy']: float(_vpos_rs[1]),
                        _vid_rs['vz']: float(_vpos_rs[2]),
                        _vid_rs['rz']: float(_rz_rs),
                        _vid_rs['ry']: float(_ry_rs),
                        _vid_rs['rx']: float(_rx_rs),
                        # v15n4: 手指不加入 vlock, 让 PD 自然处理接触力
                    }
                physics_step(
                    robot, arm_joint_indices, gi1, gi2,
                    np.asarray(arm_target) if len(arm_target) else np.array([]),
                    float(gripper_t1), float(gripper_t2),
                    self.scene,
                    lock_root_pose=None,
                    decimation=decimation,
                    virtual_lock_targets=_vlock_rs,
                )
                # 虚拟关节模式: 不做 post-step 纠正 (对齐 test5)
                # _virtual_idx = self.robot_info.get("virtual_idx", {})
                # (注释掉漂移纠正逻辑)

                # 第十九轮 v3: 提前获取 target_obj (供 CLOSE 检测和接触检测共用)
                target_obj = self.grasp_controllers[self.side].target_obj

                # === DEBUG 插桩 (opt-sim-divergence): 记录每帧状态 ===
                if target_obj is not None:
                    for actor in self.obj_actors:
                        if target_obj in actor.get_name():
                            _o_pos = np.array(actor.get_pose().p)
                            _dbg_obj_z_traj.append(float(_o_pos[2]))
                            _dbg_obj_xy_traj.append(_o_pos[:2].tolist())
                            break
                if hasattr(self, '_current_gripper_pos'):
                    _dbg_gripper_z_traj.append(float(self._current_gripper_pos[2]))
                # 第一帧详细状态 (对比 run() 的第一帧)
                if local_idx == 0 and target_obj is not None:
                    robot_qpos = robot.get_qpos().copy()
                    robot_qvel = robot.get_qvel().copy()
                    for actor in self.obj_actors:
                        if target_obj in actor.get_name():
                            _dbg_first_frame_state = {
                                'robot_qpos': robot_qpos.tolist(),
                                'robot_qvel': robot_qvel.tolist(),
                                'gripper_idx1': self.robot_info["gripper_idx1"],
                                'gripper_idx2': self.robot_info["gripper_idx2"],
                                'obj_pos': _o_pos.tolist(),
                                'root_pose': robot.get_root_pose().p.tolist(),
                            }
                            break
                # === DEBUG 插桩结束 ===

                # 第二十一轮: 全程 gripper-物体距离 (粗搜接近奖励)
                if target_obj is not None and hasattr(self, '_current_gripper_pos'):
                    for actor in self.obj_actors:
                        if target_obj in actor.get_name():
                            obj_pos = np.array(actor.get_pose().p)
                            dist = float(np.linalg.norm(self._current_gripper_pos - obj_pos))
                            _all_dists.append(dist)
                            # gripper 最低 z (用于计算 z gap)
                            if self._current_gripper_pos[2] < _gripper_z_min:
                                _gripper_z_min = float(self._current_gripper_pos[2])
                            _obj_z_target = float(obj_pos[2])
                            break

                # 检测 CLOSE 阶段
                if self._last_phase == "CLOSE" and close_start < 0:
                    close_start = local_idx
                if self._last_phase == "CLOSE":
                    close_end = local_idx
                    # 追踪 CLOSE 阶段 gripper-物体距离 + 相对偏移
                    if target_obj is not None and hasattr(self, '_current_gripper_pos'):
                        for actor in self.obj_actors:
                            if target_obj in actor.get_name():
                                obj_pos = np.array(actor.get_pose().p)
                                dist = float(np.linalg.norm(self._current_gripper_pos - obj_pos))
                                if dist < min_dist_in_close:
                                    min_dist_in_close = dist
                                _close_dists.append(dist)
                                # 物体跟随度量: 追踪相对偏移 obj_pos - gripper_pos
                                _offset = obj_pos - self._current_gripper_pos
                                _close_offsets.append(_offset.copy())
                                # 第三十二轮诊断: F50-F55 和 F60 打印手指link世界位置 vs 物体位置
                                # v14: 增加 F50-F55 诊断, 验证手指是否从两侧夹住物体
                                if local_idx in [50, 51, 52, 53, 54, 55, 60, 70, 80, 90, 100, 108]:
                                    _f1_pos, _f2_pos = None, None
                                    _glink_pos, _glink_quat = None, None
                                    _f1_qpos, _f2_qpos = None, None
                                    # v15i: 获取手指关节 qpos (实际闭合度). API: j.name
                                    _active_jts = robot.get_active_joints()
                                    for jt in _active_jts:
                                        jn = jt.name
                                        if jn == f"{self.side}_gripper_finger_joint1":
                                            _f1_qpos = float(jt.get_qpos()[0])
                                        elif jn == f"{self.side}_gripper_finger_joint2":
                                            _f2_qpos = float(jt.get_qpos()[0])
                                    for lnk in robot.get_links():
                                        n = lnk.get_name()
                                        if n == f"{self.side}_gripper_finger_link1":
                                            _f1_pos = np.array(lnk.get_entity_pose().p)
                                        elif n == f"{self.side}_gripper_finger_link2":
                                            _f2_pos = np.array(lnk.get_entity_pose().p)
                                        elif n == f"{self.side}_gripper_link":
                                            _gp = lnk.get_entity_pose()
                                            _glink_pos = np.array(_gp.p)
                                            _glink_quat = np.array(_gp.q)
                                    # 计算手指是否在物体两侧 (y 方向)
                                    _side_info = ""
                                    _y_gap_info = ""
                                    if _f1_pos is not None and _f2_pos is not None:
                                        _obj_y = obj_pos[1]
                                        _f1_y, _f2_y = _f1_pos[1], _f2_pos[1]
                                        _f1_side = "L" if _f1_y < _obj_y else "R"
                                        _f2_side = "L" if _f2_y < _obj_y else "R"
                                        _sandwich = "✓两侧" if _f1_side != _f2_side else "✗同侧"
                                        _y_gap = abs(_f1_y - _f2_y)
                                        _side_info = f", {_sandwich}(f1:{_f1_side},f2:{_f2_side})"
                                        _y_gap_info = f", y_gap={_y_gap:.4f}"
                                    # v15i: 手指 z vs 物体 z (判断是否真正夹住)
                                    _z_relation = ""
                                    if _f1_pos is not None:
                                        _f_obj_z = _f1_pos[2] - obj_pos[2]
                                        _z_relation = f", f1-obj_z={_f_obj_z:+.4f}({'上方' if _f_obj_z > 0.005 else '下方' if _f_obj_z < -0.005 else '夹住'})"
                                    _glink_info = ""
                                    if _glink_pos is not None and _f1_pos is not None:
                                        _z_diff = _f1_pos[2] - _glink_pos[2]
                                        _glink_obj_z = _glink_pos[2] - obj_pos[2]
                                        _glink_info = f", glink_z={_glink_pos[2]:.4f}, glink-obj_z={_glink_obj_z:+.4f}"
                                    # v15i: 手指 qpos (闭合度, 0=闭合, 0.05=张开)
                                    _qpos_info = ""
                                    if _f1_qpos is not None:
                                        _qpos_info = f", qpos=[{_f1_qpos:.4f},{_f2_qpos:.4f}]"
                                    logger.warning(f"  [DIAG F{local_idx}] obj={obj_pos.round(4)}, "
                                                    f"f1={(_f1_pos.round(4) if _f1_pos is not None else 'N/A')}, "
                                                    f"f2={(_f2_pos.round(4) if _f2_pos is not None else 'N/A')}, "
                                                    f"base={self._current_gripper_pos.round(4)}{_side_info}{_y_gap_info}{_z_relation}{_glink_info}{_qpos_info}")
                                break

                # 接触检测 (第十九轮修复: 只在 CLOSE 阶段计数, 避免优化器利用 APPROACH/TRANSPORT 阶段接触刷分)
                # 第二十七轮: 阈值从 2 降到 1, 让单指接触也能给优化器正向信号
                # 第二十八轮: 统一用 get_finger_contacts, 与 HybridGraspController 一致
                contacts = self.scene.get_contacts()
                if target_obj is not None and self._last_phase == "CLOSE":
                    f1_contact, f2_contact, contact_objs = get_finger_contacts(
                        robot, self.side, self.scene, self.obj_actors
                    )
                    has_obj_contact = (f1_contact or f2_contact) and target_obj in contact_objs
                    _close_contact_flags.append(1 if has_obj_contact else 0)
                    if has_obj_contact:
                        contact_frames_in_close += 1
                        _dbg_contact_traj.append(1)
                    else:
                        _dbg_contact_traj.append(0)

                # 穿透检测: 检查 contact points 的 distance (负值表示穿透)
                for c in contacts:
                    if c.points:
                        # 穿透深度 = 最大接触点的深度绝对值 (负距离)
                        for pt in c.points:
                            # SAPIEN ContactPoint 可能没有 dist 属性, 用 impulse 的 norm 代替
                            try:
                                pen_depth = float(pt.get_dist()) if hasattr(pt, 'get_dist') else 0.0
                            except Exception:
                                pen_depth = 0.0
                            if abs(pen_depth) > max_penetration:
                                max_penetration = abs(pen_depth)

            # 获取物体最终位置
            obj_final_z = 0.0
            obj_final_xy = [0.0, 0.0]
            obj_dropped = True
            if target_obj is not None and len(self.obj_actors) > 0:
                for actor in self.obj_actors:
                    if target_obj in actor.get_name():
                        pos = np.array(actor.get_pose().p)
                        obj_final_z = pos[2]
                        obj_final_xy = pos[:2].tolist()
                        obj_dropped = pos[2] < obj_init_z - 0.005  # 低于初始z 5mm 视为掉落
                        break

            # 获取 bowl 位置
            ctrl = self.grasp_controllers.get(self.side)
            bowl_xy = [0.0, 0.0] if ctrl is None or ctrl.bowl_pos is None else ctrl.bowl_pos[:2].tolist()

            # 第二十轮: 计算 CLOSE 阶段最后 5 帧的平均距离 (鼓励持续接触)
            if len(_close_dists) > 0:
                avg_dist_in_close_last5 = float(np.mean(_close_dists[-5:]))
            else:
                avg_dist_in_close_last5 = 1.0  # 无 CLOSE 帧, 给大距离惩罚

            # 第二十八轮: CLOSE 阶段最后 5 帧的接触帧数 (鼓励稳定夹持到结束)
            last_contact_count = int(np.sum(_close_contact_flags[-5:])) if len(_close_contact_flags) > 0 else 0

            # 第二十一轮: 计算全程平均距离 + gripper 最低点与物体 z 差距 (粗搜接近奖励)
            if len(_all_dists) > 0:
                avg_dist_throughout = float(np.mean(_all_dists))
            else:
                avg_dist_throughout = 1.0
            gripper_obj_z_gap = max(0.0, _gripper_z_min - _obj_z_target) if _obj_z_target is not None else 1.0

            # 物体跟随度量: 改进版 v4 — 五个条件排除假阳性
            # 条件1: 偏移方差小 = 物体跟着夹爪走
            # 条件2: 物体离开初始位置 (排除"不动"假阳性)
            # 条件3: 物体-夹爪距离在合理范围 (排除"推飞"假阳性)
            # 条件4a: 夹爪跟随 MANO 上升 (排除"推物体"假阳性 — MANO 在 CLOSE 阶段是上升的)
            # 条件4b: 物体 z 提升 (核心! 区分"抓"和"推")
            follow_score = 0.0
            follow_score_last5 = 0.0
            if len(_close_offsets) >= 3:
                _offsets_array = np.array(_close_offsets)  # (N, 3)
                _offset_var = np.var(_offsets_array, axis=0).mean()  # 平均方差
                _consistency = 1.0 / (1.0 + _offset_var * 100.0)  # 0~1, 方差越小分数越高

                # 条件2: 物体离开初始位置 (排除"不动"假阳性)
                _obj_init_xy_arr = np.array(obj_init_xy)
                _obj_final_xy_arr = np.array(obj_final_xy)
                _obj_displacement = float(np.linalg.norm(_obj_final_xy_arr - _obj_init_xy_arr))
                _obj_z_change = abs(obj_final_z - obj_init_z)
                _obj_total_move = np.sqrt(_obj_displacement**2 + _obj_z_change**2)
                _obj_moved = 1.0 - 1.0 / (1.0 + _obj_total_move * 50.0)  # sigmoid-like

                # 条件3: 物体-夹爪距离在合理范围 (排除"推飞"假阳性)
                if len(_close_dists) >= 5:
                    _mean_close_dist_last5 = float(np.mean(_close_dists[-5:]))
                else:
                    _mean_close_dist_last5 = float(np.mean(_close_dists)) if len(_close_dists) > 0 else 1.0
                _proximity = 1.0 / (1.0 + _mean_close_dist_last5 * 20.0)  # 距离<5cm 才给高分

                # 条件4a: 夹爪跟随 MANO 上升 (排除"推物体"假阳性)
                # MANO 在 CLOSE 阶段是上升的, 夹爪 z 应该跟着 MANO z 上升
                # 用 _dbg_gripper_z_traj 的末段变化率和 MANO 轨迹对比
                _follows_mano_score = 0.0
                _mano_traj = getattr(self, '_mano_gripper_traj', {}).get(self.side)
                if _mano_traj is not None and len(_dbg_gripper_z_traj) >= 5:
                    _mano_pos_arr = np.array(_mano_traj["pos"])
                    _mano_z_arr = _mano_pos_arr[:, 2] if _mano_pos_arr.ndim == 2 else np.array([])
                    if len(_mano_z_arr) >= self.num_frames:
                        _mano_z_close = _mano_z_arr[self.start_frame:self.start_frame + self.num_frames]
                        _gripper_z_arr = np.array(_dbg_gripper_z_traj)
                        if len(_gripper_z_arr) >= 5 and len(_mano_z_close) >= 5:
                            # 末段 5 帧: 夹爪 z 和 MANO z 的相关性 + MANO 上升量
                            _gz_last5 = _gripper_z_arr[-5:]
                            _mz_last5 = _mano_z_close[-5:]
                            _mano_up = float(_mz_last5[-1] - _mz_last5[0])  # MANO 末段 z 上升
                            _gripper_up = float(_gz_last5[-1] - _gz_last5[0])  # 夹爪末段 z 上升
                            # MANO 上升 > 1cm 且夹爪也上升 (> 0.5cm) 才给分
                            if _mano_up > 0.01 and _gripper_up > 0.005:
                                _follows_mano_score = min(1.0, _gripper_up / max(_mano_up, 0.001))

                # 条件4b: 物体 z 提升 (核心! 区分"抓"和"推")
                # 物体 z 提升 > 1cm 才开始给分, > 3cm 满分
                _obj_z_lift = float(obj_final_z - obj_init_z)
                _lift_score = 1.0 - 1.0 / (1.0 + max(_obj_z_lift - 0.01, 0.0) * 100.0)

                # 综合五个条件 (lift + follows_mano 都是必要条件, 用 max(x, 0.1) 避免完全归零)
                follow_score = (_consistency
                                * max(_obj_moved, 0.1)
                                * max(_proximity, 0.1)
                                * max(_lift_score, 0.1)
                                * max(_follows_mano_score, 0.1))

                # 末段偏移一致性: 最后5帧的偏移方差
                _last5_offsets = _offsets_array[-5:]
                _last5_var = np.var(_last5_offsets, axis=0).mean()
                _last5_consistency = 1.0 / (1.0 + _last5_var * 100.0)
                follow_score_last5 = (_last5_consistency
                                      * max(_obj_moved, 0.1)
                                      * max(_proximity, 0.1)
                                      * max(_lift_score, 0.1)
                                      * max(_follows_mano_score, 0.1))

                # 调试日志
                logger.debug(f"  [follow_score v4] consistency={_consistency:.3f}, "
                            f"obj_moved={_obj_moved:.3f} (move={_obj_total_move:.4f}m), "
                            f"proximity={_proximity:.3f} (dist={_mean_close_dist_last5:.4f}m), "
                            f"lift={_lift_score:.3f} (z_lift={_obj_z_lift:.4f}m), "
                            f"follows_mano={_follows_mano_score:.3f}, "
                            f"follow={follow_score:.3f}, follow_last5={follow_score_last5:.3f}")

            result = dict(
                params=opt_params,
                contact_frames_in_close=contact_frames_in_close,
                obj_init_z=obj_init_z, obj_final_z=obj_final_z,
                obj_init_xy=obj_init_xy, obj_final_xy=obj_final_xy,
                bowl_xy=bowl_xy,
                obj_dropped=obj_dropped,
                max_penetration=max_penetration,
                min_dist_in_close=min_dist_in_close,
                avg_dist_in_close_last5=avg_dist_in_close_last5,
                last_contact_count=last_contact_count,
                # 第二十一轮: 全程距离指标 (粗搜接近)
                avg_dist_throughout=avg_dist_throughout,
                gripper_obj_z_gap=gripper_obj_z_gap,
                # 第二十六轮: 帧间平滑度 (帧级窗口优化)
                smoothness_cost=smoothness_cost,
                # 物体跟随度量
                follow_score=follow_score,
                follow_score_last5=follow_score_last5,
                # 过渡帧跟踪: 过渡帧偏移 MANO 的程度 (越小越贴合 MANO)
                track_pen=getattr(self, '_current_track_pen', 0.0),
                # === DEBUG 插桩: 轨迹数据 ===
                _dbg_obj_z_traj=_dbg_obj_z_traj,
                _dbg_obj_xy_traj=_dbg_obj_xy_traj,
                _dbg_gripper_z_traj=_dbg_gripper_z_traj,
                _dbg_first_frame_state=_dbg_first_frame_state,
                # === DEBUG 插桩结束 ===
            )
            result["reward"] = getattr(self, '_reward_fn', compute_reward_xyz)(result)
            # 粗搜奖励与主奖励一致
            result["reward_coarse"] = result["reward"]
            return result

        # 3. 优化 (第二十九轮: 3 维 XYZ CEM)
        import multiprocessing as mp
        from traj_optimize import (
            compute_reward_xyz
        )

        # Phase 0: 基线扫描 — 找抓取窗口 (用 MANO 轨迹直接计算距离)
        logger.info(f"{'='*60}")
        logger.info(f"Phase 0: 基线扫描 — 找连续窗口")
        logger.info(f"{'='*60}")
        # 清除窗口参数, 确保 baseline 使用纯 MANO 轨迹
        self._window_params = None
        if hasattr(self, '_kf_cache'):
            del self._kf_cache

        # 从 MANO 轨迹计算基线距离
        s = self.side
        traj = self._mano_gripper_traj.get(s)
        ctrl = self.grasp_controllers.get(s)
        tgt = ctrl.target_obj if ctrl is not None else None
        if traj is None or len(traj["pos"]) == 0 or tgt is None or tgt not in self.obj_bbox_centers:
            logger.error(f"[Phase 0] 缺少 MANO 轨迹或目标物体, 无法扫描基线")
            return
        target_pos = np.array(self.obj_bbox_centers[tgt], dtype=np.float64)
        mano_positions = traj["pos"]
        baseline_dists = list(np.linalg.norm(mano_positions - target_pos, axis=1))
        total_frames = len(baseline_dists)

        logger.info(f"[Phase 0] 基线: {total_frames} 帧, target={tgt}, "
                    f"target_pos=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})")
        logger.info(f"[Phase 0] MANO Z: [{mano_positions[:,2].min():.3f}, {mano_positions[:,2].max():.3f}]m, "
                    f"Object Z: {target_pos[2]:.3f}m")

        dist_array = np.array(baseline_dists)
        fg = int(np.argmin(dist_array))

        # 用连续窗口代替百分位数散点窗口 — 以最接近帧为中心取 ±20 帧
        window_half = 20
        grasp_window = np.arange(
            max(0, fg - window_half),
            min(total_frames, fg + window_half + 1)
        )
        # 确保窗口至少有 3 帧
        if len(grasp_window) < 3:
            grasp_window = np.arange(max(0, fg-5), min(total_frames, fg+5))
            logger.warning(f"[Phase 0] 窗口帧数太少, 以 F{fg} 为中心取 {len(grasp_window)} 帧")

        min_dist = float(dist_array[fg])
        logger.info(f"[Phase 0] 基线: min_dist={min_dist:.4f}m at F{fg}")

        # Stage 1: 动态窗口抓取优化 (N_TRANS*2+1 帧 × 6DOF)
        from traj_optimize import cem_grasp_window_optimize, compute_reward_xyz, compute_frame_params

        self._window_params = None
        if hasattr(self, '_kf_cache'):
            delattr(self, '_kf_cache')

        # === 动态窗口计算: 根据 offset 长度自适应 ===
        _fp_s1 = compute_frame_params(len(traj["pos"]))
        _F50_s1 = _fp_s1['F50_IDX']
        # 获取物体位置 (用 bbox center, 比 _get_actual_obj_pos 更可靠)
        _ctrl_s1 = self.grasp_controllers.get(self.side)
        _target_obj_name = _ctrl_s1.target_obj if _ctrl_s1 else None
        _obj_pos_s1 = None
        if _target_obj_name and _target_obj_name in self.obj_bbox_centers:
            _obj_pos_s1 = np.array(self.obj_bbox_centers[_target_obj_name], dtype=np.float64)
        _mano_f50_pos = traj["pos"][_F50_s1]

        # 计算 F50 的完整 offset (考虑 +90°Y 旋转后的手指位置补偿)
        # +90°Y 旋转: 手指(局部+X 0.03689m) → 世界 -Z 3.689cm
        # 抓取中心(两指中点) 旋转后世界偏移 = [0, 0, -0.03689]
        # 修复: 旧值 z=-0.00589 导致手指在物体下方 3.1cm (lift=0 根因)
        _GRIPPER_LINK_OFFSET = np.array([0.0, 0.0, -0.03689])
        _init_offset_f50 = np.zeros(6)
        if _obj_pos_s1 is not None:
            _gripper_base_close = _obj_pos_s1 - _GRIPPER_LINK_OFFSET
            _init_offset_f50[:3] = _gripper_base_close - _mano_f50_pos
            logger.info(f"  [CEM init] obj_pos={_obj_pos_s1.round(4)}, "
                        f"mano_f50={_mano_f50_pos.round(4)}, "
                        f"init_offset_f50={_init_offset_f50[:3].round(4)}")

        # === 动态窗口: 根据 offset 长度计算 N_TRANS ===
        _offset_length = float(np.linalg.norm(_init_offset_f50[:3]))
        _MAX_STEP = 0.05  # 每帧最大移动 5cm (物理合理)
        _N_TRANS = max(2, int(np.ceil(_offset_length / _MAX_STEP)))
        # 限制窗口大小 (避免过大)
        _N_TRANS = min(_N_TRANS, 6)
        _GRASP_FRAMES_S1 = list(range(_F50_s1 - _N_TRANS, _F50_s1 + _N_TRANS + 1))
        _N_WINDOW = len(_GRASP_FRAMES_S1)
        _F50_LOCAL = _N_TRANS  # F50 在窗口中的局部索引
        _CEM_DIM = _N_WINDOW * 6

        # 保存窗口元数据 (供 rollout_single 和 Stage 2 使用)
        self._grasp_window_meta = {
            'n_trans': _N_TRANS,
            'frames': _GRASP_FRAMES_S1,
            'f50_local_idx': _F50_LOCAL,
            'dim': _CEM_DIM,
            'offset_f50': _init_offset_f50.copy(),
        }

        logger.info(f"  [动态窗口] offset_length={_offset_length:.4f}m, MAX_STEP={_MAX_STEP}m, "
                    f"N_TRANS={_N_TRANS}, 窗口=F{_GRASP_FRAMES_S1[0]}-F{_GRASP_FRAMES_S1[-1]} "
                    f"({_N_WINDOW}帧 × 6DOF = {_CEM_DIM}D)")

        # === init_mu: F50 指向物体, F50 之后继承 F50 偏移, F50 之前从 MANO 出发 ===
        # 关键: F50 闭合后, F51-F54 不能跳回 MANO 原始轨迹 (否则夹爪瞬间飞走)
        # F51-F54 应该继承 F50 的偏移, 然后随 MANO 轨迹变化 (跟随 MANO 的 z 和位姿)
        _init_mu = np.zeros(_CEM_DIM, dtype=np.float64)
        _f50_start = _F50_LOCAL * 6
        _init_mu[_f50_start:_f50_start+3] = _init_offset_f50[:3]  # F50 位置偏移
        # F51-F54: 继承 F50 的偏移 (保持抓取位置), CEM 在此基础上微调 xy
        for _i in range(_F50_LOCAL + 1, _N_WINDOW):
            _s = _i * 6
            _init_mu[_s:_s+3] = _init_offset_f50[:3]  # 继承 F50 位置偏移
        # F50 姿态偏移和 F50 之前帧的所有偏移都为0 (从 MANO 出发)

        logger.info(f"  [init_mu] F50 (local={_F50_LOCAL}): offset={_init_offset_f50[:3].round(4)}, "
                    f"F50之后: 继承F50偏移 (保持抓取位置), F50之前: 0 (从 MANO 出发)")

        # === 自适应搜索范围: 分三类帧 ===
        # 1. F50: 几乎固定 (指向物体, 只微调)
        # 2. F50 之前 (F46-F49): 优化 + w_track 约束, 大范围搜索 (从 MANO 到物体)
        # 3. F50 之后 (F51-F54): 跟随 MANO 的 z 和位姿, 主要改变 xy (保持抓取位置)
        #    关键: z 和姿态跟随 MANO (pos_std_z 小, rot_std 小), xy 可微调 (pos_std_xy 大)
        _trans_pos_std = max(_MAX_STEP * 0.7, _offset_length / 2.0)  # 前导帧位置搜索范围
        _pos_std_array = np.zeros(_N_WINDOW, dtype=np.float64)
        _rot_std_array = np.zeros(_N_WINDOW, dtype=np.float64)

        for _i in range(_N_WINDOW):
            if _i == _F50_LOCAL:
                # F50: 几乎固定
                _pos_std_array[_i] = 0.01
                _rot_std_array[_i] = 0.02
            elif _i < _F50_LOCAL:
                # F50 之前: 大范围搜索 (从 MANO 到物体)
                _pos_std_array[_i] = _trans_pos_std
                _rot_std_array[_i] = 0.15
            else:
                # F50 之后: 跟随 MANO 的 z 和位姿, 主要改变 xy
                # 用较小的 pos_std (限制整体偏移), 但 xy 方向允许微调
                _pos_std_array[_i] = 0.02  # 限制整体偏移幅度
                _rot_std_array[_i] = 0.03  # 姿态跟随 MANO, 几乎不变

        logger.info(f"  [搜索范围] F50: pos_std=0.01 (固定), rot_std=0.02; "
                    f"F50之前: pos_std={_trans_pos_std:.3f}, rot_std=0.15; "
                    f"F50之后: pos_std=0.02 (跟随MANO), rot_std=0.03")

        logger.info(f"{'='*60}")
        logger.info(f"Stage 1: 动态窗口 CEM 优化 ({_CEM_DIM}DOF, 窗口=F{_GRASP_FRAMES_S1[0]}-F{_GRASP_FRAMES_S1[-1]})")
        logger.info(f"{'='*60}")

        best_params, best_reward, reward_history = cem_grasp_window_optimize(
            rollout_fn=rollout_single,
            n_iterations=25,
            population_size=64,
            elite_frac=0.2,
            pos_std=0.08,           # 默认值 (会被 pos_std_array 覆盖)
            rot_std=0.15,           # 默认值 (会被 rot_std_array 覆盖)
            pos_range=0.30,
            rot_range=0.35,
            init_mu=_init_mu,
            n_frames=_N_WINDOW,
            f50_local_idx=_F50_LOCAL,
            pos_std_array=_pos_std_array,
            rot_std_array=_rot_std_array,
            frame_indices=_GRASP_FRAMES_S1,
        )

        self._opt_params_best = best_params
        self._reward_history = reward_history

        logger.info(f"[CEM-GraspWindow] 完成: best_reward={best_reward:.3f}, dim={_CEM_DIM}")
        for _gi, _gfi in enumerate(_GRASP_FRAMES_S1):
            _xyz = best_params[_gi*6:_gi*6+3]
            _rpy = best_params[_gi*6+3:_gi*6+6]
            _marker = "★" if _gi == _F50_LOCAL else " "
            logger.info(f"  {_marker}F{_gfi}: xyz=[{_xyz[0]:.4f},{_xyz[1]:.4f},{_xyz[2]:.4f}], "
                        f"rpy=[{_rpy[0]:.4f},{_rpy[1]:.4f},{_rpy[2]:.4f}]")

        # 保存结果
        np.save(self.output_dir / "opt_params.npy", best_params)
        np.save(self.output_dir / "reward_history.npy", np.array(reward_history))
        np.save(self.output_dir / "baseline_dists.npy", dist_array)

        # 4. 验证最优参数 (rollout 可重现性 + 抓取指标)
        self._opt_eval_count = 0  # 重置调试计数器
        verify_result = rollout_single(best_params)
        verify_reward = verify_result["reward"]
        verify_coarse = verify_result.get("reward_coarse", 0.0)
        logger.info(f"  [DEBUG 验证] best_reward={best_reward:.4f}, verify_reward={verify_reward:.4f}, "
                    f"diff={abs(best_reward - verify_reward):.4f}")
        logger.info(f"  [DEBUG 验证] verify: contact={verify_result.get('contact_frames_in_close', 0)}, "
                    f"last_contact={verify_result.get('last_contact_count', 0)}, "
                    f"follow_score={verify_result.get('follow_score', 0.0):.4f}, "
                    f"follow_last5={verify_result.get('follow_score_last5', 0.0):.4f}, "
                    f"min_dist={verify_result.get('min_dist_in_close', 1.0):.4f}, "
                    f"avg_dist_last5={verify_result.get('avg_dist_in_close_last5', 1.0):.4f}, "
                    f"avg_dist_all={verify_result.get('avg_dist_throughout', 1.0):.4f}, "
                    f"z_gap={verify_result.get('gripper_obj_z_gap', 1.0):.4f}, "
                    f"coarse_reward={verify_coarse:.3f}, "
                    f"xy_drift={np.linalg.norm(np.array(verify_result.get('obj_final_xy', [0,0])) - np.array(verify_result.get('obj_init_xy', [0,0]))):.4f}")
        # 保存验证轨迹供对比
        np.save(self.output_dir / "verify_reward.npy",
                np.array([best_reward, verify_reward, abs(best_reward - verify_reward)]))
        # === DEBUG 插桩 v2: 保存 verify rollout 的完整轨迹 + 第一帧状态 ===
        np.save(self.output_dir / "verify_obj_z_traj.npy", np.array(verify_result.get('_dbg_obj_z_traj', [])))
        np.save(self.output_dir / "verify_gripper_z_traj.npy", np.array(verify_result.get('_dbg_gripper_z_traj', [])))
        np.save(self.output_dir / "verify_obj_xy_traj.npy", np.array(verify_result.get('_dbg_obj_xy_traj', [])))
        _ffs = verify_result.get('_dbg_first_frame_state')
        if _ffs is not None:
            import json
            with open(self.output_dir / "verify_first_frame.json", 'w') as _f:
                json.dump(_ffs, _f, indent=2)
            gi1, gi2 = _ffs.get('gripper_idx1'), _ffs.get('gripper_idx2')
            rq, rv = _ffs['robot_qpos'], _ffs['robot_qvel']
            logger.info(f"  [DEBUG 验证] first_frame: gripper_qpos=[{rq[gi1]:.6f},{rq[gi2]:.6f}], "
                        f"gripper_qvel=[{rv[gi1]:.6f},{rv[gi2]:.6f}], "
                        f"obj_pos={_ffs['obj_pos']}, root_pose={_ffs['root_pose']}")
        # === DEBUG 插桩结束 ===

        # ============================================================
        # Stage 2: MPPI / CMA-ES 全局优化 (抓取窗口帧已固定)
        # ============================================================
        # Stage 1 已优化出 F48-F52 的30DOF, 这些帧在 Stage 2 中固定.
        # Stage 2 优化其他帧使轨迹平滑、贴近原MANO轨迹.
        # 全局线性插值轨迹 (从原始 MANO 轨迹修改)
        # ============================================================
        # 思路: 从原始 MANO 轨迹出发, 仅对 F50 施加 CEM-6D 验证的偏移,
        #       其他帧用线性插值在固定帧之间平滑过渡, 不做全局优化.
        opt_method = getattr(self, '_optimize_method', 'cem')
        if opt_method == 'mppi':
            from traj_optimize import (compute_frame_params,)

            # 动态计算帧参数 (从数据读取, 不硬编码)
            n_frames_actual = len(self._mano_gripper_traj[self.side]["pos"])
            fp = compute_frame_params(n_frames_actual)
            self._frame_params = fp  # 供 _compute_mano_neutral_target 使用
            _F0 = fp['F0_IDX']
            _F50 = fp['F50_IDX']
            _F95 = fp['F95_IDX']
            _F112 = fp['F112_IDX']
            _n_opt = fp['n_optimized']
            _n_params = fp['params_dim']

            logger.info(f"  帧参数: N_FRAMES={n_frames_actual}, "
                        f"FIXED=[{_F0},{_F50},{_F95},{_F112}], "
                        f"N_OPTIMIZED={_n_opt}, N_PARAMS={_n_params}")

            # Stage 2: 用 Stage 1 的30DOF结果构建654D参数
            # Stage 1 已优化出 F48-F52 的30DOF偏移 (每帧6DOF)
            # 这些帧固定, 其他帧在固定帧之间线性插值
            logger.info(f"{'='*60}")
            logger.info(f"[Stage 2] 用 Stage 1 的30DOF结果构建654D参数")
            logger.info(f"{'='*60}")

            # 动态窗口: 从 _grasp_window_meta 读取 (Stage 1 设置)
            _gw_meta_s2 = getattr(self, '_grasp_window_meta', None)
            if _gw_meta_s2 is not None:
                _N_TRANS_S2 = _gw_meta_s2['n_trans']
                _GRASP_FRAMES_S2 = _gw_meta_s2['frames']
            else:
                _N_TRANS_S2 = 2
                _GRASP_FRAMES_S2 = list(range(_F50 - _N_TRANS_S2, _F50 + _N_TRANS_S2 + 1))
            cem_best_30d = getattr(self, '_opt_params_best', None)
            _neutral_off = getattr(self, '_mano_neutral_offset', {}).get(self.side)

            # 抓取窗口帧偏移 (动态窗口)
            grasp_frame_offsets = {}
            _expected_dim_s2 = len(_GRASP_FRAMES_S2) * 6
            if cem_best_30d is not None and len(cem_best_30d) == _expected_dim_s2:
                for _gi, _frame_idx in enumerate(_GRASP_FRAMES_S2):
                    _off = cem_best_30d[_gi*6:(_gi+1)*6].copy()
                    # 不再叠加 _neutral_off: CEM 参数已是完整偏移
                    grasp_frame_offsets[_frame_idx] = _off
                logger.info(f"  使用CEM结果作为抓取窗口偏移 (dim={_expected_dim_s2}, "
                            f"窗口=F{_GRASP_FRAMES_S2[0]}-F{_GRASP_FRAMES_S2[-1]})")
            else:
                # 回退: 用解析偏移仅设F50
                logger.info(f"  CEM-30D无结果, 回退到解析偏移+CEM单帧")
                traj_data = self._mano_gripper_traj.get(self.side)
                grasp_offset = np.zeros(6)
                if traj_data is not None and len(traj_data["pos"]) > 0:
                    target_obj = find_pink_object(self.obj_info)
                    if target_obj is None:
                        wrist_pos_mid = np.mean(traj_data["pos"][:min(50, len(traj_data["pos"]))], axis=0)
                        target_obj = find_target_object_by_trajectory(
                            np.array([wrist_pos_mid]), self.obj_bbox_centers)
                    if target_obj and target_obj in self.obj_bbox_centers:
                        tgt_pos = np.array(self.obj_bbox_centers[target_obj], dtype=np.float64)
                        mano_pos_arr = np.array(traj_data["pos"])
                        if traj_data["R"] is not None and len(traj_data["R"]) > _F50:
                            R_at_F50 = np.array(traj_data["R"][_F50])
                            finger_forward = R_at_F50[:, 0] * FINGER_FORWARD_NEUTRAL
                        else:
                            finger_forward = np.zeros(3)
                        analytical_offset_pos = tgt_pos - mano_pos_arr[_F50] - finger_forward
                        grasp_offset[:3] = analytical_offset_pos
                        logger.info(f"  解析偏移(F50): {analytical_offset_pos.round(4)}, "
                                    f"目标={target_obj}")
                grasp_frame_offsets = {_F50: grasp_offset}

            # 提升偏移: F95 和 F112 增加 z 偏移
            LIFT_OFFSET_Z = 0.15  # 15cm 提升
            _f95_offset = np.zeros(6)
            _f95_offset[2] = LIFT_OFFSET_Z
            _f112_offset = np.zeros(6)
            _f112_offset[2] = LIFT_OFFSET_Z

            # 固定帧: F0=0, 抓取窗口帧, F95/F112=提升
            self._fixed_offsets_654 = {0: np.zeros(6)}
            self._fixed_offsets_654.update(grasp_frame_offsets)
            self._fixed_offsets_654[_F95] = _f95_offset
            self._fixed_offsets_654[_F112] = _f112_offset

            for _ff, _off in sorted(self._fixed_offsets_654.items()):
                logger.info(f"  固定帧 F{_ff}: xyz=[{_off[0]:.4f},{_off[1]:.4f},{_off[2]:.4f}], "
                            f"rpy=[{_off[3]:.4f},{_off[4]:.4f},{_off[5]:.4f}]")

            # 构建线性插值轨迹参数 (零偏移 + 固定帧之间的线性插值)
            best_params = np.zeros(_n_params, dtype=np.float64)
            fixed_frame_list = sorted(self._fixed_offsets_654.keys())
            for seg_i in range(len(fixed_frame_list) - 1):
                f_start = fixed_frame_list[seg_i]
                f_end = fixed_frame_list[seg_i + 1]
                off_start = self._fixed_offsets_654[f_start]
                off_end = self._fixed_offsets_654[f_end]
                for fi in range(f_start + 1, f_end):
                    if fi in self._fixed_offsets_654:
                        continue
                    t = (fi - f_start) / max(f_end - f_start, 1)
                    opt_idx = fi - sum(1 for f in fp['fixed_frames'] if f < fi)
                    best_params[opt_idx * 6:(opt_idx + 1) * 6] = (1 - t) * off_start + t * off_end
            logger.info(f"[MANO] 线性插值轨迹参数范数: {np.linalg.norm(best_params):.4f}")

            self._opt_params_best = best_params
            np.save(self.output_dir / "opt_params.npy", best_params)
            # 保存 frame_params 和 fixed_offsets_654 用于回放
            if hasattr(self, '_frame_params') and self._frame_params is not None:
                np.save(self.output_dir / "frame_params.npy", self._frame_params)
            if hasattr(self, '_fixed_offsets_654') and self._fixed_offsets_654 is not None:
                _fo_save = {str(k): v for k, v in self._fixed_offsets_654.items()}
                np.save(self.output_dir / "fixed_offsets_654.npy", _fo_save, allow_pickle=True)

            # 验证: 用完整 DECIMATION=8 做高精度验证
            logger.info(f"{'='*60}")
            logger.info(f"[MANO] 验证: 完整 DECIMATION=8")
            logger.info(f"{'='*60}")
            verify_result = rollout_single(best_params, decimation=8)
            contact = verify_result.get('contact_frames_in_close', 0)
            lift = verify_result.get('obj_final_z', 0) - verify_result.get('obj_init_z', 0)
            min_dist = verify_result.get('min_dist_in_close', 1.0)
            logger.info(f"  verify: contact={contact}, lift={lift:.4f}, min_dist={min_dist:.4f}")
            np.save(self.output_dir / "verify_reward.npy",
                    np.array([0.0, verify_result.get('reward', 0.0)]))

    def _compute_neutral_offsets(self):
        """计算 offset (复用 run() 中的逻辑, 供 run_optimize 调用)"""
        # 确保属性存在 (run_optimize 中可能还未初始化)
        if not hasattr(self, '_mano_neutral_offset'):
            self._mano_neutral_offset = {}
        if not hasattr(self, '_mano_grasp_frame'):
            self._mano_grasp_frame = {}
        if not hasattr(self, '_mano_target_pos'):
            self._mano_target_pos = {}
        for s in self.sides:
            traj = self._mano_gripper_traj.get(s)
            if traj is None or len(traj["pos"]) == 0 or "R" not in traj:
                self._mano_neutral_offset[s] = None
                self._mano_grasp_frame[s] = None
                continue
            ctrl = self.grasp_controllers.get(s)
            tgt = ctrl.target_obj if ctrl is not None else None
            if tgt is None or tgt not in self.obj_bbox_centers:
                self._mano_neutral_offset[s] = None
                self._mano_grasp_frame[s] = None
                continue
            obj_center = np.array(self.obj_bbox_centers[tgt], dtype=np.float64)
            mano_positions = np.asarray(traj["pos"])
            mano_R_traj = np.asarray(traj.get("R", []))
            # v4.5: 让 MANO F50 处的手指中点(不是 base)对准物体中心.
            # 夹爪 local: 手指原点位于 base +X 0.03689m 处, 因此世界坐标下手指中点为
            #   mano_pos[F50] + offset + R[F50] @ [0.03689, 0, 0]
            # 令它等于 obj_center, 得到
            #   offset = obj_center - mano_pos[F50] - R[F50] @ [0.03689, 0, 0]
            n_frames = len(mano_positions)
            F50_IDX = min(n_frames - 1, 50)
            _finger_forward = np.array([0.03689, 0.0, 0.0])
            _R_f50 = mano_R_traj[F50_IDX] if len(mano_R_traj) > F50_IDX else np.eye(3)
            _finger_mid_mano = mano_positions[F50_IDX] + _R_f50 @ _finger_forward
            offset = obj_center - _finger_mid_mano
            self._mano_neutral_offset[s] = offset
            self._mano_grasp_frame[s] = F50_IDX
            self._mano_target_pos = {s: obj_center}
            logger.info(f"[_compute_neutral_offsets] {s}: obj_center={obj_center.round(4)}, "
                        f"finger_mid_mano={_finger_mid_mano.round(4)}, "
                        f"offset={offset.round(4)}")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SAPIEN 物理仿真: R1 机器人抓取 GLB 物体"
    )
    parser.add_argument("--mode", type=str, default="full_robot",
                        choices=["full_robot", "gripper_only"],
                        help="URDF 模式: full_robot(整个机器人) / gripper_only(纯夹爪)")
    parser.add_argument("--side", type=str, default="right", choices=["right", "left", "both"],
                        help="手侧: right(右手) / left(左手) / both(双手协同)")
    parser.add_argument("--hawor-dir", type=str, default="/home/an/data/hawor/7")
    parser.add_argument("--ras-dir", type=str, default="/home/an/data/ras/my_7mp4_result")
    parser.add_argument("--output-dir", type=str, default=None,
                            help="输出目录; 默认按输入文件自动命名: <hawor名>_<ras名>_grasp_<mode>_<side>/")
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--views", type=str, default="both",
                        choices=["cam", "god", "both", "none"],
                        help="渲染视角: cam(第一人称相机视角) / god(上帝视角) / both(双视角, 默认) / none(不渲染视频)")
    parser.add_argument("--viewer", action="store_true",
                        help="启用交互式 Viewer 窗口 (实时查看仿真过程)")
    parser.add_argument("--grasp-mode", type=str, default="hybrid",
                        choices=["adaptive", "mano", "hybrid"],
                        help="抓取模式: hybrid(MANO驱动+接触力控, 默认) / adaptive(旧版MANO意图) / mano(纯MANO重放)")
    parser.add_argument("--test-stage1", action="store_true",
                        help="v4.4: 仅运行 Stage 1 单次 rollout 测试 (不渲染完整仿真)")
    parser.add_argument("--test-stage2", action="store_true",
                        help="v4.6: Stage 1 + Stage 2 轨迹重建测试")
    parser.add_argument("--test-stage3", action="store_true",
                        help="v4.6: Stage 1 + Stage 2 + Stage 3 全局优化测试")
    parser.add_argument("--optimize", action="store_true",
                        help="离线 CEM 优化轨迹参数 (借鉴 do-as-i-do Stage 5), 优化后再渲染")
    parser.add_argument("--opt-params", type=str, default=None,
                        help="手动指定优化参数文件 (npy), 跳过优化直接使用")
    parser.add_argument("--method", type=str, default="default",
                        choices=["default", "grasp-lift", "cem", "v4-pipeline"],
                        help="优化方法: default(无优化) / grasp-lift(CMA-ES 42维spline keyframes) / cem(CEM 9维) / v4-pipeline(v4.4 五阶段流水线)")
    parser.add_argument("--cmaes-gen", type=int, default=50,
                        help="CMA-ES 代数 (默认 50)")
    parser.add_argument("--cmaes-pop", type=int, default=32,
                        help="CMA-ES 种群大小")
    parser.add_argument("--cmaes-sigma", type=float, default=0.25,
                        help="CMA-ES 初始步长 (默认 0.25=25cm/25°)")
    parser.add_argument("--cmaes-sigma-end", type=float, default=0.005,
                        help="CMA-ES 终止步长 (默认 0.005=0.5°, sigma 指数衰减到此值)")
    parser.add_argument("--cmaes-sigma-decay", type=float, default=None,
                        help="CMA-ES 每代 sigma 衰减系数 (覆盖自动计算)")
    parser.add_argument("--optimizer", type=str, default="mppi",
                        choices=["mppi", "cmaes"],
                        help="优化器: mppi (默认, 654维 MPPI) / cmaes (42维两阶段 CMA-ES)")
    args = parser.parse_args()

    # v4.4: Stage 1 快速测试入口 (优先, 不进入完整优化/渲染流程)
    if args.test_stage1:
        # v4.4: Stage 1 必须在 gripper_only + 虚拟关节模式下测试,
        # 才能直接控制 6DOF 抓取位姿; grasp_mode 用 hybrid 以初始化目标物体.
        sim = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode="gripper_only",
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views=args.views,  # v4.16: 支持渲染输出视频
            grasp_mode="hybrid",
            viewer=False,
        )
        sim._test_stage1_only = True
        sim.video_tag = "stage1"  # v4.16: 视频文件名标签
        sim._align_scene()
        sim.run()  # 初始化场景+机器人+控制器+MANO轨迹
        # v4.5: test-stage1 跳过完整仿真循环, _mano_neutral_offset 未计算,
        # 需显式计算以将 MANO F50 对齐到世界坐标
        sim._compute_neutral_offsets()
        side = args.side if args.side != "both" else "right"
        logger.info("=" * 60)
        logger.info("v4.5 Stage 1 抓取姿态优化测试")
        logger.info("=" * 60)
        best_grasp = sim.cem_grasp_pose_optimize(side, n_iterations=5, population_size=8, top_k=4)
        if best_grasp is None:
            logger.error("Stage 1 优化失败")
            return
        logger.info(f"[test-stage1] 最优 grasp_pose: pos={best_grasp['pos'].round(4)}, "
                    f"euler={np.degrees(best_grasp['euler']).round(2)}, "
                    f"gripper_qpos={best_grasp['gripper_qpos']:.4f}, "
                    f"strategy={best_grasp['strategy']}")

        # v4.16: 最终最优位姿的 rollout 视频
        _has_video = (hasattr(sim, '_writer_god') and sim._writer_god is not None) or \
                     (hasattr(sim, '_writer_cam') and sim._writer_cam is not None)
        if _has_video:
            logger.info("[test-stage1] 录制最优 grasp 的 rollout 视频...")
            sim.rollout_grasp_only(best_grasp, side, strategy='pd_then_lock', n_frames=80, record_video=True)
            # 释放视频写入器
            if hasattr(sim, '_writer_cam') and sim._writer_cam is not None:
                sim._writer_cam.release()
                logger.info("[test-stage1] 相机视角视频已保存")
            if hasattr(sim, '_writer_god') and sim._writer_god is not None:
                sim._writer_god.release()
                logger.info("[test-stage1] 上帝视角视频已保存")
        return

    # v4.5: Stage 2 轨迹重建测试
    if args.test_stage2:
        from scipy.spatial.transform import Rotation as R_scipy
        sim = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode="gripper_only",
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views=args.views,  # v4.15: 支持渲染输出视频
            grasp_mode="hybrid",
            viewer=False,
        )
        sim._test_stage1_only = True
        sim.video_tag = "stage2"  # v4.16: 视频文件名标签
        sim._align_scene()
        sim.run()
        sim._compute_neutral_offsets()
        side = args.side if args.side != "both" else "right"

        # Stage 1: 先得到最优抓取姿态
        logger.info("=" * 60)
        logger.info("v4.5 Stage 2 测试: Stage 1 → Stage 2")
        logger.info("=" * 60)
        best_grasp = sim.cem_grasp_pose_optimize(side, n_iterations=5, population_size=8, top_k=4)
        if best_grasp is None:
            logger.error("Stage 1 优化失败")
            return
        logger.info(f"[test-stage2] Stage 1 最优: pos={best_grasp['pos'].round(4)}, "
                    f"gripper_qpos={best_grasp['gripper_qpos']:.4f}")

        # Stage 2: 轨迹重建 + MANO x,y 位置参考
        from traj_optimize import compute_frame_params
        traj = sim._mano_gripper_traj.get(side)
        N = len(traj["pos"])
        fp = compute_frame_params(N)
        F50 = fp['F50_IDX']

        # v4.9.2: N_TRANS 8 → 12, 让 Stage 2 重建范围更大 (F38-F62)
        # 给 PD 更多收敛时间 (F50-F56 保持 F50 位置, 让 qpos 从 0.018 收敛到 0.0034)
        recon = sim.reconstruct_trajectory(best_grasp, side, F50_IDX=F50, N_TRANS=12)
        if recon is None:
            logger.error("Stage 2 重建失败")
            return

        # 打印重建结果
        logger.info("[test-stage2] Stage 2 重建结果:")
        for f in sorted(recon['frames']):
            p = recon['pos'][f]
            g = recon['gripper_qpos'][f]
            logger.info(f"  F{f}: pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}), gripper_qpos={g:.4f}")

        # 验证: 用 Stage 2 的 F50 位姿做短 rollout
        optimized_pos = recon['pos'][F50]
        optimized_R = recon['R'][F50]
        test_pose = {
            'pos': optimized_pos,
            'R': optimized_R,
            'gripper_qpos': best_grasp['gripper_qpos'],
            'strategy': 'pd_then_lock',
        }
        sim_result = sim.rollout_grasp_only(test_pose, side, strategy='pd_then_lock', n_frames=80, record_video=True)
        if sim_result:
            lift = sim_result.get('obj_lift', 0.0)
            drift = sim_result.get('obj_xy_drift', float('inf'))
            logger.info(f"[test-stage2] 验证结果: lift={lift*100:.1f}cm, xy_drift={drift*100:.1f}cm")
            if lift > 0.02:
                logger.info("[test-stage2] Stage 2 验证通过!")
            else:
                logger.warning("[test-stage2] Stage 2 验证失败: 抬升不足 2cm")
        else:
            logger.error("[test-stage2] 验证 rollout 失败")

        # 保存 Stage 2 结果
        out_dir = Path(sim.output_dir) / "stage2"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "stage2_recon.npz",
                 frames=np.array(recon['frames']),
                 pos=np.array([recon['pos'][f] for f in recon['frames']]),
                 gripper_qpos=np.array([recon['gripper_qpos'][f] for f in recon['frames']]),
                 allow_pickle=True)
        logger.info(f"[test-stage2] Stage 2 结果已保存到 {out_dir}")

        # v4.16: 释放视频写入器
        if hasattr(sim, '_writer_cam') and sim._writer_cam is not None:
            sim._writer_cam.release()
            logger.info("[test-stage2] 相机视角视频已保存")
        if hasattr(sim, '_writer_god') and sim._writer_god is not None:
            sim._writer_god.release()
            logger.info("[test-stage2] 上帝视角视频已保存")
        return

    # v4.6: Stage 1 + Stage 2 + Stage 3 全局优化测试
    if args.test_stage3:
        from scipy.spatial.transform import Rotation as R_scipy
        from traj_optimize import compute_frame_params
        sim = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode="gripper_only",
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views=args.views,  # v4.15: 支持渲染输出视频
            grasp_mode="hybrid",
            viewer=False,
        )
        sim._test_stage1_only = True
        sim.video_tag = "stage3"  # v4.16: 视频文件名标签
        sim._align_scene()
        sim.run()
        sim._compute_neutral_offsets()
        side = args.side if args.side != "both" else "right"

        # Stage 1: 优先加载已保存的最优, 跳过 CMA-ES (避免每次跑出不同结果)
        _output_dir = args.output_dir or f"output/gripper_only_{side}"
        stage1_path = Path(_output_dir) / "stage1" / "best_grasp.npz"
        if stage1_path.exists():
            logger.info("=" * 60)
            logger.info(f"[test-stage3] Stage 1: 加载已保存的最优 {stage1_path}")
            logger.info("=" * 60)
            d = np.load(str(stage1_path), allow_pickle=True)
            best_grasp = {k: d[k] for k in d.keys()}
            best_grasp['pos'] = np.asarray(best_grasp['pos'], dtype=np.float64)
            best_grasp['gripper_qpos'] = float(best_grasp['gripper_qpos'])
            # 兼容旧 npz (无 enclose_z 或 pos[2] 仍是 hover_z): 修正 pos[2] 为 enclose_z
            _enclose_z_loaded = float(best_grasp.get('enclose_z', 0))
            if abs(_enclose_z_loaded) < 0.001:
                _obj_z = 0.0173  # 从场景获取 (粉色物体初始 z)
                _enclose_z_loaded = _obj_z + 0.030
                best_grasp['enclose_z'] = _enclose_z_loaded
                logger.info(f"[test-stage3] 旧 npz 无 enclose_z, 估计={_enclose_z_loaded:.4f}")
            if best_grasp['pos'][2] > _enclose_z_loaded + 0.005:
                # 旧 npz: pos[2] 是 hover_z, 需要替换为 enclose_z
                logger.info(f"[test-stage3] pos[2]: {best_grasp['pos'][2]:.4f} (hover_z) → {_enclose_z_loaded:.4f} (enclose_z)")
                best_grasp['pos'][2] = _enclose_z_loaded
        else:
            logger.info("=" * 60)
            logger.info("[test-stage3] Stage 1: 抓取姿态优化 (无已保存数据)")
            logger.info("=" * 60)
            best_grasp = sim.cem_grasp_pose_optimize(side, n_iterations=5, population_size=8, top_k=4)
            if best_grasp is None:
                logger.error("Stage 1 失败")
                return
        logger.info(f"[test-stage3] Stage 1: pos={best_grasp['pos'].round(4)}, "
                    f"gripper_qpos={best_grasp['gripper_qpos']:.4f}")

        # Stage 2
        logger.info("=" * 60)
        logger.info("[test-stage3] Stage 2: 轨迹重建")
        logger.info("=" * 60)
        traj = sim._mano_gripper_traj.get(side)
        if traj is None or len(traj.get("pos", [])) == 0:
            logger.error("[test-stage3] Stage 2: 无 MANO 轨迹, 跳过")
            return
        N = len(traj["pos"])
        fp = compute_frame_params(N)
        F50 = fp['F50_IDX']
        # v4.9.2: N_TRANS 8 → 12, 让 Stage 2 重建范围更大 (F38-F62)
        # 给 PD 更多收敛时间 (F50-F56 保持 F50 位置, 让 qpos 从 0.018 收敛到 0.0034)
        recon = sim.reconstruct_trajectory(best_grasp, side, F50_IDX=F50, N_TRANS=12)
        if recon is None:
            logger.error("Stage 2 失败")
            return
        logger.info(f"[test-stage3] Stage 2: {len(recon['frames'])} 帧重建完成")

        # v4.7: Stage 3 简化为平滑过渡, 不做 CEM 优化
        # 设置 Stage 2 结果
        sim._v4_stage2_recon = recon
        sim._v4_best_grasp = best_grasp
        sim._frame_params = fp

        logger.info("=" * 60)
        logger.info("[test-stage3] Stage 3: 平滑过渡 + 物理验证 (v4.7, 无CEM优化)")
        logger.info("=" * 60)

        # 直接运行 rollout (无 opt_params)
        result = sim.rollout_v4_stage3(side=side)
        lift = result.get('obj_lift', 0.0)
        drift = result.get('xy_drift', 0.0)
        obj_mano_mean = result.get('obj_mano_dist_mean', float('inf'))
        obj_mano_f50 = result.get('obj_mano_dist_f50', float('inf'))
        obj_mano_f80 = result.get('obj_mano_dist_f80', float('inf'))
        logger.info(f"[test-stage3] 结果: lift={lift*100:.1f}cm, drift={drift*100:.1f}cm")
        logger.info(f"[test-stage3] 物体-MANO距离: mean={obj_mano_mean*1000:.1f}mm, "
                    f"F50={obj_mano_f50*1000:.1f}mm, F80={obj_mano_f80*1000:.1f}mm")

        # 保存结果
        out_dir = Path(sim.output_dir) / "stage3"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "stage3_result.npz",
                 result={k: v for k, v in result.items() if isinstance(v, (int, float, np.ndarray))},
                 allow_pickle=True)
        logger.info(f"[test-stage3] 结果已保存到 {out_dir}")

        # v4.15: 释放视频写入器 (如果使用了渲染)
        if hasattr(sim, '_writer_cam') and sim._writer_cam is not None:
            sim._writer_cam.release()
            logger.info(f"[test-stage3] 相机视角视频已保存")
        if hasattr(sim, '_writer_god') and sim._writer_god is not None:
            sim._writer_god.release()
            logger.info(f"[test-stage3] 上帝视角视频已保存")
        return

    # v4.4: 五阶段流水线入口
    if args.method == "v4-pipeline":
        logger.info("=" * 60)
        logger.info("v4.4 五阶段流水线 (Stage 1 → Stage 2 → Stage 3)")
        logger.info("=" * 60)
        sim_opt = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode=args.mode,
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views="none",
            grasp_mode="hybrid",
            viewer=False,
        )
        sim_opt.run(run_main_loop=False)  # 仅初始化, 不进入主循环 (Stage 1/2/3 在此后执行)
        side = args.side if args.side != "both" else "right"
        sim_opt.run_v4_pipeline(side)

        # 渲染回放
        sim = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode=args.mode,
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views=args.views,
            grasp_mode="hybrid",
            viewer=args.viewer,
        )
        sim._opt_mano_gripper_traj = sim_opt._opt_mano_gripper_traj
        sim._frame_params = sim_opt._frame_params
        sim._fixed_offsets_654 = sim_opt._fixed_offsets_654
        sim._obj_initial_poses = sim_opt._obj_initial_poses
        sim.set_opt_params(sim_opt._opt_params_best)
        sim.run()
        return

    # 第十九轮: 支持 --method grasp-lift (CMA-ES) 和 --method cem (CEM), 以及 --opt-params
    opt_params = None
    if args.opt_params:
        opt_params = np.load(args.opt_params)
        logger.info(f"  加载手动指定参数: {args.opt_params}, shape={opt_params.shape}")
    elif args.method == "grasp-lift" or args.method == "cem" or args.optimize or args.optimizer in ("mppi", "cmaes"):
        logger.info("=" * 60)
        logger.info(f"第二十九轮: 3D XYZ CEM 优化 (3 维, "
                    f"15 iter × 32 pop)")
        logger.info("=" * 60)

        # 初始化 simulator (不渲染, 用于 rollout)
        sim_opt = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode=args.mode,
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views="none",  # 不渲染
            grasp_mode="hybrid",
        )
        # 设置优化方法和参数
        sim_opt._optimize_method = args.optimizer
        if args.method == "grasp-lift" or args.optimizer == "cmaes":
            sim_opt._cmaes_gen = args.cmaes_gen
            sim_opt._cmaes_pop = args.cmaes_pop
            sim_opt._cmaes_sigma = args.cmaes_sigma
            sim_opt._cmaes_sigma_end = args.cmaes_sigma_end
            sim_opt._cmaes_sigma_decay = args.cmaes_sigma_decay
        elif args.optimizer == "mppi":
            sim_opt._optimize_method = 'mppi'
        sim_opt.run_optimize()  # 优化 + 保存最优参数
        opt_params = getattr(sim_opt, '_opt_params_best', None)
        if opt_params is not None:
            logger.info(f"  最优参数 shape: {opt_params.shape}")
            logger.info(f"  保存最优参数: {sim_opt.output_dir / 'opt_params.npy'}")
        else:
            logger.warning("  优化未完成, 跳过最终渲染")

    # 输出目录传 None, 由 GraspSimulator 自动按输入文件命名
    sim = GraspSimulator(
        hawor_dir=args.hawor_dir,
        ras_dir=args.ras_dir,
        mode=args.mode,
        side=args.side,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        views=args.views,
        grasp_mode=args.grasp_mode,
        viewer=args.viewer,
    )

    # 设置优化参数
    if opt_params is not None:
        # 如果是从优化器来的参数, 复制相关属性
        _sim_opt = locals().get('sim_opt')
        if _sim_opt is not None:
            # === CEM 路径修复: 将 CEM 3D/6D 参数扩展为 654 维 ===
            _fp = getattr(_sim_opt, '_frame_params', None)
            _fo = getattr(_sim_opt, '_fixed_offsets_654', None)
            if len(opt_params) in (3, 6) and _fp is not None and _fo is not None:
                from traj_optimize import compute_frame_params
                _n_frames = len(_sim_opt._mano_gripper_traj.get(args.side, {}).get("pos", []))
                if _n_frames > 0:
                    _cem_dim = len(opt_params)
                    _grasp_offset = np.zeros(6)
                    _neutral_off = getattr(_sim_opt, '_mano_neutral_offset', {}).get(args.side)
                    if _neutral_off is not None:
                        _grasp_offset[:3] = _neutral_off[:3]
                    _grasp_offset[:_cem_dim] += opt_params[:_cem_dim]
                    _ff_list = sorted(_fo.keys())
                    _n_opt = _fp['n_optimized']
                    _full_params = np.zeros(_n_opt * 6, dtype=np.float64)
                    for _si in range(len(_ff_list) - 1):
                        _f_s = _ff_list[_si]
                        _f_e = _ff_list[_si + 1]
                        _off_s = _fo[_f_s]
                        _off_e = _fo[_f_e]
                        for _fi in range(_f_s + 1, _f_e):
                            if _fi in _fo:
                                continue
                            _t = (_fi - _f_s) / max(_f_e - _f_s, 1)
                            _oi = _fi - sum(1 for _f in _fp['fixed_frames'] if _f < _fi)
                            _full_params[_oi * 6:(_oi + 1) * 6] = (1 - _t) * _off_s + _t * _off_e
                    opt_params = _full_params
                    logger.info(f"  CEM {_cem_dim}D → 654D 扩展: neutral={_neutral_off}, cem={opt_params[:_cem_dim]}, "
                                f"grasp_offset={_grasp_offset.round(4)}")
            sim.set_opt_params(opt_params)
            if hasattr(_sim_opt, '_frame_params') and _sim_opt._frame_params is not None:
                sim._frame_params = _sim_opt._frame_params
                logger.info(f"  复制帧参数: FIXED={_sim_opt._frame_params['fixed_frames']}, "
                             f"params_dim={_sim_opt._frame_params['params_dim']}")
            if hasattr(_sim_opt, '_fixed_offsets_654') and _sim_opt._fixed_offsets_654 is not None:
                sim._fixed_offsets_654 = _sim_opt._fixed_offsets_654
                logger.info(f"  复制固定帧偏移: {list(_sim_opt._fixed_offsets_654.keys())}")
            if hasattr(_sim_opt, '_mano_gripper_traj') and _sim_opt._mano_gripper_traj is not None:
                sim._opt_mano_gripper_traj = _sim_opt._mano_gripper_traj
                for s in _sim_opt._mano_gripper_traj:
                    logger.info(f"  复制 MANO 轨迹 [{s}]: {len(_sim_opt._mano_gripper_traj[s]['pos'])} 帧")
            if hasattr(_sim_opt, '_obj_initial_poses') and _sim_opt._obj_initial_poses is not None:
                sim._obj_initial_poses = _sim_opt._obj_initial_poses
                logger.info(f"  复制物体初始位姿 ({len(sim._obj_initial_poses)} 个) 从优化器到渲染器")
        else:
            # 直接加载 654D 参数 (从 --opt-params 加载)
            sim.set_opt_params(opt_params)
            # 加载 frame_params 和 fixed_offsets_654
            _fp_path = sim.output_dir / "frame_params.npy"
            _fo_path = sim.output_dir / "fixed_offsets_654.npy"
            if _fp_path.exists() and _fo_path.exists():
                sim._frame_params = np.load(_fp_path, allow_pickle=True).item()
                _fo_loaded = np.load(_fo_path, allow_pickle=True).item()
                sim._fixed_offsets_654 = {int(k): v for k, v in _fo_loaded.items()}
                logger.info(f"  加载帧参数: FIXED={sim._frame_params['fixed_frames']}, "
                             f"params_dim={sim._frame_params['params_dim']}")
                logger.info(f"  加载固定帧偏移: {list(sim._fixed_offsets_654.keys())}")
            else:
                logger.info("  无 frame_params/fixed_offsets_654 文件, 跳过")
    sim.run()


if __name__ == "__main__":
    main()
