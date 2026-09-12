"""grasp_hawor.py — SAPIEN 物理仿真: 用 R1 机器人真实抓取 GLB 物体

项目目标:
  给定 HaWoR 手部重建 + RAS 场景重建 (GLB), 用 R1 机器人 URDF 在 SAPIEN 中
  复刻抓取 GLB 物体的动作, 并通过参数级验证 (物体提升/接触检测).

五阶段流水线 (v4):
  Stage 0: 候选抓取姿态生成 (无需仿真)
  Stage 1: 抓取姿态优化 (6DOF CMA-ES, ~80帧短仿真, "能不能夹住")
  Stage 2: 轨迹重建 (smoothstep插值, 确定性, F46-F54过渡)
  Stage 3: 原样重放 Stage2 后追加 MANO 携带，并在有效支撑处条件释放
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
  python grasp_hawor.py --test-stage1 --side right --output output/gripper_right
  python grasp_hawor.py --test-stage3 --side right --output output/gripper_right
  python grasp_hawor.py --mode gripper_only --side right \\
      --hawor-dir /home/an/data/hawor/7 \\
      --ras-dir /home/an/data/ras/7_vggt_omega
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
    # v6.9: 夹爪/MANO 手朝向对齐 (沿重力方向判定)
    GRAVITY_DIR, align_gripper_approach_to_gravity, flip_mano_hand_down,
    # URDF + 物理 + 接触函数
    prepare_full_robot_urdf, prepare_gripper_only_urdf,
    setup_physics_scene, setup_robot, _is_floating_root,
    physics_step, fetch_contacts,
    get_finger_contacts, get_grasp_force, is_obj_in_gripper_frame,
    # logger
    logger,
)
from mano_smoothing import smooth_gripper_traj, smooth_joints_seq
# v51 (2026-09-08): MINT EgoPipeline 参数域平滑作主选方案 (_SMOOTH_MINT=1 开启).
#   final_clean: block_hampel → fill_slerp_rot → 相机感知 UKF+RTS，在 FK 之前对
#   HaWoR 参数(pred_trans/rot/hand_pose/betas)平滑。对比实验证实比 SG 更平滑且更保形。
import os as _os_smooth
_USE_MINT = _os_smooth.environ.get('_SMOOTH_MINT', '0') == '1'
if _USE_MINT:
    from mint_cleaning.final_clean import final_clean as _mint_final_clean
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
    compute_mano_based_gripper_pose,
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
from stage1_recipe import STAGE1_CLOSE_CONTROLS, STAGE1_SETTLE_CONTROLS, load_stage1_grasp
from stage2_reconstruct_mixin import Stage2Mixin
from stage3_optimize_mixin import Stage3Mixin

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("grasp_hawor")


def transcode_to_h264(path):
    """v4.25: 将 cv2.VideoWriter 写出的 mp4v(MPEG-4 Part 2) 转码为 H.264.

    原因: mp4v 编码浏览器/IDE 内置预览器不支持, 打开提示"解析失败";
    H.264 + yuv420p + faststart 是兼容性最好的组合.
    使用 imageio_ffmpeg 自带的 ffmpeg 二进制, 无需系统安装 ffmpeg.
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        logger.warning("[video] imageio_ffmpeg 不可用, 跳过转码 (文件仍为 mp4v)")
        return False
    path = str(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    tmp = path + ".h264tmp.mp4"
    try:
        r = subprocess.run(
            [ff, "-y", "-v", "error", "-i", path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
            capture_output=True, timeout=120)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
            logger.info(f"[video] 已转码 H.264: {os.path.basename(path)}")
            return True
    except Exception as e:
        logger.warning(f"[video] 转码失败 {os.path.basename(path)}: {e}")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return False


# ============================================================
# 8. 主仿真器
# ============================================================
class GraspSimulator(Stage1Mixin, Stage2Mixin, Stage3Mixin):
    """R1 机器人 SAPIEN 仿真主类, 继承 Stage 1/2/3 Mixin"""

    def __init__(self, hawor_dir, ras_dir, mode="full_robot", side="right",
                 output_dir=None, num_frames=-1, start_frame=0, views="both",
                 grasp_mode="adaptive", viewer=False, grasp_pose_method="svd_palm",
                 root_output_dir=None):
        self.hawor_dir = Path(hawor_dir)
        self.ras_dir = Path(ras_dir)
        self.mode = mode
        self.side = side  # "left" / "right" / "both"
        self.sides = ["left", "right"] if side == "both" else [side]
        self.views = views  # "cam" / "god" / "both" — 指定渲染哪些视角
        self.video_tag = ""  # v4.16: 视频文件名标签 (如 "stage1"/"stage2"/"stage3")
        self._is_v4_pipeline = False  # v6.8: v4-pipeline 入口标记, 影响 output 路由
        self.grasp_mode = grasp_mode  # "adaptive" (MANO意图+相位) / "mano" (纯重放)
        self.grasp_pose_method = grasp_pose_method  # "analytical"(加权SVD) / "svd_palm"(5点SVD+GS)
        self.viewer = viewer  # 是否启用交互式 Viewer
        # 输出目录: 当前脚本下的 output/<mode>_<side>/ (v4.29: gripper_only_left -> gripper_left,
        # 中间产物统一收进该子目录; 最终四件套 D1-D4 由 collect_deliverable 提升到 output/ 顶层)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            script_dir = Path(__file__).resolve().parent
            self.output_dir = script_dir / "output" / f"{mode.replace('_only', '')}_{side}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # v6.7 (2026-09-01): 运行根目录 — 与 self.output_dir 区分.
        # 子实例 (如 Stage1 子仿真) 的 output_dir 是 <root>/stageN, 若不记录 root,
        # 每个子实例都会在自己的 output_dir 下重复建 alignment/ 并把交付物往上级提,
        # 造成 stageN/stageN/ 嵌套 + alignment 多份 + D*.mp4 散落到 output/ 顶层.
        # 目标结构: <root>/{alignment,stage0..3,final,test}
        self.root_output_dir = Path(root_output_dir) if root_output_dir else self.output_dir
        self.root_output_dir.mkdir(parents=True, exist_ok=True)
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.hand_indices = {"left": 0, "right": 1}  # hand_idx=0=左手, 1=右手
        self.hand_idx = 0 if side == "left" else 1  # 向后兼容 (单侧)
        # v6.9b 上游翻转开关: 在 `_mano_to_sapien` 里翻转整只 MANO 手使其朝下 (物理_utils.
        # flip_mano_hand_down). **默认关闭** — 2026-09-02 实测反而严重退步:
        #   开启: Stage1 恒定距离 ref=2.7m/31.6cm, max_dev=9-51mm FAIL (夹爪远离物体)
        #   关闭: Stage1 恒定距离 ref=3.8mm, max_dev=2.3mm PASS, lift=20cm, 验证门 3/3
        # 根因: 翻转以腕为支点把指尖上下移动 ~28cm, 改变了 MANO 轨迹的绝对位置;
        #   而 CEM 搜索空间 / enclose_z / 物体对齐都是按"未翻转轨迹"标定的, 翻转后
        #   CEM 参考位姿掉到物体下方 (base_z=0.0107 < 物体 0.0393), 候选全部偏离.
        #   下游 `align_gripper_approach_to_gravity` 只改 R 不改 MANO 指尖位置, 才是最小侵入解.
        self.flip_mano_hand = False

        # v6.8: grasp.log 的 FileHandler 移至 _align_scene() 中设置 (用 self._run_dir()),
        # 因为 __init__ 时 video_tag/_is_v4_pipeline 尚未设置, 无法决定日志子目录.
        # 此处不再调用 _setup_file_logger.

        self.scene = None
        self.robot_info = None
        self.obj_actors = []
        self.retargeting = None
        self.ik_solver = None
        self.joint_filters = {s: JointFilter() for s in self.sides}  # per-side 滤波器
        self.joint_filter = self.joint_filters[self.sides[0]]  # 向后兼容 (单侧)
        self.transform_params_path = None
        # v4.23: 统一坐标系 — SAPIEN 世界坐标 = GLB 坐标系 (z 向上, 地面 z=0)。
        # GLB 物体天然贴地 (z_min=0), 去掉非恒等 GLB→SAPIEN 变换 + 落地补偿 (~0.6m 偏差),
        # 手轨迹 SLAM→GLB 后不再乘 R_AXIS; 输出 (grasp.log) 直接是 GLB 真值坐标。
        self.glb_coords = True
        # 自适应抓取控制器 (在 run() 中加载物体后初始化)
        self.grasp_controllers = None  # {side: AdaptiveGraspController}

    def _setup_file_logger(self, log_path):
        """配置日志同时输出到终端和输出目录的 log 文件

        Args:
            log_path: 日志文件绝对路径 (由调用方计算, 支持按 run 子目录路由)
        """
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
        # v6.8: grasp.log 放在当前 run 子目录 (stageN/test/final), 由 _run_dir 决定.
        # __init__ 时 video_tag 尚未设置, 故 handler 在此延迟安装; 幂等, 多次 run() 不重复.
        self._setup_file_logger(self._run_dir() / "grasp.log")

        rec_file = self._find_reconstruction_file()
        if rec_file is None:
            raise FileNotFoundError(f"未找到 HaWoR 重建文件: {self.hawor_dir}")

        # v6.7: alignment 是全局对齐参数, 只在运行根目录存一份 (<root>/alignment).
        # 原实现用 self.output_dir, 导致每个 stage 子实例都建一份 stageN/alignment.
        align_dir = self.root_output_dir / "alignment"
        align_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Step 1: 对齐 RAS GLB → HaWoR 坐标系 (调用 01_align_scene.py)")
        logger.info("=" * 60)
        # 优先使用 hawor_dir/output/alignment 下已有的 transform_params (新版 01 输出, 含 R_hand_to_glb)
        # 不存在时才重新计算 (旧版 01 输出不含 R_hand_to_glb, 坐标变换可能不准)
        existing_params = Path(self.hawor_dir) / "output" / "alignment" / "transform_params.npz"
        if existing_params.exists():
            import shutil
            target_path = align_dir / "transform_params.npz"
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
            # APPROACH (F0-F50): 手指全开, 沿抓取位姿垂直下降 (v4.27b)
            # CLOSE (F50-F95): F50-F60 边下沉到物体中心边闭合, F60-F75 保持夹持, F75-F95 抬升
            _descend_start = max(0, _F50 - 10)   # F40 开始下降

            if local_idx_safe < _F50:
                # [修复 v4.27b] 沿 Stage1 已验证的抓取位姿垂直下降 (对齐 Stage1 rollout):
                # 全程保持 opt_R[F50] 抓取朝向、xy 对齐 opt_pos[F50], 仅 z 从安全高度
                # smoothstep 降到 opt_pos[F50].z → F50 切换时位置/姿态零跳变.
                # 原实现用 +90°Y 垂直姿态降到物体锚点, F50 瞬间切 opt 轨迹:
                # 底座横移 ~4cm + 姿态瞬转 ~48°, 指尖横扫物体 (实测 F45 时 gap=14cm 安全,
                # F50 后物体被推到 20cm 外, lift=0).
                _p_grasp = traj_cache['opt_pos'][_F50]
                _R_grasp = traj_cache['opt_R'][_F50]
                gripper_pos = np.array([_p_grasp[0], _p_grasp[1], _approach_safe_z])
                if local_idx_safe >= _descend_start:
                    _t_desc = (local_idx_safe - _descend_start) / max(_F50 - _descend_start, 1)
                    _s = _smoothstep(_t_desc)
                    gripper_pos[2] = _approach_safe_z * (1 - _s) + float(_p_grasp[2]) * _s
                # 手指全开下降 (gap≈11.9cm > 物体弦宽 11.2cm), 到 F50 再闭合
                gripper_val = GRIPPER_MAX_OPEN
                gripper_R = _R_grasp
                phase = "APPROACH"

            elif local_idx_safe < _F95:
                # [修复 v4.26] CLOSE 阶段跟随 Stage 2/3 重建的优化轨迹 (opt_pos/opt_R),
                # 而不是强制 _rot_close (垂直朝下) + 物体锚点.
                # 原因: _rot_close 把夹爪闭合轴映射到世界 Y, 而物体 Y 尺寸(11.2cm)
                # 大于手指最大闭合行程, 手指永远夹不住物体 (实测 y_gap 保持 13.5cm,
                # lift=0). Stage 2 重建的 opt_pos/opt_R 在 F50 采用 Stage 1 优化抓取
                # 位姿 (euler=[125.35,-20.83,5.57]), 该位姿在 Stage 1 已验证能夹起物体.
                gripper_pos = traj_cache['opt_pos'][local_idx_safe].copy()
                gripper_R = traj_cache['opt_R'][local_idx_safe].copy()
                gripper_val = gripper_close_qpos
                # z 高度保护:
                # 1) 从 F51 起, 夹爪 z 不低于已到达的最高点 (F50 夹持高度).
                #    否则 F56 后 smoothstep 回到预平移的 MANO 轨迹 (其 z 在地下,
                #    因为整条轨迹被 pre_translate 下移了 0.565m), 会把物体拖坠回桌面.
                # 2) F60 后主动抬升 (0.2cm/帧), 把物体带离桌面达到 lift>=3cm 验收标准.
                if local_idx_safe > _F50:
                    gripper_pos[2] = max(gripper_pos[2], float(traj_cache['z_cummax'][local_idx_safe]))
                if local_idx_safe > _F50 + 10:
                    _n_lift = local_idx_safe - (_F50 + 10)
                    _active_z = float(traj_cache['z_cummax'][_F50 + 10]) + 0.002 * _n_lift
                    gripper_pos[2] = max(gripper_pos[2], _active_z)
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
          - v6.0 统一物理仿真: 虚拟关节 (vx/vy/vz/rz/ry/rx) 与手指全部纯 PD drive
            (set_drive_target + physics_step), 与 Stage1/2/3 完全一致
          - 不使用 set_qpos 传送 / virtual_lock_targets 硬锁 (运动学操作全部废除,
            初始化摆放除外); 跟踪误差 = 真实 PD 滞后, 见 _track_history

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
                    # v6.0 统一物理仿真: 6 虚拟关节纯 PD drive (对齐 Stage1/2/3),
                    # 不再每帧 set_qpos 传送 (传送 = 无穷刚度运动学幽灵, 接触力/摩擦失真).
                    # 历史: "CLOSE阶段PD漂移3cm"的根因是 set_qpos+drive 双重控制震荡
                    # (见 physics_step docstring), 纯 drive 无此问题.
                    rz, ry, rx = rotmat_to_zyx_euler(gripper_R)
                    active_joints = robot.get_active_joints()
                    # 设置 PD 目标 (纯 drive 跟踪)
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
        if self.grasp_pose_method == "svd_palm":
            root_pos, root_R, joint1, joint2 = compute_mano_based_gripper_pose(
                joints_sapien, prefix=prefix
            )
        else:
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
        # v6.0 统一物理: actual = 当前真实 qpos (驱动前一帧的物理结果),
        # 与 expected 的差 = PD 跟踪误差 (历史: 从待写入的 qpos dict 读 actual,
        # actual==expected, 误差指标恒为 0 的假指标)
        qpos = robot.get_qpos().copy()

        if is_virtual:
            # 虚拟关节: 纯 PD drive 目标 (位置 + ZYX Euler), 不再 set_qpos 传送
            rz, ry, rx = rotmat_to_zyx_euler(root_R)
            active_joints = robot.get_active_joints()
            for vkey, vval in [('vx', root_pos[0]), ('vy', root_pos[1]),
                               ('vz', root_pos[2]), ('rz', rz), ('ry', ry), ('rx', rx)]:
                active_joints[virtual_idx[vkey]].set_drive_target(float(vval))

        gripper_idx1 = self.robot_info["gripper_idx1"]
        gripper_idx2 = self.robot_info["gripper_idx2"]

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
                current_qpos = qpos[gripper_idx1]  # 当前手指 qpos (真实物理状态)
                adapted_target, phase, grasp_info = controller.update(
                    root_pos, root_R, mano_gripper_val,
                    robot=robot, scene=self.scene, current_qpos=current_qpos
                )
                joint1 = adapted_target
                joint2 = adapted_target
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
                    # v6.0: 只改 drive 目标 (经返回值→physics_step), 不再写 qpos
                    joint1 = float(min(clamped, float(joint1)))
                    joint2 = joint1

        # 轨迹跟踪误差: 物理输出(actual) vs 真实 MANO 期望(expected)
        # 用户: "夹爪运动要物理和真实输出的误差来判断准不准确"
        # v6.0: qpos 是驱动前的真实物理状态 (previous step 结果), 误差 = PD 跟踪滞后
        actual_j1 = float(qpos[gripper_idx1])
        actual_j2 = float(qpos[gripper_idx2])
        self._last_track = {
            'root_err_mm': float(np.linalg.norm(expected_root - root_pos) * 1000.0),  # 根位置误差 mm
            'j1_err_mm': float(abs(expected_j1 - actual_j1) * 1000.0),  # 手指1误差 mm
            'j2_err_mm': float(abs(expected_j2 - actual_j2) * 1000.0),  # 手指2误差 mm
        }
        # v6.0 统一物理: 累积跟踪误差 (Stage0/主回放的物理验证指标), run() 结束时汇总
        if not hasattr(self, '_track_history'):
            self._track_history = []
        self._track_history.append(dict(self._last_track))
        # 不再 robot.set_qpos(qpos) — 全部经 drive target → physics_step 纯物理执行
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

        2026-09-06 (用户): 视频里看不到红色碰撞 box — 真实碰撞 mesh (2.47cm 指面)
        被蓝色视觉手指外壳 (9.2cm) 完全包住. 修复:
          1) 碰撞体 actor 提高不透明度 (RGBA[1,0,0,0.6]);
          2) 手指视觉外壳降为半透明 (alpha≈0.3), 红色碰撞体透出可见;
        使视频同时显示"视觉外壳(观感) + 真实碰撞体(物理接触边界)"两层.
        不影响物理 (仅 visual, 不参与碰撞).
        """
        self._collision_visual_actors = {}  # link_name -> visual actor
        self._collision_link_map = {}  # link_name -> robot link (用于取位姿)
        if not getattr(self.scene, "_render_available", True):
            return
        try:
            import sapien.render
            sides = ["left", "right"] if self.side == "both" else [self.side]
            vis_mesh_map = {}  # link_name -> 对应视觉外壳 STL (半透明化用)
            for s in sides:
                for link_name_suffix in ["finger_link1", "finger_link2"]:
                    full_link_name = f"{s}_gripper_{link_name_suffix}"
                    # 2026-09-07: 可视化读 *_scaled.STL (即 URDF 物理碰撞同文件, 显示=实际)
                    mesh_file = str(R1_MESH_DIR / f"{s}_gripper_{link_name_suffix}_scaled.STL")
                    builder = self.scene.create_actor_builder()
                    mat = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 0.6])
                    builder.add_visual_from_file(mesh_file, material=mat)
                    actor = builder.build_kinematic(name=f"_collision_vis_{full_link_name}")
                    self._collision_visual_actors[full_link_name] = actor
                    # 隐形线框栅格 (wireframe): 碰撞体棱角清晰可见, 半透红色网格包围真实碰撞边界
                    try:
                        _b = self.scene.create_actor_builder()
                        _wb = sapien.render.RenderMaterial(base_color=[1.0, 0.0, 0.0, 0.9])
                        _b.add_visual_from_file(mesh_file, material=_wb, scale=[1,1,1])
                        _wf = _b.build_kinematic(name=f"_collision_wire_{full_link_name}")
                        # wireframe 需 ray 化: 用少量细长方体勾勒 bbox 棱 (简化栅格)
                        _tgt = _wf.get_pose()
                        import numpy as _np
                        for _edge0, _edge1 in [((0,0,0),(1,0,0)),((0,0,0),(0,1,0)),((0,0,0),(0,0,1)),
                                                ((1,1,1),(-1,0,0)),((1,1,1),(0,-1,0)),((1,1,1),(0,0,-1))]:
                            pass  # 简化: wireframe mesh 直接显红边不可行, 留作后续
                    except Exception:
                        pass
                    vis_mesh_map[full_link_name] = f"{s}_gripper_{link_name_suffix}.STL"
            # 建立手指 link 映射 + 视觉外壳半透明化 (让内部红色碰撞体可见)
            for link in robot.get_links():
                lname = link.get_name()
                if lname in self._collision_visual_actors:
                    self._collision_link_map[lname] = link
                if lname in vis_mesh_map:
                    try:
                        for _vb in link.get_visual_bodies():
                            for _rs in _vb.get_render_shapes():
                                _mat = _rs.material
                                if _mat is None:
                                    continue
                                _c = list(_mat.base_color) if getattr(_mat, 'base_color', None) is not None \
                                    else [0.82, 0.82, 1.0, 1.0]
                                if len(_c) == 4:
                                    _c[3] = 0.30  # 视觉外壳半透明, 红色碰撞体透出
                                _mat.set_base_color(_c)
                    except Exception as _materr:
                        logger.warning(f"  视觉外壳半透明化失败 ({lname}): {_materr}")
            logger.info(f"  碰撞可视化已创建 ({len(self._collision_visual_actors)} 个半透明红色 actor, "
                        f"RGBA=[1,0,0,0.6], 视觉外壳 alpha≈0.3, 真实碰撞边界可见)")
            # 2026-09-07 (用户: "需要看到物体的碰撞体"): 物体的物理碰撞体 (dynamic→盒/bigger→凸包)
            # 与视觉 mesh 不同 — 叠加红色半透明 box (盒形碰撞) 或凸包, 每帧跟物体位姿,
            # 用于确认夹爪碰撞体 vs 物体碰撞体真实接触 (无空气传力)。
            self._obj_col_vis = {}
            try:
                for _ac in getattr(self, 'obj_actors', []):
                    _nm = _ac.get_name()
                    b = self.scene.create_actor_builder()
                    _ok = False
                    # obj_info 里有 bbox_size(碰撞盒半尺寸的2倍) → 红盒
                    _oi = getattr(self, 'obj_info', {}).get(_nm)
                    if _oi is not None and 'bbox_size' in _oi:
                        _hs = np.array(_oi['bbox_size'], dtype=float) / 2.0
                        b.add_box_visual(pose=sapien.Pose([0, 0, 0]),
                                         half_size=_hs.tolist(),
                                         material=sapien.render.RenderMaterial(base_color=[1.0, 0.2, 0.2, 0.35]))
                        _ok = True
                    if _ok:
                        _va = b.build_kinematic(name=f"_obj_col_vis_{_nm}")
                        # 初位姿对齐物体 (保持绕 bbox_center 对齐)
                        _va.set_pose(sapien.Pose(p=_ac.get_pose().p, q=_ac.get_pose().q))
                        self._obj_col_vis[_nm] = _va
            except Exception as _oerr:
                logger.warning(f"  [碰撞可视化] 物体碰撞体叠加失败: {_oerr}")
            if self._obj_col_vis:
                logger.info(f"  [碰撞可视化] 物体碰撞体叠加 {len(self._obj_col_vis)} 个 (红盒=物体碰撞体)")
            # 2026-09-09 (用户: "夹爪中点可能判断有问题"): 渲染夹爪中点 marker —
            # 黄球 = 当前代码用的 finger-link 原点中点 0.5*(orig1+orig2);
            # 绿球 = 修正后的指尖中点 0.5*(指尖1+指尖2) (触点/夹持中心).
            # 视频里两球并排, 可直观判断当前 fmid 是否偏移(固定偏差).
            self._fmid_vis = self._fmidtip_vis = None
            try:
                import sapien.render as _sr
                if getattr(self.scene, "_render_available", True):
                    _bm = self.scene.create_actor_builder()
                    _bm.add_sphere_visual(radius=0.008,
                        material=_sr.RenderMaterial(base_color=[1.0, 0.85, 0.0, 0.95]))
                    self._fmid_vis = _bm.build_kinematic(name="_fmid_marker")
                    _bt = self.scene.create_actor_builder()
                    _bt.add_sphere_visual(radius=0.008,
                        material=_sr.RenderMaterial(base_color=[0.1, 1.0, 0.2, 0.95]))
                    self._fmidtip_vis = _bt.build_kinematic(name="_fmidtip_marker")
            except Exception as _fme:
                logger.warning(f"  [碰撞可视化] fmid marker 创建失败: {_fme}")
            # 2026-09-06: 创建后立即同步一次初始位姿, 避免首帧 update_render 时 box 在原点
            self._update_collision_visualization()
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
            # 物体碰撞体跟随物体位姿 (每帧)
            for _nm, _va in getattr(self, '_obj_col_vis', {}).items():
                for _ac in getattr(self, 'obj_actors', []):
                    if _ac.get_name() == _nm:
                        _va.set_pose(_ac.get_pose())
                        break
            # 夹爪中点 marker: 黄=link原点中点(当前fmid), 绿=指尖中点(修正)
            _fm = getattr(self, '_fmid_vis', None)
            _ft = getattr(self, '_fmidtip_vis', None)
            if (_fm is not None or _ft is not None) and self.side:
                _l1 = self._collision_link_map.get(f"{self.side}_gripper_finger_link1")
                _l2 = self._collision_link_map.get(f"{self.side}_gripper_finger_link2")
                if _l1 is not None and _l2 is not None:
                    _p1 = np.array(_l1.get_pose().p, dtype=np.float64)
                    _p2 = np.array(_l2.get_pose().p, dtype=np.float64)
                    if _fm is not None:
                        _fm.set_pose(sapien.Pose((0.5 * (_p1 + _p2)).tolist()))
                    if _ft is not None:
                        _tip1 = np.array(_l1.get_pose().transform([0.0405, 0.0, 0.0]), dtype=np.float64)
                        _tip2 = np.array(_l2.get_pose().transform([0.0405, 0.0, 0.0]), dtype=np.float64)
                        _ft.set_pose(sapien.Pose((0.5 * (_tip1 + _tip2)).tolist()))
        except Exception:
            pass

    def collect_deliverable(self, dst_name):
        """v4.29: 将本次运行的主视角视频收集为标准交付物 (D2/D3/D4)。

        v6.7 (2026-09-01): 目标目录改为 <运行根目录>/final/ (原为 self.output_dir.parent,
        即往上一级提, 导致 D*.mp4 散落到 output/ 顶层; 且子实例会把产物提错层级)。
        目标结构: <root>/{alignment,stage0..3,final,test}

        视角优先级: right_view (用户指定右视图) > god_view > cam_view。
        返回目标路径; 找不到视频时返回 None。
        """
        import shutil
        _tag = f"_{self.video_tag}" if self.video_tag else ""
        final_dir = self.root_output_dir / "final"
        for prefix in ("right_view", "god_view", "cam_view"):
            src = self._run_dir() / f"{prefix}{_tag}_{self.mode}_{self.side}.mp4"
            if src.exists():
                final_dir.mkdir(parents=True, exist_ok=True)
                dst = final_dir / dst_name
                shutil.copy2(src, dst)
                logger.info(f"[deliverable] {src.name} -> {dst}")
                return dst
        logger.warning(f"[deliverable] 未找到可收集的视频 ({self.output_dir}, tag={self.video_tag!r})")
        return None

    def _run_subdir(self):
        """当前 run 的输出子目录名 (相对 self.output_dir).
        stage0..3 → 对应 stageN; v4-pipeline → final; 其余(ad-hoc/未标注) → test.
        """
        tag = getattr(self, 'video_tag', None) or ''
        if tag in ('stage0', 'stage1', 'stage2', 'stage3'):
            return tag
        if getattr(self, '_is_v4_pipeline', False) or (tag.startswith('v4') if tag else False):
            return 'final'
        return 'test'

    def _run_dir(self):
        """当前 run 的输出目录 (含 mkdir). 供视频/log/数据文件用."""
        d = self.output_dir / self._run_subdir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _opt_dir(self):
        """v4-pipeline 优化产物目录 = final/intermediate/."""
        d = self.output_dir / 'final' / 'intermediate'
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── 运行元数据 (2026-09-11, 用户要求) ──────────────────────────────
    def dump_run_metadata(self, tag=""):
        """把本次运行的全部物理参数 + 物体 scale 落盘到 <run_root>/run_metadata.json.

        用户要求: "每次跑的时候物理参数, 物体的scale需要记录下来".
        目的: 跨运行可追溯 (K/D 改动、物体 scale、环境开关), 避免"改了参数
        但事后对不上是哪一套"的问题. 失败不影响主流程.
        """
        import json
        import datetime

        def _get(mod, name, default=None):
            return getattr(mod, name, default)

        try:
            import physics_env as _pe
            import physics_utils as _pu

            try:
                import sapien as _sap
                _sap_ver = getattr(_sap, "__version__", None)
            except Exception:
                _sap_ver = None

            env_keys = (
                "_OBJ_SCALE", "_OBJ_SCALE_TARGET", "_OBJ_SHAPE", "_OBJ_SHAPE_TARGET",
                "_NO_GRIPPER_GROUND", "_GRIP_FRICTION", "_SMOOTH_MINT", "_Q_OPEN",
                "_VIRTUAL_PD_K", "_VIRTUAL_PD_D",
            )
            objects = {}
            for _name, _center in (getattr(self, "obj_bbox_centers", {}) or {}).items():
                _info = (getattr(self, "obj_info", {}) or {}).get(_name, {}) or {}
                _size = _info.get("bbox_size")
                objects[_name] = {
                    "bbox_center": [float(v) for v in _center],
                    "bbox_size": ([float(v) for v in _size] if _size is not None else None),
                    "body_type": _info.get("body_type"),
                }

            meta = {
                "tag": tag or getattr(self, "video_tag", "") or "",
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "sapien_version": _sap_ver,
                "env_overrides": {k: os.environ.get(k) for k in env_keys},
                "physics": {
                    # 时间/求解
                    "PHYSICS_TIMESTEP": _get(_pe, "PHYSICS_TIMESTEP"),
                    "CONTROL_FREQ": _get(_pe, "CONTROL_FREQ"),
                    "DECIMATION": _get(_pe, "DECIMATION"),
                    # 基座虚拟关节 PD (本次实际生效值, 受 _VIRTUAL_PD_* 影响)
                    "VIRTUAL_PD_STIFFNESS": _get(_pu, "VIRTUAL_PD_STIFFNESS"),
                    "VIRTUAL_PD_DAMPING": _get(_pu, "VIRTUAL_PD_DAMPING"),
                    # 手指 PD
                    "FINGER_PD_STIFFNESS": _get(_pu, "FINGER_PD_STIFFNESS"),
                    "FINGER_PD_DAMPING": _get(_pu, "FINGER_PD_DAMPING"),
                    "FINGER_PD_FORCE_LIMIT": _get(_pu, "FINGER_PD_FORCE_LIMIT"),
                    "GRIPPER_STIFFNESS": _get(_pe, "GRIPPER_STIFFNESS"),
                    "GRIPPER_DAMPING": _get(_pe, "GRIPPER_DAMPING"),
                    "GRIPPER_FORCE": _get(_pe, "GRIPPER_FORCE"),
                    # 夹爪几何/行程
                    "GRIPPER_MAX_OPEN": _get(_pe, "GRIPPER_MAX_OPEN"),
                    "GRIPPER_INIT_OPEN": _get(_pe, "GRIPPER_INIT_OPEN"),
                    "FINGER_EFFECTIVE_HALF_SPACING": _get(_pe, "FINGER_EFFECTIVE_HALF_SPACING"),
                    # 接触/物体
                    "GRIPPER_FRICTION": _get(_pe, "GRIPPER_FRICTION"),
                    "OBJECT_DENSITY": _get(_pe, "OBJECT_DENSITY"),
                    "OBJECT_MIN_MASS": _get(_pe, "OBJECT_MIN_MASS"),
                },
                "objects": objects,
                "ground_z": getattr(self, "_ground_z", None),
                "stage2_config": getattr(self, "_stage2_config", {}),
            }
            path = Path(self.root_output_dir) / "run_metadata.json"
            path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(
                "[run_metadata] 物理参数已记录 → %s (virtual PD K=%s D=%s, obj_scale=%s)",
                path,
                meta["physics"]["VIRTUAL_PD_STIFFNESS"],
                meta["physics"]["VIRTUAL_PD_DAMPING"],
                os.environ.get("_OBJ_SCALE"),
            )
            return meta
        except Exception as exc:  # 记录失败不得影响主流程
            logger.warning("[run_metadata] 落盘失败 (不影响运行): %s", exc)
            return None

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
        self._c2w_all = (R_c2w_all, t_c2w_all)  # 供 rollout 渲染逐帧更新相机位姿

        # 3. 预扫描 GLB 最低点 → 创建场景
        logger.info("=" * 60)
        logger.info("Step 3: 创建 SAPIEN 场景 + 加载 GLB")
        logger.info("=" * 60)
        glb_path = self.ras_dir / "final_scene.glb"
        ground_z = 0.0
        if glb_path.exists() and self.transform_params_path:
            ground_z = compute_glb_ground_z(glb_path, self.transform_params_path,
                                            use_glb_coords=self.glb_coords)
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
                glb_path, self.transform_params_path, self.scene, fast_collision=True,
                use_glb_coords=self.glb_coords  # 单凸包 (CoACD 太慢, 可按需切换 False)
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
        # v5.2 对齐链与 render_quick.py cam_traj_to_glb / 001_align_scene.py 一致 (用户确认正确):
        #   Step1: MANO FK 关节在 HaWoR 输入相机系 (pred_trans z>0 已证实)
        #          p_hw = R_c2w[f] @ p_cam + t_c2w[f]  ← 逐帧真实相机位姿
        #          (不用固定 Rx_hand=diag(1,-1,-1): 001 实验证实 R_X=I,
        #           相机转动后固定近似漂移可达 34°/13cm)
        #   Step2: 相似变换到 GLB (= SAPIEN 世界, glb_coords=True)
        #          p_glb = s * R_hand_to_glb @ p_hw + t_hand_to_glb
        _tp_s, _tp_R_h2g, _tp_t_h2g = 1.0, np.eye(3), np.zeros(3)
        if self.transform_params_path:
            _tp = np.load(str(self.transform_params_path))
            if 'scale_ratio' in _tp.files:
                _tp_s = float(_tp['scale_ratio'])
                _tp_R_h2g = np.asarray(_tp['R_hand_to_glb'], dtype=np.float64)
                _tp_t_h2g = np.asarray(_tp['t_hand_to_glb'], dtype=np.float64).reshape(3)
            else:  # 旧版 01 键名 (GLB→HaWoR 方向, 取逆; 历史兼容, 正常不会走到)
                _tp_s = 1.0 / max(float(_tp['s_inv']), 1e-9)
                _tp_R_h2g = np.asarray(_tp['R_inv'], dtype=np.float64)
                _tp_t_h2g = np.asarray(_tp['t_inv'], dtype=np.float64).reshape(3) / max(float(_tp['s_inv']), 1e-9)
            logger.info(f"  transform_params: s={_tp_s:.4f}, t_h2g={_tp_t_h2g.round(4)}")
        else:
            logger.warning("  无 transform_params, MANO 轨迹仅做逐帧相机提升 (可能不准)")

        # 存储到实例, 供 run_v4_pipeline 使用 (R_hand 即 R_h2g, R_X=I)
        self._mano_xform_s = _tp_s
        self._mano_xform_R_hand = _tp_R_h2g
        self._mano_xform_t = _tp_t_h2g
        # 相机变换需要 raw R_h2g
        self._cam_xform_R_h2g = _tp_R_h2g
        self._cam_xform_s = _tp_s
        self._cam_xform_t = _tp_t_h2g

        def _mano_to_sapien(pts_cam, frame_idx=0):
            """MANO 相机系 → SAPIEN/GLB 世界系.

            与 render_quick.py cam_traj_to_glb 完全一致的数据驱动链:
              Step1: 逐帧真实相机位姿提升到 HaWoR 世界系
                     p_hw = R_c2w[f] @ p_cam + t_c2w[f]
                     (不用固定 Rx_hand=diag(1,-1,-1): 001_align_scene 实验证实 R_X=I,
                      相机转动后固定近似会产生明显漂移)
              Step2: 001 相似变换 → GLB (= SAPIEN 世界, glb_coords=True)
                     p_glb = s * R_hand_to_glb @ p_hw + t_hand_to_glb
            """
            if R_c2w_all is not None and len(R_c2w_all) > 0:
                fi = min(max(int(frame_idx), 0), len(R_c2w_all) - 1)
                pts_hw = (R_c2w_all[fi] @ np.asarray(pts_cam).T).T + t_c2w_all[fi]
            else:
                pts_hw = np.asarray(pts_cam)
            pts_glb = _tp_s * (_tp_R_h2g @ pts_hw.T).T + _tp_t_h2g
            # v51 §1.1/§6 U2 (2026-09-05): GLB 即 SAPIEN 世界, 旧 R_AXIS 链分支删除.
            _out = pts_glb
            # v51 §6 U4 (2026-09-05): 独立 flip 出口删除 — 夹爪朝下由重定向 Step6
            # align_gripper_approach_to_gravity 统一保证; flip_mano_hand_down 函数保留
            # (physics_utils), 供 P2-7 上游翻转重标定实验使用.
            return _out

        # v6.9b (2026-09-02): P1-9 修复 — 若 MANO 夹爪轨迹已由 pre_translate 准备好, 复用之.
        # 原逻辑 `if not _has_opt_traj:` 无条件重建, 会在第二次 run() 时覆盖掉 pre_translate
        # 烘进 traj["pos"] 的平移 → CEM/终选读到未平移位姿.
        # 实测 (2026-09-02): pre_translate 报告 F50 finger_mid=[-0.5123 -0.795 0.5954],
        #   但 CEM 读到的 traj.pos[F50]=[-0.3157 -0.3441 -0.0008], 反算指尖偏离物体
        #   [0.0071, 0.1216, -0.0504] (≈13cm), 且帧数从 113 变成 112 — 轨迹被换掉了.
        # v51 U3 (2026-09-05): 门控改用 len() 直接判长 — 原写法 `.get("pos", []) or []`
        # 在 pos 为 numpy 数组时(pre_translate 之后必然是)触发 "truth value ambiguous" 崩溃;
        # 旧顺序 smooth 在最后会把 pos 转回 list 掩盖了该 bug, 统一 smooth 先行后暴露.
        _pre_side_traj = (getattr(self, '_mano_gripper_traj', None) or {}).get(self.side) or {}
        _pre_pos = _pre_side_traj.get("pos", [])
        _pre_ready = (getattr(self, '_mano_traj_prepared', False)
                      and _pre_pos is not None and len(_pre_pos) > 0)
        if not _has_opt_traj and _pre_ready:
            _t = self._mano_gripper_traj[self.side]
            logger.info(f"  复用已准备的 MANO 夹爪轨迹 ({len(_t['pos'])} 帧, 已含 "
                        f"pre_translate 平移), 跳过重建 (P1-9 修复)")
            wrist_positions = _t["pos"]
        elif not _has_opt_traj:
            wrist_positions = []
            # 同时预计算 MANO 解析夹爪轨迹 (用户: "mano参数为主体, 可以平移轨迹")
            # 每帧: root_pos, j1, j2 from compute_analytical_gripper_pose
            mano_gripper_traj = {}  # side -> {"pos": [], "R": [], "j1": [], "j2": [], "hand_pose": []}
            if self.side == "both":
                # 双手: 坐标变换 → (按侧)平滑 → 重定向 (v51 §1, 2026-09-05 重构)
                _seq = {s: [] for s in self.sides}
                _hp = {s: [] for s in self.sides}
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
                            _seq[s].append(_mano_to_sapien(joints, fi))
                            _hp[s].append(hd["pred_hand_pose"][fi])
                        except Exception:
                            continue
                for s in self.sides:
                    if not _seq[s]:
                        continue
                    mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": [],
                                            "hand_pose": [], "joints": []}
                    _sm = smooth_joints_seq(np.array(_seq[s], dtype=np.float64))
                    for _i in range(len(_sm)):
                        joints_sapien = _sm[_i]
                        if self.grasp_pose_method == "svd_palm":
                            root_pos, root_R, j1, j2 = compute_mano_based_gripper_pose(
                                joints_sapien, prefix=s,
                            )
                        else:
                            root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
                                joints_sapien[0, :3], joints_sapien[4, :3],
                                joints_sapien[8, :3], prefix=s,
                            )
                        _fd_real = float(np.linalg.norm(
                            joints_sapien[8, :3] - joints_sapien[4, :3])) / max(self._mano_xform_s, 1e-6)
                        j1 = j2 = float(np.clip((_fd_real - 0.01) / 2.0, 0.0, 0.05))
                        mano_gripper_traj[s]["pos"].append(root_pos)
                        mano_gripper_traj[s]["R"].append(root_R)
                        mano_gripper_traj[s]["j1"].append(j1)
                        mano_gripper_traj[s]["j2"].append(j2)
                        mano_gripper_traj[s]["hand_pose"].append(_hp[s][_i])
                        wrist_positions.append(root_pos)  # anchor 与轨迹同源 (同上)
                        # v51 §1.2/§1.5: 保存 MANO 参考点 (双手分支, 平滑后)
                        mano_gripper_traj[s]["joints"].append(joints_sapien)
            else:
                # v51 §1 (2026-09-05 重构, 用户确认顺序): 坐标变换 → 平滑 → 重定向 → (后续) pre_translate.
                # 先收集全部帧 MANO 参考点 (坐标变换后), 平滑关节序列, 再逐帧解算夹爪;
                # 旧实现"重定向后才平滑夹爪轨迹"已废弃.
                s = self.side
                mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": [],
                                        "hand_pose": [], "joints": []}
                # v51 (2026-09-08): MINT 参数域平滑 (主选, _SMOOTH_MINT=1)。
                #   final_clean 在 FK 之前对 HaWoR 参数(pred_trans/rot/hand_pose/betas)平滑，
                #   相机感知用 self._c2w_all；全帧 valid 时按帧号对齐返回同长度，否则回退原 SG。
                _MR = _MP = _MT = None
                if _USE_MINT:
                    _sl = slice(self.start_frame, self.start_frame + self.num_frames)
                    _valid_sl = np.asarray(hawor_data["pred_valid"])[_sl]
                    if _valid_sl.all():
                        _c2w = None
                        if getattr(self, "_c2w_all", None) is not None and self._c2w_all[0] is not None:
                            _c2w = np.concatenate(
                                [self._c2w_all[0][_sl], self._c2w_all[1][_sl][..., None]], axis=-1)
                        _cle = _mint_final_clean(
                            [np.stack([hawor_data["pred_trans"][_sl]] * 2, axis=0),
                             np.stack([hawor_data["pred_rot"][_sl]] * 2, axis=0),
                             np.stack([hawor_data["pred_hand_pose"][_sl]] * 2, axis=0),
                             np.stack([hawor_data["pred_betas"][_sl]] * 2, axis=0),
                             np.stack([np.asarray(hawor_data["pred_valid"])[_sl]] * 2, axis=0)],
                            cam_c2w=_c2w,
                        )
                        _MT = np.asarray(_cle[0][0], np.float64)
                        _MR = np.asarray(_cle[1][0], np.float64)
                        _MP = np.asarray(_cle[2][0], np.float64)
                        logger.info(f"  [preprocess] {s}: MINT 参数域平滑 (UKF+RTS, "
                                    f"相机感知={_c2w is not None}) → FK 前, {_MT.shape[0]} 帧")
                _joints_seq = []
                _hand_pose_seq = []
                for _k, fi in enumerate(range(self.start_frame, self.start_frame + self.num_frames)):
                    if fi < len(hawor_data["pred_valid"]) and not hawor_data["pred_valid"][fi]:
                        continue
                    if fi >= len(hawor_data["pred_trans"]):
                        continue
                    _rot = _MR[_k] if _MR is not None else hawor_data["pred_rot"][fi]
                    _pose = _MP[_k] if _MP is not None else hawor_data["pred_hand_pose"][fi]
                    _trans = _MT[_k] if _MT is not None else hawor_data["pred_trans"][fi]
                    try:
                        _, joints = compute_mano_joints(mano_layer, _rot, _pose, _trans)
                        _joints_seq.append(_mano_to_sapien(joints, fi))
                        _hand_pose_seq.append(_pose)
                    except Exception:
                        continue
                if _joints_seq:
                    _joints_arr = np.array(_joints_seq, dtype=np.float64)  # (N,21,3)
                    mano_gripper_traj[s]["joints_raw"] = _joints_arr.copy()  # 平滑前原始 (轨迹图 raw 用)
                    if _USE_MINT and _MR is not None:
                        # MINT 已在参数域平滑, 直通不再叠加 SG (避免双重平滑)
                        _sm = _joints_arr
                        logger.info(f"  [preprocess] {s}: MINT 参数域平滑生效, 跳过二次 SG")
                    else:
                        _sm = smooth_joints_seq(_joints_arr)  # v51 §1.4: ①SG + ③min-jerk (重定向前)
                    logger.info(f"  [preprocess] {s}: MANO 关节序列平滑 {_joints_arr.shape} "
                                f"(SG(17,2)+min-jerk, 重定向前, v51 §1)")
                    for _i in range(len(_sm)):
                        joints_sapien = _sm[_i]
                        if self.grasp_pose_method == "svd_palm":
                            root_pos, root_R, j1, j2 = compute_mano_based_gripper_pose(
                                joints_sapien, prefix=s,
                            )
                        else:
                            root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
                                joints_sapien[0, :3], joints_sapien[4, :3],
                                joints_sapien[8, :3], prefix=s,
                            )
                        # v51 §1.3 修正 (2026-09-05, 用户要求): 开合对齐 MANO 拇指-食指
                        # 开合 — 在真实尺度计算 (缩放系下 finger_dist/2 恒 ≥0.05 触发上限,
                        # 开合恒 0.05 不跟随); URDF gap=0.01+2q → q=(gap-0.01)/2
                        _fd_real = float(np.linalg.norm(
                            joints_sapien[8, :3] - joints_sapien[4, :3])) / max(self._mano_xform_s, 1e-6)
                        j1 = j2 = float(np.clip((_fd_real - 0.01) / 2.0, 0.0, 0.05))
                        mano_gripper_traj[s]["pos"].append(root_pos)
                        mano_gripper_traj[s]["R"].append(root_R)
                        mano_gripper_traj[s]["j1"].append(j1)
                        mano_gripper_traj[s]["j2"].append(j2)
                        mano_gripper_traj[s]["hand_pose"].append(_hand_pose_seq[_i])
                        # 2026-09-06 (用户: 初始位置=F0 且无跳变): base 初始化/放置改用
                        # 轨迹 anchor (与驱动目标同源) — 旧用腕点 j0, 与轨迹差 7-25cm
                        wrist_positions.append(root_pos)
                        # v51 §1.2/§1.5: 保存 MANO 参考点 (平滑后, 21 关节 j0腕/j4拇指尖/j8食指尖)
                        mano_gripper_traj[s]["joints"].append(joints_sapien)
            # 转为 numpy 数组, 存储到实例 (供 hybrid 中和态使用)
            for key in mano_gripper_traj:
                mano_gripper_traj[key]["pos"] = np.array(mano_gripper_traj[key]["pos"], dtype=np.float64)
                mano_gripper_traj[key]["R"] = np.array(mano_gripper_traj[key]["R"], dtype=np.float64)  # (N, 3, 3)
                mano_gripper_traj[key]["j1"] = np.array(mano_gripper_traj[key]["j1"], dtype=np.float64)
                mano_gripper_traj[key]["j2"] = np.array(mano_gripper_traj[key]["j2"], dtype=np.float64)
                mano_gripper_traj[key]["hand_pose"] = np.array(mano_gripper_traj[key]["hand_pose"], dtype=np.float64)
                if mano_gripper_traj[key]["joints"]:
                    # (N, 21, 3) MANO 参考点 (GLB 系, 未平移/未平滑 — 预处理可视化用)
                    mano_gripper_traj[key]["joints"] = np.array(
                        mano_gripper_traj[key]["joints"], dtype=np.float64)
                # 2026-09-06: 保重力连续性 — 帧间 closing 轴 (Y) 点积 <0 时绕 X 翻 180°
                # (Y,Z 反号, X 不动 → 重力朝向保持)。旧版 4 组合自由翻转会把 approach 轴
                # 翻回朝上 (实测 84/113 帧 Xz>0); 掌面法向在 n_ref 判据边界处会水平反转
                # (点积=-1.0), 此修复消除视频"机械臂翻转"且不破坏重力朝向。
                _R_seq = np.asarray(mano_gripper_traj[key]["R"], dtype=np.float64)
                _n_flip = 0
                for _fi in range(1, len(_R_seq)):
                    if float(_R_seq[_fi][:, 1] @ _R_seq[_fi - 1][:, 1]) < 0.0:
                        _R_seq[_fi] = _R_seq[_fi] @ np.diag(np.array([1.0, -1.0, -1.0]))
                        _n_flip += 1
                mano_gripper_traj[key]["R"] = _R_seq
                if _n_flip:
                    logger.info(f"  [preprocess] {key}: 修正 closing 轴反转 {_n_flip} 帧 (绕X 180°, 保重力)")
            self._mano_gripper_traj = mano_gripper_traj
            # v6.9b: 标记"已准备好" — 后续 run() (如 stage0 双重 run) 复用之,
            # 不再重建, 以免覆盖 pre_translate 烘进 traj["pos"] 的平移 (P1-9).
            self._mano_traj_prepared = True
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
        # 2026-09-12: cam 第一人称相机焦距来源对照开关 (默认关, 不影响已验证配置).
        # 问题: hawor_data["img_focal"] / est_focal.txt = 600 是 **HaWoR 自己视频** 的焦距;
        #   而 RAS 重建的真实内参是 fx=284.56, fy=286.61, cx=344, cy=192 (图像 688x384,
        #   正好等于原始第一人称视频尺寸). 两者分辨率不同, 直接混用属单位不匹配 ->
        #   视场角偏大约 32%, 画面里物体显得比原始视频小.
        # 开启 _CAM_FOCAL_FROM_RAS=1 时按渲染宽度把 RAS 内参 fx 换算过来:
        #   284.56 * (CAM_WIDTH / 688) ≈ 794 @1920 宽.
        if os.environ.get("_CAM_FOCAL_FROM_RAS", "0") == "1":
            _intr_path = Path(self.ras_dir) / "intrinsic.txt"
            _ras_fx = None
            _ras_w = None
            if _intr_path.exists():
                try:
                    _m = np.loadtxt(str(_intr_path), dtype=np.float64)
                    _ras_fx = float(_m[0, 0])
                    _ras_w = float(_m[0, 2]) * 2.0  # cx*2 = 图像宽度
                except Exception as _e:
                    logger.warning(f"  [cam focal] 读取 RAS 内参失败 ({_e}), 回退 img_focal")
            if _ras_fx and _ras_w and _ras_w > 0:
                focal = _ras_fx * (float(CAM_WIDTH) / _ras_w)
                logger.info(
                    f"  cam 焦距: RAS 内参 fx={_ras_fx:.2f}@{_ras_w:.0f} → "
                    f"换算 {CAM_WIDTH} 宽 = {focal:.1f} (原 img_focal={hawor_data.get('img_focal', HAWOR_FOCAL_DEFAULT)})"
                )
        render_cam = render_available and self.views in ("cam", "both", "all")
        render_god = render_available and self.views in ("god", "both", "god_right", "all")
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
                god_height = 0.32  # 上方 0.32m (用户反馈"太近了稍微远一点": 0.20→0.32)
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

        # 7c. 右视角 (沿 +Y 轴看向抓取区, Z-up, 近距) — 用户要求: 右视图/沿Y投影/近一点
        right_view = None
        self._right_view = None
        render_right = render_available and self.views in ("right", "both", "god_right", "all")
        if render_right:
            right_view = self.scene.add_camera(
                "right_view", CAM_WIDTH, CAM_HEIGHT, 1.0, 0.01, 100.0)
            # 目标 = 物体 SAPIEN 位置 (obj_actors 已在场景), 找不到则退回 base
            if getattr(self, 'obj_actors', None):
                _oc = np.mean([np.asarray(a.get_pose().p) for a in self.obj_actors], axis=0)
            elif self._base_pos is not None:
                _oc = np.array(self._base_pos, dtype=float)
            else:
                _oc = np.zeros(3)
            _D = 1.2  # 沿 +Y 距离 (用户 2026-09-05: "右视图离远一点": 0.85→1.2)
            right_pos = np.array([_oc[0], _oc[1] + _D, _oc[2] + 0.18])
            right_look = np.array([_oc[0], _oc[1], _oc[2] + 0.06])
            right_quat = make_look_at_camera(right_pos, right_look, up=np.array([0.0, 0.0, 1.0]))
            right_view.set_local_pose(sapien.Pose(right_pos.tolist(), right_quat.tolist()))
            # v4.28: ×2.8 -> ×8 (纯变焦, 画面放大 2.86x): 用户反馈物体太小/太远,
            # v6.3: 焦距 x8 -> x4 (用户: "视角再远一点"), 物体占画面 ~17%, 视野扩大一倍.
            right_view.set_focal_lengths(focal * 4.0, focal * 4.0)
            self._right_view = right_view
            logger.info(f"  右视角: pos={right_pos.round(3)}, 看向={right_look.round(3)} "
                        f"(沿+Y, Z-up, 距{_D}m, 焦距x4)")

        # 7d. 调试坐标轴标记 (2026-08-25 用户要求: 视频中标出 glb 坐标系和相机)
        # RGB 三色长条 = XYZ 轴; glb 标记每帧跟随目标物体, 相机标记静止.
        # v6.0: headless scene (views=none) 必须跳过 — build_kinematic 携带 visual
        # 组件在无 render system 的场景上会半创建成功, 退出时其析构抛 C++ 异常
        # std::terminate (SIGABRT), 曾使 v4-pipeline 优化完成后整体崩溃.
        self._debug_axes_glb = None
        self._debug_axes_obj_actor = None
        try:
            if not getattr(self.scene, "_render_available", True):
                raise RuntimeError("headless scene (views=none), 跳过调试坐标轴标记")
            import sapien.render as _srender

            def _mk_debug_axes(tag, pose, scale):
                b = self.scene.create_actor_builder()
                t = max(scale * 0.06, 0.0015)
                for i, color in enumerate([[1.0, 0.15, 0.15, 0.8],
                                           [0.15, 1.0, 0.15, 0.8],
                                           [0.25, 0.5, 1.0, 0.8]]):
                    half = [t, t, t]
                    half[i] = scale / 2.0
                    lp = [0.0, 0.0, 0.0]
                    lp[i] = scale / 2.0
                    b.add_box_visual(pose=sapien.Pose(lp), half_size=half,
                                     material=_srender.RenderMaterial(base_color=color))
                actor = b.build_kinematic(name=f"_dbg_axes_{tag}")
                actor.set_pose(pose)
                return actor

            # 目标物体 glb 坐标系 (动态跟随)
            _tgt_name = None
            if getattr(self, 'grasp_controllers', None):
                _ctrl = self.grasp_controllers.get(self.side)
                _tgt_name = getattr(_ctrl, 'target_obj', None) if _ctrl else None
            _tgt_actor = None
            for _a in getattr(self, 'obj_actors', []):
                if _tgt_name and _a.get_name() == _tgt_name:
                    _tgt_actor = _a
                    break
            if _tgt_actor is not None:
                self._debug_axes_obj_actor = _tgt_actor
                self._debug_axes_glb = _mk_debug_axes("glb", _tgt_actor.get_pose(), 0.05)
                logger.info(f"[调试] 已添加 glb 坐标系标记 ({_tgt_name}, 5cm)")

            # 相机位姿标记: 第一人称 (逐帧跟随相机), 上帝 / 右侧 (静止)
            self._debug_axes_cam = None
            if render_cam:
                self._debug_axes_cam = _mk_debug_axes(
                    "cam", sapien.Pose(cam_pos.tolist(), cam_quat.tolist()), 0.08)
            if render_god:
                _mk_debug_axes("god", sapien.Pose(god_pos.tolist(), god_quat.tolist()), 0.08)
            if render_right:
                _mk_debug_axes("right", sapien.Pose(right_pos.tolist(), right_quat.tolist()), 0.08)
            logger.info("[调试] 已添加相机位姿标记 (cam/god/right, 8cm)")
        except Exception as _e:
            logger.warning(f"[调试] 坐标轴标记创建失败 (不影响仿真): {_e}")

        # 8. 视频录制 (按 --views 仅创建需要的 writer)
        _video_tag = f"_{self.video_tag}" if self.video_tag else ""
        video_path_cam = str(self._run_dir() / f"cam_view{_video_tag}_{self.mode}_{self.side}.mp4")
        video_path_god = str(self._run_dir() / f"god_view{_video_tag}_{self.mode}_{self.side}.mp4")
        writer_cam = None
        writer_god = None
        if render_cam:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_cam = cv2.VideoWriter(video_path_cam, fourcc, 30, (CAM_WIDTH, CAM_HEIGHT))
            self._writer_cam = writer_cam  # v4.15: 存储供 Stage 3 rollout 录制视频
            self._video_path_cam = video_path_cam  # v4.25: 转码 H.264 用
        if render_god:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_god = cv2.VideoWriter(video_path_god, fourcc, 30, (CAM_WIDTH, CAM_HEIGHT))
            self._writer_god = writer_god  # v4.15: 存储供 Stage 3 rollout 录制视频
            self._video_path_god = video_path_god  # v4.25: 转码 H.264 用
        # 视频录制: 右视角
        video_path_right = str(self._run_dir() / f"right_view{_video_tag}_{self.mode}_{self.side}.mp4")
        writer_right = None
        if render_right:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer_right = cv2.VideoWriter(video_path_right, fourcc, 30, (CAM_WIDTH, CAM_HEIGHT))
            self._writer_right = writer_right
            self._video_path_right = video_path_right

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

        # 运行元数据落盘 (2026-09-11): 必须在下面两个提前 return 之前调用,
        # 否则 --test-stage1 / --test-stage2 路径不会记录任何参数.
        self.dump_run_metadata()

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

        # [修复 v4.26] Warmup 物理可能把目标物体弹飞:
        #   回放初始基座 = 预平移后的 MANO F0 手腕位置, 距离目标物体过近,
        #   夹爪碰撞体与物体重叠, 5 帧 warmup 求解器将物体高速弹出场景
        #   (实测 glb_4 从 (-0.302,-0.494,0.039) 被弹出至 (-0.99,-3.02,0.758)).
        #   因此 warmup 之后必须把物体重新放回初始位姿, 否则后续夹爪锚定在
        #   被弹飞的物体上, 全程 0 接触 (verify total_contacts=0).
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
                    # v6.0 统一物理: 禁用 virtual_lock_targets (每子步 set_qpos+qvel=0 的
                    # 运动学硬锁). 它与 drive 冲突会把 qpos 钉死在锁定值 —— 当年
                    # "drive 完全失效 (K=50000 无响应, qpos_z 卡死 0.092-0.098)" 的真因.
                    # 纯 PD 跟踪质量改由 [neutral] vz_err 诊断评估 (Stage1/2 已验证可行).
                    _virtual_idx_lock = self.robot_info.get("virtual_idx", {})
                    _vlock = None
                    physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                 np.array([]), j1, j2, self.scene,
                                 lock_root_pose=getattr(self, '_close_lock_pose', None))
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
                            joints_dict[s] = _mano_to_sapien(joints, global_idx)
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
                        joints_sapien = _mano_to_sapien(joints, global_idx)
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
                        # v6.0 统一物理: non-opt 路径同样禁用 vlock (纯 PD drive)
                        _vlock_nonopt = None
                        physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                                     arm_target, gripper_t1, gripper_t2, self.scene,
                                     lock_root_pose=getattr(self, '_close_lock_pose', None))
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
            if _diag_tgt is not None:  # v16: 每帧都记录 (之前只记 8 帧保持段, 青线成点)
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
            if render_available and (writer_cam is not None or writer_god is not None or writer_right is not None or viewer_win is not None):
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
                            cam_pos, cam_quat = self._cam_pose_glb(
                                R_c2w_all[global_idx], t_c2w_all[global_idx])
                            cam_view.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                            if getattr(self, '_debug_axes_cam', None) is not None:
                                self._debug_axes_cam.set_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
                        # 相机视角 (第一人称)
                        # 调试标记: glb 坐标系跟随目标物体 (用户要求视频中标出坐标系)
                        if getattr(self, '_debug_axes_glb', None) is not None \
                                and getattr(self, '_debug_axes_obj_actor', None) is not None:
                            try:
                                self._debug_axes_glb.set_pose(
                                    self._debug_axes_obj_actor.get_pose())
                            except Exception:
                                pass
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
                    if writer_right is not None:
                        # 右视角: 固定相机沿 +Y 看向抓取区, 初始化时已 set_local_pose
                        right_view.take_picture()
                        rgb_r = right_view.get_picture("Color")[..., :3]
                        bgr_r = np.ascontiguousarray((np.clip(rgb_r, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        h, w = bgr_r.shape[:2]
                        cv2.rectangle(bgr_r, (0, 0), (w, 40), (0, 0, 0), -1)
                        label = f"F{local_idx+1}/{self.num_frames} | {self.mode} | RightView | C:{n_contacts} | Grasp:{is_grasping}"
                        cv2.putText(bgr_r, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if is_grasping else (255, 255, 255), 2)
                        writer_right.write(bgr_r)
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
            transcode_to_h264(video_path_cam)  # v4.25: mp4v → H.264 (浏览器可播)
        if writer_god is not None:
            writer_god.release()
            logger.info(f"  上帝视角视频已保存: {video_path_god}")
            transcode_to_h264(video_path_god)
        if writer_right is not None:
            writer_right.release()
            logger.info(f"  右视角视频已保存: {video_path_right}")
            transcode_to_h264(video_path_right)

        # v4.29: 完整管线 (Stage1→2→3→回放) 结束, 提升为标准交付物 D4
        if run_main_loop:
            self.collect_deliverable("D4_full_pipeline.mp4")

        # 11. 参数级验证
        self._verify_results(qpos_log, contact_log, obj_pos_log, obj_initial_pos, grasp_states, gripper_pos_log, track_log)

        # v6.0 统一物理验证 (Stage0/主回放): 纯 PD 跟踪误差汇总
        # 基座虚拟关节+手指全部 drive 驱动后, 跟踪误差是真实的 PD 滞后,
        # 用于替代已废除的 set_qpos/vlock 的"完美跟踪"假象.
        _th = getattr(self, '_track_history', [])
        if len(_th) > 0:
            _root_errs = np.array([t['root_err_mm'] for t in _th])
            _j_errs = np.array([max(t['j1_err_mm'], t['j2_err_mm']) for t in _th])
            logger.info(f"[PhysicsValidation][replay] 纯PD跟踪误差: root mean={_root_errs.mean():.1f}mm, "
                        f"root max={_root_errs.max():.1f}mm | finger mean={_j_errs.mean():.1f}mm, "
                        f"finger max={_j_errs.max():.1f}mm (n={len(_th)} 帧)")

        # 12. 保存结果
        qpos_path = str(self._run_dir() / f"grasp_{self.mode}_{self.side}_qpos.npy")
        np.save(qpos_path, np.array(qpos_log))
        logger.info(f"  qpos 已保存: {qpos_path}")

        # === DEBUG 插桩 (opt-sim-divergence): 保存 run() 的轨迹, 对比 verify ===
        np.save(self._run_dir() / "run_obj_z_traj.npy", np.array(_dbg_run_obj_z_traj))
        np.save(self._run_dir() / "run_gripper_z_traj.npy", np.array(_dbg_run_gripper_z_traj))
        np.save(self._run_dir() / "run_obj_xy_traj.npy", np.array(_dbg_run_obj_xy_traj))
        if _dbg_run_first_frame_state is not None:
            import json
            with open(self._run_dir() / "run_first_frame.json", 'w') as _f:
                json.dump(_dbg_run_first_frame_state, _f, indent=2)
            gi1, gi2 = _dbg_run_first_frame_state['gripper_idx1'], _dbg_run_first_frame_state['gripper_idx2']
            rq, rv = _dbg_run_first_frame_state['robot_qpos'], _dbg_run_first_frame_state['robot_qvel']
            logger.info(f"  [DEBUG run] first_frame: gripper_qpos=[{rq[gi1]:.6f},{rq[gi2]:.6f}], "
                        f"gripper_qvel=[{rv[gi1]:.6f},{rv[gi2]:.6f}], "
                        f"obj_pos={_dbg_run_first_frame_state['obj_pos']}, root_pose={_dbg_run_first_frame_state['root_pose']}")
        # === DEBUG 插桩结束 ===

        # 运行元数据落盘 (2026-09-11): 物理参数 + 物体 scale, 供跨运行对比追溯.
        self.dump_run_metadata()

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

        verify_path = str(self._run_dir() / f"grasp_{self.mode}_{self.side}_verify.json")
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

    def run_v4_pipeline(
        self,
        side=None,
        stage3_method="smooth",
        stage2_pregrasp_retreat=None,
        stage2_approach_min_z=0.0,
        stage2_q_near=None,
        stage2_approach_start_m=0.0,
        stage2_approach_mode="mano_blend",
        stage2_axial_transfer_frac=0.35,
        stage2_axial_clearance_m=0.006,
        stage2_approach_time_scale=2.0,
        stage2_carry_time_scale=2.0,
        stage3_carry_time_scale=2.0,
        stage3_release_candidate=90,
    ):
        """Stage 1 → executable Stage 2 → append-only Stage 3.

        Args:
            side: left/right
            stage3_method: 保留兼容；CEM 已禁用，始终执行确定性 bundle 后缀。

        各阶段验收标准 (v4.25):
            Stage 1: 给定抓取位姿能接触并夹住物体 (Stage1b reward / enclose_z)
            Stage 2: 2/3 次通过同一 executable bundle 硬门
            Stage 3A: 重放同一 Stage2 后保持抓持并携带到释放候选
            Stage 3B: 仅在有支撑且低速时张开，结果与 Stage3A 分开报告

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

        # === 统一预处理入口 (v51 §6 U3/U5): smooth → pre_translate → neutral_offsets ===
        # 这样 neutral offset 接近 0, Stage3 加 offset 到全部帧时不会大幅扭曲轨迹;
        # Stage 1 需要 MANO F50 对齐到目标物体的世界坐标
        self._preprocess_mano_traj(side)

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
            # v6.7: 传运行根目录, 避免子实例在 stage1/ 下再建 alignment/ 与嵌套 stageN/
            root_output_dir=str(self.root_output_dir),
            num_frames=self.num_frames,
            start_frame=self.start_frame,
            views="none",  # Stage 1 子实例不需要渲染
            grasp_mode="hybrid",
            viewer=False,
        )
        stage1_sim.run(run_main_loop=False)
        # v6.0: 保活引用, 防止 run_v4_pipeline 返回时 stage1_sim 被 GC 回收 —
        # 其无渲染 headless scene 的 __del__ 会触发 SAPIEN std::terminate (SIGABRT),
        # 曾导致 v4-pipeline 优化完成后渲染回放阶段整体崩溃.
        self._v4_stage1_sim = stage1_sim
        # v4.25 修复双重平移: 子实例必须与主实例同样先 pre_translate.
        # 原因: 主实例轨迹已在 L2845 平移, 而子实例加载的是原始 MANO 轨迹;
        # 若子实例直接 _compute_neutral_offsets 会算出非零 offset(≈pre_offset),
        # 被 L2875 拷回主实例后与已平移轨迹叠加 → 双重平移, 手被推到物体对面
        # ~66cm 处 (Stage3 obj-mano mean=746mm, lift 仅 1.2cm 的根因).
        # 统一约定: 全管线以"平移后轨迹"为基准, offset≈0, 原始轨迹不再使用.
        stage1_sim._preprocess_mano_traj(side)  # v51 U3/U5: 与主实例同一预处理序列
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
        # v4.25 验收: Stage 1 标准 = 接触并夹住 (reward 含接触/夹持/抬升项)
        _s1_reward = float(best_grasp.get('reward', float('nan')))
        _s1_enclose = best_grasp.get('enclose_z')
        logger.info(f"[v4 pipeline][验收] Stage 1 reward={_s1_reward:.3f}, "
                    f"enclose_z={_s1_enclose if _s1_enclose is None else round(float(_s1_enclose), 4)} "
                    f"(标准: Stage1b 物理仿真中接触+夹持成功)")

        # Stage 2: 预处理 MANO + Stage1 F50 相对校正的平滑可执行时间线
        logger.info("=" * 60)
        logger.info("[v4 pipeline] Stage 2: MANO 保真平滑校正 + F50 快抓")
        logger.info("=" * 60)
        recon = self.reconstruct_trajectory(
            best_grasp,
            side,
            F50_IDX=F50,
            f_start=38,
            f_end=62,
            pregrasp_retreat=stage2_pregrasp_retreat,
            approach_min_z=stage2_approach_min_z,
            q_near=stage2_q_near,
            approach_start_distance=stage2_approach_start_m,
            approach_mode=stage2_approach_mode,
            axial_transfer_frac=stage2_axial_transfer_frac,
            axial_clearance_m=stage2_axial_clearance_m,
        )
        if recon is None:
            logger.error("[v4 pipeline] Stage 2 失败")
            return

        # Stage2 不再以“成功构造 sparse recon”作为通过。主管线执行与
        # --test-stage2 相同的 executable bundle 和 2/3 物理硬门；失败立即停止，
        # 不能让 Stage3 在一个未建立抓持的初态上继续优化。
        if not self.robot_info.get("virtual_idx"):
            logger.error("[v4 pipeline] Stage2 物理 gate 需要 gripper_only virtual joints；"
                         "当前实例不满足，停止在 Stage2")
            return
        stage2_results = [
            self.rollout_follow_trajectory(
                recon,
                side,
                F50_IDX=F50,
                approach_time_scale=stage2_approach_time_scale,
                carry_time_scale=stage2_carry_time_scale,
                record_video=False,
            )
            for _ in range(3)
        ]
        stage2_passed = [
            result for result in stage2_results
            if result.get("gate_result", {}).get("passed", False)
        ]
        if len(stage2_passed) < 2:
            failures = [
                result.get("gate_result", {}).get("failure") for result in stage2_results
            ]
            logger.error(f"[v4 pipeline] Stage2 repeat gate FAIL: "
                         f"{len(stage2_passed)}/3, failures={failures}; 停止进入 Stage3")
            return
        stage2_passed[0]["gate_result"]["repeat_pass_count"] = len(stage2_passed)
        stage2_passed[0]["gate_result"]["repeat_count"] = 3
        self._v4_stage2_bundle = stage2_passed[0]["bundle"]
        self._v4_stage2_result = stage2_passed[0]
        stage2_path = Path(self.output_dir) / "stage2" / "stage2_bundle.npz"
        self.save_stage2_bundle(stage2_path, stage2_passed[0])
        logger.info(f"[v4 pipeline] Stage2 repeat gate PASS: {len(stage2_passed)}/3, "
                    f"bundle={self._v4_stage2_bundle['bundle_hash'][:12]}")

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
        # F91~N 完全跟随 MANO (v51 §8.6.3: fixed = F38-F62 ∪ F91+; F90 归释放逻辑)
        for f in range(91, N):
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
        new_fixed = sorted(set(fp['fixed_frames']) | set(recon['frames']) | set(range(91, N)))
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

        # Stage3 的合法输入是上面通过 repeat gate 的 Stage2 bundle，而不是 sparse
        # recon 或一个手工重置出的抓取状态。rollout 会再次从共享 reset 完整重放
        # 同一 bundle；只有 formal replay PASS 才从实际末态追加携带/释放后缀。
        logger.info("=" * 60)
        logger.info("[v4 pipeline] Stage 3: Stage2 replay → Stage3A carry → conditional release")
        logger.info("=" * 60)
        if stage3_method == "cem":
            logger.warning("[v4 pipeline] --stage3-method=cem 已禁用：旧 CEM 样本不影响命令；"
                           "本次执行确定性 Stage3 后缀")
        stage3_sim = self.rollout_v4_stage3(
            None,
            side,
            carry_time_scale=stage3_carry_time_scale,
            release_candidate=stage3_release_candidate,
            record_video=False,
        )
        best_stage3_reward = float(stage3_sim.get('reward', -float('inf')))
        best_stage3_params = np.zeros(0)
        gate_a = stage3_sim.get("stage3a_gate", {})
        gate_b = stage3_sim.get("stage3b_gate", {})
        logger.info(
            "[v4 pipeline][Stage3A] %s failure=%s detail=%s",
            "PASS" if gate_a.get("passed", False) else "FAIL",
            gate_a.get("failure"),
            gate_a.get("failure_detail", ""),
        )
        logger.info(
            "[v4 pipeline][Stage3B] %s failure=%s detail=%s",
            "PASS" if gate_b.get("passed", False) else "FAIL",
            gate_b.get("failure"),
            gate_b.get("failure_detail", ""),
        )
        stage3_path = Path(self.output_dir) / "stage3" / "combined_bundle.npz"
        if not self.save_stage3_bundle(stage3_path, stage3_sim):
            logger.error("[v4 pipeline] Stage3A 未通过，不生成组合 bundle，也不进入旧式参数回放")
            return None

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
        np.save(self._opt_dir() / "v4_opt_params.npy", full_params)
        np.save(self._opt_dir() / "v4_fixed_offsets.npy",
                {str(k): v for k, v in fixed_offsets.items()}, allow_pickle=True)
        np.save(self._opt_dir() / "frame_params.npy", fp)
        logger.info(f"[v4 pipeline] 完成. Stage3 reward={best_stage3_reward:.3f}, "
                    f"params shape={full_params.shape}")

        # 设置参数供 run() 主循环回放
        self.set_opt_params(full_params)
        logger.info("[v4 pipeline] 有效交付物为 stage3/combined_bundle.npz；"
                    "旧参数仅保留可视化兼容，不作为物理验收依据")
        self._v4_stage3_result = stage3_sim
        return stage3_sim

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
                    wp_path = self._opt_dir() / "window_frames.npy"
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
            ground_z = compute_glb_ground_z(glb_path, self.transform_params_path,
                                            use_glb_coords=self.glb_coords)
        # 第十八轮: 优化模式强制 CPU 场景, 避免渲染初始化段错误
        self.scene = setup_physics_scene(ground_height=ground_z, force_cpu=True)
        render_available = getattr(self.scene, "_render_available", False)

        # 加载 GLB 物体
        self.obj_actors, _, self.obj_bbox_centers, self.obj_info = load_glb_with_physics(
            glb_path, self.transform_params_path, self.scene, fast_collision=True,
            use_glb_coords=self.glb_coords
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
        # v5.2: 与 render_quick.py cam_traj_to_glb 一致的数据驱动链 (用户确认正确)
        def _mano_to_sapien_v4(pts_cam, frame_idx=0):
            """MANO 相机系 → SAPIEN/GLB 世界系.

            Step1: 逐帧真实相机位姿提升到 HaWoR 世界系
                   p_hw = R_c2w[f] @ p_cam + t_c2w[f]
                   (不用固定 Rx_hand=diag(1,-1,-1): 001_align_scene 实验证实 R_X=I)
            Step2: 001 相似变换 → GLB (= SAPIEN 世界, glb_coords=True)
                   p_glb = s * R_hand_to_glb @ p_hw + t_hand_to_glb
            """
            s = self._mano_xform_s
            R_hand = self._mano_xform_R_hand  # = R_h2g (R_X=I)
            t = self._mano_xform_t
            if R_c2w_all is not None and len(R_c2w_all) > 0:
                fi = min(max(int(frame_idx), 0), len(R_c2w_all) - 1)
                pts_hw = (R_c2w_all[fi] @ np.asarray(pts_cam).T).T + t_c2w_all[fi]
            else:
                pts_hw = np.asarray(pts_cam)
            pts_glb = s * (R_hand @ pts_hw.T).T + t
            if getattr(self, 'glb_coords', False):
                _out = pts_glb
            else:
                _out = (R_AXIS @ pts_glb.T).T
            # v6.9b: 上游修正 MANO 手姿态 (与 run() 内同一约定); 默认关闭, 见 __init__ 注释
            return flip_mano_hand_down(_out) if self.flip_mano_hand else _out
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
                joints_sapien = _mano_to_sapien_v4(joints, fi)
                wrist_positions.append(joints_sapien[0, :3])
                s = self.side
                if s not in mano_gripper_traj:
                    mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": [], "hand_pose": []}
                if self.grasp_pose_method == "svd_palm":
                    root_pos, root_R, j1, j2 = compute_mano_based_gripper_pose(
                        joints_sapien, prefix=s,
                    )
                else:
                    root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(
                        joints_sapien[0, :3], joints_sapien[4, :3],
                        joints_sapien[8, :3], prefix=s,
                    )
                mano_gripper_traj[s]["pos"].append(root_pos)
                mano_gripper_traj[s]["R"].append(root_R)
                mano_gripper_traj[s]["j1"].append(j1)
                mano_gripper_traj[s]["j2"].append(j2)
                mano_gripper_traj[s]["hand_pose"].append(hawor_data["pred_hand_pose"][fi])
            except Exception:
                continue
        for key in mano_gripper_traj:
            mano_gripper_traj[key]["pos"] = np.array(mano_gripper_traj[key]["pos"], dtype=np.float64)
            mano_gripper_traj[key]["R"] = np.array(mano_gripper_traj[key]["R"], dtype=np.float64)
            mano_gripper_traj[key]["j1"] = np.array(mano_gripper_traj[key]["j1"], dtype=np.float64)
            mano_gripper_traj[key]["j2"] = np.array(mano_gripper_traj[key]["j2"], dtype=np.float64)
            mano_gripper_traj[key]["hand_pose"] = np.array(mano_gripper_traj[key]["hand_pose"], dtype=np.float64)
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

        # 预先平移: 把整条 MANO 轨迹整体平移, 让 F50 手指中点接近物体
        self._pre_translate_mano_trajectory(self.side)

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
            # 预先平移: 把整条 MANO 轨迹整体平移, 让 F50 手指中点接近物体
            self._pre_translate_mano_trajectory(self.side)
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
                # v6.0 统一物理: 禁用 vlock (回放路径同样纯 PD drive)
                _vlock_rs = None
                physics_step(
                    robot, arm_joint_indices, gi1, gi2,
                    np.asarray(arm_target) if len(arm_target) else np.array([]),
                    float(gripper_t1), float(gripper_t2),
                    self.scene,
                    lock_root_pose=None,
                    decimation=decimation,
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
                            # === 每帧 DIAG (L3550 修复: 之前只记 F50-F108 保持段, 青线成点) ===
                            # 取手指 link 世界位置, 计算指尖中点 fmid
                            _f1p = _f2p = None
                            for lnk in robot.get_links():
                                _n = lnk.get_name()
                                if _n == f"{self.side}_gripper_finger_link1":
                                    _f1p = np.array(lnk.get_entity_pose().p)
                                elif _n == f"{self.side}_gripper_finger_link2":
                                    _f2p = np.array(lnk.get_entity_pose().p)
                            _fmid = 0.5 * (_f1p + _f2p) if (_f1p is not None and _f2p is not None) else self._current_gripper_pos
                            logger.warning(
                                f"  [DIAG F{local_idx}] obj={obj_pos.round(4)}, "
                                f"f1={_f1p.round(4) if _f1p is not None else 'N/A'}, "
                                f"f2={_f2p.round(4) if _f2p is not None else 'N/A'}, "
                                f"base={self._current_gripper_pos.round(4)}"
                            )
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
                                # v16: 每帧都记录 (之前只记 F50-F108 保持段, 导致青线成点)
                                if True:
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
                                    # v4.50: 夹持力诊断 (对齐 Stage1 基准 9.5N)
                                    try:
                                        _gf = get_grasp_force(self.side, self.scene, self.obj_actors, robot)
                                        _force_info = f", force={_gf:.1f}N"
                                    except Exception:
                                        _force_info = ""
                                    logger.warning(f"  [DIAG F{local_idx}] obj={obj_pos.round(4)}, "
                                                    f"f1={(_f1_pos.round(4) if _f1_pos is not None else 'N/A')}, "
                                                    f"f2={(_f2_pos.round(4) if _f2_pos is not None else 'N/A')}, "
                                                    f"base={self._current_gripper_pos.round(4)}{_side_info}{_y_gap_info}{_z_relation}{_glink_info}{_qpos_info}{_force_info}")
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
        np.save(self._opt_dir() / "opt_params.npy", best_params)
        np.save(self._opt_dir() / "reward_history.npy", np.array(reward_history))
        np.save(self._opt_dir() / "baseline_dists.npy", dist_array)

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
        np.save(self._run_dir() / "verify_reward.npy",
                np.array([best_reward, verify_reward, abs(best_reward - verify_reward)]))
        # === DEBUG 插桩 v2: 保存 verify rollout 的完整轨迹 + 第一帧状态 ===
        np.save(self._run_dir() / "verify_obj_z_traj.npy", np.array(verify_result.get('_dbg_obj_z_traj', [])))
        np.save(self._run_dir() / "verify_gripper_z_traj.npy", np.array(verify_result.get('_dbg_gripper_z_traj', [])))
        np.save(self._run_dir() / "verify_obj_xy_traj.npy", np.array(verify_result.get('_dbg_obj_xy_traj', [])))
        _ffs = verify_result.get('_dbg_first_frame_state')
        if _ffs is not None:
            import json
            with open(self._run_dir() / "verify_first_frame.json", 'w') as _f:
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
            np.save(self._opt_dir() / "opt_params.npy", best_params)
            # 保存 frame_params 和 fixed_offsets_654 用于回放
            if hasattr(self, '_frame_params') and self._frame_params is not None:
                np.save(self._opt_dir() / "frame_params.npy", self._frame_params)
            if hasattr(self, '_fixed_offsets_654') and self._fixed_offsets_654 is not None:
                _fo_save = {str(k): v for k, v in self._fixed_offsets_654.items()}
                np.save(self._opt_dir() / "fixed_offsets_654.npy", _fo_save, allow_pickle=True)

            # 验证: 用完整 DECIMATION=8 做高精度验证
            logger.info(f"{'='*60}")
            logger.info(f"[MANO] 验证: 完整 DECIMATION=8")
            logger.info(f"{'='*60}")
            verify_result = rollout_single(best_params, decimation=8)
            contact = verify_result.get('contact_frames_in_close', 0)
            lift = verify_result.get('obj_final_z', 0) - verify_result.get('obj_init_z', 0)
            min_dist = verify_result.get('min_dist_in_close', 1.0)
            logger.info(f"  verify: contact={contact}, lift={lift:.4f}, min_dist={min_dist:.4f}")
            np.save(self._run_dir() / "verify_reward.npy",
                    np.array([0.0, verify_result.get('reward', 0.0)]))

    def _cam_pose_glb(self, R_c2w, t_c2w):
        """GLB 直出相机位姿 (2026-09-06 修正): hawor_cam_to_sapien_pose 内含 R_AXIS
        轴置换 (旧链), 与 glb_coords=True 的 GLB 世界不一致 → 相机轨迹/视角错位
        (用户: '坐标系绝对不可能在右上方'). 本方法去掉 R_AXIS, 其余约定一致."""
        pos = self._cam_xform_s * (self._cam_xform_R_h2g @ t_c2w) + self._cam_xform_t
        R_cam_glb = self._cam_xform_R_h2g @ R_c2w
        # SAPIEN 相机约定: col0=forward, col1=left, col2=up
        R_sc = np.eye(3)
        R_sc[:, 0] = R_cam_glb[:, 2]
        R_sc[:, 1] = -R_cam_glb[:, 0]
        R_sc[:, 2] = -R_cam_glb[:, 1]
        from scipy.spatial.transform import Rotation as _R
        _q = _R.from_matrix(R_sc).as_quat()  # [x,y,z,w]
        return pos, np.array([_q[3], _q[0], _q[1], _q[2]])

    def _sync_cam_view_to_frame(self, source_frame: float) -> None:
        """把第一人称 cam 相机同步到指定 MANO 源帧 (逐帧跟随).

        2026-09-12: 主循环 (Stage0) 一直是逐帧更新相机的, 所以 Stage0 的 cam 视频正确;
        但 Stage2/Stage3 的 rollout 有各自独立的渲染函数, 此前只调 camera.take_picture(),
        相机冻结在 init 时设的 start_frame 位姿 → cam 视频看起来像固定俯视机位
        (用户反馈: "cam 视角为什么变成了 god 视角").
        本方法提供统一入口, 供两条 rollout 渲染路径逐帧调用.
        """
        cam_view = getattr(self, "_cam_view", None)
        c2w = getattr(self, "_c2w_all", None)
        if cam_view is None or not c2w or c2w[0] is None:
            return
        R_all, t_all = c2w[0], c2w[1]
        if len(R_all) == 0:
            return
        gi = int(round(float(self.start_frame) + float(source_frame)))
        gi = max(0, min(gi, len(R_all) - 1))
        try:
            cam_pos, cam_quat = self._cam_pose_glb(R_all[gi], t_all[gi])
            cam_view.set_local_pose(sapien.Pose(cam_pos.tolist(), cam_quat.tolist()))
        except Exception as exc:
            logger.warning("[cam sync] source_frame=%s skipped: %s", source_frame, exc)

    def _pre_translate_mano_trajectory(self, side):
        """预先平移: 把整条 MANO 轨迹整体平移, 让 F50 手指中点接近物体中心.

        这样 _compute_neutral_offsets 算出的 offset 接近 0,
        Stage3 加 offset 到全部帧时不会大幅扭曲轨迹.
        """
        traj = self._mano_gripper_traj.get(side)
        if traj is None or len(traj["pos"]) == 0:
            return
        ctrl = self.grasp_controllers.get(side)
        tgt = ctrl.target_obj if ctrl is not None else None
        if tgt is None or tgt not in self.obj_bbox_centers:
            return

        obj_center = np.array(self.obj_bbox_centers[tgt], dtype=np.float64)
        mano_positions = np.asarray(traj["pos"], dtype=np.float64)
        mano_R_traj = np.asarray(traj.get("R", []), dtype=np.float64)

        n_frames = len(mano_positions)
        F50_IDX = min(n_frames - 1, 50)
        _finger_forward = np.array([0.03689, 0.0, 0.0])
        _R_f50 = mano_R_traj[F50_IDX] if len(mano_R_traj) > F50_IDX else np.eye(3)
        _finger_mid_mano = mano_positions[F50_IDX] + _R_f50 @ _finger_forward
        pre_offset = obj_center - _finger_mid_mano

        # 把整条轨迹整体平移
        traj["pos"] = mano_positions + pre_offset
        self._mano_gripper_traj[side] = traj
        # v51 §1.5: 记录平移量 — 预处理 dump 时把 MANO 参考点 (joints) 变换到同一平移系
        if not hasattr(self, '_pre_offset'):
            self._pre_offset = {}
        self._pre_offset[side] = pre_offset

        logger.info(f"[pre_translate] {side}: F50 finger_mid={_finger_mid_mano.round(4)}, "
                    f"obj_center={obj_center.round(4)}, "
                    f"pre_offset={pre_offset.round(4)}, "
                    f"|pre_offset|={np.linalg.norm(pre_offset):.4f}m")

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

    def _smooth_mano_traj(self, side):
        """v6.9 (P0-2): 离线平滑 _mano_gripper_traj[side] (pos/R/joint).
        在 pre_translate + _compute_neutral_offsets 之后调用, 让携带/释放段的 MANO 跟随变平滑."""
        if not getattr(self, '_smoothing_enabled', True):
            return
        try:
            ok = smooth_gripper_traj(self._mano_gripper_traj, side)
            if ok:
                logger.info(f"[smooth_mano] {side}: pos/R/joint 已离线平滑 (SG+quatSlerp+min-jerk)")
        except Exception as _e:
            logger.warning(f"[smooth_mano] {side} 平滑失败 (回退原始轨迹): {_e}")

    def _preprocess_mano_traj(self, side):
        """v51 §6 U3/U5 (2026-09-05, 顺序经用户确认): 统一预处理入口 — 所有路径复用同一序列.

        顺序 (v51 §1 设计): 坐标变换 → 平滑 → 重定向(解算夹爪) → pre_translate → neutral_offsets.
        前三步已在 run() 预计算内完成 (smooth_joints_seq 在重定向前平滑 MANO 关节序列),
        此处只做平移对齐与残差计算, 不再对夹爪轨迹二次平滑.
        以 [pre_translate] → offset≈0 日志为验收 (v51 §1.4 P1-1 教训).
        """
        self._check_mano_R_continuity(side)
        self._pre_translate_mano_trajectory(side)
        self._compute_neutral_offsets()
        self._dump_preprocess_traj(side)

    def _check_mano_R_continuity(self, side):
        """校验姿态序列帧间连续性 — **只报警, 不修改轨迹**.

        2026-09-11 (用户定策): align_gripper_approach_to_gravity 已改为 10° 死区判据
        (physics_utils): 向上倾角 <= 10° 一律不翻 (以帧间平滑为优先), 只有明确朝上
        (>10°) 才翻成朝下. 因此姿态序列不该再出现 180° 台阶.
        本函数只做独立校验: 若仍有 >90° 的帧间跳变, 说明预处理仍有问题 → 报警.
        """
        traj = self._mano_gripper_traj.get(side)
        if traj is None or len(traj.get("R", [])) == 0:
            return
        R = np.asarray(traj["R"], dtype=np.float64)
        if len(R) < 2:
            return

        def _rd(A, B):
            tr = float(np.clip(np.trace(A @ B.T), -1.0, 3.0))
            return float(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))))

        d = np.array([_rd(R[i], R[i + 1]) for i in range(len(R) - 1)])
        bad = [int(i) for i in np.where(d > 90.0)[0]]
        xz = R[:, 0, 2]
        logger.info(
            "[R_continuity] %s: 帧间姿态变化 mean=%.2f max=%.2f deg; X 朝下占比 %.0f%%; "
            ">90° 帧对 %s",
            side, d.mean(), d.max(), 100.0 * float(np.mean(xz < 0)),
            bad if bad else "无",
        )
        if bad:
            logger.warning(
                "[R_continuity] %s 仍有 180° 姿态台阶 @ %s — 预处理需检查", side, bad
            )

    def _dump_preprocess_traj(self, side):
        """v51 §1.5 (2026-09-05): 预处理产物 dump — 供 View/vis_preprocess_trajectories.py 出对齐轨迹图.

        保存: MANO 参考点 joints (N,21,3, 已加 pre_offset 与夹爪轨迹同系) + 重定向后夹爪
        轨迹 pos/R/j1/j2 (平滑+平移后) + obj_center. 输出 <output_dir>/preprocess_traj.npz.
        """
        try:
            traj = self._mano_gripper_traj.get(side)
            if traj is None or len(traj.get("pos", [])) == 0:
                return
            joints = traj.get("joints")
            data = {
                "side": side,
                "grasp_pose_method": getattr(self, 'grasp_pose_method', 'unknown'),
                "grip_pos": np.asarray(traj["pos"], dtype=np.float64),
                "grip_R": np.asarray(traj["R"], dtype=np.float64),
                "j1": np.asarray(traj.get("j1", []), dtype=np.float64),
                "j2": np.asarray(traj.get("j2", []), dtype=np.float64),
            }
            if joints is not None and len(joints) == len(data["grip_pos"]):
                # 参考点加同一 pre_offset, 与夹爪轨迹同处"平移后"坐标系
                _off = getattr(self, '_pre_offset', {}).get(side, np.zeros(3))
                data["joints"] = np.asarray(joints, dtype=np.float64) + np.asarray(_off)
                # 平滑/平移前的原始重定向轨迹 (v51 §1.5: 优化前后对比)
                _jraw = traj.get("joints_raw")
                if _jraw is not None and len(_jraw) == len(data["grip_pos"]):
                    data["joints_raw"] = np.asarray(_jraw, dtype=np.float64)
                data["pre_offset"] = np.asarray(_off, dtype=np.float64)
            ctrl = self.grasp_controllers.get(side)
            tgt = ctrl.target_obj if ctrl is not None else None
            if tgt is not None and tgt in self.obj_bbox_centers:
                data["obj_center"] = np.asarray(self.obj_bbox_centers[tgt], dtype=np.float64)
                data["target_obj"] = tgt
            # 相机轨迹 (GLB 系, 逐帧) — 轨迹图用 (2026-09-06)
            try:
                if getattr(self, '_c2w_all', None) is not None:
                    _Rc, _tc = self._c2w_all
                    _N = len(data["grip_pos"])
                    _cam_pts = [self._cam_pose_glb(
                        _Rc[min(self.start_frame + _i, len(_Rc) - 1)],
                        _tc[min(self.start_frame + _i, len(_tc) - 1)])[0]
                        for _i in range(_N)]
                    data["cam_traj"] = np.asarray(_cam_pts, dtype=np.float64)
            except Exception as _e:
                logger.warning(f"[preprocess_dump] 相机轨迹采集失败: {_e}")
            out_path = self.output_dir / "preprocess_traj.npz"
            np.savez(out_path, **data)
            logger.info(f"[preprocess_dump] {side}: {data['grip_pos'].shape[0]} 帧 → {out_path}")
        except Exception as _e:
            logger.warning(f"[preprocess_dump] {side} 失败 (不影响管线): {_e}")


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
    parser.add_argument("--ras-dir", type=str, default="/home/an/data/ras/7_vggt_omega",
                        help="RAS 场景根目录; 统一使用 7_vggt_omega 场景 (v16: 不再默认 my_7mp4_result)")
    parser.add_argument("--output-dir", type=str, default=None,
                            help="输出目录; 默认按输入文件自动命名: <hawor名>_<ras名>_grasp_<mode>_<side>/")
    parser.add_argument("--num-frames", type=int, default=-1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--views", type=str, default="both",
                        choices=["cam", "god", "both", "none", "right", "god_right", "all"],
                        help="渲染视角: cam(SLAM第一人称) / god(上帝俯视) / both(cam+god) / right(右视) / god_right(上帝+右视) / all(三视角齐录) / none")
    parser.add_argument("--viewer", action="store_true",
                        help="启用交互式 Viewer 窗口 (实时查看仿真过程)")
    parser.add_argument("--grasp-mode", type=str, default="hybrid",
                        choices=["adaptive", "mano", "hybrid"],
                        help="抓取模式: hybrid(MANO驱动+接触力控, 默认) / adaptive(旧版MANO意图) / mano(纯MANO重放)")
    parser.add_argument("--grasp-pose-method", type=str, default="svd_palm",
                        choices=["analytical", "svd_palm"],
                        help="夹爪位姿解算: svd_palm(5点SVD手掌平面+Gram-Schmidt, 默认, v51 U1) / analytical(加权SVD, 仅作对照保留)")
    parser.add_argument("--test-stage0", action="store_true",
                        help="v6.5: Stage 0 纯 MANO 重定向测试 (无优化, 验证管线正确性)")
    parser.add_argument("--test-stage1", action="store_true",
                        help="v4.4: 仅运行 Stage 1 单次 rollout 测试 (不渲染完整仿真)")
    parser.add_argument("--test-stage1-transfer", action="store_true",
                        help="v51: F50_TRANSFER_EQUIVALENCE — 用 Stage1 recipe 在规范入口 "
                             "(q=0.05) 重放 20+10, 3 次独立 reset 验证 anchor 可迁移")
    parser.add_argument("--f50-tilt-deg", type=float, default=0.0,
                        help="外部附加旋转角: MANO F50 姿态绕世界Y轴旋转(度). 0=关闭(默认); -90=手指倾斜向下")
    parser.add_argument("--test-stage2", action="store_true",
                        help="v4.6: Stage 1 + Stage 2 轨迹重建测试")
    parser.add_argument("--stage2-pregrasp-retreat", type=float, default=None,
                        help="已废止兼容参数；当前F38→F50不再使用retreat")
    parser.add_argument("--stage2-approach-clearance-z", type=float, default=None,
                        help="已废止兼容参数；当前F38→F50不再建立世界Z抬升走廊")
    parser.add_argument("--stage2-approach-min-z", type=float, default=0.0,
                        help="F38→F50平滑MANO校正的世界Z下界(米)，默认0；只在命令z低于该值时修正")
    parser.add_argument("--stage2-approach-start-m", type=float, default=0.0,
                        help="距离驱动切换点(米): 夹爪基座离目标中心 >= 该值的最后一帧才开始朝 "
                             "Stage1 anchor 收敛; 0 = 沿用固定 f_start=38")
    parser.add_argument("--stage2-approach-mode", choices=("mano_blend", "axial"),
                        default="mano_blend",
                        help="A_start→F50 逼近路径: mano_blend=MANO平滑混合(默认); "
                             "axial=从物体位姿反解的轴向逼近(物体只从虎口正面进入)")
    parser.add_argument("--stage2-axial-transfer-frac", type=float, default=0.35,
                        help="axial 模式: 平移到 entry pose 所占路径比例, 其余为纯轴向推进")
    parser.add_argument("--stage2-axial-clearance-m", type=float, default=0.006,
                        help="axial 模式: 物体端面到手指端面的额外间隙(米)")
    parser.add_argument("--stage2-q-near", type=float, default=None,
                        help="已废止兼容参数; 必须保持 None，F50 前夹爪固定全开0.05")
    parser.add_argument("--stage2-approach-time-scale", type=float, default=2.0,
                        help="Stage2 F38→F50 接近时间缩放, >=1, 默认2.0")
    parser.add_argument("--stage2-carry-time-scale", type=float, default=2.0,
                        help="Stage2 F51→F62 携带时间缩放, >=1, 默认2.0; F50→F51固定5个移动确认点")
    parser.add_argument("--test-stage3", action="store_true",
                        help="v4.6: Stage 1 + Stage 2 + Stage 3 全局优化测试")
    parser.add_argument("--stage3-method", type=str, default="smooth",
                        choices=["smooth", "cem"],
                        help="兼容参数: smooth=确定性bundle后缀; cem已禁用并回退到smooth")
    parser.add_argument("--stage3-carry-time-scale", type=float, default=2.0,
                        help="Stage3 F62→释放候选的时间缩放, >=1, 默认2.0")
    parser.add_argument("--stage3-release-candidate", type=int, default=90,
                        help="Stage3 释放候选 MANO 帧, 默认F90; 无支撑时不会张开")
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
    # v6.5: Stage 0 纯 MANO 重定向测试 — 无优化, 验证管线正确性
    # 核心: opt_params=None, 走 _compute_mano_neutral_target 的无优化回退路径,
    # 完整 160 帧, 用 set_drive_target 控制虚拟关节, 验证 MANO 轨迹经过
    # R_hand_to_glb 对齐后夹爪是否朝下.
    if args.test_stage0:
        sim = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode="gripper_only",
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views=args.views,
            grasp_mode="hybrid",
            grasp_pose_method=args.grasp_pose_method,
            viewer=False,
        )
        sim._test_stage1_only = False  # Stage0 需要运行完整仿真循环
        sim.video_tag = "stage0"
        sim.f50_tilt_deg = float(args.f50_tilt_deg)
        sim._align_scene()
        sim.run()
        side = args.side if args.side != "both" else "right"
        sim._preprocess_mano_traj(side)  # v51 U3/U5: smooth → pre_translate → neutral

        logger.info("=" * 60)
        logger.info("v6.5 Stage 0 纯 MANO 重定向测试 (无优化)")
        logger.info("=" * 60)

        # 检查 MANO 夹爪轨迹是否已预计算
        if not hasattr(sim, '_mano_gripper_traj') or sim._mano_gripper_traj is None:
            logger.error("[test-stage0] MANO 夹爪轨迹未预计算, 请检查 run() 输出")
            return

        mano_traj = sim._mano_gripper_traj.get(side, {})
        if not mano_traj or 'pos' not in mano_traj:
            logger.error(f"[test-stage0] 缺少 {side} 侧 MANO 夹爪轨迹")
            return

        mano_pos = mano_traj['pos']
        mano_R = mano_traj['R']
        mano_j1 = mano_traj['j1']
        mano_j2 = mano_traj['j2']
        n_mano = len(mano_pos)
        logger.info(f"[test-stage0] MANO 夹爪轨迹: {n_mano} 帧")

        # 打印 MANO F50 夹爪朝向 (GLB 坐标系)
        F50_IDX = 50
        if F50_IDX < n_mano:
            R_f50 = mano_R[F50_IDX]
            z_axis = R_f50[:, 2]
            logger.info(f"[test-stage0] MANO F50 夹爪朝向 (GLB系):")
            logger.info(f"  R Z-axis (forward): {z_axis.round(4)}")
            logger.info(f"  Z[2]={z_axis[2]:.4f} → {'朝下' if z_axis[2] < 0 else '朝上'}")
            logger.info(f"  pos: {mano_pos[F50_IDX].round(4)}")
            logger.info(f"  j1={mano_j1[F50_IDX]:.4f}, j2={mano_j2[F50_IDX]:.4f}")

        # 运行完整仿真循环 — 直接调用 run(), 它已处理渲染和物理步进
        # opt_params=None 走 _compute_mano_neutral_target 的无优化回退路径
        logger.info("[test-stage0] 开始仿真 (opt_params=None, 纯 MANO 重定向)...")
        sim._opt_params = None  # 确保走无优化路径
        sim._opt_params_side = None
        sim.run()
        logger.info("[test-stage0] Stage 0 完成")
        return

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
            grasp_pose_method=args.grasp_pose_method,  # v51 U1/P1-7: 各入口统一传参
            viewer=False,
        )
        sim._test_stage1_only = True
        sim.video_tag = "stage1"  # v4.16: 视频文件名标签
        sim.f50_tilt_deg = float(args.f50_tilt_deg)  # 外部附加旋转角 (0=关闭)
        sim._align_scene()
        sim.run()  # 初始化场景+机器人+控制器+MANO轨迹
        # v4.5: test-stage1 跳过完整仿真循环, _mano_neutral_offset 未计算,
        # 需显式计算以将 MANO F50 对齐到世界坐标
        # v4.32 统一口径: 与 v4-pipeline (L2959/L2989) 和 test-stage3 一致,
        # 先 pre_translate 平移整条轨迹, 再算残差 offset (≈0).
        # 之前仅靠 offset 机制等效平移 (数值正确但日志无 [pre_translate] 行).
        side = args.side if args.side != "both" else "right"
        sim._preprocess_mano_traj(side)  # v51 U3/U5: smooth → pre_translate → neutral
        logger.info("=" * 60)
        logger.info("v4.5 Stage 1 抓取姿态优化测试")
        logger.info("=" * 60)
        # v4.28: 5x8 -> 8x16. 原配置 CEM 随机性过大 (连续跑出 obj_lift<2cm),
        # 8 迭代 x16 种群更稳健地收敛到 ≥3cm 达标抓取, 保证交付视频质量稳定.
        # v6.1: CEM 预算与 v4-pipeline 统一 (15x32), 终选有 3 次重复验证门
        best_grasp = sim.cem_grasp_pose_optimize(side, n_iterations=15, population_size=32, top_k=6)
        if best_grasp is None:
            logger.error("Stage 1 优化失败")
            return
        logger.info(f"[test-stage1] 最优 grasp_pose: pos={best_grasp['pos'].round(4)}, "
                    f"euler={np.degrees(best_grasp['euler']).round(2)}, "
                    f"gripper_qpos={best_grasp['gripper_qpos']:.4f}, "
                    f"strategy={best_grasp['strategy']}")

        # v4.16: 最终最优位姿的 rollout 视频
        _has_video = (hasattr(sim, '_writer_god') and sim._writer_god is not None) or \
                     (hasattr(sim, '_writer_cam') and sim._writer_cam is not None) or \
                     (hasattr(sim, '_writer_right') and sim._writer_right is not None)
        if _has_video:
            logger.info("[test-stage1] 录制最优 grasp 的 rollout 视频...")
            sim.rollout_grasp_only(best_grasp, side, strategy='pd_then_lock', n_frames=160, record_video=True)
            # 释放视频写入器
            if hasattr(sim, '_writer_cam') and sim._writer_cam is not None:
                sim._writer_cam.release()
                logger.info("[test-stage1] 相机视角视频已保存")
                transcode_to_h264(getattr(sim, '_video_path_cam', ''))
            if hasattr(sim, '_writer_god') and sim._writer_god is not None:
                sim._writer_god.release()
                transcode_to_h264(getattr(sim, '_video_path_god', ''))
                logger.info("[test-stage1] 上帝视角视频已保存")
            if hasattr(sim, '_writer_right') and sim._writer_right is not None:
                sim._writer_right.release()
                transcode_to_h264(getattr(sim, '_video_path_right', ''))
                logger.info("[test-stage1] 右视角视频已保存")
            # v4.29: 提升为标准交付物 D2
            sim.collect_deliverable("D2_stage1_grasp.mp4")
        return

    # v51 (2026-09-10): T1 F50_TRANSFER_EQUIVALENCE — 在规范入口验证 Stage1 anchor
    # 可迁移到 Stage2 的起手状态 (base=Stage1 pos/R, q=0.05, 零速度, 无目标接触),
    # 再用共享 20+10 recipe 重放; 三次独立 reset 至少两次建立并保持双侧抓取.
    if args.test_stage1_transfer:
        sim = GraspSimulator(
            hawor_dir=args.hawor_dir,
            ras_dir=args.ras_dir,
            mode="gripper_only",
            side=args.side,
            output_dir=args.output_dir,
            num_frames=args.num_frames,
            start_frame=args.start_frame,
            views=args.views,
            grasp_mode="hybrid",
            grasp_pose_method=args.grasp_pose_method,
            viewer=False,
        )
        sim._test_stage1_only = True
        sim.video_tag = "stage1_transfer"
        sim.f50_tilt_deg = float(args.f50_tilt_deg)
        sim._align_scene()
        sim.run()
        side = args.side if args.side != "both" else "right"
        sim._preprocess_mano_traj(side)
        logger.info("=" * 60)
        logger.info("v51 T1: F50_TRANSFER_EQUIVALENCE (Stage1 anchor 可迁移性)")
        logger.info("=" * 60)
        _output_dir = args.output_dir or f"output/gripper_{side}"
        stage1_path = Path(_output_dir) / "stage1" / "best_grasp.npz"
        if not stage1_path.exists():
            logger.error(
                "[test-stage1-transfer] 需要先运行 --test-stage1 生成 %s (本入口不重跑 CEM)",
                stage1_path,
            )
            return
        best_grasp, recipe = load_stage1_grasp(stage1_path)
        if recipe is None:
            logger.error(
                "[test-stage1-transfer] STALE_STAGE1_ARTIFACT: %s 缺少 stage1_grasp_recipe; "
                "请用当前代码重跑 --test-stage1",
                stage1_path,
            )
            return
        logger.info(
            "[test-stage1-transfer] Stage1 recipe: target=%s q_open=%.4f q_grasp=%.4f "
            "close/settle=%d/%d digest=%s",
            recipe.get("target_object"), recipe.get("q_open"), recipe.get("q_grasp"),
            recipe.get("close_controls"), recipe.get("settle_controls"),
            str(recipe.get("scene_digest_hash", ""))[:12],
        )
        report = sim.verify_stage1_transfer_equivalence(
            side, best_grasp, n_repeat=3, record_video=bool(args.views != "none")
        )
        _report_dir = Path(sim.output_dir) / "stage2"
        _report_dir.mkdir(parents=True, exist_ok=True)
        _report_path = _report_dir / "stage1_transfer_equivalence.json"
        _report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        logger.info(
            "[test-stage1-transfer] %s (%d/%d) → %s",
            "PASS" if report["passed"] else f"FAIL({report['failure']})",
            report["n_pass"], report["n_repeat"], _report_path,
        )
        return

    # v4.5: Stage 2 轨迹重建测试
    if args.test_stage2:
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
            grasp_pose_method=args.grasp_pose_method,  # v51 U1/P1-7: 各入口统一传参
            viewer=False,
        )
        sim._test_stage1_only = True
        sim.video_tag = "stage2"  # v4.16: 视频文件名标签
        sim.f50_tilt_deg = float(args.f50_tilt_deg)  # 外部附加旋转角 (与 stage1 一致)
        sim._align_scene()
        sim.run()
        side = args.side if args.side != "both" else "right"
        sim._preprocess_mano_traj(side)  # P1-8 修复: 此前漏 pre_translate; v51 U3/U5 统一序列

        # Stage 1: 优先加载已保存的最优, 跳过 CMA-ES (避免每次跑出不同结果, v4.28)
        logger.info("=" * 60)
        logger.info("v4.5 Stage 2 测试: Stage 1 → Stage 2")
        logger.info("=" * 60)
        _output_dir = args.output_dir or f"output/gripper_{side}"
        stage1_path = Path(_output_dir) / "stage1" / "best_grasp.npz"
        if stage1_path.exists():
            logger.info(f"[test-stage2] Stage 1: 加载已保存的最优 {stage1_path}")
            best_grasp, _recipe = load_stage1_grasp(stage1_path)
            if _recipe is None:
                logger.warning(
                    "[test-stage2] STALE_STAGE1_ARTIFACT: %s 缺少 stage1_grasp_recipe, "
                    "rollout 将以 STALE_STAGE1_ARTIFACT fail-fast 拒绝运行; 请重跑 --test-stage1",
                    stage1_path,
                )
            else:
                logger.info(
                    "[test-stage2] Stage1 recipe: target=%s q_grasp=%.4f close/settle=%d/%d "
                    "scene_digest=%s",
                    _recipe.get("target_object"), _recipe.get("q_grasp"),
                    _recipe.get("close_controls"), _recipe.get("settle_controls"),
                    str(_recipe.get("scene_digest_hash", ""))[:12],
                )
        else:
            # v6.1: CEM 预算与 v4-pipeline 统一 (15x32), 终选有 3 次重复验证门
            best_grasp = sim.cem_grasp_pose_optimize(side, n_iterations=15, population_size=32, top_k=6)
            if best_grasp is None:
                logger.error("Stage 1 优化失败")
                return
        logger.info(f"[test-stage2] Stage 1: pos={best_grasp['pos'].round(4)}, "
                    f"gripper_qpos={best_grasp['gripper_qpos']:.4f}")

        # Stage 2: 轨迹重建 + MANO x,y 位置参考
        from traj_optimize import compute_frame_params
        traj = sim._mano_gripper_traj.get(side)
        N = len(traj["pos"])
        fp = compute_frame_params(N)
        F50 = fp['F50_IDX']

        # Stage2 空间职责 F38-F62；F50 仅 3 close + 2 hold，随后立即移动验证。
        recon = sim.reconstruct_trajectory(
            best_grasp,
            side,
            F50_IDX=F50,
            f_start=38,
            f_end=62,
            pregrasp_retreat=args.stage2_pregrasp_retreat,
            approach_clearance_z=args.stage2_approach_clearance_z,
            approach_min_z=args.stage2_approach_min_z,
            q_near=args.stage2_q_near,
            approach_start_distance=args.stage2_approach_start_m,
            approach_mode=args.stage2_approach_mode,
            axial_transfer_frac=args.stage2_axial_transfer_frac,
            axial_clearance_m=args.stage2_axial_clearance_m,
        )
        if recon is None:
            logger.error("Stage 2 重建失败")
            return

        # 打印重建结果
        logger.info("[test-stage2] Stage 2 重建结果:")
        for f in sorted(recon['frames']):
            p = recon['pos'][f]
            g = recon['gripper_qpos'][f]
            logger.info(f"  F{f}: pos=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}), gripper_qpos={g:.4f}")

        # Stage2 正式验证: 同一个显式 bundle 独立 reset 三次，至少 2/3 通过。
        # 第三次可录制视频；渲染只采图，不增加 physics step。
        logger.info("[test-stage2] 执行 E3→E6: approach → F50(%d STAGE1_CLOSE + %d "
                    "STAGE1_SETTLE) → moving verify → carry",
                    STAGE1_CLOSE_CONTROLS, STAGE1_SETTLE_CONTROLS)
        stage2_results = []
        for repeat_idx in range(3):
            result = sim.rollout_follow_trajectory(
                recon,
                side,
                F50_IDX=F50,
                carry_time_scale=args.stage2_carry_time_scale,
                approach_time_scale=args.stage2_approach_time_scale,
                record_video=bool(repeat_idx == 2 and args.views != "none"),
            )
            stage2_results.append(result)
            gate = result.get('gate_result', {}) if result else {}
            logger.info(
                f"[test-stage2] repeat {repeat_idx + 1}/3: "
                f"{'PASS' if gate.get('passed', False) else 'FAIL'} "
                f"failure={gate.get('failure')} detail={gate.get('failure_detail', '')}"
            )

        passed_results = [
            result for result in stage2_results
            if result and result.get('gate_result', {}).get('passed', False)
        ]
        repeat_pass = len(passed_results) >= 2
        logger.info(f"[test-stage2] repeat gate: {len(passed_results)}/3 "
                    f"→ {'PASS ✓' if repeat_pass else 'FAIL ✗'}")

        # 只保存通过硬门的完整 executable bundle；失败结果不能成为 Stage3 输入。
        out_dir = Path(sim.output_dir) / "stage2"
        out_dir.mkdir(parents=True, exist_ok=True)
        if repeat_pass:
            passed_results[0]['gate_result']['repeat_pass_count'] = len(passed_results)
            passed_results[0]['gate_result']['repeat_count'] = 3
            saved = sim.save_stage2_bundle(out_dir / "stage2_bundle.npz", passed_results[0])
            logger.info(f"[test-stage2] Stage 2 executable bundle: "
                        f"{'已保存' if saved else '保存失败'}")
        else:
            logger.warning("[test-stage2] Stage 2 未通过，不写 stage2_bundle.npz；"
                           "请按首个 failure 分类调整，不再运行 dz/dq/force 混合搜索")

        # v4.16: 释放视频写入器
        if hasattr(sim, '_writer_cam') and sim._writer_cam is not None:
            sim._writer_cam.release()
            logger.info("[test-stage2] 相机视角视频已保存")
            transcode_to_h264(getattr(sim, '_video_path_cam', ''))
        if hasattr(sim, '_writer_god') and sim._writer_god is not None:
            sim._writer_god.release()
            transcode_to_h264(getattr(sim, '_video_path_god', ''))
            logger.info("[test-stage2] 上帝视角视频已保存")
        if hasattr(sim, '_writer_right') and sim._writer_right is not None:
            sim._writer_right.release()
            transcode_to_h264(getattr(sim, '_video_path_right', ''))
            logger.info("[test-stage2] 右视角视频已保存")
        # v4.29: 提升为标准交付物 D3
        sim.collect_deliverable("D3_stage2_follow.mp4")
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
            grasp_pose_method=args.grasp_pose_method,  # v51 U1/P1-7: 各入口统一传参
            viewer=False,
        )
        sim._test_stage1_only = True
        sim.video_tag = "stage3"  # v4.16: 视频文件名标签
        sim.f50_tilt_deg = float(args.f50_tilt_deg)  # P1-10: stage3 此前漏设, 倾角固定 0
        sim._align_scene()
        sim.run()
        # v4.30 修复: test-stage3 独立路径漏了预平移, 导致 neutral_offset 高达 0.61m
        # (完整 v4-pipeline 在 Stage1 前已 pre_translate, offset≈0).
        # 统一约定: 全管线以"平移后轨迹"为基准, 必须先 pre_translate 再算 offset.
        side = args.side if args.side != "both" else "right"
        sim._preprocess_mano_traj(side)  # v51 U3/U5: 单一预处理入口 (原顺序已合规, 收拢为一行)

        # Stage 1: 优先加载已保存的最优, 跳过 CMA-ES (避免每次跑出不同结果)
        _output_dir = args.output_dir or f"output/gripper_{side}"
        stage1_path = Path(_output_dir) / "stage1" / "best_grasp.npz"
        if stage1_path.exists():
            logger.info("=" * 60)
            logger.info(f"[test-stage3] Stage 1: 加载已保存的最优 {stage1_path}")
            logger.info("=" * 60)
            best_grasp, _recipe = load_stage1_grasp(stage1_path)
            if _recipe is None:
                logger.warning(
                    "[test-stage3] STALE_STAGE1_ARTIFACT: %s 缺少 stage1_grasp_recipe; "
                    "Stage2 rollout 将 fail-fast 拒绝运行",
                    stage1_path,
                )
            # Stage2/3 必须逐值复用 Stage1 已验证 anchor。旧 artifact 只能告警，
            # 不能在下游用估算 enclose_z 静默覆盖 pos.z。
            _enclose_z_loaded = best_grasp.get('enclose_z')
            if _enclose_z_loaded is not None:
                _z_delta = abs(float(best_grasp['pos'][2]) - float(_enclose_z_loaded))
                if _z_delta > 0.005:
                    logger.warning(
                        f"[test-stage3] Stage1 pos.z/enclose_z 相差 {_z_delta * 1000.0:.1f}mm; "
                        "保留已验证 pos.z"
                    )
        else:
            logger.info("=" * 60)
            logger.info("[test-stage1] Stage 1: 抓取姿态优化 (无已保存数据)")
            logger.info("=" * 60)
            # v6.1: 8x16 → 15x32 与 v4-pipeline 完全一致 (小预算 CEM 收敛差;
            # 终选已有 3 次重复验证门兜底)
            best_grasp = sim.cem_grasp_pose_optimize(side, n_iterations=15, population_size=32, top_k=6)
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
        recon = sim.reconstruct_trajectory(
            best_grasp,
            side,
            F50_IDX=F50,
            f_start=38,
            f_end=62,
            pregrasp_retreat=args.stage2_pregrasp_retreat,
            approach_clearance_z=args.stage2_approach_clearance_z,
            approach_min_z=args.stage2_approach_min_z,
            q_near=args.stage2_q_near,
            approach_start_distance=args.stage2_approach_start_m,
            approach_mode=args.stage2_approach_mode,
            axial_transfer_frac=args.stage2_axial_transfer_frac,
            axial_clearance_m=args.stage2_axial_clearance_m,
        )
        if recon is None:
            logger.error("Stage 2 失败")
            return
        logger.info(f"[test-stage3] Stage 2: {len(recon['frames'])} 帧重建完成")

        # Stage3 的父输入必须是通过与 --test-stage2 相同 2/3 门的 executable
        # bundle。随后 Stage3 还会正式完整重放一次，不能直接从手工状态开始。
        stage2_results = [
            sim.rollout_follow_trajectory(
                recon,
                side,
                F50_IDX=F50,
                carry_time_scale=args.stage2_carry_time_scale,
                approach_time_scale=args.stage2_approach_time_scale,
                record_video=False,
            )
            for _ in range(3)
        ]
        stage2_passed = [
            item for item in stage2_results
            if item.get("gate_result", {}).get("passed", False)
        ]
        if len(stage2_passed) < 2:
            failures = [item.get("gate_result", {}).get("failure") for item in stage2_results]
            # 诊断旁路 (2026-09-11, 用户要求"赶快渲染一下 stage3"):
            # Stage3 的父输入按契约必须是过了 2/3 门的 Stage2 bundle. 当 Stage2 只差
            # 边缘门 (如 APPROACH_TARGET_CONTACT 轻擦 / 基座速度超 0.7mm/s) 时, 为让用户
            # 先看到 Stage3 的 F50→F90 表现, _S3_DIAG=1 允许强制用第 1 次 bundle 继续.
            # ⚠ 该路径产出的 Stage3 结果**不作为 PASS**, 仅供诊断/录像.
            if os.environ.get("_S3_DIAG", "0") == "1":
                logger.warning(
                    "[test-stage3][诊断] Stage2 repeat gate %d/3 未过 failures=%s; "
                    "_S3_DIAG=1 → 强制用第 1 次 bundle 继续跑 Stage3 (结果不作为 PASS)",
                    len(stage2_passed), failures,
                )
                stage2_passed = [stage2_results[0]]
            else:
                logger.error(f"[test-stage3] Stage2 repeat gate FAIL: "
                             f"{len(stage2_passed)}/3 failures={failures}; Stage3 不运行")
                return
        stage2_passed[0]["gate_result"]["repeat_pass_count"] = len(stage2_passed)
        stage2_passed[0]["gate_result"]["repeat_count"] = 3
        sim._v4_stage2_recon = recon
        sim._v4_stage2_bundle = stage2_passed[0]["bundle"]
        sim._v4_stage2_result = stage2_passed[0]
        sim._v4_best_grasp = best_grasp
        sim._frame_params = fp
        stage2_out = Path(sim.output_dir) / "stage2" / "stage2_bundle.npz"
        sim.save_stage2_bundle(stage2_out, stage2_passed[0])

        logger.info("=" * 60)
        logger.info("[test-stage3] Stage 3: exact Stage2 replay + carry + conditional release")
        logger.info("=" * 60)

        result = sim.rollout_v4_stage3(
            side=side,
            carry_time_scale=args.stage3_carry_time_scale,
            release_candidate=args.stage3_release_candidate,
            record_video=bool(args.views != "none"),
        )
        gate_a = result.get("stage3a_gate", {})
        gate_b = result.get("stage3b_gate", {})
        logger.info(f"[test-stage3] Stage3A: "
                    f"{'PASS' if gate_a.get('passed', False) else 'FAIL'} "
                    f"failure={gate_a.get('failure')} detail={gate_a.get('failure_detail', '')}")
        logger.info(f"[test-stage3] Stage3B: "
                    f"{'PASS' if gate_b.get('passed', False) else 'FAIL'} "
                    f"failure={gate_b.get('failure')} detail={gate_b.get('failure_detail', '')}")

        # 保存结果
        out_dir = Path(sim.output_dir) / "stage3"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = sim.save_stage3_bundle(out_dir / "combined_bundle.npz", result)
        logger.info(f"[test-stage3] 组合 bundle: {'已保存' if saved else '未保存(Stage3A失败)'}")

        # v4.15: 释放视频写入器 (如果使用了渲染)
        if hasattr(sim, '_writer_cam') and sim._writer_cam is not None:
            sim._writer_cam.release()
            logger.info(f"[test-stage3] 相机视角视频已保存")
            transcode_to_h264(getattr(sim, '_video_path_cam', ''))
        if hasattr(sim, '_writer_god') and sim._writer_god is not None:
            sim._writer_god.release()
            transcode_to_h264(getattr(sim, '_video_path_god', ''))
            logger.info(f"[test-stage3] 上帝视角视频已保存")
        if hasattr(sim, '_writer_right') and sim._writer_right is not None:
            sim._writer_right.release()
            transcode_to_h264(getattr(sim, '_video_path_right', ''))
            logger.info(f"[test-stage3] 右视角视频已保存")
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
            grasp_pose_method=args.grasp_pose_method,  # v51 U1/P1-7: 优化实例此前漏传, 实际跑 analytical
            viewer=False,
        )
        sim_opt._is_v4_pipeline = True
        sim_opt.run(run_main_loop=False)  # 仅初始化, 不进入主循环 (Stage 1/2/3 在此后执行)
        side = args.side if args.side != "both" else "right"
        pipeline_result = sim_opt.run_v4_pipeline(
            side,
            stage3_method=args.stage3_method,
            stage2_pregrasp_retreat=args.stage2_pregrasp_retreat,
            stage2_approach_min_z=args.stage2_approach_min_z,
            stage2_q_near=args.stage2_q_near,
            stage2_approach_start_m=args.stage2_approach_start_m,
            stage2_approach_mode=args.stage2_approach_mode,
            stage2_axial_transfer_frac=args.stage2_axial_transfer_frac,
            stage2_axial_clearance_m=args.stage2_axial_clearance_m,
            stage2_approach_time_scale=args.stage2_approach_time_scale,
            stage2_carry_time_scale=args.stage2_carry_time_scale,
            stage3_carry_time_scale=args.stage3_carry_time_scale,
            stage3_release_candidate=args.stage3_release_candidate,
        )
        if pipeline_result is None:
            logger.error("v4 pipeline 未生成通过 Stage3A 的组合 bundle，停止渲染回放")
            return

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
            grasp_pose_method=args.grasp_pose_method,  # v51 U1/P1-7: 渲染实例与优化实例保持一致
            viewer=args.viewer,
        )
        sim._is_v4_pipeline = True
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
            logger.info(f"  保存最优参数: {sim_opt._opt_dir() / 'opt_params.npy'}")
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
        grasp_pose_method=args.grasp_pose_method,
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
            # 加载 frame_params 和 fixed_offsets_654 (v6.8: 位于 _opt_dir = final/intermediate)
            _fp_path = sim._opt_dir() / "frame_params.npy"
            _fo_path = sim._opt_dir() / "fixed_offsets_654.npy"
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
