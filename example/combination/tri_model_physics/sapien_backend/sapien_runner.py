"""SAPIEN 三形式跟踪执行器

对齐 02_render_scene.py 的 IK 管线:
  - Retargeting FK → base_link 坐标系 IK (非物理仿真 get_link_pose)
  - 基座跟踪 (±4cm)
  - Warmup smoothstep 过渡
  - Warm start retargeting
  - HaWoR 相机轨迹 (第一人称视角)
"""

import logging
import sys
from pathlib import Path

import numpy as np
from pytransform3d import rotations as pr

from physics_utils import (
    RXWORLD_TO_SAPIEN, RIGHT_ARM_STARTING, LEFT_ARM_STARTING,
    COMFORT_TARGET_IN_BASE, ARM_MAX_REACH, COMFORTABLE_REACH,
    BASE_TRACKING_RANGE, BASE_TRACKING_ALPHA, SAFETY_DISTANCE,
    WARMUP_FRAMES, LP_ALPHA_EE, LP_ALPHA_JOINT,
    GRIPPER_INIT_OPEN, CONTROL_FREQ, DECIMATION,
    GRIPPER_MAX_OPEN, RIGHT_ARM_JOINT_LIMITS,
)
from sapien_backend.sapien_env import SapienEnv
from trajectory_loader import (
    load_hawor_data, load_hawor_c2w, compute_mano_joints,
    compute_analytical_gripper_pose, load_glb_transformed,
)
from grasp_controller import GraspController

logger = logging.getLogger(__name__)

# 对齐 02_render_scene.py 的常量
IK_SOLVE_PER_FRAME = 20


class LPFilter:
    """低通滤波器"""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None

    def next(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.value is None:
            self.value = x.copy()
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value.copy()


def _hawor_cam_to_sapien_pose(R_c2w, t_c2w):
    """将 HaWoR 相机位姿转换为 SAPIEN 相机位姿 (对齐 02_render_scene.py)"""
    cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w
    cam_R_sapien = RXWORLD_TO_SAPIEN @ R_c2w
    forward = -cam_R_sapien[:, 2]
    left = -cam_R_sapien[:, 0]
    up = cam_R_sapien[:, 1]
    sapien_cam_R = np.eye(3)
    sapien_cam_R[:, 0] = forward
    sapien_cam_R[:, 1] = left
    sapien_cam_R[:, 2] = up
    if np.linalg.det(sapien_cam_R) < 0:
        U, _, VH = np.linalg.svd(sapien_cam_R)
        sapien_cam_R = U @ VH
    cam_quat = pr.quaternion_from_matrix(sapien_cam_R)
    return cam_pos_sapien, cam_quat


def _make_look_at_camera(eye, target, up=np.array([0, 0, 1.0])):
    """计算 look-at 相机姿态四元数 (对齐 02_render_scene.py)"""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    else:
        right /= np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    cam_R = np.eye(3)
    cam_R[:, 0] = forward
    cam_R[:, 1] = -right
    cam_R[:, 2] = cam_up
    cam_quat = pr.quaternion_from_matrix(cam_R)
    return cam_quat


class SapienRunner:
    """SAPIEN 三形式跟踪执行器 (对齐 02_render_scene.py)"""

    def __init__(self, form_name="floating_arm", side="right", headless=True):
        self.form_name = form_name
        self.side = side
        self.env = SapienEnv(form_name, side, headless)
        self.grasp_ctrl = GraspController()
        self.joint_filter = LPFilter(LP_ALPHA_JOINT)
        self.ee_filter = LPFilter(LP_ALPHA_EE)
        self.ik_solver = None
        self.retargeting = None
        self._placed_base_pos = None
        self._placed_base_quat = None
        self._retarget2sapien = None
        self._sapien2retarget = None
        self._fixed_qpos = None
        self._ref_indices = None
        self._R_c2w_all = None
        self._t_c2w_all = None
        # 统一 8 DOF 目标 qpos 轨迹: [右臂关节1-6, 右夹爪关节1, 右夹爪关节2]
        # SAPIEN 计算, PyBullet/MuJoCo 回放
        self.target_qpos_trajectory = []

    def build(self):
        """构建环境"""
        self.env.build()
        self.grasp_ctrl.gripper_joint_names = [
            self.env.joint_names[i] for i in self.env.gripper_joint_indices
        ]
        return self

    def init_retargeting(self):
        """初始化 DexRetargeting (手部→夹爪映射) — 仅 floating_arm/full_robot"""
        if self.form_name == "gripper_only":
            return

        from dex_retargeting.constants import RobotName, HandType, RetargetingType, get_default_config_path
        from dex_retargeting.retargeting_config import RetargetingConfig

        PROJECT_ROOT = Path(__file__).resolve().parents[4]
        robot_dir = PROJECT_ROOT / "assets" / "robots" / "hands"
        RetargetingConfig.set_default_urdf_dir(str(robot_dir))

        hand_type = HandType.right if self.side == "right" else HandType.left
        config_path = get_default_config_path(RobotName.r1_full, RetargetingType.position, hand_type)

        prefix = self.side
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
        self.retargeting = config.build()
        self._ref_indices = self.retargeting.optimizer.target_link_human_indices

        # 构建 retargeting ↔ sapien 关节映射 (对齐 02_render_scene.py)
        retarget_joint_names = self.retargeting.joint_names
        self._retarget2sapien = np.array(
            [retarget_joint_names.index(n) for n in self.env.joint_names if n in retarget_joint_names]
        ).astype(int)
        self._sapien2retarget = {}
        for sapien_i, retarget_i in enumerate(self._retarget2sapien):
            self._sapien2retarget[retarget_i] = sapien_i

        # 构建 fixed_qpos
        fixed_retarget_indices = self.retargeting.optimizer.idx_pin2fixed
        self._fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        init_qpos = self.env.get_qpos()
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx in self._sapien2retarget:
                sapien_idx = self._sapien2retarget[retarget_idx]
                if sapien_idx < len(init_qpos):
                    self._fixed_qpos[i] = init_qpos[sapien_idx]

        logger.info(f"  DexRetargeting 已初始化 ({self.side}), ref_indices={self._ref_indices}")

    def init_ik(self):
        """初始化 RelaxedIK — 仅 floating_arm/full_robot"""
        if self.form_name == "gripper_only":
            return

        from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver
        from physics_utils import R1_RIGHT_SETTINGS, R1_LEFT_SETTINGS

        self.ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=[0.1] * 6,
        )
        arm_starting = RIGHT_ARM_STARTING if self.side == "right" else LEFT_ARM_STARTING
        if self.side == "right":
            self.ik_solver.relaxed_ik_right.reset(arm_starting)
        else:
            self.ik_solver.relaxed_ik_left.reset(arm_starting)
        logger.info(f"  RelaxedIK 已初始化 ({self.side})")

    def _get_gripper_pose_from_retargeting(self, retarget_qpos):
        """从 retargeting 优化器的正运动学获取夹爪位姿 (对齐 02_render_scene.py)

        这是关键: 使用 retargeting 内部机器人的 FK 获取"期望"夹爪位姿,
        而非物理仿真的 get_link_pose (滞后).
        """
        internal_robot = self.retargeting.optimizer.robot
        internal_robot.compute_forward_kinematics(retarget_qpos)
        target_name = f"{self.side}_gripper_link"
        for i, name in enumerate(internal_robot.link_names):
            if name == target_name:
                pose = internal_robot.get_link_pose(i)
                return pose[:3, 3].copy(), pose[:3, :3].copy()
        raise RuntimeError(f"内部机器人中找不到 {target_name}")

    def _compute_tracking_base_pos(self, initial_base_pos, gripper_pos_fk, arm_base_quat):
        """计算跟踪模式下的基座位置 (对齐 02_render_scene.py)

        基座在初始位置基础上, 沿 XY 方向跟踪夹爪 (±4cm).
        """
        base_R = pr.matrix_from_quaternion(arm_base_quat)
        wrist_in_base = base_R.T @ (gripper_pos_fk - initial_base_pos)
        offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE
        clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)
        delta_world = base_R @ clamped_offset
        return initial_base_pos + delta_world

    def _compute_optimal_fixed_base(self, wrist_positions_sapien):
        """计算最优固定基座位置和朝向 (对齐 02_render_scene.py)"""
        if len(wrist_positions_sapien) == 0:
            return np.array([0.0, 0.0, COMFORTABLE_REACH]), \
                   pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))

        wrist_arr = np.array(wrist_positions_sapien)
        centroid = wrist_arr.mean(axis=0)
        wrist_range = wrist_arr.max(axis=0) - wrist_arr.min(axis=0)

        arm_base_pos = centroid.copy()
        arm_base_pos[2] += COMFORTABLE_REACH

        if wrist_range[0] > 0.01:
            arm_base_pos[0] += wrist_range[0] * 0.1

        # 绕Z轴旋转180° (让机器人面朝操作者)
        z_rot_180 = pr.quaternion_from_axis_angle(np.array([0, 0, 1, np.pi]))
        arm_base_q = pr.concatenate_quaternions(z_rot_180, np.array([1, 0, 0, 0]))

        logger.info(f"  手腕质心(SAPIEN): {centroid}")
        logger.info(f"  手腕运动范围: X={wrist_range[0]:.4f} Y={wrist_range[1]:.4f} Z={wrist_range[2]:.4f}")
        logger.info(f"  最优固定基座位置: {arm_base_pos}")

        max_dist = 0
        for wp in wrist_positions_sapien:
            d = np.linalg.norm(wp - arm_base_pos)
            if d > max_dist:
                max_dist = d
        logger.info(f"  基座到最远手腕距离: {max_dist:.4f}m (臂展={ARM_MAX_REACH:.3f}m)")
        if max_dist > ARM_MAX_REACH * 0.9:
            logger.warning(f"  ⚠ 最远手腕距离 {max_dist:.4f}m 接近臂展 {ARM_MAX_REACH:.3f}m, IK可能不稳定!")

        return arm_base_pos, arm_base_q

    def _get_unified_qpos(self):
        """提取统一 8 DOF qpos: [右臂关节1-6, 右夹爪关节1, 右夹爪关节2]

        从完整 qpos 中提取右臂(6)+右夹爪(2), 不足部分补零.
        """
        full_qpos = self.env.get_qpos()
        unified = np.zeros(8, dtype=np.float64)
        for j, idx in enumerate(self.env.arm_joint_indices):
            if j < 6 and idx < len(full_qpos):
                unified[j] = full_qpos[idx]
        for j, idx in enumerate(self.env.gripper_joint_indices):
            if j < 2 and idx < len(full_qpos):
                unified[6 + j] = full_qpos[idx]
        return unified

    def run_tracking(self, hawor_dir, ras_dir, transform_params_path,
                     start_frame=0, num_frames=-1, output_video=None,
                     target_qpos_trajectory=None):
        """执行轨迹跟踪 (对齐 02_render_scene.py)"""
        # 加载数据
        hawor_data = load_hawor_data(hawor_dir, hand_idx=0 if self.side == "left" else 1)
        total_frames = hawor_data["pred_trans"].shape[0]
        if num_frames < 0 or num_frames > total_frames - start_frame:
            num_frames = total_frames - start_frame

        # 加载 HaWoR 相机轨迹
        self._R_c2w_all, self._t_c2w_all = load_hawor_c2w(hawor_dir)
        if self._R_c2w_all is not None:
            logger.info(f"  HaWoR 相机轨迹: {self._R_c2w_all.shape[0]} 帧")

        # 初始化 MANO
        sys_path = str(Path(__file__).resolve().parents[4] / "example" / "position_retargeting")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from mano_layer import MANOLayer

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_side = "left" if self.side == "left" else "right"
        mano_layer = MANOLayer(mano_side, betas_mean)

        # 加载 GLB 物体
        glb_path = Path(ras_dir) / "final_scene.glb"
        obj_actors = []
        if glb_path.exists() and Path(transform_params_path).exists():
            obj_actors = load_glb_transformed(
                glb_path, transform_params_path,
                scene=self.env.scene, backend="sapien", logger=logger,
            )

        # 初始化 retargeting 和 IK
        self.init_retargeting()
        self.init_ik()

        # 计算最优基座位置 (floating_arm/full_robot)
        if self.form_name != "gripper_only":
            self._place_robot_base(hawor_data, mano_layer, start_frame, num_frames)

            # Warm start retargeting (对齐 02_render_scene.py)
            self._warm_start_retargeting(hawor_data, mano_layer, start_frame, num_frames)

            # Warmup smoothstep 过渡 (对齐 02_render_scene.py)
            self._warmup(hawor_data, mano_layer, start_frame, num_frames)

        # 视频录制 (仅当渲染设备可用时)
        recorder = None
        camera = None
        if output_video and getattr(self.env, "render_available", False):
            from video_recorder import VideoRecorder, SapienCamera
            recorder = VideoRecorder(output_video, fps=30)
            # 相机位姿在每帧动态更新 (对齐 02_render_scene.py: 第一人称 HaWoR 相机轨迹)
            cam_pos, cam_quat = self._compute_camera_pose_frame(0, hawor_data, start_frame)
            camera = SapienCamera(self.env.scene, pos=cam_pos, target=cam_pos + np.array([1, 0, 0]))
            # 用 HaWoR 相机轨迹的位姿覆盖初始 look-at
            camera.set_pose(cam_pos, cam_quat)
        elif output_video and not getattr(self.env, "render_available", False):
            logger.warning("  渲染设备不可用，跳过视频录制（物理跟踪仍正常运行）")

        # 跟踪循环
        qpos_sequence = []
        grasp_states = []

        for local_idx in range(num_frames):
            global_idx = start_frame + local_idx

            if not hawor_data["pred_valid"][global_idx]:
                qpos_sequence.append(self._get_unified_qpos())
                grasp_states.append(False)
                # 无效帧: 重复上一个 target (或零), 保持轨迹长度一致
                if self.target_qpos_trajectory:
                    self.target_qpos_trajectory.append(self.target_qpos_trajectory[-1].copy())
                else:
                    self.target_qpos_trajectory.append(np.zeros(8))
                # 仍然录制视频帧
                if recorder and camera:
                    self._update_camera_per_frame(camera, hawor_data, global_idx)
                    frame = camera.capture()
                    recorder.add_frame(frame)
                continue

            # MANO FK
            _, joints = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][global_idx],
                hawor_data["pred_hand_pose"][global_idx],
                hawor_data["pred_trans"][global_idx],
            )
            joints_sapien = (RXWORLD_TO_SAPIEN @ joints[:, :3].T).T

            if self.form_name == "gripper_only":
                qpos = self._step_gripper_only(joints_sapien)
            else:
                qpos = self._step_arm(joints_sapien, hawor_data, global_idx)

            # 物理步进 (臂关节运动学设置, 保证 actual qpos 精确匹配 target;
            # 夹爪用PD驱动, 保留抓取力. 避免 floating_arm/full_robot 物理发散)
            self.env.step_physics(qpos, kinematic_arm=(self.form_name != "gripper_only"))

            # 接触检测
            contact_count = 0
            for actor in obj_actors:
                contact_count += len(self.env.get_contacts(actor))
            is_grasping = self.grasp_ctrl.update_grasp_state(contact_count)

            qpos_sequence.append(self._get_unified_qpos())
            grasp_states.append(is_grasping)

            # 录制视频 (每帧更新相机位姿)
            if recorder and camera:
                self._update_camera_per_frame(camera, hawor_data, global_idx)
                frame = camera.capture()
                recorder.add_frame(frame)
                # 第一帧调试信息
                if local_idx == 0:
                    cam_pos, cam_target = self._compute_camera_pose()
                    logger.info(f"  [调试] 第1帧相机: pos={cam_pos}, target={cam_target}")
                    logger.info(f"  [调试] 第1帧像素(SapienCamera): min={frame.min()}, max={frame.max()}, mean={frame.mean():.2f}")
                    base_pos = getattr(self, "_placed_base_pos", None)
                    logger.info(f"  [调试] 机器人基座: {base_pos}")

                    # 对比: 直接用 scene.add_camera 渲染
                    import sapien
                    debug_cam = self.env.scene.add_camera("debug", 640, 480, np.deg2rad(60), 0.01, 100)
                    cam_pos_arr = np.array(cam_pos)
                    cam_target_arr = np.array(cam_target)
                    forward = cam_target_arr - cam_pos_arr
                    forward = forward / np.linalg.norm(forward)
                    up = np.array([0, 0, 1.0])
                    right = np.cross(forward, up)
                    right = right / np.linalg.norm(right)
                    cam_up = np.cross(right, forward)
                    R_cam = np.eye(3)
                    R_cam[:, 0] = forward
                    R_cam[:, 1] = -right
                    R_cam[:, 2] = cam_up
                    cam_quat_dbg = pr.quaternion_from_matrix(R_cam)
                    debug_cam.set_local_pose(sapien.Pose(cam_pos_arr.tolist(), cam_quat_dbg.tolist()))
                    self.env.scene.update_render()
                    debug_cam.take_picture()
                    dbg_rgb = debug_cam.get_picture("Color")[..., :3]
                    dbg_frame = (dbg_rgb * 255).astype(np.uint8)
                    logger.info(f"  [调试] 直接渲染像素: min={dbg_frame.min()}, max={dbg_frame.max()}, mean={dbg_frame.mean():.2f}")
                    # 检查机器人 link 位姿
                    for link in self.env.robot.get_links()[:3]:
                        name = link.get_name()
                        pose = link.get_entity_pose()
                        logger.info(f"  [调试] link {name}: pos={np.array(pose.p)}")
                    # 检查 GLB 物体位姿
                    for i, actor in enumerate(obj_actors):
                        try:
                            pose = actor.get_pose()
                            logger.info(f"  [调试] 物体{i}: pos={np.array(pose.p)}")
                        except Exception:
                            pass
                    # 保存图片
                    import imageio.v2 as imageio_v2
                    debug_path = str(Path(output_video).parent / "debug_frame0.png")
                    imageio_v2.imwrite(debug_path, frame)
                    dbg_path = str(Path(output_video).parent / "debug_direct.png")
                    imageio_v2.imwrite(dbg_path, dbg_frame)
                    logger.info(f"  [调试] 图片已保存: {debug_path}, {dbg_path}")

            if (local_idx + 1) % 50 == 0:
                logger.info(f"  帧 {local_idx + 1}/{num_frames} | 抓取: {is_grasping}")

        # 保存视频
        if recorder:
            recorder.save()

        return {
            "qpos_sequence": qpos_sequence,
            "grasp_states": grasp_states,
            "target_qpos_trajectory": np.array(self.target_qpos_trajectory),
        }

    def _step_arm(self, joints_sapien, hawor_data, global_idx):
        """floating_arm/full_robot: Retargeting FK → base_link 坐标系 IK (对齐 02_render_scene.py)"""
        if self.retargeting is None:
            return self.env.get_qpos()

        from dex_retargeting.constants import HandType

        # 1. DexRetargeting: 手部关节 → 夹爪关节角
        ref_value = joints_sapien[self._ref_indices, :].astype(np.float32)
        retarget_qpos = self.retargeting.retarget(ref_value, self._fixed_qpos)

        # 提取夹爪值并严格限位
        gripper_val = 0.0
        for i, name in enumerate(self.retargeting.joint_names):
            if "gripper_finger_joint1" in name and i < len(retarget_qpos):
                gripper_val = float(retarget_qpos[i])
        gripper_val = float(np.clip(gripper_val, 0.0, GRIPPER_MAX_OPEN))

        # 2. 从 retargeting FK 获取"期望"夹爪位姿 (关键: 不用物理仿真的 get_link_pose)
        gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retarget_qpos)

        # 3. 基座跟踪: 根据夹爪位置微调基座 (±4cm)
        tracked_base = self._compute_tracking_base_pos(
            self._placed_base_pos, gripper_pos_fk, self._placed_base_quat
        )
        self.env.set_root_pose(tracked_base, self._placed_base_quat)
        # 用 step_physics (含重力补偿) 而非 scene.step, 否则 stiffness=1000 的机器人会坍塌
        self.env.step_physics()

        # 4. 获取 base_link 世界位姿 (root pose 变化后)
        base_link_p, base_link_q = self.env.get_link_pose(f"{self.side}_arm_base_link")
        if base_link_p is None:
            base_link_p = tracked_base
            base_link_q = self._placed_base_quat
        base_link_R = pr.matrix_from_quaternion(base_link_q)
        base_link_R_inv = base_link_R.T

        # 5. 计算 IK 目标 (在 base_link 坐标系下)
        mapping_offset = np.zeros(3)
        safety_offset = np.zeros(3)
        ik_target_raw = gripper_pos_fk + mapping_offset + safety_offset
        ik_target_b = base_link_R_inv @ (ik_target_raw - base_link_p)
        ee_R_base = base_link_R_inv @ R_ee_world_fk
        ee_quat_b = pr.quaternion_from_matrix(ee_R_base)

        # 6. RelaxedIK 求解 (多次求解提高精度, 对齐 02_render_scene.py)
        if self.ik_solver is not None:
            solve_fn = (self.ik_solver.solve_position_right if self.side == "right"
                        else self.ik_solver.solve_position_left)
            ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
            for _ in range(IK_SOLVE_PER_FRAME - 1):
                ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
            ik_joints = self.joint_filter.next(ik_joints)
        else:
            ik_joints = np.array(RIGHT_ARM_STARTING if self.side == "right" else LEFT_ARM_STARTING)

        # 6.5 关节限位裁剪 — 防止 IK 发散产生超限值 (如 -9, -16, 23)
        if self.side == "right" and len(ik_joints) == 6:
            ik_joints = np.clip(
                ik_joints, RIGHT_ARM_JOINT_LIMITS[:, 0], RIGHT_ARM_JOINT_LIMITS[:, 1]
            )

        # 7. 组装 qpos (物理 PD 驱动)
        qpos = self.env.get_qpos().copy()
        for j, idx in enumerate(self.env.arm_joint_indices):
            if j < len(ik_joints):
                qpos[idx] = ik_joints[j]
        for idx in self.env.gripper_joint_indices:
            name = self.env.joint_names[idx]
            if "joint1" in name:
                qpos[idx] = gripper_val
            elif "joint2" in name:
                qpos[idx] = -gripper_val

        # 8. 保存统一 8 DOF 目标 qpos: [右臂1-6, 夹爪joint1, 夹爪joint2]
        target_8dof = np.zeros(8, dtype=np.float64)
        target_8dof[:6] = ik_joints[:6]
        target_8dof[6] = gripper_val
        target_8dof[7] = -gripper_val
        self.target_qpos_trajectory.append(target_8dof)

        return qpos

    def _step_gripper_only(self, joints_sapien):
        """gripper_only: 解析映射 → 直接设置夹爪位姿"""
        wrist_pos = joints_sapien[0, :3]
        finger1_pos = joints_sapien[4, :3]
        finger2_pos = joints_sapien[8, :3]

        root_pos, root_R, joint1, joint2 = compute_analytical_gripper_pose(
            wrist_pos, finger1_pos, finger2_pos
        )

        root_quat = pr.quaternion_from_matrix(root_R)
        self.env.set_root_pose(root_pos, root_quat)

        qpos = self.env.get_qpos().copy()
        for idx in self.env.gripper_joint_indices:
            name = self.env.joint_names[idx]
            if "joint1" in name:
                qpos[idx] = joint1
            elif "joint2" in name:
                qpos[idx] = joint2

        # 保存统一 8 DOF 目标 qpos: [右臂1-6(无臂补零), 夹爪joint1, 夹爪joint2]
        target_8dof = np.zeros(8, dtype=np.float64)
        target_8dof[6] = joint1
        target_8dof[7] = joint2
        self.target_qpos_trajectory.append(target_8dof)

        return qpos

    def _place_robot_base(self, hawor_data, mano_layer, start_frame, num_frames):
        """计算最优基座位置并放置机器人 (对齐 02_render_scene.py)"""
        wrist_positions = []
        for i in range(start_frame, start_frame + num_frames):
            if not hawor_data["pred_valid"][i]:
                continue
            _, joints = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][i],
                hawor_data["pred_hand_pose"][i],
                hawor_data["pred_trans"][i],
            )
            joints_sapien = (RXWORLD_TO_SAPIEN @ joints[:, :3].T).T
            wrist_positions.append(joints_sapien[0, :3])

        if not wrist_positions:
            return

        base_pos, base_quat = self._compute_optimal_fixed_base(wrist_positions)
        self.env.set_root_pose(base_pos, base_quat)
        # 用 step_physics (含重力补偿) 而非 scene.step, 否则机器人会坍塌
        self.env.step_physics()
        self.env.scene.update_render()
        self._placed_base_pos = base_pos.copy()
        self._placed_base_quat = np.array(base_quat)
        logger.info(f"  基座位置: {base_pos}, 朝向: {base_quat}")

    def _warm_start_retargeting(self, hawor_data, mano_layer, start_frame, num_frames):
        """Warm start retargeting 优化器 (对齐 02_render_scene.py)"""
        if self.retargeting is None:
            return
        from dex_retargeting.constants import HandType

        for probe_idx in range(num_frames):
            g_idx = start_frame + probe_idx
            if not hawor_data["pred_valid"][g_idx]:
                continue
            rot = hawor_data["pred_rot"][g_idx]
            trans = hawor_data["pred_trans"][g_idx]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)):
                continue
            _, j = compute_mano_joints(mano_layer, rot,
                                       hawor_data["pred_hand_pose"][g_idx], trans)
            joints_sapien = (RXWORLD_TO_SAPIEN @ j[:, :3].T).T
            wrist_R_render = pr.matrix_from_compact_axis_angle(rot)
            wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
            wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
            hand_type = HandType.right if self.side == "right" else HandType.left
            self.retargeting.warm_start(
                joints_sapien[0, :3], wrist_quat,
                hand_type=hand_type, is_mano_convention=True,
            )
            logger.info(f"  Warm start 完成 (帧 {g_idx})")
            break

    def _warmup(self, hawor_data, mano_layer, start_frame, num_frames):
        """Warmup smoothstep 过渡 (对齐 02_render_scene.py)

        从初始关节角平滑过渡到第一个有效帧的 IK 解.
        """
        if self.ik_solver is None or self.retargeting is None:
            return

        # 找到第一个有效帧的 IK 解
        first_valid_qpos = None
        for fi in range(start_frame, start_frame + num_frames):
            if not hawor_data["pred_valid"][fi]:
                continue
            rot = hawor_data["pred_rot"][fi]
            trans = hawor_data["pred_trans"][fi]
            if np.any(np.isnan(rot)) or np.any(np.isnan(trans)):
                continue
            _, j = compute_mano_joints(mano_layer, rot,
                                       hawor_data["pred_hand_pose"][fi], trans)
            joints_sapien = (RXWORLD_TO_SAPIEN @ j[:, :3].T).T
            ref_value = joints_sapien[self._ref_indices, :].astype(np.float32)
            retarget_qpos = self.retargeting.retarget(ref_value, self._fixed_qpos)
            gripper_pos_fk, R_ee_world_fk = self._get_gripper_pose_from_retargeting(retarget_qpos)
            tracked_base = self._compute_tracking_base_pos(
                self._placed_base_pos, gripper_pos_fk, self._placed_base_quat)
            self.env.set_root_pose(tracked_base, self._placed_base_quat)
            # 用 step_physics (含重力补偿) 而非 scene.step
            self.env.step_physics()
            base_link_p, base_link_q = self.env.get_link_pose(f"{self.side}_arm_base_link")
            if base_link_p is None:
                base_link_p = tracked_base
                base_link_q = self._placed_base_quat
            base_link_R = pr.matrix_from_quaternion(base_link_q)
            base_link_R_inv = base_link_R.T
            ik_target_b = base_link_R_inv @ (gripper_pos_fk - base_link_p)
            ee_R_base = base_link_R_inv @ R_ee_world_fk
            ee_quat_b = pr.quaternion_from_matrix(ee_R_base)
            solve_fn = (self.ik_solver.solve_position_right if self.side == "right"
                        else self.ik_solver.solve_position_left)
            ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
            for _ in range(IK_SOLVE_PER_FRAME * 5 - 1):
                ik_joints = np.array(solve_fn(ik_target_b.tolist(), ee_quat_b.tolist()))
            first_valid_qpos = np.array(ik_joints)
            # 关节限位裁剪 — 防止 IK 发散值进入 warmup 导致物理失稳
            # (与 _step_arm 的裁剪一致)
            if self.side == "right" and len(first_valid_qpos) == 6:
                first_valid_qpos = np.clip(
                    first_valid_qpos,
                    RIGHT_ARM_JOINT_LIMITS[:, 0], RIGHT_ARM_JOINT_LIMITS[:, 1],
                )
            break

        if first_valid_qpos is not None:
            arm_starting = np.array(RIGHT_ARM_STARTING if self.side == "right" else LEFT_ARM_STARTING)
            for wi in range(WARMUP_FRAMES):
                t = (wi + 1) / WARMUP_FRAMES
                t = t * t * (3 - 2 * t)  # smoothstep
                interp = arm_starting * (1 - t) + first_valid_qpos * t
                qpos = self.env.get_qpos().copy()
                for j_idx, arm_idx in enumerate(self.env.arm_joint_indices):
                    if j_idx < len(interp):
                        qpos[arm_idx] = interp[j_idx]
                self.env.step_physics(qpos)
            # 初始化 joint_filter
            self.joint_filter.next(first_valid_qpos)
            logger.info(f"  Warmup 完成 ({WARMUP_FRAMES} 帧 smoothstep 过渡)")

    def _compute_camera_pose_frame(self, local_idx, hawor_data, start_frame):
        """计算指定帧的相机位姿 (对齐 02_render_scene.py: 第一人称 HaWoR 相机轨迹)

        优先使用 HaWoR 相机轨迹 (R_c2w, t_c2w), 实现第一人称视角.
        若无相机轨迹, 回退到第三人称视角.
        """
        global_idx = start_frame + local_idx
        if (self._R_c2w_all is not None and self._t_c2w_all is not None
                and global_idx < len(self._t_c2w_all)):
            R_c2w = self._R_c2w_all[global_idx]
            t_c2w = self._t_c2w_all[global_idx]
            cam_pos, cam_quat = _hawor_cam_to_sapien_pose(R_c2w, t_c2w)
            return cam_pos.tolist(), cam_quat.tolist()
        # 回退: 第三人称视角
        return self._compute_camera_pose()

    def _update_camera_per_frame(self, camera, hawor_data, global_idx):
        """每帧更新相机位姿 (对齐 02_render_scene.py: 第一人称 HaWoR 相机轨迹)

        使用 HaWoR 相机轨迹 (R_c2w, t_c2w) 实现第一人称视角,
        这样可以看到机器人跟随手部轨迹移动, 以及与物体的交互.
        """
        if (self._R_c2w_all is not None and self._t_c2w_all is not None
                and global_idx < len(self._t_c2w_all)):
            R_c2w = self._R_c2w_all[global_idx]
            t_c2w = self._t_c2w_all[global_idx]
            cam_pos, cam_quat = _hawor_cam_to_sapien_pose(R_c2w, t_c2w)
            camera.set_pose(cam_pos, cam_quat)
        else:
            # 回退: 第三人称视角
            pos, target = self._compute_camera_pose()
            cam_quat = _make_look_at_camera(np.array(pos), np.array(target))
            camera.set_pose(pos, cam_quat)

    def _compute_camera_pose(self):
        """第三人称相机位姿 — 看向机器人基座/夹爪工作空间

        对齐 02_render_scene.py: 相机看向基座位置 (不是基座下方 0.25m).
        机器人臂水平延伸, 看向基座高度才能看到臂和夹爪.
        """
        if self.form_name == "gripper_only":
            target = np.array([0.0, 0.0, 0.0])
            pos = target + np.array([0.30, 0.0, 0.20])
        else:
            base_pos = getattr(self, "_placed_base_pos", None)
            if base_pos is None:
                base_pos, _ = self.env.get_link_pose(f"{self.side}_arm_base_link")
            if base_pos is None:
                base_pos = np.array([0.0, 0.0, 0.0])
            # 看向基座位置 (臂水平延伸, 不再向下偏移 0.25m)
            target = np.array(base_pos, dtype=np.float64)
            # 相机在基座前上方, 俯视夹爪工作空间
            pos = target + np.array([0.45, -0.35, 0.15])
        return pos.tolist(), target.tolist()
