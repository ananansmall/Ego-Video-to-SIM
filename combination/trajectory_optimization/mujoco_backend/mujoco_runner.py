"""MuJoCo 三形式跟踪执行器"""

import logging
import sys
from pathlib import Path

import numpy as np
from pytransform3d import rotations as pr

from physics_utils import (
    RXWORLD_TO_SAPIEN, RIGHT_ARM_STARTING,
    GRIPPER_INIT_OPEN, LP_ALPHA_EE, LP_ALPHA_JOINT,
    GRIPPER_MAX_OPEN,
)
from mujoco_backend.mujoco_env import MuJoCoEnv
from trajectory_loader import (
    load_hawor_data, load_hawor_c2w, compute_mano_joints,
    compute_analytical_gripper_pose, load_glb_transformed,
)
from grasp_controller import GraspController

logger = logging.getLogger(__name__)


class LPFilter:
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


class MuJoCoRunner:
    """MuJoCo 三形式跟踪执行器"""

    def __init__(self, form_name="floating_arm", side="right", headless=True):
        self.form_name = form_name
        self.side = side
        self.env = MuJoCoEnv(form_name, side, headless)
        self.grasp_ctrl = GraspController()
        self.joint_filter = LPFilter(LP_ALPHA_JOINT)
        self.ik_solver = None
        self.retargeting = None
        self._R_c2w_all = None
        self._t_c2w_all = None

    def build(self):
        self.env.build()
        self.grasp_ctrl.gripper_joint_names = [
            self.env.joint_names[i] for i in self.env.gripper_joint_indices
            if i < len(self.env.joint_names)
        ]
        return self

    def _get_unified_qpos(self):
        """提取统一 8 DOF qpos: [右臂关节1-6, 右夹爪关节1, 右夹爪关节2]"""
        full_qpos = self.env.get_qpos()
        unified = np.zeros(8, dtype=np.float64)
        for j, idx in enumerate(self.env.arm_joint_indices):
            if j < 6 and idx < len(full_qpos):
                unified[j] = full_qpos[idx]
        for j, idx in enumerate(self.env.gripper_joint_indices):
            if j < 2 and idx < len(full_qpos):
                unified[6 + j] = full_qpos[idx]
        return unified

    def init_retargeting(self):
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
        logger.info(f"  DexRetargeting 已初始化 ({self.side})")

    def init_ik(self):
        if self.form_name == "gripper_only":
            return
        from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver
        from physics_utils import R1_RIGHT_SETTINGS, R1_LEFT_SETTINGS

        self.ik_solver = RelaxedIKSolver(
            left_setting_file_path=str(R1_LEFT_SETTINGS),
            right_setting_file_path=str(R1_RIGHT_SETTINGS),
            tolerances=[0.1] * 6,
        )
        logger.info(f"  RelaxedIK 已初始化 ({self.side})")

    def run_tracking(self, hawor_dir, ras_dir, transform_params_path,
                     start_frame=0, num_frames=-1, output_video=None,
                     target_qpos_trajectory=None):
        """执行轨迹跟踪

        Args:
            hawor_dir: HaWoR 数据目录
            ras_dir: RAS 数据目录
            transform_params_path: 变换参数 npz 路径
            start_frame: 起始帧
            num_frames: 帧数 (-1=全部)
            output_video: 输出视频路径 (None=不保存)
            target_qpos_trajectory: SAPIEN 计算的 8 DOF 目标 qpos (None=自行计算IK)

        Returns:
            dict: 跟踪结果
        """
        import mujoco

        hawor_data = load_hawor_data(hawor_dir, hand_idx=0 if self.side == "left" else 1)
        total_frames = hawor_data["pred_trans"].shape[0]
        if num_frames < 0 or num_frames > total_frames - start_frame:
            num_frames = total_frames - start_frame

        # 加载 HaWoR 相机轨迹 (对齐 002_render_scene.py: 第一人称视角)
        self._R_c2w_all, self._t_c2w_all = load_hawor_c2w(hawor_dir)
        if self._R_c2w_all is not None:
            logger.info(f"  HaWoR 相机轨迹: {self._R_c2w_all.shape[0]} 帧")

        sys_path = str(Path(__file__).resolve().parents[4] / "example" / "position_retargeting")
        if sys_path not in sys.path:
            sys.path.insert(0, sys_path)
        from mano_layer import MANOLayer

        betas_mean = hawor_data["pred_betas"][start_frame].astype(np.float32)
        mano_side = "left" if self.side == "left" else "right"
        mano_layer = MANOLayer(mano_side, betas_mean)

        # GLB 物体 (MuJoCo 暂不加载GLB, 仅跟踪)
        obj_actors = []

        # 回放模式: 跳过 retargeting/IK 初始化
        use_replay = target_qpos_trajectory is not None
        if not use_replay:
            self.init_retargeting()
            self.init_ik()
        else:
            logger.info(f"  回放模式: 使用 SAPIEN target_qpos ({len(target_qpos_trajectory)} 帧)")

        # 视频录制 (对齐 002_render_scene.py: 第一人称 HaWoR 相机轨迹)
        recorder = None
        camera = None
        if output_video:
            from video_recorder import VideoRecorder, MuJoCoCamera
            recorder = VideoRecorder(output_video, fps=30)
            cam_pos, cam_target = self._compute_camera_pose_frame(0, hawor_data, start_frame)
            camera = MuJoCoCamera(self.env.model, self.env.data, pos=cam_pos, target=cam_target)

        qpos_sequence = []
        grasp_states = []

        for local_idx in range(num_frames):
            global_idx = start_frame + local_idx

            if not hawor_data["pred_valid"][global_idx]:
                qpos_sequence.append(self._get_unified_qpos())
                grasp_states.append(False)
                # 仍然录制视频帧 (更新相机位姿)
                if recorder and camera:
                    self._update_camera_per_frame(camera, hawor_data, global_idx)
                    frame = camera.capture()
                    recorder.add_frame(frame)
                continue

            # MANO FK (回放模式仍需 MANO 计算 gripper_only 的根位姿)
            _, joints = compute_mano_joints(
                mano_layer,
                hawor_data["pred_rot"][global_idx],
                hawor_data["pred_hand_pose"][global_idx],
                hawor_data["pred_trans"][global_idx],
            )
            joints_sapien = (RXWORLD_TO_SAPIEN @ joints[:, :3].T).T

            if use_replay:
                # 回放模式: 直接使用 SAPIEN target_qpos, 跳过 retargeting/IK
                target_8dof = target_qpos_trajectory[local_idx]
                target_qpos = self._step_replay(target_8dof, joints_sapien)
            elif self.form_name == "gripper_only":
                target_qpos = self._step_gripper_only(joints_sapien)
            else:
                target_qpos = self._step_arm(joints_sapien, hawor_data, global_idx)

            self.env.step_physics(target_qpos, kinematic_arm=use_replay)

            # 接触检测
            contacts = self.env.get_contacts()
            contact_count = len(contacts)
            is_grasping = self.grasp_ctrl.update_grasp_state(contact_count)

            qpos_sequence.append(self._get_unified_qpos())
            grasp_states.append(is_grasping)

            # 录制视频 (每帧更新相机位姿)
            if recorder and camera:
                self._update_camera_per_frame(camera, hawor_data, global_idx)
                frame = camera.capture()
                recorder.add_frame(frame)

            if (local_idx + 1) % 50 == 0:
                logger.info(f"  帧 {local_idx + 1}/{num_frames} | 抓取: {is_grasping}")

        # 保存视频
        if recorder:
            recorder.save()

        return {"qpos_sequence": qpos_sequence, "grasp_states": grasp_states}

    def _step_replay(self, target_8dof, joints_sapien):
        """回放模式: 从 SAPIEN target_qpos (8 DOF) 提取臂+夹爪目标

        Args:
            target_8dof: [右臂1-6, 夹爪joint1, 夹爪joint2]
            joints_sapien: MANO 关节 (gripper_only 需用于根位姿)
        Returns:
            dict: {joint_index: target_angle}
        """
        target = {}
        # 臂关节 (前6)
        arm_joints = target_8dof[:6]
        for j, idx in enumerate(self.env.arm_joint_indices):
            if j < 6:
                target[idx] = float(arm_joints[j])
        # 夹爪关节 (后2)
        gripper_joints = target_8dof[6:8]
        for j, idx in enumerate(self.env.gripper_joint_indices):
            if j < 2:
                target[idx] = float(gripper_joints[j])

        # gripper_only: 需设置根位姿 (从 MANO 解析计算)
        # 注: MuJoCo 中 gripper_only 为固定基座 (nq<7), set_root_pose 为 no-op,
        # 但调用保持与 PyBullet/SAPIEN 一致; 统一 qpos 仍由夹爪关节值匹配.
        if self.form_name == "gripper_only":
            wrist_pos = joints_sapien[0, :3]
            finger1_pos = joints_sapien[4, :3]
            finger2_pos = joints_sapien[8, :3]
            root_pos, root_R, _, _ = compute_analytical_gripper_pose(
                wrist_pos, finger1_pos, finger2_pos
            )
            root_quat_wxyz = pr.quaternion_from_matrix(root_R)
            self.env.set_root_pose(root_pos, root_quat_wxyz)

        return target

    def _step_arm(self, joints_sapien, hawor_data, global_idx):
        if self.retargeting is None:
            return {}

        ref_indices = self.retargeting.optimizer.target_link_human_indices
        fixed_retarget_indices = self.retargeting.optimizer.idx_pin2fixed
        ref_value = joints_sapien[ref_indices, :].astype(np.float32)

        fixed_qpos = np.zeros(len(fixed_retarget_indices), dtype=np.float32)
        current_qpos = self.env.get_qpos()
        for i, retarget_idx in enumerate(fixed_retarget_indices):
            if retarget_idx < len(current_qpos):
                fixed_qpos[i] = current_qpos[retarget_idx]

        retarget_qpos = self.retargeting.retarget(ref_value, fixed_qpos)

        gripper_val = 0.0
        for i, name in enumerate(self.retargeting.joint_names):
            if "gripper_finger_joint1" in name and i < len(retarget_qpos):
                gripper_val = float(retarget_qpos[i])
        gripper_val = float(np.clip(gripper_val, 0.0, GRIPPER_MAX_OPEN))

        # RelaxedIK
        if self.ik_solver is not None:
            ee_link_name = f"{self.side}_gripper_link"
            ee_pos, ee_quat = self.env.get_link_pose(ee_link_name)
            if ee_pos is not None:
                if self.side == "right":
                    ik_joints = np.array(
                        self.ik_solver.solve_position_right(
                            ee_pos.tolist(), ee_quat.tolist()
                        )
                    )
                else:
                    ik_joints = np.array(
                        self.ik_solver.solve_position_left(
                            ee_pos.tolist(), ee_quat.tolist()
                        )
                    )
                ik_joints = self.joint_filter.next(ik_joints)
            else:
                ik_joints = np.array(RIGHT_ARM_STARTING)
        else:
            ik_joints = np.array(RIGHT_ARM_STARTING)

        target = {}
        for j, idx in enumerate(self.env.arm_joint_indices):
            if j < len(ik_joints):
                target[idx] = ik_joints[j]
        for idx in self.env.gripper_joint_indices:
            if idx < len(self.env.joint_names):
                name = self.env.joint_names[idx]
                if "joint1" in name:
                    target[idx] = gripper_val
                elif "joint2" in name:
                    target[idx] = -gripper_val

        return target

    def _step_gripper_only(self, joints_sapien):
        wrist_pos = joints_sapien[0, :3]
        finger1_pos = joints_sapien[4, :3]
        finger2_pos = joints_sapien[8, :3]

        root_pos, root_R, joint1, joint2 = compute_analytical_gripper_pose(
            wrist_pos, finger1_pos, finger2_pos
        )

        target = {}
        for idx in self.env.gripper_joint_indices:
            if idx < len(self.env.joint_names):
                name = self.env.joint_names[idx]
                if "joint1" in name:
                    target[idx] = joint1
                elif "joint2" in name:
                    target[idx] = joint2
        return target

    def _compute_camera_pose(self):
        """根据机器人形式计算合适的相机位姿"""
        if self.form_name == "gripper_only":
            target = np.array([0.0, 0.0, 0.0])
            pos = target + np.array([0.30, 0.0, 0.20])
        else:
            base_pos, _ = self.env.get_link_pose(f"{self.side}_arm_base_link")
            if base_pos is None:
                base_pos = np.array([0.0, 0.0, 0.0])
            target = base_pos + np.array([0.0, 0.0, 0.05])
            pos = target + np.array([0.50, 0.30, 0.25])
        return pos.tolist(), target.tolist()

    def _compute_camera_pose_frame(self, local_idx, hawor_data, start_frame):
        """计算指定帧的相机位姿 (对齐 002_render_scene.py: 第一人称 HaWoR 相机轨迹)"""
        global_idx = start_frame + local_idx
        if (self._R_c2w_all is not None and self._t_c2w_all is not None
                and global_idx < len(self._t_c2w_all)):
            R_c2w = self._R_c2w_all[global_idx]
            t_c2w = self._t_c2w_all[global_idx]
            cam_pos = RXWORLD_TO_SAPIEN @ t_c2w
            cam_R = RXWORLD_TO_SAPIEN @ R_c2w
            forward = -cam_R[:, 2]
            target = cam_pos + forward * 0.5
            return cam_pos.tolist(), target.tolist()
        return self._compute_camera_pose()

    def _update_camera_per_frame(self, camera, hawor_data, global_idx):
        """每帧更新相机位姿 (对齐 002_render_scene.py: 第一人称 HaWoR 相机轨迹)"""
        if (self._R_c2w_all is not None and self._t_c2w_all is not None
                and global_idx < len(self._t_c2w_all)):
            R_c2w = self._R_c2w_all[global_idx]
            t_c2w = self._t_c2w_all[global_idx]
            cam_pos = RXWORLD_TO_SAPIEN @ t_c2w
            cam_R = RXWORLD_TO_SAPIEN @ R_c2w
            forward = -cam_R[:, 2]
            target = cam_pos + forward * 0.5
            camera.set_pose(cam_pos, target)
        else:
            pos, target = self._compute_camera_pose()
            camera.set_pose(pos, target)

    def disconnect(self):
        self.env.disconnect()
