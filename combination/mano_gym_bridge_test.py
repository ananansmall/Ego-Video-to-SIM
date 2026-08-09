#!/usr/bin/env python3
"""
MANO → Gym 环境桥接测试
=========================
目标：验证 gym 环境能否接收 MANO 重定向的夹爪位姿，并渲染出物理仿真

步骤：
1. 生成合成 MANO 轨迹（模拟抓手机动作）
2. 重定向到夹爪位姿（Gram-Schmidt）
3. 喂给 gym 环境做物理仿真
4. 渲染视频
"""

import sys, json, tempfile, imageio
from pathlib import Path
from copy import deepcopy
import numpy as np
import gymnasium as gym
import sapien

sys.path.insert(0, '/home/an/robot_world_ws/src/GalaxeaManipSim')
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting')

from galaxea_sim.envs.robotwin.dual_bottles_pick_easy import DualBottlesPickEasyEnv
from galaxea_sim.utils.robotwin_utils import create_box
from galaxea_sim.robots.r1 import R1Robot
from galaxea_sim.planners.bimanual import BimanualPlanner
from sapien import Pose as SapienPose
from scipy.spatial.transform import Rotation as R


# ═══════════════════════════════════════════════════════════
# Step 1: 生成合成 MANO 轨迹（模拟抓手机）
# ═══════════════════════════════════════════════════════════

def generate_mano_trajectory(num_frames=200):
    """
    生成模拟抓手机的 MANO 轨迹:
    - 前 50 帧: 手腕从后方移动到手机前方
    - 中 50 帧: 手指逐渐闭合（捏手机）
    - 后 100 帧: 拿起手机并向上移动
    """
    wrist_positions = np.zeros((num_frames, 3))
    finger1_positions = np.zeros((num_frames, 3))
    finger2_positions = np.zeros((num_frames, 3))

    # 手机位置（在桌面上，右手侧）
    phone_pos = np.array([0.55, -0.15, 0.96])  # 桌面 z=0.9, 手机半高=0.04

    for i in range(num_frames):
        # 手腕位置：从后上方→手机位置→抬起
        if i < 50:
            # 接近
            t = i / 49
            wrist = phone_pos + np.array([0.15, 0, 0.15]) * (1 - t)
        elif i < 100:
            # 抓取（手腕轻微移动）
            t = (i - 50) / 49
            wrist = phone_pos + np.array([0, 0, 0.1]) * t
        else:
            # 抬起
            t = (i - 100) / 99
            wrist = phone_pos + np.array([0, 0, 0.25]) * t

        wrist_positions[i] = wrist

        # 手指位置：根据手腕位置和开合状态
        if i < 50:
            # 接近时手指张开
            finger_open = 0.06
        elif i < 100:
            # 抓取时手指闭合
            t = (i - 50) / 49
            finger_open = 0.06 - t * 0.04  # 0.06 → 0.02
        else:
            # 抬起时保持闭合
            finger_open = 0.02

        # 手指在手腕两侧
        finger1_positions[i] = wrist + np.array([0, finger_open, 0])
        finger2_positions[i] = wrist + np.array([0, -finger_open, 0])

    return wrist_positions, finger1_positions, finger2_positions


# ═══════════════════════════════════════════════════════════
# Step 2: Gram-Schmidt 重定向到夹爪位姿
# ═══════════════════════════════════════════════════════════

def retarget_mano_to_gripper(mano_wrist, mano_f1, mano_f2, hand_type='right'):
    """
    将 MANO 关键点重定向到夹爪位姿
    返回: gripper_ee_pose (7-DOF: x, y, z, qw, qx, qy, qz)
    """
    wrist = np.array(mano_wrist, dtype=np.float64)
    f1 = np.array(mano_f1, dtype=np.float64)
    f2 = np.array(mano_f2, dtype=np.float64)

    # 计算开合方向 (Y 轴)
    opening_vec = f1 - f2
    opening_norm = np.linalg.norm(opening_vec)
    if opening_norm < 1e-4:
        opening_norm = 0.02
    y_axis = opening_vec / opening_norm

    # 计算指向方向 (X 轴)
    x_raw = wrist - (f1 + f2) / 2
    x_raw = x_raw - np.dot(x_raw, y_axis) * y_axis
    x_norm = np.linalg.norm(x_raw)
    if x_norm < 1e-4:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = x_raw / x_norm

    z_axis = np.cross(x_axis, y_axis)
    R_mat = np.column_stack([x_axis, y_axis, z_axis]).T

    # 四元数 (Sapien 期望 wxyz)
    rot = R.from_matrix(R_mat)
    quat = rot.as_quat()  # xyzw
    qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]

    # 手指距离 → grasp 命令
    finger_dist = opening_norm * 1000  # mm
    grasp_cmd = min(1.0, max(0.0, (finger_dist - 10) / 60))

    # 夹爪 EE 位姿 = 手腕位置
    ee_pos = wrist

    return list(ee_pos) + [qw, qx, qy, qz], float(grasp_cmd)


# ═══════════════════════════════════════════════════════════
# Step 3: 自定义 Gym 环境（加载手机场景）
# ═══════════════════════════════════════════════════════════

class ManoPhonePickEnv(DualBottlesPickEasyEnv):
    """自定义环境：只保留右臂 + 手机（红色方块）"""

    @property
    def table_height(self):
        return 0.9

    @property
    def tabletop_center_x(self):
        return 0.7

    def _setup_red_bottle(self):
        """创建手机（红色方块）"""
        self.red_bottle_xlim = [0.5, 0.6]
        self.red_bottle_ylim = [-0.2, -0.1]
        self.red_bottle_zlim = [0.15]
        self.red_bottle_qpos = [0.707, 0.707, 0, 0]

        # 手机位置
        phone_pos = self.tabletop_center_in_world + np.array([-0.15, -0.15, 0.04])
        self.red_bottle = create_box(
            scene=self._scene,
            pose=sapien.Pose(p=phone_pos),
            half_size=[0.08, 0.04, 0.01],  # 手机尺寸: 长x宽x厚
            color=(0.3, 0.3, 0.3),  # 深灰色
            name="phone",
        )
        # 设置高摩擦
        for comp in self.red_bottle.components:
            try:
                comp.physx_material = self._scene.create_physical_material(2.0, 2.0, 0.8)
            except:
                pass

    def _setup_green_bottle(self):
        """创建隐藏的占位物（兼容基类）"""
        self.green_bottle_xlim = [-1, -1]
        self.green_bottle_ylim = [-1, -1]
        self.green_bottle_zlim = [0]
        self.green_bottle_qpos = [0, 0, 0, 0]

        # 放在看不见的地方
        hidden_pos = self.tabletop_center_in_world + np.array([-2, -2, -1])
        self.green_bottle = create_box(
            scene=self._scene,
            pose=sapien.Pose(p=hidden_pos),
            half_size=[0.01, 0.01, 0.01],
            color=(0.2, 1, 0.2),
            name="hidden_green",
        )
        # Both initial poses need to be set (red is set after _setup_red_bottle)
        self.red_bottle_initial_pose_on_table = self.red_bottle.get_pose()
        self.green_bottle_initial_pose_on_table = self.green_bottle.get_pose()

    def solution(self):
        """不做预设方案，由外部 MANO 轨迹驱动"""
        yield ("noop", {})

    def _get_info(self):
        if self.red_bottle is None:
            return dict(success=False)
        phone_pos = np.array(self.red_bottle.get_pose().p)
        return dict(
            phone_z=phone_pos[2],
            success=phone_pos[2] > 0.95,  # 手机被提起
        )

    def _check_termination(self):
        return bool(self._get_info()["success"])

    @property
    def language_instruction(self):
        return "pick up the phone"


def main():
    print("=" * 60)
    print("MANO → Gym 环境桥接测试")
    print("=" * 60)

    # ── 注册环境 ──
    gym.register(
        id='R1ManoPhonePick-v0',
        entry_point=ManoPhonePickEnv,
        disable_env_checker=True,
        order_enforce=False,
        kwargs=dict(
            robot_class=R1Robot,
            robot_kwargs=dict(init_qpos=[
                0.7005, -1.4028, -0.9996, 0.0, 0, 0,
                1.57, 1.57, -0.96, -0.96,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]),
            headless=True,
        ),
        max_episode_steps=300,
    )

    env = gym.make('R1ManoPhonePick-v0', control_freq=15, headless=True, obs_mode='state')

    # ── 初始化规划器 ──
    planner = BimanualPlanner(
        urdf_path=f"{env.unwrapped.robot.name}/robot.urdf",
        srdf_path=None,
        left_arm_move_group=env.unwrapped.left_ee_link_name,
        right_arm_move_group=env.unwrapped.right_ee_link_name,
        active_joint_names=env.unwrapped.active_joint_names,
        control_freq=env.unwrapped.control_freq,
    )

    # ── 生成 MANO 轨迹 ──
    print("生成 MANO 轨迹...")
    wrist_traj, f1_traj, f2_traj = generate_mano_trajectory(num_frames=200)
    print(f"  轨迹长度: {len(wrist_traj)} 帧")

    # ── 视频录制 ──
    frames = []

    def capture():
        rgb = env.unwrapped.render()
        frames.append(rgb)

    # ── 初始化 ──
    obs, info = env.reset()
    capture()
    print(f"环境初始化完成")

    # ── 执行 MANO 轨迹 ──
    print("开始执行 MANO 轨迹...")

    for frame_idx in range(len(wrist_traj)):
        # 重定向
        ee_pose, grasp_cmd = retarget_mano_to_gripper(
            wrist_traj[frame_idx], f1_traj[frame_idx], f2_traj[frame_idx]
        )

        # 调试: 打印前几帧
        if frame_idx < 5:
            print(f"  帧 {frame_idx}: ee_pos={ee_pose[:3]}, ee_quat={ee_pose[3:]}, grasp={grasp_cmd}")

        # 创建 substep
        substep = ("move_to_pose", {"right_pose": deepcopy(ee_pose)})

        # 规划（verbose=True 看 IK 错误）
        actions = planner.solve(substep, env.unwrapped.robot.get_qpos(),
                                [0.0, grasp_cmd], verbose=True)

        if actions is None:
            print(f"  帧 {frame_idx}: planner.solve 返回 None, 跳过")
            if frame_idx < 3:
                print(f"    ee_pose = {ee_pose}")
            capture()
            continue

        if actions is not None:
            for action in actions:
                obs, _, terminated, truncated, info = env.step(action)
                capture()

                if frame_idx % 50 == 0:
                    phone_z = info.get('phone_z', 0)
                    success = info.get('success', False)
                    print(f"  帧 {frame_idx}: 手机 z={phone_z:.3f}, 抓取={success}")
        else:
            terminated = False
            truncated = False
            info = {}

        if terminated or truncated:
            print(f"环境终止于帧 {frame_idx}")
            break

    # ── 保存视频 ──
    output_path = Path("/tmp/mano_gym_bridge.mp4")
    imageio.mimsave(str(output_path), frames, fps=30)
    print(f"\n视频已保存: {output_path} ({len(frames)} 帧)")

    # ── 结果 ──
    if env.unwrapped.red_bottle:
        phone_pos = np.array(env.unwrapped.red_bottle.get_pose().p)
        print(f"\n结果:")
        print(f"  手机最终 z: {phone_pos[2]:.3f}")
        print(f"  抓取成功: {info.get('success', False)}")

    env.close()
    return frames


if __name__ == '__main__':
    frames = main()
    print(f"\n总计 {len(frames)} 帧")
