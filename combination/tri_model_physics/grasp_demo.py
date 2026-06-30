"""GalaxeaManipSim 物理仿真抓取 demo — 使用自定义 gym 环境

借鉴 dual_bottles_pick_easy.py 模式, 但用 create_box 替代 rand_create_glb (不需要 robotwin_models).
通过 gym 环境正确处理所有偏移和校准.

运行:
    cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics
    conda run -n dex python grasp_demo.py --output output/grasp_demo.mp4
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="GalaxeaManipSim 抓取 demo")
    parser.add_argument("--output", default="output/grasp_demo.mp4", help="输出视频路径")
    args = parser.parse_args()

    # 确保 GalaxeaManipSim 在 path 中
    galaxea_src = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
    if str(galaxea_src) not in sys.path:
        sys.path.insert(0, str(galaxea_src))

    import sapien
    import gymnasium as gym
    import galaxea_sim.envs  # 注册所有环境
    from galaxea_sim.envs.robotwin.dual_bottles_pick_easy import DualBottlesPickEasyEnv
    from galaxea_sim.utils.robotwin_utils import create_box
    from galaxea_sim.robots.r1 import R1Robot
    from galaxea_sim.planners.bimanual import BimanualPlanner
    from sapien import Pose as SapienPose

    # ============ 1. 创建自定义环境类 (继承 DualBottlesPickEasyEnv) ============
    class BoxPickEnv(DualBottlesPickEasyEnv):
        """用 create_box 替代 rand_create_glb, 不需要 robotwin_models"""

        @property
        def table_height(self):
            return 0.9

        @property
        def tabletop_center_x(self):
            return 0.7

        def _setup_red_bottle(self):
            """重写: 用 create_box 替代 rand_create_glb"""
            self.red_bottle_xlim = [-0.2, 0.]
            self.red_bottle_ylim = [-0.25, -0.05]
            self.red_bottle_zlim = [0.125]
            self.red_bottle_qpos = [0.707, 0.707, 0, 0]

            # 用 create_box 创建红色方块 (替代 rand_create_glb)
            box_pos = self.tabletop_center_in_world + np.array([-0.1, -0.15, 0.05])
            self.red_bottle = create_box(
                scene=self._scene,
                pose=sapien.Pose(p=box_pos),
                half_size=[0.03, 0.03, 0.03],  # 6cm 立方体
                color=(1, 0.2, 0.2),
                name="red_box",
            )

        def _setup_green_bottle(self):
            """重写: 用 create_box 创建绿色方块"""
            self.green_bottle_xlim = [-0.2, 0.]
            self.green_bottle_ylim = [0.05, 0.25]
            self.green_bottle_zlim = [0.125]
            self.green_bottle_qpos = [0.707, 0.707, 0, 0]

            box_pos = self.tabletop_center_in_world + np.array([-0.1, 0.15, 0.05])
            self.green_bottle = create_box(
                scene=self._scene,
                pose=sapien.Pose(p=box_pos),
                half_size=[0.03, 0.03, 0.03],
                color=(0.2, 1, 0.2),
                name="green_box",
            )
            self.green_bottle_initial_pose_on_table = self.green_bottle.get_pose()
            self.red_bottle_initial_pose_on_table = self.red_bottle.get_pose()

        def solution(self):
            """抓取方案 (对齐 dual_bottles_pick_easy.py, 但只用右臂抓红方块)"""
            (right_grasp_ori := SapienPose()).set_rpy(
                rpy=(np.array([0, 0, 0.88], dtype=np.float32) + self.robot.right_ee_rpy_offset)
            )
            # 接近位姿 (方块前方 10cm)
            right_pose0 = SapienPose(
                p=self.red_bottle.get_pose().p + np.array([-0.1096, -0.1164, 0.]),
                q=right_grasp_ori.q,
            )
            # 抓取位姿 (贴近方块)
            right_pose1 = SapienPose(
                p=self.red_bottle.get_pose().p + np.array([-0.0196, -0.0164, 0.]),
                q=right_grasp_ori.q,
            )
            substeps = [
                ("move_to_pose", {"right_pose": deepcopy(right_pose0)}),
                ("open_gripper", {"action_mode": "both"}),
                ("move_to_pose", {"right_pose": deepcopy(right_pose1)}),
                ("close_gripper", {"action_mode": "both"}),
                ("move_to_pose", {"right_pose": deepcopy(self.right_target_pose)}),
                ("move_to_pose", {"right_pose": deepcopy(self.right_target_pose)}),
            ]
            for substep in substeps:
                yield substep

        def _get_info(self):
            """成功判定: 红方块被提起"""
            red_distance = np.linalg.norm(self.red_bottle.get_pose().p - self.right_target_pose.p)
            red_height = self.red_bottle.get_pose().p[2].item()
            red_success = red_distance < 0.15 and red_height >= self.right_target_pose.p[2].item() - 0.07
            return dict(
                red_success=red_success,
                success=red_success,
                red_distance=red_distance,
                red_height=red_height,
            )

        def _check_termination(self):
            return bool(self._get_info()["success"])

        @property
        def language_instruction(self):
            return "pick up the red box"

    # ============ 2. 注册自定义环境 ============
    gym.register(
        id='R1BoxPick-v0',
        entry_point=BoxPickEnv,
        disable_env_checker=True,
        order_enforce=False,
        kwargs=dict(
            robot_class=R1Robot,
            robot_kwargs=dict(init_qpos=[
                0.70050001, -1.40279996, -0.99959999, 0.0,
                0, 0, 1.57, 1.57, -0.96, -0.96,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]),
            headless=True,
        ),
        max_episode_steps=200,
    )
    logger.info("自定义环境 R1BoxPick-v0 已注册")

    # ============ 3. 创建环境 ============
    env = gym.make('R1BoxPick-v0', control_freq=15, headless=True, obs_mode='state')
    logger.info("环境已创建")

    # ============ 4. 初始化规划器 ============
    planner = BimanualPlanner(
        urdf_path=f"{env.unwrapped.robot.name}/robot.urdf",
        srdf_path=None,
        left_arm_move_group=env.unwrapped.left_ee_link_name,
        right_arm_move_group=env.unwrapped.right_ee_link_name,
        active_joint_names=env.unwrapped.active_joint_names,
        control_freq=env.unwrapped.control_freq,
    )
    logger.info("规划器已初始化")

    # ============ 5. 视频录制 ============
    import imageio
    frames = []

    def capture_frame():
        rgb = env.unwrapped.render()
        frames.append(rgb)
        return rgb

    # ============ 6. 执行抓取 (对齐 collect_demos.py) ============
    logger.info("开始抓取演示...")
    _, rest_info = env.reset()
    capture_frame()

    # 参数级验证: 记录每一步的关键参数
    param_log = []  # 每步: (step, substep_name, gripper_cmd, box_p, ee_p, qpos_arm)

    def get_right_ee_pose():
        """获取右臂末端在世界坐标系的位姿"""
        return env.unwrapped.robot.right_ee_link.get_entity_pose()

    def get_right_gripper_qpos():
        """获取右夹爪两个手指关节的 qpos"""
        qpos = env.unwrapped.robot.get_qpos()
        return qpos[env.unwrapped.right_gripper_joint_indices]

    def get_right_arm_qpos():
        """获取右臂 6 关节 qpos"""
        qpos = env.unwrapped.robot.get_qpos()
        return qpos[env.unwrapped.right_arm_joint_indices]

    # 记录初始状态
    box_initial = np.array(env.unwrapped.red_bottle.get_pose().p)
    ee_initial = np.array(get_right_ee_pose().p)
    logger.info(f"[初始] 红方块位置: {box_initial}")
    logger.info(f"[初始] 右臂末端位置: {ee_initial}")
    logger.info(f"[初始] 右臂关节角: {get_right_arm_qpos()}")
    logger.info(f"[初始] 右夹爪 qpos: {get_right_gripper_qpos()}")

    info = {}
    num_steps = 0
    current_substep = "init"
    substep_idx = 0
    for substep in env.unwrapped.solution():
        method, kwargs = substep
        substep_idx += 1
        current_substep = f"#{substep_idx}_{method}"
        logger.info(f"--- 子步骤 {current_substep} ---")

        actions = planner.solve(
            substep, env.unwrapped.robot.get_qpos(), env.unwrapped.last_gripper_cmd,
            verbose=False,
        )
        if actions is not None:
            for action in actions:
                num_steps += 1
                obs, _, _, _, info = env.step(action)
                capture_frame()

                # 记录参数
                box_p = np.array(env.unwrapped.red_bottle.get_pose().p)
                ee_p = np.array(get_right_ee_pose().p)
                gripper_qpos = get_right_gripper_qpos()
                arm_qpos = get_right_arm_qpos()
                gripper_cmd = env.unwrapped.last_gripper_cmd[1]  # 右夹爪指令
                param_log.append({
                    "step": num_steps,
                    "substep": current_substep,
                    "gripper_cmd": float(gripper_cmd),
                    "gripper_qpos": gripper_qpos.tolist(),
                    "box_p": box_p.tolist(),
                    "ee_p": ee_p.tolist(),
                    "arm_qpos": arm_qpos.tolist(),
                })

                if num_steps % 20 == 0:
                    box_ee_dist = np.linalg.norm(box_p - ee_p)
                    logger.info(
                        f"  步骤 {num_steps} | 子步骤 {current_substep} | "
                        f"夹爪cmd={gripper_cmd:.3f} qpos={gripper_qpos} | "
                        f"方块z={box_p[2]:.3f} | 末端z={ee_p[2]:.3f} | "
                        f"方块-末端距离={box_ee_dist:.3f} | "
                        f"成功: {info.get('success', False)}"
                    )

    # ============ 7. 保存视频 ============
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), frames, fps=30)
    logger.info(f"视频已保存: {output_path} ({len(frames)} 帧)")

    # ============ 8. 参数级验证 ============
    logger.info("=" * 60)
    logger.info("参数级验证 (读取实际参数判断抓取是否对应上)")
    logger.info("=" * 60)

    red_box = env.unwrapped.red_bottle
    red_final_p = np.array(red_box.get_pose().p)
    red_target_p = np.array(env.unwrapped.right_target_pose.p)
    red_height = red_final_p[2]
    target_height = red_target_p[2]

    logger.info(f"红方块最终位置: {red_final_p}")
    logger.info(f"目标位置: {red_target_p}")
    logger.info(f"方块高度: {red_height*100:.1f}cm, 目标高度: {target_height*100:.1f}cm")
    logger.info(f"抓取成功: {info.get('success', False)}")

    # 关键参数验证
    logger.info("")
    logger.info("--- 关键参数轨迹验证 ---")

    # 1. 验证夹爪开合 (open_gripper 应使 qpos 增大, close_gripper 应使 qpos 减小到 0)
    open_steps = [p for p in param_log if "open_gripper" in p["substep"]]
    close_steps = [p for p in param_log if "close_gripper" in p["substep"]]
    if open_steps:
        gripper_open_max = max(p["gripper_qpos"][0] for p in open_steps)
        logger.info(f"[夹爪开] open_gripper 阶段夹爪 qpos 最大值: {gripper_open_max:.4f} (期望 ~0.05)")
    if close_steps:
        gripper_close_min = min(p["gripper_qpos"][0] for p in close_steps)
        logger.info(f"[夹爪合] close_gripper 阶段夹爪 qpos 最小值: {gripper_close_min:.4f} (期望 ~0.0)")

    # 2. 验证方块在 close_gripper 后是否跟随末端上升
    move_after_grasp = [p for p in param_log if "move_to_pose" in p["substep"] and p["step"] > (close_steps[-1]["step"] if close_steps else 0)]
    if move_after_grasp:
        first_after = move_after_grasp[0]
        last_after = move_after_grasp[-1]
        box_delta_z = last_after["box_p"][2] - first_after["box_p"][2]
        ee_delta_z = last_after["ee_p"][2] - first_after["ee_p"][2]
        logger.info(f"[跟随上升] 抓取后方块上升: {box_delta_z*100:.1f}cm, 末端上升: {ee_delta_z*100:.1f}cm")
        if abs(box_delta_z - ee_delta_z) < 0.05:
            logger.info("  ✓ 方块与末端同步上升 (抓取牢固)")
        else:
            logger.warning(f"  ✗ 方块与末端上升差异 {abs(box_delta_z - ee_delta_z)*100:.1f}cm (可能掉落)")

    # 3. 验证方块最终高度 (应被提升到桌面以上)
    box_lifted = red_final_p[2] > box_initial[2] + 0.1  # 至少提升 10cm
    logger.info(f"[提升判定] 方块初始 z={box_initial[2]:.3f}, 最终 z={red_final_p[2]:.3f}, 提升 {(red_final_p[2]-box_initial[2])*100:.1f}cm")
    if box_lifted:
        logger.info("  ✓ 方块被成功提升 (超过 10cm)")
    else:
        logger.warning("  ✗ 方块未被提升")

    # 4. 保存参数轨迹到文件
    import json
    param_log_path = output_path.parent / (output_path.stem + "_param_log.json")
    with open(param_log_path, "w") as f:
        json.dump({
            "initial": {
                "box_p": box_initial.tolist(),
                "ee_p": ee_initial.tolist(),
                "arm_qpos": get_right_arm_qpos().tolist(),
                "gripper_qpos": get_right_gripper_qpos().tolist(),
            },
            "final": {
                "box_p": red_final_p.tolist(),
                "target_p": red_target_p.tolist(),
                "ee_p": np.array(get_right_ee_pose().p).tolist(),
                "arm_qpos": get_right_arm_qpos().tolist(),
                "gripper_qpos": get_right_gripper_qpos().tolist(),
            },
            "trajectory": param_log,
            "success": bool(info.get('success', False)),
        }, f, indent=2)
    logger.info(f"参数轨迹已保存: {param_log_path}")

    env.close()
    return info.get('success', False)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
