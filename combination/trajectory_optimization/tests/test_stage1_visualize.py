#!/usr/bin/env python3
"""Stage 1 最优 grasp 可视化: 加载 best_grasp.npz 后跑真实 rollout_grasp_only(80帧), 并渲染视频."""

import sys, os, numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "output")
sys.path.insert(0, HERE)

from grasp_hawor import GraspSimulator
import sapien


def main():
    side = "right"
    grasp_path = os.path.join(OUTPUT, f"gripper_only_{side}", "stage1", "best_grasp.npz")
    d = np.load(grasp_path, allow_pickle=True)
    grasp_pose = {"pos": d["pos"], "R": d["R"]}
    print(f"加载最优 grasp: pos={grasp_pose['pos'].round(4)}, gripper_qpos={d['gripper_qpos']:.4f}")
    print(f"  参考: lift={d['obj_lift']*100:.2f}cm, peak_force={d['peak_grip_force']:.2f}N")

    output_dir = os.path.join(OUTPUT, f"gripper_only_{side}", "stage1_demo")
    frame_dir = os.path.join(output_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # 初始化场景 (views="god" 确保 render 系统已创建)
    sim = GraspSimulator(
        hawor_dir="/home/an/data/hawor/7",
        ras_dir="/home/an/data/ras/my_7mp4_result",
        mode="gripper_only",
        side=side,
        output_dir=output_dir,
        num_frames=80, start_frame=0, views="god",
        grasp_mode="hybrid", viewer=False,
    )
    sim._test_stage1_only = True
    sim._align_scene()
    sim.run()
    sim._compute_neutral_offsets()

    # 记录物体初始位姿 (rollout_grasp_only 内部会用到)
    sim._obj_initial_poses = {a.get_name(): a.get_pose() for a in sim.obj_actors}

    # 找到 god_view camera component
    cam_entity = None
    for entity in sim.scene.get_entities():
        if entity.get_name() == "god_view":
            cam_entity = entity
            break
    if cam_entity is None:
        raise RuntimeError("未找到 god_view camera")
    cam = None
    for comp in cam_entity.get_components():
        if isinstance(comp, sapien.pysapien.render.RenderCameraComponent):
            cam = comp
            break
    if cam is None:
        raise RuntimeError("未找到 RenderCameraComponent")

    # 包装 rollout_grasp_only, 让它每帧渲染并截图
    n_frames = 80
    original_physics_step = None

    def make_recording_step(sim_instance):
        nonlocal original_physics_step
        from grasp_hawor import physics_step as orig_step
        original_physics_step = orig_step

        def wrapped_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                         arm_targets, gripper_target1, gripper_target2, scene,
                         lock_root_pose=None, virtual_lock_targets=None):
            # 先执行真实物理步
            result = orig_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                               arm_targets, gripper_target1, gripper_target2, scene,
                               lock_root_pose=lock_root_pose,
                               virtual_lock_targets=virtual_lock_targets)
            # 截图 (每步都渲染, 后续可降采样)
            scene.update_render()
            cam.take_picture()
            rgb = cam.get_picture("Color")[..., :3]
            bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
            sim_instance._stage1_frame_buffer.append(bgr.copy())
            return result

        return wrapped_step

    import grasp_hawor
    sim._stage1_frame_buffer = []
    grasp_hawor.physics_step = make_recording_step(sim)

    # 运行真实 Stage 1 rollout
    result = sim.rollout_grasp_only(grasp_pose, side, strategy='pd_then_lock', n_frames=n_frames)

    # 恢复原始 physics_step
    grasp_hawor.physics_step = original_physics_step

    # 保存图片 & 合成视频
    fps = 10
    h, w = sim._stage1_frame_buffer[0].shape[:2] if sim._stage1_frame_buffer else (1080, 1920)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, f"stage1_demo_{side}.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    saved = 0
    for i, frame in enumerate(sim._stage1_frame_buffer):
        # 每 2 帧保存一张图 (降采样), 视频用全部
        if i % 2 == 0 or i == len(sim._stage1_frame_buffer) - 1:
            cv2.imwrite(os.path.join(frame_dir, f"frame_{i:04d}.jpg"), frame)
            saved += 1
        writer.write(frame)
    writer.release()

    print(f"\n{'='*50}")
    print(f"Stage 1 真实 rollout 可视化完成")
    print(f"{'='*50}")
    print(f"  obj_lift={result.get('obj_lift', 0)*100:.2f}cm")
    print(f"  peak_force={result.get('peak_grip_force', 0):.2f}N")
    print(f"  both_contact={result.get('both_contact_count', 0)}/{n_frames}")
    print(f"  保存图片: {saved} 帧 → {frame_dir}")
    print(f"  视频: {video_path} ({len(sim._stage1_frame_buffer)} 帧)")


if __name__ == "__main__":
    main()
