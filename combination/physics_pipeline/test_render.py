"""测试 PyBullet 渲染 - 检查相机视角和夹爪加载 (修复版)"""
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pybullet_pipeline import (
    PyBulletPipeline, hawor_cam_to_sapien_pose,
    RXWORLD_TO_SAPIEN, _compute_analytical_gripper_pose,
    GRIPPER_INIT_OPEN, VIDEO_WIDTH, VIDEO_HEIGHT
)
import pybullet as p

HAWOR_DIR = Path("/home/an/data/hawor/7")
RAS_DIR = Path("/home/an/data/ras/my_7mp4_result")
GLB_PATH = RAS_DIR / "final_scene.glb"
TRANSFORM_PARAMS = Path(__file__).parent.parent / "output" / "alignment" / "transform_params.npz"

# 加载 HaWoR 数据
rec_file = HAWOR_DIR / "reconstruction" / "hawor_results_0_113.npz"
rec = np.load(str(rec_file), allow_pickle=True)
R_c2w_all = rec['R_c2w']
t_c2w_all = rec['t_c2w']

print("="*60)
print("测试 PyBullet 单夹爪渲染")
print("="*60)

# 创建单夹爪 pipeline
pipeline = PyBulletPipeline(gui=False, single_gripper=True)
print(f"Camera FOV: {pipeline.cam_fov_deg:.1f}°")

# 加载 GLB 物体
print("\n加载 GLB 物体...")
pipeline.load_glb_objects(str(GLB_PATH), str(TRANSFORM_PARAMS))

# 计算第一帧的夹爪位姿
print("\n计算第一帧夹爪位姿...")
import importlib.util
spec = importlib.util.spec_from_file_location("mod02", Path(__file__).parent.parent / "02_render_scene.py")
mod02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod02)

# 使用 04 的 load_hawor_data 和 MANOLayer
spec04 = importlib.util.spec_from_file_location("mod04", Path(__file__).parent.parent / "04_physics_simulation.py")
mod04 = importlib.util.module_from_spec(spec04)
spec04.loader.exec_module(mod04)

hawor_data = mod04.load_hawor_data(str(HAWOR_DIR), hand_idx=1)  # right hand
betas_mean = hawor_data["pred_betas"][0].astype(np.float32)
print(f"  betas shape: {betas_mean.shape}")

# 使用 04 的 MANOLayer 路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from mano_layer import MANOLayer
mano_layer = MANOLayer("right", betas_mean)
_, j = mod02.compute_mano_joints(mano_layer, hawor_data["pred_rot"][0], hawor_data["pred_hand_pose"][0], hawor_data["pred_trans"][0])
joints_sapien = (RXWORLD_TO_SAPIEN @ j.T).T

mano_wrist = joints_sapien[0, :3]
mano_finger1 = joints_sapien[4, :3]  # 拇指尖
mano_finger2 = joints_sapien[8, :3]  # 食指尖

print(f"  MANO wrist: {mano_wrist}")
print(f"  MANO finger1 (thumb): {mano_finger1}")
print(f"  MANO finger2 (index): {mano_finger2}")

root_pos, root_R, joint1, joint2 = _compute_analytical_gripper_pose(
    mano_wrist, mano_finger1, mano_finger2)
print(f"  Gripper root pos: {root_pos}")
print(f"  Gripper joint1: {joint1:.4f}, joint2: {joint2:.4f}")

# 设置夹爪位姿
from pytransform3d import rotations as pr
root_quat_wxyz = pr.quaternion_from_matrix(root_R)
pb_quat = [root_quat_wxyz[1], root_quat_wxyz[2], root_quat_wxyz[3], root_quat_wxyz[0]]
p.resetBasePositionAndOrientation(pipeline.robot_id, root_pos.tolist(), pb_quat)

# 设置夹爪关节
for i, idx in enumerate(pipeline.gripper_joint_indices):
    target = joint1 if i == 0 else joint2
    p.resetJointState(pipeline.robot_id, idx, target, targetVelocity=0.0)

# 渲染 FPV
print("\n渲染 FPV...")
cam_pos, cam_R = hawor_cam_to_sapien_pose(R_c2w_all[0], t_c2w_all[0])
print(f"  Camera pos: {cam_pos}")
print(f"  Camera forward: {cam_R[:,0]}")
print(f"  Camera up: {cam_R[:,2]}")

bgr = pipeline.render_frame(cam_pos, cam_R)
print(f"  FPV frame mean: {bgr.mean():.1f}")

out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)
cv2.imwrite(str(out_dir / "test_fpv_gripper.png"), bgr)
print(f"  Saved: test_fpv_gripper.png")

# 渲染第三人称视角 (从后上方看夹爪和物体)
print("\n渲染第三人称视角...")
scene_center = root_pos
fixed_cam_pos = scene_center + np.array([-0.3, -0.3, 0.3])
forward = scene_center - fixed_cam_pos
forward /= np.linalg.norm(forward)
up_world = np.array([0, 0, 1.0])
right = np.cross(forward, up_world)
right /= np.linalg.norm(right)
cam_up = np.cross(right, forward)
fixed_cam_R = np.column_stack([forward, -right, cam_up])

bgr3 = pipeline.render_frame(fixed_cam_pos, fixed_cam_R)
print(f"  Third-person frame mean: {bgr3.mean():.1f}")
cv2.imwrite(str(out_dir / "test_thirdperson_gripper.png"), bgr3)
print(f"  Saved: test_thirdperson_gripper.png")

# 测试翻转 up 向量 (检查是否上下颠倒)
print("\n测试翻转 up 向量 (检查图像方向)...")
cam_R_flipped = cam_R.copy()
cam_R_flipped[:, 2] = -cam_R_flipped[:, 2]  # 翻转 up
cam_R_flipped[:, 1] = -cam_R_flipped[:, 1]  # 翻转 left (保持右手系)
bgr_flipped = pipeline.render_frame(cam_pos, cam_R_flipped)
print(f"  Flipped up frame mean: {bgr_flipped.mean():.1f}")
cv2.imwrite(str(out_dir / "test_fpv_flipped.png"), bgr_flipped)
print(f"  Saved: test_fpv_flipped.png")

# 对比: 垂直翻转原始图像
bgr_vflip = cv2.flip(bgr, 0)
cv2.imwrite(str(out_dir / "test_fpv_vflip.png"), bgr_vflip)
print(f"  Saved: test_fpv_vflip.png")

# 检查夹爪是否在视野内
print("\n检查夹爪位置...")
print(f"  Gripper pos: {root_pos}")
print(f"  Camera pos:  {cam_pos}")
print(f"  Distance: {np.linalg.norm(root_pos - cam_pos):.4f}m")
print(f"  Gripper relative to camera: {root_pos - cam_pos}")
# 相机 forward 方向
cam_forward = cam_R[:, 0]
to_gripper = root_pos - cam_pos
to_gripper_norm = to_gripper / np.linalg.norm(to_gripper)
dot = np.dot(cam_forward, to_gripper_norm)
print(f"  Dot(forward, to_gripper): {dot:.4f} (>0 = 在前方)")

print("\n" + "="*60)
print("测试完成! 检查 output/ 目录下的图片")
