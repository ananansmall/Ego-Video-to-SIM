"""
验证两个 chair 的 SAM3D 输入点云在相机坐标系下的大小差异。

SAM3D 的 scale 直接取决于相机空间中点云的范围。
如果椅子距离相机远近不同，点云范围也不同，scale 就会不同。
"""
import json
import numpy as np

pose_path = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/2chair-bottle_vggt_omega/pose_changes.json"

with open(pose_path, 'r') as f:
    data = json.load(f)

objects = data["objects"]
coord_align = data["coordinate_alignment"]

# 从 pose_changes 获取 initial T (SAM3D 生成的世界坐标系变换)
chair0_T = np.array(objects["chair_0"]["stages"]["initial"]["T_matrix"])
chair1_T = np.array(objects["chair_1"]["stages"]["initial"]["T_matrix"])

chair0_pos = chair0_T[:3, 3]
chair1_pos = chair1_T[:3, 3]

# 获取 extrinsics (对齐后)
extrinsics = data["camera_extrinsics_after_alignment"]
# 用第一个椅子的最优帧 (需要找，这里用 frame 0 近似)
# 实际我们不知道哪个是最优帧，但我们知道椅子到原点的距离差异

# 椅子在世界坐标系中的位置
print("=== 世界坐标系中椅子的位置 ===")
print(f"chair_0 pos: {chair0_pos}")
print(f"chair_1 pos: {chair1_pos}")
print(f"距离原点: chair_0={np.linalg.norm(chair0_pos):.3f}, chair_1={np.linalg.norm(chair1_pos):.3f}")

# 从 T_matrix 的列向量获得尺度
def get_scale(T):
    R = T[:3, :3]
    return np.mean([np.linalg.norm(R[:, i]) for i in range(3)])

scale0 = get_scale(chair0_T)
scale1 = get_scale(chair1_T)

# 用 extrinsics 反算相机位置，看哪个 chair 离相机更远
# 取几个帧的相机位置
cam_positions = []
for frame_data in extrinsics[:5]:
    E = np.array(frame_data["extrinsic_aligned"])
    R = E[:3, :3]
    t = E[:3, 3]
    cam_pos = -R.T @ t  # world coordinates of camera
    cam_positions.append(cam_pos)

print("\n=== 前5帧相机在世界坐标系中的位置 ===")
for i, cp in enumerate(cam_positions):
    print(f"  frame {i}: cam at {cp}")

# 计算每个 chair 到相机的距离
print("\n=== 椅子到相机的距离 ===")
for i, cp in enumerate(cam_positions):
    d0 = np.linalg.norm(chair0_pos - cp)
    d1 = np.linalg.norm(chair1_pos - cp)
    print(f"  frame {i}: chair_0={d0:.3f}m, chair_1={d1:.3f}m")

print(f"\n=== 结论 ===")
print(f"chair_0 尺度: {scale0:.3f}")
print(f"chair_1 尺度: {scale1:.3f}")
print(f"比例: {scale1/scale0:.3f}")
print(f"")
print("SAM3D 的 scale 正比于相机空间中点云的大小。")
print("如果 chair_1 在相机空间中离相机更远/更大，")
print("SAM3D 就会输出更大的 scale 值。")
print(f"")
print(f"最终尺寸:")
b0 = np.array(objects["chair_0"]["stages"]["initial"]["bounds_max"]) - np.array(objects["chair_0"]["stages"]["initial"]["bounds_min"])
b1 = np.array(objects["chair_1"]["stages"]["initial"]["bounds_max"]) - np.array(objects["chair_1"]["stages"]["initial"]["bounds_min"])
print(f"  chair_0: {b0}")
print(f"  chair_1: {b1}")
print(f"  比例: {b1/b0}")
