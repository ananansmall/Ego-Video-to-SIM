"""
验证两个 chair 在 SAM3D 阶段输入点云的大小差异和输出 T_matrix 的尺度因子。
"""
import json
import numpy as np

pose_path = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/2chair-bottle_vggt_omega/pose_changes.json"

with open(pose_path, 'r') as f:
    data = json.load(f)

objects = data["objects"]

for name in ["chair_0", "chair_1"]:
    obj = objects[name]
    initial = obj["stages"]["initial"]
    T = np.array(initial["T_matrix"])
    R_cols = T[:3, :3]

    # 列向量长度 = 尺度因子
    col_scales = [np.linalg.norm(R_cols[:, i]) for i in range(3)]
    mean_scale = np.mean(col_scales)

    # bounds 尺寸
    bounds_min = np.array(initial["bounds_min"])
    bounds_max = np.array(initial["bounds_max"])
    size = bounds_max - bounds_min

    print(f"\n{name}:")
    print(f"  列向量长度 (尺度): {col_scales}")
    print(f"  平均尺度: {mean_scale:.3f}")
    print(f"  最终尺寸 (x,y,z): {size}")
    print(f"  T_matrix:")
    print(T)

print("\n\n两个 chair 尺度比:", objects["chair_1"]["stages"]["initial"]["T_matrix"])
scale_0 = np.mean([np.linalg.norm(np.array(objects["chair_0"]["stages"]["initial"]["T_matrix"][:3, :3])[:, i]) for i in range(3)])
scale_1 = np.mean([np.linalg.norm(np.array(objects["chair_1"]["stages"]["initial"]["T_matrix"][:3, :3])[:, i]) for i in range(3)])
print(f"chair_0 scale: {scale_0:.3f}")
print(f"chair_1 scale: {scale_1:.3f}")
print(f"scale ratio: {scale_1/scale_0:.3f}")
print(f"最终尺寸比: {np.array(objects['chair_1']['stages']['initial']['bounds_max']) - np.array(objects['chair_1']['stages']['initial']['bounds_min'])}")
print(f"             {np.array(objects['chair_0']['stages']['initial']['bounds_max']) - np.array(objects['chair_0']['stages']['initial']['bounds_min'])}")
