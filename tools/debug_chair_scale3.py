"""
验证两个 chair 的 VGGT world_points 在相机空间中的范围差异。
这需要加载实际的世界点数据。
"""
import json
import numpy as np

# 加载 scene_json 获取 SAM3D 最优帧信息
output_dir = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/2chair-bottle_vggt_omega"

with open(f"{output_dir}/coordinate_alignment.json", 'r') as f:
    coord = json.load(f)

# 加载 extrinsics (对齐后)
with open(f"{output_dir}/extrinsics.json", 'r') as f:
    extrinsics_data = json.load(f)

print("extrinsics_data keys:", list(extrinsics_data.keys()) if isinstance(extrinsics_data, dict) else "list")
print("type:", type(extrinsics_data))
