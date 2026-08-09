"""诊断相机位置 - 使用正确的 HaWoR 路径"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pybullet_pipeline import RXWORLD_TO_SAPIEN, hawor_cam_to_sapien_pose

HAWOR_DIR = Path("/home/an/data/hawor/7")
RAS_DIR = Path("/home/an/data/ras/my_7mp4_result")
TRANSFORM_PARAMS = Path(__file__).parent.parent / "output" / "alignment" / "transform_params.npz"

# 加载相机轨迹
rec_file = HAWOR_DIR / "reconstruction" / "hawor_results_0_113.npz"
print(f"Rec file: {rec_file}, exists: {rec_file.exists()}")
rec = np.load(str(rec_file), allow_pickle=True)
print(f"Keys: {list(rec.keys())}")

R_c2w_all = rec['R_c2w']
t_c2w_all = rec['t_c2w']
print(f"R_c2w shape: {R_c2w_all.shape}")
print(f"t_c2w shape: {t_c2w_all.shape}")

# 检查第一帧、中间帧、最后帧的相机位置
for i in [0, len(t_c2w_all)//2, len(t_c2w_all)-1]:
    cam_pos_sapien, cam_R_sapien = hawor_cam_to_sapien_pose(R_c2w_all[i], t_c2w_all[i])
    print(f"\nFrame {i}:")
    print(f"  pos (SAPIEN): {cam_pos_sapien}")
    print(f"  forward (X):  {cam_R_sapien[:,0]}")
    print(f"  up (Z):       {cam_R_sapien[:,2]}")

# 物体 Z 范围 (从之前的诊断)
print(f"\n{'='*60}")
print("Object Z range: [-0.0943, -0.0354]m")
print("Table top: -0.0963m")

# 计算相机到桌面/物体的距离
cam_z = float(cam_pos_sapien[2])  # 最后一帧
print(f"\nCamera Z (last frame): {cam_z:.4f}m")
print(f"  相机到桌面: {cam_z - (-0.0963):.4f}m = {(cam_z - (-0.0963))*100:.1f}cm")
print(f"  相机到物体最高点: {cam_z - (-0.0354):.4f}m = {(cam_z - (-0.0354))*100:.1f}cm")
print(f"  相机到物体最低点: {cam_z - (-0.0943):.4f}m = {(cam_z - (-0.0943))*100:.1f}cm")

# 检查 img_focal
if 'img_focal' in rec:
    img_focal = float(rec['img_focal'])
    print(f"\nimg_focal: {img_focal}")
