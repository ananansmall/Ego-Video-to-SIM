"""诊断相机位置和渲染效果"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from pybullet_pipeline import RXWORLD_TO_SAPIEN, hawor_cam_to_sapien_pose

# 检查多个可能的 HaWoR 目录
hawor_dirs = [
    Path(__file__).parent.parent / "output" / "7_my_7mp4_result",
    Path(__file__).parent.parent / "hand_track" / "output" / "7",
]

import importlib.util
spec = importlib.util.spec_from_file_location("mod02", Path(__file__).parent.parent / "02_render_scene.py")
mod02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod02)

for hd in hawor_dirs:
    print(f"\n{'='*60}")
    print(f"HaWoR dir: {hd}")
    print(f"Exists: {hd.exists()}")
    if not hd.exists():
        continue
    try:
        R_c2w_all, t_c2w_all = mod02.load_hawor_c2w(hd)
        if R_c2w_all is None:
            print("  R_c2w_all is None")
            continue
        print(f"  Frames: {len(t_c2w_all)}")
        for i in [0, len(t_c2w_all)//2, len(t_c2w_all)-1]:
            cam_pos = RXWORLD_TO_SAPIEN @ t_c2w_all[i]
            cam_pos_sapien, cam_R_sapien = hawor_cam_to_sapien_pose(R_c2w_all[i], t_c2w_all[i])
            print(f"  Frame {i}: pos={cam_pos_sapien}, forward={cam_R_sapien[:,0]}")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

# 物体 Z 范围 (从之前的诊断)
print(f"\n{'='*60}")
print("Object Z range: [-0.0943, -0.0354]m")
print("Table top: -0.0963m")
print("Camera Z: ~0.004m (from summary)")
print("\n距离分析:")
print(f"  相机到桌面: {0.004 - (-0.0963):.4f}m = {(0.004 - (-0.0963))*100:.1f}cm")
print(f"  相机到物体最高点: {0.004 - (-0.0354):.4f}m = {(0.004 - (-0.0354))*100:.1f}cm")
print(f"  相机到物体最低点: {0.004 - (-0.0943):.4f}m = {(0.004 - (-0.0943))*100:.1f}cm")
