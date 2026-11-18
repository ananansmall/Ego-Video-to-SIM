#!/usr/bin/env python3
"""run_ef3r_phys.py — EF3R 直连链物理接触抓取 (B 阶段泛化验证)。

与 run_agile_phys.py 的差别只在数据源: 手部轨迹 = EF3R 自标定链 (非 AGILE
物理优化), 物体 = EF3R 点云分割实例 (73×77×24mm, 非 SAM3D 50×47×25mm)。
抓握宽度/交接/接物桌全部由物体几何与 URDF 自动推导 —— 用于验证同一套逻辑
跨数据源泛化 (B 阶段)。

用法:
    python scripts/run_ef3r_phys.py                # 渲染 GIF
    python scripts/run_ef3r_phys.py --phys-only    # 只测指标 (快)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb_large_range_sim as cb

EF_DIR = "/path/to/project_data/mano_ws/EgoFound3R/outputs/pickplace_456483158"
TRAJ = os.path.join(EF_DIR, "ef3r_right_targets.npz")
OBJ = os.path.join(EF_DIR, "object_instance.npz")
CLOUD = os.path.join(EF_DIR, "pointcloud/scene_cloud.npz")

if __name__ == "__main__":
    phys_only = "--phys-only" in sys.argv
    name = "ef3r456_phys"
    print("=" * 70, flush=True)
    print(f"EF3R 直连链物理接触抓取 (object_physics=True)  name={name}  "
          f"phys_only={phys_only}", flush=True)
    print("=" * 70, flush=True)
    cb.main(targets=TRAJ, name=name, object_npz=OBJ, scene_cloud=CLOUD,
            object_physics=True, w_ori=1.0, phys_only=phys_only)
    print("\nRUN_EF3R_PHYS_DONE", flush=True)
