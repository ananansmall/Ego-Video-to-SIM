#!/usr/bin/env python3
"""run_agile_phys.py — AGILE 链物理接触抓取 (object_physics=True) 诊断/验证入口。

与 configs/agile456_large_range.json 的唯一差别: object_physics=True。
object_physics=True 时: 物体 kinematic 跟随至夹爪闭合到位 -> 切 dynamic + 重力,
之后只靠两指摩擦夹持 (真实动力学抓取), 不再刚体绑定到 pad 系。

用法:
    python scripts/run_agile_phys.py                # 渲染 GIF
    python scripts/run_agile_phys.py --phys-only    # 只测指标 (快)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cb_large_range_sim as cb

EF_DIR = "/path/to/project_data/mano_ws/EgoFound3R/outputs/pickplace_456483158"
TRAJ = "/path/to/project_data/mano_ws/AGILE/output/456483158/agile_right_targets_286.npz"
OBJ = os.path.join(EF_DIR, "object_instance_agile.npz")
CLOUD = os.path.join(EF_DIR, "pointcloud/scene_cloud.npz")

if __name__ == "__main__":
    phys_only = "--phys-only" in sys.argv
    name = "agile456_phys"
    print("=" * 70, flush=True)
    print(f"AGILE 链物理接触抓取 (object_physics=True)  name={name}  "
          f"phys_only={phys_only}", flush=True)
    print("=" * 70, flush=True)
    cb.main(targets=TRAJ, name=name, object_npz=OBJ, scene_cloud=CLOUD,
            object_physics=True, w_ori=1.0, phys_only=phys_only)
    print("\nRUN_AGILE_PHYS_DONE", flush=True)
