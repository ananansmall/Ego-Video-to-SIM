#!/usr/bin/env python3
"""
从 pose_changes.json 重建各阶段 GLB
===================================

用途:
  - 不重新运行流水线, 仅利用 pose_changes.json 中的 T_matrix 重建 GLB
  - 便于对比各阶段位姿变化
  - 调试 SP精修/穿模修复效果

GLB 命名与 mainv2.py 完全一致:
  initial          → final_scene_initial.glb
  basic_refinement → final_scene_basic.glb
  stage4           → final_scene_stage4.glb
  stage5           → final_scene_stage5.glb (无stage4时)
                   或 final_scene_stage4_5.glb (有stage4时)

输入:
  - all_instances.pkl (含 original_mesh)
  - pose_changes.json (含各阶段 T_matrix)

用法:
  python tools/rebuild_glbs_from_json.py --scene_dir output_v2/121_xxx
  python tools/rebuild_glbs_from_json.py --scene_dir output_v2/121_xxx --stage stage5
  python tools/rebuild_glbs_from_json.py --scene_dir output_v2/121_xxx --has_stage4
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

ZUP_TO_YUP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
])

# 阶段名 → mainv2.py 中的 GLB 文件名映射
STAGE_TO_GLB = {
    "initial": "final_scene_initial.glb",
    "basic_refinement": "final_scene_basic.glb",
    "stage4": "final_scene_stage4.glb",
    # stage5 的文件名取决于是否有 stage4, 在 main() 中动态决定
}


def load_all_instances(pkl_path):
    """加载 all_instances.pkl"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'all_instances' in data:
        return data['all_instances']
    return data


def load_pose_changes(json_path):
    """加载 pose_changes.json"""
    with open(json_path, 'r') as f:
        return json.load(f)


def apply_T_and_save_glb(all_instances, pose_data, stage_name, output_path, filename):
    """根据 pose_data 中指定阶段的 T_matrix 重建 GLB"""
    scene = trimesh.Scene()

    count = 0
    for category, instances in all_instances.items():
        for i, instance_info in enumerate(instances):
            obj_key = f"{category}_{i}"
            if obj_key not in pose_data:
                continue

            stages = pose_data[obj_key].get("stages", {})
            if stage_name not in stages:
                continue

            T = np.array(stages[stage_name]["T_matrix"])
            mesh = instance_info['original_mesh'].copy()
            mesh.apply_transform(T)
            scene.add_geometry(mesh, node_name=f"{category}_{i}")
            count += 1

    scene.apply_transform(ZUP_TO_YUP)
    glb_path = os.path.join(output_path, filename)
    scene.export(glb_path)
    print(f"  💾 {filename}: {count} 个物体", flush=True)
    return glb_path


def main():
    parser = argparse.ArgumentParser(description="从 pose_changes.json 重建各阶段 GLB")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景输出目录")
    parser.add_argument("--stage", type=str, default=None,
                        help="只重建指定阶段 (initial/basic_refinement/stage4/stage5), 默认全部")
    parser.add_argument("--has_stage4", action="store_true",
                        help="标记有stage4 (影响stage5的GLB命名: final_scene_stage4_5.glb)")
    args = parser.parse_args()

    pkl_path = os.path.join(args.scene_dir, "all_instances.pkl")
    json_path = os.path.join(args.scene_dir, "pose_changes.json")

    if not os.path.exists(pkl_path):
        print(f"❌ 找不到: {pkl_path}", flush=True)
        sys.exit(1)
    if not os.path.exists(json_path):
        print(f"❌ 找不到: {json_path}", flush=True)
        sys.exit(1)

    print(f"📂 场景目录: {args.scene_dir}", flush=True)
    print(f"📦 加载 all_instances.pkl...", flush=True)
    all_instances = load_all_instances(pkl_path)
    print(f"   物体类别: {list(all_instances.keys())}", flush=True)

    print(f"📦 加载 pose_changes.json...", flush=True)
    pose_data = load_pose_changes(json_path)

    # 确定可用的阶段
    sample_key = list(pose_data.keys())[0]
    available_stages = list(pose_data[sample_key]["stages"].keys())
    print(f"   可用阶段: {available_stages}", flush=True)

    # 自动检测是否有 stage4
    has_stage4 = args.has_stage4 or "stage4" in available_stages

    # 构建 stage5 的文件名
    if has_stage4:
        STAGE_TO_GLB["stage5"] = "final_scene_stage4_5.glb"
    else:
        STAGE_TO_GLB["stage5"] = "final_scene_stage5.glb"

    # 过滤阶段
    stages_to_rebuild = [args.stage] if args.stage else available_stages
    for s in stages_to_rebuild:
        if s not in available_stages:
            print(f"⚠️  阶段 '{s}' 不在 pose_changes.json 中, 跳过", flush=True)
            continue

    print(f"\n🔧 重建 GLB (命名与 mainv2.py 一致)...", flush=True)
    for stage in stages_to_rebuild:
        if stage not in available_stages:
            continue
        filename = STAGE_TO_GLB.get(stage, f"final_scene_{stage}.glb")
        apply_T_and_save_glb(all_instances, pose_data, stage, args.scene_dir, filename)

    print(f"\n✅ 完成! 输出目录: {args.scene_dir}", flush=True)
    print(f"\n📋 GLB 命名对照:", flush=True)
    for stage, glb_name in STAGE_TO_GLB.items():
        if stage in available_stages:
            print(f"   {stage:20s} → {glb_name}", flush=True)


if __name__ == "__main__":
    main()
