"""
apply_pose_changes.py — 根据位姿变化JSON对GLB应用变换

功能说明:
    从 pose_changes.json 中读取各物体在不同阶段的 T 矩阵，
    将初始GLB中的物体变换到指定阶段的位姿，输出新的GLB文件。

    pose_changes.json 由 mainv2.py 自动生成，记录了每个物体在各阶段的位姿变化:
      - initial:          Stage3刚生成后的初始位姿 (未做任何精修)
      - basic_refinement:  基础精修后的位姿 (floor/wall/embedded对齐)
      - stage5:           Stage5语义精修后的位姿 (物体间支撑关系精修)
      - physics:          物理仿真验证后的位姿 (SAPIEN仿真修正)

坐标系说明:
    pose_changes.json 中的 T 矩阵是 z-up 坐标系下的变换。
    GLB 文件是 y-up 坐标系。
    本工具在应用变换时自动处理坐标系转换:
      GLB(y-up) → z-up → 应用T矩阵 → y-up → 输出GLB

用法:
    # 查看可用阶段 (不指定 --stage 时自动列出)
    python tools/apply_pose_changes.py --glb output_v2/hoi4d/final_scene_initial.glb --json output_v2/hoi4d/pose_changes.json

    # 应用特定阶段的变换
    python tools/apply_pose_changes.py --glb output_v2/hoi4d/final_scene_initial.glb --json output_v2/hoi4d/pose_changes.json --stage basic_refinement

    # 应用Stage5变换
    python tools/apply_pose_changes.py --glb output_v2/hoi4d/final_scene_initial.glb --json output_v2/hoi4d/pose_changes.json --stage stage5

    # 指定输出文件名
    python tools/apply_pose_changes.py --glb output_v2/hoi4d/final_scene_initial.glb --json output_v2/hoi4d/pose_changes.json --stage stage5 --output my_result.glb

    # 列出所有可用阶段
    python tools/apply_pose_changes.py --glb output_v2/hoi4d/final_scene_initial.glb --json output_v2/hoi4d/pose_changes.json --list_stages

输出文件说明:
    不指定 --output 时，输出文件自动命名为:
      {输入GLB路径}_{阶段名}.glb

    例如:
      --glb final_scene_initial.glb --stage basic_refinement
      → 输出: final_scene_initial_basic_refinement.glb

      --glb final_scene_initial.glb --stage stage5
      → 输出: final_scene_initial_stage5.glb

    输出GLB中:
      - 有对应阶段数据的物体: 应用该阶段的T矩阵变换
      - 无对应阶段数据的物体: 保持原始GLB中的位姿不变

注意事项:
    - 输入GLB建议使用 final_scene_initial.glb (初始位姿)，
      而非 final_scene_basic.glb (可能已包含后续阶段的变换)
    - 如果物体在跨类去重中被合并(如chair被合并到toy)，
      该物体不会出现在pose_changes.json中，输出时保持原样
"""

import argparse
import json
import os
import numpy as np
import trimesh


def load_pose_changes(json_path):
    """加载位姿变化JSON文件

    Args:
        json_path: pose_changes.json 文件路径，由 mainv2.py 生成

    Returns:
        dict: {物体key: {category, instance_idx, relation, stages: {阶段名: {T_matrix, position, bounds_min, bounds_max, center}}}}
              物体key格式为 "{category}_{instance_idx}"，如 "table_0", "cup_0"
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def get_available_stages(pose_data):
    """获取pose_data中所有出现过的阶段名

    遍历所有物体的stages字典，收集所有阶段名并排序返回。
    典型阶段: initial, basic_refinement, stage5, physics

    Args:
        pose_data: load_pose_changes() 返回的字典

    Returns:
        list[str]: 排序后的阶段名列表，如 ['basic_refinement', 'initial', 'stage5']
    """
    stages = set()
    for obj_key, obj_data in pose_data.items():
        if "stages" in obj_data:
            stages.update(obj_data["stages"].keys())
    return sorted(stages)


def apply_stage_transform(glb_path, pose_data, target_stage, output_path=None):
    """将GLB中所有物体变换到指定阶段的位姿

    坐标系转换流程 (对每个物体):
      1. 从GLB加载mesh (y-up坐标系)
      2. y-up → z-up 变换
      3. 应用目标阶段的T矩阵 (z-up坐标系下的变换)
      4. z-up → y-up 变换
      5. 添加到输出场景

    对于pose_changes.json中没有目标阶段数据的物体，保持原始GLB位姿不变。

    Args:
        glb_path: 输入GLB文件路径，建议使用 final_scene_initial.glb
        pose_data: load_pose_changes() 返回的字典
        target_stage: 目标阶段名，如 "initial", "basic_refinement", "stage5"
        output_path: 输出GLB文件路径，None则自动生成 ({输入路径}_{阶段名}.glb)

    Returns:
        str: 输出GLB文件路径
    """
    scene = trimesh.load(glb_path)

    y_up_to_z_up = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    z_up_to_y_up = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ])

    new_scene = trimesh.Scene()

    for node_name in scene.geometry_names:
        geometry = scene.geometry[node_name]

        obj_data = pose_data.get(node_name)
        if obj_data is None or "stages" not in obj_data or target_stage not in obj_data["stages"]:
            print(f"  ⚠️ {node_name}: 无 {target_stage} 阶段数据, 保持原样")
            new_scene.add_geometry(geometry, node_name=node_name)
            continue

        T_target = np.array(obj_data["stages"][target_stage]["T_matrix"])

        mesh = geometry.copy()
        mesh.apply_transform(z_up_to_y_up)
        mesh.apply_transform(T_target)
        mesh.apply_transform(y_up_to_z_up)

        new_scene.add_geometry(mesh, node_name=node_name)
        pos = T_target[:3, 3]
        print(f"  ✅ {node_name}: {target_stage} position=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    new_scene.apply_transform(z_up_to_y_up)

    if output_path is None:
        base, ext = os.path.splitext(glb_path)
        output_path = f"{base}_{target_stage}{ext}"

    new_scene.export(output_path)
    print(f"\n💾 已保存: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="根据位姿变化JSON对GLB应用变换")
    parser.add_argument("--glb", required=True, help="输入GLB文件路径 (通常用 initial)")
    parser.add_argument("--json", required=True, help="位姿变化JSON文件路径")
    parser.add_argument("--stage", type=str, default=None,
                        help="目标阶段 (initial/basic_refinement/stage5). 不指定则列出可用阶段")
    parser.add_argument("--output", type=str, default=None, help="输出GLB文件路径")
    parser.add_argument("--list_stages", action="store_true", help="列出所有可用阶段")
    args = parser.parse_args()

    pose_data = load_pose_changes(args.json)
    available = get_available_stages(pose_data)

    if args.list_stages or args.stage is None:
        print(f"📋 可用阶段: {available}")
        for obj_key, obj_data in pose_data.items():
            obj_stages = list(obj_data.get("stages", {}).keys())
            print(f"  {obj_key}: {obj_stages}")
        if args.stage is None:
            return

    if args.stage not in available:
        print(f"❌ 阶段 '{args.stage}' 不存在. 可用: {available}")
        return

    apply_stage_transform(args.glb, pose_data, args.stage, args.output)


if __name__ == "__main__":
    main()
