"""
后处理统一管线: Stage 4 + Stage 5 分层分装
============================================

本文件负责: 在 mainv2 完成 Stage 1~3 (含基础精修) 后,
对已有输出目录进行可选的后处理。

三个独立功能, 可任意组合调用:
  1. Stage 4:  视觉-空间对齐 (ICP + MASt3R)
  2. Stage 5.1: 细化 "supported by other objects" 关系 (VLM)
  3. Stage 5.2: SP 空间位置精修 (纯几何)

数据流:
  final_scene.glb (Stage3产物)
       │
       ├──→ [Stage4] ──→ final_scene_stage4.glb
       │                      │
       │                      ├──→ [Stage5] ──→ final_scene_stage4_5.glb
       │
       └──→ [Stage5] ──→ final_scene_stage5.glb

  GLB 命名按参与的 Stage 决定:
    - final_scene.glb          = Stage3 (基础)
    - final_scene_stage4.glb   = Stage4 产物
    - final_scene_stage5.glb   = 仅 Stage5 产物
    - final_scene_stage4_5.glb = Stage4+5 产物

  Stage5 输入: 先找 final_scene_stage4.glb, 没有就用 final_scene.glb
  重复运行时直接覆盖, 不删除旧文件

坐标系约定:
  - 优先加载 all_instances.pkl (z-up, T矩阵独立, 与mainv2内部一致)
  - 回退到 GLB 加载 (y-up → z-up, T已烘焙进mesh, 精度有损)
  - 保存时统一用 save_final_glb (z-up → y-up)


使用方式:
  # 一键全功能: Stage4 + Stage5
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage4 --stage5

  # 只运行 Stage 5 (关系细化 + SP精修)
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage5

  # 只运行 Stage 4
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage4

  # 只细化关系 (Stage 5.1)
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_refine_relations

  # 只做 SP 精修 (Stage 5.2), 手动指定关系JSON
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_sp_refinement \
      --relations_json output_v2/hoi4d/hoi4d_refined.json

  # 指定 VLM 模型
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage5 \
      --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B
"""

import os
import sys
import json
import argparse
import logging
import time
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import trimesh
import cv2
import torch

os.environ["LIDRA_SKIP_INIT"] = "true"


VLM_CHECKPOINT_CANDIDATES = [
    "/mnt/data/lza/models/Qwen3.5-9B",
    "/mnt/data_8THDD/lza/models/Qwen3.5-9B",
]

Y_UP_TO_Z_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)

Z_UP_TO_Y_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.float64)


def _resolve_vlm_checkpoint(vlm_checkpoint):
    if vlm_checkpoint and os.path.exists(vlm_checkpoint):
        return vlm_checkpoint
    for cand in VLM_CHECKPOINT_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return vlm_checkpoint


def setup_logging(scene_dir, prefix="post_pipeline"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(scene_dir, f"{prefix}_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_filename, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    import builtins
    builtins._original_print = builtins.print
    def new_print(*args, sep=' ', end='\n', file=None, flush=False):
        msg = sep.join(str(arg) for arg in args) + end
        if not msg.endswith('\n'):
            msg += '\n'
        logger.info(msg.rstrip('\n'))
        builtins._original_print(*args, sep=sep, end=end, file=file or sys.stdout, flush=flush)
    builtins.print = new_print
    return log_filename


def discover_scene_files(scene_dir, run_stage4=False, run_stage5=False):
    """自动发现场景目录中的所有必要文件

    GLB 发现策略:
      - 运行 Stage4 时: 从 final_scene.glb (Stage3产物) 开始
      - 仅运行 Stage5 时: 先找 final_scene_stage4.glb, 没有就用 final_scene.glb
      - 不运行任何 Stage: 使用最完整的版本
    """
    result = {'scene_id': os.path.basename(os.path.abspath(scene_dir))}

    for name in ['color', 'depth', 'extrinsics', 'optimal_frames', 'keyframes']:
        p = os.path.join(scene_dir, name)
        result[f'{name}_dir'] = p if os.path.isdir(p) else None

    intrinsic_txt = os.path.join(scene_dir, 'intrinsic.txt')
    if not os.path.isfile(intrinsic_txt):
        intrinsic_txt = os.path.join(scene_dir, 'intrinsics.txt')
    result['intrinsic_txt'] = intrinsic_txt if os.path.isfile(intrinsic_txt) else None

    result['pkl_path'] = None
    for pkl_name in ['all_instances.pkl', 'all_instances_stage4.pkl']:
        p = os.path.join(scene_dir, pkl_name)
        if os.path.isfile(p):
            result['pkl_path'] = p
            break

    if run_stage4:
        glb_priority = ['final_scene.glb']
    elif run_stage5:
        glb_priority = ['final_scene_stage4.glb', 'final_scene.glb']
    else:
        glb_priority = ['final_scene_stage4_5.glb', 'final_scene_stage5.glb', 'final_scene_stage4.glb', 'final_scene.glb']

    result['glb_path'] = None
    for glb_name in glb_priority:
        p = os.path.join(scene_dir, glb_name)
        if os.path.isfile(p):
            result['glb_path'] = p
            break
    if result['glb_path'] is None:
        glb_cands = [f for f in os.listdir(scene_dir) if f.endswith('.glb')]
        result['glb_path'] = os.path.join(scene_dir, glb_cands[0]) if glb_cands else None

    # Stage1 JSON
    stage1_json = None
    for f in os.listdir(scene_dir):
        if f.endswith('_stage1.json') and 'refined' not in f:
            stage1_json = os.path.join(scene_dir, f)
            break
    if stage1_json is None:
        for f in os.listdir(scene_dir):
            if f.endswith('.json') and 'stage1' in f:
                stage1_json = os.path.join(scene_dir, f)
                break
    result['stage1_json'] = stage1_json

    return result


def load_vggt_results(scene_dir, files):
    """从磁盘加载 VGGT 中间结果"""
    colors, depth_list, extrinsics_list = [], [], []

    color_dir = files.get('color_dir')
    if color_dir and os.path.isdir(color_dir):
        for cf in sorted(os.listdir(color_dir), key=lambda x: int(os.path.splitext(x)[0])):
            if cf.endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(color_dir, cf))
                if img is not None:
                    colors.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    depth_dir = files.get('depth_dir')
    if depth_dir and os.path.isdir(depth_dir):
        for df in sorted(os.listdir(depth_dir), key=lambda x: int(os.path.splitext(x)[0])):
            if df.endswith('.png'):
                d = cv2.imread(os.path.join(depth_dir, df), cv2.IMREAD_UNCHANGED)
                if d is not None:
                    depth_list.append(d.astype(np.float32) / 1000.0)

    extrinsics_dir = files.get('extrinsics_dir')
    if extrinsics_dir and os.path.isdir(extrinsics_dir):
        for ef in sorted(os.listdir(extrinsics_dir), key=lambda x: int(os.path.splitext(x)[0])):
            if ef.endswith('.txt'):
                ext = np.loadtxt(os.path.join(extrinsics_dir, ef))
                if ext.shape == (4, 4):
                    extrinsics_list.append(ext.astype(np.float64))

    intrinsic = None
    intrinsic_txt = files.get('intrinsic_txt')
    if intrinsic_txt and os.path.isfile(intrinsic_txt):
        intrinsic = np.loadtxt(intrinsic_txt).astype(np.float64)

    print(f"   加载 VGGT 数据: {len(colors)} 颜色, {len(depth_list)} 深度, {len(extrinsics_list)} 外参", flush=True)

    return {
        'colors': colors if colors else None,
        'depths': depth_list if depth_list else None,
        'extrinsics': extrinsics_list if extrinsics_list else None,
        'intrinsic': intrinsic,
    }


def load_instances(scene_dir, prefer_stage4=False):
    """加载 all_instances, 优先 pkl, 回退 GLB

    pkl 方式: all_instances.pkl 保存的是 mainv2 基础精修后的原始数据
              original_mesh + T 独立, z-up 空间, 与 SP 函数完全兼容

    GLB 方式: T 已烘焙进 mesh 顶点, 加载后 T=I, 只能做增量调整
              且 y-up→z-up 转换可能有精度损失

    参数:
        scene_dir: 场景输出目录
        prefer_stage4: 是否优先加载 stage4 的 pkl (如果存在)
    返回:
        (all_instances, source_description)
    """
    import pickle

    pkl_candidates = []
    if prefer_stage4:
        pkl_candidates.append(os.path.join(scene_dir, "all_instances_stage4.pkl"))
    pkl_candidates.append(os.path.join(scene_dir, "all_instances.pkl"))

    for pkl_path in pkl_candidates:
        if os.path.isfile(pkl_path):
            with open(pkl_path, 'rb') as f:
                pkl_data = pickle.load(f)
            if isinstance(pkl_data, dict) and 'all_instances' in pkl_data:
                all_instances = pkl_data['all_instances']
            else:
                all_instances = pkl_data
            return all_instances, f"pkl: {os.path.basename(pkl_path)}"

    glb_priority = ['final_scene.glb']
    glb_path = None
    for name in glb_priority:
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            glb_path = p
            break
    if glb_path is None:
        glb_cands = [f for f in os.listdir(scene_dir) if f.endswith('.glb')]
        if glb_cands:
            glb_path = os.path.join(scene_dir, glb_cands[0])

    if glb_path is None:
        return None, "未找到 pkl 或 GLB"

    all_instances = _load_instances_from_glb(glb_path)
    return all_instances, f"glb: {os.path.basename(glb_path)} (回退, T已烘焙)"


def _load_instances_from_glb(glb_path):
    """从 GLB 加载 all_instances (y-up → z-up)

    关键: trimesh 导出GLB时, scene.apply_transform(Z_UP_TO_Y_UP) 被拆分为:
      - geometry顶点: 保持z-up不变
      - node transform: 存储Z_UP_TO_Y_UP变换
    所以加载时必须用 graph.get(frame_to=node) 获取完整的world→node变换,
    而不是只对geometry做Y_UP_TO_Z_UP (那样会得到错误结果).
    """
    scene = trimesh.load(glb_path)
    all_instances = defaultdict(list)
    g = scene.graph

    for node_name in g.nodes_geometry:
        if node_name == 'world':
            continue
        parts = node_name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        category = parts[0]
        _, geom_name = g[node_name]
        if geom_name not in scene.geometry:
            continue
        mesh = scene.geometry[geom_name].copy()
        transform_to_node, _ = g.get(frame_to=node_name)
        mesh.apply_transform(transform_to_node)
        mesh.apply_transform(Y_UP_TO_Z_UP)
        all_instances[category].append({
            "original_mesh": mesh,
            "T": np.eye(4),
        })
    return dict(all_instances)


def save_glb(all_instances, output_path, filename):
    """保存 GLB (z-up → y-up), 与 mainv2.save_final_glb 逻辑一致"""
    scene = trimesh.Scene()
    for category, instances in all_instances.items():
        for i, info in enumerate(instances):
            mesh = info['original_mesh']
            transformed = mesh.copy()
            transformed.apply_transform(info['T'])
            scene.add_geometry(transformed, node_name=f"{category}_{i}")
    scene.apply_transform(Z_UP_TO_Y_UP)
    glb_path = os.path.join(output_path, filename)
    scene.export(glb_path)
    print(f"💾 GLB 已保存: {glb_path}", flush=True)
    return glb_path


def _snapshot_positions(all_instances):
    """快照所有物体的世界坐标位置 (z-up空间, mesh@T)"""
    positions = {}
    for cat, instances in all_instances.items():
        for i, info in enumerate(instances):
            mesh = info['original_mesh'].copy()
            mesh.apply_transform(info['T'])
            key = f"{cat}_{i}"
            positions[key] = {
                'centroid': mesh.centroid.copy(),
                'bounds_min': mesh.bounds[0].copy(),
                'bounds_max': mesh.bounds[1].copy(),
            }
    return positions


def _print_position_diff(before, after, stage_name):
    """打印前后位置变化对比"""
    print(f"\n📊 {stage_name} 位置变化:", flush=True)
    print(f"   {'物体':<20} {'Δx':>8} {'Δy':>8} {'Δz':>8} {'|Δ|':>8}  详情", flush=True)
    print(f"   {'─'*70}", flush=True)
    all_keys = sorted(set(list(before.keys()) + list(after.keys())))
    for key in all_keys:
        b = before.get(key)
        a = after.get(key)
        if b is None:
            print(f"   {key:<20} {'新增':>8}", flush=True)
            continue
        if a is None:
            print(f"   {key:<20} {'删除':>8}", flush=True)
            continue
        delta = a['centroid'] - b['centroid']
        dist = np.linalg.norm(delta)
        if dist < 1e-6:
            print(f"   {key:<20} {'0':>8} {'0':>8} {'0':>8} {'0':>8}  无变化", flush=True)
        else:
            print(f"   {key:<20} {delta[0]:>+8.4f} {delta[1]:>+8.4f} {delta[2]:>+8.4f} {dist:>8.4f}  "
                  f"({b['centroid'].round(4)} → {a['centroid'].round(4)})", flush=True)


def run_stage4(scene_dir, vggt_data, all_instances, args, categories_and_relations=None):
    """Stage 4: 视觉-空间对齐 (ICP + MASt3R)

    输入: VGGT depths/extrinsics/intrinsic/colors + all_instances
    输出: 对齐后的 all_instances + final_scene_stage4.glb
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 4: 迭代视觉-空间对齐", flush=True)
    print("=" * 70, flush=True)

    if not vggt_data.get('depths') or not vggt_data.get('extrinsics'):
        print("❌ VGGT 深度/外参数据不完整, 无法运行 Stage 4", flush=True)
        return all_instances
    if vggt_data.get('intrinsic') is None:
        print("❌ VGGT 内参缺失, 无法运行 Stage 4", flush=True)
        return all_instances

    # 加载 categories_and_relations (穿模修复需要)
    if categories_and_relations is None:
        import glob as _glob
        s1_candidates = _glob.glob(os.path.join(scene_dir, "*_stage1.json"))
        if not s1_candidates:
            print("❌ 未找到 stage1 JSON, 无法运行穿模修复", flush=True)
            return all_instances
        with open(s1_candidates[0], 'r') as f:
            categories_and_relations = json.load(f)

    from stage4.run_alignment import (
        reconstruct_world_points,
        create_depth_based_masks,
        compute_optimal_frame_ids,
    )
    from stage4.combined_alignment import refine_single_instance_combined

    vggt = {
        'depths': vggt_data['depths'],
        'extrinsics': vggt_data['extrinsics'],
        'intrinsic': vggt_data['intrinsic'],
        'colors': vggt_data['colors'],
    }
    world_points = reconstruct_world_points(vggt['depths'], vggt['extrinsics'], vggt['intrinsic'])
    world_points_conf = np.ones_like(vggt['depths'], dtype=np.float32)
    vggt['world_points'] = world_points
    vggt['world_points_conf'] = world_points_conf

    all_masks = create_depth_based_masks(
        all_instances, vggt['depths'], vggt['extrinsics'], vggt['intrinsic'], world_points,
    )
    all_optimal_frame_ids = compute_optimal_frame_ids(all_masks, world_points)

    total_instances = sum(len(insts) for insts in all_instances.values())
    current_instance = 0

    for category, cat_insts in all_instances.items():
        cat_masks = all_masks.get(category, [])
        cat_fids = all_optimal_frame_ids.get(category, [])

        if len(cat_insts) != len(cat_masks):
            print(f"   [Warning] {category}: 实例/mask数量不匹配，跳过", flush=True)
            current_instance += len(cat_insts)
            continue

        for iid, (inst, masks) in enumerate(zip(cat_insts, cat_masks)):
            opt_fid = cat_fids[iid] if iid < len(cat_fids) else 0
            print(f"   [{current_instance+1}/{total_instances}] {category} #{iid}", flush=True)

            inst = refine_single_instance_combined(
                instance_info=inst,
                instance_masks=masks,
                optimal_frame_id=opt_fid,
                world_points=world_points,
                world_points_conf=world_points_conf,
                depths=vggt['depths'],
                extrinsics=vggt['extrinsics'],
                intrinsic=vggt['intrinsic'],
                colors=vggt['colors'],
                num_icp_iterations=getattr(args, 'stage4_iterations', 8),
                temporal_radius=getattr(args, 'stage4_temporal_radius', 2),
                instance_index=current_instance,
                total_instances=total_instances,
                instance_name=f"{category}_{iid}",
                use_mast3r=getattr(args, 'stage4_use_mast3r', True),
                mast3r_device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            cat_insts[iid] = inst
            current_instance += 1

    # Stage 4 后穿模修复
    from tools.refine_inter_object_placement import resolve_penetrations, check_stability
    all_instances = resolve_penetrations(all_instances, verbose=True,
                                         categories_and_relations=categories_and_relations,
                                         scene_dir=scene_dir,
                                         shared_penetration=True)
    all_instances, _ = check_stability(all_instances, verbose=True,
                                       categories_and_relations=categories_and_relations)

    print(f"✅ Stage 4 完成", flush=True)
    return all_instances


def run_refine_relations(scene_dir, stage1_json, files, vlm_checkpoint):
    """Stage 5.1: 细化 "supported by other objects" 关系

    输入: stage1 JSON + optimal_frames + keyframes
    输出: refined JSON
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 5.1: 细化关系", flush=True)
    print("=" * 70, flush=True)

    if not vlm_checkpoint:
        print("❌ 需要 VLM 模型才能细化关系", flush=True)
        return None

    scene_id = files['scene_id']
    refined_json = os.path.join(scene_dir, f"{scene_id}_refined.json")

    # 优先复用主管线 (mainv3 Stage 5.1) 已推断的场景图关系 (实例级, 如 bottle_0→table_0)
    # post_pipeline 无法复现该推断 (需要内存中的 dedup masks/colors), 丢失会导致 5.2 全 floor 跳过
    scene_graph_json = os.path.join(scene_dir, "relations_scene_graph.json")
    if os.path.isfile(scene_graph_json):
        try:
            with open(scene_graph_json, 'r') as f:
                sg_data = json.load(f)
            sg_relations = sg_data.get("refined_relations", {})
            if sg_relations:
                # 类别级关系作底, 实例级场景图关系覆盖
                with open(files['stage1_json'], 'r') as f:
                    stage1_rel = json.load(f)
                merged = dict(stage1_rel)
                merged.update(sg_relations)
                with open(refined_json, 'w') as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                print(f"   ✅ 复用场景图关系: {scene_graph_json} ({len(sg_relations)} 条实例级)", flush=True)
                print(f"   关系: {json.dumps(merged, ensure_ascii=False)}", flush=True)
                final_rel = os.path.join(scene_dir, "final_relations.json")
                with open(final_rel, 'w') as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                print(f"✅ Stage 5.1 完成 → {refined_json}", flush=True)
                return merged
        except Exception as e:
            print(f"   ⚠️ 场景图关系加载失败, 回退到 refine 流程: {e}", flush=True)

    from tools.refine_other_objects_relations import refine_other_objects_relations
    refined_relations = refine_other_objects_relations(
        stage1_json_path=stage1_json,
        output_json_path=refined_json,
        scene_dir=scene_dir,
        vlm_checkpoint=vlm_checkpoint,
        optimal_frames_dir=files.get('optimal_frames_dir'),
        keyframes_dir=files.get('keyframes_dir'),
    )

    final_rel = os.path.join(scene_dir, "final_relations.json")
    with open(final_rel, 'w') as f:
        json.dump(refined_relations, f, indent=2, ensure_ascii=False)

    print(f"✅ Stage 5.1 完成 → {refined_json}", flush=True)
    print(f"   关系: {json.dumps(refined_relations, ensure_ascii=False)}", flush=True)
    return refined_relations


def run_sp_refinement(scene_dir, all_instances, refined_relations, vlm_checkpoint,
                      categories_and_relations=None, walls_info=None):
    """Stage 5.2: SP 空间位置精修

    输入: all_instances + refined_relations
    输出: 精修后的 all_instances + final_scene_stage5.glb
    """
    print("\n" + "=" * 70, flush=True)
    print("� Stage 5.2: SP 空间位置精修", flush=True)
    print("=" * 70, flush=True)

    has_inter = any(
        rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel
        for rel in refined_relations.values()
    )
    if not has_inter:
        print("⚠️ 无物体间支撑关系, 跳过 SP 精修", flush=True)
        return all_instances

    from tools.refine_inter_object_placement import refine_inter_object_relations
    all_instances = refine_inter_object_relations(
        all_instances, refined_relations,
        walls_info=walls_info, verbose=True,
        vlm_checkpoint=vlm_checkpoint,
        scene_dir=scene_dir,
        categories_and_relations=categories_and_relations,
    )

    print(f"✅ Stage 5.2 完成", flush=True)
    return all_instances


def main(args):
    total_start = time.time()

    scene_dir = os.path.abspath(args.scene_dir)
    if not os.path.isdir(scene_dir):
        print(f"❌ 场景目录不存在: {scene_dir}", flush=True)
        sys.exit(1)

    log_filename = setup_logging(scene_dir, "post_pipeline")

    do_stage4 = args.stage4
    do_stage5 = args.stage5 or args.only_refine_relations or args.only_sp_refinement

    print("=" * 70, flush=True)
    print("🚀 后处理统一管线", flush=True)
    print("=" * 70, flush=True)
    print(f"   场景目录: {scene_dir}", flush=True)
    print(f"   日志文件: {log_filename}", flush=True)
    print(f"   Stage 4: {'✅ 启用' if do_stage4 else '⏭️  跳过'}", flush=True)
    print(f"   Stage 5: {'✅ 启用' if do_stage5 else '⏭️  跳过'}", flush=True)

    vlm_checkpoint = _resolve_vlm_checkpoint(args.vlm_checkpoint)
    if vlm_checkpoint is None:
        print("⚠️ 未找到 VLM 模型, VLM-dependent 功能将使用规则回退", flush=True)
    else:
        print(f"🤖 VLM模型: {vlm_checkpoint}", flush=True)

    files = discover_scene_files(scene_dir, run_stage4=do_stage4, run_stage5=do_stage5)
    print(f"\n📂 文件发现:", flush=True)
    for k, v in files.items():
        status = "✅" if v else "❌"
        print(f"   {status} {k}: {v}", flush=True)

    if not files.get('pkl_path') and not files.get('glb_path'):
        print("❌ 未找到 all_instances.pkl 或 GLB 文件, 无法继续", flush=True)
        sys.exit(1)
    if not files.get('stage1_json'):
        print("❌ 未找到 Stage1 JSON, 无法继续", flush=True)
        sys.exit(1)

    with open(files['stage1_json'], 'r') as f:
        categories_and_relations = json.load(f)
    print(f"\n📋 Stage1 关系: {json.dumps(categories_and_relations, ensure_ascii=False)}", flush=True)

    vggt_data = {}
    if do_stage4:
        vggt_data = load_vggt_results(scene_dir, files)
        if vggt_data.get('depths') and vggt_data.get('extrinsics') and vggt_data.get('intrinsic') is not None:
            from stage4.run_alignment import reconstruct_world_points
            world_pts = reconstruct_world_points(vggt_data['depths'], vggt_data['extrinsics'], vggt_data['intrinsic'])
            vggt_data['world_points'] = world_pts
            vggt_data['world_points_conf'] = np.ones_like(vggt_data['depths'], dtype=np.float32)

    if args.only_refine_relations:
        refined = run_refine_relations(scene_dir, files['stage1_json'], files, vlm_checkpoint)
        if refined is None:
            sys.exit(1)
        print(f"\n✅ 细化完成 | 耗时: {time.time() - total_start:.1f}s", flush=True)
        return

    if args.only_sp_refinement:
        relations_json = args.relations_json
        if not relations_json:
            relations_json = os.path.join(scene_dir, f"{files['scene_id']}_refined.json")
            if not os.path.isfile(relations_json):
                relations_json = os.path.join(scene_dir, "final_relations.json")

        if not os.path.isfile(relations_json):
            print(f"❌ 关系 JSON 不存在: {relations_json}", flush=True)
            print("   请先用 --only_refine_relations 生成, 或手动指定 --relations_json", flush=True)
            sys.exit(1)

        with open(relations_json, 'r') as f:
            refined_relations = json.load(f)

        prefer_s4 = os.path.isfile(os.path.join(scene_dir, "all_instances_stage4.pkl"))
        all_instances, source = load_instances(scene_dir, prefer_stage4=prefer_s4)
        if all_instances is None:
            print("❌ 未找到 all_instances.pkl 或 GLB, 无法继续", flush=True)
            sys.exit(1)
        print(f"\n📦 加载实例: {source}", flush=True)
        for cat, insts in all_instances.items():
            print(f"   {cat}: {len(insts)} 个实例", flush=True)

        pos_before = _snapshot_positions(all_instances)
        all_instances = run_sp_refinement(scene_dir, all_instances, refined_relations, vlm_checkpoint,
                                          categories_and_relations=categories_and_relations)
        pos_after = _snapshot_positions(all_instances)
        _print_position_diff(pos_before, pos_after, "SP 精修")

        stage4_exists = os.path.isfile(os.path.join(scene_dir, "final_scene_stage4.glb"))
        sp_output_name = "final_scene_stage4_5.glb" if stage4_exists else "final_scene_stage5.glb"
        save_glb(all_instances, scene_dir, sp_output_name)

        print(f"\n✅ SP 精修完成 | 耗时: {time.time() - total_start:.1f}s", flush=True)
        return

    all_instances, source = load_instances(scene_dir)
    if all_instances is None:
        print("❌ 未找到 all_instances.pkl 或 GLB, 无法继续", flush=True)
        sys.exit(1)
    print(f"\n📦 加载实例: {source}", flush=True)
    for cat, insts in all_instances.items():
        print(f"   {cat}: {len(insts)} 个实例", flush=True)

    if do_stage4:
        pos_before_s4 = _snapshot_positions(all_instances)
        all_instances = run_stage4(scene_dir, vggt_data, all_instances, args,
                                    categories_and_relations=categories_and_relations)
        pos_after_s4 = _snapshot_positions(all_instances)
        _print_position_diff(pos_before_s4, pos_after_s4, "Stage 4")
        import pickle
        pkl_path = os.path.join(scene_dir, "all_instances_stage4.pkl")
        with open(pkl_path, 'wb') as f:
            pickle.dump(all_instances, f)
        print(f"💾 all_instances_stage4.pkl 已保存", flush=True)
        save_glb(all_instances, scene_dir, "final_scene_stage4.glb")
    else:
        print("\n⏭️  Stage 4 已跳过", flush=True)

    if do_stage5:
        refined_relations = run_refine_relations(
            scene_dir, files['stage1_json'], files, vlm_checkpoint
        )
        if refined_relations is None:
            refined_relations = dict(categories_and_relations)

        walls_info = None
        ply_path = os.path.join(scene_dir, "point_cloud.ply")
        if os.path.exists(ply_path):
            try:
                import trimesh as _trimesh
                from src.geometry_utils import get_walls_info
                _pcd = _trimesh.load(ply_path)
                walls_info = get_walls_info(_pcd, wall_masks=None)
                print(f"   📐 walls_info 已从点云计算", flush=True)
            except Exception as e:
                print(f"   ⚠️ 无法计算 walls_info: {e}", flush=True)

        if do_stage4:
            all_instances, source = load_instances(scene_dir, prefer_stage4=True)
            print(f"\n📦 加载 Stage4 实例: {source}", flush=True)
            for cat, insts in all_instances.items():
                print(f"   {cat}: {len(insts)} 个实例", flush=True)

        pos_before_s5 = _snapshot_positions(all_instances)
        all_instances = run_sp_refinement(scene_dir, all_instances, refined_relations, vlm_checkpoint,
                                          categories_and_relations=categories_and_relations,
                                          walls_info=walls_info)
        pos_after_s5 = _snapshot_positions(all_instances)
        _print_position_diff(pos_before_s5, pos_after_s5, "Stage 5")

        if do_stage4:
            save_glb(all_instances, scene_dir, "final_scene_stage4_5.glb")
        else:
            save_glb(all_instances, scene_dir, "final_scene_stage5.glb")
    else:
        print("\n⏭️  Stage 5 已跳过", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"✅ 后处理管线完成 | 总耗时: {time.time() - total_start:.1f}s", flush=True)
    print(f"   日志文件: {log_filename}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="后处理统一管线: Stage 4 + Stage 5 分层分装",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Stage 4 + Stage 5 串联
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage4 --stage5

  # 只运行 Stage 5
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage5

  # 只细化关系
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_refine_relations

  # 手动指定关系 JSON 做 SP 精修
  python3 tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_sp_refinement \\
      --relations_json output_v2/hoi4d/hoi4d_refined.json
        """,
    )

    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景输出目录 (如 output_v2/hoi4d)")

    parser.add_argument("--stage4", action="store_true",
                        help="启用 Stage 4 视觉-空间对齐")

    parser.add_argument("--stage5", action="store_true",
                        help="启用 Stage 5 语义精修 (5.1 细化关系 + 5.2 SP精修)")

    parser.add_argument("--only_refine_relations", action="store_true",
                        help="只运行 Stage 5.1: 细化关系")

    parser.add_argument("--only_sp_refinement", action="store_true",
                        help="只运行 Stage 5.2: SP 空间位置精修")

    parser.add_argument("--relations_json", type=str, default=None,
                        help="手动指定 refined 关系 JSON (配合 --only_sp_refinement)")

    parser.add_argument("--vlm_checkpoint", type=str, default=None,
                        help="VLM 模型路径 (默认自动查找)")

    parser.add_argument("--stage4_iterations", type=int, default=8,
                        help="Stage 4 ICP 迭代次数 (默认 8)")

    parser.add_argument("--stage4_temporal_radius", type=int, default=2,
                        help="Stage 4 时序半径 (默认 2)")

    parser.add_argument("--stage4_use_mast3r", action="store_true",
                        help="Stage 4 使用 MASt3R 匹配")

    args = parser.parse_args()
    main(args)
