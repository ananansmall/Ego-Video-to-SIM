"""
ReplicateAnyScene V2 — 完整自动化流水线
========================================

只需输入视频，自动完成所有阶段，输出最终 GLB。
与 main.py 相比，V2 新增了 Stage 1 自动物体发现、Stage 5.1/5.2 物体间关系精修。

流水线:
  Stage 1: VGGT引导的智能物体发现 (generate_scene_json_stage1.py)
           输入: 视频 → 输出: 场景JSON (物体类别+关系)
           原main.py需要手动提供JSON，V2自动生成

  Stage 2: 3D重建 + SAM3空间去重 (与main.py一致，可选3种模型)
           输入: 视频 → 输出: 去重后的实例masks + 3D重建中间结果
           模型选择: --vggt_model vggt(默认) / vggt_omega / vggt4d

  Stage 3: 最优视角资产生成 + 多票验证 (与main.py一致)
           输入: masks+点云 → 输出: 3D mesh实例

  基础精修: floor/wall/embedded 精修 (始终执行，与main.py一致)
           输出: final_scene_base.glb

  Stage 4: 迭代视觉-空间对齐 (stage4/, 默认关闭，--enable_stage4 启用)
           输入: GLB+3D数据 → 输出: 对齐后的GLB
           原main.py未实现此阶段

  Stage 5: 高级语义精修 (默认关闭，--enable_stage5 启用)
    5.1: 细化 "supported by other objects" 关系 (VLM判断具体支撑物)
    5.2: 物体间支撑关系空间位置精修 (纯几何计算)

  Stage 6: 材质感知的物理参数标注 (默认关闭，--enable_stage6 启用)
           VLM 推断物体材质 → 查映射表 → 输出 SAPIEN 物理参数
           输出: materials.json + materials_debug.json

GLB 文件说明 (按启用的 Stage 决定文件名):
  final_scene_basic.glb     = Stage3 基础精修产物 (固定起点, 供 Stage4 / run_post_pipeline 使用)
                              注意: 这不是最终产物! 最终产物见 final_scene_stage5.glb
  final_scene_stage4.glb   = Stage4 产物 (视觉-空间对齐后)
  final_scene_stage5.glb   = 仅 Stage5 产物 (语义精修后) ← 通常这就是最终产物
  final_scene_stage4_5.glb = Stage4+5 产物 (全部后处理后)
  all_instances.pkl        = 基础精修后的实例数据 (供 run_post_pipeline.py 独立调用stage4/5)

使用方式:
  ────────────────────────────────────────────────────────────
  参数总览:
  ────────────────────────────────────────────────────────────
  --input_video       输入视频路径 (与--input_images二选一)
  --input_images      输入图片目录路径 (与--input_video二选一)
  --output_path       输出目录 (默认: ./output_v2/{video_stem}_{模型名})
  --category_path     手动指定场景JSON (跳过Stage1自动发现)
  --vggt_model        3D重建模型: vggt(默认) | vggt_omega | vggt4d
  --max_frames        VGGT最大帧数 (默认160)
  --max_frames_stage1 Stage1采样关键帧数 (默认10)
  --vlm_checkpoint    VLM模型路径 (默认自动查找)
  --enable_stage4     启用Stage 4视觉-空间对齐
  --enable_stage5     启用Stage 5高级语义精修 (5.1+5.2)
  --stage5_method     Stage5.1关系推断方式: scene_graph(默认) | per_object
  --enable_stage6     启用Stage 6材质感知物理标注
  --stage4_iterations Stage4 ICP迭代次数 (默认8)
  --stage4_temporal_radius Stage4时序邻域半径 (默认5)
  --stage4_use_mast3r Stage4使用MASt3R匹配
  ────────────────────────────────────────────────────────────

  # 最简调用 (默认VGGT模型)
  python mainv2.py --input_video ./assets/example/hallway.mp4

  # 使用 VGGT-Ω 模型python mainv2.py --input_video ./232.mp4 --vggt_model vggt_omega
  

  # 使用 VGGT4D 模型 (带动态mask，适合动态场景)
  python mainv2.py --input_video ./232.mp4 --vggt_model vggt4d

  # VGGT4D + 全部高级阶段
  python mainv2.py --input_video ./232.mp4 --vggt_model vggt4d --enable_stage4 --enable_stage5

  # VGGT4D + Stage5 使用逐物体推断 (旧方式)
  python mainv2.py --input_video ./232.mp4 --vggt_model vggt4d --enable_stage5 --stage5_method per_object

  # 手动指定场景JSON (跳过Stage1自动发现)
  python mainv2.py --input_video ./hallway.mp4 --category_path ./hallway.json

  # 指定输出目录
  python mainv2.py --input_video ./hallway.mp4 --output_path ./my_output

  # 启用 Stage 4 (视觉-空间对齐)
  python mainv2.py --input_video ./hallway.mp4 --enable_stage4

  # 启用 Stage 5 (高级语义精修)
  python mainv2.py --input_video ./hallway.mp4 --enable_stage5

  # 输入图片目录 (Replica/ScanNet等)
  python mainv2.py --input_images /path/to/images/ --output_path ./output_v2/scene_001

  ────────────────────────────────────────────────────────────
  后处理管线 (tools/run_post_pipeline.py):
  ────────────────────────────────────────────────────────────
  在 mainv2 完成 Stage 1~3 + 基础精修后，可单独运行后处理。
  参数:
    --scene_dir              场景输出目录 (必需)
    --stage4                 启用Stage 4
    --stage5                 启用Stage 5 (5.1+5.2)
    --only_refine_relations  只运行5.1 (细化关系)
    --only_sp_refinement     只运行5.2 (SP精修)
    --relations_json         手动指定关系JSON (配合--only_sp_refinement)
    --vlm_checkpoint         VLM模型路径
    --stage4_iterations      Stage4 ICP迭代次数 (默认8)
    --stage4_temporal_radius Stage4时序半径 (默认2)
    --stage4_use_mast3r      Stage4使用MASt3R匹配

  # 一键全功能: Stage4 + Stage5
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage4 --stage5

  # 只运行 Stage 5
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage5

  # 只运行 Stage 4
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage4

  # 只细化关系 (Stage 5.1)
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_refine_relations

  # 只做 SP 精修 (Stage 5.2), 手动指定关系JSON
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_sp_refinement \
      --relations_json output_v2/hoi4d/hoi4d_refined.json

  ────────────────────────────────────────────────────────────
  mainv2 与 run_post_pipeline 的关系:
  ────────────────────────────────────────────────────────────
  mainv2:          Stage1 → Stage2 → Stage3 → 基础精修 → [Stage4] → [Stage5] → final_scene_stage5.glb
                                                                             (或 stage4_5)
  run_post_pipeline:                                      [Stage4] → [Stage5] → final_scene_stageX.glb

  mainv2 是完整流水线 (从视频到GLB)
  run_post_pipeline 只做后处理 (从已有输出目录开始)
  两者的 Stage4/5 逻辑完全一致，只是入口不同

输出目录结构:
  output_v2/{scene}_{模型名}/
  ├── scene_{scene}_stage1.json          # Stage1: 自动发现的场景JSON
  ├── scene_{scene}_stage1_refined.json  # Stage5.1: 细化后的关系JSON
  ├── final_relations.json               # 最终关系JSON
  ├── final_scene_base.glb               # 基础精修后固定起点 (不再更改)
  ├── final_scene_basic.glb              # 基础精修产物 (固定起点, 非最终产物!)
  ├── final_scene_stage5.glb             # 最终场景GLB (语义精修后)
  ├── all_instances.pkl                  # 实例数据 (供后处理管线使用)
  ├── point_cloud.ply                    # 3D重建点云
  ├── intrinsic.txt                      # 相机内参
  ├── color/                             # RGB帧
  ├── depth/                             # 深度图 (mm uint16)
  ├── extrinsics/                        # 相机外参
  ├── optimal_frames/                    # Stage3最优视角帧
  ├── keyframes/                         # Stage1关键帧+元数据
  └── instance_masks.mp4                 # 分割mask可视化
"""

import os
os.environ["LIDRA_SKIP_INIT"] = "true"
import argparse
import json
import logging
import subprocess
import sys
import time
import torch
import cv2
import numpy as np
import trimesh
from datetime import datetime

from src.models import load_vggt_model, load_vggt_omega_model, load_vggt4d_model, load_sam3_image_model, load_sam3_video_model, unload_model
from src.utils import load_video_frames, vis_instance_masks
from src.geometry_utils import (
    align_to_room_coordinate_system,
    align_via_objects,
    align_via_vlm_floor_points,
    align_via_large_plane,
    align_via_geocalib,
    align_vggt_predictions,
    get_optimal_view_frame_id,
    get_walls_info,
)
from src.vggt_predict import vggt_predict
from src.vggt_omega_predict import vggt_omega_predict
from src.vggt4d_predict import vggt4d_predict
from src.object_segmentation import segment_wall_and_floor, segment_and_track
from src.sg_deduplication import self_category_deduplicate, cross_category_deduplicate, deduplicate_3d_assets
from src.instance_generation import generate_3d_asset_in_subprocess
from src.sp_refinement import (
    refine_supported_by_floor_object,
    refine_attached_to_wall_object,
    refine_embedded_in_wall_object,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VLM_CHECKPOINT = "/mnt/data/lza/models/Qwen3.5-9B"


def _resolve_vlm_checkpoint(checkpoint_arg):
    """自动查找可用的VLM模型路径，优先级: 用户指定 > Qwen3.5-9B > Qwen2.5-VL-3B"""
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    if os.path.exists(DEFAULT_VLM_CHECKPOINT):
        return DEFAULT_VLM_CHECKPOINT
    fallback = "/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct"
    if os.path.exists(fallback):
        snapshots_dir = os.path.join(fallback, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = [d for d in os.listdir(snapshots_dir)
                         if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])
    return None


def run_stage1(input_video, output_path, vlm_checkpoint, max_frames_stage1=12, vggt_max_frames=120):
    """
    Stage 1: VGGT引导的智能物体发现 → 场景JSON

    调用 tools/generate_scene_json_stage1.py，流程:
      Step 0: VGGT 3D场景重建
      Step 1: SimRecon 3D空间覆盖采样 → 选择关键帧
      Step 2: 提取关键帧图像
      Step 3: 第一次VLM调用 → 逐帧物体检测（名称+位置）
      Step 4: 射线投射 → 将VLM像素位置映射到3D空间
      Step 5: 名称+3D位置联合去重
      Step 6: SAM分割floor和wall
      Step 7: 第二次VLM调用 → 关系判断（4种关系+物理常识后处理）
      Step 8: 输出场景JSON

    参数:
        input_video: 输入视频路径
        output_path: 输出目录
        vlm_checkpoint: VLM模型路径
        max_frames_stage1: 采样关键帧数
    返回:
        (stage1_json路径, categories_and_relations字典)
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 1: VGGT引导的智能物体发现", flush=True)
    print("=" * 70, flush=True)
    _s1_start = time.time()

    scene_id = os.path.splitext(os.path.basename(input_video))[0]
    stage1_json = os.path.join(output_path, f"scene_{scene_id}_stage1.json")
    keyframes_dir = os.path.join(output_path, "keyframes")

    # 通过subprocess调用，因为generate_scene_json_stage1内部有自己的VGGT/VLM生命周期管理
    cmd = [
        sys.executable, "-m", "tools.generate_scene_json_stage1",
        "--input_video", input_video,
        "--output_json", stage1_json,
        "--output_dir", keyframes_dir,
        "--vlm_checkpoint", vlm_checkpoint,
        "--max_frames", str(max_frames_stage1),
        "--vggt_max_frames", str(vggt_max_frames),
    ]
    # Stage1 检测走 opencode 视觉模型 (对纯色背景/小物体更稳, 见
    # tools/generate_scene_json_stage1.py 里 API 模式的实测对比)
    if getattr(args, 'use_api_vlm', False):
        cmd.append("--use_api_vlm")

    print(f"   命令: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"❌ Stage 1 失败 (返回码 {result.returncode})", flush=True)
        sys.exit(1)

    if not os.path.exists(stage1_json):
        print(f"❌ Stage 1 输出未找到: {stage1_json}", flush=True)
        sys.exit(1)

    # 读取Stage1输出的场景JSON
    with open(stage1_json, 'r', encoding='utf-8') as f:
        categories_and_relations = json.load(f)

    # 同步 keyframes: generate_scene_json_stage1 将关键帧保存到
    # assets/key_frames/{scene_id}/，需要同步到 {output_path}/keyframes/
    # 以便后续 refine_other_objects_relations 和 refine_inter_object_placement 使用
    source_keyframes = os.path.join(REPO_ROOT, "assets", "key_frames", scene_id)
    target_keyframes = os.path.join(output_path, "keyframes")
    if os.path.isdir(source_keyframes) and not os.path.isdir(target_keyframes):
        import shutil
        shutil.copytree(source_keyframes, target_keyframes)
        print(f"   📋 同步 keyframes: {source_keyframes} → {target_keyframes}", flush=True)
    elif os.path.isdir(source_keyframes) and os.path.isdir(target_keyframes):
        # 目标已存在，检查是否缺少 metadata
        src_meta = os.path.join(source_keyframes, "keyframes_metadata.json")
        dst_meta = os.path.join(target_keyframes, "keyframes_metadata.json")
        if os.path.exists(src_meta) and not os.path.exists(dst_meta):
            import shutil
            shutil.copy2(src_meta, dst_meta)
            print(f"   📋 补充 keyframes_metadata.json → {target_keyframes}", flush=True)
        # 补充缺失的帧图片
        for fname in os.listdir(source_keyframes):
            src_f = os.path.join(source_keyframes, fname)
            dst_f = os.path.join(target_keyframes, fname)
            if os.path.isfile(src_f) and not os.path.exists(dst_f):
                import shutil
                shutil.copy2(src_f, dst_f)
                print(f"   📋 补充帧: {fname} → {target_keyframes}", flush=True)
    elif not os.path.isdir(source_keyframes) and not os.path.isdir(target_keyframes):
        print(f"   ⚠️ keyframes 目录不存在: {source_keyframes} 和 {target_keyframes}", flush=True)
        print(f"   ⚠️ 后续 VLM 帧投票可能无法使用 keyframes 来源", flush=True)

    print(f"✅ Stage 1 完成: {len(categories_and_relations)} 个物体 ({time.time()-_s1_start:.1f}s)", flush=True)
    for name, rel in sorted(categories_and_relations.items()):
        print(f"   {name}: {rel}", flush=True)

    return stage1_json, categories_and_relations


def _load_vlm_model(checkpoint):
    """加载 VLM 模型 (Stage2.5/Stage3 对齐兜底分支用).

    复用 Stage1 的加载逻辑 (支持 Qwen3.5-FP8 / Qwen2.5-VL / Qwen3-VL 等),
    懒加载避免拖慢模块导入。
    """
    from tools.generate_scene_json_stage1 import load_vlm_model
    return load_vlm_model(checkpoint)


def run_stage2(input_video, output_path, categories_and_relations, max_frames=120,
               vggt_model_type="vggt", vlm_checkpoint=None):
    """
    Stage 2: VGGT 3D重建 + SAM3空间去重 (与main.py一致)

    流程:
      1. VGGT预测3D属性（点云、深度、外参）
      2. SAM3分割floor/wall → 对齐到房间坐标系
      3. 保存VGGT中间结果（color/depth/extrinsics/intrinsic/point_cloud）
      4. SAM3 video模型逐类别分割+跟踪
      5. 类内去重（self_category_deduplicate）
      6. 跨类去重（cross_category_deduplicate）
      7. JSON白名单过滤（只保留Stage1发现的类别）
      8. 可视化mask结果

    参数:
        input_video: 输入视频路径
        output_path: 输出目录
        categories_and_relations: Stage1输出的类别-关系字典
        max_frames: VGGT最大帧数
    返回:
        (vggt_prediction_results, deduplicated_all_masks, wall_masks, floor_masks)
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 2: VGGT 3D重建 + SAM3空间去重", flush=True)
    print("=" * 70, flush=True)
    _s2_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detected_categories = list(categories_and_relations.keys())
    print(f"   检测类别: {detected_categories}", flush=True)

    # 3D重建预测
    # 注意: VGGT-omega 使用 patch_size=16, image_resolution=512
    #       VGGT/VGGT4D 使用 patch_size=14, image_resolution=518
    #       每个模型必须用自己的 load 函数，不能用 ReplicateAnyScene 的 load_video_frames

    if vggt_model_type == "vggt":
        frames = load_video_frames(input_video, max_frames).to(device)
        vggt_model = load_vggt_model().to(device)
        vggt_prediction_results = vggt_predict(frames, vggt_model)
    elif vggt_model_type == "vggt_omega":
        from src.vggt_omega_predict import load_vggt_omega_frames
        frames = load_vggt_omega_frames(input_video, max_frames).to(device)
        vggt_model = load_vggt_omega_model().to(device)
        vggt_prediction_results = vggt_omega_predict(frames, vggt_model)
    elif vggt_model_type == "vggt4d":
        from src.vggt4d_predict import load_vggt4d_frames
        frames = load_vggt4d_frames(input_video, max_frames).to(device)
        vggt_model = load_vggt4d_model().to(device)
        vggt_prediction_results = vggt4d_predict(frames, vggt_model)
    else:
        raise ValueError(f"Unknown vggt_model_type: {vggt_model_type}. Choose from: vggt, vggt_omega, vggt4d")
    print(f"   加载 {len(frames)} 帧", flush=True)
    print(f"   使用 3D重建模型: {vggt_model_type} (分辨率={frames.shape[-1]}x{frames.shape[-2]})", flush=True)
    vggt_model = unload_model(vggt_model)

    # SAM3分割floor/wall → 对齐到房间坐标系
    sam3_image_model = load_sam3_image_model()
    wall_masks, floor_masks = segment_wall_and_floor(
        vggt_prediction_results['colors'], sam3_image_model
    )

    # 五阶段坐标系对齐 (Stage1 严格 → Stage2 放宽 → Stage2.5 VLM辅助 → Stage3 大平面 → Stage4 GeoCalib)
    R, t, alignment_info = align_to_room_coordinate_system(
        vggt_prediction_results['world_points'], wall_masks, floor_masks
    )
    alignment_stage = "stage1_strict"

    if np.allclose(R, np.eye(3), atol=1e-6):
        print("   Stage1 (strict) 失败, 尝试 Stage2 (relaxed)...", flush=True)
        R, t, alignment_info = align_via_objects(
            vggt_prediction_results['world_points'], wall_masks, floor_masks
        )
        alignment_stage = "stage2_relaxed"

    # Stage 2.5: VLM 辅助 - 场景类型识别 + 针对性 floor 分割
    if np.allclose(R, np.eye(3), atol=1e-6):
        vlm_model, vlm_processor = _load_vlm_model(vlm_checkpoint)
        if vlm_model is not None:
            print("   Stage2 (relaxed) 失败, 尝试 Stage 2.5 (VLM辅助)...", flush=True)
            R, t, alignment_info = align_via_vlm_floor_points(
                vggt_prediction_results['world_points'],
                vggt_prediction_results['colors'],
                sam3_image_model,
                vlm_model,
                vlm_processor,
                num_sample_frames=8,
            )
            alignment_stage = "stage2.5_vlm_assisted"
            vlm_model = unload_model(vlm_model) if hasattr(vlm_model, 'cpu') else None
        else:
            print("   Stage 2.5 跳过 (VLM 模型不可用)", flush=True)

    if np.allclose(R, np.eye(3), atol=1e-6):
        print("   Stage 2.5 (VLM辅助) 失败, 尝试 Stage3 (large_plane)...", flush=True)
        vlm_model, vlm_processor = _load_vlm_model(vlm_checkpoint)
        # 形参是 large_plane_masks (大平面mask), 原先传floor_masks 导致
        # geometry_utils.align_via_large_plane 内L469 `enhanced_masks` 非空,
        # L471 的 VLM 大平面分割分支永不执行 —— stage3 退化成 stage2 的重复品。
        # 传空列表以启用该分支 (floor 平面由该函数自行分割+拟合)。
        R, t, alignment_info = align_via_large_plane(
            vggt_prediction_results['world_points'],
            [],   # large_plane_masks: 留空以触发 VLM 大平面分割
            vggt_prediction_results['colors'],
            sam3_image_model,
            vlm_model,
            vlm_processor,
        )
        alignment_stage = "stage3_large_plane"
        if vlm_model is not None:
            vlm_model = unload_model(vlm_model) if hasattr(vlm_model, 'cpu') else None

    if np.allclose(R, np.eye(3), atol=1e-6):
        print("   Stage3 (large_plane) 失败, 尝试 Stage4 (GeoCalib)...", flush=True)
        R, t, alignment_info = align_via_geocalib(
            vggt_prediction_results['colors'],
            vggt_prediction_results['world_points'],
            extrinsics=vggt_prediction_results['extrinsics'],
        )
        # B: align_via_geocalib 失败时返回 (np.eye(3), zeros, {'reason': ...})。
        # 原先无条件写 alignment_stage="stage4_geocalib", 于是 JSON 里同时出现
        #   alignment_stage: stage4_geocalib
        #   reason: geocalib_poor_alignment R[2,2]=0.3185
        #   R = 单位阵
        # 三者自相矛盾, 且下游把单位阵当成功用 (7a_vggt_omega 实测)。
        # stage4 是最后一级, 失败后必须显式标记, 不能记账为成功。
        _geocalib_failed = np.allclose(R, np.eye(3), atol=1e-6)
        if _geocalib_failed:
            alignment_stage = "stage4_geocalib_FAILED"
            print(f"   ⚠️ Stage4 (GeoCalib) 也失败: "
                  f"{alignment_info.get('reason', '未知原因')}", flush=True)
            print("      → 坐标系未对齐 (R 为单位阵), 下游物体会保持 VGGT 原始坐标系",
                  flush=True)
        else:
            alignment_stage = "stage4_geocalib"

    # 不再打印 R[2,2] 当质量指标 —— 它只在"VGGT 的 world 系恰好是 z-up"时才有
    # 意义, 而实测 46 个场景 |R[2,2]| 中位数仅 0.42 (最小 0.047) 却对齐良好。
    # 打出来会让人误以为对齐有问题。改打重力一致性 (不依赖坐标惯例)。
    _q = (alignment_info or {}).get('quality') if isinstance(alignment_info, dict) else None
    if _q:
        print(f"   坐标系对齐完成: {alignment_stage} "
              f"(重力一致性 median_angle={_q.get('median_angle_deg', float('nan')):.1f}°, "
              f"内点率={_q.get('inlier_ratio', float('nan')):.2f})", flush=True)
    else:
        print(f"   坐标系对齐完成: {alignment_stage}", flush=True)

    # 保存坐标系对齐变换 (供 pose_changes.json 合并 + 调试追溯)
    extrinsics_before = vggt_prediction_results['extrinsics'].copy()
    vggt_prediction_results = align_vggt_predictions(vggt_prediction_results, R, t)
    extrinsics_after = vggt_prediction_results['extrinsics']

    # ── 原始视频总帧数 ──
    # Stage 3.7 (Pick/Place 检测) 用它算 frame_ratio = total_frames / RGB帧数。
    # 该模块的时长类阈值 (MIN_MOTION_DURATION 等) 定义在原始视频帧空间, 缺了
    # total_frames 会退化成 frame_ratio=1.0, pick/place 帧号随之差几倍
    # (实测各场景 ratio 在 2.29~4.93 之间)。
    n_frames_rgb = int(extrinsics_before.shape[0])
    total_video_frames = 0
    try:
        _cap = cv2.VideoCapture(input_video)
        if _cap.isOpened():
            total_video_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _cap.release()
    except Exception as _e:
        print(f"   ⚠️ 读取视频总帧数失败: {_e}", flush=True)
    if total_video_frames <= 0:
        total_video_frames = n_frames_rgb
        print(f"   ⚠️ 无法获取原始视频帧数, total_frames 退化为 RGB 帧数 "
              f"{n_frames_rgb} (Stage 3.7 的 frame_ratio 将为 1.0)", flush=True)
    else:
        print(f"   原始视频帧数: {total_video_frames} (RGB {n_frames_rgb} 帧, "
              f"frame_ratio={total_video_frames / max(n_frames_rgb, 1):.2f})", flush=True)

    coord_align_info = {
        "alignment_stage": alignment_stage,
        "R": R.tolist(),
        "t": t.tolist(),
        "method_detail": alignment_info,
        "extrinsics_before_first_frame": extrinsics_before[0].tolist(),
        "extrinsics_after_first_frame": extrinsics_after[0].tolist(),
        "n_frames": n_frames_rgb,
        "total_frames": int(total_video_frames),
        "extrinsics_convention": "w2c (world→camera): p_cam = R @ p_world + t",
        "camera_position_formula": "cam_pos = -R_w2c.T @ t_w2c (不是 t 本身)",
    }
    coord_align_path = os.path.join(output_path, "coordinate_alignment.json")
    with open(coord_align_path, 'w', encoding='utf-8') as f:
        json.dump(coord_align_info, f, indent=2, ensure_ascii=False)
    print(f"   💾 坐标系对齐信息已保存: {coord_align_path}", flush=True)

    # 保存VGGT中间结果（供Stage4和调试使用）
    os.makedirs(os.path.join(output_path, 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'extrinsics'), exist_ok=True)
    np.savetxt(os.path.join(output_path, 'intrinsic.txt'), vggt_prediction_results['intrinsic'])
    vggt_prediction_results['point_cloud_data'].export(
        os.path.join(output_path, 'point_cloud.ply')
    )
    for i, image in enumerate(vggt_prediction_results['colors']):
        cv2.imwrite(
            os.path.join(output_path, 'color', f"{i}.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
    for i, depth in enumerate(vggt_prediction_results['depths']):
        cv2.imwrite(
            os.path.join(output_path, 'depth', f"{i}.png"),
            (depth * 1000).astype(np.uint16),  # 深度单位: mm
        )
    for i, extrinsic in enumerate(vggt_prediction_results['extrinsics']):
        np.savetxt(os.path.join(output_path, 'extrinsics', f"{i}.txt"), extrinsic)

    sam3_image_model = unload_model(sam3_image_model)

    # SAM3 video模型: 逐类别分割+跟踪
    sam3_video_model = load_sam3_video_model()
    video_path = os.path.join(output_path, 'color')
    response = sam3_video_model.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    # 逐类别分割跟踪 + 类内去重
    # box 兜底: 文本提示切不出 mask 的类别 (全库 30 类) 改用 6.8 出的框重新提示。
    # 只对"文本产出 0"的类别生效, 能切出的完全不受影响; provider 任何失败都返回
    # 空并静默退化。用 --no_vlm_box_fallback 可整体关闭。
    _box_provider = None
    if _VLM_BOX_FALLBACK:
        try:
            from src.vlm_box_provider import VLMBoxProvider
            _kf0 = os.path.join(output_path, 'color', '00000.png')
            if not os.path.isfile(_kf0):
                _cands = sorted(os.listdir(os.path.join(output_path, 'color')))
                _kf0 = os.path.join(output_path, 'color', _cands[0]) if _cands else None
            if _kf0:
                _box_provider = VLMBoxProvider(_kf0, scene_dir=output_path,
                                               verbose=True)
                print(f"   [Stage 2] box 兜底已启用: {os.path.basename(_kf0)}", flush=True)
        except Exception as _e:
            print(f"   [Stage 2] box 兜底初始化失败 (不影响主流程): {_e}", flush=True)
            _box_provider = None

    # 候选播种帧: 从 Stage1 的 keyframes_metadata.json 读"每帧可见哪些类别",
    # 作为 SAM3 多播种重试的候选。
    # 为什么需要: SAM3 视频跟踪从播种帧开始传播, 小物体在早期会被跟丢 ——
    # 实测 weixin 的 button 只在帧 64~79 产出 mask (物体从第 0 帧就清晰可见),
    # 导致首帧取到 64, 拿到的是手持高度而非桌上静止位置。
    # 这里把"VLM 在该帧看到过这个类别"的帧作为候选播种帧。
    _frame_vis = {}
    _meta_p = os.path.join(output_path, 'keyframes', 'keyframes_metadata.json')
    if os.path.isfile(_meta_p):
        try:
            with open(_meta_p, encoding='utf-8') as _f:
                _frame_vis = json.load(_f).get('frame_visibility', {}) or {}
        except Exception as _e:
            print(f"   ⚠️ 读取 keyframes_metadata.json 失败: {_e}", flush=True)
    _seed_by_cat = {}
    for _fi, _names in _frame_vis.items():
        try:
            _fi_i = int(_fi)
        except Exception:
            continue
        for _nm in _names:
            _seed_by_cat.setdefault(_nm, []).append(_fi_i)

    # ── 播种帧索引空间转换 (原视频帧号 → VGGT 槽位) ──
    # keyframes_metadata.json 的 frame_visibility 键是"原视频帧号"
    # (Stage1 关键帧按原视频帧号落盘), 而 color/ 与 VGGT 数组按采样槽位
    # (0..n_frames_rgb-1) 编号, load_video_frames 用 linspace(0, total-1, n)
    # 抽帧。直接当槽位用会: ≥n_rgb 的帧号越界丢弃, <n_rgb 的错位索引
    # (bug#1: 300帧抽80 时关键帧全落在前 26%, VLM 从没见过椅子)。
    if _seed_by_cat and total_video_frames > n_frames_rgb > 1:
        def _vid2slot(_vf):
            return int(min(n_frames_rgb - 1,
                           max(0, round(_vf * (n_frames_rgb - 1)
                                        / max(total_video_frames - 1, 1)))))
        _mapped_by_cat = {}
        for _nm, _fs in _seed_by_cat.items():
            _slots = sorted({_vid2slot(_f) for _f in _fs})
            _mapped_by_cat[_nm] = _slots
            print(f"   🔁 播种帧索引转换 ({_nm}): 原视频帧号{sorted(_fs)[:8]}... "
                  f"→ VGGT槽位{_slots[:8]}... (共{len(_slots)}个)", flush=True)
        _seed_by_cat = _mapped_by_cat

    # 没有元数据 (例如用 --scene_json 跳过 Stage1) 时, 退化成按视频长度
    # 均匀采样候选播种帧。否则 _seed_frames 为空 → 多播种重试不执行 →
    # box 兜底只能播第 0 帧, 而实测小物体在第 0 帧常常切不出
    # (weixin 的帧 0/4/12 全失败, 帧 31/35 才成功)。
    if not _seed_by_cat:
        try:
            _n = len(os.listdir(os.path.join(output_path, 'color')))
        except Exception:
            _n = 0
        if _n > 1:
            _step = max(1, _n // 16)
            _uni = sorted(set(list(range(0, _n, _step)) + [_n - 1]))
            for _cat in detected_categories:
                _seed_by_cat[_cat] = _uni
            print(f"   ℹ️ 无关键帧元数据, 候选播种帧退化为均匀采样: {_uni}",
                  flush=True)

    all_masks = {}
    for category in detected_categories:
        print(f"   分割跟踪: {category}", flush=True)
        segment_and_track._seed_frames = sorted(_seed_by_cat.get(category, []))
        # box 兜底造矩形 mask 时需要 VGGT 帧尺寸 (world_points 的 H,W)。
        # 传错尺寸会被下游按形状不匹配丢掉 (实测 wx7)。
        _wp_shape = vggt_prediction_results['world_points'].shape
        _frame_hw = (int(_wp_shape[-3]), int(_wp_shape[-2])) if len(_wp_shape) >= 3 else None
        segment_and_track._frame_hw = _frame_hw
        segment_and_track._world_points = vggt_prediction_results['world_points']
        category_masks = segment_and_track(category, sam3_video_model, session_id,
                                           box_provider=_box_provider)
        segment_and_track._seed_frames = None
        deduplicated_category_masks = self_category_deduplicate(
            category_masks,
            vggt_prediction_results['world_points'],
            vggt_prediction_results['world_points_conf'],
            category_name=category,
            overlap_thre=0.3,
        )
        all_masks[category] = deduplicated_category_masks

    # 跨类去重 — 先加载 CLIP 用于 appearance feature 比较
    print(f"   [Stage 2] 加载 CLIP 模型用于跨类合并...", flush=True)
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("/mnt/data/lza/models/clip-vit-base-patch32").cuda()
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained("/mnt/data/lza/models/clip-vit-base-patch32")
        print(f"   [Stage 2] CLIP 加载成功", flush=True)
    except Exception as e:
        print(f"   [Stage 2] CLIP 加载失败: {e}，跳过跨类合并", flush=True)
        clip_model = None
        clip_processor = None

    # 构建 images_dict: frame_id → RGB image
    colors = vggt_prediction_results['colors']
    images_dict = {}
    for fid, img in enumerate(colors):
        images_dict[fid] = img.copy()  # RGB, (H, W, 3)

    # === 保存去重前的 all_masks 用于 CLIP 测试 ===
    _pre_dedup_path = os.path.join(output_path, '_pre_dedup_masks.pkl')
    import pickle as _pk
    _pk.dump({
        'all_masks': all_masks,
        'images_dict': images_dict,
        'world_points': vggt_prediction_results['world_points'],
    }, open(_pre_dedup_path, 'wb'), protocol=5)
    print(f"   [SAVE] 去重前数据已保存到 {_pre_dedup_path}", flush=True)

    # 跨类去重
    json_categories_set = set(categories_and_relations.keys())
    deduplicated_all_masks = cross_category_deduplicate(
        all_masks,
        vggt_prediction_results['world_points'],
        vggt_prediction_results['world_points_conf'],
        images_dict=images_dict,
        clip_model=clip_model,
        clip_processor=clip_processor,
    )

    # JSON白名单过滤: 只保留Stage1发现的类别
    filtered_masks = {}
    for category, category_masks in deduplicated_all_masks.items():
        if category in json_categories_set:
            filtered_masks[category] = category_masks
        else:
            print(f"   🚫 白名单过滤: '{category}' 不在JSON中", flush=True)
    deduplicated_all_masks = filtered_masks

    # 可视化分割mask结果
    vis_instance_masks(
        vggt_prediction_results['colors'], deduplicated_all_masks,
        os.path.join(output_path, 'instance_masks.mp4'),
    )
    sam3_video_model = unload_model(sam3_video_model)

    print(f"✅ Stage 2 完成 ({time.time()-_s2_start:.1f}s)", flush=True)
    return vggt_prediction_results, deduplicated_all_masks, wall_masks, floor_masks


def run_stage3(output_path, vggt_prediction_results, deduplicated_all_masks):
    """
    Stage 3: 最优视角资产生成 + 多票验证 (与main.py一致)

    流程:
      1. 计算每个实例的最优视角帧ID（最大3D表面积）
      2. 保存最优视角帧图像（供Stage5.1 VLM使用）
      3. 在SAM3D子进程中生成3D资产（避免CUDA冲突）
      4. 多票验证: 用多帧验证生成的3D资产质量

    参数:
        output_path: 输出目录
        vggt_prediction_results: VGGT预测结果
        deduplicated_all_masks: 去重后的masks
    返回:
        (all_instances, all_optimal_frame_ids)
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 3: 最优视角资产生成", flush=True)
    print("=" * 70, flush=True)
    _s3_start = time.time()

    # 计算每个实例的最优视角帧ID
    all_optimal_frame_ids = {}
    dynamic_count = 0
    static_count = 0
    for category, category_masks in deduplicated_all_masks.items():
        all_optimal_frame_ids[category] = []
        for inst_idx, instance_masks in enumerate(category_masks):
            optimal_frame_id, is_dynamic, motion_info = get_optimal_view_frame_id(
                vggt_prediction_results['world_points'], instance_masks,
                world_points_conf=vggt_prediction_results.get('world_points_conf')
            )
            all_optimal_frame_ids[category].append(optimal_frame_id)
            tag = "DYNAMIC" if is_dynamic else "STATIC"
            print(f"   {category}_{inst_idx}: [SELECT] median_disp={motion_info['median_disp']:.4f}m, "
                  f"max_disp={motion_info['max_disp']:.4f}m, "
                  f"frames={motion_info['num_valid_frames']} → frame {optimal_frame_id}", flush=True)

    print(f"\n   📊 注意: 动态判定由第二轮滑动窗口逻辑统一处理", flush=True)

    # 保存最优视角帧图像（文件名格式: {category}_inst{idx}_frame{fid}.jpg）
    optimal_frames_dir = os.path.join(output_path, 'optimal_frames')
    os.makedirs(optimal_frames_dir, exist_ok=True)
    for category, frame_ids in all_optimal_frame_ids.items():
        for inst_idx, frame_id in enumerate(frame_ids):
            image_rgb = vggt_prediction_results['colors'][frame_id]
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            save_name = f"{category}_inst{inst_idx}_frame{frame_id}.jpg"
            cv2.imwrite(os.path.join(optimal_frames_dir, save_name), image_bgr)

    instance_visibility = {}
    for category, category_masks in deduplicated_all_masks.items():
        instance_visibility[category] = {}
        for inst_idx, instance_masks in enumerate(category_masks):
            frame_ids = sorted([im["frame_id"] for im in instance_masks])
            instance_visibility[category][str(inst_idx)] = frame_ids
    visibility_path = os.path.join(optimal_frames_dir, "instance_visibility.json")
    with open(visibility_path, 'w', encoding='utf-8') as f:
        json.dump(instance_visibility, f, indent=2, ensure_ascii=False)
    print(f"   💾 Saved instance_visibility.json ({sum(len(v) for v in instance_visibility.values())} instances)", flush=True)

    # ── Stage 3.7: Pick/Place 帧检测 (大小物体分治) ──
    # 为什么必须接这一步: 本函数后段原先用自带的三条件判动态
    #   is_dynamic = (global_disp>0.05) and (moving_ratio>0.3) and (light_moving_ratio>0.8)
    # 实测该判据在 5 个有标注场景上召回 0% (5 个全漏检)。原因是它依赖"帧间大位移
    # 占比": 真实抓取-放置是连续平滑动作, 帧间位移每帧只有 1~2cm, 永远达不到
    # moving_ratio>30% (2chair-bottle 的 global_disp 已达 1.11m, 却因 moving_ratio=2%
    # 被判静态)。pick_place_detector 的分治判据用累计路径/净位移 (小物体) 和点云
    # 重叠率 (大物体), 不依赖帧间大位移, 同组场景召回 5/5、零误检零漏检。
    #
    # 关键: 这里传 deduplicated_all_masks (去重后), 而不是让该函数去读
    # _pre_dedup_masks.pkl (去重前)。两者实例编号空间不同, 实测 7_vggt_omega 里
    # 同一物理物体在去重前叫 toy_inst1、去重后叫 toy_inst0, 直接混用会让下游
    # 拿错物体。
    _t37 = time.time()
    pick_place_result = {}
    # 动态检测自身的状态: 用于区分「场景里确实没有动态物体」和「动态检测没跑起来」。
    # 后者必须显式记录 —— 否则下游看到一个空的动态集合, 会误以为是前者。
    _pp_status = {'available': False, 'reason': 'not_run'}
    try:
        from src.pick_place_detector import run_stage3_7_pick_place
        pick_place_result = run_stage3_7_pick_place(
            output_path,
            all_masks=deduplicated_all_masks,
            world_points=vggt_prediction_results['world_points'],
        )
        _pp_interactions = list(pick_place_result.get('interactions', []))
        _pp_dynamic = {i['instance_name'] for i in _pp_interactions}
        _pp_status = {
            'available': True,
            'n_interactions': len(_pp_interactions),
            'frame_ratio': pick_place_result.get('frame_ratio'),
            # fallback_1.0 表示没拿到 total_frames, 小物体判定会偏严 (可能漏检)
            'ratio_source': pick_place_result.get('ratio_source'),
        }
        print(f"   🎯 Stage 3.7 Pick/Place: 检测到 {len(_pp_interactions)} 个交互 "
              f"(frame_ratio={pick_place_result.get('frame_ratio')}), "
              f"涉及实例 {sorted(_pp_dynamic)}", flush=True)
        for _i in _pp_interactions:
            print(f"      {_i['instance_name']}: pick={_i['pick_frame']} "
                  f"place={_i['place_frame']} conf={_i['confidence']:.2f}", flush=True)
    except Exception as _e:
        _pp_interactions, _pp_dynamic = [], set()
        _pp_status = {'available': False, 'reason': str(_e)}
        print(f"   ⚠️ Stage 3.7 (Pick/Place) 失败: {_e}", flush=True)
        print("      → 动态判定回退到兜底三条件", flush=True)
        print("        但该兜底判据在 5 个有标注场景上召回 0% (5 个全部漏检): 真实抓取是")
        print("        连续平滑动作, 帧间位移每帧仅 1~2cm, 达不到其 moving_ratio>30% 的要求")
        print("        因此本次运行很可能不会输出任何动态物体 —— 这属于「动态检测失效」,",
              flush=True)
        print("        不是「场景里没有动态物体」。如需动态物体请先修好 Stage 3.7。", flush=True)

    # 落盘动态检测状态 (成功/失败都写), 让产物自身能说明是哪一种情况
    try:
        with open(os.path.join(output_path, "dynamic_detection_status.json"),
                  'w', encoding='utf-8') as _f:
            json.dump(_pp_status, _f, indent=2, ensure_ascii=False)
    except Exception as _e2:
        print(f"   ⚠️ 动态检测状态落盘失败: {_e2}", flush=True)
    print(f"   ⏱️ Stage 3.7 耗时: {time.time() - _t37:.1f}s", flush=True)

    # ── 动态 / 静态实例拆分 ──
    # 只有"被抓取或操作"的实例必须走 SAM3D (需要完整形状支撑后续位姿与仿真);
    # 其余静态实例用 VGGT 点云直接构网 (见 src/static_mesh.py):
    #   ① 尺度天然正确 —— 点云本身是米制世界坐标, 不需要后面的逐物体校准+guard
    #   ② 快 2~3 个数量级 (凸包 2 万点 8ms, SAM3D 单实例 30~60s)
    #   ③ 不占 GPU, 不再 OOM
    # 代价: 点云构网不保证闭合/完整 (遮挡、薄壁、反光可能破面), 所以只对静态
    # 实例启用。--all_sam3d 可关掉这条快速路径 (调试用)。
    # 分流规则:
    #   动态(被抓取)      → SAM3D (需要完整形状支撑位姿与仿真)
    #   静态且是大平面     → 点云构网 (又快又准, 尺度天然正确)
    #   静态但有结构      → SAM3D (点云构网会丢结构, 实测把带腿桌子压成平板)
    # 注意: 不按类别名(table/desk)硬分 —— 实测那样会把有桌腿的桌子也压成平板。
    from src.static_mesh import is_planar_pointcloud
    _wp0 = vggt_prediction_results['world_points']
    _dyn_idx, _pc_idx = {}, {}
    for _cat, _masks in deduplicated_all_masks.items():
        keep_d, keep_p = [], []
        for _i, _imlist in enumerate(_masks):
            _nm = f"{_cat}_inst{_i}"
            if any(_it.get('instance_name') == _nm for _it in (_pp_interactions or [])):
                keep_d.append(_i)
                continue
            # 静态: 汇总该实例所有可见帧的点云, 判是否为大平面
            _pts = []
            for _im in _imlist:
                if _im['frame_id'] < len(_wp0):
                    _p = _wp0[_im['frame_id']][_im['mask']]
                    if _p is not None and len(_p):
                        _pts.append(_p)
            _pts = np.concatenate(_pts, axis=0) if _pts else np.zeros((0, 3))
            if is_planar_pointcloud(_pts):
                keep_p.append(_i)
            else:
                keep_d.append(_i)
        _dyn_idx[_cat] = keep_d
        _pc_idx[_cat] = keep_p
    # ── 平面分流可见化 ──
    # 走点云构网的实例如果是桌/柜类, 几乎必然是"桌面/侧板退化重建":
    # 桌腿在分割或动态去除中丢失, 点云只剩顶面薄壳 (实测 wxD table_0
    # 1.13x0.90x0.05 无腿薄板)。不改变分流决策 (几何判据仍是权威),
    # 只把后果显式落盘, 供下游与人工检查。
    _TABLE_LIKE = {"table", "desk", "nightstand", "bed", "shelf", "cabinet"}
    _planar_flags = []
    for _cat, _ids in _pc_idx.items():
        for _i in _ids:
            _table_like = _cat.lower() in _TABLE_LIKE
            _planar_flags.append({
                "instance": f"{_cat}_inst{_i}", "category": _cat,
                "path": "pointcloud_planar",
                "table_like": _table_like,
                "note": ("桌/柜类走点云构网: 支撑结构(腿)大概率未重建, 产物为薄板"
                         if _table_like else ""),
            })
            if _table_like:
                print(f"   ⚠️ {_cat}_inst{_i} 是桌/柜类但点云判为大平面 → 点云构网, "
                      f"支撑结构(腿)大概率丢失, 产物将是薄板", flush=True)
    try:
        with open(os.path.join(output_path, "planar_flags.json"), "w", encoding="utf-8") as _f:
            json.dump(_planar_flags, _f, ensure_ascii=False, indent=1)
    except Exception:
        pass
    _n_dyn = sum(len(v) for v in _dyn_idx.values())
    _n_pc = sum(len(v) for v in _pc_idx.values())
    _n_tot = sum(len(v) for v in deduplicated_all_masks.values())
    if getattr(args, 'all_sam3d', False) or _n_pc == 0:
        all_instances = generate_3d_asset_in_subprocess(
            deduplicated_all_masks, all_optimal_frame_ids,
            vggt_prediction_results['colors'],
            vggt_prediction_results['world_points'],
            vggt_prediction_results['extrinsics'])
        print(f"   🧊 3D 资产: SAM3D 生成全部 {_n_tot} 个实例"
              f"{' (无大平面可走点云)' if not getattr(args, 'all_sam3d', False) else ''}",
              flush=True)
    else:
        print(f"   ⚡ 分流: {_n_dyn}/{_n_tot} 走 SAM3D (动态+有结构), "
              f"{_n_pc}/{_n_tot} 是大平面改用点云构网", flush=True)
        all_instances = generate_3d_asset_in_subprocess(
            deduplicated_all_masks, all_optimal_frame_ids,
            vggt_prediction_results['colors'],
            vggt_prediction_results['world_points'],
            vggt_prediction_results['extrinsics'],
            only_instances=_dyn_idx)

        # 静态实例: 点云构网补回。
        # 必须按**原始索引**归位: SAM3D 只返回动态实例 (过滤后的顺序), 若把静态
        # 实例直接 append 到末尾, all_instances[cat] 就变成"动态在前、静态在后",
        # 与 deduplicated_all_masks[cat] (原始顺序) 错位 —— 下游
        # zip(category_masks, all_instances[cat]) 会把 A 的 mask 配到 B 的 mesh 上。
        # 因此这里重建一份按原始索引排列的列表。
        from src.static_mesh import build_mesh_from_pointcloud, build_static_instance
        _wp = vggt_prediction_results['world_points']
        _n_static_done = _n_static_fail = 0
        for _cat, _masks in deduplicated_all_masks.items():
            _dyn_list = all_instances.get(_cat, []) or []
            _full = [None] * len(_masks)
            # 动态: SAM3D 返回值按过滤后顺序排列, 逐个放回原始下标
            for _k, _oi in enumerate(_dyn_idx.get(_cat, [])):
                if _k < len(_dyn_list):
                    _full[_oi] = _dyn_list[_k]
            # 大平面静态实例: 点云构网
            for _i, _imlist in enumerate(_masks):
                if _i not in set(_pc_idx.get(_cat, [])):
                    continue
                _pts = []
                for _im in _imlist:
                    _fid = _im['frame_id']
                    if _fid >= len(_wp):
                        continue
                    _p = _wp[_fid][_im['mask']]
                    if _p is not None and len(_p):
                        _pts.append(_p)
                _pts = np.concatenate(_pts, axis=0) if _pts else np.zeros((0, 3))
                _tm = build_mesh_from_pointcloud(_pts)
                if _tm is None:
                    _n_static_fail += 1
                    print(f"      ⚠️ {_cat}_{_i}: 点云构网失败 "
                          f"({len(_pts)} 点), 退回 SAM3D", flush=True)
                    continue
                _full[_i] = build_static_instance(_tm, _cat, _i)
                _n_static_done += 1
            all_instances[_cat] = [x for x in _full if x is not None]
        print(f"   ⚡ 静态实例点云构网: {_n_static_done} 成功, "
              f"{_n_static_fail} 失败(已剔除)", flush=True)

    # 动态物体: 位置调整到首次可见帧 (用最大帧生成mesh, 但位置摆在每个实例初始被SAM3检测到的位置)
    traj_info = {}
    world_points_conf = vggt_prediction_results.get(
        'world_points_conf', np.ones_like(vggt_prediction_results['depths'], dtype=np.float32))
    total_frames = len(vggt_prediction_results['depths'])
    for category, category_masks in deduplicated_all_masks.items():
        for inst_idx, (instance_masks, instance_info) in enumerate(zip(category_masks, all_instances[category])):
            optimal_frame_id = all_optimal_frame_ids[category][inst_idx]

            sorted_masks = sorted(instance_masks, key=lambda im: im['frame_id'])
            first_visible_frame_id = sorted_masks[0]['frame_id']

            # ── 置信度加权质心计算 ──
            centroid_data = []
            for im in sorted_masks:
                fid = im['frame_id']
                mask = im['mask']
                pointmap = vggt_prediction_results['world_points'][fid]
                valid = mask > 0
                if not np.any(valid):
                    continue
                pts = pointmap[valid]
                confs = world_points_conf[fid][valid]
                finite = np.all(np.isfinite(pts), axis=-1)
                if not np.any(finite):
                    continue
                # 自适应置信度阈值: 取 conf 的 10th percentile，避免硬编码 1.5 过滤过多点
                conf_thresh = np.percentile(confs[finite], 10)
                conf_mask = finite & (confs > conf_thresh)
                if not np.any(conf_mask):
                    conf_mask = finite  # fallback
                weights = confs[conf_mask]
                w_pts = pts[conf_mask]
                centroid = np.average(w_pts, axis=0, weights=weights)
                centroid_data.append((fid, centroid))

            if len(centroid_data) < 2:
                continue

            # ── 初始帧选择 ──
            # 语义: 动态物体"被操作前"的静止位置 (预抓取状态), 场景应呈现
            # 操作开始前的状态。
            #
            # 首次可见帧 ≠ 抓取前: 分割经常在物体被拿起、变大、贴近相机之后
            # 才成功 (实测 weixin button 首个 mask 在帧 64 —— 手持空中), 直接用
            # 首帧会把物体摆在半空。这里用三级策略:
            #   ① Stage 3.7 有该实例的 pick 事件 → 取 pick 之前 (换算到 VGGT
            #      槽位) 局部运动最小的可见帧 (抓取前最稳的一帧);
            #   ② pick 前没有任何可见帧 (mask 全部出现在拿起之后, 即一直在手
            #      里) → 取最后一次 place 事件最近的可见帧 (放置后静止位置),
            #      并显式告警 —— 初始状态在视频里根本不存在;
            #   ③ 没有 Stage 3.7 结果 → 首次可见帧 (确定性兜底, 原行为)。
            _pp_inst = [i for i in _pp_interactions
                        if i['instance_name'] == f"{category}_inst{inst_idx}"]
            _ratio = float((pick_place_result or {}).get('frame_ratio') or 1.0) or 1.0
            # 每个可见帧的局部运动量 = 与前后帧质心位移之和 (越小越"静止")
            _disp_of = {}
            for _i in range(1, len(centroid_data)):
                _d = float(np.linalg.norm(centroid_data[_i][1] - centroid_data[_i-1][1]))
                _disp_of[centroid_data[_i][0]] = _disp_of.get(centroid_data[_i][0], 0.0) + _d
                _disp_of[centroid_data[_i-1][0]] = _disp_of.get(centroid_data[_i-1][0], 0.0) + _d
            _vis_ids = [fid for fid, _ in centroid_data]

            initial_frame_id = first_visible_frame_id
            _anchor_reason = "首次可见帧 (无 Stage3.7 结果)"
            if _pp_inst:
                _first_pick_slot = min(i['pick_frame'] for i in _pp_inst) / _ratio
                _pre = [fid for fid in _vis_ids if fid < _first_pick_slot]
                if _pre:
                    initial_frame_id = min(_pre, key=lambda f: _disp_of.get(f, 0.0))
                    _anchor_reason = (f"pick 前最稳帧 (pick≈{_first_pick_slot:.0f}槽, "
                                      f"候选{len(_pre)}帧)")
                else:
                    _last_place_slot = max(i['place_frame'] for i in _pp_inst) / _ratio
                    initial_frame_id = min(_vis_ids, key=lambda f: abs(f - _last_place_slot))
                    _anchor_reason = (f"⚠️ pick≈{_first_pick_slot:.0f}槽前无可见帧 "
                                      f"(mask 自 {first_visible_frame_id} 起, 物体一直在手), "
                                      f"退回 place≈{_last_place_slot:.0f}槽后最近可见帧")
            print(f"   {category}_{inst_idx}: 初始帧 = frame {initial_frame_id} "
                  f"({_anchor_reason})", flush=True)

            # ── 位移计算 + MAD 离群点剔除 ──
            disps = np.array([np.linalg.norm(centroid_data[i][1] - centroid_data[i-1][1])
                              for i in range(1, len(centroid_data))])
            med = np.median(disps)
            mad = np.median(np.abs(disps - med))
            if mad > 1e-6:
                z_scores = 0.6745 * (disps - med) / mad
                inlier_mask = np.abs(z_scores) < 3.5
            else:
                inlier_mask = np.ones(len(disps), dtype=bool)
            clean_disps = disps[inlier_mask]

            median_disp = float(np.median(clean_disps))
            mean_disp = float(np.mean(clean_disps)) if len(clean_disps) > 0 else 0.0
            max_disp = float(np.max(clean_disps)) if len(clean_disps) > 0 else 0.0

            # ── 鲁棒 global_disp: 四分位均值 (前25% vs 后25%) ──
            n_q = max(3, len(centroid_data) // 4)
            head_mean = np.mean([c for _, c in centroid_data[:n_q]], axis=0)
            tail_mean = np.mean([c for _, c in centroid_data[-n_q:]], axis=0)
            global_disp = float(np.linalg.norm(tail_mean - head_mean))

            # ── 大位移帧占比 (阈值 0.05m) ──
            moving_disps = [d for d in clean_disps if d > 0.05]
            moving_ratio = len(moving_disps) / len(clean_disps) if len(clean_disps) > 0 else 0.0

            # ── 小位移帧占比 (阈值 0.02m) ──
            light_moving_disps = [d for d in clean_disps if d > 0.02]
            light_moving_ratio = len(light_moving_disps) / len(clean_disps) if len(clean_disps) > 0 else 0.0

            # ── 连续高位移分析: 统计位移 > p50 的连续帧数最大值 ──
            #    真实运动: 有大量连续帧的中等位移 (如拿起杯子)
            #    跟踪漂移: 高位移帧是零散的, 连续帧数少
            p50 = np.percentile(clean_disps, 50) if len(clean_disps) > 0 else 0.0
            high_mask = clean_disps > p50
            max_consecutive_high = 0
            cur_consecutive = 0
            for h in high_mask:
                if h:
                    cur_consecutive += 1
                    max_consecutive_high = max(max_consecutive_high, cur_consecutive)
                else:
                    cur_consecutive = 0

            # ── 最小可见帧数: <15帧不判动态 (统计不可靠) ──
            if len(centroid_data) < 15:
                print(f"   {category}_{inst_idx}: [SKIP] only {len(centroid_data)} frames (< 15), "
                      f"too few for reliable dynamic detection", flush=True)
                continue

            # ── 可见帧占比: 防止可见帧少的物体被 VGGT 噪声误判 ──
            visible_ratio = len(centroid_data) / max(total_frames, 1)

            # ── 动态判定: 以 Stage 3.7 的 pick/place 结果为准 ──
            # 旧三条件 (global_disp / moving_ratio / light_moving_ratio) 只在
            # Stage 3.7 不可用时作兜底 —— 它在 5 个 GT 场景上召回 0% (全漏检),
            # 原因见本函数 Stage 3.7 处的说明。
            _pp_key = f"{category}_inst{inst_idx}"
            if pick_place_result:
                is_dynamic_inst = _pp_key in _pp_dynamic
            else:
                is_dynamic_inst = ((global_disp > 0.05) and (moving_ratio > 0.3)
                                   and (light_moving_ratio > 0.8))

            if not is_dynamic_inst:
                continue

            _pp_its = [i for i in _pp_interactions if i['instance_name'] == _pp_key]
            if _pp_its:
                reasons = [f"pick={i['pick_frame']} place={i['place_frame']} "
                           f"conf={i['confidence']:.2f}" for i in _pp_its]
                print(f"   {category}_{inst_idx}: [DYNAMIC] Stage3.7 pick/place: "
                      f"global_disp={global_disp:.3f}m, "
                      f"visible_ratio={visible_ratio:.0%}, "
                      f"frames={len(centroid_data)} → {'; '.join(reasons)}", flush=True)
            else:
                reasons = []
                if global_disp > 0.05:
                    reasons.append(f"global_disp={global_disp:.3f}m")
                print(f"   {category}_{inst_idx}: [DYNAMIC] (兜底三条件) "
                      f"global_disp={global_disp:.3f}m, "
                      f"moving_ratio={moving_ratio:.0%}, "
                      f"light_moving_ratio={light_moving_ratio:.0%}, "
                      f"visible_ratio={visible_ratio:.0%}, "
                      f"frames={len(centroid_data)}/{total_frames} "
                      f"→ {', '.join(reasons)}", flush=True)

            # ── 动态物体点云轨迹输出 ──
            dyn_output_dir = os.path.join(output_path, "dynamic_pointclouds")
            os.makedirs(dyn_output_dir, exist_ok=True)

            # 提取每个可见帧的点云
            frame_ids = []
            frame_pts = []
            for im in sorted_masks:
                fid = im['frame_id']
                mask = im['mask']
                pointmap = vggt_prediction_results['world_points'][fid]
                valid = mask > 0
                if not np.any(valid):
                    continue
                pts = pointmap[valid]
                finite_mask = np.all(np.isfinite(pts), axis=1)
                pts = pts[finite_mask]
                if pts.size == 0:
                    continue
                # 每帧最多采样 5000 个点
                if len(pts) > 5000:
                    indices = np.random.choice(len(pts), 5000, replace=False)
                    pts = pts[indices]
                frame_ids.append(fid)
                frame_pts.append(pts)

            # 写入 PLY 文件, 用红→蓝渐变表示时间顺序
            #
            # B1 修复: 原实现在这个逐帧循环里用「固定文件名 + 'w' 模式」写入, 每帧
            # 覆盖上一帧 —— 实测 7_vggt_omega 的日志打印了 113 次"动态点云轨迹已
            # 保存", 磁盘上却只剩最后一帧的点。而注释写明的意图是"红→蓝渐变表示
            # 时间顺序", 即一个文件里装下所有帧的点、用颜色编码时间。
            # 现在改为循环内累积、循环外只写一次。
            #
            # B2 修复: 原代码复用外层变量名 total_frames 来存"本段帧数", 导致处理
            # 完第一个物体后, 外层 total_frames(RGB 总帧数) 被永久改写, 影响后续
            # 物体的 visible_ratio。改用局部名 n_seg_frames。
            n_seg_frames = len(frame_ids)
            _all_pts = []
            for fi, (fid, pts) in enumerate(zip(frame_ids, frame_pts)):
                t_ratio = fi / max(n_seg_frames - 1, 1)
                r = int(255 * (1 - t_ratio))
                g = int(0)
                b = int(255 * t_ratio)
                color_rgb = np.full((pts.shape[0], 3), [r, g, b], dtype=np.uint8)
                _all_pts.append(np.hstack([pts.astype(np.float64),
                                           color_rgb.astype(np.float64)]))

            if _all_pts:
                combined = np.vstack(_all_pts)
                ply_header = (
                    "ply\n"
                    "format ascii 1.0\n"
                    f"element vertex {len(combined)}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "end_header\n"
                )
                ply_filename = f"{category}_inst{inst_idx}_dynamic.ply"
                ply_path = os.path.join(dyn_output_dir, ply_filename)
                with open(ply_path, 'w') as f:
                    f.write(ply_header)
                    np.savetxt(f, combined, fmt='%.6f %.6f %.6f %d %d %d')
                print(f"   {category}_{inst_idx}: 动态点云轨迹已保存: {ply_path} "
                      f"({len(frame_ids)} 帧, {len(combined)} vertices)", flush=True)
            else:
                print(f"   {category}_{inst_idx}: ⚠️ 无有效点云, 跳过轨迹 PLY", flush=True)

            # 计算轨迹信息 (复用已有的 centroid_data)
            visible_frames = [fid for fid, _ in centroid_data]
            centroids = [c.tolist() for _, c in centroid_data]

            if len(centroids) >= 2:
                total_disp = float(np.linalg.norm(np.array(centroids[-1]) - np.array(centroids[0])))
                motion_dir = np.array(centroids[-1]) - np.array(centroids[0])
                motion_dir_norm = float(np.linalg.norm(motion_dir))
                motion_dir = (motion_dir / motion_dir_norm).tolist() if motion_dir_norm > 0 else [0.0, 0.0, 0.0]
            else:
                total_disp = 0.0
                motion_dir = [0.0, 0.0, 0.0]

            traj_info[f"{category}_inst{inst_idx}"] = {
                "is_dynamic": True,
                "category": category,
                "instance_idx": inst_idx,
                "visible_frames": visible_frames,
                "centroids": centroids,
                "total_displacement": total_disp,
                "motion_direction": motion_dir,
                # Stage 3.7 的 pick/place 事件 (原始视频帧空间)。
                # 走兜底三条件时该列表为空 —— 下游可据此区分判定来源。
                "pick_place": _pp_its,
            }

            # ── 点云清理: 动态物体只保留首帧点云, 剔除其他帧的点云 ──
            # 动态物体在运动, 其他帧的点云是"残影", 会污染下游 (SP精修/Stage4/对齐)
            # 保留 first_visible_frame_id 的点云 (物体放置位置), 其余帧该实例 mask 区域置 NaN
            # *** 已注释: 会破坏下游 world_points 数据, 影响 Pick/Place 检测和 Stage4 对齐 ***
            #
            # cleaned_count = 0
            # for im_entry in sorted_masks:
            #     fid_clean = im_entry['frame_id']
            #     if fid_clean == first_visible_frame_id:
            #         continue
            #     mask_clean = im_entry['mask']
            #     vggt_prediction_results['world_points'][fid_clean][mask_clean > 0] = np.nan
            #     cleaned_count += 1
            # if cleaned_count > 0:
            #     print(f"   {category}_{inst_idx}: [DYNAMIC] 点云清理: 保留首帧{first_visible_frame_id}, "
            #           f"剔除其他 {cleaned_count} 帧的实例点云", flush=True)

            initial_centroid = None
            optimal_centroid = None
            initial_pts = None
            optimal_pts = None
            for fid, c in centroid_data:
                if fid == initial_frame_id and initial_centroid is None:
                    initial_centroid = c
                    # 提取初始帧点云 (抓取前的静止位置)
                    for im_entry in sorted_masks:
                        if im_entry['frame_id'] == initial_frame_id:
                            mask_fv = im_entry['mask']
                            pointmap_fv = vggt_prediction_results['world_points'][initial_frame_id]
                            pts_fv = pointmap_fv[mask_fv > 0]
                            finite_fv = np.all(np.isfinite(pts_fv), axis=1)
                            initial_pts = pts_fv[finite_fv]
                            break
                if fid == optimal_frame_id:
                    optimal_centroid = c
                    # 提取最优帧点云
                    for im_entry in sorted_masks:
                        if im_entry['frame_id'] == optimal_frame_id:
                            mask_opt = im_entry['mask']
                            pointmap_opt = vggt_prediction_results['world_points'][optimal_frame_id]
                            pts_opt = pointmap_opt[mask_opt > 0]
                            finite_opt = np.all(np.isfinite(pts_opt), axis=1)
                            optimal_pts = pts_opt[finite_opt]
                            break

            # ── 位置 + 旋转调整 ──
            # 平移：mesh 从最优帧位置移到「初始帧」(抓取前静止位置) 的位置
            # 旋转：用 PCA 估计两帧的朝向差，旋转到初始帧朝向
            if initial_centroid is not None and optimal_centroid is not None and initial_frame_id != optimal_frame_id:
                offset = initial_centroid - optimal_centroid
                needs_adjust = False

                # 平移调整
                if np.linalg.norm(offset) > 0.01:
                    T = instance_info["T"].copy()
                    T[:3, 3] += offset
                    needs_adjust = True

                # 旋转调整: PCA 估计主方向
                rot_applied = False
                if (initial_pts is not None and optimal_pts is not None
                        and len(initial_pts) > 10 and len(optimal_pts) > 10):
                    try:
                        # 中心化点云
                        pts_init_c = initial_pts - initial_centroid
                        pts_opt_c = optimal_pts - optimal_centroid

                        # PCA: 求协方差矩阵的主轴
                        _, _, V_init = np.linalg.svd(pts_init_c.T @ pts_init_c, full_matrices=True)
                        _, _, V_opt = np.linalg.svd(pts_opt_c.T @ pts_opt_c, full_matrices=True)

                        # V 的行就是主轴方向 (按方差降序排列)
                        # 旋转矩阵 R = V_init @ V_opt.T
                        # SVD 特征向量符号有 2^3 重任意性: 先逐轴按点积对齐符号,
                        # 再修行列式 —— 只修 det 会漏掉单轴 180° 翻转 (实测旋转
                        # 会被转到错误朝向)。
                        for _ai in range(3):
                            if float(np.dot(V_init[_ai], V_opt[_ai])) < 0:
                                V_opt[_ai] *= -1
                        R_pca = V_init @ V_opt.T

                        # 检查是否是合法旋转矩阵（反射矩阵行列式=-1需要修正）
                        if np.linalg.det(R_pca) < 0:
                            V_opt[2] *= -1
                            R_pca = V_init @ V_opt.T

                        # 旋转角 (用于日志)
                        trace_R = np.trace(R_pca)
                        rot_angle = np.degrees(np.arccos(np.clip((trace_R - 1) / 2, -1, 1)))

                        if rot_angle > 5.0:  # 旋转角 > 5度才调整
                            # 如果还没创建 T，先创建；如果有平移，复用已创建的 T
                            if not needs_adjust:
                                T = instance_info["T"].copy()
                            T[:3, :3] = R_pca @ T[:3, :3]
                            needs_adjust = True
                            rot_applied = True

                        if needs_adjust:
                            instance_info["T"] = T
                            msg = (f"   {category}_inst{inst_idx}: [DYNAMIC] 位置+旋转调整: "
                                   f"mesh帧{optimal_frame_id} → 初始帧{initial_frame_id} "
                                   f"(抓取前), offset={np.linalg.norm(offset):.4f}m")
                            if rot_applied:
                                msg += f", rot={rot_angle:.1f}deg"
                            print(msg, flush=True)

                    except Exception as e:
                        print(f"   {category}_inst{inst_idx}: [WARN] 旋转调整失败: {e}", flush=True)

    # 写入 trajectory_info.json
    if traj_info:
        traj_file = os.path.join(output_path, "dynamic_pointclouds", "trajectory_info.json")
        with open(traj_file, 'w', encoding='utf-8') as f:
            json.dump(traj_info, f, indent=2)
        print(f"   💾 trajectory_info.json 已保存 ({len(traj_info)} 个动态物体)", flush=True)

    # 3D Mesh 去重: 在生成后二次检查，移除重复3D资产
    all_instances, deduplicated_all_masks, mesh_dedup_removed = deduplicate_3d_assets(
        all_instances, all_masks=deduplicated_all_masks
    )
    if mesh_dedup_removed:
        all_optimal_frame_ids = {}
        for category, category_masks in deduplicated_all_masks.items():
            all_optimal_frame_ids[category] = []
            for instance_masks in category_masks:
                optimal_fid, _, _ = get_optimal_view_frame_id(
                    vggt_prediction_results['world_points'], instance_masks,
                    world_points_conf=vggt_prediction_results.get('world_points_conf')
                )
                all_optimal_frame_ids[category].append(optimal_fid)

    # Stage 3.5: 多票验证生成的3D资产
    from tools.asset_verifier import verify_all_instances
    all_instances = verify_all_instances(
        all_instances,
        all_optimal_frame_ids,
        deduplicated_all_masks,
        vggt_prediction_results['world_points'],
        vggt_prediction_results['world_points_conf'],
        min_votes=2,
    )

    print(f"✅ Stage 3 完成 ({time.time()-_s3_start:.1f}s)", flush=True)
    return all_instances, all_optimal_frame_ids, deduplicated_all_masks


def run_stage4(output_path, vggt_prediction_results, all_instances, args,
               categories_and_relations=None, walls_info=None,
               deduplicated_all_masks=None):
    """
    Stage 4: 迭代视觉-空间对齐 (可选，默认关闭)

    流程 (对齐论文 Section 3.4):
      1. 使用VGGT世界点云 + 置信度
      2. 使用SAM分割mask作为real mask (论文方法)，无SAM mask时回退到深度比较
      3. 对每个实例执行render-match-optimize迭代对齐:
         - Phase A: MASt3R/深度匹配 → 3D Lifting → Umeyama相似变换(含尺度)
         - Phase B: ICP精调 → 渐进阈值+RANSAC

    参数:
        output_path: 输出目录
        vggt_prediction_results: VGGT预测结果 (含 world_points, world_points_conf)
        all_instances: 3D资产实例
        args: 命令行参数（stage4_iterations, stage4_temporal_radius等）
        deduplicated_all_masks: SAM分割mask (论文的 M_real), {category: [instance_masks]}
    返回:
        对齐后的 all_instances
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 4: 迭代视觉-空间对齐", flush=True)
    print("=" * 70, flush=True)
    _s4_start = time.time()

    from stage4.run_alignment import (
        create_depth_based_masks,
        compute_optimal_frame_ids,
    )
    from stage4.combined_alignment import refine_single_instance_combined

    # 使用VGGT原始世界点云和置信度 (论文方法，不再重新计算/禁用置信度)
    world_points = vggt_prediction_results['world_points']
    world_points_conf = vggt_prediction_results.get(
        'world_points_conf', np.ones_like(vggt_prediction_results['depths'], dtype=np.float32))
    print(f"   使用 VGGT world_points_conf (min={world_points_conf.min():.3f}, "
          f"mean={world_points_conf.mean():.3f}, max={world_points_conf.max():.3f})", flush=True)

    # 使用SAM分割mask作为real mask (论文方法 M_real,v)
    # SAM mask是固定的真实物体轮廓，不随当前位姿变化，避免循环依赖
    use_sam_masks = (deduplicated_all_masks is not None
                     and set(deduplicated_all_masks.keys()) == set(all_instances.keys())
                     and all(len(deduplicated_all_masks.get(c, [])) == len(insts)
                             for c, insts in all_instances.items()))

    if use_sam_masks:
        print(f"   使用 SAM 分割 mask (论文方法 M_real)", flush=True)
        all_masks = deduplicated_all_masks
    else:
        print(f"   [Warning] SAM mask 与 all_instances 不匹配，回退到深度比较 mask", flush=True)
        all_masks = create_depth_based_masks(
            all_instances, vggt_prediction_results['depths'],
            vggt_prediction_results['extrinsics'],
            vggt_prediction_results['intrinsic'], world_points,
        )

    all_optimal_frame_ids = compute_optimal_frame_ids(all_masks, world_points)

    total_instances = sum(len(insts) for insts in all_instances.values())
    current_instance = 0

    # 逐实例执行2D-3D对齐
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

            relationship = categories_and_relations.get(category) if categories_and_relations else None
            camera_pos = None
            if relationship and walls_info:
                rel_lower = relationship.lower().replace("_", " ")
                if "wall" in rel_lower:
                    extrinsic = vggt_prediction_results['extrinsics'][opt_fid]
                    camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]

            inst = refine_single_instance_combined(
                instance_info=inst,
                instance_masks=masks,
                optimal_frame_id=opt_fid,
                world_points=world_points,
                world_points_conf=world_points_conf,
                depths=vggt_prediction_results['depths'],
                extrinsics=vggt_prediction_results['extrinsics'],
                intrinsic=vggt_prediction_results['intrinsic'],
                colors=vggt_prediction_results['colors'],
                num_icp_iterations=args.stage4_iterations,
                temporal_radius=args.stage4_temporal_radius,
                instance_index=current_instance,
                total_instances=total_instances,
                instance_name=f"{category}_{iid}",
                use_mast3r=args.stage4_use_mast3r,
                mast3r_device='cuda',
                relationship=relationship,
                walls_info=walls_info,
                camera_pos=camera_pos,
            )
            cat_insts[iid] = inst
            current_instance += 1

    print(f"✅ Stage 4 完成 ({time.time()-_s4_start:.1f}s)", flush=True)
    return all_instances


def run_stage5(output_path, categories_and_relations, all_instances,
               vggt_prediction_results, all_optimal_frame_ids,
               wall_masks, vlm_checkpoint, stage1_json,
               deduplicated_all_masks=None, stage5_method="scene_graph"):
    """
    Stage 5: 高级语义精修 (5.1 + 5.2, 需 --enable_stage5 启用)

    注意: 5.0 基础精修 (floor/wall/embedded) 已移至 main() 中始终执行，
    与 main.py 行为一致。此函数只处理 5.1 和 5.2。

    5.1 关系推断有两种方式 (--stage5_method):
      scene_graph: SimRecon风格, 在ID标注图上一次性推断所有关系 (默认, 推荐)
      per_object:  逐物体推断, 只细化 "supported by other objects" (旧方式)

    5.2 几何精修始终使用 sp_refinement.py 的方式。
    """
    print("\n" + "=" * 70, flush=True)
    print(f"🚀 Stage 5: 高级语义精修 (5.1 + 5.2) [method={stage5_method}]", flush=True)
    print("=" * 70, flush=True)

    walls_info = get_walls_info(vggt_prediction_results['world_points'], wall_masks)
    refined_relations = dict(categories_and_relations)
    vlm_for_stage52 = None

    # ── 5.1: 关系推断 ──
    _t51 = time.time()
    if stage5_method == "scene_graph":
        print("\n   📍 5.1: 场景图关系推断 (SimRecon风格)", flush=True)
        from tools.infer_relations_scene_graph import infer_relations_scene_graph
        refined_relations, vlm_for_stage52 = infer_relations_scene_graph(
            scene_dir=output_path,
            vlm_checkpoint=vlm_checkpoint,
            categories_and_relations=categories_and_relations,
            instance_masks=deduplicated_all_masks,
            colors=vggt_prediction_results['colors'],
            output_dir=output_path,
            optimal_frames_dir=os.path.join(output_path, 'optimal_frames'),
            keyframes_dir=os.path.join(output_path, 'keyframes'),
        )
    else:
        has_other_objects = any(
            rel == "supported by other objects"
            for rel in categories_and_relations.values()
        )

        if has_other_objects:
            print("\n   📍 5.1: 细化 'supported by other objects' 关系 (逐物体推断)", flush=True)
            scene_id = os.path.splitext(os.path.basename(stage1_json))[0]
            refined_json = os.path.join(output_path, f"{scene_id}_refined.json")

            from tools.refine_other_objects_relations import refine_other_objects_relations
            refined_relations = refine_other_objects_relations(
                stage1_json_path=stage1_json,
                output_json_path=refined_json,
                scene_dir=output_path,
                vlm_checkpoint=vlm_checkpoint,
                optimal_frames_dir=os.path.join(output_path, 'optimal_frames'),
                keyframes_dir=os.path.join(output_path, 'keyframes'),
            )
        else:
            print("\n   📍 5.1: 无 'supported by other objects' 关系，跳过细化", flush=True)
    _t51_time = time.time() - _t51
    print(f"   ⏱️ 5.1 耗时: {_t51_time:.1f}s", flush=True)

    # ── 5.15: 支撑物实例纠偏 ──
    # 5.1 让 VLM 从候选实例里挑 parent, 同类多实例时会挑错 (实测 weixin_v8:
    # handle_0 几何上在 table_1 上, VLM 却选了 table_0 → 悬空 15.8cm)。
    # 这里按几何校验一遍, 不成立则换成同类里最合适的实例。
    try:
        from src.sp_refinement import fix_supporter_instance
        refined_relations, _n_fix_sup = fix_supporter_instance(
            all_instances, refined_relations, verbose=True)
        if _n_fix_sup:
            print(f"   🔧 支撑物实例纠偏: {_n_fix_sup} 处", flush=True)
    except Exception as _e:
        print(f"   ⚠️ 支撑物纠偏失败 (不影响主流程): {_e}", flush=True)

    # ── 5.2: 几何精修 (只修改 "supported by <物体>" 的物体, 已精修的floor/wall不动) ──
    _t52 = time.time()
    has_inter_object = any(
        rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel
        for rel in refined_relations.values()
    )

    if has_inter_object:
        print("\n   📍 5.2: 物体间支撑关系空间位置精修 (只修改other objects物体)", flush=True)
        from tools.refine_inter_object_placement import refine_inter_object_relations
        all_instances = refine_inter_object_relations(
            all_instances, refined_relations,
            walls_info=walls_info, verbose=True,
            vlm_checkpoint=vlm_checkpoint,
            scene_dir=output_path,
            only_refine_other_objects=True,
            preloaded_vlm=vlm_for_stage52,
        )
    else:
        # 没有物体间支撑关系时也要落盘 placement_strategies.json (空字典)。
        #
        # 为什么需要: 原实现只在 has_inter_object=True 时才写该文件, 而 5.2 被
        # 跳过时文件根本不存在 —— 下游无法区分「已评估, 确实没有物体间关系」
        # 和「5.2 从未运行」这两种情况。实测 8/23 场景缺这个文件, 且全部是
        # 100% "supported by floor" 的场景 (Stage1 prompt 兜底把所有物体标成
        # floor, 见 generate_scene_json_stage1.py 的默认值修复)。
        # 空字典是诚实答案: 这些场景确实没有任何 on_top/inside 关系。
        # 下游读到空 dict 而非 None, 能正确区分"无策略"与"读取失败"。
        print("\n   📍 5.2: 无物体间支撑关系，跳过精修", flush=True)
        try:
            import json as _json
            _sp = os.path.join(output_path, "placement_strategies.json")
            if not os.path.isfile(_sp):
                with open(_sp, "w", encoding="utf-8") as _f:
                    _json.dump({}, _f, indent=2, ensure_ascii=False)
                print("   💾 放置策略已保存 (0 条, 无物体间支撑关系): "
                      f"{_sp}", flush=True)
        except Exception as _e:
            print(f"   ⚠️ 放置策略落盘失败 (不影响主流水线): {_e}", flush=True)
    _t52_time = time.time() - _t52
    print(f"   ⏱️ 5.2 耗时: {_t52_time:.1f}s", flush=True)

    # ── 5.25: 摆放补正 (必须在 5.2 之后) ──
    # 5.2 的穿模修复只做"分开"不做"定位": 关系为 floor 的会被推高, 关系为
    # on_top 的会因零间隙接触被误判穿模而推离支撑物 (实测 104 个 on_top 关系
    # 只有 3 个贴合)。这里按5.1 的实例级关系把两类摆正。
    _t525 = time.time()
    from src.sp_refinement import (ground_floating_objects, reseat_on_top_objects,
                                   dump_placement_audit,
                                   separate_object_penetrations,
                                   recenter_edge_objects)
    # 构造 ref_z: 每个实例在"最早可见帧"的 VGGT 点云底部 z, 作为参考高度。
    # 用途: 区分两种高空悬停 ——
    #   (a) 坐标系未对齐: 物体最初就在高处 (参考值也高) → 补正应跳过
    #   (b) 被上游推高:   物体最初贴地 (参考值≈0), 但 5.2 的穿模修复把它推上去了
    # 实测 basic_pick_place_17 toy_1: 最初 z≈0 → Stage5 变 0.8806m, toy_5 由 0 变
    # 0.6379m; 二者都因超过 float_z_max(0.5) 被跳过, 最终渲染时飘在空中。
    # 用"最初的点云"而不是 pose_history 作参考: 后者在 run_stage5 里并不存在
    # (只是 _record_pose_stage 的内部参数, 且该函数未被调用)。
    _ref_z = {}
    try:
        _wp = (vggt_prediction_results or {}).get("world_points")
        if _wp is not None and deduplicated_all_masks:
            for _cat, _inst_list in deduplicated_all_masks.items():
                for _idx, _im_list in enumerate(_inst_list):
                    if not _im_list:
                        continue
                    _im0 = min(_im_list, key=lambda x: x["frame_id"])
                    _fid = _im0["frame_id"]
                    _m = _im0["mask"] > 0
                    if not _m.any():
                        continue
                    _pts = _wp[_fid][_m]
                    _fin = np.all(np.isfinite(_pts), axis=-1)
                    if not _fin.any():
                        continue
                    _ref_z[f"{_cat}_{_idx}"] = float(_pts[_fin][:, 2].min())
        if _ref_z:
            print(f"   📐 落地补正参考高度: 取 {len(_ref_z)} 个实例的最早帧点云底部 z",
                  flush=True)
    except Exception as _e:
        print(f"   ⚠️ 参考高度构造失败 (不影响主流程): {_e}", flush=True)
        _ref_z = {}
    all_instances, n_grounded = ground_floating_objects(
        all_instances, refined_relations, verbose=True, ref_z=_ref_z)
    print(f"   🔧 落地补正: {n_grounded} 个悬空实例已压到地面", flush=True)
    # 注意: 穿模分离**不在这里做**, 移到本段末尾 (见下方 5.25 收敛循环)。
    # 实测 basic_pick_place_141: 分离放前面时 10 对 → 22 对 (更糟), 因为
    # 后面的重贴/居中会把刚分开的物体又拉回支撑物上重新叠在一起。
    # 实测: 分离放最后 = 3 对, 再迭代 2 轮 = 0 对。
    # 原因: Stage1 prompt 的 "If unsure, default to: supported by floor" 让俯视
    # 桌面场景的所有物体都标成 floor, 而基础精修只管单物体贴地、5.2 的穿模修复
    # 又带 only_refine_other_objects=True 跳过 floor —— 三层都没管"物体之间"的
    # 穿模。实测 basic_pick_place_111 lego_0/lego_1 重叠 85%、46 的 box_1/box_2
    # 92% 与 card_0 埋进 box_1 91.6%、118 的 croissant 埋进 pillow 68%。
    # 只推 XY 不推 Z: z 已由落地补正固定, 推 z 会破坏贴地。
    # 5.2 落盘的每实例策略 (on_top/inside/...); 缺失时reseat 退化为只看关系,
    # 此时无法区分 inside, 会在日志打印提示。
    _strategy_path = os.path.join(output_path, "placement_strategies.json")
    _strategies = None
    if os.path.isfile(_strategy_path):
        try:
            import json as _json
            with open(_strategy_path, encoding="utf-8") as _f:
                _strategies = _json.load(_f)
        except Exception as _e:
            print(f"   ⚠️ 读取 placement_strategies.json 失败: {_e}", flush=True)
    all_instances, n_reseated = reseat_on_top_objects(
        all_instances, refined_relations, placement_strategies=_strategies, verbose=True)
    if _strategies is None:
        print("   ⚠️ 未找到 placement_strategies.json, on_top 重贴无法区分 inside, "
              "可能把 inside 的物体错贴到支撑物顶面", flush=True)
    print(f"   🔧 on_top 重贴: {n_reseated} 个实例已贴回支撑物顶面", flush=True)
    # 边沿居中: reseat 只修 z 不动水平位置, "只搭上一个角"的物体仍悬在支撑物边沿。
    # 实测 vertical_pick_place_115 bottle_0 与 box_0 水平重叠仅 41% (59% 悬在盒外)。
    # 放在 reseat 之后: z 先落位, xy 再居中, 两步互不干扰。
    all_instances, n_recentered = recenter_edge_objects(
        all_instances, refined_relations, verbose=True)
    if n_recentered:
        print(f"   🔧 边沿居中: {n_recentered} 次平移", flush=True)

    # ── 穿模收敛循环 ──
    # 为什么必须在最后、且要迭代:
    #   ① 顺序: 分离若排在重贴/居中之前, 后两步会把刚分开的物体又拉回支撑物上
    #      重新叠在一起。实测 basic_pick_place_141 分离在前 = 22 对穿模 (原始 10
    #      对, 反而更糟); 分离放最后 = 3 对。
    #   ② 迭代: 分离会把物体推离支撑物 (尤其桌上物体), 需再重贴 z 把它压回顶面,
    #      而重贴又可能引入新的重叠 —— 单轮收敛不了。实测迭代 2 轮 → 0 对。
    #   ③ 循环内不再调用 recenter: 它把物体聚拢到支撑物中心, 与分离直接冲突,
    #      会在两者之间来回震荡。
    # 循环结束时最后一次执行的必是"分离" (它返回 0 才退出), 因此出口保证无穿模。
    _MAX_CONV_ITER = 3
    _n_sep_total = 0
    for _rnd in range(_MAX_CONV_ITER):
        all_instances, _ns = separate_object_penetrations(
            all_instances, refined_relations, verbose=(_rnd == 0))
        _n_sep_total += _ns
        if _ns == 0:
            break
        # 分离可能把物体推离支撑物 → 重贴 z 压回顶面 (recenter 不在此调用)
        all_instances, _ = reseat_on_top_objects(
            all_instances, refined_relations,
            placement_strategies=_strategies, verbose=False)
    if _n_sep_total:
        print(f"   🔧 穿模分离: {_n_sep_total} 次推开 (收敛 {_rnd + 1} 轮)", flush=True)

    # ── 分离后收尾落地 ──
    # 穿模分离是"把两个物体沿连线推开", 推的方向不保证是水平的 —— 当上方物体
    # 压着下方物体时, 推开会把下方物体往下压。而 ground_floating_objects 排在
    # 分离之前, 压下去之后就没人再管了。
    # 实测 wx5: chair_0 (supported by floor) 被分离推到 min_z=-0.541m,
    # 即埋进地下 54cm, 而 audit 仍报 ok (audit 只看 gap 符号)。
    # 这里在所有几何补正之后补一次落地, 保证 "supported by floor" 的物体
    # 底面最终确实在 z=0。只改 T[2,3], 不动水平位置与形状。
    all_instances, _n_reground = ground_floating_objects(
        all_instances, refined_relations, verbose=False, ref_z={})
    if _n_reground:
        print(f"   🔧 分离后收尾落地: {_n_reground} 个实例压回 z=0", flush=True)

    # 落盘摆放审计 (每实例世界 AABB / gap / 水平重叠), 供离线验收与后续修复比对。
    # 必须在所有补正之后 —— 记的是最终位姿。
    try:
        dump_placement_audit(all_instances, refined_relations,
                             placement_strategies=_strategies,
                             output_dir=output_path, verbose=True)
    except Exception as _e:
        print(f"   ⚠️ 摆放审计失败 (不影响主流水线): {_e}", flush=True)
    print(f"   ⏱️ 摆放补正耗时: {time.time() - _t525:.1f}s", flush=True)

    print(f"✅ Stage 5 完成", flush=True)
    return all_instances, refined_relations


def _render_glb_preview(glb_path):
    """GLB 保存后自动渲染预览 PNG (失败仅警告, 不影响主流水线)

    生成两张图:
      <stem>_preview.png  — 合成四视角 (front/side/top/iso)
      <stem>_camview.png  — 真实相机视角 (各实例'最大帧'/最优视角帧, 需输出目录有 extrinsics)
    """
    try:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        from tools.render_glb_preview import build_pyrender_scene, render_views, grid2x2
        scene, n_meshes = build_pyrender_scene(glb_path)
        if n_meshes == 0:
            return
        tiles = render_views(scene, scene.bounds, ["front", "side", "top", "iso"], 640)
        grid = grid2x2(tiles)
        out = os.path.splitext(glb_path)[0] + "_preview.png"
        cv2.imwrite(out, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"🖼️ 预览图已生成: {out}", flush=True)
    except Exception as e:
        print(f"⚠️ 预览图渲染失败 (不影响主流水线): {e}", flush=True)

    try:
        from tools.render_glb_preview import render_camview_auto
        render_camview_auto(glb_path)
    except Exception as e:
        print(f"⚠️ 相机视角渲染失败 (不影响主流水线): {e}", flush=True)


def save_final_glb(all_instances, output_path, filename="final_scene_basic.glb"):
    """
    保存最终GLB文件

    流程:
      1. 将每个实例的mesh应用变换矩阵T
      2. 添加到trimesh.Scene（node_name格式: {category}_{id}）
      3. z-up → y-up 变换（GLB标准坐标系）
      4. 导出GLB

    参数:
        all_instances: 精修后的3D资产实例
        output_path: 输出目录
        filename: 输出文件名
    返回:
        GLB文件路径
    """
    scene = trimesh.Scene()

    # 添加虚拟水平面标注 (z=0处的网格线, 不遮挡物体)
    grid_lines = []
    grid_range = 5.0
    grid_step = 0.5
    for v in np.arange(-grid_range, grid_range + grid_step, grid_step):
        grid_lines.append(trimesh.load_path(np.array([[v, -grid_range, 0], [v, grid_range, 0]])))
        grid_lines.append(trimesh.load_path(np.array([[-grid_range, v, 0], [grid_range, v, 0]])))
    grid = trimesh.util.concatenate(grid_lines)
    scene.add_geometry(grid, node_name="grid_z0")

    for category, category_instances in all_instances.items():
        for i, instance_info in enumerate(category_instances):
            mesh = instance_info['original_mesh']
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(instance_info['T'])
            scene.add_geometry(transformed_mesh, node_name=f"{category}_{i}")
    # z-up → y-up (GLB标准)
    scene.apply_transform(np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]))
    glb_path = os.path.join(output_path, filename)
    scene.export(glb_path)
    print(f"💾 GLB 已保存: {glb_path}", flush=True)
    _render_glb_preview(glb_path)
    return glb_path


def _record_pose_stage(all_instances, relations, stage_name, pose_history=None):
    """
    记录当前阶段每个物体的位姿到 pose_history

    Args:
        all_instances: {category: [instance_info, ...]}
        relations: {category: relationship}
        stage_name: 阶段名称 (initial / basic_refinement / stage5)
        pose_history: 已有的位姿历史 (None则新建)
    Returns:
        更新后的 pose_history
    """
    if pose_history is None:
        pose_history = {}

    for category, category_instances in all_instances.items():
        for inst_idx, info in enumerate(category_instances):
            key = f"{category}_{inst_idx}"
            T = info["T"]
            mesh = info["original_mesh"]
            transformed = mesh.copy()
            transformed.apply_transform(T)

            if key not in pose_history:
                pose_history[key] = {
                    "category": category,
                    "instance_idx": inst_idx,
                    "relation": relations.get(category, "unknown"),
                    "stages": {},
                }

            pose_history[key]["stages"][stage_name] = {
                "T_matrix": T.tolist(),
                "position": T[:3, 3].tolist(),
                "bounds_min": transformed.bounds[0].tolist(),
                "bounds_max": transformed.bounds[1].tolist(),
                "center": transformed.bounds.mean(axis=0).tolist(),
            }

            prev_stages = list(pose_history[key]["stages"].keys())
            if len(prev_stages) >= 2:
                prev_stage = prev_stages[-2]
                prev_pos = np.array(pose_history[key]["stages"][prev_stage]["position"])
                curr_pos = np.array(pose_history[key]["stages"][stage_name]["position"])
                delta = curr_pos - prev_pos
                pose_history[key]["stages"][stage_name]["delta_from_" + prev_stage] = delta.tolist()

    return pose_history


def main(args):
    """主流水线入口"""
    total_start = time.time()

    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"输入视频未找到: {args.input_video}")

    os.makedirs(args.output_path, exist_ok=True)
    
    # 配置日志输出
    # 日志名带场景名: 只用时间戳时, 不同场景的重跑会撞名 —— 实测
    # basic_pick_place_17 与 118、vertical_pick_place_101 与 111 各有同名日志
    # (如 mainv2_20261014_232716.log), 事后无法从文件名判断归属, 排查得逐个
    # grep 视频路径。加场景名后一瞥即知, 且物理上不可能再撞。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _scene_tag = os.path.basename(os.path.normpath(args.output_path))
    _scene_tag = "".join(c if (c.isalnum() or c in "-_") else "_" for c in _scene_tag)
    log_filename = os.path.join(
        args.output_path, f"mainv2_{_scene_tag}_{timestamp}.log")
    
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除已有handler

    # 格式化器: [时间戳] [级别] 消息
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # 文件handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 定义一个方便的print替代函数，同时输出到日志和控制台
    def log_print(msg, flush=False):
        logger.info(msg)
    
    # 替换print为log_print的功能
    import builtins
    original_print = builtins.print
    def new_print(*args, sep=' ', end='\n', file=sys.stdout, flush=False):
        msg = sep.join(str(arg) for arg in args) + end
        if not msg.endswith('\n'):
            msg += '\n'
        logger.info(msg.rstrip('\n'))
        original_print(*args, sep=sep, end=end, file=file, flush=flush)
    builtins.print = new_print
    
    log_print(f"📋 开始运行 ReplicateAnyScene V2")
    log_print(f"=" * 70)
    log_print(f"输入视频: {args.input_video}")
    log_print(f"输出目录: {args.output_path}")
    log_print(f"3D重建模型: {args.vggt_model}")
    log_print(f"日志文件: {log_filename}")
    log_print(f"=" * 70)

    # 自动查找VLM模型
    vlm_checkpoint = _resolve_vlm_checkpoint(args.vlm_checkpoint)
    if vlm_checkpoint is None:
        print("❌ 未找到VLM模型，请通过 --vlm_checkpoint 指定", flush=True)
        sys.exit(1)
    print(f"🤖 VLM模型: {vlm_checkpoint}", flush=True)

    # 设置 VLM 对话日志输出目录
    from src.vlm_logger import set_vlm_log_dir
    set_vlm_log_dir(args.output_path)
    print(f"📝 VLM对话日志将保存到: {os.path.join(args.output_path, 'vlm_conversation.jsonl')}", flush=True)

    # ── Stage 1: VGGT引导的智能物体发现 ──
    _t1 = time.time()
    if getattr(args, "scene_json", None):
        # 手工提供正确类别 → 跳过 VLM 检测 (它在大场景上会 hallucinate:
        # 实测 3chair-bottle 被报成 curtain x5 / router / tv / clothes rack)
        import shutil as _sh
        _sid = os.path.splitext(os.path.basename(args.input_video))[0]
        stage1_json = os.path.join(args.output_path, f"scene_{_sid}_stage1.json")
        _sh.copyfile(args.scene_json, stage1_json)
        with open(stage1_json, encoding="utf-8") as _f:
            categories_and_relations = json.load(_f)
        print(f"\n   📄 使用外部场景 JSON: {args.scene_json}", flush=True)
        print(f"      类别 ({len(categories_and_relations)}): "
              f"{sorted(categories_and_relations)}", flush=True)
    else:
        stage1_json, categories_and_relations = run_stage1(
        args.input_video, args.output_path, vlm_checkpoint,
        max_frames_stage1=args.max_frames_stage1,
        vggt_max_frames=args.max_frames,
    )
    _stage1_time = time.time() - _t1

    # ── Stage 2: VGGT 3D重建 + SAM3空间去重 ──
    _t2 = time.time()
    vggt_prediction_results, deduplicated_all_masks, wall_masks, floor_masks = run_stage2(
        args.input_video, args.output_path, categories_and_relations,
        max_frames=args.max_frames, vggt_model_type=args.vggt_model,
        vlm_checkpoint=vlm_checkpoint,
    )
    _stage2_time = time.time() - _t2

    # ── Stage 3: 最优视角资产生成 + 多票验证 ──
    _t3 = time.time()
    all_instances, all_optimal_frame_ids, deduplicated_all_masks = run_stage3(
        args.output_path, vggt_prediction_results, deduplicated_all_masks,
    )
    _stage3_time = time.time() - _t3

    # ── 尺寸校准: 点云校准 SAM3D 尺度漂移 (方案B, docs/SAM3D_SCALE_DRIFT_REPORT.md) ──
    # 必须在基础精修/all_instances.pkl 保存/Stage4/5 之前: 下游全部使用校准后尺寸
    _scale_report_p = os.path.join(args.output_path, "scale_calibration_report.json")
    try:
        from tools.rescale_meshes_by_pointcloud import rescale_all_instances
        _t_rescale = time.time()
        all_instances = rescale_all_instances(
            args.output_path, all_instances, all_optimal_frame_ids,
            report_path=_scale_report_p)
        print(f"✅ 尺寸校准完成 ({time.time() - _t_rescale:.1f}s)", flush=True)
    except Exception as _e:
        # 尺度校准失败会被静默降级 (k=1), 下游拿到的尺度漂移原样带入
        # (实测 wxD 圆凳 0.15m)。traceback + 落盘清单, 让失败可见。
        import traceback as _tb
        print(f"⚠️ 尺寸校准失败, 使用原始尺寸继续: {_e}", flush=True)
        traceback.print_exc()
        try:
            with open(_scale_report_p, "w", encoding="utf-8") as _f:
                json.dump({"status": "failed", "error": str(_e),
                           "traceback": _tb.format_exc()}, _f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    # ── 基础精修: floor/wall/embedded (始终执行，与main.py一致，在Stage4之前) ──
    _t_base = time.time()
    walls_info = get_walls_info(vggt_prediction_results['world_points'], wall_masks)

    # 同步 categories_and_relations: 移除在去重/3D资产生成中丢失的类别
    surviving_categories = set(all_instances.keys())
    lost_categories = set(categories_and_relations.keys()) - surviving_categories
    if lost_categories:
        print(f"\n⚠️  以下类别在去重/3D生成中丢失，从 categories_and_relations 中移除:", flush=True)
        for cat in sorted(lost_categories):
            print(f"    🚫 {cat}: {categories_and_relations[cat]}", flush=True)
        categories_and_relations = {k: v for k, v in categories_and_relations.items() if k in surviving_categories}

    print(f"\n{'='*70}", flush=True)
    print(f"🔧 基础精修: floor/wall/embedded", flush=True)
    print(f"{'='*70}", flush=True)

    # 保存初始3D资产摆放结果 (GLB #1)
    save_final_glb(all_instances, args.output_path, "final_scene_initial.glb")

    # 记录初始位姿
    pose_history = _record_pose_stage(all_instances, refined_relations if 'refined_relations' in dir() else categories_and_relations, "initial")

    # ── floor 关系坐实 (必须在基础精修之前) ──
    # VLM 常把桌上/托盘上的物体判成 floor; 基础精修会把 floor 物体砸到 z=0,
    # 之后 5.15 看到"在地上"就不再改挂 (实测 egodex_32 screwdriver 躺在
    # tray 上却被砸到地面)。这里用类别级关系构造实例级 dict 先做坐实检查,
    # 已改挂具名支撑物的实例在下方基础精修中跳过 (交给 Stage 5 的 5.2)。
    _floor_repaired = set()
    try:
        from src.sp_refinement import repair_floor_relations
        _inst_rel = {f"{_c}_{_i}": categories_and_relations[_c]
                     for _c, _lst in all_instances.items() for _i in range(len(_lst))}
        _r2, _n_fr = repair_floor_relations(all_instances, _inst_rel, verbose=True)
        if _n_fr:
            _floor_repaired = {n for n, rr in _r2.items() if rr != _inst_rel.get(n)}
            print(f"  🔧 floor 关系坐实: {_n_fr} 个实例改挂具名支撑物 "
                  f"{sorted(_floor_repaired)} (基础精修跳过, 交给 5.2)", flush=True)
    except Exception as _e:
        print(f"  ⚠️ floor 关系坐实检查失败 (继续基础精修): {_e}", flush=True)

    for category, category_instances in all_instances.items():
        relationship = categories_and_relations[category]
        # zip 截断防护: all_optimal_frame_ids 与实例数不匹配时, zip 会静默
        # 丢弃尾部实例 (不精修、不留痕)。提前告警, 让数量不一致暴露出来。
        _n_ids = len(all_optimal_frame_ids.get(category, []))
        if _n_ids != len(category_instances):
            print(f"  ⚠️ 基础精修: {category} 实例数 {len(category_instances)} != "
                  f"optimal_frame_ids 数 {_n_ids}, 尾部 "
                  f"{abs(len(category_instances) - _n_ids)} 个实例将被跳过精修", flush=True)
        for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(all_optimal_frame_ids[category], category_instances)):
            if f"{category}_{instance_id}" in _floor_repaired:
                print(f"  跳过基础精修: {category}_{instance_id} → 已改挂具名支撑物 (交给 5.2)", flush=True)
                continue
            if relationship == "supported_by_floor" or relationship == "supported by floor":
                print(f"  基础精修: {category}_{instance_id} → supported by floor", flush=True)
                instance_info = refine_supported_by_floor_object(instance_info)
            elif relationship == "embedded_in_wall" or relationship == "embedded in wall":
                print(f"  基础精修: {category}_{instance_id} → embedded in wall", flush=True)
                instance_info = refine_embedded_in_wall_object(instance_info, walls_info)
            elif relationship == "attached_to_wall" or relationship == "attached to wall":
                print(f"  基础精修: {category}_{instance_id} → attached to wall", flush=True)
                extrinsic = vggt_prediction_results['extrinsics'][optimal_frame_id]
                camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
                instance_info = refine_attached_to_wall_object(instance_info, walls_info, camera_pos)
            elif relationship == "held by hand":
                print(f"  跳过精修: {category}_{instance_id} → held by hand (保持VGGT位置)", flush=True)
                continue
            else:
                continue
            category_instances[instance_id] = instance_info

    # 保存基础精修后的结果 (GLB #2, Stage3产物, 供 run_post_pipeline.py 作为固定起点)
    save_final_glb(all_instances, args.output_path, "final_scene_basic.glb")

    # 记录基础精修位姿变化
    pose_history = _record_pose_stage(all_instances, categories_and_relations, "basic_refinement", pose_history)
    _base_time = time.time() - _t_base

    import pickle
    pkl_path = os.path.join(args.output_path, "all_instances.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'all_instances': all_instances,
            'all_optimal_frame_ids': all_optimal_frame_ids,
            'categories_and_relations': categories_and_relations,
            'walls_info': walls_info,
        }, f)
    print(f"💾 all_instances.pkl 已保存: {pkl_path}", flush=True)

    # ── Stage 4: 迭代视觉-空间对齐 (可选，默认关闭) ──
    _stage4_time = 0
    if args.enable_stage4:
        _t4 = time.time()
        all_instances = run_stage4(
            args.output_path, vggt_prediction_results, all_instances, args,
            categories_and_relations=categories_and_relations,
            walls_info=walls_info,
            deduplicated_all_masks=deduplicated_all_masks,
        )
        from tools.refine_inter_object_placement import resolve_penetrations
        all_instances = resolve_penetrations(all_instances, categories_and_relations, verbose=True,
                                             dry_run=True, scene_dir=args.output_path)
        save_final_glb(all_instances, args.output_path, "final_scene_stage4.glb")
        import pickle as _pkl
        s4_pkl_path = os.path.join(args.output_path, "all_instances_stage4.pkl")
        with open(s4_pkl_path, 'wb') as f:
            _pkl.dump(all_instances, f)
        print(f"💾 all_instances_stage4.pkl 已保存", flush=True)
        _stage4_time = time.time() - _t4
    else:
        print("\n⏭️  Stage 4 已跳过 (使用 --enable_stage4 启用)", flush=True)

    # ── Stage 5: 高级语义精修 (可选，默认关闭) ──
    _stage5_time = 0
    refined_relations = dict(categories_and_relations)
    if args.enable_stage5:
        _t5 = time.time()
        all_instances, refined_relations = run_stage5(
            args.output_path, categories_and_relations, all_instances,
            vggt_prediction_results, all_optimal_frame_ids,
            wall_masks, vlm_checkpoint, stage1_json,
            deduplicated_all_masks=deduplicated_all_masks,
            stage5_method=getattr(args, 'stage5_method', 'scene_graph'),
        )

        if args.enable_stage4:
            save_final_glb(all_instances, args.output_path, "final_scene_stage4_5.glb")
        else:
            # ── 重力可见化 + 退化桌面修复 (Stage 5 出口标准步骤) ──
            # 薄板桌面(桌腿未入画)合成桌腿并抬到 0.75m, 支撑物跟随; 加地板基准面。
            # 此前最终 GLB 无地板几何, 场景像漂在白空间, 无重力参考。
            try:
                from tools.ground_fix import add_ground_and_legs, verify_scene_ground
                all_instances, _ground_report = add_ground_and_legs(
                    all_instances, refined_relations, verbose=True)
                _gr = verify_scene_ground(all_instances, refined_relations, args.output_path)
                print(f"   🔬 出口自检: 沉地 {_gr['n_below_floor']} 悬空 {_gr['n_floating']} "
                      f"倒置 {_gr['n_inverted']}", flush=True)
            except Exception as _e:
                import traceback as _tb
                print(f"⚠️ 地板/桌腿修复失败 (跳过, 不影响主流水线): {_e}", flush=True)
                traceback.print_exc()
            save_final_glb(all_instances, args.output_path, "final_scene_stage5.glb")
        # 记录Stage5位姿变化
        pose_history = _record_pose_stage(all_instances, refined_relations, "stage5", pose_history)
        _stage5_time = time.time() - _t5

        # ── Stage 5 后重存 all_instances.pkl ──
        # 为什么必须重存: 原实现在 Stage 5 之前 (基础精修后) 存一次 pkl, 之后
        # Stage 4/5 改的所有位姿都不写回。下游 run_post_pipeline.py / 离线验收
        # 读这个 pkl, 拿到的是 5.25 之前的旧位姿。
        # 实测 wxD: pkl 里 table_0 顶面 z=0.050, 而同一次运行的
        # placement_audit.json 里是 0.987 —— 两者相差 0.94m, 因为 audit 在
        # Stage 5 内生成 (用的是新位姿), pkl 还是旧位姿。用 pkl 做验收会得出
        # 完全错误的结论 (比如误判"物体已贴地")。
        # 注意: 必须在 Stage 5 之后、任何下游消费之前。
        try:
            import pickle as _pk
            with open(pkl_path, 'wb') as f:
                _pk.dump({
                    'all_instances': all_instances,
                    'all_optimal_frame_ids': all_optimal_frame_ids,
                    'categories_and_relations': categories_and_relations,
                    'walls_info': walls_info,
                }, f)
            print(f"💾 all_instances.pkl 已更新 (Stage 5 后最终位姿): {pkl_path}",
                  flush=True)
        except Exception as _e:
            print(f"⚠️ Stage 5 后重存 pkl 失败 (不影响 GLB 产物): {_e}", flush=True)

        # Stage5 内置物理仿真验证 (可选, 实验性功能)
        if args.enable_physics_validation:
            print("\n  ── Stage 5.3 物理仿真验证 (SAPIEN) ──", flush=True)
            print("  ⚠️ 物理仿真是实验性功能，可能产生不合理结果", flush=True)
            from tools.physics_validator import PhysicsValidator
            validator = PhysicsValidator(
                sim_steps=args.physics_sim_steps,
                verbose=True,
            )
            all_instances_before = {cat: [dict(inst) for inst in insts] for cat, insts in all_instances.items()}
            all_instances, physics_report = validator.validate(
                all_instances,
                categories_and_relations=categories_and_relations,
                walls_info=walls_info,
                output_dir=os.path.join(args.output_path, "physics_val"),
            )
            # 位移保护: 如果物理仿真位移超过1m，拒绝该结果，恢复精修后位姿
            max_displacement = 1.0
            for cat in all_instances:
                for i, (inst_after, inst_before) in enumerate(zip(all_instances[cat], all_instances_before[cat])):
                    pos_after = inst_after["T"][:3, 3]
                    pos_before = inst_before["T"][:3, 3]
                    disp = np.linalg.norm(pos_after - pos_before)
                    if disp > max_displacement:
                        print(f"  🚫 {cat}_{i}: 位移={disp:.3f}m > {max_displacement}m, 拒绝物理仿真结果", flush=True)
                        all_instances[cat][i] = inst_before
            pose_history = _record_pose_stage(all_instances, refined_relations, "physics", pose_history)

            import json as _json
            report_path = os.path.join(args.output_path, "physics_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                _json.dump(physics_report, f, indent=2, ensure_ascii=False)
            print(f"  💾 物理验证报告: {report_path}", flush=True)
    else:
        print("\n⏭️  Stage 5 高级精修已跳过 (使用 --enable_stage5 启用)", flush=True)

    # ── Stage 6: 材质感知的物理参数标注 (可选，默认关闭) ──
    _stage6_time = 0
    if args.enable_stage6:
        _t6 = time.time()
        from tools.material_annotator import run_stage6
        materials = run_stage6(
            args.output_path, all_instances, refined_relations,
            vlm_checkpoint,
            instance_masks=deduplicated_all_masks,
            colors=vggt_prediction_results['colors'],
            preloaded_vlm=None,
        )
        _stage6_time = time.time() - _t6
    else:
        print("\n⏭️  Stage 6 材质物理标注已跳过 (使用 --enable_stage6 启用)", flush=True)

        # Stage5未启用时，物理验证仍可单独调用 (实验性功能)
        if args.enable_physics_validation:
            print("\n" + "=" * 70, flush=True)
            print("🔬 物理仿真验证 (SAPIEN, 独立模式)", flush=True)
            print("  ⚠️ 物理仿真是实验性功能，可能产生不合理结果", flush=True)
            print("=" * 70, flush=True)
            from tools.physics_validator import PhysicsValidator
            validator = PhysicsValidator(
                sim_steps=args.physics_sim_steps,
                verbose=True,
            )
            all_instances_before = {cat: [dict(inst) for inst in insts] for cat, insts in all_instances.items()}
            all_instances, physics_report = validator.validate(
                all_instances,
                categories_and_relations=categories_and_relations,
                walls_info=walls_info,
                output_dir=os.path.join(args.output_path, "physics_val"),
            )
            max_displacement = 1.0
            for cat in all_instances:
                for i, (inst_after, inst_before) in enumerate(zip(all_instances[cat], all_instances_before[cat])):
                    pos_after = inst_after["T"][:3, 3]
                    pos_before = inst_before["T"][:3, 3]
                    disp = np.linalg.norm(pos_after - pos_before)
                    if disp > max_displacement:
                        print(f"  🚫 {cat}_{i}: 位移={disp:.3f}m > {max_displacement}m, 拒绝物理仿真结果", flush=True)
                        all_instances[cat][i] = inst_before
            pose_history = _record_pose_stage(all_instances, refined_relations, "physics", pose_history)

            import json as _json
            report_path = os.path.join(args.output_path, "physics_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                _json.dump(physics_report, f, indent=2, ensure_ascii=False)
            print(f"💾 物理验证报告: {report_path}", flush=True)

    # 保存最终关系JSON
    with open(os.path.join(args.output_path, "final_relations.json"), 'w', encoding='utf-8') as f:
        json.dump(refined_relations, f, indent=2, ensure_ascii=False)

    # 保存位姿变化JSON (每个物体在各阶段的位置历史 + 坐标系对齐变换 + 相机外参变化)
    pose_path = os.path.join(args.output_path, "pose_changes.json")
    # 合并坐标系对齐信息 (由 run_stage2 保存的 coordinate_alignment.json)
    coord_align_path = os.path.join(args.output_path, "coordinate_alignment.json")
    if os.path.exists(coord_align_path):
        with open(coord_align_path, 'r', encoding='utf-8') as f:
            coord_align = json.load(f)
        # 提取所有帧的相机外参变化 (从 extrinsics/ 目录读取原始外参, 与对齐后对比)
        extrinsics_dir = os.path.join(args.output_path, "extrinsics")
        camera_changes = []
        if os.path.isdir(extrinsics_dir):
            for i in range(coord_align.get("n_frames", 0)):
                ext_file = os.path.join(extrinsics_dir, f"{i}.txt")
                if os.path.exists(ext_file):
                    ext_aligned = np.loadtxt(ext_file)
                    camera_changes.append({
                        "frame_id": i,
                        "extrinsic_aligned": ext_aligned.tolist(),
                    })
        pose_output = {
            "coordinate_alignment": coord_align,
            "camera_extrinsics_after_alignment": camera_changes,
            "objects": pose_history,
        }
    else:
        pose_output = pose_history
    with open(pose_path, 'w', encoding='utf-8') as f:
        json.dump(pose_output, f, indent=2, ensure_ascii=False)
    print(f"💾 位姿变化已保存: {pose_path}", flush=True)

    # 确定最终GLB路径用于日志输出
    if args.enable_stage4 and args.enable_stage5:
        glb_path = os.path.join(args.output_path, "final_scene_stage4_5.glb")
    elif args.enable_stage4:
        glb_path = os.path.join(args.output_path, "final_scene_stage4.glb")
    elif args.enable_stage5:
        glb_path = os.path.join(args.output_path, "final_scene_stage5.glb")
    elif args.enable_physics_validation:
        glb_path = os.path.join(args.output_path, "final_scene_physics.glb")
    else:
        glb_path = os.path.join(args.output_path, "final_scene_basic.glb")

    total_time = time.time() - total_start
    print("\n" + "=" * 70, flush=True)
    print(f"🎉 全部完成! 耗时 {total_time:.1f}s", flush=True)
    print(f"📂 输出目录: {args.output_path}", flush=True)
    print(f"📦 最终GLB: {glb_path}", flush=True)
    print("=" * 70, flush=True)

    # ── 各阶段耗时汇总 ──
    _stage_times = [
        ("Stage 1 (物体发现)", _stage1_time),
        ("Stage 2 (3D重建+去重)", _stage2_time),
        ("Stage 3 (资产生成)", _stage3_time),
        ("基础精修 (floor/wall)", _base_time),
    ]
    if _stage4_time > 0:
        _stage_times.append(("Stage 4 (视觉-空间对齐)", _stage4_time))
    if _stage5_time > 0:
        _stage_times.append(("Stage 5 (语义精修)", _stage5_time))
    if _stage6_time > 0:
        _stage_times.append(("Stage 6 (材质物理标注)", _stage6_time))

    _other_time = total_time - sum(t for _, t in _stage_times)
    _stage_times.append(("其他 (IO/初始化等)", max(0, _other_time)))

    print(f"\n⏱️ 各阶段耗时:", flush=True)
    _sorted = sorted(_stage_times, key=lambda x: x[1], reverse=True)
    for name, t in _sorted:
        pct = t / total_time * 100 if total_time > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {name:30s} {t:7.1f}s ({pct:5.1f}%) {bar}", flush=True)
    print(f"   {'总计':30s} {total_time:7.1f}s", flush=True)
    print(f"📝 运行日志已保存到: {log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReplicateAnyScene V2 — 完整自动化流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_video", type=str, default=None,
                        help="输入视频路径")
    parser.add_argument("--input_images", type=str, default=None,
                        help="输入图片目录路径 (与--input_video二选一)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="输出目录 (默认: ./output_v2/{video_stem})")
    parser.add_argument("--vlm_checkpoint", type=str, default=None,
                        help="VLM模型路径 (默认: /mnt/data/lza/models/Qwen3.5-9B)")
    parser.add_argument("--max_frames", type=int, default=160,
                        help="VGGT最大帧数 (Stage 2, 默认160)")
    parser.add_argument("--vggt_model", type=str, default="vggt",
                        choices=["vggt", "vggt_omega", "vggt4d"],
                        help="3D重建模型选择: vggt(默认), vggt_omega, vggt4d")
    parser.add_argument("--max_frames_stage1", type=int, default=10,
                        help="Stage1采样关键帧数 (默认10)")
    parser.add_argument("--enable_stage4", action="store_true",
                        help="启用Stage 4视觉-空间对齐")
    parser.add_argument("--disable_stage4", action="store_true",
                        help="关闭Stage 4视觉-空间对齐 (默认开启: 无对齐时物体位置"
                             "停留在单帧质心配准噪声水平, 实测偏离真实视野分米级)")
    parser.add_argument("--enable_stage5", action="store_true",
                        help="启用Stage 5语义感知场景精修 (默认关闭)")
    parser.add_argument("--stage5_method", type=str, default="scene_graph",
                        choices=["scene_graph", "per_object"],
                        help="Stage5.1关系推断方式: scene_graph(SimRecon风格,默认) | per_object(逐物体推断)")
    parser.add_argument("--stage4_iterations", type=int, default=8,
                        help="Stage4 ICP迭代次数 (默认8)")
    parser.add_argument("--stage4_temporal_radius", type=int, default=5,
                        help="Stage4 时序邻域半径 (默认5)")
    parser.add_argument("--stage4_use_mast3r", action="store_true",
                        help="Stage4 使用MASt3R匹配 (需要GPU)")
    parser.add_argument("--scene_json", type=str, default=None,
                        help="直接提供 Stage1 的场景 JSON (类别→关系), 跳过 VLM 检测。"
                             "用于 VLM 类别 hallucination 时手工修正后重跑。")
    parser.add_argument("--no_vlm_interaction", action="store_true",
                        help="关闭 Stage3.7 的 VLM 先行交互判断 (默认开启; "
                             "关闭则纯用几何判据)。")
    parser.add_argument("--no_vlm_box_fallback", action="store_true",
                        help="关闭 Stage2 的 box 兜底 (文本提示切不出 mask 时用 6.8 "
                             "出的框重新提示 SAM3)。默认开启; 关闭则完全走原逻辑。")
    parser.add_argument("--use_api_vlm", action="store_true",
                        help="Stage1 物体检测改用 opencode 的视觉模型 (默认本地 Qwen3.5-9B)。"
                             "实测对纯色背景+小物体更稳, 不会凭空造物体。")
    parser.add_argument("--all_sam3d", action="store_true",
                        help="所有实例都走 SAM3D (默认: 仅被抓取/操作的实例走 SAM3D, "
                             "静态实例改用 VGGT 点云直接构网以提速)")
    parser.add_argument("--enable_stage6", action="store_true",
                        help="启用Stage 6材质感知物理标注 (默认关闭)")
    parser.add_argument("--enable_physics_validation", action="store_true",
                        help="启用z轴合理性验证 (Stage5.3 或独立模式, 默认不启用)")
    parser.add_argument("--physics_sim_steps", type=int, default=300,
                        help="物理仿真检测步数 (默认300)")

    args = parser.parse_args()
    # Stage 4 默认开启 (位置配准的管线原生解; --disable_stage4 显式关闭)。
    # 实测 wxD: 无对齐时场景在真实相机视野外 (分米级偏移), 开启后桌面与
    # 真实帧完全重合。此前默认关闭是位置不准的根因之一。
    if getattr(args, "disable_stage4", False):
        args.enable_stage4 = False
    else:
        args.enable_stage4 = True
    # Stage2 box 兜底开关 (默认开)
    globals()['_VLM_BOX_FALLBACK'] = not getattr(args, 'no_vlm_box_fallback', False)
    # Stage3.7 的 VLM 交互判断开关 (传给 pick_place_detector)
    import src.pick_place_detector as _ppd
    _ppd._VLM_INTERACTION = not getattr(args, 'no_vlm_interaction', False)

    if args.input_images and args.input_video:
        parser.error("--input_video 和 --input_images 不能同时使用")
    if not args.input_images and not args.input_video:
        parser.error("必须指定 --input_video 或 --input_images")

    if args.input_images:
        if not os.path.isdir(args.input_images):
            parser.error(f"图片目录不存在: {args.input_images}")
        args.input_video = args.input_images

    if args.output_path is None:
        video_stem = os.path.splitext(os.path.basename(args.input_video))[0]
        args.output_path = os.path.join(".", "output_v2", f"{video_stem}_{args.vggt_model}")
    
    try:
        main(args)
    except Exception as e:
        # 确保异常也被记录到日志中
        import traceback
        
        # 尝试先初始化日志记录器（如果还没初始化）
        if not logging.getLogger().handlers:
            try:
                os.makedirs(args.output_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = os.path.join(args.output_path, f"mainv2_error_{timestamp}.log")
                logging.basicConfig(
                    level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler(log_filename, encoding='utf-8'),
                        logging.StreamHandler()
                    ]
                )
            except:
                pass
        
        logging.error(f"运行过程中出现异常: {e}")
        logging.error(traceback.format_exc())
        print(f"❌ 错误详情已记录到日志文件中")
        raise
