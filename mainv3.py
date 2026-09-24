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
  final_scene.glb          = Stage3 基础精修产物 (固定起点, 不再更改)
  final_scene_stage4.glb   = Stage4 产物 (视觉-空间对齐后)
  final_scene_stage5.glb   = 仅 Stage5 产物 (语义精修后)
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
  mainv2:          Stage1 → Stage2 → Stage3 → 基础精修 → [Stage4] → [Stage5] → final_scene.glb
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
  ├── final_scene.glb                    # 最终场景GLB (始终为最新结果)
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
    R, t = align_to_room_coordinate_system(
        vggt_prediction_results['world_points'], wall_masks, floor_masks
    )
    alignment_stage = "stage1_strict"
    alignment_info = {}

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
        R, t, alignment_info = align_via_large_plane(
            vggt_prediction_results['world_points'],
            floor_masks,
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
        alignment_stage = "stage4_geocalib"

    print(f"   坐标系对齐完成: {alignment_stage} (R[2,2]={R[2,2]:.4f})", flush=True)

    # 保存坐标系对齐变换 (供 pose_changes.json 合并 + 调试追溯)
    extrinsics_before = vggt_prediction_results['extrinsics'].copy()
    vggt_prediction_results = align_vggt_predictions(vggt_prediction_results, R, t)
    extrinsics_after = vggt_prediction_results['extrinsics']

    coord_align_info = {
        "alignment_stage": alignment_stage,
        "R": R.tolist(),
        "t": t.tolist(),
        "method_detail": alignment_info,
        "extrinsics_before_first_frame": extrinsics_before[0].tolist(),
        "extrinsics_after_first_frame": extrinsics_after[0].tolist(),
        "n_frames": int(extrinsics_before.shape[0]),
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
    all_masks = {}
    for category in detected_categories:
        print(f"   分割跟踪: {category}", flush=True)
        category_masks = segment_and_track(category, sam3_video_model, session_id)
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

    # 在子进程中生成3D资产（避免CUDA内存冲突）
    all_instances = generate_3d_asset_in_subprocess(
        deduplicated_all_masks,
        all_optimal_frame_ids,
        vggt_prediction_results['colors'],
        vggt_prediction_results['world_points'],
        vggt_prediction_results['extrinsics'],
    )

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

            # ── 动态判定: 三条件并置 ──
            # 条件1: global_disp > 0.05m — 首尾25%质心位移>5cm (物体真的换了位置)
            # 条件2: moving_ratio (>5cm) > 0.3 — 大位移帧占比>30% (不是偶发漂移)
            # 条件3: light_moving_ratio (>2cm) > 0.8 — 小位移帧占比>80% (持续稳定运动)
            is_dynamic_inst = (global_disp > 0.05) and (moving_ratio > 0.3) and (light_moving_ratio > 0.8)

            if not is_dynamic_inst:
                continue

            reasons = []
            if global_disp > 0.05: reasons.append(f"global_disp={global_disp:.3f}m")
            if moving_ratio > 0.3: reasons.append(f"moving_ratio={moving_ratio:.0%}")
            if light_moving_ratio > 0.8: reasons.append(f"light_moving_ratio={light_moving_ratio:.0%}")
            print(f"   {category}_{inst_idx}: [DYNAMIC] global_disp={global_disp:.3f}m, "
                  f"moving_ratio={moving_ratio:.0%}, light_moving_ratio={light_moving_ratio:.0%}, "
                  f"visible_ratio={visible_ratio:.0%}, frames={len(centroid_data)}/{total_frames} "
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
            total_frames = len(frame_ids)
            for fi, (fid, pts) in enumerate(zip(frame_ids, frame_pts)):
                t_ratio = fi / max(total_frames - 1, 1)
                r = int(255 * (1 - t_ratio))
                g = int(0)
                b = int(255 * t_ratio)
                color_rgb = np.full((pts.shape[0], 3), [r, g, b], dtype=np.uint8)
                combined = np.hstack([pts.astype(np.float64), color_rgb.astype(np.float64)])

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
                print(f"   {category}_{inst_idx}: 动态点云轨迹已保存: {ply_path} ({len(combined)} vertices)", flush=True)

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
                "motion_direction": motion_dir
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

            first_visible_centroid = None
            optimal_centroid = None
            first_visible_pts = None
            optimal_pts = None
            for fid, c in centroid_data:
                if fid == first_visible_frame_id and first_visible_centroid is None:
                    first_visible_centroid = c
                    # 提取首帧点云
                    for im_entry in sorted_masks:
                        if im_entry['frame_id'] == first_visible_frame_id:
                            mask_fv = im_entry['mask']
                            pointmap_fv = vggt_prediction_results['world_points'][first_visible_frame_id]
                            pts_fv = pointmap_fv[mask_fv > 0]
                            finite_fv = np.all(np.isfinite(pts_fv), axis=1)
                            first_visible_pts = pts_fv[finite_fv]
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
            # 平移：mesh 从最优帧位置移到首帧位置
            # 旋转：用 PCA 估计两帧的朝向差，旋转到首帧朝向
            if first_visible_centroid is not None and optimal_centroid is not None and first_visible_frame_id != optimal_frame_id:
                offset = first_visible_centroid - optimal_centroid
                needs_adjust = False

                # 平移调整
                if np.linalg.norm(offset) > 0.01:
                    T = instance_info["T"].copy()
                    T[:3, 3] += offset
                    needs_adjust = True

                # 旋转调整: PCA 估计主方向
                rot_applied = False
                if (first_visible_pts is not None and optimal_pts is not None
                        and len(first_visible_pts) > 10 and len(optimal_pts) > 10):
                    try:
                        # 中心化点云
                        pts_fv_c = first_visible_pts - first_visible_centroid
                        pts_opt_c = optimal_pts - optimal_centroid

                        # PCA: 求协方差矩阵的主轴
                        _, _, V_fv = np.linalg.svd(pts_fv_c.T @ pts_fv_c, full_matrices=True)
                        _, _, V_opt = np.linalg.svd(pts_opt_c.T @ pts_opt_c, full_matrices=True)

                        # V 的行就是主轴方向 (按方差降序排列)
                        # 旋转矩阵 R = V_fv @ V_opt.T
                        R_pca = V_fv @ V_opt.T

                        # 检查是否是合法旋转矩阵（反射矩阵行列式=-1需要修正）
                        if np.linalg.det(R_pca) < 0:
                            V_fv[:, 2] *= -1
                            R_pca = V_fv @ V_opt.T

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
                            msg = f"   {category}_inst{inst_idx}: [DYNAMIC] 位置+旋转调整: mesh帧{optimal_frame_id} → 首次可见帧{first_visible_frame_id}, offset={np.linalg.norm(offset):.4f}m"
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
        print("\n   📍 5.2: 无物体间支撑关系，跳过精修", flush=True)
    _t52_time = time.time() - _t52
    print(f"   ⏱️ 5.2 耗时: {_t52_time:.1f}s", flush=True)

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


def save_final_glb(all_instances, output_path, filename="final_scene.glb"):
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.output_path, f"mainv2_{timestamp}.log")
    
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
    try:
        from tools.rescale_meshes_by_pointcloud import rescale_all_instances
        _t_rescale = time.time()
        all_instances = rescale_all_instances(
            args.output_path, all_instances, all_optimal_frame_ids)
        print(f"✅ 尺寸校准完成 ({time.time() - _t_rescale:.1f}s)", flush=True)
    except Exception as _e:
        print(f"⚠️ 尺寸校准失败, 使用原始尺寸继续: {_e}", flush=True)

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

    for category, category_instances in all_instances.items():
        relationship = categories_and_relations[category]
        for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(all_optimal_frame_ids[category], category_instances)):
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
    save_final_glb(all_instances, args.output_path, "final_scene.glb")

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
            save_final_glb(all_instances, args.output_path, "final_scene_stage5.glb")
        # 记录Stage5位姿变化
        pose_history = _record_pose_stage(all_instances, refined_relations, "stage5", pose_history)
        _stage5_time = time.time() - _t5

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
        glb_path = os.path.join(args.output_path, "final_scene.glb")

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
                        help="启用Stage 4视觉-空间对齐 (默认关闭)")
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
    parser.add_argument("--enable_stage6", action="store_true",
                        help="启用Stage 6材质感知物理标注 (默认关闭)")
    parser.add_argument("--enable_physics_validation", action="store_true",
                        help="启用z轴合理性验证 (Stage5.3 或独立模式, 默认不启用)")
    parser.add_argument("--physics_sim_steps", type=int, default=300,
                        help="物理仿真检测步数 (默认300)")

    args = parser.parse_args()

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
