# mainv2.py 完整技术文档

> 整合时间: 2026-06-02
> 最近更新: 2026-06-25 (新增 main() 完整执行流程、修正参数表与代码对齐)
> 涵盖: mainv2 vs main 差异、新增模块逻辑、常见问题与解答

---

## 目录

1. [mainv2 vs main: 架构差异总览](#1-架构差异)
2. [mainv2 vs main: 逐阶段对比](#2-逐阶段对比)
3. [新增模块详解](#3-新增模块详解)
4. [GLB文件体系](#4-glb文件体系)
4.1. [位姿变化记录 pose_changes.json](#41-位姿变化记录-pose_changesjson)
5. [动态物体检测](#5-动态物体检测)
6. [SP精修逻辑](#6-sp精修逻辑)
7. [后处理管线 run_post_pipeline.py](#7-后处理管线)
8. [常见问题与解答](#8-常见问题)
9. [命令行参数完整参考](#9-命令行参数)
10. [GeoCalib 重力方向判断 (Stage 4 对齐)](#10-geocalib-重力方向判断-stage-4-对齐)
11. [修改记录](#11-修改记录)

---

## 1. 架构差异

| 维度 | main.py | mainv2.py |
|------|---------|-----------|
| 函数结构 | 1个平铺函数 `main(args)` | 模块化: `run_stage1/2/3/4/5` + `save_final_glb` + `_record_pose_stage` + `_resolve_vlm_checkpoint` |
| Stage1 | 手动加载 `--category_path` JSON | 自动发现 (subprocess调用 `generate_scene_json_stage1`) |
| VGGT模型 | 仅 vggt | vggt / vggt_omega / vggt4d |
| Stage4 | 注释 "代码暂不公开" | 完整实现 (可选 `--enable_stage4`) |
| Stage5 | 仅基础精修 (3种关系) | 基础精修(始终执行) + 5.1关系细化 + 5.2物体间精修 (可选) |
| VLM动态检测 | 无 | 点云位移检测 (中值位移+全局位移，无需VLM) |
| 日志系统 | 无 | 双输出 (文件+控制台)，`builtins.print` 劫持 |
| 输出目录 | 固定 | 含模型名 (`hoi4d_vggt_omega`) |
| 坐标对齐记录 | 无 | `coordinate_alignment.json` + 合并到 `pose_changes.json` |

### 核心原则

**mainv2 不指定 stage4/5 参数时，功能与 main.py 完全一致**（除Stage1自动发现外）。

```
mainv2.py (无 --enable_stage4/5):
  Stage1 → Stage2 → Stage3 → 基础精修 → final_scene.glb
  等价于 main.py 的完整流程

mainv2.py (--enable_stage4 --enable_stage5):
  Stage1 → Stage2 → Stage3 → 基础精修 → Stage4 → Stage5 → final_scene_stage4_5.glb
  (生成 5 个 GLB, 详见 §4 GLB文件体系)
```

### main() 完整执行流程 (mainv2.py 行 940-1278)

`main(args)` 是整个流水线的入口函数。以下是按代码顺序的完整执行流程，每一步标注对应行号、调用函数和产出文件。

#### 步骤 0: 初始化与日志配置 (行 942-1001)

| 行号 | 操作 | 说明 |
|------|------|------|
| 944-947 | 验证输入视频存在 + 创建输出目录 | `os.makedirs(args.output_path, exist_ok=True)` |
| 949-986 | 配置日志系统 | 文件 handler + 控制台 handler 双输出；`builtins.print` 被劫持为 `new_print`，所有 print 同时写入日志文件 |
| 997-1001 | `_resolve_vlm_checkpoint(args.vlm_checkpoint)` | 自动查找 VLM 模型 (Qwen3.5-9B → Qwen2.5-VL-3B)；找不到则 `sys.exit(1)` |

#### 步骤 1: Stage 1 物体发现 (行 1003-1010)

```python
# 行 1005-1009
stage1_json, categories_and_relations = run_stage1(
    args.input_video, args.output_path, vlm_checkpoint,
    max_frames_stage1=args.max_frames_stage1,   # 默认 10
    vggt_max_frames=args.max_frames,            # 默认 160
)
```

- **调用**: `run_stage1()` (行 206-297)，内部 subprocess 调用 `tools/generate_scene_json_stage1.py`
- **输入**: 视频 + VLM 模型
- **产出**: `scene_{id}_stage1.json` (类别+关系)、`keyframes/` 目录、`keyframes_metadata.json`
- **返回**: `(stage1_json路径, categories_and_relations字典)`

#### 步骤 2: Stage 2 3D重建 + 去重 + 坐标对齐 (行 1012-1018)

```python
# 行 1014-1017
vggt_prediction_results, deduplicated_all_masks, wall_masks, floor_masks = run_stage2(
    args.input_video, args.output_path, categories_and_relations,
    max_frames=args.max_frames, vggt_model_type=args.vggt_model,
)
```

- **调用**: `run_stage2()` (行 300-483)
- **内部流程**:
  1. VGGT/vggt_omega/vggt4d 预测 3D 点云、深度、外参 (行 336-349)
  2. SAM3 分割 floor/wall (行 351-360)
  3. **四阶段 Z 轴对齐** (行 362-411): Stage1 严格 → Stage2 放宽 → Stage3 大平面 → Stage4 GeoCalib
  4. `align_vggt_predictions()` 应用 R,t 变换到 extrinsics/world_points/point_cloud (行 395)
  5. 保存 `coordinate_alignment.json` (行 404-411)
  6. 保存 VGGT 中间结果: `color/`、`depth/`、`extrinsics/`、`intrinsic.txt`、`point_cloud.ply` (行 413-435)
  7. SAM3 video 逐类别分割+跟踪 (行 437-445)
  8. 类内去重 `self_category_deduplicate` (行 458-464)
  9. 跨类去重 `cross_category_deduplicate` (行 479-484)
  10. 全量动态分裂合并 `merge_dynamic_splits_all` (行 487-496)
  11. 白名单过滤 (只保留 Stage1 发现的类别) (行 509-514)
  12. 可视化 mask `instance_masks.mp4` (行 517-520)
- **产出**: `coordinate_alignment.json`、`point_cloud.ply`、`extrinsics/`、`instance_masks.mp4`、`deduplicated_all_masks`

#### 步骤 3: Stage 3 资产生成 (行 1020-1025)

```python
# 行 1022-1024
all_instances, all_optimal_frame_ids, deduplicated_all_masks = run_stage3(
    args.output_path, vggt_prediction_results, deduplicated_all_masks,
)
```

- **调用**: `run_stage3()` (行 485-638)
- **内部流程**:
  1. `get_optimal_view_frame_id()` 计算每个实例最优视角帧 (行 514) — 返回 `(frame_id, is_dynamic, motion_info)`
  2. 保存最优帧图像到 `optimal_frames/` (行 529-537)
  3. 保存 `instance_visibility.json` (行 539-548) — 供 Stage5.1 补充帧使用
  4. `generate_3d_asset_in_subprocess()` 在子进程中生成 3D mesh (行 551-557) — 避免 CUDA 冲突
  5. 动态物体位置调整: T 矩阵平移到首次可见帧质心 (行 559-614)
  6. `deduplicate_3d_assets()` 3D mesh 级二次去重 (行 616-622)
- **产出**: `all_instances` (核心数据结构: `{category: [instance_info, ...]}`)、`optimal_frames/`、`instance_visibility.json`
- **返回**: `(all_instances, all_optimal_frame_ids, deduplicated_all_masks)`

#### 步骤 4: 基础精修 (行 1027-1087，始终执行)

```python
# 行 1029: 计算墙壁信息
walls_info = get_walls_info(vggt_prediction_results['world_points'], wall_masks)

# 行 1031-1038: 同步 categories_and_relations (移除在去重/生成中丢失的类别)
surviving_categories = set(all_instances.keys())
lost_categories = set(categories_and_relations.keys()) - surviving_categories
categories_and_relations = {k: v for k, v in categories_and_relations.items() if k in surviving_categories}

# 行 1045: 保存初始 GLB (GLB #1)
save_final_glb(all_instances, args.output_path, "final_scene_initial.glb")

# 行 1048: 记录初始位姿
pose_history = _record_pose_stage(all_instances, ..., "initial")

# 行 1050-1069: 循环精修每个实例
for category, category_instances in all_instances.items():
    relationship = categories_and_relations[category]
    for instance_id, (optimal_frame_id, instance_info) in enumerate(...):
        if relationship == "supported by floor":
            instance_info = refine_supported_by_floor_object(instance_info)
        elif relationship == "embedded in wall":
            instance_info = refine_embedded_in_wall_object(instance_info, walls_info)
        elif relationship == "attached to wall":
            camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
            instance_info = refine_attached_to_wall_object(instance_info, walls_info, camera_pos)
        elif relationship == "held by hand":
            continue  # 跳过
        else:
            continue
        category_instances[instance_id] = instance_info  # ← 修复: 正确写回

# 行 1072: 保存基础精修后 GLB (GLB #2, 固定起点)
save_final_glb(all_instances, args.output_path, "final_scene.glb")

# 行 1075: 记录基础精修位姿
pose_history = _record_pose_stage(all_instances, categories_and_relations, "basic_refinement", pose_history)

# 行 1078-1087: 保存 all_instances.pkl
pickle.dump({'all_instances', 'all_optimal_frame_ids',
             'categories_and_relations', 'walls_info'}, ...)
```

- **产出**: `final_scene_initial.glb` (GLB #1)、`final_scene.glb` (GLB #2)、`all_instances.pkl`
- **关键**: 基础精修后的 `all_instances.pkl` 是 `run_post_pipeline.py` 独立重跑 Stage4/5 的固定起点

#### 步骤 5: Stage 4 视觉-空间对齐 (行 1089-1109，可选 `--enable_stage4`)

```python
# 行 1091-1107
if args.enable_stage4:
    all_instances = run_stage4(args.output_path, vggt_prediction_results,
                               all_instances, args, ...)
    # 行 1099-1100: 穿模修复 (dry_run 模式)
    from tools.refine_inter_object_placement import resolve_penetrations
    all_instances = resolve_penetrations(all_instances, categories_and_relations,
                                        verbose=True, dry_run=True,
                                        scene_dir=args.output_path)
    # 行 1101: 保存 Stage4 GLB (GLB #3)
    save_final_glb(all_instances, args.output_path, "final_scene_stage4.glb")
    # 行 1102-1106: 保存 all_instances_stage4.pkl
    pickle.dump(all_instances, ...)
else:
    print("⏭️  Stage 4 已跳过")
```

- **调用**: `run_stage4()` (行 640-750) — MASt3R/深度匹配 → Umeyama 变换 → ICP 精调
- **产出**: `final_scene_stage4.glb` (GLB #3)、`all_instances_stage4.pkl`

#### 步骤 6: Stage 5 语义精修 (行 1111-1166，可选 `--enable_stage5`)

```python
# 行 1113: refined_relations 初始化为 categories_and_relations 的副本
refined_relations = dict(categories_and_relations)

if args.enable_stage5:
    # 行 1116-1122: 调用 run_stage5()
    all_instances, refined_relations = run_stage5(
        args.output_path, categories_and_relations, all_instances,
        vggt_prediction_results, all_optimal_frame_ids,
        wall_masks, vlm_checkpoint, stage1_json,
        deduplicated_all_masks=deduplicated_all_masks,
        stage5_method=getattr(args, 'stage5_method', 'scene_graph'),
    )
    # 行 1124-1127: 保存 Stage5 GLB
    if args.enable_stage4:
        save_final_glb(all_instances, args.output_path, "final_scene_stage4_5.glb")  # GLB #5
    else:
        save_final_glb(all_instances, args.output_path, "final_scene_stage5.glb")   # GLB #4
    # 行 1129: 记录 Stage5 位姿
    pose_history = _record_pose_stage(all_instances, refined_relations, "stage5", pose_history)

    # 行 1132-1164: 物理仿真验证 (可选 --enable_physics_validation)
    if args.enable_physics_validation:
        from tools.physics_validator import PhysicsValidator
        validator = PhysicsValidator(sim_steps=args.physics_sim_steps, ...)
        all_instances, physics_report = validator.validate(all_instances, ...)
        # 位移保护: >1m 拒绝结果, 恢复精修后位姿
        pose_history = _record_pose_stage(all_instances, refined_relations, "physics", pose_history)
else:
    print("⏭️  Stage 5 高级精修已跳过")
```

- **调用**: `run_stage5()` (行 753-838) — 5.1 关系推断 + 5.2 SP精修 + check_stability
- **产出**: `final_scene_stage4_5.glb` 或 `final_scene_stage5.glb`、`final_relations.json` (后续保存)

#### 步骤 7: 最终输出保存 (行 1203-1235)

```python
# 行 1204-1205: 保存最终关系 JSON
json.dump(refined_relations, ...)  # → final_relations.json

# 行 1207-1235: 保存 pose_changes.json (合并三部分)
coord_align_path = os.path.join(args.output_path, "coordinate_alignment.json")
if os.path.exists(coord_align_path):
    coord_align = json.load(open(coord_align_path))
    # 逐帧读取对齐后的相机外参
    extrinsics_dir = os.path.join(args.output_path, "extrinsics")
    camera_changes = [{"frame_id": i, "extrinsic_aligned": ...} for i in range(n_frames)]
    pose_output = {
        "coordinate_alignment": coord_align,           # Stage2 坐标系对齐 R,t
        "camera_extrinsics_after_alignment": camera_changes,  # 每帧对齐后外参
        "objects": pose_history,                       # 每个物体各阶段的 T 矩阵+位置
    }
else:
    pose_output = pose_history  # 旧版回退
json.dump(pose_output, ...)  # → pose_changes.json
```

- **产出**: `final_relations.json`、`pose_changes.json`

#### 步骤 8: 耗时统计与输出 (行 1237-1278)

```python
# 行 1237-1247: 确定最终 GLB 路径
if args.enable_stage4 and args.enable_stage5:
    glb_path = ".../final_scene_stage4_5.glb"
elif args.enable_stage4:
    glb_path = ".../final_scene_stage4.glb"
elif args.enable_stage5:
    glb_path = ".../final_scene_stage5.glb"
elif args.enable_physics_validation:
    glb_path = ".../final_scene_physics.glb"
else:
    glb_path = ".../final_scene.glb"

# 行 1249-1278: 总耗时统计 + 各阶段耗时排序输出
```

#### 完整数据流总览

```
视频输入
  │
  ├─ Stage1 (run_stage1, 行 1005)
  │   └→ categories_and_relations + stage1_json + keyframes/
  │
  ├─ Stage2 (run_stage2, 行 1014)
  │   ├→ vggt_prediction_results (含对齐后的 extrinsics/world_points)
  │   ├→ deduplicated_all_masks + wall_masks + floor_masks
  │   └→ coordinate_alignment.json + point_cloud.ply + extrinsics/
  │
  ├─ Stage3 (run_stage3, 行 1022)
  │   └→ all_instances + all_optimal_frame_ids + optimal_frames/
  │
  ├─ 基础精修 (行 1050-1069)
  │   └→ all_instances (T矩阵更新) + final_scene.glb + all_instances.pkl
  │
  ├─ [Stage4] (run_stage4, 行 1093)
  │   └→ all_instances (视觉对齐) + final_scene_stage4.glb + all_instances_stage4.pkl
  │
  ├─ [Stage5] (run_stage5, 行 1116)
  │   └→ all_instances (SP精修) + refined_relations + final_scene_stage4_5.glb
  │
  └─ 最终输出 (行 1203-1235)
      └→ final_relations.json + pose_changes.json
```

---

## 2. 逐阶段对比

### Stage 1: 物体发现

| | main.py | mainv2.py |
|---|---------|-----------|
| 方式 | 手动 `--category_path` | 自动 subprocess 调用 `generate_scene_json_stage1` |
| VLM | 不需要 | Qwen3.5-9B / Qwen2.5-VL-3B (自动回退) |
| 关键帧 | 无 | 贪心采样12帧 + `keyframes_metadata.json` |
| 关系格式 | `supported_by_floor` (下划线) | 兼容下划线和空格 |
| 代码位置 | `with open(args.category_path, 'r') as f:` | `run_stage1()` 函数，subprocess调用 |
| 输入参数 | `--category_path` (必须) | `--input_video` (必须), `--max_frames_stage1` (默认12) |
| 输出文件 | 无 | `scene_{id}_stage1.json` + `keyframes/` |

### Stage 2: 3D重建 + 去重

| | main.py | mainv2.py |
|---|---------|-----------|
| VGGT模型 | 仅 `vggt_predict` | 按 `--vggt_model` 选择 (vggt/vggt_omega/vggt4d) |
| 模型加载 | `load_vggt_model()` | `load_vggt_model()` / `load_vggt_omega_model()` / `load_vggt4d_model()` |
| 帧加载 | `load_video_frames(video, max_frames)` | vggt/vggt4d: `load_video_frames`; vggt_omega: `load_vggt_omega_frames` |
| 图像分辨率 | 518 (patch_size=14) | vggt/vggt4d: 518; vggt_omega: 512 (patch_size=16) |
| 预测函数 | `vggt_predict(frames, model)` | `vggt_predict` / `vggt_omega_predict` / `vggt4d_predict` |
| protected_categories | **无** | **白名单过滤**: 跨类去重后，只保留Stage1 JSON中存在的类别 (`json_categories_set`)，过滤掉SAM3误检的类别 |
| 跨类去重调用 | `cross_category_deduplicate(all_masks, pts, conf)` | `cross_category_deduplicate(all_masks, pts, conf)` + 白名单过滤 |
| 代码位置 | 平铺在main()中 | `run_stage2()` 函数 |

**关键差异: VGGT模型选择**

```python
# main.py: 固定使用VGGT
frames = load_video_frames(args.input_video, args.max_frames).to(device)
vggt_model = load_vggt_model().to(device)
vggt_prediction_results = vggt_predict(frames, vggt_model)

# mainv2.py: 按 --vggt_model 参数选择
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
```

**关键差异: VGGT-Omega 分辨率不同**

VGGT-Omega 使用 patch_size=16, image_resolution=512，而 VGGT/VGGT4D 使用 patch_size=14, image_resolution=518。每个模型必须用自己的 `load_*_frames` 函数，不能用 `ReplicateAnyScene` 的 `load_video_frames`。

**关键差异: Stage 2 四阶段 Z 轴对齐 (新增)**

`run_stage2()` 在 SAM3 分割 floor/wall 后，按以下优先级逐级尝试对齐到房间坐标系，只有当前阶段失败（返回 identity R,t）才进入下一阶段：

| 阶段 | 函数 | 输入 | 成功条件 | 阈值 |
|------|------|------|---------|------|
| 1 (严格) | `align_to_room_coordinate_system` | SAM3 `floor`/`wall` 文本提示 mask | 同时存在有效 floor 和正交 wall 平面 | mean_distance < 0.02m, 正交角 85° |
| 2 (放宽) | `align_via_objects` | 放宽阈值的 floor (+ wall 或点云 PCA) | 存在有效 floor 平面 | mean_distance < 0.05m, 正交角 80° |
| 3 (大平面) | `align_via_large_plane` | SAM3 大平面 mask (`flat surface`/`ground`) | 存在有效大平面 | mean_distance < 0.05m |
| 4 (GeoCalib) | `align_via_geocalib` | GeoCalib 重力估计 (从图像推算) | 至少一帧重力估计成功且内点足够 | MAD 3.0 |

**实际代码** (mainv2.py 行 362-392):

```python
# Stage1 严格: 要求 floor + 正交 wall
R, t = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
alignment_stage = "stage1_strict"

# Stage1 失败 (返回 identity) → Stage2 放宽: 阈值放宽, 允许 PCA fallback
if np.allclose(R, np.eye(3), atol=1e-6):
    R, t, alignment_info = align_via_objects(world_points, wall_masks, floor_masks)
    alignment_stage = "stage2_relaxed"

# Stage2 失败 → Stage3 大平面: 用 SAM3 大平面 mask 当 floor
if np.allclose(R, np.eye(3), atol=1e-6):
    R, t, alignment_info = align_via_large_plane(world_points, floor_masks)
    alignment_stage = "stage3_large_plane"

# Stage3 失败 → Stage4 GeoCalib: 从图像估计重力方向, 不需要 floor mask!
if np.allclose(R, np.eye(3), atol=1e-6):
    R, t, alignment_info = align_via_geocalib(colors, world_points, max_frames=8)
    alignment_stage = "stage4_geocalib"
```

**为什么需要四阶段级联？** 桌面场景（如 basic_pick_place）通常无可见地面，SAM3 找不到 floor mask，Stage 1-3 全部失败。Stage 4 (GeoCalib) 从图像直接估计重力方向，不需要 floor mask，是桌面场景的关键 fallback。实测 5 个 basic_pick_place 视频中，3 个仅靠 Stage 4 才成功对齐 z 轴。

**坐标系对齐变换保存**: 对齐完成后，R, t, alignment_stage, alignment_info 保存到 `coordinate_alignment.json`，并合并到 `pose_changes.json` 的 `coordinate_alignment` 字段（详见 §4.1）。

**GeoCalib 重力方向的重要约定** (详见 §11):
- GeoCalib 的 `gravity.vec3d` 返回**重力方向**（指向 DOWN）
- `floor_normal` 应指向 UP（世界 z 轴正方向）= `-gravity`
- 代码: `floor_normal = -gravity_vec / np.linalg.norm(gravity_vec)`

**关于 SAM3 点提示**: `sam3/model/sam3_image_processor.py` 中的 `Sam3Processor` 只暴露了 `add_geometric_prompt(box=..., label=...)`，没有公开的点提示 API。底层 `FindStage` 虽然预留了 `input_points`/`input_points_mask` 字段，但图像推理封装未开放。

### Stage 3: 资产生成

| | main.py | mainv2.py |
|---|---------|-----------|
| `get_optimal_view_frame_id` | 返回3个值 (frame_id, is_dynamic, motion_info) | 一致 |
| 动态检测 | 中值位移 + 全局位移 | 一致 |
| instance_visibility | 无 | 新增，供Stage5.1使用 |
| 代码位置 | 平铺在main()中 | `run_stage3()` 函数 |

**关键差异: 动态检测返回值**

main.py 和 mainv2.py 都正确接收3个返回值:
```python
optimal_frame_id, is_dynamic, motion_info = get_optimal_view_frame_id(
    vggt_prediction_results['world_points'], instance_masks
)
```

### 基础精修 (Stage 5.0)

| | main.py | mainv2.py |
|---|---------|-----------|
| 关系匹配 | 仅 `supported_by_floor` (下划线) | 兼容 `supported_by_floor` 和 `supported by floor` (空格) |
| 精修写回 | **BUG: 未写回** | **已修复: `category_instances[instance_id] = instance_info`** |
| 代码位置 | 平铺在main()中 | main()中，Stage4之前 |
| camera_pos计算 | `- extrinsic[:3,:3].T @ extrinsic[:3,3]` | `-extrinsic[:3, :3].T @ extrinsic[:3, 3]` (仅格式差异) |

**关键BUG修复: 精修结果未写回**

```python
# main.py: BUG - 精修结果未写回
for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(...)):
    if relationship == "supported_by_floor":
        instance_info = refine_supported_by_floor_object(instance_info)
    # ...
    else:
        continue
    # ← 缺少: category_instances[instance_id] = instance_info

# mainv2.py: 已修复
for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(...)):
    if relationship == "supported_by_floor" or relationship == "supported by floor":
        instance_info = refine_supported_by_floor_object(instance_info)
    # ...
    else:
        continue
    category_instances[instance_id] = instance_info  # ← 修复: 正确写回
```

### Stage 4: 视觉-空间对齐 (mainv2新增)

| | main.py | mainv2.py |
|---|---------|-----------|
| 状态 | 注释 "代码暂不公开" | 完整实现 (需 `--enable_stage4`) |
| 功能 | — | MASt3R/深度匹配 → Umeyama变换 → ICP精调 |
| 穿模解决 | — | Stage4后执行 `resolve_penetrations` |
| 参数 | — | `--stage4_iterations`, `--stage4_temporal_radius`, `--stage4_use_mast3r` |

### Stage 5: 语义精修 (mainv2新增)

| | main.py | mainv2.py |
|---|---------|-----------|
| 基础精修 | Stage5内 (3种关系) | **独立为5.0**，始终执行 |
| 5.1 关系细化 | 无 | **新增**: VLM投票细化 "supported by other objects" → "supported by {name}" |
| 5.2 空间精修 | 无 | **新增**: 物体间支撑关系几何精修 (on_top/inside/against_side等) |

### GLB保存

| | main.py | mainv2.py |
|---|---------|-----------|
| 保存方式 | 平铺在main()中 | `save_final_glb()` 函数 |
| 输出文件 | `final_scene.glb` | `final_scene_base.glb` (固定起点) + `final_scene.glb` (最新结果) |
| z-up→y-up变换 | `[[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]` | 一致 |
| node_name | `{category}_{i}` | 一致 |

### 其他差异

| | main.py | mainv2.py |
|---|---------|-----------|
| 日志系统 | 无 | 双输出 (文件+控制台)，`builtins.print` 劫持 |
| 异常处理 | 无 | 顶层 try/except + 日志记录 |
| 耗时统计 | 无 | 总耗时统计 (`time.time()`) |
| 输出目录 | 固定 `./outputs/hallway` | 自动生成 `./output_v2/{video_stem}_{vggt_model}` |
| 最终关系JSON | 无 | `final_relations.json` |
| 实例数据pkl | 无 | `all_instances.pkl` (含all_instances, all_optimal_frame_ids, categories_and_relations, walls_info) |
| VLM动态检测 | 无 | `get_optimal_view_frame_id()` 返回 is_dynamic (纯点云位移，未接入VLM) |
| 删除的导入 | `generate_3d_asset` (直接调用) | 仅保留 `generate_3d_asset_in_subprocess` |
| `--category_path` | 必须指定 (默认 `./assets/example/hallway.json`) | **已移除**，Stage1自动发现 |
| `--input_images` | 无 | 新增，与 `--input_video` 互斥 |
| `--input_video` 默认值 | `'./assets/example/hallway.mp4'` | None (必须指定) |
| `--output_path` 默认值 | `'./outputs/hallway'` | None (自动生成) |
| 打印语言 | 英文 | 中文 |
| 注释掉的代码 | 含大段注释的替代3D资产生成代码 (Inference类) | 已移除 |
| VLM模型查找 | 无 | `_resolve_vlm_checkpoint()` 自动查找 (Qwen3.5-9B → Qwen2.5-VL-3B) |
| Stage5逻辑 | 注释占位 | `run_stage5()` 函数已定义并在 `main()` 中调用，`main()` 内只负责 GLB 命名与 pose 记录 |
| 输入验证 | 仅检查文件存在 | 互斥验证 + 目录检查 + 自动统一为input_video |

**关键差异: `--category_path` 已移除**

main.py 需要 `--category_path` 手动指定场景JSON，mainv2 通过 Stage1 自动发现，不再支持此参数。

**关键差异: `--input_images` 与 `--input_video` 互斥**

```python
# mainv2.py 输入验证
if args.input_images and args.input_video:
    parser.error("--input_video 和 --input_images 不能同时使用")
if not args.input_images and not args.input_video:
    parser.error("必须指定 --input_video 或 --input_images")
if args.input_images:
    if not os.path.isdir(args.input_images):
        parser.error(f"图片目录不存在: {args.input_images}")
    args.input_video = args.input_images  # 统一为 input_video
```

**关键差异: 输出路径自动生成**

```python
# main.py: 固定默认值
--output_path default='./outputs/hallway'

# mainv2.py: 动态生成
if args.output_path is None:
    video_stem = os.path.splitext(os.path.basename(args.input_video))[0]
    args.output_path = os.path.join(".", "output_v2", f"{video_stem}_{args.vggt_model}")
```

**关键差异: VLM模型自动查找**

```python
def _resolve_vlm_checkpoint():
    """按优先级自动查找VLM模型"""
    # 1. 用户指定路径
    # 2. /mnt/data/lza/models/Qwen3.5-9B
    # 3. /mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/{hash}
    # 找不到时 sys.exit(1)
```

**关键差异: Stage5 逻辑统一**

`run_stage5()` 函数已统一封装 5.1 关系细化与 5.2 SP 精修逻辑，`main()` 直接调用并负责最终 GLB 命名与 `pose_changes.json` 记录。

---

## 3. 新增模块详解

### 3.1 Stage 4: 迭代视觉-空间对齐 (mainv2.py 行 640-750)

**目的**: 用多帧视觉匹配修正单帧点云对齐的位姿误差。

**实际流程** (对齐论文 Section 3.4):
```
1. 准备: 用 SAM 分割 mask 作为 real mask (论文 M_real)
   - 若 SAM mask 与 all_instances 不匹配, 回退到 create_depth_based_masks
2. 对每个实例调用 refine_single_instance_combined:
   Phase A: MASt3R/深度匹配 → 3D Lifting → Umeyama 相似变换 (含尺度)
   Phase B: ICP 精调 → 渐进阈值 + RANSAC
3. 输出: 对齐后的 T 矩阵直接写回 all_instances[cat][i]
```

**关键参数** (mainv2.py 行 735-741):
- `num_icp_iterations` (`--stage4_iterations`, 默认8)
- `temporal_radius` (`--stage4_temporal_radius`, 默认5)
- `use_mast3r` (`--stage4_use_mast3r`, 默认False)

**坐标系**: Stage4 在 z-up 空间操作，输出 T 矩阵直接应用到 all_instances。

**穿模解决**: Stage4 后在 main() 中执行 `resolve_penetrations(dry_run=True)` (mainv2.py 行 1100)。

### 3.2 Stage 5.1: 关系推断 (mainv2.py 行 777-812)

**目的**: 将 "supported by other objects" 细化为 "supported by {具体物体名}"。

**两种方法** (`--stage5_method`, 默认 `scene_graph`):

| 方法 | 函数 | 流程 | 适用场景 |
|------|------|------|---------|
| `scene_graph` (默认, 推荐) | `infer_relations_scene_graph` | 在 ID 标注图上一次性推断所有关系 (SimRecon 风格) | 通用, 速度快 |
| `per_object` (旧方式) | `refine_other_objects_relations` | 逐物体裁剪图像 → VLM 判断 → 多帧投票 | 只细化 "supported by other objects" |

**scene_graph 方法流程**:
```
1. select_best_frame_for_labeling: 选 mask 覆盖率最高的帧
2. create_id_labeled_image: 在该帧上为每个实例画 ID 标注框
3. VLM 一次性推断: 给定标注图 + 物体列表 → 输出关系 JSON
4. 输出: relations_scene_graph.json + id_scene.png + id_scene_mapping.json
```

### 3.3 Stage 5.2: 物体间 SP 精修 (mainv2.py 行 815-836, refine_inter_object_placement.py)

**目的**: 根据细化后的关系，对物体间支撑关系做几何精修。

**触发条件**: refined_relations 中存在 `supported by {name}` (非 floor/other objects) 的关系。

**调用**: `refine_inter_object_relations(all_instances, refined_relations, ...)` (行 825-832)

**内部流程** (refine_inter_object_placement.py):
```
1. 主循环: 拓扑排序, 对每个 "supported by {name}" 的物体:
   - _find_supporter_instances: 解析 supporter_name (如 "table_1") → 找到实例
   - _find_nearest_supporter_instance: 用 xy 距离找最近的 supporter 实例
   - 根据 关系类型调用 sp_refine_on_top / sp_refine_inside / sp_refine_against_side
2. resolve_penetrations: 穿模修复 (AABB 预筛 + FCL 精确检测)
3. 保存 final_scene_stage5_sp.glb (行 1934)
4. check_stability: 旋转对齐 + z 轴贴合 + 悬空修复 (4个 Phase)
5. 保存最终 GLB (final_scene_stage4_5.glb 或 final_scene_stage5.glb)
```

**5种放置策略**:

| 关系 | 放置策略 | 物理约束 |
|------|---------|---------|
| supported by {name} | on_top | 底面贴顶面，不穿模不悬空 |
| inside {name} | inside | 放入容器内部，不穿出 |
| against_side of {name} | against_side | 底面贴地 + 侧面接触 |
| hanging_below {name} | hanging_below | 顶面贴支撑物底面 |
| leaning on {name} | leaning | 底面贴地 + 倾斜靠在支撑物上 |

**精修核心逻辑**: 详见[第6节](#6-sp精修逻辑)。

---

## 4. GLB文件体系

### GLB 生成流程图 (核心)

`all_instances` 是贯穿全流程的核心数据结构。每个阶段修改它，然后保存一个 GLB 快照。

```
Stage1 (物体发现)
    │  输出: categories_and_relations (JSON)
    ▼
Stage2 (3D重建 + Z轴对齐 + SAM3去重)
    │  修改: vggt_prediction_results (对齐R,t)
    │  输出: world_points, colors, extrinsics, point_cloud.ply
    ▼
Stage3 (最优视角资产生成)
    │  修改: all_instances = {category: [instance_info, ...]}
    │        每个 instance_info 含 original_mesh + T矩阵
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  save_final_glb(all_instances, "final_scene_initial.glb")       │
│  ► GLB #1: Stage3原始结果, 未做任何精修                          │
│  ► 代码: mainv2.py 行 1045                                      │
│  ► 数据来源: all_instances (Stage3刚生成)                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼  基础精修 (floor/wall/embedded, 始终执行)
       修改: all_instances 中每个物体的 T 矩阵
       (refine_supported_by_floor / refine_embedded_in_wall / refine_attached_to_wall)
┌─────────────────────────────────────────────────────────────────┐
│  save_final_glb(all_instances, "final_scene.glb")               │
│  ► GLB #2: 基础精修后结果 (固定起点)                             │
│  ► 代码: mainv2.py 行 1072                                      │
│  ► 数据来源: all_instances (基础精修后)                          │
│  ► 同时保存: all_instances.pkl (供 run_post_pipeline.py 使用)    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼  --enable_stage4?
    │
    ├── YES ──► Stage4 (迭代视觉-空间对齐)
    │           修改: all_instances (refine_single_instance_combined)
    │           + resolve_penetrations(dry_run=True)
    │           ┌─────────────────────────────────────────────────┐
    │           │ save_final_glb(all_instances,                   │
    │           │   "final_scene_stage4.glb")                     │
    │           │ ► GLB #3: Stage4对齐后结果                      │
    │           │ ► 代码: mainv2.py 行 1101                       │
    │           │ ► 数据来源: all_instances (Stage4后)             │
    │           │ ► 同时保存: all_instances_stage4.pkl             │
    │           └─────────────────────────────────────────────────┘
    │
    ▼  --enable_stage5?
    │
    ├── YES ──► Stage5 (高级语义精修, run_stage5 行 1116)
    │           │
    │           ├── 5.1: 关系推断 (scene_graph / per_object, 行 777-812)
    │           │       修改: refined_relations (不修改 all_instances)
    │           │
    │           ├── 5.2: SP精修 (有物体间关系时, refine_inter_object_placement.py)
    │           │       修改: all_instances (on_top/inside/against_side)
    │           │       + resolve_penetrations (穿模修复)
    │           │       ┌────────────────────────────────────────┐
    │           │       │ 保存: "final_scene_stage5_sp.glb"      │
    │           │       │ ► GLB #4: SP精修+穿模修复后中间结果     │
    │           │       │ ► 代码: refine_inter_object_placement  │
    │           │       │       .py 行 1934                       │
    │           │       │ ► 数据来源: all_instances (SP精修后)    │
    │           │       └────────────────────────────────────────┘
    │           │
    │           └── check_stability (稳定性检查)
    │               修改: all_instances (旋转对齐+z贴合+悬空修复)
    │               ┌────────────────────────────────────────────┐
    │               │ 保存: final_glb_name (mainv2.py 行 1125/1127)│
    │               │   有stage4 → "final_scene_stage4_5.glb"    │
    │               │   无stage4 → "final_scene_stage5.glb"      │
    │               │ ► GLB #5 (或#4无stage4时): 最终结果        │
    │               │ ► 代码: refine_inter_object_placement       │
    │               │       .py 内部 check_stability             │
    │               │ ► 数据来源: all_instances (check_stability)│
    │               └────────────────────────────────────────────┘
    │
    │           (无物体间关系时, 5.2跳过, mainv2.py 行 1125/1127 兜底保存)
    │
    ▼
结束: 输出最终GLB路径 + pose_changes.json + coordinate_alignment.json
```

### GLB 文件数量 (按启用阶段)

| 启用阶段 | 生成文件 | 总数 |
|---------|---------|------|
| 默认 (无 stage4/5) | `final_scene_initial.glb`, `final_scene.glb` | 2 |
| 仅 `--enable_stage5` | + `final_scene_stage5_sp.glb`*, `final_scene_stage5.glb` | 4* |
| 仅 `--enable_stage4` | + `final_scene_stage4.glb` | 3 |
| `--enable_stage4 --enable_stage5` | + `final_scene_stage4.glb`, `final_scene_stage5_sp.glb`*, `final_scene_stage4_5.glb` | 5* |

> *`final_scene_stage5_sp.glb` 仅在有物体间支撑关系时生成。无物体间关系时 5.2 跳过，总数减 1。

### 各 GLB 详细说明

| GLB | 保存位置 | 保存时机 | 数据来源 | 坐标系 |
|-----|---------|---------|---------|--------|
| `final_scene_initial.glb` | mainv2.py 行 1045 | Stage3 完成后、基础精修前 | all_instances (Stage3原始) | y-up |
| `final_scene.glb` | mainv2.py 行 1072 | 基础精修 (floor/wall/embedded) 后 | all_instances (基础精修后) | y-up |
| `final_scene_stage4.glb` | mainv2.py 行 1101 | Stage4 视觉-空间对齐后 | all_instances (Stage4后) | y-up |
| `final_scene_stage5_sp.glb` | refine_inter_object_placement.py 行 1934 | Stage5 SP精修+穿模修复后 | all_instances (SP精修后) | y-up |
| `final_scene_stage5.glb` | mainv2.py 行 1127 | 仅Stage5、check_stability后 | all_instances (check_stability后) | y-up |
| `final_scene_stage4_5.glb` | mainv2.py 行 1125 | Stage4+Stage5全部后处理后 | all_instances (check_stability后) | y-up |

### GLB 传递关系 (谁根据谁生成)

```
all_instances (内存中的核心数据)
    │
    ├─ Stage3完成 ──► final_scene_initial.glb  (快照#1)
    │
    ├─ 基础精修 ────► final_scene.glb          (快照#2, 固定起点)
    │                 + all_instances.pkl      (pkl快照, 供独立后处理)
    │
    ├─ Stage4 ──────► final_scene_stage4.glb   (快照#3)
    │                 + all_instances_stage4.pkl
    │
    └─ Stage5 ──────► final_scene_stage5_sp.glb  (快照#4, 中间结果)
                      final_scene_stage5.glb      (快照#4最终, 无stage4)
                      final_scene_stage4_5.glb    (快照#5最终, 有stage4)
```

**关键点**:
- 每个 GLB 都是 `all_instances` 在某个时间点的**快照**
- `all_instances` 在内存中被各阶段**就地修改** (T矩阵变化)
- GLB 之间**不相互依赖** — 都直接从 `all_instances` 生成
- `all_instances.pkl` 是 `all_instances` 的 pickle 快照, 供 `run_post_pipeline.py` 独立重跑 Stage4/5

### pkl数据结构

```python
{
    'all_instances': all_instances,           # {category: [instance_info, ...]}
    'all_optimal_frame_ids': all_optimal_frame_ids,  # {category: [frame_id, ...]}
    'categories_and_relations': categories_and_relations,  # {category: relation_str}
    'walls_info': walls_info,                 # 墙壁几何信息
}
```

### --cleanup 参数 (未实现)

> **注意**: `--cleanup` 参数在 mainv2.py 代码中**并不存在**，该功能从未实现。以下为旧版设计描述，仅供参考。

**设计意图** (未实现): 启用 `--cleanup` 后，流水线结束时自动删除中间文件：

| 设计中会删除的文件 | 说明 |
|-----------|------|
| `all_instances.pkl` | 基础精修后pkl (382M+) |
| `all_instances_stage4.pkl` | Stage4后pkl |
| `color/` | VGGT输入帧 (160张jpg) |
| `depth/` | VGGT深度图 (160张png) |
| `extrinsics/` | 相机外参 |
| `keyframes/` | Stage1关键帧 |
| `optimal_frames/` | Stage3最优视角帧 |

**设计中保留的文件**: 所有 `.glb`, `.ply`, `.json`, `.log`, `intrinsic.txt`

### GLB坐标系

- **内部处理**: z-up (与`sp_refinement.py`、`geometry_utils.py`一致)
- **GLB输出**: y-up (trimesh标准)
- **变换**: 保存时执行 z-up → y-up 变换 (仅作用于临时 Scene 对象, 不回写 all_instances)

### Stage5 是否改变坐标系？

**不改变。** Stage5 的所有操作都是**单个物体的 T 矩阵微调**, 不存在全局坐标系变换:

| 操作 | 代码位置 | 类型 |
|------|---------|------|
| SP精修 (on_top/inside/against_side等) | refine_inter_object_placement.py L608-869 | 单个物体: 旋转对齐+z贴合 |
| `_align_upright` | refine_inter_object_placement.py L590-605 | 单个物体: 旋转对齐 (当前已禁用) |
| `resolve_penetrations` | refine_inter_object_placement.py L1180-1450 | 层级穿模修复: floor→大物体→小物体, 小物体z-only+跟随supporter xy |
| `check_stability` Phase 1-4 | refine_inter_object_placement.py L1281-1504 | 单个物体: 旋转对齐+z贴合+悬空修复 |

**唯一的全局坐标系变换发生在 Stage2** (`align_to_room_coordinate_system` + `align_vggt_predictions`), 将 VGGT 原始预测对齐到房间坐标系。Stage5 只在此基础上微调各物体的位姿。

**用户观察到"不同GLB之间z轴变化"的原因**: 不是坐标系变了, 而是各阶段精修改变了物体的 T 矩阵, 导致物体在场景中的朝向和位置不同。例如:
- `final_scene_initial.glb`: 物体可能严重倾斜 (Stage2对齐失败时)
- `final_scene.glb`: 基础精修做了旋转对齐, 物体更竖直
- `final_scene_stage5_sp.glb`: SP精修进一步调整了物体位置
- `final_scene_stage5.glb`: check_stability 确保物体稳定

---

## 4.1 位姿变化记录 `pose_changes.json`

`pose_changes.json` 记录完整的坐标变化历史，包括：坐标系对齐变换、相机外参变化、每个物体在各阶段的位置。结构如下：

```json
{
  "coordinate_alignment": {
    "alignment_stage": "stage1_strict",
    "R": [[...],[...],[...]],
    "t": [x, y, z],
    "method_detail": { ... },
    "extrinsics_before_first_frame": [[...],[...],[...],[...]],
    "extrinsics_after_first_frame": [[...],[...],[...],[...]],
    "n_frames": 160
  },
  "camera_extrinsics_after_alignment": [
    { "frame_id": 0, "extrinsic_aligned": [[...],[...],[...],[...]] },
    { "frame_id": 1, "extrinsic_aligned": [[...],[...],[...],[...]] }
  ],
  "objects": {
    "table_0": {
      "category": "table",
      "instance_idx": 0,
      "relation": "supported by floor",
      "stages": {
        "initial": { "T_matrix": [...], "position": [...], "bounds_min": [...], "bounds_max": [...], "center": [...] },
        "basic_refinement": { "T_matrix": [...], "position": [...], "delta_from_initial": [...] },
        "stage4": { "T_matrix": [...], "position": [...], "delta_from_basic_refinement": [...] },
        "stage5": { "T_matrix": [...], "position": [...], "delta_from_stage4": [...] }
      }
    }
  }
}
```

### `coordinate_alignment` 字段 (Stage2 坐标系对齐)

记录 `run_stage2()` 中四阶段对齐的完整信息（详见 §2 Stage 2）:

| 字段 | 说明 |
|------|------|
| `alignment_stage` | 实际成功的阶段: `stage1_strict` / `stage2_relaxed` / `stage3_large_plane` / `stage4_geocalib` |
| `R` | 3x3 旋转矩阵 (将 VGGT 原始坐标系对齐到房间坐标系) |
| `t` | 3 维平移向量 |
| `method_detail` | 对齐方法的详细信息 (如 GeoCalib 的 gravity_vec, floor_normal, n_inliers 等) |
| `extrinsics_before_first_frame` | 对齐前第一帧的相机外参 (4x4) |
| `extrinsics_after_first_frame` | 对齐后第一帧的相机外参 (4x4) |
| `n_frames` | 总帧数 |

**变换公式** (`align_vggt_predictions`):
```
extrinsics_new[:, :3, :3] = extrinsics_old[:, :3, :3] @ R.T
extrinsics_new[:, :3, 3]  = extrinsics_old[:, :3, 3] - (R_new @ t)
world_points_new = world_points_old @ R.T + t
```

### `camera_extrinsics_after_alignment` 字段 (相机外参变化)

逐帧记录对齐后的相机外参 (从 `extrinsics/` 目录读取)。每帧一个条目，含 `frame_id` 和 4x4 外参矩阵。

### `objects` 字段 (物体位姿变化)

每个物体在各阶段的 T 矩阵和位置:

**记录规则**:
- `initial`: Stage3 完成后、基础精修前
- `basic_refinement`: 基础精修 (floor/wall/embedded) 后
- `stage4`: 仅当启用 `--enable_stage4` 时记录
- `stage5`: 仅当启用 `--enable_stage5` 时记录
- `physics`: 仅当启用 `--enable_physics_validation` 时记录

**向后兼容**: 不启用 Stage4 时，`pose_changes.json` 中不会出现 `stage4` 键，代码通过 `args.enable_stage4` 条件判断，不会出现 KeyError。如果 `coordinate_alignment.json` 不存在（如旧版输出），`pose_changes.json` 回退为纯 objects 字典。

---

## 5. 动态物体检测

### 5.1 点云位移检测 (`geometry_utils.py`)

**两个信号**:

| 信号 | 计算方式 | 阈值 | 特点 |
|------|---------|------|------|
| 逐帧中值位移 | 相邻帧质心位移的中位数 | > 0.02m | 捕捉逐帧运动 |
| 全局位移 | 首20%帧 vs 末20%帧质心距离 | > 0.04m | 捕捉VGGT漂移掩盖的真实运动 |

**判定规则**: 任一信号超过阈值 → 动态 (OR逻辑)

**为什么需要全局位移？** VGGT的漂移是系统性的，逐帧位移小但累积大。真实运动物体如果移动缓慢，逐帧中值位移可能低于阈值，但首尾帧质心距离会显著偏大。

### 5.2 VLM辅助动态检测 (未接入 mainv2)

> **注意**: 旧版文档描述了 VLM 加权投票动态检测 (`--enable_vlm_dynamic`)，但该参数**在 mainv2.py 代码中并不存在**，该功能从未接入 mainv2 的命令行参数。当前 `run_stage3()` 中的动态检测仅使用 §5.1 描述的纯点云位移信号 (中值位移 + 全局位移)，不涉及 VLM 投票。

以下为旧版设计描述 (仅供参考，代码中未实现):

**加权投票设计** (未实现):

| 信号 | 权重 | 判定方式 |
|------|------|---------|
| 点云全局位移 | 0.6 | global_disp > 0.04m |
| VLM视觉判断 | 0.4 | 观察首末帧，判断物体是否移动 |

**设计中的最终判定**: 加权得分 > 0.5 → 动态

**VLM判断流程** (未实现):
```
1. 取物体首帧和末帧图像
2. VLM提示: "Is the highlighted object moving between these frames?"
3. 回答 "moving" → VLM动态信号
```

### 5.3 动态物体帧选择

**静态物体**: 选3D表面积最大的帧（最完整建模）

**动态物体**: 选运动前的最大可见帧
```
1. 检测运动起始点 (位移突增帧)
2. 在运动前的帧中选3D表面积最大的
3. 如果没有检测到运动起始点，用首20%帧搜索
```

### 5.4 动态物体点云清理 (运动残影剔除)

**问题**: 动态物体在运动, 其在其他帧的点云是"残影" (不同帧位置不同), 会污染下游:
- SP精修: supporter 的 top_z/bounds 被残影拉偏
- Stage4 ICP: 动态点云干扰点云配准
- 坐标对齐: floor/wall 检测被残影影响

**策略**: 动态物体**只保留首帧点云** (物体放置位置), 其余帧该实例 mask 区域的 world_points 置 NaN.

**代码** (`mainv2.py` 行 598-611):
```python
# 动态物体: 保留 first_visible_frame_id 的点云, 其余帧置 NaN
for im_entry in sorted_masks:
    fid_clean = im_entry['frame_id']
    if fid_clean == first_visible_frame_id:
        continue  # 保留首帧
    mask_clean = im_entry['mask']
    vggt_prediction_results['world_points'][fid_clean][mask_clean > 0] = np.nan
```

**效果**:
- 下游 `compute_surface_area_from_pointmap`: NaN 面积被 `(triangle_areas > 0)` 过滤 → 不影响
- `_estimate_floor_centroid`: 已用 `np.isfinite` 过滤 (如有) 或 bottom-percentile 自动排除 NaN
- Stage4 ICP: NaN 点不参与配准
- 动态物体首帧点云保留 → 用于位置确认和 SP 精修

**注意**: 此清理在坐标系对齐 (行 363) **之后**执行, 不影响已完成的对齐. 但影响下游 Stage4/Stage5.

---

## 6. SP精修逻辑

> 📌 **2026-08-25 更新**: SP 几何校验、深嵌投影法修复、穿模方向/步长修复等最新改动见 [stage5_refinement_analysis.md](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/docs/pipeline/stage5_refinement_analysis.md)。本章描述的是基础框架逻辑，仍适用。

### 6.1 核心原则

**"supported by table 的精修等价于 supported by floor，只是 z=0 换成 supporter_top_z"**

这意味着:
1. 旋转对齐: 让物体上方向对齐z轴 (竖直) — 和floor完全一样
2. z轴平移: 底面对齐到支撑面 — floor对齐到z=0，table对齐到table_top_z

### 6.2 `_align_upright` 统一旋转对齐

所有SP精修函数共用 `_align_upright(info)` 做旋转对齐:

```python
def _align_upright(info):
    """旋转对齐: 让物体上方向对齐z轴 (与 refine_supported_by_floor_object 的旋转逻辑一致)"""
    transform_matrix = info["T"].copy()
    upper_real_vector = np.array([0, 0, 1])
    upper_transformed_vector = transform_matrix[:3, 1]
    # ... align_vectors 对齐 ...
    info["T"] = transform_matrix
    return info
```

这与 `sp_refinement.py` 中 `refine_supported_by_floor_object` 的旋转逻辑完全一致。

### 6.3 五种放置策略的实现

#### on_top (最常用)

```
Step 1: _align_upright(supported_info)     — 旋转对齐 (和floor一样)
Step 2: z_offset = supporter_top_z - supported_bottom_z
Step 3: 沿z轴平移 z_offset                — 底面贴顶面 (和floor对齐z=0一样)
```

#### inside

```
Step 1: _align_upright(supported_info)     — 旋转对齐
Step 2: supported底面对齐到supporter内部30%高度
Step 3: 穿模检查 (穿出顶部/底部则修正)
```

#### against_side

```
Step 1: _align_upright(supported_info)     — 旋转对齐
Step 2: z轴底面贴地
Step 3: x/y轴移动到侧面刚好接触
Step 4: 穿模检查
```

#### hanging_below

```
Step 1: _align_upright(supported_info)     — 旋转对齐
Step 2: supported顶面对齐到supporter底面
```

#### leaning

```
调用 against_side 后，允许物体有倾斜角度
```

### 6.4 与 main.py 基础精修的关系

| main.py 基础精修 | mainv2 SP精修 | 关系 |
|-----------------|--------------|------|
| `refine_supported_by_floor_object` | `sp_refine_on_top` | 旋转对齐逻辑相同，z对齐目标不同 (z=0 vs supporter_top_z) |
| `refine_embedded_in_wall_object` | — | mainv2基础精修中执行，SP精修不重复 |
| `refine_attached_to_wall_object` | — | mainv2基础精修中执行，SP精修不重复 |

**Stage5不处理floor/wall物体**: 这些已在基础精修中处理完毕。

### 6.5 层级穿模修复 (`resolve_penetrations`)

**问题背景**: 旧版穿模修复对所有物体一视同仁, 独立调整每个穿模物体的 x/y/z。
但小物体 (如杯子、玩具) 放在大物体 (如桌子) 上时, 独立调整 x/y 会导致小物体
脱离支撑面, 视觉上不合理。

**核心思路**: 按场景图层级分层处理, 小物体的 x/y 跟随支撑物的调整。

#### 6.5.1 小物体定义 (基于场景图 parent 层级)

**小物体** = `relations_scene_graph.json` 中 `parent != 1` 的物体 (即父辈不是 floor)。

```
relations_scene_graph.json 结构:
{
  "scene_graph_objects": [
    {"id": 1, "category": "floor", "parent": 1},       # floor
    {"id": 2, "category": "table", "parent": 1},        # 大物体 (parent=floor)
    {"id": 3, "category": "bowl", "parent": 2},         # 小物体 (parent=table)
    {"id": 4, "category": "cup",  "parent": 2},         # 小物体 (parent=table)
    {"id": 5, "category": "toy",  "parent": 3},         # 小物体 (parent=bowl, 层级更深)
  ],
  "category_to_display_ids": {"floor": [1], "table": [2], "bowl": [3], "cup": [4], "toy": [5]}
}
```

- `parent == 1` → 大物体 (supporter), 自由调整 x/y/z
- `parent != 1` → 小物体 (supported), 只 z 轴移动 + 跟随 supporter x/y

#### 6.5.2 层级加载方式 (优先 A, 回退 B)

**方式 A (优先)**: 从 `relations_scene_graph.json` 加载 `scene_graph_objects` 的 parent 层级
- 构建 `display_to_inst` 映射: `display_id → (category, instance_idx)`
- 遍历 `scene_graph_objects`, 对 `parent != 1` 的物体:
  - 加入 `supporter_to_supported[(supporter_cat, supporter_idx)]`
  - 类别名加入 `supported_names` (用于 z-only 判定)
- **比字符串解析更可靠**, 直接使用 scene graph 的 parent ID

**方式 B (回退)**: 从 `refined_relations` 字符串解析 "supported by X"
- 当 `scene_dir` 未提供或文件不存在时使用
- 解析 "bowl_0 supported by table_0" → `supporter_to_supported[("table", 0)] = [("bowl", 0)]`

#### 6.5.3 修复流程 (floor → 大物体 → 小物体)

```
┌─────────────────────────────────────────────────────────┐
│ 1. 穿模检测 (AABB 预筛 + FCL 精确检测)                  │
│    遍历所有物体对 (i, j), 检测是否穿模                    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 2. 移动物体选择 (层级优先)                                │
│    优先级: 小物体(supported) > 大物体 > floor/wall(不动)   │
│    - 一方是 supported, 另一方不是 → 移 supported          │
│    - 一方是 floor/wall → 移另一方                         │
│    - 两方都 supported 或都不是 → 移中心位置较高者          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 3. 小物体 z-only 强制                                    │
│    if move_is_supported and sep_axis != 2:              │
│        sep_axis = 2  # 强制 z 轴移动, 不独立调 x/y        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 4. 分离余量 (按物体大小分级)                              │
│    both_small (双方都是小物体): pen_depth + 0.005m       │
│    max_size > 0.5m:               pen_depth + 0.10m    │
│    max_size > 0.3m:               pen_depth + 0.05m    │
│    其他:                           pen_depth + 0.01m    │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 5. 应用 T 矩阵 + 地面约束                                 │
│    translation_vec[sep_axis] = direction * sep_dist    │
│    new_T = translation_matrix(translation_vec) @ old_T │
│    if new_mesh.bounds[0, 2] < 0:  # 穿出地面             │
│        z_fix = -new_mesh.bounds[0, 2]                  │
│        new_T = translation_matrix([0,0,z_fix]) @ new_T │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 6. 层级传播 (大物体移动 → 小物体跟随 x/y)                  │
│    if not move_is_supported:  # 移的是大物体             │
│        xy_delta = new_pos[:2] - old_pos[:2]             │
│        for s_cat, s_idx in supporter_to_supported[...]: │
│            s_info["T"] = translation_matrix(xy_delta)   │
│                          @ s_info["T"]                  │
│            # 小物体跟随 supporter 的 x/y 移动             │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ 7. 迭代 (最多 max_iterations=8 次)                       │
│    重复 1-6 直到无穿模或达到最大迭代次数                   │
└─────────────────────────────────────────────────────────┘
```

#### 6.5.4 调用站点

| 位置 | 代码 | scene_dir 来源 |
|------|------|----------------|
| mainv2.py 行 1102-1104 | `resolve_penetrations(..., dry_run=True, scene_dir=args.output_path)` | `args.output_path` |
| run_post_pipeline.py 行 458-460 | `resolve_penetrations(..., scene_dir=scene_dir)` | 函数参数 |
| refine_inter_object_placement.py 行 2030-2032 | `resolve_penetrations(..., scene_dir=scene_dir)` | 函数参数 |

#### 6.5.5 Scene 7 特殊情况

Scene 7 的 `relations_scene_graph.json` 中所有 9 个物体 `parent` 都是 `1` (floor):
- `category_to_display_ids`: `{"block": [3], "duck": [4], "plate": [5], "toy": [6,7,8,9,10,11]}`
- `supporter_to_supported` 为空 (无层级)
- `supported_names` 为空 (无小物体)
- 所有物体一视同仁, 正常穿模修复 (无 z-only 限制, 无 xy 跟随)

这是正确行为: Scene 7 物体都直接放在地板上, 无层级关系。



---

## 7. 后处理管线

### `run_post_pipeline.py`

**目的**: 独立于mainv2，对已有输出目录执行stage4/5。

**文件自动发现**:
```
输入: 场景目录路径 (如 output_v2/hoi4d)
自动发现:
  ├── color/              → RGB帧
  ├── depth/              → 深度帧
  ├── extrinsics/         → 外参
  ├── optimal_frames/     → 最优帧图像
  ├── keyframes/          → 关键帧
  ├── intrinsic.txt       → 内参
  ├── final_scene.glb     → 基础精修GLB (固定起点)
  ├── all_instances.pkl   → 实例数据
  └── scene_*_stage1.json → Stage1 JSON
```

**GLB发现优先级**: `final_scene.glb` > `final_scene_base.glb` (兼容旧版)

**调用方式**:
```bash
# 只执行Stage4
python tools/run_post_pipeline.py output_v2/hoi4d --stage4

# 只执行Stage5
python tools/run_post_pipeline.py output_v2/hoi4d --stage5

# 同时执行Stage4+5
python tools/run_post_pipeline.py output_v2/hoi4d --stage4 --stage5
```

**数据流**:
```
all_instances.pkl (基础精修后快照)
  → Stage4 → final_scene_stage4.glb
  → Stage5 → final_scene_stage5.glb / final_scene_stage4_5.glb
```

---

## 8. 常见问题

### Q1: 为什么有不同数量的 GLB 文件？

GLB 数量取决于启用的阶段：

| 启用阶段 | 文件 | 数量 |
|---------|------|------|
| 默认 | `final_scene_initial.glb`, `final_scene.glb` | 2 |
| `--enable_stage5` | + `final_scene_stage5_sp.glb`, `final_scene_stage5.glb` | 4 |
| `--enable_stage4` | + `final_scene_stage4.glb` | 3 |
| `--enable_stage4 --enable_stage5` | + `final_scene_stage4.glb`, `final_scene_stage5_sp.glb`, `final_scene_stage4_5.glb` | 5 |

`final_scene_base.glb` 仍作为 `run_post_pipeline.py` 的固定起点保留，但 `mainv2.py` 默认流程中不再生成（由 `final_scene.glb` 替代）。

### Q2: 为什么所有物体都被判为静态？

**原因1**: `get_optimal_view_frame_id` 返回3个值，旧版mainv2只接收1个，导致数据类型错误。

**原因2**: 逐帧中值位移对VGGT漂移太敏感，真实运动物体的逐帧位移可能低于阈值。

**修复**: 
1. 正确接收3个返回值 `(optimal_frame_id, is_dynamic, motion_info)`
2. 新增全局位移检测 (首20%帧 vs 末20%帧)

### Q3: SP精修效果为什么不好？

**原因**: 旧版 `sp_refine_on_top` 重新实现了旋转对齐逻辑，与 `refine_supported_by_floor_object` 存在差异。

**修复**: 统一使用 `_align_upright()` 函数，逻辑与 `refine_supported_by_floor_object` 的旋转对齐完全一致。

**核心原则**: "supported by table = supported by floor，只是z=0换成supporter_top_z"

### Q4: mainv2不指定参数时和main.py一样吗？

**是的**，除了Stage1自动发现外，mainv2不指定 `--enable_stage4/5` 时:
- Stage2/3逻辑与main.py一致
- 基础精修逻辑与main.py一致 (额外修复了写回BUG)
- 只输出 `final_scene_base.glb` + `final_scene.glb` (内容相同)

### Q5: 为什么动态物体没有选到运动前的帧？

**旧逻辑**: 选第一个有效帧，但物体可能刚进入画面、遮挡严重。

**新逻辑**: 
1. 检测运动起始点 (位移突增帧)
2. 在运动前的帧中选3D表面积最大的
3. 如果全局位移大但逐帧未检测到运动起始点，额外搜索首1/3帧

### Q6: 精修后物体为什么还穿模？

**可能原因**:
1. 坐标系不对齐 — Stage5在z-up空间操作，如果GLB加载时未正确转换坐标系
2. 支撑物本身位置不准 — 如果支撑物(如table)位置偏移，on_top精修会把物体放到错误高度
3. AABB碰撞检测不够精确 — 对于非规则形状物体，AABB可能过于保守

### Q7: refine_supported_by_floor_object 的 0.3m 阈值是什么意思？

`refine_supported_by_floor_object` 只对 `z_min < 0.3m` 的物体做z轴平移。如果物体离地面超过0.3m（如桌子z_min=0.47m），精修会跳过z轴调整，只做旋转对齐。

这个阈值的设计意图是：只对"接近地面"的物体做吸附，避免把桌上物体错误吸附到地面。

但在SP精修中，`_align_upright` 只做旋转对齐，不做z轴判断，z轴平移由各策略函数自己控制。

### Q8: Stage1 能不能加快速度？用 vggt_omega 行吗？还是 VLM 判断慢？

**结论：VGGT 3D重建是绝对瓶颈（~45%时间），VLM推理是第二大耗时（~30%时间）。**

Stage1 各步骤时间占比：

| 步骤 | 描述 | 时间占比 | 说明 |
|------|------|---------|------|
| Step 0 | VGGT 3D重建 | **~45%** | 120帧 × VGGT-1B完整前向推理 |
| Step 3 | 第一次VLM：物体检测 | **~18%** | 12帧逐帧VLM推理 |
| Step 6 | SAM分割floor/wall | **~12%** | SAM3模型加载+推理 |
| Step 7 | 第二次VLM：关系判断 | **~12%** | 12帧逐帧VLM推理 |
| Step 5.5 | 点云补充检测 | **~6%** | DBSCAN + ≤5次VLM |
| 其他 | 采样/帧提取/去重/射线投射 | **~7%** | 纯CPU/IO操作 |

**能否用 vggt_omega 替代？**

当前 Stage1 硬编码使用原版 VGGT（`generate_scene_json_stage1.py` 没有 `--vggt_model` 参数）。VGGT-Omega 推理速度可能快10-20%，但存在关键风险：

- VGGT-Omega 的 `depth_conf` 分布太均匀，百分位阈值无法有效区分可靠/不可靠点，导致**点云缺块**
- Stage1 的射线投射和3D覆盖采样高度依赖点云质量
- 缺块的点云会导致射线投射失败率升高、采样覆盖率下降

**不建议替换**。更好的加速方案是减少 `--vggt_max_frames`（如从120降到80），能直接减少近一半VGGT推理时间。

**VLM调用详情**：Stage1 共3类VLM调用，总计最多29次：
- 物体检测(Step3): 12次，完整帧输入，短prompt
- 补充检测(Step5.5): ≤5次，裁剪图输入，短prompt
- 关系判断(Step7): ≤12次，完整帧输入，长prompt

### Q9: 坐标系问题——点云和GLB的原点/坐标轴会有明显偏差吗？为什么会出现倒立？找不到地面就用相机吗？

**1. 点云和GLB的原点/坐标轴**

点云 (`point_cloud.ply`) 和 GLB 确实有不同的坐标系：
- 点云在 Room World 坐标系下（z-up，地板z=0，场景中心为原点）
- GLB 在 glTF 标准坐标系下（y-up），导出时做了 `z-up → y-up` 变换

两者之间是确定的轴变换关系，不会产生"偏差"，只是表示方式不同。

**2. 为什么会出现倒立坐标轴？**

倒立的根因在 `get_plane_info()` 中法线方向的确定逻辑不够鲁棒：

```python
# geometry_utils.py:193
normal = -normal if normal[0] < 0 else normal  # 仅根据x分量正负翻转
```

这个逻辑只根据法线 x 分量的正负来决定翻转，而不是根据物理含义（地板法线应该朝上）。虽然后续有修正逻辑：

```python
# geometry_utils.py:255-257
floor_to_wall_vector = wall_plane_info['centroid'] - floor_plane_info['centroid']
if np.dot(floor_to_wall_vector, floor_normal) < 0:
    floor_normal = -floor_normal
```

但这个修正**依赖墙面质心在地板上方的假设**。如果墙面分割不准确，或墙面质心恰好在地板下方，修正就会失败，导致 z 轴朝下 → 整个场景倒立。

**3. 找不到地面时是否用相机？**

**不会。** `align_to_room_coordinate_system` 在找不到地面时返回恒等变换：

```python
# geometry_utils.py:240-241
if len(floor_plane_infos) == 0:
    return np.eye(3), np.zeros(3)  # 不做任何变换，保留VGGT原始坐标系
```

没有回退到相机坐标的机制。mainv2 新增了三级坐标系后备方案（Level 1 → Level 1.5 → Level 2+3），详见 §9.9。

### Q10: SAM3D放置的姿态对吗？动态物体帧生成和摆放位置分开了吗？

**1. SAM3D的T矩阵是否正确？**

T矩阵的变换链数学推导是正确的：

```
Local(SAM3D) ──y2z──→ Local(z-up) ──l2c──→ Camera(SAM3D) ──adjust──→ Camera(VGGT) ──ext⁻¹──→ World(VGGT)
```

但T矩阵的正确性**完全依赖于VGGT预测的extrinsic精度**。VGGT可能预测了正确的相对3D结构，但相机位置偏了，导致物体绝对位置偏移。

**注意**：`_align_upright` 当前已被**禁用**（`refine_inter_object_placement.py:576-591`），函数体直接 `return info`。禁用原因是VGGT重建的物体朝向可能本身就不准确，强制旋转可能导致更差的结果。

**2. 动态物体帧生成和摆放位置是否分开？**

**是的，已部分分离。** 代码位于 `mainv2.py:507-557`：

- **mesh生成帧**：`optimal_frame_id`（3D表面积最大的帧）→ SAM3D用该帧的image/mask/pointmap/extrinsic生成mesh
- **放置位置帧**：`first_visible_frame_id`（首次被SAM3检测到的帧）→ 动态物体的T矩阵平移分量被调整到该帧的质心位置

调整方式：
```python
offset = first_visible_centroid - optimal_centroid
T[:3, 3] += offset  # 只调整平移分量，旋转不变
```

**重要限制**：
1. 只修正了位置偏移，没有修正朝向偏移（如果动态物体在不同帧朝向不同）
2. offset基于world_points的质心差，VGGT在动态区域的点云质量较差，质心本身可能有误差
3. 理想情况下应该用first_visible_frame的extrinsic重新计算整个T矩阵，但这需要重新运行SAM3D（代价太大）

### Q11: mask的给出是去重前还是去重后的？同一位置生成多个物体是去重的问题吗？和遮挡有关系吗？

**1. 传给SAM3D的masks是去重后的**

流水线中的数据流：

```
segment_and_track → category_masks (去重前)
  ↓ self_category_deduplicate
deduplicated_category_masks (类内去重后)
  ↓ cross_category_deduplicate
deduplicated_all_masks (跨类去重后)
  ↓ merge_dynamic_splits_all (动态分裂+静态遮挡合并)
deduplicated_all_masks (全量去重后)
  ↓ 白名单过滤 (json_categories_set)
filtered_masks (只保留Stage1发现的类别)
  ↓ 传入 run_stage3 → generate_3d_asset_in_subprocess
SAM3D接收去重后的masks
  ↓ deduplicate_3d_assets
3D Mesh级别二次去重
```

SAM3D接收的是经过**三层去重+白名单过滤**后的masks。

**2. 同一位置生成多个物体——根因是遮挡导致的跟踪断裂**

因果链：

```
手部遮挡/物体移动
  → SAM3跟踪断裂（同一物体被分割为多个实例）
  → self_category_deduplicate 尝试合并（基于3D点云重叠率）
  → 如果物体移动了，原位置和新位置的3D点云重叠率可能不够高
  → 合并失败 → 同一物体保留多个实例
  → SAM3D为每个实例分别生成3D资产
  → 同一位置（或相近位置）出现多个物体
```

| 场景 | 根因 | 去重能否解决 |
|------|------|-------------|
| 静态物体被遮挡后重新出现 | SAM3跟踪断裂，但3D位置不变 | **能** — 重叠率高，类内去重会合并 |
| 动态物体移动后 | SAM3在新旧位置各检测一次 | **部分能** — 新旧位置重叠率≥0.3则合并 |
| 同一物体被分为不同类别 | SAM3分割不一致 | **能** — 跨类去重+白名单过滤会处理 |

**3. 和遮挡的关系**

遮挡是间接原因。直接原因是SAM3在遮挡发生时跟踪断裂，导致同一物体被拆分为多个实例。去重机制（3D点云重叠率）可以部分修复，但对于移动过的物体，新旧位置重叠率不够高时无法合并。

### Q12: SAM floor分割报错 "boolean index did not match" 是什么问题？

**根因**：SAM对原图(如1080×W)分割的mask尺寸与VGGT输出的pointmap尺寸(如518×W_vggt)不同。`get_plane_info(pointmap, mask)` 中 `pointmap[mask]` 直接用大mask索引小pointmap导致维度不匹配。

**修复**：新增 `_resize_mask_to_pointmap(mask, pointmap)` 辅助函数，在应用mask前将其resize到pointmap尺寸（PIL NEAREST插值保持bool语义）。修复了两处：
1. `get_plane_info()` 调用前
2. 点云补充检测排除floor区域时

### Q13: 点云补充检测提取点数过少（如25个点）是什么原因？

**根因**：置信度阈值比较用 `>` 严格大于。当某帧的中位数=最小值=1.000时，`conf_frame > 1.000` 排除了所有等于1.000的点（超过一半有效点）。

**修复**：
1. `>` 改为 `>=`，保留等于阈值的点
2. 新增保底逻辑：如果过滤后点数<100，自动降低到25%分位数

修复后，同样的帧（518个有效点，中位数=1.000）会保留约259个点而非25个。

---

## 9. 命令行参数

### 9.1 mainv2.py 参数完整参考

```bash
python mainv2.py --input_video <视频路径> [选项]
python mainv2.py --input_images <图片目录> [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_video` | str | None | 输入视频路径 (与--input_images二选一，必须指定其一) |
| `--input_images` | str | None | 输入图片目录路径 (与--input_video二选一，必须指定其一) |
| `--output_path` | str | None | 输出目录 (默认自动生成: `./output_v2/{video_stem}_{vggt_model}`) |
| `--vlm_checkpoint` | str | None | VLM模型路径 (默认自动查找: Qwen3.5-9B → Qwen2.5-VL-3B) |
| `--max_frames` | int | 160 | VGGT最大帧数 (Stage2, 同时传给Stage1的vggt_max_frames) |
| `--vggt_model` | str | "vggt" | 3D重建模型选择: `vggt` / `vggt_omega` / `vggt4d` |
| `--max_frames_stage1` | int | 10 | Stage1采样关键帧数 |
| `--enable_stage4` | flag | False | 启用Stage4视觉-空间对齐 |
| `--enable_stage5` | flag | False | 启用Stage5语义感知场景精修 |
| `--stage5_method` | str | "scene_graph" | Stage5.1关系推断方式: `scene_graph` / `per_object` |
| `--stage4_iterations` | int | 8 | Stage4 ICP迭代次数 |
| `--stage4_temporal_radius` | int | 5 | Stage4时序邻域半径 |
| `--stage4_use_mast3r` | flag | False | Stage4使用MASt3R匹配 (需要GPU) |
| `--enable_physics_validation` | flag | False | 启用物理仿真验证 (SAPIEN, 实验性功能) |
| `--physics_sim_steps` | int | 300 | 物理仿真检测步数 |

**已移除的参数**: `--category_path` (main.py必须指定，mainv2通过Stage1自动发现)

> **注意**: 旧版文档中提及的 `--enable_vlm_dynamic` 参数**在 mainv2.py 代码中并不存在**。VLM 辅助动态检测功能未接入 mainv2 的命令行参数；动态检测仅使用纯点云位移信号 (中值位移 + 全局位移)，详见 §5。

### 9.2 main.py 参数参考 (对比)

```bash
python main.py --input_video <视频路径> --category_path <JSON路径> [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_video` | str | `'./assets/example/hallway.mp4'` | 输入视频或图片目录路径 |
| `--output_path` | str | `'./outputs/hallway'` | 输出目录 |
| `--category_path` | str | `'./assets/example/hallway.json'` | 类别和关系JSON文件 (必须) |
| `--max_frames` | int | 160 | 从视频中处理的最大帧数 |

### 9.3 run_post_pipeline.py 参数

```bash
python tools/run_post_pipeline.py <scene_dir> [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scene_dir` | pos | — | 场景输出目录 (如 `output_v2/hoi4d_vggt_omega`) |
| `--stage4` | flag | False | 执行Stage4视觉-空间对齐 |
| `--stage5` | flag | False | 执行Stage5 (含5.1关系细化 + 5.2 SP精修) |
| `--vlm_checkpoint` | str | None | VLM模型路径 (Stage5.1需要) |
| `--stage4_iterations` | int | 8 | Stage4 ICP迭代次数 |
| `--stage4_temporal_radius` | int | 5 | Stage4时序邻域半径 |
| `--stage4_use_mast3r` | flag | False | Stage4使用MASt3R匹配 |

### 9.4 完整调用示例

#### 基础用法 (等价于main.py)

```bash
# main.py 方式 (需手动指定category_path)
python main.py \
    --input_video ./video.mp4 \
    --category_path ./video.json \
    --output_path ./outputs/video

# mainv2 等价方式 (Stage1自动发现，无需category_path)
python mainv2.py \
    --input_video ./video.mp4
# 输出: ./output_v2/video_vggt/final_scene.glb
```

#### 使用不同3D重建模型

```bash
# VGGT (默认，patch_size=14, image_resolution=518)
python mainv2.py --input_video ./video.mp4 --vggt_model vggt

# VGGT-Omega (patch_size=16, image_resolution=512)
python mainv2.py --input_video ./video.mp4 --vggt_model vggt_omega

# VGGT4D (patch_size=14, image_resolution=518)
python mainv2.py --input_video ./video.mp4 --vggt_model vggt4d
```

#### 启用Stage4 (视觉-空间对齐)

```bash
# 基本Stage4
python mainv2.py --input_video ./video.mp4 --enable_stage4

# Stage4 + MASt3R匹配 (更精确，需要GPU)
python mainv2.py --input_video ./video.mp4 --enable_stage4 --stage4_use_mast3r

# Stage4 自定义参数
python mainv2.py --input_video ./video.mp4 \
    --enable_stage4 \
    --stage4_iterations 12 \
    --stage4_temporal_radius 3 \
    --stage4_use_mast3r
```

#### 启用Stage5 (语义精修)

```bash
# 基本Stage5 (含5.1关系细化 + 5.2 SP精修)
python mainv2.py --input_video ./video.mp4 --enable_stage5

# Stage5 + 自定义VLM模型
python mainv2.py --input_video ./video.mp4 \
    --enable_stage5 \
    --vlm_checkpoint /path/to/Qwen3.5-9B
```

#### 全流水线 (Stage1-5)

```bash
# 完整流水线: VGGT-Omega + Stage4 + Stage5
python mainv2.py --input_video ./video.mp4 \
    --vggt_model vggt_omega \
    --enable_stage4 --stage4_use_mast3r \
    --enable_stage5

# 完整流水线 + 物理仿真验证
python mainv2.py --input_video ./video.mp4 \
    --vggt_model vggt_omega \
    --enable_stage4 --stage4_iterations 10 \
    --enable_stage5 \
    --enable_physics_validation
```

#### 使用图片目录代替视频

```bash
python mainv2.py --input_images ./my_images/
```

#### 自定义输出路径

```bash
python mainv2.py --input_video ./video.mp4 --output_path ./my_output
```

#### 自定义Stage1帧数和VGGT帧数

```bash
python mainv2.py --input_video ./video.mp4 \
    --max_frames_stage1 15 \
    --max_frames 200
```

#### 后处理管线 (独立调用Stage4/5)

```bash
# 对已有输出目录执行Stage4
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega --stage4

# 对已有输出目录执行Stage5
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega --stage5

# 同时执行Stage4+5
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega --stage4 --stage5

# 后处理 + 自定义参数
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega \
    --stage4 --stage4_iterations 12 --stage4_use_mast3r \
    --stage5 --vlm_checkpoint /path/to/Qwen3.5-9B
```

### 9.5 参数组合速查

| 场景 | 命令 |
|------|------|
| 最快运行 (等价main.py) | `--input_video video.mp4` |
| 更好的3D重建 | `--input_video video.mp4 --vggt_model vggt_omega` |
| 精确位姿对齐 | `--input_video video.mp4 --enable_stage4 --stage4_use_mast3r` |
| 物体间关系精修 | `--input_video video.mp4 --enable_stage5` |
| 最佳质量 | `--input_video video.mp4 --vggt_model vggt_omega --enable_stage4 --stage4_use_mast3r --enable_stage5` |
| 物理验证 | `--input_video video.mp4 --enable_stage5 --enable_physics_validation` |
| 图片输入 | `--input_images ./images/` |

---

## 10. GeoCalib 重力方向判断 (Stage 4 对齐)

### 10.1 为什么需要 GeoCalib？

当 SAM3 找不到 floor mask（桌面场景、无可见地面）时，Stage 1-3 全部失败。GeoCalib 从**图像本身**估计相机的重力方向，不需要任何 floor/wall 分割，是最后的 z 轴对齐 fallback。

### 10.2 GeoCalib 的 gravity 约定

GeoCalib (`geocalib/gravity.py`) 的 `Gravity.vec3d` 属性返回的是**重力方向向量**——即重力把物体往下拉的方向，指向地心（DOWN）。

**源码验证** (`geocalib/gravity.py` 行 31-40):
```python
@classmethod
def from_rp(cls, roll, pitch):
    sr, cr = torch.sin(roll), torch.cos(roll)
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    return cls(torch.stack([-sr * cp, -cr * cp, sp], dim=-1))
    # roll=0, pitch=0 → [0, -1, 0]  ← 指向 y 轴负方向 (DOWN)
```

### 10.3 关键 bug 修复: gravity 相机→世界坐标变换

**原 bug 1** (`geometry_utils.py` `align_via_geocalib`):
```python
# 错误: 把 gravity (DOWN) 直接当 floor_normal (应 UP)
floor_normal = final_vec.numpy()
floor_normal = floor_normal / np.linalg.norm(floor_normal)
```

**修复 1** (方向取反):
```python
gravity_vec = final_vec.numpy()  # [3], points DOWN
floor_normal = -gravity_vec / np.linalg.norm(gravity_vec)  # negate → UP
```

**原 bug 2** (scene 15 的 R[2,2]=0.2259 根因):
```python
# 错误: GeoCalib 返回的 gravity 是相机坐标系下的向量, 直接当世界坐标用
# 当相机 z 轴不与世界 z 轴对齐时, floor_normal 偏离竖直 → R[2,2] 偏小
vec = grav.vec3d.squeeze(0).cpu()  # 相机坐标系
gravity_vecs.append(vec)  # 直接在相机坐标系平均
final_vec = spherical_mean(vecs)   # 仍是相机坐标系
floor_normal = -final_vec          # 相机坐标系的 "UP", 非世界坐标系
```

**修复 2** (相机→世界坐标变换):
```python
# 每帧 GeoCalib gravity (相机坐标系, DOWN) → 用 extrinsics 变换到世界坐标系
R_w2c = extrinsics[idx, :3, :3]          # (3,3) world→camera
grav_world = R_w2c.T @ grav_cam           # camera→world: R_c2w = R_w2c.T
gravity_world_vecs.append(grav_world)     # 世界坐标系
# 然后在世界坐标系做球面平均 + MAD 过滤
final_vec = spherical_mean(inlier_vecs, w=inlier_confs)  # 世界坐标系
floor_normal = -final_vec / norm          # 世界坐标系的 UP
```

**为什么有效**: gravity 是世界坐标系常量 (永远指向地心). 每帧相机坐标系下的 gravity 不同 (因相机朝向不同), 但变换到世界坐标系后应该一致. 在世界坐标系平均才是正确的, 在相机坐标系平均会得到一个无意义的方向.

**scene 15 验证**: 修复前 R[2,2]=0.2259 (z 轴偏 77°), 修复后预期 R[2,2]≈1.0.

#### 10.3.1 关键 bug 修复: camera_positions 提取 (w2c 的 t 不是相机位置)

**问题**: Scene 7 的 z 轴方向正好相反 (z 轴朝下)。根因是 `_orient_floor_normal` 的
fallback 分支用 `camera_positions` 判断 "上方", 但 `camera_positions` 提取错误。

**原 bug** (`geometry_utils.py` `align_via_geocalib` 行 689):
```python
# 错误: VGGT extrinsics 是 w2c (world→camera): p_cam = R @ p_world + t
# extrinsics[:, :3, 3] 给出的是 w2c 的 t (平移), 不是相机位置!
# 相机位置 = -R.T @ t (w2c 的逆变换)
camera_positions = extrinsics[:, :3, 3]  # 错: 这是 t, 不是相机位置
```

**修复** (正确提取相机位置):
```python
# VGGT extrinsics 是 w2c (world→camera): p_cam = R @ p_world + t
# 相机在世界坐标的位置 = -R.T @ t (不是 t 本身)
R_w2c_all = extrinsics[:, :3, :3]  # (N, 3, 3)
t_w2c = extrinsics[:, :3, 3]       # (N, 3)
camera_positions = -np.einsum('nji,nj->ni', R_w2c_all, t_w2c)  # (N, 3) cam_pos = -R.T @ t
```

**证据**: `test_scannet.py:260` 用同样的公式提取相机位置:
```python
camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]  # 确认 VGGT extrinsics 是 w2c
```

**影响**: `_orient_floor_normal` 在 `floor_centroid ≈ all_centroid` (退化情况) 时,
用 `mean_cam - floor_centroid` 判断 "上方"。如果 `camera_positions` 是 t 而非相机位置,
方向判断会出错, 导致 floor_normal 朝下 → z 轴方向反转。

**坐标系变换记录到 json**: `coordinate_alignment.json` 现在包含:
```json
{
  "extrinsics_convention": "w2c (world→camera): p_cam = R @ p_world + t",
  "camera_position_formula": "cam_pos = -R_w2c.T @ t_w2c (不是 t 本身)",
  "method_detail": {
    "gravity_transform": "grav_world = R_w2c.T @ grav_cam (= R_c2w @ grav_cam, camera→world)",
    "camera_position_transform": "cam_pos = -R_w2c.T @ t_w2c (w2c 的 t 不是相机位置)",
    "floor_centroid": [x, y, z]
  }
}
```



### 10.4 经验验证

对 5 个 basic_pick_place 视频统计对齐后 z>0 的点占比:

| 方案 | 平均 z>0 占比 | 判定 |
|------|-------------|------|
| `floor_normal = gravity` (原 bug) | 38.1% | z 轴朝下 ❌ |
| `floor_normal = -gravity` (修复) | 61.8% | z 轴朝上 ✅ |

### 10.5 GeoCalib 权重

- 模型: `pinhole` (针孔相机模型)
- 大小: 111MB
- 下载源: `https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar`
- 本机缓存: `/mnt/data_8THDD/lza/.cache/torch/hub/geocalib/pinhole.tar`
- 无网络时 Stage 4 不可用（会跳过，但 Stage 1-3 可能已成功）

### 10.6 完整对齐流程 (align_via_geocalib)

```
1. 从视频均匀采样 max_frames 帧 (默认8帧)
2. 对每帧用 GeoCalib 估计 gravity 向量 (相机坐标系, 指向 DOWN)
3. 用 extrinsics[idx][:3,:3].T 将每帧 gravity 从相机坐标变换到世界坐标
4. MAD 过滤离群帧 (threshold=3.0, 在世界坐标系下)
5. 置信度加权球面平均得到 final_vec (世界坐标系 gravity, 指向 DOWN)
6. floor_normal = -final_vec (取反, 指向世界坐标系 UP)
7. floor_centroid = _estimate_floor_centroid (bottom 10% 点的质心)
8. _orient_floor_normal: 用相机位置作为 "上方" 参考, 确保方向正确
9. 用 _build_R_t_from_floor 构造 R, t:
   - R[2,:] = floor_normal (z 轴 = UP)
   - R[0,:], R[1,:] = PCA 确定水平方向
   - t[2] = -rotated_floor_centroid[2] (floor 放在 z=0)
   - t[:2] = -bbox_center[:2] (xy 居中)
10. 质量检查: 若 |R[2,2]| < 0.5 (偏离竖直 > 60°), 返回 identity (对齐失败)
11. 返回 R, t, info (含 gravity_world, floor_normal, n_inliers 等)
```

### 10.6.1 中心点选择机制 (所有阶段通用)

坐标系对齐后的平移向量 `t` 由两部分决定:

| 分量 | 选择方式 | 代码位置 | 说明 |
|------|---------|---------|------|
| `t[2]` (z 轴) | `floor_centroid` 的旋转后 z 坐标 | `_build_R_t_from_floor` L318-320 | 将 floor 放到 z=0 |
| `t[:2]` (xy) | 旋转后点云 bbox 中心 `(min+max)/2` | `_build_R_t_from_floor` L321-326 | xy 居中到原点 |

**各阶段的 floor_centroid 来源**:

| 阶段 | 函数 | floor_centroid 来源 | 准确度 |
|------|------|---------------------|--------|
| Stage 1 | `align_to_room_coordinate_system` | `floor_plane_info['centroid']` (PCA 拟合的 floor 平面质心) | ★★★★★ |
| Stage 2 | `align_via_objects` | `floor_plane_info['centroid']` (同 Stage 1) | ★★★★★ |
| Stage 3 | `align_via_large_plane` | `floor_plane_info['centroid']` (大平面 PCA 质心) | ★★★★☆ |
| Stage 4 | `align_via_geocalib` | `_estimate_floor_centroid` (bottom 10% 点的质心) | ★★★☆☆ |

**Stage 4 的 floor_centroid 改进**:

- **原实现**: `floor_centroid = np.mean(all_points, axis=0)` — 用整个点云质心, z=0 平面在场景垂直中心, 不是真实 floor
- **修复后**: `floor_centroid = _estimate_floor_centroid(all_points, floor_normal, bottom_percentile=10)` — 用 floor_normal 方向上最低 10% 点的质心, 更接近真实 floor

**xy 中心 (bbox center) 的已知问题**:
- 用 `(min_coords + max_coords) / 2` 计算, 对离群点敏感
- 如果点云有离群点 (如手部云团), bbox 会被拉大, xy 中心偏移
- 这是所有阶段共有的问题, 暂未修复 (需要 percentile bbox 或 RANSAC)

### 10.7 四阶段级联测试结果 (basic_pick_place)

| 视频 | walls | floors | S1 | S2 | S3 | S4(GeoCalib) | 首个成功 |
|------|-------|--------|----|----|----|----|---------|
| 15.mp4 | 23 | 0 | ❌ | ❌ | ❌ | ✅ | stage4 |
| 109.mp4 | 15 | 0 | ❌ | ❌ | ❌ | ✅ | stage4 |
| 224.mp4 | 1 | 0 | ❌ | ❌ | ❌ | ✅ | stage4 |
| 210.mp4 | 24 | 6 | ✅ | ✅ | ✅ | ✅ | stage1 |
| 200.mp4 | 4 | 1 | ✅ | ✅ | ✅ | ✅ | stage1 |

**结论**: 桌面场景（无可见地面）3/5 视频仅靠 GeoCalib 才成功对齐 z 轴。GeoCalib 是不可或缺的 fallback。

### 10.8 Scene 15 问题分析与修复 (2026-06-26)

**Scene 15 日志** (`output_v2/15_vggt_omega/mainv2_20260625_172426.log`):

| 问题 | 现象 | 根因 | 修复 |
|------|------|------|------|
| z 轴严重偏斜 | `R[2,2]=0.2259` (偏 77°) | GeoCalib gravity 在相机坐标系平均, 未变换到世界坐标 | §10.3 修复 2: `grav_world = R_w2c.T @ grav_cam` |
| z=0 平面位置错误 | floor_centroid 用 `np.mean(all_points)` | 整个点云质心 ≠ 真实 floor | `_estimate_floor_centroid` (bottom 10%) |
| z 轴方向不确定 | `_orient_floor_normal` 退化 | `floor_centroid ≈ all_centroid` 时方向判断失效 | 新增 `camera_positions` 参数 |
| 无质量检查 | R[2,2]=0.2259 仍报 "完成" | 没有 |R[2,2]| 阈值 | `abs(R[2,2]) < 0.5` 时返回 identity |
| toy 过度分割 | 14 个原始 toy 实例 | SAM3 对 "toy" 类别过度分割 | 跨类去重 + 同位置检测 (§10.9) |
| duck/plate 误判动态 | offset 仅 2-3cm | VGGT 漂移导致位置偏移 | 位置调整逻辑下游问题, 暂未修复 |
| toy_4 跟踪丢失 | `valid_frames=5/51` | SAM3 在遮挡后丢失跟踪 | 需要时序连续性去重 (未来工作) |

### 10.9 去重架构重构: 三层流水线 (2026-07-13)

**问题**: 旧的 `cross_category_deduplicate` 同时处理同类和跨类实例，导致静态同类实例被错误合并；CLIP 外观特征在室内场景区分度极低。

**修复**: 采用三层流水线架构：

```
self_category_deduplicate (同类)   → overlap_ratio ≥ 0.3
         ↓
cross_category_deduplicate (跨类)  → 同类跳过，overlap_ratio ≥ 0.3
         ↓
merge_dynamic_splits_all (全量)    → 静态重叠 + 静态遮挡(子集关系) + 动态分裂(衔接处overlap)
```

**`merge_dynamic_splits_all` 合并条件:**
1. 静态重叠: ov1 ≥ 0.3 or ov2 ≥ 0.3 → 合并
2. 静态遮挡: 单向 overlap ≥ 0.5（子集关系）AND 子集方帧数更少 → 合并
3. 动态分裂: lifespan_ratio < 0.5 AND time_gap ≤ 10 AND 衔接帧 overlap ≥ 0.2 → 合并

**删除**: CLIP 模型加载代码、`_merge_dynamic_splits` 旧函数

---

## 11. 修改记录

| 日期 | 修改 | 文件 |
|------|------|------|
| 2026-05-29 | mainv2.py 初版 | mainv2.py |
| 2026-05-30 | 修复坐标系变换、GLB覆盖逻辑 | run_post_pipeline.py, mainv2.py |
| 2026-05-31 | 鲁棒运动检测+日志标记 | geometry_utils.py, mainv2.py, main.py |
| 2026-05-31 | protected_categories防止跨类合并 | sg_deduplication.py, mainv2.py |
| 2026-06-01 | SP精修复用refine_supported_by_floor_object | refine_inter_object_placement.py |
| 2026-06-02 | 修复get_optimal_view_frame_id返回值处理 | mainv2.py |
| 2026-06-02 | 新增全局位移检测+VLM加权投票 | geometry_utils.py, mainv2.py |
| 2026-06-02 | 统一_align_upright旋转对齐 | refine_inter_object_placement.py |
| 2026-06-02 | GLB输出简化: base(固定起点) + final(最新) | mainv2.py, run_post_pipeline.py |
| 2026-06-02 | pkl格式扩展为字典 | mainv2.py, run_post_pipeline.py |
| 2026-06-03 | 补充遗漏差异: --category_path移除、--input_images互斥、输出路径自动生成、VLM自动查找、Stage5逻辑重复等 | mainv2_technical_doc.md |
| 2026-06-03 | 补充完整参数调用示例: 各模型/Stage组合/后处理管线/参数组合速查 | mainv2_technical_doc.md |
| 2026-06-18 | 默认帧数更新: max_frames 160→120, max_frames_stage1 10→12 | mainv2.py |
| 2026-06-18 | 修复SAM floor分割mask维度不匹配: 新增_resize_mask_to_pointmap() | generate_scene_json_stage1.py |
| 2026-06-18 | 修复点云补充检测阈值: >改>=, 点数<100自动降低阈值 | generate_scene_json_stage1.py |
| 2026-06-18 | 更新技术文档: 同步默认值、更新protected_categories描述、新增Q8-Q13 | mainv2_technical_doc.md |
| 2026-06-25 | 四阶段坐标系对齐接入 mainv2 (Stage1→Stage2→Stage3→Stage4 GeoCalib) | mainv2.py, geometry_utils.py |
| 2026-06-25 | 修复 GeoCalib gravity 方向 bug: gravity 指向 DOWN, floor_normal 需取反 -gravity | geometry_utils.py |
| 2026-06-25 | 修复 logger 未定义 bug: run_stage2 中改用 print | mainv2.py |
| 2026-06-25 | 修复 id_scene.png 框选错位: create_id_labeled_image 优先选 best_frame_id 的 mask | infer_relations_scene_graph.py |
| 2026-06-25 | 修复 supporter 实例索引解析 bug: _find_supporter_info 解析 "table_1" 返回 table[1] | refine_inter_object_placement.py |
| 2026-06-25 | 修复家具间穿模被跳过: "supported by floor" 的物体不再被误判为结构元素 | refine_inter_object_placement.py |
| 2026-06-25 | pose_changes.json 扩展: 新增 coordinate_alignment + camera_extrinsics_after_alignment 字段 | mainv2.py |
| 2026-06-25 | 文档新增 §10 GeoCalib 重力方向判断, 更新 §2 四阶段对齐, §4.1 pose_changes 结构 | mainv2_technical_doc.md |
| 2026-06-25 | 文档新增 §1 "main() 完整执行流程" 章节: 8 个步骤+行号映射+完整数据流图 | mainv2_technical_doc.md |
| 2026-06-25 | 文档修正 §9.1 参数表: 移除不存在的 --enable_vlm_dynamic, 新增 --stage5_method/--enable_physics_validation/--physics_sim_steps, 修正 --max_frames 默认值 120→160, --max_frames_stage1 12→10 | mainv2_technical_doc.md |
| 2026-06-25 | 文档修正 §5.2 VLM辅助动态检测: 标注 "未接入 mainv2", --enable_vlm_dynamic 代码中不存在 | mainv2_technical_doc.md |
| 2026-06-25 | 文档修正 §4 --cleanup 参数: 标注 "未实现", 代码中不存在 | mainv2_technical_doc.md |
| 2026-06-25 | 文档修正 §1/§8 引用: 移除 --enable_vlm_dynamic 错误引用, 更新架构差异表 | mainv2_technical_doc.md |
| 2026-06-26 | 修复 GeoCalib gravity 相机→世界坐标变换: `grav_world = R_w2c.T @ grav_cam` (原直接用相机坐标 gravity, 导致 R[2,2]=0.2259) | geometry_utils.py |
| 2026-06-26 | GeoCalib floor_centroid 改用 bottom 10% 点质心 (原用整个点云质心, z=0 平面位置错误) | geometry_utils.py |
| 2026-06-26 | 新增 _estimate_floor_centroid 函数 + _orient_floor_normal 新增 camera_positions 参数 (相机位置作为 "上方" 参考) | geometry_utils.py |
| 2026-06-26 | GeoCalib 新增 R[2,2] 质量检查: abs(R[2,2]) < 0.5 时返回 identity (对齐失败) | geometry_utils.py |
| 2026-06-26 | align_via_geocalib 新增 extrinsics 参数 (必需), 更新 mainv2.py + test 调用 | mainv2.py, test_alignment_basic_pick_place.py |
| 2026-06-26 | cross_category_deduplicate 新增同位置检测: 质心距离 < 0.03m 时降低 overlap 阈值 (×0.5) | sg_deduplication.py |
| 2026-07-13 | 去重架构重构: 三层流水线 (同类去重 → 跨类去重 → 全量动态分裂合并)，删除 CLIP，新增 merge_dynamic_splits_all，新增 stage2_pre/post_dedup.pkl 保存 | sg_deduplication.py, mainv2.py |
| 2026-06-26 | 文档更新 §10.3 (gravity 相机→世界变换), §10.6 (新流程+质量检查), 新增 §10.6.1 (中心点选择), §10.8 (scene15分析), §10.9 (同位置去重) | mainv2_technical_doc.md |
| 2026-06-26 | 修复 camera_positions 提取 bug: w2c 的 t 不是相机位置, 改用 `-R.T @ t` (Scene 7 z 轴方向反转的根因) | geometry_utils.py |
| 2026-06-26 | coordinate_alignment.json 新增 extrinsics_convention + camera_position_formula 字段, GeoCalib 返回值新增 gravity_transform/camera_position_transform/floor_centroid | mainv2.py, geometry_utils.py |
| 2026-06-26 | resolve_penetrations 新增 scene_dir 参数, 优先从 relations_scene_graph.json 加载 parent 层级确定大小物体 (回退到字符串解析) | refine_inter_object_placement.py |
| 2026-06-26 | 更新 resolve_penetrations 3 个调用站点传入 scene_dir: mainv2.py, run_post_pipeline.py, refine_inter_object_placement.py | 多文件 |
| 2026-06-26 | 文档新增 §6.5 层级穿模修复 (小物体 z-only + 跟随 supporter xy), §10.3.1 camera_positions bug 修复 | mainv2_technical_doc.md |
