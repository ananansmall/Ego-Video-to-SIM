# ReplicateAnyScene 问题与解答汇总

本文档记录项目开发过程中的问题与解答。

---

## 一、LLM/VLM 调用

### Q1: mainv2.py 中大模型是如何调用的？用什么函数、怎么输入、输出什么、prompt 是什么？有哪些模型？

**回答**:

**核心结论**：mainv2.py 里没有 OpenAI API 式的 LLM 调用，全部"大模型"调用都是**本地 Qwen 视觉语言模型 (VLM)**，通过 `transformers` 加载，核心推理函数是 `src/vlm_logger.py::vlm_inference()`。

#### 1. 涉及哪些模型

| 类型 | 模型 | 路径/来源 | 用途 |
|---|---|---|---|
| 主 VLM | Qwen3.5-9B | `/mnt/data/lza/models/Qwen3.5-9B` | 物体检测/关系判断/材质识别/坐标对齐 |
| 备选 VLM | Qwen2.5-VL-3B-Instruct | `/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct` | 自动回退（见 `mainv2.py::_resolve_vlm_checkpoint`） |
| 可选辅助 VLM | 高精度模型（如 32B） | 用户指定 | Stage1 多 VLM 联合检测补全 |
| 非 LLM | VGGT / VGGT-Omega / VGGT4D | `src/models.py` | 3D 重建（点云/深度/外参） |
| 非 LLM | SAM3（image/video） | `src/models.py` | 分割 floor/wall/实例 mask |
| 非 LLM | CLIP | `tools/generate_scene_json_stage1.py` | 检测结果同义词/位置去重 |
| 非 LLM | SAM3D | `src/instance_generation.py` | 3D 资产生成（非 LLM） |

#### 2. 统一推理函数

`src/vlm_logger.py::vlm_inference(image, model, processor, prompt, max_new_tokens=1024, temperature=0.0, do_sample=False, caller="unknown")`

- **输入**：1 张 PIL 图像 + 1 段 prompt 文本（可选 system，当前未用）
- **流程**：构建 `messages=[{"role":"user","content":[{"type":"image"...},{"type":"text"...}]}]` → `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True, enable_thinking=False)` → `model.generate(max_new_tokens, pad_token_id=eos, do_sample=False)` → `batch_decode` 只取新生成部分
- **输出**：纯文本，各调用方再用正则/JSON 解析
- **日志**：每次调用都追加写入 `{output_path}/vlm_conversation.jsonl`（prompt + response + caller）

#### 3. mainv2 流程中各调用点

| 阶段 | 文件/函数 | prompt | 输入 | 输出 |
|---|---|---|---|---|
| Stage1 物体检测 | `tools/generate_scene_json_stage1.py` `VLM_DETECT_PROMPT` | "List all visible objects in this image. Output JSON only..." | 每帧图像 | `{"objects":[{"name":"cup"}]}` |
| Stage1 补充检测 | 同上 `SUPPLEMENTARY_VLM_PROMPT` | "What is the large object in the center of this cropped image?" | 点云裁剪图 | 物体名 |
| Stage1 关系判断 | 同上 `_build_relationship_prompt()` | 4 种关系 + 物理常识规则 | 物体名列表 + floor/wall 可见性 | `{"relationships":{"obj":"supported by floor"}}` |
| Stage2.5 地面参考点 | `src/object_segmentation.py` `detect_floor_reference_points_with_vlm` | "Identify N representative points on the floor/ground..." | 图像 | 归一化坐标 JSON `[[x,y],...]` |
| Stage5.1 场景图 | `tools/infer_relations_scene_graph.py` `infer_relations_from_labeled_image` | SimRecon 风格，ID 标注图 + 物理常识 | 标注图 | `{"objects":[{"instance","category","relation","parent"}]}` |
| Stage5.1 旧方式 | `tools/refine_other_objects_relations.py` `build_refinement_prompt` | 目标物体支撑关系 | 图像 | `{"relationships":{...}}` |
| Stage5.2 放置策略 | `tools/refine_inter_object_placement.py` `build_placement_prompt` | on_top/inside/against_side/hanging_below/leaning | 图像 + 支撑对 | `{"strategy":"on_top"}` |
| Stage5.2 精修程度 | 同上 `_judge_refinement_degree_with_vlm` | major/minor/correct 三选一 | 图像 | 一个词 |
| Stage6 材质 | `tools/material_annotator.py` `MATERIAL_PROMPT_TEMPLATE` / `MATERIAL_PROMPT_SINGLE` | 材质识别 | 标注图（整体/单物体） | `{"materials":{...}}` |
| Stage6.1 URDF 物理属性 | `src/stage61_urdf_generator.py` + `tools/qwen_client.py::QwenClient.query()` | EmbodiedGen PROMPT_TEMPLATE（category/description/pose/height/weight/摩擦系数） | 4 视图渲染图 | 标准格式文本 |

#### 4. 特殊说明

- **QwenClient**（`tools/qwen_client.py`）：是另一套 VLM 适配器，接口与 GPTclient 的 `query()` 兼容，仅用于 Stage6.1 URDF 物理属性生成，目前是独立脚本 `src/stage61_urdf_generator.py`，尚未接入 mainv2（`docs/design/stage61_urdf_plan.md` 标记为"待做"）。
- **mainv2 不直接调 VLM 的位置**：Stage1 通过 subprocess 调用 `tools.generate_scene_json_stage1`；Stage 5/6 在函数内 `from tools.xxx import ...` 动态导入并传入 `vlm_checkpoint`，有的还支持 `preloaded_vlm` 复用已加载模型。
- **已知隐患**：`mainv2.py::run_stage2` 第 383/401 行调用 `_load_vlm_model(vlm_checkpoint)`，但该函数在整个仓库都没有定义（仅在对齐失败的回退路径触发），且 `run_stage2` 签名里没有 `vlm_checkpoint` 参数，触发时会 `NameError`。

## 二、模型调用全流程（VGGT-Omega / SAM3 / SAM3D + 两次 VLM）

### Q2: VGGT-Omega、SAM3、SAM3D 这些模型的调用流程是怎样的？从 0 开始怎么走？加上 VLM 的两处调用呢？

**回答**: 输入是一段第一人称/手持视频。整条流水线有 3 个阶段用到这些模型，两次 VLM 调用都在 Stage1。

#### 阶段总览

| 阶段 | 主进程/子进程 | 用到的模型 |
|---|---|---|
| Stage1 物体发现 | subprocess `tools/generate_scene_json_stage1.py` | **原版 VGGT**（非 Omega）+ **VLM ×2** + SAM3(image) + CLIP |
| Stage2 3D重建+分割 | mainv2 进程内 | **VGGT / VGGT-Omega / VGGT4D**（三选一）+ SAM3(image) + SAM3(video) + CLIP |
| Stage3 资产生成 | SAM3D 子进程（spawn） | **SAM3D**（Meta SAM 3D Objects，`InferencePipelinePointMap`） |

#### Stage1（发现物体，`tools/generate_scene_json_stage1.py::main()`）

1. **Step 0 — VGGT 3D 重建**：`load_video_frames(视频, 120帧)` → `load_vggt_model()` → `vggt_predict()` → 得到 `extrinsics/world_points/world_points_conf/colors/depths/intrinsic`。用第 0 帧点云做后续采样。
2. **Step 1 — SimRecon 采样**：`extract_frames_by_vggt_sampling()` 把点云体素化 → `maximum_coverage_sampling()` 贪心选出 K=12 个视角覆盖最大的关键帧。
3. **Step 2**：`extract_specific_frames()` 用 ffmpeg 提取关键帧图片。
4. **Step 3 — 第一次 VLM 调用（物体检测）**：加载 VLM（`AutoProcessor` + `AutoModelForImageTextToText`）→ `detect_objects_in_frames()` 对每帧调 `vlm_inference(img, VLM_DETECT_PROMPT)` → JSON 物体名列表。可选辅助 VLM：主模型独立类别 < 8 时加载更高精度模型再检测一遍并合并。
5. **Step 4 — 射线投射**：`build_object_instances()` 把 VLM 像素位置用 `pixel_to_3d_position()`（深度/pointmap 反投影）映射到 3D。
6. **Step 5 — 去重**：`deduplicate_objects()`（SYNONYM_MAP + CLIP 语义匹配）。
7. **Step 5.5 — 点云补充检测**：`supplementary_detect_from_pointcloud()` 分析 VGGT 点云中未被覆盖的聚类 → 裁剪图 → VLM 用 `SUPPLEMENTARY_VLM_PROMPT` 识别（这是 VLM 的第 1.5 处调用）。
8. **Step 6 — SAM3(image) 分割 floor/wall**：`segment_floor_and_wall_sam()` 用多组文本 prompt（10 个 floor + 8 个 wall 变体）跑 `sam3_image_model.set_image → set_text_prompt → masks`。
9. **Step 7 — 第二次 VLM 调用（关系判断）**：`vlm_judge_relationships()` 用 `_build_relationship_prompt()`（4 种关系 + 物理常识 + floor/wall 可见性）→ JSON 关系 → 物理常识后处理。
10. **Step 8/9**：输出 `scene_{id}_stage1.json` + 关键帧 + `keyframes_metadata.json`。

#### Stage2（正式重建，`mainv2.py::run_stage2()`）

1. **VGGT 3D 重建（三选一，`--vggt_model`）**
   - `vggt`：`load_vggt_model()` + `vggt_predict()`（patch14 / 518）
   - `vggt_omega`：`load_vggt_omega_frames()`（ffmpeg 抽帧 + `load_and_preprocess_images`，分辨率 512 / patch16）→ `load_vggt_omega_model()`（`torch.load + load_state_dict`，不用 from_pretrained）→ `vggt_omega_predict()`：`model(images)` 前向（Aggregator + CameraHead 单次输出 pose_enc + DenseHead 输出 depth）→ `encoding_to_camera()` 转 extrinsic/intrinsic → `unproject_depth_to_world_points()` 深度反投影 → world_points（无 PointHead，置信度用 depth_conf 近似）
   - `vggt4d`：`load_vggt4d_model()` + `vggt4d_predict()`（多目标跟踪）
   - 用完后 `unload_model()` 释放显存。
2. **SAM3(image) 分 floor/wall**：`segment_wall_and_floor()`（10 floor + 8 wall prompt 变体，覆盖卧室/客厅/办公室）。
3. **坐标系对齐**：`align_to_room_coordinate_system()` 五级 fallback（strict → relaxed → **Stage2.5 VLM 找地面参考点**（`detect_floor_reference_points_with_vlm` + SAM3 box prompt）→ large_plane → GeoCalib）。⚠️ 其中 VLM 分支有 `_load_vlm_model` 未定义的隐患（见 Q1）。
4. **保存中间结果**：color/depth/extrinsics/intrinsic.txt/point_cloud.ply。
5. **SAM3(video) 逐类分割+跟踪**：`load_sam3_video_model()` → `handle_request(start_session, video_path)` → 对每个类别 `segment_and_track(category)`：`add_prompt(text=category)` → `propagate_in_video()` 逐帧传播 → 得到每个实例的 `{frame_id, mask}` 序列 → `self_category_deduplicate()` 类内去重。
6. **CLIP 跨类去重**：`cross_category_deduplicate()`（`clip-vit-base-patch32` 比较外观特征合并）。
7. **JSON 白名单过滤**：只保留 Stage1 发现的类别 → 输出 `deduplicated_all_masks`。

#### Stage3（资产生成，`mainv2.py::run_stage3()`）

1. **选最优视角**：`get_optimal_view_frame_id()` 为每个实例挑 3D 表面积最大的帧；保存 `optimal_frames/`（供 Stage5.1 VLM 用）+ `instance_visibility.json`。
2. **SAM3D 生成 3D 资产**：`generate_3d_asset_in_subprocess()`（spawn 子进程避免 CUDA 冲突）：
   - 加载 `sam-3d-objects/notebook/inference.py` 的 `Inference`（config `./models/SAM3D/checkpoints/pipeline.yaml`）。
   - 对每个实例：`generate_3d_asset(image@最优帧, mask@最优帧, pointmap, extrinsic, inference)` → SAM3D `InferencePipelinePointMap` 输出 GLB mesh + rotation/scale/translation → 组装 `T` 变换矩阵（z-up）→ `all_instances`。
3. **多票验证 + 动态物体**：多帧验证资产质量；动态物体位置摆到首次被 SAM3 检测到的帧位置。

#### 两次 VLM 调用（都在 Stage1）

| 第几次 | 位置 | prompt | 输入 | 输出 |
|---|---|---|---|---|
| 第 1 次 | Step 3 `detect_objects_in_frames` | `VLM_DETECT_PROMPT`（"List all visible objects..."，只输出 JSON） | 每张关键帧图 | `{"objects":[{"name":"cup"}]}` |
| 第 2 次 | Step 7 `vlm_judge_relationships` | `_build_relationship_prompt`（4 种关系 + 物理常识 + floor/wall 可见性） | 物体名列表 | `{"relationships":{"obj":"supported by floor"}}` |

（另有 Step 5.5 点云补充检测的 `SUPPLEMENTARY_VLM_PROMPT`，可视为第 1.5 次；Stage2.5 找地面参考点、Stage5/6 的关系/材质识别是后续阶段的可选 VLM 调用，见 Q1。）

### Q3: `_load_vlm_model` 未定义，为什么日志 `116_C5_CellPhone_185deg_vggt_omega/mainv2_20260824_235728.log` 还能跑通？

**回答**: 因为 `_load_vlm_model(vlm_checkpoint)` 只存在于 `run_stage2` 坐标系对齐的**失败兜底分支**里，而这次运行第一步对齐就成功了，该分支从未执行。

- 该日志第 46 行：`坐标系对齐完成: stage1_strict (R[2,2]=-0.7672)`。
- `mainv2.py::run_stage2` 里 `_load_vlm_model` 的调用（第 383/401 行）都包在 `if np.allclose(R, np.eye(3), atol=1e-6):` 里，即只有 `align_to_room_coordinate_system` 返回单位矩阵（对齐失败）才会进入。
- 本次 `R[2,2]=-0.7672` 不是单位矩阵 → 所有 `np.allclose(R, np.eye(3))` 判断都为 False → `_load_vlm_model` 和 `vlm_checkpoint` 都未被求值 → 不会报错。

**触发条件**：只有当地面/墙面对齐彻底失败、回退到 Stage2.5 (VLM 找地面点) 或 Stage3 (large_plane) 时才会 NameError。且 `run_stage2` 签名里也没有 `vlm_checkpoint` 参数，一旦进入该分支，`vlm_checkpoint` 本身也会先报 NameError。所以这是"潜伏 Bug"——正常场景跑不到，遇到困难场景（地面遮挡/无墙）才会炸。

### Q4: mainv3.py 运行启动 Stage5 和 Stage6 可以实现吗？

**回答**: 可以。`mainv3.py` 已内置 `--enable_stage5` 和 `--enable_stage6` 开关（argparse 定义约第 1626/1637 行），依赖模块全部存在且函数签名与调用匹配：

| Stage | 调用位置 | 依赖 | 状态 |
|---|---|---|---|
| Stage 5.1 | `mainv3.py::run_stage5` → `tools/infer_relations_scene_graph.py` (scene_graph 默认) / `tools/refine_other_objects_relations.py` (per_object) | 需 keyframes（Stage1 产出）+ optimal_frames（Stage3 产出）+ VLM | ✅ 存在 |
| Stage 5.2 | `tools/refine_inter_object_placement.py::refine_inter_object_relations` | 5.1 的 refined_relations | ✅ 存在 |
| Stage 6 | `tools/material_annotator.py::run_stage6` | VLM + `tools/materials_db.json`（已存在） | ✅ 存在 |

VLM 默认路径 `/mnt/data/lza/models/Qwen3.5-9B` 存在。启动命令：

```bash
python mainv3.py --input_video ./232.mp4 --enable_stage5 --enable_stage6
```

注意两点：
1. **无法从 Stage5 直接开始**：mainv3 总是先跑 Stage1→2→3→基础精修，Stage5/6 只是可选追加阶段。
2. **单独补跑 Stage5** 可用 `tools/run_post_pipeline.py --scene_dir <输出目录> --stage5`，但 post pipeline **没有 `--stage6` 参数**——Stage6 只能跟随完整管线一起运行（需要内存中的 `all_instances`、`deduplicated_all_masks`、`colors`）。

### Q5: mainv2.py 和 mainv3.py 目前有什么区别？

**回答**: 目前**没有任何区别**。`diff` 逐字节比对完全一致（均为 81122 字节，diff 无输出），只有文件时间戳不同（mainv2.py: 2026-09-05 15:41，mainv3.py: 2026-09-02 14:59），说明 mainv3.py 是 mainv2.py 的副本（或其中一个是另一个的直接拷贝）。

两者都包含 `--enable_stage4/5/6`、`--enable_physics_validation` 等全部开关，功能完全相同。注意 mainv3.py 内部 docstring 和日志文件名仍写的是 "V2"/"mainv2_"（如日志文件 `mainv2_{timestamp}.log`），运行 mainv3.py 时不要被日志名误导。

### Q6: 3D重建模型默认是 VGGT-Omega 吗？

**回答**: 不是。`--vggt_model` 默认是 `vggt`（mainv3.py 第 1619-1621 行，`choices=["vggt", "vggt_omega", "vggt4d"]`）。要使用 VGGT-Ω 需显式指定 `--vggt_model vggt_omega`。

注意区分：`Qwen3.5-9B`（`DEFAULT_VLM_CHECKPOINT`）是默认的 **VLM**（Stage1/5/6 语义理解用），与 3D 重建模型（VGGT 系列）是两类不同的模型。

### Q7: 帧数和设备如何指定？

**回答**:

**帧数**（2 个 CLI 参数）：
- `--max_frames`（默认 160）：Stage2 VGGT 重建最大帧数，同时作为 `vggt_max_frames` 传给 Stage1 的 3D 重建。
- `--max_frames_stage1`（默认 10）：Stage1 采样关键帧数（VLM 检测/关系判断用图）。

**设备**（无 CLI 参数，代码自动/硬编码）：
- mainv3.py 第 332 行：`device = "cuda" if torch.cuda.is_available() else "cpu"`，自动检测。
- 第 495 行：CLIP 硬编码 `.cuda()`；第 1038 行：Stage4 MASt3R 硬编码 `mast3r_device='cuda'`。
- 指定具体 GPU 用环境变量：`CUDA_VISIBLE_DEVICES=1 python mainv3.py ...`。

### Q8: Stage3 报 `KeyError: 'max_area'` 如何解决？

**回答**: 这是 mainv3.py 第 586 行日志打印引用了不存在的键。`get_optimal_view_frame_id`（src/geometry_utils.py 第 786 行）返回的 `motion_info` 只含 `median_disp / max_disp / first_valid_frame / num_valid_frames` 四个键（见函数 docstring 和第 837-838、873-878 行两处返回），**没有 `max_area`**——但第 586 行打印写了 `motion_info['max_area']`，Stage3 处理第一个实例时就抛 KeyError。

**修复**（mainv3.py 第 586-588 行）：改用实际存在的键打印位移信息：

```python
print(f"   {category}_{inst_idx}: [SELECT] median_disp={motion_info['median_disp']:.4f}m, "
      f"max_disp={motion_info['max_disp']:.4f}m, "
      f"frames={motion_info['num_valid_frames']} → frame {optimal_frame_id}", flush=True)
```

已验证：`py_compile` 通过；全文件 `motion_info[...]` 访问仅此 3 处，键均存在（早期返回与正常返回两条路径都覆盖）。

**注意**: mainv2.py 与 mainv3.py 修复前逐字节相同，存在同样的 bug，如需跑 mainv2 也要同步修改。

### Q9: 还有没有相关的 KeyError 问题？

**回答**: 对 mainv3.py 主流水线做了全量审计，KeyError 类问题已清零：

- `motion_info[...]` 仅第 586-588 行 3 处，键在 `get_optimal_view_frame_id` 两条返回路径均存在（Q8 已修）。
- `vggt_prediction_results[...]` 约 40 处：vggt / vggt_omega / vggt4d 三个 predict 函数均完整返回 7 个键（含 `world_points_conf`），换模型无风险。
- `categories_and_relations[category]`（第 1355 行）已先经 surviving_categories 过滤，安全。
- mask 字典（`frame_id`/`mask`）与实例字典（`original_mesh`/`T`）由内部模块统一生成，格式一致。

但发现 2 个同类的**潜伏 NameError**（Stage 2.5 VLM 辅助对齐兜底分支，第 383/386/401 行）：
1. `_load_vlm_model` 未定义——整个仓库无此函数定义（Q3 已记录过）。
2. `align_via_vlm_floor_points` 未 import——函数在 src/geometry_utils.py:494 存在，但 mainv3.py 导入列表缺失。

触发条件：Stage1-strict 与 Stage2-relaxed 对齐均失败（地面遮挡/无墙的困难场景）。正常场景不触发。

### Q10: Stage5.2 报 `❌ 精修失败: name 'supported_mesh' is not defined` 是什么情况？需要解决吗？

**回答**: 需要解决。这是 `tools/refine_inter_object_placement.py` 中 `sp_refine_on_top`（第 629 行起）的真 bug：

- **根因**: 第 675-676 行引用 `supported_mesh`，但该函数内从未定义此变量——被支撑物的变换后 mesh 在第 667 行叫 `transformed_mesh`（`supported_mesh` 只在别的函数如 `_compute_penetration_depth` 第 617 行中定义）。函数重命名变量时漏改了这两行。
- **影响**: `sp_refine_on_top` 每次执行到第 675 行必抛 NameError，被调用方第 2109 行的 try/except 捕获后只打印 `❌ 精修失败`——**Stage5.2 所有 on_top 策略的精修全部静默失效，物体保持原位**。仅 `refinement_degree=="correct"`（VLM 判定无需调整）时因提前 return 而幸免。
- **修复**: 第 675-676 行 `supported_mesh` → `transformed_mesh`。
- **验证**: py_compile 通过；用两个小立方体做冒烟测试（supported 悬空 0.2m），修复后正常执行且底面精确贴合支撑物顶面（z=1.2 → 1.0）。
- 其他策略函数（`sp_refine_inside`/`sp_refine_against_side`/`sp_refine_hanging_below`/`sp_refine_leaning`）逐一检查，无同类未定义变量问题。

注：此文件被 mainv2/mainv3/run_post_pipeline 共享，修复对三者同时生效。

### Q11: 能在服务器终端直接渲染 GLB 预览图，不用每次传文件吗？

**回答**: 可以，已实现 `tools/render_glb_preview.py`（pyrender EGL 无头渲染，无需显示器）：

```bash
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/render_glb_preview.py output_v2/xxx/final_scene.glb
# 支持多个 GLB、自定义视角/分辨率:
#   ... render_glb_preview.py a.glb b.glb --views front,iso --size 800
```

- 输出: GLB 同目录下 `<stem>_preview.png`，默认 2x2 四视角（front/side/top/iso），带纹理/顶点色，自动跳过 grid 线条几何。
- 查看方式: ① Trae IDE 文件树直接点开 PNG（远程文件无需传输）；② 在对话里让 Agent 渲染并用 Read 内嵌显示。
- 实现要点: `PYOPENGL_PLATFORM=egl` + `pyrender.OffscreenRenderer`；GLB 为 y-up，top 视角从 +y 俯视；相机距离 `radius*2.6` 自适应包围盒。
- 踩坑记录: pyrender `Scene` 无 `.render()` 方法，必须用 `OffscreenRenderer.render(scene)`；相机/灯节点复用 `set_pose` 而非反复 add/remove（后者会触发 networkx "node not in digraph" 错误）。

### Q12: GLB 预览渲染能集成到 mainv3 管线中吗？

**回答**: 已集成。在 mainv3.py 新增 `_render_glb_preview(glb_path)`（第 1153-1167 行），并在 `save_final_glb()` 导出 GLB 后调用（第 1215 行）。mainv3 所有 GLB 保存（`final_scene_initial.glb` / `final_scene.glb` / `final_scene_stage4.glb` / `final_scene_stage4_5.glb` / `final_scene_stage5.glb`）都经过 `save_final_glb`，因此每次保存都会自动在同目录生成 `<stem>_preview.png`（2x2 四视角）。

设计要点：
- **懒加载**: `tools.render_glb_preview` 在函数内 import，不拖慢启动。
- **失败隔离**: 整个渲染包在 try/except 中，失败只打印 `⚠️ 预览图渲染失败 (不影响主流水线)`，管线继续。
- 验证: py_compile 通过；用与 `_render_glb_preview` 完全一致的调用路径（懒加载 + EGL + 渲染 + 拼图 + 写盘）做断言测试，输出 PNG 尺寸/大小校验通过。

### Q13: render_glb_preview 需要支持真实相机视角（能看到各实例"最大帧"的视角）

**回答**: 已实现。服务器无显示器，pyrender EGL 无头渲染输出 PNG 到场景输出目录；新增真实相机视角渲染：

```bash
# 各实例最大帧视角 (Stage3 最优视角帧)
python tools/render_glb_preview.py output_v2/xxx/final_scene.glb --views none --camera_frame optimal
# 指定帧号 / 只出合成视角
python tools/render_glb_preview.py xxx.glb --camera_frame 0,85    # 指定帧
python tools/render_glb_preview.py xxx.glb                        # 默认合成四视角
```

输出 `<stem>_camview.png`（按帧标注 cam{fid}，分辨率与 color/{fid}.jpg 一致）。mainv3 集成点 `_render_glb_preview` 现在每次保存 GLB 自动生成两张图：`<stem>_preview.png`（合成四视角）+ `<stem>_camview.png`（真实相机视角）。

**关键坐标坑**（排查记录）: 首版相机视角渲染全白。诊断链：
1. 对照实验——用 stage4 proven 渲染器 + all_instances.pkl 渲染 → 图像正常，实例世界 z∈[0.51,1.81] 与点云吻合 → 管线数据没问题。
2. 根因——**trimesh 加载 GLB 时会自动把 GLB 的 y-up 约定转回 trimesh 内部 z-up**，对本管线 GLB 净效果 = 恢复管线原始 z-up 世界坐标（与 extrinsics 同一坐标系）。首版误以为加载后是 y-up，多加了一次翻转 → 相机朝向错 90°。
3. 修复——相机位姿 `inv(w2c) @ diag(1,-1,-1,1)`（OpenCV→OpenGL，世界保持 z-up）；合成视角方向也改为 z-up 语义。
4. 验证——044 场景 cam85 渲染图与真实 color/85.jpg 布局逐物体对应（椅子/茶桌镂空面板/桌边白杯）；002 场景 6 个最优帧视角全部正常。

### Q14: 最终产物是 stage5 的 GLB 吗？

**回答**: 是，按开关组合决定（mainv3.py 第 1599-1609 行）：
- 只开 `--enable_stage5` → 最终产物 `final_scene_stage5.glb`（5.1 关系精修 + 5.2 SP空间精修 + 稳定性检查后）。
- `--enable_stage4 --enable_stage5` → `final_scene_stage4_5.glb`；都不开 → `final_scene.glb`（仅基础精修）。
- **Stage6 不修改 GLB**，只输出 `materials.json` + `materials_debug.json`，最终几何仍是 stage5 那份。
- 注意区分中间文件 `final_scene_stage5_sp.glb`（refine_inter_object_placement.py 第 2121 行，SP精修+穿模修复后、稳定性检查**前**的中间产物），不要误当最终结果。看预览图认准 `final_scene_stage5_preview.png` / `final_scene_stage5_camview.png`。

### Q15: 112 场景日志显示穿模解决迭代 8 次，但最终结果左右穿模仍在，为什么？

**回答**: 不是穿模解决没生效，而是**它的成果被后续步骤作废了**。

Stage5.2 的执行顺序（refine_inter_object_relations）：
1. SP 精修 → 2. `resolve_penetrations`（迭代 8 次推开穿模对，日志中 cabinet_2↔cabinet_4 等反复修 6 次~0.6m）→ 3. 保存 `final_scene_stage5_sp.glb`（此时穿模已基本解决，渲染确认柜子间有间隙）→ 4. **`check_stability` Phase1 地面对齐：把所有落地物体旋转立正（112 场景 6 个 cabinet theta 12°~164°→0°，绕各自 mesh 原点大幅旋转）** → 5. 保存 `final_scene_stage5.glb`。

第 4 步旋转改变物体足印（如 theta=60° 的长柜 x 向足印 0.62m→0.90m），**把第 2 步推开的柜子又转回重叠状态**，且之后无任何穿模复检——这就是用户看到"左右穿模没解决"的原因。

**修复**（refine_inter_object_placement.py 第 2148-2156 行）：`check_stability` 之后、保存最终 GLB 之前，再跑一次 `resolve_penetrations(shared_penetration=True)`。

**验证**：单元测试复现完整序列——斜置 60° 长柜与正立柜初始无重叠 → check_stability 旋转后重叠 0.02m³（Bug 复现）→ 再次 resolve_penetrations 后重叠归零且两物体仍 z 贴地。渲染对比：112 的 `_sp` 版本柜子分离，final 版本柜子明显互相穿插，与诊断一致。

**已损坏的历史输出**需重跑才能修复：`python tools/run_post_pipeline.py --scene_dir output_v2/112_C5_CellPhone_191deg_vggt_omega --stage5`（会重跑 5.1+5.2，约 10 分钟）。

### Q16: Stage2 崩溃 `ValueError: max() arg is an empty sequence`（align_to_room_coordinate_system 第 246 行）是什么问题？

**回答**: Stage1-strict 坐标系对齐的健壮性 bug。SAM3 在不同帧检出的多个 floor mask 平面拟合后**法向分歧超过 30°**（如某帧把桌面/斜面误当地面）→ `vaild_floor_plane_infos` 过滤后为空 → `max()` 崩溃。代码本该像第 240 行（无 floor 时）那样返回单位阵表示"对齐失败"，让 run_stage2 走 Stage2-relaxed → Stage2.5 VLM → large_plane → GeoCalib 兜底链。

**修复**（src/geometry_utils.py 第 246-249 行）：过滤后为空时 `return np.eye(3), np.zeros(3)`。

**验证**：单元测试 3 场景——①两个 floor mask 法向差 60° → 优雅返回单位阵（原崩溃点）；②一致 floor → 正常对齐（R2[2,2]=1.000）；③无 floor → 单位阵（原有行为）。py_compile 通过。

**注意**: 重跑后该视频的坐标系对齐会落到兜底链（可能精度略低），可查看输出目录 `coordinate_alignment.json` 的 `alignment_stage` 字段确认用了哪级对齐。

### Q17: Stage2.5 兜底分支 `NameError: _load_vlm_model is not defined`（Q9 预警的潜伏 bug 触发）

**回答**: Q16 修复让 Stage1-strict 优雅降级后，对齐落入 Stage2.5（VLM 辅助）分支，暴露了 Q9 预警的两个潜伏 NameError + 一个隐藏的参数缺失，共 3 处：

1. **`_load_vlm_model` 未定义**（mainv3.py 第 383/401 行调用）→ 新增同名函数（第 306-313 行），懒加载复用 Stage1 的 `tools.generate_scene_json_stage1.load_vlm_model`（支持 Qwen3.5-FP8/Qwen2.5-VL/Qwen3-VL，含显存预算/offload 逻辑，与 Stage1 行为完全一致）。
2. **`align_via_vlm_floor_points` 未导入**（第 386 行调用）→ 加入 src.geometry_utils 导入列表（第 172 行）。
3. **`vlm_checkpoint` 不在 run_stage2 作用域**（Q3 分析过：签名无此参数）→ run_stage2 签名加 `vlm_checkpoint=None`（第 317 行），main() 调用处传入（第 1374 行）。

**验证**: py_compile 通过；进程内 import mainv3 后断言 `_load_vlm_model` 存在且解析到 `tools.generate_scene_json_stage1.load_vlm_model`、`align_via_vlm_floor_points` 已导入、run_stage2 签名含 vlm_checkpoint。VLM 实际加载逻辑与 Stage1 同源（Stage1 在多次运行中已验证可用）。

至此坐标系对齐兜底链完整可用: Stage1-strict → Stage2-relaxed → Stage2.5 VLM辅助 → Stage3 large_plane → Stage4 GeoCalib。

### Q18: run_post_pipeline 跑 5.2 时 `NameError: _load_vlm_model is not defined`——目前的大模型都得不到吗？

**回答**: 模型本身一直都在（`/mnt/data/lza/models/Qwen3.5-9B`，Stage1/5.1 多次成功加载），"得不到"是**调用侧引用了从未定义的函数名**——与 Q17 同族的第三处潜伏 bug。全仓库排查 `_load_vlm_model(` 调用点后统一修复：

1. **tools/refine_inter_object_placement.py**（本次崩溃点，第 1946 行）：新增 `_load_vlm_model`（第 180-183 行），懒加载复用 Stage1 的 `load_vlm_model`。该模块第 175 行已把仓库根目录插入 sys.path，`from tools...` 导入链可用。
2. **mainv2.py**（与 mainv3 修复前完全相同的 3 处问题）：同步修复——`_load_vlm_model` 定义（第 306-313 行）、`align_via_vlm_floor_points` 导入（第 172 行）、`run_stage2` 加 `vlm_checkpoint=None` 参数（第 316-317 行）+ main() 传入（第 1333 行）。

**验证**: py_compile 两文件通过；进程内断言 refine_inter_object_placement/mainv2 的 `_load_vlm_model` 可解析、mainv2 三修复点齐全，解析链统一指向 `tools.generate_scene_json_stage1.load_vlm_model`。

至此 VLM 加载在全仓库统一收口到 Stage1 的单一实现，`_load_vlm_model(` 的全部 6 处调用点（mainv2×2、mainv3×2、refine_inter_object_placement×1、定义×3）均有定义可用。

---

## 二点五、动态物体判断与 Pick/Place 检测

### Q9: 目前的动态物体判断和 Pick/Place 检测整个流程是什么样的？

**回答**: 完整文档已更新到 [.trae/documents/动态物体判断逻辑分析.md](../.trae/documents/动态物体判断逻辑分析.md)。核心结论：

1. **第一层** `get_optimal_view_frame_id`（[geometry_utils.py:786](../src/geometry_utils.py#L786)）：`median_disp > 0.10` 打 DYNAMIC 标签，**仅用于最优视角选择**（mainv3.py:590 注明"动态判定由第二轮滑动窗口逻辑统一处理"）。

2. **第二层**（唯一生效判定）：置信度加权质心（conf 的 10th percentile 自适应阈值）→ 相邻帧位移 → MAD 剔除离群（|z|<3.5）→ `max_consecutive_high`（位移>p50 的最长连续帧数）→ **`is_dynamic = max_consecutive_high > 30`**（[mainv2.py:709](../mainv2.py#L709) / [mainv3.py:710](../mainv3.py#L710)）。物理依据：真实运动产生大量连续中等位移；VGGT 漂移零散、连续帧数少。可见帧 <15 跳过。

3. **Pick/Place 帧检测**（大小物体分治，5 场景验证全对）：
   - 小物体 (avg_area<1000)：质心法 `path>0.15 AND net>0.08 AND pnet<5.0 AND max_run×frame_ratio≥15`
   - 大物体 (avg_area≥1000)：5帧窗口 overlap 法 `min_match<0.50 AND low_count×frame_ratio≥15`（v2 用"低匹配总帧数"替代"最长连续段 r50"，修复了 116 场景慢速移动漏检）
   - 后处理先合并（gap≤20原始帧）再过滤（持续≥15原始帧），帧号 ×frame_ratio 转到原始视频空间
   - **代码状态**：`pick_place_detector.py` 与测试脚本在工作区已丢失，输出 JSON 仍保留（output_v2/*/pick_place_frames_combined.json），待重建并集成到 mainv2/mainv3。

---

## 三、资产生成（SAM3D）

### Q13: 之前 3chair-bottle 生成的两个椅子不一样的问题有记录下来吗？

**回答**: 有记录，在 [docs/SAM3D_SCALE_DRIFT_REPORT.md](SAM3D_SCALE_DRIFT_REPORT.md)（2026-08-02）。注意：报告的触发调查场景是 **2chair-bottle_vggt_omega**（不是 3chair-bottle），现象为两把视觉上完全相同的椅子最终 mesh 尺寸相差约 1.43 倍（z 向 1.487m vs 2.125m），而两者 VGGT 点云实测尺寸几乎一致（z 向均约 0.97m）。

核心结论：

1. **根因**：SAM3D 单图模式下 `use_scene_scale=True` 的尺度归一化不可靠——scale 用"物体中心到整帧最远背景的距离"当尺子，随相机角度/物体位置变化（两帧 scene_scale 差 18%），叠加模型 exp(raw_scale) 补偿过头 28%，两因子相乘得 1.51x。不能关掉该配置（会破坏训练约定，预测崩溃）。
2. **漂移模式**：逐实例随机漂移而非全局放大（30 个实例批量诊断，ratio 有 1.0 也有 1.5~1.8x，方向不固定）；VGGT 世界坐标系本身可靠（点云实测高度均符合物理常识）。
3. **附带发现**：被移动物体跨全视频联合 mask 测尺寸会把运动轨迹算进去得到假大尺寸，测真实尺寸必须只用静止段帧。
4. **修复方案（B，推荐）**：post-hoc 点云校准——静止段（质心 K-Means 聚类最大簇）点云取 [0.5%, 99.5%] 分位得真实尺寸，对 mesh 绕质心等比缩放使其等于点云尺寸。计划实现为 `tools/rescale_meshes_by_pointcloud.py`。
5. **当前状态**：两个诊断脚本已存在（`tools/test_pointcloud_vs_mesh_size.py` 单场景、`tools/diagnose_scenes.py` 多场景批量）。**方案 B 已于 2026-09-13 落地**：`tools/rescale_meshes_by_pointcloud.py` 在 2chair-bottle_vggt_omega 验证通过（chair_0 k=0.646 / chair_1 k=0.473 / table_0 k=0.618 / bottle_0 k=1.062，校准后 z 向 ratio≈1.00，前后 GLB 预览确认两把椅子尺寸一致），相对报告原配方有三处实测驱动的偏差（最紧凑簇+移动判定、移动物体逐帧中位数、移动物体主轴 k），详见 [SAM3D_SCALE_DRIFT_REPORT.md §5.3](SAM3D_SCALE_DRIFT_REPORT.md)。下游集成已于 2026-09-13 完成：校准调用插入 mainv3.py 主流程 Stage3 之后（第 1385-1394 行），all_instances.pkl 本身即校准后数据，post_pipeline 无需改动。

### Q14: 校准后的渲染里瓶子为什么不在桌子上？是 stage5 出问题了吗？

**回答**: 校准模块没有移动瓶子（before/after 质心逐位一致 `[0.007, 0.684, 0.456]`，底面锚定缩放只改尺寸）。瓶子"不在桌上"是**既有管线问题**，证据链：

1. **瓶子在源数据里就低了 ~0.49m**：mesh 世界中瓶底 z=0.379，而桌面顶 z=0.864。根因是坐标系不一致——stage3 动态物体按"首次检测帧位置"用**原始点云坐标**摆放（该帧点云地面在 z=-0.459），而静态物体（桌/椅）被 stage5 Phase1 对齐到**地面 z=0** 的坐标系。
2. **stage5.2 精修当时静默失败**：stage5.1 已正确判定 `bottle_0: supported by table_0`（final_relations.json），本应由 `sp_refine_on_top` 把瓶子放到桌面；但该场景 2026-07-27 生成，当时 `supported_mesh` NameError（Q10，2026-09-13 才修复）使所有 on_top 精修被 try/except 吞掉，瓶子从未被放回桌面。
3. **为什么之前看起来"在桌上"**：校准前桌面被放大 1.62 倍且范围盖住瓶子位置，瓶子悬在桌体内部被遮挡；校准把桌子缩小后瓶子露出来，问题显形。
4. **附带发现**：源数据两把椅子 min_z=-0.682/-0.963（埋地下），是 stage5 Phase1 地面对齐计算偏差（报告 §7.3 已记录）。

**结论**：stage5 确实有问题（该场景生成于 Q10 修复之前）。修复方式：用当前代码重跑 `tools/run_post_pipeline.py --scene_dir <场景> --stage5`，让修复后的 on_top 精修把瓶子放回桌面，同时地面对齐会重新处理椅子埋地问题。

另外本次修了校准模块自身一个 bug：绕质心缩放使落地家具悬浮（table min_z 0→0.249），已改为**底面中心锚定**（水平位置不变、贴地物体仍贴地），k 值与校准效果不受影响。
