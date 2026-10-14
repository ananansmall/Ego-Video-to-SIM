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

## Q15: xy 不是有相对的修复吗？Phase3 悬空该怎么修？ (2026-09-28)

**问**: 现有代码不是有 xy 修复吗？接触比 0% 只修 z 的问题应该怎么修？

**答**:
1. 现有 "xy偏移" 逻辑 (refine_inter_object_placement.py L1467-1527) 是**反向跟随**:
   支撑物在穿模解决中被移动时, 其上小物体跟随同样的 xy delta。
   它不解决"物体不在支撑面上"的问题。
2. 更底层的根因: **shapely 从未安装**, `_project_footprint` 静默返回 None,
   所有历史日志的"接触比 0.0%"都是假值, Phase3 一直在盲目 z 对齐。
   → 安装 shapely 2.1.2 修复。
3. Phase3 新增 xy 拉回: 接触比 <5% 时, 把物体平移到支撑足迹内缩区最近点
   (上限 0.6m), 再重算 z。实测 bag/bottle 拉回后接触比 89%/85%。

## Q16: HOI4D 瓶子 1m 高 + 悬浮，supporter 分配有错吗？ (2026-10-01)

**问**: 082_C5 场景瓶子全 1m 高、部分悬浮，是 5.1 场景图把家具分配错了吗？

**答**: 分配本身没错（场景图区分了 table_0/table_1，bowl_2 正确分到 table_1）。
真正根因有两个:
1. 校准混合分支的 z 公式 k_z=(z_max-ground)/mesh_z 只对落地家具成立。桌上的瓶子
   gap=0.9m 时把桌高算进瓶高 → k_z≈2.5 → 1m 高瓶。guard 因 after_z≡1.0 而失明。
   → 修复: gap>=0.50 改用点云自身 z 跨度; guard 补查 k_z ∈ [0.4,2.0]。
2. Phase2 悬空修复只修 z 不修 xy（与 bag 问题同源）。
   → 修复: 抽取 _pull_xy_into_supporter() 公共函数，Phase2/Phase3 共用。

## Q17: 为什么 Stage5 跑 2h43m 这么长？ (2026-10-01)

**答**: 时间炸弹在 Stage5 的 VLM 调用数 = 实例数 × 投票帧数:
- 5.1: 每实例 3 次调用 × ~40s (25 实例 → 54min)
- 5.2: 23 对支撑 × ~11 帧投票 (帧映射修复后调用真实发生, ~25min)
已实施投票帧数上限 (5.2 [:5], 5.1 关键帧 [:1]), 预期省 ~30min。
遗留两大瓶颈: 5.1 单次调用延迟 (标注图全场景 JSON 可裁剪) 和
穿模修复 O(N²) 相交检测 (~34min), 属独立优化任务。

## Q18: `_vlm_inference is not defined` 是怎么回事？影响有多大？ (2026-10-02)

**问**: Stage1 补充帧检测 10/10 报 `❌ 错误: name '_vlm_inference' is not defined`，这个 bug 修一下。

**答**: `src/vlm_logger.py` 重构时把 `_vlm_inference` 改名为 `vlm_inference`（统一 VLM
日志入口），但 3 处调用点没跟上，NameError 被 try/except 吞掉只打 ❌:
1. generate_scene_json_stage1.py L1536（补充帧检测）
2. generate_scene_json_stage1.py L2040（Stage1 7.3 关系跨帧投票）
3. refine_inter_object_placement.py L504（Stage5.2 放置策略投票）

其中 2 和 3 是**静默失效**——比报错更严重:
- Stage1 关系投票从未生效 → stage1 关系恒为全 "supported by floor"（全靠 5.1 兜底）
- 5.2 放置策略投票从未生效 → 恒 fallback "on_top"（inside/against_side 从未触发）

已全部改为 `vlm_inference`，py_compile 通过。**行为变化提醒**: 修复后重跑，
stage1 关系将首次按 VLM 多数投票输出，5.2 策略也会开始真正区分——结果分布会变，
需对比确认合理性而非当作退化。

## Q19: 为什么 23/25 实例的尺寸校准都被 guard 拦截了？ (2026-10-03)

**问**: HOI4D 重跑后校准日志显示 23/25 实例"测量不可信, 保留原始尺寸",
z 比值一致地落在 0.06~0.28，guard 形同把校准整个取消了。

**答**: guard 的**校验参照物与实际缩放量不同源**。
混合分支用 `k_z = (z_max - z_min_pc) / mesh_z` 缩放 z，却拿
`z_range = z_max - ground_z` 做 after 校验；而 `ground_z` 是全场景 z 的 2% 分位，
被埋地家具拉到 -0.757（真实地面远高于此），z_range 被撑大约 0.76m，
于是 `after_z = z_after/z_range` 一律 0.2~0.3 —— 测量本来正确的实例
（实测点云z/meshz = 0.80~1.17，高度吻合）被全判成垃圾。

同一个错误有两面: 公式不等时被污染的 z_range **误杀好结果**；
公式恰好相等时 after_z≡1.0 **自校验失明**（此前 scissor_1 就是这样漏过去的）。

**诊断教训**: 前两轮推断都错了 — 先怀疑 moved 分支、再怀疑分支选择错，
用真实数据复现后都被推翻（moved=False；换 mesh_min_z 判据 0/25 改变）。
**真正定位靠的是直接打印 pc_z/mesh_z 这个"应该≈1"的量**，
一眼看出点云测量本来是好的，问题在校验逻辑而不在测量。
教训: 遇到"大量实例同时异常"时，先找一个"应该等于某个常数"的中间量验证，
而不是在分支逻辑里推演。

**→ 修复**: after_ratio[2] 参照物改为 `pc_z_span = z_max - z_min_pc`，
与 k_z 分子同源。实测拦截数 22/25 → 2/25，剩余 2 个（贴地剪刀 mesh_z≈0.01m）
属真实点云不可信，拦截正确。

## Q20: 5.1 关系全对，为什么 21 个物体还是没被放到桌上？ (2026-10-03)

**问**: HOI4D 082_C5 跑完，5.1 明确输出 23 个 `supported by table_0`，
但物体底面全在 z≈0 而台面 z=0.971 —— 关系对了，位置却没动。

**答**: `sp_refine_on_top` 开头有 `if refinement_degree == "correct": return`，
而 `refinement_degree` 来自 `_judge_refinement_degree_with_vlm` —— 它**看照片**回答。
视频里瓶子确实在桌上，视觉上就是 "correct"（实测 23/23 全答 correct），
但 3D 坐标是错的。**用 2D 视觉判断否决了明确的几何事实**。

**这是 Q18 那个 bug 修复后暴露的**: 修复前 `_vlm_inference` 静默失败被 except 兜底
返回 "minor"，反而触发了 z 对齐，瓶子被正确放到桌上。修复后 VLM 真正工作，
缺陷才显形。**静默失败曾意外充当了兜底** —— 修bug 时要留意这类"意外正确"，
它会在 bug 被修好时变成新问题。

**→ 修复**: 改为几何优先。`gap >= 0.15m` 说明几何上必定错位，不问 VLM 直接对齐；
只有几何已贴合时才问 VLM（保留"避免无谓调整"的原意）。副作用是 VLM 调用
23 次 → 0 次，再省约 20 分钟。

## Q21: 5.2 每个物体投票 60 多次，不是上限 5 帧吗？ (2026-10-03)

**问**: 代码里 `_MAX_VOTE_FRAMES = 5`, 但日志显示每个物体投票 60~65 次。

**答**: 上限只在单次调用内生效, 管不住外层循环。外层遍历该类别下**每个实例**
各调一次 (bottle 有 16 实例 → 16×5 = 80 次), 而要回答的问题"bottle 相对
table_0 怎么放"与具体哪个实例无关; 最终又只取第一个实例的结果。

全场景 1078 次调用 ≈ 38 分钟, 换回一句 `{"strategy": "on_top"}`。

**→ 修复**: 按类别选代表实例 (帧最多者) 只投一次 → 20 次 (降 98%)。
验证 4 个类别内部投票结果全部一致, 语义等价。

**附带发现**: 该处 `vlm_inference` 漏传 `caller` 参数, 导致这 1078 次调用
在 vlm_conversation.jsonl 里全归因为 `unknown` —— 日志归因缺失让这个问题
一开始难以定位。补上 `caller="stage5_placement_strategy"`。

## Q22: 穿模检测 FCL 5.5s/对，降密到 10% 能不能提速？ (2026-10-03)

**问**: 25 实例穿模检测跑了 40 分钟。想按"粗筛+细筛"优化。

**答**: 先测了真实瓶颈分布, 发现**粗筛（AABB）代码里已经有了**, 而且很便宜:
300 对 → AABB 相交 44 对 (过滤 85%), 仅 0.56s。瓶颈在 FCL 细筛: 44 次 ×
5.51s = 245s/轮。真正的原因是网格有 **594 万顶点**, FCL 每次碰撞都要建 BVH。

验证了两个假设, **都被否证**:
1. 复用 BVH (预构建 manager 复用查询): 摊销后 4.95s/对, 反而略慢。构建 2.6s/个,
   查询 2.35s/对 —— 开销不只在构建, 查询本身也慢。
2. 降密: 装了 `fast_simplification` 后, 降到 10% 面时**单轮检测判定完全一致**
   (18 穿透, 0 漏报 0 误报, 深度误差中位 5mm), 单对 5.51s → 0.51s。

**但最终没有采用降密** —— 因为完整验证推翻了结论:
| | 预测 | 实测 |
|---|---|---|
| 提速 | 10.8x | **仅 2.4x** (1057s → 444s) |
| 最终位置 | 一致 | **9 个实例差异, 最大 0.32m** (table_1 的 z) |

**教训**: 我只验证了"单轮检测的穿透数量一致"就外推"多轮修复结果一致"。
穿模修复是**迭代过程**, 初始 1~2cm 的深度误差在多轮累积后放大成 0.32m 位置偏差。
**验证迭代算法必须比对最终输出, 不能只比对中间判据**。
另外单对提速 10.8x 却没有换来整体 10.8x, 说明迭代轮次内AABB 命中集在变化,
降密改变了哪些对被判为穿透, 反而增加了需要精确检测的配对数。

**当前状态**: 穿模耗时未优化 (25 实例约 40min)。可行的方向改为:
(a) 减少迭代轮数(需验证收敛性); (b) 按 AABB 重叠量排序优先处理大重叠对;
(c) 接受现状 —— 该瓶颈只在 25 实例的拥挤场景出现, egodex 简单场景实例少不触发。

## Q23: inside 策略精修报 unexpected keyword argument 'max_offset' (2026-10-04)

**问**: egodex_0 重跑时 5.2 报 `sp_refine_inside() got an unexpected keyword
argument 'max_offset'`，但 HOI4D 从未出现过。

**答**: 调用点把 on_top/inside 放在同一个 else 分支，无条件传
`max_offset/initial_offset/refinement_degree`，但只有 `sp_refine_on_top` 的签名
支持这三个参数。`sp_refine_inside` / `sp_refine_hanging_below` /
`sp_refine_leaning` 都只有两个参数。TypeError 被 except 吞掉 → 精修静默失效。
`leaning` 同理：它与 `against_side` 同分支传 walls_info，但只接受 2 参。

**为什么现在才暴露**: Q18 修复 `_vlm_inference` NameError 之前，VLM 策略判定
静默失败恒fallback "on_top"，inside/leaning 分支**从未被执行过**。修复后 VLM
真正工作返回 "inside"，才撞上这条从未运行的代码路径。

**教训（与 Q20 同源，值得记住）**: 静默失败（try/except 吞异常）会让一整条代码
路径长期不被执行，其中隐藏的错误要等上游 bug 修复后才暴露。**修 bug 时要预期
会牵出"从未执行的分支"**，尤其当修复让某个条件第一次真正成立时。

**→ 修复**: 按真实签名分派（on_top 传3 参；against_side 传 walls_info；
leaning/inside/hanging_below 只传 2 参）。五策略实测 5/5 通过。

## Q24: egodex_34 在 Stage3 报 RemoteDisconnected 崩溃？ (2026-10-04)

**问**: egodex_0/1 都跑完了, egodex_34 在 Stage3 崩溃:
`hydra...InstantiationException: ...dino.Dino: RemoteDisconnected('Remote end closed connection without response')`

**答**: 不是模型或代码问题, 是 SAM3D 的 DINO backbone 依赖联网。
`sam3d_objects/model/backbone/dit/embedder/dino.py` 用
`torch.hub.load(repo_or_dir="facebookresearch/dinov2", source="github")`,
torch.hub 每次会联网访问 GitHub 确认 repo commit。**代码缓存
(`~/.cache/torch/hub/facebookresearch_dinov2_main`) 与权重
(`.../checkpoints/dinovov2_vitl14_reg4_pretrain.pth`) 明明都在**,
但联网那一步失败就让整个模型实例化崩溃。
egodex_0/1 能跑完只是当时网络恰好可用 —— 这是偶发故障, 迟早会复现。

**→ 修复**: 检测到本地 hub 缓存目录时, 把 repo_or_dir 指向该目录并用
`source="local"` 完全离线; 无缓存才回退 github 方式。
实测 `torch.hub.load(local, 'dinov2_vitl14_reg', source='local')` 正常
(304.4M 参数)。

## Q25: 更正: 关于"瓶身形状失真"的说法不可靠 (2026-10-04)

**问**: 我先后说了两个关于瓶身比例的结论，哪个是对的?

**答**: **两个都不成立，特此更正。**

1. 第一次说"mesh 长宽比 2.4 vs 真实 3.5，形状失真" —— "3.5" 是我按常见饮料瓶规格
   **估算**的，没有量过该视频里的瓶子。
2. 第二次说"形状没失真，粗的是整体尺度 (mesh 直径是点云的 1.82 倍)" —— 用点云实测
   得出，但**该数据本身不可信**。

**不可信的原因 (实测确认)**: HOI4D 的 mask 只标注**可见面**。圆柱形瓶子的侧壁
背面没有点云, 实测 xy 直径只有 0.037~0.106m (3.7~10.6cm), 明显小于真实直径 ——
**系统性低估**。细长物体尤其严重 (侧面几乎完全不可见)。

因此当前**无法判断**瓶身比例是否正确, 因为缺少可靠测量基准。
HOI4D 提供 GT mesh, 但本地只下载了 N15/N46/N47 三个子集, **没有 082**。

**各向异性残差 0.44~0.50 的正确解读**: 它不是 stage5 的问题, 而是校准环节的
测量局限 (点云只见可见面, xy 方向尤甚)。该残差**早于本轮修复存在** —— 修复前
22/25 实例被 guard 整体拦截, 残差不可见; 修复后 21/25 生效, 残差才显形。
**修复让原本被掩盖的问题暴露出来, 不是修复引入的。**

**教训**: 得出"某测量与另一测量不一致"的结论前, 必须先确认**两个测量都可信**。
我这次连续两次把不可信的点云测量当成了基准。

## Q26: 日志里 "跳过 'box_0': 支撑物 'table_0' 未找到" 是什么意思？ (2026-10-04)

**问**: egodex_1 跑完后 5.2 出现 `跳过 'box_0': 支撑物 'table_0' 未找到`，
box_0 的位置就没被修正，为什么？

**答**: **VLM 幻觉出了场景里不存在的支撑物**。Stage1 检测到的类别是
`[box, cube, marker, phone, plier, pop-it, rubik's cube, tool, toy]`,
**没有 table**; 但场景图推断时 VLM 给 box_0 投了 2 票 `supported by table_0`
成为最高票。

下游 `refine_inter_object_placement` 遇到支撑物找不到时只 `continue` 打印"跳过",
**不会修正关系** —— 于是 box_0 的位置精修被静默丢弃, 物体停在错误位置。

**→ 修复**: 在 `infer_relations_scene_graph` 的 Step 3 之前加 **Step 2.5 支撑物
存在性校验**: 支撑物必须属于场景实际存在的实例集合, 不存在则剔除该票并用其他
有效票重新投票, 无其他有效票则回退 Stage1 原关系。绝不保留幻觉支撑物。
实测 box_0 正确回退, 该物体不再丢失精修。

**顺带发现 (未改)**: 三集里 "supported by other objects" 占比都偏高
(7/ 6 / 2 个)。这些物体实际都在桌/箱子上, 说明 VLM 常常只能给出"某个物体"
而给不出具体名字。这种关系下游视作合法支撑对但无法解析到具体实例, 精修同样
不生效 —— 属独立议题(需要让 other objects 能被解析到具体支撑物)。

## Q27: Stage6 材质标注一直显示"未找到任何裁剪图"，耗时 0.0s？ (2026-10-06)

**问**: 批量跑 egodex 时发现 Stage6 每次都是
`未找到任何裁剪图，使用默认物理参数` + `Stage 6 (材质物理标注) 0.0s (0.0%)`，
但 `optimal_frames/` 目录里裁剪图明明存在。

**答**: **key 层级不匹配**。
- `mainv3` 传给 `run_stage6` 的是 `refined_relations`，而Stage5 之后它已经是
  **实例级** 的 key (如 `banana_0`、`sheet_1`)
- `material_annotator` 用这些 key 过滤 `optimal_frames` 文件名，而文件名里的
  类别段是**类别级** 的 (`banana_inst0_frame33.jpg` → `banana`)
- 实例级 key 与类别级文件名**交集恒为空** → 裁剪图全被过滤 → 走默认物理参数，
  Stage6 等于没跑

**注意**: 裁剪图存在却报"未找到"，因为过滤条件不匹配，不是文件缺失。

**→ 修复**: 比对时把 key 的实例后缀 `_N` 剥掉再匹配。
实测 egodex_15 匹配到 10 张、egodex_13 匹配到 7 张（修复前均为 0）。

**诊断弯路**: 先怀疑正则 `(\w+)_inst(\d+)_frame\d+\.jpg` 不匹配含空格/连字符的
类别名 (如 `rubik's cube`、`pop-it`)，实测这两个数据集的类别名都是纯字母数字,
正则正常匹配 —— 假设被否证。**真正原因要靠"实际跑一遍匹配逻辑"定位, 不能靠读代码猜。**

## Q28: 双卡并行跑 egodex 时报 temp_frames_stage2 文件不存在？ (2026-10-06)

**问**: 双卡并行批量跑时出现
`❌ 错误: [Errno 2] No such file or directory: '.../temp_frames_stage2/frame_000_vid1.jpg'`

**答**: **临时目录没有进程隔离**。
`generate_scene_json_stage1.py` 的 `--temp_dir` 默认值是硬编码的相对路径
`./temp_frames_stage2`。两条管线 (GPU0/GPU1) 同时跑 Stage1 时, 一个进程清理/
覆盖了另一个进程的帧文件, 于是 `shutil.copy2` 找不到源文件。

影响: 异常被捕获, Stage1 仍能完成 (实测那次的 egodex_19 报出 4 个物体),
**不致命但会丢帧** —— 多实例场景下可能漏检, 且属于随机性故障难复现。

**→ 修复**: 默认临时目录按 PID 隔离 (`./temp_frames_stage2_{os.getpid()}`),
多进程并发时各用各的目录。py_compile 通过, `os` 已在该文件导入。

**注意**: 修复前的并发运行结果理论上可能有丢帧, 严格来说受影响的那几个场景
需要重跑才能确保无漏检。

---

## 附: 双卡并行实测汇总 (2026-10-06)

14 个 egodex 场景全部 rc=0 完成。Stage6 修复后的实测:
- 裁剪图匹配: egodex_15 匹配到 10 张、egodex_13匹配到 7 张 (修复前均为 0)
- 日志里 "未找到任何裁剪图" 计数降为 0

---

## Q29: 3 个场景 SAPIEN 渲染出来全是纯灰, 物理却是正常的, 为什么？ (2026-10-08)

**问**: `7_vggt_omega`、`phone_vggt_omega`、`egodex_0_rescaled` 跑
`tools/stage61_physics_sim.py` 时, 物理明明跑了 (物体 z 有真实位移)、actor 也
装进去了 (含 RenderBodyComponent)、相机位姿读回正确、mesh 有效, 但 `get_picture`
拿到的画面**恒为纯灰**, 连额外加的 probe 测试方块都看不见。换 CPU/GPU PhysX、
换 mesh、换物体数量、整目录克隆到 /tmp 都一样。

**答**: 这是 **SAPIEN 渲染引擎内部与实体位姿的同步脱钩**, 不是数据问题也不是硬件
问题 (实测 GPU0/1 各有 38~47GB 空闲)。穷尽 6 组对照实验后确认与 mesh 内容、
物体数量、坐标正负、URDF 结构均无关 —— 根因在渲染引擎内部, 继续挖成本过高。

**→ 绕过方案 (方案 B)**: **物理与渲染解耦**。SAPIEN 只跑物理并逐帧记录位姿,
pyrender 离线重建渲染:

```bash
# 单场景
python tools/stage61_physics_sim.py --output_dir output_v2/<scene> --pyrender_render
# 批量 (含全部 8 个场景)
PYRENDER=1 bash tools/stage61_batch_run.sh 120
```

关键点:
- 完全不创建 `SapienRenderer`, 也不加任何 visual 组件 → 从根上避开该缺陷
  (物理本身不受影响, `sapien.Scene()` 默认后端实测就是 `PhysxCpuSystem`)。
- 相机用真实 VGGT `w2c` 直接构造 (`c2w = [R.T | -R.T@t]`, 再乘
  `diag(1,-1,-1,1)` 做 OpenCV→OpenGL 的 Y/Z 翻转), 保留了相机 roll,
  比原来 "由 pos+fwd 反推 look-at" 更准。
- 用**真实内参** (fx/fy/cx/cy) 构造离轴 OpenGL 投影, 分辨率取真实视频尺寸;
  原先硬编码 fovy=50° + 640x480, 与真实相机无关。

**实现中踩的两个坑** (都已在代码注释里标明):
1. 投影矩阵 `P[1,1]` 必须取 `+2fy/h`。写成负号会让画面绕主点**上下翻转**,
   表现为地面出现在画面上方、背景出现在下方 —— 容易被误判成 "相机位姿错了"。
2. pyrender 0.1.45 的 `Scene.add_node()` **不返回 node** (只有 `Scene.add()` 返回),
   写成 `node = scene.add_node(...)` 会得到 `None`, 随后 `set_pose` 报
   `Node must already be in scene`。必须自己持有 `Node` 再 `add_node`。
   (另: `set_pose` 逐帧改位姿是生效的, 早前 "改 node.matrix 不生效" 的记录
   只针对直接赋 `node.matrix`, 不适用于 `Scene.set_pose`。)

**顺带修掉一个老 bug**: MP4 里红蓝互换 —— `frames` 是 RGB 而 `VideoWriter`
期望 BGR, 两条渲染路径都补了 `cvtColor(f, COLOR_RGB2BGR)`。

**结果**: 8 个场景统一出 GIF, 3 个原失效场景的彩色像素占比中位分别为
3.2% / 58.5% / 24.3% (判据为 >0.5%), 均通过自检。其中 `phone_vggt_omega` 有
37 个 "空帧" (索引 83~119) **不是渲染失败**: 那段真实视频是手持手机特写
(相机离被摄物极近), 而 sim 里手机已落地离场, 画面只剩地面 —— 已记入报告的
`empty_frame_indices`。

### ✅ Q30-1: 上述 "Stage 6 与 6.1 断链" 已解决 (2026-10-08)

**根因最后一公里**: Stage 6 输出 key 是 `bowl_inst0`, Stage 6.1 的 URDF link 名
是 `bowl_0` —— 两边命名约定不一致, 而全仓库没有任何地方做过这两个 mapping。

**→ 修复** (`tools/stage61_physics_sim.py`):
- `resolve_material()`: `bowl_0` → `bowl_inst0`。命中优先级: 精确 → 该类别材质
  完全一致时按类别取值 → 放弃(宁可不用也不张冠李戴)。
- `build_actors` 的 `PhysxMaterial` 改为读 Stage 6 的
  `static_friction / dynamic_friction / restitution`, 不再写死 0.60/0.50/0.02。
- 报告新增 `materials_stage6` 段记录覆盖率与逐物体取值。

**关键设计决定 —— 质量不跟着换**: 直觉上应该把质量也换成 `密度×mesh体积`,
**实测证明不能这么做**: `newspaper_1` 会算出 10.9kg、`table_0` 35kg。密度质量 /
VLM 质量的中位数是 **5.03x**, 换成线性尺度就是 **≈1.71x** —— 与
`docs/SAM3D_SCALE_DRIFT_REPORT.md` 里 "SAM3D ~1.5x 绝对尺度漂移" 量级吻合,
说明是 mesh 整体偏大, 而不是密度不该用。所以:
- **摩擦 / 恢复系数: 无条件采用 Stage 6** (不依赖绝对尺度, 纯增益)
- **质量: 默认 `vlm`, `--mass_source density` 可选**, 并把两者比值做成
  **尺度漂移诊断量** `implied_linear_scale_error` 写进报告。

**验证 (受控 A/B)**: `151_C7_Plate_214deg_vggt_omega` (8 个目标场景里唯一有
materials.json 的), 材质命中 **29/29 全精确**; 仅开关 `--no_materials`, 初始 z
完全相同而最终位姿合计 |Δ| = 0.2545 m, 最大单物体 Δ = 5.9cm (`bottle_1`,
玻璃 0.4/0.4 vs 硬编码 0.6/0.5 → 滑得更远)。**Stage 6 不再是摆设。**

**剩余前提**: 另外 7 个场景当时没跑 Stage 6, 没有 materials.json, 会自动退回旧
行为并打印 WARN; 补跑 `mainv3 --enable_stage6` 后**无需改代码**即自动生效。

**遗留 (上游数据问题, 非本次引入)**: `151_C7_Plate` 物体在画面里偏小且靠边
(彩色中位 12.4%)、`082` 相机贴得很近, 属 stage2 对齐 / stage4 重建精度问题;
pyrender 路径只是把真实相机位姿如实呈现出来。

---

## Q30: 现在的 Stage 6 流程是什么？最终产出什么？有没有 HTML 渲染？ (2026-10-08)

**问**: Stage 6 现在跑到哪一步了, 最终生成的东西是什么, 还有哪些问题,
有没有 HTML 可视化。

**答**: 要先分清两个同名不同体的东西 —— **Stage 6** 和 **Stage 6.1**。

### 1. Stage 6 = 材质感知的物理参数标注 (`tools/material_annotator.py`)

- 是 `mainv3.py` 管线的**最后一步**, 但**默认关闭**, 要 `--enable_stage6` 才跑
  (`mainv3.py:31` 抬头说明, `:1585-1596` 唯一调用点, `:1749` argparse)。
- 流程: 读 `optimal_frames/*.jpg` 裁剪图 → 每个类别问一次 VLM 判断材质 →
  查 `materials_db.json` 映射成密度/静摩擦/动摩擦/恢复系数。
- **最终产物**: `<scene>/materials.json` + `<scene>/materials_debug.json`
  (`material_annotator.py:472-475` / `:486-489`)。
- ⚠️ **任何一步失败都会静默走 `_fallback_default_materials()`** (`:309/344/361/
  434/441`), 退化为全默认参数且不阻塞管线 —— 所以 "Stage 6 跑完了" 不等于
  materials.json 里有真实逐物体材质。历史 bug Q27 就是裁剪图 key 层级不匹配导致
  整段退化成 0 张匹配。

### 2. Stage 6.1 = URDF 生成 + SAPIEN 物理仿真 (**完全手工**, 未接入 mainv3)

| 步骤 | 脚本 | 产物 |
|---|---|---|
| GLB 拆解 → 逐物体 URDF → 合并 | `src/stage61_urdf_generator.py` | `urdf/scene.urdf`、`urdf/mesh/`、`urdf/<obj>/{xxx.urdf,mesh/,urdf_renders/}`、`urdf/stage61_report.json`、`urdf/stage61_validation.json`、`urdf_sim/scene_sim.urdf`+`urdf_sim/mesh/` |
| 物理仿真 + GIF | `tools/stage61_physics_sim.py` | `stage61_physics.gif` / `.mp4` / `_frame0.png` / `_report.json` |

- 输入是 **Stage 5 的 `final_scene_stage5.glb`** (`stage61_urdf_generator.py:740`
  `--glb` 必填), **不读 Stage 6 的 materials.json**。
- `mainv3.py` / `mainv2.py` 全文搜不到 `stage61` / `urdf`, `run_stage61()` 只被
  自己的 `__main__` 调用 (`stage61_urdf_generator.py:750`)。`--enable_stage61`
  至今只是旧计划文档 (`docs/design/stage61_urdf_plan.md`, 现已合并为存根) 里的计划, 没实现。
- `tools/` 下 5 个早期脚本已被弃用: `stage61_sim_sapien.py` / `render_pyrender.py`
  / `render_scene.py` / `fixed_camera.py` / `visualize_sapien.py`
  (共同缺陷: 不读 mass/inertia、或干脆不 step 物理、或用彩色 box 占位)。
  当前实际在用的只有 `stage61_urdf_generator.py` + `stage61_physics_sim.py` +
  `stage61_batch_run.sh`。

### 3. 还有的问题

1. **Stage 6 与 6.1 断链** (最大问题): Stage 6 辛苦标注的 `materials.json`
   **没有任何消费者**。URDF 里摩擦系数是 EmbodiedGen 硬编码占位
   (`<mu1>0.60</mu1><mu2>0.50</mu2>`), 质量/惯量也是占位 (`ixx=iyy=izz=1.0`),
   只能由 `stage61_physics_sim.py::recompute_inertia` 按 mesh 现算。
2. **未接入主管线**: 每次都要手工两步跑, 换场景还要手改批量脚本里的场景列表。
3. **SAPIEN 渲染确定性失效**: 3 个场景画面恒为纯灰 → 已用方案 B
   (物理+渲染解耦, pyrender 离线渲染) 绕过, 见 Q29 / CHANGE_LOG 2026-10-08 23:00。
4. **环境分裂**: 批量脚本用 `/mnt/data/lza/conda_envs/sapien` (sapien 3.0.0b1),
   而旧文档 (`stage61_urdf_plan.md`, 已合并) 写的是 `ReplicateAnyScene` (sapien 3.0.3)。
   现行口径见 `stage6_physics_sim.md` §6: ① URDF 用 `ReplicateAnyScene`, ②③ 物理+渲染用 `sapien`。
5. **mesh 命名双前缀**: `urdf/mesh/block_0_block_0.obj` (类别+实例名重复拼接)。
6. **上游对齐精度**: `151_C7_Plate` 物体在画面里偏小偏边、`082` 相机贴得过近,
   属 stage2 对齐 / stage4 重建问题, 仿真脚本如实呈现但无法修正。

### 4. HTML 渲染: **当前代码没有任何 HTML 产出**

- 全仓库 96 个 .py 搜 `.html` / `plotly` / `viser` / `trimesh scene export` →
  **0 命中** (仅在第三方 sam3/vggt 的 URL 注释里出现), 也没引入相关依赖。
- 磁盘上确实残留 2 个 HTML 文件, 但**生成脚本已从仓库删除**, 属孤儿产物:
  `output_v2/egodex_gallery.html`、
  `output_v2/7_vggt_omega/trajectory_overlay/trajectory_gallery.html`
  (后者由已删除的 `tools/analysis/make_trajectory_gallery.py` 产出,
  见 CHANGE_LOG 2026-10-06 22:40)。
- 所以准确说法是: **Stage 6/6.1 的产物全是给人看的 GIF/MP4/PNG 和给机器读的
  JSON/URDF/OBJ, 没有 HTML 环节**; 想做的话建议补一个像古人轨迹画廊那样的
  单文件汇总页 (8 个场景并排 GIF + 报告里的关键指标), 成本很低。

## Q31: Stage6 之外（Stage1~5）还有哪些残留问题？ (2026-10-09)

**问**: 批量生产跑了 28+ 个场景，除了 Stage6 的问题，其他阶段还有没有隐藏问题？

**答**: 对 mainv3.py、geometry_utils.py、sp_refinement.py、infer_relations_scene_graph.py、generate_scene_json_stage1.py 做了静态审计，排除已修复项（on_top 重贴、落地补正、穿模发散、tie-break、FLOOR_Z_MAX）后，发现 10 项，前 3 项已逐行实证确认：

### 高严重度（3 项，已确认）

1. **动态轨迹 PLY 只剩最后一帧** — mainv3.py:791-816
   ```python
   ply_filename = f"{category}_inst{inst_idx}_dynamic.ply"
   for fi, (fid, pts) in enumerate(zip(frame_ids, frame_pts)):
       with open(ply_path, 'w') as f:   # 每帧覆盖, 固定文件名
   ```
   本意是红→蓝渐变表示时间顺序的完整轨迹，实际 `dynamic_pointclouds/*.ply` 只有最后一帧的 5000 点。影响：动态轨迹下游可视化/分析拿到的数据是错的，且无任何报错。

2. **Stage3.5 验证删除实例后索引不同步** — mainv3.py:967 + asset_verifier.py:275
   `verify_all_instances` 返回删除坏实例后的 `all_instances`，但 `all_optimal_frame_ids` / `deduplicated_all_masks` 保持原数量。下游 mainv3.py:1468 `zip(all_optimal_frame_ids[category], category_instances)` 错位：被删的若是实例 0，幸存实例会拿到别人的 optimal_frame_id（attached_to_wall 的 camera_pos 用错帧、rescale 取错 opt_f、5.1 为 ghost 实例推关系）。

3. **Stage1 关键帧 timestamp 用错帧号基准** — generate_scene_json_stage1.py:312 + 2379
   `timestamp = frame_idx / fps`，其中 frame_idx 来自 VGGT 采样网格（如 120 帧），fps 是原始视频帧率。视频 2379 帧时所有"关键帧"的 timestamp 落在片头 4.8 秒内；`frame_idx >= total_frames` 的钳制 (L309) 也因此永远不触发。VLM 看到的帧与标注的"实际视频时间"完全错位。

### 中严重度（4 项）

4. **5.1 投票用错帧网格** — infer_relations_scene_graph.py:531,676,699：Stage1 元数据的 vid_idx（120 帧网格）直接作 Stage2 colors（160 帧网格）下标取帧，VLM 看的帧与可见性标注错位，代码只防越界不防错位。
5. **最优帧按文件名子串匹配** — refine_inter_object_placement.py:2298,2303："car" 命中 "toy_car_inst0"，同类别多实例恒取 inst0；匹配不到时回退取目录第一张（别的物体的帧）供 VLM 判"是否放对"。
6. **实例名双向子串首中即断** — infer_relations_scene_graph.py:537-541,683："table" 误配 "tablecloth_0"。
7. **同类别多实例校准串用点云** — rescale_meshes_by_pointcloud.py:151-157：按"opt_f 上 mask 最大"从 pre-dedup 片段猜实例来源，同类别多实例常选中同一片段，互相用对方点云校准尺寸。另 mainv3.py:790 `total_frames = len(frame_ids)` 遮蔽 657 行全视频帧数，visible_ratio 分母错（现仅污染日志）。

### 低严重度（3 项）

8. mainv3.py:1463 `refined_relations if 'refined_relations' in dir()` — 该变量 1529 行才定义，条件恒 False，死分支。
9. mainv3.py:1316,1545 stage5 位姿记录按 category 查询却传入实例级 key，relation 恒 "unknown"。

**已核实非问题**: 7a 日志 `stage1_strict (R[2,2]=-0.4286)` — stage1_strict 有内部质量门（平面拟合残差<0.02m、墙正交 5° 内），R[2,2] 只反映 VGGT 原始坐标系朝向，不表示对齐失败。

---

### Q19: FP track 的 pose 估计轨迹抖动严重（静止物体路径累积 3~7m），有什么好的修复方法？

**回答**: 对比验证了 4 类方法，结论如下（数据见 `output_v2/7_vggt_omega/fp_static_lock_eval.json`，验证脚本 `tools/analysis/fp_static_lock_eval.py`）：

**方法对比**：

| 方法 | 结论 |
|---|---|
| A. 提高 track_refine_iter (2→5) | 未测（推理翻倍，只能缓解不根除） |
| B. 静止段锁定位姿 | **有效，推荐**。8 实例 path 1.9~7.1m → 静止物体全部归零，IoU 保持 |
| C. mask 质心逐帧校正 | **有害，弃用**。mesh 质心投影 ≠ mask 质心（mesh 遮挡缺失/边界差），每帧强行对齐把 plate IoU 0.887→0.145 |
| D. 段内中值锚定 | **有害，弃用**。FP track 平移是长随机游走（plate 在 0.3m 范围游走，中值离真值 16cm），中值≠真值 |

**方案 B 正确实现要点**（两个关键踩坑）：
1. **运动判定**：用 GT mask 质心+depth 反投影到世界系，滑动窗口净位移阈值判定（win=9、thr=5cm 最优）。阈值 2cm 时反投影噪声误触发一半帧为"运动"
2. **锚点选择**：静止段必须锚定**段首**位姿（= register 或运动结束位姿，register 首帧 IoU 0.40~0.97 全部准确），不能用段内中值/四元数平均（随机游走有偏）

**锁定后效果**（win=9, thr=0.05, anchor=first, --no_correct）：

| 实例 | FP原始 path/IoU | 锁定后 path/IoU | GT净位移 |
|---|---|---|---|
| plate_inst0 | 1.920m / 0.887 | **0.000m** / 0.848 | 0.006m（真静止）|
| block_inst2 | 5.608m / 0.467 | 2.360m / 0.372 | 0.048m |
| toy_inst0 | 5.214m / 0.653 | 3.030m / 0.659 | 0.347m（真运动）|
| toy_inst1 | 7.105m / 0.318 | 4.442m / 0.327 | 0.347m |

**剩余问题**：真实运动的物体（toy_inst0/1）在运动段内 FP 抖动仍在（锁定后 path 3~4.4m vs GT path 1.1m，其中 GT path 本身含反投影噪声）。运动段的进一步改善方向：提高 track_refine_iter、运动段平移 EMA 平滑、或运动中周期性 re-register。

**与 mainv3 的集成点**：stage3 已有动态判定（质心/重叠方法），直接复用其静止/运动分段，静止段用 FP register 位姿填充，运动段用 FP track——即 stage3.5 插入点（stage3 之后、stage4 之前）。

### Q20: FoundationPose 跟踪工作目前进展到哪里？产物在哪里？还有什么问题？

**回答**:

**已完成**（7_vggt_omega 场景，8 个去重后实例）：
1. 全链路跑通：mask 自动匹配（register IoU 判据，`fp_mask_select/<obj>.json`）→ FP register+track 113 帧 → 世界系转换（ob_in_world）→ IoU 评测
2. 首帧 register 全部准确（IoU 0.40~0.97），链路正确性由 plate_inst0 全帧 IoU 0.887 背书
3. 抖动修复方案定型：静止段锁位姿（win=9/thr=5cm/段首锚定），plate path 1.92m→0，详见 Q19
4. 2D+3D 合一 GIF：8 实例 combo.gif 已生成

**产物位置**：
- 可视化目录：`ReplicateAnyScene/test/fp_vis/index.html`（索引页）+ `<obj>/combo.gif`（2D+3D 合一）+ `<obj>/small.gif` + `traj.png`
- 原始输出：`output_v2/7_vggt_omega/fp_data_<obj>/`（输入）、`fp_out_<obj>/`（位姿+IoU 报告）、`fp_vis/<obj>/`
- 汇总数据：`fp_batch_report.json`（批量评测）、`fp_static_lock_eval.json`（锁位姿验证）
- 工具脚本：`tools/analysis/fp_prepare_data.py`、`fp_select_mask.py`、`fp_run_tracking.py`、`fp_iou_eval.py`、`fp_batch_eval.py`、`fp_overlay_batch.py`、`fp_gif_2d3d.py`、`fp_static_lock_eval.py`

**遗留问题**：
1. **运动段内 FP 抖动仍在**：锁定后 toy_inst0/1 path 仍 3~4.4m（GT 1.1m），运动段需 EMA 平滑或周期性 re-register
2. **未集成 mainv3**：建议插入点 stage3 之后 stage4 之前，复用 stage3 动态判定分段，跨双 conda 环境子进程调用
3. **跨场景未验证**：2chair-bottle/3chair-bottle 仅准备了数据（fp_data_bottle0 100 帧），未跑完整流程
4. **index.html 未更新**：卡片仍只链 small.gif，未加 combo.gif 入口
5. index.html 需用系统浏览器或 `python -m http.server` 打开（IDE 内置预览注入 Vite 代理会 abort 静态页）

### Q21: 运动段内的 FP track 抖动如何解决？修复效果如何量化？

**回答**:

**方法**：在方案B（静止段锁位姿）基础上增加**运动段 savgol 平滑**，组合成完整修复管线（`fp_static_lock_eval.py`）：

1. 静止段：GT mask 世界系轨迹滑动窗口（win=9, thr=5cm）判定 → 锁定段首位姿（`lock_static`，anchor='first'）
2. 运动段：段内平移逐轴 savgol（win=21, polyorder=3）+ 旋转四元数（符号对齐后）savgol（`smooth_moving`，fp_static_lock_eval.py L154-179）

窗口选择依据（路径长度扫描，toy_inst0/1）：
| 方案 | toy_inst0 | toy_inst1 |
|---|---|---|
| 原始 FP | 5.21 | 7.10 |
| 仅锁静止 | 3.03 | 4.44 |
| savgol w9/w15/w21/w31 | 2.00/1.66/1.54/1.40 | 2.85/2.38/2.20/2.28 |
| 中值9+savgol w21 | 1.43 | 2.08 |

选定 **w21 单遍**（简单够用；w31 toy_inst1 反而回升；中值组合仅再降 0.1m，不值得加复杂度）。净位移（真实 pick-place 运动，GT 0.347m）在所有窗口下均保留（0.764/0.922m，由段锚点决定与窗口无关）。

**最终量化对比**（path 单位 m，IoU stride=2 光栅化）：

| 实例 | FP原始 path/IoU | 修复后 path/IoU | GT path/net | 运动帧 |
|---|---|---|---|---|
| plate_inst0 | 1.920/0.887 | **0.000**/0.848 | 0.280/0.006 | 0/113 |
| toy_inst0 | 5.214/0.653 | **1.539**/0.650 | 1.123/0.347 | 53/113 |
| toy_inst1 | 7.105/0.318 | **2.204**/0.317 | 1.123/0.347 | 53/113 |
| block_inst2 | 5.608/0.467 | **2.360**/0.372 | 0.389/0.048 | 11/113 |
| block_inst0 | 3.129/0.263 | 2.027/0.248 | 0.482/0.084 | 40/113 |
| block_inst1 | 5.037/0.162 | 3.436/0.141 | 0.482/0.084 | 40/113 |
| toy_inst2 | 4.569/0.146 | 3.908/0.153 | 0.475/0.083 | 41/113 |
| toy_inst3 | 3.867/0.107 | 2.135/0.157 | 0.390/0.048 | 10/113 |

结论：高 IoU 实例（plate/toy_inst0/1）抖动压掉 55%~100% 且 IoU 持平（±0.003）；block_inst2 IoU 降 0.095（低 IoU 实例本身位姿不准，平滑无从补救）；toy_inst3 IoU 反升 0.05。

**产物**：
- 修复后位姿：`fp_out_<obj>/ob_in_world_locked/{t:06d}.txt`（不动原始 ob_in_world）
- 修复后可视化：8 实例 combo.gif 已用 `fp_gif_2d3d.py --poses ob_in_world_locked` 重生成并同步 `test/fp_vis/<obj>/`
- 汇总数据：`fp_static_lock_eval.json`

**运行命令**：
```bash
python fp_static_lock_eval.py --scene_dir <SD> --win 9 --thr 0.05 --no_correct --anchor first --smooth_win 21 --save_locked
python fp_gif_2d3d.py --scene_dir <SD> --poses ob_in_world_locked
```

### Q22: 为什么对静止物体也跑了 FP 跟踪？3D 面板里物体和轨迹为什么是分开的？

**回答**:

**问题1（对静止物体也跑了跟踪）**：之前的批量脚本直接遍历 `fp_batch_report.json` 的全部 8 个实例，没有接主流程 stage3 的动态判定。实际上 `pick_place_frames_combined.json` 判定本场景**只有 toy_inst1 是动态物体**（pick 46 / place 97），其余 7 个（3 block、plate、toy_inst0/2/3）都是静止的——对静止物体跑 track 只会得到抖动漂移的垃圾轨迹，还会"莫名其妙识别不同的东西"（FP 在低纹理/相似物体间漂移，IoU 0.1~0.3 的实例即此表现）。

修复：`fp_gif_2d3d.py` 和 `fp_static_lock_eval.py` 新增 `dynamic_keys()`，默认只处理 stage3 判定为动态的实例；`--objs` 可指定、`--all` 取消过滤。

**问题2（3D 面板物体和轨迹分离）**：3D 面板里青色 mesh 点云画在位姿变换后的真实位置，但橙色 FP 轨迹画的是 `ob_in_world` 的**平移分量 t**（= mesh 坐标系原点在世界系的位置）。toy_inst1 的 mesh.obj 原点离几何中心 **0.622m**（centroid=[-0.432,-0.446,0.041]），plate 也偏 0.147m——所以轨迹整体偏离物体一团点云。

修复：轨迹改画 mesh 质心轨迹 `c_w(t) = R(t)@c + t(t)`（fp_gif_2d3d.py L102-105），与 GT 轨迹（mask 质心反投影，本来就是物体视觉中心）同一参考点，两者才可比。

**清理**：test/fp_vis 下 7 个静态实例目录已删除，只保留 toy_inst1。场景目录 output_v2 下的 fp_data_*/fp_out_* 原始产物未动。

**关于跟踪精度**：toy_inst1 修复后 path 2.20m / IoU 0.317（GT 1.12m/0.347 净位移），残余偏差是 FP track 本身在运动段的漂移，离线平滑只能压高频抖动，无法纠正低频漂移；若要更准需周期性 re-register 或换更强的跟踪器。

### Q23: GIF 里红色/绿色轮廓是什么？SAM3 逐帧都跟着物体，为什么 FP 还会检测到别的物体？有没有方法改进？其他视频测过吗？

**回答**:

**颜色含义**（fp_gif_2d3d.py）：2D 面板红轮廓 = SAM3 GT mask（物体实际位置），绿轮廓 = FP 位姿下 mesh 投影（跟踪结果），橙点 = 投影质心历史尾迹；3D 面板青点云 = 物体 mesh，橙线 = FP 质心轨迹，绿线 = GT 质心轨迹，红三角 = 当前相机。

**为什么会锁到别的物体**：fp_run_tracking.py L105-107 显示 SAM3 mask **只在帧 0 的 register 用过一次**（`est.register(..., ob_mask=mask)`），后续 track 循环是纯 RGB-D render-and-compare——FP 用自己上一帧渲染结果与当前帧匹配更新位姿，**没有用 SAM3 逐帧 mask**。位姿一旦滑动，render-and-compare 会锁到相似物体上，无约束拉回。SAM3 逐帧 mask 在管线中仅用于事后 IoU 评测，未参与跟踪。

**改进方法**（按性价比）：
1. mask 引导周期性 re-register：每隔 N 帧 / pick-place 边界（stage3 已给出帧号）用当帧 SAM3 mask + depth 重新 register，复用现有 register 代码，改动最小
2. track 每帧输入用 SAM3 mask 裁剪的 RGB-D
3. mask 反投影质心做平移软约束

**其他视频**：尚未测试。除 7_vggt_omega 外有 stage3 动态判定的场景：116_C5_CellPhone_185deg_vggt_omega（2 交互）、121_C5_CellPhone_161deg_vggt_omega（2）、2chair-bottle_vggt_omega（2）、3chair-bottle_vggt_omega（3）。工具链（fp_prepare_data → fp_select_mask → fp_pick_mesh → fp_run_tracking → fp_gif_2d3d）齐备，计划只跑各场景动态实例。

## Q32: LiteReality-Agent 能否替代 stage5 精修？agent 部分能否用 opencode？ (2026-10-13)

**回答**:

调研对象：[LiteReality-Agent](https://github.com/LiteReality/LiteReality-Agent)（Cambridge，arXiv 2610.01863，2026-10-01 发布）。核心思想是把 3D 重建变成 coding 任务：agent 迭代编写 `Room.py`，编译成 Blender/GLB/URDF/MJCF，MuJoCo 直接可仿真；通过 observe-edit-verify 循环（render 对比实拍 / critic 打分 / check_collisions 碰撞检查等 capability tool）收集证据反复修正。

### 结论：不能整体替代，但精修层（layout/QC）与 stage5 完全对位，可移植

| 本项目 stage5 | LiteReality-Agent 对应层 | 对比 |
|---|---|---|
| 5.1 VLM 投票 supported-by | `check_support`：footprint 下方最高顶面 / floor / wall anchor，纯确定性几何，强制唯一支撑者+链到地+无环 | 后者无需 VLM；VLM 只负责缺支撑物时补建模 |
| 5.2-② SP 贴合（minor 档 0.15m 限幅） | support snap：Z 平移消 gap，root-first 链式，**同样是 0.15m cap**，超出升级给 agent | 设计几乎一致，限幅值相同 |
| 5.2-③ 穿模迭代 | `resolve`：detect→nudge→detect；每个 clash 自带 metric 修复（world_dxdy）；接触由真实三角形判定，凸分解只算距离不算接触 | 解决了 121 场景"8 轮打地鼠"同款问题，更严谨 |
| 5.2-④ check_stability | 编译质量/惯性/collider 的 MuJoCo 场景 + shake test | 从几何启发式兜底升级为真实物理验证 |

### 不能整体替代的原因
1. **输入契约不同**：它要求 iOS 扫描 RGB-D + RoomPlan `room.usdz`；本项目是 ego video → VGGT 实例 mesh（all_instances.pkl），无深度、无房间结构化先验
2. **物体来源不同**：它做资产检索/生成（TRELLIS + GroundingDINO，24GB GPU 或 Modal），产出 CAD；本项目核心价值是真实重建 mesh + FP 轨迹，换成检索模型会丢失
3. **依赖重**：Blender 5.x、图像生成 API key、内置 provider 仅 `claude`/`codex` 两个

### 推荐方案 A：移植 QC 层 + opencode 驱动精修循环
1. 引入 `room_qc` 的 `check_glb`（碰撞）/`check_support`（支撑）两道确定性门控（pip 安装该包调用或按其逻辑移植）；输入为带 extras 的 GLB——可用 all_instances.pkl 实例 mesh+位姿编译，floor/wall 类实例对应其"固定实体"
2. stage 5.1 VLM 关系推断保留作先验，support check 做几何校验/纠正（与 2026-08-25 修复的"VLM 管关系、几何管摆放"分工一致）
3. 精修循环 agent 化：渲染四视图 + findings → opencode agent 编辑位姿 JSON → 重跑 gate → 收敛出 final GLB。capability tools 注册为 MCP（opencode.json 配 stdio server），`opencode run` 非交互驱动，provider 用已有配置

### 方案 B（不推荐）：完整接入 + 写第三个 provider
`agent/providers/base.py` 是 seam（AgentHarness Protocol：SessionSpec 进、归一化事件流出），把 "opencode" 加进 `PROVIDERS`、实现 opencode provider 即可（capability tools 走 stdio MCP，codex 同款路径）。但输入契约改造量太大，不划算。

### opencode 作为 harness 的已知限制（与其对 codex 的处理一致）
- 无 PreToolUse hooks → step budget 退化为硬停（无 wind-down）
- 无 cost 上报；无工具白名单（shell 始终可用）
- LiteReality 的 `describe()` 已有降级提示模式，可照搬

## Q33: 方案 A 具体是什么？ (2026-10-13)

**回答**:

方案 A = 不引入 LiteReality-Agent 整个系统（它要求 iOS RGB-D 扫描 + TRELLIS 资产生成 + Blender 5.x），只把它的"精修层"（room_qc）移植过来，替换 stage5 中最痛苦的 5.2-② SP 贴合、5.2-③ 穿模迭代、5.2-④ check_stability，agent 换成已配好的 opencode。分四步：

1. **编译带 extras 的 GLB**：把 `all_instances.pkl` 的实例 mesh+位姿、`relations_scene_graph.json` 的关系编译成一个 GLB，分类（floor/wall/普通物体）、墙线等元数据写入 glTF `extras`——这是 LiteReality `room_qc` 的输入契约；floor/wall 实例对应其"固定实体"
2. **两道确定性门控（移植）**：
   - `check_support`：对每个物体找支撑者（footprint 下方最高顶面 / floor / wall anchor），量 gap → 报 `no_contact`/`embedded`/`unsupported`；修复 = Z 向平移（≤0.15m，root-first 链式）——等价于 5.1+5.2-②，但纯几何可复现
   - `check_glb`：真实三角形判接触 + 凸分解量穿透深度 → 每个 clash 附带 metric 修复向量（`world_dxdy`）——等价于 5.2-③，不会出现 121 场景"8 轮打地鼠"
3. **opencode agent 兜底循环**：门控解决不了的（如 `unsupported`——支撑物未重建出来）交给 agent：MCP 注册 `check_support`/`check_glb`/`render` 三个工具，opencode 看到渲染对比 + findings 后编辑位姿 JSON → 重新编译 → 重跑门控，循环收敛；`opencode.json` 注册 stdio MCP server，`opencode run` 非交互驱动
4. **输出 final GLB**，替代现有 5.2 产出

与现状的对应：5.1 VLM 关系推断保留作 `check_support` 先验（"VLM 管关系、几何管摆放"，与 2026-08-25 修复确立的分工一致）；`refine_inter_object_placement.py` 本体不动，保留作回退。改动集中在：新增一个精修入口脚本 + 移植两个 check 函数 + 一个 MCP server。

## Q34: check_support / check_glb 是 agent 操作吗？ (2026-10-13)

**回答**:

不是。两者是**纯确定性 Python 函数**（无模型、无 agent），能直接命令行跑、exit 0/1 判通过：

```bash
python -m litereality_agent.pipeline.room_qc.support --glb Room.glb    # exit 0 = pass
python -m litereality_agent.pipeline.room_qc.collision --glb Room.glb
```

修复也大部分是确定性的：支撑 gap → Z 向 snap；碰撞 → 按 findings 自带的 `world_dxdy` 平移，detect→nudge→detect 循环。几何问题用几何方法解最快、最可复现，LiteReality 的设计核心是**确定性优先、agent 兜底**。

方案 A 中 agent 角色因此很窄：门控确定性修复成功（预期覆盖绝大多数贴合与毫米级穿透）时 agent 完全不参与、不花 token；仅少数边缘情况出场。121 场景"8 轮打地鼠"的根因正是几何问题被错误参数反复硬推——确定性门控每个 clash 自带修复向量，两轮即收敛。

agent 只处理确定性修不了的两类情况：

| finding | 为什么确定性修不了 | 谁修 |
|---|---|---|
| `unsupported`（footprint 下面啥都没有） | 支撑物未重建（墙隔板/支架），要补物体不是移位，补什么要看照片 | agent 读参考帧补建模 |
| `capped`（需要位移 >0.15m/nudge 上限） | 上游摆放错误，硬推会把场景推坏 | agent 重新理解该物体位置 |
| `support_cycle`（A 压 B 压 A） | 布局语义错误无解 | 直接报人 |

## Q35: github.com/agentic-real2sim/agentic_real2sim 介入进来会不会更好？ (2026-10-13)

**回答**:

结论：**不适合介入，stage5.5 继续走 opencode agent 方案**。理由逐条核对：

| 维度 | agentic_real2sim | 本项目现状 |
|---|---|---|
| 输入契约 | 需要 **DROID 机器人平台轨迹数据**（多视角 RGB-D + 机械臂动作流），输入假设是机器人操作日志 | 我们输入是单目/稀疏视角视频 + VGGT 重建，没有机器人动作流 |
| 部署规模 | 5 个模拟环境（ManiSkill 等）合计约 80GB、GPU 服务化部署 | 单场景精修，无需重型仿真集群 |
| 架构定位 | **端到端管线**：视频→场景重建→物理仿真→agent 闭环生成动作轨迹 | 我们已有完整 stage1~5 管线，缺的只是"摆放语义精修"一层 |
| agent harness | 自带 agent 框架（非 opencode），与既有 opencode 工具链不兼容 | stage5.5 已用 opencode + stage55_cli 跑通 |

可借鉴的思想（作为后续增强项，而非现在引入）：
- **物理落座仿真验证**：把 stage5.5 修正后的场景丢进 SAPIEN 做一步 settle 仿真，看物体是否稳定落座（不滑落、不弹起），作为最终质量门控——这比纯几何 AABB 检测更能发现"看似不穿模但物理上必然翻倒"的摆放。
- 其余部分（场景重建、动作生成）与本项目重叠且输入不匹配。

因此不引入该仓库，仅吸收"物理落座验证"思想作为 stage5.5 的可选 stage5.6 增强。

## Q36: stage5.5 的 prompt 是如何参考 LiteReality-Agent 的？他们是每个数据生成一个 prompt 吗？ (2026-10-14)

**回答**:

不是 per-dataset prompt。LiteReality-Agent 和本实现的 stage5.5 都是**一段写死的通用 prompt + 运行时工具喂场景信息**的三层分离：

| 层 | LiteReality-Agent | stage5.5 实现 |
|---|---|---|
| Prompt（全场景共用） | 固定任务描述：迭代编辑 Room.py 直到渲染对齐实拍、门控通过 | `build_agent_prompt()`（run_stage55.py L59-108）一段固定文本 |
| 场景差异 | 不进 prompt，由工具输出进入：render vs 实拍对比、check_collisions findings、critic 打分 | 不进 prompt：check_glb/check_support JSON findings + render 四视图 + optimal_frames 实拍参考帧，agent 用 bash/read 自取 |
| 状态载体 | Room.py（改代码=改场景） | stage55_work.glb（set_xy=改场景） |

`build_agent_prompt()` 的 f-string 只做路径注入（work.glb、参考帧目录），无任何场景相关描述——泛化靠"prompt 只描述 observe→edit→verify 通用工作流"实现。

Prompt 里的"例子"（小物件搭在支撑物 A 边缘实拍在 B 上 / 家具地面互叠实拍前后错开）是**失败模式说明书**而非场景指令，用于向 agent 解释"check 报 ok 不代表语义对"这个检测盲区；2026-10-14 已从 121 场景特有表述（杯子/高桌/矮桌）抽象化以避免过拟合单场景。

分工遵循 LiteReality"确定性优先、agent 兜底"核心：snap（Z 贴合）/nudge（MTV 向量收敛穿透）为确定性代码，agent 只处理检测器看不见的语义错误（放错支撑物、家具应错开）。后续增强：① stage5.6 物理落座验证（SAPIEN settle，对应其 MuJoCo shake test）；② opencode 无 PreToolUse hooks 导致轮次硬停，需提示词"执行完才能结束"缓解（GLM-5.2 两次失败根因）。

**Q36 补充（2026-10-14，prompt 原文核实）**：直接读取 LiteReality-Agent 仓库 `src/litereality_agent/agent/author.py` L68 的 `PROMPT` 常量确认：其 prompt 是一段写死的通用 SOP（角色定义 → "Two jobs, IN THIS ORDER"（先墙体结构后材质 FLOOR→WALLS→CEILING，再补墙挂件）→ 工具用法（render/critic/fetch_material）→ 约束（Edit ONLY Room.py、Be CONSERVATIVE and evidence-based）），**零场景内容**：无物体名、无坐标、无"移动某物"指令；场景信息靠 agent 首步 "Read Room.md and Room.py IN FULL" + head-on stitches 照片（`{stitch_lines}` 仅路径注入）+ render/critic 工具输出进入。注意细节：他们的 prompt 反而**禁止移动家具**（"do NOT move/resize the furniture object boxes"，家具来自 RoomPlan 扫描可信），agent 只管墙体/材质/挂件；stage5.5 相反——mesh 可信、摆放不可信，agent 只管 set_xy。任务分工相反、机制同构。另有 4 个 profile（base/detail/simulation/open）按任务档位分 prompt 变体，非按数据集分。

### Q24: mask 引导 re-register (--rekey) 改进效果如何？跨场景测试结果？

**回答**:

**方法**（fp_run_tracking.py 新增 `--rekey N`）：旧管线 SAM3 mask 只在帧 0 register 用一次，后续 track 是纯 RGB-D render-and-compare，无 mask 约束 → 遮挡后漂移锁到相似物体（袖子等）。改进：每隔 N 帧（默认 15）且当帧有 SAM3 mask 时重新 `est.register`（mask+depth），register 失败回退 track。mask 逐帧保存于 `fp_data_<obj>/mask/`（113 帧全齐，遮挡时面积变小但 SAM3 不丢）。

**toy_inst1（7_vggt_omega）**：IoU 0.318 → **0.523**（+64%），min 0.224。质心轨迹 path 1.71m vs GT 1.12m。旧结果备份 `fp_out_toy_inst1_orig/`。

**跨场景 4 场景 bottle_inst0 全链**（select_mask → prepare → rekey tracking → GIF）：

| 场景 | rekey path | GT path | IoU |
|---|---|---|---|
| 116_C5_CellPhone_185deg | 0.72m | 0.74m | 0.555 |
| 121_C5_CellPhone_161deg | 3.84m | 0.40m | 0.432（部分失败） |
| 2chair-bottle | 1.84m | 1.97m | 0.688 |
| 3chair-bottle | 4.23m | 4.18m | 0.603 |

116/2chair/3chair path 与 GT 几乎一致，遮挡+手举段均跟住；121 有漂移段（遮挡段 IoU 掉 0，path 虚高 10 倍）待查（候选原因：mesh 选择/depth 质量）。

**踩坑**：
1. register 在等价位姿间翻转 → 平移分量跳变 2×0.622m（mesh 原点离质心距离），**平移分量 path 是表示层假象**，应看质心轨迹（R@c+t）path；savgol 平滑跨 register 跳变会把两个等价位姿平均成错误位姿（IoU 0.523→0.361），**rekey 后不再叠加 lock+savgol**
2. fp_gif_2d3d 的 test 同步目录改为 `<场景名>_<实例名>`，避免跨场景同名实例 combo.gif 互相覆盖
3. GIF 渲染：GT mask 红色半透明填充（fill=0.45）+ 3px 轮廓、FP 投影绿色 fill=0.18——小物体（~500px）纯细轮廓在视频里像背景图案难以辨认

**产物**：`test/fp_vis/{7_vggt_omega_toy_inst1,116_..._bottle_inst0,121_...,2chair-...,3chair-...}/combo.gif`（红=SAM3 真值、绿=FP 跟踪、橙点=质心尾迹）。

### Q28: 用 SpatialTrackerV2 替代 FoundationPose 做物体跟踪的效果如何？

**回答**: 已实现 [spa_track_object.py](../tools/analysis/spa_track_object.py) 适配脚本，流程：SpaTrackerV2-Offline 前馈输出 2D 点轨迹（mask 内均匀采样 query）→ 用自有 depth/cam_K/extrinsics 反投影回 VGGT 世界系（不用模型自己的 BA 世界系，保证与 GT 同系同尺度）→ mesh 采样点 ICP 到 query 帧物体点云锚定 P0 → 逐帧加权 Kabsch（vis×conf 加权，2 轮剔离群）恢复刚体位姿。结果缓存 spa_tracks.npz，重跑位姿恢复无需 GPU。

**toy_inst1 单场景对比**：

| 方法 | 面片 IoU | 质心 path | GT path |
|---|---|---|---|
| FP rekey | **0.529** | 1.71m（+52%） | 1.12m |
| SpaTrackerV2 | 0.479 | **1.09m（-2.8%）** | 1.12m |

结论：SpaTrackerV2 的 3D 轨迹精度显著更好（路径误差 2.8% vs FP 的 52% 高估），但投影 IoU 略低，主因是深度反投影的逐帧噪声（~35mm/帧抖动）。

**实验教训**：
1. 平滑不是万能药：savgol 窗口21（项目 FP 惯例）对快速运动段过度平滑，IoU 崩溃至 0.04；窗口7 折中 0.464；位姿中值滤波+旋转测地均值在快旋转帧失败（IoU 0.00，路径暴涨 3.34m）；原始未滤波版本最优。
2. fp_data 的 mesh.obj 本身已在 VGGT 世界系摆放（bbox 与 mask+depth 点云同位置），ICP 初始值必须包含单位阵——PCA 主轴初始旋转全部错误，阈值 5cm 内零对应导致 fitness=0。
3. open3d point-to-plane 精配在噪声法向下可能发散，必须守护：精配 fitness/rmse 未变好则保留 point-to-point 粗配结果。
4. 依赖坑：pyceres 2.6 与 pycolmap 3.11.1 的 ceres 绑定冲突（ImportError: PositiveExponentialManifold），需 pyceres==2.4；PyPI utils3d 0.1.3 无 torch 子模块，需手动装 EasternJournalist/utils3d@d3a577a；moviepy 必须 1.0.3（2.x 无 moviepy.editor）。

运行：`CUDA_VISIBLE_DEVICES=1 python tools/analysis/spa_track_object.py --scene_dir output_v2/7_vggt_omega --obj toy_inst1 --n_pts 600 --depth_win 1`（foundationpose env）。GIF 对比：`fp_vis/toy_inst1/combo_spa.gif` vs `combo_fp_rekey.gif`。
