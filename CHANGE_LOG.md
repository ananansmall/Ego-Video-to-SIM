# Change Log

项目变更记录。每次 Agent 任务完成后自动追加。

---

## [2026-09-24 14:50] 打包发布 Snapshot v0.1-2026-09-24 到 GitHub

### 目的
把工作区当前状态打成一个可追溯的版本快照，方便日后回退到"这一版能跑的样子"。

### Git 操作（无代码逻辑改动）
- 分支 `main` 推送：`e1ee749` + 新 `ddecf1a` → `origin/main`
  - `origin` = `git@github.com:ananansmall/Ego-centric-Video-to-Simulation.git`
- 新 annotated tag: `v0.1-2026-09-24` → 指向 `ddecf1a41b497facbd692478e774e57a6a5fe2e9`
  - `git push origin v0.1-2026-09-24` 已同步远程

### 提交 ddecf1a 内容概览
- **新增主线入口**: `mainv3.py`（完整 Ego-centric video → simulation 管线入口）
- **新增 stage61 sapien 仿真栈**: `tools/sapien_loader.py`、`stage61_sim_sapien.py`、`stage61_render_scene.py`、`stage61_render_pyrender.py`、`stage61_fixed_camera.py`、`stage61_visualize_sapien.py`、`test_minimal_sapien.py`
- **新增 src 模块**: `pick_place_detector.py`、`stage61_urdf_generator.py`、`vlm_logger.py`、`__init__.py`
- **新增工具**: `material_annotator.py`、`materials_db.json`、`diagnose_penetration.py`、`diagnose_scenes.py`、`rescale_meshes_by_pointcloud.py`、`render_glb_preview.py`、`qwen_client.py`、`visualize_local.py`、`wrist_trajectory_3d.py`、`test_pointcloud_vs_mesh_size.py`
- **文档重组**: `docs/*.md` 重命名到 `docs/{design,pipeline,research}/` 三类目录，新增 `docs/README.md`、`SAM3D_SCALE_DRIFT_REPORT.md`、`design/stage61_urdf_plan.md`、`design/stage6_design.md`、`pipeline/stage5_refinement_analysis.md`、`research/MAIN_MODEL_CALLING_FLOWS.md`
- **测试重构**: 根目录 5 个 `test_*.py` 删除，新增 `test/` 目录下 8 个结构化子测试
- **代码增强**: `mainv2.py`、`src/geometry_utils.py`、`src/object_segmentation.py`、`src/sg_deduplication.py`、`tools/generate_scene_json_stage1.py`、`tools/infer_relations_scene_graph.py`、`tools/refine_inter_object_placement.py`、`tools/refine_other_objects_relations.py`、`stage4/run_alignment.py`、`tools/asset_verifier.py`
- **新增计划文档**: `dynamic_object_detection_plan.md`、`dynamic_pointcloud_output_plan.md`、`analyze_7a_scene.py`、`quick_verify_7a.py`、`test_verify_7a_stage3.py`
- **删除**: `docs/stage5_physics_engine_proposal.md`、`tools/debug_chair_scale*.py`、`tools/test_sam3d_scale.py`
- **`.gitignore`**: 新增 `.trae/`（IDE 目录不进仓库）

### 后续回退方式
```bash
# 回到本快照
git checkout v0.1-2026-09-24
# 回到上一次远程状态（无 tag）
git checkout 85c5949
# 看所有历史
git log --oneline --all
```

### ⚠️ Docs to Review
- `GIT_GUIDE.md`: 目前未提及 tag 用法。后续如果要建立"版本打 tag"的流程规范，可以补一节。
- `TECHNICAL_DOCUMENTATION.md`: 内容停留在 stage5 阶段，未涵盖 stage61 sapien 仿真栈。功能完善后需同步。
- `README.md`: 未提及 `mainv3.py` 这个新入口，也未链接 `docs/README.md` 分类导航。

---

## [2026-07-24 21:30] 修复CLIP去重阈值过低导致所有物体被链式合并为1个

### Changed
- `tools/generate_scene_json_stage1.py`: `CLIP_MERGE_THRESHOLD` 从 0.55 调回 0.90

### Bug Fix
- **根因**: CLIP_MERGE_THRESHOLD=0.55 时，CLIP对不相关物体也能给出0.55+相似度（如bowl≈scissors≈bottle），通过Union-Find链式合并，所有物体被归为一组，代表名称变为"cloth"，最终只剩1个物体
- **正确策略**: SYNONYM_MAP负责已知同义词映射（"mug"→"cup", "furniture legs"→"chair"），CLIP仅合并极接近的同义词（如"trash can"≈"garbage can"，阈值0.90）

---

## [2026-07-24 20:15] 修复VLM字符串列表输出导致0个物体实例

### Changed
- `tools/generate_scene_json_stage1.py` (主VLM检测路径, L1040): `for obj in objects` → `for idx, obj in enumerate(objects)`，添加 `objects[idx] = obj` 替换列表中字符串元素为dict
- `tools/generate_scene_json_stage1.py` (辅VLM检测路径, L1125): 同上修复

### Bug Fix
- **根因**: VLM输出 `{"objects": ["mug", "bottle", ...]}`（字符串列表）而非 `[{"name": "mug"}, ...]`。代码中 `if isinstance(obj, str): obj = {"name": obj}` 只创建了新局部变量，**没有替换列表中的元素**。后续 `valid_objects = [obj for obj in objects if isinstance(obj, dict)]` 过滤时列表中仍是字符串 → 全部被过滤 → 0个物体
- **影响**: 所有VLM输出字符串列表格式的帧都返回0个物体，导致stage1完全失败

---

## [2026-07-24 19:30] 修复同义词去重: merge_synonyms 无法匹配带空格名称

### Changed
- `tools/generate_scene_json_stage1.py::merge_synonyms()`: 内部添加 `replace(" ", "_")`，确保 "furniture legs" 等带空格名称能正确匹配 SYNONYM_MAP（之前只匹配下划线形式如 "furniture_legs"）
- `tools/generate_scene_json_stage1.py::merge_synonyms()`: 新增复数形式二次查表逻辑（如 "furniture_legs" → normalize → "furniture_leg" → SYNONYM_MAP → "chair"）
- `tools/generate_scene_json_stage1.py::normalize_category_name()`: 修复对含下划线复合名称的去复数逻辑，改为只对最后一个词去复数（如 "furniture_legs" → "furniture_leg" 而非错误的 "furniture_leg"）
- `tools/generate_scene_json_stage1.py::_singularize()`: 提取为独立函数
- `tools/generate_scene_json_stage1.py::SYNONYM_MAP`: 补充所有多词条目的下划线形式（如 "cell_phone", "dining_table", "potted_plant" 等），确保 `replace(" ", "_")` 后也能命中

### Bug Fix
- **根因**: `merge_synonyms("furniture legs")` → `name = "furniture legs"` → 不在 SYNONYM_MAP → 回退到 `normalize_category_name` → 去掉 's' → 返回 `"furniture leg"` → 同样不在 SYNONYM_MAP → 最终输出 `"furniture leg"` 而非 `"chair"`
- **同样受影响的**: "ottoman"→应为"couch", "cell phone"→应为"phone", "dining table"→应为"table" 等

---

## [2026-07-13 17:00] 重构 sg_deduplication.py 去重架构 + 更新 mainv2.py 调用链

### Added
- `src/sg_deduplication.py::_build_candidates()` - 提取公共 candidate 构建逻辑
- `src/sg_deduplication.py::_merge_candidates_to_masks()` - 提取公共合并结果构建逻辑
- `src/sg_deduplication.py::merge_dynamic_splits_all()` - 新增全量动态分裂 + 静态遮挡合并函数
- `mainv2.py::stage2_pre_dedup.pkl` - Stage2 去重前 mask 中间数据保存
- `mainv2.py::stage2_post_dedup.pkl` - Stage2 去重后 mask 中间数据保存

### Changed
- `src/sg_deduplication.py::self_category_deduplicate()` - 删除 `_merge_dynamic_splits` 调用，只保留 overlap_ratio ≥ 0.3 合并逻辑；默认 overlap_thre 从 0.2 改为 0.3
- `src/sg_deduplication.py::cross_category_deduplicate()` - 恢复同类跳过逻辑，删除静态遮挡/动态分裂逻辑，只保留 overlap_ratio ≥ 0.3 跨类合并；删除 images_dict/clip_model/clip_processor 参数（改用 **kwargs 保持兼容）
- `src/sg_deduplication.py` - 删除 `_merge_dynamic_splits()` 函数
- `src/sg_deduplication.py` - 删除未使用的 `instance_centroids` 变量
- `mainv2.py` - 导入 `merge_dynamic_splits_all`
- `mainv2.py` - 删除 CLIP 模型加载代码
- `mainv2.py` - 在 `cross_category_deduplicate` 之后调用 `merge_dynamic_splits_all`
- `mainv2.py` - 将 `_pre_dedup_masks.pkl` 重命名为 `stage2_pre_dedup.pkl`，新增 `stage2_post_dedup.pkl`

### Removed
- `src/sg_deduplication.py::_merge_dynamic_splits()` - 旧的动态分裂合并函数（被 `merge_dynamic_splits_all` 替代）

### 架构变更说明

**三层去重流水线：**
```
self_category_deduplicate (同类)   → overlap_ratio ≥ 0.3
         ↓
cross_category_deduplicate (跨类)  → 同类跳过，overlap_ratio ≥ 0.3
         ↓
merge_dynamic_splits_all (全量)    → 静态重叠 + 静态遮挡(子集关系) + 动态分裂(衔接处overlap)
```

**`merge_dynamic_splits_all` 合并条件：**
1. 静态重叠: ov1 ≥ 0.3 or ov2 ≥ 0.3 → 合并
2. 静态遮挡: 单向 overlap ≥ 0.5（子集关系）AND 子集方帧数更少 → 合并
3. 动态分裂: lifespan_ratio < 0.5 AND time_gap ≤ 10 AND 衔接帧 overlap ≥ 0.2 → 合并

### ⚠️ Docs to Review
- `ReplicateAnyScene/PROJECT_SUMMARY.md` - 可能涉及 Stage2 流程描述
- `ReplicateAnyScene/TECHNICAL_DOCUMENTATION.md` - 可能涉及去重算法描述

### ✅ Docs Updated
- `docs/mainv2_technical_doc.md` - §1 Stage2 流程更新（新增 merge_dynamic_splits_all 步骤），§5.2 去重流水线更新，§10.9 跨类去重同位置检测 → 去重架构重构三层流水线，修改记录表新增 2026-07-13 条目

---

## [2026-07-14 14:30] 修正动态物体判定逻辑：区分真实运动与跟踪漂移

### Problem
`trajectory_info.json` 中将椅子(chair)、两只鞋(shoe_inst0/shoe_inst1)都误判为动态物体，但场景中实际只有一个动态物体（杯子 cup）。原因是原判定逻辑 `global_disp > 0.05 OR (median_disp > 0.02 AND moving_ratio > 0.3)` 阈值过低，VGGT 跟踪漂移就能触发误判。

### Changed
- `mainv2.py` - 动态判定逻辑重构：
  - 新增 `mean_disp`（平均帧间位移）和 `max_consecutive_high`（连续高位移帧数最大值）两个指标
  - 判定条件：`mean_disp > 0.005 AND max_consecutive_high > 10`
  - 原理：真实运动（如拿起杯子）会产生大量连续帧的中等位移；跟踪漂移则是零散的小位移，连续高位移帧数很少
  - 日志输出新增 `mean_disp` 和 `consec_high` 字段

### Multi-scenario Validation (3 scenes, 9 instances)
| Scene | Instance | Mean | Consec | Result |
|-------|----------|------|--------|--------|
| beizi | cup_inst0 (DYN) | 0.0068 | 38 | ✓ DYNAMIC |
| beizi | chair_inst0 (STA) | 0.0040 | 21 | ✓ STATIC (mean<0.005) |
| beizi | shoe_inst0 (STA) | 0.0056 | 6 | ✓ STATIC (consec≤10) |
| beizi | shoe_inst1 (STA) | 0.0018 | 10 | ✓ STATIC (mean<0.005) |
| 7a | toy_inst0 (DYN) | 0.0068 | 12 | ✓ DYNAMIC |
| 7 | toy_inst2 (DYN) | 0.0090 | 12 | ✓ DYNAMIC |
| 7 | toy_inst3 (STA) | 0.0206 | 8 | ✓ STATIC (consec≤10) |
| 7 | toy_inst5 (STA) | 0.0418 | 2 | ✓ STATIC (consec≤10) |

**注意**: toy_inst3/5 虽然 mean 很高(0.02/0.04)，但 consec_high 很低(8/2)，说明位移是零散的突发值而非持续运动，属于误检测的碎片。consec 条件有效过滤了这类情况。

---

## [2026-07-14 15:00] 改进坐标系对齐：GeoCalib 采样帧数 + SAM3 prompt 扩展 + VLM 场景识别

### Problem
`7a_vggt_omega` 场景中，所有四个对齐 stage 都失败，R=I, t=0，相机第一帧 extrinsics 完全没有调整。原因：
1. GeoCalib 只采 8 帧，置信度极低（0.0005~0.004），质量阈值 0.5 太严格
2. SAM3 只用 `"floor"` + `"single wall"` 两个 prompt，对床铺/桌面等非标准地面场景无效
3. Stage 2.5 (VLM-assisted) 代码存在但未接入 pipeline，Stage 3 也没有 VLM 增强

### Added
- `src/object_segmentation.py::detect_scene_type_with_vlm()` - VLM 场景类型识别函数，返回 bedroom/living_room/office/kitchen/bathroom/outdoor/unknown
- `src/object_segmentation.py::get_scene_specific_prompts()` - 根据场景类型返回针对性 SAM3 prompt 列表
- `mainv2.py::_load_vlm_model()` - VLM 模型加载函数，用于场景识别和 floor 参考点检测
- `mainv2.py` - `run_stage2()` 新增 `vlm_checkpoint` 参数
- `mainv2.py` - `align_via_vlm_floor_points` 导入
- `mainv2.py` - Stage 2.5 (VLM辅助) 集成到对齐 cascade，位于 Stage 2 (relaxed) 和 Stage 3 (large_plane) 之间

### Changed
- `src/geometry_utils.py::align_via_geocalib()` - 默认 `max_frames` 从 8→30，新增 `quality_threshold=0.3` 参数（原硬编码 0.5）
- `src/geometry_utils.py::align_via_large_plane()` - 新增 `colors`, `sam3_image_model`, `vlm_model`, `vlm_processor` 参数，支持 VLM 增强的大平面分割
- `src/object_segmentation.py::segment_wall_and_floor()` - 从 2 个固定 prompt 扩展为 10 个 floor prompt + 8 个 wall prompt
  - floor_prompts: `["floor", "ground", "carpet", "bed", "table", "desk", "horizontal surface", "tabletop", "counter", "shelf"]`
  - wall_prompts: `["single wall", "wall", "window", "curtain", "door", "bookshelf", "cabinet", "vertical surface"]`
- `src/object_segmentation.py::segment_large_flat_surfaces()` - 新增 VLM 参数，支持场景类型驱动的 prompt 选择
- `mainv2.py` - 对齐 cascade 从 4 阶段扩展到 5 阶段: Stage1(strict) → Stage2(relaxed) → **Stage2.5(VLM assisted)** → Stage3(large_plane with VLM) → Stage4(GeoCalib)
- `mainv2.py` - 移除 `align_via_geocalib()` 调用中的硬编码 `max_frames=8`，使用默认 30

### ⚠️ Docs to Review
- `ReplicateAnyScene/docs/mainv2_technical_doc.md` - 四阶段 cascade 描述需更新为五阶段
- `ReplicateAnyScene/docs/coordinate_and_alignment.md` - 对齐流程描述需更新

---

## [2026-07-14 16:00] 坐标系对齐改进 — 单元测试 + 7a 端到端验证

### Added
- `test/test_alignment_improvements.py` - 7 个单元测试：GeoCalib 参数验证、SAM3 prompt 扩展验证、VLM 场景类型 prompt 映射、geometry_utils 核心函数验证、mainv2.py 集成验证
- `test/test_7a_alignment.py` - 4 个端到端测试：基于 7a 真实数据验证 floor 平面检测、坐标轴变换效果、R≠I 确认

### Changed
- 修复 `test_7a_alignment.py` 中 numpy 广播 bug：`extrinsics` 形状 `(N,3,4)` 不能直接 `R @ extrinsics`，改为列表推导逐帧相乘
- 修复 `test_alignment_improvements.py` 中 `test_geocalib_quality_threshold_in_code` 缺少 `import inspect`
- 修复 `test_alignment_improvements.py` 中 `test_geometry_utils_core_functions` 测试数据尺度问题（从 1.0 间距改为 0.01m 间距以匹配 `max_triangle_size=2e-4`）
- 修复 `test_alignment_improvements.py` 中 `test_mainv2_integration` 路径问题（从 test/ 目录找 mainv2.py 改为 repo_root）
- 修复 `test_7a_alignment.py` 中 `reason` 字段路径（从 `current['reason']` 改为 `current['method_detail']['reason']`）

### Test Results
- **基础单元**: 7/7 通过 ✅
- **7a 端到端**: 4/4 通过 ✅
  - Floor normal = `[-0.034, 0.035, 0.999]`（z 轴朝上，倾斜 < 2°）
  - R 与单位矩阵 Frobenius norm 差异 = **2.8284**（显著旋转）
  - 旋转矩阵差异 = **30.07**（extrinsics 被显著改变）
  - 确认改进后 Stage 3 能从 3D 点云几何找到水平面并产生有效坐标轴变化

---

## [2026-09-13] 更新动态物体判断逻辑分析文档 + questions.md 追加 Q9

### Changed
- `.trae/documents/动态物体判断逻辑分析.md`: 整体重写（仅文档，未改代码）
  - 旧版"两层 median/global OR 逻辑"描述（median>0.02 OR global>0.04）已过时删除
  - 更新为当前真实代码状态:
    - 第一层 `get_optimal_view_frame_id` (geometry_utils.py:786): `median_disp > 0.10`，仅用于最优视角选择，标签仅供参考 (mainv3.py:590)
    - 第二层 (唯一生效): 置信度加权质心 → MAD 剔除 → `is_dynamic = max_consecutive_high > 30` (mainv2.py:709 / mainv3.py:710)
    - 补充历史版本对照表 (v0 OR逻辑 → v1 三条件 → v2 mean+consec → 当前单条件)
  - 新增 Pick/Place 帧检测方案章节 (Stage 3.7 已验证设计):
    - 大小物体分治: 小物体(<1000px)质心法 path/net/pnet/max_run; 大物体(≥1000px) 5帧窗口 overlap 法
    - v2 修改: 所有帧数阈值 ×frame_ratio 统一到原始视频帧空间; low_count(低匹配总帧数)×frame_ratio≥15 替代 r50(最长连续段), 修复 116 场景慢速移动漏检
    - 后处理先合并(gap≤20原始帧)后过滤(持续≥15原始帧), 输出帧号转换到原始视频空间
    - 5 场景验证结果表 (2chair-bottle / 7_vggt / 3chair-bottle / 121 / 116 全部正确)
    - 标注代码状态: pick_place_detector.py 与测试脚本已丢失, 输出 JSON 保留, 待重建

- `docs/questions.md`: 追加 Q9 (动态物体判断与 Pick/Place 检测流程总结), 并将 Q8 的"注意"行归位到 Q8 段落

### ⚠️ Docs to Review
- `docs/pipeline/mainv2_technical_doc.md`: Stage 3 动态判定描述可能仍为旧三条件逻辑, 需同步为 max_consecutive_high > 30

---

## [2026-09-13 16:30] 重建 pick_place_detector.py + 5场景回归验证通过 + 可视化

### Added
- `src/pick_place_detector.py`: 新建 Pick/Place 帧检测完整模块 (Stage 3.7, 按流程图重建)
  - `_extract_instance_points()`: 逐实例逐帧提取 mask 内 3D 点, 有限点过滤, 每帧≥10点, 总有效帧<10丢弃
  - `compute_dynamic_info()`: 大小物体分治动态判定
    - 小物体 (avg_area<1000px) 质心位移法: path>0.15 AND net>0.08 AND pnet<5.0 AND max_run×frame_ratio≥15
    - 大物体 (≥1000px) 点云重叠法 (5帧窗口 get_overlap_ratio): min_match<0.50 AND low_count×frame_ratio≥15 AND low_count≥5
  - `_detect_interactions_small()`: 3帧平滑位移找局部峰值, 回落 baseline×1.5/×2 定 Pick/Place, 置信度=min(1,位移/0.2)
  - `_detect_interactions_large()`: 连续低重叠(<0.50)片段, 起点=Pick 终点=Place, 置信度=段长/20+(1-段内min)
  - `postprocess_interactions()`: 先合并 (gap≤20原始帧) 后过滤 (持续≥15原始帧), 输出帧号×frame_ratio 转原始视频空间
  - `run_stage3_7_pick_place()`: 主入口, 读 _pre_dedup_masks.pkl + coordinate_alignment.json (total_frames/n_frames→frame_ratio), 输出 pick_place_frames.json
- `test/test_combined.py`: 5 场景回归脚本 (支持 argv 场景过滤)
- `test/visualize_pick_place.py`: pick/place 时刻标注图生成 (color帧画框+文字, 输出 pick_place_vis/)

### Changed
- 大物体动态判定新增 `MIN_LOW_FRAMES_RGB=5` 证据数下限: 修复高 frame_ratio 场景 (3chair ratio=4.93) 下
  low_count×ratio≥15 过宽导致 curtains_inst3 (low=4, 零散噪声帧) 误检的问题。
  三条件物理含义: 深度 (min_match, 真的离开) + 时长 (×ratio, 离开够久, 跨场景等价) + 证据数 (≥5 RGB帧, 非零散噪声)

### Test Results (5 场景回归, 全部正确)
| 场景 | ratio | 动态判定 | 交互 (原始帧号) |
|------|-------|---------|----------------|
| 2chair-bottle | 2.29 | bottle_inst0 ✅ (chair_inst2 漂移排除, low=3×2.29=7) | 44→87, 133→183 |
| 7_vggt | 1.0 | toy_inst1 ✅ (11个小物体 pnet 6.6~229 全排除) | 46→97 |
| 3chair-bottle | 4.93 | bottle_inst0 ✅ (curtains_inst3 修复后排除) | 35→138, 242→291 (合并生效), 325→537 |
| 121_CellPhone | 3.0 | bottle_inst0 ✅ (mug min=0.000 但 low=2 排除) | 66→96, 198→228 |
| 116_CellPhone | 3.0 | bottle_inst0 ✅ (low_count 方案修复慢速移动漏检) | 60→99, 183→222 |

可视化抽查: PICK orig=44 手握桌上瓶子, PLACE orig=87 手放椅面 — 物理正确。

---

## [2026-09-13 18:00] Pick/Place 新增过程窗口 (pick_window / place_window)

### Added
- `src/pick_place_detector.py`: 每个交互新增 `pick_window` / `place_window` 两个字段 (原始视频帧号)
  - 大物体: pick_window = wm 从 <0.80 (开始位移, 大部分仍在原位) 降到 <0.50 的过渡带;
    place_window = 段终点回升到 ≥0.80 为止 (归位稳定)
  - 小物体: 位移 < 峰值×0.25 = "移动还小" (相对交互幅度自适应), 窗口向左/右扩到该阈值为止
  - 语义: [pick_window.start, place_window.end] = 完整"手-物体交互阶段"
    (接近→抓取→搬运→放置→稳定); pick/place_frame 仍为关键锚点帧
  - 新常量: MATCH_START_THRESH=0.80, PEAK_WINDOW_FACTOR=0.25
- `test/visualize_pick_place.py`: 标注图新增 PICK_WIN 青框 (窗口起点) / PLACE_WIN 黄框 (窗口终点)
- `.trae/documents/动态物体判断逻辑分析.md`: Step3/4/5 同步窗口规则

### Changed
- `test/test_combined.py`: 交互打印包含窗口
- `postprocess_interactions()`: 合并交互时 pick_window 取更早起点, place_window 起点同步到合并后的 place_frame

### Bug Fix
- **合并后 place_window 起点陈旧**: 合并相邻交互后 place_frame 更新为更晚终点, 但 place_window[0]
  仍是第一段的旧 place_frame (如 3chair 交互2 显示 place=291 但 place_win=[266,291])。
  修复: 合并时 place_window[0] 同步到新 place_frame。修复后全部场景 place_window[0]==place_frame。

### Test Results (5 场景, 动态判定与上轮一致, 窗口数值验证)
| 场景 | 交互 (pick_win / pick / place / place_win) |
|------|-------------------------------------------|
| 2chair-bottle | [39,44]/44/87/[87,94]; [131,133]/133/183/[183,183] |
| 7_vggt | [46,46]/46/97/[97,98] (小物体快速抓取, 窗口退化单帧, 符合动作特征) |
| 3chair-bottle | [30,35]/35/138/[138,138]; [242,242]/242/291/[291,291]; [320,325]/325/537/[537,572] |
| 121_CellPhone | [63,66]/66/96/[96,99]; [192,198]/198/228/[228,240] |
| 116_CellPhone | [60,60]/60/99/[99,99]; [180,183]/183/222/[222,225] |

### ⚠️ Docs to Review
- 集成待办: mainv2/mainv3 Stage 3 之后调用 run_stage3_7_pick_place(); coordinate_alignment.json 保存 total_frames

---

## [2026-09-02] 修复 stage5 场景图推断实例名解析崩溃

### Added
- `mainv3.py`: 新建修复版入口，为当前 `mainv2.py` 的完整备份；未改动 `mainv2.py`

### Changed
- `tools/infer_relations_scene_graph.py::create_labeled_frame_image()`: 实例名解析增加安全判断，仅当后缀为数字时才 `int(parts[1])`，其余格式整名当作类别、索引取 0

### Bug Fix
- **根因**: `create_labeled_frame_image` 假设实例名均为 `category_数字` 格式并直接 `int(parts[1])`。但实例名可能来自 `instance_names` 展开 `f"{key}_{inst_key}"`，其中 `inst_key` 为类别名本身（如 `car`），得到 `car_car`，`int("car")` 抛 `ValueError`
- **修复**: 后缀非数字时安全降级（整名当类别、索引取 0），并在 `instance_masks` 找不到类别时安全跳过，不再崩溃
- **影响**: 修复对 `mainv2.py` 与 `mainv3.py` 均生效（共享工具模块）

---
## [2026-09-13 13:58] - 修复 mainv3.py Stage3 KeyError: 'max_area'

### Changed
- `mainv3.py` (第 586-588 行): Stage3 中 `run_stage3` 的日志打印引用了 `motion_info['max_area']`，但 `get_optimal_view_frame_id`（src/geometry_utils.py）返回的 `motion_info` 只含 `median_disp / max_disp / first_valid_frame / num_valid_frames`，无 `max_area` 键 → Stage3 处理第一个实例即抛 `KeyError: 'max_area'`。改用实际存在的键打印位移信息（median_disp/max_disp/num_valid_frames）。
- 验证: `python3 -m py_compile mainv3.py` 通过；全文件 `motion_info[...]` 访问仅 3 处，键在两条返回路径（<2 有效帧早期返回 / 正常返回）均存在；模拟打印测试通过。

### ⚠️ Docs to Review
- `mainv2.py`: 与 mainv3.py 修复前逐字节相同，存在同样 bug（git status 显示 mainv2.py 有未提交修改），如需跑 mainv2 需同步修复。
- `docs/questions.md`: 已追加 Q4~Q8（Stage5/6 可行性、mainv2/v3 区别、默认模型、帧数/设备指定、本 bug 修复记录）。

---
## [2026-09-13 15:40] - 修复 Stage5.2 sp_refine_on_top NameError

### Changed
- `tools/refine_inter_object_placement.py` (第 675-676 行): `sp_refine_on_top` 引用未定义变量 `supported_mesh`（该函数内被支撑物变换后 mesh 名为 `transformed_mesh`，第 667 行；`supported_mesh` 仅在其他函数中定义）。每次执行必抛 `NameError`，被调用方 try/except 捕获打印 "❌ 精修失败" → **Stage5.2 所有 on_top 精修静默失效**。改用 `transformed_mesh`。
- 验证: `py_compile` 通过；ReplicateAnyScene conda 环境冒烟测试（两个立方体，supported 悬空 0.2m）→ 无 NameError，底面精确贴合支撑面 (z=1.2→1.0)。
- 影响: mainv2 / mainv3 / run_post_pipeline 共享此模块，修复对三者生效。

---
## [2026-09-13 16:05] - 新增 GLB 无头渲染预览工具

### Added
- `tools/render_glb_preview.py`: pyrender EGL 无头渲染 GLB → 多视角预览 PNG（默认 front/side/top/iso 2x2 拼图，支持纹理/顶点色，自适应包围盒取景）。用途: 在服务器端直接出图，Trae IDE 点开即看，无需传输 GLB。
- 验证: 在 output_v2/044_C2_Mug_high_onehand_vggt_omega/final_scene.glb 上渲染成功（4 meshes，纹理正常显示），PNG 内嵌查看确认取景正确。
- 注意: 需用 ReplicateAnyScene conda 环境运行；pyrender Scene 无 .render()，须用 OffscreenRenderer。

---
## [2026-09-13 16:20] - GLB 预览渲染集成进 mainv3 管线

### Changed
- `mainv3.py`: 新增 `_render_glb_preview(glb_path)`（第 1153-1167 行，懒加载 tools.render_glb_preview，try/except 失败隔离），并在 `save_final_glb()` 导出后调用（第 1215 行）。管线每次保存 GLB（initial/base/stage4/stage5）自动生成 `<stem>_preview.png` 四视角预览图。
- 验证: py_compile 通过；按完全一致的调用路径（懒加载+EGL+渲染+grid2x2+imwrite）断言测试通过（4 tiles 640x640 → 1280x1280 PNG）。
- 影响: mainv2.py / run_post_pipeline.py 未改动（用户仅要求 mainv3）。

---
## [2026-09-13 16:50] - 渲染预览工具新增真实相机视角（最大帧）

### Changed
- `tools/render_glb_preview.py`: 新增 `--camera_frame optimal|帧号列表` 参数与 `render_camera_views/load_camera_rig/get_optimal_frame_ids/render_camview_auto`。用输出目录 extrinsics/ + intrinsic.txt 以真实相机位姿渲染 GLB，分辨率与 color 帧一致，输出 `<stem>_camview.png`。修复坐标系 bug：trimesh 加载 GLB 后几何已是 z-up（GLB y-up 约定被 trimesh 自动转回），相机位姿改为 `inv(w2c) @ diag(1,-1,-1,1)`，不再额外翻转；合成视角方向同步改为 z-up 语义。
- `mainv3.py`: `_render_glb_preview` 现在保存 GLB 后自动生成 `<stem>_preview.png`（合成四视角）+ `<stem>_camview.png`（真实相机视角/各实例最大帧）。
- 验证: py_compile 通过；044 场景 cam85 渲染与真实帧逐物体对应；002 场景 6 个最优帧正常；对照实验（stage4 渲染器 + all_instances.pkl）确认管线数据无误。

---
## [2026-09-13 17:15] - 修复 Stage5.2 稳定性检查旋转重新引入穿模

### Changed
- `tools/refine_inter_object_placement.py` (第 2148-2156 行): `refine_inter_object_relations` 中在 `check_stability` 之后、保存最终 GLB 之前追加一次 `resolve_penetrations(shared_penetration=True)`。根因: Phase1 地面对齐将落地物体大幅旋转立正 (112 场景 6 个 cabinet theta 12°~164°→0°)，足印变化使此前 8 轮穿模解决的成果作废，且之后无穿模复检 → 最终 GLB 左右穿模。
- 验证: py_compile 通过；单元测试复现序列 (斜置60°长柜, 初始无重叠 → check_stability 后重叠 0.02m³ → 再次 resolve 后归零、z 贴地保持)；渲染对比 112 的 _sp 版本 (柜子分离) vs final 版本 (柜子穿插) 确认诊断。
- 影响: mainv2 / mainv3 / run_post_pipeline 共享此模块。历史损坏输出需用 `run_post_pipeline.py --scene_dir <dir> --stage5` 重跑。

---
## [2026-09-13 17:40] - 修复 Stage1-strict 对齐空序列崩溃

### Changed
- `src/geometry_utils.py` (第 246-249 行): `align_to_room_coordinate_system` 中各帧 floor 平面法向分歧 >30° 时 `vaild_floor_plane_infos` 为空 → `max()` 抛 ValueError 崩溃。改为返回单位阵 (对齐失败信号)，让 run_stage2 回退到 Stage2-relaxed / VLM / large_plane / GeoCalib 兜底链。
- 验证: py_compile 通过；单元测试 3 场景 (法向差 60° 优雅降级 / 一致 floor 正常对齐 / 无 floor 原有行为) 全部通过。
- 影响: mainv2 / mainv3 共享此模块；困难场景 (地面误分割) 不再中断管线。

---
## [2026-09-13 18:00] - 修复 Stage2 对齐兜底分支的三处 NameError/参数缺失

### Changed
- `mainv3.py`:
  1. 新增 `_load_vlm_model()` (第 306-313 行, 懒加载复用 tools/generate_scene_json_stage1.py 的 load_vlm_model, 支持 Qwen3.5-FP8/Qwen2.5-VL 等) —— 修复 Stage2.5/Stage3 兜底分支 NameError。
  2. src.geometry_utils 导入列表补充 `align_via_vlm_floor_points` (第 172 行) —— 修复同分支第二处 NameError。
  3. `run_stage2` 签名新增 `vlm_checkpoint=None` (第 317 行), main() 调用处传入 (第 1374 行) —— 修复该分支下 vlm_checkpoint 不在作用域的问题。
- 验证: py_compile 通过; 进程内 import mainv3 断言三个修复点全部解析成功。
- 影响: 对齐兜底链 (strict → relaxed → VLM辅助 → large_plane → GeoCalib) 首次完整可用; mainv2.py 同样存在这三处问题, 未改动。

---
## [2026-09-13 19:45] - 实现并验证 SAM3D 尺度漂移修复方案 B（点云校准）

### Added
- `tools/rescale_meshes_by_pointcloud.py`: post-hoc 点云校准脚本（docs/SAM3D_SCALE_DRIFT_REPORT.md §5.2 方案 B 首次落地）。逐实例：KMeans(K=3) 质心聚类 → 移动判定（簇间最小间距 > 0.8×物体中位尺寸）→ 移动物体取最紧凑簇静止段+逐帧 ext 中位数，静止物体全帧跨帧 min/max → uniform k 只改 T（绕世界质心缩放）→ 输出 all_instances_rescaled.pkl + rescale_check_before/after.glb。
- `output_v2/2chair-bottle_vggt_omega/all_instances_rescaled.pkl`、`rescale_check_before.glb`、`rescale_check_after.glb` 及两张 `_preview.png`: 首次验证产物。

### Changed
- `docs/SAM3D_SCALE_DRIFT_REPORT.md` §5.3: 补充实现状态与相对原方案的三处偏差。
- `docs/questions.md` Q13: 更新方案 B 落地状态。

### 验证
- 2chair-bottle_vggt_omega 全部锚点复现: chair_0 k=0.646（漂移 ~1.55x）、chair_1 k=0.473（漂移 2.11-2.25x）、table_0 k=0.618（漂移 1.62x）、bottle_0 k=1.062（报告锚点 1.0x，未被误改）；校准后 z 向 ratio 全部 ≈1.00。
- render_glb_preview 前后对比: before 两把椅子一大一小，after 尺寸一致。

### 相对报告原方案的三处偏差（实测数据驱动）
1. "点数最多的簇" → "质心最紧凑的簇 + 移动判定": 最大簇会选中 bottle 的运动段（39 帧运动 > 任一停留位）或 chair_1 的入画碎片段（24/24 打平 argmax 选中入画段）。
2. 移动物体聚合 min/max → 逐帧 ext 中位数: VGGT 帧间漂移使 bottle 在静止段内质心仍游走 0.3m，min/max 把漂移算进尺寸导致放大 6 倍；单帧测距不受漂移影响。
3. 移动物体 k 三轴中位数 → mesh 最长轴（主轴）比例: 透明瓶身深度噪声把 7cm 宽测成 22cm，2/3 轴被污染时中位数失效。

---
## [2026-09-13 20:05] - 修复校准脚本绕质心缩放导致落地家具悬浮

### Changed
- `tools/rescale_meshes_by_pointcloud.py`: 缩放锚点从 mesh 世界质心改为底面中心 (cx, cy, min_z)。绕质心缩放会让贴地家具底部抬离地面 (table min_z 0→0.249 悬浮)；底面锚定后水平位置不变、落地物体仍贴地 (table min_z=0.000)、动态物体位置不动。
- 验证: 重跑 2chair-bottle 全部 k 与 after ratio 不变 (k 只依赖尺寸, 与锚点无关); table min_z=0.000, bottle 质心逐位不变; 重渲染 after 预览确认。
- 附带诊断 (用户问"瓶子为什么不在桌上"): ①校准未移动瓶子 (before/after 质心一致 [0.007,0.684,0.456]); ②瓶子在源数据里就低了 ~0.49m —— stage3 动态物体按原始点云坐标 (地面 z=-0.459) 摆放, 静态物体被 stage5 对齐到地面 z=0, 坐标系不一致; ③该场景 2026-07-27 生成, 当时 stage5.2 sp_refine_on_top 的 supported_mesh NameError (Q10, 09-13 才修复) 使 on_top 精修全部静默失败, 瓶子未被放回桌面 (stage5.1 已判定 supported by table_0); ④源数据两把椅子 min_z=-0.682/-0.963 埋地下, 为 stage5 Phase1 地面对齐计算偏差 (报告 §7.3 已记录)。②③④均为既有管线问题, 需用修复后代码重跑 stage5 解决。

---
## [2026-09-13 18:20] - 全仓库收口 _load_vlm_model 缺失 (run_post_pipeline 5.2 崩溃)

### Changed
- `tools/refine_inter_object_placement.py` (第 180-183 行): 新增 `_load_vlm_model`, 懒加载复用 Stage1 的 load_vlm_model —— 修复 run_post_pipeline --stage5 时 `refine_inter_object_relations` 策略判定分支的 NameError。
- `mainv2.py`: 同步 mainv3 的三处修复 (Q17) —— `_load_vlm_model` 定义 (第 306-313 行)、`align_via_vlm_floor_points` 导入 (第 172 行)、run_stage2 加 vlm_checkpoint 参数 (第 316-317 行) + main() 传入 (第 1333 行)。
- 验证: py_compile 通过; 进程内断言两模块的修复点全部解析成功, 调用链统一指向 tools/generate_scene_json_stage1.load_vlm_model。
- 影响: VLM 加载在全仓库收口为单一实现, mainv2 / mainv3 / run_post_pipeline 三条入口的兜底分支均可用。

---
## [2026-09-13 21:35] - 尺寸校准接入主流程（Stage3 后）

### Changed
- `tools/rescale_meshes_by_pointcloud.py`: 核心逻辑抽为 `rescale_all_instances(output_path, all_instances, all_optimal_frame_ids)`（加载 _pre_dedup_masks.pkl、逐实例校准 T、打印报告），CLI main() 复用同一函数，逻辑与输出不变。
- `mainv3.py` (第 1385-1394 行): run_stage3 返回后、基础精修之前插入尺寸校准调用（try/except 兜底，失败回退原始尺寸不阻塞管线）。插入在此位置使 基础精修/final_scene*.glb/all_instances.pkl 保存/Stage4/5/post_pipeline 全部自动使用校准后尺寸——run_post_pipeline 读未校准 pkl 的问题随之消失。
- 验证: py_compile 通过；重构后函数在 2chair-bottle 上输出与重构前独立脚本逐位一致（T 矩阵 allclose 1e-9，k=1.062/0.646/0.473/0.618 不变，其他键保留）。
- 说明: mainv2.py 未改动（与 mainv3 维护约定一致，需要时同步同一插入点）。

---
