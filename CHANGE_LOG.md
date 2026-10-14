# Change Log

项目变更记录。每次 Agent 任务完成后自动追加。

---

## [2026-10-04 00:40] - 修复 inside/leaning 策略精修的签名不匹配 (从未真正生效)

### 问题
egodex_0 重跑时日志出现:
```
hammer_0: inside — 放在支撑物内部 (VLM)
❌ 精修失败: sp_refine_inside() got an unexpected keyword argument 'max_offset'
```

### 根因 (与 Q20 同源)
调用点把 on_top/inside 归为同一个 else 分支, 无条件传
`max_offset / initial_offset / refinement_degree`, 但只有 `sp_refine_on_top`
的签名支持这三个参数:
- `sp_refine_inside(supported, supporter)` — 2 参
- `sp_refine_hanging_below(supported, supporter)` — 2 参
- `sp_refine_leaning(supported, supporter)` — 2 参 (内部转调 against_side)
- `sp_refine_against_side(supported, supporter, walls_info)` — 3 参

TypeError 被外层 except 吞掉只打 ❌ → **该策略的精修静默失效, 物体保持原位**。
另外 `leaning` 与 `against_side` 同分支传 walls_info 也是错的 (leaning 只 2 参)。

**为什么现在才暴露**: `_vlm_inference` NameError 修复前, VLM 策略判定静默失败恒
fallback "on_top", inside/leaning 分支**从未被走到**。修复后 VLM 真正工作返回
"inside", 立刻撞上这条从未执行的代码路径。

### Changed
- `tools/refine_inter_object_placement.py`: 精修调用按各函数真实签名分派
  (on_top 传3 参; against_side 传 walls_info; leaning / inside / hanging_below
  只传 2 参)。异常日志加上 `strategy=` 便于定位。

### 验证
- py_compile 通过。
- 五个策略逐一实测 (合成场景): **5/5 通过**, 无 TypeError, 且物理约束正确
  (inside 放入箱内 z 0.210~0.410; against_side 贴侧面; hanging_below 悬挂
  z -0.200~0.000; on_top 落顶面 z 0.700~0.900)。

### 同时记录: egodex 泛化验证 (前两集)
| | egodex_0 | egodex_1 |
|---|---|---|
| 实例数 | 10 | 10 |
| 总耗时 | 2409s (40min) | 2420s (40min) |
| guard 拦截 | **1**(HOI4D 为 22/25) | **1** |
| 穿模修复次数 | 31 | 31 |
| Stage1 关系 | 非全 floor (hammer→box_0) | 正确 (box→table_0) |

校准修复在 egodex 上同样生效 (拦截 1/10, HOI4D 是 22/25); 且 egodex 实例数少
(10 vs 25), 穿模仅 31 次而 HOI4D 346 次, 印证穿模瓶颈只在拥挤大场景出现。

---

## [2026-10-03 15:20] - 修复 5.2 放置策略投票的 O(实例数) 冗余调用 (1078 → 20 次)

### 问题
HOI4D 082_C5 跑 post_pipeline 时发现 5.2 极慢: 每个物体投票 60~65 次,
而代码里的帧数上限 `_MAX_VOTE_FRAMES = 5` 本应只投 5 票 —— 超出 12 倍。
从 vlm_conversation.jsonl 统计: 全场景 1078 次投票调用 (bottle 各65 次),
按 ~2.1s/次 ≈ 38 分钟, 而结果只是一句 `{"strategy": "on_top"}`。

### 根因 (双重)
1. 外层 `for inst_key, frames in instance_to_frames.items()` 遍历该类别下
   **每一个实例**各调一次 VLM (bottle 有 16 个实例 → 16×5 = 80 次), 但要回答的
   问题"bottle 相对 table_0 怎么放"与具体哪个实例无关;
2. `instance_frames[:_MAX_VOTE_FRAMES]` 只在单次调用内截断, 管不住外层重复。
而最终只取 `best_strategy` (第一个有帧实例的结果), 其余 15 次纯属浪费。
注释里已写明"简化: 取第一个有帧实例的策略", 但代码仍在遍历全部实例。

### Changed
- `tools/refine_inter_object_placement.py`: 策略投票按类别选**代表实例**
  (帧最多者, 可见角度最充分), 只投一次。保留多实例语义扩展点: 若将来需
  按实例区分, 应改用一次调用问多个实例 (同 5.1 的 query_instances 做法)。
- 同文件: `vlm_inference` 调用补`caller="stage5_placement_strategy"`
  —— 此前漏传导致这 1078 次调用在日志里全归因为 `unknown`, 难以排查。

### 验证 (用本次真实调用的日志统计)
| (supported, supporter) | 修复前 | 修复后 | 结果一致性 |
|---|---|---|---|
| bottle→table_0 | 993 | 5 | 全部一致 |
| bowl→table_0 | 45 | 5 | 全部一致 |
| cup→table_0 | 20 | 5 | 全部一致 |
| scissor→table_0 | 20 | 5 | 全部一致 |
| **合计** | **1078** | **20 (↓98%)** | 0 处不一致 |

耗时 ~38min → ~0.7min。**4 个类别内部投票结果全部一致**, 说明同类别不同实例
本来就给相同答案, 取代表实例不丢信息 (语义等价)。

---

## [2026-10-03 11:30] - 修复 on_top 用 2D 视觉判断否决 3D 几何约束 (21物体没放到桌上)

### 问题
HOI4D 082_C5 管线跑完后发现: 5.1 关系完全正确 (23 个 `supported by table_0`),
但 21 个物体 (16 bottle + 3 bowl + 2 cup) 的底面仍在 z≈0 地面, 而台面 z=0.971m ——
**5.1 判对的关系被 5.2 整体推翻**, Stage5 最核心的功能失效。

### 根因 (这是 _vlm_inference NameError 修复后暴露的设计缺陷)
`sp_refine_on_top` 开头: `if refinement_degree == "correct": return` (不做对齐)。
而 `_judge_refinement_degree_with_vlm` 是**看照片**回答的: 视频里瓶子确实在桌上,
视觉上就是 "correct" —— 实测 23/23 全部回答 correct。但3D 坐标是错的 (悬空 0.971m)。
**2D 视觉判断不应能否决明确的几何事实**, 两者矛盾时代码无条件相信视觉。

注意因果: 修复 NameError 前, 该函数静默失败被except 兜底返回 "minor"
→ 反而触发了 z 对齐 → 瓶子被正确放到桌上。修复后 VLM 真正工作, 缺陷才暴露。
**静默失败曾意外充当了兜底**, 这是修复同类bug 时的通用风险。

### Changed
- `tools/refine_inter_object_placement.py`: on_top 精修程度判断改为**几何优先**。
  先算 `gap = supporter_top_z - supported_bottom_z`;
  - `|gap| >= GEOM_ALIGNED_TOL (0.15m)` → 几何上必定错位, **不问 VLM**, 直接对齐
  - `|gap| < 0.15m` → 几何已贴合, 才咨询 VLM 是否已正确 (保留原设计意图:
    避免对已就位的物体做无谓调整)
  新增常量 `GEOM_ALIGNED_TOL = 0.15`,含判定语义注释。

### 验证
- py_compile 通过。
- **真实场景数据** (table_0 台面 z=0.971): 21/21 物体 gap 均 > 0.9m → 全部走
  "直接对齐"; VLM 调用 23 次 → **0 次** (额外省约 20min)。
- **合成场景 8 例验证阈值语义** (桌高0.8): 精确贴合/穿透5cm/悬空2cm/悬空8cm/
  悬空13cm → 问VLM; 悬空15cm(边界)/悬空20cm/在地面 → 直接对齐。全部符合预期。
  确认已就位的物体不会被强制对齐 (未破坏原设计意图)。

---

## [2026-10-03 01:00] - 修复混合分支 guard 自校验参照物与缩放量不同源 (22/25 实例校准失效)

### 问题
HOI4D 082_C5 重跑后尺寸校准 23/25 实例被 guard 拦截, 日志显示
`校准后比值 x/0.8/0.2~0.3 超出 [0.7,1.55]`, z 轴比值一致地落在 0.06~0.28。

### 诊断过程 (两轮错误推断, 最终由数据定位)
1. 先怀疑 `moved` 分支三轴统一缩放 → 用真实点云复现, 实测 `moved=False`
   (簇间间距 0.008<< 阈值 0.196), 该分支未命中, 推断不成立。
2. 改判"落地/抬高分支选错" → 提出用 `mesh_min_z - ground_z` 替代
   `z_min_pc - ground_z`, 实测 0/25 实例改变 (ground_z 本身被污染,
   所有物体 lift 都是 0.73, 换参照物等于重复同一 bug), 修复无效已回退。
3. **关键数据**: 直接打印 `pc_z / mesh_z` 发现两者其实高度吻合
   (bottle 0.86~1.57, cup 0.80, table_0 0.80) —— 说明点云测量本来就是好的,
   是 guard 把好结果判成了垃圾。

### 真正根因
混合分支用 `k_z = (z_max - z_min_pc) / mesh_z` 缩放, 但 after_ratio[2] 的
校验参照物是 `z_range = z_max - ground_z`。两者**不同源**:
- ground_z 是全场景 z 的 2% 分位, 被埋地家具拉到 -0.757 (真实地面远高于此)
- z_range 被撑大约 0.76m → `after_z = z_after / z_range` 一律 0.2~0.3
- 22 个测量正确的实例因此被判为垃圾, 全部回退 k=1 (校准形同取消)
- 而当公式恰好相等时 after_z≡1.0 → 该分支自校验又是失明的。两个问题同源。

### Changed
- `tools/rescale_meshes_by_pointcloud.py`: 混合分支 `after_ratio[2]` 的参照物
  从 `z_range` 改为 `pc_z_span = z_max - z_min_pc`, 与 `k_z` 的分子同源,
  缩放多少就用什么校验。pc_z_span<=0 时回退 ext_pc[2] 防除零。
  分支选择逻辑 (gap 阈值) 未改动 —— 它本身自洽, 问题只在校验参照物。

### 验证 (用本次真实场景数据重跑 rescale_all_instances)
| | 修复前 | 修复后 |
|---|---|---|
| guard 拦截数 | 22/25 | **2/25** |
| bottle 校准后 z | 0.20~0.28 (回退 k=1) | **1.00 (生效)** |
| cup / bowl | 全部回退 | **全部生效** |

- py_compile 通过; 首次无效的 lift 改动已完整回退, LIFT_ELEVATED 常量已清除。
- 剩余 2 个被拦截的是 scissor_0/1: mesh_z 仅 0.010~0.012m (贴地剪刀),
  点云测出 0.037m 使k_z=3.06/8.71 —— 属真实的点云不可信, 拦截正确。

### 同时验证: 方案1 提速生效
`⚡ 25 个实例聚合为 26 帧 (去重前约 50 次调用)`,
**5.1 耗时 173.4s (上次 3002s) → 提速 17.3 倍**。

---

## [2026-10-02 23:20] - Stage5.1 消除 O(N^2) 重复 VLM 计算 (按帧聚合 + prompt 按实例裁剪)

### 问题
HOI4D 25 实例场景全流程 2h43m, 其中 Stage 5 占 77.5% (5.1 50min + 5.2 72min)。
分析 vlm_conversation.jsonl: 95 次调用中 72 次在 5.1, 且每次响应中位 2412 字符。
根因是 O(N^2) 的重复生成 —— 逐实例逐帧调用 VLM, 但每张标注图的 prompt 要求
"输出全场景 25 个实例的完整 JSON", 调用方却只取自己那一行, 其余 24 条立即丢弃
(25 实例 x 3 帧 x 25 条 = 1875 条, 有效仅 25 条)。

### Changed
1. **方案1 - 按帧聚合** (`tools/infer_relations_scene_graph.py`):
   Step 2 从"逐实例 x 逐帧调用"改为"先收集每个实例需要的帧, 按 frame_id 去重,
   每帧渲染一次标注图 + 调用一次 VLM", 结果缓存进 frame_results 再按实例汇总票。
   25 实例的实际 optimal 帧只用到 19 个不同帧, 附带 dynamic 首帧与 keyframe 帧
   也能命中已有帧。
   标注图文件名从 `label_{inst}_optimal.png` / `label_{inst}_kf{fid}.png`
   改为 `label_frame{fid}.png` (同帧共用一张)。
2. **方案2 - prompt 按实例裁剪** (`infer_relations_from_labeled_image`):
   新增可选参数 `query_instances`。调用方只传该帧真正要用的实例, prompt 的
   "exactly N objects" / query list / 末尾指令全部改为按 query 规模输出,
   但**仍完整列出全部实例名作为参考**, 保证 parent 能写成 "table_0" 之类。
   max_new_tokens 随之分档: num_query > 4 用 4096 (全量), 否则 256。

### 验证
- py_compile 通过; AST 解析通过。
- **等价性 (离线桩数据, 用上次真实运行的 25 实例/19 帧分布构造)**:
  重构前后加权票 Counter 完全一致, 最终关系决策 (vote_relations) 逐实例一致。
  调用次数 52 → 19 (降 63%)。
- **prompt 裁剪正确性**: 3 个场景断言通过 (全量 25→4096 / 裁剪 3→256 /
  中间档 5→4096), 确认 25 个实例名仍全在 prompt 中作为参考。
- **真实 VLM 冒烟测试 (GPU1, 用上次跑出的真实标注图)**:
  裁剪查询 2 实例 5.5s vs 全场景查询 26 实例 42.6s = **提速 7.7x**;
  输出 195 字符 vs 2494 字符; 两者结论一致 (bottle_0/bowl_0 均 parent=table_0)。

### 预期收益
5.1 从 ~50min 降到 ~10min 量级 (调用数降 63% x 单次耗时降 ~7.7x, 二者乘积
在共享帧时为线性叠加)。整条管线 2h43m 预计降到 1h 内。5.2 未改动 (23 次调用
响应中位仅 8 字符, 本就很快, 瓶颈不在生成)。

---

## [2026-10-02 22:30] - 修复 _vlm_inference NameError (Stage1 关系投票 + 5.2 放置投票静默失效)

### 问题
HOI4D 082_C5 重跑时 Stage1 补充帧检测 10/10 全部报 `❌ 错误: name '_vlm_inference' is not defined`。
用户贴日志报告。根因: `src/vlm_logger.py` 重构把 `_vlm_inference` 改名为 `vlm_inference`
（并统一日志入口），但两处调用点没有跟上，NameError 被外层 try/except 吞掉只打 ❌。

### 影响范围（比表面更严重）
该 bug 不仅导致报错帧的补充检测失败，还让两个投票机制**历史上从未真正生效**:
1. Stage1 7.3 关系跨帧投票（generate_scene_json_stage1.py 第 2040 行）→ 全部走
   fallback，解释了 stage1 关系恒为全 "supported by floor"。
2. Stage5.2 放置策略 VLM 投票（refine_inter_object_placement.py 第 504 行）→ 投票
   永远为空恒 fallback "on_top"。

### Changed
- `tools/generate_scene_json_stage1.py` (第 1536、2040 行): `_vlm_inference(` → `vlm_inference(`。
- `tools/refine_inter_object_placement.py` (第 504 行): `_vlm_inference(` → `vlm_inference(`。

### 验证
- 全仓库 grep `_vlm_inference`: 仅剩 vlm_logger.py 内一行注释（说明改名历史），无残留调用点。
- `py_compile` 两文件通过（conda env ReplicateAnyScene python）。
- 签名兼容: vlm_logger.py `vlm_inference(image, model, processor, prompt, max_new_tokens=1024, ...)`
  与三处调用的实参一致。

### 注意（行为变化）
修复后 Stage1 关系投票将**首次真正生效**，stage1 json 的关系不再是必然全 floor；
5.2 放置策略也会开始按 VLM 投票结果区分 on_top/inside/against_side。
重跑后需对比检查 stage1 关系与 5.2 策略分布是否合理。

---

## [2026-10-01 01:20] - Stage5 VLM 投票帧数上限 (方案1提速)

### 问题
HOI4D 25 实例场景全流程 2h43m (2chair 仅 ~30min)。时间炸弹在 Stage5:
- 5.1 场景图推断 54min: 每实例 3 次 VLM (preview + optimal + 2 keyframes),
  每次调用 ~40s (标注图含全场景实例, 生成完整场景图 JSON)
- 5.2 放置策略 ~25min: 每对象遍历全部实例帧 (bottle_0 达 11 帧) 投票,
  帧映射修复后投票真实发生, 调用数随帧数线性增长

### Changed (投票帧数上限)
- tools/refine_inter_object_placement.py::vlm_lookup_placement_strategy:
  instance_frames[:5] (optimal 优先排序, 前 5 帧足够统计意义)
- tools/infer_relations_scene_graph.py Step2: keyframe_candidates[:2] → [:1]
  (保留 preview 票 + optimal 权重2票 + 1 关键帧; 动态物体首现帧权重10仍在首位)

### 验证
py_compile 通过; 冒烟测试: 11 帧输入实际 VLM 调用 5 次, 策略输出正常;
5.1 源码含 [:1] 上限。

### 预期与遗留
- 预期节省 ~30min (5.1 -17min, 5.2 放置 -14min), 总时长 2h43m → ~2h10m
- 遗留瓶颈 (独立问题, 未动):
  1. 5.1 单次调用 ~40s — 标注图让 VLM 输出全场景 JSON 但只取目标实例, 可按实例裁剪 prompt
  2. 穿模修复尾部 ~34min — 每对相交检测 5-20s (O(N²)×迭代), 需要空间索引/降采样

## [2026-10-01 00:30] - 修复抬高物体 z 公式错误 + Phase2 补 xy拉回

### 问题 (HOI4D 082_C5_CellPhone 场景暴露)
1. 16 个瓶子全部 1m 高 (比 0.9m 桌子还高): 混合分支 z 公式 k_z=(z_max-ground)/mesh_z
   只对落地家具成立; 桌上瓶子 gap=0.9m 时该式把桌高算进瓶高 (0.9+0.25=1.15m → k_z=2.5)。
   guard 对此失明: 混合分支 after_z ≡ 1.0 (k_z 由同一条公式构造)。
2. 部分瓶子悬浮在支撑面同高但水平错位: Phase2 悬空修复只修 z 并标记 stabilized,
   Phase3 跳过它们 — Phase2 自身无 xy 逻辑 (与之前 bag 问题同源)。

### Changed
- tools/rescale_meshes_by_pointcloud.py 混合分支: gap = z_min_pc - ground_z 分三档
    gap <= 0.30: 点云 min/max (底部可见)
    0.30 < gap < 0.50: z_max - ground (落地家具, 腿被遮挡)
    gap >= 0.50: z_max - z_min_pc (抬高物体, 点云自身跨度=真实高度)
- tools/rescale_meshes_by_pointcloud.py guard: 混合分支补查 k_z ∈ [0.4, 2.0]
- tools/refine_inter_object_placement.py: 抽取 _pull_xy_into_supporter() 公共函数
  (内缩0.6×半径取最近点, 上限0.6m), Phase2 悬空修复与 Phase3 接触不足共用;
  Phase2 修 z 前先拉 xy 并重算 gap。

### 验证 (单元测试)
- 抬高物体合成场景: 旧公式 k_z=2.09→1.15m瓶 ❌; 新公式 k_z=0.45→0.25m瓶 ✅ (真值0.25)
- 落地家具回归 (gap=0.33): 仍走 z_max-ground ✅
- guard k_z: 2.5拒 / 0.45过 / 1.0过 ✅
- xy拉回: 桌沿物体+0.03m拉入 ✅; 远处0.6m外拒绝瞬移 ✅; 已在中心不动 ✅
- py_compile 通过

### 待验证
HOI4D 082_C5 全新跑: 瓶子应 ~0.25m 高且落在大桌上, 无悬浮瓶。
注: 拥挤场景瓶间穿模仍存在, 属后续独立问题。

---

## [2026-09-30 10:30] - 修复抬高物体 z 公式错误 + Phase2 补 xy拉回

### 问题 (HOI4D 082_C5_CellPhone 场景暴露)
1. 16 个瓶子全部 1m 高 (比 0.9m 桌子还高): 混合分支 z 公式 k_z=(z_max-ground)/mesh_z
   只对落地家具成立; 桌上瓶子 gap=0.9m 时该式把桌高算进瓶高 (0.9+0.25=1.15m → k_z=2.5)。
   guard 对此失明: 混合分支 after_z ≡ 1.0 (k_z 由同一条公式构造)。
2. 部分瓶子悬浮在支撑面同高但水平错位: Phase2 悬空修复只修 z 并标记 stabilized,
   Phase3 跳过它们 — Phase2 自身无 xy 逻辑 (与之前 bag 问题同源)。

### Changed
- tools/rescale_meshes_by_pointcloud.py 混合分支:
  gap = z_min_pc - ground_z 分三档
    gap <= 0.30: 点云 min/max (底部可见)
    0.30 < gap < 0.50: z_max - ground (落地家具, 腿被遮挡)
    gap >= 0.50: z_max - z_min_pc (抬高物体, 点云自身跨度=真实高度)
- tools/rescale_meshes_by_pointcloud.py guard: 混合分支补查 k_z ∈ [0.4, 2.0]
- tools/refine_inter_object_placement.py:
  抽取 _pull_xy_into_supporter() 公共函数 (内缩0.6×半径取最近点, 上限0.6m),
  Phase2 悬空修复与 Phase3 接触不足共用; Phase2 修 z 前先拉 xy 并重算 gap。

### 验证 (单元测试)
- 抬高物体合成场景: 旧公式 k_z=2.09→1.15m瓶 ❌; 新公式 k_z=0.45→0.25m瓶 ✅ (真值0.25)
- 落地家具回归 (gap=0.33): 仍走 z_max-ground ✅
- guard k_z: 2.5拒 / 0.45过 / 1.0过 ✅
- xy拉回: 桌沿物体+0.03m拉入 ✅; 远处0.6m外拒绝瞬移 ✅; 已在中心不动 ✅
- py_compile 通过

### 待验证
HOI4D 082_C5 全新跑: 瓶子应 ~0.25m 高且落在大桌上, 无悬浮瓶。
注: 拥挤场景瓶间穿模仍存在, 属后续独立问题。

---

## [2026-09-28 15:40] - 尺寸校准新增 sanity guard (垃圾测量回退 k=1)

### 问题
3chair 场景背景物体 (窗帘/电视/透明瓶) 点云测量不可靠, 校准把 curtain_6 放大到
4.75m (T缩放det=10.06)、tv_0 到 2.4m (det=8.22)、bottle k=2.80, 整个复原场景被毁。

### 方案: 校准自校验
k 由点云测得, 校准后 mesh 应与点云吻合; 吻合不上 = 测量本身不可信 → 回退 k=1
(SAM3D 原始尺寸是最可靠兜底)。

### Changed
- tools/rescale_meshes_by_pointcloud.py (rescale_all_instances 内):
  - 静止段 (移动) 分支: k ∈ [0.5, 2.0], 超出回退
  - 其他分支: 校准后各轴比值 after ∈ [0.70, 1.55], 任一轴超出回退
  - 回退实例打印 `[guard] 测量不可信, 保留原始尺寸`, 收敛统计只计通过实例

### 阈值依据 (28 实例回放, 全新跑日志数值)
- 接受集最大 1.51 (3chair box_0), 拒绝集最小 1.64 (curtain_0) → 1.55 居中
- 2chair 5/5 通过 (table_0 after 1.45 保留, 高度修复不丢)
- 3chair 9 个灾难级全拒: bottle_0(k=2.80), cloth_0(2.15), cloth_1(4.76),
  curtain_0(1.64), curtain_2(2.83), curtain_5(1.73), curtain_6(14.84), tv_0(1.78)
- 2 个轻度噪声接受: cloth_2(1.46), curtain_3(1.39) — 与 box_0(1.51) 不可分, 折中

### 备注
- 离线重跑校准读到的是已校准 pkl (主流程覆盖保存), 数值与全新跑不同;
  guard 判定以全新跑日志回放为准。
- 渲染验证待重跑管线后进行 (guard 只在 mainv3 完整管线的校准环节生效)。

---

## [2026-09-28 12:40] - Stage5 管线内置四视图预览输出

### Changed
- tools/render_glb_preview.py: 新增 `render_preview_png(glb_path)` 可复用函数
  (默认 front/side/top/iso 四视图 2x2 合成, 输出 <stem>_preview.png, 异常返回 None)
- tools/refine_inter_object_placement.py (L2228-2234): Stage5 最终 GLB 保存点后
  自动调用四视图渲染。该保存点为 mainv3 完整管线与 run_post_pipeline 共用,
  两边都覆盖。渲染失败仅打警告, 不影响管线。

### 验证
py_compile 通过; 用 2chair final_scene_stage5.glb 实测生成 preview PNG 成功。

---

## [2026-09-28 12:10] - 修复 Phase3 假接触比 (shapely 缺失) + 新增 xy 拉回

### 根因 1 (更底层): shapely 从未安装, Phase3 接触检测一直空转
`_project_footprint` (tools/refine_inter_object_placement.py:1849) 在 shapely 缺失时
静默返回 None → overlap_area 恒 0 → support_ratio 恒 0.0%。
**所有历史日志里的"接触比 0.0%"都是假值**, Phase3 实际在做"无条件 z 对齐"。
bag 悬空桌边直接源于此: z 被对齐到桌面高, xy 无任何机制修正。

### 根因 2: Phase3 只修 z, 无 xy 拉回
接触不足时只对齐 z; z 已对齐时仅标记 unstable("z已对齐但xy偏移")不修。
on_top 精修同样只算 z_offset。现有 "xy偏移" 逻辑 (L1467-1527) 是反向跟随
(支撑物移动时小物体跟随), 不是"拉回支撑面"。

### Changed
- 环境: ReplicateAnyScene conda env 安装 shapely 2.1.2 (此前缺失, rtree/scipy 已有)
- tools/refine_inter_object_placement.py Phase3 (L1696-1730): support_ratio < 5% 时
  先做 xy 拉回——支撑足迹内缩 0.6×物体半径, 取离物体中心最近点为目标;
  位移上限 0.6m (超限仅标记 unstable 防误判瞬移); 拉回后重算包围盒再对齐 z。

### 验证 (真实场景足迹离线测试)
bag:    0.0% → 平移0.334m → 89.3%  ✅
bottle: 0.0% → 平移0.153m → 84.9%  ✅
chair0(对照, floor支撑): 需0.71m > 0.6m 上限 → 拒绝瞬移 ✅
py_compile 通过。

### 待验证
post_pipeline --stage5 重跑两场景, 确认日志出现 "Phase3 xy拉回" 且
final_scene_stage5.glb 中 bag/bottle 真实落在桌面足迹内。

---

## [2026-09-28 08:50] - 修复 post_pipeline Stage 5.1 丢失场景图关系

### 问题
用 run_post_pipeline.py --stage5 重跑 2chair/3chair 时, Stage 5.1 从 stage1 JSON 重新细化,
而 stage1 关系全是 "supported by floor" → 无实例间关系 → 5.2 直接跳过 ("无物体间支撑关系"),
瓶子没有放回桌面。09-27 mainv3 完整跑时 5.1 (infer_relations_scene_graph) 推断出的
bottle_0→table_0 等关系被丢弃。

### 根因
mainv3 的 5.1 用 infer_relations_scene_graph (需内存中的 dedup masks/colors, 输出实例级关系),
post_pipeline 的 5.1 只调 refine_other_objects_relations (仅细化 "supported by other objects"),
两者不是同一实现, post_pipeline 无法复现场景图推断。

### Changed
- tools/run_post_pipeline.py::run_refine_relations: 优先复用 {scene_dir}/relations_scene_graph.json
  的 refined_relations (实例级), 以 stage1 类别级关系作底合并后写入 final_relations.json;
  加载失败时回退到原 refine 流程。

### 验证
py_compile 通过; 实际数据合并测试:
  2chair: 5 条实例级 + 4 条类别级 → has_inter=2 (bottle_0/bag_0 → table_0)
  3chair: 24 条实例级 + 13 条类别级 → has_inter=3 (bag_0→table_0, bottle_0→chair_0, box_0→table_0)

---

## [2026-09-24 15:30] - 注释掉基础精修 0.3m 阈值（测试）

### 问题
2chair-bottle_rescaled 场景中两把椅子 z_min=-0.682/-0.963，相对对齐后地面（z≈0.081）埋地 0.763m/1.044m，但基础精修未修正它们。

### 根因
`src/sp_refinement.py::refine_supported_by_floor_object` 第 90 行硬编码阈值 `abs(z_min) < 0.3`：椅子偏差 0.682/0.963 均超阈值，走 `else: translation_matrix = np.eye(4)` 分支静默跳过，仅 table_0（z_min=-0.0735）被修正。

### Changed
- `src/sp_refinement.py` (第 89-97 行): 注释掉 `if abs(z_min) < 0.3` 阈值判断，改为无条件修正 `translation_vector = np.array([0,0,-z_min])`。原代码以注释形式保留（标注 `[TEST]`）。

### 分析过程（本轮排查要点）
- 坐标系澄清: mesh 处于对齐后坐标系（R+t 已应用），地面 2% 分位=0.081，非 pkl 原始坐标系的 -0.462。table_0 z_min=0.000 离地 0.081m 属正常对齐，非悬浮。
- 椅子埋地是 Stage3 SAM3D 位姿估计偏差（0.76-1.04m），基础精修本应修正但被阈值拦截。
- 地面对齐算法目标（TECHNICAL_DOCUMENTATION.md:478 "地面在z=0平面"）实际达成，对齐本身无问题。

### 待验证
- 未跑管线验证。预期效果: chair_0/1 被拉回 z=0，但仍需观察是否引入新副作用（如 z_min>0.3 的非 floor 物体被误修正，因该函数仅对 `supported_by_floor` 关系调用，风险较低）。

### ⚠️ Docs to Review
- `TECHNICAL_DOCUMENTATION.md` §10.5 / 基础精修章节: 如移除阈值成为正式修复，需更新 0.3m 阈值说明。
- `docs/questions.md`: 本轮坐标系分析与结论尚未追加。

---

## [2026-09-27 23:20] - 修复 Stage5.2 实例→帧映射 ghost key bug

### 问题
2chair-bottle_rescaled / 3chair-bottle_rescaled 两个场景的 Stage 5.2 日志均显示所有实例 `❌ 无帧`，导致 VLM 放置策略与 SP 精修失去视觉锚点，退化为文本规则判断。

### 根因（双重命名不一致）
`tools/refine_inter_object_placement.py::refine_placement_positions` 从 5.1 关系构建 `all_categories` 时得到的是**实例名**（`['bag_0','bottle_0','table_0']`），随后：

1. `_load_optimal_frames_for_instances` 调用 `_fuzzy_match_name('bottle', [...'bottle_0'...])` 走 base 匹配返回**实例名** `bottle_0`，再拼接 `f"{cat}_{inst_idx}"` → ghost key `bottle_0_0`
2. `_build_instance_frame_map` 又把输入名剥离数字得到 `base_categories=['bag','bottle','table']`，期望 key `bottle_0`
3. `bottle_0` ∉ `{bottle_0_0, ...}` → 永远匹配不上；keyframes 路径同理（`visible_names` 存实例名 vs `base_categories` 纯类别名）

121 等旧场景关系为 `supported by other objects`，无实例名后缀，未触发该路径，故表现正常。

### Changed
- `tools/refine_inter_object_placement.py` (第 1899-1906 行): 收集 `all_categories` 后统一 `re.sub(r'_\d+$','',name)` 归一化为类别名，避免 ghost key。

### 验证
用同场景数据复现修复前/后：
```
修复前: bag_0=0帧  bottle_0=0帧  table_0=0帧  (ghost key bag_0_0/bottle_0_0/table_0_0)
修复后: bag_0=2帧  bottle_0=11帧 table_0=6帧
```
`py_compile` 通过。

### 待办（本轮未修，需重跑后观察）
- 姊妹文件 `tools/refine_other_objects_relations.py` 的 `load_optimal_frames`/`build_object_to_frames` 传入 `json_object_names`（同为实例名），结构为 `{json_name: {inst_idx: [...]}}` 嵌套 dict，需单独确认是否同类问题。
- `bag_0` 接触比 0.0% 被 Phase3 强上移 0.8m（袋状物体几何接触检测不适配）。
- 3chair 背景物体过多（13 类，含 cabinet/curtain/tv），校准偏差 13.837；属场景复原代价，非 bug，但需评估是否加前景过滤。

### 分析更正（本轮排查中的错误结论）
- 曾误判"校准未生效、bottle 高度差 5.5 倍"。实际 `all_instances.pkl` 中 bottle=0.180m / chair=0.961m / table=0.820m 均合理，校准正常。误判源于把校准日志 `before(x/y/z)` 当成尺寸（实为 mesh/点云比值）。
- 曾误判 121 场景"跑得好"。121 为 8 月运行，无校准环节，属 baseline 对照而非已修复。

### ⚠️ Docs to Review
- `TECHNICAL_DOCUMENTATION.md` Stage5 章节: 未记录实例名/类别名在 5.1→5.2 间的归一化约定。
- `docs/questions.md`: 本轮"命名不一致导致视觉锚点丢失"尚未追加。

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
## [2026-10-02 12:45] - 新增 stage61_gif.py：scene.urdf → SAPIEN 物理 → GIF

### Added
- `tools/stage61_gif.py`: 读 `<output_dir>/urdf/scene.urdf`，用 SAPIEN 跑真实物理（URDF 里的 coacd 凸分解collision mesh + gazebo mu1/mu2 摩擦系数），pyrender 光栅化 URDF 里的真实 visual mesh（带顶点色），逐帧重建场景并环绕渲染，输出 `stage61_sapien.gif` + `stage61_sapien_frame.png` + `stage61_gif_report.json`。参数：`--n_frames/--sim_steps/--res/--elev/--fps/--no_orbit/--save_mp4/--urdf/--out`。

### 环境结论（实测，非推测）
- SAPIEN 3.0.3 离屏渲染在本机画不出 actor：`cam.take_picture()` 只输出纯色，连 `add_box_visual` 的 primitive box 都渲染不出来（`scene.entities` 有对象、`get_all_actors()` 可见）。因此采用 SAPIEN 物理 + pyrender 光栅化的组合，不依赖 SAPIEN 渲染管线。这也解释了既有 `tools/stage61_visualize_sapien.py` 为何用彩色 box 占位。
- pyrender 0.1.45 相机位姿在 `add(cam, pose=...)` 时生效；后续改`node.matrix` 不生效，必须每帧重建 scene或用 add 时的pose。
- 运行环境：`/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python`（sapien/pyrender/trimesh/imageio/cv2/open3d 齐全，EGL headless 可用）。默认 python 命令不存在。

### 验证（4 个场景，全部通过）
- `7_vggt_omega`（8 物体，复用已有 urdf）
- `2chair-bottle_vggt_omega`（4 物体，复用已有 urdf）
- `laptop_vggt_omega`（3 物体，完整跑通 `stage61_urdf_generator.py` → URDF 281s/0错误 → GIF）
- `7a_vggt_omega`（8 物体，完整跑通 `stage61_urdf_generator.py` → URDF 1143s/0 错误 → GIF）

### ⚠️ Docs to Review
- `docs/design/stage61_urdf_plan.md`: 第 98-113 行的 "Step 4: SAPIEN 验证" 与第 649-677 行的 "Test 3" 只覆盖物理验证，未包含本次新增的 GIF 可视化；该文档第 3-4 行状态标注为"代码已实现"但 9.5 节测试状态仍全部为"⏳ 待执行"，实际已跑通，建议同步。
- `TECHNICAL_DOCUMENTATION.md`: 仍停留在 stage5，未涵盖 stage61 的 URDF 生成与 GIF 可视化栈。

### 已知遗留（属上游数据问题，非本次脚本缺陷）
- `2chair-bottle_vggt_omega` 的 `scene.urdf` 初始位姿本身重叠（bottle_0 在 (0.02, 0.87)，chair_1 在 (0.02, 1.27)），物理后 bottle 被chair_1 完全遮挡，GIF 中不可见。
- `stage61_urdf_generator.py::merge_to_scene_urdf` 复制 mesh 时产生双前缀文件名（如 `bottle_0_bottle_0.obj`），因目标目录已带物体名；不影响功能但命名冗余。
- `validate_with_sapien` 的 `stable` 判定仅为 `z > -0.01`，叠放物体（如 toy_0 final_z=2.93）也会判为stable，未真正校验支撑关系。
- `mainv2.py` / `mainv3.py` 仍未接入 stage61（无 `--enable_stage61`），URDF 生成与 GIF 需手动跑独立脚本。

---
## [2026-10-02 15:40] - stage61_gif 修正：忠实呈现 stage5 位姿 + 取景/地面修复

### Changed
- `tools/stage61_gif.py`:
  - **新增 `--drop` 开关，默认改为忠实呈现 stage5**：原实现把所有物体设为 dynamic自由落体，破坏了 stage5 的语义摆放（on_top / inside / against_side），渲染结果并非 stage5 成果。现默认物体为 static、位姿严格等于 URDF 初始值；`--drop` 保留原落体行为，仅用于检查 URDF 物理参数。
  - **新增位姿漂移校验**：输出每个物体的 `pose_drift` 与 `max_pose_drift` 并打印结论。stage5 模式应恒为 0（实测 4 个场景均为 0.00e+00），drop 模式 >0（实测 0.66m）——这是判断物理是否真的作用在 URDF 上的硬指标。
  - **取景改用真实顶点投影**：原先按 link 原点 + 包围盒半径 framing，扁平物体（25cm 浅盘，半径 0.178m）会撑大包围盒把相机推远导致物体显小。现按抽稀后的真实顶点（每物体≤400 点）投影求所需距离与画面覆盖率。
  - **方位角自动搜索**：原先按 X/Y span 猜方位角，但 `7_vggt_omega` 的 yspan 仅 0.028m 而 xspan/zspan≈0.5m，沿窄轴看会把长轴完全投影掉、糊成一条线。现对 180 个候选方位角搜索投影覆盖率最大者。
  - **相机中心改用顶点包围盒中心**：原先用 link 原点均值，198cm 高椅子的原点远离几何中心，导致相机偏离、画面出画。
  - **地面下沉到场景最低点**：原先固定在 z=0，当 stage5 把物体放在地面以下时（如 `laptop_vggt_omega` 的 table_0 桌面在 z=-0.77、`2chair-bottle` 整体在 z<0），俯视视线被地面遮挡导致画面全空。
  - 地面尺寸随相机距离自适应（原固定 40m，环绕低角度时会看到地面盒子边缘）。
  - 报告新增 `mode` / `pose_drift` / `max_pose_drift` 字段。

### 验证（4 个场景，位姿漂移均为 0.00e+00）
- `7_vggt_omega`（8物体）coverage 58%
- `7a_vggt_omega`（8 物体）coverage 59%
- `laptop_vggt_omega`（3 物体）coverage 70%
- `2chair-bottle_vggt_omega`（4 物体）coverage 29%
- `--drop` 模式单独验证：`7_vggt_omega` 漂移 0.66 m，物理正常作用。
- GIF 帧序列检查：4 个场景全部帧帧有变化（59/59、48/49 非零帧），无空白帧。

### 关键结论：URDF 位姿忠实于 stage5
对比 `scene.urdf` 的 joint origin 与 `final_scene_stage5.glb` 的世界质心：**8 个物体偏差 ≤ 0.08mm**，且每个 visual mesh 相对 link 原点严格居中。URDF 生成链路本身正确，此前 GIF 画面不正确是因为脚本让物体自由落体破坏了摆放。

### ⚠️ Docs to Review
- `docs/design/stage61_urdf_plan.md`: 9.5 节测试状态仍全部为"⏳ 待执行"，实际已跑通 4 个场景；Step 4 与 Test 3 未包含 GIF 可视化。

### 已知遗留（属上游 stage5 数据问题，非脚本缺陷）
- `laptop_vggt_omega`: table_0/table_1 两个 85cm 桌面重叠 43%，且整个场景位于地面以下（z 至-0.99）；笔记本悬浮于倾斜桌面上方。
- `2chair-bottle_vggt_omega`: bottle_0 被 chair_0 100% 包含、chair_0/chair_1 重叠 40%，瓶子在图中完全不可见。
- `7a_vggt_omega`: toy_1/toy_3 初始位姿 100% 重叠。
- `validate_with_sapien` 的 `stable` 判定仅为 `z > -0.01`，叠放物体也会判为 stable，未校验支撑关系。

---
## [2026-10-02 16:20] - 修复 URDF 生成丢失物体朝向的 bug（关键）

### Fixed
- `src/stage61_urdf_generator.py` (`extract_objects_from_glb`, 第 213-220 行附近):
  **导出 OBJ 时应用 `world_matrix` 的旋转**。此前 `centered_mesh = geom.copy()` 直接导出原始几何，
  从未应用 node transform 的旋转；而同一函数里 `position` 却用旋转后的质心计算 —— 位置与网格
  坐标系不一致，`merge_to_scene_urdf` 的 joint rpy 又写死 `0 0 0`，导致旋转彻底丢失。
  同时 `bbox_min/bbox_max` 改为基于 `centered_mesh`（与导出 mesh 同坐标系）。

### Bug 根因
stage5 的 GLB 里每个节点的 transform 都是一个 **-90°X 的 Y-up → Z-up 坐标系转换**（translation 为 0，
物体位置已烘焙进顶点）。诊断脚本对 4 个场景 22 个物体确认：GLB node rot 全为 90°，
而 URDF 的 rpy 全为 `0.0000 0.0000 0.0000`，且 URDF mesh 的主轴与 GLB 原始几何完全一致
—— 证明该旋转从未被应用。

### 症状（此前被误判为 stage5 的问题）
URDF 里所有物体相对 joint origin 躺倒 90°，表现为"桌椅叠放/ 笔记本悬浮在斜面上"。
`7_vggt_omega` 修复前 8 个物体主轴**全是 y**、z 高度仅 2~8cm（全部平摊在地面）。

### 修复效果（7_vggt_omega，同一 GLB 前后对比）
| 指标 | 修复前 | 修复后 |
|---|---|---|
| 主轴为 z（直立）的物体 | 0/8 | **7/8** |
| toy_2 高度 | 0.054 m | **0.121 m** |
| plate_0 高度 | 0.025 m | **0.251 m** |
| plate_0 尺寸 | 25×25×2.5cm | 25×2.5×25.1cm（标准浅盘） |

- 验证脚本：导出的 OBJ 与 GLB 变换后几何 `np.allclose(atol=1e-4)`，4 场景 22 物体**全部 OK**。
- 重跑 `7_vggt_omega`（8 物体，1123s，0 错误）与 `laptop_vggt_omega`（3 物体，281s，0 错误），
  重新体检 `7_vggt_omega`：**0 问题、无穿插**；画面中物体恢复立体形态，笔记本屏幕直立坐在水平桌面上。
- `block_1` 修复后主轴仍为 x，属该物体本身横躺的合理几何，非 bug。

### 结论修正
此前 CHANGE_LOG 中"stage5 摆放错误"的判断**不成立**。stage5 的 GLB 数据正确，
问题在 stage61 转换阶段丢失了旋转。

### ⚠️ Docs to Review
- `docs/design/stage61_urdf_plan.md`: 第 21-33 行的 Step 1 流程图与第 6.2 节描述"居中 mesh"时
  未提及需要应用 node transform 的旋转，建议补充该约束。

## [2026-10-03 21:20] - 物理仿真切到 GPU PhysX（stage61 两个脚本）

### Added
- `tools/stage61_gif.py`: 新增 `--gpu` 开关。新增 `create_scene(use_gpu)` 统一创建场景；
  `simulate()` 返回 `(trajectory, backend_name)`；报告新增 `physics_backend` 字段；
  文件头补充 GPU 使用说明与依赖库位置。
- `src/stage61_urdf_generator.py`: `validate_with_sapien()` 新增 `use_gpu` 参数；
  `run_stage61()` 新增 `use_gpu` 参数并串联；CLI 新增 `--gpu`；报告新增 `physics_backend`。

### 环境准备（关键，非代码可解决）
GPU PhysX 需要 `libPhysXGpu_64.so`。SAPIEN 的 `physx.enable_gpu()` 会自动从
GitHub 下载，但本机 **github.com 被封**（DNS 解析正常、TCP 握手超时）。
排查发现外网**部分可用**：pypi / 清华镜像 / 阿里云镜像 / gitee 均 200，
最终通过 GitHub 代理 `gh-proxy.com` 下载成功：

```bash
curl -sSL -o physx_gpu.zip \
  "https://gh-proxy.com/https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip"
unzip -o physx_gpu.zip -d ~/.sapien/physx/105.1-physx-5.3.1.patch0/
```

装入 226MB 的 `libPhysXGpu_64.so`（x86-64 ELF）后 `enable_gpu()` 即可正常返回。

### GPU PhysX 使用的三个必要步骤（实测确定，缺一不可）
1. `physx.enable_gpu()` —— 仅加载库，`scene.physx_system` 仍是 `PhysxCpuSystem`
2. `sapien.Scene([PhysxGpuSystem(), sapien.render.RenderSystem()])` —— 必须显式传入
   两套 system；只传 GPU system 会丢失 render system，导致
   `failed to get system: no system with name [render]`
3. `gpu_system.gpu_init()` —— **必须在所有 actor 添加完毕后调用**。
   实测四种时机中只有「ground → actor → init」能正常 step；提前调用报
   `failed to step: gpu simulation is not initialized`

另外 GPU 模式位姿存于显存，每步后必须 `gpu_system.sync_poses_gpu_to_cpu()`，
否则 `get_pose()` 读到过期数据，表现为**物体悬空不落**（实测 z恒为 1.0）。

### Fixed
- `src/stage61_urdf_generator.py`: `_load_actor_from_urdf_simple()` 与
  `validate_with_sapien()` 补`import sapien.render` —— 原先只 import
  `sapien.core` 和 `sapien.physx`，访问 `sapien.render` 会抛 AttributeError
  并被 except 静默吞掉，导致 `--gpu` 永远悄悄回退到 CPU。
- `validate_with_sapien()` 的回退日志补上异常类型（原先只打印消息，无法定位）。

### 验证
- CPU/GPU 一致性（`7_vggt_omega`，drop 模式 300 步）：最大位姿漂移
  CPU 0.6325 m vs GPU 0.6332 m，相对差 **0.11%**。
- 单体自由落体 100 步：CPU z=0.0449、GPU z=0.0437（CPU 参考值 0.0449）。
- `validate_with_sapien` GPU 路径：`physics_backend=PhysxGpuSystem`，8 物体 0 错误。
- `stage61_gif.py --gpu`：drop 模式漂移 0.6332 m，stage5 模式漂移 0。
- 数值稳定性差异：`toy_3`（初始 z=0.635 悬空）CPU 下穿透地面弹飞到 z=5.48，
  GPU 下0.0265 正常落地 —— GPU 稳定性更好，非 bug。

### 已知限制
- `physx.enable_gpu()` 全进程只能调用一次，且必须在任何 PhysX 代码之前。
  同进程内先跑 CPU 再跑 GPU 会报
  `GPU PhysX can only be enabled once before any other code involving PhysX`，
  故 CPU/GPU 对比测试需分进程执行。
- GPU 库安装在 `~/.sapien/`（用户目录），清理 home 或换机器需重新下载。
- Qwen VLM 推理必须 CUDA 可用；在受限沙箱环境下会报
  `cudaGetDeviceCount() Error 304` 并导致 URDF 生成失败。

## [2026-10-06 12:55] - 物体 3D 轨迹: 跑通 object_tracking 管线 + 新增轨迹可视化

### 问题
`object_tracking/run_pipeline.py` 从未成功跑通, 物体 6DoF 轨迹 npz 无产出;
且仓库内只有手腕轨迹可视化 (`wrist_trajectory_3d.py`), 没有物体轨迹可视化。

### Changed
- `object_tracking/run_pipeline.py`:
  - 删除 `run_point_tracking(object_masks=...)` 调用 (该形参在 `02_point_tracker.py`
    的函数签名中不存在, 必然 `TypeError`)。
  - 删除随之孤立的 `find_object_masks()` / `load_object_masks()` 与 `--masks_file`
    参数 (02模块不接收物体 mask, query point 是从动态 mask 减手部 mask 采样的)。
- `object_tracking/02_point_tracker.py`:
  - `unproject_tracks_to_3d()` 修复 intrinsic 维度错误。VGGT4D 逐帧预测 FoV,
    `intrinsic` 实际形状是 `(S,3,3)`, 原代码按 `(3,3)` 取 `[0,0]`, 导致 `fx` 变成
    长度 3 的数组, 报 `ValueError: inhomogeneous shape`。现按帧取 `K = intrinsic[t]`,
    同时兼容 `(3,3)`。

### Added
- `tools/analysis/object_trajectory_3d.py`: 物体 3D 轨迹可视化。
  读 `trajectories/*.npz` + `grasp/*_grasp.npz`, 输出 GLB (轨迹折线 + 红→蓝时间染色
  质心点 + 抓取闭合帧黄色标记 + 可选场景灰色背景点云) 与合并 NPZ。
- `outputs/7` → `output_v2/7_vggt_omega` 软链 (让 pipeline 按 `{base_dir}/{video_name}`
  约定找到 color/depth/extrinsics/glb)。

### 验证
`run_pipeline.py --video HaWoR/example/7.mp4 --skip_simulation` 跑通 Step 0-5:
- `outputs/7_precise/trajectories/object.npz`: trajectory (113,4,4), 113/113 帧有效
- `outputs/7_precise/grasp/object_grasp.npz`, `point_tracks.npz`, `vggt_predictions.npz`
- `outputs/7_precise/object_trajectories.glb`: 15 geometry / 200347 顶点, 可正常加载
- `object_trajectory_3d.py` 实跑通过; 三个改动文件 `py_compile` 通过

### ⚠️ Docs to Review
- `object_tracking/USAGE.md`: 仍描述已删除的 `run_tracking.py` / `--masks_file` /
  `--vggt_dir` / `vis_trajectory/` / `trajectories_3d.glb`, 实际入口是 `run_pipeline.py`。
  需同步。
- `tools/analysis/README.md`: 未收录新增的 `object_trajectory_3d.py`。需同步。

### 遗留问题 (本次未修, 与轨迹可视化无关)
- Step 6 仿真 (`--skip_simulation` 关闭时启用) 传 `env=None, robot=None` 给
  `run_simulation()`, 而后者无条件调用 `env.step()` → `AttributeError`。
  `build_scene()` 已导入但从未调用, 场景从未构建。
- `05_closed_loop_verifier.run_closed_loop_verification` 每轮迭代复用同一份
  `sim_traj`, 只调整 `current_traj` 而不重新仿真, 误差必然单调下降, 不是真闭环。
- scene 7 轨迹在第 79 帧有 0.22m 跳变, 源自 VGGT4D 深度预测抖动
  (原始 `tracks_3d` 同样), 非可视化脚本引入。

---
## [2026-10-06 15:20] - VGGT-Omega 同源物体 6DoF 轨迹(替代 VGGT4D 方案)

### 背景
上一条记录用的是 `object_tracking/run_pipeline.py`, 它内部用 **VGGT4D** 重新预测
深度与相机, 与场景 `final_scene.glb` 不同源, 轨迹会与场景错位。本次改为
直接复用 mainv2 落盘的 VGGT-Omega 结果, 保证轨迹与场景严格同源。

### 关键发现
`output_v2/7_vggt_omega/_pre_dedup_masks.pkl` (1GB) 里同时存了
- `all_masks`: 逐帧实例 mask (S, 384, 688)
- `world_points`: VGGT-Omega world_points (113, 384, 688, 3)

即 mainv2 算质心用的同一份数据, 无需重跑任何模型即可得到同源轨迹。

### Added
- `tools/analysis/object_trajectory_6dof.py`: VGGT-Omega 同源 6DoF 轨迹。
  首帧点云定义为物体局部系, 逐帧**时序链式 ICP**(cKDTree 最近邻 + trim 20%
  + Procrustes) 求位姿, 再 Savitzky-Golay 平移 + Slerp 平滑旋转。
  输出 `<obj>_traj.npz` (trajectory/centroids/local_origin/valid_frames) 与
  `object_trajectories_6dof.glb`。

### 实现踩坑(重要)
最初实现是"逐帧独立对参考帧做 RANSAC + Procrustes", 用 `ref_pts[:n]` /
`cur[:n]` 按索引截断配对。**这是错的**: 两帧点云之间根本不存在点对应关系,
按索引硬配对等于把随机配对当成正确对应, 解出的旋转是纯噪声 ——
实测帧间旋转跳变 100°~4000°, path_length 14m 而净位移仅 0.2m。
改为时序链式 ICP 后(以上一帧位姿为初值)残差降到 2~6mm, inlier_ratio 0.89~1.00。

第二个坑: 一度把 `trajectory[:3,3]`(物体局部系原点的世界位置)当成"物体质心"
去画轨迹线和算path_length, 造成 max_step 0.24m 的假象。实际世界质心是
mask 内点云均值, 二者固定相差约 0.51m。npz 现在同时存 `centroids`(世界质心,
与 mainv2 同定义) 和 `local_origin`(摆mesh 用), 避免再混淆。

### 验证 (scene 7_vggt_omega, 113 帧)
11 条轨迹全部产出(block/plate/star/toy), 每条 npz:
- ICP residual 2.0~6.4mm, inlier_ratio 0.89~1.00
- 旋转矩阵 det=1.0000, 正交误差 <1.2e-15
- 静止对照 `plate_inst0`: net_displacement 5mm, 旋转 1.3°, max_step 1.2cm
- 动态 `toy_inst1`: net_displacement 0.342m, max_step 6.1cm
- 交叉验证: 本脚本 `centroids` vs mainv2 `trajectory_info.json`
  平均差 **8.4mm** / 最大 3.4cm → 同源同定义确认
- GLB: 952 geometry / 31612 顶点, trimesh 可正常加载
- `py_compile` 通过

###⚠️ Docs to Review
- `tools/analysis/README.md`: 未收录 `object_trajectory_3d.py` 与
  `object_trajectory_6dof.py`。需同步。

### 备注
`object_trajectories_3d.py`(上一条记录的 VGGT4D 版本) 现已不推荐使用,
与场景不同源。可保留作对照或删除。

---
## [2026-10-06 15:55] - 轨迹质量门控: 用反投影验证替代"伪旋转"启发式(结论推翻前次判断)

### 背景
上一条记录里我凭"净位移接近 0 却出现大幅旋转"判定 toy_inst2/toy_inst4/toy_inst5
等轨迹"不可信", 并提议加门控。**本次用反投影实证推翻了这个判断。**

### 反投影验证方法
把轨迹声称的物体表面点投回各帧图像, 看是否落在该帧物体 mask 内。
ICP 残差低不能证明这一点(残差只比较点云自身), 只有投回图像才能。

三个关键坑(都已修正并固化到代码注释):
1. **外参约定**: `extrinsics/{i}.txt` 是 world-to-camera (p_cam = R@p_world + t)。
   用 depth + world_points 回归内参判定: w2c 得 fx=282.1, 与落盘均值 284.6 吻合,
   cx/cy=344/192 完全一致; c2w 口径下 fx 退化为 0。
2. **内参是逐帧的**: vggt-omega 反投影用逐帧内参 (predict.py:192-195), 但落盘的
   `intrinsic.txt` 是各帧均值 (predict.py:56)。必须按 world_points + 外参反推
   逐帧内参, 否则有投影偏差。
3. **参考点必须用 centroids 而非 trajectory[:3,3]**: 后者是物体局部系原点,
   与世界质心固定相差约 0.5m。误用它会导致 90 像素的系统性偏移, inler 率算成 0%。

### 验证结果 (scene 7_vggt_omega, 11 条轨迹)
| object | occl% | reproj_inlier | net_disp | rot |
|---|---|---|---|---|
| plate_inst0 | 0.0 | 87.5% | 0.005m | 1.3° |
| toy_inst1 | 0.0 | 85.1% | 0.342m | 23.8° |
| toy_inst2 | 16.2 | 84.2% | 0.023m | 24.8° |
| toy_inst4 | 26.5 | 73.9% | 0.025m | 27.8° |
| toy_inst5 | 12.9 | 66.9% | 0.032m | 61.7° |

inlier 率与遮挡率明显负相关, 说明手部遮挡确实降低了精度, 但远没到"轨迹不可用"
的程度 —— SAM3 视频传播保住的可见区域仍足以约束位姿。**我此前的判断是错的。**

### Changed
- `tools/analysis/object_trajectory_6dof.py`:
  - 新增 `reprojection_inlier_ratio()` + `recover_intrinsic()`: 反投影质量验证。
  - `assess_trajectory_quality()` 改为以 inlier 率为唯一否决项;
    移除基于 net_displacement/rotation 的启发式(该判据会误杀 inlier 达86% 的轨迹)。
  - 新增 `detect_occluded_frames()`: mask 面积 < 中位数*ratio 判为遮挡帧并丢弃。
  - npz 新增 `trusted` / `occluded_frames` 字段; 新增 `quality_report.json`;
    GLB 只绘制 trusted 轨迹。
  - CLI: `--min_inlier_ratio` (取代 `--max_frame_rot_deg` / `--max_net_disp`),
    `--occlusion_area_ratio`, `--no_occlusion_filter`, `--instance`。

### 验证
11/11 trusted, inlier 66.9%~90.8%; `py_compile` 通过。

### 附: do-as-i-do 的遮挡处理调研结论
- 用 SAM3 的 `propagate_in_video` 做掩码时序传播
  (`reconstruction/scripts/run_sam3_video.py:349-358`), 物体用点击提示、手用文本提示,
  两者是**独立会话, 没有从物体 mask 里扣除手部**(`run_pipeline.sh:69-86`)。
- TAPIR 做逐帧点轨迹, 用 `occlusion` logit + `expected_dist` 双重可见性判据
  (`tapir_velocity_tracking.py:351-353`), 且要求预测落点同时落在下一帧物体 mask 内
  (`:359-372`) —— 预测与分割互相验证, 与本脚本的反投影判据思路一致。
- 位姿由 Fast-SAM3D 的 `track_object.py`(扩散跟踪) 决定, 输出喂给
  `--rotvel_json` 当旋转速度先验。该文件在未 checkout 的 submodule 内, 无法验证。
- **无 re-identification**(全仓库 grep 零命中); 坏帧用线性/SLERP 插值
  (`process_dataset.py:605-686`), 明确不做低通平滑(`:39-40`)。

---
## [2026-10-06 17:05] - 物体轨迹 3D GIF: 按 6DoF 轨迹驱动 mesh 渲染(stage6 轨迹版)

### 背景
`tools/stage61_gif.py` 的物体位姿来自 SAPIEN 物理仿真(`simulate()`), 物体是
"模拟"出来的。本次让物体按 VGGT-Omega 观测轨迹运动, 反映真实操作。

### Added
- `tools/analysis/trajectory_gif.py`: 读 `object_trajectories/*_traj.npz`,
  复用 `stage61_gif.render_frames()` 渲染 GIF。
  运动物体按轨迹驱动, 其余 URDF link 每帧保持初始位姿作为空间参照。
  输出 `trajectory_gif/scene_trajectory.gif` + `report.json`。

### 实现要点
1. **位姿参考点必须用 `centroids`(世界质心), 不能用 `local_origin`**。
   `local_origin` 是首帧点云的局部系原点, 数值在原点附近(0.0006m), 而 URDF
   link 位姿在 (-0.43,-0.45,0.04) —— 直接用会让物体错位半个场景。
2. **实例编号不是同一套, 必须按位置匹配**。`_pre_dedup_masks.pkl` 是去重**前**
   的实例(存在同一物体被多类别重复分割: star_inst0 / block_inst0 / toy_inst3
   首帧质心几乎相同), URDF 是去重**后**的 8 个 link。脚本按首帧质心做贪心
   最近邻匹配(`match_traj_to_links`, max_dist=0.15m), 保证一个 link 只被一个
   实例占用。全景模式下 7 个匹配成功、4 个 UNMATCHED(重复实例被排除)。
3. **单物体渲染缺参照物且取景过远**。只渲运动物体时自动取景按 0.34m 位移把
   相机推到 0.82m, 物体仅占画面 ~5%。改为渲染全部 8 个 link 后,
   `visible=8/8`, 场景关系清晰可辨。
4. 坐标系与 URDF 一致(均为 SAPIEN world, Z-up, floor z=0), 无需额外转换。

### 验证 (scene 7_vggt_omega)
- `toy_inst1`(唯一 dynamic 物体, net=0.342m): GIF 40 帧 512x512, 310KB。
  目视确认首帧在场景中上方、末帧移入左侧盘子, 即抓放动作; 其余 7 物体静止。
- 全景模式: 7 个运动物体 + 1 个静态参照, 13 帧(common valid frames 交集)。
- `py_compile` 通过。

### ⚠️ Docs to Review
- `tools/analysis/README.md`: 未收录本次新增的
  `object_trajectory_6dof.py` / `trajectory_gif.py`。需同步。

---
## [2026-10-06 21:30] - 物体轨迹可视化: 投影回原视频 GIF + matplotlib 3D 轨迹图

### 坐标系(用户明确要求确认)
- **世界坐标系**: VGGT-Omega world 系, Z-up, 地板z=0。与 `extrinsics/`、
  `centroids`、URDF 完全一致, 3D 轨迹图直接用, 不做任何变换。
- **相机系**: `extrinsics/{i}.txt` 是 **world-to-camera** (p_cam = R@p_world + t),
  z_cam 即深度。用回归内参判定过: w2c 口径得 fx=282.1 与落盘均值 284.6 吻合、
  cx/cy=344/192 完全一致; c2w 口径下 fx 退化为 0。
- **内参是逐帧的**: vggt-omega 反投影用逐帧内参(predict.py:192-195), 但落盘
  `intrinsic.txt` 是各帧均值(predict.py:56)。脚本按 world_points + 外参反推
  逐帧内参, 与反投影口径一致。

### Added
- `tools/analysis/trajectory_video_overlay.py`: 一次产出三样。
  1. `trajectory_overlay/overlay/<obj>.gif` — 物体位姿叠加回原视频帧:
     青色点云为物体投影轮廓(首帧 mask 内 800 个表面点按逐帧位姿变换后投影,
     与轨迹严格同源)、橙色渐变尾迹为历史路径、紫点为当前质心、
     绿色 START / 橙色 END 标记首尾、左上角信息栏显示首尾世界坐标与净位移。
  2. `trajectory_overlay/traj3d/<obj>.png` — matplotlib 三联图(世界坐标系):
     3D 轨迹线(turbo 渐变着色)+ 起止球 + 时间标记 / X-Y-Z 位移曲线 /
     roll-pitch-yaw 欧拉角曲线。
  3. `trajectory_overlay/trajectory_report.json` — 首尾世界坐标、首尾欧拉角、
     净位移、路径长度、最大步长。

### 验证 (scene 7_vggt_omega, 11 个实例全部产出)
`toy_inst1`(唯一 dynamic 物体):
- start world = [-0.2923, -0.4111, 0.0270]  euler = [0.6, 0.1, 0.1]
- end   world = [-0.1373, -0.1068, 0.0233]  euler = [-21.9, -7.6, 7.2]
- displacement = 0.342 m, path = 1.064 m, max_step = 0.061 m
目视确认 GIF: START 绿圈落在黄色玩具上, 青色点云勾勒出被手握住的物体形状并
随手臂移动, END 橙圈落在蓝色盘子上。3D 图显示 Z 先升至 0.22m 再落回、
roll 转了约 125°, 即"拿起→抬起→移动→放下"。

`plate_inst0`(静止对照): displacement = 0.005 m, 欧拉角变化 <1.5°, 符合预期。

### 命令
```bash
python trajectory_video_overlay.py --obj toy_inst1 --n_frames 60   # 单个
python trajectory_video_overlay.py --n_frames 50                   # 全部
```
选项: `--no_mesh` 只画质心与尾迹, `--n_surface_points` 轮廓点数,
`--trail_len` 尾迹长度。

### 产物大小提示
11 个 GIF 共 60MB(单文件约 5.4MB / 50 帧 384x688 未缩放)。若需更小,
可加降采样参数。

---
## [2026-10-06 22:40] - 轨迹可视化画廊 HTML

### Added
- `tools/analysis/make_trajectory_gallery.py`: 把 `trajectory_overlay/` 的产物汇总成
  单文件 HTML 画廊, 沿用仓库既有 `output_v2/stage61_gif_gallery.html` 的形式。
  - 三个视图切换: 视频叠加 / 3D 轨迹图 / 并排
  - 两种排序: 按净位移 / 按名称
  - 汇总栏: 物体数、动态物体数、最大净位移、最低反投影 inlier
  - 图例说明各叠加元素含义; 卡片内显示首尾世界坐标 + 首尾欧拉角 + 标签
  - 点击放大(dialog)
  - 自动合并 `trajectory_gif/report.json` 的 URDF link 映射与有效帧数,
    `object_trajectories/quality_report.json` 的反投影 inlier
  输出: `trajectory_overlay/trajectory_gallery.html` (19.8 KB, 11 个物体)

### 修复的两个字段 bug
- `n_valid` 字段只在 `trajectory_gif/report.json` 里有, `trajectory_report.json`
  没有 → 卡片渲染出 `undefined 有效帧`。改为 link_map 补充 + 空值时不渲染该标签。
- `urdf_link` 为 null 时显示 `(未匹配)` → 改为 `—`。

### 验证
浏览器首次快照确认页面正常渲染(标题/汇总栏/卡片/图例齐全, 11 卡片 11 GIF),
修完字段 bug 后复查: `toy_inst1 → toy_0`、`113 有效帧`、`inlier 85%`。
离线校验脚本(占位符已替换 / 字段齐全 / 22 个资源文件存在 / 标签闭合)全部通过。

### 注意: 打开方式
GIF 单文件约 5.4MB, 浏览器会发Range 请求;Python 内置 `http.server` 是
HTTP/1.0 **不支持 Range**(返回 200 而非 206), 大 GIF 会加载失败。
直接双击 `file://` 打开最简单; 若需 HTTP 服务, 需用支持 Range 的服务器。

---
## [2026-10-06 23:00] - 轨迹 GIF 改进: 完整 mask 点云轮廓 + 全程轨迹线

### 用户反馈
"物体位姿可以完整地复现在轨迹上" —— 原实现只画 800 个采样点的散点,
且轨迹线只画最近 40 帧, 物体形状和完整路径都看不清。

### 关键实验: 为什么用 mask 点云而不是 URDF mesh
实测投影命中率(把点投回该帧图像看是否命中物体 mask):
| 方案 | inlier |
|---|---|
| A) mesh 顶点直接当世界坐标 | 0.0% |
| B) mesh 顶点 + URDF link_xyz | 15.0% |
| C) mesh 顶点重定位到轨迹质心 | 73.3% |
| D) mask 世界点(真值) | **100.0%** |

结论: URDF mesh 是重建结果, 与真实分割有形状差异, 即使正确重定位也只有 73%。
轨迹本身就是从 mask + world_points 估计的, 所以用 mask 点云最同源最准。

### Changed
- `tools/analysis/trajectory_video_overlay.py`:
  - 物体投影改用**完整**首帧 mask 点云(默认 `--n_surface_points 0` = 不抽样),
    半透明填充 + `findContours` 最大轮廓描边, 形状清晰可辨。
    实现: 投影点 → 形态学闭运算连成实心区域 → 开运算去噪 → 取最大连通域描边。
  - 轨迹线改为**全程绘制**(`--trail_len 0` = 默认): 已走过部分橙色实线,
    未走到部分灰色细线, 一眼看出物体往哪走。
  - 预投影全部轨迹点(每帧用该帧自己的外参), 避免每帧重复计算。
- `tools/analysis/make_trajectory_gallery.py`: 图例更新为新配色。

### 验证
`toy_inst1` 50 帧: 目视确认物体轮廓为实心+青色描边, 形状随位姿变化清晰
(frame 0 小手雷形 → frame 45 被手遮挡 → frame 112 圆形轮廓落在盘子上),
完整轨迹线全程可见。11 个物体全部重生成, 画廊页重新生成。
`py_compile` 通过。

---

## [2026-10-07 15:20] - 修复"关系为 floor 却悬空"(新增 5.25 落地补正)

### 问题 (用户报告: 渲染图里很多物体漂浮, 且 Stage5 修复后仍明显)

全量扫描 31 个 egodex 场景的最终位姿, 判据「关系为 supported by floor 但
min_z > 0.02」, 命中 6 个实例:

| 场景 | 实例 | 悬空高度 |
|---|---|---|
| egodex_4 | cloth_0 | 0.1065 m |
| egodex_22 | bracelet_0 | 0.0277 m |
| egodex_25 | block_0 | 0.0892 m |
| egodex_30 | stapler_0 | 0.1160 m |
| egodex_32 | screwdriver_0 / toy_1 | 0.0840 / 0.0714 m |

### 根因: 基础精修读的是 Stage1 类别级关系, 而关系到 5.1 才被推翻

`mainv3.py` 基础精修循环用 `categories_and_relations[category]` (Stage1 输出),
而 Stage1 把 cloth_0 判为 `attached to wall` → 走挂墙分支, 落地逻辑根本没被调用。
日志时间线证实:

```
20:21:24  基础精修: cloth_0 → attached to wall   ← Stage1 旧关系
20:21:26  GLB 保存
20:23:47  Stage 5 开始 → 5.1 改判 supported by floor (已太晚)
```

之后 5.2 的穿模修复只做「分开」不做「落地」, 对 cloth_0 反复触发十余次
(每次推 0.05~0.10m), **把悬空推得更高** → 用户看到「修了很多次还是飘」。

### 关键发现 1: 不能复用 refine_supported_by_floor_object (会意外旋转物体)

先直接调用现有函数测试, 5 个实例被旋转 109~119°:

```
egodex_30/stapler_0  Δrot=116.12°
egodex_25/block_0    Δrot=115.46°
```

原因: 该函数会把物体 y 轴对齐重力方向 (`sp_refinement.py` L72-84),
这是为落地家具 (椅子翻倒扶正) 设计的, 对手持小物件会转到别的朝向上。

### 关键发现 2: T 矩阵内含缩放, 用矩阵分解判旋转会得到伪值

`det(R)` 实测 0.008 / 0.010, `R @ R.T != I` —— 不是纯旋转矩阵。
`trace(R0.T @ R1)` 算出的「118°旋转」是缩放污染的伪值, 真实旋转为 0。
副作用判据改用**顶点集直接比对** (包围盒三边长 + 顶点相对包围盒中心的偏移)。

### 修改

**`src/sp_refinement.py`** — 新增 `ground_floating_objects()`:
- 只改 `T[2,3]` 一个元素, 不动旋转 / 水平位置 / mesh 顶点
- `float_eps=0.02` (判定悬空阈值) / `post_tol=0.01` (落地残差)
- 试算+ 校验通过后才写回 T, 避免半途失败留下脏 T

**`mainv3.py`** — `run_stage5()` 新增 5.25 落地补正 (5.2 之后, 穿模修复之前):
- 用 5.1 输出的**实例级** `refined_relations` (基础精修当时只有类别级)
- 位置必须在 5.2 之后: 穿模修复会推高悬空

### 验证 (5 个 egodex 场景, 几何副作用全面比对)

调用**集成后的真实函数** (非测试脚本副本), 深拷贝跑, 前后顶点集比对:

```
落地总数: 6   问题数: 0   (通过)
Δxy 全为 0.00e+00 | 落地 z 全为 ±0.0000 | 尺寸/形状变化 < 1e-17
关系非 floor 的实例全部未被改动 (逐实例断言)
```

`py_compile` 通过。验证覆盖5 组, 满足「至少 5 组 egodex」要求。

### 未解决 (待用户决策)

尺寸校准的点云尺度问题: 实测 4 个场景物体系统性偏大 1.58~2.23x
(苹果 39cm / 碗 32-41cm, 真实 6-10cm / 12-25cm)。
因缺绝对尺度锚点, 校准逻辑在原理上无法自证正确。HOI4D 官方
Annotations 与 CAD 包的 OneDrive 链接已全部失效 (403 accessDenied / 404),
无法获取尺寸真值。本轮未改动尺寸模块。

---

## [2026-10-07 19:50] - 修复 on_top 摆放被穿模修复破坏 (101/104 个 on_top 关系全坏)

### 问题 (用户质疑: "你确定都 top 都放好了吗?")

上一轮只验了 floor 关系, 漏查 on_top。补测发现**全库 104 个 on_top/inside 关系
只有 3 个贴合, 坏 101 个** (gap 中位 26.6cm, 最大 192.4cm)。

### 实测证据 (egodex_33, 破坏最严重)

5.2 的 on_top 精修本身是**正确的**:

```
02:14:23  [on_top] 大精修: z_offset=+0.0360m, 完整贴合← lego_0 贴到 plate_0 顶面
```

但随后的穿模修复把它推回去了:

```
02:22:57  穿模修复: toy_1 ↔ lego_1 | z轴分离 0.0056m   ← toy_1 被推 9 次 × 0.0056 = 0.050m
最终      lego_0 底 z=0.0000, 而 plate_0 顶面 = 0.0360
```

23 次穿模修复事件: **100% 是 z 轴分离, 100% 涉及 on_top 物体**。
`toy_1 ↔ lego_1` 单独触发 9 次。

### 两个缺陷叠加, 形成自维持循环

**缺陷 1**: on_top 把物体精确贴到支撑物顶面, **接触面间隙为 0** → 碰撞检测
必然判成重叠 → 沿 z 推开。

**缺陷 2**: 推开后物体掉到地面, 底面与地面共面 → 又被判重叠 → `bounds[0,2] < 0`
触发 `refine_inter_object_placement.py` L1508-1513 的"抬回地面"兜底 → 压到 z=0
→ 再次共面 → 循环。

排除误报的安全依据: 实测 egodex_33 plate_0 上4 个物体的 **xy 投影重叠 0%**,
不存在真穿模。

### 修改

**`tools/refine_inter_object_placement.py`**
1. `resolve_penetrations()` 新增 `on_top_pairs` 集合 (实例级), 跳过 5.1 声明的
   on_top 关系对 —— 治根, 消除误报循环
2. `refine_inter_object_relations()` 落盘 `placement_strategies.json`
   (`{实例名: on_top/inside/...}`)
   **原因**: `final_relations.json` 只写 `supported by X`, 不含策略区分。
   之前 `bread_0` 是 `inside` (面包在碗里), 但落盘后与 `block_3` (on_top)
   完全无法区分 —— 若无策略过滤, 重贴会把碗里的面包错贴到碗沿 (实测验证1
   确认了这个风险)

**`src/sp_refinement.py`** — 新增 `reseat_on_top_objects()`
- 只处理 strategy == "on_top" 的实例; `inside` / `hanging_below` /
  `against_side` 跳过 (它们的位置由各自精修函数负责)
- 只改 `T[2,3]` (gap 修正量), 不动旋转/水平/尺寸
- `gap_tol=0.03`, 校验通过才写回

**`mainv3.py`** — 5.25 扩展为"落地补正 + on_top 重贴", 读取
`placement_strategies.json` 并在缺失时打印警告

### 验证 (6 个 egodex 场景, 顶点集直接比对)

```
egodex_33  on_top  5 → 贴合 5 | 落地 0 | 重贴 5 | 副作用 0
egodex_31  on_top  4 → 贴合 4 | 落地 0 | 重贴 4 | 副作用 0
egodex_32  on_top  1 → 贴合 1 | 落地 2 | 重贴 1 | 副作用 0
egodex_4   on_top  3 → 贴合 3 | 落地 1 | 重贴 3 | 副作用 0
egodex_25  on_top  0 → 贴合 0 | 落地 1 | 重贴 0 | 副作用 0
egodex_30  on_top  2 → 贴合 2 | 落地 1 | 重贴 2 | 副作用 0
============================================================
6 场景: on_top 共 15 个 → 贴合 15 (100%) | 副作用 0
```

策略过滤专项验证 (egodex_4):
- 无策略时: `bread_0`(inside) 被错贴, gap→0 (**风险已暴露**)
- 带策略时: `bread_0` 保持 z=0.2182 未动, `bread_1`/`block_3` 正确重贴

几何副作用: 所有改动均为纯 z 平移 (Δxy / Δshape / Δsize 全部 < 1e-9)。
`py_compile` 通过。

### 影响面 (待用户决策)

本bug 意味着**当前所有已产出场景 (31 个) 的 on_top 摆放均不可信**,
gallery 的 Stage6 渲染也需重做。重跑成本约双卡并行 9 小时。

---

## [2026-10-08 02:43] - RAS + Agent 决策层接入方案(设计文档)

### 调研背景
评估 Real2Gym(Real2Sim) / R2G_Agent 与本项目 RAS 管线的结合可行性。
结论:SAPIEN 可用且 RAS 内已有接入,重建层可复用,动作层需新建。

### Added
- `docs/AGENT_DECISION_LAYER_PLAN.md`: 四阶段实施方案(基线回放 → 接口抽象 →
  双臂 stage 决策循环 → 独立评测),含阻塞清单 B1-B5 与风险表。
  已对照源码核实所有引用的接口形状。

### 调研中发现的两个既有缺陷(本次未修,仅记录)
1. **`object_tracking/simulation/run_simulation.py:151` 传 `env=None`**,
   而 `action_player.py:168` 直接调 `env.step(action)` → 必崩
   (`AttributeError`)。说明该路径从未端到端跑通,是阶段 0 的首要修复项。
2. 环境依赖缺失: 本机 `import sapien` 失败;`sys.path` 注入路径经实测**正确**
   (能解析 `galaxea_sim` 包),实际缺 `gymnasium`。

### Changed
(无代码改动,本轮仅产出设计文档)

### ⚠️ Docs to Review
- `docs/README.md`: 若需把新方案纳入文档索引,应补一条指向
  `AGENT_DECISION_LAYER_PLAN.md` 的条目。
- `README.md`: 「Object Tracking pipeline」一节提到 SAPIEN simulation,
  若阶段 0 落地后接口变化需同步。

---

## [2026-10-08 12:30] - 修复坐标系对齐 B/C/D/E/F 五项问题

### 背景

用户提供一份独立分析, 我逐条查代码核实, 结果 A~F 属实 (详见核实表)。
核实过程中发现 D (置信度阈值) 的前提被数据否证, 未实施; A 的"翻转依据"确实很弱。

### B. 失败被记成成功 (真 bug, 已修)

`mainv3.py` 五级对齐中, 前四级都用 `np.allclose(R, np.eye(3), atol=1e-6)` 判失败,
但stage4 (GeoCalib) 之后**无任何校验**, `alignment_stage` 被无条件写成
`"stage4_geocalib"`。

实测证据 (7a_vggt_omega/coordinate_alignment.json):
```
alignment_stage: stage4_geocalib          ← 记成成功
reason: geocalib_poor_alignment R[2,2]=0.3185   ← 内部说失败
R = 单位阵                                ← 实际返回值
confidence: 0.000522~0.004181              ← 几乎为零
```
JSON 里三者自相矛盾, 下游把单位阵当成功用。stage4 是最后一级, 失败后无兜底。

**修法**: 新增 `_geocalib_failed` 判定, 失败时置 `alignment_stage =
"stage4_geocalib_FAILED"` 并打印原因 + 明确告知"坐标系未对齐, 下游保持 VGGT 原始坐标系"。

### C. stage3 传错参数导致分支永不执行 (真 bug, 已修)

`mainv3.py:415` 把 `floor_masks` 传给形参 `large_plane_masks`。
`geometry_utils.align_via_large_plane` L469:
```python
enhanced_masks = list(large_plane_masks) if large_plane_masks else []
if len(enhanced_masks) == 0 and colors is not None and ...:   # ← 永不成立
    enhanced_masks = segment_large_flat_surfaces(...)# VLM 大平面分割
```
传入 floor_masks 后 enhanced_masks 恒非空 → **VLM 大平面分割分支从未执行**,
stage3 退化为 stage2 的重复品。

**修法**: 传空列表以启用该分支 (floor 平面由函数自行分割+拟合)。

### D. 置信度参与判定 (未实施 — 假设被数据否证)

分析建议"置信度低应判失败"。实测 GeoCalib 置信度:

| 场景 | 置信度中位 | 成败 |
|---|---|---|
| 7a | 0.00314 | **失败** |
| egodex_4 | 0.00270 | 成功 |
| 3chair | 0.00117 | 成功 |
| egodex_33 | 0.00079 | 成功 |

**7a(失败) 的置信度反而最高**, 所有场景都在 0.0005~0.004 量级, 差异<5倍且与成败
无相关性。推断 GeoCalib 在本数据集上普遍给出极低置信度(训练域不匹配), 用绝对阈值
会误杀全部场景。**故不实施**, 待积累更多 GeoCalib 成败样本再定。

### E. theta_gravity 门限与贴地自相矛盾 (设计缺陷, 已修)

`sp_refinement.refine_supported_by_floor_object`: L225 在 `theta_gravity ∈ (10°,170°)`
时判定"不扶正", 但 L246 **无条件**把物体贴到 z=0。物体被判定"朝向不可信"却仍被
强行摆到地面, 掩盖了上游坐标系对齐错误。

实测历史日志 290 条:
```
theta_gravity ∈ [0.1°, 177.6°]
170/290 (59%) 落在 (10°,170°) 不扶正区间, 仍全部贴到 z=0
z_min ∈ [-1.5146m, +1.7867m]  ← 负值=原在地面下 1.5m
```

**修法**: 恢复 z_min 阈值 `_FLOOR_Z_MAX = 0.5m`。超出则保持原位并告警, 让
"坐标系可能没对齐"暴露出来。

**同步修5.25**: `ground_floating_objects` 原本只判`z_min > 0.02` 就压到 0, 会把
E 跳过的 1.5m 错位物体重新拽回地面, 抵消 E 的效果。已加同样的 `float_z_max=0.5` 上界。

验证实测:
```
z_min=0.05m  → 落地 (底 z=0.0000)
z_min=1.40m  → 跳过, 保持原位, 打印告警
```

### F. URDF 转动惯量恒为占位值 1.0 (真 bug, 已修)

根源在 EmbodiedGen: `embodied_gen/validators/urdf_convertor.py:64` 模板写死
`ixx=iyy=izz=1.0`, L347-351 只更新 mass, inertia **从未计算**。
实测 URDF: `<mass value="1.6170"/><inertia ixx="1.0" .../>` — mass 真实而惯量占位。

**修法**: 由 mesh 真实几何 + 实际 mass 解析计算
`I = m/12 · diag(b²+c², a²+c², a²+b²)` (a,b,c 为包围盒边长)。

**除数修正过程**: 初版写成 `/6`, 用 trimesh.moment_inertia 交叉验证发现 2 倍偏差
(box(0.4,0.3,0.2): 公式给 0.00052, trimesh 给 0.00026)。均匀密度长方体绕质心轴的
标准解是 `/12` 而非 `/6`。修正后三组几何与 trimesh 精确一致:

```
ext=[0.4,0.3,0.2] 公式=[0.00026,0.0004,0.0005]  trimesh 同
ext=[0.1,0.1,0.1] 公式=[1.67e-06]*3              trimesh 同
ext=[0.5,0.05,0.05] 公式=[5.2e-07,2.63e-05,2.63e-05]  trimesh 同
```

### A. GeoCalib 判据丢符号 (未实施, 待用户确认阈值来源)

`geometry_utils.align_via_geocalib` L756 用 `abs(R[2,2]) < quality_threshold` 判失败,
无法区分 `+1.0`(正确) 与 `-1.0`(180°翻转)。而 `_orient_floor_normal` (L295-311)
只能做180° 翻转、**不改倾斜角**, 且其判断依据(场景质心/ 相机位置)与待判定的
floor_normal 存在循环依赖( floor_centroid 是用 floor_normal 方向算的)。

用户指出"翻转需要有依据"—— 确实如此。**建议不猜阈值, 而是先修 B 让失败暴露,
再用更多场景反推 min_cos**。未实施。

### 附带: 静默失败点排查

全库扫描发现 **45 处静默失败**(11 处高风险裸 except/except+pass, 26 处中风险)。
已给4 处主链路加日志(不改控制流, 零行为风险):
- `refine_inter_object_placement.py` VLM 投票失败 (历史 `_vlm_inference` NameError
  就藏在这里, fallback 恒返回 "on_top")
- `refine_inter_object_placement.py` optimal_frames 读取失败
- `generate_scene_json_stage1.py` 地面点云提取失败 (会导致地面剔除整体失效)
- `generate_scene_json_stage1.py` 帧图像读取失败 (会导致 Stage6 无颜色)

另修复 `sam-3d-objects/notebook/inference.py:5` 硬编码 `os.environ["CONDA_PREFIX"]`
导致 Stage3 在无 conda 环境变量时 KeyError 崩溃 (实测4 种环境组合均正常)。

---

## [2026-10-08 14:20] 修复 stage6.1 物理仿真链路并生成真实 GIF

### 问题诊断（先排查后修改）
扫描 19 个 URDF / 226 个 link 与全部 log 后，确认 stage6.1 存在 5 个问题：

1. **物理验证门控空转**（最严重）
   `physics_validator._create_object` 用 `add_multiple_convex_collisions_from_file(decomposition="coacd")`
   加载 GLB，失败则退化到 `add_convex_collision_from_file`，再失败 `return None` 静默跳过。
   151 的 `stage61_validation.json` 里29 个物体全为 `stable: true`、`final_z` 最高 **91.809m**，
   而 `errors: []`、`success: true` —— 因为没有任何 actor 参与仿真，空集与"全部稳定"无法区分。
   `_detect_unstable` 在 SAPIEN 缺失时同样 `return set()`。

2. **mass/inertia 从未生效**
   226 个 link 全部带 `<inertial>`（`real_mass = volume × density`，151 为 0.0028~6.21kg），
   但 `stage61_visualize_sapien.parse_urdf` 只读 mesh/scale/摩擦，**从不读 mass 与 inertia**，
   SAPIEN 按包围盒默认密度自算（实测 cabinet_0 被算成 71.36kg，而 URDF 是 6.2114kg）。
   且 URDF 里 `ixx=iyy=izz=1.0` 本身是 EmbodiedGen 的占位值。

3. **visual 与 collision 形状不一致**
   `add_box_visual(half_size)` 用包围盒做视觉，`add_multiple_convex_collisions_from_file` 用真mesh 碰撞。
   实测两者外形尺寸仅差约 3mm（collision mesh 是同尺寸凸包，体积大 1.4~4.1 倍），
   所以物理正确但**画面里看到的形状不是参与碰撞的形状**。

4. **z_offset 掩盖穿地**
   `z_offset = -min_z + 0.1 if min_z < 0.1 else 0.0` 把整个场景上抬。19 个 URDF 中 10 个被抬，
   151 的 `min_z=-1.296`（埋地 1.3m）被拉回 +0.100；而 `phone`/`121`/`15`/`mask` 等9 个
   `min_z>0.1` 的场景 `z_offset=0`，同管内两种行为，对比渲染自相矛盾。

5. **`stage61_sim_sapien.py` 的 SAPIEN 是摆设**
   `actor.set_pose()` 后从不 `scene.step()`，随即 `del engine`，渲染用 pyrender 画的彩色盒子，
   且 `box_half = [max(s*0.6,0.3) for s in scale]` 把缩放系数当尺寸。零步物理却输出 `stage61_sim.png`。

误报澄清：SAPIEN 3.0.0b1 实测重力沿 **-z**（物体从 `(0,0,3.0)` 落到 `z=-4.12`），是 z-up，
URDF 保留 z-up 语义正确；`stage61_render_scene.py` 里 pyrender 的 `Rot_x(-90)` Z→Y 转换也正确。

### Added
- `tools/stage61_physics_sim.py`: URDF 驱动的真实物理仿真 + GIF 输出。
  - 读 `<inertial>` 的 mass/inertia 并真正写入 SAPIEN：`PhysxRigidDynamicComponent.set_mass()` +
    `set_inertia(3 项主对角, float32)`，随后**回读校验**质量是否生效，不一致则该物体判失败。
    （SAPIEN 3.x 无 `set_mass_properties`，也无 `set_auto_compute_mass`；实测 `set_mass` 可覆盖自动计算值）
  - 占位惯量（`ixx==iyy==izz==1.0` 且无惯性积）自动按 `mass/volume` 密度从 mesh 重算主惯量
  - 碰撞用真实凸包 mesh，视觉用包围盒（SAPIEN 3.0.0b1 的 `add_visual_from_file` 实测渲染不出内容），
    报告里以 `n_visual_fallback` 标注
  - 移除 z_offset，改为如实报告每物体的初始穿透深度
  - 相机用真实 VGGT extrinsics 轨迹（`cam_pos = -R_w2c.T @ t_w2c`，光轴 `R_w2c.T @ [0,0,1]`），
    均匀采样覆盖**全部**相机帧
  - 三重门控，任一失败即拒绝出图而非产出假结果：
    ① 无物体成功装入 → 退出码 2
    ② 真实相机埋在物体内部（到最近顶点距离 <2cm）→ `--strict_cam` 退出码 3
    ③ 渲染后彩色像素占比 <0.5%（全灰画面）→ 退出码 4
  - GIF 用 imageio 写（此前用 mp4v 编解码器写 `.gif`，容器与编码不匹配），同时输出 mp4 与首帧 png
  - `--report_only` 只做质量/惯量/穿透检查；`--no_vggt_cam` 退回包围盒环绕相机

###产物（6 个场景，均物理真实运行）
| 场景 | 物体 | 质量跨度 | 惯量占位 | 初始穿透>2cm | 位移物体 | GIF 帧 | 彩色占比中位 |
|---|---|---|---|---|---|---|---|
| 2chair-bottle_rescaled | 5 | 121× | 5/5 | 0 | 3 | 99 | 52.9% |
| 3chair-bottle_rescaled | 21 | 837× | 21/21 | 1 | 19 | 100 | 38.6% |
| phone_vggt_omega | 9 | 1849× | 9/9 | 0 | 7 | 100 | 54.2% |
| 082_C5_CellPhone_rescaled | 25 | 2123× | 25/25 | 0 | 25 | 100 | 10.9% |
| 121_C5_CellPhone_161deg | 12 | 3420× | 12/12 | 0 | 5 | 100 | 26.3% |
| 151_C7_Plate_214deg | 29 | 2218× | 29/29 | 13 | 28 | 10 | 42.0% |

物理正确性验证（原先 `stage61_validation.json` 报 91.8m 的假值）：
- 2chair：`bottle_0` z 从 +0.581 → +0.035（落到地面），其余椅子/桌仅微动 2~12mm
- 082：`bowl_0` z 从 +1.384 → +0.096，`bottle_*` 从约 +1.0~1.1 落到 +0.04~0.06
- 151：28/29 物体位移，仿真后 z 收敛到 0.011~0.274（原报告为 -1.296~+0.636）

### ⚠️ Docs to Review
- `docs/design/stage61_urdf_plan.md`: 9.5 节测试状态仍为"⏳ 待执行"，且未记录本次的
  SAPIEN 3.x API 结论（无 `set_mass_properties`、mesh visual 不可用）、真实相机轨迹用法、
  三重门控语义，以及 GIF 帧覆盖全部相机帧的约定。
- `docs/README.md`: 提及 stage61 但未说明新旧产物区别
  （`stage61_sapien.gif` 是旧的 box-visual 版本，新产物为 `stage61_physics.gif`）。

### 已知限制
- 部分场景存在少量空画面帧（如 phone 47/100），因真实手持相机确实近距离掠过物体，
  该帧视野内无物体。这不是渲染 bug，GIF 保留了真实相机的全部视角。
- 连续在同一进程跑多个场景会触发 EGL 资源耗尽（`code=11` 崩溃），需每场景独立进程，
  这是 SAPIEN OffscreenRenderer 的既有问题。

---

## [2026-10-08 15:30] 更正上一条: mesh visual 本来就可用, 视觉/碰撞形状现已一致

### 更正
上一条记录里写"SAPIEN 3.0.0b1 的 `add_visual_from_file` 渲染不出内容, 视觉退回包围盒"——
**这个结论是错的, 由错误的测试方法得出。** 我当时用一个自己造的简单 `box.obj`
配 `process=False` 加载, 相机距离 1.5m, 结果连 box 都渲染不出来, 于是归咎于 mesh visual。

用**同一相机、同一批物体, 只替换 visual 类型**做对照实验:

| visual 类型 | 画面唯一色数 | 彩色像素占比 |
|---|---|---|
| `add_visual_from_file` (真 mesh) | **386** | 12.5% |
| `add_box_visual` (包围盒代理) | 6 | 54.5% |

mesh visual 不仅可用, 而且画面信息远多于包围盒 —— 椅子的镂空靠背、瓶身轮廓、
剪刀形状都能看出来。之前"渲染不出"是我的测试用例有问题(自造 box.obj + 位姿/距离不当),
不是 SAPIEN 的限制。

### Changed
- `tools/stage61_physics_sim.py`:
  - visual 改为 `add_visual_from_file`, 与 collision 用**同一个 mesh 文件**,
    因此画面里看到的形状就是参与物理碰撞的形状, 上一条遗留的
    "visual 与 collision 形状不一致"问题就此解决。
  - `add_visual_from_file` 抛异常时才退回包围盒, 报告新增 `visual.n_visual_fallback`
    统计真实回退数(实测 0), 以及 `visual.same_mesh_as_collision=True` 标明语义。
  - 移除已证明不可靠的 `place_camera_behind` 自动取景(沿视线后退会把俯视相机推到地面下,
    `fwd_z≈-0.9` 时相机落到 `z=-2.99`), 保留真实 VGGT 相机轨迹。

### 验证
- `2chair-bottle_rescaled`: 5/5 物体真mesh 渲染, 椅子镂空结构清晰可见
- `082_C5_CellPhone_rescaled`: 25/25 物体真 mesh 渲染, 瓶身/剪刀/桌面轮廓可见
- 物理结果不变(与上一条一致): 2chair `bottle_0` z +0.581→+0.035;
  082 `bowl_0` z +1.384→+0.096

### 已知限制(修正上一条)
mesh visual 比包围盒耗 GPU, 连续在同一进程跑多个场景会触发 EGL 资源耗尽
(`elf_dynamic_array_reader.h: tag not found` + `code=11` 崩溃), 需每场景独立进程。
上一条误以为是"OffscreenRenderer 的既有问题", 实际是 visual 用 mesh 后资源占用更高。

---

## [2026-10-08 14:50] - 更正: 撤回"穿模全是误报"的错误结论 + 修复 sample_surface 关键字 bug

### 错误结论的来源

14:10 左右我曾报告"5 个场景 104 对 AABB 重叠, FCL 确认真穿模 0 对", 并据此
认为穿模修复是误报、建议不改。当时的扫描脚本对 `trimesh.collision.CollisionManager`
用了非标准调用, **未做交叉验证就下了结论**, 并把这个结论写进了 CHANGE_LOG
与对外分析。

### 更正后的实测 (用 in_collision_single 重新核查)

| 场景 | AABB 重叠 | FCL 真穿模 |
|---|---|---|
| egodex_4 | 23 对 | **4 对** |
| egodex_33 | 3 对 | **2 对** |
| 082_C5_CellPhone | 40 对 | **13 对** |
| 3chair-bottle | 14 对 | **7 对** |

最严重几对 (FCL 接触点数):
```
egodex_4: block_2↔block_5 9789 | bowl_0↔bowl_1 7722 | cloth_0↔cloth_1 7578
082_C5:   table_0↔table_1 11130 | bottle_6↔bottle_9 6395 | cup_0↔cup_1 3230
3chair:   cloth_0↔drawer_0 15108 | drawer_0↔tv_0 10997 | box_0↔cabinet_1 8391
```

**结论反转: 穿模修复不是误报, 它在修真实的穿模。** cloth_0↔cloth_1 被反复推开
28 次, 是因为两个 1.3~1.7m 的床单确实叠在一起。

### 基于错误前提做的改动已回退

我曾把 `_check_mesh_penetration_fcl` 改为 `mesh_b.contains()` 包含性判断 +
收紧阈值 (0.01→0.002, 3→5点)。**该改动基于"全是误报"的前提, 已全部回退**:
- 距离判据、阈值 0.01、阈值点数 3 全部恢复原值
- 回退后判据与 FCL 实测 **6/6 一致**

### 保留的真实 bug 修复

`refine_inter_object_placement.py` L1172:
```python
trimesh.sample.sample_surface(mesh_a, n_samples)   # ❌ trimesh 4.x 无此关键字
```
trimesh 4.11.5 签名为 `sample_surface(mesh, count, ...)`。原调用**必然抛
TypeError 被 except 吞掉**, 静默退化成"只用 mesh 顶点做最近距离判断",
判据实际生效的点集与设计意图不符。已改为位置参数调用, 实测采样 2000 点生效。

### 另一处修复: 平票时 max() 随意挑选

`infer_relations_scene_graph.vote_relations` 用 `max()` 决胜, 平票时按插入序
返回第一个。实测 egodex_33 `toy_0`: 票`{'supported by bed': 2, 'supported by lego_0': 2}`,
`bed` 是幻觉支撑物被 Step 2.5 剔除后, 剩余 `lego_0` 恰好也是 2 票 → max() 随意
挑了 `lego_0` → 5.2 按 on_top 把玩具贴到积木顶面 (实际在内部, gap -0.088m)。

**修法**: 平票且无明确多数时返回 None, 由调用方回退到 Stage1 原关系。
验证: `bed:2+lego_0:2 → None` (期望 None) / `lego_0:3+lego_1:2 → lego_0` /
`floor:1 → floor` / `空 → None`, 4 项全对。

### 遗留

egodex_4 跑到 127 次穿模修复后发散未收敛 (cloth↔cloth 28 次、bowl↔bowl 22 次),
已中止。这些是真实穿模, 但迭代不收敛 —— 根因是每次推开后新的一对又触发,
缺少全局收敛判据。**未解决**, 需专门处理。

---
## [2026-10-08 18:26] - 重建 tools/analysis 被误删的 4 个轨迹可视化脚本

背景: tools/analysis 下 4 个未入 git 的脚本被删除, 仅剩 `__pycache__` 里的
Python 3.11 字节码。产物 (`dynamic_glb/`、`trajectory_overlay/`) 完好。
本次从 `.pyc` 反汇编 (dis + 常量表) 逐字恢复源码, 并全量重跑验证。

### Added (均为从 .pyc 恢复, 非 git 跟踪文件)
- `tools/analysis/gltf_builder.py` (208 行): 最小 glTF 2.0 GLB 写入器
  (带 MAT4 transform 动画)。验证: 11/11 code object 常量/名字/签名与原
  字节码全同; trimesh 立方体冒烟测试导出+回读通过。
- `tools/analysis/export_dynamic_glb.py` (281 行): stage3 动态物体 6DoF
  轨迹 + URDF mesh → 带动画 GLB。验证: 10/10 全同; 实跑
  `stage3 'toy_inst0' -> 轨迹 'toy_inst1' (首帧质心差 3.7mm)`, 产物
  `toy_inst0.glb` md5 与原文件一致 (13b9d4f6…, 16.8MB), npz 数值 0 偏差。
- `tools/analysis/render_glb_anim_gif.py` (~390 行): 读 GLB 动画关键帧 →
  pyrender EGL 离屏渲染 GIF (物体沿轨迹移动 + 橙色轨迹折线 + 灰色背景
  点云)。验证: 12/12 全同; 实跑 60 帧 512×512 0.7MB, 与原 GIF 内容一致
  (仅 np.random.choice 无种子导致的背景点云子采样差异, 原代码固有)。
- `tools/analysis/trajectory_video_overlay.py` (~560 行): 物体轨迹投影回
  原视频 GIF + matplotlib 3D 轨迹图 + trajectory_report.json。验证:
  26/26 code object 全同; 实跑 `--obj toy_inst1 --n_frames 50` 与原 GIF
  **逐像素 0 差异**, 位移 0.342m / 路径 1.064m 与历史记录一致。

### Changed
- `output_v2/7_vggt_omega/trajectory_overlay/`: `--all_objects` 全量重跑
  11 个物体, 每个产出 `overlay/<key>.gif` + `traj3d/<key>.png` +
  `trajectory_report.json`。真正被抓取的 toy_inst1 净位移 0.342m
  (roll X 峰值 125°), 其余 10 条为静态/微动物体 (位移 0.002~0.032m)。

### 恢复过程中的坑 (转写字节码时)
- 原码大量使用全关键字传参: `Trimesh(vertices=, faces=, process=)`、
  `ColorVisuals(mesh=, vertex_colors=)`、`cylinder(radius=, height=,
  sections=)`、`DirectionalLight(color=, intensity=)` — 常量表中的
  kwnames 元组是唯一证据。
- `import imageio.v2 as imageio` 的字节码特征是 `IMPORT_NAME imageio.v2;
  IMPORT_FROM v2; STORE imageio`, 与 `from ... import` 同形, 靠
  IMPORT_FROM 的名字是末段模块名来区分。
- 转写笔误 3 处, 均由实跑暴露并修正: `cam[:, 2]` 误写 `[:, :, 2]`;
  `uv[0] if ok[0] else None` 误写 `uv[ok[0]]` (布尔标量索引会升维);
  3D scatter 需 `ax.scatter(*c[i], ...)` 星号解包。

### 备份
- 原始 .pyc 已备份: `/tmp/recover/{gltf_builder,export_dynamic_glb,
  render_glb_anim_gif,trajectory_video_overlay}.ORIGINAL.pyc`
  (前两个在子任务备份, 后两个在覆盖前手工备份)。

### 遗留
- 4 个脚本仍是 untracked, **建议 `git add tools/analysis/*.py` 入库**,
  避免再次丢失。
- `tools/analysis/README.md` 未收录这 4 个脚本 (此前也未收录, 非本次引入)。

---
## [2026-10-08 18:40] - 轨迹产物收敛到被抓取物体 (stage pick/place 判定)

用户指出: 只有被抓取的物体才需要物理轨迹, stage 的 pick/place 判定
(`pick_place_frames_combined.json`) 已给出答案 — `toy_inst1`
(pick 46 -> place 97, confidence 1.0), 无需输出全部 11 条。

### Removed
- `output_v2/7_vggt_omega/trajectory_overlay/{overlay,traj3d}/`: 删除
  block_inst0/1/2、plate_inst0、star_inst0、toy_inst0/2/3/4/5 共 10 个
  静态/微动物体的 GIF 和 PNG (位移仅 0.002~0.032m, 非 pick 动作)。

### Changed
- `trajectory_overlay/trajectory_report.json`: 用
  `--obj toy_inst1 --n_frames 50` 重跑, 现只含 toy_inst1 一条,
  与 pick_place_frames_combined.json 的判定一致。

---
## [2026-10-08 23:00] - 方案 B: 绕开 SAPIEN 渲染失效, 8 个场景统一出 GIF

### 背景
`7_vggt_omega` / `phone_vggt_omega` / `egodex_0_rescaled` 三个场景 SAPIEN 渲染
与实体位姿同步脱钩 (物理真实位移、组件齐全、相机位姿读回正确、mesh 有效,
但画面恒为纯灰, 连 probe 方块也不可见)。根因在渲染引擎内部, 继续挖成本过高,
改为**物理与渲染解耦**: SAPIEN 只跑物理并逐帧记录位姿, pyrender 离线重建渲染。

### Added
- `tools/stage61_physics_sim.py::main_pyrender`: 方案 B 主流程。纯物理场景
  (`Scene(systems=[PhysxCpuSystem()])`, **完全不创建 SapienRenderer 与 visual
  组件**) → 逐帧记录全部 actor 位姿 → pyrender 渲染 → 输出三件套 + 报告。
- `tools/stage61_physics_sim.py::pyrender_render_frames`: pyrender 场景重建。
  - 相机矩阵 `_obj_c2w_gl`: 由 VGGT `w2c` 直接构造 (c2w = R.T/-R.T@t, 再乘
    `diag(1,-1,-1,1)` 做 OpenCV→OpenGL 相机系 Y/Z 翻转); 无 extrinsics 时退回
    `_look_at_c2w_gl` (Z-up 世界)。
  - `make_intrinsic_camera`: 用**真实 OpenCV 内参** (fx/fy/cx/cy) 构造离轴
    OpenGL 投影矩阵, 不再用硬编码 fovy。
  - 真实图像尺寸由 `color/0.*` 探测 (`detect_image_size`), 与真实视频同分辨率
    (688x384 或 384x688); 地面用扁平 box (该 trimesh 版本无 `creation.plane`)。
- 新 CLI: `--pyrender_render`、`--render_mesh {visual,collision}` (默认 visual)。

### Changed
- `tools/stage61_physics_sim.py`:
  - `parse_urdf` 额外返回 `visual_mesh` (渲染用真实外观 mesh)。
  - `build_actors(renderer=None)`: 支持只要物理、不建任何 visual。
  - `load_camera_trajectory` 保留原始 `w2c` 矩阵 (原先只留 pos/fwd, 丢失相机 roll)。
  - **修复 MP4 红蓝互换**: `frames` 是 RGB 而 `VideoWriter` 期望 BGR, 两条路径
    都补 `cv2.cvtColor(f, COLOR_RGB2BGR)`。
  - 修正 `setup_scene` 过时注释: 实测 `sapien.Scene()` 默认后端就是
    `PhysxCpuSystem`, 并非 "默认 GPU PhysX"。
  - `render_check` 增加 `empty_frame_indices`, 并把 "空帧" 归因写清楚。
- `tools/stage61_batch_run.sh`: 支持 `PYRENDER=1` (方案 B) / `PYRENDER=visual|collision`;
  单场景超时 1200s → 1800s。

### 修复的两个真实 bug (方案 B 实现过程中发现)
1. **投影矩阵符号**: `P[1,1]` 写成 `-2fy/h` 会让画面绕主点上下翻转 (地面跑到
   画面上方、背景跑到下方)。正确为 `+2fy/h` (与 pyrender 自带 `1/tan(fov/2)` 同号)。
2. **`Scene.add_node` 不返回 node** (只有 `Scene.add` 返回), 导致 `set_pose` 报
   `Node must already be in scene`。改为自持 `Node` 引用后 `add_node`。

### 验证
- 单场景: `7_vggt_omega` 渲染首帧与真实 `color/0.jpg` 视角/物体布局一致
  (绿积木左、粉/黄玩具中、浅蓝盘子右), 修正符号前后对比确认翻转已消除。
- 对照实验: 同一场景 `2chair-bottle_rescaled` 分别跑 SAPIEN 路径与 pyrender 路径。
  SAPIEN 路径仍正常 (无回归, 彩色中位 12.0%), pyrender 路径 83.0% 且分辨率/内参
  与真实视频一致 (SAPIEN 路径固定 640x480 + fovy 50° 与真实内参无关)。
- 8 个场景统一重跑 (`PYRENDER=1 bash tools/stage61_batch_run.sh 120`), 全部出 GIF:

| 场景 | 彩色像素中位 | 空帧 | 物体位移 | 相机帧 |
|---|---|---|---|---|
| 2chair-bottle_rescaled | 83.0% | 0/120 | 3/5 | 120 |
| 3chair-bottle_rescaled | 30.3% | 0/120 | 19/21 | 120 |
| phone_vggt_omega | 58.5% | 37/120 | 7/9 | 100 |
| 082_C5_CellPhone_rescaled | 47.3% | 0/120 | 25/25 | 120 |
| 121_C5_CellPhone_161deg | 66.9% | 0/120 | 4/12 | 100 |
| 151_C7_Plate_214deg | 12.4% | 0/120 | 29/29 | 160 |
| 7_vggt_omega | 3.2% | 0/120 | 3/8 | 113 |
| egodex_0_rescaled | 24.3% | 0/120 | 7/8 | 120 |

- `phone_vggt_omega` 的 37 个空帧 (索引 83~119) **不是渲染失败**: 该段真实视频是
  手持手机特写 (相机距被摄物极近), 而 sim 里手机已落地离场, 故画面只剩地面。
  已写入报告的 `empty_frame_indices` 以便追溯。

### ⚠️ Docs to Review
- `docs/design/stage61_urdf_plan.md`: 只覆盖 URDF 生成, 未涉及物理/渲染脚本,
  本次无需改; 若后续要沉淀 "仿真+渲染" 设计, 建议单独开一篇。
- `docs/questions.md`: 已追加 Q29 (渲染失效 + 绕过方案)。
- 已知遗留 (上游数据问题, 非本次引入): `151` 场景物体在画面里偏小且靠边、
  `082` 相机贴得很近, 属 stage2 对齐/stage4 重建精度问题, pyrender 路径只是
  把真实相机位姿如实呈现出来。

---
## [2026-10-08 23:45] - 接通 Stage6 -> Stage6.1 材质链路 + 结果画廊 HTML

### 背景
Stage 6 (`tools/material_annotator.py`) 标注的 `materials.json` **没有任何消费者**:
Stage 6.1 的 `stage61_physics_sim.py` 从不读它, URDF 里的摩擦是 EmbodiedGen 硬编码
占位 `<mu1>0.60</mu1><mu2>0.50</mu2>`、恢复系数固定 0.02, 质量是 VLM 猜的值 ——
Stage 6 等于白跑。

### 根因最后一公里
Stage 6 的输出 key 是 `bowl_inst0`, Stage 6.1 的 URDF link 名是 `bowl_0` ——
**两边命名约定不一致, 而全仓库没有任何地方做过 `inst<N>` -> `_N` 的映射**。

### Added
- `tools/analysis/make_stage61_gallery.py`: 扫描 `output_v2/` 下所有
  `stage61_physics.gif`, 生成单文件 HTML 画廊 (沿用已丢失的
  `output_v2/stage61_gif_gallery.html` 的暗色卡片形式)。每张卡显示 GIF +
  后端/物体数/位移数/彩色占比/空帧/相机帧+分辨率/是否有 Stage6 材质, 并提供
  GIF/MP4/首帧/报告/URDF/材质链接。默认输出到 `output_v2/stage61_gif_gallery.html`。
- `tools/stage61_physics_sim.py`:
  - `load_materials()` / `resolve_material()`: 读 materials.json 并做
    `bowl_0 -> bowl_inst0` 映射。命中策略: 精确 > 该类别材质完全一致时按类别取值
    > 放弃 (宁可不用也不张冠李戴)。
  - `attach_materials()`: 覆盖每个物体的静/动摩擦与恢复系数, 并写 `mass_source`、
    `mu1/mu2/restitution_effective` 等到报告。
  - `mesh_volume()`: 给密度法用的体积 + 可靠性判断。
  - `--no_materials` / `--mass_source {vlm,density}` 两个 CLI。

### Changed
- `tools/stage61_physics_sim.py`: `build_actors` 的 `PhysxMaterial` 不再写死
  restitution=0.02, 改为读 `obj["material"]`。
- 报告新增 `materials_stage6` 段 (覆盖率、逐物体材质/质量来源/有效摩擦)
  与 `render_check` 同级。

### ⚠️ 关键设计决定: 质量默认不用密度法
原打算把质量也换成 `密度 × mesh 体积` (物理上更正确), 实测后发现**不能这么做**:
- `newspaper_1` 算出 10.9kg、`table_0` 35kg `stove_0` 24.3kg, 明显不合理;
- 密度质量/VLM 质量的中位数是 **5.03x**, 换算成线性尺度误差 **≈1.71x** ——
  与 `docs/SAM3D_SCALE_DRIFT_REPORT.md` 记载的 "SAM3D ~1.5x 绝对尺度漂移"
  量级吻合, 说明是 mesh 整体偏大, 不是密度不该用。
所以默认 `vlm` (保留 URDF 的 VLM 质量), `density` 只是可选项; **摩擦/恢复系数则
无条件采用 Stage 6** (它们不依赖绝对尺度)。
顺带, 这个比值被做成**尺度漂移诊断量**写进报告: `implied_linear_scale_error`。

### 验证
- `151_C7_Plate_214deg_vggt_omega` (8 个目标场景里唯一有 materials.json 的):
  材质命中 **29/29, 全部精确匹配**; 其中 23 个体积可信、6 个体积不可信(confetti)。
- 受控 A/B (同场景、同 seeds、仅开关 `--no_materials`): 初始 z 完全相同,
  最终位姿合计 |Δ| = 0.2545 m, 最大单物体 Δ = 5.9cm (`bottle_1`,
  玻璃 0.4/0.4 vs 硬编码 0.6/0.5 -> 滑得更远), 位移物体数 29 -> 28。
  **证明 Stage 6 的材质确实在改变物理行为, 不再是摆设。**
- 画廊: 8 张卡, 49 个链接全部有效 (0 失效)。
- 其余 7 个场景没有 materials.json (Stage 6 当时没跑), 运行会打印明确 WARN 并
  退回旧行为 —— 补跑 `mainv3 --enable_stage6` 后无需改代码即可自动生效。

### ⚠️ Docs to Review
- `docs/questions.md`: Q30 已补充 "断链问题已解决" 的处理方式。
- `docs/SAM3D_SCALE_DRIFT_REPORT.md`: 本次的 `implied_linear_scale_error ≈ 1.71x`
  是该报告的一个独立实测佐证, 可交叉引用。

---
## [2026-10-09 00:50] - Stage 6 定位收敛为"物理仿真" + 10 数据集全量验证

### 定位调整 (用户明确要求)
`stage5 修复后, stage6 主要还是物理仿真, 不要有太多的功能`。
据此把 Stage 6 的官方口径 **收敛为一件事: URDF → SAPIEN 物理仿真 → 渲染出 GIF**,
把 `tools/material_annotator.py` (原 mainv3 抬头里的 "Stage 6: 材质物理标注")
**降级为 Stage 6 的输入提供方之一**, 而不是 Stage 6 本体。范围边界在文档里写死。

### Added
- `docs/design/stage6_physics_sim.md`: Stage 6 正式设计文档 (10 个数据集验证表)。
  含: 做什么/不做什么的边界表、流程图、输入输出清单、脚本清单、材质接口、
  运行命令、验证结果、已知限制。已在 `docs/README.md` 索引登记。
- `tools/run_stage6_only.py`: 给已完成场景**单独补跑** Stage 6 材质标注的最小脚本。
  只读 `all_instances.pkl` + `optimal_frames/`, 调 `material_annotator.run_stage6()`,
  不做任何 Stage 1~5 的事。支持 `--scene` / `--missing`。

### Changed
- 8 个目标场景 + 2 个补充场景补跑 Stage 6 材质; 全部 10 个场景 120 帧重跑物理仿真。
- `output_v2/stage61_gif_gallery.html`: 画廊更新为 10 个场景。

### ⚠️ 重要踩坑: Stage 6 材质标注必须显式指定空闲 GPU
默认 `device_map="auto"` 会挑到被其他进程占满的 GPU (实测挑中 GPU2, 已用 44GB),
表现为 **CUDA OOM + 权重被 offload 到 CPU**。必须 `CUDA_VISIBLE_DEVICES=<空闲卡>`。
本次: GPU0/1 空闲 ~39/47GB, GPU2/3 被占 ~44GB。

### 验证 (10 个数据集, 全部 --pyrender_render, 120 帧)
| 场景 | 物体 | 位移 | 彩色% | 空帧 | 相机帧 | Stage6命中 | 尺度误差 |
|---|---|---|---|---|---|---|---|
| 2chair-bottle_rescaled | 5 | 2 | 82.8% | 0/120 | 120 | 5/5 | 1.229x |
| 3chair-bottle_rescaled | 21 | 19 | 31.1% | 0/120 | 120 | 21/21 | 1.314x |
| phone_vggt_omega | 9 | 7 | 58.5% | 36/120 | 100 | 9/9 | 1.259x |
| 082_C5_CellPhone_rescaled | 25 | 25 | 47.8% | 0/120 | 120 | 25/25 | 1.283x |
| 121_C5_CellPhone_161deg | 12 | 5 | 66.5% | 0/120 | 100 | 12/12 | 1.204x |
| 151_C7_Plate_214deg | 29 | 28 | 12.4% | 0/120 | 160 | 29/29 | 1.713x |
| 7_vggt_omega | 8 | 3 | 3.2% | 0/120 | 113 | 8/8 | 1.402x |
| 7a_vggt_omega | 8 | 4 | 1.4% | 0/120 | 113 | 8/8 | 1.361x |
| egodex_0_rescaled | 8 | 6 | 25.2% | 0/120 | 120 | 8/8 | 1.654x |
| egodex_34_rescaled | 10 | 8 | 21.8% | 0/120 | 120 | 10/10 | 1.412x |

- 合计 **135 个物体 / 107 个发生真实位移**; 渲染自检全部通过 (>0.5%)。
- **Stage 6 材质在 10 个场景上 100% 精确命中**, 涉及 14 种材质。
- 隐含线性尺度误差 **1.20 ~ 1.71x (中位 1.36x)**, 跨 10 场景一致,
  再次佐证 SAM3D 的 ~1.5x 绝对尺度漂移。

### ⚠️ Docs to Review
- `docs/design/stage6_physics_sim.md` 与 `docs/design/stage6_design.md` 口径有重叠:
  后者仍把 material_annotator 称作 "Stage 6", 属新的定位调整后的旧口径,
  后续如需统一, 应改 `stage6_design.md` 的抬头或合并成 "Stage 6 输入准备" 小节。

---
## [2026-10-09 01:20] - Stage 6 三份文档合并为一份权威文档

### 背景
上一轮新增 `stage6_physics_sim.md` 后, 仓库里同时存在三份 "Stage 6" 文档且口径冲突:
- `stage6_design.md`: 把**材质标注**当作 Stage 6 全部职责 (2026-07 版)
- `stage61_urdf_plan.md`: 把 **URDF 生成**单独叫 "Stage 6.1" (2026-07-30 计划版)
- `stage6_physics_sim.md`: 新口径 —— Stage 6 = **物理仿真**

### Changed (合并)
- `docs/design/stage6_physics_sim.md` 扩为唯一权威文档, 新增两个附录:
  - **附录 A: Stage 6 材质标注（输入准备）** —— 吸收 `stage6_design.md`:
    材质驱动 vs 参数驱动对比、流程、输出格式(实例级 key)、两种"静默退化"陷阱
    (裁剪图 key 层级 / 缺空闲 GPU)。`materials_db.json` 的完整参数表**不再在文档里
    复制一份**, 只保留文件指针, 避免两处不同步。
  - **附录 B: URDF 生成（①）要点** —— 吸收 `stage61_urdf_plan.md`:
    五步流程、调用 EmbodiedGen 的模块、5 条踩坑 (nvdiffrast mock / real_height /
    mesh 双前缀 / gazebo 标签 / ixx=1.0 占位惯量)。
- `docs/design/stage6_design.md`: 750 行 → **33 行存根**, 含"内容去向"对照表 +
  2 条"原设计与实际实现不一致"的记录 (VLM 调用方式、输出 key 粒度)。
- `docs/design/stage61_urdf_plan.md`: 750 行 → **32 行存根**, 含"内容去向"对照表 +
  2 条需留档的历史事实 (`--enable_stage61` 从未实现; 文档里的解释器已过时)。
- `docs/README.md`: 索引改为把 `stage6_physics_sim.md` 标为权威文档, 另两份标注"已合并"。
- `docs/questions.md` Q30: 修正两处因合并而失效的行号引用 (`:131` / `:736`)。
- `README.md`: 顺手修掉 2 个**早已失效**的文档链接
  (`docs/mainv2_technical_doc.md` → `docs/pipeline/mainv2_technical_doc.md`;
  `docs/coordinate_and_alignment.md` → `docs/design/coordinate_and_alignment.md`)。

### 重新生成
- `output_v2/stage61_gif_gallery.html`: 10 卡片 / 70 链接 / **0 失效**, 15KB。
- `tools/analysis/make_stage61_gallery.py`: 缩略图样式 `object-fit:cover` →
  **`contain` + 固定高度 230px**。场景有横版(688x384)和竖版(384x688)两种,
  `cover` 会把竖版裁掉大半 (只剩中间一小块), 实测截图确认后修正。

### 验证
- 合并后全仓库文档内链扫描: 本次合并**未引入任何新的失效链接**。
  顺带把 `README.md` 那 2 个修掉后, 全仓库仅剩 **2 处**历史遗留失效链接
  (`docs/design/coordinate_and_alignment.md` 里指向一个不存在的
  `./GLB_to_HaWoR_Alignment_Plan.md`, 以及 `README.md` 曾指向它) —— 与本次无关。

### ⚠️ Docs to Review (已处理完)
- 三份 Stage 6 文档口径冲突 —— 本次已合并解决, 无遗留。

---
## [2026-10-09 11:25] 打包发布 Snapshot v0.2-2026-10-09 到 GitHub

把 Snapshot v0.1 (2026-09-24) 以来累积的全部变更打成快照入库并推送。

- 提交: `ebaff28` — "Snapshot v0.2: Stage 6 收敛为物理仿真 + 接通材质链路 +
  pyrender 方案B"; `git push origin main` 快进 `d6dd04d..ebaff28`, **未用 force**。
- 规模: **34 个文件 (17 修改 / 17 新增 / 0 删除)**, 9906 insertions(+), 1592(-)。
- 本次快照重点 (Stage 6): 定位收敛为"URDF → SAPIEN 物理仿真 → 渲染 GIF";
  接通 Stage6(materials.json) → Stage6.1 断链; SAPIEN 渲染失效走 pyrender 方案 B。
  其它累积项 (Stage5 精修、尺寸校准 guard、坐标系对齐 B~E、tools/analysis 重建等)
  逐条见上方各条目。
- **零删除**: 提交前已核验 `git status` 无任何 D/R 记录。被合并成存根的两份旧文档
  (`stage6_design.md` / `stage61_urdf_plan.md`), 其原文全文**原样归档**到
  `docs/design/archive/*_v1.md` (678/749 行完整保留), 并在存根与 `docs/README.md`
  里互相指路 —— 满足"不覆盖原文件"的要求。
- 遗留: `output_v2/` 在 `.gitignore` 内 (合理的产物目录), 所以
  `output_v2/stage61_gif_gallery.html` 与各场景 GIF **不入库**;
  入库的是它的生成脚本 `tools/analysis/make_stage61_gallery.py`, 随时可重新生成。

---

## [2026-10-09 12:30] - 用 FoundationPose 替换 ICP 管线, 修复 2D 叠加位姿漂移

### 背景
`trajectory_video_overlay.py` 产出的 2D 叠加 GIF 位姿不准。量化诊断确认:
逐帧"投影mesh vs SAM3 mask"的 IoU 在静止段 0.80~0.86, 但**举升段(帧60~80)
掉到 0.35~0.38** —— 链式 ICP 在遮挡帧累积误差导致漂移。用户判断正确。
本轮改用 FoundationPose(NVlabs, CVPR 2024)做 model-based 6DoF 跟踪。

### Added
- `tools/analysis/fp_prepare_data.py`: 为 FoundationPose 准备输入。
  从 VGGT world_points 投影生成逐帧深度(米→16bit mm png), 逐帧反推内参
  (`recover_intrinsic`, 与 vggt-omega 反投影同口径), 复制 rgb/mask, 拷 URDF mesh。
  输出 `foundationpose_data/`(113 帧)。
- `tools/analysis/fp_run_tracking.py`: FoundationPose 跟踪 runner。
  帧0 `register`(render-and-compare 自主搜初始位姿) → 后续 `track_one`。
  输出 `foundationpose_out/{ob_in_cam,ob_in_world}/`。注意 rgb 必须
  `np.ascontiguousarray(...[...,::-1])`, 否则 torch 拒绝负 stride 数组。
- `tools/analysis/fp_iou_eval.py`: **面片光栅化** IoU 评估。
  顶点投影+3×3膨胀会系统性高估投影面积, 把 IoU 压低 0.05~0.10。
- `tools/analysis/fp_overlay_gif.py`: 2D 叠加 GIF(面片级轮廓 + 历史尾迹)。
- `tools/analysis/fp_trajectory_plot.py`: 世界系 3D 轨迹三联图。
- `tools/analysis/fp_export_glb.py`: 用 FP 位姿导出内嵌动画 GLB
  (复用已验证的 `gltf_builder.export_glb_with_animation`)。
- `tools/analysis/fp_diagnose.py`: 位姿偏差来源诊断(自跟踪 vs ICP 基线对照)。

### Changed
- `tools/analysis/fp_run_tracking.py`: 撤销 `symmetry_tfs` 开关 —— 实测传入
  Z轴180° 对称变换并未减少等价解翻转(跳变 9 次→11 次), 且已验证逐帧真实
  旋转仅 4.8°/帧, 数据本身健康。保留注释说明为何不需要。

### 实现要点(踩坑记录)
1. **pytorch3d 预编译 wheel 403**(dl.fbaipublicfiles.com), 只能源码编译。
   nvdiffrast/pytorch3d 均需 `CUDA_HOME=$CONDA_PREFIX` 且
   `CPATH` 包含 `$P/targets/x86_64-linux/include` + torch 自带 `nvidia/*/include`。
   实际推理路径只用到 `so3_exp_map`/`rotation_6d_to_matrix` 等纯数学函数,
   renderer 部分无人调用。
2. **GitHub codeload 不支持断点续传**(卡在 6.5MB/37MB), 改用
   `ghfast.top` 镜像(8MB/s)。
3. **ICP 的 `centroids` 与 mesh 顶点原点不一致** —— 把它当物体原点投影会
   整体平移错, 投影直接落在画面外(IoU=0)。实测用真实深度反算物体中心
   应在相机系 (0.100, 0.222, 0.647), 而 ICP centroid 转相机系是
   (-0.622, -0.731, 0.880), 完全错误; FP 输出 (0.105, 0.220, 0.674)
   毫米级吻合。**旧"ICP 静止段 0.80"基线的评估口径本身有问题。**
4. **姿态曲线不能用欧拉角/旋转向量**: 物体总转角 167°, rotvec 在 ±π 边界
   反复跳变(看似 6.2rad), 但逐帧真实旋转仅 mean 4.8°/max 31.8°, 无一帧
   超 90° —— 纯属表示退化。改画"相对首帧"的旋转矩阵元素 R00/R11/R22。

### 验证 (scene 7_vggt_omega, toy_inst1)
- 位姿精度: 深度反算中心 vs FP 输出吻合毫米级(见要点3)。
- 投影: FP mesh 投影 bbox x[369,407] 与 SAM3 mask bbox x[369,406] 几乎重合。
- IoU(面片光栅): 全帧 mean=0.653/min=0.474;
  **静止段 IoM(真实mask被覆盖)=0.968**, 举升段 0.750(手部遮挡使SAM3
  只标可见部分, 属mask 局限)。
- 交叉验证: 世界系净位移 **0.3471m** vs stage3 pick/place 判定 0.3431m。
- `py_compile` 全部通过。

### 产物
- `foundationpose_out/overlay/toy_inst1.gif` — 2D 叠加(相机系)
- `fp_glb/gif/toy_inst1.gif` — **3D 动画(世界系 Z-up)**
- `fp_trajectory/toy_inst1_trajectory.png` — 3D 轨迹图(世界系)
- `fp_glb/toy_inst1.glb` — 内嵌动画 GLB, 可直接 Blender/MeshLab 播放

### Changed (git)
- `mainv3.py`、`src/geometry_utils.py` — 非本轮改动, 之前遗留的未提交变更。

###遗留 / 待办
- 旧 `trajectory_overlay/`(ICP 版产物)仍在目录里, 其位姿已知不准,
  建议删除或标注废弃; 本轮未擅自删除既有产物。
- FoundationPose 依赖 conda env `/mnt/data/lza/conda_envs/foundationpose`,
  不随仓库入库, 换机器需按要点1 重建。

---
## [2026-10-13 01:20] - 2D+3D 合一 GIF + 方案B(静止段锁位姿)离线验证

### Added
- tools/analysis/fp_gif_2d3d.py: 生成 2D+3D 合一 combo.gif。左半=视频叠加(绿=FP投影/红=GT mask/橙尾迹), 右半=世界系3D(mesh点云按ob_in_world变换+相机轨迹+FP轨迹橙+GT轨迹绿, 固定视角)。输出 fp_vis/<obj>/combo.gif 并自动同步 test/fp_vis/<obj>/。8/8 实例已生成(~8.5MB, 57帧)
- tools/analysis/fp_static_lock_eval.py: 方案B离线验证——用GT mask反投影世界系轨迹做滑动窗口运动判定(win=9,thr=5cm最优), 静止段锚定段首位姿(anchor=first)。对比锁定前后 path/IoU 与GT真值
- output_v2/7_vggt_omega/fp_static_lock_eval.json: 8实例验证数据

### Changed
- (无既有文件修改, 均为新增工具)

### 验证结论 (方案B定型)
- 静止段锁定(段首锚定)有效: plate_inst0 path 1.920m→0.000m, IoU 0.887→0.848(GT净位移0.006m真静止); toy_inst0 IoU 0.653→0.659 且 path 5.2→3.0m
- 弃用 mask 质心逐帧校正: mesh质心投影≠mask质心, plate IoU 0.887→0.145
- 弃用段内中值锚定: FP track平移是长随机游走, 中值离真值16cm; 必须锚定段首(register位姿, IoU 0.40~0.97全准)
- 运动段FP抖动仍在(剩余问题), 集成mainv3时复用stage3动态判定分段

---
## [2026-10-12 22:36] - FP 运动段抖动修复 (savgol 平滑) + 最终结果

### Added
- output_v2/7_vggt_omega/fp_out_<obj>/ob_in_world_locked/: 8实例修复后位姿 (113帧/实例, 静止段锁定+运动段savgol w21), 原始 ob_in_world 未动

### Changed
- tools/analysis/fp_static_lock_eval.py: 新增 smooth_moving() (运动段平移逐轴savgol+旋转四元数savgol, 段内离线零滞后) 及 --smooth_win/--smooth_order/--save_locked 参数; 定型参数 --win 9 --thr 0.05 --no_correct --anchor first --smooth_win 21
- tools/analysis/fp_gif_2d3d.py: build_frames() 新增 poses_dir 参数, main() 新增 --poses 选项, 可用修复后位姿重生成 combo.gif
- output_v2/7_vggt_omega/fp_vis/<obj>/combo.gif + test/fp_vis/<obj>/combo.gif: 8实例全部用修复后位姿重新生成 (8.1~8.7MB, 57帧)
- output_v2/7_vggt_omega/fp_static_lock_eval.json: 更新为「锁定+运动段平滑」最终数据
- docs/questions.md: 追加 Q21 (运动段抖动解决方法+最终量化对比表)

### 验证结论 (最终)
- 高IoU实例抖动压掉55%~100%且IoU持平: plate 1.92→0.00m (0.887→0.848), toy_inst0 5.21→1.54m (0.653→0.650), toy_inst1 7.11→2.20m (0.318→0.317)
- 真实pick-place运动保留: 净位移GT 0.347m, 修复后 0.764/0.922m (由段锚点决定, 与平滑窗口无关)
- 低IoU实例 (block_inst2 IoU 0.467→0.372) 平滑无法补救原始位姿误差, 属FP track本身问题
- 窗口扫描: w21最优平衡 (w31 toy_inst1回升; 中值9+savgol仅再降0.1m不引入)

---
## [2026-10-12 22:55] - FP 只跟踪动态实例 + 修复 3D 面板物体/轨迹分离

### Changed
- tools/analysis/fp_gif_2d3d.py: 新增 dynamic_keys() 按 stage3 判定 (pick_place_frames_combined.json) 过滤动态实例 (默认启用, --all 取消); 3D 面板 FP 轨迹由平移分量 t 改画 mesh 质心轨迹 R@c+t (toy_inst1 mesh 原点离几何中心 0.622m, 导致点云与轨迹分离)
- tools/analysis/fp_static_lock_eval.py: 同样接入 dynamic_keys() 过滤 + --objs/--all 参数
- test/fp_vis/: 删除 7 个静态实例目录, 只保留 toy_inst1 (stage3 判定唯一动态实例)
- output_v2/7_vggt_omega/fp_vis/toy_inst1/combo.gif + test/fp_vis/toy_inst1/combo.gif: 质心轨迹修复版重新生成
- output_v2/7_vggt_omega/fp_static_lock_eval.json: 只含 toy_inst1 (7.105/0.318 → 2.204/0.317)
- docs/questions.md: 追加 Q22

---

## [2026-10-13] - 调研 LiteReality-Agent 替代 stage5 精修的可行性

### Added
- docs/questions.md: 追加 Q32 —— LiteReality-Agent 调研结论（不能整体替代：输入契约 iOS RGB-D+usdz vs ego video→VGGT 实例 mesh、物体来源检索 CAD vs 真实 mesh+FP 轨迹；但其 layout/QC 精修层与 stage5 逐环节对位：check_support 确定性支撑判定（0.15m cap 同值）、collision resolve（detect→nudge→detect+metric 修复）、MuJoCo shake test 对应 check_stability）

### Changed
- 无代码修改（纯调研任务）

### ⚠️ Docs to Review
- docs/research/ 可考虑新增 LiteReality-Agent 分析文档，记录 room_qc 门控设计（doc/QC/collision_check.md、doc/QC/supporting_relationship.md）与 stage5 的映射关系，供后续方案 A 移植时参考

---

## [2026-10-13] - 追加 Q33：方案 A 详细说明

### Added
- docs/questions.md: Q33 —— 方案 A 四步拆解（编译带 extras GLB → check_support/check_glb 两道确定性门控 → opencode agent MCP 兜底循环 → 输出 final GLB），5.1 VLM 关系保留作先验，refine_inter_object_placement.py 保留作回退

---

## [2026-10-13] - 追加 Q34：check_support/check_glb 是确定性函数而非 agent 操作

### Added
- docs/questions.md: Q34 —— 澄清确定性优先/agent兜底分工：两 check 为纯函数可命令行直跑，agent 仅处理 unsupported（补建模）与 capped（重理解位置），support_cycle 报人

---

## [2026-10-13 22:20] - stage5.5 端到端验证完成 (121 场景)

### Added
- tools/stage55_cli.py: stage5.5 确定性 QC + 位姿调整 CLI（inspect/check_support/snap_support/check_glb/nudge/set_xy/set_z/render），GLB 加载用 X180@Y_UP_TO_Z_UP 转常规 +z 朝上，与保存侧精确互逆；修改类命令追加 stage55_moves.jsonl
- tools/run_stage55.py: 编排入口——复制 stage5 GLB 为工作副本 → 确定性预 snap → opencode agent 语义 set_xy 循环 → nudge 兜底 → 大 cap 收尾 snap → final_scene_stage55.glb + stage55_report.json + 四视图预览
- 产物: output_v2/121_*/final_scene_stage55.glb、stage55_report.json、final_scene_stage55_preview.png

### Changed
- run_stage55.py 提示词: 禁止 agent 调用 nudge/snap/set_z 强制语义 set_xy；新增"large_gap 物体是首要任务"与"必须执行修改才能结束"硬约束
- run_stage55.py 收尾 snap_support cap 0.15→0.60: agent 完成语义 XY 后，剩余垂直 gap 属合法落座行程

### 验证结果 (121_C5_CellPhone_161deg_vggt_omega)
- 碰撞对 1→0（bowl_1↔scissors_0 4.1cm，MTV nudge 3 轮收敛）；支撑问题 2→0（mug_1/scissors_0 悬空 0.446m 落座矮桌 table_1）
- 语义正确性: 矮桌物体组合（mug+scissors+bowl+bottle）与实拍参考帧 mug_inst1_frame86.jpg 一致
- agent 评估: sensenova-6.8-flash-lite 两轮均未执行任何 set_xy（读帧阶段输出截断）；确定性层独立达成正确结果，验证了"确定性优先、agent 兜底"设计

### ⚠️ Docs to Review
- docs/questions.md: Q32-Q35 已随任务同步更新，无需额外操作
- 后续若接入更强免费模型（如 glm-5.2）激活 agent 语义修正，需补测 stage55_agent_run.log 决策质量

---

## [2026-10-14 12:00] - FoundationPose mask 引导 re-register + 跨场景测试

### Added
- `tools/analysis/fp_run_tracking.py`: `--rekey N` 参数——每隔 N 帧用当帧 SAM3 mask+depth 重新 register 抗漂移（register 失败回退 track，帧 0 仍强制 register），修复旧管线 mask 仅帧 0 使用导致遮挡后锁到相似物体的问题
- 4 新场景 bottle_inst0 全链产物: output_v2/{116,121,2chair-bottle,3chair-bottle}/fp_data_|fp_out_|fp_vis/bottle_inst0

### Changed
- `tools/analysis/fp_gif_2d3d.py`: contour() 支持半透明填充（GT mask 红 fill=0.45+3px、FP 绿 fill=0.18，小物体细轮廓难辨认）；--objs 显式给定时跳过 fp_batch_report.json；test 同步目录改为 <场景名>_<实例名> 避免同名覆盖
- toy_inst1 正式结果切换为 rekey 版（IoU 0.318→0.523），旧管线结果备份 fp_out_toy_inst1_orig/

### Removed
- test/fp_vis/bottle_inst0/（跨场景同名覆盖产生的冗余副本，与 3chair 版 md5 相同）
- fp_out_toy_inst1/ob_in_world_locked/（rekey 后 savgol 跨等价位姿跳变平均导致 IoU 降至 0.361，有害无益）

### 验证
- 5 场景 IoU/path 对比：116 0.72/0.74m@0.555、2chair 1.84/1.97m@0.688、3chair 4.23/4.18m@0.603、7 1.71/1.12m@0.523 均 path 吻合；121 部分失败（3.84/0.40m@0.432）待查
- 全部 combo.gif 重生成并同步 test/fp_vis/<场景>_<实例>/

---
## [2026-10-14 15:00] - SpatialTrackerV2 替代 FoundationPose 跟踪实现与 toy_inst1 对比测试

### Added
- src/SpaTrackerV2/: 克隆 henry123-boy/SpaTrackerV2（ICCV 2025，免 mesh 先验前馈 3D 点跟踪）
- tools/analysis/spa_track_object.py: SpaTrackerV2 适配脚本——2D 点轨迹 + 自有 depth/K/ext 反投影回 VGGT 世界系 + ICP 锚定 P0 + 逐帧加权 Kabsch 恢复刚体位姿；输出 fp_out_<obj>_spa*/{ob_in_world,ob_in_cam} 与 FP 管线格式兼容；轨迹缓存 spa_tracks.npz；支持 --smooth_win/--med_win/--lock_static 滤波实验开关
- output_v2/7_vggt_omega/fp_out_toy_inst1_spa_v2/: SpaTrackerV2 最优配置输出（600 pts, depth_win 1）
- output_v2/7_vggt_omega/fp_vis/toy_inst1/combo_spa.gif + combo_fp_rekey.gif: 两方法 GIF 对比

### Changed
- tools/analysis/fp_gif_2d3d.py: --poses 支持绝对路径（非绝对路径仍按 fp_out_<obj>/<subdir> 解析）

### Removed
- tools/analysis/dbg_icp.py: ICP 调试临时脚本

### 实验结论
- toy_inst1: SpaTrackerV2 IoU 0.479 / path 1.09m（GT 1.12m，-2.8%）；FP rekey IoU 0.529 / path 1.71m（+52%）
- SpaTrackerV2 轨迹精度大幅占优，IoU 略低（深度反投影逐帧噪声 ~35mm）
- 平滑实验：savgol21 崩溃（0.04）、savgol7 0.464、中值滤波+测地均值快旋转帧失败（0.00）、原始最优
- 教训：fp_data mesh.obj 已在世界系摆放，ICP 初始值必须含单位阵；point-to-plane 精配需守护
- 依赖坑：pyceres==2.4（2.6 与 pycolmap 冲突）、utils3d 需 git@d3a577a 版、moviepy==1.0.3

### ⚠️ Docs to Review
- docs/questions.md: 已追加 Q28（本次同步完成）

---
