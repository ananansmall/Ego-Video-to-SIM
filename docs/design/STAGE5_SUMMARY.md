# Stage 5 高级语义精修 — 近一月改动总结

> 记录时间: 2026-10-14
> 覆盖范围: 2026-09-14 ~ 2026-10-14（git log 近 30 天）
> 主线代码: `mainv3.py` / `mainv2.py` / `src/sp_refinement.py` / `tools/run_stage55.py` / `tools/stage55_cli.py`

---

## 一、Stage 5 的完整子阶段

按代码实际执行顺序（`mainv3.py:1252` 起）：

```
Stage 5 (--enable_stage5)
├── 5.1  关系推断    scene_graph / per_object 两种方式
├── 5.2  SP 精修     on_top / inside / against_side / hanging_below / leaning
├── 5.25 摆放补正    ├── ground_floating_objects   (悬空/陷地 → 地面)
│                   └── reseat_on_top_objects     (on_top 物体贴回支撑物顶面)
└── 5.5  确定性 QC    tools/run_stage55.py + stage55_cli.py
```

> **注意命名撞车**：`alignment_stage` 的取值 `stage1_strict / stage2_relaxed /
> stage3_large_plane / stage4_geocalib` 是**坐标系对齐降级链**的编号，与这里
> 的 Stage 5.x 完全无关，也**不是** `mainv3.py` 里的 `--enable_stage4`（ICP 迭代
> 视觉-空间对齐）。三套 "stage" 编号体系并存，读代码时需按上下文区分。

---

## 二、新增能力（按重要性）

### 1. `ground_floating_objects` — 悬空/陷地补正（`src/sp_refinement.py:61`）

**为什么需要**：基础精修读的是 Stage1 的**类别级**关系，而关系往往到 5.1 才被
推翻。实测 `egodex_4 cloth_0`：Stage1 判 `attached to wall` → 基础精修走挂墙
分支不落地；5.1 改判 `supported by floor` 时位姿已定，且 5.2 的穿模修复只做
"分开"不做"落地"，反而把悬空推得更高。

**行为**：只改 `T[2,3]`，不扶正朝向、不动旋转/xy/mesh 顶点。

与 `refine_supported_by_floor_object`（Stage3 基础精修）的关键区别：**不扶正物体
y 轴**。那个函数为落地家具设计，会把物体 y 轴对齐重力方向，对手持小物体产生
109~119° 的意外旋转（实测 bracelet / stapler / screwdriver 等 5 个实例）。

**分支逻辑**：

| `z_min` | 行为 | 理由 |
|---|---|---|
| `abs(z_min) <= float_eps` (0.02) | 跳过 | 已贴地 |
| `z_min < 0`（陷地） | **压回 z=0** | 陷地没有"坐标系未对齐"的歧义解释，一定是几何/关系误判 |
| `z_min > float_z_max` (0.5) | 跳过 | 疑似坐标系未对齐，强行压平会掩盖真 bug |
| `0 < z_min <= 0.5` | 压到地面 | 真实落地偏差 |

**10-14 新增陷地分支**：原来 `if z_min <= float_eps: continue` 把陷地和已贴地
一起跳过。实测 `basic_pick_place_46 card_0`（5.1 把 `supported by other objects`
误判为 floor → Stage3 贴地精修没跑 → 保持 VGGT 原始位置 `z_min=-0.049m`）。

**实测**（`basic_pick_place_17`）：
```
[ground] toy_1: 悬空 z=+0.4855m → 落地 z=+0.0000m   ✅
[ground] toy_2: 跳过 (z_min=+0.5447m > 0.5m)        ← 阈值保护
[ground] toy_3: 悬空 z=+0.2120m → 落地 z=+0.0000m   ✅
[ground] toy_5: 悬空 z=+0.3770m → 落地 z=+0.0000m   ✅
```

---

### 2. `reseat_on_top_objects` — on_top 物体重贴支撑物顶面（`src/sp_refinement.py:215`）

**为什么需要**：5.2 的 on_top 精修本来贴对了，但随后的**穿模修复会把物体推开**
—— on_top 贴到顶面时接触面间隙为 0，穿模检测把这种零间隙接触判成重叠并沿 z
分离；物体掉到地面后底面共面，又被判重叠，形成自维持循环。

实测 `egodex_33`：23 次穿模修复 **100% 是 z 轴分离**且 **100% 涉及 on_top 物体**，
`toy_1` 被推 9 次 × 0.0056m = 0.050m，从 `z=0.065` 掉到地面（`plate_0` 顶面仅 0.036m）。
全库 104 个 on_top/inside 关系**只有 3 个贴合，坏 101 个**。

**行为**：只改 z 平移（gap 修正量），不动旋转/水平位置/尺寸。

**相对初版的三个修正**：

1. **拓扑序 + 多轮迭代（max_iter=4）**。初版单次遍历 dict 顺序，遇到 `A→B→C`
   链时若 A 先处理会读到 B 的旧顶面，B 之后被抬到 C 上就会让 A 悬空。
   实测 15 条多层链（`toy_0→lego_0→plate_0` 等）。
2. **环检测**。实测 `egodex_15` 存在 `sheet_0 ↔ sheet_1` 二元环，互相往上贴会
   无限发散；断开投影面积最大者的出边，让它当链的固定根。
3. **`supported by other objects` 不再躺平**。实测 52 个场景有 21 个这类物体，
   之前两道补正都跳过它们，位姿完全等于 VGGT 原始预测；现在用 `_infer_supporter`
   做几何兜底挑支撑物（`infer_min_overlap=0.30`、`max_drop=0.60`）。

**保护机制**：
- `inside` 保护：无策略文件时，陷入深度超过自身半高判为 inside 不动（避免把碗里
  的面包拽到碗沿上）
- `min_overlap=0.10`：水平重叠低于该比例认为支撑物判错，不硬拽
- `severe_penetration`：陷入深度超过支撑物高度，判为坐标系数错，不修

**离线预测全库**：463 个物体有问题 123 → 修复后剩 60。
**首次真实触发**：`121_verify_b` 的 `egg_0: gap +0.0790m → -0.0000m`
（支撑物 `plate_0` 顶面 0.0495m，重叠 86%）。

---

### 3. `dump_placement_audit` — 摆放审计落盘（`src/sp_refinement.py:396`）

每次 Stage 5.25 跑完输出 `placement_audit.json`，逐实例记录：

```json
{
  "n_objects": 9,
  "n_problem": 0,
  "n_by_status": {"ok": 9},
  "objects": [
    {"name": "toy_0", "relation": "supported by bowl_0", "min_z": 0.207,
     "max_z": 0.291, "supporter": "bowl_0", "supporter_height": 2.121,
     "gap": 0.0, "xy_overlap": 1.0, "status": "ok"}
  ]
}
```

`status` 取值：`ok` / `floating` / `sunken` / `severe_penetration` / `inside` /
`no_supporter_found` / `unsupported`。

> `unsupported` 是 `attached to wall` 等**非支撑类关系**的默认标记（挂在墙上的布
> 本来就没有"支撑物"），**不计入 `n_problem`**，不要误读为场景错误。

---

### 4. Stage 5.5 — 确定性 QC / agent 兜底（`tools/run_stage55.py` + `stage55_cli.py`）

独立可复跑的质检层，产出 `final_scene_stage5_sp.glb` / `stage55_work.glb` /
`final_scene_stage55.glb`。

---

## 三、踩过的坑（重要）

### 坑 1：`final_scene.glb` 不是最终产物（10-14 修复）

代码注释自相矛盾：
```
mainv3.py:36   final_scene.glb = Stage3 基础精修产物 (固定起点, 不再更改)
mainv3.py:141  final_scene.glb # 最终场景GLB (始终为最新结果)
```
实现是前者 —— Stage 5 的产物写到 `final_scene_stage5.glb`，而 `final_scene.glb`
在 Stage 5 **之前**就写好了。任何按名字直觉取 `final_scene.glb` 的下游（`assess/`、
`stage4/`、`object_tracking/`）拿到的都是**未精修**场景。

**后果**：批量体检时看到 `vertical_pick_place_80` 陷地 -0.885m、
`vertical_pick_place_11` 陷地 -1.168m 等"严重问题"，实际全是假象。改用
`final_scene_stage5.glb` 后，17 个场景 z_min 全部落在 `[-0.05, 0]`。

**修法**：改名 `final_scene_basic.glb`，全库 74 处引用分流：
- 起点用途（`stage4/`、`scripts/`、`tools/run_post_pipeline.py`）→ `final_scene_basic.glb`
- 最终用途（`assess/`、`object_tracking/`）→ `final_scene_stage5.glb`

### 坑 2：`_FLOOR_Z_MAX` 门限与贴地自相矛盾（E 项修复）

`refine_supported_by_floor_object` 的门限只在 `theta_gravity ∈ (10°, 170°)` 时
"不扶正"，但下面**无条件**把物体贴到 z=0。二者矛盾：物体被判定为"朝向不可信"
却仍被强行摆到地面，看起来落地了，实际掩盖了上游坐标系对齐错误。

实测历史日志 290 条：`theta_gravity ∈ [0.1°, 177.6°]`，其中 **170 条（59%）**
落在不扶正区间，`z_min ∈ [-1.5146m, 1.7867m]` 仍全部贴到 z=0。

**修法**：恢复 z_min 阈值（0.5m）。仅当物体确实接近地面时才落地，否则保持原位
并告警，让"坐标系可能没对齐"暴露出来。

### 坑 3：统计源用错

曾用 `pose_changes.json` 的 `relation`（Stage1 类别级）当 floor 标签，实际应使用
`final_relations.json`（5.1 实例级），两者不一致率 8/10。导致"floor 悬空 1.188m"
等结论作废。

---

## 四、动态物体定位（`mainv3.py` Stage 3 之后）

不属于 Stage 5，但与 5.25 的摆放补正协同，且是"渲染很奇怪"的另一半原因。

**原 bug**：动态物体位置用 `first_visible_frame_id`（**首次被 SAM3 检测到的帧**），
而不是"抓取前的静止位置"。两个问题：
1. 物体可能在视频开头就已经在动（一开始手就拿着它），那"首次可见帧"里的位置
   不是静止位置
2. `first_visible_frame_id == optimal_frame_id` 时**整个位置+旋转调整被跳过**

**修法**：Stage 3.7 已给出 `pick_frame`（抓取时刻），初始帧取 **pick 帧之前、
该实例仍可见的最后一帧**。注意帧空间换算：`pick_frame` 在原始视频帧空间，
`centroid_data` 的 fid 在 RGB 帧空间，需除以 `frame_ratio`。

**实测**：
```
vertical_pick_place_115: mesh帧41 → 初始帧22 (抓取前), offset=0.3530m
basic_pick_place_111:    mesh帧46 → 初始帧25 (抓取前), offset=0.2060m
vertical_pick_place_101: mesh帧37 → 初始帧24 (抓取前), offset=0.1465m
```

### 4.1 首帧策略演进：为什么不能取「pick 前最后一帧」

初版实现是 `initial_frame_id = max(fid for fid < pick_rgb)`，实测被两类问题打穿：

| 问题 | 实测数据 | 后果 |
|---|---|---|
| **单帧离群点** | `basic_pick_place_17` toy_inst0 在 frame 21 单帧 z 从 +0.06 跳到 **+0.24**，下一帧立刻回落 +0.08；pick=29，最后一帧正好命中它 | 初始位置被放到 **+0.242m**（正确值 +0.007，差 34 倍） |
| **pick 检测偏晚** | `basic_pick_place_118` block 从 frame 22 起**每帧只动 1cm**（渐进上升），pick 判到 31；frame 30 已累计抬升 6cm | 初始位置取到 **+0.094m**（正确 +0.035） |

**终版策略**：pick 前最长稳定段（相邻帧质心位移 ≤ 0.02m）的**中位数**帧。

**为什么不是"稳定段最后一帧"**：实测 `118` 的 0~30 帧每帧位移都 ≤0.02（渐进上升骗过单帧判据），整个 0~30 被判为一个稳定段，取最后一帧就退回 +0.094m。中位数落在段中央，天然避开尾部漂移。

| 策略 | 失败数（13 个交互实例） |
|---|---|
| S1 pick 前最后一帧 | **2** ❌ |
| **S2 最长稳定段中位数** | **0** ✅ 采用 |
| S3 最靠后接近首帧 | 0 ✅ 但容差放宽后偏差更大（0.007~0.025m） |
| S4 稳定段内最后一帧 | **1** ❌ 被渐进上升坑 |

### 4.2 「首帧错误」与「高空悬空」是两类不同问题

容易混淆，但对比如下：

| | **A. 首帧错误** | **B. 高空/陷地** |
|---|---|---|
| 对象 | **动态物体**（有 pick/place 交互） | floor 关系物体（多为静态） |
| 本质 | 用了错误的**参考帧**（mesh 生成帧 ≠ 抓取前） | 位置被上游**推离**正确值 |
| 修法 | 选对参考帧（稳定段中位） | ref_z 参照物区分根因 + 陷地分支 |
| 改什么 | **位置 + 旋转**（实测 rot 达 128~147°） | **只改 z** |
| 视觉 | 物体在错误位置 / 姿态歪 | 物体飘在空中 / 陷地 |
| 命中 | 每个动态物体必中（但 z 可能被 5.25 兜住，**旋转没人兜**） | 只命中被推高的 floor 物体 |

**简单场景（如 egodex_gripper20 单物体抓取）里 A 类更关键** —— 它直接决定"物体被抓之前在哪、什么姿态"，是整个场景复原的基准。B 类更多是穿模修复的副作用，场景性更强。

**`118` 的首帧修复视觉对比**（`bread_0` 从左上角移到紧邻绿球、`block_2` 旋转 128°）：
```
block_2  pre_fix: size=(0.123, 0.128, 0.044)   接近正方
         fixed:   size=(0.117, 0.095, 0.044)   y 明显缩短 → 朝向已变
```

---

## 五、相关测试

| 测试 | 覆盖 | 状态 |
|---|---|---|
| `test/test_reseat_topology.py` | 拓扑序 / 环断开 / other objects 兜底 / inside 保护 | ✅ 20 项 |
| `test/test_ground_sunken.py` | 陷地压平 / 悬空保护 / 非 floor 不动 / 类别级回退 | ✅ 17 项 |
| `test/test_alignment_quality.py` | 重力一致性质量判据 | ✅ 21 项 |
| `test/test_align_info.py` | 对齐链第 1 级 info | ✅ 16 项 |
| `test/test_body_part_filter.py` | Stage1 人体部位过滤 | ✅ |
| `test/test_dynamic_judge_integration.py` | Stage 3.7 集成 + ratio + key 映射 | ✅ 8 项 |
| `test/test_pipeline_vs_gt.py` | 5 个有 GT 场景召回/误检 | ✅ |

**pick_place 召回**：5 个有 GT 场景 **5/5 召回、零误检**（旧三条件判据 0/5）。

---

## 六、验证数据集（output_v3，17 场景）

`assets/egodex_gripper20_videos_results/`（7 × `basic_pick_place_*` +
10 × `vertical_pick_place_*`，每个 60~62 帧）。

**最终体检（用 `final_scene_stage5.glb`）**：

```
15/17 干净
basic_pick_place_17  toy_2 悬空 0.545m   ← 阈值保护的设计行为 (超 _FLOOR_Z_MAX=0.5)
basic_pick_place_46  card_0 陷地 0.049m  ← 10-14 已修复 (离线验证通过)
```

**交互检出**：17 场景检出 13 个交互。5 个"0 交互"经复核确认是无交互
（`net ≤ 0.006`、`pnet ≥ 36` 的来回抖动，非搬运）；`vertical_pick_place_135`
的 `toy_inst0`（path=2.028m / net=0.397m / pnet=5.1）疑似漏检，卡在
`CENTROID_PNET_THRESH = 5.0` 上，**待定是否放宽**。

---

## 七、仍未解决

| # | 问题 | 状态 |
|---|---|---|
| 1 | `ground_floating_objects` 在 output_v3 从未触发 | 俯视场景看不到地面，缺"关系是 floor 但悬空"的样本 |
| 2 | `frame_ratio` 换算的负样本 | 已有 86.mp4(3.019) / 156.mp4(2.906) 两个正样本 |
| 3 | `stage4_geocalib` 从未执行 | 17 场景 100% `stage1_strict`，"劣质成功"挡住降级链 |
| 4 | `pnet` 阈值 5.0 是否放宽 | 等决定 |
| 5 | 惯量占位符 | 旧产物 `ixx=1.0`，需重跑 |
| 6 | `card_0` 类问题需重跑管线才写入 glb | 离线已验证，未端到端 |

---

## 八、环境注意事项

- conda env: `/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python`
- GPU 归属: GPU0=`csy` vlm + `myl` sam3 服务；GPU1=`myl` sam-3d（主模型常驻）；
  GPU2/3=`root` VLLM（**禁用**）
- `myl` 的 sam-3d 服务会**动态加载到任意一张卡**（曾观测到占 18.43 GiB 导致
  4 次 CUDA OOM）。单 mainv3 峰值约 20GB（Stage1 VLM 与 Stage3 SAM3D 分阶段），
  每卡跑一个进程是安全上限
- `cusolverDnCreate` 失败也是资源竞争的表现，非代码 bug
