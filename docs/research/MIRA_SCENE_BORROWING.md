# Mira-Scene 调研与 RAS 借鉴方案

> 调研日期：2026-10-16 ~ 2026-10-18
> 关联 Q&A：questions.md Q34（区别/效果/部署）、Q35（尺度/API key）、Q36（prompt 资产）、Q37（RAS prompt 诊断）、Q38（SAM3/VLM 分工）、Q39（改进路线）

---

## 1. Mira-Scene 是什么

- **论文**: Mira-Scene: Pixel-Aligned Layouts for Generative 3D Scene Reconstruction（arXiv:2609.23796，2026-09，HKU + VAST）
- **代码**: https://github.com/VAST-AI-Research/Mira-Scene （**MIT License**，代码/两阶段 checkpoint/数据集/全部 prompt 均开源）
- **项目主页**: https://sunyangtian.github.io/Mira-Scene-web/
- **Checkpoint**: HF `Yang-Tian/Mira-Scene`；数据集: HF `Yang-Tian/Mira-Scene-Dataset`
- 本地副本: `/mnt/data_8THDD/lza/tmp/Mira-Scene`（tar 包下载，约 60MB）

**一句话**: 单张图像 → 组合式 3D 场景重建。多模态扩散 Transformer 联合预测每个物体的 mesh + Canonical Coordinate Map（CCM，像素级 canonical 坐标图），与单目深度点云（PCM）做鲁棒几何对齐得到物体位姿，再重力/支撑感知摆放，最后生成环境全景图。

### 效果（官方评测 vs 闭源 SAM3D，仅 80k 开源数据）

| 指标 | SAM3D | Mira-Scene |
|---|---|---|
| 3D-IoU ↑ | 0.520 | **0.727**（相对提升 39.8%） |
| 2D-IoU ↑ | 0.672 | **0.783** |
| CD ↓ | 0.027 | **0.021** |
| ADD-S ↓ | 0.078 | **0.031** |
| EMD ↓ | 0.163 | 0.169（略差） |

布局精度显著领先，几何质量持平。

### 与 RAS 的定位区别

| 维度 | RAS | Mira-Scene |
|---|---|---|
| 输入 | 多帧视频（VGGT + SAM3） | 单张图（生成式补全） |
| 布局 | 重建/交互检测直接测得 | CCM 稠密对应 → 几何对齐 |
| 资产 | 观测重建 mesh（保真但受漂移/校准/遮挡影响） | 生成 mesh（SAM3D/TRELLIS.2，完整但可能与真实有出入） |
| 动态 | 有（交互检测、pick/place 帧） | 无，纯静态 |
| 尺度 | 逐物体点云校准（坑多） | 有界 canonical 空间 + 单目深度全局尺度 |
| 输出 | all_instances.pkl / GLB → SAPIEN | 场景 GLB + scene graph + 环境图 → Blender/Isaac |

**结论: 互补而非替代。** RAS 走视频观测路线保动态交互；Mira 走单图生成路线保完整资产。

---

## 2. 尺度问题：RAS 的校准链条能否被替代

Mira **不需要 RAS 那套逐物体点云校准**（k 因子、底面中心锚点、自校验 guard）：

- 场景尺度由**单目深度模型统一给出**（depth 阶段: PPD / MoGe-2 / DepthAnything，输出 camera-space 点云 PCM）；
- 物体几何在**有界 canonical 空间**预测（CCM），位姿靠 canonical↔scene 稠密对应做几何对齐——没有"逐物体缩放"环节。

**注意**: 默认配置 `depth: {method: ppd, enable_metric: false}` 是**相对深度**——布局拓扑正确但非米制。物理仿真必须开 `enable_metric: true` 或换 metric 模型。且单目深度绝对尺度误差依然存在，Mira 只是把"每物体一个尺度"简化成"一个全局尺度"，不是消灭尺度误差。

---

## 3. API key 与依赖（部署门槛）

`local.yaml` 顶部 `api.base_url: https://lumina.tripo3d.com/v1` 是 VAST 自家 Lumina 图像生成 API。用到它的两处：

| 阶段 | 用途 | 可否绕过 |
|---|---|---|
| `environment` | 单图外推生成 360° 环境全景图 | **可跳过**，不影响场景几何输出 |
| `segmentation` | `vlm_model: gemini-2.5-pro` 生成物体描述/scene graph | 可换本地开源 VLM 或 Web UI 手动标注 |

其他部署要求：
1. 各阶段 Python/PyTorch/CUDA 互不兼容，需建多个 conda 环境（segmentation/depth/ccm/mesh/scene），在 `local.yaml` 配解释器路径；
2. 依赖外部仓库与权重（SAM3、MoGe/PPD、SAM3D 或 TRELLIS.2）；
3. 支持 `--from-stage/--to-stage` 断点续跑、`--gpu-ids` 多卡分片、SAM3D/TRELLIS.2 双 mesh 后端；
4. 官方建议先过 Web UI 交互审核分割与 scene graph 再跑后续（自动分割会漏检/误检）。

---

## 4. Prompt 资产清单与借鉴方案（核心）

Prompt 都在 `infer_scripts/segmentation/` 下，**单个仅 2~3KB**（只有 agentic 那份 66KB，不必搬）。

### 4.1 五份可借鉴 prompt

| 文件 | 大小 | 内容 |
|---|---|---|
| `automatic/prompts/list_objects_major_v5.txt` | 3.1KB | 房间级物体列举（SAM3 grounding 词），16 条规则 |
| `automatic/prompts/list_objects_tabletop_v1.txt` | 2.4KB | 桌面小物体专用模式 |
| `automatic/prompts/missing_objects_major_v2.txt` | 2.1KB | 漏检批判器（cyan 遮罩后找漏） |
| `automatic/prompts/agent/system_prompt.txt` | 66KB | agentic SAM3 grounding 循环（仅参考策略） |
| `interactive/scene_graph.py` L38 | 0.6KB | scene graph JSON 输出 prompt |

### 4.2 RAS 现有 prompt 的问题诊断

`tools/generate_scene_json_stage1.py`：

- **`VLM_DETECT_PROMPT`（L149-160）**: 只有 9 行规则，且 **"List each object type once per frame" 直接命令 VLM 把同类别多实例合并成一条**——Q33 统计的实例坍缩（66 场景 21→7、141 棋子 9/32）的 prompt 层根因。缺少: 合并规则（盆栽+盆、床品归床）、位置限定词消歧、遮挡描述、边界裁剪处理、颜色/材质命名。
- **`_build_relationship_prompt`（L213-252）**: 已有 has_floor 兜底补丁，但只传物体名列表不传图，VLM 无法区分同类别多实例；只有 4 种关系，无 direct_supporter 指向与 confidence。
- **SAM3/VLM 分工（Q38）**: SAM3 是条件分割器（`set_text_prompt(类别)`），不会自己发现类别；VLM 不可替代的职责是①类别发现②关系判断③场景类型/地面参考点。Mira 同分工，其 agentic 循环本质也是 VLM 驱动。**VLM 去不掉，只能改产出质量。**
- **重复分割不是问题**: `_deduplicate_masks`(IoU 0.5) 已兜住；真正吞实例的是 `CENTROID_DIST_THRESHOLD=0.15m`——棋子间距 2~4cm，32 个全被并成 1 个聚类。

### 4.3 改法（按优先级）

**改法 1 — 重写 VLM_DETECT_PROMPT（实例坍缩根因）**

```python
VLM_DETECT_PROMPT = """List every independently meaningful object visible in this image, one entry per physical instance. Output JSON only.

{{"objects": [{{"name": "<object1>"}}, {{"name": "<object2>"}}]}}

Rules:
- One entry per physical instance. When several similar objects exist, make names unique with location qualifiers (e.g. "black chair on the left", "black chair on the right"). Never write exact counts like "six chairs".
- Simple English noun phrases, lowercase, no articles. Include color/material when visually clear (e.g. "white ceramic cup").
- For thin tall cylinders prefer "bottle" over "cup".
- Merge into one entry only when physically unified: plant + its pot; cup + liquid inside; objects inside an open container are SEPARATE entries named "X in the Y".
- Ignore: hands, body parts, walls, floors, ceilings, and uninformative fragments cut off by the image border.
- An object partially occluded by another object is still one entry if recognizable.

Output JSON:"""
```

**改法 2 — 加 missing_objects critic**: 检测+SAM 后，已检出实例涂 cyan 再调一次 VLM 找漏检（照 `missing_objects_major_v2.txt`），漏的补进来。

**改法 3 — 关系 prompt 加 direct_supporter**: 输出格式改为 `{"object": ..., "supporter_index": <物体列表下标>, "relation": "...", "confidence": 0.8}`，一次调用拿到支撑对象身份，减少 `refine_other_objects_relations.py` 的纠错轮次。

**改法 4 — 去重阈值自适应（必须与改法 1 同步做）**: `CENTROID_DIST_THRESHOLD` 从固定 0.15m 改为 tabletop 场景 0.03~0.05m 或按物体中位尺寸缩放。**只改 prompt 不改去重，实例照样被吞。**

**改法 5 — relation 词表对齐**: 借鉴 Mira `rests_on / fixed_to / hangs_from`（+ motion ∈ {fixed, dynamic}），与 `relations_scene_graph.json` 结构对齐，为后续复用 Mira 风格 scene graph 铺路。

**注意**: 改成"每实例一条"后，下游 `sg_deduplication` 的 UnionFind、`SYNONYM_MAP` 会收到带限定词的名称（"black chair on the left"），同义词映射与跨帧匹配需跟着适配，否则会去重失败变重复实例。

---

## 5. 超出 Mira 的更大改进: 首帧生成资产路线（针对手部遮挡）

RAS 逐实例多帧重建的"手部遮挡"痛点，2025-2026 已有多篇工作给出"生成式资产替换观测重建"的对症方案：

| 工作 | 思路 | 对 RAS 的价值 |
|---|---|---|
| Zero-shot In-Scene Object Manipulation Reconstruction (arXiv 2512.19684) | 遮挡物体走 Amodal3R 生成，未遮挡走 Hi3DGen，FoundationPose 逐帧对齐 6D 位姿 | 与 RAS 痛点完全对症，方案可直接抄 |
| ForeHOI (CVPR 2026, arXiv 2602.06226) | 前馈"2D mask 补绘 + 3D 形状补全"，专治手持物体重度遮挡，比优化法快约 100 倍，代码+数据开源 | 逐实例重建的替换/对比基线 |
| SimRecon (arXiv 2603.02133, 清华+Galbot) | Perception-Generation-Simulation 三级管线，逐实例生成替代纯重建 | 与 RAS 定位最接近的最新管线 |
| Subsecond 3D Mesh Generation for Robot Manipulation (arXiv 2512.24428) | 生成 mesh → FPFH/RANSAC/ICP 配准到点云恢复尺度位姿，<1s，接物理仿真 | 对齐方案可整体移植到 SAPIEN 链路 |
| Scalable Real2Sim (IROS 2025, MIT, 代码开源) | alpha 透明渲染把手与物体几何分离 | 改进逐实例重建的遮挡分离 |
| EgoHOS (arXiv 2208.03826) | 手物精细分割 | 自动挑"无手干净帧"作生成输入 |

生成后端选型：
- **TRELLIS.2**（Microsoft，MIT，16-24GB 显存，512³ 约 3s / 1536³ 约 60s PBR 纹理）——首选
- **Hunyuan3D 2.x**（Tencent，代码 Apache-2.0，低至 6GB 显存）——显存受限备选
- SAM 3D Objects（Meta，遮挡鲁棒训练，license 商用受限待验证）
- Mira-Scene 的 mesh 后端即 SAM3D / TRELLIS.2，验证了这条组合的可行性

其他组件级备选（可选）: Segment Any Motion（arXiv 2503.22268，代码开源，用 VGGT 轨迹点做分割 prompt 替代质心/重叠启发式）、WHOLE（arXiv 2602.22209，手物联合位姿）、TwinTrack（arXiv 2505.22882，物理校验修正跟踪漂移）。

---

## 6. 建议实施路线

1. **近期（prompt + 去重，改动小、确定性高）**
   执行 4.3 的改法 1~4，拿 141（棋子）/ 66（跨类去重）场景重跑 stage1 验证实例数恢复。

2. **中期（资产层替换，核心）**
   EgoHOS 选无手帧 → TRELLIS.2 / Hunyuan3D 生成 → mask+depth ICP 对齐到 VGGT 帧（抄 Subsecond 3D 方案）→ **失败回退现有多帧重建**。一步同时解决: 手部遮挡、无纹理（stage4/6 全跳过问题）、部分尺度问题。

3. **远期（布局与交互升级）**
   Mira-Scene CCM 布局思想对照 stage5 摆放精度；Segment Any Motion 替代质心/重叠交互检测。

---

## 附: 复现/试跑 Mira-Scene 的检查单

```bash
# 1. 克隆（网络不稳时用 tar 包）
git clone --depth 1 https://github.com/VAST-AI-Research/Mira-Scene.git
# 或: curl -L -o mira-scene.tar.gz https://codeload.github.com/VAST-AI-Research/Mira-Scene/tar.gz/refs/heads/main

# 2. 按 docs/environment.md 建多个 conda 环境（segmentation/depth/ccm/mesh/scene 各自独立）

# 3. cp infer_scripts/config/example.yaml infer_scripts/config/local.yaml
#    - 替换所有 /path/to 外部仓库与 checkpoint 路径
#    - api.key 填 Lumina API（或跳过 environment 阶段、VLM 换本地模型）

# 4. 交互式分割审核
python infer_scripts/0_segmentation.py --web --input <img_dir> --output <out> \
  --config infer_scripts/config/local.yaml --host 0.0.0.0 --port 8890

# 5. 分支运行: depth→floor, environment(可跳过), ccm→scene
python infer_scripts/pipeline.py --input <img_dir> --output <out> \
  --config infer_scripts/config/local.yaml \
  --mesh-backend sam3d --from-stage ccm --to-stage scene
```
