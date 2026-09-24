# SimFoundry 与 PROSE 对 ReplicateAnyScene 的参考价值分析

> **两篇论文**（2026-06 同期发布，均与 RAS 技术栈高度重叠）：
> - **SimFoundry**: Modular and Automated Scene Generation for Policy Learning and Evaluation
>   - 论文: arXiv:2606.28276 (2026-06-25)
>   - 机构: NVIDIA / Stanford / Georgia Tech / UT Austin / UofT
>   - 项目主页: https://research.nvidia.com/labs/gear/simfoundry/
>   - 代码状态: **未发布**（论文标注 "Under Review"，项目页无 GitHub 链接）
> - **PROSE**: Training-Free Egocentric Scene Registration with Vision-Language Models
>   - 论文: arXiv:2606.16569 (2026-06-15)
>   - 机构: ETH Zurich / VGG-Oxford / ETH AI Center
>   - 项目主页: https://rckola.github.io/prose/
>   - 代码状态: **未发布**（项目页标注 "Code (soon)"）

---

## 〇、为什么把这两篇放一起看

两篇论文在 2026-06 同期发布，且**与 RAS 的技术栈几乎完全重叠**：

| 组件 | RAS | SimFoundry | PROSE |
|------|-----|-----------|------|
| 几何先验 | VGGT / VGGT-omega | VGGT-omega 类（深度+位姿） | **VGGT-Ω**（完全相同） |
| 分割 | SAM3 | SAM3 系列 | **SAM 3**（完全相同） |
| 语言/语义 | Qwen2.5/3.5-VL | Qwen3.6-27B（VLM） | **Qwen3.6-27B**（同家族） |
| 单图 3D 生成 | SAM3D | 2D→3D 生成模型（含 SAM3D 对比） | 不涉及 |
| 目标 | 场景级 3D 重建 (.glb) | 场景级 sim-ready 数字孪生 | 跨时间场景配准 |

**关键发现**：
- **SimFoundry 直接对比并超越了 SAM3D**（Chamfer 距离 + 位姿误差均优于 SAM3D），且其流水线与 RAS 的 Stage1→5 高度对应，并补齐了 RAS 缺失的"背景重建 / 物体去穿模 / 物理稳定性"环节。
- **PROSE 与 RAS 用的是同一套 foundation model**（VGGT-Ω + SAM3 + Qwen VLM），其"VLM 跨扫描物体匹配 + 高度分桶 + 逐实例 RANSAC 投票"思路可直接迁移到 RAS 的去重/对齐阶段。

下文按论文分别分析，最后给出综合建议。

---

## 第一部分：SimFoundry

### 一、SimFoundry 核心能力概述

SimFoundry 是 NVIDIA/Stanford 联合推出的**单视频→交互式仿真场景**全自动系统，核心定位是把真实视频变成 sim-ready 数字孪生 + 可扩展的"digital cousins"用于策略训练与评估。

| 能力 | 说明 |
|------|------|
| 单视频→数字孪生 | 从单个 RGB 视频自动重建可交互的仿真场景 |
| 模块化流水线 | 感知 / 资产生成 / 位姿对齐 / 铰接 / 物理标注 / 数据生成，各模块可替换 |
| 刚体 + 铰接物体 | 支持铰接物体（关节生成、物理参数），RAS 暂不支持 |
| 背景重建 | 前景擦除 + 3D Gaussian Splat 重建背景，RAS 无此环节 |
| 物体去穿模 + 物理稳定 | E.4 节专门处理穿透与稳定性，直接对应 RAS Stage5 痛点 |
| Digital Cousins | object/scene/task 三级变体，用于 sim2real 泛化 |
| Real2Sim 评估 | Pearson 0.911 与真实策略表现强相关 |
| Sim2Real 训练 | 仿真训练的策略可零样本迁移到真实（含未见物体/布局） |

**关键创新**：模块化设计让 foundation model 升级时只需"换组件"而非重设计系统；"digital cousins"把单个重建场景扩展成无限训练数据。

---

### 二、SimFoundry 与 RAS 的定位对比

| 维度 | SimFoundry | RAS |
|------|-----------|-----|
| **终极目标** | sim-ready 数字孪生 + 策略训练/评估 | 场景级 3D 资产 (.glb) |
| **输出** | USD/Isaac Sim 场景 + 物理参数 + 铰接 | 组合式 .glb + 空间关系 |
| **背景** | 3DGS 重建（高保真） | 无（仅物体 + 点云） |
| **3D 生成** | 2D→3D 生成模型（含 SAM3D 对比） | SAM3D |
| **去穿模** | E.4 depenetration + 物理稳定性 | Stage5 穿模修复（效果差，Q65/Q73） |
| **铰接物体** | 支持（关节+物理参数） | 不支持 |
| **位姿对齐** | 基础模型对齐 + 3min/物体人工微调 | Stage4 ICP+Umeyama |
| **多物体去重** | 有 | Stage2 类内/跨类去重 |
| **坐标系** | 仿真世界系（rigid bridge） | PCA 对齐 floor/wall → z-up |
| **策略评估/训练** | 完整闭环（Pearson 0.911, sim2real） | 无（超出范围） |

**核心差异**：SimFoundry 是**完整的 real2sim2real 闭环**，RAS 是**其中的 real2sim 几何重建子集**。SimFoundry 在 RAS 覆盖的每个环节几乎都有对应或更强的实现，并补齐了 RAS 缺失的背景、铰接、物理稳定性、sim2real 等环节。

---

### 三、SimFoundry 参考价值分析（按优先级排序）

#### 3.1 最高价值：物体去穿模 + 物理稳定性 (E.4) → 解决 Stage5 穿模痛点

**RAS 当前痛点**（Q65/Q73 已记录）：
- Stage5 穿模修复效果差，只调整小物体 x,y，层级调整缺失
- 物体间相互嵌入无法可靠分离
- 物体浮空 / 沉入地面，无物理稳定性约束

**SimFoundry 的解法**（论文 E.4 节）：
- 专门的 **depenetration** 流程：把嵌入的物体几何分离
- **物理稳定性检查**：在物理仿真器里 sanity-check 整体场景配置，确保物体不浮空/不沉地
- 论文报告：12 个重建场景 F1 0.81–0.92，3 分钟/物体微调后可达 0.93–0.99

**可借鉴方案**：

```
当前 RAS Stage5:
  relations_json → 平移调整 (仅 x,y 小物体) → 效果差

改进思路:
  Stage5 引入 depenetration + 物理稳定性验证
    → 物体间嵌入分离 (不只调 x,y，按法线方向推开)
    → floor/wall 接触约束 (物体底面贴合 z=0)
    → 重力稳定性验证 (在仿真器里测是否倾倒)
```

**可行性**：SimFoundry 此部分依赖物理仿真器（Isaac Sim），RAS 若引入需接入轻量物理引擎。路线图 Stage5 已规划物理引擎（见 `stage5_physics_engine_proposal.md`），SimFoundry 的 depenetration 流程可直接作为该引擎的核心子模块。

#### 3.2 最高价值：超越 SAM3D 的 3D 重建 → 替代/增强 Stage3

**RAS 当前痛点**（Q6/Q7）：
- SAM3D 单图生成的 mesh 质量受限，受手部遮挡污染
- VGGT pointmap 噪声 → SAM3D 几何条件错误 → l2c/T 偏移
- 大物体（桌子）精度不足

**SimFoundry 的证据**（项目页 "Reconstructed Scene v.s. SAM3D output"）：

| 场景难度 | Chamfer 距离 (m) ↓ | 位姿误差 (m) ↓ |
|---------|-------------------|---------------|
| | SimFoundry Tuned vs SAM3D Zero-Shot | SimFoundry Tuned vs SAM3D Zero-Shot |
| Easy | 0.0042 vs 0.0160 | 0.0041 vs 0.0060 |
| Medium | 0.0047 vs 0.0180 | 0.0057 vs 0.0076 |
| Hard | 0.0091 vs 0.0220 | 0.0073 vs 0.0180 |

SimFoundry 在所有难度上 Chamfer 距离和位姿误差都显著优于 SAM3D zero-shot，且在遮挡/杂乱场景优势最大。

**可借鉴思路**：
- SimFoundry 用 "Tuned" 模式（3 分钟/物体微调）达到接近真值的精度，RAS 当前 Stage3 是 zero-shot 无微调
- 论文 L.1.2 节专门描述了 "SAM3D Reconstruction Pipeline"，并做了定量对比——值得精读其 SAM3D 调用细节，看是否有可借鉴的预处理/后处理
- 长期可考虑用 SimFoundry 的 2D→3D 生成模型替换 SAM3D

#### 3.3 高价值：背景重建 (3DGS) → 补齐 RAS 缺失环节

**RAS 当前状态**：只重建物体 + 点云，无背景（墙壁/地板只是平面 mask）

**SimFoundry 的解法**（E.5 节）：
1. **前景擦除**：把前景物体从视频中抹掉，生成 background-only 视频
2. **前景 inpainting**：修复被前景遮挡的背景区域
3. **metric 深度+位姿恢复**：用 foundation model 恢复度量深度
4. **深度监督的 splat 训练**：3DGS 用 depth 监督
5. **rigid bridge 到仿真器**：把 splat 背景和物体 mesh 对齐到统一世界系

**对 RAS 的意义**：
- RAS 当前 `point_cloud.ply` 只是稀疏点云，背景质量低
- 引入 3DGS 背景可极大提升场景视觉真实感
- "rigid bridge" 思路对应 RAS 的坐标系对齐，可借鉴其对齐方法

#### 3.4 中等价值：铰接物体生成 → 扩展 RAS 能力边界

**SimFoundry 的铰接流程**（E.3 节）：
- 分割铰接物体的可动部分
- 关节生成（revolute/prismatic）
- 物理参数（摩擦、质量、阻尼）

**对 RAS 的意义**：RAS 当前只处理刚体。若要支持柜门、抽屉等铰接物体，SimFoundry 的铰接生成流程是现成参考。但这是能力扩展，非当前痛点。

#### 3.5 中等价值：代表性帧选择 → 对比 RAS 的最优视角帧

SimFoundry E.1 节有 "Representative Frame Selection"，与 RAS Stage3 的 `get_optimal_view_frame_id()` 目标相同。值得对比两者选帧策略的差异，可能改进 RAS 的选帧逻辑。

#### 3.6 低价值（超出 RAS 范围）：Digital Cousins + Sim2Real

| 特性 | 价值 | 说明 |
|------|------|------|
| Object cousins | 低 | 生成同类变体物体，属数据增强，RAS 不做训练 |
| Scene cousins | 低 | 布局变体，属 sim2real 泛化 |
| Task cousins | 低 | 任务变体，属策略训练 |
| Real2Sim 评估 | 低 | Pearson 相关性，属策略评估 |
| Sim2Real 训练 | 低 | 策略迁移，超出 RAS 重建目标 |

这些是 SimFoundry 的核心贡献，但**超出 RAS 的场景重建范围**。RAS 关注的是"把视频变成 3D 场景"，SimFoundry 关注的是"用 3D 场景训练机器人策略"。

---

### 四、SimFoundry 不适用 / 需注意的部分

| SimFoundry 特性 | 不适用/注意原因 |
|----------------|-----------------|
| 依赖 Isaac Sim | RAS 是离线管线，引入 Isaac Sim 较重 |
| 3 分钟/物体人工微调 | RAS 追求全自动，但可借鉴其交互式对齐思路（K 节） |
| NVIDIA 闭源组件 | 部分 foundation model 可能非开源 |
| 代码未发布 | "Under Review"，无法直接复用 |
| Sim2Real 闭环 | 超出 RAS 重建目标 |

---

## 第二部分：PROSE

### 一、PROSE 核心能力概述

PROSE 是 ETH Zurich 提出的**跨时间自我中心场景配准**方法：给定同一室内空间的两次 ego RGB 扫描（不同时间、物体可能移动），恢复两者的刚体变换。**核心是用同一个预训练 VLM 同时做场景理解和跨扫描物体匹配，无需训练。**

| 能力 | 说明 |
|------|------|
| 跨时间场景配准 | 两次 ego RGB 扫描 → 刚体变换 |
| 物体级 3D 场景图 | 每次扫描生成 object-level scene graph |
| 训练免费 | 0 可学习参数，全部用现成 foundation model |
| RGB-only | 无需深度传感器、无需标注图 |
| 开放词汇 | VLM 列举物体，不限于固定类别 |

**关键创新**：把"跨时间物体匹配"转化为 VLM 擅长的"视觉比较任务"，用高度分桶 + Set-of-Marks + same/different 验证 + 逐实例 RANSAC 投票实现鲁棒配准。

---

### 二、PROSE 与 RAS 的定位对比

| 维度 | PROSE | RAS |
|------|-------|-----|
| **任务** | 跨时间两次扫描配准 | 单视频场景重建 |
| **输入** | 2× ego RGB 序列 | 1× 场景漫游视频 |
| **输出** | 刚体变换 + 2× 场景图 | 组合式 3D 场景 (.glb) |
| **几何模型** | VGGT-Ω | VGGT/VGGT-omega/VGGT4D |
| **分割** | SAM 3 | SAM3 |
| **VLM** | Qwen3.6-27B | Qwen2.5/3.5-VL |
| **匹配** | VLM 跨扫描物体匹配 | 类内/跨类去重（mask IoU） |
| **位姿估计** | 逐实例 RANSAC + 投票 | ICP + Umeyama |
| **坐标系** | 重力对齐（高度分桶） | PCA 对齐 floor/wall |
| **训练** | 无 | 无 |

**核心差异**：PROSE 解决"两次扫描的对齐"，RAS 解决"单次扫描的重建"。但 PROSE 的**中间产物（物体级场景图）和匹配/对齐技术**与 RAS 的去重/对齐阶段高度同构，且**用的是同一套 foundation model**。

---

### 三、PROSE 参考价值分析（按优先级排序）

#### 3.1 最高价值：VLM 跨扫描物体匹配 → 增强 RAS 去重

**RAS 当前痛点**：
- Stage2 去重依赖 mask IoU（`self_category_deduplicate` / `cross_category_deduplicate`），纯几何
- 同一物体被 SAM3 跟踪断裂成多个实例时，IoU 去重失效（Q19/Q71）
- 无语义级别的物体对应

**PROSE 的解法**：
1. VLM 列举场景中的 landmark 物体（开放词汇）
2. SAM3 把名字转成时间一致的实例 mask
3. **VLM 跨扫描匹配物体实例**（Set-of-Marks 标注 crops）
4. **same/different 配对验证**过滤幻觉匹配

**可借鉴方案**：

```
当前 RAS 去重:
  mask IoU > 阈值 → 合并 (纯几何，跟踪断裂时失效)

改进方案:
  SAM3 分割 → VLM 列举物体 + 跨帧匹配 → same/different 验证
    → 即使 mask 不连续，只要 VLM 判定"同一物体"就合并
```

**可行性**：★★★★★。RAS 已有 Qwen VLM 和 SAM3，PROSE 的匹配逻辑可直接移植。Set-of-Marks 是标准的 VLM 提示技术，无需额外训练。

#### 3.2 最高价值：高度分桶 + 重力先验 → 改进 RAS 坐标对齐

**RAS 当前痛点**（Q67/Q74）：
- GeoCalib gravity 方向判断不稳
- floor/wall 对齐有时选错平面
- z 轴正负判断错误

**PROSE 的解法**：
- 沿重力轴把物体分到 K=5 个高度分桶（quantile bins）
- 天花板灯永不会和地板毯竞争（不同桶）
- 高度作为匹配先验，大幅缩小搜索空间

**可借鉴思路**：
- RAS 的坐标系对齐可借鉴"高度分桶"思路：先按高度分层（floor-level / table-level / shelf-level），再在层内对齐
- 这比当前的"全局 PCA"更鲁棒，因为物体高度是语义稳定的先验

#### 3.3 高价值：逐实例 RANSAC + 投票 → 替代全局 ICP

**RAS 当前痛点**（Q66/Q74）：
- Stage4 ICP 在点云噪声大时对齐差
- 全局 ICP 容易被噪声点拉偏

**PROSE 的解法**：
- 每个匹配物体对单独产生一个刚体变换假设（per-instance RANSAC）
- 用 FCGF/FPFH/GeoTrans 做描述子对应
- **场景级 inlier-ratio 投票**：选 inlier 比最高的假设
- "少数坏匹配不能污染最终变换"

**对比**：

| 方法 | 优势 | 劣势 |
|------|------|------|
| RAS Stage4 (全局 ICP) | 简单 | 噪声敏感，无语义 |
| PROSE (per-instance RANSAC + 投票) | 鲁棒，坏匹配不污染 | 需要先有物体对应 |
| FoundationPose (路线图) | 成熟 6DoF | 需额外模型 |

**可借鉴方案**：RAS 已有物体实例（SAM3 分割），可把 Stage4 从"全局 ICP"改为"逐实例 RANSAC + 投票"，用物体级对应替代全局点云对应。

#### 3.4 高价值：场景图作为可复用产物

**PROSE 的副产品**：配准过程中生成的物体级 3D 场景图可直接用于下游任务（路径规划等）。

**对 RAS 的意义**：RAS 当前 `relations_scene_graph.json` 已是场景图雏形。PROSE 的场景图构建流程（VLM 列举 + SAM3 mask + voxel-revote 融合）更系统，可参考其"voxel-revote fusion"步骤来合并多帧的物体观测。

#### 3.5 中等价值：RGB-only 鲁棒性证据

PROSE 在 VGGT-Ω 重建的（有噪声）点云上仍达 65.5% 配准召回（AEA），证明**即使几何退化，语义匹配仍鲁棒**。这给 RAS 的启示：在 VGGT 点云质量差时（Q6），应优先依赖语义（VLM）而非几何（ICP）。

#### 3.6 低价值：跨时间配准本身

PROSE 的"两次扫描配准"任务在 RAS 中不直接存在（RAS 是单视频）。但其技术组件（匹配/对齐/场景图）对 RAS 的去重和对齐阶段有迁移价值。

---

### 四、PROSE 不适用 / 需注意的部分

| PROSE 特性 | 不适用/注意原因 |
|-----------|-----------------|
| 两次扫描配准任务 | RAS 是单视频重建，无跨时间场景 |
| 代码未发布 | "Code (soon)"，无法直接复用 |
| Aria 数据集特性 | ego 头戴相机，RAS 视频可能更接近手持漫游 |
| 仅刚体变换 | RAS 需要完整 6DoF + scale |

---

## 五、综合评估与建议

### 5.1 参考价值总评

| 论文 | 维度 | 参考价值 | 说明 |
|------|------|---------|------|
| **SimFoundry** | 物体去穿模 + 物理稳定性 | ★★★★★ | 直接解决 Stage5 痛点，最高价值 |
| SimFoundry | 超越 SAM3D 的重建质量 | ★★★★★ | 证明 SAM3D 可被超越，提供对比基线 |
| SimFoundry | 背景重建 (3DGS) | ★★★★☆ | 补齐 RAS 缺失环节 |
| SimFoundry | 模块化设计 | ★★★★☆ | 与 RAS Stage 设计思路一致 |
| SimFoundry | 铰接物体 | ★★★☆☆ | 能力扩展，非当前痛点 |
| SimFoundry | Sim2Real 闭环 | ★★☆☆☆ | 超出 RAS 范围 |
| **PROSE** | VLM 跨扫描物体匹配 | ★★★★★ | 同技术栈，直接迁移到去重 |
| PROSE | 高度分桶 + 重力先验 | ★★★★★ | 改进坐标系对齐鲁棒性 |
| PROSE | 逐实例 RANSAC + 投票 | ★★★★☆ | 替代全局 ICP，更鲁棒 |
| PROSE | 场景图复用 | ★★★★☆ | 增强 relations_scene_graph |
| PROSE | RGB-only 鲁棒性证据 | ★★★☆☆ | 启示：几何差时优先语义 |

### 5.2 推荐行动（按优先级）

1. **短期（代码未发布，先吸收思路）**：
   - 研究 SimFoundry E.4 的 depenetration 算法描述，在 RAS Stage5 物理引擎提案中（`stage5_physics_engine_proposal.md`）加入"法线方向推开 + floor 接触约束 + 重力稳定性验证"子模块
   - 借鉴 PROSE 的高度分桶，在 RAS Stage2 坐标对齐中增加"按高度分层后再对齐"的前置步骤
   - 借鉴 PROSE 的 Set-of-Marks + same/different 验证，在 RAS Stage2 去重中增加"VLM 语义匹配"分支，作为 mask IoU 的补充

2. **中期（关注代码发布）**：
   - 持续关注 SimFoundry 和 PROSE 的 GitHub 代码发布
   - 待 PROSE 代码发布后，移植其 VLM 匹配 + per-instance RANSAC 到 RAS Stage4，替代全局 ICP
   - 待 SimFoundry 代码发布后，研究其 SAM3D 调用细节（L.1.2 节）和 depenetration 实现

3. **长期（架构层面）**：
   - 评估 SimFoundry 的 2D→3D 生成模型是否可替换 RAS 的 SAM3D
   - 评估引入 3DGS 背景重建的可行性（需权衡计算成本）
   - 若 RAS 未来需要支持铰接物体，SimFoundry E.3 是现成参考

### 5.3 与其他参考项目的对比

| 项目 | 核心参考价值 | 与本文两篇的关系 |
|------|------------|-----------------|
| **Do as I Do** (Q59) | 6-DoF 跟踪 + retargeting + 凸分解 | 6DoF 跟踪更先进 |
| **ForeHOI** (Q60) | 2D 遮罩修复 + 手部感知编码 | 手部遮挡专项 |
| **FoundationPose** (路线图) | 6-DoF 物体位姿跟踪 | 比 SimFoundry 的对齐更成熟 |
| **SimFoundry** (本文) | 去穿模 + 重建质量 + 背景重建 | 场景级最完整，与 RAS 流水线最对应 |
| **PROSE** (本文) | VLM 匹配 + 高度分桶 + 鲁棒对齐 | 技术栈完全一致，可迁移性最高 |

**结论**：
- **SimFoundry** 是目前与 RAS 流水线**结构最对应**的参考项目——它在 RAS 的每个 Stage 都有对应或更强的实现，并补齐了去穿模、背景、铰接、sim2real 等缺失环节。其最大直接价值是 **E.4 去穿模 + 物理稳定性**，可直接喂给 RAS Stage5 物理引擎提案。
- **PROSE** 是目前与 RAS **技术栈最一致**的参考项目——同样用 VGGT-Ω + SAM3 + Qwen VLM，其 VLM 匹配 / 高度分桶 / per-instance RANSAC 投票技术可直接迁移到 RAS 的去重和对齐阶段，迁移成本最低。
- 两篇代码均未发布，当前阶段以**吸收思路**为主；待代码发布后，PROSE 的迁移优先级最高（因为技术栈完全一致），SimFoundry 的 depenetration 模块次之。

建议将 SimFoundry 定位为**场景级流水线的完整对标**，将 PROSE 定位为**去重/对齐阶段的技术升级源**。

---

## 六、技术细节备忘

### 6.1 SimFoundry 流水线（论文 §4 + 附录 E/F）

```
输入: 单个 RGB 视频
  │
  ├── 感知模块 (E.1-E.2)
  │     ├── 代表性帧选择 (E.1)
  │     ├── Foundation model: 深度/位姿/分割 (E.2)
  │     └── 物体级 mask + 深度 + 位姿提取
  │
  ├── 资产生成模块
  │     ├── 2D→3D 生成模型 (含 SAM3D 对比, L.1.2)
  │     ├── 铰接物体生成 (E.3): 分割→关节→物理参数
  │     └── 物体去穿模 + 物理稳定性 (E.4) ★
  │
  ├── 背景重建模块 (E.5)
  │     ├── 前景擦除 + inpainting
  │     ├── metric 深度 + 位姿恢复
  │     ├── 深度监督的 3DGS 训练
  │     └── rigid bridge 到仿真世界
  │
  ├── 位姿对齐 + 物理标注
  │     ├── 自动对齐 + 3min/物体人工微调
  │     └── 物理参数标注 (质量/摩擦/阻尼)
  │
  ├── Digital Cousins 扩展 (F)
  │     ├── Object cousins (F.1)
  │     ├── Scene cousins (F.2)
  │     └── Task cousins (F.3)
  │
  └── 输出: sim-ready USD 场景 + Isaac Sim 评估
```

### 6.2 PROSE 流水线（论文 §3）

```
输入: 2× ego RGB 序列 (t0, t1)
  │
  ├── Scene Parsing (每序列独立)
  │     ├── VGGT-Ω → 深度 + 位姿 + 点云
  │     ├── VLM (Qwen3.6-27B) 列举 landmark 物体
  │     ├── SAM 3 → 名字转时间一致实例 mask
  │     └── voxel-revote fusion → 物体级 3D 场景图
  │
  ├── Height-Binned Correspondence
  │     ├── K=5 高度分桶 (沿重力轴 quantile)
  │     ├── 桶内 VLM 匹配 (Set-of-Marks crops)
  │     └── same/different 配对验证 (过滤幻觉)
  │
  ├── Pose Hypothesis & Voting
  │     ├── 每匹配对 → per-instance RANSAC (FCGF/FPFH/GeoTrans)
  │     ├── 每假设产生一个刚体变换
  │     └── 场景级 inlier-ratio 投票选最优 ★
  │
  └── 输出: 刚体变换 + 2× 物体级场景图
```

### 6.3 关键性能数据

**SimFoundry**:
- 12 场景 F1: 0.81–0.92 (zero-shot) → 0.93–0.99 (3min/物体微调)
- Real2Sim Pearson 相关性: 0.911 (vs PolaRiS 0.314)
- MMRV: 0.018 (vs PolaRiS 0.187)
- Sim2Real: object/scene/task cousins 提升 17%/21%/40%
- 重建对比 SAM3D: Chamfer 距离降低 ~2-3×，位姿误差降低 ~2×

**PROSE**:
- ADT GT 点云: RR 89.2% (vs TEASER++ 80.3%, SG-Reg 46.9%)
- ADT RGB 重建点云: RR 56.2% (vs TEASER++ 44.4%)
- AEA RGB 重建点云: RR 65.5% (vs TEASER++ 42.9%)
- 节点精度: 70.4% (vs SG-Reg 20.8%)
- 0 可学习参数，纯 training-free

### 6.4 代码发布状态（截至 2026-06-29）

| 论文 | 数据集 | 推理代码 | 训练代码 |
|------|--------|---------|---------|
| SimFoundry | 未明确发布 | 未发布 (Under Review) | 未发布 |
| PROSE | 用公开 ADT/AEA | 未发布 (Code soon) | 无需训练 |
