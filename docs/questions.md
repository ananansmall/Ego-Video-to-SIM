# Questions & Answers

## Q: GLB 物体坐标轴不对，怎么解决？

**日期**: 2026-07-25
**分类**: 调试

### 问题
`render_quick.py` 渲染时，GLB 物体的坐标轴变化错了，导致物体位置显示不正确。之前加了场景图变换后坐标系反而更差了。

### 解答
GLB 文件（glTF 标准）使用 Y-up 坐标系，但 RAS 外参（extrinsics）使用 Z-up 坐标系。关键发现：

1. 检查了 `view_all_glb.py` 的加载方式，发现它对每个 mesh 应用了场景图变换矩阵
2. 但实际 121 和 hoi4d 数据集的 GLB 文件**没有场景图节点变换**（所有 geometry 都是 `local only, no transform`）
3. 说明 GLB 在导出时已经将坐标转换到了 RAS 世界坐标系（Z-up）中，不需要再额外乘以变换矩阵

**修复方法**：使用 geometry 本地坐标直接显示，不应用场景图变换。

**验证**：121 数据集中 bottle 距相机 0.43m（合理），Frame 0 对齐误差 0.000000m。

---

## Q: 为什么 thumb 和 wrist 轨迹一样？

**日期**: 2026-07-25
**分类**: 调试

### 问题
之前 thumb 和 wrist 的轨迹显示一样，让人困惑。

### 解答
之前代码中读取 thumb 时用了 `d['pred_trans']`（即手腕位置），所以拇指和手腕轨迹完全重合。修复后通过 MANO FK 计算 21 个关节点，拇指尖 = joint 20，手腕 = joint 0，两者轨迹不同且合理。

---

## Q: vis_mano_mapping_variants.py 和 render_quick.py 各渲染了什么，有什么区别？

**日期**: 2026-09-12
**分类**: 架构

### 问题
`trajectory_optimization/View/vis_mano_mapping_variants.py` 和 `View/render_quick.py` 两个可视化脚本各自渲染了什么内容，区别是什么？

### 解答

**render_quick.py = 对齐质量检查 viewer**：把 HaWoR 重建（raw + depth 对齐两版）叠加到 RAS GLB 世界，验证手部轨迹与场景对齐。渲染内容：
- GLB 物体只画中心点 + 包围盒（原始顶点 = Z-up 世界坐标，不乘节点变换）
- 轨迹曲线：HaWoR/RAS 相机、左右手 Midpoint（拇指尖 j20+食指尖 j16 中点，Raw vs Depth）、RAS MANO 重建中点、可选 depth center / 手腕深度反投影 / 偏移轨迹
- 手部链：数据驱动（FK 关节 → 逐帧 R_c2w → transform_params 相似变换），明确不用固定 Rx_hand
- 输出 `*_overlay.png` + Plotly `*_overlay.html`，视野固定 [-2,2]³

**vis_mano_mapping_variants.py = MANO→GLB 映射链 A/B 对照工具**：2×2 变体矩阵（A: 节点世界变换+Rx_hand；B: 节点世界变换无 Rx_hand；C: 原始本地坐标+Rx_hand（当前管线链）；D: 原始本地坐标无 Rx_hand），每变体一张图：
- GLB 物体以真实面片网格渲染（独立纯色 + Lambert 着色），物体坐标处理是实验变量之一
- 完整 21 关节手部骨架（6 帧采样、时间渐变色）+ 手腕/掌心轨迹
- 客观指标：掌心轨迹到最近物体中心距离（median/min），越小越接近接触
- 输出 4 张单变体 PNG + 2×2 总览拼图，视野自动适配

**核心区别**：render_quick 只画"轨迹+包围盒"验证现有对齐；vis_mano_mapping_variants 画面片网格+完整骨架，把"物体坐标处理"和"是否乘 Rx_hand"作为实验变量供选择映射链。

**注意**：变体 A 标注 `[render_quick chain]` 带 Rx_hand，但 render_quick.py 当前代码（L159-L162）已明确不用固定 Rx_hand——该标签对应 render_quick 旧版链路（R_c2w[0]≈diag(1,-1,-1) 的第一帧特例近似），看图时勿被误导。

---

## Q: vis_preprocess_trajectories.py 渲染的是夹爪中点还是手腕？为什么看不到平滑前轨迹？

**日期**: 2026-09-12
**分类**: 概念 / 调试

### 问题
vis_preprocess_trajectories.py 渲染的是夹爪中点还是手腕？"MANO wrist raw (平滑前)" 橙色虚线在图上看不到。

### 解答
- 渲染的是**手腕级别**：橙线 = MANO 腕点 j0（raw 虚线/smooth 实线），黑线 = 夹爪 base 原点。夹爪中点只在 F50 一帧算一个验证点（绿 X），不是轨迹。
- 平滑前轨迹**已画出但被盖住**：实测 stage0_check 数据 raw vs smooth 腕点距离 mean=6.8mm / max=57mm，在全场景 ~1m 视野下仅 1-2 像素，橙虚线藏在橙实线下方。非 bug，是平滑参数（max_vel=0.020/max_acc=0.005）改动本就很小。要看清需局部放大图。

---

## Q: 对比图右侧乱糟糟的彩色长线是什么？

**日期**: 2026-09-12
**分类**: 调试

### 问题
/tmp/ras_mano_vs_stage0.png 右图出现大量横穿画面的彩色长线，完全看不懂。

### 解答
matplotlib 老版 mplot3d **不裁剪超出坐标范围的线条**：set_xlim3d 后视野外的轨迹段仍被画出来横穿画面。已改为 2D 投影特写（俯视 XY + 侧视 XZ，2D 轴天然裁剪）重出图。内容解读：RAS-depth 拇食中点在物体上方 z≈0.05-0.12 徘徊不落下，且有一段穿到桌面下（z<0）；stage0 黑线精准落到 glb_4 中心（XZ 图黑线从 0.15 下探到 0.02）。

---
