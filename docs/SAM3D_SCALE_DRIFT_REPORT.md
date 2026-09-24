# SAM3D 单图输出尺度漂移问题报告

> 生成时间: 2026-08-02
> 触发调查的场景: `2chair-bottle_vggt_omega` 两把相同椅子 mesh 尺寸差 1.5 倍
> 诊断脚本: `tools/test_pointcloud_vs_mesh_size.py`（单场景详细版）、`tools/diagnose_scenes.py`（多场景批处理版）

---

## 一、问题现象

在 `2chair-bottle_vggt_omega` 场景中，两把视觉上完全相同的椅子，最终生成的 mesh 尺寸相差约 **1.5 倍**：

| 轴 | chair_0 mesh | chair_1 mesh | 比值 |
|---|---|---|---|
| x | 0.856 m | 0.983 m | 1.15 |
| y | 0.783 m | 0.895 m | 1.14 |
| z | 1.487 m | 2.125 m | **1.43** |

但两把椅子的 **VGGT 点云实测尺寸几乎完全一致**（z 向都是 0.97 m 左右），说明问题不是点云/分割/选帧的错，而是 mesh 生成阶段引入的。

---

## 二、根因：SAM3D 的 scale 归一化机制在单图模式下不可靠

### 2.1 触发位置

`meta/sam3d_objects/data/dataset/tdfy/img_and_mask_transforms.py` 中 `ObjectCentricSSI._compute_scale_and_shift`：

```
原点(shift) ← 只用物体 mask 内的点（局部）          ← 正确
标尺(scale) ← 整张图在相机坐标系下的点云范围        ← 问题来源
```

具体流程：

1. `shift = mask_points.nanmedian(dim=-1)` — 取物体中心，合理
2. 当 `use_scene_scale == True`（pipeline.yaml 出厂配置）：
   ```python
   points_centered = pointmap_flat - shift.unsqueeze(-1)
   max_dims = points_centered.abs().max(dim=0).values
   scale = max_dims.nanmedian(dim=-1).values    # ← 用整张图最远背景当尺子
   ```
3. 这把"尺子"的物理意义是：**从物体中心到本帧可见最远背景的距离**，取决于相机角度和物体在房间中的位置

### 2.2 误差叠加机制

以 `2chair-bottle` 为例：

| 环节 | chair_0 @ frame39 | chair_1 @ frame71 |
|---|---|---|
| 物体本身 | 同一把椅子 | 同一把椅子 |
| 相机距离 | 0.66 m | 0.68 m |
| 整图 scene_scale | 0.699 | 0.824（大 **18%**）|
| 模型在归一化空间看到物体 | 占环境 72% | 占环境 61% |
| 模型 exp(raw_scale) 补偿 | 1.00 | 1.28（补偿过头 28%）|
| **最终 mesh/点云** | **1.53x** | **2.11x** |

两个因子叠加：**1.18 × 1.28 = 1.51**。

### 2.3 为什么不能关掉 `use_scene_scale`

`use_scene_scale: True` 是随 checkpoint 发布的训练约定，模型就在这个归一化下训练出来的。推理时改成 `False` 会破坏输入分布，预测直接崩溃。

---

## 三、是否所有物体都被放大？（多场景批量诊断）

用 `tools/diagnose_scenes.py` 对 4 个场景（含 2chair-bottle）全部 30 个实例跑了一遍，结论：

### 3.1 SAM3D 是逐实例随机漂移，不是全局放大

| 场景 | 静止/微动实例的 mesh/点云 |
|---|---|
| `2chair-bottle` | chair_0 1.53x、chair_1 2.11x、table_0 1.62x、bottle_0 **1.0x（正常）** |
| `121_C5_CellPhone_161deg` | door_0 1.43x、table_0 1.76x、table_1 0.86x、bowl_1 0.87x、mug_1 0.96x |
| `116_C5_CellPhone_185deg` | cabinet_0 1.07x、cabinet_4 0.93x（都正常）；cabinet_1 1.59x、cabinet_3 1.55x、chair_0 1.52x |
| `7_vggt_omega` | 大部分 mesh/点云 < 0.5x（但多数物体被大幅移动，点云被轨迹污染，不具参考价值）|

**模式：** 有时 SAM3D 能猜准（ratio ≈ 1.0），有时偏 1.5~1.8x，偏大偏小的方向都不固定，但**偏离量级稳定在 1.5x 左右**。

### 3.2 VGGT 世界坐标系本身是可信任的

`2chair-bottle` 场景只用点云做的物理先验校验：

| 物体 | 点云实测 | 真实世界常见范围 | 判定 |
|---|---|---|---|
| chair_0 高度 | 0.969 m | 0.85~1.05 m | 匹配 |
| chair_1 高度 | 0.964 m | 0.85~1.05 m | 匹配 |
| bottle 单帧高度 | ~0.19 m | 0.15~0.30 m | 匹配 |
| 桌面离地高度 | 0.857 m | 0.70~0.78 m | 接近（桌面有厚边）|

如果坐标系被整体缩放，这些值会同倍偏离常识——但它们都落在合理范围。**说明 VGGT 坐标系是可靠的"公尺"。**

---

## 四、另一个附带发现：被移动物体的点云测量陷阱

`2chair-bottle` 的 bottle 质心轨迹显示：

```
frame 0-20:  y ≈ +0.80   ← 起始位置
frame 40-60: y ≈ +0.28 → +0.19  ← 被人拿起移动
frame 70-90: y ≈ -0.27   ← 终点位置
```

跨全视频联合 mask 测尺寸，会把**运动轨迹的范围**也算进去，得到一个假大尺寸（y 向 1.4 m）。

这个问题在其他场景同样普遍：

| 场景 | 被移动实例 | 质心位移 |
|---|---|---|
| `121_C5_CellPhone_161deg` | bottle/bowl/mug/box/scissors/door/table | 0.3~1.06 m |
| `7_vggt_omega` | block/toy/plate | 0.56~0.72 m |
| `116_C5_CellPhone_185deg` | bottle/cabinet/chair | 0.17~0.94 m |

**这不是 bug，是数据性质**——视频中大部分物体都被操作过。任何需要"真实尺寸"的地方（包括下面要说的校准方案），都**必须只用物体静止段的帧**测点云，不能用全视频联合 mask。

---

## 五、修复方案：post-hoc 点云校准

### 5.1 方案选择

| 方案 | 做法 | 问题 |
|---|---|---|
| A. 同类实例取中位 scale 统一 | 假设同类必等大 | 遇到大小不一的同类别会出错 |
| **B. 用各自点云实测 rescale** ← 推荐 | 每个实例校准到它自己的 VGGT 点云 | 需处理移动物体的静止段检测 |
| C. 改 SAM3D 配置 | `use_scene_scale=False` | 会破坏训练约定 |

### 5.2 推荐方案（B）的实现要点

1. **取实例在视频中的 mask 序列**，计算每帧的质心
2. **按质心做 K-Means（K=3）聚类**，取点数最多的簇作为"静止段"
3. **对静止段帧的 mask 点云取 [0.5%, 99.5%] 分位**，得到稳健的物体真实尺寸
4. **对 mesh 应用 `T @ scale_matrix`**（绕 mesh 质心等比缩放），使得 mesh 在世界坐标系的尺寸等于点云尺寸
5. 校准因子 `k = median(点云尺寸 / mesh 当前尺寸)`，作用于 T 的三列缩放分量

### 5.3 实现位置

- 新增脚本 `tools/rescale_meshes_by_pointcloud.py`（独立，可重复跑，不侵入主流程）
- 读取 `output_v2/<scene>/_pre_dedup_masks.pkl` 的 `world_points` + `all_masks`
- 读取 `output_v2/<scene>/all_instances.pkl` 的 `T`
- 写入 `output_v2/<scene>/all_instances_rescaled.pkl`
- 下游 stage3+ 的实例加载处改为优先读 rescaled 版

> **实现状态（2026-09-13）**: 脚本已实现并在 `2chair-bottle_vggt_omega` 验证通过
> （chair_0 k=0.646 / chair_1 k=0.473 / table_0 k=0.618 / bottle_0 k=1.062，
> 校准后 z 向 ratio 全部 ≈1.00，前后 GLB 预览确认两把椅子尺寸一致）。
> 相对上文的原始配方有三处实测驱动的偏差，详见脚本 docstring：
> ① "点数最多的簇"改为"质心最紧凑簇 + 移动判定"（最大簇会选中运动段/入画碎片段）；
> ② 移动物体用静止段逐帧 ext 中位数而非 min/max（VGGT 帧间漂移会污染 min/max，bottle 被放大 6 倍）；
> ③ 移动物体 k 只取 mesh 最长轴比例（透明瓶身深度噪声污染水平向测距，三轴中位数失效）。
> "下游优先读 rescaled 版"已于 2026-09-13 以更优方式解决：校准调用直接插入 mainv3.py 主流程 Stage3 之后、基础精修之前（第 1385-1394 行，try/except 兜底），使 all_instances.pkl 本身即校准后数据，post_pipeline/Stage4/5 无需改动；mainv2.py 未同步（需要时照搬同一插入点）。

---

## 六、诊断工具

| 文件 | 用途 |
|---|---|
| `tools/test_pointcloud_vs_mesh_size.py` | 单场景详细对比（含物理先验校验、校准因子输出）|
| `tools/diagnose_scenes.py` | 多场景批量扫描，自动标注 ⚠️MESH_BIG/⚠️MESH_SMALL/⚠️MOVED |

用法：

```bash
# 单场景详细
python tools/test_pointcloud_vs_mesh_size.py [场景目录]
# 默认 2chair-bottle_vggt_omega

# 多场景批量
python tools/diagnose_scenes.py 121_C5_CellPhone_161deg_vggt_omega 7_vggt_omega 116_C5_CellPhone_185deg_vggt_omega
```

---

## 七、尚未解决

1. **Stage 5 SP 精修** 因 `name 'supported_mesh' is not defined` 报错被跳过（两个场景都出现），会进一步污染 mesh 尺寸，与本报告无关但需独立修复
2. **坐标对齐** 的 relaxed 模式引入的 T 微差，本报告未深入
3. **Stage 5 Phase1 地面对齐** 计算偏差（`2chair-bottle` chair_1 被多推 39cm），与本报告无关但需独立修复
