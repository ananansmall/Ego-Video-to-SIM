# 动态物体点云变化检测方案

## 背景

`mainv2.py` 的 Stage 3 (`run_stage3`) 中已经实现了动态物体判断，基于 SAM3 mask 跟踪的质心位移来区分动态/静态物体。
`object_tracking/` 管线也有 VGGT4D 的 `dynamic_mask` 机制。

**目标**: 在 `mainv2.py` 的流程中，增加一种基于**点云变化**的动态物体判断方法，作为现有 mask-centroid 方法的补充或替代。

---

## 现有动态判断方法回顾

### 方法1: `get_optimal_view_frame_id` (geometry_utils.py:759)
- **输入**: VGGT world_points `(T, H, W, 3)` + SAM3 instance_masks
- **逻辑**: 对每个实例，计算每帧的3D质心，相邻帧质心位移的 **median** > 0.10m → 动态
- **优点**: 简单直接，与现有 mask 跟踪天然集成
- **缺点**: 依赖 SAM3 mask 质量；VGGT world_points 有噪声，质心可能漂移

### 方法2: VGGT4D dynamic_mask (object_tracking/02_point_tracker.py)
- **输入**: VGGT4D 模型内部的 attention 聚类
- **逻辑**: 通过 TrackHead 的注意力图聚类检测运动区域
- **优点**: 端到端学习到的运动检测
- **缺点**: 需要 VGGT4D 模型，与 mainv2 的 VGGT/omega 不兼容

---

## 方案: 基于多帧点云差异的动态物体判断

### 核心思路

利用 VGGT 输出的 **world_points**（每帧的3D点云），对每个 SAM3 分割的实例，在连续帧之间计算**点云层面的变化量**，以此判断物体是否运动。

这与现有的"质心位移"方法不同之处在于：
- 质心位移只看"整体移动了多少"
- 点云变化看"物体表面点在哪里发生了变化"

### 数据流

```
mainv2.py Stage 2 输出:
  vggt_prediction_results['world_points']  → (T, H, W, 3) 每帧3D点云
  vggt_prediction_results['world_points_conf'] → (T, H, W) 点云置信度
  deduplicated_all_masks                   → {category: [instance_masks]}

↓

新增: compute_pointcloud_change(world_points, instance_masks, conf)
  → per_instance: {is_dynamic, change_metrics}

↓

mainv2.py Stage 3 的 run_stage3 中使用
```

### 具体算法

#### 1. 逐实例点云差分

对每个实例 `(category, instance_idx)`，有 `N` 个可见帧的 mask：

```python
for each instance:
    visible_frames = sorted(instance_masks, key=lambda m: m['frame_id'])
    
    # 对每对相邻帧 (t, t+1):
    for i in range(len(visible_frames) - 1):
        frame_a = visible_frames[i]
        frame_b = visible_frames[i+1]
        
        pts_a = world_points[frame_a.frame_id][frame_a.mask > 0]  # 帧a中物体表面的点
        pts_b = world_points[frame_b.frame_id][frame_b.mask > 0]  # 帧b中物体表面的点
        
        # 计算点云变化指标
        metrics = compute_pointcloud_metrics(pts_a, pts_b, ...)
```

#### 2. 点云变化指标设计

建议计算以下指标（全部基于欧氏距离，单位米）：

| 指标 | 公式 | 含义 | 动态阈值建议 |
|------|------|------|-------------|
| **Hausdorff距离** | `max(min_dist_a_to_b, min_dist_b_to_a)` | 最远点对距离 | > 0.05m |
| **Chamfer距离** | `mean(min_dist_a_to_b) + mean(min_dist_b_to_a)` | 平均双向距离 | > 0.02m |
| **质心位移** | `‖centroid_a - centroid_b‖` | 整体位移 | > 0.02m |
| **体积变化** | `‖pts_a‖ - ‖pts_b‖` (归一化) | 表观大小变化 | > 10% |
| **点密度变化** | 新出现的点数 / 总点数 | 场景变化率 | > 20% |

#### 3. 时序聚合

单对帧的差异不够鲁棒，需要在整个视频序列上聚合：

```python
# 对所有相邻帧对计算指标，然后聚合
median_hausdorff = median(all_hausdorff_distances)
mean_chamfer = mean(all_chamfer_distances)
max_chamfer = max(all_chamfer_distances)

# 判定逻辑
is_dynamic = (
    median_hausdorff > hausdorff_threshold or    # 有明显的极端运动
    mean_chamfer > chamfer_threshold or           # 平均变化显著
    (num_valid_frames > 4 and mean_chamfer > 0.01)  # 长时间段的小位移累积
)
```

#### 4. 噪声处理

VGGT world_points 有噪声，需要：
- **置信度过滤**: 只使用 `world_points_conf > percentile(90)` 的点
- **NaN 处理**: 忽略 `np.isnan(pts).any(axis=1)` 的点
- **KNN平滑**: 对每个点，用其K近邻的中值代替原始坐标

---

## 集成位置

### 方案A: 在 `run_stage3` 中新增函数 (推荐)

在 `mainv2.py` 的 `run_stage3` 函数开头，`get_optimal_view_frame_id` 调用之前，插入新的动态判断函数：

```python
# 在 run_stage3 中，替换/增强现有的 get_optimal_view_frame_id 动态判断
for category, category_masks in deduplicated_all_masks.items():
    for inst_idx, instance_masks in enumerate(category_masks):
        # 新方法: 点云变化检测
        is_dynamic_pc, pc_metrics = detect_dynamic_by_pointcloud(
            vggt_prediction_results['world_points'],
            vggt_prediction_results['world_points_conf'],
            instance_masks,
            motion_threshold=0.03,  # Chamfer median 阈值
        )
        
        # 旧方法: 质心位移检测 (作为交叉验证)
        optimal_fid, is_dynamic_centroid, centroid_info = get_optimal_view_frame_id(
            vggt_prediction_results['world_points'], instance_masks
        )
        
        # 综合判定: 任一方法判为动态 → 动态
        is_dynamic = is_dynamic_pc or is_dynamic_centroid
```

### 方案B: 作为独立模块放入 `object_tracking/`

创建 `00_dynamic_detector.py`，提供通用接口：

```python
def detect_dynamic_objects(
    world_points: np.ndarray,      # (T, H, W, 3)
    world_points_conf: np.ndarray, # (T, H, W) or None
    instance_masks: list,          # [{'frame_id': int, 'mask': ndarray}]
    method: str = "pointcloud",    # "pointcloud" | "centroid" | "hybrid"
    chamfer_threshold: float = 0.03,
    hausdorff_threshold: float = 0.05,
) -> Tuple[bool, dict]:
    """返回 (is_dynamic, metrics_dict)"""
```

---

## 实现步骤

### Step 1: 核心检测函数

新建文件 `src/dynamic_detector.py` (或 `object_tracking/00_dynamic_detector.py`):

```python
def compute_pointcloud_change_metrics(pts_a, pts_b, knn_k=5):
    """计算两组点云之间的变化指标
    
    Args:
        pts_a: (N, 3) 帧a中的物体点
        pts_b: (M, 3) 帧b中的物体点
        knn_k: KNN平滑参数
    
    Returns:
        dict: {
            'hausdorff': float,
            'chamfer': float,
            'centroid_disp': float,
            'new_point_ratio': float,
            'volume_ratio': float,
        }
    """
```

### Step 2: 时序聚合

```python
def aggregate_dynamic_signals(per_frame_metrics, num_frames, chamfer_thresh=0.03):
    """将逐帧指标聚合为最终判定
    
    返回 (is_dynamic: bool, summary_metrics: dict)
    """
```

### Step 3: 集成到 mainv2.py

修改 `run_stage3` 函数，在现有的 `get_optimal_view_frame_id` 调用处，加入点云变化检测。

### Step 4: 输出日志

在 Stage 3 的输出中，增加点云变化指标的打印：

```
   chair_0: [STATIC] median_disp=0.001m, chamfer_median=0.002m, hausdorff_max=0.008m
   cup_0:   [DYNAMIC] median_disp=0.15m, chamfer_median=0.045m, hausdorff_max=0.12m
```

---

## 关键参数建议

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chamfer_threshold` | 0.03m | 中位 Chamfer 距离阈值 |
| `hausdorff_threshold` | 0.05m | 最大 Hausdorff 距离阈值 |
| `knn_smooth` | 5 | KNN平滑邻居数 |
| `conf_percentile` | 90 | 最低置信度百分位 |
| `min_valid_frames` | 2 | 至少需要多少帧才能判定 |

---

## 与现有 VGGT4D dynamic_mask 的关系

| 维度 | VGGT4D dynamic_mask | 本方案 (点云变化) |
|------|---------------------|-------------------|
| 数据来源 | 模型内部注意力 | VGGT world_points |
| 粒度 | 像素级掩码 | 实例级点云 |
| 适用模型 | 仅 VGGT4D | VGGT / omega / 4D |
| 计算开销 | 零 (推理时附带) | 中等 (KNN距离计算) |
| 鲁棒性 | 学习到的，可能过拟合 | 几何驱动，可解释 |

**两者可以互补**: VGGT4D 的 dynamic_mask 作为先验，点云变化作为几何验证。

---

## 风险评估

1. **VGGT world_points 噪声**: 这是最大的不确定性来源。
   - 缓解: 置信度过滤 + KNN平滑 + 中值聚合
   
2. **SAM3 mask 漂移**: 如果同一物体在不同帧被分成不同实例，点云对比失效。
   - 缓解: 只在 `num_valid_frames >= 2` 时计算；mask 一致性检查

3. **计算性能**: KNN距离计算复杂度 O(N×M)。
   - 缓解: 只采样 subset (每帧最多2000点)；使用 `scipy.spatial.cKDTree`

---

## 下一步

确认方案后，我将:
1. 创建 `src/dynamic_detector.py` 或修改 `object_tracking/` 中的模块
2. 在 `mainv2.py` 的 `run_stage3` 中集成
3. 验证输出格式与现有 Stage 3 动态判断一致
