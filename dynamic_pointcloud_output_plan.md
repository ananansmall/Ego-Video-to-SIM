# 动态物体点云轨迹输出方案

## 目标

在 `mainv2.py` 的 Stage 3 中，对每个被识别为**动态**的物体，将其在各帧的 3D 点云提取并保存到独立文件夹中，形成可视化的点云运动轨迹。

## 输出目录结构

```
output_v2/{scene}/{vggt_model}/
├── dynamic_pointclouds/          ← 新增目录
│   ├── cup_inst0/               ← 每个动态物体一个子目录
│   │   ├── frame_0012.ply       ← 第12帧的点云
│   │   ├── frame_0013.ply
│   │   ├── frame_0014.ply
│   │   ├── frame_0015.ply
│   │   ├── trajectory.npz       ← 质心轨迹 + 运动方向
│   │   └── summary.json         ← 统计信息
│   ├── ball_inst1/
│   │   ├── frame_0005.ply
│   │   ├── frame_0006.ply
│   │   └── ...
│   └── _all_dynamics.json       ← 所有动态物体的索引
└── ...
```

## 实现位置

在 `mainv2.py` 的 `run_stage3()` 函数中，动态物体处理逻辑之后（约第 593-627 行），插入点云提取和保存。

## 核心逻辑

```python
# 在 run_stage3 中，对每个动态物体:
pc_dir = os.path.join(output_path, "dynamic_pointclouds", f"{category}_inst{inst_idx}")
os.makedirs(pc_dir, exist_ok=True)

for im in sorted_masks:
    fid = im['frame_id']
    mask = im['mask']
    pointmap = vggt_prediction_results['world_points'][fid]
    
    # 提取该帧中物体掩码区域内的点云
    pts = pointmap[mask > 0]
    pts = pts[np.isfinite(pts).all(axis=1)]  # 过滤 NaN
    
    if len(pts) > 0:
        # 保存为 PLY 文件
        export_pointcloud(pts, os.path.join(pc_dir, f"frame_{fid:04d}.ply"))
        
        # 记录质心
        centroids.append((fid, pts.mean(axis=0)))

# 计算运动方向
# 对相邻帧质心求差，得到运动矢量
# 保存到 trajectory.npz
```

## 每个物体的 summary.json 内容

```json
{
  "category": "cup",
  "instance_idx": 0,
  "is_dynamic": true,
  "motion_type": "translation",
  "total_frames": 12,
  "first_visible_frame": 5,
  "last_visible_frame": 16,
  "total_displacement": 0.342,
  "max_speed": 0.089,
  "centroid_trajectory": [[x1,y1,z1], [x2,y2,z2], ...],
  "motion_direction": [0.92, -0.15, 0.36]
}
```

## trajectory.npz 内容

```python
{
    "centroids": (N, 3)       每帧质心
    "frame_ids": (N,)         帧ID
    "velocities": (N, 3)      逐帧速度
    "directions": (N-1, 3)    相邻帧运动方向
    "valid_mask": (N,)        有效帧标记
}
```

## 与现有逻辑的关系

- **不影响** 现有的动态判断（median_disp > 0.02 的逻辑保持不变）
- **不影响** 现有的点云清理（动态物体其他帧点云置 NaN 的逻辑保持不变）
- **不影响** 现有的位置调整（mesh 从 optimal_frame 移到 first_visible_frame 的逻辑保持不变）
- 只是在确认物体动态后，**额外输出** 它的逐帧点云

## 性能影响

- 点云提取：每个实例每帧 O(H×W) 的掩码索引操作，极快
- PLY 写入：每个文件 ~KB 级别，I/O 可接受
- 磁盘占用：假设 10 个动态物体，每物体 20 帧，每帧 1000 点 → 约 50MB
- 时间开销：< 5 秒（对于整个 Stage 3 的几分钟来说可忽略）
