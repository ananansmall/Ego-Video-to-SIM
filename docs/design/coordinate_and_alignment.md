# ReplicateAnyScene + HaWoR 深度剖析与对齐方案

> 合并自 Coordinate_System_Alignment_Guide.md (§1-5 + 附录A-C) + GLB_to_HaWoR_Alignment_Plan.md (§6-12 + 附录D)

## 目录

**Part I — 坐标系对齐指南**
1. [ReplicateAnyScene 逐文件深度剖析](#1-replicateanyscene-逐文件深度剖析)
2. [HaWoR 逐文件深度剖析](#2-hawor-逐文件深度剖析)
3. [两系统对应关系与帧间关系](#3-两系统对应关系与帧间关系)
4. [对齐方案：GLB → HaWoR](#4-对齐方案glb--hawor)
5. [对齐正确性保证](#5-对齐正确性保证)

**Part II — GLB → HaWoR 对齐操作指南**
6. [实际文件路径与数据概况](#6-实际文件路径与数据概况)
7. [目标](#7-目标)
8. [对齐原理](#8-对齐原理)
9. [操作步骤（在终端逐段执行）](#9-操作步骤在终端逐段执行)
10. [对齐正确性分析](#10-对齐正确性分析)
11. [后续：在 HaWoR 渲染器中查看](#11-后续在-hawor-渲染器中查看)
12. [快速验证命令](#12-快速验证命令)

---

## 1. ReplicateAnyScene 逐文件深度剖析

### 1.1 管线总览

```
输入: mp4 视频 + category.json
  │
  ├─ VGGT 模型 ──→ extrinsics, depths, world_points, intrinsic
  │                    │
  │                    ├─ 保存: extrinsics/*.txt, depth/*.png, color/*.jpg, intrinsic.txt, point_cloud.ply
  │                    │
  │                    └─ SAM3 分割 ──→ wall_masks, floor_masks
  │                                        │
  │                                        └─ 房间对齐 ──→ R, t
  │                                              │
  │                                              └─ 更新 extrinsics, world_points
  │
  ├─ SAM3 Video 分割 ──→ instance_masks
  │
  ├─ 去重 ──→ deduplicated_masks
  │
  ├─ SAM3D 生成 ──→ 3D 资产 (mesh + T)
  │
  ├─ 精修 ──→ refined instances
  │
  └─ 导出 ──→ final_scene.glb
```

### 1.2 每个输出文件的坐标系剖析

#### `extrinsics/0.txt` ~ `19.txt`

**产生过程**：

```
Step 1: VGGT 模型输出 pose_enc (9维: 3平移 + 4四元数 + 2FoV)
         ↓ pose_encoding_to_extri_intri()
Step 2: 解码为 extrinsic (S, 3, 4), 格式为 w2c (world-to-camera)
         ↓ vggt_predict.py 第 62-63 行: pad 到 (S, 4, 4)
Step 3: np.pad(predictions['extrinsic'], ((0,0),(0,1),(0,0))) + [3,3]=1
         ↓ 房间对齐
Step 4: align_vggt_predictions() 更新 extrinsics
         ↓ 保存
Step 5: np.savetxt() 保存每帧 4×4 矩阵
```

**关键代码** (`vggt_predict.py:62-63`):
```python
extrinsics = np.pad(predictions['extrinsic'], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
extrinsics[:, 3, 3] = 1
```

**关键代码** (`geometry_utils.py:292-298`):
```python
c2w_old = predictions["extrinsics"]   # 变量名 c2w_old，但实际是 w2c
R_c2w_old = c2w_old[:, :3, :3]
t_c2w_old = c2w_old[:, :3, 3]
R_c2w_new = R_c2w_old @ R.T
t_c2w_new = t_c2w_old - (R_c2w_new @ t)
```

**坐标系**: Room World (z-up, 米)
**格式**: w2c 4×4 齐次矩阵
**相机约定**: OpenCV (x-right, y-down, z-forward)
**提取相机位置**: `cam_pos = -R_w2c^T @ t_w2c`

**帧间关系**: 20 帧均匀采样自原始视频（`--max_frames 160`，实际用了 20 帧），帧索引 0-19 连续。

**实测数据**:
```
Frame 0:  cam_pos ≈ [0, 0, 0],  ||R_w2c - I|| = 0.0005  ← 第一帧 ≈ 单位矩阵
Frame 1:  cam_pos = [-0.0014, 0.0004, -0.0006]
Frame 10: cam_pos = [0.0421, 0.0098, -0.0074]
Frame 19: cam_pos = [0.0679, -0.0092, 0.0403]
```

#### `intrinsic.txt`

**产生过程**:

```python
# vggt_predict.py 第 66 行
intrinsic = np.mean(predictions['intrinsic'], axis=0)
```

VGGT 为每帧估计一个内参矩阵，取所有帧的平均值。

**格式**: 3×3 针孔模型矩阵
**坐标系**: 像素坐标系 (u-right, v-down)

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

#### `depth/0.png` ~ `19.png`

**产生过程**:

```python
# vggt_predict.py 第 61 行
depths = predictions['depth'].squeeze(-1)  # (S, H, W)

# main.py 第 53 行
cv2.imwrite(..., (depth * 1000).astype(np.uint16))
```

**格式**: uint16 PNG，编码 `depth_mm = round(depth_meters * 1000)`
**解码**: `depth_meters = depth_uint16 / 1000.0`
**坐标系**: 相机坐标系 z 轴深度（即相机到场景点沿光轴的距离）
**帧间关系**: 与 extrinsics 一一对应

**实测数据**:
```
Frame 0: mean=0.8806m, median=0.8740m, range=[0.091, 1.471]
```

#### `color/0.jpg` ~ `19.jpg`

**产生过程**:

```python
# vggt_predict.py 第 60 行
colors = (predictions['images'].transpose(0, 2, 3, 1) * 255).astype(np.uint8)

# main.py 第 51 行
cv2.imwrite(..., cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
```

**格式**: BGR (OpenCV), uint8
**坐标系**: 像素坐标系 (u-right, v-down)

#### `point_cloud.ply`

**产生过程**:

```python
# vggt_predict.py 第 51-58 行
point_cloud_data = predictions_to_pcd(predictions, conf_thres=50.0, ...)

# geometry_utils.py 第 304 行 (对齐后更新)
predictions['point_cloud_data'].apply_transform(np.vstack([np.hstack([R, t.reshape(3,1)]), [0,0,0,1]]))
```

点云生成过程：
1. `unproject_depth_map_to_point_map()` 用深度图+外参+内参反投影每像素到世界坐标
2. 置信度过滤 (conf_thres=50.0，取前 50% 高置信度点)
3. 房间对齐后应用 R, t 变换

**坐标系**: Room World (z-up, 米)
**帧间关系**: 所有帧的点合并为一个点云

**实测数据**:
```
范围: x[-0.78, 1.33], y[-0.83, 0.35], z[0.01, 1.52]
中心: [0.04, -0.06, 0.84]
```

#### `final_scene.glb`

**产生过程**:

```python
# main.py 第 178-187 行
scene = trimesh.Scene()
for category, category_instances in all_instances.items():
    for i, instance_info in enumerate(category_instances):
        mesh = instance_info['original_mesh']
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(instance_info['T'])
        scene.add_geometry(transformed_mesh, node_name=f"{category}_{i}")
scene.apply_transform(np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]))  # z-up → y-up
scene.export("final_scene.glb")
```

**坐标系**: GLB 标准 (y-up)
**变换链**: SAM3D 局部(y-up) → Room World(z-up) → GLB(y-up)
**帧间关系**: 无（静态场景，不依赖帧）

**实测数据**:
```
范围 (y-up): x[-0.25, 0.39], y[0.44, 1.04], z[-0.24, -0.01]
```

### 1.3 相机产生方式详解

VGGT 的相机估计是**全局联合优化**：

```
输入: N 帧图像 (S, 3, H, W)
  ↓ VGGT Transformer
输出: pose_enc (S, 9)  ← 每帧 9 维编码
  ↓ pose_encoding_to_extri_intri()
  ├── extrinsic (S, 3, 4)  ← w2c，OpenCV 约定
  └── intrinsic (S, 3, 3)  ← 每帧独立内参
```

**特点**:
- 所有帧联合推理，帧间有全局一致性
- 深度和位姿同时估计，互相约束
- 无累积漂移（非增量式）
- 尺度由模型内部决定（声称米制，但可能有偏差）

### 1.4 房间对齐的完整机制与退化情况

#### 对齐过程详解

`align_to_room_coordinate_system()` 和 `align_vggt_predictions()` 的完整工作流程：

```
Step 1: SAM3 分割地板和墙面 → floor_masks, wall_masks
Step 2: PCA 拟合平面法线 → floor_normal, wall_normal
Step 3: 构建旋转矩阵 R = [wall_normal_1, wall_normal_2, floor_normal]
Step 4: 计算平移 t:
         t[2] = -rotated_floor_centroid[2]    ← 地板设为 z=0
         t[:2] = -scene_bbox_center[:2]       ← 场景中心设为 x-y=0
Step 5: 更新所有数据:
         world_points = world_points @ R.T + t
         extrinsics 旋转: R_c2w_new = R_c2w_old @ R.T
         extrinsics 平移: t_c2w_new = t_c2w_old - (R_c2w_new @ t)
         point_cloud: apply_transform([R|t])
```

**关键：相机位置如何变化**

对齐前，第一帧相机位置在 VGGT 原始世界坐标系中（任意位置）。
对齐后，相机位置被变换到 Room World：

```python
# align_vggt_predictions 中的变换
R_c2w_new = R_c2w_old @ R.T
t_c2w_new = t_c2w_old - (R_c2w_new @ t)

# 相机位置 (从 w2c 提取):
# 对齐前: cam_pos_old = -R_w2c_old^T @ t_w2c_old
# 对齐后: cam_pos_new = -R_w2c_new^T @ t_w2c_new
#
# 等价于: cam_pos_new = R @ cam_pos_old + t
# 即相机位置随世界坐标系一起旋转和平移
```

**实测数据验证**：
```
对齐后 cam_pos[0] ≈ [0, 0, 0]  ← 第一帧相机在场景中心附近
对齐后 cam_pos[19] ≈ [0.068, -0.009, 0.040]  ← 相机移动很小
点云 z 范围: [0.01, 1.52]  ← 地板在 z≈0，天花板在 z≈1.5m
```

原点不在第一帧相机位置，而在**地板平面的场景中心**。第一帧相机恰好在原点附近，是因为相机拍摄位置接近场景中心。

#### 退化情况：没有分割出 floor 和 wall 会怎样

`align_to_room_coordinate_system()` 中有两个关键退出点：

```python
# geometry_utils.py 第 240-241 行
if len(floor_plane_infos) == 0:
    return np.eye(3), np.zeros(3)  # ← 返回单位旋转 + 零平移

# geometry_utils.py 第 250-251 行
if len(orthogonal_wall_plane_infos) == 0:
    return np.eye(3), np.zeros(3)  # ← 返回单位旋转 + 零平移
```

**当 R=I, t=0 时，`align_vggt_predictions()` 的效果**：

```python
R_c2w_new = R_c2w_old @ I = R_c2w_old    # 旋转不变
t_c2w_new = t_c2w_old - (R_c2w_new @ 0) = t_c2w_old  # 平移不变
world_points = world_points @ I + 0 = world_points     # 点不变
```

**结论：如果没检测到地板或墙面，所有数据保持 VGGT 原始输出不变。**

| 情况 | R | t | 效果 |
|------|---|---|------|
| 地板和墙面都检测到 | 房间对齐旋转 | 地板z=0, 场景中心 | Room World (z-up) |
| 只检测到地板，没墙面 | I | 0 | VGGT 原始坐标系（任意） |
| 没检测到地板 | I | 0 | VGGT 原始坐标系（任意） |
| 地板法线方向不确定（<30°一致性） | I | 0 | VGGT 原始坐标系（任意） |

**对对齐的影响**：

如果 RAS 没有做房间对齐（R=I, t=0），那么 RAS 的世界坐标系就是 VGGT 的原始输出坐标系。此时：

1. **轴约定不再保证是 z-up**：VGGT 原始输出的朝上轴是任意的
2. **原点不再在地板中心**：原点在 VGGT 内部 SfM 决定的位置
3. **R_axis 不再适用**：因为 Room World 不再是 z-up，`[[1,0,0],[0,0,1],[0,-1,0]]` 的假设失效

**如何检测是否发生了退化**：

```python
# 加载 RAS extrinsics 后检查
ext_0 = np.loadtxt('extrinsics/0.txt')
R_w2c = ext_0[:3,:3]
cam_pos = -R_w2c.T @ ext_0[:3,3]

# 如果房间对齐成功:
#   cam_pos[2] 应该接近地板高度 (通常 1.0-1.7m，人眼高度)
#   R_w2c 应该接近某个合理朝向

# 如果房间对齐失败 (R=I, t=0):
#   cam_pos 在 VGGT 原始坐标系中 (可能远离原点)
#   R_w2c 可能朝向任意方向

# 更可靠的方法: 检查点云的 z 范围
pcd = trimesh.load('point_cloud.ply')
z_range = pcd.bounds[1,2] - pcd.bounds[0,2]
# 如果 z_range 合理 (2-4m, 典型房间高度) → 对齐成功
# 如果 z_range 异常 → 可能对齐失败
```

**退化时的对齐策略调整**：

如果确认 RAS 没有做房间对齐，需要修改对齐方案：

```
正常情况 (Room World z-up):
  R_axis = [[1,0,0],[0,0,1],[0,-1,0]]  ← 已知 z-up → y-down

退化情况 (VGGT 原始坐标系):
  R_axis 未知，需要从相机轨迹估计
  → 用 Umeyama 从相机轨迹估计完整的 R_total (含轴约定)
  → 不再分解为 R_residual @ R_axis
```

### 1.5 帧间关系

```
原始视频 (mp4)
  ↓ load_video_frames(max_frames=160)
  ↓ 均匀采样 20 帧
  ↓ 帧索引: 0, 1, 2, ..., 19
  ↓
  ├── extrinsics[0..19]  ← 每帧一个 w2c 矩阵
  ├── depths[0..19]      ← 每帧一个深度图
  ├── colors[0..19]      ← 每帧一张 RGB 图
  └── world_points[0..19] ← 每帧一个 (H,W,3) 点图

帧间约束:
  - VGGT 全局优化: 所有帧的位姿和深度联合估计
  - 房间对齐: R, t 统一应用于所有帧
  - 点云: 所有帧的 3D 点合并，置信度过滤
```

---

## 2. HaWoR 逐文件深度剖析

### 2.1 管线总览

```
输入: mp4 视频
  │
  ├─ detect_track_video ──→ extracted_images/, tracks_*/
  │
  ├─ hawor_motion_estimation ──→ cam_space/*.json (相机空间手部姿态)
  │                              │
  │                              └─ 生成 model_masks.npy (手部掩码)
  │
  ├─ hawor_slam ──→ SLAM/hawor_slam_w_scale_*.npz (SLAM 轨迹 + 尺度)
  │
  ├─ hawor_infiller ──→ world_space_res.pth (世界空间手部姿态)
  │                      │
  │                      └─ cam2world_convert: 相机空间 → 世界空间
  │
  ├─ demov2.py 后处理:
  │   ├── R_x 变换: OpenCV → OpenGL (仅用于渲染和保存)
  │   ├── 生成手部网格顶点
  │   ├── 渲染: vis_cam_*/, vis_world_*/, vis_verify/
  │   └─ 保存: reconstruction/hawor_results_*.npz
  │
  └─ 输出: hawor_results_*.npz (最终重建结果)
```

### 2.2 每个输出文件的坐标系剖析

#### `extracted_images/0000.jpg` ~ `0112.jpg`

**产生过程**: `detect_track_video()` 从 mp4 中提取所有帧

**格式**: BGR (OpenCV), uint8
**帧数**: 113 帧 (0-112)
**命名**: 4 位零填充

#### `tracks_0_113/model_masks.npy`

**产生过程**: `hawor_motion_estimation()` 中用 MANO 渲染手部掩码

```python
# hawor_video.py 第 208-215 行
cam_R = torch.eye(3).unsqueeze(0).cuda()
cam_T = torch.zeros(1, 3).cuda()
cameras, lights = renderer.create_camera_from_cv(cam_R, cam_T)
rend, mask = renderer.render_multiple(vertices_i.unsqueeze(0).cuda(), faces, ...)
model_masks[frame_ck[img_i]] += mask
```

**格式**: (N, H, W) bool 数组
**坐标系**: 像素坐标系 (u-right, v-down)
**用途**: 传给 DROID-SLAM 做手部区域掩码（避免 SLAM 跟踪手部）

#### `tracks_0_113/model_tracks.npy`, `model_boxes.npy`

**产生过程**: YOLO 检测 + 追踪

**格式**: 字典/数组，包含每帧手部检测框和追踪 ID
**坐标系**: 像素坐标系

#### `tracks_0_113/frame_chunks_all.npy`

**产生过程**: `parse_chunks()` 将连续检测到手部的帧分成块

**格式**: `{hand_id: [[frame_start, ..., frame_end], ...]}` (joblib)
**用途**: 指导 HAWOR 模型分块推理

#### `cam_space/0/0_112.json`

**产生过程**: HAWOR 模型在**相机坐标系**中推理手部姿态

```python
# hawor_video.py 第 163-169 行
results = model.inference(img_ck, boxes_ck, img_focal=img_focal, img_center=img_center, do_flip=do_flip)
data_out = {
    "init_root_orient": results["pred_rotmat"][None, :, 0],  # (B, T, 3, 3)
    "init_hand_pose": results["pred_rotmat"][None, :, 1:],   # (B, T, 15, 3, 3)
    "init_trans": results["pred_trans"][None, :, 0],          # (B, T, 3)
    "init_betas": results["pred_shape"][None, :]              # (B, T, 10)
}
```

**坐标系**: **相机坐标系** (OpenCV: x-right, y-down, z-forward)
**格式**: JSON，旋转矩阵 (3×3)
**关键**: `init_trans` 是手部根关节在**相机坐标系**中的位置

**帧间关系**: 按检测到的连续帧块存储，不是所有帧都有

#### `SLAM/hawor_slam_w_scale_0_113.npz`

**产生过程**:

```
Step 1: DROID-SLAM 增量式估计相机轨迹
         ↓ run_slam(imgfiles, masks=masks, calib=calib)
Step 2: 输出 traj (N, 7): [tx, ty, tz, qx, qy, qz, qw] ← c2w, 无尺度
         ↓
Step 3: Metric3D 估计每个关键帧的度量深度
         ↓
Step 4: est_scale_hybrid() 比较 SLAM 逆深度 vs Metric3D 深度
         ↓
Step 5: 取中位数 median_s 作为全局尺度
         ↓
Step 6: 保存 traj (未缩放), scale, tstamp, disps
```

**关键代码** (`hawor_slam.py:130-131`):
```python
median_s = np.median(scales_)
np.savez(save_path, tstamp=tstamp, disps=disps, traj=traj,
         img_focal=focal, img_center=calib[-2:], scale=median_s)
```

**坐标系**: SLAM World (OpenCV: y-down, z-forward)，**但 traj 未缩放**
**格式**: npz

| Key | 形状 | 坐标系 | 说明 |
|-----|------|--------|------|
| `traj` | (N, 7) | SLAM World (无尺度) | c2w: [tx,ty,tz,qx,qy,qz,qw] |
| `tstamp` | (N,) | - | 关键帧在原始帧序列中的索引 |
| `disps` | (N, H, W) | - | 逆深度 (视差) |
| `scale` | scalar | - | Metric3D 尺度因子 |
| `img_focal` | scalar | - | 焦距 (像素) |
| `img_center` | (2,) | - | 主点 [cx, cy] |

**帧间关系**: `tstamp` 是关键帧索引子集，不是所有帧都有 SLAM 位姿。`traj` 只包含关键帧。

**如何恢复完整帧的相机位姿**: `load_slam_cam()` 加载后，`traj` 乘以 `scale` 得到米制 c2w。但 `tstamp` 只覆盖关键帧。在 `hawor_infiller()` 中，`R_c2w_sla_all[frame_ck]` 直接用帧索引访问，说明 DROID-SLAM 的 `traj` 实际上覆盖了所有帧（或经过插值）。

#### `world_space_res.pth`

**产生过程**: `hawor_infiller()` 将相机空间手部姿态转换到世界空间

```python
# hawor_video.py 第 280 行
data_world = cam2world_convert(R_c2w_sla, t_c2w_sla, data_out, handedness)
```

**cam2world_convert 详解** (`custom_utils.py:67-97`):

```python
# 1. 旋转: 世界旋转 = SLAM c2w旋转 × 相机空间旋转
init_rot_mat = R_c2w_sla @ init_root_orient

# 2. 平移: 先用 MANO 计算根关节位置，再变换到世界坐标
root_loc = mano_output["joints"][..., 0, :]  # 相机空间根关节
offset = init_trans - root_loc                # 常数偏移
init_trans = R_c2w_sla @ root_loc + t_c2w_sla + offset
```

**坐标系**: SLAM World (OpenCV: y-down, z-forward, 米制)
**格式**: joblib，[pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid]

**帧间关系**: pred_trans 形状 (2, T, 3)，T = 全部帧数。`pred_valid` 标记哪些帧有有效手部检测，缺失帧由 infiller 填充。

#### `reconstruction/hawor_results_0_113.npz` ← **最终输出**

**产生过程**: `demov2.py` 第 537-645 行

**关键**: 保存前应用了 R_x 变换！

```python
# demov2.py 第 537-541 行
R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all[:num_total_frames])
t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all[:num_total_frames])
left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
```

**但 pred_trans 和 pred_rot 没有乘 R_x！** 它们仍然是原始 SLAM World。

```python
# demov2.py 第 635-645 行
np.savez(save_path,
         pred_trans=to_numpy(pred_trans),      # ← 原始 SLAM World (y-down, z-forward)
         pred_rot=to_numpy(pred_rot),          # ← 原始 SLAM World
         pred_hand_pose=...,                    # ← 原始
         pred_betas=...,                        # ← 原始
         pred_valid=...,                        # ← 原始
         R_c2w=to_numpy(R_c2w_sla_all),        # ← 已乘 R_x (y-up, z-backward)
         t_c2w=to_numpy(t_c2w_sla_all),        # ← 已乘 R_x
         img_focal=img_focal,
         start_idx=start_idx,
         end_idx=end_idx)
```

**坐标系混合**:

| Key | 坐标系 | 说明 |
|-----|--------|------|
| `pred_trans` | SLAM World (y-down, z-forward, 米) | **原始**，未乘 R_x |
| `pred_rot` | SLAM World | **原始**，未乘 R_x |
| `pred_hand_pose` | - | 关节旋转，与坐标系无关 |
| `pred_betas` | - | MANO 形状参数，与坐标系无关 |
| `pred_valid` | - | 有效性标记 |
| `R_c2w` | OpenGL (y-up, z-backward, 米) | **已乘 R_x** |
| `t_c2w` | OpenGL (y-up, z-backward, 米) | **已乘 R_x** |
| `img_focal` | - | 焦距 (像素) |

**恢复原始 SLAM World 相机位置**:
```python
R_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
cam_original = R_x @ t_c2w[i]   # 逆 R_x
```

**帧间关系**: 113 帧 (0-112)，连续。`pred_valid` 标记哪些帧有手部检测。

**实测数据**:
```
R_c2w[0] (已乘R_x): ||R - diag(1,-1,-1)|| ≈ 0.004  ← 确认已乘 R_x
t_c2w[0] (已乘R_x): [-0.004, -0.004, -0.001]
cam_original[0] (逆R_x): [0.004, 0.004, 0.001]  ← 原始 SLAM World
右手范围 (pred_trans, 原始SLAM World): x[-0.026,-0.005], y[0.000,0.019], z[-0.006,0.038]
手距相机: ~0.043m
```

### 2.3 相机产生方式详解

HaWoR 的相机估计是**增量式**：

```
输入: N 帧图像 + 手部掩码
  ↓ DROID-SLAM (增量式)
输出: traj (关键帧, 7维 c2w, 无尺度)
  ↓ Metric3D (单帧深度估计)
输出: 每关键帧度量深度
  ↓ est_scale_hybrid()
输出: scale (全局尺度因子)
  ↓ 应用尺度
最终: t_c2w = traj[:,:3] * scale (米制)
```

**特点**:
- 增量式，有累积漂移
- 手部区域被掩码排除，不参与 SLAM
- Metric3D 恢复绝对尺度，但可能有 10-20% 偏差
- 关键帧子集，非所有帧

### 2.4 手部姿态的产生与坐标变换

```
Step 1: HAWOR 模型推理 (相机坐标系)
  输入: 图像块 + 检测框 + 内参
  输出: init_root_orient (3×3), init_trans (3,), init_hand_pose (15×3×3), init_betas (10,)
  坐标系: 相机坐标系 (OpenCV: x-right, y-down, z-forward)
  保存: cam_space/*.json

Step 2: cam2world_convert (相机 → 世界)
  旋转: R_world = R_c2w_slam @ R_cam
  平移: t_world = R_c2w_slam @ root_loc_cam + t_c2w_slam + offset
  坐标系: SLAM World (y-down, z-forward, 米)
  保存: world_space_res.pth

Step 3: Infiller (填充缺失帧)
  输入: 已有帧的手部姿态 + pred_valid
  输出: 全部帧的手部姿态
  坐标系: SLAM World (不变)

Step 4: MANO 生成网格顶点
  输入: pred_trans, pred_rot, pred_hand_pose, pred_betas
  输出: vertices (T, V, 3)
  坐标系: SLAM World (y-down, z-forward, 米)

Step 5: R_x 变换 (仅用于渲染和保存 R_c2w/t_c2w)
  vertices_render = R_x @ vertices
  R_c2w_render = R_x @ R_c2w
  t_c2w_render = R_x @ t_c2w
  坐标系: OpenGL (y-up, z-backward, 米)
  注意: pred_trans, pred_rot 不变！
```

### 2.5 帧间关系

```
原始视频 (mp4)
  ↓ detect_track_video()
  ↓ 提取全部帧
  ↓ 帧索引: 0, 1, 2, ..., 112
  ↓
  ├── extracted_images/  ← 全部 113 帧
  ├── model_masks.npy    ← 全部 113 帧 (手部掩码)
  ├── model_tracks.npy   ← 检测到手的帧子集
  │
  ├── SLAM/
  │   ├── tstamp         ← 关键帧索引子集
  │   └── traj           ← 关键帧 c2w (未缩放)
  │
  ├── cam_space/*.json   ← 检测到手的连续帧块
  │
  ├── world_space_res.pth ← 全部帧 (infiller 填充后)
  │   ├── pred_trans     ← (2, 113, 3) 全部帧
  │   └── pred_valid     ← (2, 113) 标记哪些帧有真实检测
  │
  └── hawor_results_*.npz ← 全部帧
      ├── pred_trans     ← (2, 113, 3) 原始 SLAM World
      ├── R_c2w          ← (113, 3, 3) 已乘 R_x
      └── t_c2w          ← (113, 3) 已乘 R_x
```

---

## 3. 两系统对应关系与帧间关系

### 3.1 文件对应表

| 数据类型 | RAS 文件 | HaWoR 文件 | 格式差异 |
|---------|---------|-----------|---------|
| 帧图像 | `color/0.jpg` ~ `19.jpg` | `extracted_images/0000.jpg` ~ `0112.jpg` | 命名不同，帧数不同 |
| 外参 | `extrinsics/0.txt` ~ `19.txt` | `hawor_results_*.npz` → `R_c2w`, `t_c2w` | RAS: w2c 4×4; HaWoR: c2w (已乘R_x) |
| 内参 | `intrinsic.txt` (3×3) | `hawor_results_*.npz` → `img_focal` | RAS: 完整矩阵; HaWoR: 仅焦距 |
| 深度 | `depth/0.png` ~ `19.png` | SLAM `disps` (视差) | RAS: 度量深度; HaWoR: 逆深度 |
| 点云 | `point_cloud.ply` | 无 | RAS 独有 |
| 场景网格 | `final_scene.glb` | 无 | RAS 独有 |
| 手部姿态 | 无 | `pred_trans`, `pred_rot` | HaWoR 独有 |
| 手部掩码 | `instance_masks.mp4` | `model_masks.npy` | 格式不同 |

### 3.2 帧索引对应

```
RAS:  0, 1, 2, ..., 19          (20 帧，均匀采样)
HaWoR: 0, 1, 2, ..., 112        (113 帧，全部)

对应关系: RAS 帧 i 对应 HaWoR 帧 i (前 20 帧直接对应)
```

**问题**: RAS 用 `--max_frames` 均匀采样，HaWoR 用全部帧。如果视频有 160 帧，RAS 每 8 帧取一帧。此时 RAS 帧 0 对应原始帧 0，RAS 帧 1 对应原始帧 8，而 HaWoR 帧 1 对应原始帧 1。

**当前数据**: RAS 只有 20 帧，HaWoR 有 113 帧。需要确认 RAS 的 20 帧是否是 HaWoR 的前 20 帧。

### 3.3 相机坐标系对应

```
RAS 相机 (OpenCV):              HaWoR 相机 (OpenCV):
  x → right                       x → right
  y → down                        y → down
  z → forward                     z → forward
  ↑ 完全一致                      ↑ 完全一致
```

两个系统的**相机空间约定完全相同**（都是 OpenCV），这是对齐的基础。

### 3.4 世界坐标系对应

```
RAS Room World (z-up):          HaWoR SLAM World (y-down, z-forward):
  x → 沿主墙面                    x → right
  y → 沿次墙面                    y → down
  z → up (地板法线)               z → forward
  原点: 地板 z=0, 场景中心        原点: 第一帧相机位置
  尺度: 米 (VGGT)                 尺度: 米 (Metric3D)
```

### 3.5 帧间关系对比

| 维度 | RAS | HaWoR |
|------|-----|-------|
| 帧采样 | 均匀采样 (max_frames) | 全部帧 |
| 相机估计 | 全局联合优化 (VGGT) | 增量式 (DROID-SLAM) |
| 深度估计 | 全局联合优化 | Metric3D 单帧 |
| 尺度恢复 | 无 (模型内部) | Metric3D 比较恢复 |
| 帧间一致性 | 强 (联合优化) | 弱 (增量漂移) |
| 手部处理 | 无 | 掩码排除 + 独立估计 |

---

## 4. 对齐方案：GLB → HaWoR

### 4.1 目标

将 RAS 的 `final_scene.glb` 变换到 HaWoR SLAM World 坐标系（`pred_trans` 所在的坐标系）。

### 4.2 变换链

```
RAS GLB (y-up, VGGT单位)
    ↓ YUP_TO_ZUP = [[1,0,0],[0,0,-1],[0,1,0]]
RAS Room World (z-up, VGGT单位)
    ↓ 逆对齐: p_slam = (1/s) * R_total^T @ (p_room - t)
HaWoR SLAM World (y-down, z-forward, 米制)
```

### 4.3 关键：从 npz 恢复原始 SLAM 相机位置

`hawor_results_*.npz` 中的 `R_c2w` 和 `t_c2w` 已乘 R_x，需要逆变换：

```python
R_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
# R_x 是自逆的: R_x @ R_x = I
cam_original_pos = R_x @ t_c2w[i]   # 恢复原始 SLAM World 相机位置
```

### 4.4 参数估计

| 参数 | 值/方法 | 保证程度 |
|------|---------|---------|
| R_axis | `[[1,0,0],[0,0,1],[0,-1,0]]` |   数学推导 |
| R_residual | 第一帧外参都 ≈ I → = I |   实测验证 |
| t | `RAS_cam[0] - s * R_total @ HaWoR_cam_original[0]` |   可靠 |
| s | 深度图比较 或 手动 s=1 |   **需验证** |

### 4.5 操作步骤

详见 [GLB_to_HaWoR_Alignment_Plan.md](./GLB_to_HaWoR_Alignment_Plan.md) 第 3 节，包含完整可执行代码。

---

## 5. 对齐正确性保证

### 5.1 桥梁原理

两个系统处理同一个 mp4，同一帧的相机位置是同一个物理点在两个坐标系中的表示。相机轨迹是连接两个坐标系的可靠桥梁。

### 5.2 分项保证性

| 环节 | 保证程度 | 说明 |
|------|---------|------|
| 轴约定 R_axis |   | 两个系统都使用 OpenCV 相机约定 |
| 外参格式 |   | RAS: w2c; HaWoR: c2w (需逆R_x) |
| 帧对应 |   | 同一 mp4，前 N 帧直接对应 |
| R_residual |   | 第一帧外参都 ≈ I |
| 原点 t |   | 第一帧相机位置对齐 |
| **尺度 s** |   **⚠️** | 需要交叉验证 |

### 5.3 预期精度

| 场景 | 精度 |
|------|------|
| 理想 | < 5cm |
| 典型 | 10-20cm |
| 最坏 | 不可用 |

### 5.4 验证清单

1. 相机轨迹残差 < 0.1m
2. 手部在场景范围内
3. 手部在地板上方 (SLAM World y < 0)
4. 手部不穿模

---

## 附录 A：坐标系变换速查表

| 变换 | 矩阵 | 效果 |
|------|------|------|
| YUP_TO_ZUP (GLB→Room) | `[[1,0,0],[0,0,-1],[0,1,0]]` | (x,y,z)→(x,-z,y) |
| ZUP_TO_YUP (Room→GLB) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_axis (Room→SLAM) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_x (OpenCV→OpenGL) | `diag(1,-1,-1)` | (x,y,z)→(x,-y,-z) |

## 附录 B：坐标系全链路图

```
ReplicateAnyScene:
  VGGT World (任意) ──R,t──→ Room World (z-up, 米) ──z2y──→ GLB (y-up)
  │                            │
  │ extrinsics/*.txt (w2c)     │ depth/*.png (相机z)
  │ intrinsic.txt              │ point_cloud.ply
  │                            │
  └────────────────────────────┘

HaWoR:
  SLAM World (y-down, z-forward, 无尺度) ──×scale──→ SLAM World (米)
  │                                                   │
  │ SLAM/traj (c2w, 7维, 未缩放)                      │ pred_trans (原始SLAM World)
  │ SLAM/scale                                        │ pred_rot (原始SLAM World)
  │                                                   │
  │ cam_space/*.json (相机空间)                        │ hawor_results R_c2w/t_c2w (已乘R_x)
  │   ↓ cam2world_convert                             │
  │   └──→ world_space_res.pth (SLAM World, 米)       │
  │                                                   │
  └───────────────────────────────────────────────────┘

对齐路径:
  RAS GLB (y-up) ──y2z──→ Room World (z-up) ──逆对齐──→ SLAM World (y-down, z-forward, 米)
  ↑                                                            ↑
  final_scene.glb                                         pred_trans 所在坐标系
```

## 附录 C：HaWoR npz 数据坐标系总结

```
hawor_results_0_113.npz 中的数据分属两个坐标系:

┌─────────────────────────────────────────────────────────┐
│  原始 SLAM World (y-down, z-forward, 米)                │
│  ├── pred_trans    (2, 113, 3)  手部根关节平移          │
│  ├── pred_rot      (2, 113, 3)  手部根关节旋转(轴角)    │
│  ├── pred_hand_pose (2, 113, 45) 关节旋转               │
│  ├── pred_betas    (2, 113, 10) MANO形状参数            │
│  └── pred_valid    (2, 113)     有效性标记              │
│                                                         │
│  OpenGL (y-up, z-backward, 米) ← 已乘 R_x              │
│  ├── R_c2w         (113, 3, 3)  相机→世界旋转          │
│  └── t_c2w         (113, 3)     相机→世界平移          │
│                                                         │
│  恢复原始 SLAM World 相机位置:                           │
│  cam_original = R_x @ t_c2w[i]                          │
│  R_c2w_original = R_x @ R_c2w[i]                        │
└─────────────────────────────────────────────────────────┘
```



---



---


# GLB → HaWoR 对齐操作指南

## 6. 实际文件路径与数据概况
### 输入文件

| 系统 | 路径 | 帧数 |
|------|------|------|
| RAS 输出 | `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/my_7mp4_result` | 20 帧 (0-19) |
| HaWoR 输出 | `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7` | 113 帧 (0-112) |

### 关键文件对照

| 用途 | RAS | HaWoR |
|------|-----|-------|
| 外参（相机位姿） | `extrinsics/0.txt` ~ `19.txt` | `reconstruction/hawor_results_0_113.npz` → `t_c2w`, `R_c2w` |
| 内参 | `intrinsic.txt` | `hawor_results_0_113.npz` → `img_focal` |
| 深度图 | `depth/0.png` ~ `19.png` | SLAM `disps`（视差，非直接深度） |
| 点云 | `point_cloud.ply` | 无 |
| 场景网格 | **`final_scene.glb`** ← 待变换 | 无 |
| 手部姿态 | 无 | `hawor_results_0_113.npz` → `pred_trans`, `pred_rot` |
| SLAM 原始轨迹 | 无 | `SLAM/hawor_slam_w_scale_0_113.npz` → `traj`, `scale` |

### 实测关键数据

```
RAS (Room World, z-up):
  Camera [0]:    position ≈ [0, 0, 0],  R_w2c ≈ I  (||R-I||=0.0005)
  点云范围:       x[-0.78, 1.33],  y[-0.83, 0.35],  z[0.01, 1.52]
  点云中心:       [0.04, -0.06, 0.84]
  GLB 范围(y-up): x[-0.25, 0.39],  y[0.44, 1.04],  z[-0.24, -0.01]

HaWoR (SLAM World, y-down, z-forward):
  pred_trans:     在原始 SLAM World (OpenCV 约定, 米制)
  右手范围:       x[-0.026, -0.005], y[0.000, 0.019], z[-0.006, 0.038]
  手距相机:       ~0.043m
  R_c2w[0]:       已乘 R_x=diag(1,-1,-1) 后保存 (||R-I||=2.828)
```

### 关键发现

> **HaWoR 的 `hawor_results_0_113.npz` 中 `R_c2w` 和 `t_c2w` 已经应用了 R_x 翻转 (OpenCV→OpenGL)，不是原始 SLAM World。`pred_trans` 仍然是原始 SLAM World。**

验证方法：`R_c2w[0]` 看是否是 `[[1,0,0],[0,-1,0],[0,0,-1]]` 附近。如果是，说明已应用 R_x。

---

## 7. 目标
将 RAS 的 `final_scene.glb` 变换到 **HaWoR SLAM World 坐标系**（即 `pred_trans` 所在的坐标系）：

| 属性 | RAS GLB（当前） | 目标坐标系（SLAM World） |
|------|----------------|------------------------|
| 朝上轴 | +Y | **-Y**（即 y-down = OpenCV 约定） |
| 朝前轴 | -Z | **+Z** |
| 原点 | 地板 z=0, 场景中心 | 第一帧相机位置 ≈ (0,0,0) |
| 尺度 | VGGT 单位 | 米 (Metric3D) |

---

## 8. 对齐原理
### 8.1 变换链
```
RAS GLB (y-up, VGGT单位)
    ↓ 步骤 A: y-up → z-up
RAS Room World (z-up, VGGT单位)
    ↓ 步骤 B: 逆对齐 → 统一到 SLAM World
HaWoR SLAM World (y-down, z-forward, 米制)  ← 与 pred_trans 同一坐标系
```

### 8.2 各步骤
#### 步骤 A：GLB y-up → Room World z-up

`main.py` 导出 GLB 时做了 `z-up → y-up` 变换（`[[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]`），逆变换为 `y-up → z-up`：

```python
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
# 效果: (x, y, z) → (x, -z, y)
```

#### 步骤 B：Room World → SLAM World

三个坐标系的轴约定都是 OpenCV，但世界坐标系朝向不同：

```
Room World (z-up):         SLAM World (y-down, z-forward):
    +Z (up)                     -Y (up)
    |                            |
    +---- +Y                    +---- +Z (forward)
   /                            /
  +X                           +X (right)
```

从 Room World z-up 到 SLAM World OpenCV 的轴转换：

```python
R_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
# 效果: (x, y, z) → (x, z, -y)
# 即将 Room z-up 转为 SLAM y-down, z-forward
```

完整逆变换公式：

```
p_slam = (1/s) * R_total^T @ (p_room - t)

其中:
  R_total = R_residual @ R_axis      (旋转部分)
  s       = 尺度因子                  (VGGT单位 / 米)
  t       = 原点偏移                  (Room World → SLAM World)
```

### 8.3 参数估计
| 参数 | 估计方法 | 保证程度 |
|------|---------|---------|
| `R_axis` | **已知**：`[[1,0,0],[0,0,1],[0,-1,0]]` |   数学推导 |
| `R_residual` | 两个系统第一帧外参都 ≈ I → 残差 ≈ I |   可靠 |
| `t` | `t = RAS_cam[0] - s * R_total @ HaWoR_cam_original[0]` |   第一帧对齐 |
| `s` | 深度图比较 或 轨迹位移比例 |   **需要交叉验证** |

---

## 9. 操作步骤（在终端逐段执行）
### 9.1 加载数据
```python
import numpy as np
import cv2
import trimesh
from glob import glob
import os

# ===== RAS 侧 =====
RAS_OUT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/my_7mp4_result'

ext_files = sorted(glob(os.path.join(RAS_OUT, 'extrinsics', '*.txt')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
ras_extrinsics = []
for f in ext_files:
    ext = np.loadtxt(f)
    if ext.shape == (3, 4):
        ext = np.vstack([ext, [0, 0, 0, 1]])
    ras_extrinsics.append(ext)
ras_extrinsics = np.array(ras_extrinsics)

# RAS 相机位置：外参是 w2c，cam_pos = -R_w2c^T @ t_w2c
ras_cam = np.array([-e[:3,:3].T @ e[:3,3] for e in ras_extrinsics])

print(f"RAS: {len(ras_cam)} 帧, cam[0]={ras_cam[0]}")
print(f"RAS cam range: x[{ras_cam[:,0].min():.4f},{ras_cam[:,0].max():.4f}] "
      f"y[{ras_cam[:,1].min():.4f},{ras_cam[:,1].max():.4f}] "
      f"z[{ras_cam[:,2].min():.4f},{ras_cam[:,2].max():.4f}]")

# ===== HaWoR 侧 =====
HAWOR_RES = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7/reconstruction/hawor_results_0_113.npz'

hawor = dict(np.load(HAWOR_RES, allow_pickle=True))
hawor_t_c2w = hawor['t_c2w']          # 已乘 R_x，OpenGL 约定
hawor_R_c2w = hawor['R_c2w']          # 已乘 R_x，OpenGL 约定
pred_trans = hawor['pred_trans']       # 原始 SLAM World (OpenCV 约定，米制)
pred_valid = hawor['pred_valid']

# 恢复原始 SLAM World 的相机位置 (逆 R_x)
R_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
hawor_cam_original = (np.array([R_x @ t for t in hawor_t_c2w]))
hawor_R_c2w_original = np.array([R_x @ R for R in hawor_R_c2w])

print(f"\nHaWoR: {len(hawor_cam_original)} 帧, cam_original[0]={hawor_cam_original[0]}")
print(f"HaWoR cam_original range: x[{hawor_cam_original[:,0].min():.4f},{hawor_cam_original[:,0].max():.4f}] "
      f"y[{hawor_cam_original[:,1].min():.4f},{hawor_cam_original[:,1].max():.4f}] "
      f"z[{hawor_cam_original[:,2].min():.4f},{hawor_cam_original[:,2].max():.4f}]")

# 手部位置 (原始 SLAM World)
right_hand = pred_trans[1, pred_valid[1]]
print(f"\nRight hand range: x[{right_hand[:,0].min():.4f},{right_hand[:,0].max():.4f}] "
      f"y[{right_hand[:,1].min():.4f},{right_hand[:,1].max():.4f}] "
      f"z[{right_hand[:,2].min():.4f},{right_hand[:,2].max():.4f}]")

# 帧对应：取交集 (RAS 20帧, HaWoR 113帧)
common = list(range(min(len(ras_cam), len(hawor_cam_original))))
print(f"\n共同帧: {len(common)} (0 到 {common[-1]})")
```

### 9.2 计算轴约定旋转 R_axis
```python
# 轴约定: Room World z-up → SLAM World y-down, z-forward
# (x_room, y_room, z_room) → (x_slam, z_slam, -y_slam)
R_axis = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0,-1, 0]
], dtype=np.float64)

print("R_axis (Room z-up → SLAM World):")
print(R_axis)
print(f"R_axis 作用: (x,y,z) → (x, z, -y)")
print(f"  +Z (up) → -Y (up=down), +Y → +Z (forward)")
```

### 9.3 计算残差旋转 R_residual
```python
# 检查两个系统第一帧相机朝向是否一致
print(f"RAS 第一帧 R_w2c: ||R-I|| = {np.linalg.norm(ras_extrinsics[0,:3,:3] - np.eye(3)):.6f}")
print(f"HaWoR 第一帧 R_c2w_original: ||R-I|| = {np.linalg.norm(hawor_R_c2w_original[0] - np.eye(3)):.6f}")

# 如果两个值都很小 (< 0.1)，说明第一帧相机朝向一致，R_residual ≈ I
if np.linalg.norm(ras_extrinsics[0,:3,:3] - np.eye(3)) < 0.1 and \
   np.linalg.norm(hawor_R_c2w_original[0] - np.eye(3)) < 0.1:
    R_residual = np.eye(3)
    print("✓ 两个系统第一帧朝向一致，R_residual = I")
else:
    # Umeyama 估计残差
    src_pts = (R_axis @ hawor_cam_original[common].T).T
    dst_pts = ras_cam[common]
    src_c = src_pts - src_pts.mean(0)
    dst_c = dst_pts - dst_pts.mean(0)
    cov = dst_c.T @ src_c / len(src_pts)
    U, _, VH = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[2,2] = -1
    R_residual = U @ S @ VH
    angle = np.degrees(np.arccos(np.clip((np.trace(R_residual)-1)/2, -1, 1)))
    print(f"R_residual 旋转角度: {angle:.1f}°")

R_total = R_residual @ R_axis
print(f"R_total = R_residual @ R_axis:")
print(R_total)
```

### 9.4 估计尺度 s（选择一种方法）
#### 方法 A：深度图比较（推荐）

```python
# 比较 RAS 深度图均值 与 HaWoR 手部到相机的距离
depth_0 = cv2.imread(os.path.join(RAS_OUT, 'depth', '0.png'), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
ras_mean_depth = depth_0[depth_0 > 0].mean()

hawor_hand_dist = np.linalg.norm(right_hand[0] - hawor_cam_original[0])

s_depth = ras_mean_depth / hawor_hand_dist
print(f"\n=== 方法 A: 深度图尺度 ===")
print(f"  RAS 平均深度: {ras_mean_depth:.4f} (VGGT单位)")
print(f"  HaWoR 手-相机距离: {hawor_hand_dist:.4f} m")
print(f"  s_depth = {s_depth:.4f}")
```

#### 方法 B：轨迹位移比例（参考）

```python
# 比较相机在各自坐标系中的位移
ras_disp = np.linalg.norm(ras_cam[common[-1]] - ras_cam[common[0]])
hawor_disp = np.linalg.norm(hawor_cam_original[common[-1]] - hawor_cam_original[common[0]])
s_traj = ras_disp / hawor_disp if hawor_disp > 1e-6 else 1.0

print(f"\n=== 方法 B: 轨迹尺度 ===")
print(f"  RAS 相机位移: {ras_disp:.4f} (VGGT单位)")
print(f"  HaWoR 相机位移: {hawor_disp:.4f} m")
print(f"  s_traj = {s_traj:.4f}")
```

#### 选择尺度

```python
# 选择方法 A 或 B，或者手动指定
# s = s_depth   # 方法 A
# s = s_traj    # 方法 B
s = 1.0         # 手动指定（如果知道 VGGT 单位 ≈ 米）

print(f"\n最终使用尺度 s = {s:.4f}")
print(f"  p_ras = {s:.4f} * R_total @ p_hawor + t")
print(f"  p_hawor = {1/s:.4f} * R_total^T @ (p_ras - t)")
```

### 9.5 计算平移 t（原点对齐）
```python
# 正变换: p_ras = s * R_total @ p_hawor + t
# 对齐第一帧相机位置: RAS_cam[0] = s * R_total @ HaWoR_cam[0] + t
t = ras_cam[0] - s * (R_total @ hawor_cam_original[0])

# 逆变换参数 (用于 GLB)
s_inv = 1.0 / s
R_inv = R_total.T
t_inv = -s_inv * (R_inv @ t)

print(f"\n对齐参数:")
print(f"  正变换 (HaWoR→RAS): p_ras = {s:.4f} * R_total @ p_hawor + t")
print(f"  逆变换 (RAS→HaWoR): p_hawor = {s_inv:.6f} * R_inv @ p_ras + t_inv")
print(f"  t = {t}")
print(f"  t_inv = {t_inv}")
```

### 9.6 残差验证
```python
# 变换 HaWoR 相机到 RAS Room World，与 RAS 相机位置比较
aligned_hawor = s * (R_total @ hawor_cam_original.T).T + t
errors = np.linalg.norm(aligned_hawor[common] - ras_cam[common], axis=1)

print(f"\n=== 残差验证 ===")
print(f"  共同帧数: {len(common)}")
print(f"  对齐误差: mean={errors.mean():.6f}, median={np.median(errors):.6f}, max={errors.max():.6f}")
print(f"  每帧误差: {np.array2string(errors, precision=4, suppress_small=True)}")

if np.median(errors) < 0.1:
    print(f"  ✓ 中位误差 < 0.1m，对齐可靠")
elif np.median(errors) < 0.5:
    print(f"  ⚠ 中位误差 0.1-0.5m，可能需要深度图验证")
else:
    print(f"  ✗ 中位误差 > 0.5m，对齐不可靠！请检查帧对应和尺度")
```

### 9.7 变换 GLB
```python
# 步骤 A: GLB y-up → Room World z-up
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# 步骤 B: Room World → SLAM World
# p_hawor = s_inv * R_inv @ (p_room - t)
# 对 GLB 顶点: p_slam = s_inv * R_inv @ (YUP_TO_ZUP @ p_glb - t)

R_combined = s_inv * (R_inv @ YUP_TO_ZUP)
t_combined = s_inv * (R_inv @ (-t))

T_4x4 = np.eye(4)
T_4x4[:3, :3] = R_combined
T_4x4[:3, 3] = t_combined

print(f"\n=== 变换矩阵 (4x4) ===")
print(T_4x4)

# 加载 GLB 并应用变换
glb_path = os.path.join(RAS_OUT, 'final_scene.glb')
scene = trimesh.load(glb_path)
scene.apply_transform(T_4x4)

# 保存
OUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/aligned_output'
os.makedirs(OUT_DIR, exist_ok=True)
output_glb = os.path.join(OUT_DIR, 'scene_in_hawor_world.glb')
scene.export(output_glb)

print(f"\n变换后 GLB 已保存到: {output_glb}")
print(f"变换后场景范围 (SLAM World, 米制):")
print(f"  x: [{scene.bounds[0,0]:.4f}, {scene.bounds[1,0]:.4f}]")
print(f"  y: [{scene.bounds[0,1]:.4f}, {scene.bounds[1,1]:.4f}]")
print(f"  z: [{scene.bounds[0,2]:.4f}, {scene.bounds[1,2]:.4f}]")
```

### 9.8 验证空间关系
```python
# 检查变换后的场景范围是否与手部位置匹配
right_hand = pred_trans[1, pred_valid[1]]

print(f"\n=== 空间验证 ===")
print(f"右手范围 (SLAM World, 米):")
print(f"  x: [{right_hand[:,0].min():.4f}, {right_hand[:,0].max():.4f}]")
print(f"  y: [{right_hand[:,1].min():.4f}, {right_hand[:,1].max():.4f}]")
print(f"  z: [{right_hand[:,2].min():.4f}, {right_hand[:,2].max():.4f}]")

print(f"\n场景范围 (SLAM World, 米):")
print(f"  x: [{scene.bounds[0,0]:.4f}, {scene.bounds[1,0]:.4f}]")
print(f"  y: [{scene.bounds[0,1]:.4f}, {scene.bounds[1,1]:.4f}]")
print(f"  z: [{scene.bounds[0,2]:.4f}, {scene.bounds[1,2]:.4f}]")

# 检查手是否在场景内
hand_in_x = (right_hand[:,0].min() >= scene.bounds[0,0] - 0.5 and
             right_hand[:,0].max() <= scene.bounds[1,0] + 0.5)
hand_in_y = (right_hand[:,1].min() >= scene.bounds[0,1] - 0.5 and
             right_hand[:,1].max() <= scene.bounds[1,1] + 0.5)
hand_in_z = (right_hand[:,2].min() >= scene.bounds[0,2] - 0.5 and
             right_hand[:,2].max() <= scene.bounds[1,2] + 0.5)

if hand_in_x and hand_in_y and hand_in_z:
    print("\n✓ 手部在场景范围内（含 0.5m 容差）")
else:
    print(f"\n⚠ 手部可能超出场景范围")
    print(f"  hand_in_x={hand_in_x}, hand_in_y={hand_in_y}, hand_in_z={hand_in_z}")
```

---

## 10. 对齐正确性分析
### 10.1 为什么这个对齐在数学上是正确的
```
同一个 mp4 同一帧 i 的相机位置：
  RAS  给出 cam_ras[i]   ← Room World (z-up, VGGT单位)
  HaWoR 给出 cam_hawor[i] ← SLAM World (y-down, z-forward, 米)

→ 两个值代表同一个物理点 → 存在唯一的 {R, t, s} 使得 cam_ras = s·R·cam_hawor + t
→ 这个 {R, t, s} 适用于场景中所有 3D 点
→ 相机轨迹是连接两个坐标系的可靠桥梁
```

### 10.2 分项保证性
| 环节 | 代码来源确认 | 保证程度 |
|------|------------|---------|
| 轴约定 R_axis | VGGT + DROID-SLAM 都使用 OpenCV 约定 (x-right, y-down, z-forward) |   数学推导 |
| 外参格式 | RAS: w2c 4×4; HaWoR: c2w (R_c2w, t_c2w) 已确认 |   代码审查 |
| 帧对应 | 同一 mp4，帧索引直接对应 |   前提满足 |
| R_residual | 第一帧 ||R-I|| < 0.001，残差旋转 < 0.5° |   实测验证 |
| 原点 t | 第一帧相机位置对齐 |   可靠 |
| **尺度 s** | 需要交叉验证 |   **关键变量** |

### 10.3 尺度 s 的选择建议
| 方法 | 本数据结果 | 可靠性 |
|------|-----------|--------|
| A: 深度图比较 | s ≈ 20.6 |   VGGT 深度取全局均值，可能偏高 |
| B: 轨迹位移 | s ≈ 2.7 | ⚠️ 相机运动太小(0.03m)，不稳定 |
| C: 手动 s=1 | s = 1.0 | 假设 VGGT 单位 ≈ 米（官方声称） |

**建议**：先用 s=1 尝试，然后根据残差验证结果调整。如果手部明显不在场景内（太大或太小），调大或调小 s 直到空间关系合理。

### 10.4 验证清单
| 检查项 | 方法 | 阈值 |
|--------|------|------|
| ✓ 相机轨迹残差 | `||aligned[i] - ras_cam[i]||` | < 0.1m 可靠 |
| ✓ 手在场景范围内 | bounds 检查 | 手 xz ⊆ 场景 xz |
| ✓ 手在地板上方 | `hand_y < 0` (SLAM World y-down) | 物理合理 |
| ✓ 手在物体前面 | 深度比较 | 不穿模 |

### 10.5 如果对齐结果不对
```
症状 1: 残差很大 (> 0.5m)
  → 检查帧对应：RAS 和 HaWoR 处理的帧是否一致
  → 检查是否用了正确的 hawor_cam_original（不是 haw_c2w 原始）

症状 2: 手在场景外面
  → 调大或调小 s
  → 用手-相机距离(0.043m)除以 RAS 相应区域的深度来重新估计 s

症状 3: 场景看起来旋转了
  → 检查 R_x 是否被正确逆应用
  → 检查 YUP_TO_ZUP 矩阵方向是否正确

症状 4: 场景朝向完全错误 (不是简单旋转，而是上下颠倒等)
  → 可能是 RAS 房间对齐失败 (R=I, t=0)，见 4.6 节
```

### 10.6 房间对齐退化情况
RAS 的 `align_to_room_coordinate_system()` 在以下情况返回 `R=I, t=0`（不做任何变换）：

- 没有检测到地板 (`len(floor_plane_infos) == 0`)
- 没有检测到与地板正交的墙面 (`len(orthogonal_wall_plane_infos) == 0`)

**当 R=I, t=0 时的影响**：

| 维度 | 正常 (Room World z-up) | 退化 (VGGT 原始坐标系) |
|------|----------------------|----------------------|
| 朝上轴 | +Z (保证) | 任意 |
| 原点 | 地板z=0, 场景中心 | VGGT SfM 决定 |
| R_axis | `[[1,0,0],[0,0,1],[0,-1,0]]` (可靠) | 不适用 |

**检测方法**：

```python
# 检查点云 z 范围
pcd = trimesh.load('point_cloud.ply')
z_range = pcd.bounds[1,2] - pcd.bounds[0,2]
# z_range 在 2-4m → 对齐成功 (房间高度)
# z_range 异常 → 对齐失败

# 检查相机位置
cam_pos[0][2]  # 如果 ≈ 1.0-1.7m (人眼高度) → 对齐成功
               # 如果 ≈ 0 或异常值 → 可能对齐失败
```

**退化时的对齐策略**：

不再分解 `R_total = R_residual @ R_axis`，而是用 Umeyama 从相机轨迹直接估计完整的 `R_total`：

```python
# 退化模式: 不假设 R_axis，直接估计
src_pts = hawor_cam_original[common]   # SLAM World
dst_pts = ras_cam[common]              # VGGT 原始 World
# 用 Umeyama 估计 s, R_total, t
```

---

## 11. 后续：在 HaWoR 渲染器中查看
变换后的 GLB 在 SLAM World (y-down, z-forward)。HaWoR 的 aitviewer 渲染器使用 OpenGL (y-up, z-backward)，需要再应用 R_x：

```python
R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# p_render = R_x @ p_slam  (用于可视化渲染)
```

---

## 12. 快速验证命令
所有步骤合并为一段可复制的代码块，在 `conda run -n ReplicateAnyScene python3` 中运行：

```python
# 一行放入 align_and_check.py 运行:
# cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src
# conda run -n ReplicateAnyScene python3 align_and_check.py

import numpy as np; import cv2; import trimesh; import os; from glob import glob

RAS_OUT='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/my_7mp4_result'
HAWOR_RES='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7/reconstruction/hawor_results_0_113.npz'
OUT_DIR='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/aligned_output'

# 1. Load
e=sorted(glob(os.path.join(RAS_OUT,'extrinsics','*.txt')),key=lambda x:int(os.path.basename(x).split('.')[0]))
ras_cam=np.array([(lambda m:(-m[:3,:3].T@m[:3,3]) if m.shape==(4,4) else (-(m:=np.vstack([m,[0,0,0,1]]))[:3,:3].T@m[:3,3]))(np.loadtxt(f)) for f in e])
h=dict(np.load(HAWOR_RES,allow_pickle=True))
Rx=np.array([[1,0,0],[0,-1,0],[0,0,-1]])
hc=np.array([Rx@t for t in h['t_c2w']])
hR=np.array([Rx@R for R in h['R_c2w']])
hp=h['pred_trans'][1,h['pred_valid'][1]]
n=min(len(ras_cam),len(hc))

# 2. Params
R_axis=np.array([[1,0,0],[0,0,1],[0,-1,0]])
R_residual=np.eye(3)
R_total=R_residual@R_axis
s=1.0  # <-- 调整此值！
t=ras_cam[0]-s*(R_total@hc[0])
si,Ri,ti=1/s,R_total.T,-(1/s)*(R_total.T@t)

# 3. Verify
aligned=s*(R_total@hc[:n].T).T+t
errs=np.linalg.norm(aligned-ras_cam[:n],axis=1)
print(f'Scale s={s:.4f}, errors: mean={errs.mean():.4f}, median={np.median(errs):.4f}, max={errs.max():.4f}')
print(f'Hand  range: x[{hp[:,0].min():.3f},{hp[:,0].max():.3f}] y[{hp[:,1].min():.3f},{hp[:,1].max():.3f}] z[{hp[:,2].min():.3f},{hp[:,2].max():.3f}]')

# 4. Transform GLB
Y2Z=np.array([[1,0,0],[0,0,-1],[0,1,0]])
T4=np.eye(4); T4[:3,:3]=si*(Ri@Y2Z); T4[:3,3]=si*(Ri@(-t))
scene=trimesh.load(os.path.join(RAS_OUT,'final_scene.glb'))
scene.apply_transform(T4)
os.makedirs(OUT_DIR,exist_ok=True)
scene.export(os.path.join(OUT_DIR,'scene_in_hawor_world.glb'))
print(f'GLB saved. Bounds: [{scene.bounds[0]}, {scene.bounds[1]}]')
print(f'Hand in scene? x: {hp[:,0].min()>=scene.bounds[0,0]-0.5 and hp[:,0].max()<=scene.bounds[1,0]+0.5}, z: {hp[:,2].min()>=scene.bounds[0,2]-0.5 and hp[:,2].max()<=scene.bounds[1,2]+0.5}')
```

---

## 附录 D：坐标系变换速查表

| 变换 | 矩阵 | 效果 |
|------|------|------|
| YUP_TO_ZUP (GLB→Room) | `[[1,0,0],[0,0,-1],[0,1,0]]` | (x,y,z)→(x,-z,y) |
| ZUP_TO_YUP (Room→GLB) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_axis (Room→SLAM) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_x (OpenCV→OpenGL) | `diag(1,-1,-1)` | (x,y,z)→(x,-y,-z) |

### 坐标系一览

```
ReplicateAnyScene:
  VGGT World (任意) ──R,t──→ Room World (z-up, VGGT单位) ──z2y──→ GLB (y-up)

HaWoR:
  SLAM World (y-down, z-forward, 无尺度) ──scale──→ SLAM World (米制)
  SLAM World pred_trans (米制, y-down) ──R_x──→ R_c2w/t_c2w 保存 (y-up, z-backward)

对齐路径:
  RAS GLB (y-up) ──y2z──→ Room World (z-up) ──逆对齐──→ SLAM World (y-down, z-forward, 米制)
```