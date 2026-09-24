# HaWoR + ReplicateAnyScene 合并架构方案

## 一、目标

将两个项目合并为**一套统一体系**：共用同一个世界坐标系、同一套相机参数、同一个 3D 感知前端，最终输出一份同时包含**场景物体 + 动态双手 mesh** 的完整 3D 场景。

| 项目 | 当前职责 | 合并后角色 |
|------|---------|-----------|
| **HaWoR** | 从第一人称视频重建手部运动 (MANO) | 动态手部跟踪层 |
| **ReplicateAnyScene** | 从任意视频重建可组合式 3D 场景 | 静态场景重建层 |
| **VGGT + VGGT4D** (新引入) | 无 | 统一 3D 感知层 + 动静分离（替代 DROID-SLAM + Metric3D） |

---

## 二、深度对比：HaWoR vs ReplicateAnyScene 的本质差异

合并的难点不在于"怎么写代码"，而在于两个管线在**文件格式、相机轨迹、点云生成、坐标系标定**四个维度上存在根本性差异。不理解这些差异就无法正确合并。

### 2.1 文件输出格式对比

**HaWoR 的输出链路：**

```
视频 → detect_track → hawor_motion_estimation → hawor_slam → hawor_infiller → demov2
```

| 阶段 | 输出文件 | 格式 | 关键内容 | 坐标系 |
|------|---------|------|---------|--------|
| 检测 | `tracks_{s}_{e}/model_tracks.npy` | numpy dict | 手部检测轨迹 | 像素空间 |
| 检测 | `tracks_{s}_{e}/model_masks.npy` | npy (N,H,W) | 手部mask | 像素空间 |
| 运动估计 | `cam_space/{0\|1}/{s}_{e}.json` | JSON | init_root_orient(3,3), init_hand_pose(15,3,3), init_trans(3), init_betas(10) | **相机空间 (OpenCV)** |
| SLAM | `SLAM/hawor_slam_w_scale_{s}_{e}.npz` | NPZ | traj(N,7)[tx,ty,tz,qx,qy,qz,qw], disps(N,H,W), scale, img_focal, img_center | **SLAM世界 (OpenCV)** |
| Infiller | `world_space_res.pth` | joblib list | pred_trans(2,T,3), pred_rot(2,T,3), pred_hand_pose(2,T,45), pred_betas(2,T,10) | **SLAM世界 (OpenCV)** |
| 最终 | `reconstruction/hawor_results_{s}_{e}.npz` | NPZ | pred_trans(2,T,3), pred_rot(2,T,3), pred_hand_pose(2,T,45), pred_betas(2,T,10), R_c2w(N,3,3), t_c2w(N,3), img_focal | ⚠️ **混合！** |

**ReplicateAnyScene 的输出链路：**

```
视频 → VGGT预测 → 房间对齐 → SAM3分割 → SAM3D资产生成 → 空间精修 → GLB导出
```

| 阶段 | 输出文件 | 格式 | 关键内容 | 坐标系 |
|------|---------|------|---------|--------|
| VGGT | `intrinsic.txt` | txt | 3×3内参矩阵 (所有帧均值) | — |
| VGGT | `extrinsics/{i}.txt` | txt | 4×4 c2w矩阵 (逐帧) | **房间坐标系 (z-up)** |
| VGGT | `depth/{i}.png` | uint16 PNG | 深度图 (毫米) | — |
| VGGT | `color/{i}.jpg` | JPEG | RGB帧 | — |
| VGGT | `point_cloud.ply` | PLY | 稠密点云 (置信度>50%分位过滤) | **房间坐标系 (z-up)** |
| 资产生成 | `optimal_frames/*.jpg` | JPEG | 最优视角帧 | — |
| 最终 | `final_scene.glb` | GLB | 完整3D场景 | **y-up (GLB标准)** |

**核心差异：**

| 维度 | HaWoR | ReplicateAnyScene | 合并时的问题 |
|------|-------|-------------------|-------------|
| 数据载体 | NPZ/JSON/PTH (numpy/torch) | TXT/PLY/GLB (逐文件) | 需要统一为 SharedSceneData |
| 帧索引 | 关键帧 (tstamp, 不连续) | 全部帧 (0,1,2,...,S-1) | HaWoR 的 SLAM 只输出关键帧，Infiller 填充全帧 |
| 手部数据 | 有 (MANO参数) | 无 | 需要合并 |
| 场景数据 | 无 | 有 (GLB mesh) | 需要合并 |
| 相机数据 | R_c2w(N,3,3) + t_c2w(N,3) 分开存储 | extrinsics(S,4,4) 合并存储 | 格式不同但语义相同 |
| 深度数据 | disps (视差，无量纲) | depth (米/毫米，度量) | ⚠️ 量纲完全不同 |

### 2.2 相机轨迹预测对比

**这是合并最大的难点。** 两个管线用完全不同的方法估计相机轨迹，产生的轨迹在原点、方向、尺度上都不同。

#### HaWoR: DROID-SLAM + Metric3D (两阶段)

```
DROID-SLAM:
  输入: 视频帧 + 手部mask (屏蔽手部区域)
  输出: traj (N_keyframes, 7) = [tx, ty, tz, qx, qy, qz, qw]
  特点:
    - 只输出关键帧 (非连续), tstamp 记录关键帧索引
    - 平移无度量尺度 (需要 Metric3D 对齐)
    - 四元数格式 (x,y,z,w)
    - 坐标系: 第一帧相机位置为原点, OpenCV约定 (X右, Y下, Z前)
    - 使用光流 + 迭代更新, 对动态物体鲁棒 (因为有手部mask)

Metric3D:
  输入: 单帧RGB图
  输出: pred_depth (H, W) 度量深度 (米)
  特点:
    - 单帧估计, 无多帧一致性
    - 度量尺度 (绝对深度, 单位米)

尺度对齐 (est_scale_hybrid):
  slam_depth = 1 / disp  (SLAM视差 → 无量纲深度)
  scale = median(pred_depth / slam_depth)  (迭代10次 + Geman-McClure鲁棒优化)
  t_c2w_scaled = traj[:, :3] * scale  (只缩放平移, 旋转不变)
```

#### ReplicateAnyScene: VGGT (单阶段)

```
VGGT:
  输入: S帧图像 (均匀采样)
  输出: pose_enc → pose_encoding_to_extri_intri() → extrinsic(S,3,4) + intrinsic(S,3,3)
  特点:
    - 输出所有帧 (连续), 无关键帧概念
    - 平移自带度量尺度 (训练时有深度监督)
    - extrinsic 格式: cam-from-world (3×4)
    - 坐标系: VGGT内部世界系 (原点不确定, 但尺度正确)
    - 内参: 可学习 cx, cy (不强制中心主点)
    - 无手部mask, 动态区域会干扰位姿估计
```

#### 关键差异和合并问题

| 差异 | HaWoR (DROID-SLAM) | RAS (VGGT) | 合并时的问题 |
|------|-------------------|------------|-------------|
| **帧覆盖** | 关键帧 (稀疏, ~30%帧) | 全帧 (稠密, 100%帧) | HaWoR 的 SLAM 只有关键帧位姿，Infiller 需要全帧位姿 |
| **尺度来源** | 外部 (Metric3D) | 内部 (训练时学到的) | VGGT 的尺度可能和 Metric3D 不一致 |
| **动态处理** | 有 (YOLO手部mask → SLAM屏蔽) | 无 (VGGT4D 才有) | VGGT 的位姿会被手部运动干扰 |
| **内参假设** | cx=W/2, cy=H/2 (固定) | cx,cy 可学习 | VGGT 的 cx,cy 可能偏离中心 |
| **四元数格式** | (x,y,z,w) | 旋转矩阵 (3×3) | 需要格式转换 |
| **外参格式** | 分离 R(3,3)+t(3) | 合并 4×4 矩阵 | 需要格式对齐 |
| **世界原点** | 第一帧相机位置 | VGGT 内部 (不确定) | 两个原点完全不同 |

**合并后用 VGGT+VGGT4D 替代 DROID-SLAM+Metric3D，上述问题全部消失：**
- 帧覆盖：VGGT 输出全帧 ✅
- 尺度：VGGT 自带度量尺度 ✅
- 动态处理：VGGT4D 提供动态 mask ✅
- 但需要验证：VGGT 的位姿精度是否够 HaWoR 使用

### 2.3 相机轨迹对齐的数学分析

**核心问题：** 同一段视频跑 DROID-SLAM 和 VGGT，输出的相机轨迹在原点、方向、尺度上都不同，能不能对齐？

#### 两条轨迹的直观差异

```
DROID-SLAM 世界系:              VGGT 世界系:
  Y (下)                          Y (不确定)
  |                               |
  +--- X (右)                     +--- X (不确定)
 /                               /
Z (前)                          Z (不确定)

原点 = 第1帧相机位置              原点 = VGGT 内部决定
尺度 = 1.0 (无量纲)              尺度 = 1.0 (米)
方向 = OpenCV约定 (固定)          方向 = 模型学到的 (不确定)
```

#### 7 个根本区别

| # | 维度 | DROID-SLAM | VGGT | 为什么不同 |
|---|------|-----------|------|-----------|
| 1 | **原点** | 第1帧相机位置 | 模型内部决定 | SLAM 从零开始建图，VGGT 全局优化 |
| 2 | **方向** | OpenCV (Y下,Z前) | 不确定 | SLAM 继承相机约定，VGGT 学到的 |
| 3 | **尺度** | 无量纲 | 米 | SLAM 是单目无尺度，VGGT 有深度监督 |
| 4 | **帧覆盖** | 关键帧 (~30%) | 全帧 (100%) | SLAM 选关键帧优化，VGGT 逐帧输出 |
| 5 | **误差模式** | 累积漂移 | 全局一致但局部抖 | SLAM 增量式，VGGT 一次性 |
| 6 | **动态处理** | 有 (手部mask) | 无 (VGGT4D才有) | SLAM 可以屏蔽区域，VGGT 全图处理 |
| 7 | **帧间关系** | 光流 + 迭代更新 | 全局注意力 | SLAM 依赖时序，VGGT 依赖全局关联 |

#### 如果硬要对齐：Umeyama Sim(3) 对齐

对齐两条轨迹是一个 **Sim(3) 问题**（相似变换 = 旋转 + 平移 + 缩放）：

```
找 R_align, t_align, s_align 使得:
  P_vggt = s_align * R_align @ P_slam + t_align
```

**Step 1: 找对应帧**
```
DROID-SLAM 的 tstamp = [0, 3, 7, 12, 18, ...]  (关键帧索引)
VGGT 的帧索引 = [0, 1, 2, 3, 4, ...]           (全帧)
对应关系: tstamp[i] → VGGT 的第 tstamp[i] 帧
```

**Step 2: 提取相机中心**
```python
# DROID-SLAM
cam_center_slam = -R_c2w_slam.T @ t_c2w_slam  # (N, 3)

# VGGT
cam_center_vggt = -R_c2w_vggt.T @ t_c2w_vggt  # (S, 3)

# 取对应帧的子集
cam_slam = cam_center_slam  # (N, 3)
cam_vggt = cam_center_vggt[tstamp]  # (N, 3) 只取关键帧对应的VGGT帧
```

**Step 3: Umeyama 对齐**
```python
# 1. 去质心
centroid_slam = mean(cam_slam, axis=0)
centroid_vggt = mean(cam_vggt, axis=0)
pts_slam = cam_slam - centroid_slam
pts_vggt = cam_vggt - centroid_vggt

# 2. SVD 求旋转
H = pts_slam.T @ pts_vggt        # (3,3)
U, S, Vt = SVD(H)
R_align = Vt.T @ U.T              # (3,3) 旋转
if det(R_align) < 0:
    Vt[-1] *= -1
    R_align = Vt.T @ U.T

# 3. 求尺度
s_align = sum(S) / sum(pts_slam ** 2)

# 4. 求平移
t_align = centroid_vggt - s_align * R_align @ centroid_slam
```

**Step 4: 应用变换**
```python
# 变换 DROID-SLAM 的轨迹到 VGGT 坐标系
R_c2w_aligned = R_align @ R_c2w_slam
t_c2w_aligned = s_align * R_align @ t_c2w_slam + t_align
```

#### 对齐后仍然存在的 4 个问题

```
问题1: 累积漂移
  DROID-SLAM 的轨迹在长视频中会漂移
  即使对齐了起点和终点，中间帧可能偏离
  → Umeyama 只能做全局对齐，不能修正局部漂移

问题2: 帧覆盖不匹配
  DROID-SLAM 只有 ~30% 关键帧
  HaWoR 的 Infiller 需要全帧位姿
  → 关键帧之间的位姿需要插值，但 SLAM 没有提供

问题3: 尺度不一致
  DROID-SLAM × Metric3D 的尺度 ≈ VGGT 的尺度?
  不一定! Metric3D 是单帧估计，VGGT 是多帧联合估计
  → 即使 Umeyama 求出了 s_align，也可能在局部帧不准确

问题4: 旋转对齐不完美
  Umeyama 假设两条轨迹的旋转关系是全局一致的
  但 DROID-SLAM 的旋转可能有局部误差
  → 对齐后的 R_c2w 在某些帧可能偏差大
```

#### 结论：不要对齐两条轨迹，只用一条

```
方案 A (对齐): 两个系统各自跑 → 两条轨迹 → Umeyama 对齐 → 有误差
方案 B (统一): VGGT+VGGT4D 跑一次 → 一条轨迹 → 两个系统共用 → 无对齐误差
```

**方案 B 就是我们选择的。** VGGT+VGGT4D 输出一条轨迹，经过房间对齐后存入 SharedSceneData，两个管线共用。HaWoR 的 `cam2world_convert` 数学形式完全不变，只是 `R_c2w` 的来源从 DROID-SLAM 变成了 VGGT+VGGT4D：

```python
# 旧 (DROID-SLAM):
R_root_world = R_c2w_slam @ R_root_cam
t_world = R_c2w_slam @ root_loc + t_c2w_slam + offset

# 新 (VGGT+VGGT4D):
R_root_world = R_c2w_room @ R_root_cam    # 只是换了个 R_c2w 来源
t_world = R_c2w_room @ root_loc + t_c2w_room + offset
```

而且用房间坐标系还有一个好处：**不需要 HaWoR 的 `R_x = diag(1,-1,-1)` 翻转了**。因为房间坐标系已经是 z-up（Y 朝上），和 GLB 的 y-up 只差一个简单的轴交换，不需要那个 OpenCV→OpenGL 的翻转。

### 2.4 点云生成对比

#### HaWoR: 无点云输出

HaWoR 管线**不生成点云**。DROID-SLAM 的 `disps` (稠密视差图) 只在内部用于尺度估计，不输出给下游。最终输出只有 MANO 手部顶点 (通过 `run_mano()` 前向传播生成)。

```
HaWoR 的 3D 输出:
  只有手部顶点: (2, T, 778, 3) — 左右手各778个顶点
  没有场景点云
  没有物体点云
```

#### ReplicateAnyScene: VGGT 深度反投影 → 房间对齐点云

```
VGGT 输出:
  depth: (S, H, W, 1) — 度量深度 (米)
  extrinsic: (S, 3, 4) — cam-from-world
  intrinsic: (S, 3, 3) — 相机内参

深度反投影 (unproject_depth_map_to_point_map):
  Step 1: depth → 相机坐标
    x_cam = (u - cu) * depth / fu
    y_cam = (v - cv) * depth / fv
    z_cam = depth
    相机坐标系: OpenCV (X右, Y下, Z前)

  Step 2: 相机坐标 → 世界坐标
    cam_to_world = closed_form_inverse_se3(extrinsic)  # 3×4外参的逆
    world_coords = cam_coords @ R_c2w.T + t_c2w

  Step 3: 房间对齐
    world_points_room = world_points_vggt @ R_room.T + t_room

点云过滤:
  conf > percentile(depth_conf, 50)  # 只保留置信度>50%分位的点
  → point_cloud.ply (trimesh.PointCloud)
```

**关键差异：**

| 维度 | HaWoR | RAS | 合并时的问题 |
|------|-------|-----|-------------|
| 点云类型 | 无 (只有手部mesh顶点) | 稠密场景点云 (H×W×S 个点) | 需要把手部mesh放到场景点云中 |
| 深度来源 | DROID-SLAM视差 (无量纲) | VGGT深度 (度量, 米) | 量纲不同，合并后统一用VGGT |
| 点云坐标系 | SLAM世界 (OpenCV) | 房间坐标系 (z-up) | ⚠️ 需要对齐 |
| 动态鬼影 | SLAM有手部mask，无鬼影 | VGGT无mask，手部区域有鬼影 | ⚠️ 需要VGGT4D过滤 |

**合并后的点云生成方式：**
```
VGGT+VGGT4D → depth + extrinsics + dynamic_mask
  → world_points = unproject(depth, extrinsic, intrinsic)
  → world_points_static = world_points[dynamic_mask == False]  ← 过滤动态区域
  → 房间对齐 → point_cloud.ply (无鬼影)
```

### 2.5 坐标系标定对比

**这是最容易出错的地方。** 两个管线各自有一套坐标系变换链，中间存在多处隐式变换。

#### HaWoR 的坐标系变换链

```
① HAWOR模型输出
   坐标系: 相机空间 (OpenCV: X右, Y下, Z前)
   数据: init_root_orient(3,3), init_trans(3), init_hand_pose(15,3,3), init_betas(10)
   左手特殊处理: angle-axis的Y/Z分量取反 (因为翻转推理)

② cam2world_convert
   变换: R_root_world = R_c2w @ R_root_cam
          t_world = R_c2w @ root_loc + t_c2w + offset
   坐标系: SLAM世界 (OpenCV: X右, Y下, Z前)
   注意: offset = init_trans - root_loc (MANO平移与腕关节的偏移，在相机空间，未旋转!)

③ Infiller
   坐标系: 同② (SLAM世界, OpenCV)
   数据: pred_trans(2,T,3), pred_rot(2,T,3) — 世界空间

④ demov2 R_x翻转 (⚠️ 最容易出错的变换!)
   R_x = [[1,0,0],[0,-1,0],[0,0,-1]]
   效果: OpenCV → OpenGL (Y下→Y上, Z前→Z后)
   应用对象:
     ✅ R_c2w, t_c2w (相机矩阵)
     ✅ left_verts, right_verts (MANO顶点)
     ❌ pred_trans, pred_rot (MANO参数 — 未翻转!)

⑤ 最终NPZ保存
   ⚠️ 坐标系不一致!
   R_c2w/t_c2w → OpenGL约定 (已翻转)
   pred_trans/pred_rot → OpenCV约定 (未翻转)
   要正确使用需要: verts = R_x @ run_mano(pred_trans, pred_rot, ...)['vertices']
```

#### ReplicateAnyScene 的坐标系变换链

```
① VGGT模型输出
   坐标系: VGGT内部世界系 (原点不确定, 但尺度正确)
   数据: extrinsic(S,3,4) cam-from-world, intrinsic(S,3,3)

② 深度反投影
   坐标系: 同① (VGGT世界系)
   变换: depth → cam_coords → world_coords = cam_coords @ R_c2w.T + t_c2w

③ 房间对齐 (align_to_room_coordinate_system)
   输入: world_points + wall_masks + floor_masks
   算法:
     a. PCA拟合墙面/地面法向量
     b. floor_normal → z轴 (朝上)
     c. wall_normal_1 → x轴 (沿一面墙)
     d. wall_normal_2 = cross(z, x) → y轴
     e. R_room = [wall_normal_1; wall_normal_2; floor_normal] (3×3)
     f. t_room: 地板z=0, 场景bbox中心xy=0
   变换:
     world_points_room = world_points_vggt @ R_room.T + t_room
     R_c2w_new = R_c2w_old @ R_room.T
     t_c2w_new = t_c2w_old - R_c2w_new @ t_room
   坐标系: 房间坐标系 (z-up: X沿墙, Y沿墙, Z朝上)

④ SAM3D资产生成
   坐标系变换链 (最复杂):
     SAM3D局部(y-up) → y2z变换 → SAM3D相机 → VGGT相机 → VGGT世界 → 房间
     T_final = extrinsic⁻¹ @ M_adjust @ T_l2c @ M_y2z
     其中 M_adjust = diag(-1,-1,1,1) 是SAM3D相机系→VGGT相机系的修正
   坐标系: 房间坐标系 (z-up)

⑤ GLB导出
   变换: z-up → y-up
   [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]
   坐标系: GLB标准 (y-up)
```

#### 坐标系差异汇总

| 坐标系 | X轴 | Y轴 | Z轴 | HaWoR用 | RAS用 |
|--------|-----|-----|-----|---------|-------|
| OpenCV相机 | 右 | 下 | 前(光轴) | ✅ HAWOR模型输出 | ✅ VGGT内部 |
| OpenGL相机 | 右 | 上 | 后 | ✅ demov2翻转后 | ❌ |
| SLAM世界 | 右 | 下 | 前 | ✅ cam2world后 | ❌ |
| VGGT世界 | 不确定 | 不确定 | 不确定 | ❌ | ✅ VGGT输出 |
| 房间坐标系 | 沿墙1 | 沿墙2 | 上 | ❌ | ✅ 对齐后 |
| GLB标准 | 右 | 上 | 前 | ❌ | ✅ 最终导出 |

**合并时必须解决的坐标系问题：**

```
问题1: HaWoR 的 cam2world_convert 期望 R_c2w 在 SLAM世界系 (OpenCV: Y下, Z前)
       VGGT 的 extrinsics 在 VGGT世界系 (不确定方向)
       → 房间对齐后, extrinsics 在房间系 (z-up: Z朝上)
       → 房间系不是 OpenCV 约定! R_c2w 的 Z 轴可能朝上而非朝前

问题2: HaWoR 的 R_x 翻转 (OpenCV→OpenGL) 在合并后是否还需要?
       → 如果统一用房间坐标系 (z-up), 不需要 R_x 翻转
       → 但 MANO FK 输出的顶点在哪个坐标系? 需要明确

问题3: cam2world_convert 中的 offset 处理
       offset = init_trans - root_loc  (相机空间, 未旋转)
       这个偏移量在房间坐标系中是否仍然正确?
       → 是的, 因为 offset 是 MANO 参数空间的偏移, 与世界坐标系无关
```

### 2.6 合并后的统一坐标系方案

```
统一坐标系: 房间坐标系 (z-up)

VGGT+VGGT4D 输出 → 房间对齐 → SharedSceneData (房间坐标系)
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
               HaWoR cam2world      RAS 场景重建           GLB导出
               (不再需要R_x翻转)    (已在房间坐标系)       (z-up→y-up)

HaWoR 的 cam2world_convert 修改:
  旧: R_root_world = R_c2w_slam @ R_root_cam        (SLAM世界, OpenCV)
      t_world = R_c2w_slam @ root_loc + t_c2w_slam
  新: R_root_world = R_c2w_room @ R_root_cam        (房间世界, z-up)
      t_world = R_c2w_room @ root_loc + t_c2w_room
  → 数学形式完全相同, 只是 R_c2w/t_c2w 的来源变了
  → 不需要 R_x 翻转! 因为房间坐标系已经是 z-up (Y朝上)

MANO FK 输出:
  旧: verts 在 SLAM世界 (OpenCV: Y下, Z前), 需要 R_x 翻转
  新: verts 在 房间世界 (z-up: Z朝上), 直接可用, 不需要翻转
```

---

## 三、当前架构痛点的核心矛盾

```
现状：两套独立的 3D 感知链路

HaWoR:
  DROID-SLAM  →  traj (无尺度)       →  需要 Metric3D 估计 scale
  Metric3D    →  pred_depth (度量)    →  est_scale_hybrid()
  → traj × scale  →  R_c2w, t_c2w    →  cam2world_convert()
  → MANO mesh (世界空间)

ReplicateAnyScene:
  VGGT  →  depth + extrinsics + intrinsic
  →  align_to_room_coordinate_system()  →  房间坐标系
  →  SAM3分割 → SAM3D资产 → 空间精修
  →  GLB导出

问题:
  1. 两套坐标系：SLAM世界 ≠ VGGT世界（原点、方向、尺度都不同）
  2. 两套相机参数：单focal+中心主点 ≠ 完整3×3内参
  3. 两套深度：DROID-SLAM视差(无量纲) ≠ VGGT深度(度量)
  4. HaWoR 需要两阶段（SLAM + Metric3D），ReplicateAnyScene 一阶段（VGGT）
  5. 动态手导致点云鬼影，两个管线都受影响
```

---

## 四、合并策略：VGGT + VGGT4D 作为统一 3D 感知层

### 4.1 为什么选 VGGT + VGGT4D 而不是 VGGT-Omega？

| 维度 | VGGT-Omega | VGGT + VGGT4D | 选择理由 |
|------|------------|---------------|---------|
| world_points | 需要深度反投影 | **直接输出** (PointHead) | 更准，少一步误差 |
| 动态 mask | 无，需额外模型 | **VGGT4D 从 attention 层提取** | Training-free，零额外模型 |
| TrackHead | 无 | **有** (3D点追踪) | 物体追踪更准 |
| 位姿精度 | 单次前向 | **两次前向** (第二次屏蔽动态区域) | 动态场景位姿更准 |
| 参数量 | ~560M | ~700M | 多 140M 换来 PointHead + TrackHead + 动态mask |

**核心优势**：VGGT4D 的动态 mask 解决了两个管线共同的痛点——手部运动干扰场景重建。同时 VGGT 原版直接输出 world_points 和 TrackHead，省去了反投影和 Procrustes 对齐的额外步骤。

### 4.2 VGGT + VGGT4D 替代 DROID-SLAM + Metric3D 的理由

```
旧方案（两阶段）:
  DROID-SLAM  →  无尺度轨迹
  Metric3D    →  度量深度
  est_scale_hybrid()  →  尺度因子  (深度/位姿来自不同模型，可能不一致)
  下游: cam2world_convert(R_c2w×scale, t_c2w×scale)

新方案（VGGT4D 三阶段，training-free）:
  Stage 1: VGGT forward (不带 dyn_masks)
    → depth + extrinsics + intrinsic + world_points + Q/K attention maps
  Stage 2: VGGT4D 从 Q/K 提取动态 mask
    → 5种注意力模式融合 + KMeans去噪 + Otsu二值化
  Stage 3: VGGT forward (带 dyn_masks，屏蔽动态区域)
    → 修正后的 extrinsics (位姿更准)
    → depth 沿用 Stage 1
  → depth和位姿来自同一模型前向，天然一致
  → 动态mask同时服务 HaWoR 和 ReplicateAnyScene
  下游: cam2world_convert(R_c2w_vggt, t_c2w_vggt)  ← 完全相同的接口！
```

### 4.3 VGGT 与 VGGT4D 如何加载在一起

VGGT4D 不是独立模型，它是对 VGGT 的**推理时增强**。两者共享同一份权重，通过继承替换的方式集成：

```
加载关系:

  VGGT 原始权重 (model_tracker_fixed_e20.pt)
       │
       ▼
  VGGTFor4D (继承关系，不修改原始 VGGT 代码):
    ├─ VGGTFor4D.models.VGGTFor4D       替代 VGGT.models.VGGT
    │   └─ forward() 返回 (predictions, qk_dict, enc_feat, agg_tokens_list)
    │
    ├─ AggregatorFor4D                   替代 VGGT 的 Aggregator
    │   └─ 收集每层 Q/K 向量 + 支持 dyn_masks 参数
    │
    ├─ BlockFor4D                        替代 VGGT 的 Block
    │   └─ forward() 额外返回 Q, K
    │
    └─ AttentionFor4D                    替代 VGGT 的 Attention
        └─ attention_with_dynamic_mask() 在 global attention 前5层屏蔽动态token
```

**关键：VGGT4D 使用原始 VGGT 的预训练权重，不需要额外训练。**

```python
# 加载代码
from vggt4d.models.vggt4d import VGGTFor4D

model = VGGTFor4D()  # 内部结构和 VGGT 完全一致
model.load_state_dict(torch.load("model_tracker_fixed_e20.pt"))  # 同一份权重!
model.eval().cuda()
```

### 4.4 完整推理流程（VGGT + VGGT4D 三阶段）

```python
# ===== Stage 1: 第一次前向（不带动态mask）=====
with torch.no_grad():
    predictions, qk_dict, enc_feat, _ = model(images)

# 位姿解码
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"], images.shape[-2:]
)
# VGGT 直接输出 world_points，不需要反投影
world_points = predictions["world_points"]          # (S, H, W, 3) 直接输出!
world_points_conf = predictions["world_points_conf"] # (S, H, W)

# ===== Stage 2: VGGT4D 提取动态 mask =====
from vggt4d.utils.model_utils import organize_qk_dict
from vggt4d.masks.dynamic_mask import extract_dyn_map, cluster_attention_maps, adaptive_multiotsu_variance

qk_dict = organize_qk_dict(qk_dict, n_img=images.shape[0])
dyn_maps = extract_dyn_map(qk_dict, images)                    # 5种注意力模式融合
norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)    # KMeans语义去噪
upsampled_map = F.interpolate(norm_dyn_map, size=(H, W), ...)   # 上采样到原图
thres = adaptive_multiotsu_variance(upsampled_map)              # 自适应Otsu二值化
dyn_masks = upsampled_map > thres                               # (S, H, W) bool

# ===== Stage 3: 第二次前向（带动态mask，修正位姿）=====
with torch.no_grad():
    predictions2, _, _, _ = model(images, dyn_masks=dyn_masks.to(device))

# 仅更新位姿，深度图沿用 Stage 1
extrinsic2, intrinsic2 = pose_encoding_to_extri_intri(
    predictions2["pose_enc"], images.shape[-2:]
)
# predictions2["depth"] 沿用 Stage 1，不需要重新计算
```

### 4.5 数据流

```
                         输入视频 (MP4 / 图片序列)
                                    │
         ┌──────────────────────────┼───────────────────────────┐
         ▼                          ▼                           ▼
    YOLO 手部检测          VGGT + VGGT4D                     SAM3 分割
    (手部 bbox)       ├─ depth (S,H,W)                    ├─ wall_masks
         │            ├─ extrinsics (S,4,4) c2w (修正后)   ├─ floor_masks
         │            ├─ intrinsic (3×3)                   └─ object_masks
         │            ├─ world_points (S,H,W,3) ← 直接输出!      │
         │            ├─ dynamic_mask (S,H,W)  ← VGGT4D新增!     │
         │            └─ tracks (S,N,2+3)     ← TrackHead可选    │
         ▼                          │                            │
    HaWoR                ┌──────────▼──────────┐                 │
    位姿估计              │  房间坐标系对齐       │                 │
    (相机空间)            │  align_to_room()     │                 │
         │               │  R_room, t_room      │                 │
         │               └──────────┬──────────┘                 │
         │                          │                            │
         │               ┌──────────▼──────────┐                 │
         │               │   SharedSceneData    │◄────────────────┘
         │               │   (统一世界坐标)      │
         │               │   + dynamic_mask     │  ← 新增!
         │               └──────────┬──────────┘
         │                          │
         │             ┌────────────┼────────────┐
         │             ▼            ▼            ▼
         │       dynamic_mask   场景重建      统一 GLB 导出
         │       过滤动态区域   (更干净!)     (物体 + 手)
         │             │            │              ▲
         │             │    ┌───────┘              │
         │             │    │                      │
         │             ▼    ▼                      │
         │         SAM3D 资产生成                  │
         │             │                          │
         └── cam2world─┘                          │
              (extr)                               │
                 │                                 │
                 ▼                                 │
              MANO FK ─────────────────────────────┘
              (世界空间手部mesh)
```

**dynamic_mask 的三个消费方**：

| 消费方 | 用途 |
|--------|------|
| ReplicateAnyScene | 过滤动态区域 → 静态场景点云更干净（无鬼影） |
| HaWoR | 交叉验证 YOLO 手部检测 → 手部 mask 更准 |
| object_tracker | 区分手部点云 vs 物体点云 → 接触检测更准 |

---

## 五、世界坐标系约定

### 5.1 内部运算坐标系（房间坐标系）

```
右手系
z-up  (地面法线 = +Z)
floor z = 0
scene bbox 中心 xy = 0

R_c2w @ p_cam + t_c2w = p_world
p_cam = R_c2w^T @ (p_world - t_c2w)
```

### 5.2 导出坐标系（GLB 标准）

```
y-up (行业标准)
导出时: z-up → y-up

transform = [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0,-1, 0, 0],
             [0, 0, 0, 1]]
```

### 5.3 坐标系命名约定

| 术语 | 含义 | 示例 |
|------|------|------|
| `c2w` (camera-to-world) | 相机坐标 → 世界坐标 | `p_world = R_c2w @ p_cam + t_c2w` |
| `w2c` (world-to-camera) | 世界坐标 → 相机坐标 | `p_cam = R_w2c @ p_world + t_w2c` |
| `extrinsics` | 4×4 矩阵，不含最后一行的 c2w | `[R_c2w, t_c2w; 0,0,0,1]` |

---

## 六、核心数据结构

### 6.1 SharedSceneData（所有模块共享）

```python
@dataclass
class SharedSceneData:
    """合并后所有模块共用的世界场景数据"""
    source_model: str = "vggt_vggt4d"  # VGGT + VGGT4D

    # 图像
    colors: np.ndarray            # (S, H, W, 3) uint8 RGB
    num_frames: int               # S
    height: int                   # H
    width: int                    # W

    # 相机 (VGGT → room-aligned, VGGT4D 修正后)
    intrinsic: np.ndarray         # (3, 3)
    extrinsics: np.ndarray        # (S, 4, 4)  room-aligned c2w

    # 3D (VGGT → room-aligned)
    world_points: np.ndarray      # (S, H, W, 3)  ← VGGT PointHead 直接输出
    world_points_conf: np.ndarray # (S, H, W)
    depths: np.ndarray            # (S, H, W)
    point_cloud_data: trimesh.PointCloud

    # 动态 mask (VGGT4D 新增)
    dynamic_mask: np.ndarray      # (S, H, W) bool, True=动态区域

    # 房间参数
    R_room: np.ndarray            # (3, 3)  VGGT原始 → 房间坐标的旋转
    t_room: np.ndarray            # (3,)    VGGT原始 → 房间坐标的平移
    walls_info: list              # 墙面信息 (法线、点)

    # --- 便捷方法 ---
    def get_focal(self) -> float:
        return float(self.intrinsic[0, 0])

    def get_principal_point(self) -> tuple:
        return (float(self.intrinsic[0, 2]), float(self.intrinsic[1, 2]))

    def get_c2w_Rt(self, frame_idx: int) -> tuple:
        """返回该帧 (R_c2w 3×3, t_c2w 3,) 用于 cam2world_convert"""
        ext = self.extrinsics[frame_idx]
        return ext[:3, :3], ext[:3, 3]

    def get_w2c_Rt(self, frame_idx: int) -> tuple:
        """返回该帧 (R_w2c 3×3, t_w2c 3,) 用于投影"""
        R = self.extrinsics[frame_idx, :3, :3]
        t = self.extrinsics[frame_idx, :3, 3]
        R_w2c = R.T
        t_w2c = -R_w2c @ t
        return R_w2c, t_w2c

    def get_static_world_points(self, frame_idx: int) -> np.ndarray:
        """返回该帧的静态区域点云 (过滤掉 dynamic_mask 区域)"""
        wp = self.world_points[frame_idx]       # (H, W, 3)
        dm = self.dynamic_mask[frame_idx]       # (H, W)
        wp[dm] = 0                              # 动态区域置零
        return wp
```

---

## 七、接口改造清单

### 7.1 新增文件

| 文件 | 职责 |
|------|------|
| `src/shared_scene.py` | `SharedSceneData` 数据类定义 |
| `src/vggt4d_predict.py` | VGGT + VGGT4D 封装，三阶段推理，输出统一 dict |
| `main_merged.py` | 统一入口，编排 HaWoR + ReplicateAnyScene |

### 7.2 修改文件

| 文件 | 修改内容 |
|------|---------|
| `HaWoR/lib/eval_utils/custom_utils.py` | 新增 `load_cam_from_shared()` 函数 |
| `HaWoR/scripts/scripts_test_video/hawor_video.py` | `hawor_motion_estimation` + `hawor_infiller` 增加 `shared` 参数 |
| `HaWoR/demo.py` | 去掉 `hawor_slam` 调用，改用 VGGT+VGGT4D |
| `HaWoR/demov2.py` | 去掉 `R_x = diag(1,-1,-1)` 翻转 |

### 7.3 废弃文件

| 文件 | 原因 |
|------|------|
| `HaWoR/scripts/scripts_test_video/hawor_slam.py` | DROID-SLAM 由 VGGT+VGGT4D 替代 |

### 7.4 不改动的文件

| 文件 | 原因 |
|------|------|
| `ReplicateAnyScene/main.py` | 保留原路径，`main_merged.py` 作为新入口 |
| `HaWoR/lib/eval_utils/custom_utils.py` 的 `cam2world_convert` | c2w 语义不变，无论来源 |
| `HaWoR/lib/models/hawor.py` | 手部位姿估计不关心相机来源 |
| `ReplicateAnyScene/src/geometry_utils.py` | 房间对齐逻辑不变 |

---

## 八、关键代码对接点

### 8.1 VGGT + VGGT4D 输出 → SharedSceneData

```python
# vggt4d_predict.py
from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.utils.model_utils import organize_qk_dict
from vggt4d.masks.dynamic_mask import extract_dyn_map, cluster_attention_maps, adaptive_multiotsu_variance
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

model = VGGTFor4D().eval().cuda()
model.load_state_dict(torch.load("model_tracker_fixed_e20.pt"))

images = load_and_preprocess_images(imgfiles, image_resolution=518).cuda()

# Stage 1: 不带动态mask
predictions, qk_dict, enc_feat, _ = model(images)

# 位姿解码 (VGGT 原版接口)
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    images.shape[-2:],
)

# VGGT 直接输出 world_points (不需要反投影!)
world_points = predictions["world_points"]          # (S, H, W, 3)
world_points_conf = predictions["world_points_conf"] # (S, H, W)

# Stage 2: VGGT4D 提取动态 mask
qk_dict = organize_qk_dict(qk_dict, images.shape[0])
dyn_maps = extract_dyn_map(qk_dict, images)
norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)
upsampled_map = F.interpolate(norm_dyn_map.float(), size=(H, W), mode='bilinear')
thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
dyn_masks = upsampled_map > thres  # (S, H, W) bool

# Stage 3: 带动态mask修正位姿
predictions2, _, _, _ = model(images, dyn_masks=dyn_masks.cuda())
extrinsic2, intrinsic2 = pose_encoding_to_extri_intri(
    predictions2["pose_enc"], images.shape[-2:]
)

# 转 4×4
ext_4x4 = np.zeros((S, 4, 4))
ext_4x4[:, :3, :4] = extrinsic2
ext_4x4[:, 3, 3] = 1.0

# 输出 dict (统一格式)
vggt_results = {
    "depths": predictions["depth"][..., 0].cpu().numpy(),  # Stage 1 深度
    "extrinsics": ext_4x4,                                  # Stage 3 修正位姿
    "intrinsic": intrinsic2.cpu().numpy(),
    "world_points": world_points.cpu().numpy(),              # Stage 1 点云
    "world_points_conf": world_points_conf.cpu().numpy(),
    "dynamic_mask": dyn_masks.cpu().numpy(),                 # VGGT4D 动态mask
    "colors": (images.cpu().numpy() * 255).astype(np.uint8).transpose(0,2,3,1),
}
```

### 8.2 SharedSceneData → HaWoR cam2world

```python
# custom_utils.py (新增)
def load_cam_from_shared(shared: SharedSceneData, frame_indices):
    """从 SharedSceneData 获取相机外参，替代 load_slam_cam()"""
    ext = shared.extrinsics[frame_indices]
    R_c2w = torch.tensor(ext[:, :3, :3])
    t_c2w = torch.tensor(ext[:, :3, 3])
    R_w2c = R_c2w.transpose(-1, -2)
    t_w2c = -torch.einsum("bij,bj->bi", R_w2c, t_c2w)
    return R_w2c, t_w2c, R_c2w, t_c2w

# hawor_infiller (修改)
# 旧: R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam("SLAM/...npz")
# 新: R_w2c, t_w2c, R_c2w, t_c2w = load_cam_from_shared(shared, frames)

# cam2world_convert 不变
data_world = cam2world_convert(R_c2w, t_c2w, data_out, handedness)
# → 手部参数自动进入房间世界坐标系
```

### 8.3 SharedSceneData → ReplicateAnyScene (使用动态mask)

```python
# ReplicateAnyScene 中使用动态mask过滤点云
def get_static_point_cloud(shared: SharedSceneData, frame_idx: int):
    """获取过滤掉动态区域后的静态点云"""
    wp = shared.world_points[frame_idx]       # (H, W, 3)
    conf = shared.world_points_conf[frame_idx] # (H, W)
    dm = shared.dynamic_mask[frame_idx]       # (H, W)

    # 动态区域 (手) 的点云置零
    wp_filtered = wp.copy()
    wp_filtered[dm] = 0
    conf_filtered = conf.copy()
    conf_filtered[dm] = 0

    return wp_filtered, conf_filtered
```

### 8.4 HaWoR 相机内参 → 使用 SharedSceneData

```python
# hawor_motion_estimation (修改签名)
def hawor_motion_estimation(args, start_idx, end_idx, seq_folder,
                             shared: SharedSceneData = None):
    if shared is not None:
        img_focal = shared.get_focal()
        img_center = shared.get_principal_point()
    else:
        img_focal = args.img_focal
        img_center = [W / 2, H / 2]
```

---

## 九、Merged Pipeline 完整流程

```python
# ==================== main_merged.py ====================

# 0. VGGT + VGGT4D 3D感知 (共享前端，三阶段)
vggt_results = vggt4d_predict(images, model)
# 输出: depth, extrinsics, intrinsic, world_points, dynamic_mask, colors

# 0.5 房间坐标系对齐
wall_masks, floor_masks = segment_wall_and_floor(colors, sam3_image)
R_room, t_room       = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
aligned              = align_vggt_predictions(vggt_results, R_room, t_room)
shared = SharedSceneData(
    colors=aligned['colors'], intrinsic=aligned['intrinsic'],
    extrinsics=aligned['extrinsics'], world_points=aligned['world_points'],
    depths=aligned['depths'], dynamic_mask=aligned['dynamic_mask'],
    ...,
    R_room=R_room, t_room=t_room,
)

# 1. HaWoR 动态手部重建
hand_detections = detect_track_video(args)
cam_space_poses = hawor_motion_estimation(args, shared=shared)
# cam2world: 相机→房间坐标系 (用 VGGT4D 修正后的 extrinsics)
data_world = hawor_infiller(args, shared=shared)

# 2. ReplicateAnyScene 静态场景重建 (使用 dynamic_mask 过滤手部)
object_masks     = segment_and_track(categories, sam3_video)
dedup_masks      = cross_category_deduplicate(object_masks, shared)
optimal_views    = get_optimal_view_frame_id(shared.world_points, dedup_masks,
                                              dynamic_mask=shared.dynamic_mask)  # 过滤动态区域
scene_instances  = generate_3d_asset_in_subprocess(dedup_masks, optimal_views, ...)
scene_instances  = verify_all_instances(scene_instances, ...)
scene_instances  = refine_object_placement(scene_instances, shared.walls_info)

# 3. 统一 GLB 导出
scene = trimesh.Scene()
for inst in scene_instances:
    scene.add_geometry(inst['mesh'], transform=inst['T'])
for hand in hand_meshes:
    scene.add_geometry(hand['mesh'])
scene.apply_transform(z_up_to_y_up)
scene.export("final_scene.glb")
```

---

## 十、坐标系转换链（单帧数据流）

```
步骤1: VGGT + VGGT4D 原始输出
  depth_vggt:        (H, W)      ← 度量深度 (Stage 1)
  extrinsics_vggt:   (4, 4)      ← c2w, VGGT4D 修正后 (Stage 3)
  intrinsic_vggt:     (3, 3)
  world_points_vggt: (H, W, 3)   ← VGGT PointHead 直接输出 (Stage 1)
  dynamic_mask:      (H, W)      ← VGGT4D 提取 (Stage 2)

步骤2: 房间对齐
  R_room, t_room = align_to_room(...)
  world_points_room = (world_points_vggt - t_room) @ R_room.T
  extrinsics_room = [[R_c2w @ R_room.T, t - R_new @ t_room],
                     [0, 0, 0, 1]]
  dynamic_mask 不变 (像素空间，与坐标系无关)

步骤3: HaWoR 手部
  p_cam   = HaWoR 估计的相机空间手部坐标
  p_world = R_c2w_room @ p_cam + t_c2w_room  ← 在房间坐标系中
  → MANO FK → 手部 mesh 顶点在房间坐标系

步骤4: ReplicateAnyScene 物体
  p_world = SAM3D 生成的 mesh (VGGT坐标)
  p_room  = (p_world - t_room) @ R_room.T      ← 在房间坐标系中
  (使用 dynamic_mask 过滤手部区域，点云更干净)

步骤5: GLB 导出
  p_glb   = [[1, 0, 0, 0], [0, 0, 1, 0], [0,-1, 0, 0], [0, 0, 0, 1]] @ p_room
  → z-up → y-up, 输出标准 GLB
```

---

## 十一、实施路线

| 阶段 | 内容 | 验证方法 | 预计工作量 |
|------|------|---------|:---:|
| **P0: 验证** | VGGT+VGGT4D 对比 DROID-SLAM，对同一段视频比较 c2w 轨迹 | 轨迹可视化、重投影误差 | 1天 |
| **P1: 封装** | 创建 `vggt4d_predict.py`，三阶段推理统一输出格式 | 与旧 VGGT 输出结构比较 | 1天 |
| **P2: 共享层** | `SharedSceneData` + 房间对齐，跑通静态场景重建 | ReplicateAnyScene 的 GLB 质量不退化 | 1天 |
| **P3: 动态mask** | VGGT4D dynamic_mask 接入两个管线 | 点云鬼影减少、手部mask更准 | 1天 |
| **P4: 接入手部** | `load_cam_from_shared` + `hawor_infiller` 改造 | 手部 mesh 位置合理（与场景不穿越） | 1天 |
| **P5: 合并入口** | `main_merged.py`，端到端生成 `final_scene.glb` | 可视化检查手+场景一致性 | 半天 |

---

## 十二、风险与缓解

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| VGGT+VGGT4D 对第一人称视频位姿估计不准 | 手部在世界空间位置错误 | P0 对比验证，如果不达标回退到 DROID-SLAM |
| VGGT4D 动态mask误检（把手部区域漏检或把静态区域误检为动态） | 场景点云缺失或手部mask不准 | 调整 Otsu 阈值 + YOLO 手部检测交叉验证 |
| VGGT 两次前向推理速度慢 | 总推理时间翻倍 | Stage 3 只需更新位姿，可选择性跳过 |
| 世界坐标原点不一致（HaWoR 的相机空间 vs 房间原点） | 手和场景分离 | `cam2world_convert` 传递 room-aligned c2w 即可，数学上等价 |
| VGGT4D 代码依赖 VGGT 原版 (非 Omega) | 需要同时维护两套 VGGT | 统一使用 VGGT 原版，Omega 作为备选 |

---

## 十三、总结

> **用 VGGT + VGGT4D 作为共享 3D 感知层，两个项目的核心算法（HaWoR 手部位姿、ReplicateAnyScene 场景重建）不动，只改相机来源接口。**
>
> - **DROID-SLAM + Metric3D 删除**，由 VGGT+VGGT4D 统一提供相机位姿 + 深度 + 动态mask
> - **VGGT4D 是 training-free 的**，使用原始 VGGT 权重，通过继承替换集成，不需要额外训练
> - **新增 SharedSceneData** 作为所有模块数据交换的「通用语言」，包含 dynamic_mask
> - **房间对齐后的坐标系** 作为唯一的「真相世界」
> - **cam2world_convert、MANO FK、SAM3D、精修** 全部不需要改
> - **dynamic_mask 同时服务两个管线**：ReplicateAnyScene 过滤鬼影，HaWoR 交叉验证手部

最终效果：一段视频输入 → 一个包含墙壁、地面、家具、双手的 3D 场景 GLB 文件，无鬼影，手部与场景对齐。

---

## 十四、实测数据：7.mp4 的 SLAM vs VGGT 相机外参对比

### 14.1 测试条件

```
视频: 7.mp4 (3.77秒, 30fps, 113帧)
SLAM: DROID-SLAM + Metric3D (HaWoR 管线)
VGGT: VGGT-Omega (ReplicateAnyScene 管线, sample_fps=2.0 → 20帧)
```

### 14.2 坐标系方向

```
SLAM 平均Y轴: [0.006, 0.998, -0.059]  → Y朝上 = OpenGL
VGGT 平均Y轴: [-0.005, 0.999, 0.041]  → Y朝上 = OpenGL
→ 两者坐标系方向一致! 都是 OpenGL 约定
```

### 14.3 第一帧

```
SLAM R_c2w[0]:  近似单位矩阵 (偏差<0.01)
VGGT ext[0]:    近似单位矩阵 (偏差<0.001)
→ 第0帧两者都把相机放在原点, 朝向Z轴正方向
```

### 14.4 旋转差异

```
VGGT帧 0 ↔ SLAM帧  0: 0.46°   ← 起点一致
VGGT帧 4 ↔ SLAM帧 22: 7.00°   ← 开始偏离
VGGT帧 8 ↔ SLAM帧 45: 9.18°   ← 最大偏差!
VGGT帧12 ↔ SLAM帧 67: 7.55°
VGGT帧16 ↔ SLAM帧 90: 7.56°
VGGT帧19 ↔ SLAM帧107: 5.41°   ← 末段反而收敛
```

9° 旋转差异在 1 米距离上 → 15cm 位置偏差。SLAM 增量估计导致中间帧累积误差，VGGT 全局优化更均匀。

### 14.5 平移差异

```
SLAM: X变化0.028m, Y变化0.017m, Z变化0.026m  ← 总运动~3cm
VGGT: X变化0.084m, Y变化0.023m, Z变化0.066m  ← 总运动~8cm
→ VGGT/SLAM ≈ 2.8x 尺度差异
```

SLAM 的尺度由 Metric3D 对齐决定 (scale=0.322)，VGGT 自带度量尺度。

### 14.6 SLAM Y轴漂移分析

```
7.mp4 (3.7秒, 短视频):
  frame   0: Y=0.004m, pitch=-0.4°
  frame  50: Y=-0.001m, pitch=-5.2°
  frame 110: Y=-0.010m, pitch=-1.6°
  → Y变化1.6cm, 正常低头/抬头, 无漂移

beizi.mp4 (6秒, 较长视频):
  frame   0: Y=-0.0004m
  frame 100: Y=0.1715m  ← 漂移17cm!
  frame 150: Y=0.4984m  ← 漂移50cm!
  → pitch只有3~5°, 不是真实运动, 是SLAM累积漂移
```

**结论**: SLAM 在短视频(3-4秒)中漂移可控，在6秒+视频中漂移严重。VGGT 全局优化无此问题。

### 14.7 VGGT 降采样问题

113帧变20帧的原因: `demo_ply.py` 默认 `--sample-fps 2.0`，按FPS间隔抽帧。

```python
# demo_ply.py: extract_frames()
frame_interval = round(30 / 2.0) = 15  # 每15帧取1帧
```

**解决**: 用 `predict.py` 代替 `demo_ply.py`，或增大 `--sample-fps`：

```bash
python predict.py --video 7.mp4 --max_frames 160  # 113帧不会被降采样
python demo_ply.py --video 7.mp4 --sample-fps 10  # 30fps/10=3, 113/3≈38帧
```

### 14.8 核心差异总结

| 维度 | SLAM | VGGT | 差异性质 |
|------|:---:|:---:|------|
| 坐标系 | OpenGL (Y上) | OpenGL (Y上) | ✅ 一致 |
| 第0帧 | ≈ 单位矩阵 | ≈ 单位矩阵 | ✅ 一致 |
| 旋转 | 增量式, 中间帧5~9°偏差 | 全局优化, 更均匀 | ❌ 估计方式不同 |
| 尺度 | 3cm (scale=0.322) | 8cm (自带度量) | ❌ 尺度差2.8倍 |
| Y轴漂移 | 短视频OK, 长视频严重 | 无漂移 | ❌ SLAM缺陷 |
| 帧覆盖 | 113帧 (全帧) | 20帧 (降采样) | ❌ 可修复 |
| focal | 600px (原图1920×1080) | 236px (resize到518×294) | ⚠️ 分辨率不同, 非本质差异 |

---

## 十五、物理仿真方案：从视频到仿真复现

### 15.1 目标

将合并管线的输出（场景 mesh + 手部轨迹 + 物体轨迹 + 接触信息）导入 GalaxeaManipSim 的 SAPIEN/PhysX 物理引擎，实现**物理驱动的动作复刻**——机械臂按手部轨迹运动，物体由物理引擎计算被动响应。

### 15.2 核心原则

```
机械臂是主动方: 轨迹来自 HaWoR 手部重建 + UniDex 重定向
物体是被动方:   运动由物理引擎计算 (夹爪夹住→物体跟着走)
物体轨迹仅用于验证: 仿真结果 vs VGGT追踪轨迹 → 偏差大则调整参数
```

### 15.3 完整数据流

```
┌─────────────────────────────────────────────────────────┐
│                    输入: 第一人称操作视频                   │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
         VGGT+VGGT4D    HaWoR           SAM3
         (3D感知)       (手部重建)       (场景分割)
              │            │                │
              ▼            ▼                ▼
         统一坐标系 (房间坐标系, z-up)
              │            │                │
              ├────────────┼────────────────┤
              │            │                │
              ▼            ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ 静态场景 mesh │ │ 手部3D轨迹   │ │ 物体3D mesh  │
    │ (墙/地/桌面) │ │ (MANO FK)    │ │ + 初始6DoF   │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────────────────────────────────────────┐
    │           物理仿真场景构建 (SAPIEN)            │
    │                                              │
    │  1. 导入静态场景 → 静态碰撞体                  │
    │  2. 导入物体mesh → 动态刚体 (有质量/摩擦力)     │
    │  3. 放置机器人 (R1Lite)                       │
    │  4. 手部轨迹 → 机械臂EE轨迹 (UniDex重定向)     │
    │  5. 接触检测 → 夹爪开闭时序                    │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │           物理仿真执行 (SAPIEN/PhysX)          │
    │                                              │
    │  for t in range(total_frames):               │
    │    1. 设置机械臂目标EE位姿 (来自手部轨迹)       │
    │    2. 设置夹爪开/闭 (来自接触检测)              │
    │    3. scene.step() ← 物理引擎计算!             │
    │    4. 读取物体实际位姿 (验证用)                 │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │           仿真结果验证                          │
    │                                              │
    │  仿真物体轨迹 vs VGGT追踪物体轨迹               │
    │  → 偏差大 → 调整物理参数/抓取姿态               │
    │  → 偏差小 → 仿真成功!                          │
    └──────────────────────────────────────────────┘
```

### 15.4 GalaxeaManipSim 关键接口

| 功能 | 接口 | 说明 |
|------|------|------|
| 创建场景 | `sapien.Scene()` + `scene.add_ground(0)` | 添加地面 |
| 加载机器人 | `R1LiteRobot(scene)` | URDF 加载, 20自由度 |
| 创建物体 | `create_glb(scene, pose, modelname, is_static=False)` | 动态刚体 |
| 创建静态场景 | `create_glb(scene, pose, modelname, is_static=True)` | 静态碰撞体 |
| EE位姿控制 | `BimanualEEPoseController` action dim=16 | [左EE位置3, 左EE四元数4, 左夹爪1, 右EE位置3, 右EE四元数4, 右夹爪1] |
| 夹爪控制 | action 中第8位(左)和第16位(右) | 0.0=闭合, 0.05=张开 |
| 步进仿真 | `env.step(action)` | 内部 decimation×scene.step() |
| 读取物体位姿 | `actor.get_pose()` → `.p`(位置) + `.q`(四元数) | 验证用 |
| 读取EE位姿 | `robot.left_ee_link.get_entity_pose()` | 验证用 |

### 15.5 坐标系转换：房间坐标系 → SAPIEN 坐标系

```
房间坐标系 (合并管线输出):
  z-up, Y沿墙, X沿另一面墙
  单位: 米

SAPIEN 坐标系:
  y-up, X右, Z前
  单位: 米

转换:
  T_room_to_sapien = [[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0,-1, 0, 0],
                       [0, 0, 0, 1]]

  p_sapien = T_room_to_sapien @ p_room
  R_sapien = T_room_to_sapien[:3,:3] @ R_room @ T_room_to_sapien[:3,:3].T
```

### 15.6 手部轨迹 → 机械臂EE轨迹 (UniDex 重定向)

```
HaWoR 输出:
  pred_trans: (2, T, 3)  ← 左右手腕位置 (房间坐标系)
  pred_rot: (2, T, 3)    ← 左右手腕旋转 (axis-angle, 房间坐标系)

UniDex 重定向:
  MANO 手腕 → R1Lite 机械臂末端执行器
  输入: MANO 手腕位姿 (位置+旋转)
  输出: 机械臂 EE 位姿 (位置+四元数)

  # 坐标系转换
  ee_pose_room = mano_to_ee(mano_pose)            # MANO→EE (UniDex)
  ee_pose_sapien = T_room_to_sapien @ ee_pose_room  # 房间→SAPIEN
```

### 15.7 夹爪开闭时序 (来自 contact_detector)

```
contact_detector 输出:
  gripper_timeline: [
    {"frame": 0, "state": "open", "object": null},
    {"frame": 14, "state": "closed", "object": "cup_inst0"},
    {"frame": 46, "state": "open", "object": "cup_inst0"},
  ]

转换为 action:
  gripper_value = 0.05  # 张开
  gripper_value = 0.0   # 闭合

  for t in range(total_frames):
    if t in contact_frames:
      gripper = 0.0  # 闭合
    else:
      gripper = 0.05  # 张开
```

### 15.8 实施路线

| 阶段 | 内容 | 验证方法 |
|------|------|---------|
| **S1: 场景导入** | 静态场景 GLB + 物体 mesh → SAPIEN | 可视化检查场景布局 |
| **S2: 机器人放置** | R1Lite 导入, 基座位置对齐视频视角 | 机器人EE在正确位置 |
| **S3: 轨迹注入** | 手部轨迹 → EE轨迹 + 夹爪时序 → env.step() | 机械臂按视频动作运动 |
| **S4: 物理验证** | 仿真物体轨迹 vs VGGT追踪轨迹 | 偏差 < 阈值 |
| **S5: 参数调优** | 摩擦力/刚度/夹爪力 → 仿真更贴近视频 | 偏差进一步减小 |

### 15.9 代码文件

| 文件 | 职责 |
|------|------|
| `object_tracking/simulation/scene_builder.py` | 构建 SAPIEN 仿真场景 (静态场景+物体+机器人) |
| `object_tracking/simulation/action_player.py` | 逐帧驱动仿真 (EE轨迹+夹爪时序) |
| `object_tracking/simulation/run_simulation.py` | 入口脚本, 编排完整仿真流程 |

---

## 十六、V-Dreamer 精确追踪方案：从视频到可执行轨迹

### 16.1 核心问题

合并管线解决了"场景+手部"的3D重建问题，但**物体交互的精确追踪**仍然是瓶颈：

| 问题 | 根因 | 影响 |
|------|------|------|
| 物体6DoF漂移 | 逐帧独立 Procrustes 对齐，无时序约束 | 长序列累积误差 |
| VGGT TrackHead 闲置 | CoTracker 架构已内置但从未调用 | 丢失时序一致追踪能力 |
| 接触检测脆弱 | 仅基于深度对比，受噪声影响 | 抓取时机不准 |
| 开环无校正 | 仿真结果与视频无闭环对比 | 偏差无法自修正 |

V-Dreamer 论文的核心思路：**视频本身就是最好的运动先验**。通过联合点追踪（时序一致）→ 运动耦合检测（精确抓取）→ 闭环验证（自校正），实现精确的物体交互复刻。

### 16.2 方案对比

| 维度 | 旧方案 (run_tracking.py) | 新方案 (run_precise_tracking.py) |
|------|--------------------------|----------------------------------|
| 物体追踪 | 逐帧 Procrustes 对齐 | VGGT TrackHead 联合追踪 |
| 接触检测 | 深度对比 (5cm阈值) | 运动耦合度 + 距离稳定性 |
| 6DoF估计 | 简单 Procrustes | RANSAC + 置信度加权 + 时序平滑 |
| 验证 | 事后偏差统计 | 闭环验证 + 轨迹参数调整 |
| 时序一致性 | 无 | TrackHead CoTracker 架构保证 |

### 16.3 完整数据流

```
┌─────────────────────────────────────────────────────────┐
│                    输入: 第一人称操作视频                   │
└──────────────────────────┬──────────────────────────────┘
                           │
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
         VGGT4D +       HaWoR           SAM3
         TrackHead      (手部重建)       (场景分割)
         (3D感知+       │                │
          联合追踪)     │                │
              │            │                │
              ▼            ▼                ▼
    ┌──────────────────────────────────────────────┐
    │     Step 1: VGGT4D 三阶段 + TrackHead        │
    │                                              │
    │  Stage 1: 推理 → depth + dynamic_mask        │
    │  采样 query_points (从 dynamic_mask/SAM3)     │
    │  Stage 2: 推理(dyn_masks + query_points)      │
    │    → 精化外参 + 追踪2D轨迹 ★核心★             │
    │  Stage 3: 精化 dynamic_mask                   │
    │  深度反投影 → 3D点轨迹 (S, N, 3)              │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │     Step 2: 运动耦合检测 (grasp_controller)   │
    │                                              │
    │  物体点运动方向 vs 手部点运动方向               │
    │  耦合度 > 阈值 → 抓取                         │
    │  耦合度 < 阈值 → 释放                         │
    │  距离稳定性 → 区分"抓取中"vs"接近中"            │
    │  → gripper_timeline + grasp_poses             │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │     Step 3: 精确6DoF (trajectory_refiner)     │
    │                                              │
    │  RANSAC + Procrustes (剔除离群追踪点)          │
    │  置信度加权 (低置信度点影响小)                   │
    │  时序平滑 (Savitzky-Golay + SLERP)            │
    │  释放后静止约束 (物理一致性)                    │
    │  → object_6dof (S, 4, 4)                     │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │     Step 4: 仿真执行 (SAPIEN)                  │
    │                                              │
    │  手部轨迹 → EE轨迹 + 夹爪时序 → env.step()     │
    │  物体由物理引擎被动响应                         │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │     Step 5: 闭环验证 (closed_loop_verifier)   │
    │                                              │
    │  仿真物体轨迹 vs 追踪物体轨迹                   │
    │  偏差 > 阈值 → 调整轨迹参数 → 重新仿真          │
    │  偏差 < 阈值 → 验证通过!                       │
    └──────────────────────────────────────────────┘
```

### 16.4 TrackHead 联合追踪原理

VGGT 内置的 TrackHead 基于 CoTracker 架构，核心优势：

1. **联合追踪**: N个点在 Transformer 中相互交互，利用点间相关性提高鲁棒性
2. **迭代精化**: 默认4次迭代，每次基于相关性金字塔 + 时空注意力更新坐标
3. **时序一致性**: 所有点在所有帧中同时优化，不会出现帧间跳变

```
TrackHead 工作流程:
  query_points (N, 2) → 初始化所有帧坐标
  for iter in range(4):
    1. CorrBlock: 计算当前坐标与特征图的相关性金字塔
    2. EfficientUpdateFormer: 时空注意力更新坐标增量
    3. 强制参考帧坐标不变 (query_points是已知的)
    4. 更新追踪点特征
  输出: tracks (S, N, 2) + visibility (S, N) + confidence (S, N)
```

关键: 在 VGGT4D Stage 2 中同时传入 `query_points`，一次前向传播同时获得精化外参和追踪轨迹，无需额外推理。

### 16.5 运动耦合检测原理

传统深度对比接触检测的局限：指尖深度与场景深度差 < 5cm → 接触。但深度噪声大、阈值难调。

运动耦合检测的核心洞察：**抓取 = 物体开始跟随手部运动**。

```
耦合度计算:
  物体速度方向 vs 手部速度方向 → 余弦相似度
  滑动窗口内平均 → 耦合度 [0, 1]

距离稳定性:
  物体-手部相对距离的方差 → 稳定性 [0, 1]
  抓取中: 距离恒定 (方差小, 稳定性高)
  释放后: 距离变化 (方差大, 稳定性低)

综合判断:
  抓取 = 耦合度高 + 稳定性高 + 距离近 + 手部在运动
  释放 = 耦合度低 或 稳定性低 或 距离远
```

### 16.6 闭环验证策略

```
闭环流程:
  1. 执行仿真 → 获取物体实际轨迹
  2. 计算仿真轨迹 vs 追踪轨迹的位置偏差
  3. 偏差 > 阈值 (3cm):
     a. 计算接触帧的系统性位置偏移
     b. 用偏移量调整轨迹 → 重新仿真
     c. 或调整夹爪时序 (提前/延后1-2帧)
  4. 偏差 < 阈值 → 验证通过

最多迭代3次, 每次调整一个参数:
  - 迭代1: 位置偏移补偿
  - 迭代2: 夹爪时序微调
  - 迭代3: 物理参数调整 (摩擦力等)
```

### 16.7 代码文件

| 文件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `object_tracking/point_tracker.py` | VGGT TrackHead 激活 + 联合点追踪 | 视频 + SAM3 mask | tracks_3d + visibility + confidence |
| `object_tracking/grasp_controller.py` | 运动耦合检测 + 抓取时序 | tracks_3d + 手部顶点 | gripper_timeline + grasp_poses + segments |
| `object_tracking/trajectory_refiner.py` | RANSAC 6DoF + 时序平滑 | tracks_3d + confidence + segments | object_6dof (S,4,4) |
| `object_tracking/closed_loop_verifier.py` | 闭环验证 + 轨迹调整 | 仿真结果 + 参考轨迹 | 验证报告 + 调整后轨迹 |
| `object_tracking/run_precise_tracking.py` | 精确追踪管线入口 | MP4 视频 | 完整追踪结果 |

### 16.8 与旧管线的兼容

新管线 (`run_precise_tracking.py`) 与旧管线 (`run_tracking.py`) 并行存在：

- **旧管线**: VGGT-Omega 输出 + HaWoR + SAM3 → contact_detector → object_tracker → action_semantics
- **新管线**: VGGT4D + TrackHead + HaWoR + SAM3 → grasp_controller → trajectory_refiner → closed_loop_verifier

两者共享：
- HaWoR 手部重建结果 (`hawor_results.npz`)
- SAM3 物体分割 mask (`object_masks.npz`)
- SAPIEN 仿真模块 (`simulation/`)

新管线的关键改进是**用 TrackHead 联合追踪替代逐帧 Procrustes**，用**运动耦合检测替代深度对比接触检测**。

### 16.9 测试视频: beizi.mp4

```
视频: HaWoR/example/beizi.mp4 (杯子抓取)
帧数: 184帧 (6秒, 30fps)
HaWoR 已有输出: SLAM, tracks, cam_space, extracted_images, vis_verify
HaWoR 缺失: reconstruction/ (需补全)
VGGT4D 输出: 尚未生成

运行命令:
  # 1. 补全 HaWoR 重建
  cd HaWoR && python demov2.py --video example/beizi.mp4

  # 2. 运行精确追踪管线
  cd ReplicateAnyScene
  python object_tracking/run_precise_tracking.py \
    --video /path/to/HaWoR/example/beizi.mp4 \
    --hawor_npz /path/to/HaWoR/example/beizi/reconstruction/hawor_results_0_184.npz

  # 3. 仅追踪不仿真 (快速验证)
  python object_tracking/run_precise_tracking.py \
    --video /path/to/beizi.mp4 --skip_simulation
```
