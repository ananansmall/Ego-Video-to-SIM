# 第一人称视频 -> 夹爪替换管道

从第一人称视角（egocentric）视频中提取人手动作，映射到夹爪（Panda Gripper / R1 Gripper），并将夹爪叠加到原始视频帧上替换人手区域。

## 项目架构

```
输入视频 (MP4)
    │
    ▼
[EgocentricHandDetector] ─── MediaPipe Hands 手部检测
    │                         输出: 21个3D关键点, 手腕6DoF位姿, 2D关键点
    ▼
[RobotHandRetargeter] ────── dex_retargeting 关节重定向
    │                         输出: 夹爪关节角度
    ▼
[ContactPreservationFilter] ─ 接触保持滤波
    │                         输出: 滤波后关节值 + 抓取状态
    ▼
[TrajectorySmoother] ─────── 轨迹平滑
    │                         输出: 平滑后关节值
    ├──→ [compute_hand_mask] ── 生成手部区域掩码（凸包+膨胀）
    │
    ├──→ [GripperSAPIENRenderer] ── SAPIEN 3D 渲染夹爪
    │     │                         含 MANO 骨架 / 坐标轴 / 轨迹拖尾
    │     │                         相机视角与视频一致
    │     │                         输出: sapien_bgr + gripper_mask
    │     ▼
    │   [composite_gripper_on_frame] ── 色键合成夹爪到原始帧
    │                                    掩码区域替换为夹爪
    │
    └──→ [render_software_gripper_overlay] ── 软件渲染回退
                                               简化夹爪示意图叠加

    ▼
gripper_replaced.mp4  (夹爪替换后的视频)
sapien_render.mp4     (纯 SAPIEN 3D 渲染，含 MANO/坐标轴/轨迹)
overlay.mp4           (原始视频+手部骨架标注+手腕位置)
trajectory_viz.mp4    (关节角度/手腕位置/手指距离曲线可视化)
trajectory.pkl        (完整轨迹数据)
```

## 文件说明

| 文件 | 功能 | 说明 |
|------|------|------|
| `gripper_video_pipeline.py` | **夹爪替换管道主脚本** | 完整管道：检测→重定向→渲染→合成→可视化→保存 |
| `egocentric_hand_detector.py` | 手部检测器 | MediaPipe Hands 封装，支持 selfie 模式 |
| `arm_motion_analyzer.py` | 手臂运动分析器 | 计算手腕6DoF位姿、手指距离等 |
| `robot_hand_retargeter.py` | 重定向器封装 | SeqRetargeting 优化器接口 |
| `contact_preservation.py` | 接触保持滤波器 | 迟滞比较器 + 轨迹平滑 |

## 快速开始

### 环境配置

```bash
conda activate dex
pip install mediapipe opencv-python numpy scipy tqdm
# SAPIEN 3D 渲染（可选，需要 GPU + Vulkan）
pip install sapien
```

### 基本用法

```bash
# 使用 Panda 夹爪替换视频中的人手
conda run -n dex python gripper_video_pipeline.py \
  --video-path /path/to/video.mp4 \
  --output-dir ./output_panda \
  --robot panda

# 使用 R1 夹爪
conda run -n dex python gripper_video_pipeline.py \
  --video-path /path/to/video.mp4 \
  --output-dir ./output_r1 \
  --robot r1

# 无 GPU 时使用软件渲染
conda run -n dex python gripper_video_pipeline.py \
  --video-path /path/to/video.mp4 \
  --output-dir ./output_soft \
  --robot panda \
  --software-render

# 只看 SAPIEN 渲染效果（不合成到原始视频）
conda run -n dex python gripper_video_pipeline.py \
  --video-path /path/to/video.mp4 \
  --output-dir ./output_debug \
  --robot panda \
  --mano-scale 2.0

# 关闭部分可视化元素
conda run -n dex python gripper_video_pipeline.py \
  --video-path /path/to/video.mp4 \
  --output-dir ./output_clean \
  --robot panda \
  --no-mano --no-trail
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video-path` | (必填) | 输入 MP4 视频文件路径 |
| `--output-dir` | (必填) | 输出目录路径 |
| `--robot` | panda | 夹爪类型: panda / r1 / allegro / shadow / svh / leap / ability / inspire |
| `--hand` | right | 追踪的手别: right / left |
| `--retargeting` | position | 重定向算法: position / vector / dexpilot |
| `--fps` | 30 | 目标帧率 |
| `--start-frame` | 0 | 起始帧索引 |
| `--num-frames` | -1 | 处理帧数（-1=全部） |
| `--no-selfie` | False | 禁用自拍模式（非第一人称视角视频使用） |
| `--grasp-threshold` | 0.04 | 抓取检测阈值（手指距离） |
| `--release-threshold` | 0.06 | 释放检测阈值（手指距离） |
| `--software-render` | False | 使用软件渲染（不使用 SAPIEN 3D） |
| `--no-visualize` | False | 禁用实时可视化 |
| `--mask-dilation` | 30 | 手部掩码膨胀像素数 |
| `--mano-scale` | 2.0 | MANO 骨架缩放系数 |
| `--no-axes` | False | 隐藏坐标轴可视化 |
| `--no-trail` | False | 隐藏轨迹拖尾 |
| `--no-mano` | False | 隐藏 MANO 骨架 |

## 输出文件

| 文件 | 说明 |
|------|------|
| `gripper_replaced.mp4` | 夹爪替换后的视频（人手被夹爪替换） |
| `sapien_render.mp4` | 纯 SAPIEN 3D 渲染视频（含 MANO 骨架、坐标轴、轨迹拖尾） |
| `overlay.mp4` | 原始视频 + 手部骨架标注 + 手腕位置 + 抓取状态 |
| `trajectory_viz.mp4` | 关节角度/手腕位置/手指距离曲线可视化 |
| `trajectory.pkl` | 完整轨迹数据（关节角度 + 手腕位姿 + 抓取状态） |

## 管道步骤详解

### 步骤1：手部检测与重定向

1. **EgocentricHandDetector.detect_single()** (`egocentric_hand_detector.py`)
   - 输入: RGB 图像
   - MediaPipe Hands 检测 21 个 2D/3D 关键点
   - 深度估计: `Z = f_norm * metric_dist / img_dist`（针孔相机模型，假设 FOV=70°）
   - 手腕坐标系估计: SVD 法线估计
   - 坐标变换: MediaPipe → MANO (`OPERATOR2MANO_RIGHT/LEFT`)
   - 输出: `joint_pos`(21x3), `wrist_rot_mano`(3x3), `wrist_pos_camera`(3,), `keypoint_2d`

2. **RobotHandRetargeter.retarget()** (`robot_hand_retargeter.py`)
   - 使用 `dex_retargeting.SeqRetargeting` 优化器
   - `PositionOptimizer`: Huber Loss(机器人link位置 vs 人手参考位置) + L2正则
   - NLopt LD_SLSQP 求解
   - 输出: 夹爪关节角度（Panda: 1个关节, R1: 1个关节）

3. **ContactPreservationFilter** (`contact_preservation.py`)
   - 迟滞比较器检测抓取/释放
   - 输出: 滤波后关节值 + is_grasping 标志

4. **TrajectorySmoother** (`contact_preservation.py`)
   - 中值滤波 + 低通滤波
   - 输出: 平滑后关节值

### 步骤2：SAPIEN 3D 渲染（含可视化）

- **GripperSAPIENRenderer** (`gripper_video_pipeline.py`)
  - 加载夹爪 URDF 到 SAPIEN 场景
  - 设置夹爪位姿: 手腕相机坐标 → R_CAM2BASE → SAPIEN 世界坐标
  - 设置夹爪关节角度
  - 调用 `scene.step()` 更新物理状态
  - 渲染 MANO 骨架（5指5色球体+胶囊体）
  - 渲染坐标轴（RGB=XYZ，手腕+末端执行器+各link）
  - 渲染轨迹拖尾（黄色=MANO手腕，蓝色=夹爪末端）
  - 从与视频一致的相机视角渲染图像
  - 相机参数: FOV 与视频一致（70°），分辨率与视频一致

#### MANO 骨架可视化

在 SAPIEN 3D 场景中绘制 21 个关节点和 5 根手指的骨段：
- 球体表示关节点（手腕=肤色，各指=对应颜色）
- 胶囊体表示骨段（各指=对应颜色）
- 颜色编码：拇指=红，食指=橙，中指=绿，无名指=蓝，小指=紫
- 缩放系数 `--mano-scale` 控制骨架大小（默认 2.0）

#### 坐标轴可视化

在手腕位置和末端执行器位置绘制 RGB=XYZ 坐标轴：
- 手腕坐标轴：长度 0.25m，半径 0.012m
- 末端执行器坐标轴：长度 0.12m，半径 0.006m
- 各 link 坐标轴：长度 0.04m，半径 0.002m

#### 轨迹拖尾可视化

在 SAPIEN 场景中绘制手腕和夹爪末端的历史轨迹：
- 黄色轨迹：MANO 手腕位置
- 蓝色轨迹：夹爪末端执行器位置
- 最多保留 200 个历史点
- 轨迹透明度随时间渐变（旧→透明，新→不透明）

### 步骤3：手部掩码生成

- **compute_hand_mask()** (`gripper_video_pipeline.py`)
  - 使用 MediaPipe 2D 关键点构建凸包
  - 膨胀掩码覆盖手指间隙和手腕
  - 高斯模糊实现边缘平滑过渡
  - 输出: 二值掩码 (H, W)

### 步骤4：色键合成

- **composite_gripper_on_frame()** (`gripper_video_pipeline.py`)
  - 夹爪掩码区域：完全用夹爪图像替换（alpha=1.0）
  - 手部掩码区域（非夹爪）：用夹爪图像半透明覆盖（alpha=0.6）
  - 非手部区域：保留原始图像
  - 输出: 合成后的 BGR 图像

### 步骤5：轨迹可视化

- **render_trajectory_video()** (`gripper_video_pipeline.py`)
  - 左面板：关节角度柱状图
  - 右面板：手腕 XYZ 位置曲线 + 手指距离曲线 + 抓取状态
  - 输出: `trajectory_viz.mp4`

### 步骤6：保存输出

- 保存替换后的视频 (`gripper_replaced.mp4`)
- 保存 SAPIEN 渲染视频 (`sapien_render.mp4`)
- 保存叠加视频 (`overlay.mp4`)
- 保存轨迹可视化视频 (`trajectory_viz.mp4`)
- 保存轨迹数据 (`trajectory.pkl`)

## 关键技术参数

### 坐标系变换

```
MediaPipe 相机坐标系: X右, Y下, Z前(远离相机)
    │  R_CAM2BASE = [[0,0,1],[-1,0,0],[0,-1,0]]
    ▼
SAPIEN 世界坐标系: X前, Y左, Z上
```

### 旋转变换链（与 egocentric_sapien_render.py 一致）

```python
# MANO 坐标系变换
wrist_rot_mano_full = wrist_rot_mano @ operator2mano
# 世界坐标系旋转
wrist_rot_world = R_CAM2BASE @ wrist_rot_mano_full @ operator2mano.T
# 设置夹爪位姿
robot.set_pose(Pose(wrist_pos_base, quat_from_matrix(wrist_rot_world)))
```

### 相机视角匹配

```python
# SAPIEN 相机设置
fovy = 2 * arctan(tan(fov_h/2) * img_height / img_width)  # 从水平FOV计算垂直FOV
camera = scene.add_camera("cam", width, height, fovy, 0.01, 100)

# 动态跟随相机（与 egocentric_sapien_render.py 一致）
smooth_target = lowpass_filter(wrist_pos_base, alpha=0.08)
cam_pos = smooth_target + [-0.30, 0.0, 0.45]   # 后方偏上
cam_look_at = smooth_target + [0.10, 0.0, -0.05]  # 前方偏下
cam_rotation = look_at_matrix(cam_pos, cam_look_at, up=[0,0,1])
```

### 手部掩码生成

```python
# 凸包 + 膨胀 + 高斯模糊
hull = cv2.convexHull(keypoint_2d_pixels)
mask = cv2.fillConvexPoly(blank, hull, 255)
mask = cv2.dilate(mask, ellipse_kernel, iterations=1)
mask = cv2.GaussianBlur(mask, (2*dilation+1, 2*dilation+1), 0)
```

### 色键合成策略

```python
# 夹爪掩码（色键提取）+ 手部掩码（凸包膨胀）
blend_alpha = max(gripper_alpha, hand_mask * 0.6)
result = original * (1 - blend_alpha) + gripper * blend_alpha
```

## 调试指南

### 查看 SAPIEN 渲染效果

`sapien_render.mp4` 是纯 SAPIEN 3D 渲染视频，包含：
- 夹爪模型（URDF 加载的 3D 模型）
- MANO 骨架（5指5色球体+胶囊体）
- 坐标轴（RGB=XYZ）
- 轨迹拖尾（黄色=手腕，蓝色=末端）
- 手部骨架叠加（MediaPipe 2D 关键点投影）
- 帧号 + 检测状态 + 手腕位置文字标注

如果 `sapien_render.mp4` 中看不到夹爪，说明：
1. 手腕位置估计有误 → 检查 overlay.mp4 中的手腕位置数值
2. 相机跟随逻辑有误 → 检查 sapien_render.mp4 中手腕位置是否合理
3. 夹爪太小/太大 → 调整 `--mano-scale` 参数

### 查看轨迹数据

`trajectory_viz.mp4` 是关节角度和手腕位置的可视化：
- 左面板：各关节角度柱状图
- 右面板：手腕 XYZ 曲线 + 手指距离曲线 + 抓取状态

如果关节角度全为 0 或不变，说明重定向优化器可能未收敛。

### 查看检测结果

`overlay.mp4` 是原始视频 + 检测结果叠加：
- 手部骨架（红色关键点 + 连接线）
- 抓取状态（GRASP/OPEN）
- 手腕在相机坐标系下的 3D 位置

如果 overlay.mp4 中没有骨架，说明 MediaPipe 检测失败。

## 支持的夹爪类型

| 夹爪 | RobotName | 关节数 | 说明 |
|------|-----------|--------|------|
| Panda Gripper | `panda` | 1 (prismatic) | Franka Panda 机器人夹爪，2指平行 |
| R1 Gripper | `r1` | 1 (prismatic) | R1 机器人右夹爪，2指平行 |
| Allegro Hand | `allegro` | 16 (revolute) | 4指灵巧手（非夹爪） |
| Shadow Hand | `shadow` | 24 (revolute) | 5指灵巧手（非夹爪） |

**推荐**: 对于简单的抓取/释放动作，使用 `panda` 或 `r1` 夹爪；对于需要精细手指控制的任务，使用 `allegro` 或 `shadow` 灵巧手。

## 软件渲染回退

当 SAPIEN 不可用时（无 GPU 或 Vulkan 支持），管道自动回退到软件渲染模式：

- 在原始视频帧上绘制简化的夹爪示意图
- 矩形表示夹爪基座和手指
- 根据关节角度控制手指间距
- 颜色编码抓取状态（绿色=抓取，橙色=张开）

## 常见问题

### Q: 替换效果不理想，夹爪位置偏移？

A: 深度估计基于 FOV 假设（70°），如果实际相机 FOV 不同，会导致深度估计偏差。可以尝试：
1. 查看 `sapien_render.mp4` 确认夹爪在 3D 场景中的位置
2. 调整 `--mask-dilation` 参数扩大手部掩码范围
3. 使用 `--software-render` 查看简化渲染效果进行对比

### Q: 第一人称视角下手部检测率低？

A: 第一人称视角下手部经常被截断或遮挡，可以尝试：
1. 降低检测置信度阈值（修改 `EgocentricHandDetector` 的 `min_detection_confidence`）
2. 使用 `--grasp-threshold` 和 `--release-threshold` 调整抓取检测灵敏度

### Q: 夹爪开合与手部动作不同步？

A: 可以调整滤波参数：
1. `ContactPreservationFilter` 的 `smoothing_alpha`（0=完全平滑，1=无滤波）
2. `TrajectorySmoother` 的 `window_size`（窗口越大越平滑）和 `alpha`（低通滤波系数）

### Q: SAPIEN 渲染视频是黑屏/空白？

A: 可能原因：
1. CUDA 驱动版本不兼容 → 检查 NVIDIA 驱动和 CUDA 版本
2. Vulkan 未正确配置 → 检查 `VK_ICD_FILENAMES` 环境变量
3. 使用 `--software-render` 回退到软件渲染模式

### Q: 手指距离太小，抓取检测不灵敏？

A: 第一人称视角下手指距离通常较小，管道会自动提示建议阈值。也可以手动设置：
```bash
conda run -n dex python gripper_video_pipeline.py \
  --video-path /path/to/video.mp4 \
  --output-dir ./output \
  --grasp-threshold 0.02 \
  --release-threshold 0.03
```

## 依赖项

- Python >= 3.8
- MediaPipe
- OpenCV (cv2)
- NumPy, SciPy, tqdm
- SAPIEN (可选, 3D渲染)
- pytransform3d (旋转矩阵转换)
- dex_retargeting (核心重定向库)
