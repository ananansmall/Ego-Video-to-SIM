# 第一人称视频 -> R1 机器人仿真管道

将第一人称视角（egocentric）视频中的人手动作转换为 R1 机器人的关节轨迹，并在 SAPIEN 物理引擎中渲染仿真视频。

## 项目架构

```
输入视频 (MP4)
    │
    ▼
[EgocentricHandDetector] ─── MediaPipe Hands 手部检测
    │                         输出: 21个3D关键点, 手腕6DoF位姿
    ▼
[ArmMotionAnalyzer] ──────── 手腕位姿计算
    │                         输出: 手腕位置+四元数, 手指距离
    ▼
[RobotHandRetargeter] ────── dex_retargeting 关节重定向
    │                         输出: R1夹爪关节角度
    ▼
[ContactPreservationFilter] ─ 接触保持滤波
    │                         输出: 滤波后关节值 + 抓取状态
    ▼
[TrajectorySmoother] ─────── 轨迹平滑
    │                         输出: 平滑后关节值
    ▼
trajectory.pkl (轨迹数据)
    │
    ├── [RelaxedIK] ─────────── R1臂部逆运动学求解
    │                           输出: 6个臂关节角度
    ▼
[SAPIEN Renderer] ────────── R1全身机器人仿真渲染
                               输出: r1_sapien_sim.mp4
```

## 文件说明

| 文件 | 功能 | 说明 |
|------|------|------|
| `r1_sapien_sim.py` | R1 SAPIEN 仿真管道 | **主入口**，完整4步管道：检测→IK→渲染→日志 |
| `run_pipeline.py` | Allegro 手部重定向管道 | 独立手部（无手臂IK），适用于 Allegro 等 |
| `video_to_trajectory.py` | 轻量轨迹提取 | 仅提取轨迹，不包含渲染步骤 |
| `egocentric_hand_detector.py` | 手部检测器 | MediaPipe Hands 封装，支持 selfie 模式 |
| `arm_motion_analyzer.py` | 手臂运动分析器 | 计算手腕6DoF位姿、手指距离等 |
| `robot_hand_retargeter.py` | 重定向器封装 | SeqRetargeting 优化器接口 |
| `contact_preservation.py` | 接触保持滤波器 | 迟滞比较器 + 轨迹平滑 |
| `render_trajectory.py` | SAPIEN 手部渲染 | 独立手部渲染（Allegro等） |

## 快速开始

### 环境配置

```bash
conda activate dex
pip install mediapipe opencv-python numpy scipy tqdm
```

### 运行 R1 仿真管道

```bash
conda run -n dex python r1_sapien_sim.py \
  --video-path /path/to/video.mp4 \
  --output-dir ./output_r1_sapien
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video-path` | (必填) | 输入 MP4 视频文件路径 |
| `--output-dir` | (必填) | 输出目录路径 |
| `--fps` | 30 | 目标帧率 |
| `--hand` | auto | 手别: auto/Right/Left |
| `--grasp-threshold` | 0.04 | 抓取检测阈值 |
| `--view` | front | 相机视角: behind/front/topdown/side/closeup |

## 输出文件

| 文件 | 说明 | 生成代码位置 |
|------|------|-------------|
| `r1_sapien_sim.mp4` | R1 机器人仿真视频 | `r1_sapien_sim.py` → `_SAPIEN_RENDER_SCRIPT` (子进程) |
| `overlay.mp4` | 原始视频+手部骨架+抓取标注 | `r1_sapien_sim.py` → `_render_overlay_video()` |
| `trajectory.pkl` | 完整轨迹数据 | `r1_sapien_sim.py` → `_detect_and_retarget()` |
| `ik_result.pkl` | IK求解结果 | `r1_sapien_sim.py` → `_IK_SOLVER_SCRIPT` (子进程) |
| `pre_render_setup.pkl` | 预渲染设置 | `r1_sapien_sim.py` → `_PRE_RENDER_SETUP_SCRIPT` (子进程) |
| `render_input.pkl` | 渲染输入数据 | `r1_sapien_sim.py` → `_render_sapien()` |
| `render_evaluation.pkl` | FK验证评估 | `r1_sapien_sim.py` → `_SAPIEN_RENDER_SCRIPT` |
| `pipeline_info.json` | 结构化参数数据 | `r1_sapien_sim.py` → `_save_pipeline_info()` |
| `pipeline_info.log` | 详细运行日志 | `r1_sapien_sim.py` → `setup_logger()` |
| `_ik_solver.py` | IK求解子进程脚本 | `r1_sapien_sim.py` → `_IK_SOLVER_SCRIPT` 常量 |
| `_sapien_renderer.py` | SAPIEN渲染子进程脚本 | `r1_sapien_sim.py` → `_SAPIEN_RENDER_SCRIPT` 常量 |

## 管道步骤详解

### 步骤0：自动手别检测

均匀采样视频帧（默认30帧），使用 MediaPipe Hands 检测左右手出现频次。

### 步骤1：手部检测与重定向

**代码位置**: `r1_sapien_sim.py` → `R1SAPIENPipeline._detect_and_retarget()`

1. **EgocentricHandDetector.detect()** (`egocentric_hand_detector.py`)
   - 输入: RGB图像
   - MediaPipe Hands 检测21个2D/3D关键点
   - `_compute_wrist_pos_camera()`: 利用2D关键点图像距离+针孔相机模型估计深度 `Z = f_norm * metric_dist / img_dist`
   - `_estimate_frame_from_hand_points()`: SVD估计手腕坐标系旋转矩阵
   - 坐标变换: MediaPipe坐标 → MANO坐标 (`OPERATOR2MANO_RIGHT/LEFT`)
   - 输出: `joint_pos`(21x3), `wrist_rot_mano`(3x3), `wrist_pos_camera`(3,), `keypoint_2d`

2. **RobotHandRetargeter.retarget()** (`robot_hand_retargeter.py`)
   - 使用 `dex_retargeting.SeqRetargeting` 优化器
   - `PositionOptimizer`: Huber Loss(机器人link位置 vs 人手参考位置) + L2正则
   - NLopt LD_SLSQP 求解
   - 输出: R1夹爪2个关节角度

3. **ContactPreservationFilter** (`contact_preservation.py`)
   - 迟滞比较器检测抓取/释放（finger_distance vs threshold）
   - 输出: 滤波后关节值 + is_grasping标志

4. **TrajectorySmoother** (`contact_preservation.py`)
   - 中值滤波 + 低通滤波
   - 输出: 平滑后关节值

5. 保存 `trajectory.pkl`

### 步骤2：RelaxedIK 逆运动学求解

**代码位置**: `r1_sapien_sim.py` → `_IK_SOLVER_SCRIPT` (子进程)

1. 加载 `trajectory.pkl`，提取手腕轨迹
2. **坐标变换**: `R_cam2base = [[0,0,1],[-1,0,0],[0,-1,0]]` 将相机坐标转到臂基座坐标
3. **工作空间映射**:
   - `mapping_offset = COMFORT_TARGET - avg_wrist` (映射到舒适工作空间)
   - `safety_offset = -COMFORT_TARGET/|COMFORT_TARGET| * 0.075` (安全偏移)
4. **RelaxedIK求解** (`relaxed_ik_solver.py` → Rust FFI `librelaxed_ik_lib.so`)
   - 输入: `ik_target_base`(3D位置) + `ee_quat`(朝向四元数)
   - 输出: 6个臂关节角度
5. 后处理: 关节限位钳位 + 低通滤波 + 速度限制(30°/s)
6. 保存 `ik_result.pkl`

### 步骤3：预渲染设置

**代码位置**: `r1_sapien_sim.py` → `_PRE_RENDER_SETUP_SCRIPT` (子进程)

1. 加载R1机器人URDF到SAPIEN场景
2. 获取臂基座link的实际位姿 (`base_link_p`, `base_link_R`)
3. 保存 `pre_render_setup.pkl`

### 步骤4：SAPIEN 渲染

**代码位置**: `r1_sapien_sim.py` → `_SAPIEN_RENDER_SCRIPT` (子进程)

1. 加载 `render_input.pkl` + `pre_render_setup.pkl`
2. 创建SAPIEN场景: 地面、桌面、环境光照
3. 加载完整R1机器人URDF
4. 设置渲染相机 (look-at方式)
5. 初始化MANO 3D可视化材质和辅助函数
6. 逐帧渲染循环:
   - 设置关节角度 → `scene.step()` → `scene.update_render()`
   - FK验证: 计算末端执行器实际位置与IK目标误差
   - **MANO 3D手模型渲染** (SAPIEN internal_renderer):
     - 21个关节球体 + 5根手指胶囊体骨架
     - 深度驱动动态缩放: `mano_scale = 3.0 * (0.4 / max(depth, 0.15))`, clip=[2.0, 6.0]
     - 手腕3轴坐标轴 (R=X, G=Y, B=Z), 长0.20m
     - MCP关节迷你坐标轴, 长0.06m
     - 指尖迷你坐标轴, 长0.05m
     - 夹爪映射: 2根胶囊体手指 + 掌心方块
   - **机器人臂关节坐标轴**: 每个link处3轴, 长0.10m
   - **机器人末端姿态坐标轴**: 3轴, 长0.15m
   - **末端执行器轨迹**: 蓝色渐变胶囊体 + 球体, 最多200帧
   - **MANO手轨迹**: 黄色渐变胶囊体 + 球体, 最多200帧
   - **2D叠加标注**:
     - 21个MANO关键点投影标注 (手腕/指尖/MCP/其他)
     - 机器人末端执行器位置标注
     - 对比面板: Wrist-EE距离, 5指尖坐标+距离, MANO手腕旋转, 机器人EE旋转, 旋转差异
   - 写入MP4视频
7. 保存 `render_evaluation.pkl`

### 步骤5：参数记录

1. 保存 `pipeline_info.json`
2. 保存 `pipeline_info.log`

## 关键技术参数

### 坐标系变换

```
MediaPipe相机坐标系: X右, Y下, Z前(远离相机)
    │  R_cam2base = [[0,0,1],[-1,0,0],[0,-1,0]]
    ▼
臂基座坐标系: X前, Y左, Z上
    │  base_link_R, base_link_p
    ▼
世界坐标系 (SAPIEN)
```

### 深度估计

```python
# egocentric_hand_detector.py: _compute_wrist_pos_camera()
fov_h = 70°  # 假设水平FOV
f_norm = 1.0 / (2 * tan(fov_h/2))
depth = f_norm * metric_dist / image_dist  # 针孔相机模型
# 多对参考点取中值, clip到[0.1, 3.0]米
```

### MANO手部3D可视化缩放

```python
# r1_sapien_sim.py: 渲染循环中
depth_val = wrist_pos_camera_wt[2]  # 从深度估计获取
mano_scale = 3.0 * (0.4 / max(depth_val, 0.15))  # 近处手更大
mano_scale = clip(mano_scale, 2.0, 6.0)
# 手放在机器人末端执行器旁0.5m处
mano_offset_world = [0, 0.5, 0]  # 右手
```

### 手臂映射核心逻辑

```
手腕相机坐标 → R_cam2base → 臂基座坐标
    → 运动缩放(motion_scale)
    → + mapping_offset (映射到舒适工作空间中心[0.30, 0.0, -0.25])
    → + safety_offset (安全偏移0.075m)
    → 低通滤波(alpha=0.6)
    → RelaxedIK求解 → 6个臂关节角度
    → 关节限位 + 低通滤波 + 速度限制
```

**关键代码文件**:
- 手腕6DoF: `arm_motion_analyzer.py` → `compute_wrist_6dof()`
- 工作空间映射: `r1_sapien_sim.py` → `_IK_SOLVER_SCRIPT` (第1612-1622行)
- RelaxedIK: `GalaxeaManipSim/galaxea_sim/controllers/utils/relaxed_ik_solver.py`
- 手部重定向: `robot_hand_retargeter.py` → `SeqRetargeting.retarget()`
- 优化器: `dex_retargeting/src/dex_retargeting/optimizer.py` → `PositionOptimizer`

## 手部映射方法调研

### 当前方法: dex_retargeting (基于优化)

- **PositionOptimizer**: Huber Loss(机器人link位置 vs 人手参考位置) + L2正则, NLopt SLSQP
- **VectorOptimizer**: 关节间向量方向误差, NLopt SLSQP
- **DexPilotOptimizer**: 投影机制稳定抓取
- 优点: 通用性强(任意URDF), 考虑关节限位, 热启动加速
- 缺点: 每帧迭代优化(5-15ms), 可能局部最优, 不考虑物体交互

### 其他方法对比

| 方法类别 | 代表 | 实时性 | 精度 | 通用性 | 数据需求 |
|---------|------|--------|------|--------|---------|
| 优化 | dex-retargeting | 中(5-15ms) | 高 | 高 | 无(需URDF) |
| 优化 | DexPilot(RSS'20) | 中 | 高(抓取稳定) | 中 | 无 |
| 学习 | Hand2Hand(ECCV'22) | 高(<1ms) | 中 | 低 | 大(配对数据) |
| 学习 | AnyTeleop(RSS'23) | 中 | 高 | 高 | 无 |
| 学习 | H2O(CoRL'23) | 高 | 中 | 低 | 大(RL训练) |
| 几何 | 指尖映射 | 极高 | 低 | 高 | 无 |
| 扩散 | DexDiffuser | 低(>100ms) | 高 | 中 | 大 |

### 推荐改进方向

1. **短期**: 用HaMeR替代MediaPipe Hands提升3D手部重建精度
2. **中期**: 集成Depth Anything V2替代FOV假设深度估计
3. **长期**: ORB-SLAM3+IMU方案实现高精度大范围移动追踪

## 移动摄像头解决方案

当前项目假设相机固定，使用FOV假设估计深度。若相机移动，需要额外方案获取世界坐标系下的手部位置。

### 方案对比

| 方案 | 精度 | 实时性 | 硬件 | 鲁棒性 | 部署难度 |
|------|------|--------|------|--------|---------|
| DROID-SLAM+手部追踪 | 高 | 中(10-20fps) | GPU | 中 | 高 |
| ORB-SLAM3+IMU | 高 | 高(30fps) | CPU+IMU | 高 | 高 |
| Depth Anything V2 | 中 | 中(15-30fps) | GPU | 中 | 中 |
| 3DGS+深度(离线) | 高 | 低 | GPU | 高 | 极高 |
| Aria Glasses(VI-SLAM) | 高 | 高 | 专用硬件 | 高 | 高 |
| AnyTeleop肩部锚定 | 低-中 | 高(30fps) | 单目RGB | 低-中 | 低 |
| HaMeR 3D手部重建 | 中-高 | 中(15fps) | GPU | 中 | 中 |

### 推荐方案

1. **纯视觉低成本**: AnyTeleop方案 - 用MediaPipe Pose检测肩部锚定世界坐标
2. **中等精度**: Depth Anything V2度量深度 + HaMeR 3D手部重建
3. **高精度**: ORB-SLAM3+IMU (手机内置IMU即可) 或 Aria Glasses

## 依赖项

- Python >= 3.8
- MediaPipe
- OpenCV (cv2)
- NumPy, SciPy, tqdm
- SAPIEN (可选, 3D渲染)
- pytransform3d (旋转矩阵转换)
- dex_retargeting (核心重定向库)
- GalaxeaManipSim (R1 URDF + RelaxedIK配置)
