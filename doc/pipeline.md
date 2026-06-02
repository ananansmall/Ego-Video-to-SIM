# HaWoR + RAS → SAPIEN 仿真管线

从第一人称操作视频生成 SAPIEN 仿真场景，包含 3D 物体、MANO 手部和 R1 机械臂。

## 管线架构

```
输入数据
├── hawor_dir/                    # HaWoR 手部重建
│   ├── reconstruction/
│   │   └── hawor_results_*.npz   # 手部关节、相机轨迹
│   ├── cam_space/                # 相机空间手部参数 (2D重投影GT)
│   ├── extracted_images/         # 原始视频帧
│   └── est_focal.txt             # 焦距估计
└── ras_dir/                      # RAS 场景重建
    ├── final_scene.glb           # 3D 场景 (带顶点颜色)
    ├── extrinsics/               # 相机外参 (N个.txt)
    └── intrinsic.txt             # 相机内参

管线步骤
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: 01_align_scene.py                                      │
│    第一帧相机锚定 + Umeyama尺度校正                               │
│    输出: output/alignment/transform_params.npz                   │
│                                                                  │
│  Step 2: 02_render_scene.py --mode hand_only                    │
│    MANO手 + GLB物体 → 第一人称视频                               │
│    输出: output/videos/hand_object_hand_only.mp4                 │
│                                                                  │
│  Step 3: 02_render_scene.py --mode robot_only                   │
│    R1机器人 + GLB物体 → 机器人替代视频                            │
│    输出: output/videos/hand_object_robot_only.mp4                │
│                                                                  │
│  Step 4: 02_render_scene.py --mode robot_tracking               │
│    MANO手 + R1机器人 + GLB物体 → 对比视频                        │
│    输出: output/videos/hand_object_robot_tracking.mp4            │
│                                                                  │
│  Step 5: 04_physics_simulation.py                                │
│    物理引擎驱动仿真: PD控制 + 碰撞 + 抓取                        │
│    输出: output/videos/physics_sim_physics_tracking.mp4          │
│                                                                  │
│  Step 6: 05_video_alignment.py                                   │
│    视频-仿真对齐: 2D重投影验证 + 叠加对比 + 位姿优化              │
│    输出: output/alignment_analysis/                              │
└──────────────────────────────────────────────────────────────────┘
```

## 一键命令

### 交互式 Viewer（推荐先运行）

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

conda run -n dex python 00_run_pipeline.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --viewer
```

### 渲染全部视频

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

conda run -n dex python 00_run_pipeline.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

### 两条核心命令

| 目的 | 命令 | 说明 |
|------|------|------|
| **交互式 Viewer** | `python 00_run_pipeline.py --hawor-dir ... --ras-dir ... --viewer` | 对齐 → 启动 SAPIEN 交互式渲染 |
| **渲染视频** | `python 00_run_pipeline.py --hawor-dir ... --ras-dir ...` | 对齐 → 渲染 3 段视频 |

---

## 各阶段详细说明

### Step 0: 00_run_pipeline.py — 一键管线入口

**功能**: 串联所有步骤，处理参数传递和错误汇总。

**流程**:
1. 自动查找 `hawor_dir/reconstruction/hawor_results_*.npz`
2. 按顺序调用 Step 1~4 的子进程
3. `--viewer` 模式跳过视频渲染，直接启动交互式 Viewer
4. `--skip-align` 跳过对齐（使用已有的 `transform_params.npz`）
5. `--steps 1,2` 只运行指定步骤
6. 汇总所有步骤的成功/失败状态

---

### Step 1: 01_align_scene.py — 坐标系对齐

**目标**: 计算 RAS GLB 场景到 HaWoR 坐标系的变换参数。

**输入**: `ras_dir/extrinsics/*.txt` + `hawor_dir/reconstruction/hawor_results_*.npz`

**输出**: `output/alignment/transform_params.npz` (s_inv, R_inv, t_inv)

**示例**:
```bash
python 01_align_scene.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --output output/alignment/transform_params.npz
```

**可选参数**:
- `--force-scale 0.32` — 强制指定尺度因子

---

### Step 2-4: 02_render_scene.py — SAPIEN 渲染

**目标**: 在 SAPIEN 中统一渲染手部、场景物体和机器人。

**三种模式**:

| 模式 | 内容 | 用途 |
|------|------|------|
| `hand_only` | MANO 手 + GLB 物体 | 验证手-物对齐 |
| `robot_only` | R1 机械臂 + GLB 物体 | 机器人替代人手操作 |
| `robot_tracking` | MANO 手 + R1 机械臂 + GLB 物体 | 对比手部与机器人 |

**示例**:
```bash
# 手+物体渲染
python 02_render_scene.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode hand_only

# 机器人跟踪
python 02_render_scene.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode robot_tracking

# 交互式 Viewer
python 02_render_scene.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode hand_only --viewer
```

---

### Step 5: 04_physics_simulation.py — 物理引擎驱动仿真

**目标**: 用 SAPIEN PhysX 物理引擎实现真实的碰撞检测和抓取交互。

**与 02_render_scene.py 的区别**:

| 对比项 | 02 (运动学渲染) | 04 (物理仿真) |
|--------|----------------|--------------|
| 物体类型 | kinematic (静态) | **dynamic (动态)** |
| 碰撞检测 | 无 | **有 (CoACD凸分解)** |
| 关节控制 | 直接设置 qpos | **PD驱动 + 被动力补偿** |
| 抓取 | 不可能 | **可以 (高摩擦材质)** |
| 物理子步 | 无 | **8 (decimation=8)** |
| 地面 | 无 | **有 (防止物体下落)** |

**示例**:
```bash
# 快速模式 (凸包碰撞体, ~6秒加载)
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz \
    --fast-collision

# 精确模式 (CoACD凸分解, ~28分钟首次加载, 后续有缓存)
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz
```

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fast-collision` | False | 使用快速凸包碰撞体代替 CoACD |
| `--num-frames` | -1 | 渲染帧数 (-1=全部) |
| `--fps` | 30 | 视频帧率 |
| `--crf` | 18 | 视频质量 (0=无损, 51=最差) |
| `--viewer` | False | 交互式 Viewer |

**物理参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| PHYSICS_TIMESTEP | 1/240 s | 物理仿真时间步 |
| DECIMATION | 8 | 每控制步的物理子步数 |
| JOINT_STIFFNESS | 1000 | 臂关节PD刚度 |
| JOINT_DAMPING | 200 | 臂关节PD阻尼 |
| GRIPPER_STIFFNESS | 500 | 夹爪PD刚度 |
| GRIPPER_DAMPING | 50 | 夹爪PD阻尼 |
| 夹爪摩擦 (static) | 1.0 | 高摩擦实现稳定抓取 |
| 夹爪摩擦 (dynamic) | 1.0 | 高摩擦实现稳定抓取 |
| OBJECT_DENSITY | 1000 kg/m³ | 物体密度 |

---

### Step 6: 05_video_alignment.py — 视频-仿真对齐

**目标**: 验证和优化仿真与原始视频的对齐精度。

**四种模式**:

| 模式 | 功能 | 输出 |
|------|------|------|
| `overlay` | 视频叠加对比 | overlay_comparison.mp4 |
| `reproj_analysis` | 2D重投影误差分析 | reproj_analysis.mp4 + 统计 |
| `optimize` | 位姿优化 | pose_offset.npz + optimization_vis.mp4 |
| `full` | 完整流程 | 以上全部 |

**示例**:
```bash
# 视频叠加对比 (最简单, 快速诊断)
python 05_video_alignment.py \
    --hawor-dir /home/an/data/hawor/7 \
    --mode overlay \
    --sim-video output/videos/physics_sim_physics_tracking.mp4

# 2D重投影误差分析
python 05_video_alignment.py \
    --hawor-dir /home/an/data/hawor/7 \
    --mode reproj_analysis

# 位姿优化
python 05_video_alignment.py \
    --hawor-dir /home/an/data/hawor/7 \
    --mode optimize

# 完整流程
python 05_video_alignment.py \
    --hawor-dir /home/an/data/hawor/7 \
    --mode full
```

**2D重投影误差标准**:

| 误差范围 | 等级 | 含义 |
|----------|------|------|
| < 2 px | 优秀 | 几乎完美对齐 |
| 2-5 px | 良好 | 轻微偏差 |
| 5-15 px | 一般 | 可见偏移 |
| 15-30 px | 较差 | 明显偏移 |
| > 30 px | 失败 | 对齐无效 |

**当前管线误差**: 0.50 px (优秀)

---

## 对齐原理

### 变换链

```
RAS GLB 顶点 (Y-UP, VGGT 单位)
    ↓ p_hawor = s_inv * R_inv @ p_glb + t_inv
    ↓ (第一帧相机锚定 + Umeyama 尺度校正)
HaWoR Render World (Y-UP, 米制)
    ↓ RXWORLD_TO_SAPIEN = R_AXIS @ R_x
SAPIEN World (Z-UP, 米制)
```

### 关键矩阵

| 矩阵 | 值 | 作用 |
|------|-----|------|
| `R_x` | `diag(1,-1,-1)` | SLAM World ↔ Render World |
| `R_AXIS` | `[[1,0,0],[0,0,1],[0,-1,0]]` | Y-UP → Z-UP |
| `RXWORLD_TO_SAPIEN` | `R_AXIS @ R_x` | HaWoR Render → SAPIEN |
| `ZUP_TO_YUP` | `[[1,0,0],[0,0,1],[0,-1,0]]` | RAS Z-UP → Y-UP |
| `OPENCV_TO_OPENGL` | `diag(1,-1,-1)` | 相机约定转换 |

---

## 渲染模式

| 模式 | 内容 | 用途 |
|------|------|------|
| `hand_only` | MANO 手 + GLB 物体 | 验证手-物对齐 |
| `robot_only` | R1 机械臂 + GLB 物体 | 机器人替代人手操作 |
| `robot_tracking` | MANO 手 + R1 机械臂 + GLB 物体 | 对比手部与机器人 |
| `physics_tracking` | 物理仿真 + GLB 物体 | 真实碰撞和抓取 |
| `--viewer` | 交互式 | 实时调试 |

---

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hawor-dir` | 必需 | HaWoR 重建目录 |
| `--ras-dir` | 必需 | RAS 重建目录 |
| `--transform-params` | 自动 | 变换参数文件路径 |
| `--fps` | 30 | 视频帧率 |
| `--num-frames` | -1 | 渲染帧数 (-1=全部) |
| `--hand-idx` | -1 | 手索引 (0=左手, 1=右手, -1=自动检测) |
| `--viewer` | False | 交互式 Viewer |
| `--fast-collision` | False | 快速碰撞体 (04脚本) |
| `--crf` | 18 | 视频质量 |

---

## 依赖项目

- [SAPIEN](https://sapien.ucsd.edu/) — 仿真渲染引擎
- [HaWoR](https://github.com/ubc-vision/HaWoR) — 手部重建
- [ReplicateAnyScene](https://github.com/ubc-vision/ReplicateAnyScene) — 场景重建
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) — 手部到机器人映射
- [RelaxedIK](https://github.com/uwgraphics/RelaxedIK) — 逆运动学求解
- [pytransform3d](https://github.com/rock-learning/pytransform3d) — 3D 变换
- [trimesh](https://github.com/mikedh/trimesh) — 网格处理
