# Ego-Video-to-SIM

从第一人称操作视频到 SAPIEN 仿真的完整管线：将 HaWoR 手部重建与 RAS 场景重建融合，生成包含 3D 物体、MANO 手部和 R1 机械臂的仿真场景。

## 管线架构

```
输入数据
├── hawor_dir/                    # HaWoR 手部重建
│   └── reconstruction/
│       └── hawor_results_*.npz   # 手部关节、相机轨迹
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

## 快速开始

### 交互式 Viewer（推荐先运行）

```bash
python 00_run_pipeline.py \
    --hawor-dir /path/to/hawor/output \
    --ras-dir /path/to/ras/output \
    --viewer
```

### 渲染全部视频

```bash
python 00_run_pipeline.py \
    --hawor-dir /path/to/hawor/output \
    --ras-dir /path/to/ras/output
```

### 两条核心命令

| 目的 | 命令 | 说明 |
|------|------|------|
| **交互式 Viewer** | `python 00_run_pipeline.py --hawor-dir ... --ras-dir ... --viewer` | 对齐 → 启动 SAPIEN 交互式渲染 |
| **渲染视频** | `python 00_run_pipeline.py --hawor-dir ... --ras-dir ...` | 对齐 → 渲染 3 段视频 |

## 坐标系对齐原理

### 变换链

```
RAS GLB 顶点 (Y-UP, VGGT 单位)
    ↓ p_hawor = s_inv * R_inv @ p_glb + t_inv
    ↓ (第一帧相机锚定 + Umeyama 尺度校正)
HaWoR Render World (Y-UP, 米制)
    ↓ RXWORLD_TO_SAPIEN = R_AXIS @ R_x
SAPIEN World (Z-UP, 米制)
```

### 核心思想

RAS 和 HaWoR 处理同一个视频，第一帧相机的位姿在两个系统中描述同一个物理相机。以此为锚点，结合 OpenCV↔OpenGL 约定转换和 Umeyama 尺度校正，计算 RAS 世界 → HaWoR 世界的变换。

### 关键矩阵

| 矩阵 | 值 | 作用 |
|------|-----|------|
| `ZUP_TO_YUP` | `[[1,0,0],[0,0,1],[0,-1,0]]` | RAS Z-UP → Y-UP |
| `OPENCV_TO_OPENGL` | `diag(1,-1,-1)` | 相机约定转换 |
| `R_x` | `diag(1,-1,-1)` | SLAM World ↔ Render World |
| `R_AXIS` | `[[1,0,0],[0,0,1],[0,-1,0]]` | Y-UP → Z-UP |
| `RXWORLD_TO_SAPIEN` | `R_AXIS @ R_x` | HaWoR Render → SAPIEN |

## 渲染模式

| 模式 | 内容 | 用途 |
|------|------|------|
| `hand_only` | MANO 手 + GLB 物体 | 验证手-物对齐 |
| `robot_only` | R1 机械臂 + GLB 物体 | 机器人替代人手操作 |
| `robot_tracking` | MANO 手 + R1 机械臂 + GLB 物体 | 对比手部与机器人 |
| `physics_tracking` | 物理仿真 + GLB 物体 | 真实碰撞和抓取 |
| `--viewer` | 交互式: 手 + GLB + 机器人 | 实时调试 |

## 机器人映射链

```
人手 (MANO 21 关节, SAPIEN 坐标系)
    │ 取 3 个关键点: [4=拇指尖, 8=食指尖, 0=手腕]
    ▼
Dex Retargeting 优化器 → 夹爪开合 + 夹爪位姿
    │
    ├──→ FK 提取夹爪位姿 → RelaxedIK → 6 个 arm 关节角
    │
    └──→ finger_joint → SAPIEN 夹爪开合
```

**Dex Retargeting 管"手→夹爪"，RelaxedIK 管"夹爪→臂"**，两者串联完成"人手→机器人"的映射。

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hawor-dir` | 必需 | HaWoR 重建目录 |
| `--ras-dir` | 必需 | RAS 重建目录 |
| `--transform-params` | 自动 | 变换参数文件路径 |
| `--fps` | 30 | 视频帧率 |
| `--num-frames` | -1 | 渲染帧数 (-1=全部) |
| `--hand-idx` | -1 | 手索引 (0=左手, 1=右手, -1=自动检测) |
| `--viewer` | False | 交互式 Viewer (不生成视频) |
| `--force-scale` | None | 强制对齐尺度因子 |
| `--fast-collision` | False | 快速碰撞体 (04脚本) |
| `--crf` | 18 | 视频质量 |

## 对齐验证

| 指标 | 值 | 等级 |
|------|-----|------|
| R_align 旋转角度 | 0.47° | 几乎单位旋转 |
| 手腕→GLB最近顶点 | min=0.0004m | 帧5几乎接触 |
| 2D重投影误差 | 0.50 px | 优秀 (< 2px) |

## 项目结构

```
.
├── 00_run_pipeline.py          # 一键管线入口
├── 01_align_scene.py           # 坐标系对齐
├── 02_render_scene.py          # SAPIEN 渲染
├── 03_track_robot.py           # 独立机器人跟踪
├── 04_physics_simulation.py    # 物理引擎仿真
├── 05_video_alignment.py       # 视频-仿真对齐
├── doc/
│   ├── pipeline.md             # 管线详细文档
│   ├── alignment.md            # 对齐原理完整指南
│   ├── hawor_guide.md          # 操作步骤指南
│   ├── changelog.md            # 修改记录
│   └── question.md             # Q&A 文档
├── libs/                       # 本地依赖库
│   ├── dex_retargeting/        # 手部到机器人映射 (来自 dex-retargeting)
│   │   ├── constants.py        #   机器人名称、手型、重定向类型
│   │   ├── retargeting_config.py # 重定向配置加载
│   │   ├── seq_retarget.py     #   序列重定向优化器
│   │   ├── optimizer.py        #   NLopt 优化器
│   │   ├── optimizer_utils.py  #   低通滤波等工具
│   │   ├── yourdfpy.py         #   URDF 解析
│   │   ├── kinematics_adaptor.py # 运动学适配
│   │   ├── robot_wrapper.py    #   机器人封装
│   │   └── configs/            #   机器人配置文件 (YAML)
│   ├── galaxea_sim/            # Galaxea 仿真框架 (来自 GalaxeaManipSim)
│   │   ├── controllers/utils/
│   │   │   ├── relaxed_ik_solver.py  # RelaxedIK 求解器
│   │   │   ├── relaxed_ik.py         # RelaxedIK 核心
│   │   │   ├── kinematics.py         # 运动学工具
│   │   │   └── librelaxed_ik_lib.so  # RelaxedIK 预编译库
│   │   └── assets/r1/          #   R1 机器人模型资源
│   │       ├── configs/urdfs/  #     URDF 文件
│   │       └── meshes/         #     网格和纹理
│   └── position_retargeting/   # 位置重定向辅助 (来自 dex-retargeting/example)
│       ├── mano_layer.py       #   MANO 手部模型层
│       └── hand_robot_viewer.py #  机器人手部查看器
├── output/                     # 管线输出
│   ├── alignment/              # 对齐参数和报告
│   ├── alignment_analysis/     # 对齐分析结果
│   ├── tracking/               # 跟踪数据 (npy)
│   └── videos/                 # 渲染视频
└── test/                       # 调试和测试脚本
```

## 依赖说明

### 本地依赖 (已包含在 `libs/` 目录)

| 库 | 来源 | 用途 |
|------|------|------|
| `dex_retargeting` | [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) | 手部到机器人夹爪映射 (3约束点优化) |
| `galaxea_sim` | [OpenGalaxea/GalaxeaManipSim](https://github.com/OpenGalaxea/GalaxeaManipSim) | RelaxedIK 逆运动学求解 + R1 机器人模型 |
| `position_retargeting` | dex-retargeting/example | MANO 手部模型层 |

### pip 安装的第三方库

```bash
pip install numpy opencv-python sapien torch joblib pytransform3d tqdm trimesh scipy natsort matplotlib imageio-ffmpeg
```

### 上游数据依赖 (需单独安装)

| 项目 | 用途 | 安装方式 |
|------|------|---------|
| [HaWoR](https://github.com/ubc-vision/HaWoR) | 手部重建 | 按官方文档安装 |
| [ReplicateAnyScene](https://github.com/ubc-vision/ReplicateAnyScene) | 场景重建 | 按官方文档安装 |
| MANO 模型文件 | 手部网格生成 | 从 [MANO官网](https://mano.is.tue.mpg.com/) 下载
