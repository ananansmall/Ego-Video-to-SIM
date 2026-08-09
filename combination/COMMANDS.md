# COMMANDS.md — 调用方式大全

所有路径相对于 `example/combination/`。

---

## 1. 一键管线 (`00_run_pipeline.py`)

### 1.1 完整管线 (默认 `--mode full`)

```bash
python 00_run_pipeline.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

执行: `1(01_align) → 2(hand_only) → 3(robot_only) → 4(robot_tracking) → 5(topdown)`

### 1.2 新坐标管线 (`--mode align-render`)

```bash
python 00_run_pipeline.py --mode align-render \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --view fpv --num-frames 30
```

执行: `001_align_scene.py → 002_render_scene.py`（GLB 原始坐标系，无 ZUP_TO_YUP）

### 1.3 深度校正 + 新坐标管线 (`--mode align-render-depth`)

```bash
python 00_run_pipeline.py --mode align-render-depth \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --view fpv --num-frames 30
```

执行: `01c_depth_align.py → 001_align_scene.py → 002_render_scene.py`

### 1.4 深度校正 + 完整管线 (`--mode full-depth`)

```bash
python 00_run_pipeline.py --mode full-depth \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

执行: `01c_depth_align.py → 01_align_scene.py → 02_render_scene.py (所有mode)`

### 1.5 HandTrack 管线 (`--handtrack`)

```bash
python 00_run_pipeline.py --handtrack \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

执行: `1(01_align) → 7(render_auto.py)`
可选参数: `--optimizer` `--handedness {auto,left,right,both}`

### 1.6 灵巧手管线 (`--dexterous`)

```bash
python 00_run_pipeline.py --dexterous --robot-name allegro \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

执行: `1(01_align) → 8(render_dexterous_only.py)`
`--robot-name`: `allegro`, `inspire`, `shadow`, `ability`, `leap`, `svh`

### 1.7 自定义步骤 (`--steps`)

```bash
# 只跑对齐+hand_only+物理仿真
python 00_run_pipeline.py --steps 1,2,6 \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 跳过对齐
python 00_run_pipeline.py --skip-align --steps 2,3 \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

步骤编号: `1=对齐`, `2=hand_only`, `3=robot_only`, `4=robot_tracking`, `5=topdown`, `6=物理仿真`, `7=hand_track`, `8=灵巧手`

### 1.8 Viewer 交互模式 (`--viewer`)

```bash
# 新坐标管线 + Viewer（交互式，不生成视频）
python 00_run_pipeline.py --mode align-render --viewer \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 旧坐标管线 + Viewer（对齐后直接交互）
python 00_run_pipeline.py --viewer \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

对齐后启动 SAPIEN 交互式渲染窗口（不生成视频），按 ESC 或关闭窗口退出。

### 通用参数

| 参数 | 说明 |
|------|------|
| `--hand-idx -1` | -1=自动检测, 0=左手, 1=右手 |
| `--glb-path` | GLB 路径, 默认 `<ras-dir>/final_scene.glb` |
| `--fixed-base` | 固定基座模式 |
| `--view fpv` | fpv / topdown / behind / front |
| `--fps 60` | 视频帧率 |
| `--crf 14` | 视频质量 (0=无损) |
| `--num-frames -1` | 最大帧数, -1=全部 |

---

## 2. 坐标对齐（独立运行）

### 2.1 新对齐 `001_align_scene.py`（GLB 原始坐标）

```bash
python 001_align_scene.py \
    --ras_output /home/an/data/ras/my_7mp4_result \
    --hawor_reconstruction /home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz \
    --output_dir ./output/alignment \
    --force_scale 1.0
```

### 2.2 旧对齐 `01_align_scene.py`（SAPIEN 坐标）

```bash
python 01_align_scene.py \
    --ras_output /home/an/data/ras/my_7mp4_result \
    --hawor_reconstruction /home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz \
    --output_dir ./output/alignment
```

---

## 3. 渲染（独立运行）

### 3.1 新坐标渲染 `002_render_scene.py`（5 种 mode）

```bash
python 002_render_scene.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --glb-path /home/an/data/ras/my_7mp4_result/final_scene.glb \
    --transform-params /home/an/data/hawor/7/transform_params.npz \
    --hand-idx -1 --mode robot_tracking --view fpv \
    --fps 30 --crf 18 --num-frames 100 \
    --output /tmp/demo.mp4
```

| `--mode` | 渲染内容 |
|----------|----------|
| `gripper_only` | 夹爪 URDF（支持手臂），新策略 `--strategy aligned/analytical` |
| `hand_only` | MANO 手部关键点 + GLB 物体 |
| `robot_only` | R1 机器人 + GLB 物体 |
| `robot_tracking` | MANO 关键点 + R1 机器人 + GLB 物体 |
| `topdown` | 同 robot_only, 俯视图 |

其他参数: `--fixed-base` `--gripper-mode {gripper,gripper_arm,both}`
`--strategy {aligned,analytical}` `--open-scale 1.5` `--smooth {0,1}`
`--start-frame 0` `--width 1920` `--height 1080`

### 3.2 旧坐标渲染 `02_render_scene.py`（3 种 mode）

```bash
python 02_render_scene.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz \
    --mode robot_tracking \
    --output /tmp/old_demo.mp4
```

`--mode`: `hand_only` / `robot_only` / `robot_tracking`

---

## 4. 物理仿真 `04_physics_simulation.py`

### 4.1 完整模式 (夹爪+机械臂)

```bash
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode physics_tracking \
    --hide-hand --fast-collision --num-frames 300
```

### 4.2 单夹爪模式 (Dex Retargeting 优化器驱动)

```bash
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --single-gripper \
    --hide-hand --fast-collision --num-frames 300
```

`--single-gripper`: 只加载夹爪 URDF（无机械臂），夹爪位姿通过 Dex Retargeting 优化器计算

参数: `--viewer` `--speed 1.0` `--smooth 1` `--no-support-table`
`--fixed-base` `--no-fixed-base` `--fast-collision` `--hide-hand` `--num-frames`

---

## 5. Hand Track 子脚本

### 5.1 自动检测渲染 `hand_track/render_auto.py`

```bash
python hand_track/render_auto.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode both --hand-idx -1 \
    [--optimizer]
```

`--mode`: `gripper`(仅夹爪), `gripper_arm`(夹爪+末端), `both`(两者)

### 5.2 夹爪渲染 `hand_track/render_gripper_only.py`

```bash
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode both --arm-mode half \
    [--optimizer] [--strategy aligned] [--viewer]
```

`--arm-mode`: `half`(半臂), `full`(全臂)
`--strategy`: `aligned`(对齐), `pca`, `scaled`, `identity`
`--mode`: `gripper`, `gripper_arm`, `both`

### 5.3 灵巧手渲染 `hand_track/render_dexterous_only.py`

```bash
python hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name allegro --hand-idx -1 \
    [--viewer] [--smooth 1] [--verify]
```

`--robot-name`: `allegro`, `inspire`, `shadow`, `ability`, `leap`, `svh`

---

## 实用组合示例

```bash
# 新坐标管线 + 深度校正：只跑 50 帧，fpv 视角，最高质量
python 00_run_pipeline.py --mode align-render-depth \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --view fpv --num-frames 50 --crf 0

# 旧坐标管线 + 深度校正：完整跑一遍
python 00_run_pipeline.py --mode full-depth \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 仅新坐标机器人跟踪，自动检测双手，左右并排
python 002_render_scene.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --glb-path /home/an/data/ras/my_7mp4_result/final_scene.glb \
    --transform-params /home/an/data/hawor/7/transform_params.npz \
    --hand-idx -1 --mode robot_tracking --view fpv \
    --fps 30 --num-frames 200 \
    --output /tmp/dual_hand.mp4

# 灵巧手单独调试
python hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name shadow --hand-idx 0 --viewer

# 物理仿真
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --hide-hand --fast-collision --num-frames 300
```

---

## 6. 深度校正管线 (`01c_depth_align.py`)

> **重要**: 01c v2 必须在 001 之后运行！管线顺序: `001 → 01c → 002`
>
> v2 核心改动: 深度校正在 GLB 空间中进行，使用 001 的 scale_ratio 统一量纲。
> 不再在 HaWoR 相机坐标系中比较深度，而是把 HaWoR 手腕变换到 GLB 空间后，
> 通过 RAS 外参投影到 RAS 相机坐标系，直接和深度图的 z 比较。

### 6.1 完整管线: 001 → 01c → 002

```bash
# Step 1: 对齐 (必须先跑)
conda run -n dex python 001_align_scene.py \
    --ras_output /home/an/data/ras/my_7mp4_result \
    --hawor_reconstruction /home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz \
    --output_dir ./output/alignment

# Step 2: 深度校正 (需要 --transform-params)
conda run -n dex python 01c_depth_align.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz

# Step 3: 渲染 (自动使用 depth_aligned)
conda run -n dex python 002_render_scene.py \
    --mode hand_only \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz \
    --output ./output/videos/hand_object_depth_aligned.mp4 \
    --fps 30 --num-frames 30
```

### 6.2 01c dry-run (只看校正因子，不保存)

```bash
conda run -n dex python 01c_depth_align.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz \
    --dry-run
```

### 6.3 Viewer 交互模式 (深度校正后)

```bash
conda run -n dex python 002_render_scene.py \
    --mode hand_only \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz \
    --viewer --num-frames 30
```

### 6.4 无深度校正对比 (用原始 HaWoR 数据)

```bash
conda run -n dex python 002_render_scene.py \
    --mode hand_only \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/alignment/transform_params.npz \
    --no-depth-align --viewer --num-frames 30
```

> **01c v2 变更说明:**
> - 管线顺序: `001 → 01c` (旧版: `01c → 001`)
> - 必须参数: `--transform-params` (001 的输出)
> - 量纲统一: 用 001 的 scale_ratio 把 HaWoR 手腕变换到 GLB 空间后比较
> - 输出文件: 只生成 `_depth_aligned.npz` (不再单独生成 `_factors.npz`)
> - 02_render: 默认优先使用 `_depth_aligned.npz`，`--no-depth-align` 跳过

> **输出文件说明:**
> - 原始: `hawor_results_0_113.npz` (只读，不修改)
> - 校正: `hawor_results_0_113_depth_aligned.npz` (只改 pred_trans，其他 MANO 参数不变)