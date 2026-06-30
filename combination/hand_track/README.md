# hand_track — 自动手部检测 + 机械臂/夹爪渲染

> **核心功能**: 从 HaWoR 手部数据 + RAS 3D 场景自动渲染机械臂/夹爪跟踪视频。
> 只需两个目录即可运行，坐标对齐参数自动生成。

---

## 1. 模块组成

| 文件 | 作用 |
|---|---|
| `render_auto.py`            | **主入口** — 自动检测手部, 渲染机械臂 + 夹爪 URDF 视频 |
| `render_gripper_only.py`    | **夹爪专用** — 只渲染夹爪 URDF (不加载手臂), 支持夹爪+手臂末端模式 |
| `render_dexterous_only.py`  | **灵巧手专用** — 渲染多指灵巧手 URDF (allegro/inspire/shadow/ability/leap/svh), 使用 dex-retargeting 优化器 |
| `align_strategy.py`         | **对齐策略** — 新策略: 先对齐夹爪两点, 再用中点-手腕连线确定位姿, 机械臂手腕放到位姿线上 |
| `gripper_config.py`         | 夹爪配置 — URDF 模板、生成函数、几何常量、平滑器、retargeting 初始化 |
| `common.py`                 | 共享函数 — 场景设置、GLB 加载、关键点渲染、坐标变换、手部检测等 |
| `configs/`                  | YAML 配置 — 左右手夹爪 retargeting 配置 (`r1_gripper_left.yml`, `r1_gripper_right.yml`) |
| `run_all_hawor.py`          | 批量入口 — 扫描 `--hawor-base` 下所有 HaWoR 目录 (旧管线) |
| `docs/questions.md`         | 问答记录 |

---

## 2. 快速开始

### 2.1 最简用法 (只需两个目录)

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

# 机械臂 + 夹爪 URDF 视频 (自动检测左手/右手/双手)
python hand_track/render_auto.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result
```

**不需要手动指定 `--transform-params`** — 脚本会自动运行 `01_align_scene.py` 生成坐标对齐参数。

### 2.2 只渲染夹爪 (`render_gripper_only.py`)

**关键概念: `--mode` 和 `--arm-mode` 的关系**

| 参数 | 作用 | 取值 | 默认 |
|---|---|---|---|
| `--mode` | **选择渲染什么** | `gripper` / `gripper_arm` / `both` | `both` |
| `--arm-mode` | **`gripper_arm` 模式下, 选择哪种手臂** | `half` / `full` | `half` |

- `--mode gripper_arm` 是**开启手臂模式的开关** (必填)
- `--arm-mode full` 是**可选的手臂类型选择** (只在 `--mode gripper_arm` 时生效)
- 不加 `--arm-mode` 时默认 `half` (半个手臂 link4-6)

**所有可用命令** (已验证全部可运行):

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

# ① 默认: 同时渲染 gripper + gripper_arm (gripper_arm 用 half 手臂)
# 输出: hawor_r1_{side}_gripper_urdf.mp4 + hawor_r1_{side}_gripper_urdf_arm.mp4
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# ② 仅夹爪 (不加载手臂, finger origin=37mm, 与 02_render_scene.py 一致)
# 输出: hawor_r1_{side}_gripper_urdf.mp4
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode gripper

# ③ 夹爪 + 半个手臂 (arm_link4-6, 默认 --arm-mode half)
# 输出: hawor_r1_{side}_gripper_urdf_arm.mp4
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode gripper_arm

# ④ 夹爪 + 完整手臂 (arm_link1-6, 需显式指定 --arm-mode full)
# 输出: hawor_r1_{side}_gripper_urdf_arm.mp4
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode gripper_arm --arm-mode full

# ⑤ 使用优化器模式 (默认是解析模式)
# 输出: hawor_r1_{side}_gripper_urdf.mp4
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode gripper --optimizer

# ⑥ 强制只渲染左手 (忽略自动检测)
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --hand-idx 0

# ⑦ 使用旧策略 (Gram-Schmidt, 不带开合缩放)
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --strategy analytical

# ⑧ 调整夹爪开合缩放 (默认 3.0, 1.0=精确映射)
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --open-scale 5.0
```

> **注**: `render_auto.py` 的 `--mode` 默认是 `gripper`（不支持 `both`），只生成一个夹爪URDF视频。
> 如需同时获得 gripper 和 gripper_arm 视频，请用 `render_gripper_only.py`。

### 2.3 SAPIEN Viewer 实时循环播放

```bash
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --viewer
```

### 2.4 验证指尖误差

```bash
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --verify --num-frames 30
```

### 2.5 灵巧手渲染 (`render_dexterous_only.py`)

**核心特点**: 把 MANO 人手关节位置映射到多指灵巧手 URDF, 用 `dex_retargeting` PositionOptimizer 同时求解腕部 6DOF 位姿 (dummy free joint) 和 N 个手指关节, 仅渲染手部 (不渲染机械臂)。

#### 环境要求

```bash
# 必须使用 dex conda 环境 (含 sapien, dex_retargeting, pytransform3d, mano_layer)
PYTHON=/home/an/miniconda3/envs/dex/bin/python
```

#### 支持的灵巧手

灵巧手 URDF + retargeting YAML 配置位于 `dex-retargeting/assets/robots/hands/<name>/`:

| `--robot-name` | 总关节数 | 手指关节数 | target links | 指数 | URDF 文件 |
|---|---|---|---|---|---|
| `allegro` (默认) | 22 | 16 | 8 | 4 | `allegro_hand_{L,R}.urdf` + `_glb.urdf` |
| `inspire` | 18 | 12 | 5 | 5 | `inspire_hand_{R,L}.urdf` (无 `_glb` 版本, 自动回退) |
| `shadow`  | 30 | 24 | 10 | 5 | `shadow_hand_{L,R}.urdf` + `_glb.urdf` |
| `ability` | 16 | 10 | 5 | 5 | `ability_hand_{L,R}.urdf` + `_glb.urdf` |
| `leap`    | 22 | 16 | 8 | 4 | `leap_hand_{L,R}.urdf` + `_glb.urdf` |
| `svh`     | 26 | 20 | 10 | 5 | `svh_hand_{L,R}.urdf` + `_glb.urdf` |

> "_glb" 版本使用 glb mesh, 视觉效果更好; 不存在时自动回退到原始 URDF (如 inspire)。

#### 快速开始 (5 分钟生成第一个视频)

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
PYTHON=/home/an/miniconda3/envs/dex/bin/python

# Step 1: 默认 allegro 右手 + 自动手部检测 + 自动坐标对齐
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 输出: hand_track/output/7/videos/hawor_allegro_right_urdf.mp4
```

#### 完整命令示例

**① 切换不同灵巧手** (`--robot-name`):

```bash
# 切换为 5 指类人手 inspire
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name inspire

# 切换为 Shadow Hand (5 指, 10 target links)
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name shadow

# 切换为 Leap Hand (4 指, 与 allegro 同关节数)
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name leap
```

**② 手部选择** (`--hand-idx`):

```bash
# -1 (默认): 自动检测 (调用 detect_hands, 适用于大多数场景)
$PYTHON hand_track/render_dexterous_only.py ... --hand-idx -1

# 0: 强制渲染左手 (跳过自动检测, 用于 HaWoR 误检背景手时)
$PYTHON hand_track/render_dexterous_only.py ... --hand-idx 0

# 1: 强制渲染右手
$PYTHON hand_track/render_dexterous_only.py ... --hand-idx 1
```

**③ 双手自动渲染**: 当 `--hand-idx -1` 且 `detect_hands` 检测出双手时, 自动调用 `render_dual_dexterous_video`, 在同一场景渲染左右灵巧手, 输出 `hawor_{robot}_dual_urdf.mp4`。

**④ 相机视角** (`--view`):

```bash
# fpv (默认): 跟随 HaWoR 相机轨迹, 适合第一人称视频
$PYTHON hand_track/render_dexterous_only.py ... --view fpv

# behind: 后上方俯视, 适合看手部整体动作
$PYTHON hand_track/render_dexterous_only.py ... --view behind

# front: 正前方观察, 适合看抓取细节
$PYTHON hand_track/render_dexterous_only.py ... --view front

# topdown: 顶部俯视 (内部按 behind 处理, 同 render_gripper_only.py)
$PYTHON hand_track/render_dexterous_only.py ... --view topdown
```

**⑤ SAPIEN Viewer 实时循环** (`--viewer`): 不保存视频, 在交互窗口中循环播放动画, 适合调试:

```bash
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name allegro --viewer
```

**⑥ 验证指尖误差** (`--verify`): 不调用 `scene.step()`, 计算每帧机器人指尖 vs MANO 目标点的位置误差 (mm), 输出 mean/max 统计:

```bash
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name allegro --verify --num-frames 30
```

**⑦ 调试少量帧** (`--num-frames`): 只渲染前 N 帧, 快速验证:

```bash
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name allegro --num-frames 8 --width 640 --height 360
```

**⑧ 自定义输出目录** (`--output-dir`):

```bash
$PYTHON hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --output-dir /tmp/dex_test
```

#### 通过一键管线调用 (`00_run_pipeline.py --dexterous`)

`00_run_pipeline.py` 集成了 Step 1 (坐标对齐) + Step 8 (灵巧手渲染), 推荐用于完整生产流程:

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

# 默认: allegro, 步骤 1+8
python 00_run_pipeline.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --dexterous

# 切换 inspire
python 00_run_pipeline.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --dexterous --robot-name inspire

# 跳过对齐 (已有 transform_params.npz)
python 00_run_pipeline.py ... --dexterous --skip-align

# 仅渲染灵巧手 (跳过对齐)
python 00_run_pipeline.py ... --dexterous --steps 8

# 双手自动检测 (通过 --handedness 映射到 --hand-idx)
python 00_run_pipeline.py ... --dexterous --handedness both    # → --hand-idx -1
python 00_run_pipeline.py ... --dexterous --handedness left    # → --hand-idx 0
python 00_run_pipeline.py ... --dexterous --handedness right   # → --hand-idx 1
```

#### 输出文件命名规则

| 场景 | 输出文件名 | 说明 |
|---|---|---|
| 单手 (左手) | `videos/hawor_{robot}_left_urdf.mp4` | `--hand-idx 0` 或自动检测为左手 |
| 单手 (右手) | `videos/hawor_{robot}_right_urdf.mp4` | `--hand-idx 1` 或自动检测为右手 |
| 双手 | `videos/hawor_{robot}_dual_urdf.mp4` | `--hand-idx -1` 且 `detect_hands` 返回 `[0,1]` |
| 默认输出目录 | `hand_track/output/{hawor_name}/videos/` | 不指定 `--output-dir` 时 |
| Viewer 模式 | (无文件) | `--viewer` 不保存视频, 仅实时渲染 |

#### 执行流程 (3 步)

脚本 `main()` 函数自动执行:

```
[1/3] 手部检测
    ├── --hand-idx >= 0: 直接使用指定值
    └── --hand-idx -1:  调用 detect_hands(hawor_dir) 自动检测
        └── 返回 [0](左)/[1](右)/[0,1](双手)

[2/3] 准备 GLB 变换参数
    ├── 检查 {output_dir}/transform_params.npz 是否存在
    └── 不存在则自动运行 01_align_scene.py 生成
        (Umeyama 算法: RAS GLB → HaWoR 坐标系)

[3/3] 渲染灵巧手视频
    ├── 单手: render_dexterous_only_video()
    └── 双手: render_dual_dexterous_video()
```

#### 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: sapien` | 未使用 dex 环境 | 改用 `/home/an/miniconda3/envs/dex/bin/python` |
| `inspire_hand_right_glb.urdf is not a file` | inspire 无 `_glb` 版本 | 已自动回退, 无需处理 |
| 视频为黑屏/灰屏 | 渲染但未保存, 或 GLB 未加载 | 检查 `transform_params.npz` 是否存在, 检查日志中 `GLB 加载成功` 行 |
| `IndexError: too many indices for array` | `pred_hand_pose` 索引错误 | 已修复为 `hand_pose_frame[0:3]` (1D 索引) |
| 渲染耗时过长 | 默认 1920x1080 + 全部帧 | 用 `--num-frames 30 --width 640 --height 360` 调试 |
| 灵巧手位置偏移 | warm start 未生效 | 检查日志中 `Warm start 完成` 是否出现 |

---

## 3. CLI 参数

### render_auto.py

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hawor-dir`       | (必填) | HaWoR 数据目录 |
| `--ras-dir`         | (必填) | RAS 重建结果目录 (含 `final_scene.glb`) |
| `--output-dir`      | 自动   | 输出目录 (默认 `hand_track/output/{hawor_name}`) |
| `--mode`            | `gripper` | 夹爪URDF模式: `gripper`=仅夹爪, `gripper_arm`=夹爪+手臂末端 (不支持 `both`) |
| `--hand-idx`        | `-1`   | `-1`=自动检测, `0`=强制左手, `1`=强制右手 |
| `--fps`             | `30`   | 视频帧率 |
| `--view`            | `fpv`  | 相机视角: `fpv` / `behind` / `front` / `topdown` |
| `--width`           | `1920` | 渲染宽度 |
| `--height`          | `1080` | 渲染高度 |
| `--crf`             | `18`   | H.264 CRF 质量参数 |
| `--start-frame`     | `0`    | 起始帧索引 |
| `--num-frames`      | `-1`   | 处理帧数 (-1=全部) |

### render_gripper_only.py

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hawor-dir`       | (必填) | HaWoR 数据目录 |
| `--ras-dir`         | (必填) | RAS 重建结果目录 |
| `--output-dir`      | 自动   | 输出目录 |
| `--mode`            | `both` | `gripper`=仅夹爪, `gripper_arm`=夹爪+手臂, `both`=两者都渲染 |
| `--arm-mode`        | `half` | `gripper_arm` 模式的手臂类型: `half`=半个手臂(link4-6), `full`=完整手臂(link1-6) |
| `--smooth`          | `1`    | `0`=不平滑, `1`=EMA平滑 |
| `--hand-idx`        | `-1`   | `-1`=自动检测, `0`=强制左手, `1`=强制右手 |
| `--optimizer`       | off    | 使用优化器模式 (默认: 解析模式) |
| `--strategy`        | `aligned` | 对齐策略: `aligned`=新策略(先对齐夹爪两点+中点手腕连线确定位姿), `analytical`=旧策略(Gram-Schmidt) |
| `--open-scale`      | `3.0`  | 夹爪开合缩放因子 (仅 `aligned` 策略; 3.0=放大开合效果, 1.0=精确映射) |
| `--viewer`          | off    | SAPIEN Viewer 实时循环播放 (不保存视频) |
| `--verify`          | off    | 计算并输出指尖位置/手腕位姿误差 |
| `--fps` / `--view` / `--width` / `--height` / `--crf` | 同上 | 同 render_auto.py |
| `--start-frame` / `--num-frames` | 同上 | 同 render_auto.py |

---

## 4. 实现方式

### 4.1 手部检测 (`detect_hands`)

**数据来源**: 读取 HaWoR 输出的 `hawor_results_*.npz` 文件。

HaWoR 的数据约定：
- `pred_valid` 形状 `(2, N)`: 第0行=左手，第1行=右手，N=帧数
- `pred_trans` 形状 `(2, N, 3)`: 每只手每帧的3D平移
- `pred_betas` 形状 `(2, N, 10)`: 每只手每帧的MANO形状参数
- `hand_idx=0` → 左手 (MANO left)
- `hand_idx=1` → 右手 (MANO right)

**检测算法** (`common.py` → `detect_hands()`):

1. 找到 `hawor_dir/reconstruction/hawor_results_*.npz` 文件
2. 对每只手 (hand_idx=0, 1):
   - 统计 `pred_valid[hi]=True` 且 `pred_trans`/`pred_betas` 不含 NaN 的帧数
   - 有效帧占比 >= 5% → 进入候选
   - **运动幅度检查**: 有效帧手腕位置的 `max - min` (运动范围) 必须 >= 3cm；HaWoR 常把不存在的手输出为静止在原点的"幽灵手"
   - **原点噪声检查**: 若手腕位置均值在原点半径 5cm 内且运动范围 < 5cm，视为误检，直接排除
3. 若两只手都进入候选，检查它们平均位置的距离:
   - 距离 < 10cm → 视为同一只手被重复标记，保留运动范围更大的那只
4. 返回手部索引列表: `[]` (无有效手), `[0]`(左手), `[1]`(右手), 或 `[0,1]`(双手)

**无有效数据处理**: 如果两只手都未通过检测 (全 NaN 或持续 `pred_valid=False`)，脚本会立即报错退出，不会生成任何视频。

**覆盖检测**: `--hand-idx 0` 或 `--hand-idx 1` 强制指定，忽略自动检测结果。`render_auto.py` 和 `render_gripper_only.py` 都支持此参数，适用于 HaWoR 误检背景手的情况。

**实测结果**:

| 目录 | `pred_valid` 原始统计 | `detect_hands` 结果 | 实际视频 |
|---|---|---|---|
| `7`             | 左 100%, 右 99.1% | `[0]` 左手  | 仅左手 |
| `7_vggt-omega`  | 左 100%, 右 99.1% | `[0]` 左手  | 仅左手 |
| `hoi4d`         | 左 99.8%, 右 99.8% | `[0,1]` 双手 | 双手 |
| `laptop`        | 左 0%, 右 82.7%   | `[1]` 右手  | 仅右手 |

### 4.2 坐标对齐

`--transform-params` **不需要手动指定**。脚本内置 `_ensure_transform_params()`:
1. 检查输出目录下是否已有 `transform_params.npz`
2. 如不存在，自动运行 `01_align_scene.py` 生成
3. 生成后直接使用

`01_align_scene.py` 使用 Umeyama 算法计算 RAS → HaWoR 的坐标变换:
```
p_hawor = s_inv * R_inv @ p_glb + t_inv
```
包含缩放 (`s_inv`)、旋转 (`R_inv`)、平移 (`t_inv`) 三个分量。

### 4.3 夹爪 URDF 加载 (与 02_render_scene.py 一致)

**核心原则**: finger origin 始终使用 URDF 原始值 `0.03689` (37mm), **不缩放**, 与 `02_render_scene.py` 完全一致。

**`02_render_scene.py` 的 URDF 加载方式** (参考):
- Retargeting URDF: `assets/robots/r1_full/r1_lite_robot_glb.urdf` (完整机器人, finger origin=37mm)
- Rendering URDF: `GalaxeaManipSim/.../r1_v2_1_0_floating_{side}.urdf` (浮动 URDF, `prepare_arm_urdf` 只替换 mesh 路径 + finger joint fixed→prismatic)

**我们的 URDF 加载方式**:

| 模式 | Retargeting URDF | Rendering URDF | finger origin |
|---|---|---|---|
| `--mode gripper` | `generate_gripper_urdf()` (夹爪专用, 3 links) | 同左 | 37mm (不缩放) |
| `--mode gripper_arm --arm-mode half` | `generate_gripper_urdf()` | `prepare_half_arm_urdf()` (从浮动 URDF 提取 link4-6) | 37mm (不缩放) |
| `--mode gripper_arm --arm-mode full` | `generate_gripper_urdf()` | `prepare_full_arm_urdf()` (完整浮动 URDF) | 37mm (不缩放) |

**URDF 几何对比** (验证 finger origin 一致):

| URDF 来源 | `gripper_finger_joint1` origin X | `gripper_finger_joint2` origin X |
|---|---|---|
| `r1_lite_robot_glb.urdf` (02_render_scene.py retargeting) | 0.03689 | 0.03689 |
| `r1_v2_1_0_floating_right.urdf` (02_render_scene.py rendering) | 0.03689 | 0.03689 |
| 我们的 `generate_gripper_urdf()` | 0.03689 | 0.03689 |
| 我们的 `prepare_half/full_arm_urdf()` | 0.03689 | 0.03689 |

### 4.4 夹爪位姿计算

#### 对齐策略 (`--strategy`, 默认 `aligned`)

夹爪位姿计算有两种对齐策略，通过 `--strategy` 参数选择：

**`aligned` 策略 (默认, 推荐)** — `align_strategy.py`

用户要求的核心对齐逻辑：
> "关键先对齐夹爪两点, 最后第三个点, 是对齐位姿, 能够在同一条中轴线上即可"
> "MANO参数里面夹爪的中点和第三个手腕点的连线确定位姿"
> "对齐的主要是夹爪末端, 次要是手腕, 但位姿一定要对"

实现步骤：
1. **输入**: MANO 的3个关键点 — 手腕(joint0)、拇指尖(joint4)、食指尖(joint8)
2. **主要: 对齐夹爪两点** — 指尖中点 `midpoint = (finger1 + finger2) / 2`
3. **位姿核心: 中点-手腕连线确定 X 轴** — `X = normalize(midpoint - wrist)` (指向方向)
4. **Y 轴 (开合方向)** — `finger2-finger1` 投影到 X 垂直面后归一化
5. **Z 轴** — `Z = X × Y`, 组装旋转矩阵 `R = [X, Y, Z]`
6. **手指关节** — `joint = (指尖3D距离 - 基准距离) × open_scale / 2` (带缩放因子让开合更明显)
7. **gripper_link 位置** — `gripper_pos = midpoint - R @ finger_mid_in_gripper` (对齐夹爪两点)
8. **次要: 机械臂手腕到位姿线** — `gripper_arm` 模式下, arm wrist 通过 offset 反推, 落在位姿线上

**`analytical` 策略 (旧)** — `gripper_config.py`

Gram-Schmidt 正交化，不带开合缩放：
1. X轴 = normalize(指尖中点 - 手腕) → 指向方向
2. Y轴 = normalize(拇指→食指 投影到X垂直面) → 开合方向
3. Z轴 = X × Y
4. 手指关节: 从指尖距离残差到关节轴方向的投影 (无缩放)
5. gripper_link 位置: 匹配指尖中点, 在指尖中点后方 37mm 处
6. offset补偿: gripper_arm 模式下补偿 gripper_link 相对于 root 的偏移

**两种策略的区别**:

| 对比项 | `aligned` (新, 默认) | `analytical` (旧) |
|---|---|---|
| 夹爪开合 | 带缩放因子 (`--open-scale`, 默认 3.0), 开合明显 | 无缩放, 开合很小 (MANO 拇指-食指距离仅 ~35mm) |
| 位姿核心 | 中点-手腕连线确定 X 轴 (用户要求) | Gram-Schmidt 正交化 |
| 手腕对齐 | 次要, 落在位姿线上即可 | 不强制 |
| 适用场景 | 默认推荐, 夹爪开合可见 | 需要精确映射指尖距离时 |

**为什么需要 `--open-scale`**: MANO 拇指-食指指尖距离通常只有 ~35mm (静态捏取姿态), 而夹爪基准距离 `FINGER_BASE_DIST = 26.9mm`, 精确映射时关节值仅 ~4.3mm, 几乎看不到开合。`--open-scale 3.0` 将关节值放大到 ~12.8mm, 开合效果明显。代价是引入 ~11mm 指尖位置误差 (1-DOF 夹爪无法匹配 MANO 3D 开合方向的固有局限)。

性能 (数据集7, `--verify` 实测):

| 策略 | 模式 | open_scale | 指尖误差 | 位姿误差 | 手腕到位姿线 |
|---|---|---|---|---|---|
| `aligned` | `--mode gripper` | 3.0 | ~16.7mm | 0.02° | - |
| `aligned` | `--mode gripper_arm` | 3.0 | ~16.7mm | 0.03° | 0.0mm (完美在位姿线上) |
| `aligned` | `--mode gripper` | 1.0 | ~11.8mm | 0.02° | - |
| `analytical` | `--mode gripper` | - | 4.36mm | 0.14° | - |
| `analytical` | `--mode gripper_arm` | - | 4.36mm | 0.14° | - |

> **位姿误差 < 0.1° (优秀)**: 两种策略的位姿核心都正确, `aligned` 策略位姿更精确。
> **`gripper_arm` 模式手腕在位姿线上**: `aligned` 策略下 arm wrist 完美落在中点-手腕连线上 (0.0mm), 机械臂不会"乱飞"。
> **指尖误差权衡**: `analytical` 指尖误差小 (4.36mm) 但开合不可见; `aligned` 指尖误差大 (~16.7mm) 但开合明显。这是 1-DOF 夹爪的固有局限。

#### 优化器模式 (`--optimizer`)

使用 `dex_retargeting` PositionOptimizer + gripper-only URDF 求解:
- URDF: 8 DOF = 6 dummy free joint (root位姿) + 2 finger prismatic joint
- 约束: 3 target links × 3D = 9 约束
- 优化器: NLopt SLSQP (局部优化，可能陷入局部最优)
- Warm start: 用对齐策略位姿初始化，避免从零开始
- `--strategy` 参数对优化器模式同样生效 (影响 warm start 初始位姿)

### 4.5 相机跟踪

与 `02_render_scene.py` 一致: 当 HaWoR 相机轨迹 (`R_c2w`, `t_c2w`) 存在时，每帧更新相机位置和朝向。相机坐标变换:
```
cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w
cam_R_sapien = RXWORLD_TO_SAPIEN @ R_c2w
```
然后转换为 SAPIEN 相机坐标系 (forward=-Z, left=-X, up=Y)。

### 4.6 GLB 场景加载

1. 用 trimesh 加载 `final_scene.glb`
2. 对每个几何体的顶点应用坐标变换: `vertices_sapien = RXWORLD_TO_SAPIEN @ (s_inv * R_inv @ vertices.T + t_inv).T`
3. 导出为临时 PLY 文件，用 SAPIEN actor builder 加载
4. 创建为 kinematic actor (不受物理影响)

注意: `s_inv` 会缩放 GLB 物体大小。如果物体看起来过小，可在 `01_align_scene.py` 中用 `--force-scale` 调整。

### 4.7 平滑策略

- **解析模式**: 对 MANO 输入位置 (wrist, finger1, finger2) 做 EMA (alpha=0.5)，保持 root pose 和手指关节一致性
- **优化器模式**: 对输出 root pose 做 EMA (位置 alpha=0.6, 朝向 alpha=0.6)
- **Warmup**: 前30帧用 smoothstep 从初始位置过渡到跟踪位置

> **alpha=0.5 的选择**: alpha=0.9 会导致 ~9.5 帧延迟 (30fps 下 ~0.32s), 夹爪明显滞后于手部运动 ("跟不上")。alpha=0.5 延迟 ~1.4 帧, 平衡了平滑与实时性。可用 `--smooth 0` 完全关闭平滑。

### 4.8 有效帧检查

渲染完成后检查有效帧比例:
- 有效帧 = `pred_valid=True` 且数据不含 NaN 的帧
- 有效帧 < 50% 时输出警告，渲染结果可能不可靠

### 4.9 灵巧手渲染实现 (`render_dexterous_only.py`)

**与夹爪模式 (`render_gripper_only.py`) 的核心区别**:

| 维度 | 夹爪模式 | 灵巧手模式 |
|---|---|---|
| URDF 来源 | `gripper_config.py` 模板生成 / GalaxeaManipSim R1 | `dex-retargeting/assets/robots/hands/<name>/` 内置 6 种 |
| 手指数 | 2 (1-DOF 平行夹爪) | 5 (allegro/inspire/...) 或 4 (leap) |
| 腕部位姿 | **解析计算** (中点-手腕连线 + aligned 策略) | **优化器自动求** (dummy free joint) |
| 手指关节 | **解析** `(指尖距离 - 基准) × scale / 2` | **优化器自动求** |
| 是否可选优化器 | 是 (`--optimizer`) | 否 (固定使用 PositionOptimizer) |
| 是否支持手臂 | 是 (`--mode gripper_arm`) | 否 (仅渲染手部) |

#### 文件结构 (~940 行)

```
render_dexterous_only.py
├── 常量与导入
│   ├── SUPPORTED_ROBOTS = ["allegro","inspire","shadow","ability","leap","svh"]
│   └── 从 common.py 复用: setup_scene, load_hawor_data, load_glb_transformed, ...
├── _robot_name_to_enum(name) → RobotName
├── _load_dexterous_robot(robot_name_str, hand_type, scene, logger)
│       ← 加载 URDF + 构建 PositionRetargeting
├── _compute_wrist_quat_sapien(hand_pose_frame)
│       ← MANO 腕部 axis-angle → SAPIEN quaternion
├── render_dexterous_only_video(...)   ← 单手渲染主函数
├── render_dual_dexterous_video(...)   ← 双手渲染主函数 (同场景)
├── _ensure_transform_params(...)      ← 自动运行 01_align_scene.py
└── main()                             ← CLI 入口
```

#### 核心函数 1: `_load_dexterous_robot()`

加载灵巧手 URDF + 构建 retargeting, 返回 5 元组。参考自 `dex-retargeting/example/position_retargeting/hand_robot_viewer.py`。

```python
def _load_dexterous_robot(robot_name_str, hand_type, scene, logger):
    # 1. 设置默认 URDF 目录为 dex-retargeting/assets/robots/hands
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))

    # 2. 加载 retargeting config (强制 add_dummy_free_joint=True)
    #    dummy free joint = 6DOF (3 translation + 3 rotation)
    #    让优化器可以自由求解腕部 6D 位姿
    config_path = get_default_config_path(robot_name, RetargetingType.position, hand_type)
    override = dict(add_dummy_free_joint=True)
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()    # 构建 SeqRetargeting (含 PositionOptimizer)

    # 3. URDF 加载: 优先 _glb 版本 (有更优 mesh), 不存在则回退
    urdf_path = Path(config.urdf_path)
    if "glb" not in urdf_path.stem:
        glb_path = urdf_path.with_stem(urdf_path.stem + "_glb")
        if glb_path.exists():
            urdf_path = glb_path
        # 否则用原始 URDF (如 inspire)

    # 4. 用 yourdfpy 加载 URDF, 同样 add_dummy_free_joints=True
    robot_urdf = urdf.URDF.load(str(urdf_path), add_dummy_free_joints=True,
                                build_scene_graph=False)
    # 写入临时文件, 让 SAPIEN URDFLoader 加载
    robot = loader.load(temp_path)

    # 5. 构建 retargeting qpos → sapien qpos 索引映射
    #    retargeting 输出的 qpos 顺序与 SAPIEN robot 的 qpos 顺序可能不同
    #    需要建立索引映射, 否则关节赋值会错乱
    sapien_joint_names = [j.name for j in robot.get_active_joints()]
    retarget2sapien = np.array(
        [retargeting.joint_names.index(n) for n in sapien_joint_names]
    ).astype(int)

    target_link_human_indices = retargeting.optimizer.target_link_human_indices
    return robot, retargeting, retarget2sapien, target_link_human_indices, config
```

#### 核心函数 2: `_compute_wrist_quat_sapien()`

MANO 手腕 axis-angle → SAPIEN 空间四元数。仅用于 warm start。

```python
def _compute_wrist_quat_sapien(hand_pose_frame):
    # hand_pose_frame[0:3] = MANO 紧凑轴角 (3D), 表示腕部全局旋转 (SLAM 空间)
    # 注意: hand_pose_frame 是 1D (HaWoR pred_hand_pose[g_idx]), 不是 2D
    R_mano = pr.matrix_from_compact_axis_angle(hand_pose_frame[0:3])
    # RXWORLD_TO_SAPIEN = R_AXIS @ R_x (3x3 旋转矩阵)
    # R_AXIS: SLAM (Y-up) → SAPIEN 相机系约定
    # R_x = diag(1, -1, -1): OpenGL → OpenCV (X 右, Y 下, Z 前)
    R_sapien = RXWORLD_TO_SAPIEN @ R_mano
    return pr.quaternion_from_matrix(R_sapien)
```

#### 核心函数 3: `render_dexterous_only_video()` — 单手渲染主流程

```python
def render_dexterous_only_video(hawor_dir, ras_dir, transform_params_path, output,
                                robot_name="allegro", hand_idx=1, ...):
    # ─── 阶段 A: 数据加载 ──────────────────────────────────
    # 1. 加载 HaWoR 数据 (含 NaN 填充, 来自 common.load_hawor_data)
    hawor_data = load_hawor_data(hawor_dir, hand_idx=hand_idx)
    # 2. 加载相机位姿 (R_c2w, t_c2w, 用于第一人称相机轨迹)
    R_c2w_all, t_c2w_all = load_hawor_c2w(hawor_dir)
    # 3. 创建 MANOLayer (用于 FK 计算 MANO 关节位置)
    mano_layer = MANOLayer(prefix, betas_mean)

    # ─── 阶段 B: SAPIEN 场景 + GLB 物体 ─────────────────────
    scene = setup_scene()                          # 禁用重力, 仅用于渲染
    if glb_path.exists() and has_transform:
        obj_actors = load_glb_transformed(...)     # 加载 RAS 重建的 3D 场景

    # ─── 阶段 C: 加载灵巧手 ────────────────────────────────
    robot, retargeting, retarget2sapien, target_link_human_indices, _ = \
        _load_dexterous_robot(robot_name, hand_type, scene, logger)

    # ─── 阶段 D: Warm start (关键!) ────────────────────────
    # PositionOptimizer 是局部优化器 (NLopt SLSQP), 必须用首帧 MANO
    # 腕部位姿初始化, 否则会从原点开始陷入局部最优
    for probe_idx in range(num_frames):
        if not hawor_data["pred_valid"][g_idx]: continue
        # MANO FK: 计算关节位置 → SAPIEN 坐标系
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)
        wrist_pos = joints_sapien[0, :3]                  # 关节 0 = 手腕
        wrist_quat = _compute_wrist_quat_sapien(hand_pose)
        retargeting.warm_start(
            wrist_pos, wrist_quat,
            hand_type=hand_type, is_mano_convention=False,
        )
        break

    # ─── 阶段 E: Warmup smoothstep 过渡 (30 帧) ───────────
    # 从原点 (init_root_pos=0) 平滑过渡到首帧有效位姿
    # 避免灵巧手从原点突然"瞬移"到目标位置
    for wi in range(WARMUP_FRAMES):
        t = (wi + 1) / WARMUP_FRAMES
        t = t * t * (3 - 2 * t)                          # smoothstep
        interp_pos = init_root_pos * (1 - t) + first_valid_pos * t
        robot.set_root_pose(sapien.Pose(interp_pos, interp_quat))

    # ─── 阶段 F: 主渲染循环 ───────────────────────────────
    for local_idx in trange(num_frames):
        # 1. 更新相机 (跟随 HaWoR 轨迹)
        cam_pos, cam_quat = hawor_cam_to_sapien_pose(R_c2w_all[g_idx], t_c2w_all[g_idx])
        camera.set_local_pose(...)

        # 2. 跳过无效帧 (NaN/invalid), 写入黑色帧占位
        if not hawor_data["pred_valid"][g_idx]: continue

        # 3. MANO FK: 计算 MANO 关节 3D 位置 (SLAM 空间)
        #    → 坐标变换 → SAPIEN 空间
        _, j = compute_mano_joints(mano_layer, rot, hand_pose, trans)
        joints_sapien = _render_to_sapien(j)

        # 4. 关键点标记 (在 MANO 目标关节处画绿球, 便于对比)
        kp_nodes = _render_keypoints(joints_sapien[:, :3], ...)

        # 5. ★★★ 核心一行: retargeting 求解 ★★★
        #    ref_value = [N, 3] 数组, N = target links 数
        #    (从 MANO 关节中按 target_link_human_indices 提取)
        ref_value = joints_sapien[target_link_human_indices, :3].astype(np.float32)
        #    retargeting.retarget(ref_value) 返回完整 qpos (含 dummy free joint + 手指)
        #    [retarget2sapien] 用索引映射取出 SAPIEN 顺序的 qpos
        qpos = retargeting.retarget(ref_value)[retarget2sapien]
        robot.set_qpos(qpos)

        # 6. 验证模式: 计算指尖位置误差
        if verify:
            for k, link_name in enumerate(target_link_names):
                robot_pos = robot_link_positions[link_name]
                mano_pos = joints_sapien[mano_idx, :3]
                err = np.linalg.norm(robot_pos - mano_pos) * 1000  # mm

        # 7. 渲染: scene.update_render() → camera.take_picture() → writer.write()
        scene.update_render()
        camera.take_picture()
        rgb = camera.get_picture("Color")[..., :3]
        writer.write(bgr)

    # ─── 阶段 G: 视频重编码 ────────────────────────────────
    # 用 ffmpeg 把 mp4v 重编码为 H.264 (兼容性更好), CRF 控制质量
    cmd = [ffmpeg_exe, "-y", "-i", tmp_path, "-c:v", "libx264",
           "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p", ...]
```

#### 核心函数 4: `render_dual_dexterous_video()` — 双手渲染

与单手函数几乎相同, 区别:

1. **同场景两个 robot**: 在同一 `scene` 中分别加载左右灵巧手, 存入 `hand_states` 列表
2. **共用 GLB**: 左右手共享同一个 GLB 场景物体
3. **各自 warm start**: 每只手独立用各自首帧 MANO 数据初始化
4. **逐帧更新**: 循环内遍历 `hand_states`, 每只手分别 retargeting + set_qpos
5. **关键点**: 左手 `clear_existing=True`, 右手 `clear_existing=False` (避免覆盖)
6. **状态显示**: 视频顶部显示 `L:Y/N R:Y/N`, 表示当前帧左右手是否有效

#### 关键代码片段说明

**1. 为什么需要 `add_dummy_free_joint=True`?**

不加 dummy free joint 时, URDF 根 link 固定在世界原点, 优化器只能求解手指关节, 不能让手腕移动 → 灵巧手会"卡在原点"。

加 dummy free joint 后, URDF 变成:
```
dummy_free_joint (6DOF, fixed→ Floating) → root_link → finger joints...
```
优化器可以同时优化 6DOF 腕部位姿 + N 个手指关节 = 6+N DOF, 让机器人指尖位置匹配 MANO 目标点。

**2. 为什么需要 `retarget2sapien` 映射?**

`retargeting.retarget(ref_value)` 返回的 qpos 顺序由 retargeting config 决定 (按 URDF 出现顺序), 而 SAPIEN `robot.set_qpos()` 期望的顺序由 `robot.get_active_joints()` 决定 (按场景图加载顺序)。两者顺序可能不同 (尤其是 dummy free joint 的位置), 必须通过名字建立索引映射:

```python
sapien_joint_names = [j.name for j in robot.get_active_joints()]
retarget2sapien = np.array(
    [retargeting.joint_names.index(n) for n in sapien_joint_names]
).astype(int)
# 用法: qpos_for_sapien = retargeting_output[retarget2sapien]
```

**3. 为什么需要 warm start?**

PositionOptimizer 用 NLopt SLSQP (局部优化器), 从初始 qpos 出发找局部最优。如果不初始化:
- 初始 qpos = 0 (关节角全为 0, 灵巧手处于"默认张开"姿态, 位置在原点)
- 第一帧 MANO 手腕可能在 (0.3, 0.2, 0.5) 等远离原点的位置
- 优化器收敛到局部最优 → 灵巧手会"扭曲"去够目标, 但腕部仍在原点附近

用 MANO 腕部位姿初始化后, 优化器从正确起点出发, 收敛到正确的全局解。

**4. `_glb` URDF 与原始 URDF 的区别?**

- 原始 URDF: mesh 引用 `.stl`/`.obj` 文件 (SAPIEN 加载需要逐个 mesh 处理)
- `_glb` URDF: mesh 引用 `.glb` 文件 (SAPIEN 一次性加载, 视觉更好)
- dex-retargeting 内置 5 种灵巧手有 `_glb` 版本, inspire 没有 → 自动回退到原始 URDF

---

## 5. 输出结构

```
hand_track/output/{hawor_name}/
├── transform_params.npz                              ← 坐标对齐参数 (自动生成)
├── alignment_report.txt                              ← 对齐报告
├── videos/
│   ├── hawor_r1_{left,right}_tracking.mp4            ← 机械臂跟踪视频 (单手时, render_auto.py)
│   ├── hawor_r1_dual_tracking.mp4                    ← 双臂合成视频 (双手时, 不保留单独左/右)
│   ├── hawor_r1_dual_gripper.mp4                     ← 双夹爪关键点合成视频 (双手时, 不保留单独左/右)
│   ├── hawor_r1_{left,right}_gripper_urdf.mp4        ← 夹爪URDF视频 (单手时, --mode gripper/both)
│   ├── hawor_r1_{left,right}_gripper_urdf_arm.mp4    ← 夹爪+手臂URDF视频 (单手时, --mode gripper_arm/both)
│   ├── hawor_r1_dual_gripper_urdf.mp4                ← 同场景双夹爪URDF视频 (双手时, 仅夹爪)
│   ├── hawor_r1_dual_gripper_urdf_arm.mp4            ← 同场景双夹爪URDF视频 (双手时, 夹爪+手臂末端)
│   ├── hawor_{robot}_{left,right}_urdf.mp4           ← 灵巧手URDF视频 (单手, render_dexterous_only.py)
│   └── hawor_{robot}_dual_urdf.mp4                   ← 同场景双灵巧手URDF视频 (双手)
└── tracking/
    └── hawor_r1_{left,right}_tracking.npy            ← qpos 轨迹数据
```

> **注**: `render_gripper_only.py` 默认 `--mode both`，单手时同时生成 `_gripper_urdf.mp4` 和 `_gripper_urdf_arm.mp4` 两个视频。
> `render_auto.py` 默认 `--mode gripper`，只生成 `_gripper_urdf.mp4`；可用 `--mode gripper_arm` 改为生成 `_gripper_urdf_arm.mp4`。
> `render_dexterous_only.py` 单手时生成 `hawor_{robot}_{left,right}_urdf.mp4`, 双手时生成 `hawor_{robot}_dual_urdf.mp4` (同场景双手)。
> 双手模式下, 单独的左/右手视频在合成后自动删除, 仅保留合成视频。

---

## 6. 渲染模式说明

### 对齐策略 (`--strategy`, 默认 `aligned`)

**默认推荐**。夹爪位姿通过对齐策略计算，无需优化器。`aligned` 策略实现用户要求的核心逻辑：先对齐夹爪两点，再用中点-手腕连线确定位姿，机械臂手腕放到位姿线上。

```bash
# 默认就是 aligned 策略，无需额外参数
python hand_track/render_gripper_only.py \
    --hawor-dir /path/to/hawor --ras-dir /path/to/ras

# 切换到旧策略 (Gram-Schmidt, 不带开合缩放)
python hand_track/render_gripper_only.py \
    --hawor-dir /path/to/hawor --ras-dir /path/to/ras \
    --strategy analytical

# 调整夹爪开合缩放 (默认 3.0, 1.0=精确映射)
python hand_track/render_gripper_only.py \
    --hawor-dir /path/to/hawor --ras-dir /path/to/ras \
    --open-scale 5.0
```

详见 [Section 4.4 夹爪位姿计算](#44-夹爪位姿计算)。

### 优化器模式 (--optimizer)
使用 dex_retargeting PositionOptimizer + gripper-only URDF (8 DOF = 6 dummy + 2 finger) 求解。
可能陷入局部最优，左手误差较大 (~7-10mm)。`--strategy` 参数对优化器模式同样生效 (影响 warm start 初始位姿)。

```bash
python hand_track/render_gripper_only.py \
    --hawor-dir /path/to/hawor --ras-dir /path/to/ras \
    --optimizer
```

### 机械臂模式 (render_auto.py)
加载完整 R1 URDF，使用 IK 求解关节角度，渲染完整机械臂跟踪视频。
同时额外生成夹爪关键点视频和夹爪URDF视频（`--mode` 控制是 gripper 还是 gripper_arm，默认 gripper）。

### 夹爪模式 (--mode gripper)
只加载 gripper_link + finger_link1/2 的 URDF (finger origin=37mm, 不缩放), 排除手臂底座不确定性，只看夹爪跟踪效果。

### 夹爪+手臂模式 (--mode gripper_arm)

加载夹爪+手臂 URDF (参考 GalaxeaManipSim 的 r1_v2_1_0_floating URDF, 不缩放 finger origin, 与 02_render_scene.py 一致)。`--arm-mode` 控制手臂类型:

- **`--arm-mode half` (默认)**: 半个手臂 (arm_link4/5/6 + gripper), 从完整浮动 URDF 提取 link4-6, 移除 link1/2/3, 将 joint4 的 parent 改为 arm_base_link, **并将 joint1/2/3 的 origin 累积到 joint4** (否则 arm_link4 位置错误, "中间空了一段")
- **`--arm-mode full`**: 完整手臂 (arm_link1-6 + gripper), 直接使用完整浮动 URDF

arm 关节每帧强制设为 0 (高刚度 drive + 每帧重置), 不会因物理模拟漂移。finger origin 保持 URDF 原始 37mm 几何 (不缩放), gripper_link 位姿通过 retargeting + FK 或解析计算获取。

### 夹爪/手臂/完整手臂跟随逻辑对比

三种渲染模式使用**不同的 root 对象和求解方式**，但夹爪位姿计算都基于同一套 MANO 关键点 (手腕、拇指尖、食指尖)：

| 模式 | 加载的 URDF | 求解对象 | 跟随逻辑 | 精度 |
|---|---|---|---|---|
| **gripper 关键点** (`render_gripper_video`) | 无 URDF，只渲染 3 个绿色球体 | 无 | 直接显示 MANO 关键点 | 基准，无额外误差 |
| **夹爪 URDF** (`--mode gripper`) | `gripper_link` + `finger_link1/2` (finger origin=37mm) | `gripper_link` | 解析计算 `gripper_link` 位姿，直接设置为 root | 指尖误差 ~4-5mm |
| **夹爪+手臂** (`--mode gripper_arm --arm-mode half`) | `arm_base_link` + `arm_link4-6` + 夹爪 | `arm_base_link` | 先用解析/优化器计算 `gripper_link` 位姿，再用 offset 反推 `arm_base_link` root 位姿；arm 关节锁定为 0；finger origin=37mm | 指尖误差 ~4-5mm |
| **夹爪+完整手臂** (`--mode gripper_arm --arm-mode full`) | `arm_base_link` + `arm_link1-6` + 夹爪 | `arm_base_link` | 同上, 但使用完整 arm_link1-6 运动链 | 同上 |
| **完整机械臂** (`render_auto.py`) | 完整 R1 浮动基座 URDF | 浮动基座 + 6 arm 关节 | 用 `dex_retargeting` PositionOptimizer 求 gripper 目标位姿，再用 RelaxedIK 求解完整 arm 关节；基座会小幅跟踪手腕 | 受机械臂可达性和 IK 精度影响，误差比纯夹爪略大 |

**关键结论**:
- `gripper` 和 `gripper_arm` 的夹爪位姿计算逻辑**完全相同**，只是 root 对象不同；渲染效果一致，都精确跟踪 MANO 手部
- `render_auto.py` 的完整 arm 因为要通过 6DOF arm 关节实现位姿，且基座浮动，精度通常低于前两者
- 如果只要验证手部跟踪精度，优先使用 `--mode gripper`

### 灵巧手模式 (`render_dexterous_only.py`)

渲染多指灵巧手 URDF (allegro/inspire/shadow/ability/leap/svh), 仅渲染手部, 无机械臂。

**关键模式** (参考 `dex-retargeting/example/position_retargeting/visualize_hand_object.py`):
- URDF 加载: `add_dummy_free_joint=True` — 在根 link 加 6DOF dummy free joint, 让手腕可自由移动
- 优先使用 `_glb.urdf` 版本 (有更优 mesh), 不存在时回退到原始 URDF (如 `inspire`)
- 优化器: `dex_retargeting` PositionOptimizer, 同时求 6D 腕部位姿 (dummy free joint) + N 个手指关节
- Warm start: 用 MANO 手腕位姿初始化优化器 (`R_sapien = RXWORLD_TO_SAPIEN @ R_mano`), 避免局部最优
- Retarget2sapien 映射: 将 retargeting 输出的 qpos 索引映射到 SAPIEN robot qpos (排除 dummy free joint 的反向赋值)

**与夹爪模式对比**:

| 对比项 | 夹爪模式 (`render_gripper_only.py`) | 灵巧手模式 (`render_dexterous_only.py`) |
|---|---|---|
| URDF 来源 | `gripper_config.py` 模板生成 / GalaxeaManipSim R1 浮动 URDF | `dex-retargeting/assets/robots/hands/` 内置 6 种灵巧手 |
| 手指数 | 2 (1-DOF 平行夹爪) | 5 (allegro/inspire/shadow/ability/svh) 或 4 (leap) |
| 求解方式 | 解析计算 (aligned 策略) 或 dex-retargeting (可选) | dex-retargeting PositionOptimizer (固定使用) |
| 腕部位姿 | 解析计算 (中点-手腕连线) | 优化器自动求解 (dummy free joint) |
| 手指关节 | 解析 (指尖距离 - 基准) × scale / 2 | 优化器自动求解 |
| 是否需要手臂 | 支持 `gripper_arm` 模式 (可选 half/full) | 不支持, 仅渲染手部 |

**`00_run_pipeline.py` 集成**:
- `--dexterous` 启用灵巧手管线 (默认运行步骤 1, 8: 对齐 + 灵巧手渲染)
- `--robot-name` 指定灵巧手 (默认 allegro)
- `--handedness` 仍生效 (auto/left/right/both → 自动映射到 `--hand-idx`)

### SAPIEN Viewer 模式 (--viewer)
在 SAPIEN Viewer 窗口中实时循环播放动画, 不保存视频文件。循环播放时完全重置 robot 状态, 关闭窗口退出。

---

## 7. 已测试的 HaWoR 数据

| 目录 | 手部 | 视频 |
|---|---|---|
| `7`             | 左手  | ✓ 机械臂 + 夹爪URDF + 夹爪+手臂URDF |
| `hoi4d`         | 双手  | ✓ 机械臂 + 夹爪URDF |
| `laptop`        | 右手  | ✓ 机械臂 |
| `7_vggt-omega`  | 左手  | ✓ 机械臂 |

---

## 8. 常见问题

### Q: 不提供 --ras-dir 会怎样?
不提供 `--ras-dir` 时，不加载 GLB 场景 (只有机械臂/夹爪)。
但 `render_auto.py` 和 `render_gripper_only.py` 要求 `--ras-dir` 为必填参数。

### Q: 需要手动运行 01_align_scene.py 吗?
不需要。脚本会自动检测并运行。

### Q: 渲染的视频只有几帧?
检查输出日志中的有效帧统计。如果很多帧被 `pred_valid=False` 或 NaN 过滤掉，可用 `--num-frames` 限制调试。有效帧 < 50% 时会输出警告。

### Q: 物理仿真能实现吗?
可以。项目在 `physics_pipeline/` 目录下已有两条物理仿真管线 (PyBullet + SAPIEN)，详见 [docs/questions.md](docs/questions.md)。

### Q: GLB 物体看起来太小怎么办?
这是 Umeyama 坐标对齐的缩放因子 `s_inv` 导致的。可在 `01_align_scene.py` 中用 `--force-scale` 调整，详见 [docs/questions.md Q10](docs/questions.md)。
