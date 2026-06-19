# hand_track — 自动手部检测 + 机械臂/夹爪渲染

> **核心功能**: 从 HaWoR 手部数据 + RAS 3D 场景自动渲染机械臂/夹爪跟踪视频。
> 只需两个目录即可运行，坐标对齐参数自动生成。

---

## 1. 模块组成

| 文件 | 作用 |
|---|---|
| `render_auto.py`        | **主入口** — 自动检测手部, 渲染机械臂 + 夹爪 URDF 视频 |
| `render_gripper_only.py`| **夹爪专用** — 只渲染夹爪 URDF (不加载手臂), 支持夹爪+手臂末端模式 |
| `common.py`             | 共享函数 — 场景设置、GLB 加载、关键点渲染、坐标变换等 |
| `run_all_hawor.py`      | 批量入口 — 扫描 `--hawor-base` 下所有 HaWoR 目录 (旧管线, 调用 02_render_scene.py) |
| `__init__.py`           | 包初始化 |
| `docs/questions.md`     | 问答记录 |
| `output/`               | 默认输出目录 |

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

### 2.2 只渲染夹爪

```bash
# 默认同时渲染 gripper + gripper_arm 两种模式 (推荐)
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 仅夹爪 (不加载手臂, 排除底座不确定性)
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode gripper

# 夹爪 + 手臂末端 (arm_link4/5/6, 更生动)
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --mode gripper_arm
```

### 2.3 SAPIEN Viewer 实时循环播放

```bash
# 在 SAPIEN Viewer 窗口中实时循环播放 (不保存视频)
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --viewer
```

### 2.4 验证指尖误差

```bash
# 无头模式渲染 + 输出指尖位置/手腕位姿误差报告
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --verify --num-frames 30
```

### 2.5 调试 (只跑前 N 帧)

```bash
python hand_track/render_auto.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --num-frames 30 --view behind
```

---

## 3. CLI 参数

### render_auto.py

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hawor-dir`       | (必填) | HaWoR 数据目录 |
| `--ras-dir`         | (必填) | RAS 重建结果目录 (含 `final_scene.glb`) |
| `--output-dir`      | 自动   | 输出目录 (默认 `hand_track/output/{hawor_name}`) |
| `--mode`            | `gripper` | 夹爪URDF模式: `gripper`=仅夹爪, `gripper_arm`=夹爪+手臂末端 |
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
| `--mode`            | `both` | `gripper`=仅夹爪, `gripper_arm`=夹爪+手臂末端, `both`=两者都渲染 |
| `--smooth`          | `1`    | `0`=不平滑, `1`=EMA平滑 |
| `--viewer`          | off    | SAPIEN Viewer 实时循环播放 (不保存视频) |
| `--verify`          | off    | 计算并输出指尖位置/手腕位姿误差 |
| `--optimizer`       | off    | 使用优化器模式 (默认: 解析模式, 指尖误差≈0) |
| `--fps` / `--view` / `--width` / `--height` / `--crf` | 同上 | 同 render_auto.py |
| `--start-frame` / `--num-frames` | 同上 | 同 render_auto.py |

---

## 4. 自动行为

### 4.1 手部检测
自动检测左手/右手/双手，检测到几只手就渲染几只手：
- 单手: 渲染一个机械臂视频 + 一个夹爪URDF视频
- 双手: 渲染双臂合成视频 + 同场景双夹爪URDF视频

### 4.2 坐标对齐
`--transform-params` **不需要手动指定**。脚本内置 `_ensure_transform_params()`:
1. 检查输出目录下是否已有 `transform_params.npz`
2. 如不存在，自动运行 `01_align_scene.py` 生成
3. 生成后直接使用

### 4.3 夹爪朝向
夹爪位姿拆分处理：
- **位置**: 用 retargeting FK 给出的 gripper 位置
- **朝向**: 用 MANO 手腕朝向 (position retargeting 只保证位置对齐，朝向由手腕决定)

---

## 5. 输出结构

```
hand_track/output/{hawor_name}/
├── transform_params.npz                              ← 坐标对齐参数 (自动生成)
├── alignment_report.txt                              ← 对齐报告
├── videos/
│   ├── hawor_r1_{left,right}_tracking.mp4            ← 机械臂跟踪视频 (单手时)
│   ├── hawor_r1_dual_tracking.mp4                    ← 双臂合成视频 (双手时, 不保留单独左/右)
│   ├── hawor_r1_dual_gripper.mp4                     ← 双夹爪关键点合成视频 (双手时, 不保留单独左/右)
│   ├── hawor_r1_{left,right}_gripper_urdf.mp4        ← 夹爪URDF视频 (单手时)
│   ├── hawor_r1_dual_gripper_urdf.mp4                ← 同场景双夹爪URDF视频 (双手时, 仅夹爪)
│   └── hawor_r1_dual_gripper_urdf_arm.mp4            ← 同场景双夹爪URDF视频 (双手时, 夹爪+手臂末端)
└── tracking/
    └── hawor_r1_{left,right}_tracking.npy            ← qpos 轨迹数据
```

> **注**: 双手模式下, 默认同时生成 `gripper` 和 `gripper_arm` 两个视频 (`--mode both`)。
> 单独的左/右手视频在合成后自动删除, 仅保留合成视频。

---

## 6. 渲染模式说明

### 解析模式 (默认, analytical)
从 MANO 指尖向量直接解析计算夹爪 root 位姿和手指关节值, 绕过优化器局部最优问题。
- Y轴: finger1→finger2 方向, X轴: wrist→finger_mid 方向, Z轴: X×Y (Gram-Schmidt 正交化)
- 手指关节: 从指尖距离解析计算 (`joint = (finger_dist - base_dist) / 2`)
- 指尖误差 ≈ 0 (仅来自 EMA 平滑滞后, alpha=0.9 时 < 1.5mm)
- 平滑: 对 MANO 输入位置做 EMA (保持 root pose 和手指关节一致性)

### 优化器模式 (--optimizer)
使用 NLopt SLSQP 优化器求解夹爪关节角度, 可能有局部最优问题。
- 平滑: 对输出 root pose 做 EMA (位置 alpha=0.6, 朝向 alpha=0.6)

### 机械臂模式 (render_auto.py)
加载完整 R1 URDF，使用 IK 求解关节角度，渲染完整机械臂。

### 夹爪模式 (render_gripper_only.py --mode gripper)
只加载 gripper_link + finger_link1/2 的 URDF，排除手臂底座不确定性，只看夹爪跟踪效果。

### 夹爪+手臂末端模式 (render_gripper_only.py --mode gripper_arm)
加载 arm_link4/5/6 + gripper_link + finger_link1/2 的 URDF，比纯夹爪更生动，同时排除手臂底座不确定性。夹爪位姿和朝向均来自解析计算（或 retargeting FK），并补偿了 gripper_link 相对于 root 的 offset。

### SAPIEN Viewer 模式 (--viewer)
在 SAPIEN Viewer 窗口中实时循环播放动画, 不保存视频文件。动画播放完后自动重新开始, 关闭窗口退出。

---

## 7. 已测试的 HaWoR 数据

| 目录 | 手部 | 视频 |
|---|---|---|
| `7`             | 双手  | ✓ 机械臂 + 夹爪URDF + 夹爪+手臂URDF |
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
检查输出日志中的有效帧统计。如果很多帧被 `pred_valid=False` 或 NaN 过滤掉，可用 `--num-frames` 限制调试。

### Q: 物理仿真能实现吗?
可以。项目在 `physics_pipeline/` 目录下已有两条物理仿真管线 (PyBullet + SAPIEN)，详见 [docs/questions.md](docs/questions.md)。
