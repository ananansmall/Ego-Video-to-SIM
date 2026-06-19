# 物理仿真管线 (Physics Pipeline)

让 `02_render_scene.py` 的运动学渲染具有物理仿真属性：物体可被抓取/推动、机器人有重力补偿、桌面支撑物体不悬浮。

## 双引擎架构

| | SAPIEN (`04_physics_simulation.py`) | PyBullet (`pybullet_pipeline.py`) |
|---|---|---|
| 驱动方式 | PD 驱动 + 重力补偿 (`set_qf`/`set_drive_target`) 或运动学 (`set_qpos`) | `resetJointState` (运动学) |
| 物理引擎 | PhysX (SAPIEN 内置) | PyBullet |
| 物体交互 | 碰撞+摩擦抓取 | 碰撞+推动 |
| GPU需求 | **需要 GPU** (Vulkan 渲染) | CPU only |
| 优势 | 视觉渲染质量高，物理真实 | 无 GPU 依赖，稳定可重现 |
| 劣势 | 需要GPU，沙箱内无法运行 | 渲染质量一般，无抓取 |

---

## 各部分仿真的实现方式

### 1. GLB 真实物体的加载方式

**数据流**: `final_scene.glb` (RAS y-down) → HaWoR render world (y-up) → SAPIEN (z-up)

**变换链** (与 02_render_scene.py 完全一致):
```python
# 1. RAS GLB 顶点 → HaWoR render world (用 01_align_scene.py 的 Umeyama 对齐结果)
vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
# 2. HaWoR render world → SAPIEN (z-up)
vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
```

**物体分类** (自动判断 static/dynamic):
- **大型扁平几何体** (桌面/地板): `volume > 0.01 且 flatness < 0.3` 或 `max_extent > 0.8` → `mass=0` (static, 不受重力)
- **小物体**: `mass = volume * 1000` (dynamic, 受重力可被推动)

**碰撞体**:
- SAPIEN 04: `load_glb_with_physics()` 用 CoACD 分解凸包 (精确) 或 `--fast-collision` 用凸包 (快速)
- PyBullet: `createCollisionShape(GEOM_MESH)` 直接用 mesh 作为碰撞体

### 2. 桌面支撑的实现方式

**为什么需要桌面**: GLB 物体变换到 SAPIEN 坐标系后，最低点 Z 可能不在 0 处。如果没有支撑面，dynamic 物体会一直下落。

**桌面位置计算** (SAPIEN 04 和 PyBullet 一致):
```python
# 从 GLB 物体最低点确定桌面高度 (紧贴物体最低点下方 2mm)
min_z = min(all_verts_z)           # 所有 GLB 物体顶点的最低 Z
ground_height = min_z - 0.002      # 桌面顶部高度 (2mm 间隙让物体自然落稳)

# 桌面尺寸自适应 GLB 物体范围
table_half_x = max(0.3, extent_xy[0] / 2 + 0.1)  # 至少 0.3m, 外加 10cm 边距
table_half_y = max(0.3, extent_xy[1] / 2 + 0.1)
table_center_xy = glb_centroid_xy                  # 桌面中心 = GLB 物体质心 XY
```

**桌面属性**: `mass=0` (kinematic, 固定不动), 木色 `[0.55, 0.45, 0.35, 1.0]`, 摩擦系数 1.0

**关键**: 桌面本身 **没有物理仿真** (mass=0, kinematic), 它只是一个静态碰撞体, 用于支撑 dynamic 物体。物理仿真只作用于 dynamic 的 GLB 物体和机器人。

### 3. 机械臂的物理仿真方式

#### SAPIEN 04 (有物理仿真 + 重力补偿)

**控制策略**: PD 驱动 + 重力补偿
```python
# 每个物理子步:
qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)  # 重力补偿
robot.set_qf(qf)                                # 施加补偿力
for joint, target in zip(active_joints, target_qpos):
    joint.set_drive_target(target)               # PD 驱动目标
scene.step()                                     # PhysX 步进
```

- **有物理仿真**: 机械臂连杆有惯性, 受重力, 用 PD 控制器跟踪目标关节角
- **有重力补偿**: `compute_passive_force(gravity=True)` 计算抵消重力所需的力, 避免 arm 下垂
- **decimation=8**: 30Hz 控制频率, 240Hz 物理频率, 每帧步进 8 个子步

#### PyBullet (PD控制 + 重力补偿, 与 SAPIEN 04 一致)

**控制策略**: Computed Torque Control (PD + 重力补偿)
```python
# 每个物理子步:
current_qpos = get_qpos()           # 读取当前关节角
current_qvel = get_qvel()           # 读取当前关节速度
tau_gravity = calculateInverseDynamics(  # 重力补偿力矩 (含科氏力)
    robot_id, current_qpos, current_qvel, [0]*n  # 零加速度 → 只含重力+科氏力
)
for i, idx in enumerate(all_active_joint_indices):
    tau_pd = kp * (q_target[i] - current_qpos[i]) - kd * current_qvel[i]  # PD力矩
    tau = tau_pd + tau_gravity[i]                                         # 总力矩
    setJointMotorControl2(robot_id, idx, TORQUE_CONTROL, force=tau)       # 施加力矩
stepSimulation()                    # PyBullet 物理步进
```

- **有物理仿真**: 机械臂连杆有惯性, 受重力, 用 PD 控制器跟踪目标关节角
- **有重力补偿**: `calculateInverseDynamics(q, q_dot, 0)` 计算抵消重力+科氏力所需的力矩
- **PD增益** (与 SAPIEN 04 一致): arm kp=1000 kd=200, gripper kp=1000 kd=200
- **decimation=8**: 30Hz 控制频率, 240Hz 物理频率, 每帧步进 8 个子步
- **IK误差监控**: 每帧读取物理仿真后的实际位姿, 与 02 的 IK 目标对比, 显示 `IK err=...`
- **与 SAPIEN 04 的等价关系**:
  - SAPIEN: `compute_passive_force(gravity=True)` ↔ PyBullet: `calculateInverseDynamics(q, q_dot, 0)`
  - SAPIEN: `set_drive_target(target)` ↔ PyBullet: `tau_pd = kp*(target - q) - kd*q_dot`
  - SAPIEN: `set_qf(qf)` + `scene.step()` ↔ PyBullet: `setJointMotorControl2(TORQUE_CONTROL, force=tau)` + `stepSimulation()`

### 4. 单夹爪模式的实现方式

**单夹爪模式** (`--single-gripper`): 只加载夹爪 URDF (无机械臂), 直接用 MANO 手腕位姿驱动夹爪, 参考 `hand_track/render_gripper_only.py`。

**与完整机械臂模式的区别**:
| | 完整机械臂 | 单夹爪 |
|---|---|---|
| URDF | 完整 R1 右臂 (6 DOF arm + 2 DOF gripper) | 只有夹爪 (2 DOF finger) |
| 驱动方式 | IK (DexRetargeting + RelaxedIK) → qpos | MANO 手腕/指尖 → 解析位姿 |
| 输入 | `hand_object_robot_tracking.npy` (02 的 IK 解) | HaWoR `.npz` (MANO 关节) |
| 夹爪位姿 | 机械臂正运动学计算 | `_compute_analytical_gripper_pose` 解析计算 |

**解析夹爪位姿** (`_compute_analytical_gripper_pose`):
```python
# 输入: MANO 手腕位置, 拇指尖位置, 食指尖位置
# 输出: 夹爪 root 位姿 (pos, R) + 手指关节值 (joint1, joint2)

# Y轴 = 拇指尖→食指尖方向
y_axis = (mano_finger2 - mano_finger1) / finger_dist
# X轴 = 手腕→指尖中点方向 (Gram-Schmidt 正交化)
x_axis = (finger_mid - mano_wrist) / wrist_dist
x_axis = x_axis - dot(x_axis, y_axis) * y_axis
# Z轴 = X × Y
z_axis = cross(x_axis, y_axis)
root_R = column_stack([x_axis, y_axis, z_axis])

# 手指关节值 = (指尖距离 - 夹爪基座距离) / 2
joint1 = joint2 = clamp((finger_dist - 0.026906) / 2, 0, 0.05)
```

**SAPIEN 04 单夹爪**: `run_single_gripper_tracking()` 方法, 用 `robot.set_root_pose()` + `set_qpos()` + PD 驱动
**PyBullet 单夹爪**: `render_single_gripper_video()` 方法, 用 `resetBasePositionAndOrientation()` (root运动学) + `TORQUE_CONTROL` (手指PD+重力补偿)
- **root位姿**: 运动学控制 (直接设置, 像 SAPIEN 的 `set_root_pose`)
- **手指关节**: PD控制 + 重力补偿 (与完整机械臂模式一致)
- **PD误差监控**: 每帧显示 `err=...` (物理仿真 vs 目标)

### 5. 相机视角的实现方式

**`--view` 参数** (SAPIEN 04 和 PyBullet 都支持):

| 视角 | 说明 | 相机位置 |
|---|---|---|
| `fpv` (默认) | 第一人称, 跟随 HaWoR 相机轨迹 | 每帧用 `hawor_cam_to_sapien_pose(R_c2w, t_c2w)` 计算 |
| `topdown` | 俯视 | `scene_center + [0, 0, 1.2]`, 朝下 |
| `behind` | 后方 | `scene_center + [-0.4, -0.5, 0.3]`, look-at 场景中心 |
| `front` | 前方 | `scene_center + [0.5, 0.3, 0.3]`, look-at 场景中心 |

- `fpv` 模式: 相机每帧跟随 HaWoR 相机轨迹, 与原始视频视角一致
- 其他模式: 相机固定, 用 look-at 计算朝向

### 6. 帧数对齐 (SAPIEN 04)

**问题**: 默认 `--speed 0.5` 导致 `frame_repeat = round(1.0/0.5) = 2`, 视频帧数 = HaWoR帧数 × 2

**修复**: `--speed` 默认值改为 1.0, 使 `frame_repeat = 1`, 视频帧数 = HaWoR 帧数

```python
frame_repeat = max(1, round(1.0 / self.speed))  # speed=1.0 → frame_repeat=1
# 渲染循环: 每帧写入 frame_repeat 次
for _ in range(frame_repeat):
    writer.write(bgr)
```

---

## 仿真运行流程

### 数据流总览

```
HaWoR 手部重建 (.npz)
    │
    │  02_render_scene.py 的 run_robot_tracking
    │  (MANO → DexRetargeting → RelaxedIK → qpos)
    ▼
hand_object_robot_tracking.npy  (113, 8) = [arm_joint1..6, gripper1, gripper2]
    │
    │  04_physics_simulation.py / pybullet_pipeline.py
    │  复用相同的: 坐标变换 / 相机轨迹 / 底座位置 / GLB 加载
    ▼
物理仿真视频 (mp4)
```

### SAPIEN 04 的运行模式

#### 模式 A: 单趟物理渲染 (默认)
```
逐帧: PD控制器 → 物理仿真步进 → 渲染
```
- 每帧用 `set_drive_target` 设置目标关节角, `set_qf` 加重力补偿
- 物理引擎步进 8 个子步 (decimation=8, 30Hz控制 / 240Hz物理)

#### 模式 B: 两趟渲染 (`--two-pass`)
```
第一趟 (运动学): set_qpos 逐帧渲染 → 保存 IK 目标轨迹
第二趟 (物理):   PD驱动 + 重力补偿 → 接触检测 → 渲染
```

#### 模式 C: 交互式 Viewer (`--viewer`)
```
实时 SAPIEN Viewer 窗口，不保存视频
```

#### 模式 D: 单夹爪模式 (`--single-gripper`)
```
只加载夹爪 URDF → MANO 手腕位姿直接驱动 → 物理仿真 → 渲染
```

### PyBullet 管线运行流程

#### 完整机械臂模式 (默认)
```
1. 加载 02 的轨迹文件 hand_object_robot_tracking.npy (IK解作为PD控制目标)
2. 复用 02 的 hawor_cam_to_sapien_pose 计算相机位姿 (fpv视角, 与02一致)
3. 复用 02 的 compute_optimal_fixed_base 计算底座位置
4. 加载 GLB 场景 (与 02 相同的坐标变换) + 自适应桌面
5. 逐帧: PD控制 + 重力补偿 (calculateInverseDynamics) + PyBullet物理步进
6. 读取物理仿真实际位姿, 与IK目标对比 (IK err=...)
7. 渲染: computeViewMatrix + getCameraImage → 写入视频
```

#### 单夹爪模式 (`--single-gripper`)
```
1. 加载 HaWoR 数据 (MANO 关节)
2. 加载夹爪 URDF (无机械臂)
3. 加载 GLB 场景 + 自适应桌面
4. 逐帧: _compute_analytical_gripper_pose → resetBasePositionAndOrientation (root运动学)
   + TORQUE_CONTROL (手指PD+重力补偿) + PyBullet物理步进
5. 读取物理仿真实际位姿, 与目标对比 (err=...)
6. 渲染: computeViewMatrix + getCameraImage → 写入视频
```

---

## 调用方式

### SAPIEN 04 管线 (需要GPU，不能在沙箱内运行)

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination

# 单趟物理渲染 (默认, fpv视角)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/7_my_7mp4_result/alignment/transform_params.npz \
    --fast-collision

# 指定视角 (topdown/behind/front)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/7_my_7mp4_result/alignment/transform_params.npz \
    --view topdown --fast-collision

# 单夹爪模式 (无机械臂, MANO手腕直接驱动夹爪)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/7_my_7mp4_result/alignment/transform_params.npz \
    --single-gripper --fast-collision

# 两趟渲染 (运动学 + 物理)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ./output/7_my_7mp4_result/alignment/transform_params.npz \
    --two-pass --fast-collision

# 禁用桌面支撑 (仅物理地面)
python 04_physics_simulation.py ... --no-support-table

# 交互式 Viewer (需要显示器)
python 04_physics_simulation.py ... --viewer

# 通过 rerender.sh 入口
bash physics_pipeline/rerender.sh render        # 单趟渲染 (smooth=1)
bash physics_pipeline/rerender.sh smooth        # 两趟平滑 (smooth=2)
bash physics_pipeline/rerender.sh demo          # 交互式Viewer

# 通过 00_run_pipeline.py 一键运行 (含物理仿真)
python 00_run_pipeline.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --steps 1,2,3,4,5,6
```

### PyBullet 管线 (CPU only, 无GPU需求)

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/physics_pipeline

# 完整机械臂模式: 渲染视频 (使用默认轨迹: 02保存的hand_object_robot_tracking.npy)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video

# 指定视角 (topdown/behind/front)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video --view topdown

# 单夹爪模式 (无机械臂, MANO手腕直接驱动夹爪)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video --single-gripper

# 单夹爪 + 指定视角
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video --single-gripper --view behind

# 指定轨迹文件和输出路径
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --trajectory path/to/qpos.npy \
    --output path/to/output.mp4

# GUI模式 (需要显示器)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video --gui

# 仅基础测试
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --test
```

---

## 完整参数列表

### SAPIEN 04 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hawor-dir` | 必需 | HaWoR 重建目录 |
| `--ras-dir` | 必需 | RAS 重建目录 |
| `--transform-params` | `./output/alignment/transform_params.npz` | 01对齐结果 |
| `--mode` | `physics_tracking` | 仿真模式 |
| `--hand-idx` | -1 | 手索引 (0=左, 1=右, -1=自动) |
| `--start-frame` | 0 | 起始帧 |
| `--num-frames` | -1 | 帧数 (-1=全部) |
| `--fps` | 60 | 视频帧率 |
| `--width/--height` | 1920/1080 | 渲染分辨率 |
| `--crf` | 14 | 视频质量 (0=无损, 14=高质量) |
| `--viewer` | False | 交互式Viewer |
| `--fast-collision` | False | 用凸包代替CoACD (快速但粗糙) |
| `--hide-hand` | False | 不渲染MANO手部 |
| `--speed` | 1.0 | 播放速度倍率 (1.0=视频帧数=HaWoR帧数) |
| `--smooth` | 1 | 平滑模式 (0/1/2) |
| `--two-pass` | False | 两趟渲染 (运动学+物理) |
| `--no-support-table` | False | 禁用桌面 (仅物理地面) |
| `--view` | `fpv` | 相机视角: fpv/topdown/behind/front |
| `--single-gripper` | False | 单夹爪模式 (无机械臂) |

### PyBullet 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hawor-dir` | `/home/an/data/hawor/7` | HaWoR 重建目录 |
| `--ras-dir` | `/home/an/data/ras/my_7mp4_result` | RAS 重建目录 |
| `--transform-params` | `./output/7_my_7mp4_result/alignment/transform_params.npz` | 01对齐结果 |
| `--trajectory` | `./output/7_my_7mp4_result/tracking/hand_object_robot_tracking.npy` | qpos 轨迹文件 (单夹爪模式不需要) |
| `--output` | `output/pybullet_render.mp4` | 输出视频路径 |
| `--num-frames` | -1 | 渲染帧数 (-1=全部) |
| `--hand-idx` | -1 | 手索引 (0=左, 1=右, -1=自动) |
| `--width/--height` | 1280/720 | 渲染分辨率 |
| `--view` | `fpv` | 相机视角: fpv/topdown/behind/front |
| `--single-gripper` | False | 单夹爪模式 (无机械臂) |
| `--gui` | False | GUI模式 (需要显示器) |
| `--test` | False | 仅运行基础测试 |
| `--render-video` | False | 渲染视频 |

---

## VK_ICD_FILENAMES 环境变量

SAPIEN 需要 Vulkan 进行渲染。`VK_ICD_FILENAMES` 指定使用哪个 GPU 驱动：

- **NVIDIA GPU**: `/usr/share/vulkan/icd.d/nvidia_icd.json`
- **Intel 集显**: `/usr/share/vulkan/icd.d/intel_icd.x86_64.json`

**重要**: 
- 04 脚本会检测 `VK_ICD_FILENAMES` 是否已设置，**不覆盖**已有值
- 在 `trae-sandbox` (bwrap) 环境中无法访问 GPU，SAPIEN 渲染会失败
- 需要在有 GPU 访问权限的终端直接运行 04

---

## 文件结构

```
physics_pipeline/
├── README.md                    # 本文档
├── pybullet_pipeline.py         # PyBullet物理仿真管线 (CPU only)
├── rerender.sh                  # SAPIEN渲染入口脚本
├── run_physics_pipeline.py      # SAPIEN独立管线（对齐+仿真一键运行）
└── output/
    └── pybullet_render.mp4      # PyBullet 渲染输出

04_physics_simulation.py         # SAPIEN物理仿真 (在上级目录, 需要GPU)
```

## 04 与管线其他脚本的关系

```
00_run_pipeline.py  ← 一键入口 (steps=1,2,3,4,5,6 含物理仿真)
    │
    ├── 01_align_scene.py      → transform_params.npz
    ├── 02_render_scene.py     → hand_object_robot_tracking.npy (运动学qpos)
    ├── 03_track_robot.py      → 独立机器人跟踪 (无需RAS场景)
    ├── 04_physics_simulation.py  ← 物理仿真渲染
    └── 05_video_alignment.py  → 2D重投影验证
```

**04 的输入依赖**:
- `transform_params.npz` (来自 01): GLB→HaWoR 坐标变换参数
- HaWoR `.npz` (手部重建): 相机轨迹 + 手部数据 (用于底座位置计算)
- RAS `final_scene.glb`: 3D 场景物体

## 已知限制

1. **SAPIEN 04 无法在沙箱内测试**: `trae-sandbox` (bwrap) 环境无 GPU 访问权限，需在有 GPU 的终端运行
2. **PyBullet PD跟踪误差**: R1 URDF 连杆惯性较小, PD控制有一定跟踪误差 (IK err ~0.8 rad), 可通过调参 (PD_KP/PD_KD) 改善
3. **无摩擦力抓取**: PD控制可碰撞/推动物体, 但无摩擦力抓取 (SAPIEN 04 有抓取)
4. **单夹爪root运动学**: 单夹爪模式的root位姿是运动学控制 (直接设置), 手指关节有PD物理仿真
