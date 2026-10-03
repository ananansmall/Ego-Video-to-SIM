# 物理仿真管线 (Physics Pipeline)

让 `02_render_scene.py` 的运动学渲染具有物理仿真属性：物体可被抓取/推动、机器人有重力补偿、桌面支撑物体不悬浮。

## 双引擎架构 (对齐 04_physics_simulation.py)

SAPIEN 04 和 PyBullet 现在**完全对齐** (同一套 IK + 坐标变换 + 基座高度):

| | SAPIEN (`04_physics_simulation.py`) | PyBullet (`pybullet_pipeline.py`) |
|---|---|---|
| 驱动方式 | PD 驱动 + 重力补偿 (`set_qf`/`set_drive_target`) | 运动学控制 (`resetJointState`, 对齐 GalaxeaManipSim 效果) |
| 物理引擎 | PhysX (SAPIEN 内置) | PyBullet |
| 物体交互 | 碰撞+摩擦抓取 | 碰撞+推动 |
| GPU需求 | **需要 GPU** (Vulkan 渲染) | CPU only |
| IK 来源 | 独立重算 (DexRetargeting + RelaxedIK) | 加载 04 的轨迹 (可选 02 轨迹 `--use-02-trajectory`) |
| 基座高度 | 0.70m (固定基座) | 0.70m (固定基座, 与 04 一致) |
| 坐标变换 | `FLIP_Z_FOR_PHYSICS=True` (Z 翻转) | `FLIP_Z_FOR_PHYSICS=True` (Z 翻转, 与 04 一致) |
| 夹爪摩擦 | static=1.0, dynamic=1.0, restitution=0.6 | 碰撞+推动 (运动学控制) |
| 优势 | 视觉渲染质量高，物理真实，支持抓取 | 无 GPU 依赖，可沙箱运行 |
| 劣势 | 需要GPU，沙箱内无法运行 | 渲染质量一般，无真实抓取 |

**对齐策略** (对齐 GalaxeaManipSim):
- PD 参数: stiffness=1000, damping=200 (臂和夹爪统一)
- 基座: fix_root_link=True (固定基座)
- 抓取: 纯摩擦力抓取 (无 weld/attach, 对齐 GalaxeaManipSim)

---

## 各部分仿真的实现方式

### 1. GLB 真实物体的加载方式

**数据流**: `final_scene.glb` (RAS y-down) → HaWoR render world (y-up) → SAPIEN (z-up, Z 翻转)

**变换链** (与 04_physics_simulation.py 一致, 含 Z 翻转):
```python
# 1. RAS GLB 顶点 → HaWoR render world (用 01_align_scene.py 的 Umeyama 对齐结果)
vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
# 2. HaWoR render world → SAPIEN (z-up)
vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T
# 3. Z 翻转 (FLIP_Z_FOR_PHYSICS=True, 适配物理重力)
vertices_sapien[:, 2] = -vertices_sapien[:, 2]
```

**物体分类** (自动判断 static/dynamic):
- **大型扁平几何体** (桌面/地板): `volume > 0.01 且 flatness < 0.3` 或 `max_extent > 0.8` → `mass=0` (static, 不受重力)
- **小物体**: `mass = volume * 1000` (dynamic, 受重力可被推动)

**碰撞体**:
- SAPIEN 04: `load_glb_with_physics()` 用 CoACD 分解凸包 (精确) 或 `--fast-collision` 用凸包 (快速)
- PyBullet: `createCollisionShape(GEOM_MESH)` 直接用 mesh 作为碰撞体

### 2. 桌面支撑的实现方式

**为什么需要桌面**: GLB 物体变换到 SAPIEN 坐标系后，最低点 Z 可能不在 0 处。如果没有支撑面，dynamic 物体会一直下落。

**桌面高度计算** (SAPIEN 04 和 PyBullet 一致):
```python
# 1. Z 高度分箱 (1mm精度), 找最大水平面 (桌面表面)
# 2. 从 dynamic (小) 物体的最低 Z 确定桌面高度
# 3. 从检测到的桌面表面提取平均顶点颜色
min_z = min(dynamic_verts_z)        # dynamic 物体顶点的最低 Z
ground_height = min_z - 0.002       # 桌面顶部高度 (2mm 间隙)
```

**桌面尺寸** (防止物体被推出桌面):
- 桌面半厚度: **0.015m** (总厚度 3cm)
- XY 边距: **0.15m**
- 最小半尺寸: 0.15m

**桌面颜色**: 从 GLB 场景中检测到的桌面表面提取平均顶点颜色 (RGB)，而非硬编码木色。

**桌面属性**: `mass=0` (kinematic, 固定不动), 摩擦系数 1.0

**关键**: 桌面本身 **没有物理仿真** (mass=0, kinematic), 它只是一个静态碰撞体, 用于支撑 dynamic 物体。物理仿真只作用于 dynamic 的 GLB 物体和机器人。

### 3. 机械臂的物理仿真方式

#### SAPIEN 04 (PD 驱动 + 重力补偿)

**控制策略**: PD 驱动 + 重力补偿 (与 GalaxeaManipSim 一致)
```python
# 每个物理子步:
qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)  # 重力补偿
robot.set_qf(qf)                                # 施加补偿力
for joint, target in zip(active_joints, target_qpos):
    joint.set_drive_target(target)               # PD 驱动目标
scene.step()                                     # PhysX 步进
```

- **PD 参数**: arm stiffness=1000, damping=200 (与 GalaxeaManipSim 一致)
- **有物理仿真**: 机械臂连杆有惯性, 受重力, 用 PD 控制器跟踪目标关节角
- **有重力补偿**: `compute_passive_force(gravity=True)` 计算抵消重力所需的力, 避免 arm 下垂
- **decimation=8**: 30Hz 控制频率, 240Hz 物理频率, 每帧步进 8 个子步

#### PyBullet (运动学控制, 对齐 GalaxeaManipSim 效果)

**控制策略**: 运动学控制 (`resetJointState`)

GalaxeaManipSim 使用 PD驱动 + 重力补偿 (Kp=1000, Kd=200), 机械臂非常刚硬, 近似运动学控制。
在 PyBullet 中, R1 URDF 连杆惯性极小, computed torque control 不稳定 (IK 误差可达 25°+)。
因此采用运动学控制, 达到与 GalaxeaManipSim 同样的效果: 机械臂精确跟踪目标, 仍能推动 GLB 物体。

```python
# 每帧:
for i, idx in enumerate(all_active_joint_indices):
    target = target_qpos[i] if i < len(target_qpos) else 0.0
    p.resetJointState(robot_id, idx, target, targetVelocity=0.0)  # 运动学控制

# 物理仿真步进 (让 GLB 物体受重力/碰撞影响)
for _ in range(DECIMATION):
    p.stepSimulation()
```

- **运动学控制**: `resetJointState` 直接设置目标关节角, 零跟踪误差
- **GLB 物体物理**: 机械臂虽为运动学控制, 但碰撞检测正常工作, 仍能推动 GLB 物体
- **decimation=8**: 30Hz 控制频率, 240Hz 物理频率, 每帧步进 8 个子步
- **与 GalaxeaManipSim 的等价关系**:
  - GalaxeaManipSim: PD驱动+重力补偿 → 机械臂极刚硬 → 近似运动学控制
  - PyBullet: resetJointState → 精确运动学控制 → 同样效果

### 4. 基座控制模式 (SAPIEN 04 + PyBullet)

**固定底座 (默认)**: SAPIEN 04 和 PyBullet 都默认使用固定底座, 底座高度 `COMFORTABLE_REACH=0.55m` (让机械臂更直更舒适), 不跟随手腕移动。

#### 模式 A: 固定基座 (默认, `--fixed-base`)
- 基座固定在手腕质心正上方 0.55m 处, 不跟随手腕移动
- `BASE_TRACKING_RANGE=0.0` (无 XY 跟踪)
- **优势**: 基座完全不动, 物理仿真最稳定, 机械臂更直更舒适
- **SAPIEN 04**: `--fixed-base` (默认开启), 可用 `--no-fixed-base` 禁用
- **PyBullet**: 始终使用固定底座

#### 模式 B: 浮动基座 (SAPIEN 04, `--no-fixed-base`)
- 基座在 XY 方向跟踪手腕位置 (范围由 `BASE_TRACKING_RANGE` 控制, 默认 0.0 = 不跟踪)
- 每帧根据 IK 目标重新计算基座位置
- **问题**: 如果手腕 XY 范围超过跟踪范围, 基座饱和到边界, IK 会跳变

#### 模式 C: 分段固定基座 (SAPIEN 04, `--base-cluster`)
- 将轨迹按帧数等分为 3 段, 每段使用固定基座
- 基座间过渡用 smoothstep 插值 (10 帧过渡)
- **优势**: 基座稳定, IK 不会因基座跳变而大幅变化
- **参数**: `BASE_CLUSTER_N=3`, `BASE_CLUSTER_TRANSITION_FRAMES=10`

### 5. 单夹爪模式的实现方式

**单夹爪模式** (`--single-gripper`): 只加载夹爪 URDF (无机械臂), 直接用 MANO 手腕位姿驱动夹爪, 参考 `hand_track/render_gripper_only.py`。

**与完整机械臂模式的区别**:
| | 完整机械臂 | 单夹爪 |
|---|---|---|
| URDF | 完整 R1 右臂 (6 DOF arm + 2 DOF gripper) | 只有夹爪 (1 DOF wrist + 2 DOF finger) |
| 驱动方式 | IK (DexRetargeting + RelaxedIK) → qpos | MANO 手腕/指尖 → 解析位姿 |
| 输入 | `hand_object_robot_tracking.npy` (02 的 IK 解) | HaWoR `.npz` (MANO 关节) |
| 夹爪位姿 | 机械臂正运动学计算 | `_compute_analytical_gripper_pose` 解析计算 |

**单夹爪 URDF 结构** (参考 GalaxeaManipSim):
```
wrist_link (基座, 运动学根)
  └── [wrist_joint: revolute, axis=X, limit=-3.14~3.14] → gripper_link
        ├── [finger_joint1: prismatic, axis=-Y, limit=0~0.05] → finger_link1
        └── [finger_joint2: prismatic, axis=+Y, limit=0~0.05] → finger_link2
```

- **wrist_joint**: revolute 关节, 允许夹爪绕手腕旋转 (对齐 GalaxeaManipSim 的 arm_joint6)
- **finger_joint1/2**: prismatic 关节, 限位 0~0.05m (5cm 行程), effort=100, velocity=0.25
- **关节顺序**: `[wrist_joint, finger_joint1, finger_joint2]` (3 DOF)
- **运动学控制**: wrist_joint=0 (完整旋转由 root 位姿设置), finger_joint1/2 由 MANO 指尖距离计算

**解析夹爪位姿** (`_compute_analytical_gripper_pose`):

方法: 加权 SVD (Procrustes) + 匹配指尖中点
```python
# 输入: MANO 手腕位置, 拇指尖位置, 食指尖位置
# 输出: 夹爪 root 位姿 (pos, R) + 手指关节值 (joint1, joint2)

# 1. 计算手指关节值
finger_dist = norm(mano_finger2 - mano_finger1)
joint1 = joint2 = clamp((finger_dist - 0.026906) / 2, 0, 0.05)

# 2. 加权 SVD 最近正交旋转
# X 轴 = 手腕→指尖中点方向 (指向方向)
# Y 轴 = 拇指尖→食指尖方向 (开合方向, 权重 W_Y=5.0 更高)
W = diag([1.0, W_Y])
A = [gripper_x, gripper_y] @ W   # gripper 坐标系中的方向
B = [pointing, opening] @ W      # MANO 方向向量
H = A @ B.T
U, S, Vt = SVD(H)
root_R = Vt.T @ diag([1, 1, sign(det(Vt.T @ U.T))]) @ U.T

# 3. 匹配指尖中点确定 gripper_link 位置
root_pos = finger_mid - root_R @ finger_mid_in_gripper
```

关键: MANO 的指向方向和开合方向通常不正交。给 Y 轴更高权重 (W_Y=5.0) 优先保证开合方向精确, 从而最小化指尖位置误差。

**SAPIEN 04 单夹爪**: `run_single_gripper_tracking()` 方法, 用 `robot.set_root_pose()` + `set_qpos()` + PD 驱动
**PyBullet 单夹爪**: `render_single_gripper_video()` 方法, 用 `resetBasePositionAndOrientation()` (root运动学) + `resetJointState()` (手指运动学控制)
- **root位姿**: 运动学控制 (直接设置, 像 SAPIEN 的 `set_root_pose`)
- **手指关节**: 运动学控制 (与完整机械臂模式一致, 对齐 GalaxeaManipSim 效果)

### 6. 相机视角的实现方式

**`--view` 参数** (SAPIEN 04 和 PyBullet 都支持):

| 视角 | 说明 | 相机位置 |
|---|---|---|
| `fpv` (默认) | 第一人称, 跟随 HaWoR 相机轨迹 | 每帧用 `hawor_cam_to_sapien_pose(R_c2w, t_c2w)` 计算 |
| `topdown` | 俯视 | `scene_center + [0, 0, 1.2]`, 朝下 |
| `behind` | 后方 | `scene_center + [-0.4, -0.5, 0.3]`, look-at 场景中心 |
| `front` | 前方 | `scene_center + [0.5, 0.3, 0.3]`, look-at 场景中心 |

- `fpv` 模式: 相机每帧跟随 HaWoR 相机轨迹, 与原始视频视角一致
- 其他模式: 相机固定, 用 look-at 计算朝向

**坐标变换差异**:
- SAPIEN 04: `hawor_cam_to_sapien_pose` 使用 `FLIP_Z_FOR_PHYSICS=True` 翻转 Z 坐标
- PyBullet: `hawor_cam_to_sapien_pose` 不使用 Z 翻转 (与 02_render_scene.py 一致)

### 7. 帧数对齐 (SAPIEN 04)

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
1. 加载 02 的轨迹文件 hand_object_robot_tracking.npy (IK解作为运动学控制目标)
2. 复用 02 的 hawor_cam_to_sapien_pose 计算相机位姿 (fpv视角, 与02一致, 无Z翻转)
3. 复用 02 的 compute_optimal_fixed_base 计算底座位置
4. 加载 GLB 场景 (与 02 相同的坐标变换, 无Z翻转) + 自适应桌面
5. 逐帧: resetJointState (运动学控制, 对齐GalaxeaManipSim效果) + PyBullet物理步进
6. 渲染: computeViewMatrix + getCameraImage → 写入视频
```

#### 单夹爪模式 (`--single-gripper`)
```
1. 加载 HaWoR 数据 (MANO 关节)
2. 加载夹爪 URDF (无机械臂)
3. 加载 GLB 场景 + 自适应桌面
4. 逐帧: _compute_analytical_gripper_pose → resetBasePositionAndOrientation (root运动学)
   + resetJointState (手指运动学控制) + PyBullet物理步进
5. 渲染: computeViewMatrix + getCameraImage → 写入视频
```

---

## 调用方式

### SAPIEN 04 管线 (需要GPU，不能在沙箱内运行)

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
conda activate dex
# 默认模式: 固定基座 (0.70m高) + fpv视角 + 自适应桌面 + Z翻转
# 输出: physics_pipeline/output/physics_sim_physics_tracking.mp4
# 轨迹: output/tracking/physics_sim_physics_tracking.npy (供 PyBullet 使用)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --fast-collision

# 浮动基座模式 (基座跟踪手腕, 可能有IK跳变)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --no-fixed-base --fast-collision

# 分段固定基座模式 (3段固定基座, 减少IK跳变)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --base-cluster --fast-collision

# 指定视角 (topdown/behind/front)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --view topdown --fast-collision

# 指定帧数 (测试用)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --num-frames 30 --fast-collision

# 单夹爪模式 (无机械臂, MANO指尖直接驱动夹爪)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --single-gripper --fast-collision

# 禁用桌面支撑 (仅物理地面)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --no-support-table --fast-collision

# 交互式 Viewer (需要显示器)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --viewer --fast-collision
```

### PyBullet 管线 (CPU only)

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/physics_pipeline
conda activate dex
# 完整机械臂模式: 加载 04 的轨迹 (基座 0.70m, 推荐)
# 前提: 先运行 04 生成轨迹 (physics_sim_physics_tracking.npy)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 强制使用 02 的轨迹 (基座 0.35m, 会自动调整基座高度)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --use-02-trajectory \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 指定视角 (topdown/behind/front)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --view topdown

# 指定帧数 (测试用)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --num-frames 30

# 单夹爪模式 (解析法, 默认, 无机械臂)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --single-gripper \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 单夹爪模式 + Dex Retargeting 优化器 (解析法 warm-start + Dex 微调, 指尖精度更高)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --single-gripper --use-dex-retarget \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# GUI模式 (需要显示器)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --gui

# 手动指定轨迹和变换参数 (覆盖自动推导)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params ../output/7_my_7mp4_result/alignment/transform_params.npz \
    --trajectory ../output/tracking/physics_sim_physics_tracking.npy
```

### PyBullet 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--render-video` | - | 渲染视频 |
| `--hawor-dir` | 必填 | HaWoR 重建输出目录 |
| `--ras-dir` | 必填 | RAS 场景重建输出目录 |
| `--transform-params` | 自动推导 | 01_align_scene.py 的 transform_params.npz |
| `--trajectory` | 自动推导 | 04 的轨迹 (优先) 或 02 的轨迹 |
| `--use-02-trajectory` | False | 强制使用 02 的轨迹 (基座 0.35m) |
| `--output` | 自动生成 | 输出视频路径 (含参数后缀) |
| `--num-frames` | -1 (全部) | 渲染帧数 |
| `--hand-idx` | -1 (自动) | 手部索引: 0=左手, 1=右手 |
| `--width` | 1280 | 渲染宽度 |
| `--height` | 720 | 渲染高度 |
| `--view` | fpv | 相机视角: fpv/topdown/behind/front |
| `--single-gripper` | False | 单夹爪模式 (无机械臂) |
| `--use-dex-retarget` | False | 单夹爪模式用 Dex Retargeting |
| `--test-gripper-tracking` | - | 测试单夹爪跟踪精度 |
| `--gui` | False | GUI 模式 (需要显示器) |

### 单夹爪跟踪精度测试

使用 `hand_track` 的 gripper-only URDF + Dex Retargeting 配置（8 DOF = 6 dummy free joints + 2 finger joints，3 个目标点），对比解析法与优化器的指尖/手腕跟踪精度。

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/physics_pipeline
conda activate dex

# 解析法测试 (默认)
python pybullet_pipeline.py --test-gripper-tracking \
    --hawor-dir /home/an/data/hawor/7 \
    --hand-idx 1

# Dex Retargeting 优化器测试 (hand_track gripper-only 配置)
python pybullet_pipeline.py --test-gripper-tracking \
    --hawor-dir /home/an/data/hawor/7 \
    --hand-idx 1 \
    --use-dex-retarget

# 指定测试帧数
python pybullet_pipeline.py --test-gripper-tracking \
    --hawor-dir /home/an/data/hawor/7 \
    --hand-idx 1 \
    --num-frames 30 \
    --use-dex-retarget
```

**注意**: 必须切换到 `dex` conda 环境（系统 python 的 `pinocchio` 包版本不正确，会导致 Dex Retargeting 初始化失败）。

**测试结果示例** (数据集 `7`, 右手 `hand-idx 1`, 113 帧):

| 方法 | 手腕位置误差 | 指尖位置误差 | 手指间距误差 |
|------|-------------|-------------|-------------|
| 解析法 | 107.91 mm | **0.44 mm** | 0.00 mm |
| Dex Retargeting | 0.48 mm | **0.28 mm** | — |

说明:
- 解析法匹配指尖中点 (37mm URDF), gripper_link 在指尖中点后方 37mm 处, **不在 MANO 手腕位置** (手腕位置误差大是正常的)。2 个指尖精确对齐, 手腕方向由 MANO 指尖中点→手腕连线确定 (在中轴线上)。
- Dex Retargeting 使用 3 个目标点: 2 指尖 (位置精确对齐) + 手腕 (方向约束, 在中轴线上即可), 指尖误差可进一步降低。

### SAPIEN 04 单夹爪模式 (Dex Retargeting)

SAPIEN 04 的单夹爪渲染通过 `hand_track/render_gripper_only.py` 实现，已内置 Dex Retargeting 支持：

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
conda activate dex

# SAPIEN 单夹爪 + 解析法 (默认)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# SAPIEN 单夹爪 + Dex Retargeting 优化器 (--optimizer)
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --optimizer

# SAPIEN 单夹爪 + 验证指尖误差
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --verify --num-frames 30
```

**SAPIEN vs PyBullet 单夹爪对比**:

| 特性 | SAPIEN 04 (`hand_track`) | PyBullet (`pybullet_pipeline`) |
|---|---|---|
| 渲染引擎 | PhysX (Vulkan GPU) | PyBullet (CPU) |
| 解析法 | Gram-Schmidt (独立手指缩放) | 同左 |
| Dex Retargeting | `--optimizer` 参数 | `--use-dex-retarget` 参数 |
| URDF | gripper-only (8 DOF) | gripper-only (8 DOF) |
| 输出 | mp4 视频 | mp4 视频 |
| GPU 需求 | 需要 (Vulkan) | 不需要 |

---

## 参数自动推导

`--transform-params` 和 `--trajectory` 参数会自动从 `hawor-dir` 和 `ras-dir` 推导：

- **transform_params.npz**: 先查找 `output/{ras_bn}/alignment/transform_params.npz`，找不到则自动调用 `01_align_scene.py` 的 `compute_and_save_transform_params()` 生成
- **trajectory (.npy)**: 从 `output/{ras_bn}/tracking/hand_object_robot_tracking.npy` 自动推导

也可以手动指定覆盖自动推导：

```bash
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --transform-params path/to/custom_transform_params.npz \
    --fast-collision
```

---

## 完整参数列表

### SAPIEN 04 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hawor-dir` | 必需 | HaWoR 重建目录 |
| `--ras-dir` | 必需 | RAS 重建目录 |
| `--transform-params` | 自动推导 | 01对齐结果 |
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
| `--speed` | 1.0 | 播放速度倍率 |
| `--smooth` | 1 | 平滑模式 (0/1/2) |
| `--two-pass` | False | 两趟渲染 (运动学+物理) |
| `--no-support-table` | False | 禁用桌面 (仅物理地面) |
| `--view` | `fpv` | 相机视角: fpv/topdown/behind/front |
| `--single-gripper` | False | 单夹爪模式 (无机械臂) |
| `--base-cluster` | False | 分段固定基座 (推荐) |
| `--fixed-base` | False | 固定基座 (基座不跟随手腕移动) |

### PyBullet 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hawor-dir` | 自动推导 | HaWoR 重建目录 |
| `--ras-dir` | 自动推导 | RAS 重建目录 |
| `--transform-params` | 自动推导 | 01对齐结果 |
| `--trajectory` | 自动推导 | qpos 轨迹文件 |
| `--output` | `output/pybullet_render.mp4` | 输出视频路径 |
| `--num-frames` | -1 | 渲染帧数 |
| `--hand-idx` | -1 | 手索引 |
| `--width/--height` | 1280/720 | 渲染分辨率 |
| `--view` | `fpv` | 相机视角 |
| `--single-gripper` | False | 单夹爪模式 |
| `--gui` | False | GUI模式 |
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

## 04 与管线其他脚本的关系

```
00_run_pipeline.py  ← 一键入口
    │
    ├── 01_align_scene.py      → transform_params.npz
    ├── 02_render_scene.py     → hand_object_robot_tracking.npy (运动学qpos)
    ├── 03_track_robot.py      → 独立机器人跟踪
    ├── 04_physics_simulation.py  ← 物理仿真渲染
    └── 05_video_alignment.py  → 2D重投影验证
```

**04 的输入依赖**:
- `transform_params.npz` (来自 01): GLB→HaWoR 坐标变换参数
- HaWoR `.npz` (手部重建): 相机轨迹 + 手部数据
- RAS `final_scene.glb`: 3D 场景物体

## 已知限制

1. **SAPIEN 04 无法在沙箱内测试**: `trae-sandbox` (bwrap) 环境无 GPU 访问权限
2. **PD 跟踪误差 (SAPIEN 04)**: PD 控制有跟踪误差，qpos_set vs qpos_after 差异可达 30-90°，端部误差 50-185mm。`--base-cluster` 模式可显著降低误差
3. **坐标变换差异**: SAPIEN 04 使用 `FLIP_Z_FOR_PHYSICS=True` 翻转 Z 坐标, 而 PyBullet 和 02_render_scene.py 不使用 Z 翻转。两者渲染结果在 Z 方向上可能有差异
4. **相机 Z 翻转修复**: SAPIEN 04 的 `hawor_cam_to_sapien_pose` 已修复 Z 翻转导致的相机反向问题 (只翻转 forward/up 的 Z 分量, 重新计算 left 保证 det=+1)
5. **单夹爪手腕不对齐**: 解析法匹配指尖中点 (37mm URDF), gripper_link 在指尖中点后方 37mm 处, **不在 MANO 手腕位置**。这是设计选择: 优先保证 2 个指尖精确对齐, 手腕位置由几何关系自然决定。Dex Retargeting 也只目标 2 个指尖 (不目标手腕)。
