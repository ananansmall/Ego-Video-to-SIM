# 04_physics_simulation.py — 物理仿真驱动详解

> 文件位置: `/home/an/robot_world_ws/src/dex-retargeting/example/combination/04_physics_simulation.py`
> 最后更新: 2026-06-26

## 一、概述

`04_physics_simulation.py` 是手部→机器人映射管线的**物理仿真**环节。它在 SAPIEN 物理引擎中模拟 R1 机器人的抓取操作，考虑碰撞、摩擦和重力，输出真实的物理交互视频和轨迹文件。

与 `02_render_scene.py`（运动学渲染）不同，04 使用 **PD 控制器驱动关节**，而非直接设置关节角，因此能产生真实的抓取、推动、掉落等物理交互效果。

**核心定位**:
```
02_render_scene.py (运动学) → hand_object_robot_tracking.npy (IK 解)
                                   ↓
04_physics_simulation.py (物理) → physics_sim_physics_tracking.mp4 + .npy (物理轨迹)
                                   ↓
pybullet_pipeline.py (CPU 物理验证) → pybullet.mp4
```

---

## 二、运行方式

### 基本命令

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
conda activate dex

VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --fast-collision
```

> **必需环境**: GPU + Vulkan。在无 GPU 的沙箱中会报 `RuntimeError: failed to find a rendering device`。

### 输出

| 文件 | 路径 | 说明 |
|------|------|------|
| 物理仿真视频 | `physics_pipeline/output/physics_sim_physics_tracking.mp4` | SAPIEN 渲染视频 |
| 物理轨迹 | `output/tracking/physics_sim_physics_tracking.npy` | 供 PyBullet 使用 |

### 三种运行模式

| 模式 | 触发参数 | 说明 |
|------|----------|------|
| `run_physics_tracking` | 默认 | 完整机械臂 + DexRetargeting + RelaxedIK + PD 驱动 |
| `run_single_gripper_tracking` | `--single-gripper` | 仅夹爪（无机械臂），MANO 手腕位姿解析法直接驱动夹爪 |
| `run_bimanual_tracking` | `--hand-idx both` | 双手同时驱动两个机械臂（运动学模式） |

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--view` | `fpv` | 相机视角: `fpv`(第一人称跟随) / `topdown`(俯视) / `behind` / `front` |
| `--hand-idx` | `-1` | `0`=左手, `1`=右手, `both`=双手, `-1`=自动检测 |
| `--single-gripper` | False | 单夹爪模式（无机械臂） |
| `--base-cluster` | False | 分段固定基座（N 段，减少 IK 跳变） |
| `--fast-collision` | False | 用凸包代替 CoACD 凸分解（快但粗糙） |
| `--num-frames` | `-1` | 渲染帧数（-1=全部） |
| `--smooth` | `1` | `0`=无, `1`=在线EMA, `2`=后处理双向滤波 |

---

## 三、架构总览

```
04_physics_simulation.py
│
├── 模块级常量与导入
│   ├── GalaxeaManipSim 路径 (URDF/settings/meshes)
│   ├── 坐标变换矩阵 (RXWORLD_TO_SAPIEN 等)
│   ├── PD 驱动参数 (JOINT_STIFFNESS=1000, JOINT_DAMPING=200)
│   ├── 物理参数 (PHYSICS_TIMESTEP=1/240, DECIMATION=8)
│   ├── 夹爪几何 (_get_finger_geom: 左右镜像)
│   └── 舒适工作空间 (COMFORTABLE_REACH=0.70, COMFORT_TARGET_IN_BASE)
│
├── 工具函数
│   ├── _generate_gripper_only_urdf(prefix)     生成单夹爪 URDF
│   ├── _compute_analytical_gripper_pose(...)   加权 SVD 计算夹爪位姿
│   ├── setup_physics_scene(ground_height)      创建物理场景 + 地面
│   ├── load_glb_with_physics(...)              GLB → 动态物理物体
│   ├── _compute_optimal_fixed_base(...)        手腕质心 → 固定基座
│   └── _compute_fixed_base_clusters(...)       分段固定基座
│
├── 平滑器类
│   ├── EmaTargetSmoother                      在线 EMA 平滑
│   ├── TrajectorySmoother                     后处理双向滤波
│   └── OnlineTrajectorySmoother               速度/加速度/jerk 限幅
│
├── PhysicsSimulator 类 (核心)
│   ├── __init__                               解析参数, 加载数据
│   ├── _physics_step(robot, ...)              PD 驱动 + decimation + 重力补偿
│   ├── _kinematic_step(robot, ...)            set_qpos 运动学步 (两趟渲染用)
│   ├── _fetch_contacts(...)                   夹爪-物体接触力检测
│   ├── _get_gripper_pose_from_retargeting(...) FK 提取夹爪位姿
│   ├── run_single_gripper_tracking(...)       单夹爪模式
│   ├── run_physics_tracking(...)              完整机械臂模式 (默认)
│   └── run_bimanual_tracking(...)             双手模式
│
└── main()                                     命令行入口
```

---

## 四、运行流程详解

### 4.1 完整机械臂模式 (`run_physics_tracking`) — 8 个步骤

```
[1/8] 加载数据
  │  load_hawor_data()      → 手部 MANO 参数 (pred_trans/rot/hand_pose/betas)
  │  load_hawor_c2w()       → 相机轨迹 (R_c2w, t_c2w) 用于 FPV
  │  MANOLayer(mano_side)   → 左右手 MANO 模型
  ▼
[2/8] 创建物理场景 + 加载 GLB
  │  setup_physics_scene()  → SAPIEN Scene + 物理地面 (timestep=1/240s)
  │  load_glb_with_physics()→ GLB 物体 → 动态 actor (带 CoACD 碰撞体)
  │                           大型扁平几何体 → kinematic (桌面/地板)
  │                           小物体 → dynamic (可推动/抓取)
  ▼
[3/8] 初始化 R1 单臂机器人
  │  loader.load(FLOATING_RIGHT_URDF 或 FLOATING_LEFT_URDF)
  │  set_drive_property(stiffness=1000, damping=200)  ← 与 GalaxeaManipSim 一致
  │  set_drive_target(initial_qpos)
  │  _compute_optimal_fixed_base() → 手腕质心 + [0,0,0.70] (固定基座)
  ▼
[4/8] 初始化 Dex Retargeting
  │  RetargetingConfig.load_from_config(config_path)
  │  SeqRetargeting(...)    → 手部关节 → 夹爪 qpos
  │  retargeting.warm_start(wrist_pos, wrist_quat)  ← 用手腕朝向初始化
  ▼
[5/8] 初始化 RelaxedIK + 预计算
  │  from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver
  │  ik_solver = RelaxedIKSolver(left_setting, right_setting, tolerances)
  │  relaxed_ik = ik_solver.relaxed_ik_{prefix}
  │  预计算所有帧的 IK 解序列 (含 30 帧 warmup smoothstep 过渡)
  ▼
[6/8] 设置相机
  │  fpv:     hawor_cam_to_sapien_pose(R_c2w, t_c2w) → 相机跟随 HaWoR SLAM 轨迹
  │  topdown: 固定俯视 (高度 0.6m, 看整体场景)
  │  behind/front: 固定第三人称视角
  ▼
[7/8] Warmup + 预计算
  │  前 30 帧 smoothstep 插值: 初始关节角 → 第一帧 IK 解
  │  可选 EMA/LPF 平滑
  ▼
[8/8] 物理仿真渲染 (逐帧)
  │  for frame in range(num_frames):
  │    a. DexRetargeting: MANO 关节[0,4,8] → 夹爪 qpos
  │    b. FK 提取夹爪位姿 → base_link 坐标系
  │    c. RelaxedIK.solve() × 20 次 → 臂关节角 (6 DOF)
  │    d. _physics_step():
  │       set_drive_target(arm_target)       ← PD 目标
  │       set_drive_target(gripper_target)
  │       for _ in range(DECIMATION=8):      ← 物理子步
  │         qf = compute_passive_force(gravity, coriolis)
  │         robot.set_qf(qf)                 ← 重力补偿
  │         scene.step()                     ← PhysX 求解
  │    e. _fetch_contacts() → 检测夹爪-物体接触
  │    f. camera.take_picture() → 写入视频帧
  ▼
输出: physics_sim_physics_tracking.mp4 + .npy
```

### 4.2 单夹爪模式 (`run_single_gripper_tracking`)

不使用 DexRetargeting 和 RelaxedIK，而是用**解析法**直接从 MANO 3 个特征点计算夹爪位姿：

```
MANO 特征点: 手腕[0], 拇指尖[4], 食指尖[8]
    │
    ├── 1. 指尖距离 → 手指关节值 (joint1, joint2)
    │      finger_dist = |finger2 - finger1|
    │      joint = clamp((finger_dist - 0.0269) / 2, 0, 0.05)
    │
    ├── 2. 加权 SVD (Procrustes) → 夹爪根旋转 root_R
    │      指向方向: pointing = (finger1+finger2)/2 - wrist  → gripper X 轴
    │      开合方向: opening = y_sign * (finger2 - finger1)  → gripper Y 轴
    │      Y 轴权重 W_Y=5.0 (优先保证开合方向精确)
    │      H = A @ B.T → SVD → root_R (最近正交矩阵)
    │
    └── 3. 指尖中点匹配 → 夹爪根位置 root_pos
           finger_mid_in_gripper = (finger1_in_gripper + finger2_in_gripper) / 2
           root_pos = finger_mid - root_R @ finger_mid_in_gripper

输出: root_pos, root_R, joint1, joint2 → set_drive_target → PD 驱动
```

**左右手处理**:
- `mano_side = "left" if hand_idx[0] == 0 else "right"`
- `prefix = mano_side` — 左手用左夹爪 URDF（镜像几何），右手用右夹爪
- `_get_finger_geom(prefix)` 返回 prefix 依赖的 finger origin/axis
- `y_sign = 1.0 if prefix == "right" else -1.0` — 翻转 MANO 开合方向

---

## 五、与 02_render_scene.py 的具体区别

### 5.1 核心区别对比表

| 对比项 | 02_render_scene.py | 04_physics_simulation.py |
|--------|--------------------|---------------------------|
| **驱动方式** | 运动学 (`set_qpos` 直接设置关节角) | 动力学 (PD 驱动 `set_drive_target`) |
| **物理引擎** | 仅渲染，不步进物理 | PhysX 求解 (重力 + 碰撞 + 摩擦) |
| **碰撞** | 无（物体穿透） | 有（CoACD 凸分解碰撞体） |
| **抓取** | 无（夹爪视觉闭合） | 有（摩擦力抓取，friction=1.0） |
| **重力** | 无（物体悬浮） | 有（`[0,0,-9.81]`，物体可掉落） |
| **场景地面** | 无 | 有（`add_ground`，防无限下落） |
| **GLB 物体** | 静态可视 | 动态 actor（可推动/抓取） |
| **控制步** | 单次 `scene.step()` | `DECIMATION=8` 次物理子步 |
| **基座高度** | `COMFORTABLE_REACH=0.35` | `COMFORTABLE_REACH=0.70` |
| **单夹爪模式** | 无 | 有（`--single-gripper`，解析法） |
| **运行环境** | 仅需 CPU | 需 GPU + Vulkan |
| **映射链** | DexRetargeting + RelaxedIK | 同 02（完整模式）或解析法（单夹爪） |

### 5.2 驱动方式差异（最关键）

**02 (运动学)**:
```python
# 直接设置关节角，无物理交互
qpos[arm_indices] = arm_target
qpos[gripper_indices] = gripper_target
robot.set_qpos(qpos)
scene.step()  # 仅更新渲染，不求解物理
```

**04 (动力学)**:
```python
# PD 控制器跟踪目标，PhysX 求解力+接触
for idx, target in zip(arm_indices, arm_target):
    active_joints[idx].set_drive_target(target)
active_joints[gripper_idx1].set_drive_target(gripper_target1)
active_joints[gripper_idx2].set_drive_target(gripper_target2)

for _ in range(DECIMATION):  # 8 次物理子步
    qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
    robot.set_qf(qf)         # 重力补偿
    scene.step()             # PhysX 求解 PD 力 + 补偿力 + 接触力
```

**为什么 04 不用 `set_qpos`？**
`set_qpos` + `set_drive_target` 双重控制会导致 PhysX 求解器中 PD 力与直接位置约束冲突，产生"拉回→惯性冲出→再拉回"震荡。GalaxeaManipSim 从不在 `step()` 中调用 `set_qpos`，纯 PD 驱动保证平稳。04 遵循这一约定。

### 5.3 场景设置差异

**02**: `setup_scene()` — 仅灯光 + 环境贴图，无物理地面
**04**: `setup_physics_scene()` — 灯光 + 环境贴图 + 物理时间步 + 不可见地面

```python
# 04 独有
scene.set_timestep(PHYSICS_TIMESTEP)  # 1/240s
scene.add_ground(ground_height, render_half_size=[0, 0])  # 不可见物理地面
```

### 5.4 GLB 物体差异

**02**: `load_glb_transformed()` — 仅视觉网格，无碰撞体，无物理属性
**04**: `load_glb_with_physics()` — 视觉网格 + 碰撞体 + 物理材质 + 密度

```python
# 04 独有
actor = scene.create_actor(name)
actor.add_visual_from_file(mesh_path)
if fast_collision:
    actor.add_convex_collision_from_file(mesh_path)  # 凸包 (快)
else:
    actor.add_nonconvex_collision_from_file(mesh_path)  # CoACD (精确)
actor.set_physic_material(friction=0.5, restitution=0.3)
actor.set_density(OBJECT_DENSITY)  # 1000 kg/m³
```

### 5.5 基座高度差异

| 参数 | 02 | 04 | 原因 |
|------|----|----|------|
| `COMFORTABLE_REACH` | 0.35m | 0.70m | 04 物理仿真中机械臂需垂直抓取，避免靠近桌面碰撞；02 仅渲染无需考虑物理碰撞 |

---

## 六、为什么实现物理仿真

### 6.1 02 运动学渲染的局限

02 通过 `set_qpos` 直接设置关节角，虽然能生成视觉上合理的机器人动作，但存在根本缺陷：

1. **无碰撞响应**: 夹爪"穿透"物体，无法验证抓取是否可行
2. **无重力**: 物体悬浮，无法模拟松手后掉落
3. **无摩擦抓取**: 夹爪闭合只是视觉表现，实际无接触力
4. **无法验证轨迹可行性**: IK 解在物理上可能不可达（关节限位、奇异点、碰撞）

### 6.2 04 物理仿真的价值

04 通过 PD 驱动 + PhysX 求解，解决上述问题：

1. **真实碰撞**: CoACD 凸分解生成精确碰撞体，夹爪无法穿透物体
2. **重力模拟**: 物体受 `[0,0,-9.81]` 重力，松手会掉落到桌面/地面
3. **摩擦抓取**: 夹爪手指 friction=1.0，通过接触摩擦力抓取物体
4. **轨迹验证**: 物理仿真中若夹爪无法到达目标位置，说明轨迹在物理上不可行

### 6.3 decimation 的作用

PD 控制器需要**多次物理步**才能收敛到目标位置。04 每个控制帧执行 `DECIMATION=8` 次物理子步：

```
控制频率: 30 Hz (每 1/30s 一个控制步)
物理时间步: 1/240s
decimation = (1/30) / (1/240) = 8

每个控制步:
  set_drive_target → 8 次 (compute_passive_force + set_qf + scene.step)
```

这样 PD 控制器有足够时间收敛，同时物理引擎以 240Hz 高频率求解接触力，保证稳定性。

---

## 七、如何调用 GalaxeaManipSim

### 7.1 路径与导入

04 在文件头部（第 77-84 行）定义了 GalaxeaManipSim 的资源路径：

```python
GALAXEA_SIM_PATH = Path("/home/an/robot_world_ws/src/GalaxeaManipSim")
sys.path.insert(0, str(GALAXEA_SIM_PATH))

R1_MESH_DIR = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "meshes"
FLOATING_RIGHT_URDF = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_right.urdf"
FLOATING_LEFT_URDF  = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "urdfs" / "r1_v2_1_0_floating_left.urdf"
R1_RIGHT_SETTINGS = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_right.yaml"
R1_LEFT_SETTINGS  = GALAXEA_SIM_PATH / "galaxea_sim" / "assets" / "r1" / "configs" / "settings_left.yaml"
```

### 7.2 调用的三类资源

#### (1) URDF 机器人模型

```python
# 完整机械臂模式 (run_physics_tracking)
urdf_path = FLOATING_RIGHT_URDF if prefix == "right" else FLOATING_LEFT_URDF
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(str(urdf_path))
```

URDF 来自 GalaxeaManipSim 的 `galaxea_sim/assets/r1/configs/urdfs/`，包含完整的 R1 机器人定义（6 DOF 臂 + 2 prismatic 夹爪）。

**左右手 URDF 的镜像关系**:
- 右夹爪 `finger_joint1`: origin `xyz="0.03689 -0.013453 -0.00012053"`, axis `0 -1 0` (finger1 在 -Y)
- 左夹爪 `finger_joint1`: origin `xyz="0.03689 0.013453 0.00012067"`, axis `0 1 0` (finger1 在 +Y)

单夹爪模式（`_generate_gripper_only_urdf`）从这些 URDF 提取 finger 几何，通过 `_get_finger_geom(prefix)` 返回 prefix 依赖的常量，保证左右夹爪镜像正确。

#### (2) RelaxedIK 求解器

```python
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

ik_solver = RelaxedIKSolver(
    left_setting_file_path=str(R1_LEFT_SETTINGS),
    right_setting_file_path=str(R1_RIGHT_SETTINGS),
    tolerances=IK_TOLERANCES,
)
relaxed_ik = getattr(ik_solver, f"relaxed_ik_{prefix}")   # relaxed_ik_left 或 relaxed_ik_right
ik_solve = getattr(ik_solver, f"solve_position_{prefix}")

relaxed_ik.reset(starting)  # 用初始关节角重置
# 每帧调用 20 次 IK 迭代:
for _ in range(IK_SOLVE_PER_FRAME):
    ik_solve(ik_target_pos, ee_quat)
```

`RelaxedIKSolver` 来自 GalaxeaManipSim 的 `galaxea_sim/controllers/utils/relaxed_ik_solver.py`，是一个松驰逆运动学求解器，能在关节限位约束下平滑求解臂关节角。settings YAML 文件定义了关节限位、阻尼等 IK 参数。

#### (3) Mesh 资源

单夹爪 URDF 模板引用 GalaxeaManipSim 的 STL mesh：

```xml
<mesh filename="{mesh_dir}/{prefix}_gripper_link.STL"/>
<mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/>
```

`R1_MESH_DIR` 指向 `galaxea_sim/assets/r1/meshes/`，包含所有 R1 机器人的视觉/碰撞 mesh。

### 7.3 PD 驱动参数对齐

04 的 PD 参数严格与 GalaxeaManipSim 一致（第 158-164 行）：

```python
# PD 驱动参数: 与 GalaxeaManipSim 一致 (stiffness=1000, damping=200)
JOINT_STIFFNESS = 1000.0
JOINT_DAMPING = 200.0
GRIPPER_STIFFNESS = 1000.0
GRIPPER_DAMPING = 200.0
```

代码注释明确说明：高刚度（如 100000）配合 `set_qpos` 会导致 PD 力与位置约束冲突产生震荡。GalaxeaManipSim 使用 stiffness=1000, damping=200 的柔顺 PD，04 遵循同一参数。

---

## 八、关键参数速查

### 物理参数

| 参数 | 值 | 含义 |
|------|----|----|
| `PHYSICS_TIMESTEP` | 1/240 s | 物理引擎时间步 |
| `CONTROL_FREQ` | 30 Hz | 控制频率 |
| `DECIMATION` | 8 | 每控制步的物理子步数 |
| `JOINT_STIFFNESS` | 1000 | 臂关节 PD 刚度 |
| `JOINT_DAMPING` | 200 | 臂关节 PD 阻尼 |
| `GRIPPER_STIFFNESS` | 1000 | 夹爪关节 PD 刚度 |
| `GRIPPER_DAMPING` | 200 | 夹爪关节 PD 阻尼 |
| `OBJECT_DENSITY` | 1000 kg/m³ | GLB 物体密度 |
| `GROUND_HEIGHT` | -0.5 m | 物理地面高度 |

### 工作空间参数

| 参数 | 值 | 含义 |
|------|----|----|
| `COMFORTABLE_REACH` | 0.70 m | 基座高度（质心正上方） |
| `COMFORT_TARGET_IN_BASE` | [0.25, 0, -0.55] | base_link 下舒适目标点 |
| `ARM_MAX_REACH` | 0.713 m | 臂展最大距离 |
| `WARMUP_FRAMES` | 30 | 热身过渡帧数 |
| `IK_SOLVE_PER_FRAME` | 20 | 每帧 IK 迭代次数 |
| `BASE_TRACKING_RANGE` | 0.0 | 固定基座（不跟踪手腕） |

### 坐标变换

| 常量 | 含义 |
|------|------|
| `RXWORLD_TO_SAPIEN` | HaWoR Render World (Y-up) → SAPIEN (Z-up) |
| `FLIP_Z_FOR_PHYSICS` | `False` — 不翻转 Z（对齐 02，避免镜像手性不一致） |

---

## 九、已知问题与注意事项

1. **需要 GPU + Vulkan**: 04 必须在带 GPU 的环境运行，沙箱（bwrap）中会报 `failed to find a rendering device`。PyBullet 管线（CPU only）可作为替代验证。

2. **`run_bimanual_tracking`**: 双手模式是扩展功能，使用运动学驱动（非物理），左右臂分别独立求解 IK。当前未与物理仿真完全集成。

3. **左右夹爪镜像**: 单夹爪模式中，左右手使用不同 URDF（finger 几何镜像），通过 `_get_finger_geom(prefix)` 和 `y_sign` 处理 MANO 开合方向翻转。

4. **基座高度差异**: 04 的 `COMFORTABLE_REACH=0.70` 高于 02 的 `0.35`，因为物理仿真中机械臂需垂直抓取避免桌面碰撞。若用 02 轨迹喂给 PyBullet，需注意基座高度不匹配。

5. **碰撞体计算开销**: 默认使用 CoACD 凸分解生成精确碰撞体，首次运行较慢。`--fast-collision` 改用凸包加速，适合调试。
