# tri_model_physics — R1 机器人 SAPIEN 物理仿真抓取

用 R1 机器人 URDF（整个机器人 / 纯夹爪两种形式）在 **SAPIEN** 中复刻抓取 GLB 物体的动作，通过参数级验证（物体提升/接触检测）。

## 核心脚本: grasp_hawor.py

**功能**: 给定 HaWoR 手部重建 + RAS 场景重建 (GLB)，用 R1 机器人 URDF 在 SAPIEN 中复刻抓取 GLB 物体的动作。

**两种 URDF 模式**:
- `full_robot`: `r1_v2_1_0.urdf` (整个机器人), DexRetargeting + RelaxedIK + 纯PD驱动
- `gripper_only`: 纯夹爪 URDF (无机械臂), MANO 指尖向量解析映射

**双视角视频输出**:
- `cam_view_*.mp4`: 相机视角 (对齐 02_render_scene.py 的 `hawor_cam_to_sapien_pose`)
- `god_view_*.mp4`: 上帝视角 (高空朝下面对机器人)

### 运行命令

> **注意**: `/home/an/data/hawor/7` + `/home/an/data/ras/my_7mp4_result` 这组配套数据中，**左手 (hand_idx=0) 是抓物的手**，右手几乎不动。因此示例命令使用 `--side left`。

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics

# full_robot 模式 (整个机器人 + 夹爪, 双视角渲染)
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode full_robot --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# gripper_only 模式 (纯夹爪)
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result

# 指定帧数
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode full_robot --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --num-frames 100

# 仅渲染上帝视角 (可直观看到机器人+夹爪操作物体)
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode full_robot --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god

# 仅渲染第一人称相机视角
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode full_robot --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views cam
```

### 渲染视角 (`--views`)

| 值 | 说明 | 输出文件 |
|------|------|---------|
| `both` (默认) | 双视角: 第一人称 + 上帝视角 | `cam_view_*.mp4` + `god_view_*.mp4` |
| `cam` | 仅第一人称相机视角 (对齐 02_render_scene.py) | `cam_view_*.mp4` |
| `god` | 仅上帝视角 (高空斜俯视抓取区域, 能看到整个机器人+夹爪+物体) | `god_view_*.mp4` |

> **地面透明**: R1 机器人 ROOT 在地下 (z≈-1.0), 地面视觉已隐藏 (保留物理碰撞), 因此视频能看到整个机器人身体, 不会被地面遮挡。

### 输出目录

输出保存在当前脚本目录下的 `output/<mode>_<side>/`:

```
tri_model_physics/output/full_robot_right/
├── cam_view_full_robot_right.mp4      # 相机视角视频
├── god_view_full_robot_right.mp4      # 上帝视角视频
├── grasp.log                          # 完整日志
├── grasp_full_robot_right_qpos.npy    # 关节轨迹
├── grasp_full_robot_right_verify.json # 参数级验证结果
└── alignment/
    └── transform_params.npz           # 01_align_scene.py 对齐参数
```

### 参数级验证

运行结束自动输出验证结果:
- 机械臂关节数 (full_robot 应为 6)
- 臂关节 qpos 范围与运动幅度
- 接触检测 (夹爪-物体接触点数)
- 物体提升量 (Z 轴位移)

### 关键修复: "没有机械臂"根因

`r1_v2_1_0.urdf` 中 `<joint name="..." type="fixed">` 跨多行，旧正则要求 name/type 同行导致匹配失败，臂关节保持 fixed → "0臂关节"。
`grasp_hawor.py` 用 `re.DOTALL` + `[\s\S]*?` 匹配跨行，正确转换 12 个臂关节 fixed→revolute。

### 对齐逻辑 (调用 01_align_scene.py)

```python
# grasp_hawor.py 调用 01_align_scene.py 的 compute_and_save_transform_params()
# 对齐 RAS GLB → HaWoR 坐标系
transform_params = compute_and_save_transform_params(
    ras_output=ras_dir,
    hawor_reconstruction=hawor_npz,
    output_dir=output_dir/alignment,
)
# 变换链: p_hawor = s_inv * R_inv @ p_glb + t_inv
#         p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
```

---

## 旧版三引擎架构 (已弃用)

以下为旧版三形式×三引擎架构文档，已不再维护，推荐使用 `grasp_hawor.py`。

## 完整管线

```
┌─────────────────────────────────────────────────────────────────┐
│  输入数据                                                        │
│  ├─ HaWoR: pred_trans/rot/hand_pose (2,N,...) 手部重建          │
│  └─ RAS:   final_scene.glb + transform_params.npz 场景+变换     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. MANOLayer FK  →  21 个手部关节 3D 坐标                       │
│  2. 坐标变换 RXWORLD_TO_SAPIEN  (渲染坐标系 → SAPIEN坐标系)      │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────┬───────────────────────────────────────────┤
│  gripper_only 形式  │  floating_arm / full_robot 形式           │
│  解析映射(无IK)     │  DexRetargeting → 夹爪关节角              │
│  指尖向量→夹爪位姿  │  RelaxedIK → 6臂关节角                    │
└──────────┬──────────┴──────────────────────┬────────────────────┘
           ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. PD 位置驱动 (目标关节角 → 物理力矩)                          │
│     ├─ SAPIEN:   set_drive_target(stiffness=1000, damping=200)  │
│     ├─ PyBullet: POSITION_CONTROL(Kp=1000, Kd=200)              │
│     └─ MuJoCo:   qfrc_applied = Kp·Δq - Kd·q̇ + qfrc_bias(重力)  │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 物理仿真步进 (decimation=8, 240Hz物理 / 30Hz控制)            │
│     ├─ GLB 物体加载 (大→kinematic, 小→dynamic 凸包碰撞)         │
│     ├─ 接触检测 (SAPIEN: scene.get_contacts / PyBullet / MuJoCo)│
│     └─ 摩擦力抓取 (μ=1.0, 法向力来自PD驱动)                     │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. 输出                                                         │
│     ├─ MP4 视频: output/{form}_{backend}_{side}.mp4 (30fps)     │
│     ├─ QPOS 序列: output/{form}_{backend}_{side}_qpos.npy       │
│     └─ 抓取状态: grasp_states (每帧 bool)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
tri_model_physics/
├── __init__.py                    # 模块入口
├── physics_utils.py               # 共享物理参数与工具函数
├── trajectory_loader.py           # HaWoR手部数据+GLB场景加载
├── grasp_controller.py            # 夹取GLB物体控制逻辑
├── video_recorder.py              # 三引擎通用视频录制 (MP4输出)
├── run_tri_model.py               # 主入口脚本
├── CHANGE_LOG.md                  # 变更日志
├── models/                        # 三种机器人形式
│   ├── __init__.py
│   ├── robot_forms.py             # 形式定义与URDF解析
│   └── urdf_templates.py          # 纯夹爪URDF模板生成
├── sapien_backend/                # SAPIEN仿真后端
│   ├── __init__.py
│   ├── sapien_env.py              # SAPIEN场景+物理引擎
│   └── sapien_runner.py           # 三形式跟踪执行器
├── pybullet_backend/              # PyBullet仿真后端
│   ├── __init__.py
│   ├── pybullet_env.py            # PyBullet场景+物理引擎
│   └── pybullet_runner.py         # 三形式跟踪执行器
├── mujoco_backend/                # MuJoCo仿真后端
│   ├── __init__.py
│   ├── mujoco_env.py              # MuJoCo场景+PD力矩控制+重力补偿
│   └── mujoco_runner.py           # 三形式跟踪执行器
├── output/                        # 输出目录 (MP4视频 + QPOS npy)
└── tests/                         # 测试用例
    ├── __init__.py
    ├── test_models.py             # URDF加载与结构验证
    ├── test_sapien_backend.py     # SAPIEN后端测试
    ├── test_pybullet_backend.py   # PyBullet后端测试
    ├── test_trajectory.py         # 轨迹加载测试
    ├── test_grasp.py              # 夹取控制测试
    └── test_full_pipeline.py      # 端到端集成测试
```

## 三种机器人形式 × 三引擎

| 形式 | URDF来源 | 关节构成 | 跟踪方式 |
|------|---------|---------|---------|
| **full_robot** | `r1_v2_1_0.urdf` | 躯干4+双臂12+双夹爪4 | Retargeting+IK+PD驱动 |
| **floating_arm** | `r1_v2_1_0_floating_right.urdf` | 6臂+2夹爪(浮动底座) | Retargeting+IK+PD驱动 |
| **gripper_only** | 模板生成 | 2夹爪(prismatic) | 解析映射(无IK) |

| 引擎 | 驱动方式 | 重力补偿 | GLB物体 | 视频录制 |
|------|---------|---------|---------|---------|
| **SAPIEN** | PD位置驱动 (set_drive_target) | 高刚度误差补偿 | OBJ+凸包碰撞 | CameraEntity |
| **PyBullet** | POSITION_CONTROL (Kp/Kd) | 高刚度误差补偿 | OBJ+凸包碰撞 | getCameraImage |
| **MuJoCo** | PD力矩+重力补偿 (qfrc_applied) | qfrc_bias显式补偿 | 暂不加载 | Renderer |

## 快速开始

### 环境要求

```bash
conda activate dex  # 需要 sapien, pybullet, mujoco, trimesh, torch, manopth, imageio 等
```

### 设置 Vulkan (SAPIEN)

```bash
# NVIDIA GPU
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# Intel 集显
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
```

> **注意**: 若遇到 `RuntimeError: failed to find a rendering device`，请检查当前用户是否在 `video` 和 `render` 组，并确认对 `/dev/dri/renderD*` 有读写权限：
> ```bash
> sudo usermod -aG video,render $USER
> # 重新登录后生效
> ```

### 命令调用

```bash
# 进入目录
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics

# 1. 仅测试模型加载 (不需要数据)
conda run -n dex python run_tri_model.py --test-models

# 2. 单形式+单引擎运行 (默认全部帧, 自动生成MP4)
conda run -n dex python run_tri_model.py \
    --backend sapien \
    --form gripper_only \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --output-dir ./output

# 3. 指定帧数
conda run -n dex python run_tri_model.py \
    --backend pybullet \
    --form floating_arm \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --num-frames 100 \
    --output-dir ./output

# 4. 一键运行全部9组合 (3形式×3引擎, 子进程隔离避免GL冲突)
conda run -n dex python run_tri_model.py \
    --all \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --output-dir ./output

# 5. MuJoCo 后端
conda run -n dex python run_tri_model.py \
    --backend mujoco \
    --form gripper_only \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --output-dir ./output

# 6. GUI模式 (可视化)
conda run -n dex python run_tri_model.py \
    --backend pybullet --form gripper_only --gui \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --num-frames 50
```

### 视频输出位置

MP4 视频保存在 `--output-dir` 指定目录下，命名格式:
```
{form}_{backend}_{side}.mp4
```
例如:
- `output/gripper_only_sapien_right.mp4`
- `output/floating_arm_pybullet_right.mp4`
- `output/full_robot_mujoco_right.mp4`

同时保存 QPOS 序列:
```
{form}_{backend}_{side}_qpos.npy
```

### 运行测试

```bash
# 全部单元测试
conda run -n dex python -m pytest tests/ -v

# 单个测试文件
conda run -n dex python -m pytest tests/test_models.py -v
```

### GalaxeaManipSim 抓取演示 (grasp_demo.py)

`grasp_demo.py` 使用 **SAPIEN** 仿真器 (通过 GalaxeaManipSim 的 gym 环境封装), 与三引擎架构中的 SAPIEN 后端是**同一物理引擎**, 只是上层封装不同 (GalaxeaManipSim 的 `DualBottlesPickEasyEnv` + `BimanualPlanner`).

**不是第四种引擎**, 无需额外添加 — 抓取功能已通过 `grasp_demo.py` 提供.

```bash
# 运行抓取演示 (参数级验证抓取成功)
conda run -n dex python grasp_demo.py --output output/grasp_demo.mp4
```

**参数级验证输出**:
- 红方块最终位置 vs 目标位置 (距离 < 0.15m 容差)
- 方块提升高度 (初始 z=0.950 → 最终 z=1.178, 提升 22.8cm)
- 方块与末端同步上升 (差异 < 3mm, 抓取牢固)
- 夹爪开合 qpos 轨迹 (open→0.035, close→0.046/0.038)
- 完整参数轨迹: `output/grasp_demo_param_log.json`

### Python API 调用

```python
# SAPIEN + gripper_only (全部帧, 输出视频)
from sapien_backend.sapien_runner import SapienRunner
runner = SapienRunner("gripper_only", "right", headless=True)
runner.build()
result = runner.run_tracking(
    hawor_dir="/home/an/data/hawor/7",
    ras_dir="/home/an/data/ras/my_7mp4_result",
    transform_params_path="/home/an/data/ras/my_7mp4_result/alignment/transform_params.npz",
    num_frames=-1,  # 全部帧
    output_video="./output/gripper_only_sapien.mp4",
)
print(f"完成 {len(result['qpos_sequence'])} 帧, 抓取帧数: {sum(result['grasp_states'])}")

# MuJoCo + floating_arm
from mujoco_backend.mujoco_runner import MuJoCoRunner
runner = MuJoCoRunner("floating_arm", "right", headless=True)
runner.build()
result = runner.run_tracking(
    hawor_dir="/home/an/data/hawor/7",
    ras_dir="/home/an/data/ras/my_7mp4_result",
    transform_params_path="/home/an/data/ras/my_7mp4_result/alignment/transform_params.npz",
    num_frames=-1,
    output_video="./output/floating_arm_mujoco.mp4",
)
runner.disconnect()
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend` | `sapien` | 仿真后端: `sapien` / `pybullet` / `mujoco` |
| `--form` | `floating_arm` | 机器人形式: `full_robot` / `floating_arm` / `gripper_only` |
| `--side` | `right` | 手臂侧别: `right` / `left` |
| `--hawor-dir` | 自动查找 | HaWoR 手部重建数据目录 |
| `--ras-dir` | 自动查找 | RAS 场景数据目录 |
| `--transform-params` | 自动查找 | 变换参数 npz 路径 |
| `--num-frames` | `-1` (全部) | 运行帧数 |
| `--start-frame` | `0` | 起始帧 |
| `--headless` | `True` | 无头模式 |
| `--gui` | `False` | GUI模式 |
| `--output-dir` | `None` | 输出目录 (MP4 + QPOS npy) |
| `--all` | `False` | 运行全部9组合 (子进程隔离) |
| `--test-models` | `False` | 仅测试模型加载 |

---

## 机械臂驱动方式详解

### 1. 驱动模式: 位置PD驱动 (Position-based PD Control)

本系统采用**位置PD驱动**。每个控制周期计算目标关节角，通过PD控制器产生力矩驱动物理仿真中的关节到达目标位置。

**为什么选择位置驱动而非力矩驱动？**
- 轨迹跟踪任务的核心需求是"到达指定位置"，位置驱动天然适合
- 力矩驱动需要额外的力矩计算层（逆动力学），增加复杂度且在欠驱动系统中不稳定
- 位置PD驱动在高刚度下近似刚性跟踪，同时保留物理交互能力（碰撞响应、夹取力）

### 2. SAPIEN PD驱动实现

```python
# 设置PD驱动参数 (在URDF加载后)
for joint in robot.get_active_joints():
    if "gripper_finger" in joint.name:
        joint.set_drive_property(stiffness=1000, damping=200)  # 夹爪
    else:
        joint.set_drive_property(stiffness=1000, damping=200)  # 臂关节

# 每个控制步: 设置目标位置 → 物理引擎自动计算PD力矩
for i, val in enumerate(target_qpos):
    robot.get_active_joints()[i].set_drive_target(float(val))

# 执行 decimation=8 次物理子步
for _ in range(8):
    scene.step()
```

**PD力矩计算公式** (SAPIEN内部):
```
τ = stiffness × (q_target - q_current) - damping × q̇_current
```

### 3. PyBullet PD驱动实现

```python
# 禁用默认电机
for idx in arm_joint_indices + gripper_joint_indices:
    p.setJointMotorControl2(robot_id, idx, VELOCITY_CONTROL, force=0)

# 每个控制步: 位置控制
p.setJointMotorControl2(
    robot_id, idx, POSITION_CONTROL,
    targetPosition=float(val),
    force=PD_KP_ARM * 10,  # 最大力矩限制
)
```

### 4. MuJoCo PD驱动实现 (力矩控制 + 重力补偿)

MuJoCo 采用**直接力矩控制**，与 SAPIEN/PyBullet 不同，需要显式处理重力补偿:

```python
def step_physics(self, target_qpos):
    # 1. 计算重力补偿 (qfrc_bias 包含 C(q,q̇)q̇ + g(q))
    mujoco.mj_forward(self.model, self.data)
    gravity_comp = self.data.qfrc_bias.copy()

    # 2. PD 力矩 + 重力补偿
    for idx, target_val in target_qpos.items():
        pd_torque = kp * (target_val - qpos[idx]) - kd * qvel[idx]
        total_torque = pd_torque + gravity_comp[idx]  # 关键: 显式重力补偿
        # 3. 裁剪力矩防止仿真爆炸
        total_torque = np.clip(total_torque, -limit, limit)
        self.data.qfrc_applied[idx] = total_torque

    # 4. 物理子步
    for _ in range(DECIMATION):
        mujoco.mj_step(self.model, self.data)
```

**MuJoCo 增益参数** (比 SAPIEN/PyBullet 保守, 防止数值爆炸):
- 臂关节: Kp=100, Kd=20, 力矩限制=50 N·m
- 夹爪关节: Kp=200, Kd=40, 力矩限制=20 N·m

**为什么 MuJoCo 需要更小的增益？**
- MuJoCo 的积分方式对高刚度敏感，Kp=1000 会导致 QACC NaN
- 显式重力补偿 (qfrc_bias) 减轻了 PD 控制器的负担，不需要极高刚度
- 力矩裁剪是最后的安全网，防止大位置误差产生过大力矩

### 5. 克服重力约束

#### 5.1 SAPIEN/PyBullet: 持续位置误差补偿
```
τ_gravity_compensation ≈ stiffness × Δq_gravity_sag
```
高刚度(1000)意味着极小的下垂即可产生足够补偿力矩:
- 典型重力下垂: ~0.001 rad → 补偿力矩: 1000 × 0.001 = 1.0 Nm

#### 5.2 MuJoCo: 显式重力补偿
```
τ_total = τ_PD + qfrc_bias
```
`qfrc_bias` 是 MuJoCo 自动计算的逆动力学项，包含重力、科氏力、离心力，直接补偿无需高刚度。

#### 5.3 Decimation策略
```
控制频率: 30Hz (1/30s)
物理子步: 240Hz (1/240s)
Decimation: 8 (每个控制步执行8次物理子步)
```

#### 5.4 浮动底座跟踪 (floating_arm形式)
浮动臂的底座通过`set_root_pose()`直接设置位姿(运动学驱动)，绕过重力影响。

### 6. 夹爪驱动与夹取力

#### 6.1 夹爪关节类型
原始 GalaxeaManipSim URDF 中 `gripper_finger_joint` 是 `fixed` 类型，系统自动替换为 `prismatic`:
```python
# _make_prismatic_gripper_urdf() 自动替换
type="fixed" → type="prismatic"
<axis xyz="0 -1 0"/>  # joint1: 向内闭合
<limit lower="0" upper="0.05" effort="100" velocity="0.25"/>
```

#### 6.2 夹取力来源
夹取力完全由**摩擦力**实现:
```
F_grasp = μ × N
```
- `μ = 1.0` (高摩擦材质)
- `N` = PD驱动力矩在接触点产生的法向力

### 7. 三形式驱动差异

| 特性 | full_robot | floating_arm | gripper_only |
|------|-----------|-------------|-------------|
| 臂关节驱动 | PD位置驱动 | PD位置驱动 | 无臂关节 |
| 夹爪驱动 | PD位置驱动 | PD位置驱动 | PD位置驱动 |
| 底座 | 固定(fixed base) | 运动学驱动 | 运动学驱动 |
| 重力补偿 | PD误差补偿 | PD误差补偿+底座运动学 | 底座运动学 |
| IK求解 | RelaxedIK | RelaxedIK | 无需IK |
| 夹爪映射 | DexRetargeting | DexRetargeting | 解析映射 |

### 8. GLB物体物理分类

| 类型 | 判定条件 | 物理属性 | 交互方式 |
|------|---------|---------|---------|
| 场景结构(桌面/地板) | 体积 > 0.1m³ | kinematic(不受力) | 机器人碰撞但不移动 |
| 可抓取物体(杯子等) | 体积 < 0.1m³ | dynamic(density=1000, 凸包碰撞) | 可被夹爪夹起移动 |

---

## 测试结果

### 当前环境验证状态

测试数据: `/home/an/data/hawor/7` (113帧), `/home/an/data/ras/271_vggt_omega`

**全部 9 组合通过 (30帧快速测试)**:

| 组合 | 状态 | 视频输出 | 日志 |
|------|------|---------|------|
| full_robot × sapien | ✓ | `full_robot_sapien_right.mp4` | `full_robot_sapien_right.log` |
| floating_arm × sapien | ✓ | `floating_arm_sapien_right.mp4` | `floating_arm_sapien_right.log` |
| gripper_only × sapien | ✓ | `gripper_only_sapien_right.mp4` | `gripper_only_sapien_right.log` |
| full_robot × pybullet | ✓ | `full_robot_pybullet_right.mp4` | `full_robot_pybullet_right.log` |
| floating_arm × pybullet | ✓ | `floating_arm_pybullet_right.mp4` | `floating_arm_pybullet_right.log` |
| gripper_only × pybullet | ✓ | `gripper_only_pybullet_right.mp4` | `gripper_only_pybullet_right.log` |
| full_robot × mujoco | ✓ | `full_robot_mujoco_right.mp4` | `full_robot_mujoco_right.log` |
| floating_arm × mujoco | ✓ | `floating_arm_mujoco_right.mp4` | `floating_arm_mujoco_right.log` |
| gripper_only × mujoco | ✓ | `gripper_only_mujoco_right.mp4` | `gripper_only_mujoco_right.log` |

**113帧完整测试 (floating_arm × pybullet)**:
- 视频: 640x480, 30fps, 112帧, 3.73s
- QPOS 跟随: 臂关节总移动量 3.35, 帧间变化 mean=0.040
- 夹爪限位: `[0.0000, 0.0500]` (严格在 `[0, 0.05]` 安全范围内)

> **注意**: `--all` 模式下每个后端用独立子进程运行，避免 SAPIEN/PyBullet/MuJoCo 的 OpenGL 上下文冲突 (`gladLoadGL error`)。每个子进程的日志自动保存到 `output/{form}_{backend}_{side}.log`。

---

## 关键技术细节

### URDF处理
- 原始 GalaxeaManipSim URDF 中 `gripper_finger_joint` 是 `fixed` 类型，通过 `_make_prismatic_gripper_urdf()` 自动替换为 `prismatic`
- SAPIEN: 通过临时URDF替换 `package://` 为绝对路径
- PyBullet: 通过 symlink + 路径替换解决 mesh 查找，GLB→OBJ自动转换
- MuJoCo: 将 URDF 写入 mesh 目录，相对路径解析 STL 文件

### SAPIEN 3.0 API 适配
- `add_visual_from_trimesh` → `add_visual_from_file` (导出临时OBJ)
- `actor.get_contacts()` → `scene.get_contacts()` + 过滤
- `set_friction` → `shape.set_physical_material(PhysxMaterial(...))`

### 轨迹跟踪链路

**gripper_only (解析映射)**:
```
HaWoR pred_trans/rot/hand_pose
  → MANOLayer FK → 21关节3D坐标
  → compute_analytical_gripper_pose() → 夹爪根位姿 + 手指关节值
  → set_root_pose() + set_drive_target()
```

**floating_arm/full_robot (Retargeting+IK)**:
```
HaWoR pred_trans/rot/hand_pose
  → MANOLayer FK → 21关节3D坐标
  → DexRetargeting.retarget(ref_value, fixed_qpos) → 夹爪关节角
  → RelaxedIK.solve_position() → 6臂关节角
  → set_drive_target() → PD驱动
```

### 数据格式
HaWoR 数据 (npz) shape:
- `pred_trans`: (2, N, 3) — 2只手×N帧×3坐标
- `pred_rot`: (2, N, 3)
- `pred_hand_pose`: (2, N, 45)
- `pred_betas`: (2, N, 10)

`load_hawor_data()` 按 `hand_idx` 自动索引: 0=左手, 1=右手

可用数据集:
- `/home/an/data/hawor/7` — 113帧
- `/home/an/data/hawor/7_vggt-omega` — 113帧
- `/home/an/data/hawor/hoi4d` — 600帧
- `/home/an/data/hawor/laptop` — 600帧
