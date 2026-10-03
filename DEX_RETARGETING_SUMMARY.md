# dex-retargeting 项目全面分析与总结

---

## 一、项目概览

**项目名称**: `dex-retargeting` (v0.5.0)

**作者**: Yuzhe Qin (y1qin@ucsd.edu)

**许可证**: MIT

**仓库地址**: https://github.com/dexsuite/dex-retargeting

**项目定位**: 灵巧手姿态重定向（Hand Pose Retargeting）工具库，用于将人手姿态映射到多种机器人灵巧手上。支持三种重定向方法：**位置重定向 (Position)**、**向量重定向 (Vector)** 和 **DexPilot 重定向**。

**核心依赖**: `numpy`, `pytransform3d`, `pinocchio`, `nlopt`, `pytorch`, `pyyaml`, `lxml`

---

## 二、目录结构总览

```
dex-retargeting/
├── pyproject.toml                    # 主项目配置（构建系统、依赖）
├── README.md                         # 项目说明
├── LICENSE                           # MIT 许可证
├── CITATION.cff                      # 引用信息
│
├── src/dex_retargeting/              # ★ 核心库代码
│   ├── __init__.py                   # 包入口，检查 torch 安装
│   ├── constants.py                  # 枚举常量和映射表
│   ├── retargeting_config.py         # 重定向配置（数据类 + 解析逻辑）
│   ├── optimizer.py                  # 三种优化器实现
│   ├── seq_retarget.py               # 序列重定向器（含时序滤波）
│   ├── robot_wrapper.py              # Pinocchio 机器人封装
│   ├── kinematics_adaptor.py         # 运动学适配器（模拟关节）
│   ├── optimizer_utils.py            # 低通滤波器
│   ├── yourdfpy.py                   # URDF 解析库（内嵌修改版）
│   └── configs/                      # 配置文件
│       ├── offline/                  # 离线重定向配置（17个YAML）
│       └── teleop/                   # 遥操作重定向配置（24个YAML）
│
├── example/                          # ★ 示例代码
│   ├── position_retargeting/         # 位置重定向示例
│   │   ├── r1_hand_tracking_video.py     # ★ 本次重点：R1 追踪视频
│   │   ├── r1_relaxed_ik_from_dex.py     # R1 RelaxedIK 完整流程
│   │   ├── dataset.py                    # DexYCB 数据集加载器
│   │   ├── hand_viewer.py                # 人手可视化（SAPIEN）
│   │   ├── hand_robot_viewer.py          # 人手+机器人可视化
│   │   ├── mano_layer.py                 # MANO 手模型封装
│   │   ├── visualize_hand_object.py      # 手-物可视化
│   │   ├── render_hand_object.py         # 手-物渲染
│   │   ├── mujoco_dexycb_gripper.py      # MuJoCo 夹爪示例
│   │   ├── mujoco_dexycb_gripper_full.py # MuJoCo 夹爪完整版
│   │   ├── mujoco_ycb_objects.py         # MuJoCo YCB 物体
│   │   └── debug_sapien_pose.py          # 姿态调试
│   ├── vector_retargeting/           # 向量重定向示例
│   │   ├── detect_from_video.py
│   │   ├── single_hand_detector.py
│   │   ├── show_realtime_retargeting.py
│   │   ├── capture_webcam.py
│   │   ├── render_robot_hand.py
│   │   └── mujoco_retargeting_viewer.py
│   └── profiling/                    # 性能分析
│       ├── profile_online_retargeting.py
│       └── generate_human_data_from_video.py
│
├── tests/                            # 测试
│   ├── test_optimizer.py
│   └── test_retargeting_config.py
│
├── assets/                           # ★ 机器人资产
│   ├── pyproject.toml                # 资产子包配置
│   ├── robots/                       # 机器人 URDF/SRDF 模型
│   │   ├── arms/                     # 机械臂模型（6种）
│   │   │   ├── iiwa7/  iiwa14/       # KUKA 系列
│   │   │   ├── ur5e/   ur10e/        # Universal Robots 系列
│   │   │   ├── xarm7/                # UFactory xArm7
│   │   │   └── rm75/                 # 睿尔曼 RM75
│   │   ├── hands/                    # 灵巧手/夹爪模型（10种）
│   │   │   ├── ability_hand/         # Ability Hand（左右）
│   │   │   ├── allegro_hand/         # Allegro Hand（含变体）
│   │   │   ├── barrett_hand/         # Barrett Hand
│   │   │   ├── dclaw_gripper/        # DClaw 夹爪
│   │   │   ├── inspire_hand/         # 因时 Inspire Hand
│   │   │   ├── leap_hand/            # Leap Hand
│   │   │   ├── panda_gripper/        # Franka Panda 夹爪
│   │   │   ├── r1_gripper/           # R1 夹爪
│   │   │   ├── schunk_hand/          # Schunk SVH Hand
│   │   │   └── shadow_hand/          # Shadow Hand
│   │   ├── assembly/                 # 臂-手装配体
│   │   │   ├── rm75_inspire/
│   │   │   ├── ur5e_shadow/
│   │   │   └── xarm7_ability/
│   │   └── r1_full/                  # R1 整体机器人
│   ├── scripts/                      # 工具脚本
│   ├── tools/                        # URDF 处理工具
│   ├── example/                      # 资产渲染示例
│   └── doc/                          # 文档图片
│
└── .github/workflows/                # CI/CD
    ├── build_nightly.yml
    ├── build_stable.yml
    └── test.yml
```

---

## 三、核心模块详细分析

### 3.1 入口模块：`src/dex_retargeting/__init__.py`

- **功能**: 包初始化，唯一作用是检查 PyTorch 是否已安装
- **逻辑**: 调用 `importlib.util.find_spec("torch")` 检测，未安装则抛出 `ImportError` 并提示安装命令
- **文件位置**: [src/dex_retargeting/__init__.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/__init__.py)

### 3.2 常量定义：`constants.py`

- **文件位置**: [src/dex_retargeting/constants.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/constants.py)

定义了三组核心枚举和映射：

| 枚举/映射 | 用途 |
|---|---|
| `RobotName` | 支持的机器人名称：allegro, shadow, svh, leap, ability, inspire, panda, r1, r1_full |
| `RetargetingType` | 重定向方法：vector, position, dexpilot |
| `HandType` | 手部类型：left, right |
| `ROBOT_NAME_MAP` | RobotName → 配置文件夹名的映射 |
| `OPERATOR2MANO` | 操作员坐标系 → MANO 坐标系的旋转矩阵 |
| `get_default_config_path()` | 根据机器人、重定向类型、手型自动查找配置文件路径 |

**关键设计**: `OPERATOR2MANO_RIGHT` 和 `OPERATOR2MANO_LEFT` 定义了右手/左手从操作员坐标系到 MANO 标准坐标系的旋转矩阵，用于姿态坐标变换。

### 3.3 配置文件系统：`retargeting_config.py`

- **文件位置**: [src/dex_retargeting/retargeting_config.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/retargeting_config.py)

核心数据类 `RetargetingConfig`，实现了完整的重定向配置管理：

**配置参数**:

| 参数 | 类型 | 说明 |
|---|---|---|
| `type` | str | 重定向类型：vector/position/dexpilot |
| `urdf_path` | str | 机器人 URDF 文件路径 |
| `add_dummy_free_joint` | bool | 是否添加 6-DoF 虚拟自由关节 |
| `target_link_human_indices` | ndarray | 人手关节→机器人连杆的索引映射 |
| `target_link_names` | List[str] | 位置重定向的目标连杆名 |
| `target_joint_names` | List[str] | 优化的目标关节名 |
| `scaling_factor` | float | 向量重定向的缩放因子 |
| `normal_delta` | float | 正则化权重 |
| `huber_delta` | float | Huber 损失参数 |
| `low_pass_alpha` | float | 低通滤波系数 |

**核心方法 `build()` 流程**:

```
URDF 文件 → yourdfpy 解析 → 写临时文件 → Pinocchio 加载
→ 根据 type 选择优化器 (PositionOptimizer / VectorOptimizer / DexPilotOptimizer)
→ 解析 mimic 关节 → 设置运动学适配器
→ 创建 SeqRetargeting 实例（含低通滤波）
```

**配置文件结构示例** (`r1_full.yml`):

```yaml
retargeting:
  type: position
  urdf_path: r1_full/r1_lite_robot_glb.urdf
  target_joint_names: ["right_gripper_finger_joint1", "right_gripper_finger_joint2"]
  target_link_names: ["right_gripper_finger_link1", "right_gripper_finger_link2"]
  target_link_human_indices: [4, 8]   # thumb_tip=4, index_tip=8
  add_dummy_free_joint: True
  low_pass_alpha: 1
  ignore_mimic_joint: False
```

### 3.4 优化器模块：`optimizer.py`

- **文件位置**: [src/dex_retargeting/optimizer.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/optimizer.py)

实现三种优化器的完整类层次结构：

```
Optimizer (抽象基类)
├── PositionOptimizer    # 位置重定向：最小化 3D 位置误差
├── VectorOptimizer      # 向量重定向：最小化方向向量误差
└── DexPilotOptimizer    # DexPilot 重定向：带抓取先验的向量重定向
```

**基类 `Optimizer` 核心逻辑**:

- 使用 **NLopt** 库的 `LD_SLSQP` 算法进行非线性优化
- `idx_pin2target`: Pinocchio 关节顺序 → 优化目标关节的索引映射
- `idx_pin2fixed`: 不需要优化的固定关节索引
- `retarget(ref_value, fixed_qpos, last_qpos)`: 执行优化的主入口
  - 构建目标函数 → `self.opt.set_min_objective()` → `self.opt.optimize()`

**`PositionOptimizer` 目标函数**:

```
损失 = HuberLoss(robot_link_positions, human_joint_positions)
     + normal_delta * ||qpos - last_qpos||²

梯度 = ∂Loss/∂(link_pos) @ Jacobian + 2*normal_delta*(qpos - last_qpos)
```

工作流程:
1. 设置关节角度 → 正运动学计算 → 获取目标连杆位姿
2. 提取连杆位置 (4×4 矩阵的平移部分)
3. 与人类关节参考值比较，计算 Huber 损失
4. 使用 Pinocchio 计算空间 Jacobian
5. 通过 `adaptor.backward_jacobian()` 处理 mimic 关节约束

**`VectorOptimizer` 目标函数**:

```
loss = HuberLoss(||robot_vec - target_vec * scaling||)
```

计算的是两个连杆之间的方向向量（如手腕→指尖），而非绝对位置。

**`DexPilotOptimizer`** (参考 [DexPilot 论文](https://arxiv.org/abs/1910.03135)):

- 在 VectorOptimizer 基础上增加**投影机制**（`project_dist` / `escape_dist`）
- 当手指捏合距离 < `project_dist` 时触发投影：将手指距离固定在更小的值，模拟抓取
- 当手指张开距离 > `escape_dist` 时退出投影
- 用于实现更稳定的遥操作抓取

### 3.5 序列重定向器：`seq_retarget.py`

- **文件位置**: [src/dex_retargeting/seq_retarget.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/seq_retarget.py)

`SeqRetargeting` 类在优化器之上封装了时序相关的功能：

| 功能 | 方法 | 说明 |
|---|---|---|
| 热启动 | `warm_start()` | 解析计算初始手腕位姿，避免优化|
| 重定向 | `retarget(ref_value, fixed_qpos)` | 执行优化，带关节限制和时序滤波 |
| 状态管理 | `set_qpos()` / `get_qpos()` | 维护 `last_qpos` 用于时间正则化 |
| 性能统计 | `verbose()` | 输出累计耗时 |
| 重置 | `reset()` | 恢复到初始状态 |

**热启动 (warm_start) 算法**:
- 仅适用于有 6-DoF 虚拟自由关节的机器人手
- 通过解析方式计算手腕的初始位姿（避免第一帧的优化不稳定）
- 使用 `OPERATOR2MANO` 矩阵进行坐标系转换

### 3.6 机器人封装：`robot_wrapper.py`

- **文件位置**: [src/dex_retargeting/robot_wrapper.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/robot_wrapper.py)

基于 **Pinocchio** 的轻量级机器人封装：

```
RobotWrapper
├── 属性: joint_names, dof_joint_names, dof, link_names, joint_limits
├── 查询: get_joint_index(), get_link_index(), get_joint_parent_child_frames()
└── 运动学: compute_forward_kinematics(), get_link_pose(), get_link_pose_inv(),
            compute_single_link_local_jacobian()
```

### 3.7 运动学适配器：`kinematics_adaptor.py`

- **文件位置**: [src/dex_retargeting/kinematics_adaptor.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/kinematics_adaptor.py)

`MimicJointKinematicAdaptor` 处理 **模拟关节（Mimic Joint）**——即一个关节的运动受另一个关节驱动的场合（如夹爪的两个手指对称运动）：

```
forward_qpos():  source_qpos * multiplier + offset → mimic_qpos
backward_jacobian():  将 mimic 关节的 Jacobian 贡献合并到 source 关节
```

### 3.8 低通滤波器：`optimizer_utils.py`

- **文件位置**: [src/dex_retargeting/optimizer_utils.py](file:///home/an/robot_world_ws/src/dex-retargeting/src/dex_retargeting/optimizer_utils.py)

一阶低通滤波器 `LPFilter`：

```python
y_new = y_old + alpha * (x - y_old)
```

- `alpha=1`: 不过滤
- `alpha=0`: 不更新（冻结）
- 默认在配置中设为 1

---

## 四、示例代码模块分析

### 4.1 DexYCB 数据集加载器：`dataset.py`

- **文件位置**: [example/position_retargeting/dataset.py](file:///home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting/dataset.py)

`DexYCBVideoDataset` 负责加载 DexYCB 数据集：

**数据结构**:
- `hand_pose`: (N, 51) - MANO 参数 [pose(48) + trans(3)]
- `object_pose`: (N, K, 7) - YCB 物体位姿 [quat_xyzw(4) + pos_xyz(3)]
- `hand_shape`: (10,) - MANO 手部形状参数 betas
- `extrinsics`: (4, 4) - 外参矩阵
- `ycb_ids`: [int] - YCB 物体 ID 列表

**物体运动检测**: `is_object_move()` 方法通过比较当前帧和未来帧的位移来判断物体是否在运动，用于过滤静止不动的数据段。

### 4.2 MANO 手模型：`mano_layer.py`

- **文件位置**: [example/position_retargeting/mano_layer.py](file:///home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting/mano_layer.py)

封装 `manopth` 库的 `ManoLayer`，输入姿态参数输出手部顶点(778个)和关节(21个)。

### 4.3 可视化系统

**`hand_viewer.py`** [文件位置](file:///home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting/hand_viewer.py):
- 基于 SAPIEN 渲染引擎
- 创建场景、光照、桌面、相机
- `_compute_hand_geometry()`: 通过 MANO 层计算手部顶点和关节
- `_update_hand()`: 更新渲染节点

**`hand_robot_viewer.py`** [文件位置](file:///home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting/hand_robot_viewer.py):
- 继承 `HandDatasetSAPIENViewer`
- 额外加载多个机器人 URDF 模型
- 为每个机器人创建 `SeqRetargeting` 实例
- `render_dexycb_data()`: 同时渲染人手、YCB物体和所有机器人的重定向结果

---

## 五、支持的机器人列表

### 机械臂 (Assets: `assets/robots/arms/`)

| 机器人 | URDF文件 | DoF |
|---|---|---|
| KUKA iiwa7 | iiwa7.urdf | 7 |
| KUKA iiwa14 | iiwa14.urdf | 7 |
| Universal Robots UR5e | ur5e.urdf | 6 |
| Universal Robots UR10e | ur10e.urdf | 6 |
| UFactory xArm7 | xarm7.urdf | 7 |
| 睿尔曼 RM75 | rm75.urdf | 7 |

### 灵巧手/夹爪 (Assets: `assets/robots/hands/`)

| 手部 | 支持类型 | 说明 |
|---|---|---|
| Ability Hand | 左右 | 五指灵巧手 |
| Allegro Hand | 左右 | 四指灵巧手（含 FSR/触觉变体） |
| Barrett Hand | - | 三指灵巧手 |
| DClaw Gripper | - | 三指夹爪 |
| Inspire Hand | 左右 | 因时五指灵巧手 |
| Leap Hand | 左右 | 低成本五指灵巧手 |
| Panda Gripper | - | Franka Panda 平行夹爪 |
| R1 Gripper | - | R1 机器人夹爪 |
| Schunk SVH Hand | 左右 | Schunk 五指灵巧手 |
| Shadow Hand | 左右 | Shadow 五指灵巧手（含双手） |

### 装配体 (Assets: `assets/robots/assembly/`)

| 装配体 | 组成 |
|---|---|
| `rm75_inspire` | RM75 臂 + Inspire 手 |
| `ur5e_shadow` | UR5e 臂 + Shadow 手 |
| `xarm7_ability` | xArm7 臂 + Ability 手 |

---

## 六、⭐ 重点文件分析：`r1_hand_tracking_video.py`

### 6.1 功能目标

实现完整的 **DexYCB → Dex Retargeting（手部）→ RelaxedIK（R1右臂）→ 视频生成** 管道：

```
DexYCB 数据集
    │
    ▼
MANO 手部姿态解析（pose 48维 + trans 3维）
    │
    ▼
Dex Retargeting（PositionOptimizer）
  - 将人手关节映射到 R1 夹爪连杆
    │
    ▼
MANO 手腕 6-DoF 位姿提取
  - 位置：joints[0, :3]（手腕关节点）
  - 朝向：pose[:3] compact axis-angle → quaternion
    │
    ▼
坐标变换：MANO 坐标系 → R1 世界坐标系
  - 使用 OPERATOR2MANO 旋转矩阵
  - 施加固定平移偏移
    │
    ▼
RelaxedIK 求解 R1 右臂 6 关节
  - 输入：末端期望位姿 (pos + quat)
  - 输出：6 个关节角度
    │
    ▼
人手拇指-食指距离 → R1 夹爪开合度映射
  - [0.02m, 0.08m] → [0, 0.05]m
    │
    ▼
SAPIEN 渲染 → cv2 VideoWriter → .mp4 视频
```

### 6.2 原实现存在的问题与不足

对原始代码逐行分析后发现以下关键问题：

#### 问题 1：手腕朝向硬编码为恒等四元数

```python
# 原代码 (L149)
wrist_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 简化：单位四元数
```

**问题**: MANO 的 `pose` 参数前 3 维是**全局旋转**（手腕在相机坐标系下的朝向），用 `compact_axis_angle` 表示。直接忽略会使 R1 机械臂无法正确复现手腕旋转姿态。

**影响**: 人手转动手腕时，R1 末端保持固定朝向，丧失姿态信息。

#### 问题 2：坐标变换仅为简单加法偏移

```python
# 原代码 (L153)
r1_target_pos = wrist_pos + np.array([0.6, 0.0, 1.0])
```

**问题**:
- 使用固定偏移向量 `[0.6, 0.0, 1.0]`，未考虑不同数据样本中坐标范围的变化
- 缺少 MANO → 世界坐标系的完整旋转对齐

**影响**: 不同轨迹的坐标范围差异很大时，R1 可能求解到不可达位置。

#### 问题 3：硬编码绝对路径

```python
# 原代码 (L29, L82, L106-107)
sys.path.insert(0, '/home/an/robot_world_ws/src/GalaxeaManipSim')
urdf_path = "/home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1_lite/robot.urdf"
ik_solver = RelaxedIKSolver(
    left_setting_file_path="/home/an/.../settings_left.yaml",
    right_setting_file_path="/home/an/.../settings_right.yaml",
)
```

**问题**: 路径硬编码到作者本地环境，无法在任何其他机器上运行。

**影响**: 可移植性为零。

#### 问题 4：缺少夹爪控制映射

原代码没有将人手手指捏合程度映射到 R1 的夹爪关节。这意味着视频中的 R1 夹爪始终处于默认状态，无法反映抓取动作。

#### 问题 5：缺少 warm_start 热启动

原始代码未使用 `SeqRetargeting.warm_start()`，第一帧直接从零位姿开始优化，可能导致：
- 第一帧 IK 求解不稳定
- 较大关节跳变

#### 问题 6：未使用 SeqRetargeting 的时序功能

- 直接调用优化器底层而绕过 `SeqRetargeting.retarget()`
- 无法利用低通滤波器平滑关节轨迹

#### 问题 7：缺少帧范围有效性检查

```python
num_frames = min(args.num_frames, total_frames - start_frame)
```
只做了 `min` 处理，未检查 `start_frame >= total_frames` 的情况。

#### 问题 8：调试代码与主逻辑耦合

大量 `if i == start_frame:` 调试打印与计算逻辑混在一起，降低可读性和可维护性。

### 6.3 修改方案与改进

#### 改进 1：正确解析手腕朝向

```python
def _extract_wrist_pose(self, hand_frame, joints):
    """从 MANO 前3维 compact axis-angle 计算真实手腕四元数"""
    global_axis_angle = hand_frame[0, :3].astype(np.float64)
    wrist_quat = pr.quaternion_from_compact_axis_angle(global_axis_angle)
    wrist_pos = joints[0, :3].copy()
    return wrist_pos, wrist_quat
```

使用 `pytransform3d.rotations.quaternion_from_compact_axis_angle()` 将 MANO 的轴角参数转换为标准四元数。

#### 改进 2：完整坐标变换链

```python
# MANO 朝向矩阵 × 算子矩阵 = 世界朝向
R_mano2world = pr.matrix_from_quaternion(wrist_quat) @ operator2mano.T
ee_pos = r1_base_offset + wrist_pos
ee_quat = pr.quaternion_from_matrix(R_mano2world)
```

三条变换链：
1. **平移**: `wrist_pos + r1_base_offset → ee_pos`
2. **旋转**: `quat → matrix → @operator2mano.T → quat`

#### 改进 3：路径自动发现

```python
def _resolve_galaxea_sim_path():
    """优先环境变量 GALAXEA_SIM_PATH，其次检查默认路径"""
    env = os.environ.get("GALAXEA_SIM_PATH")
    if env and Path(env).exists():
        return Path(env)
    for candidate in [DEFAULT_PATH_1, DEFAULT_PATH_2]:
        if candidate.exists():
            return candidate
    raise RuntimeError("无法找到 GalaxeaManipSim 路径。")
```

#### 改进 4：夹爪开合度映射

```python
FINGER_CLOSE_DIST = 0.02  # 人手捏合时拇指-食指距离
FINGER_OPEN_DIST = 0.08   # 人手完全张开时距离
GRIPPER_MAX_OPEN = 0.05   # R1 夹爪最大开度

normalized_grasp = np.clip(
    (finger_dist - FINGER_CLOSE_DIST) / (FINGER_OPEN_DIST - FINGER_CLOSE_DIST), 0, 1
)
gripper_open = normalized_grasp * GRIPPER_MAX_OPEN
```

#### 改进 5：类封装架构

将原来扁平的 `main()` 函数重构为 `R1TrackingPipeline` 类：
- `__init__()`: 配置管理
- `run()`: 主流程编排
- `_precompute()`: 帧预计算
- `_extract_wrist_pose()`: 手腕位姿解析
- `_render_video()`: 视频渲染

#### 改进 6：帧范围有效性检查

```python
actual_frames = min(num_frames, total_frames - start_frame)
if actual_frames <= 0:
    raise ValueError(f"帧范围无效: start={start_frame}, num={num_frames}, total={total_frames}")
```

### 6.4 修改后的完整代码

修改后的文件位于 [r1_hand_tracking_video.py](file:///home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting/r1_hand_tracking_video.py)，共 ~340 行，主要模块：

| 模块 | 行数 | 功能 |
|---|---|---|
| 常量定义 | L36-48 | 路径常量、手指映射参数 |
| `_resolve_galaxea_sim_path()` | L51-68 | GalaxeaManipSim 路径自动发现 |
| `R1TrackingPipeline.__init__()` | L74-90 | 管道配置初始化 |
| `R1TrackingPipeline.run()` | L92-181 | 6步主流程编排 |
| `R1TrackingPipeline._precompute()` | L183-264 | 逐帧 IK 预计算 |
| `R1TrackingPipeline._extract_wrist_pose()` | L266-277 | 手腕位姿解析 |
| `R1TrackingPipeline._render_video()` | L279-311 | 视频渲染 |
| `main()` | L314-341 | CLI 入口 |

### 6.5 使用方法

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting

# 基础用法
python r1_hand_tracking_video.py \
    --dexycb-dir /path/to/dexycb \
    --output-video output.mp4

# 完整参数
python r1_hand_tracking_video.py \
    --dexycb-dir /path/to/dexycb \
    --data-id 0 \
    --start-frame 0 \
    --num-frames 100 \
    --output-video r1_tracking.mp4 \
    --width 1280 \
    --height 720 \
    --fps 30

# 如 GalaxeaManipSim 不在默认路径，设置环境变量
export GALAXEA_SIM_PATH=/your/path/to/GalaxeaManipSim
```

### 6.6 预期成果

1. **姿态准确性**: R1 右臂末端能正确跟随人手手腕的位置和朝向变化
2. **抓取同步性**: R1 夹爪开合度与人手拇指-食指捏合程度实时对应
3. **平滑性**: 通过 IK 的 `tolerances` 参数和 RelaxedIK 内置的平滑机制保证轨迹连续
4. **可移植性**: 通过环境变量和自动发现机制支持多环境部署
5. **视频输出**: 生成 640×480（可调）mp4 格式视频，展示 R1 机器人追踪人手运动

---

## 七、技术架构总结

### 整体数据流

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
│  DexYCB     │───▶│  Dex Retargeting │───▶│  RelaxedIK  │
│  人手姿态    │    │  PositionOptimizer│    │  R1 臂 IK   │
└─────────────┘    └──────────────────┘    └─────────────┘
                          │                       │
                          ▼                       ▼
                   ┌──────────────┐        ┌───────────┐
                   │  夹爪映射     │        │ SAPIEN    │
                   │  手指→开合度  │        │ 渲染输出   │
                   └──────────────┘        └───────────┘
                                                 │
                                                 ▼
                                            ┌────────┐
                                            │  .mp4  │
                                            │  视频  │
                                            └────────┘
```

### 关键设计模式

1. **策略模式**: `Optimizer` 基类 + 三种具体实现，通过配置文件选择
2. **适配器模式**: `MimicJointKinematicAdaptor` 适配 mimic 关节约束
3. **过滤器模式**: `LPFilter` 实现关节角度时间平滑
4. **数据类 + 工厂方法**: `RetargetingConfig` 用 Python dataclass 统一管理配置

### 依赖关系图

```
retargeting_config.py ──▶ yourdfpy.py (URDF解析)
     │                        │
     ▼                        ▼
optimizer.py ◀── robot_wrapper.py (Pinocchio)
     │               │
     ▼               ▼
seq_retarget.py ◀── kinematics_adaptor.py
     │
     ▼
optimizer_utils.py (LPFilter)
```

---

## 八、总结

`dex-retargeting` 是一个设计精良的灵巧手重定向工具库，具有以下特点：

**优势**:
- 支持 10 种灵巧手/夹爪，6 种机械臂，覆盖主流机器人平台
- 三种重定向算法满足不同场景（离线数据处理、实时遥操作、带抓取先验的遥操作）
- 基于 Pinocchio + NLopt 的运动学优化，支持 mimic 关节和时间平滑
- 配置文件驱动，易于扩展新机器人
- 与 DexYCB 数据集无缝集成

**`r1_hand_tracking_video.py` 改进要点**:
- 从恒等四元数改为真实的 MANO 手腕朝向解析
- 从简单偏移改为完整的坐标变换链
- 增加夹爪开合度映射
- 消除硬编码路径，支持环境变量
- 封装为管道类，提高可维护性

---

*文档生成日期: 2026-05-08*
