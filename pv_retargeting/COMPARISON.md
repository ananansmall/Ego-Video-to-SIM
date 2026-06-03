# R1 手部追踪管线对比文档

## `r1_hand_tracking_video.py` vs `r1_single_arm_follow.py`

---

## 0. DexYCB 数据集介绍

### 0.1 数据集概述

**DexYCB** (Dexterous YCB) 是一个大规模手-物体交互数据集，由 NVIDIA 收集发布。它记录了 10 个受试者抓取 21 种 YCB 物体的过程，使用 8 个 RGB-D 相机从不同角度拍摄。

| 属性 | 值 |
|------|-----|
| 受试者 | 10 人 |
| YCB 物体 | 21 种 (每次抓取 1-5 种) |
| 相机 | 8 个 RealSense (RGB-D) |
| 手部模型 | MANO (参数化) |
| 坐标系 | 相机坐标系 (需外参变换到世界坐标系) |

### 0.2 数据集目录结构

```
dex-ycb/                              # 数据集根目录
├── 20200709-subject-01/              # 10个受试者目录
├── 20200813-subject-02/
├── ...
├── 20201022-subject-10/
│   └── <capture_name>/               # 每个受试者下有多个capture目录
│       ├── meta.yml                  # 该capture的元数据
│       └── pose.npz                  # 该capture的姿态数据 (核心!)
├── calibration/                      # 标定数据
│   ├── extrinsics_<name>/            # 外参标定目录
│   │   └── extrinsics.yml            # 外参 3x4 矩阵
│   ├── intrinsics/                   # 内参标定目录
│   │   └── <serial>_color.yml        # 每个相机的内参 (fx, fy, ppx, ppy)
│   └── mano_<name>/                  # MANO手部标定目录
│       └── mano.yml                  # MANO形状参数 betas[10]
└── models/                           # YCB物体3D模型
    ├── 002_master_chef_can/
    │   └── textured_simple.obj
    ├── 003_cracker_box/
    └── ...                           # 共21种YCB物体
```

### 0.3 核心数据字段

每个 capture 的 `pose.npz` 包含两个关键数组：

| 键名 | Shape | 说明 |
|------|-------|------|
| `pose_m` | `[N_frames, 51]` | 手部 MANO 参数 (每帧) |
| `pose_y` | `[N_frames, N_objects, 7]` | 物体位姿 (每帧每物体) |

### 0.4 MANO 参数详解 (hand_pose 51维)

```
hand_pose [51维]:
  [0:3]   → 手腕 compact axis-angle 旋转 (3维)
  [3:48]  → 手指 PCA 系数 (15关节 × 3 = 45维)
  [48:51] → 手腕 3D 平移 (3维, 相机坐标系, 毫米→米)

MANO FK 前向传播:
  p = hand_pose[:, :48]    # 48维姿态参数
  t = hand_pose[:, 48:51]  # 3维平移
  vertex, joints = MANOLayer(p, t)  # → vertex[778,3], joints[21,3]
  # 输出在相机坐标系，单位米
```

**为什么需要 MANO FK？**

DexYCB **不直接提供** 3D 关节位置，只提供 MANO 的隐式参数 (axis-angle + PCA + translation)。必须通过 MANO FK 才能得到显式的 21 个 3D 关节位置和 778 个手部网格顶点。这是参数化手部模型的标准做法——51 维参数比 21×3=63 维关节位置更紧凑，且保证手部几何一致性。

### 0.5 MANO 21 关节索引

```
索引  关节名           手指
 0   wrist            手腕
 1   thumb_mcp        拇指掌指关节
 2   thumb_pip        拇指近端指间关节
 3   thumb_dip        拇指远端指间关节
 4   thumb_tip        拇指尖 ★ (retargeting约束点)
 5   index_mcp        食指掌指关节
 6   index_pip        食指近端指间关节
 7   index_dip        食指远端指间关节
 8   index_tip        食指尖 ★ (retargeting约束点)
 9   middle_mcp       中指掌指关节
10   middle_pip       中指近端指间关节
11   middle_dip       中指远端指间关节
12   middle_tip       中指尖
13   ring_mcp         无名指掌指关节
14   ring_pip         无名指近端指间关节
15   ring_dip         无名指远端指间关节
16   ring_tip         无名指尖
17   little_mcp       小指掌指关节
18   little_pip       小指近端指间关节
19   little_dip       小指远端指间关节
20   little_tip       小指尖
```

### 0.6 物体位姿格式

```python
object_pose[frame, obj_idx] = [qx, qy, qz, qw, tx, ty, tz]
# 前4维: 四元数 (xyzw格式)
# 后3维: 平移 (相机坐标系, 米)
# 使用时需转换:
pose = camera_pose * sapien.Pose(
    pos_quat[4:],                                    # [tx, ty, tz]
    np.concatenate([pos_quat[3:4], pos_quat[:3]])    # xyzw → wxyz
)
```

### 0.7 相机外参/内参

**外参** (`calibration/extrinsics_<name>/extrinsics.yml`):
```yaml
extrinsics:
  apriltag: [12个浮点数]   # 3x4 变换矩阵展平 [R|t]
```
- 3x4 矩阵补齐为 4x4 齐次矩阵
- 表示**相机坐标系 → 世界坐标系**的变换
- 使用时取逆得到 `camera_pose` (世界→相机)

**内参** (`calibration/intrinsics/<serial>_color.yml`):
```yaml
color:
  fx: 613.39    # X方向焦距 (像素)
  fy: 613.39    # Y方向焦距 (像素)
  ppx: 312.67   # X方向主点 (像素)
  ppy: 241.49   # Y方向主点 (像素)
```

### 0.8 meta.yml 结构

```yaml
mano_sides: ["right"]           # 左/右手 (用于过滤和自动检测)
ycb_ids: [1, 5, 3]             # 该capture中的YCB物体ID列表
extrinsics: "836212060125"      # 对应的外参标定名称 (相机序列号)
mano_calib: ["subject-01"]      # 对应的MANO标定名称 (取[0]获取betas)
```

### 0.9 数据流全景

```
DexYCB 数据集
    │
    ├── meta.yml ──→ mano_sides (左/右手过滤)
    │            ──→ ycb_ids (物体ID列表)
    │            ──→ extrinsics (外参名称) ──→ calibration/extrinsics_<name>/extrinsics.yml
    │            ──→ mano_calib (MANO标定名) ──→ calibration/mano_<name>/mano.yml → betas[10]
    │
    ├── pose.npz ──→ pose_m: hand_pose [N, 51]  (3手腕aa + 45PCA + 3平移)
    │            ──→ pose_y: object_pose [N, M, 7]  (4四元数xyzw + 3平移)
    │
    └── MANO FK: hand_pose[:,:48] + hand_pose[:,48:51] + betas
                  → MANOLayer.forward(p, t) → vertex[778,3] + joints[21,3] (相机坐标系)
                  → 坐标变换: joints_cam @ R^T + t → joints_world (世界坐标系)
```

---

## 1. 调用方式

### r1_hand_tracking_video.py（离线视频生成）

```bash
# 基本用法
python r1_hand_tracking_video.py --dexycb-dir /path/to/dex-ycb
# 指定数据ID和帧数
python r1_hand_tracking_video.py --dexycb-dir /path/to/dex-ycb --data-id 4 --num-frames 100
# 指定视角
python r1_hand_tracking_video.py --dexycb-dir /path/to/dex-ycb --view front
# 完整参数
python r1_hand_tracking_video.py \
    --dexycb-dir /path/to/dex-ycb \
    --data-id 4 \
    --start-frame 0 \
    --num-frames 50 \
    --output-video r1_tracking.mp4 \
    --fps 30 \
    --view behind
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dexycb-dir` | `str` | **必需** | DexYCB 数据集根目录 |
| `--data-id` | `int` | `0` | DexYCB 数据索引 |
| `--start-frame` | `int` | `0` | 起始帧 |
| `--num-frames` | `int` | `50` | 处理的帧数 |
| `--output-video` | `str` | `r1_tracking.mp4` | 输出视频路径 |
| `--fps` | `int` | `30` | 视频帧率 |
| `--view` | `str` | `behind` | 视角: `behind` / `front` / `topdown` |

### r1_single_arm_follow.py（交互式 Viewer 渲染）

```bash
# 基本用法（默认 data_id=4）
python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb
# 指定数据ID和帧数
python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb --data-id 0 --num-frames 80
# 循环播放
python r1_single_arm_follow.py --dexycb-dir /path/to/dex-ycb --loop
# 完整参数
python r1_single_arm_follow.py \
    --dexycb-dir /path/to/dex-ycb \
    --data-id 4 \
    --start-frame 0 \
    --num-frames 50 \
    --output-video r1_arm_follow.mp4
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dexycb-dir` | `str` | **必需** | DexYCB 数据集根目录 |
| `--data-id` | `int` | `4` | DexYCB 数据索引 |
| `--start-frame` | `int` | `0` | 起始帧 |
| `--num-frames` | `int` | `50` | 处理的帧数 |
| `--output-video` | `str` | `r1_arm_follow.mp4` | 输出视频路径 |
| `--loop` | `flag` | `False` | 循环播放（交互模式） |

---

## 2. 整体映射流程（核心管线）

### r1_hand_tracking_video.py 管线

```
DexYCB 数据集
    │
    ▼
[1] 加载数据: hand_pose (MANO参数), object_pose, camera_pose
    │
    ▼
[2] RobotHandDatasetSAPIENViewer 加载 R1 完整双臂 + 手部mesh + YCB物体
    │  retargeting_overrides → 3约束点配置
    │
    ▼
[3] viewer._compute_hand_geometry(hand_frame)
    │  MANO FK → vertex, joints (世界坐标系)
    │
    ▼
[4] warm_start: wrist_quat → 初始化优化器dummy关节
    │
    ▼
[5] Dex Retargeting (NLopt SLSQP):
    │  ref_value = joints[ref_indices]  (ref_indices=[4,8,0] → 拇指尖+食指尖+手腕)
    │  retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
    │  → 9维: 6 dummy自由关节 + 2 夹爪关节 + 1 mimic关节
    │
    ▼
[6] 优化器FK提取位姿:
    │  internal_robot.compute_forward_kinematics(retarget_qpos)
    │  gripper_pos_fk = get_link_pose(gripper_link_idx)[:3, 3]  ← 位置
    │  R_ee_world_fk = get_link_pose(gripper_link_idx)[:3, :3]  ← 朝向
    │
    ▼
[7] 工作空间映射:
    │  ik_target_world = gripper_pos_fk + mapping_offset + safety_offset
    │  mapping_offset = comfort_target_world - wrist_centroid
    │  safety_offset = approach_dir × 0.075m
    │
    ▼
[8] 坐标变换 (世界帧 → base_link帧):
    │  ik_target_base = base_link_R_inv @ (ik_target_world - base_link_p)
    │  R_ee_base = base_link_R_inv @ R_ee_world_fk
    │  ee_quat_base = quaternion_from_matrix(R_ee_base)
    │
    ▼
[9] RelaxedIK 求解 (容差 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    │  right_joints = ik_solver.solve_position_right(ik_target_base, ee_quat_base)
    │  → 6个臂关节角
    │
    ▼
[10] 组装qpos: right_arm_joints + gripper_joints → 26+维完整qpos
    │
    ▼
[11] 轨迹平滑: LPFilter + TrajectorySmoother (Butterworth + 速度/加速度/加加速度限幅)
    │
    ▼
[12] 离线渲染: scene.step() + camera.take_picture() → VideoWriter → MP4
```

### r1_single_arm_follow.py 管线

```
DexYCB 数据集
    │
    ▼
[1] 加载数据: hand_pose, object_pose, camera_mat (手动坐标变换)
    │  自动检测左右手: _detect_hand_type_from_dataset → meta.yml mano_sides
    │
    ▼
[2] 自建SAPIEN场景: 加载单臂URDF (8关节) + 手部mesh + 双份YCB物体
    │  retargeting override → 3约束点 + normal_delta=1e-5 + huber_delta=0.01
    │
    ▼
[3] MANOLayer FK + 手动坐标变换:
    │  _, joint = mano_layer(p, t)
    │  joint_world = joint @ camera_mat[:3, :3].T + camera_mat[:3, 3]
    │
    ▼
[4] warm_start (修正坐标系):
    │  wrist_R_world = camera_mat[:3, :3] @ matrix_from_compact_axis_angle(hp[:3])
    │  wrist_quat_world = quaternion_from_matrix(wrist_R_world)
    │  retargeting.warm_start(j0[0,:], wrist_quat_world, ...)
    │
    ▼
[5] Dex Retargeting (NLopt SLSQP):
    │  ref_value = joint_world[ref_indices]  (ref_indices=[4,8,0])
    │  retarget_qpos = retargeting.retarget(ref_value, fixed_qpos)
    │
    ▼
[6] 优化器FK提取位姿:
    │  internal_robot.compute_forward_kinematics(retarget_qpos)
    │  gripper_pos_fk, gripper_R_fk = _get_gripper_pose_from_retargeting(retarget_qpos)
    │
    ▼
[7] 工作空间映射:
    │  ik_target_raw = gripper_pos_fk + mapping_offset
    │  (无 safety_offset)
    │
    ▼
[8] 坐标变换 (世界帧 → base_link帧):
    │  ik_target_base = base_link_R_inv @ (ik_target_raw - base_link_p)
    │  ee_R_base = base_link_R_inv @ gripper_R_fk
    │
    ▼
[9] RelaxedIK 求解 (容差 [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
    │  arm_joints = ik_solver.solve_position_right(ik_target_base, ee_quat_base)
    │  → 6个臂关节角
    │
    ▼
[10] 组装qpos: arm_joints + gripper_joints → 8维单臂qpos
    │
    ▼
[11] 轨迹平滑: LPFilter + TrajectorySmoother
    │
    ▼
[12] 交互式渲染: scene.update_render() + viewer.render() (无scene.step())
    │  可选 --loop 循环播放
```

---

## 3. 总体架构对比

| 维度 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| **机器人模型** | R1 完整双臂机器人 (26+关节) | R1 单臂 (6臂关节 + 2夹爪 = 8关节) |
| **URDF来源** | `RobotHandDatasetSAPIENViewer` 内部加载 | 自定义 `_prepare_arm_urdf()` 裁剪单臂URDF |
| **场景管理** | 委托给 `RobotHandDatasetSAPIENViewer` | 自建 SAPIEN 场景，独立管理 |
| **手部类型** | 硬编码右手 | 自动检测 (`_detect_hand_type_from_dataset`) |
| **左臂处理** | 设置左臂为自然下垂姿态 | 不加载左臂 |
| **类名** | `R1TrackingPipeline` | `R1SingleArmFollower` |

---

## 4. 映射核心对比

### 4.0 Dummy 关节与正则化（通俗解释）

**Dummy 关节是什么？**

Dummy 关节是 dex-retargeting 在机器人 URDF 根部**凭空添加**的 6 个虚拟关节（3个平移 + 3个旋转），让机器人可以"自由飞行"。它们**只存在于优化器内部**（Pinocchio 数学模型），**不存在于 SAPIEN 物理场景中**。

```
比喻：
  官方示例（灵巧手）→ 夹爪装在"万向云台"上，云台直接控制位置+朝向
  R1 机械臂         → 夹爪焊死在机械臂末端，必须通过6个臂关节IK求解

官方示例: set_qpos([tx,ty,tz,rx,ry,rz, gripper1,gripper2]) → 机器人飞到位
R1:      Dummy被丢弃 → 从优化器FK提取位姿 → RelaxedIK → 6个臂关节角
```

**正则化（norm_delta）是什么？**

正则化就像一根**弹簧**，把当前帧的关节角拉向上一帧的关节角，防止帧间抖动。

```
总损失 = 位置匹配损失 + 2 × norm_delta × ||当前关节角 - 上一帧关节角||²
                                ↑ 这就是"弹簧"
```

- 弹簧太强（norm_delta=4e-3）→ Dummy旋转被锁死在上一帧 → 朝向不变
- 弹簧太弱（norm_delta=1e-5）→ 朝向可以自由变化 → 但可能抖动

**为什么2约束点时正则化会锁死朝向？**

2约束点（拇指尖+食指尖）只提供6个位置约束，但优化器有7个自由度（6个Dummy+1个夹爪）。多出的1个自由度（绕接近轴旋转）对位置匹配没有影响，所以**弹簧是唯一的作用力**，把朝向锁死。

### 4.1 Retargeting 配置 (3约束点)

两个文件现在都使用**3个约束点**的 retargeting 配置：

```yaml
# r1_full_right.yml (override)
target_link_names:
  - right_gripper_finger_link1    # 夹爪手指1
  - right_gripper_finger_link2    # 夹爪手指2
  - right_gripper_link            # 夹爪基座 (新增!)
target_link_human_indices: [4, 8, 0]  # 拇指尖 + 食指尖 + 手腕
normal_delta: 1e-5    # 正则化 (原4e-3, 降低400倍)
huber_delta: 0.01     # Huber损失阈值
```

**为什么需要第3个约束点？**

| 配置 | 约束数 | 优化DOF | 状态 | 朝向约束 |
|------|--------|---------|------|---------|
| 旧: [4, 8] | 2×3=6 | 7 (6dummy+1gripper) | 欠定1DOF | 接近轴旋转无梯度 |
| 新: [4, 8, 0] | 3×3=9 | 7 | 超定2约束 | 手腕位置→自然约束朝向 |

### 4.2 IK 目标位置

| 维度 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| **位置来源** | `gripper_pos_fk + mapping_offset + safety_offset` | `gripper_pos_fk + mapping_offset` |
| **safety_offset** | 有 (0.075m) | 无 |
| **gripper_pos_fk** | 优化器FK的gripper_link位置 | 优化器FK的gripper_link位置 |

两者都使用优化器FK的gripper位置作为IK目标位置（而非手腕位置）。

### 4.3 IK 目标朝向

| 维度 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| **朝向来源** | 优化器FK的gripper_link朝向 | 优化器FK的gripper_link朝向 |
| **提取方式** | `internal_robot.get_link_pose(gripper_link_idx)[:3,:3]` | `_get_gripper_pose_from_retargeting()` |
| **坐标系变换** | `base_link_R_inv @ R_ee_world_fk` | `base_link_R_inv @ gripper_R_fk` |

两者都使用优化器FK的gripper朝向。3约束点配置使优化器自然约束朝向（手腕位置相对于指尖的方向编码了朝向信息）。

### 4.4 RelaxedIK 容差

两个文件使用**完全相同**的容差配置：

```python
tolerances=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
# 位置容差 0.1m (10cm) → 位置权重 = 1/0.1² = 100
# 朝向容差 0.1rad (≈5.7°) → 朝向权重 = 1/0.1² = 100
# 位置权重 = 朝向权重 → 位置和朝向同等重要
```

**对比旧配置**:

| 参数 | 旧值 | 新值 | 变化 |
|------|------|------|------|
| 位置容差 | [0.001, 0.001, 0.001] | [0.1, 0.1, 0.1] | 放松100倍 |
| 朝向容差 | [10.0, 10.0, 10.0] | [0.1, 0.1, 0.1] | **收紧100倍** |

旧配置位置权重是朝向的1亿倍（位置极紧，朝向极松）。新配置位置和朝向权重相等，IK求解器同时兼顾位置和朝向。

### 4.5 warm_start 初始化

| 维度 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| **是否调用** | 是 (新增) | 是 |
| **手腕四元数** | 相机坐标系 (viewer._compute_hand_geometry) | 世界坐标系 (手动变换) |
| **坐标系** | `quaternion_from_compact_axis_angle(hand_frame[0,:3])` | `camera_mat[:3,:3] @ matrix_from_compact_axis_angle(hp[:3])` |

### 4.6 工作空间映射参数

| 参数 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| COMFORTABLE_REACH | 0.40m | 0.30m |
| COMFORT_TARGET_IN_BASE | [0.30, 0.0, -0.25] | [0.30, 0.0, -0.30] |
| SAFETY_DISTANCE | 0.075m | 0.05m (但未使用) |
| ARM_MAX_REACH | 0.713m | 0.713m |

---

## 5. 关节调用对比

### 5.1 关节索引获取

**r1_hand_tracking_video** (完整机器人):
```python
active_joints = r1_robot.get_active_joints()
joint_names = [j.get_name() for j in active_joints]
right_arm_indices = [i for i, name in enumerate(joint_names) if "right_arm" in name]
gripper_idx1 = joint_names.index("right_gripper_finger_joint1")
gripper_idx2 = joint_names.index("right_gripper_finger_joint2")
```

**r1_single_arm_follow** (单臂URDF):
```python
self.arm_joint_indices = [0, 1, 2, 3, 4, 5]  # 6个臂关节
self.gripper_idx1 = 6  # right_gripper_finger_joint1
self.gripper_idx2 = 7  # right_gripper_finger_joint2
```

### 5.2 retarget2sapien 映射

**r1_hand_tracking_video**:
```python
retarget2sapien = viewer.retarget2sapien[0]
# sapien_qpos = retarget_qpos[retarget2sapien]  → 26维
```

**r1_single_arm_follow**:
```python
self.retarget2sapien = np.array(
    [retarget_joint_names.index(n) for n in sapien_joint_names if n in retarget_joint_names]
).astype(int)
# sapien_qpos = retarget_qpos[retarget2sapien]  → 8维
```

### 5.3 qpos 组装

**r1_hand_tracking_video**:
```python
r1_qpos = r1_robot.get_qpos().copy()
for j, idx in enumerate(right_arm_indices):
    r1_qpos[idx] = right_joints[j]
r1_qpos[gripper_idx1] = gripper1
r1_qpos[gripper_idx2] = gripper2
```

**r1_single_arm_follow**:
```python
qpos = self.robot.get_qpos().copy()
for j, idx in enumerate(self.arm_joint_indices):
    qpos[idx] = arm_joints[j]
qpos[self.gripper_idx1] = gripper1
qpos[self.gripper_idx2] = gripper2
```

---

## 6. 展示/渲染对比

### 6.1 渲染方式

| 维度 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| **渲染引擎** | SAPIEN 离线相机 | SAPIEN Viewer 交互式 |
| **物理步进** | 每帧 `scene.step()` | 仅 `scene.update_render()` |
| **视角** | 3种预设 (`--view`) | 固定 `(1.5, 0, 1)` |
| **视频输出** | OpenCV VideoWriter | Viewer 交互 / 离线可选 |
| **循环播放** | 不支持 | `--loop` 参数 |

### 6.2 可视化内容

**r1_hand_tracking_video** (2D 图像标注):
- OpenCV 2D 绘制坐标轴（相机投影）
- 轨迹线（2D 投影）
- 数值面板（EE位置、误差、可达性）
- Warmup 进度条
- **MANO参考点标注**: 拇指尖(4)红色圆 + 食指尖(8)蓝色圆

**r1_single_arm_follow** (3D 场景内可视化):
- SAPIEN 3D 坐标轴（capsule 绘制 RGB 轴）
- 3D 轨迹线
- 人手 mesh + YCB 物体（两份：原始 + 偏移）
- **MANO参考点标注**: 拇指尖(4)红色球 + 食指尖(8)蓝色球

### 6.3 物体展示

| 维度 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| YCB物体 | 单份 (和人手在一起) | 双份 (原始 + mapping_offset偏移) |
| 人手mesh | `viewer._update_hand(vertex)` | 自建context渲染 |

---

## 7. 核心差异总结

### r1_hand_tracking_video
1. **完整机器人**：R1 双臂，左臂自然下垂
2. **依赖 viewer**：场景/人手/物体全委托 `RobotHandDatasetSAPIENViewer`
3. **离线渲染**：`scene.step()` + 离线相机 → MP4
4. **2D标注**：OpenCV 绘制坐标轴/轨迹/数值面板
5. **3种视角**：behind/front/topdown

### r1_single_arm_follow
1. **单臂模型**：只加载右臂 URDF，轻量高效
2. **自建场景**：独立管理 SAPIEN 场景
3. **交互式渲染**：`scene.update_render()` + viewer，无物理步进
4. **3D可视化**：SAPIEN 内绘制坐标轴/轨迹/球体
5. **自动检测左右手**：`meta.yml` → `mano_sides`
6. **双份物体**：原始 + 偏移，同时展示人手和机械臂
7. **循环播放**：`--loop` 参数
8. **修正坐标系**：`warm_start` 中 `wrist_quat` 正确转换到世界坐标系

### 精度对比

| 指标 | r1_hand_tracking_video | r1_single_arm_follow |
|------|----------------------|---------------------|
| IK目标位置 | 优化器FK gripper位置 + safety_offset | 优化器FK gripper位置 |
| IK目标朝向 | 优化器FK gripper朝向 (3约束点) | 优化器FK gripper朝向 (3约束点) |
| IK容差 | [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] | [0.1, 0.1, 0.1, 0.1, 0.1, 0.1] |
| 位置精度 | FK误差 ~5cm | FK误差 ~3-5cm |
| 朝向精度 | 由3约束点+IK容差共同保证 | 由3约束点+IK容差共同保证 |

**重要**: IK目标位姿**不是直接从MANO读取**的。MANO FK只提供21个关节3D位置作为retargeting的输入参考点，最终的位姿是retargeting优化器经过NLopt SLSQP优化后，通过内部FK计算出来的gripper_link位姿。完整链路：

```
MANO参数 → MANO FK → 21个关节3D位置(参考点)
                              ↓
              Dex Retargeting (NLopt SLSQP优化)
              输入: joints[4,8,0] (拇指尖+食指尖+手腕)
              输出: retarget_qpos (6dummy+2gripper+1mimic)
                              ↓
              优化器内部FK → gripper_link位姿 (位置+朝向)
                              ↓
              + mapping_offset → IK目标 → RelaxedIK → 6个臂关节角
```
