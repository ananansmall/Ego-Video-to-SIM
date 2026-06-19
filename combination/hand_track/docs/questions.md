# hand_track 项目问答记录

## Q1: 这个能够实现物理仿真吗？

**结论: 可以实现，且项目中已有可用的物理仿真管线。**

### 现状

项目在 `example/combination/physics_pipeline/` 目录下已经实现了两条物理仿真管线，均已通过测试：

| 管线 | 引擎 | GPU 需求 | 状态 |
|---|---|---|---|
| `pybullet_pipeline.py` | PyBullet (Bullet) | 无 (CPU only) | ✓ 全部测试通过 |
| `run_physics_pipeline.py` + `rerender.sh` | SAPIEN (PhysX) | 需要 GPU | ✓ 可用 |

### PyBullet 管线测试结果 (已通过)

```
hold_position:           PASS  (EE drift=0.0mm)
move_to_target:          PASS  (EE error=0.0mm)
arm_object_interaction:  PASS  (机械臂可推动物体)
glb_stability:           PASS  (max displacement=4.9mm)
render_video:            PASS  (113帧, 1280x720@30fps, 1.0MB)
```

### 物理仿真做了什么

让 `02_render_scene.py` 的纯运动学渲染具有物理属性：
- 物体从 kinematic 变为 dynamic (可被抓取、推动)
- 机器人关节从纯 `set_qpos` 变为 PD 驱动 + 重力补偿 (SAPIEN) 或 `resetJointState` 运动学控制 (PyBullet)
- 添加地面支撑，物体自然放置在桌面上

### 关键设计决策

1. **运动学控制策略**: R1 URDF 连杆惯性极小 (~1E-4 kg·m²)，PyBullet 的 POSITION_CONTROL 和 TORQUE_CONTROL 均无法稳定控制。采用 `resetJointState` 每步重置关节位置，物理引擎只处理 GLB 物体交互。
2. **地面高度自适应**: 自动计算 GLB 物体最低 Z 坐标，将地面放在物体下方 1cm 处。
3. **物体分类**: 大型扁平几何体 (桌面/地板) → static (mass=0)，小物体 → dynamic (mass=volume·density)。

### 调用方式

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/physics_pipeline

# PyBullet (CPU only, 无 GPU 依赖)
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --test
/home/an/miniconda3/envs/dex/bin/python pybullet_pipeline.py --render-video

# SAPIEN (需要 GPU)
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
bash physics_pipeline/rerender.sh render
```

### 局限性

- PyBullet 管线渲染质量一般 (CPU 光栅化)，但稳定且无 GPU 依赖
- SAPIEN 管线渲染质量高，但需要 GPU 且 NVIDIA 驱动需匹配
- 当前物理仿真主要针对单臂 + 物体交互，双手协同抓取等复杂场景需进一步开发

---

## Q2: 夹爪为什么之前对应不上手？

**根因: 夹爪朝向用错了坐标系。**

### 问题分析

- `dex-retargeting` 的 `RetargetingType.position` 模式只保证目标连杆 (gripper_link, finger_link1/2) 的**位置**与 MANO 关节对齐
- 之前代码用 MANO 手腕朝向 (`wrist_R_sapien`) 设置夹爪 root_pose, 但 MANO 手腕坐标系 (Z 轴指向手指) 与 R1 夹爪坐标系 (X 轴指向手指) 定义不同, 导致夹爪朝向仍然对应不上手

### 修复方案 (与 02_render_scene.py 一致)

夹爪位姿 (位置 + 朝向) 都用 retargeting FK 给出的 `gripper_pos_fk` 和 `gripper_R_fk`:
- **位置**: `gripper_pos_fk` (retargeting FK 给出的 gripper_link 位置)
- **朝向**: `gripper_R_fk` (retargeting FK 给出的 gripper_link 旋转, 即 `02_render_scene.py` 中的 `R_ee_world_fk`)

参考实现: `02_render_scene.py` 的 `run_robot_tracking` 函数, IK 目标朝向就是 `R_ee_world_fk`。

### gripper_arm 模式的 offset 补偿

`gripper_arm` 模式下 `robot.set_root_pose` 设置的是 root link (arm_base_link) 位姿, 但 `gripper_pos_fk` 是 gripper_link 的位置。需要补偿 offset:

```python
# 计算 gripper_link 相对于 root 的 offset (qpos=0 时)
offset_pos, offset_R = _compute_gripper_offset_in_root(robot, prefix)

# 设置 root pose 时补偿 offset
root_R   = gripper_R_fk @ offset_R.T
root_pos = gripper_pos_fk - root_R @ offset_pos
robot.set_root_pose(sapien.Pose(root_pos, root_quat))
```

数学推导:
```
gripper_world_pos = root_pos + root_R @ offset_pos
gripper_world_R   = root_R @ offset_R

已知 gripper_world_pos = gripper_pos_fk, gripper_world_R = gripper_R_fk:
=> root_R   = gripper_R_fk @ offset_R^T
=> root_pos = gripper_pos_fk - root_R @ offset_pos
```

---

## Q3: 双手 MANO 关键点为什么只显示一只手？

**根因: `_render_keypoints` 函数的清除逻辑 bug。**

### 问题分析

`_render_keypoints` 函数开头会执行 `kp_nodes.clear()`，在双手循环中：
1. 左手调用 → 渲染左手关键点
2. 右手调用 → 清除所有关键点 (包括左手) → 只渲染右手关键点

### 修复方案

为 `_render_keypoints` 添加 `clear_existing` 参数：
- 左手: `clear_existing=True` (清除上一帧的关键点，重新渲染左手)
- 右手: `clear_existing=False` (不清除，累加右手关键点)

---

## Q4: 能不能只展示机械臂前面几个关节 (夹爪+连接的手臂)？

**可以，已实现 `gripper_arm` 模式 (arm_link4/5/6 + 夹爪)。**

### 实现

新增 `_GRIPPER_WITH_ARM_URDF_TEMPLATE` URDF 模板，包含：
- `arm_base_link` (固定根, 代表 arm_link3 的位置)
- `arm_link4` (revolute joint, origin=`0.02735 -0.069767 0`, axis=`1 0 0`)
- `arm_link5` (revolute joint, origin=`0.2463 0.00050106 0`, axis=`0 -1 0`)
- `arm_link6` (revolute joint, origin=`0.058249 -0.00049975 0`, axis=`1 0 0`)
- `gripper_link` (fixed joint, origin=`0.1039 0 0`)
- `gripper_finger_link1/2` (两个手指, prismatic joint)

URDF 数据来源: R1 URDF (`r1_v2_1_0_floating_right.urdf`) 中 arm_joint4/5/6 的 origin 和 axis。

### 为什么选 arm_link4/5/6?

- 比纯夹爪更生动 (能看到连接的手臂段)
- 排除手臂底座 (arm_link1/2/3) 的不确定性
- arm_link4/5/6 是手腕附近的三个关节, 视觉上更像"夹爪+连接的手臂"

### offset 补偿

由于 `gripper_arm` 模式下 `robot.set_root_pose` 设置的是 root link (arm_base_link) 位姿, 但 retargeting FK 给出的是 gripper_link 的位姿, 需要补偿 offset (详见 Q2)。

### 用法

```bash
# 仅夹爪
python hand_track/render_gripper_only.py --mode gripper ...

# 夹爪 + 手臂末端 (arm_link4/5/6)
python hand_track/render_gripper_only.py --mode gripper_arm ...

# 通过管线入口
python hand_track/render_auto.py --mode gripper_arm ...
```

---

## Q5: 映射平滑性不够, 夹爪张不开, 和 02_render_scene.py 的 run_robot_tracking 有什么区别?

**已修复: 新增解析模式 (analytical mode) + MANO 输入位置 EMA 平滑。**

### 问题根因

1. **夹爪张不开**: 优化器模式 (NLopt SLSQP) 在左手数据上陷入局部最优, 指尖误差 ~38mm, 导致手指关节值不正确
2. **平滑性不够**: 之前对输出 root pose 做 EMA 平滑, 但手指关节不平滑, 造成 root pose 和手指关节不一致

### 与 02_render_scene.py 的区别

| 方面 | 02_render_scene.py (run_robot_tracking) | render_gripper_only.py (旧, 优化器模式) | render_gripper_only.py (新, 解析模式) |
|---|---|---|---|
| 夹爪位姿来源 | retargeting FK (优化器) | retargeting FK (优化器) | 解析计算 (从 MANO 指尖向量) |
| 手指关节 | retargeting FK | retargeting FK | 解析计算 (`(finger_dist - base_dist) / 2`) |
| 平滑 | 无 | root pose EMA (输出平滑) | MANO 输入位置 EMA (输入平滑) |
| 指尖误差 | 取决于优化器 | 左手 ~38mm (局部最优) | < 1.5mm |

### 解析模式原理

从 MANO 指尖向量直接计算夹爪 root 位姿:
1. **Y轴**: finger1→finger2 方向 (对应机器人 (0,1,0))
2. **X轴**: wrist→finger_mid 方向 (对应机器人 (1,0,0))
3. **Z轴**: X×Y (Gram-Schmidt 正交化)
4. **手指关节**: `joint = (finger_dist - 0.026906) / 2`, clamp 到 [0, 0.05]
5. **root_pos**: `mano_finger1 - R @ (finger1_offset)`

### 平滑策略

- **解析模式** (alpha=0.9): 对 MANO 输入位置 (wrist, finger1, finger2) 做 EMA, 保持 root pose 和手指关节一致性
- **优化器模式** (alpha=0.6): 对输出 root pose 做 EMA (位置 + 朝向)
- MANO 数据本身来自神经网络, 已经比较平滑, 只需轻微平滑

### 验证结果

```
gripper 模式:    left 1.31/1.38mm, right 0.36/0.40mm
gripper_arm 模式: left 1.31/1.38mm, right 0.36/0.40mm
```
所有指尖误差 < 1.5mm, 远小于 2% 要求。

---

## Q6: 能不能像 00_run_pipeline.py 那样有个 -View 模块, 循环播放?

**已实现: `--viewer` 参数, SAPIEN Viewer 实时循环播放。**

### 用法

```bash
python hand_track/render_gripper_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --viewer
```

### 行为

- 在 SAPIEN Viewer 窗口中实时渲染, 不保存视频文件
- 动画播放完后自动重置 qpos 和平滑器, 重新开始循环
- 关闭窗口退出
- 支持 FPV/behind/front/topdown 视角

---

## Q7: 夹爪和 gripper_arm 模式是分别渲染的吗?

**默认同时渲染两个视频 (`--mode both`)。**

### 模式说明

| `--mode` | 行为 | 输出 |
|---|---|---|
| `gripper` (默认: 否) | 仅渲染夹爪 | `*_gripper_urdf.mp4` |
| `gripper_arm` | 仅渲染夹爪+手臂末端 | `*_gripper_urdf_arm.mp4` |
| `both` (默认) | 两者都渲染 | 两个视频都生成 |

### 用法

```bash
# 默认: 同时渲染 gripper + gripper_arm
python hand_track/render_gripper_only.py --hawor-dir ... --ras-dir ...

# 仅 gripper
python hand_track/render_gripper_only.py --hawor-dir ... --ras-dir ... --mode gripper

# 仅 gripper_arm
python hand_track/render_gripper_only.py --hawor-dir ... --ras-dir ... --mode gripper_arm
```

---

## Q8: 双手模式下能不能不生成单独左/右手视频?

**已实现: 双手模式合成后自动删除单独的左/右手视频。**

### 改动

`render_auto.py` 双手模式下:
1. 渲染左臂 → 渲染右臂 → 合成双臂视频 → **删除左/右臂视频**
2. 渲染左夹爪 → 渲染右夹爪 → 合成双夹爪视频 → **删除左/右夹爪视频**
3. 渲染双夹爪 URDF (同场景, 本身就是合成视频)

### 输出文件 (双手模式)

```
videos/
├── hawor_r1_dual_tracking.mp4           ← 双臂合成 (不保留单独左/右)
├── hawor_r1_dual_gripper.mp4            ← 双夹爪关键点合成 (不保留单独左/右)
├── hawor_r1_dual_gripper_urdf.mp4       ← 双夹爪URDF (仅夹爪)
└── hawor_r1_dual_gripper_urdf_arm.mp4   ← 双夹爪URDF (夹爪+手臂末端)
```
