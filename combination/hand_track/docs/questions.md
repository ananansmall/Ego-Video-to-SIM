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

### 验证结果 (数据集7, 100帧)

| 模式 | 手 | 指尖1 | 指尖2 | 手腕 | 指向 | 开合 |
|---|---|---|---|---|---|---|
| gripper | 右 | 0.91mm | 0.74mm | 0.30mm | 0.04° | 0.03° |
| gripper | 左 | 3.51mm | 2.79mm | 0.83mm | 0.12° | 0.07° |
| gripper_arm | 右 | 与 gripper 一致 (同一位姿计算) | | | | |
| gripper_arm | 左 | 与 gripper 一致 (同一位姿计算) | | | | |

右手误差 < 1mm, 左手误差 ~3.5mm (左手人手几何与夹爪差异更大)。

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

---

## Q9: 为什么夹爪3个点不能完全对应上 MANO 手的3个点？

**根因: 夹爪和 MANO 手的几何尺度不同，3个点的相对距离不匹配，刚性变换无法改变距离。**

### 问题

夹爪有3个特征点 (gripper_link, finger_link1, finger_link2)，对应 MANO 手的3个点 (手腕joint0, 指尖joint4, 指尖joint8)。无论怎么旋转平移夹爪，3个点都无法同时精确对应。

### 数学证明

**刚性变换 (旋转+平移) 保持距离不变。** 如果两组3个点的内部距离不同，就不存在刚性变换使它们完全重合。

实测数据 (数据集7):

| 距离 | MANO 右手 | MANO 左手 | 夹爪 (URDF原始) |
|---|---|---|---|
| 拇指→腕 (finger1) | **125.5mm** | **116.5mm** | 37.0mm |
| 食指→腕 (finger2) | **144.8mm** | **127.7mm** | 37.0mm |
| 指尖间距 | 35.6mm | 59.0mm | 26.9mm |

关键发现: MANO 的拇指(finger1)和食指(finger2)的腕→指尖距离**不同** (右手差 ~19mm, 左手差 ~11mm)。夹爪原始 URDF 两个手指的 X 偏移都是 37mm。

- **腕→指尖距离**: 夹爪 37mm vs MANO 116-145mm，差 79-108mm
- **指尖间距**: 夹爪基距 26.9mm，通过关节可扩展到 126.9mm，可以覆盖 MANO 范围

### 为什么 `visualize_hand_object.py` 和 `02_render_scene.py` 也不能完全对应？

它们都用 `dex_retargeting` 优化器 (NLopt SLSQP)，优化目标是最小化3个点的位置误差。但优化器也是在做**刚性变换**——8 DOF (6 位姿 + 2 手指关节) 对 9 约束 (3点×3坐标)，超定系统只能求最小二乘解，无法让所有误差为0。

实测优化器的3点误差 vs 解析方法 (数据集7, 右手):

| 方法 | finger1 | finger2 | wrist |
|---|---|---|---|
| dex_retargeting 优化器 (2点) | 10.2mm | 10.3mm | 123mm |
| dex_retargeting 优化器 (3点) | 10.6mm | 10.5mm | 97mm |
| 解析方法 (不缩放) | 0.4mm | 0.4mm | 85mm |
| 解析方法 (单值缩放X) | 2.6mm | 3.3mm | 13mm |
| **解析方法 (独立手指缩放X)** | **0.91mm** | **0.74mm** | **0.30mm** |

### 解决方案: 独立手指缩放X方向

把 URDF 两个手指的 `finger_origin_x` 从 37mm 分别缩放到各自对应的 MANO 腕→指尖距离:

```python
finger1_origin_x = mean(wrist_to_finger1_dist)  # 拇指→腕, ~125mm(右)/~117mm(左)
finger2_origin_x = mean(wrist_to_finger2_dist)  # 食指→腕, ~145mm(右)/~128mm(左)
finger_origin_x  = (finger1 + finger2) / 2      # 平均值, 用于优化器模式回退
```

**原理**: 
1. 独立缩放后，gripper_link→finger_link1 的距离匹配 MANO 拇指→腕距离
2. gripper_link→finger_link2 的距离匹配 MANO 食指→腕距离
3. gripper_link 自然在腕部位置 (手腕误差 < 1mm)，两个指尖也精确对应

**为什么右手精度高于左手**: MANO 右手几何更接近夹爪 (右手食指 Y 分量 ~13mm vs 夹爪 13.45mm)，而左手食指 Y 分量偏大 (~19mm)，与夹爪的正交几何差异更大。

**残余误差来源**: 即使独立缩放了 X 方向，Y 方向的几何差异仍然存在:
- 夹爪的指向方向 (X轴) 和开合方向 (Y轴) **严格正交**
- MANO 手的指向方向 (腕→指尖中点) 和开合方向 (finger2-finger1) **不正交** (内积可达 0.6)
- 这个"正交性差异"是残余误差的根本原因，无法通过缩放消除

### 验证结果 (数据集7, 100帧)

| 手 | finger1 | finger2 | wrist | 指向误差 | 开合误差 |
|---|---|---|---|---|---|
| 右手 | 0.91mm | 0.74mm | 0.30mm | 0.04° | 0.03° |
| 左手 | 3.51mm | 2.79mm | 0.83mm | 0.12° | 0.07° |

### 类比

想象你有一把小尺子 (37mm) 和一大一小两把长尺子 (拇指 125mm, 食指 145mm)。你不可能把小尺子的两端同时放在两把长尺子的两端——因为长度不同。独立缩放就是把小尺子的两端分别拉长到对应 MANO 手指的长度。

---

## Q10: `render_gripper_only.py` 到底有没有调用 `detect_hands` 做手部检测?

**结论: 调用了。**

### 调用链路

```
render_gripper_only.py::main()
  └── detect_hands(args.hawor_dir)      ← 自动检测
      └── 返回 [0] / [1] / [0,1] / []
  └── render_gripper_only_video(hand_idx=...)
      └── 用传入的 hand_idx 渲染单只手
```

### 关键代码位置

[render_gripper_only.py](file:///home/an/robot_world_ws/src/dex-retargeting/example/combination/hand_track/render_gripper_only.py#L1345-L1370):

```python
if args.hand_idx >= 0:
    hand_indices = [args.hand_idx]
else:
    hand_indices = detect_hands(args.hawor_dir)
    hand_count = len(hand_indices)
    if hand_count == 0:
        logger.error("...停止生成")
        sys.exit(1)
```

### 为什么感觉"没用检测"?

检测只在 `main()` 入口做一次, 真正的渲染函数 `render_gripper_only_video()` / `render_dual_gripper_video()` 接收的是**已经确定好的 `hand_idx`**。渲染函数内部只负责根据 `hand_idx` 取数据, 不重复做检测。

这样设计的理由是:
- 检测逻辑与渲染逻辑解耦, 便于复用
- `render_auto.py` 和 `render_gripper_only.py` 共用 `common.detect_hands()`
- 用户可以通过 `--hand-idx 0/1` 覆盖检测结果

### 验证

在 `7` 上运行:
```
手部检测: 左手 (indices=[0])
```
实际视频也只渲染了左手, 与检测结果一致。

---

## Q11: 为什么 `01_align_scene.py` "对齐失败"? 明明有相机位置。

**结论: 对齐本身没失败, 失败的是最后的验证步骤。**

### 对齐到底依赖什么

`01_align_scene.py` 的核心计算只依赖两组相机位姿:
- RAS 相机轨迹 (`extrinsics/*.txt`)
- HaWoR 相机轨迹 (`R_c2w`, `t_c2w`)

基于这两组位姿, 脚本计算:
1. `R_align` — 第一帧相机朝向对齐
2. `t_align` — 第一帧相机位置对齐
3. `s_inv` — Umeyama 尺度比

这些计算**不依赖手部数据**。

### "对齐失败"的真实原因

之前的崩溃发生在 **Step 6 验证阶段**:

```python
from scipy.spatial import cKDTree
tree = cKDTree(glb_hawor)
dists, _ = tree.query(pred_trans[hand_idx, valid_frames])
```

某些数据 (如 `laptop`) 中:
- `pred_valid[hand_idx]` 为 `True`
- 但对应帧的 `pred_trans[hand_idx]` 全是 `NaN`
- `cKDTree.query()` 传入 NaN 数组, 直接崩溃

### 修复方式

已在 [01_align_scene.py](file:///home/an/robot_world_ws/src/dex-retargeting/example/combination/01_align_scene.py) 中修复:

1. 生成 `valid_frames` 时同步过滤 `pred_trans` 中的 NaN:
   ```python
   valid_mask = pred_valid[hand_idx] & ~np.isnan(pred_trans[hand_idx]).any(axis=-1)
   valid_frames = np.where(valid_mask)[0]
   ```

2. 当 `valid_frames` 为空时, 跳过手-GLB距离验证, 仍然保存 `transform_params.npz`:
   ```python
   if len(valid_frames) == 0:
       print(f"  ⚠ {hand_label} 无有效非NaN帧，跳过手-GLB距离验证")
   else:
       ...  # 正常做 cKDTree 验证
   ```

### 修复后 `laptop` 的运行结果

```
Step 5: 尺度校正 (Umeyama)
  s_inv (RAS→HaWoR 缩放): 1.025258

Step 6: 验证对齐
  GLB中心 (HaWoR): [-0.66681061  1.07988363 -0.1335726 ]
  ⚠ 左手 无有效非NaN帧，跳过手-GLB距离验证

Step 7: 保存变换参数
  保存到: .../transform_params.npz
```

对齐参数正常生成, 后续渲染脚本可以加载 GLB 场景。

### 为什么 `pred_valid=True` 但 `pred_trans=NaN`?

这是 HaWoR 重建结果本身的数据质量问题: 手部被标记为"可见/有效", 但 3D 位置估计失败, 输出默认 NaN。`detect_hands()` 已经在更高层过滤掉这类数据, 所以 `laptop` 被正确识别为只含右手; 但 `01_align_scene.py` 之前没有过滤, 导致验证步骤崩溃。

---

## Q12: 相机轨迹从 HaWoR 到 SAPIEN 的完整映射链是什么？为什么相机方向是错的？

**日期**: 2026-06-26
**分类**: 调试 / 架构

### 问题
用户感觉"相机方向是错的"，需要完整追踪相机轨迹从 HaWoR 坐标系到 SAPIEN 渲染坐标系的映射链，并给出实际数值。

### 完整映射链

#### 1. 加载 (load_hawor_c2w, common.py L289-295)
```python
rec = np.load(rec_file, allow_pickle=True)
return rec['R_c2w'], rec['t_c2w']   # 直接读取, 无任何预处理
```
- `R_c2w`: (N, 3, 3) 相机到世界的旋转
- `t_c2w`: (N, 3) 相机到世界的平移 (= 相机在世界中的位置)

#### 2. 关键常量 (common.py L74-76, 02_render_scene.py L151)
```python
R_x    = np.diag([1.0, -1.0, -1.0])                          # SLAM→render
R_AXIS = np.array([[1,0,0],[0,0,1],[0,-1,0]])                # Y-up→Z-up
RXWORLD_TO_SAPIEN = R_AXIS @ R_x
```
数值结果 (det = +1.0):
```
RXWORLD_TO_SAPIEN = [[1, 0,  0],
                     [0, 0, -1],
                     [0, 1,  0]]
```

#### 3. 转换函数 (common.py L406-420, 02_render_scene.py L1105-1137, 两处完全相同)
```python
def hawor_cam_to_sapien_pose(R_c2w, t_c2w):
    cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w          # 位置: HaWoR Y-up → SAPIEN Z-up
    cam_R_sapien   = RXWORLD_TO_SAPIEN @ R_c2w          # 旋转: 同上
    forward = -cam_R_sapien[:, 2]    # OpenGL: -Z = 前
    left    = -cam_R_sapien[:, 0]    # OpenGL: -X = 左
    up      =  cam_R_sapien[:, 1]    # OpenGL: +Y = 上  ← 可疑
    sapien_cam_R[:, 0] = forward
    sapien_cam_R[:, 1] = left
    sapien_cam_R[:, 2] = up
    if det < 0: SVD 修正
    return cam_pos_sapien, quaternion_from_matrix(sapien_cam_R)
```

#### 4. 调用点
- common.py `render_robot_video`: L814 (初始), L852 (每帧) — 仅 `view=="fpv"`
- common.py `render_gripper_only_video`: L1052 (初始), L1079 (每帧) — 仅 `view=="fpv"`
- 02_render_scene.py:
  - `run_hand_only`: L1558, L1734, L1746, L1840
  - `run_robot_tracking`: L2067, L2142
  - `run_robot_only`: L2545, L2580

#### 5. 01_align_scene.py 中 R_c2w 是否被预处理？
**否。** HaWoR 的 `R_c2w` 直接传给渲染器。仅 RAS 的 `R_c2w` 做了 `ZUP_TO_YUP` 转换。代码注释自相矛盾：文件头注释称 HaWoR 是 "OpenCV 约定 (X=right, Y=down, Z=forward)"，但 `hawor_cam_to_sapien_pose` 实际按 OpenGL 约定 (`forward=-Z, up=+Y`) 处理。

### 实际数值 (trace_camera2.py)

两个数据集第 0 帧:
```
R_c2w[0] ≈ diag(1, -1, -1)   ← 哨兵/初始化值 (非真实姿态)
```

数据集 `7` hand 0 (113 有效帧):
```
OpenGL (forward=-Z): cam→hand · forward  mean = +0.933   ← 手在相机前方
OpenCV  (forward=+Z): cam→hand · forward  mean = -0.933  ← 手在相机后方 (矛盾)

forward (函数输出, SAPIEN 世界): [ 0.0048, -0.9999,  0.0063]   → 朝 SAPIEN -Y
up      (函数输出, SAPIEN 世界): [-0.0011, -0.0063, -0.9999]   → 朝 SAPIEN -Z
up · SAPIEN +Z = -1.000   ← 相机 Z 轴 (上方) 指向世界下方！
```

数据集 `hoi4d`: 相机移动 ~50cm，yaw ∈ [-128°, -83°]，pitch ∈ [-25°, 14°]。

### 结论 (相机方向错误的根因)

1. **前向方向 (forward = -Z) 是对的**：手在相机前方 (点积 +0.933)，SAPIEN 实测 -Y 球体出现在画面中心。
2. **上方方向 (up = +cam_R_sapien[:, 1]) 是错的**：对于 `R_c2w[0] = diag(1,-1,-1)`，`up = [0,0,-1]` 在 SAPIEN 世界中指向**下方** (`up · +Z = -1.0`)，导致渲染画面**上下颠倒**。
3. **约定矛盾**：`01_align_scene.py` 注释称 HaWoR 是 OpenCV (Y=down)，但转换函数按 OpenGL (Y=up) 处理。数值证据支持 **forward = -Z (OpenGL 式)**，但相机 Y 轴在 HaWoR 世界中为 -Y，表现为 **"混合" 约定** (Y=down 如 OpenCV，Z=back 如 OpenGL)。
4. **修正方向**：在混合约定下应为 `up = -cam_R_sapien[:, 1]`，但单独翻转会使 `det = -1` (非正常旋转)，需要协调翻转其他轴或保留 SVD 修正。

---

## Q15: 如何新增一个灵巧手渲染? 为什么用 `render_dexterous_only.py` 而非扩展 `render_gripper_only.py`?

**结论**: 已通过新建 `render_dexterous_only.py` 实现, 支持 6 种灵巧手 (allegro/inspire/shadow/ability/leap/svh), 用 `--robot-name` 指定, 通过 `00_run_pipeline.py --dexterous` 调用。

### 调研结论

1. **GalaxeaManipSim 没有灵巧手**: `/home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets` 下只有 r1 / r1_lite / r1_pro 三种平行夹爪, 无多指灵巧手
2. **dex-retargeting 自带 6 种灵巧手**: `dex-retargeting/assets/robots/hands/{allegro,inspire,shadow,ability,leap,svh}/` 各有完整 URDF + mesh + YAML 配置
3. **参考实现**: `dex-retargeting/example/position_retargeting/visualize_hand_object.py` 展示了标准加载模式 (add_dummy_free_joint + RetargetingConfig)

### 为什么新建文件而非扩展?

用户原话: "我认为最终的目标是随意替换机器人能渲染, 不过你先把一个灵巧手完成"。新建独立文件有以下好处:
- **单一职责**: `render_gripper_only.py` 专注 1-DOF 平行夹爪 + 解析对齐; `render_dexterous_only.py` 专注多指灵巧手 + dex-retargeting 优化器
- **避免破坏现有功能**: `render_gripper_only.py` 已稳定 (指尖误差 0.9-3mm), 扩展会引入复杂分支
- **统一机器人接口**: 未来其他机器人 (机械臂/双足/灵巧手) 可各自有专用渲染文件, 通过 `--robot-name` 切换

### 关键模式 (来自 visualize_hand_object.py)

```python
# 1. URDF 加载 (加 6DOF dummy free joint 让手腕可移动)
robot_urdf = urdf.URDF.load(
    str(urdf_path),
    add_dummy_free_joints=True,    # 关键: 让手腕可自由移动
    build_scene_graph=False,
)

# 2. 配置加载 (从 dex-retargeting 内置 YAML)
config_path = get_default_config_path(robot_name, hand_type)
config = RetargetingConfig.load_from_file(config_path)
retargeting = config.build()

# 3. Warm start (用 MANO 手腕位姿初始化)
R_mano = pr.matrix_from_compact_axis_axis(hand_pose_frame[0:3])  # 注意 1D 索引
R_sapien = RXWORLD_TO_SAPIEN @ R_mano
wrist_quat = pr.quaternion_from_matrix(R_sapien)
retargeting.warm_start(...)

# 4. 渲染循环 (优化器自动求手腕+手指)
ref_value = joints_sapien[target_link_human_indices, :3]
qpos = retargeting.retarget(ref_value)[retarget2sapien]
robot.set_qpos(qpos)
```

### 调用方式

```bash
# 直接调用
/home/an/miniconda3/envs/dex/bin/python hand_track/render_dexterous_only.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --robot-name allegro

# 通过一键管线
python 00_run_pipeline.py \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --dexterous --robot-name inspire
```

### 6 种灵巧手规格

| 机器人 | 总关节数 | 手指关节数 | target links | 指数 | 说明 |
|---|---|---|---|---|---|
| allegro | 22 | 16 | 8 | 4 | 默认, 4 指工业灵巧手 |
| inspire | 18 | 12 | 5 | 5 | 类人手 (无 _glb 版本, 用原始 URDF) |
| shadow  | 30 | 24 | 10 | 5 | Shadow Hand |
| ability | 16 | 10 | 5 | 5 | Ability Hand (PSYONIC) |
| leap    | 22 | 16 | 8 | 4 | Leap Hand |
| svh     | 26 | 20 | 10 | 5 | Schunk SVH |

---

## Q16: `002_render_scene.py` 的双臂加载、left 手腕提取与 `--view` 模式实现情况

**日期**: 2026-07-10
**分类**: 调试 / 架构

### 问题

1. `002_render_scene.py` 的双臂加载是否继承了 `hand_track/` 里的处理方式？遇到"无法提取 left 手腕位置"时，它为什么会失败？是有错误就不提取了吗？
2. `002_render_scene.py` 是否继承了 `02_render_scene.py` 的功能？特别是与 `001_align_scene.py` 的坐标对齐，以及 `--view` 模式的输出。

### 解答

#### 1. 双臂加载与 left 手腕提取

**002_render_scene.py 没有继承 hand_track 的双臂处理方式。**

- `hand_track/render_auto.py` 对双手的处理是：分别调用 `render_robot_video(hand_idx=0)` 和 `render_robot_video(hand_idx=1)` 渲染左右臂，得到两个独立视频，最后用 `_combine_videos_side_by_side` 左右拼接成 `hawor_r1_dual_tracking.mp4`。
- `002_render_scene.py` 则自己实现了 `render_dual_robot_video`（`002_render_scene.py` L1998 起）：在同一个 SAPIEN 场景中加载左右两条 R1 浮动臂，各用各的 IK solver，共享同一个相机。

**关于"无法提取 left 手腕位置"：**

在 `render_dual_robot_video` 的"放置机器人"阶段（约 L2161-2165），会对每只手调用：

```python
wrist_positions = _compute_wrist_positions_glb(
    st["hawor_data"], st["mano_layer"], start_frame, num_frames, s, R_h2g, t_h2g)
if not wrist_positions:
    logger.error(f"无法提取 {prefix} 手腕位置")
    return None
```

如果某只手（例如 left）在 `[start_frame, start_frame+num_frames)` 范围内没有任何一帧满足 `pred_valid=True` 且 `pred_trans` 非 NaN，`_compute_wrist_positions_glb` 就会返回空列表。此时代码会**直接 `return None`，整个双臂渲染中止**，而不是跳过左手、继续渲染右手。

常见原因：
- 数据集本身左手为无效手，或 `pred_valid` 全 False / `pred_trans` 全 NaN。
- `load_hawor_data(hawor_dir, hand_idx=0)` 只按索引 0 取左手数据，内部 `_fill_nan_frames` 会把 NaN 帧标为 invalid，但不会拒绝整只手；渲染层也没有在计算腕部位置前再次兜底校验。

#### 2. 与 `02_render_scene.py` / `001_align_scene.py` 的关系

**坐标对齐：**

- `002_render_scene.py` 确实以 `001_align_scene.py` 输出的 `transform_params.npz` 作为输入，但用的是**新坐标系**参数：
  - `scale_ratio`
  - `R_hand_to_glb`
  - `t_hand_to_glb`
- 而 `02_render_scene.py` 用的是旧坐标系参数：
  - `s_inv`、`R_inv`、`t_inv`
  - 再叠加 `RXWORLD_TO_SAPIEN` 把 HaWoR SLAM world 转到 SAPIEN。

因此 `002_render_scene.py` 是 `001_align_scene.py` 的"新坐标系配套渲染脚本"，不是 `02_render_scene.py` 的直接继承者。它的核心变换是 `hand_to_glb(j, s, R_h2g, t_h2g)`，GLB 直接原样加载（`load_glb_direct`），不再做 `ZUP_TO_YUP` 或 `RXWORLD_TO_SAPIEN` 变换。

**`--view` 模式：**

- `002_render_scene.py` 保留了 `--view` 参数，支持 `"fpv", "behind", "front", "topdown"`（`002_render_scene.py` L2472），与 `02_render_scene.py` 的 `choices=["fpv", "topdown", "behind", "front"]` 基本一致。
- `render_robot_video` 中实现了对应的相机位姿计算（`002_render_scene.py` L459-480）：`fpv` 使用 HaWoR 相机轨迹转换到 GLB 坐标系；`topdown/behind/front` 使用固定视角。

**功能差异：**

- `002_render_scene.py` 没有继承 `02_render_scene.py` 的 `TrajectorySmoother`（速度/加速度/jerk 限幅 + 双向滤波），只做了关节级 `LPFilter`。
- 也没有 `hand_only / robot_only / robot_tracking` 三种模式划分，而是新增了 `gripper_only` 等模式，并在 `gripper_only` 下复用了 `hand_track/render_gripper_only.py` 的夹爪渲染逻辑。

### 关键文件位置

- `002_render_scene.py` 主入口与参数解析：L2461-2593
- `render_dual_robot_video`：L1998 起
- `hand_track/render_auto.py` 双手处理：L277 起
- `hand_track/common.py` 中的 `detect_hands` / `load_hawor_data` / `_compute_wrist_positions_glb`

---
