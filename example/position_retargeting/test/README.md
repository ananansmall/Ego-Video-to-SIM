# R1 机械臂 IK 位置跟踪测试

## 快速开始

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting/test

# 圆形轨迹（默认）
conda run -n dex python r1_ik_position_test.py --num-frames 60 --trajectory circle

# 直线 / 方形 / 螺旋
conda run -n dex python r1_ik_position_test.py --num-frames 60 --trajectory line
conda run -n dex python r1_ik_position_test.py --num-frames 60 --trajectory square
conda run -n dex python r1_ik_position_test.py --num-frames 60 --trajectory helix

# 自定义参数
conda run -n dex python r1_ik_position_test.py \
    --num-frames 120 \
    --trajectory circle \
    --ik-warmup-iters 300 \
    --fps 15 \
    --output-video my_test.mp4
```

### URDF 正确性验证（不需要 GPU）

```bash
conda run -n dex python verify_correct_urdf.py
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-frames` | 120 | 轨迹帧数 |
| `--trajectory` | circle | 轨迹类型: circle / line / square / helix |
| `--ik-warmup-iters` | 200 | IK 预热迭代次数（RelaxedIK 需多次迭代收敛） |
| `--fps` | 15 | 输出视频帧率 |
| `--output-video` | r1_ik_position_test.mp4 | 输出视频文件名 |

### 预期结果

| 指标 | 值 |
|------|-----|
| 平均误差 | ~0.5cm |
| 最大误差 | ~0.8cm |
| <2cm 帧占比 | 100% |

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `r1_ik_position_test.py` | 主测试：IK 轨迹跟随，生成视频 + 误差统计 |
| `verify_correct_urdf.py` | 验证脚本：确认 URDF 正确性（纯 CPU，无需 GPU） |

---

## 坐标系变换详解

### 涉及的三个坐标系

```
┌─────────────────────────────────────────────────────────────────┐
│                        世界帧 (World Frame)                      │
│  原点: SAPIEN 场景原点 (0, 0, 0)                                 │
│  X: 右, Y: 后, Z: 上                                             │
│  机器人根节点设为 Pose(p=[0,0,0], q=[0,0,0,1]) 即绕 Z 轴旋转 180° │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │  base_link_R_inv @ (pos_world - base_link_p)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   arm_base 帧 (Right Arm Base Frame)             │
│  原点: right_arm_base_link 的位置                                 │
│  朝向: 由于全身 URDF 的 right_arm_base_link 绕 Z 轴旋转了 180°,  │
│        所以 R ≈ diag(-1, -1, 1)                                  │
│  即: arm_base 帧的 X 轴指向世界 -X, Y 轴指向世界 -Y, Z 轴不变     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │  直接输入 RelaxedIK
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              RelaxedIK 内部帧 (Floating Arm URDF Frame)          │
│  原点: right_arm_base_link (即 arm_base 帧原点)                  │
│  朝向: 与 arm_base 帧完全一致                                     │
│  关键: 这个帧的 FK 链定义必须与 SAPIEN 加载的 URDF 一致           │
└─────────────────────────────────────────────────────────────────┘
```

### 变换公式

**世界帧 → arm_base 帧**（唯一需要的变换）:

```python
pos_base = base_link_R_inv @ (pos_world - base_link_p)
```

其中:
- `base_link_p`: SAPIEN 中 `right_arm_base_link` 的世界坐标位置
- `base_link_R`: SAPIEN 中 `right_arm_base_link` 的旋转矩阵 (3x3)
- `base_link_R_inv`: `base_link_R` 的逆，等于 `base_link_R.T`（旋转矩阵正交）

**朝向变换**:

```python
R_ee_base = base_link_R_inv @ R_ee_world
quat_base = quaternion_from_matrix(R_ee_base)   # [w, x, y, z]
```

**RelaxedIK 输入**:

```python
joints = ik_solver.solve_position_right(pos_base.tolist(), quat_base.tolist())
```

### 为什么 base_link_R ≈ diag(-1, -1, 1)?

全身 URDF 中，机器人根节点被设置为 `Pose(q=[0,0,0,1])`，这是绕 Z 轴旋转 180° 的四元数。
这个旋转会传递到 `right_arm_base_link`，使其相对于世界帧有 180° Z 旋转:

```
R_180Z = diag(-1, -1, 1)

世界帧 X → arm_base 帧 -X
世界帧 Y → arm_base 帧 -Y
世界帧 Z → arm_base 帧 Z
```

这意味着: 如果一个点在世界帧中位于 (0.3, 0.1, -0.2)，
在 arm_base 帧中会变成 (-0.3, -0.1, -0.2)。
**不做这个旋转，就会导致 X/Y 轴翻转，轨迹形状正确但位置错误。**

### 完整数据流

```
目标位置 (arm_base 帧中定义，如 [0.30, 0.0, -0.20])
    │
    ├──► pos_world = base_link_R @ pos_arm + base_link_p    (转到世界帧，用于可视化/误差计算)
    │
    └──► 直接作为 RelaxedIK 输入                              (arm_base 帧就是 IK 的输入帧)
              │
              ▼
         RelaxedIK.solve_position_right(pos_arm, quat_arm)
              │
              ▼
         关节角 joints[0..5]
              │
              ▼
         SAPIEN: robot.set_qpos(joints)
              │
              ▼
         FK 验证: ee_pos_world = right_ee_link.get_entity_pose().p
              │
              ▼
         误差 = ||ee_pos_world - target_world||
```

---

## 问题复盘：从 38cm 误差到 0.5cm 的完整排查过程

### 1. 现象

运行 `r1_ik_position_test.py`，机械臂末端与目标轨迹偏差极大:
- 平均误差 ~38cm，最大误差 ~77cm
- 轨迹**形状正确**但**位置偏移**，特别是在 X 平面上运动方向相反

### 2. 初步排查：IK 求解器是否有问题？

**假设**: RelaxedIK 本身求解不准

**测试**: 用 floating arm URDF 单独测试 RelaxedIK

```python
# 直接用 floating arm 的 FK 验证
target = [0.30, 0.0, -0.20]
joints = ik_solver.solve_position_right(target, [1,0,0,0])
# 用数学 FK 验证
math_pos, _ = floating_arm_fk(joints)
error = np.linalg.norm(math_pos - target)  # < 0.01cm
```

**结论**: RelaxedIK 本身没问题，误差 < 0.01cm。问题出在坐标变换或 URDF 上。

### 3. 第二步排查：坐标变换是否正确？

**假设**: 世界帧 → arm_base 帧的变换有误

**测试**: 对比两种变换方式

```python
# 方式 A: 完整旋转+平移变换
pos_A = base_link_R_inv @ (pos_world - base_link_p)

# 方式 B: 只做平移，不做旋转
pos_B = pos_world - base_link_p
```

**发现**: 方式 A 和 B 的结果差异巨大。因为 `base_link_R ≈ diag(-1,-1,1)`，
不做旋转会导致 X/Y 轴翻转。但即使用方式 A，误差仍然很大 (~30cm)。

**结论**: 坐标变换方向正确，但还有其他问题。

### 4. 第三步排查：SAPIEN FK 链是否与 URDF 一致？

**假设**: SAPIEN 加载的 URDF 的 FK 链与数学 FK 不一致

**测试**: 给 SAPIEN 机器人和数学 FK 模型输入相同的关节角，对比末端位置

```python
# 设置相同的关节角
joint_angles = [-1.5, 1.9508, -1.0809, -0.4438, 0.1709, 0.1985]

# SAPIEN FK
robot.set_qpos(joint_angles)
ee_sapien = right_ee_link.get_entity_pose().p
ee_sapien_in_base = base_R.T @ (ee_sapien - base_p)

# 数学 FK (floating arm URDF)
ee_math, _ = floating_arm_fk(joint_angles)

# 对比
diff = np.linalg.norm(ee_sapien_in_base - ee_math)  # 33cm！
```

**发现**: 即使在同一坐标系下，SAPIEN FK 和数学 FK 差了 33cm！
这说明 SAPIEN 加载的 URDF 与 floating arm URDF 的运动学定义不同。

### 5. 第四步排查：URDF 关节定义对比

**假设**: 全身 URDF 的关节定义与 floating arm URDF 不同

**测试**: 解析两个 URDF 的关节定义并对比

| 关节 | floating arm (正确) | r1/robot.urdf (错误) |
|------|---------------------|----------------------|
| joint1 origin | [0, 0, 0.086] | [-0.001, -0.045, 0] |
| joint1 axis | [0, 0, 1] | [0, -1, 0] |
| joint2 origin | [0, 0.031, 0.049] | [-0.001, 0.046, 0] |
| joint2 axis | [0, 1, 0] | [0, 0, 1] |

**发现**: 两个 URDF 的关节原点和旋转轴**完全不同**！
这不是坐标系变换的问题，而是 URDF 本身的定义就不一样。

### 6. 根因确认：配置文件指向了错误的 URDF

查看 `r1_full_right.yml` 配置文件:

```yaml
# 错误: 指向 GalaxeaManipSim 的 robot.urdf（关节定义不同）
urdf_path: /home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/robot.urdf
```

而 `r1_full.yml`（未被使用）指向的是:

```yaml
# 正确: 指向 dex-retargeting 的 r1_lite_robot_glb.urdf（关节定义与 floating arm 一致）
urdf_path: r1_full/r1_lite_robot_glb.urdf
```

**关键发现**: `get_default_config_path()` 对 `RobotName.r1_full + HandType.right`
会选择 `r1_full_right.yml`，而不是 `r1_full.yml`。
而 `r1_full_right.yml` 使用了绝对路径，`set_default_urdf_dir()` 对绝对路径无效。

### 7. 修复

修改 `r1_full_right.yml` 和 `r1_full_left.yml`:

```yaml
# 修复后
urdf_path: /home/an/robot_world_ws/src/dex-retargeting/assets/robots/r1_full/r1_lite_robot_glb.urdf
```

同时移除测试脚本中为错误 URDF 拟合的 `R_FK_MAP` / `T_FK_MAP` 修正矩阵，
简化坐标变换为:

```python
pos_base = base_link_R_inv @ (pos_world - base_link_p)
```

### 8. 验证结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 平均误差 | ~38cm | 0.49cm |
| 最大误差 | ~77cm | 0.77cm |
| <2cm 帧数 | 0/30 | 30/30 |

---

## 排查思路总结

```
现象: IK 误差 38cm
  │
  ├─ Q1: IK 求解器本身有问题吗？
  │   └─ 单独测试 RelaxedIK → 误差 <0.01cm → ❌ 不是 IK 的问题
  │
  ├─ Q2: 坐标变换有问题吗？
  │   ├─ 不做旋转 (方式B) → X/Y 翻转 → 轨迹形状对但位置错
  │   └─ 做旋转 (方式A) → 仍然 ~30cm → 变换方向对但还有别的问题
  │
  ├─ Q3: SAPIEN 的 FK 链对不对？
  │   └─ 同一关节角，SAPIEN FK vs 数学 FK → 差 33cm → ❌ FK 链不一致
  │
  ├─ Q4: 为什么 FK 链不一致？
  │   └─ 对比 URDF 关节定义 → 关节原点和轴完全不同 → 根因找到！
  │
  └─ Q5: 为什么加载了错误的 URDF？
      └─ r1_full_right.yml 使用绝对路径指向 robot.urdf（关节定义不同）
         而 r1_lite_robot_glb.urdf（关节定义一致）从未被使用
         → 修改配置文件指向正确 URDF → 误差降至 0.5cm ✓
```

### 关键教训

1. **URDF 一致性是基础**: IK 求解器使用的 URDF 必须与仿真加载的 URDF 关节定义一致。
   如果关节原点或旋转轴不同，无论坐标变换多么精确，FK 链本身就不匹配。

2. **绝对路径配置的风险**: 配置文件中使用绝对路径时，`set_default_urdf_dir()` 无效。
   这意味着即使你设置了正确的 URDF 目录，也不会生效。

3. **轨迹形状正确但位置偏移 → 先查坐标系旋转**: 如果轨迹的形状（圆形还是圆形）
   正确但位置不对，通常是因为缺少旋转变换（如 180° Z 旋转）。

4. **隔离变量排查**: 当问题涉及多个环节（IK 求解器、坐标变换、URDF、物理仿真）时，
   从最简单的环节开始逐个排除，而不是同时修改多个变量。

---

## RelaxedIK 求解器说明

### 工作原理

RelaxedIK 是基于梯度的连续优化 IK 求解器，核心特点:

1. **连续优化**: 每次调用 `solve_position_right()` 只走一步梯度下降，
   不是一次性求解。因此需要从接近目标的初始状态出发。

2. **预热机制**: 对第一个目标点需要多次调用（~200 次）让求解器收敛:

   ```python
   for _ in range(200):
       joints = ik_solver.solve_position_right(target_pos, target_quat)
   ```

3. **松散旋转容差**: RelaxedIK 同时优化位置和朝向。如果朝向不可达，
   会拉偏位置。设置旋转容差为 10.0 弧度（≈573°）等效于不约束朝向:

   ```python
   tolerances = [0.001, 0.001, 0.001, 10.0, 10.0, 10.0]
   #              位置x   位置y   位置z   旋转x   旋转y   旋转z
   ```

4. **四元数格式**: Python 端使用 `[w, x, y, z]`，Rust 内部使用 `[x, y, z, w]`。
   `RelaxedIKSolver` 已在内部处理了这个转换。

### 输入要求

- 位置: `right_arm_base_link` 坐标系下的 [x, y, z]
- 朝向: `right_arm_base_link` 坐标系下的四元数 [w, x, y, z]
- 关节角初始化: `ik_solver.relaxed_ik_right.reset(initial_joints)`
