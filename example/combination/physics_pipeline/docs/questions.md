# Q&A Log

## Q: SAPIEN 仿真视频帧数不匹配 + 桌子映射 + --view 参数 + README 调用说明

**日期**: 2026-06-19
**分类**: 调试 / 架构

### 问题
1. SAPIEN 渲染的还是不对。仿真视频的帧数不应该和视频展示的一样吗，hawor 是多少，仿真视频的帧数就是多少
2. 目前的桌子还是不对，没有映射过来，视频看不出仿真，有直接渲染的代码在里面吗，指定 --view 的，而且我需要你在 readme 把各种调用都告诉我
3. PyBullet 的仿真还可以，主要有几个问题：
   - 桌子还是没有找好，我现在不明白 glb 真实物体的加载方式，为什么桌子没支持还有物理仿真呢，机械臂有物理仿真吗，怎么看起来映射的还可以，现在是有物理补偿？
   - 可以在 sapien 和 pybullet 指定一个单夹爪的操作，就是只有一个夹爪的物理仿真，可以通过 -- 指定，在 hand_track 有类似的操作，可以学习一下
   - 我建议你要在 readme 里面添加各部分仿真的实现方式，然后特别是命令也得讲清楚

### 解答

#### 1. 帧数不匹配问题
**根因**: `frame_repeat = max(1, round(1.0 / self.speed))`，默认 `--speed 0.5` 导致 `frame_repeat=2`，视频帧数 = HaWoR帧数 × 2 = 226帧（HaWoR 是 113 帧）。

**修复**: `--speed` 默认值改为 1.0，使 `frame_repeat = 1`，视频帧数 = HaWoR 帧数 = 113帧。

#### 2. 桌子映射问题
**根因**: `ground_height = min_z - 0.01`（物体最低点下方1cm），导致物体悬空1cm后下落，看不出仿真效果。

**修复**: 改为 `min_z - 0.002`（2mm间隙，紧贴支撑），SAPIEN 04 和 PyBullet 都修复。

**--view 参数**: 已添加 `--view` 参数（fpv/topdown/behind/front），可以指定不同视角渲染。

#### 3. PyBullet 桌子和物理仿真

**GLB 真实物体的加载方式**:
- 数据流: `final_scene.glb` (RAS y-down) → HaWoR render world (y-up) → SAPIEN (z-up)
- 变换链: `vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv` → `vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T`
- 物体分类: 大型扁平几何体 (桌面/地板) → static (mass=0); 小物体 → dynamic (mass=volume*1000)

**为什么桌子没支持还有物理仿真**:
- 桌面本身 **没有物理仿真** (mass=0, kinematic)，它只是一个静态碰撞体，用于支撑 dynamic 物体
- 物理仿真只作用于 dynamic 的 GLB 物体和机器人

**机械臂有物理仿真吗**:
- **SAPIEN 04**: 有物理仿真 (PD 驱动 + 重力补偿 `compute_passive_force(gravity=True)`)
- **PyBullet**: 无物理仿真 (运动学控制 `resetJointState`，直接设置关节角，不受重力/惯性影响)

**为什么映射看起来还可以**:
- 因为关节角来自 02 的 IK 解 (RelaxedIK)，已经是合理的姿态
- PyBullet 只是用运动学方式重放这个姿态，所以看起来映射还可以
- **没有物理补偿**，因为机械臂是 kinematic 的，不会下落，不需要重力补偿

#### 4. 单夹爪模式
已在 SAPIEN 04 和 PyBullet 中都添加 `--single-gripper` 参数：
- 只加载夹爪 URDF (无机械臂)
- 直接用 MANO 手腕位姿驱动夹爪 (`_compute_analytical_gripper_pose`)
- 参考 `hand_track/render_gripper_only.py` 的实现

#### 5. README
已重写 README，添加：
- "各部分仿真的实现方式" 章节 (GLB加载/桌面支撑/机械臂物理仿真/单夹爪模式/相机视角/帧数对齐)
- "调用方式" 章节 (SAPIEN 04 + PyBullet 所有命令)
- "完整参数列表" 表格

---

## Q: SAPIEN 04 单夹爪bug + PyBullet物理仿真 + 相机/重力对齐02

**日期**: 2026-06-19
**分类**: 调试 / 架构

### 问题
1. SAPIEN 04 单夹爪模式报错 `IndexError: _Map_base::at` (camera.get_picture)
2. PyBullet 也需要有物理仿真，一定要有物理仿真的，读取物理仿真的姿态和 02 的 IK 对比和重力修复，单夹爪模式也需要物理仿真。最终任务是将 02 的视频仿真进阶到物理仿真
3. 相机视角要和 02_render_scene.py 相同，不要在 pybullet 仿真还是上帝视角，重力位置不对应该反了

### 解答

#### 1. SAPIEN 04 单夹爪 bug
**根因**: `camera.get_picture("Color")` 前缺少 `camera.take_picture()`，SAPIEN 渲染缓冲区未填充。
**修复**: 添加 `camera.take_picture()`，并修正颜色处理 (clip+scale 与主方法一致)。

#### 2. PyBullet 物理仿真
**原问题**: PyBullet 用 `resetJointState` (运动学控制)，机器人无物理仿真。
**修复**: 改为 computed torque control (PD + 重力补偿)，与 SAPIEN 04 一致：
- `calculateInverseDynamics(q, q_dot, 0)` 计算重力补偿力矩 (等价 SAPIEN `compute_passive_force(gravity=True)`)
- `tau = kp*(target - q) - kd*q_dot + tau_gravity` (等价 SAPIEN `set_drive_target` + `set_qf`)
- `setJointMotorControl2(TORQUE_CONTROL, force=tau)` 施加力矩
- PD增益: kp=1000, kd=200 (与 SAPIEN 04 stiffness/damping 一致)

**IK误差监控**: 每帧读取物理仿真实际位姿，与 02 IK 目标对比，显示 `IK err=...` (30帧测试: err=0.8641)

**单夹爪模式**: root位姿运动学控制 (像 SAPIEN `set_root_pose`)，手指关节 PD+重力补偿

#### 3. 相机视角和重力
**相机**: PyBullet 默认 `--view fpv`，跟随 HaWoR 相机轨迹 (与 02 完全一致)，通过 `hawor_cam_to_sapien_pose` + `sapien_cam_to_pybullet_view` 完整复用 02 的相机变换链。

**重力**: `[0, 0, -9.81]` (Z-down)，SAPIEN 和 PyBullet 都用 Z-up 坐标系，重力方向正确。桌面在物体最低点下方 2mm (`ground_z = min_z - 0.002`)，物体自然落稳。

---

## Q: GLB 物体在仿真中显得很小 + 实时渲染环境重载 + 基座固定 + 仿真测试要求

**日期**: 2026-06-22
**分类**: 调试 / 架构

### 问题
1. 实时的渲染需要重新加载仿真环境，而不是在同一个仿真环境里面不断渲染
2. 为什么感觉在仿真环境里面 GLB 的加载会那么小呢，物体大小有发生变化吗？
3. 之后的仿真需要测试渲染50%以上的视频才能结束，刚开始有点问题
4. 在 SAPIEN 仿真里面基座最好固定一下

### 解答

#### 1. 实时渲染环境重载
**修复**: PyBullet 管线新增 `reset_simulation()` 和 `_init_simulation()` 方法:
- `__init__` 只存储参数，调用 `_init_simulation()` 完成初始化
- `reset_simulation()` 断开当前连接，重新创建全新仿真环境 (物理客户端、地面、机器人)
- `render_video()` 和 `render_single_gripper_video()` 开头调用 `reset_simulation()`，确保每次渲染从干净环境开始
- GLB 物体在重置后重新加载

#### 2. GLB 物体大小问题
**分析**: PyBullet 管线中 GLB 物体**没有额外缩放**:
- `meshScale=[1, 1, 1]` (1:1 原始尺寸)
- 顶点变换链与 02_render_scene.py 完全一致: `vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv` → `vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T`
- 物体大小完全取决于 `transform_params.npz` 中的 `s_inv` (逆缩放因子)

**可能原因**:
- `s_inv` 值偏小: 01_align_scene.py 的 Umeyama 对齐计算的缩放因子可能使物体变小
- 相机视角: 第一人称视角 (fpv) 的相机位置可能离物体较远，导致物体看起来小
- 视觉参照: 没有明确的参照物 (如桌面纹理、网格线)，难以判断物体实际大小

**验证方法**:
```python
# 检查 s_inv 值
params = np.load('transform_params.npz')
print(f"s_inv = {params['s_inv']}")  # 应接近 1.0

# 检查物体实际尺寸
vertices_sapien = ...  # 变换后的顶点
bbox = vertices_sapien.max(axis=0) - vertices_sapien.min(axis=0)
print(f"物体尺寸: {bbox} 米")  # 应该是合理的物体大小 (几cm到几十cm)
```

#### 3. 仿真测试渲染50%以上视频
**修复**: `render_video()` 和 `render_single_gripper_video()` 添加渲染进度验证:
- 渲染完成后检查实际写入帧数是否 >= 目标帧数的 50%
- 如果不足 50%，打印警告并返回 False
- 确保仿真测试真正渲染了大部分视频内容

#### 4. SAPIEN 04 基座固定
**修复**: 新增 `--fixed-base` 参数:
- 默认: 浮动基座模式 (基座在 XY 方向 ±4cm 范围跟踪手腕)
- `--fixed-base`: 基座固定在初始位置，不跟随手腕移动
- 优先级: `fixed_base` > `base_cluster` > 浮动基座
- SAPIEN 已使用 `fix_root_link=True` (物理固定)，`--fixed-base` 是在此基础上不再调用 `set_root_pose` 移动基座

---
