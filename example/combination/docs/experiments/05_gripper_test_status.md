# 单夹爪物理仿真 — 完整总结 (2026-07-14)

> 本文档记录单夹爪从 MANO 参数到物理仿真的完整链路、已修复的 Bug、当前精度、以及三个抓取仿真脚本的详细工作原理。

---

## 一、涉及的源文件

| 文件 | 角色 | 今日状态 |
|------|------|---------|
| `04_physics_simulation.py` | 主物理仿真器 (~3700行) | **已修复 3 处 Bug** |
| `05_gripper_test.py` | 增量测试脚本 (新增) | **已通过 3 项测试** |
| `06_simple_grasp_test.py` | 纯夹爪抓取 Demo (虚拟关节) | 已有功能, 未修改 |
| `trajectory_optimization/grasp_hawor.py` | R1 机器人抓取 GLB 物体 (393KB) | 参考实现 |
| `trajectory_optimization/grasp_demo.py` | GalaxeaManipSim gym 环境抓取 demo | 参考实现 |
| `CHANGE_LOG.md` | 变更日志 | 待更新 |

---

## 二、今日修复的 Bug 与根因分析

### Bug 1: `finger_center` 返回值错误 (关键)

**位置**: `04_physics_simulation.py` `_compute_analytical_gripper_pose()` 函数末尾

```python
# 修改前 (BUG):
finger_center = (mano_finger1 + mano_finger2) / 2
return root_pos, R, joint, joint         # ← 第4个值传了标量 joint (~0.02), 应为 finger_center

# 修改后:
return root_pos, R, joint, finger_center  # ← 正确返回 3D 指尖中点向量
```

**根因分析**:
1. 函数内部先计算了 `finger_center`（3D 向量，指尖中点）
2. 用 `finger_center` 计算 `root_pos = finger_center - R @ [0.03689, 0, 0]`
3. 但在 return 语句中，第 4 个返回值误写为 `joint`（一个标量关节角，约 0.02m）
4. 调用方 `solve()` 解包为 `init_root_pos, init_R, joint, finger_center = ...`
5. 得到 `finger_center = 0.02`（标量），不是 3D 向量
6. 传入 `_lm_optimize()` 后，`root_pos = 0.02 - R @ offset`，位置完全错误
7. **结果**: 指尖跟踪偏差 ~20mm，看起来"不动"或"位置错乱"

**为什么这么隐蔽**: 标量赋值不报类型错误（NumPy 广播），但结果完全错位。

### Bug 2: LM 优化器恶化指尖跟踪

**位置**: `04_physics_simulation.py` `GripperIKOptimizer.solve()` 方法

```python
# 修改前: 调 LM 优化器
opt_root_pos, opt_R, opt_joint = self._lm_optimize(
    finger_center, init_R, joint, sm_f1, sm_f2, R_target)

# 修改后: 直接用解析解
opt_root_pos = init_root_pos
opt_R = init_R
opt_joint = joint
```

**根因分析**:
LM 优化器的 cost 函数:
```
cost = 10 × |finger1_pred - MANO4|²  (指尖跟踪, 权重√10)
     + 10 × |finger2_pred - MANO8|²  (指尖跟踪, 权重√10)
     +  3 × angle(R, R_target)²      (朝向跟踪, 权重√3)
```

当解析解 R 与 R_target（MANO 手腕朝向）相差 **16.7°** 时:
- 指尖 cost ≈ 10 × 0.0001² ≈ **0.000001**
- 朝向 cost ≈ 3 × 0.291² ≈ **0.254**

**朝向 cost 是指尖 cost 的 25 万倍**。LM 为了减小总 cost，旋转 R 去追 R_target，代价是放弃指尖跟踪 → 指尖误差从 0.1mm 涨到 **20mm**。

**结论**: 解析解（Gram-Schmidt 正交化）的指尖精度远优于 LM 优化，直接输出解析解即可。朝向跟踪的权重应大幅降低，或者完全移除（朝向由手指几何自动决定）。

### Bug 3: 桌面碰撞检测导致人工抬升

**位置**: `04_physics_simulation.py`（已删除整段碰撞检测代码）

```python
# 删除了: 手指穿透桌面时抬升 root_pos[2] += lift_z (约 93-108mm)
```

**根因分析**: HaWoR 重建数据的手部深度不准确（手在桌面以下 Z=-0.055m），碰撞检测将夹爪抬高 93-108mm。这并非真实的跟踪误差，而是数据问题的人工补偿。删除后误差更真实。

### Bug 4: `joint` 计算公式错误 (05_gripper_test.py)

```python
# 错误 (05 初次编写):
joint = (finger_dist / 2) / 0.013453  # → 50mm (被 clip 到最大值)

# 正确 (与 04 一致):
_FINGER_BASE_DIST = 0.026906  # 两指基础间距 (URDF 定义)
joint = max(0.0, min(0.05, (finger_dist - _FINGER_BASE_DIST) / 2))
```

**影响**: 测试3 初始 IK 误差 45.6mm → 修复后 0.1mm。

---

## 三、验证结果 (无头渲染)

| 测试 | 内容 | 最大误差 | 平均误差 | 状态 |
|------|------|---------|---------|------|
| 测试1 | 夹爪开合 (正弦波) | 6.2mm | 1.0mm | ✅ |
| 测试2 | 夹爪移动 (圆形轨迹) | **0.0mm** | **0.0mm** | ✅ |
| 测试3 | MANO 联合跟踪 | IK=0.1mm / 实际=2.0mm | IK=0.1mm / 实际=1.9mm | ✅ |

**总误差分解**:
```
总误差 1.9mm = IK 解析解 0.1mm + PD 物理跟踪 ~1.8mm
PD 跟踪误差来自手指关节 PD 参数 (stiffness=1000, damping=10)
EMA 平滑延迟贡献 3-8mm (仅当快速运动时)
```

---

## 四、解析法 IK 算法 (当前使用)

```
输入: mano_wrist, mano_f1 (关节4-拇指尖), mano_f2 (关节8-食指尖)

Step 1: EMA 平滑 (α=0.3)
  sm_wrist = 0.3*mano_wrist + 0.7*prev_sm_wrist
  sm_f1    = 0.3*mano_f1    + 0.7*prev_sm_f1
  sm_f2    = 0.3*mano_f2    + 0.7*prev_sm_f2

Step 2: Gram-Schmidt 正交化
  指尖中心:   finger_center = (sm_f1 + sm_f2) / 2
  指尖距离:   finger_dist   = ||sm_f2 - sm_f1||
  开合角度:   joint         = (finger_dist - 0.026906) / 2  [clip 0~0.05]

  Y轴 (手指开合方向):   y = normalize(sm_f2 - sm_f1)
  X轴 (手腕→指尖方向):  x_raw = finger_center - sm_wrist
                        x = normalize(x_raw - dot(x_raw, y)*y)
  Z轴 (右手系):         z = normalize(cross(x, y))
  R = [x | y | z]

Step 3: 根位置 (硬约束: 手腕在夹爪中心线上)
  root_pos = finger_center - R @ [0.03689, 0, 0]
            # 0.03689 = URDF 中 base_link → 指尖中心的 X 偏移

输出: root_pos, root_quat, joint, joint  (对称开合)
```

### URDF 几何常数 (单位: 米)

```python
_FINGER1_ORIGIN   = [0.03689, -0.013453, -0.00012053]  # 手指1关节原点
_FINGER2_ORIGIN   = [0.03689,  0.013453,  0.00012067]  # 手指2关节原点
_FINGER1_AXIS     = [0, -1, 0]  # 手指1开合方向 (沿 -Y)
_FINGER2_AXIS     = [0,  1, 0]  # 手指2开合方向 (沿 +Y)
_GRIPPER_DEPTH_OFFSET = 0.03689  # base_link → 指尖中心 (X 方向)
_FINGER_BASE_DIST     = 0.026906  # 两指基础间距 (= 0.013453×2)
```

### 坐标系确认

```python
R_x = diag(1, -1, -1)          # 无缩放
R_AXIS = [[1,0,0],[0,0,1],[0,-1,0]]  # 纯旋转
RXWORLD_TO_SAPIEN = R_AXIS @ R_x      # 纯旋转, det=1, 无缩放
```

**MANO 和 URDF 都使用"米"为单位，坐标系变换无缩放，不存在尺度问题。**

---

## 五、物理仿真流程

```
HaWoR 数据 (pred_joints, pred_rot, pred_betas)
          │
          ▼
┌─ MANO 层 ──────────────────────────────┐
│  joints_mano = MANOLayer(rotvec, betas) │
│  joints_sapien = RXWORLD_TO_SAPIEN @ joints_mano  │
│  (纯旋转变换, 无缩放)                          │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─ 提取关键点 ────────────────────────────┐
│  mano_wrist  = joints[0]  (手腕)         │
│  mano_f1     = joints[4]  (拇指尖)       │
│  mano_f2     = joints[8]  (食指尖)       │
│  R_target    = rotvec_to_R(pred_rot)     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─ EMA 平滑 ──────────────────────────────┐
│  3个输入点分别做 α=0.3 的指数移动平均     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─ 解析法 IK (Gram-Schmidt) ──────────────┐
│  输出: root_pos, R, joint, finger_center │
│  (不再做 LM 优化)                         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─ 物理驱动 ───────────────────────────────┐
│  set_root_pose(root_pos, root_quat)       │
│  set_root_linear_velocity(zeros)          │
│  set_root_angular_velocity(zeros)         │
│  set_drive_target(joint, joint)           │
│                                            │
│  for _ in range(DECIMATION=8):            │
│      qf = compute_passive_force(gravity)  │
│      set_qf(qf)                           │
│      scene.step()  ← PhysX 求解           │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─ 验证 (诊断日志) ────────────────────────┐
│  IK_err = FK指尖 vs MANO参考点            │
│  f_err  = 物理后指尖 vs MANO参考点         │
│  j_err  = 关节PD跟踪误差                  │
└──────────────────────────────────────────┘
```

---

## 六、已知限制

| 问题 | 说明 | 影响 | 优先级 |
|------|------|------|--------|
| HaWoR 深度不准 | 手部重建深度偏浅，手出现在桌面以下 | 需桌面交互时有影响 | 低 (当前不交互) |
| 朝向 16.7° 偏差 | 解析解 R 和 R_target 差 ~16.7° | 仅诊断值，不参与优化 | 低 (已知) |
| EMA 平滑延迟 | α=0.3 导致 3-8mm 轨迹延迟 | 快速运动时可见 | 低 (可调) |
| 关节 PD 跟踪 ~2mm | stiffness=1000, damping=10 | 物理仿真固有误差 | 低 (可接受) |

---

## 七、抓取仿真脚本详解

### 7.1 `06_simple_grasp_test.py` — 纯夹爪 6-DOF PD 驱动抓取

**目标**: 验证夹爪能否通过 PD 控制真正抓起物体（不依赖运动学驱动）。

**核心创新 — 6 个虚拟关节替代 `set_root_pose`**:

`set_root_pose()` 是**运动学驱动**（直接设置位置，不产生物理力），无法让夹爪真正"接触"并"推动"物体。本脚本改为 URDF 链式结构，通过虚拟关节 PD 驱动：

```
world ──[prismatic_x]──> x_link
                    ──[prismatic_y]──> y_link
                                    ──[prismatic_z]──> z_link
                                                    ──[revolute_rz]──> rz_link
                                                                    ──[revolute_ry]──> ry_link
                                                                                    ──[revolute_rx]──> rx_link
                                                                                                    ──[fixed]──> gripper_base
                                                                                                            ── gripper_link
                                                                                                                ── finger1 (prismatic)
                                                                                                                ── finger2 (prismatic)
```

**PD 参数** (对齐 GalaxeaManipSim):
| 关节类型 | stiffness | damping | effort | velocity |
|---------|-----------|---------|--------|----------|
| 平移 (virtual_x/y/z) | 1000 | 200 | 5000 | 5 |
| 旋转 (virtual_rz/ry/rx) | 200 | 50 | 500 | 10 |
| 手指 (finger1/2) | 1000 | 200 | 500 | 0.25 |

**抓取六步流程**:

```
Step 1: 初始位置 (Z=0.4m, 手指半开 0.025m)
  └─ set_target(0, 0, 0.4, 0, 0, 0, 0.025, 0.025)
  └─ 物理稳定 1 帧

Step 2: 下降到方块处 (Z=0.4→0.1m, 60帧线性插值)
  └─ z = 0.4 - frame * 0.0025
  └─ 手指保持半开

Step 3: 张开夹爪 (手指 0.025→0.05m, 30帧)
  └─ f = frame/30 * 0.05
  └─ 夹爪位置不变

Step 4: 继续下降到碰触方块 (Z=0.1→0.02m, 40帧)
  └─ z = z - 0.005, clamp 到 0.02
  └─ 手指保持全开

Step 5: 闭合夹爪 (手指 0.05→0.0m, 60帧)
  └─ f = 0.05 - frame/60 * 0.05
  └─ 夹爪位置不变, 手指压住方块

Step 6: 抬起方块 (Z=0.02→0.18m, 80帧)
  └─ z = z + 0.002
  └─ 手指保持闭合 (0.0m)
```

**碰撞组设置** (防止手指互碰):
- 手指1: group=2, mask=0b1001 (与手指2不碰撞，与方块碰撞)
- 手指2: group=2, mask=0b1001 (与手指1不碰撞，与方块碰撞)
- 方块:   group=3, mask=0b0111 (与手指碰撞)

**物理步进函数**:
```python
def step_physics(n=DECIMATION):
    for _ in range(n):  # 8 个子步
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)  # 重力 + 科里奥利力补偿
        scene.step()       # PhysX 求解
```

**验证结果**: 方块从 1.5cm 升至 19.5cm (抬升 18.0cm)，抓取成功。

**与 04_physics_simulation.py 单夹爪模式的区别**:

| 方面 | 06_simple_grasp_test.py | 04_physics_simulation.py (单夹爪) |
|------|------------------------|----------------------------------|
| 夹爪移动方式 | 6 虚拟关节 PD 驱动 | `set_root_pose()` 运动学驱动 |
| 物理交互 | 真实 PD 力，可产生挤压力 | 无物理力，不能主动挤压 |
| 旋转控制 | 3 虚拟旋转关节 (PD) | `set_root_pose()` 直接设置 |
| 适用场景 | 抓取验证，需要物理接触 | MANO 轨迹重放 |

---

### 7.2 `trajectory_optimization/grasp_hawor.py` — R1 机器人抓取 GLB 物体 (含 CEM 优化)

**目标**: 用 R1 机器人 URDF 在 SAPIEN 中复刻 HaWoR 手部重建的抓取动作，并通过 CEM 优化找出最优抓取位姿。

**两种模式**:

#### 模式 1: `full_robot` (整台机器人)
- 加载 `r1_v2_1_0.urdf`，将 arm joints 从 `fixed` → `revolute`
- 轨迹生成: **DexRetargeting**(夹爪 IK) + **RelaxedIK**(机械臂 IK) → **纯 PD 驱动**
- 机械臂 6 关节 PD 参数: `stiffness=1000, damping=200`

#### 模式 2: `gripper_only` (纯夹爪)
- 加载纯夹爪 URDF（无机械臂）
- 轨迹生成: **MANO 指尖向量** → 解析法 IK → 夹爪位姿 + 手指关节角

**关键修复**: `r1_v2_1_0.urdf` 中 `<joint type="fixed">` 跨多行，旧正则要求 name/type 同行导致匹配失败。用 `re.DOTALL + \s+` 修复。

**CEM 轨迹优化** (借鉴 do-as-i-do Stage 5):

```
优化目标: 30 DOF (5 帧 × 6 DOF)
  帧: F48, F49, F50, F51, F52 (抓取窗口)
  每帧: (dx, dy, dz, droll, dpitch, dyaw)

奖励函数 compute_reward_xyz:
  w_lift  = 800  (物体提升高度奖励)
  w_dist  = 100  (末端-物体距离惩罚)
  w_contact = 50  (接触帧数奖励)

CEM 迭代:
  1. 从当前策略采样 200 条候选参数
  2. 每条 rollout 在 SAPIEN 中跑完整轨迹
  3. 评估 reward = w_lift*lift + w_dist*dist + w_contact*contact
  4. 选 top-50 更新策略均值和协方差
  5. 迭代直到收敛
```

**优化后参数应用**:
```python
# F48-F52 设为固定帧, 各自独立偏移
_fixed_offsets_654 = {
    "F0": 0,  # 初始帧
    "F48": opt_params[0:6],   # 第1帧偏移
    "F49": opt_params[6:12],  # 第2帧偏移
    "F50": opt_params[12:18], # 第3帧偏移
    "F51": opt_params[18:24], # 第4帧偏移
    "F52": opt_params[24:30], # 第5帧偏移
    "F95": lift_target,       # 提升目标
    "F112": lift_target,      # 释放目标
}
```

**CLOSE 阶段控制策略**:
```python
# Phase A (闭合 50%): 保持在 F50 位置, 手指闭合
# 给 PD 足够时间收敛到目标
# Phase B (提升): 线性提升 z (0→0.15m)
# 手指保持闭合, 夹爪向上移动提起物体
```

**验证结果**: 1/7 物体被成功夹住 (glb_1: lift=13.6cm, xy_drift=12cm)。

---

### 7.3 `trajectory_optimization/grasp_demo.py` — GalaxeaManipSim gym 环境抓取 demo

**目标**: 用自定义 gym 环境演示 R1 机器人抓取方块，借鉴 GalaxeaManipSim 的 dual_bottles_pick_easy 模式。

**环境设计**:

```python
class BoxPickEnv(DualBottlesPickEasyEnv):
    """用 create_box 替代 rand_create_glb (不需要 robotwin_models)"""

    @property
    def table_height(self):
        return 0.9  # 桌面高度 0.9m

    def _setup_red_bottle(self):
        # 用 create_box 创建红色方块 (6cm 立方体)
        box_pos = self.tabletop_center_in_world + np.array([-0.1, -0.15, 0.05])
        self.red_bottle = create_box(
            scene=self._scene,
            pose=sapien.Pose(p=box_pos),
            half_size=[0.03, 0.03, 0.03],
            color=(1, 0.2, 0.2),
            name="red_box",
        )
```

**抓取方案** (6 个子步骤):

```python
def solution(self):
    # 1. 移动到接近位姿 (方块前方 10cm)
    right_pose0 = SapienPose(
        p=self.red_bottle.get_pose().p + np.array([-0.1096, -0.1164, 0.]),
        q=right_grasp_ori.q,
    )
    yield ("move_to_pose", {"right_pose": deepcopy(right_pose0)})

    # 2. 张开夹爪
    yield ("open_gripper", {"action_mode": "both"})

    # 3. 移动到抓取位姿 (贴近方块)
    right_pose1 = SapienPose(
        p=self.red_bottle.get_pose().p + np.array([-0.0196, -0.0164, 0.]),
        q=right_grasp_ori.q,
    )
    yield ("move_to_pose", {"right_pose": deepcopy(right_pose1)})

    # 4. 闭合夹爪
    yield ("close_gripper", {"action_mode": "both"})

    # 5-6. 移动到目标位姿 (抬起)
    yield ("move_to_pose", {"right_pose": deepcopy(self.right_target_pose)})
    yield ("move_to_pose", {"right_pose": deepcopy(self.right_target_pose)})
```

**规划器** (BimanualPlanner):
- 基于 moveit 的规划器，在 URDF 上运行
- 处理所有偏移和校准（通过 gym 环境）
- `planner.solve(substep, qpos, gripper_cmd)` 返回 action 序列

**参数级验证**:

| 验证项 | 方法 | 期望 |
|--------|------|------|
| 夹爪开合 | `open_gripper` 阶段 qpos 最大值 | ~0.05 |
| 夹爪闭合 | `close_gripper` 阶段 qpos 最小值 | ~0.0 |
| 跟随上升 | 抓取后方块 Δz vs 末端 Δz | 差值 < 5cm |
| 提升判定 | 方块最终 z > 初始 z + 0.1m | 提升 > 10cm |

---

## 八、三个抓取仿真脚本对比

| 方面 | 06_simple_grasp_test.py | grasp_hawor.py | grasp_demo.py |
|------|------------------------|---------------|---------------|
| 夹爪控制 | 6 虚拟关节 PD 驱动 | PD 驱动 (set_drive_target) | gym BimanualPlanner |
| 机械臂 | 无 | 有 (RelaxedIK) | 有 (规划器) |
| 物理引擎 | SAPIEN PhysX | SAPIEN PhysX | SAPIEN PhysX |
| 优化 | 无 | CEM 30DOF | 无 (预设方案) |
| 物体 | 方块 (actor) | GLB 模型 | 方块 (create_box) |
| 抓取成功 | ✓ (抬升 18cm) | 1/7 (抬升 13.6cm) | 待运行 |
| 参数 | stiffness=1000, damping=200 | stiffness=1000, damping=200 | GalaxeaManipSim 默认 |
| 用途 | 验证 PD 驱动能抓取 | 用 HaWoR 数据驱动抓取 | gym 环境 demo |

---

## 九、文件状态

```
04_physics_simulation.py    — 已修复 3 处 Bug, 可直接使用
05_gripper_test.py          — 新增, 3 项测试均通过 ✅
06_simple_grasp_test.py     — 已有功能, 未修改
grasp_hawor.py              — 参考实现, 393KB
grasp_demo.py               — 参考实现, 待运行
CHANGE_LOG.md               — 待更新
```

## 十、调用方式

```bash
# 单夹爪 MANO 跟踪 (推荐)
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
/home/an/miniconda3/envs/dex/bin/python 04_physics_simulation.py \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result \
  --hand-idx 1 --single-gripper --viewer

# 增量测试 (验证 IK + PD)
/home/an/miniconda3/envs/dex/bin/python 05_gripper_test.py --test 0 --num-frames 60

# 纯夹爪抓取 Demo (6-DOF PD)
/home/an/miniconda3/envs/dex/bin/python 06_simple_grasp_test.py

# R1 机器人抓取 GLB (CEM 优化)
/home/an/miniconda3/envs/dex/bin/python trajectory_optimization/grasp_hawor.py \
  --mode gripper_only \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result

# GalaxeaManipSim gym 抓取 demo
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization
/home/an/miniconda3/envs/dex/bin/python grasp_demo.py --output output/grasp_demo.mp4
```
