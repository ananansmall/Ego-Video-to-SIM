# 借鉴 do-as-i-do 的离线轨迹优化方案 (第十八轮)

**Goal**: 在 MANO 轨迹基础上, 用 CEM 采样式优化找到"最小偏离 + 抓取成功"的参数, 借鉴 do-as-i-do Stage 5 思想, 物理仿真保持 SAPIEN + PD 不变
**Architecture**: 新增离线优化模块 `traj_optimize.py`; 重构 `_compute_mano_neutral_target` 接受参数化调整; 物理仿真层回归纯 PD 控制 (删除 set_qpos / lock_root_pose hack)
**Tech Stack**: SAPIEN, NumPy, CEM (Cross-Entropy Method)
**参考**: `/home/an/robot_world_ws/src/do-as-i-do/retargeting/RETARGETING_TECHNICAL_DOC.md` Stage 5

---

## 当前 hybrid 分支诊断

### 现状 (`_compute_mano_neutral_target` L2297-2415)
手工启发式调整 MANO 轨迹:
- 位置: `mano_pos + offset` (常量平移), CLOSE 阶段前 30% blend 到 `grasp_pos`
- 姿态: APPROACH/TRANSPORT/RELEASE 跟随 MANO R, CLOSE 用 `_make_horizontal_closing_R` 水平化
- 手指: APPROACH 跟 MANO j1, CLOSE/TRANSPORT 闭合 0.0, RELEASE 张开

### 问题
1. **手工调参, 无反馈**: `close_dur=20`, `descend_t=0.3`, `FINGER_FORWARD=0.037`, `gripper_val=0.0` 都是猜的, 不知道是否最优
2. **测试 7 次都失败**: 证明手工调参无效 (要么夹不住, 要么 TRANSPORT 滑落)
3. **物理层 hack**: `set_qpos(-0.01)` 瞬移手指 + `lock_root_pose` 锁根, 绕过物理引擎

### do-as-i-do 的启示
| do-as-i-do Stage 5 | 我们的对应 |
|---------------------|-----------|
| CEM 采样式 MPC (MuJoCo Warp GPU) | CEM 离线优化 (SAPIEN CPU) |
| 奖励 = 位置跟踪 + 旋转跟踪 + 关节正则 + 穿透惩罚 | 奖励 = 偏离代价 + 接触帧数 + 提升量 + 距碗距离 |
| Warmup (焊接约束 + 零重力) | 不需要 (gripper_only 无此问题) |
| 域随机化 + 扰动 | 不需要 (单次优化, 非训练) |
| 滚动时域 | 不需要 (离线优化整条轨迹) |

---

## 方案: 离线 CEM 轨迹优化

### 核心思想
1. MANO 轨迹 = 参考轨迹 (要跟随的目标)
2. 优化变量 = 相对参考轨迹的**参数化调整** (最小偏离)
3. 在 SAPIEN 中 rollout 每条候选调整, 评估抓取质量
4. CEM 迭代: 采样 → 评估 → 精英选择 → 更新均值/方差
5. 最优参数用于生成最终视频

### 优化变量 (9 维, 关键阶段参数化)

不优化整条轨迹 (高维难收敛), 只优化关键阶段参数:

| 参数 | 维度 | 含义 | 默认值 (当前手工值) | 搜索范围 |
|------|------|------|---------------------|----------|
| `grasp_pos_delta` | 3 | CLOSE 时 grasp 位置相对 target 的偏移 | [0,0,0] | ±2cm |
| `grasp_R_euler` | 3 | CLOSE 姿态相对水平化 R 的欧拉角修正 | [0,0,0] | ±15° |
| `finger_close_target` | 1 | 手指闭合 PD target | 0.0 | [-0.01, 0.02] |
| `close_blend_ratio` | 1 | CLOSE 下降比例 | 0.3 | [0.2, 0.5] |
| `transport_vel_limit` | 1 | TRANSPORT 速度限幅 | 0.5 m/s | [0.1, 0.8] |

**为什么这 9 维**:
- `grasp_pos_delta` + `grasp_R_euler`: CLOSE 阶段抓取位姿 (决定能否夹住)
- `finger_close_target`: 手指闭合程度 (决定夹多紧)
- `close_blend_ratio`: 下降速度 (决定手指何时到位)
- `transport_vel_limit`: 运输速度 (决定物体是否滑落)

### 目标函数 (SAPIEN rollout 后评估)

```
reward = 
    - w_track   * ||params - params_default||²        # 偏离代价 (最小化调整)
    + w_contact * contact_frames_in_close             # CLOSE 阶段接触帧数
    + w_lift    * max(0, obj_final_z - obj_init_z)    # 提升量
    - w_bowl    * dist(obj_final, bowl_pos)           # 距碗距离
    - w_drop    * 100 * (obj_z < 0.01 at end)         # 掉落惩罚
    - w_pen     * max(0, penetration_depth - 0.01)    # 穿透惩罚
```

权重 (初值, 可调):
- `w_track = 1.0` (偏离代价, 主要约束)
- `w_contact = 5.0` (接触帧数, 鼓励持续接触)
- `w_lift = 50.0` (提升量, 主要目标)
- `w_bowl = 2.0` (距碗距离, pick-and-place)
- `w_drop = 1.0` (掉落惩罚)
- `w_pen = 10.0` (穿透惩罚)

### CEM 算法

```python
def cem_optimize(
    initial_params,         # 默认参数 (9维)
    rollout_fn,             # 评估函数: params -> reward
    n_iterations=5,         # 迭代轮数
    n_samples=16,           # 每轮采样数
    elite_frac=0.25,       # 精英比例
    initial_std=0.3,       # 初始标准差 (相对范围)
):
    mu = initial_params.copy()
    std = np.ones_like(mu) * initial_std * (param_range[:, 1] - param_range[:, 0])
    n_elite = max(1, int(n_samples * elite_frac))
    
    for iteration in range(n_iterations):
        # 1. 采样
        samples = mu + std * np.random.randn(n_samples, len(mu))
        samples = np.clip(samples, param_range[:, 0], param_range[:, 1])
        
        # 2. 评估 (并行可选, SAPIEN 单线程)
        rewards = np.array([rollout_fn(s) for s in samples])
        
        # 3. 精英选择
        elite_idx = np.argsort(rewards)[-n_elite:]
        elite_samples = samples[elite_idx]
        
        # 4. 更新
        mu = elite_samples.mean(axis=0)
        std = elite_samples.std(axis=0)
        
        logger.info(f"CEM iter {iteration}: best_reward={rewards[elite_idx[-1]]:.3f}, "
                    f"mu={mu.round(4)}")
    
    return mu  # 最优参数
```

**复杂度**: 10 轮 × 24 样本 = 240 次 SAPIEN rollout。每次 rollout ~2 秒 (50 帧), 共 ~480 秒 (8 分钟), 可接受。

---

## 文件改动清单

| 文件 | 改动类型 | 范围 |
|------|---------|------|
| `traj_optimize.py` (新建) | 新增 | CEM 优化器 + rollout 评估函数 + 目标函数 |
| `grasp_hawor.py` L2297-2415 | 修改 | `_compute_mano_neutral_target` 接受 `opt_params` 参数 |
| `grasp_hawor.py` L2772-2822 | 修改 | `_step_gripper_only` hybrid 分支: 回归纯 PD (删除 set_qpos / lock_root_pose) |
| `grasp_hawor.py` 主循环 L3630-3640 | 修改 | `physics_step` 不传 `lock_root_pose` |
| `grasp_hawor.py` main | 新增 | `--optimize` flag: 先优化再渲染 |
| `CHANGE_LOG.md` | 新增 | 第十八轮条目 |

---

## Task 1: 新建 `traj_optimize.py` (CEM 优化器)

**文件**: `tri_model_physics/traj_optimize.py` (新建)

**职责**: 
- `cem_optimize()`: CEM 算法主循环
- `rollout_and_evaluate()`: 给定参数, 在 SAPIEN 中跑完整轨迹, 返回 reward
- `compute_reward()`: 从 rollout 结果计算多目标 reward
- `PARAM_RANGE`, `DEFAULT_PARAMS`, `PARAM_NAMES`: 参数定义

**关键代码骨架**:

```python
"""离线 CEM 轨迹优化 (借鉴 do-as-i-do Stage 5)

在 MANO 参考轨迹基础上, 用 Cross-Entropy Method 优化关键阶段参数,
最小化偏离代价同时保证抓取成功. 物理仿真保持 SAPIEN + PD 控制.
"""
import numpy as np
from typing import Callable

# 优化参数定义 (9 维)
PARAM_NAMES = [
    "grasp_pos_delta_x", "grasp_pos_delta_y", "grasp_pos_delta_z",  # 3D
    "grasp_R_euler_x", "grasp_R_euler_y", "grasp_R_euler_z",        # 3D
    "finger_close_target",   # 1D
    "close_blend_ratio",     # 1D
    "transport_vel_limit",   # 1D
]
DEFAULT_PARAMS = np.array([
    0.0, 0.0, 0.0,        # grasp_pos_delta
    0.0, 0.0, 0.0,        # grasp_R_euler
    0.0,                   # finger_close_target
    0.3,                   # close_blend_ratio
    0.5,                   # transport_vel_limit
])
PARAM_RANGE = np.array([
    [-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02],  # grasp_pos_delta ±2cm
    [-0.26, 0.26], [-0.26, 0.26], [-0.26, 0.26],  # grasp_R_euler ±15°
    [-0.01, 0.02],                                  # finger_close_target
    [0.2, 0.5],                                     # close_blend_ratio
    [0.1, 0.8],                                     # transport_vel_limit
])

# 奖励权重
REWARD_WEIGHTS = dict(
    w_track=1.0, w_contact=5.0, w_lift=50.0,
    w_bowl=2.0, w_drop=1.0, w_pen=10.0,
)
```

**验证**: 单元测试 `cem_optimize` 用 mock rollout_fn 验证收敛。

---

## Task 2: 重构 `_compute_mano_neutral_target` 接受 `opt_params`

**文件**: `grasp_hawor.py` L2297-2415

**改动**: 把硬编码常数替换为 `opt_params` 参数 (默认值 = 当前值, 保持行为不变)

```python
def _compute_mano_neutral_target(self, local_idx, side, opt_params=None):
    """
    opt_params: 9 维 np.ndarray, None 时用默认值 (DEFAULT_PARAMS)
        [0:3]   grasp_pos_delta: CLOSE 时 grasp 位置偏移
        [3:6]   grasp_R_euler: CLOSE 姿态欧拉角修正 (rad)
        [6]     finger_close_target: 手指闭合 PD target
        [7]     close_blend_ratio: CLOSE 下降比例
        [8]     transport_vel_limit: TRANSPORT 速度限幅
    """
    # 默认参数 (保持向后兼容)
    if opt_params is None:
        opt_params = np.array([0,0,0, 0,0,0, 0.0, 0.3, 0.5])
    
    grasp_pos_delta = opt_params[0:3]
    grasp_R_euler = opt_params[3:6]
    finger_close_target = float(opt_params[6])
    close_blend_ratio = float(opt_params[7])
    transport_vel_limit = float(opt_params[8])
    
    # ... traj/offset/f_grasp 获取 (不变) ...
    
    if phase == "CLOSE":
        gripper_R = self._make_horizontal_closing_R(mano_R)
        # 应用姿态修正
        if np.linalg.norm(grasp_R_euler) > 1e-6:
            R_correction = pr.matrix_from_euler("xyz", grasp_R_euler)
            gripper_R = R_correction @ gripper_R
        grasp_pos = target_pos - gripper_R[:, 0] * FINGER_FORWARD_NEUTRAL + grasp_pos_delta
        close_progress = (local_idx - f_grasp) / max(close_dur, 1)
        descend_t = min(1.0, close_progress / close_blend_ratio)
        t = descend_t * descend_t * (3 - 2 * descend_t)
        gripper_pos = mano_target_pos * (1 - t) + grasp_pos * t
        gripper_val = finger_close_target  # 用优化值代替硬编码 0.0
    # ... 其他阶段类似, transport_vel_limit 用于 _step_gripper_only ...
    
    # 缓存 transport_vel_limit 供 _step_gripper_only 使用
    self._current_transport_vel_limit = transport_vel_limit
```

**验证**: `opt_params=None` 时行为与当前完全一致 (回归测试)。

---

## Task 3: `_step_gripper_only` hybrid 分支回归纯 PD 控制

**文件**: `grasp_hawor.py` L2772-2822

**改动**: 删除 `set_qpos` 和 `lock_root_pose`, 回归 PD + kinematic 根驱动

```python
# 第十八轮: 回归纯 PD + kinematic 根 (删除 set_qpos / lock_root_pose)
opt_params = getattr(self, '_opt_params', None)
neutral_target = self._compute_mano_neutral_target(local_idx, self.side, opt_params)
if neutral_target is not None:
    gripper_pos, gripper_R, gripper_val, phase = neutral_target
    self._last_phase = phase
    root_quat = pr.quaternion_from_matrix(gripper_R)
    robot = self.robot_info["robot"]
    
    # 根: set_root_pose + set_root_linear_velocity (kinematic 浮动根标准驱动)
    robot.set_root_pose(sapien.Pose(gripper_pos.tolist(), root_quat.tolist()))
    prev_pos = getattr(self, '_prev_demo_root_pos', None)
    transport_vel_limit = getattr(self, '_current_transport_vel_limit', 0.5)
    if prev_pos is not None:
        root_vel = (gripper_pos - prev_pos) * float(CONTROL_FREQ)
        vel_norm = float(np.linalg.norm(root_vel))
        if vel_norm > transport_vel_limit:
            root_vel = root_vel * (transport_vel_limit / vel_norm)
    else:
        root_vel = np.zeros(3)
    robot.set_root_linear_velocity(root_vel.tolist())
    robot.set_root_angular_velocity([0.0, 0.0, 0.0])
    self._prev_demo_root_pos = gripper_pos.copy()
    self._close_lock_pose = None  # 不锁根
    
    # 手指: 纯 set_drive_target (PD, 不用 set_qpos)
    return (), (float(gripper_val), float(gripper_val))
```

**主循环 `physics_step` 调用**: 不传 `lock_root_pose` (始终 None)

**验证**: 代码中 hybrid 分支不再有 `set_qpos`, 不再有 `_close_lock_pose` 赋值。

---

## Task 4: 实现 `rollout_and_evaluate` (rollout + reward)

**文件**: `traj_optimize.py`

**职责**: 给定 `opt_params`, 在 SAPIEN 中跑完整轨迹, 返回 rollout 结果

```python
def rollout_and_evaluate(opt_params, simulator):
    """在 SAPIEN 中跑完整轨迹, 评估抓取质量
    
    simulator: 已初始化的 GraspSimulator 实例 (含 scene/robot/objects)
    opt_params: 9 维优化参数
    """
    # 1. 重置仿真到初始状态
    simulator.reset()
    simulator.set_opt_params(opt_params)
    
    # 2. 跑完整轨迹 (113 帧)
    contact_frames_in_close = 0
    obj_init_z = simulator.get_obj_z(target_obj_name)
    max_penetration = 0.0
    close_start, close_end = simulator.get_close_frame_range()
    
    for frame_idx in range(simulator.num_frames):
        simulator.step()  # 单帧: _step_gripper_only + physics_step + 接触检测
        
        # 统计 CLOSE 阶段接触帧数
        if close_start <= frame_idx <= close_end:
            if simulator.get_contact_count(target_obj_name) >= 2:
                contact_frames_in_close += 1
        
        # 穿透检测
        pen = simulator.get_max_penetration()
        max_penetration = max(max_penetration, pen)
    
    # 3. 评估最终状态
    obj_final_pos = simulator.get_obj_pos(target_obj_name)
    obj_final_z = obj_final_pos[2]
    obj_final_xy = obj_final_pos[:2]
    obj_dropped = obj_final_z < 0.01
    bowl_xy = simulator.get_bowl_pos()[:2]
    
    rollout_result = dict(
        params=opt_params,
        contact_frames_in_close=contact_frames_in_close,
        obj_init_z=obj_init_z, obj_final_z=obj_final_z,
        obj_final_xy=obj_final_xy, bowl_xy=bowl_xy,
        obj_dropped=obj_dropped,
        max_penetration=max_penetration,
    )
    rollout_result["reward"] = compute_reward(rollout_result)
    return rollout_result
```

**关键**: `simulator` 需要暴露 `reset()`, `set_opt_params()`, `step()`, `get_contact_count()`, `get_obj_pos()`, `get_bowl_pos()`, `get_close_frame_range()`, `get_max_penetration()` 接口。

**验证**: 用默认参数 rollout 一次, 确认返回合理的 reward (应该 < 最优值, 因为默认参数没优化)。

---

## Task 5: 集成 `--optimize` flag 到 main

**文件**: `grasp_hawor.py` main 函数

**新增逻辑**:
```python
parser.add_argument("--optimize", action="store_true", 
                    help="离线 CEM 优化轨迹参数 (借鉴 do-as-i-do Stage 5)")
parser.add_argument("--opt-params", type=str, default=None,
                    help="直接加载已优化的参数文件 (npy), 跳过 CEM")

# main 中:
if args.optimize:
    from traj_optimize import cem_optimize, rollout_and_evaluate, DEFAULT_PARAMS
    logger.info("=== 离线 CEM 轨迹优化开始 (借鉴 do-as-i-do Stage 5) ===")
    logger.info(f"参数空间: {len(DEFAULT_PARAMS)} 维, 采样数: 24, 迭代: 10 轮")
    
    # 初始化 simulator (不渲染, 用于 rollout)
    sim = GraspSimulator(args, render=False)
    sim.setup()
    
    # CEM 优化
    best_params, best_reward = cem_optimize(
        rollout_fn=lambda p: rollout_and_evaluate(p, sim),
        n_iterations=10, n_samples=24,
    )
    logger.info(f"=== 优化完成: best_reward={best_reward:.3f}, params={best_params.round(4)} ===")
    
    # 保存最优参数
    np.save(output_dir / "opt_params.npy", best_params)
    
    # 用最优参数生成最终视频 (带渲染)
    sim_final = GraspSimulator(args, render=True)
    sim_final.setup()
    sim_final.set_opt_params(best_params)
    sim_final.run()  # 渲染 + 保存视频
elif args.opt_params:
    # 直接加载优化参数
    best_params = np.load(args.opt_params)
    sim = GraspSimulator(args, render=True)
    sim.setup()
    sim.set_opt_params(best_params)
    sim.run()
else:
    # 不优化, 用默认参数运行
    sim = GraspSimulator(args, render=True)
    sim.setup()
    sim.run()
```

**验证**: `--optimize` 时日志显示 CEM 迭代过程, 10 轮后输出最优参数。

---

## Task 6: 运行优化 + 验证

**命令**:
```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics

# 1. 离线优化 (约 8 分钟)
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py --mode gripper_only --side left \
  --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result \
  --views god --grasp-mode hybrid --optimize --num-frames 50

# 2. (可选) 用优化后的参数重新渲染
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py --mode gripper_only --side left \
  --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result \
  --views god --grasp-mode hybrid \
  --opt-params output/gripper_only_left/opt_params.npy
```

**成功标准**:
1. ✅ CEM 10 轮迭代收敛 (best_reward 单调上升)
2. ✅ 最优参数下: 物体被夹起 (lift > 3cm)
3. ✅ 最优参数下: 物体在 TRANSPORT 阶段跟随夹爪 (xy_drift > 5cm)
4. ✅ 最优参数下: 物体最终在碗附近 (距碗心 < 10cm)
5. ✅ 偏离代价合理: ||opt_params - DEFAULT||² < 0.01 (调整量小)
6. ✅ 视频中可见半透明红色碰撞体与物体真实接触

---

## Task 7: 失败处理

### 7.1 若 CEM 不收敛 (best_reward 不上升)
**原因**: 参数空间不对 / reward 设计不合理
**方案**:
- 检查 rollout 是否正常 (默认参数 reward 应 < 0)
- 调整 reward 权重 (增大 w_lift, 减小 w_track)
- 扩大搜索范围 (PARAM_RANGE)
- 增加迭代轮数 (10 → 15)

### 7.2 若收敛但抓不住物体
**原因**: 参数空间不够 / 物理层问题
**方案**:
- 扩展参数空间 (加 `close_dur` 参数)
- 检查 PD 是否闭合 (可能需要增大 GRIPPER_STIFFNESS)
- 检查手指几何 (FINGER_BASE_DIST 是否太大)

### 7.3 若抓住但 TRANSPORT 滑落
**原因**: 摩擦不够 / 速度太快
**方案**:
- `transport_vel_limit` 优化范围下限调到 0.05
- 增大 GRIPPER_FRICTION (1.0 → 1.5)
- 增加 TRANSPORT 阶段持续帧数

### 7.4 若穿透严重
**原因**: PD 刚度太高 / DECIMATION 太小
**方案**:
- 增大 `w_pen` 权重
- 减小 GRIPPER_STIFFNESS (1000 → 500)
- 增大 DECIMATION (8 → 16)

---

## Task 8: 文档同步

### 8.1 CHANGE_LOG.md
新增第十八轮条目:
- 用户反馈 (借鉴 do-as-i-do, 最小偏离代价)
- 修改内容 (新增 traj_optimize.py, 重构 _compute_mano_neutral_target, 回归纯 PD)
- 验证结果 (CEM 收敛 + 抓取成功)

### 8.2 docs/grasp_hawor_analysis.md
新增"离线轨迹优化"章节 (借鉴 do-as-i-do Stage 5)

### 8.3 调用 change-log skill
任务结束前调用 change-log skill 输出修改总结。

---

## 执行顺序

1. **Task 1** (新建 traj_optimize.py) — 核心优化模块
2. **Task 2** (重构 _compute_mano_neutral_target) — 参数化
3. **Task 3** (回归纯 PD) — 删除 hack
4. **Task 4** (rollout_and_evaluate) — 评估函数
5. **Task 5** (--optimize flag) — 集成
6. **Task 6** (运行优化) — 验证
7. **Task 7** (失败处理) — 按需
8. **Task 8** (文档同步) — 收尾

---

## 与 do-as-i-do 的对应关系

| do-as-i-do Stage 5 | 本方案 | 差异 |
|---------------------|--------|------|
| MuJoCo Warp (GPU 并行) | SAPIEN (CPU 单线程) | 不能并行 rollout, 但 240 次 × 2s = 8min 可接受 |
| CEM 采样式 MPC | CEM 离线优化 | 不做滚动时域, 一次性优化整条轨迹 |
| 奖励: 位置/旋转跟踪 + 穿透 | 奖励: 偏离代价 + 接触 + 提升 + 距碗 | 适配 pick-and-place 场景 |
| 优化整条控制轨迹 (高维) | 优化 9 个关键参数 (低维) | 降维加速收敛, 适合 CPU |
| Warmup (焊接+零重力) | 不需要 | gripper_only 无此问题 |
| 域随机化 + 扰动 | 不需要 | 单次优化, 非训练策略 |
| 接触引导 (Contact Guidance) | 不需要 | 离线优化已含接触奖励 |

---

## 核心原则 (本轮)

> **在 MANO 轨迹基础上, 用 CEM 优化找到最小偏离的抓取参数。**
> 
> 1. MANO 是参考, 不是要绕过的约束 (用户: "在轨迹基础上最小化偏离代价")
> 2. 物理仿真是基础, 不是要 hack 的对象 (用户: "物理仿真以之前的为基础")
> 3. 借鉴 do-as-i-do 的采样式优化思想, 适配 SAPIEN + gripper_only
> 4. 优化 9 个关键参数, 不是整条轨迹 (降维可收敛)
> 5. 删除 set_qpos / lock_root_pose hack, 回归纯 PD 控制

---

## Task 4.5: 实现细节修复

**文件**: `grasp_hawor.py`

### 4.5.1 `setup_physics_scene` 加 `force_cpu` 参数
优化模式 (`--optimize`) 下直接创建 CPU 场景, 避免 Vulkan 渲染初始化段错误:
```python
def setup_physics_scene(ground_height=GROUND_HEIGHT, force_cpu=False):
    if force_cpu:
        return sapien.Scene(systems=[sapien.physx.PhysxCpuSystem()])
    # 正常模式: 先尝试渲染, 失败再降级 CPU
```
`run_optimize` 中调用时传 `force_cpu=True`。

### 4.5.2 接触检测 API 修复
SAPIEN Contact API 是 `c.bodies[0]/[1]`, `c.points` (不是 `c.actor0`/`actor1`):
```python
# 旧: a0, a1 = c.actor0.name, c.actor1.name
# 新: b0, b1 = c.bodies[0], c.bodies[1]; a0, a1 = b0.get_name(), b1.get_name()

# 旧: pen = c.get_depth(0)
# 新: pen = float(pt.get_dist()) if hasattr(pt, 'get_dist') else 0.0
```

### 4.5.3 `compute_reward` 签名修复
```python
# 旧: result["reward"] = compute_reward(result, REWARD_WEIGHTS)
# 新: result["reward"] = compute_reward(result)  # REWARD_WEIGHTS 在函数内部定义
```

### 4.5.4 `_compute_neutral_offsets` 属性初始化
`run_optimize` 中直接调用时属性未初始化, 加 fallback:
```python
def _compute_neutral_offsets(self):
    if not hasattr(self, '_mano_neutral_offset'):
        self._mano_neutral_offset = {}
    # ...
```

**验证**: 以上修复均已在 `--optimize` 模式下验证通过。

---

## 最终实施结果 (已完成)

### CEM 优化结果

```
最优参数: [-0.0015 -0.0012 -0.0028 -0.0009  0.0019  0.0045  0.0035  0.2979  0.5034]
```

**核心发现: 默认参数已经接近最优**

| 参数 | 最优值 | 默认值 | 偏离 |
|------|--------|--------|------|
| grasp_pos_delta | [-0.0015, -0.0012, -0.0028] | [0,0,0] | mm 级 |
| grasp_R_euler | [-0.0009, 0.0019, 0.0045] | [0,0,0] | mm 级 |
| finger_close_target | 0.0035 | 0.0 | 0.0035 (轻微张开) |
| close_blend_ratio | 0.2979 | 0.3 | 几乎不变 |
| transport_vel_limit | 0.5034 | 0.5 | 几乎不变 |

**这意味着**: CEM 优化没有带来显著改善。核心问题不在参数调优,而在 PD 跟随方法本身。

### 已完成的修改

1. **新建 `traj_optimize.py`** — CEM 优化器 + rollout 评估函数 + 多目标奖励函数
2. **`_compute_mano_neutral_target` 参数化** — 接受 `opt_params` 参数, 支持 9 维优化
3. **`_step_gripper_only` 回归纯 PD 控制** — 删除 `set_qpos` 和 `lock_root_pose` hack
4. **`setup_physics_scene` 加 `force_cpu` 参数** — 优化模式直接创建 CPU 场景, 避免 Vulkan 段错误
5. **新增 `--optimize` 和 `--opt-params` 命令行参数**
6. **`run_optimize` 方法** — 无头模式 CEM 优化, 10 轮 × 24 采样 = 240 次 rollout

### 运行命令

```bash
# 离线优化 (无头模式, 约 8 分钟)
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god --grasp-mode hybrid \
    --optimize --num-frames 50

# 使用优化后的参数渲染视频
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
    --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god --grasp-mode hybrid \
    --opt-params output/gripper_only_left/opt_params.npy
```