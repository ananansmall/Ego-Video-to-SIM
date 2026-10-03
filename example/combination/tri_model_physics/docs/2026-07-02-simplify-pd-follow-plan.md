# 回归 PD 跟随 MANO 位姿曲线 — 简化方案 (第十八轮)

**Goal**: 去掉所有 `set_qpos` / `lock_root_pose` kinematic hack, 回归 "PD 控制跟随 MANO 位姿曲线 + 物理引擎自然算接触力" 的极简方案
**Architecture**: 全部改动在 `grasp_hawor.py` 的 `_step_gripper_only` hybrid 分支 + 主循环 `physics_step` 调用处; 位姿曲线 = MANO (pos, R, j1, j2) + offset
**Tech Stack**: SAPIEN, NumPy

---

## 用户反馈与我的错误认知

**用户的反问 (核心):**
1. "本身肯定是有一个位姿曲线你需要跟随, 难道你想直接启动吗?" — 必须跟随 MANO 位姿曲线, 不能瞬移
2. "为什么会说没有物理仿真呢" — 物理仿真一直在跑 (scene.step 每帧调)
3. "kinematic 推 dynamic 当然有力" — kinematic body 推 dynamic body 有接触力, 这是物理引擎基本行为
4. "你越讲越复杂了" — 回归简单

**我之前的错误:**
| 错误 | 后果 |
|------|------|
| 用 `set_qpos(-0.01)` 强制手指闭合 | 手指变 kinematic, teleport 绕过 PD, 物理引擎接触力不连续 |
| 用 `set_root_pose` 每子步 `lock_root_pose` 锁根 | CLOSE 阶段根完全不动, 物体被压在地面 |
| 解释 "kinematic 与 dynamic 互斥" | 错误理论, 让用户更困惑 |
| CLOSE/TRANSPORT 分离控制 | 方案越来越复杂, 用户难以理解 |

---

## 正确认知 (回归第一性原理)

### 1. 位姿曲线 = MANO 提供的目标轨迹
- 位置: `mano_pos + offset` (常量平移对齐目标, 不偏离轨迹形状)
- 姿态: `mano_R` (CLOSE 阶段可水平化 R_Y, 这是唯一姿态调整)
- 手指开合: `mano_j1` (APPROACH) / `0.0` (CLOSE/TRANSPORT) / `GRIPPER_MAX_OPEN` (RELEASE)

### 2. 物理仿真一直在跑
- `physics_step` 中 `for _ in range(DECIMATION): scene.step()`
- 接触力由 PhysX 求解器计算
- 手指是 dynamic link (PD 驱动), 物体是 dynamic, 接触力自然产生

### 3. kinematic 根推 dynamic 物体有力 (用户对)
- 根是 kinematic-like (`set_root_pose` 驱动, 这是浮动根的标准方式)
- 手指是 dynamic (PD 驱动, `set_drive_target`)
- 手指接触物体 → 物理引擎算接触力 → 物体被推动
- 关键: **手指不能 set_qpos**, 否则变 kinematic, 接触力不连续

### 4. 根为什么必须 set_root_pose
- URDF 根是 free joint (浮动根), 但 R1 robot 的根在地下 z≈-1.0, 不参与物理
- 根由 `set_root_pose` + `set_root_linear_velocity` kinematic-like 驱动 (对齐 04_physics_simulation.py)
- 这不是 "瞬移破坏接触", 而是浮动根的标准驱动方式
- `set_root_linear_velocity` 让根有速度, 摩擦力才能带动物体

---

## 文件改动清单

| 文件 | 改动类型 | 范围 |
|------|---------|------|
| `grasp_hawor.py` L2772-2822 | 重写 | `_step_gripper_only` hybrid 分支: 去掉 set_qpos, 去掉 lock_root_pose |
| `grasp_hawor.py` 主循环 L3630-3640 | 修改 | `physics_step` 调用: 不传 `lock_root_pose` |
| `CHANGE_LOG.md` | 新增 | 第十八轮条目 |

---

## Task 1: 重写 _step_gripper_only hybrid 分支 (核心)

**文件**: `grasp_hawor.py` L2772-2822

**改动**: 删除所有 `set_qpos` 和 `lock_root_pose` 调用, 回归纯 PD + kinematic 根驱动

```python
# 第十八轮: 回归极简 PD 跟随方案 (用户: "位姿曲线需要跟随")
# - 根: set_root_pose (kinematic 浮动根标准驱动) + set_root_linear_velocity (摩擦力生效)
# - 手指: 纯 set_drive_target (PD 驱动, 不用 set_qpos 瞬移)
# - 物理引擎自然算接触力, 物体被推动
prev_pos = getattr(self, '_prev_demo_root_pos', None)
root_pose = sapien.Pose(gripper_pos.tolist(), root_quat.tolist())
robot.set_root_pose(root_pose)

# 根速度 = delta_pos * CONTROL_FREQ, 让摩擦力生效带动物体
if prev_pos is not None:
    root_vel = (gripper_pos - prev_pos) * float(CONTROL_FREQ)
    vel_norm = float(np.linalg.norm(root_vel))
    if vel_norm > 0.5:  # 限幅 0.5 m/s 防止甩飞
        root_vel = root_vel * (0.5 / vel_norm)
else:
    root_vel = np.zeros(3)
robot.set_root_linear_velocity(root_vel.tolist())
robot.set_root_angular_velocity([0.0, 0.0, 0.0])
self._prev_demo_root_pos = gripper_pos.copy()
self._close_lock_pose = None  # 不再使用 lock_root_pose

joint1 = float(gripper_val)
joint2 = float(gripper_val)
if local_idx == 0 or local_idx % 30 == 0:
    logger.info(f"  [neutral][{self.side}] F{local_idx}: phase={phase}, "
                f"pos={gripper_pos.round(3)}, grip_cmd={gripper_val:.4f}")
return (), (joint1, joint2)
```

**关键变化:**
- 删除 `if phase == "CLOSE":` 特殊分支 (不再 set_qpos(-0.01))
- 删除 `elif phase in ("TRANSPORT", "HOLD"):` 特殊分支 (不再 set_qpos 保持)
- 删除 `self._close_lock_pose = root_pose` (不再锁根)
- 全阶段统一: set_root_pose + set_root_linear_velocity + set_drive_target

**验证**: 代码中不再有 `set_qpos` 在 hybrid 分支的调用, 不再有 `_close_lock_pose` 赋值。

---

## Task 2: 主循环 physics_step 不传 lock_root_pose

**文件**: `grasp_hawor.py` L3630-3640 附近

**改动**: 删除 `lock_root_pose=_lock_pose` 参数, 让根自然跟随 set_root_pose + velocity

```python
# 旧:
if self._last_phase == "CLOSE" and hasattr(self, '_close_lock_pose') and self._close_lock_pose is not None:
    _lock_pose = self._close_lock_pose
else:
    _lock_pose = None
physics_step(..., lock_root_pose=_lock_pose)

# 新:
# 第十八轮: 不锁根, 让根自然跟随 set_root_pose + set_root_linear_velocity
# 物理引擎在 8 个子步中自然积分, 手指 PD 接触物体产生摩擦力带动物体
physics_step(..., lock_root_pose=None)
```

**验证**: `physics_step` 调用处不再有 `lock_root_pose` 实参 (或始终为 None)。

---

## Task 3: 运行测试验证

**命令**:
```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py --mode gripper_only --side left \
  --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result \
  --views god --grasp-mode hybrid
```

**成功标准:**
1. ✅ 视频中物体被夹起 (lift > 3cm)
2. ✅ 物体在 TRANSPORT 阶段跟随夹爪移动 (xy_drift > 5cm)
3. ✅ 物体最终在碗附近 (距碗心 < 10cm)
4. ✅ 日志确认: phase=CLOSE 时 grip_cmd=0.0 (PD 目标), 不再 set_qpos
5. ✅ 日志确认: 不再出现 "lock_root_pose" 字样

---

## Task 4: 失败处理 (按可能性排序)

### 4.1 若 PD 闭合不够快 (物体没夹住)
**现象**: CLOSE 阶段接触=0, 物体未被提起
**原因**: stiffness=1000, 8 子步内 qpos 收敛慢
**方案**:
- 优先: 增大 CLOSE 阶段持续帧数 (从 20 → 30 帧, 给 PD 更多时间收敛)
- 次选: 减小 DECIMATION (8→4, 控制更频繁, PD 响应更快)
- 最后: 增大 GRIPPER_STIFFNESS (但用户要求对齐 R1=1000, 谨慎)

### 4.2 若根漂移 (手指到不了物体)
**现象**: set_root_pose 设 z=0.017, 但手指实际 z=0.043
**原因**: 根速度大 + PD 反作用力, 8 子步内根漂移
**方案**:
- 优先: 限制 root_vel 限幅更严 (0.5 → 0.2 m/s)
- 次选: 在 set_root_pose 后立即清零根速度, 只用 set_root_pose 驱动
- 注意: 不要回到 lock_root_pose, 那是 hack

### 4.3 若 TRANSPORT 物体掉落
**现象**: CLOSE 夹住, 但 TRANSPORT 移动时物体滑落
**原因**: 根移动太快, 摩擦力不够
**方案**:
- 优先: 限制 TRANSPORT 阶段 root_vel (0.5 → 0.2 m/s)
- 次选: 增大 GRIPPER_FRICTION (1.0 → 1.5)
- 注意: 不要回到 set_qpos, 那会破坏接触连续性

---

## Task 5: 文档同步

### 5.1 CHANGE_LOG.md
新增第十八轮条目, 记录:
- 用户反馈 (回归 PD 跟随, 去掉 kinematic hack)
- 修改内容 (删除 set_qpos, 删除 lock_root_pose)
- 验证结果

### 5.2 调用 change-log skill
任务结束前调用 change-log skill 输出修改总结。

---

## 执行顺序

1. **Task 1** (重写 hybrid 分支) — 核心, 5 分钟
2. **Task 2** (主循环不传 lock_root_pose) — 配套, 1 分钟
3. **Task 3** (运行测试) — 验证, 3 分钟
4. **Task 4** (失败处理) — 按需, 视测试结果
5. **Task 5** (文档同步) — 收尾, 2 分钟

---

## 核心原则 (本轮)

> **跟随意图, 不要 hack 物理引擎。**
> - MANO 给位姿曲线 → PD 跟随
> - 物理引擎算接触力 → 物体被推动
> - 不要用 set_qpos 瞬移手指 (绕过物理)
> - 不要用 lock_root_pose 每子步锁根 (压制物理)
> - 根用 set_root_pose + set_root_linear_velocity 是标准驱动, 不是 hack
