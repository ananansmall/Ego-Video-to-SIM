# 05_gripper_test.py 开发上下文 (Context)

> 本文档整理 `05_gripper_test.py` 相关的全部对话与开发脉络，作为后续会话的上下文恢复入口。
> 时间范围: 2026-08-28 会话
> 主脚本: `/home/an/robot_world_ws/src/dex-retargeting/example/combination/05_gripper_test.py`

---

## 1. 任务总览

围绕 `05_gripper_test.py` 的多个 test 用例展开，核心脉络：

1. **test8 渲染 + PhysX 驱动管线文档** — 已完成
2. **test8 夹爪眼在手相机 (D405)** — 已完成
3. **阻抗 vs 导纳控制原理讲解** — 已完成
4. **test10 导纳力控抓取** — 已完成（方块升至 23.2cm）
5. **test8 轨迹平滑性分析 + test9 是否适合导纳** — 已分析
6. **test11 单臂机械臂抓取（固定底座 + RelaxedIK）** — 规划中，未实现

---

## 2. 关键文件

| 文件 | 作用 |
|------|------|
| `example/combination/05_gripper_test.py` | 主测试脚本，含 test1~test10 |
| `example/combination/02_render_scene.py` | 单臂渲染参考（浮动臂 + RelaxedIK） |
| `example/combination/docs/pipeline/05_gripper_test.md` | PhysX 驱动管线五层文档 |
| `example/combination/docs/experiments/05_gripper_test_test9.md` | test9 MANO 轨迹跟随开发总结 |
| `example/combination/docs/experiments/05_gripper_test_dev_context.md` | 本文档 |
| `docs/CHANGE_LOG.md` | 变更日志 |
| `GalaxeaManipSim/galaxea_sim/assets/r1/configs/urdfs/r1_v2_1_0_floating_right.urdf` | 浮动右臂 URDF（6 关节 + 二指夹爪） |
| `GalaxeaManipSim/galaxea_sim/controllers/utils/relaxed_ik_solver.py` | RelaxedIK 求解器 |

---

## 3. 已完成工作

### 3.1 test8 渲染 + 管线文档

- 重渲染 test8（`--test 8 --num-frames 800`），输出：
  - `test8_complex_pick_place.mp4` (127KB)
  - `test8_wrist_view.mp4` (369KB，D405 眼在手相机)
- 按用户四节结构重写 `docs/pipeline/05_gripper_test.md`：
  1. `set_drive_target()` — 设置目标位置
  2. `compute_passive_force()` — 计算被动力（重力/摩擦/阻尼）
  3. `set_qf()` — 注入外力
  4. `scene.step() × 16` — 物理子步（DECIMATION=16, PHYSICS_TIMESTEP=1/240s）
- 参数表（已修正）：
  - test8: `VIRTUAL_K=5000, VIRTUAL_D=1000, GRIPPER_K=3000`
  - test9: `K=30000, D=10000, GRIPPER_K=10000`
- 修正：`set_drive_property` 每帧调用 → 仅一次初始化

### 3.2 test8 夹爪眼在手相机

- 移植 test5 的 D405 眼在手相机到 test8
- 挂载点：`right_D405_link`，下倾 35°
- 输出 `test8_wrist_view.mp4`

### 3.3 test10 导纳力控抓取

**目标**：按给定力要求（`F_desired`）运行抓取，验证导纳控制可行性。

**关键常量**：
```python
ADM_GAIN = 0.0008
_GRIPPER_Q_OPEN = 0.025
_GRIPPER_Q_BASE = 0.012
_GRIPPER_Q_CLOSED = 0.0
_ADM_DELTA_MAX = 0.012
_ADM_GRIP_K = 5000.0
```

**力估计函数** `_compute_finger_contact_force`：
- 用 entity id 匹配 cube 与手指 link（`right_gripper_finger_link1/2`）
- 取 `points[].impulse` 峰值 ÷ DT 得力
- 无接触时回退弹簧代理力：`proxy = grip_k * (squeeze_q - q_actual)`

**力控外环**（grasp/transport 阶段）：
```python
F_err = F_desired - F_ext
delta = clip(ADM_GAIN * F_err / 30, -_ADM_DELTA_MAX, _ADM_DELTA_MAX)
q_target = _GRIPPER_Q_BASE - delta
```

**结果**：方块最高 23.2cm（test8 为 21.8cm），力误差均 27-39N，证明导纳力控可行。

**输出**：
- `test10_admittance_grasp.mp4`
- `test10_wrist_view.mp4`

### 3.4 test8 轨迹平滑性分析

**结论**：test8 轨迹规划本身已平滑（min-jerk + Bezier S 曲线），不抖。

- 位置：`_min_jerk` 最小加加速度曲线（C² 连续）
- 角度：`_min_jerk` 插值
- 搬运段：3 段 cubic Bezier 拼接的 S 曲线（C¹ 连续）

**唯一可优化点**：Phase 子段拼接处（0.08 / 0.63 / 0.88）加速度未必连续，可能顿挫。

**test8 驱动刚度偏低**（K=5000/1000/3000），跟踪"肉"、滞后，但非抖动。

### 3.5 test9 是否适合导纳

**结论**：不适合，导纳反而会变差。

| | test8（合成轨迹） | test9（MANO 跟随） |
|---|---|---|
| 目标来源 | 自己生成的平滑曲线 | HaWoR 实时视觉估计（含噪） |
| 主要噪声 | 无 | 大（逐帧 MANO 估计跳变） |
| 当前控制 | 阻抗 K=5000/1000/3000 | 阻抗 K=30000/10000/10000（极硬） |
| 平滑手段 | 轨迹本身平滑 | SG 滤波 + 高斯 Slerp + min-jerk + EMA 四重平滑 |

test9 的问题是**"命令目标本身有噪声"**，解法是"命令侧重平滑 + 高刚度阻抗"。导纳会放大视觉噪声 → 力估计噪声 → 位置漂移。

**导纳的正确场景**：按给定力工艺约束（易碎物体、装配插入、恒力打磨）——即 test10 那种"强制按给定力运行"的场合。

---

## 4. 技术决策

### 4.1 阻抗 vs 导纳

- **阻抗控制**：给定目标位置 → 位置误差 → 力（`F = K(q_t - q) - D*q_dot`）
  - 适合：轨迹精度优先、目标干净
- **导纳控制**：给定目标力 → 力误差 → 运动
  - 适合：力约束优先、易碎/变刚度物体、装配插入
  - 需要力传感器（仿真用 `scene.get_contacts()` 代理）

### 4.2 test10 导纳设计

- 比例外环（非积分器，避免发散）
- 双层力估计（接触点 + 弹簧代理力回退）
- 初始位姿预热（`set_qpos` + 100 步预热，避免首帧扫飞物体）

### 4.3 test11 单臂机械臂规划（未实现）

**目标**：把 test8 的浮动夹爪换成固定底座单臂 + RelaxedIK。

**设计**：
- URDF：`r1_v2_1_0_floating_right.urdf`（6 关节右臂 + 二指夹爪）
- 底座固定：`loader.fix_root_link=True`，底座 `(0.0, 0.0, 0.0)`
- 关节：`right_arm_joint1~6`（6 旋转）+ `right_gripper_finger_joint1/2`
- 控制：RelaxedIK 末端位姿 → 6 关节角，`set_drive_property(K=100000, D=10000)` 高刚度阻抗
- 轨迹：复用 test8 的 4 阶段 progress（高位悬停→接近→抓取→Bezier 搬运→释放→撤退）
- 方块位置：`(0.28, 0.0, 0.025)`，放置 `(0.42, 0.0, 0.025)`

---

## 5. 错误与修复（test10 调试链）

| 错误 | 修复 |
|------|------|
| 手指 link 名错误（`right_finger1_link`） | 改为 `right_gripper_finger_link1/2` |
| `PhysxContact` 无 `impulse` 属性、link entity 共享根 id | 改用 `bodies.get_name()` + `points[].impulse` |
| cube/table `get_name()` 均返回 `''` | 改用 entity id 精确匹配 |
| 导纳积分器发散（79N 误差） | 改比例外环 |
| 出力估计不足（饱和 ~21N） | 双层估计 + 代理刚度提至 5000 + `_GRIPPER_Q_CLOSED=0.0` |
| 姿态角双叠加（`ry + _HOME_RY` 重复） | 移除驱动行多余叠加 |
| 无初始位姿预热（首帧扫飞方块） | 加 `set_qpos` 初始化 + 100 步预热 |

---

## 6. 当前状态

- **test8**：已渲染，含眼在手相机，轨迹平滑性已分析
- **test9**：已开发完成，MANO 轨迹跟随，四重平滑 + 高刚度阻抗
- **test10**：已跑通，导纳力控抓取，方块升至 23.2cm
- **test11**：规划中，单臂机械臂抓取，未实现

---

## 7. 后续待办

1. **test11 单臂机械臂抓取**（用户最新请求）
   - 复用 test8 轨迹规划
   - 浮动夹爪 → 固定底座单臂 + RelaxedIK
   - 底座固定 `(0.0, 0.0, 0.0)`
   - 关节：`right_arm_joint1~6` + `right_gripper_finger_joint1/2`
   - 高刚度阻抗驱动（K=100000, D=10000）

2. **文档补充**（用户尚未确认）
   - 将"阻抗 vs 导纳 + test10 设计说明"补进 `docs/pipeline/05_gripper_test.md`
   - 更新 `docs/CHANGE_LOG.md`（test10 + 眼在手相机条目）

3. **test8 轨迹优化**（可选）
   - Phase 子段拼接处加速度连续性
   - 可把各子段用单一 min-jerk 或 B 样条整体参数化

---

## 8. 关键命令

```bash
# test8 渲染
python example/combination/05_gripper_test.py --test 8 --num-frames 800

# test10 导纳力控
python example/combination/05_gripper_test.py --test 10 --num-frames 800

# test9 MANO 跟随
python example/combination/05_gripper_test.py --test 9 --num-frames 600
```

---

## 9. 参考文档

- `docs/pipeline/05_gripper_test.md` — PhysX 驱动管线五层文档
- `docs/experiments/05_gripper_test_test9.md` — test9 MANO 轨迹跟随开发总结
- `docs/pipeline/04_physics_simulation.md` — 物理仿真基础
- `docs/pipeline/002_render_scene_plan.md` — 渲染场景规划
- `docs/experiments/questions.md` — 问答记录
- `docs/CHANGE_LOG.md` — 变更日志
