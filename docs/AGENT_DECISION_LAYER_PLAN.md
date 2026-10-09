# RAS + Agent 决策层接入方案

> 目标:在 ReplicateAnyScene 已有重建与 SAPIEN 仿真基础上,引入 R2G_Agent 式的**模型闭环决策**,替换现有轨迹回放。
> 硬件基线:R1Lite 双臂(SAPIEN + PhysX)
> 状态:设计文档,未实施。文中所有接口均已对照源码核实。

---

## 1. 现状盘点(已核实)

### 1.1 RAS 已具备的能力

| 环节 | 代码位置 | 状态 |
|---|---|---|
| 场景重建 | `mainv2.py` → `output_v2/<scene>/final_scene.glb` | 已跑通,有多组实测产物 |
| 物体实例 | `final_scene.glb` + `all_instances.pkl` | 已跑通 |
| 手部轨迹 | HaWoR → `hawor_results.npz` | 已跑通 |
| 点追踪 | `object_tracking/02_point_tracker.py` | 已跑通 |
| 抓取时序 | `object_tracking/03_grasp_controller.py` | 已跑通 |
| 轨迹精修 | `object_tracking/04_trajectory_refiner.py` | 已跑通 |
| 闭环验证 | `object_tracking/05_closed_loop_verifier.py` | **有,但语义不同**(见 1.3) |

### 1.2 SAPIEN 接入现状

`object_tracking/simulation/` 三个模块已实现:

- `scene_builder.py` — 建场景、载 GLB、载 R1Lite、坐标系转换
- `action_player.py` — 16 维 action 构造、MANO 轨迹 → EE 轨迹、夹爪时序 → 信号
- `run_simulation.py` — 端到端入口

**坐标系转换已实现**(z-up 房间系 → y-up SAPIEN 系):

```python
T_ROOM_TO_SAPIEN = [[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]
```

物理引擎为 PhysX。碰撞几何用 `add_multiple_convex_collisions_from_file(decomposition="coacd")`,凸分解质量可控。

### 1.3 关键差异:现有"闭环"不是 Agent 闭环

`05_closed_loop_verifier.py` 的 docstring 自述为 **V-Dreamer 思路**:

```
1. 执行仿真 → 获取物体实际轨迹
2. 渲染仿真画面 (从原视频相机视角)
3. 对比仿真画面 vs 原视频画面
4. 计算偏差指标 (物体位置误差 + 视觉相似度)
5. 偏差 > 阈值 → 调整轨迹参数 → 重新仿真
```

这是**用真实视频当参考去调轨迹参数**。与 R2G_Agent 的根本区别:

| 维度 | RAS 现状 | R2G_Agent 目标 |
|---|---|---|
| 动作来源 | HaWoR 轨迹回放 | 模型现场生成 |
| 物体运动 | 轨迹引导 | 纯物理,只靠接触 |
| 参考信号 | 原视频轨迹(特权) | 无特权参考 |
| 决策者 | 偏差阈值调节器 | LLM stage 循环 |
| 失败语义 | 轨迹误差数值 | 任务成功判据 |

**结论:RAS 的重建可复用,动作层需要新建,不能改造。**

### 1.4 GalaxeaManipSim 提供的现成能力

`GalaxeaManipSim/galaxea_sim/` 是完整 RL 基础设施,可直接复用:

- `envs/robotwin/` — 19 个已定义任务环境
- `controllers/` — `bimanual_ee_pose` / `bimanual_joint_position` / `bimanual_relaxed_ik`
- `planners/` — 带运动规划的抓取
- `utils/mplib_utils.py` — MPlib 规划器集成

R1Lite 已具备([r1_lite.py](../GalaxeaManipSim/galaxea_sim/robots/r1_lite.py)):

- **3 个相机**:`head`(1280×720)、`left_wrist` / `right_wrist`(320×240,D405)
- **触觉 link**:4 个 `*_gripper_finger_link{1,2}`
- **IK**:左右臂各一份 `settings_{left,right}.yaml`
- **关节构型**:3 + 6 + 2 / 6 + 2 = 19 DOF,含 torso 与升降

---

## 2. 阻塞项清单

按修复成本排序。

### B1 — `env=None` 导致必崩(代码缺陷)

`run_simulation.py:151` 传入 `env=None`,但 `action_player.py:168` 直接用 `env.step(action)`:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

**结果**:任何走`run_simulation()` 的调用都会 `AttributeError: 'NoneType' object has no attribute 'step'`。

现状说明这条路径**从未被端到端跑通过**。修复方向:改为从 `GalaxeaManipSim` 的 `BimanualManipulationEnv` 构造真实 env,或改成直接驱动 `robot` + `scene.step()`。

### B2 — SAPIEN 及Gymnasium 依赖未安装

本机 `import sapien` 失败。实测 `sys.path` 注入路径**正确**(能解析到 `galaxea_sim` 包),但缺 `gymnasium` 等依赖:

```
ModuleNotFoundError: No module named 'gymnasium'
```

需装SAPIEN 3.x(代码用 `PhysxRigidDynamicComponent` 等 v3 风格 API)+ `galaxea_sim` 声明的其余依赖。建议直接按 [pyproject.toml](../GalaxeaManipSim/pyproject.toml) 装。

### B3 — 感知服务缺失

Agent 决策层需要分割与抓取候选。RAS 侧有 SAM3 分割经验(Stage 1),但需确认是否可服务化复用;GraspNet 类服务需要单独准备。

若第一阶段只用 RGB + 深度(不做文本提示分割),可绕过 B3。

### B4 — 模型 CLI

R2G_Agent 硬编码 Codex CLI 私有 flag(见 `command.py`)。若换用其他模型,需重写 `build_command`,并确认该 CLI 支持:
`exec --ignore-user-config --ignore-rules --ephemeral --skip-git-repo-check -c <k=v> --disable <feature> ...`

### B5 — 任务判据(criteria)

RAS 现有验证是轨迹误差,**不是任务成功判据**。需按任务定义判据(物体最终位姿、是否在容器内、稳定性等),并冻结哈希。

---

## 3. 分阶段实施方案

### 阶段 0:打通现有 SAPIEN 仿真(不引入 Agent)

**目标**:先证明 RAS 重建 → SAPIEN 物理能跑通,拿到基线。这是后续所有工作的地基。

改动:

1. 修 B1(`env=None`)
2. 装B2 依赖
3. 用 `output_v2/` 已有的实测场景跑通:载入 GLB → 加动态物体 → 载入 HaWoR 轨迹 → 回放 → 出视频

验证标准:

- [ ] SAPIEN 场景加载无报错
- [ ] 机械臂按轨迹运动,无 NaN/发散
- [ ] 至少一个物体在物理作用下产生合理位移
- [ ] 渲染出可用视频

**这是本方案的第一个决策点。** 若这里跑不通,后面全部不成立,应先修重建质量。

### 阶段 1:把回放层抽象成可查询接口

**目标**:不改行为,只把 `action_player` 的直连回放改成"状态查询 + 动作执行"分离。为 Agent 层预留入口。

设计:

```
SimBackend(协议)
  reset()                      -> Observation
  step(action)                 -> Observation      # 现有回放逻辑搬进来
  query_state()                -> RobotState       # 关节/位姿/接触力
  render(camera)               -> RGB / Depth
  # 以下为新增,阶段 2 才实现
  set_joint_targets(left,right)
  set_gripper(left,right)
  solve_ik(arm, pos, quat)     -> JointTargets
```

复用 R2G_Agent 的接口形状([batch_backend.py](../../../r2g_review/R2G_Agent/roboagent/backends/batch_backend.py) 的 `run_code` API),但**只借形状,不抄实现**——R2G_Agent 那个是 MuJoCo 专用的。

验证标准:

- [ ] 阶段 0 的回放行为用新接口重跑,结果与修改前逐帧一致
- [ ] `query_state()` 返回值含双臂关节角、EE 位姿、指尖接触力

### 阶段 2:实现双臂 stage 决策循环

**目标**:把 R2G_Agent 的控制器模式移植到 SAPIEN。

移植范围(只取设计,不取代码):

| R2G_Agent 组件 | 文件 | 移植到 |
|---|---|---|
| 决策 schema | `resources/decision.schema.json` | 直接复用 schema |
| stage prompt | `resources/stage_decision.txt` | 改写硬件相关部分 |
| 单会话控制器 | `continuous_policy.py` | 改 CLI 构造 + 工具名 |
| 决策校验 | `policy.py: validate_decision` | 直接复用 |
| 输入边界 | `boundary.py: split_observation` | 直接复用 |

**必须重写**(硬件契约完全不同):

- 后端 API:R1Lite 是 19 DOF 双臂 + torso,R2G_Agent 是 7 DOF 单臂 + 独立夹爪索引
- 机器人契约:禁止沿用 `self.aa = np.arange(7)` / `ga = 7` / `gq = 14` 这类硬编码索引
- 相机:用 R1Lite 的 head + 双腕,替换 `ext1` / `wrist`
- 坐标变换:`room_pose_to_sapien` 已实现,直接用

新增能力(原轨迹回放没有):

- 双臂协调:哪只手主动、哪只手保持
- 阶段划分:接近 → 抓取 → 搬运 → 释放 → 落定
- 每阶段独立步数上限

验证标准:

- [ ] 模型能在**无轨迹参考**下完成任务
- [ ] 决策失败时如实报错,不静默回退
- [ ] 输出视频与真值对比可用人工核查

### 阶段 3:独立评测

**目标**:把成功判据从轨迹误差换成任务判据,且评测器与策略**进程隔离**。

照搬 R2G_Agent 的两条硬规则([batch_profile.py](../../../r2g_review/R2G_Agent/roboagent/backends/batch_profile.py)):

1. 评测器读仿真器特权状态(物体精确位姿、接触力、求解器警告),但**只在执行后运行**
2. 判据文件绑 SHA-256,执行前校验,拒绝篡改

判据设计(以 cup-on-plate 为例):

```json
{
  "task_id": "cup_on_plate",
  "required": ["inside_destination", "supported", "released", "stable"],
  "thresholds": {
    "plate_inner_radius_m": 0.06,
    "target_radius_m": 0.03,
    "support_N": 0.5,
    "release_N": 0.1,
    "speed_m_s": 0.02,
    "settle_s": 1.0,
    "stable_s": 0.5
  }
}
```

验证标准:

- [ ] 判据哈希被篡改时任务失败退出
- [ ] 策略无法读到 `inside_destination` 之类的真值
- [ ] 每个判据有实测数值,不含 `not_evaluated` 伪装成通过

---

## 4. 架构

```
视频
 │
 ├─ VGGT + SAM3 ────────┐
 ├─ HaWoR ──────────────┤
 └─ GLB 合成 ──────────┴→ final_scene.glb + 物体位姿
                          │
                          ▼
                   ┌─────────────┐
                   │ 阶段 0/1 基线 │  SAPIEN 场景 + 轨迹回放
                   └──────┬──────┘
                          │  SimBackend 协议
                          ▼
                   ┌─────────────┐
                   │ 阶段 2 决策层 │  单会话 stage 循环
                   │  (LLM 生成代码)│  25 轮 / 4000 步
                   └──────┬──────┘
                          ▼
                   ┌─────────────┐
                   │ 阶段 3 评测  │  独立进程 + 哈希判据
                   └─────────────┘
```

保留:VGGT / SAM3 / HaWoR / GLB 合成 / 点追踪 / SAPIEN 场景构建
替换:轨迹回放动作层
新建:决策层、任务判据、双臂协调

---

## 5. 不可省的设计约束

从 Real2Gym 规范与 R2G_Agent 实现中提炼,这些是项目可信度的来源:

1. **禁止动作回放冒充策略** — 模型必须自己产生动作,不给它参考轨迹
2. **禁止特权状态泄漏** — 评测真值不回灌给策略
3. **禁止掩盖失败** — 命令成功不等于接触成功;判据要有实测数值
4. **初态显式加载** — 不用物理稳定代替正确初态
5. **禁止辅助执行** — 不加 weld/父子附着冒充抓取
6. **参数来源可追溯** — 质量/摩擦/增益标明是实测、标称、估计还是校准

第 1 和第 3 条是 R2G_Agent 做得最好的地方(它甚至在代码里检查策略是否调用了非预期工具,见 `continuous_policy.py:63`),值得直接继承。

---

## 6. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| SAPIEN API 版本不匹配 | 高 | 先装环境验证 `galaxea_sim` 能import |
| GLB 碰撞质量差 | 高 | coacd 凸分解参数需调;容器类物体需专门处理 |
| 重建物体质量/惯量失真 | 高 | 物理调参会变成主要工作量 |
| 双臂 IK 不可达 | 中 | 沿用 `relaxed_ik` + 拒绝策略 |
| 坐标系三重转换出错 | 中 | RAS 房间系 / HaWoR 伪相机系 / SAPIEN 系,统一在 `scene_builder` 处理 |
| 模型 CLI 不可用 | 中 | 阶段 2 才暴露,可先用脚本控制器占位 |

**最大的不确定性不是代码,是物体重建质量。** 若 GLB 碰撞体不可信,后续调参成本会超过写控制器。

---

## 7. 下一步

立即可做的是**阶段 0**,它不依赖任何未决问题:

1. 装 SAPIEN,确认 `galaxea_sim` 可导入
2. 修 `env=None`(B1)
3. 用 `output_v2/002_C1_Bowl_364deg_vggt_omega` 等已有场景跑通回放

阶段 0 完成后,再根据实测结果决定阶段 1 的接口细节。

若确认要做双模型路线(保留回放基线 + 加 Agent),阶段 0 的回放视频正好可作为阶段 2 的对照组。