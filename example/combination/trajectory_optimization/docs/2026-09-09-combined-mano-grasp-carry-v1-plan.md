# MANO 保真抓取与携带统一标准计划（2026-09-10 v6，Stage2 重放方向）

> 文档性质：当前 Stage0–Stage3 的目标标准与测试合同。
>
> - `2026-09-01-combined-mano-pretranslate-v51-plan.md` 是只读历史设计，不修改；
> - `2026-09-04-implementation-change-plan.md` v14 记录实现差异；
> - `2026-09-10-stage2-preprocessed-mano-grasp-replay-plan.md` 是当前 Stage2
>   的详细参数、时间线与验收标准；
> - `2026-09-10-stage2-stage3-physical-test-and-diagnosis.md` 保存旧代码物理失败证据；
> - 本文统一阶段语义、当前参数、碰撞作用域、产物与测试方法。
>
> 代码状态：共享 F50 recipe、scene digest、`stage1_target_isolated` 和接触三分类
> 已经实现，T0 23/23；Stage2 v2 仍为 0/3。用户已决定冻结 Stage1 成功结果，
> 不再重复测试 Stage1，也不以旧 transferable 门阻止 Stage2。下一版 Stage2
> 从预处理 MANO 运行，采用最高 0.12 m 的安全抬升和 F50 后下降—闭合 wrapper。
>
> 本文 v5 中关于“必须重新认证 Stage1、F0-F38 safety_offset 必须为 0、
> q=0.05 必须先静止到旧 Stage1 anchor”的条款保留为历史审计，但不再是当前执行门。
> Stage2 发生冲突时，以 2026-09-10 Stage2 详细计划为准。

---

## 0. v6 统一决定

当前各阶段边界统一为：

| 阶段 | 当前处理 |
|---|---|
| Stage0 | 已通过，预处理算法冻结 |
| Stage1 | v2 artifact 冻结只读，不重跑 CEM/transfer |
| Stage2 | 从预处理 MANO F0 连续执行；负责安全到达、复现抓取末态、携带与 bundle 重放 |
| Stage3 | 只消费已通过 Stage2 的 bundle；Stage2 未通过时拒绝 |

Stage2 的“完全复用 Stage1”现在指：复用目标/场景物理、Stage1 pos/R/q_grasp、
20 点 q 闭合、10 点 settle、PD/force/control rate；不复制 Stage1 的 set_qpos
瞬时初始化。Stage2 可以在 F50 后加入下降和额外保持帧，并让最后 0.03 m 下降
与 20 点闭合同步，以避免张开夹爪直接进入旧 anchor。

当前固定基线参数：

| 参数 | 值 |
|---|---:|
| 最大 world-Z clearance | 0.12 m |
| P0 clearance 范围 | F0-F50 全段固定 +0.12 m |
| pregrasp settle | 5 controls |
| open descent | 45 controls |
| close insertion | 0.03 m / 20 controls |
| Stage1 settle | 10 controls |
| Stage2 grasp confirm | 10 controls |
| moving carry verify | 5 controls |
| Stage2 功能重复 | 至少 2/3 |
| 同 bundle 重放 | 至少 2/3 |

当前核心完成条件是动态目标被抓住、沿预处理 MANO 相对运动移动到 F62，并且同一
bundle 可重复重放。只有该基线通过以后，才按 0.12、0.10、0.08、0.06 的顺序
减小偏移；在此之前不优化 w、不恢复邻物碰撞、不重跑 Stage1。

---

## 1. 愿景与当前优先级

输入 HaWoR/MANO 轨迹和带物理属性的场景，夹爪应当：

1. 尽量跟随统一预处理后的 MANO 夹爪轨迹；
2. 在 F50 以前不接触目标；
3. 在 F50 完整复现 Stage1 的 20 点闭合 + 10 点稳定配方并建立抓取；
4. 抓住后保持夹爪—目标相对位姿，并沿 MANO 运动携带；
5. F90 左右仅在有支撑时释放；
6. 释放后重新并入预处理 MANO 轨迹并到达终点。

验收等级：

| 等级 | 定义 | 当前优先级 |
|---|---|---|
| G1 | F50 建立稳定双侧目标接触 | 必须 |
| G2 | F50 后目标随夹爪沿 MANO 携带到 F90 | 当前核心 |
| G3 | 有支撑时释放，目标稳定，夹爪到达终点 | 后续 |
| G4 | full_robot IK、限位和全场景碰撞可执行 | 最终部署 |

reward 只能给已经通过硬门的候选排序；任何缺帧、NaN、错误碰撞作用域或状态不可复现均为 FAIL。

---

## 2. 7a 左手冻结输入

| 参数 | 标准值 |
|---|---:|
| 工作目录 | `/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization` |
| Python | `/home/an/miniconda3/envs/dex/bin/python` |
| `hawor_dir` | `/home/an/data/hawor/7a` |
| `ras_dir` | `/home/an/data/ras/7_vggt_omega` |
| `side` | `left` |
| `start_frame` | `0` |
| `num_frames` | `113` |
| 目标物体 | `glb_4` |
| `_OBJ_SCALE` | `0.6` |
| `_OBJ_SCALE_TARGET` | `glb_4` |
| `_GRIP_FRICTION` | `2.0` |
| `_SMOOTH_MINT` | `0` |
| 位姿解算 | `svd_palm` |
| `f50_tilt_deg` | `0.0` |

锚点固定为：

```text
F0 ───── F38 ───── F50 ───── F62 ───────── F90 ─── F93 ───── F112
MANO跟随   接近调整    抓取      短携带结束      释放候选  并回MANO     终点
```

这些帧号只适用于当前 113 帧 7a 数据。更换数据必须重新确认，不能把 F50 当成跨数据集常量。

---

## 3. 预处理事实与唯一轨迹来源

### 3.1 统一顺序

```text
MANO joints
  → 每帧 camera-to-world
  → HaWoR world-to-GLB 相似变换
  → joints 序列平滑
  → SVD 夹爪重定向
  → 整条轨迹一次 pre_translate
  → neutral residual
```

Stage1、Stage2、Stage3 必须共享 `_preprocess_mano_traj(side)` 的结果。Stage2/3 不允许重新使用未平移轨迹，也不允许再加第二个大 offset。

### 3.2 2026-09-10 证据

当前 Stage2 物理日志已经证明预处理被执行：

```text
pre_offset = [-0.1213, 0.0494, 0.0058] m
neutral residual after pre_translate = [0, 0, 0]
```

因此旧物理失败不能直接归因为“Stage2 完全没有预处理”。需要分别检查：

- Stage2 是否从预处理数组构造 F0–F38；
- F38–F50 是否被另外的调整路径改坏；
- 夹爪与哪些物体实际建立了碰撞对；
- 接触统计是否把地面和邻物混成目标提前接触。

Stage0 的 `grasp_gripper_only_left_verify.json` 当前记录 `total_contacts=56`，并不是全程零接触。该值来自 Stage0 全局统计，不能与 Stage2 的逐控制点目标接触门直接等价。以后比较必须使用统一 trace：`source_frame/phase/contact_objects/contact_links`。

### 3.3 Stage0 门禁

- F50 finger midpoint 到目标中心误差 ≤1 mm；
- neutral residual 范数 ≤1 mm；
- `R` 有限、正交且 `det(R)≈1`；
- 同输入重复预处理产物一致；
- 保存 `preprocess_traj.npz`，下游只读这套坐标约定；
- Stage0 的接触只作诊断，不等于 Stage2 抓取成功或失败。

Stage0 预处理算法保持不变。Stage1 已在 v2 把入口从 `_DESCEND_OPEN=0.012` 改为 `q_open=0.05` 并导出 recipe；这属于新的 Stage1 revision，不能再描述成“Stage1 未改”。

---

## 4. 当前物理作用域：先验证目标交互

### 4.1 为什么先缩小作用域

旧 Stage2 的 F38–F50 接近同时接触 `ground/glb_7/glb_5/glb_4`。这把三类问题混在一起：目标抓取是否可行、接近轨迹是否合理、邻物碰撞几何是否可信。当前优先验证 G1/G2，因此先隔离夹爪—目标物理交互。

这是用户明确允许的临时“邻物隔离”：当前只关闭目标任务与非目标 GLB 之间的干扰，以最小 MANO 偏离验证抓取和携带。ground 不是“其他物体”，夹爪—ground 与目标—ground 必须继续使用 Stage1 的真实碰撞；后期必须恢复邻物碰撞，不能把本阶段结果当作最终避碰成功。

### 4.2 `stage1_target_isolated` 碰撞合同

这不是删除非目标物体，也不是把它们改成纯视觉体。采用 SAPIEN g0/g1 contact type/affinity：

| 碰撞对 | 是否建立 |
|---|---:|
| 夹爪 ↔ 目标 `glb_4` | 是 |
| 夹爪 ↔ 非目标 GLB | 否 |
| 夹爪 ↔ ground/默认环境 | 是 |
| 目标 ↔ ground/默认环境 | 是 |
| 非目标 GLB ↔ ground/默认环境 | 是 |
| 目标 ↔ 非目标 GLB | 否 |
| 非目标 GLB ↔ 非目标 GLB | 是 |

所以所有物体继续受重力、阻尼、质量、摩擦和地面约束；只隔离非目标物体对当前目标任务的作用。夹爪内部原有 g2/g3 self-collision filter 保留。

每次 Stage2 rollout 必须把如下内容写入 manifest：

```text
collision_scope = stage1_target_isolated
gripper_target_pair = true
gripper_non_target_pair = false
gripper_ground_pair = true
target_environment_pair = true
target_non_target_pair = false
non_target_environment_pair = true
各类别 collision shape 数量
```

如果 trace 仍出现夹爪或目标与非目标物体接触，失败码必须是 `COLLISION_FILTER_LEAK`，不能当成轨迹问题继续调插值。gripper-ground contact 是真实物理结果，不是过滤泄漏；它应独立记录，只有造成穿透、跟踪失败或目标扰动超门槛时，才按对应物理门失败。

### 4.3 恢复全场景碰撞的条件

`stage1_target_isolated` 是目标物理闭环基线，不是 G4 最终标准。只有在 G1/G2 连续 2/3 通过后，才依次恢复：

1. 目标—单个最近邻物；
2. 夹爪—同一最近邻物；
3. 逐个恢复其余邻物，最终恢复全部邻物；
4. full_robot 全链路。

ground 碰撞从本基线开始始终开启。每恢复一级，只允许调整或重新规划 F38–F50 接近路径，不允许改变 F50 Stage1 anchor、夹持力或 F50 20+10 控制合同，以便定位问题来源。

---

## 5. v5 历史时间线（Stage2 已由 v6 详细计划替代）

### 5.1 源帧与控制点不是一一对应

`source_frame` 表示参考 MANO 帧，可以重复或为小数；`control_index` 是实际 30 Hz 命令序号。F50 的 30 个抓取控制点都使用 `source_frame=50`，但物理时间继续前进。

### 5.2 分段职责

| 源帧 | 负责人 | phase | 位姿 | 开度与目标接触 |
|---|---|---|---|---|
| F0–F38 | Stage0 数据，Stage2 重放验证 | `MANO_FOLLOW` | 原样跟随预处理轨迹 | 全开 0.05；禁止目标接触 |
| F38–F50 前 | Stage2 | `APPROACH_ADJUST` | 沿 MANO 平滑加入 Stage1 相对校正 | 全开 0.05；禁止目标接触 |
| F50 前最后控制点 | Stage2 | `FINAL_APPROACH` | 精确到 Stage1 F50 anchor | 全开；静止/低速/无目标接触 |
| F50，20 点 | Stage2，逐值复用 Stage1 | `STAGE1_CLOSE` | 位姿固定 | 使用 Stage1 smoothstep 从 0.05 闭合到 `q_grasp` |
| F50，10 点 | Stage2，逐值复用 Stage1 | `STAGE1_SETTLE` | 位姿固定 | 保持 `q_grasp`，要求稳定双侧目标接触 |
| F50→F51，5 点 | Stage2 | `CARRY_VERIFY` | 立即边走边抓 | 保持 `q_grasp`，至少 4/5 双侧 |
| F51–F62 | Stage2 | `CARRY` | MANO 相对运动 + 固定 Stage1 校正 | 始终保持抓取 |
| F62–F90 | Stage3A | `STAGE3_CARRY` | 精确继承 Stage2 末态，继续 MANO | 始终保持抓取 |
| F90 左右 | Stage3B | `PRE_RELEASE/RELEASE` | 只有支撑检查通过才固定并张开 | 最多 5 点张开至 0.05 |
| F91–F93 | Stage3B | `POST_RELEASE_BLEND` | 3 帧消除携带校正 | 全开，不再接触目标 |
| F94–F112 | Stage3B | `MANO_FOLLOW` | 预处理 MANO 原轨迹 | 全开到终点 |

Stage2 的空间职责严格是 F38–F62，但其可执行 bundle 包含 F0–F38 前缀，目的是验证它没有通过“瞬移到 F38”绕过真实历史状态。

---

## 6. v5 Stage1 严格迁移合同（历史审计，不再是 Stage2 前置门）

Stage1 输出必须区分五层结果，不能再用一个“3/3 PASS”同时代表它们：

| 层级 | 回答的问题 | 当前结果 |
|---|---|---|
| `legacy_local_lift_pass` | 历史门 `lift_final>=3cm` 是否通过 | v1 3/3、v2 3/3，证据保留 |
| `entry_clearance_pass` | q=0.05 到达该位姿时，目标是否保持 reset 状态且无 overlap | v1 FAIL、v2 FAIL |
| `grasp_establish_pass` | 20+10 是否建立双侧且 reset→settle 位移≤2mm | v1/v2 未通过新门 |
| `local_carry_pass` | Stage1 竖直提升中是否保持双侧和完整 `T_GO` | 旧验证未按三次完整记录，待验证 |
| `transfer_equivalence_pass` | 规范入口状态与 20+10 是否共同可重复 | v1 0/3；v2 工具报 3/3，但严格审计为 0/3 |

输出必须至少包含：

```text
pos, R, gripper_qpos, enclose_z, force_limit
side, target_object, object_scale, grasp_frame
物理配置、碰撞几何与预处理 hash
legacy-lift/entry/establish/local-carry/transfer 独立验证结果
legacy_local_lift_pass, entry_clearance_pass, grasp_establish_pass
local_carry_pass, transfer_equivalence_pass, transferable
```

Stage2 只能加载 `transferable=true` 的 Stage1 artifact，并精确复用其 entry/local primitive、`pos/R/gripper_qpos` 和 20 close + 10 settle。新标准中 `transferable = entry_clearance AND grasp_establish AND local_carry AND transfer_equivalence`。禁止把 `legacy_local_lift_pass=true` 当成可迁移证明。若 entry anchor 或抓取 primitive 修改，生成新的 Stage1 revision 并重新验证受影响的门。

`enclose_z` 只能作为旧 artifact 的一致性诊断字段，不能覆盖已验证的 `pos.z`；字段缺失时尤其不能默认成 0。

当前统一物理参数：

| 参数 | 值 |
|---|---:|
| physics timestep | 1/240 s |
| control frequency | 30 Hz |
| decimation | 8 |
| gravity | `[0,0,-9.81] m/s²` |
| virtual base K/D | 50000 / 600 |
| finger K/D | 1000 / 200 |
| finger force limit | 100 N |
| friction | 2.0 |
| Stage2 full-open q | 0.05 m |

Stage2 不允许在 F50 后偷偷提高 K/D/force，也不允许 runtime clamp 伪造保持。

### 6.1 “完全复用 Stage1”的边界

Stage1 是局部可行性试验，会用 `set_qpos` 直接初始化到 F50 抓取位姿；这次瞬移不能成为 Stage2 可执行轨迹的一部分。完整复用的准确含义是：

- Stage2 负责新增 F0→F50 的连续到达过程；
- 到达 F50 后，`pos/R/q_grasp`、20+10 命令序列、控制频率、PD/force、robot/scene/ground、目标物理属性和初始状态全部来自 Stage1 同一 revision；
- 只有“目标任务与非目标 GLB 的碰撞对关闭”列入差异白名单；
- F50 后 Stage2 才新增 `CARRY_VERIFY/CARRY`，并始终保持 Stage1 的 `q_grasp`。

Stage1 artifact 必须保存、Stage2 manifest 必须比较：

```text
stage1 revision / exported anchor hash
close_controls=20, settle_controls=10
close_profile=cubic_smoothstep: t=linspace(0,1,20), s=t²(3-2t)
q_open, q_grasp, pos, R
physics_hz, control_hz, decimation, gravity
virtual-base K/D, finger K/D/force_limit
target name/scale/mass/inertia/friction/collision-shape digest
ground pose/material, solver/contact parameters
all-object initial pose/velocity digest, scene asset digest
```

recipe 必须在 Stage1 导出时用规范化 `pos/R/q_grasp` 计算 `anchor_hash`；Stage2 从加载值重算并比较。当前实现只在 Stage2 内现算 contract hash、recipe 本身未绑定 pos/R，不足以证明完整 anchor 未漂移。比较结果保存为 `stage1_reuse_report`；除非目标隔离白名单外，只要有一个字段不同，就以 `STAGE1_REUSE_MISMATCH` 停止，不能用后续 reward 或抓取结果覆盖。

### 6.2 F50 入口状态等价门

旧 v1 Stage1 以 `_DESCEND_OPEN=0.012` 起步；v2 虽已改成命令 q=0.05，但 T1 在检查入口前推进了 5 个物理 warmup，目标已经被推走约 44.9 mm。随后“当前无 contact”不能证明初始入口合法。

在 Stage2 前必须执行 `F50_TRANSFER_EQUIVALENCE`：

```text
reset 到与 Stage1 相同 scene/object/robot 状态
→ base 设为 Stage1 pos/R
→ actual/command q=0.05，robot/object 速度=0
→ 在目标仍处于 reset pose 时验证静态 overlap=false
→ warmup 全程目标接触=0，reset→entry 漂移 <=2mm/2deg
→ 验证 base <=2mm/2deg、q max error <=3mm、robot/object 速度门
→ 调用 Stage1/2 共用的 20 close + 10 settle recipe
→ reset→settle 目标位移 <=2mm
→ 三次独立 reset 至少两次通过 Stage1 双侧接触/保持门
```

单次 PASS 是上述条件的逻辑 AND；未采集的指标按 FAIL。`object_drift_ok` 必须参与 `run_pass`，不能只记录。该测试禁止 `_DESCEND_OPEN=0.012` 的隐藏初态，也禁止先用 warmup 推开目标再检查。通过后，Stage1 anchor 才标记为 `transferable=true`。

Stage1 的 `grasp_establish_pass` 要求 settle 双侧 ≥8/10、末 3/3 双侧且 reset→settle 位移 ≤2 mm。`local_carry_pass` 必须在竖直提升段报告完整 `T_GO` 平移/旋转漂移 ≤5 mm/5°、双侧比例 ≥0.8 和目标实际进度；旧的单一 fingertip-distance 与 `lift_final>=3cm` 只能作为兼容诊断，不能替代这组门。

### 6.3 什么时候需要重测 Stage1

旧成功永远可以作为对应历史协议的证据保留。只有以下情况使 Stage1→Stage2 资格失效并要求新 revision：

- Stage1 entry/anchor/primitive 或 20+10 recipe 改变；
- target collision geometry、scale、pose、mass/inertia/material 改变；
- ground、solver/contact、PD/force、control/physics timestep 改变；
- artifact 缺 recipe/digest/transferable 字段；
- 严格 transfer gate 未通过。

其他情况下 Stage2 应直接复用冻结 artifact，不得每次重新跑 CEM。当前 v1/v2 都保留，但都应标记 `transferable=false`。

---

## 7. v5 Stage2 轨迹构造（已由 0.12 + F50 wrapper 替代）

### 7.1 F0–F38

直接复制已对齐的：

```text
preprocessed_pos[0:39]
preprocessed_R[0:39]
```

这一段 `safety_offset=0`，不得使用 Stage1 correction；夹爪保持全开。

### 7.2 当前优先项：目标碰撞几何审计

v2 的确定性结果是 Stage2 `prefix/adjust=15/24`，而 Stage1 v2 候选在 q_open 入口也接触 `glb_4`。在生成新 anchor 或修改轨迹前，必须核查当前 dynamic bbox box collision 是否相对 visual mesh 包大、错位或 margin 过大。

审计必须覆盖 visual/collision AABB 与 pose、夹爪各 link/shape、首个 contact frame、point/normal/separation/impulse，以及**任何 physics step 之前**的静态 overlap/distance。结论只能二选一：

- `FALSE_COLLISION_GEOMETRY`：修正 `glb_4` collision representation，并因 physics digest 改变重新生成 Stage1 revision；
- `REAL_PATH_OVERLAP`：保留碰撞几何，将 q_open clearance 和 reset→entry drift 变成 Stage1 硬约束；若单一 pose 无解，Stage1 输出可复用的局部 `pregrasp→insertion→20+10` primitive。

禁止将 prefix contact 改为 report-only。目标是 dynamic actor，提前接触已经造成 37–45 mm 级位移；放宽日志门不会得到正确的 F50 初态。

### 7.3 F38–F50 MANO 保真平滑校正

旧 `-anchor_R[:,0]` retreat 和 world-Z“抬升—水平—下降”都已废止。前者朝邻物方向移动，后者在邻物碰撞已经关闭时仍会无理由抬高轨迹，都会破坏 MANO 保真。

当前确定性路径为：

```text
delta_p = p_stage1 - p_mano[F50]
delta_R = R_stage1 · R_mano[F50]^T
alpha[f] = quintic_smoothstep((f-38)/(50-38))

p_cmd[f] = p_mano[f] + alpha[f] · delta_p
R_cmd[f] = Slerp(I, delta_R, alpha[f]) · R_mano[f]
```

它满足：F38 不改变；F50 精确命中通过入口硬门的 Stage1 entry anchor；中间继续跟随预处理 MANO 的逐帧变化；校正速度与加速度在两端为零。当前 v2 anchor 未达到该前提，暂时不得作为正式终点。

不允许额外构造 z 高台。只有 `p_cmd.z < approach_min_z` 时才施加：

```text
p_cmd.z = approach_min_z
safety_offset.z = approach_min_z - original_p_cmd.z
```

默认参数：

| 参数 | 默认 | 可调整范围 | 说明 |
|---|---:|---:|---|
| `approach_min_z` | 0.0 m | 场景地面高度 | 只阻止进入 z<0，不主动抬升 |
| `approach_time_scale` | 2.0 | 1.0–4.0 | 只改变控制采样密度，不改变空间路径 |
| `q_start` | 0.050 m | 固定 | F50 前不允许 `q_near`/预闭合 |

CLI 使用 `--stage2-approach-min-z 0.0`。`--stage2-pregrasp-retreat`、`--stage2-approach-clearance-z` 和 `--stage2-q-near` 均已废止。

### 7.4 F50–F62 边走边抓

- F50 恰好逐值复现 Stage1 的 `20 STAGE1_CLOSE + 10 STAGE1_SETTLE`，共 30 个控制点；
- 这 30 点在 30 Hz 下占 1.0 s；本阶段接受 F50 的局部时间扩张以换取已验证抓取复用，但 base spatial anchor 不得漂移；
- 第 31 个控制点必须已经朝 F51 移动；
- F50→F51 插入 5 个移动确认点，而不是静止等待；
- F51–F62 使用 `MANO[f] + delta_p` 与 `delta_R·MANO_R[f]`；
- correction 在 Stage2/Stage3 携带期间保持，不在 F55–F62 淡出，否则会把夹爪从物体上拉走；
- `carry_time_scale=2.0` 默认只放慢时间，不改 MANO 空间形状。

如果以后要求缩短 F50 停留，必须先由 Stage1 使用新时长完成 2/3 物理重验并产生新 revision；Stage2 不能单独压缩。

---

## 8. v5 目标函数说明（仅作历史参考）

统一权重列：

```text
[mano_pose,
 avoid_target_contact,
 maintain_bilateral_contact,
 maintain_tgo,
 target_carry_progress]
```

| phase | 权重 | 语义 |
|---|---|---|
| `MANO_FOLLOW/APPROACH_ADJUST/FINAL_APPROACH` | `[1,1000,0,0,0]` | 还没抓，目标接触是错误 |
| `STAGE1_CLOSE` | `[0.2,0,100,100,0]` | 允许按 Stage1 配方建立接触，降低位姿强跟踪权重 |
| `STAGE1_SETTLE/CARRY_VERIFY/CARRY/STAGE3_CARRY` | `[1,0,1000,1000,1000]` | 已抓住，必须保持双侧、T_GO 和物体进度 |
| `RELEASE` | `[0.5,0,0,0,0]` | 主动解除抓取 |
| `RELEASE_SETTLE/POST_RELEASE` | `[0.5或1,1000,0,0,0]` | 张开后不再接触目标 |

非目标碰撞不进入当前轨迹代价，因为碰撞对本身已关闭；若出现非目标 contact，属于配置泄漏，直接硬失败。

这些权重必须逐控制点写入 bundle，并参与 `bundle_hash`。当前确定性实现中它们首先是相位合同和后续优化接口，不得声称已经有优化器在真正最小化所有列。

---

## 9. v5 Stage2 物理门禁（已由 P0/P1 两级门替代）

### 9.1 Reset 与 F50 前

每次独立重复必须恢复：所有 GLB pose、线/角速度、robot qpos/qvel、root pose/velocity 和 drive target。

进入本节前还必须满足：artifact 的 `entry_clearance_pass/grasp_establish_pass/local_carry_pass/transfer_equivalence_pass/transferable` 全为 true，`stage1_reuse_report` 只有非目标隔离白名单差异，且严格 `F50_TRANSFER_EQUIVALENCE` 至少 2/3 PASS。否则 Stage2 直接拒绝运行。历史 `legacy_local_lift_pass` 单独不能晋级。

| 指标 | 门槛 |
|---|---:|
| reset→warmup 目标平移 | ≤2 mm |
| reset→warmup 目标旋转 | ≤2° |
| F50 前目标接触控制点 | 0 |
| F50 前目标平移漂移 | ≤2 mm |
| FINAL_APPROACH base position error | ≤2 mm |
| FINAL_APPROACH rotation error | ≤2° |
| FINAL_APPROACH linear speed | ≤0.01 m/s |
| FINAL_APPROACH angular speed | ≤5°/s |
| gripper/target ↔ 非目标 contact | 0；否则 `COLLISION_FILTER_LEAK` |
| gripper ↔ ground contact | 独立记录；不是 filter leak，不要求人为清零 |

F0–F38 与 F38–F50 的接触计数必须分开报告。这样能区分“预处理原轨迹已经碰目标”和“Stage2 调整路径造成提前接触”。

### 9.2 抓取与短携带

| 指标 | 门槛 |
|---|---:|
| F50 20+10 命令复用 | 与 Stage1 recipe 逐值一致 |
| F50 settle 末段 | 双侧目标接触稳定，门槛与 Stage1 同 revision 一致 |
| F50→F51 `CARRY_VERIFY` | ≥4/5 双侧目标接触 |
| CARRY_VERIFY `T_GO` 平移最大漂移 | ≤5 mm |
| CARRY_VERIFY `T_GO` 旋转最大漂移 | ≤5° |
| CARRY_VERIFY 相对速度 | ≤0.02 m/s |
| F51–F62 双侧比例 | ≥0.8 |
| F51–F62 `T_GO` 平移/旋转漂移 | ≤5 mm / 5° |
| 目标沿命令方向进度 | `≥max(20 mm, 0.5×commanded_progress)` |
| base tracking mean/max | ≤5 mm / 15 mm |
| 独立重复 | 至少 2/3 PASS |

只有通过结果可以保存为 `stage2/stage2_bundle.npz`。最佳失败结果不得成为 Stage3 父输入。

优先按首个 failure 处理：

```text
STAGE1_REUSE_MISMATCH → 恢复 Stage1 同一场景/物理/配方，不开始 rollout
ENTRY_CLEARANCE_FAIL / F50_TRANSFER_FAIL → 静态 collision audit；不运行 Stage2
NO_TRANSFERABLE_ENTRY → 明确报告无解，不保存最佳失败 anchor
COLLISION_FILTER_LEAK → 修非目标碰撞配置，不调轨迹
INITIAL_TARGET_CONTACT / RESET_DRIFT → 修 reset 或首帧状态
APPROACH_TARGET_CONTACT → 先定位 collision shape/link；真实接触做最小 MANO 偏离重规划
PRE_F50_READY → 调 time scale/控制跟踪，不改 F50 anchor
PROVISIONAL_GRASP → 先做 F50 transfer-equivalence，核对 Stage1 20+10 合同
CARRY_VERIFY / CARRY → 调 carry time scale，检查 T_GO 与夹持力
TRACK → 修控制跟踪
```

---

## 10. Stage3 复用与释放

### 10.1 Stage3A

Stage3 必须先从共同 reset 正式重放相同 Stage2 executable bundle；hash、物理 manifest 或正式重放 gate 不通过就停止。

F62–F90：

- 保持 Stage2 末端 `q_grasp`；
- 固定 Stage2 末端相对 MANO 的 `delta_p/delta_R`；
- 只允许搜索 `stage3_carry_time_scale`，默认 2.0；
- CEM 继续禁用，空间路径不得被无效参数覆盖；
- 持续验证双侧接触、`T_GO` 和任务进度。

### 10.2 Stage3B

F90 是候选，不是强制释放。释放前必须满足：

- 目标有非夹爪支撑；
- 目标底面到支撑 gap ≤5 mm；
- 目标线速度 ≤0.02 m/s；
- 目标角速度 ≤5°/s；
- `T_GO` 漂移仍合格。

不满足时 Stage3A 可以 PASS，但 Stage3B 返回 `NO_SUPPORT`，继续保持抓取，不能悬空张开。

满足时：5 个 `RELEASE` 控制点张开到 0.05，观察 10 个 settle 控制点；F91–F93 用 3 帧淡出携带 correction，F94–F112 精确跟随预处理 MANO 到终点。

---

## 11. MANO 保真指标

允许一次全局 pre_translate 和抓取后固定 Stage1 correction，因此比较以锚点为中心的运动形状：

```text
e_p[f] = ||(p_cmd[f]-p_cmd[anchor]) - (p_mano[f]-p_mano[anchor])||
e_R[f] = geodesic(R_cmd[f] R_cmd[anchor]^T,
                  R_mano[f] R_mano[anchor]^T)
```

每个正式 artifact 至少报告：

- approach 与 carry 的 `e_p mean/max/p95`；
- `e_R mean/max/p95`；
- 命令/MANO 路径长度比；
- 速度方向余弦；
- 最大 safety offset；
- source frame 到 control index 映射；
- Stage1 anchor 相对 MANO F50 的 `delta_p/delta_R`。

排序顺序：硬门全部通过 → repeat 成功率 → `T_GO` 漂移 → MANO 误差 → jerk。不能用更接近 MANO 抵消没有抓住。

---

## 12. 可执行产物合同

Stage2/3 bundle 至少包含：

```text
source_frame[N]
phase[N]
base_pos[N,3]
base_R[N,3,3]
gripper_q[N,2]
control_dt[N]
expected_contact_policy[N]
objective_weight_names[5]
objective_weights[N,5]
safety_offset[N,3]
grasp_anchor
timeline_meta
stage1_contract_hash
physics_hash
manifest_hash / manifest_json
bundle_hash
```

Stage2 trace 还要保存：目标 pose、夹爪 pose、`T_GO`、实际/命令 q、双侧接触、`target_contact/ground_contact/non_target_contact`、接触对象/links 和力。禁止继续用一个 `environment_contact` 同时表示 ground 与非目标物体；碰撞过滤审计必须扫描全局 contact pair，不能只扫描夹爪 link，否则看不到 target—non-target 泄漏。Stage3 combined bundle 的 Stage2 前缀必须逐字节等于父 bundle。

---

## 13. v5 测试命令（历史记录，不再直接执行）

当前已知结果：T0 23/23；v1 local 3/3 但 transfer 0/3；v2 local 3/3，但严格 entry/transfer 审计为 FAIL；Stage2 0/3 `APPROACH_TARGET_CONTACT 15/24`；Stage3 正确拒绝。下面是下一 revision 的执行顺序，不把旧工具返回的 T1 3/3 当作有效通过。

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization

export PY=/home/an/miniconda3/envs/dex/bin/python
export _OBJ_SCALE=0.6
export _OBJ_SCALE_TARGET=glb_4
export _GRIP_FRICTION=2.0
export _SMOOTH_MINT=0

# 1. 保持现有 23 个测试，并增加严格 entry/transfer 回归测试
$PY -m pytest -q tests/test_stage2_timeline.py tests/test_stage3_timeline.py

# 2. Stage0：算法不变；新输出目录只确认预处理 hash
$PY grasp_hawor.py --test-stage0 --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3

# 3. 先对 v2 做只读静态碰撞几何审计；该 CLI 需实现
$PY grasp_hawor.py --test-stage1-contact-audit --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --output-dir output/hawor7a_left_stage1_target_isolated_v2

# 4. 只在审计后完成 collision 或 entry/primitive 单一分支时生成 Stage1 v3
$PY grasp_hawor.py --test-stage1 --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3

# 5. 严格 F50 迁移等价验证
$PY grasp_hawor.py --test-stage1-transfer --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3

# 6. 只有 transferable=true 后才运行 Stage2
$PY grasp_hawor.py --test-stage2 --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --stage2-approach-min-z 0.0 \
  --stage2-approach-time-scale 2.0 \
  --stage2-carry-time-scale 2.0 \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3

# 7. Stage3；Stage2 未达到 2/3 时必须自动停止
$PY grasp_hawor.py --test-stage3 --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --stage2-approach-min-z 0.0 \
  --stage2-approach-time-scale 2.0 \
  --stage2-carry-time-scale 2.0 \
  --stage3-carry-time-scale 2.0 \
  --stage3-release-candidate 90 \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3
```

第一轮 Stage2 不做空间参数网格；先检查日志必须出现：

```text
[pre_translate] ...
[_compute_neutral_offsets] ... offset=[0. 0. 0.]
[Stage2 collision scope] stage1-target-isolated ...
[Stage2 reuse] mismatch=0 except non-target collision whitelist
artifact transferable=true
strict transfer reset-to-entry drift/base/q/speed gates PASS >=2/3
F0-F38 phase=MANO_FOLLOW
F38-F50 phase=APPROACH_ADJUST/FINAL_APPROACH
prefix/adjust target contact = 0/0
F50 exactly 20 STAGE1_CLOSE + 10 STAGE1_SETTLE, equal to Stage1 recipe
repeat gate >= 2/3
```

如果仍是 `APPROACH_TARGET_CONTACT`，继续看具体 link/shape/contact point，不能改为 report-only，也不能通过增加任意 z clearance 绕过。只有提前接触为 0 后，才处理 `PRE_F50_READY/TRACK`，并且一次只改 `approach_time_scale=3/4`。不要把插值、time scale、力、q、F50 帧数一起搜索。

---

## 14. v5 禁止项（与 v6 冲突的条款已撤销）

- 修改 2026-09-01 历史计划；
- 把 Stage0 全局接触数当作 Stage2 无提前目标接触证明；
- F50 前预闭合或接触目标；
- warmup 推动物体后再把“当前无接触”判为合法入口；
- 忽略 `object_drift_ok=false`、base/q error 或缺失速度指标后仍报告 transfer PASS；
- 把动态目标的 prefix contact 降级为 report-only；
- 因为要接口复验而删除/否定旧 Stage1 legacy local-lift 成功证据；
- 把 Stage1 的 20+10 抓取配方压缩、延长或另写一套闭合曲线；
- Stage2 隐式修改 Stage1 `dz/dq/dR/force`；
- F50 后让 correction 淡出造成脱手；
- Stage3 重新生成另一条 Stage2 前缀；
- 保存未通过物理门的正式 bundle；
- 在非目标隔离作用域下宣称已经完成全场景避碰或 full_robot 验收；
- 同时改变物理参数和轨迹参数后仍复用旧 artifact。

---

## 15. v5 晋级顺序（已由 v6 T0-T4 替代）

```text
Stage0 preprocess PASS
  → 保留 Stage1 legacy local lift 证据
  → collision geometry audit 给出单一结论并完成对应修复
  → Stage1 q=0.05 静态入口 clearance/reset-drift hard gate PASS
  → Stage1 grasp_establish + local_carry PASS
  → 严格 F50_TRANSFER_EQUIVALENCE PASS ≥2/3，artifact transferable=true
  → Stage2 stage1_target_isolated 目标抓取+短携带 PASS ≥2/3
  → Stage3A stage1_target_isolated 全程携带 PASS ≥2/3
  → Stage3B 条件释放
  → 分级恢复目标/夹爪与邻物的碰撞（ground 始终开启）
  → full_robot
```

当前下一步唯一目标是：**先完成静态碰撞几何审计并修正严格入口门**。v1/v2 的 legacy local-lift 成功都保留，但二者都不是 transferable revision。只有新 revision 在目标未被提前推动的情况下通过 entry/establish/local-carry/transfer，才重新运行 Stage2；随后依次要求 prefix/adjust=0/0、F50 20+10 抓住、F50–F62 实际移动 `glb_4`。这以前不继续用 `w`、CEM、释放或邻物优化掩盖根因。
