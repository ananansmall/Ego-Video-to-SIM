# Do-as-I-do Retargeting 参考分析与 Stage4 修订建议

> 日期：2026-09-09  
> 面向计划：`2026-09-09-combined-mano-grasp-carry-v1-plan.md`  
> 参考项目：`/home/an/robot_world_ws/src/do-as-i-do/retargeting`

## 1. 结论

当前计划中**仍然有 Stage4**，但它属于 `TARGET`，不是当前已完成、已验证的生产阶段。

现有计划对 Stage4 的定义是：

- 当最终任务只要求验证夹爪能否抓取和携带时，Stage0–Stage3 可以先在 `gripper_only` 下建立物理闭环；
- 当结果需要由完整机器人执行或部署到真实机器人时，Stage4 必须执行；
- 当前代码中存在 `full_robot` 模式，不代表它已经满足 Stage4 验收要求。

因此建议将原来的单一 Stage4 拆成：

```text
Stage4A full_robot kinematic validation
  输入：Stage3A/Stage3B 已通过的 executable bundle
  输出：整机关节轨迹、IK/限位/碰撞报告

Stage4B full_robot dynamic replay
  输入：Stage4A 整机关节轨迹 + 同一动态物体场景
  输出：动力学实际轨迹、接触与抓取携带指标
```

晋级语义应写成：

```text
Stage3A PASS
  → G2 gripper-level grasp-and-carry PASS

Stage4A PASS + Stage4B PASS
  → full-robot executable PASS

Stage4A/4B 未运行
  → 不得声称整机可执行或可部署
```

## 2. 为什么当前 full_robot 不能直接视为 Stage4 完成

当前 SAPIEN 管线的 `full_robot` 路径仍包含以下工程妥协：

1. 浮动根可能通过 `set_root_pose` 在每个物理子步前后直接设置；
2. 非夹爪 link 的大部分碰撞被关闭；
3. robot link 重力被关闭；
4. Stage0–Stage3 的 gripper-only 成功没有自动证明真实机械臂可达；
5. 尚未形成“整机 IK → 关节限位 → 全身碰撞 → 动力学重放 → 重新验证物体携带”的闭环。

正式 Stage4B 必须满足：

- 机器人基座固定在真实安装位姿；
- 禁止通过 root teleport 跟踪末端轨迹；
- 只允许实际机械臂和夹爪执行器驱动；
- 启用机械臂、夹爪、环境和物体的真实碰撞；
- 保留全部关节位置、速度、加速度和力矩/力限制；
- 把物体作为动态自由体重新仿真；
- 重新执行双侧接触、`T_GO`、物体进度、跟踪误差和扰动恢复门禁。

## 3. Do-as-I-do retargeting 的实际结构

该项目的 retargeting 流程为：

```text
重建手/物轨迹
  → 重力对齐与异常帧修复
  → 物体 CoACD 凸分解
  → 生成机器人手 + 动态物体 MuJoCo 场景
  → mink 手腕/指尖 IK
  → 运动学轨迹前向动力学检查
  → MuJoCo Warp 并行物理优化
  → trajectory_mjwp.npz
```

值得借鉴的重点不是它的具体增益或摩擦数值，而是：

- 数据修复保持手和物体时间一致；
- 可视网格与物理碰撞网格分离；
- IK 与动力学验证分离；
- 从参考控制出发优化小残差，而不是任意重写轨迹；
- 并行扰动 rollout 检验抓取鲁棒性；
- 保存实际执行的 `qpos/qvel/ctrl/time`，而不只保存命令轨迹。

## 4. 对 Stage0 的参考意见

### 4.1 增加共享异常帧修复

Do-as-I-do 在手或物体任一跟踪结果出现缺帧、负深度或速度尖峰时，使用共享 mask 同时修复手和物体，避免只平滑其中一方后破坏相对运动。

Stage0 产物建议增加：

```text
repair_mask[N]
repair_reason[N]
mano_joints_raw[N,21,3]
mano_joints_repaired[N,21,3]
object_pose_raw[N,7]
object_pose_repaired[N,7]
```

新增门禁：

- 修复帧比例必须报告；
- 抓取锚点附近禁止跨越长缺失区间插值；
- 平移使用线性插值，旋转使用 SLERP；
- 修复后手—物相对位姿不得产生新的跳变；
- 原始数据和修复数据必须同时保留。

### 4.2 冻结的是参考，不是物理物体

Do-as-I-do 会在参考轨迹被判断为尚未抓取或已经释放时，将物体参考位姿冻结，防止优化器为了追逐重建噪声而提前推动物体。

这个原则可以借鉴，但必须明确：

- 只允许冻结 `object_reference_pose`；
- 仿真中的动态物体仍由重力、接触和惯性决定；
- 不允许每帧把模拟物体 `set_pose` 到参考位姿。

## 5. 对碰撞与惯性资产的参考意见

### 5.1 使用双层碰撞模型

当前管线对动态目标主要使用 bbox box collision。建议引入：

```yaml
collision:
  fast_model: bbox
  validation_model: coacd
  coacd:
    thicken_m: 0.002
    dilate_m: 0.002
    max_hulls: null
```

建议流程：

- Stage1 大规模候选预筛选可使用 bbox；
- Top-K 物理验证使用 CoACD；
- Stage2、Stage3 和 Stage4B 正式门禁必须使用 CoACD；
- bbox 与 CoACD 结论不一致时不得晋级；
- 保存原 mesh hash、collision mesh hash、缩放和分解参数。

注意：Do-as-I-do 的 2 mm thickening/dilation 对小物体可能过大，必须做单变量灵敏度测试，不能直接照抄。

### 5.2 物理 manifest 补充项

计划中的 physics 配置建议增加：

```yaml
physics:
  object_static_friction: 2.0
  object_dynamic_friction: 2.0
  gripper_static_friction: 2.0
  gripper_dynamic_friction: 2.0
  friction_combine_mode: null
  object_restitution: 0.0
  gripper_restitution: 0.6
  object_linear_damping: 0.5
  object_angular_damping: 50.0
  robot_gravity_enabled: false
  object_gravity_enabled: true
  collision_model: coacd
  solver_position_iterations: null
  solver_velocity_iterations: null
  contact_offset_m: null
  rest_offset_m: null
  ccd_enabled: null
  sleep_threshold: null
```

`null` 表示当前尚未完成显式配置或系统识别，不能默认为已知。

质量记录应从“density + minimum mass”扩展为：

```text
mass
center_of_mass
inertia_tensor
inertia_source
collision_volume
visual_volume
```

## 6. 对 IK 与轨迹验证的参考意见

Do-as-I-do 的 IK 首帧先求手腕，再加入指尖目标；后续帧使用前一帧 warm start。其思想适合用于 Stage4A：

```text
末端位姿 IK
  → 关节连续性
  → 关节限位
  → 速度/加速度/jerk
  → 全身碰撞
  → 前向运动学复核
```

自由关节和旋转关节速度不能简单对四元数逐元素差分，应使用流形上的位姿差分。

Stage4A 建议门禁：

- 所有帧 IK 有解；
- 末端位置误差 mean ≤5 mm、max ≤15 mm；
- 末端姿态误差按任务设定独立阈值；
- 无关节越界；
- 无环境、自碰撞、另一机械臂碰撞；
- 所有速度、加速度和 jerk 满足硬件限制；
- 手指关节映射后仍在真实 Sharpa/夹爪范围内。

## 7. 对物理优化的参考意见

### 7.1 保留确定性基线，再增加 residual MPC

Do-as-I-do 使用的控制形式可以抽象为：

```text
u = u_reference + delta_u
```

其中 `delta_u` 在稀疏时间 knot 上采样并平滑插值，通过并行物理 rollout 评价，再执行短前缀并滚动重规划。

建议顺序：

1. v1 继续完成当前 Stage2/Stage3 确定性、可解释的分层搜索；
2. 确定性基线通过后，新增可选 `stage3a_residual_mpc`；
3. MPC 只能在当前 executable bundle 上增加有界残差；
4. 不得重新生成或覆盖 Stage2 已验证前缀。

建议残差边界：

```yaml
residual_mpc:
  enabled: false
  base_translation_limit_m: 0.010
  base_rotation_limit_deg: 5.0
  gripper_residual_limit_m: 0.008
  zero_first_knot: true
  preserve_stage2_prefix: true
```

样本选择仍必须遵循：

```text
先硬拒绝提前接触、穿透、掉落、单侧接触和 T_GO 越界
  → 再按 repeat success
  → 再按 T_GO 漂移
  → 再按 MANO 保真
  → 最后按控制平滑度
```

不能用一个加权 reward 抵消抓取失败。

### 7.2 增加扰动鲁棒性门

参考并行扰动 rollout，但使用尺寸和质量归一化参数：

```yaml
robustness:
  force_scales_mg: [0.1, 0.25, 0.5]
  torque_scales_mgr: [0.1, 0.25, 0.5]
  directions: 6
  seeds: 4
  settle_after_disturbance_s: 0.5
```

扰动只允许在以下条件同时成立后施加：

- 已形成稳定双侧接触；
- 物体已经离开有效支撑面；
- 相对速度已稳定；
- 不处于 CLOSE、RELEASE 或接触切换区间。

门禁应检查：

- 扰动期间物体不掉落；
- 双侧接触比例仍达标；
- `T_GO` 不越界；
- 扰动结束后相对位姿能够恢复；
- 输出最大通过扰动包络。

## 8. 对 warmup 和支撑建模的限制

Do-as-I-do 会在 warmup 中把物体 weld 到世界并关闭重力，结束后再解除 weld、恢复重力。这个方法可以用于优化器初始化，但不能计入正式抓取验证。

如果采用类似辅助初始化，计划必须规定：

```text
辅助约束阶段标记为 NON_EVALUATION
  → 解除 weld
  → 恢复完整重力
  → 无控制 settle
  → 重新记录 T_reset/T_robot_set/T_warmup/T_approach
  → S2-0 PASS 后才进入正式优化
```

Do-as-I-do 还会生成 pedestal 和焊接在物体上的 support geom。你的正式 Stage3B 不应照搬这种隐藏支撑；只能使用输入场景真实存在的桌面、容器、碗或支撑面。

## 9. 对抓取判定的参考边界

Do-as-I-do 使用手 mesh 与物体 mesh 的最小距离阈值判断参考轨迹是否处于 `in_hand`。该距离门适合生成相位 mask，但不能作为成功判定。

正式抓取仍应由以下联合证据决定：

```text
双侧目标接触
+ 接触法向方向合理
+ 相对速度稳定
+ T_GO 稳定
+ 物体完成任务进度
+ 扰动后可恢复
```

禁止使用单一 proximity、单一 lift 或单一 reward 宣称成功。

## 10. 统一产物补充

Stage2 起建议把 executable bundle 划分为四层：

```text
reference:
  mano_base_pose
  mano_gripper_q
  object_reference_pose

command:
  base_pose_target
  gripper_drive_target
  source_frame
  control_time_s
  phase

sim_actual:
  robot_qpos
  robot_qvel
  actual_base_pose
  actual_gripper_q
  actuator_force_or_torque
  force_limit_saturation
  object_pose
  object_twist
  contact_pairs
  contact_points
  normal_impulse
  tangent_impulse

hardware_actual:
  timestamp
  target_q
  actual_q
  joint_current
  tracking_error
```

正式 artifact 必须保存实际执行轨迹，不能只保存优化器临时使用的命令。

## 11. 计划修改优先级

### P0：v1 必须增加

1. Stage0 共享异常修复 mask；
2. bbox/CoACD 双层碰撞资产；
3. reference/command/sim_actual 分层产物；
4. Stage4 拆成 Stage4A 和 Stage4B；
5. 明确 Stage4 未通过时不得声称整机可执行；
6. warmup 辅助约束退出后的重新验收。

### P1：确定性 Stage2/3 通过后增加

1. 扰动恢复 gate；
2. Stage3A residual MPC；
3. 完整接触 wrench 与执行器饱和记录；
4. Stage4B 动力学域随机化。

### P2：真机部署前增加

1. 仿真 actuator force 与真实电流/力矩的系统标定；
2. 真机 target/actual/current 同步记录；
3. 低速 dry-run 和工作空间边界检查；
4. 跟踪误差、过流、通信超时和接触异常保护；
5. Stage4B 与真机回放指标对照。

## 12. 最终推荐架构

```text
Stage0 统一坐标、共享异常修复、MANO 重定向
  ↓
Stage1 理想就位局部抓取验证
  ↓
Stage2 无提前扰动的接近、闭合、稳定、短携带
  ↓
Stage3A gripper-only 全程携带
  ├─ deterministic baseline
  └─ optional bounded residual MPC
  ↓
Stage3B 可选真实支撑放置
  ↓
Stage4A full_robot IK、限位、连续性和全身碰撞
  ↓
Stage4B full_robot 动力学回放并重新验证物体
  ↓
Hardware dry-run / low-speed replay / feedback validation
```

Stage3A 通过表示“夹爪级抓取携带”成立；只有 Stage4A 和 Stage4B 都通过，才表示“整机可执行”成立。
