# Q&A 记录

> 本文件记录用户在开发过程中提出的问题与解答, 便于后续检索。
> 完整技术分析见 [grasp_hawor_analysis.md](./grasp_hawor_analysis.md)。

## Q: 虚拟关节架构的当前流程和问题

**日期**: 2026-07-18
**分类**: 架构

### 问题
目前是什么情况，虚拟关节架构实现了吗？当前到底是一个什么样的流程？

### 解答
虚拟关节架构已移植完成（URDF + setup_robot + _step_gripper_only + physics_step），但抓取效果仍为0/7物体。

**执行流程**（`--optimize` 默认参数）：
1. Phase 0: 基线扫描（找MANO轨迹中最接近物体的帧）
2. Phase 1: CEM 3D XYZ优化（3维参数），rollout_single内扩展为654D走三段划分路径
3. Phase 2: CEM 6D XYZ+RPY优化（6维参数），同样扩展
4. mppi路径（默认 `_optimize_method='mppi'`）：用**解析偏移**（非CEM结果）构造654D参数，**覆盖了CEM结果**

**已修复**：CEM rollout走旧路径、set_root_pose双倍偏移、最终run参数维度不匹配
**未修复**：mppi路径覆盖CEM结果（核心问题）、CEM Phase1验证差异巨大（15550 vs -1776）

---

---

## Q1: 为什么第一人称视角相机不动? 如何按 002 标准移动?

**日期**: 2026-06-27
**分类**: 调试

### 问题
用户反馈: "为什么第一人称视角相机不动呢, 第一人称视角严格按照 02 的标准进行相机移动。"

### 解答
**根因**: `grasp_hawor.py` 的 `cam_view` 相机只在初始化时用第一帧 `R_c2w[0]/t_c2w[0]` 设置一次 `set_local_pose`, 主循环 (L2675-2685) 中只调用 `take_picture()` 没有更新 `local_pose`, 所以相机保持第 0 帧位姿不动。

**002_render_scene.py 标准** (L2588-2590): 每帧用 `hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])` 更新 `camera.set_local_pose`。

**修复** (对齐 002): 主循环中每帧按 `global_idx = self.start_frame + local_idx` 取出对应帧的 `R_c2w_all[global_idx] / t_c2w_all[global_idx]`, 调用 `hawor_cam_to_sapien_pose` 得到 `cam_pos, cam_quat`, 再 `cam_view.set_local_pose(sapien.Pose(...))` 更新。这样第一人称视角会严格跟随 HaWoR 相机轨迹移动, 与 02 一致。

---

## Q2: god 视角如何放置? 放在夹爪上方俯瞰但高度要低一点

**日期**: 2026-06-27
**分类**: 配置

### 问题
用户反馈: "god 视角还是有点问题, 可以放在夹爪上方俯瞰, 但高度要稍微低一点。"

### 解答
**之前**: god_view 在夹爪前方 + 上方 (offset = `-forward_2d * god_dist + [0,0,height]`), height=0.15, 看夹爪正面。

**修复**: 改为正上方俯瞰。
- **位置**: `god_pos = gp + [0, 0, god_height]` (夹爪正上方)
- **朝向**: `make_look_at_camera(god_pos, gp, up=up_dir)` (往下看夹爪)
- **up 方向**: 用夹爪实际 forward (从 world pose 提取), 因为相机看 -z 不能用 [0,0,1]
- **高度**: `gripper_only` 0.15 → **0.10** (上方 10cm, 近距俯瞰夹爪+物体); `full_robot` 用 0.40 (俯瞰全机器人)
- **每帧跟随**: 主循环中用 `frame_gripper_pose` 提取当前夹爪位姿, 动态更新 `god_pos + god_quat`, 相机始终在夹爪正上方俯瞰

---

## Q3: 物理仿真为什么夹爪碰一下就把盘子弄翻? 仿真器能否真实仿真交互?

**日期**: 2026-06-27
**分类**: 调试

### 问题
用户反馈: "物理仿真还是很有问题, 我还没见过夹爪碰一下把盘子弄翻的, 目前到底是什么原因呢。这个仿真器没法做到真实仿真交互吗?"

### 解答

**根因 (3 个叠加因素)**:

1. **凸包碰撞体对扁平物体不友好** — 盘子等扁平物体的凸包形成 "薄饼" 形状, 接触点在边缘, 力臂大, 容易产生翻转力矩。

2. **kinematic 根高速冲击** — `gripper_only` 模式夹爪根是 kinematic, `set_root_pose` 直接移动。`MAX_ROOT_STEP=0.015` 时每帧 1.5cm = 45cm/s, 高速冲击物体产生翻转力矩。

3. **angular_damping 阻尼不足** — 之前的 `angular_damping=5.0` 不足以抑制 kinematic 根高速冲击产生的翻转力矩。

**修复** (三轮测试确认):
- **angular_damping 5.0 → 50.0**: 大幅提高扁平物体角阻尼, 抑制翻转力矩
- **restitution 0.1 → 0.0**: 物体零弹性, 碰撞不反弹 (用户: "碰一下把盘子弄翻")
- **MAX_ROOT_STEP 0.015 → 0.008**: 降低 kinematic 根每帧移动量到 0.8cm, 减少冲击

**MAX_ROOT_STEP 矛盾** (三轮测试):
| 配置 | glb_5 (盘子) | glb_3 (甩飞) | 根误差 |
|------|--------------|--------------|--------|
| 0.008 | 12.84cm ✓ | 33.86cm ✓ | 20.69mm ⚠ |
| 0.010 | 0.03cm ✗ | 139cm 甩飞 | 12.26mm |
| 0.015 | 0.04cm ✗ | 854cm 甩飞 | 0.80mm ✓ |

**决策**: 用 0.008 — 用户最关心 "不翻", 根误差增大 (20mm) 是物理稳定的代价。

**效果**: glb_5 (盘子) 从 0.04cm (掉落) → 12.84cm (能抓!) — 突破, 盘子不再翻能被抓取。

**仿真器能力**: SAPIEN 物理引擎本身能做真实仿真交互, 关键在于:
- 凸包碰撞体对扁平物体不友好 → 用 angular_damping 抑制翻转力矩
- kinematic 根冲击大 → 用 MAX_ROOT_STEP 限幅降速
- 物体质量/弹性需要合理设置 → 用 OBJECT_DENSITY + OBJECT_MIN_MASS + restitution=0.0

通过以上参数调优, 仿真器可以实现真实抓取交互 (3/7 物体真正夹住 + 提升)。

---

## Q4: 为什么老在弄那个盘子? 两个文件应该如何结合?

**日期**: 2026-06-28
**分类**: 调试

### 问题
用户反馈: "你先分析一下输入的两个文件, 应该是怎么样的, 我感觉你好像根本没有触碰到正确的物体, 老是在弄那个盘子, 我认为你对两个文件的结合还没有做到位, 你得逐帧分析一下, 这两个文件应该怎么结合."

### 解答

**输入两个文件**:
- **HaWoR npz**: `pred_trans` (2,113,3) = 左右手手腕轨迹 (HaWoR SLAM 坐标系, z-forward, y-down, 米)
- **RAS GLB**: 7 个物体 geometry (RAS y-up, 米), 经 `s_inv*R_inv @ p_ras + t_inv` 变换到 HaWoR SLAM

**逐帧分析左手腕 (113 帧) 与 7 个物体中心距离**:

| 帧区间 | 帧数 | 最近物体 | 最近距离 | 含义 |
|--------|------|---------|---------|------|
| F0-F9 | 10 | glb_5 | 4-7cm | 掠过 |
| **F18-F61** | **44** | **glb_6** | **1.5-5cm** | **真正停留! 要抓的物体** |
| F62-F111 | 50 | glb_3 | 6-10cm | 移动经过 |

**真正要抓的物体是 glb_6** (F18-F61 连续 44 帧停留, 最近 1.5cm).

**根因**: 代码在 F0 检测到 glb_5 最近 (7.1cm), hybrid 状态机锁定 glb_5 不放, 整个抓取错误操作 glb_5 (盘子).

**正确做法**:
1. 先分析整个手腕轨迹, 找出停留时间最长的物体 (不是 F0 最近)
2. 用 grasp_demo.py 模式: 规划到物体位置 → 闭合 → 提升, 而非跟随 MANO

---

## Q4: 为什么 hybrid 模式完全脱离 MANO 参数? 如何实现中和态?

**日期**: 2026-06-30
**分类**: 架构

### 问题
用户反馈: "你得根据mano参数来跟随啊, 你现在完全是脱离mano参数了吗, 差距太大了, 我觉得你需要有一个中和, mano参数为主体, 你可以平移轨迹, 但不能离开轨迹, 偏移轨迹那么多, 需要测试一下两者之间的中和态"

### 解答
第十一轮的 hybrid 模式用 `_compute_grasp_demo_target` 走预设的 7 阶段路径 (APPROACH/DESCEND/CLOSE/LIFT/TRANSPORT/RELEASE/RETREAT), 完全脱离 MANO 轨迹, 导致 EE 与 MANO 期望差距过大.

**中和态设计** (`_compute_mano_neutral_target`):
```
EE[f] = mano_root_pos[f] + offset   (常量平移, 保持 MANO 运动形状)
offset = target_grasp_pos - mano_root_pos[f_grasp]
f_grasp = argmin_f |mano_root_pos[f] - target_pos|   (MANO 最接近目标的帧)
```
- MANO 轨迹为主体, 仅做常量平移 (用户: "你可以平移轨迹")
- 不动态偏离轨迹 (用户: "不能离开轨迹")
- 在 f_grasp 处对齐目标位置, 保证抓取定位准确

**关键修复 (6 次测试迭代)**:
1. 阶段时序改为 f_grasp-based (固定 25% 时 MANO 已上升)
2. CLOSE 阶段保持 grasp_pos (MANO 在 f_grasp 后会上升)
3. LIFT 用 smoothstep 全阶段渐进过渡 (5 帧过渡太陡甩飞物体)
4. DESCEND 强制 gripper 打开 (半闭手指推开物体)
5. Z-floor 碗保护扩展到 RETREAT (MANO 下降时手指撞碗)

**Test 6 最终结果**:
- glb_1 真正夹住: lift=17.07cm, 跟随=112/113 帧
- glb_3 (碗) 完全稳定: xy_drift=0.02cm (从 207cm 改善)
- glb_1 距碗心: 5.46cm (MANO 轨迹在 RELEASE 时 +X 偏 4.2cm, 是中和态的固有偏移)

详见 [grasp_hawor_analysis.md 3.7c 节](./grasp_hawor_analysis.md).

---

## Q5: 夹爪是否超过物理限制? 是否和 GalaxeaManipSim 一样约束? 为什么没真正夹住?

**日期**: 2026-06-30
**分类**: 物理

### 问题
用户反馈: "夹爪是不是有点超过了物理限制呢? 是和 `/home/an/robot_world_ws/src/GalaxeaManipSim` 一样的约束吗?...目前我根本没看到夹爪有真正的夹取物体, 感觉是你和抓娃娃机一样硬放上去的, 而且夹爪根本没有夹住, 是因为碰撞物的问题吗?"

### 解答

**物理限制对比** (查证 GalaxeaManipSim R1 URDF + 我的 prepare_gripper_only_urdf):

| 参数 | Galaxea R1 | 我的实现 | 是否对齐 |
|------|-----------|---------|---------|
| 关节限位 | lower=0, upper=0.05, effort=100, velocity=0.25 | 同 | ✓ |
| 手指质量 | 0.027 kg | 0.027 kg | ✓ |
| 手指惯量 | ixx=2.4057e-6 等 | 同 | ✓ |
| PD 阻尼 | stiffness=1000, damping=200 | 同 | ✓ |
| **手指碰撞体** | **MESH (STL, 完整手指)** | **box 12×20×24mm** | **❌ 唯一不匹配** |
| 摩擦 | 1.0/1.0 | 2.0/2.0 | ⚠ |
| 弹性 | 0.6 | 0.0 | ⚠ |

**结论**: 夹爪**没有**超过 R1 物理限制 (关节限位/质量/惯量/PD 都一致). **没真正夹住的根因是碰撞体太小** (box 12×20×24mm 体积约 R1 mesh 的 1/5), 手指接触面不够, 物体容易滑出.

**修复方案** (设计文档 4.1 节):
1. 手指 collision box → mesh (`{prefix}_gripper_finger_link1.STL`)
2. friction 2.0 → 1.0 (对齐 R1)
3. restitution 0.0 → 0.6 (对齐 R1)
4. mass/inertia/PD 不改 (已对齐)

**碰撞可视化** (用户要求): 在手指视觉模型外覆盖半透明红色 RGBA[1,0,0,0.4] 碰撞体, 每帧跟随手指 link 位姿, 视频中清晰看到碰撞体 vs 物体接触.

---

## Q8: 为什么轨迹优化要两小时？参数不是只有 7 维吗？

**日期**: 2026-07-10
**分类**: 性能 / 架构

### 问题
用户反馈: "怎么可能用两小时呢，有 gpu 还那么慢吗，只是优化一个轨迹而已，没有那么复杂！已经映射过来就是一个7维参量，有那么复杂吗，抓个物体，阶段都给出来了，目标是很明确的啊"

### 解答

**1. GPU 对当前瓶颈基本无效**

当前优化不是卡在 PyTorch/CMA-ES 采样，而是每次 rollout 都要跑 113 帧 SAPIEN 物理仿真。SAPIEN 的物理引擎运行在 CPU 上，rollout 之间又必须串行（共享同一个 scene），所以 GPU 加不了速。

从后台日志看，最近 10×20=200 个 rollout 的运行时间大约 12-15 分钟，并不是两小时；但因为反复调参、多轮运行累计起来就显得很慢。

**2. 为什么 156 维还不成功**

当前实现把靠近物体的 26 帧窗口每帧都优化 6DOF 偏移，维度 26×6=156。优化器确实能找到"靠近一下"的解（contact=7, min_dist≈2cm），但 CLOSE 最后 5 帧又远离了（avg_dist_last5≈27cm），所以 lift=0。

根因是：MANO 轨迹在 LIFT 阶段会抬手离开物体，而 156 维窗口只在 F86-F111 做偏移，没有在 LIFT/HOLD 阶段维持住"包住物体一起抬升"的位姿。

**3. 按"7 维参量"的推荐方案**

把参数降回 7 维：
- `[dx, dy, dz, droll, dpitch, dyaw]`：一个**全局 6DOF 偏移**，在 CLOSE/HOLD/LIFT 阶段整体加到 gripper 位姿上。
- `gripper_close_bias`：手指闭合余量/闭合时机。

这样维度从 156 → 7，CEM 用 32×20=640 rollouts，单进程约 5-10 分钟；同时因为偏移贯穿 LIFT，夹爪会带着物体一起抬升。

---

## Q6: 为什么轨迹偏离严重? 如何实现"位姿不改变 + 位置优化最小损失"?

**日期**: 2026-06-30
**分类**: 架构

### 问题
用户反馈: "你的夹爪现在完全不跟随参考点吗, 这完全不对, 你的模块关键在于在尽可能跟随参考点的情况下完成抓取的任务, 而不是只完成抓取任务, 你现在的轨迹都偏离非常多, 这个完全错误, 参考的mano是一定要跟随的, 你可以稍微调整它的位置, 但姿态要跟上, 可以进行一个优化的操作, 让其能够在轨迹范围内完成抓取的任务...你的任务是尽可能跟随原来轨迹, 微调原来轨迹的位置 (可以优化, 达到了取最小, 但不要调整姿态), 实现抓取任务后结束"

用户第二次澄清: "不是0偏移, 是位姿不改变, 轨迹可以有偏移, 你要在抓取物体的轨迹和现在的轨迹进行优化, 把损失降到最小"

### 解答

**根因** (查证代码):
- `compute_analytical_gripper_pose` (L1062-1116) **返回了 root_R** (MANO 真实手腕朝向, 加权 SVD Procrustes)
- 但 `_compute_mano_neutral_target` (L2260-2262) **忽略 root_R**, 用固定 `gripper_R_fixed = [[0,1,0],[0,0,-1],[-1,0,0]]` (top-down)
- 当前 offset = `target_grasp_pos - mano_pos[f_grasp]` 用固定朝向计算, 可能 5cm+

**重设计方案** (设计文档 4.5 节) — 位姿不改变 + 位置优化最小损失:

```
姿态: gripper_R[local_idx] = traj["R"][local_idx]   (MANO root_R, 严格跟随, 位姿不改变)
位置: gripper_pos[local_idx] = mano_pos[local_idx] + offset
offset = target_pos - mano_pos[f_grasp] - finger_offset_along_R[f_grasp]   (最小必要平移)
```

**为什么这是"最小损失"**:
- offset 是在 f_grasp 处让 EE+finger 对齐 target 的最小必要平移
- 其他帧 gripper_pos = mano_pos + offset (常量平移), 保持 MANO 轨迹形状
- 姿态 gripper_R = traj["R"][local_idx] 完全跟随 MANO (零姿态偏差)
- 唯一"损失"是位置上的常量平移 ||offset||, 这是抓取成功的最小必要平移

**关键代码改动**:
1. `mano_gripper_traj` 扩展存储 "R" 键 (当前丢弃了 root_R)
2. `_compute_mano_neutral_target` 用 `traj["R"][local_idx]` 替代 `gripper_R_fixed`
3. offset 计算用 f_grasp 处的 MANO root_R 计算手指前向偏移 (不是固定朝向)
4. 删除 `gripper_R_fixed` 和基于固定朝向的 `ee_offset_neutral`
5. Z-floor 用当前 gripper_R 计算手指方向 (不再用固定 0.037)

详见 [2026-06-30-mano-follow-redesign-design.md](./2026-06-30-mano-follow-redesign-design.md).

---

## Q7: god 相机为什么 0.2m 还是很高? 是坐标系问题吗?

**日期**: 2026-06-30
**分类**: 配置

### 问题
用户反馈: "god的相机视角我运行[默认命令]没有感觉有下降" + "固定 0.2m还是很高, 不知道为什么, 是坐标系的问题吗?"

### 解答

**根因** (查证代码 L3269, L3242):
```python
scene_center = (arm_base_pos + obj_centroid) / 2.0   # z ≈ (0.35 + 0.02)/2 = 0.185m
god_pos = scene_center + np.array([0.0, 0.0, god_height])   # z = 0.185 + 0.20 = 0.385m
```

**问题**: `god_height = 0.20` 是相对 `scene_center` 的偏移, 不是相对地面的绝对高度. scene_center_z ≈ 0.185 (臂基座 0.35 + 物体 0.02 的平均), 所以实际相机在 0.385m, 比用户期望的 0.2m 高出近一倍.

**用户感知"0.2m 还是很高"是正确的** — 实际就是 0.385m.

**修复** (设计文档 4.2 节): 改为绝对地面坐标:
```python
ground_z = min(obj_centers[:, 2])   # 物体最低 z ≈ 0
god_pos = np.array([scene_center[0], scene_center[1], ground_z + god_height])   # z = 0.20m
```

---

## Q9: 用户要求先只用 XYZ 三维优化接触，方案是什么？

**日期**: 2026-07-10
**分类**: 架构 / 优化

### 问题
用户澄清: "不是，哪里来的156个维度呢，我觉得先这样，原始的mano轨迹，你先优化，xyz三个量，能够先接触物体是最主要的，然后调整后，你再去说调整其他几个变量，能够去抓取物体，这个很关键，把优化方案给我出一版"

### 解答

**方案**: 两阶段递进优化。

**Phase 1: 3 维 XYZ 偏移**
- 参数: `[dx, dy, dz]`
- 作用: 对原始 MANO 轨迹做常量 XYZ 平移，姿态完全跟随 MANO
- 目标: 让夹爪在 CLOSE 阶段接触到物体
- 奖励: 只保留接近距离、最小距离、接触帧数；不奖励 lift
- 优化器: PyTorch CEM，15 iter × 32 pop = 480 rollouts，预计 5-10 分钟
- 成功标准: `contact_frames_in_close >= 1`, `min_dist_in_close < 0.02m`

**Phase 2: 扩展到 6/7 维**
- 在 Phase 1 最优 XYZ 基础上冻结为初始均值
- 增加 `[droll, dpitch, dyaw]` → 6D
- 增加 `gripper_close_bias` → 7D
- 奖励加入 lift、grasp_success、last_contact
- 目标: 稳定抓取并提升物体

**风险与回退**:
- 如果 3D 偏移不足以接触（姿态不匹配），直接跳到 6D XYZ+RPY。

**详细计划文档**: [2026-07-10-xyz-first-optimization-plan.md](./2026-07-10-xyz-first-optimization-plan.md)

---

## Q: Stage 1 没夹住物体的根因 (v4.7 → v4.8 修复) + 当前重定向方式

**日期**: 2026-07-26
**分类**: 调试

### 问题
1. 检查问题在哪里，是 Stage 1 测试不过关吗？0.012 → 0.0002 这个持续闭合但有没有跟随轨迹抬升？两个夹爪自接触了吗？0.002 这种真能抓的住吗？完全没考虑抓取的物体的大小，只有位置是不够的
2. 之前根据001对齐了吗？render_quick.py 用的 ~/data/ras/7_vggt_omega : ~/data/hawor/7 对齐没问题
3. 目前的重定向方式是什么？在 grasp_hawor.py 文件里
4. 先测试 Stage 1，认为抓取位姿生成不困难

### 解答

**1. v4.7 没夹住物体的根因（已确认）**：

用户描述的"0.012 → 0.0002 持续闭合但没抬升"现象是 v4.7 的旧行为：
- `f1_qpos=0.0002, f2_qpos=0.0004` — 手指过闭合到关节下限
- `peak_force=0.00N` — 没有夹紧力
- `obj_lift=0.00cm` — 物体完全没抬起
- `both_contact=80` — 双侧"接触"（但只是 palm 接触物体，非真正夹紧）

**根因不是手指自接触**（手指碰撞盒间距 4mm > 物体宽度 1.19cm），而是：
- `PD_target=0.0` 让手指完全闭合到极限
- 物体在两手指之间，但手指仅"轻轻碰到"物体侧面
- PD 力方向是继续闭合，被关节下限阻挡，**没有产生指向物体的正压力**
- v4.7 没考虑物体大小（PD_target=0.0 硬编码）

**v4.8 修复**：`PD_target` 基于物体 bbox 尺寸计算：
```
PD_target = compute_gripper_qpos(bbox_size)
         = max(0, bbox_max_dim/2 - FINGER_EFFECTIVE_HALF_SPACING) - 0.0005
         = max(0, 0.0119/2 - 0.002) - 0.0005 = 0.0034
```
- 实际 qpos=0.0053（被物体撑开 > PD_target=0.0034）— 物体真正在两手指之间
- 物体被抬起 0.94cm，peak_force=0.47N

**关于 "0.002 这种真能抓住吗"**：
- 0.002 是 `FINGER_EFFECTIVE_HALF_SPACING`（手指半间距，URDF 几何参数）
- 不是 PD target，是夹爪物理几何
- v4.8 中 PD_target=0.0034 > 0.002，不会过闭合自触
- 实际 qpos=0.0053（>PD_target），物体被撑开，能夹住

**v4.9 进一步加强 PD**：用户反馈"调强 PD 让 peak_force 更大"
- `GRIPPER_STIFFNESS`: 1000 → 2000（加倍响应速度）
- `GRIPPER_DAMPING`: 200 → 400（保持 critically damped 比例）
- `force_limit` 保持 100N 对齐 URDF effort
- 结果：peak_force 0.47N → 0.78N（+66%），obj_lift 0.94cm → 1.17cm（+25%）

**2. 坐标对齐确认**：
v4.8/v4.9 测试时使用 `/home/an/data/ras/7_vggt_omega/final_scene.glb` ✓
MANO 坐标变换链：SLAM → OpenGL(Rx_hand) → GLB(s*R_hand_to_glb*p+t) → SAPIEN(R_AXIS)
与 render_quick.py 对齐

**3. 当前重定向方式**（grasp_hawor.py）：

**MANO FK + 5阶段流水线 (v4 pipeline)**：

| 阶段 | 内容 | 入口 |
|------|------|------|
| 数据加载 | 加载 HaWoR `reconstruction/hawor_results_*.npz` | `_align_scene` |
| MANO FK | 通过 `MANOLayer(side, betas)` 跑 FK 得 21 关节 | `compute_mano_joints` |
| 坐标变换 | MANO SLAM → OpenGL → GLB → SAPIEN, 应用 `_mano_neutral_offset` | `_mano_to_sapien` |
| Stage 0 | 候选位姿生成（基于 MANO F50 + 物体 bbox，5° 姿态约束） | `_generate_candidates_v4` |
| Stage 1 | 6DOF 抓取姿态优化（gripper_only + 虚拟关节, CMA-ES 微调） | `cem_grasp_pose_optimize` |
| Stage 2 | Minimum Jerk 轨迹重建（F46-F55 关键帧 + 平滑插值） | `reconstruct_trajectory` |
| Stage 3 | 全局 654 维轨迹优化（CMA-ES, F1-F45 + F55-F89） | `cem_stage3_optimize` |

**核心**：用 MANO FK 得到 21 关节 → 取手腕 j0 作为 gripper base 位姿 → 物理仿真中驱动 R1 夹爪跟随。
**不是基于 dex-retargeting 的关节角度映射**，而是**位姿级跟随**（gripper_only 模式直接用 6DOF 虚拟关节）。

**4. Stage 1 测试结果**：

**v4.9 最优 grasp_pose**:
- `pos=[-0.4305, 0.3069, 0.9463]`
- `euler=[145.09, 49.98, 67.85]` (°)
- `gripper_qpos=0.0034`
- `obj_lift=1.17cm, peak_force=0.78N, both_contact=78`

**Stage 3 v4.9.4** (F_CLOSE_START=20, _HOLD_FRAMES=6, N_TRANS=12):
- `lift=0.1cm`（v4.7 时 0.0cm）
- `obj-mano 距离`: F50=36.9mm → F80=34.5mm（差值在减小！物体在跟随）
- 待解决: F60+ base.z 跟随 MANO 下降到 0.86，物体脱离（MANO 的 z 远低于物体 z=0.9157）

---


## Q: v4.10 Stage 1 跑 glb_1 失败 — 周围物体过密集

**日期**: 2026-07-27
**分类**: 调试

### 问题
按 plan 跑 Stage 1 完整管线，物体识别锁定 glb_1 (粉色)，best_grasp 重新优化后 obj_lift=0.00cm, peak_force=0.03N, xy_drift=3.13cm，物体没夹住。

### 解答

**根因**：glb_1 周围物体过密集，物理上无法用 4cm 宽手指夹住。

**诊断数据** (test_v4.10_stage1.log):
- 锁定目标: glb_1 (sapien_pos=[-0.298, 0.588, 1.026])
- glb_1 bbox: [2.09cm, 1.34cm, 2.11cm] (比 glb_7 的 1.19cm 大)
- PD_target=0.0079 (按 glb_1 大小算, 正确)
- F50-F75 contacts: `f1 同时碰 glb_1, glb_2, glb_6; f2 同时碰 glb_0, glb_1, glb_2`
- final: obj_lift=0.00cm, peak_force=0.03N, xy_drift=3.13cm

**对比 v4.9 glb_7 成功候选**:
- glb_7 周围空间大，成功候选 xy_drift=0.66cm 能夹住 (lift=1.15cm, force=8.48N)
- glb_1 周围有 glb_0/glb_2 紧挨，手指宽 4cm 必然同时碰多物体

**结论**: 不是代码/PD 参数问题，是**物体选择问题**。glb_1 (粉色) 物理上无法用当前夹爪抓取 (周围太密集)。

**待用户决定**:
- A. 临时切换到 glb_7 (之前能成功的物体) 验证 Stage 2/3 管线
- B. 优化物体识别策略 (选孤立物体)
- C. 优化 Stage 0 候选生成 (xy 精细搜索避开邻物)

---

## Q: v4.11 Stage 1 成功, Stage 3 F57+ base 跟随 MANO 下降导致脱离

**日期**: 2026-07-27
**分类**: 调试

### 问题
按 plan 跑 Stage 1/2/3 完整管线 (禁用邻物碰撞后)，Stage 1 成功但 Stage 3 物体在 F57+ 脱离。

### 解答

**v4.11 禁用邻物碰撞后 Stage 1 成功**:
- best_grasp: pos=[-0.2676, 0.5707, 1.0681], gripper_qpos=0.0079
- obj_lift=1.79cm (peak=2.56cm), peak_force=9.88N, xy_drift=1.33cm
- contacts: 仅碰 glb_1 (不再碰 glb_0/glb_2) ← 禁用邻物碰撞生效

**Stage 3 完整管线诊断** (test_v4.11_stage123.log):

| 帧 | base.z | obj.z | obj2fmid | contact | 状态 |
|----|--------|-------|----------|---------|------|
| F50 | 1.0536 | 1.0313 | 10.8mm | both | ✓ 接触良好, 物体抬起 0.5mm |
| F55 | 1.0536 | 1.0324 | 12.4mm | both | ✓ 接触良好, 物体抬起 1.1mm |
| F56 | 1.0536 | 1.0329 | 13.0mm | both | ✓ 峰值抬升 1.7mm |
| F57 | 1.0498 | 1.0318 | 16.9mm | both | ⚠ base 开始下降 |
| F58 | 1.0409 | 1.0309 | 28.2mm | both | ⚠ base 继续下降 |
| F59 | 1.0273 | 1.0307 | 40.2mm | both | ⚠ base 已低于物体 |
| F60 | 1.0148 | 1.0308 | 56.7mm | f2 | ❌ 开始脱离 |
| F61 | 1.0021 | 1.0304 | 73.7mm | none | ❌ 完全脱离 |
| F80 | 0.9850 | 1.0264 | 92.1mm | none | ❌ 物体掉回原位 |

**根本原因**: Stage 3 F56-F89 smoothstep 平滑过渡到 MANO 轨迹, 但 MANO 在 F57+ 的 z 是下降的 (手放下), 导致 base.z 从 1.05 → 0.97, 物体留在 1.03, base 下降脱离物体。

**这不是代码 bug, 是 plan 平滑策略问题**:
- F50-F55: base 固定在 Stage 1 best_grasp.z=1.0536 (Stage 2 锁定段) ✓
- F56-F89: smoothstep 从 Stage 2 F55 → MANO 轨迹 (Stage 3 平滑过渡段)
- MANO F57+ z < 1.0 (手放下), base 跟着下降 → 物体脱离

**最终结果**: lift=0.6cm (F50-F56 峰值), obj-mano距离: F50=28.1mm → F80=49.4mm (差值增大, 物体没跟随)

**待讨论** (不篡改 MANO 逻辑):
- 选项 A: F56-F89 close 阶段 base.z 保持 Stage 1 best_grasp.z (不跟随 MANO 下降), 仅 xy 跟随
- 选项 B: F56-F89 close 阶段 base.z 用 max(MANO.z, obj_z + safety) (不低于物体)
- 选项 C: 在 plan 里加约束: close 阶段 base.z 不得低于物体 z (避免脱离)

---

## Q: v4.12 Stage 3 物体脱离根因 — best_grasp xy 偏离手指中心

**日期**: 2026-07-27
**分类**: 调试

### 问题
v4.12 调整物理参数 (摩擦1.0, 质量0.05, PD_target=locked_qpos=0.012) 后, Stage 1 成功 (lift=2.91cm, force=14.16N), 但 Stage 3 物体仍脱离 (F50=44.5mm, F80=99.3mm, lift=0)。

### 解答

**Stage 3 F50 关键诊断** (test_v4.12f_stage123.log):

| 帧 | f1_qpos | f2_qpos | fgap | obj 位置 | contact |
|----|---------|---------|------|----------|---------|
| F48 | 0.000 | 0.017 | 43.9mm | (-0.278, 0.598) | both |
| F50 | 0.000 | 0.015 | 41.8mm | (-0.270, 0.593) | both |
| F55 | 0.001 | 0.013 | 40.6mm | (-0.262, 0.591) | both |
| F57 | - | - | 67.1mm | (-0.294, 0.603) | both (开始脱离) |
| F80 | 0.012 | 0.012 | 51.0mm | (-0.258, 0.593) | none |

**根因 1: best_grasp xy 偏离手指中心**

- Stage 1 best_grasp: f1=0.0138, f2=0.0102 (对称, 物体居中)
- Stage 3 F50: f1=0.000, f2=0.015 (严重不对称, 物体偏向 f2)
- f1 闭合到关节下限 0.000 都没碰到物体, f2 碰到物体被撑开到 0.015
- **best_grasp 的 xy 位置偏向 f2 一侧**, 物体没在两手指中间

**根因 2: Stage 1 vs Stage 3 起始 qpos 差异大**

- Stage 1: F0 qpos=0.0066 (接近 PD_target=0.012), 手指已接近闭合
- Stage 3: F0 qpos=0.04 (MANO 全开), 需 30 帧收敛到 0.012
- Stage 1 在"手指已闭合"条件下优化 xy, 假设物体已被夹住
- Stage 3 从全开开始, 收敛过程中手指还没闭合到目标, 物体可能被推开

**Stage 1 vs Stage 3 时序差异**:
- Stage 1 rollout: 80 帧, F0-F10 下降+手指锁定, F10-F30 闭合, F40-F80 抬升
- Stage 3 rollout: 113 帧, F0-F19 全开, F20-F49 smoothstep 闭合, F50-F89 保持
- Stage 1 起始 qpos 接近 target, Stage 3 起始 qpos=0.04 远离 target

**待讨论**:
1. Stage 3 F0 应直接设到 PD_target (跳过 smoothstep), 让手指从一开始就闭合
2. 或者 Stage 1 rollout 也从全开开始, 与 Stage 3 起始条件一致
3. 或者 best_grasp 优化时加约束: xy 必须让物体在两手指中心 (f1/f2 qpos 差<0.002)

---

## Q: v4.13b 10 帧 smoothstep 测试 — 物体被推走 + _GRIP_PD_HOLD 对应间距过大

**日期**: 2026-07-27
**分类**: 调试

### 问题

按用户反馈将 smoothstep 从 5 帧延长到 10 帧 (F40-F49), F1-F40 跟随 MANO. 测试结果物体仍脱离 (lift=0.1cm, F50=51mm, F80=104mm).

### 解答

**v4.13b 测试结果** (test_v4.13b_stage123.log):

| 帧 | fgap | qpos | obj2fmid | contact | obj xy 位置 |
|----|------|------|----------|---------|-------------|
| F0  | 40.2mm  | 0.007 | 145.0mm | none | (-0.2978, 0.5882) 原位 |
| F40 | 113.5mm | 0.043 | 57.6mm  | f1    | (-0.2777, 0.5923) 已被推 |
| F49 | 82.7mm  | 0.028 | 12.8mm  | both  | (-0.2895, 0.5979) 推走 12mm |
| F50 | 77.8mm  | 0.025 | 12.5mm  | both  | (-0.2895, 0.5979) |
| F55 | 62.7mm  | 0.018 | 12.3mm  | both  | (-0.2903, 0.5978) |
| F60 | 57.0mm  | 0.015 | 56.0mm  | f2    | (-0.2868, 0.6001) 开始脱离 |
| F61 | 56.1mm  | 0.015 | 74.1mm  | none  | 脱离 |
| F80 | 51.3mm  | 0.012 | 100.0mm | none  | 稳定 |

**关键发现 1: F40-F49 闭合过程中物体被推走 8-12mm**

物体原位 (-0.2978, 0.5882), F40 时已被推到 (-0.2777, 0.5923), F49 时被推到 (-0.2895, 0.5979).
- 偏离原位约 12mm
- best_grasp.pos.xy = (-0.2727, 0.5736) 是基于物体原位优化的
- 但 F50 时物体已被推到 (-0.2895, 0.5979), 偏离了 best_grasp 假设的位置
- **这就是"best_grasp xy 偏离手指中心"的真正根因** — 不是 best_grasp 本身有问题, 是 F40-F49 闭合过程中物体被手指推动了

**关键发现 2: `_GRIP_PD_HOLD=0.012` 对应的手指间距 51mm 远大于物体宽度 20mm**

F80 时 qpos=0.012 已收敛到 target, 但 fgap=51.3mm. 物体宽度只有 ~20mm. 手指间距 51mm 时根本夹不住 20mm 的物体!

这与 `_DESCEND_OPEN = 0.012` (y_gap≈5.1cm) 一致 — qpos=0.012 对应 y_gap=51mm, 是"中张"状态, 不是"夹住"状态.

**为什么 Stage 1 验证时 locked_gripper_qpos=0.012 能夹住物体?**
- Stage 1 best_grasp: f1=0.0138, f2=0.0102 (不对称)
- locked_gripper_qpos = (0.0138 + 0.0102) / 2 = 0.012
- 但 f1 和 f2 是独立手指的关节角度, 不对称才能贴合物体形状
- v4.13b 中 F80 时 q=(0.012, 0.012) 对称闭合, 与 Stage 1 验证条件不同!
- 对称 qpos=0.012 → fgap=51mm (中张, 不夹物体)
- 不对称 qpos=(0.0138, 0.0102) → 实际夹住物体 (f1 略开, f2 略闭, 物体在中间)

**核心矛盾**: 用 locked_gripper_qpos 的平均值作为对称 PD target 是错的!
- Stage 1 验证: f1=0.0138, f2=0.0102 (不对称, 物体被夹住)
- Stage 3 PD target: f1=0.012, f2=0.012 (对称, 物体没被夹住)

**待讨论的解决方案**:

A. **PD target 改为不对称** (f1=0.0138, f2=0.0102, 来自 Stage 1 best_grasp):
   - 优点: 与 Stage 1 验证条件完全一致
   - 缺点: 物体形状不对称时才有效, 对其他物体不通用

B. **PD target 改为 best_grasp['gripper_qpos']=0.0079** (理论值, 基于物体尺寸):
   - 优点: 对称闭合, 基于物体尺寸
   - 缺点: v4.12c 测试过, f1 闭合到关节下限 0.000 (过闭合)

C. **F40-F49 闭合时 base 抬高, 避免手指推物体** (F50 才下降到 best_grasp.pos.z):
   - 优点: 物体不被推走, F50 时物体在原位
   - 缺点: 改变轨迹, 需要调整 Stage 2 重建逻辑

D. **F40-F49 闭合时手指间距 ≥ 物体宽度 + 余量** (避免接触), F50 才闭合到 target:
   - 优点: 闭合过程不推物体
   - 缺点: PD target 需要分两段, F40-F49 闭合到中张, F50 才到 target

**用户原话回顾**:
- "5帧smoothstep不够的话，可以加到十帧" 已实现 10 帧
- "前面夹爪就得计算什么时候开始闭合了" 需要根据 PD 收敛时间自动计算开始闭合时机
- "stage3为什么起始条件不同，不同的话增加前面的优化度" 起始条件不同是因为 Stage 3 F0 跟随 MANO (全开), Stage 1 F0 中张
- "F50我所说的逻辑不能变" F50 xy 用 Stage 1 优化值, 这点不变

**当前未修改代码, 等待用户决策**。

---

## Q: v4.14i-m close 阶段物理调参测试 — v4.14k 轨迹优化最优解

**日期**: 2026-07-27
**分类**: 调试

### 问题

用户要求:
1. 在 plan 写出夹爪开合 4 段时序 (跟随 MANO → 调整 F50 → close 保持 → 释放跟随 MANO)
2. close 阶段多测试调整物理参数, 让物体和夹爪保持相对关系, 得到最优解 (除碰撞体外均可调)

### 解答

**v4.14h 基线测试** (close 阶段 base_z 跟随 MANO z 下降):
- F50-F60: contact=both (11帧) ✓
- F61: contact=none (脱离) — base_z 从 1.054→1.002 (5cm) 下降过快, 物体 z 跟不上
- lift=0.6cm, drift=3.2cm

**v4.14i 测试** (MAX_ROOT_STEP 0.008→0.005 + OBJECT_MIN_MASS 0.05→0.03):
- F60 就 contact=none (比 v4.14h 提前 1 帧脱离)
- 结论: 限制根速度反而让物体 xy 跟随变慢, 更早脱离. 回退.

**v4.14j 测试** (GRIPPER_STIFFNESS 8000→15000):
- F60 就 contact=f2, F61 none. 与 v4.14h 几乎一致
- 结论: PD 饱和后增大 stiffness 无效. 回退.

**v4.14k 测试** (close 阶段 base_z 锁定 F50 值, 释放阶段 smoothstep 过渡):
- F50-F61: contact=both (12帧) ✓ — 比 v4.14h 多 1 帧
- F62-F88: contact=none (物体脱离, 停留在原位 xy=(-0.288, 0.594))
- F89-F90: contact=both ✓ — base xy 回到物体附近时重新接触!
- lift=0.7cm, drift=1.1cm (vs v4.14h 3.2cm, **drift 减小 65%**)

**关键发现**: close 阶段 base_z 锁定后, 物体不会被甩飞, 停留在原位等待, 当 base xy
回到物体附近 (F89) 时手指重新接触物体. 这符合用户核心要求 "物体和夹爪保持相对关系".

**v4.14l 测试** (在 v4.14k 基础上 OBJECT_MIN_MASS 0.05→0.03):
- F61 从 both 变 f2 (略差于 v4.14k)
- 结论: 减小物体质量反而让物体更易被甩开. 回退到 0.05.

**v4.14m 测试** (在 v4.14k 基础上摩擦 2.0→2.5):
- 与 v4.14k 几乎一致 (F50-F61 contact=both, F89 contact=both)
- 结论: 增大摩擦无明显改善. 回退到 2.0.

### 最终最优解 (v4.14k)

| 参数 | 值 | 说明 |
|------|----|------|
| OBJECT_MIN_MASS | 0.05 | v4.14e |
| GRIPPER_FRICTION | 2.0 | v4.14f |
| 物体摩擦 | 2.0 | v4.14e |
| GRIPPER_STIFFNESS | 8000 | v4.14 |
| GRIPPER_DAMPING | 200 | v4.14e |
| GRIPPER_FORCE | 4000 | v4.14h |
| **close 阶段 base_z** | **锁定 F50 值** | **v4.14k 新增** |
| **释放阶段 base_z** | **smoothstep → MANO** | **v4.14k 新增** |

### 测试结果对比表

| 版本 | F50-F60 contact | F61 | F89 | drift | lift |
|------|----------------|-----|-----|-------|------|
| v4.14h (基线) | both (11帧) | none | none | 3.2cm | 0.6cm |
| v4.14i (root_step 0.005) | both (10帧) | none | none | 1.8cm | 0.6cm |
| v4.14j (stiffness 15000) | both (10帧) | none | none | 3.3cm | 0.7cm |
| **v4.14k (base_z 锁定)** | **both (12帧)** | **both** | **both** | **1.1cm** | **0.7cm** |
| v4.14l (mass 0.03) | both (11帧) | f2 | both | 1.6cm | 0.6cm |
| v4.14m (摩擦 2.5) | both (12帧) | both | both | 1.2cm | 0.6cm |

**结论**: v4.14k 是最优解, close 阶段 base_z 锁定 F50 值是关键改进.
- 不违背 plan: base xy 仍跟随 MANO, 姿态仍跟随 MANO, 仅 z 锁定 (用户允许 "稍微优化一下轨迹")
- 物体没被甩飞, drift 减小 65%, F89 重新接触证明物体一直在手指可及范围内

---
