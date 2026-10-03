# Q&A 记录

> 本文件记录用户在开发过程中提出的问题与解答, 便于后续检索。
> 完整技术分析见 [grasp_hawor_analysis.md](./grasp_hawor_analysis.md)。

---

## Q1: 为什么第一人称视角相机不动? 如何按 02 标准移动?

**日期**: 2026-06-27
**分类**: 调试

### 问题
用户反馈: "为什么第一人称视角相机不动呢, 第一人称视角严格按照 02 的标准进行相机移动。"

### 解答
**根因**: `grasp_hawor.py` 的 `cam_view` 相机只在初始化时用第一帧 `R_c2w[0]/t_c2w[0]` 设置一次 `set_local_pose`, 主循环 (L2675-2685) 中只调用 `take_picture()` 没有更新 `local_pose`, 所以相机保持第 0 帧位姿不动。

**02_render_scene.py 标准** (L2588-2590): 每帧用 `hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])` 更新 `camera.set_local_pose`。

**修复** (对齐 02): 主循环中每帧按 `global_idx = self.start_frame + local_idx` 取出对应帧的 `R_c2w_all[global_idx] / t_c2w_all[global_idx]`, 调用 `hawor_cam_to_sapien_pose` 得到 `cam_pos, cam_quat`, 再 `cam_view.set_local_pose(sapien.Pose(...))` 更新。这样第一人称视角会严格跟随 HaWoR 相机轨迹移动, 与 02 一致。

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

