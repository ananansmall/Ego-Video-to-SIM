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

