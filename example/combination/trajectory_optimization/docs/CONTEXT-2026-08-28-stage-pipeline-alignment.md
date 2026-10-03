# Context: Stage 0/1/2/3 抓取管线 — 轨迹对齐与可视化诊断

> 生成时间: 2026-08-28
> 范围: `example/combination/trajectory_optimization/` 的 4-stage 抓取管线
> 状态: **诊断中，对齐问题未解决**

---

## 1. 管线总览

4 个 stage，每个 stage 是**独立的 simulator 实例**，各自调用 `_align_scene()`：

| Stage | 作用 | 帧数 | 关键产物 |
|-------|------|------|----------|
| Stage0 | 纯 MANO retargeting（`set_drive_target` 跟踪） | 113 | `grasp_gripper_only_right_qpos.npy` (8维: pos+rot_euler_ZYX+gripper) |
| Stage1 | CEM 抓取位姿优化 | 160 | `best_grasp.npz` (obj_lift=18.87cm) |
| Stage2 | 轨迹重建 + dz 网格搜索 | 120 | `stage2_recon.npz` (25帧), dz=+0.0100 |
| Stage3 | 完整管线 | 112 | `stage3_result.npz` (obj_lift=2.98cm) |

输出目录结构（已重组）:
```
output/gripper_right/
├── stage0/  (113帧)  cam/ + stage0.log + grasp.log + qpos.npy + alignment/
├── stage1/  (160帧)  3 videos + best_grasp.npz + alignment/
├── stage2/  (120帧)  3 videos + best_grasp.npz + stage2_recon.npz + alignment/
└── stage3/  (112帧)  3 videos + best_grasp.npz + stage3_result.npz + alignment/
```

---

## 2. 关键代码位置

- `grasp_hawor.py`
  - `_pre_translate_mano_trajectory` (line 2670): 把 MANO 轨迹平移到物体中心，计算 `pre_offset`
  - `_compute_mano_neutral_target` (line 4355): MANO 指尖 → 夹爪 base，用 `FINGER_OFFSET = [0.03689, 0, 0]`
  - `rollout_v4_stage3` (line 4730)
  - `reconstruct_trajectory` (line 4680): 只生成 25 帧 (frame 38-62)
- `stage2_reconstruct_mixin.py` (line 201): `rollout_follow_trajectory` 定义
- `05_gripper_test.py` (line 1348-1900): test8 — `plan_err` 是 **base 位置误差**，不是指尖误差

---

## 3. 已澄清的误解（重要）

### 3.1 "19cm PD 跟踪误差" 是错的
最初把 Stage0 的夹爪 base 位置 (`qpos[:3]`) 和 Stage3 的 MANO 指尖位置直接对比，得到 17.9cm 误差。
**这是坐标系统错误**：base 位置 ≠ 指尖位置。正确做法是从 `qpos[3:6]` (euler ZYX) 提取旋转，算 `fingertip = base + R @ [0.03689, 0, 0]`。

### 3.2 "平移不一致" 是错的
用户纠正：**只有一个 `pre_offset` 平移向量，不存在"不一致"**。
Stage0 执行顺序：
1. `sim.run()` — 第一次运行（无平移）
2. `sim._pre_translate_mano_trajectory(side)` — 计算 `pre_offset` 并平移整条 MANO 轨迹
3. `sim._compute_neutral_offsets()` — 计算 `offset`（已平移后 ≈ 0）
4. `sim.run()` — 第二次运行（用平移后轨迹）

最终夹爪位置 ≈ `pre_translated_mano_pos`，逻辑正确。

### 3.3 跨 Stage 比较绝对坐标没有意义
每个 Stage 是独立 simulator 实例，各自 `_align_scene()`，物体在世界坐标中的绝对位置可能不同，`pre_offset` 也不同。
**跨实例比较绝对坐标无意义**。Stage0 内部跟踪误差应为 PD 控制器级别（几毫米）。

### 3.4 `rollout_follow_trajectory` 是插帧回放
用户确认理解正确：把 25 帧重建轨迹 (frame 38-62) 插值展开成 120 帧完整仿真（接近→抓取→抬起→搬运），**仅用于可视化**，不是真正的控制策略。

---

## 4. 当前未解决的问题（用户最新反馈）

用户指出（2026-08-28）：

1. **Stage0/1/2/3 差距过大**，特别是 F50 附近。Stage1 看上去明显偏差很大，和原始 F50 相比。
2. **怀疑完全没有读对轨迹**。
3. **对齐顺序问题**：目前是"先平滑后平移"，用户认为对齐问题仍然很大。
4. **`render_quick.py` 已给出正确的对齐方式**，需要参考它。

### 4.1 `render_quick.py` 的正确对齐方式（参考基准）

`example/combination/View/render_quick.py` 的 `load_data` 函数（line 141+）：
- 通过 **MANO FK** 计算关节点（拇指尖 j20、食指尖 j16、手腕 j0）
- 使用**逐帧真实相机位姿** `R_c2w[f]` / `t_c2w[f]` 转到 HaWoR 世界系
- 再过相似变换 `(s, R, t)` 到 GLB 坐标
- **关键注释**：不使用固定 `Rx_hand=diag(1,-1,-1)` —— 那只是 `R_c2w[0]≈diag(1,-1,-1)` 的第一帧特例近似，相机转动后（实测可达 34°）会产生明显漂移

**这是正确的对齐方式**，trajectory_optimization 里的对齐逻辑需要对照检查。

---

## 5. 待办事项

### 5.1 生成新的绘图代码（用户明确要求）
- **只要一张图**：左手 3D trajectory
- 不要很多其他分析
- 图片放置到 `output/gripper_right/photo/` 目录
- 前面的绘图代码先保留

### 5.2 排查轨迹读取问题
- 检查 trajectory_optimization 里 MANO 轨迹的读取方式是否与 `render_quick.py` 一致
- 重点检查：是否用了固定旋转近似而非逐帧相机位姿
- 检查"先平滑后平移"的顺序是否正确

### 5.3 排查 Stage1 偏差
- Stage1 的 CEM 优化结果与原始 F50 偏差大，需确认是否读对了轨迹

---

## 6. 相关文件清单

### 绘图脚本（View/）
- `vis_stage_trajectories_right.py` — 4 子图轨迹对比（3D/XZ/Z-frame/XY）
- `vis_stage0_error_analysis.py` — Stage0 误差分析 + 平滑
- `vis_stage_trajectories_left_v2.py` — 左手版本（用户提到的参考）

### 参考对齐
- `example/combination/View/render_quick.py` — **正确对齐方式基准**

### 测试参考
- `example/combination/05_gripper_test.py` — test8 跟踪误差计算

### 外部参考
- `do-as-i-do/retargeting/utils/tracking_error.py` — 跟踪误差计算参考
- `do-as-i-do/retargeting/pipeline/solve_ik.py` (line 265) — `moving_average_filter` 参考

---

## 7. 关键结论

1. Stage0 内部跟踪逻辑正确，17.9cm 是跨实例坐标比较错误
2. `rollout_follow_trajectory` 是插帧回放，仅用于可视化
3. **当前核心问题**：trajectory_optimization 的轨迹读取/对齐方式可能不对，需对照 `render_quick.py` 修正
4. 用户要求：生成一张左手 3D trajectory 图，放到 `output/gripper_right/photo/`
