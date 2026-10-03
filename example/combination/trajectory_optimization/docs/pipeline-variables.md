# 管线变量参考手册

> 本文档逐步说明 MANO→夹爪重定向管线中每个阶段用了什么变量、从哪来、怎么算。

---

## 管线总览

```
HaWoR SLAM 追踪
  │
  ▼
① 原始数据 (pred_trans/rot/hand_pose/betas)
  │
  ▼
② 坐标变换 (SLAM → GLB)
  │  用 transform_params.npz 的 R_hand_to_glb, t_hand_to_glb, scale_ratio
  │
  ▼ mano_palm_glb (113帧手腕轨迹)
  │
③ pre_translate (场景对齐)
  │  F50 手指中点 → 物体中心，整体平移
  │
  ▼ traj["pos"] (平移后的113帧位置)
  │
④ neutral_offsets (微调)
  │  再算一次 obj_center - finger_mid，供 stage2/3 用
  │
  ▼ _mano_neutral_offset
  │
⑤ 平滑 (P0-2, 可选)
  │  对 pos/R/joint 做 Savitzky-Golay + Min-Jerk 平滑
  │
  ▼ 平滑后的 traj["pos"]/["R"]/["j1"]/["j2"]
  │
⑥ Stage1: CMA-ES 抓取优化
  │  输入: 平移后 MANO 轨迹 + 物体位姿
  │  输出: best_grasp (F50 的 pos/R/gripper)
  │
  ▼ best_grasp.npz
  │
⑦ Stage2: 轨迹重建
  │  输入: best_grasp + MANO 轨迹 + neutral_offset
  │  输出: stage2_recon (F38-F62 平滑重建)
  │
  ▼ stage2_recon.npz
  │
⑧ Stage3: 物理仿真优化
  │  输入: MANO 轨迹 + stage2_recon + neutral_offset
  │  输出: opt_pos (最终夹爪轨迹) + 物理仿真结果
  │
  ▼ stage3_result.npz
```

---

## ① 原始 SLAM 数据

**来源**: HaWoR 手部追踪系统（`/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz`）

**文件内容**:
| 字段 | 形状 | 含义 |
|---|---|---|
| `pred_trans` | `(1, N, 3)` | 每帧手腕平移（HaWoR 坐标系） |
| `pred_rot` | `(1, N, 3, 3)` | 每帧手腕旋转矩阵（HaWoR 坐标系） |
| `pred_hand_pose` | `(1, N, 45)` | 每帧 15 个关节 × 3 轴角 |
| `pred_betas` | `(1, N, 10)` | 手部形状参数 |
| `pred_valid` | `(1, N)` | 有效帧标记（bool） |

**处理**: 过滤无效帧（NaN / pred_valid=False），得到 `N_valid` 帧。

**坐标系**: HaWoR 自己的坐标系，跟 GLB 场景不一样。手在 HaWoR 坐标系里 Z>0（地面上方），但跟场景地面没有对齐关系。

---

## ② 坐标变换 (SLAM → GLB)

**目的**: 把 HaWoR 坐标系的手部数据转换到 GLB 世界坐标系（跟场景物体同一坐标系）。

**参数来源**: `alignment/transform_params.npz`（由 `_align_scene` 步骤计算）

**关键参数**:
| 参数 | 值/形状 | 含义 |
|---|---|---|
| `scale_ratio` | float (3.168) | HaWoR→GLB 缩放比 |
| `R_hand_to_glb` | (3,3) | 旋转矩阵 |
| `t_hand_to_glb` | (3,) | 平移向量 |
| `Rx_hand` | (3,3) = diag(1,-1,-1) | 手部坐标系翻转（Y/Z 轴反向） |

**变换公式**:
```python
# R_hand = R_hand_to_glb @ Rx_hand
# p_glb = scale_ratio * R_hand @ p_hawor + t_hand_to_glb

mano_palm_glb = scale_ratio * (R_hand_to_glb @ Rx_hand @ mano_palm.T).T + t_hand_to_glb
```

**结果**: `mano_palm_glb` — (N, 3) 的手腕在 GLB 世界里的轨迹。

**⚠️ 注意**: 此时轨迹还没跟场景对齐。手腕可能在地面以下（Z<0），因为 SLAM 有累积漂移。这是正常的，下一步 pre_translate 会修正。

---

## ③ pre_translate (场景对齐)

**目的**: 把整条 MANO 轨迹平移，让第 50 帧（F50，抓取帧）的手指中点正好在物体中心。

**代码**: `grasp_hawor.py:4359` `_pre_translate_mano_trajectory(side)`

**输入**:
- `traj["pos"]`: 113 帧的位置（坐标变换后，pre_translate 前）
- `traj["R"]`: 113 帧的旋转矩阵
- `obj_center`: 目标物体的 bounding box 中心（从 `self.obj_bbox_centers[tgt]` 获取）

**计算过程**:
```python
# 1. 取 F50 帧的基座位置和旋转
F50_IDX = min(n_frames - 1, 50)
R_f50 = traj["R"][F50_IDX]          # F50 的旋转矩阵
pos_f50 = traj["pos"][F50_IDX]      # F50 的基座位置

# 2. 算手指中点（基座前方 3.689cm，沿夹爪 X 轴）
finger_forward = [0.03689, 0.0, 0.0]  # 夹爪本体系：X 轴 = 手指伸出方向
finger_mid = pos_f50 + R_f50 @ finger_forward

# 3. 算平移量
pre_offset = obj_center - finger_mid

# 4. 整条轨迹平移
traj["pos"] = traj["pos"] + pre_offset   # 所有 113 帧都加这个偏移
```

**输出**:
- `traj["pos"]`: 平移后的位置轨迹
- `pre_offset`: 平移向量（日志里会打印）

**效果**: F50 帧时，手指中点 ≈ 物体中心。整条轨迹相对位置不变，只是整体平移了。

---

## ④ neutral_offsets

**目的**: 再算一次偏移量，供 Stage2/Stage3 使用。pre_translate 已经平移了轨迹，neutral_offsets 理论上 ≈ 0。

**代码**: `grasp_hawor.py:4393` `_compute_neutral_offsets()`

**计算过程**（跟 pre_translate 完全一样的逻辑）:
```python
obj_center = self.obj_bbox_centers[tgt]
finger_mid = mano_positions[F50] + R[F50] @ [0.03689, 0, 0]
offset = obj_center - finger_mid

self._mano_neutral_offset[side] = offset    # ≈ 0（因为 pre_translate 已对齐）
self._mano_grasp_frame[side] = F50_IDX      # 抓取帧索引
self._mano_target_pos[side] = obj_center    # 物体中心
```

**输出**:
| 变量 | 含义 |
|---|---|
| `_mano_neutral_offset[side]` | (3,) 偏移向量（pre_translate 后 ≈ 0） |
| `_mano_grasp_frame[side]` | int, 抓取帧索引（= 50） |
| `_mano_target_pos[side]` | (3,) 物体中心坐标 |

**为什么需要**: Stage2/3 的代码用 `_mano_neutral_offset` 来确保轨迹在正确坐标系。如果 pre_translate 已经跑了，这个 offset ≈ 0，不影响结果。

---

## ⑤ 平滑 (P0-2)

**目的**: 消除 SLAM 原始数据的帧间抖动，让携带段轨迹更平滑。

**代码**: `grasp_hawor.py:4435` `_smooth_mano_traj(side)` + `mano_smoothing.py`

**输入**: `traj["pos"]`, `traj["R"]`, `traj["j1"]`, `traj["j2"]`

**平滑对象**:
| 字段 | 平滑方法 | 参数 |
|---|---|---|
| `pos` (位置) | Savitzky-Golay 滤波 | window=17, polyorder=3 |
| `R` (旋转) | 先转四元数 → Gaussian 滤波 → 转回矩阵 | sigma=3 |
| `j1/j2` (关节) | Min-Jerk 插值（>4cm 跳变时触发） | max_vel=0.020, max_acc=0.005 |

**⚠️ 顺序问题**: 平滑必须在 `pre_translate` **之前**调用（让 pre_translate 在平滑后的轨迹上重新对齐 F50）。当前代码里 stage0/1/2 的顺序还有问题（平滑在 neutral_offsets 之后），stage3 已修正。

**输出**: 平滑后的 `traj["pos"]`/`["R"]`/`["j1"]`/`["j2"]`（原地修改）

---

## ⑥ Stage1: CMA-ES 抓取优化

**目的**: 找到最优的夹爪位姿（F50 帧的 pos/R/gripper），让夹爪能成功夹住物体。

**输入**:
- 平移后 MANO 轨迹（`traj["pos"]`, `traj["R"]`）
- 物体位姿（位置 + 包围盒）
- `_mano_neutral_offset`（≈ 0）

**优化方法**: CMA-ES（协方差矩阵自适应进化策略），42 维参数（spline keyframes）

**输出**: `best_grasp.npz`
| 字段 | 含义 |
|---|---|
| `pos` | (3,) F50 帧夹爪基座位置 |
| `R` | (3,3) F50 帧夹爪旋转矩阵 |
| `gripper` | float, 夹爪开合度（0=全开, 1=全闭） |
| `obj_name` | str, 目标物体名称 |

---

## ⑦ Stage2: 轨迹重建

**目的**: 从 Stage1 的单帧抓取位姿，重建 F38-F62 的平滑轨迹（接近→抓取→保持）。

**代码**: `stage2_reconstruct_mixin.py:19` `reconstruct_trajectory()`

**输入**:
- `best_grasp`: Stage1 输出的抓取位姿
- `traj["pos"]`/`["R"]`/`["j1"]`/`["j2"]`: MANO 轨迹
- `_mano_neutral_offset`: 偏移量（≈ 0）

**重建逻辑**:
| 帧段 | 策略 |
|---|---|
| F38-F49 (接近) | 从 MANO 位置插值到 F50，夹爪逐渐闭合 |
| F50 (抓取) | = Stage1 优化结果（pos/R/gripper） |
| F51-F58 (保持) | 位置锁 F50，夹爪保持闭合 |
| F59-F62 (过渡) | smoothstep 过渡回 MANO 轨迹 |

**关键设计**: 用 Minimum Jerk（最小加加速度）五次插值，保证轨迹平滑（位置/速度连续，加速度为 0）。

**输出**: `stage2_recon.npz`
| 字段 | 含义 |
|---|---|
| `pos` | dict: {帧号: (3,)} 重建位置 |
| `R` | dict: {帧号: (3,3)} 重建旋转 |
| `frames` | list: 重建帧号列表 |
| `gripper` | dict: {帧号: float} 夹爪开合度 |

---

## ⑧ Stage3: 物理仿真优化

**目的**: 在 SAPIEN 物理引擎中验证轨迹，生成最终的 113 帧夹爪轨迹。

**代码**: `stage3_optimize_mixin.py:128` `rollout_v4_stage3()`

**输入**:
- `traj["pos"]`/`["R"]`/`["j1"]`/`["j2"]`: MANO 轨迹（平滑+平移后）
- `_v4_stage2_recon`: Stage2 重建轨迹
- `_mano_neutral_offset`: 偏移量（≈ 0）
- `_v4_best_grasp`: Stage1 抓取位姿

**全局基底构建**:
```python
# 基底 = MANO 轨迹 + neutral_offset（≈ MANO 本身）
opt_pos_full = mano_pos + _neutral_off

# 分段覆盖:
# F0-F49:   MANO 轨迹（v4.41 高空飞行 + 边降边闭）
# F38-F62:  Stage2 重建（覆盖 MANO）
# F56-F89:  Stage2 F55 → MANO smoothstep 过渡（携带段）
# F90-F112: MANO 轨迹（释放段，j1/j2 信号检测释放帧）
```

**夹爪控制**:
- F50-F95: 保持夹持（gripper = 闭合值）
- F90+: 检测 j1/j2 信号（手指重新张开）→ 释放

**输出**: `stage3_result.npz`
| 字段 | 含义 |
|---|---|
| `mano_pos` | (N,3) 平移后 MANO 位置（pipeline ref） |
| `opt_pos` | (N,3) 最终优化后的夹爪基座位置 |
| `obj_pos_traj` | (N,3) 物体在仿真中的实际位置 |
| `result` | dict: 包含所有上述字段 + 物理仿真结果 |

---

## 验证指标

### validate_carry (恒定距离验证)

**代码**: `physics_utils.py:189`

**用途**: 验证携带段（F50→F95）手指中点与物体保持恒定距离（刚性跟随）。

**输入**:
- `fmid_traj`: (N,3) 手指中点轨迹 = `mano_pos + R @ [0.03689, 0, 0]`
- `obj_traj`: (N,3) 物体位置轨迹
- `grasp_frame`: 夹持起始帧（= 50）

**计算**:
```python
d = ||fmid_traj - obj_traj||           # 每帧距离
ref = d[grasp_frame]                    # 基准距离（F50 时）
dev = |d - ref|                         # 偏差
max_dev = max(dev[grasp_frame:end])     # 最大偏差
```

**输出**:
| 字段 | 含义 | 判定 |
|---|---|---|
| `ref_dist_mm` | F50 时手指中点↔物体距离 (mm) | — |
| `max_dev_mm` | 携带段最大偏差 (mm) | ≤ 5mm → PASS |
| `mean_dev_mm` | 携带段平均偏差 (mm) | — |
| `passed` | bool | `max_dev ≤ CARRY_DEV_TOL_M` |

### compute_tracking_error (跟随误差)

**代码**: `physics_utils.py:145`

**用途**: 验证仿真中夹爪实际位置 vs 目标轨迹的跟随精度。

**输入**:
- `actual_pos_traj`: (N,3) 仿真实际基座位置（从 qpos 提取）
- `target_pos_traj`: (N,3) 目标轨迹（送进 PD 控制器的 target_pos）

**输出**:
| 字段 | 含义 |
|---|---|
| `mean_mm` | 平均跟随误差 (mm) |
| `max_mm` | 最大跟随误差 (mm) |
| `final_mm` | 末帧跟随误差 (mm) |

### 其他 Stage3 指标

| 指标 | 含义 | 来源 |
|---|---|---|
| `lift` | 物体被提起的最高 z 坐标 (cm) | `obj_pos_traj[:,2].max()` |
| `lift_final` | 物体最终 z 坐标 (cm) | `obj_pos_traj[-1,2]` |
| `drift` | 携带段物体 xy 漂移 (cm) | `norm(obj_xy[F50] - obj_xy[F95])` |
| `release_z_std` | 释放后物体 z 稳定性 (mm) | `std(obj_z[release:])` |
| `obj_mano_mean` | 物体↔MANO 参考点平均距离 (mm) | `mean(norm(obj - mano_ref))` |

---

## 关键帧索引

| 帧号 | 含义 |
|---|---|
| F0-F37 | 接近期（手从远处移向物体） |
| F38-F49 | 接近末段（边走边闭合） |
| **F50** | **抓取帧**（手指中点对齐物体中心） |
| F51-F58 | 夹持保持段（锁 F50 位姿） |
| F59-F62 | Stage2 过渡段（smoothstep 回 MANO） |
| F63-F89 | 携带段（跟随 MANO 轨迹，物体被夹着移动） |
| F90-F112 | 释放段（手指张开，物体放下） |

---

## 变量依赖关系图

```
HaWoR .npz
  ├─ pred_trans/rot/hand_pose/betas
  │     │
  │     ▼
  │   MANOLayer FK → mano_palm (手腕3D坐标)
  │     │
  │     ▼
  │   xform(mano_palm, scale, R_hand, T_H2G)  ← transform_params.npz
  │     │
  │     ▼
  │   mano_palm_glb (113帧, GLB坐标)
  │     │
  │     ▼
  │   _mano_gripper_traj[side] = {pos, R, j1, j2}
  │     │
  │     ├── pre_translate → traj["pos"] += pre_offset
  │     │
  │     ├── _compute_neutral_offsets → _mano_neutral_offset ≈ 0
  │     │
  │     ├── _smooth_mano_traj → traj["pos"/"R"/"j1"/"j2"] 平滑
  │     │
  │     ├── Stage1 CMA-ES → best_grasp.npz
  │     │
  │     ├── Stage2 reconstruct → stage2_recon.npz
  │     │
  │     └── Stage3 rollout → stage3_result.npz
  │           ├─ opt_pos (最终轨迹)
  │           ├─ obj_pos_traj (物体轨迹)
  │           └─ validate_carry → max_dev, ref, passed
  │
  └─ obj_bbox_centers[tgt] → obj_center (物体中心, pre_translate 的目标)
```
