# STAGE 问题收集库

> **用途**: 管线**当前仍存在**的问题清单。只列开放问题, 不列已修复。
> **维护**: 问题解决后**从此文件移除**, 并记入 `CHANGE_LOG.md`。本文件不留 ✅ 条目。
> **最近更新**: 2026-09-11

---

## 优先级定义

| 级别 | 含义 |
|------|------|
| **P0** | 阻塞: 结果不可信或产物不可用, 必须优先解决 |
| **P1** | 高: 影响正确性/可复现性, 或明确的功能缺口 |
| **P2** | 中: 影响可维护性/一致性, 不阻塞当前结果 |
| **P3** | 低: 清理/文档类, 可择机处理 |

---

## 顶层摘要 — 开放问题由大到小

> 按对**生产管线 (v4-pipeline / stage1-3) 正确性的真实影响**排序, 兼顾作用范围。

| 序 | ID | 问题 (一句话) | 级别 | 改动量 |
|---|----|------|------|------|
| 1 | P0-1 | Stage3 夹持滑脱: 恒定距离 max_dev=8.6mm FAIL (主症, v6.9 已改善 63%) | P0 | 中 |
| 2 | P0-2 | MANO 原始逐帧无滤波接入 PD (P0-1 潜在诱因) | P0 | 中 |
| 3 | P2-7 | MANO 变换链疑整体 Z 翻转 (指尖比手腕高 13.45cm) | P2 | 中 |
| 4 | P1-5 | Stage3 残留两个碰撞 hack, 验证环境≠真实环境 | P1 | 中 |
| 4 | P1-7 | `--grasp-pose-method` 漏传 stage1/2/3, svd_palm 对管线无效 | P1 | 3 行 |
| 5 | P1-8 | stage2 漏 `_pre_translate_mano_trajectory`, 用已废弃大 offset 形式 | P1 | 1 行 |
| 6 | P1-6 | `--side right` 可能跟到 HaWoR 幻觉手 | P1 | 小 |
| 7 | P1-3 | npz 无序 glob 取首个, 加载不确定 (不可复现) | P1 | 小 |
| 8 | P1-9 | stage0 双重 run() 第二次重建 precompute, 覆盖平移轨迹 | P1 | 中 |
| 9 | P1-10 | `--f50-tilt-deg` 漏设 stage3, stage3 倾角固定 0 | P1 | 1 行 |
| 10 | P1-4 | F50 硬编码 50, 轨迹长度变化时语义漂移 | P1 | 小 |
| 11 | P1-1 | 旧分析脚本 vis_stage0_error_analysis 仍报假指标 19.36cm | P1 | 小 |
| 12 | P2-3 | 恒定距离 ref 绝对值 59cm 异常 (max_dev 判定仍可信) | P2 | 小 |
| 13 | P2-1 | v4.49 抬升上限 3.1cm (F63-F89 硬编码 1.2mm/帧, 设计约束) | P2 | — |
| 14 | P2-2 | View 脚本 best_grasp/stage2_recon 路径三处不一致 | P2 | 小 |
| 15 | P2-4 | `_mano_to_sapien` 两处等价定义, 未复用公共函数 | P2 | 小 |
| 16 | P2-5 | 几何可行性硬钳制 (enclose_z/ground_clearance/feasible 元数据) 缺失 | P2 | 中 |
| 17 | P2-6 | stage3 旧 npz 兜底硬编码 `_obj_z=0.0173` | P2 | 小 |
| 18 | P3-1 | README/CHANGE_LOG/docs 未同步近期改动 | P3 | 小 |
| 19 | P3-3 | 根目录 s2/s3/run_stage*.log 调试产物未归类 | P3 | 小 |
| 20 | P3-4 | stage2_recon.npz 缺 R 字段 (当前无读取方) | P3 | 1 行 |
| 21 | P1-11 | Stage2 接近段轻擦目标物体 (APPROACH_TARGET_CONTACT 未过) | P1 | 小 |
| 22 | P2-8 | Stage0 映射过零硬阈值在近水平段产生 180° 姿态台阶 | P2 | 小 |

**最小修复集 (5 行)**: P1-7 + P1-8 + P1-10 — 4 个 stage 构造参数与对齐序列即完全一致。
**P0-1/P0-2** 为生产管线主症; **P1-9 (stage0 双重 run)** 建议先实测确认再改。

---

## P0 — 阻塞

### P0-1 Stage3 夹持滑脱 (恒定距离 FAIL)
**现象**: 手指咬住物体但抬升中滑脱, 物体未跟随 MANO 轨迹。
```
[v6.9 修正后 2026-09-02 复跑]
[PhysicsValidation][stage3] 恒定距离: ref=31.3mm, max_dev=7.8mm (tol=5mm) → FAIL ✗
[Stage 3] obj-mano距离: mean=57.1mm, F50=39.2mm, F80=77.3mm
[Stage3 偏差②] 跟随误差: mean=1.0mm, max=41.5mm
[修正前] 恒定距离 ref=7.1mm, max_dev=23.5mm | obj-mano mean=100.6mm, F80=152.9mm
```
**关键洞察**: 跟随误差仅 `mean=1.0mm` → **控制精度没问题, 是夹不紧/夹持几何问题**。
**v6.9 进展 (复跑稳定)**: 夹爪 approach 轴朝下对齐后 `max_dev 23.5mm → 7.8mm` (改善 67%),
obj-mano mean `100.6 → 57.1mm` (-43%), F50 `54.2 → 39.2mm`, F80 `152.9 → 77.3mm`。
方向正确但未达 5mm 容差。
**证据**: `stage3_optimize_mixin.py` 闭合时序; `docs/2026-08-31-stage3-flow-and-problems.md` §2.1
**状态**: 🔴 未解决 — 主症 (v6.9 大幅改善, 剩余 8.6mm vs 容差 5mm)

### P0-2 MANO 轨迹未平滑
**现象**: 管线直接把 MANO 原始逐帧输出喂 PD, 无滤波。而 `05_gripper_test.py:269` 已有四重平滑流水线
(Savitzky-Golay + 高斯四元数 Slerp + min-jerk + EMA), **当前管线完全没接入**。
**证据**: `trajectory_optimization/` 内搜 `smooth_mano_trajectory|savgol_filter|min_jerk_smooth|EmaTargetSmoother` → **0 结果**
**影响**: MANO 逐帧噪声直接驱动 PD, 可能加剧夹持抖动/滑脱 (P0-1 的潜在诱因)。
**注意**: 但实测平滑对"恒定系统偏移"无效 — 若偏差是标定/坐标系问题, 平滑无用 (见 P1-1 教训)。
**状态**: 🔴 未解决

---

## P1 — 高

### P1-1 旧分析脚本报假指标
**现象**: `View/vis_stage0_error_analysis.py` 报 "Stage0 vs MANO = 19.36cm", 实为**基准差** (Stage0 neutral_offset 0.517m vs Stage3 pre_translate 0.666m 的恒定平移), 非跟踪误差; 速度完全吻合 (0.3511 vs 0.3534 m/s)。
**另**: 该脚本 L19-20 直接取 `s0_qpos[:, :3]` 当基座位置, 未用 `virtual_idx`, 列序依赖脆弱。
**待办**: 旧脚本改用 `physics_utils.compute_tracking_error()` (同基准口径) 或废弃。
**状态**: 🔴 旧脚本仍报假指标, 待修正/废弃

### P1-3 npz 加载非确定性
**现象**: 无序 glob 取第一个, 不保证选到 `depth_aligned` 版本。
**证据**: `trajectory_loader.py:48-50` 与 `:125-127`
```python
for f in rec_dir.glob("hawor_results_*.npz"):
    npz_file = f
    break
```
**影响**: 不同环境/文件系统顺序下可能加载不同 npz → 结果不可复现。
对照 `View/vis_left_hand_3d_trajectory.py:44` 已正确优先 depth_aligned。
**状态**: 🔴 未解决

### P1-4 F50 硬编码
**证据**: `stage1_grasp_mixin.py:54` `f50_idx = min(50, n - 1)`; `_pre_translate_mano_trajectory` 内同有硬编码 50。
**影响**: 轨迹长度变化时 F50 语义漂移, 应接 `traj_optimize.compute_frame_params(N)['F50_IDX']`。
**状态**: 🔴 未解决

### P1-5 Stage3 残留两个碰撞 hack
**现象**: Stage3 仍禁用碰撞, 与"优化环境 = 验收环境"原则冲突 (Stage1 已删除)。
- 手指-地面碰撞禁用
- 邻物碰撞禁用 (7 shape, 仅保留 target `glb_4`)
**证据**: `stage3_optimize_mixin.py` rollout 开头; `docs/2026-08-25-full-pipeline-diagnosis.md` P0-2
**影响**: 通过验证的解在无 hack 的真实环境下可能不可行。
**状态**: 🔴 未解决

### P1-6 side 与 hand_idx 未解耦
**现象**: `--side right` 时 `hand_idx` 仍可能跟到 HaWoR **幻觉手** (HaWoR 对单手序列会输出两只手, 其中一只是幻觉)。
**证据**: `grasp_hawor.py` side→hand_idx 映射 (left→0 / right→1); `View/vis_left_hand_3d_trajectory.py:45`
注释明确 `HAND_IDX = 0  # 0=左手(真实操作手); 1=右手(HaWoR 幻觉, pred_trans z≈0)`
**状态**: 🔴 未解决

### P1-7 `--grasp-pose-method` 漏传 stage1/2/3
**证据**: 仅 stage0 构造传 `grasp_pose_method=args.grasp_pose_method` (L4467); stage1(L4523)/stage2(L4590)/stage3(L4733) 构造都漏 → `__init__` 默认 `"analytical"` (L185)。
**影响**: CLI `--grasp-pose-method svd_palm` **只对 stage0 生效**; stage1/2/3 的 `_mano_gripper_traj` 预计算永远走 analytical (加权 SVD), 用户选择的 svd_palm 对生产管线静默无效。
**修复**: stage1/2/3 构造各加 `grasp_pose_method=args.grasp_pose_method,` (共 3 行)。
**状态**: 🔴 未解决

### P1-8 stage2 漏 `_pre_translate_mano_trajectory`
**证据**: stage0/1/3 都 `pre_translate_mano_trajectory(side) → _compute_neutral_offsets()`; **stage2 (L4607) 直接 `_compute_neutral_offsets()`, 跳过 pre_translate**。
**机制**: 两函数同公式 `offset = obj_center - (mano_pos[F50]+R[F50]@[0.03689,0,0])`。pre_translate 把 pre_offset≈0.59m 烘进 `traj["pos"]` → neutral_offset≈0; stage2 不烘入 → neutral_offset≈0.6m (大)。rollout 若均匀加 offset 则数值等价, 但任何直接读 `traj["pos"]` 的下游 (recon/指标/viz) 看到未平移版本; 大 offset 形式正是 stage3 v4.30 专门修掉的形态。
**修复**: stage2 在 `_compute_neutral_offsets()` 前加 `sim._pre_translate_mano_trajectory(side)` (1 行)。
**状态**: 🔴 未解决

### P1-9 stage0 双重 run() 覆盖平移轨迹
**现象**: stage0 跑两次 run(): 第一次 run(L4474) → pre_translate(L4476, 烘进平移) → neutral_offsets(L4477, offset≈0) → 第二次 run(L4516, `_opt_params=None`)。
**根因**: 第二次 run 里 `_has_opt_traj=None`, precompute 无条件重建 `mano_gripper_traj={}`(L1617) → `self._mano_gripper_traj=...`(L1697), **覆盖** L4476 的平移轨迹, 回到原始未平移位置。
**后果**: F50 日志(L4493-4509)打印平移后轨迹 (对齐物体), 但第二次 run 实际运动用未平移轨迹 → 日志≠实际; 回退目标 offset≈0 + traj 未平移 → stage0 最终夹爪实际不对齐物体 (差 ~0.59m)。即 stage0"平移后 MANO→夹爪运动"的"平移"在最终输出里被覆盖丢失。
**确认**: precompute 无条件重建 (L1617/L1697 + 门控 L1490/L1613) 已读证; 回退目标用 `traj["pos"]+neutral_offset` 已读证 (L643-666)。结论为分析确证, 建议改前用一次带日志的 run 实测复核。
**修复方向**: A 去掉第二次 run; B 二次 run 前不重建 precompute (加"已填充则跳过"门控); C stage0 不做 pre_translate/neutral_offsets, 纯跑一次基线 MANO 重定向。需实测后定。

**2026-09-02 实测确认 ✅ (Stage1 同样受影响)**: 连续两次 `--test-stage1` 稳定复现
"CEM 能找到好候选, 但最终选出失败位姿":
```
候选分布: obj_lift=23.44 / 23.41 / 23.15 / 20.53 ... (1226 个候选)
最终选出: reward=0.000, lift_final=[0.0, 0.0, 0.0]cm → 验证门 0/3 通过
```
即 CEM 评估时 lift=23cm, 验证/终选时 lift=0 — **评估与终选用的不是同一条轨迹**,
与 P1-9 描述的"precompute 被覆盖/平移丢失"现象一致。
**升级**: 由"待实测确认"改为**已确认**, 且不只是 stage0 — 影响 Stage1 验证门。
**下一步**: 先按方案 B (加"已填充则跳过"门控) 改, 改动最小; 改完用
`--test-stage1` 验证终选位姿的 lift 是否与 CEM 最优候选一致。
**状态**: 🔴 未解决 (现象已确认, 待修)

### P1-11 Stage2 接近段轻擦目标物体
**现象** (2026-09-11, scale 0.45 / q_open 0.05 / d_stop 0.16 / axial):
```
[Stage2 gate] APPROACH_TARGET_CONTACT: prefix/adjust=0/22, objects=['glb_4']
[Stage2 contact] left_gripper_finger_link2 x glb_4  n=71 first_ctrl=39 max_pen=1.9807mm
[Stage2 contact] left_gripper_finger_link1 x glb_4  n=59 first_ctrl=47 max_pen=0.1409mm
```
接近段仍有 22 个控制点轻擦目标物体 (穿透 0.14~1.98mm)。**但物体未被推走**:
anchor 处基座-物体 = `37.6mm` ≈ 设计值 `36.9mm` (= 手指伸出 3.689cm), 携带段稳定 `37.9mm`,
且 SETTLE `cmd=0.0086 / act=0.0248` → 确实夹住。scale 0.6 同配置则被推走 3.6cm (anchor 73.6mm)。

**根因 (已量化)**: 轴向逼近余量薄 —— 物体在 anchor 系 `y ∈ [-32.6, +47.9]mm`, 虎口内侧 `±55.0mm`
(q_open=0.05) → **+y 侧仅 7.1mm**。finger2 在 +y 侧, 实测 `first_ctrl=39` 先碰, 与预测一致。
q_open=0.075 时余量 +33.3mm, adjust 40→24; scale 0.45 时余量 +23.6mm, adjust 40→22。

**已排除 (本轮逐一验证)**:
- 碰撞检测假阳性 → 接触对就是 `finger_link × glb_4`, 且有真实穿透 (0.14~1.98mm), 视频可见手指确实擦到
- 碰撞体/渲染/scale 不一致 → `data_loader.py` L290-401 同一批 `vertices_sapien` 同时导出渲染 PLY 与
  `bbox_size/2` 碰撞盒, actor pose = bbox 中心 (严格一致)
- PD 跟踪 → FINAL_APPROACH `0.18mm / 0.02°`
- Stage0 映射被改动 → 已回退, 逐位复现基准 (帧间姿态 mean=7.63 max=179.54, >90° 帧对 [7,9,13,108])

**可行方向**: (a) 改判定口径为"接触后物体位移 < 阈值"或只在 FINAL_APPROACH 最后 2 点要求零接触;
(b) Stage2 侧把 anchor 沿夹持轴横向对中 (余量 7.1mm → ~20mm, 属 Stage2 自身位姿处理, 不碰 Stage0/1);
(c) 维持较小物体 scale。
**证据**: `output/deliverables_videos/STAGE2_scale045_q005_dstop016.mp4`; `output/distance_profile.csv`;
`output/check_zoom_scale060.png` (ctrl45 可见物体被挤偏) / `check_zoom_scale045.png` (物体不动)
**状态**: 🔴 未解决 (抓取成立, 仅门禁口径未过)

### P1-10 `--f50-tilt-deg` 漏设 stage3
**证据**: stage0(L4472)/stage1/stage2(L4604) 都设 `sim.f50_tilt_deg = float(args.f50_tilt_deg)`; **stage3 块 (L4719-4852) 全程无此赋值** → `__init__` 默认 0。
**影响**: stage3 夹爪倾角固定 0, 与 stage0/1/2 不一致; 用户传 `--f50-tilt-deg` 在 stage3 不生效。
**修复**: stage3 构造后加 `sim.f50_tilt_deg = float(args.f50_tilt_deg)` (1 行)。
**状态**: 🔴 未解决

---

## P2 — 中

### P2-1 v4.49 抬升上限 3.1cm
`F63-F89 线性 1.2mm/帧` 硬编码, Stage3 lift 上限即 3.1cm。
**注意**: 这导致 Stage3 的 lift 与 Stage1 (19.5cm) **不可比**, 解读结果时勿混淆。
若希望 Stage3 真正复现"人手抓起平移", 此约束是根本限制。
**状态**: 🔴 未解决 (设计约束, 需决策是否放宽)

### P2-2 View 脚本路径不一致
- `vis_stage2_trajectory.py:33` 用 `stage1/best_grasp.npz`
- `vis_stage0_error_analysis.py:31` 用 `stage1/stage1/best_grasp.npz`
- `vis_left_hand_3d_trajectory.py:113` 用 `glob("stage2/**/stage2_recon.npz")` 通配
**注**: v6.8 输出重构后路径已稳定 (`gripper_{side}/stageN/...`), 这些 View 脚本可改用新路径。
**状态**: 🔴 未解决

### P2-3 恒定距离 ref 绝对值异常
Stage2 报 `ref=590.5mm` (59cm) — 手指中点离物体不应这么远。
**影响**: `max_dev` (相对变化) 判定仍可信, ref 绝对值不可信。疑为 `get_entity_pose` vs `get_pose` 坐标系差异。
**证据**: `stage2_reconstruct_mixin.py` `fmid_obj_dist_history` 采集处
**状态**: 🟡 待排查 (不影响 max_dev 判定)

### P2-4 坐标变换代码冗余
`_mano_to_sapien` 两处等价定义; 未复用 `hand_track/common.py::_render_to_sapien`; `RXWORLD_TO_SAPIEN` 别名仍在使用。
**证据**: `grasp_hawor.py` `run` 与 `run_optimize` 内各一处

**另 (v6.9 新增)**: `compute_analytical_gripper_pose` 共 **3 份副本**, 仅 `data_loader.py` 那份
(主管线) 同步了 v6.9 的 `align_gripper_approach_to_gravity`:
- `data_loader.py:443` ← 主管线用 ✅ 已修复
- `View/vis_trajectory.py:80` ← 可视化, ❌ 未同步
- `trajectory_loader.py:171` ← ❌ 未同步 (且无调用方, 疑似死代码)
**影响**: 可视化脚本画出的夹爪姿态与仿真不一致 (仍朝上), 误导调试。
**状态**: 🔴 未解决

### P2-5 几何可行性硬钳制缺失
`enclose_z` 硬钳制 (obj_top+margin / 地面+指长)、`_probe_ground_clearance()`、`best_grasp.npz` 写 `feasible` 元数据 + 加载端校验 — 均缺失。
**证据**: `docs/2026-08-25-full-pipeline-diagnosis.md` P0-1/P0-3; 全仓无 `feasible` 校验
**状态**: 🔴 未解决

### P2-6 stage3 旧 npz 兜底硬编码 `_obj_z=0.0173`
**证据**: stage3 旧 `best_grasp.npz` 兼容兜底 (L4767-4777):
```python
_obj_z = 0.0173  # 从场景获取 (粉色物体初始 z)   ← 硬编码, 注释说"从场景获取"实际没取
_enclose_z_loaded = _obj_z + 0.030
```
**影响**: 换物体/场景, `enclose_z` 估计错误。仅旧 npz (无 enclose_z 字段) 兜底路径触发, 新 npz 不触发。
**修复**: `_obj_z` 从 `sim.obj_info`/`obj_bbox_centers` 实际读取。
**状态**: 🔴 未解决

### P2-7 MANO 变换链疑整体 Z 翻转
**现象**: 即便 v6.9 已把夹爪 R 翻转为朝下, **MANO 手本身在该变换链下仍是朝上的**:
```
MANO 指尖相对手腕高度: mean=+13.45cm  (指尖比手腕高 13cm)
```
人手伸向桌面物体时指尖应**低于**手腕 (负值)。量级 13cm ≈ 手掌+手指长度, 说明手几乎竖直朝天。
**疑点**: 上游 `R_h2g` / `R_c2w` / `R_AXIS` 变换链可能存在整体 Z 翻转。
**现状**: v6.9 在 R 构造处做 `align_gripper_approach_to_gravity` 属**下游修正**, 已验证有效
(Stage1 2/3→3/3, Stage3 滑脱 -63%), 但未触及上游根因。
**待办**: 核查 `_mano_to_sapien` 变换链 (grasp_hawor.py L1640) 是否应整体翻 Z。

**上游翻转实验 (v6.9b, 2026-09-02) — 已实现但默认关闭**:
实现了 `physics_utils.flip_mano_hand_down()` (绕水平轴 `h = cross(z_up, pointing)` 转 180°,
以腕为支点; 重力点积判据; 幂等), 挂在 `_mano_to_sapien` 出口, 由 `GraspSimulator.flip_mano_hand`
开关控制 (**默认 False**)。

- 单独测试该函数**指标全部正确**: 掌法 z `+0.846→-0.846`, 指尖-腕 `+14.31→-14.31cm`,
  指尖高度 `0.574→0.288` (逼近物体 0.039), 解算 R 的 `dot(X,重力)` 自然为 `+0.326`
- 但**接入管线后 Stage1 严重退步**: 恒定距离 `ref=2.7m / 31.6cm`, `max_dev=9-51mm` FAIL
  (关闭时 `ref=3.8mm, max_dev=2.3mm` PASS)
- 根因: 翻转以腕为支点把指尖上下移动 ~28cm, 改变了 MANO 轨迹的**绝对位置**;
  而 CEM 搜索空间 / enclose_z / 物体对齐按"未翻转轨迹"标定, 翻转后 CEM 参考位姿
  掉到物体下方 (`base_z=0.0107` < 物体 `0.0393`), 候选全部偏离
- 结论: 下游 `align_gripper_approach_to_gravity` 只改 R、不动 MANO 指尖位置, 才是最小侵入解。
  上游翻转若要启用, 需**同步重新标定** CEM 搜索空间与 enclose_z
**证据**: `test/diag_mano_R_orientation.py` 【F】段; `test/run_s1_upflip.log`
**状态**: 🔴 待核查 (上游翻转能力已备好, 待重新标后再启用)

### P2-8 Stage0 映射过零硬阈值产生 180° 姿态台阶
**现象** (2026-09-11): 预处理后的夹爪姿态序列 `grip_R` 在相邻帧间出现**真实的 180° 台阶**:
```
帧间姿态变化 mean=7.63 max=179.54 deg; >90° 帧对 [7, 9, 13, 108]
F9->F10: 179.1° (施加 R@diag(-1,1,-1) 后残差 0.97°)  F13->F14: 178.8° (1.87°)  F108->F109: 179.3° (2.54°)
该段 X.z = -0.085 … -0.003 (approach 轴几乎水平)
```
**根因**: `physics_utils.align_gripper_approach_to_gravity` 用 `dot(X, 重力) < 0` 这一**过零硬阈值**,
被 `data_loader.compute_mano_based_gripper_pose` / `compute_analytical_gripper_pose` **逐帧独立**调用。
approach 轴接近水平时符号在相邻帧间来回跳 → 逐帧翻/不翻 → 180° 台阶。

**影响 (实测评估: 小)**: 台阶全落在 F0–F25 前缀段 (离物体 30cm+), 只造成 1 个控制周期的瞬态;
`MANO_FOLLOW` 段跟踪 `mean=1.10° / max=21.44°`, 去掉台阶后同段为 `1.10/21.44` (完全一致) → **非 Stage2 瓶颈**。
F108/F109 在 Stage2 用的 F0–F62 范围之外。

**注意**: 本函数位于 **Stage0 的 MANO→夹爪映射链路**, 它的输出就是 Stage0 复现的夹爪姿态本身。
2026-09-11 一度被改成"向上倾角 >10° 才翻"的死区判据, 因**改动了 Stage0 的输出**已按用户要求回退
(回退后逐位复现基准: mean=7.63 max=179.54, 帧对 [7,9,13,108])。
若要修, 应在**选符号那一步**取"与重力朝向更一致"的分支 (用户方案), 但**必须重新验证 Stage0**。
**证据**: `output/stage0_check/preprocess_traj.npz`; `physics_utils.align_gripper_approach_to_gravity`
**状态**: 🔴 未解决 (已知, 影响小, 待与 Stage0 重验一并处理)

---

## P3 — 低

### P3-1 文档同步
- `README.md` 描述的"两阶段 3D→6D CEM"已非主流程 (现主推 `--method v4-pipeline`)
- `docs/questions.md` 需同步近期改动

### P3-3 临时日志文件
根目录 `s2.log` / `s3.log` / `run_stage*.log` / `run_param_check.log` 为调试产物, 应归入 `test/` 或清理。

### P3-4 stage2_recon.npz 缺 R 字段
**证据**: stage2 保存 (L4694-4698) 只存 `frames/pos/gripper_qpos`, 无 `R`。
**影响**: grep 确认当前无任何读取方, 无功能影响, 仅记录不完整。
**修复**: 顺手补存 `R` (`np.array([recon['R'][f] for f in recon['frames']])`)。
**状态**: 🔴 未解决 (次要)

---

## 附: 复现命令

```bash
cd example/combination/trajectory_optimization
PY=/home/an/miniconda3/envs/dex/bin/python
$PY grasp_hawor.py --test-stage0 --views none --side right
$PY grasp_hawor.py --test-stage1 --views none --side right
$PY grasp_hawor.py --test-stage2 --views none --side right
$PY grasp_hawor.py --test-stage3 --views none --side right
```

关键日志标记:
- `[PhysicsValidation][stageN]` — 恒定距离 (偏差①)
- `[StageN 偏差②]` — 跟随误差 (偏差②)
- `[Stage3 到位确认][诊断]` — 三条件实测量级
