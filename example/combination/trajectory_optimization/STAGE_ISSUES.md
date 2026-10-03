# STAGE 问题收集库

> **用途**: 管线**当前仍存在**的问题清单。只列开放问题, 不列已修复。
> **维护**: 问题解决后**从此文件移除**, 并记入 `CHANGE_LOG.md`。本文件不留 ✅ 条目。
> **最近更新**: 2026-09-12

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

| 序 | ID | 问题 (一句话) | 级别 | 改动量 |
|---|----|------|------|------|
| 1 | **N-5** | Stage2 接近段无局部优化, 靠手调 d_stop/q_open 规避碰撞 | P0 | 大 |
| 2 | **N-2** | CoACD 在被缩放后的网格上分解 → 121 退化凸包 → 物体 warmup 自转 | P0 | 中 |
| 3 | **N-4** | Stage1 位姿选择无物理依据 (6DOF CMA-ES 全局搜), Stage1/2 太慢 | P1 | 中 |
| 4 | **N-1** | K=1000/D=200 对 0.72kg 虚拟基座过阻尼 (ζ=3.72), 跟不住 | P1 | 小 |
| 5 | **N-3** | F90 释放段冗余添加帧过多 (96 → 可裁到 ~60) | P1 | 小 |
| 6 | **N-6** | scale 0.6 阻塞: 手指碰撞盒撞 4.72cm 物体, 开大夹爪反而更糟 | P1 | 见 N-5 |
| 7 | P1-9 | stage0 双重 run() 第二次重建 precompute, 覆盖平移轨迹 | P1 | 中 |
| 8 | P1-5 | Stage3 残留两个碰撞 hack, 验证环境≠真实环境 | P1 | 中 |
| 9 | P1-6 | `--side right` 可能跟到 HaWoR 幻觉手 | P1 | 小 |
| 10 | P1-3 | npz 无序 glob 取首个, 加载不确定 (不可复现) | P1 | 小 |
| 11 | P1-4 | F50 硬编码 50, 轨迹长度变化时语义漂移 | P1 | 小 |
| 12 | P1-1 | 旧分析脚本 vis_stage0_error_analysis 仍报假指标 19.36cm | P1 | 小 |
| 13 | P1-11 | Stage2 接近段轻擦目标物体 (已降级为软指标, 残留) | P1 | 见 N-5 |
| 14 | P2-7 | MANO 变换链疑整体 Z 翻转 (指尖比手腕高 13.45cm) | P2 | 中 |
| 15 | P2-8 | Stage0 映射过零硬阈值产生 180° 姿态台阶 | P2 | 小 |
| 16 | P2-3 | 恒定距离 ref 绝对值异常 (max_dev 判定仍可信) | P2 | 小 |
| 17 | P2-4 | 坐标变换代码冗余 (3 份 `compute_analytical_gripper_pose`) | P2 | 小 |
| 18 | P2-5 | 几何可行性硬钳制 (enclose_z/ground_clearance/feasible) 缺失 | P2 | 中 |
| 19 | P2-6 | stage3 旧 npz 兜底硬编码 `_obj_z=0.0173` | P2 | 小 |
| 20 | P2-2 | View 脚本 best_grasp/stage2_recon 路径三处不一致 | P2 | 小 |
| 21 | P3-1 | README/CHANGE_LOG/docs 未同步近期改动 | P3 | 小 |
| 22 | P3-3 | 根目录 s2/s3/run_stage*.log 调试产物未归类 | P3 | 小 |
| 23 | P3-4 | stage2_recon.npz 缺 R 字段 (当前无读取方) | P3 | 1 行 |

### 本会话 (2026-09-11/12) 已解决并移除的条目

| 原 ID | 问题 | 解决方式 |
|---|---|---|
| P0-1 | Stage3 夹持滑脱 (恒定距离 max_dev=8.6mm FAIL) | **`_GRIP_PRELOAD` 夹持预紧** — 手指命令 0.0086→0.0006, 夹持力 16.2N→24.0N (+48%); 携带段 `dist2obj 37.3/37.3mm` 稳定, Stage3A PASS |
| P0-2 | MANO 原始逐帧无滤波接入 PD | 已实现 `mano_smoothing.py` (SG) + `mint_cleaning/final_clean` (MINT 参数域 UKF+RTS), `_SMOOTH_MINT` 开关; MINT 下 MANO_FOLLOW max 误差 3.97→1.90mm |
| P1-7 | `--grasp-pose-method` 漏传 stage1/2/3 | 各入口已 `grasp_pose_method=args.grasp_pose_method` |
| P1-10 | `--f50-tilt-deg` 漏设 stage3 | stage3 已补 `sim.f50_tilt_deg = float(args.f50_tilt_deg)` |
| P2-1 | v4.49 抬升上限 3.1cm (F63-F89 硬编码 1.2mm/帧) | Stage3 已改用 MANO 携带轨迹 + `release_candidate`, 不再受该硬编码限制 |
| — | `RESET_DRIFT` 全流程 0/3 (物体 warmup 自转 29°) | = N-2 CoACD 问题, 已临时 `_COACD_TARGET=none` 规避, 根治见 N-2 |
| — | Approach 段基座速度门 10mm/s 逼停 | `PRE_READY_LINEAR_SPEED_MPS` 10→30mm/s (实测仅超 7%, 位置/姿态/漂移均远优于容差) |
| — | cam 视角"像 god 视角" | 已定位: **相机从未被替换**(基准 1.23% vs 旧版 1.30% 内容占比/画面中心几乎重合); Stage2/3 rollout 补 `_sync_cam_view_to_frame` 逐帧同步; 焦距源修正 `_CAM_FOCAL_FROM_RAS=1` (600→794, 画面放大 1.32×) |

**最小修复集**: N-3 (删 F90 冗余帧) + N-1 (K/D 阻尼修正) — 都是小改动, 当天可验证。

---

## N — 本会话新增 (2026-09-12)

### N-1 K=1000 对虚拟基座过阻尼, 跟不住

**现象**: `_VIRTUAL_PD_K=1000, _VIRTUAL_PD_D=200` (GalaxeaManipSim / 04 原始值) 下, MANO 自由段跟踪误差 **116.51mm / 45.85°**, 是 K=50000/D=600 的 **77 倍**。

**根因 (已算清, 非刚度不足)**:
```
虚拟 6 关节链接 0.01kg×6 = 0.06kg
+ gripper_link 0.604kg + 两指 0.027×2 = 0.72kg    ← 运动总质量 m
阻尼比 ζ = D / (2√(K·m))
```
| K | D | ζ | τ=D/K | 30Hz 滞后 |
|---|---|---|---|---|
| 1000 | **200** | **3.72** | 0.2 s = 6 控制周期 | ~100mm ← 与实测 116mm 吻合 |
| 1000 | **50** | **0.83** | 0.05 s | ~25mm |
| 50000 | 600 | 1.58 | 12 ms | ~6mm |

GalaxeaManipSim 的 (K=1000, D=200, dt=0.01) 是给**大惯量真实机械臂**标定的; 我们 0.72kg 虚拟基座套同一组 D 就显得"黏"。

**待办**: K/D 矩阵 A/B (见下), 判据 MANO_FOLLOW 自由段 pos max < 2mm。
| 组 | K | D | ζ |
|---|---|---|---|
| A0 | 1000 | 200 | 3.72 (复现) |
| A1 | 1000 | 100 | 1.86 |
| **A2** | **1000** | **50** | **0.83 (目标)** |
| A3 | 1000 | 25 | 0.42 |
| A4 | 50000 | 600 | 1.58 (基线) |

**注意**: 改 K/D 会进 physics digest → 必须连带重跑 Stage1。
**附带**: 夹爪开合已就是上限 `GRIPPER_MAX_OPEN=0.05` (`GRIPPER_INIT_OPEN=0.04`), "降低到 0.05" 已是当前状态。
**证据**: `physics_utils.py:101-113`; `physics_env.py` URDF 质量/effort 定义

**状态**: 🔴 未解决 (根因已定论, 待 A/B 验证)

---

### N-2 CoACD 在被缩放后的网格上分解 → 121 个退化凸包

**现象** (2026-09-12, scale 0.45):
```
[load_glb] 测试缩放 toy_0 ×0.45 (center 不变)
[CoACD] Compute Time: 202.3s   # Convex Hulls: 121
[load_glb] toy_0: CoACD 凸分解 = 121 个凸包 (碰撞体)
[SAPIEN] [critical] QuickHullConvexHullLib::findSimplex: Simplex input points appers to be almost at the same place
→ warmup object drift=8.63mm / 29.09deg → RESET_DRIFT → Stage2 0/3 → 全流程不运行
```
关闭 CoACD (`_COACD_TARGET=none`) 后 `target=1` (box), 立即 **3/3 PASS**。

**根因 (代码顺序)**: `data_loader.py` 中 **`_OBJ_SCALE` 缩放发生在 CoACD 之前** —— 先缩放 `vertices_sapien` → 导出 `temp_ply` → 再喂给 `_coacd_decompose`。网格绝对尺寸变小后, `preprocess_resolution=30` / `max_ch_vertex=256` / merge 步骤切出大量碎屑。
**另一注意**: `self.coacd_target = os.environ.get("_COACD_TARGET", "glb_4")` 默认**开启**; 且 `_OBJ_SHAPE` 判断在 `elif` 分支里, **CoACD 一生效 `_OBJ_SHAPE` 就被整个跳过**。
**空串陷阱**: `_COACD_TARGET=` (空) 会让 `"" in real_name` 恒真 → **所有物体**都被分解, 更慢更糟。必须写 `none`。

**修复方案 (按优先级)**
1. **先分解后缩放**: 用**原始尺度**网格做 CoACD, 再把每个凸包顶点乘 `_OBJ_SCALE` (绕 bbox 中心)。分解质量与缩放无关, 且更快。
2. **限数与过滤**: `max_convex_hull` -1 → **16~32**; 丢弃体积 < bbox 体积 0.1% 的碎屑; 超限回退 box。
3. **缓存**: 键 = `(网格 hash, threshold, max_hull, resolution)`。**202s/次不可接受**。
4. **稳定性准入门**: 加载后静置 warmup, 要求漂移 < 0.5mm / < 0.5° 才放行; 否则回退 box 并记录。
5. **验证收益**: 对比 box / CoACD 的 anchor 间隙、双侧接触时长、夹持力。

**2026-09-13 进展 — 问题分三层, 前两层已解决, 第三层待查**

子模块 `coacd_collision.py` 已实现并可用: `decompose`(先分解后缩放/g退化过滤/限数/缓存/收缩) +
`export_hulls` + `check_stability`; `data_loader.py` 29 行钩子, 默认关闭 (`_COACD_TARGET=glb_4` 开启)。

| 层 | 现象 | 处置 | 状态 |
|---|---|---|---|
| 1 | **121 个退化凸包** (202s + QuickHull 退化警告) | 先分解后缩放 + `max_hull=32` + 退化过滤 + 缓存 | ✅ 已修 (实测 32 凸包 / target=31 / 缓存命中) |
| 2 | **物体静置自转 26.45°** (8.12mm/27.39°) | 定位为**外逼近凸包底面不平 → 物体自找支撑面**, 确定性、**期间无任何夹爪接触** (`全程无夹爪相关接触`)。**不是穿地** (按凸包落地修正未触发)。`RESET_DRIFT` 参考位姿改到静置之后 (用户定策 A), 静置量只记指标 | ✅ 已修 (RESET_DRIFT 不再阻断) |
| 3 | **物体被弹射 1.4m** (`object_drift_mm: 1406/1150`) | 无夹爪接触 (`target_contact=False`) 却位移 1.4m → 疑多凸包 compound 惩罚力弹射 | 🔴 **待查** |

**第 3 层的候选修法**:
- ① **静置提前到场景加载时** (`load_glb_with_physics` 之后、Stage1 之前), 使**所有阶段共享静置后的物体位姿** —— 现仅 Stage2 reset 静置, Stage1 仍用未静置位姿
- ② **`max_hull` 32 → 8~16** (凸包越少越"瘦", 越不易重叠)
- ③ **`shrink` 0.994 → 0.97~0.98** (直接消除外逼近体积超出, 2~3% ≈ 1~1.5mm)
- ④ 用户观察: **物体离其他物体蛮远, 应不会自碰撞** → 需专项验证"物体自身是否稳定"(单物体静置 90 步的漂移), 而非归因于邻物

**状态**: 🔴 未解决 (第 1/2 层已修, 第 3 层待查; 用户要求必须用 CoACD 提取被抓取物体, 这是真实仿真实物)

---

### N-3 F90 释放段冗余添加帧过多

**现象**: Stage3 后缀 96 控制点, 其中释放段为
`PRE_RELEASE_SETTLE(n)` → `RELEASE(smoothstep 渐变 m)` → `RELEASE_SETTLE(k)`。
用户定策: **F50 的添加帧是预设的 (20 close + 10 settle, 保留)**, 但 **F90 放置不需要添加帧, 直接打开夹爪**。

**改法**:
```
现状: CARRY(n) → PRE_RELEASE_SETTLE → RELEASE(smoothstep) → RELEASE_SETTLE → 退避
目标: CARRY(n) → RELEASE(单步, q 直接置 GRIPPER_MAX_OPEN) → 退避
```
- 删除 `pre_release_settle_controls` 整段
- `release_controls: N → 1` (去掉 `_smoothstep5` 渐变)
- `release_settle_controls: N → 0` (或留 2 步让 trace 记录"已张开")

**收益**: 后缀 96 → ~60 控制点 (**-37%**), 视频与整体耗时同步下降。
**回归检查**: `OPEN_TRACK` (`min(final_q) ≥ q_open − 3mm`) 与 `NOT_RELEASED` 必须仍通过 —— 单步张开更突然, 需确认 `final_q` 在退避前收敛 (必要时保留 2~3 步 hold)。
**注意**: 之前 `_GRIP_PRELOAD` 必须在 `RELEASE`/`RELEASE_SETTLE`/`POST_RELEASE_BLEND`/`MANO_FOLLOW` **都不施加**, 否则手指张不到位 (曾导致 `OPEN_TRACK` 失败, final_q=[0.0544, 0.0396] < 0.047)。
**证据**: `stage3_optimize_mixin.py` `build_stage3_suffix` 的 `pre_release/release/release_settle` 段

**状态**: 🔴 未解决

---

### N-4 Stage1 位姿选择无物理依据 + Stage1/2 太慢

**现象**: Stage1 用 6DOF CMA-ES 全局搜索 (数百次 ~80 控制点 rollout), **慢**, 且选出的位姿不理想 (当前 anchor = F50 的 MANO 位姿加偏移, 并未比较过其他帧)。
**用户方案**: **在 F25–F50 里逐帧挑位姿, 看哪个"稍微平移一下"能夹住物体最好, 必须以物理仿真为准。**

**方案**
1. **候选生成**: `f ∈ [F25, F50]` (26 帧), 取该帧 MANO 夹爪位姿为种子。
2. **零成本几何预筛** (解析 AABB, 不做物理): 只留"手指包围盒与物体无穿透、虎口内侧间隙 > 0"的候选。
3. **微平移搜索 (粗到细)**: 粗 `Δ∈{−10,0,+10}mm` 三轴 (27/帧) → 细 `Δ∈{−3,0,+3}mm` (27)。
4. **物理验证**: 每个候选跑 **30 控制点短 rollout** (20 close + 10 settle, 复用现有 recipe), 打分 = 双侧接触控制点数 (主) / anchor-物体距离偏差 / 物体被推动位移 / 夹持力。
5. **预算**: 26×27=702 次解析预筛 (近免费) → 预计剩 50~80 → 各 30 控制点。对比 CMA-ES 数百次×80 控制点, **预计加速 3~5 倍**, 且位姿有物理依据。

**Stage2 加速**: 当前 = 3 次独立 reset 的完整 136 控制点 rollout + Stage3 一次 Stage2 重放 = **4 次全量**。方案: 第 1 次通过后, 后两次只做 digest/轨迹一致性校验而不重跑物理; 并降低 `--stage2-approach-time-scale` (现 2.0)。
**证据**: `stage1_grasp_mixin.py`; `traj_optimize.py`

**状态**: 🔴 未解决

---

### N-5 Stage2 接近段无局部轨迹优化 (靠手调参数规避碰撞)

**现象**: 接近段 (d_stop 切换点 → F50) 目前是**固定轴向直线**, 碰撞与否完全靠手调 `--stage2-approach-start-m` / `_Q_OPEN` / 物体 scale。后果:
- scale 0.45: 轻擦物体 (`max_pen 0.14~1.98mm`), 靠把门禁降级为软指标通过;
- scale 0.6: **真撞** (`max_pen 3.08mm`, 手指被顶开 13.57mm → `PRE_F50_READY` 失败);
- scale 0.6 + `_Q_OPEN=0.075` (开大夹爪): **更糟** —— `max_pen 11.51mm`, 手指被卡死 (`cmd 0.075 / act 0` → 误差 75mm)。

**用户方案 (接近段局部优化)**: 处理的原始参数 = **预处理后的夹爪轨迹参数**, 代价函数三项 + 终点固定:
1. **`C_collide` (不碰到)**: 夹爪碰撞体 vs 目标物体的穿透深度 (逐控制点 max, 段内求和), 用解析手指盒-物体检查做快速近似 + 物理接触查询确认。**大权重障碍项**
2. **`C_follow` (跟随预处理轨迹)**: 与预处理后夹爪轨迹的逐控制点位姿偏差 (平移+旋转, 加权)
3. **`C_feasible` (物理可行)**: 虚拟基座关节速度/加速度限、PD 跟踪误差、不得接触地面/非目标物
- **末状态硬约束 = Stage1 的最终固定位姿**
- **热启动** = 当前已验证的轴向直线逼近

**实现要点**: 决策变量低维化 (3~5 个控制点参数化的横向/垂向偏移 + 可选手指 q 剖面 → 维度 ~9~15), CMA-ES/MPPI 20~40 次评估即可。现成 `--stage3-method cem` 桩可复用 (当前标记"已禁用并回退 smooth")。

**⇒ 这一项顺带解决 N-6 / P1-11 / scale 0.6 阻塞。**

**状态**: 🔴 未解决 (核心能力)

---

### N-6 scale 0.6 阻塞 (手指碰撞盒撞 4.72cm 物体)

**现象** (2026-09-12, scale 0.6, q_open 0.05, d_stop 0.16, axial):
```
[Stage2 gate] PRE_F50_READY: q tracking error=13.57mm
[Stage2 contact] finger_link2 x glb_4  n=22  max_pen=3.0810mm
[Stage2 contact] finger_link1 x glb_4  n=31  max_pen=1.3064mm
controls=79 (走到 close_start) → 0/3 FAIL
```
**对照**: scale 0.45 同配置只擦 `0.34mm`。
**已排除的方向**: **开大夹爪是错的** —— `_Q_OPEN=0.075` 时 `max_pen` 恶化到 `11.51mm`, 手指被完全卡住 (误差 75mm)。
**注意**: `output/hawor7a_left/` 是 **scale 0.6** 的历史成功版本, 但**只做到 stage2**(stage3 标注"待做"); 0.6 从未跑到 stage3。基准实验记录: `物体缩放 glb_4 ×0.6 (4.72cm/0.17kg)`。

**解决方向**: 只能走 N-5 (接近段轨迹优化, `C_collide` 会自动把路径绕开物体), 或改 `--stage2-approach-mode mano_blend`。

**状态**: 🔴 未解决 (阻塞 0.6 复现基准 cam 取景)

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

### P1-9 stage0 双重 run() 覆盖平移轨迹
**现象**: stage0 跑两次 run(): 第一次 run → pre_translate (烘进平移) → neutral_offsets (offset≈0) → 第二次 run (`_opt_params=None`)。
**根因**: 第二次 run 里 `_has_opt_traj=None`, precompute 无条件重建 `mano_gripper_traj={}` → **覆盖**第一次的平移轨迹, 回到原始未平移位置。
**后果**: 日志打印平移后轨迹 (对齐物体), 但第二次 run 实际运动用未平移轨迹 → 日志≠实际; stage0 最终夹爪实际不对齐物体 (差 ~0.59m)。
**2026-09-02 实测确认 (Stage1 同样受影响)**: 连续两次 `--test-stage1` 稳定复现 "CEM 能找到好候选, 但最终选出失败位姿":
```
候选分布: obj_lift=23.44 / 23.41 / 23.15 / 20.53 ... (1226 个候选)
最终选出: reward=0.000, lift_final=[0.0, 0.0, 0.0]cm → 验证门 0/3 通过
```
即 CEM 评估时 lift=23cm, 验证/终选时 lift=0 — **评估与终选用的不是同一条轨迹**。
**修复方向**: A 去掉第二次 run; B 二次 run 前不重建 precompute (加"已填充则跳过"门控); C stage0 不做 pre_translate/neutral_offsets, 纯跑一次基线。需实测后定。
**状态**: 🔴 未解决 (现象已确认, 待修)

### P1-11 Stage2 接近段轻擦目标物体 (已降级为软指标)
**现象** (2026-09-11, scale 0.45 / q_open 0.05 / d_stop 0.16 / axial):
```
[Stage2 gate][软指标·不阻断] APPROACH_TARGET_CONTACT: target contact prefix/adjust=0/22~40, objects=['glb_4']
[Stage2 contact] finger_link2 x glb_4  n=22~71  max_pen=0.34~1.98mm
[Stage2 contact] finger_link1 x glb_4  n=31~59  max_pen=0.14~1.31mm
```
接近段仍有数十个控制点轻擦目标物体。**但物体未被推走**: anchor 处基座-物体 `37.8mm` ≈ 设计值 `36.9mm`, 携带段稳定 `38.3/38.2mm`, 且预紧后 `cmd=0.0006/act=0.0246` 确实夹住。
**2026-09-12 门禁定策**: `APPROACH_TARGET_CONTACT` **降级为软指标** (记录+告警, 不阻断)。理由: 手指碰撞盒长 7.74cm, 要绕过 3~5cm 物体, 轻擦几乎不可避免, 且不影响抓取成立。
**根因 (已量化)**: 轴向逼近余量薄 —— 物体在 anchor 系 `y ∈ [-32.6, +47.9]mm`, 虎口内侧 `±55.0mm` (q_open=0.05) → **+y 侧仅 7.1mm**。
**根治**: 走 N-5 (接近段轨迹优化)。
**状态**: 🔴 未解决 (门禁已放宽, 物理现象仍在; 根治见 N-5)

---

## P2 — 中

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

**另 (v6.9 新增)**: `compute_analytical_gripper_pose` 共 **3 份副本**, 仅 `data_loader.py` 那份 (主管线) 同步了 v6.9 的 `align_gripper_approach_to_gravity`:
- `data_loader.py:443` ← 主管线用 ✅
- `View/vis_trajectory.py:80` ← 可视化, ❌ 未同步
- `trajectory_loader.py:171` ← ❌ 未同步 (且无调用方, 疑似死代码)
**影响**: 可视化脚本画出的夹爪姿态与仿真不一致 (仍朝上), 误导调试。
**状态**: 🔴 未解决

### P2-5 几何可行性硬钳制缺失
`enclose_z` 硬钳制 (obj_top+margin / 地面+指长)、`_probe_ground_clearance()`、`best_grasp.npz` 写 `feasible` 元数据 + 加载端校验 — 均缺失。
**证据**: `docs/2026-08-25-full-pipeline-diagnosis.md` P0-1/P0-3; 全仓无 `feasible` 校验
**状态**: 🔴 未解决

### P2-6 stage3 旧 npz 兜底硬编码 `_obj_z=0.0173`
**证据**: stage3 旧 `best_grasp.npz` 兼容兜底:
```python
_obj_z = 0.0173  # 从场景获取 (粉色物体初始 z)   ← 硬编码, 注释说"从场景获取"实际没取
_enclosed_z_loaded = _obj_z + 0.030
```
**影响**: 换物体/场景, `enclose_z` 估计错误。仅旧 npz (无 enclose_z 字段) 兜底路径触发。
**修复**: `_obj_z` 从 `sim.obj_info`/`obj_bbox_centers` 实际读取。
**状态**: 🔴 未解决

### P2-7 MANO 变换链疑整体 Z 翻转
**现象**: 即便 v6.9 已把夹爪 R 翻转为朝下, **MANO 手本身在该变换链下仍是朝上的**:
```
MANO 指尖相对手腕高度: mean=+13.45cm  (指尖比手腕高 13cm)
```
人手伸向桌面物体时指尖应**低于**手腕 (负值)。量级 13cm ≈ 手掌+手指长度, 说明手几乎竖直朝天。
**疑点**: 上游 `R_h2g` / `R_c2w` / `R_AXIS` 变换链可能存在整体 Z 翻转。
**现状**: v6.9 在 R 构造处做 `align_gripper_approach_to_gravity` 属**下游修正**, 已验证有效, 但未触及上游根因。
**上游翻转实验 (v6.9b) — 已实现但默认关闭**:
实现了 `physics_utils.flip_mano_hand_down()` (绕水平轴转 180°, 以腕为支点; 重力点积判据; 幂等), 挂在 `_mano_to_sapien` 出口, 由 `flip_mano_hand` 开关控制 (**默认 False**)。
- 单独测试指标全部正确 (掌法 z `+0.846→-0.846`, 指尖-腕 `+14.31→-14.31cm`)
- 但**接入管线后 Stage1 严重退步** (恒定距离 `ref=2.7m/31.6cm`, `max_dev=9-51mm` FAIL)
- 根因: 翻转以腕为支点把指尖上下移动 ~28cm, 改变 MANO 轨迹**绝对位置**; 而 CEM 搜索空间/`enclose_z`/物体对齐按"未翻转轨迹"标定
- 结论: 上游翻转若要启用, 需**同步重新标定** CEM 搜索空间与 enclose_z
**证据**: `test/diag_mano_R_orientation.py` 【F】段; `test/run_s1_upflip.log`
**状态**: 🔴 待核查 (上游翻转能力已备好, 待重新标后再启用)

### P2-8 Stage0 映射过零硬阈值产生 180° 姿态台阶
**现象** (2026-09-11): 预处理后的夹爪姿态序列 `grip_R` 在相邻帧间出现**真实的 180° 台阶**:
```
帧间姿态变化 mean=7.63 max=179.54 deg; >90° 帧对 [7, 9, 13, 108]
F9->F10: 179.1°   F13->F14: 178.8°   F108->F109: 179.3°
该段 X.z = -0.085 … -0.003 (approach 轴几乎水平)
```
**根因**: `physics_utils.align_gripper_approach_to_gravity` 用 `dot(X, 重力) < 0` 这一**过零硬阈值**, 被 `data_loader.compute_mano_based_gripper_pose` / `compute_analytical_gripper_pose` **逐帧独立**调用。approach 轴接近水平时符号在相邻帧间来回跳 → 180° 台阶。
**影响 (实测: 小)**: 台阶全落在 F0–F25 前缀段 (离物体 30cm+), 只造成 1 个控制周期瞬态; `MANO_FOLLOW` 段跟踪 `mean=1.10°/max=21.44°`, 去掉台阶后完全一致 → **非 Stage2 瓶颈**。
**注意**: 本函数位于 **Stage0 的 MANO→夹爪映射链路**, 它的输出就是 Stage0 复现的夹爪姿态本身。2026-09-11 一度被改成死区判据, 因**改动 Stage0 输出**已按要求回退 (回退后逐位复现基准)。
**若要修**: 应在**选符号那一步**取"与重力朝向更一致"的分支, 但**必须重新验证 Stage0**。
**状态**: 🔴 未解决 (已知, 影响小)

---

## P3 — 低

### P3-1 文档同步
- `README.md` 描述的"两阶段 3D→6D CEM"已非主流程 (现主推 `--method v4-pipeline`)
- `docs/questions.md` 需同步近期改动
- 新增成功记录: `docs/成功管线记录_s045_f4_pre008.md`

### P3-3 临时日志文件
根目录 `s2.log` / `s3.log` / `run_stage*.log` / `run_param_check.log` 为调试产物, 应归入 `test/` 或清理。

### P3-4 stage2_recon.npz 缺 R 字段
**证据**: stage2 保存只存 `frames/pos/gripper_qpos`, 无 `R`。
**影响**: grep 确认当前无任何读取方, 无功能影响, 仅记录不完整。
**修复**: 顺手补存 `R`。
**状态**: 🔴 未解决 (次要)

---

## 附: 复现命令

```bash
cd example/combination/trajectory_optimization
PY=/home/an/miniconda3/envs/dex/bin/python

# 当前已验证成功的完整配置 (scale 0.45)
export _OBJ_SCALE=0.45 _OBJ_SCALE_TARGET=glb_4 _OBJ_SHAPE=box _OBJ_SHAPE_TARGET=
export _GRIP_FRICTION=4.0 _GRIP_PRELOAD=0.008 _Q_OPEN=0.05 _NO_GRIPPER_GROUND=1
export _VIRTUAL_PD_K=50000 _VIRTUAL_PD_D=600 _SMOOTH_MINT=1
export _COACD_TARGET=none          # ★ 必须, 见 N-2
export _CAM_FOCAL_FROM_RAS=1       # cam 焦距对齐 RAS 内参 (600→794)

$PY grasp_hawor.py --test-stage3 --side left --views all \
    --hawor-dir /home/an/data/hawor/7a \
    --ras-dir /home/an/data/ras/7_vggt_omega \
    --start-frame 0 --num-frames 113 --grasp-pose-method svd_palm \
    --stage2-approach-mode axial --stage2-approach-start-m 0.16 \
    --output-dir output/s045_mint_camfocal
```

关键日志标记:
- `[Stage2 gate] PASS/FAIL` — Stage2 硬门 (需 3/3)
- `[Stage3] A=PASS B=PASS` — 携带 / 释放双门
- `[Stage2 track] <phase> n= pos= rot= dist2obj= finger_q=` — 分段跟踪与夹持
- `[Stage2 gate][软指标·不阻断]` — 已降级、不阻断的观察量
