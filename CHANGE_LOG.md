## [2026-09-01] render_quick: 手腕深度反投影补齐 HaWoR→GLB 相似变换 + 轨迹上图

**类型**: Bug 修复
**影响范围**: example/combination/View/render_quick.py

### 问题
`wrist_dpj` (手腕深度反投影验证线) 只把 `pred_trans` 从 HaWoR 输入相机系转到 **HaWoR 世界系**，
就直接拿 **GLB 世界系** 的 RAS 外参去投影。两个世界系差 scale+旋转
(实测 7: 3.17×/65°, 121: 1.17×/76°, hoi4d: 0.55×)，投影被打飞:
- 121 右手 oob=431/600，仅 55 帧有效
- 7 左手 113 帧"全有效"但 Z=[-0.002,0.004]，全打在桌面
- hoi4d 双手深度一致性 mae 0.65~0.71

代码注释里的假设 "HaWoR 世界系 == 外参规范世界" 不成立。

### 修改内容
- `wrist_ext = R_c2w[fi] @ wrist_h + t_c2w[fi]` 之后补 `xform(p_hawor, s, R, t)`，
  与 `cam_traj_to_glb` / 主轨迹走同一个全局相似变换 `(s,R,t)`。
  **注**: (s,R,t) 是全局常量，与帧号无关；帧号只用来选 RAS 深度图。
- 新增颜色 `wrist_dpj_l` (darkviolet) / `wrist_dpj_r` (sienna)。
- **该轨迹此前算完从未绘制**（只在控制台打印），现已接入 matplotlib + plotly 两张图 + LEGEND 打印。

### 验证结果（真实手）

| 数据集 | 真实手 | 指标 | 修复前 | 修复后 |
|---|---|---|---|---|
| 121 | 右 | valid 帧 / oob | 55 / 431 | **179 / 306** |
| | | 深度一致性 mae | 0.852 | **0.388** |
| 7 | 左 | Z 范围 | [-0.002, 0.004]（贴桌面） | **[-0.016, 0.348]** |
| | | 深度一致性 mae | 0.381 | **0.174** |
| hoi4d | 左 / 右 | 深度一致性 mae | 0.710 / 0.650 | **0.400 / 0.442** |

hoi4d 的 valid 帧数下降 (537→196) 是因为错误投影点此前落在图内读到假深度，
修正后被正确判为 oob 剔除 —— 属于正确行为。

### 产出
- `example/combination/View/out/align_{121,7,hoi4d}_overlay.{png,html}`
- 三组 Frame 0 对齐距离均为 0.000000 m（transform_params.npz 锚点正确）

### 遗留
- 7 号左手 Midpoint Raw/Depth 仍穿透桌面 (min Z=-0.330/-0.170)，属 HaWoR 上游重建问题，非渲染错误。
- hoi4d RAS=160 帧 vs HaWoR=600 帧的**深度图选取**仍用均匀拉伸
  `ri = round(fi*159/599)`；这只影响取哪张深度图，不影响坐标变换正确性。

## [2026-08-29c] 新增 docs: 仿真测试参数、误差计算与调优方向

**类型**: 文档
**影响范围**: example/combination/docs/context/仿真测试参数与误差计算.md

### 内容
- GalaxeaManipSim 原始基准参数: R1 统一驱动增益 1000/200, control 15Hz, timestep 0.01,
  decimation 6, 触摸摩擦 (1.0,1.0,0.6), 手指 sign [1,1];
- 步进对比: 我们 240Hz/30Hz/DECIMATION=16 (控制周期 1/15s 与原仓库一致);
- 驱动增益四列对比 (原仓库/test8/9/11) + 手指增益与抓取力标定的耦合警告;
- 手指关节限位专节: URDF 0~0.05m/effort100/vel0.25, 行程语义 (q=0 合拢~-3mm 过盈,
  q=0.014 触碰 25mm 方块, q=0.012 squeeze, q=0.05 全开), 各测试驱动目标范围;
- 误差计算逐 test 公式与实测值 (test9 1.1mm/0.8°, test11 96.9mm/46.9°), 不可比性解释,
  test11 钳制口径注意点;
- 调整方向×预期效果表 (7 项, 3 项已完成) + RelaxedIK 弃用记录与 pybullet IK 三坑。

### 关联
- 回答用户三问: 原仓库参数 / 手指限位 / 误差计算完整口径。

## [2026-08-29b] test11 跟踪基座: 小幅平移 + 高平滑 + 舒适解保证

**类型**: 功能增量
**影响范围**: example/combination/05_gripper_test.py

### 修改内容
- **跟踪基座** (test11_mano_arm_follow): 姿态固定, 位置小幅平移跟随 MANO 轨迹质心:
  - 两级低通保证平滑度: 质心 EMA (α=0.04, 时间常数~0.8s) → 基座 EMA (α=0.10);
  - 限幅: 基座离初始位最大位移 8cm; 实测平均速度 2.39cm/s;
  - 每帧同步 SAPIEN (set_root_pose) 与 pybullet (resetBasePositionAndOrientation, 新增 set_base_pose);
  - 基座初始位 = 轨迹质心 + (-0.10,-0.10,+0.15), 可达钳制半径 0.45m.
- **舒适逆解保证** (_PybulletIK.solve): 候选集 = roll 扫描 × {当前构型(连续性), RIGHT_ARM_STARTING(舒适位姿)};
  评分 = 位置 + 0.10×朝向(rad) + 0.02×姿态偏离(Σ|q-STARTING|) — 等价逆解中自动挑最舒适的.

### 验证结果 (300帧, 与固定基座对比)
| 指标 | 固定基座 | 跟踪基座 |
|---|---|---|
| 位置误差 avg / max | 89.5 / 234.1 mm | 96.9 / **166.0 mm** |
| 朝向误差 avg / max | 51.9 / 152.7° | 46.9 / **60.6°** |
| 基座移动 | 0 | ≤8cm, 2.39cm/s |

最坏情况显著改善 (位置 max -29%, 朝向 max -60%), 平均持平; 剩余朝向 ~47° 为
R1 手腕限位 (q5±95°) vs MANO 自由手腕的物理鸿沟, 与基座无关.

## [2026-08-29] test11 重写: 固定底座单臂 + test9 MANO 管线 + pybullet IK (RelaxedIK 弃用)

**类型**: 重写 + 诊断结论
**影响范围**: example/combination/05_gripper_test.py

### 背景与诊断
- test11 v1 (test8轨迹+固定底座+RelaxedIK) 抓取失败, 根因链:
  1. R1 臂 URDF 手指 STL 全长碰撞使手指提前撑在桌面 → 已用 test8 同款简化 box 替换 (prepare_arm_urdf);
  2. RelaxedIK 冷启动大跳困在坏盆地 → 行走式预热 (多种子+pybullet... SAPIEN FK 择优);
  3. **致命: RelaxedIK 的解在 SAPIEN 中位置准但朝向差 100~167° (构型相关)**,
     朝向伺服外环也无法收敛; 且全仓库 (02/002/04/D4) 从未真正用固定基座
     6-DOF 追踪过 (D4 是 gripper_only 浮动夹爪, 02 视频为 INVALID 空帧),
     该问题无先例可循 → **弃用 RelaxedIK, 改用 pybullet IK** (与 SAPIEN 同一 URDF, 帧约定一致).

### 修改内容
- `_precompute_mano_targets()`: test9 预计算段抽为共享函数 (HaWoR 加载 + MANO 目标
  + anti-flip + 插值 + 四重平滑), test9/test11 共用; 顺带修复 test9 失效的
  `compute_mano_based_gripper_pose` 导入 (现从 trajectory_optimization/data_loader.py 加载).
- `_PybulletIK`: pybullet DIRECT 模式纯 IK 服务, 关键点:
  - 四元数直接用 scipy as_quat() 的 xyzw (多 roll 一次会差 ~90°);
  - null-space 四件套必须齐全 (lowerLimits+upperLimits+jointRanges+restPoses), 否则解无视限位;
  - DLS 对远目标会停在 rest-pose 吸引盆 → 多 rest-pose 重启 + pybullet 自身 FK 残差择优
    + roll 扫描 (MANO SVD 绕手指轴的 roll 本就任意);
  - 评分 = 位置 + 0.10×朝向(rad), 位置<10mm 且朝向<8.6° 提前退出.
- `test11_mano_arm_follow` (替换原 test11_arm_pick_place): 基座自动选址
  (轨迹质心后 0.10m/上 0.15m, 扫描最优), 命令 EMA 与 test9 同参, 可达钳制 0.45m,
  手指目标 = test9 MANO 关节值, step_physics_drive 纯物理, 5 球 MANO 参考标记 + 视频输出.
- `load_r1_arm(base_pos, yaw_toward)` 参数化.

### 验证结果 (300帧, --test 11)
- 端到端跑通, 产出 test11_mano_arm_follow.mp4 + test11_frame_log.log, 视频确认臂跟随 MANO 标记.
- 朝向误差 avg 41.9° (此前 RelaxedIK 为 126°, 帧内常见 5° 级跟踪段);
  位置误差 avg 84~90mm (基座扫描前 192mm).
- 已知残留: 固定基座工作空间 < MANO 自由轨迹 (物理约束, 非代码 bug) → 位置钳制后
  远端目标有 ~10-20cm 跟随 shortfall; 朝向受 R1 手腕限位 (q5 ±95°) 限制.
- 后续建议: 04 式浮动/跟踪基座 (基座平移跟随质心) 可同时解决位置与朝向可达性.

## [2026-08-29] 05_gripper_test 四重平滑统一: test9 内嵌实现提升为模块级 smooth_mano_trajectory

**类型**: 重构 (行为不变的代码提升 + 统一入口) + 文档同步
**影响范围**: example/combination/05_gripper_test.py, example/combination/docs/context/四重平滑与对话问题总结.md

### 背景
- 确认 `docs/context/四重平滑与对话问题总结.md` (2026-08-28) 所述"四重平滑未实现"已过时:
  test9 实际已实现 ①SG ②高斯Slerp ③min-jerk ④EMA 四重平滑, 但全部内嵌在
  `test9_mano_trajectory()` 函数内部 (~200 行), 无法被其他测试复用。

### 修改内容
- `05_gripper_test.py`
  - 将 test9 内嵌的 `_gaussian_quat_smooth` / `_apply_smooth_sg` / `_apply_smooth_dai` /
    `min_jerk_smooth` 四个函数原样提升到模块级 (置于 `_min_jerk` 缓动函数之后),
    `_apply_smooth_dai` 内部的 `from scipy.spatial.transform import Slerp` 改用已导入的 `sst.Slerp`。
  - 新增模块级统一入口 `smooth_mano_trajectory(raw_pos, raw_quat, raw_joint, raw_valid,
    max_vel=0.020, max_acc=0.005)`: 串接 ①SG+②高斯Slerp (`_apply_smooth_sg`) → ③min-jerk
    (位置三轴 + 关节 1/20 约束 + clip[0,0.05]); 第④重 EMA 保持命令级在线 (test9 主循环)。
  - test9 改为调用 `smooth_mano_trajectory()`, 删除内嵌定义与局部 savgol 导入
    (`from scipy.signal import savgol_filter` 移至模块顶部); 跟踪误差统计、
    max_jump_after / 平滑变化量打印等周边逻辑不变。
- `docs/context/四重平滑与对话问题总结.md`
  - 头部加 [2026-08-29 更新] 说明; 遗留待办 #1/#3 标记完成;
    新增「五、2026-08-29 更新」小节记录四重平滑落地现状与函数清单。

### 验证结果
- `python -m py_compile 05_gripper_test.py` 通过。
- dex 环境冒烟测试: 合成 120 帧序列 (注入 0.15m 尖峰 + 5 帧无效帧 + 带噪关节) 跑
  `smooth_mano_trajectory()` — 最大帧间跳变 265.2mm → 0.6mm, 四元数范数恒 1 (atol 1e-4),
  关节值保持在 [0, 0.05], 输出全为有限值。
- test9 调用链行为不变 (纯提升重构, 参数/顺序/clip 均与原内嵌实现一致)。

## [2026-08-28] 新增 tools/smooth_mano.py —— MANO 轨迹两种平滑器 (Kalman / Savitzky-Golay)

**类型**: 新增 (tools 工具模块 + 验证脚本)
**影响范围**: example/combination/tools/smooth_mano.py, example/combination/tools/test_smooth.py

### 需求
- 用户要求先用独立脚本验证两种平滑方案（卡尔曼滤波 vs Savitzky-Golay 滤波），作用于 MANO 参数，得到"新的 mano"后测试跳变消除效果。

### 实现内容
- `tools/smooth_mano.py`
  - `smooth_mano_kalman(raw, hand_idx, q=None, r=None, jump_gate_sigma=5.0)`:
    随机游走(RW)模型逐标量卡尔曼滤波，作用在 pred_trans / pred_rot / pred_hand_pose 参数轨迹上。
    采用**测量主导高增益配置 (q=delta_mad², r=0.01*delta_mad²)** 使滤波器无滞后跟随慢变旋转；
    单帧脉冲由 outlier gate (基于帧间差 MAD + 2% 信号幅度绝对下限) 识别并拒绝，保留预测。
  - `smooth_mano_sg(raw, hand_idx, window=9, poly=3, median_clip=True)`:
    Savitzky-Golay 零相位多项式滤波，先对离群帧做 MAD median-clip 再拟合，作用同上三个参数。
  - `smooth_mano(raw, method, ...)`: 调度入口。
  - 关键设计决策: 轴角参数**不做 rotate-vector unwrap**（原始 HaWoR 序列连续未跨 ±pi 缝），
    且卡尔曼用 RW 而非 CV 模型（CV 在旋转通道累积速度偏置导致 RMSE ~1 rad 发散，RW 稳定）。
- `tools/test_smooth.py`
  - 加载真实 hawor_results_*.npz，对每手 FK 出 21 关节轨迹，对比 raw / Kalman / SG 的
    midpoint (拇指尖 j20 与食指尖 j16 中点) 逐帧跳变与 RMSE。
  - 支持 `--inject-spike 0.15` 注入 15cm 合成平移脉冲验证 outlier 消除。

### 验证结果 (数据集 7, hawor_results_0_113.npz)
- 正常数据 RMSE (midpoint vs raw): Hand0 KF 4.58cm / SG 2.26cm; Hand1 KF 2.14cm / SG 0.39cm。
- 注入 15cm 脉冲 @frame50: F50->F51 跳变 原始 15.12cm → KF 0.58cm / SG 0.42cm (Hand0)，
  → KF 0.15cm / SG 0.21cm (Hand1)。两种平滑器均成功消除单帧跳变。
- 调参历程 (已收敛): CV 模型发散 → RW+低增益滞后 22cm → RW+测量主导高增益 (当前, 无滞后仅去 spike)。

## [2026-08-27] v6.0-v6.4 Stage2 修复"夹爪朝上"+ 手指合拢 + 简单优化

**类型**: 对齐 bugfix + 物理机制修复 + 简单优化

### v6.0 夹爪朝上根因 (用户多次指出"夹爪朝上")
- 根因: `data_loader.py` `compute_analytical_gripper_pose` 对 right 手 `y_sign=+1`,
  同一 MANO 变换链下 left 手朝下(-0.51)但 right 手翻转朝上(+0.94).
  render_quick/001_align 渲染的 left 手朝下, Stage2 用 right 手朝上.
- 修复: right 手 `y_sign` +1 → -1. 纯朝向修正 (root_pos 不变).
- 验证: 重跑 Stage1, 夹爪朝下 reward 61.46 (旧朝上 61.47), lift 18.34cm — 朝下完全能抓.

### v6.1 动态 F50 z 对齐
- Stage1 动态 enclose_z = obj_init_z - finger_offset_z; Stage2 之前用死固定值.
- 实测当前场景 delta=0 (两者一致), 非主因.

### v6.2→v6.4b 手指合不拢 (g_act 卡 0.074) 根因
- 现象: Stage2 物理回放手指被撑开到 0.074 (间距16cm>物体11.2cm), 物体只能被"架着"抬 3-4cm;
  Stage1 手指能合到 0.044 (过盈夹紧) 抬 18cm.
- v6.2 误改"边夹边提"(F50 起位置立即上提+闭合重叠) → 手指没来得及夹紧物体就被抬,
  物体被桌面顶住反推手指 → 合不拢.
- **v6.4b 修复**: F50 就位后先原地闭合 (位置保持 ~5帧/≈25仿真帧, 对齐 Stage1 F0-19 闭合),
  手指在物体表面施压夹紧 (g_act 0.071→0.043), 再携带抬升.
- 结果: g_act 合拢到 0.043 (对齐 Stage1 0.044), both_contact=120, peak_force=25N.

### v6.4 Stage2 简单优化 (用户: "stage2 应该需要优化")
- 新增 `optimize_stage2_dz`: 对 F50 就位 z 偏移 dz 做简单网格搜索 (9 值),
  reward = obj_lift + 双侧接触 - 漂移惩罚, 选出最优 dz 再录视频.
- 说明: Stage2 为确定性 Minimum Jerk 重建 + 物理回放, 无 CEM/MPPI;
  简单网格搜索即满足"轻量优化完成任务".

### v6.3 相机视角 (用户: "right_view 视角再远一点")
- right_view 焦距 x8 → x4 (视野扩大一倍, 物体占画面 ~17%).

## [2026-08-26] v5.3/v5.3b Stage2 跟随修复: 免下降直接就位 + 物体清零速度, 物体 1:1 跟随夹爪

**类型**: 物理 rollout bugfix (stage2_reconstruct_mixin.py)
**触发**: 用户指出 stage2 物体没有跟着夹爪走 ("夹爪的中心肯定得有物体才叫夹取")

### 根因诊断 (v5.2 失败分析)
1. **带速重置**: Stage1 验证回放结束时物体被夹在 0.228m 高空, 跟随 rollout 只
   `set_pose` 重置不清速度 → 物体带 ~2m/s 穿透桌面弹跳 (f00 obj_z=0.078),
   撞飞悬停中的夹爪 (vx 偏 2.8cm)
2. **下降楔死**: 全开间隙 11.9cm vs 物体弦宽 11.2cm 仅 0.35mm 单侧余量,
   PD xy 稳态误差 ~5mm 使手指架在物体肩部 → 下降卡在 z=0.044 (目标 0.030),
   闭合点比优化位姿高 1.5cm → 抓错截面 → 携带段滑落 (obj_lift 仅 0.47cm)

### 修改内容 (rollout_follow_trajectory)
1. 物体重置时清零线/角速度 (复用 stage3 的 components 遍历模式)
2. **免下降就位 (v5.3b)**: `_fd0` 与主循环 `fd` 钳位到 `max(dense, F50)`,
   set_qpos 直接就位 F50 抓取位姿 (复刻 Stage1 就位方式), 前 ~50 帧静止保持
3. 原地闭合时序对齐 Stage1: F50→F55 闭合 (~21 帧) → 稳定 (~12 帧) → 携带 (~17 帧);
   `_HOLD_FRAMES` 6→8, 携带起点 F56→F58; n_frames_out 80→100

### 效果对比
| 指标 | v5.2 | v5.3b |
|------|------|-------|
| final obj_lift | 0.47cm | **2.02cm** (峰值 3.85cm 含初始弹跳帧) |
| 双侧接触 | 80/80 | **100/100** |
| 携带段跟踪 | 滑落回桌面 | **物体与夹爪 1:1 同步上升** (+9mm vs +9.5mm) |

已知残留: 基座 z 稳态停在 0.038 vs 目标 0.030 (手指搭肩支撑), 闭合截面比优化值高
~9mm 但不影响夹持; 携带指令 +3.2cm 因 PD 滞后实际可见提升 ~1.5cm.

## [2026-08-26] v5.5 002 mano_only 新增 --hand-space: 双坐标系手部同屏对比 (对应原 Midpoint Depth / Midpoint RAS-GLB 轨迹)

**类型**: 功能增强 (仅 mano_only 分支, 主管线零改动)
**触发**: 用户澄清要的是原四条轨迹中 "Midpoint Depth" 与 "Midpoint RAS-GLB" 对应的手部 mesh, depth_with_mano 图仅为参考

### 修改内容 (002_render_scene.py)
1. `render_mano_only_video` 新增 `hand_space="both"` 参数:
   - `glb`: RAS-GLB 场景对齐版 (肤色) — 完整 h2g 链 `_render_to_sapien(verts)` (= 原 Midpoint RAS-GLB 轨迹)
   - `slam`: HaWoR SLAM 原始版 (蓝色) — 仅 Rx_hand 翻转 + R_AXIS, 不做场景对齐 (= 原 Midpoint Depth 轨迹)
   - `both`: 双版本同屏对比 (默认), 肤手在 GLB 场景中, 蓝手显示 SLAM 原始位姿
2. 材质拆分: mat_hand_glb 肤色 [0.96,0.75,0.69] / mat_hand_slam 蓝色 [0.30,0.55,0.90]
3. ras 数据源顶点已在 GLB 空间, 无 SLAM 版 → slam 分支自动跳过
4. argparse 新增 `--hand-space {glb,slam,both}` (默认 both)

### 验证 (数据集 7, 5 帧冒烟测试)
- `--data-source depth --hand-space both` → depth_aligned 加载正常, FPV 相机 113 帧, 渲染 18.2s 完成

## [2026-08-26] v5.4 002 mano_only --data-source 新增 ras: RAS 管线 MANO 重建 (depth_with_mano 渲染同源)

**类型**: 功能增强 (仅 mano_only 分支, 主管线零改动)
**触发**: 用户指出除 HaWoR 的 depth_aligned 外, 还有 depth_with_mano 深度渲染

### 数据源发现
- `/home/an/data/ras/7_vggt_omega/depth_with_mano/`: 113 张 PNG, 深度伪彩图 + MANO 手部 mesh (蓝) + 关节点 (黄/红), 由 RAS 管线生成 (本仓库无生成代码)
- 其手部数据源为 `hawor/7/reconstruction/mano_ras_reconstruction_0_113.npz`: 直接含 `verts_ras_world_left/right` (113,778,3) RAS/GLB 世界坐标顶点 + `valid_left/right`, 免 MANO FK

### 修改内容 (002_render_scene.py)
1. `_load_hawor_data_by_source` 新增 `ras` 分支: 加载 mano_ras_reconstruction npz, 按 hand_idx 取 left/right, 返回 verts_world + pred_valid + pred_trans(顶点均值, 仅供 centroid) + img_focal(fx); 缺失报错
2. `render_mano_only_video`:
   - betas 兜底 (ras 无 pred_betas → 零 betas, MANOLayer 仅用于面索引)
   - `_update_hand` 分支: ras 数据源顶点直用, `R_AXIS @ vw.T` 变换 RAS/GLB(Z-UP)→SAPIEN, 免 FK
   - centroid 分支: ras 用 `R_AXIS @ 手部中心`
3. argparse `--data-source` choices 增加 `ras`

### 现有四种数据源对照
| data-source | 文件 | 说明 |
|---|---|---|
| auto | depth_aligned 优先 | 默认 |
| depth | hawor_results_*_depth_aligned.npz | HaWoR 深度校正版 |
| raw | hawor_results_*.npz | HaWoR 原始版 |
| ras | mano_ras_reconstruction_*.npz | RAS 管线重建 (= depth_with_mano 渲染图的手) |

### 验证 (数据集 7, 5 帧冒烟测试)
- `--data-source ras` → 日志确认加载 `mano_ras_reconstruction_0_113.npz`, 有效帧 113/113, FPV 相机 113 帧, 渲染正常

## [2026-08-26] v5.3 002 mano_only 新增 --data-source: 手部 mesh 可选 depth_aligned / 原始版

**类型**: 功能增强 (仅 mano_only 分支, 主管线零改动)
**触发**: 用户指出轨迹本身有 depth / 非 depth 两种数据源区别, mano_only 手部 mesh 也应可选

### 修改内容 (002_render_scene.py)
1. 新增 `_load_hawor_data_by_source(hawor_dir, hand_idx, source, logger)`:
   - `auto`: 优先 `hawor_results_*_depth_aligned.npz` (与 load_hawor_data 行为一致)
   - `depth`: 强制深度校正版, 缺失报错
   - `raw`: 强制原始 `hawor_results_*.npz`, 缺失报错
   - 复用 `_fill_nan_frames`, 加载时打印实际文件名与版本标签
2. `render_mano_only_video(...)` 新增 `data_source="auto"` 参数
3. argparse 新增 `--data-source {auto,depth,raw}` (默认 auto)

### 验证 (数据集 7, 5 帧冒烟测试)
- `--data-source raw` → 日志确认加载 `hawor_results_0_113.npz (raw(npz))`, 渲染正常

## [2026-08-26] v5.2 对齐链修正: MANO 轨迹改用逐帧相机位姿提升 (与 render_quick.py 一致)

**类型**: 坐标对齐 bugfix + Stage1 重跑
**触发**: 用户确认 `001_align_scene.py` / `View/render_quick.py` 为正确版本, 要求 stage1 对齐方式与其学习

### 修改内容 (grasp_hawor.py + physics_env.py)
1. `physics_env._ALIGN_SCRIPT`: `01_align_scene.py` → `001_align_scene.py`
2. `_mano_to_sapien` / `_mano_to_sapien_v4`: 弃用固定 `Rx_hand=diag(1,-1,-1)` 近似,
   改为数据驱动链: `p_hw = R_c2w[f] @ p_cam + t_c2w[f]` (逐帧真实相机位姿) →
   `p_glb = s * R_hand_to_glb @ p_hw + t_hand_to_glb` (001 相似变换, R_X=I)
3. 所有调用点传入帧索引 (`fi` / `global_idx`)
4. `_mano_xform_R_hand` 语义变更: `R_h2g @ Rx_hand` → 纯 `R_h2g` (已确认无外部消费者)

### 效果验证
- 手腕轨迹移动 4~8cm (x=[-0.130,-0.057]→[-0.172,-0.087], y 同向), 对齐修正确实生效
- CMA-ES 8 迭代正常收敛: iter0 mean=18.78/best=60.98 → iter7 mean=60.91/best=61.15, Stage1d 精调至 **61.475**
- 最优位姿: pos=[-0.2827, -0.5192, 0.0295], euler=[125.91, -15.33, 9.98]
- 物理指标: obj_lift=18.83cm, both_contact=160/160, xy_drift=0.26cm, peak_force=7.21N
- 三视角视频已转码 H.264, 交付物 `output/D2_stage1_grasp.mp4`

## [2026-08-26] v5.1 CEM 重跑: 新模板下找到真正成功的抓取 (reward 61.85, lift 18.92cm)

**类型**: Stage1 重优化结果归档
**命令**: `python grasp_hawor.py --test-stage1 --side right --output output/gripper_right --views both` (dex 环境)

### CEM 收敛曲线 (8 迭代 × 16 种群, v5.1 直接闭合模板)
| iter | best | mean | std_pos |
|------|------|------|---------|
| 0 | 61.502 | 43.511 | ~3.7mm |
| 4 | 61.821 | 60.183 | ~0.5mm |
| 7 | **61.853** | 61.745 | ~0.3mm |

### 最优位姿 vs 旧位姿 (同一 v5.1 标准)
| 指标 | 旧 best_grasp (v4.37 时代) | 新 best_grasp (v5.1) |
|------|------|------|
| obj_lift_final | 0 cm (峰值 0.86cm 滑落) | **18.92cm** |
| both_contact | 单侧先接触推歪 | **160/160 帧** |
| peak_force | 2.7N | **7.24N** |
| xy_drift | 3.71cm (被推歪) | **0.77cm** |
| reward | -6.0 | **+61.85** |

### 产物
- `output/gripper_right/stage1/best_grasp.npz` — pos=[-0.277, -0.5255, 0.0271], euler=[125.68, -19.31, 8.83]
- 三视角视频已转码 H.264: cam/god/right_view_stage1_gripper_only_right.mp4 → deliverable D2_stage1_grasp.mp4

### 结论
v5.0/v5.1 的 "跟随提升" reward 有效区分了假成功与真抓取; CEM 在新标准下收敛到全程双侧接触、
干净竖直提升 18.92cm 的位姿. 用户反馈: reward 设计仍偏简单, 需参考 GraspGen/DexGraspNet 等论文改进.

## [2026-08-26] 002_render_scene.py: traj_only 改造为 mano_only (纯 MANO 手部 mesh, 无轨迹)

**类型**: 模式重做 (用户要求: "不需要绘制轨迹, 只渲染 mano 的手部, 像 02_render_scene.py 类似")
**影响范围**: 002_render_scene.py

### 修改
- `--mode traj_only` → `--mode mano_only`; 删除 `--trajs` 参数与全部轨迹计算/绘制代码
- 从 02_render_scene.py 移植 mesh 渲染: `compute_smooth_shading_normal()` +
  `update_hand_mesh_nodes()` (每帧重建顶点 mesh, 肤色材质 [0.96,0.75,0.69])
- 手部渲染从红色关节球升级为 MANO mesh (778 顶点); GLB 场景 + FPV 相机逻辑不变
- 输出文件 hand_traj.mp4 → hand_mano.mp4
- 原有 robot_tracking/topdown/gripper_only 分支零改动

### 数据集 7 验证
`example/combination/output/7_mano/hand_mano.mp4` (113 帧 @30fps, 55 秒)

## [2026-08-26] 002_render_scene.py: traj_only 新增 --trajs 参数自选轨迹

**类型**: 参数化 (用户反馈: "太多了, 我自己来选择")
**影响范围**: 002_render_scene.py (仅 traj_only 模式)

### 修改
- 新增 `--trajs` 参数, 逗号分隔可选 `raw,depth,glb,center`, 默认 `depth,center`
- 未选中的轨迹直接跳过计算 (省 FK/深度反投影时间), 不只是不画
- TRAJ_SPECS 加 key 字段过滤; 其余模式零改动

### 用法示例
```bash
--trajs depth          # 只看 Midpoint Depth (蓝)
--trajs depth,center   # Depth + RAS depth center 对照 (默认)
--trajs raw,depth      # 原始 vs 矫正对比
```

## [2026-08-26] 002_render_scene.py: 新增 --mode traj_only (手部+四条轨迹, 无机器人)

**类型**: 新增渲染模式 (用户要求: "先别重定向, 这几个轨迹都帮我渲染一下")
**影响范围**: 002_render_scene.py

### 修改
- parse_args choices 增加 `traj_only`; main 在 [3/4] 验证后分支调用, 跳过机器人加载
- 新增 `render_traj_only_video()`: SAPIEN 场景 = GLB + MANO 手部关节球(红, 每帧) + 4 条静态轨迹
  - Midpoint Raw 红 / Midpoint Depth 蓝 / Midpoint RAS-GLB 品红 / RAS depth center 青
  - Raw 中点显式加载非 depth_aligned 原始 npz (注意原始 npz 带 (2,N,C) 手部维度需切片)
  - RAS depth center 复用 View/render_quick 的 load_ras_depth_data + unproject_mask_center_to_glb
    (导入路径必须用 Path(__file__).parent/"View", PROJECT_ROOT 指向 robot_world_ws 不是仓库根)
  - FPV 相机默认跟 RAS; 支持 behind/front/topdown 与 --viewer 实时预览
- 踩坑: 此 SAPIEN 版本 get_picture("Color") 前必须 camera.take_picture(), 返回 float[0,1] 需转 uint8

### 数据集 7 输出
`example/combination/output/7_traj/hand_traj.mp4` (113 帧 @30fps, dex conda 环境运行)

## [2026-08-26] render_quick.py: 新增 Midpoint RAS-GLB 轨迹 (mano_ras_reconstruction npz)

**类型**: 新增可视化数据源
**影响范围**: render_quick.py

### 修改
- 新增 `load_ras_mano_glb()`: 加载 `reconstruction/mano_ras_reconstruction_*.npz` 的
  `joints_ras_world_{left,right}` (RAS 深度反投影重建, 原生 GLB 世界系, 零变换直接用),
  取 (j20+j16)/2 中点, 颜色 magenta(L)/teal(R), 标签 "Midpoint RAS-GLB"
- matplotlib + plotly 双端接入; 自动发现文件, 不存在则跳过

### 数据集 7 实测对比 (113 帧, 左手)
- Raw vs RAS-GLB: mean 0.686 m (原始深度严重偏离)
- Depth vs RAS-GLB: mean 0.336 m, 按轴分解 X -0.04/Y +0.20/Z -0.17, 各轴 std ~0.14 m
  → 无单一主导轴, 非标定问题; 是 RAS MANO 逐帧拟合的抖动 vs HaWoR 平滑轨迹的本质差异
- Depth 轨迹仍以 RAS 深度中位数为锚, 更平滑; RAS-GLB 提供独立参照

## [2026-08-26] render_quick.py v2: 改为只渲染 Midpoint, Wrist/Thumb/Index 注释掉

**类型**: 可视化调整 (用户指定: "只保留 midpoint, 其他留着")
**影响范围**: render_quick.py (matplotlib + plotly 双端)

### 修改
- 恢复 Midpoint Raw/Depth 轨迹渲染; Wrist(j0)/Thumb(j20)/Index(j16) 关节标记注释保留在代码中
- 偏移后 Midpoint 轨迹仍保持注释 (--offset-frame 调试用, 需要时取消注释)
- docstring 同步更新; 两数据集已重渲染至 output/align_{7,121}_overlay.png/.html

### 附: Depth 轨迹的计算流程 (01c_depth_align.py, 默认 mode=camframe)
1. 加载 HaWoR reconstruction + model_masks + RAS depth/extrinsics + 001 的 transform_params(scale_ratio)
2. 帧配对: N_ras 帧 RAS 深度图 ↔ 等比例采样的 HaWoR 帧
3. 每对帧: trans_z = pred_trans 的 z (HaWoR 输入相机系); ras_z = RAS 深度图手部 mask 区域中位数 z (不膨胀, mm→m)
4. cf = (ras_z / scale_ratio) / |trans_z| — 只缩放 z, 保留 HaWoR 回归的 u,v 方向
5. 校正因子线性插值到全部 HaWoR 帧
6. 应用: pred_trans[...,2] *= cf, 其余 MANO 参数不变 → 存 hawor_results_*_depth_aligned.npz
依据: HaWoR R_c2w 与 RAS extrinsics 旋转逐帧一致 (残差 0.0°), 故 RAS 深度图 z 可直接视为 HaWoR 输入相机系 z.

## [2026-08-26] render_quick.py v1: 手部渲染只保留 Wrist, Midpoint/Thumb/Index 注释掉

**类型**: 可视化简化 (用户指定: "只留下 wrist, 其他的先注释掉, midpoint 也不需要")
**影响范围**: render_quick.py (matplotlib + plotly 双端) — 已被上一条覆盖, 仅存档

## [2026-08-26] v5.1: 删除下降段 — 位姿直接就位 → 原地闭合 → 竖直提起 20cm

**类型**: 动作模板简化 (用户指定: "给定位姿是直接闭合, 而不需要下降")
**影响范围**: stage1_grasp_mixin.py (rollout_grasp_only)

### 修改
- 删除 hover 下降段 (原 F0-F10 从上方 3cm 下降 + F10-F30 边走边夹)
- 新帧结构: 就位 (set_qpos 到位姿, 手指张开包住) + warmup → F0-F19 原地闭合 → F20-F29 稳定 → F30-末 竖直提起 20cm
- CEM 的 z 候选改为在 enclose_z ±1cm 内搜索 (直接决定初始夹持高度, 防插地)
- 删除 _DESCEND_FRAMES/_hover_z/_PRELOAD_QPOS/_close2 相关代码

### 回答 "Stage1 还要设计动作吗"
不需要. 动作模板完整且固定, Stage1 唯一任务是优化位姿本身; 模板越简单, 评估越纯粹.

### 验证 (eval_stage1_v5.py, 旧 best_grasp.npz)
| 指标 | v5.0 (带下降) | v5.1 (直接闭合) |
|------|------|------|
| lift_peak | 1.34cm | 0.86cm |
| peak_force | 10.7N | **2.7N** |
| xy_drift | 3.4cm | 3.71cm (物体被推歪) |
| reward | -5.0 | -6.0 |

结论: 直接闭合暴露得更彻底 — 旧位姿手指没对准物体中心, 闭合时单侧先接触把物体推走
(力只有 2.7N, 物体被推歪 3.71cm). 正是用户要求的 "两个 gripper 中间必须有物体" 的硬检验.
下一步: 重跑 CEM 在新模板下重新搜索位姿.

## [2026-08-26] v5.0: Stage1 统一动作模板 (竖直提起 20cm) + reward 重写, 旧最优位姿 16.63→-5.0 分

**类型**: Reward 设计重构 / 用户指定范式统一
**影响范围**: stage1_grasp_mixin.py (rollout_grasp_only / stage1_reward / _eval_pose), grasp_hawor.py (3 处调用点), eval_stage1_v5.py (新增验证脚本)

### 背景
用户指出: Stage1 只读 MANO F50 一个位姿, 抓取动作全是自设计且各版本不一致; 要求统一固定为
"给定位姿 → 夹爪闭合 → 竖直提起 20cm", reward 只看"物体最终有没有和夹爪一起提升", 且此项目标价很大.

### 动作模板统一 (rollout_grasp_only)
- 默认 n_frames 80→160 (4s @ 40Hz): 40 帧接近闭合 + 120 帧竖直抬升
- 抬升总高度固定 `_LIFT_TOTAL_M=0.20`, 速度自适应 `0.20/_LIFT_FRAMES` (≈1.67mm/帧 ≈ 66mm/s)

### reward v5.0 (stage1_reward)
- 主导项: `40*min(1, lift_final/0.20)` — 每 1cm +2 分, 提满 20cm 才拿满 (旧版 3.2cm 即拿满 28 分)
- ≥15cm 强奖励 +10; 双侧接触≥10帧且提起>5cm +6; 压住不夹 -6
- 夹爪中心 gate: `4*exp(-dist/0.02)`; 推飞检测: 提起但 drift>3cm -15; 中途滑落: peak>8cm 且 final<半峰值 -8
- 删除接近过程距离主导项与搬运期 xy_drift 矛盾 (竖直提升范式下 drift 语义=推飞检测)

### 验证 (eval_stage1_v5.py, 加载旧 best_grasp.npz 单次 rollout)
| 指标 | 旧模板(80帧/3.2cm) | 新模板(160帧/20cm) |
|------|------|------|
| lift_peak | 1.24cm | 1.34cm (中途掉落) |
| lift_final | 1.24cm | **0.00cm** |
| both_contact | 80 帧 | 97 帧 |
| reward | 16.63 | **-5.0** |

结论: 旧 16.63 分是"假成功"——双侧接触正常但夹持力 (10.7N) 不足以克服重力,
物体被短暂带起 1.34cm 即滑落. 新标准下该位姿不合格, CEM 必须搜索真正能带起物体 20cm 的位姿.

### 待办
重跑 CEM (`--test-stage1`, rollout 已自动 160 帧): 8 迭代×16 种群×160 帧, 预计耗时约为原来 2 倍.

## [2026-08-26] reward 诊断: 用 test8 已知成功轨迹反测 stage1_reward v4.37, 暴露无梯度缺陷

**类型**: 分析 / Reward 设计诊断
**影响范围**: example/combination/gripper_test_output/analyze_test8_reward.py (新增)

### 背景
用户指出当前优化"没有梯度、2cm 移动只是碰到物体、缺硬约束 (双侧夹爪+中间必须有物体)"。
要求用当前 reward 反测 05_gripper_test.py 的 test8 (已验证成功: 双侧 43N 夹紧、提起 20cm、搬运 40cm)。

### 方法
解析 test8_frame_log.log (800 帧), 把每个时刻当作 "rollout 终点", 代入 stage1_reward v4.37 公式逐项评分。
接触判定: 双侧 j <= 14.05mm+0.8mm 容差。

### 关键发现 (reward 缺陷实证)
1. **帧240 双侧 43N 夹紧完成**: reward=-1.5 — 接触项反而 -4 (`both_contact>0 且 lift<=5mm → -4`), 抓得越早越罚
2. **帧260 提起 12cm**: reward=41.4 全程最高 — reward 只对终点快照敏感
3. **帧280 提到 20cm 高空**: 漂移项 -12 重罚搬运 (任务本身要求位移!)
4. **帧799 完整抓-搬-放执行完**: reward=-16.7 接近最低分
5. 结论: 当前 reward 是"终点快照"型, 无过程梯度; 对成功抓取存在负激励; xy_drift 与搬运任务冲突

### 结论提取脚本
`gripper_test_output/analyze_test8_reward.py` 可复用于后续 reward 改版回归测试

## [2026-08-26] v4.19: Stage2 动态抓取跑通 — 纯物理化 + 垂直下降配方, lift 0→1.15cm

**类型**: 关键修复 / 对齐 Stage1 配方
**影响范围**: example/combination/trajectory_optimization/{stage2_reconstruct_mixin.py, grasp_hawor.py}

### 根因 (三轮迭代定位)
1. **best_grasp.npz 被覆盖**: +90° tilt 测试 (--output output/gripper_right) 把 Stage1 成功结果 (reward=16.97) 覆盖成坏结果 (reward=-2.61), 导致首次 test-stage2 验证 both_contact=0
2. **接近段斜线切入撞物体**: 重建轨迹 F38-F49 的 xy 从 MANO F38 位置 (偏物体 4.8cm) lerp 到抓取点, 手指全开下降时侧向压住物体 → 反力顶住虚拟关节 PD (K=50000) → act_z 卡死 0.090 全程不跟随 ([S2诊断] 实证)
3. **接近段姿态混入 MANO R**: MANO 原始姿态"夹爪朝上"已被诊断不可用
4. **携带段无有效抬升**: follow 的 _interp 覆盖 reconstruct 输出, 仅 +3cm smoothstep 且起步过缓

### 修改内容
- stage2_reconstruct_mixin.py `rollout_follow_trajectory`:
  - 删除碰撞禁用 hack (手指-地面 + 邻物) 及恢复块; 删除主循环每帧 set_qpos teleport — 完全对齐 Stage1/test8 纯 driver 范式
  - 新增轨迹起点一次性 set_qpos + warmup 5 帧 (对齐 Stage1)
  - 携带段抬升 0.03→0.032m 线性 (对齐 Stage1 提起配方)
  - 新增 [S2诊断] 每 8 帧日志: drive target vs 实际 qpos / 手指闭合 / obj_z / 接触
- stage2_reconstruct_mixin.py `reconstruct_trajectory`:
  - 接近段改垂直下降: xy 全程锁定抓取点正上方, hover offset 0.058→0.021 (同 Stage1)
  - 姿态全程 grasp_R (删除 MANO R 渐变/slerp)
- grasp_hawor.py: test-stage2 入口接线 `--f50-tilt-deg` (与 stage1 一致)

### 验证结果
- `--test-stage2 --output output/gripper_right`: **both_contact=80/80, obj_lift=1.15cm** (Stage1 验证水平 1.13cm), peak_force=17.68N, xy_drift=2.99cm ✅
- 虚拟关节恢复跟踪 (f72: tgt_z=0.059 / act_z=0.059); 物体跟随手上升 (obj_z 0.0393→0.0508)
- 视频: output/D3_stage2_follow.mp4 (三视角)

## [2026-08-26] render_quick.py GLB 加载修复 + 手部关节渲染

**类型**: 关键修复 / 功能增强
**影响范围**: example/combination/View/render_quick.py

### 根因 (GLB Z 轴虚高)
- 实证 (数据集 7): final_scene.glb 每个节点的 T 都是导出时加的 **Z-up→Y-up 纯旋转 (平移=0)**; 原始 geom.vertices 才是 Z-up 外参/GLB 世界坐标 (物体平铺在 z≈0)
- 此前 load_glb_objects 应用 T@v 把物体转立起来, z 方向虚高 ~0.6m ("z 轴上升那么多")
- 原版 Ego-Video-to-SIM 的 load_glb_to_sapien 同样直接用 geom.vertices

### 修改内容
- load_glb_objects 改用原始顶点 (删除节点变换), 文件头/注释同步修正
- 新增手部关节轨迹渲染: 拇指尖 j20 (^/v)、食指尖 j16 (s/D)、腕 j0 (x/+), Raw 实心 / Depth 半透明, matplotlib 与 plotly 双端
- 修复空心标记 facecolors='none' 在 mpl 3D scatter 的 RGBA 广播崩溃

### 验证结果
- 数据集 7/121 均出图成功, Frame 0 对齐 dist=0.000000m ✅
- 数据集 7: 物体 z≈0 平铺, 相机 z≈0.49-0.51, 右手中点 z∈[0.58,0.72] ✅
- 颜色验证: 顶点色 uint8 正确归一化, 各物体颜色可区分 (含粉色目标物) ✅
- 注意: 左手 (幻觉手) 中点穿透桌面 z=-0.33, 属上游 HaWoR 数据问题, 已有告警提示

## [2026-08-26] render_quick.py 回归 Ego-Video-to-SIM 正统对齐流程

**类型**: 重构 / 关键修复
**影响范围**: example/combination/View/render_quick.py

### 根因
- render_quick 自行发明了两层坐标变换 (GLB auto-calibration T_cal + 在线 HaWoR→GLB 标定), 与原版管线世界观冲突
- 实证 (数据集 7): HaWoR 世界 ≠ RAS 外参世界 — 外参经 stage1_strict 重对齐 (coordinate_alignment.json), 第 0 帧相机在外参世界位于 (-0.33,-0.90,0.51), 在 HaWoR 世界位于原点; 逐帧旋转差恒定 ~65°。此前注释"旋转差 0.0°"是 121 场景结论, 不适用于数据集 7
- 正统流程 (对照 /home/an/robot_world_ws/src/Ego-Video-to-SIM): 规范世界 = GLB 原生世界; RAS 外参原生就在 GLB 世界; 唯一矩阵是 001_align_scene.py 预计算的 transform_params.npz (第一帧相机锚定 + Umeyama 尺度), 渲染脚本只加载应用

### 修改内容
- 删除 calibrate_glb_to_world / _depth_proj_score / load_ras_c2w / calibrate_hw_to_glb 四个自造函数 (~200 行)
- 主流程改为直接加载 transform_params.npz (优先 hawor_dir/transform_params.npz, 回退 output/alignment/), 缺失时报错提示先跑 001_align_scene.py
- RAS 相机轨迹 / mask 中心反投影 / 手腕深度反投影: 删除多余的 ext_to_glb 变换 (原生在 GLB 世界)
- HaWoR 相机朝向: R_inv @ R_c2w → R_h2g @ R_c2w (与原版 hawor_cam_to_sapien_pose 一致)
- 文件头 docstring 重写为正统世界观; matplotlib/plotly 标题去掉 cal_rep 引用

### 验证结果
- 数据集 7 完整运行通过, 输出 align_overlay.png + .html ✅
- **Frame 0 alignment: dist=0.000000m** (HaWoR 与 RAS 相机中心完全重合; 此前相差 1.49m) ✅
- transform_params: scale=3.1678, 加载自 /home/an/data/hawor/7/transform_params.npz

## [2026-08-25] v4.52: Stage3 base 指令通道失效修复 — lift 0→2.7cm 突破

**类型**: 关键 Bug 修复
**影响范围**: example/combination/trajectory_optimization/stage3_optimize_mixin.py, physics_env.py

### 根因 (v4.42-v4.51 十余版 lift=0 的真正原因)
- Stage3 rollout 的虚拟关节驱动只调了 `set_drive_target`, **缺失 Stage1/Stage2 都有的 `set_qpos` 硬设**
- 实测 ([vdrive] 诊断): cmd_z 全程变化 (0.025→-0.004→+0.026), qpos_z 卡死 0.092-0.098 **完全无响应**, K=50000 近 5000N 驱动力亦然 → drive_target 通道在该场景下失效
- 后果: base 从未执行过任何轨迹 — "close 段下沉"实为自由坠落被地面撑住, "抬升"从未发生; 物体自然不动
- 上层策略参数 (夹持深度/冻结时机/抬升速度) 怎么调都无效, 因为轨迹根本没被执行

### 修改内容
- `stage3_optimize_mixin.py` rollout_v4_stage3: 虚拟关节改为 set_qpos 硬设 + set_drive_target 双保险 (对齐 Stage1/Stage2 成功实现)
- `physics_env.py`: VIRTUAL_STIFFNESS 5000→50000 / DAMPING 1000→2500 (v4.51, 消除 M·g/K≈71mm 重力下垂; 单独无效但保留)
- `stage3_optimize_mixin.py`: v4.49 高点冻结 (F63) + v4.50 提速抬升 (1.2mm/帧); DIAG 增加 [vdrive] cmd vs qpos 对比

### 验证结果
- **lift=2.7cm** (此前十余版全部 0.0cm), 接近 Stage1 基准 3.47cm ✅
- base 跟踪误差 dz: 70-97mm → **0.7mm**, contact=both 全程保持 ✅
- 遗留: drift=6.6cm 偏大 (夹持段仅 ~1.1cm, 主要来自释放段扰动), 可作为后续优化项

### 方法论教训
- "策略怎么调都没反应"时优先怀疑基础设施层 (指令通道/执行链路), 用 cmd vs actual 直接对比数据定位, 而非继续调参



**类型**: 文档 / 重渲染
**影响范围**: example/combination/docs/05_physics_drive_pipeline.md (新增), gripper_test_output/test8_complex_pick_place.mp4 (重生成)

### 修改内容
- 重新渲染 test8 复杂抓取放置测试 (`python 05_gripper_test.py --test 8 --num-frames 800`): 覆盖 8月2日 旧视频, 新视频 127KB/800帧
- 新增文档 `docs/05_physics_drive_pipeline.md`: 完整五层调用链 (L0 set_drive_property 注册K/D → L1 控制层轨迹规划+set_drive_target×8 → L2 step_physics_drive 内 compute_passive_force/set_qf/scene.step ×16 → L3 pysapien .so 绑定层 → L4 PhysX 引擎内部 F_drive/F_total 合成/碰撞检测/M(q)q̈=F_total 求解/半隐式欧拉积分)
- 文档含时间尺度表 (1/240s 子步 × DECIMATION=16 → 2x 慢动作)、test8 各关节 PD 参数表、实测佐证与 GalaxeaManipSim 一致性说明

### 验证结果
- test8 完整周期通过: 方块最高升至 22.1cm, 最终放置 (43.7, 20.3) 高度 1.2cm 稳定落桌 ✅
- 视频已输出 gripper_test_output/test8_complex_pick_place.mp4

## [2026-08-25] test5 加入眼在手上 D405 相机 (URDF + 挂载渲染)

**类型**: 功能
**影响范围**: example/combination/05_gripper_test.py (GRIPPER_ONLY_URDF_TEMPLATE / generate_gripper_urdf / test5_grasp_release)

### 修改内容
- 调研确认: 真实 R1 机器人 URDF (assets/robots/hands/r1_gripper/r1_gripper_glb.urdf) 中夹爪上挂有 Intel RealSense D405 深度相机 (`right_D405_link`, fixed joint 挂 `right_gripper_link`, 偏移 xyz=(-0.0192, 0.009, 0.0748) rpy=(-2.268, 0, -1.5708)), 头部另有左右两个全局相机; 但 galaxea_sim 的 meshes 目录无 D405 文件, 故精简模板当初未包含
- `GRIPPER_ONLY_URDF_TEMPLATE` 加入 D405 link + fixed joint: 位姿/inertial 与真实 URDF 一致 (mass=0.1kg), visual mesh 走 `./d405_meshes/` 相对路径; 不加 collision, 相机外壳不参与接触, 抓取物理不受影响
- `generate_gripper_urdf` 新增 `d405_meshes` 符号链接 → 本仓库 `assets/robots/hands/r1_gripper/meshes` (D405 mesh 所在处)
- test5 新增眼在手上相机: `scene.add_mounted_camera()` 挂载到 `right_D405_link.entity` (get_links() 返回的是 Component, 需取 .entity); 挂载朝向补偿 D405 joint 旋转 (`R_local = R_joint^T @ R_desired`), 前向 = 手指伸出方向下倾 35°, 使方块始终在画面中央
- 渲染不复用 `render_frame` (其内部有 scene.step() 会改变物理时序), 只做 update_render + take_picture; 输出 `gripper_test_output/test5_wrist_view.mp4`

### 验证结果
- 抓取物理不受影响: 方块仍真实升至 14.2cm 稳定保持, 释放落回桌面 ✅
- 眼在手上画面正常: 可见双指夹持方块、桌面与场景, 视频已输出 test5_wrist_view.mp4

## [2026-08-24] test5 改为纯物理抓取, 修复双重驱动失稳

**类型**: 修复
**影响范围**: example/combination/05_gripper_test.py (test5_grasp_release)

### 修改内容
- 删除 `cube.set_pose()` 瞬移作弊: 原代码在抬升/释放阶段直接传送方块跟随夹爪, 抓取是假的; 现在方块完全由物理引擎驱动, 靠手指摩擦力真实夹持
- 修复双重驱动冲突: 原实现手动 PD (`step_physics joint_pd`) 与 `load_gripper` 设置的引擎内置 drive (K=5000) 叠加, 提高手动增益后手指失稳抽打方块 (方块被撞飞出桌面); 改为与 test8 一致的纯引擎驱动模式 (`step_physics_drive`), 无手动 PD、无 fixed_qpos 钉死
- 驱动增益对齐 test8 已验证值: 平移 K=5000/D=1000, 夹爪 K=3000/D=300

### 验证结果
- 完整周期通过: 下降(不碰倒方块) → 闭合 → 方块真实跟随夹爪升至 14.2cm 并稳定保持 → 张开释放落回桌面 (2.0cm)
- 抓取判定 ✅ 成功, 视频已输出 gripper_test_output/test5_grasp_release.mp4

## [2026-08-22] 修复 v4 流水线最终回放物体被弹飞 / CLOSE 夹不住物体

**类型**: 修复
**影响范围**: example/combination/trajectory_optimization/grasp_hawor.py

### 修改内容
- 最终回放 Warmup 后物体被物理弹飞 (基座与目标物体重叠), 已修复为 warmup 后重置物体位姿
- CLOSE 阶段改为跟随 Stage 1 优化抓取位姿 (opt_pos/opt_R), 不再强制垂直朝下 _rot_close
- 详见 example/combination/trajectory_optimization/CHANGE_LOG.md

### 验证结果
- 回放从 total_contacts=0 (物体在 2.5m 外) 改善为手指到达物体并夹持 (obj 抬至 6~8cm)
- 遗留: 抬升时物体滑落, lift<3cm (继续调优)

## [2026-07-18] 为 05_gripper_test.py 关键函数添加中文 docstring

**类型**: 修改
**影响范围**: dex-retargeting/example/combination/05_gripper_test.py

### 修改内容
- [05_gripper_test.py] 为 19 个关键函数添加/改进中文 docstring:
  - 工具函数: generate_gripper_urdf, create_scene, load_gripper, step_physics, rotmat_to_zyx_euler, fk_fingertips, create_camera, set_camera_pose, render_frame
  - 测试函数: test1_gripper_open_close, test2_gripper_movement, test3_mano_tracking, test4_combined_motion, test5_grasp_release, test6_mano_grasp
  - 辅助函数: compute_analytical_pose, create_cube_object, create_table_surface
  - 主函数: main
- 所有改动仅涉及 docstring 和注释, 未修改任何代码逻辑

### 验证结果
- 通过逐函数 Edit 操作, 确认每次替换唯一匹配
- 未改动代码逻辑, 仅增强可读性
