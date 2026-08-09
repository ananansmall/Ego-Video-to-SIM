# Change Log

本文件记录 `dex-retargeting/example/combination` 目录下代码的变更历史。

---

## [2026-08-06] test9 MANO 轨迹跟随: 帧数匹配数据 + 无效帧插值补齐

**类型**: 修改
**影响范围**: `05_gripper_test.py` (test9_mano_trajectory)

### 问题背景
- 用户使用 `/home/an/data/hawor/121_C5_CellPhone_161deg` 数据 (600 帧)，但代码用 `--num-frames 600` 硬编码，导致数据帧数不匹配
- 数据存在无效帧间隙 (帧 268-325 无效)，旧代码对无效帧"保持上一帧目标"，导致物理仿真的残余 PD 力产生朝向跳动 (帧 128 出现 40.7° 朝向误差)
- 帧 551 的 `hand_pose` 全零 (`hp_norm=0.000`) 但 `pred_valid=True`，属于 HaWoR 跟踪器损坏但未标记的帧

### 修改内容
- [05_gripper_test.py] 加载数据后设置 `num_frames = n_frames_data`，使用数据实际帧数，移除 `frame_skip`
- [05_gripper_test.py] 预计算循环: 无效帧存储 `np.nan` 而非保持上一帧值
- [05_gripper_test.py] 新增插值补齐块 (do-as-i-do 风格): 位置线性插值 + 朝向 SLERP 插值 + 边界 clamp
- [05_gripper_test.py] 新增 MANO 参数突变 TOP-5 调试输出 (Δhand_pose / Δ位置 / Δ朝向 / SVD 平面法向量)

### 验证结果
- 数据 `121_C5_CellPhone_161deg`: 162 帧无效帧插值补齐，600 帧全部有效
- 位置误差 avg=2.8mm (改善 26%)，帧 128 的 40.7° 朝向误差消除
- 剩余问题: 数据段边界 (帧 266) 朝向误差 58.3°，帧 551 的损坏数据待 MAD 速度检测处理

---

## [2026-08-06] render_quick.py: 将渲染轨迹从拇指尖/手腕切换为拇指+食指中点

**类型**: 修改
**影响范围**: `dex-retargeting/example/combination/View/render_quick.py`

### 修改内容
- [render_quick.py] 数据打印: 从 thumbs_raw/thumbs_da/wrists_raw/wrists_da 改为 mids_raw/mids_da
- [render_quick.py] 偏移计算: `_offset_wrist_trajs` 重命名为 `_offset_mid_trajs`，对 mids_raw/mids_da 施加偏移
- [render_quick.py] matplotlib渲染: 移除拇指尖和手腕轨迹，改为渲染拇指+食指中点 Raw/Depth 轨迹
- [render_quick.py] Plotly渲染: 同上，移除旧轨迹，渲染中点轨迹
- [render_quick.py] 图例: 更新为 mid_raw/mid_depth 系列颜色键
- [render_quick.py] 自动范围计算: 从 [thumbs_raw, thumbs_da, wrists_raw, wrists_da] 改为 [mids_raw, mids_da]
- [render_quick.py] 修复 `ras_depth_ctr_off` 颜色键（matplotlib/Plotly offset mask center 均使用此颜色）

### 验证结果
- 语法检查: 无 `_offset_wrist_trajs` / `l_thumb_raw` / `r_wrist_raw` 等旧引用残留
- `thumbs_raw`/`indexs_raw`/`wrists_raw` 等数据加载函数保持不变，仍用于计算中点和 MANO 帧统计

---

## [2026-07-20] 新增极简纯夹爪物理抓取验证脚本

**类型**: 新增
**影响范围**: `06_simple_grasp_test.py`

### 修改内容
- [06_simple_grasp_test.py] 新增独立脚本，用 GalaxeaManipSim 的 R1 右手夹爪 mesh + 6 个虚拟关节（3 平移 + 3 旋转）实现纯夹爪物理抓取
- [06_simple_grasp_test.py] 物理驱动完全照搬 GalaxeaManipSim 的 `step()` 方法：`set_drive_target()` → `compute_passive_force()` → `scene.step()`
- [06_simple_grasp_test.py] 设置碰撞组过滤：手指之间不碰撞，手指与方块碰撞，避免手指自碰撞导致无法闭合
- [06_simple_grasp_test.py] 方块放在手指正前方（base 前方 3.689cm），确保手指能真正对准并夹住方块
- [06_simple_grasp_test.py] 输出视频 `output/simple_grasp.mp4` 和参数日志 `output/simple_grasp_param_log.json`

### 验证结果
- **抓取成功**: 方块从 1.5cm 升至 19.5cm（抬升 18.0cm）
- 手指正常闭合：40mm → 0mm
- 方块跟随夹爪同步上升，未掉落

---

## [2026-07-19] MANO 轨迹抓取: 修正四阶段策略, 成功抓取方块

**类型**: 修复
**影响范围**: `05_gripper_test.py` (test6)

### 修改内容
- [05_gripper_test.py] 策略从"展开手指→下降→闭合→跟随"改为"悬停闭合→下降→跟随→释放"
  - Phase 1 (0-10%): 夹爪停在方块上方 3cm, 手指在空中间闭合 (避免方块被 PD 力弹飞)
  - Phase 2 (10-20%): 手指保持闭合, 夹爪下降至方块高度
  - Phase 3 (20-70%): 跟随 MANO 轨迹, 抬起方块
  - Phase 4 (70-100%): 手指张开释放
- [05_gripper_test.py] 修正 Z 轴参考系: 使用 `cube_z_val` (方块中心高度) 替代 `z_base_corrected` 作为 Phase 3 的 Z 基准
- [05_gripper_test.py] 删除旧的 `close_end`/`HOVER_END`/`CLOSE_END` 变量, 统一为 `CLOSE_END`/`DESCEND_END`

### 验证结果
- **抓取成功**: 方块从 1.2cm 最高升至 6.1cm (抬升 ~5cm)
- 手指闭合正常: 50mm→0.2mm (Phase 1), 不产生弹飞力
- 方块跟随夹爪沿 MANO 轨迹移动 (XY 从 8.7→11.5, Z 从 1.2→3.7cm)
- Phase 4 释放后方块正确回到桌面 (1.2cm)

---

## [2026-07-19] 移除预闭合逻辑: 改为"一边跑一边调整夹爪"

**类型**: 重构
**影响范围**: `05_gripper_test.py` (test6)

### 修改内容
- [05_gripper_test.py] 移除预闭合逻辑 (set_qpos 手指到0.0 + 200步稳定)
- [05_gripper_test.py] 手指 PD 刚度 (stiffness=8000, damping=800) 在方块创建后、主循环前设置
- [05_gripper_test.py] 主循环使用合成开合 (基于最低Z帧的时序)，夹爪跟随 MANO 轨迹的同时调整手指开合

### 验证结果
- 抓取成功: 方块最高升至 2.7cm (初始 1.2cm)
- 手指在 0-7% 进度内从 50mm 闭合到 9mm，实现"一边跑一边调整夹爪"
- 释放正常: 55-70% 进度内手指打开，方块回到桌面

## [2026-07-18] 30DOF抓取窗口CEM优化 + CLOSE阶段策略优化: 首次成功抓取(1/7)

**类型**: 修改/新增
**影响范围**: `trajectory_optimization/grasp_hawor.py`, `trajectory_optimization/traj_optimize.py`

### 修改内容
- [traj_optimize.py] 新增 `cem_grasp_window_optimize`: 30DOF CEM优化 (F48-F52 × 6DOF)
  - 替代原来3D/6D单帧优化, 给优化器更多自由度调整抓取位姿
  - 5帧独立偏移, 每帧6DOF (dx,dy,dz,droll,dpitch,dyaw)
- [traj_optimize.py] `compute_reward_xyz` 新增 lift 奖励 (w_lift=800), 驱动CEM优化出真正能夹住的位姿
- [grasp_hawor.py] `rollout_single`: 支持30DOF CEM参数扩展到654D向量
  - F48-F52 设为固定帧, 各自独立偏移
  - `_fixed_offsets_654` 包含 {F0:0, F48:off, F49:off, F50:off, F51:off, F52:off, F95:lift, F112:lift}
- [grasp_hawor.py] `run_optimize`: Stage 1 用30DOF CEM替代3D/6D, 删除旧Phase 2(6D CEM)
- [grasp_hawor.py] `_step_gripper_only`: 虚拟关节控制策略优化
  - APPROACH: set_qpos + set_drive_target (位置跳变大, 需要加速收敛)
  - CLOSE 第一帧: set_qpos 跳转到F50位置 (避免PD跟踪延迟ee_err=8cm)
  - CLOSE 后续帧: 只用 set_drive_target (让PD自然产生力交互, 有利于抓取)
- [grasp_hawor.py] `_compute_mano_neutral_target`: CLOSE阶段分Phase A(闭合50%) + Phase B(提升)
  - Phase A: 保持在F50位置, 手指闭合 (充足时间让PD收敛)
  - Phase B: 线性提升z (0→0.15m)
- [grasp_hawor.py] APPROACH阶段手指闭合时机: F50前2帧开始smoothstep闭合
- [grasp_hawor.py] 移除主循环和rollout中的post-step漂移纠正 (CLOSE阶段)
- [grasp_hawor.py] 虚拟关节PD参数: stiffness 5000→1000, damping 500→200 (对齐test5)
- [grasp_hawor.py] mppi路径: 用CEM-30D结果构建654D参数 (F48-F52固定帧)

### 验证结果
- **1/7 物体被成功夹住** (glb_1: lift=13.6cm, xy_drift=12cm)
- CEM验证: contact=44帧, lift=12.8cm, min_dist=1.17cm
- 对比之前: 0/7物体被夹住, 虚拟关节架构首次成功抓取

## [2026-07-18] test6重写: MANO轨迹对齐抓取参考系, 成功抓取物块

**类型**: 修复/重构
**影响范围**: `05_gripper_test.py` (test6, URDF, load_gripper)

### 修改内容
- [05_gripper_test.py] URDF 添加 3 个虚拟 revolute 关节 (Rz/Ry/Rx), 支持 6-DOF 夹爪控制
- [05_gripper_test.py] 添加 rotmat_to_zyx_euler: 旋转矩阵→ZYX Euler角分解
- [05_gripper_test.py] load_gripper 返回 9 个值 (6虚拟+2手指)
- [05_gripper_test.py] test6 重写: MANO轨迹对齐到抓取参考系
  - XY: 方块对齐 + MANO wrist偏移
  - Z: 抓取序列 (接近→下降→闭合→抬升→释放)
  - 姿态: MANO rpy (保持手部朝向)
  - 开合: MANO指距变化够大时映射, 否则合成

### 验证结果
- MANO轨迹抓取成功: 方块从1.3cm升至7.6cm
- 姿态变化正确: yaw [-130°,-121°], pitch [40°,43°], roll [168°,173°]

---

## [2026-07-18] 添加 test6: MANO轨迹驱动夹爪抓取物体 (混合模式)

**类型**: 新增
**影响范围**: `05_gripper_test.py` (test6)

### 修改内容
- [05_gripper_test.py] 新增 test6_mano_grasp: MANO轨迹驱动夹爪抓取物体
- [05_gripper_test.py] 混合模式: MANO提供XY位置+开合意图, 合成Z轨迹提供下降/抬升
- [05_gripper_test.py] MANO finger_dist变化太小时自动切换合成开合
- [05_gripper_test.py] 方块位置自动从MANO wrist推算

### 验证结果
- MANO轨迹抓取成功: 方块从1.3cm升至11.1cm (提升~10cm)
- hawor/7数据手指变化仅0.1cm, 自动使用合成开合

---

## [2026-07-18] 修复夹爪物理抓取: 运动学驱动→PD驱动, 对齐GalaxeaManipSim

**类型**: 修复
**影响范围**: `05_gripper_test.py` (test5 抓取释放测试)

### 修改内容
- [05_gripper_test.py] URDF 添加 3 个虚拟 prismatic 关节 (virtual_x/y/z), 夹爪通过 PD 驱动移动而非 set_root_pose 运动学驱动
- [05_gripper_test.py] PD 参数对齐 GalaxeaManipSim: stiffness=1000, damping=200
- [05_gripper_test.py] 恢复 compute_passive_force + set_qf 重力补偿
- [05_gripper_test.py] close_gripper 目标改为 0.0 (完全闭合, 持续挤压力)
- [05_gripper_test.py] 物体摩擦改为 sf=0.5/df=0.5/rest=0.6 (与 GalaxeaManipSim create_box 一致)
- [05_gripper_test.py] 手指碰撞盒 Z 尺寸 0.04→0.02 避免接触桌面
- [05_gripper_test.py] load_gripper 返回 6 个值: robot, idx_vx, idx_vy, idx_vz, idx1, idx2

### 验证结果
- 抓取测试通过: 2.5cm 方块被提起至 13.2cm (提升 ~11cm)
- 释放后方块正确掉回桌面
- 根因分析: set_root_pose (运动学驱动) 不产生真实动量和接触力 → 改用虚拟关节 PD 驱动

---

## [2026-07-12] 修复相机轨迹对比脚本的坐标系问题

**类型**: 修复
**影响范围**: `View/vis_camera_trajectories.py`

### 修改内容
- `vis_camera_trajectories.py`: 在投影到 camera-0 空间前，对 RAS 数据添加 ZUP→YUP 坐标系转换（`ZUP_TO_YUP @ t_c2w_r`）
  - 原因：HaWoR 是 Y-UP，RAS 是 Z-UP，直接对比形状时 up-axis 不统一导致轨迹形状显示异常
  - 方法：与 `vis_camera_comparison.py` 的 01 方法一致，对位置和旋转都做 ZUP→YUP 转换
- `vis_camera_aligned.py` 无需修改：HaWoR→GLB 变换和 RAS 数据都在 Z-UP 空间，坐标系一致

### 验证结果
- 修改后 RAS 和 HaWoR 轨迹在 camera-0 空间中 up-axis 统一，形状对比有效

---

## [2026-07-12] 可视化脚本迁移到 View/ 目录 + 改用 plt.show()

**类型**: 重构
**影响范围**: `View/` 目录下 7 个 vis_*.py 文件

### 修改内容
- 创建 `View/` 目录，将 7 个可视化脚本集中管理：
  - `vis_hoi4d_raw.py` — HaWoR/RAS 原始相机轨迹 + 手在 GLB 中的位置
  - `vis_camera_aligned.py` — 001 变换后的 HaWoR 相机轨迹 vs RAS 逐帧距离
  - `vis_alignment_3d.py` — 通用 3D 对齐可视化（手骨架+GLB物体+相机轨迹）
  - `vis_hoi4d_alignment.py` — 全面对齐分析（01c vs raw、Z深度、直方图）
  - `vis_depth_align.py` — 深度校正+坐标对齐完整管线可视化（含子进程调用）
  - `vis_camera_comparison.py` — 01 vs 001 方法相机对齐误差对比
  - `vis_camera_trajectories.py` — 相机轨迹形状对比（frame-0 坐标系）
- 所有脚本移除 `matplotlib.use('Agg')`，改为交互式后端
- 所有 `plt.savefig(...)` 改为 `plt.show()`，直接显示图像不保存

### 验证结果
- 所有 7 个文件已移动并修改完毕

---

## [2026-07-10] 更新 system_architecture.md: 添加 01c/001/002 三脚本实现细节

**类型**: 文档更新
**影响范围**: `system_architecture.md`

### 修改内容
- [system_architecture.md] 重写管线概览，区分新旧两套并行管线（新：01c→001→002，旧：01→02→04）
- [system_architecture.md] 添加阶段 0 完整分析：`01c_depth_align.py`（RAS 深度图校正）
  - HaWoR 手腕深度计算、RAS 深度提取（dilated mask + 中位数 + 回退机制）、校正因子插值 + 低通滤波
  - 深度校正原理：沿相机射线方向缩放，保持视线方向不变
- [system_architecture.md] 添加阶段 1a 完整分析：`001_align_scene.py`（第一帧相机坐标系对齐）
  - 核心变换链、Umeyama 尺度估计、静态相机启发式回退、cKDTree 尺度验证
  - 完整 `transform_params.npz` 字段说明
- [system_architecture.md] 添加阶段 2a 完整分析：`002_render_scene.py`（新坐标系运动学渲染）
  - 5 个核心渲染函数详解、5 种渲染模式、gripper_only 子模式、对齐策略
  - 自动手部检测（单/双手分发）、验证模式（指尖/手腕误差报告）
- [system_architecture.md] 更新数据流图：添加 01c 预处理和新管线数据流
- [system_architecture.md] 更新共享组件表：添加 `hand_to_glb`、`hawor_cam_to_glb_pose` 等新组件
- [system_architecture.md] 添加新旧管线关键区别对比表（02 vs 002）
- [system_architecture.md] 更新调用示例：包含新旧两套管线和一键管线命令
- [system_architecture.md] 添加第 12 节：管线选择指南 + 快速命令速查
- [system_architecture.md] 更新坐标系对照表，添加 GLB 原始坐标系（新管线）

### 验证结果
- 文件完整写入，所有 mermaid 流程图语法正确
- 三脚本的算法流程、输入输出、核心函数均已准确描述

---

## [2026-07-10] 单夹爪 Retargeting 加权优化

**类型**: 修改 / 修复
**影响范围**: 单夹爪物理仿真、retargeting 配置

### 修改内容
- [hand_track/configs/r1_gripper_right.yml] 恢复 3 目标点(9约束 vs 7DOF=超定); normal_delta 1e-5→5e-3; huber_delta 0.01→0.05; low_pass_alpha 1→0.3
- [04_physics_simulation.py] 新增 RetargetingSmoother 类: per-DOF-group EMA(alpha_finger=0.5, alpha_root_pos=0.4, alpha_root_rot=0.25) + 帧间限幅
- [04_physics_simulation.py] URDF 模板添加 mimic 关节(right_gripper_finger_joint2 mimic joint1)
- [04_physics_simulation.py] 修复 gripper_offset_pos: zeros→[0.1039,0,0] 补偿 10.39cm base→gripper offset
- [04_physics_simulation.py] 双手模式 override 参数同步: normal_delta=5e-3, huber_delta=0.05, low_pass_alpha=0.3

### 验证结果
- 无头测试 10 帧通过: root_pos 帧间差从大幅跳变降至约 4mm, j1=j2 对称
- 轨迹连贯平滑, 无突然跳变

---

## [2026-07-07] 深度对齐模块实现 + 文档更新

**类型**: 新增 / 修改
**影响范围**: 深度校正模块、管线集成、渲染脚本

### 修改内容

- [01c_depth_align.py] 新增深度校正脚本，用 RAS 深度图校正 HaWoR 弱透视深度误差
  - 核心函数: `compute_depth_hawor()`, `extract_hand_depth_ras()`, `compute_correction_factors()`
  - 相机坐标系深度: `R_c2w^T @ (wrist - t_c2w)` 投影取 z
  - 中位数统计 + 线性插值 + 5帧移动平均平滑
  - 支持 `--hawor-dir` 自动发现和 `--dry-run` 验证模式
- [00_run_pipeline.py] 新增 `--depth-align` 参数，Step 0 运行深度校正
  - 适配 full mode (01+02) 和 align-render mode (001+002) 两套管线
- [02_render_scene.py] `_find_reconstruction_file()` 优先加载 `*_depth_aligned.npz`
- [hand_track/common.py] `_find_reconstruction_file()` 优先加载 `*_depth_aligned.npz`
- [深度对齐.md] 重写文档以匹配实际实现（校正位置、相机坐标系深度、中位数、插值+平滑、管线集成）

### 验证结果

- `py_compile` 通过: `01c_depth_align.py`, `00_run_pipeline.py`
- 待验证: 使用实际数据端到端运行

---

## [2026-07-01] 修正文档中的文件名 — raw_track_data.pkl → hawor_results_*.npz

**类型**: 文档修正
**影响范围**: `system_architecture.md`

### 修改内容

- 将所有 `raw_track_data.pkl` 替换为正确的 `hawor_results_*.npz`（HaWoR 重建输出的实际文件名）
- 类型描述从 `pickle 字典` 改为 `NumPy .npz 字典`
- 修正 CLI 调用示例：`--raw-tracking` → `--hawor_reconstruction`（01）和 `--hawor-dir`（02/04）
- 实际存在的数据文件路径：`hand_track/output/*/hawor_world_space/reconstruction/hawor_results_*.npz`

---

## [2026-07-01] 生成 system_architecture.md — 三阶段管线综合分析文档

**类型**: 文档
**影响范围**: `system_architecture.md` (新增)

### 修改内容

- [system_architecture.md] 新增 100% 中文版综合架构分析文档（约 14KB），覆盖全部三阶段管线:
  - **1. Pipeline Overview**: 三阶段数据流总览 + mermaid 流程图
  - **2. Coordinate System Architecture**: 多坐标系变换链 (Haworth→MANO→AAB→SAPIEN), GLB up-axis 检测, `RXWORLD_TO_SAPIEN` 组合
  - **3. 01_align_scene.py (Umeyama Alignment)**: SVD/Kabsch 算法, 网格搜索尺度优化 (s ∈ [0.1, 0.5], 41 步), R_c2w 条件化转换
  - **4. 02_render_scene.py (Kinematic Rendering)**: Z-UP→Y-UP GLB 转换, MANO→retargeting→IK→set_qpos 循环, 三种基座策略, smoothstep 热身
  - **5. 04_physics_simulation.py (Physics Simulation)**: 物体分类 (static/dynamic/heavy/skip), TrajectorySmoother (Butterworth+clamping), PD 驱动 (Kp=1000, Kd=200), 接触检测, 两趟渲染
  - **6. Kinematic vs Dynamic Comparison**: 22 维度对比表格
  - **7. Pipeline Integration**: 共享组件/数据流/调用示例/坐标系一致性
- 包含 5 个 mermaid 流程图 (Pipeline Data Flow, Frame Chain, Alignment, Kinematic Loop, Physics Loop)

---

## [2026-06-30] 完整分析 04_physics_simulation.py 物理仿真管线

**类型**: 分析 + 文档
**影响范围**: `04_physics_simulation.py` (完整分析, 无代码修改)

### 分析内容

完整分析了 `04_physics_simulation.py` (3620行) 的物理仿真管线，涵盖:

1. **文件头与 Vulkan 设置** (line 1-80): SAPIEN 初始化、Vulkan 后端配置、环境检测
2. **导入与工具函数** (line 82-334): ffmpeg 重编码、矩阵转换、坐标系工具
3. **GLB 物体分类** (line 336-433): 静态 vs 动态物体自动分类 (基于体积/扁平度/最大延伸)
4. **TrajectorySmoother** (line 435-510): 低通滤波器 + 迭代裁剪 + 平滑指标计算
5. **基座位置计算** (line 1651-1748): `_compute_tracking_base_pos`, `_compute_fixed_base_clusters`, `_compute_frame_base_positions`
6. **接触检测** (line 1981-2034): `_fetch_contacts` 遍历 PhysX 接触对，统计夹爪-物体冲量
7. **物理控制步** (line 1935-1980): `_physics_step` (纯PD驱动+重力补偿+decimation) vs `_kinematic_step` (set_qpos直接设置)
8. **主仿真循环** (line 2800-3620): 实时IK渲染、PD驱动、两趟渲染、轨迹平滑后处理、视频导出

### 核心发现: 运动学 vs 动力学控制差异

| 维度 | 02_render_scene.py (运动学) | 04_physics_simulation.py (动力学) |
|------|---------------------------|----------------------------------|
| 控制方式 | `set_qpos` 直接设关节角 | `set_drive_target` + PD控制器 |
| 物理交互 | 无 (关节瞬达目标) | 有 (PD力逐步逼近, 接触检测) |
| 确定性 | 完全确定性 | 非确定性 (受物理求解器影响) |
| 重力补偿 | 不需要 | 每子步 `compute_passive_force` + `set_qf` |
| Decimation | 1次/帧 | DECIMATION次子步/帧 |
| 适用场景 | 视觉回放/演示 | 物理真实/抓取测试 |

### 关键设计决策

- **纯PD驱动**: 不调用 `set_qpos`，避免 PhysX 求解器中 PD力与直接位置约束冲突产生震荡
- **两趟渲染**: 第一趟运动学(set_qpos)产生确定性参考轨迹，第二趟PD驱动跟踪轨迹
- **基座策略**: 3种模式 (fixed_base/base_cluster/浮动基座)
- **物体分类**: 静态 if volume>0.01 AND (flatness<0.3 OR max_extent>0.8)
- **夹爪摩擦力**: 高摩擦(friction=1.0)确保稳定抓取
- **平滑模式**: 0=不平滑, 1=在线EMA, 2=后处理双向滤波

### 完整流水线

```
01_align_scene.py  →  坐标系统一 (RAS↔GLB↔HaWoR↔SAPIEN)
        ↓
02_render_scene.py  →  运动学回放 (set_qpos, 确定性)
        ↓
04_physics_simulation.py  →  动力学仿真 (PD驱动, 物理交互)
```

---

## [2026-06-26] 相机帧一致性修复 (仅相机, 手部/GLB 不变) + 重建 COMMANDS.md

**类型**: 修复 + 新增
**影响范围**: `hand_track/common.py`, `02_render_scene.py`, `COMMANDS.md`

### 问题诊断

用户反馈: "相机左右反, 手部和 GLB 是正确的". 经 4 配置实测 (transform × extraction),
发现根因是相机 transform 用错矩阵:

- 手部 = `RXWORLD_TO_SAPIEN @ SLAM_data` = `R_AXIS @ R_x @ SLAM` = `R_AXIS @ OpenGL` (帧: R_AXIS@OpenGL)
- GLB   = `RXWORLD_TO_SAPIEN @ SLAM_data` (同上, 与手部一致) ✓
- 相机 (BUG) = `RXWORLD_TO_SAPIEN @ stored` = `R_AXIS @ R_x @ R_x @ SLAM` = `R_AXIS @ SLAM` (帧: R_AXIS@SLAM)
- 相机与手部/GLB 帧相差 R_x = diag(1,-1,-1) → 相机位置轨迹与手部不同帧 → "左右反"

### 修改内容 (仅改相机, 手部/GLB 保持 RXWORLD_TO_SAPIEN 不变)

- [hand_track/common.py] `hawor_cam_to_sapien_pose` 相机 transform 改为 R_AXIS
  - 修复前: `cam_pos = RXWORLD_TO_SAPIEN @ t_c2w`, `cam_R = RXWORLD_TO_SAPIEN @ R_c2w` (帧: R_AXIS@SLAM, 与手部不同帧)
  - 修复后: `cam_pos = R_AXIS @ t_c2w`, `cam_R = R_AXIS @ R_c2w` (帧: R_AXIS@OpenGL, 与手部同帧)
  - 原理: stored = R_x @ SLAM = OpenGL, 所以 R_AXIS @ stored = R_AXIS @ OpenGL = 手部帧
- [hand_track/common.py] `hawor_cam_to_sapien_pose` 相机约定提取改为 OpenCV
  - 修复前: `forward=-col2, up=+col1` (OpenGL 约定, 导致 forward 朝错误方向)
  - 修复后: `forward=+col2, up=-col1` (OpenCV 约定, R_c2w 已应用 R_x 但相机约定仍是 OpenCV)
- [02_render_scene.py] 同步上述 2 处相机修复 (transform + extraction)
- [02_render_scene.py] 文档注释更新 (坐标系变换链说明三者同帧 = R_AXIS@OpenGL)
- **手部/GLB 保持不变**: `_render_to_sapien`, `load_glb_transformed`, `ras_origin_sapien` 仍用 `RXWORLD_TO_SAPIEN`
- [COMMANDS.md] 重新创建 (原文件丢失), 汇总所有 12 个可调用命令 + 2 个示例

### 4 配置实测结果 (`/tmp/test_camera_configs.py`)

| Config | transform | extract | fwd·c2h | up·WU(-Z) | 帧一致 | 判定 |
|--------|-----------|---------|---------|-----------|--------|------|
| A (BUG) | RXWORLD | OpenGL | +0.933 | +0.998 | ✗ (hoi4d: 0.826 vs 0.478) | ✗ |
| B | RXWORLD | OpenCV  | -0.933 | -0.998 | ✗ | ✗ |
| C | R_AXIS  | OpenGL  | -0.934 | -0.998 | ✓ | ✗ |
| D (修复)| R_AXIS  | OpenCV  | +0.934 | +0.998 | ✓ | ✓ |

### 验证结果

- 语法检查通过: common.py, 02_render_scene.py
- `/tmp/verify_actual_functions.py` 用 common.py 实际函数验证 (2 个数据集):
  - 数据集 7 (113帧): forward·cam2hand=+0.934, up·WORLD_UP=+0.998, |c2h|_sapien/|c2h|_slam=1.0000 ✓
  - 数据集 hoi4d (599帧): forward·cam2hand=+0.771, up·WORLD_UP=+0.953, |c2h|_sapien/|c2h|_slam=1.0000 ✓
- 关键: 帧一致性 (|c2h|_sapien == |c2h|_slam) 证明相机与手部/GLB 在同一帧中
- 注: 相机抖动来自 SLAM 原始数据 (jitter std/mean=1.375~1.780), 与本次约定修复无关

---

## [2026-06-26] dual_tracking 固定基座 + 删除关键点视频 + dex优化器选项

**类型**: 修复 + 新增
**影响范围**: `hand_track/common.py`, `hand_track/render_auto.py`, `00_run_pipeline.py`

### 修改内容

- [hand_track/common.py] `render_robot_video` 添加 `fixed_base=True` 参数, 默认固定基座不移动
  - 修复前: 每帧用 `_compute_tracking_base_pos` 移动基座 (±4cm), 导致双臂位姿不协调
  - 修复后: 基座固定在 `_compute_optimal_fixed_base` 计算的初始位置
- [hand_track/render_auto.py] 删除双手路径的关键点球体视频 (`hawor_r1_dual_gripper.mp4`)
  - 用户不需要分开渲染的关键点视频, 只需要 `hawor_r1_dual_gripper_urdf.mp4`
- [hand_track/render_auto.py] 添加 `--optimizer` 参数, 可选 dex_retargeting PositionOptimizer
  - 默认: `analytical=True` (Gram-Schmidt 解析法, 夹爪开合较生硬)
  - `--optimizer`: `analytical=False` (dex PositionOptimizer, 指尖精度更高, 开合更自然)
  - 传递到 `render_gripper_only_video` 和 `render_dual_gripper_video`
- [00_run_pipeline.py] 添加 `--optimizer` 参数, 传递到 render_auto.py

### 验证结果

- 语法检查通过: common.py, render_auto.py, 00_run_pipeline.py
- `render_dual_gripper_video` 和 `render_gripper_only_video` 均接受 `analytical` 参数
- `--optimizer` flag 正确映射到 `analytical=not args.optimizer`

---

## [2026-06-26] robot_only MANO side 修复 + handtrack both 模式 + 相机轨迹分析

**类型**: 修复 + 新增
**影响范围**: `02_render_scene.py`, `hand_track/render_auto.py`, `00_run_pipeline.py`, `COMMANDS.md`

### 修改内容

- [02_render_scene.py:2403] 修复 `run_robot_only` 的 MANO side 硬编码 bug:
  - 修复前: `MANOLayer(prefix, betas_mean)` — prefix 永远 "right", 左手数据被错误用右手模型解读
  - 修复后: `mano_side = "left" if hi == 0 else "right"; MANOLayer(mano_side, betas_mean)`
  - 这导致 robot_only 和 robot_tracking 渲染出不同的机器人运动 (tracking 是正确的)
- [hand_track/render_auto.py] 添加 `--mode both` 选项, 同时渲染夹爪 + 带机械臂夹爪两种URDF视频
  - tracking 和 keypoint 视频只渲染一次 (与 with_arm 无关)
  - 夹爪URDF视频按 render_modes 循环渲染
- [00_run_pipeline.py] `--handtrack` 模式自动传 `--mode both`
- [COMMANDS.md] 添加 handtrack 输出文件说明

### 验证结果

- 语法检查通过: 02_render_scene.py, render_auto.py, 00_run_pipeline.py
- argparse 验证: `--mode` choices = ['gripper', 'gripper_arm', 'both']
- render_modes 逻辑: both → [False, True], 正常模式 → [单值]
- for 循环出现 2 次 (单手 + 双手各一处)

### 相机轨迹分析结论

- 相机轨迹代码无 bug: `hawor_cam_to_sapien_pose` 对 7 和 hoi4d 处理完全相同
- 两个数据集 R_c2w/t_c2w 约定一致 (右手系 c2w, OpenGL 相机, det=+1)
- 所有 R_align det=+1 (无反射镜像)
- hoi4d 相机偏航变化 -37° vs 7 的 -0.43°, 是 SLAM 数据本身差异
- slam_scale/img_center 差异不影响渲染 (代码不读这两个键)

---

## [2026-06-26] 00_run_pipeline.py 添加 --handtrack 选项

**类型**: 新增
**影响范围**: `00_run_pipeline.py`, `COMMANDS.md`

### 修改内容

- [00_run_pipeline.py] 添加 `--handtrack` 参数, 启用时默认步骤变为 `1,7` (对齐+hand_track), 替代 02 的步骤 2-5
- [00_run_pipeline.py] `--steps` 默认值改为 None, 根据 `--handtrack` 动态决定; 用户仍可通过 `--steps` 覆盖
- [00_run_pipeline.py] Step 7 增加 `--hand-idx` 传递, 从 `--handedness` 映射 (auto→-1, left→0, right→1, both→-1)
- [COMMANDS.md] 新增 "0. 一键管线" 章节, 包含传统模式和 handtrack 模式的完整命令示例
- [COMMANDS.md] 数据路径约定增加 hoi4d 路径

### 验证结果

- `python 00_run_pipeline.py --help` 正确显示 `--handtrack` 选项
- 默认步骤逻辑测试: 无 `--handtrack` → `1,2,3,4,5`; 有 `--handtrack` → `1,7`; `--steps` 覆盖正常

---

## [2026-06-25] hand_track Z-UP GLB 修复 + 文件夹审查

**类型**: 修复
**影响范围**: `hand_track/common.py`, `docs/questions.md`

### 修改内容

- [hand_track/common.py] `load_glb_transformed()` 添加 Z-UP GLB 顶点转换:
  - 新增 `ZUP_TO_YUP` 常量 (line 72)
  - 从 transform_params.npz 读取 `glb_up_axis` (line 334)
  - Z-UP GLB 顶点先做 `ZUP_TO_YUP @ vertices.T` 转换再应用 R_inv (line 347-348)
- [docs/questions.md] 追加 Q&A: hand_track 文件夹审查结果

### 审查结论

审查了 hand_track 文件夹全部 7 个 Python 文件和 2 个配置文件:
- 无功能性 bug (除已修复的 Z-UP GLB 问题)
- 发现代码重复: `_combine_videos_side_by_side` 和 `_ensure_transform_params` 在 render_gripper_only.py 和 render_auto.py 各有一份
- 不影响正确性，可后续提取到 common.py 统一维护

---

## [2026-06-25] 对齐公式修复: R_c2w 转换条件化 + 尺度验证

**类型**: 修复
**影响范围**: `01_align_scene.py`

### 问题

hoi4d1_vggt_omega 数据集的对齐结果手→GLB距离过大 (min=0.40m)，手没有靠近GLB物体。
根因: GLB 使用地面坐标系 (Z-UP) 时，R_c2w 的 Y-UP 转换方式不同于相机坐标系 (Y-UP GLB)。

### 修改内容

- [01_align_scene.py] R_c2w Y-UP 转换条件化:
  - Y-UP GLB (相机坐标系, 如 "7" 数据): 相似变换 `ZUP_TO_YUP @ R @ ZUP_TO_YUP.T` (方式A)
  - Z-UP GLB (地面坐标系, 如 hoi4d1 数据): 直接乘 `ZUP_TO_YUP @ R` (方式B)
  - 在 `compute_and_save_transform_params()` 和 `main()` 两处同步修改
- [01_align_scene.py] 添加 Umeyama 尺度验证:
  - 计算手→GLB最近顶点距离，若 > 10cm 则自动网格搜索更优 s_inv
  - 在 `compute_and_save_transform_params()` 和 `main()` 两处同步添加
- [01_align_scene.py] main() 添加 GLB 坐标系检测 (Step 1.5)
- [01_align_scene.py] 保存 glb_up_axis 使用步骤 1.5 的检测结果 (统一)

### 验证结果

| 数据集 | GLB | R_c2w 方法 | 尺度来源 | 手→GLB min | 手→GLB mean |
|--------|-----|-----------|---------|-----------|------------|
| "7" | y-up | 相似变换 | Umeyama | 0.0041m | 0.0451m |
| hoi4d1 | z-up | 直接乘 | 网格搜索 | 0.0002m | 0.1973m |

---

## [2026-06-25] FPV 第一人称跟随 + Topdown 上帝视角 + 双手 (Bimanual) 逻辑

**类型**: 新增 / 修改 / 修复
**影响范围**: `04_physics_simulation.py` (主文件, ~3720 行)

### 任务背景

用户需求:
1. 相机是第一人称视角的跟随，不要去更改相机位置，更改基座的位置，基座可以调整低一点
2. 应该有 topdown 的视角 (上帝视角，负责查看整体的情况)
3. (扩展) 添加双手同时驱动两个机械臂的能力，含自动检测

### 修改内容

#### 任务1: FPV 跟随 + Topdown 视角

**常量调整** (`04_physics_simulation.py:104-111`):
- `COMFORTABLE_REACH`: 0.70 → 0.40 (基座高度降低，让机械臂水平延伸，不从上方挡住相机视野)
- 新增 `BASE_OFFSET_Y = 0.30` (基座 Y 偏移到相机后方，相机看 -Y，基座在 +Y 不挡视野)
- `COMFORT_TARGET_IN_BASE`: Z 从 -0.55 → -0.35 (跟随基座高度调整)
- 新增 `LEFT_ARM_STARTING` (镜像 RIGHT，joint1/4/6 取反)
- 新增 `FLOATING_LEFT_URDF` 路径常量

**基座位置计算** (`_compute_optimal_fixed_base`, `_compute_fixed_base_clusters`):
- 3 处分支添加 `arm_base_pos[1] += BASE_OFFSET_Y` (基座偏移到相机后方)

**视角支持** (已有, 验证可用):
- `--view fpv`: 第一人称跟随 (相机位于手腕位置，看前方)
- `--view topdown`: 上帝视角 (高度 0.6m，俯视整个场景)
- `--view behind`: 后方第三人称视角

#### 任务2: 双手 (Bimanual) 逻辑

**2.1 `_detect_hand_idx` 改返回 list** (`04_physics_simulation.py:466-488`):
- 改前: 返回 `int` (0/1/None)，两者都有时默认左手
- 改后: 返回 `list[int]`，支持 `[]` / `[0]` / `[1]` / `[0,1]` (双手)

**2.2 `load_hawor_data` 支持 list** (`04_physics_simulation.py:491+`):
- 改前: `hand_idx: int` → 返回单 dict
- 改后: `hand_idx: int | list[int]`，list 时返回 `{0: dict, 1: dict}` 双手数据

**2.3 `PhysicsSimulator.__init__` 支持 list** (`04_physics_simulation.py:1434-1435`):
- `self.hand_idx = hand_idx if isinstance(hand_idx, list) else [hand_idx]`
- `self.bimanual = len(self.hand_idx) == 2`

**2.4 单手流程参数化** (`run_physics_tracking`):
- 引入 `mano_side` / `prefix` / `hand_type` / `starting` 动态变量
- 左手 (`hand_idx=0`): `prefix="left"`, `hand_type=HandType.left`, `starting=LEFT_ARM_STARTING`
- 右手 (`hand_idx=1`): `prefix="right"`, `hand_type=HandType.right`, `starting=RIGHT_ARM_STARTING`
- `_get_gripper_pose_from_retargeting` 增加 `gripper_link_name` 参数
- `_fetch_contacts` 增加 `prefix` 参数 (检测对应侧夹爪接触)
- IK 求解器动态选择: `getattr(ik_solver, f"relaxed_ik_{prefix}")` / `getattr(ik_solver, f"solve_position_{prefix}")`
- `warm_start` 参数化 `hand_type`
- 所有 `RIGHT_ARM_STARTING` 硬编码替换为 `starting` 变量 (7 处)

**2.5 新增 `run_bimanual_tracking` 方法** (`04_physics_simulation.py:3333+`):
- 双手追踪主流程 (运动学模式，简化版)
- 共享同一 SAPIEN 场景和相机
- 两只手分别 retargeting + IK + `set_qpos` 驱动
- 支持 FPV / topdown / behind 三种视角
- 不含两趟渲染/轨迹平滑/接触检测 (单手版的增强功能)

**2.6 CLI `--hand-idx` 改为 str** (`04_physics_simulation.py:3638`):
- 改前: `type=int, default=0`
- 改后: `type=str, default="-1"`，接受 `"0"` / `"1"` / `"both"` / `"-1"` (自动检测)
- 自动检测逻辑: 调用 `_detect_hand_idx`，根据返回 list 长度分发到单手/双手模式

**2.7 主入口分发** (`04_physics_simulation.py:3745+`):
- `sim.bimanual=True` → `run_bimanual_tracking`
- `sim.bimanual=False` → `run_physics_tracking`

#### Bug 修复 (附带)

- **GLB path None 修复** (`04_physics_simulation.py:2621`):
  - 改前: `if not glb_path.exists():` (当 glb_path 为 None 时崩溃)
  - 改后: `if glb_path is None or not glb_path.exists():`
- **Triple `main()` 调用修复** (文件末尾):
  - 改前: `main()` 被调用 3 次 (导致程序运行 3 遍)
  - 改后: 单次 `main()` 调用

### 验证结果

**任务1 验证** (10 帧测试):
- 左手 FPV: 10/10 ✓
- 右手 FPV: 10/10 ✓ (无回归)
- 左手 topdown: 10/10 ✓ (27.8% 彩色内容，俯视可见整体)
- 右手 topdown: 10/10 ✓

**任务2 验证** (10 帧测试):
- 自动检测 (单手数据): 10/10 ✓ (正确检测 "左手")
- 双手 topdown: 10/10 ✓ (左右两臂同时加载并追踪)
  - 日志: `✓ left 臂已加载: 6 臂关节 + 2 夹爪关节`
  - 日志: `✓ right 臂已加载: 6 臂关节 + 2 夹爪关节`

**测试命令示例**:
```bash
# 单手 FPV
python 04_physics_simulation.py --hand-idx 0 --num-frames 10 --view fpv
# 单手 topdown
python 04_physics_simulation.py --hand-idx 0 --num-frames 10 --view topdown
# 自动检测
python 04_physics_simulation.py --hand-idx -1 --num-frames 10 --view topdown
# 双手 topdown
python 04_physics_simulation.py --hand-idx both --num-frames 10 --view topdown
```

### 已知限制 / 待解决 (非本次任务)

- `run_bimanual_tracking` 为运动学模式简化版，未含两趟渲染/轨迹平滑/接触检测
- PyBullet 管道中盘子不动 (prior issue, 非本次范围)
- 单夹爪 MANO 参考点 (prior issue, 非本次范围)
- 手指 link 摩擦 (prior issue, 非本次范围)

### 相关文档

- 实现计划: `docs/specs/2026-06-25-camera-view-and-bimanual-plan.md` (本次任务的完整步骤计划)

---

## [2026-06-26] 修复 06_visualize_alignment.py 相机箭头方向

**类型**: 修复
**影响范围**: `06_visualize_alignment.py`

### 问题诊断

用户反馈 "相机箭头完全错误". 干运行显示 `R_AXIS @ t` 位置在相机远离原点时与场景坐标系不一致:
- 600 帧中仅 246 帧 (41%) 朝向场景 (mean dot = -0.03)
- 参考代码 `01_align_scene.py` 使用 `RXWORLD_TO_SAPIEN @ t` + `-(RXWORLD @ R)[:,2]`, 600 帧中 472 帧 (79%) 朝向场景 (mean dot = 0.40)

### 修改内容

- [06_visualize_alignment.py] `load_hawor_cameras`:
  - 位置: `R_AXIS @ t` → `RXWORLD_TO_SAPIEN @ t` (与 GLB/手部同帧)
  - 朝向: `R_AXIS @ R[:,2]` → `-(RXWORLD_TO_SAPIEN @ R)[:,2]` (匹配参考代码)
- [06_visualize_alignment.py] 诊断 `scene_to_cam` → `cam_to_scene` (点积符号修正)

### 验证结果

- `Frame 0 forward·scene_dir dot: 0.6812 ✓ 朝向场景`
- 参考方法: 472/600 帧正向 (79%), my_method: 246/600 (41%)
- 位置差: 帧 100 前差异 <0.01m, 帧 200+ 差异可达 0.89m

---

## [2026-06-27] 简化 06_visualize_alignment.py：轨迹线替代逐帧标记 + 死代码清理

**类型**: 重构 + 修复
**影响范围**: `06_visualize_alignment.py`

### 修改内容

- [06_visualize_alignment.py] `RXWORLD_TO_SAPIEN` 从 `R_AXIS @ R_x` 改为 `R_AXIS`，与 `01_align_scene.py` 一致
- [06_visualize_alignment.py] 相机 forward: `RXWORLD_TO_SAPIEN @ R[:,2]` → `-RXWORLD_TO_SAPIEN @ R[:,2]`，匹配 01 参考代码
- [06_visualize_alignment.py] 删除 `_init_hand_markers`, `_set_hand_positions`, `_set_hand_lines`, `_hide_all`, `_update_frame`, 帧前进回调
- [06_visualize_alignment.py] 新增 `_build_hand_trajectories()`: 从所有有效帧的关节中心生成两条静态轨迹线 (蓝=右手, 橙=左手)
- [06_visualize_alignment.py] `AlignmentVisualizer.__init__` 参数简化: 接收 `hand_traj_lines` 替代 `avg_joints`/`all_mano_kps`/`wrist_fb`/`n_frames`
- [06_visualize_alignment.py] 清理死代码: 重复 `run()` 方法、遗留 `_on_h` 方法、重复的视图初始化块
- [06_visualize_alignment.py] Legend 更新: 蓝/橙轨迹线替代原来的 Cyan/Orange 球体 + 手指线

### 验证结果

- 干运行: `Frame 0 forward·scene_dir dot: 0.6812 ✓ 朝向场景`
- 482/599 帧 (80.5%) 相机朝向最近手部, mean dot = 0.588

---

## [2026-06-27] 修复相机 forward yaw 反转: 移除 R_X 和全局负号

**类型**: 修复
**影响范围**: `06_visualize_alignment.py`, `01_align_scene.py`

### 修改内容

- [06_visualize_alignment.py] `load_hawor_cameras`: `forwards[i] = -RXWORLD_TO_SAPIEN @ fwd_render` → `forwards[i] = R_AXIS @ fwd_render`
- [01_align_scene.py] `forward = -cam_R_sapien[:, 2]` → `forward = R_AXIS @ R_c2w_hawor[:, 2]`

### 根因

`R_X = diag(1,-1,-1)` 对 forward 向量的 Y/Z 取负, 全局 `-` 撤销了 Z 负号但同时也翻转了 X → yaw 反向。
新旧公式在 frame 0 (f_x≈0) 时点积几乎相同 (~0.89), 所以在静态测试中无法发现问题。

### 验证结果

- Frame 400→500: raw f_x `-0.0069 → +0.5905` (Δ=+0.5974, 右转)
  - OLD x_sap: `+0.0069 → -0.5905` (Δ=-0.5974, 左转 ✗)
  - NEW x_sap: `-0.0069 → +0.5905` (Δ=+0.5974, 右转 ✓)
- 干运行: `Frame 0 forward·scene_dir dot: 0.9223 ✓ 朝向场景`

---

## [2026-07-02] 添加夹爪位姿日志 `_log_gripper_pose`

**类型**: 新增
**影响范围**: `04_physics_simulation.py`

### 修改内容

- [04_physics_simulation.py:1570] 新增 `_log_gripper_pose(robot, label)` 方法:
  - 记录 `root_pos=(p0, p1, p2)` 和 `root_quat=(q0, q1, q2, q3)`
  - 格式与 `_log_object_positions` 对齐，使用 `── 标签 ──` 分隔
- [04_physics_simulation.py:2439] `run_single_gripper_tracking` 物理稳定化后调用:
  - `self._log_gripper_pose(robot, "夹爪初始位姿 (warm-up后)")`
- [04_physics_simulation.py:2526] `run_single_gripper_tracking` 渲染循环中每帧调用:
  - `self._log_gripper_pose(robot, f"帧 {global_idx}")`

### 用途

诊断单夹爪模式下夹爪渲染坐标与 GLB/手部坐标不一致的问题，提供夹爪位置 + 四元数旋转的完整日志。

---

## [2026-07-03] CoACD 缓存写入 + 碰撞体可视化修复

**类型**: 新增 / 修复
**影响范围**: `04_physics_simulation.py`

### 修改内容

- [04_physics_simulation.py] **CoACD 缓存写入**: 原先 CoACD 结果只读不写，每次运行都重新分解。现在使用 `coacd` Python 包直接运行凸分解，结果保存到 `physics_cache/*.npz`，下次运行时直接读取缓存（秒级加载）
  - `else` (cache miss) 分支: 用 `coacd.Mesh` + `coacd.run_coacd` 分解 → `np.savez` 保存 → 逐个凸部件 `add_convex_collision_from_file`
  - `except` (cache 加载失败) 回退分支: 同样用 coacd 包重算并保存到缓存
  - 两个分支都保留 `coacd` 包未安装时的回退（SAPIEN 内部 CoACD，不缓存）
  - 安装: `pip install coacd`
- [04_physics_simulation.py] **碰撞体可视化 (collision_vis)**: 在 CoACD 缓存路径中，为每个凸部件单独生成可视化 PLY，确保红色半透明外壳正确显示

### 验证结果

- `py_compile` 语法检查通过
- `coacd` 包 (v1.0.5) 已安装，API 验证通过
- 缓存路径: `final_scene.physics_cache/` 目录下，每个几何体对应一个 `.npz` 文件
- 缓存格式: `convex_parts` 为 `list[tuple(verts, faces)]`，world-frame 坐标

---

## [2026-07-03] 致命 Bug 修复: 物体穿地 + CoACD 缓存序列化 + 缓存目录只读

**类型**: 修复
**影响范围**: `04_physics_simulation.py`

### 修改内容

- [04_physics_simulation.py] **修复 `np.savez` 导致凸部件未添加的 Bug**: 原代码在添加凸部件到 builder **之前**调用 `np.savez`，而 `np.savez` 对变长凸部件数组失败（`inhomogeneous shape`），导致整个 `try` 块跳到 `except`，凸部件从未添加进 builder，回退到非凸碰撞体 → 所有物体穿地到 Z=-30.6
  - **修复**: 先循环添加凸部件到 builder，**再**保存缓存。保存失败不影响碰撞体
- [04_physics_simulation.py] **缓存序列化改用 pickle**: `np.savez` 不支持变长数组（不同凸部件有不同顶点数），改用 `pickle.dump/load` 支持任意形状
- [04_physics_simulation.py] **缓存目录从只读文件系统迁移**: 原 `glb_path.parent/physics_cache/` 位于只读文件系统（`/home/an/data/ras/`），无法创建缓存文件。改为 `physics_pipeline/output/physics_cache/`（可写）
- [04_physics_simulation.py] **缓存文件名扩展名改为 `.pickle`**: 与旧 `.npz` 文件区分
- [04_physics_simulation.py] **缓存命中时 PLY 文件名冲突修复**: 原代码所有凸部件写入同一个 PLY 文件，后一个覆盖前一个 → builder 中只有最后一个凸部件。改为 `_p{part_i}.ply` 唯一文件名
- [04_physics_simulation.py] **CoACD 失败回退策略改进**: 从非凸碰撞 → SAPIEN 内部 CoACD（生成正确凸碰撞体，不穿地），非凸碰撞作为三重保底
- [04_physics_simulation.py] **碰撞体可视化错误日志**: 修复 `except Exception: pass` 静默吞掉错误的问题，现在输出具体错误信息和文件路径
- [04_physics_simulation.py] **添加 `import pickle`**

### 验证结果

- `py_compile` 语法检查通过
- 旧 `.npz` 缓存目录为只读文件系统，已无法删除（不影响运行，新缓存写入新目录）

### 关键数据流 (修复后)

1. `run_coacd` 成功 → 得到 5 个凸部件 ✓
2. **先** 循环添加每个凸部件到 builder（`add_convex_collision_from_file`）
3. **再** 用 `pickle.dump` 保存到 `physics_pipeline/output/physics_cache/*.pickle`
4. 若步骤 1 失败 → 回退 SAPIEN 内部 CoACD → 若再失败 → 非凸碰撞保底
