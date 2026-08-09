# Q&A

## Q: grasp_hawor.py 与 v4.4 计划的差距分析 (Stage 1 已验证, Stage 2/3 有问题)

**日期**: 2026-07-24
**分类**: 架构 / 调试

### 问题
对比 `trajectory_optimization/grasp_hawor.py` 与 `.trae/documents/grasp_pose_optimization_plan.md`,
分析完成度、偏离项、潜在问题 (坐标系映射 / 渲染 / 摩擦力)。

### 解答

**完成度估算**: ~75% (Stage 1 已验证通过; Stage 2/3 代码已写但存在关键偏离 + 1 个 bug)

**Stage 0** (`generate_grasp_candidates` L5035): ✅ 基本符合
- 候选生成网格 (4×4×2×3=96→32) 与 plan 的 4×4×2 一致
- ⚠️ `z_step=0.005` (±5mm) 而 plan 要求 ±3mm

**Stage 1** (`cem_grasp_pose_optimize` L5480): ✅ Stage 1b/1c 架构符合 plan
- ⚠️ Stage 1 rollout 中 `_GRIP_TARGET = _GRIP_CLOSE` (L5220) **覆盖了** `compute_gripper_qpos` 结果 — 与 plan 第 246-253 行矛盾
- ⚠️ `target_pos[2] = _hover_z` (L5189) **强制覆盖 z**, 让 CMA-ES 的 dz 搜索失效
- ⚠️ Stage 1 rollout 禁用手指-地面碰撞 (L5247-5267), 与 Stage 4 回放物理不一致
- 🔴 **`_check_mano_hard_constraint` bug** (L5524-5538): 实现比较 R vs `ref_base_R` (`_rot_close`), 而 plan 明确要求 "候选 R 与 MANO F50 R 的角度差 ≤ 5°"。由于所有候选都是 `_rot_close` + ±1.1° 扰动, 硬约束实际失效, 没有真正约束与 MANO 姿态的相似度。

**Stage 2** (`reconstruct_trajectory` L5713): ⚠️ 部分偏离
- ✅ 位置/姿态/夹爪开合度全部插值
- ⚠️ 用 **Minimum Jerk 五次插值** (C2) 而非 plan 的 "Hermite 三次 (C1)" — 实际更平滑, 不算 bug
- ⚠️ `run_v4_pipeline` 调用时 `N_TRANS=6` (F44~F56, 13帧), 而 plan 要求 `N_TRANS=4` (F46~F54, 9帧)
- ⚠️ Stage 2 输出转 `_fixed_offsets_654` 时, 旋转偏移用 `R_scipy.from_matrix(R_corr).as_euler('xyz')`, 必须与 `generate_trajectory_from_params` 的解码方式一致 (待验证)

**Stage 3** (`stage3_reward` L5832 / `cem_stage3_optimize` L6081): ⚠️ 偏离较多
- ✅ 分区间权重 (F1-45: 1/30, F55-90: 2/40) 与 plan 完全一致
- ✅ F46-F54 跳过, F91-F112 锁定跟随 MANO
- ✅ 平滑 10.0, 物体跟随 500.0, 弹飞 1000.0 — 与 plan 一致
- 🔴 `max_penetration` 在 `rollout_v4_stage3` L6070 **硬编码为 0.0**, 穿透惩罚永远不触发
- 🔴 默认 `n_iterations=10, population_size=12`, 远低于 plan 的 `20×64`
- ⚠️ `rollout_v4_stage3` 用 `decimation = DECIMATION // 2 = 4` (半精度), 验证不充分

**坐标系映射问题** (用户关注点):
- ✅ `001_align_scene.py` 的 `R_hand_to_glb` / `t_hand_to_glb` / `scale_ratio` 已建立 HaWoR→GLB 变换
- ✅ `vis_camera_trajectories.py` 验证相机轨迹对齐 (用户已确认完成)
- ⚠️ `run_v4_pipeline` 中 Stage 1 用 `gripper_only` 模式跑 (虚拟关节直接设世界位姿), Stage 3/4 用 `full_robot` 模式跑 (需 IK 解算)。`_mano_neutral_offset` 从 stage1_sim 复制到主 sim, 但 full_robot IK 可能无法到达 target_pos → 抓取失败
- ⚠️ `_compute_mano_neutral_target` (L2582) 主路径只读 `opt_pos[_F50]` 作为初始物体位置 (L2640), 没有同步 Stage 1 的 `best_grasp['pos']`, Stage 3/4 用的 F50 位置可能 ≠ Stage 1 优化结果

**摩擦力对比**:
- `grasp_hawor.py`: `GRIPPER_FRICTION=1.0` (L200)
- `05_gripper_test.py` test7 (2026-07-24 成功): 摩擦 3.0/3.0
- 1.0 vs 3.0 差 3 倍, 配合 force_limit=40N 可能仍能抓, 但若 full_robot IK 跟踪误差大, 摩擦裕度不足

**优先修复顺序建议** (用户后续任务):
1. 修 `_check_mano_hard_constraint` — 改为比较 R vs MANO F50 R (`_get_mano_f50_pose`)
2. 修 `_GRIP_TARGET` override — 让 Stage 1 用 `compute_gripper_qpos(bbox)` 的真实值
3. 修 `target_pos[2]` override — 让 z 搜索生效
4. 验证 Stage 2 输出 → `_fixed_offsets_654` 的旋转解码一致性
5. 修 `max_penetration` 计算 (或暂时移除该惩罚项)
6. 提高Stage 3 默认 `n_iterations/population_size` 到 plan 的 20/64
7. 同步 Stage 1 `best_grasp['pos']` 到 `_compute_mano_neutral_target` 的 F50 位置
8. 考虑提高 `GRIPPER_FRICTION` 到 3.0 (对齐 test7 成功参数)
9. 移除 Stage 1 的"禁用手指-地面碰撞" hack, 或在 Stage 4 回放时也禁用 (保持一致)

---

## Q: dual_tracking位姿错误 + 删除关键点视频 + dex优化器 + 相机轨迹映射

**日期**: 2026-06-26
**分类**: 调试 / 修复

### 问题

1. `hawor_r1_dual_tracking.mp4` track完全没法看, 机械臂夹爪位置位姿都不对
2. `hawor_r1_dual_gripper.mp4` 分开渲染的部分可以删除, 只需要 `hawor_r1_dual_gripper_urdf.mp4`; 夹爪开合生硬, 有用dex优化吗
3. 相机轨迹确定用hawor的吗? 方向有点问题, 怎么映射到当前坐标系?

### 解答

1. **dual_tracking 固定基座 (已修复)**: `render_robot_video` 之前每帧用 `_compute_tracking_base_pos` 移动基座 (±4cm), 双臂各自独立移动导致不协调。添加 `fixed_base=True` 参数, 基座固定在初始位置。双臂仍分开渲染后合成, 不生成单手视频。

2. **删除关键点视频 + dex优化器 (已实现)**:
   - 删除双手路径的 `hawor_r1_dual_gripper.mp4` (关键点球体) 渲染
   - 之前默认 `analytical=True` (Gram-Schmidt), **没有用 dex PositionOptimizer**
   - 添加 `--optimizer` 选项: `analytical=False` 时调用 `retargeting.retarget()` 走 PositionOptimizer + 3点约束, 指尖精度更高, 开合更自然

3. **相机轨迹映射链**:
   ```
   HaWoR npz (R_c2w, t_c2w) → load_hawor_c2w() 直接读取, 无预处理
   → hawor_cam_to_sapien_pose():
     ① cam_pos = RXWORLD_TO_SAPIEN @ t_c2w  (位置: Y-up→Z-up)
     ② cam_R = RXWORLD_TO_SAPIEN @ R_c2w    (旋转: Y-up→Z-up)
     ③ forward = -cam_R[:,2]  (OpenGL: -Z前)
     ④ left = -cam_R[:,0]     (OpenGL: -X左)
     ⑤ up = cam_R[:,1]        (OpenGL: +Y上)
     ⑥ 构建 SAPIEN 相机旋转矩阵 [forward|left|up]
   → sapien.Pose(cam_pos, quaternion)
   ```
   `RXWORLD_TO_SAPIEN = [[1,0,0],[0,0,-1],[0,1,0]]` (det=+1, 正常旋转)
   数值验证: 手在相机前方 (点积 +0.933), 前向方向正确。两个数据集约定一致 (右手系 c2w, OpenGL相机, det=+1)。

---

## Q: SAPIEN 穿桌/无夹取 + PyBullet 与 SAPIEN 统一方案

**日期**: 2026-06-24
**分类**: 架构 / 调试

### 问题
用户提出 4 点:
1. 基座高度不够，机械臂应垂直抓取，PyBullet 也改成固定基座
2. SAPIEN 机械臂穿桌，物理仿真是否真正实现？文件命名要包含参数避免覆盖
3. SAPIEN 没有夹取物体，分析原因
4. PyBullet 效果不如 SAPIEN，机械臂位置错误，没和 GLB 交互，要求相机/机械臂/夹爪/GLB 全部对齐 SAPIEN，统一两个项目

### 解答

#### 1. 基座高度问题
- **SAPIEN 04**: `COMFORTABLE_REACH` 已从 0.55 → 0.70m，`COMFORT_TARGET_IN_BASE=[0.25, 0.0, -0.55]`，机械臂垂直下垂抓取
- **PyBullet**: 临时回退到 0.35m，因为 PyBullet 直接使用 02 的 IK 轨迹 (基于 0.35m 基座计算)
- **根本矛盾**: PyBullet 想要 0.70m 基座，但 IK 轨迹是基于 0.35m 计算的，强行提高基座会导致机械臂位置偏移 0.35m

#### 2. SAPIEN 机械臂穿桌原因
物理仿真**确实实现了** (PhysX 求解器 + 碰撞检测 + 重力)，但穿桌的原因:
1. **PD 跟踪误差**: SAPIEN 04 用 PD 驱动 (stiffness=1000, damping=200)，不是运动学控制。PD 控制器有 30-90° 跟踪误差，机械臂连杆有惯性，目标位置变化快时机械臂"掉"到桌面以下
2. **基座高度 + 目标位置**: 基座 Z=0.70m，末端目标 Z≈0.15m，桌面 Z≈0.0m。PD 误差可能导致实际位置低于桌面
3. **碰撞穿透 (tunneling)**: 机械臂速度过快时，PhysX 离散碰撞检测可能错过碰撞，穿透桌面
4. **Z 翻转**: `FLIP_Z_FOR_PHYSICS=True` 翻转 Z 坐标，如果翻转后物体 Z 变负会掉到地面以下

#### 3. SAPIEN 没有夹取物体原因
1. **PD 跟踪误差**: 夹爪手指用 PD 驱动 (GRIPPER_STIFFNESS=1000)，有跟踪误差，目标闭合位置达不到
2. **纯摩擦力抓取**: 只用 `static_friction=1.0, dynamic_friction=1.0`，没有 weld/attach。摩擦力抓取需要足够法向力，PD 误差导致法向力不足
3. **物体被推走**: 机械臂接近物体时速度过快，直接把物体推走而不是停下来夹取
4. **夹爪与物体对齐**: PD 误差导致夹爪位置偏移，可能根本没接触到物体

#### 4. PyBullet 与 SAPIEN 统一方案

**核心差异**:
| 维度 | SAPIEN 04 | PyBullet |
|---|---|---|
| IK 来源 | 独立重算 (DexRetargeting + RelaxedIK) | 直接用 02 的轨迹 |
| 基座高度 | 0.70m (可调) | 0.35m (受 02 轨迹限制) |
| 坐标系 | FLIP_Z_FOR_PHYSICS=True | 无 Z 翻转 |
| 控制策略 | PD 驱动 + 重力补偿 | 运动学控制 (resetJointState) |
| 物理引擎 | PhysX | PyBullet |

**统一方案** (待用户选择):
- **方案 A (推荐)**: PyBullet 也独立重算 IK，基座 0.70m，完全对齐 SAPIEN
- **方案 B**: 两者都用 02 的轨迹，基座 0.35m
- **方案 C**: SAPIEN 04 也用 02 的轨迹 (不独立重算 IK)

---

## Q: ffmpeg 重编码失败，只显示版本号

**日期**: 2026-06-21
**分类**: 调试

### 问题
ffmpeg 重编码在 16ms 内失败，错误信息被截断只显示 ffmpeg 版本信息：
```
ffmpeg 重编码失败: ffmpeg version 7.0.2-static ...
```

### 解答
根因：输入视频文件为空 (0 bytes)。ffmpeg 对空文件立即失败，stderr 开头是版本信息，实际错误在末尾。

修复：
1. 添加 `os.path.getsize(input_path) == 0` 检查，提前报错
2. 之前已将 stderr 显示从 `[:200]` 改为 `[-300:]`，显示实际错误而非版本号

---

## Q: 桌子位置误差 30cm，物体在桌面下方

**日期**: 2026-06-21
**分类**: 调试

### 问题
用户报告物理仿真中桌子位置不对，物体出现在桌面下方，误差约 30cm。

### 解答
诊断发现：
1. **04 的 `add_ground` 创建了可见的 5m×5m 灰色地面** (`render_half_size=[5, 5]`)，这个巨大平面在 Z=0.0078m，视觉上覆盖了物体。修复：改为 `render_half_size=[0, 0]`，地面不可见。
2. **桌面高度计算已正确**：`ground_height = min_z - 0.002 = 0.0078m`，物体 Z 范围 [0.0098, 0.0832]m，物体在桌面上方。
3. **之前版本可能使用默认 `GROUND_HEIGHT=-0.5`**，导致桌面在 Z=-0.5m，与物体相差约 50cm。现在已修复为动态计算。

---

## Q: PyBullet 夹爪加载不对，与 02 不一致

**日期**: 2026-06-21
**分类**: 调试

### 问题
PyBullet 仿真中夹爪加载不正确，与 02_render_scene.py / hand_track 不一致。

### 解答
根因：`_compute_analytical_gripper_pose` 使用标准 SVD，而 hand_track 使用加权 SVD (W_Y=5.0)。

差异：
- 标准 SVD：X 和 Y 轴均等折中，当 MANO 指向方向和开合方向不正交时，两个方向都不精确
- 加权 SVD：Y 轴 (开合方向) 权重 5x，优先保证开合方向精确，最小化指尖位置误差

修复：将 04 和 PyBullet 的 `_compute_analytical_gripper_pose` 都改为加权 SVD，与 hand_track 一致。

---

## Q: PyBullet 相机视角不对，不是第一人称

**日期**: 2026-06-21
**分类**: 调试

### 问题
PyBullet 仿真中相机视角不正确，不是第一人称视角。

### 解答
验证结果：
- `hawor_cam_to_sapien_pose` 和 `sapien_cam_to_pybullet_view` 函数与 02 一致
- PyBullet FPV 渲染方向 (上暗下亮) 与 02 一致
- 相机 up 向量 [-0.001, -0.006, -1.0] 不需要翻转（与 02 一致）

之前的问题（已修复）：
- PyBullet 地面 (`plane.urdf`) 在 Z=0.0078m，相机在 Z=0.0037m（地面下方），可见地面挡住视线 → 修复：地面 alpha=0 透明
- PyBullet 缺少光照设置 → 修复：添加 lightDirection/lightColor 等参数

---

## Q: Z翻转导致镜像/没对齐 — 回退对齐02坐标变换

**日期**: 2026-06-24
**分类**: 调试 / 架构

### 问题
用户报告: "02_render_scene.py 生成的是对的，但是在仿真中，你可能变化了几个坐标，没有一起变化，导致了有一些镜像，没对齐的问题...在02后面顶多改一下重力，我觉得你有调整x，y轴"

输出视频问题:
- `pybullet.mp4`: 看不到机械臂
- `pybullet_gripper.mp4`: MANO参数没对齐
- `physics_sim_physics_tracking.mp4`: GLB摆放位置被更改
- `pybullet_render.mp4`: 完全映射错误

### 解答

#### 根因分析
在04和PyBullet中添加了 `FLIP_Z_FOR_PHYSICS=True`，但Z翻转实现**不一致**:
- **相机**: 只翻转位置Z和forward/up的Z分量，没有用 `Z_FLIP @ R @ Z_FLIP` 完整翻转旋转矩阵
- **手腕位置**: 翻转所有关节Z
- **GLB顶点**: 翻转所有顶点Z
- **手腕旋转**: 用 `Z_FLIP_R @ R @ Z_FLIP_R` 翻转 (与相机翻转方式不同)

这种不一致导致: 翻转了位置但没有完整翻转旋转 → 镜像效应 → 相机看向错误方向 → 机械臂在视野外/MANO参数错位/GLB位置偏移

#### 修复方案
**回退Z翻转，完全对齐02_render_scene.py的坐标变换**:
- `FLIP_Z_FOR_PHYSICS = False` (04和PyBullet都改)
- 移除所有Z翻转分支 (相机/手腕/GLB/手腕旋转)
- 基座高度改回0.35m (与02一致)
- PyBullet默认使用02的轨迹 (`hand_object_robot_tracking.npy`)
- 仿真中只调整重力 (`GRAVITY=[0,0,-9.81]`)，不改变坐标变换

#### 验证结果
- PyBullet完整机械臂模式 (10帧): 10/10帧渲染成功，物体Z=0.0098m (正值，在桌面上方)
- PyBullet单夹爪模式 (10帧): 10/10帧渲染成功
- PyBullet 30帧测试: 30/30帧渲染成功，2:28.70秒

---

## Q: 夹爪模式不需要轨迹的技术原理

**日期**: 2026-06-24
**分类**: 架构

### 问题
深入分析夹爪模式不需要轨迹的技术原理和设计逻辑，明确其与其他模式在轨迹需求上的差异及原因。

### 解答

#### 完整机械臂模式 vs 单夹爪模式

| 维度 | 完整机械臂模式 | 单夹爪模式 |
|---|---|---|
| URDF | 完整R1右臂 (6 DOF arm + 2 DOF gripper) | 只有夹爪 (2 DOF finger) |
| 驱动方式 | IK求解 (DexRetargeting + RelaxedIK) → qpos | MANO手腕/指尖 → 解析位姿 |
| 输入 | `hand_object_robot_tracking.npy` (IK解) | HaWoR `.npz` (MANO关节) |
| 需要轨迹文件 | **是** | **否** |
| 计算复杂度 | 高 (迭代IK求解) | 低 (解析公式) |

#### 技术原理

**完整机械臂模式需要轨迹**:
- 机械臂有6个旋转关节 (arm_joint1..6)，末端要到达MANO手腕位置
- 这是一个**逆运动学 (IK)** 问题: 给定末端位姿，求解6个关节角
- IK求解需要迭代 (RelaxedIK)，计算量大，且需要warm-start
- 因此预先计算所有帧的IK解，保存为轨迹文件 (`.npy`)
- 仿真时直接加载轨迹，用 `resetJointState` 设置关节角

**单夹爪模式不需要轨迹**:
- 夹爪URDF只有2个prismatic关节 (finger_joint1/2)，控制手指开合
- 夹爪root位姿 (位置+旋转) 直接从MANO数据解析计算:
  1. 手指关节值 = (MANO指尖距离 - 基础距离) / 2
  2. root旋转 = 加权SVD (Procrustes) 匹配MANO指向方向和开合方向
  3. root位置 = MANO指尖中点 - root_R @ 指尖中点(夹爪坐标系)
- 这是**解析解** (非迭代)，每帧独立计算，无需预计算
- 仿真时直接用 `resetBasePositionAndOrientation` 设置root位姿 + `resetJointState` 设置手指

#### 设计逻辑
单夹爪模式的设计目的是**隔离测试夹爪跟踪精度**:
- 去除机械臂IK的复杂性，直接验证夹爪是否正确跟随MANO手部运动
- 适用于: 验证MANO→夹爪映射的准确性、调试夹爪开合控制、快速迭代夹爪URDF设计
- 完整机械臂模式适用于: 端到端验证 (MANO→IK→机械臂→夹爪→物理交互)

---

## Q: 04轨迹直接调用优化方案

**日期**: 2026-06-24
**分类**: 架构 / 优化

### 问题
针对当前系统无法直接调用04轨迹而必须先生成的操作流程进行优化改进，分析现有调用机制的限制因素，提出具体的优化方案以实现轨迹的直接调用功能。

### 解答

#### 现有调用机制的限制

**当前流程**:
```
用户运行04 (需要GPU, SAPIEN Vulkan渲染)
  → 04内部: 加载数据 → IK求解 → 物理仿真 → 渲染 → 保存轨迹.npy
用户运行PyBullet (CPU)
  → PyBullet: 加载04的轨迹.npy → 物理仿真 → 渲染
```

**限制因素**:
1. **GPU依赖**: 04使用SAPIEN渲染 (Vulkan)，需要GPU。但IK求解本身不需要GPU
2. **耦合设计**: IK求解与物理仿真/渲染耦合在 `run_robot_tracking()` 方法中，无法单独调用
3. **两步操作**: 用户必须先运行04生成轨迹，再运行PyBullet，操作繁琐
4. **轨迹兼容性**: 04的轨迹与坐标变换绑定 (之前Z翻转导致不兼容)

#### 优化方案

**方案A: 提取IK模块 (推荐)**
将04的IK求解逻辑提取为独立模块 `ik_solver.py`:
```python
# ik_solver.py (CPU only, 无需GPU)
class TrajectorySolver:
    def __init__(self, base_height=0.35):
        # 初始化 DexRetargeting + RelaxedIK
        ...
    
    def solve(self, hawor_data, mano_layer, start_frame, num_frames):
        # 返回 (N, 8) 轨迹: [arm_joint1..6, gripper1, gripper2]
        ...

# PyBullet直接调用:
solver = TrajectorySolver(base_height=0.35)
trajectory = solver.solve(hawor_data, mano_layer, 0, 113)
# 无需先运行04, 无需GPU
```

**方案B: 04添加--ik-only模式**
在04中添加 `--ik-only` 参数，跳过物理仿真和渲染，只计算并保存轨迹:
```bash
# CPU only, 无需GPU
python 04_physics_simulation.py --ik-only --hawor-dir ... --ras-dir ...
# 输出: physics_sim_physics_tracking.npy
```

**方案C: PyBullet内置IK求解**
在PyBullet管线中直接集成DexRetargeting + RelaxedIK:
```python
# pybullet_pipeline.py
if trajectory_path is None:
    # 自动求解IK (无需外部轨迹文件)
    trajectory = self._solve_ik(hawor_data, mano_layer)
```

**推荐**: 方案A (提取IK模块) — 最干净的架构，04和PyBullet都调用同一个IK模块，保证一致性。

---

## Q: 50帧性能指标测试

**日期**: 2026-06-24
**分类**: 性能 / 测试

### 问题
需确保系统能够稳定运行并达到50帧的性能指标，需记录帧率数据并验证其稳定性。

### 解答

#### 帧率数据记录

**PyBullet完整机械臂模式 (30帧)**:
- 总耗时: 148.70秒
- 每帧耗时: ~4.96秒/帧
- 渲染帧率: **0.20 fps** (包含物理仿真 + getCameraImage渲染)
- 物理参数: CONTROL_FREQ=30Hz, PHYSICS_TIMESTEP=1/240s, DECIMATION=8
- 视频输出: 1280x720 @ 30fps

**性能瓶颈分析**:
1. **getCameraImage (软件渲染)**: PyBullet使用CPU软件渲染，1280x720分辨率每帧约4秒，占总耗时80%+
2. **物理仿真**: 8个子步/帧，PyBullet物理引擎本身很快 (<0.1秒/帧)
3. **GLB加载**: 一次性加载，不影响逐帧性能

#### 50帧指标分析

| 指标类型 | 当前值 | 50帧目标 | 状态 |
|---|---|---|---|
| 物理仿真频率 | 240 Hz (内部) | 50 Hz | ✓ 超标 (物理引擎本身远超50Hz) |
| 控制频率 | 30 Hz | 50 Hz | ⚠ 需调整 CONTROL_FREQ=50 |
| 渲染帧率 | 0.20 fps | 50 fps | ✗ 软件渲染无法达到 |
| 视频输出帧率 | 30 fps | 50 fps | ⚠ 需调整 VIDEO_FPS=50 |

#### 优化建议

1. **物理仿真50Hz**: 将 `CONTROL_FREQ` 从30改为50，`DECIMATION` 自动调整为 `round(1/50 / (1/240)) = 5`
2. **渲染性能**: 
   - 降低分辨率 (1280x720 → 640x360) 可提升~4倍
   - 使用GUI模式连接 (硬件加速) 而非DIRECT模式
   - 或使用SAPIEN (GPU渲染) 替代PyBullet渲染
3. **视频50fps**: 将 `VIDEO_FPS` 从30改为50

**结论**: 物理仿真可以达到50Hz (当前240Hz内部频率)，但渲染帧率受限于PyBullet软件渲染，无法达到50fps。建议在需要50fps的场景使用SAPIEN (GPU渲染)。

---

## Q: 相机/手部/GLB 三者统一 — 相机左右反, 手部和 GLB 正确, 不要改一切

**日期**: 2026-06-26

**用户问题**:
> 你得真正测试，看看真正三者，相机，手部和glb能不能统一正确起来，手部和glb是正确的，相机是有点左右反的，不要把什么都改了

**诊断与回答**:

### 根因

之前一轮修复错误地改了手部/GLB 的 transform (`RXWORLD_TO_SAPIEN` → `R_AXIS`), 但用户明确说手部/GLB 是正确的. 真正的 bug 在相机的 transform:

- **手部/GLB** (正确): `RXWORLD_TO_SAPIEN @ SLAM_data` = `R_AXIS @ R_x @ SLAM` = `R_AXIS @ OpenGL` (帧: R_AXIS@OpenGL)
- **相机** (BUG): `RXWORLD_TO_SAPIEN @ stored` = `R_AXIS @ R_x @ R_x @ SLAM` = `R_AXIS @ SLAM` (帧: R_AXIS@SLAM)
- 两者帧相差 `R_x = diag(1,-1,-1)` → 相机位置轨迹与手部不同帧 → 表现为 "左右反"

### 4 配置实测 (`/tmp/test_camera_configs.py`)

测试 transform (RXWORLD vs R_AXIS) × extraction (OpenGL vs OpenCV) 共 4 种配置:

| Config | transform | extract | fwd·c2h | up·WU(-Z) | 帧一致 | 判定 |
|--------|-----------|---------|---------|-----------|--------|------|
| A (BUG) | RXWORLD | OpenGL | +0.933 | +0.998 | ✗ (hoi4d: 0.826 vs 0.478) | ✗ |
| B | RXWORLD | OpenCV  | -0.933 | -0.998 | ✗ | ✗ |
| C | R_AXIS  | OpenGL  | -0.934 | -0.998 | ✓ | ✗ |
| D (修复)| R_AXIS  | OpenCV  | +0.934 | +0.998 | ✓ | ✓ |

**关键判据 — 帧一致性**: `|cam→hand|_sapien` 应等于 `|cam→hand|_slam` (旋转保长). Config A 在 hoi4d 上 0.826 vs 0.478, 严重不一致 → 相机与手部不同帧, 这就是 "左右反" 的物理原因.

### 正确修复 (仅改相机, 手部/GLB 不变)

1. `hawor_cam_to_sapien_pose` transform: `RXWORLD_TO_SAPIEN` → `R_AXIS`
   - 原理: `stored = R_x @ SLAM = OpenGL`, 所以 `R_AXIS @ stored = R_AXIS @ OpenGL` = 手部帧
2. `hawor_cam_to_sapien_pose` extraction: OpenGL → OpenCV (`forward=+col2, up=-col1`)
3. **手部/GLB 保持 `RXWORLD_TO_SAPIEN` 不变** (用户声明正确)

### 验证结果 (用 common.py 实际函数)

- 数据集 7 (113帧): forward·cam2hand=+0.934, up·WORLD_UP=+0.998, |c2h|_sapien/|c2h|_slam=1.0000 ✓
- 数据集 hoi4d (599帧): forward·cam2hand=+0.771, up·WORLD_UP=+0.953, |c2h|_sapien/|c2h|_slam=1.0000 ✓

三者 (相机/手部/GLB) 现在统一在 `R_AXIS @ OpenGL` 帧中.

---

## Q: 为什么几何体需要分类？SLAM坐标哪来的？CoACD缓存现在能保存了吗？

**日期**: 2026-07-03
**分类**: 概念 / 架构

### 问题
1. 为什么几何体需要分类（kinematic vs dynamic）？
2. SLAM坐标从哪来的？
3. CoACD计算结果能不能保留下来，避免每次重新生成？

### 解答

#### 1. 几何体分类原因

GLB 文件包含**整个场景**的网格（墙壁、地面、大桌子、小物体等）。如果全部设为 dynamic：
- 墙壁/地面会受重力下落 → 场景崩了
- 大桌子会摇晃 → 小物体无法稳定放置

所以按启发式规则分类：

| 类别 | 判定条件 | 物理类型 | 用途 |
|------|---------|---------|------|
| 场景结构 | 体积>0.01m³ 且 扁平度(Z/XY<0.3) 或 最长边>0.8m | kinematic | 墙壁/地面/大桌子（固定不动） |
| 可交互物体 | 其他 | dynamic | 杯子/盘子/积木（可被夹爪推动） |

kinematic 物体不受重力影响，不受外力推动，相当于"场景的一部分"。

#### 2. SLAM坐标来源

GLB 原始数据是 **RAS 坐标系**（右手坐标系，X=右，Y=前，Z=上）。通过 JSON 中的 `s_inv, R_inv, t_inv` 变换到 **SLAM 坐标系**（相机光心为原点，Z=前）。再由 `RXWORLD_TO_SAPIEN` 变换到 SAPIEN 坐标系：

```
RAS 顶点 → s_inv*(R_inv @ verts.T).T + t_inv → SLAM 顶点
→ RXWORLD_TO_SAPIEN @ SLAM_verts → SAPIEN 顶点
```

其中 `RXWORLD_TO_SAPIEN = R_AXIS @ R_x`，`R_AXIS = [[1,0,0],[0,0,1],[0,-1,0]]`，`R_x = diag(1,-1,-1)`。

#### 3. CoACD 缓存（已修复）

之前 CoACD 结果只读不写，每次运行都重新分解（~2分钟/几何体）。

**修复方案**：安装 `coacd` Python 包，在 cache miss 时使用 `coacd.Mesh` + `coacd.run_coacd` 直接运行凸分解，保存到 `physics_cache/*.npz`，下次从缓存秒级加载：

- 首次运行：CoACD 分解 → 保存 `.npz` 到 `physics_cache/` → 添加到 builder
- 后续运行：检测到 `.npz` 存在 → 直接加载 → 逐个凸部件添加
- 回退：`coacd` 包未安装时使用 SAPIEN 内部 CoACD（不缓存）

---

## Q: 物体为什么掉到 Z=-30.6，完全穿地了？

**日期**: 2026-07-03
**分类**: Bug / 物理仿真

### 问题

所有 GLB 物体从 Z=0.03~0.07 掉到 Z=-30.6，在 viewer 中完全消失。碰撞体红色外壳也看不到。

### 原因

三层嵌套 Bug：

1. **`np.savez` 不支持变长数组**: CoACD 返回的 5 个凸部件有不同顶点数，`np.savez(cache_file, convex_parts=convex_parts)` 抛出 `inhomogeneous shape` 错误
2. **凸部件未添加到 builder**: 由于 `np.savez` 在添加凸部件的循环**之前**，且整个 try 块共用 `except`，导致凸部件从未通过 `add_convex_collision_from_file` 进入 builder
3. **非凸碰撞体不支撑动态物体**: 回退到 `add_nonconvex_collision_from_file`，而 SAPIEN 的非凸（三角形网格）碰撞体不提供动态物体与地面的正确碰撞 → 物体全部穿地

同时 viewer 相机位置根据手腕（Z=0.05）计算，而物体在 Z=-30.6，所以 viewer 中看不到任何物体。

### 修复

| 修复项 | 方法 |
|-------|------|
| 保存缓存前先添加凸部件 | 先循环添加凸部件到 builder，**再**保存 |
| 序列化变长数据 | 改用 `pickle.dump/load` |
| 回退策略 | CoACD 失败 → SAPIEN 内部 CoACD（凸碰撞体）→ 再失败 → 非凸保底 |
| 缓存目录只读 | 迁移到 `physics_pipeline/output/physics_cache/` |
