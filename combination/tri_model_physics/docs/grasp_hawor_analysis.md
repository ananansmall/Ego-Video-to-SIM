# grasp_hawor.py 分析文档 (整合版)

> 整合自: grasp_hawor_flow.md + grasp_redesign.md + questions.md (2026-06-27)
> 核心目标: 给定 HaWoR 手部重建 + RAS 场景重建 (GLB), 用 R1 机器人 URDF 在 SAPIEN 中复刻抓取 GLB 物体的动作, 并通过参数级验证 (物体提升/接触检测/综合抓取判定/轨迹跟踪误差).

---

## 1. 架构核心: 夹爪与机器人任务统一

**用户要求 (2026-06-27)**: "夹爪和整个机器人的任务是一样的, 除了加载和映射, 两者应该是一样的完成任务. 我认为你可以完全先测试夹爪, 这个很关键."

### 1.1 两种模式的差异 (仅加载 + 映射)

| 维度 | `gripper_only` | `full_robot` |
|------|----------------|--------------|
| **加载 URDF** | 纯夹爪 URDF (gripper_base_link → gripper_link → finger1/2) | r1_v2_1_0.urdf (整个机器人, 臂关节 fixed→revolute) |
| **映射方式** | 解析 SVD Procrustes: MANO 手腕+指尖 → root_pos + joint1/joint2 | DexRetargeting + RelaxedIK: MANO 指尖 → retarget_qpos → IK → arm_target |
| **根运动** | kinematic (set_root_pose 直接移动) | PD 驱动 (set_drive_target) |

### 1.2 抓取逻辑统一 (两者共享)

两种模式的 `_step_*` 函数都按相同顺序处理三种 `--grasp-mode`:

```python
if self.grasp_mode == "hybrid":      # MANO 定位 + 接触力控状态机 (6 相位)
    controller.update(...)            # → adapted_target
elif self.grasp_mode == "adaptive":  # 旧版 MANO 意图 + 简单状态机
    controller.update(...)            # → adapted_target
elif self.grasp_mode == "mano":      # 纯 MANO 重放 + 接触维持夹紧
    # 接触前: 纯重放 MANO
    # 接触后: 维持 qpos0 - CLAMP_OFFSET_MAX * max(curl, CLAMP_CURL_FLOOR)
    # MANO 张开 (curl < RELEASE_TRIGGER_CURL) → 释放
```

**关键**: `mano` 接触维持逻辑在 `_step_gripper_only` (L2025-2063) 和 `_step_full_robot` (L1903-1935) 中**完全一致**, 只是变量名不同 (gripper_only 用 `qpos[gripper_idx1]`, full_robot 用 `gripper_val`).

### 1.3 测试策略: 先夹爪后机器人

用户明确: "我认为你可以完全先测试夹爪, 这个很关键".

- **第一阶段 (当前)**: `gripper_only` 模式, 验证夹爪能真正夹住物体 + 轨迹跟踪误差达标
- **第二阶段 (后续)**: `full_robot` 模式, 复用相同的抓取逻辑, 只是映射改为 IK

`full_robot` 的 `mano` 分支已就位 (L1903-1935), 架构统一完成, 等 gripper_only 验证通过后可直接应用.

---

## 2. 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    GraspSimulator.run() 主循环                   │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: _align_scene()                                         │
│    01_align_scene.compute_and_save_transform_params             │
│    对齐 RAS GLB → HaWoR 坐标系 (s_inv, R_inv, t_inv)            │
│  Step 2: 加载 HaWoR 数据 + MANO FK                              │
│    load_hawor_data → pred_trans/rot/hand_pose/betas/valid       │
│    MANOLayer(side, betas) → joints_sapien (wrist + 21 joints)  │
│  Step 3: setup_physics_scene + load_glb_with_physics            │
│    7 个 dynamic 物体, 统一 OBJECT_DENSITY + OBJECT_MIN_MASS     │
│  Step 4: _compute_optimal_base (full_robot) / 首帧手腕 (gripper)│
│  Step 5: setup_robot (加载 URDF + PD 参数 stiffness=1000,d=200) │
│  Step 6: _init_retargeting + _init_ik (仅 full_robot)            │
│  Step 7-8: 相机设置 + 视频录制 (按 --views)                     │
│  Step 9: Warmup (smoothstep 过渡到起始位姿)                      │
│  Step 10: 主循环 (num_frames 帧)                                 │
│    ├─ MANO FK → joints_sapien                                   │
│    ├─ full_robot: _step_full_robot (DexRetargeting + IK)         │
│    │  OR gripper_only: _step_gripper_only (SVD Procrustes)      │
│    │  → 两者都调 hybrid/adaptive/mano 抓取逻辑 (统一)           │
│    ├─ physics_step (PD drive + 重力补偿, DECIMATION=8)           │
│    ├─ 记录 qpos/obj_pos/gripper_pos/track_log                   │
│    └─ 渲染 (try/except GPU 降级)                                │
│  Step 11: _verify_results (参数级验证 + 轨迹跟踪误差)            │
│  Step 12: 保存 qpos.npy + verify.json + 视频                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 关键模块

### 3.1 URDF 准备 (L151-287)

- **`prepare_full_robot_urdf`**: r1_v2_1_0.urdf → 臂关节 `fixed → revolute` (`re.DOTALL` 匹配跨行), 夹爪 `fixed → prismatic`
- **`prepare_gripper_only_urdf`**: 生成纯夹爪 URDF, `gripper_base_link → gripper_link → finger_link1/2` (prismatic, axis 相反)

### 3.2 场景与物体加载 (L293-590)

- **`setup_physics_scene`**: 尝试 `sapien.Scene()`, 失败降级 `PhysxCpuSystem`
- **`load_glb_with_physics`**:
  - 顶点变换: `p_hawor = s_inv * R_inv @ p_ras + t_inv`, `p_sapien = RXWORLD_TO_SAPIEN @ p_hawor`
  - 每个 geometry 创建 dynamic actor (凸包碰撞体)
  - **统一基础惯性变量** (用户: "基础的惯性变量"): `obj_mass = max(volume * OBJECT_DENSITY, OBJECT_MIN_MASS)`
  - **angular_damping=50.0** (用户: "夹爪碰一下把盘子弄翻"): 大幅提高扁平物体角阻尼, 抑制 kinematic 根高速冲击的翻转力矩 (5.0 不足以抑制, 提到 50.0). 不加 linear_damping (避免影响提升, 历史教训 glb_5 lift=-26cm). 物体 restitution=0.0 (零弹性, 碰撞不反弹).
  - **返回 obj_info 字典** (第十一轮): `{name: {color, bbox_size, bbox_min, bbox_max, volume, flatness, body_type}}`, 用于颜色/几何识别范式

### 3.2b 颜色+几何识别范式 (第十一轮, L1479-1575)

用户: "我需要夹住的是那个粉色的东西, 放到碗里面; 注意你的这个需要形成一个范式, 在不同的文件夹里面都可以使用上"

**`find_pink_object(obj_info)` — 颜色识别范式**:
- 粉色判定 (RGB [0,1]): `R>0.4 and G<0.35 and 0.15<B<0.6 and B>G`
- 评分: `pinkness = R * (1-G) * (B-G)`, 取最高
- 测试: glb_1 (0.58, 0.06, 0.33) ✓ (排除 glb_4 橙色 B<G, glb_3 蓝灰 R<G)

**`find_bowl(obj_info, exclude_names)` — 几何识别范式**:
- 碗判定: `dynamic + volume>1e-4 + flatness<0.55` (大体积且扁平容器)
- 评分: `bowlness = volume * (1 - flatness)`, 取最高
- 支持 exclude_names 排除已锁定为抓取目标的物体 (避免粉色物体既是抓取目标又是碗)
- 测试: glb_3 (vol=0.0002, flat=0.446) ✓

### 3.3 物理参数 (统一基础变量, L123-137)

```python
# 基础惯性变量 (用户: "你的仿真没有一个基础的惯性变量吗")
OBJECT_DENSITY = 1000.0  # 所有物体统一密度 (kg/m³), 对齐水密度
OBJECT_MIN_MASS = 0.15   # 物体质量下限 (kg), 防止轻物被 kinematic 根弹飞

# PD 参数 (对齐 GalaxeaManipSim)
JOINT_STIFFNESS = 1000.0; JOINT_DAMPING = 200.0
GRIPPER_STIFFNESS = 1000.0; GRIPPER_DAMPING = 200.0

# 物理仿真
PHYSICS_TIMESTEP = 1 / 240.0; CONTROL_FREQ = 30; DECIMATION = 8

# 夹爪
GRIPPER_FRICTION = 1.0; GRIPPER_INIT_OPEN = 0.04; GRIPPER_MAX_OPEN = 0.05
MAX_ROOT_STEP = 0.008  # 根速度限制: 每帧 ≤ 0.8cm (第五轮: 0.015→0.008, 仅 0.008 让盘子不翻; 根误差增大是物理稳定代价)
```

**说明**: 所有物体用同一 `OBJECT_DENSITY` (不是每个物体不同), 质量公式 `max(volume * OBJECT_DENSITY, OBJECT_MIN_MASS)` 统一应用. 盘子等扁平物体体积小, 走 `OBJECT_MIN_MASS` 下限 (0.15kg), 大物体按密度计算.

### 3.4 gripper_only 单帧步进 (`_step_gripper_only`, L1944-2075)

**用户**: "夹爪运动是要物理和真实输出的误差来判断准不准确".

```
1. compute_analytical_gripper_pose (加权 SVD Procrustes, W_Y=5.0)
   → root_pos, root_R, joint1, joint2 (限幅前 = MANO 期望)
2. expected_root = root_pos.copy()           # 记录 MANO 期望 (限幅前)
3. 根速度限幅: root_pos 变化 ≤ MAX_ROOT_STEP  # 防止 kinematic 根冲击
4. set_root_pose + set_qpos (立即设手指位置, 对齐 04)
5. expected_j1/j2 = joint1/joint (调整前 = MANO 期望)
6. 抓取逻辑: hybrid / adaptive / mano (统一)
7. 记录轨迹跟踪误差:
   self._last_track = {
     'root_err_mm': |expected_root - actual_root| * 1000,
     'j1_err_mm':   |expected_j1 - actual_j1| * 1000,
     'j2_err_mm':   |expected_j2 - actual_j2| * 1000,
   }
```

**三重设置** (对齐 04_physics_simulation.py L2436-2458):
1. `set_root_pose` (kinematic 根跟随手腕)
2. `set_qpos` (立即设手指位置产生接触)
3. `set_drive_target` (PD 保持, 在 `physics_step` 中)

### 3.5 full_robot 单帧步进 (`_step_full_robot`, L1847-1942)

```
1. DexRetargeting.retarget(ref_value) → retarget_qpos (含 gripper + arm)
2. 提取 gripper_val (从 retarget_qpos)
3. _get_gripper_pose_from_retargeting → gripper_pos_fk, R_ee_world_fk
4. 抓取逻辑: hybrid / adaptive / mano (统一, 与 gripper_only 一致)
5. IK: solve_position(ik_target_b, ee_quat_b) → arm_targets
6. 返回 (arm_targets, gripper_vals)
```

### 3.6 mano 接触维持 (两模式共享逻辑)

```
接触前: 纯重放 MANO (不改 qpos/gripper_val)
检测: 双指接触 + MANO curl > GRASP_TRIGGER_CURL → 进入维持
接触后: 维持 qpos0 - CLAMP_OFFSET_MAX * max(curl, CLAMP_CURL_FLOOR)
        (只夹紧不松开, 防 MANO 抖动导致物体滑落)
释放: MANO 张开 (curl < RELEASE_TRIGGER_CURL) → 回到纯重放
```

- `_step_gripper_only` (L2025-2063): 用 `qpos[gripper_idx1]`, `_mano_state`
- `_step_full_robot` (L1903-1935): 用 `gripper_val`, `_mano_state_fr[s]` (per-side)

### 3.7 HybridGraspController (hybrid 模式, L1164-1460)

6 相位状态机: `APPROACH → CLOSE → FORCE_CONTROL → LIFT → HOLD → RELEASE`

- **MANO 参数驱动**: curl 决定力度 (curl=0.5→2.5N, curl=1.0→5.0N), 腕部 z 变化决定提升/释放
- **力控闭环**: 接触前位置控制 (跟随 MANO), 接触后力控 (MANO curl → 目标力)
- **固定夹紧** (用户: "接触肯定还要力来控制啊"): `qpos_at_contact - CLAMP_OFFSET_MAX * max(curl, CLAMP_CURL_FLOOR)` (不是持续闭合, 防止物体被挤出飞出)

### 3.7b Pick-and-Place 7 阶段范式 (第十一轮, L2223-2377)

用户: "夹住粉色的东西, 放到碗里面"

`_compute_grasp_demo_target` 在原 4 阶段 (APPROACH→DESCEND→CLOSE→LIFT) 基础上, 当检测到碗时启用 7 阶段:

| 阶段 | 帧占比 | 行为 | 速度控制 |
|------|--------|------|---------|
| APPROACH | 0-10% | EE → approach_pos (grasp+8cm) | teleport + ff |
| DESCEND | 10-25% | EE → grasp_pos, gripper open | teleport + ff |
| CLOSE | 25-40% | gripper 立即设为 0, PD 收敛 | teleport + ff |
| LIFT | 40-55% | EE → lift_pos (grasp+12cm) | **velocity + P** (不 teleport) |
| TRANSPORT | 55-75% | EE 水平移动到 bowl_release_pos | **velocity + P** (不 teleport) |
| RELEASE | 75-85% | 在碗上方打开夹爪 | teleport + ff |
| RETREAT | 85-100% | 后退到 bowl_release_pos + 15cm | teleport + ff |

关键: LIFT 和 TRANSPORT 都不能 teleport (会破坏接触). 用 `velocity = ff + (target-actual)*KP` (KP=CONTROL_FREQ/4) 控制根平滑移动, 维持接触摩擦.

### 3.7c MANO+offset 中和态 (第十二轮, L2224-2360)

用户: "我觉得你需要有一个中和, mano参数为主体, 你可以平移轨迹, 但不能离开轨迹, 偏移轨迹那么多"

**设计目标**: 在 "纯 MANO 重放" (轨迹形状好, 但定位不准) 和 "纯预设路径" (定位准, 但脱离 MANO) 之间找中和态.

**核心公式**:
```
EE[f] = mano_root_pos[f] + offset   (常量平移, 保持 MANO 运动形状)
offset = target_grasp_pos - mano_root_pos[f_grasp]   (在 f_grasp 处对齐)
f_grasp = argmin_f |mano_root_pos[f] - target_pos|   (MANO 最接近目标的帧)
```

**关键改进**: 阶段判定基于 f_grasp (MANO 实际到达目标时刻), 不是固定帧占比.
- 上一轮问题: CLOSE 在 F28 (固定 25%), 但 MANO 在 F19 最接近目标, F28 时已上升 → 抓不到
- 修复: CLOSE 起点改为 f_grasp, 让抓取发生在 MANO 真正到达目标高度时

**阶段行为**:
| 阶段 | EE 行为 | gripper_val | 速度控制 |
|------|---------|-------------|---------|
| APPROACH | MANO+offset | 跟随 MANO | teleport |
| DESCEND | MANO+offset | **强制打开** (避免推开物体) | teleport |
| CLOSE | **保持 grasp_pos** (MANO 此时会上升, 不跟随) | 强制闭合 (0) | teleport |
| LIFT | smoothstep: grasp_pos → MANO+offset (整个 LIFT 渐进) | 强制闭合 | velocity+P |
| TRANSPORT | MANO+offset + **Z-floor** | 强制闭合 | velocity+P |
| RELEASE | MANO+offset + **Z-floor** | 强制打开 | teleport |
| RETREAT | MANO+offset + **Z-floor** | 跟随 MANO | teleport |

**Z-floor 碗保护** (Test 4-6 关键修复):
```python
if has_bowl and phase in ("TRANSPORT", "RELEASE", "RETREAT"):
    bowl_safe_z = bowl_z + 0.15 + 0.037   # 碗上方 15cm + 手指偏移
    gripper_pos[2] = max(gripper_pos[2], bowl_safe_z)
```
- 根因: MANO+offset 轨迹经过碗位置, 闭合夹爪撞碗 → 碗飞 207cm
- 根因: RETREAT 时 MANO 下降, 手指 z=0.026 低于碗心 0.027 → 碗被推 16.75cm
- 修复: 扩展 Z-floor 到 RETREAT → 碗 xy_drift 0.02cm (完全稳定)

**预计算优化** (避免每帧重复 SVD):
- 启动时一次性计算所有帧的 `mano_gripper_traj[side] = {"pos": [], "j1": [], "j2": []}`
- 主循环中 `_compute_mano_neutral_target` 直接查表

**Test 6 验证结果**:
- glb_1 (粉) 真正夹住: lift=17.07cm, 跟随=112/113 帧 ✓
- glb_3 (碗) 完全稳定: xy_drift=0.02cm ✓
- glb_1 距碗心: 5.46cm (MANO 轨迹在 RELEASE 时 +X 偏 4.2cm, 落地后弹开 1.3cm)
- 6 次测试迭代: glb_3 xy_drift 41→10→50→207→16.75→0.02cm (持续改进)

### 3.8 轨迹跟踪误差验证 (`_verify_results`, L2823-2852)

**用户 (2026-06-27)**: "夹爪运动是要物理和真实输出的误差来判断准不准确, 而不是只有一个开合的判断".

每帧记录:
- `expected_root`: 限幅前 MANO 期望根位置
- `expected_j1/j2`: 抓取调整前 MANO 期望手指
- `actual_root/j1/j2`: 限幅 + 抓取调整后 SAPIEN 实际输出

统计输出 (verify.json `track_error`):
```json
{
  "n_frames": 113,
  "root_err_mean_mm": 0.80,    // < 10mm = 准确
  "root_err_max_mm": 8.80,
  "finger1_err_mean_mm": 3.74, // < 5mm = 准确
  "finger1_err_max_mm": 4.95,
  "finger2_err_mean_mm": 3.74,
  "finger2_err_max_mm": 4.95
}
```

### 3.9 综合抓取判定 (`_evaluate_grasp_quality`, L2029-2138)

三项都满足才算 "真正夹住":
1. **接触**: 连续 >= 10 帧 n_contacts >= 2
2. **跟随**: 物体相对最近夹爪位置变化 < 5cm 持续 10 帧
3. **提升**: 物体 z 提升量 > 5cm

---

## 4. 数据流

```
输入:
  --hawor-dir  → hawor_results_*.npz (pred_trans/rot/hand_pose/betas/valid)
  --ras-dir    → final_scene.glb + extrinsics/*.txt
  --mode       → full_robot / gripper_only
  --side       → left / right / both
  --grasp-mode → hybrid (默认) / mano / adaptive
  --views      → cam / god / both

处理:
  01_align_scene → transform_params.npz
  HaWoR npz → MANO FK → joints_sapien
  GLB → load_glb_with_physics → obj_actors (统一 OBJECT_DENSITY)

  gripper_only: compute_analytical_gripper_pose → root_pos + joint1/joint2
  full_robot:   DexRetargeting + RelaxedIK → arm_target + gripper_val

  抓取逻辑 (统一): hybrid/adaptive/mano → adapted_target

  physics_step: set_drive_target → compute_passive_force(gravity=True) → scene.step (×8)

输出:
  output/<mode>_<side>/
    cam_view_<mode>_<side>.mp4   (第一视角视频, HaWoR 相机轨迹, --views both/cam)
    god_view_<mode>_<side>.mp4   (上帝视角视频, 相机在夹爪前方看手指, --views both/god)
    grasp_<mode>_<side>_qpos.npy (关节角轨迹)
    grasp_<mode>_<side>_verify.json (验证 + 综合抓取判定 + 轨迹跟踪误差)
    grasp.log
```

---

## 5. 使用示例

```bash
# gripper_only + mano (推荐, 先测试夹爪)
python grasp_hawor.py --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god --grasp-mode mano

# full_robot + hybrid (后续, 复用相同抓取逻辑)
python grasp_hawor.py --mode full_robot --side both \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views both --grasp-mode hybrid

# CPU 无头模式 (GPU 损坏时)
VK_ICD_FILENAMES= python grasp_hawor.py --mode gripper_only --side left \
    --hawor-dir /home/an/data/hawor/7 \
    --ras-dir /home/an/data/ras/my_7mp4_result \
    --views god --grasp-mode mano
```

---

## 6. 验证结果参考 (gripper_only + hybrid, 113 帧, 2026-06-30, 第十二轮: MANO+offset 中和态)

```
配置: 颜色识别粉色 + 几何识别碗 + MANO+offset 中和态 + Z-floor 碗保护 + god_height=0.2m

范式识别结果:
  glb_1 (0.58, 0.06, 0.33): ✓ 粉色物体 (pinkness=R*(1-G)*(B-G)=0.21 最高)
  glb_3 (vol=0.0002, flat=0.446): ✓ 碗 (bowlness=vol*(1-flat)=1.1e-4 最高)

MANO+offset 中和态轨迹 (113帧, 基于 f_grasp 时序):
  APPROACH → DESCEND → CLOSE (f_grasp) → LIFT (smoothstep blend) →
  TRANSPORT (Z-floor) → RELEASE (Z-floor) → RETREAT (Z-floor)

  核心: EE[f] = mano_root_pos[f] + offset (常量平移, 保持 MANO 形状)
  f_grasp: MANO 轨迹最接近目标的帧 (不是固定 25%)
  Z-floor: TRANSPORT/RELEASE/RETREAT 阶段 EE_z >= bowl_z + 0.15 + 0.037

物体最终位置 (verify.json):
  glb_1 (粉): 初始 [0.033, -0.151, 0.017] → 最终 [0.147, -0.297, 0.017]
              xy_drift=18.48cm (被运输到碗附近), max lift=17.07cm ✓
              最终距碗中心: 5.46cm (MANO 轨迹在 RELEASE 时 +X 偏 4.2cm)
  glb_3 (碗): 初始 [0.094, -0.300, 0.027] → 最终 [0.093, -0.300, 0.027]
              xy_drift=0.02cm ✓ (Z-floor 保护下碗完全稳定)

>>> glb_1 真正夹住: 接触=True (61/113 连续), 跟随=True (112/113), 提升=True (17.07cm)
>>> MANO 轨迹形状保持: 仅常量平移, 不动态偏离 (用户: "mano参数为主体, 你可以平移轨迹")
>>> 碗完全稳定: xy_drift 207→16.75→0.02cm (Z-floor 三阶段保护)
>>> god 视角: 0.2m (从 0.5m 降低 60%, 用户: "god降低到0.2")
>>> 视频已生成: god_view_gripper_only_left.mp4
```

### 6b. 历史验证结果 (gripper_only + hybrid, 2026-06-29, 第十一轮: Pick-and-Place 范式)

```
配置: 颜色识别粉色物体 + 几何识别碗 + 7 阶段 pick-and-place (脱离 MANO) + god_height=0.5m

范式识别结果:
  glb_1 (0.58, 0.06, 0.33): ✓ 粉色物体 (pinkness=R*(1-G)*(B-G)=0.21 最高)
  glb_3 (vol=0.0002, flat=0.446): ✓ 碗 (bowlness=vol*(1-flat)=1.1e-4 最高)

Pick-and-Place 轨迹 (113帧, 7 阶段):
  F0-F11:    APPROACH  EE → approach_pos
  F11-F28:   DESCEND   EE → grasp_pos (gripper open)
  F28-F45:   CLOSE     gripper=0, PD 收敛
  F45-F62:   LIFT      EE → lift_pos (12cm 提升)
  F62-F84:   TRANSPORT EE 水平移动到 bowl_release_pos (19cm)
  F84-F96:   RELEASE   gripper open, 物体掉入碗
  F96-F113:  RETREAT   EE 后退到 bowl_release_pos + 15cm

物体最终位置 (verify.json):
  glb_1 (粉): 初始 [0.033, -0.151, 0.017] → 最终 [0.114, -0.326, 0.051]
              xy_drift=19.21cm ✓ (被运输到碗位置), max lift=15.54cm ✓
              最终距碗中心: 4cm (物体落入碗中)
  glb_3 (碗): 初始 [0.094, -0.300, 0.027] → 最终 [0.094, -0.300, 0.027]
              xy_drift=0.01cm ✓ (碗稳定未动)

>>> glb_1 真正夹住: 接触=True, 跟随=True (112/113), 提升=True (15.54cm)
>>> Pick-and-Place 成功: 粉色物体 → 碗 (距离 4cm)
>>> god 视角: 0.5m (从 1.0m 降低 50%)
>>> 范式通用化: 不硬编码物体名, 基于颜色+几何识别, 不同场景文件夹可用
>>> 视频已生成: god_view_gripper_only_left.mp4 (677KB)
```

### 6c. 历史验证结果 (gripper_only + hybrid, 2026-06-28, 第五轮+补充: 物体甩飞修复)

```
配置: MAX_ROOT_STEP=0.008 + angular_damping=50.0 + linear_damping=0.5 + restitution=0.0 + GRIPPER_FRICTION=2.0

轨迹跟踪误差 (物理输出 vs MANO 期望, 113帧):
  根位置: mean=20.69mm, max=84.75mm  (超 10mm ⚠ — MAX_ROOT_STEP=0.008 限幅代价)
  手指1/2: mean=2.56mm, max=3.70mm  (< 5mm ✓ 准确)

物体位置变化 (对比第五轮前 → 第五轮补充):
  glb_2: lift=39.76cm (上轮 40.66cm)  ✓ 真正夹住
  glb_3: lift=33.24cm (上轮 33.86cm)  ✓ 真正夹住, xy_drift 53→8.87cm ↓83%
  glb_5: lift=29.97cm (上轮 12.84cm → 29.97cm +17cm)  ★★ 突破! 盘子提升翻倍
         xy_drift: 224cm → 38.58cm ↓83% (不再飞2.2米!)
  glb_6: lift=0.80cm  (上轮 3.60cm)

>>> 真正夹住物体数: 3/7 (glb_2/3/5)
>>> 核心突破: glb_5 (盘子) lift 12.84→29.97cm, xy_drift 224→38cm (linear_damping=0.5 + 摩擦2.0)
>>> 相机: cam_view 每帧按 HaWoR 相机轨迹更新 (对齐 02 标准); god_view 在夹爪正上方俯瞰 (height=0.10)
>>> 物体翻转: angular_damping=50.0 + MAX_ROOT_STEP=0.008 + restitution=0.0 三重抑制
>>> 物体甩飞: linear_damping=0.5 + GRIPPER_FRICTION=2.0 双重抑制

物体甩飞修复对比 (linear_damping + 摩擦力):
                    修改前          修改后
  glb_5 lift:      12.84cm        29.97cm  (+17cm)
  glb_5 xy_drift:  224.64cm       38.58cm  (↓83%)
  glb_3 xy_drift:  53.74cm        8.87cm   (↓83%)
  力控最大力:      2.84N          3.6N
```

---

## 7. 关键修复历史

| 日期 | 修复 | 根因 |
|------|------|------|
| 2026-06-27 (第五轮) | cam_view 每帧更新 (对齐 02 标准) | 之前只初始化用第一帧, 主循环只 take_picture 没更新 local_pose → 相机不动 |
| 2026-06-27 (第五轮) | god_view 正上方俯瞰 + 降高度 (0.15→0.10) | 之前在夹爪前方+上方看正面, 用户要正上方俯瞰但高度低一点 |
| 2026-06-27 (第五轮) | 物体翻转修复 (angular_damping 5→50 + restitution 0.1→0 + MAX_ROOT_STEP 0.015→0.008) | 凸包碰撞体对扁平物体力臂大 + kinematic 根高速冲击 + 阻尼不足 → 盘子被碰翻. 5.0 阻尼不足以抑制 0.015 冲击, 三轮测试确认仅 0.008+50.0 让盘子不翻 (glb_5 0.04→12.84cm) |
| 2026-06-27 (第五轮) | GPU 检查修复 (nvidia-smi returncode=9) | returncode=9 (ECC 错误) 时 GPU 仍可用, 旧代码只检查 returncode==0 误判 → SAPIEN 渲染降级. 改用 stdout 非空判断 |
| 2026-06-27 | 统一 OBJECT_DENSITY 基础惯性变量 | 之前硬编码 500, 与 OBJECT_DENSITY=1000 不一致 |
| 2026-06-27 | 轨迹跟踪误差验证 | 之前只有开合判断, 无物理输出 vs MANO 期望误差 |
| 2026-06-27 | full_robot 添加 mano 分支 | 之前 full_robot 缺 mano, 与 gripper_only 架构不统一 |
| 2026-06-27 | 相机方向修复 (前方看手指) | 之前固定 offset 夹爪旋转不跟随朝向 → 每帧从 world pose 提取实际朝向动态计算, +forward 看手指侧 |
| 2026-06-27 | 物体翻转修复 (angular_damping) | 盘子等扁平物体被碰翻 → dynamic 物体加 angular_damping=5.0 抑制翻转力矩 (不加 linear_damping, 避免影响提升) |
| 2026-06-27 | 物体弹飞修复 | restitution 0.6→0.1 + 物体质量下限 0.15kg + MAX_ROOT_STEP 0.03→0.015 |
| 2026-06-26 | mano 接触维持 | 之前 mano 模式纯重放无接触维持, 物体易掉 |
| 2026-06-26 | 固定夹紧替代持续闭合 | 持续闭合把物体挤出飞出 (glb_6 xy_drift=389cm) |
| 2026-06-26 | gripper_only 夹爪不动 | physics_step 不调 set_qpos, 手指靠 PD 收敛来不及 |
| 2026-06-26 | 机器人朝向偏转 90° | root_quat=identity (面向+X), 相机看向 -Z |
| 2026-06-24 | 0 臂关节 bug | URDF joint 跨行, 旧正则要求 name/type 同行 → re.DOTALL |

---

## 8. Q&A 要点 (整合自 questions.md)

### Q1: 为什么仿真没有机械臂? 如何用两个 URDF 实现真实抓取?
**根因**: r1_v2_1_0.urdf joint 跨多行, 旧正则匹配失败 → 臂关节保持 fixed → "0 臂关节".
**解决**: 新建 grasp_hawor.py, `re.DOTALL` 匹配跨行, 12 个臂关节 fixed→revolute. 两种 URDF 模式: full_robot + gripper_only.

### Q2: grasp_demo.py 用的是什么仿真器?
SAPIEN (通过 GalaxeaManipSim gym 封装), 与 tri_model_physics SAPIEN 后端是同一物理引擎, 不是第四种引擎.

### Q3 (2026-06-27): 仿真没有一个基础的惯性变量吗, 每个物体的仿真都不一样吗?
**答**: 有统一基础惯性变量. `OBJECT_DENSITY=1000.0` (所有物体统一密度) + `OBJECT_MIN_MASS=0.15` (质量下限). 所有物体用同一公式 `max(volume * OBJECT_DENSITY, OBJECT_MIN_MASS)`, 不是每个物体不同. 之前代码硬编码 500 与常量不一致, 已修复为用 OBJECT_DENSITY.

### Q4 (2026-06-27): 夹爪和整个机器人的任务是一样的吗?
**答**: 是的, 除加载和映射外, 两者抓取逻辑完全一致. `_step_gripper_only` 和 `_step_full_robot` 都按相同顺序处理 hybrid/adaptive/mano 三种模式. `mano` 接触维持逻辑在两模式中代码一致 (只是变量名不同). 先测试夹爪, 验证通过后 full_robot 可直接应用相同逻辑.

### Q5 (2026-06-27): 夹爪运动如何判断准不准确?
**答**: 用轨迹跟踪误差 (物理输出 vs 真实 MANO 期望), 不是只有开合判断. 每帧记录:
- `expected_root` (限幅前 MANO 期望根位置) vs `actual_root` (限幅后 SAPIEN 实际)
- `expected_j1/j2` (抓取调整前 MANO 期望手指) vs `actual_j1/j2` (调整后实际)

统计 mean/max 误差 (mm), 阈值: 根 < 10mm, 手指 < 5mm. 实测 (第五轮 MAX_ROOT_STEP=0.008): 根 mean=20.69mm (超阈值, 是物理稳定代价), 手指达标. 第四轮 (MAX_ROOT_STEP=0.015): 根 mean=0.80mm 达标但盘子翻. 详见 Q8.

### Q6 (2026-06-27): 为什么第一人称视角相机不动? 如何按 02 标准移动?
**根因**: cam_view 只在初始化用第一帧 R_c2w[0]/t_c2w[0] 设置一次 set_local_pose, 主循环只 take_picture 没更新 local_pose → 相机保持第 0 帧不动.
**解决** (对齐 02_render_scene.py L2588-2590): 主循环每帧用 `global_idx = start_frame + local_idx` 取 `R_c2w_all[global_idx]`, 调 `hawor_cam_to_sapien_pose` 得 cam_pos/cam_quat, 再 `cam_view.set_local_pose` 更新. 严格按 02 标准每帧跟随 HaWoR 相机轨迹.

### Q7 (2026-06-27): god 视角如何放置? 放在夹爪上方俯瞰但高度低一点
**解决**: god_view 改为正上方俯瞰.
- 位置: `god_pos = gp + [0, 0, god_height]` (夹爪正上方)
- 朝向: `make_look_at_camera(god_pos, gp, up=夹爪forward)` (往下看)
- 高度: gripper_only=0.10 (从 0.15 降低), full_robot=0.40
- 每帧跟随: 用 frame_gripper_pose 提取当前夹爪位姿动态更新

### Q8 (2026-06-27): 物理仿真为什么夹爪碰一下就把盘子弄翻? 仿真器能否真实仿真交互?
**根因 (3 个叠加)**: ① 凸包碰撞体对扁平物体接触点偏边缘产生大力臂;② kinematic 根每帧 1.5cm (45cm/s) 高速冲击;③ angular_damping=5.0 阻尼不足.
**修复** (三轮测试): angular_damping 5→50 + restitution 0.1→0 + MAX_ROOT_STEP 0.015→0.008. 效果: glb_5 (盘子) 从 0.04cm (掉落) → 12.84cm (能抓!).
**MAX_ROOT_STEP 矛盾**: 0.008 盘子不翻但根误差 20mm; 0.015 根误差 0.8mm 但盘子翻+甩飞. 决策用 0.008 (用户最关心 "不翻").
**仿真器能力**: SAPIEN 能做真实仿真交互, 关键在参数调优 (凸包碰撞体 + kinematic 根冲击 + 物体质量/弹性). 调优后 3/7 物体真正夹住 + 提升.
