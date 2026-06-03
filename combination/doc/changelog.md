# 管线修改文档

> 最后更新: 2026-05-28
> 文件位置: `/home/an/robot_world_ws/src/dex-retargeting/example/combination/CHANGELOG.md`

***

## 1. GLB 坐标系对齐（核心问题）

### 1.1 问题背景

RAS 场景重建输出的 GLB 和 HaWoR 手部重建处于不同坐标系，需要对齐后才能在同一个 SAPIEN 场景中渲染。

### 1.2 坐标系分析

| 坐标系                        | 方向   | 单位 | 说明                 |
| -------------------------- | ---- | -- | ------------------ |
| RAS 外参 (extrinsics/\*.txt) | Z-UP | 米  | 房间对齐后的相机位姿         |
| RAS GLB (final\_scene.glb) | Y-UP | 米  | 导出时做了 z-up→y-up 转换 |
| HaWoR render world         | Y-UP | 米  | 手部重建的世界坐标系         |
| SAPIEN 世界                  | Z-UP | 米  | 物理仿真引擎坐标系          |

### 1.3 对齐方法：第一帧相机位姿锚定

**核心思想**：RAS 和 HaWoR 的第一帧相机位姿描述的是同一个真实相机，以此为锚点计算坐标系变换。

**变换链**：

```
RAS GLB 顶点 (Y-UP)
    ↓ p_hawor = s_inv * R_inv @ p_ras + t_inv
HaWoR render world (Y-UP)
    ↓ p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
SAPIEN 世界 (Z-UP)
```

**R\_inv 和 t\_inv 的计算**（在 `01_align_scene.py` 中）：

1. RAS 外参是 Z-UP，需要先转为 Y-UP：
   ```
   R_c2w_ras_yup = ZUP_TO_YUP @ R_c2w_ras_zup
   t_c2w_ras_yup = ZUP_TO_YUP @ t_c2w_ras_zup
   ```
2. RAS 外参是 OpenCV 相机模型（Z 朝后），HaWoR 是 OpenGL 模型（Z 朝前）：
   ```
   R_c2w_ras_gl = OPENCV_TO_OPENGL @ R_c2w_ras_yup
   ```
3. 第一帧相机位姿对齐：
   ```
   R_align = R_c2w_hawor @ R_c2w_ras_gl.T
   t_align = t_c2w_hawor - R_align @ t_c2w_ras_yup
   ```
4. Umeyama 尺度校正：
   ```
   s_inv ≈ 0.321 (RAS 尺度 → HaWoR 尺度)
   ```
5. 最终变换参数：
   ```
   R_inv = R_align
   t_inv = t_align
   ```

### 1.4 关键矩阵

```python
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
OPENCV_TO_OPENGL = np.diag([1, -1, -1])
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
RXWORLD_TO_SAPIEN = R_AXIS @ R_x  # y-up → z-up + SLAM→render
```

### 1.5 验证结果

| 指标            | 值                         | 说明             |
| ------------- | ------------------------- | -------------- |
| R\_align 旋转角度 | 0.47°                     | 几乎单位旋转，方向对齐正确  |
| 手腕→GLB中心      | 0.1923m                   | 合理（手腕在 GLB 旁边） |
| 手腕→GLB最近顶点    | min=0.0004m, mean=0.0824m | 帧5几乎接触（0.4mm）  |
| 指尖→GLB最近顶点    | min=0.0006m, mean=0.0623m | 指尖频繁接触 GLB 表面  |

**逐帧手腕→GLB最近距离趋势**：

- 帧0-5: 0.042→0.0004m（手接近GLB）
- 帧5-10: 0.0004→0.036m（手离开GLB）
- 帧10-19: 0.045→0.084m（手持续远离）

**结论**：手部与 GLB 对齐正确，手确实在操作过程中接近并接触 GLB 物体。视频看起来"远"可能是因为相机 FOV=120° 导致的广角畸变。

### 1.6 之前的错误方法

- **yingshe.py (Umeyama)**：假设 RAS 外参是 Y-UP（实际是 Z-UP），导致旋转角度 178.83°（几乎翻转），手-GLB 距离 0.5071m

---

## 2. 修改记录

### 2026-05-31 修改（第六轮）：映射修复 & 夹爪校正

#### 3.1 修复 `warm_start` 中 `wrist_quat` 坐标系不匹配

**问题**: `warm_start` 传入的 `wrist_quat` 来自 `pred_rot`（Render World Y-UP），但 `joints_sapien[0, :3]` 已经在 SAPIEN (Z-UP) 坐标系中。位置和旋转不在同一坐标系，导致优化器初始化方向错误，这是"关节反转"的根本原因。

**修改**: 将 `wrist_quat` 也转到 SAPIEN 坐标系：
```python
wrist_R_render = pr.matrix_from_compact_axis_angle(pred_rot)
wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
```

#### 3.2 修复夹爪初始化值超出关节限位

**问题**: `init_qpos[gripper_idx2] = -0.04`，但 r1_v2_1_0.urdf 中 joint2 的限位是 `[0, 0.05]`，-0.04 超出限位。

**修改**: 改为 `init_qpos[gripper_idx2] = 0.04`，与 joint1 一致。两处 warm_start 和三处初始化都已修复。

#### 3.3 确认容差和约束点映射与参考文件一致

对比 `r1_hand_tracking_video.py`，确认以下配置已一致：
- `IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]` ✓
- `target_link_human_indices = np.array([4, 8, 0])` ✓
- `target_link_names = ["right_gripper_finger_link1", "right_gripper_finger_link2", "right_gripper_link"]` ✓
- 映射关系: 拇指尖(4)→finger_link1, 食指尖(8)→finger_link2, 手腕(0)→gripper_link ✓

---

### 2026-05-31 修改（第五轮）：鲁棒性修复 & 代码清理

#### 2.1 修复 `wrist_pos_sapien` 未定义 bug

**问题**: `run_robot_tracking` 渲染循环中引用 `wrist_pos_sapien` 计算末端误差，但该变量从未定义，导致 `NameError`。

**修改**: 初始化 `wrist_pos_sapien = None`，在每帧手部数据有效时更新 `wrist_pos_sapien = joints_sapien[0, :3].copy()`。

#### 2.2 修复 `obj_nodes` 变量名不一致

**问题**: `run_robot_tracking` 和 `run_robot_only` 中 GLB 加载失败时创建了 `obj_nodes = []`，但实际变量名是 `obj_actors`，`obj_nodes` 从未使用。

**修改**: 移除多余的 `obj_nodes = []`。

#### 2.3 修复 `run_robot_only` 预计算循环未更新 base_link 位姿

**问题**: `run_robot_only` 的预计算循环中，底座跟踪后未更新 `base_link_p`/`base_link_q`/`base_link_R`/`base_link_R_inv`，导致后续帧的 IK 目标使用过时的 base_link 位姿。`run_robot_tracking` 中有此更新但 `run_robot_only` 遗漏了。

**修改**: 在预计算循环中添加底座跟踪后的 base_link 位姿更新逻辑。

#### 2.4 移除未使用的代码

- 移除 `saved_qpos` 变量（2处，从未使用）
- 移除 `axis_nodes` 变量（5处，位姿标识已删除后不再使用）
- 移除 `_render_camera_axes()` 和 `_render_pose_axes()` 方法（不再被调用）

#### 2.5 `01_align_scene.py` 增加 `--force_scale` 参数和验证

**新增**: `--force_scale` 参数，允许强制指定尺度因子覆盖自动计算值。

**新增验证**:
- R_align 行列式检查（应为 1.0）
- R_align 正交性检查（R^T R ≈ I）
- s_inv 范围检查（0.01~10.0 之外发出警告）
- s_inv ≤ 0 时抛出异常

#### 2.7 静态相机自动检测和回退

**问题**: Umeyama 通过相机轨迹离散度计算尺度比，静态相机时 sigma→0 导致尺度不稳定。用户不应手动判断。

**修改**: 自动检测静态相机 (sigma < 0.01)，回退到基于手-GLB 距离的启发式估算：
1. 计算 GLB 质心在 HaWoR 坐标系中的位置（未缩放）
2. 计算手部平均位置到 GLB 质心的距离
3. 假设手-物距离约 0.15m，反推 s_inv

#### 2.6 文档润色

- `Combination_pipeline.md`: 更新对齐原理描述（第一帧相机锚定而非旧 Umeyama 方式），移除已删除功能的描述（相机柱子、位姿标识），增加机器人映射链和底座跟踪说明，精简文件说明表格。

---

### 2026-05-29 修改（第四轮）：映射方法重构 & 视觉清理

#### 2.1 映射方法重构 — 采用 r1_hand_tracking_video.py 的3约束点方案

**问题**: 旧方法使用2约束点[4,8] + `_compute_ee_orientation_from_wrist()`直接计算朝向，夹爪开合和姿态映射都不准确。

**旧方法 vs 新方法对比**:

| 对比项 | 旧方法 | 新方法 (参考r1_hand_tracking_video.py) |
|--------|--------|--------------------------------------|
| 约束点 | 2个 [4,8] (食指尖+中指尖) | **3个 [4,8,0] (食指尖+中指尖+手腕)** |
| 朝向来源 | `_compute_ee_orientation_from_wrist()` 直接从手腕旋转计算 | **FK提取** `_get_gripper_pose_from_retargeting()` |
| IK目标位置 | 手腕位置 `wrist_pos + offset` | **FK夹爪位置** `gripper_pos_fk + offset` |
| normal_delta | 默认 4e-3 (锁死朝向) | **1e-5** (朝向可自由变化) |
| huber_delta | 默认 | **0.01** |
| IK容差 | [0.001,0.001,0.001,0.002,0.002,0.002] | **[0.1,0.1,0.1,0.1,0.1,0.1]** |
| warm_start | 无 | **有** (用第一帧手腕四元数初始化) |

**3约束点为什么能约束朝向**:
- 2约束点: 6约束 vs 7DOF → 欠定1DOF → 接近轴旋转无梯度
- 3约束点: 9约束 vs 7DOF → 超定2约束 → 手腕位置相对指尖方向编码朝向，优化器自然产生朝向梯度

**IK容差为什么改为0.1**:
- 容差是RelaxedIK目标函数的权重分母: `L = Σ(err_i / tol_i)²`
- 旧值0.001: 位置权重=10⁶, 朝向权重=2.5×10⁵ → 位置主导，朝向被拉偏
- 新值0.1: 位置权重=100, 朝向权重=100 → 位置和朝向平衡

#### 2.2 去除蓝色相机柱子

**问题**: 视频中相机位置有蓝色柱状标记，影响观感。

**修改**: 移除所有 `_render_camera_axes()` 调用（4处），相机位置更新保留但不再渲染标记。

#### 2.3 去除位姿标识

**问题**: 视频中末端执行器（手腕/夹爪）位姿处有坐标轴标识，影响观感且与最终展示无关。

**修改**: 移除所有位姿坐标轴渲染调用，位姿计算保留但不再渲染标识。

#### 2.4 Output文件夹说明

example目录下的output文件夹是测试输出，包含：

| 文件夹 | 内容 | 建议 |
|--------|------|------|
| `combination/output/` | 对齐报告、变换参数、视频、npy | **保留** — 管线正式输出 |
| `combination/test/` | 调试脚本、旧对齐视频 | 可删除 |
| `position_retargeting/pv_retargeting/` | r1跟踪视频和日志 | **保留** — 参考实现输出 |
| `position_retargeting/test/` | IK测试视频和日志 | 可删除 |
| `simulation/` | 仿真测试视频和日志 | 可删除 |
| `video_egocentric_retargeting/output_*/` | 第一人称管线输出 | 可删除 |
| `output_pin_base_frame/` (项目根) | R1跟踪视频和日志 | 可删除 |

---

### 2026-05-29 修改（第三轮）：IK收敛 & 底座跟踪 & 关键点精简

#### 2.1 RelaxedIK 姿态收敛改善

**问题**: RelaxedIK每次调用只做一步梯度下降，5次调用不足以收敛到目标姿态，导致末端姿态误差大。

**根因分析**:
1. RelaxedIK是增量式求解器，每次`solve_position`只做一步优化
2. 位置梯度量级大于姿态梯度，位置先收敛但姿态还差很远
3. 求解器内部状态未同步——每次求解后不reset，下一帧优化起点偏移
4. R1 Lite只有6个关节（非冗余臂），姿态可达空间受限

**修改**:
- 求解次数从5增加到20: `IK_SOLVE_PER_FRAME = 20`
- 旋转容差从0.005收紧到0.002: `IK_TOLERANCES = [0.001, 0.001, 0.001, 0.002, 0.002, 0.002]`
- 每帧求解后reset求解器内部状态: `ik_solver.relaxed_ik_right.reset(list(arm_joints))`

#### 2.2 关键点显示精简

**问题**: 视频中显示了所有21个MANO关节点，视觉混乱，与夹爪映射无关。

**修改**:
- `_render_keypoints()` 只显示`ref_indices`中的关节（4=食指尖, 8=中指尖）
- 移除其他关节的渲染，移除`FINGER_GROUP_COLORS`在该函数中的使用

#### 2.3 机械臂底座小范围跟踪

**问题**: 底座固定在手腕轨迹质心上方，当手腕远离质心时IK不稳定。

**修改**:
- 新增`_compute_tracking_base_pos()`方法，根据当前手腕位置计算底座偏移
- 偏移量在base_link坐标系中计算，限制在±`BASE_TRACKING_RANGE`(0.08m)范围内
- 三个`run_*`方法中每帧更新底座位置并重新获取base_link位姿
- 新增常量: `BASE_TRACKING_RANGE = 0.08`, `BASE_TRACKING_ALPHA = 0.15`

---

### 2026-05-29 修改（第二轮）：机械臂映射精度 & 相机视角修正

#### 2.1 GLB 渲染方式改为 SAPIEN 公开 API

| 修改项    | 修改前                                                   | 修改后                                                       |
| ------ | ----------------------------------------------------- | --------------------------------------------------------- |
| 渲染 API | `context.create_mesh_from_array()` + `internal_scene` | `scene.create_actor_builder()` + `add_visual_from_file()` |
| 加载方式   | 内存中构建 mesh 数组                                         | 导出变换后 PLY → SAPIEN 加载 PLY                                 |
| 颜色设置   | `context.create_material()`                           | `sapien.render.RenderMaterial(base_color=...)`            |
| 简化     | `MAX_FACES_PER_OBJECT=0`（不简化）                         | 不简化，SAPIEN 公开 API 自动处理大 mesh                              |
| 渲染效果   | 点云状（内部 API 处理大 mesh 异常）                               | 实物质感（公开 API 正确渲染）                                         |

**参照文件**：`/home/an/robot_world_ws/src/ReplicateAnyScene/View/pipeline_universal.py`

#### 2.2 机械臂末端朝向综合

| 修改项   | 修改前                                    | 修改后                                                        |
| ----- | -------------------------------------- | ---------------------------------------------------------- |
| 朝向来源  | 仅 `wrist_R_sapien`                     | `_compute_combined_ee_orientation()`：手腕旋转(40%) + 手部几何(60%) |
| IK 容差 | `[0.001, 0.001, 0.001, 0.1, 0.1, 0.1]` | `[0.001, 0.001, 0.001, 0.03, 0.03, 0.03]`                  |
| 位置映射  | `mapping_offset=zeros`                 | `mapping_offset=zeros, safety_offset=zeros`（手部实际位置）        |

**综合朝向算法**：

1. 计算手部几何朝向 R\_hand（拇指→食指 + 手腕→MCP）
2. 计算 MANO 手腕朝向 R\_wrist
3. 逐轴加权融合：`R_combined[:, col] = 0.4 * R_wrist[:, col] + 0.6 * R_hand[:, col]`
4. SVD 正交化保证 R\_combined 是合法旋转矩阵
5. 如果 R\_hand 计算失败，回退到 R\_wrist

#### 2.3 GPU 渲染

| 修改项        | 修改前                   | 修改后                                |
| ---------- | --------------------- | ---------------------------------- |
| Vulkan ICD | 硬编码 `nvidia_icd.json` | 自动检测：nvidia-smi 可用→NVIDIA，否则→Intel |

#### 2.4 `_extract_wrist_pose` 状态

`_extract_wrist_pose` 方法**未被使用**。当前末端朝向由 `_compute_combined_ee_orientation()` 计算，位置由手腕关节 `joints_sapien[0, :3]` 直接获取。

#### 2.5 `03_track_robot.py` 状态

`03_track_robot.py` **仍然有效**，是独立机器人跟踪脚本（不需要 RAS GLB）。与 `02_render_scene.py --mode robot_only` 的区别：

- 不需要 RAS GLB 场景
- 不需要第一人称相机轨迹
- 支持多种第三人称视角
- 适合快速验证手部→机器人映射

### 2026-05-28 修改（第一轮）

#### GLB 像点云的根因

`fast_simplification` 未安装 → `simplify_quadric_decimation` 抛异常 → fallback 的 `faces[::step]` 把连续三角面拆成孤立碎片 → 看起来像点云。

修复：安装 `fast_simplification`，最终改为 SAPIEN 公开 API 加载。

***

## 3. 文件清单

| 文件                          | 功能        | 关键修改                                   |
| --------------------------- | --------- | -------------------------------------- |
| `01_align_scene.py`         | 计算对齐参数    | 第一帧相机位姿锚定 + ZUP→YUP + OPENCV→OPENGL    |
| `02_render_scene.py`        | 渲染仿真场景    | SAPIEN 公开 API 加载 GLB + GPU 自动检测 + 综合朝向 |
| `03_track_robot.py`         | 独立机器人跟踪   | 无修改，仍然有效                               |
| `hand_object.py`            | 独立渲染脚本    | IK 容差修改                                |
| `yingshe.py`                | 旧对齐脚本（参考） | Umeyama 方式，RAS Y-UP 假设错误               |
| `r1_hand_tracking_video.py` | 参考实现      | DexYCB 管线，综合朝向算法来源                     |
| `pipeline_universal.py`     | GLB 可视化参考 | SAPIEN 公开 API 加载 GLB 的参考实现             |
| `test/analyze_distance.py`  | 距离分析脚本    | 手-GLB 逐帧距离分析                           |

***

## 4. 坐标系变换速查

```
GLB顶点 (RAS Y-UP)
  → s_inv * R_inv @ p + t_inv
HaWoR render world (Y-UP)
  → RXWORLD_TO_SAPIEN @ p
SAPIEN world (Z-UP)

HaWoR 手部顶点 (render world Y-UP)
  → RXWORLD_TO_SAPIEN @ p
SAPIEN world (Z-UP)

HaWoR 相机 (R_c2w, t_c2w, render world Y-UP)
  → hawor_cam_to_sapien_pose()
SAPIEN 相机

MANO 手腕旋转 (render world Y-UP)
  → RXWORLD_TO_SAPIEN @ wrist_R_render
SAPIEN world (Z-UP)

机械臂末端朝向 (SAPIEN world Z-UP)
  → _compute_combined_ee_orientation(joints_sapien, wrist_R_sapien)
  → 40% wrist_R_sapien + 60% hand_geometry_R
  → SVD 正交化
```

***

## 5. 手-GLB 距离详细分析

### 5.1 GLB 在 SAPIEN 坐标系

| 属性 | 值                                                   |
| -- | --------------------------------------------------- |
| 中心 | \[-0.006, -0.238, 0.053]                            |
| 边界 | \[-0.074, -0.335, 0.010] \~ \[0.132, -0.140, 0.083] |
| 尺寸 | 0.205 × 0.194 × 0.073 m                             |

### 5.2 手部在 SAPIEN 坐标系

| 属性   | 值                       |
| ---- | ----------------------- |
| 手腕中心 | \[0.176, -0.283, 0.008] |
| 指尖中心 | \[0.069, -0.249, 0.103] |

### 5.3 距离统计

| 指标         | min     | mean    | max     |
| ---------- | ------- | ------- | ------- |
| 手腕→GLB最近顶点 | 0.0004m | 0.0824m | 0.1507m |
| 指尖→GLB最近顶点 | 0.0006m | 0.0623m | 0.1858m |

### 5.4 逐帧手腕→GLB最近距离（前20帧）

| 帧  | 距离(m)      | 说明       |
| -- | ---------- | -------- |
| 0  | 0.0423     | <br />   |
| 1  | 0.0363     | <br />   |
| 2  | 0.0350     | <br />   |
| 3  | 0.0268     | <br />   |
| 4  | 0.0187     | <br />   |
| 5  | **0.0004** | 手几乎接触GLB |
| 6  | 0.0124     | <br />   |
| 7  | 0.0211     | <br />   |
| 8  | 0.0290     | <br />   |
| 9  | 0.0363     | <br />   |
| 10 | 0.0451     | <br />   |
| 15 | 0.0658     | <br />   |
| 19 | 0.0843     | 手远离GLB   |

