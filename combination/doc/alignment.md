# RAS GLB → HaWoR 坐标系对齐 — 完整指南

## 0. 涉及的坐标系一览

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  RAS (ReplicateAnyScene)                                            │
│  ─────────────────────────                                          │
│  VGGT World (任意) ──R,t──→ Room World (Z-UP) ──z2y──→ GLB (Y-UP) │
│                              地板z=0           导出时做了z→y转换     │
│                                                                     │
│  HaWoR                                                              │
│  ─────                                                              │
│  SLAM World (Y-down, Z-forward, 无尺度) ──scale──→ SLAM World (米) │
│  pred_trans 在此坐标系 (OpenCV约定)                                  │
│                                                                     │
│  SLAM World (米, Y-down) ──R_x──→ Render World (Y-UP, Z-backward)  │
│                              diag(1,-1,-1)    R_c2w/t_c2w 保存在此  │
│                                                                     │
│  SAPIEN                                                             │
│  ──────                                                             │
│  Z-UP, X=right, Y=forward, Z=up                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 1. 之前犯的错误（逐步分析）

### 错误 1：混淆了 HaWoR 的两个坐标系

**现象**：手部位置和场景对不上，手飘在场景外面。

**原因**：HaWoR 的 `hawor_results_*.npz` 中保存了两种坐标系的数据：
- `pred_trans` / `pred_rot`：在**原始 SLAM World**（OpenCV 约定，Y-down, Z-forward）
- `R_c2w` / `t_c2w`：已经乘了 `R_x = diag(1,-1,-1)`，在 **Render World**（OpenGL 约定，Y-up, Z-backward）

**错误做法**：直接用 `R_c2w` / `t_c2w` 的坐标系来放置 `pred_trans` 的手部，导致手和相机不在同一坐标系。

**正确做法**：
```python
# pred_trans 在 SLAM World (Y-down)
# R_c2w/t_c2w 在 Render World (Y-up)
# 两者之间差 R_x = diag(1,-1,-1)

# 从 R_c2w/t_c2w 恢复 SLAM World 相机位置:
R_x = np.diag([1.0, -1.0, -1.0])
cam_slam = R_x @ t_c2w  # Render World → SLAM World

# 或者从 SLAM World 转到 Render World:
cam_render = R_x @ pred_trans  # SLAM World → Render World
```

### 错误 2：GLB 加载后不可见（内部API vs 公共API）

**现象**：GLB 场景在 viewer 中可见，但在视频渲染中消失。

**原因**：SAPIEN 有两套渲染管线：
- **内部 API**（`context.create_mesh_from_array`）：只在 viewer 窗口可见，不被 `camera.take_picture()` 捕获
- **公共 API**（`scene.create_actor_builder().add_visual_from_file()`）：在 viewer 和视频渲染中都可见

**错误做法**：用内部 API 直接创建 mesh 节点。
```python
# 错误：内部API，视频不可见
mesh = context.create_mesh_from_array(vertices, faces, mat)
node = internal_scene.add_node(mesh)
```

**正确做法**：用公共 API，通过临时 PLY 文件加载。
```python
# 正确：公共API，视频可见
builder = scene.create_actor_builder()
builder.add_visual_from_file(filename=temp_ply, material=material)
actor = builder.build_kinematic(name=geom_name)
```

### 错误 3：末端朝向计算不稳定

**现象**：机械臂末端朝向抖动严重，与实际手腕旋转不匹配。

**原因**：旧方法 `_compute_combined_ee_orientation()` 基于手部关节几何（拇指尖→食指尖、手腕→MCP中心）构造旋转矩阵，与手腕旋转加权混合。关节位置噪声大，且未正确处理 MANO 坐标系到夹爪坐标系的映射。

**错误做法**：
```python
# 错误：从关节几何猜测朝向，不稳定
y_axis = index_tip - thumb_tip
x_axis = mcp_center - wrist
R_ee = np.column_stack([x_axis, y_axis, z_axis])
```

**正确做法**：使用 HaWoR 的手腕旋转（`pred_rot`）+ 正确的坐标系转换。
```python
# 正确：从手腕旋转 + MANO→夹爪坐标系转换
wrist_R_sapien = RXWORLD_TO_SAPIEN @ axis_angle_to_matrix(pred_rot)
R_ee_world = wrist_R_sapien @ OPERATOR2MANO_RIGHT.T @ R_GRIPPER_ALIGN
```

### 错误 4：RelaxedIK 朝向容差太松

**现象**：机械臂末端位置有几厘米偏差。

**原因**：容差 `[0.001, 0.001, 0.001, 0.03, 0.03, 0.03]` 中旋转容差 0.03rad ≈ 1.7°，在 0.5m 臂长下导致 ~15mm 末端偏差。

**错误做法**：
```python
tolerances = [0.001, 0.001, 0.001, 0.03, 0.03, 0.03]  # 旋转太松
```

**正确做法**：
```python
tolerances = [0.001, 0.001, 0.001, 0.005, 0.005, 0.005]  # 旋转收紧到0.29°
```

### 错误 5：相机 FOV 硬编码

**现象**：渲染视角与原视频不匹配，场景看起来太远。

**原因**：硬编码 `HAWOR_FOCAL = 600.0`，但实际视频的焦距可能不同。

**错误做法**：
```python
HAWOR_FOCAL = 600.0  # 硬编码
CAM_FOV = 2 * np.arctan(CAM_HEIGHT / 2.0 / HAWOR_FOCAL)
```

**正确做法**：从 HaWoR 数据读取实际焦距。
```python
# 优先级: hawor_results.npz → est_focal.txt → 默认600
img_focal = hawor_data.get("img_focal", None)
if img_focal is None:
    est_focal_file = hawor_path / "est_focal.txt"
    if est_focal_file.exists():
        img_focal = float(est_focal_file.read_text().strip())
    else:
        img_focal = 600.0

# 按渲染分辨率缩放
focal_for_render = img_focal * CAM_WIDTH / 1280.0
cam_fov = 2 * np.arctan(CAM_HEIGHT / 2.0 / focal_for_render)
```

---

## 2. 正确的对齐方案

### 2.1 完整变换链

```
RAS GLB (Y-UP, VGGT单位)
    │
    │  Step 1: 01_align_scene.py 计算 RAS Y-UP → HaWoR Render World 变换
    │  p_hawor = s_inv * R_inv @ p_glb + t_inv
    │  参数保存在 transform_params.npz
    ↓
HaWoR Render World (Y-UP, 米制)
    │
    │  Step 2: RXWORLD_TO_SAPIEN = R_AXIS @ R_x
    │  R_AXIS = [[1,0,0],[0,0,1],[0,-1,0]]  (Y-UP → Z-UP)
    │  R_x    = diag(1,-1,-1)                (Render → SLAM 再回到 Z-UP)
    │  p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
    ↓
SAPIEN World (Z-UP, 米制)
```

### 2.2 01_align_scene.py 的对齐原理

**核心思想**：同一个视频的第一帧相机，在 RAS 和 HaWoR 中描述同一个物理相机。

```
RAS 第一帧相机:  R_c2w_ras (Y-UP, OpenCV约定)
HaWoR 第一帧相机: R_c2w_hawor (Y-UP, OpenGL约定)

同一个物理相机，OpenCV和OpenGL之间差 R_x = diag(1,-1,-1):
  R_align = R_c2w_hawor @ R_x @ R_c2w_ras.T
  t_align = t_c2w_hawor - R_align @ t_c2w_ras

尺度校正 (Umeyama):
  s_inv = sigma_hawor / sigma_ras

最终变换 (GLB Y-UP → HaWoR Render World):
  p_hawor = s_inv * R_align @ p_glb + t_align_scaled
```

### 2.3 02_render_scene.py 中的坐标系变换

```python
# 常量定义
R_x = np.diag([1.0, -1.0, -1.0])           # OpenCV ↔ OpenGL
R_AXIS = np.array([[1,0,0],[0,0,1],[0,-1,0]]) # Y-UP ↔ Z-UP
RXWORLD_TO_SAPIEN = R_AXIS @ R_x            # HaWoR Render → SAPIEN

# 手部位置/朝向变换
joints_sapien = (RXWORLD_TO_SAPIEN @ joints_render.T).T
wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render

# 相机位姿变换
cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w
cam_R_sapien = RXWORLD_TO_SAPIEN @ R_c2w

# GLB 变换 (由 01_align_scene.py 预计算)
p_hawor = s_inv * R_inv @ p_glb + t_inv
p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
```

### 2.4 末端朝向的正确计算

```python
# 1. HaWoR 手腕旋转 (轴角 → 旋转矩阵 → SAPIEN坐标系)
wrist_R_render = axis_angle_to_matrix(pred_rot)
wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render

# 2. MANO 操作者坐标系转换
R_mano2world = wrist_R_sapien @ OPERATOR2MANO_RIGHT.T

# 3. 夹爪对齐
R_ee_world = R_mano2world @ R_GRIPPER_ALIGN

# 4. 转换到 base_link 坐标系 (RelaxedIK 需要)
ee_R_base = base_link_R_inv @ R_ee_world
ee_quat_base = quaternion_from_matrix(ee_R_base)
```

其中：
- `OPERATOR2MANO_RIGHT = [[0,0,-1],[-1,0,0],[0,1,0]]`：MANO 右手坐标系到操作者坐标系的转换
- `R_GRIPPER_ALIGN = [[0,0,1],[0,1,0],[-1,0,0]]`：操作者坐标系到夹爪坐标系的对齐

---

## 3. 坐标系变换速查表

| 变换 | 矩阵 | 效果 |
|------|------|------|
| YUP_TO_ZUP (GLB→Room) | `[[1,0,0],[0,0,-1],[0,1,0]]` | (x,y,z)→(x,-z,y) |
| ZUP_TO_YUP (Room→GLB) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_AXIS (Y-UP→Z-UP) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_x (OpenCV↔OpenGL) | `diag(1,-1,-1)` | (x,y,z)→(x,-y,-z) |
| RXWORLD_TO_SAPIEN | `R_AXIS @ R_x` | HaWoR Render → SAPIEN |
| OPERATOR2MANO_RIGHT | `[[0,0,-1],[-1,0,0],[0,1,0]]` | MANO右手→操作者 |
| R_GRIPPER_ALIGN | `[[0,0,1],[0,1,0],[-1,0,0]]` | 操作者→夹爪 |

### 各坐标系轴约定

```
RAS Room World (Z-UP):     RAS GLB (Y-UP):          HaWoR SLAM World (Y-down):
    +Z (up)                    +Y (up)                   -Y (up)
    |                          |                          |
    +---- +Y                  +---- -Z                  +---- +Z (forward)
   /                          /                          /
  +X                         +X                         +X (right)

HaWoR Render World (Y-UP):  SAPIEN (Z-UP):
    +Y (up)                    +Z (up)
    |                          |
    +---- -Z (backward)       +---- +Y (forward)
   /                          /
  +X (right)                 +X (right)
```

---

## 4. 验证清单

| 检查项 | 方法 | 通过标准 |
|--------|------|---------|
| 手在场景范围内 | bounds 检查 | 手 xz ⊆ 场景 xz ± 0.5m |
| 手在相机前方 | 方向点积 > 0 | 相机→手 方向 · 相机forward > 0 |
| 相机轨迹残差 | `‖aligned[i] - ras_cam[i]‖` | 中位误差 < 0.1m |
| GLB 在视频中可见 | 渲染检查 | 公共API加载的actor可见 |
| 末端朝向合理 | 可视化坐标轴 | 夹爪朝向与手腕旋转一致 |
| 末端位置误差 | FK验证 | < 5mm (收紧容差后) |

---

## 5. 关键文件对照

| 文件 | 作用 |
|------|------|
| `01_align_scene.py` | 计算 RAS→HaWoR 变换参数，保存到 `transform_params.npz` |
| `02_render_scene.py` | 加载变换后的 GLB + HaWoR 手部 + R1 机器人，SAPIEN 渲染 |
| `transform_params.npz` | 包含 `s_inv`, `R_inv`, `t_inv` (GLB→HaWoR 变换参数) |
| `changelog.md` | 代码修改记录 |
| `question.md` | 问题与回答 |
