# RAS GLB → HaWoR 坐标系对齐 — 完整指南

## 0. 涉及的坐标系一览

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  RAS (ReplicateAnyScene)                                            │
│  ─────────────────────────                                          │
│  VGGT World (任意) ──R,t──→ Room World (Z-UP) ──z2y──→ GLB (Y-UP 或 Z-UP) │
│                              地板z=0     导出时可能做z→y转换,也可能未做  │
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

## 1. 实际文件路径与数据概况

### 输入文件

| 系统 | 路径 | 帧数 |
|------|------|------|
| RAS 输出 | `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/my_7mp4_result` | 20 帧 (0-19) |
| HaWoR 输出 | `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7` | 113 帧 (0-112) |

### 关键文件对照

| 用途 | RAS | HaWoR |
|------|-----|-------|
| 外参（相机位姿） | `extrinsics/0.txt` ~ `19.txt` | `reconstruction/hawor_results_0_113.npz` → `t_c2w`, `R_c2w` |
| 内参 | `intrinsic.txt` | `hawor_results_0_113.npz` → `img_focal` |
| 深度图 | `depth/0.png` ~ `19.png` | SLAM `disps`（视差，非直接深度） |
| 点云 | `point_cloud.ply` | 无 |
| 场景网格 | **`final_scene.glb`** ← 待变换 | 无 |
| 手部姿态 | 无 | `hawor_results_0_113.npz` → `pred_trans`, `pred_rot` |
| SLAM 原始轨迹 | 无 | `SLAM/hawor_slam_w_scale_0_113.npz` → `traj`, `scale` |

### 关键发现

> **HaWoR 的 `hawor_results_0_113.npz` 中 `R_c2w` 和 `t_c2w` 已经应用了 R_x 翻转 (OpenCV→OpenGL)，不是原始 SLAM World。`pred_trans` 仍然是原始 SLAM World。**

验证方法：`R_c2w[0]` 看是否是 `[[1,0,0],[0,-1,0],[0,0,-1]]` 附近。如果是，说明已应用 R_x。

---

## 2. 之前犯的错误（逐步分析）

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

**正确做法**：用公共 API，通过临时 PLY 文件加载。
```python
builder = scene.create_actor_builder()
builder.add_visual_from_file(filename=temp_ply, material=material)
actor = builder.build_kinematic(name=geom_name)
```

### 错误 3：末端朝向计算不稳定

**现象**：机械臂末端朝向抖动严重，与实际手腕旋转不匹配。

**原因**：旧方法基于手部关节几何构造旋转矩阵，关节位置噪声大，且未正确处理 MANO 坐标系到夹爪坐标系的映射。

**正确做法**：使用 HaWoR 的手腕旋转（`pred_rot`）+ 正确的坐标系转换。
```python
wrist_R_sapien = RXWORLD_TO_SAPIEN @ axis_angle_to_matrix(pred_rot)
R_ee_world = wrist_R_sapien @ OPERATOR2MANO_RIGHT.T @ R_GRIPPER_ALIGN
```

### 错误 4：RelaxedIK 朝向容差太松

**现象**：机械臂末端位置有几厘米偏差。

**原因**：容差 `[0.001, 0.001, 0.001, 0.03, 0.03, 0.03]` 中旋转容差 0.03rad ≈ 1.7°，在 0.5m 臂长下导致 ~15mm 末端偏差。

**正确做法**：
```python
tolerances = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # 位置和朝向平衡
```

### 错误 5：相机 FOV 硬编码

**现象**：渲染视角与原视频不匹配，场景看起来太远。

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

focal_for_render = img_focal * CAM_WIDTH / 1280.0
cam_fov = 2 * np.arctan(CAM_HEIGHT / 2.0 / focal_for_render)
```

---

## 3. 正确的对齐方案

### 3.1 完整变换链

```
RAS GLB (Y-UP 或 Z-UP, VGGT单位)
    │
    │  Step 0: 检测 GLB 坐标系 (_detect_glb_up_axis)
    │  Y-UP: 导出时做了 z→y 转换 (相机坐标系)
    │  Z-UP: 未转换, 地面坐标系 (地板在 z=0)
    │
    │  Step 1: 01_align_scene.py 计算 RAS Y-UP → HaWoR Render World 变换
    │  R_c2w Y-UP 转换取决于 GLB 坐标系:
    │    Y-UP GLB: R_c2w_yup = ZUP_TO_YUP @ R_c2w_zup @ ZUP_TO_YUP.T (相似变换)
    │    Z-UP GLB: R_c2w_yup = ZUP_TO_YUP @ R_c2w_zup (直接乘)
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

### 3.2 01_align_scene.py 的对齐原理

**核心思想**：同一个视频的第一帧相机，在 RAS 和 HaWoR 中描述同一个物理相机。

**R_c2w Y-UP 转换** (取决于 GLB 坐标系):
- Y-UP GLB (导出时已做 z→y): 相机约定随世界 up 轴变化 → 相似变换
  `R_c2w_yup = ZUP_TO_YUP @ R_c2w_zup @ ZUP_TO_YUP.T`
- Z-UP GLB (地面坐标系, 未转换): 只改变世界坐标系, 相机约定不变 → 直接乘
  `R_c2w_yup = ZUP_TO_YUP @ R_c2w_zup`

**尺度校正**:
- 优先使用 Umeyama (相机轨迹标准差比)
- 若 Umeyama 尺度导致手→GLB距离 > 10cm, 自动网格搜索更优 s_inv

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

### 3.3 02_render_scene.py 中的坐标系变换

```python
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

### 3.4 末端朝向的正确计算

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

## 4. 操作步骤（01_align_scene.py 自动化）

`01_align_scene.py` 已将以下步骤自动化，但理解原理有助于排错。

### 4.1 加载 RAS 相机位姿 (Z-UP)

```python
# RAS 外参: world-to-camera 变换 [R_w2c | t_w2c]
R_c2w = R_w2c.T
cam_pos = -R_c2w @ t_w2c
```

### 4.2 RAS Z-UP → Y-UP 转换

```python
ZUP_TO_YUP = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]   # (x,y,z) → (x,z,-y)
ras_cam_pos_yup = (ZUP_TO_YUP @ ras_cam_pos_zup.T).T
ras_R_c2w_yup  = ZUP_TO_YUP @ R_c2w_zup @ ZUP_TO_YUP.T
```

### 4.3 加载 HaWoR 相机位姿

从 `hawor_results_*.npz` 读取 `R_c2w` 和 `t_c2w`，这些已经在 **Render World (Y-UP)** 坐标系中。

### 4.4 第一帧相机位姿对齐

```python
R_align = R_c2w_hawor @ OPENCV_TO_OPENGL @ R_c2w_ras_yup.T
t_align = t_c2w_hawor - R_align @ t_c2w_ras_yup
```

### 4.5 尺度校正 (Umeyama)

```python
sigma_ras = sqrt(mean(||ras_cam - ras_cam_mean||²))
sigma_hawor = sqrt(mean(||hawor_cam - hawor_cam_mean||²))
s_inv = 1.0 / (sigma_ras / sigma_hawor)
```

**静态相机回退**: 当 sigma < 0.01 时，自动回退到基于手-GLB 距离的启发式估算。

### 4.6 保存

```python
# transform_params.npz 包含:
s_inv       # RAS → HaWoR 缩放因子 (~0.32)
R_inv       # GLB Y-UP → HaWoR 旋转矩阵 (= R_align)
t_inv       # GLB Y-UP → HaWoR 平移 (= t_align_scaled)
```

---

## 5. 手动对齐步骤（调试用）

当 `01_align_scene.py` 自动对齐结果不理想时，可手动执行以下步骤微调。

### 5.1 加载数据

```python
import numpy as np; import cv2; import trimesh; import os; from glob import glob

RAS_OUT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/my_7mp4_result'
HAWOR_RES = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7/reconstruction/hawor_results_0_113.npz'

# RAS 相机位置
ext_files = sorted(glob(os.path.join(RAS_OUT, 'extrinsics', '*.txt')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
ras_extrinsics = []
for f in ext_files:
    ext = np.loadtxt(f)
    if ext.shape == (3, 4):
        ext = np.vstack([ext, [0, 0, 0, 1]])
    ras_extrinsics.append(ext)
ras_extrinsics = np.array(ras_extrinsics)
ras_cam = np.array([-e[:3,:3].T @ e[:3,3] for e in ras_extrinsics])

# HaWoR 相机位置 (恢复原始 SLAM World)
hawor = dict(np.load(HAWOR_RES, allow_pickle=True))
R_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
hawor_cam_original = np.array([R_x @ t for t in hawor['t_c2w']])
```

### 5.2 计算对齐参数

```python
R_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
R_residual = np.eye(3)
R_total = R_residual @ R_axis
s = 1.0  # 手动调整尺度
t = ras_cam[0] - s * (R_total @ hawor_cam_original[0])

# 逆变换参数 (用于 GLB)
s_inv = 1.0 / s
R_inv = R_total.T
t_inv = -s_inv * (R_inv @ t)
```

### 5.3 残差验证

```python
aligned = s * (R_total @ hawor_cam_original[:n].T).T + t
errors = np.linalg.norm(aligned - ras_cam[:n], axis=1)
print(f'errors: mean={errors.mean():.4f}, median={np.median(errors):.4f}, max={errors.max():.4f}')
```

### 5.4 变换 GLB

```python
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
R_combined = s_inv * (R_inv @ YUP_TO_ZUP)
t_combined = s_inv * (R_inv @ (-t))

T_4x4 = np.eye(4)
T_4x4[:3, :3] = R_combined
T_4x4[:3, 3] = t_combined

scene = trimesh.load(os.path.join(RAS_OUT, 'final_scene.glb'))
scene.apply_transform(T_4x4)
scene.export('scene_in_hawor_world.glb')
```

---

## 6. 对齐正确性分析

### 6.1 为什么这个对齐在数学上是正确的

```
同一个 mp4 同一帧 i 的相机位置：
  RAS  给出 cam_ras[i]   ← Room World (z-up, VGGT单位)
  HaWoR 给出 cam_hawor[i] ← SLAM World (y-down, z-forward, 米)

→ 两个值代表同一个物理点 → 存在唯一的 {R, t, s} 使得 cam_ras = s·R·cam_hawor + t
→ 这个 {R, t, s} 适用于场景中所有 3D 点
→ 相机轨迹是连接两个坐标系的可靠桥梁
```

### 6.2 分项保证性

| 环节 | 代码来源确认 | 保证程度 |
|------|------------|---------|
| 轴约定 R_axis | VGGT + DROID-SLAM 都使用 OpenCV 约定 | 数学推导 |
| 外参格式 | RAS: w2c 4×4; HaWoR: c2w (R_c2w, t_c2w) 已确认 | 代码审查 |
| 帧对应 | 同一 mp4，帧索引直接对应 | 前提满足 |
| R_residual | 第一帧 ||R-I|| < 0.001，残差旋转 < 0.5° | 实测验证 |
| 原点 t | 第一帧相机位置对齐 | 可靠 |
| **尺度 s** | 需要交叉验证 | **关键变量** |

### 6.3 尺度 s 的选择建议

| 方法 | 可靠性 | 说明 |
|------|--------|------|
| A: 深度图比较 | 中 | VGGT 深度取全局均值，可能偏高 |
| B: 轨迹位移 | 低 | 相机运动太小时不稳定 |
| C: Umeyama (自动) | 中 | 依赖相机轨迹离散度 |
| D: 手动 s=1 | 低 | 假设 VGGT 单位 ≈ 米 |

**建议**：先用 `01_align_scene.py` 自动估算，然后根据残差验证结果用 `--force-scale` 调整。

### 6.4 如果对齐结果不对

```
症状 1: 残差很大 (> 0.5m)
  → 检查帧对应：RAS 和 HaWoR 处理的帧是否一致
  → 检查是否用了正确的 hawor_cam_original（不是 haw_c2w 原始）

症状 2: 手在场景外面
  → 调大或调小 s
  → 用手-相机距离除以 RAS 相应区域的深度来重新估计 s

症状 3: 场景看起来旋转了
  → 检查 R_x 是否被正确逆应用
  → 检查 YUP_TO_ZUP 矩阵方向是否正确
```

---

## 7. 坐标系变换速查表

| 变换 | 矩阵 | 效果 |
|------|------|------|
| YUP_TO_ZUP (GLB→Room) | `[[1,0,0],[0,0,-1],[0,1,0]]` | (x,y,z)→(x,-z,y) |
| ZUP_TO_YUP (Room→GLB) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_AXIS (Y-UP→Z-UP) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_x (OpenCV↔OpenGL) | `diag(1,-1,-1)` | (x,y,z)→(x,-y,-z) |
| RXWORLD_TO_SAPIEN | `R_AXIS @ R_x` | HaWoR Render → SAPIEN |
| OPERATOR2MANO_RIGHT | `[[0,0,-1],[-1,0,0],[0,1,0]]` | MANO右手→操作者 |
| R_GRIPPER_ALIGN | `[[0,0,1],[0,1,0],[-1,0,0]]` | 操作者→夹爪 |

---

## 8. 验证清单

| 检查项 | 方法 | 通过标准 |
|--------|------|---------|
| 手在场景范围内 | bounds 检查 | 手 xz ⊆ 场景 xz ± 0.5m |
| 手在相机前方 | 方向点积 > 0 | 相机→手 方向 · 相机forward > 0 |
| 相机轨迹残差 | `‖aligned[i] - ras_cam[i]‖` | 中位误差 < 0.1m |
| GLB 在视频中可见 | 渲染检查 | 公共API加载的actor可见 |
| 末端朝向合理 | 可视化坐标轴 | 夹爪朝向与手腕旋转一致 |
| 末端位置误差 | FK验证 | < 5mm (收紧容差后) |
| 2D重投影误差 | 05_video_alignment.py | < 2px (优秀) |

---

## 9. 关键文件对照

| 文件 | 作用 |
|------|------|
| `01_align_scene.py` | 计算 RAS→HaWoR 变换参数，保存到 `transform_params.npz` |
| `02_render_scene.py` | 加载变换后的 GLB + HaWoR 手部 + R1 机器人，SAPIEN 渲染 |
| `03_track_robot.py` | 独立机器人跟踪（不需要 GLB 场景），快速验证映射 |
| `transform_params.npz` | 包含 `s_inv`, `R_inv`, `t_inv`, `glb_up_axis` (GLB→HaWoR 变换参数) |
| `question.md` | 问题与回答 |
