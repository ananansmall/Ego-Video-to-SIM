# GLB → HaWoR 对齐操作指南

## 0. 实际文件路径与数据概况

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

### 实测关键数据

```
RAS (Room World, z-up):
  Camera [0]:    position ≈ [0, 0, 0],  R_w2c ≈ I  (||R-I||=0.0005)
  点云范围:       x[-0.78, 1.33],  y[-0.83, 0.35],  z[0.01, 1.52]
  点云中心:       [0.04, -0.06, 0.84]
  GLB 范围(y-up): x[-0.25, 0.39],  y[0.44, 1.04],  z[-0.24, -0.01]

HaWoR (SLAM World, y-down, z-forward):
  pred_trans:     在原始 SLAM World (OpenCV 约定, 米制)
  右手范围:       x[-0.026, -0.005], y[0.000, 0.019], z[-0.006, 0.038]
  手距相机:       ~0.043m
  R_c2w[0]:       已乘 R_x=diag(1,-1,-1) 后保存 (||R-I||=2.828)
```

### 关键发现

> **HaWoR 的 `hawor_results_0_113.npz` 中 `R_c2w` 和 `t_c2w` 已经应用了 R_x 翻转 (OpenCV→OpenGL)，不是原始 SLAM World。`pred_trans` 仍然是原始 SLAM World。**

验证方法：`R_c2w[0]` 看是否是 `[[1,0,0],[0,-1,0],[0,0,-1]]` 附近。如果是，说明已应用 R_x。

---

## 1. 目标

将 RAS 的 `final_scene.glb` 变换到 **HaWoR SLAM World 坐标系**（即 `pred_trans` 所在的坐标系）：

| 属性 | RAS GLB（当前） | 目标坐标系（SLAM World） |
|------|----------------|------------------------|
| 朝上轴 | +Y | **-Y**（即 y-down = OpenCV 约定） |
| 朝前轴 | -Z | **+Z** |
| 原点 | 地板 z=0, 场景中心 | 第一帧相机位置 ≈ (0,0,0) |
| 尺度 | VGGT 单位 | 米 (Metric3D) |

---

## 2. 对齐原理

### 2.1 变换链

```
RAS GLB (y-up, VGGT单位)
    ↓ 步骤 A: y-up → z-up
RAS Room World (z-up, VGGT单位)
    ↓ 步骤 B: 逆对齐 → 统一到 SLAM World
HaWoR SLAM World (y-down, z-forward, 米制)  ← 与 pred_trans 同一坐标系
```

### 2.2 各步骤

#### 步骤 A：GLB y-up → Room World z-up

`main.py` 导出 GLB 时做了 `z-up → y-up` 变换（`[[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]`），逆变换为 `y-up → z-up`：

```python
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
# 效果: (x, y, z) → (x, -z, y)
```

#### 步骤 B：Room World → SLAM World

三个坐标系的轴约定都是 OpenCV，但世界坐标系朝向不同：

```
Room World (z-up):         SLAM World (y-down, z-forward):
    +Z (up)                     -Y (up)
    |                            |
    +---- +Y                    +---- +Z (forward)
   /                            /
  +X                           +X (right)
```

从 Room World z-up 到 SLAM World OpenCV 的轴转换：

```python
R_axis = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
# 效果: (x, y, z) → (x, z, -y)
# 即将 Room z-up 转为 SLAM y-down, z-forward
```

完整逆变换公式：

```
p_slam = (1/s) * R_total^T @ (p_room - t)

其中:
  R_total = R_residual @ R_axis      (旋转部分)
  s       = 尺度因子                  (VGGT单位 / 米)
  t       = 原点偏移                  (Room World → SLAM World)
```

### 2.3 参数估计

| 参数 | 估计方法 | 保证程度 |
|------|---------|---------|
| `R_axis` | **已知**：`[[1,0,0],[0,0,1],[0,-1,0]]` |   数学推导 |
| `R_residual` | 两个系统第一帧外参都 ≈ I → 残差 ≈ I |   可靠 |
| `t` | `t = RAS_cam[0] - s * R_total @ HaWoR_cam_original[0]` |   第一帧对齐 |
| `s` | 深度图比较 或 轨迹位移比例 |   **需要交叉验证** |

---

## 3. 操作步骤（在终端逐段执行）

### 3.1 加载数据

```python
import numpy as np
import cv2
import trimesh
from glob import glob
import os

# ===== RAS 侧 =====
RAS_OUT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/my_7mp4_result'

ext_files = sorted(glob(os.path.join(RAS_OUT, 'extrinsics', '*.txt')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
ras_extrinsics = []
for f in ext_files:
    ext = np.loadtxt(f)
    if ext.shape == (3, 4):
        ext = np.vstack([ext, [0, 0, 0, 1]])
    ras_extrinsics.append(ext)
ras_extrinsics = np.array(ras_extrinsics)

# RAS 相机位置：外参是 w2c，cam_pos = -R_w2c^T @ t_w2c
ras_cam = np.array([-e[:3,:3].T @ e[:3,3] for e in ras_extrinsics])

print(f"RAS: {len(ras_cam)} 帧, cam[0]={ras_cam[0]}")
print(f"RAS cam range: x[{ras_cam[:,0].min():.4f},{ras_cam[:,0].max():.4f}] "
      f"y[{ras_cam[:,1].min():.4f},{ras_cam[:,1].max():.4f}] "
      f"z[{ras_cam[:,2].min():.4f},{ras_cam[:,2].max():.4f}]")

# ===== HaWoR 侧 =====
HAWOR_RES = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7/reconstruction/hawor_results_0_113.npz'

hawor = dict(np.load(HAWOR_RES, allow_pickle=True))
hawor_t_c2w = hawor['t_c2w']          # 已乘 R_x，OpenGL 约定
hawor_R_c2w = hawor['R_c2w']          # 已乘 R_x，OpenGL 约定
pred_trans = hawor['pred_trans']       # 原始 SLAM World (OpenCV 约定，米制)
pred_valid = hawor['pred_valid']

# 恢复原始 SLAM World 的相机位置 (逆 R_x)
R_x = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
hawor_cam_original = (np.array([R_x @ t for t in hawor_t_c2w]))
hawor_R_c2w_original = np.array([R_x @ R for R in hawor_R_c2w])

print(f"\nHaWoR: {len(hawor_cam_original)} 帧, cam_original[0]={hawor_cam_original[0]}")
print(f"HaWoR cam_original range: x[{hawor_cam_original[:,0].min():.4f},{hawor_cam_original[:,0].max():.4f}] "
      f"y[{hawor_cam_original[:,1].min():.4f},{hawor_cam_original[:,1].max():.4f}] "
      f"z[{hawor_cam_original[:,2].min():.4f},{hawor_cam_original[:,2].max():.4f}]")

# 手部位置 (原始 SLAM World)
right_hand = pred_trans[1, pred_valid[1]]
print(f"\nRight hand range: x[{right_hand[:,0].min():.4f},{right_hand[:,0].max():.4f}] "
      f"y[{right_hand[:,1].min():.4f},{right_hand[:,1].max():.4f}] "
      f"z[{right_hand[:,2].min():.4f},{right_hand[:,2].max():.4f}]")

# 帧对应：取交集 (RAS 20帧, HaWoR 113帧)
common = list(range(min(len(ras_cam), len(hawor_cam_original))))
print(f"\n共同帧: {len(common)} (0 到 {common[-1]})")
```

### 3.2 计算轴约定旋转 R_axis

```python
# 轴约定: Room World z-up → SLAM World y-down, z-forward
# (x_room, y_room, z_room) → (x_slam, z_slam, -y_slam)
R_axis = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0,-1, 0]
], dtype=np.float64)

print("R_axis (Room z-up → SLAM World):")
print(R_axis)
print(f"R_axis 作用: (x,y,z) → (x, z, -y)")
print(f"  +Z (up) → -Y (up=down), +Y → +Z (forward)")
```

### 3.3 计算残差旋转 R_residual

```python
# 检查两个系统第一帧相机朝向是否一致
print(f"RAS 第一帧 R_w2c: ||R-I|| = {np.linalg.norm(ras_extrinsics[0,:3,:3] - np.eye(3)):.6f}")
print(f"HaWoR 第一帧 R_c2w_original: ||R-I|| = {np.linalg.norm(hawor_R_c2w_original[0] - np.eye(3)):.6f}")

# 如果两个值都很小 (< 0.1)，说明第一帧相机朝向一致，R_residual ≈ I
if np.linalg.norm(ras_extrinsics[0,:3,:3] - np.eye(3)) < 0.1 and \
   np.linalg.norm(hawor_R_c2w_original[0] - np.eye(3)) < 0.1:
    R_residual = np.eye(3)
    print("✓ 两个系统第一帧朝向一致，R_residual = I")
else:
    # Umeyama 估计残差
    src_pts = (R_axis @ hawor_cam_original[common].T).T
    dst_pts = ras_cam[common]
    src_c = src_pts - src_pts.mean(0)
    dst_c = dst_pts - dst_pts.mean(0)
    cov = dst_c.T @ src_c / len(src_pts)
    U, _, VH = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[2,2] = -1
    R_residual = U @ S @ VH
    angle = np.degrees(np.arccos(np.clip((np.trace(R_residual)-1)/2, -1, 1)))
    print(f"R_residual 旋转角度: {angle:.1f}°")

R_total = R_residual @ R_axis
print(f"R_total = R_residual @ R_axis:")
print(R_total)
```

### 3.4 估计尺度 s（选择一种方法）

#### 方法 A：深度图比较（推荐）

```python
# 比较 RAS 深度图均值 与 HaWoR 手部到相机的距离
depth_0 = cv2.imread(os.path.join(RAS_OUT, 'depth', '0.png'), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
ras_mean_depth = depth_0[depth_0 > 0].mean()

hawor_hand_dist = np.linalg.norm(right_hand[0] - hawor_cam_original[0])

s_depth = ras_mean_depth / hawor_hand_dist
print(f"\n=== 方法 A: 深度图尺度 ===")
print(f"  RAS 平均深度: {ras_mean_depth:.4f} (VGGT单位)")
print(f"  HaWoR 手-相机距离: {hawor_hand_dist:.4f} m")
print(f"  s_depth = {s_depth:.4f}")
```

#### 方法 B：轨迹位移比例（参考）

```python
# 比较相机在各自坐标系中的位移
ras_disp = np.linalg.norm(ras_cam[common[-1]] - ras_cam[common[0]])
hawor_disp = np.linalg.norm(hawor_cam_original[common[-1]] - hawor_cam_original[common[0]])
s_traj = ras_disp / hawor_disp if hawor_disp > 1e-6 else 1.0

print(f"\n=== 方法 B: 轨迹尺度 ===")
print(f"  RAS 相机位移: {ras_disp:.4f} (VGGT单位)")
print(f"  HaWoR 相机位移: {hawor_disp:.4f} m")
print(f"  s_traj = {s_traj:.4f}")
```

#### 选择尺度

```python
# 选择方法 A 或 B，或者手动指定
# s = s_depth   # 方法 A
# s = s_traj    # 方法 B
s = 1.0         # 手动指定（如果知道 VGGT 单位 ≈ 米）

print(f"\n最终使用尺度 s = {s:.4f}")
print(f"  p_ras = {s:.4f} * R_total @ p_hawor + t")
print(f"  p_hawor = {1/s:.4f} * R_total^T @ (p_ras - t)")
```

### 3.5 计算平移 t（原点对齐）

```python
# 正变换: p_ras = s * R_total @ p_hawor + t
# 对齐第一帧相机位置: RAS_cam[0] = s * R_total @ HaWoR_cam[0] + t
t = ras_cam[0] - s * (R_total @ hawor_cam_original[0])

# 逆变换参数 (用于 GLB)
s_inv = 1.0 / s
R_inv = R_total.T
t_inv = -s_inv * (R_inv @ t)

print(f"\n对齐参数:")
print(f"  正变换 (HaWoR→RAS): p_ras = {s:.4f} * R_total @ p_hawor + t")
print(f"  逆变换 (RAS→HaWoR): p_hawor = {s_inv:.6f} * R_inv @ p_ras + t_inv")
print(f"  t = {t}")
print(f"  t_inv = {t_inv}")
```

### 3.6 残差验证

```python
# 变换 HaWoR 相机到 RAS Room World，与 RAS 相机位置比较
aligned_hawor = s * (R_total @ hawor_cam_original.T).T + t
errors = np.linalg.norm(aligned_hawor[common] - ras_cam[common], axis=1)

print(f"\n=== 残差验证 ===")
print(f"  共同帧数: {len(common)}")
print(f"  对齐误差: mean={errors.mean():.6f}, median={np.median(errors):.6f}, max={errors.max():.6f}")
print(f"  每帧误差: {np.array2string(errors, precision=4, suppress_small=True)}")

if np.median(errors) < 0.1:
    print(f"  ✓ 中位误差 < 0.1m，对齐可靠")
elif np.median(errors) < 0.5:
    print(f"  ⚠ 中位误差 0.1-0.5m，可能需要深度图验证")
else:
    print(f"  ✗ 中位误差 > 0.5m，对齐不可靠！请检查帧对应和尺度")
```

### 3.7 变换 GLB

```python
# 步骤 A: GLB y-up → Room World z-up
YUP_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

# 步骤 B: Room World → SLAM World
# p_hawor = s_inv * R_inv @ (p_room - t)
# 对 GLB 顶点: p_slam = s_inv * R_inv @ (YUP_TO_ZUP @ p_glb - t)

R_combined = s_inv * (R_inv @ YUP_TO_ZUP)
t_combined = s_inv * (R_inv @ (-t))

T_4x4 = np.eye(4)
T_4x4[:3, :3] = R_combined
T_4x4[:3, 3] = t_combined

print(f"\n=== 变换矩阵 (4x4) ===")
print(T_4x4)

# 加载 GLB 并应用变换
glb_path = os.path.join(RAS_OUT, 'final_scene.glb')
scene = trimesh.load(glb_path)
scene.apply_transform(T_4x4)

# 保存
OUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/aligned_output'
os.makedirs(OUT_DIR, exist_ok=True)
output_glb = os.path.join(OUT_DIR, 'scene_in_hawor_world.glb')
scene.export(output_glb)

print(f"\n变换后 GLB 已保存到: {output_glb}")
print(f"变换后场景范围 (SLAM World, 米制):")
print(f"  x: [{scene.bounds[0,0]:.4f}, {scene.bounds[1,0]:.4f}]")
print(f"  y: [{scene.bounds[0,1]:.4f}, {scene.bounds[1,1]:.4f}]")
print(f"  z: [{scene.bounds[0,2]:.4f}, {scene.bounds[1,2]:.4f}]")
```

### 3.8 验证空间关系

```python
# 检查变换后的场景范围是否与手部位置匹配
right_hand = pred_trans[1, pred_valid[1]]

print(f"\n=== 空间验证 ===")
print(f"右手范围 (SLAM World, 米):")
print(f"  x: [{right_hand[:,0].min():.4f}, {right_hand[:,0].max():.4f}]")
print(f"  y: [{right_hand[:,1].min():.4f}, {right_hand[:,1].max():.4f}]")
print(f"  z: [{right_hand[:,2].min():.4f}, {right_hand[:,2].max():.4f}]")

print(f"\n场景范围 (SLAM World, 米):")
print(f"  x: [{scene.bounds[0,0]:.4f}, {scene.bounds[1,0]:.4f}]")
print(f"  y: [{scene.bounds[0,1]:.4f}, {scene.bounds[1,1]:.4f}]")
print(f"  z: [{scene.bounds[0,2]:.4f}, {scene.bounds[1,2]:.4f}]")

# 检查手是否在场景内
hand_in_x = (right_hand[:,0].min() >= scene.bounds[0,0] - 0.5 and
             right_hand[:,0].max() <= scene.bounds[1,0] + 0.5)
hand_in_y = (right_hand[:,1].min() >= scene.bounds[0,1] - 0.5 and
             right_hand[:,1].max() <= scene.bounds[1,1] + 0.5)
hand_in_z = (right_hand[:,2].min() >= scene.bounds[0,2] - 0.5 and
             right_hand[:,2].max() <= scene.bounds[1,2] + 0.5)

if hand_in_x and hand_in_y and hand_in_z:
    print("\n✓ 手部在场景范围内（含 0.5m 容差）")
else:
    print(f"\n⚠ 手部可能超出场景范围")
    print(f"  hand_in_x={hand_in_x}, hand_in_y={hand_in_y}, hand_in_z={hand_in_z}")
```

---

## 4. 对齐正确性分析

### 4.1 为什么这个对齐在数学上是正确的

```
同一个 mp4 同一帧 i 的相机位置：
  RAS  给出 cam_ras[i]   ← Room World (z-up, VGGT单位)
  HaWoR 给出 cam_hawor[i] ← SLAM World (y-down, z-forward, 米)

→ 两个值代表同一个物理点 → 存在唯一的 {R, t, s} 使得 cam_ras = s·R·cam_hawor + t
→ 这个 {R, t, s} 适用于场景中所有 3D 点
→ 相机轨迹是连接两个坐标系的可靠桥梁
```

### 4.2 分项保证性

| 环节 | 代码来源确认 | 保证程度 |
|------|------------|---------|
| 轴约定 R_axis | VGGT + DROID-SLAM 都使用 OpenCV 约定 (x-right, y-down, z-forward) |   数学推导 |
| 外参格式 | RAS: w2c 4×4; HaWoR: c2w (R_c2w, t_c2w) 已确认 |   代码审查 |
| 帧对应 | 同一 mp4，帧索引直接对应 |   前提满足 |
| R_residual | 第一帧 ||R-I|| < 0.001，残差旋转 < 0.5° |   实测验证 |
| 原点 t | 第一帧相机位置对齐 |   可靠 |
| **尺度 s** | 需要交叉验证 |   **关键变量** |

### 4.3 尺度 s 的选择建议

| 方法 | 本数据结果 | 可靠性 |
|------|-----------|--------|
| A: 深度图比较 | s ≈ 20.6 |   VGGT 深度取全局均值，可能偏高 |
| B: 轨迹位移 | s ≈ 2.7 | ⚠️ 相机运动太小(0.03m)，不稳定 |
| C: 手动 s=1 | s = 1.0 | 假设 VGGT 单位 ≈ 米（官方声称） |

**建议**：先用 s=1 尝试，然后根据残差验证结果调整。如果手部明显不在场景内（太大或太小），调大或调小 s 直到空间关系合理。

### 4.4 验证清单

| 检查项 | 方法 | 阈值 |
|--------|------|------|
| ✓ 相机轨迹残差 | `||aligned[i] - ras_cam[i]||` | < 0.1m 可靠 |
| ✓ 手在场景范围内 | bounds 检查 | 手 xz ⊆ 场景 xz |
| ✓ 手在地板上方 | `hand_y < 0` (SLAM World y-down) | 物理合理 |
| ✓ 手在物体前面 | 深度比较 | 不穿模 |

### 4.5 如果对齐结果不对

```
症状 1: 残差很大 (> 0.5m)
  → 检查帧对应：RAS 和 HaWoR 处理的帧是否一致
  → 检查是否用了正确的 hawor_cam_original（不是 haw_c2w 原始）

症状 2: 手在场景外面
  → 调大或调小 s
  → 用手-相机距离(0.043m)除以 RAS 相应区域的深度来重新估计 s

症状 3: 场景看起来旋转了
  → 检查 R_x 是否被正确逆应用
  → 检查 YUP_TO_ZUP 矩阵方向是否正确
```

---

## 5. 后续：在 HaWoR 渲染器中查看

变换后的 GLB 在 SLAM World (y-down, z-forward)。HaWoR 的 aitviewer 渲染器使用 OpenGL (y-up, z-backward)，需要再应用 R_x：

```python
R_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
# p_render = R_x @ p_slam  (用于可视化渲染)
```

---

## 6. 快速验证命令

所有步骤合并为一段可复制的代码块，在 `conda run -n ReplicateAnyScene python3` 中运行：

```python
# 一行放入 align_and_check.py 运行:
# cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src
# conda run -n ReplicateAnyScene python3 align_and_check.py

import numpy as np; import cv2; import trimesh; import os; from glob import glob

RAS_OUT='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/my_7mp4_result'
HAWOR_RES='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR/example/7/reconstruction/hawor_results_0_113.npz'
OUT_DIR='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/aligned_output'

# 1. Load
e=sorted(glob(os.path.join(RAS_OUT,'extrinsics','*.txt')),key=lambda x:int(os.path.basename(x).split('.')[0]))
ras_cam=np.array([(lambda m:(-m[:3,:3].T@m[:3,3]) if m.shape==(4,4) else (-(m:=np.vstack([m,[0,0,0,1]]))[:3,:3].T@m[:3,3]))(np.loadtxt(f)) for f in e])
h=dict(np.load(HAWOR_RES,allow_pickle=True))
Rx=np.array([[1,0,0],[0,-1,0],[0,0,-1]])
hc=np.array([Rx@t for t in h['t_c2w']])
hR=np.array([Rx@R for R in h['R_c2w']])
hp=h['pred_trans'][1,h['pred_valid'][1]]
n=min(len(ras_cam),len(hc))

# 2. Params
R_axis=np.array([[1,0,0],[0,0,1],[0,-1,0]])
R_residual=np.eye(3)
R_total=R_residual@R_axis
s=1.0  # <-- 调整此值！
t=ras_cam[0]-s*(R_total@hc[0])
si,Ri,ti=1/s,R_total.T,-(1/s)*(R_total.T@t)

# 3. Verify
aligned=s*(R_total@hc[:n].T).T+t
errs=np.linalg.norm(aligned-ras_cam[:n],axis=1)
print(f'Scale s={s:.4f}, errors: mean={errs.mean():.4f}, median={np.median(errs):.4f}, max={errs.max():.4f}')
print(f'Hand  range: x[{hp[:,0].min():.3f},{hp[:,0].max():.3f}] y[{hp[:,1].min():.3f},{hp[:,1].max():.3f}] z[{hp[:,2].min():.3f},{hp[:,2].max():.3f}]')

# 4. Transform GLB
Y2Z=np.array([[1,0,0],[0,0,-1],[0,1,0]])
T4=np.eye(4); T4[:3,:3]=si*(Ri@Y2Z); T4[:3,3]=si*(Ri@(-t))
scene=trimesh.load(os.path.join(RAS_OUT,'final_scene.glb'))
scene.apply_transform(T4)
os.makedirs(OUT_DIR,exist_ok=True)
scene.export(os.path.join(OUT_DIR,'scene_in_hawor_world.glb'))
print(f'GLB saved. Bounds: [{scene.bounds[0]}, {scene.bounds[1]}]')
print(f'Hand in scene? x: {hp[:,0].min()>=scene.bounds[0,0]-0.5 and hp[:,0].max()<=scene.bounds[1,0]+0.5}, z: {hp[:,2].min()>=scene.bounds[0,2]-0.5 and hp[:,2].max()<=scene.bounds[1,2]+0.5}')
```

---

## 附录：坐标系变换速查表

| 变换 | 矩阵 | 效果 |
|------|------|------|
| YUP_TO_ZUP (GLB→Room) | `[[1,0,0],[0,0,-1],[0,1,0]]` | (x,y,z)→(x,-z,y) |
| ZUP_TO_YUP (Room→GLB) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_axis (Room→SLAM) | `[[1,0,0],[0,0,1],[0,-1,0]]` | (x,y,z)→(x,z,-y) |
| R_x (OpenCV→OpenGL) | `diag(1,-1,-1)` | (x,y,z)→(x,-y,-z) |

### 坐标系一览

```
ReplicateAnyScene:
  VGGT World (任意) ──R,t──→ Room World (z-up, VGGT单位) ──z2y──→ GLB (y-up)

HaWoR:
  SLAM World (y-down, z-forward, 无尺度) ──scale──→ SLAM World (米制)
  SLAM World pred_trans (米制, y-down) ──R_x──→ R_c2w/t_c2w 保存 (y-up, z-backward)

对齐路径:
  RAS GLB (y-up) ──y2z──→ Room World (z-up) ──逆对齐──→ SLAM World (y-down, z-forward, 米制)
```