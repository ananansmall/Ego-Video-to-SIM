# 坐标变换审查与重构报告

> 日期：2026-08-02
> 状态：坐标变换链已修改为 001/002 链；代码冗余待清理

---

## 一、核心问题

### 背景

MANO 手部与 GLB 物体不在同一 SAPIEN 空间中，导致仿真中手部与物体位置不匹配。

根因：`grasp_hawor.py` 中的 `_mano_to_sapien` 和 `data_loader.py` 中的 GLB→SAPIEN 变换使用了**错误的旋转矩阵**：
- 旧代码使用 `R_x = diag(1, -1, -1)`（01/02 链）
- 正确应为 `R_x = I`（001/002 链，经验验证 HaWoR 和 RAS 相机坐标系一致）

### 变换链对比

| | 旧链（错误，01/02） | 新链（正确，001/002） |
|---|---|---|
| SLAM → OpenGL | `p_opengl = R_x @ p_slam`, `R_x = diag(1,-1,-1)` | `p_opengl = p_slam`（无需 R_x） |
| HaWoR → GLB | `p_glb = s * R_h2g @ p_opengl + t_h2g` | 同上 |
| GLB → SAPIEN | `p_sapien = R_AXIS @ p_glb` | 同上 |
| 合写（手部） | `p_sapien = R_AXIS @ R_x @ (s * R_h2g @ R_x @ slam + t_h2g)` ❌ | `p_sapien = R_AXIS @ (s * R_h2g @ Rx_hand @ slam + t_h2g)` ✓ |
| 合写（GLB） | `p_sapien = R_AXIS @ R_x @ (s_inv * R_inv @ v + t_inv)` ❌ | `p_sapien = R_AXIS @ (s_inv * R_h2g.T @ (v - t_h2g))` ✓ |

> 注意：手部使用 `Rx_hand = diag(1,-1,-1)`（HaWoR → OpenGL 翻转 Y/Z），GLB 使用 `R_x = I`（SLAM → OpenGL 无需翻转）。

---

## 二、已完成的文件修改

### 1. `physics_env.py` L119

```python
# 改前: R_x = np.diag([1.0, -1.0, -1.0])
# 改后: R_x = np.eye(3)  # 001 链: R_X = I
```

### 2. `data_loader.py`（两处：`load_glb_with_physics` 和 `compute_glb_ground_z`）

```python
# 改前:
s_inv = float(params['s_inv'])
R_inv = params['R_inv']
t_inv = params['t_inv']
vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv
vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T

# 改后:
s_inv = float(params['s_inv'])
R_h2g = params['R_hand_to_glb']
t_h2g = params['t_hand_to_glb']
vertices_hawor = s_inv * (R_h2g.T @ (vertices - t_h2g).T).T  # 注意: R_h2g.T
vertices_sapien = (R_AXIS @ vertices_hawor.T).T
```

### 3. `grasp_hawor.py` L1568 — `_mano_to_sapien`（setup 内嵌套定义）

```python
# 改前:
def _mano_to_sapien(pts_slam):
    return (RXWORLD_TO_SAPIEN @ pts_slam.T).T

# 改后:
def _mano_to_sapien(pts_slam):
    pts_glb = _tp_s * (_tp_R_h2g @ _tp_Rx @ pts_slam.T).T + _tp_t_h2g
    return (R_AXIS @ pts_glb.T).T
```

### 4. `grasp_hawor.py` L3046 — `_mano_to_sapien_v4`（run_v4_pipeline 内嵌套定义）

```python
# 改前:
def _mano_to_sapien_v4(pts_slam):
    return (RXWORLD_TO_SAPIEN @ pts_slam.T).T

# 改后:
def _mano_to_sapien_v4(pts_slam):
    pts_glb = s * (R_hand @ pts_slam.T).T + t
    return (R_AXIS @ pts_glb.T).T
```

### 5. `vis_stage_trajectories_left_v2.py`（三处逆变换）

| 函数 | 改前 | 改后 |
|------|------|------|
| `mano_sapien_to_glb` | `SCALE * R_H2G @ R_AXIS.T @ s + T_H2G` | `R_AXIS.T @ s`（因 002 链中 sapien = R_AXIS @ glb，逆变换即 R_AXIS.T） |
| `sapien_to_glb` | `SCALE * R_H2G.T @ (R_x @ R_AXIS.T @ s - T_H2G)` | `T_H2G + SCALE * R_H2G @ R_AXIS.T @ s` |
| `mano_palm_sapien` | `R_AXIS @ R_H2G.T @ (palm_glb - T_H2G) / SCALE` | `R_AXIS @ palm_glb` |

### 6. `diagnose_mano.py`（两处逆变换）

```python
# palm_sapien:  R_AXIS @ R_H2G.T @ (palm_glb - T_H2G) / SCALE
#     → 改后:   R_AXIS @ palm_glb

# s3_mano_glb:  SCALE * R_H2G @ R_AXIS.T @ s3_sapien + T_H2G
#     → 改后:   R_AXIS.T @ s3_sapien
```

### 7. `diagnose_full.py` L30

```python
# 改前:
def _mano_to_sapien(pts_slam):
    return (RXWORLD_TO_SAPIEN @ pts_slam.T).T

# 改后:
def _mano_to_sapien(pts_slam):
    pts_glb = SCALE * (R_hand @ pts_slam.T).T + T_H2G
    return (R_AXIS @ pts_glb.T).T
```

---

## 三、待处理的代码冗余问题

### 问题 1：`_mano_to_sapien` 两处嵌套定义，完全等价

- **L1568**（`setup()` 内）和 **L3046**（`run_v4_pipeline()` 内）定义了两个数学上完全相同的函数
- 只是读取的变量名不同（局部 vs 实例变量），但实例变量 `_mano_xform_*` 就是局部变量的拷贝
- **建议**：提取为实例方法 `def _mano_to_sapien(self, pts_slam)`，Stage 1/2/3/4 统一调用 `self._mano_to_sapien(joints)`

### 问题 2：与 `hand_track/common.py` 的 `_render_to_sapien` 重复

- `hand_track/common.py` L480 已实现 `_render_to_sapien(pts, R_h2g, t_h2g, Rx_hand, s)`，逻辑完全相同
- `grasp_hawor.py` 没有导入，而是自己重写
- **建议**：`from hand_track.common import _render_to_sapien, set_render_transform_params`，调用 `set_render_transform_params()` 初始化全局参数，然后直接调用无参数的 `_render_to_sapien(joints)`

### 问题 3：`hawor_cam_to_sapien_pose` 两份独立实现

- `hand_track/common.py` L508 和 `data_loader.py` L80 逻辑完全相同
- `grasp_hawor.py` 从 `data_loader.py` 导入
- **建议**：统一到 `hand_track/common.py`，`data_loader.py` 从那里 re-export

### 问题 4：transform_params 加载逻辑重复

- `grasp_hawor.py` L1542-1566 手动加载并存 6 个实例变量
- `hand_track/common.py` 的 `set_render_transform_params()` 做同样的事（存储到模块级全局变量）
- 且 `_render_to_sapien` / `hawor_cam_to_sapien_pose` 的无参数版本会自动使用这些全局变量
- **建议**：在 `grasp_hawor.py` 中调用 `set_render_transform_params(R_h2g, t_h2g, Rx_hand, s)`，然后使用无参数版本

### 问题 5：`R_x` / `RXWORLD_TO_SAPIEN` 符号冗余

- 001 链中 `R_x = I`，所以 `RXWORLD_TO_SAPIEN = R_AXIS @ I = R_AXIS`
- 代码中有的地方用 `R_AXIS`，有的用 `RXWORLD_TO_SAPIEN`，容易混淆
- **建议**：统一使用 `R_AXIS`，废弃 `RXWORLD_TO_SAPIEN` 别名

---

## 四、当前状态

- [x] 所有文件的坐标变换链已改为 001/002 链
- [x] `R_x = I` 已生效
- [x] GLB→SAPIEN 使用 `R_h2g.T @ (v - t_h2g)` 逆变换
- [x] 手部→SAPIEN 使用 `R_h2g @ Rx_hand @ slam + t_h2g` 前向变换
- [ ] 代码冗余清理（问题 1-5）
- [ ] test-stage3 验证运行
- [ ] CHANGE_LOG.md 更新
