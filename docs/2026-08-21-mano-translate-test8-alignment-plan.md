# 新抓取计划：MANO 预先平移 + Stage123 微调

> 日期: 2026-08-21
> 目标: 让机器人复现人手抓取，青线（机器人输出）贴合绿线（人手输入）
> 核心: **预先平移 MANO 轨迹**，使 Stage123 只做微调而非大幅扭曲

---

## 一、问题诊断：青线为什么失真

### 1.1 青线 = Stage1→Stage2→Stage3 全管线输出

```
Stage1  CMA-ES 搜索最优抓取位姿 → best_grasp.npz (pos + R)
Stage2  重建 F46-F62 过渡轨迹（用 smoothstep 插值）
Stage3  平滑优化 + 物理 rollout → grasp.log 的 fmid_pos
```

### 1.2 失真根因：Stage2 用直线"缝"两条不重合的轨迹

`stage2_reconstruct_mixin.py` L108-148 的位置重建逻辑：

```
F46~F49:  smoothstep 从 MANO[F46] 直线插值到 Stage1 的 F50 位置
F50~F56:  完全固定 = Stage1 的 F50 位置（7 帧不动）
F57~F62:  smoothstep 从 F50 直线插值回 MANO[F62]
```

关键代码：
```python
# L132: xy 直线插值
xy_interp = _lerp(pos_traj[f_start][:2], optimized_f50_pos[:2], s)
# L147: 回程也是直线
pos_interp[f] = optimized_f50_pos * (1 - s) + pos_traj[f] * s
```

**两条直线 + 中间 7 帧静止** = 青线走直线、MANO 曲线参数全丢。

### 1.3 为什么首尾位置被改

Stage1 搜出的 `optimized_f50_pos` 和 MANO 的 F50 位置差距很大（`xy_gap` 可能几十毫米）。Stage2 强行把 F46 拉到 F50、F62 又拉回 MANO，smoothstep 扭曲了 F46 和 F62 的 xy。

### 1.4 为什么姿态失真

L154-175 用 `grasp_R`（Stage1 姿态）做 slerp，F50-F54 完全用 Stage1 姿态覆盖 MANO 姿态。

### 1.5 根本矛盾

```
Stage1 搜出的抓取位姿  ≠  MANO 轨迹的 F50 位置
         ↓
Stage2 被迫用直线"缝"起来
         ↓
MANO 的曲线参数（位置+姿态）全丢 → 青线失真
```

---

## 二、解决方案：预先平移

### 2.1 核心思路

**在 Stage1 之前**，把整条 MANO 轨迹整体平移，让 MANO 的 F50 手指中点**本来就落在 Stage1 能搜到的抓取位置上**。

```
原始 MANO 轨迹
    ↓ 预先平移（恒定 delta，位置平移，姿态不变）
平移后 MANO 轨迹（F50 手指中点 ≈ 物体中心）
    ↓ Stage1 搜索
best_grasp.pos ≈ MANO 平移后 F50  →  xy_gap ≈ 0
    ↓ Stage2 重建
smoothstep 两端几乎重合 → 保留 MANO 曲线
    ↓ Stage3 微调
几毫米级优化，不再大幅扭曲
```

### 2.2 平移量计算

```
delta = 物体中心 - MANO_F50_手指中点
```

- **物体中心**：`7_vggt_omega` 场景的 `toy_0` 中心（GLB 坐标）
- **MANO_F50_手指中点**：MANO FK 后拇指尖+食指尖中点（GLB 坐标）
- **delta 是恒定向量**，加到整条 MANO 轨迹的每帧位置上
- **姿态完全不变**（不旋转、不缩放）

### 2.3 为什么这样能解决失真

| 问题 | 旧方案 | 新方案 |
|------|--------|--------|
| F50 位置 | Stage1 搜出 ≠ MANO F50 | 平移后 MANO F50 ≈ Stage1 搜出 |
| Stage2 插值 | 两端差距大 → 直线 | 两端几乎重合 → 保留曲线 |
| 首尾位置 | 被 smoothstep 扭曲 | 基本不变 |
| 姿态 | F50-F54 被 Stage1 覆盖 | MANO 姿态基本保留 |
| Stage3 优化 | 大幅调整 | 几毫米微调 |

---

## 三、实现步骤

### Step 1: 预先平移函数

**文件**: `grasp_hawor.py`（新增方法）

```python
def _pre_translate_mano(self, side, obj_center_glb):
    """预先平移 MANO 轨迹，使 F50 手指中点对齐物体中心。

    Args:
        side: 'left' 或 'right'
        obj_center_glb: 物体中心 (GLB 坐标, 3)

    Returns:
        delta: 平移向量 (3)
    """
    traj = self._mano_gripper_traj.get(side)
    if traj is None:
        return np.zeros(3)

    pos_traj = np.asarray(traj["pos"])
    j1 = np.asarray(traj["j1"])
    j2 = np.asarray(traj["j2"])
    mano_f50_mid = (j1[50] + j2[50]) / 2.0  # F50 手指中点 (GLB)

    delta = obj_center_glb - mano_f50_mid
    logger.info(f"[PreTranslate] {side}: MANO_F50_mid={mano_f50_mid.round(4)}, "
                f"obj_center={obj_center_glb.round(4)}, delta={delta.round(4)}")

    # 平移整条轨迹位置（姿态不变）
    self._mano_gripper_traj[side]["pos"] = pos_traj + delta
    return delta
```

### Step 2: 在 Stage1 之前调用

**文件**: `grasp_hawor.py` 主循环（Stage1 之前）

```python
# 获取物体中心 (GLB 坐标)
obj_center_glb = self._get_object_center_glb(target_obj_name)

# 预先平移 MANO 轨迹
self._pre_translate_mano(side, obj_center_glb)

# 然后正常跑 Stage1
self.cem_grasp_pose_optimize(...)
```

### Step 3: Stage1 保持原样

Stage1 仍然用 CMA-ES 搜索，但因为 MANO 已经平移，搜索起点就在物体附近，结果更准。

### Step 4: Stage2 保持原样

Stage2 的 smoothstep 插值逻辑不变。但因为 `xy_gap ≈ 0`，插值两端几乎重合，自然保留 MANO 曲线。

### Step 5: Stage3 保持原样

Stage3 仍然做平滑优化，但幅度很小（几毫米级）。

### Step 6: 物理参数对齐 test8

| 参数 | 当前 | test8 | 改动 |
|------|------|-------|------|
| 夹爪摩擦 | 2.0 | 3.0 | `physics_env.py` GRIPPER_FRICTION |
| 物体密度 | 1000 | 2000 | `physics_env.py` OBJECT_DENSITY |
| 夹爪 K | 8000 | 5000 | `physics_env.py` GRIPPER_STIFFNESS |
| 夹爪 D | 200 | 500 | `physics_env.py` GRIPPER_DAMPING |
| 物体摩擦 | 0.5 | 3.0-5.0 | `data_loader.py` 物体创建 |
| 夹爪闭合 | compute_gripper_qpos | 0.012 | `physics_env.py` |

---

## 四、验证标准

1. **xy_gap < 5mm**：Stage2 日志中 `xy_gap` 应接近 0
2. **青线贴合绿线**：轨迹图中 F46-F62 段青线应跟随绿线曲线
3. **首尾位置不变**：F46 和 F62 的青线位置 ≈ 绿线位置
4. **lift > 3cm**：物体被成功抓起
5. **drift < 2cm**：物体不掉落

---

## 五、与旧计划的区别

| 维度 | 旧计划（v4 pipeline） | 新计划 |
|------|----------------------|--------|
| MANO 轨迹 | 原始，不修改 | 预先平移到物体位置 |
| Stage1 | CMA-ES 搜索（起点远） | CMA-ES 搜索（起点近） |
| Stage2 | 直线缝接（失真） | 保留 MANO 曲线 |
| Stage3 | 大幅优化 | 微调 |
| 物理参数 | 默认 | 对齐 test8 |
| 核心改动 | 优化器 | 预先平移 |

---

## 六、注意事项

1. **平移是恒定的**：整条轨迹加同一个 delta，不逐帧变化
2. **姿态不变**：只平移位置，不旋转
3. **平移在 Stage1 之前**：确保 Stage1 搜索起点正确
4. **平移量可逆**：保存 delta，便于调试和回退
5. **物体中心用 GLB 坐标**：与 MANO 轨迹同坐标系
