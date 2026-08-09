# test9 MANO 轨迹跟随 — 开发总结

## 1. 任务目标

在 SAPIEN 物理仿真中，让夹爪跟随 MANO 手部轨迹，并比较**命令轨迹（平滑后）**与**物理实际跟踪**之间的偏差。

- 输入: HaWoR 数据 (`pred_rot`, `pred_trans`, `pred_hand_pose`, `pred_valid`)
- 重定向映射: `svd_palm` 策略 (5 个 MANO 关节点 SVD 拟合手掌平面 + Gram-Schmidt 正交化)
- 输出: 位置/朝向/关节的跟踪误差统计 + 视频

## 2. 关键文件

| 文件 | 作用 |
|------|------|
| `/home/an/robot_world_ws/src/dex-retargeting/example/combination/05_gripper_test.py` | 主要测试脚本，`test9_mano_trajectory` 函数 |
| `/home/an/robot_world_ws/src/dex-retargeting/example/combination/hand_track/gripper_config.py` | `compute_mano_based_gripper_pose` (SVD 手掌平面映射) |
| `/home/an/robot_world_ws/src/do-as-i-do/retargeting/retargeting/pipeline/process_dataset.py` | do-as-i-do 的 MAD 速度阈值检测 + 插值方法 |
| `/home/an/robot_world_ws/src/do-as-i-do/retargeting/retargeting/utils/interp.py` | 上采样插值工具 |

## 3. 已解决的问题

### 3.1 帧数不匹配 (num_frames 硬编码)

**问题**: 代码用 `--num-frames 600` 硬编码，但数据实际帧数不同 (如 `/home/an/data/hawor/7` 仅 113 帧)。

**表现**: `frame_skip = max(1, 113 // 600) = 1`，帧 113+ 全部映射到数据索引 112，而 `pred_valid[112] = False`，导致大量无效帧。

**修复**: 加载数据后设置 `num_frames = n_frames_data`，移除 `frame_skip`。

```python
# 05_gripper_test.py (L2055-2057)
n_frames_data = hawor_data['pred_trans'].shape[0]
num_frames = n_frames_data  # 使用数据实际帧数
```

### 3.2 无效帧导致 PD 残余力跳动 (帧 128 的 40.7° 朝向误差)

**问题**: 旧代码对无效帧"保持上一帧驱动目标"，但 SAPIEN 物理仿真继续步进，PD 控制器在残留力作用下产生朝向跳动。

**修复**: 改为存储 `np.nan`，然后在循环后做插值补齐 (do-as-i-do 风格):

```python
# 预计算循环: 无效帧存 NaN
for i in range(num_frames):
    if not hawor_data["pred_valid"][data_idx]:
        raw_pos[i] = np.nan; raw_quat[i] = np.nan; raw_joint[i] = np.nan
        continue
    # ... 计算夹爪位姿 ...
    raw_valid[i] = True

# 插值补齐: 位置线性插值 + 朝向 SLERP
_good = np.where(raw_valid)[0]
_bad = np.where(~raw_valid)[0]
for d in range(3):
    raw_pos[_bad, d] = np.interp(_bad, _good, raw_pos[_good, d])
# ... SLERP for quaternions ...
```

### 3.3 帧 0 初始化偏移

**问题**: 机器人初始化到 `_init_pos + (0,0,0.05)`，但帧 0 命令是 `_init_pos`，5cm 偏移导致 415.7mm 帧 0 误差。

**修复**: 使用第一个有效帧的平滑目标初始化，移除 Z 偏移。

### 3.4 夹爪漂移

**问题**: warmup 使用 `load_gripper` 默认 PD (K=5000, D=500)，重力补偿力与 PD 交互产生稳态误差。

**修复**: 初始化时对齐第一个有效帧的平滑目标 (位置+朝向+关节值)。

### 3.5 PD 参数调优

| 参数 | 旧值 | 新值 | 效果 |
|------|------|------|------|
| 位置 K/D | 5000/1000 (过阻尼 ζ≈7) | 10000/200 (临界阻尼 ζ≈1) | 响应速度提升 5 倍 |
| 朝向 K/D | 5000/1000 | 5000/150 | 稳定跟踪 |
| 夹爪 K/D | 5000/1000 | 5000/150 | 稳定跟踪 |

## 4. 剩余问题

### 4.1 MANO 参数突变 (帧 552→551)

**现象**:
```
#1 帧552→551: Δhp=3.3514 | Δpos=670.2mm | Δgq=170.8° | hp_norm=0.000
```

**根因**: `hp_norm=0.000` 说明 frame 551 的 `hand_pose` 全零——HaWoR 跟踪器输出损坏，但 `pred_valid=True`。当前插值只处理 `pred_valid=False` 的帧，漏掉了这种"被标记为有效但数据损坏"的帧。

**do-as-i-do 的解决方案**: 使用 MAD (Median Absolute Deviation) 速度阈值检测，不依赖 `pred_valid`:

```python
# do-as-i-do 的检测参数
CLEAN_CONFIG = {
    "pos": dict(k_mad=8.0, v_cap=0.20, window=31),   # 位置速度阈值 0.20 m/fr
    "rot": dict(k_mad=8.0, v_cap=0.40, window=31),   # 朝向速度阈值 0.40 rad/fr
}
```

670.2mm 的跳跃远超 0.20 m/fr 阈值，MAD 方法能检测到。

**三种方法对比**:

| 方法 | 能处理帧 551 吗？ | 原因 |
|------|-------------------|------|
| 当前 pred_valid 插值 | ❌ 不能 | `pred_valid=True` 所以跳过 |
| do-as-i-do MAD 检测 | ✅ 能 | 检测帧间速度，不依赖标记 |
| SG + 高斯 Slerp 平滑 | ⚠️ 部分能 | 180° 翻转导致 Slerp 不稳定 |

### 4.2 数据段边界朝向误差 (帧 266 的 58.3°)

**现象**: 数据有 2 个有效段 (帧 56-267 和帧 326-551)，中间 268-325 帧无效。SLERP 插值在 58 帧间隙内创建了平滑过渡，但两个段的手部姿态完全不同 (帧 268→267: Δhp=1.0756, Δgq=175.1°)，物理跟踪跟不上。

**本质**: 这不是算法问题，而是数据本身的问题——两个段的手部姿态发生了真实的大幅度变化。SLERP 插值在间隙内平滑过渡，但 PD 控制器在段边界处仍有较大跟踪误差。

## 5. 性能指标对比

| 指标 | 修复前 (121_C5_CellPhone_161deg) | 修复后 | 改善 |
|------|----------------------------------|--------|------|
| 有效帧数 | 438/600 | 600/600 | +37% |
| 位置误差 avg | 3.8mm | 2.8mm | -26% |
| 位置误差 max | 22.3mm (帧212) | 20.6mm (帧506) | -8% |
| 朝向误差 avg | 0.9° | 0.9° | 持平 |
| 朝向误差 max | 40.7° (帧128) | 58.3° (帧266) | 段边界问题 |
| 关节误差 avg | 1.0mm | 0.9mm | -10% |

## 6. 下一步建议

1. **添加 MAD 速度阈值检测**: 在基本插值之后，增加 do-as-i-do 风格的 MAD 检测+插值，捕获 `pred_valid` 漏标的损坏帧 (如帧 551)
2. **处理段边界大突变**: 在插值后检测帧间变化，如果 Δpos > 阈值或 Δgq > 阈值，对该段做额外平滑
3. **数据质量改进**: 建议在数据预处理阶段 (如 `process_dataset.py`) 修复 `pred_valid` 标记，确保损坏帧被正确标记