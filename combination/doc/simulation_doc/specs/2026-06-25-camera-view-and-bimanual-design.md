# SAPIEN 04 相机视角修复 + 双手逻辑设计

**日期**: 2026-06-25
**作者**: 协作设计 (用户 + assistant)
**状态**: 待用户审查

## 1. 问题分析

### 1.1 用户反馈

1. "之前的相机视角是对的，修改回来相机视角又发生了问题" — FLIP_Z=True 视角对，FLIP_Z=False 视角错
2. "上帝视角，gui这些都有运行吗，桌面还有吗？为什么变成镜像回来之后有些东西都没有了"
3. "自动检测手能不能添加一个双手的逻辑？"

### 1.2 诊断结果（基于实际 HaWoR 数据）

| 元素 | 位置 (SAPIEN, FLIP_Z=False) |
|---|---|
| 相机 | [0.004, -0.001, +0.004] |
| 相机 forward | [0.005, -1.0, +0.006] (主要看 -Y 方向, 略朝上) |
| 手腕质心 | [-0.014, -0.012, +0.009] |
| 物体 Z 范围 | [+0.010, +0.083] (正 Z, 地上) |
| 桌面顶部 Z | +0.036 |
| 机械臂基座 | [-0.014, -0.012, +0.709] (手腕正上方 0.70m) |

**核心问题不是 Z 翻转，而是机械臂基座位置**：
- 相机离手腕只有 2.2cm（HaWoR 重建的真实第一人称轨迹）
- 机械臂基座放在手腕质心正上方 0.70m，从上方延伸到手腕
- 机械臂充满整个画面（fpv 视频中机械臂银灰色像素占 91%，桌面颜色像素 0.00%）
- 桌面物理上存在（topdown 视频能检测到 2-3% 桌面颜色），但 fpv 视角被机械臂完全挡住

### 1.3 Z 翻转的真相

| | FLIP_Z=True (之前) | FLIP_Z=False (当前) |
|---|---|---|
| 相机 Z | -0.004 (朝下看) | +0.004 (朝上看) |
| 物体 Z | [-0.083, -0.010] (地下!) | [+0.010, +0.083] (地上) |
| 桌面顶部 Z | -0.036 | +0.036 |
| 机械臂基座 Z | -0.709 | +0.709 |
| GLB 镜像 | 是 (improper 变换 det=-1) | 否 (proper) |
| 物理正确性 | 物体在地下，物理 bug | 物体在正 Z，物理正确 |

Z 翻转把整个场景翻转：相机从朝上看变朝下看，机械臂从上方变下方。视觉感受不同源于机械臂 mesh 不对称（连杆朝下延伸，从上看稀疏，从下看密集）。但 Z 翻转有物理 bug（物体在地下）和镜像 bug（GLB improper 变换）。

### 1.4 GUI/topdown 现状

- `--viewer` 默认 False，命令没指定 → 无 GUI
- `--view topdown` 默认 fpv → 默认 fpv 视角
- topdown 视频是诊断时单独生成的

### 1.5 双手检测现状

`_detect_hand_idx` (line 463-491)：两只手都存在时**默认左手 (idx=0)**，不支持双手。

---

## 2. 任务1：移动机械臂基座到相机视野外

### 2.1 修复方案

基座位置从手腕正上方改为相机后方：

| | 当前 | 修复后 |
|---|---|---|
| 基座位置 | 手腕质心 + [0, 0, +0.70] | 手腕质心 + [0, +0.30, +0.55] |
| 距手腕最大 | 0.710m | 0.644m (< 臂展 0.713m ✓) |
| 基座相对相机 angle | 88.8° (边缘) | 117.1° (后方) |
| 机械臂路径在视野内 | 100% (垂直下方) | ~10% (仅手腕附近) |

机械臂路径分析（方案A）：
- t=0.0 到 t=0.9: angle > 100° (视野外，相机后方)
- t=1.0 (手腕): angle=60.3° (视野边缘)

### 2.2 代码改动

**常量修改** (line 106-108):
```python
# 改前
COMFORTABLE_REACH = 0.70
COMFORT_TARGET_IN_BASE = np.array([0.25, 0.0, -0.55])
BASE_TRACKING_RANGE = 0.0

# 改后
COMFORTABLE_REACH = 0.55  # 降低高度, 让基座在相机后方时仍在臂展内
BASE_OFFSET_Y = 0.30      # 基座 Y 偏移到相机后方 (相机看 -Y, 基座在 +Y)
COMFORT_TARGET_IN_BASE = np.array([0.25, -0.30, -0.50])  # 舒适目标点跟随基座偏移
BASE_TRACKING_RANGE = 0.0
```

**`_compute_optimal_fixed_base`** (line 1574-1575):
```python
# 改前
arm_base_pos = centroid.copy()
arm_base_pos[2] += COMFORTABLE_REACH

# 改后
arm_base_pos = centroid.copy()
arm_base_pos[1] += BASE_OFFSET_Y  # 相机后方
arm_base_pos[2] += COMFORTABLE_REACH
```

**`_compute_fixed_base_clusters`** (line 1647-1648, 1671-1672): 同样添加 `base_pos[1] += BASE_OFFSET_Y`

### 2.3 验证方法

1. 运行 `04_physics_simulation.py --hawor-dir ... --ras-dir ... --fast-collision`
2. 抽取视频第一帧，统计：
   - 桌面颜色像素 > 5% (当前 0.00%)
   - 机械臂银灰色像素 < 50% (当前 91%)
3. 确认物体在正 Z (FLIP_Z=False 不变)
4. 确认 GLB 不镜像 (FLIP_Z=False 不变)

---

## 3. 任务2：双手同时驱动两个机械臂

### 3.1 设计方案

支持三种模式：
- **单手 (现有)**: `--hand-idx 0` 或 `--hand-idx 1`，加载单 URDF
- **双手 (新增)**: `--hand-idx both` 或自动检测到双手时，加载双 URDF (FLOATING_LEFT_URDF + FLOATING_RIGHT_URDF)
- **自动检测 (改进)**: `--hand-idx -1` (默认)，检测到双手时进入双手模式

### 3.2 代码改动

#### 3.2.1 `_detect_hand_idx` 改为返回 list

```python
def _detect_hand_idx(hawor_path):
    """自动检测 HaWoR 数据中活跃的手

    返回:
        list[int]: 活跃手的索引列表 (空/单/双)
            []: 无法检测
            [0]: 仅左手
            [1]: 仅右手
            [0, 1]: 双手都活跃
    """
    cam_dir = Path(hawor_path) / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        return sorted(detected)
    return []
```

**注意**: 还需检查 `pred_valid` 是否真的有效（cam_space 有目录但 pred_valid 全 False 的情况）。在 main 中加载后验证。

#### 3.2.2 `load_hawor_data` 支持双手

新增 `load_hawor_data_multi` 函数，或修改 `load_hawor_data` 接受 `hand_idx` 为 int 或 list:

```python
def load_hawor_data(hawor_dir, hand_idx=0):
    """加载 HaWoR 手部重建数据

    Args:
        hand_idx: int (单手 0/1) 或 list[int] (双手 [0, 1])
    Returns:
        单手: dict (现有格式)
        双手: {0: dict, 1: dict}
    """
    # ... 加载原始数据 ...
    if isinstance(hand_idx, list):
        return {
            idx: {
                "pred_trans": pred_trans[idx],
                "pred_rot": pred_rot[idx],
                # ... 其他字段
            }
            for idx in hand_idx
        }
    # 单手: 现有逻辑
    return {"pred_trans": pred_trans[hand_idx], ...}
```

#### 3.2.3 `_setup_robot` 支持双 URDF

新增 `_setup_robots` 方法（或修改 `_setup_robot`）:

```python
def _setup_robots(self, scene, hand_indices, base_positions, base_quats):
    """加载单/双机械臂

    Args:
        hand_indices: [int] 单手或双手
        base_positions: {hand_idx: (3,)} 每只手的基座位置
        base_quats: {hand_idx: (4,)} 每只手的基座朝向

    Returns:
        {hand_idx: robot_info_dict}
    """
    robots = {}
    for hand_idx in hand_indices:
        prefix = "left" if hand_idx == 0 else "right"
        urdf = FLOATING_LEFT_URDF if hand_idx == 0 else FLOATING_RIGHT_URDF
        starting = LEFT_ARM_STARTING if hand_idx == 0 else RIGHT_ARM_STARTING
        # ... 加载 URDF, 设置 PD 驱动, 初始关节角, 夹爪摩擦 ...
        robots[hand_idx] = {
            "robot": robot,
            "joint_names": joint_names,
            "arm_joint_indices": arm_joint_indices,
            "gripper_idx1": gripper_idx1,
            "gripper_idx2": gripper_idx2,
            "ee_link": ee_link,
        }
    return robots
```

#### 3.2.4 `run_physics_tracking` 双手分支

新增 `run_bimanual_tracking` 方法（或在现有方法中加双手分支）:

```python
def run_bimanual_tracking(self, start_frame=0, num_frames=-1):
    """双手追踪: 同时驱动左右机械臂"""
    # [1/8] 加载双手数据
    hawor_data_multi = load_hawor_data(self.hawor_dir, hand_idx=[0, 1])
    # [2/8] 创建场景 + GLB (与单手相同)
    # [3/8] 加载双机械臂 (FLOATING_LEFT_URDF + FLOATING_RIGHT_URDF)
    #   - 左手基座: 左手腕质心 + [0, +0.30, +0.55]
    #   - 右手基座: 右手腕质心 + [0, +0.30, +0.55]
    # [4/8] 初始化双 Dex Retargeting (HandType.left + HandType.right)
    # [5/8] 初始化 RelaxedIK (左右两个 solver)
    # [6/8] 设置相机 (与单手相同)
    # [7/8] Warmup 双手
    # [8/8] 实时 IK 渲染: 每帧对两只手分别 retargeting + IK
```

#### 3.2.5 命令行参数

```python
parser.add_argument("--hand-idx", type=str, default="-1",
                    help="手: 0=左手, 1=右手, both=双手, -1=自动检测")
```

main 中处理:
```python
if args.hand_idx == "-1":
    detected = _detect_hand_idx(Path(args.hawor_dir))
    hand_indices = detected if detected else [0]
elif args.hand_idx == "both":
    hand_indices = [0, 1]
else:
    hand_indices = [int(args.hand_idx)]

if len(hand_indices) == 1:
    # 单手模式 (现有 run_physics_tracking)
elif len(hand_indices) == 2:
    # 双手模式 (新 run_bimanual_tracking)
```

### 3.3 验证方法

1. 单手模式回归测试: `--hand-idx 0` 和 `--hand-idx 1` 应正常工作
2. 双手模式: `--hand-idx both` 应同时渲染左右机械臂
3. 自动检测: `--hand-idx -1` 在双手数据上应进入双手模式
4. 双手基座位置不冲突: 左右手基座 XY 不重叠

### 3.4 已知风险

1. **基座冲突**: 左右手基座都在 +Y 方向 0.30m，可能太近。需要看实际手腕位置。如果左右手腕 XY 距离够大（>0.3m），基座不会重叠。
2. **IK 收敛**: 双手同时 IK 可能更慢，但不影响正确性。
3. **RelaxedIK**: 现有 `RelaxedIKSolver` 已支持 left + right setting，无需改动。
4. **Retargeting 配置**: 需要确认 `HandType.left` 的 config 存在。

---

## 4. 实现顺序

按用户要求"同时实现两个任务"，但内部仍有依赖关系：

1. **任务1 (基座位置)**: 改 4 处常量 + 函数，独立可验证
2. **任务2 (双手逻辑)**: 改造数据加载/机器人设置/追踪流程，依赖任务1 的基座位置逻辑

建议实现顺序：
1. 实现任务1 (基座位置) → 验证 fpv 视角能看到桌面
2. 实现任务2 (双手逻辑) → 验证双机械臂同时工作
3. 集成验证: 双手模式 + 新基座位置

---

## 5. 待确认问题

1. **基座偏移方向**: 当前方案是 +Y (相机后方)。如果用户的相机看向其他方向（非 -Y），需要动态计算偏移方向。当前 HaWoR 数据相机看向 -Y，方案可行。
2. **双手基座冲突**: 如果左右手腕 XY 距离 < 0.3m，基座可能重叠。需要看实际数据。如果重叠，可以让左手基座偏移 +Y+X，右手基座偏移 +Y-X。
3. **COMFORT_TARGET_IN_BASE**: 基座偏移后，舒适目标点是否需要调整？当前方案改为 [0.25, -0.30, -0.50]，让目标点在世界坐标的手腕附近。
