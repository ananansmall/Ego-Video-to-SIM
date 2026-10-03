# 2026-06-26 全部修改汇总

> 本文档整理 2026-06-25 ~ 2026-06-26 两次会话中对 `combination/` 目录的所有修改。
> 按时间顺序记录，包含文件级变更和代码级细节。

---

## 一、修改文件清单

| # | 文件 | 操作 | 状态 |
|---|------|------|------|
| 1 | `04_physics_simulation.py` | 修改 | ✓ 语法检查通过 |
| 2 | `doc/04_physics_simulation.md` | 新建 | ✓ 已复制到 docs/ |
| 3 | `docs/04_physics_simulation.md` | 新建 | 从 doc/ 复制 |
| 4 | `CHANGE_LOG.md` | 修改 | ✓ 已更新 |
| 5 | `COMMANDS.md` | 修改 | 前一会话 (2026-06-25) |

---

## 二、04_physics_simulation.py 代码变更

### 变更 1: 回退基座/相机改动

**原因**: 前一会话 Task 1 降低了基座高度 (0.70→0.40) 并添加 Y 偏移 (0.30)，导致 fpv 视角下机械臂位置完全错误。用户明确要求退回。

**具体修改** (共 6 处):

| 位置 | 修改前 | 修改后 |
|------|--------|--------|
| line 108 | `COMFORTABLE_REACH = 0.40` | `COMFORTABLE_REACH = 0.70` |
| line 109 | `BASE_OFFSET_Y = 0.30` | 整行删除 |
| line 110 | `COMFORT_TARGET_IN_BASE = [0.25, 0.0, -0.35]` | `COMFORT_TARGET_IN_BASE = [0.25, 0.0, -0.55]` |
| `_compute_optimal_fixed_base` | `arm_base_pos[1] += BASE_OFFSET_Y` | 删除该行 |
| `_compute_fixed_base_clusters` (3处) | `base_pos[1] += BASE_OFFSET_Y` | 删除该行 (3处) |

**回退后的基座位置计算**:
```python
# 修改前 (错误)
arm_base_pos = centroid.copy()
arm_base_pos[1] += 0.30  # Y 偏移
arm_base_pos[2] += 0.40  # 低基座

# 修改后 (恢复)
arm_base_pos = centroid.copy()
arm_base_pos[2] += 0.70  # 高基座, 机械臂垂直抓取
```

### 变更 2: 单夹爪左右开合不对称修复

**根因**: 3 层 bug:
1. URDF 模板用右夹爪 finger 几何硬编码，左夹爪也用了右夹爪的值
2. `_compute_analytical_gripper_pose` 用硬编码右夹爪常量
3. `run_single_gripper_tracking` 硬编码 `prefix = "right"`

**关键发现**: GalaxeaManipSim 的左右夹爪 URDF 是镜像的:
- 右夹爪 finger_joint1: origin `xyz="0.03689 -0.013453 -0.00012053"`, axis `0 -1 0` (finger1 在 -Y)
- 左夹爪 finger_joint1: origin `xyz="0.03689 0.013453 0.00012067"`, axis `0 1 0` (finger1 在 +Y)

**具体修改** (共 6 处):

#### (a) 新增左右夹爪 finger 几何常量 (lines 171-193)

```python
# 右夹爪 finger 几何 (从 r1_v2_1_0_floating_right.urdf 提取)
_FINGER1_ORIGIN_RIGHT = np.array([0.03689, -0.013453, -0.00012053])
_FINGER1_AXIS_RIGHT = np.array([0, -1, 0])
_FINGER2_ORIGIN_RIGHT = np.array([0.03689, 0.013453, 0.00012067])
_FINGER2_AXIS_RIGHT = np.array([0, 1, 0])
# 左夹爪 finger 几何 (镜像, 与 r1_v2_1_0_floating_left.urdf 一致: finger1/2 互换)
_FINGER1_ORIGIN_LEFT = np.array([0.03689, 0.013453, 0.00012067])
_FINGER1_AXIS_LEFT = np.array([0, 1, 0])
_FINGER2_ORIGIN_LEFT = np.array([0.03689, -0.013453, -0.00012053])
_FINGER2_AXIS_LEFT = np.array([0, -1, 0])
_FINGER_BASE_DIST = abs(_FINGER1_ORIGIN_RIGHT[1] - _FINGER2_ORIGIN_RIGHT[1])  # 0.026906
GRIPPER_INIT_OPEN = 0.04


def _get_finger_geom(prefix):
    """返回 (finger1_origin, finger1_axis, finger2_origin, finger2_axis)
    左右夹爪 finger 几何不同 (镜像), 与 GalaxeaManipSim URDF 一致:
    - right: finger1 在 -Y, finger2 在 +Y
    - left:  finger1 在 +Y, finger2 在 -Y (镜像)
    """
    if prefix == "left":
        return _FINGER1_ORIGIN_LEFT, _FINGER1_AXIS_LEFT, _FINGER2_ORIGIN_LEFT, _FINGER2_AXIS_LEFT
    return _FINGER1_ORIGIN_RIGHT, _FINGER1_AXIS_RIGHT, _FINGER2_ORIGIN_RIGHT, _FINGER2_AXIS_RIGHT
```

**替换**: 原来的 4 个全局常量 `_FINGER1_ORIGIN`, `_FINGER1_AXIS`, `_FINGER2_ORIGIN`, `_FINGER2_AXIS` (只有右夹爪值)。

#### (b) URDF 模板 finger joint 改为占位符 (lines 233-268)

```xml
<!-- 修改前: 硬编码右夹爪值 -->
<origin xyz="0.03689 -0.013453 -0.00012053" rpy="0 0 0"/>
<axis xyz="0 -1 0"/>

<!-- 修改后: 占位符 -->
<origin xyz="{f1_origin}" rpy="0 0 0"/>
<axis xyz="{f1_axis}"/>
```

同样对 `finger_joint2` 用 `{f2_origin}` 和 `{f2_axis}`。

#### (c) `_generate_gripper_only_urdf` 填充占位符 (lines 302-310)

```python
# 修改前
xml = _GRIPPER_ONLY_URDF_TEMPLATE.format(prefix=prefix, mesh_dir=str(R1_MESH_DIR))

# 修改后
f1o, f1a, f2o, f2a = _get_finger_geom(prefix)
xml = _GRIPPER_ONLY_URDF_TEMPLATE.format(
    prefix=prefix, mesh_dir=str(R1_MESH_DIR),
    f1_origin=f"{f1o[0]} {f1o[1]} {f1o[2]}",
    f1_axis=f"{f1a[0]} {f1a[1]} {f1a[2]}",
    f2_origin=f"{f2o[0]} {f2o[1]} {f2o[2]}",
    f2_axis=f"{f2a[0]} {f2a[1]} {f2a[2]}",
)
```

#### (d) `_compute_analytical_gripper_pose` 使用 prefix 依赖几何 (lines 361-365)

```python
# 修改前: 硬编码右夹爪常量
finger1_in_gripper = _FINGER1_ORIGIN + _FINGER1_AXIS * joint1
finger2_in_gripper = _FINGER2_ORIGIN + _FINGER2_AXIS * joint2

# 修改后: 根据 prefix 选择左右夹爪几何
f1_origin, f1_axis, f2_origin, f2_axis = _get_finger_geom(prefix)
finger1_in_gripper = f1_origin + f1_axis * joint1
finger2_in_gripper = f2_origin + f2_axis * joint2
```

#### (e) `run_single_gripper_tracking` prefix 修复 (line 2319)

```python
# 修改前: 硬编码右夹爪
prefix = "right"

# 修改后: 使用对应侧夹爪 URDF (镜像几何)
prefix = mano_side  # mano_side = "left" if hand_idx[0] == 0 else "right"
```

### 变更 3: 前一会话已做的修改 (仍在文件中)

以下修改来自 2026-06-25 会话，目前仍在文件中，**未被本次回退**:

| 修改项 | 描述 |
|--------|------|
| `FLIP_Z_FOR_PHYSICS = False` | Z 不翻转，对齐 02 坐标变换 |
| `run_bimanual_tracking` 方法 | 双手追踪（运动学驱动，非物理），`--hand-idx both` 触发 |
| `_detect_hand_idx` 返回 list | 支持检测双手 `[-1, 0, 1, "both"]` |
| `--hand-idx` 命令行参数 | `0`=左手, `1`=右手, `both`=双手, `-1`=自动 |
| `--view` 命令行参数 | `fpv`/`topdown`/`behind`/`front` 多视角 |
| `--base-cluster` | 分段固定基座模式 |
| `--single-gripper` | 单夹爪模式 |

---

## 三、新建文件

### 3.1 doc/04_physics_simulation.md (后复制到 docs/)

**用途**: 04 物理仿真驱动的完整技术文档。

**9 个章节**:
1. **概述** — 核心定位，管线中的位置
2. **运行方式** — 命令/输出/三种模式/常用参数
3. **架构总览** — 模块结构树（常量/工具函数/平滑器/PhysicsSimulator 类）
4. **运行流程详解** — 8 步完整机械臂模式 + 单夹爪解析法流程图
5. **与 02 的具体区别** — 对比表 + 驱动方式差异代码 + 场景/GLB/基座差异
6. **为什么实现物理仿真** — 02 局限 + 04 价值 + decimation 原理
7. **如何调用 GalaxeaManipSim** — 路径/URDF/RelaxedIK/Mesh + PD 参数对齐
8. **关键参数速查** — 物理参数/工作空间参数/坐标变换
9. **已知问题** — GPU 需求/双手模式/左右镜像/基座高度/碰撞体开销

---

## 四、CHANGE_LOG.md 变更

追加条目 `## [2026-06-26] 04 回退基座改动 + 单夹爪左右镜像修复 + 04 详解文档`，记录:
- 回退 COMFORTABLE_REACH/BASE_OFFSET_Y/COMFORT_TARGET_IN_BASE 的 6 处修改
- 单夹爪镜像修复的 6 处修改
- 新建文档
- 验证结果 (语法检查通过, GPU 测试待做)

---

## 五、COMMANDS.md 变更 (前一会话)

前一会话修改的 COMMANDS.md 包含:
- 04 的 `--hand-idx` 参数文档 (0/1/both/-1)
- 04 的 `--view` 参数文档 (fpv/topdown/behind/front)
- 04 的 `--single-gripper` 参数文档
- 04 的 `--base-cluster` 参数文档
- `--num-frames` 测试参数
- 完整管线运行顺序说明

---

## 六、未完成的修改

| 项目 | 状态 | 原因 |
|------|------|------|
| 单夹爪左右镜像修复 GPU 验证 | 未验证 | 沙箱无 GPU/Vulkan，报 `failed to find a rendering device` |
| `run_bimanual_tracking` 决定 | 待定 | 用户表示困惑但未明确要求删除 |
| 基座高度 + fpv 视角问题 | 回退到 0.70 | fpv 视角下机械臂仍可能挡视野，需后续解决 |
