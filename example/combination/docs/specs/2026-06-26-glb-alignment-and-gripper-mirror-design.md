# 2026-06-26 GLB 对齐与夹爪镜像修复设计

> 范围: `04_render_dual_arm.py` 微调, 不重写
> 基准: `02_render_scene.py` 源代码
> 目标: 修复 GLB 摆放位置错乱 + 夹爪左右开合不对称 + 自动手部检测
> 后续延后: 50fps 性能优化、视频文件重编码细节

***

## 一、问题诊断（根因定位）

通过逐行对比 `04_render_dual_arm.py` 与 `02_render_scene.py` 源码, 定位到 **4 处根因**:

### 根因 1: GLB 加载逻辑不完整 (GLB 对齐问题主因)

`04_render_dual_arm.py` 的 `load_glb_transformed` (line 362) 缺少 02 中已有的关键步骤:

| 步骤                  | 02 (正确)                    | 04 (缺失) |
| ------------------- | -------------------------- | ------- |
| 自动检测 GLB 坐标系        | `_detect_glb_up_axis` 启发式  | ❌ 缺失    |
| Z-UP → Y-UP 转换      | 必要时应用 `ZUP_TO_YUP`         | ❌ 缺失    |
| 读取 `glb_up_axis` 参数 | 优先用 transform\_params 保存的值 | ❌ 缺失    |
| 内存释放                | `gc.collect()`             | ❌ 缺失    |

**影响**: 若 GLB 是 Z-UP (RAS 导出常见), 04 把地板当墙处理, 物体平躺, 这就是"GLB 摆放位置被更改"现象。

### 根因 2: 夹爪 finger joint 符号错误 (单个夹爪只开合一边的主因, 用户指正)

**通过查看 URDF 文件确认** (此前未做物理仿真实验, 现已查证):

`r1_v2_1_0_floating_right.urdf` line 542-610:
```
right_gripper_finger_joint1: axis="0 -1 0", limit=[0, 0.05]
right_gripper_finger_joint2: axis="0  1 0", limit=[0, 0.05]   ← axis 相反!
```

`r1_v2_1_0_floating_left.urdf` line 542-610:
```
left_gripper_finger_joint1:  axis="0  1 0", limit=[0, 0.05]
left_gripper_finger_joint2:  axis="0 -1 0", limit=[0, 0.05]   ← axis 相反!
```

**关键事实**: 两个 finger joint 的 axis 相反 + limit 都是非负 [0, 0.05]。
**因此两个 joint 必须用同号值才能对称开合**:
- joint1=0.04, axis=(0,-1,0) → finger1 向 -Y 移动 0.04
- joint2=0.04, axis=(0, 1,0) → finger2 向 +Y 移动 0.04  ← 对称开合 ✓

**04 (和 02) 代码的 BUG** (`04_render_dual_arm.py` line 521-522):
```python
init_qpos[self.gripper_idx1] = 0.04
init_qpos[self.gripper_idx2] = -0.04   # ← 负值! 违反 limit [0, 0.05], 且符号错了
```

代入计算:
- joint1=0.04, axis=(0,-1,0) → finger1 向 -Y 移动 0.04
- joint2=-0.04, axis=(0, 1,0) → finger2 向 -Y 移动 0.04 (负值取反了 axis)
- **两个 finger 都向 -Y 移动 → 只开合一边**

**retargeting 输出也有同样问题** (`04_render_dual_arm.py` line 677-680):
```python
qpos[self.gripper_idx1] = float(sapien_qpos[self.gripper_idx1])
qpos[self.gripper_idx2] = float(sapien_qpos[self.gripper_idx2])  # ← 两个独立值, 不保证对称
```

retargeting 优化器不知道 URDF axis 方向, 输出两个独立值, 可能一正一负 → 夹爪歪斜。
用户洞察正确: **夹爪应该是单控制量, 不应该有两个独立值**。

### 根因 2b: 左臂初始关节角错误 (次要, 加剧观感)

```python
# 02_render_scene.py line 154 (正确, 与 hand_track/common.py 一致)
LEFT_ARM_STARTING = [-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]

# 04_render_dual_arm.py line 122 (错误!)
LEFT_ARM_STARTING = [1.5, 1.9508, 1.0809, 0.4438, -0.1709, 0.1985]
```

R1 URDF 左臂的镜像关系是 joint2/4 取反。04 错误地把 joint1/2/4 取反, 导致左臂初始姿态错误。
此问题与夹爪开合无关, 但会影响左臂整体姿态, 仍需修正。

### 根因 3: 手部数据二次变换 (加剧 GLB 错位观感)

`04_render_dual_arm.py` 的 `load_hawor_npz` (line 201-207) 额外应用了:

```python
pred_trans[hand_i, frame_i] = R_c2w[frame_i] @ pred_trans[hand_i, frame_i] + t_c2w[frame_i]
rot_mat_world = R_c2w[frame_i] @ rot_mat
```

但 `02_render_scene.py` 的 `load_hawor_data` (line 688-752) 直接使用 npz 中的 `pred_trans/pred_rot`, 02 注释 (line 23-24) 明确说明 `pred_trans` 已是世界坐标。

**影响**: 若数据已是世界坐标, 04 的二次变换让手部位置偏移 → 机械臂跟随到错误位置 → GLB 看起来"位置被改了"。

### 根因 4: hand\_detector 模块缺失

`04_render_dual_arm.py` line 158 `from hand_detector import HandDetector`, 但 `combination/hand_track/` 下并无 `hand_detector.py`。`02_render_scene.py` 中已有等效的 `_detect_hands` 函数 (line 629-685) 可复用。

**影响**: 04 直接运行会 ImportError。

***

## 二、修复方案 (方案 B: 微调对齐 + 夹爪镜像 + 性能基线)

### 修复 1: 替换 GLB 加载函数为 02 版本

**改动**: 将 04 的 `load_glb_transformed` 函数体替换为 02 中的版本 (line 902-1022)。

**关键代码差异** (新版本 04 应包含):

```python
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

def _detect_glb_up_axis(all_vertices):
    # 启发式判断 z-up vs y-up (复制自 02 line 867-896)
    ...

def load_glb_transformed(glb_path, transform_params_path, scene, logger=None):
    params = np.load(str(transform_params_path))
    saved_glb_up_axis = str(params.get('glb_up_axis', 'y-up')) if 'glb_up_axis' in params else None
    # 检测或读取 glb_up_axis
    # 必要时 ZUP_TO_YUP 转换
    # 再应用 s_inv * R_inv @ vertices + t_inv
    # 最后 RXWORLD_TO_SAPIEN 转换
    # 添加 gc.collect() 内存释放
```

**验证**: GLB 在 SAPIEN 中保持与 02 输出相同的地板朝向和物体位置。

### 修复 2: 修正 LEFT\_ARM\_STARTING

**改动** (1 行):

```python
# 04_render_dual_arm.py line 122
LEFT_ARM_STARTING = [-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]  # 与 02 一致
```

**验证**: 左臂初始姿态与 02 一致, 关节角符号正确镜像 (joint2/4 取反)。

### 修复 3: 移除 load\_hawor\_npz 中的 R\_c2w/t\_c2w 转换

**改动**: 移除 04 `load_hawor_npz` 中的二次变换循环 (line 201-207), 直接使用 npz 中的 `pred_trans/pred_rot` 作为世界坐标, 与 02 行为一致。

```python
# 修改前 (二次变换, 与 02 不一致)
for frame_i in range(pred_trans.shape[1]):
    for hand_i in range(pred_trans.shape[0]):
        if pred_valid[hand_i, frame_i]:
            pred_trans[hand_i, frame_i] = R_c2w[frame_i] @ pred_trans[hand_i, frame_i] + t_c2w[frame_i]
            rot_mat = pr.matrix_from_compact_axis_angle(pred_rot[hand_i, frame_i])
            rot_mat_world = R_c2w[frame_i] @ rot_mat
            pred_rot[hand_i, frame_i] = pr.compact_axis_angle_from_matrix(rot_mat_world)

# 修改后 (移除变换, 与 02 一致)
# npz 中 pred_trans/pred_rot 已是世界坐标, 直接使用
```

同时移除 `R_c2w`/`t_c2w` 的加载逻辑 (除非其他地方仍需要, 如相机轨迹)。`load_hawor_npz` 不再返回 R\_c2w/t\_c2w, 但 `R_c2w/t_c2w` 仍可在主流程中按需加载用于相机轨迹 (与 02 的 `load_hawor_c2w` 一致)。

**验证**: 同一 npz 输入下, 04 与 02 加载出的 `pred_trans/pred_rot` 完全一致。

### 修复 4: 用 02 的 \_detect\_hands 替换缺失的 hand\_detector

**改动**: 将 04 的 `detect_hands` 函数 (line 145-163) 替换为 02 的 `_detect_hands` (line 629-685) + 自定义 Handedness 枚举包装:

```python
class Handedness:
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    NONE = "none"

class HandDetectionResult:
    def __init__(self, handedness, left_valid_frames, right_valid_frames, method):
        self.handedness = handedness
        self.left_valid_frames = left_valid_frames
        self.right_valid_frames = right_valid_frames
        self.detection_method = method
        self.description = f"hands={handedness}"

def detect_hands(hawor_dir, logger):
    from 02_render_scene import _detect_hands  # 不行, 不能跨文件
    # 直接复制 _detect_hands 函数体到 04
    hands = _detect_hands_local(Path(hawor_dir))  # 复制 02 的实现
    ...
```

实际做法: 直接把 02 的 `_detect_hands` 函数复制到 04 内部, 不再 import 不存在的 hand\_detector。

**验证**: 04 能正确检测左手/右手/双手。

### 修复 5: 夹爪统一为单控制量 (用户指正, 真正根因修复)

**改动**: 因为 URDF 中两个 finger joint 的 axis 相反 + limit=[0, 0.05] (非负), 必须用**同号值**才会对称开合。

**位置 1** (`04_render_dual_arm.py` line 521-522, 初始姿态):

```python
# 修改前 (BUG: 负值违反 limit, 且 axis 相反时负值导致两 finger 同向移动)
init_qpos[self.gripper_idx1] = 0.04
init_qpos[self.gripper_idx2] = -0.04

# 修改后 (同号, axis 相反, 对称开合)
init_qpos[self.gripper_idx1] = 0.04
init_qpos[self.gripper_idx2] = 0.04
```

**位置 2** (`04_render_dual_arm.py` line 677-680, retargeting 输出赋值):

```python
# 修改前 (两个独立值, 不保证对称, 可能一正一负)
if self.gripper_idx1 < len(sapien_qpos):
    qpos[self.gripper_idx1] = float(sapien_qpos[self.gripper_idx1])
if self.gripper_idx2 < len(sapien_qpos):
    qpos[self.gripper_idx2] = float(sapien_qpos[self.gripper_idx2])

# 修改后 (单控制量 + clamp 到 limit [0, 0.05])
if self.gripper_idx1 < len(sapien_qpos):
    gripper_open = float(sapien_qpos[self.gripper_idx1])
    gripper_open = max(0.0, min(0.05, gripper_open))   # clamp 到 URDF limit
    qpos[self.gripper_idx1] = gripper_open
    if self.gripper_idx2 < len(sapien_qpos):
        qpos[self.gripper_idx2] = gripper_open          # 同号, axis 相反, 对称开合
```

**原理说明** (axis 相反 + 同号 → 对称):
- 右臂: joint1 axis=(0,-1,0), joint2 axis=(0,1,0); 同号 0.04 → finger1 向 -Y, finger2 向 +Y ✓
- 左臂: joint1 axis=(0, 1,0), joint2 axis=(0,-1,0); 同号 0.04 → finger1 向 +Y, finger2 向 -Y ✓

**验证**: 单个夹爪的两个 finger 对称开合 (向相反方向移动相同距离), 不再只开合一边。

### 修复 6: 添加 FPS 性能基线测量

**改动**: 在 04 主循环中添加详细的计时器, 量化当前 FPS:

```python
import time
_frame_times = []

for local_idx in trange(num_frames, ...):
    _t0 = time.time()
    # ... 原有渲染逻辑 ...
    _frame_times.append(time.time() - _t0)

# 渲染完成后输出性能统计
fps_mean = 1.0 / np.mean(_frame_times)
fps_p50 = 1.0 / np.percentile(_frame_times, 50)
fps_p95 = 1.0 / np.percentile(_frame_times, 95)
logger.info(f"  性能: 平均 {fps_mean:.1f} fps, P50 {fps_p50:.1f} fps, P95 {fps_p95:.1f} fps")
logger.info(f"  目标: 50 fps {'✓' if fps_mean >= 50 else '✗ 未达标'}")
```

**验证**: 输出日志明确显示当前 FPS 是否达到 50fps, 为后续优化提供基线数据。

***

## 三、不在本次范围

为遵循"以源代码为基准做微调"原则, 以下项目本次**不修改**, 留待后续:

1. **重力参数调整**: 用户明确"仅允许调整重力参数"。本次不动重力。
2. **50fps 性能优化实施**: 本次只添加测量基线, 不优化代码。
3. **X/Y 轴坐标修改**: 用户明确禁止。
4. **视频文件重编码细节**: ffmpeg 重编码逻辑已与 02 一致, 不动。
5. **量化验证方案设计**: 99% MANO 一致性量化方案留待性能基线后处理。

***

## 四、验证策略

### 4.1 静态验证 (无需 GPU)

1. **语法检查**: `python -c "import ast; ast.parse(open('04_render_dual_arm.py').read())"`
2. **常量对比**: 写脚本对比 04 与 02 的所有常量值
3. **import 检查**: `python -c "import sys; sys.path.insert(0, '.'); import 04_render_dual_arm"` 验证无 ImportError

### 4.2 动态验证 (需 GPU, 由用户执行)

1. **GLB 对齐验证**: 运行 04 与 02, 对比同一帧的 GLB 物体位置是否一致
2. **左右夹爪对称验证**: 双手数据下, 检查左/右夹爪开合角度对称性
3. **手部检测验证**: 用左手/右手/双手数据各测一次, 检测结果正确

### 4.3 量化指标 (用户要求 99% MANO 一致性)

后续单独设计, 本次仅修复对齐 bug。

***

## 五、风险评估

| 风险                                 | 等级 | 缓解                                   |
| ---------------------------------- | -- | ------------------------------------ |
| 02 的 `_detect_hands` 在 04 上下文行为不一致 | 低  | 直接复制函数体, 不修改逻辑                       |
| `R_c2w/t_c2w` 移除后相机轨迹仍能正常工作        | 中  | 相机轨迹单独加载 (load\_hawor\_c2w), 与手部数据解耦 |
| `LEFT_ARM_STARTING` 改动影响 IK 收敛     | 低  | 改为与 02 一致, 02 已验证可用                  |
| GLB 加载改动引入新 bug                    | 低  | 直接复制 02 的实现, 风险最小                    |

***

## 六、用户已确认决策

1. ✅ 方案 B: 微调对齐 4 处 + 夹爪镜像 + 性能基线
2. ✅ 数据加载: 对齐 02, 移除 R\_c2w/t\_c2w 二次变换

***

## 七、下一步

1. 用户审查本设计文档 → 批准后进入 writing-plans 阶段
2. writing-plans 输出具体实施任务清单 (按文件 + 行号 + 改动)
3. 按计划执行, 每步验证

