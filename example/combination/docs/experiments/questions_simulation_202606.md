# Q&A

## Q: GLB/夹爪镜像根因 — Z轴翻转的反射变换 + 地面下移方案

**日期**: 2026-06-27
**分类**: 调试 / 修复

### 问题

用户反馈: "这个物体 glb 的初始坐标是错误的, 所以导致了这个问题的存在, 我建议你调整就是 z 轴和重力, 其他不要动, 可以把 z=0 平面降下去, 如果在反面的话"

前情: 用户之前已指出 "z 轴都是对的, x, y 的坐标问题, 有点镜像"。用户运行 `--single-gripper` 模式。要求以 `02_render_scene.py` 为基线 (它的映射是正确的), 04 只需"加重力", 不应改其他坐标变换。

### 解答

#### 根因: FLIP_Z_FOR_PHYSICS=True 是反射变换 (det=-1)

对比 02 (正确) 和 04 (镜像) 的坐标变换差异:

| 位置 | 02 (正确) | 04 (镜像, 修复前) |
|------|----------|----------|
| `_render_to_sapien` (手部点) | `RXWORLD_TO_SAPIEN @ pts` | 同上 + **`result[..., 2] = -result[..., 2]`** |
| GLB 顶点 | `RXWORLD_TO_SAPIEN @ vertices` | 同上 + **`vertices[:, 2] = -vertices[:, 2]`** |
| 手腕旋转 | `RXWORLD_TO_SAPIEN @ R @ RXWORLD_TO_SAPIEN.T` | 同上 + **`Z_FLIP_R @ R @ Z_FLIP_R`** (det=-1 共轭) |
| 相机 | 用 `R_AXIS` | 用 `RXWORLD_TO_SAPIEN` + Z 翻转分支 |

**关键数学洞察**: `result[..., 2] = -result[..., 2]` 不是旋转, 而是关于 xy 平面的**反射 (reflection)**。其雅可比矩阵 `diag(1, 1, -1)` 的**行列式 det = -1**, 改变坐标系手性 (右手系 ↔ 左手系)。后果:
- Z 坐标正确 (位置上看起来对)
- 但 X, Y 方向发生镜像 (左右手翻转)
- 旋转矩阵用 `Z_FLIP_R @ R @ Z_FLIP_R` 共轭会得到转置矩阵的镜像版本, 进一步放大手性翻转

这正是用户观察到的 "z 轴都是对的, x, y 的坐标问题, 有点镜像"。

#### 为什么 04 引入了 Z 翻转而 02 没有

- **02 用 kinematic**: `set_qpos` 直接设置物体位姿, 物体不会因重力下落, 即使 Z 坐标为负 (SAPIEN_Z = HaWoR_Y, HaWoR SLAM Y 轴向下, "上方"对应 Y 负 → SAPIEN Z 负, 物体在地下) 也能正常显示。
- **04 用 dynamic physics**: 物体受重力影响, 如果 Z 为负 (在地下), 重力 -Z 会把物体拉得更深, 视觉上消失。
- 原作者用 `FLIP_Z=True` "翻转 Z" 来让物体 Z 变正 (从地下到地上), 但**反射变换**带来镜像副作用。

#### 用户建议的方案: 不翻转 Z, 而是把地面降下去

用户的核心思路: 既然物体初始 Z 为负是物理事实 (HaWoR SLAM 坐标系决定), 不要去翻转 Z (会引入镜像), 而是把**地面 (z=0 平面) 降下去**, 让物体仍在地面之上。

这正是已有代码的工作机制:
1. `_compute_object_support_plane(glb_path, transform_params_path)` 计算 GLB 顶点的 Z 范围
2. `ground_height = support_plane['min_z'] - 0.002` (地面自动降到物体最低点下方 2mm)
3. `scene.add_ground(ground_height)` 地面在 Z 负值
4. 重力保持默认 `-Z` (SAPIEN 默认 `[0, 0, -9.81]`)
5. 物体自然落在降低后的地面上

#### 修复

`04_physics_simulation.py` L95: `FLIP_Z_FOR_PHYSICS = True → False`

修复后 5 个 Z 翻转点全部不执行, GLB/手部/手腕映射与 `02_render_scene.py` 完全一致:
1. 相机: 走 else 分支, 不翻转
2. `_compute_object_support_plane`: 不翻转
3. `load_glb_with_physics`: 不翻转 (消除 GLB 镜像)
4. `_render_to_sapien`: 不翻转 (手部点与 02 一致)
5. `warm_start`: 不做 `Z_FLIP_R` 共轭 (消除夹爪手性翻转)

#### 为什么不动其他东西

用户明确要求 "其他不要动":
- 相机变换: 04 用 `RXWORLD_TO_SAPIEN`, 02 用 `R_AXIS` — 不动 (虽然形式不同, 但都在 SAPIEN 坐标系中)
- 碰撞禁用代码 (L1872-1895, L2377-2389): 不动 (修复夹爪不对称, 与镜像问题独立)
- 重力: 保持默认 `-Z` (地面降低后, 物体自然下落)
- GLB Z-UP 检测: 不动 (与镜像问题无关)

#### 验证

- 语法检查: 通过
- 运行验证: 待用户运行 `python 04_physics_simulation.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result --single-gripper --fast-collision`
- 预期: GLB 不镜像, 夹爪朝向正确, 物体在 Z 负值但地面自动降低支撑, 重力正常

---

## Q: 相机轨迹左右反转 + robot_only vs tracking 不一致 + handtrack 渲染两种夹爪

**日期**: 2026-06-26
**分类**: 调试 / 修复

### 问题

1. 相机轨迹用 hawor 的吗？感觉左右转动方向相反，7 和 hoi4d 的 hawor 文件有区别吗？坐标系有区别吗？
2. hand_object_robot_only.mp4 和 hand_object_robot_tracking.mp4 为什么映射的机器人不一样？tracking 是对的
3. handtrack 命令要渲染夹爪和带机械臂的夹爪两种

### 解答

1. **相机轨迹无 bug**: 两个数据集都从 hawor npz 读 `R_c2w`/`t_c2w`，约定完全一致 (右手系 c2w, OpenGL 相机, det=+1)。`hawor_cam_to_sapien_pose` 对两者处理相同，所有 R_align det=+1 (无反射镜像)。hoi4d 相机偏航变化 -37° vs 7 的 -0.43°，是 SLAM 数据本身差异，不是代码问题。`slam_scale`/`img_center` 差异不影响渲染 (代码不读这两个键)。

2. **robot_only MANO side bug (已修复)**: `run_robot_only` (02_render_scene.py:2403) 的 MANOLayer side 硬编码为 `prefix="right"`，而 `run_robot_tracking` (line 1924) 正确地根据 hand_idx 选择 side。当左手数据 (hand_idx=0) 时，robot_only 用右手 MANO 模型解读左手数据，导致指尖位置被镜像 → IK 目标错误 → 机器人运动不同。修复: `mano_side = "left" if hi == 0 else "right"`。

3. **handtrack both 模式 (已实现)**: render_auto.py 添加 `--mode both`，tracking 和 keypoint 视频只渲染一次，夹爪URDF视频循环渲染两轮 (gripper + gripper_arm)。`--handtrack` 模式自动传 `--mode both`。

---

## Q: 相机轨迹来源 + hoi4d 能否在 handtrack 跑 + --handtrack 管线集成

**日期**: 2026-06-26
**分类**: 架构

### 问题

1. 相机轨迹读取的是哪个文件？感觉变化有点大，对齐能应对所有内容吗？hoi4d 能不能在 handtrack 里跑？
2. hand_track 的手部检测/双手/双夹爪逻辑能否应用到 02_render_scene.py，通过管线 --handtrack 调用？

### 解答

1. **相机轨迹来源**: hand_track 和 02 都一样，从 hawor npz 的 `R_c2w`/`t_c2w` 读取（HaWoR SLAM 原始输出）。变化大是 SLAM 本身特性：hoi4d 相机步长均值 6.17mm vs hawor/7 的 1.5mm。对齐能应对，因为 01_align_scene.py 有 grid-search 尺度验证（手→GLB 距离 >10cm 时自动搜索更优 s_inv）。hoi4d 可以在 handtrack 跑：npz 有完整的 R_c2w (600,3,3)、t_c2w (600,3)、双手 pred_valid (599/600帧)。

2. **--handtrack 管线**: 已在 00_run_pipeline.py 添加 `--handtrack` 参数，启用时默认步骤变为 `1,7`（对齐+hand_track），跳过 02 的步骤 2-5。hand_track 的自动手部检测、双手、双夹爪逻辑全部通过 render_auto.py 调用，不修改 02_render_scene.py 原始代码。

---

## Q: SAPIEN 穿桌/无夹取 + PyBullet 与 SAPIEN 统一方案

**日期**: 2026-06-24
**分类**: 架构 / 调试

### 问题
用户提出 4 点:
1. 基座高度不够，机械臂应垂直抓取，PyBullet 也改成固定基座
2. SAPIEN 机械臂穿桌，物理仿真是否真正实现？文件命名要包含参数避免覆盖
3. SAPIEN 没有夹取物体，分析原因
4. PyBullet 效果不如 SAPIEN，机械臂位置错误，没和 GLB 交互，要求相机/机械臂/夹爪/GLB 全部对齐 SAPIEN，统一两个项目

### 解答

#### 1. 基座高度问题
- **SAPIEN 04**: `COMFORTABLE_REACH` 已从 0.55 → 0.70m，`COMFORT_TARGET_IN_BASE=[0.25, 0.0, -0.55]`，机械臂垂直下垂抓取
- **PyBullet**: 临时回退到 0.35m，因为 PyBullet 直接使用 02 的 IK 轨迹 (基于 0.35m 基座计算)
- **根本矛盾**: PyBullet 想要 0.70m 基座，但 IK 轨迹是基于 0.35m 计算的，强行提高基座会导致机械臂位置偏移 0.35m

#### 2. SAPIEN 机械臂穿桌原因
物理仿真**确实实现了** (PhysX 求解器 + 碰撞检测 + 重力)，但穿桌的原因:
1. **PD 跟踪误差**: SAPIEN 04 用 PD 驱动 (stiffness=1000, damping=200)，不是运动学控制。PD 控制器有 30-90° 跟踪误差，机械臂连杆有惯性，目标位置变化快时机械臂"掉"到桌面以下
2. **基座高度 + 目标位置**: 基座 Z=0.70m，末端目标 Z≈0.15m，桌面 Z≈0.0m。PD 误差可能导致实际位置低于桌面
3. **碰撞穿透 (tunneling)**: 机械臂速度过快时，PhysX 离散碰撞检测可能错过碰撞，穿透桌面
4. **Z 翻转**: `FLIP_Z_FOR_PHYSICS=True` 翻转 Z 坐标，如果翻转后物体 Z 变负会掉到地面以下

#### 3. SAPIEN 没有夹取物体原因
1. **PD 跟踪误差**: 夹爪手指用 PD 驱动 (GRIPPER_STIFFNESS=1000)，有跟踪误差，目标闭合位置达不到
2. **纯摩擦力抓取**: 只用 `static_friction=1.0, dynamic_friction=1.0`，没有 weld/attach。摩擦力抓取需要足够法向力，PD 误差导致法向力不足
3. **物体被推走**: 机械臂接近物体时速度过快，直接把物体推走而不是停下来夹取
4. **夹爪与物体对齐**: PD 误差导致夹爪位置偏移，可能根本没接触到物体

#### 4. PyBullet 与 SAPIEN 统一方案

**核心差异**:
| 维度 | SAPIEN 04 | PyBullet |
|---|---|---|
| IK 来源 | 独立重算 (DexRetargeting + RelaxedIK) | 直接用 02 的轨迹 |
| 基座高度 | 0.70m (可调) | 0.35m (受 02 轨迹限制) |
| 坐标系 | FLIP_Z_FOR_PHYSICS=True | 无 Z 翻转 |
| 控制策略 | PD 驱动 + 重力补偿 | 运动学控制 (resetJointState) |
| 物理引擎 | PhysX | PyBullet |

**统一方案** (待用户选择):
- **方案 A (推荐)**: PyBullet 也独立重算 IK，基座 0.70m，完全对齐 SAPIEN
- **方案 B**: 两者都用 02 的轨迹，基座 0.35m
- **方案 C**: SAPIEN 04 也用 02 的轨迹 (不独立重算 IK)

---

## Q: ffmpeg 重编码失败，只显示版本号

**日期**: 2026-06-21
**分类**: 调试

### 问题
ffmpeg 重编码在 16ms 内失败，错误信息被截断只显示 ffmpeg 版本信息：
```
ffmpeg 重编码失败: ffmpeg version 7.0.2-static ...
```

### 解答
根因：输入视频文件为空 (0 bytes)。ffmpeg 对空文件立即失败，stderr 开头是版本信息，实际错误在末尾。

修复：
1. 添加 `os.path.getsize(input_path) == 0` 检查，提前报错
2. 之前已将 stderr 显示从 `[:200]` 改为 `[-300:]`，显示实际错误而非版本号

---

## Q: 桌子位置误差 30cm，物体在桌面下方

**日期**: 2026-06-21
**分类**: 调试

### 问题
用户报告物理仿真中桌子位置不对，物体出现在桌面下方，误差约 30cm。

### 解答
诊断发现：
1. **04 的 `add_ground` 创建了可见的 5m×5m 灰色地面** (`render_half_size=[5, 5]`)，这个巨大平面在 Z=0.0078m，视觉上覆盖了物体。修复：改为 `render_half_size=[0, 0]`，地面不可见。
2. **桌面高度计算已正确**：`ground_height = min_z - 0.002 = 0.0078m`，物体 Z 范围 [0.0098, 0.0832]m，物体在桌面上方。
3. **之前版本可能使用默认 `GROUND_HEIGHT=-0.5`**，导致桌面在 Z=-0.5m，与物体相差约 50cm。现在已修复为动态计算。

---

## Q: PyBullet 夹爪加载不对，与 02 不一致

**日期**: 2026-06-21
**分类**: 调试

### 问题
PyBullet 仿真中夹爪加载不正确，与 02_render_scene.py / hand_track 不一致。

### 解答
根因：`_compute_analytical_gripper_pose` 使用标准 SVD，而 hand_track 使用加权 SVD (W_Y=5.0)。

差异：
- 标准 SVD：X 和 Y 轴均等折中，当 MANO 指向方向和开合方向不正交时，两个方向都不精确
- 加权 SVD：Y 轴 (开合方向) 权重 5x，优先保证开合方向精确，最小化指尖位置误差

修复：将 04 和 PyBullet 的 `_compute_analytical_gripper_pose` 都改为加权 SVD，与 hand_track 一致。

---

## Q: PyBullet 相机视角不对，不是第一人称

**日期**: 2026-06-21
**分类**: 调试

### 问题
PyBullet 仿真中相机视角不正确，不是第一人称视角。

### 解答
验证结果：
- `hawor_cam_to_sapien_pose` 和 `sapien_cam_to_pybullet_view` 函数与 02 一致
- PyBullet FPV 渲染方向 (上暗下亮) 与 02 一致
- 相机 up 向量 [-0.001, -0.006, -1.0] 不需要翻转（与 02 一致）

之前的问题（已修复）：
- PyBullet 地面 (`plane.urdf`) 在 Z=0.0078m，相机在 Z=0.0037m（地面下方），可见地面挡住视线 → 修复：地面 alpha=0 透明
- PyBullet 缺少光照设置 → 修复：添加 lightDirection/lightColor 等参数

---

## Q: Z翻转导致镜像/没对齐 — 回退对齐02坐标变换

**日期**: 2026-06-24
**分类**: 调试 / 架构

### 问题
用户报告: "02_render_scene.py 生成的是对的，但是在仿真中，你可能变化了几个坐标，没有一起变化，导致了有一些镜像，没对齐的问题...在02后面顶多改一下重力，我觉得你有调整x，y轴"

输出视频问题:
- `pybullet.mp4`: 看不到机械臂
- `pybullet_gripper.mp4`: MANO参数没对齐
- `physics_sim_physics_tracking.mp4`: GLB摆放位置被更改
- `pybullet_render.mp4`: 完全映射错误

### 解答

#### 根因分析
在04和PyBullet中添加了 `FLIP_Z_FOR_PHYSICS=True`，但Z翻转实现**不一致**:
- **相机**: 只翻转位置Z和forward/up的Z分量，没有用 `Z_FLIP @ R @ Z_FLIP` 完整翻转旋转矩阵
- **手腕位置**: 翻转所有关节Z
- **GLB顶点**: 翻转所有顶点Z
- **手腕旋转**: 用 `Z_FLIP_R @ R @ Z_FLIP_R` 翻转 (与相机翻转方式不同)

这种不一致导致: 翻转了位置但没有完整翻转旋转 → 镜像效应 → 相机看向错误方向 → 机械臂在视野外/MANO参数错位/GLB位置偏移

#### 修复方案
**回退Z翻转，完全对齐02_render_scene.py的坐标变换**:
- `FLIP_Z_FOR_PHYSICS = False` (04和PyBullet都改)
- 移除所有Z翻转分支 (相机/手腕/GLB/手腕旋转)
- 基座高度改回0.35m (与02一致)
- PyBullet默认使用02的轨迹 (`hand_object_robot_tracking.npy`)
- 仿真中只调整重力 (`GRAVITY=[0,0,-9.81]`)，不改变坐标变换

#### 验证结果
- PyBullet完整机械臂模式 (10帧): 10/10帧渲染成功，物体Z=0.0098m (正值，在桌面上方)
- PyBullet单夹爪模式 (10帧): 10/10帧渲染成功
- PyBullet 30帧测试: 30/30帧渲染成功，2:28.70秒

---

## Q: 夹爪模式不需要轨迹的技术原理

**日期**: 2026-06-24
**分类**: 架构

### 问题
深入分析夹爪模式不需要轨迹的技术原理和设计逻辑，明确其与其他模式在轨迹需求上的差异及原因。

### 解答

#### 完整机械臂模式 vs 单夹爪模式

| 维度 | 完整机械臂模式 | 单夹爪模式 |
|---|---|---|
| URDF | 完整R1右臂 (6 DOF arm + 2 DOF gripper) | 只有夹爪 (2 DOF finger) |
| 驱动方式 | IK求解 (DexRetargeting + RelaxedIK) → qpos | MANO手腕/指尖 → 解析位姿 |
| 输入 | `hand_object_robot_tracking.npy` (IK解) | HaWoR `.npz` (MANO关节) |
| 需要轨迹文件 | **是** | **否** |
| 计算复杂度 | 高 (迭代IK求解) | 低 (解析公式) |

#### 技术原理

**完整机械臂模式需要轨迹**:
- 机械臂有6个旋转关节 (arm_joint1..6)，末端要到达MANO手腕位置
- 这是一个**逆运动学 (IK)** 问题: 给定末端位姿，求解6个关节角
- IK求解需要迭代 (RelaxedIK)，计算量大，且需要warm-start
- 因此预先计算所有帧的IK解，保存为轨迹文件 (`.npy`)
- 仿真时直接加载轨迹，用 `resetJointState` 设置关节角

**单夹爪模式不需要轨迹**:
- 夹爪URDF只有2个prismatic关节 (finger_joint1/2)，控制手指开合
- 夹爪root位姿 (位置+旋转) 直接从MANO数据解析计算:
  1. 手指关节值 = (MANO指尖距离 - 基础距离) / 2
  2. root旋转 = 加权SVD (Procrustes) 匹配MANO指向方向和开合方向
  3. root位置 = MANO指尖中点 - root_R @ 指尖中点(夹爪坐标系)
- 这是**解析解** (非迭代)，每帧独立计算，无需预计算
- 仿真时直接用 `resetBasePositionAndOrientation` 设置root位姿 + `resetJointState` 设置手指

#### 设计逻辑
单夹爪模式的设计目的是**隔离测试夹爪跟踪精度**:
- 去除机械臂IK的复杂性，直接验证夹爪是否正确跟随MANO手部运动
- 适用于: 验证MANO→夹爪映射的准确性、调试夹爪开合控制、快速迭代夹爪URDF设计
- 完整机械臂模式适用于: 端到端验证 (MANO→IK→机械臂→夹爪→物理交互)

---

## Q: 04轨迹直接调用优化方案

**日期**: 2026-06-24
**分类**: 架构 / 优化

### 问题
针对当前系统无法直接调用04轨迹而必须先生成的操作流程进行优化改进，分析现有调用机制的限制因素，提出具体的优化方案以实现轨迹的直接调用功能。

### 解答

#### 现有调用机制的限制

**当前流程**:
```
用户运行04 (需要GPU, SAPIEN Vulkan渲染)
  → 04内部: 加载数据 → IK求解 → 物理仿真 → 渲染 → 保存轨迹.npy
用户运行PyBullet (CPU)
  → PyBullet: 加载04的轨迹.npy → 物理仿真 → 渲染
```

**限制因素**:
1. **GPU依赖**: 04使用SAPIEN渲染 (Vulkan)，需要GPU。但IK求解本身不需要GPU
2. **耦合设计**: IK求解与物理仿真/渲染耦合在 `run_robot_tracking()` 方法中，无法单独调用
3. **两步操作**: 用户必须先运行04生成轨迹，再运行PyBullet，操作繁琐
4. **轨迹兼容性**: 04的轨迹与坐标变换绑定 (之前Z翻转导致不兼容)

#### 优化方案

**方案A: 提取IK模块 (推荐)**
将04的IK求解逻辑提取为独立模块 `ik_solver.py`:
```python
# ik_solver.py (CPU only, 无需GPU)
class TrajectorySolver:
    def __init__(self, base_height=0.35):
        # 初始化 DexRetargeting + RelaxedIK
        ...
    
    def solve(self, hawor_data, mano_layer, start_frame, num_frames):
        # 返回 (N, 8) 轨迹: [arm_joint1..6, gripper1, gripper2]
        ...

# PyBullet直接调用:
solver = TrajectorySolver(base_height=0.35)
trajectory = solver.solve(hawor_data, mano_layer, 0, 113)
# 无需先运行04, 无需GPU
```

**方案B: 04添加--ik-only模式**
在04中添加 `--ik-only` 参数，跳过物理仿真和渲染，只计算并保存轨迹:
```bash
# CPU only, 无需GPU
python 04_physics_simulation.py --ik-only --hawor-dir ... --ras-dir ...
# 输出: physics_sim_physics_tracking.npy
```

**方案C: PyBullet内置IK求解**
在PyBullet管线中直接集成DexRetargeting + RelaxedIK:
```python
# pybullet_pipeline.py
if trajectory_path is None:
    # 自动求解IK (无需外部轨迹文件)
    trajectory = self._solve_ik(hawor_data, mano_layer)
```

**推荐**: 方案A (提取IK模块) — 最干净的架构，04和PyBullet都调用同一个IK模块，保证一致性。

---

## Q: 50帧性能指标测试

**日期**: 2026-06-24
**分类**: 性能 / 测试

### 问题
需确保系统能够稳定运行并达到50帧的性能指标，需记录帧率数据并验证其稳定性。

### 解答

#### 帧率数据记录

**PyBullet完整机械臂模式 (30帧)**:
- 总耗时: 148.70秒
- 每帧耗时: ~4.96秒/帧
- 渲染帧率: **0.20 fps** (包含物理仿真 + getCameraImage渲染)
- 物理参数: CONTROL_FREQ=30Hz, PHYSICS_TIMESTEP=1/240s, DECIMATION=8
- 视频输出: 1280x720 @ 30fps

**性能瓶颈分析**:
1. **getCameraImage (软件渲染)**: PyBullet使用CPU软件渲染，1280x720分辨率每帧约4秒，占总耗时80%+
2. **物理仿真**: 8个子步/帧，PyBullet物理引擎本身很快 (<0.1秒/帧)
3. **GLB加载**: 一次性加载，不影响逐帧性能

#### 50帧指标分析

| 指标类型 | 当前值 | 50帧目标 | 状态 |
|---|---|---|---|
| 物理仿真频率 | 240 Hz (内部) | 50 Hz | ✓ 超标 (物理引擎本身远超50Hz) |
| 控制频率 | 30 Hz | 50 Hz | ⚠ 需调整 CONTROL_FREQ=50 |
| 渲染帧率 | 0.20 fps | 50 fps | ✗ 软件渲染无法达到 |
| 视频输出帧率 | 30 fps | 50 fps | ⚠ 需调整 VIDEO_FPS=50 |

#### 优化建议

1. **物理仿真50Hz**: 将 `CONTROL_FREQ` 从30改为50，`DECIMATION` 自动调整为 `round(1/50 / (1/240)) = 5`
2. **渲染性能**: 
   - 降低分辨率 (1280x720 → 640x360) 可提升~4倍
   - 使用GUI模式连接 (硬件加速) 而非DIRECT模式
   - 或使用SAPIEN (GPU渲染) 替代PyBullet渲染
3. **视频50fps**: 将 `VIDEO_FPS` 从30改为50

**结论**: 物理仿真可以达到50Hz (当前240Hz内部频率)，但渲染帧率受限于PyBullet软件渲染，无法达到50fps。建议在需要50fps的场景使用SAPIEN (GPU渲染)。

---

## Q: 镜像修复后相机视角/桌面/GUI/上帝视角消失原因 + 双手逻辑

**日期**: 2026-06-25
**分类**: 调试 / 架构

### 问题
用户提出 2 点:
1. 觉得很奇怪，之前的相机视角是对的，修改回来相机视角又发生了问题，不明白原因。上帝视角、GUI 这些都有运行吗？桌面还有吗？为什么变成镜像回来之后有些东西都没有了？
2. 自动检测手能不能添加一个双手的逻辑？

### 解答

#### 1. 相机视角/桌面/GUI/上帝视角状态确认

**先明确结论：所有功能都还在代码里，没有任何功能被删除。** 用户感觉"消失了"是视觉假象，不是代码移除。

**代码实际状态** (2026-06-25):

| 功能 | 代码位置 | 默认状态 | 启用方式 |
|---|---|---|---|
| `FLIP_Z_FOR_PHYSICS` | 04_physics_simulation.py:101 | `False` (从 True 改来) | 改常量 |
| 相机视角 fpv/topdown/behind/front | 04_physics_simulation.py:3323-3325 | `fpv` | `--view topdown` 等 |
| 交互式 GUI (Viewer) | 04_physics_simulation.py:3310, 2337-2344 | 关闭 (写视频) | `--viewer` |
| 可见桌面 actor | 04_physics_simulation.py:2216-2242 | 开启 (`support_table=True`) | `--no-support-table` 关闭 |
| 物理地面 | setup_physics_scene (line 881) | 总是开启 | 不可关闭 |

#### 2. 相机视角"变错"的真正根因

**用户感觉"之前的相机视角是对的"** 指的是 `FLIP_Z_FOR_PHYSICS=True` 时的画面。诊断结果:

| FLIP_Z | 相机 Z | 相机朝向 | 物体 Z | 物理正确性 | GLB 手性 |
|---|---|---|---|---|---|
| True (旧) | -0.004 (地下) | 朝下看场景 | [-0.083, -0.010] (地下) | ✗ 物体在地下 | ✗ 镜像 |
| False (新) | +0.004 (地上) | 朝前看 (水平) | [+0.010, +0.080] (桌上) | ✓ 物理正确 | ✓ 不镜像 |

**关键洞察**:
- 旧版 (True) 看着"对"，是因为相机被翻到地下朝上看，恰好看到桌面底部 + 物体，画面"有内容"。但这是物理错误 (物体在地下) + 视觉错误 (GLB 镜像)。
- 新版 (False) 物理正确 (物体在桌上)，但相机现在在手腕高度 (Z≈0.004) **水平看前方**，而机械臂基座在手腕正上方 0.70m，机械臂从上垂下来充满整个画面，挡住了桌面和物体。

**所以"东西消失了"的真相**: 不是桌面/GUI/上帝视角被删了，而是 **fpv 视角下机械臂挡住了视野**。

#### 3. 上帝视角/GUI 怎么用

```bash
# 上帝视角 (topdown) - 看清整个场景, 不被机械臂挡
python 04_physics_simulation.py --view topdown --hawor-dir ... --ras-dir ... --fast-collision

# GUI 交互模式 (不保存视频, 可鼠标拖动相机)
python 04_physics_simulation.py --viewer --hawor-dir ... --ras-dir ... --fast-collision

# 上帝视角 + GUI 同时
python 04_physics_simulation.py --view topdown --viewer --hawor-dir ... --ras-dir ...

# 后方视角 (从机械臂后方看)
python 04_physics_simulation.py --view behind --hawor-dir ... --ras-dir ...
```

#### 4. 真正的修复方案 (已在设计文档中)

既然根因是"机械臂基座挡视野"，修复方向就是 **移动基座到相机视野外**:

```python
# 当前 (有问题): 基座在手腕正上方 0.70m, 挡住相机
arm_base_pos = centroid.copy()
arm_base_pos[2] += 0.70  # 正上方

# 修复后: 基座在相机后方 (+Y 0.30m, Z 降到 0.55m), 不挡视野
arm_base_pos = centroid.copy()
arm_base_pos[1] += 0.30   # 后方 (相机看 -Y, 基座在 +Y)
arm_base_pos[2] += 0.55   # 高度降到 0.55m (仍在臂展 0.713m 内)
```

**安全性论证**:
- 距离手腕 √(0.30² + 0.55²) = 0.626m < ARM_MAX_REACH (0.713m) ✓
- 机械臂从相机后方延伸到手腕，90% 路径在相机视野外 ✓
- fpv 视角下桌面/物体可见，机械臂只在画面边缘 ✓

#### 5. 双手逻辑实现方案

**当前 `_detect_hand_idx`** (line 463-491) 只返回单个 int (0 或 1):

```python
def _detect_hand_idx(hawor_path):
    # 检测 cam_space/0 (左手) 或 cam_space/1 (右手)
    # 返回 int 或 None
```

**改为返回 list[int]**:

```python
def _detect_hand_idx(hawor_path):
    # 返回 [], [0], [1], 或 [0, 1] (双手)
    cam_dir = Path(hawor_path) / "cam_space"
    detected = set()
    if cam_dir.exists():
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
    return sorted(detected)  # 空/单/双手
```

**双手驱动**: 加载左右两个机械臂 URDF (`r1_v2_1_0_floating_left.urdf` + `r1_v2_1_0_floating_right.urdf`)，分别用左手 MANO 数据驱动左机械臂，右手 MANO 数据驱动右机械臂。

**命令行**:
```bash
# 自动检测 (单手/双手都支持)
python 04_physics_simulation.py --hand-idx -1 ...

# 强制双手
python 04_physics_simulation.py --hand-idx both ...

# 强制单手 (兼容旧用法)
python 04_physics_simulation.py --hand-idx 0 ...
```

#### 6. 实现计划

已在 `docs/specs/2026-06-25-camera-view-and-bimanual-plan.md` 中详细规划:
- **任务1** (基座位置): 4 步骤, 改 4 处常量 + 函数
- **任务2** (双手逻辑): 7 步骤, 改 5 处 + 新增 2 处
- 用户已批准"同时实现两个任务"

---

## Q: 为什么 hoi4d1_vggt_omega 数据下 01_align_scene 和 02_render_scene 没有对齐？

**日期**: 2026-06-25
**分类**: 调试

### 问题
运行 `conda run -n dex python 00_run_pipeline.py --hawor-dir /home/an/data/hawor/hoi4d/ --ras-dir /home/an/data/ras/hoi4d1_vggt_omega/ --viewer` 时，01_align_scene.py 和 02_render_scene.py 的场景没有对齐，GLB 场景位置偏移约 1.2m。

### 解答
**根因**: RAS 导出的 GLB 可能是 Z-UP 坐标系（地板在 z=0，物体在 z>0），但代码假设所有 GLB 都是 Y-UP。对齐变换 `R_inv` 将 Y-UP 映射到 HaWoR render world，直接对 Z-UP 顶点应用会导致场景错位。

**具体数据验证**:
- hoi4d1_vggt_omega GLB: min_z=0.0, min_y=-5.0 → 地板在 z=0 → Z-UP
- 修复前 GLB center→hand 距离: 2.31m（错误）
- 修复后 GLB center→hand 距离: 1.14m（正确）

**修复方案**:
1. 新增 `_detect_glb_up_axis()` 函数，通过地板位置启发式检测 GLB 坐标系
2. 在 01_align_scene.py 中检测 GLB 坐标系，对 Z-UP 顶点先做 ZUP_TO_YUP 转换
3. 将 `glb_up_axis` 保存到 transform_params.npz
4. 在 02_render_scene.py 和 04_physics_simulation.py 中读取保存的值，对 Z-UP 顶点先转换再应用对齐变换

**涉及文件**: 01_align_scene.py, 02_render_scene.py, 04_physics_simulation.py

---

## Q: 对齐公式中 R_c2w Y-UP 转换方式和 GLB 坐标系的关系

**日期**: 2026-06-25
**分类**: 调试 / 概念

### 问题
用户指出: "这和坐标系没关系，只要你把相机的第一帧视角进行对齐，那么就是对的。hoi4d1_vggt_omega是因为它的glb是地面的坐标系，之前的7的相机坐标系就是glb坐标系。对齐正不正确你得看手有没有靠近glb。"

### 解答

**核心洞察**: 对齐正确性的唯一判断标准是**手有没有靠近GLB物体**。

**两种GLB坐标系**:
1. **相机坐标系** (如 "7" 数据): GLB 导出时做了 z→y 转换，R_c2w_ras ≈ I，相机在原点
2. **地面坐标系** (如 hoi4d1 数据): GLB 未转换，地板在 z=0，相机有非平凡旋转

**R_c2w Y-UP 转换必须条件化**:
- Y-UP GLB (相机坐标系): `R_c2w_yup = ZUP_TO_YUP @ R @ ZUP_TO_YUP.T` (相似变换) — 相机约定随世界 up 轴变化
- Z-UP GLB (地面坐标系): `R_c2w_yup = ZUP_TO_YUP @ R` (直接乘) — 只改变世界坐标系，相机约定不变

**尺度校正**:
- Umeyama 对 hoi4d1 不可靠 (给出 s_inv=1.82，实际应为 ~0.89)
- 添加手→GLB距离验证：若 > 10cm，自动网格搜索更优 s_inv

**验证结果**:
| 数据集 | GLB | 方法 | 手→GLB min |
|--------|-----|------|-----------|
| "7" | y-up | 相似变换 | 0.0041m |
| hoi4d1 | z-up | 直接乘 | 0.0002m |

---

## Q: hand_track 文件夹还有什么问题？

**日期**: 2026-06-25
**分类**: 架构

### 问题
hand_track 这个文件夹里面还有什么问题吗？

### 解答

审查了 hand_track 文件夹中所有 7 个 Python 文件 (common.py, gripper_config.py, align_strategy.py, render_gripper_only.py, render_auto.py, verify_optimizer_3points.py, __init__.py) 和 2 个配置文件。

**之前已修复的问题**:
- `common.py` 的 `load_glb_transformed()` 缺少 Z-UP GLB 处理 — 已添加 `glb_up_axis` 读取和 Z-UP→Y-UP 顶点转换

**新发现的问题** (均为代码质量/维护性问题，非功能性 bug):
1. **`_combine_videos_side_by_side` 重复代码**: `render_gripper_only.py` 和 `render_auto.py` 各有一份几乎完全一样的实现 (仅标签文字略有不同)
2. **`_ensure_transform_params` 重复代码**: `render_gripper_only.py` 和 `render_auto.py` 各有一份完全相同的实现
3. **临时文件命名**: `common.py:load_glb_transformed` 使用 `os.getpid()_geom_name` 命名临时 PLY，同一进程多次调用可能重名覆盖（实际双手模式只创建一个场景，不会触发）

**结论**: 除之前已修复的 Z-UP GLB bug 外，没有功能性 bug。上面的问题都是代码重复和维护性问题，不影响正确性。如需要，可将重复函数提取到 common.py 中统一维护。

---

## Q: 夹爪不对称打开 + 输出mp4无变化 + 用户提到可能性的调研

**日期**: 2026-06-26
**分类**: 调试 / 修复

### 问题
1. 运行 `python 04_physics_simulation.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result --fast-collision`，输出的 mp4 完全没有变化，夹爪仍然不对称打开
2. 对于夹爪问题，用户提到的那些可能性（mimic joint、URDF参数化、mapping）有进行调研吗？
3. 用户强调："首先我说的是单独夹爪，和地面有什么关系，夹爪的开合都是对称的"

### 调研过程

#### 1. 用户提到的可能性调研结果

**(1) mimic joint (夹爪应该只有一个变量控制开合)**:
- 所有 URDF 文件都**没有 `<mimic>` 标签**
- 但 retargeting 和 SAPIEN 都用**约定 A**（joint1 axis=(0,-1,0) limit=[0,0.05]，joint2 axis=(0,1,0) limit=[0,0.05]）
- 约定 A 只需 `q1 = q2` 即可对称张开，等价于 multiplier=1 的隐式 mimic
- 已做的 v2 修改 `gripper_target2 = gripper_target1` 就是这个隐式 mimic
- **结论: 不是问题根源**

**(2) URDF 参数化**:
- 存在两套约定（A/B），但 retargeting（`robot.urdf`）和 SAPIEN（`r1_v2_1_0_floating_right.urdf`）**都用约定 A，一致**
- `r1_gripper_glb.urdf` 用约定 B，但**全臂模式不用这个 URDF**（红鲱鱼）
- **结论: 不是问题根源**

**(3) mapping (retarget2sapien)**:
- DEBUG 输出证明: `sapien_qpos[idx1]=0.025095, sapien_qpos[idx2]=0.005457`，与 retargeting 输出一致
- **结论: 不是问题根源**

#### 2. 为什么 mp4 没有变化
- DEBUG 日志只打印到控制台，**不影响 mp4 内容**
- v2 修改（`gripper_target2 = gripper_target1`）目标层面正确（target=0.025, 0.025 对称）
- 但**物理碰撞**导致实际 qpos 仍然不对称

#### 3. 真正根本原因：物理碰撞 + SRDF 未加载

DEBUG 输出（帧0）:
```
gripper_target1=0.025095, gripper_target2=0.025095  (对称目标 ✓)
actual_qpos[idx1]=0.000000, actual_qpos[idx2]=0.037781  (严重不对称!)
finger_link1 接触数=3, finger_link2 接触数=7
接触: finger_link2 <-> finger_link1 (自碰撞!)
      finger_link2 <-> right_realsense_link (相机碰撞!)
      finger_link1 <-> unknown (GLB/桌面)
```

- `robot.srdf` 已声明 `right_gripper_finger_link1` 和 `right_gripper_finger_link2` 互不碰撞（`reason="Default"`）
- 但 `_setup_robot` 用 `loader.load(arm_urdf_path)` 加载 URDF 时**没有传入 SRDF 文件**
- `_prepare_arm_urdf` 也没处理 SRDF
- 所以 SRDF 声明**没有生效**，手指之间碰撞未被禁用

### 修复方案

在 `_setup_robot`（全臂模式，line 1872-1895）和 `run_single_gripper_tracking`（单夹爪模式，line 2377-2389）中加载 robot 后，用 SAPIEN 的 `set_collision_groups` API：

1. **禁用 finger1-finger2 之间的碰撞**: 用 ignore group（g2 bit + 相同 g3 id），让两个手指共享同一 ignore bit，互相忽略碰撞
2. **禁用 realsense_link 的所有碰撞**: 用 `[0,0,0,0]` 完全禁用（相机不需要物理碰撞）
3. **保留 finger 与 GLB/桌面碰撞**: 用于抓取物体

```python
# ignore group: 两 shape 共享同一 g2 bit 且 g3 相同时, 互相忽略碰撞
finger_ignore_bit = 1 << 0
finger_ignore_id = 1
finger_link_names = {"right_gripper_finger_link1", "right_gripper_finger_link2"}
for link in robot.get_links():
    if link.get_name() in finger_link_names:
        for component in link.entity.components:
            if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                for cs in component.get_collision_shapes():
                    g = list(cs.get_collision_groups())
                    g[2] |= finger_ignore_bit
                    g[3] = finger_ignore_id
                    cs.set_collision_groups(g)
    elif link.get_name() == "right_realsense_link":
        for component in link.entity.components:
            if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                for cs in component.get_collision_shapes():
                    cs.set_collision_groups([0, 0, 0, 0])
```

参考项目已有范式: `tri_model_physics/grasp_hawor.py` L691-709（用 `set_collision_groups([0,0,0,0])` 禁用非夹爪 link 碰撞）。

### SAPIEN collision_groups API 说明

碰撞判定公式: 当且仅当以下两条件**同时**成立才碰撞:
1. `(A.g0 & B.g1) or (A.g1 & B.g0)` — 双方 contact type 与对方 affinity 有交集
2. `not ((A.g2 & B.g2) and (A.g3 & 0xffff == B.g3 & 0xffff))` — 不在「忽略对」中

- g0 = contact type group
- g1 = contact affinity group
- g2 = ignore group（位掩码）
- g3 = id group（仅低 16 位有效）

默认值 `[1, 1, 0, 0]`：互相能接触，无忽略、无 ID。

---
