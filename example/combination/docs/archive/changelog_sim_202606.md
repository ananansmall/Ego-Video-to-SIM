## [2026-06-29] 04_physics_simulation.py 单夹爪模式物理稳定修复 (settle period + 参数统一)

**类型**: 修复
**影响范围**: `04_physics_simulation.py` (单夹爪模式 + 物理参数)

### 根因分析

用户反馈: "渲染的仿真能一下子把盘子给砸飞了，物理仿真引擎不对"

**根因**: 单夹爪模式 (`run_single_gripper_tracking`) **没有物体稳定化 (settle period)**！

| 模式 | 函数 | 物体稳定化 |
|------|------|-----------|
| 机械臂模式 | `run_robot_tracking` (L2640-2645) | ✓ 500+100=600步物理稳定 |
| **单夹爪模式** | `run_single_gripper_tracking` | **✗ 无稳定化！** |

GLB 的 7 个几何体全是 dynamic，初始位置可能互相接触/重叠。机械臂模式有 600 步让物体自然落下稳定，但单夹爪模式直接开始渲染——第一个物理步就把重叠的物体弹飞了。

### 修改内容

#### Fix 1: 单夹爪模式加 600 步 settle period (L2315-2337)

```python
# 加载 GLB 物体 (带碰撞体)
glb_actors = []
if glb_path is not None and ...:
    glb_actors = load_glb_with_physics(...)
    if glb_actors:
        self._log_object_positions(glb_actors, "GLB物体初始位置")
        # 物理稳定: 让dynamic物体自然落下, 消除初始重叠/穿透
        dynamic_actors = [a for a in glb_actors if ...]
        if dynamic_actors:
            for _ in range(500):
                self.scene.step()
            for _ in range(100):
                self.scene.step()
        self._log_object_positions(glb_actors, "GLB物体稳定后位置")
```

效果: 物体先下落稳定, 消除初始穿透, 再开始渲染。

#### Fix 2: GLB 物体物理参数调整 (L1023-1027)

| 参数 | 修改前 | 修改后 | 原因 |
|------|--------|--------|------|
| static_friction | 0.5 | 0.8 | 增加摩擦, 防止物体滑动 |
| dynamic_friction | 0.5 | 0.8 | 同上 |
| restitution | 0.3 | 0.0 | 完全不弹, 防止碰撞弹飞 |

#### Fix 3: 桌面参数统一 (单夹爪 + 机械臂模式)

| 参数 | 修改前 | 修改后 | 位置 |
|------|--------|--------|------|
| 桌面半厚度 | 0.015 (3cm) | 0.025 (5cm) | L2306, L2595 |
| 桌面 restitution | 0.1 | 0.0 | L2308, L2624 |

桌面加厚到 5cm, 更容易看到。restitution 降到 0.0, 物体落在桌面上不弹。

### 验证结果

运行命令:
```
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json python 04_physics_simulation.py \
    --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result --single-gripper --fast-collision
```

日志关键数据:
- 物体初始 Z: 0.0294~0.0683 (悬空)
- 物体稳定后 Z: 0.0113~0.0189 (落在桌面, ground_height=0.0078)
- 渲染: 113帧, 46秒, 2.45 fps
- 输出: physics_sim_physics_tracking_gripper.mp4 (0.7MB)
- **物体没有飞！** ✓

### 文档同步
- `doc/simulation_doc/questions.md` — 追加 Q&A 记录 settle period 修复

---

## [2026-06-27] 04_physics_simulation.py 坐标系对齐修复 (对齐 02: 相机 R_AXIS + GLB/手部 RXWORLD_TO_SAPIEN + FLIP_Z=False)

**类型**: 修复
**影响范围**: `04_physics_simulation.py` (相机函数 + FLIP_Z 常量, 回退之前的 R_AXIS 错误方案)

### 失败方案回顾 (已回退)

| 尝试 | 方案 | 结果 | 原因 |
|------|------|------|------|
| 第1次 | FLIP_Z=False, 相机用 RXWORLD_TO_SAPIEN | 相机错、桌子消失、盘子飞 | 相机多乘 R_x, 与 GLB/手部不同帧 |
| 第2次 | GLB/手部改 R_AXIS | 物体和夹爪消失 | 手部数据是 SLAM (非 OpenGL), 少乘 R_x |

### 根因分析 (对比 02_render_scene.py)

通过 search agent 完整研究 02_render_scene.py 发现关键差异:

**02 的设计**: 所有变换最终同帧 `R_AXIS @ OpenGL_world`
- 相机: `R_AXIS @ t_c2w` (t_c2w 是 OpenGL 数据, R_x 已应用)
- 手部/GLB: `RXWORLD_TO_SAPIEN @ SLAM = R_AXIS @ R_x @ SLAM = R_AXIS @ OpenGL`
- **同帧!** 02 无 FLIP_Z 逻辑

**04 之前的问题**:
- 相机用 `RXWORLD_TO_SAPIEN` → `R_AXIS @ R_x @ OpenGL` (多乘 R_x) ✗
- GLB/手部用 `RXWORLD_TO_SAPIEN` + FLIP_Z=True → 镜像 (FLIP_Z det=-1) ✗

### 正确方案 (对齐 02)

#### Fix 1: FLIP_Z_FOR_PHYSICS = False (L89)

关闭所有 Z 翻转, 消除镜像 (FLIP_Z = diag(1,1,-1), det=-1 是反射变换)。

#### Fix 2: 相机改用 R_AXIS (L690-694, L705-709)

```python
# 修改前: RXWORLD_TO_SAPIEN @ t_c2w (多乘 R_x, 与 GLB/手部不同帧)
# 修改后: R_AXIS @ t_c2w (对齐 02, OpenGL 数据用 R_AXIS)
cam_pos_sapien = R_AXIS @ t_c2w
cam_R_sapien = R_AXIS @ R_c2w
# forward/left/up 符号也完全对齐 02 的 hawor_cam_to_sapien_pose
forward = cam_R_sapien[:, 2]
left = -cam_R_sapien[:, 0]
up = -cam_R_sapien[:, 1]
```

#### Fix 3: GLB/手部/手腕保持 RXWORLD_TO_SAPIEN (不改)

- L793: `vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T` (GLB 顶点)
- L994: 同上 (load_glb_with_physics)
- L1520: `result = (RXWORLD_TO_SAPIEN @ pts_render.T).T` (手部点)
- L2746: `wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T` (手腕旋转)

**为什么不改**: pred_trans 是 SLAM 数据 (非 OpenGL), 需完整 `R_AXIS @ R_x` 变换。02 也是这样做的。

### 同帧验证

| 数据 | 输入帧 | 变换 | 最终帧 |
|------|--------|------|--------|
| 相机 t_c2w/R_c2w | OpenGL (R_x 已应用) | R_AXIS @ OpenGL | R_AXIS @ OpenGL ✓ |
| 手部 pred_trans | SLAM | RXWORLD_TO_SAPIEN @ SLAM = R_AXIS @ R_x @ SLAM | R_AXIS @ OpenGL ✓ |
| GLB 顶点 | RAS → SLAM (经 R_inv) | RXWORLD_TO_SAPIEN @ SLAM | R_AXIS @ OpenGL ✓ |

三者同帧, det=+1 (无镜像)。

### 物体 Z 负值处理 (地面自动降低)

FLIP_Z=False 后物体可能在 Z 负值 (SLAM Y 向下), 但 `support_plane` 机制兜底:
- `_compute_object_support_plane` 计算 GLB 顶点 Z 范围 (min_z)
- `ground_height = support_plane['min_z'] - 0.002` (地面降到物体下方)
- `scene.add_ground(ground_height)` (地面在 Z 负值, 物体自然落在地面上)
- 桌子 actor 在 `ground_height - 0.015` (桌子可见)

### GUI 模式

04 支持 `--viewer` 参数 (L3456), 启用交互式 Viewer (不保存视频):
```bash
python 04_physics_simulation.py --hawor-dir ... --ras-dir ... --single-gripper --viewer
```

### 验证
- 语法检查: ✓ 通过 (GetDiagnostics 无错误)
- 变换矩阵检查: ✓ 相机用 R_AXIS (L693-694), GLB/手部用 RXWORLD_TO_SAPIEN (L793/994/1520/2746), FLIP_Z=False
- 运行验证: 待用户运行
- 预期: 相机与 GLB/手部同帧, 无镜像, 桌子可见, 物体不弹飞

### 文档同步
- `doc/simulation_doc/questions.md` — 追加 Q&A 记录三种方案对比

---

## [2026-06-26] 04_physics_simulation.py GLB/夹爪镜像根因修复 (FLIP_Z_FOR_PHYSICS=False)

**类型**: 修复
**影响范围**: `04_physics_simulation.py` (1 处常量修改, 影响 5 个使用点)

### 根因分析 (对比 02_render_scene.py)

用户反馈: "z轴都是对的, x,y的坐标问题, 有点镜像"。对比 02 和 04 发现:

| 位置 | 02 (正确) | 04 (镜像) |
|------|----------|----------|
| `_render_to_sapien` (手部点) | `RXWORLD_TO_SAPIEN @ pts` | 同上 + **`result[..., 2] = -result[..., 2]`** |
| GLB 顶点 | `RXWORLD_TO_SAPIEN @ vertices` | 同上 + **`vertices[:, 2] = -vertices[:, 2]`** |
| 手腕旋转 | `RXWORLD_TO_SAPIEN @ R @ RXWORLD_TO_SAPIEN.T` | 同上 + **`Z_FLIP_R @ R @ Z_FLIP_R`** (det=-1 共轭) |
| 相机 | 用 `R_AXIS` | 用 `RXWORLD_TO_SAPIEN` + Z 翻转分支 |

**根因**: `FLIP_Z_FOR_PHYSICS=True` (L93) 在 5 个地方翻转 Z。Z 翻转 `result[..., 2] = -result[..., 2]` 是**反射变换 (det=-1)**, 虽然让 Z 位置变正 (物体从地下到地上), 但改变了坐标系手性, 导致 GLB/夹爪在 **X,Y 方向镜像** (Z 位置对, 左右手翻转)。

**为什么 02 没问题**: 02 用 kinematic + `set_qpos`, 物体在 Z 负值不掉, 不需要翻转 Z。04 用 dynamic, 物体在地下会掉, 所以错误地用 Z 翻转"修复", 但导致镜像。

### 修改内容

#### Fix: `FLIP_Z_FOR_PHYSICS = True → False` (L93)

```python
# 修改前
FLIP_Z_FOR_PHYSICS = True

# 修改后
FLIP_Z_FOR_PHYSICS = False
```

修改后 5 个 Z 翻转点全部不执行, GLB/手部/手腕映射与 02_render_scene.py 完全一致:
1. L697-710 相机: 走 else 分支, 不翻转
2. L799-800 _compute_object_support_plane: 不翻转
3. L1000-1001 load_glb_with_physics: 不翻转 (消除 GLB 镜像)
4. L1533-1534 _render_to_sapien: 不翻转 (手部点与 02 一致)
5. L2759-2761 warm_start: 不做 Z_FLIP_R 共轭 (消除夹爪手性翻转)

### 物理重力处理 (无需额外修改)

FLIP_Z=False 后物体在 Z 负值 (地下), 但:
- `_compute_object_support_plane` 计算物体 Z 范围 (min_z 为负值)
- `ground_height = support_plane['min_z'] - 0.002` (地面自动降到物体下方)
- `scene.add_ground(ground_height)` 地面在 Z 负值, 物体在地面之上
- 重力保持默认 `-Z`, 物体自然落在地面上

这正是用户建议的"可以把 z=0 平面降下去, 如果在反面的话"。

### 验证
- 语法检查: ✓ 通过 (GetDiagnostics 无错误)
- 运行验证: 待用户运行 `python 04_physics_simulation.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result --single-gripper --fast-collision`
- 预期: GLB 不镜像, 夹爪朝向正确, 物体在 Z 负值但地面自动降低支撑

### 文档同步
- `doc/simulation_doc/questions.md` — 追加 Q&A 记录对比分析

---

## [2026-06-26] 04_physics_simulation.py 夹爪不对称根因修复 (物理碰撞禁用)

**类型**: 修复
**影响范围**: `04_physics_simulation.py` (2 处: `_setup_robot` 全臂模式 + `run_single_gripper_tracking` 单夹爪模式)

### 根因分析 (基于 DEBUG 日志 + URDF/SRDF/SAPIEN API 调研)

v2 修改 (`gripper_target2 = gripper_target1`) 目标层面正确, 但 mp4 无可见变化, 因为实际 qpos 仍不对称。

DEBUG 输出 (帧0):
```
gripper_target1=0.025095, gripper_target2=0.025095  (对称目标 ✓)
actual_qpos[idx1]=0.000000, actual_qpos[idx2]=0.037781  (严重不对称!)
finger_link1 接触数=3, finger_link2 接触数=7
接触: finger_link2 <-> finger_link1 (自碰撞!)
      finger_link2 <-> right_realsense_link (相机碰撞!)
```

**调研用户提到的 3 种可能性** (均非根因):
1. **mimic joint**: URDF 无 `<mimic>` 标签, 但约定 A (retargeting 和 SAPIEN 都用) 只需 `q1 = q2` 即可对称, 等价 multiplier=1 的隐式 mimic, v2 已做
2. **URDF 参数化**: 存在两套约定 (A/B), 但 retargeting (`robot.urdf`) 和 SAPIEN (`r1_v2_1_0_floating_right.urdf`) 都用约定 A, **一致**
3. **mapping**: DEBUG 证明 `sapien_qpos` 与 retargeting 输出一致, 正确

**真正根因**: 物理碰撞 + SRDF 未加载
- `robot.srdf` L260 已声明 `right_gripper_finger_link1` 和 `right_gripper_finger_link2` 互不碰撞 (`reason="Default"`)
- 但 `_setup_robot` 用 `loader.load(arm_urdf_path)` 加载 URDF 时**没传 SRDF**, `_prepare_arm_urdf` 也没处理 SRDF
- 所以 SRDF 声明未生效, 手指之间碰撞未被禁用
- 碰撞力把 joint1 卡在 0, joint2 推到 0.038

### 修改内容

#### Fix 1: `_setup_robot` (全臂模式, L1872-1895) - 禁用手指间自碰撞 + realsense_link 碰撞

用 SAPIEN `set_collision_groups` API:
- **finger1-finger2**: 设置相同 ignore group bit (g2=1) + 相同 id (g3=1), 两手指互相忽略碰撞
- **realsense_link**: 设置 `[0,0,0,0]` 完全禁用碰撞 (相机不需要物理碰撞)
- **保留** finger 与 GLB/桌面碰撞 (用于抓取物体)

```python
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

#### Fix 2: `run_single_gripper_tracking` (单夹爪模式, L2377-2389) - 禁用手指间自碰撞

单夹爪模式无 realsense_link (用 `_GRIPPER_ONLY_URDF_TEMPLATE`), 只需禁用 finger1-finger2 碰撞, 逻辑同 Fix 1。

### SAPIEN collision_groups API 说明

碰撞判定公式 (两 shape A, B):
1. `(A.g0 & B.g1) or (A.g1 & B.g0)` — contact type 与对方 affinity 有交集
2. `not ((A.g2 & B.g2) and (A.g3 & 0xffff == B.g3 & 0xffff))` — 不在忽略对中

两条件同时成立才碰撞。默认 `[1, 1, 0, 0]`。

参考范式: `tri_model_physics/grasp_hawor.py` L691-709 (`set_collision_groups([0,0,0,0])` 禁用非夹爪 link 碰撞)。

### 验证
- 语法检查: ✓ 通过 (GetDiagnostics 无错误)
- 运行验证: 待用户运行 `python 04_physics_simulation.py --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result --fast-collision`
- 预期: actual_qpos[idx1] ≈ actual_qpos[idx2] (都接近 target 0.025), 夹爪对称打开

### 文档同步
- `doc/simulation_doc/questions.md` — 追加 Q&A 记录完整调研过程

---

## [2026-06-26] 04_physics_simulation.py GLB 对齐 + 夹爪单控制量 + 手部自动识别 v2

**类型**: 修复
**影响范围**: `04_physics_simulation.py` (4 类共 11 处微调)

### 根因分析 (基于 URDF 实际读取)
1. **GLB 镜像/未对齐**: RAS 导出的 GLB 可能是 Z-UP 坐标系, 但 `load_glb_with_physics` 未做检测, 直接按 Y-UP 处理, 导致物体在 SAPIEN 中朝向错误。
2. **夹爪只开合一边**: URDF `right_gripper_finger_joint1` axis=`0 -1 0`, `joint2` axis=`0 1 0` (axis 相反), limit 均为 `[0, 0.05]` (非负)。原代码 `idx2 = -0.04` 违反 limit, 且符号与 idx1 相反导致两指同向运动 (不对称)。正确做法: 两指**同号**值, 利用 axis 相反实现镜像开合。
3. **手部识别单一**: `_detect_hand_idx` 只返回 int (0/1/None), 无法表达 "双手" 场景。

### 修改内容

#### Fix 1: GLB Z-UP 自动检测与转换
- L98 添加 `ZUP_TO_YUP = np.array([[1,0,0],[0,0,1],[0,-1,0]])` 常量
- L130 添加 `_detect_glb_up_axis(all_vertices)` 函数 (复制自 02)
- `load_glb_with_physics` 中 (L949/L976/L982/L996):
  - 读取 `saved_glb_up_axis` from transform_params
  - 缺失时调用 `_detect_glb_up_axis` 自动判断
  - `need_zup_to_yup = (glb_up_axis == "z-up")` 时执行 `vertices = (ZUP_TO_YUP @ vertices.T).T`

#### Fix 2: 夹爪统一为单控制量 (5 处 -0.04 → 0.04 + 2 处镜像 + 2 处 clamp)
- L1823/L1824 init_qpos: `idx1 = idx2 = 0.04`
- L2823 warmup `_physics_step` 调用: `0.04, 0.04` (含说明注释)
- L3060/L3061 second-pass init_qpos: `0.04, 0.04`
- L3084/L3085 reset_qpos: `0.04, 0.04`
- L3208/L3209 second-pass reset_qpos: `0.04, 0.04`
- L2978-2980 单臂 PD 驱动:
  ```python
  gripper_target1 = max(0.0, min(0.05, gripper_target1))  # clamp
  gripper_target2 = gripper_target1  # 同号 → axis 相反 → 对称开合
  ```
- L3164-3166 second-pass PD 驱动: 同上模式 (gripper_t1/gripper_t2)

#### Fix 3: 手部自动识别 (左/右/双手)
- L481-530 `_detect_hand_idx` 重构:
  - 返回类型: `int` → `tuple(hand_idx, handedness_str)`
  - handedness_str ∈ `{"left", "right", "both", "unknown"}`
  - 新增 cam_space 双目录检测 → "both"
  - 新增 pred_valid npz 回退检测 (cam_space 缺失时)
- L3420-3427 调用方适配:
  ```python
  detected_idx, handedness = _detect_hand_idx(Path(args.hawor_dir))
  args.hand_idx = detected_idx
  # handedness 日志 + 中文标签 (含 "双手(默认左手)" 提示)
  ```
  注意: 04 仅支持单臂渲染 (与 02 一致), 双手场景默认取 idx=0 (左手), 调用方按需切换。

### 验证结果
- 语法检查: ✓ 通过 (`python3 -m py_compile` 返回 0)
- 9 项关键修复点: ✓ 全部通过
  - ZUP_TO_YUP 常量, _detect_glb_up_axis, saved_glb_up_axis 读取
  - need_zup_to_yup 判断, vertices ZUP_TO_YUP 转换
  - gripper_target1/gripper_t1 clamp [0, 0.05]
  - gripper_target2/gripper_t2 = gripper_target1/gripper_t1 (镜像)
  - _detect_hand_idx 返回元组 (4 种 handedness)
  - 调用方正确解包元组
- `-0.04` 残留检查: ✓ 文件中无 `-0.04`

### 文档同步
- `doc/simulation_doc/specs/2026-06-26-04-physics-glb-and-gripper-design-v2.md` — 基于实际 URDF 读取的 v2 设计文档 (含 3 根因 + 4 修复 + 验证策略)

---

## [2026-06-26] 04_render_dual_arm.py GLB 对齐与夹爪镜像修复

**类型**: 修复
**影响范围**: `04_render_dual_arm.py` (7 处微调)

### 修改内容
- `04_render_dual_arm.py` 修正 LEFT_ARM_STARTING 为 `[-1.5, -1.9508, 1.0809, -0.4438, -0.1709, 0.1985]` (与 02 一致)
- `04_render_dual_arm.py` 添加 ZUP_TO_YUP 常量 + `_detect_glb_up_axis` 函数 (复制自 02)
- `04_render_dual_arm.py` 替换 `load_glb_transformed` 函数体 (含 Z-UP 自动检测、saved_glb_up_axis 读取、gc.collect 内存释放)
- `04_render_dual_arm.py` 移除 `load_hawor_npz` 中 R_c2w/t_c2w 二次变换 (npz 数据已是世界坐标)
- `04_render_dual_arm.py` 替换 `detect_hands` 为内联实现 (Handedness/HandDetectionResult/_detect_hands_local), 移除缺失的 hand_detector import
- `04_render_dual_arm.py` 夹爪统一为单控制量: 初始姿态 idx2=0.04 同号 + retargeting 输出 gripper_open clamp [0,0.05] + idx2 镜像 idx1 (用户指正: URDF axis 相反+limit 非负, 同号才对称开合)
- `04_render_dual_arm.py` 添加 FPS 性能基线测量 (_frame_times + fps_mean/p50/p95 日志)

### 验证结果
- 语法检查: ✓ 通过
- 15 项关键修复检查: ✓ 全部通过
  - LEFT_ARM_STARTING 修复, ZUP_TO_YUP, _detect_glb_up_axis, saved_glb_up_axis, gc.collect
  - R_c2w 二次变换移除, hand_detector import 移除, Handedness 类, _detect_hands_local
  - 夹爪初始姿态同号/无负值, 单控制量, clamp, idx2 镜像 idx1, FPS 基线

### 文档同步
- `doc/simulation_doc/specs/2026-06-26-glb-alignment-and-gripper-mirror-design.md` — 已更新 (根因 2 修正为 URDF axis 分析, 修复 5 修正为单控制量)
- `doc/simulation_doc/specs/2026-06-26-glb-alignment-and-gripper-mirror-plan.md` — 已更新 (任务 6 修正为夹爪单控制量, 验证项更新)
