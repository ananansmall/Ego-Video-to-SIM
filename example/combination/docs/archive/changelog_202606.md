# 管线修改文档

> 最后更新: 2026-06-09
> 文件位置: `/home/an/robot_world_ws/src/dex-retargeting/example/combination/CHANGELOG.md`

***

## 1. GLB 坐标系对齐（核心问题）

### 1.1 问题背景

RAS 场景重建输出的 GLB 和 HaWoR 手部重建处于不同坐标系，需要对齐后才能在同一个 SAPIEN 场景中渲染。

### 1.2 坐标系分析

| 坐标系                        | 方向   | 单位 | 说明                 |
| -------------------------- | ---- | -- | ------------------ |
| RAS 外参 (extrinsics/\*.txt) | Z-UP | 米  | 房间对齐后的相机位姿         |
| RAS GLB (final\_scene.glb) | Y-UP | 米  | 导出时做了 z-up→y-up 转换 |
| HaWoR render world         | Y-UP | 米  | 手部重建的世界坐标系         |
| SAPIEN 世界                  | Z-UP | 米  | 物理仿真引擎坐标系          |

### 1.3 对齐方法：第一帧相机位姿锚定

**核心思想**：RAS 和 HaWoR 的第一帧相机位姿描述的是同一个真实相机，以此为锚点计算坐标系变换。

**变换链**：

```
RAS GLB 顶点 (Y-UP)
    ↓ p_hawor = s_inv * R_inv @ p_ras + t_inv
HaWoR render world (Y-UP)
    ↓ p_sapien = RXWORLD_TO_SAPIEN @ p_hawor
SAPIEN 世界 (Z-UP)
```

**R\_inv 和 t\_inv 的计算**（在 `01_align_scene.py` 中）：

1. RAS 外参是 Z-UP，需要先转为 Y-UP：
   ```
   R_c2w_ras_yup = ZUP_TO_YUP @ R_c2w_ras_zup
   t_c2w_ras_yup = ZUP_TO_YUP @ t_c2w_ras_zup
   ```
2. RAS 外参是 OpenCV 相机模型（Z 朝后），HaWoR 是 OpenGL 模型（Z 朝前）：
   ```
   R_c2w_ras_gl = OPENCV_TO_OPENGL @ R_c2w_ras_yup
   ```
3. 第一帧相机位姿对齐：
   ```
   R_align = R_c2w_hawor @ R_c2w_ras_gl.T
   t_align = t_c2w_hawor - R_align @ t_c2w_ras_yup
   ```
4. Umeyama 尺度校正：
   ```
   s_inv ≈ 0.321 (RAS 尺度 → HaWoR 尺度)
   ```
5. 最终变换参数：
   ```
   R_inv = R_align
   t_inv = t_align
   ```

### 1.4 关键矩阵

```python
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
OPENCV_TO_OPENGL = np.diag([1, -1, -1])
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
RXWORLD_TO_SAPIEN = R_AXIS @ R_x  # y-up → z-up + SLAM→render
```

### 1.5 验证结果

| 指标            | 值                         | 说明             |
| ------------- | ------------------------- | -------------- |
| R\_align 旋转角度 | 0.47°                     | 几乎单位旋转，方向对齐正确  |
| 手腕→GLB中心      | 0.1923m                   | 合理（手腕在 GLB 旁边） |
| 手腕→GLB最近顶点    | min=0.0004m, mean=0.0824m | 帧5几乎接触（0.4mm）  |
| 指尖→GLB最近顶点    | min=0.0006m, mean=0.0623m | 指尖频繁接触 GLB 表面  |

**逐帧手腕→GLB最近距离趋势**：

- 帧0-5: 0.042→0.0004m（手接近GLB）
- 帧5-10: 0.0004→0.036m（手离开GLB）
- 帧10-19: 0.045→0.084m（手持续远离）

**结论**：手部与 GLB 对齐正确，手确实在操作过程中接近并接触 GLB 物体。视频看起来"远"可能是因为相机 FOV=120° 导致的广角畸变。

### 1.6 之前的错误方法

- **yingshe.py (Umeyama)**：假设 RAS 外参是 Y-UP（实际是 Z-UP），导致旋转角度 178.83°（几乎翻转），手-GLB 距离 0.5071m

---

## 2. 修改记录

### 2026-06-09 修改：hand_track 模块整合 & 批量自动检测管线

**类型**: 新增 / 重构
**影响范围**: hand_track/ 模块、手部检测、机械臂映射

#### 修改内容

- [hand_track/run_all_hawor.py] 新增：批量自动检测 + 映射管线，基于 03_track_robot.py 逻辑，支持左手/右手/双手自动检测
- [hand_track/hand_detector.py] 保留：手部类型自动检测 (左手/右手/双手)
- [hand_track/robot_arm_config.py] 保留：根据手部类型生成机械臂配置
- [hand_track/render_auto.py] 保留：GLB 场景渲染模式
- [hand_track/test_pipeline.py] 保留：检测 + 映射验证测试
- [hand_track/__init__.py] 更新：模块说明文档
- [combination/hand_detector.py] 删除：已迁移至 hand_track/
- [combination/robot_arm_config.py] 删除：已迁移至 hand_track/
- [combination/utils/hand_detector.py] 删除：已迁移至 hand_track/
- [combination/utils/__init__.py] 更新：指向 hand_track/ 模块
- [test/test_hand_detection_and_mapping.py] 修改：导入路径从根目录改为 hand_track/

#### 关键改进

1. **自动跳过无效起始帧**: 当数据开头全是NaN时，自动找到第一个有效帧作为起始
2. **左手臂支持**: MANOLayer("left"), HandType.left, IK solve_position_left
3. **坐标变换自动检测**: 根据手腕质心和运动范围判断是否需要 hawor→SAPIEN 变换
4. **4个hawor目录全部验证通过**:

| 目录 | 检测结果 | 映射臂 | 有效帧 | qpos |
|------|---------|--------|--------|------|
| 7 | 左手 | left | 113/113 | ✓ |
| 7_vggt-omega | 左手 | left | 113/113 | ✓ |
| hoi4d | 双手 | left+right | 599/600 | ✓ |
| laptop | 右手 | right | 496/600 | ✓ |

---

### 2026-06-09 修改（第七轮）：函数注释补全 & 可视化输出

**类型**: 文档 / 用户体验
**影响范围**: hand_track/ 全模块

#### 修改内容

- [hand_track/render_auto.py] 为 `ArmController` 类及所有方法 (`__init__`, `_init_retargeting`, `_init_ik`, `_update_base_link`, `compute_optimal_base`, `compute_tracking_base_pos`, `get_gripper_pose_from_retargeting`, `solve_ik`, `compute_retargeting`, `solve_ik_for_result`, `compute_wrist_positions_sapien`, `do_warmup`, `do_mini_warmup`) 添加详细 docstring, 含流程说明、Args、Returns
- [hand_track/render_auto.py] `run_robot_only` 增补完整 docstring, 列出 7 步流程和关键设计
- [hand_track/render_auto.py] `_render_keypoints` 增补 docstring
- [hand_track/render_auto.py] `main` 增补 docstring, 含完整命令行参数说明
- [hand_track/render_auto.py] 新增 `_print_banner()` 启动横幅
- [hand_track/render_auto.py] 渲染主循环增加进度日志 (约 10 个进度点) 和有效/无效帧统计
- [hand_track/render_auto.py] `run_robot_only` 入口增加数据源/输出/视角/帧数/有效帧的可见输出
- [hand_track/render_auto.py] qpos 保存路径增加成功/失败日志, 末尾输出 "渲染统计" 表格

#### 用户体验提升

1. **启动横幅**: 清晰展示脚本名称和特性
2. **参数配置打印**: 启动时列出所有 CLI 参数值, 便于确认
3. **手部检测可见化**: 打印检测结果描述、方法、cam_space 状态
4. **7 步流程标记**: `[1/7] [2/7] ... [7/7]` 方便定位卡在哪一步
5. **进度日志**: 渲染过程中约 10 个进度点, 显示有效/无效帧数
6. **结尾统计**: 总耗时、渲染统计、输出路径全部可见

#### 验证结果

- `python -c "import ast; ast.parse(...)"` 5 个文件语法全部通过
- `python hand_track/test_pipeline.py` 全部 35 个测试通过 (0 失败, 0 跳过)

---

### 2026-06-09 修改（第八轮）：日志可见性修复 & 数据源明确化

**类型**: 修复 / 用户体验
**影响范围**: hand_track/run_all_hawor.py

#### 背景

用户反馈: `conda run -n dex python hand_track/run_all_hawor.py` 运行后看不到任何输出,
无法判断脚本是否在运行、卡在哪一步、是否成功检测到 `reconstruction/` 文件。

**根本原因**: Python 在非 TTY 场景下 (如 `conda run`、`nohup`、管道) 默认使用
**全缓冲** stdout, 导致 `logger.info` 输出被累计到 4KB+ 才 flush 一次。

#### 修复内容

- [run_all_hawor.py] **顶部** 强制设置 `PYTHONUNBUFFERED=1` + `sys.stdout.reconfigure(line_buffering=True)` + 提前 `import sys`
- [run_all_hawor.py] **最早** 添加 `print("[run_all_hawor.py] 启动, 正在加载依赖 ...", flush=True)`, 即使后续 `import` 卡住也能看到提示
- [run_all_hawor.py] `main()` 入口在 logger 初始化前先用 `print(..., flush=True)` 打印关键参数
- [run_all_hawor.py] `main()` 给 logging StreamHandler 重新设置 `line_buffering=True`
- [run_all_hawor.py] `process_hawor_dir` 增加 `_find_reconstruction_file` 预先检查, 找不到时立即打印期望路径
- [run_all_hawor.py] `load_hawor_data` 新增 `logger` 参数, 逐步打印: 找到的文件、shape 维度、NaN 帧数等
- [run_all_hawor.py] `_find_reconstruction_file` 文档更新, 明确主路径是 `reconstruction/hawor_results_*.npz`
- [run_all_hawor.py] 关键步骤后调用 `sys.stdout.flush()`, 避免缓冲

#### 数据源说明 (回答用户问题)

**只需要 `reconstruction/` 这个文件夹**:
- 主路径: `<hawor_dir>/reconstruction/hawor_results_*.npz`
- 兜底 1: `<hawor_dir>/*.npz` (根目录的 npz 文件)
- 兜底 2: `<hawor_dir>/world_space_res.pth` (兼容旧版 HaWoR)

**完全不需要**:
- `cam_space/` (仅作辅助检测手部, 不用于 retargeting)
- `cam_param/`
- 其他元数据目录

#### 验证结果

```
$ conda run -n dex python hand_track/run_all_hawor.py --hawor-dirs /home/an/data/hawor/7 --no-render --num-frames 30
[run_all_hawor.py] 启动, 正在加载依赖 ...
[run_all_hawor.py] 参数解析完成, 启动时间: 12:10:48
[run_all_hawor.py] --hawor-base = /home/an/data/hawor
12:10:48 [INFO]   ✓ 发现 reconstruction 文件: reconstruction/hawor_results_0_113.npz
12:10:48 [INFO]   正在检测手部类型 ...
12:10:48 [INFO]   检测结果: 左手 (有效帧: 113/113, 占比: 100.0%)
12:10:48 [INFO]   检测方法: pred_valid+cam_space
12:10:48 [INFO]   [load_hawor_data] ✓ 找到 reconstruction npz: hawor_results_0_113.npz
12:10:48 [INFO]   [load_hawor_data] pred_trans.shape=(2, 113, 3), pred_valid.shape=(2, 113)
12:10:48 [INFO]   [load_hawor_data] 已完成相机→世界坐标变换 (113 帧)
12:10:48 [INFO]   [load_hawor_data] NaN 帧: 0 (已在 pred_valid 中置为 False)
```

完整输出含: 启动横幅、参数配置、扫描到的目录数、每个目录的 reconstruction 状态、
手部检测过程、数据加载详情、NaN 过滤、warmup、IK 进度、视频渲染、汇总表、总耗时。

---

### 2026-06-15 修改（第十六轮）：run_all_hawor.py 改用浮动臂 + 单场景 + 单视频

**类型**: 核心重构
**影响范围**: hand_track/run_all_hawor.py, doc/pipeline.md

#### 背景

用户要求 `run_all_hawor.py` 与 `02_render_scene.py` 的 `run_robot_only` 方法完全一致:
- 旧版使用 `RobotHandDatasetSAPIENViewer` 加载**完整 R1 机器人** (body + 双臂)
- 新版应使用 `prepare_arm_urdf` + `loader.load` 加载**浮动臂** (只有单臂, 无机器人身体)
- 旧版双手时生成两个视频, 新版应生成一个视频

#### 修改

**1. 移除 `RobotHandDatasetSAPIENViewer` 依赖**

不再 import `hand_robot_viewer`，改用 `setup_scene()` + `prepare_arm_urdf()` + `loader.load()` 加载浮动臂。

**2. 新增函数 (与 02_render_scene.py 一致)**

- `setup_scene()` — 创建 SAPIEN 渲染场景 (灯光 + 环境贴图)
- `_compute_optimal_fixed_base(wrist_positions, logger)` — 计算浮动臂基座最优位置
- `_compute_tracking_base_pos(initial_base_pos, gripper_pos, arm_base_q)` — 基座小范围跟踪夹爪
- `_get_arm_base_link_pose(robot, base_link_name)` — 获取浮动臂 base_link 世界位姿

**3. 删除旧函数**

- `run_single_arm_pipeline()` — 旧的单臂管线 (基于完整 R1 机器人)
- `precompute_frames()` — 旧的预计算函数

**4. 重写 `process_hawor_dir()`**

- 所有臂在同一个 SAPIEN 场景中
- 每个臂独立加载浮动 URDF + 初始化 Retargeting
- 共享一个 `RelaxedIKSolver`
- 使用 `_compute_optimal_fixed_base` + `_compute_tracking_base_pos` (与 02 一致)
- 浮动臂不需要 `mapping_offset` / `safety_offset`
- 双手输出: `hawor_r1_dual_tracking.mp4` + qpos dict `{left: np.array, right: np.array}`
- 单手输出: `hawor_r1_{left|right}_tracking.mp4` + qpos np.array

**5. 重写 `render_video()`**

- 接受 `robots` (list) + `arm_states` (list) 参数
- 所有臂在同一画面渲染
- 相机基于所有臂基座质心定位

**6. 常量更新 (与 02 一致)**

| 常量 | 旧值 | 新值 |
|---|---|---|
| `SAFETY_DISTANCE` | 0.075 | 0.05 |
| `COMFORTABLE_REACH` | 0.40 | 0.35 |
| `COMFORT_TARGET_IN_BASE` | [0.30, 0.0, -0.25] | [0.30, 0.0, -0.30] |
| `BASE_TRACKING_RANGE` | (无) | 0.04 |
| `ARM_BASE_OFFSET_LOCAL` | [0.09, -0.34, 0.97] | (删除) |

**7. 修复 `_print_summary_table()`**

双臂 qpos 为 dict 类型，需 `allow_pickle=True` 加载。

**8. 更新 doc/pipeline.md**

移除 `RobotHandDatasetSAPIENViewer` 引用，更新关键特性描述。

#### 验证

- ✅ 单手 (hawor/7): `hawor_r1_left_tracking.mp4` + qpos (40, 8)
- ✅ 双手 (hoi4d): `hawor_r1_dual_tracking.mp4` + qpos dict {left: (40,8), right: (40,8)}

---

### 2026-06-15 修改（第十五轮）：run_all_hawor.py 与 02_render_scene.py 对齐

**类型**: 关键修复 (核心算法一致化)
**影响范围**: hand_track/run_all_hawor.py

#### 背景 (用户反馈)

用户测试 `hand_track/run_all_hawor.py` 后指出: **"生成的怎么还是和 02_render_scene.py 不一样呢"**。

#### 根因分析 (3 个核心差异)

通过逐行对比 02_render_scene.py 的 `run_robot_only()`, 发现 3 个关键算法差异:

| 维度 | 02_render_scene.py | run_all_hawor.py (之前) |
|---|---|---|
| **夹爪位姿** | `_get_gripper_pose_from_retargeting()` 用 retargeting 内部 FK 算 (考虑手指弯曲) | `R_mano2world @ gripper_align_R` 合成 (只考虑手腕) |
| **目标平滑** | `EmaTargetSmoother(pos_alpha=0.6, ori_alpha=0.6)` | 无 |
| **IK 迭代次数** | 1000 次 warmup + 每帧 20 次反复求解 | 200 次 warmup + 每帧 1 次 |

**夹爪位姿差异是最关键的**:
- 02 用 retargeting 内部 FK 算出 `gripper_R` (考虑手指实际弯曲)
- 我的代码用 `wrist_R @ operator2mano.T @ gripper_align_R` 合成, 只看手腕

#### 修改

**1. 新增 `_get_gripper_pose_from_retargeting()`** (与 02 一致)

```python
def _get_gripper_pose_from_retargeting(retargeting, retarget_qpos, arm_prefix="right"):
    internal_robot = retargeting.optimizer.robot
    internal_robot.compute_forward_kinematics(retarget_qpos)
    target_name = f"{arm_prefix}_gripper_link"
    for i, name in enumerate(internal_robot.link_names):
        if name == target_name:
            pose = internal_robot.get_link_pose(i)
            return pose[:3, 3].copy(), pose[:3, :3].copy()
```

**2. 新增 `EmaTargetSmoother` 类** (与 02 一致)

```python
class EmaTargetSmoother:
    def __init__(self, pos_alpha=0.3, ori_alpha=0.3):
        self.pos_alpha = pos_alpha
        self.ori_alpha = ori_alpha
        self.pos = None
        self.ori_quat = None

    def smooth(self, pos, ori_quat):
        if self.pos is None:
            self.pos = pos.copy()
            self.ori_quat = ori_quat.copy()
            return self.pos.copy(), self.ori_quat.copy()
        self.pos = self.pos + self.pos_alpha * (pos - self.pos)
        self.ori_quat = self.ori_quat + self.ori_alpha * (ori_quat - self.ori_quat)
        ...
```

**3. 改用 `R_ee_world_fk`** (替换 `R_mano2world @ gripper_align_R`)

```python
# 之前
R_ee_base = base_link_R_inv @ R_mano2world @ gripper_align_R

# 现在
gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(
    retargeting, retarget_qpos, arm_prefix
)
ee_R_base = base_link_R_inv @ R_ee_world_fk
```

**4. 应用 EMA 目标平滑** (主循环中)

```python
ik_target_base, ee_quat_base = target_smoother.smooth(ik_target_base, ee_quat_base)
```

**5. 每帧 20 次 IK 反复求解**

```python
arm_joints = np.array(ik_solver.solve_position_...(ik_target_base.tolist(), ee_quat_base.tolist()))
for _ in range(IK_SOLVE_PER_FRAME - 1):
    arm_joints = np.array(ik_solver.solve_position_...(ik_target_base.tolist(), ee_quat_base.tolist()))
```

**6. 新增 `IK_SOLVE_PER_FRAME = 20` 常量**

#### 验证

测试 `--hawor-dirs /home/an/data/hawor/7 --num-frames 5`:
- ✓ 35 帧全部有效 (30 warmup + 5 data)
- ✓ 视频生成: `hawor_r1_left_tracking.mp4` (480KB)
- ✓ qpos shape: (35, 26) - 26 维 SAPIEN qpos
- ✓ 总耗时 7.3s

#### 后续

`04_render_dual_arm.py` 已经使用类似 02 的 `_get_gripper_pose` 模式 (内部已直接用 FK 算 EE), 不需要修改。

---

### 2026-06-15 修改（第十四轮）：04_render_dual_arm.py 双臂同视频渲染 + 综合输出日志

**类型**: 用户体验 + 验证
**影响范围**: 04_render_dual_arm.py

#### 背景 (用户 3 个核心要求)

1. **双手 → 一个视频**: 双手数据时, 左右臂必须**同时**渲染到**一个**mp4 文件
2. **相对自由的机械臂**: 渲染方式像 02_render_scene.py 一样, 用 floating URDF, 机械臂可以在 3D 空间自由移动 (set_root_pose 跟踪手腕)
3. **输出日志**: 每次输出视频后必须有综合 log (路径/帧数/大小/qpos)

#### 验证结果

**1. BOTH 手 → 一个视频** ✓

测试 `--hawor-dir /home/an/data/hawor/hoi4d` (BOTH):
```
✓ 动态创建 2 个臂: ['left', 'right']
[left] 基座=(-0.21, 0.44, -0.04)
[right] 基座=(0.15, 0.45, -0.06)
渲染臂数: 2 (left, right)
qpos 形状: left=(15, 8), right=(15, 8)
视频路径: /tmp/test_dual_both.mp4
```

**2. 自由浮动机械臂 (FLOATING URDF)** ✓

测试启动时打印:
```
RelaxedIK is using below URDF file:
  /home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/configs/urdfs/r1_v2_1_0_floating_left.urdf
RelaxedIK is using below URDF file:
  /home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/configs/urdfs/r1_v2_1_0_floating_right.urdf
```

完全模仿 02_render_scene.py 的 robot_only 模式:
- `loader.fix_root_link = True` 锁住 root_link (但 root 是 free joint, 可平移旋转)
- `self.robot.set_root_pose()` 每帧更新浮动基座位置
- 浮动基座跟随手腕质心 (BASE_TRACKING_RANGE=0.04m 范围)
- 与手腕相对位置保持在 COMFORT_TARGET_IN_BASE = [0.30, 0.0, -0.30]

**3. 输出综合日志** ✓ (用户关键要求)

视频输出后的汇总块:
```
================================================================================
  ✓ 视频 + qpos 输出完成
================================================================================
  视频路径:   /tmp/test_dual_left.mp4
  视频格式:   H.264  帧率: 30  分辨率: 1920x1080
  视频帧数:   10 (与请求 --num-frames=10 一致)
  视频大小:   22.6 KB
  渲染臂数:   1 (left)
  视角:       behind
  渲染耗时:   1.7 秒 (FPS: 5.8 帧/秒)

  qpos 路径:  /tmp/test_dual_left.npy
  qpos 形状:  left=(10, 8)
  qpos 含义:  每帧 SAPIEN qpos (6 臂关节 + 2 夹爪 = 8 维)
              left: range=[-0.764, 2.221], mean=0.569
================================================================================
```

#### 关键改进

- 新增 `_t_render_start` / `_t_render_end` 计算渲染耗时和实际 FPS
- 修复 `reencode_with_ffmpeg` 中小文件显示 "0.0MB" 的 bug (改用 KB)
- 输出汇总块包含: 路径/格式/帧数/大小/臂数/视角/耗时/FPS/qpos 路径/qpos 形状/qpos 含义
- 保留旧 log 兼容: 同时显示 mp4v/H.264 格式, 显示 SAPIEN qpos 范围/均值

---

### 2026-06-15 修改（第十三轮）：run_all_hawor.py 输出追踪修复

**类型**: 用户体验 (关键修复)
**影响范围**: hand_track/run_all_hawor.py

#### 背景 (用户反馈)

用户报告: `conda run -n dex python hand_track/run_all_hawor.py --hawor-dirs /home/an/data/hawor/hoi4d ...` 
**完全看不到有没有正在运行的痕迹**, 太差。

#### 根因分析

经调试, 发现两个核心问题:
1. **SAPIEN viewer 初始化慢/失败**: 24+ 秒静默 (导入 MANO 模型 + SAPIEN 内部初始化)
2. **GPU/Vulkan 设备不可用时直接 segfault**: `RuntimeError: failed to find a rendering device` 后立即 `段错误 (核心已转储)`, 错误信息混杂在 logger 输出中难以发现

#### 修改内容

**1. 8 步进度标记 (核心改进)**

在 `run_single_arm_pipeline` 每个关键步骤添加 `time.time()` + 日志:

| Step | 操作 | 典型耗时 |
|---|---|---|
| 1/8 | 初始化 MANOLayer | 0-5s (首次加载) |
| 2/8 | 确定坐标变换 | <1s |
| 3/8 | 初始化 SAPIEN viewer | 1.5-10s |
| 4/8 | 分析手部轨迹 | <1s |
| 5/8 | 配置 R1 机器人 + 工作区映射 | <1s |
| 6/8 | 初始化 RelaxedIK | 1-2s |
| 7/8 | 预计算帧 (含 30 帧 warmup) | 0.1-N 秒 |
| 8/8 | 渲染视频 | 取决于帧数 |

每步都打印 `[step N/8] ...` 和完成后的耗时 `✓ ... (耗时 X.Xs)`。

**2. 帧循环进度 (10% 间隔)**

在 `precompute_frames` 和 `render_video` 的主循环中, 每完成 10% 打印一次:
```
[precompute] 帧 60/600 (10%)
[precompute] 帧 120/600 (20%)
...
[render] 10% (60/600)
[render] 20% (120/600)
```

**3. GPU/Vulkan 设备预检**

新增 `_check_rendering_device()` 函数, 在 SAPIEN init 之前运行:
- 检查 `nvidia-smi -L` 是否能找到 GPU
- 备选检查 `vulkaninfo --summary`
- 没有 GPU 时给出明确警告 + 解决建议

```
[环境检查] 检测渲染设备 ...
[GPU] ✓ 检测到 NVIDIA GPU: GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU
```

**4. SAPIEN 错误捕获**

在 `viewer = RobotHandDatasetSAPIENViewer(...)` 外层加 try/except, 捕获 `RuntimeError("rendering device")`:
```
[step 3/8] ✗ SAPIEN 初始化失败: failed to find a rendering device
[step 3/8]   原因: 没有找到可用的渲染设备 (Vulkan/GPU)
[step 3/8]   建议:
            1) 运行 nvidia-smi 确认 GPU 存在
            2) 检查 /usr/share/vulkan/icd.d/ 下的 ICD 文件
            3) 更新 NVIDIA 驱动 (CUDA 12020 太旧)
            4) 尝试 headless 渲染: export SAPIEN_HEADLESS=1
```

**5. 修复路径 bug**

`robot_dir` 之前是 `PROJECT_ROOT / "dex-retargeting" / "assets" / "robots" / "hands"` (双重 dex-retargeting), 
改为 `PROJECT_ROOT / "assets" / "robots" / "hands"` (PROJECT_ROOT 已经是 dex-retargeting)。

#### 验证

- **GPU 缺失环境**: 立即显示 `[环境检查] ⚠ 未检测到可用的渲染设备`, 用户清楚知道问题
- **GPU 正常环境**: 显示完整 8 步进度, 总耗时 1.8s (5 帧 7 目录)

```
22:29:55 [INFO]   [GPU] ✓ 检测到 NVIDIA GPU: GPU 0: NVIDIA GeForce RTX 3050 Laptop GPU
22:29:55 [INFO]   [step 1/8] 初始化 MANOLayer (side=left) ...
22:29:55 [INFO]   [step 1/8] ✓ MANOLayer 就绪 (耗时 0.0s)
22:29:55 [INFO]   [step 2/8] 确定坐标变换 ...
22:29:55 [INFO]   [step 3/8] 初始化 SAPIEN 场景 + R1 机器人 (约 5-10s) ...
22:29:57 [INFO]   [step 3/8] ✓ SAPIEN viewer 就绪 (耗时 1.5s)
22:29:57 [INFO]   ✓ Dex Retargeting 就绪 (HandType=left)
22:29:57 [INFO]   [step 4/8] 分析手部轨迹 ...
22:29:57 [INFO]   [step 5/8] 配置 R1 机器人 + 计算工作区映射 ...
22:29:57 [INFO]   [step 6/8] 初始化 RelaxedIK ...
22:29:57 [INFO]   ✓ RelaxedIK 就绪
22:29:57 [INFO]   [step 7/8] 预计算 5 帧 (含 30 帧warmup) ...
22:29:57 [INFO]   [step 7/8] ✓ 预计算完成: 35/35 帧有效 (耗时 0.1s)
22:29:57 [INFO]   --no-render 模式: 跳过视频渲染
22:29:57 [INFO]   7                    left     left     ok         (35, 26)       ✗ 
22:29:57 [INFO]   总耗时:   1.8 秒
22:29:57 [INFO]   ✓ 完成!
```

---

### 2026-06-10 修改（第十二轮）：04_render_dual_arm.py 泛化重构

**类型**: 关键修复
**影响范围**: 04_render_dual_arm.py

#### 背景 (用户关键反馈)

用户指出之前的 `04_render_dual_arm.py` 实际上**不是泛化的**:
- 硬编码 `hand_idx=0` (左) 和 `hand_idx=1` (右)
- 不管检测到几只手, 都同时创建 2 个臂
- 真正的目标: **脚本不预先知道是哪些手, 由 reconstruction 文件自动决定**

#### 核心重构: 自动检测 + 动态创建

**新设计**:
```
HandDetector.detect() 读取 reconstruction/*.npz
  ├─ LEFT  → 动态创建 1 个 ArmInstance("left",  0)
  ├─ RIGHT → 动态创建 1 个 ArmInstance("right", 1)
  ├─ BOTH  → 动态创建 2 个 ArmInstance
  └─ NONE  → 退出 (无手部数据)
```

#### 关键修改

1. **新增 `build_arms_from_detection()` 函数**:
   - 接收 `Handedness` 枚举值
   - 返回 0/1/2 个 `ArmInstance` 列表
   - 不再有任何手部索引硬编码

2. **`ArmInstance` 类重写**:
   - 接收 `prefix` (left/right) + `hand_idx` 作为参数
   - 内部根据 `prefix` 自动选择 URDF/IK 方法/初始 qpos
   - 与 hand_track/SingleArmConfig 兼容但解耦

3. **CLI 完全保留 `--hawor-dir`**:
   - 不再有任何手部指定参数
   - 所有手部信息从 reconstruction 文件读取

4. **修复**:
   - `robot_dir` 路径错误 (`/assets/robots/hands` → `/dex-retargeting/assets/robots/hands`)
   - 抑制 5 类无关警告
   - 添加启动横幅

#### 验证 (4 个 hawor 目录)

| 目录 | 检测 | 创建 | qpos | 状态 |
|---|---|---|---|---|
| `7`             | 左手 113/113 | `['left']`        | `left=(5, 8)`      | ✓ |
| `7_vggt-omega`  | 左手 113/113 | `['left']`        | `left=(5, 8)`      | ✓ |
| `laptop`        | 右手 578/600 | `['right']`       | `right=(200, 8)`   | ✓ |
| `hoi4d`         | 双手 599/600 | `['left', 'right']` | `left=(5, 8), right=(5, 8)` | ✓ |

**关键证据 — 真正泛化**:
- `7` 目录只有左手 → 脚本**只创建 1 个左臂**, 没有右臂
- `laptop` 目录只有右手 → 脚本**只创建 1 个右臂**, 没有左臂
- `hoi4d` 目录有双手 → 脚本创建 2 个臂, **同时运动**

---

### 2026-06-10 修改（第十一轮）：双臂协同运动渲染 04_render_dual_arm.py

**类型**: 新功能
**影响范围**: 04_render_dual_arm.py (新增)

#### 背景

用户需求: 实现与 02_render_scene.py 同类的双臂协同运动映射, 两个机械臂必须**同时运动**,
且专注于**运动映射实现**而非整个机器人系统。

#### 架构

- **单臂封装**: `SingleArm` 类 — 每个臂独立 URDF + Retargeting + IK 链
- **双臂协调**: 在同一个 SAPIEN 场景中, 双手 MANO FK → 双手 retarget → 双手 IK → 同步 `set_qpos` → `scene.step()`
- **共享元素**: scene, camera, GLB/ground, internal_scene
- **独立元素**: URDF, robot, retargeting, mano_layer, joint_filter, ik_method

#### 关键功能

1. **`SingleArm` 类** (1.7KB 代码):
   - 解析臂关节 (6个) + 夹爪关节 (2个) 索引
   - 初始化 Retargeting, 约束点 `[4, 8, 0]` (食指尖/中指尖/手腕)
   - `retarget_and_solve_ik()` 一步完成 retarget + IK + 滤波
   - `compute_mano()` MANO FK → SAPIEN 坐标系
   - `warm_start()` 首帧预热

2. **`run_dual_arm_only()`** 主函数:
   - 7 步流程: 加载数据 → 场景 → 双臂 → 预热 → 基座 → 相机 → 渲染
   - 双手腕质心分别计算, 独立放置左右臂基座
   - 双手同步 IK, 同一 `scene.step()` 调用

3. **`simple_ground()`** 默认场景:
   - 4x4m 地面网格, 灰色
   - 不需 GLB / transform_params
   - 配合 `render_coordinate_axes()` 显示 XYZ 坐标系

4. **关键点显示** (用户要求):
   - 左手 3 个 retargeting 关键点: 粉色球
   - 右手 3 个 retargeting 关键点: 蓝色球
   - 半径 0.006m, 跟随手腕实时更新

5. **GLB 可选加载** (用户要求):
   - 指定 `--ras-dir` + `--transform-params` → 加载 GLB
   - 不指定 → simple_ground + 坐标系

6. **qpos 输出格式**:
   ```python
   np.save(qpos_path, {
       "left":  np.array([q["left"]  for q in qpos_log]),  # (N, 8)
       "right": np.array([q["right"] for q in qpos_log]),  # (N, 8)
   })
   ```
   每臂 8 维 = 6 臂关节 + 2 夹爪

#### CLI

```bash
# 默认 (simple_ground + 坐标系)
python 04_render_dual_arm.py --hawor-dir /home/an/data/hawor/hoi4d

# 加载 GLB
python 04_render_dual_arm.py --hawor-dir /home/an/data/hawor/hoi4d \
    --ras-dir /path/to/ras \
    --transform-params ./output/alignment/transform_params.npz

# 视角: behind/front/topdown/side
python 04_render_dual_arm.py --hawor-dir /home/an/data/hawor/hoi4d \
    --view behind --num-frames 100
```

#### 验证

- `python -c "import ast; ast.parse(...)"` 语法通过
- `conda run -n dex python 04_render_dual_arm.py --hawor-dir /home/an/data/hawor/hoi4d --num-frames 20`
  - ✓ 双手 20 帧同时渲染
  - ✓ 左 qpos 形状 (20, 8), 右 qpos 形状 (20, 8)
  - ✓ 左 qpos[0][0]=1.29, 右 qpos[0][0]=-1.49 (左右对称初始位置)
  - ✓ mp4 视频 29KB, npy 3KB
  - ✓ 总耗时 ~5s

#### 与 02_render_scene.py 的区别

| 维度 | 02_render_scene.py | 04_render_dual_arm.py |
|---|---|---|
| 臂数 | 1 (单臂 robot_only/robot_tracking/hand_only) | **2 (双臂)** |
| 模式 | 3 种 (hand_only/robot_only/robot_tracking) | **1 种 (robot_only + 关键点)** |
| MANO 手渲染 | 可选 (hand_only / robot_tracking) | **不渲染 (只显示 3 个关键点)** |
| GLB | 必需 | **可选** |
| 场景 | GLB 为主 | **GLB 或 simple_ground** |
| 输出 | 单一 qpos.npy | **左右分开的 dict** |

---

### 2026-06-09 修改（第十轮）：抑制无关警告, 输出清洁化

**类型**: 用户体验
**影响范围**: hand_track/run_all_hawor.py

#### 背景

用户反馈输出被大量无关警告污染:
- `pkg_resources is deprecated` (来自 sapien)
- `np.bool/np.object/np.str will be defined as the corresponding NumPy scalar` (来自 mano_layer.py)
- `SAPIEN warning: loading multiple convex collision meshes from STL file` (C++ 端, 重复数十次)
- `PyTorch CUDA initialization: The NVIDIA driver on your system is too old`
- PyTorch `not writable tensor` 警告

#### 修改

- [run_all_hawor.py] 顶部添加 `import warnings` 和 6 条 `warnings.filterwarnings()` 规则
- [run_all_hawor.py] 设置 `os.environ["SAPIEN_LOG_LEVEL"] = "ERROR"`
- [run_all_hawor.py] 在 `import sapien` 后调用 `sapien.set_log_level("error")` 抑制 C++ 端日志
- [README.md] FAQ 新增 Q1.5 说明如何恢复警告 (`PYTHONWARNINGS=default`)

#### 验证

- `conda run -n dex python hand_track/run_all_hawor.py --hawor-dirs /home/an/data/hawor/7 --no-render --num-frames 30`
  - **修改前**: 输出 30+ 行警告
  - **修改后**: 输出 0 行警告, 全部是脚本自己的 [INFO] 日志

---

### 2026-06-09 修改（第九轮）：hand_track/README.md 文档

**类型**: 文档
**影响范围**: hand_track/README.md (新增)

#### 内容

- **模块组成**: 列出 5 个核心文件 + `__init__.py` + `output/` 的作用
- **数据源约定**: 明确说明只需要 `reconstruction/hawor_results_*.npz`
- **数据流架构图**: ASCII 框图展示从 HaWoR 数据到 qpos.mp4 的完整管线
- **快速开始**: 7 种调用示例 (单目录/多目录/全扫描/--no-render/前 N 帧/测试)
- **CLI 参数表**: `--hawor-base`, `--hawor-dirs`, `--output-base`, `--fps`, `--view`, `--no-render`, `--start-frame`, `--num-frames`
- **输出结构**: 树形图展示 `output/<dir_name>/` 下的 qpos.npy 和 mp4
- **qpos 格式说明**: (有效帧数, 14) 形状, 14 = 6 臂关节 + 2 夹爪 + 6 浮动基座
- **00_run_pipeline.py 集成方式**: `--use-auto` 透传到 `render_auto.py`
- **已测试 HaWoR 目录**: 7 / 7_vggt-omega / hoi4d / laptop 四个目录
- **常见问题 FAQ**: 4 个常见问题 + 解决方案
- **开发指南**: 单元测试覆盖、添加新 hawor 目录、调试新数据

#### 验证

- `python hand_track/test_pipeline.py` 35 个测试全部通过

---

### 2026-05-31 修改（第六轮）：映射修复 & 夹爪校正

#### 3.1 修复 `warm_start` 中 `wrist_quat` 坐标系不匹配

**问题**: `warm_start` 传入的 `wrist_quat` 来自 `pred_rot`（Render World Y-UP），但 `joints_sapien[0, :3]` 已经在 SAPIEN (Z-UP) 坐标系中。位置和旋转不在同一坐标系，导致优化器初始化方向错误，这是"关节反转"的根本原因。

**修改**: 将 `wrist_quat` 也转到 SAPIEN 坐标系：
```python
wrist_R_render = pr.matrix_from_compact_axis_angle(pred_rot)
wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T
wrist_quat = pr.quaternion_from_matrix(wrist_R_sapien)
```

#### 3.2 修复夹爪初始化值超出关节限位

**问题**: `init_qpos[gripper_idx2] = -0.04`，但 r1_v2_1_0.urdf 中 joint2 的限位是 `[0, 0.05]`，-0.04 超出限位。

**修改**: 改为 `init_qpos[gripper_idx2] = 0.04`，与 joint1 一致。两处 warm_start 和三处初始化都已修复。

#### 3.3 确认容差和约束点映射与参考文件一致

对比 `r1_hand_tracking_video.py`，确认以下配置已一致：
- `IK_TOLERANCES = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]` ✓
- `target_link_human_indices = np.array([4, 8, 0])` ✓
- `target_link_names = ["right_gripper_finger_link1", "right_gripper_finger_link2", "right_gripper_link"]` ✓
- 映射关系: 拇指尖(4)→finger_link1, 食指尖(8)→finger_link2, 手腕(0)→gripper_link ✓

---

### 2026-05-31 修改（第五轮）：鲁棒性修复 & 代码清理

#### 2.1 修复 `wrist_pos_sapien` 未定义 bug

**问题**: `run_robot_tracking` 渲染循环中引用 `wrist_pos_sapien` 计算末端误差，但该变量从未定义，导致 `NameError`。

**修改**: 初始化 `wrist_pos_sapien = None`，在每帧手部数据有效时更新 `wrist_pos_sapien = joints_sapien[0, :3].copy()`。

#### 2.2 修复 `obj_nodes` 变量名不一致

**问题**: `run_robot_tracking` 和 `run_robot_only` 中 GLB 加载失败时创建了 `obj_nodes = []`，但实际变量名是 `obj_actors`，`obj_nodes` 从未使用。

**修改**: 移除多余的 `obj_nodes = []`。

#### 2.3 修复 `run_robot_only` 预计算循环未更新 base_link 位姿

**问题**: `run_robot_only` 的预计算循环中，底座跟踪后未更新 `base_link_p`/`base_link_q`/`base_link_R`/`base_link_R_inv`，导致后续帧的 IK 目标使用过时的 base_link 位姿。`run_robot_tracking` 中有此更新但 `run_robot_only` 遗漏了。

**修改**: 在预计算循环中添加底座跟踪后的 base_link 位姿更新逻辑。

#### 2.4 移除未使用的代码

- 移除 `saved_qpos` 变量（2处，从未使用）
- 移除 `axis_nodes` 变量（5处，位姿标识已删除后不再使用）
- 移除 `_render_camera_axes()` 和 `_render_pose_axes()` 方法（不再被调用）

#### 2.5 `01_align_scene.py` 增加 `--force_scale` 参数和验证

**新增**: `--force_scale` 参数，允许强制指定尺度因子覆盖自动计算值。

**新增验证**:
- R_align 行列式检查（应为 1.0）
- R_align 正交性检查（R^T R ≈ I）
- s_inv 范围检查（0.01~10.0 之外发出警告）
- s_inv ≤ 0 时抛出异常

#### 2.7 静态相机自动检测和回退

**问题**: Umeyama 通过相机轨迹离散度计算尺度比，静态相机时 sigma→0 导致尺度不稳定。用户不应手动判断。

**修改**: 自动检测静态相机 (sigma < 0.01)，回退到基于手-GLB 距离的启发式估算：
1. 计算 GLB 质心在 HaWoR 坐标系中的位置（未缩放）
2. 计算手部平均位置到 GLB 质心的距离
3. 假设手-物距离约 0.15m，反推 s_inv

#### 2.6 文档润色

- `Combination_pipeline.md`: 更新对齐原理描述（第一帧相机锚定而非旧 Umeyama 方式），移除已删除功能的描述（相机柱子、位姿标识），增加机器人映射链和底座跟踪说明，精简文件说明表格。

---

### 2026-05-29 修改（第四轮）：映射方法重构 & 视觉清理

#### 2.1 映射方法重构 — 采用 r1_hand_tracking_video.py 的3约束点方案

**问题**: 旧方法使用2约束点[4,8] + `_compute_ee_orientation_from_wrist()`直接计算朝向，夹爪开合和姿态映射都不准确。

**旧方法 vs 新方法对比**:

| 对比项 | 旧方法 | 新方法 (参考r1_hand_tracking_video.py) |
|--------|--------|--------------------------------------|
| 约束点 | 2个 [4,8] (食指尖+中指尖) | **3个 [4,8,0] (食指尖+中指尖+手腕)** |
| 朝向来源 | `_compute_ee_orientation_from_wrist()` 直接从手腕旋转计算 | **FK提取** `_get_gripper_pose_from_retargeting()` |
| IK目标位置 | 手腕位置 `wrist_pos + offset` | **FK夹爪位置** `gripper_pos_fk + offset` |
| normal_delta | 默认 4e-3 (锁死朝向) | **1e-5** (朝向可自由变化) |
| huber_delta | 默认 | **0.01** |
| IK容差 | [0.001,0.001,0.001,0.002,0.002,0.002] | **[0.1,0.1,0.1,0.1,0.1,0.1]** |
| warm_start | 无 | **有** (用第一帧手腕四元数初始化) |

**3约束点为什么能约束朝向**:
- 2约束点: 6约束 vs 7DOF → 欠定1DOF → 接近轴旋转无梯度
- 3约束点: 9约束 vs 7DOF → 超定2约束 → 手腕位置相对指尖方向编码朝向，优化器自然产生朝向梯度

**IK容差为什么改为0.1**:
- 容差是RelaxedIK目标函数的权重分母: `L = Σ(err_i / tol_i)²`
- 旧值0.001: 位置权重=10⁶, 朝向权重=2.5×10⁵ → 位置主导，朝向被拉偏
- 新值0.1: 位置权重=100, 朝向权重=100 → 位置和朝向平衡

#### 2.2 去除蓝色相机柱子

**问题**: 视频中相机位置有蓝色柱状标记，影响观感。

**修改**: 移除所有 `_render_camera_axes()` 调用（4处），相机位置更新保留但不再渲染标记。

#### 2.3 去除位姿标识

**问题**: 视频中末端执行器（手腕/夹爪）位姿处有坐标轴标识，影响观感且与最终展示无关。

**修改**: 移除所有位姿坐标轴渲染调用，位姿计算保留但不再渲染标识。

#### 2.4 Output文件夹说明

example目录下的output文件夹是测试输出，包含：

| 文件夹 | 内容 | 建议 |
|--------|------|------|
| `combination/output/` | 对齐报告、变换参数、视频、npy | **保留** — 管线正式输出 |
| `combination/test/` | 调试脚本、旧对齐视频 | 可删除 |
| `position_retargeting/pv_retargeting/` | r1跟踪视频和日志 | **保留** — 参考实现输出 |
| `position_retargeting/test/` | IK测试视频和日志 | 可删除 |
| `simulation/` | 仿真测试视频和日志 | 可删除 |
| `video_egocentric_retargeting/output_*/` | 第一人称管线输出 | 可删除 |
| `output_pin_base_frame/` (项目根) | R1跟踪视频和日志 | 可删除 |

---

### 2026-05-29 修改（第三轮）：IK收敛 & 底座跟踪 & 关键点精简

#### 2.1 RelaxedIK 姿态收敛改善

**问题**: RelaxedIK每次调用只做一步梯度下降，5次调用不足以收敛到目标姿态，导致末端姿态误差大。

**根因分析**:
1. RelaxedIK是增量式求解器，每次`solve_position`只做一步优化
2. 位置梯度量级大于姿态梯度，位置先收敛但姿态还差很远
3. 求解器内部状态未同步——每次求解后不reset，下一帧优化起点偏移
4. R1 Lite只有6个关节（非冗余臂），姿态可达空间受限

**修改**:
- 求解次数从5增加到20: `IK_SOLVE_PER_FRAME = 20`
- 旋转容差从0.005收紧到0.002: `IK_TOLERANCES = [0.001, 0.001, 0.001, 0.002, 0.002, 0.002]`
- 每帧求解后reset求解器内部状态: `ik_solver.relaxed_ik_right.reset(list(arm_joints))`

#### 2.2 关键点显示精简

**问题**: 视频中显示了所有21个MANO关节点，视觉混乱，与夹爪映射无关。

**修改**:
- `_render_keypoints()` 只显示`ref_indices`中的关节（4=食指尖, 8=中指尖）
- 移除其他关节的渲染，移除`FINGER_GROUP_COLORS`在该函数中的使用

#### 2.3 机械臂底座小范围跟踪

**问题**: 底座固定在手腕轨迹质心上方，当手腕远离质心时IK不稳定。

**修改**:
- 新增`_compute_tracking_base_pos()`方法，根据当前手腕位置计算底座偏移
- 偏移量在base_link坐标系中计算，限制在±`BASE_TRACKING_RANGE`(0.08m)范围内
- 三个`run_*`方法中每帧更新底座位置并重新获取base_link位姿
- 新增常量: `BASE_TRACKING_RANGE = 0.08`, `BASE_TRACKING_ALPHA = 0.15`

---

### 2026-05-29 修改（第二轮）：机械臂映射精度 & 相机视角修正

#### 2.1 GLB 渲染方式改为 SAPIEN 公开 API

| 修改项    | 修改前                                                   | 修改后                                                       |
| ------ | ----------------------------------------------------- | --------------------------------------------------------- |
| 渲染 API | `context.create_mesh_from_array()` + `internal_scene` | `scene.create_actor_builder()` + `add_visual_from_file()` |
| 加载方式   | 内存中构建 mesh 数组                                         | 导出变换后 PLY → SAPIEN 加载 PLY                                 |
| 颜色设置   | `context.create_material()`                           | `sapien.render.RenderMaterial(base_color=...)`            |
| 简化     | `MAX_FACES_PER_OBJECT=0`（不简化）                         | 不简化，SAPIEN 公开 API 自动处理大 mesh                              |
| 渲染效果   | 点云状（内部 API 处理大 mesh 异常）                               | 实物质感（公开 API 正确渲染）                                         |

**参照文件**：`/home/an/robot_world_ws/src/ReplicateAnyScene/View/pipeline_universal.py`

#### 2.2 机械臂末端朝向综合

| 修改项   | 修改前                                    | 修改后                                                        |
| ----- | -------------------------------------- | ---------------------------------------------------------- |
| 朝向来源  | 仅 `wrist_R_sapien`                     | `_compute_combined_ee_orientation()`：手腕旋转(40%) + 手部几何(60%) |
| IK 容差 | `[0.001, 0.001, 0.001, 0.1, 0.1, 0.1]` | `[0.001, 0.001, 0.001, 0.03, 0.03, 0.03]`                  |
| 位置映射  | `mapping_offset=zeros`                 | `mapping_offset=zeros, safety_offset=zeros`（手部实际位置）        |

**综合朝向算法**：

1. 计算手部几何朝向 R\_hand（拇指→食指 + 手腕→MCP）
2. 计算 MANO 手腕朝向 R\_wrist
3. 逐轴加权融合：`R_combined[:, col] = 0.4 * R_wrist[:, col] + 0.6 * R_hand[:, col]`
4. SVD 正交化保证 R\_combined 是合法旋转矩阵
5. 如果 R\_hand 计算失败，回退到 R\_wrist

#### 2.3 GPU 渲染

| 修改项        | 修改前                   | 修改后                                |
| ---------- | --------------------- | ---------------------------------- |
| Vulkan ICD | 硬编码 `nvidia_icd.json` | 自动检测：nvidia-smi 可用→NVIDIA，否则→Intel |

#### 2.4 `_extract_wrist_pose` 状态

`_extract_wrist_pose` 方法**未被使用**。当前末端朝向由 `_compute_combined_ee_orientation()` 计算，位置由手腕关节 `joints_sapien[0, :3]` 直接获取。

#### 2.5 `03_track_robot.py` 状态

`03_track_robot.py` **仍然有效**，是独立机器人跟踪脚本（不需要 RAS GLB）。与 `02_render_scene.py --mode robot_only` 的区别：

- 不需要 RAS GLB 场景
- 不需要第一人称相机轨迹
- 支持多种第三人称视角
- 适合快速验证手部→机器人映射

### 2026-05-28 修改（第一轮）

#### GLB 像点云的根因

`fast_simplification` 未安装 → `simplify_quadric_decimation` 抛异常 → fallback 的 `faces[::step]` 把连续三角面拆成孤立碎片 → 看起来像点云。

修复：安装 `fast_simplification`，最终改为 SAPIEN 公开 API 加载。

***

## 3. 文件清单

| 文件                          | 功能        | 关键修改                                   |
| --------------------------- | --------- | -------------------------------------- |
| `01_align_scene.py`         | 计算对齐参数    | 第一帧相机位姿锚定 + ZUP→YUP + OPENCV→OPENGL    |
| `02_render_scene.py`        | 渲染仿真场景    | SAPIEN 公开 API 加载 GLB + GPU 自动检测 + 综合朝向 |
| `03_track_robot.py`         | 独立机器人跟踪   | 无修改，仍然有效                               |
| `hand_object.py`            | 独立渲染脚本    | IK 容差修改                                |
| `yingshe.py`                | 旧对齐脚本（参考） | Umeyama 方式，RAS Y-UP 假设错误               |
| `r1_hand_tracking_video.py` | 参考实现      | DexYCB 管线，综合朝向算法来源                     |
| `pipeline_universal.py`     | GLB 可视化参考 | SAPIEN 公开 API 加载 GLB 的参考实现             |
| `test/analyze_distance.py`  | 距离分析脚本    | 手-GLB 逐帧距离分析                           |

***

## 4. 坐标系变换速查

```
GLB顶点 (RAS Y-UP)
  → s_inv * R_inv @ p + t_inv
HaWoR render world (Y-UP)
  → RXWORLD_TO_SAPIEN @ p
SAPIEN world (Z-UP)

HaWoR 手部顶点 (render world Y-UP)
  → RXWORLD_TO_SAPIEN @ p
SAPIEN world (Z-UP)

HaWoR 相机 (R_c2w, t_c2w, render world Y-UP)
  → hawor_cam_to_sapien_pose()
SAPIEN 相机

MANO 手腕旋转 (render world Y-UP)
  → RXWORLD_TO_SAPIEN @ wrist_R_render
SAPIEN world (Z-UP)

机械臂末端朝向 (SAPIEN world Z-UP)
  → _compute_combined_ee_orientation(joints_sapien, wrist_R_sapien)
  → 40% wrist_R_sapien + 60% hand_geometry_R
  → SVD 正交化
```

***

## 5. 手-GLB 距离详细分析

### 5.1 GLB 在 SAPIEN 坐标系

| 属性 | 值                                                   |
| -- | --------------------------------------------------- |
| 中心 | \[-0.006, -0.238, 0.053]                            |
| 边界 | \[-0.074, -0.335, 0.010] \~ \[0.132, -0.140, 0.083] |
| 尺寸 | 0.205 × 0.194 × 0.073 m                             |

### 5.2 手部在 SAPIEN 坐标系

| 属性   | 值                       |
| ---- | ----------------------- |
| 手腕中心 | \[0.176, -0.283, 0.008] |
| 指尖中心 | \[0.069, -0.249, 0.103] |

### 5.3 距离统计

| 指标         | min     | mean    | max     |
| ---------- | ------- | ------- | ------- |
| 手腕→GLB最近顶点 | 0.0004m | 0.0824m | 0.1507m |
| 指尖→GLB最近顶点 | 0.0006m | 0.0623m | 0.1858m |

### 5.4 逐帧手腕→GLB最近距离（前20帧）

| 帧  | 距离(m)      | 说明       |
| -- | ---------- | -------- |
| 0  | 0.0423     | <br />   |
| 1  | 0.0363     | <br />   |
| 2  | 0.0350     | <br />   |
| 3  | 0.0268     | <br />   |
| 4  | 0.0187     | <br />   |
| 5  | **0.0004** | 手几乎接触GLB |
| 6  | 0.0124     | <br />   |
| 7  | 0.0211     | <br />   |
| 8  | 0.0290     | <br />   |
| 9  | 0.0363     | <br />   |
| 10 | 0.0451     | <br />   |
| 15 | 0.0658     | <br />   |
| 19 | 0.0843     | 手远离GLB   |

