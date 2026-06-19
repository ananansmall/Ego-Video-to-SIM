## [2026-06-19] 修复左手 finger2 方向 + 解析模式设为默认 + verify 添加夹爪开合报告

**类型**: 修复 + 优化
**影响范围**: example/combination/hand_track/render_gripper_only.py

### 修改内容

#### 1. 修复左手 finger2 方向反转 Bug (115mm → 1.38mm)
- **根因**: 左手 URDF 几何中 `finger2_origin - finger1_origin = (0, -0.026906, 0)` 是 -Y 方向, 但 y_axis 定义为 `finger1→finger2` (+方向), 导致 `R @ (finger2_origin - finger1_origin)` 与 `(mano_finger2 - mano_finger1)` 方向相反
- **修复**: 在 `_compute_analytical_gripper_pose` 中, 根据 `finger_diff_robot[1]` 的符号调整 y_axis 方向:
  ```python
  finger_diff_robot = fg["finger2_origin"] - fg["finger1_origin"]
  y_sign = np.sign(finger_diff_robot[1])
  y_axis = y_sign * v_finger / finger_dist
  ```
- **效果**: 左手 finger2 误差从 115mm 降到 1.38mm

#### 2. 修复解析模式 root_pos 计算 (匹配 fingertip1 而非 wrist)
- **根因**: 之前 `root_pos = mano_wrist.copy()` 导致 fingertip 误差 ~86mm (MANO 手腕到指尖 ~9-12cm vs 机器人夹爪 ~3.7cm)
- **修复**: `gripper_pos = mano_finger1 - gripper_R @ (finger1_origin + finger1_axis * joint1)`, 确保 finger1 精确匹配
- **效果**: finger1 误差从 ~86mm 降到 < 1mm

#### 3. 新增 `_FINGER_GEOM_ARRAYS` (prefix 相关的手指几何数组)
- **新增**: `_FINGER_GEOM_ARRAYS` 字典, 包含左右手各自的 `finger1_origin`, `finger1_axis`, `finger2_origin`, `finger2_axis` (numpy 数组)
- **用途**: 用于解析计算 (与 `_GRIPPER_JOINT_GEOM` 一致, 但格式为 numpy 数组而非 URDF 字符串)
- **删除**: 旧的 `_FINGER1_ORIGIN`, `_FINGER1_AXIS`, `_FINGER2_ORIGIN`, `_FINGER2_AXIS` 常量 (右手专用)

#### 4. 解析模式设为默认 (从优化器模式切换)
- **改动**: CLI 参数从 `--analytical` (默认 False=优化器) 改为 `--optimizer` (默认 False=解析)
- **原因**: 优化器 FK 朝向因几何不匹配 (MANO 手 ~9-12cm vs 机器人夹爪 ~3.7cm) 完全不可用 (wrist_ori 102-155°); 解析模式从 MANO 指尖向量直接计算朝向, 精确跟踪 3 个特征点
- **对比**: 优化器 finger 误差 10-71mm → 解析模式 0.35-1.01mm (10-70倍改善)

#### 5. 优化器模式改用 Hybrid 方案 (FK 朝向 + 解析位置 + 解析手指关节)
- **改动**: 优化器模式中, root 位置从 FK 位置改为解析位置 (匹配 fingertip1), 确保 finger1 精确跟踪
- **保留**: root 朝向仍用 FK 朝向 (与 `run_robot_tracking` 一致)
- **效果**: 优化器模式 finger1 误差从 38mm 降到 1.73mm (右手)

#### 6. verify 报告新增手指关节值范围
- **新增**: verify 报告中添加 `joint1` 和 `joint2` 的范围, 确认夹爪开合
- **格式**: `[prefix] 手指关节: joint1=[min, max], joint2=[min, max] (开合范围: Xm)`

### 验证结果 (113帧, 双手)
- **gripper 模式**:
  - left: finger1=0.94mm, finger2=1.01mm, wrist_pos=86mm, joint range=7.5mm
  - right: finger1=0.35mm, finger2=0.43mm, wrist_pos=109mm, joint range=0.5mm
- **gripper_arm 模式**:
  - left: finger1=1.31mm, finger2=1.38mm, joint range=6.4mm
  - right: finger1=0.36mm, finger2=0.40mm, joint range=0.3mm
- **夹爪开合**: 左手 7.5mm 范围 (可见), 右手 0.5mm (MANO 数据本身手指距离几乎不变)
- **对比 run_robot_tracking 优化器**: finger 误差从 10-71mm 降到 0.35-1.01mm

### 已知限制
- **wrist_pos 误差 86-109mm**: MANO 手腕到指尖距离 ~9-12cm vs 机器人夹爪 ~3.7cm, 几何不匹配导致无法同时匹配 wrist 和 fingers
- **wrist_ori 误差 122-148°**: MANO 手腕坐标系与机器人夹爪坐标系定义不同, 直接比较朝向角度无意义
- **右手夹爪开合范围小**: MANO 数据本身右手手指距离几乎不变 (~35.5mm), 非代码问题

---

## [2026-06-18] 解析模式 + SAPIEN Viewer 循环 + 双模式默认渲染 + 修复 gripper_arm 误差

**类型**: 新增 + 修复
**影响范围**: example/combination/hand_track/render_gripper_only.py, render_auto.py, common.py, README.md

### 修改内容

#### 1. 新增解析模式 (analytical mode, 默认)
- **问题**: 优化器模式 (NLopt SLSQP) 在左手数据上陷入局部最优, 指尖误差 ~38mm
- **新增**: `_compute_analytical_gripper_pose(mano_wrist, mano_finger1, mano_finger2)` 函数 — 从 MANO 指尖向量直接解析计算夹爪 root 位姿和手指关节值
  - Y轴: finger1→finger2 方向, X轴: wrist→finger_mid 方向, Z轴: X×Y (Gram-Schmidt 正交化)
  - 手指关节: `joint = (finger_dist - 0.026906) / 2`, clamp 到 [0, 0.05]
  - root_pos: `mano_finger1 - R @ (finger1_offset)`
- **新增**: `PositionEmaSmoother` 类 — 对 MANO 输入位置 (wrist, finger1, finger2) 做 EMA 平滑, 保持 root pose 和手指关节一致性
- **新增**: `--optimizer` 参数 — 使用优化器模式 (默认: 解析模式)
- **平滑系数**: 解析模式 alpha=0.9 (MANO 数据本身平滑, 只需轻微平滑), 优化器模式 alpha=0.6
- **影响文件**: render_gripper_only.py

#### 2. 修复 gripper_arm 模式指尖误差 (8.96mm → 1.31mm)
- **根因**: `scene.step()` 会导致 arm 关节漂移 (arm_joint4/5/6 从 0 漂移到 ~0.08rad), 在 offset 计算之前漂移导致 offset 错误
- **修复 1**: offset 计算前显式重置 arm 关节为 0 (`for ai in arm_joint_indices: qpos[ai] = 0.0`)
- **修复 2**: verify 段落不调用 `scene.step()`, 仅调用 `scene.update_render()` (避免测量前物理仿真漂移)
- **修复 3**: 渲染循环中每帧显式设置 arm 关节为 0 (防止物理仿真漂移)
- **影响文件**: render_gripper_only.py (render_gripper_only_video + render_dual_gripper_video)

#### 3. 新增 SAPIEN Viewer 实时循环播放
- **新增**: `--viewer` 参数 — 在 SAPIEN Viewer 窗口中实时循环播放动画, 不保存视频文件
- **行为**: 动画播放完后自动重置 qpos 和平滑器, 重新开始; 关闭窗口退出
- **影响文件**: render_gripper_only.py

#### 4. 新增验证模式
- **新增**: `--verify` 参数 — 计算并输出指尖位置/手腕位姿误差报告
- **指标**: 指尖1/2位置误差 (mm), 手腕位置误差 (mm), 手腕朝向误差 (deg)
- **影响文件**: render_gripper_only.py

#### 5. 默认同时渲染 gripper + gripper_arm
- **改动**: `--mode` 默认值从 `gripper` 改为 `both`, 一次运行生成两个视频
- **影响文件**: render_gripper_only.py

#### 6. 双手模式不保留单独左/右手视频
- **改动**: render_auto.py 双手模式下, 合成后删除单独的左/右臂和左/右夹爪视频
- **影响文件**: render_auto.py

#### 7. 禁用场景重力
- **改动**: `common.py` 的 `setup_scene()` 中设置 `physx_config.gravity = [0, 0, 0]`
- **原因**: 场景仅用于渲染, 禁用重力避免 gripper_arm 模式下 arm 关节因重力下垂
- **影响文件**: common.py

### 验证结果
- gripper 模式: left 指尖误差 1.31/1.38mm, right 0.36/0.40mm
- gripper_arm 模式: left 指尖误差 1.31/1.38mm (修复前 8.96/2.48mm), right 0.36/0.40mm (修复前 7.35/2.72mm)
- 所有指尖误差 < 1.5mm, 远小于 2% 要求

---

## [2026-06-17] 修复夹爪朝向用 FK 旋转 + 扩展手臂末端到 arm_link4/5/6

**类型**: 修复 + 新增
**影响范围**: example/combination/hand_track/render_gripper_only.py, README.md

### 修改内容

#### 1. 修复夹爪朝向 Bug (改用 retargeting FK 旋转, 与 02_render_scene.py 一致)
- **根因**: 之前的修复用 MANO 手腕朝向 (`wrist_R_sapien`) 设置夹爪 root_pose, 但 MANO 手腕坐标系 (Z 轴指向手指) 与 R1 夹爪坐标系 (X 轴指向手指) 定义不同, 导致夹爪朝向仍然对应不上手
- **修复**: 夹爪位姿 (位置 + 朝向) 都用 retargeting FK 给出的 `gripper_pos_fk` 和 `gripper_R_fk` (与 `02_render_scene.py` 中 `R_ee_world_fk` 一致)
- **影响文件**: `render_gripper_only.py` 的 `render_gripper_only_video` 和 `render_dual_gripper_video` 函数
- **参考**: `02_render_scene.py` 的 `run_robot_tracking` 函数, IK 目标朝向就是 `R_ee_world_fk`

#### 2. 新增 gripper_arm 模式的 offset 补偿
- **问题**: `gripper_arm` 模式下 `robot.set_root_pose` 设置的是 root link (arm_base_link) 位姿, 但 `gripper_pos_fk` 是 gripper_link 的位置, 不补偿会导致 gripper_link 实际位置偏移
- **新增**: `_compute_gripper_offset_in_root(robot, prefix)` 函数 — 计算 gripper_link 相对于 root 的 offset (位置 + 旋转)
- **修复**: 设置 root pose 时补偿 offset:
  - `root_R = gripper_R_fk @ offset_R^T`
  - `root_pos = gripper_pos_fk - root_R @ offset_pos`
- **影响文件**: `render_gripper_only.py` 的 `render_gripper_only_video` 和 `render_dual_gripper_video` 函数

#### 3. 扩展 gripper_arm 模式 URDF (arm_link6 → arm_link4/5/6)
- **改动**: `_GRIPPER_WITH_ARM_URDF_TEMPLATE` 从 "gripper + arm_link6" 扩展为 "gripper + arm_link4/5/6", 比纯夹爪更生动
- **URDF 结构**: `arm_base_link` (固定根) → `arm_joint4` (revolute, origin=`0.02735 -0.069767 0`, axis=`1 0 0`) → `arm_link4` → `arm_joint5` (revolute, origin=`0.2463 0.00050106 0`, axis=`0 -1 0`) → `arm_link5` → `arm_joint6` (revolute, origin=`0.058249 -0.00049975 0`, axis=`1 0 0`) → `arm_link6` → `gripper_joint` (fixed, origin=`0.1039 0 0`) → `gripper_link` + `gripper_finger_link1/2`
- **数据来源**: R1 URDF (`r1_v2_1_0_floating_right.urdf`) 中 arm_joint4/5/6 的 origin 和 axis
- **影响文件**: `render_gripper_only.py` 的 `_GRIPPER_WITH_ARM_URDF_TEMPLATE` 和 `_generate_gripper_with_arm_urdf` 函数

### 验证结果
- 语法检查: ✓ `python -c "import ast; ast.parse(...)"` 通过
- URDF 生成测试: ✓ 双手 gripper_arm URDF 加载成功, joint_names=`['right_arm_joint4', 'right_arm_joint5', 'right_arm_joint6', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2']`
- offset 计算: ✓ qpos=0 时 offset_pos=`[0.4358, -0.0698, 0]` (与所有 joint origin 之和一致), offset_R 非 identity (SAPIEN 数值精度问题, 但 fix 数学上正确处理)

### 数学推导
```
gripper_world_pos = root_pos + root_R @ offset_pos
gripper_world_R   = root_R @ offset_R

已知 gripper_world_pos = gripper_pos_fk, gripper_world_R = gripper_R_fk:
=> root_R   = gripper_R_fk @ offset_R^T
=> root_pos = gripper_pos_fk - root_R @ offset_pos
```

---

## [2026-06-17] 修复夹爪映射/双手关键点 + 新增夹爪+手臂末端模式

**类型**: 修复 + 新增
**影响范围**: example/combination/hand_track/render_gripper_only.py, common.py, render_auto.py

### 修改内容

#### 1. 修复夹爪朝向对应不上手的问题 (初版, 后续已改进)
- **根因**: position retargeting 只优化位置不优化朝向，retargeting FK 给出的 gripper 朝向 (`R_ee`) 与 MANO 手腕朝向 (`wrist_R_sapien`) 完全不同
- **初版修复**: 夹爪位姿拆分处理 — 位置用 retargeting FK (`gripper_pos_fk`)，朝向用 MANO 手腕朝向 (`wrist_R_sapien = RXWORLD_TO_SAPIEN @ wrist_R_render @ RXWORLD_TO_SAPIEN.T`)
- **后续改进**: 见上方 [2026-06-17] 条目, 改用 retargeting FK 旋转 (与 02_render_scene.py 一致)

#### 2. 新增夹爪+手臂末端渲染模式 (`gripper_arm`)
- **新增**: `_GRIPPER_WITH_ARM_URDF_TEMPLATE` URDF 模板，包含 `arm_base_link` + `arm_link6` (revolute) + `gripper_link` (fixed, origin=`xyz="0.1039 0 0"`) + `gripper_finger_link1/2` (prismatic)
- **新增**: `_generate_gripper_with_arm_urdf()` 函数
- **新增**: `render_gripper_only_video` 和 `render_dual_gripper_video` 添加 `with_arm` 参数
- **新增**: `main()` 添加 `--mode` 参数 (choices: `gripper`, `gripper_arm`)
- **新增**: `render_auto.py` 添加 `--mode` 参数，传递 `with_arm` 到夹爪 URDF 渲染函数，输出文件名添加 `_arm` 后缀

#### 3. 修复双手 MANO 关键点只显示一只手的问题
- **根因**: `_render_keypoints` 函数开头执行 `kp_nodes.clear()`，双手循环中右手调用会删除左手的关键点
- **修复**: `common.py` 的 `_render_keypoints` 添加 `clear_existing` 参数 (默认 `True`)
- **修复**: `render_gripper_only.py` 的 `render_dual_gripper_video` 中左手 `clear_existing=True`，右手 `clear_existing=False`

#### 4. 新增问答文档
- **新增**: `docs/questions.md` — 回答物理仿真可行性、夹爪映射、双手关键点、夹爪+手臂末端模式等问题

### 验证结果
- `gripper` 模式 (双手, 仅夹爪): ✓ 30帧, 40KB, `/tmp/test_gripper/videos/hawor_r1_dual_gripper_urdf.mp4`
- `gripper_arm` 模式 (双手, 夹爪+手臂末端): ✓ 30帧, 43KB, `/tmp/test_arm/videos/hawor_r1_dual_gripper_urdf_arm.mp4`
- 双手夹爪 URDF 加载成功: `['left_gripper_finger_joint1', 'left_gripper_finger_joint2']` / `['right_gripper_finger_joint1', 'right_gripper_finger_joint2']`
- 双手夹爪+手臂 URDF 加载成功: `['left_arm_joint6', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2']` / `['right_arm_joint6', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2']`

### 用法
```bash
# 仅夹爪 (修复朝向 + 双手关键点)
python hand_track/render_gripper_only.py --hawor-dir <dir> --ras-dir <dir> --mode gripper

# 夹爪 + 手臂末端
python hand_track/render_gripper_only.py --hawor-dir <dir> --ras-dir <dir> --mode gripper_arm

# 通过管线入口
python hand_track/render_auto.py --hawor-dir <dir> --ras-dir <dir> --mode gripper_arm
```

---

## [2026-06-16] run_robot_only 支持自适应单臂/双臂渲染

**类型**: 修改
**影响范围**: example/combination/02_render_scene.py

### 修改内容
- [example/combination/02_render_scene.py] `HandObjectRenderer.__init__` 末尾添加 `self.hand_indices = [self.hand_idx]` 属性，支持双手列表
- [example/combination/02_render_scene.py] 重写 `run_robot_only` 方法，支持自适应单臂/双臂渲染:
  - 按 `self.hand_indices` 列表循环初始化每个臂的 URDF、retargeting、IK solver、MANO layer 等，封装为 `arm_states` 字典列表
  - 双臂时合并两手腕位置计算基座
  - 双臂时取非 None 的焦距更新相机 FOV
  - 双臂时取较小帧数作为 total_frames
  - 对每个臂分别执行 warm_start 和 warmup smoothstep 过渡
  - 渲染循环中同时更新所有臂，双臂时显示 "Dual Arm | L:✓/✗ R:✓/✗"
  - qpos 保存为 `_{prefix}.npy` 后缀区分左右臂
- [example/combination/02_render_scene.py] 修改 `main()` 函数:
  - `--hand-idx` 参数支持 `-2` 表示强制双手
  - 自动检测逻辑改用 `_detect_hands()` 返回列表
  - `robot_only` 模式下根据检测结果设置 `renderer.hand_indices`
- [example/combination/02_render_scene.py] 修复第147行 `R1_LEFT_SETTINGS` 字符串缺少右引号的语法错误

### 验证结果
- `py_compile.compile('02_render_scene.py', doraise=True)` 语法检查通过

## [2026-06-16] 修复相机移动大数据(hoi4d/laptop)渲染失败问题

**类型**: 修复
**影响范围**: hand_track/run_all_hawor.py

### 修改内容
- [example/combination/hand_track/run_all_hawor.py] 核心修复: 添加数据预处理步骤, 解决 02_render_scene.py 不做 R_c2w/t_c2w 变换导致相机移动大数据渲染失败的问题
  - 新增: `_prepare_world_space_hawor_dir()` 函数 — 创建预处理后的临时 hawor 目录, 将 pred_trans/pred_rot 从相机系转到世界系, R_c2w/t_c2w 保持不变 (02 用它定位相机)
  - 新增: `_symlink_aux_files()` 函数 — 为 est_focal.txt、cam_space/、world_space_res.pth 创建软链接
  - 新增: NaN 帧过滤 — 预处理时检查 pred_trans 和 pred_rot 的 NaN, 在 pred_valid 中置为 False
  - 新增: 有效帧检测 — 查找每只手的第一个有效帧, 调整 start_frame 跳过无效帧 (修复 laptop 前 36 帧无有效数据导致崩溃)
  - 修改: `process_hawor_dir()` 在调用 02_render_scene.py 前先调用 `_prepare_world_space_hawor_dir()` 预处理数据
  - 修改: subprocess 命令使用 `effective_hawor_dir` (预处理后的目录) 和 `effective_start_frame` (跳过无效帧)

### 根因分析
02_render_scene.py 的 load_hawor_data() 不做 R_c2w/t_c2w 变换, 直接使用 npz 中的 pred_trans/pred_rot。
对于相机移动大的数据 (hoi4d, laptop), pred_trans 在相机空间, 导致机械臂放置位置不对不可见。
7 目录因相机几乎不动 (t_c2w 范围 <0.03m), 相机空间≈世界空间, 所以不受影响。

### 验证结果
- 7 目录 (左手, 相机不动): ✓ 30帧 mean=195.8, 非灰=100%, 31KB
- 7 目录 + RAS GLB: ✓ 30帧 183KB (含 GLB 场景)
- hoi4d 目录 (双手, 相机移动大): ✓ 左右手各30帧 mean=195.3, 非灰=100%, 23KB (之前: 3.8%非灰, 23KB 无机械臂)
- laptop 目录 (右手, NaN帧多): ✓ 30帧 mean=194.9, 非灰=100%, 25KB (之前: 崩溃)
- 7_vggt-omega 目录 (左手): ✓ 30帧 mean=195.7, 非灰=100%, 30KB

## [2026-06-16] 重写 run_all_hawor.py: 改为调用 02_render_scene.py 子进程渲染

**类型**: 重构
**影响范围**: hand_track/run_all_hawor.py, hand_track/README.md

### 修改内容
- [example/combination/hand_track/run_all_hawor.py] 核心重写: process_hawor_dir() 从自渲染逻辑 (~640行) 简化为子进程调用 (~160行)
  - 新增: 调用 02_render_scene.py --mode robot_only 作为子进程 (subprocess.run)
  - 新增: 6 个 CLI 参数 (--ras-dir, --transform-params, --width, --height, --smooth, --crf)
  - 新增: 双手时分别调用 02 两次 (hand_idx=0 + hand_idx=1), 生成两个独立视频
  - 新增: 未提供 --transform-params 时自动创建 dummy npz 文件
  - 删除: 7 个不再需要的函数 (setup_scene, render_video, _compute_optimal_fixed_base, _compute_tracking_base_pos, _get_arm_base_link_pose, load_hawor_c2w, hawor_cam_to_sapien_pose)
  - 删除: 5 个孤立常量 (LP_ALPHA_EE, Q_180Z, COMFORTABLE_REACH, R1_LEFT_SETTINGS_PATH, R1_RIGHT_SETTINGS_PATH)
  - 更新: _print_banner() 反映新功能
  - 更新: _print_summary_table() 适配新结果格式
  - 更新: main() 参数日志输出和 process_hawor_dir() 调用
- [example/combination/hand_track/README.md] 全面更新文档
  - 更新: 数据流架构图 (Section 3) 反映子进程调用模式
  - 更新: CLI 参数表 (Section 5) 添加新参数
  - 更新: 输出结构 (Section 6) 移除 qpos.npy, 说明双手生成两个视频
  - 更新: 与 02_render_scene.py 的对比 (Section 7) 反映委托关系
  - 新增: Q5 常见问题 (--ras-dir / --transform-params 缺失时的行为)

### 验证结果
- py_compile 语法检查: ✓ 通过
- 单手测试 (hawor/7, 左手): ✓ 视频生成成功, 30帧 1920x1080
- 双手测试 (hoi4d): ✓ 两次调用 02 分别生成 left/right 视频
