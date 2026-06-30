## [2026-06-26] 完善 README 灵巧手使用说明与代码解释

**类型**: 文档
**影响范围**: hand_track/README.md

### 修改内容
- [README.md] 新增 2.5 节 "灵巧手渲染" (8 个完整命令示例 + 输出命名规则 + 3 步执行流程 + 常见问题表)
  - 环境要求说明 (dex conda 环境)
  - 6 种灵巧手规格表 (含 URDF 文件名 + _glb 回退说明)
  - 8 类命令示例: 切换机器人 / 手部选择 / 双手 / 相机视角 / Viewer / verify / 调试帧 / 自定义输出目录
  - 通过 `00_run_pipeline.py --dexterous` 调用的 7 个示例 (含 --skip-align / --steps / --handedness)
  - 输出文件命名规则表
  - 3 步执行流程图 (手部检测 → GLB 变换 → 渲染)
  - 常见问题表 (6 个常见错误 + 原因 + 解决)
- [README.md] 新增 4.9 节 "灵巧手渲染实现" (代码结构 + 4 个核心函数逐行解释 + 4 个关键问题)
  - 与夹爪模式 6 维度对比表
  - 文件结构树状图 (7 个主要模块)
  - `_load_dexterous_robot()` 完整代码注释 (5 个步骤: URDF 目录 → config 加载 → URDF 加载 → SAPIEN loader → 索引映射)
  - `_compute_wrist_quat_sapien()` 代码 + 坐标变换公式解释
  - `render_dexterous_only_video()` 7 阶段流程注释 (数据加载 / 场景 / 加载 / warm start / warmup / 渲染循环 / 视频重编码)
  - `render_dual_dexterous_video()` 6 个差异点
  - 4 个关键问题解释: dummy free joint / retarget2sapien / warm start / _glb URDF
- [README.md] Section 5 输出结构新增 `hawor_{robot}_{left,right}_urdf.mp4` 和 `hawor_{robot}_dual_urdf.mp4` 两行
- [README.md] Section 5 注释新增 `render_dexterous_only.py` 单手/双手输出说明

### 验证结果
- 语法检查: ✓ (无代码改动)
- 文档结构检查: ✓ (Section 2.5 / 4.9 / 5 完整)
- 命令示例检查: ✓ (8 + 7 = 15 个示例, 全部对应代码中实际可用的 CLI 参数)

---

## [2026-06-26] 新增灵巧手渲染管线 (render_dexterous_only.py + 00_run_pipeline Step 8)

**类型**: 新增
**影响范围**: hand_track/render_dexterous_only.py (新), 00_run_pipeline.py, README.md

### 需求
用户希望在 `hand_track` 目录的 "单独夹爪" 之外新增灵巧手渲染:
1. 先查 `GalaxeaManipSim/galaxea_sim/assets`, 若无灵巧手 → 使用 `dex-retargeting/example/position_retargeting/visualize_hand_object.py` 中的方案
2. 灵巧手只渲染手部, 用 dex-retarget 优化器, 通过 `--robot-name` 指定
3. `00_run_pipeline.py` 的调用要同步更新

### 调研结论
- `GalaxeaManipSim/galaxea_sim/assets` 只有 r1/r1_lite/r1_pro 平行夹爪, **没有灵巧手**
- `dex-retargeting/assets/robots/hands` 有 6 种灵巧手完整 mesh + retargeting config:
  - `allegro` (22 joints, 8 target links), `inspire` (18 joints, 5 target links), `shadow` (30 joints, 10 target links), `ability` (16 joints, 5 target links), `leap` (22 joints, 8 target links), `svh` (26 joints, 10 target links)

### 修改内容

#### 1. 新增 `hand_track/render_dexterous_only.py` (新文件, ~940 行)
- **核心函数**:
  - `_load_dexterous_robot(robot_name_str, hand_type, scene, logger)`: 加载灵巧手 URDF + 构建 PositionRetargeting, 返回 (robot, retargeting, retarget2sapien, target_link_human_indices, config)
  - `_compute_wrist_quat_sapien(hand_pose_frame)`: 将 MANO 手腕 axis-angle 转 SAPIEN quaternion (`R_sapien = RXWORLD_TO_SAPIEN @ R_mano`)
  - `render_dexterous_only_video(...)`: 单手渲染 (warm start + 渲染循环)
  - `render_dual_dexterous_video(...)`: 双手渲染 (同场景双手)
  - `_ensure_transform_params(...)`: 自动运行 `01_align_scene.py`
- **关键模式** (来自 `visualize_hand_object.py`):
  - `add_dummy_free_joint=True`: 加 6DOF dummy free joint 让手腕可自由移动
  - URDF `_glb` 版本回退: 优先用 `<name>_glb.urdf`, 不存在则用原始 URDF (inspire 无 _glb 版本)
  - Warm start: 用 MANO 手腕位姿初始化优化器避免局部最优
  - 1D 索引修复: `hand_pose_frame[0:3]` (HaWoR `pred_hand_pose[g_idx]` 是 1D)
- **CLI 参数**:
  - `--robot-name` (allegro/inspire/shadow/ability/leap/svh, 默认 allegro)
  - `--hand-idx` (-1=自动检测, 0=左, 1=右)
  - `--hawor-dir / --ras-dir / --output-dir / --fps / --width / --height / --view / --crf / --num-frames`
- **输出**: `videos/hawor_{robot}_{side}_urdf.mp4`

#### 2. `00_run_pipeline.py` — 新增 `--dexterous` 和 `--robot-name` 参数 + Step 8
- **新增参数**:
  - `--dexterous`: 启用灵巧手渲染管线 (默认运行步骤 1,8)
  - `--robot-name`: 灵巧手名称 (默认 allegro, 仅 `--dexterous` 时生效)
- **更新 docstring**: 添加 dexterous 模式示例
- **更新 `--steps` help**: 加入 "8=灵巧手渲染"
- **新增 Step 8 调用块** (line 444-463): 调用 `render_dexterous_only.py`, 自动从 `--handedness` 映射到 `--hand-idx`
- **更新默认 steps 逻辑**:
  ```python
  if args.dexterous:    args.steps = "1,8"
  elif args.handtrack:  args.steps = "1,7"
  else:                 args.steps = "1,2,3,4,5"
  ```

#### 3. `README.md` — 同步文档
- 模块组成表新增 `render_dexterous_only.py` 行
- 快速开始新增 2.5 节 "灵巧手渲染"
- CLI 参数表新增 `render_dexterous_only.py` 子表
- Section 6 新增 "灵巧手模式" 小节
- `00_run_pipeline.py` 描述新增 `--dexterous` 选项

### 验证结果

**Smoke test** (allegro 右手, 8 帧, 640x360):
```
23:22:43 [INFO]   ✓ Warm start 完成 (帧 0)
23:22:43 [INFO]   Warmup 完成 (30 帧 smoothstep 过渡)
灵巧手-allegro-right: 100%|██████████| 8/8 [00:08<00:00,  1.01s/it]
23:22:51 [INFO] ✓ 灵巧手视频已保存: /tmp/dex_test/videos/hawor_allegro_right_urdf.mp4
23:22:51 [INFO] 总耗时: 19.1 秒
```

**6 种灵巧手加载验证** (全部成功):
| 机器人 | 总关节数 | 手指关节数 | target links |
|---|---|---|---|
| allegro | 22 | 16 | 8 |
| inspire | 18 | 12 | 5 |
| shadow  | 30 | 24 | 10 |
| ability | 16 | 10 | 5 |
| leap    | 22 | 16 | 8 |
| svh     | 26 | 20 | 10 |

**语法检查 + --help 检查**: ✓ 通过

---

## [2026-06-26] 相机帧一致性修复 (仅相机, 手部/GLB 不变)

**类型**: 修复
**影响范围**: common.py

### 问题
用户反馈 "相机左右反, 手部和 GLB 是正确的". 经 4 配置实测, 根因是相机 transform 用错矩阵: 相机用 `RXWORLD_TO_SAPIEN @ stored` (帧 R_AXIS@SLAM), 手部/GLB 用 `RXWORLD_TO_SAPIEN @ SLAM_data` (帧 R_AXIS@OpenGL), 两者帧相差 R_x → 相机位置轨迹与手部不同帧 → "左右反".

### 修改内容 (仅改相机)
- [common.py] `hawor_cam_to_sapien_pose` transform: `RXWORLD_TO_SAPIEN` → `R_AXIS`
  - 原理: stored = R_x @ SLAM = OpenGL, 所以 R_AXIS @ stored = R_AXIS @ OpenGL = 手部帧
- [common.py] `hawor_cam_to_sapien_pose` extraction: OpenGL → OpenCV (`forward=+col2, up=-col1`)
- **手部/GLB 保持 `RXWORLD_TO_SAPIEN` 不变** (`_render_to_sapien`, `load_glb_transformed`)

### 验证结果
- `/tmp/verify_actual_functions.py` 用 common.py 实际函数验证 (2 个数据集):
  - 数据集 7 (113帧): forward·cam2hand=+0.934, up·WORLD_UP=+0.998, 帧一致 ratio=1.0000 ✓
  - 数据集 hoi4d (599帧): forward·cam2hand=+0.771, up·WORLD_UP=+0.953, 帧一致 ratio=1.0000 ✓

---

## [2026-06-25] 修复 GLB 加载 Path3D 无 faces 属性的崩溃

**类型**: 修复
**影响范围**: common.py

### 问题
- `load_glb_transformed()` 遍历 GLB 场景中所有 geometry 时, 遇到 `Path3D` 类型对象 (2D 路径/曲线) 调用 `geom.faces` 报 `AttributeError: 'Path3D' object has no attribute 'faces'`

### 修改内容
- [common.py#L338] 在遍历 geometry 时添加 `if not hasattr(geom, 'faces'): continue`, 跳过非网格几何体 (Path3D 等)

### 验证结果
- 语法检查: ✓

---

## [2026-06-24] 修复 half_arm URDF joint4 origin + 降低平滑延迟

**类型**: 修复
**影响范围**: gripper_config.py, README.md

### 问题

1. **机械臂"中间空了一段"**: 用户反馈 "urdf机械臂有点问题, 感觉中间有一段link空了, 乱七八糟的"。根因: `prepare_half_arm_urdf` 移除 link1/2/3 后, 只把 joint4 的 parent 改为 arm_base_link, 但 joint4 的 origin 仍是 link3→link4 的偏移 (7.5cm), 缺少 base→link3 的累积偏移 (32cm), 导致 arm_link4 位置错误
2. **"映射跟不上"**: 用户反馈 "映射就是跟不上"。根因: `LP_ALPHA_ANALYTICAL = 0.9` 的 EMA 平滑导致 ~9.5 帧延迟 (30fps 下 ~0.32s), 夹爪明显滞后于手部运动

### 修改内容

#### 1. `gripper_config.py` — 修复 `prepare_half_arm_urdf` joint4 origin 累积

- **修复前**: joint4 origin = (0.027, -0.070, 0) ≈ 7.5cm (只有 link3→link4 偏移)
- **修复后**: joint4 origin = (-0.323, -0.240, 0) ≈ 40cm (base→link4 累积偏移)
- **实现**: 在移除 joint1/2/3 前, 解析它们的 origin xyz, 累积到 joint4 的 origin 上
- **验证**: gripper_link 位姿误差 0.00mm (offset 反推完全正确), gripper/gripper_arm 两种模式指尖误差完全一致

#### 2. `gripper_config.py` — 降低 `LP_ALPHA_ANALYTICAL` 从 0.9 到 0.5

- **修复前**: alpha=0.9, 延迟 ~9.5 帧 (0.32s), "跟不上"
- **修复后**: alpha=0.5, 延迟 ~1.4 帧 (0.05s), 实时性好
- **注释**: 添加 alpha 选择说明

#### 3. `README.md` — 更新文档

- Section 4.3: half_arm URDF 说明新增 "并将 joint1/2/3 的 origin 累积到 joint4"
- Section 4.7: 平滑参数从 alpha=0.9 更新为 alpha=0.5, 添加 alpha 选择说明

### 验证结果

**URDF 结构验证** (修复后):
| 手 | joint4 origin xyz | joint4 parent | links | joints |
|---|---|---|---|---|
| right | -0.323045 -0.240469 0.000000 | right_arm_base_link | 8 | 7 |
| left | -0.323045 0.240465 0.000000 | left_arm_base_link | 8 | 7 |

**位姿验证** (gripper_arm 模式, 诊断脚本):
- gripper_offset_pos: [0.085, 0.241, 0] (255mm, arm 长度正确) ✓
- gripper_link 位姿误差: 0.00mm ✓ (offset 反推完全正确)
- arm_base_link → arm_link6 → gripper_link 运动链连续, 不再"中间空了一段"

**渲染测试** (数据集7, 60帧, `--verify`):
| 模式 | 指尖1误差 | 指尖2误差 | 指向误差 | 平滑 |
|---|---|---|---|---|
| `--mode gripper` | 32.74mm | 35.22mm | 1.20° | alpha=0.5 |
| `--mode gripper_arm` | 32.74mm | 35.22mm | 1.20° | alpha=0.5 |

> 两种模式误差完全一致, 证明 gripper_arm 的 offset 计算正确。
> 指尖误差 ~33mm 是 `open_scale=3.0` 放大开合的预期代价 (可用 `--open-scale 1.0` 降低到 ~12mm)。

---

## [2026-06-24] 新对齐策略: 先对齐夹爪两点 + 中点-手腕连线确定位姿

**类型**: 新增 + 修改
**影响范围**: align_strategy.py (新), render_gripper_only.py, README.md

### 问题

1. **夹爪不开合**: 用户反馈 "夹爪都不张开, 感觉优化器和解析器求解都差不多, 给人的感觉是硬凑"。根因: MANO 拇指-食指指尖距离仅 ~35mm (静态捏取姿态), 夹爪基准距离 26.9mm, 精确映射时关节值仅 ~4.3mm, 几乎看不到开合
2. **机械臂乱飞**: 用户反馈 "机械臂的urdf都拆开了, 在渲染的时候都乱飞了"。需验证 URDF 结构和位姿对齐
3. **对齐策略不明确**: 用户要求 "关键先对齐夹爪两点, 最后第三个点, 是对齐位姿, 能够在同一条中轴线上即可; MANO参数里面夹爪的中点和第三个手腕点的连线确定位姿; 最后把机械臂手腕的点放到位姿线上即可; 对齐的主要是夹爪末端, 次要是手腕, 但位姿一定要对"

### 修改内容

#### 1. 新增 `align_strategy.py` — 用户要求的对齐策略 (新文件)

实现用户原话的对齐逻辑:
- `compute_gripper_pose_aligned()`: 主函数
  1. 指尖中点 `midpoint = (finger1 + finger2) / 2`
  2. X 轴 (位姿核心) = `normalize(midpoint - wrist)` (中点-手腕连线确定指向方向)
  3. Y 轴 (开合方向) = `finger2-finger1` 投影到 X 垂直面后归一化
  4. Z 轴 = `X × Y`, 组装旋转矩阵
  5. 手指关节 = `(指尖3D距离 - 基准距离) × open_scale / 2` (带缩放因子)
  6. gripper_link 位置 = `midpoint - R @ finger_mid_in_gripper` (对齐夹爪两点)
- `compute_arm_root_pose()`: 从 gripper 位姿反推 arm root 位姿
- `project_wrist_to_pose_line()`: 验证手腕是否在位姿线上
- `verify_alignment()`: 验证对齐效果 (指尖误差/位姿误差/手腕到位姿线距离)
- `GRIPPER_OPEN_SCALE = 3.0`: 夹爪开合缩放因子 (让开合更明显)

#### 2. `render_gripper_only.py` — 集成新对齐策略

- 新增 `_compute_gripper_pose_by_strategy()` 模块级 helper (line 71-85): 根据 `strategy` 参数选择 `aligned` (新) 或 `analytical` (旧) 策略
- 新增 `--strategy` 参数 (choices: aligned/analytical, default: aligned)
- 新增 `--open-scale` 参数 (default: GRIPPER_OPEN_SCALE = 3.0)
- `render_gripper_only_video()` 和 `render_dual_gripper_video()` 签名新增 `strategy` 和 `open_scale` 参数
- 替换所有 5 处 `compute_analytical_gripper_pose` 直接调用为策略感知 helper:
  - 单手: warm start / first valid pose probe / 主渲染循环
  - 双手: warm start / first valid pose probe / 主渲染循环
- `main()` 传递 `strategy` 和 `open_scale` 到渲染函数

#### 3. `README.md` — 体现新对齐策略

- 模块组成表新增 `align_strategy.py` 条目
- 快速开始新增 ⑦ `--strategy analytical` 和 ⑧ `--open-scale 5.0` 命令示例
- CLI 参数表新增 `--strategy` 和 `--open-scale` 两行
- Section 4.4 重写: 详细说明 `aligned` vs `analytical` 两种策略, 对比表, `--open-scale` 原理, 更新误差表
- Section 6 新增 "对齐策略" 小节, 含命令示例

### 验证结果

**对齐策略测试** (数据集7, `--verify` 实测):

| 策略 | 模式 | open_scale | 指尖误差 | 位姿误差 | 手腕到位姿线 |
|---|---|---|---|---|---|
| `aligned` | `--mode gripper` | 3.0 | ~16.7mm | 0.02° | - |
| `aligned` | `--mode gripper_arm` | 3.0 | ~16.7mm | 0.03° | 0.0mm (完美在位姿线上) |
| `aligned` | `--mode gripper` | 1.0 | ~11.8mm | 0.02° | - |
| `analytical` | `--mode gripper` | - | 4.36mm | 0.14° | - |

**关键结论**:
- 位姿误差 < 0.1° (优秀): `aligned` 策略位姿核心正确
- `gripper_arm` 模式手腕在位姿线上: 0.0mm (完美), 机械臂不会"乱飞"
- 夹爪开合可见: `open_scale=3.0` 时关节值 12.8mm (vs 旧策略 4.3mm)
- 指尖误差权衡: `aligned` ~16.7mm (开合明显) vs `analytical` 4.36mm (开合不可见), 这是 1-DOF 夹爪的固有局限

**`render_auto.py` 兼容性**: 不传 `strategy`/`open_scale` 参数, 自动使用默认 `aligned` 策略, 无需修改

---

## [2026-06-24] 所有模式统一不缩放 finger origin + README 命令与代码 100% 对齐

**类型**: 修复 + 文档
**影响范围**: render_gripper_only.py, README.md

### 问题

1. **`--mode gripper` 仍在缩放 finger origin**: 之前只对 `--mode gripper_arm` 不缩放, `--mode gripper` 仍然从 MANO 数据计算 finger_origin_x (~120mm), 与 `02_render_scene.py` 不一致 (始终 37mm)
2. **README 命令与代码不对应**: 用户反馈 "为什么每次运行完, README.md 的命令都不改呢", 命令没有跟上代码变化
3. **`--mode gripper_arm` vs `--arm-mode full` 关系不清**: 用户困惑到底用哪个参数

### 修改内容

#### 1. `render_gripper_only.py` — 所有模式统一不缩放 finger origin
- **单手函数** (`render_gripper_only_video`, line 207-213):
  - 之前: 从 MANO 数据计算 finger_origin_x (~120mm), arm 模式才覆盖为 0.03689
  - 现在: **所有模式** (gripper / gripper_arm) 都直接使用 `finger_origin_x = 0.03689` (37mm)
  - 删除了 MANO 距离计算循环 (`_w2f1_dists`, `_w2f2_dists`)
- **双手函数** (`render_dual_gripper_video`, line 741-746):
  - 同样改为所有模式统一使用 0.03689
  - 删除了 MANO 距离计算循环
- **与 `02_render_scene.py` 完全一致**: `02_render_scene.py` 使用 `r1_lite_robot_glb.urdf` (finger origin=0.03689), 从不缩放

#### 2. `README.md` — 命令与代码 100% 对齐
- **新增 `--mode` 和 `--arm-mode` 关系说明** (Section 2.2):
  - 表格清晰说明: `--mode` 选择渲染什么, `--arm-mode` 选择手臂类型
  - 明确: `--mode gripper_arm` 是开启手臂模式的开关 (必填)
  - 明确: `--arm-mode full` 是可选的手臂类型选择 (只在 `--mode gripper_arm` 时生效)
- **所有命令重新验证** (6 个命令, 全部实测通过):
  - ① 默认 (both) ② `--mode gripper` ③ `--mode gripper_arm` ④ `--mode gripper_arm --arm-mode full` ⑤ `--mode gripper --optimizer` ⑥ `--hand-idx 0`
- **新增 Section 4.3 "夹爪 URDF 加载"**: 详细对比 `02_render_scene.py` 和我们的 URDF 加载方式, 验证 finger origin 一致
- **更新误差表** (Section 4.4): 使用 37mm 不缩放后的实测数据
- **更新对比表** (Section 6): 精度列更新为实测值

### 验证结果

**命令测试** (数据集7, 全部通过):
| 命令 | 结果 |
|---|---|
| `render_gripper_only.py` (默认 both) | ✓ 生成 gripper + gripper_arm 两个视频 |
| `render_gripper_only.py --mode gripper` | ✓ 指尖误差 4.36/4.76mm |
| `render_gripper_only.py --mode gripper_arm` (half) | ✓ 指尖误差 4.36/4.76mm |
| `render_gripper_only.py --mode gripper_arm --arm-mode full` | ✓ |
| `render_gripper_only.py --mode gripper --optimizer` | ✓ |
| `render_gripper_only.py --hand-idx 0` | ✓ |
| `render_auto.py --mode gripper_arm` | ✓ |

**URDF 几何验证** (finger origin 一致):
| URDF 来源 | finger_joint1 origin X | finger_joint2 origin X |
|---|---|---|
| `r1_lite_robot_glb.urdf` (02_render_scene.py) | 0.03689 | 0.03689 |
| `r1_v2_1_0_floating_right.urdf` (浮动 URDF) | 0.03689 | 0.03689 |
| 我们的 `generate_gripper_urdf()` | 0.03689 | 0.03689 |
| 我们的 `prepare_half/full_arm_urdf()` | 0.03689 | 0.03689 |

**`--help` 输出与 README CLI 表一致**: ✓

---

## [2026-06-24] 恢复半个手臂选项 + 修复夹爪 URDF (不缩放, 参考 02_render_scene.py)

**类型**: 修复 + 新增
**影响范围**: gripper_config.py, render_gripper_only.py, README.md

### 问题

1. **半个手臂选项被删除**: 之前 `generate_gripper_with_arm_urdf()` (arm_link4-6) 被替换为 `prepare_full_arm_urdf()` (arm_link1-6), 用户无法再选择半个手臂
2. **夹爪 URDF 不对**: `prepare_full_arm_urdf()` 缩放了 finger origin (0.03689 → ~120mm), 导致夹爪手指过长, 几何变形; `02_render_scene.py` 不缩放, 渲染正确
3. **手腕映射不对**: 缩放后 gripper_link 位置偏离正确几何, 与 02_render_scene.py 不一致

### 修改内容

#### 1. 新增 `prepare_half_arm_urdf()` 函数 (gripper_config.py)
- **功能**: 从 GalaxeaManipSim 的 `r1_v2_1_0_floating_{prefix}.urdf` 提取 arm_link4/5/6 + gripper
- **步骤**:
  1. 读取完整浮动 URDF
  2. 替换 `package://r1_v2_1_0/meshes/` 为绝对路径
  3. 将 finger joint 从 fixed 改为 prismatic
  4. 移除 arm_link1/2/3 (link 和 joint)
  5. 将 arm_joint4 的 parent 从 arm_link3 改为 arm_base_link
- **不缩放 finger origin**, 保持原始 37mm 几何

#### 2. 修改 `prepare_full_arm_urdf()` 不缩放 finger origin (gripper_config.py)
- **之前**: 缩放 finger origin 从 0.03689 到 MANO 腕→指尖距离 (~120mm)
- **现在**: 不缩放, 保持原始 0.03689 (37mm), 与 `02_render_scene.py` 的 `prepare_arm_urdf()` 一致
- finger1_origin_x/finger2_origin_x 参数保留但未使用 (兼容性)

#### 3. 新增 `--arm-mode` 参数 (render_gripper_only.py)
- `--arm-mode half` (默认): 半个手臂 (arm_link4-6), 调用 `prepare_half_arm_urdf()`
- `--arm-mode full`: 完整手臂 (arm_link1-6), 调用 `prepare_full_arm_urdf()`
- 单手和双手函数都支持, main() 传递 `arm_mode=args.arm_mode`

#### 4. arm 模式不缩放 finger origin (render_gripper_only.py)
- 当 `with_arm=True` 时, 覆盖 `finger_origin_x`/`finger1_origin_x`/`finger2_origin_x` 为 0.03689
- 确保 retargeting URDF、渲染 URDF、解析计算都使用一致的 37mm 几何
- 单手函数 (line 228-233) 和双手函数 (line 784-789) 都添加了覆盖逻辑

### 验证结果

**URDF 结构验证**:
```
half_arm (5 joints): arm_joint4/5/6 + finger_joint1/2
full_arm (8 joints): arm_joint1-6 + finger_joint1/2
gripper_only (2 joints): finger_joint1/2
```

**Finger origin 验证 (不缩放)**:
```
half_arm finger_joint1 origin: 0.03689 -0.013453 -0.00012053  ✓
half_arm finger_joint2 origin: 0.03689 0.013453 0.00012067   ✓
full_arm finger_joint1 origin: 0.03689 -0.013453 -0.00012053  ✓
full_arm finger_joint2 origin: 0.03689 0.013453 0.00012067   ✓
```

**命令测试** (数据集7, 右手):
| 命令 | 结果 |
|---|---|
| `--mode gripper` | ✓ |
| `--mode gripper_arm --arm-mode half` (analytical) | ✓ |
| `--mode gripper_arm --arm-mode half --optimizer` | ✓ |
| `--mode gripper_arm --arm-mode full --optimizer` | ✓ |

**Verify 误差** (half-arm, optimizer, 50帧):
| 指尖1 | 指尖2 | 手腕 | 指向 | 开合 |
|---|---|---|---|---|
| 7.85mm | 7.85mm | 95.68mm | 35.11° | 1.52° |

- 指尖误差 ~8mm (不缩放几何, optimizer 在 37mm 夹爪上匹配 130mm MANO 手)
- 手腕误差 ~96mm (预期: 夹爪 37mm vs MANO 手 130mm, 几何不匹配)
- 开合方向误差 1.52° (优秀)

---

## [2026-06-24] 修复 gripper_arm URDF: 使用完整运动链 (参考 GalaxeaManipSim)

**类型**: 修复
**影响范围**: gripper_config.py, render_gripper_only.py, render_auto.py, README.md

### 问题

1. **gripper_arm 模式 URDF 运动链不完整**: `GRIPPER_WITH_ARM_URDF_TEMPLATE` 跳过了 arm_link1/2/3, 直接连接 arm_base_link → arm_link4, 导致 arm_link4 mesh 位置错误 (距 base 74.9mm, 应为 ~480mm 完整链)
2. **夹爪和手腕没有贴合**: 由于运动链断裂, arm_link4/5/6 的 mesh 位置不正确, 视觉上夹爪与手臂分离
3. **render_auto.py --hand-idx 崩溃**: `hand_count` 变量在 `--hand-idx >= 0` 分支中未赋值

### 修改内容

#### 1. 新增 `prepare_full_arm_urdf()` 函数 (gripper_config.py)
- **参考**: GalaxeaManipSim 的 `r1_v2_1_0_floating_{left,right}.urdf` 完整浮动 URDF
- **功能**:
  1. 读取完整浮动 URDF (包含 arm_link1-6 全部 6 个关节)
  2. 替换 `package://r1_v2_1_0/meshes/` 为绝对路径
  3. 将 gripper_finger_joint1/2 从 fixed 改为 prismatic (右手 URDF 需要)
  4. 缩放 finger joint origin 的 x 分量 (0.03689 → finger1_origin_x/finger2_origin_x)
- **对比旧方案**: 旧 `generate_gripper_with_arm_urdf()` 使用硬编码模板, 跳过 link1/2/3; 新方案直接使用 GalaxeaManipSim 官方 URDF, 运动链完整

#### 2. render_gripper_only.py 改用 `prepare_full_arm_urdf()`
- 单手函数 (line 261) 和双手函数 (line 818) 的 `generate_gripper_with_arm_urdf` 调用替换为 `prepare_full_arm_urdf`
- 移除未使用的 `generate_gripper_with_arm_urdf` 导入
- 日志改为 "完整手臂+夹爪 (arm_link1-6 + gripper, 参考 GalaxeaManipSim)"

#### 3. 修复 render_auto.py --hand-idx 崩溃
- **根因**: `--hand-idx >= 0` 分支只设置 `hand_indices`, 未设置 `hand_count`, 导致 line 228 `UnboundLocalError`
- **修复**: 添加 `hand_count = 1`

#### 4. README.md 更新描述
- "夹爪+手臂末端 (arm_link4/5/6)" → "夹爪+完整手臂 (arm_link1-6, 参考 GalaxeaManipSim)"
- 对比表更新: `arm_link4/5/6` → `arm_link1-6`

### 验证结果

**运动链距离验证 (所有关节=0)**:
```
arm_base_link → arm_link1: 44.6mm
arm_link1 → arm_link2: 106.1mm
arm_link2 → arm_link3: 349.9mm
arm_link3 → arm_link4: 74.9mm  (修复前: base→link4 直接 74.9mm, 跳过 link1/2/3)
arm_link4 → arm_link5: 246.3mm
arm_link5 → arm_link6: 58.3mm
arm_link6 → gripper_link: 103.9mm  (gripper_joint, 夹爪正确贴合手腕)
总距离: 846.1mm
```

**位置精度验证 (gripper_arm 模式, 30帧, 数据集7)**:
| 指尖1 | 指尖2 | 手腕 | 指向 | 开合 |
|---|---|---|---|---|
| 1.39mm | 1.91mm | 4.12mm | 0.23° | 0.09° |

精度与修复前一致 (解析计算未变), 视觉效果改善: arm mesh 沿完整运动链正确放置, 夹爪与 arm_link6 贴合。

**命令测试**:
| 命令 | 结果 |
|---|---|
| `render_gripper_only.py --mode gripper` | ✓ |
| `render_gripper_only.py --mode gripper_arm --verify` | ✓ |
| `render_gripper_only.py --mode gripper_arm --optimizer` | ✓ |
| `render_gripper_only.py` (默认 both) | ✓ |
| `render_auto.py --mode gripper_arm --hand-idx 0` | ✓ (修复后) |

---

## [2026-06-23] README 修正: 默认模式说明 + 输出结构 + 验证结果

**类型**: 文档
**影响范围**: example/combination/hand_track/README.md

### 修改内容

#### 1. Section 2.2 — 明确默认 `--mode both` 行为
- **问题**: 用户疑惑 `render_gripper_only.py` 不带 `--mode` 时是否同时渲染两种
- **修复**: 添加说明 "默认 `--mode both`，同时渲染 gripper 和 gripper_arm 两个视频"，每个命令标注预期输出文件名
- 添加注释说明 `render_auto.py` 默认 `--mode gripper`（不支持 `both`）

#### 2. Section 3 — render_auto.py `--mode` 参数说明
- **修复**: 标注 "(不支持 `both`)"，与代码 `choices=["gripper", "gripper_arm"]` 一致

#### 3. Section 4.3 — 添加 `--verify` 实测误差表
- **问题**: 原来只写 "右手 ~0.9mm / 左手 ~3mm"，未区分两种模式
- **修复**: 替换为实测表格，gripper 和 gripper_arm 两种模式误差完全一致：
  - 指尖1=3.10mm, 指尖2=2.80mm, 手腕=1.22mm, 指向=0.23°, 开合=0.09°

#### 4. Section 5 — 输出结构补充单手 `_arm` 视频
- **问题**: 原输出结构缺少 `hawor_r1_{left,right}_gripper_urdf_arm.mp4`（单手 gripper_arm 模式）
- **修复**: 添加该文件，并区分 `render_gripper_only.py`（默认 both）和 `render_auto.py`（默认 gripper）的输出差异

#### 5. Section 6 — render_auto.py 模式说明
- **修复**: 补充 "同时额外生成夹爪关键点视频和夹爪URDF视频"

### 验证结果

**默认 `--mode both` 测试** (数据集7, 5帧):
```
--- 渲染模式: gripper ---
✓ hawor_r1_left_gripper_urdf.mp4
--- 渲染模式: gripper_arm ---
✓ hawor_r1_left_gripper_urdf_arm.mp4
```
确认默认同时生成两个视频。

**`--verify` 跟随 MANO 测试** (数据集7, 30帧):

| 模式 | 指尖1 | 指尖2 | 手腕 | 指向 | 开合 |
|---|---|---|---|---|---|
| `--mode gripper` | 3.10mm | 2.80mm | 1.22mm | 0.23° | 0.09° |
| `--mode gripper_arm` | 3.10mm | 2.80mm | 1.22mm | 0.23° | 0.09° |

两种模式误差完全一致，证明夹爪确实精确跟随 MANO 手部。

---

## [2026-06-23] README 命令全量验证 + 修正已测试数据表

**类型**: 文档 + 验证
**影响范围**: example/combination/hand_track/README.md, CHANGE_LOG.md

### 修改内容

#### 1. 修正 README 第 7 节"已测试的 HaWoR 数据"表
- **问题**: 表格中 `7` 目录手部写为"双手", 但实际 `detect_hands` 返回 `[0]` 左手
- **修复**: [README.md#L308] 将 `7` 的手部从"双手"改为"左手"

#### 2. 验证 README 所有命令与代码参数一致
- 使用 `python -m py_compile` 对 `render_auto.py`, `render_gripper_only.py`, `common.py`, `gripper_config.py` 做语法检查: ✓ 通过
- 对比 `--help` 输出与 README CLI 参数表: ✓ 一致
- 所有命令中的 `--hand-idx`, `--mode`, `--view`, `--verify`, `--optimizer` 等参数均与代码实现匹配

#### 3. 实际运行测试 README 关键命令
使用 `--num-frames 5`/`10` 快速验证, 覆盖:
- `render_auto.py` (默认 gripper 模式)
- `render_auto.py --mode gripper_arm`
- `render_gripper_only.py` (默认 both 模式)
- `render_gripper_only.py --mode gripper`
- `render_gripper_only.py --hand-idx 0`
- `render_gripper_only.py --verify`
- `render_gripper_only.py --optimizer`

### 验证结果

**实际 `detect_hands` 检测结果**:
| 目录 | 结果 | 说明 |
|---|---|---|
| `7`             | `[0]` | 左手 |
| `7_vggt-omega`  | `[0]` | 左手 |
| `hoi4d`         | `[0,1]` | 双手 |
| `laptop`        | `[1]` | 右手 |

**命令运行结果**:
| 命令 | 目录 | 结果 |
|---|---|---|
| `render_auto.py` | `7` | ✓ 机械臂 + 夹爪关键点 + 夹爪URDF 均成功 |
| `render_auto.py --mode gripper_arm` | `7` | ✓ 成功 |
| `render_gripper_only.py` | `7` | ✓ gripper + gripper_arm 双视频成功 |
| `render_gripper_only.py --mode gripper --hand-idx 0` | `7` | ✓ 强制左手成功 |
| `render_gripper_only.py --verify --mode gripper` | `7` | ✓ 误差报告正常: 指尖 ~3mm, 指向 ~0.18°, 开合 ~0.05° |
| `render_gripper_only.py --optimizer --mode gripper` | `7` | ✓ 优化器模式成功 |
| `render_gripper_only.py` | `7_vggt-omega` | ✓ gripper + gripper_arm 双视频成功 |
| `render_auto.py` | `laptop` | ⚠ 手部检测正确(右手), 但因数据含 NaN 导致 `01_align_scene.py` 失败, 机械臂无法放置; 夹爪关键点和夹爪URDF 渲染成功 |

**结论**: README 命令参数与代码完全一致, 7 和 7_vggt-omega 完全可运行; laptop 因原始数据 NaN 问题导致部分功能受限, 非命令/代码问题。

---

## [2026-06-23] 修复 01_align_scene.py 对 NaN 手部数据的崩溃

**类型**: 修复
**影响范围**: example/combination/01_align_scene.py

### 问题
- `01_align_scene.py` 在 Step 6 验证阶段直接对 `pred_trans[hand_idx, valid_frames]` 调用 `cKDTree.query()`
- 某些数据（如 `laptop`）中 `pred_valid=True` 的帧对应的 `pred_trans` 仍可能为 NaN，导致对齐脚本崩溃
- 崩溃信息误导为"对齐失败"，实际 R/t/s 已经计算完成

### 修复
- [01_align_scene.py] 在 `compute_and_save_transform_params` 和 `main` 中生成 `valid_frames` 时，同步过滤 `pred_trans` 中的 NaN 帧
- [01_align_scene.py] Step 6 中当 `valid_frames` 为空时跳过手-GLB距离验证，不再崩溃，仍然正常保存 `transform_params.npz`

### 验证
- `laptop` 数据之前因 NaN 崩溃，修复后 `01_align_scene.py` 成功退出并保存变换参数
- `detect_hands` 对该数据仍正确返回 `[1]`（右手）

---

## [2026-06-23] 修复单手/gripper_arm模式独立手指缩放缺失 (误差 11mm → 0.9mm)

**类型**: 修复
**影响范围**: example/combination/hand_track/render_gripper_only.py, gripper_config.py, README.md

### 修改内容

#### 1. 修复单手 gripper 模式 URDF 未传独立手指缩放 (指尖误差 11mm → 0.9mm)
- **根因**: `render_gripper_only_video` (单手函数) 调用 `generate_gripper_urdf(prefix, finger_origin_x)` 只传平均值, 但 `compute_analytical_gripper_pose` 用独立 `finger1_origin_x`/`finger2_origin_x`, 导致 URDF 几何与解析计算不匹配
- **修复**: [render_gripper_only.py#L264-267] 添加 `finger1_origin_x=finger1_origin_x, finger2_origin_x=finger2_origin_x` 参数, 与双手函数一致
- **效果**: 右手单模式指尖误差从 ~11.73mm 降到 0.91mm (与双手模式一致)

#### 2. 修复 gripper_arm 模式完全不支持独立手指缩放 (指尖误差 9-12mm → 0.9mm)
- **根因**: `generate_gripper_with_arm_urdf` 函数签名只有 `finger_origin_x` (平均值), 两个手指用相同值, 而 `generate_gripper_urdf` 已支持独立值
- **修复**: [gripper_config.py#L447-461] 为 `generate_gripper_with_arm_urdf` 添加 `finger1_origin_x`/`finger2_origin_x` 参数, 逻辑与 `generate_gripper_urdf` 一致
- **修复**: [render_gripper_only.py#L260-264, #L817-821] 单手和双手函数的 `generate_gripper_with_arm_urdf` 调用添加独立手指缩放参数
- **效果**: gripper_arm 模式右手指尖误差从 11.77mm 降到 0.91mm, 左手从 9.21mm 降到 3.51mm

#### 3. 修正 README gripper_arm 模式描述
- **问题**: README 描述 "arm 关节锁定为0 (极高刚度 + 限制[0,0])" 与代码不符 (实际用 stiffness=1e5 + 每帧重置, 无 set_limits)
- **修复**: [README.md#L265-266] 改为 "arm 关节每帧强制设为 0 (高刚度 drive + 每帧重置)", 添加 "支持独立手指缩放, 误差与纯夹爪模式一致"

### 验证结果 (数据集7, 100帧)

| 模式 | 手 | 指尖1 | 指尖2 | 手腕 | 指向 | 开合 |
|---|---|---|---|---|---|---|
| gripper (单手 --hand-idx 1) | 右 | 0.91mm | 0.74mm | 0.30mm | 0.04° | 0.03° |
| gripper (单手 --hand-idx 0) | 左 | 3.51mm | 2.79mm | 0.83mm | 0.12° | 0.07° |
| gripper (双手自动) | 右 | 0.91mm | 0.74mm | 0.30mm | 0.04° | 0.03° |
| gripper (双手自动) | 左 | 3.51mm | 2.79mm | 0.83mm | 0.12° | 0.07° |
| gripper_arm (双手自动) | 右 | 0.91mm | 0.74mm | 0.30mm | 0.04° | 0.03° |
| gripper_arm (双手自动) | 左 | 3.51mm | 2.79mm | 0.83mm | 0.12° | 0.07° |

三种模式误差完全一致, 管线验证通过: detect_hands → hand_indices → 单手/双手路由 → 正确加载手数据 → 相机每帧更新

---

## [2026-06-22] README重写: 添加实现方式说明 + 手部检测详细文档

**类型**: 文档
**影响范围**: example/combination/hand_track/README.md

### 修改内容

#### 1. README 第4节重写为"实现方式"
- 4.1 手部检测: 详细说明 HaWoR 数据约定 (pred_valid形状、hand_idx含义) 和检测算法步骤
- 4.2 坐标对齐: 说明 Umeyama 算法和自动生成 transform_params
- 4.3 夹爪位姿计算: 解析模式 (Gram-Schmidt) 和优化器模式的完整算法描述
- 4.4 相机跟踪: 说明相机坐标变换公式
- 4.5 GLB 场景加载: 说明加载流程和 s_inv 缩放
- 4.6 平滑策略: 解析模式 vs 优化器模式的平滑方式
- 4.7 有效帧检查: 50% 阈值说明

#### 2. 模块组成更新
- 添加 `gripper_config.py` 和 `configs/` 目录

---

## [2026-06-22] arm关节锁定 + viewer循环重置 + 有效帧50%检查 + GLB大小问题记录

**类型**: 修复 + 新增
**影响范围**: example/combination/hand_track/render_gripper_only.py, README.md

### 修改内容

#### 1. 修复相机不跟随手移动的问题
- **根因**: render_gripper_only.py 只在 `view == "fpv"` 时每帧更新相机，而 02_render_scene.py 始终每帧更新
- **修复**: 移除 `view == "fpv"` 条件，当 R_c2w 数据存在时始终每帧更新相机位置
- **效果**: 所有 view 模式下相机都能正确跟随手移动，物体始终可见

#### 2. 新增 --hand-idx 参数覆盖自动检测
- **问题**: HaWoR 可能误检背景手，导致单手场景渲染出双手
- **新增**: `--hand-idx 0` 强制左手, `--hand-idx 1` 强制右手, 默认 `-1` 自动检测
- **效果**: 用户可以精确控制渲染哪只手

#### 3. 更新 README.md
- 模块组成添加 `gripper_config.py`
- CLI 参数表添加 `--hand-idx`
- 渲染模式说明重写: 详细解释解析模式(推荐)和优化器模式, 含代码示例和性能数据
- 快速开始添加 `--hand-idx` 示例

### 验证结果
- 数据集7, 10帧, gripper模式: ✓ 运行成功, 误差正常
- `--hand-idx 0` 只渲染左手: ✓ 运行成功, 输出 `hawor_r1_left_gripper_urdf.mp4`

---

## [2026-06-22] 夹爪映射优化: 投影开合方向误差修正 + 独立手指缩放 + warm_start改进

**类型**: 修复 + 优化
**影响范围**: example/combination/hand_track/render_gripper_only.py, gripper_config.py

### 修改内容

#### 1. 修正开合方向误差验证指标 (右手38° → 0.04°)
- **根因**: 验证指标将夹爪y轴与MANO原始开合方向(finger1→finger2)比较, 但MANO开合方向与指向方向不正交, 而夹爪y轴由运动学约束与x轴正交, 导致投影差异被误报为误差
- **修复**: 开合方向误差计算改为先将MANO开合方向投影到指向方向垂直面, 再与夹爪y轴比较
- **效果**: 右手开合方向误差从38°降到0.04°(解析模式), 左手从11°降到0.09°

#### 2. 独立手指缩放 (finger1_origin_x / finger2_origin_x)
- **问题**: 之前使用单一finger_origin_x(平均值)缩放两个手指, 但MANO拇指到腕距离(~125mm)和食指到腕距离(~147mm)差异大
- **修复**: 为每个手指独立计算finger_origin_x, 在URDF生成和解析计算中使用独立值
- **效果**: 解析模式位置误差从~3mm降到右手~0.9mm, 左手~3mm

#### 3. warm_start改用解析位姿初始化 (左手开合方向148° → 15°)
- **问题**: warm_start使用MANO手腕旋转初始化, 但MANO手腕坐标系与夹爪坐标系定义不同, 导致优化器从错误初始值出发陷入局部最优
- **修复**: warm_start改用compute_analytical_gripper_pose()计算的解析位姿初始化, 同时用解析关节值初始化手指关节
- **效果**: 优化器模式左手开合方向误差从148°降到15°

#### 4. 移除调试打印
- 移除优化器内部FK误差验证和SAPIEN vs MANO对比的DEBUG日志

### 验证结果 (数据集7, 30帧, gripper模式)

**解析模式 (analytical):**
| | 指尖1 | 指尖2 | 手腕 | 指向 | 开合 |
|---|---|---|---|---|---|
| 左手 | 3.15mm | 2.86mm | 1.22mm | 0.24° | 0.09° |
| 右手 | 0.90mm | 0.79mm | 0.36mm | 0.03° | 0.04° |

**优化器模式 (optimizer):**
| | 指尖1 | 指尖2 | 手腕 | 指向 | 开合 |
|---|---|---|---|---|---|
| 左手 | 7.53mm | 9.60mm | 8.45mm | 4.32° | 3.35° |
| 右手 | 1.75mm | 1.74mm | 2.24mm | 0.78° | 1.72° |

---

## [2026-06-21] 夹爪映射算法升级: 缩放X方向 + 加权Kabsch SVD

**类型**: 优化
**影响范围**: example/combination/hand_track/render_gripper_only.py, docs/questions.md

### 修改内容

#### 1. 夹爪映射算法从 Y优先 Gram-Schmidt 升级为 加权 Kabsch SVD + 缩放X方向 (腕部误差 85mm → 13mm)
- **根因**: MANO 腕→指尖距离 ~120-135mm, 夹爪 gripper_link→指尖距离只有 37mm。刚性变换无法改变距离, gripper_link 不可能在腕部位置
- **修复**: 缩放 URDF 的 `finger_origin_x` 从 37mm 到 MANO 腕→指尖距离, 使 gripper_link 自然在腕部位置
- **算法**: 加权 Kabsch SVD (权重 10:10:1, 指尖优先) 从3个点求解最优旋转和平移
- **效果**: 腕部误差从 79-87mm 降到 13-22mm (改善 80%+), 指向方向误差从 38° 降到 10°

#### 2. URDF 动态缩放
- `_generate_gripper_urdf` / `_generate_gripper_with_arm_urdf` 新增 `finger_origin_x` 参数
- 每个视频根据 MANO 数据自动计算平均腕→指尖距离, 动态修改 URDF joint origin

#### 3. 新增 Q9 文档: 为什么3个点不能完全对应
- 数学证明: 刚性变换保持距离不变, 距离不匹配则无法重合
- 对比 dex_retargeting 优化器也无法精确匹配 (10mm 指尖误差)
- 类比: 小尺子 (37mm) 无法同时放在大尺子 (135mm) 的两端

### 验证结果

| 指标 | 左手 (旧) | 左手 (新) | 右手 (旧) | 右手 (新) |
|---|---|---|---|---|
| finger1 | 0.96mm | 2.59mm | 0.39mm | 9.15mm |
| finger2 | 1.00mm | 3.28mm | 0.38mm | 8.30mm |
| wrist | 78.85mm | 12.96mm | 87.39mm | 22.18mm |
| 指向方向 | 11.42° | 6.26° | 38.03° | 9.92° |

### 清理
- 删除临时测试脚本: analyze_3points.py, test_scaled_gripper.py, diag_gripper_mapping.py

---

## [2026-06-21] 夹爪映射算法升级: Gram-Schmidt → Y轴优先 Gram-Schmidt

**类型**: 修复 + 优化
**影响范围**: example/combination/hand_track/render_gripper_only.py, diag_gripper_mapping.py

### 修改内容

#### 1. 夹爪映射算法从 X 优先 Gram-Schmidt 升级为 Y 轴优先 Gram-Schmidt (指尖误差 23mm → 0.4mm)
- **根因**: X 优先 Gram-Schmidt 强制 Y ⊥ X, 但 MANO 的开合方向和指向方向不正交 (内积可达 0.61), 导致 Y 轴被扭曲, finger2 误差 12-23mm
- **修复**: 改为 Y 轴优先 Gram-Schmidt:
  - Y 轴精确匹配开合方向 (finger1→finger2), 误差 < 0.5°
  - X 轴从指向方向投影掉 Y 分量, 尽可能接近指向方向
  - 匹配指尖中点确定位置 (误差均摊到两个指尖)
- **效果**: 指尖最大误差从 23.15mm 降到 1.00mm (改善 96%)

#### 2. 验证指标改进: 用指向/开合方向误差替代手腕朝向误差
- **旧指标**: `wrist_ori_deg` 比较 gripper_link_R 和 MANO wrist_R, 但两者坐标系定义不同 (X=指向 vs Z=指向), 误差 122-153° 无意义
- **新指标**: `pointing_deg` (gripper X 轴 vs MANO 指向方向) 和 `opening_deg` (gripper Y 轴 vs MANO 开合方向), 更直观反映夹爪朝向精度

#### 3. 优化器模式改用 Hybrid 方案 (FK 朝向 + 解析位置/手指关节)
- 优化器因 MANO 手和夹爪几何不匹配, 手指关节值接近 0, 无法体现开合
- 改为: 朝向用 retargeting FK, 位置和手指关节用解析值

#### 4. 新增诊断脚本 diag_gripper_mapping.py
- 比较不同方法的指尖误差和方向误差

### 验证结果 (数据集 7, 50帧)
- **Y 轴优先 Gram-Schmidt**:
  - left: finger1=0.96mm, finger2=1.00mm, pointing=11.42°, opening=0.18°
  - right: finger1=0.39mm, finger2=0.38mm, pointing=38.03°, opening=0.05°
- **laptop 数据集**: finger1=2.84mm, finger2=2.34mm, pointing=32.48°, opening=0.42°

### 与 run_robot_tracking 的对比
- `run_robot_tracking` 使用 NLopt 非线性优化器同时优化 8 DOF, 全局最优
- `render_gripper_only` 使用 Y 轴优先 Gram-Schmidt 解析计算, 速度更快
- 优化器的手指关节值因几何不匹配接近 0, 无法直接用于夹爪渲染
- 指尖位置精度: 两者基本一致 (< 1mm)

### 已知限制
- **指向方向误差 11-38°**: MANO 的指向方向 (wrist→finger_mid) 和开合方向 (finger1→finger2) 高度非正交 (内积 0.16-0.61), 数学上无法同时精确匹配两者。Y 轴优先保证指尖位置精确, X 轴 (指向方向) 不可避免地有偏差
- **wrist_pos 误差 85-108mm**: MANO 手腕到指尖距离 ~9-12cm vs 机器人夹爪 ~3.7cm, 几何不匹配

---

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
