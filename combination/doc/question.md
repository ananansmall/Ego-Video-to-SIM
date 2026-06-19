# Q&A 文档

## Q1: 目前相机是静止的吗？

**不是静止的。** 相机跟随 HaWoR 的 SLAM 相机轨迹运动。

代码中有三种相机模式：

| 模式 | 相机行为 |
|------|---------|
| **hand_only (viewer)** | 每帧跟随 `hawor_cam_to_sapien_pose(R_c2w_all[global_idx], t_c2w_all[global_idx])` |
| **robot_tracking** | 每帧跟随 HaWoR 相机轨迹 |
| **robot_only** | 每帧跟随 HaWoR 相机轨迹 |

如果 `R_c2w_all` 和 `t_c2w_all` 为 None（没有相机轨迹数据），则回退到固定视角。

**结论**：当有 HaWoR 相机轨迹时，相机是动态的，逐帧跟随 SLAM 估计的相机位姿。这模拟了第一人称视角。

---

## Q2: 1.7° 的朝向容差还不够吗？

**不够。** 1.7° 看似很小，但对机械臂末端来说影响很大：

| 容差 | 角度 | 在0.5m臂长下的末端偏差 | 效果 |
|------|------|----------------------|------|
| 0.03 rad | 1.72° | ~15mm (1.5cm) | 几乎无朝向约束 |
| 0.01 rad | 0.57° | ~5mm | 轻度约束 |
| 0.005 rad | 0.29° | ~2.5mm | 严格约束 |
| **0.002 rad** | **0.11°** | **~1mm** | **当前设置** |
| 0.001 rad | 0.057° | ~0.5mm | 非常严格，可能不可达 |

当前已收紧到 0.002 rad (0.11°)。

---

## Q3: 这两个文件夹还有什么可以展示融合的？

RAS (ReplicateAnyScene) 和 HaWoR 的数据可以融合的方向：

### 已在管线中使用的融合

| 融合数据 | RAS侧 | HaWoR侧 | 用途 |
|----------|--------|---------|------|
| 相机位姿 | `extrinsics/*.txt` | `R_c2w`/`t_c2w` | 坐标系对齐 |
| 3D场景 | `final_scene.glb` | 手部`pred_trans` | 场景+手部共渲染 |
| 相机内参 | `intrinsic.txt` | `img_focal`/`est_focal.txt` | 渲染FOV |

### 尚未充分利用的融合方向

| 方向 | RAS数据 | HaWoR数据 | 价值 |
|------|---------|-----------|------|
| **深度图交叉验证** | `depth/*.png` (VGGT) | SLAM `disps` (Metric3D) | 两种深度估计互补，提升精度 |
| **手部去除重建** | `depth/*.png` | `model_masks.npy` (手部掩码) | 从深度图中减去手部，提升场景重建质量 |
| **操作物体标注** | `final_scene.glb` (物体网格) | `pred_trans` (手部轨迹) | 自动标注手操作了哪个物体 |
| **手-物接触检测** | `point_cloud.ply` | `pred_trans` | 手部到场景最近点距离 → 接触判定 |
| **碰撞仿真** | `final_scene.glb` | MANO完整参数 | 手-物交互物理仿真 |
| **尺度自校正** | `point_cloud.ply` | SLAM `scale` | 用SLAM米制尺度校正VGGT尺度偏差 |

### 最有价值的下一步融合

1. **手-物接触检测**：用 RAS 点云和 HaWoR 手部3D位置，计算最近点距离，判断手是否接触物体
2. **操作物体标注**：根据手部轨迹和场景物体边界框，自动标注被操作的物体类别
3. **深度图融合**：VGGT 深度在非手部区域更准，Metric3D 深度在手部区域更准，可以融合

---

## Q4: 夹爪映射只显示4、8两点

**已修改。** `_render_keypoints()` 现在只显示 `ref_indices` 中的关节（4=食指尖, 8=中指尖），移除了其他21个关节的显示。

关节索引对照：
- 0: wrist
- 1-4: index (MCP, PIP, DIP, **tip**)
- 5-8: middle (MCP, PIP, DIP, **tip**)
- 9-12: ring
- 13-16: pinky
- 17-20: thumb

---

## Q5: RelaxedIK为什么姿态误差那么大？

**根本原因**：RelaxedIK是增量式求解器，每次`solve_position`只做一步梯度下降优化。

### 详细分析

1. **单步优化**：每次调用只做一步，位置梯度量级大于姿态梯度，位置先收敛但姿态还差很远
2. **内部状态不同步**：求解器维护内部"当前关节位置"状态，如果与仿真实际关节位置不一致，优化起点偏移
3. **6自由度非冗余臂**：R1 Lite只有6个关节，没有额外自由度调节姿态
4. **容差设置**：旋转容差太松时，求解器认为姿态"足够好"就停止优化

### 修复措施

| 措施 | 修改 | 效果 |
|------|------|------|
| 增加求解次数 | 5→20次/帧 | 充分收敛 |
| 收紧旋转容差 | 0.005→0.002 rad | 增加姿态优化压力 |
| 每帧reset求解器 | `ik_solver.relaxed_ik_right.reset(list(arm_joints))` | 同步内部状态 |

---

## Q6: 机械臂底座不要固定

**已修改。** 底座现在根据手腕位置做小范围跟踪。

### 实现方式

```python
# 新增方法 _compute_tracking_base_pos()
# 1. 计算手腕在base_link坐标系中的位置
# 2. 计算与舒适目标的偏移
# 3. 限制偏移在 ±BASE_TRACKING_RANGE (0.08m) 范围内
# 4. 转回世界坐标系作为新底座位置
```

### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `BASE_TRACKING_RANGE` | 0.08m | 底座跟踪范围（±8cm） |
| `BASE_TRACKING_ALPHA` | 0.15 | 平滑系数（预留） |
| `COMFORT_TARGET_IN_BASE` | [0.30, 0.0, -0.30] | 手腕在base_link中的舒适位置 |

---

## Q7: changelog 合并

已合并为一个 `CHANGELOG.md`，删除了重复的小写 `changelog.md`。

---

## Q8: 坐标系对齐整合

见 `alignment.md`，已整合了之前犯的错误和正确方案。

---

## Q9: 蓝色相机柱子已去除

视频中相机位置的蓝色柱状标记已移除，所有 `_render_camera_axes()` 调用已删除（4处）。相机位置更新逻辑保留。

---

## Q10: 夹爪映射方法重构

### 旧方法 vs 新方法

| 对比项 | 旧方法 | 新方法 |
|--------|--------|--------|
| 约束点 | 2个 [4,8] | **3个 [4,8,0]** |
| 朝向来源 | 手腕旋转直接计算 | **FK提取** |
| IK目标位置 | 手腕位置 | **FK夹爪位置** |
| normal_delta | 4e-3 | **1e-5** |
| IK容差 | [0.001,...,0.002,...] | **[0.1,...,0.1,...]** |
| warm_start | 无 | **有** |

### 3约束点为什么能约束朝向？

2约束点只有6个位置约束，但优化变量有7DOF（6个dummy+1个gripper），欠定1DOF——恰好是夹爪接近轴旋转。添加手腕→gripper_link约束后，9约束 vs 7DOF，超定2约束，手腕位置相对指尖方向编码了朝向信息。

### IK容差为什么用0.1而不是0.001？

容差是RelaxedIK目标函数的权重分母：`L = Σ(err_i / tol_i)²`
- tol=0.001 → 权重=10⁶，位置主导，朝向被拉偏
- tol=0.1 → 权重=100，位置和朝向平衡

---

## Q11: output文件夹是否需要删除？

| 文件夹 | 建议 | 原因 |
|--------|------|------|
| `combination/output/` | **保留** | 管线正式输出 |
| `combination/test/` | 可删除 | 调试脚本和旧视频 |
| `position_retargeting/pv_retargeting/` | **保留** | 参考实现输出 |
| `position_retargeting/test/` | 可删除 | IK测试视频 |
| `simulation/` | 可删除 | 仿真测试视频 |
| `video_egocentric_retargeting/output_*/` | 可删除 | 旧管线输出 |
| `output_pin_base_frame/` (项目根) | 可删除 | 旧测试输出 |

---

## Q12: 为什么移除了 `_render_camera_axes()` 和 `_render_pose_axes()`？

这两个方法在之前的修改中被移除调用，但方法定义还残留。第五轮清理时彻底删除了。

**移除原因**：
- `_render_camera_axes()`: 之前在相机位置渲染蓝色柱状标记 + 坐标轴。用户反馈视频中蓝色柱子影响观感，已在第四轮修改中移除所有调用。
- `_render_pose_axes()`: 之前在手腕/末端执行器位置渲染 XYZ 坐标轴标识。用户反馈位姿标识影响观感且与最终展示无关，已在第四轮修改中移除所有调用。

**保留的计算，移除的只是可视化**：相机位姿计算、手腕旋转计算、末端执行器位姿计算都完整保留，只是不再渲染视觉标记。

---

## Q13: 帧对应和 GLB 对齐有什么关系？

**关键认识：GLB 是静态的，对齐参数一旦确定就不依赖帧对应。**

帧对应只影响 **Umeyama 尺度估算**这一个环节：

| 环节 | 是否依赖帧对应 | 原因 |
|------|---------------|------|
| R_align 计算 | ❌ 不依赖 | 只用第一帧相机位姿 |
| t_align 计算 | ❌ 不依赖 | 只用第一帧相机位姿 |
| **s_inv 尺度** | ✅ 依赖 | 用全部相机轨迹计算离散度比 |
| GLB 顶点变换 | ❌ 不依赖 | 用 R_inv + t_inv + s_inv 一次性变换 |
| 渲染时相机跟随 | ❌ 不依赖 | 各自用各自的相机轨迹 |

所以帧对应只影响 s_inv 的精度。如果帧映射错误，s_inv 可能偏差，导致 GLB 整体偏大或偏小。但 R_align 和 t_align 不受影响。

**当前帧映射方式**：线性插值 `hi = round(ri * (n_hawor-1) / (n_ras-1))`，假设 RAS 和 HaWoR 处理的是同一段视频的相同帧范围。如果两端帧数不同但确实是同一段视频，这个映射是合理的。

---

## Q14: R_c2w 是怎么得来的？

R_c2w 是 camera-to-world 旋转矩阵，表示从相机坐标系到世界坐标系的旋转。

**RAS 侧**：
```
RAS 外参文件 (extrinsics/*.txt) 存储的是 world-to-camera 变换 [R_w2c | t_w2c] (3x4 矩阵)
R_c2w = R_w2c.T  (转置即逆，因为旋转矩阵正交)
cam_pos = -R_c2w @ t_w2c  (相机在世界坐标系中的位置)
```

**HaWoR 侧**：
```
HaWoR SLAM (DROID-SLAM) 输出相机轨迹 traj (N, 7): [tx, ty, tz, qx, qy, qz, qw]
这是 camera-to-world 位姿，经 R_x = diag(1,-1,-1) 变换后存储在 hawor_results_*.npz 中
R_c2w 直接从 npz 文件读取，已在 Render World (Y-UP) 坐标系中
```

**两者约定差异**：
- RAS: OpenCV 约定 (X=right, Y=down, Z=forward)
- HaWoR: OpenGL 约定 (X=right, Y=up, Z=backward)
- 转换: `OPENCV_TO_OPENGL = diag(1, -1, -1)`

---

## Q15: GLB 对齐潜在问题分析

### 🔴 高风险

**1. 静态相机时 Umeyama 尺度退化**

Umeyama 通过相机轨迹的离散度 (sigma) 计算尺度比。静态相机 sigma→0，尺度比不稳定。

**已修复**: 自动检测静态相机 (sigma < 0.01)，回退到基于手-GLB 距离的启发式估算。

**2. GLB 场景图变换被忽略**

trimesh `geometry.items()` 返回几何体局部坐标，不包含场景图中的变换节点。如果 RAS 导出的 GLB 使用了场景图变换（平移/旋转/缩放），这些变换会被忽略。

**影响**: 如果 GLB 有场景图变换，顶点位置整体偏移。当前数据未发现此问题。

### 🟡 中等风险

**3. R_c2w 可能不正交**

从外参 txt 读取的 R_w2c 经数值计算后可能不是严格正交矩阵，R_c2w = R_w2c.T 也不正交，R_align 会累积误差。

**已缓解**: 添加了行列式和正交性检查。

**4. 外参文件名排序假设**

假设文件名是纯数字 (如 `0.txt`)。如果 RAS 输出含前缀 (如 `frame_0.txt`)，`int()` 解析会失败。

### 🟢 低风险

**5. 临时 PLY 文件名冲突**: 同一进程多次调用时 geom_name 相同可能覆盖。当前管线只调用一次。

**6. 颜色丢失**: 只取平均顶点颜色作为整体材质，原始逐顶点颜色信息在 SAPIEN 渲染时丢失。

**7. 验证步骤坐标系混用**: pred_trans 在 SLAM World (Y-down)，glb_hawor 在 Render World (Y-up)，距离计算差 R_x 变换，但绝对值影响不大。

---

## Q16: 管线中有多少种坐标系？它们之间的关系是什么？

### 总览：7 种坐标系

| # | 名称 | UP轴 | 来源 | 有关节？ | 用途 |
|---|------|------|------|---------|------|
| 1 | RAS World (Z-UP) | Z | VGGT 场景重建 | ❌ | RAS 外参和 GLB 的原始坐标系 |
| 2 | RAS Y-UP | Y | Z-UP 经 ZUP_TO_YUP | ❌ | 对齐时与 HaWoR 统一 UP 轴 |
| 3 | HaWoR SLAM World (Y-down) | -Y | DROID-SLAM | ❌ | pred_trans/pred_rot 的存储坐标系 |
| 4 | HaWoR Render World (Y-UP) | Y | SLAM World 经 R_x | ❌ | MANO 输出、相机轨迹的坐标系 |
| 5 | SAPIEN World (Z-UP) | Z | Render World 经 RXWORLD_TO_SAPIEN | ❌ | 最终渲染坐标系 |
| 6 | 机器人 Base Link | Z | SAPIEN World 的子坐标系 | ✅ | RelaxedIK 求解坐标系 |
| 7 | 相机坐标系 | - | 依附于 SAPIEN World | ❌ | 渲染视角 |

### 变换链总图

```
                    ┌─────────────────────┐
                    │  RAS World (Z-UP)   │ ← VGGT 外参 + GLB 顶点
                    │  X=right, Y=fwd, Z=up│
                    └──────────┬──────────┘
                               │ ZUP_TO_YUP
                               ▼
                    ┌─────────────────────┐
                    │  RAS Y-UP           │ ← 对齐中间态
                    │  X=right, Y=up, Z=bwd│
                    └──────────┬──────────┘
                               │ s_inv * R_inv @ p + t_inv  (01_align_scene)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  HaWoR Render World (Y-UP)                                  │
│  X=right, Y=up, Z=backward (OpenGL)                         │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────┐              │
│  │ SLAM World       │    │ MANO 输出         │              │
│  │ (Y-down, OpenCV) │    │ 顶点 + 21 关节    │              │
│  │ pred_trans/rot   │    │ R_c2w / t_c2w     │              │
│  └────────┬─────────┘    └────────┬──────────┘              │
│           │ R_x=diag(1,-1,-1)     │                          │
│           └──────────┬────────────┘                          │
│                      ▼                                       │
│           都在 Render World 中                               │
└──────────────────────┬──────────────────────────────────────┘
                       │ RXWORLD_TO_SAPIEN = R_AXIS @ R_x
                       │ = [[1,0,0],[0,0,-1],[0,1,0]]
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SAPIEN World (Z-UP)                                        │
│  X=right, Y=forward, Z=up                                   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ MANO 手部     │  │ GLB 物体      │  │ 相机 (第一人称)   │  │
│  │ 顶点+关节     │  │ 变换后顶点    │  │ SAPIEN 相机约定   │  │
│  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
│         │                                                    │
│         │ Dex Retargeting (3约束点: 拇指尖/食指尖/手腕)      │
│         │ → 夹爪 qpos (7 DOF: 6 dummy + 1 gripper)         │
│         │ → FK → gripper_pos + gripper_R (SAPIEN World)     │
│         │                                                    │
│         │ + mapping_offset + safety_offset                   │
│         │ → ik_target_world (SAPIEN World)                   │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────┐               │
│  │ 机器人 Base Link 坐标系 (Z-UP, Z旋转180°) │ ← 唯一有关节  │
│  │                                          │               │
│  │ ik_target_b = base_R⁻¹ @ (target - base) │               │
│  │ ee_R_base   = base_R⁻¹ @ R_ee_world      │               │
│  │                                          │               │
│  │ RelaxedIK 求解 → 6个arm关节角             │               │
│  └──────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

### 关键变换矩阵

| 变换 | 矩阵 | 行列式 | 作用 |
|------|------|--------|------|
| `R_x` | `diag(1,-1,-1)` | 1 | SLAM World (Y-down) → Render World (Y-up) |
| `R_AXIS` | `[[1,0,0],[0,0,1],[0,-1,0]]` | 1 | Y-UP → Z-UP |
| `RXWORLD_TO_SAPIEN` | `[[1,0,0],[0,0,-1],[0,1,0]]` | 1 | Render World → SAPIEN |
| `ZUP_TO_YUP` | `[[1,0,0],[0,0,1],[0,-1,0]]` | 1 | RAS Z-UP → Y-UP |
| `OPENCV_TO_OPENGL` | `diag(1,-1,-1)` | 1 | OpenCV 相机 → OpenGL 相机 |
| `OPERATOR2MANO_RIGHT` | `[[0,0,-1],[-1,0,0],[0,1,0]]` | 1 | 操作员手 → MANO 手 (右手) |

所有变换矩阵的行列式都为 1，都是合法旋转（无镜像）。

---

## Q17: Dex Retargeting 到底起了什么作用？举个例子

### 一句话概括

**Dex Retargeting 解决的问题是：已知人手 3 个关键点的 3D 位置，求机器人夹爪应该怎么张开、朝向哪里，才能让夹爪的对应位置最接近人手的这 3 个点。**

### 具体例子

假设某一帧，MANO 输出了 21 个手部关节在 SAPIEN 坐标系中的 3D 位置：

```
关节 0 (手腕):     [0.30, -0.05, 0.15]
关节 4 (拇指尖):   [0.35, -0.08, 0.12]
关节 8 (食指尖):   [0.38, -0.02, 0.13]
... 其余 18 个关节 ...
```

我们只取 3 个关键点（`ref_indices = [4, 8, 0]`）：

```
ref_value = [
    [0.35, -0.08, 0.12],   ← 拇指尖位置
    [0.38, -0.02, 0.13],   ← 食指尖位置
    [0.30, -0.05, 0.15],   ← 手腕位置
]
```

### 优化器内部做了什么

优化器有 **7 个优化变量**（自由度）：

| 变量 | 含义 | 范围 |
|------|------|------|
| `dummy_x_translation` | 夹爪整体 X 平移 | [-5, 5] m |
| `dummy_y_translation` | 夹爪整体 Y 平移 | [-5, 5] m |
| `dummy_z_translation` | 夹爪整体 Z 平移 | [-5, 5] m |
| `dummy_x_rotation` | 夹爪整体绕 X 旋转 | [-2π, 2π] rad |
| `dummy_y_rotation` | 夹爪整体绕 Y 旋转 | [-2π, 2π] rad |
| `dummy_z_rotation` | 夹爪整体绕 Z 旋转 | [-2π, 2π] rad |
| `right_gripper_finger_joint1` | 夹爪手指1 开合 | [0, 0.05] m |

优化器通过 **NLopt SLSQP** 求解：

```
minimize:  huber_loss(FK(q)[finger_link1], 拇指尖位置)
         + huber_loss(FK(q)[finger_link2], 食指尖位置)
         + huber_loss(FK(q)[gripper_link],  手腕位置)
         + λ * ||q - q_last||²              (平滑正则化)
```

### 优化器输出怎么用？

**第 1 部分：夹爪位姿（6 个 dummy 变量）→ 提取出来给 RelaxedIK**

```python
gripper_pos_fk, R_ee_world_fk = _get_gripper_pose_from_retargeting(retargeting, retarget_qpos)
ik_target = gripper_pos_fk + mapping_offset
ik_joints = ik_solver.solve_position_right(ik_target_b, ee_quat_b)
```

**第 2 部分：夹爪开合（1 个 joint 变量）→ 直接设置到 SAPIEN 机器人**

```python
sapien_qpos = retarget_qpos[retarget2sapien]
qpos[gripper_idx1] = float(sapien_qpos[gripper_idx1])
qpos[gripper_idx2] = float(sapien_qpos[gripper_idx2])
```

### 完整数据流图

```
人手 (MANO 21 关节, SAPIEN 坐标系)
    │
    │ 取 3 个关键点: [4=拇指尖, 8=食指尖, 0=手腕]
    ▼
ref_value (3×3 矩阵)
    │
    │ Dex Retargeting 优化器
    │   7 个变量: [dummy_x, dummy_y, dummy_z, dummy_rx, dummy_ry, dummy_rz, finger_joint1]
    │   目标: min huber_loss(FK(q)[3个link], ref_value[3个点])
    ▼
retarget_qpos (7 维)
    │
    ├──→ FK 提取夹爪位姿 ──→ RelaxedIK ──→ 6 个 arm 关节角 (肩/肘/腕)
    │    (3 位置 + 3 旋转)        ↑
    │                          "臂到达夹爪"
    │
    └──→ finger_joint1 ──→ SAPIEN 夹爪开合
         (1 个标量)          "夹爪张开多少"
```

### 和 RelaxedIK 的区别

| | Dex Retargeting | RelaxedIK |
|---|---|---|
| **输入** | 3 个人手关键点位置 | 1 个末端位姿 (位置+朝向) |
| **输出** | 夹爪开合 + 夹爪位姿 | 6 个臂关节角 |
| **优化变量** | 7 (6 dummy + 1 gripper) | 6 (6 arm joints) |
| **求解什么** | "夹爪怎么张开才能匹配人手" | "臂怎么弯才能到达夹爪位置" |

简单说：**Dex Retargeting 管"手→夹爪"，RelaxedIK 管"夹爪→臂"**。两者串联完成"人手→机器人"的映射。

---

## Q18: 为什么不能直接给定位置？为什么要优化？6+1 和 26 是什么？

### 为什么直接给定行不通？

**问题是：夹爪是一个刚体结构，3 个 link 的位置不能独立设置。**

假设人手某一帧的 3 个关键点位置：

```
手腕 (joint 0):  [0.30, -0.05, 0.15]
拇指尖 (joint 4): [0.35, -0.08, 0.12]
食指尖 (joint 8): [0.38, -0.02, 0.13]
```

但夹爪的物理结构是：

```
gripper_link (底座)
    ├── finger_link1 (沿 Y 轴滑动, 范围 0~5cm)
    │     关闭时相对底座: [0.037, -0.013, 0.000]
    │     打开时相对底座: [0.037, -0.063, 0.000]
    └── finger_link2 (沿 Y 轴滑动, 范围 0~5cm)
          关闭时相对底座: [0.037, +0.013, 0.000]
          打开时相对底座: [0.037, +0.063, 0.000]
```

**结论：无论你怎么旋转和打开夹爪，都不可能同时让 3 个 link 精确到达 3 个人手关键点的位置。** 因为夹爪的几何形状和人手的几何形状不同。

### 那优化做了什么？

优化不是"精确匹配"，而是"**尽可能接近**"：

```
目标: 找到 [底座位置, 底座朝向, 手指开合度]
使得:  finger_link1 尽量靠近 拇指尖
      finger_link2 尽量靠近 食指尖
      gripper_link  尽量靠近 手腕
```

### 6+1 是什么？

**8 个优化变量**（6 dummy + 2 gripper finger）：

| 变量 | 含义 | 为什么需要 |
|------|------|-----------|
| dummy_x/y/z_translation | 夹爪底座在空间中的位置 | 对应"圆规轴放在哪" |
| dummy_x/y/z_rotation | 夹爪底座的朝向 | 对应"圆规朝哪个方向" |
| finger_joint1/2 | 手指开合度 | 对应"两脚张开多少" |

6 个 dummy 就是"圆规的轴"——它不是物理关节，只是优化变量，让优化器可以自由调整夹爪在 3D 空间中的位姿。

### 26 是什么？

**26 = Dex Retargeting 内部 pinocchio 模型的总 DOF**（完整 R1 机器人）：

| 关节 | 数量 | 状态 |
|------|------|------|
| dummy (虚拟底座) | 6 | 优化 |
| torso (躯干) | 4 | 固定为0 |
| left_arm (左臂) | 6 | 固定为0 |
| left_gripper (左夹爪) | 2 | 固定为0 |
| right_arm (右臂) | 6 | 固定为0 |
| right_gripper (右夹爪) | 2 | 优化 |

### 三层 qpos 的关系

```
Dex Retargeting 内部 (26 维)
    │ idx_pin2target = [0,1,2,3,4,5,24,25]  → 8个优化变量
    │ idx_pin2fixed  = [6,7,...,23]           → 18个固定为0
    ▼
retarget_qpos (8 维): [dummy_x, ..., dummy_rz, finger1, finger2]
    │
    ├──→ FK 提取夹爪位姿 → RelaxedIK → 6 个 arm 关节角
    │
    └──→ finger_joint1/2 → 直接设置 SAPIEN 夹爪开合

最终 SAPIEN qpos = [arm_joint1~6, finger1, finger2]  (8维)
```

---

## Q19: 2D重投影误差的标准是什么？怎么看待这些数值？

### 什么是 2D 重投影误差？

2D 重投影误差 = 将 3D 手部关节投影到 2D 图像平面后，与 GT（真值）2D 关节位置的像素距离。

```
3D 关节 (世界坐标系) → 相机投影 → 2D 像素坐标 (u, v)
                                         ↓
                               与 GT 2D 坐标比较 → 像素距离 (px)
```

### 误差等级和含义

| 误差范围 | 等级 | 含义 | 对应3D偏差 (在0.3m深度) |
|----------|------|------|----------------------|
| **< 2 px** | 优秀 | 几乎完美对齐，视觉上无法区分 | < 1mm |
| **2-5 px** | 良好 | 轻微偏差，整体一致 | 1-2.5mm |
| **5-15 px** | 一般 | 可见偏移，但动作趋势正确 | 2.5-7.5mm |
| **15-30 px** | 较差 | 明显偏移，手和物体位置不准 | 7.5-15mm |
| **> 30 px** | 失败 | 严重偏移，对齐无效 | > 15mm |

> 注：3D偏差 = error_px × depth_m / focal_px。以 focal=600, depth=0.3m 为例：1px ≈ 0.5mm

### 为什么用像素而不是毫米？

1. **GT 来自 2D**：cam_space 的 MANO 参数投影到 2D 后，与图像像素对齐。GT 本身就是像素精度
2. **与视频对齐**：我们关心的是"仿真渲染与视频看起来是否一致"，这是像素级的问题
3. **深度无关**：同一3D偏差，近处物体像素误差大，远处小。用像素更直观

### 当前管线的误差

| 指标 | 修复前 | 修复后 | 等级 |
|------|--------|--------|------|
| 平均误差 | 32.02 px | **0.50 px** | 优秀 |
| 中位误差 | 29.43 px | **0.38 px** | 优秀 |
| 手腕平均 | 36.68 px | **0.51 px** | 优秀 |
| 指尖平均 | 31.56 px | **0.49 px** | 优秀 |

修复后误差 < 1px，说明 HaWoR 自身的 world→cam 变换链是完全正确的。

### 但这验证的是什么？

**重要**：这个误差验证的是 **HaWoR 内部一致性**（cam_space GT vs world→cam 重投影），不是仿真与视频的对齐。真正的"仿真与视频对齐"需要：

1. **仿真渲染帧 vs 原始视频帧** — 像素级对比（需要 05_video_alignment.py 的 overlay 模式）
2. **仿真3D物体 vs 视频2D投影** — 物体位姿是否正确（需要物体级别的重投影验证）
3. **仿真物理交互 vs 视频交互** — 抓取时机、力度是否正确（需要时序对齐）

---

## Q20: 用 MASt3R 和 SAM3 可以吗？主要调整哪方面的位姿？

### 可以，而且非常推荐

MASt3R (Matching and Stereo 3D Reconstruction) 和 SAM3 (Segment Anything Model 3) 是目前最强的 2D-3D 对齐工具组合。

### 调整的是什么？——三者都需要

| 调整对象 | 当前问题 | MASt3R/SAM3 如何帮助 |
|----------|---------|---------------------|
| **物体位姿** | GLB 物体经 R_inv/t_inv/s_inv 变换后位置可能偏移 | SAM3 分割视频中的物体 → MASt3R 建立 2D-3D 对应 → 优化物体 6DoF 位姿 |
| **手部位姿** | HaWoR 手部重建有累积漂移 | SAM3 分割手部 → MASt3R 建立手-场景对应 → 优化手部轨迹 |
| **交互轨迹** | 手-物接触时序可能不准 | SAM3 追踪接触帧 → MASt3R 验证接触几何 → 修正交互时序 |

### 具体方案

#### 方案 A：物体位姿优化（最直接、最有价值）

```
输入: 原始视频帧 + GLB 3D 物体 + 初始位姿
    ↓
SAM3: 分割每帧中的物体 mask
    ↓
MASt3R: 建立视频帧间的 2D-2D 对应 + 3D 点云
    ↓
渲染: 将 GLB 物体按当前位姿渲染到每帧 → 得到仿真 mask
    ↓
优化: min Σ (SAM3_mask - rendered_mask) + Σ (MASt3R_2D - projected_3D)²
    ↓
输出: 优化后的物体 6DoF 位姿序列
```

**调整量**：每个物体 6 个参数 (3 平移 + 3 旋转)，通常只优化平移即可

#### 方案 B：手-物交互轨迹修正

```
输入: HaWoR 手部轨迹 + 优化后物体位姿
    ↓
SAM3: 检测手-物接触帧 (手部 mask 与物体 mask 重叠)
    ↓
MASt3R: 在接触帧建立手-物 3D 对应
    ↓
约束: 接触时手和物体必须 3D 距离 < 阈值
    ↓
优化: 修正手部轨迹使接触约束满足
    ↓
输出: 修正后的手部轨迹 + 交互时序标注
```

**调整量**：手部轨迹的全局偏移 + 每帧微调

#### 方案 C：端到端场景优化（最完整但最复杂）

```
输入: 视频帧 + GLB 场景 + HaWoR 手部
    ↓
SAM3: 全场景语义分割 (手/物体/背景)
    ↓
MASt3R: 稠密 3D 重建 + 2D 对应
    ↓
可微渲染: 渲染仿真场景 → 与视频帧比较
    ↓
优化: 场景参数 (物体位姿 + 相机参数 + 手部轨迹)
    ↓
输出: 完整优化后的仿真场景
```

### 推荐优先级

| 优先级 | 方案 | 原因 |
|--------|------|------|
| **1** | 物体位姿优化 | 效果最明显，物体位置偏移是当前最大问题 |
| **2** | 手-物交互修正 | 抓取时序对物理仿真至关重要 |
| **3** | 端到端优化 | 精度最高但实现复杂，需要可微渲染 |

### MASt3R vs 其他方法对比

| 方法 | 2D-3D 对应 | 稠密重建 | 需要训练 | 精度 | 速度 |
|------|-----------|---------|---------|------|------|
| **MASt3R** | ✅ 稠密 | ✅ | ✅ (预训练) | 高 | 中 |
| DUSt3R | ✅ 稀疏 | ✅ | ✅ | 中 | 快 |
| LoFTR | ✅ 稀疏 | ❌ | ✅ | 中 | 快 |
| SuperPoint+SuperGlue | ✅ 稀疏 | ❌ | ✅ | 中 | 快 |
| 可微渲染 (nvdiffrast) | ✅ 像素级 | ❌ | ❌ | 最高 | 慢 |

### 实现建议

如果本地算力有限，推荐分步实现：

1. **第一步**：用 SAM3 做物体分割（轻量，可本地运行）
2. **第二步**：用分割 mask 做 IoU 优化（不需要 MASt3R，只需渲染 mask 比较）
3. **第三步**：如果精度不够，再引入 MASt3R 做稠密对应

---

## Q21: 05_video_alignment.py 修复了什么坐标系 Bug？

### Bug 描述

`world_to_cam_hawor()` 函数的坐标系变换顺序错误。

### 错误的变换（修复前）

```python
# 错误: 先做 w2c，再翻转
points_cam = (R_w2c @ points_world.T).T + t_w2c
points_cam_flipped = R_x @ points_cam.T  # R_x 在 w2c 之后
```

手腕 2D 误差: **18.1 px**

### 正确的变换（修复后）

```python
# 正确: 先翻转 world 到 SLAM 坐标系，再做 w2c
points_world_slam = (R_x @ points_world.T).T  # R_x 在 w2c 之前
points_cam = (R_w2c @ points_world_slam.T).T + t_w2c
```

手腕 2D 误差: **0.2 px**

### 原因

HaWoR 的 `pred_trans` 在 **Render World (Y-UP)** 坐标系中（已经过 R_x 翻转），而 `R_c2w/t_c2w` 描述的是 **SLAM World (Y-down)** 中的相机位姿。所以需要先把点从 Render World 翻转回 SLAM World，再做 world-to-camera 变换。

```
pred_trans (Render World, Y-UP)
    ↓ R_x = diag(1,-1,-1)  ← 先翻转回 SLAM 坐标系
pred_trans_slam (SLAM World, Y-down)
    ↓ R_w2c @ p + t_w2c    ← 再做标准 w2c
p_cam (Camera Space)
```

### 修复结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 平均重投影误差 | 32.02 px | **0.50 px** |
| 中位误差 | 29.43 px | **0.38 px** |
| 最大误差 | 76.09 px | **1.34 px** |
| 手腕平均 | 36.68 px | **0.51 px** |
| 指尖平均 | 31.56 px | **0.49 px** |

---

## Q22: 有了轨迹还是没有真实的交互，物理仿真到底实现了什么？

### 问题本质

用户说的对：**有了轨迹 ≠ 有了真实交互**。当前 04_physics_simulation.py 的物理仿真实际上是 **"轨迹驱动的物理"**，而不是 **"交互驱动的物理"**。

### 当前实现 vs 真实交互

| 对比项 | 当前实现 | 真实交互 |
|--------|---------|---------|
| 机器人控制 | 跟随 HaWoR 轨迹 (PD 控制器追踪目标位置) | 根据接触力反馈调整动作 |
| 夹爪开合 | 从 Dex Retargeting 直接映射 | 检测到接触后闭合，力控保持 |
| 物体运动 | 被动响应碰撞 (被推动/被重力拉下) | 主动被夹爪抓取、提起、放置 |
| 抓取成功率 | 低 (轨迹不一定让夹爪碰到物体) | 高 (闭环控制确保接触) |
| 失败模式 | 夹爪穿过物体 / 物体滑落 / 抓空 | 力反馈检测到失败后重试 |

### 当前物理仿真实现了什么？

1. **碰撞检测**: 夹爪和物体之间有真实的碰撞响应，不会穿透
2. **重力**: 物体会受重力影响下落，有地面支撑
3. **摩擦力**: 夹爪手指的高摩擦材质 (static=1.0, dynamic=1.0) 提供了摩擦抓取的**可能性**
4. **被动力补偿**: `compute_passive_force(gravity=True, coriolis_and_centrifugal=True)` 补偿重力和科氏力

### 当前物理仿真没有实现什么？

1. **接触感知闭环**: 不知道"是否真的碰到了物体"，只是盲目跟随轨迹
2. **力控抓取**: 夹爪开合度来自人手映射，不是根据物体大小和接触力动态调整
3. **抓取策略**: 没有接近→接触→闭合→提起的抓取策略，只有"跟随人手开合"
4. **失败恢复**: 如果抓空了，不知道重试

### 如何实现真正的交互？

#### 方案 A: 接触感知的力控抓取 (推荐，改动最小)

```
当前: retargeting → gripper_opening → set_drive_target
改进: retargeting → gripper_opening → 接触检测 → 力控闭合

具体:
1. 每帧检测夹爪与物体的接触力 (_fetch_contacts 已有)
2. 当接触力 > 阈值 且 人手在"闭合"状态 → 切换到力控模式
3. 力控模式: 夹爪持续闭合直到接触力达到目标值
4. 当人手"张开" → 释放物体 (切换回位置控制)
```

#### 方案 B: 基于视频的抓取时序标注

```
1. 用 SAM3 分割视频中的手和物体
2. 检测手-物接触帧 (mask 重叠)
3. 标注: 接近帧 / 接触帧 / 抓取帧 / 提起帧 / 放置帧
4. 在物理仿真中根据标注切换控制策略
```

#### 方案 C: 端到端强化学习 (最完整但最复杂)

```
1. 用当前管线生成初始轨迹
2. 在物理仿真中用 RL 优化抓取策略
3. 奖励: 抓取成功 + 物体到达目标位置 + 轨迹平滑
4. 训练后机器人能自主完成抓取，不依赖人手轨迹
```

### 推荐实现路径

| 阶段 | 改动 | 效果 |
|------|------|------|
| **1** | 接触检测 + 力控闭合 | 夹爪碰到物体后自动闭合，不再抓空 |
| **2** | 视频抓取时序标注 | 知道什么时候该抓、什么时候该放 |
| **3** | 物体位姿优化 (MASt3R/SAM3) | 物体位置准确，夹爪能碰到 |
| **4** | RL 微调 | 机器人自主完成抓取任务 |

---

## Q23: --transform-params 参数是做什么的？

### 一句话概括

**`--transform-params` 指定的是 01_align_scene.py 输出的坐标系对齐参数文件，它告诉后续脚本如何把 RAS 重建的 GLB 物体放到 HaWoR 手部所在的坐标系中。**

### 为什么需要这个参数？

HaWoR 和 RAS 是两个独立的重建系统，它们各自有独立的世界坐标系：

| 数据源 | 坐标系 | 原点 | 朝向 | 尺度 |
|--------|--------|------|------|------|
| HaWoR | SLAM World (Y-down) → Render World (Y-up) | SLAM 初始位置 | SLAM 估计 | Metric3D 深度 (米制) |
| RAS | Z-UP → Y-UP | VGGT 场景中心 | 场景对齐 | VGGT 深度 (非米制) |

如果不做对齐，直接把 GLB 物体和 HaWoR 手部放到同一个场景中，会出现：

```
❌ 没有对齐时:
  - 物体在 (5.2, -3.1, 0.8)，手在 (0.01, -0.02, 0.15) → 完全不在同一位置
  - 物体朝向错误 → 桌子可能倒着放
  - 物体太大或太小 → RAS 尺度可能是 HaWoR 的 3 倍
```

### transform_params.npz 包含什么？

| 参数 | 形状 | 含义 |
|------|------|------|
| `s_inv` | 标量 | RAS → HaWoR 的缩放因子 (通常 ~0.32，即 RAS 比 HaWoR 大约 3 倍) |
| `R_inv` | (3,3) | GLB Y-UP → HaWoR 的旋转矩阵 |
| `t_inv` | (3,) | GLB Y-UP → HaWoR 的平移向量 |

### 如何使用？

在 `load_glb_with_physics()` 中：

```python
params = np.load(str(transform_params_path))
s_inv = float(params['s_inv'])     # 缩放
R_inv = params['R_inv']            # 旋转
t_inv = params['t_inv']            # 平移

# 对 GLB 的每个顶点做变换
vertices_hawor = s_inv * (R_inv @ vertices.T).T + t_inv   # GLB → HaWoR
vertices_sapien = (RXWORLD_TO_SAPIEN @ vertices_hawor.T).T  # HaWoR → SAPIEN
```

### 数据流图

```
RAS GLB 顶点 (Y-UP, VGGT 单位, 尺度≈3x)
    │
    │ s_inv * R_inv @ p + t_inv    ← transform_params.npz 的作用
    │ (缩放 + 旋转 + 平移)
    ▼
HaWoR Render World (Y-UP, 米制, 与手部同一坐标系)
    │
    │ RXWORLD_TO_SAPIEN
    ▼
SAPIEN World (Z-UP, 米制)
    │
    │ 此时手和物体在同一场景中，位置/朝向/尺度一致
    ▼
物理仿真 / 渲染
```

### 如果不提供这个参数会怎样？

04_physics_simulation.py 会在启动时检查文件是否存在，不存在则报错：

```
FileNotFoundError: 未找到变换参数文件: ./output/alignment/transform_params.npz
请先运行: python 01_align_scene.py --ras_output ... --hawor_reconstruction ...
```

---

## Q24: 为什么需要精确模式凸分解？太慢了吧？SAPIEN 为什么不能直接设定仿真？

### 核心原因：PhysX 物理引擎只支持凸体碰撞

SAPIEN 底层使用 NVIDIA PhysX 物理引擎。PhysX 对 **动态刚体 (dynamic rigid body)** 的碰撞检测有一个硬性要求：

> **碰撞形状必须是凸的 (convex)**

### 什么是凸体？为什么非凸不行？

```
凸体 (Convex):  任意两点连线都在物体内部
  ■ ← 正方体: 凸的
  ● ← 球体: 凸的
  ▲ ← 三角锥: 凸的

非凸体 (Non-convex):  存在两点连线穿过物体外部
  ☐ ← 杯子: 非凸 (内部有凹陷)
  ⌐ ← L形: 非凸 (有内角)
  🪑 ← 椅子: 非凸
```

PhysX 的碰撞检测算法 (GJK/EPA) **只对凸体有效**。对非凸体，算法会给出错误结果或崩溃。

### SAPIEN 提供的三种碰撞体选项

| 方法 | 速度 | 精度 | 适用场景 |
|------|------|------|---------|
| `add_convex_collision_from_file` | **快** (~1秒) | **低** (凸包包裹整个物体) | 调试、简单物体 |
| `add_multiple_convex_collisions_from_file(decomposition="coacd")` | **慢** (~28分钟首次) | **高** (多个凸体逼近原形状) | 正式仿真、抓取 |
| `add_nonconvex_collision_from_file` | 中 | 精确 | **仅静态物体**，不能用于动态物体 |

### 凸包 vs CoACD 的区别

以一个杯子为例：

```
原始杯子:        凸包 (fast mode):     CoACD (精确模式):
  ┌──┐             ┌──┐                 ┌──┐
  │  │             │  │                 │╶╶│  ← 3个凸部件
  │  │             │  │                 │  │     逼近杯壁
  │  │             │  │                 │╶╶│  ← 1个凸部件
  └──┘             └──┘                 └──┘     逼近杯底

  杯内空间:       杯内空间被填满!       杯内空间保留!
  可以放东西       无法放入任何东西       可以放入小物体
  (凹陷保留)      (凸包填平了凹陷)      (多个凸体逼近)
```

**关键区别**：凸包把所有凹陷都填平了，所以：
- 用凸包：夹爪**无法**伸入杯内抓取，因为杯内空间被碰撞体填满了
- 用 CoACD：夹爪**可以**伸入杯内，因为凹陷区域由多个凸体包围，中间留有空间

### 为什么 CoACD 这么慢？

CoACD (Approximate Convex Decomposition) 算法步骤：

1. **体素化**: 将三角网格转为体素表示 (~1秒)
2. **递归分割**: 找到最佳切割平面，将非凸体切成两个更凸的子体 (~20分钟)
3. **合并优化**: 合并过于细碎的凸体 (~5分钟)
4. **凸包计算**: 对每个子体计算凸包 (~1分钟)

复杂物体 (如桌子、椅子) 有很多凹陷，需要更多次分割，所以更慢。

### 缓存机制

首次 CoACD 计算后，结果会缓存到 `physics_cache/` 目录：

```
physics_cache/
├── final_scene_transform_params_table_0.npz    ← 桌子的凸分解缓存
├── final_scene_transform_params_mug_0.npz      ← 杯子的凸分解缓存
└── ...
```

后续运行直接加载缓存，**不需要重新计算**。所以慢只在第一次。

### 为什么 SAPIEN 不能直接用原始网格仿真？

这不是 SAPIEN 的限制，而是 **PhysX 物理引擎的根本限制**：

1. **碰撞检测算法**: GJK (Gilbert-Johnson-Keerthi) 算法只对凸体有效
2. **接触点计算**: EPA (Expanding Polytope Algorithm) 需要凸体的支撑映射
3. **性能**: 非凸碰撞检测的复杂度是 O(n²)，凸体是 O(log n)

其他物理引擎 (Bullet, MuJoCo, Isaac Sim) 也有同样的限制——动态物体必须用凸碰撞体。

### 实际建议

| 场景 | 推荐选项 | 原因 |
|------|---------|------|
| 调试轨迹 | `--fast-collision` | 快速验证轨迹是否合理 |
| 验证抓取 | CoACD (默认) | 需要精确碰撞才能抓取 |
| 简单物体 (球/方块) | `--fast-collision` | 简单物体凸包≈原形状 |
| 复杂物体 (杯子/工具) | CoACD (默认) | 凹陷区域对抓取至关重要 |

---

## Q25: 从视频到完整的机器人仿真数据集，还需要哪些步骤？

### 当前管线完成的部分

```
✅ 视频输入
✅ 手部 3D 重建 (HaWoR)
✅ 场景 3D 重建 (RAS)
✅ 坐标系对齐 (01_align_scene.py)
✅ 手部→机器人映射 (Dex Retargeting + RelaxedIK)
✅ 运动学渲染 (02_render_scene.py)
✅ 物理仿真 (04_physics_simulation.py)
✅ 视频-仿真对齐验证 (05_video_alignment.py)
```

### 还需要完成的步骤

#### 第一层：基础数据完善 (必须)

| # | 步骤 | 当前状态 | 说明 |
|---|------|---------|------|
| 1 | **物体 6DoF 位姿标注** | ❌ 缺失 | 每个物体在每帧的精确位置和朝向。当前只有 GLB 静态位置，物体被碰后位姿未知 |
| 2 | **物体语义标注** | ❌ 缺失 | 每个物体是什么 (杯子/桌子/书本...)。当前 GLB 只有几何体名称 |
| 3 | **手-物接触标注** | ❌ 缺失 | 哪些帧手在接触物体、接触点在哪。当前只有物理仿真的接触力检测 |
| 4 | **动作分割** | ❌ 缺失 | 将连续操作分割为原子动作: reach → grasp → lift → transport → place |
| 5 | **任务描述** | ❌ 缺失 | 自然语言描述: "拿起杯子放到桌子上" |

#### 第二层：仿真数据增强 (推荐)

| # | 步骤 | 当前状态 | 说明 |
|---|------|---------|------|
| 6 | **多视角渲染** | ❌ 缺失 | 从多个相机角度渲染同一操作，用于多视角训练 |
| 7 | **域随机化** | ❌ 缺失 | 随机化纹理、光照、物体位置，提升策略泛化能力 |
| 8 | **物体变体** | ❌ 缺失 | 同一语义类别的不同形状/大小物体 (5种不同的杯子) |
| 9 | **扰动轨迹** | ❌ 缺失 | 在原始轨迹上添加噪声，生成多样化的操作数据 |
| 10 | **失败案例** | ❌ 缺失 | 抓取失败、物体滑落等负样本，对 RL 训练至关重要 |

#### 第三层：策略学习数据 (进阶)

| # | 步骤 | 当前状态 | 说明 |
|---|------|---------|------|
| 11 | **成功/失败标签** | ❌ 缺失 | 每次操作是否成功完成 |
| 12 | **奖励函数** | ❌ 缺失 | RL 训练需要的稀疏/稠密奖励 |
| 13 | **观测数据** | ❌ 缺失 | 相机图像 + 关节角 + 夹爪状态 + 物体位姿 |
| 14 | **动作数据** | ❌ 缺失 | 末端执行器位姿增量 / 关节角增量 |
| 15 | **数据格式化** | ❌ 缺失 | 转换为 RLDS / Open X-Embodiment / DROID 等标准格式 |

### 完整数据集生成管线

```
原始视频
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  当前管线 (已完成)                                        │
│  HaWoR → 手部3D  +  RAS → 场景3D  +  对齐 → 物理仿真     │
└────────────────────────┬────────────────────────────────┘
                         │
    ┌────────────────────┼────────────────────────────┐
    ▼                    ▼                            ▼
物体位姿标注          动作分割+标注               接触标注
(MASt3R/SAM3)        (视频理解模型)             (SAM3+接触检测)
    │                    │                            │
    ▼                    ▼                            ▼
每帧6DoF位姿         原子动作序列                接触帧+接触点
    │                    │                            │
    └────────────────────┼────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  仿真数据生成        │
              │  多视角 + 域随机化   │
              │  + 物体变体 + 扰动   │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  数据格式化          │
              │  RLDS / OXE / DROID  │
              │  观测+动作+奖励      │
              └──────────┬──────────┘
                         │
                         ▼
              机器人操作数据集
```

### 各步骤的工具推荐

| 步骤 | 推荐工具 | 难度 |
|------|---------|------|
| 物体位姿标注 | MASt3R + SAM3 + 可微渲染 | ★★★ |
| 物体语义标注 | SAM3 + CLIP | ★★ |
| 接触标注 | SAM3 (手物分割) + 物理仿真接触力 | ★★ |
| 动作分割 | 视频LLM (GPT-4V) / 专用模型 | ★★★ |
| 任务描述 | 视频LLM (GPT-4V) | ★ |
| 多视角渲染 | SAPIEN 多相机 | ★ |
| 域随机化 | SAPIEN 材质/光照随机化 | ★★ |
| 数据格式化 | tensorflow/datasets (RLDS) | ★★ |

---

## Q26: 映射是否真的正确？两个文件如何合并？为什么需要 Umeyama 对齐？

### 映射是否正确？

**部分正确，但有误差来源。** 当前管线的映射链是：

```
人手 (MANO 21关节, HaWoR 坐标系)
    │ Dex Retargeting (3约束点优化)
    ▼
夹爪位姿 (位置 + 朝向 + 开合度)
    │ RelaxedIK (6DOF逆运动学)
    ▼
机器人关节角 (6臂关节 + 2夹爪关节)
    │ PD控制 + 物理仿真
    ▼
机器人运动
```

每个环节都有误差：

| 环节 | 误差来源 | 量级 |
|------|---------|------|
| HaWoR 手部重建 | SLAM 漂移、深度估计误差 | 1-5cm |
| Dex Retargeting | 3约束点无法完全约束7DOF | 0.5-2cm |
| RelaxedIK | 增量求解、关节限位 | 1-3cm |
| 物理仿真 | PD跟踪误差、接触力扰动 | 0.5-2cm |
| **GLB对齐** | **尺度、旋转、平移误差** | **1-10cm** |

**最大的误差来源是 GLB 对齐**，也就是两个文件合并的过程。

### 两个文件如何合并？

HaWoR 和 RAS 输出的是两个独立的数据文件：

| 文件 | 内容 | 坐标系 |
|------|------|--------|
| `hawor_results_*.npz` | 手部关节、相机轨迹 | HaWoR SLAM World |
| `final_scene.glb` | 3D 场景网格 | RAS World |

**合并 = 把 GLB 物体变换到 HaWoR 坐标系中**，使手和物体在同一空间。

合并过程 (01_align_scene.py)：

```
Step 1: RAS 相机位姿 Z-UP → Y-UP
    RAS 外参在 Z-UP 坐标系，GLB 在 Y-UP，需要统一

Step 2: 第一帧相机锚定
    RAS 和 HaWoR 处理同一个视频，第一帧相机是同一个物理相机
    R_align = R_c2w_hawor @ OPENCV_TO_OPENGL @ R_c2w_ras.T
    t_align = t_c2w_hawor - R_align @ t_c2w_ras

Step 3: 尺度校正 (Umeyama)
    RAS 和 HaWoR 的世界尺度不同，需要缩放

Step 4: 变换 GLB 顶点
    p_hawor = s_inv * R_inv @ p_glb + t_inv
```

合并后的数据流：

```
hawor_results_*.npz                    final_scene.glb
    │                                      │
    │ 手部关节 (HaWoR坐标)                  │ GLB顶点 (RAS坐标)
    │                                      │
    │                                      │ s_inv * R_inv @ p + t_inv
    │                                      │ (对齐到HaWoR坐标)
    │                                      │
    └──────────── 同一坐标系 ──────────────┘
                         │
                         │ RXWORLD_TO_SAPIEN
                         ▼
                    SAPIEN 仿真场景
```

### 为什么需要 Umeyama 对齐？

**核心问题：RAS 和 HaWoR 的世界尺度不同。**

| | RAS (VGGT) | HaWoR (Metric3D) |
|---|---|---|
| 深度估计方法 | VGGT (基于视频的深度) | Metric3D (单目深度) |
| 世界尺度 | 非米制 (相对尺度) | 米制 (绝对尺度) |
| 同一物体的距离 | 可能是 1.5 (VGGT单位) | 可能是 0.5 (米) |
| 尺度比 | ≈ 3x | 1x |

**如果不做尺度校正**：

```
❌ 没有 Umeyama:
  GLB 物体 (RAS单位): 桌子在距离 1.5 处
  手部 (HaWoR米制): 手在距离 0.5 处
  → 物体是手的 3 倍远，完全对不上
  → 或者等价地：物体比手大 3 倍

✅ 有 Umeyama:
  s_inv ≈ 0.32 (RAS 单位 → HaWoR 米制的缩放)
  GLB 物体缩放后: 桌子在距离 0.5 处
  手部: 手在距离 0.5 处
  → 物体和手在同一位置，尺度一致
```

### Umeyama 尺度是怎么算的？

Umeyama 利用**相机轨迹的离散度**来估算尺度比：

```python
# RAS 相机轨迹的离散度 (相机移动了多远)
sigma_ras = sqrt(mean(||ras_cam_pos - ras_cam_mean||²))

# HaWoR 相机轨迹的离散度
sigma_hawor = sqrt(mean(||hawor_cam_pos - hawor_cam_mean||²))

# 尺度比 = RAS轨迹范围 / HaWoR轨迹范围
scale_ratio = sigma_ras / sigma_hawor

# RAS → HaWoR 缩放因子
s_inv = 1.0 / scale_ratio
```

**直觉**：同一个人拿着同一个相机拍同一段视频，相机移动的物理距离是一样的。但 RAS 和 HaWoR 对这个距离的"刻度"不同。通过比较两者对同一运动的"测量值"，就能算出刻度比。

### Umeyama 的局限

| 情况 | 问题 | 解决方案 |
|------|------|---------|
| 静态相机 (相机不动) | sigma→0，尺度比不稳定 | 回退到手-GLB距离启发式估算 |
| 快速运动 | 轨迹噪声大，尺度比不准 | 多段取平均 |
| 尺度漂移 | SLAM 累积漂移导致局部尺度不一致 | 分段 Umeyama 或 MASt3R 稠密对应 |

### 如何验证对齐是否正确？

1. **2D 重投影误差** (05_video_alignment.py): 将 3D 手部投影到 2D，与视频对比。当前误差 0.5px (优秀)
2. **手-GLB 距离**: 手到最近 GLB 顶点的距离。min < 0.001m 为优秀
3. **视觉检查**: 在 Viewer 中看手和物体是否在同一位置
4. **视频叠加**: 将仿真渲染叠加到原始视频上，看是否对齐

### 总结：为什么需要两步对齐？

```
第一步: 旋转+平移 (R_align, t_align)
  → 解决"方向和位置"问题
  → 基于第一帧相机位姿锚定
  → 这个是精确的 (同一个物理相机)

第二步: 尺度 (s_inv)
  → 解决"大小"问题
  → 基于 Umeyama 相机轨迹离散度比
  → 这个是近似的 (依赖深度估计一致性)
  → 最容易出错的环节
```

**如果对齐不准，最可能的原因是 s_inv (尺度) 不对**。可以用 `--force-scale` 手动调整，或用 MASt3R/SAM3 做更精确的尺度估计。

---

## Q27: V-Dreamer 是如何从对齐到仿真的？我们可以借鉴什么？

### 论文信息

**V-Dreamer: Automating Robotic Simulation and Trajectory Synthesis via Video Generation Priors**
南京大学, 西北工业大学, 2026.03
arXiv: 2603.18811

### V-Dreamer 的三阶段管线

```
自然语言指令 (可选: 真实场景照片)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: Semantic-to-Physics Scene Synthesis                 │
│  文本 → LLM 资产清单 → Flux 扩散模型生成2D图 → SAM3 去背景   │
│  → 3D 重建 → 物理验证布局                                     │
│  输出: 仿真就绪的3D场景 (SAPIEN/Isaac)                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: Video-Prior-Based Trajectory Generation             │
│  仿真场景渲染初始帧 → 视频生成模型 (CogVideoX) "做梦"        │
│  → 生成操作视频 (2D像素空间)                                  │
│  输出: 2D 操作视频                                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3: Sim-to-Gen Alignment                                │
│  2D视频 → CoTracker3 追踪末端执行器 → VGGT 深度估计          │
│  → 3D运动提升 → IK求解 → 机器人关节角                        │
│  输出: 可执行的机器人轨迹 (LeRobot格式)                        │
└──────────────────────────────────────────────────────────────┘
```

### V-Dreamer vs 当前管线对比

| 对比项 | V-Dreamer | 当前管线 |
|--------|-----------|---------|
| **输入** | 自然语言指令 (+ 可选照片) | 第一人称操作视频 |
| **场景来源** | LLM + 扩散模型生成3D场景 | RAS (VGGT) 从视频重建3D场景 |
| **轨迹来源** | 视频生成模型 "做梦" 生成操作视频 | HaWoR 从视频重建手部3D轨迹 |
| **2D→3D提升** | CoTracker3 + VGGT | HaWoR SLAM + Metric3D |
| **坐标系对齐** | Sim-to-Gen (渲染帧→生成视频→3D提升) | 第一帧相机锚定 + Umeyama |
| **机器人映射** | 3D末端轨迹 → IK | Dex Retargeting → RelaxedIK |
| **物理仿真** | SAPIEN/Isaac 仿真验证 | SAPIEN PhysX 物理仿真 |
| **输出格式** | LeRobot 格式轨迹 | MP4视频 + NPZ关节角 |

### V-Dreamer 的三个关键创新

#### 1. 视频生成模型作为运动先验 (Video-Prior)

**核心思想**：不依赖人手重建，而是让视频生成模型"想象"机器人应该如何操作。

```
传统方法 (我们的管线):
  视频 → 人手3D重建 → 手→夹爪映射 → IK → 机器人轨迹
  问题: 人手重建有误差，手→夹爪映射有信息损失

V-Dreamer:
  仿真场景初始帧 → 视频生成模型 → 机器人操作视频 → 2D追踪+3D提升 → 机器人轨迹
  优势: 直接生成机器人操作，跳过人手重建
```

**具体做法**：
1. 将 Stage 1 生成的仿真场景渲染第一帧
2. 用 CogVideoX 等视频生成模型，以第一帧+任务描述为条件，生成操作视频
3. 使用 negative prompting 避免物理不合理的动作 (如物体穿透、悬浮等)

#### 2. Sim-to-Gen 对齐模块 (最值得借鉴)

**这是 V-Dreamer 最核心的技术贡献**，解决了"2D视频→3D机器人轨迹"的转换问题。

```
2D 操作视频
    │
    ├─① Mask-Restricted Tracking (CoTracker3)
    │   在第一帧标注末端执行器mask → CoTracker3 追踪后续帧的2D位置
    │   输出: 末端执行器的2D像素轨迹 (u, v) 序列
    │
    ├─② Metric Depth Estimation (VGGT)
    │   对每帧估计米制深度图
    │   输出: 每个像素的深度值 d (米)
    │
    ├─③ 3D Motion Lifting
    │   2D轨迹 + 深度图 + 相机内参 → 3D点轨迹
    │   p_3d = d * K^{-1} @ [u, v, 1]^T
    │   输出: 末端执行器的3D轨迹 (x, y, z) 序列
    │
    └─④ IK Solving
        3D末端轨迹 → 逆运动学 → 关节角序列
        输出: 可执行的机器人关节角
```

**关键细节**：
- **Mask-Restricted Tracking**: 不是追踪整个视频，而是只在末端执行器的mask区域内追踪，避免误追踪
- **VGGT 深度**: VGGT (Video Grounded Geometry Transformer) 提供米制深度估计，比 Metric3D 更准确
- **3D Lifting**: 利用已知的仿真场景相机参数，将2D+深度精确转换为3D世界坐标

#### 3. Real2Sim2Real 零样本迁移

```
真实场景照片 + 语言指令
    │
    ▼
V-Dreamer 生成匹配的仿真场景 (photo-conditioned)
    │
    ▼
在仿真中生成1条专家轨迹
    │
    ▼
训练策略 (仅1条演示!)
    │
    ▼
零样本迁移到真实机器人
```

### 我们可以借鉴什么？

#### 借鉴1: 用 CoTracker3 替代/补充 HaWoR 手部追踪 (★★★ 最实用)

**问题**: HaWoR 手部重建有 SLAM 漂移和深度估计误差
**V-Dreamer 方案**: 用 CoTracker3 在2D追踪末端执行器

**我们的实现思路**:
```python
# 当前: 依赖 HaWoR 的 3D 手部轨迹
pred_trans = hawor_data["pred_trans"]  # 有累积漂移

# 改进: 用 CoTracker3 在2D追踪手腕，再用 VGGT 深度提升到3D
# 1. 在第一帧标注手腕位置 (从 HaWoR 2D 关节点初始化)
# 2. CoTracker3 追踪手腕的2D轨迹
# 3. VGGT 估计深度
# 4. 2D + 深度 → 3D 轨迹

# 优势: CoTracker3 是2D追踪，不受3D重建漂移影响
# 劣势: 需要准确的深度估计
```

**实现步骤**:
1. 安装 CoTracker3: `pip install cotracker`
2. 从 HaWoR 的 cam_space 数据获取第一帧手腕2D位置
3. 运行 CoTracker3 追踪手腕2D轨迹
4. 用 VGGT 或 Metric3D 估计深度
5. 2D+深度 → 3D 轨迹，替代 pred_trans

#### 借鉴2: 用 VGGT 替代 Metric3D 做深度估计 (★★★ 最实用)

**问题**: 当前 Umeyama 尺度校正依赖相机轨迹离散度比，静态相机时不稳定
**V-Dreamer 方案**: 用 VGGT 直接估计米制深度

**我们的实现思路**:
```python
# 当前: Umeyama 尺度校正 (间接，依赖相机轨迹)
s_inv = sigma_hawor / sigma_ras  # 静态相机时不稳定

# 改进: 用 VGGT 直接估计深度，计算尺度
# 1. VGGT 估计视频帧的米制深度图
# 2. 在深度图中找到手部区域和物体区域的深度
# 3. 与 RAS GLB 的已知尺度对比，直接计算 s_inv

# 优势: 不依赖相机运动，静态相机也有效
# 劣势: VGGT 需要GPU，推理速度较慢
```

#### 借鉴3: Mask-Restricted Tracking 做物体位姿追踪 (★★ 有价值)

**问题**: 当前 GLB 物体只有静态位姿，被碰后位姿未知
**V-Dreamer 方案**: 用 CoTracker3 追踪物体的2D位置

**我们的实现思路**:
```python
# 1. SAM3 分割视频中的物体 mask
# 2. CoTracker3 追踪物体 mask 的2D轨迹
# 3. VGGT 估计物体深度
# 4. 2D + 深度 → 物体3D位姿序列

# 输出: 每帧每个物体的6DoF位姿
# 用途: 物理仿真中验证物体位姿是否正确
```

#### 借鉴4: 视频生成模型做轨迹增强 (★ 算力要求高)

**问题**: 当前只有一条人手轨迹，无法生成多样化数据
**V-Dreamer 方案**: 用视频生成模型生成多条操作视频

**我们的实现思路** (轻量版):
```python
# 完整版 (需要大算力):
# 仿真场景渲染 → CogVideoX 生成多条操作视频 → Sim-to-Gen 对齐

# 轻量版 (当前算力可行):
# 1. 在已有轨迹上添加高斯噪声 (位置±2cm, 朝向±5°)
# 2. 用 SAPIEN 物理仿真验证哪些扰动轨迹仍然可行
# 3. 可行的轨迹作为数据增强

# 优势: 不需要视频生成模型，只需要物理仿真
# 劣势: 多样性不如视频生成
```

#### 借鉴5: LeRobot 数据格式 (★★ 标准化)

**问题**: 当前输出是自定义的 MP4 + NPZ 格式
**V-Dreamer 方案**: 使用 LeRobot 标准格式

**我们的实现思路**:
```python
# LeRobot 格式: 每个 episode 包含
# - observation.images.cam_high: 高分辨率相机图像
# - observation.images.cam_wrist: 手腕相机图像
# - observation.state: 机器人关节角
# - action: 末端执行器位姿增量
# - language_instruction: 任务描述

# 转换脚本
def convert_to_lerobot(qpos_sequence, video_frames, task_description):
    dataset = {
        "observation.state": qpos_sequence,        # (N, 8) 关节角
        "action": compute_ee_delta(qpos_sequence),  # (N, 7) 位姿增量
        "language_instruction": task_description,
    }
    return dataset
```

### 实现优先级建议

| 优先级 | 借鉴内容 | 算力需求 | 效果 |
|--------|---------|---------|------|
| **1** | CoTracker3 追踪手腕2D轨迹 | 低 (CPU可跑) | 减少SLAM漂移影响 |
| **2** | VGGT 替代 Umeyama 尺度校正 | 中 (需GPU) | 静态相机也能对齐 |
| **3** | LeRobot 数据格式化 | 低 | 标准化输出 |
| **4** | Mask-Restricted 物体追踪 | 中 | 物体位姿标注 |
| **5** | 轨迹扰动增强 | 低 | 数据多样化 |
| **6** | 视频生成模型轨迹合成 | 高 (8×4090) | 最高多样性 |

### V-Dreamer 的局限

| 局限 | 说明 |
|------|------|
| **场景类型** | 仅限桌面操作 (tabletop)，不支持大范围移动操作 |
| **物体复杂度** | 生成的3D物体质量有限，复杂物体 (如杯子凹陷) 可能不准 |
| **视频物理一致性** | 视频生成模型可能违反物理约束 (物体穿透、悬浮) |
| **深度估计精度** | VGGT 在遮挡区域深度不准 |
| **算力需求** | 8×RTX 4090 才能实现 600 trajectories/hour |
| **与我们的差异** | V-Dreamer 从文本生成，我们从真实视频重建——后者更准确但更受限 |

### 总结：V-Dreamer 的核心启示

1. **2D→3D 的 Sim-to-Gen 对齐思路**比纯3D重建更鲁棒——先在2D追踪，再用深度提升
2. **视频生成模型可以作为运动先验**——不依赖人手重建，直接"想象"机器人操作
3. **Mask-Restricted Tracking** 是追踪精度和鲁棒性的关键——只在感兴趣区域追踪
4. **VGGT 米制深度**比 Umeyama 尺度校正更直接——不依赖相机运动
5. **Real2Sim2Real** 是可行的——1条仿真演示就能实现零样本迁移

---

## Q28: 05_video_alignment.py 目前的对齐方式是什么？

### 一句话概括

**05_video_alignment.py 通过将 3D 手部关节投影到 2D 图像平面，与 GT 2D 关键点比较像素距离来量化对齐质量，并可通过 L-BFGS-B 优化平移偏移来改善对齐。**

### 三种对齐模式

| 模式 | 功能 | 输出 |
|------|------|------|
| `overlay` | 视频叠加对比：仿真渲染半透明叠加到原始视频帧 | overlay_comparison.mp4 |
| `reproj_analysis` | 2D 重投影误差分析：逐帧计算 GT vs Sim 的像素误差 | reproj_analysis.mp4 + 统计 |
| `optimize` | 位姿优化：基于重投影误差优化 3D 平移偏移 | pose_offset.npz + optimization_vis.mp4 |
| `full` | 完整流程：叠加 → 分析 → 优化 | 以上全部 |

### 对齐的核心流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 获取 GT 2D 关节 (cam_space)                           │
│                                                                 │
│  cam_space MANO 参数 (init_trans, init_root_orient, ...)        │
│      ↓ MANOLayer FK                                            │
│  joints_cam (相机坐标系 3D 关节)                                 │
│      ↓ project_3d_to_2d(focal, cx, cy)                         │
│  joints_2d_gt (GT 2D 像素坐标)                                  │
│      绿色骨架显示                                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 获取 Sim 2D 关节 (world → cam)                        │
│                                                                 │
│  pred_trans + pred_rot + pred_hand_pose (Render World, Y-UP)    │
│      ↓ MANOLayer FK                                            │
│  joints_world (Render World 3D 关节)                            │
│      ↓ world_to_cam_hawor()                                    │
│         ① R_x 翻转: points_slam = R_x @ points_render          │
│         ② w2c 变换: points_cam = R_w2c @ points_slam + t_w2c  │
│  joints_cam_sim (相机坐标系 3D 关节)                             │
│      ↓ project_3d_to_2d(focal, cx, cy)                         │
│  joints_2d_sim (Sim 2D 像素坐标)                                │
│      红色骨架显示                                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 3: 计算重投影误差                                         │
│                                                                 │
│  valid_mask = gt有效 & sim有效 (z > 0.01)                       │
│  diff = joints_2d_gt[valid] - joints_2d_sim[valid]             │
│  per_joint_err = ||diff||₂ (像素距离)                           │
│  mean_err = per_joint_err.mean()                                │
│                                                                 │
│  统计指标: mean, median, max, wrist_err, fingertip_err          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Step 4: 位姿优化 (optimize 模式)                               │
│                                                                 │
│  优化变量: offset_trans = [dx, dy, dz] (3D 平移偏移)            │
│  优化目标: min Σ  mean_reproj_error(pred_trans + offset)        │
│  优化方法: L-BFGS-B, maxiter=200, ftol=1e-6                    │
│  采样帧: 10帧 (均匀采样有效帧)                                   │
│                                                                 │
│  输出: pose_offset.npz → offset_trans, before/after error      │
└─────────────────────────────────────────────────────────────────┘
```

### 关键坐标系变换: `world_to_cam_hawor()`

这是对齐的核心函数，将 Render World (Y-UP) 中的 3D 点变换到相机坐标系：

```python
def world_to_cam_hawor(points_world, R_c2w, t_c2w):
    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ t_c2w
    # ① 先翻转: Render World → SLAM World
    points_world_slam = (R_x @ points_world.T).T
    # ② 再做 w2c: SLAM World → Camera
    points_cam = (R_w2c @ points_world_slam.T).T + t_w2c
    return points_cam
```

**变换顺序至关重要**：先 R_x 翻转回 SLAM 坐标系，再做标准 w2c 变换。之前的 Bug 就是顺序反了（先 w2c 再翻转），导致 18.1px 误差；修复后降到 0.2px。

### 2D 投影: `project_3d_to_2d()`

```python
def project_3d_to_2d(points_3d_cam, focal, cx, cy):
    valid = points_3d_cam[:, 2] > 0.01  # 只投影在相机前方的点
    pts_2d[valid, 0] = focal * x / z + cx  # u = f * X/Z + cx
    pts_2d[valid, 1] = focal * y / z + cy  # v = f * Y/Z + cy
    return pts_2d, valid
```

### 误差计算: `compute_reprojection_error()`

```python
def compute_reprojection_error(joints_2d_gt, joints_2d_sim, valid_mask=None):
    diff = joints_2d_gt[valid_mask] - joints_2d_sim[valid_mask]
    per_joint_err = np.linalg.norm(diff, axis=1)  # L2 像素距离
    return mean_err, {
        "mean": mean_err, "median": ..., "max": ...,
        "wrist_err": per_joint_err[0],           # 手腕误差
        "fingertip_err": per_joint_err[[4,8,12,16,20]].mean(),  # 五指尖平均
    }
```

### 位姿优化: `PoseOptimizer.optimize_offset()`

```python
class PoseOptimizer:
    def optimize_offset(self, frame_indices=None, method="L-BFGS-B"):
        def loss_fn(params):
            offset = np.array(params[:3])  # 只优化 3D 平移
            total_err = Σ compute_frame_error(fi, offset) / n_valid
            return total_err

        result = minimize(loss_fn, np.zeros(3), method="L-BFGS-B",
                          options={"maxiter": 200, "ftol": 1e-6})
        return offset, final_err, result
```

### 当前对齐验证的是什么？

| 验证对象 | 说明 |
|----------|------|
| **HaWoR 内部一致性** | cam_space GT vs world→cam 重投影，当前误差 0.50px (优秀) |
| ❌ 仿真与视频对齐 | 需要仿真渲染帧 vs 原始视频帧的像素级对比 |
| ❌ 物体位姿对齐 | 需要 3D 物体投影到 2D 与视频物体位置比较 |
| ❌ 交互时序对齐 | 需要抓取时机、力度的时序验证 |

### 对齐方式的局限

| 局限 | 说明 |
|------|------|
| 只验证手部 | 没有验证 GLB 物体在视频中的位置是否正确 |
| 只优化平移 | 不优化旋转和尺度偏移 |
| GT 来自 cam_space | 验证的是 HaWoR 自身一致性，不是仿真与真实视频的对齐 |
| 优化采样少 | 只用 10 帧采样，可能不具全局代表性 |

### 如何使用优化结果

```bash
# 1. 运行优化
python 05_video_alignment.py --hawor-dir ... --mode optimize

# 2. 加载偏移量
offset = np.load("output/alignment_analysis/pose_offset.npz")["offset_trans"]

# 3. 在 04 脚本中将 offset_trans 加到 pred_trans 上
pred_trans_corrected = pred_trans + offset
```

---

## Q29: 底座位置是如何决定的？为什么底座变化太大？

### 底座位置决策机制

底座位置由两个函数控制：`_compute_optimal_fixed_base()` 和 `_compute_tracking_base_pos()`。

#### 第一步：计算初始固定底座位置 (`_compute_optimal_fixed_base`)

```python
# 输入: 所有帧的手腕位置 (SAPIEN 坐标系)
wrist_arr = np.array(wrist_positions_sapien)
centroid = wrist_arr.mean(axis=0)          # 手腕质心
wrist_range = wrist_arr.max(axis=0) - wrist_arr.min(axis=0)  # 手腕运动范围

arm_base_pos = centroid.copy()
arm_base_pos[2] += COMFORTABLE_REACH      # Z轴上方 0.35m (臂舒适工作距离)
if wrist_range[0] > 0.01:
    arm_base_pos[0] += wrist_range[0] * 0.1  # X方向微调 (手腕运动范围的10%)

arm_base_q = 绕Z轴旋转180°  # 机器人面朝下
```

**关键参数**:

| 参数 | 值 | 含义 |
|------|-----|------|
| `COMFORTABLE_REACH` | 0.35m | 底座到手腕质心的Z轴距离 |
| `COMFORT_TARGET_IN_BASE` | [0.30, 0.0, -0.30] | 手腕在base_link坐标系中的舒适位置 |
| `ARM_MAX_REACH` | 0.713m | 臂最大伸展距离 |

**决策逻辑**：底座放在手腕质心正上方 0.35m 处，使手腕大致在臂的舒适工作空间内。绕Z轴旋转180°是因为 R1 机器人底座朝下安装。

#### 第二步：逐帧跟踪底座 (`_compute_tracking_base_pos`)

```python
# 输入: initial_base_pos (初始底座位置), wrist_pos_sapien (当前帧手腕位置), arm_base_q (底座朝向)
base_R = pr.matrix_from_quaternion(arm_base_q)
wrist_in_base = base_R.T @ (wrist_pos_sapien - initial_base_pos)  # 手腕在base_link坐标系中的位置
offset_in_base = wrist_in_base - COMFORT_TARGET_IN_BASE            # 与舒适目标的偏移
clamped_offset = np.clip(offset_in_base, -BASE_TRACKING_RANGE, BASE_TRACKING_RANGE)  # 限制偏移
delta_world = base_R @ clamped_offset                                # 转回世界坐标系
return initial_base_pos + delta_world                                # 新底座位置
```

**跟踪逻辑**：
1. 计算当前手腕在 base_link 坐标系中的位置
2. 计算与舒适目标 `COMFORT_TARGET_IN_BASE` 的偏移
3. 将偏移限制在 `±BASE_TRACKING_RANGE` 范围内
4. 转回世界坐标系，加到初始底座位置上

### 为什么底座变化太大？

**原因1：`BASE_TRACKING_RANGE = 0.08m` 太大**

底座每帧最多可以移动 8cm，如果手腕在舒适目标两侧来回运动，底座会频繁大幅移动。

**原因2：没有平滑**

`_compute_tracking_base_pos` 是逐帧独立计算的，没有低通滤波或指数平滑。即使 `BASE_TRACKING_ALPHA = 0.15` 已定义但未使用。

**修复**：
- `BASE_TRACKING_RANGE` 从 0.08m 减小到 0.04m
- 后续可添加指数平滑：`tracked_base = (1-α) * prev_tracked_base + α * new_tracked_base`

---

## Q30: 夹爪有时候不会张开，在多次迭代后才张开，是什么问题？

### 现象描述

在 `robot_only` 和 `robot_tracking` 模式中，人手已经明显张开，但夹爪仍然闭合。需要多帧之后夹爪才逐渐张开。

### 根因分析

夹爪开合度来自 Dex Retargeting 优化器的 `finger_joint1/2` 输出。优化器是增量式求解器（NLopt SLSQP），依赖 `last_qpos` 作为初始点。夹爪不张开有以下几个原因：

#### 原因1：Retargeting 优化器的惯性正则化

`SeqRetargeting.retarget()` 内部目标函数包含平滑正则化项：

```
minimize: huber_loss(FK(q)[3个link], ref_value) + λ * ||q - q_last||²
```

`λ * ||q - q_last||²` 项使优化器倾向于保持上一帧的解。当上一帧夹爪闭合时，正则化项会**阻碍夹爪张开**，需要多帧才能"拉"开。

#### 原因2：3 个约束点对夹爪开合的约束力不足

3 个约束点 [4=拇指尖, 8=食指尖, 0=手腕] 中：
- 手腕 (joint 0) 约束的是 `gripper_link` 的位置，与夹爪开合无关
- 拇指尖 (joint 4) 约束的是 `finger_link1` 的位置
- 食指尖 (joint 8) 约束的是 `finger_link2` 的位置

当人手张开时，拇指尖和食指尖距离增大，但 `huber_delta` 和 `normal_delta` 的设置使得优化器对位置偏差的容忍度较高，不会立即调整夹爪。

#### 原因3：`normal_delta` 太小

当前 `normal_delta = 1e-5`，这是 Huber loss 的阈值。当位置误差小于 `normal_delta` 时，梯度为零。这意味着微小的手指位置变化不会触发优化器调整夹爪。

#### 原因4：IK 求解的夹爪映射延迟

即使 retargeting 输出了正确的夹爪开合度，`sapien_qpos[gripper_idx1/2]` 的设置还受到 `TrajectorySmoother` 的平滑处理。平滑器的 `max_velocity=1.5 rad/s` 和低通滤波会进一步延迟夹爪张开。

### 实时监测方案

可以在渲染循环中添加夹爪状态监测：

```python
# 在每帧渲染后检查夹爪状态
gripper_q1 = frame_data[gripper_idx1]
gripper_q2 = frame_data[gripper_idx2]
gripper_opening = abs(gripper_q1 - gripper_q2)  # 开合度

# 从 retargeting 获取目标开合度
ref_value = joints_sapien[ref_indices, :]
retarget_qpos = retargeting.retarget(ref_value)
target_opening = abs(retarget_qpos[gripper_idx1] - retarget_qpos[gripper_idx2])

# 如果目标开合度与实际差距过大，发出警告
if abs(target_opening - gripper_opening) > 0.02:
    self.logger.warning(f"  帧 {frame_idx}: 夹爪延迟! 目标={target_opening:.3f}, 实际={gripper_opening:.3f}")
```

### 可能的改进

| 改进 | 方法 | 效果 |
|------|------|------|
| 增大 `normal_delta` | `1e-5` → `1e-3` | 增大位置偏差的梯度，加速夹爪响应 |
| 减小平滑正则化 | 减小 `λ` | 减少对上一帧的依赖 |
| 夹爪开合直接映射 | 不经过优化器，直接从手指距离计算开合度 | 消除延迟，但可能不连续 |
| 夹爪优先策略 | 检测到"张开"意图时，直接设置夹爪目标值 | 立即响应，但可能过冲 |

---

## Q31: 夹爪前面不张开后面才张开——系统原因分析

### 现象

- **pipeline 模式**（预计算+渲染）：夹爪前面不张开，后面才逐渐张开
- **viewer 模式**（实时）：夹爪响应更即时

### 根本原因：TrajectorySmoother 平滑了夹爪关节

预计算模式的数据流：

```
所有帧 qpos → TrajectorySmoother(含夹爪关节) → 平滑后 qpos → 渲染
```

`smooth_indices = list(arm_joint_indices) + [gripper_idx1, gripper_idx2]`

**夹爪关节被包含在平滑范围内**，导致：

1. **低通滤波延迟**：`lp_alpha=0.25` 使夹爪开合变化被严重平滑，张开动作需要多帧才能体现
2. **速度钳制**：`max_velocity=1.5 rad/s` 限制了夹爪开合速度，从闭合(0.04)到张开(-0.04)需要 `0.08/1.5 ≈ 0.053s ≈ 1.6帧`，但低通滤波使实际延迟远大于此
3. **双向滤波**：`_bidirectional_lowpass` 前向+后向滤波进一步抹平了快速变化

### 为什么 viewer 模式没有这个问题？

Viewer 模式**没有 TrajectorySmoother**，每帧直接设置 retargeting 输出的夹爪值，响应即时。

### Pipeline 比 Viewer 计算量大的原因

| 对比项 | Viewer 模式 | Pipeline 模式 |
|--------|------------|--------------|
| IK 求解 | 每帧 20 次 | 每帧 20 次 |
| TrajectorySmoother | **无** | **有**（10 次迭代 × N 帧） |
| 渲染 | 实时，可能跳帧 | 逐帧渲染，不跳帧 |
| MANO 计算 | 每帧 1 次 | 每帧 2 次（预计算 + 渲染） |
| 视频编码 | 无 | 有（ffmpeg 重编码） |
| 总帧数 | 循环播放 | 固定 N 帧 |

主要差异：
1. **TrajectorySmoother**：10 次迭代 × N 帧，计算量 O(N)
2. **MANO 重复计算**：预计算时算一次，渲染时又算一次
3. **视频编码**：ffmpeg 重编码耗时

### 修复方案

**方案 A（推荐）：从 smooth_indices 中排除夹爪关节**

```python
# 只平滑臂关节，不平滑夹爪
smooth_indices = list(arm_joint_indices)  # 不含 gripper_idx1/2
```

夹爪开合不需要平滑——它来自 retargeting 优化器的连续输出，本身已经比较平滑。臂关节需要平滑是因为 RelaxedIK 的增量求解可能产生抖动。

**方案 B：减小夹爪平滑强度**

如果仍想保留夹爪平滑，可以减小 `lp_alpha` 对夹爪的影响，或增大 `max_velocity`。

---

## Q32: mp4 中机械臂底座位置与 viewer 不同

### 现象

- `--viewer` 模式：底座跟随手腕位置小范围移动，跟随效果好
- mp4 视频：底座位置似乎固定不动，和 viewer 中看到的位置不一样

### 根因

预计算模式的数据流：

```
预计算循环: 每帧计算 tracked_base → robot.set_root_pose(tracked_base) → scene.step()
    → qpos 保存到 qpos_sequence
    → 但 tracked_base 没有保存！

渲染循环: robot.set_qpos(frame_data) → scene.step()
    → 底座位置没有被恢复，使用的是 scene.step() 后的默认位置
```

**问题**：`qpos_sequence` 只保存了关节角，没有保存底座位置。渲染时 `robot.set_qpos()` 只设置关节角，不改变底座位置。而 `scene.step()` 可能会改变底座位置（因为物理引擎的积分）。

### 修复

在预计算时同时保存 `base_pos_sequence`，渲染时用 `robot.set_root_pose()` 恢复底座位置：

```python
# 预计算时
qpos_sequence.append(qpos)
base_pos_sequence.append(tracked_base.copy())

# 渲染时
frame_data = qpos_sequence[frame_idx]
base_data = base_pos_sequence[frame_idx]
if frame_data is not None:
    robot.set_qpos(frame_data)
if base_data is not None:
    robot.set_root_pose(sapien.Pose(base_data.tolist(), arm_base_q.tolist()))
```

### 其他差异

| 对比项 | Viewer 模式 | 预计算模式（修复后） |
|--------|------------|---------------------|
| 底座位置 | 每帧实时计算并设置 | 预计算保存，渲染时恢复 ✓ |
| IK 求解 | 每帧实时 | 预计算 ✓ |
| `ee_pos_filter` | 无 | 有（低通滤波 IK 目标） |
| `TrajectorySmoother` | 无 | 有（只平滑臂关节） |
| `scene.step()` | 每帧 | 每帧 ✓ |

修复后两者应该基本一致，剩余差异来自 `ee_pos_filter` 和 `TrajectorySmoother`，这些是为了视频平滑性而添加的后处理。

---

## Q33: 机器人正手/反手（夹爪朝向）问题

### 问题描述

当前 Dex Retargeting 使用 3 个约束点 [拇指尖(4), 食指尖(8), 手腕(0)] 来映射夹爪位姿。3 个点提供 9 个位置约束，优化变量为 7 DOF（6 dummy + 1 gripper），超定 2 个约束。但夹爪的**绕接近轴旋转**（即正手/反手）仍然存在模糊性。

### 什么是正手/反手？

```
正手 (Palm-down / Supination):     反手 (Palm-up / Pronation):
    手心朝下                              手心朝上
    ┌──┐                                 ┌──┐
    │🤚│ ← 手心向下                       │🤚│ ← 手心向上
    └──┘                                 └──┘
    适合: 从上方抓取桌面物体               适合: 从下方托举物体

机械臂对应:
    正手: 夹爪手指朝下 (默认)              反手: 夹爪手指朝上 (需旋转180°)
```

### 为什么 3 个约束点不能完全确定正手/反手？

3 个约束点的几何关系：

```
手腕 (0) ─── 食指尖 (8)
  │
  └── 拇指尖 (4)
```

这 3 个点定义了一个平面，确定了：
- ✅ 夹爪位置（3D 平移）
- ✅ 夹爪接近方向（平面法线方向）
- ✅ 夹爪开合方向（拇指→食指方向）
- ❌ **绕接近轴的旋转**（正手 vs 反手）

原因：当夹爪绕接近轴旋转 180° 时，拇指尖和食指尖的位置会互换，但 3 个约束点仍然可以被满足（优化器只需调整夹爪开合度和微小朝向）。

### 当前代码如何处理？

当前代码通过 `R_GRIPPER_ALIGN` 矩阵和 `OPERATOR2MANO` 矩阵隐式地选择了正手：

```python
# 02_render_scene.py / 02_render_scene_auto.py
R_ee_base = base_link_R_inv @ R_ee_world_fk  # FK 提取的朝向
# R_ee_world_fk 来自 Dex Retargeting 的内部 FK

# 03_track_robot.py
R_ee_base = base_link_R_inv @ R_mano2world @ R_GRIPPER_ALIGN
# R_mano2world = wrist_rot_sapien @ OPERATOR2MANO_RIGHT.T
```

`OPERATOR2MANO_RIGHT = [[-1,0,0],[0,0,1],[0,1,0]]` 将操作员手坐标系映射到 MANO 手坐标系，这个映射隐含了"正手"假设。

### 问题场景

| 场景 | 期望 | 当前行为 | 问题 |
|------|------|---------|------|
| 从上方抓取桌面物体 | 正手 (palm-down) | ✅ 正确 | 无 |
| 从下方托举物体 | 反手 (palm-up) | ❌ 仍然是正手 | 夹爪方向错误，无法托举 |
| 翻转手腕操作 | 动态切换 | ❌ 始终正手 | 夹爪不跟随手腕翻转 |
| 左手操作 | 左手正手 | ⚠️ 可能不正确 | OPERATOR2MANO_LEFT 不同 |

### 解决方案

#### 方案 A: 利用手腕旋转检测正手/反手 (推荐)

```python
def detect_gripper_orientation(wrist_rot_sapien, R_coord):
    """根据手腕旋转判断正手/反手
    
    核心思想: 手心法向量在世界坐标系中的朝向决定了正手/反手
    - 手心法向量朝下 (Z < 0) → 正手 (palm-down)
    - 手心法向量朝上 (Z > 0) → 反手 (palm-up)
    """
    # MANO 手心法向量在操作员坐标系中 = [0, -1, 0] (指向手掌内侧)
    palm_normal_operator = np.array([0, -1, 0])
    
    # 变换到世界坐标系
    R_mano2world = wrist_rot_sapien @ OPERATOR2MANO_RIGHT.T
    palm_normal_world = R_mano2world @ palm_normal_operator
    
    # 在 SAPIEN Z-UP 坐标系中判断
    if palm_normal_world[2] < 0:
        return "palm_down"  # 正手
    else:
        return "palm_up"   # 反手
```

#### 方案 B: 增加第 4 个约束点

```
当前 3 点: [拇指尖(4), 食指尖(8), 手腕(0)]
增加第 4 点: [中指根(9)] 或 [小指尖(16)]

4 个约束点 → 12 个位置约束 vs 7 DOF → 超定 5 个约束
足以完全确定夹爪朝向（包括正手/反手）
```

**优点**: 优化器自动选择正确的朝向，不需要额外逻辑
**缺点**: 需要修改 retargeting 配置，可能影响收敛性

#### 方案 C: 基于手腕旋转的动态 R_GRIPPER_ALIGN

```python
# 根据手腕旋转动态选择对齐矩阵
if is_palm_up:
    R_GRIPPER_ALIGN = R_GRIPPER_ALIGN_FLIPPED  # 绕接近轴旋转180°
else:
    R_GRIPPER_ALIGN = R_GRIPPER_ALIGN_DEFAULT
```

**优点**: 改动最小，只需修改 IK 目标朝向
**缺点**: 正手/反手切换时可能产生不连续

### 推荐实现路径

| 优先级 | 方案 | 改动量 | 效果 |
|--------|------|--------|------|
| **1** | 方案 A: 手腕旋转检测 | 小 | 能检测正手/反手，但需要配合方案 C |
| **2** | 方案 C: 动态 R_GRIPPER_ALIGN | 小 | 实现正手/反手切换 |
| **3** | 方案 B: 增加第 4 约束点 | 中 | 最根本的解决方案，优化器自动处理 |

### 对左手的影响

左手的 `OPERATOR2MANO_LEFT = [[0,0,-1],[1,0,0],[0,-1,0]]` 与右手不同，这意味着左手的"正手"定义也不同。需要分别处理：

```python
if hand_type == "left":
    OPERATOR2MANO = OPERATOR2MANO_LEFT
    palm_normal_operator = np.array([0, 1, 0])  # 左手心法向量方向相反
else:
    OPERATOR2MANO = OPERATOR2MANO_RIGHT
    palm_normal_operator = np.array([0, -1, 0])
```

---

## Q34: hawor/7 与 hawor/laptop 数据差异及管线兼容性

### 原版 `02_render_scene.py` 的检测和生成逻辑

原版 `02_render_scene.py` **有自己的手部检测逻辑**，但比 `hand_detector.py` 简单得多：

```python
# 02_render_scene.py 中的 _detect_hand_idx()
def _detect_hand_idx(hawor_path):
    """仅通过 cam_space/ 子目录判断手部索引"""
    cam_dir = hawor_path / "cam_space"
    if cam_dir.exists():
        detected = set()
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))
        if 0 in detected and 1 not in detected:
            return 0  # 仅左手
        if 1 in detected and 0 not in detected:
            return 1  # 仅右手
        if 0 in detected:
            return 0  # 都有 → 默认左手
        if 1 in detected:
            return 1
    return None  # 无法检测
```

**与 `hand_detector.py` 的区别**：

| 特性 | `02_render_scene.py._detect_hand_idx()` | `hand_detector.HandDetector` |
|------|---------------------------------------|------------------------------|
| 检测方式 | 仅 cam_space 目录 | pred_valid + cam_space + world_space_res |
| 返回值 | 单个 hand_idx (0 或 1) | Handedness 枚举 (LEFT/RIGHT/BOTH/NONE) |
| 运动范围 | 不考虑 | 考虑 (区分主要操作手) |
| 双手支持 | ❌ (返回单个 idx) | ✅ (返回 BOTH) |
| 降级策略 | 无 | cam_space → world_space_res 逐级回退 |

原版的生成逻辑也是硬编码为右手：
- 固定使用 `FLOATING_RIGHT_URDF`
- 固定使用 `OPERATOR2MANO_RIGHT`
- 固定使用 `R1_RIGHT_SETTINGS`
- IK 只调用 `solve_position_right()`

### hawor/7 与 hawor/laptop 的具体区别

#### 目录结构对比

```
hawor/7/                              hawor/laptop/
├── cam_space/                        (无)
│   └── 0/
│       └── 0_112.json (569KB)
├── reconstruction/                   (无)
│   └── hawor_results_0_113.npz (62KB)
├── SLAM/                             (无)
│   └── hawor_slam_w_scale_0_113.npz (6.6MB)
├── tracks_0_113/                     (无)
│   ├── frame_chunks_all.npy (1.2KB)
│   ├── model_boxes.npy (280B)
│   ├── model_masks.npy (224MB)
│   └── model_tracks.npy (17KB)
├── extracted_images/                 (无)
├── est_focal.txt (内容: "600")       (无)
├── world_space_res.pth (56KB)        (无)
├── vis_cam_0_112/ (PNG)             ├── vis_cam_0_600/ (PNG)
├── vis_cam_0_113/ (PNG)             (无)
├── vis_verify/hand_0/ (PNG+MP4)     (无)
├── vis_world_0_112/ (PNG+MP4)       ├── vis_world_0_600/ (PNG+MP4)
└── vis_world_0_113/ (PNG+MP4)       └── (仅以上两个目录)
```

#### 文件级对比

| 文件/目录 | hawor/7 | hawor/laptop | 管线是否需要 | 说明 |
|-----------|---------|-------------|-------------|------|
| `reconstruction/hawor_results_*.npz` | ✅ 62KB | ❌ | **必须** | 手部关节+相机轨迹核心数据 |
| `cam_space/*/` | ✅ 0/ (左手) | ❌ | 可选 | 手部检测辅助 |
| `world_space_res.pth` | ✅ 56KB | ❌ | 备选 | 旧版数据格式 |
| `est_focal.txt` | ✅ "600" | ❌ | 备选 | 焦距估计 |
| `SLAM/hawor_slam_*.npz` | ✅ 6.6MB | ❌ | 不需要 | SLAM 数据 (管线未使用) |
| `tracks_*/` | ✅ 224MB | ❌ | 不需要 | 2D 跟踪数据 (管线未使用) |
| `extracted_images/` | ✅ | ❌ | 不需要 | 原始视频帧 |
| `vis_cam_*/` | ✅ 112+113帧 | ✅ 600帧 | 不需要 | 可视化: 相机视角渲染 |
| `vis_world_*/` | ✅ 112+113帧 | ✅ 600帧 | 不需要 | 可视化: 世界视角渲染 |
| `vis_verify/` | ✅ | ❌ | 不需要 | 可视化: 验证视频 |

#### 数据量对比

| 指标 | hawor/7 | hawor/laptop |
|------|---------|-------------|
| 帧数 | 113 | 600 |
| 可视化图片 | 294 张 | 453 张 |
| 核心数据文件 | 5 个 | 0 个 |
| 总数据量 | ~224MB (含 tracks) | ~453 张 PNG |

### hawor/laptop 数据状态 (已验证可用)

`hawor/laptop` **有完整的 reconstruction 数据**，但存在 NaN 问题：

| 数据 | 状态 | 说明 |
|------|------|------|
| `reconstruction/hawor_results_0_600.npz` | ✅ 62KB | 600帧，完整数据 |
| `cam_space/1/` | ✅ | 仅有右手 (idx=1) |
| `est_focal.txt` | ✅ "600" | 焦距 |
| `world_space_res.pth` | ❌ | 缺失 |
| `R_c2w` / `t_c2w` | ✅ | 相机轨迹完整 |

**NaN 问题**：
- 左手 (idx=0): 477/600 有效帧，但有 1431 个 NaN → 过滤后 0 有效帧
- 右手 (idx=1): 578/600 有效帧，但有 246 个 NaN → 过滤后 496 有效帧 (82.7%)
- NaN 原因: 画面中没有手时，HaWoR 输出 pred_trans 为 NaN

**hand_track/render_auto.py 的 NaN 处理**：
1. `load_hawor_data()` 中过滤 NaN 帧：`pred_valid = pred_valid & ~has_nan`
2. `compute_retargeting()` 中检测 NaN：NaN 帧返回 None
3. 无效帧时只显示 GLB 场景，不渲染机器人
4. 手重新出现时做 mini-warmup（重新初始化 filter 和 IK）

**测试结果**：
```
# 前 10 帧 (全部无效)
python hand_track/render_auto.py --hawor-dir /home/an/data/hawor/laptop ... --num-frames 10
→ Warmup 失败 (无有效帧), 但正常渲染 GLB 场景

# 帧 100-130 (有手出现)
python hand_track/render_auto.py --hawor-dir /home/an/data/hawor/laptop ... --start-frame 100 --num-frames 30
→ 自动检测为右手, Warmup 完成, mini-warmup 触发, 视频保存成功 (242KB)
```

### 如何让 laptop 数据可用？

需要重新运行 HaWoR 推理，确保输出包含 `reconstruction/` 目录。HaWoR 的完整推理命令通常为：

```bash
# 假设原始视频为 laptop.mp4
python -m hawor.inference \
    --input_video laptop.mp4 \
    --output_dir /home/an/data/hawor/laptop \
    --save_reconstruction  # 关键: 保存 reconstruction/
```

---

## Q35: 底座位置到底是怎么确定的？和 IK 有什么关系？

### 一句话概括

**底座位置不是 IK 算出来的，而是根据手腕轨迹的质心预先算好的。IK 是在底座位置确定之后，算"臂该怎么弯才能到达目标"。**

### 打个比方

想象你站在一个固定的位置（底座），伸手去够桌上的东西（手腕位置）：

```
你（底座）─── 手臂 ─── 手（目标位置）
   ↑                       ↑
 固定位置               由HaWoR给出
```

- **底座位置**：你站在哪？→ 根据桌上东西的位置，选一个"站着最舒服、伸手都能到"的位置
- **IK**：你手臂该怎么弯？→ 知道目标在哪、你站在哪，算出肩/肘/腕的角度

**底座位置是"站在哪"的问题，IK 是"怎么弯"的问题。先确定站在哪，再算怎么弯。**

### 底座位置的具体计算

```
第1步: 收集所有帧的手腕位置 (来自HaWoR)
  帧0: 手腕在 [0.30, -0.05, 0.15]
  帧1: 手腕在 [0.32, -0.04, 0.14]
  ...
  帧112: 手腕在 [0.28, -0.06, 0.16]

第2步: 算质心
  centroid = mean(所有手腕位置) = [0.31, -0.05, 0.15]

第3步: 底座放在质心正上方 0.35m
  base_pos = centroid + [0, 0, 0.35] = [0.31, -0.05, 0.50]
  
  为什么是上方0.35m？因为R1机器人是倒挂安装的（底座朝下），
  臂从底座向下伸展，0.35m大约是臂展的一半，是最舒适的工作距离。

第4步: 朝向绕Z轴旋转180°
  因为R1底座朝下安装，需要翻转180°让臂朝向操作区域。

第5步: 验证可达性
  最远手腕距离 = max(||手腕 - 底座||) = 0.41m
  臂展 = 0.713m
  0.41m < 0.713m × 0.9 = 0.64m → ✓ 可达
```

### 为什么不是根据 IK 确定底座位置？

**因为 IK 需要先知道底座位置才能求解。** IK 的输入是"目标在 base_link 坐标系中的位置"，base_link 坐标系的原点就是底座位置。如果底座位置不确定，IK 就没法算。

```
IK 的输入:
  ik_target_base = base_R⁻¹ @ (target_world - base_pos)
                              ↑               ↑
                         目标世界坐标      底座世界坐标

如果 base_pos 不知道，ik_target_base 就算不出来。
```

### 底座跟踪（±4cm 微调）

底座位置虽然预先算好了，但不是完全固定。每帧会根据当前手腕位置做小范围跟踪：

```
初始底座: [0.31, -0.05, 0.50]
当前手腕: [0.35, -0.05, 0.15]  ← 手腕偏右了4cm

1. 手腕在base_link坐标系中的位置:
   wrist_in_base = base_R⁻¹ @ (wrist - base) = [0.35, 0.0, -0.30]

2. 与舒适目标的偏移:
   COMFORT_TARGET_IN_BASE = [0.30, 0.0, -0.30]
   offset = [0.35-0.30, 0-0, -0.30-(-0.30)] = [0.05, 0, 0]

3. 限制偏移范围 (±4cm):
   clamped = clip(offset, -0.04, 0.04) = [0.04, 0, 0]

4. 底座向右移4cm:
   new_base = base + base_R @ [0.04, 0, 0] = [0.35, -0.05, 0.50]
```

**为什么需要跟踪？** 手腕在整个操作过程中可能移动很远（比如从桌子左边移到右边），如果底座完全固定，手腕可能移出臂的可达范围。跟踪让底座"跟着"手腕微调，保证手腕始终在舒适工作空间内。

### COMFORT_TARGET_IN_BASE 是什么？

`COMFORT_TARGET_IN_BASE = [0.30, 0.0, -0.30]` 是在底座坐标系（base_link）下，臂**最舒服**的工作点：

```
base_link 坐标系 (底座朝下):
  X轴: 臂前方
  Y轴: 臂侧方
  Z轴: 臂上方（但因为底座朝下，实际是向下）

COMFORT_TARGET_IN_BASE = [0.30, 0.0, -0.30]
  → 臂前方30cm、正中、下方30cm
  → 这是臂在这个位置时关节角度最自然、力矩最小的位置
```

**底座跟踪的目标就是让手腕始终在这个舒适点附近**。当手腕偏离舒适点时，底座就往那个方向微调，把手腕"拉回"舒适区域。

---

## Q36: 相机轨迹为什么会影响 GLB 对齐？不就是一个坐标变换吗？

### 一句话概括

**GLB 对齐确实就是一个坐标变换，但这个变换的参数（旋转、平移、缩放）需要从相机轨迹中计算出来。没有相机轨迹，就不知道该旋转多少、平移多少、缩放多少。**

### 通俗理解

想象两个画师分别画了同一间房间的画：

```
画师A (RAS): 画了一张房间俯视图，桌子在左边，椅子在右边
画师B (HaWoR): 画了同一间房间，但用的是正面视角，桌子在下方，椅子在上方
```

**问题**：怎么把画师A画的物体（桌子、椅子）放到画师B的画面中？

**答案**：需要知道三个东西：
1. **旋转**：画师A的"上"和画师B的"上"方向差多少？
2. **平移**：画师A的原点和画师B的原点差多远？
3. **缩放**：画师A的1厘米等于画师B的多少厘米？

**这三个参数从哪来？从相机轨迹来。**

### 相机轨迹是什么？

RAS 和 HaWoR 都处理了同一段视频，它们各自估计了"拍摄这段视频时相机在哪里、朝向哪里"：

```
RAS 估计: 帧0时相机在 (0, 0, 0)，朝向正前方
         帧1时相机在 (0.001, 0, 0)，朝向略偏右
         ...

HaWoR 估计: 帧0时相机在 (0.004, 0.004, 0.001)，朝向几乎正前方
            帧1时相机在 (0.005, 0.004, 0.001)，朝向略偏右
            ...
```

**关键事实**：RAS 和 HaWoR 处理的是同一个物理相机拍的同一段视频，所以相机的真实运动是一样的。但它们各自用不同的算法估计相机位置，导致坐标系不同。

### 从相机轨迹到对齐参数

#### 旋转和平移：第一帧相机锚定

```
同一个物理相机，在帧0时:
  RAS说: 相机在 (0, 0, 0)，朝向 R_ras
  HaWoR说: 相机在 (0.004, 0.004, 0.001)，朝向 R_hawor

因为RAS和HaWoR用的是不同的相机约定:
  RAS: OpenCV约定 (X=右, Y=下, Z=前)
  HaWoR: OpenGL约定 (X=右, Y=上, Z=后)

先统一约定: R_ras_opengl = R_ras @ OPENCV_TO_OPENGL

然后: R_align = R_hawor @ R_ras_opengl.T
      t_align = t_hawor - R_align @ t_ras

R_align 就是"RAS坐标系旋转多少能和HaWoR对齐"
t_align 就是"RAS原点平移多少能到HaWoR原点"
```

**为什么只用第一帧？** 因为第一帧最可靠——SLAM 的累积漂移还没开始，两个系统对第一帧的估计最接近真实值。

#### 缩放：Umeyama 尺度校正

```
同一个相机移动了同样的物理距离，但RAS和HaWoR的"刻度"不同:

RAS测量的相机移动范围: sigma_ras = 0.15 (RAS单位)
HaWoR测量的相机移动范围: sigma_hawor = 0.05 (米)

RAS单位 / 米 = sigma_ras / sigma_hawor = 0.15 / 0.05 = 3.0

所以: RAS的1个单位 = HaWoR的 1/3 米
      s_inv = 1/3 ≈ 0.32 (RAS→HaWoR的缩放因子)
```

**为什么需要缩放？** RAS 用的是 VGGT 深度估计，输出的是相对尺度（不知道1单位等于多少米）；HaWoR 用的是 Metric3D，输出的是米制尺度。同一个桌子，RAS 可能说是 1.5 单位宽，HaWoR 说 0.5 米宽。不缩放的话，桌子会大三倍。

### 完整的对齐公式

```
GLB顶点 (RAS坐标系) → HaWoR坐标系:

p_hawor = s_inv × R_inv × p_glb + t_inv

  s_inv: 缩放 (从相机轨迹离散度比算出)
  R_inv: 旋转 (从第一帧相机朝向算出)  
  t_inv: 平移 (从第一帧相机位置算出)
```

**就是一个坐标变换，但变换的参数必须从相机轨迹中计算。** 没有相机轨迹，就不知道参数，变换就做不了。

### 如果相机不动呢？

如果相机是固定的（比如固定在三脚架上拍摄），相机轨迹几乎是一条直线，离散度 sigma ≈ 0，缩放因子 s_inv 就算不准了（0/0 不确定）。

这时代码会回退到**启发式估算**：用手到 GLB 最近顶点的距离来估算尺度。

### 总结

| 对齐参数 | 从哪来 | 依赖什么 |
|----------|--------|----------|
| R_inv (旋转) | 第一帧相机朝向 | 两个系统的相机约定差异 |
| t_inv (平移) | 第一帧相机位置 | 两个系统的原点差异 |
| s_inv (缩放) | 相机轨迹离散度比 | 相机必须有运动，否则退化 |

**相机轨迹不是"影响"对齐，而是"提供"了对齐所需的参数。**

如果原始视频已丢失，则无法恢复核心数据，laptop 目录只能用于查看可视化。

---

## Q37: 能不能通过2D视频推断最优底座位置？让机械臂和人手臂在画面上对齐

### 核心需求

当前底座位置是启发式计算的（手腕质心正上方0.35m），没有考虑"机械臂在2D画面上是否和人的手臂对齐"。用户希望：

```
输入: 第一人称操作视频（人的手臂在画面中可见）
目标: 找到最优底座位置，使得机械臂渲染到画面上时，和人的手臂（肩→肘→腕）的位置/姿态最大程度一致
```

**注意：对比的是人的手臂（shoulder→elbow→wrist），不是人手（手指/夹爪）。** 这意味着需要检测/估计人在2D画面中的手臂骨架，然后让机械臂的骨架在画面上与之对齐。

### 关键澄清：HaWoR 已经提供什么，还需要什么

**HaWoR 已经提供**（**不需要再检测手部**）：
- ✅ 手腕 3D 位置 `pred_trans` (N, 3)
- ✅ 手腕 3D 朝向 `pred_rot` (N, 3) axis-angle
- ✅ 21 个手指 3D 关节点 `pred_joints_3d` (N, 21, 3)
- ✅ 相机参数 `R_c2w`, `t_c2w`, `img_focal`

**HaWoR 没有提供**（**这些才是真正需要的**）：
- ❌ 肩膀 3D/2D 位置
- ❌ 肘部 3D/2D 位置
- ❌ 上臂、前臂方向

**关键结论**：在 2D 视频中检测**肩/肘**才是核心需求，**绝不是为了检测手部**（HaWoR 已经把 3D 手部重建得很准了，再检测一次是浪费）。MediaPipe Pose、OpenPose 等方案都是**为了填补 HaWoR 没有的手臂信息**。

### 重新设计的 5 种方案（按推荐度排序）

#### 方案 1 (推荐) : VGGT-Omega 附带人体关键点 (零额外依赖)

7_vggt-omega 目录中多出 `vggt_omega_cam/` 目录，说明 VGGT-Omega 重建的不仅是手部，还包括相机+场景。

**关键检查**: VGGT-Omega 是否输出人体 3D 关键点？

如果 VGGT-Omega 的输出包含人体肩/肘关键点（很多 VGGT 衍生模型都支持人体 3D 重建，如 SMPLer-X/VIBE 等）：

```python
rec = np.load('7_vggt-omega/vggt_omega_cam/vggt_omega_cam.npz')
# 假设有 human_joints_3d: (N, 17, 3)  17 个 SMPL 关键点
shoulder_3d = rec['human_joints_3d'][:, 12]  # 右肩 SMPL index 12
elbow_3d = rec['human_joints_3d'][:, 14]     # 右肘
wrist_3d = rec['human_joints_3d'][:, 16]     # 右腕 (可与 HaWoR 校准)
```

**优点**:
- 零外部依赖，与现有管线完美兼容
- 3D 关键点 → 投影到 2D → 直接和机械臂 2D 骨架对齐
- 包含深度信息，不受第一人称视角影响

**缺点**:
- 需要查 VGGT-Omega 是否真支持人体 3D 关键点输出
- 需验证 7_vggt-omega 数据集中是否真的包含这些字段

**优先级**: 最高 — 先验证此方案，如果可行就直接用，省下所有 2D 检测的工作

#### 方案 2 (次选) : MediaPipe Pose 检测 2D 肩/肘 (简单可靠)

如果方案 1 不可行，退而求其次用 MediaPipe Pose 检测 2D 关键点。

MediaPipe Pose 可检测 33 个关键点，手臂相关：
```
左臂: left_shoulder(11) → left_elbow(13) → left_wrist(15)
右臂: right_shoulder(12) → right_elbow(14) → right_wrist(16)
```

```python
import mediapipe as mp
mp_pose = mp.solutions.pose.Pose(static_image_mode=False)
results = mp_pose.process(frame)
if results.pose_landmarks:
    r_shoulder = results.pose_landmarks.landmark[12]  # 右肩 (注意: 不是检测手)
    r_elbow = results.pose_landmarks.landmark[14]     # 右肘
    r_wrist = results.pose_landmarks.landmark[16]     # 右腕 (2D)
```

**关键点**:
- MediaPipe 检测的 `wrist` (2D) 可以**和 HaWoR 投影的 wrist 2D 做校准**
- 用 HaWoR 的 `pred_trans` + 相机内参投影作为更精确的 wrist 2D
- 肩/肘 2D 仍然用 MediaPipe（HaWoR 没有这俩）

**优点**: 实时 (30+ FPS)、无需 GPU、Python API 简单
**缺点**: 第一人称视角下肩膀经常被裁剪出画面；2D 坐标无深度

**开源**: https://google.github.io/mediapipe/solutions/pose.html

#### 方案 3 (备选) : 物理约束 + 人体运动学 (零外部依赖)

如果不想引入 MediaPipe，可用人体运动学模型从手腕 3D 推算肩/肘 3D：

```python
# 假设
FOREARM_LEN = 0.26  # 前臂长度 (成人平均)
UPPERARM_LEN = 0.33  # 上臂长度
ELBOW_BEND_ANGLE = np.deg2rad(90)  # 肘部弯曲角 (假设)

# 从手腕 3D + 手腕朝向 推算
wrist_3d = hawor_pred_trans[frame]      # HaWoR 已知
wrist_rot = hawor_pred_rot[frame]       # HaWoR 已知
# 手腕到肘方向: 沿手腕局部 -X 方向延伸 FOREARM_LEN
# 肘到肩方向: 沿重力反方向延伸 UPPERARM_LEN
```

**优点**: 零外部依赖
**缺点**: 假设的臂长/弯曲角不准确；不同人差异大

#### 方案 4 (不推荐) : OpenPose

CMU OpenPose 可检测 25 个关键点，包括手臂。比 MediaPipe 更准确，但速度慢、依赖多。

**开源**: https://github.com/CMU-Perceptual-Computing-Lab/openpose
**不推荐原因**: 重型依赖、第一人称视频优势不明显

#### 方案 5 (高级) : UpperLimbs 3D 上肢追踪

专门用于单目摄像头的 3D 上肢追踪，基于 MediaPipe，输出统一的 3D 坐标。

**开源**: https://github.com/sthasmn/UpperLimbs
**不推荐原因**: 长期方案，集成成本高，短期不必要

### 2D 手臂对齐优化底座位置（基于方案 1 或 2）

#### 核心思路

```
1. 获取人手臂 3D 骨架 (肩→肘→腕): 来自方案1 (VGGT) 或 HaWoR + 方案2 (MediaPipe)
2. 投影到 2D:  P_shoulder_2D, P_elbow_2D, P_wrist_2D
3. 对每个候选底座位置:
   a. IK 求解 → 机械臂关节角
   b. FK → 机械臂 3D 骨架 (肩→肘→腕)
   c. 投影到 2D → 机械臂 2D 骨架
   d. 计算 2D 骨架距离
4. 选距离最小的底座位置
```

#### 2D 骨架距离度量

```
人手臂 2D 骨架:  P_shoulder → P_elbow → P_wrist
机械臂 2D 骨架:  R_shoulder → R_elbow → R_wrist

距离 = w_shoulder * ||P_shoulder - R_shoulder||
     + w_elbow    * ||P_elbow    - R_elbow||
     + w_wrist    * ||P_wrist    - R_wrist||
     + λ          * |angle(P_elbow) - angle(R_elbow)|  ← 肘部弯曲角度
```

**权重推荐**: `w_shoulder=1.0, w_elbow=1.5, w_wrist=2.0` (末端权重更高)

**为什么加肘部角度？** 如果只对齐关键点位置，可能出现"关键点位置对了但肘弯方向反了"的情况。

#### 第一人称视角的特殊问题

在第一人称视频中，人的肩膀经常**被裁剪出画面**，只有肘部和手腕可见。这时：

```
方案 A: 只用可见关键点
  - 肩膀权重设为 0，只用肘+腕计算距离
方案 B: 用 VGGT-Omega 的 3D 肩位置 (方案1优势)
方案 C: 用人体运动学推算 (方案3)
```

#### 完整算法流程（推荐版）

```python
def optimize_base_with_arm_alignment(self, video_frames, camera_intrinsics):
    # Step 1: 获取人手臂 3D 骨架 (按方案优先级)
    human_arm_3d = []
    if HAS_VGGT_HUMAN_KEYPOINTS:  # 方案1
        rec = np.load('vggt_omega_cam.npz')
        human_arm_3d = rec['human_joints_3d'][:, [12, 14, 16]]  # 肩肘腕
    elif HAS_MEDIAPIPE:  # 方案2
        # MediaPipe 给出 2D, 用 HaWoR 深度 + 三角化得到 3D
        mp_pose = mp.solutions.pose.Pose()
        for frame in video_frames:
            results = mp_pose.process(frame)
            landmarks = results.pose_landmarks.landmark
            human_arm_3d.append({
                'shoulder': np.array([landmarks[12].x, landmarks[12].y]),
                'elbow':    np.array([landmarks[14].x, landmarks[14].y]),
                'wrist':    np.array([landmarks[16].x, landmarks[16].y]),
                'shoulder_visible': landmarks[12].visibility > 0.5,
            })
    else:  # 方案3: 物理约束
        # 用 HaWoR wrist_3D + 人体比例推算
        ...
    
    # Step 2: 投影到 2D (对方案1,3) 或直接用 2D (对方案2)
    human_arm_2d = []
    for frame_data in human_arm_3d:
        if 'shoulder_3d' in frame_data:
            # 方案1,3: 3D→2D
            K = camera_intrinsics
            P_shoulder_2d = K @ frame_data['shoulder_3d']  # 简化
            P_elbow_2d = K @ frame_data['elbow_3d']
            P_wrist_2d = K @ frame_data['wrist_3d']
        else:
            P_shoulder_2d = frame_data['shoulder']
            P_elbow_2d = frame_data['elbow']
            P_wrist_2d = frame_data['wrist']
        human_arm_2d.append({...})
    
    # Step 3: 网格搜索候选底座位置
    centroid = np.mean(hawor_pred_trans, axis=0)  # 手腕质心
    best_score = float('inf')
    best_base = None
    
    for dx in np.linspace(-0.2, 0.2, 9):
        for dy in np.linspace(-0.2, 0.2, 9):
            for dz in np.linspace(-0.1, 0.3, 5):
                base_pos = centroid + [dx, dy, dz + 0.35]
                
                total_error = 0
                ik_fail_count = 0
                for i, wrist_pos in enumerate(hawor_pred_trans):
                    ik_target = base_R.T @ (wrist_pos - base_pos)
                    arm_joints = relaxed_ik.solve(ik_target, ee_quat)
                    if arm_joints is None:
                        ik_fail_count += 1
                        continue
                    
                    # FK 机械臂 3D 骨架 → 投影到 2D
                    robot_shoulder_3d, robot_elbow_3d, robot_wrist_3d = \
                        forward_kinematics(arm_joints, base_pos, base_R)
                    R_s_2d = K @ robot_shoulder_3d
                    R_e_2d = K @ robot_elbow_3d
                    R_w_2d = K @ robot_wrist_3d
                    
                    # 2D 骨架距离
                    human = human_arm_2d[i]
                    error = 0
                    if human['shoulder_visible']:
                        error += 1.0 * np.linalg.norm(human['shoulder_2d'] - R_s_2d[:2])
                    error += 1.5 * np.linalg.norm(human['elbow_2d'] - R_e_2d[:2])
                    error += 2.0 * np.linalg.norm(human['wrist_2d'] - R_w_2d[:2])
                    total_error += error
                
                score = ik_fail_count * 100 + total_error
                if score < best_score:
                    best_score = score
                    best_base = base_pos
    
    return best_base
```

### 方案对比 (更新版)

| 方案 | 检测部位 | 3D信息 | 2D对齐 | 3D可达 | 外部依赖 | 推荐度 |
|------|---------|--------|--------|--------|---------|--------|
| **1. VGGT-Omega** | **肩→肘→腕 (3D)** | **✅** | **✅** | **❌** | **零** | **⭐⭐⭐ 最高** |
| 2. MediaPipe Pose | 肩→肘→腕 (2D) | ❌ | ✅ | ❌ | mediapipe | ⭐⭐ 次选 |
| 3. 物理约束 | 肩→肘→腕 (3D 推算) | 估算 | ✅ | ❌ | 零 | ⭐ 备选 |
| 4. OpenPose | 肩→肘→腕 (2D) | ❌ | ✅ | ❌ | 重型 | ⭐ 不推荐 |
| 5. UpperLimbs | 肩→肘→腕 (3D) | ✅ | ✅ | ❌ | mediapipe+ | ⭐⭐⭐ 长期 |
| **6. 方案1+B\*** | **肩→肘→腕 (3D)** | **✅** | **✅** | **✅** | **零** | **⭐⭐⭐ 最高** |

### 推荐路线 (重写)

**Step 1 (立即)**: 验证方案 1 (VGGT-Omega 是否有 human_joints_3d)
- 检查 `/home/an/data/hawor/7_vggt-omega/vggt_omega_cam/vggt_omega_cam.npz` 的所有字段
- 如果有 `*joint*` 或 `*smpl*` 字段，直接用方案 1
- 工作量: 0.5 天 (纯调研)

**Step 2 (短期, 1-2 天)**: 方案 2 兜底
- 用 MediaPipe Pose 检测 2D 肩/肘
- 用 HaWoR 投影的 2D wrist 做校准
- 集成到 02_render_scene.py 作为可选步骤

**Step 3 (中期, 1-2 周)**: 方案 6 (2D骨架 + 3D可达联合优化)
- 方案 1/2 + B* 优化器 (3D 可达性约束)

**Step 4 (长期)**: 集成 UpperLimbs 做完整 3D 上肢追踪

### 关键依赖

| 方案 | 依赖 | 安装 |
|------|------|------|
| 1 (VGGT) | 零 | 已含 |
| 2 (MediaPipe) | mediapipe | `pip install mediapipe` |
| 3 (物理约束) | 零 | - |
| 6 (B* 联合) | B* | https://github.com/leiyaocui/B_STAR |
| 5 (UpperLimbs) | mediapipe + | https://github.com/sthasmn/UpperLimbs |

---

## Q35: 04 中 GLB 物体支撑点/位置管理

**日期**: 2026-06-09
**分类**: 调试

### 问题
用户反馈："glb 支撑点都一样吗？最好先固定一个位置，等待机械臂的交互"

### 解答
GLB 物体在 04 中默认是 `kinematic` 类型（不会被重力掉落），初始位置硬编码为 `(0, 0, 0)`（与 02 一致）。
GLB 顶点已经在 SAPIEN 坐标系中经过变换（通过临时 PLY 文件），所以 actor pose 为 (0,0,0) 是正确的。

修复了 04 中两趟渲染物体初始化的 bug：
- 旧代码用 `initial_obj_poses[-1]` 把所有物体都设置到最后一个的位置
- 修复后用 zip 循环分别记录和重置每个物体的 pose

用户可以执行 `bash physics_pipeline/rerender.sh demo` 直接在交互式 viewer 中查看 GLB 物体位置。

---

## Q36: 04 中"机械臂乱动"的潜在 bug

**日期**: 2026-06-09
**分类**: 调试

### 问题
用户反馈："关节和 glb 的映射以 02 为主是对的，这些抖动问题都应该解决，我觉得可能是有一些 bug 你没有解决，会导致机械臂乱动"

### 解答
定位并修复了以下 bug：

#### Bug 1: GLB 物体初始化位置覆盖
原代码：
```python
for actor in obj_actors:
    actor.set_pose(sapien.Pose(initial_obj_poses[-1][0].tolist(), initial_obj_poses[-1][1].tolist()))
```
把所有 GLB 物体都设置到最后一个物体的位置。修复后删除该行（每帧用 zip 重置）。

#### Bug 2: viewer 模式使用 PD 控制导致抖动
viewer 模式 (`self.viewer=True`) 之前用 `_physics_step` (高刚度PD+多步scene.step)，会引入 PD 震荡。
修复后 viewer 模式用 `set_drive_target + set_qpos + 单次 scene.step`（与 02_render_scene.py 完全一致），避免 PD 震荡。

#### Bug 3: 第二趟渲染物体初始化 bug（已修复）

### 验证
1. **离线数据诊断** (`diagnose_data.py`): 0个>10°的跳变点
2. **端到端模拟** (`end_to_end_test.py`): 最大跳变 2.87°（↓97.4%）
3. **新 demo 模式**: `bash physics_pipeline/rerender.sh demo` 可直接在 viewer 中查看仿真效果

### 用法
```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination
bash physics_pipeline/rerender.sh demo     # 交互式 3D 查看器
bash physics_pipeline/rerender.sh video    # 两趟渲染视频
bash physics_pipeline/rerender.sh render   # 单趟渲染（快速预览）
bash physics_pipeline/rerender.sh dataset  # 渲染 + 数据集生成
```

