# 2026-06-26 04_physics_simulation.py GLB 对齐与夹爪对称修复设计 (v2 基于实际代码)

> 范围: `04_physics_simulation.py` 微调, 不重写, 不加双手支持
> 基准: `02_render_scene.py` 源代码 (但 02 也有同样的夹爪负值 bug, 本次修正 04)
> 目标: 修复 GLB 摆放位置错乱 + 夹爪只开合一边 + 手部检测返回类型

---

## 一、问题诊断 (基于实际读取 URDF 和源码)

### 根因 1: GLB 加载缺少 Z-UP 坐标系检测 (GLB 对齐问题主因)

`04_physics_simulation.py` 的 `load_glb_with_physics` (line 878-1098) 直接应用 `s_inv * (R_inv @ vertices.T).T + t_inv`, 没有:
- 自动检测 GLB 坐标系 (Z-UP vs Y-UP)
- 必要时应用 `ZUP_TO_YUP` 转换
- 读取 transform_params 中保存的 `glb_up_axis`

而 `02_render_scene.py` 的 `load_glb_transformed` (line 902-1022) 有完整的 `_detect_glb_up_axis` 启发式检测 + `ZUP_TO_YUP` 转换。

**影响**: 若 GLB 是 Z-UP (RAS 导出常见), 04 把地板当墙处理, 物体平躺 → "GLB 摆放位置被更改"。

### 根因 2: 夹爪 finger joint 负值违反 URDF limit + 符号错误 (只开合一边主因)

**通过读取 URDF 文件确认** (非猜测):

`r1_v2_1_0_floating_right.urdf` line 541-558:
```xml
<joint name="right_gripper_finger_joint1" type="fixed">  ← _prepare_arm_urdf 改为 prismatic
  <axis xyz="0 -1 0" />      ← joint1 axis
  <limit lower="0" upper="0.05" .../>  ← 非负!
</joint>
<joint name="right_gripper_finger_joint2" type="fixed">
  <axis xyz="0 1 0" />       ← joint2 axis (与 joint1 相反!)
  <limit lower="0" upper="0.05" .../>  ← 非负!
</joint>
```

`r1_v2_1_0_floating_left.urdf` line 542-610:
```xml
<joint name="left_gripper_finger_joint1" type="prismatic">
  <axis xyz="0 1 0" />        ← joint1 axis (与右臂相反)
  <limit lower="0" upper="0.05" .../>
</joint>
<joint name="left_gripper_finger_joint2" type="prismatic">
  <axis xyz="0 -1 0" />       ← joint2 axis (与 joint1 相反)
  <limit lower="0" upper="0.05" .../>
</joint>
```

**关键事实**:
1. 两个 finger joint 的 **axis 相反** (一个 +Y 一个 -Y)
2. limit 都是 **[0, 0.05] 非负**
3. 因此两个 joint 必须用**同号值**才会对称开合:
   - joint1=0.04, axis=(0,-1,0) → finger1 向 -Y 移动 0.04
   - joint2=0.04, axis=(0, 1,0) → finger2 向 +Y 移动 0.04  ← 对称 ✓
4. 若用异号 (joint1=0.04, joint2=-0.04):
   - joint2=-0.04, axis=(0,1,0) → finger2 向 -Y 移动 0.04 (负值取反了 axis)
   - **两个 finger 都向 -Y 移动 → 只开合一边!**

**04 代码的 BUG** (5 处负值):
- line 1779: `init_qpos[gripper_idx2] = -0.04`
- line 3015: `init_qpos[gripper_idx2] = -0.04`
- line 3039: `reset_qpos[gripper_idx2] = -0.04`
- line 3065: `init_qpos[gripper_idx2] = -0.04`
- line 3162: `reset_qpos[gripper_idx2] = -0.04`
- line 2913: `gripper_target2 = ... else -0.04` (默认值)
- line 3098: `gripper_t2 = ... else -0.04` (默认值)

**注**: 02_render_scene.py line 2360 也有 `init_qpos[gripper_idx2] = -0.04`, 但 02 用 `set_qpos` (运动学), SAPIEN 会 clamp 到 limit [0, 0.05], 实际等价于 0; 而 04 用 PD 驱动, `set_drive_target(-0.04)` 会持续施加力拉向 -0.04, 违反 limit 导致震荡/只开合一边。

**注2**: 单夹爪模式的 `_compute_analytical_gripper_pose` (line 307-308) 已经是同号 `joint1 = joint2 = required_open_sum / 2`, **不需要修改**。

### 根因 3: 手部检测返回单个 idx, 不支持双手

`_detect_hand_idx` (line 455-483) 只返回单个 int (0 或 1), 无法表达"双手"。用户要求"手部自动识别 左/右/双手"。但用户已确认**不加双手支持**, 所以只需改进返回类型让调用方能区分左/右/双手 (即使双手时只用一只)。

---

## 二、修复方案 (最小改动, 仅修复现有右臂)

### 修复 1: 添加 ZUP_TO_YUP 常量 + _detect_glb_up_axis 函数

**位置**: 常量区 (line 95 附近)

**插入**:
```python
# GLB 坐标系转换: Z-UP (RAS 导出常见) → Y-UP (SAPIEN 标准)
ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def _detect_glb_up_axis(all_vertices):
    """检测 GLB 坐标系是 Z-UP 还是 Y-UP (复制自 02_render_scene.py)."""
    FLOOR_THRESHOLD = 0.1
    min_z = all_vertices[:, 2].min()
    min_y = all_vertices[:, 1].min()
    z_is_floor = abs(min_z) < FLOOR_THRESHOLD
    y_is_floor = abs(min_y) < FLOOR_THRESHOLD
    if z_is_floor and not y_is_floor:
        return "z-up"
    if y_is_floor and not z_is_floor:
        return "y-up"
    if z_is_floor and y_is_floor:
        z_at_floor = (abs(all_vertices[:, 2]) < FLOOR_THRESHOLD).sum()
        y_at_floor = (abs(all_vertices[:, 1]) < FLOOR_THRESHOLD).sum()
        return "z-up" if z_at_floor > y_at_floor else "y-up"
    return "y-up"
```

### 修复 2: load_glb_with_physics 添加 Z-UP 检测

**位置**: `load_glb_with_physics` 函数 (line 878+)

**改动**:
1. 读取 `saved_glb_up_axis` from transform_params
2. 自动检测 GLB 坐标系
3. 必要时应用 `ZUP_TO_YUP` 转换 (在 `vertices_hawor` 计算之前)

### 修复 3: 夹爪统一为单控制量 (用户洞察: "夹爪应该只有一个变量控制开合")

**原理**: URDF axis 相反 + limit 非负 → 必须同号才对称。

**改动 3a** (5 处初始/重置姿态): `gripper_idx2 = -0.04` → `gripper_idx2 = 0.04`
- line 1779, 3015, 3039, 3065, 3162

**改动 3b** (retargeting 输出, line 2912-2913):
```python
# 修改前
gripper_target1 = float(sapien_qpos[gripper_idx1]) if gripper_idx1 < len(sapien_qpos) else 0.04
gripper_target2 = float(sapien_qpos[gripper_idx2]) if gripper_idx2 < len(sapien_qpos) else -0.04

# 修改后 (单控制量 + clamp)
gripper_target1 = float(sapien_qpos[gripper_idx1]) if gripper_idx1 < len(sapien_qpos) else 0.04
gripper_target1 = max(0.0, min(0.05, gripper_target1))  # clamp 到 URDF limit
gripper_target2 = gripper_target1  # 同号, axis 相反, 对称开合
```

**改动 3c** (第二趟 PD 驱动, line 3097-3098):
```python
# 修改前
gripper_t1 = float(target_qpos[gripper_idx1]) if gripper_idx1 < len(target_qpos) else 0.04
gripper_t2 = float(target_qpos[gripper_idx2]) if gripper_idx2 < len(target_qpos) else -0.04

# 修改后
gripper_t1 = float(target_qpos[gripper_idx1]) if gripper_idx1 < len(target_qpos) else 0.04
gripper_t1 = max(0.0, min(0.05, gripper_t1))
gripper_t2 = gripper_t1
```

**改动 3d** (warmup 调用, line 2823, 3060, 3180): `0.04, -0.04` → `0.04, 0.04`

**改动 3e** (注释, line 1744): `gripper=[0.04, -0.04]` → `gripper=[0.04, 0.04]`

### 修复 4: 手部检测返回类型改进 (支持左/右/双手识别, 但不加双手渲染)

**改动**: 增强 `_detect_hand_idx` 返回 (idx, handedness_str), 让调用方日志能显示"左手/右手/双手":

```python
def _detect_hand_idx(hawor_path):
    """自动检测 HaWoR 数据中活跃的手 (改进: 返回 handedness 字符串)

    Returns:
        tuple: (hand_idx, handedness_str)
            hand_idx: int (0=左手, 1=右手) - 用于现有单臂渲染
            handedness_str: "left" / "right" / "both" / "unknown"
    """
    cam_dir = Path(hawor_path) / "cam_space"
    detected = set()
    if cam_dir.exists():
        for d in cam_dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                detected.add(int(d.name))

    if 0 in detected and 1 in detected:
        return 0, "both"   # 双手数据, 单臂渲染默认用左手
    if 1 in detected:
        return 1, "right"
    if 0 in detected:
        return 0, "left"

    # 回退: 通过 npz pred_valid 检测
    rec_file = _find_reconstruction_file(Path(hawor_path))
    if rec_file is not None:
        rec = np.load(str(rec_file), allow_pickle=True)
        if 'pred_valid' in rec:
            pred_valid = rec['pred_valid']
            if pred_valid.ndim == 2 and pred_valid.shape[0] >= 2:
                left_active = pred_valid[0].any()
                right_active = pred_valid[1].any()
                if left_active and right_active:
                    return 0, "both"
                if right_active:
                    return 1, "right"
                if left_active:
                    return 0, "left"
    return 0, "unknown"
```

**调用方适配**: 现有调用 `hand_idx = _detect_hand_idx(...)` 改为 `hand_idx, handedness = _detect_hand_idx(...)` 并记录日志。

---

## 三、不在本次范围

1. **双手渲染支持** (用户已确认不加)
2. **50fps 性能优化** (本次只添加测量基线, 不优化)
3. **重力参数调整** (用户禁止)
4. **X/Y 轴坐标修改** (用户禁止)
5. **02_render_scene.py 的同 bug** (本次只修 04, 02 留待后续)

---

## 四、验证策略

### 4.1 静态验证
```bash
python -c "
import ast
src = open('04_physics_simulation.py').read()
ast.parse(src)
checks = [
    ('ZUP_TO_YUP 添加', 'ZUP_TO_YUP' in src),
    ('_detect_glb_up_axis 添加', '_detect_glb_up_axis' in src),
    ('saved_glb_up_axis 添加', 'saved_glb_up_axis' in src),
    ('need_zup_to_yup 添加', 'need_zup_to_yup' in src),
    ('夹爪无负值 -0.04', '-0.04' not in src),
    ('夹爪 gripper_target2 镜像', 'gripper_target2 = gripper_target1' in src),
    ('夹爪 gripper_t2 镜像', 'gripper_t2 = gripper_t1' in src),
    ('夹爪 clamp', 'max(0.0, min(0.05, gripper_target1))' in src),
    ('手部检测返回 handedness', '\"both\"' in src and '\"left\"' in src and '\"right\"' in src),
]
for name, ok in checks:
    print(f'  {\"✓\" if ok else \"✗\"} {name}')
"
```

### 4.2 动态验证 (需 GPU, 由用户执行)
1. 运行 04, 检查 GLB 物体位置与 02 一致
2. 检查夹爪两个 finger 对称开合
3. 检查日志显示 "左手/右手/双手"

---

## 五、用户已确认决策

1. ✅ 仅修复现有右臂, 不加双手支持
2. ✅ 夹爪统一为单控制量 (用户洞察: "夹爪应该只有一个变量控制开合")
3. ✅ URDF axis 相反 + limit 非负是根因 (用户怀疑 URDF, 已通过实际读取 URDF 确认)
