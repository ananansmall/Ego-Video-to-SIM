# MANO 跟随重设计 (第十三轮)

> 日期: 2026-06-30
> 状态: 待用户审阅 (已根据用户第二次澄清更新)
> 上游: 用户 Round 13 反馈 (4 项问题) + 第二次澄清

## 1. 背景

用户在观看 Round 12 测试视频后提出 4 项关键反馈:

1. **物理限制不匹配** — 夹爪是否超过物理限制? 是否和 `/home/an/robot_world_ws/src/GalaxeaManipSim` 一样的约束?
2. **god 相机未下降** — 运行默认命令 (不带 `--views god`) 时没感觉到 god 相机下降; 第二次澄清: "固定 0.2m 还是很高, 不知道为什么, 是坐标系的问题吗?"
3. **没有真正夹住** — 视频像抓娃娃机, 物体最终掉下来; verify.json 报告"真正夹住"是误报; 要求把碰撞模型也展示到视频里
4. **轨迹偏离严重** (最关键) — 夹爪完全不跟随 MANO 参考点, 当前用固定 top-down 朝向, 偏离非常大; 要求: 必须跟随 MANO 位置+姿态, 只能微调位置不改姿态, 取最小偏移, 完成抓取后结束

**用户第二次澄清 (最关键)**:
- 轨迹策略: "不是 0 偏移, 是位姿不改变, 轨迹可以有偏移, 你要在抓取物体的轨迹和现在的轨迹进行优化, 把损失降到最小"
- 含义: 姿态 (orientation) **必须**跟随 MANO root_R (位姿不改变); 位置 (position) **可以**有偏移; 但偏移要**最小化** (在保证抓取成功的前提下)
- 碰撞可视化颜色: 半透明红色 RGBA[1,0,0,0.4] (用户选第一个选项)
- god 相机: 固定, 但 0.2m 还是偏高, 怀疑坐标系问题

## 2. 诊断 (已查证数据)

### 2.1 物理参数对比

| 参数 | Galaxea R1 (参考标准) | 我的实现 | 状态 |
|------|---------------------|---------|------|
| 手指碰撞体 | MESH (STL, 完整手指) | box 12×20×24mm | ❌ 太小 (体积约 1/5, **唯一物理不匹配**) |
| static/dynamic friction | 1.0 / 1.0 | 2.0 / 2.0 | ⚠ 偏高 (黏滞) |
| restitution | 0.6 | 0.0 | ⚠ 无弹性 |
| 关节限位 | 0~0.05m, effort=100, velocity=0.25 | 同 | ✓ 一致 |
| 手指质量 | 0.027 kg | 0.027 kg | ✓ **已对齐** (URDF L330, L347) |
| 手指惯量 | ixx=2.4057e-6 等 | 同 | ✓ **已对齐** (URDF L331-332) |
| PD 阻尼 | stiffness=1000, damping=200 | 同 | ✓ 一致 |

**结论**: 物理限制**没有**超过 R1 (关节限位/质量/惯量/PD 都一致), **唯一不匹配的是手指碰撞体** (box vs mesh). 用户问"是不是超过物理限制", 答案: 没有, 但碰撞体太小导致夹不住.

参考文件: `/home/an/robot_world_ws/src/GalaxeaManipSim/galaxea_sim/assets/r1/meshes/r1_mjcf_usf87pxx/r1_gripper_only_right.urdf`

### 2.2 god 相机 (确认是坐标系问题, 已查证)

- L3264: `god_height = 0.20` 已生效
- L3269: `god_pos = scene_center + np.array([0.0, 0.0, god_height])`
- **关键**: `scene_center = (arm_base_pos + obj_centroid) / 2.0` (L3242)
  - `arm_base_pos` z ≈ 0.35 (臂基座高度)
  - `obj_centroid` z ≈ 0.02 (物体在地面上)
  - → `scene_center_z` ≈ (0.35 + 0.02) / 2 = **0.185m**
- → `god_pos_z` = 0.185 + 0.20 = **0.385m** (不是 0.2m!)
- **用户感知"0.2m 还是很高"是正确的** — 实际相机在 0.385m, 比用户期望的 0.2m 高出近一倍

**修复**: `god_pos_z` 应该用**绝对坐标** (相对地面), 不是相对 scene_center:
- 方案 A (推荐): `god_pos = np.array([scene_center[0], scene_center[1], ground_z + god_height])`
  - ground_z = 物体最低 z (≈0), god_pos_z = 0 + 0.20 = 0.20m (用户期望)
- 方案 B: `god_pos = np.array([scene_center[0], scene_center[1], max_obj_z + god_height])`
  - max_obj_z = 最高物体顶 z, 确保能看到所有物体

### 2.3 verify.json "真正夹住" 误报 (核心矛盾)

代码 `_evaluate_grasp_quality` (L3804-3913) 三项判据都有问题:

| 判据 | 当前实现 | 漏洞 |
|------|---------|------|
| 接触 | 全局 `n_contacts >= 2` 持续 10 帧 | ❌ 全局非 per-object, glb_1 的"61 帧"可能来自其他物体; `n_contacts>=2` 不等于"对向夹紧" |
| 跟随 | `\|\|Δ(obj_p - gripper_p)\|\| < 5cm` 持续 10 帧 | ❌ 物体坐在闭合夹爪上方也会通过 (娃娃机式) |
| 提升 | `max(z) - init_z > 5cm` | ❌ 用 max 不是 final; glb_1 init_z=0.017=final_z, 但 max=17.07cm → 托起又掉 |

JSON 证据: glb_1 initial_pos=[0.033,-0.151,0.017], final_pos=[0.147,-0.297,**0.017**] — z 完全相同, 物体掉回原位.

### 2.4 轨迹偏离 (最关键)

- `compute_analytical_gripper_pose` (L1062-1116) **返回了 root_R** (MANO 真实手腕朝向, 加权 SVD Procrustes)
- 但 `_compute_mano_neutral_target` (L2224-2360) **忽略 root_R**, 用固定的 `gripper_R_fixed = [[0,1,0],[0,0,-1],[-1,0,0]]` (top-down)
- 当前 offset = `target_grasp_pos - mano_pos[f_grasp]` 可能达 5cm+, 用户说"偏离非常多"
- `mano_gripper_traj` 当前只存 `{"pos", "j1", "j2"}`, **没存 R** — 需要扩展

## 3. 设计决策 (用户已确认 + 第二次澄清修正)

| 决策 | 选择 | 理由 |
|------|------|------|
| 轨迹策略 | **位姿不改变 + 位置优化最小损失** | 姿态**必须**跟 MANO root_R (位姿不改变); 位置**可以**有偏移但需最小化 (在抓取轨迹和当前轨迹间优化, 把损失降到最小). 不是零偏移, 是优化最小偏移. |
| 成功标准 | **A: 夹住+不掉** | 末段 10 帧: 双指对向接触 + z 稳定 (方差<1cm) + final_z - init_z > 3cm |
| 碰撞可视化 | **半透明红色 RGBA[1,0,0,0.4]** | 在视觉模型外覆盖半透明红色碰撞体, 跟随手指位姿, 不影响物理 (用户选第一个选项) |
| 物理对齐 | **手指碰撞改 mesh + friction/restitution 对齐** | 手指碰撞改 mesh(STL); friction 1.0/1.0; restitution 0.6; mass/inertia/PD **已对齐无需改** |
| god 相机 | **绝对坐标 0.2m** | `god_pos_z = ground_z + 0.20` (不是 scene_center_z + 0.20); 解决 0.385m 偏高问题 |

## 4. 实施方案

### 4.1 物理参数对齐 R1 (Issue 1)

**文件**: `grasp_hawor.py` `prepare_gripper_only_urdf` (L291-367, URDF 模板) + 物理参数区 (L123-137) + load_glb_with_physics 接触材质 (L529-532 附近)

**改动 (仅碰撞体 + 摩擦/弹性, mass/inertia 已对齐)**:
1. URDF L336-337: `finger_link1` collision 从 `<box size="0.012 0.020 0.024"/>` 改为 `<mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/>`
2. URDF L354-355: `finger_link2` collision 同样改为 mesh
3. 全局参数 L134: `GRIPPER_FRICTION = 1.0` (从 2.0 降低, 对齐 R1)
4. 物体材质: `restitution = 0.6` (从 0.0 改, 对齐 R1; 物体弹性可让夹爪闭合时更稳定)
5. 接触材质 `set_friction`: `static=1.0, dynamic=1.0` (对齐 R1)

**不改**: mass=0.027, inertia, PD stiffness/damping, 关节限位 — 这些已经和 R1 一致.

**验证**: 加载后日志打印 finger collision 类型, 确认是 mesh.

### 4.2 god 相机坐标系修复 (Issue 2)

**文件**: `grasp_hawor.py` L3269

**改动** (从相对 scene_center 改为绝对地面坐标):
```python
# 旧 (错误):
god_pos = scene_center + np.array([0.0, 0.0, god_height])  # god_pos_z ≈ 0.385m

# 新 (正确):
ground_z = min(obj_centers[:, 2])  # 物体最低 z ≈ 0
god_pos = np.array([scene_center[0], scene_center[1], ground_z + god_height])  # god_pos_z = 0.20m
```

**验证**: 日志打印 god_pos, 确认 z ≈ 0.20m (不是 0.385m).

### 4.3 验证逻辑修复 (Issue 3)

**文件**: `grasp_hawor.py` `_evaluate_grasp_quality` (L3804-3913) + 主循环接触记录

**改动**:
1. **接触 per-object 化**: 主循环记录 `contact_log[obj_name] = [n_contacts_per_frame]`, 不是全局
   - 用 `contact[0]` 和 `contact[1]` 的 actor 名匹配物体
2. **对向夹紧检测**: 检查两指接触点的法向是否对向 (点积 < 0), 不是同侧推
3. **提升判据改 final**: `lift = final_z - init_z` (不是 max), 阈值 3cm (从 5cm 降低, 因为不再要求 max)
4. **跟随加 z 稳定性**: 末段 10 帧物体 z 方差 < 1cm (排除"托起又掉")
5. **新增"末段夹紧维持"**: 末段 10 帧持续 `n_contacts_per_object >= 2` 且 z 方差小

**新成功标准** (全部满足才算"真正夹住"):
- 末段 10 帧 per-object 接触 ≥ 2 (双指接触同一物体)
- 末段 10 帧物体 z 方差 < 1cm (稳定被夹, 不是掉落中)
- `final_z - init_z > 3cm` (最终被抬起, 不是掉回原位)
- (可选) 末段 10 帧双指接触法向对向 (真的夹紧, 不是同侧推)

### 4.4 碰撞模型可视化 (Issue 3 配套)

**文件**: `grasp_hawor.py` 主循环渲染部分 (L3601 附近)

**改动**:
1. URDF 加载后, 用 `scene.create_actor_builder()` 为每个手指创建半透明红色 visual actor
   - 几何: 与 collision 相同 (mesh 或 box)
   - 材质: `sapien.render.PBRMaterial(base_color=[1,0,0,0.4])` (红色半透明)
2. 主循环每帧 `collision_visual_actor.set_local_pose(finger_link.get_local_pose())`
3. 渲染时半透明红色叠加在真实视觉模型上, 清晰看到碰撞体 vs 物体接触

### 4.5 轨迹重设计 (Issue 4, 最关键) — 位姿不改变 + 位置优化最小损失

**文件**: `grasp_hawor.py` L3018 (预计算) + L2224-2356 (中和态) + L3179-3207 (offset 计算)

**核心思想** (用户第二次澄清):
- 姿态 `gripper_R` = MANO `root_R[local_idx]` (**位姿不改变**, 严格跟随)
- 位置 `gripper_pos` = MANO `root_pos[local_idx] + offset` (位置可以有偏移)
- `offset` 通过**优化**求得: 在保证抓取成功的前提下, 最小化 `||offset||` (把损失降到最小)

**改动 1: mano_gripper_traj 扩展存储 R**

预计算循环 (L3018-3082) 中, `compute_analytical_gripper_pose` 返回 `(root_pos, root_R, j1, j2)`, 当前 `_,` 丢弃了 root_R, **改为存储**:
```python
# L3018: 加 "R" 键
mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": []}
# L3040/L3068: 不再丢弃 root_R
root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(...)
mano_gripper_traj[s]["R"].append(root_R)
# L3078-3081: R 转 numpy
mano_gripper_traj[key]["R"] = np.array(mano_gripper_traj[key]["R"], dtype=np.float64)  # (N, 3, 3)
```

**改动 2: _compute_mano_neutral_target 用 MANO root_R (位姿不改变)**

```python
def _compute_mano_neutral_target(self, local_idx, side):
    traj = getattr(self, '_mano_gripper_traj', {}).get(side)
    offset = getattr(self, '_mano_neutral_offset', {}).get(side)
    f_grasp = getattr(self, '_mano_grasp_frame', {}).get(side)
    if traj is None or offset is None or f_grasp is None or len(traj["pos"]) == 0:
        return None
    if local_idx >= len(traj["pos"]):
        local_idx = len(traj["pos"]) - 1

    # === 位姿不改变: 姿态严格跟 MANO root_R ===
    # CRITICAL FIX: 不再用 gripper_R_fixed (top-down), 改用 traj["R"][local_idx]
    gripper_R = traj["R"][local_idx]

    # === 位置优化最小损失: MANO root_pos + minimal offset ===
    mano_pos = traj["pos"][local_idx]
    mano_j1 = float(traj["j1"][local_idx])

    # 阶段判定 (基于 f_grasp, 保留现有逻辑)
    # ... [APPROACH/DESCEND/CLOSE/LIFT/TRANSPORT/RELEASE/RETREAT] ...

    # 位置计算 (保留现有阶段逻辑, 但所有阶段都用 MANO root_R):
    # - APPROACH/DESCEND/TRANSPORT/RELEASE/RETREAT: gripper_pos = mano_pos + offset
    # - CLOSE: gripper_pos = grasp_pos (f_grasp 处的 mano_pos + offset)
    # - LIFT: smoothstep 从 grasp_pos 过渡到 mano_pos + offset
    # ... [保留现有位置计算] ...

    # Z-floor 碗保护: 保留 (但 FINGER_FORWARD_OFFSET_NEUTRAL 改为基于当前 gripper_R 计算)
    # 因为手指偏移方向取决于 gripper_R 的 X 轴 (现在是 MANO root_R 的 X 轴, 不是固定的 [0,1,0])
    if has_bowl and phase in ("TRANSPORT", "RELEASE", "RETREAT"):
        # 用当前 gripper_R 的 X 轴计算手指前向偏移 (不再用固定 0.037)
        finger_forward = gripper_R[:, 0]
        # EE 需在碗上方安全高度 (沿世界 Z)
        bowl_safe_z = float(ctrl.bowl_pos[2]) + 0.15
        if gripper_pos[2] < bowl_safe_z:
            gripper_pos = gripper_pos.copy()
            gripper_pos[2] = bowl_safe_z

    # gripper_val: 保留现有阶段强制逻辑 (DESCEND 打开, CLOSE/LIFT/TRANSPORT 闭合, RELEASE 打开)
    # ...

    return gripper_pos, gripper_R, gripper_val, phase
```

**改动 3: offset 计算 — 优化最小损失 (替换 L3179-3207)**

```python
# 删除 gripper_R_fixed 和 ee_offset_neutral (基于固定朝向的偏移)
# 改为: 优化求解最小 offset, 使 MANO 轨迹在 f_grasp 处对齐目标抓取点
self._mano_neutral_offset = {}
self._mano_grasp_frame = {}
for s in self.sides:
    tgt = target_objs.get(s)
    traj = self._mano_gripper_traj.get(s)
    if tgt is None or tgt not in self.obj_bbox_centers or traj is None or len(traj["pos"]) == 0:
        self._mano_neutral_offset[s] = None
        self._mano_grasp_frame[s] = None
        continue
    target_pos = np.array(self.obj_bbox_centers[tgt], dtype=np.float64)

    # === 优化目标: 最小化 ||offset||, s.t. MANO 轨迹 + offset 在 f_grasp 处能抓到目标 ===
    # 抓取点 = target_pos 沿 gripper_R[f_grasp] 的 X 轴后退 finger_offset (让手指夹住物体)
    # 即: mano_pos[f_grasp] + offset + finger_offset_along_R = target_pos
    # → offset = target_pos - mano_pos[f_grasp] - finger_offset_along_R[f_grasp]

    mano_positions = traj["pos"]
    dists = np.linalg.norm(mano_positions - target_pos, axis=1)
    f_grasp = int(np.argmin(dists))

    # 用 f_grasp 处的 MANO root_R 计算手指前向偏移 (位姿不改变, 所以用真实朝向)
    R_at_grasp = traj["R"][f_grasp]
    FINGER_FORWARD = 0.037  # 手指在 EE 前方 3.7cm (沿 gripper_R 的 X 轴)
    finger_offset = R_at_grasp[:, 0] * FINGER_FORWARD

    # 最小 offset: 让 EE + finger_offset 对齐 target_pos
    offset = target_pos - mano_positions[f_grasp] - finger_offset

    self._mano_neutral_offset[s] = offset
    self._mano_grasp_frame[s] = f_grasp
    logger.info(f"  [neutral][{s}] MANO 最接近 {tgt} @ F{f_grasp} "
                f"(dist={dists[f_grasp]:.3f}m), minimal offset={offset.round(3)} "
                f"(||offset||={np.linalg.norm(offset):.3f}m)")
```

**为什么这是"最小损失"**:
- `offset` 是在 f_grasp 处让 EE+finger 对齐 target 的**最小必要平移**
- 在其他帧, `gripper_pos = mano_pos[f] + offset` (常量平移), 保持 MANO 轨迹形状
- 姿态 `gripper_R = traj["R"][local_idx]` 完全跟随 MANO (零姿态偏差)
- 唯一的"损失"是位置上的常量平移 `||offset||`, 这是抓取成功的最小必要平移

**改动 4: 删除 gripper_R_fixed 相关代码**

- L3179-3183: 删除 `self._gripper_R_fixed = ...`
- L3184-3185: 删除 `FINGER_FORWARD_OFFSET_NEUTRAL` 和 `ee_offset_neutral` (基于固定朝向)
- L2260-2262: 删除 `_compute_mano_neutral_target` 中的 `gripper_R = getattr(self, '_gripper_R_fixed', ...)`

**阶段行为调整**:
- 所有阶段: `gripper_R = traj["R"][local_idx]` (MANO 真实朝向, 位姿不改变)
- APPROACH/DESCEND: `gripper_pos = mano_pos + offset` (跟 MANO 轨迹形状)
- CLOSE: `gripper_pos = grasp_pos` (f_grasp 处, 让夹爪闭合)
- LIFT: smoothstep 从 grasp_pos 过渡到 `mano_pos + offset`
- (无碗场景): LIFT 后即结束 (用户: "实现抓取任务后结束")
- (有碗场景): 保留 TRANSPORT/RELEASE/RETREAT, 但 Z-floor 用 gripper_R 计算手指方向

## 5. 文件改动清单

| 文件 | 改动类型 | 范围 |
|------|---------|------|
| `grasp_hawor.py` L123-137 | 修改 | 物理参数 (friction 2.0→1.0, restitution 0→0.6) |
| `grasp_hawor.py` L291-367 | 修改 | `prepare_gripper_only_urdf`: 手指 collision 改 mesh, 加 mass/inertia |
| `grasp_hawor.py` L1062-1116 | 不变 | `compute_analytical_gripper_pose` 已返回 root_R |
| `grasp_hawor.py` L2224-2360 | 重写 | `_compute_mano_neutral_target`: 用 MANO root_R, 零偏移 |
| `grasp_hawor.py` L2880-2947 | 修改 | `mano_gripper_traj` 新增 "R" 存储 |
| `grasp_hawor.py` L3179-3210 | 重写 | offset 计算改零偏移起步 |
| `grasp_hawor.py` L3601 附近 | 新增 | 碰撞模型半透明红色可视化 |
| `grasp_hawor.py` L3804-3913 | 重写 | `_evaluate_grasp_quality`: per-object 接触 + final_z + z 稳定性 |
| `grasp_hawor.py` 主循环 | 修改 | 接触记录 per-object 化 |
| `CHANGE_LOG.md` | 新增 | 第十三轮条目 |
| `docs/grasp_hawor_analysis.md` | 更新 | 3.7d 节 (MANO 跟随重设计) + 第 6 章验证结果 |
| `docs/questions.md` | 新增 | Q5: 为什么轨迹偏离大 + Q6: 为什么"真正夹住"是误报 |

## 6. 成功标准

实施完成后, 运行测试:
```bash
python grasp_hawor.py --mode gripper_only --side left \
  --hawor-dir /home/an/data/hawor/7 --ras-dir /home/an/data/ras/my_7mp4_result \
  --views god --grasp-mode hybrid
```

通过条件 (全部满足):
1. ✅ 视频中可看到半透明红色碰撞体, 与物体真实接触可见
2. ✅ 物体在视频末段仍在夹爪中 (没掉回原位)
3. ✅ verify.json 新标准: final_z - init_z > 3cm, 末段 10 帧 z 方差 < 1cm, per-object 接触 ≥ 2
4. ✅ 夹爪姿态跟随 MANO root_R (不再是固定 top-down)
5. ✅ 物理参数与 Galaxea R1 一致 (mesh collision, friction 1.0, restitution 0.6)
6. ✅ god 视频明显能看到 0.2m 高度 (相对夹爪或场景)

## 7. 已确认的点 (用户第二次澄清)

1. **轨迹策略**: 位姿不改变 (用 MANO root_R) + 位置优化最小损失 (不是零偏移, 是最小化 ||offset||)
2. **碰撞可视化颜色**: 半透明红色 RGBA [1,0,0,0.4] (用户选第一个选项)
3. **god 相机策略**: 固定, 但需修复坐标系问题 (从 scene_center 相对改为地面绝对, god_pos_z=0.20m)
4. **有碗场景**: 保留 TRANSPORT/RELEASE/RETREAT 阶段 (用户没明确要求简化, 默认保留 pick-and-place); 但 Z-floor 需用 MANO root_R 计算手指方向 (不再是固定 0.037)
