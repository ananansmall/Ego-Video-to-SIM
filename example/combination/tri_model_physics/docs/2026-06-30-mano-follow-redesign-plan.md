# MANO 跟随重设计 实施计划 (第十三轮)

**Goal**: 修复 4 项问题 (物理碰撞体/god坐标系/验证误报/轨迹偏离), 实现"位姿不改变 + 位置优化最小损失"的 MANO 跟随抓取
**Architecture**: 全部改动在 `grasp_hawor.py` 单文件; URDF 模板内联在 `prepare_gripper_only_urdf` 函数中; 预计算轨迹扩展存储 R; 中和态用 MANO root_R 替代固定 top-down
**Tech Stack**: SAPIEN, NumPy, trimesh, OpenCV
**上游设计**: [2026-06-30-mano-follow-redesign-design.md](./2026-06-30-mano-follow-redesign-design.md)

---

## 文件改动清单

| 文件 | 改动类型 | 范围 |
|------|---------|------|
| `grasp_hawor.py` L123-137 | 修改 | 物理参数: friction 2.0→1.0 |
| `grasp_hawor.py` L291-367 | 修改 | URDF 模板: finger collision box→mesh |
| `grasp_hawor.py` L529-532 附近 | 修改 | 物体材质: restitution 0.0→0.6, friction 对齐 |
| `grasp_hawor.py` L2224-2356 | 重写 | `_compute_mano_neutral_target`: 用 traj["R"][local_idx] |
| `grasp_hawor.py` L3018-3082 | 修改 | `mano_gripper_traj` 加 "R" 键 |
| `grasp_hawor.py` L3179-3207 | 重写 | offset 计算: 用 MANO root_R 计算手指偏移 |
| `grasp_hawor.py` L3269 | 修改 | god_pos: 改为绝对地面坐标 |
| `grasp_hawor.py` L3601 附近 | 新增 | 碰撞模型半透明红色可视化 |
| `grasp_hawor.py` L3804-3913 | 重写 | `_evaluate_grasp_quality`: per-object 接触 + final_z |
| `grasp_hawor.py` 主循环接触记录 | 修改 | 接触记录 per-object 化 |
| `CHANGE_LOG.md` | 新增 | 第十三轮实施条目 |
| `docs/grasp_hawor_analysis.md` | 更新 | 3.7d 节 + 第 6 章验证结果 |

---

## Task 1: 物理参数对齐 R1 (Issue 1)

### Task 1.1: URDF 手指 collision box → mesh

**文件**: `grasp_hawor.py` L336-337, L354-355

**改动**: finger_link1 和 finger_link2 的 `<collision>` 从 box 改为 mesh

```python
# L336-337 (finger_link1) 旧:
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.012 0.020 0.024"/></geometry></collision>

# 新:
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link1.STL"/></geometry></collision>

# L354-355 (finger_link2) 同样改:
    <collision><origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{prefix}_gripper_finger_link2.STL"/></geometry></collision>
```

**验证**: 加载 URDF 后日志打印 finger collision 类型, 确认是 mesh。

### Task 1.2: friction 2.0 → 1.0

**文件**: `grasp_hawor.py` L134 附近 (GRIPPER_FRICTION 常量)

```python
# 旧:
GRIPPER_FRICTION = 2.0  # 或当前值
# 新:
GRIPPER_FRICTION = 1.0  # 对齐 Galaxea R1 (static=1.0, dynamic=1.0)
```

### Task 1.3: 物体 restitution 0.0 → 0.6

**文件**: `grasp_hawor.py` L529-532 附近 (load_glb_with_physics 接触材质)

查证当前代码:
```python
# 旧: restitution = 0.0
# 新: restitution = 0.6  # 对齐 Galaxea R1
```

**验证**: 启动时日志打印物体材质参数, 确认 restitution=0.6, friction=1.0。

---

## Task 2: god 相机坐标系修复 (Issue 2)

### Task 2.1: god_pos 改为绝对地面坐标

**文件**: `grasp_hawor.py` L3269

```python
# 旧 (L3269):
god_pos = scene_center + np.array([0.0, 0.0, god_height])

# 新:
# god_pos_z 用绝对地面坐标 (用户: "0.2m还是很高" 是因为 scene_center_z≈0.185 叠加)
ground_z = float(obj_centers[:, 2].min()) if len(obj_centers) > 0 else 0.0
god_pos = np.array([scene_center[0], scene_center[1], ground_z + god_height])
```

**验证**: 日志打印 `god_pos`, 确认 z ≈ 0.20m (不是 0.385m)。

---

## Task 3: 轨迹重设计 (Issue 4, 最关键)

### Task 3.1: mano_gripper_traj 扩展存储 "R" 键

**文件**: `grasp_hawor.py` L3018, L3038-3039, L3040, L3044-3046, L3066-3067, L3068, L3072-3074, L3078-3082

**改动 1**: L3018 初始化加 "R":
```python
# 旧:
mano_gripper_traj = {}  # side -> {"pos": [], "j1": [], "j2": []}
# 新:
mano_gripper_traj = {}  # side -> {"pos": [], "R": [], "j1": [], "j2": []}
```

**改动 2**: L3038-3039, L3066-3067 双手/单手分支初始化加 "R":
```python
# 旧:
mano_gripper_traj[s] = {"pos": [], "j1": [], "j2": []}
# 新:
mano_gripper_traj[s] = {"pos": [], "R": [], "j1": [], "j2": []}
```

**改动 3**: L3040, L3068 不再丢弃 root_R, 并 append:
```python
# 旧:
root_pos, _, j1, j2 = compute_analytical_gripper_pose(...)
# 新:
root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(...)
mano_gripper_traj[s]["R"].append(root_R)
```

**改动 4**: L3078-3082 转 numpy 时加 R:
```python
# 新增:
mano_gripper_traj[key]["R"] = np.array(mano_gripper_traj[key]["R"], dtype=np.float64)
```

**验证**: 启动后日志打印 `mano_gripper_traj[side]["R"].shape`, 确认是 (N, 3, 3)。

### Task 3.2: offset 计算用 MANO root_R (替换 L3179-3207)

**文件**: `grasp_hawor.py` L3179-3207

**删除**:
- L3179-3183: `self._gripper_R_fixed = ...`
- L3184-3185: `FINGER_FORWARD_OFFSET_NEUTRAL` 和 `ee_offset_neutral`

**重写** L3186-3207 offset 循环:
```python
self._mano_neutral_offset = {}
self._mano_grasp_frame = {}
for s in self.sides:
    tgt = target_objs.get(s)
    traj = self._mano_gripper_traj.get(s)
    if tgt is None or tgt not in self.obj_bbox_centers or traj is None or len(traj["pos"]) == 0 or "R" not in traj:
        self._mano_neutral_offset[s] = None
        self._mano_grasp_frame[s] = None
        continue
    target_pos = np.array(self.obj_bbox_centers[tgt], dtype=np.float64)

    mano_positions = traj["pos"]
    dists = np.linalg.norm(mano_positions - target_pos, axis=1)
    f_grasp = int(np.argmin(dists))

    # 用 f_grasp 处的 MANO root_R 计算手指前向偏移 (位姿不改变, 用真实朝向)
    R_at_grasp = traj["R"][f_grasp]
    FINGER_FORWARD = 0.037  # 手指在 EE 前方 3.7cm (沿 gripper_R 的 X 轴)
    finger_offset = R_at_grasp[:, 0] * FINGER_FORWARD

    # 最小 offset: EE + finger_offset 对齐 target_pos
    offset = target_pos - mano_positions[f_grasp] - finger_offset

    self._mano_neutral_offset[s] = offset
    self._mano_grasp_frame[s] = f_grasp
    logger.info(f"  [neutral][{s}] MANO 最接近 {tgt} @ F{f_grasp} "
                f"(dist={dists[f_grasp]:.3f}m), minimal offset={offset.round(3)} "
                f"(||offset||={np.linalg.norm(offset):.3f}m)")
```

### Task 3.3: _compute_mano_neutral_target 用 traj["R"][local_idx]

**文件**: `grasp_hawor.py` L2260-2262, L2333-2338

**改动 1**: L2260-2262 删除固定朝向, 改用 MANO root_R:
```python
# 旧 (L2259-2262):
# 固定 top-down 朝向 (gripper X 朝下, 稳定抓取)
gripper_R = getattr(self, '_gripper_R_fixed', None)
if gripper_R is None:
    gripper_R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float64)

# 新:
# 位姿不改变: 姿态严格跟 MANO root_R (用户: "位姿不改变, 姿态要跟上")
gripper_R = traj["R"][local_idx]
```

**改动 2**: L2333-2338 Z-floor 用当前 gripper_R (不再用固定 0.037):
```python
# 旧 (L2333-2338):
if has_bowl and phase in ("TRANSPORT", "RELEASE", "RETREAT"):
    bowl = ctrl.bowl_pos
    bowl_safe_z = float(bowl[2]) + 0.15 + 0.037  # 碗上方 15cm + 手指偏移
    if gripper_pos[2] < bowl_safe_z:
        gripper_pos = gripper_pos.copy()
        gripper_pos[2] = bowl_safe_z

# 新:
if has_bowl and phase in ("TRANSPORT", "RELEASE", "RETREAT"):
    bowl = ctrl.bowl_pos
    # 手指前向偏移沿当前 gripper_R 的 X 轴 (位姿不改变, 用真实朝向)
    finger_forward_z = abs(gripper_R[2, 0]) * 0.037  # 手指在世界 z 方向的分量
    bowl_safe_z = float(bowl[2]) + 0.15 + finger_forward_z
    if gripper_pos[2] < bowl_safe_z:
        gripper_pos = gripper_pos.copy()
        gripper_pos[2] = bowl_safe_z
```

**验证**: 运行后日志打印 `gripper_R` 不再是固定 `[[0,1,0],[0,0,-1],[-1,0,0]]`。

---

## Task 4: 验证逻辑修复 (Issue 3)

### Task 4.1: 接触记录 per-object 化

**文件**: `grasp_hawor.py` 主循环接触记录部分 (搜索 `contact_log`)

查证当前接触记录是全局的。改为 per-object:
```python
# 旧: contact_log.append(n_contacts)  # 全局
# 新: contact_per_obj = {name: 0 for name in obj_names}
#     for c in scene.get_contacts():
#         a0, a1 = c.actor0.name, c.actor1.name
#         for obj_name in obj_names:
#             if obj_name in (a0, a1):
#                 contact_per_obj[obj_name] += 1
#     contact_log.append(contact_per_obj.copy())
```

### Task 4.2: _evaluate_grasp_quality 重写

**文件**: `grasp_hawor.py` L3804-3913

**新成功标准** (全部满足才算"真正夹住"):
1. **末段 10 帧 per-object 接触 ≥ 2** (双指接触同一物体, 不是全局)
2. **末段 10 帧物体 z 方差 < 1cm** (稳定被夹, 不是掉落中)
3. **`final_z - init_z > 3cm`** (最终被抬起, 不是 max(z) 然后)

```python
def _evaluate_grasp_quality(self, ...):
    results = {}
    last_n = 10
    for obj_name in obj_names:
        positions = np.array([pos_log[i][obj_name] for i in range(len(pos_log))])
        init_z = positions[0, 2]
        final_z = positions[-1, 2]
        # 末段 10 帧 z 方差
        last_z = positions[-last_n:, 2]
        z_var = float(np.var(last_z))
        # 末段 10 帧 per-object 接触
        last_contacts = [contact_log[i].get(obj_name, 0) for i in range(len(contact_log)-last_n, len(contact_log))]
        contact_sustained = sum(1 for c in last_contacts if c >= 2) >= last_n * 0.8
        # 判定
        lift_ok = (final_z - init_z) > 0.03  # 3cm
        z_stable = z_var < 1e-4  # 1cm 方差
        results[obj_name] = {
            "真正夹住": contact_sustained and z_stable and lift_ok,
            "末段接触": contact_sustained,
            "末段z方差_cm": z_var * 100,
            "final_z - init_z_cm": (final_z - init_z) * 100,
            "lift_ok": lift_ok,
            "z_stable": z_stable,
        }
    return results
```

**验证**: verify.json 中 glb_1 的"真正夹住" 应该是 False (因为之前是托起又掉, final_z=init_z)。

---

## Task 5: 碰撞模型可视化 (Issue 3 配套)

### Task 5.1: 创建半透明红色碰撞可视化 actor

**文件**: `grasp_hawor.py` setup_robot 之后, 主循环之前

```python
# 在 robot 加载后, 创建手指碰撞可视化 actor (半透明红色)
self._collision_visual_actors = {}  # link_name -> visual actor
if render_available:
    try:
        import sapien.render
        for link_name in ["finger_link1", "finger_link2"]:
            builder = self.scene.create_actor_builder()
            # 用与 collision 相同的 mesh
            mesh_path = str(R1_MESH_DIR / f"{self.side}_gripper_{link_name}.STL")
            mat = sapien.render.PBRMaterial(base_color=[1.0, 0.0, 0.0, 0.4])  # 半透明红色
            builder.add_visual_from_file(mesh_path, material=mat)
            actor = builder.build_kinematic(name=f"_collision_vis_{link_name}")
            self._collision_visual_actors[link_name] = actor
        logger.info("  碰撞可视化 actor 已创建 (半透明红色 RGBA[1,0,0,0.4])")
    except Exception as e:
        logger.warning(f"  碰撞可视化创建失败: {e}")
        self._collision_visual_actors = {}
```

### Task 5.2: 主循环每帧更新碰撞可视化位姿

**文件**: `grasp_hawor.py` 主循环渲染部分 (L3601 附近)

```python
# 每帧更新碰撞可视化位姿 (跟随手指 link)
if hasattr(self, '_collision_visual_actors') and self._collision_visual_actors:
    for link_name, vis_actor in self._collision_visual_actors.items():
        # 找到对应的手指 link
        finger_link = ...  # 从 robot.get_links() 找
        if finger_link is not None:
            vis_actor.set_pose(finger_link.get_pose())
```

**验证**: 视频中可看到半透明红色碰撞体叠加在手指视觉模型上, 与物体接触可见。

---

## Task 6: 测试 + 验证

### Task 6.1: 运行测试

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/tri_model_physics
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
  --mode gripper_only --side left \
  --hawor-dir /home/an/data/hawor/7 \
  --ras-dir /home/an/data/ras/my_7mp4_result \
  --views god --grasp-mode hybrid
```

### Task 6.2: 验证成功标准

通过条件 (全部满足):
1. ✅ 视频中可看到半透明红色碰撞体, 与物体真实接触可见
2. ✅ 物体在视频末段仍在夹爪中 (没掉回原位)
3. ✅ verify.json 新标准: final_z - init_z > 3cm, 末段 10 帧 z 方差 < 1cm, per-object 接触 ≥ 2
4. ✅ 夹爪姿态跟随 MANO root_R (日志确认 gripper_R 不是固定 top-down)
5. ✅ 物理参数与 Galaxea R1 一致 (mesh collision, friction 1.0, restitution 0.6)
6. ✅ god 视频中相机高度明显是 0.2m (日志确认 god_pos_z ≈ 0.20)

### Task 6.3: 失败处理

如果 Task 6.2 未通过:
- 若"没夹住": 检查碰撞可视化视频确认手指是否真的接触物体; 调整 FINGER_FORWARD 或 offset
- 若"姿态不对": 检查 traj["R"] 是否正确加载, 日志打印 gripper_R 矩阵
- 若"轨迹仍偏离": 检查 offset 计算, 日志打印 ||offset||

---

## Task 7: 文档同步

### Task 7.1: CHANGE_LOG.md 更新

在第十三轮条目中补充"实施完成"小节, 记录验证结果。

### Task 7.2: docs/grasp_hawor_analysis.md 更新

新增 3.7d 节 (MANO 跟随重设计), 更新第 6 章验证结果。

### Task 7.3: 调用 change-log skill

任务结束前调用 change-log skill 输出修改总结。

---

## 执行顺序

按依赖关系执行:
1. **Task 1** (物理参数) — 独立, 可先做
2. **Task 2** (god 相机) — 独立, 可先做
3. **Task 3** (轨迹重设计) — 最关键, Task 3.1 → 3.2 → 3.3 顺序
4. **Task 4** (验证逻辑) — 独立, 但需 Task 3 完成后才能验证
5. **Task 5** (碰撞可视化) — 独立, 可并行
6. **Task 6** (测试) — Task 1-5 都完成后
7. **Task 7** (文档) — Task 6 通过后
