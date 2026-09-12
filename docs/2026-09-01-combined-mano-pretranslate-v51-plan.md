# 全部 Stage 由完整管线串接（2026-09-05 增补，便于整体把握）

> 每个 Stage 都是在**同一套坐标系（GLB=SAPIEN）**、**同一平滑+重定向+平移后的轨迹**上工作，只做各自职责，不重复预处理。

```
预处理链路（前置，所有 Stage 共用）:
  MANO FK(compute_mano_joints) → 平滑 21 关节 → svd 重定向(compute_mano_based_gripper_pose)
    → pre_translate(整体平移, F50 手指中点对齐物体中心) → neutral_offsets(残差≈0)

Stage0（预处理链路验证 / 基线）:
  纯重定向+平移后的夹爪轨迹回放，不优化。验证平滑→重定向→平移后夹爪能否跟随 MANO
  （轨迹图三色锚: 腕/拇指尖/食指尖 + 夹爪 URDF 跟随轨迹）。

Stage1（抓取位姿搜索）:
  gripper_only 短仿真（v4-pipeline 内 stage1_sim），cem_grasp_pose_optimize 网格+CMA-ES
  找"给定 base 位姿能否夹住并提升15cm" → best_grasp{pos,R,gripper_qpos,enclose_z}。
  通过判定: both_contact≥10 且 lift≥15cm（§12.2）。

Stage2（宽窗加权整合 F38-F62）:
  reconstruct_trajectory 用 w 加权(w_max=0.5 起步)把 Stage1 抓取位姿与 MANO 平滑融合,
  产出 recon{frames=F38-F62,pos,R,gripper_qpos} → 转 _fixed_offsets_654。
  通过判定: max_dev≤5mm + both_contact≥10 + obj_carry_travel≥4cm（§12.3）。

Stage3（全局合成+F90放置）:
  rollout_v4_stage3 以 MANO 为底, 覆盖 Stage2 段(F38-F62), F62+跟随 MANO, F90±3帧张开放物。
  通过判定(S3 最终验收): carry_validation.passed + release_z_std<2mm, 物体跟随夹爪（§12.4）。
```

> 顺序关系（v51 U3，2026-09-05 修订）：**平滑 → 重定向 → 平移** 在前置链路里完成，Stage0-3 全部在这条"已平滑+已重定向+已平移"的轨迹上复用，不再各自预处理。

---

## 1. Stage 0 预处理：坐标变换与 MANO 参数优化（重定向前，直接映射）

> **Stage 0 = 直接映射（无搜索、无姿态锁定）**：坐标变换 → 平滑（优化 MANO 参数）→ SVD 重定向，全为确定性变换/滤波，输出"重定向+平顺后的 MANO 轨迹"（§1.5 最终仿真实现）驱动夹爪物理回放。"姿态锁定 MANO F50 R / 240 位置网格"是 **Stage 1** 候选生成层的搜索约束（§2/§8.4.1），不属于 Stage 0。

> 现状缺口（**v51 重构前**的旧入口）：管线把 MANO 原始逐帧直接喂 PD，无统一重定向、无平滑（STAGE_ISSUES P0-2；**该描述已过时**——2026-09-05 v51 重构后 `run()` 预计算已接入重定前平滑，见 §1.2 实现状态）。
> 设计：**在重定向前**先做坐标变换（拿 MANO 参考点），再平滑，然后解算夹爪。

### 1.1 坐标变换：HaWoR → GLB（只一步，不用旧 RXWORLD_TO_SAPIEN）

**原则：GLB 即世界即仿真坐标系（001/002 链，R_X=I 已实测证实）。**
只需把 HaWoR 手部数据变换到 GLB 系，无需 `RXWORLD_TO_SAPIEN`/`R_AXIS`/`ZUP_TO_YUP`（旧链已废弃）。

**两步坐标链（唯一正确链，与 `render_quick.cam_traj_to_glb`(L63) 一致）**：
```
                                          ┌─ 第1步: 逐帧(必选) ─┐  ┌─ 第2步: 统一 ─┐
 MANO FK joints(相机系) ──R_c2w[f]/t_c2w[f]──→  HaWoR 世界系  ──(s,R_h2g,t_h2g)──→  GLB=SAPIEN
                                                                  (来自001首帧锚定)
```
```python
# 第1步: 逐帧 real camera pose → HaWoR 世界系
#   (pred_trans / MANO FK joints 在 "HaWoR 输入相机系", z>0; SLAM 相机每帧在动, 故必须逐帧)
p_hw = R_c2w[f] @ p_cam + t_c2w[f]          # _mano_to_sapien_v4 L3337

# 第2步: 统一相似变换 (s,R_hand_to_glb,t_hand_to_glb) 为全局常量, 与帧号无关
#   R_hand_to_glb 来自 001_align_scene "首帧相机锚定"(render_quick L18-20); R_X=I, 无需额外 R_x
p_glb = scale_ratio * (R_hand_to_glb @ p_hw) + t_hand_to_glb   # L3340
```
> 澄清"逐帧 vs 统一"：**第1步逐帧**（手在动的相机系里，每帧 c2w 不同，提到 HaWoR 世界系）；**第2步统一**（`(s,R_h2g,t_h2g)` 全局常量，锚定首帧）。到 GLB 只有一次统一变换，不是每个 mano 一个变换。

**产物 = MANO 参考点轨迹**：`compute_mano_joints`(FK) 得到 21 关节 `joints_glb`，其中 j0=腕 / j4=拇指尖 / j8=食指尖 就是之后可视化的参考锚。

### 1.2 平滑（四重）（重定前）
`smooth_mano_trajectory`（05_gripper_test L269）：①Savitzky-Golay ②高斯四元数 Slerp ③min-jerk ④EMA（在线命令级）。
> 注：实测对"恒定系统偏移"无效——若偏差是标定/坐标问题，平滑不治病（P1-1 教训）。平滑的作用是消除**逐帧抖动**，不替代坐标系校正。
> **位置**：坐标变换（§1.1）之后、重定向（§1.4）之前——先变换坐标，再平滑（优化 MANO 参数），再解算夹爪。

### 1.3 MANO 参考点 ≠ 夹爪重定向（两节分离，不混用）

| | MANO 参考点（数据/观测） | 夹爪重定向（解算/控制） |
|---|---|---|
| 来自 | `compute_mano_joints` → `hand_to_glb` 变换 | SVD / 解析式 从 joints 解算 |
| 是什么 | 手腕/拇指/食指原始 3D 轨迹（GLB 系观测） | `_mano_gripper_traj[pos,R,j1,j2]` 夹爪 base 位姿 |
| 用于 | pre_translate 的 F50 手指中点基准、可视化对比锚 | Stage1/2/3 驱动的轨迹本体 |

> 分层：**实例化一条 `_mano_gripper_traj`（夹爪重定向结果）时，它派生自 MANO 参考点（坐标变换结果），但二者是不同层级的量**，计划不把 `_render_to_sapien`/`RXWORLD_TO_SAPIEN` 算作夹爪重定向，只作为坐标变换链里已弃用的一步。

### 1.4 夹爪重定向方法：SVD 具体实现细节（与 MANO 参考点区分）

**入口**：`compute_mano_based_gripper_pose(joints_glb, prefix, finger_origin_x=0.03689)`（data_loader L521-590）。

```
输入: joints_glb (21×3, GLB 系, 来自 §1.1 坐标变换 + §1.2 平滑)
  j[3]=拇指PIP  j[4]=拇指尖  j[5]=食指MCP  j[6]=食指PIP  j[7]=食指DIP  j[8]=食指尖
```

**Step 1：锚点**
```python
anchor = (j[3] 拇指PIP + j[5] 食指MCP) / 2        # dL543: 夹爪 base 位置
```

**Step 2：SVD 拟合手掌平面 → Z 轴（平面法向）**
```python
palm_pts = [j[3], j[5], j[6], j[7], j[8]]            # dL545: 5 点
centered = palm_pts - palm_pts.mean(axis=0)
_, _, Vt = np.linalg.svd(centered, full_matrices=True)
z_axis = Vt[2] / np.linalg.norm(Vt[2])               # dL547-548
```

**Step 3：Z 符号确定性（消除跨帧抖动）**
```python
# SVD 的 Vt[2] 符号在帧间可任意翻转 → 用解剖学参考 n_ref 定符号
n_ref = normalize(cross(j[6]-j[5], j[5]-j[3]))       # 食指MCP→PIP × 拇指PIP→食指MCP
if dot(z_axis, n_ref) < 0: z_axis = -z_axis          # dL558-561
```
> 注意：此处只保证**符号稳定**，不负责夹爪朝下（朝下由 Step 6 重力对齐统一保证）。

**Step 4：X 轴 = 食指方向，对 Z 做 Gram-Schmidt 正交化**
```python
x_raw = normalize(j[6]-j[5])                          # 食指 MCP→PIP; 退化时用 j[7]-j[5]
x_orth = x_raw - dot(x_raw, z_axis)·z_axis
x_axis = normalize(x_orth)                            # 严格 ⊥ z_axis
```

**Step 5：Y = Z × X，再重算 Z = X × Y（三轴严格正交）**
```python
y_axis = cross(z_axis, x_axis)
z_axis = cross(x_axis, y_axis)
R = column_stack([x_axis, y_axis, z_axis])            # dL575-577
```

**Step 6：重力对齐（approach 轴 = gripper X 朝下）**
```python
R = align_gripper_approach_to_gravity(R)              # dL582
# 判据：dot(R[:,0]=X, g=-Z) < 0 → 手指朝上 → 绕开合轴 Y 旋转 180° (X→-X,Z→-Z,Y不变)
```

**Step 7：base 位置与夹爪开合量**
```python
root_pos = anchor                                     # dL584: 无偏移, 对齐 viz_web
thumb_tip = j[4]; finger_dist = ||j[8]-j[4]||         # 拇指尖↔食指尖距离
joint = clip(finger_dist / 2, 0, 0.05)                # dL586-588: 开合量 = 指距/2
```

**输出契约** `(root_pos, R, joint, joint)`（joint1=joint2=指距/2，与 `compute_analytical_gripper_pose` 返回一致，可互换调用点）。

> 与解析式的差异：SVD 用 5 点手掌平面拟合（姿态更稳、对坏帧鲁棒），解析式只用腕/两指尖三点（L445）。本计划 §6 U1 统一为 SVD。


### 1.5 输出
- **轨迹图片**：MANO 参考点（红=腕 / 绿=拇指尖 / 蓝=食指尖）叠加夹爪 URDF 的跟踪轨迹图（对齐 test9 可视化 05_gripper_test L2810）。
- **最终仿真实现**：重定向+平顺后的 MANO 轨迹驱动夹爪的物理回放——**这就是 Stage 0 的操作**（直接映射，无搜索、无姿态锁定）。

---

## 2. Stage 1：F50 最优抓取位姿（gripper_only 子实例 CEM）

> **一句话**：在 Stage 0 的 MANO 轨迹上，求 **F50 的最优抓取位姿 `{pos, R, gripper_qpos}`**——这个位姿就是"直接提起物体"的位姿；gripper_only 短仿真验证"能否接触、夹住、抬升"，**抬升量 `obj_lift` 进 reward**（§8.3.1）。

### 2.1 输入 / 目标 / 输出
- **输入**：Stage 0 产物（平滑+重定向后的 MANO 轨迹，F50 参考 `mano_f50_pos`/`mano_f50_R`）；物体 bbox `obj_bbox_size`。
- **目标**：F50 抓取位姿 `{pos, R, gripper_qpos}`。**搜索变量 = 位置**（1c 加姿态微调 ≤5°）；**夹爪开合 g 不搜索**，由 `compute_gripper_qpos(obj_bbox_size)` 解析求得——使手指接触并挤压物体表面（stage1_grasp_mixin L29/L114）。
- **输出**：`best_grasp.npz = {pos:(3,), R:(3,3), gripper_qpos:(2,), enclose_z:(3,)}`，其中 `pos[2]=enclose_z`（手指接触物体时的 base z，L995-1003）。

### 2.2 "就是这个位姿，直接提起"
F50 是 MANO 真实抓取帧。Stage 1 把该帧重定向成夹爪位姿后**原地就位 → PD 闭指 → 提起**（`rollout_grasp_only`：hover→enclose_z 就位，z 在 enclose±1cm 内由 CEM 决定 → 闭指到 `compute_gripper_qpos` 施加持续压力 → 抬升）。能提起 = 抓取成立，`obj_lift`（V1）直接进 reward；`both_contact_count`（V2）作门禁（§12.2）。

### 2.3 搜索结构（5 层，详见 §8.4.1）
```
候选生成(240 位置网格 close8×depth6×z5; R 锁定=MANO F50 R; g=compute_gripper_qpos)
  → 1a 预筛(score_grasp_quality + MANO 硬约束 旋转≤5°) → Top8
  → 1b 物理验证(rollout_grasp_only ×8, 取 reward 最大)
  → 1c CMA-ES 6DOF 精英重采样(15×32): (dx,dy,dz,dry,drx,drz),
      R_actual=delta_rot@ref_R; 违反≤5° → reward=-50; 均值漂出可行域 → 重置
  → 1d 纯平移精修(±4mm step 2mm, 姿态不变)
```

### 2.4 入口与子实例
**入口**：`cem_grasp_pose_optimize`（stage1_grasp_mixin L723），跑在 **gripper_only 子实例**（独立 6DOF 虚拟关节，与 full_robot 臂 IK 解耦），保证"给定 base 位姿能否夹住"被干净评估：
```
新建 stage1_sim(mode="gripper_only", views="none", output_dir=.../stage1)
  ├─ 子实例 pre_translate + compute_neutral_offsets   (防双重平移)
  ├─ cem_grasp_pose_optimize(n_iterations=15, population_size=32)
  │     └ generate_grasp_candidates / rollout_grasp_only 做 Top-K 物理验证
  └─ 校验非空 → 把 _mano_neutral_offset/_mano_grasp_frame/best_grasp 同步回主实例
```

### 2.5 验收 / 门禁
reward 含接触 / 夹持 / 抬升项 → 能接触并夹住，输出合法 `enclose_z`（日志 `[v4 pipeline][验收] Stage 1`）。门禁 §12.2：`both_contact_count≥10` + 跟随提升。

---

## 3. Stage 2：宽窗加权整合 F38-F62（扩宽方案 + 新增 w 加权）

> 现状：`reconstruct_trajectory(N_TRANS=4)` → 重建 **F46-F54(9帧)**，用 Smoothstep+hold；**没有 w 加权**（代码 L132-153 是 smoothstep，L144-146 是 hold）。
> 设计（与代码差异已确认）：**扩宽到 F38-F62**，并**新增 w 加权机制**（现有代码未实现，作为待物理验证的尝试方案）。

### 3.0 对齐原则（Stage0-3 帧边界必须统一，不能各自为政）
```
Frame0          F38        F46   F50   F54   F62
 |────接近──────|──预抓取──|抓取|保持|过渡─|──跟随 MANO──→
                         抓取窗口 F38-F62（Stage2 重建）
                         Stage3 锁定段 F38-F62（不优化，直接复用 S2）
```
- **Stage2 重建窗口 = F38-F62（25 帧）**：`f_start=F50-12, f_end=F50+12`（替换现有 `F50±N_TRANS`，N_TRANS 4→12）。
- **Stage3 锁定区间必须同步改为 F38-F62**：现有 `stage3_reward`(L53) 是 `f<=54`，扩宽后应 `f<=62`，否则 Stage3 会在 F54-F62 覆盖 S2 结果 → 管线不一致。此点列入 §11 待确认核验。

### 3.1 帧语义（预对齐后讨论）
预对齐（#0）后 MANO 已平移到物体附近，故 MANO 与 Stage1 的 F50 位姿都属于**同一坐标系**，可安全混用。

### 3.2 w 加权机制（新增，替代纯 smoothstep）
> 现有代码 L153 是 `pos = optimized_f50_pos*(1-s) + pos_traj[f]*s`（s=smoothstep）——本质是"单一过渡"，不含用户控制。
> 新增 w 让重建结果 = **w·grasp + (1-w)·MANO**，w 每段不同，且可多档平滑。

```
位置:  pos(f) = w(f)·grasp_pos[F50]  +  (1-w(f))·MANO_pos[offset](f)
姿态:  R(f)   = slerp(MANO_R(f), F50_grasp_R, w(f))     # 四元数 slerp
夹爪:  g(f)   = lerp(张开, g_grasp, w(f))                # 随 w 收拢
```
| 段 | 帧 | w(f) | 行为 |
|---|---|---|---|
| 接近 | F38-F49 | 0 → 1（平滑上升） | MANO 切入 → 渐变到 F50 抓取位姿；边走边预闭合 |
| 抓取 | F50 | 1 | 直接用 Stage1 best_grasp（enclose_z） |
| 夹持 | F51-F58 | 1（hold） | 锁 F50 位姿，原地闭合+稳定 |
| 过渡 | F58-F62 | 1 → 0（平滑下降） | 从 F50 grasp 平滑回 MANO+offset |
| 跟随 | F62+ | 0 | 完全跟随 MANO |

> w 的过渡用多段 smoothstep（≥2 次幂）保证位置/速度连续；"多加权多平滑" = w 曲线本身也做平滑（避免 w 突变引起抖动）。这是**待物理验证的尝试方案**：先试固定 α（如 w 峰=1、平滑窗=6 帧），不行再加一维 w 也入优化网格。

### 3.3 F50 → F62（夹持 + 过渡回 MANO）
- F50 夹住后，w 从 1 逐渐降到 0，位置平滑回 MANO+offset；
- **F62 端点固定为 MANO 原始位姿**；F62 之后 w=0 完全跟随 MANO。

### 3.4 全轨迹夹爪开合设计（2026-09-05 定案：F50 夹住后恒定 g_grasp + F90 固定张开放置）
> 结论：开合量**不是靠 SVD 优化**——SVD（§1.4 Step7）只给**初始估计** `指距/2`；**实际抓取开合**由 Stage1 物理验证确定 `g_grasp = best_grasp.gripper_qpos`；此后到释放前**恒定**，不逐帧跟随 MANO。

**开合量来源（三层，与位姿分开讲）**：
| 层 | 开合量 | 来源 |
|---|---|---|
| 初始估计（几何） | `joint = clip(指距/2, 0, 0.05)`（拇指尖↔食指尖距离/2） | SVD 重定向 Step7 |
| 实际抓取值（物理） | `g_grasp = best_grasp.gripper_qpos`（夹住物体并提升15cm 的开合） | Stage1 物理验证 |
| 时间插值（合成） | `g(f)=lerp(张开, g_grasp, w(f))`（随 w 收拢/张开） | Stage2 w 加权 |

**全轨迹开合量形状（用户定案）**：
```
开合  | 张开__                     __张开(释放时张开放物)
gqpos |       \                   /
      |        \___ g_grasp 恒定 ___/__ → 跟随 MANO 退开
      F0        F50  F51~F58 F59~释放  F90±3帧       F90+
      接近(随w收拢) 夹住  夹持    携带(恒定)  张开放置     退开
```
| 段 | 开合量 | 行为 |
|---|---|---|
| 接近 F38~F50 | `lerp(张开, g_grasp, w)` 随 w 上升 | 边走边预闭合 |
| **F50 夹住** | **g_grasp（恒定）** | 夹住物体（enclose_z）|
| 夹持 F51~F58 | **g_grasp（恒定 hold）** | 锁 F50 位姿，原地稳定 |
| 携带 F59~F90 | **g_grasp（恒定）** | 物体被夹着移动，不随 MANO 变 |
| **F90 固定张开放置** | **张开放物（恒定到释放完成）** | 抛弃 MANO 信号自动检测，§11 第2项 |
| F90+ 退开 | 跟随 MANO 开合 | 物体已放下，手退开 |

**约束**：
- **夹持+携带段（F50→释放前）开合量恒为 g_grasp**，绝不逐帧跟随 MANO 手指距离（否则引入抖动，可能夹持不稳/滑脱）；
- **仅 §11 第2项 F90 固定 ±3 帧 smoothstep 从 g_grasp → 张开**完成放置，抛弃现代码 `_mano_open=(j1+j2)/2` 的释放自动检测（[stage3 L562-566](file:///home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization/stage3_optimize_mixin.py#L562-L566)），改由固定 F90 触发（需落码）。
- 初始开合 `指距/2`（SVD）与最终 `g_grasp`（Stage1）可不同：Stage1 物理验证会**覆盖** 初始值，以能夹住为标准；
- "物体跟随夹爪"≠"让开合量去追 MANO"：开合恒定保证夹持刚性，位置/姿态由 §3.2 w 加权负责。

### 3.5 Stage2 物理验证与代价（沿用 `optimize_stage2_dz`，不重写）
现有代码 post-2026-08-31 已有完整物理验证 + 代价，计划沿用它：

**代价（`optimize_stage2_dz`，2026-09-05 改锚点=F50 设计帧）**：
```python
# 携带锚点 _t0 = F50 设计帧 (rollout f50_frame), 不是首触帧 —
# F38-F50 接近段(接触前)的 fmid-obj 距离不作代价项 (本来就没接触, §3.2).
# F50 前双侧接触 = w 过渡不安全 (设计违规): both 只计 F50 后, 且每早 1 帧扣分.
reward = min(obj_carry_travel/0.04, 1.0)*30      # 主任务: F50 后物体被带走 ≥4cm 记满 30
         - carry_dev*300                           # 刚性跟随罚 (F50→末 恒定距离偏差)
         + 0.5*both_ratio*10                        # 稳定双侧夹持 (仅 F50 后)
         + lift*30                                  # 次要: 提升
         - 5.0*drift*100                            # 推飞罚
         - early_pen                                # F50 前接触罚 (每帧 EARLY_CONTACT_PEN_PER_FRAME=10)
```

**物理验证（`rollout_follow_trajectory` L542-562）**：
| 指标 | 来源 | 判定 |
|---|---|---|
| 恒定距离 `max_dev` | `validate_carry(fmid, obj, grasp_frame=f50_frame)`（F50 设计帧锚点，2026-09-05） | ≤5mm PASS |
| 跟随误差 `mean_mm` | `compute_tracking_error(base_actual, base_target)` | 打印比对 |
| `obj_lift` / `obj_lift_peak` | `obj_z峰值 - obj_z初` | 观察 |
| 双侧接触 `both_contact_count` | 双侧同时接触帧 | ≥10 |
| `obj_xy_drift` | 物体 xy 位移 | 观察 |

**Stage2 优化方式**：沿用现有 `optimize_stage2_dz` 的 **dz 网格搜索**（F50 就位 z 微调），reward 最大选最优；**先单调 w**（固定 α/平滑窗）观察效果，再加 w 到搜索网格。

---

## 4. Stage 3：全局合成（复用 S2）+ 跟随 MANO + F90 放置

> 现状：F90 仍是保持夹持，无放置段；抬升硬编码 1.2mm/帧。
> 设计：stage3 **基本跟随 MANO，只覆盖 stage2 给出的部分**，并在 **F90 放置**。

### 4.1 分段
```
opt_pos_full = MANO(平滑后) 作底
  ├─ F0~F38   跟随 MANO（可轻微平滑）        ← 接近
  ├─ F38~F62  覆盖 = S2 宽窗加权结果（复用 Stage2）← 抓取+夹持
  ├─ F62~F89  跟随 MANO（携带）             ← 抓走
  └─ F90~F112 放置：夹爪张开、物体放下        ← 释放
```
- **F90 放置（重要新增）**：F90 夹爪张开让物体落放；**F90 附近 ± 几帧都可放置**；
- 放置需**平滑连贯**：状态切换用 smoothstep，避免瞬时张开导致物体被捕明弹飞；
- 放置完成后，剩余帧跟随 MANO 退开。

### 4.2 与上表对应的物理策略
| 段 | 夹爪 | 物体 |
|---|---|---|
| F0-F38 | 张开，跟随 MANO | 自由/未接触 |
| F38-F62 | 加权闭合到夹持 | 接触→被抓紧 |
| F62-F89 | 保持夹持(恒定距离) | 随夹爪移动 |
| F90+ | 平滑张开放置 | 落下，平滑过渡 |

---

## 5. 重定向、仿真参数、物体/夹爪设计（固化到计划）

### 5.1 坐标变换与重定向
| 项 | 方案 | 来源 |
|---|---|---|
| 坐标变换 | `hand_to_glb`：`s·R_hand_to_glb·p + t_hand_to_glb`（R_X=I，弃用 RXWORLD_TO_SAPIEN） | `common.py::hand_to_glb`(L1361) |
| 夹爪重定向 | svd_palm：5 点手掌平面 SVD 取法线 + Gram-Schmidt | `data_loader::compute_mano_based_gripper_pose`(L521/L547) |
| 向下对齐 | 重力 -Z 判据 `dot(X,g)<0` → 绕 Y 翻转180° | `physics_utils::align_gripper_approach_to_gravity`(L174) |
| MANO 参考点 | `compute_mano_joints`(FK) → `hand_to_glb`（j0/j4/j8 = 腕/拇指尖/食指尖） | `common.py::compute_mano_joints`(L366) |

### 5.2 仿真参数（单一来源 `physics_utils.py`）
| 参数 | 值 | 位置 |
|---|---|---|
| OBJECT_DENSITY | 1000 kg/m³ | physics_utils L66 |
| OBJECT_MIN_MASS | 0.05 kg | physics_env L199 |
| GRIPPER_FRICTION | 2.0（single source） | physics_utils L87 |
| 虚拟关节 PD | VIRTUAL 5000/1000, ANGULAR 500/200, GRIPPER 5000/500 | 05_gripper_test L591-603 |
| DECIMATION | 16 | 05_gripper_test L322 |

### 5.3 夹爪设计
- gripper_only URDF：手指 prismatic 关节（行程约束一致）
- 手指原点在 base +X `_GRIPPER_DEPTH_OFFSET=0.03689` 处
- 重力补偿：`step_physics(gravity_compensation=True)`（05_gripper_test L629）

---

## 6. 统一化清单（当前代码未统一处 → 本计划目标）

> 明确以下"同一 CLII 参数/同一语义在不同处不一致"的地方，本计划逐条收拢。**原则：任何影响抓取结果的参数必须在 4 个 Stage 一致生效，否则不引入。**

| # | 现状（不统一） | 代码证据 | 计划目标 |
|---|---|---|---|
| U1 | 重定向方法按 `grasp_pose_method` 分支（svd_palm/analytical） | L3365-3373 真假两路 | **统一 svd_palm**，解析式仅作对照 |
| U2 | 世界约定混用：`glb_coords` 有时乘 `R_AXIS`，旧 `_render_to_sapien` 仍存在 | `_mano_to_sapien_v4` L3341-3344 | **只保留 GLB 系**（glb_coords=True），删 `R_AXIS`/`RXWORLD_TO_SAPIEN` 路径 |
| U3 | 平滑时机在各子实例不一致（stage0/1/2 在 neutral_offsets 之后，stage3 已修正） | pipeline-variables.md §⑤ | **统一：先坐标变换 → 平滑（MANO 关节）→ 重定向 → pre_translate**（§1/§8.5 口径，2026-09-05 定案） |
| U4 | `flip_mano_hand_down` 仅部分路径生效（默认关） | `_mano_to_sapien_v4` L3345 | 统一并入重力对齐，删独立 flip |
| U5 | pre_translate 在 `run_v4_pipeline`(L3030) 与 test-stage3（v4.30）两处实现 | §0/§5 旧文 | 单一预处理入口，所有路径复用 |
| U6 | MANO 参考点与夹爪重定向混在 `_mano_gripper_traj` 里，职责不清 | L3362-3377 | 计划明确分层：参考点(数据) 派生 → 夹爪重定向(控制)，两表分离 |

---

## 7. 最终测试参数与验证指标

### 7.1 运行配置（单一事实源）
| 参数 | 值/入口 |
|---|---|
| 主入口 | `grasp_hawor.py --method v4-pipeline --stage3-method smooth --side left` |
| 分 Stage 调试 | `--test-stage0/1/2/3`（stage1/2/3 均 `--views none` 快速验证） |
| Python 环境 | `/home/an/miniconda3/envs/dex/bin/python` |
| 场景 | `7_vggt_omega`（`toy_0` 为 target） |
| 重定向 | svd_palm + gravity align（U1） |

### 7.2 复现命令
```bash
cd example/combination/trajectory_optimization
PY=/home/an/miniconda3/envs/dex/bin/python
# 全管线（预处理+预对齐+S1+S2+S3+回放）
$PY grasp_hawor.py --method v4-pipeline --stage3-method smooth --side left
# 分 Stage 验证/回归
$PY grasp_hawor.py --test-stage1 --views none --side left   # S1 抓取位姿
$PY grasp_hawor.py --test-stage2 --views none --side left   # S2 宽窗整合
$PY grasp_hawor.py --test-stage3 --views none --side left   # S3 全局+放置
# 输出交付
$PY grasp_hawor.py --test-stage2 --views both --side left   # 三视角交付视频
```

### 7.3 验证指标（判定阈值）
> **判定明细/如何计算/代码溯源 → 见 §8.5 验证指标总表（本 plan 唯一指标口径源）**；此处仅列各 Stage 主判定阈值速查：
| 指标 | 目标/判定 | 主用 Stage |
|---|---|---|
| 恒定距离 `max_dev` | ≤ 5mm PASS | S2/S3 |
| 跟随误差 `mean_mm` | ≤ 5mm | 观察（S2/S3） |
| 释放稳定性 `release_stability_mm` | < 2mm | S3（F90 放置） |
| `lift` | **观察项**（跟随 MANO 自然抬升，不设硬阈值，§11 第5项/§12.4） | S1/S2/S3 |
| `both_contact_count` | ≥10（S1 门禁） | S1/S2 |
| 物体↔MANO 距离 `obj_mano_dist_mean` | 有界（目标最小化） | S3 |
| `obj_carry_travel` | ≥4cm（S2 reward 主项） | S2 |

### 7.4 关键日志标记
- `[pre_translate]` — 平移量
- `[平滑]` — 预处理平滑生效
- `[v4 pipeline][验收] Stage N ✓/✗` — 各 Stage 通过/失败
- `[PhysicsValidation][stageN]` — 恒定距离 + 跟随误差
- `[Stage3 到位确认][诊断]` — 闭合三条件实测量级

---

## 8. 验证指标与变量：如何计算、如何得来（2026-09-05 整合 §7.3+§8+§9）

> 定位：本 plan **唯一**的指标/变量口径源。整合了原 §7.3（验证指标）、§8（代价+下降）、§9（变量/数据流）。
> 每个指标/变量都给出 **① 如何计算（公式）② 代码溯源（文件+L）③ 判定/用途**——全部经代码核实。
> 统一定案：**GLB 即 SAPIEN 仿真系**（`glb_coords=True`，`load_glb_direct` 原样加载，旧链 `RXWORLD_TO_SAPIEN`/`R_AXIS`/`ZUP_TO_YUP` 弃用）。

### 8.1 坐标一致性（每个 Stage 的前提）
```
场景 GLB（load_glb_direct）→ SAPIEN 原样加载
手 data（hand_to_glb，p_glb = s·R_hand_to_glb@p_hawor + t_hand_to_glb）→ 同一 SAPIEN/GLB 系
对象 bbox（GLB）→ 同一 SAPIEN/GLB 系；全部 Stage 只在一套 SAPIEN=GLB 系运算。
```

---

## 8.2 一、验证指标总表（如何算 + 代码溯源 + 判定）

> 这是各 Stage 回放/优化产出物转成的**观测指标**，与 §8.3-8.5 的 reward（代价）分开：reward 供优化器选参，指标供 §12 判定通过。

| # | 指标 | 如何计算（公式） | 代码溯源 | 判定/用途 |
|---|---|---|---|---|
| V1 | `obj_lift` | `max(0, obj_z_final - obj_init_z)`（S1/S2 用峰值 `obj_lift_peak = max(0, obj_z峰值-obj_init_z)`） | stage1_grasp_mixin L599/L609；stage2_reconstruct_mixin L488/L492 | 观察项（§11-5：跟随 MANO 自然抬升，不设硬阈值） |
| V2 | `both_contact_count` | 逐帧两侧手指同时接触物体 ++ | stage1 L427/536；stage2 L356/436 | S1 门禁 ≥10（§12.2）；S2 观察 |
| V3 | 恒定距离 `max_dev` | `max|‖fmid[f]-obj[f]‖ - ref|`，`ref=‖fmid[grasp_frame]-obj[grasp_frame]‖`（F50 设计帧→释放，abs 偏差），≤tol_m=CARRY_DEV_TOL_M；代码先取 `d=‖fmid-obj‖` 再 `dev=|d-ref|`，`max_dev=dev.max()` | physics_utils.`validate_carry` L249/L275-278 | ≤5mm PASS（S2/S3） |
| V4 | 跟随误差 `mean_mm` | `mean(‖actual-target‖)` 逐帧，实际基座 vs 送 PD 的目标轨迹 | physics_utils.`compute_tracking_error` L205 | ≤5mm（观察） |
| V5 | 释放稳定性 `release_z_std` | `std(释放窗口 obj_z) * 1000` → `release_stability_mm` | stage3_optimize_mixin L645-646 | <2mm（S3 F90 放置） |
| V6 | `drift`（xy 漂移） | `‖obj_xy_final - obj_xy_init‖`（全程物体相对**初始**位置的水平位移，两处同式） | stage1_grasp_mixin L593；stage2_reconstruct_mixin L486 | 观察（§8.3.1 用于积分罚） |
| V7 | 物体↔MANO 距离 | `obj_mano_dists[f]=‖obj[f]-mano_ref[f]‖`，agg=mean/f50/f80 | stage3_optimize_mixin L610-619 | 有界（S3，目标最小化） |
| V8 | `obj_carry_travel` | 首次双侧接触后 `‖obj[-1]-obj[首触]‖`（物体实际位移） | stage2_reconstruct_mixin L581-584 | S2 reward 主项 ≥4cm |
| V9 | `peak_grip_force` | 最大夹持力 | stage1 L617 附近 | 观察 |
| V10 | `best_grasp` 合法 | `enclose_z` 非 NaN/有限/在物体底面~顶面 | data_loader 重定向输出 | S1 门禁（§12.2） |

---

## 8.3 二、Stage 代价函数（reward，供优化器选参）

### 8.3.1 Stage 1 — `stage1_reward`（stage1_grasp_mixin L663）
三种结局分类，不是连续逼近：
```python
lift_final = obj_lift_final;  lift_peak = obj_lift;  dist = fingertip_obj_dist
both_contact = both_contact_count;  xy_drift = obj_xy_drift;  peak_force
r  = 40.0 * min(1.0, max(0, lift_final)/0.20)        # 主导: 提升20cm满分
r += 10.0 if lift_final >= 0.15                       # ≥15cm 稳定夹持奖励
if both_contact>=10 and lift_final>0.05: r += 6.0     # 真抓取
elif both_contact>0 and lift_final<=0.005: r -= 6.0   # 压住不夹
r += 4.0 * exp(-dist/0.02)                            # 抓中心
if lift_final>0.02:
    if xy_drift>0.03: r -= 15.0; elif xy_drift>0.015: r -= 6.0; else: r += 3.0
else:
    if xy_drift>0.05: r -= 8.0
if lift_peak>0.08 and lift_final<0.5*lift_peak: r -= 8.0   # 中途滑落
if 5.0<peak_force<200.0: r += 1.0
```

### 8.3.2 Stage 2 — `optimize_stage2_dz` reward（stage2_reconstruct_mixin，2026-09-05 改锚点=F50 设计帧）
主任务=“边走边抓”；提升次要。携带锚点 `_t0 = f50_frame`（F50 设计帧，rollout 坐标），**不是首触帧**：F38-F50 接近段（接触前）的 fmid-obj 距离不作代价项（本来就没接触）。F50 前双侧接触 = w 过渡不安全（设计违规）：`both` 只计 F50 后，且每早 1 帧扣 `EARLY_CONTACT_PEN_PER_FRAME=10`。
```python
reward = min(obj_carry_travel/0.04, 1.0)*30      # ① F50 后物体被带走≥4cm 记满30
         - (carry_dev if not nan else 0)*300      # ② 刚性跟随罚（F50→末 恒定距离偏差）
         + 0.5*both_ratio*10                       # ③ 稳定双侧夹持（仅 F50 后）
         + lift*30                                 # ④ 次要：提升
         - 5.0*drift*100                           # ⑤ 推飞罚（xy漂移）
         - early_pen                               # ⑥ F50 前接触罚（每帧 10）
```
- `obj_carry_travel`=F50 后物体 3D 位移；`carry_dev`=F50 后 `‖fmid-obj‖` 相对 F50 基准最大偏差（滑脱检测）；`both_ratio`=F50 后双侧同时接触帧占比（F50 前接触归零）。

### 8.3.3 Stage 3 — `stage3_reward`（stage3_optimize_mixin L28）
分区间权重，跟随 MANO：
```python
for f: 
  f<46  → w_pos,w_rot = 1.0,30.0    # 接近贴合 MANO
  f<=62 → continue                   # F38-F62 锁定 Stage2（扩宽后，须同步 f<=54→f<=62）
  f<=90 → w_pos,w_rot = 2.0,40.0    # 携带更高权重
  r -= w_pos*||opt_pos[f]-mano_pos[f]||²
  r -= w_rot * rotation_distance(opt_R[f],mano_R[f])²
  r -= 10.0*||opt_pos[f+1]-2opt_pos[f]+opt_pos[f-1]||²   # 二阶平滑
if lift>0.03 and contact>=3: r += 100; elif lift>0.03: r += 50; ...
```

---

## 8.4 三、各 Stage 参数调整与“优化怎么下降”（迭代机制，代码实证）

> 优化器 = 搜索变量 + §8.3 代价 + 下降规则（如何更新搜索分布）。

### 8.4.1 Stage 1 — 分层搜索 + CMA-ES（`cem_grasp_pose_optimize` L723）
> 注意：这里的"候选生成"是 **Stage 1 的搜索层**，不是 Stage 0（§1 直接映射）。姿态基准全程 = MANO F50 R。
- **搜索变量**：位置（网格 → 1c 加姿态微调）；**夹爪开合 g 不搜索** = `compute_gripper_qpos(obj_bbox)` 解析求得（L29/L114）
- **姿态轨迹**：候选生成 **R 锁定 = MANO F50 R**（L263-264）→ 1a 旋转≤5° 硬约束（L790-805）→ 1c 可微调 `R=delta_rot@ref_R`（std 15°，违反≤5°→-50，L899-907）→ 1d 姿态不变（纯平移）
```
240位置网格(close8×depth6×z5) --1a预筛(score_grasp_quality+旋转≤5°)--> Top8
  --1b物理验证(rollout_grasp_only)--> 最优 --1c CMA-ES 6DOF 精英重采样(15×32)--> best_grasp
  --1d 纯平移精修(±4mm step 2mm, 姿态不变)
  1c 下降: top25% elite → mean=elite.mean, std=elite.std+1e-4 → 迭代收敛
```

### 8.4.2 Stage 2 — 1D dz 网格穷举（`optimize_stage2_dz` L587）
- 搜索变量：F50 就位 z 偏移 `dz`；网格 `[-14,-10,-6,-2,0,2,6,10,14]`mm 9 档（L591）
- 下降：**非迭代穷举 + argmax**（每 dz 一次 rollout→reward→取最大，L642-643）
- 本 plan 再加一维 w（先固定 w=0.5，§3.2）

### 8.4.3 Stage 3 — 默认 smooth（无迭代）+ CEM 备选（`cem_stage3_optimize` L880）
- 默认 `--stage3-method smooth`：直接合成（MANO 基底 + 覆盖 S2 段 + F90 放置），一次 rollout 验证——**无下降**
- CEM 备选：非锁定帧×6DOF，top25% 精英重采样，`mu=elite.mean,std=elite.std+1e-4`，10 轮（L942-953）

---

## 8.5 四、变量与数据流（原 §9，吸收至此；左手专用）

> **数据定案（2026-09-05）**：只用**左手**。数据源 `~/data/hawor/7a/reconstruction/hawor_results_0_113.npz`，`pred_*` 首维=(2,113)：**index0=真实左手**，index1=占位手（忽略）。CLI `--side left`→`hand_idx=0`（`hand_indices={"left":0,"right":1}`，grasp_hawor L209）。

```
pred_trans/rot/hand_pose(45)/betas (HaWoR 输入系, hand_idx=0 左手)
  → MANOLayer FK → mano joints 21×3 (j0腕/j4拇指尖/j8食指尖)
  → 平滑 MANO 21 关节（重定前，SG+Slerp+min-jerk）
  → hand_to_glb → MANO 参考点（GLB 系）
  → svd 重定向 compute_mano_based_gripper_pose(+gravity align) → _mano_gripper_traj{pos,R,j1,j2}
  → pre_translate（整体平移）→ neutral_offsets（≈0 残差）

S1 → best_grasp.npz {pos:(3,), R:(3,3), gripper_qpos:(2,), enclose_z:(3,)}
S2 → stage2_recon.npz {frames:[38..62]{25}, pos, R, gripper_qpos}
S3 → stage3_result.npz {mano_pos, opt_pos, opt_R, obj_pos_traj, result, carry_validation, obj_mano_dist*}
```

| 中间变量 | 如何得来 | 代码溯源 | 口径 |
|---|---|---|---|
| `joints_glb` | FK 21 关节 + hand_to_glb | trajectory_loader L143；data_loader L449 | (21,3) m |
| `g_grasp` | Stage1 `best_grasp.gripper_qpos` = `compute_gripper_qpos(obj_bbox)` 解析求得（手指挤压物体表面；**非搜索变量**） | stage1_grasp_mixin L29/L114 | 开合量 |
| `enclose_z` | F50 手指接触物体时的 base z | data_loader 重定向 / stage1 | m |
| `w(f)` | §3.2 加权曲线：F50=1，非平台 ×w_max(=0.5 起步) | stage2_reconstruct_mixin w_curve L21 | 0~1 |
| `f_start/f_end` | Stage2 窗口 (38,62) | stage2 L48 | 帧 |
| 顺序（U3） | `平滑 MANO 关节 → svd 重定向 → pre_translate → neutral_offsets` | 各 Stage 统一 | — |

---
## 10. 与现状代码的差异对照（实现前必读）

| 设计点 | 现状 | 目标（本文） |
|---|---|---|
| 坐标变换 | 旧 `_render_to_sapien`+`R_AXIS`/`RXWORLD_TO_SAPIEN` 仍存在 | `hand_to_glb`（GLB 即世界），删旧链 |
| MANO 参考点 vs 重定向 | 混在 `_mano_gripper_traj` 职责不清 | 分两层：参考点(数据) → 重定向(控制) |
| 重定向 | stage 分支 svd/analytical 未统一(U1) | 统一 svd_palm + 重力对齐 |
| MANO 预处理 | 直接喂 PD，无平滑(P0-2) | 重定向前 坐标变换+平滑 |
| Stage2 帧窗 | N_TRANS=4 → F46-F54，垂直下降+锁定 | **F38-F62** 宽窗，加权平滑插值整合 |
| Stage2 夹爪 | 走边阖 | 调整闭合，夹爪中心对齐物体中心 |
| Stage3 F90 | 保持夹持，无放置 | **F90 放置**，顺滑张开落物 |
| Stage3 抬升 | 硬编码 1.2mm/帧 上限 | 跟随 MANO 真实抬升，无硬编码上限 |
| Stage3 合成 | opt_pos_full 拼 S2 + smoothstep | 基本跟随 MANO，仅覆盖 S2 段 + F90 放置 |

---

## 11. 已定案设计参数（不再变更）

1. **Stage3 锁定边界**：扩宽 Stage2 到 F38-F62，`stage3_reward` 的 `continue` 边界 `f<=54` **同步改为 `f<=62`**，与 Stage2 对齐（§8.4）。
2. **F90 放置**：±3 帧 smoothstep 张开放物。
3. **重定向方法**：**先用 SVD**（`svd_palm`，U1 定案），解析式仅作对照保留、不再驱动管线。
4. **w 加权初始值**：**从 w=0.5 起步**，平滑窗≈6 帧；物理验证后逐步调 w，若有效再加一维 w 入搜索网格。
5. **抬升**：跟随 MANO 自然抬升（lift 仅作观察项）。
6. **代价函数/验证指标**：沿用现有代码实现（§8），不重写。
7. **坐标系与手性（2026-09-05 强化定案）**：GLB 即 SAPIEN 仿真系（002 链，R_X=I）；**数据集只用左手（`hawor_results` 的 index0 为真实左手，index1 为占位，实测 rot≈0/trans≈原点）**。CLI 统一 `--side left` → `hand_idx=0`（代码 `hand_indices={"left":0,"right":1}`, grasp_hawor L209），所有复现命令/主入口已改为 `--side left`。**plan 只处理左手，右/双手不做**（用户定案）。
8. **平滑参数先简单（2026-09-06 定案）**：`max_vel=0.020/max_acc=0.005` 等用**默认值起步**，**不追平 MANO 原始速度（~43mm/帧）**；**接受中段滞后**（最大 ~14cm，F50 处 11.5cm，不影响 F50 对齐与 Stage1 验证）；全链路达标后再按需精调，调参必须带根因记录、不许盲试（§1.2 参数策略）。

> 核心一句话：**预处理（坐标两步链 + SVD 重定向 + 平滑）→ Stage0 纯跟随回放射出基线 → Stage1 抓取位姿 → Stage2 用 w=0.5 起步的 F38-F62 加权整合 → Stage3 复用 S2 跟随 MANO + F90 放置，全程沿用现有代价/指标。**

---

## 12. 各 Stage 成功判定与通过门禁（独立定案，2026-09-04 讨论收敛）

> 定位：把散落在 §2/§3/§7/§8 的"验收/指标"收拢为一套**分 Stage 的 Pass/Fail 判定**。
> **reward 与通过判定脱钩**：reward（§8.2/§8.3/§8.4）是**搜索器**——S1 选 best_grasp、S2 调 dz/w、S3 CEM 打分，只用于挑参数；**通过判定统一看"物体跟夹爪的运动"**（本节的硬门禁），两者不混用。
> **数值口径全部沿用现有代码与上文（§7.3/§8.2/§8.5），不新造公式。** 分两层：**硬门禁**（失败阻断/标记 FAIL）与**观察项**（只打印，不阻断）。

### 12.1 串行门禁
- **S1/S2 失败 → 阻断**：不进下一 Stage（日志 `[v4 pipeline][验收] Stage N ✗`，管线中止）；
- **S3 是最终验收**：失败不阻断回放，但日志标 `FAIL`，交付视频带失败标记。

### 12.2 Stage 1 通过（门禁：夹住 + 跟随提升）
- `best_grasp` 非空（CEM 收敛出结果）；
- 其物理验证满足**跟随提升**（S1 主轴，对齐 §8.2 的 `+10` 稳定夹持项）：`both_contact≥10 且 lift≥15cm`（物体被夹住并跟随夹爪竖直提升 ≥15cm）；
- `enclose_z` 合法：非 NaN、有限、在物体底面~顶面合理区间。
- **观察项**：`stage1_reward` 数值（搜索用，不作为通过判定）、`enclose_z` 与物体 bbox 的关系。

### 12.3 Stage 2 通过（门禁：恒定距离 + 双侧接触 + 物体被带走）
- `max_dev≤5mm`（`validate_carry` → `carry_validation.passed`，§3.5/§8.5）；
- `both_contact≥10`（§3.5 L192）；
- `obj_carry_travel≥4cm`（**物体被夹爪带走**，§8.3 reward 主导项，直接度量"物体跟夹爪运动"）。
- reward（§8.3）仅用于 dz/w 选优，**不作为通过判定**。
- **观察项**：`mean_mm`（跟随误差）、`obj_lift`、`obj_xy_drift`。

### 12.4 Stage 3 通过（最终验收：物体跟随夹爪移动）
核心 = **F50-F90 物体是否跟随夹爪移动**（§8.5 修正：lift 不是 S3 成功指标，是跟随 MANO 的副产品）：
- **主项（carry，物体跟夹爪）**：`carry_validation.passed`（max_dev≤5mm，F50-F95 恒定距离刚性跟随）；
- **辅助（物体跟 MANO）**：`obj_mano_dist_mean` 有界（核对 S2 的 F38-F62 段仍完整跟随 MANO，参考 `obj_mano_dist_f50/f80`）——只记录，不作为硬门禁；
- `release_z_std<2mm`（F90 落放稳定，§7.3/§8.5）。
- **观察项**：`obj_lift`（跟随 MANO 自然抬升，不设硬阈值，§11 第5项）、`mean_mm`（跟踪误差）、`drift`。
- **矛盾修正**：§7.3 的 `lift≥3cm` 硬阈值 → **降为观察项**；§8.5 "S3 不设硬阈值" → 明确为"**lift 不设阈值，但 carry/跟随 设门禁**"。

### 12.5 与旧判定（§7.3/§8.5）差异一览
| 项 | 旧（§7.3/§8.5） | 新（本节） | 变化 |
|---|---|---|---|
| S1 | 定性"能接触并夹住" | `both_contact≥10 且 lift>0.05` + `enclose_z` 合法 | 数值化（沿用 §8.2） |
| S2 | max_dev≤5mm、both_contact≥10 | 同左 + reward 不作通过判定 | 明确边界 |
| S3 | lift≥3cm(§7.3) vs 观察(§8.5) **矛盾** | carry_validation.passed + obj_mano_dist 有界 + release_z_std<2mm | 消除矛盾，以"跟随夹爪"为核心 |
| 串行 | 无 | S1/S2 阻断、S3 最终验收 | 新增 |