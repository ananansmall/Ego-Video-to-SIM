# 全管线诊断报告：Stage1 / Stage2 / Stage3（2026-08-25）

> 应用户要求，将诊断范围从 Stage3 扩展到全部三个 Stage，逐模块核对：
> 管线结构、数据传输、驱动范式、物理参数、验证结果与 plan 差距。
> 本报告取代同日早先的 stage3 单独报告（结论并入 §4）。

---

## 0. 一句话总结

**三个 Stage 都没有在"纯物理驱动"下真正成功过。** Stage1 的成功建立在 3 个物理 hack 上，
它输出的抓取点 `enclose_z=0.0254` 本身穿地；Stage2 忠实地把这个坏点织进轨迹；
Stage3 继承后，纯 drive 被 7cm 的不可达差距卡死（lift=0 十余版），v4.52 用 set_qpos
穿透作弊出 lift=2.7cm。整条链的公共病根是：**MANO 人手轨迹 → 夹爪 base 轨迹的映射
从未做几何可行性约束，且每个 Stage 用各自的 hack 掩盖它**。

---

## 1. 全管线数据流与断点图

```
MANO 轨迹 (data_loader, N帧)
   │ ① pre_translate: neutral_offset 平移 (v4.30 修复过双重平移 bug)
   ▼
┌─ Stage1: cem_grasp_pose_optimize ──────────────────────────────┐
│ │ Stage0: MANO F50 姿态扰动 ×240 候选                          │
│ │ Stage1a: GraspQuality 无仿真筛选 + MANO 旋转硬约束 ≤5°        │
│ │          (xyz 位置不约束!)                                    │
│ │ Stage1b: Top-K 80帧物理验证 (rollout_grasp_only)              │
│ │ Stage1c: CMA-ES 6DOF 微调                                     │
│▼ 输出 best_grasp{pos, R, gripper_qpos, enclose_z}               │
│ ✗ 断点A: enclose_z=obj_z+(R@[0.03689,0,0])z — 姿态倾斜时         │
│   offset_z<0 → 抓取点 z 可低于物体中心甚至穿地 (实测 0.0254)      │
│ ✗ 断点B: 验证环境含 3 个 hack (见 §2)，坏解在 hack 内"可行"       │
└──────────────────────────────────────────────────────────────────┘
   ▼
┌─ Stage2: reconstruct_trajectory (F38-F62, N_TRANS=12) ──────────┐
│ │ F50 = best_grasp.pos (直接采用, 含断点A)                       │
│ │ F50-F56 保持; F46-F49 hover→enclose smoothstep                │
│ │ F57-F62: F50 → MANO[offset] 过渡                              │
│▼ 输出 recon{pos,R,gripper_qpos,frames}                           │
│ ✗ 断点C: F57+ 交还未补偿的 MANO 手腕轨迹 (手腕贴物面, z 更低)     │
│ ✗ 断点D: v4.28 自述"原 F57-F62 16cm 快速上升会脱手→改温和携带    │
│   3cm" — 用改写轨迹迁就动力学, 而非修映射                        │
│ ⚠ rollout_follow_trajectory 驱动=set_qpos 硬设虚拟关节+drive 双保│
└──────────────────────────────────────────────────────────────────┘
   ▼
┌─ Stage3: rollout_v4_stage3 ─────────────────────────────────────┐
│ │ 合成: F1-45 MANO / F46-55 S2recon / F56-89 smoothstep→MANO    │
│ │       / F90+ MANO; v4.41 复刻 Stage1 时序(高空+边降边闭)       │
│ │ 抓取点 = opt_pos_full[50] = S2 F50 = 0.0254 (继承断点A)        │
│ │ 抬升窗 F63-F91: cmd_z -0.005→+0.026, 幅度仅 ~3cm              │
│▼                                                                 │
│ ✗ 断点E: 目标高度比手指触物后的物理下限 (~0.098) 低 7cm           │
│   → 纯 drive 被接触硬约束挡死 → lift=0 (v4.49-v4.51 十余版)      │
│ ✗ 断点F: v4.52 加 set_qpos 硬设穿透 → lift=2.7cm 作弊成功        │
│   (drift 6.6cm, F93 后单指接触, fgap 110→140mm)                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Stage1 诊断

### 2.1 它真的成功了吗？—— 名义成功，环境失真

| 项目 | 状态 |
|---|---|
| 表面指标 | ✓ lift_peak=3.47cm（历史基准），CEM 8×16 稳定收敛 ≥3cm |
| 驱动范式 | ✗ 混合：F0-F10 下降段 set_qpos 硬设虚拟关节+手指+qvel清零（纯运动学）；F10 后虚拟关节纯 PD |
| Hack① | ✗ rollout 期间禁用手指-地面碰撞（结束才恢复）→ 手指穿地无阻力 |
| Hack② | ✗ 禁用邻物碰撞 → 多物体场景失真 |
| Hack③ | ✗ gripper_only 全 link disable_gravity（test8 是重力补偿不禁用） |

**关键推论**：在 Hack①③ 环境里，"base 降到 0.0254、手指插入地平面"是一个可执行的解，
CEM 自然会选中它；换到纯物理环境（Stage3 减少了 hack），同一解立即不可执行。
**Stage1 的"成功"是 hack 环境内的局部最优，不是物理可行的抓取。**

### 2.2 enclose_z 公式缺陷（断点A 实锤）

```python
enclose_z = obj_init_z + (base_R @ [0.03689, 0, 0])[2]
```
- 设计意图：手指沿 base x 轴伸出 3.689cm，把 base 抬到"指尖恰在物体中心"的高度。
- 缺陷：R 来自 CEM/MANO（允许侧抓、斜抓）。当 R 使 x 轴朝下倾时 offset_z<0，
  enclose_z < obj_z。实测缓存 npz 中 **pos[2]=0.0254 且被标记"已经是 enclose_z"**
  （stage3z 日志 L100），而物体中心 z≈0.058 —— 抓取点比物体中心还低 3.3cm。
- 无任何下限校验（如 `enclose_z ≥ obj_top_z + margin` 或 `≥ 地面+finger_len`）。

### 2.3 reward 演化史暴露的问题

v4.36/v4.37 注释自述：距离项曾盖过抬升项 → 收敛到"双指压住不夹起"（lift 0.3cm 拿 18 分）；
随后又出现"铲起推飞"（lift 3.6cm 但 drift 5.3cm）。reward 打了 37+ 版补丁，
本质是在 hack 环境里对抗 Goodhart 定律——**环境失真时，reward 修不完**。

### 2.4 数据传输

- ✓ pre_translate 双重平移 bug 已修（v4.25/v4.30 注释）。
- ✗ 输出侧：best_grasp.npz 无"物理可行性"元数据；旧 npz 兼容代码里还有硬编码
  `_obj_z = 0.0173`（grasp_hawor.py L4637），与当前场景物体实际 z 不符。

### 2.5 视频生成

✓ test-stage1 路径录制 cam/god/right 三视角并转码 H264。

---

## 3. Stage2 诊断

### 3.1 模块性质与验证方式

- 本体是**纯数学插值**（smoothstep + Minimum Jerk 五次多项式，边界速度匹配），无自身物理正确性概念。
- 配套 `rollout_follow_trajectory` 做物理回放验证（v4.28 边走边抓视频）。

### 3.2 数据传输正确性

| 环节 | 判定 |
|---|---|
| F50 ← best_grasp.pos | ✓ 忠实传递（连缺陷一起传） |
| F50 z 兼容逻辑 | ✓ 有 hover→enclose 修正；✗ 但不校验 enclose 本身合理性 |
| F51-F56 保持 | ✓ 给 PD 收敛时间（τ≈10 帧，留 7 帧） |
| **F57-F62 → MANO[offset]** | ✗ **断点C**：交还未补偿的手腕轨迹，z 立即掉向贴物面区域 |
| 夹爪曲线 | ⚠ 与 Stage1 曲线不同源（_PRELOAD=0.004 vs Stage1 GRIPPER_MAX_OPEN 流程），rollout 时又弃用 recon 曲线改用 Stage1 安全曲线——**recon 的 gripper 输出实际未被消费** |

### 3.3 驱动范式

`rollout_follow_trajectory`：set_qpos 硬设虚拟关节 + drive_target 双保险（与 Stage1 下降段同款）。
→ Stage2 的"验证通过"同样发生在准运动学保护下，无法暴露轨迹可达性问题。

### 3.4 已知妥协（断点D）

v4.28 注释：原 F57-F62 的 16cm 快速上升会让物体脱手 → 改成温和垂直携带 3cm。
这是典型的"改轨迹迁就动力学"。MANO 原始抬升语义（16cm 大幅上移）被静默丢弃，
下游 Stage3 又以 MANO 为过渡目标——**各 Stage 对"轨迹真值"的定义已经互相矛盾**。

---

## 4. Stage3 诊断（承接当日 stage3 报告，补充新证据）

### 4.1 问题表现（铁证）

```
stage3y (v4.51b 纯 drive):
F20 dz=4.5mm   ← drive 通道健康
F44 cmd_z=0.0254 qpos_z=0.0985 dz=73mm  ← 手指触物后被接触硬约束挡死
F63 cmd_z=-0.0052 (地面以下!)            ← 指令物理不可达
F91 cmd_z=+0.0260                        ← "抬升完"仍比卡死高度低 7cm
```

### 4.2 根因链（新证据闭环）

1. Stage1 缓存 npz：`pos[2]=0.0254`（断点A 产物）；
2. test-stage3 加载缓存，仅做 hover→enclose 兼容修正，无合理性校验；
3. Stage2 F50=0.0254，Stage3 opt_pos_full[50]=0.0254，v4.41 时序"垂直降至 enclose z=0.025"
   （stage3z 日志 L108 明文确认）；
4. 纯 drive 下 base 物理下限 ≈0.098（手指触物顶面）→ 7.3cm 死区 → lift=0；
5. v4.49-v4.51 调抬升速度/PD 全部无效（目标本身不可达）；v4.52 set_qpos 穿透作弊 lift=2.7cm，
   代价 drift 6.6cm + 单指接触 + fgap 发散。

### 4.3 与 05_gripper_test.py 的范式差异

| 维度 | test8 成功配方 | Stage3 现状 |
|---|---|---|
| 驱动 | 纯 driver + compute_passive_force(gravity=True) 补偿 | v4.51 纯 drive(轨迹坏)；v4.52 set_qpos 作弊 |
| 轨迹 | 手工设计：下探到表面上方即停→合指→垂直上提，全程几何可行 | MANO 直映 + 各代补丁，抓取点穿地 |
| VIRTUAL PD | 5000/1000 | 50000/2500（盲调） |
| GRIPPER PD | 5000/500 无力限 | 16000/200 force_limit=40N |
| DECIMATION | 16 | 8 |
| 重力 | 不禁用，被动力量补偿 | 全 link 禁重力 |

用户判断正确：**test8 架构就是 driver；set_qpos 与参考实现和 plan 均相悖。**

---

## 5. 跨 Stage 系统性问题汇总

### 5.1 MANO→base 映射缺失（总病根）
人手手腕可贴物面运动（手指短、环腕分布）；夹爪手指单向伸出 ~3.7cm，base 必须
保持在"物体表面+指长"之上。三个 Stage 都没有这层几何约束，各自打补丁：
Stage1 靠 hack 硬塞、Stage2 靠改写携带段、Stage3 靠 set_qpos 穿透。

### 5.2 hack 依赖链
```
Stage1 (禁指地碰撞+set_qpos下降+禁重力) ─产出坏 enclose_z→ Stage2 (set_qpos跟随)
                                              ↓ 忠实传递
                                          Stage3 (纯drive→卡死; set_qpos→作弊)
```
上游在失真环境里优化，下游在更真实环境里执行 —— **优化与验证环境不一致是结构性错误**。

### 5.3 物理参数三套并存（未统一）
- `physics_env.py`: GRIPPER 16000/200/FORCE4000N, VIRTUAL 50000/2500(v4.51 盲调), DECIM 8
- `physics_utils.py`: GRIPPER 1000/200/FORCE 10N（被 grasp_controller/trajectory_loader 引用）
- `05_gripper_test.py`(基准): GRIPPER 5000/500, VIRTUAL 5000/1000, DECIM 16
- 2026-08-21 对齐 plan 只执行了一半。

### 5.4 与 plan 的差距
| plan 要求 | 现状 |
|---|---|
| 物理层回归纯 PD，删除 set_qpos/lock_root_pose hack | 三个 Stage 全依赖 set_qpos |
| Stage3 CEM 参数化优化 (9 维) | v4.7 起退化为固定平滑轨迹（opt_params 弃用） |
| 验收 lift≥3cm | 作弊后 2.7cm，仍未达标 |
| MANO 旋转硬约束 ≤5° | ✓ 已实现（唯一严格落地项） |

### 5.5 视频与可观测性
- 视频：三个 Stage 均有生成路径 ✓（cam/god/right 三视角 + H264 转码）。
- 可观测性缺口：直到 v4.51b 才加 cmd-vs-qpos 对比诊断；此前十余版失败均无执行链路证据。
  **应把 [vdrive] 类对比设为常开低成本采样**（如每 20 帧）。

---

## 6. 修复路线图（按依赖顺序）

### P0 — 抓取点几何合法性（治本，一处修复三处受益）
1. Stage1 enclose_z 计算后强制钳制：
   `enclose_z = max(enclose_z, obj_top_z + 0.005, ground + finger_len*cos(tilt))`
   或直接限制 CEM 姿态使 offset_z ≥ 0（只允许"指尖向下"族姿态）。
2. Stage1 rollout **去掉 Hack①（禁指地碰撞）**，让 CEM 在真实碰撞下重选解；
   若无解达标，说明该物体需要不同抓取策略而非调参。
3. best_grasp.npz 写入 `feasible=true` 元数据；加载端校验失败即拒绝使用并报警。

### P1 — 轨迹全程可达性
4. Stage2/Stage3 合成轨迹后统一后处理：
   `base_z[f] = max(base_z[f], grasp_z_min)`（grasp_z_min 由 P0 的合法 enclose_z 决定），
   F90+ MANO 段同样钳制；携带段语义改为"保持夹持高度跟随 MANO xy"。
5. 删除 v4.28 的"温和携带 3cm"特判（P0/P1 生效后不再需要）。

### P2 — 回归纯 driver（对齐 test8/plan）
6. 撤销 v4.52 set_qpos（保留 debug 开关默认关）；撤销 v4.51 的 VIRTUAL 50000 盲调，
   回 5000/1000；GRIPPER 回 5000/500 去 force_limit；DECIMATION 16。
7. 重力策略二选一并全线统一：要么全 link 禁重力（现状），要么对齐 test8 用
   compute_passive_force(gravity=True) 补偿——禁止混用。

### P3 — 常态化验收门
8. 每个 Stage 输出固定指标卡：自由空间跟踪 dz<5mm、contact=both 占比、fgap 波动、
   drift、lift；任一越界即 fail-fast，不允许带病进入下一 Stage。
9. 合并 physics_utils/physics_env 常量为单一来源。

### 验收标准（全管线）
纯 drive 下：Stage1 lift≥3cm 且无碰撞 hack；Stage2 跟随全程 contact=both；
Stage3 lift≥3cm、drift<2cm、fgap 波动<10mm；三视角视频人工复核无穿透。

---

## 7. 方法论教训

1. **先验证执行通道与环境真实性，再调参数**：F20 帧 dz=4.5mm 一条数据即可证明 drive 健康，
   应在第 1 版失败时就采集，而不是十几版后才加诊断。
2. **上游在失真环境优化的解，不能假设下游可用**：hack 环境里的"最优"是环境的函数，
   不是物理的函数。优化环境必须与验收环境一致。
3. **指标要拆到物理量纲**：lift=0 背后是"目标低于硬约束下限 7cm"的几何事实，
   任何 PD/速度参数都无法跨越。
4. **参考实现对齐要到范式级**：只抄 test8 的 PD 数值而不抄"可行轨迹+纯动力学"的架构，
   等于没对齐。
