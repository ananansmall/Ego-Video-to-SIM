# Stage 3 流程与问题诊断

**日期**: 2026-08-31
**数据**: 右手 (`--side right`), hawor/7 + ras/7_vggt_omega, 目标物体 `glb_4`
**代码**: `stage3_optimize_mixin.py::rollout_v4_stage3` (入口 `grasp_hawor.py --test-stage3`)

---

## 0. 结论速览

| Stage | 目标 | lift | drift | 恒定距离(容差5mm) | 与 MANO 偏差 | 判定 |
|-------|------|------|-------|-------------------|--------------|------|
| 0 | 纯重定向(不夹) | 0cm (设计如此) | — | — | offset=**0** | ✅ 跟随 9/10 |
| 1 | 找可抓位姿 | **19.5cm** | 0.69cm | max_dev=**1.0mm** PASS | F50 xy **8.8mm** | ✅ |
| 2 | 轨迹重建 F38-62 | **19.5cm** | 0.7cm | — | — | ✅ 验证通过 |
| 3 | 全局执行+释放 | **2.8cm** | 1.8cm | max_dev=**23.5mm** FAIL | F50 **54mm** | ❌ |

**Stage1/2 与 MANO 高度一致 (8.8mm) 且能真抓起 19.5cm；Stage3 退化到 2.8cm 且恒定距离 FAIL。**

---

## 1. Stage 3 完整流程

### 1.1 位置/姿态轨迹 (v4.41 + v4.49)

```
F0-F9    抬升至 hover z=0.209
F10-F24  高空飞向抓取点上空 (水平转移)
F25-F44  垂直下降至 enclose z≈0.029
F45-F49  到达抓取点 (原代码此区间无分支覆盖 → 见 §3.2)
F50      抓取帧 _F_GRASP
F50-F89  hold: 跟随 MANO 轨迹
F63-F89  v4.49 高点冻结+慢速抬升: 锁定 xy/R, 线性 1.2mm/帧
F89-F95  释放 (8 帧 smoothstep)
F96+     跟随 MANO 开合
```

**关键约束**: v4.49 把抬升段硬编码为 `1.2mm/帧 × 26帧 = 3.1cm` 上限。
→ **Stage3 的 lift 上限就是 3.1cm**, 与 Stage1 的 19.5cm 不是同一路径,
两者 lift 数值**不可直接比较**。

### 1.2 夹爪时序 (4 段)

| 帧 | 阶段 | PD 目标 |
|----|------|---------|
| F0-24 | 全开 | `GRIPPER_MAX_OPEN` (0.05) |
| F25-49 | 接近触发闭合 | 10 帧 smoothstep → `_GRIP_PD_HOLD` |
| F50-89 | hold 夹持 | `_GRIP_PD_HOLD` |
| F89-95 | 释放 | 8 帧 smoothstep → `_mano_g` |
| F96+ | 跟随 MANO | `_mano_g = (j1+j2)/2` |

`_GRIP_PD_HOLD = best_grasp['gripper_qpos'] - 0.004` (v4.42 深压, 见 §4.2)

### 1.3 物理参数 (v6.0 统一后)

```
虚拟关节: K=50000, D=600      (VIRTUAL_PD_STIFFNESS/DAMPING)
手指 PD : K=1000,  D=200, F=100N
摩擦    : 2.0 (v6.6 统一, 见 §5)
DECIMATION = 8 (240Hz 物理 / 30Hz 控制)
驱动方式: 纯 set_drive_target + physics_step, 禁止 set_qpos 传送
重力补偿: compute_passive_force(gravity=True)
```

### 1.4 两个残留的碰撞 hack

Stage3 仍在禁用碰撞 (Stage1 已删除, 造成"优化环境≠验收环境"):

- 手指-地面碰撞禁用
- 邻物碰撞禁用 (7 个 shape, 仅保留 target `glb_4`)

---

## 2. 当前问题 (量化)

### 2.1 主症: 夹持滑脱

```
[PhysicsValidation][stage3] 恒定距离: ref=7.1mm, max_dev=23.5mm (tol=5mm) → FAIL
[Stage 3] obj-mano距离: mean=100.6mm, F50=54.2mm, F80=152.9mm
lift_peak=2.8cm, lift_final=2.7cm
```

- `ref=7.1mm`: 手指中点确实**夹住了**物体 (间距 7mm)
- `max_dev=23.5mm`: 夹持过程中**滑脱 2.35cm**
- `obj-mano F50=54mm → F80=153mm`: 物体**没有跟随 MANO 轨迹**移动, 越落越远

→ 手指咬住了, 但抬升过程中从指间滑走。

### 2.2 lift 2.8cm 是设计上限内的表现, 非故障

v4.49 轨迹把抬升限制为 3.1cm, 实测 2.8cm 已接近上限。
**不代表 Stage3 "抓不起", 而是它本就没被设计成大幅抬升。**

---

## 3. 2026-08-31 修复的 Bug

### 3.1 F55 兜底是死代码 (引入)

```python
elif local_idx < 45:
    ...
    if _close_t0 is None and local_idx >= 55:   # ← 与外层 <45 互斥, 永不执行
```
兜底写在 `local_idx < 45` 块内而触发条件是 `local_idx >= 55`, 条件互斥。
→ 日志中从未出现 `F55 兜底强制闭合`。已改为 F45 兜底且置于可达位置。

### 3.2 F45-F49 分支 gap (既有)

```
local_idx < 25      → 阶段1
local_idx < 45      → 阶段2
local_idx <= 89     → 阶段3 (hold)
```
F45-F49 **无任何分支覆盖**, `gripper_pd_target` 继承 F44 的值。
原代码因触发条件宽松 (Δz<3cm) 在 F42 前就闭合, gap 被掩盖;
阈值收紧后暴露 → 下降段全程张开。
已将阶段2边界改为 `< _F_GRASP` (50) 补齐。

**修复效果**: `lift: 0.0cm → 2.8cm`

---

## 4. 被实测证伪的 plan 假设 (2026-08-29 计划)

### 4.1 plan ① "到位确认" 三条件物理上不可满足

实测诊断 (F25→F45, 逐 5 帧采样):

```
F25: dz=180.2mm  dxy=0.1mm   |vz|=0.00mm/帧
F30: dz=160.1mm  dxy=24.3mm  |vz|=8.41mm/帧
F35: dz=98.0mm   dxy=20.6mm  |vz|=14.01mm/帧
F40: dz=34.8mm   dxy=4.1mm   |vz|=16.84mm/帧
F45: dz=0.5mm    dxy=0.6mm   |vz|=1.61mm/帧   ← 唯一满足的一帧
```

- `|vz| < 2mm/帧` 在下降段**不可能满足** (实测 8-17mm/帧)
- 等 `dz<5mm` (F45) 才触发 + 10 帧过渡 = **F55 才闭合完**, 越过 F50 抓取帧

**结构性矛盾**: plan ① 的"到位确认"与 v4.41 的"边降边闭"互斥 ——
等到位再闭合, 闭合需要时间, 必然错过咬合窗口。

**修正**: 改为"接近触发" `dz<45mm / dxy<25mm / |vz|<20mm/帧`。
反推依据: 抓取帧 F50 − 10 帧过渡 ≈ F40 触发, F40 实测 dz=34.8mm。
修后 F42 触发闭合。

### 4.2 plan ② 去掉 -0.004 深压 → 无效

去掉深压: `lift=2.8cm, max_dev=23.5mm`
恢复深压: `lift=2.8cm, max_dev=23.5mm` (**一字不差**)

→ `_GRIP_PD_HOLD` 深浅对结果**毫无影响**: 手指被物体挡住停在接触点,
PD 目标深浅不改变最终位置。已回退 v4.42 值 (历史 v4.48 亦证明改浅会失败)。

### 4.3 plan ③ 释放自动检测未触发

MANO 开合信号 `(j1+j2)/2` 在 hold 段**从未回升**, 每次都走 `F95 兜底强制释放`。
序列本身不含"张开"意图, 自动检测逻辑正确但无条件可触发。

---

## 5. 物理参数统一 (v6.6, 2026-08-31)

修复前同参数两份定义:

| 参数 | physics_utils.py | physics_env.py | 修复后 |
|------|------------------|----------------|--------|
| `GRIPPER_FRICTION` | 2.5 | 2.0 (本地) | **2.0 单一来源** (utils 定义, env 导入) |
| `GRIPPER_FORCE` | 10.0 (死值) | =100N | 删除死值, 唯一来源 `FINGER_PD_FORCE_LIMIT` |

依据: `GRIPPER_FRICTION=2.0` 是 v4.14f 实验最佳 (3.0 会把物体卡住推不开 f1,
2.5 与 2.0 无差别)。`GRIPPER_FORCE=10.0` 与 100N 差 10 倍且从未被引用。

---

## 6. 遗留问题 (按优先级)

1. **夹持滑脱 23.5mm** (容差 5mm) — 主症。需动夹持时序/深度/几何楔紧,
   可能涉及 `wedge` 几何、接触点分布或摩擦系数实验。
   (CHANGE_LOG L81/L88 已记录为 v6.0b 时代欠债, 非本次引入)
2. **Stage3 两个碰撞 hack** (指地/邻物) 与"优化环境=验收环境"原则冲突,
   Stage1 已删, Stage3 未删。
3. **v4.49 抬升上限 3.1cm** — 若希望 Stage3 真正复现"人手抓起平移",
   该硬编码约束是根本限制, 需重新设计抬升段。
4. `enclose_z` / 抓取位姿几何可行性硬钳制、`_probe_ground_clearance`、
   `best_grasp.feasible` 元数据校验 (P0/P1 项, 见 2026-08-25 诊断)。

---

## 7. 复现命令

```bash
cd example/combination/trajectory_optimization
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py --test-stage0 --views none --side right
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py --test-stage1 --views none --side right
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py --test-stage2 --views none --side right
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py --test-stage3 --views none --side right
```

关键日志标记:
`[Stage3 到位确认][诊断]` (三条件实测) / `[PhysicsValidation][stage3]` (三指标验收)
