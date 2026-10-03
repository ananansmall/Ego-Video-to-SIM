# PhysX 驱动管线 (05_gripper_test.py)

> 对应代码: `example/combination/05_gripper_test.py`
> 核心函数: `step_physics_drive()` (L528-555)
> 结论: **每一步物理推进都在用 PhysX，它是唯一的力学计算者。**

---

## 1. 关节驱动参数

文件里有两组刚度，按测试用途选用（均为初始化时通过 `set_drive_property` 注册一次）:

```python
# ── A组: 抓取场景 (test5/test6/test7/test8, L1467-1469) ──
# 平移关节 (vx/vy/vz)
active_joints[idx_vx].set_drive_property(stiffness=5000, damping=1000)
# 旋转关节 (rz/ry/rx)
active_joints[idx_rz].set_drive_property(stiffness=1000, damping=200)
# 夹爪关节 (j1/j2) — K=3000: 25cm高空抓持, 更强夹力防滑落
active_joints[idx1].set_drive_property(stiffness=3000, damping=300)

# ── B组: MANO 轨迹跟踪 (test9_mano_trajectory, L2409-2411) ──
# 平移关节 (vx/vy/vz)
active_joints[idx_vx].set_drive_property(stiffness=30000, damping=3000)
# 旋转关节 (rz/ry/rx)
active_joints[idx_rz].set_drive_property(stiffness=10000, damping=1000)
# 夹爪关节 (j1/j2)
active_joints[idx1].set_drive_property(stiffness=10000, damping=1000)
```

| 关节组 | A组 抓取 (test8) | B组 轨迹跟踪 (test9) | 选型原因 |
|---|---|---|---|
| 平移 vx/vy/vz | K=5000, D=1000 | K=30000, D=3000 | 跟踪 MANO 手要极硬；抓取太硬会把方块撞飞 |
| 旋转 rz/ry/rx | K=1000, D=200 | K=10000, D=1000 | 同上 |
| 夹爪 j1/j2 | K=3000, D=300 | K=10000, D=1000 | 抓取靠接触力，适度柔软更稳 |

> 注意: `set_drive_property` 只在测试开始时调用**一次**，
> 之后 PhysX 引擎每个子步自动用这组 K/D 算弹簧力，Python 不再干预。

## 2. 时间参数

| 参数 | 值 | 含义 |
|---|---|---|
| PHYSICS_TIMESTEP | 1/240s | 物理引擎每步时间 |
| 物理频率 | 240Hz | 引擎每秒算 240 次 |
| DECIMATION | 16 | 每帧渲染前跑 16 步 |
| 每帧模拟时间 | 16 × 1/240 = 1/15s | 一帧覆盖 1/15 秒 |
| 视频 fps | 30 | 视频每秒播 30 帧 → 2x 慢动作 |

## 3. 每帧物理步进

每帧 (1/15s 模拟时间) → 16 次子步 (1/240s each):

```
初始化 (仅一次)                     SAPIEN 引擎
──────────                        ─────────────
set_drive_property(K, D)  ──────→  注册弹簧参数

每帧循环 (30Hz)
──────────
set_drive_target(x)        ──────→  更新目标位置 (只写内存)
compute_passive_force()    ←────  算出重力+科氏补偿 qf
set_qf(qf)                 ──────→  施加被动力

scene.step() × 16  ────────────→  引擎内部 (每个子步):
                                    F_drive = K·(target-qpos) - D·qvel
                                    F_total = F_passive + F_drive + F_contact
                                    碰撞检测 (手指↔方块↔桌面)
                                    求解 M(q)·q̈ = F_total
                                    积分更新 qpos, qvel
```

分步说明:

① `set_drive_target()`
   ← Python 层设置每个关节的目标值

② `compute_passive_force(gravity=True, coriolis_and_centrifugal=True)`
   ← Python 层计算被动力（重力 + 科里奥利力）

③ `set_qf(qf)`
   ← 施加被动力，抵消重力，机械臂悬浮

④ `scene.step()` × 16
   ← SAPIEN 引擎内部:
     - 对每个关节计算 F_drive = K·(target - qpos) - D·qvel
     - 合成总力 = F_passive + F_drive
     - 求解 M(q)·q̈ = F_total
     - 积分更新 qpos, qvel

## 4. 核心机制

- **弹簧模型**: 每个关节连着一根弹簧，另一端固定在目标位置。
  离目标越远拉力越大（K），动得越快阻力越大（D）
- **重力补偿**: 每帧算出重力，反向施加等大力抵消，机械臂悬浮在空中
- **16 步子步**: 一帧覆盖 1/15 秒，拆成 16 步每步 1/240 秒，
  运动平滑、数值稳定

---

## 附: 与 GalaxeaManipSim 的一致性

`step_physics_drive` 的 docstring 明确写着:

> Mirrors BimanualManipulationEnv.step() exactly:
>   qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
>   robot.set_qf(qf)
>   scene.step()

即本文件的驱动链路与 `src/GalaxeaManipSim` 的环境步进完全同构，
是同一套 PhysX 驱动范式在独立场景中的复现。

实测佐证 (2026-08-25 test8): 方块最高升至 22.1cm 并稳定保持，
最终放置 (43.7, 20.3) 落桌 —— 方块能克服重力升起，
只能是 PhysX 接触求解算出的手指夹持力，运动学脚本做不到。
