# Stage2/Stage3 物理仿真测试与诊断总结（2026-09-10）

> 本日任务：测试 `--test-stage2` / `--test-stage3` 物理仿真情况并对照 `2026-09-04-implementation-change-plan.md`；随后按决策表调参数验证；再逐帧诊断接触根因；最后在代码两次外部更新（clearance 走廊版 → min-z 平滑插值版）后复测。**本会话全程未修改任何仓库代码**（诊断脚本在 /tmp，通过包装 `_target_finger_contacts` 记录接触）。

---

## 1. 当前状态（文档最新结论，以此为准）

- **代码为 min-z 平滑插值版（v3）**：F38→F50 不再建立世界 Z 抬升走廊，采用平滑 MANO 修正（`min_z=0mm`）；`--stage2-pregrasp-retreat` 与 `--stage2-approach-clearance-z` 已废止，**兼容接收但提示废止并忽略**（[grasp_hawor.py#L4863-L4870](../grasp_hawor.py)）。
- **碰撞"小权限"政策已明确写进计划**：当前夹爪只与目标碰撞，可穿过 ground 和非目标物体；其他物体保留动力学；后期完成抓取、携带和放置后再逐级恢复完整碰撞。见 [2026-09-09-combined-mano-grasp-carry-v1-plan.md#L115](2026-09-09-combined-mano-grasp-carry-v1-plan.md)、[2026-09-04-implementation-change-plan.md#L96](2026-09-04-implementation-change-plan.md)。
- **单元测试 16/16 通过**（`tests/test_stage2_timeline.py + test_stage3_timeline.py`）。
- **Stage2 物理验证 FAIL（0/3）**：`APPROACH_TARGET_CONTACT prefix/adjust=15/24, objects=['glb_4']`，tracking=17.53/38.53mm，controls=63，bundle=891c0c1afad2。
- **Stage3 正确拒绝运行**（Stage2 gate 0/3 后入口直接返回，符合"失败不晋级"合同），Stage3A/3B 物理仍未验证。
- 产物：`stage2_bundle.npz` / `combined_bundle.npz` 均未生成（gate 未过，符合合同）。

### 1.1 核心问题：prefix=15 三个代码世代恒定，Z 高度手段已穷尽

| 代码世代 | approach 策略 | prefix 接触 | adjust 接触 |
|---|---|---|---|
| 旧版（retreat） | 起点后撤 0.08→0.20 | 15 | 25~37 |
| clearance 版 | 世界 Z 抬升 +0.080 | 15 | 24 |
| clearance 版 | 世界 Z 抬升 +0.120 | 15 | 18 |
| **min-z 版（当前）** | **平滑 MANO 修正，min_z=0** | **15** | **24** |

- **prefix=15（不可变 MANO_FOLLOW F0–F38 与 glb_4 重叠）在所有世代中恒定**，retreat 0.08→0.20、clearance 0.08→0.12、min-z=0 全部无效——这是**数据/场景级冲突**：pre_translate 把 F50 对齐物体中心后，MANO 早期路径本身穿过物体。
- adjust 段在 min-z=0 时回到 24（走廊贴住 MANO F38→F50，手指张开 0.05 从物体旁/顶部扫过）；clearance 0.120 曾降到 18 但从不清零。
- **结论（遵照计划约定）：不能再靠增加 Z 高度掩盖**。prefix=15 的解决路径只有两条：检查目标/夹爪碰撞体，或重新决定 prefix 接触政策。

### 1.2 下一步（需用户决策，二选一）

- **C. 检查碰撞体尺寸**（09-09 遗留排查点）：核查 finger_link2 / 夹爪碰撞几何与 glb_4 碰撞体，确认 prefix 重叠是真实手物重叠还是碰撞几何越界（资产级修复）。
- **A. 重新决定 prefix 接触政策**：若确认是"手本来就贴着物体"的真实重叠，则 MANO_FOLLOW 段目标接触只报告不计 fail，验收聚焦 APPROACH_ADJUST 及之后（gate 政策级修改）。

---

## 2. 代码版本

- min-z 平滑插值版（**当前 v3**）：`grasp_hawor.py`（`--stage2-approach-min-z` 默认 0.0，L4867；废止参数兼容 L4864-L4866）、`stage2_reconstruct_mixin.py`（gate 拆分、prefix/adjust 分段计数、碰撞 scope target-only）。
- 中间版（clearance 走廊）：世界 Z 抬升走廊 `safe_z = max(F38.z, F50.z) + clearance`。
- 旧版（retreat）：起点后撤走廊。
- bundle hash：旧版 baseline `972f8ba3b636`；clearance=0.080 `213a2faf1d9b`；clearance=0.120 `11655606f200`；min-z=0 `891c0c1afad2`。

## 3. 运行命令（canonical env）

```bash
export PY=/home/an/miniconda3/envs/dex/bin/python
export _OBJ_SCALE=0.6 _OBJ_SCALE_TARGET=glb_4 _GRIP_FRICTION=2.0 _SMOOTH_MINT=0
# 单测（16/16）
$PY -m pytest -q tests/test_stage2_timeline.py tests/test_stage3_timeline.py
# Stage2 min-z 版（当前主线 v3，输出目录独立隔离）
$PY grasp_hawor.py --test-stage2 --views none --side left \
    --hawor-dir /home/an/data/hawor/7a --ras-dir /home/an/data/ras/7_vggt_omega \
    --start-frame 0 --num-frames 113 --grasp-pose-method svd_palm \
    --stage2-approach-min-z 0.0 \
    --stage2-approach-time-scale 2.0 --stage2-carry-time-scale 2.0 \
    --output-dir output/hawor7a_left_target_only_v3
# Stage3 同参数将 --test-stage2 换为 --test-stage3
```

## 4. 测试演进（三个阶段）

### 4.1 阶段一：旧代码（retreat）参数实验（输出目录 `output/hawor7a_left/`）

旧版 gate 判定 `PRE_F50_READY: unexpected approach contacts`，决策表三旋钮穷尽无效：

| 实验 | 参数 | 接触帧数 | 结论 |
|---|---|---|---|
| baseline | retreat=0.08, ats=2.0 | 25/25 | 参数生效性参照 |
| E1 | retreat=0.120 | 25/25 | F38 实际后移 4cm、hash 变化 → 生效但无效 |
| E2 | q_near=0.040 | 25/25 | F46-49 开度实际 0.04 → 生效但无效 |
| E3 | ats=3.0 | 37/37 | 控制帧变多 → 全程接触，放慢更糟 |
| E4 | 组合 | 37/37 | 同上 |
| E5 | retreat=0.200 | 25/25 | 起点 z=0.126m、距目标 ~23cm 仍 25/25 → 接触源非"起点太近" |

### 4.2 阶段二：逐帧诊断（定旧根因，两根因已在后续版本修复）

/tmp 诊断脚本包装 `_target_finger_contacts` 逐帧记录（未改仓库代码）：

| 控制帧区间 | base_z | 接触对象 | 手指 |
|---|---|---|---|
| 帧 0-9（高空 src 38→41） | 0.126→0.108 | **glb_7** | finger_link2 |
| 帧 10-13（下降 src 42→45） | 0.097→0.071 | ground、**glb_7、glb_5**、glb_4(帧13起) | link1+link2 |
| 帧 14-24（低位 src 46→50） | 0.069→0.073 | **ground + glb_4** | link1+link2 |

旧版根因：① gate 把静态 ground 与邻物（glb_7/glb_5）计入意外接触；② 接近走廊被邻物阻断。

> 修复确认：clearance 版起 gate 拆分（目标接触 `APPROACH_TARGET_CONTACT` / 环境泄漏 `COLLISION_FILTER_LEAK` 分列）+ 碰撞 scope target-only（`gripper=2 target=1 non-target=7 shapes`），符合计划"碰撞小权限"政策——阶段一、二两条根因均消除，环境对象不再触发 gate。

### 4.3 阶段三：target-only 复测（两代代码，输出目录 `output/hawor7a_left_target_only_v2/` 与 `..._v3/`）

| 运行 | 代码 | 参数 | 结果 | 失败细节 |
|---|---|---|---|---|
| 单测 | clearance 版 | 两个 timeline 测试文件 | 14/14 通过 | — |
| C1 | clearance 版 | clearance-z=0.080 | 0/3 FAIL | `APPROACH_TARGET_CONTACT prefix/adjust=15/24`；tracking=12.19/38.39mm；bundle=213a2faf1d9b |
| C2 | clearance 版 | clearance-z=0.120 | 0/3 FAIL | `15/18`；bundle=11655606f200 |
| 单测 | min-z 版 | 两个 timeline 测试文件 | **16/16 通过** | — |
| **V3** | **min-z 版（当前）** | **min-z=0.0（用户给定命令）** | **0/3 FAIL** | **`APPROACH_TARGET_CONTACT prefix/adjust=15/24`；tracking=17.53/38.53mm；bundle=891c0c1afad2** |

min-z 版走廊（日志实测）：F38→F50 平滑 MANO 修正，`min_z=0mm`，q_start=0.0500（F50 前固定全开）→ q_grasp=0.0170，controls=63，三次 repeat 完全一致（确定性）。

## 5. 与 2026-09-04 计划的对照（按当前状态）

| 计划条款 | 状态 |
|---|---|
| P0-5"先解决 approach 提前接触" | 环境误报已消除（碰撞小权限政策落地）；剩 glb_4 目标接触：prefix=15 为数据/场景级冲突（待碰撞体核查或政策决策），adjust=24 为几何逼近擦碰 |
| §7.6 Stage2 门禁（repeat≥2/3, tracking≤5/15mm） | 未达成（tracking 17.53/38.53mm） |
| §4.2"失败不晋级、只保存通过门禁的 bundle" | 已实现并生效（未写 stage2_bundle.npz） |
| §8.1"Stage3 必须消费通过 2/3 门 bundle，失败即停" | 已实现并生效 |
| §5.3 F50 3+2 快抓时序 / 显式 timeline / relative SE(3) | 代码已落地（16 单测过）；CLOSE/HOLD/CARRY 因 approach 门禁前置仍未物理验证 |
| 碰撞"小权限"政策（v1 计划 L115 / 实施计划 L96） | 已实现并生效（target-only scope；ground 与非目标可穿过，物体保留动力学） |

## 6. 日志路径清单

### 当前主线（min-z 版 v3，`output/hawor7a_left_target_only_v3/`）

| 日志 | 内容 | 结果 |
|---|---|---|
| `stage2_minz_v3_0910.log` | V3：min-z=0.0（用户命令原样） | 0/3 FAIL(APPROACH_TARGET_CONTACT 15/24) |
| `stage1/best_grasp.npz`、`stage1/grasp.log` | V3 运行时自动生成的 Stage1 产物 | — |

### 中间版（clearance，`output/hawor7a_left_target_only_v2/`）

| 日志 | 内容 | 结果 |
|---|---|---|
| `stage2_clearance080_0910.log` | C1：clearance=0.080 | 0/3 FAIL(15/24) |
| `stage2_clearance120_0910.log` | C2：clearance=0.120 | 0/3 FAIL(15/18) |
| `stage1/best_grasp.npz` | C1 运行时自动生成的 Stage1 产物 | — |

### 历史（旧版 retreat，`output/hawor7a_left/`）

| 日志 | 内容 | 结果 |
|---|---|---|
| `stage2_run_0910_physical.log` | 旧版 Stage2 基线 | 0/3 FAIL(PRE_F50_READY, 25) |
| `stage3_run_0910_physical.log` | 旧版 Stage3 入口测试 | Stage2 gate FAIL → 不运行 |
| `stage2_e1_retreat120.log` ~ `stage2_e5_retreat200.log` | E1-E5 参数实验 | 全 FAIL（见 §4.1） |
| `stage2_gate.log` | 09-09 21:11 旧基线（对照） | 0/3 FAIL(25) |

### /tmp 诊断（非仓库产物）

| 文件 | 内容 |
|---|---|
| `/tmp/s2_contact_diag.py` | 接触逐帧诊断脚本（包装 `_target_finger_contacts`） |
| `/tmp/s2_contact_diag.log`、`/tmp/s2_contact_diag_r200.log` | 逐帧接触表（retreat=0.08 / 0.20，含 base_z） |

**无崩溃型报错**：所有运行 exit 0、无 Python traceback；"失败"均为 gate 判定（当前搜 `APPROACH_TARGET_CONTACT`，历史搜 `PRE_F50_READY`）。

## 7. 相关文档

- 计划：[docs/2026-09-04-implementation-change-plan.md](2026-09-04-implementation-change-plan.md)、[docs/2026-09-09-combined-mano-grasp-carry-v1-plan.md](2026-09-09-combined-mano-grasp-carry-v1-plan.md)
- Q&A 记录：[docs/questions.md](questions.md)
