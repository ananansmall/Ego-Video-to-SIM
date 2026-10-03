# Stage2 / Stage1-recipe 实施与复测报告（2026-09-10 晚）

> 本日任务：按 `2026-09-04-implementation-change-plan.md` §3 实施代码改动，并按 §4 顺序执行
> T0 → T1 → T2 → T3，给出可复核的物理结果。
>
> 本轮**修改了仓库代码**（见 §2），与本目录下前一份诊断文档
> `2026-09-10-stage2-stage3-physical-test-and-diagnosis.md`（全程未改代码）不同。
>
> 原工具返回：T0 通过、T1 v2 返回 3/3、T2 为 0/3、T3 正确拒绝。
>
> **2026-09-10 复核更正：T1 v2 的 3/3 不能作为有效 PASS。**入口检查前的 5 个 warmup 已把目标从 reset pose 推移约 44.9 mm，且 base/q 也未达到规范入口；严格合同下 T1 应记为 0/3。完整更正见 §0，原始运行返回值和日志仍保留，避免篡改实验记录。
>
> **2026-09-10 后续执行决定：本报告只保留为历史实验记录。**Stage1 v2 的局部成功
> 已冻结，本轮不再重测 Stage1，也不再要求独立 transfer 测试先 PASS。入口审计发现的
> “张开夹爪不能直接静止到旧 anchor”问题，改由 Stage2 的 F0-F50 全段 0.12 m
> 预抓取抬升、
> F50 后下降—闭合 wrapper 和额外保持帧解决。当前执行计划见
> `2026-09-10-stage2-preprocessed-mano-grasp-replay-plan.md`；这不修改本报告中的
> 原始结果，只改变下一步工程路线。

---

## 0. 复核更正：区分“工具返回 PASS”和“物理合同 PASS”

本报告后文的 `T1 v2 3/3 PASS` 是当时实现的 gate 返回值，不是复核后的最终验收结论。原始 JSON/日志显示：

```text
target reset      = [-0.3022,-0.4945,0.0236]
target at entry   = [-0.2844,-0.4573,0.0415]  → drift ≈44.9 mm
base target       = [-0.3208,-0.5016,0.0505]
base actual entry = [-0.3255,-0.5111,0.0606]  → error ≈14.6 mm
q command         = [0.05,0.05]
q actual entry    = [0.0750,0.0149]            → max error ≈35.1 mm
recorded final object_drift=37.13 mm, object_drift_ok=false
```

`verify_stage1_transfer_equivalence()` 当前只用
`not entry_target_contact and settle_pass` 构造 `run_pass`：它在物体已被 warmup 推开后才检查 contact，并未把 drift、base/q tracking、速度或缺失指标纳入硬门。因此最终审计状态为：

| 项目 | 原工具结果 | 复核结论 |
|---|---:|---:|
| Stage1 v1 local lift | 3/3 | 保留，`legacy_local_lift_pass` |
| Stage1 v1 transfer | 0/3 | FAIL |
| Stage1 v2 local lift | 3/3 | 保留，`legacy_local_lift_pass`；不等于新标准完整抓取 |
| Stage1 v2 entry clearance | 未设硬门 | FAIL |
| Stage1 v2 transfer | 3/3 | **严格合同下 0/3 FAIL** |
| Stage2 v2 | 0/3 | FAIL，且尚未执行到 F50 20+10 |

所以旧 Stage1 成功不删除；问题是它只证明“局部放置后能夹起”，没有证明“能从 Stage2 的张开入口无扰动到达”。下一步先做静态 collision geometry audit 和严格入口门，不能把 prefix contact 改为 report-only。

另一个限制是：当前 `stage1_reuse_report PASS` 只证明 recipe 已纳入的 scene 字段和 `q_grasp` 一致；recipe 尚未保存并校验导出时的 `pos/R` anchor hash，也未覆盖完整 collision-shape/inertia/material/solver 摘要。因此该 PASS 保留为“当前字段一致”，不能表述为完整 Stage1 物理身份已经全部绑定。

---

## 1. 结论摘要

| 步骤 | 内容 | 结果 |
|---|---|---|
| T0 | `tests/test_stage2_timeline.py` + `tests/test_stage3_timeline.py` | **23/23 PASS** |
| Stage1 v1 | 旧入口（含 `_DESCEND_OPEN` 隐藏 warmup）→ 生成 recipe | 验证门 3/3；anchor `pos=[-0.3181,-0.5123,0.0427]` |
| T1（对 v1） | `F50_TRANSFER_EQUIVALENCE` | **0/3 FAIL**，物体被弹飞 `drift=627.01mm` |
| Stage1 v2 | 规范入口 `q_open=0.05`（去掉隐藏 warmup）→ 新 revision | 验证门 3/3（lift 19.4cm）；anchor `pos=[-0.3208,-0.5016,0.0505]` |
| T1（对 v2） | `F50_TRANSFER_EQUIVALENCE` | 原工具 **3/3 PASS**；严格复核 **0/3 FAIL**（入口前目标已漂移约 44.9 mm） |
| T2 | `--test-stage2` stage1_target_isolated 基线 | **0/3 FAIL** `APPROACH_TARGET_CONTACT prefix/adjust=15/24` |
| T3 | `--test-stage3` | **正确拒绝运行**（Stage2 gate 0/3） |

三条关键事实：

1. **scale / 位置一致性合同已经生效**：Stage2 启动时逐字段比对 Stage1 recipe 的
   `scene_digest`，不一致 → `STAGE1_REUSE_MISMATCH`，缺 recipe → `STALE_STAGE1_ARTIFACT`，
   都不进入 rollout。T2 三次运行的 `stage1_reuse_report` 全部 PASS（`digest=d86661319f87`）。
2. **Stage1 anchor 的“可迁移性”是真实存在的门槛**：旧 Stage1 的抓取成功依赖
   `qpos[gripper]=_DESCEND_OPEN=0.012` 的隐藏 warmup。换成规范入口 `q=0.05` 后，
   同一个 anchor 会与目标互穿、把物体弹飞 35cm（T1 0/3）。去掉隐藏 warmup 重新生成
   revision 后，旧 gate 返回 3/3；但严格复核发现 warmup 已推动目标，因此不能晋级。
3. **T2 的首个 failure 与之前三个代码世代完全相同**（`prefix/adjust = 15/24`）。
   这证明本轮 recipe/碰撞改动没有解决该冲突；它强烈指向数据/碰撞几何或入口设计，
   但仅凭相同计数不能严格证明与所有代码变化完全无关。

---

## 2. 本轮的代码改动

按计划 §3.5 的“最小修改顺序”实施：

### 2.1 新增 `stage1_recipe.py`（共享配方 + 场景/物体摘要）

- `STAGE1_CLOSE_CONTROLS=20`、`STAGE1_SETTLE_CONTROLS=10`、
  `close_profile=cubic_smoothstep`（`s(t)=t²(3-2t)`）、`q_open` / `q_grasp`；
- `stage1_close_q_sequence()`：Stage1 与 Stage2 共用同一条闭合序列，禁止复制常量；
- `build_scene_digest()`：把**必须一致**的字段固定下来
  `target_object / obj_scale_env / obj_scale_target_env / bbox_size / bbox_min / bbox_max /
  body_type / initial_pose_pos / initial_pose_quat / mass / ground_z / gravity / physics`；
- `recipe_mismatches()` / `scene_digest_hash()` / `load_stage1_grasp()`。

> 注意 `bbox_size` 是 `_OBJ_SCALE` 施加之后的真实几何，`initial_pose_pos` 是两阶段
> rollout 共同恢复的**重置位姿**——这正是“抓取物体的 scale 与位置必须一致”的落点。

### 2.2 `stage1_grasp_mixin.py`

1. 闭合序列改为调用 `stage1_close_q_sequence(GRIPPER_MAX_OPEN, _GRIP_TARGET, 20)`
   （数值与历史 `_close1_smooth` 逐值一致，仅去掉重复实现）；
2. **入口改为规范状态**：`qpos[gripper]=GRIPPER_MAX_OPEN`，warmup 命令保持 `q_open`，
   删除 `_DESCEND_OPEN=0.012` 的隐藏 warmup（计划 §3.1 第 2 条）；
3. 新增**入口自检**：warmup 后若已与目标接触则记 `_last_entry_target_contact`，
   `stage1_reward` 对其扣 20 分（引导 CEM 选择可迁移 anchor）；
4. 导出 `stage1_grasp_recipe`：写入 `stage1/best_grasp.npz`（新增
   `target_object/q_open/close_controls/settle_controls/scene_digest_hash/stage1_recipe_json`）
   并并列 `stage1/stage1_grasp_recipe.json`；
5. 同进程内把 recipe 挂在 `best_output['stage1_recipe']` 上，供 Stage2/Stage3 直接复用。

### 2.3 `stage2_reconstruct_mixin.py`

1. **F50 时间线改为共享 20+10**：`F50_CLOSE_CONTROLS=20`、`F50_HOLD_CONTROLS=10`，
   phase 名 `STAGE1_CLOSE` / `STAGE1_SETTLE`，第 31 个 F50 控制点之后才进入
   5 点 `CARRY_VERIFY`；`build_stage2_timeline(recipe=...)` 会校验闭合命令与
   Stage1 recipe 逐值一致，不一致直接抛错；
2. **碰撞矩阵反转**：重命名 `_configure_stage1_target_isolated_collisions`，
   `STAGE1_TARGET_ISOLATED_CONTACT_MASKS` = `gripper↔target ON`、
   `gripper↔ground/default ON`（复用 Stage1 真实物理）、`target↔ground ON`、
   `gripper↔non_target OFF`、`target↔non_target OFF`、
   `non_target↔ground ON`、`non_target↔non_target ON`；
3. **接触分类拆分**：新增全局 `_audit_contact_pairs()`，trace 分列
   `target_contact` / `ground_contact` / `non_target_contact`；
   只有被禁用的两对（`gripper|target` ↔ `non_target`）才触发
   `COLLISION_FILTER_LEAK`，ground 接触只进诊断；
4. **新增 `_stage1_reuse_report()`**：在 rollout 之前逐字段比对 recipe 与当前场景，
   生成 `stage1_reuse_report` 并写入 manifest / bundle / 保存产物；
5. `PROVISIONAL_GRASP` 改为按 10 点 settle 判定（≥8/10 且末 3 点双侧）；
6. **新增 `verify_stage1_transfer_equivalence()`**（T1 的实现）。

### 2.4 `grasp_hawor.py`

- 新增 `--test-stage1-transfer` 入口；
- `--test-stage2` / `--test-stage3` 改用 `load_stage1_grasp()`，缺 recipe 时明确告警
  `STALE_STAGE1_ARTIFACT`；
- 日志文案更新为 `F50(20 STAGE1_CLOSE + 10 STAGE1_SETTLE)`。

### 2.5 单元测试

`tests/test_stage2_timeline.py` 重写断言：20/10 控制点数与 phase 名、闭合命令与 recipe
逐值一致、recipe 不可复现时抛错、`stage1_target_isolated` 碰撞矩阵 7 对关系、
`stage1_reuse_report` 四种结论（PASS / scale 不符 / 位置不符 / 缺 recipe / anchor 与
recipe 冲突）。

---

## 3. 运行环境与命令

```bash
export PY=/home/an/miniconda3/envs/dex/bin/python
export _OBJ_SCALE=0.6 _OBJ_SCALE_TARGET=glb_4 _GRIP_FRICTION=2.0 _SMOOTH_MINT=0
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization

# T0 单元测试
$PY -m pytest -q tests/test_stage2_timeline.py tests/test_stage3_timeline.py

# Stage1（规范入口）→ 生成带 recipe 的 revision
$PY grasp_hawor.py --test-stage1 --views none --side left \
    --hawor-dir /home/an/data/hawor/7a --ras-dir /home/an/data/ras/7_vggt_omega \
    --start-frame 0 --num-frames 113 --grasp-pose-method svd_palm \
    --output-dir output/hawor7a_left_stage1_target_isolated_v2

# T1
$PY grasp_hawor.py --test-stage1-transfer --views none --side left \
    --hawor-dir /home/an/data/hawor/7a --ras-dir /home/an/data/ras/7_vggt_omega \
    --start-frame 0 --num-frames 113 --grasp-pose-method svd_palm \
    --output-dir output/hawor7a_left_stage1_target_isolated_v2

# T2（计划 §4 T2 参数原样）
$PY grasp_hawor.py --test-stage2 --views none --side left \
    --hawor-dir /home/an/data/hawor/7a --ras-dir /home/an/data/ras/7_vggt_omega \
    --start-frame 0 --num-frames 113 --grasp-pose-method svd_palm \
    --stage2-approach-min-z 0.0 \
    --stage2-approach-time-scale 2.0 --stage2-carry-time-scale 2.0 \
    --output-dir output/hawor7a_left_stage1_target_isolated_v2

# T3（同参数，--test-stage3 + --stage3-carry-time-scale 2.0 --stage3-release-candidate 90）
```

---

## 4. T0 单元测试

```text
TOTAL 23 PASSED 23 FAILED 0
```

覆盖点：F50 = 20 `STAGE1_CLOSE` + 10 `STAGE1_SETTLE`；`source_frame` 单调不减且
`close_start` 前最后一个 source < 50；`verify_start=50.2`/`verify_end=51.0`；
`pre_f50_end` 的 objective 为 `avoid_target_contact`、`close_start` 起切到
`maintain_bilateral_contact`；闭合命令与 `stage1_recipe.stage1_close_q_sequence` 逐值一致；
`stage1_target_isolated` 七对碰撞关系；`stage1_reuse_report` 的 PASS / bbox_size 不符 /
initial_pose_pos 不符 / 缺 recipe / q_grasp 冲突；Stage3 前缀逐字节不变。

> 初始版本有 2 个断言写错（JSON 9 位小数取整、移动用例改的是 actor 当前位姿而 digest 依据
> 重置位姿），已修正后 23/23。

---

## 5. Stage1 revision 对比（v1 旧入口 vs v2 规范入口）

两轮的**场景/物体 digest 完全相同**（`d86661319f87`），说明差异只在 anchor：

| 项 | v1（旧入口，含隐藏 warmup） | v2（规范入口 q_open=0.05） |
|---|---|---|
| `qpos[gripper]` 入口 | `_DESCEND_OPEN=0.012` | `GRIPPER_MAX_OPEN=0.05` |
| 验证门 `lift_final` | `[20.9, 20.0, 20.9] cm` → 3/3 | `[19.4, 19.4, 19.4] cm` → 3/3 |
| anchor `pos` | `[-0.3181, -0.5123, 0.0427]` | `[-0.3208, -0.5016, 0.0505]` |
| `gripper_qpos` | `0.0170` | `0.0170` |
| 入口自检（q_open 下是否已接触目标） | — | **所有候选 `f2=True`（已接触）** |
| T1 结果 | **0/3 FAIL**，`drift=627.01mm` | 原工具 3/3；严格复核 **0/3 FAIL**，reset→entry≈44.9mm |

两轮 recipe 的 `scene_digest` 一致：

```text
target=glb_4
bbox=[0.053878659, 0.067053777, 0.047149126]      # _OBJ_SCALE=0.6 之后
init_pos=[-0.3021809, -0.4944604, 0.0235746]
mass=0.170338839
q_open=0.0500  q_grasp=0.0170  close/settle=20/10
digest=d86661319f87…cb567f
```

---

## 6. T1 `F50_TRANSFER_EQUIVALENCE`

### 6.1 对 v1（旧 anchor）：0/3 FAIL —— 定位到“隐藏初始化依赖”

逐 warmup 步追迹（`--output-dir ..._v1`）：

```text
reset_obj = [-0.3022, -0.4945, 0.0236]        # 与 recipe 一致
warmup 0  obj = [-0.3138, -0.5225, 0.1264]    # 一个控制步(33ms)内 z +10cm
warmup 4  obj = [-0.3575, -0.7163, 0.3727]    # 共 +35cm，随后抛体下落
k=00..29  f1=False f2=False  contacts=['ground']
repeat 1/3: FAIL entry_target_contact=False settle_bilateral=0/10 drift=627.01mm
```

即在旧 anchor 处、夹爪张开（q=0.05）时**与目标实体互穿**，一上物理就把物体弹飞。
旧 Stage1 之所以“抓成功”，是因为它先把手指收到 `0.012` 再 warmup，物体只被顶起
~4.5cm（Stage1 debug `obj_z=0.0683 @F0`）后被空中夹住——这正是计划 §3.1 说的
**“旧 Stage1 成功依赖隐藏初始化”**，结论是回到 Stage1 生成并验证新 revision，
不得在 Stage2 内补丁式修复。

### 6.2 对 v2：原工具返回 3/3，严格入口合同为 0/3 FAIL

```text
repeat 1/3: PASS entry_target_contact=False settle_bilateral=10/10 drift=37.13mm
repeat 2/3: PASS entry_target_contact=False settle_bilateral=10/10 drift=37.13mm
repeat 3/3: PASS entry_target_contact=False settle_bilateral=10/10 drift=37.13mm
F50_TRANSFER_EQUIVALENCE: 3/3 → PASS
```

上面是原工具日志，不是复核结论。工具确实证明 20+10 与 artifact 一致、settle 为
10/10；但它没有证明规范入口。入口检查前 warmup 已造成约 44.9mm 目标漂移，实际 base/q
也偏离命令。`object_drift=37.13mm`（`object_drift_ok=False`）同样必须纳入入口/迁移硬门，
不能再描述成“只记录、以后再判”。因此本节最终结论为严格 T1 0/3。

碰撞作用域（T1 报告）：

```text
scope=stage1_target_isolated  gripper=2 shapes  target=1 shapes  non-target=7 shapes
gripper_ground_pair=True   gripper_target_pair=True
gripper_non_target_pair=False  target_non_target_pair=False
target_environment_pair=True   non_target_environment_pair=True  non_target_non_target_pair=True
```

T1 末态 `final_contacts=['glb_4','ground']` 且**未**触发 `COLLISION_FILTER_LEAK`
—— 证明 ground 接触不再被误判为泄漏（计划 §3.4 的拆分生效）。

---

## 7. T2 Stage2 物理基线

```text
[Stage2] stage1_reuse_report PASS: {... 'scene_digest_hash':'d86661319f87…'}
[Stage2] nominal geometry F38-F62: …, q_start=0.0500, q_grasp=0.0170, bundle=2c9a4b7d08d8
[Stage2 collision scope] stage1_target_isolated: gripper=2 target=1 non-target=7 shapes
[Stage2 gate] FAIL(APPROACH_TARGET_CONTACT) bundle=2c9a4b7d08d8 controls=63 tracking=16.94/38.39mm
[test-stage2] repeat 1/3: FAIL failure=APPROACH_TARGET_CONTACT
             detail=target contact prefix/adjust=15/24, objects=['glb_4']
[test-stage2] repeat 2/3: FAIL （同上，逐值一致）
[test-stage2] repeat 3/3: FAIL （同上，逐值一致）
[test-stage2] repeat gate: 0/3 → FAIL ✗
```

- `stage1_reuse_report` 三次全部 **PASS**：Stage2 确认自己重放的是同一场景、同一物体
  scale（bbox）与同一重置位置、同一质量/地面/求解器参数，非目标隔离是唯一白名单差异。
- 失败发生在 `close_start`（第 63 个控制点 = 39 prefix + 24 approach），因此
  `trace` 只有 63 个控制点，**20+10 闭合段在物理上没有被执行到**
  （其正确性由 T0 单测与 T1 的 10/10 settle 双侧共同覆盖）。
- `prefix/adjust = 15/24` 与旧三代码世代的记录完全一致：
  prefix=15 是 F0–F38 预处理 MANO 路径（q=0.05）与 `glb_4` 重叠的控制点数；
  adjust=24 是 F38–F50 平滑校正路径**每一点**都接触。
- 三次 repeat 结果逐值一致（确定性）。
- 未生成 `stage2_bundle.npz`（合同要求“失败不保存”）。

---

## 8. T3 Stage3

```text
[test-stage3] Stage 1: 加载 …_v2/stage1/best_grasp.npz
[test-stage3] Stage 2: 25 帧重建完成
[test-stage3] Stage2 repeat gate FAIL: 0/3
              failures=['APPROACH_TARGET_CONTACT','APPROACH_TARGET_CONTACT','APPROACH_TARGET_CONTACT'];
              Stage3 不运行
```

符合“Stage3 必须消费通过 2/3 门的 bundle，失败即停”的合同，Stage3A/3B 本轮仍未物理验证。

---

## 9. 与 2026-09-04 计划的对照

| 计划条款 | 状态 |
|---|---|
| §3.1 共享 20+10 recipe，Stage2 只读不硬编码 | **已实现**（T0 逐值断言 + T1 物理 10/10） |
| §3.1 Stage1 artifact 输出 recipe + 场景/物体 digest | **已实现**（npz + JSON，`revision=stage1_grasp_recipe_v1`） |
| §3.1 `STALE_STAGE1_ARTIFACT` / `STAGE1_REUSE_MISMATCH` fail-fast | **已实现**（T0 覆盖 4 种结论；T2 三次 PASS） |
| §3.1 F50_TRANSFER_EQUIVALENCE 规范入口 ≥2/3 | 已实现初版，但 gate 漏掉 drift/base/q/speed；原返回 3/3，严格复核 0/3 |
| §3.1 F0–F38 prefix / F38–F50 平滑校正 / F50 30 点 / F50→F51 5 点 / F51–F62 carry | 结构已实现；**F50 之后的物理段尚未到达** |
| §3.2 撤销 Z 抬升走廊，仅 min-z 下界 | 保持（`min_z=0.0`，未新增抬升） |
| §3.3 分相位 objective 权重 | 已实现（`STAGE1_CLOSE` / `STAGE1_SETTLE` 分列） |
| §3.4 `stage1_target_isolated` 碰撞矩阵 + 接触拆分 | **已实现**（矩阵/分类均有单测；T1 实测 ground 不误判） |
| §4 T0 全部通过 | **达成 23/23** |
| §4 T1 三次至少两次同时满足规范入口和双侧保持 | **未达成**；旧 gate 只覆盖双侧，严格复核 0/3 |
| §4 T2 2/3 物理门、tracking ≤5/15mm | **未达成**（0/3；tracking 16.94/38.39mm） |
| §4 T3 Stage2 replay hash/物理门 | 未运行（Stage2 未通过，符合合同） |
| §7 完成定义 | **未达成**，状态仍是“待测试，不是已完成” |

---

## 10. 当前根因与待决策项

T2 的首个 failure 在三个旧世代中同为 `15/24`，说明本轮代码改动**没有解决它**；当前证据更支持数据/碰撞几何或入口设计问题，但不把“计数相同”写成严格因果证明。本轮新增的证据是：

1. T1 显示 **anchor 本身在 q_open 入口就与 `glb_4` 有接触**（自检 `f2=True`，
   闭合过程中把物体推移 37mm）。也就是说 MANO 派生几何与目标的贴近/重叠不是
   prefix 独有现象，而是贯穿整条轨迹。
2. 因此 `prefix=15` / `adjust=24` 更像**几何/碰撞体层面的贴近**，而不是姿态或时间参数问题。

按 v13/v5 计划，下一步先固定为静态碰撞几何审计，再根据证据选择一个分支：

- **C. 碰撞体核查**（推荐）：核查 `finger_link1/2`、夹爪 palm 的碰撞几何与 `glb_4`
  的碰撞盒（当前 dynamic 物体用 bbox 盒形碰撞）是否与视觉/真实几何越界；
  同时用同一 trace 口径对照 Stage0 的 `phase/contact_objects/contact_links`。
- **真实重叠分支**：保持目标接触硬门，Stage1 将 q_open clearance/reset→entry drift
  作为硬约束；单一 pose 无解时输出可复用的 `pregrasp→insertion→20+10` 局部 primitive。

不再采用“prefix 只报告不 fail”：目标是 dynamic actor，提前接触会真实改变 F50 初态。
在审计完成前，**不应**再调 time-scale / min-z / 力的旋钮，也不应重复跑 CEM 或放宽 T2 门限。

---

## 11. 产物与日志清单

### 输出目录

| 目录 | 内容 | 结果 |
|---|---|---|
| `output/hawor7a_left_stage1_target_isolated_v1/` | 旧入口 Stage1 + T1（3 次带诊断重跑） | Stage1 3/3；**T1 0/3** |
| `output/hawor7a_left_stage1_target_isolated_v2/` | 规范入口 Stage1、T1、T2、T3 | Stage1 local 3/3；T1 原工具 3/3、严格复核 0/3；**T2 0/3**；T3 不运行 |

v2 目录产出：

| 文件 | 内容 |
|---|---|
| `stage1/best_grasp.npz` | anchor + `stage1_recipe_json` + `scene_digest_hash` |
| `stage1/stage1_grasp_recipe.json` | 可读的 recipe（含 20 点 `close_ctrl_q`） |
| `stage2/stage1_transfer_equivalence.json` | T1 报告（每 repeat 的 warmup 位姿、逐控制双侧、漂移） |
| `stage2/grasp.log` | T2 运行日志（读目录时会被覆盖，见 /tmp 备份） |

### 日志（仓库外，供复核）

| 日志 | 内容 | 结果 |
|---|---|---|
| 进程内直跑（无管道，见 §4 输出） | T0 单测 | 23/23 |
| `/tmp/s1_isolated.log` | Stage1 **v1**（旧入口） | 3/3，anchor z=0.0427 |
| `/tmp/s1_transfer.log` `/tmp/s1_transfer2.log` `/tmp/s1_transfer3.log` | T1 对 v1（逐 warmup 追迹） | **0/3**，drift 627mm |
| `/tmp/s1_iso_v2.log` | Stage1 **v2**（规范入口） | 3/3，anchor z=0.0505 |
| `/tmp/s1_transfer_v2.log` `/tmp/s1_transfer_v2b.log` | T1 对 v2 | 原工具 3/3；严格复核 0/3，reset→entry≈44.9mm |
| `/tmp/s2_v2.log` | T2 | **0/3** `APPROACH_TARGET_CONTACT 15/24` |
| `/tmp/s3_v2.log` | T3 | 正确拒绝运行 |

**无崩溃型报错**：全部运行 exit 0、无 Python traceback；“失败”均为 gate 判定。

## 12. 相关文档

- 实施计划：[docs/2026-09-04-implementation-change-plan.md](2026-09-04-implementation-change-plan.md)
- 组合标准：[docs/2026-09-09-combined-mano-grasp-carry-v1-plan.md](2026-09-09-combined-mano-grasp-carry-v1-plan.md)
- 前一次物理诊断（未改代码）：[docs/2026-09-10-stage2-stage3-physical-test-and-diagnosis.md](2026-09-10-stage2-stage3-physical-test-and-diagnosis.md)
