# Stage2 / Stage3 实施更改计划（2026-09-10 v14，预处理轨迹抓取重放方向）

## 2026-09-25晚更新：当前实施入口与未完成项

**以下v14正文是9月10日历史记录，不再代表当前代码状态。** 统一执行标准见[最新主计划第8～9节](2026-09-23-物理仿真控制与奖励联合设计-推荐实施方案.md#8-2026-09-25晚更新stage1为什么仍失败先修什么)。9月1日原计划保持只读。本次只改文档，未修改代码、重跑仿真或干预正在运行的实验。

当前Stage1问题主要分两类：

1. **接触/物理观测不可信：** KΔq仍参与双侧判定和降速；历史接触累计被当成当前同时接触；sep单位和单侧/双侧条件混用；目标shape材质/offset读回0/0报错后仍继续运行。先修采样窗口、同子步交集和profile失败阻断，不先调大力或改奖励。
2. **命令/交付链不一致：** 预紧已改成减q，但可能被名义终点限幅抵消；冻结后保持修补分支不可达；闭合和保持使用两套命令状态。t2j CSV在159→160控制帧跳q约1.018mm。启动配置130点闭合/15点稳定，导出仍是`regenerated_nominal`的16/10，尚未完整复用实际Stage1。

已结束的t2j终验0/3、最终提升三次均0cm，Stage2正确拒绝进入；读取时t2k/t2l仍运行，不提前断言最终结果。当前主入口仍CEM，新计划21帧→3初值→L-BFGS-B尚未完整落地。

下一步按主计划8.4实施：**物理profile/观测 → 单一闭合预紧状态机 → 每候选完整实际T/q/t冻结及逐点重放 → 固定候选诊断 → 恢复计划筛选。** 普通Stage2改接近不触发重测已成功Stage1；当前CoACD/物理/动作已变更，则属于新revision，不能借用旧成功身份。

Stage2停顿单列在主计划第9节：修正入口速度参数被归一化抵消、轴向entry速度拼接不连续，取消额外张开空等；保留必要末端减速和真实闭合/稳定动作。先查对应旧bundle，不能把当前未运行的Stage2当作旧停顿复现。Stage0保持不动，不新增全局clearance，不复制Stage1的20cm测试提升到参考搬运。

---

## 以下为9月10日v14历史正文（与上节冲突时以上节及最新主计划为准）

> 本文只回答“当前代码已经改了什么、接下来怎样修改和测试”。
>
> 当前 Stage2 完整执行标准见
> `2026-09-10-stage2-preprocessed-mano-grasp-replay-plan.md` v1；
> `2026-09-01-combined-mano-pretranslate-v51-plan.md` 是只读历史设计，不修改；
> 旧物理实验见 `2026-09-10-stage2-stage3-physical-test-and-diagnosis.md`。
>
> 当前决策：Stage1 成功结果冻结，不再重复测试，也不再把
> `--test-stage1-transfer` 作为 Stage2 前置门。Stage2 从预处理 MANO 连续运行，
> **不施加任何 clearance 抬升**，F50 只保留 Stage1 的 20 点闭合 + 10 点夹持，
> 最终复现 Stage1 抓取末态、移动目标并可重放。
>
> 本轮只更新计划，不修改代码、不重新运行仿真。下方 v13 的严格复核保留为历史诊断，
> 但凡与本节冲突的“重测 Stage1 / transferable 前置门 / F0-F38 不可修正 /
> q=0.05 必须先静止到旧 anchor”要求均已撤销。

---

## 0. v14 当前实施决定

| 项目 | 当前状态 |
|---|---|
| v14 设计与参数 | 已写入计划 |
| Stage1 v2 artifact 路径 | 已确认存在 |
| v14 代码 | 尚未实现 |
| 新 T0 / dry-build / Stage2 v3 / replay | 尚未运行 |
| 2026-09-01 历史计划 | 未修改 |

### 0.1 先冻结不动的部分

以下内容保持现状，不应因为 Stage2 失败而重写：

- Stage0 预处理算法和已通过测试；
- Stage1 v2 artifact：
  `output/hawor7a_left_stage1_target_isolated_v2/stage1/best_grasp.npz`；
- `stage1_recipe.py` 中 20 close + 10 settle 的 q 序列；
- Stage1 的 pos/R/q_grasp、30 Hz 控制、240 Hz 物理、PD、force 和目标物理；
- `stage1_target_isolated` 碰撞矩阵与接触三分类；
- v1/v2 失败产物和日志。

Stage1 的旧局部成功已经回答“这个夹爪末态能否抓住并提升目标”。本轮不再要求它同时
证明“q=0.05 可静止进入相同 anchor”。后者改由 Stage2 的连续过渡解决。

只有主动改变 Stage1 抓取末态、recipe、目标碰撞几何或物理属性时，才创建新 Stage1
revision。普通 Stage2 调轨迹不得触发 Stage1 重跑。

### 0.2 当前失败真正说明什么

当前 Stage2 0/3 需要按 contact 口径复核，而不是用 clearance 掩盖：

| 旧测试 | prefix F0-F38 | adjust F38-F50 |
|---|---:|---:|
| clearance 0.08 | 15 | 24 |
| clearance 0.12 | 15 | 18 |
| min-z 当前实现 | 15 | 24 |

抬升 0.08 与 0.12 的 prefix 接触同为 15，说明病根不在绝对高度；而且该统计尚未与
`contact_objects` / `contact_links` 对齐（§A.1）。因此本轮**不采用全局抬升**，改为：

1. F0-F50 clearance 恒为 0，`p_cmd` 只含预处理 MANO + anchor_correction；
2. q=0.05 不再直接下降到旧 Stage1 anchor，最后 0.03 m 与 20 点闭合同步；
3. 不做独立的沉降段、下降段和抓取确认段；
4. 抓稳后立即开始 CARRY_VERIFY；
5. 若实测出现真实目标接触，先核对是否 bbox 假碰撞，再考虑 3D waypoint，
   不允许退回全局抬升。

### 0.2.1 当前代码与 v14 的精确差距

| 位置 | 当前行为 | v14 要求 |
|---|---|---|
| `reconstruct_trajectory` | 删除并忽略 clearance；F0-F38 prefix 原样 | clearance 恒为 0，只加 anchor_correction |
| `build_stage2_timeline` | 强制 F50 恰好 20+10，base 固定 | 20 close 时允许最后 0.03 m insertion；不增加 descent/settle/confirm |
| `_objective_weights_for_phase` | 只认识旧 phase | 覆盖全部新 phase；继续注明只是 metadata |
| `rollout_follow_trajectory` | 在 close_start 前按旧 phase 统计接触 | 按 expected_contact_policy 统一判定 |
| provisional grasp gate | settle 时 reset drift 必须 <=2 mm | 只在允许接触前限制 5 mm；接触后允许受控居中 |
| T_GO reference | Stage1 settle 结束 | Stage2 settle 结束（confirm 已撤销） |
| `_timeline_hash` | 不含 trajectory_config、MANO nominal、Euler command | 全部纳入 |
| `save_stage2_bundle` | 失败结果完全不保存 | 失败保存 debug，正式 bundle 仍只保存 PASS |
| `--test-stage2` | Stage1 路径依赖 output-dir，缺失时自动 CEM | 显式只读 stage1-artifact，缺失立即失败 |
| Stage2 replay | 没有显式 load/replay CLI | 新增 load、hash 校验和原命令重放 |

### 0.3 需要修改的代码

#### A. `stage2_reconstruct_mixin.py`

当前问题：

- F0-F38 被复制为不可修改 prefix，`safety_offset=0`；
- F50 要求先以 q=0.05 到达 Stage1 anchor；
- timeline 只允许 20 close + 10 settle + 5 moving verify；
- 曾用全局 clearance 抬升绕过提前接触，但抬 0.08 与 0.12 的 prefix 接触同为 15。

目标修改：

1. P0 **不加任何 clearance**：

       clearance_wrapper_offset[f] = [0, 0, 0]
       p_cmd[f] = p_mano_ref[f] + anchor_correction[f]

   不引入风险窗口扫描器，也不构造 ramp。接触判定交给 09-10 §7.1 的逐控制点门禁。
2. F38-F50 平滑到：

       p_pregrasp = stage1_pos + [0, 0, close_insertion_z]   # 0.03 m
       R_pregrasp = stage1_R
       q_pregrasp = [0.05, 0.05]

4. 新 timeline 固定为（与 09-10 v1 §5.1 逐值一致）：

       MANO_FOLLOW                       39 点
       -> PREGRASP_ALIGN                 12 点  (approach_time_scale=1.0)
       -> 20 STAGE1_CLOSE_INSERT         20 点
       -> 10 STAGE1_SETTLE               10 点
       -> 5 CARRY_VERIFY                  5 点
       -> F51-F62 CARRY                  11 点  (carry_time_scale=1.0)
                                        ─────
                                         97 点

5. PREGRASP_SETTLE、OPEN_DESCENT 与 STAGE2_GRASP_CONFIRM 均已撤销（各 0 点）。
   readiness 改由 STAGE1_CLOSE_INSERT 首点检查承担。
6. F0-F50 不做 clearance 抬升；F38-F50 的终点是 Stage1 anchor 上方
   `close_insertion_z=0.03 m`，最后 0.03 m 在 20 点 CLOSE_INSERT 内与闭合同步下降完。
7. STAGE1_CLOSE_INSERT 在 20 点内从上方 0.03 m 平滑下降到 Stage1 anchor，
   q 必须逐值复用 Stage1 的 20 点闭合序列；最后一点严格等于
   Stage1 pos/R/q_grasp。
8. STAGE1_SETTLE 保留原 10 点，不改变 force、PD 或 q。
9. CARRY_VERIFY 保留当前 5 点且必须已经移动；F51-F62 使用预处理
   MANO 相对 F50 的位移/旋转，不生成另一条 carry 轨迹。
10. 所有 phase、offset、控制点数和参数进入 manifest、trace 和 bundle hash。
11. 当前函数在 reconstruct 和 rollout 中各构建一次 bundle。新实现必须收敛成
    单一 trajectory_config 和单次正式 build，避免两次使用不同默认参数。

内部数据不要继续让 `recon["pos"]` 同时表示 MANO、anchor 校正和最终命令。建议
逐控制点明确保存：

    mano_source_pos / mano_source_R
    anchor_correction_pos / anchor_correction_R
    total_offset
    base_pos / base_R

关系必须满足：

    base_pos = mano_source_pos + anchor_correction_pos

F0-F50 的 clearance 恒为 0；CLOSE_INSERT 内 base 从锚点上方 0.03 m 平滑降到锚点。
anchor correction 在 F38-F50 平滑加入，抓取后固定，从而保留 MANO 相对运动。

当前 rollout 每个控制点临时把 R 转成 ZYX Euler。为了避免分支跳变并保证重放使用
完全相同的 virtual-joint 命令，build 阶段应一次性生成连续 unwrap 后的
`base_euler_zyx_cmd`，检查相邻角速度，再写入 bundle/hash；rollout 和 replay
都直接消费这个数组，不再各自重新转换 R。

#### B. `grasp_hawor.py`

`--stage2-approach-clearance-z`、`--stage2-pregrasp-settle-controls` 与
`--stage2-open-descent-controls` 本轮**不再作为输入**（抬升和这两段均已撤销）。
需要保留并新增：

| CLI 参数 | 初值 |
|---|---:|
| `--stage1-artifact` | 冻结 v2 best_grasp.npz 路径 |
| `--stage2-close-insertion-z` | 0.03 |
| `--stage2-post-grasp-hold-controls` | 0 |
| `--stage2-carry-verify-controls` | 5 |
| `--stage2-approach-time-scale` | 1.0 |
| `--stage2-carry-time-scale` | 1.0 |
| `--stage2-dry-build` | 只构建/校验 timeline，不推进物理 |
| `--replay-stage2-bundle` | 显式 Stage2 bundle 路径 |

加载 Stage1 artifact 和 recipe 仍保留，但 Stage2 启动不再要求执行
`--test-stage1-transfer`，也不以旧 `transferable=false` 阻止 rollout。
`--test-stage1-transfer` 可保留为历史诊断命令，不删除旧结果。

当前 Stage1 路径由 `--output-dir/stage1/best_grasp.npz` 隐式推导，文件不存在时还会
自动重跑 CEM；这与“冻结 Stage1”冲突。必须改为：

- `--stage1-artifact` 只读加载冻结 artifact；
- `--output-dir` 只负责写新的 Stage2 产物；
- artifact 缺失立即 `MISSING_FROZEN_STAGE1_ARTIFACT`，禁止自动调用 CEM；
- 第一轮 Stage2 使用新目录
  `output/hawor7a_left_stage2_preprocessed_v4/`；
- replay 只读 `--replay-stage2-bundle`，禁止重新调用 reconstruct。

#### C. `stage1_recipe.py` / `stage1_grasp_mixin.py`

本轮原则上不改 Stage1 行为：

- 20 close + 10 settle 不变；
- Stage1 artifact 不覆盖；
- Stage2 只读 recipe；
- Stage2 的 anchor_correction 与 close_insertion_z 属于 Stage2，不能写回 Stage1 recipe。

若只为兼容 bundle schema 增加只读字段，不能触发重新运行 Stage1。

#### D. `tests/test_stage2_timeline.py`

旧“F50 只有 30 点、prefix offset 恒为 0、close 时 base 固定”的断言需要替换。
新增断言：

1. P0 的 F0-F50 clearance 逐值为 0；
2. F50 预抓取终点为 Stage1 pos+[0,0,close_insertion_z]；
3. 控制点严格等于 39+12+20+10+5+11 = 97，且不含 PREGRASP_SETTLE /
   OPEN_DESCENT / STAGE2_GRASP_CONFIRM；approach/carry time scale 为 1.0；
4. 20 点 q 与 Stage1 recipe 逐值一致；
5. close-insert 末点与 Stage1 pos/R/q_grasp 一致；
6. MANO_FOLLOW 与 PREGRASP_ALIGN 不允许目标接触；
7. `pregrasp_settle_controls` 与 `post_grasp_hold_controls` 为 0 时仍进入 hash
   （证明未使用默认值）；
8. 5 点 verify 已经发生空间移动；
9. Stage2 构建不依赖 transfer 测试结果；
10. Stage1 输入目录与 Stage2 输出目录分离；
11. artifact 缺失时不允许回退 CEM；
12. load 后重算 timeline/manifest/physics hash，不匹配拒绝 replay；
13. 旧 Stage1 artifact 和历史测试文件不被写入。

#### E. `rollout_follow_trajectory` 的门禁位置

当前 gate 仍按旧 phase 名累计接触，并在 Stage1 settle 结束时要求目标相对 reset
位移不超过 2 mm。新 wrapper 中闭合接触可以对目标产生小幅居中运动，因此应改成：

1. 接触是否允许直接读取每个控制点的 `expected_contact_policy`，不要再靠
   `MANO_FOLLOW/APPROACH_ADJUST` 名字硬编码。
2. `MANO_FOLLOW` 和 `PREGRASP_ALIGN` 都是 NO_CONTACT；任一点接触目标立即
   `PREGRASP_TARGET_CONTACT`。
3. CLOSE_INSERT 首点检查 base position/rotation、linear/angular speed、
   q tracking 和目标 reset drift；失败时在闭合开始前退出。
4. 允许接触前目标 drift 使用 P0 的 5 mm / 5 deg。开始 CLOSE_INSERT 后，
   不再使用“reset 到 settle 必须 <=2 mm”作为失败条件；改为记录首个双侧接触后的
   位姿变化、速度和 impulse，防止爆炸，但允许正常居中。
5. `tgo_reference` 在 `STAGE1_SETTLE` 的最后一点建立（confirm 已撤销）；
   carry 的 object_start 和 commanded_start 也使用 settle_end。
6. P0 使用 T_GO 15 mm / 15 deg、F51-F62 双侧比例 70%；P1 再收紧到
   5 mm / 5 deg 和 80%。current tracking mean/max 5/15 mm 继续作为控制器
   tracking 硬门，因为它比较 actual 与 p_cmd，不是 MANO 偏移。
7. `_objective_weights_for_phase` 要覆盖所有新 phase，但这些权重只是 bundle
   元数据；在当前确定性执行器中不能把它们描述成已经参与优化。

#### F. bundle、失败证据和 replay

当前 `save_stage2_bundle` 会拒绝所有失败结果，这是正式产物正确、调试证据不足。
应拆成：

- `save_stage2_debug_bundle`：每个 repeat 都保存，文件名带 repeat 和 failure；
- `save_stage2_bundle`：仍只保存 P0 通过结果；
- `load_stage2_bundle`：allow_pickle=False，重算 timeline、manifest、physics
  三个 hash，并检查控制数组长度/有限性；
- `replay_stage2_bundle`：只执行已加载的 base_pos/base_R/gripper_q/control_dt，
  禁止调用 reconstruct/build/CEM。

正式和 debug bundle 都要新增：

- `mano_source_pos/mano_source_R`；
- `stage2_offset`；
- `trajectory_config_json`；
- `stage1_artifact_path` 和文件 SHA256；
- `control_index`、phase 边界和初始 robot/object state。

`trajectory_config_json` 必须参与 timeline hash。只依靠最终数组变化虽然多数时候也会
改变 hash，但不能证明未使用的默认参数和实际执行参数一致。

### 0.4 参数调整顺序

第一轮只跑**无抬升**基线：

| 参数 | 第一轮 |
|---|---:|
| clearance | 无（F0-F50 恒为 0） |
| close_insertion_z | 0.03 m |
| close_controls | 20，固定 |
| settle_controls | 10，固定 |
| post_grasp_hold_controls | **0**（撤销） |
| carry_verify_controls | 5，固定 |
| approach_time_scale | **1.0** |
| carry_time_scale | **1.0** |

失败时只按首个失败相位调整：

- F0-F50 出现真实目标接触：输出首个 source/link/shape 与 contact_objects/links，
  先核对是否 bbox 假碰撞，确认真实后再做 3D waypoint；**不恢复全局抬升**；
- PREGRASP 未 ready：把 approach_time_scale 从 1.0 增到 1.5 再 2.0；
- CLOSE_INSERT 无双侧：只调整 insertion_z 或下降/闭合相位关系；
- settle 抓住但移动即掉：post_grasp_hold_controls 0 -> 5 -> 10；
- tracking 门不过（time_scale=1.0 太快）：对应 time_scale 回到 1.5 再 2.0；
- 抓住但携带滑移：先检查 T_GO、接触 impulse 和 carry 加速度；
- replay 不一致：检查 bundle/reset/seed，不重新优化。

在无抬升基线获得 2/3 抓取携带和 2/3 bundle 重放以前，不优化 w、不恢复邻物碰撞、
不重跑 Stage1、不扫描 force/PD。

### 0.5 新测试顺序

| 阶段 | 内容 | 通过后才能进入 |
|---|---|---|
| T0 | timeline/recipe/hash/collision 单测 | T1 |
| T1 | Stage2 dry-build，校验冻结 artifact、完整 timeline 和 hash | T2 |
| T2 | 无抬升基线 Stage2，独立 reset 三次 | T3 |
| T3 | 固定同一 Stage2 bundle 重放三次 | T4 |
| T4 | F38-F50 anchor_correction 窗口单变量收紧 | Stage3 |

T2 功能门：

- 允许接触前目标接触=0，reset 漂移不超过 5 mm / 5 deg；
- settle 最后 10 点至少 8 点双侧，最后 3/3 双侧；
- verify 至少 4/5 双侧；
- F51-F62 双侧比例至少 70%；
- target progress 至少 20 mm 且至少为夹爪命令位移 50%；
- 携带末 T_GO 不超过 15 mm / 15 deg；
- 三次至少 2/3 通过。

T3 必须直接加载 T2 bundle，不重建轨迹；三次至少 2/3 才称为实现重放。

### 0.6 完成标准

这轮工具修改完成的标志不是 Stage1 再次 PASS，也不是单测通过，而是：

1. Stage1 artifact 保持只读；
2. Stage2 从预处理 MANO F0 连续运行；
3. 目标在 CLOSE_INSERT 前未被推走；
4. Stage2 在 wrapper 末点精确复现 Stage1 抓取末态；
5. 真实动态目标被携带到 F62；
6. 同一 bundle 可独立重放至少 2/3。

---

## A. v13 历史复核（只保留证据，不再作为当前门禁）

### 1.1 不是“完全没有预处理”

`--test-stage2` 与 `--test-stage3` 都在轨迹重建前调用 `_preprocess_mano_traj(side)`。9 月 10 日已有日志显示：

```text
pre_translate |offset| = 0.1311 m
neutral residual after pre_translate = [0, 0, 0]
```

所以 Stage2 确实拿到了平移后的 MANO 轨迹。旧失败更可能来自：

1. Stage2 以前从 F38 开始验证，相当于绕过 F0–F38 的实际执行历史；
2. 旧 `-gripper-X` retreat 在当前姿态下朝 `glb_7` 方向移动；
3. 夹爪默认与 ground 和所有邻物建立碰撞对；
4. Stage0 与 Stage2 接触统计口径不一致。

Stage0 verify 当前有 `total_contacts=56`，不能继续描述成“其他碰撞严格为 0”。应比较同一 trace 中的 `phase/contact_objects/contact_links`。

clearance 版最近一次物理结果为 `APPROACH_TARGET_CONTACT prefix/adjust=15/24`（clearance 0.12 为 `15/18`）。其中 prefix=15 是预处理轨迹本身与目标的冲突，撤销 z 抬升不会自动消除。新插值必须重新测试；若 prefix 仍接触，先核查真实碰撞体，不能把它误归因于插值曲线。

### 1.2 当前策略

先验证目标抓取和携带，不同时解决邻物避碰。采用 `stage1_target_isolated`：

- Stage2 与 Stage1 使用相同 scene、robot、ground、目标 actor、物体初始状态和物理参数；
- 夹爪与目标 `glb_4`、ground/默认环境的碰撞保持真实；
- 目标与 ground/默认环境的碰撞保持真实；
- 暂时关闭夹爪—非目标、目标—非目标碰撞，避免其他物体干扰本轮“抓住并移动目标”的判断；
- 所有 GLB 仍是动态物体；
- 非目标仍受重力并与 ground/彼此碰撞；
- 该作用域通过后，再分级恢复目标/夹爪与邻物的碰撞，最后进入 full_robot。

这不是最终全场景成功标准，而是用于隔离 G1/G2 根因的物理基线。与 Stage1 相比，唯一允许的物理差异是**隔离非目标物体**；不得借此更改目标质量、尺度、惯量、摩擦、碰撞几何、ground、求解器、PD 或 force。

---

## A.2 v13 当前实现状态与结果复核（历史）

| 项目 | 状态 | 当前代码 |
|---|---|---|
| 项目 | 实现/测试状态 | 结论 |
|---|---|---|
| 共享 20 close + 10 settle recipe | 已实现；T0 逐值通过 | 保留 |
| Stage1 scene/object/physics digest | 已实现；Stage2 reuse report PASS | 只证明当前已纳入字段一致；还缺 anchor pos/R 绑定、碰撞几何/惯量/材料/solver 等完整性 |
| `stage1_target_isolated` 矩阵 | 已实现；T0 与 T1 ground contact 证明分类生效 | 保留 |
| target/ground/non-target 接触三分类 | 已实现 | 保留 |
| Stage1 v1 旧协议局部抬升 | 3/3 | **保留为 `legacy_local_lift_pass` 证据**，不再作为 Stage2 可迁移证据 |
| Stage1 v1 规范入口迁移 | 0/3，drift 627.01 mm | FAIL |
| Stage1 v2 局部抬升 | lift 19.4 cm，3/3 | `legacy_local_lift_pass=true`；入口先扰动物体，不能单独称完整抓取通过 |
| Stage1 v2 入口自检 | 日志中候选在 q_open 下仍接触目标 | `ENTRY_CLEARANCE_FAIL` |
| T1 v2 工具返回值 | 报告为 3/3 PASS | **门禁实现有漏洞，审计结论改为 0/3 FAIL** |
| Stage2 v2 | 0/3，`APPROACH_TARGET_CONTACT 15/24` | 未到达 20+10 或 carry |
| Stage3 v2 | 正确拒绝 | 保留这一行为 |

### 2.1 为什么旧 Stage1 成功仍保留，但不能直接给 Stage2 使用

旧 Stage1 回答的是：“把夹爪直接放到局部抓取位置，并使用 `_DESCEND_OPEN=0.012` 的隐藏初态后，能否夹住并竖直提升？”答案仍然是 3/3，可以保留，不能抹掉。

Stage2 需要回答的是另一个问题：“夹爪从 MANO 轨迹以 `q=0.05` 张开状态连续到达 F50，期间不移动目标，然后是否能执行同一个抓取 primitive？”旧 Stage1 没有验证这个接口。因此重新测试不是否定旧成功，而是做 **Stage1→Stage2 接口资格认证**。只有 Stage1 代码、场景/物理摘要、入口状态或 recipe 发生变化，或者旧 artifact 缺合同字段时才重测；合格 revision 以后应冻结复用，不得每次 Stage2 都重跑 CEM。

本轮 v2 实际修改了 Stage1 的入口（0.012→0.05）和候选评分，所以它本来就不是对旧 Stage1 的简单重复测试，而是新 revision。v1 与 v2 应同时保留：

```text
v1: legacy_local_lift_pass=true, transferable=false
v2: legacy_local_lift_pass=true, entry_clearance_pass=false, transferable=false
```

### 2.2 为什么 T1 v2 的“3/3 PASS”必须撤回

原始 T1 日志显示，每次 reset 后的目标位置为 `[-0.3022,-0.4945,0.0236]`；5 个 warmup 后才做入口接触检查，此时目标已经变为 `[-0.2844,-0.4573,0.0415]`，reset→entry 平移约 **44.9 mm**。同时：

- base 目标约 `[-0.3208,-0.5016,0.0505]`，入口实际值 `[-0.3255,-0.5111,0.0606]`，位置误差约 **14.6 mm**；
- q 命令为 `[0.05,0.05]`，入口实际 q 约 `[0.0750,0.0149]`，最大误差约 **35.1 mm**；
- 报告保存的闭合后 `object_drift=37.13 mm`，且明确为 `object_drift_ok=false`；
- 旋转误差、目标/夹爪速度没有成为 T1 PASS 必要条件。

当前 `run_pass = not entry_target_contact and settle_pass` 只在物体被推开以后检查“现在是否接触”，并忽略 `object_drift_ok`、base/q tracking 和速度。因此 3/3 只是**旧实现门禁返回值**，不是计划定义的入口等价。按本计划应改判 `F50_TRANSFER_FAIL 0/3`。

---

## A.3 v13 代码变更要求（已由 v14 替代）

### 3.1 Stage2 时间线

必须保持以下不变量：

```text
F0-F38       MANO_FOLLOW，复制预处理 pos/R，q=0.05
F38-F50      APPROACH_ADJUST，q=0.05，目标接触=0
F50          20 STAGE1_CLOSE + 10 STAGE1_SETTLE
F50-F51      5 CARRY_VERIFY，已经运动
F51-F62      CARRY，q=q_grasp
```

- `source_frame` 单调不减；
- 只有上述 30 个抓取控制点等于 F50；
- `close_start` 前最后一个 source 必须小于 50；
- `verify_start=50.2`、`verify_end=51.0`；
- F0–F38 的 `safety_offset=0`；
- bundle hash 包含 objective weights。

Stage1 的 `pos/R/gripper_qpos` 是已验证 anchor。Stage2 在 F50 必须复用 Stage1 的同一条 20 点 smoothstep 闭合曲线和 10 点夹持稳定段，并使用相同 control frequency、decimation、PD、force limit 与目标物理参数。不得把 20+10 压缩成 3+2 后仍称为“完全复用 Stage1”。

20+10 在 30 Hz 下占 1.0 s，这是本阶段为了复用已验证抓取配方而接受的局部时间扩张；base pose 在这 1.0 s 内固定，但空间 anchor 不偏离 MANO。只有 Stage1 用更短配方重新验证并生成新 revision 后，Stage2 才能同步缩短，不能单独为“更像 MANO 时间”压缩 Stage2。

Stage1 rollout 中用 `set_qpos` 直接把夹爪初始化到 F50 周围，这是局部可行性测试的初始化，不是可执行轨迹动作，Stage2 不能复制这次瞬移。Stage2 新增的唯一职责是：从预处理 MANO F0 连续到达同一 F50 anchor；到达后再逐控制点复现 Stage1 的 20+10 抓取实验。`enclose_z` 只用于旧 artifact 一致性诊断；缺失时不能默认为 0，更不能覆盖 `pos.z`。

为避免“同名参数、实际数值不同”，Stage1 artifact 必须输出 `stage1_grasp_recipe`，Stage2 只读，不得重新硬编码：

```text
close_controls=20, settle_controls=10
q_open=0.05, q_grasp=<Stage1 result>
close_profile=cubic_smoothstep: t=linspace(0,1,20), s=t²(3-2t)
control_hz=30, physics_hz=240, decimation=8
virtual_base K/D, finger K/D/force_limit
target name/scale/mass/inertia/friction/collision-shape digest
ground material/pose, gravity, solver/contact parameters
object initial pose/velocity digest, scene asset digest
```

Stage2 启动时逐字段比较并生成 `stage1_reuse_report`；任何不在“非目标碰撞隔离白名单”内的差异都以 `STAGE1_REUSE_MISMATCH` fail-fast，禁止开始 rollout。

在正式 Stage2 前必须有严格的 `F50_TRANSFER_EQUIVALENCE`：同一 scene reset 后，构造 Stage2 应达到的规范入口状态——base=`Stage1 pos/R`、实际/命令 q=`0.05`、robot/object 速度为 0、目标未接触——然后调用共享 20+10 recipe。禁止沿用 `_DESCEND_OPEN=0.012` 的隐藏初态。

严格的单次 PASS 必须同时满足，不能只看 settle contact：

```text
pre-step collision overlap = false
AND warmup target-contact controls = 0
AND reset→entry target drift <= 2 mm / 2 deg
AND entry base error <= 2 mm / 2 deg
AND entry gripper-q max error <= 3 mm
AND entry base linear/angular speed <= 0.01 m/s / 5 deg/s
AND entry target linear/angular speed <= 0.02 m/s / 5 deg/s
AND disabled non-target contact pairs = 0
AND settle bilateral >= 8/10 and last 3/3 bilateral
AND reset→settle target displacement <= 2 mm
```

其中任一指标没有采集也不能 PASS。连续 3 次至少 2 次全部满足，才允许 `transferable=true`。当前 v2 的 `object_drift_ok=false` 必须参与 `run_pass`；还需把 base/q/速度门加入判定。若失败，先进入下面的碰撞几何核查，不能直接重复 CEM、改 Stage2 anchor、加力或预闭合。

### 3.2 先做碰撞几何核查，再决定是否重跑 Stage1

当前最早且重复一致的事实是：Stage1 候选在 q_open 入口接触目标，Stage2 prefix/adjust 为 15/24。目标 `glb_4` 当前使用尺寸约 `[53.9,67.1,47.1] mm` 的 dynamic bbox box collision。下一步必须先回答“这是错误的粗碰撞盒，还是 MANO/anchor 的真实穿入”，而不是放宽接触门。

实现一个只读 `STAGE1_CONTACT_GEOMETRY_AUDIT`，至少输出：

- `glb_4` visual mesh、collision shape 的类型、局部/世界 AABB、shape pose、margin/contact offset；
- q=`0.05` 时夹爪 palm、finger1、finger2 的 collision shape AABB/pose；
- MANO F0–F50 与 v2 anchor 的首个 overlap/contact control、link、shape pair、contact point、normal、separation/penetration、impulse；
- reset 后任何 physics step 之前的静态 overlap/distance；若引擎没有无步进 query，用隔离的几何 checker，不能先推动动态目标再判断；
- visual surface 与 collision surface 在这些 contact point 的偏差；
- Stage0、Stage1、Stage2 对同一 pose/q 的分类逐值对照。

决策树固定为：

1. **若 bbox collision 明显包大/错位**：只修 `glb_4` collision representation（优先 convex hull/合适 primitive；复杂时再 VHACD），视觉 mesh、scale 和 pose 不动。碰撞几何变化必须进入 scene digest，生成新的 Stage1 revision 并重跑局部/入口/迁移三门。
2. **若碰撞是真实的**：当前 v2 anchor 不能作为开口入口。Stage1 搜索必须把 q_open overlap、reset→entry drift 做成硬约束，而不是 `-20` 软罚。如果不存在“同一 base pose、q_open 无接触、20+10 可抓”的解，Stage1 应输出一个完整局部 primitive：`pregrasp_pose(q_open)` → 小幅无接触 insertion → `grasp_pose` → 20+10；Stage2 必须逐值复用整个 primitive。
3. **禁止方案**：把 prefix target contact 改为 report-only。目标是 dynamic body，当前日志已经证明提前接触会改变其状态；放宽 gate 会让 F50 面对一个被提前推走的物体，无法证明抓取与移动成功。

### 3.3 F38–F50 空间路径

world-Z 三段抬升走廊已撤销。F38–F50 以预处理 MANO 为主体，只平滑加入 Stage1 F50 的相对校正：

```text
delta_p = p_stage1 - p_mano[F50]
delta_R = R_stage1 · R_mano[F50]^T
alpha[f] = quintic_smoothstep((f-38)/(50-38))

p_cmd[f] = p_mano[f] + alpha[f] · delta_p
R_cmd[f] = Slerp(I, delta_R, alpha[f]) · R_mano[f]
```

这样 F38 严格等于预处理 MANO，F50 严格等于**通过入口硬门的新 Stage1 entry anchor**，中间保留 MANO 的弯曲与相对运动，不额外抬高 z。v2 anchor 未通过入口硬门，暂时不得继续作为该终点。

唯一允许的 z 修正：当 `p_cmd.z < approach_min_z` 时夹到下界。默认 `approach_min_z=0.0 m`，因此只有轨迹进入世界 z<0 时才向上修正；否则 `safety_offset=0`。

`--stage2-pregrasp-retreat` 和 `--stage2-approach-clearance-z` 都只保留 CLI 兼容，当前实现忽略；新参数是 `--stage2-approach-min-z`。

### 3.4 目标函数分相位

统一列为：

```text
[mano_pose, avoid_target_contact,
 maintain_bilateral_contact, maintain_tgo, target_carry_progress]
```

F50 前 `[1,1000,0,0,0]`；闭合 `[0.2,0,100,100,0]`；抓住后 `[1,0,1000,1000,1000]`。

这些权重当前首先是明确的优化接口/阶段合同；不能写成“优化器已经收敛”。

### 3.5 碰撞组

SAPIEN g0/g1 类型分配（名称改为 `STAGE1_TARGET_ISOLATED_CONTACT_MASKS`）：

```text
default environment = bit 0
gripper            = bit 1, affinity=default|target
target             = bit 2, affinity=default|gripper|target
non_target         = bit 3, affinity=default|non_target
```

由此得到的明确矩阵：

| 碰撞对 | 本阶段 |
|---|---:|
| gripper ↔ target | 开启 |
| gripper ↔ ground/default | 开启，复用 Stage1 |
| target ↔ ground/default | 开启，复用 Stage1 |
| gripper ↔ non-target | 暂时关闭 |
| target ↔ non-target | 暂时关闭 |
| non-target ↔ ground/default | 开启 |
| non-target ↔ non-target | 开启 |

现有 g2/g3 夹爪内部自碰过滤不覆盖。每次 rollout 都重新应用同一设置，确保独立重复与 Stage3 replay 一致。ground contact 是本阶段真实物理的一部分，不能再被归为 `COLLISION_FILTER_LEAK`；只有出现 gripper/target 与非目标物体接触才是 filter leak。

若 target/gripper collision shape 数量为 0，应 fail-fast；若 trace 出现非目标接触，应返回 `COLLISION_FILTER_LEAK`。

### 3.6 给实现工具的下一轮最小修改顺序

现有共享 recipe、reuse report、collision matrix、接触三分类和 Stage3 拒绝逻辑先冻结。不得一次同时调路径、力和物理参数。下一轮按以下提交边界实现：

1. **先修 T1 判据，不改物理**：将 `object_drift_ok`、base/rotation/q tracking、robot/object speed、warmup contact 全部并入 `run_pass`；缺指标即 FAIL。增加复现当前 v2 “旧工具 PASS、严格门 FAIL”的回归测试。
2. **把 Stage1 入口从软罚改成硬门**：`entry_target_contact`、pre-step overlap 或 reset→entry drift 超阈值的候选不得参与 elite/best，也不得通过终选验证；若所有候选失败，应明确返回 `NO_TRANSFERABLE_ENTRY`，不能保存 `transferable=true` artifact。
3. **补 artifact 状态分层**：保存 `legacy_local_lift_pass`、`entry_clearance_pass`、`grasp_establish_pass`、`local_carry_pass`、`transfer_equivalence_pass`、`transferable`。新标准下 `transferable` 是 entry、establish、local-carry、transfer 四门的 AND；Stage2 只接受 true。v1/v2 的历史 lift 证据继续保留，但 transferable 标记 false。
4. **完成静态碰撞几何审计**：实现 §3.2 的 shape/link/contact-point 报告，并把 collision type、shape pose/dimensions、collision digest、惯量、材料、solver/contact offset 纳入 scene digest。
5. **绑定 anchor 身份**：recipe 保存规范化 `pos/R/q_grasp` 与 `anchor_hash`；Stage2 从加载值重新计算并比较。当前 fresh `stage1_contract_hash` 不能替代“Stage1 导出时的 anchor hash”。测试必须覆盖 pos、R、q 任一被改后 fail-fast。
6. **按审计单选分支**：假碰撞只改 target collision representation；真实碰撞只改 Stage1 entry/primitive 搜索。一次只选一条，不同时调 `min_z/time_scale/force/q_grasp`。
7. **必要时生成 v3 Stage1 revision**：只有碰撞几何或 Stage1 primitive 改变后才重跑 Stage1；要求 entry clearance、grasp establish、local carry、strict transfer 全部达到各自门禁。
8. **再跑 Stage2**：先确认 prefix/adjust=0/0，再看 PRE_F50_READY、20+10、carry。tracking 16.94/38.39 mm 是第二顺位，不能在提前接触解决前用 time scale 掩盖。
9. **Stage2 通过后才跑 Stage3**；此前不启用 `w`、CEM、释放或邻物避碰。

涉及文件预计包括：`stage1_grasp_mixin.py`、`stage2_reconstruct_mixin.py`、`stage1_recipe.py`、`grasp_hawor.py`、`tests/test_stage2_timeline.py`；若新增 collision representation，还需修改实际创建 `glb_4` collision shape 的场景加载模块，并由 digest 自动使旧 artifact 失效。

---

## A.4 v13 测试顺序与要求（已由 v14 替代）

测试由外部工具执行，必须按顺序，禁止直接先跑 Stage3。

### T0 静态与单元测试

```bash
cd /home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization
/home/an/miniconda3/envs/dex/bin/python -m pytest -q \
  tests/test_stage2_timeline.py tests/test_stage3_timeline.py
```

当前已有 23/23 PASS，证明 20+10、recipe identity、collision matrix、reuse mismatch 和 Stage3 前缀结构正确。下一轮必须增加严格入口门的回归测试；原有 23 个测试继续全部通过，但不能把单元测试等同于物理成功。

### T1A 静态碰撞几何审计

优先新增只读入口（建议 `--test-stage1-contact-audit`），复用 v2 artifact，不运行 CEM、不移动动态目标：

```bash
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
  --test-stage1-contact-audit --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --output-dir output/hawor7a_left_stage1_target_isolated_v2
```

要求产出 §3.2 的 collision/visual 对照，并明确给出 `FALSE_COLLISION_GEOMETRY` 或 `REAL_PATH_OVERLAP`，不能只给截图。

### T1B 条件式生成 Stage1 v3

只有 T1A 已选择并完成一个修复分支时才运行：

```bash
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
  --test-stage1 --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3
```

要求：3 次 legacy local lift 至少 2 次；入口 hard gate 3/3；闭合到 settle 的目标位移 ≤2 mm；Stage1 竖直携带段以 `T_GO`≤5 mm/5°、双侧比例≥0.8 做 local-carry 门。终选 artifact 必须明确 entry/establish/local-carry 均 true。若没有候选满足入口，不得用最高 reward 失败候选兜底。

### T1C 严格 F50 迁移等价验证

使用 v3 的规范入口和共享 20+10：

```bash
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
  --test-stage1-transfer --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3
```

- 禁止 `_DESCEND_OPEN=0.012` 隐藏初态；
- 必须执行 §3.1 的全部入口状态门，包括 reset→entry drift、base/q error 和速度；
- 20+10 命令与 Stage1 artifact 完全一致，settle 达门；
- 三次独立 reset 至少两次**同时**通过入口与抓取；
- 未通过时不得继续 Stage2。

### T2 Stage2 Stage1-target-isolated 基线

```bash
export _OBJ_SCALE=0.6
export _OBJ_SCALE_TARGET=glb_4
export _GRIP_FRICTION=2.0
export _SMOOTH_MINT=0

/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
  --test-stage2 --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --stage2-approach-min-z 0.0 \
  --stage2-approach-time-scale 2.0 \
  --stage2-carry-time-scale 2.0 \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3
```

必须检查：

- pre_translate 后 neutral residual ≤1 mm；
- artifact 明确 `transferable=true`，T1C ≥2/3；
- `stage1_reuse_report` 除“非目标碰撞隔离”外零差异；
- manifest 的碰撞作用域与 shape 数量正确，gripper-ground/target-ground 为 true；
- F0–F38 和 F38–F50 的目标接触分别为 0；
- F50 的 20 close + 10 settle 命令与 Stage1 逐值一致，settle 末段稳定双侧；
- moving verify ≥4/5 双侧；
- `T_GO` 平移/旋转 ≤5 mm/5°；
- tracking mean/max ≤5/15 mm；
- 至少 2/3 PASS 后才生成正式 bundle。

### T3 Stage3

Stage2 通过后，使用相同输入与 Stage2 参数：

```bash
/home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
  --test-stage3 --views none --side left \
  --hawor-dir /home/an/data/hawor/7a \
  --ras-dir /home/an/data/ras/7_vggt_omega \
  --start-frame 0 --num-frames 113 \
  --grasp-pose-method svd_palm \
  --stage2-approach-min-z 0.0 \
  --stage2-approach-time-scale 2.0 \
  --stage2-carry-time-scale 2.0 \
  --stage3-carry-time-scale 2.0 \
  --stage3-release-candidate 90 \
  --output-dir output/hawor7a_left_stage1_target_isolated_v3
```

要求：Stage3 的 Stage2 replay hash/物理门通过；F62–F90 保持抓取。无支撑时允许 Stage3A PASS、Stage3B `NO_SUPPORT`，不得悬空释放。

---

## A.5 v13 失败调整（已由 v14 替代）

| 首个 failure | 处理方向 | 禁止操作 |
|---|---|---|
| `STAGE1_REUSE_MISMATCH` | 比较 Stage1/2 manifest，恢复同一场景与配方 | 不进入物理 rollout |
| `ENTRY_CLEARANCE_FAIL` / `F50_TRANSFER_FAIL` | 先做静态 collision audit；假碰撞修 shape，真碰撞重做 entry/primitive | 不用 warmup 把目标推开后再检查 |
| `NO_TRANSFERABLE_ENTRY` | 明确报告当前姿态/几何下无可迁移候选 | 不保存“最佳失败”供 Stage2 使用 |
| `COLLISION_FILTER_LEAK` | 检查非目标 shape 分类与 g0/g1 | 不把真实 ground contact 当泄漏 |
| `INITIAL_TARGET_CONTACT` | 检查首帧 pose、warmup、目标 reset | 不跳过 F0–F38 |
| prefix target contact | 用 contact geometry audit 定位首个 link/shape；确认真实后做最小 MANO deviation 重规划 | 不改成 report-only，不让目标提前移动 |
| adjust target contact | 先确认终点 entry anchor 本身无 overlap，再优化 correction 曲线 | 不增加任意 z 抬升、不沿用 v2 失败 anchor |
| `PRE_F50_READY` | approach time scale 2/3/4 | 不增加 F50 静止帧 |
| `PROVISIONAL_GRASP` | 回到 Stage1 anchor 与同一 20+10 recipe；先做 F50 transfer-equivalence test | 不提高隐藏 force/K/D、不缩短闭合 |
| `CARRY_VERIFY` | carry time scale 2/3/4，查双侧和 T_GO | 不恢复旧 w |
| `CARRY` | 查 q 保持、目标进度、T_GO | 不让 correction fade |
| `TRACK` | 查 virtual drive 和空间步长 | 不改物体质量掩盖 |

每轮只改一个变量，使用新输出目录，并保留 bundle/physics hash。

---

## A.6 v13 后续工作（历史）

按以下顺序恢复真实环境约束：

1. 保持 ground 碰撞始终开启，先恢复目标—最近邻物；
2. 再恢复夹爪—最近邻物，必要时只重新规划 F38–F50；
3. 逐个恢复其余非目标物，最后恢复全部邻物；
4. 将同一 bundle 接入 full_robot IK 与关节限位；
5. 全部成功后才把非目标隔离作用域从标准基线中移除。

环境恢复阶段只能修改 approach corridor 或使用正式 motion planner。Stage1 anchor、F50 20+10、F50 后 q 与 target carry 合同必须保持冻结，这样才能知道失败来自环境可达性而不是抓取配方漂移。

---

## A.7 v13 完成定义（已由 v14 替代）

本实施计划只有在以下条件同时成立时才算 Stage2 完成：

```text
代码单元测试通过
AND `MANO_FOLLOW/APPROACH_ADJUST/FINAL_APPROACH` 的目标接触为 0
AND Stage1/Stage2 物理与物体配置除非目标隔离外逐字段一致
AND Stage1 artifact 明确 entry_clearance/grasp_establish/local_carry/transferable 全部通过
AND 严格 F50_TRANSFER_EQUIVALENCE 在 q=0.05 规范入口至少 2/3 PASS
AND F50 20+10 完整复现 Stage1 并建立稳定双侧抓取
AND F50-F62 边走边抓通过 T_GO/进度/跟踪门
AND 三次独立 reset 至少两次通过
AND 正式 bundle 可保存并由 Stage3 原样重放
```

当前尚未获得这些物理结果。准确状态是：**共享基础设施已完成；Stage1 v1/v2 的局部抓取证据保留，但没有可供 Stage2 消费的 transferable revision；Stage2 仍为 0/3**。下一步不是继续调 Stage2，而是先完成静态碰撞几何审计并修正严格入口门。
