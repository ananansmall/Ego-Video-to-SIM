# Stage2 预处理 MANO 抓取、携带与重放计划（2026-09-10 v1）

> 本文是当前 Stage2 的执行标准，优先级高于 2026-09-09 标准计划中与 Stage1 重新认证、F0-F38 不允许修正、F50 必须以张开状态静止到达旧锚点有关的条款。
>
> 2026-09-01 历史计划保持只读；2026-09-04 文档记录代码修改清单；既有 v1/v2 日志和失败产物全部保留，不覆盖。

---

## 1. 唯一核心目标

给定已经统一预处理的 MANO 夹爪轨迹和动态目标物体 glb_4：

1. 夹爪从预处理轨迹开始连续执行，不瞬移到 F38 或 F50；
2. 抓取前不把目标提前推走；
3. 在 F50 附近复现 Stage1 已证明可用的抓取末态；
4. 抓住目标后沿预处理 MANO 的相对运动移动目标；
5. 保存全部控制点、物理配置和状态，使同一抓取可以确定性重放。

当前验收只要求“抓住、移动、可重放”。非目标 GLB 的夹爪/目标碰撞继续隔离；释放和 full_robot 放到后续阶段。

成功优先级固定为：

1. 目标未在抓取前被推走；
2. 建立双侧抓取；
3. 携带目标产生有效位移；
4. 同一 bundle 可重放；
5. 在前四项通过后，再减小对 MANO 的偏移。

---

## 2. Stage1 冻结，不再重新测试

### 2.1 冻结对象

当前冻结并只读使用：

    output/hawor7a_left_stage1_target_isolated_v2/stage1/best_grasp.npz

已经从 artifact 核对的固定值：

| 字段 | 冻结值 |
|---|---|
| pos | [-0.32075208, -0.50156775, 0.05051774] m |
| q_grasp | 0.0169798885 m |
| q_open | 0.05 m |
| force_limit | 100 N |
| close / settle | 20 / 10 controls |
| recipe revision | stage1_grasp_recipe_v1 |
| scene digest | d86661319f87... |
| artifact SHA256 | 096c1e7b12da96e9a35082ffd74c4b6683cbfd1f300c0bfa1f2c36bada60f35d |

artifact 中 enclose_z=0.04257832 m，与 pos.z 相差约 7.94 mm。Stage2 必须以 pos
为权威，enclose_z 只保留为历史诊断字段，不能覆盖 anchor z。

Stage1 v1/v2 的历史测试和日志继续作为证据保存。本计划不再要求：

- 重跑 Stage1 CEM；
- 每次 Stage2 前运行 --test-stage1；
- 把 --test-stage1-transfer 作为 Stage2 的前置门；
- 因为 Stage2 失败而修改或覆盖 Stage1 artifact。

如果将来主动修改目标碰撞几何、目标质量/摩擦、夹爪 URDF、Stage1 pos/R/q_grasp 或 20+10 recipe，才建立新 Stage1 revision；本轮不做。

### 2.2 Stage2 究竟复用 Stage1 什么

固定复用：

- 目标物体、尺度、质量、摩擦和场景初始状态；
- Stage1 终端抓取 base pos/R；
- Stage1 q_grasp；
- 20 点 q 闭合序列；
- 10 点 q_grasp 稳定序列；
- control_hz、physics_hz、decimation、PD 和 force limit；
- target-isolated 碰撞作用域。

不再要求复现：

- Stage1 用 set_qpos 构造局部试验的瞬时初态；
- 张开夹爪 q=0.05 在旧抓取 anchor 静止且无接触；
- Stage1 局部试验开始前的隐藏 warmup；
- 独立的 Stage1 transfer 认证 rollout。

因此“复刻 Stage1”的定义是复现已成功的抓取终态和抓取控制配方，而不是复现 Stage1 的不可执行初始化。

### 2.3 允许 Stage2 做的适配

Stage2 可以且必须负责把预处理 MANO 接到 Stage1 抓取末态：

- 不加任何 world-Z 抬升，F0-F50 clearance 恒为 0；
- 20 点 Stage1 q 闭合期间允许 base 从锚点上方 0.03 m 平滑下降到 Stage1 anchor；
- 抓稳后直接进入移动验证和 F51-F62 携带（grasp-confirm 保持帧本轮设为 0；仅在
  “抓住但移动即掉”时按 §10 分流逐级恢复 5 / 10）。

这些适配全部属于 Stage2，不能回写 Stage1 recipe，也不能隐藏在 physics substep 中。

---

## 3. 统一坐标和轨迹来源

Stage2 唯一名义轨迹为：

    MANO joints
      -> camera-to-world
      -> world-to-GLB
      -> smoothing
      -> SVD gripper retargeting
      -> one-time pre_translate
      -> p_mano[f], R_mano[f]

Stage2 命令由两部分显式相加，不能都混在一个 pos 字段里：

    p_cmd = p_mano_ref + anchor_correction

其中 anchor_correction 负责把 MANO F50 接到冻结 Stage1 anchor；F0-F50 段**不再叠加
任何 clearance 抬升**。两项都必须逐控制点保存。

抓住后保留预处理 MANO 的相对运动：

    p_carry[f] = p_stage1_grasp
                 + (p_mano[f] - p_mano[50])

    R_carry[f] = R_stage1_grasp
                 * R_mano[50]^T
                 * R_mano[f]

每个控制点都必须保存 p_mano/R_mano、anchor_correction、total_offset、p_cmd/R_cmd
和 source_frame。这样“偏了多少”和“是否仍沿 MANO 运动”可以直接审计。

---

## 4. 不做 clearance 抬升

本轮 P0 **不施加任何 world-Z 抬升**：`p_cmd = p_mano_ref + anchor_correction`，
F0-F50 的手臂高度完全由预处理轨迹和 Stage1 anchor 决定。

理由：

- 预处理 `pre_translate` 已经把 F50 手指中点对齐物体中心，Stage1 anchor 与 MANO
  F50 处于同一坐标系，接近段不存在需要绕行的固定障碍；
- 物体总高约 47 mm，而 v13 的 0.12 m 抬升是物体高度的 2.5 倍，会让夹爪动作完全
  脱离人手轨迹，并额外引入 12 点下降与 12 点放慢；
- 抬升对 F0-F38 的接触问题并未生效——历史记录里 0.12 与 0.08 的 prefix 接触同为 15，
  说明病根不在绝对高度。

历史 v13 记录过 `F0-F38 prefix 目标接触=15`，但该统计尚未与 `contact_objects` /
`contact_links` 口径对齐（见 09-04 §A.1 关于 `total_contacts=56` 的记录），不足以
作为施加 0.12 m 全局抬升的依据。

如果 T2 实测确认 F0-F50 存在**真实**目标接触，按 §10 分流处理；不允许把接触门禁
改成 report-only，也**不允许静默接受**（目标会被提前推走，抓取就失去意义）。

---

## 5. Stage2 新时间线

### 5.1 阶段表

| 区间 | phase | 控制点 | base 轨迹 | q | 目标接触 |
|---|---|---:|---|---|---|
| F0 到 F38 | MANO_FOLLOW | 39 | 预处理 MANO 原样，不叠加任何偏移 | 0.05 | 0 |
| F38 到 F50 | PREGRASP_ALIGN | 12 | 保留 MANO 变化并到达 Stage1 anchor 上方 0.03 | 0.05 | 0 |
| F50 重复 20 点 | STAGE1_CLOSE_INSERT | 20 | 从 +0.03 下降到 Stage1 anchor，同时逐值使用 Stage1 20 点 q 序列 | 0.05 到 q_grasp | 允许目标接触 |
| F50 重复 10 点 | STAGE1_SETTLE | 10 | 固定 Stage1 pos/R | q_grasp | 要求双侧 |
| F50 到 F51，5 点 | CARRY_VERIFY | 5 | 已经开始移动 | q_grasp | 要求保持 |
| F51 到 F62 | CARRY | 11 | Stage1 anchor + MANO 相对运动 | q_grasp | 要求保持 |

Stage2 合计 **97 个控制点**（F0-F62 原始 63 帧），加 Stage3 的 56 点，全程 **153 点
= 5.10 s**，相对原始 113 帧为 **1.35 倍**。

这里不再要求“张开夹爪先到 Stage1 anchor”。夹爪在 anchor 上方保持张开，最后 0.03 m 与闭合同步完成；第 20 个 close-insert 控制点才第一次精确到达 Stage1 pos/R/q_grasp。

### 5.1.1 控制点预算（相对 v14 原设计的精简）

v14 原设计全程 276 点（2.44 倍）。本轮撤销 123 点：

| 撤销项 | 原值 | 现值 | 省 |
|---|---:|---:|---:|
| clearance 全局 world-Z 抬升 | +0.12 m | **无** | 0 |
| `pregrasp_settle_controls` | 5 | **0** | 5 |
| `open_descent_controls` | 45 | **0** | 45 |
| `post_grasp_hold_controls` | 10 | **0** | 10 |
| `approach_time_scale` | 2.0 | **1.0** | 12 |
| `carry_time_scale`（Stage2） | 2.0 | **1.0** | 11 |
| `carry_time_scale`（Stage3） | 2.0 | **1.0** | 28 |
| Stage3 释放段 | 3+5+10 | **1+5+0** | 12 |
| **合计** | | | **123** |

`close_controls=20` 与 `settle_controls=10` 一个不动，**Stage1 无需重跑**。

不抬升之后，原 12 点 OPEN_DESCENT 整段消失：剩下的 0.03 m 插入本来就在 20 点
CLOSE_INSERT 内完成。F50 恢复为 **20+10 = 30 点（1.0 s）**，这是 Stage1 配方的
物理下限（见 §2.2）。

### 5.2 close_insertion_z

F38-F50 的终点停在 Stage1 anchor 上方 0.03 m（`close_insertion_z`），最后 0.03 m
在 20 点 CLOSE_INSERT 内与手指闭合同步下降完成。

它只用于减少手指尚未合拢时在水平方向顶到物体的概率，**不是避碰抬升**，因此远小于
v13 的 0.12 m。F0 是 rollout 起点，不构造任何 ramp；F0-F50 全段 clearance 恒为 0。

当前 SAPIEN 3.0.0b0 CPU Scene 没有直接的 distance/overlap query，只有物理 step 后的
contact。因此不引入独立的风险窗口扫描器作为 P0 前置；接触判定完全依赖 §7.1 的
逐控制点门禁。

### 5.3 F38-F50 对齐

F38-F50 的终点是 Stage1 anchor 上方 close_insertion_z：

    p_pregrasp = p_stage1_grasp + [0, 0, close_insertion_z]    # close_insertion_z = 0.03 m
    R_pregrasp = R_stage1_grasp
    q_pregrasp = [0.05, 0.05]

位置和旋转都使用 quintic/slerp 平滑连接。要求 F38 左连续、F50 到达 p_pregrasp/R_pregrasp，不能用单帧覆盖。

到达 p_pregrasp 后直接进入 STAGE1_CLOSE_INSERT：没有独立的沉降段，也没有独立的
下降段。readiness 检查在 CLOSE_INSERT 首点承担；若 T2 出现 PREGRASP 未 ready，
优先把 `approach_time_scale` 从 1.0 提到 1.5。

### 5.4 F50 闭合与稳定

参数初值：

| 参数 | 初值 | 允许范围 |
|---|---:|---:|
| close_insertion_z | 0.03 m | 0.015 到 0.04 m |
| close_controls | 20 | 固定，来自 Stage1 |
| settle_controls | 10 | 固定，来自 Stage1 |
| post_grasp_hold_controls | **0** | 0、5、10、15、20 |
| carry_verify_controls | 5 | 固定（代码 `CARRY_VERIFY_CONTROLS`） |
| approach_time_scale | **1.0** | >= 1.0（代码下限） |
| carry_time_scale | **1.0** | >= 1.0（代码下限） |

PREGRASP_SETTLE 与 OPEN_DESCENT 均已撤销，这两个 phase 不再出现在 timeline 中。
readiness 由 CLOSE_INSERT 首点的 base tracking/velocity 检查承担。

STAGE1_CLOSE_INSERT：

- source_frame 固定 50；
- q 序列逐值来自 stage1_close_q_sequence；
- z 从 close_insertion_z 平滑减到 0；
- R 平滑到 Stage1 R；
- 最后一点严格等于 Stage1 pos/R/q_grasp；
- 允许从此阶段开始产生目标接触。

STAGE1_SETTLE：

- 原样保留 10 点 Stage1 settle；
- base 固定，q=q_grasp；
- 末 3 点必须保持双侧。

STAGE2_GRASP_CONFIRM：

- 初值 **0**，本轮该 phase 不出现；
- 若出现“抓住但移动即掉”，按 §10 分流把 `post_grasp_hold_controls` 逐级恢复
  0 -> 5 -> 10，恢复后这些控制点必须进入 timeline、manifest 和 bundle hash。

### 5.5 携带

STAGE1_SETTLE 结束后，不能直接跳到 F51。保留当前 5 个 CARRY_VERIFY：

- source_frame 从 50.2 到 51.0；
- base 已经按 MANO 相对增量移动；
- q=q_grasp；
- 至少 4/5 控制点保持双侧；
- 目标沿命令方向产生非零进度。

之后执行 F51-F62。物体目标轨迹由抓取建立时的 T_GO 和夹爪命令共同确定，不把物体 teleport 到参考位置。

---

## 6. 物理作用域

继续使用 stage1_target_isolated：

| 碰撞对 | 当前设置 |
|---|---:|
| gripper 与 glb_4 | 开启 |
| glb_4 与 ground/default environment | 开启 |
| gripper 与 ground/default environment | 开启 |
| gripper 与非目标 GLB | 关闭 |
| glb_4 与非目标 GLB | 关闭 |
| 非目标 GLB 与 ground | 开启 |
| 非目标 GLB 彼此 | 开启 |

以下参数必须与冻结 Stage1 一致：

| 参数 | 值 |
|---|---:|
| physics timestep | 1/240 s |
| control frequency | 30 Hz |
| decimation | 8 |
| gravity | [0,0,-9.81] m/s^2 |
| virtual base K/D | 50000 / 600 |
| finger K/D | 1000 / 200 |
| finger force limit | 100 N |
| friction | 2.0 |
| full-open q | 0.05 m |

当前不允许为了通过而增加 force、改变质量、关闭 gripper-target 碰撞、冻结目标或运行时钳制目标位姿。

---

## 7. Stage2 两级验收

### 7.1 P0 功能通过：先证明抓住、移动、重放

单次必须满足：

- timeline/source_frame 完整、有限、单调不减；
- MANO_FOLLOW、PREGRASP_ALIGN 的目标接触控制点为 0；F0-F50 全段 clearance 恒为 0；
- 允许接触前，目标相对 reset 平移不超过 5 mm、旋转不超过 5 deg；
- STAGE1_SETTLE 最后 10 点中双侧接触不少于 8 点，最后 3 点为 3/3；
- CARRY_VERIFY 至少 4/5 双侧；
- F51-F62 双侧比例不少于 70%；
- target carry progress 不少于 20 mm，且不少于夹爪命令位移的 50%；
- 携带末端 T_GO 平移误差不超过 15 mm、旋转误差不超过 15 deg；
- 无非目标碰撞泄漏、NaN、爆炸速度或 runtime clamp。

独立 reset 三次至少 2/3 通过，形成 P0 Stage2 bundle。

### 7.2 P1 保真优化：P0 以后再做

锁定 P0 的 q、force、physics 和全部控制点数，只做单变量 approach 保真调整：

- `approach_time_scale` 已在下限 1.0，无法继续下探，因此 P1 改为**收紧 F38-F50 的
  anchor_correction 曲线**（缩短校正窗口，让接近段更贴近 MANO）；
- 每轮仍执行 3 次，必须保持 P0 至少 2/3；选择满足 P0 的最小校正量，而不是 reward
  最高但抓取失败的候选。

P1 目标：

- 抓取前 p_cmd 相对 p_mano 的平均、P95、最大误差完整报告；
- 除 F38-F50 校正窗口外，offset 必须为 0；
- 携带段使用 MANO 相对位移，不另行平移轨迹；
- T_GO 收紧到 5 mm / 5 deg。

如果 1.0 基线无法通过，不进入权重 w 优化，也不同时扫描多组参数；先输出第一失败
相位、首个接触 link/shape/source_frame 和目标漂移。

---

## 8. 可重放合同

每个 Stage2 候选都保存 debug bundle；只有 P0 通过者可保存正式 stage2_bundle。

bundle 至少包含：

- frozen Stage1 artifact path、digest、pos/R/q_grasp 和 20+10 recipe；
- preprocess trajectory digest；
- p_mano/R_mano；
- 每控制点 p_cmd/R_cmd/q_cmd/source_frame/phase；
- 连续 unwrap 后的 base_euler_zyx_cmd；rollout/replay 直接消费，禁止各自重新转换 R；
- P0 的 F0-F50 clearance 恒为 0（记录该不变量）；P1 若加校正窗口则保存窗口和曲线；
- close insertion、settle、post-grasp hold、carry verify 的控制点数；
- collision matrix、物理参数、随机种子；
- 初始 robot/object state；
- 接触 trace、T_GO、目标轨迹和全部 gate 结果；
- objective/参数摘要和 bundle hash。

重放时只加载 bundle 中的命令，不重新构造轨迹，不重新优化，也不调用 Stage1 CEM。三次独立 reset 至少 2/3 满足 P0，才称为“实现重放”。

Stage3 当前职责只是在 Stage2 P0 bundle 通过后继续 F62-F90；Stage2 未通过时，Stage3 必须拒绝。释放仍延后。

Stage3 的控制点预算（相对 v14 原设计一并精简）：

| 段 | 控制点 | 说明 |
|---|---:|---|
| CARRY F62-F90 | 28 | `carry_time_scale=1.0`（原 2.0 → 56 点） |
| PRE_RELEASE_SETTLE | 1 | 原 3 |
| RELEASE | 5 | 张开手指，不变 |
| RELEASE_SETTLE | **0** | 原 10，撤销；不再等落稳 |
| RETREAT F91-F112 | 22 | 1:1 |
| **合计** | **56** | 原 96 |

**不等物体落稳**：RELEASE 张开后直接进入 RETREAT，落稳与否由 Stage3 的支撑门
（`_support_ready`）在 RELEASE 之前判定，不用额外的保持帧确认。

---

## 9. 实施顺序

### T0：纯逻辑测试

先修改并运行单测，覆盖：

1. P0 的 F0-F50 clearance 逐值为 0（p_cmd 只含 p_mano_ref + anchor_correction）；
2. F50 终点是 p_stage1 + [0, 0, close_insertion_z]；
3. 控制结构逐段等于 39（MANO_FOLLOW）+ 12（PREGRASP_ALIGN）
   + 20（CLOSE_INSERT）+ 10（SETTLE）+ 5（CARRY_VERIFY）+ 11（CARRY）= 97，且
   不含 PREGRASP_SETTLE / OPEN_DESCENT / STAGE2_GRASP_CONFIRM；
4. close-insert 的 q 序列与 Stage1 20 点逐值一致；
5. close-insert 最后一点与 Stage1 pos/R/q_grasp 逐值一致；
6. CARRY_VERIFY 仍为 5 点且已经移动；
7. 所有参数（含 approach/carry time scale）进入 manifest 和 hash；
8. Stage2 不再要求运行或通过 --test-stage1-transfer。

### T1：无物理 dry-build

显式加载冻结 Stage1 artifact，构建一次完整 timeline，不推进物理。检查：

- Stage1 artifact digest 和 recipe；
- F0-F50 全段 clearance 为 0；
- 39+12+20+10+5+11 = 97 控制结构（见 §5.1.1）；
- manifest/hash 可重复；
- Stage2 输出目录与 Stage1 artifact 目录彻底分离。

### T2：无抬升基线

只运行 Stage2，固定：

    clearance           = 无（F0-F50 不施加任何抬升）
    close_insertion_z   = 0.03
    close_controls      = 20
    settle_controls     = 10
    post_grasp_hold_controls = 0
    carry_verify_controls    = 5
    approach_time_scale = 1.0
    carry_time_scale    = 1.0

执行三次独立 reset。任何一次失败都保存完整 trace；不得覆盖 v1/v2 失败产物。

实现完成后的标准命令应为：

    /home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
      --test-stage2 \
      --views none \
      --side left \
      --hawor-dir /home/an/data/hawor/7a \
      --ras-dir /home/an/data/ras/7_vggt_omega \
      --stage1-artifact output/hawor7a_left_stage1_target_isolated_v2/stage1/best_grasp.npz \
      --output-dir output/hawor7a_left_stage2_preprocessed_v4 \
      --stage2-close-insertion-z 0.03 \
      --stage2-post-grasp-hold-controls 0 \
      --stage2-carry-verify-controls 5 \
      --stage2-approach-time-scale 1.0 \
      --stage2-carry-time-scale 1.0

这里的 --output-dir 只写 Stage2；不得再用它反推 Stage1 输入。

### T3：正式重放

T2 至少 2/3 通过后，固定同一 bundle 重放三次。通过后才把它交给 Stage3 继续携带。

    /home/an/miniconda3/envs/dex/bin/python grasp_hawor.py \
      --replay-stage2-bundle \
      output/hawor7a_left_stage2_preprocessed_v4/stage2/stage2_bundle.npz \
      --views none \
      --side left \
      --hawor-dir /home/an/data/hawor/7a \
      --ras-dir /home/an/data/ras/7_vggt_omega \
      --output-dir output/hawor7a_left_stage2_preprocessed_v4

### T4：最小校正

仅在 T3 通过后按 §7.2 收紧 F38-F50 的 anchor_correction。不得在抓取尚未成功时把精力
转向 w 或全场景避碰。

---

## 10. 失败分流

| 首个失败 | 下一步 | 禁止动作 |
|---|---|---|
| F0-F50 出现真实目标接触 | 输出首个 source/link/shape 与 contact_objects/links；先核对是否 bbox 假碰撞，确认真实后再做 3D waypoint | 加夹持力、加全局抬升 |
| PREGRASP 未达到 readiness | approach time scale 1.0 -> 1.5 -> 2.0 | 同时改闭合参数 |
| CLOSE_INSERT 无双侧 | 只调整 close_insertion_z 或下降/闭合相位关系 | 同时改 force、q 和物理 |
| settle 有双侧但一移动就掉 | 先把 post_grasp_hold 0 增到 5/10，再检查 T_GO | 重跑 Stage1 |
| 抓住但目标不移动 | 检查接触 impulse、滑移、T_GO 和 carry 加速度 | teleport 目标 |
| Stage2 通过但 replay 失败 | 检查 bundle 漏字段、reset/seed/初态漂移 | 重新优化掩盖不可复现 |

每轮只改变一类参数。首个目标是获得第一条可抓取、可携带、可重放的 Stage2 轨迹；
之后才按 §7.2 收紧 F38-F50 的校正量。

---

## 11. 完成定义

本计划的 Stage2 完成必须同时满足：

- Stage1 artifact 未被重跑或覆盖；
- Stage2 从统一预处理 MANO 起点连续运行；
- 抓取允许接触前目标保持稳定；
- F50 局部 wrapper 终点精确复现 Stage1 抓取末态；
- F51-F62 真实移动动态目标；
- 正式 bundle 在独立 reset 下至少 2/3 重放成功；
- F0-F50 clearance 恒为 0、F50 控制点结构（20+10）和所有物理设置完全可审计。

只显示夹爪动画、只通过数组单测、只在局部 Stage1 抬升，或者靠隐藏状态/运行时钳制移动目标，都不算完成。
