# R1 从原片到移动操作的五样本验收

公开版收录文字结果。表中提到的视频、物理 trace 和逐帧 CSV 随原始实验保管。

2026-11-18。五条必测原片是 456 耳机盒、Chairbottle、HOT3D 杯子、HOI4D 纸盒、EgoExplore ep4 小烧瓶。按原片计一次。EgoDex 桌面作为第六条独立对照。固定底盘适用于桌面示范，跨工位示范须完成实际底盘移动。每个数据集适配器可以读取不同格式，下游使用相同的指垫定义、源帧、世界系、求解、物理和判定。

## 目前各条能证明什么

本轮在 simulation-host 用同一份公共回放代码重新执行了耳机盒（2,500 个物理子步）、ep4（5,171 个）和 HOI4D（1,300 个）。独立读取三份新旧 NPZ，源帧、实际指垫、实际物体、双指接触步数、承重授权、实际开爪、独立支撑步数逐元素一致。HOT3D 重新执行固定台架的 150 帧 IK/FK。Chairbottle 复核已有的 592 帧连续物理回放。这一轮测试覆盖五条，不新增 Chairbottle 调参实验。

| 原片与运行身份 | 已有证据 | 按统一条件还缺什么 |
|---|---|---|
| 456 耳机盒，共用 R1 对照，300 帧 | 椅面连续承重 F240–242，F243 实测开爪，F244–299 独立承重 1.868 s。实测对该 pair 的指垫位置/姿态 P95 为 66.72 mm / 38.82°。 | 26 帧机器人碰支撑。另一份有冻结黑线的诊断在进入回放前就有 289.41 mm 的位置 P95 差距。需先验证手到物体、物体到场景的注册以及实际写入 pair 的来源。|
| Chairbottle，完整 F0–591，joint-seat/recovery | 双指夹持，F42 首次通过离台门控。木椅具备实际载荷，但合格承重最长连续 2 帧。 | 门槛为 3 帧，未授权开爪，后续木椅重抓和白椅放稳均未发生。实际站位距可达候选约 176 mm，躯干超限。将木椅到站作为公共放置逻辑的检验案例。|
| HOT3D，clip-002088 花纹杯，150 帧 | 官方双手观测可用于检查手系。固定底盘和躯干的右臂 FK 对目标位置 P95 为 8.87 mm，姿态 9.84°。 | 缺同一杯子自由刚体的受力、抬升、目标支撑和实际开爪记录。先确认原片覆盖的动作及可见支撑，若源片没有完整释放，保留腕轴结果并另选有完整交互的 HOT3D 原片做物理验收。|
| HOI4D 纸盒，ZY20210800001_H1_C5_N15_S57_s03_T1，F31–186 | 公共 R1 实际双指接触，承重后 F178 开爪。实测指垫对 pair 的位置 P95 为 26.87 mm。 | 独立支撑 0.200 s，该区间最大横移 14.17 mm，末帧落点距目标 23.76 mm，并有 1 帧机器人碰支撑。物体中段参考由手轨迹连接深度端点，须保留来源标记。|
| EgoExplore ep4 小烧瓶，file-003，F531–840 | 公共 R1 无延迟版 F549 起升，F827 授权、F828 开爪。独立承重 0.868 s，横移 0.176 mm。 | 有效右手观测 248/310；中段物体轨迹尚缺独立验证。实测对 pair 的位置 P95 为 105.64 mm。另一次冻结 Stage 0 对下发命令的观测帧位置 P95 为 402.37 mm。完整验收还需长尾、跟随与配准。|

上表的黑线到命令、实测到 pair 和 FK 到目标是三种不同误差。数值来自表内指明的运行，跨版本比较须核对源片、pair、场景和时钟。历史测试保留原始判定。按本轮同一入口、真实物体、源轨迹和完整动力学的联合条件，目前没有一条证据齐全。

**456 有可保留的物理正例。** `examples/456483158-mobile-grasp/scene_agile/456-physregress.OIEMlx` 的冻结回归在 F80 起升、F245 开爪，开爪后独立支撑 1.836 s，末帧水平落点误差 0.908 mm。历史和公共包中耳机盒网格的 SHA-256 一致（`3fb7ed863fc096573874af6661cd11b8c42f286b16d723e09ee8f7912bb18878`）。这能排除两版使用了不同网格文件，但不证明源手注册、物体初始姿态、碰撞参数或控制路径一致。冻结这条正例用于回归物理，保留上表公共入口的独立成绩。

## 共用管线和代码入口

```text
原片、时间戳、相机模型
  ├─ RAS mainv3 → 米制场景、重力、地板 → 物体网格和指定承重面碰撞
  └─ EgoFound3R 或数据集原生标注 → 左右手、物体轨迹、观测置信度、接触候选
      ↓ 同源帧注册；在留出的手、物、支撑观测上独立验投影与深度
冻结 Stage 0：同一解剖/工具定义的 R1 指垫目标、手侧、物体 ID、缺测掩码
      ↓ AGILE 只改有原片证据的接触窗；保留优化输出、窗边界和物体系抓姿
新 pair → 连续全身 IK（底盘＋躯干＋双臂按观测启用）→ 限速后复查净空
      ↓ R1/SAPIEN 回读实际位姿、双指接触、物体运动和支撑冲量
整段回放：夹稳再抬 → 持物移动 → 指定支撑承重 → 开爪 → 独立放稳 → 下一接触
      ↓ 同一 trace 生成黑红四视图、机器人第一视角、右腕、逐帧表及 verdict
```

**目前的代码断点。** 在 simulation-host 仓库，`examples/egoexplore/video_to_r1/run_video_to_r1.sh:34` 起已有 EF3R、RAS 和注册调用。第 59–65 行读取外部样本配置，第 72–80 行等待外部提供原始 pair 和 AGILE 窗结果，再调用配置指定的 R1 runner。只提供 MP4 的新样本会在缺少这些输入时退出。第 81–92 行另外要求渲染和报告所用的产物。公共物理入口 `examples/r1_multicontact/replay_multicontact.py:89` 先作物理探测再作接触回放，第 101 行开始读包。其数据合约见同目录 `README.md:9`。两个入口之间应明确谁产出每个文件、数据用哪个坐标系、如何传给下一步。

**最优先修的共性问题。** 在已有 456 和 ep4 的抓取候选窗，按原存储坐标比较，旧审计给出的指尖对物体网格距离统计 P95 分别为 323.21 和 387.13 mm。两只物体自身的长边均不足 70 mm。这个量级值得在调 IK 之前先核对手系、工具中心、物体位姿和源帧。这个结果能定位检查边界，还不能判定具体哪个变换错。验收应保存 Stage 0 → 接触窗修正 → 后处理 → IK FK → 物理读回各级目标，逐级同帧相减。依据为 `examples/r1_multicontact/runs/gripper-hoi-20261118-controls/source_grasp_geometry_check.json`。

**实测轨迹的文件接口也要统一。** 公共回放器在 `examples/r1_multicontact/replay_multicontact.py:611` 写 `multicontact_trace.npz`，现有入口最后调用的 `scripts/unified_trajectory_report.py:106` 读取 `exec_trace.npz`。入口还把 `ik_plan.npz` 单独传入报告。公共回放器的原生产物不能直接满足这些要求。另一个容易误判的问题是入口第 85–86 行合成视频时只传源视频和仿真视频，合成器在 `scripts/render_source_and_sim.py:37` 默认从原片 F0 开始。对 ep4 这类从 F531 开始的回放，要从实际 trace 传入时间映射，并核对首、中、末帧。此前针对具体样本生成的视频各自有独立入口，不能凭这段代码推断每一个旧视频都错位。下一步优先让报告和合成器消费运行器的同一份源帧、实测指垫与物理时钟。

## 三步横向测试

1. **五条一起验映射。** 每条固定源片、对象、支持面、相机、工具轴、源帧和时钟。输出手/物体/支撑在原片上的回投，分别记录实测、补帧、缺测和左右手覆盖率。在原片能确认接触的帧检验两指与网格、物底与承重面的空间关系。配准不可靠的样本仍可跑物理诊断，轨迹保真一栏保留 BLOCKED。EgoDex 可用真实标注对照这个环节。
2. **同一份输入拆开查目标和执行。** 每条冻结 Stage 0，只在有源片证据的接触窗用 AGILE。调用 `examples/egoexplore/video_to_r1/merge_agile_contact_windows.py:39`，核对窗外原始目标和开合逐元素相等。优化或平滑后另存命令，接触时锁物体系抓姿，在同一时间表记录腕、底盘和躯干的实际速度/加速度。按源帧列出首次超差的环节、位置和 SO(3) 姿态 P95、最大值、覆盖率。沿用 `docs/21_retarget_render_acceptance.md:36` 的窗外工程门槛 30 mm / 12°（P95）和 60 mm / 25°（最大值）；有效原片证据少的段单列。
3. **每条一次连续物理验收。** 从接近开始启用重力和碰撞，用实际双指载荷和脱离原支撑验证起升，检查持物相对位姿、实际底盘到位与夹具/家具净空。逐个接触周期查目标支撑连续至少 3 个采样帧承重后才开爪，脱手后独立承重至少 1 s。沿用现有 10 mm 搬运滑移峰值、末 15 帧 5 mm 漂移和 5° 末态目标姿态门槛，记录任何机器人禁碰或躯干越限。源片没有 1 s 尾段时固定末次机器人撤手命令并续跑物理，续跑时间单列。多工位路径上发生的重抓必须在同一个仿真实例内完成。冻结旧版做 A/B，任何共用改动均回归五条。

**输出。** 每条在同一源帧和视角下提供原片＋完整四个仿真视图、独立机身第一视角和独立右腕。本轮正式执行视频统一用黑线表示冻结 Stage 0 指垫中心，红线表示物理实测指垫中心，同一工具偏置、同一源帧。在逐帧 CSV 中另存优化命令。旧图的红线有些表示命令，保留旧图例和来源；运动学对照显式标 FK。补帧断点及原片可见的手均标出。逐帧 CSV 配源帧、物理时间、各级误差、载荷、滑移、关节限位、底盘和放置事件。REPORT 逐条写首个失败环节及支持它的文件。只有上述五个原片各项都通过才宣告这组回归成功。

**复查位置。** 在 simulation-host 仓库根目录，耳机盒和 ep4 的公共对照位于 `examples/r1_multicontact/runs/gripper-hoi-20261118-controls/{456,ep4}`。HOT3D 和 HOI4D 对照位于 `examples/r1_multicontact/runs/humannet-alignment-20261117`，核对 `render/ik_comparison/fk_comparison.json` 和 `hoi4d-clearance-guard/after/physics/contact_replay_audit.json`。Chairbottle 逐步接触和命令见 `examples/r1_multicontact/runs/pixel-umi-ear-20261118/chair_joint_seat/recovery/physics`。交付门槛、日志字段与历史 7 号回归沿用 `docs/21_retarget_render_acceptance.md`。

## EgoDex 桌面对照

simulation-host 的 `/path/to/project_data/dataset/egodex/test/basic_pick_place/` 有 277 对同名 MP4 和 HDF5。第 0 条源数据标注的是坐姿右手将黑色订书机从金属桌移到盒盖上。原片 126 帧、30 Hz，HDF5 中相机和两手各有 126 个 4×4 变换。先用这条检查指垫重定向和相机投影，再从视频独立确定订书机姿态、桌面与盒盖承重碰撞。HDF5 的手和头部位姿不提供物理接触力。可在固定底盘上测试完整抓放，单列桌面结果。[EgoDex 官方数据说明](https://github.com/apple-aiml-research/ml-egodex)

EgoDex 本轮已用官方投影约定生成整段 原片右手回投 MP4 (`egodex-0/egodex_0_right_hand_source_diagnostic.mp4`)，126/126 帧成功解码。投影、置信度和输入哈希 (`egodex-0/projection_audit.json`) 可独立复核。抽看 F36、F60、F84、F96，可以看到订书机依次在手下、离桌、进入盒盖、右手撤离。该片具备真实抓放的可见首尾。右手整体置信度中位数 0.995；部分掌骨点约 0.003，拟合应按关节置信度计权。这个视频只画原始手标注，下一步用独立订书机位姿、盒盖底面/边沿建碰撞后做 R1 动态测试。

## 复现与逐帧证据

耳机盒逐帧接触 (`456/contact_replay_frames.csv`) · ep4 逐帧接触 (`ep4/contact_replay_frames.csv`) · HOI4D 逐帧接触 (`hoi4d/contact_replay_frames.csv`)。

耳机盒、ep4 和 HOI4D 各自冻结输入包后，可从项目根目录按 三条公共物理命令与逐帧审计命令 (`gripper-hoi-RUN_COMMANDS.md`) 回放。ep4 沿用已有基准的 0.5 倍速度，其他两条为原速；各自与同一原片的基准比，跨片不直接比较加速度绝对值。

在 simulation-host 执行下列已有检查，两项均通过。覆盖双窗按源帧写入、窗外逐元素相等、关联位姿一致，以及对窗重叠、物体 ID 错误、观测缺失的拒绝。这是合并层的测试。某条真实视频是否确实运行了 AGILE 并改善接触，还须给出该窗优化器输入、输出、pair 差分和物理 trace。

```bash
cd /path/to/Real2SimGripper
p=python3
"$p" examples/egoexplore/video_to_r1/contact_merge_smoke.py
"$p" examples/egoexplore/video_to_r1/stage2_contact_smoke.py
```

HOT3D 的运动学复现命令如下。原始输入和输出都冻结在同一实验下面，复跑时选新的输出目录。

```bash
cd /path/to/Real2SimGripper
p=python3
h=examples/r1_multicontact/runs/humannet-alignment-20261117
out=$(mktemp -d "$h/five-case-review/hot3d-ik.XXXXXX")
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$p" "$h/render/hot3d_ik.py" \
  "$h/render/hot3d_ik_inputs.npz" /path/to/project_data/dataset/hot3d/train_aria/clip-002088.tar "$out"
```

本次对应 `$h/five-case-review/hot3d-ik/fk_comparison.json`。Chairbottle 复核的 250 Hz 原始记录是 `chair_joint_seat/recovery/physics/exec_trace.npz`，SHA-256 为 `b714362cfa56e8793b0e207b844d61b295ed377781fccead7c07a2ad27faa6f4`。
