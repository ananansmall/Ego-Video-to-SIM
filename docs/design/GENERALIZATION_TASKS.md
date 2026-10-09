# 手物交互管线任务泛化方案

> 状态：调研 + 方案草案，待决策
> 日期：2026-10-08
> 范围：ReplicateAnyScene (RAS) `object_tracking/` 管线，HoI4D / EgoDex 数据集

---

## 0. 核心结论

**数据侧已经泛化，代码侧没有。**

- 本地 `HOI4D_selected_200/manifest.json` 的 200 条 clip，`source_path` 中已含 `/T1/`–`/T6/` 任务 ID，覆盖 6 个不同 task
- `egodex_test.zip` 含 **107 个任务子目录**，本地仅解压 `basic_pick_place` 一个
- 但管线把一切都当 pick-and-place 处理，且存在若干会主动破坏轨迹的硬编码假设

因此泛化的第一步不是加模型，而是**把隐式的 pick-place 假设显式化并拆掉**。

---

## 1. 现状诊断：pick-and-place 硬编码的 4 个层次

pick-and-place 不是以显式状态机存在的，而是以「抓取/释放二值信号 + 自由刚体 + 单一夹爪开闭」的形式隐式分布在 4 层。

### L1 语义层 — 二值抓取判据

文件：`object_tracking/03_grasp_controller.py`

| 位置 | 问题 |
|---|---|
| L59-L106 `compute_motion_coupling()` | 耦合度只看**速度方向一致性**。推物体时手与物体同向移动 → `cos_sim ≈ 1` → 判为高耦合 |
| L90 | `moving = (obj_s > 0.005) & (hand_s > 0.005)`，**完全静止的物体耦合度强制归零**。抽屉已拉开、灯已打开这类"状态改变但无平移"永远检测不到 |
| L184-L197 | `grasp_score` = coupling(0.4) + stability(0.3) + distance(0.3)，阈值 0.7。三项全是抓取语义标定 |
| L150 | `distance_threshold=0.15`，超过 15cm 的交互（handover、抛掷、推远处物体）检测不到 |
| L210-L224 | `type` 只有 `grasp_and_release` / `held` 两个正常字面量 |
| L236-L238 | 多段全部退化为 `repeated_manipulation` |
| L243-L294 `compute_grasp_pose()` | 单一 6DoF 构型（指尖中心 + 掌心法线），无 grasp type 枚举。`hand_label` 参数在函数体内**完全未被使用**（L243 签名有，L259-L294 未引用） |

### L2 规划层 — 释放后静止约束（最严重）

文件：`object_tracking/04_trajectory_refiner.py` L258-L284

```python
def apply_release_constraint(poses, release_frame, total_frames):
    """释放后物体保持静止"""
    for t in range(release_frame + 1, total_frames):
        poses[t]["R"] = release_pose["R"]
        poses[t]["t"] = release_pose["t"]
```

对以下任务**主动破坏**轨迹：
- 开抽屉 / 开门（释放后 drawer 仍在运动）
- 推倒多米诺（释放后继续滑行）
- 倾倒（释放后瓶内液体继续流动）

模块 docstring L12 明确写「释放后静止约束 → 物理一致性」——这是把 pick-place 的偶然特征当成了物理规律。

### L3 物理层 — 只有 static / dynamic 二分

文件：`object_tracking/simulation/scene_builder.py`

| 位置 | 问题 |
|---|---|
| L203-L223 `_create_mesh_object()` | `if is_static: "static" else: "dynamic"`，**无关节模型** |
| L152-L167 `_create_dynamic_object()` | `is_static = obj_desc.get("is_static", False)`，布尔二分 |
| L130-L146 `_load_static_scene()` | 静态场景整体作为单个 static body |

EmbodiedGen 侧同样如此，`embodied_gen/models/layout.py` L76 明确写：
> `{MANIPULATED_OBJS}` and `{DISTRACTOR_OBJS}` are rigid bodies and not include flexible objects.

**后果**：HoI4D 中 C3 Laptop（开合屏幕）、C4 StorageFurniture（抽屉/门）、C6 Safe（门）、C17 Lamp（折叠灯臂）、C18 Stapler 的任务**必须**有关节模型才能仿真。

### L4 执行层 — action 空间过窄

文件：`object_tracking/simulation/action_player.py`

| 位置 | 问题 |
|---|---|
| L29 | `ACTION_DIM = 16` = 双 EE 位姿(7+7) + 双夹爪标量(2)。无腕部旋转模态、无手指关节 |
| L98-L121 `gripper_timeline_to_signal()` | 接收 schema `{"frame", "state", "object"}` |
| `03_grasp_controller.py` L313-L322 | 产出 schema `{"frame", "action", "hand"}` |

**`action` vs `state` 字段名断裂**，两模块对接处会 `KeyError` 或静默失效。

---

## 2. 现有数据已覆盖的任务

### 2.1 HoI4D 任务定义（26 个动词）

来源：`hoi4d/HOI4D-Instructions/definitions/task/task_definitions.csv`

| 类别 | T1 | T2 | T3 | T4 | T5 | T6 |
|---|---|---|---|---|---|---|
| C1 Toy Car | Pick and place | **Push toy car** | Put it in the drawer | Take it out of the drawer | - | - |
| C2 Mug | Pick and place | Put it in the drawer | Take it out of the drawer | **Fill with water by a kettle** | **Pour water into another mug** | Pick and place (with water) |
| C3 Laptop | Pick and place | **Open and close the display** | - | - | - | - |
| C4 Storage Furniture | **Open and close the drawer** | **Open and close the door** | Put the drink in the door | Put the drink in the drawer | - | - |
| C5 Bottle | Pick and place | **Pour all the water into a mug** | Put it in the drawer | Take it out of the drawer | **Reposition the bottle** | Pick and place (with water) |
| C6 Safe | **Open and close the door** | **Put something in it** | **Take something out of it** | - | - | - |
| C7 Bowl | Pick and place | Put it in the drawer | Take it out of the drawer | **Take the ball out of the bowl** | **Put the ball in the bowl** | Pick and place (with ball) |
| C8 Bucket | Pick and place | **Pour water into another bucket** | - | - | - | - |
| C9 Scissors | Pick and place | **Cut something** | - | - | - | - |
| C11 Pliers | Pick and place | Take it out of the drawer | Put it in the drawer | **Clamp something** | - | - |
| C12 Kettle | Pick and place | **Pour water into a mug** | - | - | - | - |
| C13 Knife | Pick and place | Take it out of the drawer | Put it in the drawer | **Cut apple** | - | - |
| C14 Trash Can | **Open and close** | **Throw something in it** | - | - | - | - |
| C17 Lamp | Pick and place | **Turn and fold** | **Turn on and turn off** | - | - | - |
| C18 Stapler | Pick and place | **Bind the paper** | - | - | - | - |
| C20 Chair | Pick and place to the original position | (同上) | Pick and place to a new position | (同上) | - | - |

### 2.2 本地 HOI4D_selected_200 的 task 分布

从 `manifest.json` 的 `source_path` 解析 `/T(\d+)/` 统计：

| 类别 | 出现的 task ID | clip 数 |
|---|---|---|
| C5_CellPhone | T1–T6 | 20 |
| C2_Mug | T1–T6 | 20 |
| C7_Plate | T1–T6 | 20 |
| C4_Box | T1–T4 | 20 |
| C1_Bowl | T1–T4 | 20 |
| C6_Pan | T1–T3 | 19 |
| C9_Cup | T1–T2 | 20 |
| C8_TrashCan | T1–T2 | 20 |
| C3_Bottle | T1–T2 | 20 |
| C11_Scissors / C12_Laptop / C13_Pliers / C14_Stapler / C17_Kettle / C18_Knife / C20_Paper | 部分 | 3 each |

### 2.3 EgoDex 任务清单（107 个子目录）

`egodex_test.zip` 的顶层目录，按交互类型归类：

| 类型 | EgoDex 任务目录 |
|---|---|
| 抓取搬运 | `basic_pick_place` `vertical_pick_place` `pick_place_food` `stack` `stack_unstack_bowls` `stack_unstack_cups` `stack_unstack_plates` `stack_unstack_tupperware` `stack_remove_jenga` `insert_dump_blocks` `build_unstack_lego` |
| 推拉滑移 | `push_pop_toy` `roll_ball` `gather_roll_dice` `arrange_topple_dominoes` `rake_smooth_zen_garden` `sweep_dustpan` `deal_gather_cards` |
| 铰链开合 | `add_remove_lid` `insert_remove_drawer` `insert_remove_bookshelf` `assemble_disassemble_furniture_bench_drawer` `insert_remove_furniture_bench_cabinet` `insert_remove_furniture_bench_round_table` `assemble_disassemble_furniture_bench_desk` `assemble_disassemble_furniture_bench_chair` `assemble_disassemble_furniture_bench_lamp` `assemble_disassemble_furniture_bench_square_table` `assemble_disassemble_furniture_bench_stool` `open_close_insert_remove_box` `open_close_insert_remove_case` `open_close_insert_remove_tupperware` `insert_remove_plug_socket` `insert_remove_usb` |
| 旋转 | `lock_unlock_key` `screw_unscrew_allen_fixture` `screw_unscrew_bottle_cap` `screw_unscrew_fingers_fixture` `charge_uncharge_airpods` `charge_uncharge_device` `slot_batteries` `turn_...` `use_rubiks_cube` `fidget_magnetic_spinner_rings` |
| 倾倒浇注 | `pour` `load_dispense_ice` `scoop_dump_ice` `boil_serve_egg` `make_sandwich` |
| 工具使用 | `use_chopsticks` `play_piano` `play_mancala` `play_reset_connect_four` `set_up_clean_up_chessboard` `write` `type_keyboard` `point_and_click_remote` |
| 柔性物 | `crumple_flatten_paper` `fold_stack_unstack_unfold_cloths` `basic_fold` `fold_unfold_paper_basic` `fold_unfold_paper_origami` `wrap` `wrap_unwrap_food` `zip_unzip_bag` `zip_unzip_case` `tie_and_untie_shoelace` `tie_and_untie_rubberband` `braid_unbraid` `knead_slime` |
| 双手协同 | `assemble_disassemble_legos` `assemble_disassemble_soft_legos` `assemble_disassemble_structures` `assemble_disassemble_tiles` `assemble_jenga` `thread_unthread_bead_necklace` `insert_remove_shirt_in_tube` |
| 精细操作 | `use_rubiks_cube` `peel_place_sticker` `zip_unzip_bag` `slot_batteries` `clip_unclip_papers` `turn_fidget...` `measure_objects` `sort_beads` |
| 其他 | `wipe_screen` `wipe_kitchen_surfaces` `clean_surface` `clean_cups` `clean_tableware` `wash_fruit` `wash_kitchen_dishes` `dry_hands` `declutter_desk` `flip_pages` `flip_coin` `color` `throw_and_catch_ball` `throw_collect_objects` |

> `docs/research/REFERENCED_PAPERS_DETAIL.md` L68 引自 EgoSim 论文，注明 EgoDex **全部是 tabletop 场景**，L125 Limitations 明确「只支持桌面场景」。

---

## 3. 数据 bug：类别映射完全错位

`/mnt/data_8THDD/lza/dataset/HOI4D_RGB/select_200.py` L23-L28 的 `CATEGORY_NAMES` 与 HoI4D 官方 mapping（`hoi4d/HOI4D-Instructions/README.md` L35-L39）**完全错位**：

| ID | select_200.py 写的 | 官方实际 |
|---|---|---|
| C1 | Bowl | **ToyCar** |
| C2 | Mug | Mug ✓ |
| C3 | Bottle | **Laptop** |
| C4 | Box | **StorageFurniture** |
| C5 | CellPhone | Bottle |
| C6 | Pan | **Safe** |
| C7 | Plate | **Bowl** |
| C8 | TrashCan | **Bucket** |
| C9 | Cup | **Scissors** |
| C11 | Scissors | **Pliers** |
| C12 | Laptop | **Kettle** |
| C13 | Pliers | **Knife** |
| C14 | Stapler | **TrashCan** |
| C17 | Kettle | **Lamp** |
| C18 | Knife | **Stapler** |
| C20 | Paper | **Chair** |

### 连带后果

`select_200.py` L29-L30 依据错误名字划分：

```python
RIGID_CATS = ["C1","C2","C3","C4","C5","C6","C7","C8","C9"]
ARTICULATED_CATS = ["C11","C12","C13","C14","C17","C18","C20"]
```

按官方 mapping，真正**带关节**的类别是：
- C3 Laptop（屏幕开合）
- C4 StorageFurniture（抽屉/门）
- C6 Safe（门）
- C8 Bucket（桶把手）
- C9 Scissors（两片开合）
- C14 TrashCan（盖）
- C17 Lamp（灯臂折叠）
- C20 Chair（靠背/座面）

其中 **C3、C4、C6 被当成 rigid 各采了 20 条**。C4_Box 的 T1–T4 恰好就是「开抽屉 / 开门 / 放饮料进柜 / 取出」——数据极好但被标成刚体。

**建议：先修此映射，再谈泛化。** 否则按类别筛选数据会筛反，且 joint 标注取不到。

---

## 4. 缺失的标注数据

### 4.1 本地现状

```
/mnt/data_8THDD/lza/dataset/HOI4D_RGB/
├── HOI4D_release/          # 仅 align_rgb/image.mp4
├── HOI4D_annotations.zip   # 不存在
├── download_annotations.sh # 脚本存在，OneDrive 分享链接已硬编码
└── HOI4D_selected_200/     # 200 clips + manifest.json（含 T ID）
```

HoI4D 数据结构（`README.md` L14-L30）：
```
./ZY2021080000*/H*/C*/N*/S*/s*/T*/
|--align_rgb/image.mp4
|--align_depth/depth_video.avi
|--objpose/*.json            # 物体位姿
|--action/color.json         # 动作分段标注 ← 缺失
|--3Dseg/{raw_pc.pcd,label.pcd,output.log}
|--2Dseg/mask
```

### 4.2 关节标注格式（泛化 L3 的关键）

`README.md` L60 明确：

> For each articulated object, we provide the articulated part meshes as well as joint annotations. […] joint annotations `{category}/{object id}/mobility_v2.json` including the **origin, direction, and rotation (for revolute joint) or translation (for prismatic joint) limit for each joint axis**. In addition […] the canonical frame of each articulated part `{category}/{object id}/{objs}/{part name}_align.obj` that is used to define the part pose.

**关节轴 + 限位 + part canonical frame 都现成**，接入 scene_builder 的成本远低于从零估计。

### 4.3 动作分段标注格式（泛化 L1 的关键）

`README.md` L62-L64：
> We present Segmentation to record **per-frame action class**. `duration` denotes the length of video, `event` denotes the label of the clip, `startTime` and `endTime` denotes the range of the clip.

- 文件：`action/color.json`
- 字段：`duration` / `event` / `startTime` / `endTime`
- **`event` 即动作片段标签字符串**，是 clip 级区间标注（不是逐帧），需展开成 per-frame 序列

这是现成的「视频帧 → 动作阶段」ground truth，可直接替代 L1 的手写规则。

### 4.4 运动分割标注（含左右手）

`definitions/motion segmentation/label.csv`，字段 `Category ID, Category, Lable 1..Lable 6`（原文拼写 "Lable"）：

| 类别 | L1 | L2 | L3 | L4 | L5 | L6 |
|---|---|---|---|---|---|---|
| C4 StorageFurniture | Door | **Right Hand** | Body of the Cabinet | **Drawer** | Bottle | |
| C8 Bucket | Body of the Bucket | **Right Hand** | Handle | Body of the Bucket | Handle | **Left Hand** |
| C17 Lamp | Base | **Right Hand** | Tube | Top Link | Bottom Link | **Left Hand** |
| C20 Chair | Chair | **Right Hand** | **Left Hand** | | | |

**唯一标注左右手的表**，对双手任务直接可用。

---

## 5. 三条泛化路线

### 路线 A：接触类型分类器替换 grasp_score（低成本，先做）

**不依赖任何新标注，纯运动学。**

把 L184-L197 的二值判据换成接触模式分类：

| 判据 | 判为 | 需要的物理模型 |
|---|---|---|
| 耦合 + 支撑关系不变（物体始终在桌面/地面） | `push` | 自由刚体 + 摩擦 |
| 耦合 + 手-物距离持续增大 | `pull` | 自由刚体 + 摩擦 |
| 角速度耦合、力矩残差呈螺旋 | `rotate` | 自由刚体 + 力矩 |
| 耦合 + 支撑关系改变（离开支撑面） | `lift` / `grasp` | 自由刚体 + 夹爪 |
| 支撑面拓扑变化（抽屉滑出） | `insert` | **铰链**（依赖路线 B） |

同时：
- 删除 `apply_release_constraint()`，改为「释放后按物理自由演化」
- 放宽 `distance_threshold`（handover / 抛掷场景）
- 修 L90 `moving` 判据，让静止物体的状态改变可被检测

**产出**：可跑通 pick / place / push / pull / rotate / lift，不动物理引擎。

### 路线 B：scene_builder 加关节（中成本，必做）

`static/dynamic` 二分 → `fixed / revolute / prismatic / free`。

- 输入：HoI4D `mobility_v2.json`（origin / direction / limit）+ `{part}_align.obj`
- 打开抽屉这类任务在 L1 分类出 `insert` 后，物理层才有对应能力

**前置依赖**：必须先落地 annotations 包。

### 路线 C：VLM 打标替代手写规则（较高成本，最通用）

`EmbodiedGen/embodied_gen/scripts/affordance_annot/partsemantics_annot.py` L99-L119 已有验证过的 prompt 模板，输出：
- `graspable` (bool)
- `grasp_scenarios`（带 confidence）
- `functional_labels` — 已含 `support` / `contain` / `cut` / `press` / `cover` / `connect` / `stabilize` / `indicate orientation`
- `semantic_description`

**优势**：能覆盖 EgoDex 107 个任务而无需逐个写规则。
**代价**：需要标注数据做验证集（HoI4D `action/color.json` 可充当 ground truth）。

`docs/research/EGO_VIDEO_TO_SIM_ROADMAP.md` L159-L184 已定义目标 schema（`task_semantics` + `primitive_sequence`），L699-L724 有 `segment_primitives()` 伪代码草案，但 **`segment_primitives.py` / `task_semantics.py` 均不存在**。

---

## 6. 必须先修的隐性 bug

### Bug 1：左手夹爪恒为 OPEN（双手任务全废）

`03_grasp_controller.py` L313-L322，`generate_gripper_timeline()` 事件 `hand` 硬编码 `"right"`：

```python
timeline.append({"frame": close_frame, "action": "close", "hand": "right"})
timeline.append({"frame": seg["release_frame"] + 1, "action": "open", "hand": "right"})
```

而 L344-L346 的消费端按 hand 过滤：

```python
for event in timeline:
    if event["hand"] != hand:
        continue
```

L478-L483 对左右手各调一次，因 timeline 里全为 `"right"`，`gripper_signal_left` **永远全 OPEN**。

影响：HoI4D C8 Bucket（双手把 Pour）、C17 Lamp、C20 Chair，以及 EgoDex 全部双手任务在此处失效。

L470-L476 的 `active_hand` 兜底逻辑本身是合理的（选有 segment 的那只手），但结果只喂给一只手。

### Bug 2：schema 字段名断裂

| 模块 | 字段 |
|---|---|
| `03_grasp_controller.py` L313-L322 产出 | `{"frame", "action", "hand"}` |
| `action_player.py` L102 消费 | `{"frame", "state", "object"}` |

`action` vs `state` 不一致。

### Bug 3：`compute_grasp_pose` 的 `hand_label` 参数未使用

L243 签名含 `hand_label`，L259-L294 函数体内未引用。双手场景下左右手会用同一套构型逻辑（当前因为硬编码 right 尚未暴露）。

### Bug 4：仿真环节未真正接入

`run_pipeline.py` L499-L505：

```python
sim_result = run_simulation(
    env=None,
    robot=None,
    ...
)
```

`env` / `robot` 均为 None，`run_simulation` L168 的 `env.step(action)` 必然 `AttributeError`。

### Bug 5：两份副本会分叉

`Ego-Video-to-SIM/ReplicateAnyScene/object_tracking/` 是 `ReplicateAnyScene/object_tracking/` 的完整副本（7 个文件同名同结构）。修改需两处同步，否则立刻分叉。

### Bug 6：目录/字段命名以 grasp 为中心

`run_pipeline.py` L410-L420 输出目录硬编码 `grasp`，字段名 `gripper_signal_*`。泛化到 push/rotate 后命名体系需重构。

---

## 7. 建议执行顺序

```
阶段 0（前置，无代码）
  ├─ 修 select_200.py 类别映射
  ├─ 下载并解压 HOI4D annotations 包
  └─ 解压 EgoDex 更多任务子目录

阶段 1（低成本，立即可做）
  ├─ 修 Bug 1（左右手夹爪）
  ├─ 修 Bug 2（schema 对齐）
  ├─ 修 Bug 4（仿真接入）
  └─ 重命名 grasp/ → interaction/（或先不改，避免额外 churn）

阶段 2（路线 A，核心收益）
  ├─ 抽掉 apply_release_constraint
  ├─ grasp_score → 接触类型分类器
  └─ 验证：push / pull / rotate / lift 四类各跑通

阶段 3（路线 B，突破瓶颈）
  ├─ 解析 mobility_v2.json
  ├─ scene_builder 加 revolute / prismatic
  └─ 验证：insert_remove_drawer / open_close 类任务

阶段 4（路线 C，覆盖长尾）
  ├─ VLM 打标 task / contact 语义
  └─ 用 action/color.json 做 ground truth 评测
```

---

## 8. 遗留问题

1. **优先哪条路线？** A（低成本先见效）还是 B（直接突破瓶颈）。若目标是论文，B 是必经之路且必须先补标注；若目标是工程可用，A 先跑。
2. **路线 C 是否现在做？** 它能覆盖 EgoDex 107 任务，但需要评测集设计。
3. **两份副本如何管理？** 合并成一份，还是继续双份同步。
4. **`grasp/` 目录与字段重命名的时机？** 阶段 2 之后重命名成本最低，但会让 diff 变大。