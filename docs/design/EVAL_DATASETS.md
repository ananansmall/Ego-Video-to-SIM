# 评测集清单与数据获取状态

> 更新：2026-10-08
> 用途：手物交互管线（RAS `object_tracking/`）任务泛化的评测集梳理

---

## 0. 结论速览

| 数据集 | 本地状态 | 任务数 | clip 数 | 动作标注 | 可用性 |
|---|---|---|---|---|---|
| **EgoDex test** | ✅ **已完整解压** | 111 | 3243 | ✅ 逐 episode 动词级 | **立即可用** |
| EgoDex extra | ⏳ 下载中 | — | — | 同上 | 下载中 |
| EgoDex train (part1-5) | ❌ 未下载 | ~194 | ~33万 | 同上 | Apple CDN 可达 |
| HoI4D RGB | ✅ 已下载 | 26 动词 | 2969 mp4 | ❌ 无 action 标注 | 仅视频 |
| HoI4D annotations | ❌ **权限失效** | — | — | 目标 | 阻塞 |

**核心发现**：EgoDex 的 HDF5 里带 **逐 episode 的动词级动作标注**（`llm_verbs` / `llm_type` / `llm_objects`），这是路线 A（接触类型分类器）缺失的监督信号，且**本地已有 3243 条**，无需等 HoI4D。

---

## 1. EgoDex test split（本地已有，主力评测集）

### 1.1 位置与完整性

```
/mnt/data_8THDD/lza/dataset/test/<task_name>/{<idx>.mp4, <idx>.hdf5}
```

完整性核验结果：

| 检查项 | 结果 |
|---|---|
| 任务目录数 | 111 / 111（与 `egodex_test.zip` 内清单完全一致，无缺失无多余） |
| clip 数 | 3243 mp4 + 3243 hdf5（严格 1:1 配对） |
| 抽样任务 clip 数比对 | `pour` 33=33、`insert_remove_drawer` 17=17、`push_pop_toy` 29=29、`screw_unscrew_bottle_cap` 29=29、`wipe_kitchen_surfaces` 6=6 |
| 来源确认 | `egodex_test.zip` 大小 17304529397 与 Apple 官方 `test.zip` 的 `content-length` **完全一致** → 本地即官方原版 |

### 1.2 HDF5 标注字段（关键）

单个 clip 的 `pour/0.hdf5` 实测：

```python
attrs: ['annotated', 'annotator_version', 'environment', 'extra',
        'llm_description', 'llm_description2', 'llm_objects', 'llm_type',
        'llm_verbs', 'object', 'session_name', 'task', 'trim']

llm_type        = 'reset'
llm_verbs       = ['pour']
llm_objects     = ['gray cup', 'coke']
llm_description = 'Pour coke into the gray cup on the green tablecloth.'
task            = 'pour'
environment     = 'tablecloth:green, position:standing, background:none'
trim            = [0, 4386]

transforms: 69 joints, 每个 (N, 4, 4) float32
  camera / hip / leftArm / leftForearm / leftHand
  leftIndexFinger{Tip,Knuckle,Metacarpal,IntermediateBase,IntermediateTip} ... (双手 × 每指 5 节)
  → 双手 + 双手腕 + 全部手指关节的完整 SE(3)
camera/intrinsic: (3, 3)
confidences: 每关节标量置信度 [0,1]，0 表示完全遮挡
```

**`llm_verbs` 是关键字段** —— 这是逐 episode 的动词标签，可直接作为分类器的 ground truth。

### 1.3 全量动词标注统计（3243 clips 全扫描）

**97 个去重动词**，按频次 top 40：

| 频次 | 动词 | 频次 | 动词 |
|---|---|---|---|
| 546 | remove | 65 | scoop |
| 508 | insert | 65 | dump |
| 496 | assemble | 59 | zip |
| 490 | place | 59 | unzip |
| 489 | disassemble | 58 | screw |
| 479 | pick | 58 | unscrew |
| 266 | stack | 50 | put |
| 228 | unstack | 46 | thread |
| 122 | fold | 46 | unthread |
| 122 | clean | 45 | unfold |
| 121 | charge | 43 | throw |
| 121 | uncharge | 39 | dry |
| 79 | play | 32 | gather |
| 78 | open | 31 | wrap |
| 78 | close | 29 | roll |
| 66 | tie | 29 | push |
| 66 | untie | 28 | take |
| 65 | use | 27 | pour |

完整 97 动词表：
```
add arrange assemble away bagging boil braid build catch charge clean clean up
cleanup click clip close collect color crumple deal declutter disassemble dispense
dry dump fidget flatten flip fold fry gather insert knead load lock make measure
open paint peel pick pick up place play point pop pour push put put away put down
rake remove reset roll scoop screw serve set up setup sleeve slot sort spin stack
staple stock sweep take thread throw tie topple twist type unbraid uncharge unclip
unfold unlock unpeel unscrew unsleeve unstack unstake unstock unthread untie unwrap
unzip use wash wipe wrap wrapping write zip
```

`llm_type` 分布：

| 类型 | 数量 | 含义 |
|---|---|---|
| `reversible` | 2240 | 可逆（打开→关闭、插入→取出） |
| `reset` | 619 | 需要重置状态 |
| `reset-free` | 384 | 无需重置（一次性消耗） |

> `reversible` 占 69%，天然适合做「状态变化」类任务（开合类）的评测。

### 1.4 按交互类型分组的任务清单

标注 ✅ = 现管线可跑；⚠️ = 会被误判为 grasp；❌ = 需新增物理模型

#### A. 抓取 / 搬运 / 放置 — 766 clips ✅ 现管线已覆盖

| clips | 任务 | 动词 |
|---|---|---|
| 277 | `basic_pick_place` | pick, place |
| 153 | `vertical_pick_place` | pick, place |
| 86 | `stack_unstack_plates` | stack, unstack |
| 80 | `stack` | stack, unstack |
| 49 | `pick_place_food` | pick, place |
| 36 | `stack_unstack_bowls` | stack, unstack |
| 22 | `stack_remove_jenga` | remove, stack |
| 16 | `gather_roll_dice` | gather, roll |
| 13 | `build_unstack_lego` | build, unstake |
| 12 | `stack_unstack_cups` | stack, unstack |
| 12 | `stack_unstack_tupperware` | stack, unstack |
| 10 | `insert_dump_blocks` | insert |

#### B. 推 / 拉 / 滑移 — 92 clips ⚠️ 当前会被误判为 grasp

| clips | 任务 | 动词 | 误判原因 |
|---|---|---|---|
| 29 | `push_pop_toy` | pop, push | 手物同向移动 → coupling≈1 |
| 20 | `rake_smooth_zen_garden` | rake | 同上 |
| 16 | `deal_gather_cards` | deal, gather | 同上 |
| 13 | `roll_ball` | roll | 同上 |
| 7 | `arrange_topple_dominoes` | arrange, topple | 同上 |
| 7 | `sweep_dustpan` | sweep | 同上 |

**这组是路线 A 的首要验证目标** —— 只需改分类器，不动物理引擎。

#### C. 铰链开合 — 667 clips ❌ 需 revolute/prismatic 关节

| clips | 任务 | 动词 |
|---|---|---|
| 84 | `insert_remove_furniture_bench_cabinet` | insert, remove |
| 63 | `charge_uncharge_device` | charge, uncharge |
| 58 | `charge_uncharge_airpods` | charge, uncharge |
| 55 | `insert_remove_usb` | insert, remove |
| 50 | `insert_remove_furniture_bench_round_table` | insert, remove |
| 49 | `insert_remove_plug_socket` | insert, remove |
| 43 | `insert_remove_airpods` | insert, remove |
| 41 | `open_close_insert_remove_tupperware` | close, insert, open, remove |
| 35 | `zip_unzip_bag` | unzip, zip |
| 32 | `insert_remove_bookshelf` | insert, remove |
| 29 | `screw_unscrew_bottle_cap` | screw, unscrew |
| 26 | `add_remove_lid` | add, remove |
| 24 | `open_close_insert_remove_case` | close, insert, open, remove |
| 24 | `zip_unzip_case` | unzip, zip |
| 23 | `stock_unstock_fridge` | stock, unstock |
| 21 | `lock_unlock_key` | lock, unlock |
| 20 | `paint_clean_brush` | clean, paint |
| 17 | `insert_remove_drawer` | insert, remove |
| 17 | `screw_unscrew_fingers_fixture` | screw, unscrew |
| 13 | `open_close_insert_remove_box` | close, insert, open, remove |
| 12 | `screw_unscrew_allen_fixture` | screw, unscrew |
| 12 | `slot_batteries` | slot |
| 5 | `clip_unclip_papers` | clip, unclip |

> 注意 `screw_unscrew_*` / `lock_unlock_key` 明确是**旋转关节**，`insert_remove_*` 是**平移关节**。这批数据同时覆盖 revolute 和 prismatic 两类。

#### D. 装配 / 拆解（多物体）— 165 clips ❌ 需多物体协同

| clips | 任务 | 动词 |
|---|---|---|
| 38 | `insert_remove_utensils` | insert, remove |
| 37 | `assemble_disassemble_legos` | assemble, disassemble |
| 28 | `put_in_take_out_glasses` | put, take |
| 19 | `insert_remove_cups_from_rack` | insert, remove |
| 14 | `insert_remove_tennis_ball` | insert, remove |
| 8 | `assemble_disassemble_soft_legos` | assemble, disassemble |
| 7 | `assemble_disassemble_structures` | assemble, disassemble |
| 7 | `assemble_disassemble_tiles` | assemble, disassemble |
| 7 | `assemble_jenga` | assemble |

#### E. 倾倒 / 浇注 — 129 clips ❌ 需流体

| clips | 任务 | 动词 |
|---|---|---|
| 65 | `scoop_dump_ice` | dump, scoop |
| 33 | `pour` | pour |
| 18 | `make_sandwich` | make |
| 8 | `load_dispense_ice` | dispense, load |
| 5 | `boil_serve_egg` | boil, serve |

#### F. 柔性物 / 形变 — 241 clips ❌ 需 deformable

| clips | 任务 | 动词 |
|---|---|---|
| 77 | `basic_fold` | fold |
| 26 | `tie_and_untie_shoelace` | tie, untie |
| 19 | `wrap_unwrap_food` | unwrap, wrap |
| 19 | `knead_slime` | knead |
| 18 | `fold_unfold_paper_basic` | fold, unfold |
| 17 | `fold_stack_unstack_unfold_cloths` | fold, stack, unfold, unstack |
| 17 | `wrap` | fold, wrapping |
| 16 | `wipe_screen` | wipe |
| 11 | `braid_unbraid` | braid, unbraid |
| 10 | `fold_unfold_paper_origami` | fold, unfold |
| 6 | `wipe_kitchen_surfaces` | wipe |
| 5 | `crumple_flatten_paper` | crumple, flatten |

> **EgoDex 不含拖cloth / 拖擦这类大面积柔性体** —— 这是它相对 HoI4D 的短板。`wipe_*` 属刚体表面摩擦，`crumple_flatten_paper` 涉及纸张形变但幅度小。

#### G. 工具 / 精细操作 — 352 clips ❌ 需工具耦合

| clips | 任务 | 动词 |
|---|---|---|
| 61 | `use_rubiks_cube` | twist, use |
| 46 | `thread_unthread_bead_necklace` | thread, unthread |
| 36 | `play_piano` | play |
| 22 | `play_reset_connect_four` | play, reset |
| 22 | `point_and_click_remote` | click, point |
| 21 | `play_mancala` | play |
| 21 | `color` | color |
| 20 | `write` | write |
| 20 | `insert_remove_shirt_in_tube` | insert, remove |
| 12 | `type_keyboard` | type |
| 12 | `sort_beads` | sort |
| 11 | `peel_place_sticker` | peel, place, unpeel |
| 11 | `fidget_magnetic_spinner_rings` | fidget, spin, stack |
| 9 | `measure_objects` | measure |
| 7 | `use_chopsticks` | use |
| 6 | `staple_paper` | staple |
| 3 | `set_up_clean_up_chessboard` | clean up, set up |
| 2 | `flip_coin` | flip |

#### H. 双手协同 — 435 clips ⚠️ 左手夹爪 bug 会失效

| clips | 任务 | 动词 |
|---|---|---|
| 90 | `assemble_disassemble_furniture_bench_chair` | assemble, disassemble |
| 87 | `assemble_disassemble_furniture_bench_square_table` | assemble, disassemble |
| 87 | `assemble_disassemble_furniture_bench_stool` | assemble, disassemble |
| 76 | `assemble_disassemble_furniture_bench_drawer` | assemble, disassemble |
| 59 | `assemble_disassemble_furniture_bench_desk` | assemble, disassemble |
| 36 | `assemble_disassemble_furniture_bench_lamp` | assemble, disassemble |

> **这组直接暴露 [03_grasp_controller.py](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/object_tracking/03_grasp_controller.py#L313-L322) 的左手夹爪 bug**：`hand` 硬编码 `"right"`，`gripper_signal_left` 恒 OPEN。

#### I. 清洁 / 擦拭 — 170 clips ⚠️ 无平移的接触

| clips | 任务 | 动词 |
|---|---|---|
| 44 | `clean_surface` | clean |
| 40 | `dry_hands` | dry |
| 30 | `clean_cups` | clean |
| 28 | `clean_tableware` | clean |
| 12 | `declutter_desk` | declutter |
| 6 | `setup_cleanup_table` | cleanup, setup |
| 6 | `put_away_set_up_board_game` | put away, set up |
| 2 | `wash_put_away_dishes` | away, put, wash |
| 1 | `wash_fruit` | wash |
| 1 | `wash_kitchen_dishes` | wash |

> **这组暴露 `moving` 判据的盲区**：[03_grasp_controller.py](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/object_tracking/03_grasp_controller.py#L90) 的 `moving = (obj_s > 0.005) & (hand_s > 0.005)`，擦拭时物体几乎不动 → 耦合度强制归零 → 完全检测不到。

#### J. 其他 — 243 clips

| clips | 任务 | 动词 |
|---|---|---|
| 41 | `open_close_insert_remove_tupperware` | close, insert, open, remove |
| 22 | `throw_and_catch_ball` | catch, throw |
| 21 | `throw_collect_objects` | collect, throw |
| 20 | `put_toothpaste_on_toothbrush` | put |
| 15 | `flip_pages` | flip |
| 14 | `pick_up_and_put_down_case_or_bag` | pick up, put down |
| 13 | `roll_ball` | roll |
| 11 | `insert_remove_bagging` | bagging, insert, remove |
| 42 | `tie_untie_rubberband` | tie, untie |
| 4 | `sleeve_unsleeve_cards` | sleeve, unsleeve |

### 1.5 汇总

| 交互类型 | clips | 占比 | 需要的能力 |
|---|---|---|---|
| A 抓取/搬运/放置 | 766 | 23.6% | ✅ 已有 |
| C 铰链开合 | 667 | 20.6% | revolute + prismatic 关节 |
| H 双手协同 | 435 | 13.4% | 双手夹爪（修 bug 即可） |
| G 工具/精细 | 352 | 10.9% | 工具耦合 |
| F 柔性物 | 241 | 7.4% | deformable |
| J 其他 | 243 | 7.5% | 混合 |
| I 清洁/擦拭 | 170 | 5.2% | 接触态检测（修 moving 判据） |
| D 装配/拆解 | 165 | 5.1% | 多物体协同 |
| E 倾倒/浇注 | 129 | 4.0% | 流体 |
| B 推/拉/滑移 | 92 | 2.8% | 接触类型分类（修 grasp_score） |

---

## 2. EgoDex extra / train（Apple 官方源，可下载）

### 2.1 官方 CDN 探测结果（2026-10-08 实测）

| 文件 | HTTP | 大小 |
|---|---|---|
| `part1.zip` | 200 | 321,588,941,964 (300G) |
| `part2.zip` | 200 | 327,274,733,628 (305G) |
| `part3.zip` | 200 | 326,263,668,370 (304G) |
| `part4.zip` | 200 | 329,365,549,875 (307G) |
| `part5.zip` | 200 | 331,277,374,642 (309G) |
| `extra.zip` | 200 | 223,924,972,352 (209G) |
| `test.zip` | 200 | 17,304,529,397 (17G) ← **本地已有** |

**全部 HTTP 200，无需鉴权**（与 HoI4D 的 OneDrive 403 完全不同）。

- 训练集总量：~1525 GB（5 卷）
- `extra.zip`：209 GB（冻结后补充的样本）
- 磁盘剩余：1.6 TB → 可容纳 `extra.zip` + 1~2 个 train分卷

### 2.2 train 分卷任务分布

README 明确：

> basic_pick_place 与 furniture_bench 类任务在 **part2.zip**

因卷内任务混合，下载 train 分卷前需先确认目标任务的卷号。`extra.zip` 是全任务混合，适合直接补充任务多样性。

### 2.3 下载记录

```
目标: /mnt/data_8THDD/lza/dataset/egodex_extra.zip   (209 GB)
源:   https://ml-site.cdn-apple.com/datasets/egodex/extra.zip
方式: aria2c 8 连接 + 127.0.0.1:17890 代理
日志: /tmp/egodex_extra_dl.log
状态: 下载中
```

---

## 3. HoI4D（本地部分可用，标注阻塞）

### 3.1 现状

```
/mnt/data_8THDD/lza/dataset/HOI4D_RGB/
├── HOI4D_release/          # 2969 个 .mp4，无任何标注
├── HOI4D_selected_100/ 300/ # RAS 预处理产物
├── HOI4D_selected_200/     # 200 clips + manifest.json（含 T 任务 ID）
├── select_100.py / 200.py / 300.py
├── download_annotations.sh # 脚本存在，但分享链接已失效
└── HOI4D_annotations.zip   # ❌ 不存在
```

`HOI4D_release/` 内容统计：
```
文件类型: 2969 mp4 + 1 md
objpose: 0   action: 0   2Dseg: 0   3Dseg: 0   align_depth: 0
```

**完全没有 `action/color.json`（动作分段）和 `mobility_v2.json`（关节标注）。**

### 3.2 官方源探测结果（2026-10-08）

| 资源 | Graph API 结果 |
|---|---|
| RGB Video | ✅ 200，`HOI4D_release.zip` 23,269,777,121 |
| **Annotations** | ❌ **403 accessDenied** — "The sharing link no longer exists, or you do not have permission to access it" |
| CAD Models | ❌ 403 accessDenied |
| Depth Video | ❌ 404 itemNotFound |
| Camera Parameters | ❌ 404 itemNotFound |
| Hand Pose | ❌ 404 itemNotFound |
| ActionSeg Test Set | ❌ 403 accessDenied |
| SemSeg Test Set | ❌ 403 accessDenied |

**根因**：不是网络问题。`rclone.conf` 里 `lza` 这个 OneDrive 账号对数据集所有者 cid `12E5C3DBEFFD0594` 只有 RGB Video 一个 share 的权限，其余 share 已对该账号关闭。官网 [hoi4d.github.io](https://hoi4d.github.io/)（2026-03-27 更新）的链接与本地脚本硬编码完全一致，即**官网页面本身未更新失效链接**。

百度云备用源同样不可用（匿名分享已改版需登录态，`share/init` 返回「页面不存在」，提取码 verify 返回 `errno -12`）。

### 3.3 仍可利用的部分

`manifest.json` 保留了每条 clip 的 `source_path`，其中含 `/T1/`–`/T6/` 任务 ID：

```python
"source_path": ".../ZY20210800004/H4/C1/N34/S190/s01/T3/align_rgb/image.mp4"
                                                     ^^^^ 任务 ID
```

配合 [task_definitions.csv](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/hoi4d/HOI4D-Instructions/definitions/task/task_definitions.csv) 可反查任务名 —— **这是目录名丢失后唯一残留的任务标签来源**。

### 3.4 ⚠️ 类别映射 bug（必须修）

`select_200.py` L23-L28 的 `CATEGORY_NAMES` 与官方 mapping 完全不同：

| ID | select_200.py | 官方实际 | 后果 |
|---|---|---|---|
| C1 | Bowl | **ToyCar** | 类别名全错 |
| C3 | Bottle | **Laptop** | C3 屏幕开合数据被标成刚体瓶 |
| C4 | Box | **StorageFurniture** | **T1–T4 恰是开抽屉/开门，却标成刚体** |
| C6 | Pan | **Safe** | C6 保险箱门数据丢失 |
| C12 | Laptop | Kettle | |
| C20 | Paper | Chair | |

`ARTICULATED_CATS = [C11..C20]` 这个划分建立在错误名字上。按官方 mapping，真正带关节的是 **C3 Laptop / C4 StorageFurniture / C6 Safe / C8 Bucket / C9 Scissors / C14 TrashCan / C17 Lamp / C20 Chair**。

---

## 4. 动词体系对照（HoI4D 26 ↔ EgoDex 97）

| 物理交互类型 | HoI4D 动词 | EgoDex 动词 | 需关节 |
|---|---|---|---|
| 抓取搬运放置 | Pick and place, Put/Take in drawer | pick, place, put, take, stack, unstack | — |
| 推拉滑移 | Push toy car | push, roll, gather, arrange, topple | — |
| 铰链旋转 | Open/close (display/drawer/door), Turn and fold | open, close, screw, unscrew, lock, unlock, twist, spin | ✅ revolute |
| 铰链平移 | — | insert, remove, charge, uncharge, zip, unzip, clip, unclip, slot, stock, unstock | ✅ prismatic |
| 倾倒浇注 | Pour water, Fill with kettle | pour, scoop, dump, dispense, boil | ❌ 流体 |
| 工具使用 | Cut apple, Cut something, Clamp | use, write, type, play, click, staple, paint, peel | ❌ 工具耦合 |
| 柔性形变 | Bind the paper | fold, unfold, crumple, flatten, wrap, tie, braid, knead | ❌ deformable |
| 双手协同 | Pour water（需双手） | assemble/disassemble furniture_bench | — |
| 状态变化 | Turn on and turn off | charge, uncharge, plug, switch | ✅ revolute |

**结论**：EgoDex 的动词体系**完全覆盖** HoI4D，且额外多了 prismatic（插入/取出）、柔性形变、工具使用三大类。**单靠 EgoDex 即可覆盖路线 A + 路线 B 的评测需求**，HoI4D 标注不再是硬阻塞。

---

## 5. 标注产物

### 5.1 动词索引表（已生成）

```
/mnt/data_8THDD/lza/dataset/egodex_test_verbs.json
```

结构：
```json
[
  {
    "task": "pour",
    "clip": "0",
    "verbs": ["pour"],
    "type": "reset",
    "desc": "Pour coke into the gray cup on the green tablecloth.",
    "objects": ["gray cup", "coke"]
  },
  ...
]
```

3243 条记录，可直接作为分类器评测的 ground truth。

### 5.2 建议补充的索引

- 按 `llm_verbs` 反查 → 每类动词对应的 clip 列表
- 按 `llm_type`（reversible/reset/reset-free）分组 → 用于「状态变化」任务评测
- HoI4D 侧：从 `manifest.json` 的 `source_path` 解析 `/T(\d+)/` → task 名

---

## 6. 待办

| 项 | 状态 |
|---|---|
| `egodex_extra.zip` 下载 | ⏳ 进行中 |
| `extra.zip` 解压 + 动词扫描 | 待下载完成 |
| 评估是否下载 `part2.zip`（含更多 pick_place / furniture_bench） | 待定（305 GB） |
| HoI4D annotations | ❌ 外部阻塞，需换账号或联系数据集作者 |
| 修 `select_200.py` 类别映射 | 待办 |
| 编写动词索引生成脚本（`build_verb_index.py`） | 待办 |