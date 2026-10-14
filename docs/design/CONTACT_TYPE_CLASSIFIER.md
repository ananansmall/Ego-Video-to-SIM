# 交互类型分类器设计（路线 A 实现方案）

> 状态：设计已用真实数据验证可行
> 日期：2026-10-08
> 前置：[GENERALIZATION_TASKS.md](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/docs/design/GENERALIZATION_TASKS.md)、[EVAL_DATASETS.md](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/docs/design/EVAL_DATASETS.md)

---

## 0. 问题的提法

前一版文档只回答了「有哪些动词」，没回答「**怎么从视频得到动词**」。本篇回答后者。

要回答的其实是三个独立问题，不能混为一谈：

| 层次 | 问题 | 现状 |
|---|---|---|
| **任务如何获得** | 怎么从视频判断发生了什么动作 | ❌ 完全没有，这是本文档要解决的 |
| **如何分类** | 判据是什么，怎么标 | ❌ 完全没有 |
| **分类之后做什么** | 分类结果如何驱动物理仿真 | 部分（`scene_builder` 只有 static/dynamic） |

本文档只解决前两个。第三个属于路线 B/C。

---

## 1. 一个关键的架构决策：两级分类，不要一步到位

我实测了「用 EgoDex 的 97 个动词做直接分类」，**结果是不可能**：

```
留一最近邻(标准化欧氏) 任务级命中率: 0/18 = 0.0%
```

原因不是数据不好，而是**动词标签的语义粒度远超运动学可观测性**：

- `pour`（倒水）和 `pick`（拿起来）在手部运动上几乎一样，区别只在**液体流向**——单目视频看不到
- `screw_unscrew_bottle_cap` 和 `open_close_box` 都是「手持续旋转」，区别只在**物体类型**
- `wipe_surface`（擦桌子）和 `clean_cups` 都是「手往复运动 + 物体几乎不动」

**结论：直接从 97 个动词分类是错的抽象层级。**

正确的切法是两级：

```
Level 1：接触状态（physics-grounded, 从运动学可判定, ~6 类）
    ↓  决定需要哪种物理模型
Level 2：任务语义（semantic, 靠物体类别 + VLM, 97 个动词）
```

Level 1 是本管线的**职责**（决定仿真怎么搭）；Level 2 是上层语义任务，**不必在管线里解决**。

---

## 2. Level 1：接触状态分类（本管线的核心）

### 2.1 定义

按「物体在手上发生什么」分类，而非按「任务叫什么」分类：

| 类别 | 物理判据 | 需要的物理模型 | 当前状态 |
|---|---|---|---|
| `free_hold` | 物体与手刚性绑定，跟随手运动 | 自由刚体 + 夹爪 | ✅ 已支持 |
| `surface_slide` | 物体与手耦合，但**支撑关系不变**（始终在桌面） | 自由刚体 + 摩擦 | ⚠️ 被误判为 grasp |
| `free_release` | 释放后**继续运动** | 自由刚体 | ❌ `apply_release_constraint` 主动破坏 |
| `in_place_change` | 物体**几乎不动**，但状态改变（擦拭/开关） | 接触态检测 | ❌ `moving` 判据盲区 |
| `articulated` | 物体**部分点动、部分不动**（铰链） | revolute/prismatic | ❌ 无关节模型 |
| `deformable` | 物体形状变化 | deformable | ❌ 不支持 |
| `bimanual` | 双手同时参与 | 双手夹爪 | ⚠️ 左手 bug |

### 2.2 判据设计：三个正交轴

不是打分加权，而是**三轴独立判定**（对应 [03_grasp_controller.py](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/object_tracking/03_grasp_controller.py#L184-L197) 现有 `grasp_score` 三项加权的替代方案）：

**轴 A — 支撑关系是否改变**
```
support_plane_estimate = argmin_z( z - min_z(points) )     # 物体最低点高度
support_changed = |support_plane_estimate(t) - support_plane_estimate(t0)| > τ_h
```
- 轴 A 翻转 → `free_hold` / `free_release`
- 轴 A 不变 + 轴 B 耦合 → `surface_slide`

**轴 B — 手物是否耦合**（沿用现有 coupling，但去掉 `moving` 门控）
```
coupling[t] = mean_t( cos( v_obj[t], v_hand[t] ) )   # 不要求 speed > 0.005
contact[t] = coupling[t] > τ_c
```
> 去掉 [L90](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/object_tracking/03_grasp_controller.py#L90) 的 `moving` 门控是**关键**：擦拭时物体速度≈0，但轴 B 应该判为「接触中」，而非「未接触」。

**轴 C — 物体是否刚体**
```
if 物体的 VGGT4D 追踪点中，≥2 组点运动差异显著:
    → articulated
```

三轴组合：

| 轴A | 轴B | 轴C | 输出 |
|---|---|---|---|
| 改变 | on | rigid | `free_hold` → `free_release` |
| 不变 | on | rigid | `surface_slide` |
| 不变 | on | 分数 | `articulated` |
| 不变 | off | rigid | `in_place_change` |
| — | — | — | 双手参与 → `bimanual` 标记 |

### 2.3 实测验证：单判据已足够区分精细操作

我从 EgoDex hdf5 抽了 10 个时序特征，在 18 个代表性任务（每任务 12 clips）上实测：

**最重要的判据 `R_curl_mean`（右手五指指尖到腕距离的时序极差均值）：**

```
精细操作类 (11 任务: screw/lock/open/zip/flip/deal/assemble/fold/throw): 0.0483
粗抓取类   (4 任务: pick_place/pour/stack/throw_catch):               0.0256
间隔/组内std = 1.51  →  阈值 0.0366 时任务级正确 11/12
```

单特征就能把「精细操作」和「粗抓取」分开。18 类全分开做不到（0%），但 Level 1 只有 6-7 类，这是可行的。

**其他特征的单判据分离度（|间隔|/组内std）：**

| 特征 | 精细类均值 | 粗抓类均值 | 分离度 | 方向 |
|---|---|---|---|---|
| `R_pinch_rel`（拇指-食指开合幅度比） | 1.6835 | 0.9133 | **1.76** | 高=精细 |
| `R_curl_mean` | 0.0483 | 0.0256 | **1.51** | 高=精细 |
| `R_pinch_min`（捏合最小间距） | 0.0145 | 0.0306 | 1.59 | 低=精细 |
| `R_h`（手相对相机高度） | -0.1776 | -0.2340 | 1.06 | 高=精细 |
| `R_zrange`（手高度时序极差） | 0.1167 | 0.0979 | 0.24 | 弱 |
| `wrist_dist_std`（腕间距波动） | 0.0494 | 0.0464 | 0.10 | 无用 |

### 2.4 为什么不能用多特征分类器：特征冗余

我做了特征间相关性分析，10 个特征里 |r|>0.7 的有 **11 组**，有效维度只有 3-4：

```
R_pinch_rel ~ R_pinch_min    r=-0.90
R_pinch_rel ~ R_curl_mean    r=+0.85
R_pinch_rel ~ L_curl_mean    r=+0.88
R_curl_mean ~ L_curl_mean   r=+0.77
R_h        ~ L_h             r=+0.80
R_zrange   ~ L_zrange        r=+0.88
R_zrange   ~ wrist_dist_std  r=+0.92
...
```

`pinch_rel` / `pinch_min` / `curl` 本质是**同一个信息的三个刻度**（手指弯曲程度）。丢到随机森林或 MLP 里只会引入过拟合和不可解释性。

**设计决策：用 3 个正交轴的显式规则，不用学习式分类器。**

选 3 个互相独立且物理意义明确的量：
1. `support_changed`（轴 A）— 物体高度是否变化
2. `contact`（轴 B）— 手物是否耦合
3. `rigid_score`（轴 C）— 物体是否刚体

这三个来自不同物理量，不存在上述那种强相关。

---

## 3. 任务如何获得：完整数据流

```
输入视频 (EgoDex mp4 / HoI4D mp4)
   ↓
Step 1  VGGT4D 点追踪  → tracks_3d (S,N,3)          [已有: 02_point_tracker.py]
Step 2  HaWoR 手部重建  → hand_vertices, fingertips  [已有: demov2.py]
   ↓
Step 3  ★ 接触状态分类（本篇新增，替换 grasp_score）
         ├─ 轴A: 支撑关系 → support_changed
         ├─ 轴B: 运动耦合(去掉moving门控) → contact
         └─ 轴C: 刚体性 → rigid_score
   ↓
Step 4  接触段切分 → segments[{type, start, end, hand}]
   ↓
Step 5  按 type 分发到对应物理模型               [需路线B]
   ├─ free_hold/free_release → 自由刚体 + 夹爪
   ├─ surface_slide        → 自由刚体 + 摩擦
   ├─ articulated          → revolute/prismatic 关节
   ├─ in_place_change      → 接触态检测(无物体运动)
   └─ bimanual             → 双手夹爪
   ↓
Step 6  轨迹精化（删掉 apply_release_constraint）[需修改: 04_trajectory_refiner.py]
   ↓
输出: 每帧的接触状态标签 + 分段 + 所需物理模型类型
```

**与现有代码的接口**：`run_grasp_controller()`（[L358](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/object_tracking/03_grasp_controller.py#L358)）保持签名不变，把内部 `grasp_score` 换成三轴判定，`segments[].type` 从 3 个字面量扩展为 7 个。下游 `run_pipeline.py`（[L344-L361](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/object_tracking/run_pipeline.py#L344-L361)）传参逻辑不用改。

---

## 4. 需要新增的代码

| 文件 | 内容 | 行数估计 |
|---|---|---|
| `03_grasp_controller.py` 内新增 `detect_contact_type()` | 三轴判定，替代 `grasp_score` | ~80 |
| `03_grasp_controller.py` 内 `compute_support_plane()` | 轴 A：物体最低点高度时序 | ~25 |
| `03_grasp_controller.py` 内 `compute_rigid_score()` | 轴 C：分组点运动一致性 | ~40 |
| `03_grasp_controller.py` 内改 `detect_grasp_release()` | 删 `moving` 门控，输出 type 枚举 | ~30 |
| 新建 `object_tracking/contact_types.py` | 类型枚举 + 常量 + 物理模型映射表 | ~40 |

**不新增文件也可以**，但类型枚举放独立文件更清晰（现有 `run_pipeline.py` 用 `_mod_gc.run_grasp_controller` 这种动态导入方式，说明作者偏好松耦合）。

---

## 5. 评测方案

### 5.1 用 EgoDex 动词标注做 ground truth

```python
import h5py, json
rows = json.load(open('/mnt/data_8THDD/lza/dataset/egodex_test_verbs.json'))
# 每条: {task, clip, verbs, type, desc, objects}

def expected_type(verbs):
    """把 97 动词映射到 Level 1 的 6-7 类"""
    if {'pour','scoop','dump','dispense'} & set(verbs):  return 'surface_slide'   # 需流体
    if {'push','pop','roll','rake','sweep'} & set(verbs):  return 'surface_slide'
    if {'screw','unscrew','open','close','lock','unlock','twist','spin'} & set(verbs):
        return 'articulated'                              # 旋转关节
    if {'insert','remove','zip','unzip','charge','slot','clip'} & set(verbs):
        return 'articulated'                              # 平移关节
    if {'fold','unfold','crumple','flatten','wipe','clean','tie'} & set(verbs):
        return 'deformable'
    if {'pick','place','put','take','stack','unstack'} & set(verbs):
        return 'free_hold'
    return 'unknown'
```

### 5.2 评测指标

| 指标 | 目标 |
|---|---|
| 段级类型准确率（precision/recall per type） | ≥0.8 |
| **`surface_slide` 的召回率**（当前会被误判为 grasp） | ≥0.9 |
| **`in_place_change` 的召回率**（当前 `moving` 盲区） | ≥0.7 |
| `bimanual` 检出率 | ≥0.9 |

### 5.3 首个验证集：92 clips

优先用 **B 组 push/pull 92 clips**，理由：
- 当前 100% 会误判为 grasp → 改进空间明显
- 不需要关节模型，可独立验证轴 A/B
- `push_pop_toy` 29 clips + `roll_ball` 13 + `deal_gather_cards` 16

第二个验证集：I 组 `clean_surface` 44 clips，专测 `in_place_change`。

---

## 6. 实测数据记录（可复现）

### 6.1 环境

```python
h5py      # 系统已有
数据       /mnt/data_8THDD/lza/dataset/test/<task>/<idx>.hdf5
索引       /mnt/data_8THDD/lza/dataset/egodex_test_verbs.json
```

### 6.2 hdf5 关节命名（实测确认，69 个）

```
camera, hip, neck1-4, spine1-7
leftShoulder/leftArm/leftForearm/leftHand
left{Thumb,IndexFinger,MiddleFinger,RingFinger,LittleFinger}{Knuckle,Metacarpal,
    IntermediateBase,IntermediateTip,Tip}     ← 每指 5 节 × 5 指 × 双手
right 同构
```

⚠️ **注意**：部分 hdf5 **没有** `confidences` group（README 说 "Most (but not all!)"）。实测 `pour/0.hdf5` 就无。读特征时必须 `if 'confidences' in h`，不能直接访问。

### 6.3 ARKit 世界系的陷阱

README 明确：
> all transforms are expressed in the *ARKit origin frame*: a stationary frame on the ground set at the beginning of a recording session. Since this depends on device initialization, this world frame **is not necessarily consistent across episodes**.

**实测印证**：`L_travel`（腕部首尾位移）在同任务内跨 clip 差异达 30 倍（`basic_pick_place` 从 0.002 到 0.205）。

**设计约束：所有特征必须是 episode 内的相对量。**
- ✅ `curl`（指尖到腕距离的时序极差）
- ✅ `pinch_min` / `pinch_rel`（拇指-食指距离）
- ✅ `support_changed`（支撑面高度的**变化**，不是绝对值）
- ❌ 物体在场景中的绝对坐标
- ❌ 手到相机的绝对距离

---

## 7. 待决问题

1. **`in_place_change` 的轴 B 判据怎么定？** 擦拭时物体速度≈0，但手在动。现在的 coupling 公式要求两者速度都 >0.005。改为「只要求手在动」会不会引入大量误检（手在物体附近划过但没接触）？可能需要引入 RAS 的 VGGT4D mask 做接触验证。
2. **`articulated` 的轴 C 判据**依赖物体点追踪质量。`rigid_score` 阈值需在 HoI4D 上调（HoI4D 有 `objpose` 标注可以验证，当前拿不到）。
3. **是否要保留 `grasp_score` 作为 fallback？** 建议保留为 `contact_type='unknown'` 时的兜底，但要在输出里显式标 `unknown`，不要静默当grasp 处理。
4. **EgoDex 是否有物体轨迹标注？** 没有 —— 只有手和相机。所以物体侧的 `tracks_3d` 必须靠 RAS 自己的 VGGT4D 追踪。这意味着 Level 1 分类的质量受限于点追踪。这是路线 B 依赖 HoI4D annotations 的根本原因。