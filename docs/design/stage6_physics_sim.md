# Stage 6: URDF 驱动的物理仿真

> **状态**: ✅ 已实现并在 **10 个数据集**上验证通过 (2026-10-09)
> **一句话**: Stage 5 把场景摆好之后，Stage 6 **只做一件事** —— 把场景跑一遍物理，然后出 GIF。

---

## 1. 范围：Stage 6 做什么、不做什么

Stage 5 修完之后，`final_scene_stage5.glb` 里物体的位置/朝向/尺度已经是最终值。
Stage 6 接在这个 GLB 之后，**职责边界刻意做窄**：

| Stage 6 **做** | Stage 6 **不做** |
|---|---|
| GLB → 每物体 OBJ → URDF（含碰撞体/质量/摩擦） | ❌ 不做 3D 重建（Stage 2/3） |
| PhysX 真实物理仿真（重力/碰撞/摩擦/惯量） | ❌ 不做空间对齐（Stage 4） |
| 逐帧记录位姿 → 渲染 GIF/MP4/PNG + JSON 报告 | ❌ 不做语义/关系推理（Stage 5） |
| 读取 Stage 6 材质标注作为物理输入之一 | ❌ 不做尺度校准、不做裁剪图提取 |

**为什么不做**：这些事 Stage 1~5 各自有专职环节，把它们塞进 Stage 6 会让
"仿真出图"这件唯一的正事被一堆前置/后置逻辑淹没（历史教训：`stage61_*` 曾经
衍生出 5 个互不相同又互相重复的渲染脚本，最后全部废弃）。

> 历史口径提示：`mainv3.py` 抬头里写的 "Stage 6: 材质感知的物理参数标注"
> 指的是 `tools/material_annotator.py`。按现在的分工，**它只是 Stage 6 的输入
> 提供方之一**（产出 `materials.json`），Stage 6 的主线是物理仿真本身。

---

## 2. 流程图

```
final_scene_stage5.glb  +  materials.json(可选)
        │
        ▼
┌──────────────────────────────────────────────────┐
│ ① URDF 生成   src/stage61_urdf_generator.py      │
│   GLB 拆解 → coacd 凸分解碰撞体 → 逐物体 URDF     │
│   → 合并 scene.urdf + urdf/mesh/                 │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ ② 物理仿真    tools/stage61_physics_sim.py       │
│   ├─ 解析 URDF: 质量 / 惯量 / 摩擦 / 初始位姿     │
│   ├─ materials.json 覆盖摩擦 & 恢复系数           │
│   ├─ PhysX step(): 每个 physics step 都真的跑     │
│   └─ 逐帧记录全部 actor 的 (p, q)                │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│ ③ 渲染        pyrender 离屏 (方案 B)             │
│   真实 VGGT 相机轨迹 + 真实 OpenCV 内参           │
│   → GIF / MP4 / frame0.png / report.json         │
└──────────────────────────────────────────────────┘
```

**为什么要分 ②③ 两步而不是让 SAPIEN 顺手渲染**：实测 3 个场景的 SAPIEN 渲染引擎
与实体位姿同步脱钩（物理正常、组件齐全、相机位姿读回正确，但画面恒为纯灰）。
现在 SAPIEN **完全不创建渲染器**，只跑物理；画面交给 pyrender。详见 §7。

---

## 3. 输入 / 输出

### 输入（`output_v2/<scene>/`）

| 文件 | 必需 | 用途 |
|---|---|---|
| `final_scene_stage5.glb` | ✅ | 物体 geometry + 世界位姿，①的输入 |
| `extrinsics/{i}.txt` | ✅ | VGGT 相机轨迹（world→camera 4×4），③的相机 |
| `intrinsic.txt` | ✅ | 相机内参 3×3（fx/fy/cx/cy），③的投影 |
| `color/` | ⭕ | 用来探测真实图像分辨率 |
| `materials.json` | ⭕ | Stage 6 材质，见 §5。没有就用默认值 |

### 输出（`output_v2/<scene>/`）

| 文件 | 说明 |
|---|---|
| `urdf/scene.urdf` | 合并后的场景 URDF |
| `urdf/mesh/*.obj` | visual + collision mesh |
| `urdf/stage61_report.json` / `stage61_validation.json` | ①的生成与静置校验报告 |
| **`stage61_physics.gif`** | **主产物**：真实相机视角的物理仿真动画 |
| `stage61_physics.mp4` | 同内容无损全帧 |
| `stage61_physics_frame0.png` | 首帧，便于快速预览 |
| `stage61_physics_report.json` | 质量/惯量/位移/渲染自检/材质对接全量报告 |

---

## 4. 就这几个脚本（保持稳定，不再增加）

| 脚本 | 职责 |
|---|---|
| `src/stage61_urdf_generator.py` | ① URDF 生成 |
| `tools/stage61_physics_sim.py` | ②③ 物理 + 渲染，唯一的成品脚本 |
| `tools/stage61_batch_run.sh` | 逐场景**独立进程**跑上面那个（同进程反复创建/销毁 Scene 会让 EGL 上下文失效） |
| `tools/run_stage6_only.py` | 辅助：给已有场景单独补跑 Stage 6 材质标注 |
| `tools/analysis/make_stage61_gallery.py` | 辅助：把所有场景的 GIF 汇总成单文件 HTML 画廊 |

### 已废弃（不要用）

`stage61_sim_sapien.py`、`stage61_render_pyrender.py`、`stage61_render_scene.py`、
`stage61_fixed_camera.py`、`stage61_visualize_sapien.py`
—— 共同问题：不读 mass/inertia、或干脆不 step 物理、或用彩色 box 占位。
判断标准很简单：**写 `stage61_physics.*` 的是现行版本，写 `stage61_sim.png` /
`stage61_render.png` / `stage61_sapien_*` 的都是旧版本。**

---

## 5. 材质输入：唯一需要注意的接口

> 材质标注本身怎么做的，见 [附录 A](#9-附录-astage-6-材质标注输入准备)。
> 本节只讲**仿真侧怎么消费它**。

`materials.json` 的 key 是 `bowl_inst0`，而 URDF 的 link 名是 `bowl_0`。
**两边命名约定不一致**，是历史上"Stage 6 标注结果无人消费"的直接原因。

现在的处理（`stage61_physics_sim.py::resolve_material`）：
1. 精确匹配 `bowl_0 → bowl_inst0`
2. 实例编号对不上时，仅当**该类别所有实例材质完全一致**才按类别取值
3. 否则宁可放弃

覆盖到的物理量：

| 物理量 | 来源 | 说明 |
|---|---|---|
| 静/动摩擦 mu1/mu2 | **Stage 6 材质** | 取代 URDF 里 EmbodiedGen 硬编码的 0.60/0.50 |
| 恢复系数 restitution | **Stage 6 材质** | 取代硬编码的 0.02 |
| 质量 mass | URDF 的 VLM 值（默认） | `--mass_source density` 可换成 密度×体积 |
| 惯量 inertia | 由 mesh + 最终质量现算 | URDF 里的 `ixx=iyy=izz=1.0` 是占位值，必须重算 |

**质量为什么没跟着换成密度法**：试过，结果是 `newspaper_1` 算出 10.9kg、
`table_0` 35kg。密度质量 / VLM 质量的中位数约 **2~5x**，换算成线性尺度误差
**1.2~1.7x** —— 与 `SAM3D_SCALE_DRIFT_REPORT.md` 记载的 ~1.5x 漂移吻合，说明是
mesh 整体偏大而不是密度不该用。所以只用那些**不依赖绝对尺度**的量（摩擦/恢复），
质量留给 VLM。这个比值本身也被做成尺度漂移诊断量写进报告（见 §7 表格）。

---

## 6. 怎么跑

```bash
cd ReplicateAnyScene
PY=/mnt/data/lza/conda_envs/sapien/bin/python

# ① URDF（通常会重跑 VLM + coacd，较慢；已有 urdf/scene.urdf 时可跳过）
$PY src/stage61_urdf_generator.py \
    --glb output_v2/<scene>/final_scene_stage5.glb \
    --output_dir output_v2/<scene> \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B

# ②③ 物理 + pyrender 渲染出 GIF（约 20~40s / 场景）
$PY tools/stage61_physics_sim.py --output_dir output_v2/<scene> --pyrender_render

# 批量（默认 8 个场景，必须每场景独立进程）
PYRENDER=1 bash tools/stage61_batch_run.sh 120

# 缺 materials.json 时先补跑 Stage 6 材质（需要空闲 GPU）
CUDA_VISIBLE_DEVICES=1 $PY tools/run_stage6_only.py --scene <scene>

# 汇总画廊
$PY tools/analysis/make_stage61_gallery.py   # -> output_v2/stage61_gif_gallery.html
```

> 环境：②③ 用 `conda_envs/sapien`（sapien 3.0.0b1 + pyrender 0.1.45）；
> ① 和 Stage 6 材质标注用 `conda_envs/ReplicateAnyScene`（sapien 3.0.3 + torch）。
> Stage 6 材质标注**需要空闲 GPU**，务必 `CUDA_VISIBLE_DEVICES` 指定。

---

## 7. 验证结果（10 个数据集）

全部走 `--pyrender_render` 方案 B，120 帧 @30fps，真实 VGGT 相机轨迹 + 真实内参：

| 场景 | 物体 | 位移 | 彩色占比 | 空帧 | 相机帧 | Stage6 命中 | 隐含尺度误差 | 分辨率 |
|---|---|---|---|---|---|---|---|---|
| 2chair-bottle_rescaled | 5 | 2 | 82.8% | 0/120 | 120 | 5/5 | 1.229x | 384×688 |
| 3chair-bottle_rescaled | 21 | 19 | 31.1% | 0/120 | 120 | 21/21 | 1.314x | 384×688 |
| phone_vggt_omega | 9 | 7 | 58.5% | 36/120 | 100 | 9/9 | 1.259x | 384×688 |
| 082_C5_CellPhone_rescaled | 25 | 25 | 47.8% | 0/120 | 120 | 25/25 | 1.283x | 688×384 |
| 121_C5_CellPhone_161deg | 12 | 5 | 66.5% | 0/120 | 100 | 12/12 | 1.204x | 688×384 |
| 151_C7_Plate_214deg | 29 | 28 | 12.4% | 0/120 | 160 | 29/29 | 1.713x | 688×384 |
| 7_vggt_omega | 8 | 3 | 3.2% | 0/120 | 113 | 8/8 | 1.402x | 688×384 |
| 7a_vggt_omega | 8 | 4 | 1.4% | 0/120 | 113 | 8/8 | 1.361x | 688×384 |
| egodex_0_rescaled | 8 | 6 | 25.2% | 0/120 | 120 | 8/8 | 1.654x | 688×384 |
| egodex_34_rescaled | 10 | 8 | 21.8% | 0/120 | 120 | 10/10 | 1.412x | 688×384 |

- **合计 135 个物体，107 个发生真实位移**（阈值 5mm）——物理确实在跑，不是摆pose。
- **Stage 6 材质 100% 精确命中**（十场景全部），涉及 14 种材质：
  aluminum / cardboard / ceramic / fabric / glass / hard_plastic / metal / paper /
  plastic / rubber / soft_plastic / steel / wood / default。
- **渲染自检全部通过**（判据：彩色像素占比 > 0.5%）。
- 隐含线性尺度误差 **1.20 ~ 1.71x（中位 1.36x）**，跨 10 个场景一致 →
  这是 SAM3D 尺度漂移的一个独立佐证。

---

## 8. 已知限制

1. **未接入 mainv3**：`run_stage61()` 只被自己的 `__main__` 调用，必须手工跑。
   `--enable_stage61` 从旧计划文档（`stage61_urdf_plan.md`，已合并为存根）起就是
   计划，**至今未实现**。
2. **SAPIEN 渲染失效 → 走方案 B**：3 个场景（7 / phone / egodex_0）画面恒为纯灰，
   根因在渲染引擎内部。方案 B 是**绕过不是修复**。
3. **`phone_vggt_omega` 有 36 个"空帧"**（索引 83~119，见报告 `empty_frame_indices`）：
   不是渲染失败——那段真实视频是手持手机特写、相机离被摄物极近，而 sim 里手机已
   落地离场，画面只剩地面。
4. **质量依赖 VLM 猜测**：见 §5，受制于上游的尺度漂移。
5. **环境分裂**：②③ 用 `sapien` env，① 用 `ReplicateAnyScene` env，文档别写混。
6. **mesh 命名双前缀**：`urdf/mesh/block_0_block_0.obj`（类别+实例名重复拼接）。

---

## 9. 附录 A：Stage 6 材质标注（输入准备）

> 实现：`tools/material_annotator.py` + `tools/materials_db.json`
> 入口：`mainv3.py --enable_stage6`（默认关闭），或单独补跑 `tools/run_stage6_only.py`

### 9.1 设计取向：材质驱动，而不是参数驱动

| | ❌ 参数驱动 | ✅ 材质驱动（现行） |
|---|---|---|
| VLM 输出 | 直接生成 density/friction 数值 | 只输出**材质名**（离散类别） |
| 稳定性 | 自由生成数值，不可控 | 离散类别，稳定可复现 |
| 扩展 | 换材质要改 prompt | 只改 `materials_db.json` |
| 可解释 | 数值无来源 | 材质有语义 |

物理数值由 `materials_db.json` 统一映射，VLM 不碰数值。
**该表是人工整理的参考值表**（工程手册量级），不是权威数据库——SAPIEN 参数本就
是"看起来合理的近似"，同量级正确即可；表里收录 17 种材质 + `default` 兜底。

### 9.2 流程

```
optimal_frames/*.jpg  +  all_instances.pkl
        │
        ▼  逐实例裁剪图 → VLM 识别材质名（一次一实例）
        ▼  material → materials_db.json 查表
        ▼  未识别 / 未列出的实例 → default
materials.json  +  materials_debug.json
```

### 9.3 输出格式（实例级 key）

```json
{
  "bowl_inst0": {"material": "ceramic", "density": 2250,
                 "static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.3},
  "bottle_inst0": {"material": "glass", "density": 2500, ...}
}
```

- key 是 **`<类别>_inst<编号>`**（实例级）；`materials_debug.json` 另存
  VLM 识别链路与 mapping log 供排查。
- **仿真侧只消费 `static_friction / dynamic_friction / restitution`**（见 §5）。

### 9.4 两种已知的"静默退化"

1. **裁剪图 key 层级不匹配**：实例级 key（`banana_0`）对类别级文件名
   （`banana_inst0_frame33.jpg`）过滤 → 交集为空 → 全部走默认参数、耗时 0.0s。
   已在 `material_annotator.py` 里剥离 `_N` 后缀修复。
2. **缺空闲 GPU**：`device_map="auto"` 可能挑到被占满的卡，表现为 CUDA OOM +
   权重 offload 到 CPU。**必须 `CUDA_VISIBLE_DEVICES=<空闲卡>`**。

两者都不会中断管线（有 `_fallback_default_materials()` 兜底），所以"跑完了"不等于
"参数是真实的"——要看 `materials.json` 里是否有非 `default` 的材质。

---

## 10. 附录 B：① URDF 生成（`src/stage61_urdf_generator.py`）要点

### 10.1 五步

```
final_scene_stage5.glb
  Step1 GLB 拆解     : 每物体居中 → 导出裸 OBJ + world_position + ground_z
  Step2 逐物体 URDF  : Open3D 渲染4视图 → Qwen VLM 判属性(类别/高度/质量/摩擦)
                       → coacd 凸分解 → EmbodiedGen generate_urdf()
  Step3 合并         : 复制 mesh 到 urdf/mesh/ + 统一前缀 → scene.urdf
                       (fixed joint, parent=base, origin=world_position)
  Step4 SAPIEN 静置校验: 装进场景 step 若干步 → stage61_validation.json
  Step5 报告          : urdf/stage61_report.json
```

### 10.2 调用 EmbodiedGen 的部分（sys.path 导入，不复制代码）

| 模块 | 用途 |
|---|---|
| `embodied_gen.validators.urdf_convertor.URDFGenerator` | VLM 输出解析 + URDF 模板填充 + coacd |
| `embodied_gen.utils.simulation.load_actor_from_urdf` | SAPIEN 校验加载 |

### 10.3 踩过的坑（都已在代码里处理）

1. **nvdiffrast 缺失** → `import URDFGenerator` 时会连带 import
   `render_asset3d` 而崩溃。用 mock module 顶上即可（`generate_urdf` 根本不调它）。
2. **`attr_dict["real_height"]` 缺失** → `generate_urdf()` KeyError。
   要自己按 `(min_height+max_height)/2` 补。
3. **mesh 双前缀** → 合并阶段会给已带前缀的名字再加一次，产出
   `urdf/mesh/block_0_block_0.obj`。功能无影响，仅命名难看。
4. **gazebo 标签必须保留** → `load_actor_from_urdf()` 硬依赖
   `.//collision/gazebo/mu1`，合并 scene.urdf 时不能丢。
5. **URDF 里的 `ixx=iyy=izz=1.0` 是占位值**（EmbodiedGen 模板），转动惯量必须
   由 ② 按 mesh + 最终质量重算，否则物体会以错误角加速度翻滚。
