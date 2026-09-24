# mainv2.py 里是怎么调模型的（通俗版）

> 一句话：**输入一段视频，跑完所有模型后输出一个 GLB 3D 场景**。
> 中间用到 4 类模型：VLM（聊天/看图问答）、VGGT 系列（3D 重建）、SAM3（分割跟踪）、SAM3D（生成 3D 资产）。

---

## 第 0 步：整体流程图

把整条流水线画出来，你就能明白每个模型在哪个位置干活：

```
                        ┌──────────────────────────────────────────────┐
                        │  输入：一段视频 mp4                             │
                        └──────────────────────┬───────────────────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          ▼                    ▼                    ▼
                    ┌─────────┐         ┌────────────┐       ┌──────────┐
                    │Stage 1 │         │  Stage 2  │       │ Stage 3 │
                    │ 物体发现│         │ 3D 重建+   │       │ 资产生成│
                    └────┬────┘         │  分割去重  │       └────┬─────┘
                         │              └─────┬──────┘            │
                         │                    │                   │
                    用到什么？              用到什么？            用到什么？
                    · VGGT（原版）         · VGGT-Omega         · SAM3D
                    · VLM ×2               · SAM3 image         · 每实例调 1 次
                    · SAM3 image           · SAM3 video         · 输出 3D mesh
                    · CLIP 去重            · CLIP 去重           │
                         │                    │                   │
                    ┌────┴────┐         ┌────┴────┐        ┌────┴─────┐
                    │ 输出：   │         │ 输出：   │        │ 输出：    │
                    │ scene_   │         │ 每个物体 │        │ GLB 模型 │
                    │ _stage1  │         │ 的逐帧   │        │ 列表     │
                    │ .json    │         │ mask     │        └─────────┘
                    └─────────┘         └──────────┘
```

**耗时分布**（按日志 `116_C5_CellPhone_185deg_vggt_omega` 实测）：

| 阶段 | 耗时 | 占比 | 干的主要活 |
|---|---|---|---|
| Stage 1 | 436 s | 16% | 发现场景里有什么物体 |
| Stage 2 | 876 s | 32% | 重建 3D + 每个物体贴标签 |
| Stage 3 | 770 s | 28% | 每个物体生成 3D 模型 |
| Stage 5 | 510 s | 19% | 精修物体间关系 |
| 基础精修 | 131 s | 5% | 把物体摆到地上/墙上 |

---

## 第 1 步：VLM（本地 Qwen，看图聊天）

### 先讲人话

VLM 就是个**能看图说话的大模型**。在这个项目里它不是远程 API，是本地加载的 Qwen 模型，放在 `/mnt/data/lza/models/Qwen3.5-9B`。

它的用途是：你给它一张图 + 一段话（"这图里有啥？"），它用文字回答你（"图里有杯子、椅子"）。

### 统一入口：所有 VLM 调用都走这一个函数

不管你是要检测物体、判断关系、还是识别材质，最终都调**同一个函数**：

```python
from src.vlm_logger import vlm_inference, set_vlm_log_dir
```

**调用前先设置一次日志目录**（把对话记录到文件里，方便调试）：

```python
set_vlm_log_dir("./output_v2/xxx/")   # 之后所有调用都会写到这个目录下的 vlm_conversation.jsonl
```

**这个函数的样子**：

```python
response_text = vlm_inference(
    image,             # PIL 图像（一张）
    model,             # Qwen 模型对象（你自己加载的）
    processor,         # 对应的处理器（tokenizer）
    prompt,            # 发给模型的文本指令
    max_new_tokens=1024,   # 最多生多少 token
    caller="stage1_detect", # 打日志用的标签
)
```

**输入**：一张图 + 一段 prompt
**输出**：模型生成的纯文本（你要自己解析 JSON 或关键词）

**这个函数内部做了什么**（`src/vlm_logger.py` 第 33-94 行）：

```
第 1 步：把图和文字拼成 chat 格式
第 2 步：processor 转成模型能吃的 tensor
第 3 步：model.generate() 前向推理（do_sample=False，贪心解码）
第 4 步：切掉输入部分，只留新 token
第 5 步：batch_decode 还原成文本
第 6 步：写到 vlm_conversation.jsonl 做日志
```

### 模型怎么加载到 `model` 和 `processor`

两种写法，各阶段用不同的：

| 谁在用 | 怎么加载 | 代码位置 |
|---|---|---|
| Stage1 | `load_vlm_model(checkpoint_path)`（内部用 AutoModelForImageTextToText） | `tools/generate_scene_json_stage1.py` 第 2145 行 |
| Stage5 / Stage6 | 内联写 `AutoModelForVision2Seq.from_pretrained(...)` | `tools/infer_relations_scene_graph.py` 第 559-573 行 |

两种写法最终得到的 `model` / `processor` 都长一样，都能传给 `vlm_inference()`。

### 主线里两次 VLM 调用（都在 Stage1）

#### VLM ① 物体检测 —— "图里都有啥？"

**在哪调**：`tools/generate_scene_json_stage1.py::detect_objects_in_frames()`，第 3 步

**输入**：Stage1 用 SimRecon 从视频里抽出来的 10~12 张关键帧
**prompt**（发给模型的文本）：

```
List all visible objects in this image. Output JSON only.

{"objects": [{"name": "cup"}, {"name": "chair"}]}

Rules:
- Simple singular names: "cup", "chair", "table"
- Ignore: hands, body parts, walls, floors, ceilings
- List each object type once per frame

Output JSON:
```

**调用方式**：

```python
for i, frame_path in enumerate(frame_paths):
    image = Image.open(frame_path).convert("RGB")
    output_text = vlm_inference(
        image, model, processor, VLM_DETECT_PROMPT,
        max_new_tokens=1024, caller="stage1_detect"
    )
    json_str = _extract_json_from_text(output_text)   # 从文本里抠 JSON
    result = json.loads(json_str)
    # result["objects"] = [{"name":"cup"}, ...]
```

**输出**：`{"objects":[{"name":"cup"}, {"name":"chair"}]}`
**之后**：可选辅助 VLM 补测（主模型类别少时再跑一个高精度模型合并）

#### VLM ② 关系判断 —— "谁在谁上面？"

**在哪调**：`tools/generate_scene_json_stage1.py`，第 7 步
**输入**：VLM ① 找到的物体名列表 + floor/wall 是否可见（Step6 SAM3 分割的结果）

**prompt**（动态拼出来的）：

```python
def _build_relationship_prompt(object_names, has_floor, has_wall):
    return f"""You are a precise spatial relationship analysis API. Output ONLY valid JSON.

**Scene information**:
- Objects in the scene: [{", ".join(object_names)}]
- FLOOR is {"VISIBLE" if has_floor else "NOT visible"}
- WALL is {"VISIBLE" if has_wall else "NOT visible"}

**Valid relationships** (MUST use EXACTLY one of these 4):
- "supported by floor"
- "supported by other objects"
- "attached to wall"
- "embedded in wall"

**Output format**:
{{"relationships": {{"object_name": "relationship"}}}}

**CRITICAL REQUIREMENTS**:
- You MUST output exactly {len(object_names)} objects
- If unsure, default to: "supported by floor"

Now analyze this scene and OUTPUT JSON:"""
```

**输出**：`{"relationships":{"cup":"supported by floor","painting":"attached to wall"}}`
**后处理**：`_post_process_relationships()` 用物理常识再纠正一次（比如柜子不可能被杯子支撑）

### 后续阶段（Stage5/6）的 VLM 调用速查

| 阶段 | 问什么 | prompt 文件 | 输出什么 |
|---|---|---|---|
| Stage1 补充检测 | 点云裁剪图里是什么大物体 | `SUPPLEMENTARY_VLM_PROMPT` | 物体名 |
| Stage5.1 场景图 | ID 标注图里的物体关系 | SimRecon 风格 prompt | `{"objects":[{"instance","category","relation","parent"}]}` |
| Stage5.2 放置策略 | 应该摆在哪（on_top/inside/...） | `build_placement_prompt()` | `{"strategy":"on_top"}` |
| Stage5.2 精修程度 | 当前位置对不对 | 内联 prompt | major / minor / correct |
| Stage6 材质 | 这个物体是什么材质 | `MATERIAL_PROMPT_TEMPLATE` | `{"materials":{...}}` |

---

## 第 2 步：VGGT 系列（3D 重建）

### 先讲人话

VGGT 系列是**把一堆图片转成 3D 点云**的模型。你给它 120 张视频帧，它输出每个像素对应的 3D 坐标、相机的位姿、深度图。

项目里支持 3 个版本，用 `--vggt_model` 命令行参数选：

```
--vggt_model vggt         (原版，默认)
--vggt_model vggt_omega   (Omega 版)
--vggt_model vggt4d       (4D 版)
```

### 在 mainv2 里怎么选

`mainv2.py::run_stage2()` 第 341-356 行：

```python
if vggt_model_type == "vggt":
    frames = load_video_frames(input_video, max_frames=120)  # 抽帧
    model  = load_vggt_model()                                # 加载
    result = vggt_predict(frames, model)                      # 推理

elif vggt_model_type == "vggt_omega":
    frames = load_vggt_omega_frames(input_video, max_frames=120)
    model  = load_vggt_omega_model()
    result = vggt_omega_predict(frames, model)

elif vggt_model_type == "vggt4d":
    ...
```

三个版本输出的 dict 结构完全一样：

```python
{
    "world_points":      numpy.ndarray,   # (S, H, W, 3) 每帧每像素的世界坐标
    "world_points_conf": numpy.ndarray,   # 置信度
    "colors":            numpy.ndarray,   # 颜色
    "depths":            numpy.ndarray,   # 深度
    "extrinsics":        numpy.ndarray,   # 相机外参
    "intrinsic":         numpy.ndarray,   # 相机内参
}
```

**下游代码不需要知道用的是哪个版本，直接读这个 dict 就行。**

### 三个版本的区别

| | VGGT 原版 | VGGT-Omega | VGGT4D |
|---|---|---|---|
| 分辨率 | 518 | 512 | 518 |
| 加载 | `from_pretrained` | `torch.load` 手动加载权重 | `from_pretrained` |
| 出 world_points | 模型直接输出 | 用 depth 反投影得到 | 模型直接输出 |
| 速度 | 中等 | 较快 | 中等 |

### VGGT-Omega 详细调用链

Omega 版的加载方式比较特别，不是 `from_pretrained`，而是手动 `torch.load` + `load_state_dict`：

**抽帧**：
```
load_vggt_omega_frames("xxx.mp4", max_frames=120)
  └─ ffmpeg 抽帧到临时目录
  └─ 读成 (S, 3, 512, 512) tensor，值域 [0, 1]
```

**加载模型**：
```
load_vggt_omega_model(".../vggt_omega_1b_512.pt")
  └─ VGGTOmega().eval()
  └─ torch.load(checkpoint, weights_only=True)
  └─ model.load_state_dict(state_dict)
```

**前向推理**：
```
vggt_omega_predict(images, model)
  └─ model(images)       # 一次前向，内部跑 Aggregator + CameraHead + DenseHead
  └─ encoding_to_camera(predictions["pose_enc"])  # 9D → R, t, fx, fy, cx, cy
  └─ unproject_depth_to_world_points(depth, extrinsic, intrinsic)  # 深度反投影
  └─ 返回 dict（和上面一样的结构）
```

### 坐标系对齐（自动把 Z 轴转正）

拿到 3D 点云后，相机坐标系可能是任意朝向的，要把它摆正（Z 轴向上，地面在 z=0）。对齐失败会**一路回退**：

```
第 1 级：align_to_room_coordinate_system()      ← 几何法，大多数场景一次过
    └─ 失败
第 2 级：align_via_objects()                     ← 用物体几何
    └─ 失败
第 3 级：[align_via_vlm_floor_points()]          ← 用 VLM 找地面点
    └─ 失败
第 4 级：[align_via_large_plane()]               ← 用 SAM3 找大平面
    └─ 失败
第 5 级：align_via_geocalib()                    ← 兜底算法
```

你的日志里第 46 行：`stage1_strict 一步成功 (R[2,2]=-0.7672)`，说明第一级就过了，后 4 级都没走。

---

## 第 3 步：SAM3（分割 + 跟踪）

### 先讲人话

SAM3 干两件事：
1. **Image 版**：给一张图 + 文字提示，把它分割出来（比如"floor"）
2. **Video 版**：给一段视频 + 文字提示，自动跟踪每个实例的逐帧 mask

用的是同一个权重文件：`./models/SAM3/sam3.pt`。

### SAM3 Image —— 分 floor/wall

**在哪用**：`src/object_segmentation.py::segment_wall_and_floor()`
**怎么调**：对每帧图、每个 prompt 各跑一次：

```python
# 先给图片编码一次
inference_state = sam3_image_model.set_image(image)

# 对每个 floor prompt 各跑一次
for prompt in ["floor", "ground", "carpet", "bed", "table", ...]:
    sam3_image_model.reset_all_prompts(inference_state)     # 清上轮
    inference_state = sam3_image_model.set_text_prompt(
        state=inference_state, prompt=prompt
    )
    masks = inference_state['masks'].cpu().numpy()          # 拿到 mask
```

**18 个 prompt 覆盖不同场景**：
- floor: `floor`, `ground`, `carpet`, `bed`, `table`, `desk`, `horizontal surface`, `tabletop`, `counter`, `shelf`
- wall: `single wall`, `wall`, `window`, `curtain`, `door`, `bookshelf`, `cabinet`, `vertical surface`

### SAM3 Video —— 逐类跟踪每个物体

**在哪用**：`mainv2.py::run_stage2()` 第 470-489 行

**核心逻辑**：

```python
# 第 1 步：创建 session
sam3_video_model = load_sam3_video_model()
session_id = sam3_video_model.handle_request(
    request=dict(type="start_session", resource_path=video_path)
)["session_id"]

# 第 2 步：对每个类别跑一次
for category in ["bottle", "cabinet", "chair"]:   # Stage1 发现的类别
    category_masks = segment_and_track(
        category, sam3_video_model, session_id
    )
```

**`segment_and_track` 内部做了什么**：

```
第 1 步：reset_session（清空旧 prompt）
第 2 步：add_prompt(text="bottle", frame_index=0)  ← 只在第 0 帧给一个文字提示
第 3 步：propagate_in_video()  ← 从第 0 帧一路传播到最后一帧
第 4 步：汇总 → 每个实例 = [{frame_id:0, mask:...}, {frame_id:1, mask:...}, ...]
```

**输入**：一个类别名（比如 "bottle"）
**输出**：每个 bottle 实例的逐帧 mask 列表

**之后做类内去重**（3D 点云重叠度，overlap_thre=0.3），最后跨类用 CLIP 再合并一次。

---

## 第 4 步：SAM3D（生成 3D 模型）

### 先讲人话

SAM3D 干一件事：**给一张图 + 一个 mask，生成这个物体的 3D mesh（GLB 模型）**。

### 在哪用

`mainv2.py::run_stage3()` 第 613 行：

```python
all_instances = generate_3d_asset_in_subprocess(
    deduplicated_all_masks,       # 每类的逐帧 mask（SAM3 video 的输出）
    all_optimal_frame_ids,        # 每个实例的最佳视角帧号
    colors,                       # 每帧 RGB
    world_points,                 # 每帧 3D 点云
    extrinsics,                   # 每帧相机外参
)
```

### 为什么要开子进程

SAM3D 内部用 `torch.compile` 或自定义算子，和 VGGT/VLM 直接放同一个进程会 CUDA 冲突，所以单独开一个 spawn 子进程：

```python
ctx = mp.get_context("spawn")
process = ctx.Process(target=_generate_all_instances_worker, ...)
process.start()
```

### 每个实例的调用

```python
# 子进程里：
inference = Inference(config_file="./models/SAM3D/checkpoints/pipeline.yaml")

for 每个实例:
    image    = colors[最佳帧]          # 一张 RGB 图
    mask     = 该实例在该帧的 mask
    pointmap = world_points[最佳帧]    # 每像素的世界坐标
    extrinsic = extrinsics[最佳帧]     # 相机外参

    result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
```

**`generate_3d_asset` 做了什么**（`src/instance_generation.py` 第 12-54 行）：

```
第 1 步：把世界坐标转成相机坐标（矩阵乘）
第 2 步：调 SAM3D → inference(image, mask, pointmap=...)
         返回 GLB mesh + rotation + scale + translation
第 3 步：把相机坐标变换成世界坐标（z-up）
         最终 T = inv(extrinsic) @ adjust @ l2c @ y2z
```

**输出**：`{"original_mesh": trimesh, "T": 4×4 变换矩阵}`

---

## 第 5 步：把整件事串起来

### 实际运行回放（日志 `116_C5_CellPhone_185deg_vggt_omega`）

```
23:57:28  启动 mainv2，参数：
          --vggt_model vggt_omega     决定走 Omega 版
          --vlm_checkpoint Qwen3.5-9B

23:57:28  Stage 1：subprocess 跑 generate_scene_json_stage1.py
          ├─ VGGT 原版重建 100 帧
          ├─ SimRecon 选 10 张关键帧
          ├─ VLM ① 每帧调 vlm_inference 问"图里有啥"
          ├─ 射线投射到 3D
          ├─ SAM3 image 分 floor/wall
          ├─ VLM ② 问"谁支撑谁"
          └─ 输出：bottle / cabinet / chair（436 s）

00:04:44  Stage 2：mainv2 进程内
          ├─ VGGT-Omega 重建 100 帧 → 得到世界点云
          ├─ 坐标系对齐：第 1 级就成功了（R[2,2]=-0.7672）
          │    后面 4 级回退都没走
          ├─ SAM3 video 逐类跟踪：
          │     bottle → 1 个 / cabinet → 6 个 / chair → 1 个 = 8 个实例
          ├─ CLIP 跨类去重 → 8 个不变
          └─ 输出 deduplicated_all_masks（876 s）

00:19:20  Stage 3：spawn 子进程 SAM3D
          ├─ 选最优视角：bottle→帧67, cabinet→多帧
          ├─ 8 个实例各调一次 SAM3D 生成 3D mesh
          ├─ 3D Mesh 去重 + 3 维投票 → 全部保留
          └─ 输出 all_instances.pkl（770 s）

00:34:22  基础精修：把 8 个物体摆到 floor 上（131 s）

00:34:22  Stage 5：语义精修
          ├─ 5.1 场景图：VLM 全场景预览 + 每实例 3 帧投票
          │    bottle_0 → supported by chair_0
          │    chair_0  → supported by floor
          ├─ 5.2 几何精修：
          │    VLM 判定 bottle_0 的放置策略 = on_top
          │    VLM 判定精修程度 = correct
          │    实际执行：bottle 摆到 chair 顶面
          └─ 输出 final_scene_stage5.glb（510 s）

00:42:52  结束，总耗时 2723 s
```

### 参数从哪来

| 参数 | 在哪设 | 怎么用 |
|---|---|---|
| `--vggt_model` | 命令行 | 决定 Stage2 用 vggt / vggt_omega / vggt4d |
| `--vlm_checkpoint` | 命令行 | 传给 `_resolve_vlm_checkpoint()`，决定用哪个 Qwen |
| `--vlm_checkpoint_aux` | 命令行 | Stage1 辅助 VLM（可选） |
| `--enable_stage5` | 命令行 | 打开 Stage5 语义精修 |
| `--enable_stage6` | 命令行 | 打开 Stage6 材质物理 |
| `--enable_multi_vlm` | 命令行 | Stage1 启用主+辅 VLM 联合检测 |
| `max_frames` | 命令行（默认 120） | Stage2 重建抽多少帧 |

### `vlm_checkpoint` 怎么被传到各阶段

```
命令行 --vlm_checkpoint /path/to/model
    │
    ▼  mainv2.py:_resolve_vlm_checkpoint()
    │
    ▼  run_stage1()  传入 subprocess 参数
    │
    ▼  run_stage2()  （注意：这个函数签名里还没加 vlm_checkpoint 参数！）
    │
    ▼  run_stage5()   显式传入
    │
    ▼  run_stage6()   显式传入
```

---

## 附录：文件索引

| 文件 | 干嘛的 |
|---|---|
| `mainv2.py` | 主流程，调 Stage1/2/3/5/6 |
| `tools/generate_scene_json_stage1.py` | Stage1，里面有 VLM ①② 的全部代码 |
| `src/vggt_omega_predict.py` | VGGT-Omega 的加载和推理 |
| `src/object_segmentation.py` | SAM3 image（分 floor/wall）+ SAM3 video（跟踪）|
| `src/instance_generation.py` | SAM3D 的生成（spawn 子进程） |
| `src/vlm_logger.py` | `vlm_inference` 统一推理函数 |
| `tools/infer_relations_scene_graph.py` | Stage5.1 场景图 |
| `tools/refine_inter_object_placement.py` | Stage5.2 几何精修 |
| `tools/material_annotator.py` | Stage6 材质 |
| `src/models.py` | 所有 `load_*_model()` 和 `unload_model()` |
