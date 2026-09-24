# Stage 6.1: URDF 生成 + SAPIEN 物理仿真

> **状态**: ✅ 代码已实现 (2026-07-30)

## 整体流程图

```
final_scene_stage5.glb
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 0: 路径配置 & 导入                                      │
│  ├─ sys.path.insert(EmbodiedGen_ROOT)                      │
│  ├─ sys.path.insert(ReplicateAnyScene_ROOT)               │
│  ├─ import trimesh, open3d, numpy, os, ET, sapien         │
│  ├─ import URDFGenerator (EmbodiedGen)                     │
│  ├─ import load_actor_from_urdf (EmbodiedGen)              │
│  └─ import QwenClient (本地新建)                            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: GLB 拆解                                             │
│  ├─ trimesh.load(glb_path)                                  │
│  ├─ for node_name, mesh in scene.geometry.items():          │
│  │   ├─ centroid = mesh.vertices.mean()   → 世界位置        │
│  │   ├─ bbox_min = mesh.vertices.min()      → 接地高度       │
│  │   ├─ category = node_name.split("_")[0]                 │
│  │   ├─ mesh.vertices -= centroid        → 居中             │
│  │   └─ mesh.export(f"{node_name}.obj") → 本地保存          │
│  └─ 输出: objects_info = [{name, category, obj_path,        │
│                              position, ground_z}, ...]      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 逐个生成 URDF (for obj in objects_info)            │
│                                                              │
│  ├─ Step 2a: 渲染 4 视图                                     │
│  │   └─ open3d_offscreen_render(obj_path)                   │
│  │       ├─ 前视图: rotation=(0, 0, 0)                       │
│  │       ├─ 右视图: rotation=(0, 90, 0)                      │
│  │       ├─ 后视图: rotation=(0, 180, 0)                     │
│  │       └─ 左视图: rotation=(0, -90, 0)                     │
│  │       输出: ["renders/{obj}_0.png", ..., "renders/{obj}_3.png"]
│  │       → 保存到 output/urdf/{obj}/urdf_renders/            │
│  │                                                      │
│  ├─ Step 2b: VLM 推理                                       │
│  │   └─ QwenClient.query(prompt_template, render_paths)    │
│  │       prompt = EmbodiedGen 的 prompt_template            │
│  │       内部: Qwen3.5-9B 看图 → 输出                       │
│  │       返回: 格式化后的标准文本 (GPT兼容格式)               │
│  │                                                      │
│  ├─ Step 2c: 解析 VLM 输出                                  │
│  │   └─ URDFGenerator.parse_response(response)              │
│  │       输出: attr_dict = {category, description,           │
│  │                         min_height, max_height,           │
│  │                         min_mass, max_mass, mu1, mu2}     │
│  │                                                      │
│  └─ Step 2d: 生成单个 URDF                                  │
│      └─ URDFGenerator.generate_urdf(                         │
│            obj_path,                                        │
│            output_dir,                                      │
│            attr_dict,                                       │
│            output_name=node_name)                           │
│      内部:                                                    │
│      ├─ 加载 mesh → 归一化 → 缩放到真实高度                   │
│      ├─ 保存缩放的 mesh                                       │
│      ├─ 复制 .mtl/.png (如果存在)                             │
│      ├─ coacd 凸分解 → 碰撞 mesh                             │
│      ├─ 填充 URDF XML (visual + collision + inertial)        │
│      └─ 输出: output/urdf/{obj}/{obj}.urdf                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 合并为 scene.urdf                                   │
│                                                              │
│  ├─ 创建 <robot name="scene">                                │
│  ├─ <link name="base">                                       │
│  ├─ for each obj:                                            │
│  │   ├─ 读取 {obj}.urdf                                      │
│  │   ├─ 更新 mesh 引用:                                       │
│  │   │   "mesh.obj" → "{obj}_mesh.obj"                      │
│  │   ├─ 复制 mesh/collision 到 output/urdf/mesh/            │
│  │   ├─ 更新 link name → {obj}                               │
│  │   ├─ 创建 <joint type="fixed">                            │
│  │   │   ├─ parent: base                                     │
│  │   │   ├─ child: {obj}                                     │
│  │   │   ├─ origin: xyz={world_position}                     │
│  │   │   └─ origin: rpy=0 0 0                                │
│  │   └─ append link + joint to scene URDF                    │
│  └─ 输出: output/urdf/scene.urdf                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: SAPIEN 验证 (可选, --validate)                      │
│                                                              │
│  ├─ scene = sapien.Scene()  ← 无头模式, 无 renderer           │
│  ├─ scene.add_ground(0)                                      │
│  ├─ for each obj:                                            │
│  │   └─ load_actor_from_urdf(scene, {obj}.urdf, pose)        │
│  │       ← EmbodiedGen 的 load_actor_from_urdf               │
│  │       内部:                                                │
│  │       ├─ 解析 URDF XML                                    │
│  │       ├─ 创建 PhysxMaterial (mu1/mu2)                     │
│  │       ├─ 加载 collision mesh (凸分解)                     │
│  │       └─ 加载 visual mesh                                 │
│  ├─ for 200 steps:                                           │
│  │   └─ scene.step()                                         │
│  └─ 输出: validation_report.json                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 文件位置总览

### 新建的文件

| 文件 | 路径 | 职责 |
|------|------|------|
| `stage61_urdf_generator.py` | `ReplicateAnyScene/src/stage61_urdf_generator.py` | 主入口，三步：拆解→生成→合并→验证 |
| `qwen_client.py` | `ReplicateAnyScene/tools/qwen_client.py` | Qwen VLM 适配器 |

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `mainv2.py` | 添加 `--enable_stage61` 参数，在 Stage 5 后调用 `src.stage61_urdf_generator.run_stage61()` |

### 调用的 EmbodiedGen 代码（不复制，sys.path 导入）

| 文件 | 导入方式 | 用途 |
|------|---------|------|
| `embodied_gen/validators/urdf_convertor.py` | `from embodied_gen.validators.urdf_convertor import URDFGenerator` | 解析 VLM 输出 + 生成 URDF |
| `embodied_gen/data/convex_decomposer.py` | 被 URDFGenerator 内部调用 | 凸分解碰撞体 |
| `embodied_gen/utils/simulation.py` | `from embodied_gen.utils.simulation import load_actor_from_urdf` | SAPIEN 验证阶段加载 URDF |

### 调用的 EmbodiedGen 但**不直接使用**的代码（不修改、不导入）

| 文件 | 原因 |
|------|------|
| `embodied_gen/utils/process_media.py` (render_asset3d) | 依赖 nvdiffrast，用 Open3D 替代 |
| `embodied_gen/data/differentiable_render.py` | nvdiffrast 依赖 |
| `embodied_gen/utils/gpt_clients.py` (GPTclient) | 用 QwenClient 替代 |
| `embodied_gen/models/layout.py` | 布局生成，不相关 |
| `embodied_gen/scripts/room_gen/export_scene.py` | Blender 脚本，不相关 |

---

## 2. 冲突分析

### 2.1 路径导入冲突

```
src/stage61_urdf_generator.py:
  ├─ from src.* import *              # 需要 ReplicateAnyScene_ROOT 在 sys.path
  ├─ from tools.* import *            # 需要 ReplicateAnyScene_ROOT 在 sys.path
  └─ from embodied_gen.* import *     # 需要 EmbodiedGen_ROOT 在 sys.path
```

**问题**：`mainv2.py` 运行在 ReplicateAnyScene 根目录，`src/stage61_urdf_generator.py` 运行在 src/ 下，两者路径不同。

**解决**：在 `stage61_urdf_generator.py` 顶部用绝对路径解析：

```python
_REPLICATE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EMBODIED_GEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "EmbodiedGen"))

for p in [_REPLICATE_ROOT, _EMBODIED_GEN_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)
```

### 2.2 GPTclient 单例冲突

```
EmbodiedGen 内部定义:
  GPT_CLIENT = GPTclient(endpoint=..., api_key=...)   ← 模块级单例
  URDFGenerator(GPT_CLIENT)                           ← 默认参数引用它
```

**问题**：即使我们传了自定义 `QwenClient`，EmbodiedGen 模块中 `GPT_CLIENT` 单例仍然存在。如果其他 EmbodiedGen 模块（如 quality_checkers）被 import，它们也会引用这个全局单例。但 Stage 6.1 不调用 quality_checkers，所以**实际不影响**。

**解决**：`URDFGenerator(qwen_client, ...)` 传入自定义 client，不引用 `GPT_CLIENT` 单例。

### 2.3 模块导入顺序冲突

```
import 链:
  stage61_urdf_generator.py
    → URDFGenerator (urdf_convertor.py)
      → from embodied_gen.data.convex_decomposer import decompose_convex_mesh
      → from embodied_gen.utils.process_media import render_asset3d  ← 导入但不用
      → from embodied_gen.utils.gpt_clients import GPT_CLIENT, GPTclient
```

**问题**：导入 `urdf_convertor.py` 时，Python 会执行文件顶部的所有 `import`，包括 `render_asset3d` 和 `GPT_CLIENT`。`render_asset3d` 依赖 `nvdiffrast`，`GPT_CLIENT` 依赖 `openai`。如果环境里没有这些库，import 就会失败。

**解决**：`openai` 已安装（Qwen 用 transformers，不影响），`nvdiffrast` 没安装。**必须处理 `render_asset3d` 的 import 失败问题**。

```python
# 方案: 在 urdf_convertor.py 的 import 之前做 patch
# 在 urdf_convertor.py 的顶部, render_asset3d 的 import 失败时,
# 用 mock object 替代, 因为 generate_urdf 不调用 render_asset3d

# 实现:
import sys
class _MockModule:
    def __getattr__(self, name):
        return lambda *a, **k: None

sys.modules.setdefault("embodied_gen.utils.process_media", _MockModule())

# 然后再 import URDFGenerator
from embodied_gen.validators.urdf_convertor import URDFGenerator
```

### 2.4 `generate_urdf()` 对 `attr_dict["real_height"]` 的依赖

```python
# urdf_convertor.py 第 267-269 行:
real_height = attr_dict["real_height"]
scale = round(real_height / raw_height, 6)
mesh = mesh.apply_scale(scale)
```

**问题**：`generate_urdf()` 期望 `attr_dict` 有 `"real_height"` 键。我们的 `parse_response()` 不生成这个键。

**解决**：`parse_response()` 只返回 min/max height/mass。`real_height` 需要在 `__call__` 中计算（`__call__` 第 516-518 行）。所以我们直接调 `generate_urdf()` 时，需要手动添加：

```python
attr_dict["real_height"] = round((attr_dict["min_height"] + attr_dict["max_height"]) / 2, 4)
```

### 2.5 `generate_urdf()` 对 mtl/png 的复制

```python
# urdf_convertor.py 第 280-285 行:
for file in self.mesh_file_list:  # ["material_0.png", "material_1.png", "material.mtl", ...]
    src = os.path.join(input_dir, file)
    if os.path.isfile(src):
        shutil.copy(src, mesh_folder)
```

**问题**：从 GLB 提取的 mesh 是裸 OBJ，没有 .mtl 和 .png。

**解决**：`self.mesh_file_list` 默认为空列表，不会复制。URDFGenerator 的 `__init__` 默认 `mesh_file_list=[]`，所以不需要额外处理。

### 2.6 `load_actor_from_urdf()` 的硬编码路径假设

```python
# simulation.py 第 142-143 行:
visual_file = os.path.join(file_dir, visual_file)
collision_file = os.path.join(file_dir, collision_file)
```

**问题**：`file_dir` = URDF 所在目录（即 `output/urdf/chair_0/`），视觉 mesh 在 `output/urdf/chair_0/mesh/mesh.obj`，所以路径拼接是：`output/urdf/chair_0/` + `mesh/mesh.obj` = 正确。

**结论**：`load_actor_from_urdf()` 的路径拼接是正确的，不需要额外处理。

### 2.7 `load_actor_from_urdf()` 对 gazebo 标签的硬依赖

```python
# simulation.py 第 144-145 行:
static_fric = root.find(".//collision/gazebo/mu1").text
dynamic_fric = root.find(".//collision/gazebo/mu2").text
```

**问题**：如果 URDF 中没有 gazebo 标签，`.text` 访问会抛出 AttributeError。

**解决**：EmbodiedGen 的 `generate_urdf()` 生成的 URDF 包含 gazebo 标签（来自 URDF_TEMPLATE），所以不会出问题。但如果 SAPIEN 验证时加载的是 `scene.urdf`（我们自己合并的），需要确保合并时保留 gazebo 标签。

### 2.8 `generate_urdf()` 对 `VERSION` 的依赖

```python
# urdf_convertor.py 第 337 行:
"version": VERSION,
```

`VERSION` 来自 `embodied_gen.utils.tags`，如果导入失败会报错。

**解决**：`embodied_gen.utils.tags` 是纯 Python 文件（只有一个 `VERSION = "0.0.1"`），无外部依赖，import 不会失败。

---

## 3. 冲突解决策略总表

| 冲突 | 影响 | 解决方案 | 解决位置 |
|------|------|---------|---------|
| nvdiffrast 未安装 → `render_asset3d` import 失败 | `from embodied_gen.validators.urdf_convertor import URDFGenerator` 时崩溃 | 用 `_MockModule` 替换 `embodied_gen.utils.process_media` | `stage61_urdf_generator.py` 顶部 |
| `GPT_CLIENT` 单例存在但不用 | 无功能影响 | 传自定义 QwenClient，忽略 GPT_CLIENT | `URDFGenerator(qwen_client, ...)` |
| `attr_dict["real_height"]` 缺失 | `generate_urdf()` 第 267 行 KeyError | 手动计算 `real_height` 后传入 | `stage61_urdf_generator.py` Step 2c→2d 之间 |
| mesh 无 .mtl/.png | `generate_urdf()` 会跳过 | 无需处理（mesh_file_list 默认为空） | — |
| scene.urdf 缺少 gazebo 标签 | `load_actor_from_urdf()` 第 144 行 AttributeError | 合并 URDF 时保留 gazebo 子元素 | `stage61_urdf_generator.py` Step 3 |
| 路径导入冲突 | 无法 import | 用绝对路径解析后 `sys.path.insert` | `stage61_urdf_generator.py` 顶部 |

---

## 4. 输出目录结构

```
output_v2/{scene_name}/
├── final_scene_stage5.glb                    # 输入
├── urdf/
│   ├── scene.urdf                            # 合并版 URDF
│   ├── chair_0/                              # 每个物体独立目录
│   │   ├── chair_0.urdf
│   │   ├── mesh/
│   │   │   ├── chair_0.obj
│   │   │   ├── chair_0_collision.obj
│   │   │   ├── material_0.png
│   │   │   └── material.mtl
│   │   └── urdf_renders/
│   │       ├── chair_0_0.png                 # 前视图
│   │       ├── chair_0_1.png                 # 右视图
│   │       ├── chair_0_2.png                 # 后视图
│   │       └── chair_0_3.png                 # 左视图
│   ├── bottle_0/
│   │   └── ...
│   └── mesh/                                 # 合并后的 mesh 文件
│       ├── chair_0_mesh.obj
│       ├── chair_0_collision.obj
│       ├── bottle_0_mesh.obj
│       ├── bottle_0_collision.obj
│       ├── material_0.png
│       └── material.mtl
├── stage61_report.json                       # 生成报告
└── stage61_validation.json                   # SAPIEN 验证报告 (--validate)
```

---

## 5. 实现要点

### 5.1 Open3D 渲染函数

```python
def open3d_render_mesh(mesh_path, output_dir, obj_name):
    """用 Open3D 离屏渲染 4 个视角，返回渲染图片路径列表"""
    import open3d as o3d
    import numpy as np
    from PIL import Image

    o3d_mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d_mesh.compute_vertex_normals()

    camera_angles = [
        ("0",   [0, 0, 0]),
        ("1",   [0, 90, 0]),
        ("2",   [0, 180, 0]),
        ("3",   [0, -90, 0]),
    ]

    render = o3d.visualization.rendering.OffscreenRenderer(512, 512)
    render.scene.set_background([0, 0, 0, 1])
    render.scene.remove_geometry("mesh")
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    render.scene.add_geometry("mesh", o3d_mesh, mat)

    paths = []
    os.makedirs(os.path.join(output_dir, "urdf_renders"), exist_ok=True)

    for idx, angle in camera_angles:
        # 设置相机位姿
        cam = o3d.geometry.TriangleMesh.create_box()  # 用 bbox 估算距离
        aabb = o3d_mesh.get_axis_aligned_bounding_box()
        diag = aabb.get_max_bound() - aabb.get_min_bound()
        dist = max(diag) * 1.5
        center = (aabb.get_max_bound() + abb.get_min_bound()) / 2

        # 旋转
        q = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # 设置相机位置 (front camera, 围绕 z 轴旋转)
        angle_rad = np.radians(angle[1])
        cam_pos = o3d.core.Vector3d(center[0] + dist, center[1], center[2])
        render.setup_camera(50, center=center.tolist(), eye=cam_pos.tolist(), up=[0, 0, 1])

        img = render.render_to_image()
        img_pil = Image.fromarray(np.array(img))
        out_path = os.path.join(output_dir, "urdf_renders", f"{obj_name}_{idx}.png")
        img_pil.save(out_path)
        paths.append(out_path)

    return paths
```

### 5.2 QwenClient 格式化函数

```python
def _format_response(self, raw):
    """将 Qwen 输出格式化为 GPT 兼容格式"""
    import re
    # 用正则从 Qwen 输出中提取各字段
    # 然后拼装成: "Category: xxx\nDescription: xxx\n..." 标准格式
    # 确保 parse_response() 可以按行索引解析
```

### 5.3 SAPIEN 验证

```python
def validate_with_sapien(urdf_dir, output_dir):
    """用 SAPIEN 无头模式验证 scene.urdf"""
    import sapien.core as sapien
    from embodied_gen.utils.simulation import load_actor_from_urdf

    scene = sapien.Scene()  # 无头
    scene.set_timestep(1/200)
    scene.add_ground(0)

    scene_urdf = os.path.join(urdf_dir, "scene.urdf")
    actors = []
    errors = []

    for obj_name in os.listdir(urdf_dir):
        if obj_name.endswith(".urdf") and obj_name != "scene.urdf":
            try:
                actor = load_actor_from_urdf(
                    scene,
                    os.path.join(urdf_dir, obj_name, f"{obj_name}.urdf"),
                    pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
                )
                actors.append(actor)
            except Exception as e:
                errors.append({"object": obj_name, "error": str(e)})

    # 运行仿真
    for _ in range(200):
        scene.step()

    # 检查稳定性
    results = []
    for i, actor in enumerate(actors):
        z = actor.get_pose().p[2]
        results.append({"actor": actor.name, "final_z": float(z), "stable": z > -0.01})

    with open(os.path.join(output_dir, "stage61_validation.json"), "w") as f:
        json.dump({"errors": errors, "results": results}, f, indent=2)

    return len(errors) == 0
```

---

## 6. 关键实现细节

### 6.1 `generate_urdf()` 不是 `__call__()`

关键区别：
- `URDFGenerator.__call__()` = render + VLM + parse + generate（完整流程，依赖 render_asset3d）
- `URDFGenerator.parse_response()` = 解析文本 → attr_dict
- `URDFGenerator.generate_urdf()` = attr_dict + mesh → URDF 文件

我们调用 **后两个**，不调用 `__call__()`。

### 6.2 `generate_urdf()` 修改了 mesh

`generate_urdf()` 内部会修改 mesh 的顶点（归一化+缩放），所以传入的 mesh 文件需要是**原始裸 OBJ**，不能是已经变换过的。

### 6.3 mesh 引用路径

`generate_urdf()` 输出的 URDF 中 mesh 引用是相对于 URDF 文件位置的：

```xml
<visual>
    <mesh filename="mesh/chair_0.obj"/>
</visual>
```

所以 `chair_0.urdf` 必须在 `chair_0/` 目录下，`mesh/` 子目录必须在 `chair_0/` 下。

### 6.4 SAPIEN 验证时 mesh 路径

`load_actor_from_urdf()` 解析 mesh 路径时用 `os.path.join(file_dir, visual_file)`，其中 `file_dir` = URDF 文件所在目录。所以验证时路径解析是正确的，不需要修改 URDF 内容。

### 6.5 合并 scene.urdf 时的 mesh 文件复制

合并 `scene.urdf` 时，需要把每个物体的 mesh 文件复制出来（避免路径混乱），并在 URDF 中更新 mesh 引用为相对于 `scene.urdf` 的路径。

---

## 7. 运行方式

```bash
# 独立运行（完整流程）
python src/stage61_urdf_generator.py \
    --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \
    --output_dir output_v2/2chair-bottle_vggt_omega \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B

# 独立运行（跳过 SAPIEN 验证）
python src/stage61_urdf_generator.py \
    --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \
    --output_dir output_v2/2chair-bottle_vggt_omega \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \
    --no_validate

# 嵌入 mainv2
python mainv2.py --input_video ./232.mp4 --enable_stage61
```

---

## 8. 测试计划

### 8.1 两个新建文件的位置

| 文件 | 完整绝对路径 |
|------|------------|
| `qwen_client.py` | `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/tools/qwen_client.py` |
| `stage61_urdf_generator.py` | `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/src/stage61_urdf_generator.py` |

### 8.2 测试数据准备

只需要一个文件：

```
output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb
```

---

### 8.3 Test 0: 依赖验证（环境准备，先跑）

| # | 测试项 | 命令 | 预期效果 |
|---|--------|------|---------|
| 0.1 | ReplicateAnyScene 基础依赖 | `python -c "import trimesh, numpy, scipy, cv2, PIL, imageio, matplotlib, networkx"` | 全部 OK |
| 0.2 | coacd 凸分解 | `python -c "import coacd; print('OK')"` | OK |
| 0.3 | sapien 仿真 | `python -c "import sapien; print('OK')"` | OK |
| 0.4 | transformers (Qwen) | `python -c "import transformers; print('OK')"` | OK |
| 0.5 | open3d 离屏渲染 | `python -c "import open3d as o3d; r=o3d.visualization.rendering.OffscreenRenderer(224,224); print('OK')"` | OK (EGL headless) |
| 0.6 | nvdiffrast 确认未安装 | `python -c "import nvdiffrast"` | **报错**（确认未安装，预期行为） |

---

### 8.4 Test 1: QwenClient 独立测试

**被测文件**: `tools/qwen_client.py`

**运行方式**: 独立脚本 `test_qwen_client.py`

| # | 测试内容 | 方法 | 预期效果 |
|---|---------|------|---------|
| 1.1 | 模型加载 | `QwenClient("/mnt/data/lza/models/Qwen3.5-9B")` | 无报错，`model` + `processor` 加载成功 |
| 1.2 | 单图推理 | `client.query("描述这个物体", ["test_render_0.png"])` | 返回文本，含 Category / Height / Weight / Friction 关键字 |
| 1.3 | 四图推理 | `client.query(prompt_template, 4张渲染图)` | 返回标准格式文本 |
| 1.4 | 格式兼容验证 | response → `URDFGenerator.parse_response(response)` | 不抛异常，返回 dict 含 `min_height` / `max_height` / `min_mass` / `max_mass` / `mu1` / `mu2` |
| 1.5 | 格式容错测试 | 给非标准 Qwen 输出走 `_format_response()` | 不崩溃，返回标准格式，`parse_response()` 能解析 |

**测试脚本**: `test_qwen_client.py`

```python
#!/usr/bin/env python3
"""test_qwen_client.py — 独立测试 QwenClient"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from tools.qwen_client import QwenClient

def test_load():
    print("Test 1.1: 加载模型...")
    client = QwenClient("/mnt/data/lza/models/Qwen3.5-9B")
    assert client.model is not None
    assert client.processor is not None
    print("  PASS")
    return client

def test_query(client, image_path):
    print("Test 1.2: 单图推理...")
    response = client.query("描述这个物体", [image_path])
    assert isinstance(response, str) and len(response) > 0
    print(f"  PASS (len={len(response)})")
    print(f"  内容预览: {response[:200]}")
    return response

def test_parse_compat(response):
    print("Test 1.4: parse_response 兼容性...")
    _EG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "EmbodiedGen"))
    sys.path.insert(0, _EG)
    sys.modules.setdefault(
        "embodied_gen.utils.process_media",
        type("M", (), {"render_asset3d": lambda *a, **k: []})()
    )
    from embodied_gen.validators.urdf_convertor import URDFGenerator
    parsed = URDFGenerator.parse_response(response)
    assert "min_height" in parsed
    assert "max_height" in parsed
    assert "mu1" in parsed
    assert "mu2" in parsed
    print(f"  PASS: {parsed}")

if __name__ == "__main__":
    client = test_load()
    response = test_query(client, "/path/to/test_render_0.png")
    test_parse_compat(response)
    print("\n全部通过")
```

---

### 8.5 Test 2: stage61_urdf_generator 完整流程测试

**被测文件**: `src/stage61_urdf_generator.py`

**运行方式**:

```bash
python src/stage61_urdf_generator.py \
    --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \
    --output_dir output_v2/2chair-bottle_vggt_omega \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \
    --no_validate
```

| # | 测试内容 | 验证方式 | 预期效果 |
|---|---------|---------|---------|
| 2.1 | EmbodiedGen import + MockModule | 程序启动不崩溃 | 成功导入 URDFGenerator、load_actor_from_urdf |
| 2.2 | GLB 拆解 | `print(objects_info)` | 输出列表，每个含 name/category/obj_path/position |
| 2.3 | Open3D 渲染 | `ls output/urdf/{obj}/urdf_renders/` | 每个物体 4 张 PNG，尺寸 512×512，黑底 |
| 2.4 | VLM 推理 | 检查 console 输出 | 返回格式化文本，parse_response() 不抛异常 |
| 2.5 | URDF 生成 | 检查目录结构 | 每个物体输出 `{obj}.urdf` + `mesh/` + `urdf_renders/` |
| 2.6 | 合并 scene.urdf | `ET.parse("output/urdf/scene.urdf")` | 根节点 name="scene"，含多个 `<link>` + `<joint type="fixed">` |
| 2.7 | 输出目录检查 | `tree output/urdf/` | 符合计划目录结构 |

**Test 2.5 验证点**:

```python
assert os.path.exists(f"output/urdf/chair_0/chair_0.urdf")
assert os.path.exists(f"output/urdf/chair_0/mesh/chair_0.obj")
assert os.path.exists(f"output/urdf/chair_0/mesh/chair_0_collision.obj")
assert os.path.exists(f"output/urdf/bottle_0/bottle_0.urdf")
```

**Test 2.6 验证点**:

```python
tree = ET.parse("output/urdf/scene.urdf")
assert tree.getroot().get("name") == "scene"
links = tree.findall(".//link")
joints = tree.findall(".//joint[@type='fixed']")
assert len(links) > 0
assert len(joints) > 0
```

---

### 8.6 Test 3: SAPIEN 仿真验证（最终测试）

**运行方式**:

```bash
python src/stage61_urdf_generator.py \
    --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \
    --output_dir output_v2/2chair-bottle_vggt_omega \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \
    --validate
```

| # | 测试内容 | 方法 | 预期效果 |
|---|---------|------|---------|
| 3.1 | URDF 加载 | `load_actor_from_urdf(scene, {obj}.urdf)` | 不报错，actor 创建成功 |
| 3.2 | 物理稳定性 | `scene.step()` × 200 步 | 物体 z 坐标变化 < 0.05m，不穿模 |
| 3.3 | 多物体共存 | 所有 actor 一起加载 | 物体间不穿模 |

**验证输出** (`stage61_validation.json`):

```json
{
    "errors": [],
    "results": [
        {"actor": "chair_0", "final_z": 0.01, "stable": true},
        {"actor": "bottle_0", "final_z": 0.01, "stable": true}
    ]
}
```

---

### 8.7 测试执行顺序

```
Test 0 (依赖) → Test 1 (QwenClient) → Test 2 (完整生成) → Test 3 (SAPIEN 验证)
```

**规则**: 每个 Test 通过后才进下一个，失败则定位问题，修好再继续。

---

## 9. 实现状态总览

### 9.1 已实现的文件

| 文件 | 状态 | 职责 |
|------|------|------|
| `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/src/stage61_urdf_generator.py` | ✅ 完成 | 主入口，5步流程：拆解→渲染→VLM推理→生成URDF→合并→验证 |
| `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/tools/qwen_client.py` | ✅ 完成 | Qwen3.5-9B VLM 适配器，兼容 GPTclient query() 接口 |

### 9.2 待实现

| 文件 | 状态 | 说明 |
|------|------|------|
| `mainv2.py` 修改 | ⏳ 待做 | 添加 `--enable_stage61` 参数，Stage 5 后调用 `run_stage61()` |

### 9.3 调用 EmbodiedGen 的代码（直接导入，未复制）

| EmbodiedGen 模块 | 导入方式 | 用途 |
|------------------|---------|------|
| `embodied_gen.validators.urdf_convertor.URDFGenerator` | `from embodied_gen.validators.urdf_convertor import URDFGenerator` | 解析 VLM 输出 + 生成 URDF + 凸分解 |
| `embodied_gen.utils.simulation.load_actor_from_urdf` | 延迟导入（仅在 `--validate` 时） | SAPIEN 验证 |

### 9.4 Mock 的依赖

| 缺失依赖 | 原模块 | Mock 方式 |
|----------|--------|-----------|
| `nvdiffrast` | `embodied_gen.utils.process_media.render_asset3d` | `_MockProcessMedia` → 返回空列表 |
| `openai` | `embodied_gen.utils.gpt_clients.GPTclient` | `_MockGPTClientsModule` → 存根 |

### 9.5 测试执行状态

| Test | 描述 | 状态 |
|------|------|------|
| Test 0 | 依赖验证 | ⏳ 待执行 |
| Test 1 | QwenClient 独立测试 | ⏳ 待执行 |
| Test 2 | 完整流程测试 (`--no_validate`) | ⏳ 待执行 |
| Test 3 | SAPIEN 仿真验证 (`--validate`) | ⏳ 待执行 |

### 9.6 实际运行命令

```bash
# 从 ReplicateAnyScene 根目录运行
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene

# 完整测试（跳过 SAPIEN 验证）
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python src/stage61_urdf_generator.py \
    --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \
    --output_dir output_v2/2chair-bottle_vggt_omega \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \
    --no_validate

# 含 SAPIEN 验证
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python src/stage61_urdf_generator.py \
    --glb output_v2/2chair-bottle_vggt_omega/final_scene_stage5.glb \
    --output_dir output_v2/2chair-bottle_vggt_omega \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \
    --validate
```

