# Stage 6 设计方案：Material-aware Physics Annotation（材质感知的物理参数标注）

---

## 1. 目标

为 `mainv2.py` 流水线新增 **Stage 6**，将核心定位从"物理参数标注"升级为 **"材质感知的物理标注"**。

VLM 只负责完成 **Visual Analysis → Material Recognition**（视觉分析 → 材质识别），输出每个物体的材质类型（如 hardwood, glass, hard_plastic, steel）。物理参数的映射由独立的 **Material Database**（`materials_db.json`）统一完成。

架构对比：

```
❌ 旧方案（Parameter-driven）:
  Visual Analysis → VLM 直接输出 density/friction/restitution
  → VLM 自由生成数值，不稳定、不可控

✅ 新方案（Material-driven）:
  Visual Analysis → Material Recognition (VLM 输出材质名)
  → Material Database 映射为具体物理参数
  → 扩展新材质/新平台只需改数据库，不重跑 VLM
```

---

## 2. 背景与上下文

### 2.1 当前 SAPIEN 物理参数设置方式

参考 `04_physics_simulation.py` 中 `load_glb_with_physics()` 函数：

```python
# 全局密度（硬编码，所有物体统一）
OBJECT_DENSITY = 1000.0

# 物理材质（硬编码）
phys_material = scene.create_physical_material(
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.3,
)
```

**问题**：所有物体统一使用 density=1000、friction=0.5、restitution=0.3，但实际场景物体差异巨大：
- 塑料瓶 vs 木桌 → 密度完全不同
- 玻璃杯 vs 橡胶球 → 摩擦和弹性完全不同

### 2.2 现有流水线输出

Stage 5 完成后，`mainv2.py` 输出：
- `final_scene_stage5.glb`（或 `final_scene_stage4_5.glb`）
- `final_relations.json`（物体类别 + 关系）
- `optimal_frames/`（每个实例的最优视角帧图片）

这些资源可被 Stage 6 消费。

---

## 3. Stage 6 设计

### 3.1 整体流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                        Stage 6 流水线                            │
│                                                                 │
│  input: final_relations.json + optimal_frames/ + glb           │
│                                                                 │
│  Step 1: 构建场景图（ID 标注图，与 Stage 5.1 复用）             │
│  Step 2: VLM 推断材质（仅输出材质名 + 简短推理）                │
│  Step 3: 查材料数据库 → 映射为物理参数                            │
│  Step 4: 组装输出文件                                            │
│                                                                 │
│  output: materials.json + materials_debug.json + materials_db.json │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 VLM 推理方式

**只使用场景图方式（scene_graph），与 Stage 5.1 完全一致**：
- 1 次 VLM 调用，一次性推断所有物体的材质
- 上下文完整，推理一致性好
- 仅 1 次调用，token 友好（约 300 tokens prompt，1024 max_new_tokens）
- 不提供 `--stage6_method` 切换

### 3.3 VLM Prompt（材质识别专用）

Prompt 极简设计，只要求 VLM 识别材质，不做参数映射。参考 PhysGS（分割掩码+隔离图像）、CCoT（链式推理）：

**输入**：ID 标注图（每个物体用不同颜色填充）+ 物体类别列表

```
You are a material recognition expert. Identify the PRIMARY material of each
colored object region in this scene image.

## Supported Materials
hardwood, softwood, hard_plastic, soft_plastic, glass, ceramic,
steel, aluminum, copper, rubber, fabric, paper, foam, leather,
concrete, stone, marble, ice, wood (generic), plastic (generic),
metal (generic)

## Instructions

For each colored region in the image:
1. Look at the object's surface appearance: color, reflectance
   (glossy/matte/metallic/translucent), texture (smooth/rough/grainy)
2. Identify its PRIMARY material from the list above
3. Be specific when possible: "hardwood" instead of just "wood"
4. If uncertain or looks synthetic, use generic names (plastic, metal, wood)

## Objects in This Scene
{object_list_str}

## Important Warning
⚠ The category labels may be INACCURATE (e.g., "toy" instead of the real object).
ALWAYS trust your visual analysis of the actual object appearance over the text labels.

## Output Format

Respond with a JSON object ONLY (no markdown, no explanations):
{{
  "materials": {{
    "<category_name>": "{{material_name}}",
    ...
  }}
}}
```

**Prompt 分析**：
- 长度：约 300 tokens（不含 object list）
- 支持的材质列表：17 种，覆盖常见室内外场景
- 输出结构：极简 JSON，只有材质名映射，不含推理文本

### 3.4 材料参数映射表（Material Mapping Table）

**文件**：`tools/materials_db.json`（人工整理，非权威数据库）

⚠ **重要说明**：`materials_db.json` 是**人工整理的参考值表**，不是某个权威来源的数据库。表中的密度、摩擦系数、恢复系数均来自常见工程手册/经验值，是合理的起点，但并非精确值。SAPIEN 仿真器中的物理参数本就不是真实世界的精确映射，而是"看起来合理的近似值"——所以这组参数只要在同一量级上正确即可。

**维护方式**：随代码分发，更新时直接编辑 JSON 文件。可参考网上公开的 Engineering Toolbox、MatWeb 等资料来校准。

```json
{
  "materials": {
    "hardwood": {
      "description": "硬木家具材料",
      "sapien": {
        "density": 750,
        "static_friction": 0.60,
        "dynamic_friction": 0.50,
        "restitution": 0.20
      }
    },
    "softwood": {
      "description": "松木等软木材料",
      "sapien": {
        "density": 550,
        "static_friction": 0.50,
        "dynamic_friction": 0.40,
        "restitution": 0.20
      }
    },
    "hard_plastic": {
      "description": "硬质塑料（瓶、盒等）",
      "sapien": {
        "density": 1375,
        "static_friction": 0.40,
        "dynamic_friction": 0.30,
        "restitution": 0.50
      }
    },
    "soft_plastic": {
      "description": "软质塑料（玩具、软包）",
      "sapien": {
        "density": 935,
        "static_friction": 0.30,
        "dynamic_friction": 0.20,
        "restitution": 0.40
      }
    },
    "glass": {
      "description": "玻璃（杯、窗、容器）",
      "sapien": {
        "density": 2500,
        "static_friction": 0.40,
        "dynamic_friction": 0.40,
        "restitution": 0.60
      }
    },
    "ceramic": {
      "description": "陶瓷（碗、花瓶、瓷砖）",
      "sapien": {
        "density": 2250,
        "static_friction": 0.50,
        "dynamic_friction": 0.40,
        "restitution": 0.30
      }
    },
    "steel": {
      "description": "钢铁（工具、结构件）",
      "sapien": {
        "density": 7900,
        "static_friction": 0.70,
        "dynamic_friction": 0.60,
        "restitution": 0.20
      }
    },
    "aluminum": {
      "description": "铝合金（灯架、电子设备外壳）",
      "sapien": {
        "density": 2700,
        "static_friction": 0.60,
        "dynamic_friction": 0.50,
        "restitution": 0.35
      }
    },
    "rubber": {
      "description": "橡胶（垫、球、玩具）",
      "sapien": {
        "density": 1000,
        "static_friction": 0.90,
        "dynamic_friction": 0.80,
        "restitution": 0.70
      }
    },
    "fabric": {
      "description": "布料/纺织品（靠垫、窗帘、衣物）",
      "sapien": {
        "density": 200,
        "static_friction": 0.60,
        "dynamic_friction": 0.50,
        "restitution": 0.10
      }
    },
    "paper": {
      "description": "纸张/纸板（书本、盒子、杂志）",
      "sapien": {
        "density": 750,
        "static_friction": 0.50,
        "dynamic_friction": 0.40,
        "restitution": 0.10
      }
    },
    "foam": {
      "description": "泡沫/海绵（坐垫、玩具、包装）",
      "sapien": {
        "density": 35,
        "static_friction": 0.40,
        "dynamic_friction": 0.30,
        "restitution": 0.20
      }
    },
    "leather": {
      "description": "皮革（沙发、包、座椅）",
      "sapien": {
        "density": 950,
        "static_friction": 0.60,
        "dynamic_friction": 0.50,
        "restitution": 0.30
      }
    },
    "concrete": {
      "description": "混凝土（地面、柱子）",
      "sapien": {
        "density": 2400,
        "static_friction": 0.60,
        "dynamic_friction": 0.50,
        "restitution": 0.10
      }
    },
    "stone": {
      "description": "石头/石材（桌面、装饰）",
      "sapien": {
        "density": 2600,
        "static_friction": 0.60,
        "dynamic_friction": 0.50,
        "restitution": 0.10
      }
    },
    "marble": {
      "description": "大理石（台面、雕塑）",
      "sapien": {
        "density": 2700,
        "static_friction": 0.40,
        "dynamic_friction": 0.30,
        "restitution": 0.15
      }
    },
    "ice": {
      "description": "冰（特殊场景）",
      "sapien": {
        "density": 917,
        "static_friction": 0.05,
        "dynamic_friction": 0.03,
        "restitution": 0.30
      }
    },
    "wood": {
      "description": "通用木材（未区分软硬）",
      "sapien": {
        "density": 650,
        "static_friction": 0.55,
        "dynamic_friction": 0.45,
        "restitution": 0.20
      }
    },
    "plastic": {
      "description": "通用塑料（未区分软硬）",
      "sapien": {
        "density": 1150,
        "static_friction": 0.35,
        "dynamic_friction": 0.25,
        "restitution": 0.45
      }
    },
    "metal": {
      "description": "通用金属（未区分种类）",
      "sapien": {
        "density": 7000,
        "static_friction": 0.65,
        "dynamic_friction": 0.55,
        "restitution": 0.25
      }
    }
  },
  "default": {
    "description": "默认材质（VLM 未识别或未匹配的物体）",
    "sapien": {
      "density": 1000,
      "static_friction": 0.50,
      "dynamic_friction": 0.50,
      "restitution": 0.30
    }
  }
}
```

### 3.5 VLM 输出解析与参数映射

```python
# Stage 6 核心逻辑
def run_stage6(output_path, ...):
    # Step 1: 构建场景图 ID 标注图
    scene_image_path = build_id_labeled_image(...)

    # Step 2: VLM 推断材质
    material_map = query_vlm_materials(vlm_checkpoint, scene_image_path, object_list)
    # 输出格式: {"materials": {"bottle": "hard_plastic", "table": "hardwood", ...}}

    # Step 3: 查材料数据库映射为物理参数
    materials_db = load_materials_db()
    per_category = {}
    for cat_name, material_name in material_map["materials"].items():
        if material_name in materials_db["materials"]:
            params = materials_db["materials"][material_name]["sapien"].copy()
            params["material"] = material_name
        else:
            params = materials_db["default"]["sapien"].copy()
            params["material"] = "default"
        per_category[cat_name] = params

    # Step 4: 输出文件
    save_materials_json(per_category, output_path)
    save_materials_debug(material_map, output_path)
```

### 3.6 输出文件

**① `materials.json`**（仿真器直接消费，极简格式）

```json
{
  "bottle": {
    "material": "hard_plastic",
    "density": 1375,
    "static_friction": 0.40,
    "dynamic_friction": 0.30,
    "restitution": 0.50
  },
  "bowl": {
    "material": "ceramic",
    "density": 2250,
    "static_friction": 0.50,
    "dynamic_friction": 0.40,
    "restitution": 0.30
  },
  "table": {
    "material": "hardwood",
    "density": 750,
    "static_friction": 0.60,
    "dynamic_friction": 0.50,
    "restitution": 0.20
  }
}
```

**② `materials_debug.json`**（调试用，包含 VLM 推理链路）

```json
{
  "material_recognition": {
    "materials": {
      "bottle": "hard_plastic",
      "bowl": "ceramic",
      "table": "hardwood"
    }
  },
  "material_db_version": "v1.0",
  "mapping_log": [
    {"category": "bottle", "recognized_material": "hard_plastic", "mapped_params": {...}},
    {"category": "bowl", "recognized_material": "ceramic", "mapped_params": {...}},
    {"category": "table", "recognized_material": "hardwood", "mapped_params": {...}}
  ],
  "unrecognized": []
}
```

**③ `materials_db.json`**（独立维护，随代码分发）
- 位置：`tools/materials_db.json`
- 更新材质只需编辑此文件，无需改任何 Python 代码
- 可扩展至其他仿真器：在 JSON 中增加 `"isaac_sim"`, `"mujoco"`, `"pybullet"` 等字段

### 3.7 与 SAPIEN 的集成方式

#### 方案：修改 `04_physics_simulation.py` 支持外部材质文件

在 `load_glb_with_physics()` 中新增可选参数 `materials_path`：

```python
def load_glb_with_physics(glb_path, transform_params_path, scene,
                           logger=None, fast_collision=False,
                           materials_path=None):
    if materials_path and os.path.exists(materials_path):
        with open(materials_path) as f:
            materials = json.load(f)

    for geom_idx, (geom_name, geom) in enumerate(trimesh_scene.geometry.items()):
        cat_name = geom_name.split('_')[0]  # "bottle_0" → "bottle"

        if materials and cat_name in materials:
            m = materials[cat_name]
        else:
            m = {
                "density": OBJECT_DENSITY,
                "static_friction": 0.5,
                "dynamic_friction": 0.5,
                "restitution": 0.3,
            }

        builder.use_density = True
        builder.set_density(m["density"])

        phys_material = scene.create_physical_material(
            static_friction=m["static_friction"],
            dynamic_friction=m["dynamic_friction"],
            restitution=m["restitution"],
        )
```

---

## 4. mainv2.py 修改点

### 4.1 新增函数

```python
def run_stage6(output_path, all_instances, categories_and_relations,
               vlm_checkpoint, optimal_frames_dir=None):
    """
    Stage 6: Material-aware Physics Annotation

    流程:
      1. 构建场景图（ID 标注图，复用 Stage 5.1 逻辑）
      2. 调用 VLM 推断每个物体的材质类型（仅输出材质名）
      3. 查材料数据库 → 映射为物理参数
      4. 输出 materials.json + materials_debug.json

    参数:
        output_path: 输出目录
        all_instances: 3D资产实例
        categories_and_relations: 类别-关系字典
        vlm_checkpoint: VLM 模型路径
        optimal_frames_dir: 最优视角帧目录
    返回:
        materials_dict (dict) — 与 materials.json 同格式
    """
```

### 4.2 命令行参数新增

```python
parser.add_argument("--enable_stage6", action="store_true",
                    help="启用Stage 6材质感知物理标注 (默认关闭)")
```

### 4.3 main() 函数新增调用

在 Stage 5 之后、最终保存之前：

```python
# ── Stage 6: 材质感知的物理标注 (可选) ──
materials = None
if args.enable_stage6:
    _t6 = time.time()
    from tools.material_annotator import run_stage6
    materials = run_stage6(
        args.output_path, all_instances, categories_and_relations,
        vlm_checkpoint,
        optimal_frames_dir=os.path.join(args.output_path, 'optimal_frames'),
    )
    _stage6_time = time.time() - _t6
```

### 4.4 输出文件汇总

Stage 6 完成后，output 目录新增：
```
output_v2/{scene}_{model}/
  ├── materials.json                 # 材质+物理参数（仿真器消费）
  ├── materials_debug.json           # VLM推理链路+映射日志（调试用）
  └── ...（其他现有文件）
```

---

## 5. 工具实现结构

新增文件 `tools/material_annotator.py`：

```python
def build_id_labeled_image(optimal_frames_dir, categories_and_relations,
                           all_instances, output_path):
    """生成实例 ID 标注图（与 Stage 5.1 完全一致，可复用）"""

def query_vlm_materials(vlm_checkpoint, scene_image_path, object_list):
    """调用 VLM 推断场景图中所有物体的材质类型"""

def load_materials_db():
    """加载 tools/materials_db.json"""

def map_materials_to_params(material_map, materials_db):
    """将 VLM 输出的材质名映射为物理参数"""

def run_stage6(output_path, all_instances, categories_and_relations,
               vlm_checkpoint, optimal_frames_dir=None):
    """Stage 6 主入口"""
```

新增文件 `tools/materials_db.json`：
```json
{
  "materials": { "hardwood": {...}, "glass": {...}, ... },
  "default": { "sapien": {...} }
}
```

---

## 6. SAPIEN 集成改造

### 6.1 `04_physics_simulation.py` 修改

**最小改动方案**：新增命令行参数 `--materials`，在 `load_glb_with_physics()` 中读取：

```python
# main() 新增参数
parser.add_argument("--materials", type=str, default=None,
                    help="Stage6 输出的 materials.json 路径")

# load_glb_with_physics() 修改签名
def load_glb_with_physics(glb_path, transform_params_path, scene,
                           logger=None, fast_collision=False,
                           materials=None):
    for geom_idx, (geom_name, geom) in enumerate(...):
        cat_name = geom_name.split('_')[0]
        m = materials.get(cat_name, {"density": 1000, "static_friction": 0.5,
                                       "dynamic_friction": 0.5, "restitution": 0.3})
        builder.use_density = True
        builder.set_density(m["density"])
        phys_material = scene.create_physical_material(
            static_friction=m["static_friction"],
            dynamic_friction=m["dynamic_friction"],
            restitution=m["restitution"],
        )
```

### 6.2 命令行调用示例

```bash
# mainv2.py 启用 Stage 6
python mainv2.py --input_video ./test.mp4 --enable_stage6

# 04_physics_simulation.py 使用 Stage 6 输出
python 04_physics_simulation.py \
    --hawor-dir ./output/hawor/ \
    --ras-dir ./output/ras/ \
    --materials ./output/ras/materials.json
```

---

## 7. 实现优先级

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | `materials_db.json` 建立 | 17种材质 + SAPIEN 参数 + default 兜底 |
| P0 | `query_vlm_materials()` + 场景图方式 | VLM 材质识别，仅 1 次调用 |
| P0 | `map_materials_to_params()` | 材质名 → 物理参数映射 |
| P0 | `mainv2.py` 集成 Stage 6 调用 | 命令行参数 + 主流程接入 |
| P0 | `materials.json` + `materials_debug.json` 输出 | 两文件分离：仿真用 vs 调试用 |
| P1 | `04_physics_simulation.py` 集成读取 | 读取 materials.json |
| P2 | 扩展至 Isaac Sim / MuJoCo / PyBullet | 在 materials_db.json 中新增字段 |

---

## 8. 扩展性说明

### 8.1 添加新材质

只需编辑 `tools/materials_db.json`，新增一个条目即可。无需改动 VLM 代码：

```json
"silicone": {
  "description": "硅胶",
  "sapien": {
    "density": 1100,
    "static_friction": 0.80,
    "dynamic_friction": 0.70,
    "restitution": 0.40
  }
}
```

### 8.2 添加高级物理属性（杨氏模量、泊松比）

在 `materials_db.json` 中新增字段：

```json
"hardwood": {
  "description": "硬木家具材料",
  "sapien": { ... },
  "elastic_properties": {
    "youngs_modulus": 10000000000,
    "poissons_ratio": 0.35
  }
}
```

### 8.3 迁移至新仿真平台

在 `materials_db.json` 中增加新字段，各仿真器各自读取对应的子字段：

```json
"hardwood": {
  "sapien": { ... },
  "isaac_sim": { ... },
  "mujoco": { ... },
  "pybullet": { ... }
}
```

### 8.4 论文表达优势

| 旧方案（Parameter-driven） | 新方案（Material-driven） |
|---------------------------|--------------------------|
| VLM 直接输出物理参数 | VLM 输出材质类型 |
| 不稳定性高（自由生成数值） | 稳定性高（离散类别） |
| 扩展新平台需改 VLM prompt | 扩展新平台只需改 JSON |
| 参数无解释 | 材质有语义可解释 |
| 不符合真实建模流程 | 符合真实建模流程 |

---

## 9. 风险提示

1. **VLM 材质识别不确定性**：VLM 可能将材质识别错误，但离散类别（20种）比连续数值更稳定
2. **材质粒度选择**：当前 17 种材质是合理起点，可后续扩展
3. **GLB 命名映射**：GLB node 名称 → 类别的映射关系需要稳定（当前用 `{category}_{i}` 格式）
4. **SAPIEN density 范围**：SAPIEN 对极端密度值（>50000）可能不稳定，materials_db.json 已限制在合理范围