# 环境配置说明

本文档记录 ReplicateAnyScene 的**实测通过环境**与**外部依赖**。
所有版本号均为实际运行验证过的，不是推测值。

---

## 1. 硬件 / 驱动（实测）

| 项 | 值 |
|---|---|
| GPU | 4 × NVIDIA RTX A6000 (48 GB) |
| 驱动 | 580.105.08 |
| CUDA Toolkit | 12.1（torch 内置） |
| 显存需求 | 单场景跑全流程约需 **22~26 GB**；SAM3D 阶段峰值最高 |

> 显存不足的典型报错：`torch.OutOfMemoryError: CUDA out of memory`。
> SAM3D（Stage 3）和 Qwen3.5-9B（Stage 3.7 / 5.1）是最吃显存的两处。
> 同一张卡不要并行跑两个场景 —— 会 OOM。

---

## 2. Python 环境

```bash
conda create -n ReplicateAnyScene python=3.11 -y
conda activate ReplicateAnyScene

# torch 必须 >=2.0（VGGT 依赖 scaled_dot_product_attention）
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

实测版本（关键项）：

```
python       3.11.0
torch        2.5.1+cu121
torchvision  0.20.1+cu121
numpy        1.26.4
scipy        1.17.1
trimesh      4.11.5
open3d       0.18.0
pyrender     0.1.45
opencv       4.13.0
pillow       9.5.0
transformers 5.6.2
sapien       3.0.3
pytorch3d    0.7.8
```

完整 469 包快照见 `requirements-freeze.txt`。

---

## 3. 外部依赖（不在 pip 里）

### 3.1 模型权重

| 权重 | 建议路径 | 来源 |
|---|---|---|
| VGGT-Omega | `/mnt/data/lza/models/vggt_omega/vggt_omega_1b_512.pt` | huggingface.co/facebook/VGGT-Omega |
| SAM3D | `./models/SAM3D/checkpoints/{ss_generator,slat_generator}.ckpt` | sam3d-objects 发布页 |
| Qwen3.5-9B（VLM） | `/mnt/data/lza/models/Qwen3.5-9B` | 本地已有 |
| CLIP ViT-B/32 | `/mnt/data/lza/models/clip-vit-base-patch32` | 跨类去重的 appearance feature |
| **EgoFound3R**（可选） | `EgoFound3R.pt` | huggingface.co/Frank-F2021/EgoFound3R |

> 权重文件均已被 `.gitignore` 排除（`*.pt / *.ckpt / *.safetensors` 等），仓库只存代码。

### 3.2 外部代码仓库（同级目录）

```
robot_world_ws/src/
├── ReplicateAnyScene/     ← 本仓库
├── vggt/                  ← VGGT（Stage 2 骨干）
├── sam3/                  ← SAM3（Stage 2 分割）
├── EgoFound3R/            ← 手部重建（可选，与本文档第 6 节相关）
└── ...
```

`mainv3.py` 通过 `PYTHONPATH` 引用它们：

```bash
export PYTHONPATH="$PWD:$PWD/vggt:$PWD/sam3"
```

---

## 4. 已知环境坑（都实际踩过）

### 4.1 pyrender 离屏渲染必须有 EGL

```python
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"   # 必须在 import pyrender 之前
```

否则报 `NoSuchDisplayException: Cannot connect to "None"`。

### 4.2 torch 版本对 VGGT 是硬要求

`torch < 2.0` 缺 `torch.nn.functional.scaled_dot_product_attention`，
VGGT 直接 `AttributeError`。hawor 自带环境是 torch 1.13，**不能**在那里跑 VGGT；
本仓库的 `hand_removal/` 必须在 ReplicateAnyScene 环境里跑。

### 4.3 `pynvml` 的 import 噪音

启动时会打印 `import pynvml  # type: ignore[import]`，无害，可忽略或
`pip install pynvml` 消除。

### 4.4 GPU 争用

```bash
# 选空闲最多的卡
G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | sort -t, -k2 -rn | head -1 | cut -d, -f1 | tr -d ' ')
CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python mainv3.py ...
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 能显著减少碎片化 OOM。

---

## 5. 最小验证

```bash
# 编译检查
python -m py_compile mainv3.py src/*.py tools/*.py

# 回归测试（17 个）
for t in test/test_*.py; do python "$t" >/dev/null 2>&1 && echo "PASS $t" || echo "FAIL $t"; done

# 单场景端到端（约 35~50 分钟）
python mainv3.py \
    --input_video assets/example/3chair-bottle.mp4 \
    --output_path output_v3/demo \
    --vggt_model vggt_omega \
    --enable_stage5 \
    --max_frames 80 \
    --scene_json scene_jsons/3chair-bottle.json \
    --disable_stage4
```

> `--scene_json` 用手写类别绕过 VLM 检测（VLM 在大场景会 hallucinate）；
> `--disable_stage4` 跳过 ICP（对 3cm 级小物体，其绝对距离阈值 0.075m
> 大于物体本身，迭代数学上无效，实测无收益）。

---

## 6. 可选：EgoFound3R 手部重建

EgoFound3R 与本管线**共用 VGGT-Omega 骨干**，输出世界坐标下的：

- 手 root / joints / vertices
- `contact`（逐点手-物接触）
- `visibility`（逐点可见性，可替代手遮挡 mask）
- 米制 MANO 手（`--upsample-mano`）

理论上的四个接口：

| 接口 | 能替代的现有启发式 |
|---|---|
| 手部 3D | `pick_place_detector.py` 的位移/重叠率判据 |
| `contact` | 「哪个物体被操作」的几何猜测 |
| `visibility` | `hand_removal/`（hawor）的手部 mask |
| 米制手 | `tools/rescale_meshes_by_pointcloud.py` 的尺度标定 |

**当前状态**：`EgoFound3R.pt` 未下载，尚未接入。

---

## 7. 目录约定（均已在 `.gitignore` 排除）

```
output_v3/     每个场景一个子目录（color/ depth/ extrinsics/ *.glb / *.png / *.pkl）
output_v2/     旧管线产物
assets/        输入视频与关键帧
models/        本地权重
outputs/       hand_removal 等中间产物
```

仓库只包含**代码**（约 200 MB 工作区里的源码部分），产物与权重不入库。
