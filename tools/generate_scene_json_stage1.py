"""
ReplicateAnyScene Stage 2: VGGT引导的智能物体发现与3D去重

核心流程（参考SimRecon infer_scene_graph.py）:
  第一次VLM调用: 物体检测（仅名称+位置）
  → 去重合并
  → SAM分割floor/wall
  → 第二次VLM调用: 关系判断（基于物体列表+floor/wall信息，4种关系）

  Step 0: VGGT 3D场景重建 → 获取点云和相机位姿
  Step 1: SimRecon 3D空间覆盖采样 → 选择关键帧
  Step 2: 提取关键帧图像
  Step 3: 第一次VLM调用 → 逐帧物体检测（仅名称+位置，不判断关系）
  Step 4: 射线投射 → 将VLM像素位置映射到3D空间
  Step 5: 名称+3D位置联合去重
  Step 6: SAM分割floor和wall
  Step 7: 第二次VLM调用 → 关系判断（独立prompt，4种关系+物理常识后处理）
  Step 8: 输出场景JSON
"""
import argparse
import json
import math
import os
import re
import sys
import time
import subprocess
import shutil
import tempfile
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from PIL import Image
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    from transformers import AutoProcessor

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from src.models import load_vggt_model, unload_model
from src.utils import load_video_frames
from src.vggt_predict import vggt_predict
from src.sg_deduplication import UnionFind
from src.vlm_logger import vlm_inference, set_vlm_log_dir

# ============================================================
# 常量定义
# ============================================================

# 4种有效空间关系
VALID_SPATIAL_RELATIONSHIPS = [
    "supported by floor",        # 被地面支撑（直接放在地面上）
    "supported by other objects", # 被其他物体支撑（放在家具上等）
    "attached to wall",          # 附着在墙上（挂在/固定在墙面）
    "embedded in wall"           # 嵌入墙体（门、窗、插座等）
]

# 需要过滤的类别（属于floor/wall本身，不是独立物体）
FILTER_CATEGORIES = {
    'floor', 'wall', 'ground', 'ceiling', 'floor_area', 'wall_section',
    'flooring', 'wall_surface', 'room', 'space',
    # VLM 照抄提示词占位符 / 泛指的产物。提示词已明确禁止, 这里是兜底 ——
    # 实测 2026-10-19 weixin_195833_224 场景 VLM 仍返回了 "object",
    # 占掉一个类别名额并把 Stage1 输出污染成 4 类 (真实只有 3 类)。
    'object', 'objects', 'thing', 'things', 'item', 'items', 'stuff',
    'noun', 'object1', 'object2', 'entity',
}

# 必须由地面支撑的类别（不可能挂在墙上）
MUST_BE_FLOOR_SUPPORTED = {
    'cabinet', 'wardrobe', 'closet', 'dresser', 'bookshelf', 'bookcase',
    'shelf', 'shelving', 'desk', 'table', 'chair', 'sofa', 'couch',
    'bed', 'mattress', 'nightstand', 'bench', 'stool', 'armchair',
    'refrigerator', 'fridge', 'washing_machine', 'dryer', 'oven', 'stove',
    'tv_stand', 'entertainment_center', 'sideboard', 'buffet',
    'filing_cabinet', 'storage_cabinet', 'kitchen_cabinet',
    'plant', 'potted_plant', 'trash_can', 'garbage_can', 'bin',
    'box', 'luggage', 'suitcase', 'bag', 'backpack',
    'carpet', 'rug', 'mat', 'ottoman', 'footstool'
}

# 必须附着/嵌入墙面的类别（不可能放在地面上）
MUST_BE_WALL_ATTACHED = {
    'picture', 'painting', 'photo', 'photo_frame', 'frame',
    'mirror', 'clock', 'wall_clock', 'poster', 'art',
    'whiteboard', 'blackboard', 'bulletin_board',
    'light_switch', 'outlet', 'socket',
    'curtain', 'drape', 'blind', 'shade',
    'window', 'door', 'vent', 'air_vent'
}

# 去重参数
CENTROID_DIST_THRESHOLD = 0.15  # 3D质心距离阈值（米）
RAY_CAST_TOP_K = 5              # 射线投射取最近K个点

FLOOR_OVERLAP_THRESHOLD = 0.3   # SAM floor重叠检测阈值：物体底部30%与floor mask重叠则判定为在地板上

# ============================================================
# 同义词映射表
# ============================================================

SYNONYM_MAP = {
    "ground": "floor",
    "flooring": "floor",
    "carpet": "floor",
    "rug": "floor",
    "walling": "wall",
    "walls": "wall",
    "ceiling": "ceiling",
    "ceilings": "ceiling",
    "cardboard box": "box",
    "cardboard": "box",
    "carton": "box",
    "mug": "cup",
    "glass": "cup",
    "tumbler": "cup",
    "sneaker": "shoe",
    "slipper": "shoe",
    "footwear": "shoe",
    "sofa": "couch",
    "settee": "couch",
    "desk": "table",
    "dining table": "table",
    "laptop computer": "laptop",
    "notebook": "laptop",
    "cell phone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "television": "tv",
    "monitor": "tv",
    "display": "tv",
    "potted plant": "plant",
    "flowerpot": "plant",
    "remote control": "remote",
    "pillow case": "pillow",
}


# ============================================================
# 第一次VLM调用: 物体检测提示词（仅名称，不判断位置和关系）
# ============================================================

# 检测提示词。
#
# 关键约束 (改之前务必先读): "name" 会被直接当作 SAM3 的分割文本提示
# (mainv3.py → segment_and_track(category=name) → add_prompt(text=name))。
# 因此 name 必须是"简单名词", 不能写成 "black chair on the left" 这种描述性
# 短语 —— 文本提示越具体, SAM3 越容易切不出东西。想表达"有几个/在哪", 只能
# 用附加字段, 不能污染 name。
#
# 相对旧版的改动及原因:
#   ① 旧版有 "List each object type once per frame", 完全不给实例数量信息,
#      SAM3 只能靠自身密集检测硬猜。实测 basic_pick_place_141 实拍约 8~11 个
#      棋子, 最终只产出 9 个实例, log 里无从判断是"VLM 少数"还是"SAM3 漏切"。
#      现在要求输出 count, 让这两件事第一次可分离诊断。
#   ② 旧版 Ignore 列表只有 hands/body parts/walls/floors/ceilings, 实测噪声里
#      出现 wall outlet×8、table, 都不在列表 → 靠下游 BODY_PART_CATEGORIES
#      硬编码兜底。改为把背景结构显式列出。
#   ③ 旧版没有合并规则, 盆栽+盆、杯+液体常被拆成两个实例。补一条, 并明确
#      "容器内的物体是独立实例", 避免反向误合并。
VLM_DETECT_PROMPT = """List all visible objects in this image. Output JSON only.

{{"objects": [{{"name": "<noun>", "count": 1}}, {{"name": "<noun>", "count": 2}}]}}

Rules:
- "name" MUST be a simple singular English noun in lowercase, no articles
  (e.g. "bottle", "chess piece"). It is used as a segmentation text prompt,
  so NEVER use descriptive phrases like "white cup on the left".
- Describe what you actually see; do NOT copy the placeholder "<noun>" above.
  Never output a name that is a generic word such as "object", "thing", "item".
- "count" = how many separate physical instances of this type are visible.
  Count carefully: identical-looking small objects must each be counted
  (e.g. 8 chess pieces -> {{"name": "chess piece", "count": 8}}).
- Report each object TYPE once, together with its instance count.
- For thin tall cylinders prefer "bottle" over "cup" (cup is short and wide).
- Merge into one entry only when physically inseparable: a plant with its pot,
  a cup with the liquid inside it.
  Objects stored inside an open container are SEPARATE entries.
- Ignore: hands, body parts, walls, floors, ceilings, wall outlets, sockets,
  light switches.
- Table / desk / counter surfaces ARE objects — report them as "table",
  "desk" or "counter". They are the support surface everything else rests on,
  so they must NOT be treated as background.
  (实测 weixin_195833_224: 旧版把白桌面当背景丢掉, 导致唯一的物体悬空在
  空场景里, 只能贴到虚拟 z=0。)

Output JSON:"""


# ============================================================
# API 模式: 用 opencode 的视觉模型替代本地 Qwen3.5-9B
# ============================================================
# 实测对比 (weixin_195833_224, 手持小物体 + 纯白背景):
#   本地 Qwen3.5-9B + 旧 prompt → earbud case / eraser / object / stone
#                                 (3 个是从白背景幻觉出来的, object 是占位符污染)
#   6.8 + 新 prompt             → {"objects":[{"name":"button","count":1}]}
#                                 (干净, 简单名词, 带数量)
# 强视觉模型对"大面积纯色背景 + 小物体"这类场景明显更稳, 不会凭空造物体。
API_VLM_MODEL = os.environ.get("RAS_STAGE1_VLM_MODEL",
                               "sensenova/sensenova-6.8-flash-lite")
USE_API_VLM = os.environ.get("RAS_STAGE1_USE_API", "0") == "1"
OPENCODE_BIN = shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse_batch_detect(raw, n):
    """把批量检测的 VLM 输出拆成逐帧的 JSON 字符串列表。

    VLM 返回 {"0": {"objects": [...]}, "1": {...}, ...}; 这里按 key 取出并
    重新序列化成与逐帧模式相同的字符串, 让下游 _extract_json_from_text 无感知。
    任何结构不符都返回 None, 由调用方退回逐帧。
    """
    if not raw:
        return None
    js = _extract_json_from_text(raw)
    if js is None:
        return None
    out = []
    for i in range(n):
        v = js.get(str(i))
        if isinstance(v, dict) and "objects" in v:
            out.append(json.dumps(v, ensure_ascii=False))
        elif isinstance(v, list):
            out.append(json.dumps({"objects": v}, ensure_ascii=False))
        else:
            return None
    return out


def vlm_inference_api(image_path, prompt, model=None, timeout=420):
    """用 opencode 调视觉模型做一次检测。

    关键: cwd 必须是项目根、图片路径必须是项目内路径。
    实测把图拷到 /tmp 并用它当 cwd 时, opencode 报
        permission requested: external_directory (/tmp/*); auto-rejecting
    然后静默返回空 —— 表现为"检测失败", 但看不出原因。

    Returns:
        模型原始输出文本; 失败返回空串 (由调用方回退到本地模型)。
    """
    model = model or API_VLM_MODEL
    if not os.path.isfile(image_path) or not os.path.exists(OPENCODE_BIN):
        return ""
    cfg = {"$schema": "https://opencode.ai/config.json",
           "permission": {"edit": "deny", "bash": {"*": "allow"},
                          "webfetch": "deny"}}
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
            cfg_path = f.name
        env = dict(os.environ)
        env["OPENCODE_CONFIG"] = cfg_path
        rel = os.path.relpath(os.path.abspath(image_path), PROJECT_ROOT)
        cmd = [OPENCODE_BIN, "run", "--model", model,
               "Read the image at %s, then follow these instructions.\n\n%s"
               % (rel, prompt)]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                              capture_output=True, text=True, timeout=timeout)
        return (proc.stdout or "").strip()
    except Exception:
        return ""
    finally:
        try:
            os.unlink(cfg_path)
        except Exception:
            pass


# ============================================================
# 第二次VLM调用: 关系判断提示词构建
# ============================================================

def _build_relationship_prompt(object_names, has_floor, has_wall):
    """
    构建第二次VLM调用的关系判断提示词

    只使用4种关系，不涉及树状结构:
      - supported by floor: 被地面支撑
      - supported by other objects: 被其他物体支撑
      - attached to wall: 附着在墙上
      - embedded in wall: 嵌入墙体

    参数:
        object_names: 去重后的物体名称列表
        has_floor: SAM是否检测到地面
        has_wall: SAM是否检测到墙面
    """
    objects_str = ", ".join(object_names)
    num_objects = len(object_names)

    floor_info = "FLOOR is VISIBLE in this scene." if has_floor else "FLOOR is NOT visible in this scene."
    wall_info = "WALL is VISIBLE in this scene." if has_wall else "WALL is NOT visible in this scene."

    # 兜底默认关系必须与"地面是否可见"一致。
    # 原提示词无条件是 'If unsure, default to: "supported by floor"', 与上面刚写的
    # "FLOOR is NOT visible" 自相矛盾; 俯视桌面场景 SAM 检测不到地面 (has_floor=False),
    # VLM 对"靠什么支撑"确实没把握 → 按兜底把全部物体标成 floor。
    # 实测后果 (basic_pick_place_111/14/46/47/64 五个场景): 100% 物体都是
    # "supported by floor" → mainv3 里 has_inter_object=False → 5.2 段落整体跳过:
    #   ① resolve_penetrations 不跑 → 物体间穿模残留 (46 的 box_1/box_2 重叠 92%)
    #   ② placement_strategies.json 不写 → 5.25 的 on_top 无法区分 inside
    # 看不到地面时, 物体必然是靠在场景内其它物体上 (桌面/托盘/盒面),
    # 故该情形下默认 "supported by other objects"。
    # 兜底默认关系统一用 "supported by other objects"。
    #
    # 为什么不再依赖 has_floor: 实测 SAM3 的地面检测对桌面特写场景不可靠 ——
    # basic_pick_place_46 报 "floor=✅ (43 个 mask)", 把整个桌面当成了地面。
    # 于是 has_floor=True, 若按它决定默认值, 仍然会把全部物体标成 floor。
    # 一旦全标 floor:
    #   ① mainv3 的 has_inter_object=False → 5.2 段落整体跳过
    #      → resolve_penetrations 不跑 → 物体间穿模残留
    #   ② placement_strategies.json 不写 → 5.25 的 on_top 无法区分 inside
    #
    # 改法: 默认交给下游的几何推断 (_infer_supporter 按 xy 重叠 + z 关系找真实
    # 支撑物), 它比 VLM 的兜底和 SAM3 的 floor 检测都可靠 —— 真在地面上的物体
    # 找不到下方支撑物时, 落地补正仍会把它压到 z=0。
    _default_rule = ('- If unsure, default to: "supported by other objects" '
                     '(the geometric stage will resolve the actual supporter; '
                     'do NOT fall back to "supported by floor" as a catch-all)')
    _overhead_rule = ('5. Tabletop / close-up / overhead views: do NOT use '
                      '"supported by floor" as a catch-all. Objects resting on a '
                      'table, tray, box or other item are '
                      '"supported by other objects"')

    prompt = f"""You are a precise spatial relationship analysis API. Output ONLY valid JSON.

**CRITICAL RULES**:
- Output MUST start with {{ and end with }}
- NO explanations, NO thinking process, NO text before/after JSON

**Scene information**:
- Objects in the scene: [{objects_str}] (Total: {num_objects} objects)
- {floor_info}
- {wall_info}

**Task**: For each object, determine which of the 4 relationships applies.

**Valid relationships** (MUST use EXACTLY one of these 4):
- "supported by floor": Object rests DIRECTLY on the floor/ground (e.g., table, chair, cabinet on floor)
- "supported by other objects": Object rests ON TOP of another object (e.g., cup on table, pillow on chair, book on shelf)
- "attached to wall": Object is mounted/hanging on a wall (e.g., picture frame, clock, curtain)
- "embedded in wall": Object is built into/flush with a wall (e.g., door, window, wall socket)

**Key judgment rules**:
1. Furniture on the ground (desk, table, chair, sofa, bed, cabinet, shelf) → "supported by floor"
2. Small items on furniture (cup on table, lamp on desk, pillow on chair, keyboard on desk) → "supported by other objects"
3. Things hanging on wall (picture, mirror, clock, poster, curtain) → "attached to wall"
4. Things built into wall (door, window, outlet, vent) → "embedded in wall"
{_overhead_rule}

**Output format** (STRICT JSON):
{{
    "relationships": {{
        "object_name": "relationship",
        ...
    }}
}}

**CRITICAL REQUIREMENTS**:
- You MUST output exactly {num_objects} objects (one for each: [{objects_str}])
- Do NOT skip any object!
{_default_rule}

**Now analyze this scene and OUTPUT JSON:**"""
    return prompt


# ============================================================
# 视频处理工具
# ============================================================

def _get_video_info(video_path):
    """
    获取视频基本信息（帧率、总帧数）

    参数:
        video_path: 视频文件路径
    返回:
        (fps, total_frames): 帧率和总帧数
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate,nb_frames,duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split('\n')
        fps = 30.0
        total_frames = None

        for line in lines:
            line = line.strip()
            if '/' in line and total_frames is None:
                try:
                    num, den = line.split('/')
                    fps = float(num) / float(den)
                except:
                    pass
            elif line.replace('.', '').isdigit():
                try:
                    val = float(line)
                    if val > 100:
                        total_frames = int(val)
                except:
                    pass

        if total_frames is None:
            try:
                result2 = subprocess.run(
                    ['ffprobe', '-v', 'error', '-count_frames',
                     '-select_streams', 'v:0',
                     '-show_entries', 'stream=nb_read_frames',
                     '-of', 'default=nokey=1:noprint_wrappers=1', video_path],
                    capture_output=True, text=True, timeout=30
                )
                total_frames = int(result2.stdout.strip())
            except:
                total_frames = int(fps * 60)

        return fps, total_frames
    except Exception:
        return 30.0, 1800


def extract_specific_frames(video_path, frame_indices, output_dir):
    """
    从视频中提取指定帧索引的图像

    参数:
        video_path: 视频文件路径
        frame_indices: 要提取的帧索引列表
        output_dir: 输出目录
    返回:
        [(vid_idx, frame_path), ...]: 成功提取的帧列表
    """
    print(f"\n📹 正在提取 {len(frame_indices)} 个关键帧...", flush=True)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fps, total_frames = _get_video_info(video_path)
    print(f"   视频信息: fps={fps}, 总帧数={total_frames}", flush=True)

    frame_paths_with_indices = []
    for i, frame_idx in enumerate(frame_indices):
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1

        timestamp = frame_idx / fps
        frame_path = os.path.join(output_dir, f"frame_{i:03d}_vid{frame_idx}.jpg")

        cmd = [
            'ffmpeg', '-y', '-ss', f'{timestamp:.6f}',
            '-i', video_path, '-frames:v', '1',
            '-q:v', '2', '-update', '1', frame_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 100:
                frame_paths_with_indices.append((frame_idx, frame_path))
            else:
                print(f"   ⚠️  帧#{frame_idx} 提取失败", flush=True)
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  帧#{frame_idx} 提取超时", flush=True)

    print(f"✅ 成功提取 {len(frame_paths_with_indices)} 帧\n", flush=True)
    return frame_paths_with_indices


# ============================================================
# SimRecon 3D空间覆盖采样
# ============================================================

def compute_voxel_sets(world_points, world_points_conf_mask, x_min, y_min, z_min, voxel_size):
    """
    计算每帧的体素集合（用于3D空间覆盖采样）

    参数:
        world_points: 3D点云 (T, H, W, 3)
        world_points_conf_mask: 置信度掩码
        x_min, y_min, z_min: 体素偏移量
        voxel_size: 体素大小
    返回:
        [set of voxel tuples, ...]: 每帧的体素集合
    """
    T = world_points.shape[0]
    voxel_sets = []
    offset = np.array([x_min, y_min, z_min])

    for t in range(T):
        mask = world_points_conf_mask[t].flatten().astype(bool)
        pts_flat = world_points[t].reshape(-1, 3)
        valid_pts = pts_flat[mask[:len(pts_flat)]] if len(mask) >= len(pts_flat) else pts_flat

        if len(valid_pts) == 0:
            voxel_sets.append(set())
            continue

        voxel_coords = np.floor((valid_pts - offset) / voxel_size).astype(int)
        unique_voxels = set(map(tuple, np.unique(voxel_coords, axis=0)))
        voxel_sets.append(unique_voxels)

    return voxel_sets


def maximum_coverage_sampling(voxel_sets, K):
    """
    贪心最大覆盖采样算法（SimRecon核心）

    参数:
        voxel_sets: 每帧的体素集合
        K: 目标采样帧数
    返回:
        选中帧的索引列表
    """
    selected = []
    covered = set()
    remaining_frames = set(range(len(voxel_sets)))

    for iteration in range(K):
        if not remaining_frames:
            break

        max_gain = -1
        best_frame = None

        for frame in remaining_frames:
            gain = len(voxel_sets[frame] - covered)
            if gain > max_gain:
                max_gain = gain
                best_frame = frame

        if best_frame is None or max_gain <= 0:
            break

        selected.append(best_frame)
        covered.update(voxel_sets[best_frame])
        remaining_frames.remove(best_frame)

        if iteration < 3 or iteration >= K - 2:
            print(f"   🎯 第{iteration+1}轮: 选择帧#{best_frame}, 新增{max_gain}个体素, 累计覆盖{len(covered)}个", flush=True)

    total_voxels = set()
    for vs in voxel_sets:
        total_voxels.update(vs)

    coverage_ratio = len(covered) / len(total_voxels) if total_voxels else 0
    print(f"   📊 最终覆盖率: {len(covered)}/{len(total_voxels)} 体素 ({coverage_ratio:.1%})", flush=True)

    return sorted(selected)


def extract_frames_by_vggt_sampling(extrinsics, world_points, world_points_conf, max_frames=10):
    """
    基于VGGT点云的3D空间覆盖采样，选择关键帧

    参数:
        extrinsics: 相机外参列表
        world_points: VGGT输出的3D点云
        world_points_conf: VGGT输出的置信度
        max_frames: 最大采样帧数
    返回:
        关键帧索引列表
    """
    S = len(extrinsics)
    T_wp = world_points.shape[0]
    T_usable = min(S, T_wp)

    if T_usable <= max_frames:
        return list(range(T_usable))

    print(f"   🎯 使用 SimRecon 3D空间覆盖采样...", flush=True)
    print(f"   📐 VGGT数据维度: wp={world_points.shape}, wpc={world_points_conf.shape}, ext={S}", flush=True)
    print(f"   📐 可用帧数: min(ext={S}, wp={T_wp}) = {T_usable}", flush=True)

    wp_subset = world_points[:T_usable]
    wpc_subset = world_points_conf[:T_usable]

    wp_flat = wp_subset.reshape(-1, 3)
    wpc_flat = wpc_subset.flatten()

    min_len = min(len(wpc_flat), len(wp_flat))
    wpc_flat = wpc_flat[:min_len]
    wp_flat = wp_flat[:min_len]

    init_threshold = np.percentile(wpc_flat[wpc_flat > 0], 50) if np.any(wpc_flat > 0) else 0.1
    valid_flat_mask = (wpc_flat >= init_threshold) & (wpc_flat > 0.1)

    if not valid_flat_mask.any():
        # 均匀覆盖全视频, 而不是 range(S)[:N] —— 后者全挤在视频开头,
        # VLM 只看到前段 (与 bug#1 的关键帧偏置同性质)。
        _n = min(max_frames, S)
        _fallback = sorted({int(round(i * (S - 1) / max(_n - 1, 1))) for i in range(_n)})
        print(f"   ⚠️  警告: 没有有效的点云数据, 关键帧退化为均匀采样 {_fallback}", flush=True)
        return _fallback

    valid_points = wp_flat[valid_flat_mask]
    x_min, y_min, z_min = valid_points[:, 0].min(), valid_points[:, 1].min(), valid_points[:, 2].min()
    x_max, y_max, z_max = valid_points[:, 0].max(), valid_points[:, 1].max(), valid_points[:, 2].max()

    print(f"   📏 边界盒: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}], z=[{z_min:.2f}, {z_max:.2f}]", flush=True)

    scene_extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
    voxel_size = max(scene_extent / 20.0, 0.01)
    print(f"   📏 体素大小: {voxel_size:.4f}m", flush=True)

    T = T_usable
    pts_per_frame = wp_flat.shape[0] // T
    conf_mask_reshaped = valid_flat_mask.reshape(T, pts_per_frame) if pts_per_frame * T == len(valid_flat_mask) else None

    if conf_mask_reshaped is None:
        conf_mask_reshaped = np.ones_like(wpc_subset, dtype=bool)

    print(f"   🎯 计算体素集合...", flush=True)
    voxel_sets = compute_voxel_sets(wp_subset, conf_mask_reshaped, x_min, y_min, z_min, voxel_size)

    print(f"   🎯 运行贪心最大覆盖算法 (目标{max_frames}帧)...", flush=True)
    keyframe_indices = maximum_coverage_sampling(voxel_sets, max_frames)

    return sorted(keyframe_indices)


# ============================================================
# CLIP 语义匹配器
# ============================================================

class CLIPSemanticMatcher:
    """CLIP语义相似度匹配器，用于图像和文本的语义比较"""

    CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
    CLIP_MIRROR_ID = "AI-ModelScope/clip-vit-base-patch32"
    CLIP_LOCAL_PATH = "/mnt/data/lza/models/clip-vit-base-patch32"

    def __init__(self, device="cuda"):
        """
        初始化CLIP模型，依次尝试本地/HuggingFace/ModelScope加载

        参数:
            device: 计算设备
        """
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP 模型不可用")

        print(f"📦 加载 CLIP 模型...", flush=True)
        self.device = device

        model_candidates = [
            (self.CLIP_LOCAL_PATH, "本地模型"),
            (self.CLIP_MODEL_ID, "HuggingFace"),
            (self.CLIP_MIRROR_ID, "ModelScope镜像"),
        ]

        loaded = False
        for model_id, source_name in model_candidates:
            try:
                print(f"   尝试从 {source_name} 加载: {model_id}", flush=True)
                self.clip_model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
                self.clip_processor = CLIPProcessor.from_pretrained(model_id)
                print(f"✅ CLIP 模型加载完成 ({source_name}, safetensors)", flush=True)
                loaded = True
                break
            except Exception as e:
                print(f"   ⚠️  {source_name} safetensors 失败: {e}", flush=True)
                try:
                    self.clip_model = CLIPModel.from_pretrained(model_id).to(device)
                    self.clip_processor = CLIPProcessor.from_pretrained(model_id)
                    print(f"✅ CLIP 模型加载完成 ({source_name})", flush=True)
                    loaded = True
                    break
                except Exception as e2:
                    print(f"   ⚠️  {source_name} 加载失败: {e2}", flush=True)
                    continue

        if not loaded:
            raise RuntimeError("CLIP 模型加载失败，所有源均不可用")

        self.clip_model.eval()

    def _extract_features(self, outputs):
        """从CLIP输出中提取特征向量"""
        if isinstance(outputs, torch.Tensor):
            features = outputs
        elif hasattr(outputs, 'image_embeds'):
            features = outputs.image_embeds
        elif hasattr(outputs, 'text_embeds'):
            features = outputs.text_embeds
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            features = outputs[0] if hasattr(outputs, '__getitem__') else outputs
        if features.dim() == 3:
            features = features[:, 0, :]
        return features

    def compute_image_similarity(self, image1, image2):
        """
        计算两张图像的CLIP语义相似度

        参数:
            image1, image2: PIL图像
        返回:
            相似度分数 [0, 1]
        """
        try:
            inputs = self.clip_processor(
                images=[image1, image2],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            pixel_values = inputs.get("pixel_values")
            if pixel_values is None:
                return 0.0

            with torch.no_grad():
                image_out = self.clip_model.get_image_features(pixel_values=pixel_values)

            if isinstance(image_out, torch.Tensor):
                image_features = image_out
            elif hasattr(image_out, 'pooler_output'):
                image_features = image_out.pooler_output
            else:
                return 0.0

            if image_features is None or image_features.shape[0] != 2:
                return 0.0

            norms = image_features.norm(dim=1, keepdim=True)
            if (norms < 1e-8).any():
                return 0.0

            image_features = image_features / norms
            similarity = torch.dot(image_features[0], image_features[1]).item()

            if math.isnan(similarity) or math.isinf(similarity):
                return 0.0

            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0

    def compute_text_similarity(self, text1, text2):
        """
        计算两段文本的CLIP语义相似度

        参数:
            text1, text2: 文本字符串
        返回:
            相似度分数 [0, 1]
        """
        try:
            tokenizer = self.clip_processor.tokenizer
            tokens = tokenizer(
                [text1, text2],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                text_out = self.clip_model.get_text_features(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"]
                )

            if isinstance(text_out, torch.Tensor):
                text_features = text_out
            elif hasattr(text_out, 'pooler_output'):
                text_features = text_out.pooler_output
            else:
                return 0.0

            if text_features is None or text_features.shape[0] != 2:
                return 0.0

            norms = text_features.norm(dim=1, keepdim=True)
            if (norms < 1e-8).any():
                return 0.0

            text_features = text_features / norms
            similarity = torch.dot(text_features[0], text_features[1]).item()

            if math.isnan(similarity) or math.isinf(similarity):
                return 0.0

            return max(0.0, min(1.0, similarity))
        except Exception:
            return 0.0


# ============================================================
# 名称标准化
# ============================================================

def normalize_category_name(category_name):
    """
    标准化类别名称（小写、去复数）

    参数:
        category_name: 原始类别名称
    返回:
        标准化后的名称
    """
    name = category_name.lower().strip()
    if name.endswith('ies'):
        name = name[:-3] + 'y'
    elif name.endswith('ves'):
        name = name[:-3] + 'f'
    elif name.endswith('es') and not name.endswith('ses'):
        name = name[:-2]
    elif name.endswith('s') and not name.endswith('ss'):
        name = name[:-1]
    return name


def merge_synonyms(category_name):
    """
    合并同义词：先查SYNONYM_MAP，再标准化

    参数:
        category_name: 原始类别名称
    返回:
        合并后的名称 (空格转下划线, 与 MUST_BE_WALL_ATTACHED 等下划线
        形式的规则集一致 —— 否则 "wall clock" 绕过物理纠错)
    """
    name = category_name.lower().strip()
    if name in SYNONYM_MAP:
        return SYNONYM_MAP[name]
    return normalize_category_name(name.replace(" ", "_"))


def _merge_multi_vlm_detections(main_detections, aux_detections):
    """
    合并主VLM和辅助VLM的检测结果（并集 + 同义词归一化 + 位置去重）

    合并策略:
      1. 按 frame_idx 分组，主模型已有该帧的以其为主
      2. 对辅助模型检测到的物体:
         - 若主模型已检测到同义类别（经merge_synonyms归一化后相同），跳过
         - 若主模型未检测到（漏检），添加到该帧的 objects 列表
         - 若同帧位置重复（中心距离<50px），仅保留主模型的检测

    参数:
        main_detections: [{frame_idx, frame_path, objects}]
        aux_detections: [{frame_idx, frame_path, objects}]
    返回:
        合并后的 detections，结构同 main_detections
    """
    main_by_frame = {d["frame_idx"]: d for d in main_detections}

    for aux_det in aux_detections:
        fid = aux_det["frame_idx"]
        if fid not in main_by_frame:
            main_by_frame[fid] = {
                "frame_idx": fid,
                "frame_path": aux_det["frame_path"],
                "objects": list(aux_det["objects"]),
            }
            continue

        main_objs = main_by_frame[fid]["objects"]
        main_std_names = set()
        for obj in main_objs:
            if "name" in obj:
                main_std_names.add(merge_synonyms(obj["name"].lower().strip().replace(" ", "_")))

        main_centers = []
        for obj in main_objs:
            if "center_x" in obj and "center_y" in obj:
                main_centers.append((obj["center_x"], obj["center_y"]))

        for aux_obj in aux_det.get("objects", []):
            if "name" not in aux_obj:
                continue
            std = merge_synonyms(aux_obj["name"].lower().strip().replace(" ", "_"))
            if std in main_std_names:
                continue
            if std in FILTER_CATEGORIES:
                continue
            ax = aux_obj.get("center_x")
            ay = aux_obj.get("center_y")
            if ax is not None and ay is not None:
                if any((ax - mx) ** 2 + (ay - my) ** 2 < 50 ** 2 for mx, my in main_centers):
                    continue
            main_objs.append(aux_obj)
            main_std_names.add(std)
            if ax is not None and ay is not None:
                main_centers.append((ax, ay))

    merged = sorted(main_by_frame.values(), key=lambda d: d["frame_idx"])
    return merged


# ============================================================
# JSON 解析工具
# ============================================================

def _extract_json_from_text(text):
    """
    从VLM输出文本中提取JSON字符串

    依次尝试: 去除think标签 → 代码块 → 从第一个{匹配完整JSON → 首尾花括号

    参数:
        text: VLM原始输出文本
    返回:
        JSON字符串，或None
    """
    if '</think_>' in text:
        text = text.split('</think_>')[-1].strip()

    code_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[-1]

    code_block_pattern2 = r'```\s*({[\s\S]*?})\s*```'
    matches2 = re.findall(code_block_pattern2, text, re.DOTALL)
    if matches2:
        return matches2[-1]

    first_brace = text.find('{')
    if first_brace != -1:
        brace_count = 0
        for i in range(first_brace, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    complete_json = text[first_brace:i+1]
                    try:
                        json.loads(complete_json)
                        return complete_json
                    except json.JSONDecodeError:
                        continue
        if first_brace < len(text):
            last_brace = text.rfind('}')
            if last_brace > first_brace:
                return text[first_brace:last_brace+1]

    return None


def _fix_json_string(json_str):
    """
    修复常见的JSON格式错误（尾逗号、注释、无引号键名）

    参数:
        json_str: 有格式问题的JSON字符串
    返回:
        修复后的JSON字符串
    """
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    return json_str


# VLM 推理函数已从 src/vlm_logger 导入，不再在此处定义


# ============================================================
# 射线投射: 将VLM像素位置映射到3D空间
# ============================================================

def pixel_to_3d_position(u, v, extrinsic, intrinsic, world_points_frame, world_points_conf_frame):
    """
    通过射线投射将2D像素坐标映射到3D空间位置

    参数:
        u, v: 像素坐标
        extrinsic: 相机外参矩阵 (4x4)
        intrinsic: 相机内参矩阵 (3x3)
        world_points_frame: 该帧的3D点云
        world_points_conf_frame: 该帧的置信度
    返回:
        3D位置 numpy数组(3,)，或None
    """
    pts_flat = world_points_frame.reshape(-1, 3)
    conf_flat = world_points_conf_frame.flatten()

    min_len = min(len(conf_flat), len(pts_flat))
    pts_flat = pts_flat[:min_len]
    conf_flat = conf_flat[:min_len]

    valid_mask = conf_flat > 0.1
    valid_pts = pts_flat[valid_mask]

    if len(valid_pts) < 3:
        return None

    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    cam_pos = -R.T @ t

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    if fx < 1 or fy < 1:
        return None

    ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    ray_norm = np.linalg.norm(ray_cam)
    if ray_norm < 1e-8:
        return None
    ray_cam = ray_cam / ray_norm
    ray_world = R.T @ ray_cam

    PO = valid_pts - cam_pos
    cross = np.cross(PO, ray_world)
    distances = np.linalg.norm(cross, axis=1)

    dot_products = np.sum(PO * ray_world, axis=1)
    in_front = dot_products > 0

    if not in_front.any():
        return None

    distances[~in_front] = float('inf')

    sorted_indices = np.argsort(distances)
    top_k = min(RAY_CAST_TOP_K, len(sorted_indices))
    nearest_indices = sorted_indices[:top_k]

    nearest_pts = valid_pts[nearest_indices]
    nearest_dists = distances[nearest_indices]

    valid_nearest = nearest_pts[nearest_dists < float('inf')]
    if len(valid_nearest) == 0:
        return None

    position_3d = np.median(valid_nearest, axis=0)

    if np.any(np.isnan(position_3d)):
        return None

    return position_3d


# ============================================================
# Step 3: 第一次VLM调用 — 物体检测（仅名称+位置）
# ============================================================

def detect_objects_in_frames(frame_paths_with_indices, model, processor, aux_model=None, aux_processor=None, enable_multi_vlm=False, aux_min_objects=8):
    """
    第一次VLM调用: 逐帧检测物体，只获取名称，不判断位置和关系

    支持多VLM联合检测（可选）:
      - 主模型对所有帧做检测
      - 若主模型检测到的独立物体数 < aux_min_objects，触发辅助VLM补全
      - 辅助VLM也做逐帧检测，结果与主模型合并（并集 + 同义词归一）
      - 注意: 辅助模型不同时加载，检测完主模型后释放显存再加载

    参数:
        frame_paths_with_indices: [(vid_idx, frame_path), ...]
        model: 主VLM模型
        processor: 主VLM processor
        aux_model: 辅助VLM模型 (可选, 高精度模型如32B)
        aux_processor: 辅助VLM processor
        enable_multi_vlm: 是否启用多VLM
        aux_min_objects: 主模型独立物体数 < 该值时触发辅助VLM
    返回:
        [{"frame_idx", "frame_path", "objects": [{name}]}]
    """
    print(f"\n{'='*70}", flush=True)
    print(f"🔍 Step 3: 第一次VLM调用 — 物体检测 ({len(frame_paths_with_indices)} 帧)", flush=True)
    print(f"{'='*70}\n", flush=True)

    all_detections = []
    total_objects = 0

    # ── 批量检测 ──
    # 原来逐帧调一次 VLM: N 个关键帧 = N 次调用。实测 weixin (10 帧, API 模式)
    # Stage1 光检测就 861s, 其中逐帧调用是主要开销 (每次约 40s, 含模型往返)。
    # 改成每次带 BATCH 帧: 本地 VLM 支持一条消息多图, API 模式把多个路径写进
    # prompt 让它逐个读。BATCH=5 时 10 帧从 10 次降到 2 次。
    # 输出格式相应改成按帧索引分组的 JSON, 解析后拆回逐帧结果 —— 下游看到的
    # 数据结构和逐帧版完全一致。
    _BATCH = int(os.environ.get("STAGE1_VLM_BATCH", "5"))
    _batches = [frame_paths_with_indices[i:i + _BATCH]
                for i in range(0, len(frame_paths_with_indices), _BATCH)]
    print(f"   批量检测: {len(frame_paths_with_indices)} 帧分 "
          f"{len(_batches)} 批 (每批最多 {_BATCH} 帧)", flush=True)

    for _bi, _batch in enumerate(_batches):
        print(f"[{_bi+1}/{len(_batches)}] [主VLM] 帧 "
              f"{[v for v, _ in _batch]} ...", end=" ", flush=True)
        _texts = None
        try:
            if USE_API_VLM and len(_batch) > 1:
                _paths = "\n".join(p for _, p in _batch)
                _bp = (f"Read these {len(_batch)} image files (one per line):\n"
                       f"{_paths}\n\nFor EACH image, list the visible objects.\n"
                       f"Output ONE JSON object keyed by image order "
                       f"(0-based), no markdown fence:\n"
                       f'{{"0": {{"objects": [{{"name": "..."}}]}}, '
                       f'"1": {{"objects": [...]}}}}\n\n'
                       f"Rules per image:\n"
                       f"- Simple singular English nouns, lowercase\n"
                       f"- Ignore: hands, body parts, walls, floors, ceilings\n"
                       f"- List each object type once\n")
                _raw = vlm_inference_api(_batch[0][1], _bp)
                _texts = _parse_batch_detect(_raw, len(_batch))
                if _raw and _texts is None:
                    print("(批量解析失败, 退回逐帧) ", end="", flush=True)
                    _texts = None
        except Exception as _e:
            print(f"(批量异常 {type(_e).__name__}, 退回逐帧) ", end="", flush=True)
            _texts = None

        for _j, (vid_idx, frame_path) in enumerate(_batch):
            try:
                image = Image.open(frame_path).convert("RGB")
                start_time = time.time()
                if _texts is not None and _j < len(_texts) and _texts[_j]:
                    output_text = _texts[_j]
                    elapsed = 0.0
                elif USE_API_VLM:
                    output_text = vlm_inference_api(frame_path, VLM_DETECT_PROMPT)
                    if not output_text:
                        print("(API失败, 回退本地VLM) ", end="", flush=True)
                        output_text = vlm_inference(
                            image, model, processor, VLM_DETECT_PROMPT,
                            max_new_tokens=1024, caller="stage1_detect")
                else:
                    output_text = vlm_inference(image, model, processor, VLM_DETECT_PROMPT, max_new_tokens=1024, caller="stage1_detect")
                elapsed = time.time() - start_time

                json_str = _extract_json_from_text(output_text)
                if json_str is None:
                    print(f"❌ 帧#{vid_idx} 无法提取JSON ({elapsed:.1f}s)")
                    continue

                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e1:
                    json_str_fixed = _fix_json_string(json_str)
                    try:
                        result = json.loads(json_str_fixed)
                    except json.JSONDecodeError as e2:
                        print(f"❌ JSON解析失败 ({elapsed:.1f}s)")
                        print(f"      🔍 提取的JSON字符串 (前500字符): {json_str[:500]}", flush=True)
                        print(f"      🔧 修复后JSON (前500字符): {json_str_fixed[:500]}", flush=True)
                        print(f"      ⚠️  原始错误: {str(e1)[:200]}", flush=True)
                        print(f"      ⚠️  修复后错误: {str(e2)[:200]}", flush=True)
                        print(f"      📝 VLM完整输出:\n{output_text}", flush=True)
                        continue
    
                if isinstance(result, list):
                    objects = result
                else:
                    objects = result.get("objects", result.get("detections", []))
    
                img_w, img_h = image.size
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    if "label" in obj and "name" not in obj:
                        obj["name"] = obj.pop("label")
                    bbox = None
                    if "bbox" in obj:
                        bbox = obj["bbox"]
                    elif "bbox_2d" in obj:
                        bbox = obj["bbox_2d"]
                    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        obj["center_x"] = (int(bbox[0]) + int(bbox[2])) // 2
                        obj["center_y"] = (int(bbox[1]) + int(bbox[3])) // 2
                        obj["bbox"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    if "center_x" not in obj or "center_y" not in obj:
                        obj["center_x"] = img_w // 2
                        obj["center_y"] = img_h // 2
    
                valid_objects = [obj for obj in objects if "name" in obj]
                total_objects += len(valid_objects)
                all_detections.append({
                    "frame_idx": vid_idx,
                    "frame_path": frame_path,
                    "objects": valid_objects
                })
    
                obj_names = [obj["name"] for obj in valid_objects]
                print(f"✅ {len(valid_objects)}个: {', '.join(obj_names)} ({elapsed:.1f}s)", flush=True)
    
            except Exception as e:
                print(f"❌ 错误: {e}", flush=True)
    unique_main = set()
    for det in all_detections:
        for obj in det["objects"]:
            std = merge_synonyms(obj["name"].lower().strip().replace(" ", "_")) if "name" in obj else None
            if std and std not in FILTER_CATEGORIES:
                unique_main.add(std)
    print(f"\n📊 主VLM总计: {total_objects} 个物体实例, {len(unique_main)} 个独立类别", flush=True)

    # count 字段诊断: 每类别取各帧中的最大自报实例数。
    # 目的: 让"VLM 少数"与"SAM3 漏切"第一次可区分 —— 把这里的数字和下游
    # _pre_dedup_masks.pkl 里该类别实际切出的 mask 数对比即可。
    # 例如 141 若这里报 chess piece=8 而 SAM3 切出 9, 说明是 SAM3 侧问题;
    # 若这里报 3 而实拍有 8, 才是 VLM 少数。
    _cnt = {}
    for _det in all_detections:
        for _obj in _det.get("objects", []):
            if "name" not in _obj:
                continue
            _std = merge_synonyms(_obj["name"].lower().strip().replace(" ", "_"))
            if not _std or _std in FILTER_CATEGORIES:
                continue
            try:
                _c = int(_obj.get("count", 1) or 1)
            except Exception:
                _c = 1
            _cnt[_std] = max(_cnt.get(_std, 0), _c)
    if _cnt:
        _tot = sum(_cnt.values())
        _top = dict(sorted(_cnt.items(), key=lambda x: -x[1]))
        print(f"   🔢 VLM 自报实例数合计 {_tot}: {_top}", flush=True)

    if enable_multi_vlm and aux_model is not None and aux_processor is not None and len(unique_main) < aux_min_objects:
        print(f"\n{'='*70}", flush=True)
        print(f"🤝 多VLM联合检测: 主VLM独立类别 {len(unique_main)} < {aux_min_objects}, 触发辅助VLM补全", flush=True)
        print(f"{'='*70}\n", flush=True)

        aux_detections = []
        aux_total = 0
        for i, (vid_idx, frame_path) in enumerate(frame_paths_with_indices):
            print(f"[{i+1}/{len(frame_paths_with_indices)}] [辅VLM] 帧 #{vid_idx}...", end=" ", flush=True)
            try:
                image = Image.open(frame_path).convert("RGB")
                start_time = time.time()
                output_text = vlm_inference(image, aux_model, aux_processor, VLM_DETECT_PROMPT, max_new_tokens=1024, caller="stage1_detect_aux")
                elapsed = time.time() - start_time

                json_str = _extract_json_from_text(output_text)
                if json_str is None:
                    print(f"❌ 无法提取JSON ({elapsed:.1f}s)")
                    continue

                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        result = json.loads(_fix_json_string(json_str))
                    except Exception:
                        print(f"❌ JSON解析失败 ({elapsed:.1f}s)")
                        continue

                if isinstance(result, list):
                    objects = result
                else:
                    objects = result.get("objects", result.get("detections", []))

                img_w, img_h = image.size
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    if "label" in obj and "name" not in obj:
                        obj["name"] = obj.pop("label")
                    bbox = None
                    if "bbox" in obj:
                        bbox = obj["bbox"]
                    elif "bbox_2d" in obj:
                        bbox = obj["bbox_2d"]
                    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        obj["center_x"] = (int(bbox[0]) + int(bbox[2])) // 2
                        obj["center_y"] = (int(bbox[1]) + int(bbox[3])) // 2
                        obj["bbox"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    if "center_x" not in obj or "center_y" not in obj:
                        obj["center_x"] = img_w // 2
                        obj["center_y"] = img_h // 2

                valid_objects = [obj for obj in objects if "name" in obj]
                aux_total += len(valid_objects)
                aux_detections.append({
                    "frame_idx": vid_idx,
                    "frame_path": frame_path,
                    "objects": valid_objects,
                })

                obj_names = [obj["name"] for obj in valid_objects]
                print(f"✅ {len(valid_objects)}个: {', '.join(obj_names)} ({elapsed:.1f}s)", flush=True)
            except Exception as e:
                print(f"❌ 错误: {e}", flush=True)

        unique_aux = set()
        for det in aux_detections:
            for obj in det["objects"]:
                std = merge_synonyms(obj["name"].lower().strip().replace(" ", "_")) if "name" in obj else None
                if std and std not in FILTER_CATEGORIES:
                    unique_aux.add(std)
        new_objects = unique_aux - unique_main
        print(f"\n📊 辅助VLM总计: {aux_total} 个物体实例, {len(unique_aux)} 个独立类别", flush=True)
        print(f"🆕 辅助VLM新发现: {len(new_objects)} 个 → {new_objects}", flush=True)

        all_detections = _merge_multi_vlm_detections(all_detections, aux_detections)
        print(f"🔗 合并后总物体实例: {sum(len(d['objects']) for d in all_detections)}", flush=True)
    elif enable_multi_vlm and len(unique_main) >= aux_min_objects:
        print(f"\n✅ 主VLM独立类别 {len(unique_main)} ≥ {aux_min_objects}, 跳过辅助VLM", flush=True)
    elif enable_multi_vlm:
        print(f"\n⚠️  多VLM已启用但未提供辅助模型, 跳过补全", flush=True)

    return all_detections


# ============================================================
# Step 4: 构建物体实例（射线投射法）
# ============================================================

def build_object_instances(all_detections, vggt_results, total_video_frames=0,
                           keyframe_size=None):
    """
    将VLM检测到的2D像素位置通过射线投射映射到3D空间

    参数:
        all_detections: detect_objects_in_frames的输出 (frame_idx 是原视频帧号)
        vggt_results: VGGT重建结果
        total_video_frames: 原视频总帧数; > VGGT 帧数时 frame_idx 被视为
            原视频帧号并转换为 VGGT 槽位 (bug#1: 关键帧按原视频帧号落盘,
            直接当槽位用会越界丢弃或错位索引)
        keyframe_size: (W, H) 原分辨率关键帧尺寸, 用于把检测框中心从原图
            像素缩放到 VGGT 帧像素 (内参对应的是 VGGT 帧)
    返回:
        [{frame_idx, name, description, center_x, center_y, centroid}]
    """
    print(f"\n{'='*70}", flush=True)
    print(f"🎯 Step 4: 射线投射 → 将VLM像素位置映射到3D空间", flush=True)
    print(f"{'='*70}\n", flush=True)

    world_points = vggt_results['world_points'][0]
    world_points_conf = vggt_results['world_points_conf'][0]
    extrinsics = vggt_results['extrinsics']
    intrinsic = vggt_results['intrinsic']

    T = world_points.shape[0]
    T_ext = len(extrinsics)
    T_usable = min(T, T_ext)
    print(f"   📐 VGGT数据: wp={world_points.shape}, wpc={world_points_conf.shape}, ext={T_ext}, T_usable={T_usable}", flush=True)

    # 原视频帧号 → VGGT 槽位 (load_video_frames 用 linspace(0, total-1, T) 抽帧)
    _need_map = bool(total_video_frames and total_video_frames > T_usable > 1)
    if _need_map:
        print(f"   🔁 检测帧号索引转换: 原视频 {total_video_frames} 帧 → VGGT {T_usable} 槽位", flush=True)

    def _vid2slot(f):
        if not _need_map:
            return int(f)
        return int(min(T_usable - 1, max(0, round(f * (T_usable - 1) / (total_video_frames - 1)))))

    # ── 像素空间对齐 ──
    # keyframe_size=(W,H): 原分辨率关键帧尺寸。VGGT 预处理 (resize/crop) 后
    # 内参对应的是 VGGT 帧像素, 两个空间不缩放会整体错位。
    _uv_scale = None
    if keyframe_size:
        _Hv, _Wv = world_points.shape[0], world_points.shape[1]
        _Wo, _Ho = keyframe_size
        if _Wo > 0 and _Ho > 0:
            _uv_scale = (_Wv / _Wo, _Hv / _Ho)
            _ar_vggt = _Wv / max(_Hv, 1)
            _ar_orig = _Wo / max(_Ho, 1)
            if abs(_ar_vggt / max(_ar_orig, 1e-9) - 1.0) > 0.02:
                print(f"   ⚠️ 关键帧宽高比 ({_Wo}x{_Ho}) 与 VGGT 帧 ({_Wv}x{_Hv}) "
                      f"不一致 (crop 模式), 线性缩放可能有残余偏差", flush=True)
            print(f"   🔍 像素空间对齐: 关键帧 {_Wo}x{_Ho} → VGGT {_Wv}x{_Hv} "
                  f"(缩放 {_uv_scale[0]:.3f}/{_uv_scale[1]:.3f})", flush=True)

    object_instances = []
    total_instances = sum(len(d['objects']) for d in all_detections)
    success_count = 0
    n_dropped = 0

    print(f"   为 {total_instances} 个物体实例估计3D位置...", flush=True)

    for detection in all_detections:
        frame_idx = _vid2slot(detection["frame_idx"])
        if frame_idx >= T_usable:
            n_dropped += 1
            continue

        wp_frame = world_points[frame_idx]
        wpc_frame = world_points_conf[frame_idx]
        ext_frame = extrinsics[frame_idx]

        for obj in detection["objects"]:
            center_x = obj.get("center_x", 0)
            center_y = obj.get("center_y", 0)
            # 检测框中心是"原分辨率关键帧"像素 (VLM 看的是原图), 而 intrinsic
            # 是 VGGT 预处理帧的 —— 不缩放射线会偏离数倍 (实测 1920x1080 关键帧
            # vs 688x384 VGGT 帧, 图像中心被算成 ~50° 偏轴)。
            if _uv_scale:
                center_x = center_x * _uv_scale[0]
                center_y = center_y * _uv_scale[1]

            centroid = pixel_to_3d_position(
                center_x, center_y,
                ext_frame, intrinsic,
                wp_frame, wpc_frame
            )

            if centroid is not None:
                success_count += 1

            object_instances.append({
                "frame_idx": frame_idx,
                "name": obj["name"],
                "description": obj.get("description", ""),
                "center_x": center_x,
                "center_y": center_y,
                "centroid": centroid,
            })

    print(f"   ✅ {len(object_instances)} 个物体实例，{success_count} 个成功映射到3D", flush=True)
    if n_dropped:
        print(f"   ⚠️ {n_dropped} 个检测因帧号超出 VGGT 范围被丢弃 "
              f"(索引转换后仍越界, 检查帧号空间映射)", flush=True)
    print(flush=True)
    return object_instances


# ============================================================
# Step 5: 名称 + 3D位置联合去重
# ============================================================

CLIP_MERGE_THRESHOLD = 0.90


def deduplicate_objects(object_instances, centroid_dist_thre=CENTROID_DIST_THRESHOLD, clip_matcher=None):
    """
    基于CLIP语义匹配 + SYNONYM_MAP的联合去重

    去重策略（优先级从高到低）:
      1. SYNONYM_MAP精确匹配: "mug"→"cup", "sofa"→"couch" 等
      2. CLIP语义相似度: 文本相似度 >= 0.85 视为同义 (如 "trash can" ≈ "garbage can")
      3. 名称标准化: 去复数、小写等

    参数:
        object_instances: build_object_instances的输出
        centroid_dist_thre: 3D质心距离阈值（米），仅用于非同名物体
        clip_matcher: CLIPSemanticMatcher实例（可选，无则仅用SYNONYM_MAP）
    返回:
        去重后的唯一物体列表
    """
    print(f"\n{'='*70}", flush=True)
    print(f"🔗 Step 5: 语义去重（SYNONYM_MAP + CLIP）", flush=True)
    print(f"{'='*70}\n", flush=True)

    name_groups = defaultdict(list)
    for inst in object_instances:
        std_name = merge_synonyms(inst["name"])
        name_groups[std_name].append(inst)

    if clip_matcher and len(name_groups) > 1:
        print(f"   🧠 CLIP语义匹配中 ({len(name_groups)} 个候选名称)...", flush=True)
        group_names = list(name_groups.keys())
        merged_pairs = []

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name_i = group_names[i]
                name_j = group_names[j]
                if name_i in FILTER_CATEGORIES or name_j in FILTER_CATEGORIES:
                    continue
                try:
                    sim = clip_matcher.compute_text_similarity(
                        name_i.replace("_", " "), name_j.replace("_", " ")
                    )
                    if sim >= CLIP_MERGE_THRESHOLD:
                        merged_pairs.append((name_i, name_j, sim))
                        print(f"   🔗 CLIP合并: '{name_i}' ≈ '{name_j}' (相似度={sim:.3f})", flush=True)
                except Exception:
                    pass

        for name_i, name_j, sim in merged_pairs:
            if name_i in name_groups and name_j in name_groups:
                if name_i in SYNONYM_MAP.values() or name_i in SYNONYM_MAP:
                    keep, remove = name_i, name_j
                elif len(name_groups[name_i]) >= len(name_groups[name_j]):
                    keep, remove = name_i, name_j
                else:
                    keep, remove = name_j, name_i
                name_groups[keep].extend(name_groups.pop(remove))
                print(f"   🔄 '{remove}' → '{keep}' (合并 {len(name_groups[keep])} 个实例)", flush=True)
    else:
        if not clip_matcher:
            print(f"   ⚠️  CLIP不可用，仅使用SYNONYM_MAP去重", flush=True)

    unique_objects = []
    for std_name, instances in name_groups.items():
        # 跨帧多数投票: 按原始名称计数取最高, 而非 instances[0]
        raw_counts = Counter(inst["name"].lower().strip() for inst in instances)
        most_name, most_count = raw_counts.most_common(1)[0]
        # 选中该名称对应实例作为代表, 保持原始大小写
        representative = next(
            inst for inst in instances
            if inst["name"].lower().strip() == most_name
        ).copy()
        representative["name"] = representative["name"].lower().strip()
        representative["instance_count"] = len(instances)

        centroids = [inst["centroid"] for inst in instances if inst.get("centroid") is not None]
        if centroids:
            representative["centroid"] = np.median(centroids, axis=0)

        unique_objects.append(representative)

        if len(instances) > 1:
            frame_indices = [inst['frame_idx'] for inst in instances]
            print(f"   🔄 合并: '{std_name}' {len(instances)}个实例 → 1个 (帧: {frame_indices})", flush=True)

    print(f"\n📊 去重后剩余 {len(unique_objects)} 个唯一物体类别\n", flush=True)
    return unique_objects


# ============================================================
# Step 5.5: 基于点云的补充检测 — 发现VLM遗漏的远端大物体
# ============================================================

SUPPLEMENTARY_CROP_SIZE = 300
SUPPLEMENTARY_VLM_PROMPT = """What is the large object in the center of this cropped image?

If you can clearly identify an object, reply with JSON:
{{"name": "object_name"}}

Use simple singular names like: cabinet, bookshelf, plant, door, window, refrigerator.
If no clear object is visible, reply: {{"name": "none"}}

Output JSON:"""


def _voxel_downsample(points, voxel_size=0.05):
    h = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(h, axis=0, return_index=True)
    return points[idx]


def _project_3d_to_2d(centroid, world_points_frame):
    diff = world_points_frame - centroid
    dist_map = np.sum(diff ** 2, axis=2)
    min_pos = np.unravel_index(np.argmin(dist_map), dist_map.shape)
    return int(min_pos[1]), int(min_pos[0])


def _video_frame_to_slot(vid_idx, total_video_frames, t_vggt):
    """原视频帧号 → VGGT 槽位。

    load_video_frames 用 np.linspace(0, total-1, T) 抽帧, 所以槽位 i 对应
    原视频帧 round(i*(total-1)/(T-1))。total<=T 时两个索引空间重合, 原样返回。
    (bug#1 家族: frame_paths_with_indices 存的是原视频帧号, 直接当槽位索引
    world_points 会越界丢弃或错位取点。)
    """
    vid_idx = int(vid_idx)
    if not total_video_frames or total_video_frames <= t_vggt or t_vggt <= 1:
        return vid_idx
    return int(min(t_vggt - 1, max(0, round(vid_idx * (t_vggt - 1) / (total_video_frames - 1)))))


def supplementary_detect_from_pointcloud(vggt_results, unique_objects,
                                          frame_paths_with_indices, model, processor,
                                          sam_results=None):
    """
    Step 5.5: 基于点云的补充检测 — 发现VLM遗漏的远端大物体

    保留原有VLM检测结果，通过分析VGGT点云中未被检测物体覆盖的
    显著3D聚类，补充检测远端大物体。

    流程:
      1. 从VGGT点云中提取高置信度3D点，体素降采样
      2. 排除已知物体附近和floor/wall区域的点
      3. DBSCAN聚类找出独立的3D区域
      4. 过滤: 排除过小(噪声)、过大(墙/地板)、过扁(平面)的聚类
      5. 对剩余聚类，投影到2D并裁剪图像区域
      6. 用VLM识别裁剪区域中的物体
      7. 用点云大小/位置决定是否采纳

    参数:
        vggt_results: VGGT重建结果
        unique_objects: Step 5去重后的物体列表
        frame_paths_with_indices: 关键帧路径列表
        model: VLM模型
        processor: VLM处理器
        sam_results: SAM分割结果（可选，用于排除floor/wall区域）
        total_video_frames: 原视频总帧数; > VGGT 帧数时 frame_paths_with_indices
            里的帧号是原视频帧号, 索引 world_points 前必须换算成 VGGT 槽位
    返回:
        补充检测到的物体列表 [{name, centroid, source}]
    """
    print(f"\n{'='*70}", flush=True)
    print(f"🔎 Step 5.5: 点云补充检测 — 发现遗漏的远端大物体", flush=True)
    print(f"{'='*70}\n", flush=True)

    world_points = vggt_results['world_points'][0]
    world_points_conf = vggt_results['world_points_conf'][0]
    T, H, W = world_points.shape[:3]
    _need_map = bool(total_video_frames and total_video_frames > T > 1)
    if _need_map:
        print(f"   🔁 帧号索引转换: 原视频 {total_video_frames} 帧 → VGGT {T} 槽位", flush=True)

    def _vid2slot(vid_idx):
        if not _need_map:
            return int(vid_idx)
        return int(min(T - 1, max(0, round(vid_idx * (T - 1) / (total_video_frames - 1)))))

    known_centroids = []
    for obj in unique_objects:
        c = obj.get('centroid')
        if c is not None and not np.any(np.isnan(c)):
            known_centroids.append(np.array(c))
    print(f"   已知物体: {len(known_centroids)} 个", flush=True)

    # ---- 1. 提取高置信度3D点 ----
    sample_step = max(1, T // 5)
    all_points = []
    frame_sample_indices = list(range(0, T, sample_step))

    print(f"   🔍 采样帧索引: {frame_sample_indices}", flush=True)
    
    for t in frame_sample_indices:
        # 诊断：打印置信度统计信息
        conf_frame = world_points_conf[t]
        valid_mask = conf_frame > 0
        if valid_mask.any():
            conf_valid = conf_frame[valid_mask]
            print(f"      帧#{t}: 置信度范围 [{conf_valid.min():.3f}, {conf_valid.max():.3f}], "
                  f"均值={conf_valid.mean():.3f}, 中位数={np.median(conf_valid):.3f}, "
                  f"有效点数={valid_mask.sum()}", flush=True)
        
        # 使用动态阈值：50%分位数，但不低于0.1
        # 注意用>=而非>，避免当中位数=最小值时排除大量点
        if valid_mask.any():
            conf_valid = conf_frame[valid_mask]
            threshold = max(np.percentile(conf_valid, 50), 0.1)
            conf_mask = conf_frame >= threshold
            # 如果阈值过滤后点太少(<100)，降低到25%分位数
            if conf_mask.sum() < 100:
                threshold = max(np.percentile(conf_valid, 25), 0.1)
                conf_mask = conf_frame >= threshold
        else:
            conf_mask = conf_frame >= 0.1
        
        pts = world_points[t][conf_mask]
        if len(pts) > 0:
            ds = _voxel_downsample(pts, voxel_size=0.05)
            all_points.append(ds)
            print(f"      → 提取 {len(ds)} 个点 (阈值={threshold:.3f})", flush=True)

    if not all_points:
        print("   ⚠️  无有效点云数据，跳过补充检测", flush=True)
        return []

    all_points = np.vstack(all_points).astype(np.float32)
    print(f"   📊 降采样后点云: {all_points.shape[0]} 个点", flush=True)

    # ---- 2. 排除已知物体附近的点 ----
    if known_centroids:
        known_arr = np.array(known_centroids)
        dists = np.min(np.linalg.norm(all_points[:, None, :] - known_arr[None, :, :], axis=2), axis=1)
        far_mask = dists > 0.3
        all_points = all_points[far_mask]
        print(f"   排除已知物体附近点后: {all_points.shape[0]} 个点", flush=True)

    if len(all_points) < 100:
        print("   ⚠️  排除后点云过少，跳过补充检测", flush=True)
        return []

    # ---- 3. 排除floor/wall区域的点 ----
    keyframe_floor_masks = sam_results.get('keyframe_floor_masks', {}) if sam_results else {}
    if keyframe_floor_masks:
        floor_3d_points = set()
        # 原为静默 pass: 任一帧失败都无日志。若所有帧都失败, floor_3d_points
        # 为空 → 走else 分支 → 地面点云剔除逻辑整体失效且完全无声。
        _floor_fail = 0
        _floor_first_err = None
        for frame_idx, floor_mask in keyframe_floor_masks.items():
            if frame_idx < T:
                try:
                    pointmap = world_points[frame_idx]
                    # resize mask到pointmap尺寸，避免维度不匹配
                    floor_mask_pm = _resize_mask_to_pointmap(floor_mask, pointmap)
                    fp = pointmap[floor_mask_pm]
                    fp_ds = _voxel_downsample(fp, voxel_size=0.1)
                    for p in fp_ds:
                        floor_3d_points.add(tuple(np.floor(p / 0.1).astype(int)))
                except Exception as _e:
                    _floor_fail += 1
                    _floor_first_err = _floor_first_err or f"frame {frame_idx}: {type(_e).__name__}: {_e}"
        if _floor_fail:
            print(f"   ⚠️ [floor_3d_points] {len(keyframe_floor_masks)} 帧中 {_floor_fail} 帧提取失败"
                  f" (首个: {_floor_first_err})", flush=True)
        if not floor_3d_points and keyframe_floor_masks:
            print(f"   ⚠️ [floor_3d_points] 全部帧均未提取到地面点 → 地面剔除将跳过", flush=True)

        if floor_3d_points:
            point_voxels = set(tuple(row) for row in np.floor(all_points / 0.1).astype(int))
            non_floor_voxels = point_voxels - floor_3d_points
            keep_mask = np.array([tuple(row) in non_floor_voxels
                                   for row in np.floor(all_points / 0.1).astype(int)])
            all_points = all_points[keep_mask]
            print(f"   排除floor区域点后: {all_points.shape[0]} 个点", flush=True)

    if len(all_points) < 100:
        print("   ⚠️  排除floor后点云过少，跳过补充检测", flush=True)
        return []

    # ---- 4. DBSCAN聚类 ----
    try:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.10, min_samples=20).fit(all_points)
        labels = clustering.labels_
    except ImportError:
        print("   ⚠️  sklearn不可用，使用简单体素聚类", flush=True)
        voxel_idx = np.floor(all_points / 0.1).astype(np.int64)
        voxel_keys = voxel_idx[:, 0] * 1000000 + voxel_idx[:, 1] * 1000 + voxel_idx[:, 2]
        unique_keys, inverse = np.unique(voxel_keys, return_inverse=True)
        labels = inverse

    unique_labels = set(labels) - {-1}
    print(f"   🔮 DBSCAN聚类: {len(unique_labels)} 个聚类 (eps=0.10, min_samples=20, 阈值降低可发现小物体)", flush=True)

    # ---- 5. 过滤聚类 ----
    candidate_clusters = []
    for label in unique_labels:
        cluster_pts = all_points[labels == label]
        if len(cluster_pts) < 30:
            continue

        centroid = np.mean(cluster_pts, axis=0)
        bbox_extent = np.ptp(cluster_pts, axis=0)
        volume = float(np.prod(bbox_extent))

        if volume < 0.002:
            continue

        if volume > 8.0:
            continue

        sorted_ext = sorted(bbox_extent, reverse=True)
        if sorted_ext[2] < 0.03 and sorted_ext[0] > 1.5:
            continue

        min_dim = min(bbox_extent)
        max_dim = max(bbox_extent)
        if min_dim > 0.01 and max_dim / min_dim > 30:
            continue

        candidate_clusters.append({
            'centroid': centroid,
            'bbox_extent': bbox_extent,
            'volume': volume,
            'point_count': len(cluster_pts),
        })

    candidate_clusters.sort(key=lambda x: x['volume'], reverse=True)
    print(f"   ✅ 过滤后候选聚类: {len(candidate_clusters)} 个 (阈值降低: vol<0.002, pts<30)", flush=True)

    if not candidate_clusters:
        print("   无显著遗漏聚类，跳过补充检测", flush=True)
        return []

    for i, c in enumerate(candidate_clusters[:10]):
        print(f"      候选{i+1}: 质心={c['centroid'].round(2)}, "
              f"尺寸={c['bbox_extent'].round(2)}, 体积={c['volume']:.3f}, "
              f"点数={c['point_count']}", flush=True)

    # ---- 6. VLM识别 ----
    new_objects = []
    max_candidates = min(8, len(candidate_clusters))

    for ci, cluster in enumerate(candidate_clusters[:max_candidates]):
        centroid = cluster['centroid']
        print(f"\n   [{ci+1}/{max_candidates}] 识别候选聚类 (质心={centroid.round(2)})...", flush=True)

        best_frame = None
        best_score = -1
        best_uv = None

        for vid_idx, frame_path in frame_paths_with_indices:
            slot = _vid2slot(vid_idx)
            if slot >= T:
                continue
            wp_frame = world_points[slot]
            conf_frame = world_points_conf[slot]
            valid_mask = conf_frame > 0.5
            if not np.any(valid_mask):
                continue

            u, v = _project_3d_to_2d(centroid, wp_frame)
            img_w, img_h = W, H

            margin = SUPPLEMENTARY_CROP_SIZE // 2
            if u < margin or u > img_w - margin or v < margin or v > img_h - margin:
                continue

            local_dist = np.linalg.norm(wp_frame[v, u] - centroid)
            if local_dist > 0.5:
                continue

            score = 1.0 / (local_dist + 0.01)
            if score > best_score:
                best_score = score
                best_frame = (vid_idx, frame_path)
                best_uv = (u, v)

        if best_frame is None:
            print(f"      ⚠️  无合适关键帧可投影，跳过", flush=True)
            continue

        vid_idx, frame_path = best_frame
        u, v = best_uv

        try:
            image = Image.open(frame_path).convert("RGB")
            # u,v 是 pointmap (VGGT 帧 H,W) 空间的坐标; 裁剪的是原分辨率关键帧,
            # 必须按尺寸比例缩放, 否则裁剪位置偏 ~2.8 倍 (实测 1920x1080 vs 688x384)
            _sxu = image.width / float(W)
            _syv = image.height / float(H)
            u_img, v_img = u * _sxu, v * _syv
            half = SUPPLEMENTARY_CROP_SIZE // 2
            crop_box = (int(u_img - half), int(v_img - half), int(u_img + half), int(v_img + half))
            crop_box = (
                max(0, crop_box[0]),
                max(0, crop_box[1]),
                min(image.width, crop_box[2]),
                min(image.height, crop_box[3]),
            )
            cropped = image.crop(crop_box)

            output_text = vlm_inference(cropped, model, processor, SUPPLEMENTARY_VLM_PROMPT, max_new_tokens=256)
            json_str = _extract_json_from_text(output_text)

            if json_str is None:
                print(f"      ⚠️  VLM输出无法解析: {output_text[:80]}", flush=True)
                continue

            result = json.loads(json_str)
            obj_name = result.get("name", "none") if isinstance(result, dict) else "none"

            if obj_name.lower() in ("none", "null", "n/a", "nothing", "unknown", ""):
                print(f"      ❌ VLM未识别到物体", flush=True)
                continue

            std_name = merge_synonyms(obj_name.lower().strip().replace(" ", "_"))
            if std_name in FILTER_CATEGORIES:
                print(f"      ❌ 识别为场景结构: {std_name}，跳过", flush=True)
                continue

            already_detected = any(merge_synonyms(o['name']) == std_name for o in unique_objects)
            if already_detected:
                print(f"      ❌ 已存在同名物体: {std_name}，跳过", flush=True)
                continue

            new_objects.append({
                'name': std_name,
                'centroid': centroid.tolist(),
                'source': 'pointcloud_supplementary',
                'volume': cluster['volume'],
                'point_count': cluster['point_count'],
            })
            print(f"      ✅ 新发现: {std_name} (体积={cluster['volume']:.3f}, "
                  f"点数={cluster['point_count']})", flush=True)

        except Exception as e:
            print(f"      ❌ VLM识别失败: {e}", flush=True)

    if new_objects:
        print(f"\n   🎉 补充检测发现 {len(new_objects)} 个新物体:", flush=True)
        for obj in new_objects:
            print(f"      - {obj['name']} (质心={[round(c, 2) for c in obj['centroid']]})", flush=True)
    else:
        print(f"\n   📋 补充检测未发现新物体", flush=True)

    return new_objects


# ============================================================
# Step 6: SAM 分割 floor 和 wall
def _resize_mask_to_pointmap(mask, pointmap):
    """将SAM mask resize到与VGGT pointmap相同的尺寸

    SAM对原图分割的mask尺寸(如1080xW)与VGGT输出的pointmap尺寸(如518xW)不同，
    直接用mask索引pointmap会导致boolean index维度不匹配错误。

    参数:
        mask: numpy (H_img, W_img) bool数组，SAM分割的mask
        pointmap: numpy (H_pm, W_pm, 3) 数组，VGGT点云
    返回:
        numpy (H_pm, W_pm) bool数组，resize后的mask
    """
    target_h, target_w = pointmap.shape[:2]
    if mask.shape[0] == target_h and mask.shape[1] == target_w:
        return mask
    # 用PIL resize：先转uint8再resize再转回bool
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
    mask_resized = mask_img.resize((target_w, target_h), Image.NEAREST)
    return np.array(mask_resized) > 127


# ============================================================
# SAM floor/wall分割
# ============================================================

def segment_floor_and_wall_sam(frame_paths_with_indices, vggt_results,
                               total_video_frames=0):
    """
    使用SAM3分割floor和wall（复用object_segmentation.py）

    参数:
        frame_paths_with_indices: 关键帧路径列表
        total_video_frames: 原视频总帧数 (帧号→VGGT槽位换算用, 与 Step5.5 同因)
        vggt_results: VGGT重建结果
    返回:
        {'has_floor': bool/None, 'has_wall': bool/None,
         'floor_masks': list, 'wall_masks': list,
         'keyframe_floor_masks': {frame_idx: numpy(H,W)} }
    """
    print(f"\n{'='*70}", flush=True)
    print(f"🧱 Step 6: SAM分割Floor和Wall", flush=True)
    print(f"{'='*70}\n", flush=True)

    try:
        from src.models import load_sam3_image_model
        from src.object_segmentation import segment_wall_and_floor
    except ImportError as e:
        print(f"   ⚠️  SAM3 模块不可用: {e}", flush=True)
        print(f"   ⚠️  回退到VLM视觉判断floor/wall", flush=True)
        return {'has_floor': None, 'has_wall': None, 'floor_masks': [], 'wall_masks': [], 'keyframe_floor_masks': {}}

    colors = vggt_results.get('colors')
    if colors is None:
        print(f"   ⚠️  VGGT结果中无colors数据，尝试从帧图像构建", flush=True)
        images = []
        _img_fail = 0
        _img_first_err = None
        for vid_idx, frame_path in frame_paths_with_indices:
            try:
                img = np.array(Image.open(frame_path).convert("RGB"))
                images.append(img)
            except Exception as _e:
                # 原为静默 pass: 读图全失败时返回空 colors, Stage6 材质标注
                # 拿不到颜色却不会报错, 表现为"材质标注结果异常但无原因"。
                _img_fail += 1
                _img_first_err = _img_first_err or f"{os.path.basename(str(frame_path))}: {type(_e).__name__}"
        if _img_fail:
            print(f"   ⚠️ [colors] {len(frame_paths_with_indices)} 帧中 {_img_fail} 帧读取失败"
                  f" (首个: {_img_first_err})", flush=True)
        if not images:
            return {'has_floor': None, 'has_wall': None, 'floor_masks': [], 'wall_masks': [], 'keyframe_floor_masks': {}}
        colors = np.array(images)

    if colors.ndim == 4 and colors.shape[0] > 10:
        step = max(1, colors.shape[0] // 10)
        colors_subset = colors[::step]
    else:
        colors_subset = colors

    print(f"   📦 加载 SAM3 模型...", flush=True)
    try:
        sam3_model = load_sam3_image_model()
    except Exception as e:
        print(f"   ⚠️  SAM3 模型加载失败: {e}", flush=True)
        print(f"   ⚠️  回退到VLM视觉判断floor/wall", flush=True)
        return {'has_floor': None, 'has_wall': None, 'floor_masks': [], 'wall_masks': [], 'keyframe_floor_masks': {}}

    print(f"   🔍 SAM3分割中 ({len(colors_subset)} 帧)...", flush=True)
    try:
        wall_masks, floor_masks = segment_wall_and_floor(colors_subset, sam3_model)
    except Exception as e:
        print(f"   ⚠️  SAM3分割失败: {e}", flush=True)
        del sam3_model
        torch.cuda.empty_cache()
        return {'has_floor': None, 'has_wall': None, 'floor_masks': [], 'wall_masks': [], 'keyframe_floor_masks': {}}

    has_floor = len(floor_masks) > 0
    has_wall = len(wall_masks) > 0

    print(f"   ✅ VGGT帧分割完成: floor={'✅' if has_floor else '❌'} ({len(floor_masks)} 个mask), "
          f"wall={'✅' if has_wall else '❌'} ({len(wall_masks)} 个mask)", flush=True)

    keyframe_floor_masks = {}
    print(f"   🔍 SAM3关键帧floor分割中 ({len(frame_paths_with_indices)} 帧)...", flush=True)

    world_points = vggt_results.get('world_points')
    if world_points is not None:
        world_points = world_points[0]

    for vid_idx, frame_path in frame_paths_with_indices:
        try:
            img = np.array(Image.open(frame_path).convert("RGB"))
            _, kf_floor_masks = segment_wall_and_floor(np.array([img]), sam3_model)
            if not kf_floor_masks:
                continue

            if len(kf_floor_masks) == 1:
                m = np.atleast_2d(kf_floor_masks[0]['mask'])
                if m.ndim == 3:
                    m = m[0]
                chosen_mask = m.astype(bool)
                if np.sum(chosen_mask) > 500:
                    keyframe_floor_masks[vid_idx] = chosen_mask
                continue

            # frame_paths_with_indices 的帧号是原视频帧号 (total>T 时),
            # 索引 world_points 前必须换算成 VGGT 槽位 (bug#1 同族)
            _slot = _video_frame_to_slot(vid_idx, total_video_frames,
                                         world_points.shape[0])
            if world_points is not None and _slot < world_points.shape[0]:
                from src.geometry_utils import get_plane_info
                pointmap = world_points[_slot]
                plane_infos = []
                for fm in kf_floor_masks:
                    if fm['frame_id'] != 0:
                        continue
                    m = np.atleast_2d(fm['mask'])
                    if m.ndim == 3:
                        m = m[0]
                    mask = m.astype(bool)
                    if np.sum(mask) <= 500:
                        continue
                    # resize mask到pointmap尺寸，避免维度不匹配
                    mask_pm = _resize_mask_to_pointmap(mask, pointmap)
                    info = get_plane_info(pointmap, mask_pm)
                    if info['mean_distance'] < 0.02:
                        plane_infos.append((mask, info))

                if len(plane_infos) == 0:
                    combined = np.zeros(img.shape[:2], dtype=bool)
                    for fm in kf_floor_masks:
                        if fm['frame_id'] == 0:
                            m = np.atleast_2d(fm['mask'])
                            if m.ndim == 3:
                                m = m[0]
                            combined |= m.astype(bool)
                    if np.sum(combined) > 500:
                        keyframe_floor_masks[vid_idx] = combined
                    continue

                if len(plane_infos) == 1:
                    keyframe_floor_masks[vid_idx] = plane_infos[0][0]
                    continue

                mean_normal = np.mean([info['normal'] for _, info in plane_infos], axis=0)
                mean_normal = mean_normal / np.linalg.norm(mean_normal)
                valid = [(m, info) for m, info in plane_infos
                         if abs(np.dot(info['normal'], mean_normal)) > np.cos(np.radians(30))]

                if valid:
                    chosen_mask, chosen_info = max(valid, key=lambda x: x[1]['area'])
                else:
                    chosen_mask, chosen_info = max(plane_infos, key=lambda x: x[1]['area'])

                keyframe_floor_masks[vid_idx] = chosen_mask
                print(f"      帧#{vid_idx}: {len(kf_floor_masks)} 个候选 → 选择面积最大+法向量一致的mask "
                      f"(面积={chosen_info['area']:.2f}, 拟合误差={chosen_info['mean_distance']:.4f})", flush=True)
            else:
                combined = np.zeros(img.shape[:2], dtype=bool)
                for fm in kf_floor_masks:
                    if fm['frame_id'] == 0:
                        combined |= fm['mask'].astype(bool)
                if np.sum(combined) > 500:
                    keyframe_floor_masks[vid_idx] = combined
        except Exception as e:
            print(f"      ⚠️  帧#{vid_idx} SAM floor分割失败: {e}", flush=True)

    print(f"   ✅ 关键帧floor分割完成: {len(keyframe_floor_masks)}/{len(frame_paths_with_indices)} 帧检测到floor", flush=True)

    del sam3_model
    torch.cuda.empty_cache()

    print(f"   📊 SAM3总结果: floor={'✅' if has_floor else '❌'}, wall={'✅' if has_wall else '❌'}, "
          f"关键帧floor masks: {len(keyframe_floor_masks)}\n", flush=True)

    return {
        'has_floor': has_floor,
        'has_wall': has_wall,
        'floor_masks': floor_masks,
        'wall_masks': wall_masks,
        'keyframe_floor_masks': keyframe_floor_masks,
    }


# ============================================================
# SAM floor重叠检测
# ============================================================

def _check_object_on_floor(bbox, floor_mask, threshold=FLOOR_OVERLAP_THRESHOLD):
    """
    检查物体bbox底部是否与SAM floor mask重叠

    判定逻辑: 取物体bbox底部30%区域，计算该区域与floor mask的重叠比例
    若重叠比例 >= threshold，则判定物体在地板上

    参数:
        bbox: [x1, y1, x2, y2] 物体边界框（像素坐标）
        floor_mask: numpy数组 (H, W)，bool类型，True表示floor区域
        threshold: 重叠比例阈值，默认0.3
    返回:
        (is_on_floor, overlap_ratio): 是否在地板上 + 重叠比例
    """
    if floor_mask is None or bbox is None:
        return False, 0.0

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = floor_mask.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return False, 0.0

    bbox_height = y2 - y1
    bottom_height = max(1, int(0.3 * bbox_height))
    bottom_y1 = y2 - bottom_height

    bottom_region = floor_mask[bottom_y1:y2, x1:x2]
    bottom_area = bottom_region.size
    if bottom_area == 0:
        return False, 0.0

    floor_pixels = np.sum(bottom_region > 0)
    overlap_ratio = float(floor_pixels) / float(bottom_area)

    return overlap_ratio >= threshold, overlap_ratio


def _sam_prejudge_floor(all_detections, keyframe_floor_masks):
    """
    使用SAM floor mask预判断哪些物体在地板上

    对每个关键帧中检测到的物体，检查其bbox底部是否与floor mask重叠
    若任一帧中重叠比例 >= 阈值，则判定该物体在地板上

    参数:
        all_detections: Step 3的检测结果 [{frame_idx, objects: [{name, bbox}]}]
        keyframe_floor_masks: {frame_idx: numpy(H,W)} SAM关键帧floor mask
    返回:
        sam_floor_objects: set of std_name，SAM判定在地板上的物体
        sam_floor_details: {std_name: [(frame_idx, overlap_ratio)]} 详细信息
    """
    sam_floor_objects = set()
    sam_floor_details = defaultdict(list)

    for detection in all_detections:
        frame_idx = detection["frame_idx"]
        if frame_idx not in keyframe_floor_masks:
            continue

        floor_mask = keyframe_floor_masks[frame_idx]

        for obj in detection["objects"]:
            if "name" not in obj:
                continue
            std_name = merge_synonyms(obj["name"])
            if std_name in FILTER_CATEGORIES:
                continue

            bbox = obj.get("bbox")
            if bbox is None:
                continue

            is_on_floor, overlap_ratio = _check_object_on_floor(bbox, floor_mask)
            sam_floor_details[std_name].append((frame_idx, overlap_ratio))

            if is_on_floor:
                sam_floor_objects.add(std_name)

    return sam_floor_objects, dict(sam_floor_details)


# ============================================================
# Step 7: 第二次VLM调用 — 关系判断（独立prompt + 后处理纠错）
# ============================================================

def _post_process_relationships(relationships, object_names):
    """
    关系判断后处理纠错（参考SimRecon post_process_objects）

    基于物理常识规则修正VLM可能犯的错误:
      1. 过滤掉floor/wall本身的类别
      2. 柜子/桌子等不应attached to wall → 修正为supported by floor
      3. 画/镜子等不应supported by floor → 修正为attached to wall
      4. 门/窗等不应supported by floor → 修正为embedded in wall
      5. 确保所有物体都有关系判断

    参数:
        relationships: VLM输出的关系字典 {name: relationship}
        object_names: 物体名称列表（用于兜底）
    返回:
        纠错后的关系字典 {std_name: relationship}
    """
    corrected = {}

    for obj_name, rel in relationships.items():
        name = obj_name.lower().strip().replace(" ", "_")
        std_name = merge_synonyms(name)

        if std_name in FILTER_CATEGORIES:
            print(f"      [过滤] 跳过 '{std_name}': 属于floor/wall类别", flush=True)
            continue

        if rel not in VALID_SPATIAL_RELATIONSHIPS:
            # 兜底用 "supported by other objects" 而非 floor: prompt 的默认
            # 已改成 other objects (见 build_relationship_prompt 处的注释 ——
            # 全部 floor 会让 mainv3 判 has_inter_object=False 而整段跳过 5.2),
            # 代码级兜底必须与之一致, 否则 garbage 输出会把场景拖回全 floor。
            rel = "supported by other objects"

        category_base = std_name.split("_")[0] if "_" in std_name else std_name

        if std_name in MUST_BE_FLOOR_SUPPORTED or category_base in MUST_BE_FLOOR_SUPPORTED:
            if rel in ("attached to wall", "embedded in wall"):
                print(f"      [纠错] '{std_name}': {rel} → supported by floor (物理常识)", flush=True)
                rel = "supported by floor"

        if std_name in MUST_BE_WALL_ATTACHED or category_base in MUST_BE_WALL_ATTACHED:
            if rel == "supported by floor":
                if category_base in {'picture', 'painting', 'photo', 'mirror', 'clock', 'poster'}:
                    print(f"      [纠错] '{std_name}': supported by floor → attached to wall (物理常识)", flush=True)
                    rel = "attached to wall"
                elif category_base in {'window', 'door', 'vent', 'air_vent', 'outlet', 'socket'}:
                    print(f"      [纠错] '{std_name}': supported by floor → embedded in wall (物理常识)", flush=True)
                    rel = "embedded in wall"

        corrected[std_name] = rel

    for name in object_names:
        std_name = merge_synonyms(name)
        if std_name not in corrected:
            corrected[std_name] = "supported by other objects"
            print(f"      [兜底] '{std_name}': 无VLM判断，默认 supported by other objects", flush=True)

    return corrected


def vlm_judge_relationships(unique_objects, sam_results, frame_paths_with_indices, model, processor, all_detections=None):
    """
    第二次VLM调用: 使用独立的prompt判断物体空间关系

    流程:
      1. SAM floor预判断: 用SAM floor mask检测哪些物体在地板上（重叠>=30%）
      2. 构建per-frame可见性映射: 只对每帧中实际出现的物体调用VLM
      3. 对每个关键帧分别调用VLM（仅判断该帧可见的物体）
      4. 每帧结果经过后处理纠错
      5. 多帧投票取最终结果
      6. SAM预判断的物体直接标记为"supported by floor"

    参数:
        unique_objects: 去重后的物体列表
        sam_results: SAM分割结果 {'has_floor', 'has_wall', 'keyframe_floor_masks', ...}
        frame_paths_with_indices: 关键帧路径列表
        model: VLM模型
        processor: VLM处理器
        all_detections: Step 3的检测结果（用于per-frame可见性）
    返回:
        {物体名称: 关系} 字典
    """
    print(f"\n{'='*70}", flush=True)
    print(f"🔗 Step 7: 第二次VLM调用 — 关系判断", flush=True)
    print(f"{'='*70}\n", flush=True)

    object_names_raw = [obj["name"] for obj in unique_objects]
    object_names = [n for n in object_names_raw if merge_synonyms(n) not in FILTER_CATEGORIES]

    has_floor = sam_results['has_floor']
    has_wall = sam_results['has_wall']

    if has_floor is None and has_wall is None:
        print(f"   ⚠️  SAM未提供floor/wall信息，默认floor和wall均可见", flush=True)
        has_floor = True
        has_wall = True

    filtered = set(object_names_raw) - set(object_names)
    if filtered:
        print(f"   🏗️  场景结构物体（不参与关系判断）: {filtered}", flush=True)
    print(f"   待判断物体列表: {object_names}", flush=True)
    print(f"   Floor可见: {has_floor}, Wall可见: {has_wall}", flush=True)

    if not object_names:
        print(f"   ⚠️  无需判断的物体（全部为场景结构）", flush=True)
        return {}

    # ---- 7.1 SAM floor预判断 ----
    keyframe_floor_masks = sam_results.get('keyframe_floor_masks', {})
    sam_floor_objects = set()
    sam_floor_details = {}

    if keyframe_floor_masks and all_detections:
        print(f"\n   🏠 7.1 SAM floor预判断...", flush=True)
        sam_floor_objects, sam_floor_details = _sam_prejudge_floor(all_detections, keyframe_floor_masks)

        for std_name, details in sorted(sam_floor_details.items()):
            max_overlap = max(r for _, r in details)
            is_floor = std_name in sam_floor_objects
            status = "✅ 在地板上" if is_floor else "❌ 不在地板上"
            print(f"      {std_name}: {status} (最大重叠率: {max_overlap:.1%}, 检查帧数: {len(details)})", flush=True)

        if sam_floor_objects:
            print(f"   🏠 SAM判定在地板上的物体: {sam_floor_objects}", flush=True)
        else:
            print(f"   🏠 SAM未检测到任何物体在地板上", flush=True)
    else:
        print(f"\n   ⚠️  无SAM关键帧floor mask，跳过SAM预判断", flush=True)

    # ---- 7.2 构建per-frame可见性映射 ----
    frame_visibility = defaultdict(set)
    if all_detections:
        for detection in all_detections:
            frame_idx = detection["frame_idx"]
            for obj in detection["objects"]:
                if "name" not in obj:
                    continue
                std_name = merge_synonyms(obj["name"])
                if std_name not in FILTER_CATEGORIES:
                    frame_visibility[frame_idx].add(std_name)
        print(f"\n   👁️  7.2 Per-frame可见性映射:", flush=True)
        for frame_idx, visible in sorted(frame_visibility.items()):
            print(f"      帧#{frame_idx}: {visible}", flush=True)
    else:
        print(f"\n   ⚠️  无all_detections，所有帧使用完整物体列表", flush=True)
        for vid_idx, _ in frame_paths_with_indices:
            frame_visibility[vid_idx] = set(object_names)

    # ---- 7.3 多帧VLM推理（per-frame可见性过滤） ----
    # 名字空间统一: frame_visibility 里存的是 merge_synonyms 标准名
    # ("cup"/"chair"), 而 object_names 可能是合并组的原始代表名
    # ("mug"/"chairs"/"remote control") —— 直接 in 匹配永远查不到, 物体在
    # 所有帧都被跳过查询, 最终静默兜底成 floor (下游 5.1/5.2 随之跳过)。
    # 成员判断统一走标准名, 投票与提示词仍用原始名。
    _std_of = {n: merge_synonyms(n.lower().strip().replace(" ", "_"))
               for n in object_names}
    _sam_floor_std = {merge_synonyms(s.lower().strip().replace(" ", "_"))
                      for s in sam_floor_objects}
    objects_to_query = [n for n in object_names
                        if _std_of.get(n, n) not in _sam_floor_std]

    if objects_to_query:
        print(f"\n   🤖 7.3 VLM关系判断: 需判断 {len(objects_to_query)} 个物体 (SAM已处理 {len(sam_floor_objects)} 个)", flush=True)
    else:
        print(f"\n   ✅ 所有物体已由SAM预判断，无需VLM推理", flush=True)

    all_votes = defaultdict(lambda: defaultdict(int))
    num_frames_to_query = len(frame_paths_with_indices)

    for qi, (vid_idx, frame_path) in enumerate(frame_paths_with_indices if objects_to_query else []):
        visible_in_frame = frame_visibility.get(vid_idx, set())
        frame_objects = [n for n in objects_to_query
                         if _std_of.get(n, n) in visible_in_frame]

        if not frame_objects:
            print(f"   [{qi+1}/{num_frames_to_query}] 帧 #{vid_idx}: 无待判断物体可见，跳过", flush=True)
            continue

        print(f"   [{qi+1}/{num_frames_to_query}] 帧 #{vid_idx}: 可见物体 {frame_objects}...", end=" ", flush=True)

        try:
            image = Image.open(frame_path).convert("RGB")
            start_time = time.time()
            frame_prompt = _build_relationship_prompt(frame_objects, has_floor, has_wall)
            output_text = vlm_inference(image, model, processor, frame_prompt, max_new_tokens=1024)
            elapsed = time.time() - start_time

            json_str = _extract_json_from_text(output_text)

            if json_str is None:
                print(f"❌ 无法提取JSON ({elapsed:.1f}s)")
                print(f"      📝 VLM原始输出 (前300字符): {output_text[:300]}", flush=True)
                continue

            try:
                result = json.loads(json_str)
            except json.JSONDecodeError as e1:
                json_str_fixed = _fix_json_string(json_str)
                try:
                    result = json.loads(json_str_fixed)
                except json.JSONDecodeError as e2:
                    print(f"❌ JSON解析失败 ({elapsed:.1f}s)")
                    print(f"      🔍 提取的JSON字符串 (前500字符): {json_str[:500]}", flush=True)
                    print(f"      🔧 修复后JSON (前500字符): {json_str_fixed[:500]}", flush=True)
                    print(f"      ⚠️  原始错误: {str(e1)[:200]}", flush=True)
                    print(f"      ⚠️  修复后错误: {str(e2)[:200]}", flush=True)
                    print(f"      📝 VLM完整输出:\n{output_text}", flush=True)
                    continue

            raw_rels = result.get("relationships", {})

            if isinstance(raw_rels, dict):
                frame_corrected = _post_process_relationships(raw_rels, frame_objects)
                for obj_name, category in frame_corrected.items():
                    if category in VALID_SPATIAL_RELATIONSHIPS:
                        all_votes[obj_name][category] += 1
                print(f"✅ {len(raw_rels)} 条关系 → {len(frame_corrected)} 个物体 ({elapsed:.1f}s)", flush=True)
            elif isinstance(raw_rels, list):
                rel_dict = {}
                for item in raw_rels:
                    if isinstance(item, dict) and "name" in item:
                        name_key = item["name"]
                        if "relationship" in item:
                            rel_dict[name_key] = item["relationship"]
                        elif "relation" in item:
                            rel_dict[name_key] = item["relation"]
                if rel_dict:
                    frame_corrected = _post_process_relationships(rel_dict, frame_objects)
                    for obj_name, category in frame_corrected.items():
                        if category in VALID_SPATIAL_RELATIONSHIPS:
                            all_votes[obj_name][category] += 1
                    print(f"✅ {len(raw_rels)} 条关系 → {len(frame_corrected)} 个物体 ({elapsed:.1f}s)", flush=True)
                else:
                    print(f"❌ 无法从list格式中提取关系 ({elapsed:.1f}s)", flush=True)

        except Exception as e:
            print(f"❌ 错误: {e}", flush=True)

    # ---- 7.4 汇总结果: SAM预判断 + VLM投票 ----
    print(f"\n   📊 7.4 汇总结果:", flush=True)
    final_objects = {}
    for obj in unique_objects:
        name = obj["name"]
        std_name = merge_synonyms(name)

        if std_name in FILTER_CATEGORIES:
            continue

        if std_name in sam_floor_objects:
            final_objects[std_name] = "supported by floor"
            details = sam_floor_details.get(std_name, [])
            max_overlap = max(r for _, r in details) if details else 0
            print(f"   🏠 {std_name}: supported by floor (SAM预判断, 最大重叠率: {max_overlap:.1%})", flush=True)
        elif std_name in all_votes and all_votes[std_name]:
            best_rel = max(all_votes[std_name], key=all_votes[std_name].get)
            final_objects[std_name] = best_rel
            vote_info = dict(all_votes[std_name])
            print(f"   📋 {std_name}: {best_rel} (VLM投票: {vote_info})", flush=True)
        else:
            final_objects[std_name] = "supported by other objects"
            print(f"   ⚠️  {std_name}: supported by other objects (无SAM判断也无VLM投票，兜底默认)", flush=True)

    print(f"\n📊 关系判断完成，{len(final_objects)} 个物体\n", flush=True)
    return final_objects


# ============================================================
# Step 8: 输出场景JSON
# ============================================================

# 人体部位 / 操作者身体部分。
# VLM_DETECT_PROMPT 里已写了 "Ignore: hands, body parts, walls, floors,
# ceilings", 但实测靠不住: 86.mp4 (人在桌前操作食物) 仍然输出了 "arm",
# 而 156.mp4 (静态摆物的场景) 没有 —— 说明 VLM 会把画面中突出的肢体当成物体。
# 放任的后果不只是多生成一个模型: 手臂是「操作者」, 会被 pick_place_detector
# 判成抓取事件 (86 的 arm_inst1 被判出 4 次 pick/place), 还会污染支撑关系
# (plate_0 → supported by arm_1, 实际水平重叠 0%)。
# 因此这里在 Stage1 的唯一出口统一拦掉, 不依赖 VLM 自觉。
BODY_PART_CATEGORIES = {
    # 上肢
    'arm', 'hand', 'finger', 'thumb', 'fist', 'palm', 'wrist', 'forearm', 'elbow',
    'shoulder',
    # 下肢
    'leg', 'foot', 'toe', 'knee', 'ankle', 'thigh', 'hip',
    # 躯干 / 头
    'body', 'torso', 'head', 'face', 'neck', 'hair', 'chest', 'back', 'belly',
    'waist',
    # 感官
    'eye', 'ear', 'nose', 'mouth', 'lip', 'teeth', 'tongue',
    # 人本身
    'person', 'people', 'human', 'man', 'woman', 'child', 'kid', 'baby', 'boy',
    'girl', 'guy', 'lady',
}


def _is_body_part(category_name):
    """判断类别名是否为人体部位 / 操作者身体部分。

    会同时处理带实例后缀的形式 (arm_0 → arm)。故意用「精确匹配」而不是子串匹配,
    避免误伤 armchair / chair / handle / footstool / alarm 这些合法类别。
    """
    if not category_name:
        return False
    name = str(category_name).lower().strip()
    if name in BODY_PART_CATEGORIES:
        return True
    # 去掉 _<数字> 实例后缀再判一次
    head, sep, tail = name.rpartition('_')
    if sep and tail.isdigit():
        return head in BODY_PART_CATEGORIES
    return False


def generate_scene_json(category_registry, output_path):
    """
    将最终的场景类别-关系映射保存为JSON文件

    参数:
        category_registry: {物体名称: 关系} 字典
        output_path: 输出JSON文件路径
    """
    print(f"\n{'='*70}", flush=True)
    print(f"💾 Step 8: 保存场景JSON", flush=True)
    print(f"{'='*70}\n", flush=True)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # 过滤人体部位 (说明见 BODY_PART_CATEGORIES 处注释)。
    # 兜底: 若过滤后什么都不剩, 说明这个场景几乎全是肢体 (极端情况), 保留原清单
    # 而不是输出空场景 —— 空场景会让后续 Stage 无物体可处理。
    body_parts = sorted(n for n in category_registry if _is_body_part(n))
    if body_parts:
        print(f"   🚫 过滤人体部位 {body_parts} (VLM 未遵守 'Ignore: hands, body parts')",
              flush=True)
    filtered = {k: v for k, v in category_registry.items() if not _is_body_part(k)}
    if not filtered:
        print("   ⚠️ 过滤后无剩余物体, 保留原清单", flush=True)
        filtered = dict(category_registry)

    sorted_registry = dict(sorted(filtered.items()))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_registry, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON 已保存: {output_path}", flush=True)
    print(f"\n📦 最终类别清单 ({len(sorted_registry)} 个):", flush=True)
    for i, (name, rel) in enumerate(sorted_registry.items(), 1):
        print(f"   {i:2d}. {name:20s} → {rel}", flush=True)


# ============================================================
# VLM 模型加载
# ============================================================

def load_vlm_model(checkpoint_path):
    """
    加载VLM模型和处理器

    支持的模型:
      - Qwen3.5/Qwen3.6-27B-FP8 (AutoModelForImageTextToText)
      - Qwen2.5-VL/Qwen3-VL 系列
      - 通用 AutoModelForCausalLM 兜底

    FP8 量化模型由 transformers >= 4.45 自动识别 (config.json 含 quantization_config)

    参数:
        checkpoint_path: 模型权重路径
    返回:
        (model, processor)
    """
    print(f"📦 加载 VLM 模型: {checkpoint_path}", flush=True)
    print(f"   路径存在: {os.path.exists(checkpoint_path)}", flush=True)

    print("   [1/2] 加载 Processor...", flush=True)
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)

    print("   [2/2] 加载 Model...", flush=True)
    arch = ""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        arch = "/".join(cfg.architectures) if cfg.architectures else ""
        print(f"   架构: {arch}", flush=True)
    except Exception as e:
        print(f"   ⚠️ 读取 config 失败: {e}", flush=True)

    ModelClass = None
    if "ConditionalGeneration" in arch or "ForVision2Seq" in arch or "ForImageTextToText" in arch:
        try:
            from transformers import AutoModelForImageTextToText
            ModelClass = AutoModelForImageTextToText
            print(f"   使用 AutoModelForImageTextToText", flush=True)
        except ImportError:
            pass
    if ModelClass is None:
        try:
            from transformers import AutoModelForVision2Seq
            ModelClass = AutoModelForVision2Seq
            print(f"   使用 AutoModelForVision2Seq (兼容旧版)", flush=True)
        except ImportError:
            pass
    if ModelClass is None:
        from transformers import AutoModelForCausalLM
        ModelClass = AutoModelForCausalLM
        print(f"   兜底使用 AutoModelForCausalLM", flush=True)

    if torch.cuda.is_available():
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_budget = int(gpu_total * 0.85)
        max_memory_map = {0: f"{gpu_budget}GiB", "cpu": "200GiB"}
        print(f"   GPU 显存: {gpu_total:.1f} GiB, 分配: {gpu_budget} GiB (留15%给激活值)", flush=True)
    else:
        max_memory_map = {"cpu": "200GiB"}
        print(f"   ⚠️  无 GPU, 全部加载到 CPU (推理会很慢)", flush=True)

    os.makedirs("/tmp/vlm_offload", exist_ok=True)
    model = ModelClass.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        max_memory=max_memory_map,
        offload_folder="/tmp/vlm_offload",
        offload_state_dict=True,
    )
    model.eval()

    print(f"✅ 模型加载完成! 设备: {model.device}", flush=True)
    return model, processor


# ============================================================
# 主函数
# ============================================================

def main():
    """主流程: VGGT重建 → 采样 → 第一次VLM检测 → 去重 → SAM分割 → 第二次VLM关系判断 → 输出"""
    parser = argparse.ArgumentParser(description='ReplicateAnyScene Stage 2')
    parser.add_argument('--input_video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output_json', type=str, default=None, help='输出 JSON 路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录 (保存关键帧和元数据, 默认与output_json同目录)')
    parser.add_argument('--vlm_checkpoint', type=str, default=None, help='主VLM 模型路径 (快速模型, 用于检测)')
    parser.add_argument('--vlm_checkpoint_aux', type=str, default=None, help='辅助VLM模型路径 (高精度模型, 如32B, 用于主模型漏检物体补全, 可选)')
    parser.add_argument('--max_frames', type=int, default=12, help='贪心采样关键帧数 (默认12, 提高视角覆盖度)')
    parser.add_argument('--vggt_max_frames', type=int, default=120, help='VGGT 3D重建最大帧数 (默认120, 室内场景足够且提速25%)')
    # 临时帧目录按 PID 隔离: 硬编码路径在多进程并发(如双卡并行跑不同场景)时会被
    # 另一进程清理/覆盖, 实测报 FileNotFoundError: temp_frames_stage2/frame_000_vid1.jpg
    # (丢帧, 异常虽被捕获但可能导致漏检)。加 PID 后各进程互不干扰。
    parser.add_argument('--temp_dir', type=str,
                        default=f'./temp_frames_stage2_{os.getpid()}',
                        help='临时帧目录 (默认按 PID 隔离, 避免多进程并发时互相覆盖)')
    parser.add_argument('--centroid_dist_thre', type=float, default=0.15, help='3D去重质心距离阈值/米')
    parser.add_argument('--use_sam', type=str, default='auto', choices=['auto', 'yes', 'no'], help='SAM3 floor/wall分割')
    parser.add_argument('--supplementary_detect', action='store_true', default=True, help='启用点云补充检测（发现远端大物体）')
    parser.add_argument('--no_supplementary_detect', action='store_true', default=False, help='禁用点云补充检测')
    parser.add_argument('--enable_multi_vlm', action='store_true', default=False, help='启用多VLM联合检测 (主+辅模型, 需要同时设置vlm_checkpoint_aux)')
    parser.add_argument('--aux_min_object_threshold', type=int, default=8, help='主模型检测物体数<该值时触发辅助VLM补全 (默认8)')
    parser.add_argument('--use_api_vlm', action='store_true', default=False,
                        help='用 opencode 的视觉模型做第一次检测 (替代本地 Qwen3.5-9B)。'
                             '模型可用环境变量 RAS_STAGE1_VLM_MODEL 覆盖; '
                             'API 失败会自动回退本地模型。')

    args = parser.parse_args()
    # 命令行开关优先于环境变量
    if getattr(args, 'use_api_vlm', False):
        globals()['USE_API_VLM'] = True
    if getattr(args, 'stage1_vlm_model', None):
        globals()['API_VLM_MODEL'] = args.stage1_vlm_model

    if args.no_supplementary_detect:
        args.supplementary_detect = False

    if args.output_json is None:
        video_stem = Path(args.input_video).stem
        args.output_json = f"./assets/json_configs/scene_{video_stem}_stage2.json"

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.output_json))

    if args.vlm_checkpoint is None:
        candidates_fast = [
            "/mnt/data/lza/models/Qwen3.5-9B",
            "/mnt/data/lza/models/Qwen3.5-7B",
        ]
        for cand in candidates_fast:
            if os.path.exists(cand):
                args.vlm_checkpoint = cand
                print(f"   ✅ 默认使用快速模型: {cand} (~18GB, 全部加载到GPU, 推理快)", flush=True)
                break

        if args.vlm_checkpoint is None:
            candidates_high_precision = [
                "/mnt/data/lza/models/Qwen3.6-27B-FP8",
                "/mnt/data/lza/models/Qwen3.5-32B",
                "/mnt/data/lza/models/Qwen2.5-VL-32B-Instruct",
                "/mnt/data/lza/models/Qwen3-32B",
                "/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-32B-Instruct",
            ]
            for cand in candidates_high_precision:
                if os.path.exists(cand):
                    args.vlm_checkpoint = cand
                    print(f"   ⚠️  9B不可用, 改用高精度模型: {cand} (可能需CPU offload, 速度较慢)", flush=True)
                    break

        if args.vlm_checkpoint is None:
            fallback = "/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct"
            if os.path.exists(fallback):
                snapshots_dir = os.path.join(fallback, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        args.vlm_checkpoint = os.path.join(snapshots_dir, snapshots[0])
                        print(f"   ⚠️  9B/高精度都不可用，回退到 3B: {args.vlm_checkpoint}", flush=True)

    if args.vlm_checkpoint is None:
        print("❌ 错误: 未找到 VLM 模型", flush=True)
        return

    if args.vlm_checkpoint_aux is None:
        if "9b" in args.vlm_checkpoint.lower() or "7b" in args.vlm_checkpoint.lower() or "3b" in args.vlm_checkpoint.lower():
            aux_candidates = [
                "/mnt/data/lza/models/Qwen3.6-27B-FP8",
                "/mnt/data/lza/models/Qwen3.5-32B",
                "/mnt/data/lza/models/Qwen2.5-VL-32B-Instruct",
            ]
            for cand in aux_candidates:
                if cand != args.vlm_checkpoint and os.path.exists(cand):
                    args.vlm_checkpoint_aux = cand
                    print(f"   💡 主VLM较小, 已配高精度辅助: {cand} (需 --enable_multi_vlm 触发)", flush=True)
                    break

    print("=" * 70, flush=True)
    print("🚀 ReplicateAnyScene Stage 2: VGGT引导的智能物体发现", flush=True)
    print("=" * 70, flush=True)
    print(f"📥 输入视频: {args.input_video}")
    print(f"📤 输出 JSON: {args.output_json}")
    print(f"📂 输出目录: {args.output_dir}")
    print(f"🤖 主VLM模型: {args.vlm_checkpoint}")
    if args.enable_multi_vlm:
        if args.vlm_checkpoint_aux:
            print(f"🤖 辅助VLM模型: {args.vlm_checkpoint_aux}")
        else:
            print(f"⚠️  启用多VLM但未找到辅助模型, 仅使用主模型", flush=True)
    print(f"🖼️  贪心采样帧数: {args.max_frames}")
    print(f"🖼️  VGGT重建帧数: {args.vggt_max_frames}")
    print("=" * 70 + "\n", flush=True)

    # 设置 VLM 对话日志输出目录
    set_vlm_log_dir(args.output_dir)

    try:
        # Step 0: VGGT 3D场景重建
        print(f"{'='*70}", flush=True)
        print(f"🔮 Step 0: VGGT 3D场景重建", flush=True)
        print(f"{'='*70}", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        frames = load_video_frames(args.input_video, max_frames=args.vggt_max_frames).to(device)
        print(f"   加载 {len(frames)} 帧用于VGGT重建", flush=True)

        vggt_model = load_vggt_model().to(device)
        vggt_results = vggt_predict(frames, vggt_model)

        extrinsics = vggt_results['extrinsics']
        print(f"✅ VGGT重建完成: {len(extrinsics)} 帧\n", flush=True)

        # Step 1: SimRecon采样
        world_points = vggt_results['world_points'][0]
        world_points_conf = vggt_results['world_points_conf'][0]

        keyframe_indices = extract_frames_by_vggt_sampling(
            extrinsics, world_points, world_points_conf, max_frames=args.max_frames
        )

        # ── 索引空间转换 (VGGT 采样帧下标 → 原视频帧号) ──
        # keyframe_indices 是 VGGT 采样帧(0..T_vggt-1)的下标, 而
        # extract_specific_frames 要的是原视频帧号。load_video_frames 用
        # np.linspace(0, total-1, max_frames) 抽帧, 所以必须做逆映射。
        # 实测 weixin (300帧抽80): 下标 78 被当成原视频第 78 帧(26%处,
        # 全是手+耳机盒特写), 而它真正对应第 292 帧(97%处, 广角能看到
        # 2 把椅子) → VLM 从没见过椅子, JSON 里不可能有 chair。
        _fps, _total = _get_video_info(args.input_video)
        # 用 extrinsics 的数量取 VGGT 帧数, 比 world_points.shape 可靠
        # (后者可能是 torch tensor, np.asarray 会抛错)。
        try:
            _t_vggt = len(extrinsics)
        except Exception:
            _t_vggt = 0
        if _t_vggt > 1 and _total > _t_vggt:
            _mapped = sorted({int(round(i * (_total - 1) / (_t_vggt - 1)))
                              for i in keyframe_indices})
            print(f"   🔁 关键帧索引空间转换: VGGT下标{keyframe_indices[:6]}... "
                  f"→ 原视频帧号{_mapped[:6]}... (共{len(_mapped)}个, "
                  f"VGGT {_t_vggt}帧 / 原视频 {_total} 帧)", flush=True)
            keyframe_indices = _mapped

        # Step 2: 提取关键帧
        frame_paths_with_indices = extract_specific_frames(
            args.input_video, keyframe_indices, args.temp_dir
        )

        if not frame_paths_with_indices:
            print("❌ 未能提取任何帧", flush=True)
            return

        # 清理VGGT模型
        vggt_model = unload_model(vggt_model)
        del frames
        torch.cuda.empty_cache()

        # Step 3: 第一次VLM调用 — 物体检测（仅名称+位置）
        model, processor = load_vlm_model(args.vlm_checkpoint)

        aux_model, aux_processor = None, None
        if args.enable_multi_vlm and args.vlm_checkpoint_aux and os.path.exists(args.vlm_checkpoint_aux):
            print(f"\n🤝 多VLM模式: 检测完主模型后, 将在主VLM独立物体<阈值时加载辅助VLM", flush=True)
            print(f"   主VLM: {args.vlm_checkpoint}", flush=True)
            print(f"   辅助VLM (按需): {args.vlm_checkpoint_aux}", flush=True)

        all_detections = detect_objects_in_frames(
            frame_paths_with_indices, model, processor,
            aux_model=aux_model, aux_processor=aux_processor,
            enable_multi_vlm=args.enable_multi_vlm and args.vlm_checkpoint_aux is not None,
            aux_min_objects=args.aux_min_object_threshold,
        )

        if args.enable_multi_vlm and args.vlm_checkpoint_aux and os.path.exists(args.vlm_checkpoint_aux):
            unique_count = 0
            seen = set()
            for det in all_detections:
                for obj in det["objects"]:
                    if "name" in obj:
                        std = merge_synonyms(obj["name"].lower().strip().replace(" ", "_"))
                        if std and std not in FILTER_CATEGORIES and std not in seen:
                            seen.add(std)
                            unique_count += 1
            if unique_count < args.aux_min_object_threshold:
                print(f"\n🤝 加载辅助VLM进行补全 (主模型独立类别 {unique_count} < {args.aux_min_object_threshold})", flush=True)
                del model, processor
                torch.cuda.empty_cache()

                aux_model, aux_processor = load_vlm_model(args.vlm_checkpoint_aux)
                all_detections = detect_objects_in_frames(
                    frame_paths_with_indices, aux_model, aux_processor,
                    aux_model=None, aux_processor=None,
                    enable_multi_vlm=False, aux_min_objects=0,
                )
                model, processor = aux_model, aux_processor
                print(f"\n✅ Step 7 关系判断将使用辅助VLM (高精度)", flush=True)
            else:
                print(f"\n✅ 主VLM独立类别 {unique_count} ≥ {args.aux_min_object_threshold}, 无需辅助VLM", flush=True)
        elif args.vlm_checkpoint_aux and os.path.exists(args.vlm_checkpoint_aux) \
                and args.vlm_checkpoint_aux != args.vlm_checkpoint:
            is_32b_main = "32b" in args.vlm_checkpoint.lower()
            is_9b_aux = "9b" in args.vlm_checkpoint_aux.lower() or "7b" in args.vlm_checkpoint_aux.lower()
            if is_32b_main and is_9b_aux:
                print(f"\nℹ️  当前主VLM(32B)已足够, 跳过辅助VLM(9B). 如需强制启用, 加 --enable_multi_vlm", flush=True)

        if not all_detections:
            print("❌ 未能检测到任何物体", flush=True)
            return

        # Step 4: 射线投射 → 3D位置估计
        _kf_size = None
        try:
            if frame_paths_with_indices:
                from PIL import Image as _PILImage
                with _PILImage.open(frame_paths_with_indices[0][1]) as _kf_img:
                    _kf_size = _kf_img.size  # (W, H) 原分辨率关键帧
        except Exception as _e:
            print(f"   ⚠️ 读取关键帧尺寸失败 (跳过像素空间对齐): {_e}", flush=True)
        object_instances = build_object_instances(all_detections, vggt_results,
                                                  total_video_frames=_total,
                                                  keyframe_size=_kf_size)

        if not object_instances:
            print("❌ 没有有效的物体实例", flush=True)
            return

        # Step 5: 去重（SYNONYM_MAP + CLIP语义匹配）
        clip_matcher = None
        if CLIP_AVAILABLE:
            try:
                clip_matcher = CLIPSemanticMatcher(device=device)
            except Exception as e:
                print(f"   ⚠️  CLIP加载失败: {e}，仅使用SYNONYM_MAP去重", flush=True)

        unique_objects = deduplicate_objects(
            object_instances, centroid_dist_thre=args.centroid_dist_thre,
            clip_matcher=clip_matcher
        )

        if clip_matcher is not None:
            del clip_matcher
            torch.cuda.empty_cache()

        if not unique_objects:
            print("❌ 去重后没有剩余物体", flush=True)
            return

        # Step 6: SAM分割floor/wall
        sam_results = {'has_floor': None, 'has_wall': None, 'floor_masks': [], 'wall_masks': [], 'keyframe_floor_masks': {}}
        if args.use_sam != 'no':
            try:
                sam_results = segment_floor_and_wall_sam(
                    frame_paths_with_indices, vggt_results,
                    total_video_frames=_total
                )
            except Exception as e:
                print(f"   ⚠️  SAM分割失败: {e}，回退到VLM视觉判断", flush=True)

        # Step 5.5: 点云补充检测（在SAM之后，可利用floor mask排除floor区域）
        if args.supplementary_detect:
            try:
                new_objects = supplementary_detect_from_pointcloud(
                    vggt_results, unique_objects, frame_paths_with_indices,
                    model, processor, sam_results=sam_results,
                    total_video_frames=_total
                )
                if new_objects:
                    for obj in new_objects:
                        unique_objects.append(obj)
                    print(f"\n   📋 补充检测后物体总数: {len(unique_objects)}", flush=True)
            except Exception as e:
                print(f"   ⚠️  点云补充检测失败: {e}", flush=True)

        # Step 7: 第二次VLM调用 — 关系判断（独立prompt）
        final_objects = vlm_judge_relationships(
            unique_objects, sam_results, frame_paths_with_indices, model, processor,
            all_detections=all_detections
        )

        # 清理VLM模型
        del model
        del processor
        torch.cuda.empty_cache()

        if not final_objects:
            print("❌ 关系判断后没有剩余物体", flush=True)
            return

        # Step 8: 保存JSON
        generate_scene_json(final_objects, args.output_json)

        # Step 9: 保存贪心采样关键帧 + 每帧可见物体映射到输出目录
        video_stem = Path(args.input_video).stem
        keyframes_out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "assets", "key_frames", video_stem
        )
        keyframes_out_dir = os.path.normpath(keyframes_out_dir)
        os.makedirs(keyframes_out_dir, exist_ok=True)

        import shutil
        saved_keyframes = []
        for vid_idx, frame_path in frame_paths_with_indices:
            dst_name = f"frame_vid{vid_idx}.jpg"
            dst_path = os.path.join(keyframes_out_dir, dst_name)
            shutil.copy2(frame_path, dst_path)
            saved_keyframes.append({"vid_idx": vid_idx, "path": dst_name})

        frame_visibility = {}
        for detection in all_detections:
            fidx = detection["frame_idx"]
            visible_names = list(set(
                merge_synonyms(obj["name"])
                for obj in detection["objects"]
                if "name" in obj and merge_synonyms(obj["name"]) not in FILTER_CATEGORIES
            ))
            frame_visibility[str(fidx)] = visible_names

        metadata = {
            "keyframe_indices": keyframe_indices,
            "keyframes": saved_keyframes,
            "frame_visibility": frame_visibility,
            "scene_objects": {name: rel for name, rel in final_objects.items()},
        }
        metadata_path = os.path.join(keyframes_out_dir, "keyframes_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*70}", flush=True)
        print(f"💾 Step 9: 保存贪心采样关键帧", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"   📁 关键帧目录: {keyframes_out_dir}", flush=True)
        print(f"   📋 元数据: {metadata_path}", flush=True)
        print(f"   🖼️  保存 {len(saved_keyframes)} 个关键帧", flush=True)
        print(f"   👁️  可见性映射: {len(frame_visibility)} 帧", flush=True)
        for fidx_str, names in sorted(frame_visibility.items(), key=lambda x: int(x[0])):
            print(f"      帧#{fidx_str}: {names}", flush=True)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断", flush=True)
    except Exception as e:
        import traceback
        print(f"\n❌ 错误: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
    finally:
        if os.path.exists(args.temp_dir):
            import shutil
            shutil.rmtree(args.temp_dir)
            print(f"\n🧹 已清理临时目录: {args.temp_dir}", flush=True)


if __name__ == '__main__':
    main()
