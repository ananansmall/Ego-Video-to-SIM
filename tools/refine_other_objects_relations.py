"""
ReplicateAnyScene Stage 5.1: 细化 "supported by other objects" 关系
================================================================

管线概述
--------
Stage1 (generate_scene_json_stage1.py) 通过VGGT重建+贪心采样+VLM检测，生成了
场景JSON，其中部分物体的关系被标记为 "supported by other objects"（泛化关系）。
本脚本的目标是将这些泛化关系细化为具体的支撑关系，例如：
  "bowl": "supported by other objects" → "bowl": "supported by table"

帧来源（两层，合并去重后投票）
------------------------------
1. optimal_frames/ (mainv2.py生成)
   - 路径: outputs/{scene_id}/optimal_frames/
   - 每个物体实例的最优视角帧（最大3D表面积/首次出现帧）
   - 文件名格式: {category}_inst{inst_idx}_frame{frame_id}.jpg
   - 例: bowl_inst0_frame107.jpg → bowl在帧107的最优视角

2. keyframes/ (stage1贪心采样生成)
   - 包含最优视角帧 + stage1贪心采样中选择该物体可见的帧
   - 元数据: keyframes/keyframes_metadata.json → frame_visibility 映射

补充帧: 若合并后不足5帧，从 instance_visibility (mask数据) 中按视角距离从大到小
  补充到 color/ 按需加载，优先选视角差异大的帧（更多样化）。

对每个 "other objects" 物体，按实例独立投票:

以场景232为例
-------------
输入数据:
  - 场景JSON: assets/json_configs/232.json
    7个物体: bowl, box, card, cloth, donut, table, toy
    其中5个 "supported by other objects": bowl, box, card, donut, toy
    1个 "supported by floor": table
    1个 "attached to wall": cloth

  - optimal_frames: outputs/232/optimal_frames/
    bowl_inst0_frame107.jpg   → bowl 在帧107的最优视角
    cloth_inst0_frame58.jpg   → cloth 在帧58的最优视角
    donut_inst0_frame14.jpg   → donut 帧14
    donut_inst1_frame65.jpg   → donut 帧65
    donut_inst2_frame112.jpg  → donut 帧112
    table_inst0_frame101.jpg  → table 帧101
    toy_inst0_frame108.jpg    → toy 帧108
    toy_inst1_frame69.jpg     → toy 帧69
    toy_inst2_frame64.jpg     → toy 帧64

  - keyframes: assets/key_frames/232/
    10个贪心采样关键帧 (vid0, 21, 34, 55, 76, 95, 119, 139, 140, 156)
    frame_visibility 映射:
      帧0:   [card, bowl, toy]
      帧21:  [box, bowl, toy]
      帧34:  [box, bowl, toy]
      帧55:  [card, bowl, toy]
      帧76:  [box, bowl, toy]
      帧95:  [card, donut, bowl]
      帧119: [donut, table, bowl]
      帧139: [cloth, donut, bowl]
      帧140: [card, donut, bowl]
      帧156: [card, bowl, toy]

  按实例扩展后:
    bowl_0:  optimal(inst0_frame107) + keyframes(0,21,34,55,76,95,119,139,140,156) = 11帧
    box_0:   keyframes(21,34,76) = 3帧
    card_0:  keyframes(0,55,95,140,156) = 5帧
    donut_0: optimal(inst0_frame14) + keyframes(95,119,139,140) = 5帧
    donut_1: optimal(inst1_frame65) + keyframes(95,119,139,140) = 5帧
    donut_2: optimal(inst2_frame112) + keyframes(95,119,139,140) = 5帧
    toy_0:   optimal(inst0_frame108) + keyframes(0,21,34,55,76,156) = 7帧
    toy_1:   optimal(inst1_frame69) + keyframes(0,21,34,55,76,156) = 7帧
    toy_2:   optimal(inst2_frame64) + keyframes(0,21,34,55,76,156) = 7帧

调用方式
--------
# 基本调用（自动推断路径）
python tools/refine_other_objects_relations.py \
    --stage1_json ./assets/json_configs/232.json \
    --scene_dir ./outputs/232

# 完整调用（指定所有路径）
python tools/refine_other_objects_relations.py \
    --stage1_json ./assets/json_configs/232.json \
    --output_json ./assets/json_configs/232_refined.json \
    --scene_dir ./outputs/232 \
    --optimal_frames_dir ./outputs/232/optimal_frames \
    --keyframes_dir ./assets/key_frames/232 \
    --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B

# 指定输出路径
python tools/refine_other_objects_relations.py \
    --stage1_json ./assets/json_configs/232.json \
    --output_json ./outputs/232/232_refined.json \
    --scene_dir ./outputs/232

路径自动推断规则:
  - output_json: 默认 {stage1_json同名}_refined.json
  - optimal_frames_dir: 默认 {scene_dir}/optimal_frames
  - keyframes_dir: 默认 项目根目录/assets/key_frames/{scene_id}/
    其中 scene_id 从 scene_dir 路径末尾提取

输出:
  更新后的场景JSON，"supported by other objects" 被替换为具体关系
  多实例物体扩展为独立key: toy → toy_0, toy_1, toy_2（各自独立关系）
"""

import argparse
import json
import os
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from src.vlm_logger import vlm_inference, set_vlm_log_dir

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    from transformers import AutoProcessor


VALID_RELATIONSHIPS = [
    "supported by floor",
    "supported by other objects",
    "attached to wall",
    "embedded in wall",
]

FILTER_CATEGORIES = {
    'floor', 'wall', 'ground', 'ceiling', 'floor_area', 'wall_section',
    'flooring', 'wall_surface', 'room', 'space'
}

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

MUST_BE_WALL_ATTACHED = {
    'picture', 'painting', 'photo', 'photo_frame', 'frame',
    'mirror', 'clock', 'wall_clock', 'poster', 'art',
    'whiteboard', 'blackboard', 'bulletin_board',
    'light_switch', 'outlet', 'socket',
    'curtain', 'drape', 'blind', 'shade',
}

MUST_BE_WALL_EMBEDDED = {
    'window', 'door', 'vent', 'air_vent',
}


def build_refinement_prompt(all_objects_with_rel, target_objects):
    """构建VLM推理提示词（含物理常识，适度简洁）

    给出物体列表和物理常识规则，让VLM判断每个target物体的支撑关系。
    比纯选择题多了物理常识引导，但比原始长prompt精简。

    参数:
        all_objects_with_rel: {name: current_relation} 全部物体及当前关系
        target_objects: [name, ...] 需要细化的物体列表
    返回:
        prompt文本
    """
    objects_str = "\n".join([f'  - "{name}": {rel}' for name, rel in all_objects_with_rel.items()])
    targets_str = ", ".join([f'"{t}"' for t in target_objects])
    num_targets = len(target_objects)

    prompt = f"""Look at the image. Determine what supports each target object.

**Scene objects** (name: relationship):
{objects_str}

**Target**: {targets_str}

**Valid relationships**:
- "supported by floor" — rests on ground
- "supported by <object_name>" — rests ON TOP of another object (e.g., cup on table, pillow on bed, book on shelf)
- "attached to wall" — mounted/hanging on wall (e.g., picture, clock, curtain)
- "embedded in wall" — built into wall (e.g., door, window, outlet)

**Physical common sense**:
- Heavy furniture (table, chair, sofa, cabinet, shelf, bed) → "supported by floor"
- Small items on furniture (cup on table, lamp on desk, pillow on chair) → "supported by <that furniture>"
- Wall-mounted items (picture, mirror, clock, curtain) → "attached to wall"
- Structural elements (door, window, vent, outlet) → "embedded in wall"
- Items on floor-level objects (toy on carpet, box on floor) → "supported by floor"

JSON only, {num_targets} entries, no explanation:
{{"relationships":{{"target_name":"relationship"}}}}"""

    return prompt


def _extract_json_from_text(text):
    """从VLM输出文本中提取JSON字符串

    处理多种格式:
      - Qwen3.5 thinking标签: <think_>...</think_> 或 <think...</think...>
      - 代码块: ```json ... ``` 或 ``` ... ```
      - 裸JSON对象: {...}
      - 带前后文字: ...text...{...}...text...
    """
    if not text or not text.strip():
        return None

    for tag in ['</think_>', '</think >', '</think\n', '</think\r\n']:
        if tag in text:
            text = text.split(tag)[-1].strip()

    if '</think' in text:
        idx = text.rfind('</think')
        after = text[idx:]
        close_pos = after.find('>')
        if close_pos != -1:
            text = after[close_pos+1:].strip()

    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*({[\s\S]*?})\s*```']:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1]

    first_brace = text.find('{')
    if first_brace != -1:
        brace_count = 0
        for i in range(first_brace, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[first_brace:i+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
        last_brace = text.rfind('}')
        if last_brace > first_brace:
            return text[first_brace:last_brace+1]
    return None


def _fix_json_string(json_str):
    """修复常见的JSON格式错误

    处理: 尾部逗号、注释、无引号键名
    """
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    return json_str


# VLM 推理函数已从 src/vlm_logger 导入，不再在此处定义


def parse_optimal_frame_filename(filename):
    """解析optimal_frames文件名，提取物体类别、实例索引和帧号

    文件名格式: {category}_inst{inst_idx}_frame{frame_id}.jpg
    例:
        bowl_inst0_frame107.jpg → ("bowl", 0, 107)
        donut_inst2_frame112.jpg → ("donut", 2, 112)
        coffee_table_inst0_frame5.jpg → ("coffee_table", 0, 5)

    参数:
        filename: 文件名（不含目录）
    返回:
        (category, inst_idx, frame_id) 或 (None, None, None)
    """
    name = os.path.splitext(filename)[0]
    match = re.match(r'(.+)_inst(\d+)_frame(\d+)', name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None


def _fuzzy_match_name(query, candidates):
    """模糊匹配名称到候选列表

    匹配策略（按优先级）:
      1. 精确匹配（忽略大小写和空格/下划线差异）
      2. 基词匹配（取第一个下划线前的词）
      3. 子串匹配

    参数:
        query: 待匹配名称
        candidates: 候选名称列表
    返回:
        匹配到的候选名称，或None
    """
    if not query or not candidates:
        return None

    q = query.lower().strip().replace(' ', '_')

    for c in candidates:
        if c.lower().strip().replace(' ', '_') == q:
            return c

    q_base = q.split('_')[0]
    for c in candidates:
        c_base = c.lower().strip().replace(' ', '_').split('_')[0]
        if q_base == c_base:
            return c

    for c in candidates:
        c_low = c.lower().strip().replace(' ', '_')
        if q in c_low or c_low in q:
            return c

    return None


def load_optimal_frames(optimal_frames_dir, json_object_names):
    """加载optimal_frames/目录，按实例索引区分帧映射

    从文件名解析物体类别和实例索引（如 toy_inst0_frame108.jpg → "toy", inst=0），
    再通过模糊匹配关联到JSON中的物体名称。

    参数:
        optimal_frames_dir: optimal_frames目录路径
        json_object_names: JSON中的物体名称列表
    返回:
        object_instance_frames: {json_name: {inst_idx: [(image, source_str), ...]}}
            每个物体的每个实例的帧列表
        instance_count: {json_name: int} 每个物体的实例数量
    """
    if not os.path.isdir(optimal_frames_dir):
        print(f"   ⚠️  optimal_frames/ 目录不存在: {optimal_frames_dir}", flush=True)
        return {}, {}

    object_instance_frames = defaultdict(lambda: defaultdict(list))

    for fname in sorted(os.listdir(optimal_frames_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        obj_name, inst_idx, frame_id = parse_optimal_frame_filename(fname)
        if obj_name is None:
            continue

        json_name = _fuzzy_match_name(obj_name, json_object_names)

        img_path = os.path.join(optimal_frames_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"   ⚠️  无法加载图像 {fname}: {e}", flush=True)
            continue

        if json_name is not None:
            object_instance_frames[json_name][inst_idx].append((img, f"optimal_frames/{fname}"))
            print(f"      {fname} → {json_name}[inst{inst_idx}] (帧#{frame_id})", flush=True)
        else:
            object_instance_frames[obj_name][inst_idx].append((img, f"optimal_frames/{fname}"))
            print(f"      {fname} → {obj_name}[inst{inst_idx}] ⚠️ 无JSON匹配 (帧#{frame_id})", flush=True)

    instance_count = {}
    total = 0
    for json_name, inst_dict in object_instance_frames.items():
        instance_count[json_name] = len(inst_dict)
        for inst_idx, frames in inst_dict.items():
            total += len(frames)

    print(f"   📷 optimal_frames/: {total} 帧, 覆盖 {len(object_instance_frames)} 个物体类别", flush=True)
    for json_name, inst_dict in sorted(object_instance_frames.items()):
        for inst_idx, frames in sorted(inst_dict.items()):
            print(f"      {json_name}[inst{inst_idx}]: {len(frames)} 帧", flush=True)

    return dict(object_instance_frames), instance_count


def load_instance_visibility(optimal_frames_dir, json_object_names):
    """加载instance_visibility.json，获取每个实例在哪些帧有mask

    instance_visibility.json 由 main.py 生成，结构:
      {category: {"0": [frame_id, ...], "1": [frame_id, ...]}}

    通过模糊匹配关联到JSON中的物体名称。

    参数:
        optimal_frames_dir: optimal_frames目录路径（含instance_visibility.json）
        json_object_names: JSON中的物体名称列表
    返回:
        visibility: {json_name: {inst_idx_str: [frame_id, ...]}}
        instance_count_from_vis: {json_name: int}
    """
    vis_path = os.path.join(optimal_frames_dir, "instance_visibility.json")
    if not os.path.exists(vis_path):
        print(f"   ⚠️  instance_visibility.json 不存在: {vis_path}", flush=True)
        return {}, {}

    with open(vis_path, 'r') as f:
        raw_vis = json.load(f)

    visibility = {}
    instance_count_from_vis = {}

    for category, inst_dict in raw_vis.items():
        json_name = _fuzzy_match_name(category, json_object_names)
        if json_name is None:
            json_name = category
            print(f"      ⚠️  '{category}' 无JSON匹配，保留原名", flush=True)

        visibility[json_name] = {}
        for inst_idx_str, frame_ids in inst_dict.items():
            visibility[json_name][inst_idx_str] = frame_ids

        instance_count_from_vis[json_name] = len(inst_dict)
        print(f"      {category} → {json_name}: {len(inst_dict)} 个实例, "
              f"帧数={[len(v) for v in inst_dict.values()]}", flush=True)

    print(f"   📋 instance_visibility: {sum(len(v) for v in visibility.values())} 个实例", flush=True)
    return visibility, instance_count_from_vis


def load_keyframes_with_visibility(keyframes_dir, json_object_names):
    """加载keyframes/目录及可见性元数据

    从 keyframes_metadata.json 读取每帧可见物体列表，
    通过模糊匹配关联到JSON中的物体名称。

    参数:
        keyframes_dir: keyframes目录路径（含keyframes_metadata.json）
        json_object_names: JSON中的物体名称列表
    返回:
        keyframe_images: {vid_idx: (image, source_str)} 帧索引→图像
        frame_visibility: {vid_idx: [json_name, ...]} 帧索引→可见物体列表（已匹配到JSON名称）
    """
    metadata_path = os.path.join(keyframes_dir, "keyframes_metadata.json")
    if not os.path.exists(metadata_path):
        print(f"   ⚠️  keyframes_metadata.json 不存在: {metadata_path}", flush=True)
        return {}, {}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    raw_visibility = metadata.get("frame_visibility", {})
    keyframe_list = metadata.get("keyframes", [])

    keyframe_images = {}
    for kf in keyframe_list:
        vid_idx = kf["vid_idx"]
        path = kf["path"]
        full_path = os.path.join(keyframes_dir, path)
        if os.path.exists(full_path):
            try:
                img = Image.open(full_path).convert("RGB")
                keyframe_images[vid_idx] = (img, f"keyframes/{path}")
            except Exception as e:
                print(f"   ⚠️  无法加载关键帧 {path}: {e}", flush=True)

    frame_visibility = {}
    for fidx_str, visible_names in raw_visibility.items():
        vid_idx = int(fidx_str)
        matched = []
        for raw_name in visible_names:
            json_name = _fuzzy_match_name(raw_name, json_object_names)
            matched.append(json_name if json_name else raw_name)
        frame_visibility[vid_idx] = matched

    print(f"   📷 keyframes/: {len(keyframe_images)} 个关键帧图像, {len(frame_visibility)} 帧可见性数据", flush=True)
    for vid_idx, names in sorted(frame_visibility.items()):
        print(f"      帧#{vid_idx}: {names}", flush=True)

    return keyframe_images, frame_visibility


def build_object_to_frames(
    optimal_instance_frames: Dict[str, Dict[int, List]],
    instance_visibility: Dict[str, Dict[str, List]],
    keyframe_images: Dict[int, Tuple],
    keyframe_visibility: Dict[int, List[str]],
    other_objects: List[str],
    instance_count: Dict[str, int],
    scene_dir: str = None,
    min_frames: int = 5,
):
    """高效的帧选择：optimal_frames + keyframes，不足min_frames才从color/补充

    选帧优先级:
      1. optimal_frames/ — 每个实例的最优视角帧（1帧/实例）
      2. keyframes/ — stage1贪心采样帧中该类别可见的帧（所有实例共享）

    若合并后仍不足min_frames帧，从instance_visibility中按视角距离补充。
    补充时从 color/ 按需加载图像（不预加载全量，避免速度慢）。

    参数:
        optimal_instance_frames: {json_name: {inst_idx: [(image, src), ...]}}
        instance_visibility: {json_name: {inst_idx_str: [frame_id, ...]}}
        keyframe_images: {vid_idx: (image, source_str)}
        keyframe_visibility: {vid_idx: [json_name, ...]}
        other_objects: 需要细化的物体名称列表
        instance_count: {json_name: int} 每个物体的实例数量
        scene_dir: 场景目录（用于从color/按需加载补充帧）
        min_frames: 每个实例最少需要的帧数
    返回:
        instance_to_frames: {instance_key: [(image, source_str), ...]}
        instance_to_category: {instance_key: category_name}
        missing_instances: [instance_key, ...] 无任何帧的实例
    """
    instance_to_frames = defaultdict(list)
    instance_to_category = {}
    seen_sources = defaultdict(set)

    for obj_name in other_objects:
        n_inst = instance_count.get(obj_name, 0)
        if n_inst == 0:
            n_inst = 1

        for inst_idx in range(n_inst):
            inst_key = f"{obj_name}_{inst_idx}"
            instance_to_category[inst_key] = obj_name

            if obj_name in optimal_instance_frames:
                inst_dict = optimal_instance_frames[obj_name]
                if inst_idx in inst_dict:
                    for img, src in inst_dict[inst_idx]:
                        if src not in seen_sources[inst_key]:
                            instance_to_frames[inst_key].append((img, src))
                            seen_sources[inst_key].add(src)

    for vid_idx, visible_names in keyframe_visibility.items():
        if vid_idx not in keyframe_images:
            continue
        img, src = keyframe_images[vid_idx]
        for obj_name in other_objects:
            if obj_name in visible_names:
                n_inst = instance_count.get(obj_name, 0)
                if n_inst == 0:
                    n_inst = 1
                for inst_idx in range(n_inst):
                    inst_key = f"{obj_name}_{inst_idx}"
                    if src not in seen_sources[inst_key]:
                        instance_to_frames[inst_key].append((img, src))
                        seen_sources[inst_key].add(src)

    for inst_key in list(instance_to_category.keys()):
        current_frames = instance_to_frames.get(inst_key, [])
        if len(current_frames) >= min_frames:
            continue

        obj_name = instance_to_category[inst_key]
        inst_idx = int(inst_key.split('_')[-1])
        inst_idx_str = str(inst_idx)

        existing_fids = set()
        for _, src in current_frames:
            m = re.search(r'(\d+)\.(jpg|png|jpeg)$', src)
            if m:
                existing_fids.add(int(m.group(1)))
            m2 = re.search(r'vid(\d+)', src)
            if m2:
                existing_fids.add(int(m2.group(1)))

        vis_dict = instance_visibility.get(obj_name, {})
        if inst_idx_str in vis_dict:
            all_vis_fids = vis_dict[inst_idx_str]
            remaining = [fid for fid in all_vis_fids if fid not in existing_fids]

            def _view_distance(fid):
                if not existing_fids:
                    return fid
                return min(abs(fid - ef) for ef in existing_fids)

            remaining.sort(key=lambda fid: -_view_distance(fid))
        else:
            color_dir_check = os.path.join(scene_dir, "color") if scene_dir else None
            if color_dir_check and os.path.isdir(color_dir_check):
                total_color_frames = len([f for f in os.listdir(color_dir_check)
                                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                if total_color_frames > 0:
                    all_fids = list(range(total_color_frames))
                    remaining = [fid for fid in all_fids if fid not in existing_fids]
                    remaining.sort(key=lambda fid: -(
                        min(abs(fid - ef) for ef in existing_fids) if existing_fids else fid))
                    print(f"      📐 {inst_key}: 无visibility数据，从color/均匀采样补充", flush=True)
                else:
                    remaining = []
            else:
                remaining = []

        need = min_frames - len(current_frames)
        supplemented = 0
        color_dir = os.path.join(scene_dir, "color") if scene_dir else None
        for fid in remaining:
            if supplemented >= need:
                break
            if color_dir is None:
                break
            img = None
            for ext in ['.jpg', '.png', '.jpeg']:
                img_path = os.path.join(color_dir, f"{fid}{ext}")
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path).convert("RGB")
                        break
                    except Exception:
                        continue
            if img is None:
                continue
            src = f"color/{fid}{ext}"
            if src not in seen_sources[inst_key]:
                instance_to_frames[inst_key].append((img, src))
                seen_sources[inst_key].add(src)
                supplemented += 1

        if supplemented > 0:
            print(f"      📐 {inst_key}: 补充 {supplemented} 帧（视角距离优先）", flush=True)

    missing_instances = [k for k in instance_to_category if len(instance_to_frames.get(k, [])) == 0]

    print(f"\n   📋 实例→帧映射（合并+按需补充后）:", flush=True)
    for inst_key in sorted(instance_to_category.keys()):
        frames = instance_to_frames.get(inst_key, [])
        if frames:
            sources = [src for _, src in frames]
            print(f"      {inst_key}: {len(frames)} 帧 → {sources}", flush=True)
        else:
            print(f"      {inst_key}: ❌ 无帧（保留原关系）", flush=True)

    return dict(instance_to_frames), instance_to_category, missing_instances


def post_process_relationships(relationships, all_relations):
    """物理常识后处理纠错

    规则:
      1. 家具类不应 attached/embedded wall → 修正为 supported by floor
      2. 挂墙类不应 supported by floor → 修正为 attached to wall
      3. 嵌入墙类不应 supported by floor → 修正为 embedded in wall
      4. "other objects"关系的物体不能作为支撑物
      5. 确保所有目标物体都有关系判断

    参数:
        relationships: {name: rel} VLM判断的关系
        all_relations: {name: rel} 当前所有物体的关系
    返回:
        corrected: {name: rel} 纠错后的关系
    """
    corrected = {}
    all_names = set(all_relations.keys())

    for target_name, rel in relationships.items():
        name = target_name.lower().strip().replace(' ', '_')
        category_base = name.split('_')[0]

        if name in FILTER_CATEGORIES:
            print(f"      [过滤] 跳过 '{name}': 属于floor/wall类别", flush=True)
            continue

        if name in MUST_BE_FLOOR_SUPPORTED or category_base in MUST_BE_FLOOR_SUPPORTED:
            if "attached to wall" in rel or "embedded in wall" in rel:
                print(f"      [纠错] '{name}': {rel} → supported by floor (物理常识)", flush=True)
                rel = "supported by floor"

        if name in MUST_BE_WALL_ATTACHED or category_base in MUST_BE_WALL_ATTACHED:
            if rel == "supported by floor":
                if category_base in {'picture', 'painting', 'photo', 'mirror', 'clock', 'poster'}:
                    print(f"      [纠错] '{name}': supported by floor → attached to wall (物理常识)", flush=True)
                    rel = "attached to wall"
                elif category_base in {'window', 'door', 'vent', 'air_vent', 'outlet', 'socket'}:
                    print(f"      [纠错] '{name}': supported by floor → embedded in wall (物理常识)", flush=True)
                    rel = "embedded in wall"

        if "supported by" in rel and "floor" not in rel and "other objects" not in rel:
            supporter_name = rel.replace("supported by ", "").strip()
            if supporter_name in all_names:
                supporter_rel = all_relations.get(supporter_name, "")
                if supporter_rel == "supported by other objects":
                    print(f"      [纠错] '{name}': 支撑物 '{supporter_name}' 本身也是'other objects'关系 → supported by floor", flush=True)
                    rel = "supported by floor"

        corrected[target_name] = rel

    return corrected


def _match_name(vlm_output, candidate_list):
    """模糊匹配VLM输出到候选列表中的名称

    参数:
        vlm_output: VLM输出的名称字符串
        candidate_list: 候选名称列表
    返回:
        匹配到的候选名称，或None
    """
    if not vlm_output:
        return None
    vlm_lower = vlm_output.lower().strip()

    if vlm_lower == "floor":
        return "floor"

    for cand in candidate_list:
        if cand.lower() == vlm_lower:
            return cand

    for cand in candidate_list:
        if vlm_lower in cand.lower() or cand.lower() in vlm_lower:
            return cand

    vlm_base = vlm_lower.split('_')[0].split(' ')[0]
    for cand in candidate_list:
        cand_base = cand.lower().split('_')[0].split(' ')[0]
        if vlm_base == cand_base:
            return cand

    return None


def _parse_vlm_relationships(output_text, target_objects, all_object_names):
    """解析VLM输出为关系字典

    VLM可能输出:
      {"relationships": {"bowl": "supported by table", "toy": "supported by floor"}}
    或:
      {"bowl": "supported by table", "toy": "attached to wall"}

    参数:
        output_text: VLM原始输出文本
        target_objects: 目标物体名称列表
        all_object_names: 所有物体名称列表
    返回:
        {target_name: relationship_string} 或 None
    """
    json_str = _extract_json_from_text(output_text)
    if json_str is None:
        return None

    try:
        result = json.loads(json_str)
    except json.JSONDecodeError:
        json_str = _fix_json_string(json_str)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError:
            return None

    if "relationships" in result and isinstance(result["relationships"], dict):
        raw_rels = result["relationships"]
    elif isinstance(result, dict):
        raw_rels = result
    else:
        return None

    parsed = {}
    for vlm_key, vlm_val in raw_rels.items():
        matched_target = _match_name(vlm_key, target_objects)

        if not matched_target:
            if vlm_key.lower().strip().replace(' ', '_') in ('target_name', 'target', 'object_name', 'name') \
                    and len(target_objects) == 1:
                matched_target = target_objects[0]
            else:
                continue

        if not isinstance(vlm_val, str):
            vlm_val = str(vlm_val)

        vlm_val_lower = vlm_val.lower().strip()

        if vlm_val_lower in ("supported by floor", "attached to wall", "embedded in wall"):
            parsed[matched_target] = vlm_val_lower
        elif vlm_val_lower.startswith("supported by "):
            supporter_raw = vlm_val_lower.replace("supported by ", "").strip()
            matched_supporter = _match_name(supporter_raw, all_object_names + ["floor"])
            if matched_supporter == "floor":
                parsed[matched_target] = "supported by floor"
            elif matched_supporter:
                parsed[matched_target] = f"supported by {matched_supporter}"
            else:
                parsed[matched_target] = "supported by floor"
        elif vlm_val_lower in ("floor", "ground"):
            parsed[matched_target] = "supported by floor"
        elif vlm_val_lower in ("wall",):
            parsed[matched_target] = "attached to wall"
        else:
            matched_supporter = _match_name(vlm_val_lower, all_object_names + ["floor"])
            if matched_supporter == "floor":
                parsed[matched_target] = "supported by floor"
            elif matched_supporter:
                parsed[matched_target] = f"supported by {matched_supporter}"
            else:
                parsed[matched_target] = "supported by floor"

    return parsed


def _parse_vlm_text_fallback(output_text, target_objects, all_object_names):
    """纯文本回退解析：当VLM不输出JSON时，从文本中提取关系

    尝试匹配文本中的关系关键词:
      - "supported by floor" / "on the floor" / "on the ground"
      - "supported by <name>" / "on the <name>" / "on a <name>"
      - "attached to wall" / "on the wall" / "hanging on wall"
      - "embedded in wall" / "in the wall"

    参数:
        output_text: VLM原始输出文本
        target_objects: 目标物体名称列表
        all_object_names: 所有物体名称列表
    返回:
        {target_name: relationship_string} 或 None
    """
    text_lower = output_text.lower()

    rel_patterns = [
        (r'supported by floor', "supported by floor"),
        (r'on the floor', "supported by floor"),
        (r'on the ground', "supported by floor"),
        (r'rests? on the floor', "supported by floor"),
        (r'attached to wall', "attached to wall"),
        (r'hanging on (?:the )?wall', "attached to wall"),
        (r'on (?:the )?wall', "attached to wall"),
        (r'embedded in wall', "embedded in wall"),
        (r'built into (?:the )?wall', "embedded in wall"),
        (r'in (?:the )?wall', "embedded in wall"),
    ]

    for pattern, rel in rel_patterns:
        if re.search(pattern, text_lower):
            if len(target_objects) == 1:
                return {target_objects[0]: rel}

    supported_by_match = re.search(r'supported by (\w[\w\s]*?)(?:\.|,|$)', text_lower)
    if supported_by_match:
        supporter_raw = supported_by_match.group(1).strip()
        matched = _match_name(supporter_raw, all_object_names + ["floor"])
        if matched == "floor" and len(target_objects) == 1:
            return {target_objects[0]: "supported by floor"}
        elif matched and len(target_objects) == 1:
            return {target_objects[0]: f"supported by {matched}"}

    on_match = re.search(r'on (?:a |the )?(\w[\w\s]*?)(?:\.|,|$)', text_lower)
    if on_match:
        supporter_raw = on_match.group(1).strip()
        if supporter_raw in ('floor', 'ground'):
            if len(target_objects) == 1:
                return {target_objects[0]: "supported by floor"}
        elif supporter_raw in ('wall',):
            if len(target_objects) == 1:
                return {target_objects[0]: "attached to wall"}
        else:
            matched = _match_name(supporter_raw, all_object_names)
            if matched and len(target_objects) == 1:
                return {target_objects[0]: f"supported by {matched}"}

    return None


def _infer_keyframes_dir(scene_dir):
    """从scene_dir推断keyframes目录路径

    规则: 项目根目录/assets/key_frames/{scene_id}/
    其中 scene_id 从 scene_dir 路径末尾提取

    例: scene_dir="./outputs/232" → "./assets/key_frames/232"
    """
    scene_id = os.path.basename(os.path.normpath(scene_dir))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    keyframes_dir = os.path.join(project_root, "assets", "key_frames", scene_id)
    if os.path.isdir(keyframes_dir):
        return keyframes_dir
    fallback = os.path.join(scene_dir, "keyframes")
    return fallback


def refine_other_objects_relations(
    stage1_json_path: str,
    output_json_path: str,
    scene_dir: str,
    vlm_checkpoint: str,
    optimal_frames_dir: str = None,
    keyframes_dir: str = None,
    min_frames: int = 5,
):
    """细化 "supported by other objects" 关系的主函数（按实例投票）

    流程:
      1. 读取Stage1 JSON，分离出 "other objects" 物体
      2. 加载 optimal_frames/ 图像，按实例索引区分（如 toy_inst0, toy_inst1）
      3. 加载 keyframes/ 图像和可见性元数据
      4. 合并帧来源，构建每个实例的去重帧列表
      5. 扩展JSON key: toy → toy_0, toy_1, toy_2（多实例独立投票）
      6. 对每个实例的帧用VLM判断具体支撑关系
      7. 多帧投票 + 物理常识后处理纠错
      8. 更新JSON，输出细化后的场景JSON

    参数:
        stage1_json_path: Stage1场景JSON路径
        output_json_path: 输出JSON路径
        scene_dir: 场景输出目录 (如 outputs/232)
        vlm_checkpoint: VLM模型路径
        optimal_frames_dir: optimal_frames目录路径 (默认自动推断)
        keyframes_dir: keyframes目录路径 (默认自动推断)
    返回:
        final_relations: 细化后的关系字典
    """
    print("=" * 70, flush=True)
    print("🔧 Stage 4: VLM细化'supported by other objects'关系 (按实例投票)", flush=True)
    print("   帧来源: optimal_frames/ + instance_visibility + keyframes/", flush=True)
    print("=" * 70, flush=True)

    print(f"\n📂 加载Stage1 JSON: {stage1_json_path}", flush=True)
    with open(stage1_json_path, 'r') as f:
        stage1_relations = json.load(f)

    other_objects = [n for n, r in stage1_relations.items() if r == "supported by other objects"]
    floor_objects = [n for n, r in stage1_relations.items() if r == "supported by floor"]
    wall_attached = [n for n, r in stage1_relations.items() if r == "attached to wall"]
    wall_embedded = [n for n, r in stage1_relations.items() if r == "embedded in wall"]

    print(f"   总物体数: {len(stage1_relations)}", flush=True)
    print(f"   supported by other objects: {len(other_objects)} → {other_objects}", flush=True)
    print(f"   supported by floor: {len(floor_objects)} → {floor_objects}", flush=True)
    print(f"   attached to wall: {len(wall_attached)} → {wall_attached}", flush=True)
    print(f"   embedded in wall: {len(wall_embedded)} → {wall_embedded}", flush=True)

    if not other_objects:
        print(f"\n✅ 无需细化", flush=True)
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(stage1_relations, f, indent=2, ensure_ascii=False)
        return stage1_relations

    json_object_names = list(stage1_relations.keys())

    if optimal_frames_dir is None:
        optimal_frames_dir = os.path.join(scene_dir, "optimal_frames")
    if keyframes_dir is None:
        keyframes_dir = _infer_keyframes_dir(scene_dir)

    print(f"\n📷 [来源1] 加载optimal_frames: {optimal_frames_dir}", flush=True)
    optimal_instance_frames, instance_count_from_opt = load_optimal_frames(optimal_frames_dir, json_object_names)

    print(f"\n📋 [来源2] 加载instance_visibility: {optimal_frames_dir}", flush=True)
    instance_visibility, instance_count_from_vis = load_instance_visibility(optimal_frames_dir, json_object_names)

    instance_count = {}
    for name in set(list(instance_count_from_opt.keys()) + list(instance_count_from_vis.keys())):
        instance_count[name] = max(instance_count_from_opt.get(name, 0), instance_count_from_vis.get(name, 0))
    for obj_name in other_objects:
        if obj_name not in instance_count:
            instance_count[obj_name] = 1
    print(f"   合并后instance_count: {instance_count}", flush=True)

    print(f"\n📷 [来源2] 加载keyframes: {keyframes_dir}", flush=True)
    keyframe_images, keyframe_visibility = load_keyframes_with_visibility(keyframes_dir, json_object_names)

    print(f"\n🔗 帧选择 (optimal + keyframes, 不足{min_frames}帧按需从color/补充)", flush=True)
    for obj_name in other_objects:
        opt_n = len(optimal_instance_frames.get(obj_name, {}))
        vis_dict = instance_visibility.get(obj_name, {})
        vis_total = sum(len(fids) for fids in vis_dict.values())
        kf_n = sum(1 for names in keyframe_visibility.values() if obj_name in names)
        est = opt_n + kf_n
        flag = "" if est >= min_frames else f" ⚠️需补充至{min_frames}"
        print(f"   {obj_name}: optimal={opt_n}, keyframes≈{kf_n}, visibility可用={vis_total}帧 → 预估{est}帧{flag}", flush=True)

    instance_to_frames, instance_to_category, missing_instances = build_object_to_frames(
        optimal_instance_frames, instance_visibility,
        keyframe_images, keyframe_visibility, other_objects, instance_count,
        scene_dir=scene_dir, min_frames=min_frames,
    )

    present_instances = [k for k in instance_to_category if k not in missing_instances]

    final_relations = {}
    for name, rel in stage1_relations.items():
        if rel != "supported by other objects":
            final_relations[name] = rel

    if missing_instances:
        print(f"\n   ⚠️  以下实例在optimal_frames/和keyframes/中均无对应帧，保留原关系:", flush=True)
        for k in missing_instances:
            print(f"      ❌ {k} → supported by other objects (保留)", flush=True)
            final_relations[k] = "supported by other objects"

    if not present_instances:
        print(f"\n✅ 所有实例均无可用帧，已默认处理", flush=True)
        os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
        sorted_relations = dict(sorted(final_relations.items()))
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_relations, f, indent=2, ensure_ascii=False)
        return final_relations

    print(f"\n📋 实例扩展映射:", flush=True)
    for obj_name in other_objects:
        n_inst = instance_count.get(obj_name, 0)
        if n_inst == 0:
            n_inst = 1
        if n_inst == 1:
            print(f"   {obj_name} → {obj_name}_0 (单实例)", flush=True)
        else:
            keys = [f"{obj_name}_{i}" for i in range(n_inst)]
            print(f"   {obj_name} → {', '.join(keys)} ({n_inst}个实例)", flush=True)

    print(f"\n🤖 加载VLM: {vlm_checkpoint}", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        vlm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(vlm_checkpoint, trust_remote_code=True)
    print(f"   ✅ VLM模型加载完成", flush=True)

    all_votes = defaultdict(lambda: defaultdict(int))

    print(f"\n🔍 VLM推理 (按实例投票, {len(present_instances)} 个实例)", flush=True)

    for inst_key in present_instances:
        frames = instance_to_frames.get(inst_key, [])
        if not frames:
            continue

        category = instance_to_category[inst_key]
        print(f"\n   🔍 判断 '{inst_key}' ({len(frames)} 帧)...", flush=True)

        prompt_relations = {}
        for name, rel in stage1_relations.items():
            if name == category:
                prompt_relations[inst_key] = "supported by other objects"
            else:
                prompt_relations[name] = rel
        prompt = build_refinement_prompt(prompt_relations, [inst_key])

        for fi, (image, src) in enumerate(frames):
            print(f"      [{fi+1}/{len(frames)}] {src}...", end=" ", flush=True)

            try:
                output_text = vlm_inference(image, model, processor, prompt, max_new_tokens=1024, caller="stage5_refine_relations")
                frame_rels = _parse_vlm_relationships(output_text, [inst_key], list(prompt_relations.keys()))

                if not frame_rels:
                    frame_rels = _parse_vlm_text_fallback(output_text, [inst_key], list(prompt_relations.keys()))

                if frame_rels:
                    frame_corrected = post_process_relationships(frame_rels, final_relations)
                    for name, rel in frame_corrected.items():
                        if name == inst_key and (rel in VALID_RELATIONSHIPS or rel.startswith("supported by ")):
                            all_votes[inst_key][rel] += 1
                    vote_rel = frame_rels.get(inst_key, "unknown")
                    print(f"✅ → {vote_rel}", flush=True)
                else:
                    print(f"❌ 解析失败, VLM原始输出: {output_text[:300]}", flush=True)

            except Exception as e:
                print(f"❌ 错误: {e}", flush=True)

    del model
    del processor
    torch.cuda.empty_cache()

    print(f"\n📊 投票汇总:", flush=True)
    for inst_key in sorted(instance_to_category.keys()):
        votes = all_votes.get(inst_key, {})
        if votes:
            best_rel = max(votes, key=votes.get)
            n_vote = votes[best_rel]
            n_total = sum(votes.values())

            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_votes) >= 2 and sorted_votes[0][1] == sorted_votes[1][1]:
                tied_rels = [rel for rel, cnt in sorted_votes if cnt == sorted_votes[0][1]]
                floor_candidates = [r for r in tied_rels if "floor" in r.lower()]
                table_candidates = [r for r in tied_rels if "table" in r.lower()]
                if floor_candidates and table_candidates:
                    best_rel = f"tied:{','.join(tied_rels)}"
                    print(f"   {inst_key}: ⚠️平票 {sorted_votes} → 标记为'{best_rel}' (后续用坐标系判断)", flush=True)
                else:
                    best_rel = sorted_votes[0][0]
                    print(f"   {inst_key}: '{best_rel}' ({n_vote}/{n_total}) [平票但非floor/table]", flush=True)
            else:
                print(f"   {inst_key}: '{best_rel}' ({n_vote}/{n_total})", flush=True)
        else:
            best_rel = "supported by other objects"
            if inst_key in missing_instances:
                print(f"   {inst_key}: 无帧可用 → supported by other objects (保留原关系)", flush=True)
            else:
                print(f"   {inst_key}: 无投票 → supported by other objects (保留原关系)", flush=True)

        final_relations[inst_key] = best_rel

    os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
    sorted_relations = dict(sorted(final_relations.items()))
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_relations, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 细化完成! 保存至: {output_json_path}", flush=True)

    rel_counts = defaultdict(int)
    for rel in final_relations.values():
        rel_counts[rel] += 1

    print(f"\n📊 最终关系统计:", flush=True)
    for rel, count in sorted(rel_counts.items()):
        print(f"   {rel}: {count} 个物体", flush=True)

    print(f"\n📦 最终类别清单 ({len(sorted_relations)} 个):", flush=True)
    for i, (name, rel) in enumerate(sorted_relations.items(), 1):
        print(f"   {i:2d}. {name:20s} → {rel}", flush=True)

    return final_relations


def main():
    parser = argparse.ArgumentParser(
        description='Stage 4: 细化supported by other objects关系',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
调用示例:
  # 基本调用（自动推断路径）
  python tools/refine_other_objects_relations.py \\
      --stage1_json ./assets/json_configs/232.json \\
      --scene_dir ./outputs/232

  # 完整调用
  python tools/refine_other_objects_relations.py \\
      --stage1_json ./assets/json_configs/232.json \\
      --output_json ./assets/json_configs/232_refined.json \\
      --scene_dir ./outputs/232 \\
      --optimal_frames_dir ./outputs/232/optimal_frames \\
      --keyframes_dir ./assets/key_frames/232 \\
      --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B
        """)
    parser.add_argument('--stage1_json', type=str, required=True,
                        help='Stage1场景JSON路径 (如 ./assets/json_configs/232.json)')
    parser.add_argument('--output_json', type=str, default=None,
                        help='输出JSON路径 (默认: {stage1_json同名}_refined.json)')
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='场景输出目录 (如 ./outputs/232，用于定位optimal_frames/)')
    parser.add_argument('--optimal_frames_dir', type=str, default=None,
                        help='optimal_frames目录 (默认: {scene_dir}/optimal_frames)')
    parser.add_argument('--keyframes_dir', type=str, default=None,
                        help='keyframes目录 (默认: 项目根目录/assets/key_frames/{scene_id}/)')
    parser.add_argument('--vlm_checkpoint', type=str, default=None,
                        help='VLM模型路径')

    args = parser.parse_args()

    if args.output_json is None:
        base_path = Path(args.stage1_json)
        args.output_json = str(base_path.parent / f"{base_path.stem}_refined{base_path.suffix}")

    if args.optimal_frames_dir is None:
        args.optimal_frames_dir = os.path.join(args.scene_dir, "optimal_frames")
    if args.keyframes_dir is None:
        args.keyframes_dir = _infer_keyframes_dir(args.scene_dir)

    if args.vlm_checkpoint is None:
        default_model = "/mnt/data/lza/models/Qwen3.5-9B"
        if os.path.exists(default_model):
            args.vlm_checkpoint = default_model
        else:
            fallback = "/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct"
            if os.path.exists(fallback):
                snapshots_dir = os.path.join(fallback, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        args.vlm_checkpoint = os.path.join(snapshots_dir, snapshots[0])

    if args.vlm_checkpoint is None:
        print("❌ 未找到VLM模型", flush=True)
        return

    print("=" * 70, flush=True)
    print("🚀 Stage 4: 细化'other objects'关系", flush=True)
    print("=" * 70, flush=True)
    print(f"📂 Stage1 JSON: {args.stage1_json}")
    print(f"📂 场景目录: {args.scene_dir}")
    print(f"📂 optimal_frames: {args.optimal_frames_dir}")
    print(f"📂 keyframes: {args.keyframes_dir}")
    print(f"📤 输出: {args.output_json}")
    print(f"🤖 VLM: {args.vlm_checkpoint}")
    print("=" * 70 + "\n", flush=True)

    try:
        refine_other_objects_relations(
            stage1_json_path=args.stage1_json,
            output_json_path=args.output_json,
            scene_dir=args.scene_dir,
            vlm_checkpoint=args.vlm_checkpoint,
            optimal_frames_dir=args.optimal_frames_dir,
            keyframes_dir=args.keyframes_dir,
        )
    except KeyboardInterrupt:
        print("\n⚠️  用户中断", flush=True)
    except Exception as e:
        import traceback
        print(f"\n❌ 错误: {e}\n{traceback.format_exc()}", flush=True)


if __name__ == '__main__':
    main()
