"""
Stage 5.1 替代方案: 使用 SimRecon 场景图推断方式获取物体间关系
================================================================

与 refine_other_objects_relations.py 的区别:
  旧方式: 逐个物体查看图像，判断 "supported by other objects" 的具体支撑物
  新方式: 在一张带ID标签的场景图上，一次性推断所有物体的类别和关系

优势:
  1. 支持多级嵌套关系 (floor → table → cup)
  2. 单次VLM调用推断所有关系，上下文更完整
  3. 物理常识后处理更完善 (柜子不应挂在墙等)

精修方式仍使用 sp_refinement.py 的几何精修函数。

使用方式:
  # 从 mainv2.py 自动调用 (推荐)
  python mainv2.py --input_video ./232.mp4 --enable_stage5 --stage5_method scene_graph

  # 独立调用
  python tools/infer_relations_scene_graph.py \
      --scene_dir output_v2/232_vggt \
      --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B

  # 指定 ID 标注图
  python tools/infer_relations_scene_graph.py \
      --scene_dir output_v2/232_vggt \
      --id_scene_path output_v2/232_vggt/id_scene.png \
      --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from PIL import Image

from src.vlm_logger import vlm_inference, set_vlm_log_dir


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
    'tv', 'television', 'monitor',
    'light_switch', 'outlet', 'socket',
    'curtain', 'drape', 'blind', 'shade',
    'window', 'door', 'vent', 'air_vent'
}


def build_instance_names(categories_and_relations: Dict[str, str]) -> List[str]:
    """从 categories_and_relations 构建实例名列表

    规则:
      - 如果 key 带 _数字 后缀 (如 toy_0, duck_1) → 保留原样
      - 如果 key 不带 _数字 后缀 (如 toy, duck) → 推断实例数量后展开
        从 scene_dir/optimal_frames/instance_visibility.json 读取实例数

    返回:
      实例名列表，如 ["duck_0", "duck_1", "plate_0", "toy_0", ...]
    """
    instance_names = []
    for key in categories_and_relations:
        if key.lower() in FILTER_CATEGORIES:
            continue
        if '_' in key and key.rsplit('_', 1)[-1].isdigit():
            # 已经是实例级 key
            instance_names.append(key)
        else:
            # 需要展开为实例级
            instance_names.append(key)  # 暂时用类别名，调用者负责展开

    return instance_names


def create_labeled_frame_image(
    frame_image: np.ndarray,
    visible_instances: List[str],
    instance_masks: Dict[str, List[Dict]],
    frame_id: int,
) -> np.ndarray:
    """
    在帧图像上为每个可见实例绘制彩色 bounding box + 实例名

    参数:
        frame_image: RGB图像 (H, W, 3)
        visible_instances: 该帧中可见的实例名列表 (如 ["duck_0", "plate_0", "toy_0"])
        instance_masks: mask 数据 {category: [{frame_id, mask}, ...], ...}
        frame_id: 当前帧的 ID
    返回:
        标注后的RGB图像 (带彩色框和实例名)
    """
    annotated = frame_image.copy()
    h, w = annotated.shape[:2]

    # 为每个实例生成固定颜色 (基于实例名哈希, 保证同实例同色)
    instance_colors = {}
    for inst_name in visible_instances:
        hash_val = hash(inst_name) % (256 * 256 * 256)
        r = (hash_val >> 16) & 0xFF
        g = (hash_val >> 8) & 0xFF
        b = hash_val & 0xFF
        instance_colors[inst_name] = (r, g, b)

    # 为每个可见实例查找 mask 并绘制 bounding box
    for inst_name in visible_instances:
        color = instance_colors[inst_name]
        # 解析实例名: "duck_0" → category="duck", idx=0
        parts = inst_name.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            category, inst_idx = parts[0], int(parts[1])
        else:
            # 非 "category_idx" 格式的实例名 (如纯类别名 "car" 或 "car_car"),
            # 整名当作类别, 索引取 0; 若在 instance_masks 中找不到整名对应的类别,
            # 会安全跳过 (见下方 cat_masks 判空), 避免 int() 抛 ValueError 崩溃
            category, inst_idx = inst_name, 0

        # 从 instance_masks 中查找该实例的 mask
        cat_masks = instance_masks.get(category)
        if not cat_masks:
            continue
        if inst_idx >= len(cat_masks):
            continue
        instance_mask_list = cat_masks[inst_idx]

        # 找到 frame_id 对应的 mask
        bbox = None
        for mask_entry in instance_mask_list:
            if mask_entry.get('frame_id') == frame_id:
                mask = mask_entry.get('mask')
                if mask is not None and mask.ndim >= 2:
                    coords = np.where(mask > 0)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                break

        if bbox is None:
            # 没有 mask，跳过
            continue

        x, y, bw, bh = bbox
        # 边界保护
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))

        # 绘制矩形框
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)

        # 绘制实例名标签 (在框上方)
        label = inst_name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        label_y = max(y - text_size[1] - 5, 5)

        # 标签背景
        cv2.rectangle(annotated,
                      (x, label_y - text_size[1] - 3),
                      (x + text_size[0], label_y + 3),
                      color, -1)
        # 标签文字
        cv2.putText(annotated, label, (x + 2, label_y),
                    font, font_scale, (255, 255, 255), thickness)

    return annotated


def infer_relations_from_labeled_image(
    model,
    processor,
    image_path: str,
    instance_names: List[str],
    query_instances: Optional[List[str]] = None,
) -> str:
    """
    使用 SimRecon 风格的 VLM prompt 推断物体间关系
    (复用现有 prompt 模板, 只替换 object name hints)

    参数:
        model: VLM模型
        processor: VLM处理器
        image_path: 标注图路径
        instance_names: 场景中全部实例名 (如 ["duck_0", "plate_0", "toy_0"])
        query_instances: 本次实际需要结果的实例名; None 表示全都要。
            非 None 时 prompt 只要求输出这些实例 —— 调用方通常只取其中一条,
            让模型生成全场景 JSON 是 O(N^2) 的浪费 (25 实例 x 3 帧, 每次生成
            25 条记录, 其中 24 条立即被丢弃)。非 None 时仍完整列出
            instance_names 作为名称参考, 保证 parent 能写成 "table_0" 之类。
    返回:
        VLM 原始输出文本
    """
    # 构建 prompt
    num_objects = len(instance_names)
    instance_names_str = ", ".join(instance_names)
    id_name_hints = "\n".join([f"- {name}" for name in instance_names])

    # 只查询部分实例时裁剪输出规模 (方案2: 消除 O(N^2) 的生成浪费)
    query = list(query_instances) if query_instances else list(instance_names)
    num_query = len(query)
    query_str = ", ".join(query)

    prompt = f'''You are analyzing a room scene image with labeled objects.

**Visible objects**: {instance_names_str} (Total: {num_objects} objects)

**Hidden root nodes** (not shown in image):
- "floor" (ground surface)
- "wall" (vertical wall surface)

**Object name hints** (use these as the primary reference):
{id_name_hints}

## PHYSICAL COMMON SENSE RULES (use internally, do NOT repeat in output):
1. Cabinets, wardrobes, bookshelves, tables, desks, chairs, beds -> supported by floor.
2. Wall-attached is RARE, only for: pictures, mirrors, clocks, posters, wall-mounted TVs, curtains, windows, doors.
3. Objects resting on furniture -> parent is that furniture's instance name.
4. Small items on a desk (lamp, monitor, keyboard, cup) -> supported by that desk.
5. Objects held by hand or floating -> assign their NATURAL support (where they belong when not held); never use relation="held".

## OUTPUT FORMAT:
```json
{{
  "objects": [
    {{"instance": "<instance_name>", "category": "<name>", "relation": "support|attached", "parent": "<parent_instance_name_or_floor_or_wall_or_other_objects>"}}
  ]
}}
```
Where parent is one of:
- another object's instance name (e.g., "table_0") if the object clearly rests on / is attached to it;
- "floor" if it stands on the ground;
- "wall" if it is wall-mounted;
- "other_objects" ONLY IF you CANNOT determine the exact supporter but it is clearly ON/ASSOCIATED WITH some other object (NOT the floor/wall).

## CRITICAL REQUIREMENTS:
- Output ONLY the JSON code block above. No explanations, no reasoning, no text outside the JSON.
- You MUST output exactly {num_query} objects (one for each: {query_str}). Do NOT skip any object.
- Do NOT output any object outside the query list [{query_str}].
- Do NOT use relation="held" - every object must have a physical support relation.
- If you are unsure of the exact supporter but the object is on/with another object, use parent="other_objects". Do NOT default to "floor" unless it clearly stands on the ground.

Now output the complete JSON with all {num_query} objects.
'''

    caller = "stage5_scene_graph"
    # Qwen3.5-VL processor 对字符串只按 URL/base64 解析, 不认本地文件路径, 需先转 PIL Image
    image = Image.open(image_path).convert("RGB")
    # prompt 已要求"只输出 JSON、不写推理"。全场景 (25 物体≈1800 token) 用 4096
    # 规避截断; 按实例裁剪后仅数条记录, 256 足够且生成时间随 token 数线性下降。
    max_new_tokens = 4096 if num_query > 4 else 256
    return vlm_inference(image, model, processor, prompt, max_new_tokens=max_new_tokens, caller=caller)


def parse_scene_graph_output(output_text: str) -> Dict[str, Any]:
    """解析VLM输出为场景图JSON (支持新旧两种格式)

    新格式: {"objects": [{"instance": "duck_0", "parent": "plate_0", ...}]}
    旧格式: {"objects": [{"id": 3, "category": "duck", "parent": 4, ...}]}

    处理截断情况: 如果 JSON 不完整 (被 max_new_tokens 截断), 尝试提取前 N 个完整对象
    """
    # 1. 优先从 ```json ... ``` 代码块中提取
    if '```json' in output_text:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', output_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # 2. 从普通 ``` ... ``` 代码块中提取
    json_match = re.search(r'```\s*([\s\S]*?)\s*```', output_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 尝试匹配完整 JSON (无代码块)
    json_match = re.search(r'[{]\s*"objects"\s*:\s*\[[\s\S]*?\]\s*[{}]', output_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # 4. 截断兜底: 找到 JSON 开头, 逐对象截断直到能解析
    start = output_text.find('{"objects"')
    if start >= 0:
        truncated = output_text[start:]
        # 删除 ``` 代码块标记
        truncated = re.sub(r'```[\s\S]*?```', '', truncated)
        try:
            data = json.loads(truncated)
            return data
        except json.JSONDecodeError:
            # 逐步截断最后一个不完整对象
            obj_pattern = re.compile(r'[{]\s*"instance"\s*:\s*"[^"]+"\s*,\s*"category"\s*:\s*"[^"]+"\s*,\s*"relation"\s*:\s*"[^"]+"\s*,\s*"parent"\s*:\s*"[^"]*"\s*[{}]')
            matches = obj_pattern.findall(truncated)
            if matches:
                # 用前 N 个完整对象重建 JSON
                for n in range(len(matches), 0, -1):
                    partial = {"objects": matches[:n]}
                    try:
                        return json.loads(json.dumps(partial))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    # 如果字符串里的 matches 已经是 dict 格式需要重新构造
                # fallback: 用字符串拼接
                for n in range(len(matches), 0, -1):
                    objs_json = "[" + ", ".join(matches[:n]) + "]"
                    try:
                        return json.loads('{"objects":' + objs_json + "}")
                    except json.JSONDecodeError:
                        continue

    return {"objects": [], "parse_error": True, "raw_output": output_text}


def post_process_scene_graph(objects: List[Dict]) -> List[Dict]:
    """
    物理常识后处理: 纠正VLM的常见错误 (适配新格式)

    新格式: {"instance": "duck_0", "category": "duck", "relation": "support", "parent": "plate_0"}
    """
    filtered = []
    for obj in objects:
        category = obj.get('category', '').lower().replace(' ', '_')
        relation = obj.get('relation', 'support')
        parent = obj.get('parent', 'floor')

        if category in FILTER_CATEGORIES:
            continue
        if 'duplicate' in category or 'same_as' in category:
            continue

        category_base = category.split('_')[0] if '_' in category else category

        # 规则 1: 家具类不应挂在墙上
        if relation == 'attached' and (parent == 'wall' or parent == 2):
            if category in MUST_BE_FLOOR_SUPPORTED or category_base in MUST_BE_FLOOR_SUPPORTED:
                relation = 'support'
                parent = 'floor'

        # 规则 2: 挂墙类不应站在地上
        if relation == 'support' and (parent == 'floor' or parent == 1):
            if category in MUST_BE_WALL_ATTACHED or category_base in MUST_BE_WALL_ATTACHED:
                if category_base in {'picture', 'painting', 'photo', 'mirror', 'clock', 'poster'}:
                    relation = 'attached'
                    parent = 'wall'
                elif category_base in {'window', 'door', 'vent', 'air_vent', 'outlet', 'socket'}:
                    relation = 'attached'
                    parent = 'wall'

        filtered.append({
            'instance': obj.get('instance', category),
            'category': obj.get('category', 'unknown'),
            'relation': relation,
            'parent': parent,
        })

    return filtered


def vote_relations(votes: List[Tuple[str, int]]) -> str:
    """
    多数投票决定关系

    参数:
        votes: [(关系字符串, 权重), ...]
    返回:
        投票决定的关系字符串
    """
    weighted_votes = defaultdict(int)
    for rel, weight in votes:
        weighted_votes[rel] += weight

    if not weighted_votes:
        # 返回 None 表示"无法判断"，调用方需根据原始关系决定保留还是降级
        return None

    # 平票且无明确多数时返回 None, 而非让 max() 按插入序随意挑一个。
    # 实测 egodex_33 toy_0: 票为 {'supported by bed': 2, 'supported by lego_0': 2},
    # 'bed' 是不存在的幻觉支撑物被 Step 2.5 剔除后, 剩下的 lego_0 恰好与它
    # 同为 2 票 —— max() 会返回其中一个, 使"支撑物是否存在"退化成随机选择。
    # 这里显式把平票判为"无法判断", 由调用方回退到Stage1 原关系 (真实关系)。
    top_count = max(weighted_votes.values())
    if sum(1 for c in weighted_votes.values() if c == top_count) > 1:
        return None

    return max(weighted_votes, key=weighted_votes.get)


def _get_original_relation(categories_and_relations: Dict[str, str], inst_name: str) -> str:
    """
    按实例名查询原始 (Stage 1) 关系，兼容 key 形式不一致:

      - 直接命中实例级 key (如 "bottle_0") → 返回其值
      - 命中类别级 key (如 "bottle"，inst_name 去掉 _数字 后缀) → 返回其值
      - 都查不到 → 返回默认 "supported by floor"

    这是防止 "original=other objects 被错误降级为 floor" 的关键:
    若查询不兼容，inst_name 永远查不到，会回退成 floor，破坏 Stage 1 输入。
    """
    if inst_name in categories_and_relations:
        return categories_and_relations[inst_name]
    # 去掉末尾 _<数字> 后缀，回退到类别级 key
    if '_' in inst_name:
        base = inst_name.rsplit('_', 1)
        if len(base) == 2 and base[1].isdigit():
            category_key = base[0]
            if category_key in categories_and_relations:
                return categories_and_relations[category_key]
    return "supported by floor"


def infer_relations_scene_graph(
    scene_dir: str,
    vlm_checkpoint: str,
    categories_and_relations: Dict[str, str],
    instance_masks: Dict[str, List[Dict]] = None,
    colors: List[np.ndarray] = None,
    id_scene_path: str = None,
    output_dir: str = None,
    preloaded_vlm: tuple = None,
    optimal_frames_dir: str = None,
    keyframes_dir: str = None,
    dynamic_frames_dir: str = None,
) -> tuple:
    """
    使用多帧投票方式推断物体间关系

    参数:
        scene_dir: 场景输出目录
        vlm_checkpoint: VLM模型路径
        categories_and_relations: 原始关系字典
        instance_masks: mask数据 (如果为None, 尝试从磁盘加载)
        colors: 帧图像列表 (如果为None, 尝试从磁盘加载)
        id_scene_path: 已有的ID标注图路径 (如果为None, 自动创建)
        output_dir: 输出目录 (默认为scene_dir)
        preloaded_vlm: 预加载的VLM (model, processor), 避免重复加载
        optimal_frames_dir: optimal_frames 目录路径 (新增)
        keyframes_dir: keyframes 目录路径 (新增)
        dynamic_frames_dir: dynamic_pointclouds 目录路径 (用于动态物体首次出现帧权重)
    返回:
        (refined_relations, vlm_or_None): 更新后的关系字典 + VLM模型(供后续使用)
    """
    if output_dir is None:
        output_dir = scene_dir
    output_dir = os.path.abspath(output_dir)

    # 创建 labeled_images 子目录
    labeled_images_dir = os.path.join(output_dir, "labeled_images")
    os.makedirs(labeled_images_dir, exist_ok=True)

    # ── Step 0: 准备数据 ──
    print(f"   📷 准备数据...", flush=True)

    # 加载 instance_visibility.json 获取实例名
    instance_visibility = {}
    if optimal_frames_dir:
        vis_path = os.path.join(optimal_frames_dir, "instance_visibility.json")
        if os.path.exists(vis_path):
            with open(vis_path, 'r') as f:
                instance_visibility = json.load(f)

    # 构建实例名列表
    instance_names = []
    for key in categories_and_relations:
        if key.lower() in FILTER_CATEGORIES:
            continue
        if '_' in key and key.rsplit('_', 1)[-1].isdigit():
            instance_names.append(key)
        else:
            # 类别级 key，从 instance_visibility 展开
            if key in instance_visibility:
                for inst_key in instance_visibility[key]:
                    instance_names.append(f"{key}_{inst_key}")
            else:
                instance_names.append(key)

    if not instance_names:
        print("   ⚠️ 无可见物体，跳过场景图推断", flush=True)
        return dict(categories_and_relations), None

    print(f"   📋 实例列表: {instance_names}", flush=True)

    # 加载 keyframes 可见性数据
    keyframe_visibility = {}
    if keyframes_dir:
        meta_path = os.path.join(keyframes_dir, "keyframes_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            raw_visibility = metadata.get("frame_visibility", {})
            for fidx_str, visible_names in raw_visibility.items():
                vid_idx = int(fidx_str)
                matched = []
                for raw_name in visible_names:
                    # 模糊匹配到实例名
                    found = False
                    for inst_name in instance_names:
                        base = inst_name.split('_')[0]
                        raw_base = raw_name.lower().strip().replace(' ', '_')
                        if raw_base in base or base in raw_base:
                            matched.append(inst_name)
                            found = True
                            break
                    if not found:
                        matched.append(raw_name)
                keyframe_visibility[vid_idx] = matched

    # 加载动态物体 trajectory 信息 — 记录每个动态实例的首次出现帧号
    # 注意: traj_info key 格式为 "{category}_inst{idx}" (如 bottle_inst0),
    # 而 instance_names 格式为 "{category}_{idx}" (如 bottle_0)，需要归一化
    dynamic_first_frame_map = {}  # {inst_name: first_visible_frame_id}
    if dynamic_frames_dir and os.path.isdir(dynamic_frames_dir):
        traj_path = os.path.join(dynamic_frames_dir, "trajectory_info.json")
        if os.path.exists(traj_path):
            with open(traj_path, 'r') as f:
                traj_info = json.load(f)
            for traj_key, info in traj_info.items():
                if not info.get("is_dynamic") or not info.get("visible_frames"):
                    continue
                # 将 traj key 转为 instance_name 格式: bottle_inst0 → bottle_0
                if '_inst' in traj_key:
                    cat, idx = traj_key.split('_inst')
                    inst_name = f"{cat}_{idx}"
                else:
                    inst_name = traj_key
                first_frame = min(info["visible_frames"])
                dynamic_first_frame_map[inst_name] = first_frame
                print(f"   🔄 动态物体 {inst_name}: 首次出现在帧 #{first_frame}", flush=True)

    # 加载 colors 目录
    if colors is None:
        colors = _load_colors_from_disk(scene_dir)

    if colors is None:
        print("   ❌ 无法加载图像数据，跳过场景图推断", flush=True)
        return dict(categories_and_relations), None

    # ── Step 1: 全场景预览 (1 次 VLM 调用，仅供参考) ──
    print(f"   📷 Step 1: 全场景预览...", flush=True)

    # 加载 VLM
    if preloaded_vlm is not None:
        model, processor = preloaded_vlm
        print(f"   🤖 使用预加载VLM", flush=True)
    else:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq, AutoProcessor
            except ImportError:
                print("   ❌ 无法导入transformers，跳过场景图推断", flush=True)
                return dict(categories_and_relations), None

        print(f"   🤖 加载VLM: {vlm_checkpoint}", flush=True)
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(vlm_checkpoint, trust_remote_code=True)

    # 选一帧能显示最多物体的帧做预览
    if keyframe_visibility:
        best_frame_id = max(keyframe_visibility, key=lambda k: len(keyframe_visibility[k]))
    else:
        best_frame_id = 0

    # ── Step 1: 全场景预览 + 投票 ──
    print(f"   📷 Step 1: 全场景预览投票 (frame {best_frame_id})...", flush=True)
    preview_votes = defaultdict(lambda: defaultdict(int))  # {inst_name: {rel_str: weight}}

    if best_frame_id < len(colors):
        frame_image = colors[best_frame_id]
        visible_in_frame = keyframe_visibility.get(best_frame_id, instance_names)
        labeled_image = create_labeled_frame_image(frame_image, visible_in_frame, instance_masks, best_frame_id)
        preview_path = os.path.join(output_dir, "labeled_images", "preview_id_scene.png")
        cv2.imwrite(preview_path, cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))

        raw_output = infer_relations_from_labeled_image(model, processor, preview_path, instance_names)
        parsed = parse_scene_graph_output(raw_output)
        for obj in parsed.get("objects", []):
            inst = obj.get("instance", "")
            if not inst or inst not in instance_names:
                continue
            relation = obj.get("relation", "support")
            # 无法判断具体支撑物时默认 "other_objects" (保留原 stage1 语义)，不默认 floor
            parent = obj.get("parent", "other_objects")
            if parent == "other_objects":
                rel_str = "supported by other objects"
            elif relation == "support" and parent == "floor":
                rel_str = "supported by floor"
            elif relation == "support":
                rel_str = f"supported by {parent}"
            elif relation == "attached":
                rel_str = f"attached to {parent}"
            else:
                rel_str = "supported by other objects"
            preview_votes[inst][rel_str] += 1  # 全场景投票权重 1
            print(f"   📊 preview: {inst} → {rel_str}", flush=True)
        print(f"   📊 全场景预览共解析 {len(parsed.get('objects', []))} 个物体", flush=True)
    else:
        raw_output = ""
        print(f"   ⚠️ 预览帧不可用，跳过全场景投票", flush=True)

    # ── Step 2: 逐实例 3 图投票 ──
    print(f"   📷 Step 2: 逐实例多帧投票...", flush=True)

    refined_relations = {}
    inference_details = {}
    optimal_frame_map = {}

    # 从 optimal_frames 目录解析每个实例的最优帧号
    if optimal_frames_dir and os.path.isdir(optimal_frames_dir):
        for fname in os.listdir(optimal_frames_dir):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            match = re.match(r'(.+)_inst(\d+)_frame(\d+)', os.path.splitext(fname)[0])
            if match:
                category = match.group(1)
                inst_idx = match.group(2)
                frame_id = int(match.group(3))
                inst_name = f"{category}_{inst_idx}"
                optimal_frame_map[inst_name] = frame_id

    # ── 方案1: 按帧分组, 每帧只调用一次 VLM ──
    # 原代码对每个实例单独推理, 但同一帧的标注图 + prompt 完全相同 (只是
    # 事后从全场景 JSON 里挑出自己那一行, 其余全部丢弃) —— 25 实例时是
    # O(N^2) 的重复生成。先把每个实例需要的帧收集起来, 按帧去重, 每帧一次
    # 调用 + query_instances 只问该帧真正要用的实例 (配合方案2 裁剪输出)。
    print(f"   📷 Step 2: 按帧分组投票...", flush=True)

    # 收集每个实例的投票来源: {frame_id: [(inst_name, weight, kind), ...]}
    frame_queries = defaultdict(list)
    for inst_name in instance_names:
        # 图1: optimal_frames 对应的帧 (权重 2)
        if inst_name in optimal_frame_map:
            frame_id = optimal_frame_map[inst_name]
            if frame_id < len(colors):
                frame_queries[frame_id].append((inst_name, 2, "optimal"))

        # 图2~3: 从 keyframes 中找该实例可见的其他帧
        keyframe_candidates = []
        for vid_idx, visible in keyframe_visibility.items():
            if inst_name in visible or any(inst_name.split('_')[0] in str(v) for v in [visible]):
                keyframe_candidates.append(vid_idx)

        # 动态物体: 锁定 trajectory 记录的首次出现帧作为主导帧 (权重 10)
        is_dynamic = inst_name in dynamic_first_frame_map
        if is_dynamic:
            first_dynamic_frame = dynamic_first_frame_map[inst_name]
            ordered = [first_dynamic_frame] + [f for f in keyframe_candidates if f != first_dynamic_frame]
            keyframe_candidates = ordered
        else:
            keyframe_candidates.sort()

        # 关键帧投票上限 1 帧 (每帧 VLM 调用 ~40s)
        for kf_idx, kf_frame_id in enumerate(keyframe_candidates[:1]):
            if kf_frame_id == optimal_frame_map.get(inst_name):
                continue
            if kf_frame_id >= len(colors):
                continue
            weight = 10 if (is_dynamic and kf_idx == 0) else 1
            frame_queries[kf_frame_id].append((inst_name, weight, "keyframe"))

    # 每帧渲染一次标注图 + 调用一次 VLM, 结果缓存到 frame_results
    frame_results = {}   # (frame_id, inst_name) -> rel_str
    n_frames = len(frame_queries)
    print(f"   ⚡ {len(instance_names)} 个实例聚合为 {n_frames} 帧 (去重前约 "
          f"{len(instance_names) * 2} 次调用)", flush=True)

    for frame_id, queries in sorted(frame_queries.items()):
        query_instances = [inst for inst, _, _ in queries]
        frame_image = colors[frame_id]
        visible_in_frame = keyframe_visibility.get(frame_id, query_instances)
        labeled_image = create_labeled_frame_image(frame_image, visible_in_frame, instance_masks, frame_id)
        labeled_path = os.path.join(output_dir, "labeled_images", f"label_frame{frame_id}.png")
        cv2.imwrite(labeled_path, cv2.cvtColor(labeled_image, cv2.COLOR_RGB2BGR))

        # 只问该帧真正需要的实例 (方案2: 输出规模从 N 降到 len(query_instances))
        raw_out = infer_relations_from_labeled_image(
            model, processor, labeled_path, instance_names,
            query_instances=query_instances)
        parsed = parse_scene_graph_output(raw_out)
        for obj in parsed.get("objects", []):
            inst = obj.get("instance", "")
            if inst not in query_instances:
                continue
            relation = obj.get("relation", "support")
            parent = obj.get("parent", "other_objects")
            if parent == "other_objects":
                rel_str = "supported by other objects"
            elif relation == "support" and parent == "floor":
                rel_str = "supported by floor"
            elif relation == "support":
                rel_str = f"supported by {parent}"
            elif relation == "attached":
                rel_str = f"attached to {parent}"
            else:
                rel_str = "supported by other objects"
            frame_results[(frame_id, inst)] = rel_str

    # 汇总每个实例的加权票 (含全场景预览票)
    for inst_name in instance_names:
        votes = list(preview_votes[inst_name].items())  # 先加入全场景预览投票

        for frame_id, queries in frame_queries.items():
            for inst, weight, kind in queries:
                if inst != inst_name:
                    continue
                rel_str = frame_results.get((frame_id, inst_name))
                if rel_str:
                    votes.append((rel_str, weight))

        # 投票决定关系
        # 注意: categories_and_relations 的 key 可能是类别级 ("bottle") 或实例级
        # ("bottle_0")。inst_name 永远是实例级，需兼容两种 key 形式，否则查不到
        # 时会错误回退成 floor，导致原 "other objects" 被降级。
        original_relation = _get_original_relation(categories_and_relations, inst_name)
        if votes:
            decision = vote_relations(votes)
        else:
            decision = None

        if decision is None:
            # VLM 无法判断关系: 若原关系就是"与其他物体有交互"，则保持原判定，
            # 不降级为 floor；否则默认 floor
            if original_relation == "supported by other objects":
                final_relation = original_relation
                print(f"   ⚠️ {inst_name}: VLM未判断出交互对象，保留原始关系"
                      f"'{original_relation}' (不降级为 floor)", flush=True)
            else:
                final_relation = "supported by floor"
        else:
            # VLM 有投票结果: 但仍需保护 Stage 1 输入不被错误覆盖
            #   - 原关系是 "other objects" 且 VLM 仍没给出具体支撑物
            #     (decision 是 floor 或 other objects) → 保持原 other objects，不降级
            #   - 原关系是 floor 且 VLM 给出 other objects(未指明具体物) → 保持 floor
            if original_relation == "supported by other objects":
                if decision == "supported by floor" or decision == "supported by other objects":
                    final_relation = original_relation
                    print(f"   ⚠️ {inst_name}: 原关系为 other objects，VLM未明确具体支撑物"
                          f"({decision})，保留原始关系不降级", flush=True)
                else:
                    final_relation = decision
            elif original_relation == "supported by floor":
                if decision == "supported by other objects":
                    final_relation = original_relation
                    print(f"   ⚠️ {inst_name}: 原关系为 floor，VLM未明确具体支撑物"
                          f"(other objects)，保持 floor 不被篡改", flush=True)
                else:
                    final_relation = decision
            else:
                final_relation = decision

        refined_relations[inst_name] = final_relation
        inference_details[inst_name] = {
            "votes": dict(defaultdict(int, {v: w for v, w in votes})),
            "frame_count": len(votes),
        }
        vote_counts = defaultdict(int)
        for v, w in votes:
            vote_counts[v] += w
        print(f"   ✅ {inst_name}: {final_relation} ({len(votes)} 票: {dict(vote_counts)})", flush=True)

    # ── Step 2.5: 支撑物存在性校验 (剔除幻觉支撑物) ──
    # VLM 可能给出场景里根本不存在的支撑物 (实测 egodex_1: 检测类别里没有 table,
    # 却推断出 "supported by table_0")。下游遇到不存在的支撑物只会 "跳过",
    # 该物体的位置精修被静默丢掉。必须在关系落盘前拦掉。
    valid_supporters = set(instance_names) | {"floor", "wall", "other_objects"}

    def _parent_of(rel):
        return rel.split("supported by ", 1)[1].strip() if "supported by " in rel else None

    fixed_relations = {}
    n_halluc = 0
    for inst_name, rel in refined_relations.items():
        parent = _parent_of(rel)
        if parent is not None and parent not in valid_supporters:
            n_halluc += 1
            print(f"   ⚠️ {inst_name}: 支撑物 '{parent}' 不存在于场景, 剔除该票", flush=True)
            # 用该实例的其他有效票重新决策
            all_votes = inference_details.get(inst_name, {}).get("votes", {})
            alt = [(v, w) for v, w in all_votes.items()
                   if not (v.startswith("supported by ") and
                           v.split("supported by ", 1)[1].strip() not in valid_supporters)]
            new_rel = vote_relations(alt) if alt else None
            if new_rel is None:
                # 无可用票: 回退 Stage1 原关系 (至少是真实关系)
                orig = _get_original_relation(categories_and_relations, inst_name)
                new_rel = orig if orig != "supported by other objects" else "supported by floor"
                print(f"      无其他有效票, 回退原关系 '{new_rel}'", flush=True)
            else:
                print(f"      改用其他有效票 -> '{new_rel}'", flush=True)
            fixed_relations[inst_name] = new_rel
            continue
        fixed_relations[inst_name] = rel

    if n_halluc:
        print(f"   🔍 共剔除 {n_halluc} 个幻觉支撑物", flush=True)
    refined_relations = fixed_relations

    # ── Step 3: 物理常识后处理 ──
    print(f"   🔧 Step 3: 物理常识后处理...", flush=True)
    corrected_relations = post_process_relationships(refined_relations, refined_relations)

    # ── Step 4: 构建 scene_graph_objects ──
    scene_graph_objects = []
    for inst_name, rel in corrected_relations.items():
        category = inst_name.rsplit('_', 1)[0] if '_' in inst_name else inst_name
        try:
            inst_idx = int(inst_name.rsplit('_', 1)[1])
        except (ValueError, IndexError):
            inst_idx = 0

        if "attached to" in rel:
            sg_relation = "attached"
            sg_parent = rel.replace("attached to ", "")
        elif "supported by" in rel:
            sg_relation = "support"
            sg_parent = rel.replace("supported by ", "")
        else:
            sg_relation = "support"
            sg_parent = "floor"

        scene_graph_objects.append({
            "instance": inst_name,
            "category": category,
            "inst_idx": inst_idx,
            "relation": sg_relation,
            "parent": sg_parent,
        })

    # ── Step 5: 保存结果 ──
    refined_json_path = os.path.join(output_dir, "relations_scene_graph.json")
    with open(refined_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'refined_relations': corrected_relations,
            'scene_graph_objects': scene_graph_objects,
            'inference_details': inference_details,
        }, f, indent=2, ensure_ascii=False)
    print(f"   💾 场景图关系已保存: {refined_json_path}", flush=True)

    # 不卸载VLM, 返回给调用者供后续使用 (5.2 SP精修需要VLM判断精修程度)
    return corrected_relations, (model, processor)


def post_process_relationships(relationships: Dict[str, str], all_relations: Dict[str, str]) -> Dict[str, str]:
    """
    物理常识后处理: 纠正VLM投票后的关系

    参数:
        relationships: {实例名: 关系} VLM投票后的关系
        all_relations: {实例名: 关系} 当前所有物体的关系
    返回:
        corrected: {实例名: 关系} 纠错后的关系
    """
    corrected = {}
    for target_name, rel in relationships.items():
        name = target_name.lower().strip().replace(' ', '_')
        category_base = name.split('_')[0]

        # 规则 1: 家具类不应挂在墙上
        if name in MUST_BE_FLOOR_SUPPORTED or category_base in MUST_BE_FLOOR_SUPPORTED:
            if "attached to wall" in rel or "embedded in wall" in rel:
                print(f"  [纠错] '{target_name}': {rel} → supported by floor (物理常识)")
                rel = "supported by floor"

        # 规则 2: 挂墙类不应站在地上
        if name in MUST_BE_WALL_ATTACHED or category_base in MUST_BE_WALL_ATTACHED:
            if rel == "supported by floor":
                if category_base in {'picture', 'painting', 'photo', 'mirror', 'clock', 'poster'}:
                    print(f"  [纠错] '{target_name}': supported by floor → attached to wall (物理常识)")
                    rel = "attached to wall"
                elif category_base in {'window', 'door', 'vent', 'air_vent', 'outlet', 'socket'}:
                    print(f"  [纠错] '{target_name}': supported by floor → embedded in wall (物理常识)")
                    rel = "embedded in wall"

        # 规则 3: "supported by X" 中 X 本身也是 "supported by other objects" → 修正为 supported by floor
        if "supported by" in rel and "floor" not in rel and "other objects" not in rel:
            supporter_name = rel.replace("supported by ", "").strip()
            if supporter_name in all_relations:
                supporter_rel = all_relations.get(supporter_name, "")
                if supporter_rel == "supported by other objects":
                    print(f"  [纠错] '{target_name}': 支撑物 '{supporter_name}' 本身也是 'other objects' → supported by floor")
                    rel = "supported by floor"

        corrected[target_name] = rel

    return corrected


def _load_instance_masks_from_disk(scene_dir):
    """从磁盘加载mask数据 (回退方案: 使用optimal_frames推断)"""
    visibility_path = os.path.join(scene_dir, "optimal_frames", "instance_visibility.json")
    if not os.path.exists(visibility_path):
        return None

    with open(visibility_path, 'r') as f:
        visibility = json.load(f)

    instance_masks = {}
    for category, instances in visibility.items():
        cat_masks = []
        for inst_idx_str, frame_ids in instances.items():
            mask_list = []
            for fid in frame_ids:
                mask_list.append({'frame_id': fid, 'mask': np.zeros((1, 1), dtype=np.uint8)})
            cat_masks.append(mask_list)
        instance_masks[category] = cat_masks

    return instance_masks


def _load_colors_from_disk(scene_dir):
    """从磁盘加载帧图像"""
    color_dir = os.path.join(scene_dir, "color")
    if not os.path.isdir(color_dir):
        return None

    colors = []
    for cf in sorted(os.listdir(color_dir), key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0):
        if cf.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(color_dir, cf))
            if img is not None:
                colors.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return colors if colors else None


def main():
    parser = argparse.ArgumentParser(
        description="SimRecon风格的场景图关系推断",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景输出目录")
    parser.add_argument("--vlm_checkpoint", type=str, required=True,
                        help="VLM模型路径")
    parser.add_argument("--id_scene_path", type=str, default=None,
                        help="已有的ID标注图路径 (默认自动创建)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认为scene_dir)")

    args = parser.parse_args()

    # 加载原始关系
    stage1_json = None
    for f in os.listdir(args.scene_dir):
        if f.endswith('_stage1.json') and 'refined' not in f:
            stage1_json = os.path.join(args.scene_dir, f)
            break
    if stage1_json is None:
        for f in os.listdir(args.scene_dir):
            if f.endswith('.json') and 'stage1' in f:
                stage1_json = os.path.join(args.scene_dir, f)
                break

    if stage1_json is None:
        final_rel = os.path.join(args.scene_dir, "final_relations.json")
        if os.path.exists(final_rel):
            with open(final_rel, 'r') as f:
                categories_and_relations = json.load(f)
        else:
            print(f"❌ 未找到场景JSON", flush=True)
            sys.exit(1)
    else:
        with open(stage1_json, 'r') as f:
            categories_and_relations = json.load(f)

    print(f"📋 原始关系: {json.dumps(categories_and_relations, ensure_ascii=False)}", flush=True)

    refined, _ = infer_relations_scene_graph(
        scene_dir=args.scene_dir,
        vlm_checkpoint=args.vlm_checkpoint,
        categories_and_relations=categories_and_relations,
        id_scene_path=args.id_scene_path,
        output_dir=args.output_dir or args.scene_dir,
    )

    print(f"\n📋 推断后关系: {json.dumps(refined, ensure_ascii=False)}", flush=True)

    final_rel = os.path.join(args.output_dir or args.scene_dir, "final_relations.json")
    with open(final_rel, 'w') as f:
        json.dump(refined, f, indent=2, ensure_ascii=False)
    print(f"💾 最终关系已保存: {final_rel}", flush=True)


if __name__ == "__main__":
    main()
