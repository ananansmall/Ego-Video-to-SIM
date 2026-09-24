"""
Stage 6: Material-aware Physics Annotation
==========================================

VLM 推断场景中每个物体的材质类型（如 hardwood, glass, hard_plastic），
再通过 materials_db.json 映射为 SAPIEN 物理参数（密度/摩擦/恢复系数）。

架构: Visual Analysis → Material Recognition (VLM) → Material DB → Physics Params

使用方式:
  # 从 mainv2.py 自动调用 (推荐)
  python mainv2.py --input_video ./232.mp4 --enable_stage6

  # 独立调用
  python tools/material_annotator.py \
      --scene_dir output_v2/232_vggt \
      --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B
"""

import os
import sys
import json
import re
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

# 复用 Stage 5.1 的过滤类别常量
from tools.infer_relations_scene_graph import FILTER_CATEGORIES


MATERIALS_DB_PATH = os.path.join(os.path.dirname(__file__), "materials_db.json")

SUPPORTED_MATERIALS = [
    "hardwood", "softwood", "hard_plastic", "soft_plastic", "glass", "ceramic",
    "steel", "aluminum", "copper", "rubber", "fabric", "paper", "foam", "leather",
    "concrete", "stone", "marble", "ice", "wood", "plastic", "metal"
]

MATERIAL_PROMPT_TEMPLATE = """You are a material recognition expert. Identify the PRIMARY material of each colored object region in this scene image.

## Supported Materials
hardwood, softwood, hard_plastic, soft_plastic, glass, ceramic, steel, aluminum, copper, rubber, fabric, paper, foam, leather, concrete, stone, marble, ice, wood, plastic, metal

## Instructions

For each colored region in the image:
1. Look at the object's surface appearance: color, reflectance (glossy/matte/metallic/translucent), texture (smooth/rough/grainy)
2. Identify its PRIMARY material from the list above
3. Be specific when possible: "hardwood" instead of just "wood"
4. If uncertain or looks synthetic, use generic names (plastic, metal, wood)

## Objects in This Scene
{object_list_str}

## Important Warning
The category labels may be INACCURATE (e.g., "toy" instead of the real object). ALWAYS trust your visual analysis of the actual object appearance over the text labels. You are looking at REAL physical objects in a real scene.

## Output Format

Respond with a JSON object ONLY (no markdown, no explanations):
{{
  "materials": {{
    "<category_name>": "<material_name>",
    ...
  }}
}}"""

MATERIAL_PROMPT_SINGLE = """You are a material recognition expert. Identify the PRIMARY material of this object.

## Supported Materials
{materials_list}

## Object Category
{object_list}

## Instructions
Examine the object's surface: color, reflectance (glossy/matte/metallic), texture (smooth/rough/grainy). Choose the most specific material from the list.

## Output Format
Respond with a JSON object ONLY:
{{
  "material": "<material_name>"
}}"""


def load_materials_db(db_path: str = MATERIALS_DB_PATH) -> Dict[str, Any]:
    with open(db_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _extract_single_material(output_text: str) -> str:
    """从 VLM 输出中提取单个材质名"""
    # 尝试从 markdown code block 中提取
    if '```json' in output_text:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', output_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "material" in parsed:
                    return parsed["material"]
            except json.JSONDecodeError:
                pass
    
    # 尝试直接解析 JSON
    json_match = re.search(r'\{\s*"material"\s*:\s*"([^"]+)"\s*\}', output_text)
    if json_match:
        return json_match.group(1)
    
    # 纯文本：直接匹配材质名
    for mat in SUPPORTED_MATERIALS:
        if mat in output_text.lower():
            return mat
    
    return ""


def query_vlm_materials(
    model,
    processor,
    image_path: str,
    category_names: List[str],
) -> Dict[str, str]:
    """
    调用 VLM 推断场景图中所有物体的材质类型

    参数:
        model: VLM 模型
        processor: VLM 处理器
        image_path: ID 标注图路径
        category_names: 类别名列表（去重后）
    返回:
        {category_name: material_name} 映射
    """
    object_list_str = "\n".join(f"- {name}" for name in category_names)
    prompt = MATERIAL_PROMPT_TEMPLATE.format(object_list_str=object_list_str)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def parse_material_output(output_text: str) -> Dict[str, str]:
    """解析 VLM 输出为 {category: material} 映射"""
    # 尝试从 markdown code block 中提取
    if '```json' in output_text:
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', output_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "materials" in parsed:
                    return parsed["materials"]
            except json.JSONDecodeError:
                pass

    if '```' in output_text:
        json_match = re.search(r'```\s*([\s\S]*?)\s*```', output_text)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if "materials" in parsed:
                    return parsed["materials"]
            except json.JSONDecodeError:
                pass

    # 尝试直接解析 JSON
    json_match = re.search(r'\{\s*"materials"\s*:\s*\{[\s\S]*?\}\s*\}', output_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "materials" in parsed:
                return parsed["materials"]
        except json.JSONDecodeError:
            pass

    return {}


def map_materials_to_params(
    material_map: Dict[str, str],
    materials_db: Dict[str, Any],
) -> Tuple[Dict[str, Dict], List[Dict], List[str]]:
    """
    将 VLM 输出的材质名映射为物理参数

    参数:
        material_map: {instance_key (如 bottle_inst0): material_name}
        materials_db: materials_db.json 内容
    返回:
        (per_category_params, mapping_log, unrecognized)
        per_category_params: {instance_key: {material, density, static_friction, ...}}
        mapping_log: 映射日志
        unrecognized: 未识别的材质名列表
    """
    per_category = {}
    mapping_log = []
    unrecognized = []

    for cat_name, material_name in material_map.items():
        if material_name in materials_db.get("materials", {}):
            params = materials_db["materials"][material_name]["sapien"].copy()
            params["material"] = material_name
            mapping_log.append({
                "category": cat_name,
                "recognized_material": material_name,
                "mapped_params": {k: v for k, v in params.items() if k != "material"},
            })
        else:
            params = materials_db.get("default", {}).get("sapien", {
                "density": 1000,
                "static_friction": 0.50,
                "dynamic_friction": 0.50,
                "restitution": 0.30,
            }).copy()
            params["material"] = "default"
            unrecognized.append(material_name)
            mapping_log.append({
                "category": cat_name,
                "recognized_material": material_name,
                "fallback": True,
                "mapped_params": {k: v for k, v in params.items() if k != "material"},
            })

        per_category[cat_name] = params

    return per_category, mapping_log, unrecognized


def run_stage6(
    output_path: str,
    all_instances: Dict[str, list],
    categories_and_relations: Dict[str, str],
    vlm_checkpoint: str,
    instance_masks: Dict[str, List[Dict]] = None,
    colors: List[np.ndarray] = None,
    preloaded_vlm: tuple = None,
) -> Dict[str, Dict]:
    """
    Stage 6: Material-aware Physics Annotation

    流程:
      1. 从 optimal_frames/ 加载每个类别的最佳裁剪图
      2. 对每个类别单独调用 VLM 判断材质（一张图一个类别）
      3. 查 materials_db.json → 映射为物理参数
      4. 输出 materials.json + materials_debug.json

    参数:
        output_path: 输出目录
        all_instances: 3D 资产实例
        categories_and_relations: 类别-关系字典
        vlm_checkpoint: VLM 模型路径
        instance_masks: mask 数据 (可选，为 None 时从磁盘加载)
        colors: 帧图像列表 (可选，为 None 时从磁盘加载)
        preloaded_vlm: 预加载的 VLM (model, processor)
    返回:
        materials_dict — 与 materials.json 同格式
    """
    print("\n" + "=" * 70, flush=True)
    print("Stage 6: Material-aware Physics Annotation", flush=True)
    print("=" * 70, flush=True)
    _t6 = time.time()

    # ── Step 1: 加载所有实例的裁剪图 ──
    optimal_dir = os.path.join(output_path, "optimal_frames")
    if not os.path.isdir(optimal_dir):
        print(f"   optimal_frames 目录不存在: {optimal_dir}", flush=True)
        materials_db = load_materials_db()
        return _fallback_default_materials(categories_and_relations, materials_db)

    # 过滤掉背景类别
    category_names = [cat for cat in categories_and_relations.keys() if cat.lower() not in FILTER_CATEGORIES]
    
    # 从 optimal_frames 中找到所有实例的裁剪图
    # 格式: {instance_key: (crop_path, area)}  如 "bottle_inst0": ("/path/to/file.jpg", 69858)
    instance_crops = {}
    for fname in os.listdir(optimal_dir):
        if not fname.endswith('.jpg'):
            continue
        match = re.match(r'(\w+)_inst(\d+)_frame\d+\.jpg', fname)
        if not match:
            continue
        cat = match.group(1)
        if cat not in category_names:
            continue
        fpath = os.path.join(optimal_dir, fname)
        area = os.path.getsize(fpath)
        inst_key = f"{cat}_inst{match.group(2)}"
        if inst_key not in instance_crops or area > instance_crops[inst_key][1]:
            instance_crops[inst_key] = (fpath, area)

    if not instance_crops:
        print("   未找到任何裁剪图，使用默认物理参数", flush=True)
        materials_db = load_materials_db()
        return _fallback_default_materials(categories_and_relations, materials_db)

    print(f"   找到 {len(instance_crops)} 个实例的裁剪图: {list(instance_crops.keys())}", flush=True)

    # ── Step 2: 加载 VLM ──
    if preloaded_vlm is not None:
        model, processor = preloaded_vlm
        print(f"   使用预加载 VLM", flush=True)
    else:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq, AutoProcessor
            except ImportError:
                print("   无法导入 transformers，使用默认物理参数", flush=True)
                materials_db = load_materials_db()
                return _fallback_default_materials(categories_and_relations, materials_db)

        print(f"   加载 VLM: {vlm_checkpoint}", flush=True)
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(vlm_checkpoint, trust_remote_code=True)

    # ── Step 3: 对每个实例单独调用 VLM 判断材质 ──
    material_map = {}  # {instance_key: material_name}
    per_call_debug = []

    for inst_key, (crop_path, _) in instance_crops.items():
        cat_name = inst_key.rsplit('_inst', 1)[0]
        print(f"   [{inst_key}] 裁剪图: {os.path.basename(crop_path)}", flush=True)

        # 调用 VLM 判断单个实例的材质
        materials_list = "\n".join(SUPPORTED_MATERIALS)
        prompt = MATERIAL_PROMPT_SINGLE.format(materials_list=materials_list, object_list=inst_key)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False
            )
        except TypeError:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        inputs = inputs.to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.1, repetition_penalty=1.1)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # 解析材质名
        material_name = _extract_single_material(output_text)
        if material_name:
            material_map[inst_key] = material_name
            print(f"   [{inst_key}] → {material_name}", flush=True)
        else:
            print(f"   [{inst_key}] VLM 输出解析失败: {output_text[:100]}", flush=True)

        per_call_debug.append({
            "instance": inst_key,
            "category": cat_name,
            "crop_path": crop_path,
            "raw_output": output_text[:500],
            "material": material_name,
        })

    if not material_map:
        print("   所有类别 VLM 解析失败，使用默认物理参数", flush=True)
        materials_db = load_materials_db()
        return _fallback_default_materials(categories_and_relations, materials_db)

    print(f"\n   材质识别结果: {json.dumps(material_map, ensure_ascii=False)}", flush=True)

    if not material_map:
        print("   VLM 输出解析失败，使用默认物理参数", flush=True)
        materials_db = load_materials_db()
        return _fallback_default_materials(categories_and_relations, materials_db)

    print(f"   材质识别结果: {json.dumps(material_map, ensure_ascii=False)}", flush=True)

    # ── Step 5: 映射为物理参数 ──
    materials_db = load_materials_db()
    per_category, mapping_log, unrecognized = map_materials_to_params(material_map, materials_db)

    if unrecognized:
        print(f"   未识别的材质 (已使用默认参数): {unrecognized}", flush=True)

    # ── Step 6: 为未在 VLM 输出中出现的实例补充默认参数 ──
    # 从 all_instances 中获取所有需要处理的实例
    expected_instances = set()
    for cat, insts in all_instances.items():
        if cat.lower() in FILTER_CATEGORIES:
            continue
        for i in range(len(insts)):
            expected_instances.add(f"{cat}_inst{i}")
    
    for inst_key in expected_instances:
        if inst_key not in per_category:
            default_params = materials_db.get("default", {}).get("sapien", {
                "density": 1000, "static_friction": 0.50,
                "dynamic_friction": 0.50, "restitution": 0.30,
            }).copy()
            default_params["material"] = "default"
            per_category[inst_key] = default_params

    # ── Step 7: 输出文件 ──
    # materials.json — 仿真器消费
    materials_json_path = os.path.join(output_path, "materials.json")
    with open(materials_json_path, 'w', encoding='utf-8') as f:
        json.dump(per_category, f, indent=2, ensure_ascii=False)
    print(f"   materials.json 已保存: {materials_json_path}", flush=True)

    # materials_debug.json — 调试用
    debug_out = {
        "material_recognition": {
            "materials": material_map,
        },
        "mapping_log": mapping_log,
        "unrecognized": unrecognized,
        "per_category_calls": per_call_debug,
    }
    debug_json_path = os.path.join(output_path, "materials_debug.json")
    with open(debug_json_path, 'w', encoding='utf-8') as f:
        json.dump(debug_out, f, indent=2, ensure_ascii=False)
    print(f"   materials_debug.json 已保存: {debug_json_path}", flush=True)

    _stage6_time = time.time() - _t6
    print(f"Stage 6 完成 ({_stage6_time:.1f}s)", flush=True)

    return per_category


def _fallback_default_materials(
    categories_and_relations: Dict[str, str],
    materials_db: Dict[str, Any],
) -> Dict[str, Dict]:
    """当 VLM 不可用时，为所有类别生成默认物理参数"""
    default_params = materials_db.get("default", {}).get("sapien", {
        "density": 1000, "static_friction": 0.50,
        "dynamic_friction": 0.50, "restitution": 0.30,
    }).copy()
    default_params["material"] = "default"

    result = {}
    for cat in categories_and_relations:
        if cat.lower() in FILTER_CATEGORIES:
            continue
        result[cat] = dict(default_params)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6: Material-aware Physics Annotation",
    )
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景输出目录")
    parser.add_argument("--vlm_checkpoint", type=str, required=True,
                        help="VLM 模型路径")
    parser.add_argument("--id_scene_path", type=str, default=None,
                        help="已有的 ID 标注图路径 (默认自动创建)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认为 scene_dir)")

    args = parser.parse_args()
    output_dir = args.output_dir or args.scene_dir

    # 加载原始关系
    final_rel = os.path.join(args.scene_dir, "final_relations.json")
    if os.path.exists(final_rel):
        with open(final_rel, 'r', encoding='utf-8') as f:
            categories_and_relations = json.load(f)
    else:
        # 查找 stage1 JSON
        stage1_json = None
        for fname in os.listdir(args.scene_dir):
            if fname.endswith('_stage1.json') and 'refined' not in fname:
                stage1_json = os.path.join(args.scene_dir, fname)
                break
        if stage1_json:
            with open(stage1_json, 'r', encoding='utf-8') as f:
                categories_and_relations = json.load(f)
        else:
            print(f"未找到场景 JSON", flush=True)
            sys.exit(1)

    print(f"类别: {list(categories_and_relations.keys())}", flush=True)

    result = run_stage6(
        output_path=output_dir,
        all_instances={},
        categories_and_relations=categories_and_relations,
        vlm_checkpoint=args.vlm_checkpoint,
    )

    print(f"\n材质参数结果:", flush=True)
    for cat, params in result.items():
        print(f"  {cat}: material={params['material']}, density={params['density']}, "
              f"friction={params['static_friction']}/{params['dynamic_friction']}, "
              f"restitution={params['restitution']}", flush=True)


if __name__ == "__main__":
    main()
