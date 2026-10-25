"""
物体间支撑关系空间位置精修 (refine_inter_object_placement.py)
============================================================

本文件负责: 已知物体间支撑关系后，判定放置策略，并用几何计算精修物体空间位置。

输入:
  - all_instances: 3D资产实例 (每个含 original_mesh + T变换矩阵)
  - refined_relations: 细化后的关系字典
      { "bowl_0": "supported by table", "table": "supported by floor", ... }
  - vlm_checkpoint + scene_dir: VLM模型路径 + 场景目录 (用于策略判定)

输出:
  - 更新后的 all_instances (T矩阵已精修)

================================================================================
  核心流程
================================================================================

  refined_relations
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Step 1: 策略判定                        │
  │  输入: 物体名称 + 场景图像               │
  │  VLM看图判断: bowl放在table上是什么方式?  │
  │  输出: "on_top" / "inside" / ...         │
  │                                          │
  │  帧来源: optimal_frames/ + keyframes/    │
  │  投票: 多帧VLM推理 → 取多数策略          │
  └─────────────┬───────────────────────────┘
                │
                ▼
  ┌─────────────────────────────────────────┐
  │  Step 2: SP几何精修                      │
  │  根据策略施加物理约束，修改T矩阵          │
  │  只做约束方向的最小移动，保留原始位置      │
  └─────────────────────────────────────────┘

================================================================================
  五种放置策略及物理约束实现
================================================================================

  ── on_top: 放在支撑物顶面 ──

    场景: 杯子在桌上、枕头在床上、书在架子上

    物理约束:
      supported.bottom_z >= supporter.top_z  (不穿入支撑物)
      supported.bottom_z <= supporter.top_z  (不悬空)

    实现:
      z_offset = supporter.top_z - supported.bottom_z

      z_offset < 0     → 物体穿入支撑物，向上推到顶面
      z_offset ≈ 0     → 已接触，无需调整
      0 < z_offset < 阈值 → 物体悬空，向下落到顶面
      z_offset 很大    → 物体远在上方，可能是误判，跳过

      supporter顶面z ───────┐
                           │ z_offset
      supported底面z ───────┘ → 对齐到顶面

  ── inside: 放在支撑物内部 ──

    场景: 衣服在抽屉里、碗在柜子里

    物理约束:
      supported.bottom_z >= supporter.bottom_z  (不穿出底部)
      supported.top_z <= supporter.top_z        (不穿出顶部)
      supported.bottom_z ≈ supporter内部30%高度  (合理放置)

    实现:
      1. 先把 supported 底面对齐到 supporter 内部30%高度
      2. 检查是否穿出顶部 → 如果穿出则上移到刚好不穿出
      3. 检查是否穿出底部 → 如果穿出则上移到刚好在内部

      supporter顶面z ───────┐
                           │ 70%
      target_z (30%) ──────┤ ← supported底面对齐到这里
                           │ 30%
      supporter底面z ───────┘

  ── against_side: 靠在支撑物侧面 ──

    场景: 柜子靠墙、画贴墙

    物理约束:
      1. 底面贴地: supported.bottom_z >= 0
      2. 侧面接触: supported与supporter在x/y轴刚好接触
      3. 不穿模: 侧面接触后，如果z轴穿入supporter内部，上移到顶面

    实现:
      # 约束1: 底面贴地
      if supported.bottom_z < 0:
          z_offset = -supported.bottom_z

      # 约束2: 侧面接触
      遍历 x/y 轴 (axis_idx=0,1) 和 正/负方向 (sign=1,-1):
          if sign > 0:  offset = supporter.min_x - supported.max_x
          else:         offset = supporter.max_x - supported.min_x
      选择 |offset| 最小的方向 (最近侧面)

      # 约束3: 穿模检查
      if 侧面移动后 AABB 在x/y上重叠 且 z轴穿入:
          z_fix = supporter.top_z - supported.bottom_z

      supporter ┃  supported
               ┃← offset →│
               ┃           │  ← 侧面接触

  ── hanging_below: 悬挂在支撑物下方 ──

    场景: 吊灯、吊扇

    物理约束:
      supported.top_z <= supporter.bottom_z  (不穿入支撑物)
      supported.top_z >= supporter.bottom_z  (不远离)

    实现:
      z_offset = supporter.bottom_z - supported.top_z

      z_offset > 0 → 物体在下方太远，上移到刚好接触
      z_offset < 0 → 物体穿入支撑物，下移到刚好接触
      z_offset ≈ 0 → 已接触，无需调整

      supporter底面z ───────┐
                           │ z_offset
      supported顶面z ───────┘ → 对齐到底面

  ── leaning: 斜靠在支撑物上 ──

    场景: 画靠墙、梯子靠墙

    物理约束: 同 against_side (底面贴地 + 侧面接触 + 穿模检查)

    实现: 直接调用 sp_refine_against_side

================================================================================
  使用方式
================================================================================

  Python API:

    from tools.refine_inter_object_placement import refine_inter_object_relations

    all_instances = refine_inter_object_relations(
        all_instances=all_instances,
        refined_relations=refined_relations,
        walls_info=walls_info,
        vlm_checkpoint="/mnt/data/lza/models/Qwen3.5-9B",
        scene_dir="outputs/232",
    )

  命令行:

    python3 -m tools.refine_inter_object_placement \
        --input_glb outputs/232/final_scene_basic.glb \
        --relations_json assets/json_configs/232_refined.json \
        --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B

    输出: outputs/232/final_scene_refined.glb
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import trimesh
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vlm_logger import vlm_inference, set_vlm_log_dir


def _load_vlm_model(checkpoint):
    """加载 VLM 模型 (策略判定/精修程度判断用). 懒加载复用 Stage1 的加载逻辑."""
    from tools.generate_scene_json_stage1 import load_vlm_model
    return load_vlm_model(checkpoint)


PLACEMENT_STRATEGIES = {
    "on_top": "放在支撑物顶面",
    "inside": "放在支撑物内部",
    "against_side": "靠在支撑物侧面",
    "hanging_below": "悬挂在支撑物下方",
    "leaning": "斜靠在支撑物上",
}

VALID_STRATEGIES = set(PLACEMENT_STRATEGIES.keys())
# on_top 几何对齐阈值(m): 物体底面与支撑物顶面的 z 偏差小于此值才算
# "几何上已贴合", 此时才咨询 VLM 是否已正确 (VLM 看的是照片, 不能推翻
# 明确的几何事实 —— HOI4D 082_C5: 悬空 0.971m 却被 VLM 判为 correct,
# 导致 21 个物体全部没被放到桌面上)。偏差大于此值时直接执行对齐。
GEOM_ALIGNED_TOL = 0.15

SP_REFINE_MAP: Dict[str, callable] = {}

DEFAULT_VLM_CHECKPOINT = "/mnt/data/lza/models/Qwen3.5-9B"


def _resolve_vlm_checkpoint(checkpoint_arg):
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    if os.path.exists(DEFAULT_VLM_CHECKPOINT):
        return DEFAULT_VLM_CHECKPOINT
    return None


# VLM 模型加载和推理函数已从 src/vlm_logger 导入，不再在此处定义


def _extract_json_from_text(text):
    """从VLM输出文本中提取JSON"""
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
            text = after[close_pos + 1:].strip()

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
                    candidate = text[first_brace:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
        last_brace = text.rfind('}')
        if last_brace > first_brace:
            return text[first_brace:last_brace + 1]
    return None


def build_placement_prompt(supported_name, supporter_name, all_pairs):
    """构建VLM放置策略判定提示词

    参数:
        supported_name: 被支撑物体名称
        supporter_name: 支撑物名称
        all_pairs: 所有支撑关系对 [(supported, supporter), ...]
    返回:
        prompt文本
    """
    pairs_str = "\n".join([f'  - "{s}" is supported by "{t}"' for s, t in all_pairs])

    prompt = f"""Look at the image. Determine how "{supported_name}" is physically placed on/against "{supporter_name}".

**All support relationships in the scene**:
{pairs_str}

**Question**: How is "{supported_name}" placed relative to "{supporter_name}"?

**Choose ONE placement strategy**:
- "on_top" — {supported_name} rests ON TOP of {supporter_name} (e.g., cup on table, pillow on bed, book on shelf, toy on table)
- "inside" — {supported_name} is INSIDE {supporter_name} (e.g., clothes in drawer, dish in cabinet, shoe in shoe rack)
- "against_side" — {supported_name} is against the SIDE of {supporter_name} (e.g., cabinet against wall, picture on wall)
- "hanging_below" — {supported_name} hangs BELOW {supporter_name} (e.g., lamp below ceiling, chandelier)
- "leaning" — {supported_name} leans against {supporter_name} at an angle (e.g., painting leaning on wall, ladder against wall)

**Physical common sense**:
- Most small objects on furniture → "on_top"
- Objects stored in containers/drawers → "inside"
- Objects mounted flush on walls → "against_side"
- Suspended from ceiling/overhang → "hanging_below"
- Tilted/angled contact → "leaning"

JSON only, no explanation:
{{"strategy": "one_of_on_top_inside_against_side_hanging_below_leaning"}}"""

    return prompt


def _parse_optimal_frame_filename(filename):
    """解析optimal_frames文件名: {category}_inst{idx}_frame{fid}.jpg → (category, idx, fid)"""
    name = os.path.splitext(filename)[0]
    match = re.match(r'(.+)_inst(\d+)_frame(\d+)', name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None


def _fuzzy_match_name(query, candidates):
    """模糊匹配名称到候选列表"""
    if not query or not candidates:
        return None
    q = query.lower().strip().replace(' ', '_')
    for c in candidates:
        if c.lower().strip().replace(' ', '_') == q:
            return c
    for c in candidates:
        c_base = c.lower().split('_')[0].split(' ')[0]
        q_base = q.split('_')[0].split(' ')[0]
        if c_base == q_base:
            return c
    for c in candidates:
        if q in c.lower() or c.lower() in q:
            return c
    return None


def _load_optimal_frames_for_instances(optimal_frames_dir, target_names):
    """加载optimal_frames/，按实例索引区分帧映射

    参数:
        optimal_frames_dir: optimal_frames目录路径
        target_names: 目标物体名称列表（如 ["bowl", "table", "toy"]）
    返回:
        instance_frames: {instance_key: [(PIL.Image, source_str), ...]}
            instance_key 如 "bowl_0", "toy_1"
        instance_count: {category: int} 每个类别的实例数量
    """
    from PIL import Image as PILImage

    if not os.path.isdir(optimal_frames_dir):
        return {}, {}

    instance_frames = defaultdict(list)
    instance_count = defaultdict(int)
    seen = defaultdict(set)

    for fname in sorted(os.listdir(optimal_frames_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        obj_name, inst_idx, frame_id = _parse_optimal_frame_filename(fname)
        if obj_name is None:
            continue

        matched = _fuzzy_match_name(obj_name, target_names)
        cat = matched if matched else obj_name

        inst_key = f"{cat}_{inst_idx}"
        src = f"optimal_frames/{fname}"
        if src in seen[inst_key]:
            continue

        img_path = os.path.join(optimal_frames_dir, fname)
        try:
            img = PILImage.open(img_path).convert("RGB")
            instance_frames[inst_key].append((img, src))
            seen[inst_key].add(src)
            instance_count[cat] = max(instance_count[cat], inst_idx + 1)
        except Exception as _e:
            # 原为静默 continue: 读不到最优视角图会导致该物体没有投票帧,
            # 策略恒回退 "on_top" 而日志无任何提示。
            print(f"      ⚠️ [load_optimal_frame] 读取失败 {img_path}: "
                  f"{type(_e).__name__}: {_e}", flush=True)
            continue

    return dict(instance_frames), dict(instance_count)


def _load_keyframes_for_instances(keyframes_dir, target_names):
    """加载keyframes/及可见性元数据

    参数:
        keyframes_dir: keyframes目录路径（含keyframes_metadata.json）
        target_names: 目标物体名称列表
    返回:
        keyframe_images: {vid_idx: (PIL.Image, source_str)}
        frame_visibility: {vid_idx: [matched_name, ...]}
    """
    from PIL import Image as PILImage

    metadata_path = os.path.join(keyframes_dir, "keyframes_metadata.json")
    if not os.path.exists(metadata_path):
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
                img = PILImage.open(full_path).convert("RGB")
                keyframe_images[vid_idx] = (img, f"keyframes/{path}")
            except Exception:
                continue

    frame_visibility = {}
    for fidx_str, visible_names in raw_visibility.items():
        vid_idx = int(fidx_str)
        matched = []
        for raw_name in visible_names:
            json_name = _fuzzy_match_name(raw_name, target_names)
            matched.append(json_name if json_name else raw_name)
        frame_visibility[vid_idx] = matched

    return keyframe_images, frame_visibility


def _build_instance_frame_map(optimal_instance_frames, instance_count,
                               keyframe_images, keyframe_visibility,
                               target_categories):
    """合并 optimal_frames + keyframes 为每个实例的帧列表（去重）

    与 refine_other_objects_relations.py 的 build_object_to_frames 逻辑一致。

    参数:
        optimal_instance_frames: {inst_key: [(PIL.Image, src), ...]}
        instance_count: {category: int}
        keyframe_images: {vid_idx: (PIL.Image, src)}
        keyframe_visibility: {vid_idx: [name, ...]}
        target_categories: 目标类别列表 (可包含 "bottle" 或 "bottle_0" 格式)
    返回:
        instance_to_frames: {inst_key: [(PIL.Image, src), ...]}
        instance_to_category: {inst_key: category}
    """
    import re as _re
    instance_to_frames = defaultdict(list)
    instance_to_category = {}
    seen_sources = defaultdict(set)

    # 过滤 target_categories: 去掉实例级 key (如 "bottle_0"),
    # 只保留类别级 key (如 "bottle"), 避免生成 "bottle_0_0" 这种 ghost key
    base_categories = []
    for cat in target_categories:
        if _re.match(r'^[a-zA-Z_]+_\d+$', cat):
            # "bottle_0" -> "bottle"
            base = _re.sub(r'_\d+$', '', cat)
            if base not in base_categories:
                base_categories.append(base)
        else:
            if cat not in base_categories:
                base_categories.append(cat)

    for cat in base_categories:
        n_inst = instance_count.get(cat, 1)
        if n_inst == 0:
            n_inst = 1

        for inst_idx in range(n_inst):
            inst_key = f"{cat}_{inst_idx}"
            instance_to_category[inst_key] = cat

            if inst_key in optimal_instance_frames:
                for img, src in optimal_instance_frames[inst_key]:
                    if src not in seen_sources[inst_key]:
                        instance_to_frames[inst_key].append((img, src))
                        seen_sources[inst_key].add(src)

    for vid_idx, visible_names in keyframe_visibility.items():
        if vid_idx not in keyframe_images:
            continue
        img, src = keyframe_images[vid_idx]
        for cat in base_categories:
            if cat in visible_names:
                n_inst = instance_count.get(cat, 1)
                if n_inst == 0:
                    n_inst = 1
                for inst_idx in range(n_inst):
                    inst_key = f"{cat}_{inst_idx}"
                    if src not in seen_sources[inst_key]:
                        instance_to_frames[inst_key].append((img, src))
                        seen_sources[inst_key].add(src)

    return dict(instance_to_frames), instance_to_category


def vlm_lookup_placement_strategy(supported_name, supporter_name, all_pairs,
                                  model, processor, instance_frames):
    """用VLM判断放置策略，按实例帧投票

    参数:
        supported_name: 被支撑物体名称（如 "bowl_0"）
        supporter_name: 支撑物名称（如 "table"）
        all_pairs: 所有支撑关系对
        model: VLM模型
        processor: VLM处理器
        instance_frames: [(PIL.Image, source_str), ...] 该实例的帧列表
    返回:
        策略字符串 (on_top/inside/against_side/hanging_below/leaning)
    """
    prompt = build_placement_prompt(supported_name, supporter_name, all_pairs)

    # 投票帧数上限: 帧列表按 optimal 优先排序, 取前 5 足够统计意义
    # (HOI4D 23 对支撑 × ~11 帧曾导致 5.2 放置策略 ~25min)
    _MAX_VOTE_FRAMES = 5
    votes = Counter()
    # 静默失败计数: 曾因 NameError(_vlm_inference 未定义) 导致投票全失败却
    # 无任何日志, fallback 恒返回 "on_top", 缺陷隐藏数月。这里显式记录失败数,
    # 让"策略恒为 on_top"能被日志识别, 而不是靠再次踩坑才发现。
    n_vote_total = 0
    n_vote_failed = 0
    first_error = None
    for img, src in instance_frames[:_MAX_VOTE_FRAMES]:
        n_vote_total += 1
        try:
            raw_text = vlm_inference(img, model, processor, prompt, max_new_tokens=256,
                                caller="stage5_placement_strategy")
            json_str = _extract_json_from_text(raw_text)
            if json_str:
                data = json.loads(json_str)
                strategy = data.get("strategy", "").strip().lower()
                if strategy in VALID_STRATEGIES:
                    votes[strategy] += 1
            else:
                n_vote_failed += 1
                first_error = first_error or "VLM 未返回可解析的 JSON"
        except Exception as e:
            n_vote_failed += 1
            first_error = first_error or f"{type(e).__name__}: {e}"

    if n_vote_failed:
        print(f"      ⚠️ [strategy_vote] {supported_name}: "
              f"{n_vote_failed}/{n_vote_total} 帧投票失败 ({first_error})"
              f" → 该物体策略将回退默认", flush=True)

    if votes:
        best_strategy = votes.most_common(1)[0][0]
        return best_strategy

    return "on_top"


def lookup_placement_strategy(supported_name, supporter_name):
    """无VLM时的规则回退，默认 on_top"""
    return "on_top"


def _get_transformed_mesh(instance_info, transform_matrix=None):
    """获取变换后的mesh"""
    mesh = instance_info["original_mesh"].copy()
    mesh.apply_transform(transform_matrix if transform_matrix is not None else instance_info["T"])
    return mesh


def _get_transformed_bounds(instance_info, transform_matrix=None):
    mesh = _get_transformed_mesh(instance_info, transform_matrix)
    return mesh.bounds


def _get_transformed_center(instance_info, transform_matrix=None):
    if transform_matrix is None:
        transform_matrix = instance_info["T"]
    center_local = np.mean(instance_info["original_mesh"].vertices, axis=0)
    center_world = trimesh.transformations.transform_points(
        np.array([center_local]), transform_matrix
    )[0]
    return center_world


def _align_upright(info):
    """旋转对齐: 让物体上方向对齐z轴

    当前已禁用旋转对齐，只做z轴平移。保留函数接口以便后续启用。

    禁用原因: VGGT重建的物体朝向可能本身就不准确，强制旋转可能
    导致更差的结果。当前优先保证z轴位置正确(底面贴支撑面)，
    旋转对齐待后续验证效果后再启用。

    Args:
        info: 物体信息字典，含 original_mesh 和 T

    Returns:
        dict: 不做任何修改，原样返回
    """
    return info


def _judge_refinement_degree_with_vlm(vlm_model, vlm_processor, image,
                                       object_name, supporter_name,
                                       max_new_tokens=64):
    """用VLM判断物体放置的精修程度.

    根据图像中物体与支撑物的关系, 判断需要大精修还是小精修:
    - major: 物体明显穿入支撑物内部或位置严重偏离
    - minor: 物体接近正确位置, 只需微调
    - correct: 物体已正确放置, 无需调整

    Args:
        vlm_model: 已加载的VLM模型
        vlm_processor: VLM processor
        image: PIL.Image 或 numpy array (H, W, 3) RGB
        object_name: 被支撑物体名称
        supporter_name: 支撑物名称
        max_new_tokens: 最大输出token数

    Returns:
        str: "major" / "minor" / "correct"
    """
    try:
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)

        prompt = (
            f"Look at the {object_name} and {supporter_name} in this image.\n"
            f"Is the {object_name} correctly resting on top of the {supporter_name}?\n\n"
            f"Answer with ONE word:\n"
            f"- major: the {object_name} is deeply embedded inside the {supporter_name}, "
            f"or far from the correct position\n"
            f"- minor: the {object_name} is close to correct but slightly off "
            f"(floating or slightly embedded)\n"
            f"- correct: the {object_name} is correctly resting on the {supporter_name}\n\n"
            f"Answer:"
        )

        result = vlm_inference(image, vlm_model, vlm_processor, prompt,
                                max_new_tokens=max_new_tokens, caller="stage5_refinement_degree")
        result = result.strip().lower()

        if "major" in result:
            return "major"
        elif "minor" in result:
            return "minor"
        elif "correct" in result:
            return "correct"
        else:
            # 无法解析, 默认 minor
            return "minor"
    except Exception as e:
        print(f"      ⚠️ VLM精修程度判断失败: {e}, 默认 minor", flush=True)
        return "minor"


def _compute_penetration_depth(supported_info, supporter_info):
    """计算物体与支撑物的穿模深度 (z轴方向).

    Returns:
        float: 穿模深度 (m), 正值表示穿入, 0表示无穿模
    """
    supported_mesh = _get_transformed_mesh(supported_info)
    supporter_mesh = _get_transformed_mesh(supporter_info)

    supported_bottom_z = supported_mesh.bounds[0, 2]
    supporter_top_z = supporter_mesh.bounds[1, 2]

    # 穿模深度: 物体底部在支撑物顶部以下多少
    penetration = supporter_top_z - supported_bottom_z
    return max(0.0, penetration)


def sp_refine_on_top(supported_info, supporter_info, max_offset=0.3, initial_offset=None,
                     refinement_degree=None):
    """supported底面贴supporter顶面 (只做z轴对齐)

    大精修 / 小精修 分级策略:
      - refinement_degree="major": 大精修 — 物体严重穿入支撑物
        始终执行完整 z 轴贴合 (移动到支撑物顶面)
      - refinement_degree="minor": 小精修 — 物体接近正确位置
        执行 z 轴贴合, 但限制最大移动距离为 0.15m (避免过度调整)
      - refinement_degree="correct": 不调整
      - refinement_degree=None: 默认行为, 始终执行完整 z 轴贴合

    z_offset = supporter_top_z - supported_bottom_z
    正值: 物体在支撑物下方或穿入，向上推
    负值: 物体悬空在支撑物上方，向下落
    接近0: 已接触，无需调整

    Args:
        supported_info: 被支撑物体的信息字典，含 original_mesh 和 T
        supporter_info: 支撑物体的信息字典，含 original_mesh 和 T
        max_offset: (已废弃, 保留参数兼容)
        initial_offset: (已废弃, 保留参数兼容)
        refinement_degree: VLM判断的精修程度 "major"/"minor"/"correct"/None

    Returns:
        dict: T矩阵已更新的 supported_info
    """
    # correct: 不调整
    if refinement_degree == "correct":
        print(f"      [on_top] VLM判定 correct, 无需调整", flush=True)
        return supported_info

    old_pos = supported_info["T"][:3, 3].copy()

    supported_info = _align_upright(supported_info)

    transform_matrix = supported_info["T"].copy()

    transformed_mesh = supported_info["original_mesh"].copy()
    transformed_mesh.apply_transform(transform_matrix)
    supported_bottom_z = transformed_mesh.bounds[0, 2]

    supporter_mesh = _get_transformed_mesh(supporter_info)
    
    # 从瓶子底部投影到支撑物：取瓶子底部顶点，投影到支撑物表面，取支撑面 z 的平均
    supporter_vertices = supporter_mesh.vertices
    supported_center_xy = transformed_mesh.bounds.mean(axis=0)[:2]
    supported_half_size = (transformed_mesh.bounds[1, :2] - transformed_mesh.bounds[0, :2]) / 2
    
    # 取瓶子底部 z 值接近 minimum 的顶点（整个底面，不是单个点）
    bottom_z_threshold = 0.01
    bottom_verts_mask = np.abs(transformed_mesh.vertices[:, 2] - supported_bottom_z) < bottom_z_threshold
    bottom_vertices = transformed_mesh.vertices[bottom_verts_mask]
    
    if len(bottom_vertices) > 0:
        # 对每个底部顶点, 在支撑物上找 (x,y) 邻域内的"局部顶面"高度。
        # 注意必须取窗口内顶点的 z 最大值 (局部顶面), 而不是"低于物体底面的
        # 顶点 z 均值" —— 均值会把桌腿/底面/隔板的顶点也算进来, 估计值被拉到
        # 桌面之下: 悬空物体勉强能落, 已沉入桌面的物体会被越按越深
        # (2chair 实测: 物体在桌面时该均值 ≈0.3 vs 桌面 0.49)。
        local_top_zs = []
        for bv in bottom_vertices:
            bx, by = bv[0], bv[1]
            in_x = np.abs(supporter_vertices[:, 0] - bx) < supported_half_size[0]
            in_y = np.abs(supporter_vertices[:, 1] - by) < supported_half_size[1]
            mask = in_x & in_y
            matching_pts = supporter_vertices[mask]
            if len(matching_pts) > 0:
                local_top_zs.append(float(np.max(matching_pts[:, 2])))

        if local_top_zs:
            # 各底部顶点的局部顶面取中位数 (对个别悬空窗口离群鲁棒)
            supporter_top_z = float(np.median(local_top_zs))
        else:
            # fallback: 用 bounding box 顶面
            supporter_top_z = supporter_mesh.bounds[1, 2]
    else:
        # fallback: 用 bounding box 顶面
        supporter_top_z = supporter_mesh.bounds[1, 2]

    z_offset = supporter_top_z - supported_bottom_z

    if abs(z_offset) < 1e-6:
        return supported_info

    # 根据精修程度决定调整策略
    if refinement_degree == "minor":
        reason = "小精修"
        max_minor_move = 0.15
        if abs(z_offset) > max_minor_move:
            z_offset = max_minor_move if z_offset > 0 else -max_minor_move
            print(f"      [on_top] 小精修: z_offset截断到 {z_offset:+.4f}m", flush=True)
        else:
            print(f"      [on_top] 小精修: z_offset={z_offset:+.4f}m", flush=True)
    elif refinement_degree == "major":
        reason = "大精修"
        print(f"      [on_top] 大精修: z_offset={z_offset:+.4f}m, 完整贴合", flush=True)
    else:
        reason = "默认精修"
        print(f"      [on_top] z_offset={z_offset:+.4f}m, 贴合支撑物顶面", flush=True)

    # 应用 z 轴平移
    translation_vector = np.array([0, 0, z_offset])
    translation_matrix = trimesh.transformations.translation_matrix(translation_vector)
    transform_matrix = translation_matrix @ transform_matrix

    supported_info["T"] = transform_matrix

    # 后检查: 验证贴合是否成功
    post_mesh = supported_info["original_mesh"].copy()
    post_mesh.apply_transform(supported_info["T"])
    post_bottom_z = post_mesh.bounds[0, 2]
    residual = post_bottom_z - supporter_top_z
    if abs(residual) > 1e-4:
        correction = supporter_top_z - post_bottom_z
        supported_info["T"][2, 3] += correction
        print(f"      ⚠️ [on_top] 残余间隙={residual:.4f}m, 修正{correction:+.4f}m", flush=True)

    new_pos = supported_info["T"][:3, 3].copy()
    print(f"      [on_top] {reason}, supporter_top={supporter_top_z:.4f}m, "
          f"pos: ({old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}) → ({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})", flush=True)
    return supported_info


def sp_refine_inside(supported_info, supporter_info):
    """supported放入supporter内部，防止穿模

    物理约束:
      supported.bottom_z >= supporter.bottom_z (不穿出底部)
      supported.bottom_z <= supporter内部30%高度 (放在合理位置)
      supported.top_z <= supporter.top_z (不穿出顶部)

    实现:
      Step 1: 旋转对齐 — 让物体竖直
      Step 2: z轴平移 — supported底面对齐到supporter内部30%高度
      Step 3: 穿模检查 — 穿出顶部/底部则修正
    """
    old_pos = supported_info["T"][:3, 3].copy()
    supported_info = _align_upright(supported_info)

    transform_matrix = supported_info["T"].copy()

    supporter_bounds = _get_transformed_bounds(supporter_info)
    supported_bounds = _get_transformed_bounds(supported_info)

    supporter_bottom_z = supporter_bounds[0, 2]
    supporter_top_z = supporter_bounds[1, 2]
    supporter_height = supporter_top_z - supporter_bottom_z

    target_z = supporter_bottom_z + supporter_height * 0.3
    z_offset = target_z - supported_bounds[0, 2]

    translation_vec = np.array([0.0, 0.0, z_offset])
    transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    refined_bounds = _get_transformed_bounds(supported_info, transform_matrix)

    if refined_bounds[1, 2] > supporter_top_z:
        z_fix = supporter_top_z - refined_bounds[1, 2]
        transform_matrix = trimesh.transformations.translation_matrix(
            np.array([0.0, 0.0, z_fix])
        ) @ transform_matrix

    refined_bounds = _get_transformed_bounds(supported_info, transform_matrix)
    if refined_bounds[0, 2] < supporter_bottom_z:
        z_fix = supporter_bottom_z - refined_bounds[0, 2]
        transform_matrix = trimesh.transformations.translation_matrix(
            np.array([0.0, 0.0, z_fix])
        ) @ transform_matrix

    supported_info["T"] = transform_matrix
    new_pos = supported_info["T"][:3, 3].copy()
    print(f"      [inside] pos: ({old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}) → ({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})", flush=True)
    return supported_info


def sp_refine_against_side(supported_info, supporter_info, walls_info=None):
    """supported靠在supporter侧面：底面贴地 + 侧面接触，防止穿模

    物理约束:
      1. 底面贴地: supported.bottom_z >= 0
      2. 侧面接触: supported与supporter在x/y轴上刚好接触，不穿入也不远离
      3. 不穿模: 侧面接触后，检查z轴是否穿入supporter内部，如果是则上移

    实现:
      Step 1: 旋转对齐 — 让物体竖直
      Step 2: z轴: 底面贴地
      Step 3: x/y轴: 找最近侧面方向，移动到刚好接触
      Step 4: 穿模检查
    """
    old_pos = supported_info["T"][:3, 3].copy()
    supported_info = _align_upright(supported_info)

    transform_matrix = supported_info["T"].copy()

    supported_bounds = _get_transformed_bounds(supported_info)
    if supported_bounds[0, 2] < 0.0:
        z_offset = -supported_bounds[0, 2]
        transform_matrix = trimesh.transformations.translation_matrix(
            np.array([0.0, 0.0, z_offset])
        ) @ transform_matrix

    supported_mesh = _get_transformed_mesh(supported_info, transform_matrix)
    supported_vertices = supported_mesh.vertices

    supporter_mesh = _get_transformed_mesh(supporter_info)
    supporter_vertices = supporter_mesh.vertices

    best_offset = 0.0
    best_dist = float("inf")
    best_axis = 0

    for axis_idx in [0, 1]:
        for sign in [1.0, -1.0]:
            if sign > 0:
                s_max = supported_vertices[:, axis_idx].max()
                r_min = supporter_vertices[:, axis_idx].min()
                offset = r_min - s_max
            else:
                s_min = supported_vertices[:, axis_idx].min()
                r_max = supporter_vertices[:, axis_idx].max()
                offset = r_max - s_min

            if abs(offset) < abs(best_dist):
                best_dist = offset
                best_offset = offset
                best_axis = axis_idx

    if abs(best_offset) < 1.0:
        translation_vec = np.array([0.0, 0.0, 0.0])
        translation_vec[best_axis] = best_offset
        transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    refined_mesh = _get_transformed_mesh(supported_info, transform_matrix)
    refined_bounds = refined_mesh.bounds
    supporter_bounds = supporter_mesh.bounds

    if (refined_bounds[0, 0] < supporter_bounds[1, 0] and
        refined_bounds[1, 0] > supporter_bounds[0, 0] and
        refined_bounds[0, 1] < supporter_bounds[1, 1] and
        refined_bounds[1, 1] > supporter_bounds[0, 1] and
        refined_bounds[0, 2] < supporter_bounds[1, 2]):

        z_fix = supporter_bounds[1, 2] - refined_bounds[0, 2]
        if z_fix > 0 and z_fix < supporter_bounds[1, 2] - supporter_bounds[0, 2]:
            transform_matrix = trimesh.transformations.translation_matrix(
                np.array([0.0, 0.0, z_fix])
            ) @ transform_matrix

    supported_info["T"] = transform_matrix
    new_pos = supported_info["T"][:3, 3].copy()
    print(f"      [against_side] pos: ({old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}) → ({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})", flush=True)
    return supported_info


def sp_refine_hanging_below(supported_info, supporter_info):
    """supported悬挂在supporter下方：顶面贴支撑物底面，防止穿模

    物理约束:
      supported.top_z <= supporter.bottom_z (不穿入支撑物)
      supported.top_z >= supporter.bottom_z - eps (不远离)

    实现:
      Step 1: 旋转对齐 — 让物体竖直
      Step 2: z轴平移 — supported顶面对齐到supporter底面
    """
    old_pos = supported_info["T"][:3, 3].copy()
    supported_info = _align_upright(supported_info)

    transform_matrix = supported_info["T"].copy()

    supporter_bounds = _get_transformed_bounds(supporter_info)
    supported_bounds = _get_transformed_bounds(supported_info)

    supporter_bottom_z = supporter_bounds[0, 2]
    supported_top_z = supported_bounds[1, 2]

    z_offset = supporter_bottom_z - supported_top_z

    if abs(z_offset) < 1e-6:
        return supported_info

    translation_vec = np.array([0.0, 0.0, z_offset])
    transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    supported_info["T"] = transform_matrix
    new_pos = supported_info["T"][:3, 3].copy()
    print(f"      [hanging_below] z_offset={z_offset:.4f}m, pos: ({old_pos[0]:.3f},{old_pos[1]:.3f},{old_pos[2]:.3f}) → ({new_pos[0]:.3f},{new_pos[1]:.3f},{new_pos[2]:.3f})", flush=True)
    return supported_info


def sp_refine_leaning(supported_info, supporter_info):
    """斜靠在支撑物上：等同于 against_side"""
    return sp_refine_against_side(supported_info, supporter_info)


SP_REFINE_MAP = {
    "on_top": sp_refine_on_top,
    "inside": sp_refine_inside,
    "against_side": sp_refine_against_side,
    "hanging_below": sp_refine_hanging_below,
    "leaning": sp_refine_leaning,
}


def _find_supporter_instances(supporter_name, all_instances):
    """在all_instances中按名称查找实例列表，返回 (category, instances) 或 (None, None)

    支持实例级匹配:
      - "toy_2" → toy类别第2个实例(索引2)
      - "toy" → toy类别全部实例
      - "table" → table类别全部实例
    """
    supporter_lower = supporter_name.lower().strip().replace(" ", "_")

    instance_idx = None
    cat_name = supporter_lower
    if "_" in supporter_lower:
        parts = supporter_lower.rsplit("_", 1)
        if parts[1].isdigit():
            cat_name = parts[0]
            instance_idx = int(parts[1])

    for category, instances in all_instances.items():
        cat_lower = category.lower().strip().replace(" ", "_")
        if cat_lower == cat_name:
            if instance_idx is not None and instance_idx < len(instances):
                return category, [instances[instance_idx]]
            return category, instances

    for category, instances in all_instances.items():
        cat_base = category.lower().split("_")[0].split(" ")[0]
        supp_base = cat_name.split("_")[0].split(" ")[0]
        if cat_base == supp_base:
            if instance_idx is not None and instance_idx < len(instances):
                return category, [instances[instance_idx]]
            return category, instances

    for category, instances in all_instances.items():
        if cat_name in category.lower() or category.lower() in cat_name:
            if instance_idx is not None and instance_idx < len(instances):
                return category, [instances[instance_idx]]
            return category, instances

    return None, None


def _find_nearest_supporter_instance(supported_info, supporter_instances):
    """在多个支撑物实例中找到最近的那个"""
    if len(supporter_instances) == 1:
        return supporter_instances[0]

    supported_center = _get_transformed_center(supported_info)
    best_instance = supporter_instances[0]
    best_dist = float("inf")

    for inst in supporter_instances:
        inst_center = _get_transformed_center(inst)
        dist = np.linalg.norm(supported_center - inst_center)
        if dist < best_dist:
            best_dist = dist
            best_instance = inst

    return best_instance


def _find_relationship_for_category(category, categories_and_relations):
    """模糊匹配找到某个 category 对应的 relationship（支持 instance-level key）"""
    cat_lower = category.lower().strip()
    for key, rel in categories_and_relations.items():
        key_lower = key.lower().strip()
        if key_lower == cat_lower:
            return rel
        if key_lower.startswith(cat_lower) and (len(key_lower) == len(cat_lower) or key_lower[len(cat_lower)] == '_'):
            return rel
    return "supported by floor"


def _topological_sort_pairs(supported_pairs):
    """依赖排序: 支撑物先处理，被支撑物后处理

    例如: toy→box→table 的处理顺序应为 table, box, toy
    这样 box 放到 table 上之后，toy 再放到 box 上时位置才正确。
    """
    supporter_set = set()
    supported_set = set()
    for supported, supporter in supported_pairs:
        supporter_set.add(supporter)
        supported_set.add(supported)

    roots = supporter_set - supported_set

    order = []
    visited = set()
    queue = list(roots)

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for supported, supporter in supported_pairs:
            if supporter == current and supported not in visited:
                order.append((supported, supporter))
                queue.append(supported)

    for supported, supporter in supported_pairs:
        if (supported, supporter) not in order:
            order.append((supported, supporter))

    return order


def _get_aabb_overlap(mesh_a, mesh_b):
    """计算两个mesh的AABB重叠量

    返回: (overlaps, overlap_x, overlap_y, overlap_z)
        overlaps: 是否有重叠
        overlap_x/y/z: 各轴的重叠深度（负值表示不重叠）
    """
    bounds_a = mesh_a.bounds
    bounds_b = mesh_b.bounds

    overlap_x = min(bounds_a[1, 0], bounds_b[1, 0]) - max(bounds_a[0, 0], bounds_b[0, 0])
    overlap_y = min(bounds_a[1, 1], bounds_b[1, 1]) - max(bounds_a[0, 1], bounds_b[0, 1])
    overlap_z = min(bounds_a[1, 2], bounds_b[1, 2]) - max(bounds_a[0, 2], bounds_b[0, 2])

    overlaps = overlap_x > 0 and overlap_y > 0 and overlap_z > 0
    return overlaps, overlap_x, overlap_y, overlap_z


# 穿模检测参数 (原先硬编码在函数体内, 无集中定义):
# _PEN_SURFACE_SAMPLES 表面采样点数——trimesh 4.x 的关键字是 count
# _PEN_STALL_LIMIT 连续多少轮"总分离量未下降"即判定不收敛并停止。
#   单靠 max_iterations 兜底会在发散时空转 (实测 127 次仍未收敛)
_PEN_STALL_LIMIT = 3
_PEN_SURFACE_SAMPLES = 2000
# _PEN_SURFACE_TOL_FALLBACK 采样点到对方 mesh 表面的距离阈值 (m):
#   与改动前一致 (0.01)。曾尝试收紧到 0.002 + 改用 contains(), 但该改动基于
#   "穿模全属误报"的错误前提 (已用 FCL 纠正: egodex_4 有 4 对真穿模,
#   cloth_0↔cloth_1 有 7578 个 FCL 接触点), 故已回退。
_PEN_SURFACE_TOL_FALLBACK = 0.01
# _PEN_MIN_INSIDE 判定穿模所需的最少采样点数, 与改动前一致
_PEN_MIN_INSIDE = 3


def _check_mesh_penetration_fcl(mesh_a, mesh_b):
    """使用FCL精确碰撞检测判断mesh_a是否穿入mesh_b

    相比旧版AABB+顶点采样方法，FCL提供:
      - 精确的mesh-mesh碰撞检测 (非AABB近似)
      - 精确的穿透深度和分离方向
      - 不受顶点密度影响

    Returns:
        (penetrates, penetration_depth, sep_axis):
            penetrates: bool, whether penetration exists
            penetration_depth: float, estimated penetration depth
            sep_axis: int (0=x, 1=y, 2=z), best separation axis
    """
    try:
        from trimesh.collision import CollisionManager
        manager = CollisionManager()
        manager.add_object("b", mesh_b)
        collides, contacts = manager.in_collision_single(mesh_a, return_data=True)
        if not collides or not contacts:
            return False, 0.0, 2

        max_depth = 0.0
        avg_normal = np.zeros(3)
        for c in contacts:
            if c.depth > max_depth:
                max_depth = c.depth
            avg_normal += np.array(c.normal)
        avg_normal /= len(contacts)
        norm = np.linalg.norm(avg_normal)
        if norm > 1e-8:
            avg_normal /= norm

        sep_axis = int(np.argmax(np.abs(avg_normal)))
        return True, float(max_depth), sep_axis
    except (ImportError, ValueError):
        return _check_mesh_penetration_legacy(mesh_a, mesh_b)


def _check_mesh_penetration_legacy(mesh_a, mesh_b, n_samples=500):
    """旧版AABB+顶点采样碰撞检测 (FCL不可用时的回退方案)

    Returns:
        (penetrates, penetration_depth, sep_axis):
            penetrates: bool, whether penetration exists
            penetration_depth: float, estimated penetration depth
            sep_axis: int (0=x, 1=y, 2=z), best separation axis
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        overlaps, ox, oy, oz = _get_aabb_overlap(mesh_a, mesh_b)
        if not overlaps:
            return False, 0.0, 2
        min_overlap = min(ox, oy, oz)
        if oz == min_overlap:
            return True, oz, 2
        elif oy == min_overlap:
            return True, oy, 1
        return True, ox, 0

    if len(mesh_a.vertices) < 10 or len(mesh_b.vertices) < 10:
        overlaps, ox, oy, oz = _get_aabb_overlap(mesh_a, mesh_b)
        if not overlaps:
            return False, 0.0, 2
        min_overlap = min(ox, oy, oz)
        if oz == min_overlap:
            return True, oz, 2
        elif oy == min_overlap:
            return True, oy, 1
        return True, ox, 0

    overlaps, ox, oy, oz = _get_aabb_overlap(mesh_a, mesh_b)
    if not overlaps:
        return False, 0.0, 2

    # 表面采样点。trimesh 4.x 的 sample_surface 签名是 (mesh, count, ...),
    # 关键字叫 count 而非 n_samples —— 之前误用 n_samples 会抛 TypeError 被
    # except 吞掉, 静默退化成"只用 mesh 顶点做最近距离判断"。
    try:
        pts_a, _ = trimesh.sample.sample_surface(mesh_a, _PEN_SURFACE_SAMPLES)
    except Exception:
        pts_a = mesh_a.vertices
    if len(pts_a) == 0:
        pts_a = mesh_a.vertices

    tree_b = cKDTree(mesh_b.vertices)
    dists, _ = tree_b.query(pts_a)

    inside_count = (dists < _PEN_SURFACE_TOL_FALLBACK).sum()
    if inside_count < _PEN_MIN_INSIDE:
        return False, 0.0, 2

    inside_pts = pts_a[dists < _PEN_SURFACE_TOL_FALLBACK]
    center_a = mesh_a.bounds.mean(axis=0)
    center_b = mesh_b.bounds.mean(axis=0)
    diff = center_a - center_b

    abs_diff = np.abs(diff)
    sep_axis = int(np.argmax(abs_diff))

    penetration_depth = float(np.mean(dists[dists < _PEN_SURFACE_TOL_FALLBACK])) + 0.005

    return True, penetration_depth, sep_axis


def resolve_penetrations(all_instances, refined_relations=None, verbose=True,
                         categories_and_relations=None, dry_run=False,
                         max_iterations=8, scene_dir=None,
                         shared_penetration=False):
    """全局穿模检测与解决

    策略:
      1. 对每对物体用FCL精确碰撞检测 (回退到AABB+顶点采样)
      2. 如果穿模，沿FCL返回的分离方向推开（优先推被支撑物，保留支撑物不动）
      3. 推开后确保不穿出地面(z=0)
      4. 迭代直到无穿模或达到最大迭代次数
      5. 大物体穿模: 增加分离余量 (按穿模深度比例增加)
      6. 层级修复: 小物体 (parent != floor) 只 z 轴移动, 跟随 supporter x,y

    参数:
        all_instances: {category: [instance_info, ...]}
        refined_relations: 关系字典，用于判断谁是被支撑物（优先移动）
        verbose: 是否打印详细信息
        dry_run: 如果为True, 只检测和警告, 不实际修改T矩阵
        max_iterations: 最大迭代次数 (默认8, 比旧版5更多)
        scene_dir: 场景目录路径. 若提供, 尝试加载 relations_scene_graph.json,
                   使用 scene_graph_objects 的 parent 层级确定大小物体 (比字符串解析更可靠)
        shared_penetration: bool, 是否启用共享穿模修复 (默认False)
                           True时，穿模的两个物体各自退一半距离（sep_dist/2），
                           离原始位置都更近，点云匹配度更高。
                           仅对两个可移动物体有效（排除floor/wall/被支撑小物体）。
                           如果一方不可移动，自动回退为单方面推开。
    返回:
        更新后的 all_instances
    """
    if verbose:
        mode_str = " (仅检测, 不修改)" if dry_run else ""
        print(f"\n🛡️ 全局穿模检测与解决{mode_str}...", flush=True)

    supported_names = set()
    floor_names = set()
    wall_names = set()
    _all_rels = {}
    if categories_and_relations:
        _all_rels.update(categories_and_relations)
    if refined_relations:
        _all_rels.update(refined_relations)
    for name, rel in _all_rels.items():
        rel_lower = rel.lower()
        if rel.startswith("supported by ") and "floor" not in rel_lower and "other objects" not in rel_lower:
            base = name.rsplit('_', 1)[0] if '_' in name else name
            supported_names.add(base.lower().strip())
            supported_names.add(name.lower().strip())
        # 注意: 只有当物体本身就是 floor/wall (rel == "floor"/"wall") 时才算结构元素.
        # "supported by floor" 的物体 (如 table, cabinet) 是放在地板上的家具,
        # 它们之间应该正常检测穿模, 不能跳过.
        if rel_lower == "floor" or name.lower() == "floor":
            base = name.rsplit('_', 1)[0] if '_' in name else name
            floor_names.add(base.lower().strip())
            floor_names.add(name.lower().strip())
        if rel_lower == "wall" or name.lower() == "wall":
            base = name.rsplit('_', 1)[0] if '_' in name else name
            wall_names.add(base.lower().strip())
            wall_names.add(name.lower().strip())

    # 构建 on_top 声明关系对 (实例级): A 的关系是 "supported by B" 且 B 不是
    # floor/wall/other_objects。这类 pair 的垂直位置由 5.2 的 on_top 精修负责。
    #
    # 必须从穿模检测中排除: on_top 把物体精确贴到支撑物顶面, 接触面间隙为 0,
    # 碰撞检测必然判成重叠 → 沿 z 推开 → 物体掉到地面 → 底面共面又判重叠 →
    # 形成自维持循环。实测 egodex_33: 23 次修复 100% 是 z 轴分离且 100% 涉及
    # on_top 物体, toy_1 被推 9 次 x 0.0056m = 0.050m; 全库 104 个 on_top 关系
    # 只有 3 个贴合。这些 pair 的 xy 实测无重叠 (egodex_33 plate_0 上 4 个物体
    # xy 重叠 0%), 排除检测不会漏掉真穿模。
    on_top_pairs = set()
    for name, rel in _all_rels.items():
        if not rel.startswith("supported by "):
            continue
        sup = rel[len("supported by "):].strip()
        if sup in ("floor", "wall", "other objects") or "_" not in sup:
            continue
        on_top_pairs.add(frozenset((name.lower().strip(), sup.lower().strip())))
        # 同类别的不同实例也算(类别级关系时 name 可能是 base)
        base = name.rsplit("_", 1)[0] if "_" in name else name
        on_top_pairs.add(frozenset((base.lower().strip(), sup.rsplit("_", 1)[0].lower().strip())))

    # 构建 supporter→supported 映射, 用于层级穿模修复
    # 当 supporter (大物体) 移动时, 其 supported (小物体) 跟随移动 x,y
    # 小物体定义: scene_graph_objects 中 parent != 1 (floor) 的物体
    supporter_to_supported = {}

    # 方式 A (优先): 从 relations_scene_graph.json 加载 parent 层级
    # 新格式: scene_graph_objects.parent 是实例名字符串 (如 "table_0") 或 "floor"
    sg_loaded = False
    if scene_dir:
        sg_path = os.path.join(scene_dir, "relations_scene_graph.json")
        if os.path.isfile(sg_path):
            try:
                with open(sg_path, 'r') as f:
                    sg_data = json.load(f)
                sg_objects = sg_data.get("scene_graph_objects", [])

                # 用 parent 层级构建 supporter_to_supported
                # parent="floor" → 大物体/supporter (跳过); parent="X_0" → 小物体 (supported)
                for obj in sg_objects:
                    parent = obj.get("parent", "floor")
                    if parent == "floor":
                        continue
                    supported_key = (obj.get("category", "unknown").lower().strip(), obj.get("inst_idx", 0))

                    # parent 是 "table_0" 格式，需要解析
                    if '_' in parent:
                        s_cat, s_idx_str = parent.rsplit('_', 1)
                        try:
                            s_idx = int(s_idx_str)
                        except ValueError:
                            if verbose:
                                print(f"   ⚠️ 解析 parent '{parent}' 失败", flush=True)
                            continue
                    else:
                        if verbose:
                            print(f"   ⚠️ parent '{parent}' 不是实例格式，跳过", flush=True)
                        continue

                    supporter_key = (s_cat.lower().strip(), s_idx)
                    supporter_to_supported.setdefault(supporter_key, []).append(supported_key)
                    # 同步更新 supported_names: 小物体 (parent != floor) 的类别加入 supported_names
                    supported_cat_name = supported_key[0]
                    supported_names.add(supported_cat_name)

                if supporter_to_supported and verbose:
                    print(f"   📊 使用 scene_graph_objects parent 层级: "
                          f"{len(supporter_to_supported)} 个 supporter → "
                          f"{sum(len(v) for v in supporter_to_supported.values())} 个 supported", flush=True)
                sg_loaded = True
            except Exception as e:
                if verbose:
                    print(f"   ⚠️ 加载 relations_scene_graph.json 失败: {e}, 回退到字符串解析", flush=True)

    # 方式 B (回退): 从 refined_relations 字符串解析 "supported by X"
    if not sg_loaded:
        for name, rel in _all_rels.items():
            if rel.startswith("supported by ") and "floor" not in rel.lower() and "other objects" not in rel.lower():
                supporter_name = rel[len("supported by "):].strip()
                if '_' in supporter_name:
                    s_cat, s_idx_str = supporter_name.rsplit('_', 1)
                    try:
                        s_idx = int(s_idx_str)
                    except ValueError:
                        continue
                else:
                    continue
                if '_' in name:
                    d_cat, d_idx_str = name.rsplit('_', 1)
                    try:
                        d_idx = int(d_idx_str)
                    except ValueError:
                        continue
                else:
                    continue
                key = (s_cat.lower().strip(), s_idx)
                val = (d_cat.lower().strip(), d_idx)
                supporter_to_supported.setdefault(key, []).append(val)

    all_meshes = []
    for category, instances in all_instances.items():
        for idx, info in enumerate(instances):
            mesh = _get_transformed_mesh(info)
            all_meshes.append((category, idx, mesh, info))


    penetration_warnings = []
    effective_max_iter = 1 if dry_run else max_iterations
    _prev_total_sep = None      # 上一轮的总分离量 (收敛判据用)
    _stall_rounds = 0           # 连续无进展轮数
    _iteration_total_sep = 0.0  # 本轮总分离量
    for iteration in range(effective_max_iter):
        any_resolved = False
        _iteration_total_sep = 0.0

        for i in range(len(all_meshes)):
            for j in range(i + 1, len(all_meshes)):
                cat_i, idx_i, mesh_i, info_i = all_meshes[i]
                cat_j, idx_j, mesh_j, info_j = all_meshes[j]

                overlaps, ox, oy, oz = _get_aabb_overlap(mesh_i, mesh_j)
                if not overlaps:
                    continue

                penetrates, pen_depth, sep_axis = _check_mesh_penetration_fcl(mesh_i, mesh_j)
                if not penetrates:
                    continue

                cat_i_is_supported = cat_i.lower().strip() in supported_names
                cat_j_is_supported = cat_j.lower().strip() in supported_names
                cat_i_is_floor = cat_i.lower().strip() in floor_names
                cat_j_is_floor = cat_j.lower().strip() in floor_names
                cat_i_is_wall = cat_i.lower().strip() in wall_names
                cat_j_is_wall = cat_j.lower().strip() in wall_names

                if (cat_i_is_floor or cat_i_is_wall) and (cat_j_is_floor or cat_j_is_wall):
                    continue

                # 跳过 5.1 声明的 on_top 关系对: 其垂直位置由 5.2 on_top 精修负责,
                # 零间隙接触会被误判为穿模并把物体推离支撑物 (见上方说明)。
                _pair_key = frozenset((f"{cat_i}_{idx_i}".lower(),
                                       f"{cat_j}_{idx_j}".lower()))
                if _pair_key in on_top_pairs:
                    continue

                pair_desc = f"{cat_i}_{idx_i} ↔ {cat_j}_{idx_j}"
                axis_name = ['x', 'y', 'z'][sep_axis]

                if dry_run:
                    penetration_warnings.append({
                        "pair": pair_desc,
                        "axis": axis_name,
                        "depth": round(float(pen_depth), 4),
                    })
                    if verbose:
                        print(f"      ⚠️ 穿模警告: {pair_desc} | {axis_name}轴 穿模深度 {pen_depth:.4f}m", flush=True)
                    continue

                # --- 层级穿模修复: floor → 大物体 → 小物体 ---
                if cat_i_is_supported and not cat_j_is_supported:
                    move_idx = i
                elif cat_j_is_supported and not cat_i_is_supported:
                    move_idx = j
                elif (cat_i_is_floor or cat_i_is_wall) and not (cat_j_is_floor or cat_j_is_wall):
                    move_idx = j
                elif (cat_j_is_floor or cat_j_is_wall) and not (cat_i_is_floor or cat_i_is_wall):
                    move_idx = i
                else:
                    center_i = mesh_i.bounds.mean(axis=0)
                    center_j = mesh_j.bounds.mean(axis=0)
                    if center_i[sep_axis] > center_j[sep_axis]:
                        move_idx = i
                    else:
                        move_idx = j

                move_cat, move_idx_orig, move_mesh, move_info = all_meshes[move_idx]
                other_mesh = mesh_j if move_idx == i else mesh_i

                move_is_supported = cat_i_is_supported if move_idx == i else cat_j_is_supported
                other_is_supported = cat_j_is_supported if move_idx == i else cat_i_is_supported
                both_small = move_is_supported and other_is_supported

                # 小物体 (supported) 不独立调整 x,y: 强制 z 轴移动
                if move_is_supported and sep_axis != 2:
                    sep_axis = 2
                    axis_name = 'z'

                move_center = move_mesh.bounds.mean(axis=0)[sep_axis]
                other_center = other_mesh.bounds.mean(axis=0)[sep_axis]

                if move_center >= other_center:
                    direction = 1.0
                else:
                    direction = -1.0

                # 分离余量: 小物体之间更小
                move_size = np.max(move_mesh.bounds[1] - move_mesh.bounds[0])
                other_size = np.max(other_mesh.bounds[1] - other_mesh.bounds[0])
                max_size = max(move_size, other_size)
                if both_small:
                    sep_dist = pen_depth + 0.005
                elif max_size > 0.5:
                    sep_dist = pen_depth + 0.10
                elif max_size > 0.3:
                    sep_dist = pen_depth + 0.05
                else:
                    sep_dist = pen_depth + 0.01

                # ── 发散修复: 分离量必须覆盖实际重叠, 而非仅 pen_depth ──
                # pen_depth 是"采样点到对方表面的最近距离 + 0.005", 量级仅厘米级;
                # 而 AABB 重叠可达 0.57m (实测 egodex_4 cloth_0↔cloth_1)。
                # 每轮只推开厘米级而重叠量不变 → 同一对反复触发, 迭代发散
                # (实测 127 次后仍未收敛, cloth↔cloth 28 次、bowl↔bowl 22 次)。
                # 这里取 max(pen_depth 推出的距离, 该轴的实际 AABB 重叠 + 余量)。
                _ov_axis = {0: ox, 1: oy, 2: oz}[sep_axis]
                if _ov_axis > 0:
                    sep_dist = max(sep_dist, float(_ov_axis) + 0.005)

                # ── 共享穿模修复: 两物体各自退一半 ──
                # 条件: shared_penetration=True, 且两个物体都可移动（非 floor/wall, 非 z-only 受限小物体）
                use_shared = (
                    shared_penetration
                    and not both_small
                    and not (cat_i_is_floor or cat_i_is_wall)
                    and not (cat_j_is_floor or cat_j_is_wall)
                    and not move_is_supported
                )

                def _apply_translation(mesh_key, info_key, translate_vec, half=False):
                    """应用平移并更新 all_meshes 和 all_instances"""
                    scale = 0.5 if half else 1.0
                    actual_vec = translate_vec * scale
                    old_pos = info_key["T"][:3, 3].copy()
                    new_T = trimesh.transformations.translation_matrix(actual_vec) @ info_key["T"]
                    info_key["T"] = new_T
                    new_mesh = _get_transformed_mesh(info_key)
                    if new_mesh.bounds[0, 2] < 0.0:
                        z_fix = -new_mesh.bounds[0, 2]
                        info_key["T"] = trimesh.transformations.translation_matrix(
                            np.array([0.0, 0.0, z_fix])
                        ) @ info_key["T"]
                        new_mesh = _get_transformed_mesh(info_key)
                    all_meshes[mesh_key] = (
                        all_meshes[mesh_key][0],
                        all_meshes[mesh_key][1],
                        new_mesh,
                        info_key,
                    )
                    return old_pos

                if use_shared:
                    # 共享模式: 两个物体各退一半
                    half_dist = sep_dist / 2.0
                    # move_idx 方向: 推离对方
                    translate_move = np.array([0.0, 0.0, 0.0])
                    translate_move[sep_axis] = direction * half_dist
                    old_pos_move = _apply_translation(move_idx, move_info, translate_move, half=True)

                    # other_idx 方向: 反向推
                    other_idx = j if move_idx == i else i
                    other_cat = all_meshes[other_idx][0]
                    other_info = all_meshes[other_idx][3]
                    other_cat_key = other_cat.lower().strip()
                    translate_other = np.array([0.0, 0.0, 0.0])
                    translate_other[sep_axis] = -direction * half_dist
                    old_pos_other = _apply_translation(other_idx, other_info, translate_other, half=True)

                    all_instances[move_cat][move_idx_orig] = move_info
                    all_instances[other_cat][all_meshes[other_idx][1]] = other_info

                    # 层级传播: 共享模式下, 两个物体都是可移动大物体, 各自传播
                    for side_name, side_idx, side_info, side_old_pos in [
                        ("move", move_idx, move_info, old_pos_move),
                        ("other", other_idx, other_info, old_pos_other),
                    ]:
                        side_cat = all_meshes[side_idx][0]
                        side_is_supported = (
                            cat_i_is_supported if side_idx == i else cat_j_is_supported
                        )
                        if not side_is_supported:
                            new_pos = side_info["T"][:3, 3]
                            xy_delta = np.array([new_pos[0] - side_old_pos[0],
                                                 new_pos[1] - side_old_pos[1], 0.0])
                            if np.linalg.norm(xy_delta[:2]) > 1e-6:
                                supporter_key = (side_cat.lower().strip(), all_meshes[side_idx][1])
                                for s_cat, s_idx in supporter_to_supported.get(supporter_key, []):
                                    if s_cat not in all_instances or s_idx >= len(all_instances[s_cat]):
                                        continue
                                    s_info = all_instances[s_cat][s_idx]
                                    s_info["T"] = trimesh.transformations.translation_matrix(xy_delta) @ s_info["T"]
                                    for m_idx in range(len(all_meshes)):
                                        m_cat, m_idx_orig, _, _ = all_meshes[m_idx]
                                        if m_cat.lower().strip() == s_cat and m_idx_orig == s_idx:
                                            all_meshes[m_idx] = (m_cat, m_idx_orig, _get_transformed_mesh(s_info), s_info)
                                            break
                                    if verbose:
                                        print(f"      📎 层级跟随(共享): {s_cat}_{s_idx} 跟随 {side_cat}_{all_meshes[side_idx][1]} "
                                              f"xy偏移 ({xy_delta[0]:+.4f}, {xy_delta[1]:+.4f})", flush=True)

                    if verbose:
                        print(f"      🔧 共享穿模修复: {pair_desc} | {axis_name}轴各退 {half_dist:.4f}m"
                              f" (共 {sep_dist:.4f}m)", flush=True)
                else:
                    # 单侧模式（原逻辑）
                    translation_vec = np.array([0.0, 0.0, 0.0])
                    translation_vec[sep_axis] = direction * sep_dist

                    old_pos = move_info["T"][:3, 3].copy()
                    new_T = trimesh.transformations.translation_matrix(translation_vec) @ move_info["T"]
                    move_info["T"] = new_T

                    new_mesh = _get_transformed_mesh(move_info)
                    if new_mesh.bounds[0, 2] < 0.0:
                        z_fix = -new_mesh.bounds[0, 2]
                        move_info["T"] = trimesh.transformations.translation_matrix(
                            np.array([0.0, 0.0, z_fix])
                        ) @ move_info["T"]
                        new_mesh = _get_transformed_mesh(move_info)

                    all_meshes[move_idx] = (move_cat, move_idx_orig, new_mesh, move_info)
                    all_instances[move_cat][move_idx_orig] = move_info

                # 层级传播: supporter (大物体) 移动时, supported (小物体) 跟随 x,y
                # 共享模式下已在 if use_shared 内部完成层级传播, 这里跳过
                if not use_shared and not move_is_supported:
                    new_pos = move_info["T"][:3, 3]
                    xy_delta = np.array([new_pos[0] - old_pos[0], new_pos[1] - old_pos[1], 0.0])
                    if np.linalg.norm(xy_delta[:2]) > 1e-6:
                        supporter_key = (move_cat.lower().strip(), move_idx_orig)
                        for s_cat, s_idx in supporter_to_supported.get(supporter_key, []):
                            if s_cat not in all_instances or s_idx >= len(all_instances[s_cat]):
                                continue
                            s_info = all_instances[s_cat][s_idx]
                            s_info["T"] = trimesh.transformations.translation_matrix(xy_delta) @ s_info["T"]
                            for m_idx in range(len(all_meshes)):
                                m_cat, m_idx_orig, _, _ = all_meshes[m_idx]
                                if m_cat.lower().strip() == s_cat and m_idx_orig == s_idx:
                                    all_meshes[m_idx] = (m_cat, m_idx_orig, _get_transformed_mesh(s_info), s_info)
                                    break
                            if verbose:
                                print(f"      📎 层级跟随: {s_cat}_{s_idx} 跟随 {move_cat}_{move_idx_orig} "
                                      f"xy偏移 ({xy_delta[0]:+.4f}, {xy_delta[1]:+.4f})", flush=True)

                any_resolved = True
                _iteration_total_sep += sep_dist
                if verbose:
                    extra = " (z-only, 小物体)" if move_is_supported and sep_axis == 2 else ""
                    print(f"      🔧 穿模修复: {pair_desc} | {axis_name}轴分离 {sep_dist:.4f}m{extra}", flush=True)

        # ── 收敛判据 ──
        # 仅有 max_iterations 兜底不够: 每轮都在"修"(any_resolved=True) 但实际
        # 重叠没减少时会一直空转。加"连续 N 轮总分离量未下降则停止"。
        # _prev_total_sep 由外层循环维护 (见下方), 首轮为None 表示无历史。
        if dry_run:
            break
        if not any_resolved:
            break
        _total_sep = _iteration_total_sep
        if _prev_total_sep is not None and _total_sep >= _prev_total_sep * 0.9:
            _stall_rounds += 1
            if verbose:
                print(f"      ⚠️ 穿模修复无进展 ({_stall_rounds}/{_PEN_STALL_LIMIT}): "
                      f"总分离量 {_prev_total_sep:.4f} → {_total_sep:.4f} m, 停止迭代",
                      flush=True)
            if _stall_rounds >= _PEN_STALL_LIMIT:
                if verbose:
                    print(f"      ⚠️ 穿模修复未能收敛, 保留当前位姿(仍有残留重叠)", flush=True)
                break
        else:
            _stall_rounds = 0
        _prev_total_sep = _total_sep

    if verbose:
        if dry_run:
            if penetration_warnings:
                print(f"   ⚠️ 检测到 {len(penetration_warnings)} 处穿模 (未自动修复, 请检查 pose_changes.json)", flush=True)
            else:
                print(f"   ✅ 无穿模", flush=True)
        else:
            print(f"   ✅ 穿模检测完成 (迭代{iteration + 1}次)", flush=True)

    return all_instances


def check_stability(all_instances, refined_relations=None, categories_and_relations=None,
                    contact_threshold=0.2, gap_threshold=0.05, verbose=True):
    """检查物体稳定性: 接触面积是否足够、是否悬空

    对每个 "supported by {name}" 的物体:
      1. 检查是否与支撑物有接触 (非悬空)
      2. 检查接触面积占比是否足够 (非边缘放置)

    对每个 "supported by floor" 的物体:
      1. 检查是否接触地面 (bottom_z ≈ 0)

    参数:
        all_instances: {category: [instance_info, ...]}
        refined_relations: 关系字典
        categories_and_relations: 原始关系字典
        contact_threshold: 接触面积占比阈值 (默认0.2, 即底面20%需有接触)
        gap_threshold: 悬空判定阈值 (默认0.05m, 即5cm间隙视为悬空)
        verbose: 是否打印详细信息
    返回:
        (all_instances, unstable_list): 更新后的实例 + 不稳定物体列表
    """
    _all_rels = {}
    if categories_and_relations:
        _all_rels.update(categories_and_relations)
    if refined_relations:
        _all_rels.update(refined_relations)

    unstable_list = []
    fixed_count = 0

    floor_items = []
    supported_items = []
    for category, instances in all_instances.items():
        for idx, info in enumerate(instances):
            inst_key = f"{category}_{idx}"
            rel = _all_rels.get(inst_key, _all_rels.get(category, ""))
            rel_lower = rel.lower()
            if "floor" in rel_lower:
                floor_items.append((category, idx, info, rel, rel_lower))
            elif rel.startswith("supported by ") and "other objects" not in rel_lower and "floor" not in rel_lower:
                supported_items.append((category, idx, info, rel, rel_lower))

    # ── Phase 1: 地面物体旋转对齐 + z轴贴合 ──
    # 学习 sp_refinement.py 的 theta_gravity 逻辑，但阈值更宽松
    # 基础精修只在 theta_gravity < 10° 时对齐旋转，这里放宽到 1°
    # 记录已稳定的物体, Phase 4 不再重复修复
    stabilized = set()
    for category, idx, info, rel, rel_lower in floor_items:
        T = info["T"]
        upper_transformed = T[:3, 1] / np.linalg.norm(T[:3, 1])
        theta_gravity = np.arccos(np.clip(np.dot(np.array([0, 0, 1]), upper_transformed), -1.0, 1.0)) / np.pi * 180

        rotation_fixed = False
        if theta_gravity > 1.0 and theta_gravity < 179.0:
            # ── 扁平物体保护 ──
            # 为什么需要: Phase1 会把物体 y 轴对齐重力方向 (为落地家具设计)。
            # 但对**扁平物体** (桌面、床板、平板) 这是灾难 —— 一个 5cm 厚的
            # 桌面被转 54.1° 后会竖起来, AABB 高度从 0.05m 变成 0.99m。
            # 而 Phase4 随后会读这个被抬高的 supporter_top 去贴桌上物体,
            # 把耳机盒推到 1 米高的空中 (实测 wxD: earbud case_0 z +0.9365m,
            # table_0 顶面 0.050 → 0.987)。
            # 判据是纯几何、可测量的: 若当前 AABB 已经"扁" (最薄维度 / 最大
            # 维度 < 0.25) 且旋转后高度会显著增大, 说明该物体的"自然朝向"
            # 就是躺着的, 不该被扶正。不依赖类别名。
            _m0 = _get_transformed_mesh(info)
            _e0 = _m0.bounds[1] - _m0.bounds[0]
            _mx, _mn = float(max(_e0)), float(min(_e0))
            _is_flat = _mx > 1e-6 and (_mn / _mx) < 0.35
            _rot_would_raise = False
            if _is_flat:
                # 估算旋转后的高度: 旋转把"最薄维"之一转到 z 轴的极端情形
                _srt = np.sort(_e0)
                _worst_h = float(_srt[1])      # 旋转后可能的最大高度
                if _worst_h > float(_e0[2]) * 1.5 + 0.02:
                    _rot_would_raise = True
            if _rot_would_raise:
                # 只贴地, 不扶正 —— 保持物体原有的躺倒姿态
                if abs(_m0.bounds[0, 2]) > 0.001:
                    T[2, 3] -= float(_m0.bounds[0, 2])
                parts = [f"扁平物体保护: 跳过扶正 (尺寸={_e0.round(3).tolist()}, "
                         f"theta={theta_gravity:.1f}°), 仅贴地"]
                if verbose:
                    print(f"      🔧 Phase1 地面对齐: {category}_{idx} "
                          f"{', '.join(parts)}", flush=True)
                continue
            if theta_gravity < 90.0:
                align_matrix = trimesh.geometry.align_vectors(upper_transformed, np.array([0, 0, 1]))
            else:
                # 网格 y-up: theta>90° = 物体倒置, 应翻回 +z (180° 翻正)。
                # 旧实现 align 到 -z 会把倒置物体转成"完美倒置" (wxD chair_0)。
                align_matrix = trimesh.geometry.align_vectors(upper_transformed, np.array([0, 0, 1]))
            T[:3, :3] = align_matrix[:3, :3] @ T[:3, :3]
            rotation_fixed = True

        # z 轴贴合: 确保底面在 z=0。
        # 门限与 sp_refinement 基础精修一致: 沉地/轻微悬空正常贴合;
        # 但 bottom_z > 0.5m 的高空物体不自动砸地 —— 那几乎总是上游
        # 关系/位姿错误的信号 (实测 wxD: 被错误翻成 floor 的桌上物体
        # 从 0.9m 砸到 0), 保留悬空并告警, 让问题可见。
        mesh = _get_transformed_mesh(info)
        bottom_z = mesh.bounds[0, 2]
        z_fixed = False
        if bottom_z < -0.001:
            T[2, 3] -= bottom_z
            z_fixed = True
        elif bottom_z > 0.5:
            if verbose:
                print(f"      ⚠️ Phase1 贴地: {category}_{idx} bottom_z="
                      f"{bottom_z:+.3f}m 离地过远 → 不自动砸地 (疑似上游错误)",
                      flush=True)
        elif abs(bottom_z) > 0.001:
            T[2, 3] -= bottom_z
            z_fixed = True

        if rotation_fixed or z_fixed:
            fixed_count += 1
            if verbose:
                parts = []
                if rotation_fixed:
                    parts.append(f"旋转对齐 (theta={theta_gravity:.1f}°→0°)")
                if z_fixed:
                    parts.append(f"z轴贴合 (bottom_z={bottom_z:.4f}m→0)")
                print(f"      🔧 Phase1 地面对齐: {category}_{idx} {', '.join(parts)}", flush=True)

        # 验证稳定: 底面 z=0 且 theta<10°
        mesh_after = _get_transformed_mesh(info)
        if abs(mesh_after.bounds[0, 2]) < 0.001 and theta_gravity < 10:
            stabilized.add(f"{category}_{idx}")

    # ── Phase 2: 支撑物体旋转对齐 + 间隙检测 + 修复 ──
    # 学习 sp_refinement.py 的 theta_gravity 逻辑，但调低阈值
    # 基础精修阈值 10°，这里用 5° 以识别更多需要旋转对齐的物体
    for category, idx, info, rel, rel_lower in supported_items:
        T = info["T"]
        upper_transformed = T[:3, 1] / np.linalg.norm(T[:3, 1])
        theta_gravity = np.arccos(np.clip(np.dot(np.array([0, 0, 1]), upper_transformed), -1.0, 1.0)) / np.pi * 180

        rotation_fixed = False
        if theta_gravity > 5.0 and theta_gravity < 175.0:
            if theta_gravity < 90.0:
                align_matrix = trimesh.geometry.align_vectors(upper_transformed, np.array([0, 0, 1]))
            else:
                # 网格 y-up: theta>90° = 倒置, 翻回 +z。旧实现 align 到 -z,
                # 会把半翻倒的物体主动转成"完美倒置"。
                align_matrix = trimesh.geometry.align_vectors(upper_transformed, np.array([0, 0, 1]))
            T[:3, :3] = align_matrix[:3, :3] @ T[:3, :3]
            rotation_fixed = True
            if verbose:
                print(f"      🔧 Phase2 旋转对齐: {category}_{idx} theta={theta_gravity:.1f}°→0°", flush=True)

        mesh = _get_transformed_mesh(info)
        bottom_z = mesh.bounds[0, 2]
        supporter_name = rel[len("supported by "):].strip()
        supporter_cat = supporter_name.rsplit("_", 1)[0] if "_" in supporter_name else supporter_name

        supporter_info = _find_supporter_info(all_instances, supporter_name, supporter_cat)
        if supporter_info is None:
            continue

        supporter_mesh = _get_transformed_mesh(supporter_info)
        supporter_top_z = supporter_mesh.bounds[1, 2]

        gap = bottom_z - supporter_top_z
        if gap > gap_threshold:
            unstable_list.append({
                "name": f"{category}_{idx}",
                "issue": "悬空",
                "detail": f"距支撑面 {gap:.3f}m (阈值 {gap_threshold}m)",
            })
            # xy 拉回: 只修 z 会把物体放到"支撑面同高但水平错位"的位置
            _pull = _pull_xy_into_supporter(mesh, supporter_mesh, verbose, f"{category}_{idx}")
            if _pull is not None and _pull[2] > 0:
                _dx, _dy, _shift = _pull
                info["T"] = trimesh.transformations.translation_matrix(
                    np.array([_dx, _dy, 0.0])) @ info["T"]
                mesh = _get_transformed_mesh(info)  # 倾斜物体的 bbox 会随 xy 变化, 重算
                bottom_z = mesh.bounds[0, 2]
                gap = bottom_z - supporter_top_z
                if verbose:
                    print(f"      🔧 Phase2 xy拉回: {category}_{idx} 平移({_dx:+.3f},{_dy:+.3f})m "
                          f"→ 支撑物足迹内 (距离 {_shift:.3f}m)", flush=True)
            z_fix = gap
            info["T"] = trimesh.transformations.translation_matrix(
                np.array([0.0, 0.0, -z_fix])
            ) @ info["T"]
            fixed_count += 1
            if verbose:
                print(f"      🔽 Phase2 悬空修复: {category}_{idx} 悬空 {gap:.3f}m → 落到 {supporter_name} 顶面", flush=True)
            stabilized.add(f"{category}_{idx}")
            continue

        # ── Phase 3: 接触不足检测 + z轴修复 ──
        supported_proj = _project_footprint(mesh)
        supporter_proj = _project_footprint(supporter_mesh)

        try:
            overlap = supported_proj.intersection(supporter_proj)
            overlap_area = overlap.area if overlap is not None and not overlap.is_empty else 0.0
        except Exception:
            overlap_area = 0.0

        supported_area = supported_proj.area if supported_proj is not None and not supported_proj.is_empty else 1e-6
        support_ratio = overlap_area / supported_area if supported_area > 1e-6 else 0.0

        if support_ratio < contact_threshold:
            # ── xy 拉回: 足迹几乎无重叠时, 先把物体 xy 拉回支撑物足迹内 ──
            if support_ratio < 0.05:
                _pull = _pull_xy_into_supporter(mesh, supporter_mesh, verbose, f"{category}_{idx}")
                if _pull is None:
                    unstable_list.append({
                        "name": f"{category}_{idx}",
                        "issue": "接触不足",
                        "detail": "xy距支撑物 >0.6m, 不自动拉回, 需人工确认",
                    })
                    continue
                _dx, _dy, _shift = _pull
                if _shift > 0:
                    info["T"][0, 3] += _dx
                    info["T"][1, 3] += _dy
                    mesh = _get_transformed_mesh(info)  # xy 变了, 重算包围盒供 z 对齐
                    if verbose:
                        print(f"      🔧 Phase3 xy拉回: {category}_{idx} 平移({_dx:+.3f},{_dy:+.3f})m "
                              f"→ 支撑物足迹内 (距离 {_shift:.3f}m)", flush=True)

            # 接触不足: 直接对齐 z 轴
            supporter_top_z = supporter_mesh.bounds[1, 2]
            supported_bottom_z = mesh.bounds[0, 2]
            z_gap = supporter_top_z - supported_bottom_z

            if abs(z_gap) > 0.001:
                correction = z_gap
                info["T"][2, 3] += correction
                fixed_count += 1
                if verbose:
                    print(f"      🔧 Phase3 接触不足修复: {category}_{idx} z轴修正{correction:+.4f}m "
                          f"(接触比 {support_ratio:.1%} < {contact_threshold:.0%})", flush=True)
                stabilized.add(f"{category}_{idx}")
            else:
                unstable_list.append({
                    "name": f"{category}_{idx}",
                    "issue": "接触不足",
                    "detail": f"接触面积比 {support_ratio:.1%} (阈值 {contact_threshold:.0%}), z已对齐但xy偏移",
                })

    # ── Phase 4: 最终 z 轴强制贴合 (兜底) ──
    # 仅对 Phase 1-3 未修复的物体进行, 避免重复修复
    phase4_fixed = 0
    for category, idx, info, rel, rel_lower in floor_items:
        inst_key = f"{category}_{idx}"
        if inst_key in stabilized:
            continue  # Phase 1 已对齐
        mesh = _get_transformed_mesh(info)
        bottom_z = mesh.bounds[0, 2]
        if abs(bottom_z) > 0.001:
            info["T"][2, 3] -= bottom_z
            phase4_fixed += 1
            if verbose:
                print(f"      🔧 Phase4 地面贴合: {category}_{idx} bottom_z={bottom_z:.4f}m→0", flush=True)

    for category, idx, info, rel, rel_lower in supported_items:
        inst_key = f"{category}_{idx}"
        if inst_key in stabilized:
            continue  # Phase 2/3 已对齐, 不再重复
        mesh = _get_transformed_mesh(info)
        bottom_z = mesh.bounds[0, 2]
        supporter_name = rel[len("supported by "):].strip()
        supporter_cat = supporter_name.rsplit("_", 1)[0] if "_" in supporter_name else supporter_name

        supporter_info = _find_supporter_info(all_instances, supporter_name, supporter_cat)
        if supporter_info is None:
            continue

        supporter_mesh = _get_transformed_mesh(supporter_info)
        supporter_top_z = supporter_mesh.bounds[1, 2]

        z_gap = supporter_top_z - bottom_z
        if abs(z_gap) > 0.001:
            info["T"][2, 3] += z_gap
            phase4_fixed += 1
            if verbose:
                print(f"      🔧 Phase4 支撑贴合: {category}_{idx} z修正{z_gap:+.4f}m "
                      f"(bottom_z={bottom_z:.4f} → supporter_top={supporter_top_z:.4f})", flush=True)

    fixed_count += phase4_fixed

    if verbose:
        if unstable_list:
            contact_issues = sum(1 for u in unstable_list if u["issue"] == "接触不足")
            gap_issues = sum(1 for u in unstable_list if u["issue"] == "悬空")
            msg = f"   ⚠️ 稳定性检查: {len(unstable_list)} 个不稳定"
            if gap_issues:
                msg += f" (悬空 {gap_issues} 个, 已修复 {fixed_count} 个)"
            if contact_issues:
                msg += f" (接触不足 {contact_issues} 个, 需人工确认)"
            if phase4_fixed:
                msg += f" [Phase4额外修复 {phase4_fixed} 个]"
            print(msg, flush=True)
        else:
            extra = f" [Phase4额外修复 {phase4_fixed} 个]" if phase4_fixed else ""
            print(f"   ✅ 稳定性检查: 所有物体稳定{extra}", flush=True)

    return all_instances, unstable_list


def _find_supporter_info(all_instances, supporter_name, supporter_cat):
    """在all_instances中查找支撑物的instance_info

    supporter_name 可能是 "table_1" (带实例索引) 或 "table" (不带).
    all_instances 的 key 是类别名 (如 "table"), value 是实例列表.
    当 supporter_name 带索引时, 解析出索引返回对应实例, 而非总是 [0].
    """
    # 解析实例索引: "table_1" → cat="table", idx=1
    inst_idx = None
    if "_" in supporter_name:
        parts = supporter_name.rsplit("_", 1)
        if parts[1].isdigit():
            inst_idx = int(parts[1])

    # 按类别查找实例列表
    instances = None
    if supporter_cat in all_instances and all_instances[supporter_cat]:
        instances = all_instances[supporter_cat]
    else:
        for cat, insts in all_instances.items():
            cat_base = cat.rsplit("_", 1)[0] if "_" in cat else cat
            if cat_base.lower() == supporter_cat.lower() or cat.lower() == supporter_name.lower():
                if insts:
                    instances = insts
                    break

    if not instances:
        return None

    # 有明确索引且在范围内 → 返回对应实例
    if inst_idx is not None and 0 <= inst_idx < len(instances):
        return instances[inst_idx]

    # 无索引或索引越界 → 回退到 [0]
    return instances[0]


def _project_footprint(mesh):
    """将mesh投影到xy平面, 返回shapely Polygon"""
    try:
        from shapely.geometry import MultiPoint
        vertices_xy = mesh.vertices[:, :2]
        if len(vertices_xy) < 3:
            return None
        return MultiPoint(vertices_xy).convex_hull
    except ImportError:
        return None


def _pull_xy_into_supporter(mesh, supporter_mesh, verbose=False, name=""):
    """把物体 xy 平移到支撑物足迹内缩区的最近点 (Phase2/Phase3 共用)

    重建/SAM3D 给出的 xy 可能偏离支撑物, 只修 z 会造成"同高但水平悬空"。
    内缩 0.6×物体半径, 允许轻微出沿; 位移上限 0.6m 防误判瞬移。

    返回:
        (dx, dy, shift) — 需要平移时 shift>0; 无需移动 (0,0,0.0);
        距离超限返回 None (调用方应标记 unstable)
    """
    try:
        from shapely.ops import nearest_points as _nearest_points
        supported_proj = _project_footprint(mesh)
        supporter_proj = _project_footprint(supporter_mesh)
        if (supported_proj is None or supporter_proj is None
                or supported_proj.is_empty or supporter_proj.is_empty):
            return (0.0, 0.0, 0.0)
        obj_half = max(supported_proj.bounds[2] - supported_proj.bounds[0],
                       supported_proj.bounds[3] - supported_proj.bounds[1]) / 2
        inner = supporter_proj.buffer(-0.6 * obj_half)
        if inner.is_empty:
            inner = supporter_proj
        obj_center = supported_proj.centroid
        target = _nearest_points(inner, obj_center)[0]
        dx, dy = target.x - obj_center.x, target.y - obj_center.y
        shift = float(np.hypot(dx, dy))
        if shift <= 1e-3:
            return (0.0, 0.0, 0.0)
        if shift > 0.6:
            if verbose:
                print(f"      ⚠️ {name} xy距支撑物 {shift:.2f}m > 0.6m, 不自动拉回", flush=True)
            return None
        return (dx, dy, shift)
    except Exception as e:
        if verbose:
            print(f"      ⚠️ {name} xy拉回失败: {e}", flush=True)
        return (0.0, 0.0, 0.0)

def refine_inter_object_relations(all_instances, refined_relations,
                                  walls_info=None, verbose=True,
                                  vlm_checkpoint=None, scene_dir=None,
                                  categories_and_relations=None,
                                  only_refine_other_objects=False,
                                  initial_T_snapshot=None,
                                  final_glb_name="final_scene_stage5.glb",
                                  preloaded_vlm=None):
    """
    主函数: 精修物体间支撑关系的空间位置

    策略判定优先级:
      1. VLM (如果提供 vlm_checkpoint + scene_dir) → 按实例帧投票
         帧来源: optimal_frames/ + keyframes/ (与 refine_other_objects_relations.py 一致)
      2. 规则回退 → 默认 on_top

    参数:
        all_instances: {category: [instance_info, ...]}
        refined_relations: {name: relationship}  细化后的关系
        walls_info: 墙面信息（供 against_side 策略使用）
        verbose: 是否打印详细信息
        vlm_checkpoint: VLM模型路径（可选）
        scene_dir: 场景输出目录（如 outputs/232，包含 optimal_frames/ 和 keyframes/）
        only_refine_other_objects: 只精修 "supported by other objects" 的物体,
                                   已精修的 floor/wall 不动
    返回:
        更新后的 all_instances
    """
    if verbose:
        print("=" * 70, flush=True)
        print("🔧 物体间支撑关系空间位置精修", flush=True)
        print("=" * 70, flush=True)

    supported_pairs = []
    for name, rel in refined_relations.items():
        if rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel:
            supporter_name = rel.replace("supported by ", "").strip()
            supported_pairs.append((name, supporter_name))

    if verbose:
        print(f"\n📋 发现 {len(supported_pairs)} 对物体间支撑关系:", flush=True)
        for supported, supporter in supported_pairs:
            print(f"   {supported} ← {supporter}", flush=True)

    if not supported_pairs:
        if verbose:
            print(f"\n✅ 无需精修", flush=True)
        return all_instances

    # ── 策略判定 ──
    placement_strategies = {}
    _final_strategies = {}   # {实例名: 最终策略}, 循环结束后落盘供 5.25 使用

    use_vlm = vlm_checkpoint is not None and scene_dir is not None

    if use_vlm:
        resolved_ckpt = _resolve_vlm_checkpoint(vlm_checkpoint)
        if not resolved_ckpt:
            use_vlm = False
            if verbose:
                print(f"\n⚠️ VLM模型未找到，回退到规则匹配", flush=True)

    if use_vlm:
        if verbose:
            print(f"\n📐 放置策略（VLM判定，帧来源: optimal_frames/ + keyframes/）:", flush=True)

        # 收集所有涉及的物体类别（被支撑物 + 支撑物）
        all_categories = set()
        for supported_name, supporter_name in supported_pairs:
            all_categories.add(supported_name)
            all_categories.add(supporter_name)
        # 也加入 refined_relations 中的所有 key 以支持模糊匹配
        for name in refined_relations.keys():
            all_categories.add(name)

        # 归一化为类别名: 实例名 "bottle_0" -> "bottle"
        # 否则 _fuzzy_match_name 会匹配到 "bottle_0" 并生成 ghost key "bottle_0_0"
        import re as _re_norm
        normalized_categories = set()
        for name in all_categories:
            stripped = _re_norm.sub(r'_\d+$', '', name)
            normalized_categories.add(stripped)
        all_categories = list(normalized_categories)

        # 加载帧: optimal_frames/ + keyframes/
        optimal_frames_dir = os.path.join(scene_dir, "optimal_frames")
        keyframes_dir = os.path.join(scene_dir, "keyframes")

        # 回退: 如果 {scene_dir}/keyframes/ 不存在，尝试 assets/key_frames/{scene_id}/
        if not os.path.isdir(keyframes_dir):
            scene_id = os.path.basename(os.path.normpath(scene_dir))
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            fallback_dir = os.path.join(project_root, "assets", "key_frames", scene_id)
            if os.path.isdir(fallback_dir):
                keyframes_dir = fallback_dir
                if verbose:
                    print(f"   ⚠️ {scene_dir}/keyframes/ 不存在，回退到 {fallback_dir}", flush=True)

        if verbose:
            print(f"\n   📷 [来源1] 加载optimal_frames: {optimal_frames_dir}", flush=True)
        optimal_instance_frames, instance_count = _load_optimal_frames_for_instances(
            optimal_frames_dir, list(all_categories)
        )
        if verbose:
            total_opt = sum(len(v) for v in optimal_instance_frames.values())
            print(f"      optimal_frames/: {total_opt} 帧, 覆盖 {len(optimal_instance_frames)} 个实例", flush=True)

        if verbose:
            print(f"   📷 [来源2] 加载keyframes: {keyframes_dir}", flush=True)
        keyframe_images, keyframe_visibility = _load_keyframes_for_instances(
            keyframes_dir, list(all_categories)
        )
        if verbose:
            print(f"      keyframes/: {len(keyframe_images)} 个关键帧, {len(keyframe_visibility)} 帧可见性数据", flush=True)

        # 合并帧来源
        instance_to_frames, instance_to_category = _build_instance_frame_map(
            optimal_instance_frames, instance_count,
            keyframe_images, keyframe_visibility,
            list(all_categories),
        )

        if verbose:
            print(f"\n   📋 实例→帧映射:", flush=True)
            for inst_key in sorted(instance_to_category.keys()):
                frames = instance_to_frames.get(inst_key, [])
                if frames:
                    sources = [src for _, src in frames]
                    print(f"      {inst_key}: {len(frames)} 帧 → {sources}", flush=True)
                else:
                    print(f"      {inst_key}: ❌ 无帧", flush=True)

        # 加载VLM (优先使用预加载的, 避免重复加载)
        if preloaded_vlm is not None:
            vlm_model, vlm_processor = preloaded_vlm
            if verbose:
                print(f"   🤖 使用预加载VLM", flush=True)
        else:
            vlm_model, vlm_processor = _load_vlm_model(resolved_ckpt)

        # 按实例投票判定策略
        for supported_name, supporter_name in supported_pairs:
            # 收集属于该类别的实例帧。
            # 只取帧最多的一个实例作代表: 要回答的问题是"supported_name 相对
            # supporter_name 怎么放", 与具体哪个实例无关。此前遍历该类别下
            # 每个实例各投一次 (bottle 有 16 实例 → 16x5=80 次VLM 调用),
            # 但结果只取第一个 (best_strategy), 其余 15 个纯属浪费。
            # HOI4D 082_C5 实测: 每物体 60~65 票, 全场景 1078 次调用 (~50min)。
            # 多实例投票语义在此保留扩展点: 若将来需要按实例区分, 应改用
            # 一次调用问多个实例 (如 5.1 的 query_instances 做法)。
            cat_frames = []
            for inst_key, frames in instance_to_frames.items():
                cat = instance_to_category.get(inst_key, "")
                # 模糊匹配: inst_key 的 category 部分匹配 supported_name
                if cat != supported_name and not _fuzzy_match_name(supported_name, [cat]):
                    continue
                if not frames:
                    continue
                cat_frames.append((len(frames), inst_key, frames))

            best_strategy = None
            if cat_frames:
                # 代表实例: 帧最多者 (可见角度最充分)
                _, rep_key, rep_frames = max(cat_frames, key=lambda x: x[0])
                best_strategy = vlm_lookup_placement_strategy(
                    rep_key, supporter_name, supported_pairs,
                    vlm_model, vlm_processor, rep_frames,
                )

            if best_strategy is None:
                best_strategy = "on_top"

            placement_strategies[supported_name] = best_strategy
            desc = PLACEMENT_STRATEGIES.get(best_strategy, "")
            if verbose:
                print(f"   {supported_name}: {best_strategy} — {desc} (VLM)", flush=True)

        # 不卸载VLM, 保留用于SP精修程度判断
        if verbose:
            print(f"   ℹ️ VLM保留加载, 用于SP精修程度判断", flush=True)
    else:
        if verbose:
            print(f"\n📐 放置策略（规则回退，默认 on_top）:", flush=True)
        for supported_name, supporter_name in supported_pairs:
            strategy = lookup_placement_strategy(supported_name, supporter_name)
            placement_strategies[supported_name] = strategy
            desc = PLACEMENT_STRATEGIES.get(strategy, "")
            if verbose:
                print(f"   {supported_name}: {strategy} — {desc}", flush=True)
        vlm_model = None
        vlm_processor = None

    # ── SP精修 (按依赖排序: 支撑物先处理) ──
    if verbose:
        print(f"\n🔧 SP精修开始 (依赖排序: 支撑物优先)...", flush=True)

    sorted_pairs = _topological_sort_pairs(supported_pairs)
    if verbose and sorted_pairs != supported_pairs:
        print(f"   📋 处理顺序:", flush=True)
        for supported, supporter in sorted_pairs:
            print(f"      {supporter} → {supported}", flush=True)

    for supported_name, supporter_name in sorted_pairs:
        strategy = placement_strategies.get(supported_name, "on_top")

        supported_cat, supported_instances = _find_supporter_instances(supported_name, all_instances)
        if supported_cat is None:
            if verbose:
                print(f"   ⚠️ 跳过 '{supported_name}': 未找到对应实例", flush=True)
            continue

        supporter_cat, supporter_instances = _find_supporter_instances(supporter_name, all_instances)
        if supporter_cat is None:
            if verbose:
                print(f"   ⚠️ 跳过 '{supported_name}': 支撑物 '{supporter_name}' 未找到", flush=True)
            continue

        sp_refine_fn = SP_REFINE_MAP.get(strategy, sp_refine_on_top)

        for inst_idx, supported_info in enumerate(supported_instances):
            supporter_info = _find_nearest_supporter_instance(supported_info, supporter_instances)

            if verbose:
                print(f"   🔧 {supported_name}[{inst_idx}] → {strategy} → {supporter_name}", flush=True)

            # 计算 initial_offset (基础精修前的 z_offset), 用于 sp_refine_on_top 判断 supporter 是否被抬升
            initial_offset = None
            if initial_T_snapshot is not None and strategy == "on_top":
                try:
                    _init_supporter = initial_T_snapshot.get(supporter_cat, [])
                    _init_supported = initial_T_snapshot.get(supported_cat, [])
                    if _init_supported and _init_supporter:
                        # 找最接近的 supporter 实例 (用 xy 距离)
                        _sup_xy = supporter_info['T'][:2, 3]
                        _sup_idx_init = 0
                        if len(_init_supporter) > 1:
                            _best_d = float('inf')
                            for _k, _init_T in enumerate(_init_supporter):
                                _init_xy = _init_T[:2, 3]
                                _d = np.linalg.norm(_init_xy - _sup_xy)
                                if _d < _best_d:
                                    _best_d = _d
                                    _sup_idx_init = _k
                        # 复算 supported/supporter 的初始 bottom/top (用快照中的 T)
                        from copy import deepcopy
                        _init_supported_T = deepcopy(_init_supported[inst_idx]) if inst_idx < len(_init_supported) else deepcopy(supported_info['T'])
                        _init_supporter_T = deepcopy(_init_supporter[_sup_idx_init])
                        _init_supported_mesh = supported_info['original_mesh'].copy()
                        _init_supported_mesh.apply_transform(_init_supported_T)
                        _init_supported_bottom_z = _init_supported_mesh.bounds[0, 2]
                        _init_supporter_mesh = supporter_info['original_mesh'].copy()
                        _init_supporter_mesh.apply_transform(_init_supporter_T)
                        _init_supporter_top_z = _init_supporter_mesh.bounds[1, 2]
                        initial_offset = float(_init_supporter_top_z - _init_supported_bottom_z)
                except Exception as e:
                    if verbose:
                        print(f"      ⚠️  initial_offset 计算失败: {e}", flush=True)
                    initial_offset = None

            # 精修程度 (大精修/小精修/正确) —— 几何优先, VLM 只在几何已接近时表态
            #
            # VLM 是看**照片**回答的, 而照片里物体往往确实在支撑物上 (视频就是这么拍的),
            # 但 3D 坐标可能完全错位。HOI4D 082_C5 实测: 瓶底 z≈0 而台面 z=0.971
            # (悬空 0.971m), VLM 却 23/23 回答 "correct", 导致 on_top 对齐被整体跳过,
            # 21 个物体全留在地面 —— 5.1 判对的关系被 5.2 推翻。
            # 因此: 先看几何。明显错位 (|z_offset| 大) 时直接执行对齐, 不问 VLM;
            # 只有几何本来就接近贴合时, VLM 的"是否已正确"才有意义 (此时它用于
            # 避免无谓调整)。顺带省掉大量无意义调用 (原本每对支撑都问一次 VLM)。
            refinement_degree = None
            if vlm_model is not None and strategy == "on_top":
                # 几何前置检查: 物体底面与支撑物顶面的 z 偏差
                _gap = None
                try:
                    _sup_mesh = _get_transformed_mesh(supporter_info)
                    _sup_top_z = float(_sup_mesh.bounds[1, 2])
                    _sup_mesh2 = _get_transformed_mesh(supported_info)
                    _sup_bottom_z = float(_sup_mesh2.bounds[0, 2])
                    _gap = _sup_top_z - _sup_bottom_z
                except Exception:
                    _gap = None

                if _gap is not None and abs(_gap) >= GEOM_ALIGNED_TOL:
                    # 几何上明显没对齐 → 必定需要调整, 不问VLM
                    if verbose:
                        print(f"      📐 几何偏差 {_gap:+.3f}m ≥ {GEOM_ALIGNED_TOL}m, "
                              f"跳过 VLM 判定直接对齐", flush=True)
                    refinement_degree = None      # None = 始终执行完整对齐
                else:
                    # 几何已接近贴合 → 问 VLM 是否已正确 (避免无谓调整)
                    _optimal_frame_path = None
                    _optimal_dir = os.path.join(scene_dir, "optimal_frames") if scene_dir else None
                    if _optimal_dir and os.path.isdir(_optimal_dir):
                        # 查找匹配该类别的帧
                        for _fname in sorted(os.listdir(_optimal_dir)):
                            if supported_cat.lower() in _fname.lower() or supported_name.lower() in _fname.lower():
                                _optimal_frame_path = os.path.join(_optimal_dir, _fname)
                                break
                        # 回退: 取第一张帧
                        if _optimal_frame_path is None:
                            _all_frames = sorted(os.listdir(_optimal_dir))
                            if _all_frames:
                                _optimal_frame_path = os.path.join(_optimal_dir, _all_frames[0])

                    if _optimal_frame_path and os.path.isfile(_optimal_frame_path):
                        try:
                            from PIL import Image as PILImage
                            _img = PILImage.open(_optimal_frame_path).convert("RGB")
                            refinement_degree = _judge_refinement_degree_with_vlm(
                                vlm_model, vlm_processor, _img,
                                supported_name, supporter_name
                            )
                            if verbose:
                                print(f"      📊 VLM精修程度: {refinement_degree}", flush=True)
                        except Exception as e:
                            if verbose:
                                print(f"      ⚠️ VLM精修程度判断失败: {e}", flush=True)
                            refinement_degree = None
                    if verbose and _gap is not None:
                        print(f"      📐 几何偏差 {_gap:+.3f}m < {GEOM_ALIGNED_TOL}m, 已咨询 VLM",
                              flush=True)

            try:
                # 按各精修函数的真实签名分派。
                # sp_refine_on_top(supported, supporter, max_offset, initial_offset,
                #                   refinement_degree) —— 支持全部 3 个额外参数;
                # sp_refine_inside / sp_refine_hanging_below 只有 (supported, supporter);
                # sp_refine_against_side / leaning 额外需要 walls_info。
                # 原代码对 on_top/inside 共用一个 else 分支无条件传 3 个参数,
                # 导致 inside 等策略抛 TypeError 被 except 吞掉 → 静默失效。
                if strategy == "against_side":
                    supported_instances[inst_idx] = sp_refine_fn(
                        supported_info, supporter_info, walls_info
                    )
                elif strategy == "leaning":
                    # sp_refine_leaning 只接受 2 个参数 (内部转调 against_side),
                    # 不能传 walls_info
                    supported_instances[inst_idx] = sp_refine_fn(
                        supported_info, supporter_info
                    )
                elif strategy == "on_top":
                    supported_instances[inst_idx] = sp_refine_fn(
                        supported_info, supporter_info,
                        max_offset=0.3, initial_offset=initial_offset,
                        refinement_degree=refinement_degree
                    )
                else:
                    # inside / hanging_below: 仅两个位置参数
                    supported_instances[inst_idx] = sp_refine_fn(
                        supported_info, supporter_info
                    )
            except Exception as e:
                if verbose:
                    print(f"      ❌ 精修失败 (strategy={strategy}): {e}", flush=True)

            # 落盘每实例的最终策略: refined_relations 只写 "supported by X",
            # 不含on_top / inside / against_side 的区分。5.25 的 on_top 重贴必须
            # 知道策略,否则会把 inside 的物体也贴到支撑物顶面 (实测 bread_0 是
            # inside, 会被错贴到碗沿)。
            # supported_name 已是完整实例名 (如 lego_0); 类别级时是 lego。
            # 不能再拼 _inst_idx —— 那会得到 lego_0_0, 与 5.25 查的 key 不一致,
            # 导致策略查不到、inside 被当on_top 错贴到支撑物顶面。
            _final_strategies[supported_name] = strategy
            if supported_cat is not None and inst_idx > 0:
                # 类别级名称 (如 "lego") 下的每个实例单独记录
                _final_strategies[f"{supported_cat}_{inst_idx}"] = strategy

    # 落盘策略供 5.25 使用 (best-effort: 失败不影响主流水线)
    if scene_dir:
        try:
            with open(os.path.join(scene_dir, "placement_strategies.json"), "w",
                      encoding="utf-8") as _f:
                json.dump(_final_strategies, _f, indent=2, ensure_ascii=False)
            if verbose:
                print(f"   💾 放置策略已保存 ({len(_final_strategies)} 条): "
                      f"{os.path.join(scene_dir, 'placement_strategies.json')}", flush=True)
        except Exception as e:
            if verbose:
                print(f"   ⚠️ 放置策略保存失败 (不影响主流水线): {e}", flush=True)

    # ── 全局穿模解决 ──
    all_instances = resolve_penetrations(all_instances, refined_relations, verbose=verbose,
                                         categories_and_relations=categories_and_relations,
                                         scene_dir=scene_dir,
                                         shared_penetration=True)

    # 保存 SP精修+穿模修复后的中间结果 (GLB #3)
    if scene_dir is not None:
        _sp_glb = os.path.join(scene_dir, "final_scene_stage5_sp.glb")
        try:
            _scene = trimesh.Scene()
            for _cat, _insts in all_instances.items():
                for _i, _info in enumerate(_insts):
                    _m = _info["original_mesh"].copy()
                    _m.apply_transform(_info["T"])
                    _scene.add_geometry(_m, node_name=f"{_cat}_{_i}")
            _scene.apply_transform(np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ]))
            _scene.export(_sp_glb)
            if verbose:
                print(f"   💾 SP精修中间结果已保存: {_sp_glb}", flush=True)
        except Exception as e:
            if verbose:
                print(f"   ⚠️ SP精修中间结果保存失败: {e}", flush=True)

    # ── 稳定性检查 ──
    all_instances, unstable_list = check_stability(
        all_instances, refined_relations=refined_relations,
        categories_and_relations=categories_and_relations, verbose=verbose
    )

    # Phase1 旋转对齐/Phase2 悬空修复会改变物体足印, 可能重新引入穿模
    # (此前 resolve_penetrations 的成果被旋转作废) → 稳定性检查后再解决一次
    if verbose:
        print(f"\n🛡️ 稳定性检查后再次穿模检测与解决...", flush=True)
    all_instances = resolve_penetrations(
        all_instances, refined_relations, verbose=verbose,
        categories_and_relations=categories_and_relations,
        scene_dir=scene_dir,
        shared_penetration=True)

    # 保存 check_stability 后的最终结果 (GLB #4 / GLB #5)
    if scene_dir is not None:
        _final_glb = os.path.join(scene_dir, final_glb_name)
        try:
            _scene = trimesh.Scene()
            for _cat, _insts in all_instances.items():
                for _i, _info in enumerate(_insts):
                    _m = _info["original_mesh"].copy()
                    _m.apply_transform(_info["T"])
                    _scene.add_geometry(_m, node_name=f"{_cat}_{_i}")
            _scene.apply_transform(np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1],
            ]))
            _scene.export(_final_glb)
            if verbose:
                print(f"   💾 Stage5最终结果已保存: {_final_glb}", flush=True)
            # Stage5 四视图预览 (front/side/top/iso), 渲染失败不影响管线
            try:
                from tools.render_glb_preview import render_preview_png
                render_preview_png(_final_glb)
            except Exception as _e:
                if verbose:
                    print(f"   ⚠️ Stage5 预览图生成失败: {_e}", flush=True)
        except Exception as e:
            if verbose:
                print(f"   ⚠️ Stage5最终结果保存失败: {e}", flush=True)

    if verbose:
        print(f"\n✅ 物体间关系精修完成!", flush=True)

    # 卸载VLM
    if vlm_model is not None:
        try:
            import torch
            import gc
            del vlm_model, vlm_processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    return all_instances


def refine_full_scene(all_instances, categories_and_relations, walls_info,
                      extrinsics=None, all_optimal_frame_ids=None, verbose=True,
                      vlm_checkpoint=None, scene_dir=None):
    """
    完整场景精修: 处理 floor + wall + inter-object 所有关系

    参数:
        all_instances: {category: [instance_info, ...]}
        categories_and_relations: {category: relationship}
        walls_info: 墙面信息
        extrinsics: 外参矩阵列表（用于 attached_to_wall 的 camera_pos）
        all_optimal_frame_ids: {category: [frame_id, ...]}
        verbose: 是否打印详细信息
        vlm_checkpoint: VLM模型路径（可选，用于策略判定）
        scene_dir: 场景输出目录（可选，包含 optimal_frames/ 和 keyframes/）
    返回:
        更新后的 all_instances
    """
    if verbose:
        print("=" * 70, flush=True)
        print("🔧 完整场景精修 (floor + wall + inter-object)", flush=True)
        print("=" * 70, flush=True)

    from src.sp_refinement import (
        refine_supported_by_floor_object,
        refine_embedded_in_wall_object,
        refine_attached_to_wall_object,
    )

    for category, category_instances in all_instances.items():
        relationship = _find_relationship_for_category(category, categories_and_relations)

        for instance_id, instance_info in enumerate(category_instances):
            if verbose:
                print(f"   {category}[{instance_id}]: {relationship}", flush=True)

            if relationship == "supported by floor" or relationship == "supported_by_floor":
                instance_info = refine_supported_by_floor_object(instance_info)

            elif relationship == "embedded in wall" or relationship == "embedded_in_wall":
                instance_info = refine_embedded_in_wall_object(instance_info, walls_info)

            elif relationship == "attached to wall" or relationship == "attached_to_wall":
                if extrinsics is not None and all_optimal_frame_ids is not None:
                    optimal_frame_id = all_optimal_frame_ids.get(category, [0])[
                        min(instance_id, len(all_optimal_frame_ids.get(category, [0])) - 1)
                    ]
                    extrinsic = extrinsics[optimal_frame_id]
                    camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
                else:
                    camera_pos = None
                instance_info = refine_attached_to_wall_object(instance_info, walls_info, camera_pos)

            category_instances[instance_id] = instance_info

    all_instances = refine_inter_object_relations(
        all_instances, categories_and_relations,
        walls_info=walls_info, verbose=verbose,
        vlm_checkpoint=vlm_checkpoint, scene_dir=scene_dir,
    )

    return all_instances


def _load_instances_from_glb(glb_path):
    """从 GLB 文件加载 all_instances (y-up → z-up)

    GLB文件是y-up格式，但SP精修函数需要z-up格式。
    加载时先做 y-up → z-up 反变换，精修完成后再转回y-up保存。
    """
    scene = trimesh.load(glb_path)

    y_up_to_z_up = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    all_instances = defaultdict(list)
    g = scene.graph

    for node_name in g.nodes_geometry:
        if node_name == 'world' or node_name.startswith('grid'):
            # 'world' 是根节点; 'grid_z0' 是 save_final_glb 画的地面网格线,
            # 当实例加载会被精化并再导出 (旧实现把它当成 "grid" 类实例)。
            continue
        parts = node_name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        category = parts[0]

        _, geom_name = g[node_name]
        if geom_name not in scene.geometry:
            continue

        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(y_up_to_z_up)
        all_instances[category].append({
            "original_mesh": mesh,
            "T": np.eye(4),
        })

    return dict(all_instances)


def _save_scene_to_glb(all_instances, output_glb):
    """将 all_instances 导出为 GLB 文件"""
    scene = trimesh.Scene()
    for category, category_instances in all_instances.items():
        for i, instance_info in enumerate(category_instances):
            mesh = instance_info['original_mesh'].copy()
            mesh.apply_transform(instance_info['T'])
            scene.add_geometry(mesh, node_name=f"{category}_{i}")
    scene.apply_transform(np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]))
    os.makedirs(os.path.dirname(output_glb) or '.', exist_ok=True)
    scene.export(output_glb)


def main():
    parser = argparse.ArgumentParser(description='物体间支撑关系空间位置精修')
    parser.add_argument('--input_glb', type=str, default=None,
                        help='输入 GLB 文件路径')
    parser.add_argument('--relations_json', type=str, required=True,
                        help='细化后的关系 JSON 路径')
    parser.add_argument('--output_glb', type=str, default=None,
                        help='输出 GLB 文件路径（默认为输入同目录下 *_refined.glb）')
    parser.add_argument('--instances_pkl', type=str, default=None,
                        help='[可选] all_instances pickle 路径')
    parser.add_argument('--vlm_checkpoint', type=str, default=None,
                        help=f'VLM模型路径 (默认: {DEFAULT_VLM_CHECKPOINT})')
    parser.add_argument('--scene_dir', type=str, default=None,
                        help='场景输出目录（含 optimal_frames/ 和 keyframes/，VLM用）')

    args = parser.parse_args()

    if not args.input_glb and not args.instances_pkl:
        parser.error("请提供 --input_glb 或 --instances_pkl 中的一个")

    print("=" * 70, flush=True)
    print("🚀 物体间支撑关系空间位置精修", flush=True)
    print("=" * 70, flush=True)

    with open(args.relations_json, 'r') as f:
        refined_relations = json.load(f)

    print(f"\n📋 关系列表:", flush=True)
    for name, rel in sorted(refined_relations.items()):
        print(f"   {name}: {rel}", flush=True)

    inter_object_pairs = [
        (name, rel.replace("supported by ", "").strip())
        for name, rel in refined_relations.items()
        if rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel
    ]

    if not inter_object_pairs:
        print(f"\n✅ 无物体间支撑关系，无需精修", flush=True)
        return

    import pickle

    # pkl 回写追踪: mainv3 的 pkl 是带包装的 blob
    # {'all_instances': ..., 'all_optimal_frame_ids': ..., ...}。
    # 记录来源与包装结构, 精修完成后把新位姿写回同一文件 —— 否则 pkl 停留在
    # 修复前位姿, 下游读 pkl 与 GLB 矛盾 (bug#12 的离线缺口)。
    _pkl_blob, _pkl_path = None, None

    if args.input_glb:
        print(f"\n📂 从 GLB 加载: {args.input_glb}", flush=True)
        all_instances = _load_instances_from_glb(args.input_glb)
    elif args.instances_pkl and os.path.exists(args.instances_pkl):
        print(f"\n📂 从 pickle 加载: {args.instances_pkl}", flush=True)
        with open(args.instances_pkl, 'rb') as f:
            _pkl_blob = pickle.load(f)
        _pkl_path = args.instances_pkl
        if isinstance(_pkl_blob, dict) and isinstance(_pkl_blob.get('all_instances'), dict):
            all_instances = _pkl_blob['all_instances']
        else:
            all_instances = _pkl_blob
    else:
        print(f"\n❌ 无有效输入文件", flush=True)
        return

    print(f"📦 加载 {sum(len(v) for v in all_instances.values())} 个物体, "
          f"{len(all_instances)} 个类别", flush=True)

    scene_dir = args.scene_dir
    if scene_dir is None and args.input_glb:
        scene_dir = os.path.dirname(args.input_glb)

    all_instances = refine_inter_object_relations(
        all_instances, refined_relations,
        vlm_checkpoint=args.vlm_checkpoint,
        scene_dir=scene_dir,
    )

    if args.input_glb:
        output_glb = args.output_glb or f"{os.path.splitext(args.input_glb)[0]}_refined.glb"
    else:
        output_glb = args.output_glb or os.path.join(
            os.path.dirname(args.relations_json), "refined_scene.glb"
        )

    _save_scene_to_glb(all_instances, output_glb)
    print(f"\n💾 保存精修后的场景: {output_glb}", flush=True)

    # ── pkl 回写 (bug#12 离线缺口) ──
    if _pkl_path is not None:
        try:
            if isinstance(_pkl_blob, dict) and isinstance(_pkl_blob.get('all_instances'), dict):
                _pkl_blob['all_instances'] = all_instances
                pickle.dump(_pkl_blob, open(_pkl_path, 'wb'))
            else:
                pickle.dump(all_instances, open(_pkl_path, 'wb'))
            print(f"💾 all_instances.pkl 已回写 (与精修后位姿一致): {_pkl_path}", flush=True)
        except Exception as _e:
            print(f"⚠️ pkl 回写失败 (GLB 已保存): {_e}", flush=True)

    print(f"\n✅ 完成!", flush=True)


if __name__ == '__main__':
    main()
