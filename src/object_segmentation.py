import numpy as np
from PIL import Image
import torch

def segment_wall_and_floor(images, sam3_image_model):
    """
    Use SAM3 to segment wall and floor from the input images.

    增强版: 使用多组 prompt 变体来覆盖不同场景类型 (卧室/客厅/办公室等).
    原代码只用 "floor" 和 "single wall", 对床铺/桌面等非标准地面场景效果差.

    Args:
        images: numpy array of shape (S, H, W, 3)
        sam3_image_model: The loaded SAM3 image model object.
    Returns:
        wall_masks: A list of dictionaries containing 'frame_id' and 'mask'.
        floor_masks: A list of dictionaries containing 'frame_id' and 'mask'.
    """
    # 多组 prompt 变体, 覆盖不同场景类型的"地面"和"墙面"
    floor_prompts = [
        "floor",           # 通用地面
        "ground",          # 室外/室内地面
        "carpet",          # 地毯 (卧室常见)
        "bed",             # 床铺 (卧室大平面)
        "table",           # 桌面 (桌面场景)
        "desk",            # 书桌
        "horizontal surface",  # 水平面 (通用)
        "tabletop",        # 桌子表面
        "counter",         # 柜台
        "shelf",           # 架子
    ]
    wall_prompts = [
        "single wall",     # 单面墙
        "wall",            # 墙 (通用)
        "window",          # 窗户 (常在大面积墙上)
        "curtain",         # 窗帘 (大面积垂直面)
        "door",            # 门 (大面积垂直面)
        "bookshelf",       # 书架 (大面积垂直面)
        "cabinet",         # 柜子 (大面积垂直面)
        "vertical surface",    # 垂直面 (通用)
    ]

    wall_masks = []
    floor_masks = []
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam3_image_model.set_image(image)

        # 尝试多个 floor prompt
        for floor_prompt in floor_prompts:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sam3_image_model.reset_all_prompts(inference_state)
                inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt=floor_prompt)
                masks = inference_state['masks'].cpu().numpy()
            for mask in masks:
                mask_arr = np.atleast_2d(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                if np.sum(mask_arr) > 500:
                    floor_masks.append({
                        'frame_id': i,
                        'mask': mask_arr,
                        'prompt': floor_prompt,
                    })

        # 尝试多个 wall prompt
        for wall_prompt in wall_prompts:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sam3_image_model.reset_all_prompts(inference_state)
                inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt=wall_prompt)
                masks = inference_state['masks'].cpu().numpy()
            for mask in masks:
                mask_arr = np.atleast_2d(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                if np.sum(mask_arr) > 500:
                    wall_masks.append({
                        'frame_id': i,
                        'mask': mask_arr,
                        'prompt': wall_prompt,
                    })

    return wall_masks, floor_masks


def _deduplicate_masks(masks, iou_threshold=0.5):
    """去重: 合并 IoU > threshold 的 mask"""
    if len(masks) <= 1:
        return masks
    unique = []
    for mask in masks:
        is_dup = False
        for existing in unique:
            intersection = np.sum(mask & existing)
            union = np.sum(mask | existing)
            if union > 0 and intersection / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(mask)
    return unique


def segment_large_flat_surfaces(images, sam3_image_model, vlm_model=None, vlm_processor=None,
                                 min_mask_area=5000, sample_frame_for_vlm=0):
    """
    用宽泛文本提示分割大平面物体 (阶段3 fallback).

    当 "floor" 文本提示失败时, 用多个宽泛提示词尝试分割大平面.
    增强版: 如果提供了 VLM 模型, 先用 VLM 识别场景类型, 再用针对性的 prompt.

    Args:
        images: numpy array of shape (S, H, W, 3)
        sam3_image_model: SAM3 image model
        vlm_model: VLM 模型 (可选, 用于场景类型识别)
        vlm_processor: VLM processor (可选)
        min_mask_area: 最小 mask 面积 (像素), 过滤小 mask (默认5000)
        sample_frame_for_vlm: 用哪一帧做 VLM 场景类型识别 (默认0)
    Returns:
        large_plane_masks: list of dicts with 'frame_id' and 'mask'
    """
    large_plane_masks = []

    # 如果有 VLM, 先用它识别场景类型
    scene_type = "unknown"
    if vlm_model is not None and vlm_processor is not None and 0 <= sample_frame_for_vlm < len(images):
        print(f"   VLM 场景类型识别中...", flush=True)
        scene_type = detect_scene_type_with_vlm(
            images[sample_frame_for_vlm], vlm_model, vlm_processor
        )
        print(f"   识别场景类型: {scene_type}", flush=True)
        scene_floor_prompts, scene_wall_prompts = get_scene_specific_prompts(scene_type)
        prompts = scene_floor_prompts  # 大平面主要找地面类
    else:
        prompts = ["flat surface", "ground", "horizontal surface"]

    for i, image in enumerate(images):
        image = Image.fromarray(image)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam3_image_model.set_image(image)

        frame_masks = []
        for prompt in prompts:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sam3_image_model.reset_all_prompts(inference_state)
                inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt=prompt)
                masks = inference_state['masks'].cpu().numpy()
            for mask in masks:
                mask_arr = np.atleast_2d(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                if np.sum(mask_arr) > min_mask_area:
                    frame_masks.append(mask_arr)

        unique_masks = _deduplicate_masks(frame_masks)
        for mask_arr in unique_masks:
            large_plane_masks.append({
                'frame_id': i,
                'mask': mask_arr
            })

        sam3_image_model.reset_all_prompts(inference_state)

    return large_plane_masks


def detect_scene_type_with_vlm(image, vlm_model, vlm_processor, max_new_tokens=256):
    """
    用 VLM 识别场景类型, 用于指导 SAM3 prompt 选择.

    通过分析单张图像判断场景属于哪类房间/环境, 返回场景类型标签.
    场景类型与 SAM3 prompt 的映射:
      bedroom → bed, carpet, floor, shelf
      living_room → floor, sofa, table, carpet
      office → desk, table, cabinet, bookshelf
      kitchen → counter, cabinet, table
      bathroom → floor, wall
      outdoor → ground, floor, wall

    Args:
        image: PIL.Image 或 numpy array (H, W, 3) RGB
        vlm_model: 已加载的 VLM 模型
        vlm_processor: 已加载的 VLM processor
        max_new_tokens: VLM 最大输出 token 数

    Returns:
        scene_type: str, 场景类型标签 (bedroom/living_room/office/kitchen/bathroom/outdoor/unknown)
    """
    try:
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)

        prompt = (
            "What type of room or scene is this? "
            "Choose ONE of these categories: bedroom, living_room, office, kitchen, bathroom, outdoor, unknown. "
            "Return ONLY the category name, no explanation. "
            "Look at the furniture, layout, and overall environment to decide."
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]

        try:
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False
            )
        except TypeError:
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        inputs = inputs.to(vlm_model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": vlm_processor.tokenizer.eos_token_id,
            "do_sample": False,
        }

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip().lower()

        # 提取场景类型关键词
        valid_types = ["bedroom", "living_room", "office", "kitchen", "bathroom", "outdoor"]
        for scene_type in valid_types:
            if scene_type in output_text:
                return scene_type

        return "unknown"
    except Exception as e:
        print(f"   ⚠️ VLM 场景类型识别失败: {e}", flush=True)
        return "unknown"


def get_scene_specific_prompts(scene_type):
    """
    根据场景类型返回针对性的 SAM3 prompt 列表.

    Args:
        scene_type: 场景类型标签

    Returns:
        (floor_prompts, wall_prompts): 针对该场景优化的 prompt 列表
    """
    prompt_map = {
        "bedroom": {
            "floor": ["bed", "carpet", "floor", "rug", "wooden floor", "blanket"],
            "wall": ["curtain", "wall", "window", "wardrobe", "dresser", "nightstand"],
        },
        "living_room": {
            "floor": ["floor", "carpet", "rug", "tile floor"],
            "wall": ["wall", "window", "curtain", "sofa", "tv stand", "bookshelf"],
        },
        "office": {
            "floor": ["floor", "carpet", "tile"],
            "wall": ["wall", "whiteboard", "bookshelf", "cabinet", "desk"],
        },
        "kitchen": {
            "floor": ["floor", "tile"],
            "wall": ["wall", "cabinet", "counter", "kitchen counter"],
        },
        "bathroom": {
            "floor": ["floor", "tile floor"],
            "wall": ["wall", "tile", "mirror", "cabinet"],
        },
        "outdoor": {
            "floor": ["ground", "floor", "path", "grass", "pavement"],
            "wall": ["wall", "building", "fence", "tree trunk"],
        },
        "unknown": {
            "floor": ["floor", "ground", "carpet", "bed", "table", "horizontal surface", "tabletop", "counter"],
            "wall": ["wall", "single wall", "window", "curtain", "vertical surface"],
        },
    }

    prompts = prompt_map.get(scene_type, prompt_map["unknown"])
    return prompts["floor"], prompts["wall"]


def detect_floor_reference_points_with_vlm(image, vlm_model, vlm_processor,
                                           num_points=4, max_new_tokens=256):
    """
    用VLM识别图像中地面的代表性参考点.

    SAM3的Sam3Processor没有直接的point prompt API, 但支持box prompt.
    因此先用VLM生成地面参考点, 再围绕每个点构造小box作为SAM3输入.

    Args:
        image: PIL.Image 或 numpy array (H, W, 3) RGB
        vlm_model: 已加载的VLM模型
        vlm_processor: 已加载的VLM processor
        num_points: 期望返回的参考点数量 (默认4)
        max_new_tokens: VLM最大输出token数

    Returns:
        points: list of (x, y) 归一化坐标 [0, 1], 或空列表
    """
    try:
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)

        prompt = (
            f"Identify {num_points} representative points on the floor/ground in this image. "
            "Return ONLY a JSON list of [x, y] normalized coordinates (0-1). "
            "Example: [[0.25, 0.75], [0.50, 0.80], [0.75, 0.75], [0.40, 0.85]]. "
            "Do not include any explanation."
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]

        try:
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False
            )
        except TypeError:
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        inputs = inputs.to(vlm_model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": vlm_processor.tokenizer.eos_token_id,
            "do_sample": False,
        }

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 解析JSON
        import json
        import re
        # 尝试直接解析
        try:
            points = json.loads(output_text.strip())
            if isinstance(points, list) and len(points) > 0:
                normalized = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        x, y = float(p[0]), float(p[1])
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            normalized.append((x, y))
                return normalized[:num_points]
        except Exception:
            pass

        # 尝试从文本中提取JSON数组
        matches = re.findall(r'\[\s*\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]\s*(,\s*\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]\s*)*\]', output_text)
        for match in matches:
            try:
                points = json.loads(match)
                normalized = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        x, y = float(p[0]), float(p[1])
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            normalized.append((x, y))
                if normalized:
                    return normalized[:num_points]
            except Exception:
                continue

        return []
    except Exception as e:
        print(f"   ⚠️ VLM地面参考点检测失败: {e}", flush=True)
        return []


def segment_floor_with_box_prompts(images, sam3_image_model, reference_points_per_frame,
                                   box_size=0.05, min_mask_area=500):
    """
    用VLM提供的地面参考点, 通过SAM3 box prompt生成floor mask.

    SAM3的Sam3Processor不支持point prompt, 但支持add_geometric_prompt(box).
    这里用围绕参考点的小box近似point prompt.

    Args:
        images: numpy array (S, H, W, 3) RGB
        sam3_image_model: SAM3 image processor
        reference_points_per_frame: dict {frame_id: list of (x, y) normalized coords}
        box_size: 小box的宽高 (归一化坐标, 默认0.05)
        min_mask_area: 最小mask面积 (像素)

    Returns:
        floor_masks: list of dicts with 'frame_id' and 'mask'
    """
    floor_masks = []

    for frame_id, points in reference_points_per_frame.items():
        if frame_id >= len(images):
            continue
        image = Image.fromarray(images[frame_id])

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam3_image_model.set_image(image)

        frame_masks = []
        for (x, y) in points:
            # 围绕点构造小box [center_x, center_y, width, height], 归一化
            box = [x, y, box_size, box_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sam3_image_model.reset_all_prompts(inference_state)
                inference_state = sam3_image_model.add_geometric_prompt(
                    box=box, label=True, state=inference_state
                )
            masks = inference_state['masks'].cpu().numpy()
            for mask in masks:
                mask_arr = np.atleast_2d(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                if np.sum(mask_arr) > min_mask_area:
                    frame_masks.append(mask_arr)

        sam3_image_model.reset_all_prompts(inference_state)

        unique_masks = _deduplicate_masks(frame_masks)
        for mask_arr in unique_masks:
            floor_masks.append({
                'frame_id': frame_id,
                'mask': mask_arr
            })

    return floor_masks


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def _box_fallback_segment(category, video_predictor, session_id, box_provider,
                          verbose=True, max_boxes=3, frame_hw=None, world_points=None):
    """文本提示切不出时的 box 兜底: 用外部给的框重新提示 SAM3 视频跟踪。

    设计约束 (与 mainv3 的"只补不换"约定一致):
      - 只在文本提示返回 0 结果时才调用, 已经能切出的类别完全不受影响
      - box_provider 抛异常 / 返回空 / 超时, 一律静默返回 [], 退化成原行为
      - 最多试 max_boxes 个框, 避免单类别拖慢整条管线

    Returns:
        与 segment_and_track 同结构; 兜底失败时为 []
    """
    if box_provider is None:
        return []
    try:
        boxes01 = box_provider(category) or []
    except Exception as e:
        if verbose:
            print(f"    [box兜底] {category}: 取框失败 ({type(e).__name__}: {e}), "
                  f"跳过", flush=True)
        return []
    if not boxes01:
        return []

    # SAM3 视频模型的 bounding_boxes 用像素坐标 [x1, y1, x2, y2]
    try:
        frame0 = video_predictor._get_session(session_id)["state"].get("images")
        if frame0 is not None and len(frame0) > 0:
            import numpy as _np
            _a = _np.asarray(frame0[0])
            H, W = _a.shape[-3], _a.shape[-2]
        else:
            H = W = None
    except Exception:
        H = W = None
    # H/W 取不到时 (session 状态异常/测试替身) 必须给默认值, 不能留 None:
    # 后续 \"if H and W and max(...) <= 1.5\" 会因 H 为 None 而跳过归一化→像素
    # 的转换, 框坐标停在 0.3/0.55 这种小数域, int() 后全变 0, 框退化成空,
    # 最终 continue 掉, 矩形 mask 兜底静默失效 (实测: 日志打了\"直接用矩形
    # mask\"但返回 0 个实例)。
    if not H or not W:
        H, W = 480, 854

    best = []
    for bi, b in enumerate(boxes01[:max_boxes]):
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        if H and W and max(x1, y1, x2, y2) <= 1.5:   # 归一化 → 像素
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H
        if x2 <= x1 or y2 <= y1:
            continue
        # 播种帧: 不只第 0 帧。
        # 只播第 0 帧的问题: 物体在开头可能很小/被遮挡/模糊, SAM3 视频模型从
        # 第 0 帧播种后一路跟丢。实测 weixin_v9: 7 个物体类别全部文本切不出,
        # box 兜底取到了框但从 frame 0 播种仍然 0 目标。
        # 改为在"VLM 确实看到过该物体"的关键帧上播种 (见 _seed_frames),
        # 取覆盖帧数最多的那次。没有候选帧时才退回第 0 帧。
        _seed_cands = [s for s in (getattr(segment_and_track, '_seed_frames',
                                           None) or []) if s > 0]
        _seed_cands = _seed_cands[:4] or [0]
        _seg_outs = []            # 所有播种帧产出的 propagate 结果 (全部保留)
        _best_ids = set()         # 仅用于日志统计
        for _sf in _seed_cands:
            try:
                video_predictor.handle_request(
                    request=dict(type="reset_session", session_id=session_id))
                video_predictor.handle_request(request=dict(
                    type="add_prompt", session_id=session_id, frame_index=_sf,
                    bounding_boxes=[[x1, y1, x2, y2]], bounding_box_labels=[1]))
                outputs = propagate_in_video(video_predictor, session_id)
            except Exception as e:
                if verbose:
                    print(f"    [box兜底] {category} 框{bi} 帧{_sf}: SAM3 报错 "
                          f"({type(e).__name__}: {str(e)[:50]}), 换下一帧", flush=True)
                continue
            if not outputs:
                continue
            _ids = set()
            for fr in outputs.keys():
                _ids.update(outputs[fr].get('out_obj_ids', []))
            if not _ids:
                continue
            if verbose and len(_seed_cands) > 1:
                print(f"    [box兜底] {category} 框{bi} 帧{_sf} 播种 → "
                      f"{len(_ids)} 个目标", flush=True)
            # ── 合并而非择优 ──
            # 原来每个框只保留"覆盖帧最多"的那一次播种。问题是: 小物体在不同
            # 播种帧上跟出的是**不同时间段**的片段 —— 实测 weixin 的 earbud case,
            # 帧 31 播种只覆盖 29~38, 帧 35 播种只覆盖 30~38, 而第一次跑
            # (wx4) 确实从帧 15 就开始跟。取"最多"的那一次就把其它段丢了,
            # 于是同一物体时有时无 (wx4 有 mask, wx5 完全没有) —— 分割结果
            # 不确定, 直接导致交互检测与首帧选择反复失败。
            # 现在把所有播种帧产出的轨迹全部合并, 交给下游 self_category_
            # deduplicate 去重 (它按点云重叠判定, 同一物体的不同时间段会
            # 被正确合并, 不会产生重复实例)。
            _seg_outs.append(outputs)
            # 提前停止: 覆盖率已过半就不必再试其余播种帧。
            # 每次播种都要跑一遍完整 propagate (80 帧约 14s), 4 个候选帧就是
            # ~56s/框; Stage2 有十几个框时这是主要耗时来源。
            _tot_frames = max(outputs.keys()) + 1 if outputs else 0
            if _tot_frames and len(_ids) > 0.5 * _tot_frames:
                break

        if not _seg_outs:
            # ── 最后兜底: 把框当矩形 mask 直接用 ──
            # 为什么需要: box provider (6.8) 给出的框是准的 (实测 weixin_v6:
            # 框 [0.305,0.555,0.445,0.645] 精确落在黑色耳机盒上, 红框画在帧上
            # 与物体完全重合), 但 SAM3 视频模型即使拿到精确框也可能产出 0 个
            # 目标 —— 框只有 67x77 像素, 小于视频模型的跟踪门限。
            # 此时框本身已经携带了"物体在哪、有多大"的全部信息, 直接栅格化成
            # 矩形 mask 即可, 不需要再依赖 SAM3。代价是 mask 是矩形而非精确
            # 轮廓, 但比"物体整个丢失"好得多 —— 下游 SAM3D 用的是同一帧的
            # 图像, mesh 形状由 SAM3D 决定, 不受 mask 形状影响。
            if verbose:
                print(f"    [box兜底] {category} 框{bi}: SAM3 有框也切不出 → "
                      f"直接用矩形 mask (位置/尺寸来自该框)", flush=True)
            # mask 必须是 VGGT 帧尺寸 (world_points 的 H,W), 不是 session
            # 图像尺寸 —— SAM3 原生 mask 就是 (688,384), 与 world_points 对齐;
            # 用错尺寸会被下游按形状不匹配直接丢掉 (实测 wx7: 造了 (480,854)
            # 的矩形 mask, 结果 earbud case 实例数仍是 0)。
            if frame_hw:
                _H, _W = int(frame_hw[0]), int(frame_hw[1])
            elif H and W:
                _H, _W = int(H), int(W)
            else:
                _H = _W = None
            if not _H or not _W:
                continue
            # 框 -> VGGT 帧 mask 的坐标映射。难点: VGGT 帧与 session 图像不只
            # 是缩放, 还差 90° 旋转 (实测 weixin: 关键帧 480x854 横, VGGT 的
            # color/world_points 是 688x384 竖)。直接按比例缩放会把框映射到
            # 完全错误的位置, mask 落在空白上, 反投影点云全无效, 实例被当空
            # 实例丢掉。做法: 4 个朝向都生成矩形 mask, 用"反投影出的有效 VGGT
            # 点数"自校验 —— 正确朝向落在真实物体上(有深度), 错误朝向落在
            # 背景/画面外。可测量, 不依赖对旋转方向的假设。
            _ow, _oh = float(W or 854), float(H or 480)
            _n = (x1 / _ow, y1 / _oh, x2 / _ow, y2 / _oh)
            _cands = {
                'r0':   (_n[0] * _W, _n[1] * _H, _n[2] * _W, _n[3] * _H),
                'r90':  ((1 - _n[3]) * _W, _n[0] * _H,
                         (1 - _n[1]) * _W, _n[2] * _H),
                'r180': ((1 - _n[2]) * _W, (1 - _n[3]) * _H,
                         (1 - _n[0]) * _W, (1 - _n[1]) * _H),
                'r270': (_n[1] * _W, (1 - _n[2]) * _H,
                         _n[3] * _W, (1 - _n[0]) * _H),
            }
            _best_mask, _best_score = None, -1.0
            _wp = np.asarray(world_points) if world_points is not None else None
            for _a, _b, _c, _d in _cands.values():
                _bx1, _by1 = max(0, int(min(_a, _c))), max(0, int(min(_b, _d)))
                _bx2, _by2 = min(_W, int(max(_a, _c))), min(_H, int(max(_b, _d)))
                if _bx2 <= _bx1 or _by2 <= _by1:
                    continue
                _m = np.zeros((_H, _W), dtype=bool)
                _m[_by1:_by2, _bx1:_bx2] = True
                # 判据: 深度起伏 (z 的标准差)。正确朝向落在凸出桌面的物体上,
                # 深度有明显起伏; 错误朝向落在平坦桌面上, 深度几乎不变。
                # 实测 weixin: r0=0.0068 / r270=0.0038 / r180=0.0025 / r90=0.0011,
                # 正确朝向 (r0) 的起伏是错误朝向的 6 倍, 可稳定区分。
                # (不能用"有效点比例": VGGT 填满全帧, 四个朝向都是 1.000。)
                if _wp is not None and _wp.ndim >= 3 and _wp.shape[0] > 0:
                    _p = _wp[0][_m]
                    _p = _p[np.isfinite(_p).all(axis=-1)]
                    _score = float(_p[:, 2].std()) if len(_p) > 4 else -1.0
                else:
                    _score = float(_m.sum())
                if _score > _best_score:
                    _best_score, _best_mask = _score, _m
            if _best_mask is None:
                continue
            # ── 矩形收缩 ──
            # VLM 给的框是"外接框", 比物体本身大一圈; 直接栅格化会把框内的
            # **背景点**也包进来。小物体坐在大平面上时后果严重: 实测 wx7 的
            # earbud case 兜底矩形面积 3762px (真实物体约 1500px), 框内包含
            # 桌面点 → 与 table 点云重叠率达 0.604, 超过官方跨类 OR@0.5 阈值
            # → 耳机盒被并进桌子, 整个物体消失。
            # 这里把矩形向中心收缩 25% (保留中心 75%), 排除框边缘的背景。
            _ys, _xs = np.where(_best_mask)
            if len(_ys) > 8:
                _cy0, _cy1 = _ys.min(), _ys.max()
                _cx0, _cx1 = _xs.min(), _xs.max()
                _ky = int((_cy1 - _cy0) * 0.125)
                _kx = int((_cx1 - _cx0) * 0.125)
                _tight = np.zeros_like(_best_mask)
                _tight[_cy0 + _ky:_cy1 - _ky + 1, _cx0 + _kx:_cx1 - _kx + 1] = True
                if _tight.sum() > 8:
                    _best_mask = _tight
            _mask = _best_mask
            # 注意结构: _split_instances 返回的是"实例列表", 每项是"帧字典列表"。
            # 这里必须同样返回 [[{frame_id,mask}, ...]] (一个实例, 多帧),
            # 不能返回扁平的 [{frame_id,mask}, ...] —— 那样下游 self_category_
            # deduplicate 会把每个帧字典当成一个实例迭代, 直接 TypeError
            # (实测 wx9: 矩形 mask 造对了但 earbud case 实例数仍是 0)。
            res = [[{'frame_id': int(_sf), 'mask': _mask.copy()}
                    for _sf in _seed_cands]]
            if res:
                best.extend(res)
            continue

        if verbose:
            print(f"    [box兜底] {category} 框{bi}: 文本切不出 → box 切出 "
                  f"{len(_seg_outs)} 段轨迹, 合并  ✅ (播种帧 {_seed_cands})",
                  flush=True)
        # 复用主流程的实例切分逻辑: 多段播种结果各切各的, 合并后由去重收口
        res = []
        for _o in _seg_outs:
            res.extend(_split_instances(_o))
        if res:
            best.extend(res)
    return best


def _split_instances(outputs_per_frame):
    """把 propagate 输出按 obj_id 切分成逐实例轨迹 (与 segment_and_track 主体同逻辑)。"""
    all_obj_ids = set()
    for frame_idx in outputs_per_frame.keys():
        all_obj_ids.update(outputs_per_frame[frame_idx]['out_obj_ids'])
    if not all_obj_ids:
        return []
    final_results = []
    for obj_id in sorted(all_obj_ids):
        traj = []
        for frame_idx in sorted(outputs_per_frame.keys()):
            fr = outputs_per_frame[frame_idx]
            if obj_id not in fr['out_obj_ids']:
                continue
            pos = list(fr['out_obj_ids']).index(obj_id)
            mask = fr['out_binary_masks'][pos]
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            import numpy as _np
            mask = _np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[0]
            if mask.sum() > 0:
                traj.append({'frame_id': int(frame_idx), 'mask': mask.astype(bool)})
        if traj:
            final_results.append(traj)
    return final_results


def segment_and_track(category, video_predictor, session_id, box_provider=None,
                      verbose=True):
    '''
    Segment raw instance masks using sam3 video tracking
    Args:
        category: A str means the category to segment
        video_predictor: the loaded sam3 video model
        session_id: the session with loaded video frames corresponding to the video_predictor
        box_provider: 可选回调 (category) -> list of (x1, y1, x2, y2) 归一化 0~1。
            仅当文本提示切不出任何 mask 时才会调用它做 box 兜底。
            实测 (basic_pick_place_141): "mouse"/"soap" 用文本提示 SAM3 视频
            跟踪产出 0 个 mask, 但用 6.8 出的框作为 box 提示能切出 8~11 个。
            全库统计: 30 个类别属于"文本切不出", 是物体丢失的最大单一来源。
            传 None 则完全走原逻辑 (零风险)。
        verbose: 是否打印兜底过程
    Returns:
        A list of list, each list represents a segmented instance and is composed
        of dicts with keys frame_id and mask (binary mask of the instance in that frame)
        [
            [
                {
                    'frame_id': int,
                    'mask': numpy array of shape (H, W) with binary values (True or False)
                },
                ...
            ],
            ...
        ]
    '''
    # Reset session and add text prompt for the category to segment
    _ = video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    video_predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=category))
    outputs_per_frame = propagate_in_video(video_predictor, session_id)
    if not outputs_per_frame:
        return _box_fallback_segment(category, video_predictor, session_id,
                                     box_provider, verbose,
                                     frame_hw=getattr(segment_and_track, '_frame_hw', None),
                                     world_points=getattr(segment_and_track, '_world_points', None))

    # Collect all object IDs across frames, discontinuous segments will be split into different instances.
    all_obj_ids = set()
    for frame_idx in outputs_per_frame.keys():
        all_obj_ids.update(outputs_per_frame[frame_idx]['out_obj_ids'])

    # ── 播种帧重试 ──
    # 为什么需要: SAM3 视频跟踪从"播种帧"开始传播。播种帧不好时, 小物体
    # (实测 weixin 的耳机盒只有 40x50 像素, 占画面 0.4%) 早期会被跟丢,
    # 传播结果只覆盖后段。实测 button 只在帧 64~79 产出 mask, 而物体从第 0
    # 帧就清晰可见 —— 首帧取"首次可见"就变成 64, 拿到的是手持高度而非桌上
    # 静止位置。
    # 做法: 首轮帧覆盖率偏低时, 换候选播种帧重试, 取覆盖帧最多的那次。
    # 候选帧由调用方通过 segment_and_track._seed_frames 提供 (通常是 VLM
    # 检出该类别的关键帧); 未设置时完全走原行为 (只在 frame 0 播种)。
    if getattr(segment_and_track, '_seed_frames', None) and outputs_per_frame:
        _total_frames = max(outputs_per_frame.keys()) + 1
        _covered = sum(1 for f in outputs_per_frame.keys()
                       if len(outputs_per_frame[f]['out_obj_ids']) > 0)
        _min_cov = getattr(segment_and_track, '_min_coverage', 0.25)
        if _total_frames > 0 and _covered < _min_cov * _total_frames:
            _best_out, _best_cov = outputs_per_frame, _covered
            for _seed in segment_and_track._seed_frames:
                if not _seed or _seed <= 0:
                    continue
                try:
                    video_predictor.handle_request(
                        request=dict(type="reset_session", session_id=session_id))
                    video_predictor.handle_request(request=dict(
                        type="add_prompt", session_id=session_id,
                        frame_index=int(_seed), text=category))
                    _out = propagate_in_video(video_predictor, session_id)
                except Exception:
                    continue
                if not _out:
                    continue
                _cov = sum(1 for f in _out.keys()
                           if len(_out[f]['out_obj_ids']) > 0)
                if verbose:
                    print(f"    [多播种] {category} 帧 {_seed} 播种 → "
                          f"覆盖 {_cov}/{_total_frames} 帧", flush=True)
                if _cov > _best_cov:
                    _best_out, _best_cov = _out, _cov
            if _best_cov > _covered:
                outputs_per_frame = _best_out
                all_obj_ids = set()
                for frame_idx in outputs_per_frame.keys():
                    all_obj_ids.update(outputs_per_frame[frame_idx]['out_obj_ids'])
                if verbose:
                    print(f"    [多播种] {category} 采用覆盖最多的播种: "
                          f"{_best_cov}/{_total_frames} 帧 (原 {_covered})",
                          flush=True)

        # 多播种后覆盖率仍然低 → 再试 box 兜底, 合并后直接返回。
        # 原来 box 兜底只在"0 个目标"时触发, 但实测更常见的是"跟出了结果但只
        # 覆盖后段" (button 只有帧 64~79) —— 那种情况文本提示并非完全失败,
        # 却同样导致首帧取错。box 提示是几何约束, 对小物体比文本词更硬。
        _cov2 = sum(1 for f in outputs_per_frame.keys()
                    if len(outputs_per_frame[f]['out_obj_ids']) > 0)
        _tot2 = max(outputs_per_frame.keys()) + 1 if outputs_per_frame else 0
        # "出现太晚" 也是触发条件。
        # 实测 weixin 的耳机盒: 文本跟踪只覆盖帧 51~79 (共 80 帧), 物体第一次
        # 出现在 64% 处 —— 而那一帧它已经在手里了。抓取前的帧完全缺失, 于是
        # "从桌上到手"的位移在数据里不存在, 交互检测必然判不出 (0 个交互)。
        # 覆盖率 36% 高于 25% 阈值, 所以旧的"低覆盖"条件不会触发兜底。
        # 这里补一条: 首次出现晚于总帧数 25% 就重试, 用 box 从第 0 帧重新播种,
        # 把跟踪往前延伸。
        _first_vis = min((f for f in outputs_per_frame
                          if len(outputs_per_frame[f]['out_obj_ids']) > 0),
                         default=_tot2)
        _late = _tot2 > 0 and _first_vis > 0.25 * _tot2
        if _tot2 > 0 and (_cov2 < _min_cov * _tot2 or _late):
            _extra = _box_fallback_segment(category, video_predictor, session_id,
                                           box_provider, verbose,
                                     frame_hw=getattr(segment_and_track, '_frame_hw', None),
                                     world_points=getattr(segment_and_track, '_world_points', None))
            if _extra:
                _text_traj = _split_instances(outputs_per_frame)
                _merged = _text_traj + list(_extra)
                if verbose:
                    print(f"    [低覆盖兜底] {category} 文本 {_cov2}/{_tot2} 帧 "
                          f"(首见 {_first_vis}, {'出现太晚' if _late else '覆盖不足'}), "
                          f"box 兜底补 {len(_extra)} 条 → 合并 {len(_merged)} 条",
                          flush=True)
                return _merged

    if len(all_obj_ids) == 0:
        print(f'No object detected for {category}.')
        # 必须在这里也触发 box 兜底。
        # 之前的兜底只门禁在 `if not outputs_per_frame` 上, 但实测真正的
        # "没检出"不是 propagate 无输出, 而是: outputs_per_frame 非空、
        # 逐帧的 out_obj_ids 加起来却是空 —— 也就是 propagate 跑了但一个目标
        # 都没跟到。那种情况下函数走到这里直接 return [], 兜底永远不触发。
        # 实测 weixin_195833_224 (竖屏 480x854, 80 帧): 4 个类别全部走这条
        # "No object detected" 分支, box 兜底一次都没被调用 —— 整场景 0 实例。
        return _box_fallback_segment(category, video_predictor, session_id,
                                     box_provider, verbose,
                                     frame_hw=getattr(segment_and_track, '_frame_hw', None),
                                     world_points=getattr(segment_and_track, '_world_points', None))
    final_results = []
    sorted_obj_ids = sorted(list(all_obj_ids))
    for obj_id in sorted_obj_ids:
        raw_frame_ids = sorted([
            id for id in outputs_per_frame.keys() 
            if obj_id in outputs_per_frame[id]['out_obj_ids']
        ])
        
        if not raw_frame_ids:
            continue
        segments = []
        if len(raw_frame_ids) > 0:
            current_segment = [raw_frame_ids[0]]
            for i in range(1, len(raw_frame_ids)):
                if raw_frame_ids[i] == raw_frame_ids[i-1] + 1:
                    current_segment.append(raw_frame_ids[i])
                else:
                    segments.append(current_segment)
                    current_segment = [raw_frame_ids[i]]
            segments.append(current_segment)
        for frame_ids in segments:
            instance_track = []
            
            for frame_id in frame_ids:
                obj_indices = np.where(outputs_per_frame[frame_id]['out_obj_ids'] == obj_id)[0]
                
                if len(obj_indices) > 0:
                    idx = obj_indices[0]
                    raw_mask = outputs_per_frame[frame_id]['out_binary_masks'][idx].squeeze()
                    binary_mask = raw_mask > 0
                    
                    instance_track.append({
                        'frame_id': frame_id,
                        'mask': binary_mask
                    })
            if instance_track:
                final_results.append(instance_track)

    return final_results