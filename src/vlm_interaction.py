#!/usr/bin/env python3
"""VLM 先行判断"哪个物体被操作" —— 交互检测的语义层。

为什么需要 (与 src/pick_place_detector.py 的几何判据互补):
  几何判据靠点云位移/重叠率, 对两类情况会失效:
    1. 大静态物体被相机移动欺骗 (桌子被判成"在动") —— 已用质心门槛挡住,
       但门槛是魔法数字, 换个相机运动幅度就可能失效
    2. 小物体轨迹太短 (mask 只覆盖抓取后的帧), 位移统计不充分
  VLM 直接看画面回答"手拿起了什么", 是语义判断, 不依赖阈值。
  实测 test_vlm_pick_place.py: VLM 与几何判据在 bottle 上给出一致结论。

分工:
  VLM      → 哪个**类别**被操作, 大致在哪些帧 (语义, 粗)
  几何     → 哪个**实例**, 精确 pick/place 帧 (精确, 细)
  二者一致 → 采用; 只有一方有结论 → 用有结论的那方并记录来源

任何失败 (模型加载失败/超时/解析失败) 都返回 [], 由调用方退化成纯几何判据。
"""
import json
import os
import re

PROMPT = """You are given {n} frames sampled in temporal order from a first-person video. Each frame has its frame index drawn in the top-left corner (0 to {n}).

Task: For EACH object that is picked up and put down by the hand in the video, report:
1. The object name (short English name)
2. pick_frame: the frame index (in the ORIGINAL video timeline, estimated by interpolating between the sampled frame indices) where the hand starts lifting the object off its resting surface
3. place_frame: the frame index where the object is put down onto a surface and released

Output ONLY a JSON array, no other text:
[
  {{"object": "bottle", "pick_frame": 40, "place_frame": 90}}
]

If no object is manipulated, output []"""

# 与 Stage1 一致: 这些词不是可操作物体, VLM 有时会顺手报出来
_IGNORE = {"hand", "hands", "arm", "arms", "human arm", "finger", "fingers",
           "body", "person", "people", "wall", "floor", "ceiling", "table",
           "desk", "counter", "countertop", "wall outlet", "socket", "door"}


DEFAULT_CKPT = "/mnt/data/lza/models/Qwen3.5-9B"
# 全局 VLM 缓存: 加载一次, Stage3.7 与 Stage5.1 复用。
# 为什么要共享: Qwen3.5-9B 从磁盘加载实测要 ~10 分钟。原实现里 Stage3.7
# (交互判断) 和 Stage5.1 (关系判断) 各加载一次, 纯浪费。
_VLM_CACHE = {}


def get_shared_vlm(checkpoint=None, device="cuda"):
    """返回 (model, processor); 同一 checkpoint 只加载一次。

    失败会抛异常, 由调用方决定退化策略 (交互判断退化成纯几何, 不中断管线)。
    """
    ckpt = checkpoint or os.environ.get("VLM_CHECKPOINT") or DEFAULT_CKPT
    if ckpt not in _VLM_CACHE:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            ckpt, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map=device)
        model.eval()
        _VLM_CACHE[ckpt] = (model, processor)
    return _VLM_CACHE[ckpt]


def release_shared_vlm():
    """释放缓存 (管线结束时调用)。"""
    global _VLM_CACHE
    for m, _ in _VLM_CACHE.values():
        try:
            m.to("cpu")
        except Exception:
            pass
    _VLM_CACHE = {}
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _draw_index(img, idx, ratio_hint):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 340, 46], fill=(0, 0, 0))
    d.text((10, 8), f"S{idx} orig~{int(idx * ratio_hint)}", fill=(0, 255, 0))
    return img


def _extract_json_array(text):
    if not text:
        return []
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i = t.find("[")
    if i < 0:
        return []
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "[":
            depth += 1
        elif t[j] == "]":
            depth -= 1
            if depth == 0:
                try:
                    v = json.loads(t[i:j + 1])
                    return v if isinstance(v, list) else []
                except Exception:
                    return []
    return []


def vlm_detect_interactions(frame_paths, model_name=None, max_frames=12,
                            frame_ratio=1.0, device="cuda", verbose=True):
    """用本地 VLM 判断哪些物体被手拿起/放下。

    Args:
        frame_paths: 均匀采样的帧路径列表 (会自动再抽到 max_frames)
        model_name: VLM checkpoint; None 用环境变量/默认
        frame_ratio: 采样帧号 → 原始帧号 的换算系数
    Returns:
        list of {"object": str, "pick_frame": int, "place_frame": int}
        (帧号在**原始视频**时间轴)。失败时返回 []。
    """
    if not frame_paths:
        return []
    # 再采样到 max_frames
    if len(frame_paths) > max_frames:
        idx = [round(i * (len(frame_paths) - 1) / (max_frames - 1))
               for i in range(max_frames)]
        frame_paths = [frame_paths[i] for i in idx]

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except Exception as e:
        if verbose:
            print(f"   [VLM交互] 依赖缺失 ({type(e).__name__}), 退化为纯几何判据",
                  flush=True)
        return []

    ckpt = model_name or os.environ.get("VLM_CHECKPOINT") or DEFAULT_CKPT
    try:
        if verbose:
            print(f"   [VLM交互] 获取 VLM {os.path.basename(ckpt)} "
                  f"(共享缓存, 重复调用不重新加载) ...", flush=True)
        # 用共享缓存: 同一进程内 Stage3.7 / Stage5.1 复用, 省掉一次 ~10min 加载
        model, processor = get_shared_vlm(ckpt, device=device)

        images = []
        for i, p in enumerate(frame_paths):
            im = Image.open(p).convert("RGB")
            images.append(_draw_index(im, i, frame_ratio))

        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text",
                        "text": PROMPT.format(n=len(images))})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        text = processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        raw = _extract_json_array(text)
        if verbose:
            print(f"   [VLM交互] 原始输出: {text.strip()[:200]}", flush=True)
    except Exception as e:
        if verbose:
            print(f"   [VLM交互] 推理失败 ({type(e).__name__}: {str(e)[:80]}), "
                  f"退化为纯几何判据", flush=True)
        return []
    finally:
        # 不再主动卸载 —— 模型在共享缓存里, 交给 release_shared_vlm() 统一释放。
        # 原来这里 del + empty_cache, 会让下一次调用重新付 ~10min 加载代价。
        pass

    out = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        name = str(it.get("object", "")).strip().lower()
        if not name or name in _IGNORE:
            continue
        try:
            pf = int(float(it.get("pick_frame", -1)))
            pl = int(float(it.get("place_frame", -1)))
        except Exception:
            continue
        if pf < 0 or pl < 0 or pl <= pf:
            continue
        out.append({"object": name, "pick_frame": pf, "place_frame": pl})
    if verbose:
        print(f"   [VLM交互] 判定 {len(out)} 个被操作物体: "
              f"{[o['object'] for o in out]}", flush=True)
    return out


def match_vlm_to_instances(vlm_items, categories, synonym_map=None):
    """把 VLM 报的类别名匹配到 Stage1 检出的类别上。

    VLM 的自由文本名 (如 "earbud case"/"small black box") 和 Stage1 的类别名
    (如 "case"/"button") 往往不一致, 用子串 + 同义词表双向匹配。
    Returns:
        {vlm_object名: Stage1类别名 or None}
    """
    if synonym_map is None:
        try:
            from tools.generate_scene_json_stage1 import SYNONYM_MAP
            synonym_map = SYNONYM_MAP
        except Exception:
            synonym_map = {}

    def norm(s):
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

    cats_norm = {c: norm(c) for c in categories}
    # 同义词展开: 类别 → 一组别名
    alias = {}
    for c in categories:
        al = {norm(c)}
        for k, v in (synonym_map or {}).items():
            if norm(k) == norm(c) or norm(v) == norm(c):
                al.add(norm(k)); al.add(norm(v))
        alias[c] = al

    res = {}
    for it in vlm_items:
        nm = norm(it["object"])
        best, best_score = None, 0
        for c in categories:
            for a in alias[c]:
                if not a:
                    continue
                if a == nm:
                    best, best_score = c, 100
                    break
                # 子串匹配: "earbud case" ⊂ "case" 或反之
                if a in nm or nm in a:
                    sc = min(len(a), len(nm)) / max(len(a), len(nm)) * 50
                    if sc > best_score:
                        best, best_score = c, sc
            if best_score == 100:
                break
        res[it["object"]] = best
    return res
