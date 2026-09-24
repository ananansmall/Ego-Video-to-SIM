#!/usr/bin/env python3
"""VLM (Qwen3.5-9B) Pick/Place 帧检测 vs 几何方法对比测试

方法: 仿 Stage1 的 VLM 加载方式, 给模型均匀抽帧的 color 图 (带帧号标注),
prompt 要求输出每个物体抓取(pick)/放置(place)的大概帧号, 输出 JSON。
然后与 pick_place_frames_combined.json (几何方法) 对比。
"""
import os, sys, json, re
import torch
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, REPO)

BASE = os.path.join(REPO, 'output_v2')
MODEL = "/mnt/data/lza/models/Qwen3.5-9B"
N_SAMPLES = 12          # 均匀抽帧数 (与 Stage1 关键帧数一致)
MAX_NEW_TOKENS = 1024

SCENES = {
    '2chair-bottle_vggt_omega':            2.29,
    '7_vggt_omega':                        1.00,
    '3chair-bottle_vggt_omega':            4.93,
    '121_C5_CellPhone_161deg_vggt_omega':  3.00,
    '116_C5_CellPhone_185deg_vggt_omega':  3.00,
}

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


def draw_index(img, idx, total_hint):
    """左上角画帧号 (采样索引 + 原始帧号换算提示)."""
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 340, 46], fill=(0, 0, 0))
    d.text((10, 8), f"S{idx} orig~{int(idx * total_hint)}", fill=(0, 255, 0))
    return img


def load_vlm():
    from transformers import AutoProcessor, AutoModelForImageTextToText
    print(f"📦 加载 {MODEL} ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda:0")
    model.eval()
    print("✅ 模型加载完成", flush=True)
    return model, processor


def vlm_query(model, processor, prompt, images):
    messages = [{"role": "user", "content":
        [{"type": "text", "text": prompt}] +
        [{"type": "image", "image": im} for im in images]}]
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False)
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                             pad_token_id=processor.tokenizer.eos_token_id,
                             do_sample=False)
    gen = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def parse_json(text):
    m = re.search(r'\[.*\]', text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


def main():
    only = sys.argv[1:]
    model, processor = load_vlm()
    for scene, ratio in SCENES.items():
        if only and scene not in only:
            continue
        scene_dir = os.path.join(BASE, scene)
        n_rgb = len([f for f in os.listdir(os.path.join(scene_dir, 'color'))
                     if f.endswith('.jpg')])
        # 均匀抽帧
        fids = [round(i * (n_rgb - 1) / (N_SAMPLES - 1)) for i in range(N_SAMPLES)]
        # 帧号换算: 采样索引 → 原始视频帧 (画在图上, 帮模型对齐时间轴)
        orig_hint = ratio * (n_rgb - 1) / (N_SAMPLES - 1) if N_SAMPLES > 1 else ratio
        images = []
        for k, fid in enumerate(fids):
            im = Image.open(os.path.join(scene_dir, 'color', f'{fid}.jpg')).convert('RGB')
            im = draw_index(im, k, orig_hint)
            images.append(im)
        print(f"\n{'='*70}\n{scene} (n_rgb={n_rgb}, ratio={ratio}, 抽帧={fids})", flush=True)

        raw = vlm_query(model, processor, PROMPT.format(n=N_SAMPLES), images)
        print(f"--- VLM raw ---\n{raw}\n--- end ---", flush=True)
        vlm_items = parse_json(raw)

        # 几何方法结果 (转 RGB 空间对比)
        geo_path = os.path.join(scene_dir, 'pick_place_frames_combined.json')
        geo_items = []
        if os.path.exists(geo_path):
            with open(geo_path) as f:
                gd = json.load(f)
            for it in gd['interactions']:
                geo_items.append({
                    'object': it['instance_name'],
                    'pick_rgb': round(it['pick_frame'] / gd['frame_ratio']),
                    'place_rgb': round(it['place_frame'] / gd['frame_ratio']),
                    'pick_orig': it['pick_frame'],
                    'place_orig': it['place_frame'],
                })

        print(f"VLM  检出: {json.dumps(vlm_items, ensure_ascii=False)}", flush=True)
        print(f"几何 检出 (RGB空间): {json.dumps(geo_items, ensure_ascii=False)}", flush=True)

        out = {'vlm_raw': raw, 'vlm_parsed': vlm_items, 'geometry': geo_items,
               'sampled_fids': fids, 'frame_ratio': ratio, 'n_rgb': n_rgb}
        with open(os.path.join(scene_dir, 'vlm_pick_place_compare.json'), 'w') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
