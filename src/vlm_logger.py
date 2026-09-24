"""
VLM 对话记录工具
==================
将每次 VLM 调用的 prompt + response 记录到独立文件 {output_dir}/vlm_conversation.jsonl，
不与 mainv2.log 等其他日志混合。

使用方式:
    from src.vlm_logger import vlm_inference, set_vlm_log_dir

    set_vlm_log_dir(output_path)       # 设置输出目录
    response = vlm_inference(           # 等价于原来的 _vlm_inference()
        image, model, processor, prompt,
        max_new_tokens=1024, caller="stage1_detect"
    )
"""

import json
import os
import time

import torch

_vlm_log_path = None


def set_vlm_log_dir(log_dir: str):
    """设置 VLM 对话日志的输出目录，会在其中创建 vlm_conversation.jsonl"""
    global _vlm_log_path
    os.makedirs(log_dir, exist_ok=True)
    _vlm_log_path = os.path.join(log_dir, "vlm_conversation.jsonl")


def vlm_inference(image, model, processor, prompt, max_new_tokens=1024, temperature=0.0, do_sample=False, caller="unknown"):
    """
    调用VLM进行单次推理，并记录 prompt + response 到对话日志文件。

    参数:
        image: PIL图像 或 图像路径字符串
        model: VLM模型
        processor: VLM处理器
        prompt: 提示词文本
        max_new_tokens: 最大生成token数
        temperature: 采样温度 (默认0.0=贪心)
        do_sample: 是否采样 (默认False)
        caller: 调用方标识，用于日志溯源 (如 "stage1_detect", "stage5_scene_graph")
    返回:
        VLM输出的文本
    """
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
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

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = max(temperature, 0.01)
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # 记录对话到独立文件
    if _vlm_log_path:
        _log_vlm(caller, prompt, output_text)

    return output_text


def _log_vlm(caller: str, prompt: str, response: str):
    """将一次 VLM 对话追加写入 jsonl 文件"""
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "caller": caller,
        "prompt": prompt,
        "response": response,
    }
    try:
        with open(_vlm_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"  ⚠️ 写入VLM对话日志失败: {e}", flush=True)