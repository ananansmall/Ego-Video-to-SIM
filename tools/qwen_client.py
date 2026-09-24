"""
QwenClient — Qwen3.5-9B VLM 适配器，实现与 EmbodiedGen GPTclient 相同的 query() 接口。

输出格式化为 EmbodiedGen URDFGenerator 的 prompt_template 期望的标准格式：
    Category: cup
    Description: shiny golden cup with floral design
    Pose: upright
    Height: 0.10-0.15 m
    Weight: 0.3-0.6 kg
    Static friction coefficient: 0.6
    Dynamic friction coefficient: 0.5
"""

import logging
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

logger = logging.getLogger(__name__)

# EmbodiedGen prompt_template — 直接从 urdf_convertor.py 复制
PROMPT_TEMPLATE = """This is an orthographic projection showing the front(1st image), right(2nd), back(3rd), and left(4th) views of the 3D object asset, category: {category}.
You are an expert in 3D object analysis and physical property estimation.
Give the category of this object asset (within 3 words), (if category is already provided, use it directly), accurately describe this 3D object asset (within 15 words), Determine the pose of the object in the first image based on all views and estimate the true vertical height (vertical projection) range of the object (in meters), i.e., how tall the object appears from top to bottom in the first image. also weight range (unit: kilogram), the average static friction coefficient of the object relative to rubber and the average dynamic friction coefficient of the object relative to rubber.

Output Example:
Category: cup
Description: shiny golden cup with floral design
Pose: <short_description_within_10_words>
Height: 0.10-0.15 m
Weight: 0.3-0.6 kg
Static friction coefficient: 0.6
Dynamic friction coefficient: 0.5"""

# Qwen3.5-9B-VL prompt — 直接让模型输出标准格式
SYSTEM_PROMPT = """You are an expert in 3D object analysis and physical property estimation.
You are given up to 4 orthographic projection views of a 3D object (front, right, back, left).
Based on all views, estimate the following physical properties.
Output ONLY in the exact format below, no extra text:

Category: {category_in_3_words}
Description: {short_15_word_description}
Pose: {short_10_word_pose_description}
Height: {min}-{max} m
Weight: {min}-{max} kg
Static friction coefficient: {number}
Dynamic friction coefficient: {number}"""


def _format_response(raw: str) -> str:
    """从 Qwen 原始输出中提取字段，格式化为 GPT 兼容的标准格式。"""
    category = re.search(r"[Cc]ategory:\s*(\w+)", raw)
    description = re.search(r"[Dd]escription:\s*(.+)", raw)
    height = re.search(r"[Hh]eight:\s*([\d.]+)\s*-\s*([\d.]+)\s*m", raw)
    weight = re.search(r"[Ww]eight:\s*([\d.]+)\s*-\s*([\d.]+)\s*kg", raw)
    mu1 = re.search(r"[Ss]tatic.*?[Cc]oefficient:\s*([\d.]+)", raw)
    mu2 = re.search(r"[Dd]ynamic.*?[Cc]oefficient:\s*([\d.]+)", raw)

    cat = category.group(1).lower() if category else "unknown"
    desc = description.group(1).strip() if description else "unknown"
    pose = "upright"

    if height:
        h_min, h_max = height.group(1), height.group(2)
    else:
        h_min, h_max = "0.1", "0.2"

    if weight:
        w_min, w_max = weight.group(1), weight.group(2)
    else:
        w_min, w_max = "0.1", "0.3"

    mu1_val = mu1.group(1) if mu1 else "0.6"
    mu2_val = mu2.group(1) if mu2 else "0.5"

    return (
        f"Category: {cat}\n"
        f"Description: {desc}\n"
        f"Pose: {pose}\n"
        f"Height: {h_min}-{h_max} m\n"
        f"Weight: {w_min}-{w_max} kg\n"
        f"Static friction coefficient: {mu1_val}\n"
        f"Dynamic friction coefficient: {mu2_val}"
    )


class QwenClient:
    """Qwen VLM 适配器，兼容 EmbodiedGen GPTclient 的 query() 接口。"""

    def __init__(self, vlm_checkpoint: str, device: str = "cuda", max_new_tokens: int = 512):
        """
        Args:
            vlm_checkpoint: Qwen VLM 模型路径，如 "/mnt/data/lza/models/Qwen3.5-9B"
            device: 推理设备
            max_new_tokens: 最大生成长度
        """
        logger.info(f"Loading Qwen VLM from {vlm_checkpoint} on {device}...")
        self.vlm_checkpoint = vlm_checkpoint
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(vlm_checkpoint, trust_remote_code=True)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            vlm_checkpoint,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        logger.info("Qwen VLM loaded successfully.")

    def query(self, text_prompt: str, image_paths: Optional[list[str]] = None,
              system_role: Optional[str] = None, params: Optional[dict] = None) -> str:
        """
        调用 Qwen VLM 推理，与 mainv2 的 vlm_logger.py 使用相同的接口。

        Args:
            text_prompt: 提示文本（通常是 EmbodiedGen 的 prompt_template）
            image_paths: 图片路径列表（最多 4 张，对应 4 个视图）
            system_role: 系统提示（不使用，保留兼容 GPTclient 接口）
            params: 额外参数（不使用，保留兼容 GPTclient 接口）

        Returns:
            格式化后的标准文本，与 GPT 输出格式一致
        """
        if image_paths is None or len(image_paths) == 0:
            logger.warning("No images provided to QwenClient.query(), returning empty.")
            return ""

        # 过滤有效路径并加载 PIL 图像
        valid_paths = [p for p in image_paths if os.path.isfile(p)]
        if not valid_paths:
            logger.error(f"All image paths invalid: {image_paths}")
            return ""

        images = [Image.open(p).convert("RGB") for p in valid_paths]

        # 构建 messages: 系统提示（如有）+ 用户消息（文本+图片）
        messages = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": text_prompt}] +
                       [{"type": "image", "image": img} for img in images]
        })

        # 编码 (tokenize=True + return_tensors="pt")，与 vlm_logger 一致
        try:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False
            )
        except TypeError:
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        inputs = inputs.to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "do_sample": False,
        }

        # AutoModelForImageTextToText 的 prepare_inputs_for_generation 支持
        # pixel_values, image_grid_thw, mm_token_type_ids 等多模态参数。
        # 使用完整 inputs，模型会自动处理图像嵌入。
        gen_inputs = dict(inputs)

        with torch.no_grad():
            generated_ids = self.model.generate(**gen_inputs, **gen_kwargs)

        # 解码（只取新生成的文本）
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # 格式化为 GPT 兼容格式
        formatted = _format_response(raw)
        logger.info(f"Qwen raw response:\n{raw[:300]}")
        logger.info(f"Formatted response:\n{formatted}")

        return formatted
