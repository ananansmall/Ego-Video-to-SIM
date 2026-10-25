#!/usr/bin/env python3
"""SenseNova 云端 VLM 适配器 (OpenAI 兼容接口)。

与 tools/qwen_client.QwenClient 接口保持一致, 便于在 stage61_urdf_generator 中互换:

    client = SenseNovaClient(model="sensenova-6.8-flash-lite")
    text = client.query(prompt_text, [img1, img2, ...])   # 返回格式化后的文本

apiKey 解析顺序:
    1. 显式传入 api_key
    2. 环境变量 SENSENOVA_API_KEY
    3. ~/.config/opencode/opencode.jsonc 里 provider.sensenova.options.apiKey
       (复用 opencode 已配置的凭据, 不额外落盘)

注意: 该免费/低额度 key 有 tpm/rpm 限制, 实测会返回 429。
本模块内置指数退避重试 (max_retries 次), 仍失败则抛出 RuntimeError。
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://token.sensenova.cn/v1"
DEFAULT_MODEL = "sensenova-6.8-flash-lite"
_OPENCODE_CONFIG = os.path.expanduser("~/.config/opencode/opencode.jsonc")


def _load_opencode_key() -> Optional[str]:
    """从 opencode 配置里读 sensenova 的 apiKey (容错: jsonc 去注释后解析)。"""
    if not os.path.isfile(_OPENCODE_CONFIG):
        return None
    try:
        txt = open(_OPENCODE_CONFIG, encoding="utf-8").read()
        txt = re.sub(r"^\s*//.*$", "", txt, flags=re.M)
        cfg = json.loads(txt)
        return cfg.get("provider", {}).get("sensenova", {}) \
                  .get("options", {}).get("apiKey")
    except Exception as e:  # 配置损坏不应让整个流程挂掉
        logger.warning(f"解析 opencode 配置失败: {e}")
        return None


def _resolve_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    env = os.environ.get("SENSENOVA_API_KEY")
    if env:
        return env
    cfg = _load_opencode_key()
    if cfg:
        return cfg
    raise RuntimeError(
        "未找到 SenseNova apiKey: 请传入 api_key, 或设置 SENSENOVA_API_KEY, "
        f"或在 {_OPENCODE_CONFIG} 中配置 provider.sensenova.options.apiKey"
    )


def _image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def _load_formatter():
    """加载 qwen_client 的 _format_response; 兼容脚本被直接运行(此时 sys.path 无项目根)。"""
    try:
        from tools.qwen_client import _format_response
        return _format_response
    except ImportError:
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from tools.qwen_client import _format_response
        return _format_response


class SenseNovaClient:
    """SenseNova VLM 适配器, 兼容 QwenClient.query() 接口。"""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 base_url: str = DEFAULT_BASE_URL, max_retries: int = 4,
                 timeout: float = 90.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = _resolve_api_key(api_key)
        self.max_retries = max_retries
        self.timeout = timeout
        logger.info(f"SenseNova VLM 就绪: model={self.model} base={self.base_url} "
                    f"(key {len(self.api_key)} 字符)")

    # ---- 内部: 带退避的请求 ----
    def _post(self, payload: dict) -> dict:
        last = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=payload, timeout=self.timeout,
                )
                if r.status_code == 200:
                    return r.json()
                # 429 限流 / 5xx 服务端: 退避重试
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = min(2 ** attempt * 2, 30)
                    logger.warning(f"SenseNova HTTP {r.status_code}, "
                                   f"{wait}s 后重试 ({attempt + 1}/{self.max_retries})")
                    time.sleep(wait)
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue
                # 其它 4xx 直接失败, 重试无意义
                raise RuntimeError(f"SenseNova 请求失败 HTTP {r.status_code}: {r.text[:300]}")
            except requests.RequestException as e:
                wait = min(2 ** attempt * 2, 30)
                logger.warning(f"SenseNova 网络异常 {type(e).__name__}, {wait}s 后重试")
                time.sleep(wait)
                last = str(e)
        raise RuntimeError(f"SenseNova 重试 {self.max_retries} 次仍失败: {last}")

    @staticmethod
    def _extract_text(data: dict) -> str:
        """兼容不同返回结构: content 可能为空, 文本可能在 reasoning_content 里。"""
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError):
            return ""
        if not isinstance(msg, dict):
            return str(msg)
        for k in ("content", "reasoning_content", "reasoning", "text"):
            v = msg.get(k)
            if v:
                # content 也可能是 [{type:text, text:...}] 形式
                if isinstance(v, list):
                    parts = [seg.get("text", "") for seg in v if isinstance(seg, dict)]
                    v = "".join(parts)
                if v and str(v).strip():
                    return str(v)
        return ""

    # ---- 对外接口 (与 QwenClient 一致) ----
    def query(self, text_prompt: str, image_paths: Optional[list[str]] = None,
              system_role: Optional[str] = None, params: Optional[dict] = None) -> str:
        """调用 SenseNova 多模态接口, 返回格式化后的文本 (与 QwenClient.query 同约定)。"""
        _format_response = _load_formatter()  # 复用 qwen_client 的格式化逻辑

        valid = [p for p in (image_paths or []) if os.path.isfile(p)]
        if not valid:
            logger.warning("SenseNovaClient.query() 未收到有效图片, 返回空串")
            return ""

        content: list[dict] = [{"type": "text", "text": text_prompt}]
        for p in valid:
            content.append({"type": "image_url",
                            "image_url": {"url": _image_to_data_url(p)}})

        messages = []
        if system_role:
            messages.append({"role": "system", "content": system_role})
        messages.append({"role": "user", "content": content})

        # 注意: 6.8-flash-lite 是推理模型, 会先在内部做长推理再输出答案。
        # 实测 max_tokens=512 时答案还没输出就被截断, 解析到的只是推理片段;
        # 给到 4000 才稳定拿到完整的 7 行结果。默认值因此调高。
        payload = {"model": self.model, "messages": messages,
                   "max_tokens": int((params or {}).get("max_new_tokens", 4000))}
        data = self._post(payload)
        raw = self._extract_text(data)
        if not raw:
            logger.warning(f"SenseNova 返回内容为空, 原始响应: "
                           f"{json.dumps(data, ensure_ascii=False)[:300]}")
            return ""

        logger.info(f"SenseNova raw response:\n{raw[:300]}")
        formatted = _format_response(raw)
        logger.info(f"Formatted response:\n{formatted}")
        return formatted


if __name__ == "__main__":
    # 自测: 对给定的物体渲染图跑一次标注
    import argparse
    from glob import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--prompt", default="Describe this object briefly.")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    imgs = []
    for pat in a.images:
        imgs.extend(sorted(glob(pat)) or [pat])
    print(SenseNovaClient(model=a.model).query(a.prompt, imgs))
