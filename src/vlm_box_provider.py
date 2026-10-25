#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 opencode + VLM (默认 sensenova-6.8-flash-lite) 为 SAM3 提供 box 兜底提示。

用途: src/object_segmentation.segment_and_track 的 box_provider 回调。
      仅当文本提示切不出任何 mask 时被调用 (见该函数的注释)。

设计要点:
  1. 全部失败路径都返回 [] —— 调用方拿到空就退化成原行为, 绝不阻断主流程。
     (实测 stage5.5 的 opencode agent 有 6/17 因限流/超时失败, 这条很重要)
  2. 结果按 (scene_dir, category) 落盘缓存, 同一场景重跑不重复烧 API。
  3. 并发由调用方控制; 这里只做单次的超时保护。

注意: opencode 是 agent CLI, 对"外部目录"有权限拦截 (实测拒绝读项目目录)。
      因此调用方需要先把图拷到允许的位置, 或让 cwd 落在图所在目录。
"""
import json
import os
import re
import shutil
import subprocess
import tempfile

DEFAULT_MODEL = "sensenova/sensenova-6.8-flash-lite"
OPENCODE_BIN = shutil.which("opencode") or os.path.expanduser("~/.opencode/bin/opencode")

_BOX_PROMPT = """In this image, find the {cls}.

Give its bounding box. Output JSON only, no markdown fence:
{{"boxes": [{{"type": "{cls}", "bbox": [x1, y1, x2, y2]}}]}}

bbox = integers 0-1000 normalized to image width/height (x1,y1 top-left; x2,y2 bottom-right).
If the {cls} is not visible, output {{"boxes": []}}."""


def _extract_json(text):
    """从可能带 markdown fence / 前后废话的输出里抠出 JSON。"""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    i = t.find("{")
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    return None
    return None


def _call_opencode(prompt, cwd, model, timeout):
    cfg = {"$schema": "https://opencode.ai/config.json",
           "permission": {"edit": "deny", "bash": {"*": "allow"},
                          "webfetch": "deny"}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name
    env = dict(os.environ)
    env["OPENCODE_CONFIG"] = cfg_path
    cmd = [OPENCODE_BIN, "run", "--model", model, prompt]
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                           text=True, timeout=timeout)
        return (p.stdout or "").strip()
    except Exception:
        return ""
    finally:
        try:
            os.unlink(cfg_path)
        except Exception:
            pass


class VLMBoxProvider:
    """给 segment_and_track 用的 box_provider。

    Args:
        image_path: 要问的那一帧图片路径 (会被拷到临时目录后再给 opencode)
        scene_dir: 缓存落盘位置 (可选)
        model: opencode 模型
        timeout: 单次调用超时秒数
        enabled: 总开关; False 时 __call__ 直接返回 []
    """

    def __init__(self, image_path, scene_dir=None, model=DEFAULT_MODEL,
                 timeout=300, enabled=True, verbose=True):
        self.image_path = image_path
        self.scene_dir = scene_dir
        self.model = model
        self.timeout = timeout
        self.enabled = enabled and os.path.exists(OPENCODE_BIN)
        self.verbose = verbose
        self._cache = {}
        self._tmpdir = None
        self.stats = {"called": 0, "hit": 0, "fail": 0}
        if scene_dir:
            self._cache_path = os.path.join(scene_dir, "vlm_box_cache.json")
            if os.path.isfile(self._cache_path):
                try:
                    self._cache = json.load(open(self._cache_path, encoding="utf-8"))
                except Exception:
                    self._cache = {}
        else:
            self._cache_path = None

    def _prepared_image(self):
        """返回 (cwd, image_path)。

        实测踩过的坑 (2026-10-19): 把图拷到 /tmp 下的临时目录再以该目录为 cwd,
        opencode 会报
            permission requested: external_directory (/tmp/*); auto-rejecting
        并静默返回空 —— 表现为"取到 0 个框", 但 stats 里 fail=0, 极难排查。

        可行做法是 Stage 5.5 在用的那套: cwd = 项目根目录, 图片路径用项目内路径。
        因此这里不再拷贝, 直接用原路径 + 项目根作 cwd。
        """
        if self._tmpdir is None:
            # 项目根目录: 向上找到含 models/ 与 tools/ 的那一层
            cur = os.path.abspath(self.image_path)
            for _ in range(8):
                cur = os.path.dirname(cur)
                if os.path.isdir(os.path.join(cur, "tools")) and \
                        os.path.isdir(os.path.join(cur, "src")):
                    self._tmpdir = cur
                    break
            else:
                self._tmpdir = os.path.dirname(os.path.abspath(self.image_path))
        rel = os.path.relpath(os.path.abspath(self.image_path), self._tmpdir)
        return self._tmpdir, rel

    def __call__(self, category):
        """返回归一化 0~1 的框列表 [(x1,y1,x2,y2), ...]; 任何失败都返回 []。"""
        if not self.enabled:
            return []
        if category in self._cache:
            self.stats["hit"] += 1
            return self._cache[category]
        self.stats["called"] += 1
        try:
            cwd, img = self._prepared_image()
            prompt = ("Read the image file at %s, then follow these instructions.\n\n%s"
                      % (img, _BOX_PROMPT.format(cls=category)))
            out = _call_opencode(prompt, cwd, self.model, self.timeout)
            d = _extract_json(out)
            boxes = []
            if d and isinstance(d.get("boxes"), list):
                for b in d["boxes"]:
                    bb = b.get("bbox")
                    if isinstance(bb, list) and len(bb) == 4:
                        try:
                            x1, y1, x2, y2 = [float(v) / 1000.0 for v in bb]
                        except Exception:
                            continue
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2))
            self._cache[category] = boxes
            self._save()
            if self.verbose:
                print(f"    [boxProvider] {category}: 取到 {len(boxes)} 个框", flush=True)
            return boxes
        except Exception as e:
            self.stats["fail"] += 1
            if self.verbose:
                print(f"    [boxProvider] {category}: 失败 "
                      f"({type(e).__name__}: {str(e)[:60]})", flush=True)
            return []

    def _save(self):
        if not self._cache_path:
            return
        try:
            json.dump(self._cache, open(self._cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        except Exception:
            pass
