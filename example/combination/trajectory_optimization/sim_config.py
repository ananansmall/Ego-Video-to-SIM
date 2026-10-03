"""★ 统一仿真配置加载器 (2026-10-02) ✓

设计 ✓: yaml 的 key **就是真实 env 名** ⇒ 写入 `os.environ` ⇒ 现有代码**零改动**生效 ✓
优先级 ✓: **CLI/已有 env > yaml > 代码默认** (绝不覆盖已显式设置的值 ✗)
用法 ✓: 在入口最早处 `from sim_config import load_sim_config; load_sim_config()`
"""
from __future__ import annotations

import os
from pathlib import Path

_CFG = Path(__file__).resolve().parent / "config" / "sim.yaml"


def load_sim_config(path=None, verbose=True):
    """把 yaml 里的 env 键写入环境 (不覆盖已存在的) ⇒ 返回本次实际设置项 ✓"""
    p = Path(path) if path else _CFG
    if not p.exists():
        return {}
    try:
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        print("[sim_config] yaml 读取失败(忽略): %s" % e)
        return {}
    applied, skipped = {}, []
    for _sec, _kv in data.items():
        if not isinstance(_kv, dict):
            continue
        for _k, _v in _kv.items():
            if not str(_k).startswith("_"):
                continue
            if _k in os.environ:                 # 已有 ⇒ 尊重 (CLI/env 优先 ✓)
                skipped.append(_k)
                continue
            os.environ[_k] = ("1" if _v is True else "0" if _v is False else str(_v))
            applied[_k] = os.environ[_k]
    if verbose:
        print("[sim_config] 已应用 %d 项 (来自 %s); 跳过 %d 项(已有 env 优先 ✓)"
              % (len(applied), p.name, len(skipped)))
        if skipped:
            print("[sim_config]   跳过: %s" % ", ".join(sorted(skipped)[:12]))
    return applied


if __name__ == "__main__":
    load_sim_config()
