#!/usr/bin/env python3
"""两个 VLM 后端交叉校验 —— 找出"质量估计不可信"的物体。

动机
----
质量只有两个来源，且都不可靠：
  1) 体积 × 类别密度    —— 依赖 SAM3D 重建尺度，实测个别物体差 40x
  2) VLM 直接估计(离散区间) —— 定性猜测，实测个别物体差 10x

单一来源不可信，但**两个独立来源互相印证**能给出置信度：
  一致 -> 可信
  冲突 -> 至少一个错，需要人工复核

为什么必须读日志而不能读 URDF
-----------------------------
`stage61_urdf_generator.py` 在拿到 VLM 结果后会执行
    attr_dict["min_mass"] = real_mass
    attr_dict["max_mass"] = real_mass
即**用体积法覆盖掉 VLM 给的区间**，所以两个后端的 URDF 里
<extra_info> 的 min_mass/max_mass 完全相同（都等于体积法值），读 URDF 无法区分后端。
VLM 的原始区间只存在于运行日志的 `Parsed: {...}` 行里。

用法
----
    python tools/eval/cross_check_vlm.py
    python tools/eval/cross_check_vlm.py --scenes basic_pick_place_14 --tol 1.5
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# 两侧 VLM 质量区间中值的比值超过该倍数 -> 判为"冲突"
RATIO_TOL = 1.5

# 本地 Qwen 日志的候选路径（历次批量跑留下的）
QWEN_LOG_CANDIDATES = [
    "/tmp/v3b_{scene}.log",      # 第三批(local Qwen)
    "/tmp/v3_{scene}.log",       # 早期批
    "/tmp/v3_bpp14.log",         # basic_pick_place_14 的首次运行
]
SN_LOG = "/tmp/sn_{scene}.log"

PARSED_RE = re.compile(r"Parsed:\s*(\{.*?\})", re.S)
MASS_RE = re.compile(r"=\s*([\d.]+)\s*kg")


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def parse_log(path: str) -> Dict[str, dict]:
    """从一份 stage61_urdf_generator 日志里提取每个物体的 VLM 区间与体积法质量。

    返回 {物体名: {"vlm_min", "vlm_max", "volume_mass"}}
    """
    txt = _read(path)
    if not txt:
        return {}
    out: Dict[str, dict] = {}
    # 按 "=== Generating URDF for <name>" 切块
    blocks = re.split(r"=== Generating URDF for ", txt)[1:]
    for b in blocks:
        name = b.split(" ")[0].strip()
        if not name:
            continue
        rec: dict = {}
        m = PARSED_RE.search(b)
        if m:
            try:
                d = ast.literal_eval(m.group(1))
                rec["vlm_min"] = float(d.get("min_mass", 0))
                rec["vlm_max"] = float(d.get("max_mass", 0))
            except (ValueError, SyntaxError, TypeError):
                pass
        m2 = re.search(r"= ([\d.]+) kg", b)
        if m2:
            rec["volume_mass"] = float(m2.group(1))
        if rec:
            out[name] = rec
    return out


def find_qwen_log(scene: str) -> Optional[str]:
    for pat in QWEN_LOG_CANDIDATES:
        p = pat.format(scene=scene)
        if not os.path.isfile(p):
            continue
        txt = _read(p)
        # 排除误命中 SenseNova 日志
        if "SenseNova VLM 就绪" in txt:
            continue
        return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="VLM 后端交叉校验")
    ap.add_argument("--scenes", nargs="*", default=None, help="限定场景名")
    ap.add_argument("--tol", type=float, default=RATIO_TOL,
                    help="两侧中值比值超过该倍数判为冲突 (默认 1.5)")
    a = ap.parse_args()

    scenes = a.scenes or [
        "basic_pick_place_14", "basic_pick_place_118", "basic_pick_place_17",
        "vertical_pick_place_115", "vertical_pick_place_42",
        "vertical_pick_place_64",
    ]

    rows: List[Tuple[str, str, Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    for sc in scenes:
        qlog = find_qwen_log(sc)
        slog = SN_LOG.format(scene=sc)
        if not qlog:
            print(f"[skip] {sc}: 找不到本地 Qwen 日志")
            continue
        if not os.path.isfile(slog):
            print(f"[skip] {sc}: 找不到 SenseNova 日志 {slog}")
            continue
        q = parse_log(qlog)
        s = parse_log(slog)
        for name in sorted(set(q) & set(s)):
            rows.append((sc, name,
                         q[name].get("vlm_min"), q[name].get("vlm_max"),
                         s[name].get("vlm_min"), s[name].get("vlm_max")))

    if not rows:
        print("没有可比对的数据（需要同一场景同时存在 Qwen 日志与 sn_*.log）")
        return 1

    print(f"{'场景':26s} {'物体':16s} {'Qwen':>14s} {'SenseNova':>14s} {'比值':>7s} {'判定':>6s}")
    print("-" * 92)
    same = conflict = 0
    for sc, name, qmin, qmax, smin, smax in rows:
        if None in (qmin, qmax, smin, smax):
            print(f"{sc:26s} {name:16s} {'(缺失)':>14s}")
            continue
        qmid = (qmin + qmax) / 2
        smid = (smin + smax) / 2
        ratio = max(qmid, smid) / max(min(qmid, smid), 1e-9)
        ok = ratio <= a.tol
        same += ok
        conflict += not ok
        print(f"{sc:26s} {name:16s} {f'{qmin:.2f}-{qmax:.2f}':>14s} "
              f"{f'{smin:.2f}-{smax:.2f}':>14s} {ratio:6.2f}x "
              f"{'一致' if ok else '冲突':>6s}")
    print("-" * 92)
    n = same + conflict
    if n:
        print(f"可比物体: {n}   一致: {same} ({same/n:.0%})   冲突: {conflict} ({conflict/n:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
