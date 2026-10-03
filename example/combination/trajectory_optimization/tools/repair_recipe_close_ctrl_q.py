"""R03 诊断工具 —— 把 recipe 的 `close_ctrl_q` 从 **全段** 修成 **close 相位实际段** ✓

背景 (2026-09-28 实测, 三处现场一致 ✓)
----------------------------------------
`output/{s31,s44,gif_s31}/stage1/stage1_grasp_recipe.json` 都是:
    `close_ctrl_q` 长度 = **350** (= 全段实际 q ✗, 而 `close_ctrl_t` 只有 16 ✗ 语义打架)
    `phase`        = close:130 / settle:10 / lift:210 ✓
    `phase_boundaries.close = [0, 130]` ✓
消费侧 (`stage2_reconstruct_mixin.verify_stage1_transfer_equivalence:5261` /
`_stage1_reuse_report`) 期望:
    `close_ctrl_q` == `split_trace_by_phase(recipe)["close"]["q"]` (**130 点**, 逐位 atol=1e-12)
⇒ 长度 350 ≠ 130 ⇒ `--test-stage1-transfer` **恒 FAIL(STAGE1_REUSE_MISMATCH)** ✗
  ⇒ 回放入口从未通过 ⇒ "成功抬起录像"永远拿不到 ✗

本工具 ✓
--------
· **只读**原 recipe, **只写** `--out` 指定目录 (默认 `output/replay_fixed/stage1`) ⇒ 不动 s31/s44 现场 ✓
· 用 recipe 自带的 `phase`/`phase_boundaries` 切出 close 段 ⇒ 逐位保留, 一个点不重算 ✓
· 同时写 `close_ctrl_q_source` 标注来源 (可追溯 ✓)
· 随后把同目录的 `best_grasp.npz` 复制过去 (若给出 `--copy-best`)

用法
----
    python tools/repair_recipe_close_ctrl_q.py \\
        --recipe output/gif_s31/stage1/stage1_grasp_recipe.json \\
        --out output/replay_s31b/stage1 --copy-best output/gif_s31/stage1/best_grasp.npz
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def split_close_segment(recipe: dict) -> list[float]:
    """按 phase 切 close 段 (优先用 phase_boundaries ⇒ 与消费侧同口径 ✓)"""
    q = [float(x) for x in (recipe.get("q_cmd") or [])]
    ph = [str(x) for x in (recipe.get("phase") or [])]
    bnd = (recipe.get("phase_boundaries") or {}).get("close") or []

    seg_b, seg_p = [], []
    if q and len(ph) == len(q) and bnd:
        lo, hi = int(bnd[0][0]), int(bnd[0][1])
        # ★ 半开区间 [lo, hi) ✓ —— 与 `stage1_recipe.split_trace_by_phase` 的 `q[a:b]` **同口径** ✓
        #   ⚠ 早先写成 `hi + 1` ⇒ 131 点 ⇒ 与消费侧 130 点仍差 1 ✗ (又被 MISMATCH 拦下)
        seg_b = q[lo: hi]
    if q and len(ph) == len(q):
        seg_p = [v for v, p in zip(q, ph) if p == "close"]
    return seg_b or seg_p


def repair_npz(src_npz: Path, out_npz: Path, new_recipe: dict) -> str:
    """★ 关键 ✓: `load_stage1_grasp` **优先**读 npz 里的 `stage1_recipe_json`
    (实测 ✗: 只改 JSON 文件 ⇒ 载入的 recipe 仍是旧 350 ⇒ 仍 FAIL)
    ⇒ 必须**同时**把 npz 内嵌 recipe 换成修正版 ✓
    """
    import numpy as np

    d = np.load(str(src_npz), allow_pickle=True)
    kw = {k: d[k] for k in d.keys()}
    old_len = -1
    try:
        _old = json.loads(str(kw.get("stage1_recipe_json")))
        old_len = len(_old.get("close_ctrl_q") or [])
    except Exception:  # noqa: BLE001
        pass
    kw["stage1_recipe_json"] = np.array(json.dumps(new_recipe, default=str))
    np.savez(str(out_npz), **kw)
    return f"npz 内嵌 recipe: close_ctrl_q {old_len} → {len(new_recipe.get('close_ctrl_q') or [])}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True, help="输出目录 (会创建 <out>/stage1_grasp_recipe.json)")
    ap.add_argument("--copy-best", default=None, help="best_grasp.npz 路径 (复制到同目录)")
    ap.add_argument("--repair-npz", default=None,
                    help="★ 同时把该 npz 内嵌的 stage1_recipe_json 换成修正版 (必需 ✓)")
    args = ap.parse_args()

    src = Path(args.recipe)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(src.read_text())

    old = recipe.get("close_ctrl_q") or []
    seg = split_close_segment(recipe)
    if not seg:
        print("✗ 无法切出 close 段 (缺 q_cmd/phase) ⇒ 不改动")
        return 2

    print(f"[R03] 原 close_ctrl_q 长度 = {len(old)}")
    print(f"[R03] close 相位实际段长度 = {len(seg)} (phase_boundaries={recipe.get('phase_boundaries')})")
    if old[:len(seg)] == [round(x, 9) for x in seg[:len(old) and len(seg)]] or old == seg:
        print("[R03] 前段逐位一致 ✓ (说明只是**长度**写错)")
    recipe["close_ctrl_q"] = seg
    recipe["close_ctrl_q_source"] = (
        "repaired_close_phase_segment (R03 诊断: 由 tools/repair_recipe_close_ctrl_q.py 从 "
        "phase/q_cmd 切出; 原字段长度为 %d 全段 ✗)" % len(old))

    dst = out_dir / "stage1_grasp_recipe.json"
    dst.write_text(json.dumps(recipe, indent=2, ensure_ascii=False, default=str))
    print(f"[R03] ✓ 写出修正版: {dst} (close_ctrl_q {len(old)} → {len(seg)})")

    if args.copy_best:
        b = Path(args.copy_best)
        if b.exists():
            shutil.copy2(b, out_dir / "best_grasp.npz")
            print(f"[R03] ✓ 已复制 {b.name} → {out_dir / 'best_grasp.npz'}")
        else:
            print(f"[R03] ⚠ best_grasp 不存在: {b}")

    if args.repair_npz:
        n_src = Path(args.repair_npz)
        if not n_src.exists():
            print(f"[R03] ✗ npz 不存在: {n_src}")
            return 3
        n_dst = out_dir / n_src.name
        print(f"[R03] {repair_npz(n_src, n_dst, recipe)} ⇒ {n_dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
