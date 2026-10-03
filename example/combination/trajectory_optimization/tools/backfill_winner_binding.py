#!/usr/bin/env python
"""★ md §04 历史产物绑定回填 (2026-09-30) ✓

问题 ✓ (实测 `output/gif_s31` 等):
    `verification_results.runs` = [128, 129, 130] (passed=True)
    但 `winner_binding.episode_id` = **s1ep-000124** (搜索最优, **未在 passed 列表**)
    且 `verification_results.selected_episode` = **None** ⇒ md §04 的绑定不一致 ✗

原则 ✓ (只补绑定, **不改事实** ✗):
  · `selected_episode` := **第一个 passed 的 episode_id** (进入 passed 列表 ✓)
  · `winner_binding.episode_id` **保留原值** (那是"搜索最优"这一**事实** ✗ 不篡改 ✓)
    另加 `verified_episode_ids` / `binding_note` 说明两者语义 ✓
  · 仅处理"**成功产物**" (export_valid ∧ n_pass>=2) ✓; 搜索中途产物不动 ✓
  · 覆盖前把原文件备份到 `trimesh/backfill_<时间戳>/` ✓ (遵守 safe-delete: 不真删 ✓)
  · npz 内嵌 `stage1_recipe_json` 是**真源** ⇒ 同步回填 ✓

用法:
    python tools/backfill_winner_binding.py                 # dry-run (只打印计划)
    python tools/backfill_winner_binding.py --apply         # 实际写入
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _plan_one(path: Path, data: dict) -> dict | None:
    vr = data.get("verification_results") or {}
    runs = list(vr.get("runs") or [])
    passed = [str(r.get("episode_id")) for r in runs
              if isinstance(r, dict) and r.get("passed")]
    ok = bool(data.get("export_valid")) and int(vr.get("n_pass") or 0) >= 2
    if not ok or not passed:
        return None
    sel = vr.get("selected_episode")
    wb = data.get("winner_binding") or {}
    need = (sel in (None, "")) or (str(sel) not in passed)
    if not need:
        return None
    return {
        "path": str(path.relative_to(ROOT)),
        "old_selected_episode": sel,
        "new_selected_episode": passed[0],
        "passed": passed,
        "winner_episode_id_kept": wb.get("episode_id"),
    }


def _apply(data: dict) -> dict:
    vr = dict(data.get("verification_results") or {})
    passed = [str(r.get("episode_id")) for r in (vr.get("runs") or [])
              if isinstance(r, dict) and r.get("passed")]
    vr["selected_episode"] = passed[0]
    # n_pass 由列表**重算** ✓ (md §04: 不能来自搜索)
    vr["n_pass"] = len(passed)
    data["verification_results"] = vr
    wb = dict(data.get("winner_binding") or {})
    if wb:
        wb["verified_episode_ids"] = passed
        wb["binding_note"] = ("winner=搜索最优(事实, 保留 ✗); 交付以 selected_episode/"
                              "verified_episode_ids 为准 ✓ (md §04)")
        data["winner_binding"] = wb
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入 (默认 dry-run)")
    ap.add_argument("--exclude", action="append", default=[],
                    help="排除的 run 目录名 (可多次; 如正在使用的 s31/verify_s31 ✓)")
    args = ap.parse_args()
    _EX = set(str(x) for x in (args.exclude or []))

    def _excluded(_pl):
        return any(("/%s/" % _e) in ("/" + _pl["path"] + "/") for _e in _EX)

    plans, npz_plans = [], []
    for p in sorted(ROOT.glob("output/*/stage1/stage1_grasp_recipe.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        pl = _plan_one(p, d)
        if pl and not _excluded(pl):
            pl["_data"] = d
            plans.append(pl)

    # npz 内嵌 (真源 ✓)
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        np = None
    if np is not None:
        for p in sorted(ROOT.glob("output/*/stage1/best_grasp.npz")):
            try:
                with np.load(p, allow_pickle=True) as z:
                    if "stage1_recipe_json" not in z:
                        continue
                    d = json.loads(str(z["stage1_recipe_json"]))
            except Exception:  # noqa: BLE001
                continue
            pl = _plan_one(p, d)
            if pl and not _excluded(pl):
                pl["_data"] = d
                npz_plans.append(pl)

    print("=== JSON recipe 待回填 %d 个 ===" % len(plans))
    for pl in plans:
        print("  %-42s sel %s -> %s | passed=%s | winner(保留)=%s"
              % (pl["path"], pl["old_selected_episode"], pl["new_selected_episode"],
                 pl["passed"], pl["winner_episode_id_kept"]))
    print("=== npz 内嵌 待回填 %d 个 ===" % len(npz_plans))
    for pl in npz_plans:
        print("  %-42s sel %s -> %s" % (pl["path"], pl["old_selected_episode"],
                                        pl["new_selected_episode"]))

    if not args.apply:
        print("\n[dry-run] 未写入。加 --apply 实际执行。")
        return 0

    bak = ROOT / "trimesh" / ("backfill_%s" % time.strftime("%Y%m%d-%H%M%S"))
    bak.mkdir(parents=True, exist_ok=True)
    n = 0
    for pl in plans:
        src = ROOT / pl["path"]
        shutil.copy2(src, bak / (src.parent.name + "__" + src.name))     # 备份 ✓ (不真删 ✓)
        src.write_text(json.dumps(_apply(pl["_data"]), indent=2, sort_keys=True),
                       encoding="utf-8")
        n += 1
    for pl in npz_plans:
        src = ROOT / pl["path"]
        shutil.copy2(src, bak / (src.parent.name + "__" + src.name))     # 备份 ✓
        newd = _apply(pl["_data"])
        try:
            with np.load(src, allow_pickle=True) as z:
                arrs = {k: z[k] for k in z.files if k != "stage1_recipe_json"}
            arrs["stage1_recipe_json"] = np.array(json.dumps(newd))
            np.savez_compressed(src, **arrs)                             # 同步内嵌真源 ✓
            n += 1
        except Exception as e:  # noqa: BLE001
            print("  ✗ npz 回填失败 %s: %s" % (src, e))
    print("\n✓ 已回填 %d 个文件; 原文件备份于 %s" % (n, bak))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
