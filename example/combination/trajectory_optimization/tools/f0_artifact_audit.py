#!/usr/bin/env python
"""★ md 06.4-f0 (2026-09-30) 【固定产物身份与负 q 语义 · 0 物理】 ✓

做四件事 ✓ (只读 npz/JSON/URDF 模板, 不改任何原件 ✗):
  ① 三处一致性比对: npz 外层 / npz 内嵌 recipe / 磁盘 JSON
  ② **分开**算出各 q 语义 (f0 明令: 不能用整条 `min(q)` 代替指定 run 的终态 ✗):
       · q_nominal          = 获胜候选的名义闭合目标 (原始字段 ✓)
       · q_effective_min    = 全段实际下发最小值 (仅**诊断**用途 ✓)
       · q_close_end        = close 段末命令 (index = close-1 ✓)
       · q_settle_end       = **settle 段末命令** (index = close+settle-1 ✓) ← anchor 应取此 ✓
  ③ 读 URDF 模板真实关节 limit ⇒ 判"实际 q 越界"与"drive 参考越界"的**区别** ✓
  ④ 输出 `artifact_migration_audit.json` + `joint_command_policy.json`

⚠ f0 红线 ✓: "未定义(负 drive)则**拒绝该控制模式**, 不自动改 URDF 或钳成 0" ✗
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _urdf_finger_limits() -> dict:
    """从 URDF **模板** 读手指关节 limit (本机不落静态 .urdf ✓)"""
    out = {"lower": None, "upper": None, "source": None}
    p = ROOT / "models" / "urdf_templates.py"
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        ms = re.findall(r'<limit\s+lower="([-\d.]+)"\s+upper="([-\d.]+)"', txt)
        if ms:
            out["lower"] = float(ms[0][0])
            out["upper"] = float(ms[0][1])
            out["source"] = "models/urdf_templates.py (<limit> 首个匹配)"
    except Exception as e:  # noqa: BLE001
        out["source"] = "读取失败: %s" % e
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="output/s102_deep4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    st = ROOT / a.run_dir / "stage1"
    pj = st / "stage1_grasp_recipe.json"
    pn = st / "best_grasp.npz"
    out_dir = Path(a.out) if a.out else (ROOT / a.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aud = {"run_dir": a.run_dir, "checks": [], "q_semantics": {},
           "identity": {}, "verdict": {}}

    # ── ① 三处一致性 ──
    J = json.loads(pj.read_text())
    with np.load(pn, allow_pickle=True) as z:
        qpos_outer = float(z["gripper_qpos"])
        inner = json.loads(str(z["stage1_recipe_json"]))
    q_json = float(J.get("q_grasp"))
    q_inner = float(inner.get("q_grasp"))
    aud["identity"] = {
        "json_q_grasp": q_json, "npz_outer_gripper_qpos": qpos_outer,
        "npz_inner_q_grasp": q_inner,
        "revision": J.get("revision"),
        "export_valid": bool(J.get("export_valid")),
        "n_pass": int((J.get("verification_results") or {}).get("n_pass") or 0),
        "consistent": bool(q_json == qpos_outer == q_inner),
    }
    aud["checks"].append({
        "name": "三处一致性 (JSON vs npz外层 vs npz内嵌)",
        "ok": aud["identity"]["consistent"],
        "detail": "q_json=%s qpos_outer=%s q_inner=%s" % (q_json, qpos_outer, q_inner),
    })

    # ── ② 各 q 语义 (分开算 ✓ 不用 min 代替终态 ✗) ──
    aqc = np.asarray(J.get("actual_q_commands"), dtype=float).ravel()
    qc = np.asarray(J.get("q_cmd"), dtype=float).ravel()
    cc = int(J.get("close_controls") or 0)
    sc = int(J.get("settle_controls") or 0)
    i_close_end = max(0, cc - 1)
    i_settle_end = max(0, cc + sc - 1)
    sem = {
        "q_nominal(候选名义)": q_json,
        "q_effective_min(全段最小·仅诊断 ✗ 不作 anchor)":
            float(aqc.min()) if aqc.size else None,
        "q_close_end_cmd(index=%d)" % i_close_end:
            float(aqc[i_close_end]) if aqc.size > i_close_end else None,
        "q_settle_end_cmd(index=%d·★anchor 应取此)" % i_settle_end:
            float(aqc[i_settle_end]) if aqc.size > i_settle_end else None,
        "q_actual_settle_end(实测)": "MISSING(本 artifact 未逐指存 q_actual ⇒ 标注缺测 ✓)",
        "n_points": int(aqc.size),
        "close_controls": cc, "settle_controls": sc,
    }
    aud["q_semantics"] = sem
    aud["checks"].append({
        "name": "q 语义分离 (名义/全段min/close末/settle末 ✓)",
        "ok": True,
        "detail": "settle 末 = %s (anchor 应取此 ✓); 全段 min = %s (仅诊断 ✗)"
                  % (sem["q_settle_end_cmd(index=%d·★anchor 应取此)" % i_settle_end],
                     sem["q_effective_min(全段最小·仅诊断 ✗ 不作 anchor)"]),
    })

    # ── ③ URDF limit 与越界 ──
    lim = _urdf_finger_limits()
    lo = lim.get("lower")
    qmin = float(aqc.min()) if aqc.size else float("nan")
    viol = (lo is not None) and (qmin < lo - 1e-9)
    aud["urdf_limits"] = lim
    aud["checks"].append({
        "name": "实际 q 越界 (URDF lower=0 ⇒ q<0 越界 ✗)",
        "ok": (not viol),
        "detail": ("URDF lower=%s upper=%s; 实际下发 min=%s ⇒ %s"
                   % (lo, lim.get("upper"), round(qmin, 6),
                      "越界 ✗" if viol else "未越界 ✓")),
    })
    aud["checks"].append({
        "name": "负 drive 是否在 spec 中声明",
        "ok": False,
        "detail": ("**未声明** ✗ —— f0: '未定义则**拒绝该控制模式**, "
                   "不自动改 URDF 或钳成 0' ⇒ 该 artifact 的负 q 控制模式应被拒绝 ✓"),
    })

    aud["verdict"] = {
        "anchor_should_be": sem["q_settle_end_cmd(index=%d·★anchor 应取此)" % i_settle_end],
        "min_q_violates_urdf": bool(viol),
        "negative_drive_declared": False,
        "action": ("REJECT_CONTROL_MODE(负 drive 未声明 ✗)"
                   if viol else "OK"),
        "note": ("先前用 `min(q)` 当 anchor 是 f0 明令禁止的 ✗; "
                 "正确 anchor = settle 段末命令 ✓"),
    }

    (out_dir / "artifact_migration_audit.json").write_text(
        json.dumps(aud, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    pol = {
        "schema": "joint_command_policy/v1",
        "finger_joint_type": "prismatic",
        "urdf_limits": lim,
        "policy": {
            "nominal_q_range": [lo if lo is not None else "UNKNOWN",
                                lim.get("upper") if lim.get("upper") is not None
                                else "UNKNOWN"],
            "negative_drive": {
                "declared": False,
                "decision": "REJECT",
                "reason": ("f0: 允许负 drive 作限位外预紧必须**在 spec 中明确**: "
                           "范围 / 力限与行程保护 / 最终 drive 读回; "
                           "不能把 env 过闭合上限当关节限位, 也不得自动改 URDF 或钳成 0"),
                "required_to_enable": [
                    "_S1_ALLOW_NEGATIVE_DRIVE=1 (显式开关)",
                    "声明 q 下限 (≥ -? mm) 与力限、行程保护",
                    "最终 drive 读回校验 (末子步前记录六轴与双指 target)",
                ],
            },
            "q_grasp_anchor_source": "settle 段末实际下发命令 (index=close+settle-1)",
            "forbidden": ["用整条 min(q) 当 anchor", "原地改证书使校验通过",
                          "改 URDF 限位或把负 q 静默钳成 0"],
        },
    }
    (out_dir / "joint_command_policy.json").write_text(
        json.dumps(pol, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== f0 审计 (%s) ===" % a.run_dir)
    for c in aud["checks"]:
        print("  [%s] %s\n        %s" % ("OK " if c["ok"] else "FAIL", c["name"], c["detail"]))
    print("  verdict:", json.dumps(aud["verdict"], ensure_ascii=False)[:200])
    print("交付: %s / %s" % (out_dir / "artifact_migration_audit.json",
                             out_dir / "joint_command_policy.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
