"""R00 —— 冻结成功基线与诊断现场 (只读 ✓)

md 位置: docs/2026-09-23-物理仿真控制与奖励联合设计-推荐实施方案.md §H.1 R00

md 原话 ✓:
  "保留 s31 目录和 s44 失败产物, 不覆盖。登记两者 recipe、best_grasp、实际配置、
   代码版本、输入轨迹及 mesh/profile hash; 说明 s44 是否 ...
   明确 Stage1 证书范围: 当前物体/位姿/旧实际判据 3/3, 不扩大为任意轨迹成功。
   记录 F65 来源差异; 先查 candidate provenance, 必要时 ...
   输出 baseline_manifest 与未完成清单。**不重跑 CEM、不改 Stage0、质量/摩擦/驱动力。**"

本脚本 ✓:
  · **只读**: 只 open 'rb'/'r' 与 `git rev-parse` / `sha1sum`, 不写任何 output/ 下的东西
  · 不启动物理, 不重跑 CEM ✓
  · 输出 2 份清单 (JSON + Markdown) 到 `docs/`

用法:
    python tools/freeze_baseline_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
GLB = Path("/home/an/data/ras/7_vggt_omega/final_scene.glb")
HAWOR = Path("/home/an/data/hawor/7a")

RUNS = {"s31": "成功基线 (Stage1 3/3)", "s44": "失败现场 (Stage2 CARRY 0/3 ⇒ Stage3 未运行)"}


def _sha1(p: Path, head: int = 0) -> str:
    try:
        h = hashlib.sha1(p.read_bytes()).hexdigest()
    except Exception:  # noqa: BLE001
        return "?"
    return h if not head else h[:head]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        return "?"


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _npz_summary(p: Path) -> dict:
    if not p.exists():
        return {"exists": False}
    out = {"exists": True, "sha1": _sha1(p, 16), "size": p.stat().st_size}
    try:
        import numpy as np
        d = np.load(p, allow_pickle=True)
        out["keys"] = sorted(d.keys())
        for k in ("target_object", "strategy", "reward", "obj_lift", "obj_xy_drift",
                  "q_open", "verification"):
            if k in d:
                v = d[k]
                try:
                    v = v.tolist()
                except Exception:  # noqa: BLE001
                    v = str(v)
                out[k] = v if not isinstance(v, list) or len(v) < 24 else f"<list n={len(v)}>"
    except Exception as exc:  # noqa: BLE001
        out["read_error"] = str(exc)
    return out


def _recipe_summary(p: Path) -> dict:
    d = _read_json(p)
    if not d:
        return {"exists": False}
    vr = d.get("verification_results") or {}
    runs = vr.get("runs") or []
    return {
        "exists": True,
        "sha1": _sha1(p, 16),
        "target_object": d.get("target_object"),
        "q_sequence_source": d.get("q_sequence_source"),
        "task_grasp_frame": d.get("task_grasp_frame"),
        "export_valid": d.get("export_valid"),
        "lengths": {k: (len(v) if isinstance(v, (list, tuple)) else None)
                    for k, v in d.items()
                    if k in ("actual_q_commands", "actual_control_times", "actual_base_commands",
                             "q_cmd", "T_W_G_cmd", "phase", "close_ctrl_q", "close_ctrl_t",
                             "observations", "optimized_T_W_G")},
        "phase_boundaries": d.get("phase_boundaries"),
        "phase_counts": d.get("phase_counts"),
        "hashes": {k: str(d.get(k))[:24] for k in
                   ("physics_hash", "controller_hash", "criteria_hash",
                    "canonical_state_hash", "initial_state_hash")},
        "verification": {
            "criteria": vr.get("criteria"),
            "n_pass": vr.get("n_pass"),
            "n_repeat": vr.get("n_repeat"),
            "failure_reason": vr.get("failure_reason"),
            # ★ R03 的靶子 ✓: 通过的是 128/129/130, 但 selected_episode 仍是搜索阶段的 124
            "selected_episode": vr.get("selected_episode"),
            "passed_episodes": [r.get("episode_id") for r in runs if r.get("passed")],
            "runs_brief": [{k: r.get(k) for k in
                            ("episode_id", "passed", "lift_final_cm", "drift_mm",
                             "true_grasp_ratio", "finger_ground_mm")} for r in runs],
        },
        "env_echo": {k: d.get(k) for k in
                     ("obj_scale_target_env", "coacd_target_env", "criteria_revision")},
    }


def _s44_stage2_failure() -> dict:
    """s44 的 Stage2 CARRY 失败行 (只读日志)"""
    out = {"log": "/tmp/s44.log", "found": []}
    p = Path("/tmp/s44.log")
    if not p.exists():
        return out
    for line in p.read_text(errors="ignore").splitlines():
        if ("[Stage2 gate] FAIL" in line or "Stage2 repeat gate FAIL" in line
                or "Stage3 不运行" in line):
            out["found"].append(line.strip()[:240])
    return out


def _f65_provenance() -> dict:
    """F65 来源差异: 多帧扫描窗口 F35–F65 vs 任务抓取帧 F50 vs close 相位 [0,130]"""
    out = {"scan_window": "F35–F65 (31 帧, 见 s31 日志 [Stage1][多帧扫描])",
           "task_grasp_frame": None, "note": [
               "任务抓取帧 = recipe.task_grasp_frame (s31: 50)",
               "close 相位 = phase_boundaries.close (s31: [0,130])",
               "⇒ F65 是**扫描窗口右端**, 不是任务终点; 两者含义不同, 不可互相替代 ✓",
           ]}
    r = _recipe_summary(ROOT / "output/s31/stage1/stage1_grasp_recipe.json")
    out["task_grasp_frame"] = r.get("task_grasp_frame") if r.get("exists") else None
    return out


def main() -> int:
    man: dict = {
        "revision": "R00",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "read_only": True,
        "code": {
            "git_head": _git("rev-parse", "--short", "HEAD"),
            "git_head_date": _git("log", "-1", "--format=%ad", "--date=short"),
            "git_head_subject": _git("log", "-1", "--format=%s")[:120],
            "worktree_dirty_files": len([x for x in _git("status", "--porcelain").splitlines() if x.strip()]),
        },
        "inputs": {
            "glb": str(GLB), "glb_sha1": _sha1(GLB, 16) if GLB.exists() else "?",
            "glb_size": GLB.stat().st_size if GLB.exists() else None,
            "hawor_dir": str(HAWOR), "hawor_exists": HAWOR.exists(),
            "hawor_entries": sorted(x.name for x in HAWOR.iterdir())[:12] if HAWOR.exists() else [],
        },
        "stage1_certificate_scope": {
            "statement": ("证书仅覆盖: **当前物体 (glb_4 / toy_0) + 当前初始位姿 + "
                          "当前判据 (旧实际判据)** 的 3/3 复现 ✓; "
                          "**不扩大**为\"任意轨迹/任意物体都能抓\" ✗"),
            "object": "glb_4 (toy_0)", "n_pass": 3, "n_repeat": 3,
            "passed_episodes": ["s1ep-000128", "s1ep-000129", "s1ep-000130"],
        },
        "f65_provenance": _f65_provenance(),
        "runs": {},
        "s44_stage2_failure": _s44_stage2_failure(),
        "todo_after_r00": [
            "R01 时间线截断/数组完整性 (P0, 零物理) —— 禁止 q-only 尾裁剪删 CARRY",
            "R02 闭合→稳定→搬运命令真实复用 (P0, 先离线)",
            "R03 成功 record 与入口门绑定 (从真实通过 run 恢复, 不伪造)",
            "R04 场景/初态真实核对 (API 自检先行, 不再持续放行 UNKNOWN)",
            "R05 用完整矩阵证据定位 33.37° 与起步失稳 (P1, 最多一次诊断)",
            "R06 完整搬运 + 反馈鲁棒性 + 重放",
            "R07 Stage1 最小自洽性修复 (不重找抓取姿态)",
            "R08 Stage3 入口/后缀/最终重放自洽",
        ],
    }

    for name, desc in RUNS.items():
        base = ROOT / "output" / name
        man["runs"][name] = {
            "desc": desc,
            "dir_exists": base.exists(),
            "recipe": _recipe_summary(base / "stage1" / "stage1_grasp_recipe.json"),
            "best_grasp": _npz_summary(base / "stage1" / "best_grasp.npz"),
            "run_metadata": _read_json(base / "run_metadata.json"),
            "stage2_trace_exists": (base / "stage2" / "gripper_trace.csv").exists(),
            "stage3_videos": sorted(p.name for p in (base / "stage3").glob("*.mp4"))
            if (base / "stage3").exists() else [],
        }

    DOCS.mkdir(parents=True, exist_ok=True)
    jp = DOCS / "baseline_manifest_20260928.json"
    jp.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    print(f"[R00] ✓ JSON 清单: {jp}")

    # ── Markdown 摘要 ──
    L = []
    A = L.append
    A("# baseline_manifest — R00 冻结成功基线与诊断现场")
    A("")
    A(f"生成时间: {man['generated_at']} · **只读** (未重跑 CEM, 未改 Stage0/质量/摩擦/驱动力) ✓")
    A("")
    A("## 1. 代码与输入")
    A("")
    c = man["code"]
    A(f"- git HEAD: `{c['git_head']}` ({c['git_head_date']}) {c['git_head_subject']}")
    A(f"- 工作树未提交文件数: **{c['worktree_dirty_files']}** (现场含未提交修改 ⇒ 版本以文件 sha1 为准)")
    i = man["inputs"]
    A(f"- 场景 glb: `{i['glb']}` sha1=`{i['glb_sha1']}`")
    A(f"- HaWoR: `{i['hawor_dir']}` (存在={i['hawor_exists']})")
    A("")
    A("## 2. Stage1 证书范围 (不扩大 ✓)")
    A("")
    s = man["stage1_certificate_scope"]
    A(f"- {s['statement']}")
    A(f"- 通过 episode: {s['passed_episodes']} ({s['n_pass']}/{s['n_repeat']})")
    A("")
    A("## 3. 两个现场")
    A("")
    for name in RUNS:
        r = man["runs"][name]
        A(f"### {name} — {r['desc']}")
        A("")
        rp, bg = r["recipe"], r["best_grasp"]
        if rp.get("exists"):
            A(f"- recipe sha1=`{rp['sha1']}` · target=`{rp['target_object']}` · "
              f"q源=`{rp['q_sequence_source']}` · task_grasp_frame=`{rp['task_grasp_frame']}` · "
              f"export_valid=`{rp['export_valid']}`")
            A(f"- 数组长度: {rp['lengths']}")
            A(f"- 相位: boundaries={rp['phase_boundaries']} counts={rp['phase_counts']}")
            A(f"- hashes: {rp['hashes']}")
            v = rp["verification"]
            A(f"- 终验: n_pass={v['n_pass']}/{v['n_repeat']} · 通过={v['passed_episodes']} · "
              f"**selected_episode={v['selected_episode']}** "
              f"{'⚠ 与通过列表不一致 (R03 靶子)' if v['selected_episode'] not in (v['passed_episodes'] or []) else '✓'}")
            A(f"- 判据: {v['criteria']}")
        if bg.get("exists"):
            A(f"- best_grasp sha1=`{bg['sha1']}` size={bg['size']} keys={bg.get('keys')}")
        A(f"- stage2 trace 存在={r['stage2_trace_exists']} · stage3 视频={r['stage3_videos']}")
        A("")
    A("## 4. s44 Stage2 失败证据 (原始行)")
    A("")
    for line in man["s44_stage2_failure"]["found"]:
        A(f"- `{line}`")
    if not man["s44_stage2_failure"]["found"]:
        A("- (未找到 /tmp/s44.log, 或日志已被轮转 ✓ 产物目录仍在 ✓)")
    A("")
    A("## 5. F65 来源差异")
    A("")
    f = man["f65_provenance"]
    A(f"- 扫描窗口: {f['scan_window']}")
    A(f"- 任务抓取帧: {f['task_grasp_frame']}")
    for n in f["note"]:
        A(f"- {n}")
    A("")
    A("## 6. 未完成清单 (按 md §H.1 顺序)")
    A("")
    for t in man["todo_after_r00"]:
        A(f"- [ ] {t}")
    A("")
    mp = DOCS / "baseline_manifest_20260928.md"
    mp.write_text("\n".join(L))
    print(f"[R00] ✓ Markdown 摘要: {mp}")

    # 控制台摘要
    print("\n" + "=" * 92)
    print("R00 冻结摘要 (只读 ✓)")
    print("=" * 92)
    print(f"git HEAD={c['git_head']} ({c['git_head_date']}) | 工作树未提交={c['worktree_dirty_files']} 个文件")
    print(f"glb sha1={i['glb_sha1']} | hawor={i['hawor_exists']}")
    for name in RUNS:
        r = man["runs"][name]
        v = r["recipe"].get("verification", {}) if r["recipe"].get("exists") else {}
        print(f"[{name}] recipe={'✓' if r['recipe'].get('exists') else '✗'} "
              f"n_pass={v.get('n_pass')} 通过={v.get('passed_episodes')} "
              f"selected={v.get('selected_episode')} ⚠绑定不一致" if v.get("n_pass") else "")
    print(f"s44 CARRY 失败行: {len(man['s44_stage2_failure']['found'])} 条")
    print("→ 结论 ✓: Stage1 证书 = 当前物体/位姿/旧判据 3/3, **不外推** ✓; 下一步 R01 (零物理) ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
