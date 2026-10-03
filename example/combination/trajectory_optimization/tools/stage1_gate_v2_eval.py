"""任务03 离线评估器 —— 用**现有产物**按 md 的新口径判 Stage1 (零物理 ✓ 不改原码 ✓)

md 位置: §4 任务 03（Stage1 只保留一条执行语义）
md 原话 ✓:
  "旧'真夹持率'改名 `joint_block_ratio` **诊断**; 真实门检查**目标双指载荷、目标离桌、
   **固定 C0 的相对平移/旋转**和**任务进度**。v1/v2 先**并列报告**" ✓
  "搜索与终验共用一次解析的 criteria 对象; 明确单位, 不按数值大小猜 m/cm" ✓

数据来源 (实测确认 ✓, `output/<run>/geom_snapshot.jsonl`, 910 帧 × 7 相位)
  · `obj_pos_m` / `obj_quat`          —— 物体世界位姿 ✓
  · `finger1_pos_m` / `finger2_pos_m` —— 双指尖世界位置 ✓
  · `finger1_sum_imp_raw` / `finger2_sum_imp_raw` —— **双指法向冲量 = 载荷** ✓
  · `obj_supported` / `obj_supported_state`       —— **目标是否仍由桌面支撑** ✓
  · `q_cmd` / `q_act1` / `q_act2` / `left_nearest` / `right_nearest` ✓
  · `phase` ∈ {OPEN_READY, CLOSE_END, PRE_LIFT, LIFT+15/30/60/90} ✓

v2 四项门 (本脚本可算的 ✓) + 一项诊断
  1. `dual_finger_load`  目标双指载荷: 搬运段内 `f1_imp>0 且 f2_imp>0` 的帧占比 ≥ 阈值 ✓
  2. `off_table`         目标离桌: 抬升段 `obj_supported is False` 的帧占比 ≥ 阈值 ✓
  3. `c0_rigid`          固定 C0 相对位姿: C0 定义在**稳定末**(PRE_LIFT 末帧) **一次**,
                         之后每帧算平移/旋转偏差 ⇒ ≤ **5mm / 5°** (md §06 ✓)
                         ⚠ 诚实标注 ✓: 现场无"夹爪 6DOF 位姿"字段 ⇒ 用**双指尖中点**
                           近似夹爪原点 (搬运段夹爪姿态基本固定) ⇒ 平移项可靠 ✓,
                           旋转项用 `obj_quat` 相对 C0 的变化 ✓ (等价于相对夹爪的旋转变化)
  4. `progress`          任务进度: 抬升段物体 z 增量 ≥ 阈值 ✓
  诊断 `joint_block_ratio`: 旧"真夹持率" —— 用 trace 里两指同时被挡停的帧占比 ✓ (仅诊断 ✗)

用法 ✓
    python tools/stage1_gate_v2_eval.py --run output/s31
    python tools/stage1_gate_v2_eval.py --run output/s31 --json /tmp/s31_v2.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

# ★ 阈值集中一处 ⇒ 与 criteria 同源, 单位写在名字里 (md: 明确单位, 不猜 m/cm ✓)
GATE = {
    "dual_load_ratio_min": 0.90,      # 双指同时有载荷的帧占比
    "off_table_ratio_min": 0.90,      # 目标不再由桌面支撑的帧占比
    "c0_trans_mm_max": 5.0,           # md §06: 5mm
    "c0_rot_deg_max": 5.0,            # md §06: 5°
    "progress_cm_min": 3.0,           # md §03/§5.1: 与旧门同量级 (lift ≥3cm)
}

LIFT_PHASES = ("LIFT+15", "LIFT+30", "LIFT+60", "LIFT+90")


def _quat_to_R(q):
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def evaluate(run_dir: Path) -> dict:
    """★ 2026-09-28 **口径修正** ✓ (第一次我按"逐帧连续"算, 那是**错的** ✗)

    实测结构 ✓: `geom_snapshot.jsonl` = **每次 rollout 7 个检查点**的离散采样
      frame 序列固定 = [0(OPEN_READY), 130(CLOSE_END), 140(PRE_LIFT),
                        155/170/200/230(LIFT+15/30/60/90)] ✓
      910 行 = **130 次 rollout × 7 检查点** ⇒ 行号**不是**时间轴 ✗
    ⇒ 正确做法 ✓: 按 **7 行一组**切分, **组内**定义 C0 并比较 (同一 episode 内 ✓)
      ⚠ 跨组比较 (我第一版干的事) 会把"不同 rollout"混成一条轨迹 ⇒ 数值全错 ✗
    """
    snap_path = run_dir / "geom_snapshot.jsonl"
    trace_path = run_dir / "stage1" / "gripper_trace.csv"
    out: dict = {"run": str(run_dir), "data": {}}

    rows = []
    if snap_path.exists():
        for line in snap_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    out["data"]["snapshot_rows"] = len(rows)
    if not rows:
        out["error"] = f"缺 {snap_path} ⇒ 无法评估 v2 门 ✗"
        return out

    # ── 按 7 行一组 (一次 rollout 的 7 个检查点) 切分 ✓ ──
    GROUPS = 7
    groups = [rows[i:i + GROUPS] for i in range(0, len(rows) - GROUPS + 1, GROUPS)]
    valid = [g for g in groups
             if [int(r.get("frame", -1)) for r in g] == [0, 130, 140, 155, 170, 200, 230]]
    out["data"]["episodes(7-point groups)"] = len(valid)
    out["data"]["group_shape_ok"] = len(valid) == len(groups)
    if not valid:
        out["error"] = "没有一条 7 点完整 episode ⇒ 无法评估 ✗"
        return out

    def _obj_pos(r):
        return np.asarray([_f(x) for x in (r.get("obj_pos_m") or [np.nan] * 3)], dtype=np.float64)

    def _grip_origin(r):
        f1 = np.asarray([_f(x) for x in (r.get("finger1_pos_m") or [np.nan] * 3)], dtype=np.float64)
        f2 = np.asarray([_f(x) for x in (r.get("finger2_pos_m") or [np.nan] * 3)], dtype=np.float64)
        return (f1 + f2) / 2.0

    def _ok(v, lo=None, hi=None):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        if lo is not None and v < lo:
            return False
        if hi is not None and v > hi:
            return False
        return True

    per_ep, n_all_pass = [], 0
    agg = {"dual": [], "off": [], "tr": [], "rot": [], "prog": []}
    imp1, imp2 = [], []
    for g in valid:
        r_pre, r_lift = g[2], g[3:]                      # 140 = 稳定末 (定义 C0 ✓); 155.. = 抬升 ✓
        C0_t = _obj_pos(r_pre) - _grip_origin(r_pre)
        C0_R = _quat_to_R(r_pre.get("obj_quat") or [0, 0, 0, 1])

        dual = off = 0
        tr, rot = [], []
        for r in r_lift:
            a, b = _f(r.get("finger1_sum_imp_raw")), _f(r.get("finger2_sum_imp_raw"))
            imp1.append(a)
            imp2.append(b)
            dual += int(a > 0 and b > 0)
            off += int(bool(r.get("obj_supported")) is False)
            t = _obj_pos(r) - _grip_origin(r)
            if np.all(np.isfinite(t)):
                tr.append(float(np.linalg.norm(t - C0_t)) * 1000.0)
            rot.append(_rot_deg(C0_R, _quat_to_R(r.get("obj_quat") or [0, 0, 0, 1])))
        dual_r = dual / float(len(r_lift))
        off_r = off / float(len(r_lift))
        tr_max = max(tr) if tr else float("nan")
        rot_max = max(rot) if rot else float("nan")
        prog = (_obj_pos(r_lift[-1])[2] - _obj_pos(r_pre)[2]) * 100.0

        ok = {
            "目标双指载荷": _ok(dual_r, lo=GATE["dual_load_ratio_min"]),
            "目标离桌": _ok(off_r, lo=GATE["off_table_ratio_min"]),
            "固定C0平移≤5mm": _ok(tr_max, hi=GATE["c0_trans_mm_max"]),
            "固定C0旋转≤5°": _ok(rot_max, hi=GATE["c0_rot_deg_max"]),
            "任务进度≥3cm": _ok(prog, lo=GATE["progress_cm_min"]),
        }
        all_ok = all(v is True for v in ok.values())
        n_all_pass += int(all_ok)
        agg["dual"].append(dual_r)
        agg["off"].append(off_r)
        agg["tr"].append(tr_max)
        agg["rot"].append(rot_max)
        agg["prog"].append(prog)
        per_ep.append({"all_pass": all_ok, **{k: ("PASS" if v else "FAIL") if v is not None else "?"
                                             for k, v in ok.items()}})

    def _stat(key, pct=True):
        v = np.asarray([x for x in agg[key] if np.isfinite(x)], dtype=np.float64)
        if v.size == 0:
            return None
        d = {"mean": round(float(v.mean()), 3), "max": round(float(v.max()), 3),
             "min": round(float(v.min()), 3), "pass_rate": round(float(np.mean(
                 [1 if p["all_pass"] else 0 for p in per_ep])), 3)}
        return d

    out["v2_gates_per_episode"] = {
        "episodes": len(valid),
        "all_5_gates_pass_episodes": n_all_pass,
        "v2_pass_rate": round(n_all_pass / float(len(valid)), 3),
        "dual_finger_load_ratio": _stat("dual"),
        "off_table_ratio": _stat("off"),
        "c0_trans_mm_max": _stat("tr"),
        "c0_rot_deg_max": _stat("rot"),
        "progress_cm": _stat("prog"),
    }
    out["imp_ranges"] = {"f1": [round(float(np.nanmin(imp1)), 2), round(float(np.nanmax(imp1)), 2)],
                         "f2": [round(float(np.nanmin(imp2)), 2), round(float(np.nanmax(imp2)), 2)]}

    # ── 诊断: joint_block_ratio (旧"真夹持率", 仅诊断 ✗) ──
    jbr = float("nan")
    if trace_path.exists():
        with open(trace_path) as fh:
            tr_rows = list(csv.DictReader(fh))
        if tr_rows:
            q1 = np.asarray([_f(r.get("finger1_q")) for r in tr_rows])
            q2 = np.asarray([_f(r.get("finger2_q")) for r in tr_rows])
            gap = np.asarray([_f(r.get("finger_gap_m")) for r in tr_rows])
            seg = slice(int(0.4 * len(tr_rows)), len(tr_rows))
            blocked = (np.abs(np.diff(q1[seg])) < 1e-6) & (np.abs(np.diff(q2[seg])) < 1e-6)
            jbr = float(np.mean(blocked)) if len(blocked) else float("nan")
            out["data"]["trace_frames"] = len(tr_rows)
            out["data"]["gap_mm"] = {"close": round(float(gap[int(0.38 * len(tr_rows))]) * 1000, 1),
                                     "lift_end": round(float(gap[-1]) * 1000, 1)}
    out["diagnostic_only"] = {"joint_block_ratio(旧'真夹持率')": round(jbr, 3)}

    # ── v1 (旧判据) 对照: 直接读 recipe 的 verification ✓ ──
    rec_p = run_dir / "stage1" / "stage1_grasp_recipe.json"
    if rec_p.exists():
        try:
            rec = json.loads(rec_p.read_text())
            v1 = rec.get("verification_results") or {}
            out["v1_legacy"] = {"n_pass": v1.get("n_pass"), "n_repeat": v1.get("n_repeat"),
                                "selected_episode": v1.get("selected_episode"),
                                "lift_cm": [r.get("lift_final_cm") for r in (v1.get("runs") or [])],
                                "true_grasp_ratio": [r.get("true_grasp_ratio")
                                                     for r in (v1.get("runs") or [])]}
        except Exception:  # noqa: BLE001
            pass
    out["v2_verdict"] = {
        "note": ("v2 门逐 **episode** 判定 (每次 rollout 的 7 检查点, C0 在**组内**稳定末定义 ✓); "
                 "v1 = md §03 要降级为诊断的旧口径 ✓"),
        "v2_pass_rate": out["v2_gates_per_episode"]["v2_pass_rate"],
        "any_episode_all_pass": n_all_pass > 0,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run 目录 (如 output/s31)")
    ap.add_argument("--json", default=None, help="把结果写到此 JSON 路径")
    args = ap.parse_args()

    rep = evaluate(Path(args.run))
    print("=" * 92)
    print(f"【任务03 离线评估】run={rep['run']}   (零物理 ✓ 只读产物 ✓)")
    print("=" * 92)
    if "error" in rep:
        print("✗", rep["error"])
        return 1
    print("数据:", json.dumps(rep["data"], ensure_ascii=False))
    if "v1_legacy" in rep:
        print("\nv1 (旧判据, md §03 要求降级为诊断 ✓):",
              json.dumps(rep["v1_legacy"], ensure_ascii=False))
    g = rep["v2_gates_per_episode"]
    print(f"\nv2 门 (md §03 新口径 ✓, 按 episode 判定):")
    print(f"  episodes = {g['episodes']} | **5 项门全过 = {g['all_5_gates_pass_episodes']} "
          f"⇒ v2 通过率 = {g['v2_pass_rate']:.1%}**")
    for k in ("dual_finger_load_ratio", "off_table_ratio", "c0_trans_mm_max",
              "c0_rot_deg_max", "progress_cm"):
        s = g.get(k)
        if s:
            print(f"  {k:<24} mean={s['mean']:<10} max={s['max']:<10} min={s['min']}")
    print("\n手指载荷范围:", json.dumps(rep.get("imp_ranges", {}), ensure_ascii=False))
    print("仅诊断 (不作门 ✗):", json.dumps(rep["diagnostic_only"], ensure_ascii=False))
    print(f"\n阈值: {json.dumps(GATE, ensure_ascii=False)}")
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(f"✓ 已写 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
