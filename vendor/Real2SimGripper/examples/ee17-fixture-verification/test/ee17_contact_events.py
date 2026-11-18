#!/usr/bin/env python3
"""ee17_contact_events.py — 逐帧接触事件验收 (用户交付要求第 4 步)。

从 exec_trace.npz 提取关键接触事件帧 + 放稳判定:
  关键帧: 双指开始承载 / 离开起始支撑 / 首次碰环 / 真正落座 / 张爪 / 夹爪撤离
  放稳通过: 先落座再张爪, 撤爪后铁圈独立承重 ≥0.5 物理秒,
            漏斗位置/姿态稳定, 无地面接触、无夹爪接触。

trace 必需字段 (新版 cb 输出): obj/grip_target/frame/fps/phys_time/
  contact_impulses_Ns/contact_impulses_signed_z/object_dynamic/measured_gap_m/base
用法:
    python test/ee17_contact_events.py <trace.npz> [--out report.json]
"""
import argparse
import json
import os

import numpy as np


def quat_deg(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2 * np.arccos(np.clip(d, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rest-s", type=float, default=0.5,
                    help="铁圈独立承重时长 (物理秒, 默认 0.5)")
    ap.add_argument("--pos-tol-mm", type=float, default=3.0,
                    help="落座稳定位置漂移 (mm)")
    ap.add_argument("--rot-tol-deg", type=float, default=2.5,
                    help="落座稳定姿态漂移 (deg)")
    a = ap.parse_args()

    z = np.load(a.trace, allow_pickle=True)
    obj = np.asarray(z["obj"], np.float64)            # (N,7) pos+quat
    fps = float(z["fps"]) if "fps" in z.files else 30.0
    pt = (np.asarray(z["phys_time"], np.float64) if "phys_time" in z.files
          else np.arange(len(obj)) / fps)
    imp = np.asarray(z["contact_impulses_Ns"], np.float64)
    imp_z = (np.asarray(z["contact_impulses_signed_z"], np.float64)
             if "contact_impulses_signed_z" in z.files else None)
    dyn = np.asarray(z["object_dynamic"], bool)
    gt = np.asarray(z["grip_target"], np.float64)
    names = [str(x) for x in np.asarray(z["contact_names"])]
    i_f1, i_f2 = names.index("finger1"), names.index("finger2")
    i_sup, i_gnd = names.index("support"), names.index("ground")
    N = len(obj)

    ev = {}
    bil = (imp[:, i_f1] > 1e-7) & (imp[:, i_f2] > 1e-7)
    idx = np.flatnonzero(bil)
    ev["bilateral_hold_start"] = int(idx[0]) if len(idx) else None
    k0 = ev["bilateral_hold_start"]
    if k0 is not None:
        rise = obj[k0:, 2] - obj[k0, 2]
        left = np.flatnonzero(dyn[k0:] & (rise > 0.02))
        ev["leave_support"] = int(k0 + left[0]) if len(left) else None
    sup = imp[:, i_sup] > 1e-7
    idx = np.flatnonzero(sup)
    ev["first_ring_touch"] = int(idx[0]) if len(idx) else None
    seated = None
    if len(idx):
        need = int(0.1 * fps)
        run = 0
        for s in idx:
            run = run + 1 if (seated is None or s - seated == run) else 1
            if run >= need:
                seated = int(s - need + 1)
                break
    ev["ring_seated"] = seated
    open_t = gt >= 0.035
    idx = np.flatnonzero(open_t)
    ev["gripper_open"] = int(idx[0]) if len(idx) else None
    ev["order_ok"] = bool(ev["ring_seated"] is not None
                          and ev["gripper_open"] is not None
                          and ev["ring_seated"] < ev["gripper_open"])
    ev["independent_support_s"] = 0.0
    ev["rest_pass"] = False
    if ev["order_ok"]:
        k_o = ev["gripper_open"]
        tail = slice(k_o, N)
        indep = sup[tail] & ~bil[tail] & ~(imp[tail, i_gnd] > 1e-7)
        best = cur = 0
        for v in indep:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        ev["independent_support_s"] = float(best / fps)
        pos_drift = float(np.linalg.norm(
            obj[tail, :2] - obj[tail, :2].mean(0), axis=1).max() * 1000)
        rot_drift = max(quat_deg(obj[-1, 3:7], obj[q, 3:7]) for q in range(k_o, N))
        ev["pos_drift_mm"] = pos_drift
        ev["rot_drift_deg"] = rot_drift
        no_gnd = bool(np.all(imp[tail, i_gnd] <= 1e-7))
        ev["rest_pass"] = bool(ev["independent_support_s"] >= a.rest_s
                               and pos_drift <= a.pos_tol_mm
                               and rot_drift <= a.rot_tol_deg
                               and no_gnd)
    if imp_z is not None:
        ev["ring_signed_z_total"] = float(imp_z[:, i_sup].sum())
        ev["ring_signed_z_positive_s"] = float((imp_z[:, i_sup] > 0).sum() / fps)
        ev["ring_side_hit_s"] = float((imp_z[:, i_sup] < 0).sum() / fps)
        ev["ground_signed_z_total"] = float(imp_z[:, i_gnd].sum())
    ev["frames"] = N
    ev["phys_time_s"] = float(pt[-1]) if len(pt) else 0.0
    ev["fps"] = fps
    ev["pass"] = bool(ev["order_ok"] and ev["rest_pass"]
                      and (imp_z is None or ev["ring_side_hit_s"] < 0.5))
    out = a.out or os.path.join(os.path.dirname(a.trace), "contact_events.json")
    json.dump(ev, open(out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(ev, indent=1, ensure_ascii=False))
    print(("PASS" if ev["pass"] else "FAIL") + ": contact events -> " + out)


if __name__ == "__main__":
    main()
