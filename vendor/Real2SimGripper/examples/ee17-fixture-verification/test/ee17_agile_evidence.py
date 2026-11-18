#!/usr/bin/env python3
"""ee17_agile_evidence.py — 步骤④ 验收证据汇编 (用户配方)。

汇集: AGILE 链产物哈希、mocap 锚定配准误差、600 帧释放窗 A/B (先验代理 vs
AGILE 精细几何)、执行层证据 (底盘到站/躯干峰值/双指承载/重力接管/松爪时序/
实际加载)、FPV/动图清单 + 复现命令。输出 results/large_range/ee17_agile_evidence.json
+ 一份人读 MD。

用法 (robotwin_curobo env): python test/ee17_agile_evidence.py
"""
import hashlib
import json
import os

import numpy as np

ROOT = "/path/to/Real2SimGripper"
LR = f"{ROOT}/results/large_range"
DEMO = f"{ROOT}/demo_output/large_range"
WIN = "/path/to/project_data/mano_ws/EgoFound3R/outputs/ee17_win"
AGILE_DIR = "/path/to/project_data/mano_ws/AGILE/output/ee17_release_s"


def sha256(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(n), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def exec_layer(tag):
    """从 exec_trace 提取执行层证据。"""
    z = np.load(f"{LR}/{tag}_exec_trace.npz", allow_pickle=True)
    obj = z["obj"]; grip = z["grip"]; base = z["base"]; q = z["q"]
    imp = z["contact_impulses_Ns"]; dyn = z["object_dynamic"]
    gap = z["measured_gap_m"]; cmd = z["cmd_base"]
    live = np.isfinite(obj[:, 0])
    k_lift = int(np.argmax(dyn)) if dyn.any() else -1          # 重力接管帧
    fin = (imp[:, :2] > 1e-7)
    bil = fin.all(1)
    # 松爪时序: 双指接触结束帧 (最后一次双侧接触) 与之后物体离手
    bil_idx = np.flatnonzero(bil)
    k_release = int(bil_idx[-1]) if len(bil_idx) else -1
    # 实际加载: 接管后双指冲量
    seg = slice(k_lift, k_release if k_release > k_lift else len(fin))
    load = imp[seg, :2].sum(1)
    # 底盘到站: cmd_base = [abs_x, abs_y, yaw, frame] —— 末帧指令 vs 实际底盘
    arrive_mm = float(np.linalg.norm(cmd[-1, :2] - base[-1, :2]) * 1000)
    # 躯干峰值: JSON 已有 torso 段; 这里从 q 取臂根 3 关节幅值参考
    return {
        "frames": int(len(obj)),
        "gravity_handoff_frame": k_lift,
        "bilateral_contact_frames": int(bil.sum()),
        "release_frame_last_bilateral": k_release,
        "release_open_gap_m": (float(np.nanmean(gap[k_release:k_release + 30]))
                               if k_release > 0 else None),
        "finger_load_mean_N": float(load.mean()) if len(load) else None,
        "finger_load_peak_N": float(load.max()) if len(load) else None,
        "base_arrival_err_mm": arrive_mm,
        "base_travel_m": float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1).sum()),
        "object_held_frames": int(((np.linalg.norm(np.diff(obj[live, :3], axis=0), axis=1)
                                    < 0.02).mean() * len(obj)) if live.sum() > 2 else 0),
    }


def main():
    ev = {"run": "ee17_agile (SAM3D→AGILE→R1, mocap-anchored)"}

    # ---- 1) 产物哈希 ----
    files = {
        "targets": f"{ROOT}/dataset/ee17_mocap/robot/targets.npz",
        "object_agile": f"{WIN}/object_instance_agile.npz",
        "registration": f"{WIN}/agile_ee17_registration.json",
        "cb": f"{ROOT}/scripts/cb_large_range_sim.py",
        "lrt": f"{ROOT}/scripts/large_range_track.py",
        "bridge": f"{ROOT}/scripts/agile_ee17_bridge.py",
        "subset_map": f"{AGILE_DIR}/subset_frame_map.npy",
        "agile_mesh": f"{AGILE_DIR}/runs/opt/aligned/object_mesh.obj",
    }
    ev["sha256"] = {k: sha256(v) for k, v in files.items() if os.path.exists(v)}

    # ---- 2) 配准误差 (桥接 JSON) ----
    ev["registration"] = json.load(open(files["registration"]))

    # ---- 3) 600 帧释放窗 A/B ----
    def acc_of(tag):
        d = json.load(open(f"{LR}/{tag}_large_range.json"))
        o = d.get("object") or {}
        return {"grasped": o.get("grasped"), "lift_mm": o.get("lift_mm"),
                "criteria": o.get("acceptance", {}).get("criteria"),
                "phys_pos_err_mm": d.get("phys_pos_err_mm", {}).get("mean"),
                "base_travel_m": d.get("base", {}).get("travel_m"),
                "torso": d.get("torso")}
    ev["window600"] = {
        "prior_proxy_object": acc_of("ee17_posture_fix_release_after"),
        "agile_geometry": acc_of("ee17_posture_agile_object"),
        "agile_geometry_before": (acc_of("ee17_posture_agile_before")
                                  if os.path.exists(f"{LR}/ee17_posture_agile_before_large_range.json")
                                  else "pending"),
    }

    # ---- 4) 执行层证据 ----
    ev["exec_layer"] = {"agile_geometry": exec_layer("ee17_posture_agile_object")}
    if os.path.exists(f"{LR}/ee17_posture_agile_before_exec_trace.npz"):
        ev["exec_layer"]["agile_geometry_before"] = exec_layer("ee17_posture_agile_before")

    # ---- 5) 1050 帧回归 (prior archive, 哈希复核) ----
    ev["regression1050"] = {
        "task": "ee17_posture_fix_archive_after (prior, 1050 帧 file-016 全段)",
        "trace": f"{LR}/ee17_posture_fix_archive_after_exec_trace.npz",
        "json": f"{LR}/ee17_posture_fix_archive_after_large_range.json",
        "trace_sha256_16": sha256(f"{LR}/ee17_posture_fix_archive_after_exec_trace.npz"),
    }

    # ---- 6) 交付物清单 + 复现命令 ----
    ev["deliverables"] = {
        "fpv_full": f"{DEMO}/ee17_agile_fpv_full.mp4",
        "fpv_windows": [f"{DEMO}/ee17_agile_fpv_{w}.mp4" for w in ("grasp", "hold", "ring")],
        "gif_after": f"{DEMO}/ee17_posture_agile_object_large_range.gif",
        "gif_before": f"{DEMO}/ee17_posture_agile_before_large_range.gif",
        "fpv_compare": [f"{DEMO}/ee17_agile_fpv_compare_{w}.gif"
                        for w in ("grasp", "hold", "ring")],
        "agile_bundle": f"{AGILE_DIR}/runs/bundle",
        "agile_motion": f"{AGILE_DIR}/runs/bundle/export/motion",
        "follow_plot": f"{LR}/ee17_posture_agile_object_follow_compare.png",
    }
    ev["commands"] = [
        "# AGILE 链 (55 真值帧子集, 渲染 cap 补丁):",
        "bash tools/run_ee17_release_s.sh 6  # (AGILE repo, stages 0-8)",
        "# 桥接 (mocap 深度锚定 s=0.539, PCA 对齐, 尺度收敛 1.097):",
        "python scripts/agile_ee17_bridge.py",
        "# R1 释放窗 A/B (AGILE 几何):",
        "python test/ee17_posture.py after --object "
        f"{WIN}/object_instance_agile.npz --name ee17_posture_agile_object",
        "python test/ee17_posture.py before --object "
        f"{WIN}/object_instance_agile.npz --name ee17_posture_agile_before",
        "# FPV/对比动图:",
        "python test/ee17_fpv.py after && python test/ee17_fpv.py compare",
    ]

    out = f"{LR}/ee17_agile_evidence.json"
    json.dump(ev, open(out, "w"), ensure_ascii=False, indent=1, default=float)
    print(f"✓ {out}")
    # 人读摘要
    a = ev["window600"]["agile_geometry"]
    p = ev["window600"]["prior_proxy_object"]
    r = ev["registration"]
    print(f"""
===== ee17 AGILE 接入验收摘要 =====
尺度收敛 (先验腕锚定 / mocap 深度锚定 AGILE): {r['mesh_scale_prior_over_anchor']:.3f}
c2w 搬运一致性: {r['carry_residual_hold_mm']:.1f} mm | 手链交叉: {r['hand_chain_mean_mm']['raw_zfwd']:.0f} mm
释放窗 A/B:  先验代理 抓起={'✓' if p['grasped'] else '✗'} {p['lift_mm']:.0f}mm | AGILE 几何 抓起={'✓' if a['grasped'] else '✗'} {a['lift_mm']:.0f}mm
物理跟随: 先验 21.2mm / AGILE {a['phys_pos_err_mm']:.1f}mm
执行层: 接管帧 {ev['exec_layer']['agile_geometry']['gravity_handoff_frame']}, 双指冲量峰值 {ev['exec_layer']['agile_geometry']['finger_load_peak_N']:.2f} N·s, 到站误差 {ev['exec_layer']['agile_geometry']['base_arrival_err_mm']:.0f}mm
""")


if __name__ == "__main__":
    main()
