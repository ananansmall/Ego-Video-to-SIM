#!/usr/bin/env python3
"""ee17_ras_compare.py — RAS→EE17 配准方案对比 (A: frame0 锚定; B: umeyama+偏移修正)。"""
import json
import os
import sys

import numpy as np

W = open("/tmp/ee17-ras-workdir.txt").read().strip()
OUT = open("/tmp/ee17-release600-out.txt").read().strip().split("=", 1)[1]
MOCAP = ("/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
         "/episode_meta/episode_000017/smplx/smplx_episode.npz")
M_Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
BASE_ABS = 378 * 30


def rot_norm(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def main():
    z = np.load(os.path.join(W, "raw_map.npz"))
    w2c = np.asarray(z["w2c"], np.float64)
    f10 = np.asarray(z["frame_10fps"], np.int64)
    S = len(f10)
    C_ras = np.array([-w2c[k, :3, :3].T @ w2c[k, :3, 3] for k in range(S)])
    R_ras_c2w = np.array([w2c[k, :3, :3].T for k in range(S)])   # 各帧相机朝向 (RAS world)

    d = np.load(MOCAP, allow_pickle=True)
    vfi = np.asarray(d["video_frame_index"], np.int64)
    tpos = np.asarray(d["tracker_position"], np.float64)
    trot = np.asarray(d["tracker_rotation"], np.float64)
    idx = np.searchsorted(vfi, BASE_ABS + f10 * 3)
    hit = (idx < len(vfi)) & (vfi[np.clip(idx, 0, len(vfi) - 1)] == BASE_ABS + f10 * 3)
    assert hit.mean() > 0.9, f"mocap 命中 {hit.mean():.0%}"
    idx = np.clip(idx, 0, len(vfi) - 1)
    C_ee = np.array([M_Y2Z @ tpos[i, 0] for i in idx])            # z-up
    R_ee = np.array([M_Y2Z @ trot[i, 0] @ M_Y2Z.T for i in idx])  # 旋转同系
    R_ee = np.array([rot_norm(r) for r in R_ee])

    holdout = set(np.arange(S)[::4].tolist())
    train = np.array([i for i in range(S) if i not in holdout])
    test = np.array(sorted(holdout))

    # ---- 方案 A: frame0 锚定 + 轨迹中位尺度 ----
    dist_ras = np.linalg.norm(C_ras - C_ras[0], axis=1)
    dist_ee = np.linalg.norm(C_ee - C_ee[0], axis=1)
    ratio = dist_ee[1:] / np.maximum(dist_ras[1:], 1e-6)
    ok = ratio < np.percentile(ratio, 80) * 2
    sA = float(np.median(ratio[ok]))
    R_A = rot_norm(R_ee[0] @ R_ras_c2w[0].T)
    t_A = C_ee[0] - sA * (R_A @ C_ras[0])
    eA_tr = np.linalg.norm((sA * (C_ras[train] @ R_A.T) + t_A) - C_ee[train], axis=1) * 1000
    eA_te = np.linalg.norm((sA * (C_ras[test] @ R_A.T) + t_A) - C_ee[test], axis=1) * 1000
    # 相机朝向一致性 (旋转误差 deg)
    R_ras_pred = np.array([R_A @ r for r in R_ras_c2w])
    angA = np.array([np.degrees(np.arccos(np.clip(
        (np.trace(R_ee[k].T @ R_ras_pred[k]) - 1) / 2, -1, 1))) for k in range(S)])

    # ---- 方案 B: umeyama + tracker→camera 偏移修正 (2 轮) ----
    sys.path.insert(0, os.path.join(W, "ReplicateAnyScene"))
    _ns = {}
    exec(compile(open(os.path.join(W, "ReplicateAnyScene", "stage4", "umeyama.py")).read(),
                 "umeyama.py", "exec"), _ns)
    um = _ns["umeyama_alignment"]
    sB, R_B, t_B, _ = um(C_ras[train], C_ee[train], with_scale=True)
    for _ in range(2):
        e = C_ee - (sB * (C_ras @ R_B.T) + t_B)          # 世界系残差
        d_cam = np.array([R_ee[k].T @ e[k] for k in range(S)])  # 相机系偏移样本
        d_fix = np.median(d_cam[train], axis=0)          # 固定安装偏移 (相机系)
        C_ee_corr = C_ee + np.array([R_ee[k] @ d_fix for k in range(S)])
        sB, R_B, t_B, _ = um(C_ras[train], C_ee_corr[train], with_scale=True)
    eB_tr = np.linalg.norm((sB * (C_ras[train] @ R_B.T) + t_B) - C_ee_corr[train], axis=1) * 1000
    eB_te = np.linalg.norm((sB * (C_ras[test] @ R_B.T) + t_B) - C_ee_corr[test], axis=1) * 1000

    print(f"方案A (frame0 锚定): scale={sA:.4f} | train {eA_tr.mean():.1f}mm "
          f"holdout {eA_te.mean():.1f}mm (p95 {np.percentile(eA_te,95):.1f}) | "
          f"朝向误差 mean {angA.mean():.1f}° p95 {np.percentile(angA,95):.1f}°")
    print(f"方案B (umeyama+偏移): scale={sB:.4f} 偏移 {np.round(d_fix*1000,1)}mm | "
          f"train {eB_tr.mean():.1f}mm holdout {eB_te.mean():.1f}mm "
          f"(p95 {np.percentile(eB_te,95):.1f})")
    report = {
        "A": {"scale": sA, "R": R_A.tolist(), "t": t_A.tolist(),
              "train_mm": float(eA_tr.mean()), "holdout_mm": float(eA_te.mean()),
              "holdout_p95_mm": float(np.percentile(eA_te, 95)),
              "rot_err_deg_mean": float(angA.mean())},
        "B": {"scale": sB, "R": R_B.tolist(), "t": t_B.tolist(),
              "cam_offset_m": d_fix.tolist(),
              "train_mm": float(eB_tr.mean()), "holdout_mm": float(eB_te.mean()),
              "holdout_p95_mm": float(np.percentile(eB_te, 95))}}
    json.dump(report, open(os.path.join(OUT, "ee17_ras_align_compare.json"), "w"),
              indent=1)
    print("✓", os.path.join(OUT, "ee17_ras_align_compare.json"))


if __name__ == "__main__":
    main()
