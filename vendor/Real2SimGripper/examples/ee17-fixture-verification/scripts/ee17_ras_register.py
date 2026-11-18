#!/usr/bin/env python3
"""ee17_ras_register.py — RAS 建图 → EE17 坐标配准 (用户步骤 2)。

输入: RAS raw_map.npz (39 帧 VGGT-Omega, w2c/points/conf/rgb/depth)
目标: 米制 +Z 向上的 EE17 仿真输入系 (mocap tracker 相机位置经 M_y2z;
      npz 标 mocap_world=[1], cb 入口统一应用 yaw+z_shift)。
产出: ee17_ras_scene.npz (points/colors/mocap_world) + 变换 JSON +
      三联图 (原帧 | 配准点云投影 | 叠加) F47/F448/F534 +
      桌面高度/铁圈中心偏差报告。
"""
import json
import os
import sys

import numpy as np
import cv2

OUT = open("/tmp/ee17-release600-out.txt").read().strip().split("=", 1)[1]
W = open("/tmp/ee17-ras-workdir.txt").read().strip()
RAS = os.path.join(W, "ReplicateAnyScene")
# stage4/__init__ 连带 import pyrender (vggt-omega env 无) —— 绕过包加载,
# 直接从文件 exec umeyama_alignment
_um_src = open(os.path.join(RAS, "stage4", "umeyama.py")).read()
_ns = {}
exec(compile(_um_src, "umeyama.py", "exec"), _ns)  # noqa: S102
umeyama_alignment = _ns["umeyama_alignment"]

MOCAP = ("/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
         "/episode_meta/episode_000017/smplx/smplx_episode.npz")
SRC_SEQ = "/path/to/project_data/mano_ws/AGILE/output/ee17_release/seq"
BASE_ABS = 378 * 30
M_Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
CONF_THR = 1.5
VOX = 0.02
PHASES = {"PICK": (47, 16), "PLACE": (448, 149), "RELEASED": (534, 178)}


def main():
    z = np.load(os.path.join(W, "raw_map.npz"))
    w2c = np.asarray(z["w2c"], np.float64)
    pts = np.asarray(z["points"], np.float64)
    conf = np.asarray(z["confidence"], np.float64)
    cols = np.asarray(z["rgb"], np.float64)
    f10 = np.asarray(z["frame_10fps"], np.int64)
    S, H, Wd = pts.shape[:3]
    print(f"RAS map: {S} 帧, 网格 {H}x{Wd}, conf>={CONF_THR}")

    # ---- EE17 mocap 相机位置 (z-up 仿真输入系) ----
    d = np.load(MOCAP, allow_pickle=True)
    vfi = np.asarray(d["video_frame_index"], np.int64)
    tpos = np.asarray(d["tracker_position"], np.float64)
    C_ee = np.zeros((S, 3))
    hit = 0
    for k, f in enumerate(f10):
        row = int(np.searchsorted(vfi, BASE_ABS + f * 3))
        if row < len(vfi) and int(vfi[row]) == BASE_ABS + f * 3:
            C_ee[k] = M_Y2Z @ tpos[row, 0]
            hit += 1
    assert hit >= S - 2, f"mocap 行映射失败 ({hit}/{S})"
    print(f"mocap 相机位置: {hit}/{S} 帧命中 (z-up)")

    # ---- RAS 相机中心 C = -R^T t ----
    C_ras = np.zeros((S, 3))
    for k in range(S):
        Rk, tk = w2c[k, :3, :3], w2c[k, :3, 3]
        C_ras[k] = -Rk.T @ tk

    # ---- train/holdout ----
    all_idx = np.arange(S)
    holdout = set(all_idx[::4].tolist())
    train = np.array([i for i in all_idx if i not in holdout])
    test = np.array(sorted(holdout))

    # ---- Umeyama ----
    s, R, t, T = umeyama_alignment(C_ras[train], C_ee[train], with_scale=True)
    print(f"Umeyama: scale={s:.4f}")
    err_tr = np.linalg.norm((s * (C_ras[train] @ R.T) + t) - C_ee[train], axis=1) * 1000
    err_te = np.linalg.norm((s * (C_ras[test] @ R.T) + t) - C_ee[test], axis=1) * 1000
    print(f"  train: mean {err_tr.mean():.1f}mm p95 {np.percentile(err_tr,95):.1f}mm")
    print(f"  holdout: mean {err_te.mean():.1f}mm p95 {np.percentile(err_te,95):.1f}mm "
          f"max {err_te.max():.1f}mm")

    T_json = {"scale": float(s), "R": R.tolist(), "t": t.tolist(),
              "convention": "RAS world -> EE17 z-up sim input (mocap y-up via M_y2z)",
              "camera_center_formula": "C = -R_w2c^T t_w2c",
              "train_frames_10fps": f10[train].tolist(),
              "holdout_frames_10fps": f10[test].tolist(),
              "train_cam_err_mm": {"mean": float(err_tr.mean())},
              "holdout_cam_err_mm": {"mean": float(err_te.mean()),
                                     "p95": float(np.percentile(err_te, 95)),
                                     "max": float(err_te.max())}}

    # ---- 点云变换 + 剔手/漏斗重影 + 聚合 ----
    all_P, all_C, drop_tot = [], [], 0
    for k in range(S):
        Pk = s * (pts[k].reshape(-1, 3) @ R.T) + t
        Ck = cols[k].reshape(-1, 3)
        confk = conf[k].reshape(-1)
        f = int(f10[k])
        keep = confk >= CONF_THR
        n0 = int(keep.sum())
        for mdir in ("mask/hand", "mask/object"):
            mp = os.path.join(SRC_SEQ, mdir, f"{f:05d}.png")
            if not os.path.exists(mp):
                continue
            m = cv2.imread(mp, 0)
            if m is None:
                continue
            ms = cv2.resize(m, (Wd, H), interpolation=cv2.INTER_NEAREST)
            keep &= ~(ms.reshape(-1) > 0)
        drop_tot += n0 - int(keep.sum())
        all_P.append(Pk[keep])
        all_C.append(Ck[keep])
    P_all = np.concatenate(all_P)
    C_all = np.concatenate(all_C)
    print(f"聚合点: {len(P_all)} (剔除手/漏斗重影 {drop_tot} 点)")

    keep = ((P_all[:, 2] > -1.8) & (P_all[:, 2] < 2.6)
            & (np.abs(P_all[:, 0]) < 8) & (np.abs(P_all[:, 1]) < 8))
    P_all, C_all = P_all[keep], C_all[keep]
    voxel = np.floor(P_all / VOX).astype(np.int64)
    _, uniq = np.unique(voxel, axis=0, return_index=True)
    order = np.sort(uniq)
    P_all, C_all = P_all[order], C_all[order]
    print(f"体素降采样({VOX}m): {len(P_all)} 点")

    # ---- 桌面高度 ----
    z_hist, z_edges = np.histogram(P_all[:, 2], bins=200,
                                   range=(P_all[:, 2].min(), P_all[:, 2].max()))
    i_top = int(np.argmax(z_hist))
    table_z = float((z_edges[i_top] + z_edges[i_top + 1]) / 2)
    pr = np.load("/tmp/ee17-fix-vkg3vuoa/assets/release_object_proxy.npz",
                 allow_pickle=True)
    ref_table = float(np.median(np.asarray(pr["surface_pick_V"])[:, 2]))
    print(f"桌面高度: RAS {table_z:.3f} m | 参考 proxy {ref_table:.3f} m | "
          f"偏差 {abs(table_z - ref_table)*1000:.0f} mm")

    # ---- 铁圈中心对照 ----
    rp = np.load("/tmp/ee17-e2e-r2dkbroz/release_window/object_instance_ring_proxy.npz",
                 allow_pickle=True)
    ring_c = np.asarray(rp["verts"], np.float64).mean(0)
    nb = P_all[np.linalg.norm(P_all[:, :3] - ring_c[:3], axis=1) < 0.25]
    print(f"铁圈中心 (proxy 世界系): {np.round(ring_c[:3], 3)} | 附近 0.25m 点云 "
          f"{len(nb)} 点")
    ring_report = {"ring_center_proxy_xyz": ring_c[:3].tolist(),
                   "points_near_ring_025m": int(len(nb))}

    out_npz = os.path.join(OUT, "ee17_ras_scene.npz")
    C_out = C_all / 255.0 if C_all.max() > 1.5 else C_all
    np.savez(out_npz, points=P_all.astype(np.float32),
             colors=C_out.astype(np.float32), mocap_world=np.array([1]))
    T_json.update({"table_z_ras": table_z, "table_z_ref_proxy": ref_table,
                   "table_dev_mm": float(abs(table_z - ref_table) * 1000),
                   "ring_report": ring_report,
                   "agg_points": int(len(P_all)),
                   "ghost_dropped_px": int(drop_tot)})
    json.dump(T_json, open(os.path.join(OUT, "ee17_ras_transform.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"✓ {out_npz} ({len(P_all)} 点, mocap_world=1)")

    # ---- 三联图 F47/F448/F534 ----
    K = np.asarray(z["K_mean"], np.float64).reshape(3, 3)
    for nm, (wf, f10i) in PHASES.items():
        k = int(np.argmin(np.abs(f10 - f10i)))
        orig = cv2.imread(os.path.join(SRC_SEQ, "rgb", f"{f10i:05d}.jpg"))
        if orig is None:
            continue
        Pk_ee = s * (pts[k].reshape(-1, 3) @ R.T) + t
        confk = conf[k].reshape(-1)
        Pk_ras = ((Pk_ee - t) / s) @ R           # 回 RAS world
        Rk, tk = w2c[k, :3, :3], w2c[k, :3, 3]
        Pc = (Rk @ Pk_ras.T + tk[:, None]).T
        valid = (Pc[:, 2] > 0.05) & (confk >= CONF_THR)
        uv = (K @ Pc[valid].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv[:, 0] *= 1920.0 / Wd
        uv[:, 1] *= 970.0 / H
        depth_img = np.zeros((970, 1920), np.float32)
        ui = np.clip(uv[:, 0].astype(int), 0, 1919)
        vi = np.clip(uv[:, 1].astype(int), 0, 969)
        depth_img[vi, ui] = Pc[valid, 2]
        dn = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        proj = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
        proj[depth_img == 0] = 0
        overlay = cv2.addWeighted(orig, 0.55, proj, 0.45, 0)
        tri = np.concatenate([orig, proj, overlay], axis=1)
        tri = cv2.resize(tri, (tri.shape[1] // 2, tri.shape[0] // 2))
        p3 = os.path.join(OUT, f"triptych_{nm}_F{wf}.jpg")
        cv2.imwrite(p3, tri)
        print(f"✓ 三联图 {nm} F{wf}: {p3}")


if __name__ == "__main__":
    main()
