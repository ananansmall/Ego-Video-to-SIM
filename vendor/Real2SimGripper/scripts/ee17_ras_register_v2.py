#!/usr/bin/env python3
"""ee17_ras_register_v2.py — RAS→EE17 终版配准: B 方案(偏移修正) + 鲁棒剔离群。

输出: ee17_ras_scene.npz / ee17_ras_transform.json / 三联图 F47/F448/F534
"""
import json
import os
import sys

import numpy as np
import cv2

W = open("/tmp/ee17-ras-workdir.txt").read().strip()
OUT = open("/tmp/ee17-release600-out.txt").read().strip().split("=", 1)[1]
RAS = os.path.join(W, "ReplicateAnyScene")
_ns = {}
exec(compile(open(os.path.join(RAS, "stage4", "umeyama.py")).read(),
             "umeyama.py", "exec"), _ns)
um = _ns["umeyama_alignment"]

MOCAP = ("/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
         "/episode_meta/episode_000017/smplx/smplx_episode.npz")
SRC_SEQ = "/path/to/project_data/mano_ws/AGILE/output/ee17_release/seq"
BASE_ABS = 378 * 30
M_Y2Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
CONF = 0.5
VOX = 0.02
PHASES = {"PICK": (47, 16), "PLACE": (448, 149), "RELEASED": (534, 178)}


def rot_norm(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def main():
    z = np.load(os.path.join(W, "raw_map.npz"), allow_pickle=True)
    w2c = np.asarray(z["w2c"], np.float64)
    pts = np.asarray(z["points"], np.float64)
    conf = np.asarray(z["confidence"], np.float64)
    cols = np.asarray(z["rgb"], np.float64)
    f10 = np.asarray(z["frame_10fps"], np.int64)
    S, H, Wd = pts.shape[:3]

    d = np.load(MOCAP, allow_pickle=True)
    vfi = np.asarray(d["video_frame_index"], np.int64)
    tpos = np.asarray(d["tracker_position"], np.float64)
    trot = np.asarray(d["tracker_rotation"], np.float64)
    idx = np.searchsorted(vfi, BASE_ABS + f10 * 3)
    idx = np.clip(idx, 0, len(vfi) - 1)
    C_ee = np.array([M_Y2Z @ tpos[i, 0] for i in idx])
    R_ee = np.array([rot_norm(M_Y2Z @ trot[i, 0] @ M_Y2Z.T) for i in idx])
    C_ras = np.array([-w2c[k, :3, :3].T @ w2c[k, :3, 3] for k in range(S)])

    holdout = set(np.arange(S)[::4].tolist())
    train = np.array([i for i in range(S) if i not in holdout])
    test = np.array(sorted(holdout))

    # ---- B 方案 + 迭代偏移修正 + 离群帧剔除 ----
    active = set(range(S))
    for it in range(3):
        tr = np.array([i for i in train if i in active])
        sB, R_B, t_B, _ = um(C_ras[tr], C_ee[tr], with_scale=True)
        e = np.linalg.norm((sB * (C_ras @ R_B.T) + t_B) - C_ee, axis=1) * 1000
        # 偏移修正
        e_vec = (C_ee - (sB * (C_ras @ R_B.T) + t_B))
        d_fix = np.median(np.array([R_ee[k].T @ e_vec[k] for k in tr]), axis=0)
        C_corr = C_ee + np.array([R_ee[k] @ d_fix for k in range(S)])
        sB, R_B, t_B, _ = um(C_ras[tr], C_corr[tr], with_scale=True)
        e = np.linalg.norm((sB * (C_ras @ R_B.T) + t_B) - C_corr, axis=1) * 1000
        worst = int(np.argmax(e))
        print(f"iter{it}: train {e[train].mean():.1f}mm holdout "
              f"{e[test].mean():.1f}mm | worst idx{worst}(f{f10[worst]}) "
              f"{e[worst]:.0f}mm")
        if e[worst] > 200 and len(active) > S - 6:
            active.discard(worst)
            print(f"  剔除离群帧 idx{worst}")
        else:
            break
    # 偏移修正后的最终误差
    e_final = np.linalg.norm((sB * (C_ras @ R_B.T) + t_B) - C_corr, axis=1) * 1000
    tr_err = e_final[[i for i in train if i in active]]
    te_err = e_final[[i for i in test if i in active]]
    print(f"最终: train {tr_err.mean():.1f}mm holdout {te_err.mean():.1f}mm "
          f"(p95 {np.percentile(te_err, 95):.1f}) scale={sB:.4f} "
          f"偏移={np.round(d_fix*1000,1)}mm")

    T_json = {"scheme": "B umeyama+cam-offset iter+outlier-reject",
              "scale": float(sB), "R": R_B.tolist(), "t": t_B.tolist(),
              "cam_offset_m": d_fix.tolist(),
              "dropped_frames_10fps": [int(f10[i]) for i in range(S)
                                       if i not in active],
              "train_mm": float(tr_err.mean()), "holdout_mm": float(te_err.mean()),
              "holdout_p95_mm": float(np.percentile(te_err, 95)),
              "convention": "RAS world -> EE17 z-up sim input (mocap via M_y2z)",
              "camera_center_formula": "C = -R_w2c^T t_w2c"}

    # ---- 点云聚合 (conf 0.5 + mask 剔重影 + 帧计数静态过滤) ----
    all_P, all_C, all_F, drop_tot = [], [], [], 0
    for k in range(S):
        Pk = sB * (pts[k].reshape(-1, 3) @ R_B.T) + t_B
        Ck = cols[k].reshape(-1, 3)
        confk = conf[k].reshape(-1)
        f = int(f10[k])
        keep = confk >= CONF
        for mdir in ("mask/hand", "mask/object"):
            mp = os.path.join(SRC_SEQ, mdir, f"{f:05d}.png")
            if os.path.exists(mp):
                m = cv2.imread(mp, 0)
                if m is not None:
                    ms = cv2.resize(m, (Wd, H), interpolation=cv2.INTER_NEAREST)
                    keep &= ~(ms.reshape(-1) > 0)
        drop_tot += int(confk.sum() - keep.sum())
        all_P.append(Pk[keep])
        all_C.append(Ck[keep])
        all_F.append(np.full(int(keep.sum()), k, np.int64))
    P_all = np.concatenate(all_P)
    C_all = np.concatenate(all_C)
    F_all = np.concatenate(all_F)
    keep = ((P_all[:, 2] > -1.8) & (P_all[:, 2] < 2.6)
            & (np.abs(P_all[:, 0]) < 8) & (np.abs(P_all[:, 1]) < 8))
    P_all, C_all, F_all = P_all[keep], C_all[keep], F_all[keep]
    # 计数式静态过滤: 体素被 >=MIN_FRAMES 个不同帧观察才保留 (手臂/手/漏斗
    # 只在 1-2 帧出现, 桌面/支架被 20+ 帧看 —— 一次去掉全部动态残影)
    MIN_FRAMES = 6
    voxel = np.floor(P_all / VOX).astype(np.int64)
    vu, inv = np.unique(voxel, axis=0, return_inverse=True)
    pairs = np.stack([inv, F_all], axis=1)
    up, _ = np.unique(pairs, axis=0, return_counts=True)
    vox_count = np.zeros(len(vu), np.int64)
    np.add.at(vox_count, up[:, 0], 1)
    stat = vox_count[inv] >= MIN_FRAMES
    P_all, C_all = P_all[stat], C_all[stat]
    voxel2 = np.floor(P_all / VOX).astype(np.int64)
    _, uniq2 = np.unique(voxel2, axis=0, return_index=True)
    order2 = np.sort(uniq2)
    P_all, C_all = P_all[order2], C_all[order2]
    print(f"静态过滤 (体素被>={MIN_FRAMES}帧观察): {len(P_all)} 点 "
          f"(剔除动态残影 {int((~stat).sum())})")

    # ---- 桌面/铁圈对照 (release_scene 为现行已对齐参考, 同 mocap z-up 系) ----
    rs = np.load("/tmp/ee17-fix-vkg3vuoa/assets/release_scene.npz")
    P_ref = np.asarray(rs["points"], np.float64)
    z_hist_r, ze_r = np.histogram(P_ref[:, 2], bins=200,
                                  range=(P_ref[:, 2].min(), P_ref[:, 2].max()))
    table_ref = float((ze_r[int(np.argmax(z_hist_r))] + ze_r[int(np.argmax(z_hist_r)) + 1]) / 2)
    z_hist, ze = np.histogram(P_all[:, 2], bins=200,
                              range=(P_all[:, 2].min(), P_all[:, 2].max()))
    table_ras = float((ze[int(np.argmax(z_hist))] + ze[int(np.argmax(z_hist)) + 1]) / 2)
    print(f"桌面 z: RAS {table_ras:.3f} vs release_scene {table_ref:.3f} "
          f"(偏差 {(table_ras-table_ref)*1000:+.0f} mm)")
    rp = np.load("/tmp/ee17-e2e-r2dkbroz/release_window/object_instance_ring_proxy.npz",
                 allow_pickle=True)
    ring_c = np.asarray(rp["verts"], np.float64).mean(0)
    nb = P_all[np.linalg.norm(P_all[:, :3] - ring_c[:3], axis=1) < 0.25]
    nb_ref = P_ref[np.linalg.norm(P_ref[:, :3] - ring_c[:3], axis=1) < 0.25]
    print(f"铁圈中心 {np.round(ring_c[:3],3)}: RAS 点云附近 {len(nb)} 点 | "
          f"release_scene 附近 {len(nb_ref)} 点")

    out_npz = os.path.join(OUT, "ee17_ras_scene.npz")
    C_out = C_all / 255.0 if C_all.max() > 1.5 else C_all
    np.savez(out_npz, points=P_all.astype(np.float32),
             colors=C_out.astype(np.float32), mocap_world=np.array([1]))
    T_json.update({"table_z_ras": table_ras, "table_z_release_scene": table_ref,
                   "table_dev_mm": float((table_ras - table_ref) * 1000),
                   "ring_report": {"center_xyz": ring_c[:3].tolist(),
                                   "ras_pts_near": int(len(nb)),
                                   "release_scene_pts_near": int(len(nb_ref))},
                   "agg_points": int(len(P_all)), "conf_thr": CONF,
                   "ghost_dropped_px": int(drop_tot)})
    json.dump(T_json, open(os.path.join(OUT, "ee17_ras_transform.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"✓ {out_npz}")

    # ---- 三联图 (投影点膨胀 2px) ----
    K = np.asarray(z["K_mean"], np.float64).reshape(3, 3)
    for nm, (wf, f10i) in PHASES.items():
        k = int(np.argmin(np.abs(f10 - f10i)))
        orig = cv2.imread(os.path.join(SRC_SEQ, "rgb", f"{f10i:05d}.jpg"))
        if orig is None:
            continue
        Pk_ee = sB * (pts[k].reshape(-1, 3) @ R_B.T) + t_B
        confk = conf[k].reshape(-1)
        Pk_ras = ((Pk_ee - t_B) / sB) @ R_B
        Rk, tk = w2c[k, :3, :3], w2c[k, :3, 3]
        Pc = (Rk @ Pk_ras.T + tk[:, None]).T
        valid = (Pc[:, 2] > 0.05) & (confk >= CONF)
        uv = (K @ Pc[valid].T).T
        uv = uv[:, :2] / uv[:, 2:3]
        uv[:, 0] *= 1920.0 / Wd
        uv[:, 1] *= 970.0 / H
        depth_img = np.zeros((970, 1920), np.float32)
        ui = np.clip(uv[:, 0].astype(int), 0, 1919)
        vi = np.clip(uv[:, 1].astype(int), 0, 969)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                depth_img[np.clip(vi+dv, 0, 969), np.clip(ui+du, 0, 1919)] = Pc[valid, 2]
        dn = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        proj = cv2.applyColorMap(dn, cv2.COLORMAP_TURBO)
        proj[depth_img == 0] = 0
        overlay = cv2.addWeighted(orig, 0.5, proj, 0.5, 0)
        tri = np.concatenate([orig, proj, overlay], axis=1)
        tri = cv2.resize(tri, (tri.shape[1] // 2, tri.shape[0] // 2))
        p3 = os.path.join(OUT, f"triptych_{nm}_F{wf}.jpg")
        cv2.imwrite(p3, tri)
        print(f"✓ 三联图 {nm} F{wf}")


if __name__ == "__main__":
    main()
