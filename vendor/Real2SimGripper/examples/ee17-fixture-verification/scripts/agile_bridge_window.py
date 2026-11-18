#!/usr/bin/env python3
"""agile_bridge_window.py — 任意窗口的 AGILE→R1 桥接 (ee17 桥的参数化版)。

与 agile_ee17_bridge.py 同一套方法 (mocap 深度锚定 + 相机约定 4 候选择优 +
48 候选 PCA 对齐 + 尺度收敛记录), 三处泛化:
  1. 参数化: --agile-dir (含 seq_s 子集) / --episode / --t0 / --out-dir /
     --prior-object / --targets
  2. 阶段帧从 targets grip_states 推导 (grip_state_names:
     APPROACH/PICK/LIFT/HOLD/PLACE/RELEASED), 不再硬编码 ee17 时刻
  3. 窗帧 = t0*30 + 3*orig (AGILE SUBSAMPLE=3)

用法 (agile-optim env, 链完成后):
    python scripts/agile_bridge_window.py --agile-dir <AGILE/output/SEQ> \
        --episode 12 --t0 12.5 --out-dir <EE_DIR>/w0 \
        --prior-object <EF_DIR>/object_instance.npz \
        --targets <EE_DIR>/w0/ef3r_right_targets.npz
输出: <out-dir>/object_instance_agile.npz + agile_registration.json
"""
import argparse
import itertools
import json
import os

import numpy as np
from scipy.spatial import cKDTree

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agile_ee17_bridge import load_obj, pca_frame  # noqa: E402

EE = "/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
FLIP_X = np.diag([1.0, -1.0, -1.0])          # Unity(z-back) → OpenCV(z-fwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agile-dir", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--t0", type=float, required=True, help="窗起点秒 (视频绝对)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prior-object", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--side", type=int, default=2, help="tracker 列: 2=右腕")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    aligned = os.path.join(a.agile_dir, "runs", "opt", "aligned")
    SEQDIR = os.path.join(a.agile_dir, "seq_s")

    # ---- AGILE aligned ----
    Jc = np.load(os.path.join(aligned, "optimized_mano_joints.npz"))["mano_joints"]
    Jc = Jc[:, 0].astype(np.float64)
    op = np.load(os.path.join(aligned, "object_poses.npz"))
    T_co = op["object_poses_in_cam"].astype(np.float64)
    obj_scale = float(np.asarray(op["object_scale"]).ravel()[0])
    pred = np.asarray(np.load(os.path.join(aligned, "camera_data.npz"))
                      ["prediction_frame_ids"], np.int64)
    fmap_p = os.path.join(a.agile_dir, "subset_frame_map.npy")
    orig = (np.load(fmap_p) if os.path.exists(fmap_p) else np.arange(len(pred)))[pred]
    win_frames = np.round(a.t0 * 30).astype(int) + orig * 3   # 绝对视频帧
    F = len(pred)
    print(f"AGILE aligned: {F} 帧 | object_scale {obj_scale:.4f}")

    # ---- mocap ----
    d = np.load(f"{EE}/episode_meta/episode_{a.episode:06d}/smplx/smplx_episode.npz",
                allow_pickle=True)
    vfi = np.asarray(d["video_frame_index"], np.int64)
    tpos = np.asarray(d["tracker_position"], np.float64)
    trot = np.asarray(d["tracker_rotation"], np.float64)
    att_r = np.asarray(d["attachment_rotation"], np.float64)[0]
    rows = np.searchsorted(vfi, win_frames)
    ok = (rows < len(vfi)) & (vfi[np.clip(rows, 0, len(vfi) - 1)] == win_frames)
    assert ok.mean() > 0.8, f"mocap 行映射 {ok.mean():.0%}"
    rows = np.clip(rows, 0, len(vfi) - 1)
    rw_w = tpos[rows, a.side]

    # ---- mocap 深度锚定 ----
    from PIL import Image
    w2d = np.load(f"{SEQDIR}/wilor_raw/mano_joints_2d.npz")["mano_joints_2d"][:, 0].astype(float)
    ratios = []
    for k, p in enumerate(pred):
        R0 = trot[rows[k], 0]
        rw_cam = R0.T @ (tpos[rows[k], a.side] - tpos[rows[k], 0])
        u, v = w2d[p, 0].astype(int)
        dep = np.array(Image.open(f"{SEQDIR}/moge/depth/{p:05d}.png"), float) / 1000.0
        v2, u2 = int(np.clip(v, 0, dep.shape[0] - 1)), int(np.clip(u, 0, dep.shape[1] - 1))
        pv = dep[max(0, v2 - 4):v2 + 5, max(0, u2 - 4):u2 + 5]
        pv = pv[pv > 0]
        if len(pv) and rw_cam[2] > 0.02:
            ratios.append(rw_cam[2] / float(np.median(pv)))
    s_anchor = float(np.median(ratios)) if ratios else 1.0
    print(f"mocap 深度锚定: s={s_anchor:.3f} (n={len(ratios)})")
    Jc = Jc * s_anchor
    T_co[:, :3, 3] *= s_anchor

    # ---- 相机约定 4 候选 (手链交叉择优) ----
    tg = np.load(a.targets, allow_pickle=True)
    gs = np.asarray(tg["grip_states"], np.int64)
    names = [str(x) for x in np.asarray(tg["grip_state_names"])]
    # 阶段帧 (targets 时间轴 = 窗帧系)
    def phase_range(nm):
        i = names.index(nm) if nm in names else -1
        idx = np.flatnonzero(gs == i)
        return (int(idx[0]), int(idx[-1])) if len(idx) else (0, len(gs) - 1
                                                            if len(gs) else 0)
    h0, h1 = phase_range("HOLD")
    p0 = phase_range("PICK")[0]
    hold_tw = (win_frames >= h0) & (win_frames <= h1)
    static_tw = win_frames < p0
    convs = {}
    for nm_att, R0 in (("raw", trot[rows, 0]),
                       ("attachment", trot[rows, 0] @ att_r[None])):
        for nm_flip, FL in (("zfwd", np.eye(3)), ("zback", FLIP_X)):
            R = R0 @ FL[None]
            jw = np.einsum("fij,fj->fi", R, Jc[:, 0]) + tpos[rows, 0]
            convs[f"{nm_att}_{nm_flip}"] = (
                R, float(np.linalg.norm(jw - rw_w, axis=1).mean() * 1000))
    c2w_name = min(convs, key=lambda k: convs[k][1])
    R_cam = convs[c2w_name][0]
    print(f"相机约定: {c2w_name} (手链 mean {convs[c2w_name][1]:.0f} mm)")

    T_cw = np.tile(np.eye(4), (F, 1, 1))
    T_cw[:, :3, :3] = R_cam
    T_cw[:, :3, 3] = tpos[rows, 0]
    T_wo = np.einsum("fij,fjk->fik", T_cw, T_co)
    t_wo = T_wo[:, :3, 3]

    d_obj = np.diff(t_wo[hold_tw], axis=0) if hold_tw.sum() >= 2 else None
    carry_mm = (float(np.median(np.linalg.norm(
        d_obj - np.diff(rw_w[hold_tw], axis=0), axis=1)) * 1000)
        if d_obj is not None and len(d_obj) else None)

    dist_agile = np.linalg.norm(T_co[:, :3, 3] - Jc[:, 0], axis=1) * 1000
    pre_mm = float(dist_agile[static_tw].mean()) if static_tw.any() else None
    hold_mm = float(dist_agile[hold_tw].mean()) if hold_tw.any() else None

    # ---- 几何: mesh → 先验 PCA 框架 (48 候选 chamfer) ----
    V_m, Fc = load_obj(os.path.join(aligned, "object_mesh.obj"))
    V_m = V_m * obj_scale * s_anchor
    c_m, ax_m = pca_frame(V_m)
    prior = np.load(a.prior_object, allow_pickle=True)
    Vp = np.asarray(prior["verts"], np.float64)
    c_p, ax_p = pca_frame(Vp)
    ext_m = np.ptp((V_m - c_m) @ ax_m, axis=0)
    ext_p = np.ptp((Vp - c_p) @ ax_p, axis=0)
    Vs = V_m[np.random.default_rng(0).choice(len(V_m), min(1500, len(V_m)), replace=False)]
    tree = cKDTree(Vp - c_p)
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            A = ax_m[:, list(perm)] * np.array(signs)[None, :]
            dd, _ = tree.query((Vs - c_m) @ A)
            sc = float(dd.mean())
            if best is None or sc < best[0]:
                best = (sc, perm, signs)
    chamfer0, perm, signs = best
    A = ax_m[:, list(perm)] * signs[None, :]
    s_snap = float(ext_p.mean() / np.sort(ext_m[list(perm)]).mean())
    print(f"PCA 对齐 perm {perm} signs {signs} chamfer {chamfer0*1000:.1f}mm | "
          f"尺寸比 (先验/锚定AGILE) {s_snap:.3f}")
    verts_w = (V_m - c_m) @ A.T + c_p
    if len(Fc) > 4000:
        import open3d as o3d
        _m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts_w),
            o3d.utility.Vector3iVector(np.asarray(Fc, np.int64)))
        _m = _m.simplify_quadric_decimation(3000)
        verts_w = np.asarray(_m.vertices, np.float64)
        Fc = np.asarray(_m.triangles, np.int64)
    _cc = verts_w.mean(0)
    flip = 0
    for i, (a1, b1, c1) in enumerate(Fc):
        n = np.cross(verts_w[b1] - verts_w[a1], verts_w[c1] - verts_w[a1])
        fc = (verts_w[a1] + verts_w[b1] + verts_w[c1]) / 3
        if n @ (fc - _cc) < 0:
            Fc[i] = [a1, c1, b1]
            flip += 1
    ctr_w, axes_w = pca_frame(verts_w)
    ext_w = np.ptp((verts_w - ctr_w) @ axes_w, axis=0)

    out_npz = os.path.join(a.out_dir, "object_instance_agile.npz")
    op_sample = verts_w[:: max(1, len(verts_w) // 400)].copy()
    np.savez(out_npz,
             verts=verts_w, faces=Fc.astype(np.int64),
             center_pre=prior["center_pre"], center_placed=prior["center_placed"],
             obj_points=op_sample, n_points=len(op_sample),
             pca_center=ctr_w, pca_axes=axes_w, pca_ext=ext_w,
             support_placed=prior["support_placed"],
             surface_pick_V=prior["surface_pick_V"], surface_pick_F=prior["surface_pick_F"],
             surface_place_V=prior["surface_place_V"], surface_place_F=prior["surface_place_F"],
             obj_scale_m=obj_scale, traj_T_pos=t_wo, traj_T_rot=T_wo[:, :3, :3],
             provenance=f"agile {os.path.basename(a.agile_dir)} (mocap-anchored)")
    reg = {"c2w_convention": c2w_name,
           "hand_chain_mean_mm": {k: v[1] for k, v in convs.items()},
           "carry_residual_hold_mm": carry_mm,
           "obj_wrist_pre_agile_mm": pre_mm,
           "obj_wrist_hold_agile_mm": hold_mm,
           "mocap_depth_anchor_s": s_anchor,
           "mesh_scale_prior_over_anchor": s_snap,
           "mesh": {"verts": int(len(verts_w)), "faces": int(len(Fc)),
                    "flip_faces": flip, "pca_ext_mm": ext_m.tolist(),
                    "prior_pca_ext_mm": ext_p.tolist(),
                    "snapshot_chamfer_mm": chamfer0 * 1000,
                    "pca_perm": list(perm), "pca_signs": list(signs)},
           "object_scale_agile": obj_scale,
           "window_frames": win_frames.tolist(),
           "phase_frames": {"PICK": p0, "HOLD": [h0, h1]}}
    reg_p = os.path.join(a.out_dir, "agile_registration.json")
    json.dump(reg, open(reg_p, "w"), indent=1, ensure_ascii=False)
    print(f"✓ {out_npz}\n✓ {reg_p}")


if __name__ == "__main__":
    main()
