#!/usr/bin/env python3
"""agile_release600_bridge.py — ee17 释放窗口 (600 帧, file-016 378-398s) 联调桥。

把 AGILE 优化 (ee17_release_s, 55 帧真值子集 → 补跑 rel-opt 全程锚点)
接成 R1 可直接消费的两个 npz:
  1. targets_agile_600.npz — **AGILE MANO 手链驱动**的 pad 目标
     (optimized_mano_joints j0/j4/j8 → mocap c2w → sim z-up → SVD 重定向),
     grip_states 继承 targets_labeled 的源标签 (F47 PICK / F53 LIFT /
     F106 HOLD / F448 PLACE / F534 RELEASED), 时间轴上采样到 600 帧。
  2. object_agile_textured.npz — 保留 SAM3D **UV/MTL/贴图**的纹理物体
     (视觉模型用真实纹理, 碰撞体另用抽稀面), 快照摆先验 PCA 框架。

相机链与 agile_ee17_bridge 相同 (4 候选约定择优 + mocap 深度锚定); 输出
路径全部参数化 (--name 保留旧结果)。

用法 (agile-optim env, rel-opt 完成后):
    python scripts/agile_release600_bridge.py \
      --agile-dir /tmp/ee17-release600-agile.4oBckD/rel-opt \
      --labels /tmp/ee17-e2e-r2dkbroz/release_window/targets_labeled.npz \
      --scene /tmp/ee17-fix-vkg3vuoa/assets/release_scene.npz \
      --mesh-dir /path/to/project_data/mano_ws/AGILE/output/ee17_release_s/generated/object \
      --prior-obj .../ee17_win/object_instance.npz \
      --out-dir /tmp/ee17-release600-agile.4oBckD --name ee17_release600 <work>
"""
import argparse
import itertools
import json
import os
import shutil
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/path/to/Real2SimGripper"
SCRATCH = "/path/to/workspace/_scratch"
sys.path.insert(0, SCRATCH)
from cb_svd_retarget import svd_gripper_pose  # noqa: E402

MOCAP = ("/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
         "/episode_meta/episode_000017/smplx/smplx_episode.npz")
FLIP_X = np.diag([1.0, -1.0, -1.0])          # Unity(z-back) → OpenCV(z-fwd)
BASE_ABS = 378 * 30                          # 窗起点绝对视频帧
# 释放窗阶段标签 (600 帧窗系)
PHASES = {"PICK": 47, "LIFT": 53, "HOLD": 106, "PLACE": 448, "RELEASED": 534}


def load_obj(path):
    vs, fs = [], []
    for line in open(path):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            fs.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(vs, np.float64), np.array(fs, np.int64)


def pca_frame(V):
    c = V.mean(0)
    _, _, Vt = np.linalg.svd(V - c)
    return c, Vt.T


def quat_slerp(q0, q1, t):
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 1.0 - 1e-9:
        q = q0 + t * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    return (np.sin((1 - t) * th) * q0 + np.sin(t * th) * q1) / s


def rotmat_slerp_seq(Rs, xq):
    from scipy.spatial.transform import Rotation as Rot
    qs = Rot.from_matrix(Rs).as_quat()       # (N,4) xyzw
    qs = qs[:, [3, 0, 1, 2]]                 # → wxyz
    N = len(qs)
    out = np.zeros((len(xq), 3, 3))
    for i, x in enumerate(np.asarray(xq, np.float64)):
        k = min(int(np.floor(x)), N - 2)
        t = float(x - k)
        q = quat_slerp(qs[k], qs[k + 1], t)
        out[i] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return out


def load_mocap(win_frames):
    d = np.load(MOCAP, allow_pickle=True)
    vfi = np.asarray(d["video_frame_index"], np.int64)
    tpos = np.asarray(d["tracker_position"], np.float64)
    trot = np.asarray(d["tracker_rotation"], np.float64)
    att_r = np.asarray(d["attachment_rotation"], np.float64)[0]
    rows = np.searchsorted(vfi, BASE_ABS + win_frames)
    ok = (rows < len(vfi)) & (vfi[np.clip(rows, 0, len(vfi) - 1)] == BASE_ABS + win_frames)
    assert ok.mean() > 0.8, f"mocap 行映射失败 ({ok.mean():.0%})"
    rows = np.clip(rows, 0, len(vfi) - 1)
    return {"vfi": vfi, "tpos": tpos, "trot": trot, "att_r": att_r,
            "rows": rows, "ok": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agile-dir", required=True, help="rel-opt 结果目录 (含 aligned)")
    ap.add_argument("--labels", required=True, help="targets_labeled.npz")
    ap.add_argument("--scene", required=True, help="release_scene.npz")
    ap.add_argument("--mesh-dir", required=True, help="SAM3D generated/object 目录")
    ap.add_argument("--prior-obj", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="ee17_release600")
    ap.add_argument("--subset-map", default=None,
                    help="release_s/subset_frame_map.npy (缺省用 pred 直映射)")
    ap.add_argument("--seq-dir", default=None,
                    help="本轮 AGILE seq 目录 (深度/2D手点从此读; 缺省 agile-dir 同级 seq)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    aligned = os.path.join(a.agile_dir, "aligned")

    # ---- AGILE aligned (rel-opt 补跑全程锚点) ----
    Jc = np.load(os.path.join(aligned, "optimized_mano_joints.npz"))["mano_joints"]
    Jc = Jc[:, 0].astype(np.float64)                       # (F,21,3) 相机系 m
    op = np.load(os.path.join(aligned, "object_poses.npz"))
    T_co = op["object_poses_in_cam"].astype(np.float64)    # (F,4,4) 相机→物
    obj_scale = float(np.asarray(op["object_scale"]).ravel()[0])
    cd = np.load(os.path.join(aligned, "camera_data.npz"))
    pred = np.asarray(cd["prediction_frame_ids"], np.int64)  # 子集帧号
    F = len(pred)
    src_seq = os.path.dirname(os.path.dirname(aligned))     # $work/rel-opt → 但 aligned 在 rel-opt/aligned... 实际是 rel-opt
    # subset_frame_map 指向 release_s/seq 的 55 帧; rel-opt 用的 data-dir 是
    # ee17_release_s/seq (55帧真值), pred 就是 55 帧里的子索引, 直接 ×3 → 600 窗
    fmap = np.load(a.subset_map) if a.subset_map else None
    orig = (fmap[pred] if fmap is not None else pred)      # 55帧系 原200帧
    win_frames = orig * 3
    print(f"AGILE aligned: {F} 锚点 (子集 {pred} → 原 {orig} → 600窗 {win_frames}) "
          f"| object_scale {obj_scale:.4f}")
    # 锚点覆盖检查
    for nm, f in PHASES.items():
        d = int(np.abs(win_frames - f).min())
        print(f"  [{nm}] F{f} 最近锚点距 {d} 帧 {'(覆盖)' if d <= 60 else '(缺口!)'}")

    # ---- mocap c2w ----
    mm = load_mocap(win_frames)
    tpos, trot, att_r, rows = mm["tpos"], mm["trot"], mm["att_r"], mm["rows"]
    rw_w = tpos[rows, 2]

    # ---- mocap 深度锚定 ----
    from PIL import Image
    seqd = (a.seq_dir if a.seq_dir
            else os.path.join(a.agile_dir, "..", "seq"))
    seqd = os.path.normpath(seqd)
    depthd = os.path.join(seqd, "moge", "depth")
    w2d_p = os.path.join(seqd, "wilor_raw", "mano_joints_2d.npz")
    w2d = np.load(w2d_p)["mano_joints_2d"][:, 0].astype(float) if os.path.exists(w2d_p) else None
    ratios = []
    if w2d is not None:
        for k, p in enumerate(pred):
            iw = orig[k]
            R0 = trot[rows[k], 0]
            rw_cam = R0.T @ (tpos[rows[k], 2] - tpos[rows[k], 0])
            u, v = w2d[iw if iw < len(w2d) else p, 0].astype(int)
            dep = np.array(Image.open(f"{depthd}/{iw:05d}.png"), float) / 1000.0 if os.path.exists(f"{depthd}/{iw:05d}.png") else None
            if dep is None:
                continue
            v2, u2 = int(np.clip(v, 0, dep.shape[0] - 1)), int(np.clip(u, 0, dep.shape[1] - 1))
            patch = dep[max(0, v2 - 4):v2 + 5, max(0, u2 - 4):u2 + 5]
            pv = patch[patch > 0]
            if len(pv) and rw_cam[2] > 0.02:
                ratios.append(rw_cam[2] / float(np.median(pv)))
    s_anchor = float(np.median(ratios)) if ratios else 1.0
    print(f"mocap 深度锚定: s={s_anchor:.3f} (n={len(ratios)})")
    # 剔除优化发散锚点 (实测 idx24/46 Jc 达 ±1.2km -> 腕距 1.3km 拉爆 mean)
    bad = np.flatnonzero(np.abs(Jc).max(axis=(1, 2)) > 3.0)
    if len(bad):
        print(f"  剔除 |Jc|>3m 发散锚点: {bad.tolist()} (子集帧 {pred[bad].tolist()})")
        keepm = np.ones(len(Jc), bool)
        keepm[bad] = False
        Jc, T_co, pred, orig, win_frames = (Jc[keepm], T_co[keepm], pred[keepm],
                                            orig[keepm], win_frames[keepm])
        rows = rows[keepm]
        rw_w = tpos[rows, 2]                               # 同步剔除
        F = len(pred)
    Jc *= s_anchor
    T_co[:, :3, 3] *= s_anchor

    # ---- 相机约定 4 候选择优 (手链交叉距离) ----
    convs = {}
    for nm_att, R0 in (("raw", trot[rows, 0]),
                       ("attachment", trot[rows, 0] @ att_r[None])):
        for nm_flip, FL in (("zfwd", np.eye(3)), ("zback", FLIP_X)):
            R = R0 @ FL[None]
            jw = np.einsum("fij,fj->fi", R, Jc[:, 0]) + tpos[rows, 0]
            convs[f"{nm_att}_{nm_flip}"] = (R, float(np.linalg.norm(jw - rw_w, axis=1).mean() * 1000))
    for k, (R, mmd) in convs.items():
        print(f"  相机约定 {k}: MANO腕-mocap腕 mean {mmd:.0f} mm")
    c2w_name = min(convs, key=lambda k: convs[k][1])
    R_cam = convs[c2w_name][0]
    print(f"  选用: {c2w_name}")

    # ---- 撤爪轨迹检查 (用户: 张爪后手确有撤离) ----
    rel_f = PHASES["RELEASED"]
    k_rel = int(np.argmin(np.abs(win_frames - rel_f)))
    tail = win_frames >= rel_f
    if tail.sum() >= 2:
        retreat = float(np.linalg.norm(jw[tail][-1] - jw[tail][0]) * 1000)
        print(f"  撤爪检查: RELEASED F{rel_f} 锚点 idx{k_rel}; 张爪后手位移 "
              f"{retreat:.0f}mm ({'有撤离' if retreat > 30 else '无撤离!'})")
    else:
        print(f"  撤爪检查: RELEASED 后无锚点 (缺口 F{win_frames[-1]}-F600)")

    # ---- 物体世界轨迹 (mocap 世界 y-up) ----
    T_cw = np.tile(np.eye(4), (F, 1, 1))
    T_cw[:, :3, :3] = R_cam
    T_cw[:, :3, 3] = tpos[rows, 0]
    T_wo = np.einsum("fij,fjk->fik", T_cw, T_co)
    t_wo = T_wo[:, :3, 3]

    # ---- targets: AGILE MANO → sim z-up → SVD 重定向 → pad 600 帧 ----
    # mocap world y-up → sim z-up (与 ee_mocap_targets 同构)
    M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    jw_ef = np.einsum("fij,fnj->fni", R_cam, Jc) + tpos[rows, 0][:, None, :]   # MANO 关节世界 (F,21,3) y-up
    jw = np.einsum("ij,fnj->fni", M, jw_ef)                      # sim z-up
    valid = np.ones(F, bool)
    pos = np.zeros((F, 3)); rot = np.zeros((F, 3, 3)); grip = np.zeros((F, 2))
    for k in range(F):
        p, Rk, jv = svd_gripper_pose(jw[k, 0], jw[k, 4], jw[k, 8], "right")
        pos[k], rot[k], grip[k] = p, Rk, [jv, jv]
    print(f"SVD 重定向: {F} 锚点 → sim pad ({pos.shape} / {rot.shape}) | "
          f"grip {grip[:, 0].min():.4f}~{grip[:, 0].max():.4f} m")

    # 时间轴: 600 帧 (窗) —— 锚点在 win_frames, 上采样补间
    N_OUT = 600
    f_vid = np.arange(N_OUT)
    t_out = np.stack([np.interp(f_vid, win_frames, pos[:, a]) for a in range(3)], 1)
    _sp = float(np.median(np.diff(win_frames))) if F > 1 else 30.0
    rot_u = rotmat_slerp_seq(rot, f_vid / _sp)
    grip_u = np.stack([np.interp(f_vid, win_frames, grip[:, a]) for a in range(2)], 1)

    # grip_states: 继承源标签 (0=APPROACH,1=PICK,2=LIFT,3=HOLD,4=PLACE,5=RELEASED)
    labels = np.load(a.labels, allow_pickle=True)
    src_gs = np.asarray(labels["grip_states"], np.int64) if "grip_states" in labels.files else None
    if src_gs is not None and len(src_gs) >= N_OUT:
        gs = src_gs[:N_OUT].astype(np.int64)
    else:
        gs = np.zeros(N_OUT, np.int64)
        for i, nm in enumerate(["PICK", "LIFT", "HOLD", "PLACE", "RELEASED"]):
            gs[PHASES[nm]:] = i + 1
    print(f"grip_states: 源标签 {'继承' if src_gs is not None and len(src_gs) >= N_OUT else '从 PHASES 重建'}; "
          f"各态分布 {dict(zip(*np.unique(gs, return_counts=True)))}")

    out_tgt = os.path.join(a.out_dir, f"{a.name}_targets_agile_600.npz")
    np.savez(out_tgt, pos=t_out, rot=rot_u, grip=grip_u, valid=np.ones(N_OUT, bool),
             R_c2w=np.tile(np.eye(3), (N_OUT, 1, 1)),
             t_c2w=np.zeros((N_OUT, 3)),
             grip_states=gs,
             grip_state_names=np.array(["APPROACH", "PICK", "LIFT", "HOLD", "PLACE", "RELEASED"]),
             src_frames=win_frames, video_fps=30.0,
             anchor_pos=pos, anchor_anchor_k=win_frames,
             fallback={} )
    print(f"✓ {out_tgt} ({N_OUT} 帧)")

    # ---- 物体: 保留纹理 (UV/MTL/贴图) 的快照 + 碰撞体 ----
    mesh_dir = a.mesh_dir
    mesh_p = os.path.join(aligned, "object_mesh.obj")   # FP scale_mesh 后的米制 mesh
    V_m, Fc = load_obj(mesh_p)
    V_m = V_m * obj_scale * s_anchor
    print(f"textured mesh: {len(V_m)} 顶点 {len(Fc)} 面 (obj_scale {obj_scale:.4f} × anchor {s_anchor:.3f})")
    # 快照先验 PCA 框架 (48 候选 chamfer)
    prior = np.load(a.prior_obj, allow_pickle=True)
    Vp = np.asarray(prior["verts"], np.float64)
    c_m, ax_m = pca_frame(V_m)
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
    A = ax_m[:, list(perm)] * np.array(signs)[None, :]   # signs 是 tuple, 需转 np.array 才能 [None,:]
    s_snap = float(ext_p.mean() / np.sort(ext_m[list(perm)]).mean())
    verts_w = (V_m - c_m) @ A.T + c_p
    print(f"  PCA 对齐 perm {perm} signs {signs} chamfer {chamfer0*1000:.1f}mm | "
          f"尺寸比 {s_snap:.3f}")
    # 碰撞体: 抽稀 (视觉保持密 mesh)
    import open3d as o3d
    _c = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts_w),
        o3d.utility.Vector3iVector(np.asarray(Fc, np.int64)))
    _c = _c.simplify_quadric_decimation(3000)
    V_coll = np.asarray(_c.vertices, np.float64)
    Fc_coll = np.asarray(_c.triangles, np.int64)
    _cc = V_coll.mean(0)
    flip = 0
    for i, (x1, y1, z1) in enumerate(Fc_coll):
        n = np.cross(V_coll[y1] - V_coll[x1], V_coll[z1] - V_coll[x1])
        fc = (V_coll[x1] + V_coll[y1] + V_coll[z1]) / 3
        if n @ (fc - _cc) < 0:
            Fc_coll[i] = [x1, z1, y1]
            flip += 1
    ctr_w, axes_w = pca_frame(V_coll)
    ext_w = np.ptp((V_coll - ctr_w) @ axes_w, axis=0)
    # 视觉面朝向
    for i, (x1, y1, z1) in enumerate(Fc):
        n = np.cross(verts_w[y1] - verts_w[x1], verts_w[z1] - verts_w[x1])
        fc = (verts_w[x1] + verts_w[y1] + verts_w[z1]) / 3
        if n @ (fc - verts_w.mean(0)) < 0:
            Fc[i] = [x1, z1, y1]

    out_obj = os.path.join(a.out_dir, f"{a.name}_object_agile_textured.npz")
    np.savez(out_obj,
             # cb 契约: verts = 世界系 (cb 转 M_EF3R 后算 ext/PCA 建盒碰撞);
             # visual_local_V/F = **质心居中局部系** (cb 写 OBJ, actor 位姿=obj_center)
             verts=V_coll, faces=Fc_coll.astype(np.int64),
             visual_local_V=verts_w - verts_w.mean(0),
             visual_local_F=Fc.astype(np.int64),  # 密纹理
             geometry_kind="agile_textured",
             center_pre=prior["center_pre"], center_placed=prior["center_placed"],
             pca_center=ctr_w, pca_axes=axes_w, pca_ext=ext_w,
             support_placed=prior["support_placed"],
             surface_pick_V=prior["surface_pick_V"], surface_pick_F=prior["surface_pick_F"],
             surface_place_V=prior["surface_place_V"], surface_place_F=prior["surface_place_F"],
             obj_scale_m=obj_scale, obj_scale_anchored=obj_scale * s_anchor,
             # 校准铁圈 contract (cb ring-place 必需, cb:388)
             ring_segment_count=np.array([16]),
             ring_radius_m=np.array([0.1102]),
             traj_T_pos=t_wo, traj_T_rot=T_wo[:, :3, :3],
             texture_mtl=f"{mesh_dir}/material.mtl",
             texture_png=f"{mesh_dir}/material_0.png",
             texture_uv_exists=os.path.exists(f"{mesh_dir}/material_0.png"),
             provenance=f"AGILE rel-opt ({c2w_name}) + SAM3D textured, PCA snap "
                        f"chamfer {chamfer0*1000:.1f}mm, coll decim 3k")
    print(f"✓ {out_obj} (碰撞 {len(V_coll)}v/{len(Fc_coll)}f + 视觉 "
          f"{len(verts_w)}v/{len(Fc)}f 纹理保留)")

    reg = {"c2w_convention": c2w_name,
           "hand_chain_mean_mm": {k: v[1] for k, v in convs.items()},
           "mocap_depth_anchor_s": s_anchor,
           "mesh_scale_prior_over_anchor": s_snap,
           "anchor_frames_600": win_frames.tolist(),
           "phase_gaps": {nm: int(np.abs(win_frames - f).min()) for nm, f in PHASES.items()},
           "targets": out_tgt, "object": out_obj,
           "obj_scale_agile": obj_scale,
           "deliverable_note": "锚点从 rel-opt (55帧真值全程) 覆盖 F0..597; "
                               "后段补间/回退帧在 anchor_anchor_k 标注"}
    reg_p = os.path.join(a.out_dir, f"{a.name}_registration.json")
    json.dump(reg, open(reg_p, "w"), indent=1, ensure_ascii=False)
    print(f"✓ {reg_p}")


if __name__ == "__main__":
    main()
