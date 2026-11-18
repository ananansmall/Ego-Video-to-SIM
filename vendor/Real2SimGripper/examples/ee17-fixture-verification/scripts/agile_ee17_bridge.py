#!/usr/bin/env python3
"""agile_ee17_bridge.py — AGILE 链 (ee17 漏斗) → R1 object npz + 配准误差记录。

用户配方步骤②的 ee17 适配 (mocap 直通路径, 无 EF3R):
  - c2w: EgoExplorer mocap tracker_rotation[:,0] (camera tracker, 逐帧世界旋转)
    + tracker_position[:,0] —— y-up mocap 世界。**相机约定数据驱动择优**:
    VR tracker 常见 Unity 约定 (z-backward) vs AGILE OpenCV (z-forward) 差
    diag(1,-1,-1); 4 候选 (raw/attachment × ±flip) 用"MANO 腕 vs mocap
    right_wrist tracker"距离择优 —— 正确约定应落到解剖偏置量级 (~0.1-0.2m)。
  - 几何: aligned/object_mesh.obj (AGILE 优化后 mesh) 按**先验 object npz 的
    PCA 框架**摆快照 (48 候选: 6 轴排列 × 8 符号, chamfer 择优), 均匀尺度对齐
    先验 (MoGe 绝对尺度不可靠, 先验尺度已经物理抓取验证) —— 尺度比入 JSON。
  - targets: R1 继续用 mocap 直通 (真值); AGILE MANO 手链只做交叉验证。
  - 配准误差 (无 Δ 原点/旋转依赖):
      1. c2w_carry_residual_mm   HOLD 段 Δ物体世界 ≈ Δ腕世界 (c2w 质量探针)
      2. obj_wrist_dist_mm       抓取前 |物-腕| AGILE(MoGe 米制) vs
         mocap/sim (center_pre vs pad) —— 尺度/位置一致性 (单帧级)
      3. hand_chain_check_mm     MANO 腕 (j0) 经 c2w → mocap 世界 vs
         right_wrist tracker (mean=解剖偏置, std=配准质量; 兼作约定择优)
      4. mesh_scale_ratio        AGILE(MoGe) / 先验(腕锚定) 尺寸比

用法 (agile-optim env, 链完成后):
    python scripts/agile_ee17_bridge.py
输出: <out>/object_instance_agile.npz + <out>/agile_ee17_registration.json
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = "/path/to/Real2SimGripper"

AGILE = "/path/to/project_data/mano_ws/AGILE/output/ee17_release_s"
MOCAP = ("/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
         "/episode_meta/episode_000017/smplx/smplx_episode.npz")
OUT_DIR = "/path/to/project_data/mano_ws/EgoFound3R/outputs/ee17_win"
PRIOR_OBJ = os.path.join(OUT_DIR, "object_instance.npz")
TARGETS = os.path.join(ROOT, "dataset/ee17_mocap/robot/targets.npz")
FRAME_MAP = os.path.join(AGILE, "subset_frame_map.npy")

# 释放窗阶段帧 (600 帧窗系): PICK 47 / LIFT 53 / HOLD 106 / PLACE 448 / RELEASED 534
PICK, HOLD0, HOLD1 = 47, 106, 448
FLIP_X = np.diag([1.0, -1.0, -1.0])          # Unity(z-back) → OpenCV(z-fwd)


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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    aligned = os.path.join(AGILE, "runs", "opt", "aligned")

    # ---- AGILE aligned 输出 ----
    Jc = np.load(os.path.join(aligned, "optimized_mano_joints.npz"))["mano_joints"]
    Jc = Jc[:, 0].astype(np.float64)                       # (F,21,3) 相机系 m
    op = np.load(os.path.join(aligned, "object_poses.npz"))
    T_co = op["object_poses_in_cam"].astype(np.float64)    # (F,4,4) 相机→物
    obj_scale = float(np.asarray(op["object_scale"]).ravel()[0])
    cd = np.load(os.path.join(aligned, "camera_data.npz"))
    pred = np.asarray(cd["prediction_frame_ids"], np.int64)  # 子集帧号
    F = len(pred)
    fmap = np.load(FRAME_MAP)                              # 子集→原200帧
    orig = fmap[pred]                                      # 原 200 帧系
    win_frames = orig * 3                                  # 600 帧窗系 (30fps)
    print(f"AGILE aligned: {F} 帧 (子集 {pred} → 原 {orig} → 窗 {win_frames}) "
          f"| object_scale {obj_scale:.4f}")

    # ---- mocap 相机 c2w (窗帧 j → 绝对视频帧 378*30+j → mocap 行) ----
    d = np.load(MOCAP, allow_pickle=True)
    vfi = np.asarray(d["video_frame_index"], np.int64)
    tpos = np.asarray(d["tracker_position"], np.float64)   # (N,5,3)
    trot = np.asarray(d["tracker_rotation"], np.float64)   # (N,5,3,3)
    att_r = np.asarray(d["attachment_rotation"], np.float64)[0]  # cam 挂载
    base_abs = 378 * 30
    rows = np.searchsorted(vfi, base_abs + win_frames)
    ok = (rows < len(vfi)) & (vfi[np.clip(rows, 0, len(vfi) - 1)] == base_abs + win_frames)
    assert ok.mean() > 0.8, f"mocap 行映射失败 ({ok.mean():.0%} 命中)"
    rows = np.clip(rows, 0, len(vfi) - 1)
    rw_w = tpos[rows, 2]                                   # right_wrist 世界
    print(f"mocap c2w: {len(rows)} 行映射 (命中 {ok.mean():.0%})")

    # ---- mocap 深度锚定: MoGe 绝对尺度不可靠 (FOV/f 标定误差 ~2×, 实测腕深
    # 0.15-0.41m vs MoGe 0.29-0.76m) —— 以 mocap 真值腕深为锚, 全链米制 ×s 重标定
    from PIL import Image
    w2d = np.load(f"{AGILE}/seq/wilor_raw/mano_joints_2d.npz")["mano_joints_2d"][:, 0].astype(float)
    ratios = []
    for k, p in enumerate(pred):
        R0 = trot[rows[k], 0]
        rw_cam = R0.T @ (tpos[rows[k], 2] - tpos[rows[k], 0])
        u, v = w2d[p, 0].astype(int)
        dep = np.array(Image.open(f"{AGILE}/seq/moge/depth/{p:05d}.png"), float) / 1000.0
        v2, u2 = int(np.clip(v, 0, dep.shape[0] - 1)), int(np.clip(u, 0, dep.shape[1] - 1))
        patch = dep[max(0, v2 - 4):v2 + 5, max(0, u2 - 4):u2 + 5]
        pv = patch[patch > 0]
        if len(pv) and rw_cam[2] > 0.02:
            ratios.append(rw_cam[2] / float(np.median(pv)))
    s_anchor = float(np.median(ratios))
    print(f"mocap 深度锚定: s = {s_anchor:.3f} "
          f"(n={len(ratios)}, IQR {np.percentile(ratios,25):.3f}~"
          f"{np.percentile(ratios,75):.3f}) —— AGILE 米制 ×s")
    Jc *= s_anchor
    T_co[:, :3, 3] *= s_anchor

    # ---- 相机约定 4 候选, 用手链交叉距离择优 (正确约定 → 解剖偏置量级) ----
    static = win_frames < PICK
    hold = (win_frames >= HOLD0) & (win_frames < HOLD1)
    convs = {}
    for nm_att, R0 in (("raw", trot[rows, 0]), ("attachment", trot[rows, 0] @ att_r[None])):
        for nm_flip, FL in (("zfwd", np.eye(3)), ("zback", FLIP_X)):
            R = R0 @ FL[None]
            jw = np.einsum("fij,fj->fi", R, Jc[:, 0]) + tpos[rows, 0]
            convs[f"{nm_att}_{nm_flip}"] = (R, float(np.linalg.norm(jw - rw_w, axis=1).mean() * 1000))
    for k, (R, mm) in convs.items():
        print(f"  相机约定 {k}: MANO腕-mocap腕 mean {mm:.0f} mm")
    c2w_name = min(convs, key=lambda k: convs[k][1])
    R_cam = convs[c2w_name][0]
    print(f"  选用: {c2w_name}")

    # ---- 物体世界轨迹 (y-up mocap 世界): T_wo = T_cw @ T_co ----
    T_cw = np.tile(np.eye(4), (F, 1, 1))
    T_cw[:, :3, :3] = R_cam
    T_cw[:, :3, 3] = tpos[rows, 0]
    T_wo = np.einsum("fij,fjk->fik", T_cw, T_co)
    t_wo = T_wo[:, :3, 3]

    # ---- 指标 1: HOLD 搬运增量一致性 (Δ物体 ≈ Δ腕) ----
    d_obj = np.diff(t_wo[hold], axis=0)
    d_rw = np.diff(rw_w[hold], axis=0)
    carry_mm = float(np.median(np.linalg.norm(d_obj - d_rw, axis=1)) * 1000)
    print(f"  HOLD 搬运增量残差 median {carry_mm:.1f} mm")

    # ---- 指标 2: 抓取前 |物-腕| 一致性 (AGILE 米制 vs mocap/sim 系) ----
    dist_agile = np.linalg.norm(T_co[:, :3, 3] - Jc[:, 0], axis=1) * 1000
    tg = np.load(TARGETS)
    pos_sim = np.asarray(tg["pos"], np.float64)            # (600,3) sim pad
    prior = np.load(PRIOR_OBJ, allow_pickle=True)
    c_pre = np.asarray(prior["center_pre"], np.float64)
    w0 = np.clip(win_frames[static], 0, len(pos_sim) - 1)
    dist_mocap_pre = float(np.linalg.norm(pos_sim[w0] - c_pre[None], axis=1).mean() * 1000) \
        if static.any() else None
    dist_agile_pre = float(dist_agile[static].mean()) if static.any() else None
    hold_off = float(dist_agile[hold].mean()) if hold.any() else None
    print(f"  抓取前 |物-腕|: AGILE {dist_agile_pre} vs mocap/sim {dist_mocap_pre} mm")
    print(f"  HOLD 抓取偏移 |物-腕| (AGILE): {hold_off} mm")

    # ---- 几何: AGILE 优化 mesh → 先验 PCA 框架快照 (48 候选 chamfer 择优) ----
    V_m, Fc = load_obj(os.path.join(aligned, "object_mesh.obj"))
    V_m = V_m * obj_scale * s_anchor                       # 模型→米 (锚定尺度)
    print(f"AGILE mesh: {len(V_m)} 顶点 {len(Fc)} 面, 包围盒 "
          f"{np.round(np.ptp(V_m, 0) * 1000, 1)} mm")
    c_m, ax_m = pca_frame(V_m)
    Vp = np.asarray(prior["verts"], np.float64)            # 先验世界快照
    c_p, ax_p = pca_frame(Vp)
    ext_m = np.ptp((V_m - c_m) @ ax_m, axis=0)
    ext_p = np.ptp((Vp - c_p) @ ax_p, axis=0)
    print(f"  PCA 轴长 AGILE {np.round(ext_m*1000,1)} vs 先验 {np.round(ext_p*1000,1)} mm")
    # 轴排列 × 符号 全搜: 先验顶点是最粗真值 (同漏斗), chamfer 最小者即对齐
    Vs = V_m[np.random.default_rng(0).choice(len(V_m), min(1500, len(V_m)), replace=False)]
    Vs_c = Vs - c_m
    tree = cKDTree(Vp - c_p)
    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            A = ax_m[:, list(perm)] * np.array(signs)[None, :]
            P = Vs_c @ A                                   # AGILE 点转到候选框架
            dd, _ = tree.query(P)
            sc = float(dd.mean())
            if best is None or sc < best[0]:
                best = (sc, np.array(perm), np.array(signs))
    chamfer0, perm, signs = best
    print(f"  PCA 对齐: perm {perm} signs {signs} chamfer {chamfer0*1000:.1f} mm")
    A = ax_m[:, list(perm)] * signs[None, :]               # 模型系→先验轴向
    # 尺度: mocap 锚定后的 AGILE 米制 (不再强推先验尺寸) —— 与先验的比值入 JSON
    ext_agile_matched = np.sort(ext_m[list(perm)])
    s_snap = float(ext_p.mean() / ext_agile_matched.mean())
    print(f"  尺寸比 (先验/锚定后AGILE): {s_snap:.3f} (≈1 = 两独立尺度链收敛)")
    verts_w = (V_m - c_m) @ A.T + c_p
    # 抽稀: 20k 面 SAPIEN 凸分解过重 (先验代理 110 面), 二次误差 3k 面 ≈ 亚毫米级
    if len(Fc) > 4000:
        import open3d as o3d
        _m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts_w),
            o3d.utility.Vector3iVector(np.asarray(Fc, np.int64)))
        _m = _m.simplify_quadric_decimation(3000)
        verts_w = np.asarray(_m.vertices, np.float64)
        Fc = np.asarray(_m.triangles, np.int64)
        print(f"  抽稀: -> {len(verts_w)} 顶点 {len(Fc)} 面")
        _cc = verts_w.mean(0)
    # 面片朝向 (凸体质心在内, SAPIEN 背面剔除坑)
    _cc = verts_w.mean(0)
    flip = 0
    for i, (a, b, c3) in enumerate(Fc):
        n = np.cross(verts_w[b] - verts_w[a], verts_w[c3] - verts_w[a])
        fc = (verts_w[a] + verts_w[b] + verts_w[c3]) / 3
        if n @ (fc - _cc) < 0:
            Fc[i] = [a, c3, b]
            flip += 1
    ctr_w, axes_w = pca_frame(verts_w)
    ext_w = np.ptp((verts_w - ctr_w) @ axes_w, axis=0)

    out_npz = os.path.join(OUT_DIR, "object_instance_agile.npz")
    op_sample = verts_w[:: max(1, len(verts_w) // 400)].copy()
    np.savez(out_npz,
             verts=verts_w, faces=Fc.astype(np.int64),
             center_pre=prior["center_pre"], center_placed=prior["center_placed"],
             obj_points=op_sample, n_points=len(op_sample),
             pca_center=ctr_w, pca_axes=axes_w, pca_ext=ext_w,
             support_placed=prior["support_placed"],
             surface_pick_V=prior["surface_pick_V"], surface_pick_F=prior["surface_pick_F"],
             surface_place_V=prior["surface_place_V"], surface_place_F=prior["surface_place_F"],
             obj_scale_m=obj_scale,
             traj_T_pos=t_wo, traj_T_rot=T_wo[:, :3, :3],
             provenance="agile_ee17_release_s (SAM3D+FoundationPose+AGILE contact opt); "
                        "snapshot in prior PCA frame, uniform scale to prior size")
    reg = {
        "c2w_convention": c2w_name,
        "hand_chain_mean_mm": {k: v[1] for k, v in convs.items()},
        "carry_residual_hold_mm": carry_mm,
        "obj_wrist_pre_agile_mm": dist_agile_pre,
        "obj_wrist_pre_mocap_sim_mm": dist_mocap_pre,
        "obj_wrist_hold_agile_mm": hold_off,
        "mesh_scale_prior_over_anchor": s_snap,
        "mocap_depth_anchor_s": s_anchor,
        "mesh": {"verts": int(len(V_m)), "faces": int(len(Fc)),
                 "flip_faces": flip, "pca_ext_mm": ext_m.tolist(),
                 "prior_pca_ext_mm": ext_p.tolist(),
                 "snapshot_chamfer_mm": chamfer0 * 1000,
                 "pca_perm": perm.tolist(), "pca_signs": signs.tolist()},
        "object_scale_agile": obj_scale,
        "aligned_frames": {int(p): int(o) for p, o in zip(pred, orig)},
        "window_frames": win_frames.tolist(),
    }
    reg_p = os.path.join(OUT_DIR, "agile_ee17_registration.json")
    json.dump(reg, open(reg_p, "w"), indent=1, ensure_ascii=False)
    print(f"✓ {out_npz}")
    print(f"✓ {reg_p}")


if __name__ == "__main__":
    main()
