#!/usr/bin/env python3
"""agile_object.py — AGILE 物体 (SAM3D mesh + FoundationPose 6D 轨迹) → object_instance.npz。

与 ef3r_object.py 的输出契约完全一致 (cb_large_range_sim.py 的 object_npz 消费端
不区分来源), 但物体几何/轨迹来自 AGILE 链 (物理优化后的 mesh 尺度 + 逐帧位姿):

  SAM3D textured_mesh → FoundationPose --scale_mesh → <prep>/object_pose/scaled_mesh.obj
  AGILE 优化 object_scale (aligned/object_poses.npz) → 最终模型系顶点 = scaled_mesh × scale
  T_world_obj(k) = EF3R c2w[video_frame(k)] @ T_cam_obj(k)   → EF3R 世界 (y-up)
  上采样到视频 300 帧时间轴 = traj_T

被抓手 actor 在 cb 端出生在夹爪抓取点 (pad@k_lift) 并做 PCA 轴对齐, 因此
verts 只需在世界系里"摆一个合理快照" (取 PICK 段中帧的物体位姿), 形状/尺寸/
PCA 轴才是被真正消费的字段。支撑台面 (surface_pick/place, support_placed) 是
场景几何, 与手链无关 → 直接复用 EF3R object_instance.npz 的拟合结果。

输出: <ef3r-dir>/object_instance_agile.npz
  verts/faces (EF3R 世界系, PICK 段快照) · center_pre/center_placed ·
  pca_center/pca_axes/pca_ext · traj_T (300,4,4) · surface_pick_V/F ·
  surface_place_V/F · support_placed · obj_scale_m (AGILE 标定, 米)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agile_to_targets import rotmat_slerp_seq  # noqa: E402  复用同一条时间轴上采样


def load_obj(path):
    vs, fs = [], []
    for line in open(path):
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            fs.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
    return np.array(vs, np.float64), np.array(fs, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agile-dir", required=True, help="AGILE output/<seq> 目录")
    ap.add_argument("--ef3r-dir", required=True, help="EgoFound3R outputs/<clip> 目录")
    ap.add_argument("--subsample", type=int, default=3)
    ap.add_argument("--out-num", type=int, default=300)
    ap.add_argument("--snapshot-frame", type=int, default=None,
                    help="静态快照帧 (视频时间轴); 默认 PICK 段中帧")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    aligned = os.path.join(args.agile_dir, "runs", "opt", "aligned")
    op = np.load(os.path.join(aligned, "object_poses.npz"))
    T_co = op["object_poses_in_cam"].astype(np.float64)      # (F,4,4) 相机→物体
    obj_scale = float(np.asarray(op["object_scale"]).ravel()[0])
    cd = np.load(os.path.join(aligned, "camera_data.npz"))
    vid = np.asarray(cd["prediction_frame_ids"], np.int64) * args.subsample

    # 相机系相似变换 (agile_to_targets.py 拟合并落盘): AGILE (MoGe 米制) → EF3R 米制
    # 物体位姿左乘 D; mesh 尺寸同乘 s (两条链米制标定差 ~1.8x, 见 agile_to_targets)
    sim_p = os.path.join(args.agile_dir, "cam_similarity_to_ef3r.npz")
    if os.path.exists(sim_p):
        sm = np.load(sim_p)
        s_al, R_al, t_al = float(sm["s"]), np.asarray(sm["R"]), np.asarray(sm["t"])
        print(f"相机系相似变换: s={s_al:.3f} (残差 {sm['resid_mean_mm']:.1f}mm)")
    else:
        raise SystemExit(f"缺 {sim_p} —— 先跑 agile_to_targets.py (相似变换在那里拟合)")
    D = np.eye(4); D[:3, :3] = s_al * R_al; D[:3, 3] = t_al
    T_co = np.einsum("ij,fjk->fik", D, T_co)

    c2w = np.load(os.path.join(args.ef3r_dir, "camera", "traj_c2w.npy")).astype(np.float64)
    Nv = c2w.shape[0]

    # ---- 模型系顶点 (EF3R 米制): scaled_mesh × AGILE object_scale × 相似尺度 ----
    prep = os.path.join(args.agile_dir, "seq", "object_pose")
    mesh_p = os.path.join(prep, "scaled_mesh.obj")
    V_m, Fc = load_obj(mesh_p)
    V_m = V_m * obj_scale * s_al
    print(f"物体 mesh: {len(V_m)} 顶点 {len(Fc)} 面 (scaled_mesh × {obj_scale:.4f} × s{s_al:.3f}), "
          f"模型系包围盒 {np.round(np.ptp(V_m, 0) * 1000, 1)} mm")

    # ---- 逐帧世界系位姿 → 上采样到视频时间轴 ----
    T_w = np.einsum("fij,fjk->fik", c2w[vid], T_co)          # (F,4,4) EF3R 世界
    n_out = args.out_num or len(vid)
    f_out = np.arange(n_out)
    f_vid = f_out * (Nv / n_out)                              # 输出帧 → 视频帧
    _sp = float(np.median(np.diff(vid)))
    t_out = np.stack([np.interp(f_vid, vid, T_w[:, a, 3]) for a in range(3)], 1)
    R_out = rotmat_slerp_seq(T_w[:, :3, :3], f_vid / _sp)
    traj_T = np.concatenate([R_out, t_out[:, :, None]], 2)   # (300,4,4)
    print(f"物体轨迹: AGILE {len(vid)} 帧 → 上采样 {n_out} 帧 (EF3R 世界系 y-up)")

    # ---- 尺寸/PCA (模型系算, 与世界系快照同构 —— 刚体不变形) ----
    ctr = V_m.mean(0)
    _, _, Vt = np.linalg.svd(V_m - ctr)
    axes_m = Vt.T                                            # [:,0] 最长 … [:,2] 最薄
    ext_m = np.ptp((V_m - ctr) @ axes_m, axis=0)
    print(f"PCA 轴长 {np.round(ext_m * 1000, 1)} mm (长→薄)")

    # ---- 静态快照: PICK 段中帧的物体世界位姿 ----
    states = {}
    sl = os.path.join(args.ef3r_dir, "state_labels.json")
    if os.path.exists(sl):
        states = json.load(open(sl))["states"]
    if args.snapshot_frame is not None:
        f_snap = args.snapshot_frame
    elif "PICK" in states:
        f_snap = (states["PICK"][0] + states["PICK"][1]) // 2
    else:
        f_snap = int(vid[len(vid) // 2])
    k_snap = int(np.argmin(np.abs(vid - f_snap)))            # 最近 AGILE 帧
    R_s, t_s = T_w[k_snap, :3, :3], T_w[k_snap, :3, 3]
    verts_w = (R_s @ V_m.T).T + t_s                          # 模型系→世界系刚体变换
    print(f"快照: 视频帧 {f_snap} (AGILE 帧 {vid[k_snap]}), 物体原点 {np.round(t_s, 3)}")

    # 面片朝向修正 (凸包质心在内部; SAPIEN 静默背面剔除, 与 ef3r_object 同坑)
    _c = verts_w.mean(0)
    for i, (a, b, c3) in enumerate(Fc):
        n = np.cross(verts_w[b] - verts_w[a], verts_w[c3] - verts_w[a])
        fc = (verts_w[a] + verts_w[b] + verts_w[c3]) / 3
        if n @ (fc - _c) < 0:
            Fc[i] = [a, c3, b]

    # PCA 在世界系快照上重算 (verts 与 pca_* 必须同系, cb 端同变换)
    ctr_w = verts_w.mean(0)
    _, _, Vt_w = np.linalg.svd(verts_w - ctr_w)
    axes_w = Vt_w.T
    ext_w = np.ptp((verts_w - ctr_w) @ axes_w, axis=0)

    # ---- 放置参考: RELEASED 段末端的物体中心 ----
    if "RELEASED" in states:
        f_end = min(states["RELEASED"][1] - 5, n_out - 1)
    else:
        f_end = n_out - 1
    c_placed = traj_T[f_end, :3, 3]
    c_pre = t_s
    print(f"center_pre {np.round(c_pre, 3)}  center_placed {np.round(c_placed, 3)}")

    # ---- 支撑台面/放置支撑: 复用 EF3R 场景拟合 (与手链无关) ----
    ef3r_obj_p = os.path.join(args.ef3r_dir, "object_instance.npz")
    extra = {}
    if os.path.exists(ef3r_obj_p):
        eo = np.load(ef3r_obj_p)
        for k in ("surface_pick_V", "surface_pick_F", "surface_place_V",
                  "surface_place_F", "support_placed"):
            if k in eo.files:
                extra[k] = eo[k]
        print(f"复用 EF3R 支撑台面/支撑面字段: {sorted(k for k in extra)}")

    out = args.out or os.path.join(args.ef3r_dir, "object_instance_agile.npz")
    np.savez(out,
             verts=verts_w.astype(np.float64), faces=Fc.astype(np.int64),
             center_pre=c_pre, center_placed=c_placed,
             obj_points=V_m, n_points=len(V_m),
             pca_center=ctr_w, pca_axes=axes_w, pca_ext=ext_w,
             traj_T=traj_T, obj_scale_m=obj_scale, snapshot_frame=f_snap,
             **extra)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
