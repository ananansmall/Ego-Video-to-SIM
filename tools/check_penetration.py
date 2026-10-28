"""
check_penetration.py - 手部与物体 mesh 的穿模/接触检查
=====================================================

前提: RAS 场景目录里已有物体 mesh (all_instances.pkl, 含 4x4 变换 T, T 内带缩放)
      且 extrinsics/ 已应用房间对齐, 与物体在同一世界系。

做法:
  1) 读 EgoFound3R 的手 (metric), × g 换算到 RAS raw 尺度, 再加精修平移
  2) 用 RAS 的外参 (w2c) 逆变换到世界系:  X_w = R^T (x_c - t)
  3) 对每个物体 mesh (open3d 降采样) 求有符号距离 (trimesh signed_distance)
        d < 0     → 穿模 (手在物体内部)
        d < 1cm   → 接触
  4) 渲染: 物体点云(灰) + 手, 穿模顶点标红; 输出统计 json

用法:
  python tools/check_penetration.py \
      --scene-dir output_v3/egofound_verify/bpp111 \
      --ras-dir output_v3/basic_pick_place_111
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer_html as MV

SIDE_COLORS_BGR = [(0, 165, 255), (255, 128, 0)]


def load_ras(dirpath: Path, simplify_faces=6000):
    K = np.loadtxt(dirpath / "intrinsic.txt")
    ext_files = sorted((dirpath / "extrinsics").glob("*.txt"), key=lambda p: int(p.stem))
    E = np.stack([np.loadtxt(p) for p in ext_files])
    import pickle
    d = pickle.load(open(dirpath / "all_instances.pkl", "rb"))
    objs = []
    for cat, lst in d["all_instances"].items():
        for k, o in enumerate(lst):
            m = o["original_mesh"].copy()
            T = np.asarray(o["T"], dtype=np.float64)
            m.apply_transform(T)
            if simplify_faces and len(m.faces) > simplify_faces:
                try:
                    import open3d as o3d
                    o3 = o3d.geometry.TriangleMesh(
                        o3d.utility.Vector3dVector(np.asarray(m.vertices)),
                        o3d.utility.Vector3iVector(np.asarray(m.faces)))
                    o3 = o3.simplify_quadric_decimation(simplify_faces)
                    m = trimesh.Trimesh(np.asarray(o3.vertices), np.asarray(o3.triangles), process=False)
                except Exception as e:
                    print(f"[WARN] 降采样失败 {cat}[{k}]: {e}")
            objs.append({"name": f"{cat}[{k}]", "mesh": m})
            print(f"[RAS] {cat}[{k}] 顶点={len(m.vertices)} 面={len(m.faces)} "
                  f"bbox={np.ptp(m.vertices, axis=0).round(3)}")
    return K, E, objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--ras-dir", type=Path, required=True)
    ap.add_argument("--ego-root", type=Path, default=MV.DEFAULT_EGO_ROOT)
    ap.add_argument("--contact-cm", type=float, default=1.0)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--yaw-deg", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    import torch
    from PIL import Image

    D = args.scene_dir
    payload = torch.load(D / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    m = json.loads((D / "verify" / "metrics.json").read_text())
    g = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter") or \
        (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
    V = MV.tnp(payload["hand"]["vertex_xyz_metric"])
    hv = MV.tnp(payload["hand"]["valid"]).astype(bool)
    CP = MV.tnp(payload["hand"]["vertex_contact_probability"])
    faces = MV.load_marker_faces(args.ego_root)
    print(f"[EGO] g={g:.4f}  T_R={V.shape[0]}")

    # 精修平移 (若已跑)
    tpath = D / "refine" / "refined_hand.npz"
    tm = {}
    if tpath.exists():
        r = np.load(tpath)
        tm = {int(i): r["t"][k] for k, i in enumerate(r["frames"])}
        print(f"[EGO] 使用精修平移 ({len(tm)} 帧)")

    K, E, objs = load_ras(args.ras_dir, simplify_faces=6000)
    # RAS 的深度点云 (渲染背景用)
    dep_files = sorted((args.ras_dir / "depth").glob("*"), key=lambda p: int(p.stem))
    N = min(len(E), V.shape[0])
    CM2RAW = 0.01 * g

    # 相机系点云: 用 RAS 深度 (与 RAS 外参/物体同一来源, 保证一致性)
    def ras_cloud(fi, step=2):
        f = dep_files[fi]
        if f.suffix == ".npy":
            d = np.load(f).astype(np.float32)
        else:
            d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32)
            if d is not None and d.max() > 200:
                d = d / 1000.0
        if d is None:
            return None
        if d.ndim == 3:
            d = d[..., 0]
        H, W = d.shape
        v = np.isfinite(d) & (d > 1e-6)
        s = np.zeros_like(v); s[::step, ::step] = True; s &= v
        ys, xs = np.nonzero(s)
        z = d[ys, xs]
        return np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                         (ys - K[1, 2]) / K[1, 1] * z, z], 1)

    stats, gif = [], []
    th = np.deg2rad(args.yaw_deg); cy, sy = np.cos(th), np.sin(th)
    light = np.array([0.35, -0.55, -0.76]); light /= np.linalg.norm(light)

    for fi in range(N):
        best = None
        for sd in range(2):
            if not hv[fi, sd]:
                continue
            v = V[fi, sd] * g + tm.get(fi, np.zeros(3))
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 30:
                continue
            R, t = E[fi][:3, :3], E[fi][:3, 3]
            vw = (R.T @ (v - t).T).T
            # 有符号距离: 取所有物体的最小值 (负=穿模)
            sd_all = np.full(len(vw), 1e3)
            for o in objs:
                try:
                    pq = trimesh.proximity.ProximityQuery(o["mesh"])
                    # trimesh >= 4.11: signed_distance 正=内部, 负=外部 (与旧版相反)
                    # 统一成 "负=穿模" 的常规约定
                    d = -pq.signed_distance(vw)
                except Exception:
                    d = trimesh.proximity.closest_point(o["mesh"], vw)[1]
                sd_all = np.minimum(sd_all, d)
            pen = sd_all < 0
            rec = {
                "frame": fi, "side": sd,
                "n_vertices": int(ok.sum()),
                "penetrating_vertices": int(pen.sum()),
                "max_penetration_cm": float(-sd_all.min() / CM2RAW) if sd_all.min() < 0 else 0.0,
                "contact_vertices_1cm": int((sd_all < args.contact_cm * CM2RAW).sum()),
                "min_dist_cm": float(sd_all.min() / CM2RAW),
                "median_dist_cm": float(np.median(sd_all) / CM2RAW),
            }
            stats.append(rec)
            if best is None or rec["penetrating_vertices"] < best[0]["penetrating_vertices"]:
                best = (rec, vw, ok, sd, sd_all)
        if best is None:
            continue
        rec, vw, ok, sd, sd_all = best

        # ---- 渲染 ----
        Pc = ras_cloud(fi)
        S = args.size
        img = np.zeros((S, S, 3), np.uint8)
        def proj(Q):
            return cy * Q[:, 0] + sy * Q[:, 2], Q[:, 1], -sy * Q[:, 0] + cy * Q[:, 2]
        allp = [vw] + [o["mesh"].vertices[::7] for o in objs]
        if Pc is not None:
            allp.append((E[fi][:3, :3].T @ (Pc - E[fi][:3, 3]).T).T)
        allp = np.vstack(allp)
        x, y, _ = proj(allp)
        lo = np.array([np.percentile(x, 2), np.percentile(y, 2)])
        hi = np.array([np.percentile(x, 98), np.percentile(y, 98)])
        span = float(max(hi - lo)) * 1.05; ctr = (hi + lo) / 2
        px_ = lambda X: ((X - ctr[0]) / span * (S - 20) + S / 2).astype(np.int32)
        py_ = lambda Y: ((Y - ctr[1]) / span * (S - 20) + S / 2).astype(np.int32)

        if Pc is not None:
            Pw = (E[fi][:3, :3].T @ (Pc - E[fi][:3, 3]).T).T
            xx, yy, dd = proj(Pw)
            X, Y = px_(xx), py_(yy)
            okp = (X >= 0) & (X < S - 1) & (Y >= 0) & (Y < S - 1)
            for j in np.nonzero(okp)[0][::2]:
                img[Y[j], X[j]] = (90, 90, 90)
        # 物体 mesh 顶点
        for o in objs:
            ov = o["mesh"].vertices[::5]
            xx, yy, dd = proj(ov)
            X, Y = px_(xx), py_(yy)
            okp = (X >= 0) & (X < S - 1) & (Y >= 0) & (Y < S - 1)
            for j in np.nonzero(okp)[0]:
                img[Y[j], X[j]] = (200, 200, 160)
        # 手 (穿模顶点红, 接触橙, 其他蓝/橙)
        hx, hy, hd = proj(vw)
        hpx, hpy = px_(hx), py_(hy)
        V3 = np.stack([hx, hy, hd], 1)
        tris = []
        for f in faces:
            tri = np.stack([hpx[list(f)], hpy[list(f)]], 1)
            if not ((tri[:, 0] > -S).all() and (tri[:, 0] < 2 * S).all()
                    and (tri[:, 1] > -S).all() and (tri[:, 1] < 2 * S).all()):
                continue
            p0, p1, p2 = V3[f[0]], V3[f[1]], V3[f[2]]
            n = np.cross(p1 - p0, p2 - p0); ln = np.linalg.norm(n)
            sh = 0.45 + 0.55 * abs(float(n @ light) / ln) if ln > 1e-12 else 0.8
            base = SIDE_COLORS_BGR[sd]
            if sd_all[list(f)].min() < 0:
                base = (40, 40, 255)          # 穿模: 红
            elif sd_all[list(f)].min() < args.contact_cm * CM2RAW:
                base = (60, 190, 255)         # 接触: 橙黄
            tris.append((float(hd[list(f)].mean()), tri,
                         [min(255, int(c * sh)) for c in base]))
        for _, tri, col in sorted(tris, key=lambda t: -t[0]):
            cv2.fillPoly(img, [tri.astype(np.int32)], col, cv2.LINE_AA)
        cv2.putText(img, f"f{fi} side{sd} | pen verts={rec['penetrating_vertices']} "
                         f"max={rec['max_penetration_cm']:.1f}cm | contact<1cm={rec['contact_vertices_1cm']} "
                         f"| min={rec['min_dist_cm']:.2f}cm",
                    (8, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (225, 225, 225), 1, cv2.LINE_AA)
        gif.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                   .convert("P", palette=Image.ADAPTIVE))

    out_dir = D / "penetration"
    out_dir.mkdir(parents=True, exist_ok=True)
    if stats:
        pen = [s["penetrating_vertices"] for s in stats]
        mx = [s["max_penetration_cm"] for s in stats]
        con = [s["contact_vertices_1cm"] for s in stats]
        print(f"\n[PEN] 帧数={len(stats)}")
        print(f"[PEN] 穿模顶点: 中位 {int(np.median(pen))}, 最多 {max(pen)}, 有穿模的帧 {int(sum(1 for p in pen if p>0))}/{len(pen)}")
        print(f"[PEN] 最大穿深: 中位 {np.median(mx):.2f} cm, 最深 {max(mx):.2f} cm")
        print(f"[PEN] 接触顶点(<1cm): 中位 {int(np.median(con))}, 最多 {max(con)}")
        (out_dir / "penetration_stats.json").write_text(
            json.dumps({"g": float(g), "per_frame": stats}, indent=2, ensure_ascii=False))
    if gif:
        gp = out_dir / "penetration.gif"
        gif[0].save(gp, save_all=True, append_images=gif[1:],
                    duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"[PEN] {gp}  ({len(gif)} 帧, {gp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
