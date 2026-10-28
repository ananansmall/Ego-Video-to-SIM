"""
analyze_object_contact.py - 手-物体接触分析 (不需要物体 mesh)
=============================================================

思路很简单: 点云里每个点都来自深度图的某个像素, 所以
  1) 把手部 mesh 用 K 投影到图像 → 得到"手的轮廓"掩码
  2) 落在轮廓内的点云点 = 手自身的表面点; 轮廓外的 = 场景(物体/背景)
  3) 对轮廓外、但 3D 上离手很近 (<10cm) 的点, 算它到手 mesh 的距离:
       < 1cm  → 正在接触
       < 0    → 穿模 (在手内部)
  4) 统计接触点数量/距离分布, 并渲染成 GIF (物体点按到手的距离着色)

用法:
  python tools/analyze_object_contact.py --scene-dir output_v3/egofound_verify/0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer_html as MV

SIDE_COLORS_BGR = [(0, 165, 255), (255, 128, 0)]
NEAR_CM = 10.0     # 3D 上"靠近手"的判定 (厘米, 米制)
CONTACT_CM = 1.0   # 接触判定 (厘米, 米制)


def hand_silhouette(V, faces, K, H, W):
    """手部 mesh 投影到图像 → 布尔轮廓掩码 (H,W)."""
    z = V[:, 2]
    ok = np.isfinite(V).all(axis=1) & (z > 1e-4)
    mask = np.zeros((H, W), np.uint8)
    if ok.sum() < 10:
        return mask
    vv = V[ok]
    uv = np.stack([K[0, 0] * vv[:, 0] / vv[:, 2] + K[0, 2],
                   K[1, 1] * vv[:, 1] / vv[:, 2] + K[1, 2]], 1)
    remap = -np.ones(len(V), int)
    remap[np.nonzero(ok)[0]] = np.arange(int(ok.sum()))
    tris = []
    for a, b, c in faces:
        if ok[a] and ok[b] and ok[c]:
            tris.append(np.stack([uv[remap[a]], uv[remap[b]], uv[remap[c]]], 0))
    if not tris:
        return mask
    cv2.fillPoly(mask, [t.astype(np.int32) for t in tris], 1, cv2.LINE_AA)
    # 轻微膨胀, 覆盖手部边缘的采样误差
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1).astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--ego-root", type=Path, default=MV.DEFAULT_EGO_ROOT)
    ap.add_argument("--pix-step", type=int, default=2)
    ap.add_argument("--size", type=int, default=620)
    ap.add_argument("--yaw-deg", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=10.0)
    args = ap.parse_args()

    import torch
    from PIL import Image

    D = args.scene_dir
    out_dir = D / "contact"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(D / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    data = np.load(D / "local_vggt" / "local_pred.npz", mmap_mode="r")
    depths = np.asarray(data["depths"]); colors = np.asarray(data["colors"])
    K = np.asarray(data["intrinsic"])
    m = json.loads((D / "verify" / "metrics.json").read_text())
    g = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter") or \
        (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
    # 米制 <-> raw 换算: 1 米 = 1/g raw? 手部 metric*g = raw, 所以 raw->meter = /g... 
    # 注意: g = local_raw / ego_metric, 且 ego 的 metric 就是米, 故 raw -> 米 = /g 不成立;
    # 正确: 1 米(ego metric) = g raw  →  1 cm = 0.01*g raw
    CM2RAW = 0.01 * g
    V_all = MV.tnp(payload["hand"]["vertex_xyz_metric"])
    hv = MV.tnp(payload["hand"]["valid"]).astype(bool)
    CP = MV.tnp(payload["hand"]["vertex_contact_probability"])
    faces = MV.load_marker_faces(args.ego_root)
    N, H, W = depths.shape
    print(f"[CONTACT] N={N} g={g:.4f} → 1cm = {CM2RAW:.4f} raw")

    # 精修平移量（若存在）
    tpath = D / "refine" / "refined_hand.npz"
    t_ref = np.load(tpath)["t"] if tpath.exists() else None
    tidx = np.load(tpath)["frames"] if tpath.exists() else None
    if t_ref is not None:
        tm = {int(i): t_ref[k] for k, i in enumerate(tidx)}
        print(f"[CONTACT] 使用精修平移 ({len(tm)} 帧)")
    else:
        tm = {}

    stats, gif = [], []
    for i in range(N):
        d = depths[i]
        valid = np.isfinite(d) & (d > 1e-6)
        s = np.zeros_like(valid); s[::args.pix_step, ::args.pix_step] = True; s &= valid
        ys, xs = np.nonzero(s)
        if len(xs) < 200:
            continue
        z = d[ys, xs]
        P = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                      (ys - K[1, 2]) / K[1, 1] * z, z], 1)
        Cb = colors[i][ys, xs]

        hands = []
        for sd in range(2):
            if not hv[i, sd]:
                continue
            v = V_all[i, sd] * g
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            v = v + tm.get(i, np.zeros(3))            # 应用精修平移
            remap = -np.ones(len(v), int)
            remap[np.nonzero(ok)[0]] = np.arange(int(ok.sum()))
            kept = [(remap[a], remap[b], c) for a, b, c in faces
                    if ok[a] and ok[b] and ok[c]]
            if len(kept) < 20:
                continue
            hands.append({"sd": sd, "V": v, "cp": CP[i, sd],
                          "mesh": trimesh.Trimesh(vertices=v[ok].astype(np.float64),
                                                  faces=np.array(kept), process=False)})
        if not hands:
            continue

        # ---- 轮廓: 手自身的点 vs 场景点 ----
        sil = np.zeros((H, W), bool)
        for h in hands:
            sil |= hand_silhouette(h["V"], faces, K, H, W)
        in_hand = sil[ys, xs]
        obj_idx = np.nonzero(~in_hand)[0]
        if len(obj_idx) < 50:
            continue

        # ---- 靠近手的场景点 → 到手 mesh 的距离 ----
        near_dist, near_mask = [], []
        for h in hands:
            tree = cKDTree(h["mesh"].vertices)
            dist, _ = tree.query(P[obj_idx])
            near = dist < NEAR_CM * CM2RAW            # 10cm (raw 单位)
            near_dist.append(dist); near_mask.append(near)
        if not near_dist:
            continue
        # 取到任一手的最小距离
        dmin = np.min(np.stack(near_dist, 0), axis=0)
        near_any = np.max(np.stack(near_mask, 0), axis=0)
        obj_near = obj_idx[near_any]
        d_near = dmin[near_any]
        if len(obj_near) < 10:
            continue
        contact_n = int((d_near < CONTACT_CM * CM2RAW).sum())   # <1cm
        stats.append({
            "frame": i,
            "object_points_near_hand": int(len(obj_near)),
            "contact_points_cm1": contact_n,
            "median_dist_cm": float(np.median(d_near) / CM2RAW),
            "min_dist_cm": float(d_near.min() / CM2RAW),
            "hand_contact_prob_mean": float(np.mean([h["cp"].mean() for h in hands])),
        })

        # ---- 渲染: 场景点按到手的距离着色 (红=接触) ----
        th = np.deg2rad(args.yaw_deg); cy, sy = np.cos(th), np.sin(th)
        def proj(Q):
            return cy * Q[:, 0] + sy * Q[:, 2], Q[:, 1], -sy * Q[:, 0] + cy * Q[:, 2]
        S = args.size
        allp = np.vstack([P] + [h["V"] for h in hands])
        x, y, _ = proj(allp)
        lo = np.array([np.percentile(x, 2), np.percentile(y, 2)])
        hi = np.array([np.percentile(x, 98), np.percentile(y, 98)])
        span = float(max(hi - lo)) * 1.05; ctr = (hi + lo) / 2
        img = np.zeros((S, S, 3), np.uint8)
        px_ = lambda X: ((X - ctr[0]) / span * (S - 20) + S / 2).astype(np.int32)
        py_ = lambda Y: ((Y - ctr[1]) / span * (S - 20) + S / 2).astype(np.int32)
        px, py, dep = proj(P)
        px, py = px_(px), py_(py)
        okp = (px >= 0) & (px < S - 1) & (py >= 0) & (py < S - 1)
        order = np.argsort(-dep[okp])
        base = 0.55 * Cb[okp][order]                      # 场景点压暗
        img[py[okp][order], px[okp][order]] = base.astype(np.uint8)
        # 接触的物体点: 红→黄按距离
        dcm = np.full(len(P), 999.0)
        dcm[obj_near] = d_near / CM2RAW          # 到手的距离 (厘米)
        hot = np.nonzero((dcm < 3.0) & okp)[0]
        if len(hot):
            t = np.clip(dcm[hot] / 3.0, 0, 1)
            hotcol = np.stack([(255 * (1 - t)).astype(np.uint8),
                               (200 * (1 - t * 0.7)).astype(np.uint8),
                               np.full(len(hot), 40, np.uint8)], 1)   # BGR: 近=红
            for k, j in enumerate(hot):
                cv2.circle(img, (int(px[j]), int(py[j])), 2,
                           [int(c) for c in hotcol[k]], -1, cv2.LINE_AA)
        # 手部
        light = np.array([0.35, -0.55, -0.76]); light /= np.linalg.norm(light)
        for h in hands:
            hx, hy, hd = proj(h["V"])
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
                tris.append((float(hd[list(f)].mean()), tri,
                             [min(255, int(c * sh)) for c in SIDE_COLORS_BGR[h["sd"]]]))
            for _, tri, col in sorted(tris, key=lambda t: -t[0]):
                cv2.fillPoly(img, [tri.astype(np.int32)], col, cv2.LINE_AA)
        cv2.putText(img, f"frame {i} | object points near hand={len(obj_near)} | "
                         f"contact(<1cm)={contact_n} | median={np.median(d_near)/CM2RAW:.1f}cm",
                    (8, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1, cv2.LINE_AA)
        gif.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                   .convert("P", palette=Image.ADAPTIVE))

    if stats:
        med = [s["median_dist_cm"] for s in stats]
        con = [s["contact_points_cm1"] for s in stats]
        print(f"\n[CONTACT] 帧数={len(stats)}")
        print(f"[CONTACT] 手附近物体点: 中位 {int(np.median([s['object_points_near_hand'] for s in stats]))}")
        print(f"[CONTACT] 接触点(<1cm):  中位 {int(np.median(con))}, 最多 {max(con)}")
        print(f"[CONTACT] 手-物中位间距: {np.median(med):.2f} cm (最近帧 {min(med):.2f} cm)")
        (out_dir / "contact_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    if gif:
        gp = out_dir / "object_contact.gif"
        gif[0].save(gp, save_all=True, append_images=gif[1:],
                    duration=int(1000 / args.fps), loop=0, optimize=True)
        print(f"[CONTACT] {gp}  ({len(gif)} 帧, {gp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
