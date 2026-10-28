"""
make_perframe_gif.py - 逐帧点云 + 手部 GIF (判断 3D 对齐)
========================================================

每一帧: 只用该帧自己的深度反投影点云 (不叠加其它帧, 无重影)
      + 该帧的手部 mesh (x g 换算到同一 gauge)
固定斜视角 (yaw 35°, 绕竖直轴旋转), 正交投影 + 画家算法 + 手部逐面光照。

这样移动的手/物体会在点云里同步移动, 能直接看出手是否贴在该帧点云表面上。

用法:
  python tools/make_perframe_gif.py --scene-dir output_v3/egofound_verify/weixin_195833_224_h264
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer_html as MV

SIDE_COLORS_BGR = [(0, 165, 255), (255, 128, 0)]   # 橙=side0 左手, 蓝=side1 右手


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--ego-root", type=Path, default=MV.DEFAULT_EGO_ROOT)
    ap.add_argument("--step", type=int, default=2, help="每隔几帧取一帧")
    ap.add_argument("--size", type=int, default=680)
    ap.add_argument("--yaw-deg", type=float, default=35.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--pix-step", type=int, default=3, help="深度图像素采样步长")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--mp4", action="store_true", help="同时输出 MP4 视频 (30fps 全帧)")
    args = ap.parse_args()

    import torch
    from PIL import Image

    scene_dir = args.scene_dir
    payload = torch.load(scene_dir / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    data = np.load(scene_dir / "local_vggt" / "local_pred.npz", mmap_mode="r")
    depths = np.asarray(data["depths"]); colors = np.asarray(data["colors"])
    K = np.asarray(data["intrinsic"])
    extr = np.asarray(data["extrinsics"])
    m = json.loads((scene_dir / "verify" / "metrics.json").read_text())
    g = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter") or \
        (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
    print(f"[GIF] g={g:.4f} (手部 metric x g -> 本地 raw gauge)")

    faces = MV.load_marker_faces(args.ego_root)
    verts = MV.tnp(payload["hand"]["vertex_xyz_metric"])
    hvalid = MV.tnp(payload["hand"]["valid"]).astype(bool)
    N = depths.shape[0]
    sel_frames = list(range(0, N, args.step))

    # ---- 1) 逐帧准备点云(相机系 raw) 与手部(相机系, x g) ----
    per = []
    for i in sel_frames:
        d = depths[i]
        valid = np.isfinite(d) & (d > 1e-6)
        s = np.zeros_like(valid); s[::args.pix_step, ::args.pix_step] = True; s &= valid
        ys, xs = np.nonzero(s)
        if len(xs) == 0:
            continue
        z = d[ys, xs]
        P = np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                      (ys - K[1, 2]) / K[1, 1] * z, z], axis=1)
        C = colors[i][ys, xs]
        hands = []
        for sd in range(2):
            if not hvalid[i, sd]:
                continue
            v = verts[i, sd] * g
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            remap = -np.ones(len(v), int)
            remap[np.nonzero(ok)[0]] = np.arange(int(ok.sum()))
            kept = [(remap[a], remap[b], remap[c]) for a, b, c in faces
                    if ok[a] and ok[b] and ok[c]]
            if len(kept) >= 20:
                hands.append({"V": v[ok], "F": np.array(kept), "side": sd})
        per.append({"i": i, "P": P, "C": C, "H": hands})
    print(f"[GIF] 帧数={len(per)}  平均点数={int(np.mean([len(p['P']) for p in per]))}")

    # 每帧量化: 手部顶点 -> 该帧点云最近点距离 (手长归一化)
    from scipy.spatial import cKDTree
    for f in per:
        best = None
        if len(f["P"]) > 100:
            tree = cKDTree(f["P"])
            for h in f["H"]:
                vv = h["V"]
                ok = np.isfinite(vv).all(axis=1) & (vv[:, 2] > 1e-4)
                if ok.sum() < 20:
                    continue
                dist, _ = tree.query(vv[ok])
                hl = np.linalg.norm(np.ptp(vv[ok], axis=0))
                r = float(np.median(dist) / max(hl, 1e-9))
                if best is None or r < best:
                    best = r
        f["ratio"] = best
    print("[GIF] 平均贴合误差: " + ", ".join(
        f"f{f['i']}={f['ratio']*100:.0f}%" for f in per if f["ratio"] is not None))

    th = np.deg2rad(args.yaw_deg)
    cy, sy = np.cos(th), np.sin(th)

    def proj(Q):
        x = cy * Q[:, 0] + sy * Q[:, 2]
        dep = -sy * Q[:, 0] + cy * Q[:, 2]
        return x, Q[:, 1], dep

    # ---- 2) 固定视口: 覆盖所有帧 ----
    allx, ally = [], []
    for p in per:
        x, y, _ = proj(p["P"])
        allx.append(x); ally.append(y)
        for h in p["H"]:
            hx, hy, _ = proj(h["V"])
            allx.append(hx); ally.append(hy)
    allx = np.concatenate(allx); ally = np.concatenate(ally)
    lo = np.array([np.percentile(allx, 2), np.percentile(ally, 2)])
    hi = np.array([np.percentile(allx, 98), np.percentile(ally, 98)])
    span = float(max(hi - lo)) * 1.06
    ctr = (hi + lo) / 2

    light = np.array([0.35, -0.55, -0.76]); light /= np.linalg.norm(light)
    frames = []
    bgr_frames = []
    for p in per:
        img = np.zeros((args.size, args.size, 3), np.uint8)
        x, y, dep = proj(p["P"])
        px = ((x - ctr[0]) / span * (args.size - 30) + args.size / 2).astype(np.int32)
        py = ((y - ctr[1]) / span * (args.size - 30) + args.size / 2).astype(np.int32)
        ok = (px >= 0) & (px < args.size - 2) & (py >= 0) & (py < args.size - 2)
        order = np.argsort(-dep[ok])
        po, qo = px[ok][order], py[ok][order]
        cols = p["C"][ok][order][:, ::-1]
        for dy in (0, 1):
            for dx in (0, 1):
                img[qo + dy, po + dx] = cols

        # 手部: 逐面光照 + 画家算法
        tris = []
        for h in p["H"]:
            hx, hy, hd = proj(h["V"])
            hpx = ((hx - ctr[0]) / span * (args.size - 30) + args.size / 2)
            hpy = ((hy - ctr[1]) / span * (args.size - 30) + args.size / 2)
            V3 = np.stack([hx, hy, hd], 1)
            for f in h["F"]:
                tri = np.stack([hpx[f], hpy[f]], 1)
                if not ((tri[:, 0] > -args.size).all() and (tri[:, 0] < 2 * args.size).all()
                        and (tri[:, 1] > -args.size).all() and (tri[:, 1] < 2 * args.size).all()):
                    continue
                p0, p1, p2 = V3[f[0]], V3[f[1]], V3[f[2]]
                n = np.cross(p1 - p0, p2 - p0); ln = np.linalg.norm(n)
                sh = 0.45 + 0.55 * abs(float(n @ light) / ln) if ln > 1e-12 else 0.8
                col = [min(255, int(c * sh)) for c in SIDE_COLORS_BGR[h["side"]]]
                tris.append((float(hd[f].mean()), tri, col))
        for _, tri, col in sorted(tris, key=lambda t: -t[0]):
            cv2.fillPoly(img, [tri.astype(np.int32)], col, cv2.LINE_AA)

        cv2.putText(img, f"frame {p['i']} | point cloud = this frame only | hand = this frame",
                    (8, args.size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1, cv2.LINE_AA)
        r = p.get("ratio")
        if r is not None:
            col = (120, 230, 120) if r < 0.10 else ((90, 220, 240) if r < 0.25 else (90, 90, 240))
            cv2.putText(img, f"hand-to-cloud dist = {r*100:.0f}% of hand length",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        bgr_frames.append(img.copy())
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                      .convert("P", palette=Image.ADAPTIVE))

    out = args.out or (scene_dir / "verify" / "perframe_pointcloud.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"[GIF] {out}  ({len(frames)} 帧, {out.stat().st_size/1e6:.1f} MB)")

    if args.mp4:
        mp4 = out.with_suffix(".mp4")
        vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps * args.step, (args.size, args.size))
        for im in bgr_frames:
            vw.write(im)
        vw.release()
        print(f"[MP4] {mp4}  ({mp4.stat().st_size/1e6:.1f} MB, {len(frames)} 帧 @ {args.fps*args.step:.0f}fps)")


if __name__ == "__main__":
    main()
