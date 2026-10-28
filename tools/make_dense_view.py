"""
make_dense_view.py - 稠密点云 GLB + 双面板视频
=============================================

解决两个问题:
  1) 之前的点云每 3 像素才取 1 个 → 稀疏、有横条纹, 场景看不清
     → 这里用 ::2 或 ::1 采样 (每帧 6~26 万点), 视觉上接近实心
  2) 单视角斜视图难判断"手在动、场景是什么"
     → 双面板视频: 左=相机视角(能认出场景) 右=3D 斜视角(能看出手的立体位置)

产物:
  dense/scene_and_hands_all.glb   合并点云 + 每一帧的手 (节点名 hand_f000..)
  dense/perframe_dual.mp4         左:相机视角  右:3D斜视  (逐帧)
  dense/perframe_dual.gif         同上 (小体积预览)

用法:
  python tools/make_dense_view.py --scene-dir output_v3/egofound_verify/weixin_195833_224_h264
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

SIDE_COLORS_BGR = [(0, 165, 255), (255, 128, 0)]   # 橙=side0, 蓝=side1
SIDE_COLORS_RGB = [(255, 165, 0), (0, 128, 255)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--ego-root", type=Path, default=MV.DEFAULT_EGO_ROOT)
    ap.add_argument("--pix-step", type=int, default=2, help="点云像素采样步长 (1=全像素)")
    ap.add_argument("--hand-every", type=int, default=2, help="每隔几帧导出一只手")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--yaw-deg", type=float, default=32.0)
    ap.add_argument("--panel-h", type=int, default=720)
    ap.add_argument("--max-points", type=int, default=700_000)
    ap.add_argument("--no-glb", action="store_true")
    ap.add_argument("--contact", action="store_true",
                    help="手部按 EgoFound3R 的接触概率上色 (红=接触), 并显示接触距离")
    ap.add_argument("--suffix", default="", help="输出文件名后缀")
    args = ap.parse_args()

    import torch
    from PIL import Image

    scene_dir = args.scene_dir
    out_dir = scene_dir / "dense"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(scene_dir / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    data = np.load(scene_dir / "local_vggt" / "local_pred.npz", mmap_mode="r")
    depths = np.asarray(data["depths"]); colors = np.asarray(data["colors"])
    wp = np.asarray(data["world_points"]); extr = np.asarray(data["extrinsics"])
    K = np.asarray(data["intrinsic"])
    m = json.loads((scene_dir / "verify" / "metrics.json").read_text())
    g = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter") or \
        (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
    faces = MV.load_marker_faces(args.ego_root)
    verts = MV.tnp(payload["hand"]["vertex_xyz_metric"])
    hvalid = MV.tnp(payload["hand"]["valid"]).astype(bool)
    # EgoFound3R 自带的接触信息 (逐顶点, 距离单位为米)
    cont_prob = MV.tnp(payload["hand"]["vertex_contact_probability"]) if args.contact else None
    cont_dist = MV.tnp(payload["hand"]["vertex_contact_distance_metric"]) if args.contact else None
    N, H, W = depths.shape
    print(f"[DENSE] N={N} H={H} W={W} g={g:.4f} pix_step={args.pix_step}")

    # ---------------- 1) 稠密 GLB ----------------
    if not args.no_glb:
        print("[GLB] 合并稠密点云 + 每帧手部 ...")
        pts, cols = [], []
        for i in range(N):
            d = depths[i]
            v = np.isfinite(d) & (d > 1e-6)
            s = np.zeros_like(v); s[::args.pix_step, ::args.pix_step] = True; s &= v
            ys, xs = np.nonzero(s)
            if len(xs) == 0:
                continue
            p = wp[i][ys, xs]
            ok = np.isfinite(p).all(axis=1)
            pts.append(p[ok].astype(np.float32)); cols.append(colors[i][ys, xs][ok].astype(np.uint8))
        P = np.concatenate(pts); C = np.concatenate(cols)
        if len(P) > args.max_points:
            idx = np.random.choice(len(P), args.max_points, replace=False)
            P, C = P[idx], C[idx]
        cloud = trimesh.PointCloud(P, colors=C)
        print(f"[GLB] 点云 {len(P)} 点, bbox 对角 {np.linalg.norm(np.ptp(P, axis=0)):.3f}")

        geoms = [cloud]
        for i in range(0, N, args.hand_every):
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
                if len(kept) < 20:
                    continue
                R, t = extr[i, :3, :3], extr[i, :3, 3]
                vw = (R.T @ (v - t).T).T
                color = np.tile(np.array(SIDE_COLORS_RGB[sd], np.uint8), (int(ok.sum()), 1))
                geoms.append(trimesh.Trimesh(vertices=vw[ok].astype(np.float64),
                                             faces=np.array(kept), vertex_colors=color,
                                             process=False))
        glb = out_dir / "scene_and_hands_all.glb"
        trimesh.Scene(geoms).export(glb)
        print(f"[GLB] {glb}  ({glb.stat().st_size/1e6:.1f} MB, {len(geoms)-1} 只手, 每只一个图层)")

    # ---------------- 2) 双面板逐帧视频 ----------------
    th = np.deg2rad(args.yaw_deg); cy, sy = np.cos(th), np.sin(th)
    BW = int(args.panel_h * W / H)          # 左面板宽 (保持画幅)
    size = args.panel_h
    light = np.array([0.35, -0.55, -0.76]); light /= np.linalg.norm(light)

    def proj(Q):
        x = cy * Q[:, 0] + sy * Q[:, 2]
        dep = -sy * Q[:, 0] + cy * Q[:, 2]
        return x, Q[:, 1], dep

    # 固定 3D 视口 (所有帧一致)
    allx, ally = [], []
    for i in range(0, N, 4):
        d = depths[i]; v = np.isfinite(d) & (d > 1e-6)
        s = np.zeros_like(v); s[::4, ::4] = True; s &= v
        ys, xs = np.nonzero(s)
        if len(xs) == 0:
            continue
        z = d[ys, xs]
        Q = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
        x, y, _ = proj(Q)
        allx.append(x); ally.append(y)
    allx = np.concatenate(allx); ally = np.concatenate(ally)
    lo = np.array([np.percentile(allx, 2), np.percentile(ally, 2)])
    hi = np.array([np.percentile(allx, 98), np.percentile(ally, 98)])
    span = float(max(hi - lo)) * 1.05
    ctr = (hi + lo) / 2

    writer = None
    gif_frames = []
    for i in range(N):
        d = depths[i]
        valid = np.isfinite(d) & (d > 1e-6)
        s = np.zeros_like(valid); s[::args.pix_step, ::args.pix_step] = True; s &= valid
        ys, xs = np.nonzero(s)
        if len(xs) == 0:
            continue
        z = d[ys, xs]
        P = np.stack([(xs - K[0, 2]) / K[0, 0] * z, (ys - K[1, 2]) / K[1, 1] * z, z], 1)
        Cb = colors[i][ys, xs][:, ::-1]

        # ---- 左: 相机视角 (源分辨率绘制后 resize, 避免跳行出现的网格线) ----
        ysa, xsa = np.nonzero(valid)
        src = np.zeros((H, W, 3), np.uint8)
        src[ysa, xsa] = colors[i][ysa, xsa][:, ::-1]
        left = cv2.resize(src, (BW, size), interpolation=cv2.INTER_LINEAR)
        left = cv2.convertScaleAbs(left, alpha=1.45, beta=22)

        # ---- 右: 3D 斜视 ----
        right = np.zeros((size, size, 3), np.uint8)
        x, y, dep = proj(P)
        px = ((x - ctr[0]) / span * (size - 24) + size / 2).astype(np.int32)
        py = ((y - ctr[1]) / span * (size - 24) + size / 2).astype(np.int32)
        ok = (px >= 0) & (px < size - 1) & (py >= 0) & (py < size - 1)
        order = np.argsort(-dep[ok])
        po, qo = px[ok][order], py[ok][order]
        # 右面板用深度着色, 让 3D 结构更易读 (纯 RGB 在浅色场景上几乎看不出形貌)
        zz = dep[ok][order]
        zn = np.clip((zz - np.percentile(zz, 2)) /
                     max(np.percentile(zz, 98) - np.percentile(zz, 2), 1e-9), 0, 1)
        zc = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
        zc = (0.45 * np.array(Cb[ok][order], np.float32) + 0.55 * zc).astype(np.uint8)
        for dy in (0, 1):
            right[np.clip(qo + dy, 0, size - 1), po] = zc

        # 手部 (逐面光照)
        tris = []
        for sd in range(2):
            if not hvalid[i, sd]:
                continue
            v = verts[i, sd] * g
            okv = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if okv.sum() < 20:
                continue
            vv = v[okv]
            remap = -np.ones(len(v), int)
            remap[np.nonzero(okv)[0]] = np.arange(int(okv.sum()))
            kept = [(remap[a], remap[b], c) for a, b, c in faces if okv[a] and okv[b] and okv[c]]
            if len(kept) < 20:
                continue
            hx, hy, hd = proj(vv)
            hpx = (hx - ctr[0]) / span * (size - 24) + size / 2
            hpy = (hy - ctr[1]) / span * (size - 24) + size / 2
            V3 = np.stack([hx, hy, hd], 1)
            vcp = cont_prob[i, sd][okv] if cont_prob is not None else None
            for f in kept:
                tri = np.stack([hpx[list(f)], hpy[list(f)]], 1)
                if not ((tri[:, 0] > -size).all() and (tri[:, 0] < 2 * size).all()
                        and (tri[:, 1] > -size).all() and (tri[:, 1] < 2 * size).all()):
                    continue
                p0, p1, p2 = V3[f[0]], V3[f[1]], V3[f[2]]
                n = np.cross(p1 - p0, p2 - p0); ln = np.linalg.norm(n)
                sh = 0.45 + 0.55 * abs(float(n @ light) / ln) if ln > 1e-12 else 0.8
                base = np.array(SIDE_COLORS_BGR[sd], dtype=float)
                if vcp is not None:
                    # 接触概率越高越红 (红=正在接触物体)
                    pc = float(np.clip(vcp[list(f)].mean(), 0, 1))
                    base = base * (1 - pc) + np.array([0, 0, 255], dtype=float) * pc
                col = [min(255, int(c * sh)) for c in base]
                tris.append((float(hd[list(f)].mean()), tri, col))
            # 接触顶点高亮点
            if vcp is not None:
                hot = np.nonzero(vcp > 0.5)[0]
                for vi in hot:
                    x, y = int(hpx[vi]), int(hpy[vi])
                    if 0 <= x < size and 0 <= y < size:
                        cv2.circle(right, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
        for _, tri, col in sorted(tris, key=lambda t: -t[0]):
            cv2.fillPoly(right, [tri.astype(np.int32)], col, cv2.LINE_AA)

        cv2.putText(left, f"camera view  (frame {i})", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(right, f"3D view  (yaw {args.yaw_deg:.0f}deg)", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        frame = np.hstack([left, right])

        if writer is None:
            vw_path = out_dir / "perframe_dual.mp4"
            writer = cv2.VideoWriter(str(vw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if i % 2 == 0:
            gif_frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                              .convert("P", palette=Image.ADAPTIVE))
    if writer is not None:
        writer.release()
        print(f"[MP4] {out_dir/'perframe_dual.mp4'}  ({N} 帧 @ {args.fps}fps)")

    if gif_frames:
        gp = out_dir / "perframe_dual.gif"
        gif_frames[0].save(gp, save_all=True, append_images=gif_frames[1:],
                           duration=int(2000 / args.fps), loop=0, optimize=True)
        print(f"[GIF] {gp}  ({len(gif_frames)} 帧, {gp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
