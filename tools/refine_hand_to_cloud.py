"""
refine_hand_to_cloud.py - 逐帧手部→点云配准精修 (ICP + 时间平滑)
================================================================

问题: EgoFound3R 的相机与本地 VGGT 相机逐帧存在 3~6 度旋转差 (实测),
      导致手部在点云里"飘"。尺度 (g) 已经标定好, 需要修的是逐帧刚体偏差。

做法: 每帧求一个小 SE(3) (δR, δt), 把 ego 相机系的手部顶点映射到
      本地相机系, 使手部顶点贴合本地点云 (点云里本来就含手自身的表面点):
        1) 初值: δR = R_local @ R_ego^T (相机旋转差), δt = 0
        2) ICP 迭代 4 次: 最近点 + 截尾 80% + point-to-point Procrustes
        3) 时间平滑: 对 δR/δt 做滑窗中值+均值平滑, 再作为初值做一次轻量 ICP
      尺度不优化 (g 已标定), 避免把手缩放到噪声上。

输出: <scene>/refine/
  refined_hand.npz          精修后的手部顶点 (ego 相机系 + 需要应用的 (R,t))
  refine_metrics.json       逐帧 before/after 距离统计
  refine_before_after.gif   左右对照 (左=原始 右=精修)

用法:
  python tools/refine_hand_to_cloud.py --scene-dir output_v3/egofound_verify/weixin_195833_224_h264
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer_html as MV

SIDE_COLORS_BGR = [(0, 165, 255), (255, 128, 0)]


def rigid_from_corr(src, dst):
    """point-to-point Procrustes: 求 R,t 使 R·src + t ≈ dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def frame_cloud(depth, K, step=2):
    v = np.isfinite(depth) & (depth > 1e-6)
    s = np.zeros_like(v); s[::step, ::step] = True; s &= v
    ys, xs = np.nonzero(s)
    z = depth[ys, xs]
    return np.stack([(xs - K[0, 2]) / K[0, 0] * z,
                     (ys - K[1, 2]) / K[1, 1] * z, z], 1), ys, xs


def refine_step(V, P, tree, iters=6, trim=70, damp=0.5, cap=0.05):
    """只解平移 (195 顶点手近似平面, 解旋转病态会发散).
    迭代: 最近点 -> 截尾 -> 中位位移 * 阻尼, 每步限幅 cap (raw 单位)."""
    t = np.zeros(3)
    for _ in range(iters):
        Vt = V + t
        d, idx = tree.query(Vt)
        thr = np.percentile(d, trim)
        keep = d <= thr
        if keep.sum() < 20:
            break
        res = P[idx[keep]] - Vt[keep]
        dt = np.median(res, axis=0) * damp
        n = float(np.linalg.norm(dt))
        if n > cap:
            dt *= cap / max(n, 1e-9)
        t += dt
        if n < 2e-4:
            break
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--ego-root", type=Path, default=MV.DEFAULT_EGO_ROOT)
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--pix-step", type=int, default=2)
    ap.add_argument("--size", type=int, default=520)
    ap.add_argument("--yaw-deg", type=float, default=30.0)
    args = ap.parse_args()

    import torch
    from PIL import Image

    D = args.scene_dir
    out_dir = D / "refine"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(D / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    data = np.load(D / "local_vggt" / "local_pred.npz", mmap_mode="r")
    depths = np.asarray(data["depths"]); colors = np.asarray(data["colors"])
    K = np.asarray(data["intrinsic"]); extr = np.asarray(data["extrinsics"])
    m = json.loads((D / "verify" / "metrics.json").read_text())
    g = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter") or \
        (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
    V = MV.tnp(payload["hand"]["vertex_xyz_metric"])
    hv = MV.tnp(payload["hand"]["valid"]).astype(bool)
    CP = MV.tnp(payload["hand"]["vertex_contact_probability"])
    ego_R = MV.tnp(payload["camera_pose_metric_full"])[:, :3, :3]
    faces = MV.load_marker_faces(args.ego_root)
    N = depths.shape[0]
    print(f"[REFINE] N={N}, g={g:.4f}, 平滑窗口={args.smooth_window}")

    # ---------- 1) 逐帧准备 ----------
    frames = []
    for i in range(N):
        P, ys, xs = frame_cloud(depths[i], K, args.pix_step)
        if len(P) < 200:
            continue
        s = None
        for sd in range(2):
            if hv[i, sd]:
                s = sd
                break
        if s is None:
            frames.append({"i": i, "P": P, "ys": ys, "xs": xs, "V": None})
            continue
        v = V[i, s] * g
        ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
        frames.append({"i": i, "P": P, "ys": ys, "xs": xs, "V": v[ok],
                        "cp": CP[i, s][ok], "sd": s, "ok": ok})
    print(f"[REFINE] 有效帧 {sum(1 for f in frames if f['V'] is not None)}/{len(frames)}")

    # ---------- 2) 逐帧两阶段平移精修 + 择优 (保证不劣化) ----------
    def refine_step_w(V, P, tree, w=None, iters=6, trim=70, damp=0.5, cap=0.05):
        """带权平移精修 (w: 顶点权重, 例如 1 + k*接触概率)."""
        t = np.zeros(3)
        for _ in range(iters):
            Vt = V + t
            d, idx = tree.query(Vt)
            thr = np.percentile(d, trim)
            keep = d <= thr
            if keep.sum() < 20:
                break
            res = P[idx[keep]] - Vt[keep]
            ww = None if w is None else w[keep]
            dt = np.median(res, axis=0) if ww is None else \
                (res * ww[:, None]).sum(0) / max(ww.sum(), 1e-9)
            dt = dt * damp
            n = float(np.linalg.norm(dt))
            if n > cap:
                dt *= cap / max(n, 1e-9)
            t += dt
            if n < 2e-4:
                break
        return t

    def score(V, P, tree, cw):
        d, _ = tree.query(V)
        hl = max(np.linalg.norm(np.ptp(V, axis=0)), 1e-9)
        if cw is None:
            return float(np.median(d) / hl)
        w = 1.0 + 8.0 * cw
        return float((d * w).sum() / max(w.sum(), 1e-9) / hl)

    ts, chosen, base_sc, best_sc = [], [], [], []
    for f in frames:
        if f["V"] is None:
            ts.append(np.zeros(3)); chosen.append("none")
            base_sc.append(None); best_sc.append(None); continue
        tree = cKDTree(f["P"])
        cw = f["cp"]
        s0 = score(f["V"], f["P"], tree, cw)
        # 阶段1: 全顶点
        t1 = refine_step_w(f["V"], f["P"], tree, None, cap=0.05)
        s1 = score(f["V"] + t1, f["P"], tree, cw)
        # 阶段2: 接触加权 (从阶段1出发, 小步)
        t2 = refine_step_w(f["V"] + t1, f["P"], tree, cw, cap=0.02, iters=5, damp=0.35)
        t2 = t1 + t2
        s2 = score(f["V"] + t2, f["P"], tree, cw)
        cand = [(np.zeros(3), s0, "none"), (t1, s1, "global"), (t2, s2, "contact")]
        cand.sort(key=lambda c: (c[1], np.linalg.norm(c[0])))
        ts.append(cand[0][0]); chosen.append(cand[0][2])
        base_sc.append(s0); best_sc.append(cand[0][1])
    ts = np.asarray(ts, dtype=float)
    print("[REFINE] 采纳方案: global=%d, contact=%d, 保持原样=%d" %
          (chosen.count("global"), chosen.count("contact"), chosen.count("none")))
    print("[REFINE] 平移模长: 中位 %.3f, 最大 %.3f (raw)" %
          (np.median(np.linalg.norm(ts, axis=1)), np.linalg.norm(ts, axis=1).max()))

    # ---------- 3) 时间平滑 ----------
    w = args.smooth_window
    def smooth(a):
        out = a.copy()
        for i in range(len(a)):
            lo, hi = max(0, i - w), min(len(a), i + w + 1)
            seg = a[lo:hi]
            out[i] = 0.5 * np.median(seg, axis=0) + 0.5 * seg.mean(axis=0)
        return out
    ts_s = smooth(ts)

    # ---------- 4) 平滑后逐帧择优 (平滑值 vs 未平滑值) ----------
    ts_f = []
    for k, f in enumerate(frames):
        if f["V"] is None:
            ts_f.append(ts_s[k]); continue
        tree = cKDTree(f["P"])
        cw = f["cp"]
        a, b = ts_s[k], ts[k]
        sa = score(f["V"] + a, f["P"], tree, cw)
        sb = score(f["V"] + b, f["P"], tree, cw)
        ts_f.append(a if sa <= sb else b)
    ts_f = np.asarray(ts_f, dtype=float)
    Rs_f = np.tile(np.eye(3), (len(frames), 1, 1))

    # ---------- 5) 指标 ----------
    def ratio(Vf, P, tree=None):
        tree = tree or cKDTree(P)
        d, _ = tree.query(Vf)
        hl = np.linalg.norm(np.ptp(Vf, axis=0))
        return float(np.median(d) / max(hl, 1e-9))
    before, after, frames_id = [], [], []
    for k, f in enumerate(frames):
        if f["V"] is None:
            continue
        tree = cKDTree(f["P"])
        before.append(ratio(f["V"], f["P"], tree))
        after.append(ratio(f["V"] @ Rs_f[k].T + ts_f[k], f["P"], tree))
        frames_id.append(f["i"])
    before, after = np.array(before), np.array(after)
    print("\n[REFINE] 手长归一化中位距离:  before=%.1f%%  after=%.1f%%" %
          (np.median(before) * 100, np.median(after) * 100))
    print("[REFINE] 超 25%% 的帧数:        before=%d/%d  after=%d/%d" %
          (int((before > 0.25).sum()), len(before), int((after > 0.25).sum()), len(after)))
    print("[REFINE] 最差帧:                before=%.0f%%  after=%.0f%%" %
          (before.max() * 100, after.max() * 100))

    np.savez(out_dir / "refined_hand.npz",
             frames=np.array(frames_id), R=Rs_f, t=ts_f,
             before=before, after=after, g=g)
    (out_dir / "refine_metrics.json").write_text(json.dumps({
        "g": float(g), "n_frames": len(frames_id),
        "median_ratio_before": float(np.median(before)),
        "median_ratio_after": float(np.median(after)),
        "frames_over_25pct_before": int((before > 0.25).sum()),
        "frames_over_25pct_after": int((after > 0.25).sum()),
        "worst_before": float(before.max()), "worst_after": float(after.max()),
        "per_frame": [{"frame": int(i), "before": float(b), "after": float(a)}
                      for i, b, a in zip(frames_id, before, after)],
    }, indent=2, ensure_ascii=False))

    # ---------- 6) 左右对照 GIF ----------
    th = np.deg2rad(args.yaw_deg); cy, sy = np.cos(th), np.sin(th)
    def proj(Q):
        return cy * Q[:, 0] + sy * Q[:, 2], Q[:, 1], -sy * Q[:, 0] + cy * Q[:, 2]
    allx, ally = [], []
    for k, f in enumerate(frames):
        x, y, _ = proj(f["P"]); allx.append(x); ally.append(y)
        if f["V"] is not None:
            for Vf in (f["V"], f["V"] @ Rs_f[k].T + ts_f[k]):
                x, y, _ = proj(Vf); allx.append(x); ally.append(y)
    allx = np.concatenate(allx); ally = np.concatenate(ally)
    lo = np.array([np.percentile(allx, 2), np.percentile(ally, 2)])
    hi = np.array([np.percentile(allx, 98), np.percentile(ally, 98)])
    span = float(max(hi - lo)) * 1.05; ctr = (hi + lo) / 2
    light = np.array([0.35, -0.55, -0.76]); light /= np.linalg.norm(light)
    S = args.size

    def draw(P, Cb, Vf, sd, label):
        img = np.zeros((S, S, 3), np.uint8)
        x, y, dep = proj(P)
        px = ((x - ctr[0]) / span * (S - 20) + S / 2).astype(np.int32)
        py = ((y - ctr[1]) / span * (S - 20) + S / 2).astype(np.int32)
        ok = (px >= 0) & (px < S - 1) & (py >= 0) & (py < S - 1)
        order = np.argsort(-dep[ok])
        img[py[ok][order], px[ok][order]] = Cb[ok][order]
        if Vf is not None:
            hx, hy, hd = proj(Vf)
            hpx = (hx - ctr[0]) / span * (S - 20) + S / 2
            hpy = (hy - ctr[1]) / span * (S - 20) + S / 2
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
                             [min(255, int(c * sh)) for c in SIDE_COLORS_BGR[sd]]))
            for _, tri, col in sorted(tris, key=lambda t: -t[0]):
                cv2.fillPoly(img, [tri.astype(np.int32)], col, cv2.LINE_AA)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return img

    gif = []
    for k, f in enumerate(frames):
        if f["V"] is None:
            continue
        Cb = colors[f["i"]][f["ys"], f["xs"]][:, ::-1]
        b = before[frames_id.index(f["i"])] if f["i"] in frames_id else 0
        a = after[frames_id.index(f["i"])] if f["i"] in frames_id else 0
        left = draw(f["P"], Cb, f["V"], f["sd"], f"frame {f['i']}  BEFORE  {b*100:.0f}%")
        right = draw(f["P"], Cb, f["V"] @ Rs_f[k].T + ts_f[k], f["sd"],
                     f"REFINED  {a*100:.0f}%")
        cmp = np.hstack([left, right])
        gif.append(Image.fromarray(cv2.cvtColor(cmp, cv2.COLOR_BGR2RGB))
                   .convert("P", palette=Image.ADAPTIVE))
    gp = out_dir / "refine_before_after.gif"
    gif[0].save(gp, save_all=True, append_images=gif[1:], duration=100, loop=0, optimize=True)
    print(f"[REFINE] {gp}  ({len(gif)} 帧, {gp.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
