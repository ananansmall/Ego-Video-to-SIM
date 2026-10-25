"""
make_viewer_gif.py - 生成可内联查看的 GIF (替代 HTML 查看器)
============================================================

用途: HTML 查看器在部分环境加载不便时, 用 GIF 在 Markdown / GitHub / VS Code
预览里直接观看。包含两类:

  1) turntable: 3D 场景点云 + 手部 mesh 绕竖直轴旋转一周 (正交投影, 画家算法)
     -> 相当于 HTML 查看器的自动旋转效果
  2) 已有 2D 渲染序列 -> GIF (overlay2d / render3d / sideview / compare)

用法:
  cd ReplicateAnyScene
  /mnt/.../python tools/make_viewer_gif.py --scene-dir output_v3/egofound_verify/weixin_195833_224_h264
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_viewer_html as MV  # 复用数据准备 (官方几何函数)

DEFAULT_EGO_ROOT = MV.DEFAULT_EGO_ROOT
SIDE_COLORS_BGR = [(0, 165, 255), (255, 128, 0)]   # side0 橙, side1 蓝 (BGR)


def turntable_gif(P, C, hands, out_path, n_frames=36, size=720, fps=14, pitch_deg=18.0):
    """正交投影 turntable: 点云 + 手部绕竖直轴(Y)旋转一周."""
    from PIL import Image

    # 固定视口范围 (所有帧一致)
    lo = np.percentile(P, 1, axis=0)
    hi = np.percentile(P, 99, axis=0)
    for h in hands:
        V = np.asarray(h["vw"])
        lo = np.minimum(lo, V.min(axis=0)); hi = np.maximum(hi, V.max(axis=0))
    ctr = (hi + lo) / 2.0
    span = max(float(np.max(hi - lo)), 1e-6) * 1.08   # 最大边长 (np.ptp 会算成跨轴范围)

    # 预处理手部 (转到统一数组)
    Hs = []
    for h in hands:
        Hs.append({"V": np.asarray(h["vw"]), "F": np.asarray(h["faces"]),
                   "col": SIDE_COLORS_BGR[int(h["side"])], "name": h["name"]})

    pitch = np.deg2rad(pitch_deg)
    frames = []
    for k in range(n_frames):
        yaw = 2 * np.pi * k / n_frames
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        def proj(Q):
            x = cy * Q[:, 0] + sy * Q[:, 2]
            z = -sy * Q[:, 0] + cy * Q[:, 2]
            y = Q[:, 1]
            # 屏幕: 横向=x, 纵向 = cp*y - sp*z (轻微俯视)
            v = cp * y - sp * z
            depth = sp * y + cp * z
            px = ((x - ctr[0]) / span * (size - 40) + size / 2).astype(np.int32)
            py = ((v - ctr[1]) / span * (size - 40) + size / 2).astype(np.int32)
            return px, py, depth

        img = np.zeros((size, size, 3), np.uint8)
        px, py, dep = proj(P)
        ok = (px >= 0) & (px < size - 2) & (py >= 0) & (py < size - 2) & np.isfinite(dep)
        order = np.argsort(-dep[ok])          # 远 -> 近
        po, qo = px[ok][order], py[ok][order]
        cols = C[ok][order][:, ::-1]          # RGB -> BGR
        for dy in (0, 1):                     # 2x2 splat, 否则单像素在 720 画布上几乎看不见
            for dx in (0, 1):
                img[qo + dy, po + dx] = cols

        # 手部 (画家算法: 按面深度排序)
        tris = []
        for H in Hs:
            hpx, hpy, hdep = proj(H["V"])
            for f in H["F"]:
                if not (np.isfinite(hdep[f]).all()):
                    continue
                tri = np.stack([hpx[f], hpy[f]], 1)
                if (tri[:, 0] > -size).all() and (tri[:, 0] < 2 * size).all() and \
                   (tri[:, 1] > -size).all() and (tri[:, 1] < 2 * size).all():
                    tris.append((float(hdep[f].mean()), tri, H["col"]))
        for _, tri, col in sorted(tris, key=lambda t: -t[0]):
            cv2.fillPoly(img, [tri.astype(np.int32)], col, cv2.LINE_AA)

        cv2.putText(img, f"yaw {np.rad2deg(yaw):.0f} deg | {len(P)} pts | {len(Hs)} hands",
                    (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE))

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"[GIF] {out_path}  ({n_frames} 帧, {out_path.stat().st_size/1e6:.1f} MB)")


def seq_gif(paths, out_path, width=400, fps=10, max_frames=30):
    """把已有 PNG 序列压成 GIF (控制体积)."""
    from PIL import Image
    step = max(1, len(paths) // max_frames)
    sel = sorted(paths, key=lambda p: p.name)[::step][:max_frames]
    frames = []
    for p in sel:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (width, int(h * width / w)))
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE))
    if not frames:
        return None
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    print(f"[GIF] {out_path}  ({len(frames)} 帧, {out_path.stat().st_size/1e6:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
def gather_with_world(scene_dir, gauge, ego_root, max_points=150_000, frames=None):
    """取点云 + 手部(世界系顶点+面), 供 turntable 使用."""
    import torch
    import trimesh
    payload = torch.load(scene_dir / "egofound3r" / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    T = payload["hand"]["valid"].shape[0]
    frames = frames or sorted(set([int(np.argmax(payload["hand"]["valid"].numpy().sum(axis=1))),
                                   T // 2, T - 1]))
    if gauge == "ego":
        P, C, _, meta = MV.gather_ego(scene_dir, max_points, frames, ego_root)
        # 手部世界系顶点 (与 gather_ego 一致: 官方 w2c_to_c2w)
        sys.path.insert(0, str(ego_root))
        from egohandmetric_prompt.models.camera_temporal_refiner import w2c_to_c2w
        cam_full = MV.tnp(payload["camera_pose_metric_full"])
        verts = MV.tnp(payload["hand"]["vertex_xyz_metric"])
        valid = MV.tnp(payload["hand"]["valid"]).astype(bool)
    else:
        m = json.loads((scene_dir / "verify" / "metrics.json").read_text())
        g = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter") or \
            (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
        P, C, _, meta = MV.gather_ras(scene_dir, g, max_points, frames, ego_root)
        extr = np.asarray(np.load(scene_dir / "local_vggt" / "local_pred.npz", mmap_mode="r")["extrinsics"])
        cam_full = None
        verts = MV.tnp(payload["hand"]["vertex_xyz_metric"])
        valid = MV.tnp(payload["hand"]["valid"]).astype(bool)
    faces = MV.load_marker_faces(ego_root)
    hands = []
    for f in frames:
        for s in range(2):
            if not valid[f, s]:
                continue
            v = verts[f, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            if gauge == "ego":
                import torch
                c2w = w2c_to_c2w(torch.as_tensor(cam_full[f], dtype=torch.float32)[None]).numpy()[0]
                vw = v @ c2w[:3, :3].T + c2w[:3, 3]
            else:
                R, t = extr[f, :3, :3], extr[f, :3, 3]
                vw = (R.T @ (v * g - t).T).T
            # 只保留三顶点都有效的面
            remap = -np.ones(len(vw), int)
            remap[np.nonzero(ok)[0]] = np.arange(int(ok.sum()))
            kept = [(remap[a], remap[b], remap[cc]) for a, b, cc in faces
                    if ok[a] and ok[b] and ok[cc]]
            if len(kept) < 20:
                continue
            hands.append({"vw": vw[ok], "faces": np.array(kept), "side": s,
                          "name": f"frame{f}_side{s}"})
    return P, C, hands, meta, frames


def main():
    ap = argparse.ArgumentParser(description="生成可内联查看的 GIF")
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--ego-root", type=Path, default=DEFAULT_EGO_ROOT)
    ap.add_argument("--gauge", choices=["ego", "ras", "both"], default="ego")
    ap.add_argument("--max-points", type=int, default=150_000)
    ap.add_argument("--turntable", type=int, default=36, help="turntable 帧数")
    ap.add_argument("--size", type=int, default=720)
    ap.add_argument("--seq", action="store_true", help="同时把 2D 序列压成 GIF")
    args = ap.parse_args()

    scene_dir = args.scene_dir
    verify = scene_dir / "verify"
    verify.mkdir(parents=True, exist_ok=True)

    for gauge in (["ego", "ras"] if args.gauge == "both" else [args.gauge]):
        P, C, hands, meta, frames = gather_with_world(scene_dir, gauge, args.ego_root,
                                                      args.max_points)
        print(f"[GIF] gauge={gauge}  点云={len(P)}  手={len(hands)} ({[h['name'] for h in hands]})")
        turntable_gif(P, C, hands, verify / f"turntable_{gauge}.gif",
                      n_frames=args.turntable, size=args.size)

    if args.seq:
        for name in ("overlay2d", "render3d", "sideview"):
            paths = sorted(verify.glob(f"{name}_frame*.png"))
            if paths:
                seq_gif(paths, verify / f"{name}.gif")
        cmp_mp4 = verify / "compare_ego_vs_local.mp4"
        if cmp_mp4.exists():
            subprocess_gif(cmp_mp4, verify / "compare.gif")


def subprocess_gif(mp4_path, out_path, width=640, fps=10, max_frames=30):
    """从 mp4 抽帧压 GIF."""
    from PIL import Image
    cap = cv2.VideoCapture(str(mp4_path))
    frames, i = [], 0
    step = 2
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            h, w = f.shape[:2]
            f = cv2.resize(f, (width, int(h * width / w)))
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE))
        i += 1
    cap.release()
    if frames:
        frames = frames[:max_frames]
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        print(f"[GIF] {out_path}  ({len(frames)} 帧, {out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
