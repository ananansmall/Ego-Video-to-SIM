"""
export_egofound_assets.py - 导出可直接打开的 3D 资产 (PLY / GLB)
=================================================================

目的: 把 "场景点云" 与 "EgoFound3R 手部 mesh" 导出到同一坐标系、同一比例尺,
方便在 MeshLab / Blender / CloudCompare 里直接观察手与场景点云的尺度/位置关系。

两种 gauge 都要 (--source ras|ego|both, 默认 both):

A. RAS gauge (--source ras):  目录 assets/
   场景点 = 本地 VGGT-Omega world_points (不乘任何因子)
   手部点 = hand_metric x g -> 本地相机系 -> 本地世界系 (本地 extrinsics 求逆)
   -> 用于和 RAS 物体/GLB 混放 (需要标定 g)

B. EgoFound3R 原生 gauge (--source ego):  目录 assets_ego/
   EgoFound3R 一次前向同时输出 metric 场景深度/相机 与 metric 手部 mesh,
   两者天然同 gauge, 零标定:
   场景点 = depth_metric_global 用 intrinsics_global + camera_pose_metric_global 反投影
   手部点 = hand 的 vertex_xyz_metric 用 camera_pose_metric_full 逆变换到世界系
   -> 用于验证 "尺度比例本来就对" (只看效果用这个)

用法:
  cd ReplicateAnyScene
  /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/export_egofound_assets.py \
      --scene-dir output_v3/egofound_verify/weixin_195833_224_h264
"""

import argparse
import json
import os
from pathlib import Path

os.environ["LIDRA_SKIP_INIT"] = "true"

import importlib.util
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EGO_ROOT = REPO_ROOT.parent / "EgoFound3R"

SIDE_COLORS_RGB = [(255, 165, 0), (0, 128, 255)]   # side0 橙, side1 蓝


def tnp(t):
    return t.detach().cpu().float().numpy()


def load_marker_mesh_module(ego_root: Path):
    path = ego_root / "egohandmetric_prompt" / "data" / "marker_mesh.py"
    spec = importlib.util.spec_from_file_location("ego_marker_mesh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_hand_mesh(verts_world, faces, side, valid_mask):
    """用手部顶点+marker 面构建实体 mesh; 仅保留三顶点全有效的面, 并重映射顶点索引."""
    remap = -np.ones(len(verts_world), int)
    remap[np.nonzero(valid_mask)[0]] = np.arange(int(valid_mask.sum()))
    kept = []
    for a, b, c in faces:
        if valid_mask[a] and valid_mask[b] and valid_mask[c]:
            kept.append((remap[a], remap[b], remap[c]))
    if len(kept) < 20:
        return None
    color = np.tile(np.array(SIDE_COLORS_RGB[side], np.uint8), (int(valid_mask.sum()), 1))
    return trimesh.Trimesh(vertices=verts_world[valid_mask].astype(np.float64),
                           faces=np.array(kept),
                           vertex_colors=color,
                           process=False)


def decode_rgb_frames(video_path: Path, count: int, H: int, W: int):
    """从 sampled.mp4 解码 RGB 帧并缩放到模型输入尺寸 (与 ego 预处理一致)."""
    import cv2
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    out = []
    while len(out) < count:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR)
        out.append(f)
    cap.release()
    return out if len(out) == count else None


# ---------------------------------------------------------------------------
# A. RAS gauge
# ---------------------------------------------------------------------------
def export_ras_assets(payload, npz_path, out_dir, ego_root, frames, g_value):
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path, mmap_mode="r")
    extr = np.asarray(data["extrinsics"])
    wp = data["world_points"]; colors = data["colors"]; depths = data["depths"]
    N, H, W, _ = wp.shape
    print(f"[RAS] gauge g={g_value:.4f}  世界系=本地 VGGT-Omega")

    # 场景点云: 合并多帧
    pts, cols = [], []
    for i in range(N):
        d = np.asarray(depths[i])
        valid = np.isfinite(d) & (d > 1e-6)
        sel = np.zeros_like(valid); sel[::4, ::4] = True; sel &= valid
        ys, xs = np.nonzero(sel)
        if len(xs) == 0:
            continue
        p = np.asarray(wp[i])[ys, xs]
        ok = np.isfinite(p).all(axis=1)
        pts.append(p[ok].astype(np.float32))
        cols.append(np.asarray(colors[i])[ys, xs][ok].astype(np.uint8))
    P = np.concatenate(pts); C = np.concatenate(cols)
    if len(P) > 900_000:
        idx = np.random.choice(len(P), 900_000, replace=False)
        P, C = P[idx], C[idx]
    cloud = trimesh.PointCloud(P, colors=C)
    cloud.export(out_dir / "scene_pointcloud.ply")
    print(f"[RAS] scene_pointcloud.ply  点数={len(P)}  bbox对角={np.linalg.norm(np.ptp(P, axis=0)):.3f}")

    # 手部 mesh
    mm = load_marker_mesh_module(ego_root)
    faces = mm.marker_faces_for_count(195).numpy()
    verts = tnp(payload["hand"]["vertex_xyz_metric"])
    valid = tnp(payload["hand"]["valid"]).astype(bool)
    meshes = []
    for f in frames:
        for s in range(2):
            if not valid[f, s]:
                continue
            v = verts[f, s] * g_value
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            R, t = extr[f, :3, :3], extr[f, :3, 3]
            vw = (R.T @ (v - t).T).T
            m = build_hand_mesh(vw, faces, s, ok)
            if m is None:
                continue
            m.export(out_dir / f"hand_mesh_frame{f:04d}_side{s}.glb")
            meshes.append((f, s, m))
            print(f"[RAS] hand_mesh_frame{f:04d}_side{s}.glb  手长={np.linalg.norm(np.ptp(vw, axis=0)):.3f}")
    if meshes:
        try:
            trimesh.Scene([cloud] + [m for _, _, m in meshes]).export(out_dir / "scene_and_hand.glb")
            print("[RAS] scene_and_hand.glb (点云+手部合并)")
        except Exception as e:
            print(f"[RAS][WARN] 合并 GLB 失败: {e}")
    return meshes


# ---------------------------------------------------------------------------
# B. EgoFound3R 原生 gauge (零标定)
# ---------------------------------------------------------------------------
def export_ego_assets(payload, scene_dir, out_dir, ego_root, frames):
    import cv2
    import sys
    import torch
    out_dir.mkdir(parents=True, exist_ok=True)
    # 复用 EgoFound3R 官方几何函数, 保证与它内部 (MetricHead 估尺度用的就是这条路径)
    # 的约定 100% 一致, 而不是我手写的反投影
    sys.path.insert(0, str(ego_root))
    from egohandmetric_prompt.metric_geometry import unproject_depth_to_camera_points
    from egohandmetric_prompt.models.camera_temporal_refiner import w2c_to_c2w

    def cam_to_world(P_cam, w2c):
        c2w = w2c_to_c2w(torch.as_tensor(w2c, dtype=torch.float32)[None]).numpy()[0]
        return P_cam @ c2w[:3, :3].T + c2w[:3, 3]

    d_met = payload["depth_metric_global"]
    d_met = tnp(d_met[..., 0]) if d_met.ndim == 4 else tnp(d_met)
    conf = payload["depth_confidence_global"]
    conf = tnp(conf[..., 0]) if conf.ndim == 4 else tnp(conf)
    K_g = tnp(payload["intrinsics_global"])                 # (T_G,3,3)
    cam_g = tnp(payload["camera_pose_metric_global"])       # (T_G,4,4) metric w2c
    cam_full = tnp(payload["camera_pose_metric_full"])      # (T_R,4,4) metric w2c
    H, W = payload["image_size_hw"]
    g_idx = payload["global_frame_indices"].tolist()
    T_G = len(g_idx)

    rgb = decode_rgb_frames(scene_dir / "sampled.mp4", payload["T_R"], H, W)

    # 场景点云: ego 的 metric 深度 -> 世界系 (零标定)
    pts, cols = [], []
    for j in range(T_G):
        d, c, K, C = d_met[j], conf[j], K_g[j], cam_g[j]
        valid = np.isfinite(d) & (d > 1e-3) & np.isfinite(c)
        if valid.sum() > 100:      # 置信度低分位过滤
            thr = np.percentile(c[valid], 30)
            valid &= (c >= thr)
        sel = np.zeros_like(valid); sel[::3, ::3] = True; sel &= valid
        ys, xs = np.nonzero(sel)
        if len(xs) == 0:
            continue
        # 官方反投影 (与 MetricHead 估尺度同一条路径)
        P_cam_full = unproject_depth_to_camera_points(
            torch.as_tensor(d, dtype=torch.float32)[None, None],
            torch.as_tensor(K, dtype=torch.float32)[None, None],
        )[0, 0].numpy()
        P_cam = P_cam_full[ys, xs]
        Pw = cam_to_world(P_cam, C)          # 官方 w2c_to_c2w
        pts.append(Pw.astype(np.float32))
        if rgb is not None:
            cols.append(rgb[g_idx[j]][ys, xs])
        else:
            zn = np.clip((z - np.percentile(z, 2)) /
                         max(np.percentile(z, 98) - np.percentile(z, 2), 1e-9), 0, 1)
            cols.append(cv2.applyColorMap((zn * 255).astype(np.uint8),
                                          cv2.COLORMAP_TURBO).reshape(-1, 3)[:, ::-1])
    if not pts:
        print("[EGO] 无有效深度点")
        return []
    P = np.concatenate(pts); C = np.concatenate(cols)
    if len(P) > 900_000:
        idx = np.random.choice(len(P), 900_000, replace=False)
        P, C = P[idx], C[idx]
    cloud = trimesh.PointCloud(P, colors=C)
    cloud.export(out_dir / "scene_pointcloud.ply")
    print(f"[EGO] scene_pointcloud.ply  点数={len(P)}  bbox对角={np.linalg.norm(np.ptp(P, axis=0)):.3f} (米制单位)")

    # 手部 mesh (metric 世界系, 零标定)
    mm = load_marker_mesh_module(ego_root)
    faces = mm.marker_faces_for_count(195).numpy()
    verts = tnp(payload["hand"]["vertex_xyz_metric"])
    valid = tnp(payload["hand"]["valid"]).astype(bool)
    meshes = []
    for f in frames:
        for s in range(2):
            if not valid[f, s]:
                continue
            v = verts[f, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            vw = cam_to_world(v, cam_full[f])   # 官方 w2c_to_c2w
            m = build_hand_mesh(vw, faces, s, ok)
            if m is None:
                continue
            m.export(out_dir / f"hand_mesh_frame{f:04d}_side{s}.glb")
            meshes.append((f, s, m))
            print(f"[EGO] hand_mesh_frame{f:04d}_side{s}.glb  手长(米)={np.linalg.norm(np.ptp(vw, axis=0)):.3f}")
    if meshes:
        try:
            trimesh.Scene([cloud] + [m for _, _, m in meshes]).export(out_dir / "scene_and_hand.glb")
            print("[EGO] scene_and_hand.glb (点云+手部合并, 零标定)")
        except Exception as e:
            print(f"[EGO][WARN] 合并 GLB 失败: {e}")
    return meshes


def main():
    ap = argparse.ArgumentParser(description="导出 RAS 点云 + EgoFound3R 手部 mesh 的 PLY/GLB")
    ap.add_argument("--scene-dir", type=Path, required=True,
                    help="含 local_vggt/local_pred.npz 与 egofound3r/inference_output.pt 的目录")
    ap.add_argument("--ego-root", type=Path, default=DEFAULT_EGO_ROOT)
    ap.add_argument("--source", choices=["ras", "ego", "both"], default="both")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="导出手部 mesh 的帧号 (默认: 自动挑 3 帧 = argmax, 中, 末)")
    ap.add_argument("--g", type=float, default=None, help="RAS gauge: 覆盖 g (默认用 g_core)")
    args = ap.parse_args()

    scene_dir = args.scene_dir
    npz_path = scene_dir / "local_vggt" / "local_pred.npz"
    payload_path = scene_dir / "egofound3r" / "inference_output.pt"
    metrics_path = scene_dir / "verify" / "metrics.json"

    import torch
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    hand_valid = tnp(payload["hand"]["valid"]).astype(bool)
    n_valid = hand_valid.sum(axis=1)
    T = hand_valid.shape[0]
    if args.frames:
        frames = list(args.frames)
    else:
        frames = sorted(set([int(np.argmax(n_valid)), T // 2, T - 1]))

    summary = {"frames": frames, "outputs": {}}

    # ---- A. RAS gauge ----
    if args.source in ("ras", "both"):
        g_core = g_pooled = None
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            g_core = (m.get("gauge_calibration_core") or {}).get("g_core_local_raw_per_meter")
            g_pooled = (m.get("gauge_calibration") or {}).get("g_local_raw_per_meter")
        g_value = args.g if args.g is not None else (g_core if g_core else g_pooled)
        if g_value is None:
            raise SystemExit("未找到 g, 请先跑 verify_egofound_scale.py 或 --g 指定")
        print(f"[RAS] g={g_value:.4f} (g_core={g_core}, g_pooled={g_pooled})")
        out_dir = scene_dir / "assets"
        meshes = export_ras_assets(payload, npz_path, out_dir, args.ego_root, frames, g_value)
        summary["outputs"]["ras"] = {
            "dir": str(out_dir), "g_used": float(g_value), "g_core": g_core, "g_pooled": g_pooled,
            "hand_meshes": [f"frame{f}_side{s}" for f, s, _ in meshes],
        }

    # ---- B. EgoFound3R 原生 gauge ----
    if args.source in ("ego", "both"):
        print("[EGO] 零标定: 场景与手部均来自 EgoFound3R 单次前向 (metric)")
        out_dir = scene_dir / "assets_ego"
        meshes = export_ego_assets(payload, scene_dir, out_dir, args.ego_root, frames)
        summary["outputs"]["ego"] = {
            "dir": str(out_dir), "gauge": "metric (米)",
            "world_scope": "ego stitched_video_world",
            "hand_meshes": [f"frame{f}_side{s}" for f, s, _ in meshes],
        }

    spath = scene_dir / "assets_summary.json"
    spath.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[OUT] {spath}")


if __name__ == "__main__":
    main()
