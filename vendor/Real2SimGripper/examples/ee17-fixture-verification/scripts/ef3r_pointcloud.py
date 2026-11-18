#!/usr/bin/env python3
"""ef3r_pointcloud.py — 用 EgoFound3R 的 VGGT 米制深度 + 相机位姿重建场景点云,
并把**被抓取物体**单独分割渲染 (EF3R 世界系, y-up → 可视化转 z-up)。

输入: EgoFound3R outputs/<clip>/ 的
  camera/depth_metric_global.npy (21 锚点帧, 512x288 米制深度)
  camera/K_full.npy              (画布像素系内参)
  camera/traj_c2w.npy            (300,4,4) c2w
  hands/joint_xyz_world.npy      (300,2,21,3)
  meta.json                      (anchors: 21 锚点的帧号)
  + 原始抽帧目录 (彩色化, 默认 3d_ws/<clip>/extracted_images)

物体分割: 抓握段 (HOLD) 手部质心 ≈ 物体中心 (物体在手里);
  物体点 = 所有锚点中距该中心 < --obj-radius 的场景点
  (自然覆盖 "桌上→在手→放下" 全程的物体表面)。

输出: <out>/scene_cloud.ply (含色), 物体红色/手关节品红高亮;
      3 个视角 PNG (matplotlib, 无额外依赖)。
"""
import argparse
import json
import os

import numpy as np

# EF3R 世界 (y-up) → 可视化 (z-up): 与 ef3r_to_targets 同款轴变换
M_YUP2ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)


def load_frames_rgb(frame_dir, w, h, frame_idx):
    from PIL import Image
    p = os.path.join(frame_dir, f"{frame_idx:04d}.jpg")
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("RGB").resize((w, h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef3r-dir", required=True)
    ap.add_argument("--frames-dir", default=None, help="原始抽帧目录 (彩色化)")
    ap.add_argument("--hand", default="right", choices=["left", "right"])
    ap.add_argument("--stride", type=int, default=2, help="像素抽样步长")
    ap.add_argument("--obj-radius", type=float, default=0.08, help="物体分割半径 (m)")
    ap.add_argument("--voxel", type=float, default=0.006, help="体素降采样 (m), PLY 输出用")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.ef3r_dir, "pointcloud")
    os.makedirs(out_dir, exist_ok=True)
    h_i = 1 if args.hand == "right" else 0

    meta = json.load(open(os.path.join(args.ef3r_dir, "meta.json")))
    anchors = meta["camera"]["anchors"]
    W, H = meta["camera"]["image_size_hw"][1], meta["camera"]["image_size_hw"][0]
    K_all = np.load(os.path.join(args.ef3r_dir, "camera/K_full.npy")).reshape(-1, 3, 3)
    c2w = np.load(os.path.join(args.ef3r_dir, "camera/traj_c2w.npy"))
    dep = np.load(os.path.join(args.ef3r_dir, "camera/depth_metric_global.npy"))
    Jw = np.load(os.path.join(args.ef3r_dir, "hands/joint_xyz_world.npy"))[:, h_i]
    N = Jw.shape[0]
    print(f"锚点 {len(anchors)} 帧, 画布 {W}x{H}, K fx={K_all[anchors[0]][0,0]:.1f} (逐帧内参)")

    # HOLD 段手部质心 = 抓取物体中心 (无 state_labels 时用全程中位)
    hold = (0, N)
    sl = os.path.join(args.ef3r_dir, "state_labels.json")
    if os.path.exists(sl):
        s = json.load(open(sl))["states"]
        if "HOLD" in s:
            hold = tuple(s["HOLD"])
    palm = Jw[hold[0]:hold[1]][:, [0, 5, 9, 13, 17], :].reshape(-1, 3).mean(0)
    print(f"抓取物体中心 (HOLD 手部质心): {np.round(palm, 3)}")

    us, vs = np.meshgrid(np.arange(0, W, args.stride), np.arange(0, H, args.stride))
    pts_all, col_all, obj_all, pts_hand = [], [], [], []
    for i, f in enumerate(anchors):
        if f >= N:
            continue
        fx, fy, cx, cy = K_all[f][0, 0], K_all[f][1, 1], K_all[f][0, 2], K_all[f][1, 2]
        D = dep[i]
        z = D[vs, us]
        m = np.isfinite(z) & (z > 0.05) & (z < 2.5)
        if not m.any():
            continue
        u, v, zz = us[m], vs[m], z[m]
        p_cam = np.stack([(u - cx) / fx * zz, (v - cy) / fy * zz, zz], axis=1)
        T = c2w[f]
        p_w = p_cam @ T[:3, :3].T + T[:3, 3]
        d_hand = np.linalg.norm(p_w[:, None, :] - Jw[f][None, :, :], axis=2).min(1)
        near_obj = np.linalg.norm(p_w - palm, axis=1) < args.obj_radius
        obj_all.append(p_w[near_obj])
        pts_hand.append(Jw[f])
        keep = ~near_obj & (d_hand > 0.02)          # 物体/手单独管理, 场景排除手附近
        pts_all.append(p_w[keep])
        if args.frames_dir:
            rgb = load_frames_rgb(args.frames_dir, W, H, f)
            if rgb is not None:
                col_all.append(rgb[v, u][keep])
    P = np.concatenate(pts_all); O = np.concatenate(obj_all); HJ = np.concatenate(pts_hand)
    C = np.concatenate(col_all) / 255.0 if col_all else np.full((len(P), 3), 0.75)
    print(f"场景点 {len(P)/1e3:.0f}k, 物体点 {len(O)/1e3:.0f}k, 手关节采样 {len(HJ)}")

    # ---- PLY (体素降采样) ----
    def voxel_sub(P_, C_, vox):
        key = np.floor(P_ / vox).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        return P_[idx], C_[idx]
    Pv, Cv = voxel_sub(P, C, args.voxel)
    Ov, _ = voxel_sub(O, np.zeros((len(O), 3)), args.voxel)
    np.savez(os.path.join(out_dir, "scene_cloud.npz"), points=P.astype(np.float32),
             colors=C.astype(np.float32), object_points=O.astype(np.float32))
    with open(os.path.join(out_dir, "scene_cloud.ply"), "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(Pv)+len(Ov)+len(HJ)}\n"
                "property float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(Pv, Cv):
            f.write("%.4f %.4f %.4f %d %d %d\n" % (*p, *(c * 255).astype(int)))
        for p in Ov:
            f.write("%.4f %.4f %.4f 255 60 60\n" % tuple(p))       # 物体=红
        for p in HJ:
            f.write("%.4f %.4f %.4f 255 0 255\n" % tuple(p))       # 手关节=品红
    print(f"✓ PLY -> {out_dir}/scene_cloud.ply ({len(Pv)+len(Ov)+len(HJ)} 点, 体素 {args.voxel*1000:.0f}mm)")

    # ---- 渲染 (z-up) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Pz = P @ M_YUP2ZUP.T; Oz = O @ M_YUP2ZUP.T; Hz = HJ @ M_YUP2ZUP.T
    views = [("overview", 22, -60), ("top", 88, -90), ("object_closeup", 18, -45)]
    for nm, elev, azim in views:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        s = 0.5 if nm != "object_closeup" else 2.0
        ax.scatter(Pz[::3, 0], Pz[::3, 1], Pz[::3, 2], s=s, c=C[::3], linewidths=0)
        ax.scatter(Oz[:, 0], Oz[:, 1], Oz[:, 2], s=1.5, c="red", label="grasped object")
        ax.scatter(Hz[:, 0], Hz[:, 1], Hz[:, 2], s=4, c="magenta", label="hand joints")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z (up)")
        if nm == "object_closeup":
            ctr = (Oz.mean(0) if len(Ov) else Hz.mean(0))
            r = 0.25
            ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r)
            ax.set_zlim(ctr[2]-r, ctr[2]+r)
        ax.view_init(elev=elev, azim=azim)
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out = os.path.join(out_dir, f"render_{nm}.png")
        plt.savefig(out, dpi=110)
        plt.close()
        print(f"✓ {out}")


if __name__ == "__main__":
    main()
