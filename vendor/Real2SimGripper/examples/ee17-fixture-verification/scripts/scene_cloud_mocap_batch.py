#!/usr/bin/env python3
"""scene_cloud_mocap_batch.py — 全 episode 批量: tracker 相机位姿定位 ego 深度点云
→ mocap 世界系场景云 (env 碰撞升级前置)。

与 test/ee5_scene_mocap.py 同逻辑 (ep5 验证过的路径), 三处泛化:
  1. 参数化 ep 号 + EF3R 输出目录自动发现 (mocap_batch 同选择规则)
  2. 锚帧→mocap 行映射用 video_frame_index 精确查找 (原版直接下标, mocap 网格
     与视频帧 ~1:1 但有 ±16 行漂移)
  3. 视频缺省时灰度降级 (碰撞只用 points, 颜色仅渲染)

用法 (robotwin_curobo 或带 torch 的 env):
    python scripts/scene_cloud_mocap_batch.py 1 2 3 5 7 8 9 10 11 12 13 15 16 17 18
输出: <ef3r>/pointcloud/scene_cloud_mocap.npz (already_sim=1, cb 直通)
"""
import glob
import os
import sys

import numpy as np

EE = "/path/to/project_data/dataset/EgoExplorer/dataset/lerobot"
OUTS = "/path/to/project_data/mano_ws/EgoFound3R/outputs"
CLIPS = "/path/to/workspace"


def find_ef(ep):
    cands = [d for d in sorted(glob.glob(f"{OUTS}/ee{ep}_*"))
             if os.path.exists(d + "/meta.json") and not d.endswith("_mocap")]
    return cands[0] if cands else None


def build_scene(ep):
    EF = find_ef(ep)
    if EF is None:
        return f"ee{ep}: 无 EF3R 输出"
    out_p = EF + "/pointcloud/scene_cloud_mocap.npz"
    if os.path.exists(out_p):
        return f"ee{ep}: 已存在 (跳过)"
    import torch
    d = np.load(f"{EE}/episode_meta/episode_{int(ep):06d}/smplx/smplx_episode.npz",
                allow_pickle=True)
    tr_p = np.asarray(d["tracker_position"], np.float64)
    tr_r = np.asarray(d["tracker_rotation"], np.float64)
    vfi = (np.asarray(d["video_frame_index"], np.int64)
           if "video_frame_index" in d.files else None)
    cam_p = tr_p[:, 0, :]
    ef = torch.load(EF + "/inference_output.pt", map_location="cpu", weights_only=False)
    dep = ef["depth"].numpy()                    # (A,H,W) 米制
    Ka = ef["K"].numpy()
    dfr = ef["depth_frame_indices"].numpy().tolist()

    # 锚帧视频帧 → mocap 行 (精确); 无 vfi 时回退直下标
    def row_of(fa):
        if vfi is not None:
            i = int(np.searchsorted(vfi, fa))
            if i < len(vfi) and abs(int(vfi[i]) - fa) <= 2:
                return i
        return fa

    # 颜色: 视频存在则采样, 否则灰度 (碰撞不需要颜色)
    frames = {}
    clip = f"{CLIPS}/{os.path.basename(EF)}.mp4"
    if os.path.exists(clip):
        import imageio.v2 as imageio
        r = imageio.get_reader(clip)
        for i, img in enumerate(r):
            if i in dfr:
                frames[i] = img
            if i > max(dfr):
                break
        r.close()

    rng = np.random.RandomState(0)
    pts_all, col_all = [], []
    for ai, fa in enumerate(dfr):
        z = dep[ai]
        vs, us = np.mgrid[0:z.shape[0], 0:z.shape[1]]
        m = np.isfinite(z) & (z > 0.1) & (z < 6.0)
        zz = z[m]
        K = Ka[ai]
        pc = np.stack([(us[m] - K[0, 2]) / K[0, 0] * zz,
                       (vs[m] - K[1, 2]) / K[1, 1] * zz, zz], 1)   # 相机系
        rw = row_of(fa)
        Rw = tr_r[rw, 0] if tr_r.ndim == 4 else np.eye(3)
        pw = pc @ Rw.T + cam_p[rw]                                   # mocap y-up
        img = frames.get(fa)
        if img is not None:
            Hh, Ww = img.shape[:2]
            uu = (us[m] * (Ww / z.shape[1])).round().astype(int).clip(0, Ww - 1)
            vv = (vs[m] * (Hh / z.shape[0])).round().astype(int).clip(0, Hh - 1)
            col = img[vv, uu] / 255.0
        else:
            col = np.full((len(pw), 3), 0.6)
        sel = rng.choice(len(pw), min(len(pw), 30000), replace=False)
        pts_all.append(pw[sel])
        col_all.append(col[sel])
        print(f"  ee{ep} 锚帧 F{fa} (row {rw}): {len(pw)} 点", flush=True)

    P = np.concatenate(pts_all)
    C = np.concatenate(col_all)
    # mocap y-up → sim z-up (与 ee_mocap_targets 同变换) + 地板对齐
    ankle_y = float(np.min(tr_p[:, 3:5, 1]))
    floor_y = ankle_y - 0.10
    M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    P = P @ M.T
    P[:, 2] -= (-floor_y)
    os.makedirs(EF + "/pointcloud", exist_ok=True)
    np.savez(out_p, points=P.astype(np.float32), colors=C.astype(np.float32),
             already_sim=np.array([1]))
    print(f"✓ ee{ep}: {out_p} ({len(P)} 点, z 范围 "
          f"{P[:,2].min():.2f}~{P[:,2].max():.2f})", flush=True)
    return f"ee{ep}: ok ({len(P)} 点)"


if __name__ == "__main__":
    eps = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
    for ep in eps:
        try:
            print(build_scene(ep), flush=True)
        except Exception as e:
            print(f"ee{ep}: 失败 {str(e)[:140]}", flush=True)
