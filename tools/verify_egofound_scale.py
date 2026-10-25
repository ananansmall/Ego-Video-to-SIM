"""
verify_egofound_scale.py — EgoFound3R 手部 mesh ↔ VGGT-Omega 点云 尺度对齐验证
================================================================================

目的:
  验证 EgoFound3R (MINT-SJTU) 输出的 metric 手部 mesh 能否与本地 ReplicateAnyScene
  VGGT-Omega 管线的点云在同一视角、同一尺度下渲染对齐。

原理 (已通过双方源码确认):
  1. EgoFound3R 的几何骨干是冻结的 VGGT-Omega (同权重 vggt_omega_1b_512.pt,
     同 512/patch16 预处理), 因此同一帧的 raw 深度/相机/内参天然一致。
  2. EgoFound3R 的 metric 化 = 全局均匀缩放因子 f (raw→meter, per-clip 标量):
        depth_metric  = depth_raw × f
        camera_metric = camera_raw (仅平移 × f, 旋转不变)
     且 scale_contract.metric_scale_applies_to_hand = False →
     手部 mesh 自身即为 metric (米), 与 depth_metric_global 同在 metric 相机系。
  3. 重要实测结论 (2026-10 验证): EgoFound3R 的 raw 几何走 "多速率 + 第18层后
     手部 prompt 注入" 前向, 与本地干净 VGGT-Omega 前向存在系统性差异 (实测约 2x),
     因此 ego 的 f 不能直接乘到本地点云上。两种正确渲染方式:
       A. 本地 gauge (脚本默认渲染): 场景=本地 raw 点云; 手部 × g (g 由 ego 内部
          同款估计器经验校准: 手部顶点深度 vs 点云深度 的加权中位数)
       B. ego 自洽 gauge: ego depth_metric + ego metric 手部 (仅 G 锚帧有深度)

帧对齐 (严格 1:1):
  - 从源视频均匀采样 N 帧存 PNG → 本地 VGGT-Omega 输入
    (与项目 load_and_preprocess_images 完全一致的预处理路径)
  - 同 N 帧无损重编码为 sampled.mp4 → EgoFound3R 输入
  - EgoFound3R 参数: --input-fps F --high-fps F (R率=全部输入帧, 1:1),
    --global-stride 5 (训练拓扑), G率锚帧索引入 global_frame_indices 显式导出
  - 深度/相机数值对比按 global_frame_indices 映射到本地帧索引;
    手部渲染按 R 索引 (source_frame_indices = arange(N), 1:1)

用法:
  cd ReplicateAnyScene
  /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/verify_egofound_scale.py \
      --video assets/weixin_195833_224_h264.mp4 --num-frames 60 --device cuda:1

输出 (output_dir):
  frames/frame_XXXX.png          采样帧 (本地 VGGT-Omega 输入)
  sampled.mp4                    无损采样视频 (EgoFound3R 输入)
  local_vggt/local_pred.npz      本地 VGGT-Omega 原始输出
  egofound3r/inference_output.pt EgoFound3R 完整输出
  egofound3r/overlay.mp4         EgoFound3R 自带 2D 叠加 (参考)
  verify/metrics.json            全部对比指标
  verify/overlay2d_*.png         手部 + 场景点云 2D 叠加 (模型视角)
  verify/render3d_*.png          手部 + 场景点云 同视角 3D 渲染 (相机系)
  verify/world_frame*.ply        手部 + 场景点云 (metric 世界系, 可用 MeshLab 打开)
  verify/contact_check.png       手部深度 vs 场景深度 一致性散点
"""

import os
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import gc
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]                     # ReplicateAnyScene/
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_VGGT_OMEGA_CKPT = "/mnt/data/lza/models/vggt_omega/vggt_omega_1b_512.pt"
DEFAULT_EGO_CKPT = "/mnt/data/lza/models/egofound3r/EgoFound3R.pt"
DEFAULT_EGO_ROOT = REPO_ROOT.parent / "EgoFound3R"                  # src/EgoFound3R/

# MANO 21 关节骨架 (与 EgoFound3R infer_marker_video.py 一致)
MANO_JOINT_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

SIDE_COLORS_BGR = [(0, 165, 255), (255, 128, 0)]   # side0 橙, side1 蓝
SIDE_COLORS_RGB = [(255, 165, 0), (0, 128, 255)]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def log(msg=""):
    print(msg, flush=True)


def tnp(t):
    """安全 tensor→numpy (处理 bfloat16 等 numpy 不支持的类型)."""
    return t.detach().cpu().float().numpy()


def load_marker_mesh_module(ego_root: Path):
    """独立加载 EgoFound3R 的 marker_mesh.py (只依赖 torch, 不触发包 __init__,
    避免与本地已导入的 vggt_omega 模块冲突)."""
    path = ego_root / "egohandmetric_prompt" / "data" / "marker_mesh.py"
    spec = importlib.util.spec_from_file_location("ego_marker_mesh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_video_frames(video_path: Path, num_frames: int, frames_dir: Path):
    """从源视频均匀采样 num_frames 帧, 保存 PNG, 返回 (png路径list, 源帧索引list)."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()
    if len(all_frames) == 0:
        raise RuntimeError(f"视频无帧: {video_path}")
    if len(all_frames) < total:
        total = len(all_frames)
    indices = np.linspace(0, total - 1, num_frames).astype(int).tolist()
    png_paths = []
    for k, src_idx in enumerate(indices):
        p = frames_dir / f"frame_{k:04d}.png"
        cv2.imwrite(str(p), all_frames[src_idx])
        png_paths.append(p)
    del all_frames
    gc.collect()
    return png_paths, indices


def build_lossless_sampled_video(png_paths, out_path: Path, fps: float):
    """把采样帧无损重编码为视频 (EgoFound3R 的 cv2 可解码), 保证两边输入帧一致."""
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(png_paths[0].parent / "frame_%04d.png"),
        "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sample_depth_bilinear(depth, uv):
    """双线性采样深度图. depth: (H,W) float32; uv: (M,2) 像素坐标(x,y)."""
    H, W = depth.shape
    x = np.clip(uv[:, 0], 0, W - 1).astype(np.float64)
    y = np.clip(uv[:, 1], 0, H - 1).astype(np.float64)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    wx = x - x0; wy = y - y0
    d = (depth[y0, x0] * (1 - wx) * (1 - wy) + depth[y0, x1] * wx * (1 - wy) +
         depth[y1, x0] * (1 - wx) * wy + depth[y1, x1] * wx * wy)
    valid = (depth[y0, x0] > 0) & (depth[y0, x1] > 0) & (depth[y1, x0] > 0) & (depth[y1, x1] > 0)
    return d.astype(np.float32), valid


def project_points(points_cam, K):
    """相机系 3D 点 → 像素 uv. points_cam: (M,3); K: (3,3). 返回 (uv, valid)."""
    z = points_cam[:, 2]
    valid = np.isfinite(points_cam).all(axis=1) & (z > 1e-6)
    safe_z = np.where(valid, z, 1.0)
    uv = np.stack([K[0, 0] * points_cam[:, 0] / safe_z + K[0, 2],
                   K[1, 1] * points_cam[:, 1] / safe_z + K[1, 2]], axis=1)
    return uv, valid


def rotation_angle_deg(R_a, R_b):
    """两个旋转矩阵之间的夹角 (度)."""
    R = R_a @ R_b.T
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


# ---------------------------------------------------------------------------
# Step 1/2: 本地 VGGT-Omega
# ---------------------------------------------------------------------------
def run_local_vggt(png_paths, ckpt, device, out_dir: Path):
    from src.vggt_omega_predict import load_vggt_omega_model, vggt_omega_predict
    from vggt_omega.utils.load_fn import load_and_preprocess_images

    log("[LOCAL] 加载帧并预处理 (VGGT-Omega balanced 512/patch16) ...")
    frames = load_and_preprocess_images([str(p) for p in png_paths], image_resolution=512)
    frames = frames.to(device)
    N, C, H, W = frames.shape
    log(f"[LOCAL] 输入张量: {tuple(frames.shape)}  (H={H}, W={W})")

    model = load_vggt_omega_model(ckpt).to(device)
    t0 = time.time()
    pred = vggt_omega_predict(frames, model)
    log(f"[LOCAL] VGGT-Omega 推理完成 ({time.time()-t0:.1f}s)")
    del model
    torch.cuda.empty_cache(); gc.collect()

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "local_pred.npz",
        depths=pred["depths"].astype(np.float32),
        extrinsics=pred["extrinsics"].astype(np.float64),
        intrinsic=pred["intrinsic"].astype(np.float64),
        world_points=pred["world_points"].astype(np.float32),
        world_points_conf=pred["world_points_conf"].astype(np.float32),
        colors=pred["colors"],
    )
    import trimesh
    pred["point_cloud_data"].export(out_dir / "point_cloud.ply")
    log(f"[LOCAL] 保存 → {out_dir/'local_pred.npz'}")
    return {"H": H, "W": W, "N": N}


# ---------------------------------------------------------------------------
# Step 3: EgoFound3R (subprocess)
# ---------------------------------------------------------------------------
def run_egofound3r(sampled_video: Path, ego_root: Path, ego_ckpt: Path,
                   vggt_ckpt: Path, out_dir: Path, device: str,
                   num_frames: int, fps: float, in_h: int, in_w: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 自定义 config: 覆盖 vggt_checkpoint_path (其余用仓库 default.toml)
    cfg_path = out_dir / "egofound3r_config.toml"
    cfg_path.write_text(
        "[marker_runtime]\n"
        f'vggt_checkpoint_path = "{vggt_ckpt}"\n', encoding="utf-8")

    # 单 chunk: chunk_seconds 取 (N/fps + 1) → clip_frames >= N → 单一窗口, 单一 f
    chunk_seconds = num_frames / fps + 1.0
    cmd = [
        sys.executable, "infer_marker_video.py",
        "--video-path", str(sampled_video),
        "--checkpoint", str(ego_ckpt),
        "--output-dir", str(out_dir),
        "--device", device,
        "--config", str(cfg_path),
        "--input-fps", str(fps),
        "--high-fps", str(fps),          # H率=R率=全部输入帧 → 1:1
        "--global-stride", "5",          # 训练拓扑
        "--chunk-seconds", str(chunk_seconds),
        "--overlap-seconds", "0.5",
        "--input-resize-mode", "stretch",
        "--input-height", str(in_h),
        "--input-width", str(in_w),
        "--max-frames", str(num_frames),
        "--save-debug-tensors",          # 导出 depth_raw_global / camera_pose_raw_global
        "--no-overlay-frame-images",
        "--allow-output-overwrite",
    ]
    log(f"[EGO] 运行 EgoFound3R: {' '.join(cmd)}")
    log(f"[EGO] chunk_seconds={chunk_seconds} (单chunk → 单一 metric 因子 f)")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(out_dir / "run.log", "w") as lf:
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ego_root), stdout=lf,
                            stderr=subprocess.STDOUT, env=env).returncode
    log(f"[EGO] EgoFound3R 推理完成 (rc={rc}, {time.time()-t0:.1f}s), 日志: {out_dir/'run.log'}")
    if rc != 0:
        tail = (out_dir / "run.log").read_text(errors="ignore")[-4000:]
        raise RuntimeError(f"EgoFound3R 运行失败 (rc={rc}):\n{tail}")

    payload = torch.load(out_dir / "inference_output.pt",
                         map_location="cpu", weights_only=False)
    return payload


# ---------------------------------------------------------------------------
# Step 4: 对比指标
# ---------------------------------------------------------------------------
def compare_and_metrics(payload, local, g_idx, f_value, f_valid):
    """返回 metrics dict. g_idx: G帧→本地帧索引 映射."""
    m = {}
    K_ego = tnp(payload["intrinsics_full"])             # (T_R,3,3) 逐R帧内参
    depths = local["depths"]                       # (N,H,W) raw
    extr = local["extrinsics"]                     # (N,4,4) w2c raw
    intr = local["intrinsic"]                      # (3,3)

    # ---- 4.1 raw 深度对比: EgoFound3R depth_raw_global vs 本地 depths ----
    if "depth_raw_global" in payload:
        d_raw = payload["depth_raw_global"]        # (T_G,H,W,1)
        d_raw = tnp(d_raw[..., 0]) if d_raw.ndim == 4 else tnp(d_raw)
        ratios = []
        for j, li in enumerate(g_idx):
            dl = depths[li]
            de = d_raw[j]
            valid = np.isfinite(dl) & np.isfinite(de) & (dl > 1e-6) & (de > 1e-6)
            if valid.sum() < 100:
                continue
            r = de[valid] / dl[valid]
            ratios.append(np.median(r))
        ratios = np.array(ratios)
        m["raw_depth"] = {
            "n_frames": int(len(ratios)),
            "median_ratio_ego_over_local": float(np.median(ratios)) if len(ratios) else None,
            "ratio_std": float(np.std(ratios)) if len(ratios) else None,
            "per_frame_median_ratio": [float(x) for x in ratios],
        }

    # ---- 4.2 raw 相机对比 (debug tensors) ----
    if "camera_pose_raw_global" in payload:
        c_raw = tnp(payload["camera_pose_raw_global"])   # (T_G,4,4) w2c raw
        rot_err, t_ratio = [], []
        for j, li in enumerate(g_idx):
            R_e, t_e = c_raw[j, :3, :3], c_raw[j, :3, 3]
            R_l, t_l = extr[li, :3, :3], extr[li, :3, 3]
            rot_err.append(rotation_angle_deg(R_e, R_l))
            t_ratio.append(np.linalg.norm(t_e) / max(np.linalg.norm(t_l), 1e-9))
        m["raw_camera"] = {
            "rotation_angle_deg_median": float(np.median(rot_err)),
            "translation_norm_ratio_median": float(np.median(t_ratio)),
        }

    # ---- 4.3 内参对比: ego intrinsics_full (逐R帧) vs 本地 intrinsic (均值) ----
    K_ego = tnp(payload["intrinsics_full"])           # (T_R,3,3)
    m["intrinsics"] = {
        "ego_fx_mean": float(K_ego[:, 0, 0].mean()), "local_fx": float(intr[0, 0]),
        "ego_fy_mean": float(K_ego[:, 1, 1].mean()), "local_fy": float(intr[1, 1]),
        "ego_cx_mean": float(K_ego[:, 0, 2].mean()), "local_cx": float(intr[0, 2]),
        "ego_cy_mean": float(K_ego[:, 1, 2].mean()), "local_cy": float(intr[1, 2]),
        "fx_rel_diff": float(abs(K_ego[:, 0, 0].mean() - intr[0, 0]) / intr[0, 0]),
        "fy_rel_diff": float(abs(K_ego[:, 1, 1].mean() - intr[1, 1]) / intr[1, 1]),
    }

    # ---- 4.4 metric 尺度因子 ----
    m["metric_scale"] = {
        "raw_to_meter_effective": float(f_value),
        "valid": bool(f_valid),
        "note": "depth_metric = depth_raw × f; 手部 mesh 自身 metric, 不乘 f",
    }

    # ---- 4.5 gauge 校准 + 手部 vs 本地点云深度 一致性 (核心验证) ----
    # 重要: EgoFound3R 的 raw 几何走 "多速率 + 第18层后手部 prompt 注入" 前向,
    # 与本地干净 VGGT-Omega 前向存在系统性差异 (见 4.7, 实测 ~2.2x),
    # 因此不能把 ego 的 f 直接乘到本地点云上。
    # 正确做法: 用 EgoFound3R 内部的同款估计器 (手部顶点深度 vs 点云深度的
    # 加权中位数) 经验校准本地 gauge 因子 g, 再评估贴合残差。
    hand = payload["hand"]
    verts = tnp(hand["vertex_xyz_metric"])            # (T_R,2,V,3) metric 相机系
    hand_valid = tnp(hand["valid"]).astype(bool)      # (T_R,2)
    vis = hand.get("vertex_visibility_probability")
    vis = tnp(vis) if vis is not None else None       # (T_R,2,V)
    T_R = verts.shape[0]
    calib_ratio = []      # local_raw_depth / hand_z (metric)
    per_frame = []
    for i in range(T_R):
        for s in range(2):
            if not hand_valid[i, s]:
                continue
            v = verts[i, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 20:
                continue
            if vis is not None:
                ok = ok & (vis[i, s] >= 0.5)
            vv = v[ok]
            # 手部用 ego 自身内参投影 (手部解算与 K_full 自洽; 已在 ego overlay 中验证贴合)
            uv, uv_ok = project_points(vv, K_ego[i])
            inb = uv_ok & (uv[:, 0] >= 0) & (uv[:, 0] < local["W"] - 1) & \
                  (uv[:, 1] >= 0) & (uv[:, 1] < local["H"] - 1)
            if inb.sum() < 20:
                continue
            d_s, d_ok = sample_depth_bilinear(depths[i], uv[inb])
            good = d_ok & (d_s > 1e-6)
            if good.sum() < 20:
                continue
            z_hand = vv[inb][good, 2]
            calib_ratio.append(d_s[good] / z_hand)
            per_frame.append({"frame": i, "side": s, "n": int(good.sum())})
    g_value = None
    if calib_ratio:
        cr = np.concatenate(calib_ratio)
        g_value = float(np.median(cr))
        lr = np.log(cr / g_value)     # 校准后残差
        m["gauge_calibration"] = {
            "n_points": int(cr.size),
            "g_local_raw_per_meter": g_value,
            "note": "hand_in_local_gauge = hand_metric × g; 场景=本地 raw 点云 (不乘任何因子)",
        }
        # 稳健校准: 手指缝隙/边缘会采样到背景 (ratio 偏大), 取最小 60% 分位作为
        # "手部是前景表面" 的核心簇, 给出核心簇标定与残差
        q60 = np.percentile(cr, 60)
        core = cr[cr <= q60]
        g_core = float(np.median(core))
        lr_core = np.log(core / g_core)
        m["gauge_calibration_core"] = {
            "n_points": int(core.size),
            "g_core_local_raw_per_meter": g_core,
            "pooled_median": g_value,
            "frac_core": float(core.size / cr.size),
            "residual_median_log_ratio_core": float(np.median(lr_core)),
            "residual_mad_core": float(np.median(np.abs(lr_core - np.median(lr_core)))),
            "note": "剔除背景采样后的核心簇校准; 与 pooled 的差异反映前景/背景污染程度",
        }
        m["hand_vs_local_depth"] = {
            "n_points": int(lr.size),
            "median_log_ratio": float(np.median(lr)),
            "mad_log_ratio": float(np.median(np.abs(lr - np.median(lr)))),
            "frac_within_5pct": float(np.mean(np.abs(lr) < np.log(1.05))),
            "frac_within_10pct": float(np.mean(np.abs(lr) < np.log(1.10))),
            "note": "校准后 log(hand_z×g / local_raw_depth); 中位≈0, 展宽=真实贴合质量",
        }
    else:
        m["hand_vs_local_depth"] = {"n_points": 0}
    m["g_value"] = g_value
    m["hand_calib_per_frame"] = per_frame

    # ---- 4.6 EgoFound3R 自洽性: 手部深度 vs 自身 metric 深度 (G帧) ----
    if "depth_metric_global" in payload:
        d_met = payload["depth_metric_global"]
        d_met = tnp(d_met[..., 0]) if d_met.ndim == 4 else tnp(d_met)
        K_g = tnp(payload["intrinsics_global"])
        self_lr = []
        for j, li in enumerate(g_idx):
            for s in range(2):
                if not hand_valid[li, s]:
                    continue
                v = verts[li, s]
                ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
                if vis is not None:
                    ok = ok & (vis[li, s] >= 0.5)
                if ok.sum() < 20:
                    continue
                vv = v[ok]
                uv, uv_ok = project_points(vv, K_g[j])
                inb = uv_ok & (uv[:, 0] >= 0) & (uv[:, 0] < d_met.shape[2] - 1) & \
                      (uv[:, 1] >= 0) & (uv[:, 1] < d_met.shape[1] - 1)
                if inb.sum() < 20:
                    continue
                d_s, d_ok = sample_depth_bilinear(d_met[j], uv[inb])
                good = d_ok & (d_s > 1e-6)
                if good.sum() < 20:
                    continue
                self_lr.append(np.log(vv[inb][good, 2] / d_s[good]))
        if self_lr:
            lr = np.concatenate(self_lr)
            m["ego_self_consistency"] = {
                "n_points": int(lr.size),
                "median_log_ratio": float(np.median(lr)),
                "frac_within_10pct": float(np.mean(np.abs(lr) < np.log(1.10))),
            }

    # ---- 4.7 EgoFound3R raw 几何 vs 本地 raw 几何 (架构差异量化) ----
    # ego_raw = ego_metric / f; 比值 = local_raw / ego_raw。
    # ≈1 → 两边 raw 一致 (可互乘 f); 明显偏离 → ego 多速率+prompt 前向的
    # 系统性差异, 必须用 g 校准或改用 ego 自身几何渲染。
    if "depth_metric_global" in payload and f_valid and f_value > 0:
        d_met_all = payload["depth_metric_global"]
        d_met_all = tnp(d_met_all[..., 0]) if d_met_all.ndim == 4 else tnp(d_met_all)
        ratios = []
        for j, li in enumerate(g_idx):
            dm, dl = d_met_all[j], depths[li]
            valid = np.isfinite(dm) & np.isfinite(dl) & (dm > 1e-6) & (dl > 1e-6)
            if valid.sum() < 100:
                continue
            ratios.append(float(np.median(dl[valid] * f_value / dm[valid])))
        if ratios:
            m["ego_raw_vs_local_raw"] = {
                "n_frames": len(ratios),
                "median_local_raw_over_ego_raw": float(np.median(ratios)),
                "note": "≈1 为一致; 本视频实测值用于判断两边 raw 几何是否可直接互换算",
            }
    return m


# ---------------------------------------------------------------------------
# Step 5: 渲染
# ---------------------------------------------------------------------------
def draw_hand_2d(img_bgr, K, joints, verts, valid, marker_edges, side):
    """在 BGR 图像上画一只手 (marker 网格 + MANO 关节骨架)."""
    if not valid:
        return
    color = SIDE_COLORS_BGR[side]
    # marker 顶点
    uv, ok = project_points(verts, K)
    H, W = img_bgr.shape[:2]
    inb = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    pts = uv[inb].astype(np.int32)
    for p in pts:
        cv2.circle(img_bgr, tuple(p), 1, color, -1, cv2.LINE_AA)
    # marker 边 (隔采样)
    if marker_edges is not None and len(marker_edges):
        for a, b in marker_edges[::3]:
            if inb[a] and inb[b]:
                cv2.line(img_bgr, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                         color, 1, cv2.LINE_AA)
    # MANO 关节
    if joints is not None and len(joints):
        uvj, okj = project_points(joints, K)
        inbj = okj & (uvj[:, 0] >= 0) & (uvj[:, 0] < W) & (uvj[:, 1] >= 0) & (uvj[:, 1] < H)
        for a, b in MANO_JOINT_EDGES:
            if inbj[a] and inbj[b]:
                cv2.line(img_bgr, tuple(uvj[a].astype(int)), tuple(uvj[b].astype(int)),
                         (255, 255, 255), 1, cv2.LINE_AA)
        for k in range(len(joints)):
            if inbj[k]:
                cv2.circle(img_bgr, tuple(uvj[k].astype(int)), 2, color, -1, cv2.LINE_AA)


def render_2d_overlay(payload, local, frames_out: Path, render_frames, K_ego, g_value):
    """2D 叠加 (本地 gauge): 场景=本地 raw 深度像素(深染色) + 手部=ego metric × g.
    手部用 ego 内参 K_ego 投影 (与手部解算自洽, ego overlay 已验证贴合)."""
    depths = local["depths"]; colors = local["colors"]
    H, W = local["H"], local["W"]
    hand = payload["hand"]
    joints = payload.get("joint_xyz")
    joints = tnp(joints) if joints is not None else None
    marker_edges = local["marker_edges"]
    hand_valid = tnp(hand["valid"]).astype(bool)
    verts_all = tnp(hand["vertex_xyz_metric"])
    paths = []
    for i in render_frames:
        img = cv2.cvtColor(colors[i].copy(), cv2.COLOR_RGB2BGR)
        # 场景点云: 有效深度像素隔点采样, 按相对深度着色
        d = depths[i]
        valid = np.isfinite(d) & (d > 1e-6)
        ys, xs = np.nonzero(valid)[0][::6], np.nonzero(valid)[1][::6]
        if len(xs):
            zz = d[ys, xs]
            zn = (zz - np.percentile(zz, 2)) / max(np.percentile(zz, 98) - np.percentile(zz, 2), 1e-9)
            zn = np.clip(zn, 0, 1)
            cmap = (cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    .reshape(-1, 3))
            for (x, y), c in zip(zip(xs, ys), cmap):
                img[y, x] = (0.72 * img[y, x] + 0.28 * c).astype(np.uint8)
        # 手部 (metric × g → 本地 raw gauge, 本地 K 投影)
        for s in range(2):
            if not hand_valid[i, s]:
                continue
            v = verts_all[i, s] * g_value
            j = joints[i, s] * g_value if joints is not None else None
            draw_hand_2d(img, K_ego[i], j, v, True, marker_edges, s)
        out = frames_out / f"overlay2d_frame{i:04d}.png"
        cv2.imwrite(str(out), img)
        paths.append(out)
    return paths


def render_ego_metric_view(payload, local, g_idx, frames_out: Path, g_frames):
    """EgoFound3R 自洽视角 (metric gauge): ego metric 深度点云 + ego metric 手部,
    两者同 gauge 无需校准 (仅 G 锚帧有深度)."""
    d_met_all = payload["depth_metric_global"]
    d_met_all = tnp(d_met_all[..., 0]) if d_met_all.ndim == 4 else tnp(d_met_all)
    K_g = tnp(payload["intrinsics_global"])
    hand = payload["hand"]
    verts_all = tnp(hand["vertex_xyz_metric"])
    valid_all = tnp(hand["valid"]).astype(bool)
    H, W = local["H"], local["W"]
    marker_edges = local["marker_edges"]
    paths = []
    for j in g_frames:
        if j >= len(g_idx):
            continue
        li = g_idx[j]
        img = np.zeros((H, W, 3), np.uint8)
        d = d_met_all[j]
        valid = np.isfinite(d) & (d > 1e-6)
        ys, xs = np.nonzero(valid)[0][::4], np.nonzero(valid)[1][::4]
        if len(xs):
            zz = d[ys, xs]
            zn = np.clip((zz - np.percentile(zz, 2)) /
                         max(np.percentile(zz, 98) - np.percentile(zz, 2), 1e-9), 0, 1)
            cols = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
            for (x, y), c in zip(zip(xs, ys), cols):
                img[y, x] = tuple(int(v) for v in c)
        for s in range(2):
            if not valid_all[li, s]:
                continue
            v = verts_all[li, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 10:
                continue
            vv = v[ok]
            uv, uv_ok = project_points(vv, K_g[j])
            inb = uv_ok & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
            pts = uv[inb].astype(np.int32)
            for p in pts:
                cv2.circle(img, tuple(p), 2, SIDE_COLORS_BGR[s], -1, cv2.LINE_AA)
            if marker_edges is not None and len(marker_edges):
                okidx = np.nonzero(ok)[0]
                remap = -np.ones(len(v), int); remap[okidx] = np.arange(len(okidx))
                for a, b in marker_edges:
                    ia, ib = remap[a], remap[b]
                    if ia < 0 or ib < 0:
                        continue
                    pa, pb = uv[ia], uv[ib]
                    if not (0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H):
                        continue
                    cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                             (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"G{j} (R{li}) | EgoFound3R self-consistent | scene=depth_metric | hand=metric",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        out = frames_out / f"egometric_G{j:02d}.png"
        cv2.imwrite(str(out), img)
        paths.append(out)
    return paths


def render_3d_camera_view(payload, local, frames_out: Path, render_frames,
                          K_ego, g_value):
    """相机系软件渲染 (本地 gauge): 场景=本地 raw 深度(像素网格) + 手部=ego metric × g.
    手部用 ego K_ego 投影, 不影响场景点 (场景点即像素位置)."""
    depths = local["depths"]; colors = local["colors"]
    H, W = local["H"], local["W"]
    hand = payload["hand"]
    verts_all = tnp(hand["vertex_xyz_metric"])
    valid_all = tnp(hand["valid"]).astype(bool)
    marker_edges = local["marker_edges"]
    paths = []
    for i in render_frames:
        img = np.zeros((H, W, 3), np.uint8)
        # ---- 场景点云 (本地 raw 相机系) ----
        d = depths[i]
        valid = np.isfinite(d) & (d > 1e-6)
        sel = np.zeros_like(valid); sel[::4, ::4] = True; sel &= valid
        ys, xs = np.nonzero(sel)
        if len(xs) > 60000:  # 上限
            idx = np.random.choice(len(xs), 60000, replace=False)
            ys, xs = ys[idx], xs[idx]
        if len(xs):
            z = d[ys, xs]
            zn = np.clip((z - np.percentile(z, 2)) /
                         max(np.percentile(z, 98) - np.percentile(z, 2), 1e-9), 0, 1)
            cols = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
            # 逐点画 (画家算法: 按深度从远到近)
            order = np.argsort(-z)
            for oi in order:
                cv2.circle(img, (int(xs[oi]), int(ys[oi])), 1,
                           tuple(int(c) for c in cols[oi]), -1, cv2.LINE_AA)
        # ---- 手部 (metric × g → 本地 raw 相机系) ----
        for s in range(2):
            if not valid_all[i, s]:
                continue
            v = verts_all[i, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 10:
                continue
            vv = v[ok] * g_value
            col = SIDE_COLORS_RGB[s]
            uv, uv_ok = project_points(vv, K_ego[i])
            inb = uv_ok & (uv[:, 0] >= 0) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < H - 1)
            pts = uv[inb].astype(np.int32)
            zz = vv[inb, 2]
            for p, z in sorted(zip(pts, zz), key=lambda t: -t[1]):
                cv2.circle(img, tuple(p), 2, col[::-1], -1, cv2.LINE_AA)
            # marker 边
            if marker_edges is not None and len(marker_edges):
                okidx = np.nonzero(ok)[0]
                remap = -np.ones(len(v), int); remap[okidx] = np.arange(len(okidx))
                for a, b in marker_edges:
                    ia, ib = remap[a], remap[b]
                    if ia < 0 or ib < 0:
                        continue
                    pa, pb = uv[ia], uv[ib]
                    if not (0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pb[0] < W and 0 <= pb[1] < H):
                        continue
                    cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)),
                             (255, 255, 255), 1, cv2.LINE_AA)
        # 视角标注
        cv2.putText(img, f"frame {i} | camera space | scene=local_VGGT_raw | hand=EgoFound3R_metric x g={g_value:.4f}",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
        out = frames_out / f"render3d_frame{i:04d}.png"
        cv2.imwrite(str(out), img)
        paths.append(out)
    return paths


def render_side_view(payload, local, frames_out: Path, render_frames, K_local, g_value,
                     yaw_deg: float = 55.0, zoom: float = 2.4):
    """侧视正交渲染 (以手为中心): 场景=本地 raw 深度反投影, 手部=metric × g,
    绕 Y 轴旋转后正交投影 + z-buffer 上色. 手画成网格 (顶点+拓扑边), 便于判断
    手是贴在点云表面上还是悬浮/陷入."""
    depths = local["depths"]
    hand = payload["hand"]
    H, W = local["H"], local["W"]
    verts_all = tnp(hand["vertex_xyz_metric"])
    valid_all = tnp(hand["valid"]).astype(bool)
    marker_edges = local["marker_edges"]
    th = np.deg2rad(yaw_deg)
    c, s = np.cos(th), np.sin(th)

    def rot(Q):
        xr = c * Q[:, 0] + s * Q[:, 2]
        zr = -s * Q[:, 0] + c * Q[:, 2]
        return xr, Q[:, 1], zr

    paths = []
    for i in render_frames:
        # ---- 手部顶点 (metric × g) + 拓扑边 ----
        hp, hcols, hedge = [], [], []
        for sd in range(2):
            if not valid_all[i, sd]:
                continue
            v = verts_all[i, sd] * g_value
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if ok.sum() < 10:
                continue
            idx = np.nonzero(ok)[0]
            remap = -np.ones(len(v), int); remap[idx] = np.arange(len(idx))
            hp.append(v[idx])
            hcols.append(np.array(SIDE_COLORS_RGB[sd], np.float32))
            e = []
            for a_, b_ in marker_edges:
                ia, ib = remap[a_], remap[b_]
                if ia >= 0 and ib >= 0:
                    e.append((ia, ib))
            hedge.append(np.array(e, int) if e else np.zeros((0, 2), int))
        if not hp:
            continue
        # ---- 画布范围: 以手为中心 ----
        hx_all, hy_all, _ = rot(np.concatenate(hp))
        hx_all, hy_all = hx_all[np.isfinite(hx_all)], hy_all[np.isfinite(hy_all)]
        if hx_all.size < 10:
            continue
        cx, cy = np.median(hx_all), np.median(hy_all)
        span = max(np.ptp(hx_all), np.ptp(hy_all), 0.04) * zoom
        x0, x1 = cx - span / 2, cx + span / 2
        y0, y1 = cy - span / 2, cy + span / 2
        Hc = Wc = 700

        def to_px(qx, qy):
            px = ((qx - x0) / (x1 - x0) * (Wc - 1)).astype(int)
            py = ((qy - y0) / (y1 - y0) * (Hc - 1)).astype(int)
            return px, py

        img = np.zeros((Hc, Wc, 3), np.uint8)
        # ---- 场景点 (仅视野内, 按旋转后深度着色, 远→近) ----
        d = depths[i]
        valid = np.isfinite(d) & (d > 1e-6)
        sel = np.zeros_like(valid); sel[::2, ::2] = True; sel &= valid
        ys, xs = np.nonzero(sel)
        z = d[ys, xs]
        X = (xs - K_local[0, 2]) / K_local[0, 0] * z
        Y = (ys - K_local[1, 2]) / K_local[1, 1] * z
        sx, sy, sz = rot(np.stack([X, Y, z], 1))
        px, py = to_px(sx, sy)
        inb = (px >= 0) & (px < Wc) & (py >= 0) & (py < Hc) & np.isfinite(sz)
        if inb.sum() > 50:
            zz = sz[inb]
            zn = np.clip((zz - np.percentile(zz, 2)) /
                         max(np.percentile(zz, 98) - np.percentile(zz, 2), 1e-9), 0, 1)
            scols = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
            order = np.argsort(-zz)
            pxi, pyi = px[inb][order], py[inb][order]
            for oi in range(len(order)):
                cv2.circle(img, (int(pxi[oi]), int(pyi[oi])), 1,
                           tuple(int(v) for v in scols[order][oi]), -1)
        # ---- 手部网格 (边 + 顶点) ----
        for Q, col, E in zip(hp, hcols, hedge):
            qx, qy, qz = rot(Q)
            hpx, hpy = to_px(qx, qy)
            fin = np.isfinite(qx) & np.isfinite(qy)
            bgr = tuple(int(v) for v in col[::-1])
            for a_, b_ in E:
                if fin[a_] and fin[b_]:
                    cv2.line(img, (int(hpx[a_]), int(hpy[a_])), (int(hpx[b_]), int(hpy[b_])),
                             bgr, 1, cv2.LINE_AA)
            for oi in np.argsort(-qz):
                if fin[oi] and 0 <= hpx[oi] < Wc and 0 <= hpy[oi] < Hc:
                    cv2.circle(img, (int(hpx[oi]), int(hpy[oi])), 2, bgr, -1, cv2.LINE_AA)
        cv2.putText(img, f"frame {i} | side view yaw={yaw_deg:.0f}deg | scene=local_raw | hand=metric x g",
                    (10, Hc - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        out = frames_out / f"sideview_frame{i:04d}.png"
        cv2.imwrite(str(out), img)
        paths.append(out)
    return paths


def export_world_plys(payload, local, frames_out: Path, render_frames, g_value):
    """导出世界系 PLY (本地 gauge): 场景=本地 world_points (raw) + 手部=ego metric × g
    经本地相机外参逆变换到世界系. 可用 MeshLab/Blender 打开."""
    import trimesh
    depths = local["depths"]; colors = local["colors"]
    world_points = local["world_points"]
    extr = local["extrinsics"]                    # (N,4,4) 本地 w2c raw
    H, W = local["H"], local["W"]
    hand = payload["hand"]
    verts_all = tnp(hand["vertex_xyz_metric"])
    valid_all = tnp(hand["valid"]).astype(bool)
    paths = []
    for i in render_frames:
        # 场景: 本地 world_points (raw, 不乘任何因子)
        d = depths[i]
        sel = np.zeros((H, W), bool); sel[::6, ::6] = True
        sel &= np.isfinite(d) & (d > 1e-6)
        ys, xs = np.nonzero(sel)
        if len(xs) > 80000:
            idx = np.random.choice(len(xs), 80000, replace=False)
            ys, xs = ys[idx], xs[idx]
        P = world_points[i][ys, xs]
        C = colors[i][ys, xs]
        parts_p, parts_c = [P], [C]
        # 手部: metric × g → 本地相机系 → 本地世界系 (本地 w2c 求逆)
        R, t = extr[i, :3, :3], extr[i, :3, 3]
        for s in range(2):
            if not valid_all[i, s]:
                continue
            v = verts_all[i, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            vv = v[ok] * g_value
            vw = (R.T @ (vv - t).T).T
            parts_p.append(vw)
            parts_c.append(np.tile(np.array(SIDE_COLORS_RGB[s], np.uint8), (len(vw), 1)))
        cloud = trimesh.PointCloud(np.vstack(parts_p), colors=np.vstack(parts_c))
        out = frames_out / f"world_frame{i:04d}.ply"
        cloud.export(out)
        paths.append(out)
    return paths


def compile_frames_to_video(paths, out_path: Path, fps: float = 30.0):
    """把渲染帧序列合成 mp4 (便于看运动过程中的贴合)."""
    paths = sorted(paths, key=lambda p: p.name)
    if not paths:
        return None
    img = cv2.imread(str(paths[0]))
    if img is None:
        return None
    h, w = img.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))
    if not writer.isOpened():
        return None
    for p in paths:
        f = cv2.imread(str(p))
        if f is None:
            continue
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        writer.write(f)
    writer.release()
    return out_path


def make_montage(paths, out_path: Path, cols: int = 4, tile: tuple = (320, 300)):
    """把渲染帧拼成一张网格图, 便于快速浏览."""
    paths = sorted(paths, key=lambda p: p.name)
    if not paths:
        return None
    imgs = []
    for p in paths:
        f = cv2.imread(str(p))
        if f is None:
            continue
        imgs.append(cv2.resize(f, tile))
    if not imgs:
        return None
    rows = []
    for r in range(0, len(imgs), cols):
        row = imgs[r:r + cols]
        while len(row) < cols:
            row.append(np.zeros((tile[1], tile[0], 3), np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path


def plot_contact_check(payload, local, out_path: Path, g_value, K_ego):
    """校准后 手部深度×g vs 本地深度 的 log-ratio 直方图 + 散点 (ego K 投影)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = local["depths"]
    hand = payload["hand"]
    verts = tnp(hand["vertex_xyz_metric"])
    valid_all = tnp(hand["valid"]).astype(bool)
    vis = hand.get("vertex_visibility_probability")
    vis = tnp(vis) if vis is not None else None
    lrs, zs = [], []
    for i in range(verts.shape[0]):
        for s in range(2):
            if not valid_all[i, s]:
                continue
            v = verts[i, s]
            ok = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-4)
            if vis is not None:
                ok = ok & (vis[i, s] >= 0.5)
            if ok.sum() < 20:
                continue
            vv = v[ok]
            uv, uv_ok = project_points(vv, K_ego[i])
            inb = uv_ok & (uv[:, 0] >= 0) & (uv[:, 0] < local["W"] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < local["H"] - 1)
            if inb.sum() < 20:
                continue
            d_s, d_ok = sample_depth_bilinear(depths[i], uv[inb])
            good = d_ok & (d_s > 1e-6)
            if good.sum() < 20:
                continue
            lrs.append(np.log(vv[inb][good, 2] * g_value / d_s[good]))
            zs.append(vv[inb][good, 2])
    if not lrs:
        return None
    lrs = np.concatenate(lrs); zs = np.concatenate(zs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(lrs, bins=80, color="steelblue")
    axes[0].axvline(0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("log(hand_z x g / local_raw_depth)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"median={np.median(lrs):+.4f}  (0=perfect fit)")
    axes[1].scatter(zs[::max(1, len(zs)//4000)], lrs[::max(1, len(zs)//4000)], s=1, alpha=0.3)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set_xlabel("hand vertex z (m)")
    axes[1].set_ylabel("log ratio")
    axes[1].set_title("log ratio vs hand depth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EgoFound3R 手部 ↔ VGGT-Omega 点云 尺度对齐验证")
    ap.add_argument("--video", type=Path,
                    default=REPO_ROOT / "assets/weixin_195833_224_h264.mp4",
                    help="输入视频 (需 cv2 可解码, 如 h264 版本)")
    ap.add_argument("--num-frames", type=int, default=60)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--vggt-omega-ckpt", type=Path, default=Path(DEFAULT_VGGT_OMEGA_CKPT))
    ap.add_argument("--ego-ckpt", type=Path, default=Path(DEFAULT_EGO_CKPT))
    ap.add_argument("--ego-root", type=Path, default=DEFAULT_EGO_ROOT)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--skip-local", action="store_true", help="复用已有 local_pred.npz")
    ap.add_argument("--skip-ego", action="store_true", help="复用已有 inference_output.pt")
    ap.add_argument("--max-render-frames", type=int, default=8)
    ap.add_argument("--render-all", action="store_true",
                    help="渲染全部帧 (并合成 mp4), 便于看运动过程中的贴合")
    args = ap.parse_args()

    out_dir = args.output_dir or (REPO_ROOT / "output_v3" / "egofound_verify" / args.video.stem)
    # 必须绝对路径: EgoFound3R 子进程以仓库目录为 cwd, 相对路径会解析错
    out_dir = out_dir.expanduser().resolve()
    frames_dir = out_dir / "frames"
    verify_dir = out_dir / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    log(f"输出目录: {out_dir}")

    # ---- Step 1: 帧采样 + 无损视频 ----
    npz_path = out_dir / "local_vggt" / "local_pred.npz"
    if args.skip_local and npz_path.exists():
        png_paths = sorted(frames_dir.glob("frame_*.png"))
        log(f"[FRAMES] 复用已有 {len(png_paths)} 帧")
    else:
        log(f"[FRAMES] 从 {args.video} 均匀采样 {args.num_frames} 帧 ...")
        png_paths, src_idx = sample_video_frames(args.video, args.num_frames, frames_dir)
        (out_dir / "source_frame_indices.json").write_text(json.dumps(src_idx))
        sampled_video = out_dir / "sampled.mp4"
        log(f"[FRAMES] 无损重编码 → {sampled_video}")
        build_lossless_sampled_video(png_paths, sampled_video, args.fps)
    sampled_video = out_dir / "sampled.mp4"

    # ---- Step 2: 本地 VGGT-Omega ----
    if args.skip_local and npz_path.exists():
        data = np.load(npz_path)
        local = {"depths": data["depths"], "extrinsics": data["extrinsics"],
                 "intrinsic": data["intrinsic"], "world_points": data["world_points"],
                 "world_points_conf": data["world_points_conf"], "colors": data["colors"],
                 "H": int(data["depths"].shape[1]), "W": int(data["depths"].shape[2])}
        log(f"[LOCAL] 复用 {npz_path}")
    else:
        info = run_local_vggt(png_paths, args.vggt_omega_ckpt, args.device,
                              out_dir / "local_vggt")
        data = np.load(npz_path)
        local = {"depths": data["depths"], "extrinsics": data["extrinsics"],
                 "intrinsic": data["intrinsic"], "world_points": data["world_points"],
                 "world_points_conf": data["world_points_conf"], "colors": data["colors"],
                 "H": info["H"], "W": info["W"]}
    N = local["depths"].shape[0]

    # marker 拓扑 (边)
    mm = load_marker_mesh_module(args.ego_root)
    local["marker_edges"] = mm.marker_edges_for_count(195).numpy()

    # ---- Step 3: EgoFound3R ----
    ego_out = out_dir / "egofound3r"
    payload_path = ego_out / "inference_output.pt"
    if args.skip_ego and payload_path.exists():
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        log(f"[EGO] 复用 {payload_path}")
    else:
        payload = run_egofound3r(sampled_video, args.ego_root, args.ego_ckpt,
                                 args.vggt_omega_ckpt, ego_out, args.device,
                                 N, args.fps, local["H"], local["W"])

    # ---- Step 4: 对齐与指标 ----
    g_idx = payload["global_frame_indices"].tolist()      # G帧 → R(=本地) 帧索引
    log(f"[ALIGN] T_R={payload['T_R']}  T_G={len(g_idx)}  "
        f"image_size_hw={payload['image_size_hw']}  (本地 {local['H']}x{local['W']})")
    assert payload["image_size_hw"] == [local["H"], local["W"]], \
        f"输入尺寸不一致: ego={payload['image_size_hw']} local={[local['H'], local['W']]}"

    # metric 因子 f (单 chunk)
    clip_meta = payload["clip_metadata"]
    f_value = float(payload["metric_scale"]["raw_to_meter_effective"].reshape(-1)[0]) \
        if "raw_to_meter_effective" in payload.get("metric_scale", {}) \
        else float(clip_meta[0]["raw_to_meter_fused"])
    f_valid = bool(payload["metric_scale"].get("valid", torch.tensor([True]))[0]) \
        if "valid" in payload.get("metric_scale", {}) else bool(clip_meta[0]["metric_scale_valid"])
    log(f"[ALIGN] chunk 数={len(clip_meta)}  raw_to_meter f={f_value:.6f}  valid={f_valid}")

    metrics = compare_and_metrics(payload, local, g_idx, f_value, f_valid)
    metrics["frame_alignment"] = {
        "T_R": int(payload["T_R"]), "T_G": len(g_idx),
        "global_frame_indices": g_idx,
        "ego_image_size_hw": payload["image_size_hw"],
        "local_image_size_hw": [local["H"], local["W"]],
        "f_raw_to_meter": f_value, "f_valid": f_valid,
        "n_clips": len(clip_meta),
    }
    (verify_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    # ---- Step 5: 渲染 ----
    K_local = local["intrinsic"]                              # 本地内参 (场景反投影)
    K_ego = tnp(payload["intrinsics_full"])                   # ego 内参 (手部投影)
    g_value = metrics.get("g_value")
    hand_valid = tnp(payload["hand"]["valid"]).astype(bool)   # (T_R,2)
    n_valid_per_frame = hand_valid.sum(axis=1)
    # 选帧: --render-all 渲染全部帧, 否则取手部最有效的 top-k + 均匀分布
    if args.render_all:
        render_frames = list(range(N))
    else:
        k = args.max_render_frames
        top = np.argsort(-n_valid_per_frame)[:max(1, k // 2)]
        even = np.linspace(0, N - 1, max(1, k - len(top))).astype(int)
        render_frames = sorted(set(top.tolist()) | set(even.tolist()))
    log(f"[RENDER] 渲染帧数: {len(render_frames)}")

    if g_value is None:
        log("[RENDER] ⚠️ 无法校准 g (无有效手部点), 跳过渲染")
    else:
        log(f"[RENDER] 本地 gauge 校准因子 g = {g_value:.6f} (hand_local = hand_metric × g)")
        p2d = render_2d_overlay(payload, local, verify_dir, render_frames, K_ego, g_value)
        log(f"[RENDER] 2D 叠加 ×{len(p2d)} → {verify_dir}")
        p3d = render_3d_camera_view(payload, local, verify_dir, render_frames, K_ego, g_value)
        log(f"[RENDER] 3D 同视角渲染 ×{len(p3d)} → {verify_dir}")
        pside = render_side_view(payload, local, verify_dir, render_frames, K_local, g_value)
        log(f"[RENDER] 侧视图 ×{len(pside)} → {verify_dir}")
        pworld = export_world_plys(payload, local, verify_dir, render_frames, g_value)
        log(f"[RENDER] 世界系 PLY ×{len(pworld)} → {verify_dir}")
        # EgoFound3R 自洽视角 (ego metric 深度 + ego metric 手部, 仅 G 锚帧)
        g_sel = sorted(set([0, len(g_idx) // 2, len(g_idx) - 1]))
        pego = render_ego_metric_view(payload, local, g_idx, verify_dir, g_sel)
        log(f"[RENDER] EgoFound3R 自洽视角 ×{len(pego)} → {verify_dir}")
        pcheck = plot_contact_check(payload, local, verify_dir / "contact_check.png",
                                    g_value, K_ego)
        log(f"[RENDER] 一致性检查图 → {pcheck}")

        # ---- 合成视频 / 网格图 ----
        make_montage(p2d, verify_dir / "montage_overlay2d.png", cols=4, tile=(320, 336))
        make_montage(pside, verify_dir / "montage_sideview.png", cols=4, tile=(320, 320))
        if args.render_all:
            for name, plist in (("overlay2d", p2d), ("render3d", p3d), ("sideview", pside)):
                v = compile_frames_to_video(plist, verify_dir / f"{name}.mp4", fps=args.fps)
                if v:
                    log(f"[RENDER] {name}.mp4 → {v}")
            # 与 EgoFound3R 自带 overlay 的左右对比视频
            ego_overlay = ego_out / "overlay.mp4"
            v_local = verify_dir / "overlay2d.mp4"
            if ego_overlay.exists() and v_local.exists():
                cmp_path = verify_dir / "compare_ego_vs_local.mp4"
                rc = subprocess.run([
                    "ffmpeg", "-y", "-v", "error",
                    "-i", str(v_local), "-i", str(ego_overlay),
                    "-filter_complex",
                    "[0:v]scale=384:688[v0];[1:v]scale=384:688[v1];[v0][v1]hstack=inputs=2",
                    "-c:v", "libx264", "-qp", "0", str(cmp_path),
                ]).returncode
                log(f"[RENDER] 左右对比视频 → {cmp_path}" if rc == 0
                    else "[WARN] 对比视频合成失败")

    # ---- 报告 ----
    log("\n" + "=" * 72)
    log("验证报告")
    log("=" * 72)
    rd = metrics.get("raw_depth", {})
    if rd:
        log(f"[1] raw 深度一致 (ego depth_raw vs 本地 depths): "
            f"中位比值={rd['median_ratio_ego_over_local']:.4f} (期望≈1.0), std={rd['ratio_std']:.4f}")
    else:
        log("[1] ego depth_raw 未导出 (需 --save-debug-tensors), 用 [7] 的 metric 反推替代")
    rc = metrics.get("raw_camera", {})
    if rc:
        log(f"[2] raw 相机一致: 旋转差={rc['rotation_angle_deg_median']:.4f}°, "
            f"平移模长比={rc['translation_norm_ratio_median']:.4f} (期望≈1.0)")
    ki = metrics["intrinsics"]
    log(f"[3] 内参: ego(fx={ki['ego_fx_mean']:.1f},fy={ki['ego_fy_mean']:.1f}) vs "
        f"本地(fx={ki['local_fx']:.1f},fy={ki['local_fy']:.1f})  fx相对差={ki['fx_rel_diff']*100:.2f}%")
    log(f"[4] EgoFound3R raw→meter 因子 f = {f_value:.6f} (valid={f_valid}, 仅作用于 ego 自身几何)")
    el = metrics.get("ego_raw_vs_local_raw", {})
    if el:
        log(f"[5] ego raw / 本地 raw 几何比 = {el['median_local_raw_over_ego_raw']:.4f} "
            f"(≈1 则可直乘 f, 明显偏离则两边前向有系统差)")
    hv = metrics.get("hand_vs_local_depth", {})
    if hv.get("n_points"):
        verdict = "✅ 贴合" if abs(hv["median_log_ratio"]) < 0.05 else "⚠️ 偏差"
        log(f"[6] 手部(×g) vs 本地点云深度 (核心): n={hv['n_points']}, "
            f"g={g_value:.4f}, 中位log比={hv['median_log_ratio']:+.4f} ({verdict}), "
            f"MAD={hv['mad_log_ratio']:.4f}, ±5%内={hv['frac_within_5pct']*100:.1f}%, "
            f"±10%内={hv['frac_within_10pct']*100:.1f}%")
    gc_ = metrics.get("gauge_calibration_core", {})
    if gc_:
        log(f"[6b] 稳健核心簇校准: g_core={gc_['g_core_local_raw_per_meter']:.4f} "
            f"(占 {gc_['frac_core']*100:.0f}% 点), 核心残差MAD={gc_['residual_mad_core']:.4f}; "
            f"与 pooled g 的差异体现手指缝隙采样到背景的污染")
    es = metrics.get("ego_self_consistency", {})
    if es:
        log(f"[7] EgoFound3R 自洽 (手 vs 自身metric深度): 中位log比={es['median_log_ratio']:+.4f}, "
            f"±10%内={es['frac_within_10pct']*100:.1f}%")
    log("=" * 72)
    log(f"指标: {verify_dir/'metrics.json'}")
    log(f"2D叠加/3D渲染/世界系PLY: {verify_dir}")


if __name__ == "__main__":
    main()
