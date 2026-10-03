#!/usr/bin/env python3
"""
生成深度图可视化视频:
  - HaWoR mask 轮廓 (白色)
  - 手腕 MANO j0 (青色)
  - mask 中心 (黄色)
  - 手掌中心 vertex 268 (品红)
投影路径: MANO 3D (HaWoR相机空间) → 自动推导的有效焦距投影到HaWoR图像(1920x1080)
            → resize 到 RAS 图像(688x384) — 与 mask 走相同的投影路径
焦距推导: 用 mask 中心像素 + pred_trans 3D 反推 fx, 取中位数"""
import argparse, numpy as np, cv2, os, torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / 'position_retargeting'))
from mano_layer import MANOLayer

H_SID = {0: 'left', 1: 'right'}

def load_ras_depth_files(ras_dir):
    """只加载 RAS 深度图文件列表"""
    depth_dir = Path(ras_dir) / 'depth'
    depth_files = sorted(depth_dir.glob('*.png'), key=lambda x: int(x.stem))
    return depth_files

def derive_focal_length(pred_trans, masks, hidx, n_frames):
    """从 mask 中心像素 + pred_trans 3D 反推有效焦距 fx
    
    对每帧: fx = (mc_y - cy) * tz / ty
    只用 y 方向 (t_x 接近 0 时 fx_x 会爆炸)
    取中位数作为数据集的有效焦距
    """
    cx, cy = 960.0, 540.0
    fx_samples = []
    for fi in range(n_frames):
        t = pred_trans[hidx, fi]
        if np.any(np.isnan(t)) or abs(t[1]) < 0.01:
            continue
        mask = masks[fi] if fi < masks.shape[0] else masks[-1]
        if (mask > 0).sum() < 5:
            continue
        ys, xs = np.where(mask > 0)
        mc_y = float(ys.mean())
        mc_x = float(xs.mean())
        fx_y = (mc_y - cy) * t[2] / t[1]
        fx_samples.append(fx_y)
        if abs(t[0]) > 0.01:
            fx_x = (mc_x - cx) * t[2] / t[0]
            fx_samples.append(fx_x)
    
    fx_arr = np.array(fx_samples)
    valid = (fx_arr > 100) & (fx_arr < 2000)
    if valid.sum() > 0:
        fx = float(np.median(fx_arr[valid]))
        print(f"  自动推导焦距: median fx = {fx:.0f} (from {valid.sum()} samples)")
        print(f"  范围: [{np.min(fx_arr[valid]):.0f}, {np.max(fx_arr[valid]):.0f}]")
        return fx
    return 600.0  # fallback

def project_mano_to_ras(p_cam_mano, fx, cx=960.0, cy=540.0,
                        hawor_w=1920, hawor_h=1080, ras_w=688, ras_h=384):
    """MANO 3D (HaWoR相机空间) → 投影到HaWoR图像 → resize 到 RAS"""
    if p_cam_mano[2] <= 0:
        return None
    u_h = fx * p_cam_mano[0] / p_cam_mano[2] + cx
    v_h = fx * p_cam_mano[1] / p_cam_mano[2] + cy
    u_r = u_h * ras_w / hawor_w
    v_r = v_h * ras_h / hawor_h
    ui, vi = int(round(u_r)), int(round(v_r))
    if 0 <= ui < ras_w and 0 <= vi < ras_h:
        return ui, vi
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--output', default='/tmp/depth_video.mp4')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--hand-idx', type=int, default=None, help='0=left, 1=right, None=auto')
    parser.add_argument('--focal-length', type=float, default=None, help='Override focal (default: 600)')
    args = parser.parse_args()

    hawor_dir = Path(args.hawor_dir)
    ras_dir = Path(args.ras_dir)

    # 1. 加载 HaWoR 数据
    print("加载 HaWoR ...")
    rec_dir = hawor_dir / 'reconstruction'
    npz_files = sorted(rec_dir.glob('hawor_results_*.npz'))
    npz_files = [f for f in npz_files if '_depth_aligned' not in str(f) and '_aligned' not in str(f)]
    if not npz_files:
        npz_files = sorted(rec_dir.glob('*.npz'))
    d = dict(np.load(str(npz_files[0]), allow_pickle=True))
    pred_trans = d['pred_trans']
    pred_valid = d['pred_valid']
    n_frames = pred_trans.shape[1]
    print(f"  HaWoR: {n_frames} 帧, pred_trans shape: {pred_trans.shape}")

    # 自动选择手
    if args.hand_idx is not None:
        hidx = args.hand_idx
    else:
        hidx = 1 if pred_valid.shape[0] > 1 and pred_valid[1].sum() > pred_valid[0].sum() else 0
    side = H_SID[hidx]
    print(f"  使用手: {side} (hidx={hidx})")

    # 2. 加载 MANO 模型
    betas_arr = d['pred_betas'][hidx]
    first_valid = next((i for i in range(n_frames) if not np.any(np.isnan(betas_arr[i]))), None)
    if first_valid is None:
        raise ValueError("No valid beta found")
    ml = MANOLayer(side, betas_arr[first_valid].astype(np.float32))

    # 3. 加载 HaWoR mask
    tracks_dirs = sorted(hawor_dir.glob('tracks_*'))
    if not tracks_dirs:
        raise FileNotFoundError("No tracks directory found")
    mask_file = tracks_dirs[0] / 'model_masks.npy'
    if not mask_file.exists():
        raise FileNotFoundError(f"Mask not found: {mask_file}")
    masks = np.load(str(mask_file), allow_pickle=True)
    if masks.ndim == 0: masks = masks.item()
    if masks.ndim > 3:
        if masks.shape[0] == 1: masks = masks[0]
    print(f"  Masks shape: {masks.shape}")

    # 4. 自动推导焦距 (从 mask + pred_trans 反推)
    print("推导焦距 ...")
    if args.focal_length is not None:
        fx = args.focal_length
        print(f"  使用指定焦距: fx = {fx}")
    else:
        fx = derive_focal_length(pred_trans, masks, hidx, n_frames)
    print(f"  使用焦距: fx = {fx}")

    # 5. 加载 RAS 深度图文件
    print("加载 RAS 深度 ...")
    depth_files = load_ras_depth_files(ras_dir)
    n_ras = len(depth_files)
    ras_h, ras_w = 384, 688
    print(f"  RAS: {n_ras} 帧, 分辨率 {ras_w}x{ras_h}")

    # 6. 统计跟踪
    stats = {'wrist_in': 0, 'wrist_oob': 0, 'v268_in': 0, 'v268_oob': 0, 'nan': 0}

    # 7. 生成视频帧
    print("生成视频帧 ...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(args.output), fourcc, args.fps, (ras_w, ras_h))
    if not out.isOpened():
        out = cv2.VideoWriter(str(args.output), fourcc, args.fps, (ras_w, ras_h), isColor=True)

    depth_cache = {}
    frame_count = 0
    for ri in range(n_ras):
        # 映射到 HaWoR 帧
        hi = round(ri * (n_frames - 1) / (n_ras - 1)) if n_ras > 1 else 0
        hi = min(hi, n_frames - 1)

        # 读取深度图
        if ri not in depth_cache:
            depth_img = cv2.imread(str(depth_files[ri]), cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                continue
            depth_cache[ri] = depth_img
        else:
            depth_img = depth_cache[ri]

        # 彩色深度图
        depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # HaWoR mask → resize to RAS
        mask = masks[hi] if hi < masks.shape[0] else masks[-1]
        mask_ras = cv2.resize(mask.astype(np.uint8), (ras_w, ras_h), interpolation=cv2.INTER_NEAREST)

        # mask 轮廓 (白色)
        contours, _ = cv2.findContours(mask_ras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_color, contours, -1, (255, 255, 255), 1)

        # mask 中心 (黄色)
        ys, xs = np.where(mask_ras > 0)
        if len(xs) > 0:
            mc_x, mc_y = int(xs.mean()), int(ys.mean())
            cv2.circle(depth_color, (mc_x, mc_y), 4, (0, 255, 255), -1)
            cv2.putText(depth_color, f'MC({mc_x},{mc_y})', (mc_x+5, mc_y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # 计算 MANO 点
        trans = pred_trans[hidx, hi].astype(np.float32)
        rot = d['pred_rot'][hidx, hi].astype(np.float32)
        hp = d['pred_hand_pose'][hidx, hi].astype(np.float32)
        valid_frame = not (np.any(np.isnan(trans)) or np.all(np.abs(trans) < 1e-6) or
                          np.any(np.isnan(rot)) or np.any(np.isnan(hp)))

        if valid_frame:
            p_t = torch.from_numpy(np.concatenate([rot, hp])).unsqueeze(0)
            t_t = torch.from_numpy(trans).unsqueeze(0)
            with torch.no_grad():
                v, j = ml(p_t, t_t)
            jv = j[0].numpy()
            vv = v[0].numpy()

            # 手腕 j0 (青色)
            pt = project_mano_to_ras(jv[0], fx)
            if pt is not None:
                ui, vi = pt
                stats['wrist_in'] += 1
                cv2.circle(depth_color, (ui, vi), 5, (255, 255, 0), -1)
                cv2.putText(depth_color, f'Wrist({ui},{vi})', (ui+5, vi-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                z_val = depth_img[vi, ui].astype(float) / 1000.0
                cv2.putText(depth_color, f'Z={z_val:.3f}m', (ui+5, vi+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            else:
                stats['wrist_oob'] += 1

            # 手掌中心 vertex 268 (品红)
            pt = project_mano_to_ras(vv[268], fx)
            if pt is not None:
                ui, vi = pt
                stats['v268_in'] += 1
                cv2.circle(depth_color, (ui, vi), 5, (255, 0, 255), -1)
                cv2.putText(depth_color, f'v268({ui},{vi})', (ui+5, vi-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                z_val = depth_img[vi, ui].astype(float) / 1000.0
                cv2.putText(depth_color, f'Z={z_val:.3f}m', (ui+5, vi+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
            else:
                stats['v268_oob'] += 1
        else:
            stats['nan'] += 1

        # 帧信息
        cv2.putText(depth_color, f'HaWoR Frame {hi} / RAS Frame {ri}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(depth_color, f'Hand: {side} (hidx={hidx})', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 图例
        cv2.putText(depth_color, 'WHITE=mask', (10, ras_h-55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(depth_color, 'YELLOW=center', (10, ras_h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        cv2.putText(depth_color, 'CYAN=wrist', (10, ras_h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        cv2.putText(depth_color, 'MAGENTA=mano268', (10, ras_h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        out.write(depth_color)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧...")

    out.release()
    print(f"\n视频已保存: {args.output} ({frame_count} 帧, {args.fps} fps)")
    print(f"统计: wrist_in={stats['wrist_in']}, wrist_oob={stats['wrist_oob']}, "
          f"v268_in={stats['v268_in']}, v268_oob={stats['v268_oob']}, nan={stats['nan']}")

if __name__ == '__main__':
    main()