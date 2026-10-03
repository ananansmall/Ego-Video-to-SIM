#!/usr/bin/env python3
"""测试: HaWoR 手腕在深度图上的投影 → 与 mask 走同一 pipeline"""
import argparse, numpy as np, cv2, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--output', default='/tmp/wrist_depth_test')
    parser.add_argument('--depth-scale', type=float, default=1000.0)
    args = parser.parse_args()

    hawor_dir = Path(args.hawor_dir)
    ras_dir = Path(args.ras_dir)

    # 1. 加载 HaWoR
    rec_dir = hawor_dir / 'reconstruction'
    npz_file = sorted(rec_dir.glob('hawor_results_*.npz'))
    npz_file = [f for f in npz_file if '_depth_aligned' not in str(f)]
    d = dict(np.load(str(npz_file[0]), allow_pickle=True))
    pred_trans, pred_valid = d['pred_trans'], d['pred_valid']
    R_c2w, t_c2w = d['R_c2w'], d['t_c2w']
    img_focal = float(d['img_focal'])
    n_frames = pred_trans.shape[1]
    H_h, W_h = 1080, 1920  # HaWoR image size
    cx_h, cy_h = W_h / 2, H_h / 2

    # 2. 加载 masks
    tracks_dir = sorted(hawor_dir.glob('tracks_*'))
    masks = np.load(str(tracks_dir[0] / 'model_masks.npy'), allow_pickle=True)
    if masks.ndim == 0: masks = masks.item()
    if masks.ndim > 3: masks = masks[0] if masks.shape[0] == 1 else masks
    print(f"Masks shape: {masks.shape}")

    # 3. 加载 RAS
    depth_dir = ras_dir / 'depth'
    depth_files = sorted(depth_dir.glob('*.png'), key=lambda x: int(x.stem))
    K = np.loadtxt(str(ras_dir / 'intrinsic.txt'))
    if K.shape == (3, 4): K = K[:, :3]
    elif K.size == 9: K = K.reshape(3, 3)
    n_ras = len(depth_files)
    print(f"RAS depth: {n_ras} frames, K={K}")
    H_r, W_r = 384, 688  # RAS depth image size

    # 4. 选择有效手 (右手, 121 数据集右手是 idx=1)
    hand_idx = 1  # 右手
    if not pred_valid[hand_idx, args.frame]:
        # 尝试左手
        hand_idx = 0
        if not pred_valid[hand_idx, args.frame]:
            print("No valid hand!")
            return

    # 5. 计算手腕在 HaWoR 图像中的 2D 位置
    fi = args.frame
    p_world = pred_trans[hand_idx, fi]
    R_w2c = R_c2w[fi].T  # world → camera
    t_w2c = -R_w2c @ t_c2w[fi]
    p_cam = R_w2c @ p_world + t_w2c  # camera coordinates

    # OpenGL 投影: HaWoR 相机看向 -z
    z_neg = -p_cam[2]  # OpenGL: 用 -z 做除法
    if z_neg > 0:
        u_h = int(round(cx_h + img_focal * p_cam[0] / z_neg))
        v_h = int(round(cy_h + img_focal * p_cam[1] / z_neg))
    else:
        u_h, v_h = -1, -1

    print(f"Frame {fi}: 手{hand_idx} 世界坐标={p_world}")
    print(f"  相机坐标={p_cam}, z_neg={z_neg}")
    print(f"  手腕预测: u={u_h}, v={v_h} (HaWoR 图像 {W_h}x{H_h})")

    # 6. 加载该帧 mask 并检查手腕是否在 mask 内
    mask_h = masks[fi]
    in_mask_h = False
    if 0 <= u_h < W_h and 0 <= v_h < H_h:
        in_mask_h = mask_h[v_h, u_h] > 0
        print(f"  在 HaWoR mask 内: {in_mask_h}")

    # 7. 映射到 RAS 深度图尺寸
    scale_x = W_r / W_h  # 688/1920
    scale_y = H_r / H_h  # 384/1080
    u_r = int(round(u_h * scale_x))
    v_r = int(round(v_h * scale_y))
    u_r = np.clip(u_r, 0, W_r - 1)
    v_r = np.clip(v_r, 0, H_r - 1)

    # 8. 读取 RAS 深度
    ras_idx = round(fi * (n_ras - 1) / max(n_frames - 1, 1))
    ras_idx = min(ras_idx, n_ras - 1)
    depth_img = cv2.imread(str(depth_files[ras_idx]), cv2.IMREAD_UNCHANGED)
    depth_val = depth_img[v_r, u_r].astype(float) / args.depth_scale
    valid_depth = depth_val > 0 and np.isfinite(depth_val)

    print(f"  RAS 映射: u={u_r}, v={v_r} (深度图 {W_r}x{H_r})")
    print(f"  RAS 深度值: {depth_val:.3f}m {'✓' if valid_depth else '✗'}")

    # 9. 加载 mask 在 RAS 深度图上的投影
    mask_r = cv2.resize(mask_h.astype(np.uint8), (W_r, H_r), interpolation=cv2.INTER_NEAREST)
    in_mask_r = mask_r[v_r, u_r] > 0 if 0 <= u_r < W_r and 0 <= v_r < H_r else False
    print(f"  在 RAS mask 内: {in_mask_r}")

    # 10. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 左: HaWoR mask + 手腕投影
    ax = axes[0]
    ax.imshow(mask_h, cmap='gray', alpha=0.8)
    if 0 <= u_h < W_h and 0 <= v_h < H_h:
        ax.plot(u_h, v_h, 'r.', markersize=12, label='手腕')
    # mask 中心
    ys, xs = np.where(mask_h > 0)
    if len(xs) > 0:
        ax.plot(xs.mean(), ys.mean(), 'g.', markersize=10, label='mask中心')
    ax.set_title(f'HaWoR mask 帧{fi}')
    ax.legend(fontsize=8)
    ax.set_xlim(0, W_h); ax.set_ylim(H_h, 0)

    # 中: RAS 深度图 + mask 轮廓 + 手腕投影
    ax = axes[1]
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    ax.imshow(depth_norm, cmap='viridis')
    # mask 轮廓
    mask_contour = cv2.dilate(mask_r.astype(np.uint8), None, iterations=2) - mask_r
    mask_contour = cv2.dilate(mask_contour, None, iterations=1)
    ax.imshow(mask_contour, cmap='Reds', alpha=0.5)
    # 手腕点
    ax.plot(u_r, v_r, 'r.', markersize=12, label='手腕')
    # mask 中心
    ys_r, xs_r = np.where(mask_r > 0)
    if len(xs_r) > 0:
        ax.plot(xs_r.mean(), ys_r.mean(), 'g.', markersize=10, label='mask中心')
    ax.set_title(f'RAS 深度图 帧{ras_idx}')
    ax.legend(fontsize=8)
    ax.set_xlim(0, W_r); ax.set_ylim(H_r, 0)

    # 右: 对比分析
    ax = axes[2]
    ax.axis('off')
    info = [
        f"HaWoR 帧 {fi} | RAS 帧 {ras_idx}",
        f"手: {'左' if hand_idx==0 else '右'}",
        f"手腕 2D (HaWoR): ({u_h}, {v_h})",
        f"手腕 in HaWoR mask: {in_mask_h}",
        f"手腕 2D (RAS): ({u_r}, {v_r})",
        f"手腕 in RAS mask: {in_mask_r}",
        f"手腕 RAS 深度: {depth_val:.3f}m" if valid_depth else "手腕 RAS 深度: 无效",
        f"focal: {img_focal}",
        f"scale_x: {scale_x:.4f}, scale_y: {scale_y:.4f}",
    ]
    # 从 mask 区域读深度统计
    if len(xs_r) > 0:
        z_vals = depth_img[ys_r, xs_r].astype(float) / args.depth_scale
        valid_z = z_vals > 0
        if valid_z.any():
            info.append(f"mask 深度: median={np.median(z_vals[valid_z]):.3f}m")
            info.append(f"mask 深度范围: [{z_vals[valid_z].min():.3f}, {z_vals[valid_z].max():.3f}]m")
    for i, line in enumerate(info):
        ax.text(0.05, 0.95 - i*0.08, line, fontsize=11, fontfamily='monospace',
                transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    png = f'{args.output}_frame{fi}.png'
    fig.savefig(png, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {png}")
    plt.close()

    # 输出总结
    print(f"\n{'='*60}")
    print(f"总结: 帧{fi} 手{['左','右'][hand_idx]}")
    print(f"{'='*60}")
    print(f"  手腕 HaWoR 2D: ({u_h}, {v_h}) {'✓在mask内' if in_mask_h else '✗不在mask内'}")
    print(f"  手腕 RAS 2D: ({u_r}, {v_r}) {'✓在mask内' if in_mask_r else '✗不在mask内'}")
    print(f"  手腕深度: {depth_val:.3f}m")
    if len(xs_r) > 0:
        z_vals = depth_img[ys_r, xs_r].astype(float) / args.depth_scale
        valid_z = z_vals > 0
        if valid_z.any():
            print(f"  mask 深度中值: {np.median(z_vals[valid_z]):.3f}m")
            print(f"  深度差 (手腕-mask): {depth_val - np.median(z_vals[valid_z]):.3f}m")

if __name__ == '__main__':
    main()