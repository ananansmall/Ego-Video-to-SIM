#!/usr/bin/env python3
"""
可视化 HaWoR mask 在 RAS 深度图上的投影 → GLB 3D 点云
对比: RAS 深度图点云 vs HaWoR 手腕轨迹
"""
import argparse, numpy as np, cv2, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_ras_depth_dir(ras_dir):
    depth_dir = os.path.join(ras_dir, 'depth')
    return sorted(
        [os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')],
        key=lambda x: int(os.path.basename(x).split('.')[0]))

def load_ras_intrinsics(ras_dir):
    K = np.loadtxt(os.path.join(ras_dir, 'intrinsic.txt'))
    if K.shape == (3, 4): K = K[:, :3]
    elif K.size == 9: K = K.reshape(3, 3)
    return K

def load_ras_extrinsics(ras_dir):
    ext_dir = os.path.join(ras_dir, 'extrinsics')
    files = sorted(Path(ext_dir).glob('*.txt'), key=lambda x: int(x.stem))
    R_c2w, t_c2w = [], []
    for f in files:
        ext = np.loadtxt(str(f))
        if ext.shape == (3, 4): ext = np.vstack([ext, [0,0,0,1]])
        Rw2c, tw2c = ext[:3,:3], ext[:3,3]
        Rc2w = Rw2c.T
        R_c2w.append(Rc2w)
        t_c2w.append(-Rc2w @ tw2c)
    return np.array(R_c2w), np.array(t_c2w)

def unproject_depth_to_glb(u, v, z_val, K, R_c2w, t_c2w, depth_scale=1000.0):
    """单个像素反投影到 GLB"""
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    z_m = z_val / depth_scale
    x_cam = (u - cx) * z_m / fx
    y_cam = (v - cy) * z_m / fy
    p_cam = np.array([x_cam, y_cam, z_m])
    return R_c2w @ p_cam + t_c2w

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--transform-params', required=True)
    parser.add_argument('--frame', type=int, default=0, help='要显示的帧号')
    parser.add_argument('--output', default='/tmp/mask_depth_vis')
    parser.add_argument('--depth-scale', type=float, default=1000.0)
    args = parser.parse_args()

    # 1. 加载 transform_params
    print("加载 transform_params ...")
    tp = dict(np.load(args.transform_params))
    s = float(tp.get('scale_ratio', tp.get('scale', 1.0)))
    R_h2g = tp.get('R_hand_to_glb', tp.get('R', np.eye(3)))
    t_h2g = tp.get('t_hand_to_glb', tp.get('t', np.zeros(3)))
    Rx_hand = np.diag([1, -1, -1])
    R_hand = R_h2g @ Rx_hand

    # 2. 加载 HaWoR
    print("加载 HaWoR ...")
    rec_dir = Path(args.hawor_dir) / 'reconstruction'
    npz_file = sorted(rec_dir.glob('hawor_results_*.npz'))
    if not npz_file: raise FileNotFoundError(f"no npz in {rec_dir}")
    # 排除 _depth_aligned
    npz_file = [f for f in npz_file if '_depth_aligned' not in str(f)]
    d = dict(np.load(str(npz_file[0]), allow_pickle=True))
    pred_trans = d['pred_trans']
    pred_valid = d['pred_valid']
    n_frames = pred_trans.shape[1]

    # 3. 加载 masks
    print("加载 masks ...")
    tracks_dir = sorted(Path(args.hawor_dir).glob('tracks_*'))
    mask_file = tracks_dir[0] / 'model_masks.npy' if tracks_dir else None
    if not mask_file or not mask_file.exists():
        raise FileNotFoundError(f"masks not found")
    masks = np.load(str(mask_file), allow_pickle=True)
    if masks.ndim == 0: masks = masks.item()
    if masks.ndim > 3:
        if masks.shape[0] == 1: masks = masks[0]
    print(f"  masks shape: {masks.shape}")

    # 4. 加载 RAS
    print("加载 RAS ...")
    depth_files = load_ras_depth_dir(args.ras_dir)
    K = load_ras_intrinsics(args.ras_dir)
    R_c2w, t_c2w = load_ras_extrinsics(args.ras_dir)
    n_ras = len(depth_files)
    print(f"  RAS: {n_ras} frames, depth={depth_files[0]}")

    # 找到最近的 RAS 帧
    ras_idx = round(args.frame * (n_ras - 1) / (n_frames - 1))
    ras_idx = min(ras_idx, n_ras - 1)

    # 5. 加载该帧深度图
    depth_img = cv2.imread(depth_files[ras_idx], cv2.IMREAD_UNCHANGED)
    print(f"  RAS 深度图: {depth_img.shape}, dtype={depth_img.dtype}")

    # 6. 获取 mask 并 resize 到深度图分辨率
    mask = masks[args.frame]
    if mask.shape[:2] != depth_img.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (depth_img.shape[1], depth_img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 0

    # 7. 提取深度图 mask 区域的像素
    ys, xs = np.where(mask_bool)
    z_vals = depth_img[ys, xs].astype(np.float32) / args.depth_scale
    valid = z_vals > 0
    xs, ys, z_vals = xs[valid], ys[valid], z_vals[valid]
    print(f"  mask 有效像素: {len(xs)}/{mask_bool.sum()}")

    # 8. 反投影到 GLB
    pts_glb = []
    for u, v, z in zip(xs, ys, z_vals):
        p = unproject_depth_to_glb(u, v, z, K, R_c2w[ras_idx], t_c2w[ras_idx], args.depth_scale)
        pts_glb.append(p)
    pts_glb = np.array(pts_glb)
    print(f"  GLB 点云: {pts_glb.shape}, Z=[{pts_glb[:,2].min():.3f},{pts_glb[:,2].max():.3f}]")

    # 9. HaWoR 手腕在 GLB 空间
    # 自动选择有效手
    hand_idx = 0
    for h in range(pred_trans.shape[0]):
        if pred_valid[h, args.frame] and not np.any(np.isnan(pred_trans[h, args.frame])):
            hand_idx = h
            break
    p_hawor = pred_trans[hand_idx, args.frame]
    p_hawor_glb = s * R_hand @ p_hawor + t_h2g
    print(f"  HaWoR 手{['左','右'][hand_idx]} 手腕 GLB: {p_hawor_glb}")

    # 10. 可视化
    fig = plt.figure(figsize=(16, 7))

    # 左: 深度图 + mask 叠加
    ax1 = fig.add_subplot(121)
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    ax1.imshow(depth_norm, cmap='viridis')
    # 叠加 mask 轮廓
    mask_contour = cv2.dilate(mask.astype(np.uint8), None, iterations=2) - mask
    mask_contour = cv2.dilate(mask_contour, None, iterations=1)
    ax1.imshow(mask_contour, cmap='Reds', alpha=0.5)
    ax1.set_title(f'RAS 深度图 + HaWoR mask 帧{args.frame} (RAS帧{ras_idx})')
    ax1.text(10, 30, f'手{["左","右"][hand_idx]} mask 像素: {len(xs)}', color='white',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    # 深度值统计
    if len(z_vals) > 0:
        ax1.text(10, depth_img.shape[0]-20,
                 f'深度: median={np.median(z_vals):.3f}m, range=[{z_vals.min():.3f},{z_vals.max():.3f}]m',
                 color='white', fontsize=9, bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # 右: GLB 3D 点云
    ax2 = fig.add_subplot(122, projection='3d')
    # RAS 点云
    if len(pts_glb) > 0:
        # 按距离相机远近着色
        cam_pos = t_c2w[ras_idx]
        dists = np.linalg.norm(pts_glb - cam_pos, axis=1)
        sc = ax2.scatter(pts_glb[:,0], pts_glb[:,1], pts_glb[:,2],
                        c=dists, cmap='viridis', s=1.0, alpha=0.6)
        plt.colorbar(sc, ax=ax2, label='RAS dist to cam')
    # HaWoR 手腕位置
    ax2.scatter(*p_hawor_glb, c='red', s=80, marker='o', label=f'HaWoR 手腕 (手{["左","右"][hand_idx]})')
    # RAS 相机位置
    ax2.scatter(*cam_pos, c='orange', s=60, marker='^', label='RAS 相机')

    # 坐标轴等比例
    all_pts = [pts_glb] if len(pts_glb) > 0 else []
    all_pts.append(p_hawor_glb.reshape(1,3))
    all_pts.append(cam_pos.reshape(1,3))
    ag = np.concatenate(all_pts, axis=0)
    lo, hi = ag.min(0), ag.max(0)
    rng = max(np.maximum(hi-lo, 0.5))
    pad = rng * 0.15
    half = rng/2 + pad
    center = (lo + hi).mean() / 2
    lim_lo, lim_hi = -half, half
    ax2.set_xlim3d(lim_lo, lim_hi)
    ax2.set_ylim3d(lim_lo, lim_hi)
    ax2.set_zlim3d(lim_lo, lim_hi)
    ax2.set_box_aspect([1,1,1])
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title(f'GLB 空间: RAS 点云 vs HaWoR 手腕')
    ax2.legend(fontsize=9)

    # 坐标轴
    for c, lbl, v in [('red','X',[half*0.5,0,0]),('green','Y',[0,half*0.5,0]),('blue','Z',[0,0,half*0.5])]:
        ax2.plot([0,v[0]],[0,v[1]],[0,v[2]], color=c, lw=2)
        ax2.text(v[0],v[1],v[2], lbl, color=c, fontsize=12, fontweight='bold')

    plt.tight_layout()
    png = f'{args.output}_frame{args.frame}.png'
    plt.savefig(png, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {png}")
    plt.close()

    # 输出数值对比
    print(f"\n{'='*50}")
    print(f"帧{args.frame} 对比")
    print(f"{'='*50}")
    print(f"HaWoR 手腕 GLB: {p_hawor_glb}")
    if len(pts_glb) > 0:
        ras_center = pts_glb.mean(0)
        print(f"RAS 点云中心 GLB: {ras_center}")
        print(f"RAS 点云范围 Z: [{pts_glb[:,2].min():.3f}, {pts_glb[:,2].max():.3f}]")
        print(f"距离: {np.linalg.norm(p_hawor_glb - ras_center):.3f}m")
        print(f"RAS 手腕投影帧: {ras_idx} (RAS帧率)")

if __name__ == '__main__':
    main()