#!/usr/bin/env python3
"""可视化 MANO vertex 268 在 RAS 深度图上的投影位置"""
import argparse, numpy as np, cv2, os, torch
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加 MANO 路径
import sys
_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE.parent / 'position_retargeting'))
from mano_layer import MANOLayer

H_SID = {0: 'left', 1: 'right'}

def load_ras_depth_data(ras_dir):
    depth_dir = os.path.join(ras_dir, 'depth')
    files = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith('.png')],
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
    K = np.loadtxt(os.path.join(ras_dir, 'intrinsic.txt'))
    if K.shape == (3, 4): K = K[:, :3]
    elif K.size == 9: K = K.reshape(3, 3)
    ext_dir = Path(ras_dir)/'extrinsics'
    ext_files = sorted(ext_dir.glob('*.txt'), key=lambda x: int(x.stem))
    R_c2w, t_c2w = [], []
    for f in ext_files:
        ext = np.loadtxt(str(f))
        if ext.shape == (3, 4): ext = np.vstack([ext, [0,0,0,1]])
        Rw2c, tw2c = ext[:3,:3], ext[:3,3]
        Rc2w = Rw2c.T
        R_c2w.append(Rc2w)
        t_c2w.append(-Rc2w @ tw2c)
    return K, np.array(R_c2w), np.array(t_c2w), files

def compute_palm_vertex(d, hawor_dir, hidx=1):
    """计算 MANO vertex 268 的世界坐标"""
    n = len(d['pred_trans'][hidx])
    betas_arr = d['pred_betas'][hidx]
    first_valid = None
    for fi in range(n):
        if not np.any(np.isnan(betas_arr[fi])):
            first_valid = fi
            break
    if first_valid is None:
        return np.full((n, 3), np.nan)
    betas = betas_arr[first_valid].astype(np.float32)
    ml = MANOLayer(side='right' if hidx == 1 else 'left', betas=betas)
    palms = np.full((n, 3), np.nan, dtype=np.float64)
    for fi in range(n):
        trans = d['pred_trans'][hidx, fi].astype(np.float32)
        rot = d['pred_rot'][hidx, fi].astype(np.float32)
        hp = d['pred_hand_pose'][hidx, fi].astype(np.float32)
        if np.any(np.isnan(trans)) or np.all(np.abs(trans) < 1e-6):
            continue
        if np.any(np.isnan(rot)) or np.any(np.isnan(hp)):
            continue
        p_t = torch.from_numpy(np.concatenate([rot, hp])).unsqueeze(0)
        t_t = torch.from_numpy(trans).unsqueeze(0)
        with torch.no_grad():
            v, _ = ml(p_t, t_t)
        vv = v[0].numpy()
        if not np.any(np.isnan(vv)):
            palms[fi] = vv[268]
    return palms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--output', default='/tmp/palm_proj_viz')
    args = parser.parse_args()

    # 加载数据
    print("加载数据 ...")
    d = dict(np.load(str(Path(args.hawor_dir)/'reconstruction'/'hawor_results_0_600.npz'), allow_pickle=True))
    par = dict(np.load(Path(args.hawor_dir)/'output/alignment/transform_params.npz'))
    s = float(par['scale_ratio'])
    R = par['R_hand_to_glb']
    t = par['t_hand_to_glb']
    Rx_hand = np.diag([1, -1, -1])
    R_hand = R @ Rx_hand
    K_ras, R_c2w, t_c2w, depth_files = load_ras_depth_data(args.ras_dir)
    n_ras = len(depth_files)

    # 计算手掌顶点
    palms = compute_palm_vertex(d, args.hawor_dir, hidx=1)
    n_hawor = len(palms)

    # 选择几个关键帧
    sel_frames = [0, 50, 100, 150, 200]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, fi in enumerate(sel_frames):
        if idx >= len(axes): break
        ax = axes[idx]
        p_world = palms[fi]
        if np.any(np.isnan(p_world)):
            ax.text(0.5, 0.5, f'Frame {fi}: NaN', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Frame {fi}')
            continue

        # 变换到 GLB
        palm_glb = s * R_hand @ p_world + t

        # 投影到 RAS 相机
        ri = round(fi * (n_ras - 1) / max(n_hawor - 1, 1))
        ri = min(ri, n_ras - 1)
        R_w2c = R_c2w[ri].T
        t_w2c = -R_w2c @ t_c2w[ri]
        p_cam = R_w2c @ palm_glb + t_w2c
        z_cam = p_cam[2]

        # 读取深度图
        depth_img = cv2.imread(depth_files[ri], cv2.IMREAD_UNCHANGED)
        depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)

        # 显示深度图
        ax.imshow(cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB))

        if z_cam > 0:
            u = K_ras[0,0] * p_cam[0] / z_cam + K_ras[0,2]
            v = K_ras[1,1] * p_cam[1] / z_cam + K_ras[1,2]
            ui, vi = int(round(u)), int(round(v))
            in_bounds = 0 <= vi < depth_img.shape[0] and 0 <= ui < depth_img.shape[1]
            if in_bounds:
                z_depth = depth_img[vi, ui].astype(float) / 1000.0
                # 画点
                ax.plot(ui, vi, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                ax.text(ui+5, vi-5, f'v268\nz={z_depth:.3f}m', color='red', fontsize=9,
                        fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))
                # 标注深度值
                info = f'RAS z={z_depth:.3f}m | MANO z={p_world[2]:.3f}m'
                ax.text(10, 30, info, color='white', fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.5))
            else:
                ax.text(10, 30, f'OOB (u={ui:.0f}, v={vi:.0f})', color='white', fontsize=9,
                        bbox=dict(facecolor='red', alpha=0.7))
                # 画出界箭头
                ax.plot(min(max(ui, 0), depth_img.shape[1]-1),
                        min(max(vi, 0), depth_img.shape[0]-1), 'rx', markersize=12, markeredgewidth=2)
        else:
            ax.text(10, 30, f'z_cam={z_cam:.3f} <= 0', color='white', fontsize=9,
                    bbox=dict(facecolor='red', alpha=0.7))

        ax.set_title(f'Frame {fi} (RAS帧 {ri}) | MANO z={p_world[2]:.3f}m')
        ax.axis('off')

    # 隐藏多余的子图
    for idx in range(len(sel_frames), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'MANO Vertex 268 投影到 RAS 深度图 | {args.hawor_dir.split("/")[-1]}', fontsize=14)
    plt.tight_layout()
    png = f'{args.output}_depth_overlay.png'
    plt.savefig(png, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {png}")
    plt.close()

    # ===== 第二张图: 3D 轨迹对比 =====
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')

    # 计算两个轨迹
    palms_glb_traj = []
    palms_dpj_traj = []
    for fi in range(n_hawor):
        p_world = palms[fi]
        if np.any(np.isnan(p_world)): continue
        # Palm Raw
        palm_glb = s * R_hand @ p_world + t
        palms_glb_traj.append(palm_glb)
        # Palm Dep Proj
        ri = round(fi * (n_ras - 1) / max(n_hawor - 1, 1))
        ri = min(ri, n_ras - 1)
        R_w2c = R_c2w[ri].T
        t_w2c = -R_w2c @ t_c2w[ri]
        p_cam = R_w2c @ palm_glb + t_w2c
        z_cam = p_cam[2]
        if z_cam <= 0: continue
        u = K_ras[0,0] * p_cam[0] / z_cam + K_ras[0,2]
        v = K_ras[1,1] * p_cam[1] / z_cam + K_ras[1,2]
        ui, vi = int(round(u)), int(round(v))
        depth_img = cv2.imread(depth_files[ri], cv2.IMREAD_UNCHANGED)
        if depth_img is None: continue
        if vi < 0 or vi >= depth_img.shape[0] or ui < 0 or ui >= depth_img.shape[1]: continue
        z_depth = depth_img[vi, ui].astype(float) / 1000.0
        if z_depth <= 0 or not np.isfinite(z_depth): continue
        p_cam_dpj = np.array([(ui - K_ras[0,2]) * z_depth / K_ras[0,0],
                              (vi - K_ras[1,2]) * z_depth / K_ras[1,1], z_depth])
        p_glb_dpj = R_c2w[ri] @ p_cam_dpj + t_c2w[ri]
        palms_dpj_traj.append(p_glb_dpj)

    palms_glb_traj = np.array(palms_glb_traj)
    palms_dpj_traj = np.array(palms_dpj_traj)

    # 绘制轨迹
    ax2.plot(palms_glb_traj[:,0], palms_glb_traj[:,1], palms_glb_traj[:,2],
             'coral', lw=1.5, label=f'Palm Raw ({len(palms_glb_traj)} fr)')
    ax2.plot(palms_dpj_traj[:,0], palms_dpj_traj[:,1], palms_dpj_traj[:,2],
             'darkred', lw=1.5, label=f'Palm Dep Proj ({len(palms_dpj_traj)} fr)')

    # 画起点和终点
    ax2.scatter(*palms_glb_traj[0], c='orange', s=80, marker='o', edgecolors='black')
    ax2.scatter(*palms_dpj_traj[0], c='red', s=80, marker='o', edgecolors='black')

    # 坐标轴
    for c, lbl, v in [('red','X',[0.5,0,0]),('green','Y',[0,0.5,0]),('blue','Z',[0,0,0.5])]:
        ax2.plot([0,v[0]],[0,v[1]],[0,v[2]], color=c, lw=2)
        ax2.text(v[0],v[1],v[2], lbl, color=c, fontsize=12, fontweight='bold')

    # 自动范围
    all_pts = [palms_glb_traj, palms_dpj_traj]
    ag = np.concatenate(all_pts, axis=0)
    lo, hi = ag.min(0), ag.max(0)
    rng = max(np.maximum(hi-lo, 0.5))
    pad = rng * 0.15
    half = rng/2 + pad
    lim_lo, lim_hi = -half, half
    ax2.set_xlim3d(lim_lo, lim_hi)
    ax2.set_ylim3d(lim_lo, lim_hi)
    ax2.set_zlim3d(lim_lo, lim_hi)
    ax2.set_box_aspect([1,1,1])
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    ax2.set_title('Palm Raw vs Palm Dep Proj 3D 轨迹对比')
    ax2.legend()

    png2 = f'{args.output}_3d_trajectory.png'
    plt.savefig(png2, dpi=150, bbox_inches='tight')
    print(f"Saved: {png2}")
    plt.close()

    print(f"\nPalm Raw: {len(palms_glb_traj)} fr, Z=[{palms_glb_traj[:,2].min():.3f},{palms_glb_traj[:,2].max():.3f}]")
    print(f"Palm Dep Proj: {len(palms_dpj_traj)} fr, Z=[{palms_dpj_traj[:,2].min():.3f},{palms_dpj_traj[:,2].max():.3f}]")

if __name__ == '__main__':
    main()