#!/usr/bin/env python3
"""batch_vis_3d.py — 批量生成四个例子的 3D 轨迹对比图 (RAS vs Default vs SVD)"""
import numpy as np, glob, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EXAMPLES = {
    '2chair-bottle': {
        'ras': '/home/an/data/ras/2chair-bottle_vggt_omega',
        'hawor': '/home/an/data/hawor/2chair-bottle',
        'default': './output/alignment_2chair_default/transform_params.npz',
        'svd': './output/alignment_2chair_svd/transform_params.npz'},
    '121': {
        'ras': '/home/an/data/ras/121_C5_CellPhone_161deg_vggt_omega',
        'hawor': '/home/an/data/hawor/121_C5_CellPhone_161deg',
        'default': '/home/an/data/hawor/121_C5_CellPhone_161deg/output/alignment/transform_params.npz',
        'svd': './output/alignment_121_svd/transform_params.npz'},
    'hoi4d': {
        'ras': '/home/an/data/ras/hoi4d1_vggt_omega',
        'hawor': '/home/an/data/hawor/hoi4d',
        'default': '/home/an/data/hawor/hoi4d/output/alignment/transform_params.npz',
        'svd': './output/alignment_hoi4d_svd/transform_params.npz'},
    '7': {
        'ras': '/home/an/data/ras/7_vggt_omega',
        'hawor': '/home/an/data/hawor/7',
        'default': '/home/an/data/hawor/7/output/alignment/transform_params.npz',
        'svd': './output/alignment_7_svd/transform_params.npz'},
}

def load_ras(ras_dir):
    ext_dir = os.path.join(ras_dir, 'extrinsics')
    files = sorted(glob.glob(os.path.join(ext_dir, '*.txt')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
    R, t = [], []
    for f in files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4): ext = np.vstack([ext, [0,0,0,1]])
        Rw = ext[:3,:3]; tw = ext[:3,3]
        Rc = Rw.T; R.append(Rc); t.append(-Rc @ tw)
    return np.array(R), np.array(t)

def load_hawor(hawor_dir):
    rec_dir = f'{hawor_dir}/reconstruction'
    matches = sorted(glob.glob(f'{rec_dir}/hawor_results_*.npz'))
    h = dict(np.load(matches[-1], allow_pickle=True))
    return h['R_c2w'], h['t_c2w']

def load_transform(params_path):
    p = dict(np.load(params_path))
    return p['R_hand_to_glb'], float(p['scale_ratio']), p['t_hand_to_glb']

def transform_hawor(R_hand_to_glb, scale_ratio, t_hand_to_glb, R_c2w_h, t_c2w_h):
    s = scale_ratio
    t_glb = s * (R_hand_to_glb @ t_c2w_h.T).T + t_hand_to_glb
    R_glb = R_hand_to_glb @ R_c2w_h
    return t_glb, R_glb

def pca_angle(a, b):
    ac = a - a.mean(0); bc = b - b.mean(0)
    _, _, Va = np.linalg.svd(ac, full_matrices=False)
    _, _, Vb = np.linalg.svd(bc, full_matrices=False)
    d = np.clip(np.dot(Va[0], Vb[0]), -1, 1)
    if d < 0: d = -d
    return np.degrees(np.arccos(d))

def resample(p, n):
    return p[np.linspace(0, len(p)-1, n, dtype=int)]

out_dir = './output/trajectory_plots_3d'
os.makedirs(out_dir, exist_ok=True)

for name, ex in EXAMPLES.items():
    print(f'\n{"="*60}\n  {name}\n{"="*60}')
    Rr, tr = load_ras(ex['ras'])
    Rh, th = load_hawor(ex['hawor'])

    # Load transforms
    R_d, s_d, t_d = load_transform(ex['default'])
    R_s, s_s, t_s = load_transform(ex['svd'])

    # Transform HaWoR trajectories
    t_d_glb, R_d_glb = transform_hawor(R_d, s_d, t_d, Rh, th)
    t_s_glb, R_s_glb = transform_hawor(R_s, s_s, t_s, Rh, th)

    # Resample to RAS frames
    t_d_s = resample(t_d_glb, len(tr))
    t_s_s = resample(t_s_glb, len(tr))

    # Compute PCA angles
    pca_d = pca_angle(t_d_s, tr)
    pca_s = pca_angle(t_s_s, tr)
    mean_dist_d = np.linalg.norm(t_d_s - tr, axis=1).mean()
    mean_dist_s = np.linalg.norm(t_s_s - tr, axis=1).mean()

    print(f'  默认: PCA={pca_d:.1f}°  平均距离={mean_dist_d:.4f}m')
    print(f'  SVD:  PCA={pca_s:.1f}°  平均距离={mean_dist_s:.4f}m')

    # === 生成 3D 图 ===
    fig = plt.figure(figsize=(20, 6))
    colors = {'RAS': '#2196F3', 'Default': '#FF9800', 'SVD': '#4CAF50'}
    styles = {'RAS': '-', 'Default': '-', 'SVD': '-'}
    lw = {'RAS': 3, 'Default': 2, 'SVD': 2}

    # 第一帧朝向
    fwd_ras = Rr[0, :, 2]
    fwd_d = R_d_glb[0, :, 2]
    fwd_s = R_s_glb[0, :, 2]
    fwd_dict = {'RAS': fwd_ras, 'Default': fwd_d, 'SVD': fwd_s}

    trajs = {'RAS': tr, 'Default': t_d_s, 'SVD': t_s_s}

    for idx, view in enumerate(['XY', 'XZ', 'YZ']):
        ax = fig.add_subplot(1, 3, idx+1, projection='3d')
        for label, traj in trajs.items():
            c = colors.get(label, '#999')
            pca_label = {'RAS': '-', 'Default': f'{pca_d:.1f}°', 'SVD': f'{pca_s:.1f}°'}[label]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=c, linewidth=lw[label], label=f'{label} PCA={pca_label}')
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color=c, s=80, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color=c, s=50, marker='^', edgecolors='k', linewidths=0.5, zorder=5)

        # 朝向箭头 (forward = +Z column)
        arrow_len = 0.02
        for label in ['RAS', 'Default', 'SVD']:
            fwd = fwd_dict[label]
            pos = trajs[label][0]
            c = colors[label]
            ax.quiver(pos[0], pos[1], pos[2], fwd[0], fwd[1], fwd[2],
                      length=arrow_len, color=c, linewidth=2.5, alpha=0.8)

        if view == 'XY':
            ax.view_init(elev=90, azim=-90)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        elif view == 'XZ':
            ax.view_init(elev=0, azim=-90)
            ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('Y (m)')
        else:
            ax.view_init(elev=0, azim=0)
            ax.set_xlabel('Y (m)'); ax.set_ylabel('Z (m)'); ax.set_zlabel('X (m)')

        ax.set_title(f'{name} — {view} View', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7)

    plt.suptitle(f'{name}: RAS (blue) vs Default (orange) vs SVD (green) — PCA: default={pca_d:.1f}°, SVD={pca_s:.1f}°',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f'{name}_3d.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  保存: {path}')

print('\n所有 3D 图生成完成!')
