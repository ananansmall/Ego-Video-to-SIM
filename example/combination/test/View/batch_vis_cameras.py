#!/usr/bin/env python3
"""
batch_vis_cameras.py — 为所有例子生成 render_quick 风格的 3D 相机轨迹对比图

输出到 output/trajectory_plots/ 下, 每个例子一张 PNG + HTML
每条轨迹独立颜色, 与 render_quick.py 风格一致。
"""
import argparse, sys, warnings, os, glob
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  注册 3d 投影

EXAMPLES = {
    '121': {
        'hawor': '/home/an/data/hawor/121_C5_CellPhone_161deg',
        'ras':   '/home/an/data/ras/121_C5_CellPhone_161deg_vggt_omega',
        'params': '/home/an/data/hawor/121_C5_CellPhone_161deg/output/alignment/transform_params.npz',
    },
    '2chair-bottle': {
        'hawor': '/home/an/data/hawor/2chair-bottle',
        'ras':   '/home/an/data/ras/2chair-bottle_vggt_omega',
        'params': '/home/an/robot_world_ws/src/dex-retargeting/example/combination/test/output/alignment_2chair_default/transform_params.npz',
    },
    'hoi4d': {
        'hawor': '/home/an/data/hawor/hoi4d',
        'ras':   '/home/an/data/ras/hoi4d1_vggt_omega',
        'params': '/home/an/data/hawor/hoi4d/output/alignment/transform_params.npz',
    },
    '3chair-bottle': {
        'hawor': '/home/an/data/hawor/3chair-bottle',
        'ras':   '/home/an/data/ras/3chair-bottle_vggt_omega',
        'params': '/home/an/data/hawor/3chair-bottle/output/alignment/transform_params.npz',
    },
    '7': {
        'hawor': '/home/an/data/hawor/7',
        'ras':   '/home/an/data/ras/7_vggt_omega',
        'params': '/home/an/data/hawor/7/output/alignment/transform_params.npz',
    },
}

COLORS = {
    'hawor_cam': 'navy',
    'ras_cam': 'limegreen',
}


def load_hawor_cameras(hawor_dir, params_path, s_override=None, R_override=None, t_override=None):
    """加载 HaWoR 重建 + transform_params, 返回 (t_glb, R_glb)"""
    rec_dir = f'{hawor_dir}/reconstruction'
    matches = sorted(glob.glob(f'{rec_dir}/hawor_results_*.npz'))
    # 优先 depth-aligned 版本, 排除 _factors 文件
    candidates = [m for m in matches if 'factors' not in m]
    npz_path = candidates[-1]
    h = dict(np.load(npz_path, allow_pickle=True))
    R_c2w_h = h['R_c2w']
    t_c2w_h = h['t_c2w']

    p = dict(np.load(params_path))
    required_keys = {'scale_ratio', 'R_hand_to_glb', 't_hand_to_glb'}
    missing = required_keys - set(p.keys())
    if missing:
        raise KeyError(f'params.npz missing keys: {missing}')
    if s_override is not None:
        s, R, t = s_override, R_override, t_override
    else:
        s, R, t = float(p['scale_ratio']), p['R_hand_to_glb'], p['t_hand_to_glb']

    t_glb = s * (R @ t_c2w_h.T).T + t
    R_glb = R @ R_c2w_h
    return t_glb, R_glb, s, R, t


def load_ras_cameras(ras_dir):
    """加载 RAS 外参, 返回 (t_c2w, R_c2w)"""
    ext_dir = f'{ras_dir}/extrinsics'
    files = sorted(glob.glob(f'{ext_dir}/*.txt'),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
    R, t = [], []
    for f in files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        Rc = ext[:3, :3].T
        t_c = -Rc @ ext[:3, 3]
        R.append(Rc)
        t.append(t_c)
    return np.array(R), np.array(t)


def resample(p, n):
    return p[np.linspace(0, len(p) - 1, n, dtype=int)]


def plot_3d_matplotlib(name, t_hawor, R_hawor, t_ras, R_ras, s, R_hand, save_dir):
    """生成 render_quick 风格的 3D 图"""
    os.makedirs(save_dir, exist_ok=True)
    n_ras = len(t_ras)
    n_h = len(t_hawor)

    # 重采样
    t_h_s = resample(t_hawor, n_ras)
    R_h_s = resample(R_hawor, n_ras)

    # 统计
    pos0 = np.linalg.norm(t_hawor[0] - t_ras[0])
    dists = np.linalg.norm(t_h_s - t_ras, axis=1)
    fwd_h0 = R_hawor[0, :, 2]
    fwd_r0 = R_ras[0, :, 2]
    fwd0 = np.degrees(np.arccos(np.clip(np.dot(fwd_h0, fwd_r0), -1, 1)))

    print(f'\n{name}:')
    print(f'  HaWoR: {n_h}f, RAS: {n_ras}f')
    print(f'  Frame0 距离: {pos0:.6f}m, 朝向: {fwd0:.2f}°')
    print(f'  逐帧距离: min={dists.min():.4f}m max={dists.max():.4f}m mean={dists.mean():.4f}m')

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 坐标轴范围
    all_pts = [t_hawor, t_ras]
    ag = np.concatenate(all_pts, axis=0)
    lo, hi = ag.min(0), ag.max(0)
    half = max(0.5, max((hi - lo).max() / 2, 0.5) * 1.3)
    lim_lo, lim_hi = -half, half
    ax.set_xlim3d(lim_lo, lim_hi)
    ax.set_ylim3d(lim_lo, lim_hi)
    ax.set_zlim3d(lim_lo, lim_hi)
    try:
        ax.set_box_aspect([(lim_hi - lim_lo)] * 3)
    except AttributeError:
        pass

    # 世界坐标轴
    o = np.zeros(3)
    for c, v in [('red', [0.3, 0, 0]), ('green', [0, 0.3, 0]), ('blue', [0, 0, 0.3])]:
        ax.plot([o[0], v[0]], [o[1], v[1]], [o[2], v[2]], color=c, lw=3, alpha=0.7)

    # HaWoR 轨迹 (深蓝)
    ax.plot(t_hawor[:, 0], t_hawor[:, 1], t_hawor[:, 2],
            color=COLORS['hawor_cam'], lw=1.5, alpha=0.7,
            label=f'HaWoR cam ({n_h}f)')
    ax.scatter(*t_hawor[0], c=COLORS['hawor_cam'], s=80, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
    ax.scatter(*t_hawor[-1], c=COLORS['hawor_cam'], s=50, marker='x', edgecolors='k', linewidths=0.5, zorder=5)

    # RAS 轨迹 (亮绿)
    ax.plot(t_ras[:, 0], t_ras[:, 1], t_ras[:, 2],
            color=COLORS['ras_cam'], lw=2, alpha=0.9,
            label=f'RAS cam ({n_ras}f)')
    ax.scatter(*t_ras[0], c=COLORS['ras_cam'], s=80, marker='s', edgecolors='k', linewidths=0.5, zorder=5)
    ax.scatter(*t_ras[-1], c=COLORS['ras_cam'], s=50, marker='D', edgecolors='k', linewidths=0.5, zorder=5)

    # 第一帧朝向箭头
    arrow_len = 0.1
    for pos, fwd, color, label in [
        (t_hawor[0], R_hawor[0, :, 2], 'blue', 'HaWoR'),
        (t_ras[0], R_ras[0, :, 2], 'red', 'RAS'),
    ]:
        ax.quiver(pos[0], pos[1], pos[2],
                  fwd[0] * arrow_len, fwd[1] * arrow_len, fwd[2] * arrow_len,
                  color=color, lw=3, alpha=0.8)
        ax.text(pos[0] + fwd[0] * arrow_len * 1.2,
                pos[1] + fwd[1] * arrow_len * 1.2,
                pos[2] + fwd[2] * arrow_len * 1.2,
                f'{label} fwd', color=color, fontsize=9, fontweight='bold')

    # 标签
    for pos, color in [(t_hawor[0], COLORS['hawor_cam']), (t_ras[0], COLORS['ras_cam'])]:
        ax.text(pos[0], pos[1], pos[2] + 0.05, 'cam0', color=color, fontsize=9,
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    rot_angle = np.degrees(np.arccos(np.clip((np.trace(R_hand) - 1) / 2, -1, 1)))
    ax.set_title(f'{name}: HaWoR vs RAS | s={s:.3f}x R_rot={rot_angle:.1f}° | '
                 f'frame0={pos0:.4f}m {fwd0:.1f}° | mean_dist={dists.mean():.4f}m',
                 fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    png = os.path.join(save_dir, f'{name}_3d.png')
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  PNG: {png}')
    return png


def plot_3d_html(name, t_hawor, R_hawor, t_ras, R_ras, s, R_hand, save_dir):
    """生成交互 Plotly HTML"""
    os.makedirs(save_dir, exist_ok=True)
    n_ras = len(t_ras)
    n_h = len(t_hawor)

    pos0 = np.linalg.norm(t_hawor[0] - t_ras[0])
    dists = np.linalg.norm(resample(t_hawor, n_ras) - t_ras, axis=1)
    fwd_h0 = R_hawor[0, :, 2]
    fwd_r0 = R_ras[0, :, 2]
    fwd0 = np.degrees(np.arccos(np.clip(np.dot(fwd_h0, fwd_r0), -1, 1)))
    rot_angle = np.degrees(np.arccos(np.clip((np.trace(R_hand) - 1) / 2, -1, 1)))

    import json
    traces = []

    def add_line(pts, color, label, lw=2, size=3):
        traces.append({
            'type': 'scatter3d', 'mode': 'lines',
            'x': pts[:, 0].tolist(), 'y': pts[:, 1].tolist(), 'z': pts[:, 2].tolist(),
            'line': {'color': color, 'width': lw}, 'name': label, 'hoverinfo': 'skip'
        })
        for pos, sym in [(pts[0], 'circle'), (pts[-1], 'triangle-up')]:
            traces.append({
                'type': 'scatter3d', 'mode': 'markers',
                'x': [pos[0]], 'y': [pos[1]], 'z': [pos[2]],
                'marker': {'size': 8, 'color': color, 'symbol': sym},
                'name': f'{label} end', 'hoverinfo': 'skip'
            })
        # forward arrow
        fwd = None
        if label.startswith('HaWoR'):
            fwd = R_hawor[0, :, 2]
        else:
            fwd = R_ras[0, :, 2]
        fsc = 0.03
        traces.append({
            'type': 'cone',
            'x': [pts[0, 0]], 'y': [pts[0, 1]], 'z': [pts[0, 2]],
            'u': [fwd[0] * fsc], 'v': [fwd[1] * fsc], 'w': [fwd[2] * fsc],
            'color': color, 'sizemode': 'absolute',
            'name': f'{label} fwd', 'hoverinfo': 'skip'
        })

    add_line(t_hawor, COLORS['hawor_cam'], f'HaWoR cam ({n_h}f)', 1.5)
    add_line(t_ras, COLORS['ras_cam'], f'RAS cam ({n_ras}f)', 2.5)

    stats = (f'<b>{name}</b><br>'
             f'🔵 HaWoR: {n_h}f<br>'
             f'🟢 RAS: {n_ras}f<br>'
             f's={s:.3f}x, R_rot={rot_angle:.1f}°<br>'
             f'frame0: dist={pos0:.4f}m, fwd={fwd0:.1f}°<br>'
             f'mean dist: {dists.mean():.4f}m')

    data_json = json.dumps(traces)
    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{name} 相机轨迹 3D</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{margin:0;background:#111;color:#fff;font-family:sans-serif;}}
#stats{{position:fixed;top:10px;left:12px;background:rgba(0,0,0,.85);
  padding:12px 16px;border-radius:8px;font-size:13px;line-height:1.6;z-index:10;
  border-left:3px solid #2196F3;}}
#chart{{width:100vw;height:100vh;}}</style></head>
<body>
<div id="stats">{stats}</div>
<div id="chart"></div>
<script>
const data = {data_json};
Plotly.newPlot("chart", data, {{
  showlegend: false, margin: {{l:0,r:0,t:0,b:0}},
  paper_bgcolor:"#111", plot_bgcolor:"#111",
  scene: {{
    xaxis: {{title:"X (m)", showgrid:true, color:"#555", gridcolor:"#222"}},
    yaxis: {{title:"Y (m)", showgrid:true, color:"#555", gridcolor:"#222"}},
    zaxis: {{title:"Z (m)", showgrid:true, color:"#555", gridcolor:"#222"}},
    bgcolor:"#111",
    camera: {{eye: {{x:1.8, y:1.5, z:1.2}}}},
    aspectratio: {{x:1, y:1, z:1}},
  }},
}}, {{responsive:true, displayModeBar:false}});
window.addEventListener("resize", ()=>Plotly.Plots.resize("chart"));
</script></body></html>'''

    path = os.path.join(save_dir, f'{name}_3d.html')
    with open(path, 'w') as f:
        f.write(html)
    print(f'  HTML: {path}')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='./output/trajectory_plots')
    parser.add_argument('--html', action='store_true',
                        help='额外生成可交互 Plotly HTML')
    args = parser.parse_args()

    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)

    print('=' * 70)
    print('  批量生成相机轨迹对比图')
    print('=' * 70)

    for name, ex in EXAMPLES.items():
        R_ras, t_ras = load_ras_cameras(ex['ras'])
        t_hawor, R_hawor, s, R_hand, t_hand = load_hawor_cameras(
            ex['hawor'], ex['params'])
        plot_3d_matplotlib(name, t_hawor, R_hawor, t_ras, R_ras, s, R_hand, save_dir)

        if args.html:
            plot_3d_html(name, t_hawor, R_hawor, t_ras, R_ras, s, R_hand, save_dir)

    print('\n✅ 完成!')
    print(f'   输出目录: {os.path.abspath(save_dir)}')


if __name__ == '__main__':
    main()
