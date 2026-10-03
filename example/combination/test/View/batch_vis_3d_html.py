#!/usr/bin/env python3
"""batch_vis_3d_html.py — 生成可交互的 Plotly 3D 轨迹对比"""
import numpy as np, glob, os, json

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
        Rc = ext[:3,:3].T; R.append(Rc); t.append(-Rc @ ext[:3,3])
    return np.array(R), np.array(t)

def load_hawor(hawor_dir):
    rec_dir = f'{hawor_dir}/reconstruction'
    matches = [f for f in sorted(glob.glob(f'{rec_dir}/hawor_results_*.npz'))
               if 'factors' not in f]
    h = dict(np.load(matches[-1], allow_pickle=True))
    return h['R_c2w'], h['t_c2w']

def load_transform(path):
    p = dict(np.load(path))
    return p['R_hand_to_glb'], float(p['scale_ratio']), p['t_hand_to_glb']

def transform_hawor(Rh, s, t, R_c2w_h, t_c2w_h):
    return s * (Rh @ t_c2w_h.T).T + t, Rh @ R_c2w_h

def pca_angle(a, b):
    ac = a - a.mean(0); bc = b - b.mean(0)
    _, _, Va = np.linalg.svd(ac, full_matrices=False)
    _, _, Vb = np.linalg.svd(bc, full_matrices=False)
    return np.degrees(np.arccos(np.clip(np.abs(np.dot(Va[0], Vb[0])), -1, 1)))

def resample(p, n):
    return p[np.linspace(0, len(p)-1, n, dtype=int)]

def rot_to_aa(R):
    a = np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))
    if a < 1e-10: return [1,0,0], 0.0
    rx = R[2,1]-R[1,2]; ry = R[0,2]-R[2,0]; rz = R[1,0]-R[0,1]
    return [rx, ry, rz], np.degrees(a)

out_dir = './output/trajectory_plots_3d'
os.makedirs(out_dir, exist_ok=True)

print(f"\n{'例子':<14} {'默认PCA':>8} {'SVD PCA':>8} {'默认距离':>8} {'SVD距离':>8} {'R_cam_align':>12}")
print('-'*62)

for name, ex in EXAMPLES.items():
    Rr, tr = load_ras(ex['ras'])
    Rh_h, th_h = load_hawor(ex['hawor'])
    Rd, sd, td = load_transform(ex['default'])
    Rs, ss, ts = load_transform(ex['svd'])
    t_d_glb, R_d_glb = transform_hawor(Rd, sd, td, Rh_h, th_h)
    t_s_glb, R_s_glb = transform_hawor(Rs, ss, ts, Rh_h, th_h)
    t_d_s = resample(t_d_glb, len(tr))
    t_s_s = resample(t_s_glb, len(tr))
    pca_d = pca_angle(t_d_s, tr)
    pca_s = pca_angle(t_s_s, tr)
    md = np.linalg.norm(t_d_s - tr, axis=1).mean()
    ms = np.linalg.norm(t_s_s - tr, axis=1).mean()
    _, ang = rot_to_aa(Rs @ Rd.T)

    print(f"{name:<14} {pca_d:>7.1f}° {pca_s:>7.1f}° {md:>7.4f}m {ms:>7.4f}m {ang:>10.2f}°")

    # Build traces for Plotly
    traces = []
    for label, traj, color, lw, leg in [
        ('RAS', tr, '#2196F3', 5, 'RAS (true)'),
        ('Default', t_d_s, '#FF9800', 3, f'Default (PCA={pca_d:.1f}°)'),
        ('SVD', t_s_s, '#4CAF50', 3, f'SVD (PCA={pca_s:.1f}°)'),
    ]:
        traces.append({
            'type': 'scatter3d', 'mode': 'lines',
            'x': traj[:,0].tolist(), 'y': traj[:,1].tolist(), 'z': traj[:,2].tolist(),
            'line': {'color': color, 'width': lw}, 'name': leg,
            'hoverinfo': 'skip'
        })

    # Forward arrows at frame 0
    # Scale forward vectors to be visible (0.03m)
    f_scale = 0.03
    for pos, fwd, color in [
        (tr[0], Rr[0,:,2] * f_scale, '#2196F3'),
        (t_d_s[0], R_d_glb[0,:,2] * f_scale, '#FF9800'),
        (t_s_s[0], R_s_glb[0,:,2] * f_scale, '#4CAF50'),
    ]:
        traces.append({
            'type': 'cone',
            'x': [pos[0]], 'y': [pos[1]], 'z': [pos[2]],
            'u': [fwd[0]], 'v': [fwd[1]], 'w': [fwd[2]],
            'color': color, 'sizemode': 'absolute',
            'name': 'forward', 'hoverinfo': 'skip'
        })

    # Start/end markers
    for label, pos, color in [
        ('RAS start', tr[0], '#2196F3'),
        ('Default start', t_d_s[0], '#FF9800'),
        ('SVD start', t_s_s[0], '#4CAF50'),
        ('RAS end', tr[-1], '#2196F3'),
        ('Default end', t_d_s[-1], '#FF9800'),
        ('SVD end', t_s_s[-1], '#4CAF50'),
    ]:
        traces.append({
            'type': 'scatter3d', 'mode': 'markers',
            'x': [pos[0]], 'y': [pos[1]], 'z': [pos[2]],
            'marker': {'size': 10, 'color': color},
            'name': label, 'hoverinfo': 'skip'
        })

    data = json.dumps(traces)
    html = f'''<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{name} 3D轨迹</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>body{{margin:0;background:#111;color:#fff;font-family:sans-serif;}}
#stats{{position:fixed;top:10px;left:12px;background:rgba(0,0,0,.85);padding:12px 16px;
  border-radius:8px;font-size:13px;line-height:1.6;z-index:10;
  border-left:3px solid #2196F3;}}
#chart{{width:100vw;height:100vh;}}</style></head>
<body>
<div id="stats">
<b style="font-size:16px">{name}</b><br>
🔵 RAS (真值): {len(tr)}f<br>
🟠 Default (第一帧对齐): PCA={pca_d:.1f}°, 距离={md:.3f}m<br>
🟢 SVD (全部帧对齐): PCA={pca_s:.1f}°, 距离={ms:.3f}m<br>
<i>R_cam_align (SVD vs Default): {ang:.2f}°</i>
</div>
<div id="chart"></div>
<script>
const data = {data};
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

    path = os.path.join(out_dir, f'{name}_3d.html')
    with open(path, 'w') as f:
        f.write(html)
    print(f"  → {path}")

print('\n✅ 所有 3D HTML 生成完成!')
