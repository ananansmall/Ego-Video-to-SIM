#!/usr/bin/env python3
"""
Stage 1 完整可视化: GLB 场景 + 相机轨迹 + MANO 手部 + Stage 1 抓取轨迹
所有数据统一到 SAPIEN 坐标系 (与仿真一致)
输出 PNG + 交互式 HTML (plotly)
"""
import sys, os, numpy as np, warnings, re, argparse
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "output")
sys.path.insert(0, HERE)

# ── SAPIEN 坐标变换 (与 grasp_hawor.py 一致) ──
R_x = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_x  # [[1,0,0],[0,0,-1],[0,1,0]]


def xform(pts, s, R, t):
    return s * (R @ pts.T).T + t


def transform_ras_to_sapien(pts, s_inv, R_inv, t_inv):
    """RAS/GLB 坐标 → SAPIEN 坐标 (与 load_glb_with_physics 一致)"""
    p_hawor = s_inv * (R_inv @ pts.T).T + t_inv
    p_sapien = (RXWORLD_TO_SAPIEN @ p_hawor.T).T
    return p_sapien


def load_glb_geoms_sapien(glb_path, s_inv, R_inv, t_inv):
    """加载 GLB 并变换到 SAPIEN 坐标, 地面对齐 (与 load_glb_with_physics 一致)
    返回: (geoms, ground_z)  ground_z 用于对齐其他数据"""
    if not os.path.exists(glb_path):
        return [], 0.0
    scene = trimesh.load(glb_path)
    all_min_z = []
    geoms_raw = []
    for name, geom in scene.geometry.items():
        if not hasattr(geom, 'bounds') or geom.bounds is None:
            continue
        verts = geom.vertices
        verts_sapien = transform_ras_to_sapien(verts, s_inv, R_inv, t_inv)
        bbox_min = verts_sapien.min(0)
        bbox_max = verts_sapien.max(0)
        bbox_center = (bbox_min + bbox_max) / 2
        all_min_z.append(bbox_min[2])
        geoms_raw.append((name, bbox_center, bbox_min, bbox_max))
    # 地面对齐: 与 load_glb_with_physics 一致, 减去 ground_z
    ground_z = min(all_min_z) if all_min_z else 0.0
    geoms = []
    for name, center, bmin, bmax in geoms_raw:
        center_aligned = center.copy()
        center_aligned[2] -= ground_z
        bounds_aligned = np.array([bmin, bmax])
        bounds_aligned[:, 2] -= ground_z
        geoms.append((name, center_aligned, bounds_aligned))
    print(f"  ground_z={ground_z:.4f}, 对齐后物体 z≈{geoms[0][2][0,2]:.3f}" if geoms else "  无物体")
    return geoms, ground_z


def load_hawor_data_sapien(hawor_dir, params_p, s_inv, R_inv, t_inv, use_depth=False):
    """加载 HaWoR 数据并变换到 SAPIEN 坐标"""
    from pathlib import Path
    rec = Path(hawor_dir) / 'reconstruction'
    files = sorted(rec.glob('hawor_results_*.npz'))
    if use_depth:
        p = next((f for f in files if '_depth_aligned' in str(f)), files[0])
    else:
        p = next((f for f in files if '_depth_aligned' not in str(f)), files[0])
    d = dict(np.load(str(p), allow_pickle=True))
    par = dict(np.load(params_p))
    s = float(par['scale_ratio'])
    R = par['R_hand_to_glb']
    t = par['t_hand_to_glb']
    Rx_hand = par.get('Rx_hand', np.diag([1, -1, -1]))
    R_hand = R @ Rx_hand

    rot_angle = np.arccos(min(1, max(-1, (np.trace(R)-1)/2))) * 180 / np.pi

    # 相机 -> GLB -> SAPIEN
    t0_h = d['t_c2w'][0]
    t0_glb = xform(np.array([t0_h]), s, R, t)[0]
    t0_sapien = transform_ras_to_sapien(np.array([t0_h]), s_inv, R_inv, t_inv)[0]

    cam_glb = xform(d['t_c2w'], s, R, t)
    cam_sapien = transform_ras_to_sapien(cam_glb, s_inv, R_inv, t_inv)
    cam_valid = cam_sapien[:, 2] >= 0

    # 手腕 -> GLB -> SAPIEN
    hands = {}
    for hidx in [0, 1]:
        pts = d['pred_trans'][hidx]
        x_glb = xform(pts, s, R_hand, t)
        x_sapien = transform_ras_to_sapien(x_glb, s_inv, R_inv, t_inv)
        ok = ~np.any(np.isnan(x_sapien), axis=1)
        if ok.any():
            hands[hidx] = x_sapien[ok]
    return d, s, R, t, rot_angle, t0_sapien, cam_sapien, cam_valid, hands


def load_ras_cameras_sapien(ras_dir, s_inv, R_inv, t_inv):
    """加载 RAS 相机位置并变换到 SAPIEN 坐标"""
    from pathlib import Path
    ext_dir = Path(ras_dir) / 'extrinsics'
    files = sorted(ext_dir.glob('*.txt'), key=lambda x: int(x.stem))
    ras_pos = []
    for f in files:
        ext = np.loadtxt(str(f))
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_w2c, t_w2c = ext[:3, :3], ext[:3, 3]
        R_c2w = R_w2c.T
        ras_pos.append(-R_c2w @ t_w2c)
    ras_pos = np.array(ras_pos)
    ras_pos_sapien = transform_ras_to_sapien(ras_pos, s_inv, R_inv, t_inv)
    ras_valid = ras_pos_sapien[:, 2] >= 0
    return ras_pos_sapien, ras_valid


def draw_axes_arrows(ax, origin, scale=0.1, lw=4):
    o = np.array(origin)
    for c, lbl, v in [('red', 'X', [scale, 0, 0]), ('green', 'Y', [0, scale, 0]), ('blue', 'Z', [0, 0, scale])]:
        tip = o + v
        ax.plot([o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]], color=c, lw=lw, alpha=0.9, zorder=10)
        ax.text(tip[0] + v[0] * 0.1, tip[1] + v[1] * 0.1, tip[2] + v[2] * 0.1, lbl, color=c, fontsize=12, fontweight='bold')


def draw_bbox(ax, bounds, color, alpha=0.2):
    p0, p1 = bounds
    c = np.array([[p0[0], p0[1], p0[2]], [p1[0], p0[1], p0[2]], [p1[0], p1[1], p0[2]], [p0[0], p1[1], p0[2]],
                  [p0[0], p0[1], p1[2]], [p1[0], p0[1], p1[2]], [p1[0], p1[1], p1[2]], [p0[0], p1[1], p1[2]]])
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot3D(*c[[a, b]].T, color=color, alpha=alpha, lw=1.0)


def parse_stage1_log(log_path):
    """解析 Stage 1 最终 rollout 轨迹"""
    frames = []
    if not os.path.exists(log_path):
        return frames
    with open(log_path) as f:
        lines = f.readlines()
    debug_lines = [l for l in lines if "Stage1 debug" in l and "f1_qpos" in l]
    debug_lines = debug_lines[-17:]  # 最后一次完整 rollout
    for line in debug_lines:
        m = re.search(r'F(\d+)', line)
        if not m: continue
        frame = int(m.group(1))
        kv = {}
        for p in line.replace('[', ' ').replace(']', ' ').replace(',', ' ').split():
            if '=' in p:
                k, v = p.split('=')
                try:
                    kv[k.strip()] = float(v)
                except:
                    pass
        frames.append({
            'frame': frame,
            'pos_z': kv.get('pos_z', 0),
            'obj_z': kv.get('obj_z', 0),
            'force': kv.get('force', 0),
        })
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default="right")
    parser.add_argument("--output", default=os.path.join(OUTPUT, "gripper_only_right", "stage1_full"))
    args = parser.parse_args()

    side = args.side
    hawor_dir = "/home/an/data/hawor/7"
    ras_dir = "/home/an/data/ras/my_7mp4_result"
    params_p = os.path.join(hawor_dir, 'output/alignment/transform_params.npz')
    glb_path = os.path.join(ras_dir, 'final_scene.glb')
    log_path = os.path.join(OUTPUT, f"gripper_only_{side}", "grasp.log")
    grasp_path = os.path.join(OUTPUT, f"gripper_only_{side}", "stage1", "best_grasp.npz")

    # 加载 alignment 参数
    par = np.load(params_p)
    s_inv = float(par['s_inv'])
    R_inv = par['R_inv']
    t_inv = par['t_inv']
    s = float(par['scale_ratio'])
    R = par['R_hand_to_glb']
    t = par['t_hand_to_glb']

    # 1. 加载 GLB → SAPIEN (地面对齐)
    geoms, ground_z = load_glb_geoms_sapien(glb_path, s_inv, R_inv, t_inv)
    print(f"GLB 物体 (SAPIEN): {len(geoms)}, ground_z={ground_z:.4f}")

    # 2. 加载 HaWoR 数据 → SAPIEN
    d, _, _, _, ra, t0_sapien, cam_sapien, cam_valid, hands = load_hawor_data_sapien(
        hawor_dir, params_p, s_inv, R_inv, t_inv)
    # 对相机和手部数据应用 ground_z 偏移 (与 GLB 物体对齐)
    cam_sapien[:, 2] -= ground_z
    for hidx in [0, 1]:
        if hidx in hands:
            hands[hidx][:, 2] -= ground_z
    cam_valid = cam_sapien[:, 2] >= 0  # 重新计算有效帧
    print(f"HaWoR 相机: {cam_valid.sum()} 有效帧, 手: {[k for k in hands.keys()]}")

    # 3. 加载 RAS 相机 → SAPIEN
    ras_pos_sapien, ras_valid = load_ras_cameras_sapien(ras_dir, s_inv, R_inv, t_inv)
    ras_pos_sapien[:, 2] -= ground_z  # 也应用 ground_z
    ras_valid = ras_pos_sapien[:, 2] >= 0
    print(f"RAS 相机: {ras_valid.sum()} 有效帧")

    # 4. 加载 Stage 1 数据
    d_grasp = np.load(grasp_path, allow_pickle=True)
    grasp_pos = d_grasp["pos"]
    log_frames = parse_stage1_log(log_path)
    obj_lift = float(d_grasp.get("obj_lift", 0))
    peak_force = float(d_grasp.get("peak_grip_force", 0))
    print(f"Stage 1: grasp_pos={grasp_pos.round(4)}, lift={obj_lift*100:.1f}cm, log={len(log_frames)}帧")

    # 构建轨迹 (SAPIEN 坐标, 与仿真一致)
    base_xy = grasp_pos[:2]
    if log_frames:
        gripper_pts = [[base_xy[0], base_xy[1], f['pos_z']] for f in log_frames]
        obj_pts = [[base_xy[0], base_xy[1], f['obj_z']] for f in log_frames]
        gripper_traj = np.array(gripper_pts)
        obj_traj = np.array(obj_pts)
        obj_traj[:, :2] = [base_xy[0], base_xy[1]]
    else:
        gripper_traj = np.array([])
        obj_traj = np.array([])

    # ═══════════════════════════════════════════════
    # MATPLOTLIB 渲染
    # ═══════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_title(f"Stage 1 Grasp + GLB Scene [SAPIEN Coord]\n"
                 f"obj_lift={obj_lift*100:.1f}cm  peak_force={peak_force:.1f}N  Scale={s:.3f}×  Rot={ra:.1f}°",
                 fontweight='bold')

    draw_axes_arrows(ax, [0, 0, 0], scale=0.1)

    # GLB 物体 bbox (SAPIEN 坐标)
    for i, (nm, ct, bd) in enumerate(geoms):
        col = plt.cm.tab10(i)
        draw_bbox(ax, bd, col, alpha=0.15)
        ax.scatter(*ct, c=col, s=50, marker='o', edgecolors='k', zorder=5)
        ax.text(ct[0], ct[1], ct[2] + 0.02, f'{nm[:8]}', fontsize=6, color=col)

    # HaWoR 相机轨迹 (SAPIEN)
    if cam_valid.any():
        c = cam_sapien[cam_valid]
        ax.plot(c[:, 0], c[:, 1], c[:, 2], 'b-', lw=1.5, alpha=0.5, label='HaWoR cam')
        ax.scatter(*c[0], c='blue', s=50, marker='d', zorder=6, label='HaWoR start')

    # RAS 相机轨迹 (SAPIEN)
    if ras_valid.any():
        r = ras_pos_sapien[ras_valid]
        ax.plot(r[:, 0], r[:, 1], r[:, 2], '-', color='orange', lw=1.5, alpha=0.8, label='RAS cam')
        ax.scatter(*r[0], c='orange', s=50, marker='d', zorder=6, label='RAS start')

    # MANO 手部轨迹 (SAPIEN)
    for hidx, col, lbl in [(0, 'green', 'L Hand'), (1, 'gold', 'R Hand')]:
        if hidx in hands:
            pts = hands[hidx]
            v = pts[:, 2] >= 0
            if v.any():
                ax.plot(pts[v, 0], pts[v, 1], pts[v, 2], color=col, lw=1.0, ls='-', alpha=0.7, label=lbl)
                ax.scatter(pts[v, 0], pts[v, 1], pts[v, 2], s=1.5, color=col, alpha=0.4, marker='.')

    # Stage 1 夹爪轨迹 (SAPIEN, 无需变换)
    if len(gripper_traj) > 0:
        ax.plot(gripper_traj[:, 0], gripper_traj[:, 1], gripper_traj[:, 2],
                'cyan', lw=3.0, label='Gripper Traj', alpha=0.9)
        ax.scatter(*gripper_traj[0], c='cyan', s=80, edgecolors='k', zorder=7, label='Gripper Start')
        ax.scatter(*gripper_traj[-1], c='cyan', s=80, marker='*', edgecolors='k', zorder=7, label='Gripper End')

    # Stage 1 物体轨迹
    if len(obj_traj) > 0:
        ax.plot(obj_traj[:, 0], obj_traj[:, 1], obj_traj[:, 2],
                'magenta', lw=2.5, ls='--', label='Object Traj', alpha=0.8)
        ax.scatter(*obj_traj[0], c='magenta', s=60, edgecolors='k', zorder=7, label='Object Start')
        ax.scatter(*obj_traj[-1], c='magenta', s=60, marker='*', edgecolors='k', zorder=7, label='Object End')

    # Grasp pose
    ax.scatter(*grasp_pos, c='lime', s=200, marker='*', edgecolors='k', zorder=10, label='Grasp Pose')

    # 标注关键帧
    if len(gripper_traj) > 0:
        n = len(gripper_traj)
        for fid, col, lbl in [(10, 'cyan', 'F10'), (30, 'orange', 'F30'), (min(65, n - 1), 'red', 'F65')]:
            if fid < n:
                ax.scatter(*gripper_traj[fid], c=col, s=60, edgecolors='k', zorder=6)
                ax.text(*(gripper_traj[fid] + [0, 0, 0.008]), lbl, fontsize=8, color=col)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper left', fontsize=8)

    # 自动范围
    all_pts = [bd for _, _, bd in geoms]
    if cam_valid.any(): all_pts.append(cam_sapien[cam_valid])
    if ras_valid.any(): all_pts.append(ras_pos_sapien[ras_valid])
    for hidx in [0, 1]:
        if hidx in hands: all_pts.append(hands[hidx])
    if len(gripper_traj) > 0: all_pts.append(gripper_traj)
    if len(obj_traj) > 0: all_pts.append(obj_traj)
    all_pts = [p for p in all_pts if len(p) > 0]
    if all_pts:
        ag = np.concatenate(all_pts, axis=0)
        ag = ag[np.isfinite(ag).all(axis=1)]
        if len(ag) > 0:
            lo, hi = ag.min(0), ag.max(0)
            rng = np.maximum(hi - lo, 0.2)
            pad = rng * 0.12
            ax.set_xlim3d(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim3d(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_zlim3d(max(lo[2] - pad[2], -0.05), hi[2] + pad[2])

    plt.tight_layout()
    png_out = f"{args.output}_overview.png"
    plt.savefig(png_out, dpi=150, bbox_inches='tight')
    print(f"Saved: {png_out}")
    plt.close()

    # ═══════════════════════════════════════════════
    # PLOTLY HTML (交互式)
    # ═══════════════════════════════════════════════
    fig_p = go.Figure()

    ax_s = 0.1
    for i, (clr, lbl) in enumerate([('red', 'X'), ('green', 'Y'), ('blue', 'Z')]):
        v = np.zeros(3); v[i] = ax_s
        fig_p.add_trace(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode='lines+text', line=dict(color=clr, width=6),
            text=['', f' {lbl}'], textfont=dict(color=clr, size=18), showlegend=False))

    # GLB 物体 (SAPIEN)
    for i, (nm, ct, bd) in enumerate(geoms):
        col = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'][i % 6]
        p0, p1 = bd
        corners = np.array([[p0[0], p0[1], p0[2]], [p1[0], p0[1], p0[2]], [p1[0], p1[1], p0[2]], [p0[0], p1[1], p0[2]],
                            [p0[0], p0[1], p1[2]], [p1[0], p0[1], p1[2]], [p1[0], p1[1], p1[2]], [p0[0], p1[1], p1[2]]])
        for e in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            fig_p.add_trace(go.Scatter3d(x=corners[e, 0], y=corners[e, 1], z=corners[e, 2],
                mode='lines', line=dict(color=col, width=2), showlegend=False))

    # HaWoR 相机 (SAPIEN)
    if cam_valid.any():
        c = cam_sapien[cam_valid]
        fig_p.add_trace(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2],
            mode='lines', line=dict(color='blue', width=2), name='HaWoR cam'))

    # RAS 相机 (SAPIEN)
    if ras_valid.any():
        r = ras_pos_sapien[ras_valid]
        fig_p.add_trace(go.Scatter3d(x=r[:, 0], y=r[:, 1], z=r[:, 2],
            mode='lines', line=dict(color='orange', width=2.5), name='RAS cam'))

    # MANO 手部 (SAPIEN)
    for hidx, col, lbl in [(0, 'rgba(0,255,0,0.6)', 'L Hand'), (1, 'rgba(255,215,0,0.8)', 'R Hand')]:
        if hidx in hands:
            pts = hands[hidx]
            v = pts[:, 2] >= 0
            if v.any():
                fig_p.add_trace(go.Scatter3d(x=pts[v, 0], y=pts[v, 1], z=pts[v, 2],
                    mode='lines+markers', marker=dict(size=1.5, color=col),
                    line=dict(color=col, width=1.5), name=lbl))

    # Stage 1 夹爪轨迹 (SAPIEN)
    if len(gripper_traj) > 0:
        fig_p.add_trace(go.Scatter3d(x=gripper_traj[:, 0], y=gripper_traj[:, 1], z=gripper_traj[:, 2],
            mode='lines+markers', marker=dict(size=3, color='cyan'),
            line=dict(color='cyan', width=4), name='Gripper Traj'))

    # Stage 1 物体轨迹 (SAPIEN)
    if len(obj_traj) > 0:
        fig_p.add_trace(go.Scatter3d(x=obj_traj[:, 0], y=obj_traj[:, 1], z=obj_traj[:, 2],
            mode='lines+markers', marker=dict(size=3, color='magenta'),
            line=dict(color='magenta', width=3, dash='dash'), name='Object Traj'))

    # Grasp pose
    fig_p.add_trace(go.Scatter3d(x=[grasp_pos[0]], y=[grasp_pos[1]], z=[grasp_pos[2]],
        mode='markers+text', marker=dict(size=12, color='lime', symbol='diamond'),
        text=['GRASP'], textfont=dict(color='lime', size=14), name='Grasp Pose'))

    fig_p.update_layout(
        title=f'Stage 1 Grasp + GLB Scene [SAPIEN Coord] | lift={obj_lift*100:.1f}cm',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        width=1200, height=900,
        legend=dict(font=dict(size=10), x=0, y=1)
    )
    html_out = f"{args.output}_overview.html"
    fig_p.write_html(html_out)
    print(f"Saved: {html_out}")

    print("\n✅ 完成! 所有数据统一在 SAPIEN 坐标系")
    print(f"  PNG: {png_out}")
    print(f"  HTML: {html_out}")
    print(f"  GLB 物体 SAPIEN z 范围: {[f'{bd[0,2]:.3f}-{bd[1,2]:.3f}' for _, _, bd in geoms]}")


if __name__ == "__main__":
    main()