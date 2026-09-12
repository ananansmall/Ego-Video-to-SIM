#!/usr/bin/env python3
"""
vis_mano_mapping_variants.py — MANO 映射变体对比图生成

目的: 用同一数据源 (/home/an/data/ras/7_vggt_omega + /home/an/data/hawor/7)
     渲染几种 MANO→GLB 映射变体, 由用户挑选正确的映射逻辑。

变体定义 (2x2 完整矩阵):
  A: 物体=节点世界变换 | MANO=s*R_h2g*Rx_hand+t   (render_quick.py 链)
  B: 物体=节点世界变换 | MANO=s*R_h2g+t           (无 Rx_hand)
  C: 物体=原始本地坐标 | MANO=s*R_h2g*Rx_hand+t   (当前管线链, use_glb_coords=True)
  D: 物体=原始本地坐标 | MANO=s*R_h2g+t           (补全对照)

参考物: HaWoR 相机轨迹 cam_glb = s*R_h2g*t_c2w + t_h2g (所有变体相同, 无 Rx_hand)

输出: output/mapping_variants/D1_mapping_{A,B,C,D}_*.png + overview 拼图
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import FixedLocator

_BASE = Path(__file__).resolve().parent.parent          # trajectory_optimization/
sys.path.insert(0, str(_BASE))
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')

import trimesh
from mano_layer import MANOLayer

H_SID = {0: 'left', 1: 'right'}

# MANO 关节运动树 (manopth 21 关节: 0=wrist, 1-15手指链, 16-20指尖)
BONES = [(0, 1), (1, 2), (2, 3), (3, 16),
         (0, 4), (4, 5), (5, 6), (6, 17),
         (0, 7), (7, 8), (8, 9), (9, 18),
         (0, 10), (10, 11), (11, 12), (12, 19),
         (0, 13), (13, 14), (14, 15), (15, 20)]
PALM_IDS = [0, 1, 4, 7, 10, 13]  # wrist + 5个 MCP


def xform(pts, s, R, t):
    """p' = s * R @ p + t"""
    return s * (R @ pts.T).T + t


def compute_mano_joints(d, hand_idx):
    """MANO FK, 返回 (N,21,3) SLAM 坐标 (无效帧为 NaN)"""
    import torch
    side = H_SID[hand_idx]
    n = len(d['pred_trans'][hand_idx])
    betas_arr = d['pred_betas'][hand_idx]
    fv = next((fi for fi in range(n) if not np.any(np.isnan(betas_arr[fi]))), None)
    if fv is None:
        return np.full((n, 21, 3), np.nan)
    ml = MANOLayer(side, betas_arr[fv].astype(np.float32))
    joints = np.full((n, 21, 3), np.nan, dtype=np.float64)
    for fi in range(n):
        trans = d['pred_trans'][hand_idx, fi].astype(np.float32)
        rot = d['pred_rot'][hand_idx, fi].astype(np.float32)
        hp = d['pred_hand_pose'][hand_idx, fi].astype(np.float32)
        if np.any(np.isnan(trans)) or np.all(np.abs(trans) < 1e-6):
            continue
        if np.any(np.isnan(rot)) or np.any(np.isnan(hp)):
            continue
        with torch.no_grad():
            _, j = ml(torch.from_numpy(np.concatenate([rot, hp])).unsqueeze(0),
                      torch.from_numpy(trans).unsqueeze(0))
        jv = j[0].numpy()
        if not np.any(np.isnan(jv)):
            joints[fi] = jv
    return joints


def load_glb_geoms(ras_dir, max_faces_per_geom=6000):
    """加载 GLB 几何体, 同时保留 本地顶点 与 节点世界变换 T

    返回 list of dict(name, verts_local, tris_local, tris_world, colors)
    过滤大地面 (世界坐标下任一维 > 0.5m)
    """
    scene = trimesh.load(str(Path(ras_dir) / 'final_scene.glb'), force='scene')
    out = []
    for node_name in scene.graph.nodes_geometry:
        try:
            T, geom_key = scene.graph.get(node_name)
        except Exception:
            continue
        if geom_key is None or geom_key not in scene.geometry:
            continue
        geom = scene.geometry[geom_key]
        if not hasattr(geom, 'faces') or len(geom.vertices) == 0 or len(geom.faces) == 0:
            continue
        v = np.asarray(geom.vertices, dtype=np.float64)
        f = np.asarray(geom.faces, dtype=np.int64)
        v_world = v @ T[:3, :3].T + T[:3, 3]
        size = v_world.max(0) - v_world.min(0)
        if np.any(size > 0.5):  # 大地面等
            continue
        # 面片降采样
        nf = len(f)
        if nf > max_faces_per_geom:
            idx = np.linspace(0, nf - 1, max_faces_per_geom).astype(int)
            f_ds = f[idx]
        else:
            f_ds = f
        # 每面平均颜色
        fc = None
        vc = getattr(getattr(geom, 'visual', None), 'vertex_colors', None)
        if vc is not None and len(vc) == len(v):
            c = np.asarray(vc, dtype=np.float64)[:, :3]
            fc = c[f_ds].mean(axis=1)
            fc = np.clip(fc / 255.0, 0, 1)
        else:
            fc = np.full((len(f_ds), 3), 0.75)
        out.append({
            'name': node_name,
            'verts_local': v,
            'tris_local': v[f_ds],
            'tris_world': v_world[f_ds],
            'face_colors': fc,
        })
    return out


def shade_face_colors(tris, fc):
    """简单 Lambert 着色, 让白色物体也有立体感"""
    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    n = np.cross(v1, v2)
    nl = np.linalg.norm(n, axis=1)
    nl[nl == 0] = 1
    n = n / nl[:, None]
    light = np.array([0.4, -0.5, 0.75])
    light = light / np.linalg.norm(light)
    lam = np.abs(n @ light)
    return np.clip(fc * (0.35 + 0.65 * lam)[:, None], 0, 1)


def draw_bbox_wire(ax, tris, color, lw=1.2):
    v = tris.reshape(-1, 3)
    p0, p1 = v.min(0), v.max(0)
    c = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                  [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
    for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot3D(*c[[e[0], e[1]]].T, color=color, alpha=0.8, lw=lw)


OBJ_COLORS = plt.cm.tab10.colors


def draw_objects(ax, geoms, use_node_transform):
    """每个物体用独立纯色 + Lambert 着色 + 同色包围盒, 保证清晰可辨"""
    for i, g in enumerate(geoms):
        tris = g['tris_world'] if use_node_transform else g['tris_local']
        col = np.array(OBJ_COLORS[i % 10])[:3]
        base = np.tile(col, (len(tris), 1))
        pc = Poly3DCollection(tris, facecolors=shade_face_colors(tris, base),
                              edgecolors=(0, 0, 0, 0.18), linewidths=0.15,
                              zsort='average')
        ax.add_collection3d(pc)
        draw_bbox_wire(ax, tris, OBJ_COLORS[i % 10])
        c = tris.reshape(-1, 3).mean(axis=0)
        ax.text(c[0], c[1], c[2] + 0.02, g['name'].split('/')[-1][:16],
                fontsize=6, color=col, fontweight='bold')


def draw_hand_skeleton(ax, joints_glb, frame_ids, cmap_name='cool'):
    """在指定帧画完整手部骨架, 颜色随时间渐变"""
    cmap = plt.get_cmap(cmap_name)
    n = len(frame_ids)
    for k, fi in enumerate(frame_ids):
        J = joints_glb[fi]
        if not np.isfinite(J).all():
            continue
        color = cmap(k / max(n - 1, 1))
        for a, b in BONES:
            ax.plot(*J[[a, b]].T, color=color, lw=1.8, alpha=0.95)
        ax.scatter(J[:, 0], J[:, 1], J[:, 2], s=6, color=color, depthshade=False)
        ax.scatter(*J[0], s=60, marker='*', color=color,
                   edgecolors='k', linewidths=0.4, depthshade=False)


def plot_traj(ax, pts, color, label, lw=1.6, alpha=0.85):
    pts = np.asarray(pts)
    v = np.isfinite(pts).all(axis=1)
    if not v.any():
        return None
    pts = pts[v]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=lw, alpha=alpha, label=label)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, color=color, alpha=alpha * 0.5, marker='.')
    return pts


def draw_axes_arrows(ax, origin, scale):
    o = np.array(origin)
    for c, lbl, vec in [('red', 'X', [scale, 0, 0]),
                        ('green', 'Y', [0, scale, 0]),
                        ('blue', 'Z', [0, 0, scale])]:
        tip = o + np.array(vec)
        ax.plot([o[0], tip[0]], [o[1], tip[1]], [o[2], tip[2]], color=c, lw=4, alpha=0.9)
        ax.text(tip[0] * 1.05, tip[1] * 1.05, tip[2] * 1.05, lbl, color=c,
                fontsize=12, fontweight='bold')


def set_equal_view(ax, all_pts):
    ag = np.concatenate([p.reshape(-1, 3) for p in all_pts if len(p) > 0])
    ag = ag[np.isfinite(ag).all(axis=1)]
    lo, hi = ag.min(0), ag.max(0)
    ctr = (lo + hi) / 2
    half = max((hi - lo).max() / 2, 0.08) * 1.15
    ax.set_xlim3d(ctr[0] - half, ctr[0] + half)
    ax.set_ylim3d(ctr[1] - half, ctr[1] + half)
    ax.set_zlim3d(max(ctr[2] - half, -0.02), ctr[2] + half)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ras-dir', default='/home/an/data/ras/7_vggt_omega')
    ap.add_argument('--hawor-dir', default='/home/an/data/hawor/7')
    ap.add_argument('--params',
                    default=str(_BASE / 'output' / 'gripper_left' / 'alignment' / 'transform_params.npz'))
    ap.add_argument('--hand-idx', type=int, default=0)
    ap.add_argument('--n-skeletons', type=int, default=6)
    ap.add_argument('--output', default=str(_BASE / 'output' / 'mapping_variants'))
    args = ap.parse_args()

    ras_dir, hawor_dir = Path(args.ras_dir), Path(args.hawor_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 对齐参数 ===
    par = dict(np.load(args.params))
    s = float(par['scale_ratio'])
    R = par['R_hand_to_glb']
    t = par['t_hand_to_glb']
    Rx_hand = par.get('Rx_hand', np.diag([1, -1, -1]))
    R_rx = R @ Rx_hand
    ra = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f'[params] scale={s:.4f} rot={ra:.1f}deg')

    # === 数据 ===
    rec_file = sorted((hawor_dir / 'reconstruction').glob('hawor_results_*.npz'))[0]
    d = dict(np.load(str(rec_file), allow_pickle=True))
    hand_idx = args.hand_idx
    side_label = H_SID[hand_idx].upper()

    print('[fk] computing MANO joints ...')
    joints_slam = compute_mano_joints(d, hand_idx)      # (N,21,3)
    valid = np.isfinite(joints_slam[:, 0]).all(axis=1)
    print(f'[fk] valid frames: {valid.sum()}/{len(valid)}')

    print('[glb] loading objects ...')
    geoms = load_glb_geoms(ras_dir)
    print(f'[glb] {len(geoms)} objects')

    # 相机轨迹 (参照, 所有变体一致)
    cam_glb = xform(d['t_c2w'], s, R, t)
    ras_pos = []
    for f in sorted((ras_dir / 'extrinsics').glob('*.txt'), key=lambda x: int(x.stem)):
        ext = np.loadtxt(str(f))
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        ras_pos.append(-ext[:3, :3].T @ ext[:3, 3])
    ras_pos = np.array(ras_pos)

    # 骨架帧选取: 有效帧均匀采样
    vid = np.where(valid)[0]
    sel = np.unique(np.linspace(0, len(vid) - 1, args.n_skeletons).astype(int))
    skel_frames = vid[sel]

    # === 变体定义 ===
    VARIANTS = [
        ('A', True,  R_rx, 'obj: node-T (world) | MANO: s*R_h2g*Rx_hand+t  [render_quick chain]'),
        ('B', True,  R,    'obj: node-T (world) | MANO: s*R_h2g+t  [no Rx_hand]'),
        ('C', False, R_rx, 'obj: raw local       | MANO: s*R_h2g*Rx_hand+t  [current pipeline chain]'),
        ('D', False, R,    'obj: raw local       | MANO: s*R_h2g+t'),
    ]

    saved = []
    dist_table = []
    for tag, use_node_T, R_mano, desc in VARIANTS:
        joints_glb = xform(joints_slam.reshape(-1, 3), s, R_mano, t).reshape(joints_slam.shape)
        wrist = joints_glb[:, 0, :]
        palm = joints_glb[:, PALM_IDS, :].mean(axis=1)

        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'Variant {tag}  ({side_label} hand)\n{desc}',
                     fontsize=11, fontweight='bold')

        draw_objects(ax, geoms, use_node_T)
        draw_hand_skeleton(ax, joints_glb, skel_frames)
        plot_traj(ax, wrist, 'magenta', f'{side_label} wrist traj', lw=1.8)
        plot_traj(ax, palm, 'red', f'{side_label} palm mid traj', lw=1.4)
        plot_traj(ax, cam_glb, 'navy', 'HaWoR cam traj', lw=1.2, alpha=0.5)
        rv = np.isfinite(ras_pos).all(axis=1)
        ax.scatter(*ras_pos[rv].T, c='limegreen', s=4, label='RAS cam', alpha=0.7)
        draw_axes_arrows(ax, [0, 0, 0], scale=0.1)

        # 视野只取 物体+手 的交互区域 (相机轨迹仅作参照, 超出部分裁剪)
        all_pts = [wrist.reshape(-1, 3), palm.reshape(-1, 3),
                   joints_glb[skel_frames].reshape(-1, 3)]
        for g in geoms:
            all_pts.append(g['tris_world'] if use_node_T else g['tris_local'])
        set_equal_view(ax, all_pts)

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        plt.tight_layout()
        out_path = out_dir / f'D1_mapping_{tag}_{side_label.lower()}.png'
        plt.savefig(str(out_path), dpi=140, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_path)
        print(f'[out] {out_path}')

        # 客观距离参考: 手掌轨迹到最近物体中心的最小距离均值
        centers = []
        for g in geoms:
            vv = g['tris_world'].reshape(-1, 3) if use_node_T else g['tris_local'].reshape(-1, 3)
            centers.append(vv.mean(0))
        centers = np.array(centers)
        pv = palm[np.isfinite(palm).all(axis=1)]
        if len(pv) and len(centers):
            dmin = np.linalg.norm(pv[:, None, :] - centers[None, :, :], axis=2).min(axis=1)
            dist_table.append((tag, float(np.median(dmin)), float(dmin.min())))
            print(f'      palm->nearest-object-center: median={np.median(dmin):.4f} m, '
                  f'min={dmin.min():.4f} m')

    # === 总览拼图 ===
    import matplotlib.image as mpimg
    fig, axes = plt.subplots(2, 2, figsize=(22, 17))
    for ax, p, (tag, _, _, desc) in zip(axes.ravel(), saved, VARIANTS):
        ax.imshow(mpimg.imread(str(p)))
        ax.set_title(f'Variant {tag}', fontsize=14, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    ov_path = out_dir / f'D1_mapping_overview_{side_label.lower()}.png'
    plt.savefig(str(ov_path), dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'[out] {ov_path}')

    print('\n=== palm->nearest-object-center distance (smaller = closer to touching) ===')
    for tag, med, mn in dist_table:
        print(f'  Variant {tag}: median={med:.4f} m  min={mn:.4f} m')


if __name__ == '__main__':
    main()
