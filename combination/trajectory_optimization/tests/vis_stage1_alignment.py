#!/usr/bin/env python3
"""可视化 Stage 1 对齐: MANO 手部轨迹 + GLB 场景 + Grasp Pose

展示所有数据在 SAPIEN 坐标系下的对齐情况。
"""
import argparse, sys, warnings, os
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

# ── 路径设置 ──
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# ── 坐标变换 ──
RXWORLD_TO_SAPIEN = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

def xform(pts, s, R, t):
    return s * (R @ pts.T).T + t

def load_transform_params(params_path):
    """加载 transform_params.npz, 返回 (R_hand_to_glb, t_hand_to_glb, scale, s_inv, R_inv, t_inv)"""
    p = np.load(str(params_path))
    R_htg = p.get('R_hand_to_glb', p.get('R', np.eye(3)))
    t_htg = p.get('t_hand_to_glb', p.get('t', np.zeros(3)))
    scale = float(p.get('scale_ratio', p.get('scale', 1.0)))
    s_inv = float(p.get('s_inv', 1.0 / scale))
    R_inv = p.get('R_inv', R_htg.T)
    t_inv = p.get('t_inv', -R_inv @ t_htg)
    return R_htg, t_htg, scale, s_inv, R_inv, t_inv

def hawor_to_sapien(pts, s_inv, R_inv, t_inv):
    """HaWoR 坐标 → SAPIEN 坐标"""
    pts_sapien = s_inv * (R_inv @ (pts - t_inv).T).T
    pts_sapien = (RXWORLD_TO_SAPIEN @ pts_sapien.T).T
    return pts_sapien

def glb_to_sapien(pts, R_htg, t_htg, scale, s_inv, R_inv, t_inv):
    """GLB 坐标 → SAPIEN 坐标"""
    pts_hawor = (pts / scale - t_htg) @ R_htg.T  # GLB → HaWoR
    return hawor_to_sapien(pts_hawor, s_inv, R_inv, t_inv)

def load_mano_traj(hawor_dir):
    """加载 MANO 轨迹数据"""
    rec_dir = Path(hawor_dir) / "reconstruction"
    if not rec_dir.exists():
        return None, None
    files = sorted(rec_dir.glob("hawor_results_*.npz"))
    if not files:
        return None, None
    d = dict(np.load(str(files[0]), allow_pickle=True))
    pos_traj = np.array(d.get('pred_trans', []))
    if len(pos_traj) == 0:
        return None, None
    # pos_traj shape: [n_hands, n_frames, 3]
    return pos_traj, d

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', type=str, default=None,
                        help='HaWoR 数据目录 (默认自动查找)')
    parser.add_argument('--ras-dir', type=str, default=None,
                        help='RAS 数据目录 (默认自动查找)')
    parser.add_argument('--params', type=str, default=None,
                        help='transform_params.npz 路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出前缀')
    parser.add_argument('--best-grasp', type=str, default=None,
                        help='best_grasp.npz 路径')
    args = parser.parse_args()

    # ── 自动查找数据目录 ──
    data_root = Path("/home/an/data")
    hawor_dir = Path(args.hawor_dir) if args.hawor_dir else (data_root / "hawor" / "7")
    ras_dir = Path(args.ras_dir) if args.ras_dir else (data_root / "ras" / "my_7mp4_result")

    if not hawor_dir.exists():
        # 尝试其他路径
        for p in sorted(data_root.glob("hawor/*")):
            if p.is_dir():
                hawor_dir = p
                break
    if not ras_dir.exists():
        for p in sorted(data_root.glob("ras/*")):
            if p.is_dir():
                ras_dir = p
                break

    # ── 变换参数 ──
    params_path = Path(args.params) if args.params else (
        SCRIPT_DIR / "output" / "gripper_only_right" / "alignment" / "transform_params.npz")
    if not params_path.exists():
        params_path = Path(args.params) if args.params else (
            SCRIPT_DIR / "output" / "gripper_only_right" / "stage1_demo" / "alignment" / "transform_params.npz")
    print(f"变换参数: {params_path} (存在={params_path.exists()})")
    R_htg, t_htg, scale, s_inv, R_inv, t_inv = load_transform_params(params_path)
    print(f"  scale={scale:.4f}, s_inv={s_inv:.4f}")

    # ── GLB 场景 ──
    glb_path = ras_dir / "final_scene.glb"
    glb_geoms = []
    if glb_path.exists():
        try:
            import trimesh
            scene = trimesh.load(str(glb_path))
            for name, geom in scene.geometry.items():
                verts = geom.vertices
                center = verts.mean(0)
                bounds = geom.bounds  # (2, 3)
                # 变换到 SAPIEN 坐标
                center_sapien = glb_to_sapien(center[None], R_htg, t_htg, scale, s_inv, R_inv, t_inv)[0]
                bounds_sapien = glb_to_sapien(bounds, R_htg, t_htg, scale, s_inv, R_inv, t_inv)
                glb_geoms.append((name, center_sapien, bounds_sapien))
                print(f"  GLB物体: {name}, center={center_sapien.round(4)}, bounds_z=[{bounds_sapien[:,2].min():.3f},{bounds_sapien[:,2].max():.3f}]")
        except Exception as e:
            print(f"  GLB加载失败: {e}")
    else:
        print(f"  GLB未找到: {glb_path}")

    # ── MANO 手部轨迹 ──
    mano_pos_traj, mano_data = load_mano_traj(hawor_dir)
    mano_hands = {}
    if mano_pos_traj is not None:
        n_hands = min(2, len(mano_pos_traj))
        for hidx in range(n_hands):
            pts = np.array(mano_pos_traj[hidx])
            pts_sapien = hawor_to_sapien(pts, s_inv, R_inv, t_inv)
            valid = ~np.any(np.isnan(pts_sapien), axis=1) & (pts_sapien[:, 2] >= -0.5)
            if valid.any():
                mano_hands[hidx] = pts_sapien[valid]
                print(f"  MANO 手{hidx}: {valid.sum()} 帧, z=[{pts_sapien[valid,2].min():.3f},{pts_sapien[valid,2].max():.3f}]")

    # ── MANO 相机轨迹 ──
    cam_glb = None
    if mano_data is not None:
        t_c2w = np.array(mano_data.get('t_c2w', []))
        if len(t_c2w) > 0:
            cam_glb = hawor_to_sapien(t_c2w, s_inv, R_inv, t_inv)
            print(f"  MANO 相机: {len(cam_glb)} 帧, z=[{cam_glb[:,2].min():.3f},{cam_glb[:,2].max():.3f}]")

    # ── RAS 相机轨迹 ──
    ras_pos = None
    ext_dir = ras_dir / "extrinsics"
    if ext_dir.exists():
        files = sorted(ext_dir.glob("*.txt"), key=lambda x: int(x.stem))
        ras_pos_raw = []
        for f in files:
            ext = np.loadtxt(str(f))
            if ext.shape == (3, 4):
                ext = np.vstack([ext, [0, 0, 0, 1]])
            R_w2c, t_w2c = ext[:3, :3], ext[:3, 3]
            R_c2w = R_w2c.T
            ras_pos_raw.append(-R_c2w @ t_w2c)
        if ras_pos_raw:
            ras_glb = np.array(ras_pos_raw)
            ras_sapien = glb_to_sapien(ras_glb, R_htg, t_htg, scale, s_inv, R_inv, t_inv)
            ras_pos = ras_sapien
            print(f"  RAS 相机: {len(ras_pos)} 帧, z=[{ras_pos[:,2].min():.3f},{ras_pos[:,2].max():.3f}]")

    # ── Stage 1 Grasp Pose ──
    best_grasp_path = Path(args.best_grasp) if args.best_grasp else (
        SCRIPT_DIR / "output" / "gripper_only_right" / "stage1" / "best_grasp.npz")
    grasp_pos = None
    grasp_euler = None
    obj_lift = 0
    if best_grasp_path.exists():
        g = np.load(str(best_grasp_path))
        grasp_pos = g.get('pos', np.zeros(3))
        grasp_euler = g.get('euler', np.zeros(3))
        obj_lift = float(g.get('obj_lift', 0))
        print(f"  Grasp Pose: pos={grasp_pos.round(4)}, euler={np.degrees(grasp_euler).round(2) if isinstance(grasp_euler, np.ndarray) else grasp_euler}, lift={obj_lift*100:.1f}cm")

    # ── Stage 1 rollout 轨迹 ──
    rollout_data = None
    rollout_path = SCRIPT_DIR / "output" / "gripper_only_right" / "stage1"
    run_z_path = rollout_path / "run_gripper_z_traj.npy"
    obj_z_path = rollout_path / "run_obj_z_traj.npy"
    if run_z_path.exists():
        gripper_z = np.load(str(run_z_path))
    else:
        gripper_z = None
    if obj_z_path.exists():
        obj_z = np.load(str(obj_z_path))
    else:
        obj_z = None
    if gripper_z is not None and obj_z is not None:
        print(f"  Rollout: {len(gripper_z)} 帧, gripper_z=[{gripper_z.min():.3f},{gripper_z.max():.3f}]")

    # ── 创建 MANO F50 位姿 ──
    mano_f50_pos = None
    if mano_pos_traj is not None and len(mano_pos_traj) > 0 and len(mano_pos_traj[0]) > 50:
        f50 = np.array(mano_pos_traj[0][50])
        mano_f50_pos = hawor_to_sapien(f50[None], s_inv, R_inv, t_inv)[0]
        print(f"  MANO F50: pos={mano_f50_pos.round(4)}")

    # ================================================================
    # 绘图
    # ================================================================
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    output_prefix = args.output or str(SCRIPT_DIR / "output" / "gripper_only_right" / "stage1_alignment")

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Stage 1 Alignment: MANO + GLB + Grasp Pose (SAPIEN coords)', fontweight='bold', fontsize=13)

    # 坐标轴
    origin = np.zeros(3)
    axis_len = 0.2
    for c, lbl, v in [('red', 'X', [axis_len, 0, 0]), ('green', 'Y', [0, axis_len, 0]), ('blue', 'Z', [0, 0, axis_len])]:
        tip = origin + v
        ax.plot([origin[0], tip[0]], [origin[1], tip[1]], [origin[2], tip[2]], color=c, lw=3, alpha=0.8)
        ax.text(tip[0], tip[1], tip[2], lbl, color=c, fontsize=12, fontweight='bold')

    # GLB 物体 bbox
    for i, (nm, ct, bd) in enumerate(glb_geoms):
        col = plt.cm.tab10(i % 10)
        p0, p1 = bd[0], bd[1]
        corners = np.array([[p0[0], p0[1], p0[2]], [p1[0], p0[1], p0[2]], [p1[0], p1[1], p0[2]], [p0[0], p1[1], p0[2]],
                            [p0[0], p0[1], p1[2]], [p1[0], p0[1], p1[2]], [p1[0], p1[1], p1[2]], [p0[0], p1[1], p1[2]]])
        for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot3D(*corners[[e[0], e[1]]].T, color=col, alpha=0.3, lw=0.8)
        ax.scatter(*ct, c=col, s=40, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
        ax.text(ct[0], ct[1], ct[2]+0.01, f'obj{i}', color=col, fontsize=7)

    # MANO 手部轨迹
    for hidx, col, lbl in [(0, 'green', 'Left Hand'), (1, 'gold', 'Right Hand')]:
        if hidx in mano_hands:
            pts = mano_hands[hidx]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=col, lw=1.0, alpha=0.7, label=lbl)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.0, color=col, alpha=0.3)

    # MANO F50 标记
    if mano_f50_pos is not None:
        ax.scatter(*mano_f50_pos, c='black', s=100, marker='*', zorder=12, label='MANO F50')
        ax.text(mano_f50_pos[0], mano_f50_pos[1], mano_f50_pos[2]+0.01, 'F50', fontsize=9, fontweight='bold')

    # MANO 相机
    if cam_glb is not None:
        ax.plot(cam_glb[:, 0], cam_glb[:, 1], cam_glb[:, 2], 'b-', lw=1.5, alpha=0.5, label='HaWoR cam')

    # RAS 相机
    if ras_pos is not None:
        ax.plot(ras_pos[:, 0], ras_pos[:, 1], ras_pos[:, 2], '-', color='orange', lw=1.5, alpha=0.8, label='RAS cam')

    # Grasp Pose
    if grasp_pos is not None:
        ax.scatter(*grasp_pos, c='cyan', s=150, marker='D', edgecolors='k', linewidths=1, zorder=15, label='Grasp Pose')
        ax.text(grasp_pos[0], grasp_pos[1], grasp_pos[2]+0.01, f'Grasp (lift={obj_lift*100:.1f}cm)', fontsize=8, color='cyan')

    # Rollout 轨迹
    if gripper_z is not None and obj_z is not None:
        # 从 best_grasp 获取 xy 位置
        if grasp_pos is not None:
            gripper_xyz = np.zeros((len(gripper_z), 3))
            gripper_xyz[:, 0] = grasp_pos[0]
            gripper_xyz[:, 1] = grasp_pos[1]
            gripper_xyz[:, 2] = gripper_z
            ax.plot(gripper_xyz[:, 0], gripper_xyz[:, 1], gripper_xyz[:, 2], 'c-', lw=2.0, alpha=0.8, label='Gripper Z (rollout)')
            ax.scatter(gripper_xyz[:, 0], gripper_xyz[:, 1], gripper_xyz[:, 2], s=3, color='cyan', alpha=0.5)

            # 物体轨迹
            obj_xyz = np.zeros((len(obj_z), 3))
            obj_xyz[:, 0] = grasp_pos[0]
            obj_xyz[:, 1] = grasp_pos[1]
            obj_xyz[:, 2] = obj_z
            ax.plot(obj_xyz[:, 0], obj_xyz[:, 1], obj_xyz[:, 2], 'm--', lw=1.5, alpha=0.8, label='Obj Z (rollout)')

    # 自动范围
    all_pts = []
    for nm, ct, bd in glb_geoms:
        all_pts.append(bd.reshape(-1, 3))
        all_pts.append(ct.reshape(1, 3))
    for hidx in [0, 1]:
        if hidx in mano_hands:
            all_pts.append(mano_hands[hidx])
    if cam_glb is not None:
        all_pts.append(cam_glb)
    if ras_pos is not None:
        all_pts.append(ras_pos)
    if grasp_pos is not None:
        all_pts.append(grasp_pos.reshape(1, 3))
    if gripper_z is not None:
        all_pts.append(gripper_xyz)

    all_pts = [p for p in all_pts if len(p) > 0]
    if all_pts:
        ag = np.concatenate([p for p in all_pts if len(p) > 0], axis=0)
        ag = ag[np.isfinite(ag).all(axis=1)]
        if len(ag) > 0:
            lo, hi = ag.min(0), ag.max(0)
            rng = np.maximum(hi - lo, 0.3)
            pad = rng * 0.15
            ax.set_xlim3d(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim3d(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_zlim3d(max(lo[2] - pad[2], -0.1), hi[2] + pad[2])

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    plt.tight_layout()

    png_path = f"{output_prefix}.png"
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"\n保存: {png_path}")
    plt.close()

    # ── Plotly HTML ──
    try:
        import plotly.graph_objects as go

        fig_p = go.Figure()

        # 坐标轴
        for c, lbl, v in [('red', 'X', [axis_len, 0, 0]), ('green', 'Y', [0, axis_len, 0]), ('blue', 'Z', [0, 0, axis_len])]:
            fig_p.add_trace(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
                mode='lines+text', line=dict(color=c, width=6),
                text=['', f' {lbl}'], textfont=dict(color=c, size=16), showlegend=False))

        # GLB 物体
        for i, (nm, ct, bd) in enumerate(glb_geoms):
            col = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'][i % 5]
            p0, p1 = bd[0], bd[1]
            corners = np.array([[p0[0], p0[1], p0[2]], [p1[0], p0[1], p0[2]], [p1[0], p1[1], p0[2]], [p0[0], p1[1], p0[2]],
                                [p0[0], p0[1], p1[2]], [p1[0], p0[1], p1[2]], [p1[0], p1[1], p1[2]], [p0[0], p1[1], p1[2]]])
            for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
                fig_p.add_trace(go.Scatter3d(x=corners[e, 0], y=corners[e, 1], z=corners[e, 2],
                    mode='lines', line=dict(color=col, width=2), showlegend=False))

        # MANO 手部
        for hidx, col, lbl in [(0, 'rgba(0,255,0,0.7)', 'Left Hand'), (1, 'rgba(255,215,0,0.7)', 'Right Hand')]:
            if hidx in mano_hands:
                pts = mano_hands[hidx]
                fig_p.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='lines+markers', marker=dict(size=1.5, color=col),
                    line=dict(color=col, width=1.5), name=lbl))

        # MANO F50
        if mano_f50_pos is not None:
            fig_p.add_trace(go.Scatter3d(x=[mano_f50_pos[0]], y=[mano_f50_pos[1]], z=[mano_f50_pos[2]],
                mode='markers+text', marker=dict(size=10, color='black', symbol='diamond'),
                text=['F50'], textfont=dict(size=12, color='black'), name='MANO F50'))

        # 相机
        if cam_glb is not None:
            fig_p.add_trace(go.Scatter3d(x=cam_glb[:, 0], y=cam_glb[:, 1], z=cam_glb[:, 2],
                mode='lines', line=dict(color='blue', width=2), name='HaWoR cam'))
        if ras_pos is not None:
            fig_p.add_trace(go.Scatter3d(x=ras_pos[:, 0], y=ras_pos[:, 1], z=ras_pos[:, 2],
                mode='lines', line=dict(color='orange', width=2.5), name='RAS cam'))

        # Grasp Pose
        if grasp_pos is not None:
            fig_p.add_trace(go.Scatter3d(x=[grasp_pos[0]], y=[grasp_pos[1]], z=[grasp_pos[2]],
                mode='markers+text', marker=dict(size=12, color='cyan', symbol='diamond'),
                text=[f'Grasp ({obj_lift*100:.1f}cm)'], textfont=dict(size=10, color='cyan'),
                name='Grasp Pose'))

        # Rollout
        if gripper_z is not None:
            fig_p.add_trace(go.Scatter3d(x=gripper_xyz[:, 0], y=gripper_xyz[:, 1], z=gripper_xyz[:, 2],
                mode='lines+markers', marker=dict(size=2, color='cyan'),
                line=dict(color='cyan', width=2), name='Gripper (rollout)'))
            fig_p.add_trace(go.Scatter3d(x=obj_xyz[:, 0], y=obj_xyz[:, 1], z=obj_xyz[:, 2],
                mode='lines+markers', marker=dict(size=2, color='magenta'),
                line=dict(color='magenta', width=1.5, dash='dash'), name='Object (rollout)'))

        fig_p.update_layout(title=f'Stage 1 Alignment: MANO + GLB + Grasp (lift={obj_lift*100:.1f}cm)',
            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
            width=1200, height=900)

        html_path = f"{output_prefix}.html"
        fig_p.write_html(html_path)
        print(f"保存: {html_path}")
    except Exception as e:
        print(f"Plotly 保存失败: {e}")

    print("\nDone.")

if __name__ == '__main__':
    main()