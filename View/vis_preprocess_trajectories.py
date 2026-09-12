#!/usr/bin/env python3
"""
vis_preprocess_trajectories.py — v51 §1.5 预处理对齐轨迹图 (2026-09-05)

用途: 校验"重定向前 MANO 参考点 vs 重定向后夹爪轨迹"的预处理效果 (stage0 阶段即可出图).
数据源: <output/gripper_{side}/preprocess_traj.npz> — 由 grasp_hawor.py
`_preprocess_mano_traj → _dump_preprocess_traj` 在预处理完成后自动 dump.

内容 (对齐 test9 可视化约定, 05_gripper_test L2810):
  - MANO 参考点轨迹: 红=腕 j0 / 绿=拇指尖 j4 / 蓝=食指尖 j8 (compute_mano_joints FK, 已加 pre_offset)
  - 夹爪重定向轨迹 (_mano_gripper_traj, svd_palm + smooth + pre_translate 后): 黑线 = base 位置,
    暗色三轴 = base 姿态 (每 5 帧采样)
  - GLB 物体线框 + 目标物体中心 (星标) + F50 手指中点 (验证 finger_mid ≈ obj_center)

用法:
  python View/vis_preprocess_trajectories.py --side right
  python View/vis_preprocess_trajectories.py --side left --traj output/gripper_left/preprocess_traj.npz
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GLB = '/home/an/data/ras/7_vggt_omega/final_scene.glb'
# glb_N bbox 中心 (data_loader/MuJoCo 命名) — 节点名↔glb_N 最近邻对应 (同 vis_stage_trajectories_left_v2)
_GLB_N_CTRS = {
    'glb_1': [-0.249,-0.267,0.011], 'glb_2': [-0.640,-0.366,0.025], 'glb_3': [-0.642,-0.205,0.017],
    'glb_4': [-0.302,-0.494,0.039], 'glb_5': [-0.433,-0.447,0.042], 'glb_6': [-0.426,-0.204,0.027],
    'glb_7': [-0.386,-0.635,0.018], 'glb_8': [-0.118,-0.087,0.013],
}
OBJ_CENTER_DEFAULT = np.array(_GLB_N_CTRS['glb_4'])
_FINGER_FORWARD = np.array([0.03689, 0.0, 0.0])


def load_objects():
    """GLB 物体线框 (本地顶点 Z-up 贴地, 同 v2 脚本口径)."""
    scene = trimesh.load(GLB, force='scene')
    objs = []
    for node_name in scene.graph.nodes_geometry:
        if 'grid' in node_name.lower():
            continue
        _, geom_name = scene.graph.get(node_name)
        if geom_name is None or geom_name not in scene.geometry:
            continue
        v = scene.geometry[geom_name].vertices.copy()
        center = v.mean(0)
        gname = ''
        _bn, _bc = min(_GLB_N_CTRS.items(), key=lambda kv: np.linalg.norm(np.array(kv[1]) - center))
        if np.linalg.norm(np.array(_bc) - center) < 0.08:
            gname = _bn
        objs.append((node_name, gname, center, np.array([v.min(0), v.max(0)])))
    return objs


def draw_objects(ax, alpha=0.15):
    for i, (nm, gname, ct, bd) in enumerate(load_objects()):
        c = f"C{i % 10}"
        p0, p1 = bd
        corners = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                            [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
        for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot3D(*corners[[e[0], e[1]]].T, color=c, alpha=alpha, lw=0.8)
        ax.scatter(*ct, c=c, s=40, marker='o', edgecolors='k', linewidths=0.5)
        ax.text(ct[0], ct[1], ct[2]+0.04, gname or nm, color=c, fontsize=6)


def draw_gripper_axes(ax, pos, R, length=0.03, lw=1.2):
    """夹爪 base 姿态三轴: 暗红/暗绿/暗蓝 (与 MANO 三色锚区分)."""
    for k, (c) in enumerate(['#8B0000', '#006400', '#00008B']):
        d = R[:, k] * length
        ax.plot([pos[0], pos[0]+d[0]], [pos[1], pos[1]+d[1]], [pos[2], pos[2]+d[2]],
                color=c, lw=lw, alpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', default='right', choices=['right', 'left'])
    ap.add_argument('--traj', default=None, help='preprocess_traj.npz 路径 (默认 output/gripper_<side>/)')
    ap.add_argument('--out', default=None, help='输出 PNG (默认 output/gripper_<side>/stage0/)')
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__)) + '/..'
    traj_path = args.traj or f"{root}/output/gripper_{args.side}/preprocess_traj.npz"
    if not os.path.exists(traj_path):
        print(f"✗ 未找到 {traj_path} — 先跑一次 --test-stage0 (预处理会自动 dump)")
        sys.exit(1)
    d = np.load(traj_path, allow_pickle=True)
    grip_pos = d['grip_pos']
    grip_R = d['grip_R']
    N = len(grip_pos)
    side = str(d['side']) if 'side' in d else args.side
    method = str(d['grasp_pose_method']) if 'grasp_pose_method' in d else '?'
    obj_center = d['obj_center'] if 'obj_center' in d else OBJ_CENTER_DEFAULT
    pre_off = d['pre_offset'] if 'pre_offset' in d else np.zeros(3)
    joints = d['joints'] if 'joints' in d else None
    print(f"载入 {traj_path}: N={N}, side={side}, method={method}, |pre_offset|={np.linalg.norm(pre_off):.4f}m")

    F50 = min(50, N - 1)
    finger_mid_f50 = grip_pos[F50] + grip_R[F50] @ _FINGER_FORWARD
    fm_err = float(np.linalg.norm(finger_mid_f50 - obj_center))
    print(f"F50 finger_mid={finger_mid_f50.round(4)}, obj_center={np.asarray(obj_center).round(4)}, 偏差={fm_err*1000:.1f}mm")

    fig = plt.figure(figsize=(24, 10.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for a in (ax, ax2):
        draw_objects(a)

    # ── 四条轨迹 (2026-09-06 用户指定): MANO 腕点平滑前/后 + 夹爪 wrist + SLAM 相机 ──
    joints_raw = d['joints_raw'] if 'joints_raw' in d else None
    _off_v = np.asarray(pre_off, dtype=np.float64)
    if joints_raw is not None:
        _w_raw = joints_raw[:, 0, :] + _off_v  # MANO 腕点 (平滑前, +pre_offset 与平滑后同系)
        ax.plot(_w_raw[:, 0], _w_raw[:, 1], _w_raw[:, 2], '--',
                color='orange', lw=1.4, alpha=0.6, label='MANO wrist raw (平滑前)')
    if joints is not None:
        _w_sm = joints[:, 0, :]  # MANO 腕点 (平滑后, 已平移)
        ax.plot(_w_sm[:, 0], _w_sm[:, 1], _w_sm[:, 2], '-',
                color='orange', lw=1.8, alpha=0.85, label='MANO wrist smooth (平滑后)')
    cam_traj = d['cam_traj'] if 'cam_traj' in d else None
    if cam_traj is not None:
        ax.plot(cam_traj[:, 0], cam_traj[:, 1], cam_traj[:, 2], '-',
                color='c', lw=1.8, alpha=0.8, label='SLAM camera trajectory')
        ax.scatter(*cam_traj[0], c='c', s=50, marker='o', edgecolors='k', linewidths=0.6)
    else:
        print("⚠ npz 无 cam_traj (旧版 dump), 跳过相机轨迹")

    # ── 夹爪重定向轨迹: base 位置 (黑) + 姿态三轴 (暗色, 每5帧) ──
    ax.plot(grip_pos[:, 0], grip_pos[:, 1], grip_pos[:, 2], '-', color='k', lw=1.8,
            alpha=0.9, label='Gripper base (svd_palm+smooth+pre_translate)')
    for f in range(0, N, 5):
        draw_gripper_axes(ax, grip_pos[f], grip_R[f])
    draw_gripper_axes(ax, grip_pos[F50], grip_R[F50], length=0.05, lw=2.0)
    ax.scatter(*grip_pos[F50], c='gold', s=120, marker='*', edgecolors='k',
               linewidths=1.0, zorder=9, label='F50 base')
    # F50 手指中点 (base +X 前方 3.689cm) — 应落在物体中心
    ax.scatter(*finger_mid_f50, c='lime', s=90, marker='X', edgecolors='k',
               linewidths=0.8, zorder=9, label='F50 finger_mid')
    ax.scatter(*np.asarray(obj_center), c='m', s=160, marker='p', edgecolors='k',
               linewidths=1.0, zorder=10, label='obj_center')

    ax.set_xlabel('GLB X (m)'); ax.set_ylabel('GLB Y (m)'); ax.set_zlabel('GLB Z (m, up)')
    ax.set_title(f'Left: 预处理对齐总览 — MANO 腕点(橙虚=平滑前/橙实=平滑后) + 夹爪wrist(黑) + 相机轨迹(青), side={side}, {method}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.92)
    ax.view_init(elev=20, azim=45)
    ax.set_box_aspect([1, 1, 0.8])

    # ── 右图: F50 物体附近特写 (验证 finger_mid ≈ obj_center) ──
    c0 = np.asarray(obj_center)
    pad = 0.12
    if joints is not None:
        seg_w = joints[max(0, F50-15):F50+1, 0, :]
        ax2.plot(seg_w[:, 0], seg_w[:, 1], seg_w[:, 2], '-', color='orange', lw=2.0, alpha=0.9)
    seg = grip_pos[max(0, F50-15):F50+1]
    ax2.plot(seg[:, 0], seg[:, 1], seg[:, 2], '-', color='k', lw=1.8)
    for k, f in enumerate(range(max(0, F50-14), F50+1, 2)):
        draw_gripper_axes(ax2, grip_pos[f], grip_R[f], length=0.025, lw=1.0)
    ax2.scatter(*grip_pos[F50], c='gold', s=130, marker='*', edgecolors='k', zorder=9)
    ax2.scatter(*finger_mid_f50, c='lime', s=100, marker='X', edgecolors='k', zorder=9)
    ax2.scatter(*c0, c='m', s=170, marker='p', edgecolors='k', zorder=10)
    ax2.set_xlim(c0[0]-pad, c0[0]+pad)
    ax2.set_ylim(c0[1]-pad, c0[1]+pad)
    _zs = np.vstack([seg, c0.reshape(1, 3)] +
                    ([joints[max(0, F50-15):F50+1].reshape(-1, 3)] if joints is not None else []))
    ax2.set_zlim(max(0.0, float(_zs[:, 2].min()) - 0.02), float(_zs[:, 2].max()) + 0.02)
    ax2.set_xlabel('GLB X (m)'); ax2.set_ylabel('GLB Y (m)'); ax2.set_zlabel('GLB Z (m, up)')
    ax2.set_title(f'Right: F50 特写 — finger_mid 偏差 {fm_err*1000:.1f}mm (绿X vs 品红五角星), 黑虚=优化前wrist',
                  fontsize=11, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    ax2.set_box_aspect([1, 1, 0.8])

    fig.text(0.5, 0.012,
             f"v51 §1.5 预处理轨迹图 | side={side}, method={method}, N={N}, "
             f"|pre_offset|={np.linalg.norm(pre_off):.4f}m | F50 finger_mid↔obj_center 偏差 {fm_err*1000:.1f}mm | "
             "黑实=优化后 wrist, 黑虚=优化前 wrist, 青=SLAM 相机轨迹, 暗三轴=base 姿态",
             ha='center', fontsize=8, color='#333333')

    out_png = args.out or f"{root}/output/gripper_{args.side}/stage0/preprocess_traj.png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"✓ 已保存: {out_png}")


if __name__ == '__main__':
    main()
