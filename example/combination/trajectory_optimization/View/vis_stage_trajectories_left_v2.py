#!/usr/bin/env python3
"""
vis_stage_trajectories_left_v2.py — GLB space, matching render_quick.py exactly.
Shows: GLB objects + Original MANO + Stage 1/2/3 trajectories.
Everything in GLB world coordinates (Z-up).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Zen Hei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import torch

import sys
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/combination')
sys.path.insert(0, '/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths (v5 2026-09-05: side 可由命令行首参指定, 默认 left 兼容旧用法) ──
SIDE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('left', 'right') else 'left'
_ROOT = '/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization'
GLB = '/home/an/data/ras/7_vggt_omega/final_scene.glb'
TRANSFORM = f'{_ROOT}/output/hawor7a_{SIDE}/alignment/transform_params.npz'
OUT_DIR = f'{_ROOT}/output/hawor7a_{SIDE}'
OUT_PNG = f'{_ROOT}/output/hawor7a_{SIDE}/D1_trajectory_comparison_mint.png'

# ── Transform params (same as render_quick.py) ──
p = np.load(TRANSFORM)
SCALE = float(p['scale_ratio'])          # 3.168
R_H2G = p['R_hand_to_glb']               # 3x3
T_H2G = p['t_hand_to_glb'].ravel()       # 3
# v5 2026-09-05: 坐标链修复 (v51 §1.1) — 弃用固定 Rx_hand=diag(1,-1,-1)
# (001_align_scene 实验证实 R_X=I, 固定近似在相机转动后漂移 34°/13cm),
# 改用与主管线一致的逐帧真实相机位姿链:
#   p_hw = R_c2w[f] @ p_cam + t_c2w[f]  →  p_glb = s * (R_h2g @ p_hw) + t_h2g
from hand_track.common import load_hawor_c2w
R_c2w_all, t_c2w_all = load_hawor_c2w('/home/an/data/hawor/7a')

def xform(pts, s, R, t):
    """Same as render_quick.py: s * (R @ pts.T).T + t"""
    return s * (R @ pts.T).T + t

# ═══════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════

# ── GLB objects (SAME as render_quick.py: 顶点经节点变换到 GLB 世界坐标) ──
# 关键: final_scene.glb 节点变换非恒等 (R_node: Y-up 本地 → Z-up 世界),
#       必须应用 T = 节点世界变换, 物体才能落在与 MANO 一致的 GLB 世界.
#       之前 D1 改成 "不取节点变换" (用 geom.vertices 原始坐标),
#       等于把物体留在 Y-up 本地系, 缺了一次 "方向旋转", 看起来手像缺旋转.
#       这就是 render_quick.py 正确版本的做法, 这里恢复对齐.
scene = trimesh.load(GLB, force='scene')
# v4.33: glb_N (data_loader/MuJoCo 命名) bbox 中心 — 用于 节点名↔glb_N 最近邻对应
_GLB_N_CTRS = {
    'glb_1': [-0.249,-0.267,0.011], 'glb_2': [-0.640,-0.366,0.025], 'glb_3': [-0.642,-0.205,0.017],
    'glb_4': [-0.302,-0.494,0.039], 'glb_5': [-0.433,-0.447,0.042], 'glb_6': [-0.426,-0.204,0.027],
    'glb_7': [-0.386,-0.635,0.018], 'glb_8': [-0.118,-0.087,0.013],
}
geoms = []
for node_name in scene.graph.nodes_geometry:
    T, geom_name = scene.graph.get(node_name)
    if geom_name is None or geom_name not in scene.geometry:
        continue
    geom = scene.geometry[geom_name]
    # v4.31 [再修复] 直接用本地顶点 (Z-up 重建世界系, z_min=0 贴地), 与 MANO 映射同系.
    # 节点旋转 R=[x,z,-y] 只是 GLB 导出的 Y-up 存储标记, 应用它会把物体抬到 z=0.2~0.65 悬空.
    v_world = geom.vertices.copy()
    center = v_world.mean(0)
    bounds = np.array([v_world.min(0), v_world.max(0)])
    # v4.33 [颜色修复] 提取真实材质颜色 (vertex_colors 优先, 退回 material.diffuse).
    # 之前用 matplotlib 调色板按顺序上色, 把粉色目标 toy_0 画成橙色、橙色 block_1 画成粉色, 严重误导.
    rgb = None
    _vis = getattr(geom, 'visual', None)
    if _vis is not None:
        _vc = getattr(_vis, 'vertex_colors', None)
        if _vc is not None and len(_vc) > 0:
            rgb = np.array(_vc)[:, :3].mean(0) / 255.0
        else:
            _mat = getattr(_vis, 'material', None)
            _col = getattr(_mat, 'diffuse', None) if _mat is not None else None
            if _col is not None:
                _col = np.array(_col[:3], dtype=float)
                rgb = _col / 255.0 if _col.max() > 1.5 else _col
    # 节点名 ↔ glb_N (最近邻匹配, 阈值 8cm)
    gname = ''
    if 'grid' not in node_name.lower():
        _bn, _bc = min(_GLB_N_CTRS.items(), key=lambda kv: np.linalg.norm(np.array(kv[1]) - center))
        if np.linalg.norm(np.array(_bc) - center) < 0.08:
            gname = _bn
    geoms.append((node_name, center, bounds, rgb, gname))
    print(f"  Object {node_name} [{gname}]: rgb={None if rgb is None else rgb.round(3)}, center={center.round(4)}")

# ── HaWoR MANO reference (same as render_quick.py: xform with s, R_hand, t) ──
# v5 2026-09-05: hand_idx 随 SIDE (0=left 真实操作手 / 1=right, 见 vis_left_hand_3d_trajectory 注)
HAND_IDX = 0 if SIDE == 'left' else 1
d = np.load('/home/an/data/hawor/7a/reconstruction/hawor_results_0_113.npz', allow_pickle=True)
v = d['pred_valid'][HAND_IDX] & ~np.isnan(d['pred_trans'][HAND_IDX]).any(1) & ~np.isnan(d['pred_rot'][HAND_IDX]).any(1)
trans_v = d['pred_trans'][HAND_IDX][v]
rot_v = d['pred_rot'][HAND_IDX][v]
hp_v = d['pred_hand_pose'][HAND_IDX][v]
betas = d['pred_betas'][HAND_IDX][v].mean(0) if v.any() else np.zeros(10)
print(f"HaWoR MANO: {v.sum()} valid frames")

from mano_layer import MANOLayer
ml = MANOLayer(SIDE, betas=betas)
mano_palm = []
for i in range(len(trans_v)):
    r = torch.from_numpy(rot_v[i:i+1].astype(np.float32))
    hp = torch.from_numpy(hp_v[i:i+1].astype(np.float32))
    t = torch.from_numpy(trans_v[i:i+1].astype(np.float32))
    p = torch.cat([r, hp], dim=1)
    vo = ml(p, t)
    if isinstance(vo, (tuple, list)): vo = vo[0]
    vn = vo.detach().cpu().numpy()
    mano_palm.append(vn[:, 745, :] if vn.shape[1] >= 745 else vn[:, 0, :])
mano_palm = np.concatenate(mano_palm, axis=0)  # HaWoR SLAM

# v5 2026-09-05: 逐帧 c2w 链 (R_X=I) — 与主管线 _mano_to_sapien 完全一致
_orig_idx = np.where(v)[0]
_n_c2w = len(R_c2w_all)
mano_palm_glb = np.stack([
    SCALE * (R_H2G @ (R_c2w_all[min(fi, _n_c2w - 1)] @ mano_palm[i]
                      + t_c2w_all[min(fi, _n_c2w - 1)])) + T_H2G
    for i, fi in enumerate(_orig_idx)
])
print(f"  Original MANO palm GLB: first={mano_palm_glb[0].round(4)}, last={mano_palm_glb[-1].round(4)}")
print(f"  Y range: {mano_palm_glb[:,1].min():.4f} ~ {mano_palm_glb[:,1].max():.4f}")

# ── Stage 1/2/3 data (SAPIEN -> GLB world) ──
# [修复 2026-08-23] 实测 final_scene.glb 为 Z-up, 且管线以 glb_coords=True 运行:
# SAPIEN 世界 = GLB 坐标系 (恒等映射), 显示无需任何变换.
# (旧 R_AXIS.T 是把 GLB 误当 Y-up 的产物, 造成 ~0.46m 假偏移)
s_inv = 1.0 / SCALE

def mano_sapien_to_glb(pts):
    """glb_coords=True: SAPIEN == GLB, 恒等."""
    return np.asarray(pts)

def sapien_to_glb(pts):
    """glb_coords=True: SAPIEN == GLB, 恒等."""
    return np.asarray(pts)

# Stage 1 grasp (in data_loader space)
s1 = np.load(os.path.join(OUT_DIR, 'stage1', 'best_grasp.npz'), allow_pickle=True)
s1_pos = np.array(s1['pos'])
s1_glb = sapien_to_glb(s1_pos)
print(f"Stage1 grasp GLB: {s1_glb.round(4)}")

# Stage 2 lift (in data_loader space)
s2 = np.load(os.path.join(OUT_DIR, 'stage2', 'stage2_recon.npz'), allow_pickle=True)
s2_pos = s2['pos']
s2_frames = s2['frames']
s2_glb = sapien_to_glb(s2_pos)
print(f"Stage2 GLB: {s2_glb.shape}, F{s2_frames[0]}-F{s2_frames[-1]}")
print(f"  Y range: {s2_glb[:,1].min():.4f} ~ {s2_glb[:,1].max():.4f}")

# Stage 3 data (optional — may not exist for all sides)
try:
    s3 = np.load(os.path.join(OUT_DIR, 'stage3', 'stage3_result.npz'), allow_pickle=True)
    r3 = s3['result'].item()
    s3_mano = np.array(r3['mano_pos'])
    s3_opt = np.array(r3['opt_pos'])
    s3_obj = np.array(r3['obj_pos_traj'])
    s3_mano_glb = mano_sapien_to_glb(s3_mano)
    s3_opt_glb = sapien_to_glb(s3_opt)
    s3_obj_glb = sapien_to_glb(s3_obj)
    has_s3 = True
    print(f"Stage3 mano GLB: {s3_mano_glb.shape}, Y range: {s3_mano_glb[:,1].min():.4f} ~ {s3_mano_glb[:,1].max():.4f}")
    print(f"Stage3 opt GLB: {s3_opt_glb.shape}, Y range: {s3_opt_glb[:,1].min():.4f} ~ {s3_opt_glb[:,1].max():.4f}")
    print(f"Stage3 object GLB: {s3_obj_glb.shape}, Y range: {s3_obj_glb[:,1].min():.4f} ~ {s3_obj_glb[:,1].max():.4f}")
    print(f"  --- MANO comparison (should be close) ---")
    print(f"  Original MANO palm GLB:  first={mano_palm_glb[0].round(4)}, Y=[{mano_palm_glb[:,1].min():.3f},{mano_palm_glb[:,1].max():.3f}]")
    print(f"  Stage3 mano GLB:         first={s3_mano_glb[0].round(4)}, Y=[{s3_mano_glb[:,1].min():.3f},{s3_mano_glb[:,1].max():.3f}]")
    print(f"  Original MANO palm SAPIEN: first={mano_palm_glb[0].round(4)}, Z=[{mano_palm_glb[:,2].min():.3f},{mano_palm_glb[:,2].max():.3f}]")
    print(f"  Stage3 mano SAPIEN (raw):  first={s3_mano[0].round(4)}, Z=[{s3_mano[:,2].min():.3f},{s3_mano[:,2].max():.3f}]")
except FileNotFoundError:
    has_s3 = False
    _dummy = np.zeros((1, 3))
    s3_mano_glb = _dummy
    s3_opt_glb = _dummy
    s3_obj_glb = _dummy
    s3_mano = _dummy
    print("  ⚠ Stage3 result not found, skipping stage3 trajectories in plot")

# ── v5 2026-09-05: 预处理产物叠加 (v51 §1.5) — MANO 三色参考锚 + 夹爪重定向轨迹 ──
try:
    _pp = np.load(f'{OUT_DIR}/preprocess_traj.npz', allow_pickle=True)
    _pp_joints = _pp['joints'] if 'joints' in _pp else None          # 平移后 MANO 21 关节 (+pre_offset)
    _pp_jraw = _pp['joints_raw'] if 'joints_raw' in _pp else None    # 平移前原始重定向关节 (无 +off)
    _pp_preoff = _pp['pre_offset'] if 'pre_offset' in _pp else None  # pre_translate 平移量
    _pp_gpos = _pp['grip_pos']
    _pp_gR = _pp['grip_R']
    _pp_F50 = min(50, len(_pp_gpos) - 1)
    has_pp = True
    print(f"Preprocess dump: N={len(_pp_gpos)}, method={_pp['grasp_pose_method'] if 'grasp_pose_method' in _pp else '?'}")
except FileNotFoundError:
    has_pp = False
    _pp_joints, _pp_gpos, _pp_gR, _pp_F50 = None, None, None, 50
    print("⚠ 无 preprocess_traj.npz (跑 --test-stage0 自动生成), 跳过预处理叠加")

# ═══════════════════════════════════════════════════════════════
# 2. PRINT SCALE ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SCALE ANALYSIS (GLB world, Z-up)")
print("="*70)

all_pts = np.vstack([mano_palm_glb, s1_glb.reshape(1,3), s2_glb, s3_mano_glb, s3_opt_glb, s3_obj_glb])
print(f"  Overall range: X=[{all_pts[:,0].min():.4f},{all_pts[:,0].max():.4f}] "
      f"Y=[{all_pts[:,1].min():.4f},{all_pts[:,1].max():.4f}] "
      f"Z=[{all_pts[:,2].min():.4f},{all_pts[:,2].max():.4f}]")
print(f"  Overall extent: {all_pts.max(0)-all_pts.min(0)} (m)")

# ═══════════════════════════════════════════════════════════════
# 3. PLOT — 左: 全景 (objects + MANO + Stage1/2/3); 右: 三条 MANO 轨迹对比
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(24, 10.5))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
fig.patch.set_facecolor('white')

def draw_objects(target_ax, alpha=0.15):
    obj_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#a65628','#f781bf','#17becf']
    for i, (nm, ct, bd, _rgb, _gname) in enumerate(geoms):
        if 'grid' in nm.lower():  # skip ground grid
            continue
        # 必须用 GLB 真实颜色 (vertex_colors/material.diffuse), 与原始场景一致; 无颜色才回退调色板
        c = _rgb if (_rgb is not None and np.asarray(_rgb).shape == (3,)) else obj_colors[i % len(obj_colors)]
        p0, p1 = bd
        corners = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                            [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
        for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            target_ax.plot3D(*corners[[e[0],e[1]]].T, color=c, alpha=alpha, lw=1.0)
        target_ax.scatter(*ct, c=c, s=60, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
        target_ax.text(ct[0], ct[1], ct[2]+0.05, nm, color=c, fontsize=7)

draw_objects(ax)

# ── World axes (same as render_quick.py draw_axes_arrows) ──
AX_LEN = 0.5
origin = np.array([0,0,0])
for c, lbl, v in [('red','X',[AX_LEN,0,0]),('green','Y',[0,AX_LEN,0]),('blue','Z',[0,0,AX_LEN])]:
    tip = origin + v
    ax.plot([origin[0],tip[0]],[origin[1],tip[1]],[origin[2],tip[2]], color=c, lw=3, alpha=0.9, zorder=10)
    ax.text(tip[0]+v[0]*0.1, tip[1]+v[1]*0.1, tip[2]+v[2]*0.1, lbl, color=c, fontsize=12, fontweight='bold')

# ── Original MANO (HaWoR FK palm, same transform as render_quick.py) ──
ax.plot(mano_palm_glb[:,0], mano_palm_glb[:,1], mano_palm_glb[:,2], '-', color='magenta', lw=2.5, alpha=0.8, zorder=6,
        label=f'Original MANO (HaWoR FK, palm, n={len(mano_palm_glb)})')
ax.scatter(*mano_palm_glb[0], c='magenta', s=40, marker='o', zorder=7)
ax.scatter(*mano_palm_glb[-1], c='magenta', s=80, marker='*', zorder=7, edgecolors='k')
ax.text(mano_palm_glb[-1,0], mano_palm_glb[-1,1], mano_palm_glb[-1,2], 'MANO_end', fontsize=7, color='magenta')
ax.scatter(*mano_palm_glb[50], c='magenta', s=90, marker='*', edgecolors='k', zorder=8)
ax.text(mano_palm_glb[50,0], mano_palm_glb[50,1], mano_palm_glb[50,2]+0.03, 'MANO F50', fontsize=7, color='magenta', ha='center')

# ── Pre-translated MANO (PIPELINE START) ──
# v4.30: test-stage3 已补 _pre_translate_mano_trajectory, stage3_result.npz 的
# mano_pos 即"平移后的 MANO 参考轨迹"(管线实际使用的基准), 不再用 S1 代理近似.
ax.plot(s3_mano_glb[:,0], s3_mano_glb[:,1], s3_mano_glb[:,2], '-', color='cyan', lw=2.2, alpha=0.9, zorder=6,
        label=f'Pre-translated MANO (PIPELINE START, n={len(s3_mano_glb)})')
_f50 = min(len(s3_mano_glb) - 1, 50)
ax.scatter(*s3_mano_glb[_f50], c='cyan', s=110, marker='D', edgecolors='k', linewidths=1.0, zorder=8)
ax.text(s3_mano_glb[_f50,0], s3_mano_glb[_f50,1], s3_mano_glb[_f50,2]+0.03,
        'PreTrans F50', fontsize=7, color='teal', ha='center', fontweight='bold')
ax.scatter(*s3_mano_glb[0], c='cyan', s=45, marker='o', zorder=7)
ax.scatter(*s3_mano_glb[-1], c='cyan', s=85, marker='*', zorder=7, edgecolors='k')

# ── v5: 预处理叠加 — MANO 参考锚 (红腕/绿拇指/蓝食指) + 夹爪 base 重定向轨迹 (黑) ──
if has_pp:
    # 只渲染 两指尖中点 (thumb-tip j4 与 index-tip j8 的中点) 的轨迹:
    #   平移前 (joints_raw, 无 pre_offset) vs 平移后 (joints, +pre_offset) 对比
    _off = _pp_preoff if _pp_preoff is not None else np.zeros(3)
    _md_pre = None
    if _pp_jraw is not None and len(_pp_jraw) == len(_pp_gpos):
        _md_pre = 0.5 * (_pp_jraw[:, 4] + _pp_jraw[:, 8])          # (N,3) 平移前两指中点
    if _pp_joints is not None and len(_pp_joints) == len(_pp_gpos):
        _md_post = 0.5 * (_pp_joints[:, 4] + _pp_joints[:, 8])     # (N,3) 平移后两指中点 (+off)
        if _md_pre is not None and (_md_pre + _off - _md_post).shape[0] == len(_md_post):
            _md_pre = _md_pre  # 保留原始做对比
        ax.plot(_md_post[:, 0], _md_post[:, 1], _md_post[:, 2], '-', color='#ff7f0e', lw=2.8,
                alpha=0.95, zorder=6, label='两指尖中点 (平移后, +pre_offset)')
        ax.scatter(*_md_post[_pp_F50], c='#ff7f0e', s=120, marker='*', edgecolors='k', zorder=9)
        ax.text(_md_post[_pp_F50, 0], _md_post[_pp_F50, 1], _md_post[_pp_F50, 2] + 0.03,
                'Mid(F50) post', fontsize=7, color='#b25a00', ha='center', fontweight='bold')
    if _md_pre is not None:
        ax.plot(_md_pre[:, 0], _md_pre[:, 1], _md_pre[:, 2], '--', color='gray', lw=2.4,
                alpha=0.9, zorder=5, label=f'两指尖中点 (平移前, raw, |off|={np.linalg.norm(_off)*1000:.0f}mm)')
        ax.scatter(*_md_pre[_pp_F50], c='gray', s=100, marker='s', edgecolors='k', zorder=8)
        ax.text(_md_pre[_pp_F50, 0] + 0.02, _md_pre[_pp_F50, 1], _md_pre[_pp_F50, 2],
                'Mid(F50) pre', fontsize=7, color='gray', ha='left')
        # 用箭头/连线标出 pre_translate 平移方向 (取 3 帧示意)
        for _fi in [0, _pp_F50, len(_md_pre) - 1]:
            _a = _md_pre[_fi]; _b = _md_post[_fi]
            ax.plot([_a[0], _b[0]], [_a[1], _b[1]], [_a[2], _b[2]], ':', color='#7f7f7f',
                    lw=1.0, alpha=0.8)
    ax.set_title('夹爪中心(两指中点)轨迹 · 平移前 vs 平移后 (MINT 平滑)', fontsize=11)

# ── 数值验证: 平移后 MANO 相对原始 MANO 应为常数平移 (= pre_offset), std≈0 ──
_n = min(len(mano_palm_glb), len(s3_mano_glb))
_pre_off = s3_mano_glb[:_n] - mano_palm_glb[:_n]
print(f"  [verify] mano_pos - rawGLB: mean={_pre_off.mean(0).round(4)}, "
      f"std={_pre_off.std(0).round(5)} (std≈0 ⇒ 纯平移, |mean|≈|pre_offset|)")

# ── Stage 1 grasp ──
ax.scatter(*s1_glb, c='green', s=200, marker='D', edgecolors='k', linewidths=1.5, zorder=10)
ax.text(s1_glb[0], s1_glb[1], s1_glb[2]+0.03, 'S1 Grasp', fontsize=9, color='darkgreen', fontweight='bold', ha='center')

# ── Stage 2 lift ──
ax.plot(s2_glb[:,0], s2_glb[:,1], s2_glb[:,2], '-', color='darkorange', lw=2.5, alpha=0.9, zorder=6,
        label=f'Stage 2 Lift (FIXED reconstruction, F{s2_frames[0]}-F{s2_frames[-1]})')
ax.scatter(*s2_glb[0], c='darkorange', s=80, marker='^', zorder=8, edgecolors='k')
ax.scatter(*s2_glb[-1], c='darkorange', s=80, marker='v', zorder=8, edgecolors='k')
ax.text(s2_glb[0,0], s2_glb[0,1], s2_glb[0,2]-0.03, 'S2_start', fontsize=7, color='darkorange', ha='center')
ax.text(s2_glb[-1,0], s2_glb[-1,1], s2_glb[-1,2]-0.03, 'S2_end', fontsize=7, color='darkorange', ha='center')
ax.scatter(*s2_glb[50 - int(s2_frames[0])], c='darkorange', s=90, marker='*', edgecolors='k', zorder=8)
ax.text(s2_glb[50 - int(s2_frames[0]),0], s2_glb[50 - int(s2_frames[0]),1],
        s2_glb[50 - int(s2_frames[0]),2]+0.03, 'S2 F50', fontsize=7, color='darkorange', ha='center')

# ── Stage 3 Opt ──
ax.plot(s3_opt_glb[:,0], s3_opt_glb[:,1], s3_opt_glb[:,2], '-', color='royalblue', lw=2.5, alpha=0.9, zorder=6,
        label=f'Stage 3 Opt (F0-F{len(s3_opt_glb)-1})')
ax.scatter(*s3_opt_glb[0], c='royalblue', s=30, marker='o', zorder=7)
ax.scatter(*s3_opt_glb[-1], c='royalblue', s=100, marker='*', zorder=8, edgecolors='k')
ax.text(s3_opt_glb[-1,0], s3_opt_glb[-1,1], s3_opt_glb[-1,2]+0.03, 'S3_end', fontsize=7, color='royalblue', fontweight='bold', ha='center')

# ── Stage 3 object trajectory ──
ax.plot(s3_obj_glb[:,0], s3_obj_glb[:,1], s3_obj_glb[:,2], ':', color='brown', lw=2.0, alpha=0.8, zorder=5,
        label=f'Object (simulation)')

# ── Axis limits (左图) ──
all_plot = np.vstack([mano_palm_glb, s1_glb.reshape(1,3), s2_glb, s3_mano_glb, s3_opt_glb, s3_obj_glb])
pad = 0.15
ax.set_xlim(all_plot[:,0].min()-pad, all_plot[:,0].max()+pad)
ax.set_ylim(all_plot[:,1].min()-pad, all_plot[:,1].max()+pad)
ax.set_zlim(all_plot[:,2].min()-pad, all_plot[:,2].max()+pad)

ax.set_xlabel('GLB X (m)', fontsize=10)
ax.set_ylabel('GLB Y (m)', fontsize=10)
ax.set_zlabel('GLB Z (m, up)', fontsize=10)
ax.set_title(f'Left: Full Pipeline Overview (GLB Z-up, objects=node-transform)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', framealpha=0.92, ncol=2)
ax.view_init(elev=20, azim=45)
ax.set_box_aspect([1,1,0.8])

# ═══════════════════════════════════════════════════════════════
# 右图 (最后一张): 原始 MANO → 平移后 MANO → 管线最终轨迹 三代对比
# ═══════════════════════════════════════════════════════════════
draw_objects(ax2, alpha=0.10)

# ① 原始 MANO (SLAM→GLB 直接映射, 未平移)
ax2.plot(mano_palm_glb[:,0], mano_palm_glb[:,1], mano_palm_glb[:,2], '-', color='magenta', lw=2.5, alpha=0.85,
         zorder=6, label=f'① Original MANO (SLAM→GLB raw, n={len(mano_palm_glb)})')
ax2.scatter(*mano_palm_glb[_f50], c='magenta', s=90, marker='D', edgecolors='k', linewidths=0.8, zorder=8)

# ② 平移后的 MANO (管线基准 = stage3 mano_pos, 已 pre_translate 对齐物体)
ax2.plot(s3_mano_glb[:,0], s3_mano_glb[:,1], s3_mano_glb[:,2], '-', color='cyan', lw=2.2, alpha=0.9,
         zorder=6, label=f'② Pre-translated MANO (pipeline ref)')
ax2.scatter(*s3_mano_glb[_f50], c='cyan', s=110, marker='D', edgecolors='k', linewidths=1.0, zorder=8)
# F50 对应点连线: 展示 pre_offset 平移向量
ax2.plot([mano_palm_glb[_f50,0], s3_mano_glb[_f50,0]],
         [mano_palm_glb[_f50,1], s3_mano_glb[_f50,1]],
         [mano_palm_glb[_f50,2], s3_mano_glb[_f50,2]], '--', color='gray', lw=1.2, alpha=0.7, zorder=4)
ax2.text((mano_palm_glb[_f50,0]+s3_mano_glb[_f50,0])/2,
         (mano_palm_glb[_f50,1]+s3_mano_glb[_f50,1])/2,
         (mano_palm_glb[_f50,2]+s3_mano_glb[_f50,2])/2 + 0.04,
         f'pre_offset\n|{_pre_off.mean(0).round(2)}|', fontsize=7, color='#555555', ha='center')

# ③ 最终管线生成轨迹 (Stage3 输出 opt_pos)
ax2.plot(s3_opt_glb[:,0], s3_opt_glb[:,1], s3_opt_glb[:,2], '-', color='royalblue', lw=2.5, alpha=0.9,
         zorder=6, label=f'③ Final pipeline traj (Stage3 out, n={len(s3_opt_glb)})')
ax2.scatter(*s3_opt_glb[-1], c='royalblue', s=100, marker='*', zorder=8, edgecolors='k')

# Stage1 抓取点参考
ax2.scatter(*s1_glb, c='green', s=180, marker='D', edgecolors='k', linewidths=1.5, zorder=10)
ax2.text(s1_glb[0], s1_glb[1], s1_glb[2]+0.03, 'S1 Grasp', fontsize=8, color='darkgreen',
         fontweight='bold', ha='center')

ax2.set_xlim(all_plot[:,0].min()-pad, all_plot[:,0].max()+pad)
ax2.set_ylim(all_plot[:,1].min()-pad, all_plot[:,1].max()+pad)
ax2.set_zlim(all_plot[:,2].min()-pad, all_plot[:,2].max()+pad)
ax2.set_xlabel('GLB X (m)', fontsize=10)
ax2.set_ylabel('GLB Y (m)', fontsize=10)
ax2.set_zlabel('GLB Z (m, up)', fontsize=10)
ax2.set_title('Right: MANO Trajectory Evolution — Original → Pre-translated → Final',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left', framealpha=0.92)
ax2.view_init(elev=20, azim=45)
ax2.set_box_aspect([1,1,0.8])

# 阶段说明
_stage_note = ("Pipeline: Original MANO → [pre_translate: F50 finger_mid→obj_center] → Pre-translated MANO (pipeline ref) → "
               "Stage1 CMA-ES grasp → Stage2 FIXED reconstruction (locked) → Stage3 smooth blend → Final traj. "
               "v4.30: test-stage3 补回缺失的 pre_translate, neutral_offset 回归≈0.")
fig.text(0.5, 0.012, _stage_note, ha='center', fontsize=7.5, color='#333333')

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {OUT_PNG}")