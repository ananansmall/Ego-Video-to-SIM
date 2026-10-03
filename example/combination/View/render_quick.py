#!/usr/bin/env python3
"""
Render alignment comparison (raw vs depth-aligned) of HaWoR + RAS data, overlayed in one GLB view.
Generic across examples (121 / 7 / hoi4d / ...); pass any hawor-dir/ras-dir with matching structure.
Usage:
  python3 View/render_quick.py --hawor-dir /path/to/hawor --ras-dir /path/to/ras --output /tmp/align --show-depth-cloud

Reference points (canonical world = GLB NATIVE world, i.e. the RAS reconstruction world):
  * Camera trajectory      : RAS extrinsics are natively in the GLB world (load_ras_cameras,
                             Z-up, no extra transform). HaWoR cameras/hands live in the HaWoR
                             world which is a DIFFERENT frame (dataset 7: ~65 deg + translation;
                             its extrinsics were re-aligned by stage1_strict, see
                             coordinate_alignment.json).
  * Hand joints (MANO FK)  : FK joints live in the HaWoR INPUT CAMERA frame (pred_trans z>0).
    Data-driven chain (no assumed flips):
      p_hawor = R_c2w_hawor[f] @ p_cam + t_c2w_hawor[f]
      p_glb   = s * R_hand_to_glb @ p_hawor + t_hand_to_glb
    where (s, R_hand_to_glb, t_hand_to_glb) come from transform_params.npz produced by
    001_align_scene.py (first-frame camera anchoring + Umeyama scale) - the SAME file the
    canonical pipeline (00_run_pipeline.py -> 002_render_scene.py) consumes. No online
    calibration here: render_quick is just a viewer.
  * Wrist (j0)             : MANO FK root joint (== pred_trans, the hand-root anchor)
  * Thumb tip  (joint 20)  : MANO FK joint 20
  * Index tip  (joint 16)  : MANO FK joint 16
  * Midpoint (plotted)     : (thumb_tip_j20 + index_tip_j16) / 2  -- the pinch/grasp point.
                             "Raw" uses the un-aligned HaWoR npz; "Depth" uses the *_depth_aligned npz.
  * RAS depth center       : independent ground truth = HaWoR hand mask center unprojected through
                             RAS depth + extrinsics. Used to validate the hand placement.
Note: GLB objects use RAW geom.vertices = Z-up extrinsics/GLB world coords (objects lie
      flat at z~0, consistent with cameras at z~0.5). Do NOT apply the per-node T:
      dataset 7 nodes carry only a Z-up->Y-up export rotation (translation=0); applying
      it stands objects up and inflates their Z extent (~0.6 m) - verified against
      load_glb_to_sapien in the original Ego-Video-to-SIM which also uses raw vertices.
Hands: --hands left|right|both selects which hand to render (datasets physically contain
      one hand only; the other is an upstream HaWoR hallucination - never auto-judged).
Plotted hand points: Midpoint = (thumb_tip_j20 + index_tip_j16)/2 pinch point
      (Raw solid line vs Depth corrected line), same color family per side.
      Wrist/Thumb tip/Index tip markers are kept in code but commented out
      (disabled 2026-08-26 by user request).
"""
import argparse, sys, warnings, os
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go
import trimesh
import cv2
import torch

# MANO FK 导入
_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE.parent / 'position_retargeting'))
from mano_layer import MANOLayer

def xform(pts, s, R, t): return s * (R @ pts.T).T + t

def cam_traj_to_glb(pts, R_c2w_h, t_c2w_h, s, R, t):
    """(N,3) HaWoR 输入相机系点 -> 逐帧真实相机位姿 R_c2w[f]/t_c2w[f] 转到 HaWoR 世界系
    -> 相似变换到 GLB 世界系. 与 vis_camera_trajectories.py 的相机轨迹处理完全一致
    (相机轨迹也是世界系点直接过同一个相似变换), 不引入任何假设翻转."""
    p_hw = np.einsum('fij,fj->fi', R_c2w_h, pts) + t_c2w_h
    return xform(p_hw, s, R, t)


H_SID = {0: 'left', 1: 'right'}

def compute_mano_joints(d, hawor_dir, hands=(0, 1)):
    """通过 MANO FK 计算21个关节点, 返回 {hidx: (N,21,3)} (NaN for invalid). 只算 hands 里的手"""
    joints_all = {}
    for hidx in hands:
        side = H_SID[hidx]
        n = len(d['pred_trans'][hidx])
        betas_arr = d['pred_betas'][hidx]
        # 找第一个有效 beta
        first_valid = None
        for fi in range(n):
            if not np.any(np.isnan(betas_arr[fi])):
                first_valid = fi
                break
        if first_valid is None:
            joints_all[hidx] = np.full((n, 21, 3), np.nan)
            continue
        betas = betas_arr[first_valid].astype(np.float32)
        ml = MANOLayer(side, betas)
        joints = np.full((n, 21, 3), np.nan, dtype=np.float64)
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
                _, j = ml(p_t, t_t)
            jv = j[0].numpy()
            if not np.any(np.isnan(jv)):
                joints[fi] = jv
        joints_all[hidx] = joints
        nv = np.sum(~np.any(np.isnan(joints[:, 0]), axis=1))
        print(f'  MANO {side}: {nv}/{n} valid frames')
    return joints_all

def draw_axes_arrows(ax, origin, scale=5.0, lw=6, axes=None):
    o = np.array(origin)
    if axes is None:
        axes = [('red','X',[1,0,0]),('green','Y',[0,1,0]),('blue','Z',[0,0,1])]
    for c, lbl, v in axes:
        v = np.asarray(v, dtype=float); v = v / max(np.linalg.norm(v), 1e-9) * scale
        tip = o + v
        ax.plot([o[0],tip[0]],[o[1],tip[1]],[o[2],tip[2]], color=c, lw=lw, alpha=0.9, zorder=10)
        ax.text(tip[0]+v[0]*0.1,tip[1]+v[1]*0.1,tip[2]+v[2]*0.1, lbl, color=c, fontsize=20, fontweight='bold')

def draw_camera_axes(ax, origin, R_cam, scale=0.35):
    o = np.array(origin)
    for i,(c,lbl) in enumerate([('red','X'),('green','Y'),('blue','Z')]):
        v = R_cam[:,i]*scale; tip = o+v
        ax.plot([o[0],tip[0]],[o[1],tip[1]],[o[2],tip[2]], color=c, lw=2, ls='--', alpha=0.7, zorder=10)
        ax.text(tip[0]+v[0]*0.1,tip[1]+v[1]*0.1,tip[2]+v[2]*0.1, f'cam{lbl}', color=c, fontsize=10, fontweight='bold')
    look = R_cam[:,2]*scale*0.8; tip_l = o+look
    ax.plot([o[0],tip_l[0]],[o[1],tip_l[1]],[o[2],tip_l[2]], color='orange', lw=4, alpha=1.0, zorder=11)
    ax.text(tip_l[0]+look[0]*0.15,tip_l[1]+look[1]*0.15,tip_l[2]+look[2]*0.15, 'Look', color='darkorange', fontsize=11, fontweight='bold')
    ax.scatter(*o, c='black', s=80, marker='d', zorder=10)
    ax.text(o[0],o[1],o[2]+0.12, 'cam0', fontsize=10, color='black', fontweight='bold')

def draw_bbox(ax, bounds, color, alpha=0.2):
    p0,p1 = bounds
    c = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                  [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
    for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot3D(*c[[e[0],e[1]]].T, color=color, alpha=alpha, lw=1.0)

def load_data(hawor_dir, ras_dir, use_depth, params_p, depth_npz=None, raw_hw=False, hands=(0, 1)):
    """加载数据, 通过 MANO FK 计算 拇指尖(j20) 和 手腕(j0) 的世界坐标.

    params_p: transform_params.npz 路径 (001_align_scene.py 产出).
    raw_hw=False (默认): 过相似变换 (s,R,t) -> GLB 坐标.
    raw_hw=True: 留在 HaWoR 世界系 (仅调试用; 注意 HaWoR 世界 ≠ GLB 世界)."""
    rec = Path(hawor_dir)/'reconstruction'
    if use_depth and depth_npz and Path(depth_npz).exists():
        p = str(depth_npz)
    else:
        files = sorted(rec.glob('hawor_results_*.npz'))
        if use_depth:
            p = next((f for f in files if '_depth_aligned' in str(f)), files[0])
        else:
            p = next((f for f in files if '_depth_aligned' not in str(f)), files[0])
    d = dict(np.load(str(p), allow_pickle=True))
    par = params_p if isinstance(params_p, dict) else dict(np.load(params_p))
    s, R, t = float(par['scale_ratio']), par['R_hand_to_glb'], par['t_hand_to_glb']
    # 数据驱动旋转链: MANO FK 关节在 HaWoR 输入相机系 (pred_trans z>0 已证实),
    # 用逐帧真实相机位姿 R_c2w[f]/t_c2w[f] 转到 HaWoR 世界系, 再过相似变换到 GLB.
    # 注意: 不使用固定 Rx_hand=diag(1,-1,-1) —— 那只是 R_c2w[0]≈diag(1,-1,-1) 的
    # 第一帧特例近似, 相机转动后 (实测可达 34°) 会产生明显漂移.
    R_c2w_h, t_c2w_h = d['R_c2w'], d['t_c2w']  # (N,3,3), (N,3)
    rot_angle = np.arccos(min(1, max(-1, (np.trace(R)-1)/2))) * 180 / np.pi
    R0, t0_h = d['R_c2w'][0], d['t_c2w'][0]
    if raw_hw:
        t0_glb = t0_h                       # 相机中心在 HaWoR 世界 (未变换, 仅调试)
        cam_glb = d['t_c2w'].copy()
    else:
        t0_glb = xform(np.array([t0_h]), s, R, t)[0]
        cam_glb = xform(d['t_c2w'], s, R, t)
    cam_valid = np.ones(len(cam_glb), dtype=bool)

    # 通过 MANO FK 计算关节点
    joints_all = compute_mano_joints(d, hawor_dir, hands)

    thumbs = {}  # 拇指尖 j20
    indexs = {}  # 食指尖 j16
    wrists = {}  # 手腕 j0
    for hidx in hands:
        j = joints_all[hidx]  # (N, 21, 3)
        def _traj(pts):
            if raw_hw:
                # 留在 HaWoR 世界系: 逐帧相机位姿变换即可 (仅调试)
                return np.einsum('fij,fj->fi', R_c2w_h, pts) + t_c2w_h
            return cam_traj_to_glb(pts, R_c2w_h, t_c2w_h, s, R, t)
        # 拇指尖 j20
        thumb_pts = j[:, 20, :]  # (N, 3)
        x = _traj(thumb_pts)
        ok = ~np.any(np.isnan(x), axis=1)
        if ok.any():
            thumbs[hidx] = x[ok]
        # 食指尖 j16
        index_pts = j[:, 16, :]  # (N, 3)
        x = _traj(index_pts)
        ok = ~np.any(np.isnan(x), axis=1)
        if ok.any():
            indexs[hidx] = x[ok]
        # 手腕 j0 (MANO FK 的 j0 等同于 pred_trans)
        wrist_pts = j[:, 0, :]
        x2 = _traj(wrist_pts)
        ok2 = ~np.any(np.isnan(x2), axis=1)
        if ok2.any():
            wrists[hidx] = x2[ok2]
    return d, s, R, t, rot_angle, t0_glb, cam_glb, cam_valid, thumbs, indexs, wrists

def load_ras_cameras(ras_dir):
    ext_dir = Path(ras_dir)/'extrinsics'
    files = sorted(ext_dir.glob('*.txt'), key=lambda x: int(x.stem))
    ras_pos = []
    for f in files:
        ext = np.loadtxt(str(f))
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0,0,0,1]])
        R_w2c, t_w2c = ext[:3,:3], ext[:3,3]
        R_c2w = R_w2c.T
        ras_pos.append(-R_c2w @ t_w2c)
    ras_pos = np.array(ras_pos)
    ras_valid = np.ones(len(ras_pos), dtype=bool)
    return ras_pos, ras_valid

def refit_similarity(hawor_dir, ras_dir):
    """用 HaWoR 相机轨迹 -> RAS 相机轨迹 做完整 Umeyama 相似变换拟合, 替代 001 的 (s,R,t).

    001 用"第一帧锚定 + Umeyama 尺度"; 当相机移动量很小 (轨迹近乎退化) 时尺度估计病态,
    实测 dataset 7 相机只移动 ~2.9cm -> scale 偏大 10.4%, 手部被压到桌面下.
    这里直接用整条轨迹重拟合, 返回 (s, R, t); 失败返回 None."""
    rec = Path(hawor_dir) / 'reconstruction'
    files = sorted(rec.glob('hawor_results_*.npz'))
    if not files:
        return None
    f = next((x for x in files if '_depth_aligned' not in str(x)), files[0])
    d = dict(np.load(str(f), allow_pickle=True))
    if 't_c2w' not in d:
        return None
    A = np.asarray(d['t_c2w'], dtype=np.float64)
    B, _ = load_ras_cameras(ras_dir)
    n_h, n_r = len(A), len(B)
    if n_h < 4 or n_r < 4:
        return None
    if n_h != n_r:  # 帧数不等: 均匀重采样 HaWoR 到 RAS 长度
        print(f'  [warn] HaWoR {n_h} 帧 != RAS {n_r} 帧: 只能均匀重采样建立对应关系, '
              f'Umeyama 拟合的 R/t 不可靠 (实测 121 上相机误差反而 0.0119->0.1637). '
              f'仅当帧数相等时重拟合才有意义.')
        A = A[np.linspace(0, n_h - 1, n_r).round().astype(int)]
    a, b = A.T, B.T                                  # (3,N)
    ma, mb = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
    aa, bb = a - ma, b - mb
    H = aa @ bb.T / a.shape[1]
    U, S, Vt = np.linalg.svd(H)
    dd = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1.0, 1.0, dd]) @ Vt
    var_a = (aa ** 2).sum() / a.shape[1]
    s = float((S @ np.array([1.0, 1.0, dd])) / max(var_a, 1e-12))
    tv = (mb - s * (Rm @ ma)).ravel()
    return s, Rm, tv


def camera_traj_error(A, B):
    """两条相机轨迹的逐帧距离统计 (帧数不等时均匀重采样)."""
    n_h, n_r = len(A), len(B)
    if n_h != n_r:
        A = A[np.linspace(0, n_h - 1, n_r).round().astype(int)]
    e = np.linalg.norm(np.asarray(A) - np.asarray(B), axis=1)
    return e.mean(), e.max()


def load_ras_depth_data(ras_dir):
    ras_dir = Path(ras_dir)
    K = np.loadtxt(str(ras_dir / 'intrinsic.txt'))
    if K.shape == (3, 4): K = K[:, :3]
    elif K.size == 9: K = K.reshape(3, 3)
    ext_dir = ras_dir / 'extrinsics'
    ext_files = sorted(ext_dir.glob('*.txt'), key=lambda x: int(x.stem))
    R_c2w, t_c2w = [], []
    for f in ext_files:
        ext = np.loadtxt(str(f))
        if ext.shape == (3, 4): ext = np.vstack([ext, [0,0,0,1]])
        Rw2c, tw2c = ext[:3,:3], ext[:3,3]
        Rc2w = Rw2c.T
        R_c2w.append(Rc2w)
        t_c2w.append(-Rc2w @ tw2c)
    R_c2w, t_c2w = np.array(R_c2w), np.array(t_c2w)
    depth_dir = ras_dir / 'depth'
    depth_files = sorted(depth_dir.glob('*.png'), key=lambda x: int(x.stem))
    return K, R_c2w, t_c2w, depth_files

def unproject_mask_center_to_glb(mask, depth_img, K, R_c2w, t_c2w, depth_scale=1000.0):
    """HaWoR mask 中心 → RAS 深度 → GLB 3D"""
    if mask.shape[:2] != depth_img.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (depth_img.shape[1], depth_img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    cy, cx = ys.mean(), xs.mean()
    u, v = int(round(cx)), int(round(cy))
    u = np.clip(u, 0, depth_img.shape[1]-1)
    v = np.clip(v, 0, depth_img.shape[0]-1)
    z_val = depth_img[v, u].astype(float) / depth_scale
    if z_val <= 0 or not np.isfinite(z_val):
        z_vals = depth_img[ys, xs].astype(float) / depth_scale
        valid = z_vals > 0
        if not valid.any():
            return None
        z_val = np.median(z_vals[valid])
        u, v = int(round(xs[valid].mean())), int(round(ys[valid].mean()))
        u = np.clip(u, 0, depth_img.shape[1]-1)
        v = np.clip(v, 0, depth_img.shape[0]-1)
    fx, fy = K[0,0], K[1,1]
    cx_k, cy_k = K[0,2], K[1,2]
    x_cam = (u - cx_k) * z_val / fx
    y_cam = (v - cy_k) * z_val / fy
    p_cam = np.array([x_cam, y_cam, z_val])
    p_glb = R_c2w @ p_cam + t_c2w
    return p_glb, u, v, z_val

def get_geom_color(geom):
    """提取 GLB 几何体原始颜色 (顶点色均值 > 材质漫反射/baseColor > 灰色), 返回 0-1 RGB"""
    try:
        if hasattr(geom, 'visual') and geom.visual is not None:
            vc = getattr(geom.visual, 'vertex_colors', None)
            if vc is not None and len(vc) > 0:
                return np.array(vc[:, :3]).astype(float).mean(0) / 255.0
            mat = getattr(geom.visual, 'material', None)
            if mat is not None:
                col = getattr(mat, 'diffuse', None)
                if col is not None:
                    return np.array(col[:3]).astype(float) / 255.0
                bcf = getattr(mat, 'baseColorFactor', None)
                if bcf is not None:
                    return np.array(bcf[:3]).astype(float) / 255.0
    except Exception:
        pass
    return np.array([0.5, 0.5, 0.5])

# ============================================================
# 颜色方案: 轨迹用固定颜色, 物体方框用 GLB 原色
# ============================================================
COLORS = {
    'hawor_cam':      'navy',         # 深蓝 - HaWoR 相机轨迹
    'ras_cam':        'limegreen',    # 亮绿 - RAS 相机轨迹
    'l_mid_raw':      'red',          # 红 - 左手拇指食指中点 Raw
    'l_mid_depth':    'dodgerblue',   # 蓝 - 左手拇指食指中点 Depth
    'r_mid_raw':      'darkorange',   # 橙 - 右手拇指食指中点 Raw
    'r_mid_depth':    'gold',         # 金 - 右手拇指食指中点 Depth
    'l_mid_glb':      'magenta',      # 品红 - 左手 RAS MANO 中点 (GLB 原生)
    'r_mid_glb':      'teal',         # 青 - 右手 RAS MANO 中点 (GLB 原生)
    'l_mid_raw_off':  'pink',         # 粉 - 左手拇指食指中点 Raw (偏移后)
    'l_mid_depth_off':'turquoise',    # 青绿 - 左手拇指食指中点 Depth (偏移后)
    'r_mid_raw_off':  'magenta',      # 品红 - 右手拇指食指中点 Raw (偏移后)
    'r_mid_depth_off':'yellow',       # 黄 - 右手拇指食指中点 Depth (偏移后)
    'ras_depth_ctr':  'cyan',         # 青 - RAS depth center (原始)
    'ras_depth_ctr_off':'brown',      # 棕 - RAS depth center (偏移后)
    'wrist_dpj_l':    'darkviolet',   # 紫 - 左手手腕深度反投影 (HaWoR→GLB)
    'wrist_dpj_r':    'sienna',       # 赭 - 右手手腕深度反投影 (HaWoR→GLB)
}


# ===== GLB 物体加载: 唯一入口. =====
# 实证 (2026-08, 数据集7): final_scene.glb 每个节点的 T 都是导出时加的 Z-up->Y-up
# 纯旋转 (平移=0). 原始 geom.vertices 才是 Z-up 外参/GLB 世界坐标 (物体平铺在
# z≈0 地面); 若做 T@v 会把物体转立起来 (z 方向虚高 ~0.6m), 与相机轨迹 (z≈0.5)
# 不自洽. 原版 Ego-Video-to-SIM 的 load_glb_to_sapien 同样直接用 geom.vertices.
def load_glb_objects(glb_p):
    """唯一 GLB 加载入口. 返回 [(name, verts(N,3), color_rgb(3,))].
    verts = 原始顶点 = Z-up 外参/GLB 世界坐标 (不做节点变换, 见上实证)."""
    scene = trimesh.load(str(glb_p), force='scene')
    out = []
    for node_name in scene.graph.nodes_geometry:
        _, geom_name = scene.graph.get(node_name)
        if geom_name is None or geom_name not in scene.geometry:
            continue
        geom = scene.geometry[geom_name]
        out.append((node_name, np.asarray(geom.vertices, dtype=float),
                    get_geom_color(geom)))
    return out


def load_ras_mano_glb(hawor_dir, hands):
    """加载 mano_ras_reconstruction_*.npz 的 joints_ras_world_{left,right}.

    这些关节由 RAS 深度直接反投影重建, 原生就在 GLB 世界系 (Z-up),
    不需要过 transform_params 相似变换, 也不需要旋转链提升.
    返回 {hidx: (N,3) 拇指尖(j20)+食指尖(j16) 中点轨迹}."""
    rec = Path(hawor_dir)/'reconstruction'
    files = sorted(rec.glob('mano_ras_reconstruction_*.npz'))
    if not files:
        return {}
    d = dict(np.load(str(files[-1])))
    out = {}
    for hidx in hands:
        side = 'left' if hidx == 0 else 'right'
        key = f'joints_ras_world_{side}'
        if key not in d:
            continue
        j = d[key]  # (N, 21, 3)
        mid = 0.5 * (j[:, 20, :] + j[:, 16, :])
        ok = np.isfinite(mid).all(axis=1)
        vkey = f'valid_{side}'
        if vkey in d:
            ok &= d[vkey].astype(bool)
        if ok.any():
            out[hidx] = mid[ok]
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hawor-dir', required=True)
    parser.add_argument('--ras-dir', required=True)
    parser.add_argument('--output', default='/tmp/align')
    parser.add_argument('--axis-scale', type=float, default=5.0)
    parser.add_argument('--camera-scale', type=float, default=0.35)
    parser.add_argument('--no-axes', action='store_true',
                        help='不绘制全局 XYZ 与相机坐标系箭头 (默认绘制)')
    parser.add_argument('--depth-npz', type=str, default=None)
    parser.add_argument('--show-depth-cloud', action='store_true',
                        help='将手部 mask 中心和手腕的 RAS 深度反投影到 GLB')
    parser.add_argument('--offset-frame', type=int, default=-1,
                        help='对 mask center 轨迹的第 N 帧施加偏移 (默认-1=不偏移)')
    parser.add_argument('--offset-rgb', type=str, default=None,
                        help='目标物体 RGB (0-255), 如 "200,50,180", 自动从 GLB 找最近物体')
    parser.add_argument('--glb', type=str, default='final_scene.glb',
                        help='GLB 文件名 (默认 final_scene.glb, 可选 final_scene1.glb 等)')
    parser.add_argument('--refit-transform', action='store_true',
                        help='不用 001 的 (s,R,t), 改用 HaWoR 相机轨迹 -> RAS 相机轨迹 的 '
                             'Umeyama 重拟合 (相机移动量小的轨迹上尺度更稳)')
    parser.add_argument('--hands', choices=['both', 'left', 'right'], default='both',
                        help='渲染哪只手. 数据集实际只有一只手 (如 7 只有左手, 121 只有右手), '
                             '另一只是 HaWoR 上游幻觉输出, 直接不渲染, 不做任何自动判断')
    parser.add_argument('--stage0-npz', type=str, default=None,
                        help='stage0 预处理产物 preprocess_traj.npz: 叠加两个路径 — '
                             '处理前 MANO 拇食中点 (joints_raw j20+j16, 未平滑未重定向) + '
                             '处理后夹爪 finger_mid (grip_pos+R*0.03689, 重定向+pre_translate 后)')
    parser.add_argument('--ras-final-npz', type=str, default=None,
                        help='RAS-GLB 合成版 ras_glb_final_left.npz (output/ras_glb_synthesized/): '
                             '叠加第 4 条中点轨迹 mid_smooth (savgol 平滑的 RAS 深度反投影拇食中点)')
    parser.add_argument('--zoom', action='store_true',
                        help='视野聚焦到所有轨迹包围盒 (默认固定 [-2,2]^3)')
    parser.add_argument('--minimal', action='store_true',
                        help='精简模式: 只画 GLB 物体 + Midpoint Raw vs Depth 对比 '
                             '(隐藏相机轨迹/RAS-GLB/depth center/wrist dpj)')
    args = parser.parse_args()
    hand_ids = {'left': [0], 'right': [1], 'both': [0, 1]}[args.hands]

    hawor_dir, ras_dir = Path(args.hawor_dir), Path(args.ras_dir)
    # transform_params.npz 由 001_align_scene.py 预先产出 (第一帧相机锚定 + Umeyama 尺度),
    # 与正统管线 (00_run_pipeline -> 002_render_scene) 消费的是同一个文件. 这里只加载.
    params_p = None
    for cand in (hawor_dir/'transform_params.npz',
                 hawor_dir/'output/alignment/transform_params.npz'):
        if cand.exists():
            params_p = str(cand)
            break
    if params_p is None:
        raise FileNotFoundError(
            f'transform_params.npz not found under {hawor_dir}; '
            f'run 001_align_scene.py first')
    glb_p = ras_dir/args.glb

    # ===== 规范世界 = GLB 原生世界 (= RAS 重建世界, Z-up). 与原版 Ego-Video-to-SIM 一致:
    #   * GLB 物体: 原生在 GLB 世界, 节点变换 @ 局部顶点, 零改动;
    #   * RAS 相机 / mask 中心 / 深度反投影: 外参原生就在 GLB 世界, 直接用;
    #   * MANO 手 / HaWoR 相机: 在 HaWoR 世界, 经 transform_params.npz 的
    #     (s, R_hand_to_glb, t_hand_to_glb) 相似变换进 GLB 世界 (load_data 内完成).
    # 没有其他坐标变换层. =====
    par = dict(np.load(params_p))
    print(f'transform_params: {params_p}  scale={float(par["scale_ratio"]):.4f}')

    # 可选: 用整条相机轨迹重拟合相似变换, 替代 001 的 (s,R,t)
    params_eff = par
    if args.refit_transform:
        fit = refit_similarity(hawor_dir, ras_dir)
        if fit is None:
            print('  [warn] 相机轨迹重拟合失败, 沿用 001 的变换')
        else:
            s_n, R_n, t_n = fit
            s_o = float(par['scale_ratio'])
            R_o, t_o = par['R_hand_to_glb'], par['t_hand_to_glb']
            rec_h = hawor_dir / 'reconstruction'
            _f = next((x for x in sorted(rec_h.glob('hawor_results_*.npz'))
                       if '_depth_aligned' not in str(x)), None)
            if _f is not None:
                _A = dict(np.load(str(_f), allow_pickle=True))['t_c2w']
                _B, _ = load_ras_cameras(ras_dir)
                eo = camera_traj_error(xform(_A, s_o, R_o, t_o), _B)
                en = camera_traj_error(xform(_A, s_n, R_n, t_n), _B)
                print(f'  相机轨迹误差: 001 mean={eo[0]:.4f} max={eo[1]:.4f}'
                      f'  ->  refit mean={en[0]:.4f} max={en[1]:.4f}')
            print(f'  refit scale: {s_o:.4f} -> {s_n:.4f}  ({s_o/s_n-1:+.1%})')
            params_eff = {'scale_ratio': s_n,
                          'R_hand_to_glb': R_n,
                          't_hand_to_glb': t_n}

    # Load both raw & depth-aligned (通过 MANO FK 计算 拇指尖和手腕)
    d_raw, s, R, t, ra, t0_glb, cam_glb, cam_valid, thumbs_raw, indexs_raw, wrists_raw = load_data(hawor_dir, ras_dir, False, params_eff, hands=hand_ids)
    d_da, s2, R2, t2, ra2, t0_glb2, cam_glb2, cam_valid2, thumbs_da, indexs_da, wrists_da = load_data(hawor_dir, ras_dir, True, params_eff, depth_npz=args.depth_npz, hands=hand_ids)
    # 手/相机已经过相似变换在 GLB 世界 (load_data 内完成), 无需再变换
    # 计算拇指+食指中点轨迹
    mids_raw = {}
    mids_da = {}
    for hidx in [0, 1]:
        if hidx in thumbs_raw and hidx in indexs_raw:
            mids_raw[hidx] = 0.5 * (thumbs_raw[hidx] + indexs_raw[hidx])
        if hidx in thumbs_da and hidx in indexs_da:
            mids_da[hidx] = 0.5 * (thumbs_da[hidx] + indexs_da[hidx])
    # RAS MANO 重建中点 (原生 GLB 世界系, 零变换直接用)
    mids_glb = load_ras_mano_glb(hawor_dir, hand_ids)
    for hidx in hand_ids:
        if hidx in mids_glb:
            n = len(mids_glb[hidx])
            c = mids_glb[hidx].mean(0)
            print(f'mids_glb[h{hidx}]: {n} pts, center=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})')

    # RAS-GLB 合成版 (第 4 条中点轨迹): ras_glb_final_left.npz 的 mid_smooth
    mid_final = None
    if args.ras_final_npz:
        if not Path(args.ras_final_npz).exists():
            print(f'  [warn] --ras-final-npz 不存在: {args.ras_final_npz}')
        else:
            _rf = np.load(args.ras_final_npz)
            _key = 'mid_smooth' if 'mid_smooth' in _rf else 'mid_raw'
            mid_final = np.asarray(_rf[_key], dtype=np.float64)
            print(f'ras-final[{_key}]: {len(mid_final)} pts, '
                  f'center=({mid_final[:,0].mean():.2f},{mid_final[:,1].mean():.2f},'
                  f'{mid_final[:,2].mean():.2f})')

    # ===== stage0 预处理产物: 两个路径 (2026-09-13, 中点口径, 不画手腕) =====
    # 路径1 处理前: joints_raw (平滑前 FK 关节, _mano_to_sapien 已到 GLB 系, 未加 pre_offset)
    #          → MANO 拇食中点 (j20+j16)/2
    # 路径2 处理后: grip_pos + grip_R*[0.03689,0,0] (重定向+pre_translate 后的夹爪 finger_mid)
    s0_raw_mid, s0_grip_mid = None, None
    if args.stage0_npz:
        if not Path(args.stage0_npz).exists():
            print(f'  [warn] --stage0-npz 不存在: {args.stage0_npz}')
        else:
            _s0 = dict(np.load(args.stage0_npz, allow_pickle=True))
            _jr = _s0.get('joints_raw')
            if _jr is not None and np.asarray(_jr).ndim == 3:
                _jr = np.asarray(_jr, dtype=np.float64)
                s0_raw_mid = 0.5 * (_jr[:, 16, :] + _jr[:, 20, :])
                print(f'stage0 处理前 MANO mid: {len(s0_raw_mid)} pts, '
                      f'center=({s0_raw_mid[:,0].mean():.2f},{s0_raw_mid[:,1].mean():.2f},'
                      f'{s0_raw_mid[:,2].mean():.2f})')
            _gp, _gR = _s0.get('grip_pos'), _s0.get('grip_R')
            if _gp is not None and _gR is not None:
                _ff = np.array([0.03689, 0.0, 0.0])
                s0_grip_mid = np.asarray(_gp, dtype=np.float64) + \
                    np.einsum('fij,j->fi', np.asarray(_gR, dtype=np.float64), _ff)
                print(f'stage0 处理后夹爪 mid: {len(s0_grip_mid)} pts, '
                      f'center=({s0_grip_mid[:,0].mean():.2f},{s0_grip_mid[:,1].mean():.2f},'
                      f'{s0_grip_mid[:,2].mean():.2f})')

    # Objects — 唯一加载入口: 原始顶点 = Z-up 外参/GLB 世界坐标 (见 load_glb_objects 实证)
    geoms = []
    try:
        for node_name, v_world, col in load_glb_objects(glb_p):
            center = v_world.mean(0)
            bounds = np.array([v_world.min(0), v_world.max(0)])
            geoms.append((node_name, center, bounds, col))
    except Exception as e:
        print(f'  [warn] GLB load failed: {e}')

    for hidx, side in [(0,'Left'),(1,'Right')]:
        for tag, data in [('Midpoint Raw', mids_raw), ('Midpoint Depth', mids_da)]:
            if hidx in data:
                pts = data[hidx]
                print(f'{side} {tag}: {pts.shape[0]} fr, Z=[{pts[:,2].min():.3f},{pts[:,2].max():.3f}]')

    # ===== 桌面穿透检测: 手部轨迹 Z 低于场景物体最低点时告警 (说明上游数据有问题) =====
    real_geoms = [(nm, bd) for nm, _, bd, _ in geoms
                  if not any(k in nm.lower() for k in ('grid','floor','wall'))
                  and abs(bd[1,0]-bd[0,0]) < 2.0]
    if real_geoms:
        floor_z = min(bd[0, 2] for _, bd in real_geoms)
        for hidx, side in [(0,'Left'),(1,'Right')]:
            for tag, data in [('Raw', mids_raw), ('Depth', mids_da)]:
                if hidx in data and data[hidx][:, 2].min() < floor_z - 0.02:
                    print(f'  [warn] {side} Midpoint {tag} 穿透桌面: min Z={data[hidx][:,2].min():.3f} < 场景最低点 {floor_z:.3f} '
                          f'(HaWoR 上游重建/深度矫正问题, 非渲染错误)')

    # Load RAS camera positions
    ras_pos, ras_valid = load_ras_cameras(ras_dir)
    print(f'RAS camera: {ras_valid.sum()} valid frames, Z=[{ras_pos[ras_valid,2].min():.3f},{ras_pos[ras_valid,2].max():.3f}]')

    # 偏移轨迹变量初始化 (仅 --offset-frame 时在下方深度云分支内赋值)
    _offset_mask_traj = None
    _offset_mid_trajs = None

    # ===== MASK CENTER TRAJECTORY: HaWoR mask 中心 → RAS depth → GLB 3D =====
    mask_center_traj = []
    if args.show_depth_cloud:
        try:
            tracks_dirs = sorted(Path(hawor_dir).glob('tracks_*'))
            if tracks_dirs:
                mask_file = tracks_dirs[0] / 'model_masks.npy'
                if mask_file.exists():
                    masks = np.load(str(mask_file), allow_pickle=True)
                    if masks.ndim == 0: masks = masks.item()
                    if masks.ndim > 3:
                        if masks.shape[0] == 1: masks = masks[0]
                    n_frames = masks.shape[0]
                    K_ras, R_c2w_ras, t_c2w_ras, depth_files = load_ras_depth_data(ras_dir)
                    n_ras = len(depth_files)
                    # 对每个 RAS 帧尝试反投影
                    all_results = [None] * n_ras
                    for ri in range(n_ras):
                        fi = round(ri * (n_frames - 1) / max(n_ras - 1, 1))
                        fi = min(fi, n_frames - 1)
                        depth_img = cv2.imread(str(depth_files[ri]), cv2.IMREAD_UNCHANGED)
                        if depth_img is None: continue
                        mask = masks[fi]
                        result = unproject_mask_center_to_glb(mask, depth_img, K_ras,
                                                              R_c2w_ras[ri], t_c2w_ras[ri])
                        if result is not None:
                            all_results[ri] = result[0]
                    # 前向填充 + 后向填充补全所有帧
                    for i in range(1, n_ras):
                        if all_results[i] is None and all_results[i-1] is not None:
                            all_results[i] = all_results[i-1]
                    for i in range(n_ras - 2, -1, -1):
                        if all_results[i] is None and all_results[i+1] is not None:
                            all_results[i] = all_results[i+1]
                    for i in range(n_ras):
                        if all_results[i] is not None:
                            mask_center_traj.append((all_results[i], i, i))
                    if mask_center_traj:
                        traj_arr = np.array([p for p,_,_ in mask_center_traj])
                        mask_center_traj = [(traj_arr[i], a, b) for i, (_, a, b) in enumerate(mask_center_traj)]
                        print(f'RAS depth center: {len(mask_center_traj)} fr (RAS={n_ras}), Z=[{traj_arr[:,2].min():.3f},{traj_arr[:,2].max():.3f}]')

                    # ===== 帧偏移: 将指定帧的相机位置对齐到目标物体质心 =====
                    _offset_mask_traj = None
                    _offset_mid_trajs = None
                    if args.offset_frame >= 0 and mask_center_traj:
                        try:
                            objs = load_glb_objects(glb_p)   # 与 geoms 同一入口, 世界坐标
                            target_rgb = np.array([int(x) for x in args.offset_rgb.split(',')]) / 255.0 if args.offset_rgb else None
                            # 找粉色目标 (R>0.4, G<0.35, B>G)
                            if target_rgb is None:
                                for gn, v, c in objs:
                                    if 'grid' in gn: continue
                                    if c[0] > 0.4 and c[1] < 0.35 and c[2] > c[1]:
                                        target_rgb = c
                                        break
                            if target_rgb is not None:
                                # 找最近物体 (verts 即世界坐标)
                                best_dist, best_center, best_name = 1e9, None, None
                                for gn, v, c in objs:
                                    if 'grid' in gn: continue
                                    center = v.mean(0)
                                    dist = np.linalg.norm(c - target_rgb)
                                    if dist < best_dist:
                                        best_dist, best_center, best_name = dist, center, gn
                                if best_center is not None:
                                    best_center = best_center   # 已是 GLB 规范世界坐标
                                    f = min(args.offset_frame, len(mask_center_traj) - 1)
                                    _, _, ri = mask_center_traj[f]
                                    cam_pos = ras_pos[ri]
                                    offset_vec = best_center - cam_pos
                                    # 保存原始轨迹作为偏移对比
                                    _offset_mask_traj = np.array([p for p,_,_ in mask_center_traj]) + offset_vec
                                    # 对 MANO 手腕轨迹也施加同样的偏移
                                    _offset_mid_trajs = {}
                                    for hidx in [0, 1]:
                                        if hidx in mids_raw:
                                            _offset_mid_trajs[f'l_mid_raw_off' if hidx==0 else 'r_mid_raw_off'] = mids_raw[hidx] + offset_vec
                                        if hidx in mids_da:
                                            _offset_mid_trajs[f'l_mid_depth_off' if hidx==0 else 'r_mid_depth_off'] = mids_da[hidx] + offset_vec
                                    print(f'  帧偏移: frame={args.offset_frame} target={best_name} color=({target_rgb[0]:.2f},{target_rgb[1]:.2f},{target_rgb[2]:.2f})')
                                    print(f'         相机({ras_pos[ri][0]:.4f},{ras_pos[ri][1]:.4f},{ras_pos[ri][2]:.4f}) -> 物体({best_center[0]:.4f},{best_center[1]:.4f},{best_center[2]:.4f})')
                                    print(f'         offset=({offset_vec[0]:.4f},{offset_vec[1]:.4f},{offset_vec[2]:.4f}) 距离={np.linalg.norm(offset_vec):.4f}m')
                            else:
                                print('  未找到粉色物体作为偏移目标')
                        except Exception as e:
                            import traceback; traceback.print_exc()
                            print(f'  帧偏移失败: {e}')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  Mask center 加载失败: {e}')

    # ===== 手腕深度反投影轨迹: HaWoR 手腕 GLB → RAS 相机投影 → 深度读取 → 反投影回 GLB =====
    wrist_dpj = {}  # {hidx: np.array (M, 3)}
    if args.show_depth_cloud:
        try:
            K_ras, R_c2w_ras, t_c2w_ras, depth_files = load_ras_depth_data(ras_dir)
            n_ras = len(depth_files)
            # 缓存深度图
            depth_cache = {}
            n_hawor = len(d_raw['pred_trans'][0])
            for hidx in hand_ids:
                traj_pts = []
                skip_reasons = {'nan': 0, 'z_neg': 0, 'oob': 0, 'no_depth': 0, 'zero_depth': 0}
                for fi in range(n_hawor):
                    ri = round(fi * (n_ras - 1) / max(n_hawor - 1, 1))
                    ri = min(ri, n_ras - 1)
                    # pred_trans 在 HaWoR 输入相机系: 逐帧相机位姿 -> HaWoR 世界系,
                    # 再过 transform_params 的 (s,R,t) 相似变换 -> GLB 世界系, 才能投到 RAS 外参.
                    # HaWoR 世界 != GLB 世界 (实测 7: 3.17x/65deg, 121: 1.17x/76deg), 不过变换会打飞.
                    # (s,R,t) 是全局常量, 与帧号无关; 帧号只用来选 RAS 深度图.
                    wrist_h = d_raw['pred_trans'][hidx, fi]
                    if np.any(np.isnan(wrist_h)):
                        skip_reasons['nan'] += 1
                        continue
                    p_hawor = d_raw['R_c2w'][fi] @ wrist_h + d_raw['t_c2w'][fi]
                    wrist_ext = xform(np.array([p_hawor]), s, R, t)[0]
                    R_w2c_ras = R_c2w_ras[ri].T
                    t_w2c_ras = -R_w2c_ras @ t_c2w_ras[ri]
                    p_cam_ras = R_w2c_ras @ wrist_ext + t_w2c_ras
                    z_cam = p_cam_ras[2]
                    if z_cam <= 0:
                        skip_reasons['z_neg'] += 1
                        continue
                    # 像素坐标
                    u = K_ras[0,0] * p_cam_ras[0] / z_cam + K_ras[0,2]
                    v = K_ras[1,1] * p_cam_ras[1] / z_cam + K_ras[1,2]
                    ui, vi = int(round(u)), int(round(v))
                    # 读取深度 (缓存)
                    if ri not in depth_cache:
                        depth_cache[ri] = cv2.imread(str(depth_files[ri]), cv2.IMREAD_UNCHANGED)
                    depth_img = depth_cache[ri]
                    if depth_img is None:
                        skip_reasons['no_depth'] += 1
                        continue
                    if vi < 0 or vi >= depth_img.shape[0] or ui < 0 or ui >= depth_img.shape[1]:
                        skip_reasons['oob'] += 1
                        continue
                    z_depth = depth_img[vi, ui].astype(float) / 1000.0
                    if z_depth <= 0 or not np.isfinite(z_depth):
                        skip_reasons['zero_depth'] += 1
                        continue
                    # 用深度值反投影到 GLB
                    x_cam_dpj = (ui - K_ras[0,2]) * z_depth / K_ras[0,0]
                    y_cam_dpj = (vi - K_ras[1,2]) * z_depth / K_ras[1,1]
                    p_cam_dpj = np.array([x_cam_dpj, y_cam_dpj, z_depth])
                    p_ext_dpj = R_c2w_ras[ri] @ p_cam_dpj + t_c2w_ras[ri]
                    traj_pts.append(p_ext_dpj)   # 原生在 GLB 世界 (=外参世界)
                side = H_SID[hidx]
                print(f'  Wrist depth proj {side}: skips={skip_reasons}, valid={len(traj_pts)}')
                if traj_pts:
                    wrist_dpj[hidx] = np.array(traj_pts)
                    print(f'    Z=[{wrist_dpj[hidx][:,2].min():.3f},{wrist_dpj[hidx][:,2].max():.3f}]')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  Wrist depth projection failed: {e}')

    # ===== MATPLOTLIB =====
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=22, azim=-55)
    ax.set_title(f'GLB Alignment | transform_params: Scale={s:.3f}x Rot={ra:.1f}deg',
                 fontweight='bold', fontsize=11)

    if not args.no_axes:
        # 世界数据轴 = GLB 原生坐标轴 (无显示基旋转)
        draw_axes_arrows(ax, [0,0,0], scale=args.axis_scale*0.1, lw=6,
                         axes=[('red','Xw',[1,0,0]), ('green','Yw',[0,1,0]), ('blue','Zw',[0,0,1])])
        if not args.minimal:
            draw_camera_axes(ax, t0_glb, R @ d_raw['R_c2w'][0], scale=args.camera_scale)

    # Objects (用节点名, 方框颜色 = GLB 原色; 跳过 grid/floor/wall 辅助几何)
    for nm,ct,bd,col in geoms:
        low = nm.lower()
        if any(k in low for k in ('grid','floor','wall')): continue
        c = tuple(col)
        draw_bbox(ax, bd, c, alpha=0.15)
        ax.scatter(*ct, color=c, s=60, marker='o', edgecolors='k', linewidths=0.5, zorder=5)
        ax.text(ct[0],ct[1],ct[2]+0.05, nm, color=c, fontsize=7)

    # Alignment check
    h0 = cam_glb[cam_valid][:1] if cam_valid.any() else None
    r0 = ras_pos[ras_valid][:1] if ras_valid.any() else None
    if h0 is not None and r0 is not None:
        d0 = np.linalg.norm(h0[0] - r0[0])
        print(f'Frame 0 alignment: HaWoR={h0[0]}, RAS={r0[0]}, dist={d0:.6f}m')

    # ===== 所有轨迹，每个独立颜色 =====
    def plot_traj(ax, pts, color, label, lw=1.0, alpha=0.8, scatter=True):
        v = np.isfinite(pts).all(axis=1)  # 只过滤 NaN
        if not v.any(): return
        ax.plot(pts[v,0], pts[v,1], pts[v,2], color=color, lw=lw, alpha=alpha, label=label)
        if scatter:
            ax.scatter(pts[v,0], pts[v,1], pts[v,2], s=0.5, color=color, alpha=alpha*0.6, marker='.')
        mid = pts[v].mean(0)
        ax.text(mid[0], mid[1], mid[2]+0.05, label, color=color, fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Camera (--minimal 时隐藏)
    if not args.minimal:
        if cam_valid.any():
            plot_traj(ax, cam_glb[cam_valid], COLORS['hawor_cam'], 'HaWoR cam', lw=1.5, alpha=0.5)
        if ras_valid.any():
            plot_traj(ax, ras_pos[ras_valid], COLORS['ras_cam'], 'RAS cam', lw=1.5, alpha=0.8)

    # ===== [2026-08-26] 用户要求: 只保留 Midpoint, Wrist/Thumb/Index 注释掉 =====
    # Midpoint Raw (拇指+食指中点)
    for hidx, key, side in [(0,'l_mid_raw','L'), (1,'r_mid_raw','R')]:
        if hidx in mids_raw:
            plot_traj(ax, mids_raw[hidx], COLORS[key], f'{side} Midpoint Raw')
    # Midpoint Depth (拇指+食指中点, 深度矫正)
    for hidx, key, side in [(0,'l_mid_depth','L'), (1,'r_mid_depth','R')]:
        if hidx in mids_da:
            plot_traj(ax, mids_da[hidx], COLORS[key], f'{side} Midpoint Depth')
    # Midpoint RAS-GLB (RAS MANO 重建中点, 原生 GLB 世界系) [2026-09-13 补: 与 Plotly 对齐]
    if not args.minimal:
        for hidx, key, side in [(0,'l_mid_glb','L'), (1,'r_mid_glb','R')]:
            if hidx in mids_glb:
                plot_traj(ax, mids_glb[hidx], COLORS[key], f'{side} Midpoint RAS (mano_ras_reconstruction)')
    # RAS-GLB 合成版 (第 4 条): ras_glb_final_left.npz mid_smooth [2026-09-13]
    if mid_final is not None:
        plot_traj(ax, mid_final, 'darkorange', 'L Midpoint RAS-GLB final (synthesized)', lw=2.2)
    # 手部关节标记: 已注释 (2026-08-26 只保留 Midpoint)
    # for hidx, key, side in [(0,'l_mid_raw','L'), (1,'r_mid_raw','R')]:
    #     if hidx not in wrists_raw:
    #         continue
    #     c = COLORS[key]
    #     for jname, jd, mk in [('Thumb', thumbs_raw, '^'), ('Index', indexs_raw, 's'),
    #                           ('Wrist', wrists_raw, 'x')]:
    #         if hidx in jd:
    #             p = jd[hidx]
    #             ax.scatter(p[:,0], p[:,1], p[:,2], c=[c], marker=mk, s=22,
    #                        depthshade=False, label=f'{side} {jname} Raw')
    # for hidx, key, side in [(0,'l_mid_depth','L'), (1,'r_mid_depth','R')]:
    #     if hidx not in wrists_da:
    #         continue
    #     c = COLORS[key]
    #     for jname, jd, mk in [('Thumb', thumbs_da, 'v'), ('Index', indexs_da, 'D'),
    #                           ('Wrist', wrists_da, '+')]:
    #         if hidx in jd:
    #             p = jd[hidx]
    #             ax.scatter(p[:,0], p[:,1], p[:,2], c=[c], marker=mk, s=26,
    #                        alpha=0.45, depthshade=False,
    #                        label=f'{side} {jname} Depth')
    # RAS depth center / offset / wrist dpj (--minimal 时隐藏)
    if not args.minimal:
        if mask_center_traj:
            traj = np.array([p for p,_,_ in mask_center_traj])
            plot_traj(ax, traj, COLORS['ras_depth_ctr'], 'RAS depth center', lw=2.0, alpha=0.9)
        if _offset_mask_traj is not None:
            plot_traj(ax, _offset_mask_traj, COLORS['ras_depth_ctr_off'], 'RAS depth center (offset)', lw=2.0, alpha=0.5)
        # Offset midpoints (偏移后) [已注释 2026-08-26: Midpoint 不再渲染]
        # if _offset_mid_trajs is not None:
        #     for key, mpts in _offset_mid_trajs.items():
        #         plot_traj(ax, mpts, COLORS[key], key.replace('_', ' ').replace('off', '(offset)'), lw=1.0, alpha=0.5)
        # 手腕深度反投影 (HaWoR→GLB→RAS相机→读深度→GLB)
        for hidx, key, side in [(0,'wrist_dpj_l','L'), (1,'wrist_dpj_r','R')]:
            if hidx in wrist_dpj:
                plot_traj(ax, wrist_dpj[hidx], COLORS[key], f'{side} Wrist DepthProj', lw=2.0, alpha=0.9)

    # ===== stage0 两个路径 (2026-09-13, 中点口径): 处理前 MANO mid (紫) / 处理后夹爪 mid (黑) =====
    if s0_raw_mid is not None:
        plot_traj(ax, s0_raw_mid, 'purple', 'stage0 PRE Mano mid (j20+j16)/2 raw', lw=2.2)
    if s0_grip_mid is not None:
        plot_traj(ax, s0_grip_mid, 'black', 'stage0 POST Gripper finger_mid', lw=2.2)

    # 显示范围: --zoom 聚焦轨迹包围盒, 否则固定 [-2,2]^3 (用户指定)
    if args.zoom:
        _cand = [] if args.minimal else \
            [cam_glb[cam_valid], ras_pos[ras_valid]] + \
            [mids_glb[h] for h in mids_glb] + [wrist_dpj[h] for h in wrist_dpj]
        _cand += [mids_raw[h] for h in mids_raw] + [mids_da[h] for h in mids_da]
        if mid_final is not None: _cand.append(mid_final)
        if s0_raw_mid is not None: _cand.append(s0_raw_mid)
        if s0_grip_mid is not None: _cand.append(s0_grip_mid)
        _all = np.vstack([p[np.isfinite(p).all(axis=1)] for p in _cand if len(p)])
        _lo, _hi = _all.min(0) - 0.06, _all.max(0) + 0.06
        _span = float((_hi - _lo).max() / 2); _ctr = (_lo + _hi) / 2
        ax.set_xlim3d(_ctr[0]-_span, _ctr[0]+_span)
        ax.set_ylim3d(_ctr[1]-_span, _ctr[1]+_span)
        ax.set_zlim3d(_ctr[2]-_span, _ctr[2]+_span)
        print(f'zoom view: center={_ctr.round(3)}, span={_span:.3f}')
    else:
        # 固定显示范围 [-2, 2]^3 (用户指定; 自动包围盒过紧导致比例尺过大)
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError: pass
    nticks = 9
    from matplotlib.ticker import FixedLocator
    _r0, _r1 = (-2, 2) if not args.zoom else (_ctr[0]-_span, _ctr[0]+_span)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        ticks = np.linspace(_r0, _r1, nticks)
        axis.set_major_locator(FixedLocator(ticks))

    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=2)

    print("\n" + "=" * 70)
    print("LEGEND:")
    if s0_raw_mid is not None:
        print("  purple       = stage0 PRE Mano mid = raw (j20+j16)/2, before smooth/retarget")
    if s0_grip_mid is not None:
        print("  black        = stage0 POST Gripper finger_mid = grip_pos + R*0.03689, after pre_translate")
    if mid_final is not None:
        print("  darkorange   = L Midpoint RAS-GLB final = mid_smooth (synthesized savgol)")
    for key, desc in [
        ('hawor_cam',     'HaWoR camera trajectory'),
        ('ras_cam',       'RAS camera trajectory'),
        ('l_mid_raw',     'L Midpoint Raw (thumb+index)'),
        ('l_mid_depth',   'L Midpoint Depth (thumb+index, corrected)'),
        ('r_mid_raw',     'R Midpoint Raw (thumb+index)'),
        ('r_mid_depth',   'R Midpoint Depth (thumb+index, corrected)'),
        ('l_mid_glb',     'L Midpoint RAS (mano_ras_reconstruction, native GLB)'),
        ('r_mid_glb',     'R Midpoint RAS (mano_ras_reconstruction, native GLB)'),
        ('ras_depth_ctr', 'RAS depth center (mask→depth→GLB)'),
        ('ras_depth_ctr_off', 'RAS depth center (offset)'),
        ('l_mid_raw_off', 'L Midpoint Raw (offset)'),
        ('l_mid_depth_off', 'L Midpoint Depth (offset)'),
        ('r_mid_raw_off', 'R Midpoint Raw (offset)'),
        ('r_mid_depth_off', 'R Midpoint Depth (offset)'),
        ('wrist_dpj_l',   'L Wrist depth projection (HaWoR→GLB→RAS depth→GLB)'),
        ('wrist_dpj_r',   'R Wrist depth projection (HaWoR→GLB→RAS depth→GLB)'),
    ]:
        print(f"  {COLORS[key]:12s} = {desc}")
    print("=" * 70)

    plt.tight_layout()
    png = f'{args.output}_overlay.png'
    plt.savefig(png, dpi=150, bbox_inches='tight')
    print(f'Saved: {png}')
    plt.close()

    # ===== PLOTLY =====
    fig_p = go.Figure()
    ax_s, cam_s = args.axis_scale, args.camera_scale
    R_cam_glb = R @ d_raw['R_c2w'][0]

    if not args.no_axes:
        # World axes (GLB 原生坐标轴, 无显示基旋转)
        for clr,lbl,v in [('red','Xw',(1,0,0)),('green','Yw',(0,1,0)),('blue','Zw',(0,0,1))]:
            v = np.asarray(v, dtype=float)*ax_s
            fig_p.add_trace(go.Scatter3d(x=[0,v[0]],y=[0,v[1]],z=[0,v[2]], mode='lines+text',
                line=dict(color=clr,width=10), text=['',f' {lbl}'], textfont=dict(color=clr,size=24), showlegend=False))
        # Camera axes (--minimal 时隐藏)
        if not args.minimal:
            for i,(clr,lbl) in enumerate([('red','X'),('green','Y'),('blue','Z')]):
                v = R_cam_glb[:,i]*cam_s; tip = t0_glb+v
                fig_p.add_trace(go.Scatter3d(x=[t0_glb[0],tip[0]],y=[t0_glb[1],tip[1]],z=[t0_glb[2],tip[2]],
                    mode='lines+text', line=dict(color=clr,width=4,dash='dash'), text=['',f'cam{lbl}'], textfont=dict(color=clr,size=14), showlegend=False))
            look = R_cam_glb[:,2]*cam_s*0.8; tip_l = t0_glb+look
            fig_p.add_trace(go.Scatter3d(x=[t0_glb[0],tip_l[0]],y=[t0_glb[1],tip_l[1]],z=[t0_glb[2],tip_l[2]],
                mode='lines+text', line=dict(color='orange',width=6), text=['','Look'], textfont=dict(color='darkorange',size=14), showlegend=False))

    # Objects — center点 + bbox + 名称标签 (方框颜色 = GLB 原色; 跳过 grid/floor/wall 辅助几何)
    for nm,ct,bd,col in geoms:
        if any(k in nm.lower() for k in ('grid','floor','wall')): continue
        col = f'rgb({int(round(col[0]*255))},{int(round(col[1]*255))},{int(round(col[2]*255))})'
        p0,p1 = bd
        corners = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                           [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
        for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            fig_p.add_trace(go.Scatter3d(x=corners[e,0],y=corners[e,1],z=corners[e,2],
                mode='lines', line=dict(color=col,width=2), showlegend=False, hoverinfo='skip'))
        # 物体中心点 + 名称
        fig_p.add_trace(go.Scatter3d(x=[ct[0]],y=[ct[1]],z=[ct[2]], mode='markers+text',
            marker=dict(size=6, color=col, opacity=0.9), text=[nm], textposition='top center',
            textfont=dict(size=10, color=col), name=f'GLB: {nm}', showlegend=True))

    def add_plotly_traj(pts, color, name, width=2, size=3):
        v = np.isfinite(pts).all(axis=1)
        if not v.any(): return
        fig_p.add_trace(go.Scatter3d(x=pts[v,0],y=pts[v,1],z=pts[v,2], mode='lines+markers',
            marker=dict(size=size, color=color, opacity=0.7), line=dict(color=color, width=width), name=name,
            legendgroup=name))

    # Camera (--minimal 时隐藏)
    if not args.minimal:
        if cam_valid.any():
            add_plotly_traj(cam_glb[cam_valid], COLORS['hawor_cam'], 'HaWoR cam', width=2)
        if ras_valid.any():
            add_plotly_traj(ras_pos[ras_valid], COLORS['ras_cam'], 'RAS cam', width=2.5)

    # ===== [2026-08-26] 用户要求: 只保留 Midpoint, Wrist/Thumb/Index 注释掉 =====
    # Midpoint Raw (MANO thumb+index)
    for hidx, key, side in [(0,'l_mid_raw','L'), (1,'r_mid_raw','R')]:
        if hidx in mids_raw:
            add_plotly_traj(mids_raw[hidx], COLORS[key], f'{side} Midpoint Raw', size=3)
    # Midpoint Depth (MANO thumb+index, depth corrected)
    for hidx, key, side in [(0,'l_mid_depth','L'), (1,'r_mid_depth','R')]:
        if hidx in mids_da:
            add_plotly_traj(mids_da[hidx], COLORS[key], f'{side} Midpoint Depth', size=3)
    # Midpoint RAS-GLB (RAS MANO 重建, 原生 GLB 世界系, 零变换; --minimal 时隐藏)
    if not args.minimal:
        for hidx, key, side in [(0,'l_mid_glb','L'), (1,'r_mid_glb','R')]:
            if hidx in mids_glb:
                add_plotly_traj(mids_glb[hidx], COLORS[key], f'{side} Midpoint RAS (mano_ras_reconstruction)', size=3)
    # RAS-GLB 合成版 (第 4 条): ras_glb_final_left.npz mid_smooth
    if mid_final is not None:
        add_plotly_traj(mid_final, 'darkorange', 'L Midpoint RAS-GLB final (synthesized)', width=4)
    # 手部关节标记: 已注释 (2026-08-26 只保留 Midpoint)
    # for hidx, key, side in [(0,'l_mid_raw','L'), (1,'r_mid_raw','R')]:
    #     if hidx not in wrists_raw:
    #         continue
    #     c = COLORS[key]
    #     for jname, jd in [('Thumb', thumbs_raw), ('Index', indexs_raw), ('Wrist', wrists_raw)]:
    #         if hidx in jd:
    #             add_plotly_traj(jd[hidx], c, f'{side} {jname} Raw', size=2)
    # for hidx, key, side in [(0,'l_mid_depth','L'), (1,'r_mid_depth','R')]:
    #     if hidx not in wrists_da:
    #         continue
    #     c = COLORS[key]
    #     for jname, jd in [('Thumb', thumbs_da), ('Index', indexs_da), ('Wrist', wrists_da)]:
    #         if hidx in jd:
    #             add_plotly_traj(jd[hidx], c, f'{side} {jname} Depth (open)', size=2)
    # RAS depth center / offset / wrist dpj (--minimal 时隐藏)
    if not args.minimal:
        if mask_center_traj:
            traj = np.array([p for p,_,_ in mask_center_traj])
            add_plotly_traj(traj, COLORS['ras_depth_ctr'], 'RAS depth center', width=3, size=2.5)
        if _offset_mask_traj is not None:
            add_plotly_traj(_offset_mask_traj, COLORS['ras_depth_ctr_off'], 'RAS depth center (offset)', width=3, size=2.5)
        # 手腕深度反投影 (HaWoR→GLB→RAS相机→读深度→GLB)
        for hidx, key, side in [(0,'wrist_dpj_l','L'), (1,'wrist_dpj_r','R')]:
            if hidx in wrist_dpj:
                add_plotly_traj(wrist_dpj[hidx], COLORS[key], f'{side} Wrist DepthProj', width=3, size=2.5)
    # stage0 两个路径 (中点口径): 处理前 MANO mid (紫) / 处理后夹爪 mid (黑)
    if s0_raw_mid is not None:
        add_plotly_traj(s0_raw_mid, 'purple', 'stage0 PRE Mano mid (j20+j16)/2 raw', width=4)
    if s0_grip_mid is not None:
        add_plotly_traj(s0_grip_mid, 'black', 'stage0 POST Gripper finger_mid', width=4)
    # Offset midpoints (偏移后) [已注释 2026-08-26: Midpoint 不再渲染]
    # if _offset_mid_trajs is not None:
    #     for key, mpts in _offset_mid_trajs.items():
    #         add_plotly_traj(mpts, COLORS[key], key.replace('_', ' ').replace('off', '(offset)'), width=2, size=3)

    fig_p.update_layout(
        title=f'GLB Alignment | transform_params: Scale={s:.3f}x Rot={ra:.1f}deg'
              + (' | ZOOM' if args.zoom else ''),
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                   xaxis=dict(range=[-2, 2] if not args.zoom else [_ctr[0]-_span, _ctr[0]+_span]),
                   yaxis=dict(range=[-2, 2] if not args.zoom else [_ctr[1]-_span, _ctr[1]+_span]),
                   zaxis=dict(range=[-2, 2] if not args.zoom else [_ctr[2]-_span, _ctr[2]+_span]),
                   aspectmode='cube'),
        legend=dict(font=dict(size=11), itemsizing='constant', itemwidth=40),
        width=1400, height=900)
    html = f'{args.output}_overlay.html'
    fig_p.write_html(html)
    print(f'Saved: {html}')

if __name__ == '__main__':
    main()