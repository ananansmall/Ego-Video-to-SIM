#!/usr/bin/env python3
"""Debug: 对比 test_multi_displacement 和 001_align_scene 的差异"""
import numpy as np
import os, glob, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib import import_module

# 直接复制 001_align_scene.py 中的函数逻辑
def load_ras_extrinsics(ras_dir):
    ext_dir = os.path.join(ras_dir, 'extrinsics')
    files = sorted(glob.glob(os.path.join(ext_dir, '*.txt')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
    R_list, t_list = [], []
    for f in files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        R_c2w = R_w2c.T
        t_list.append(-R_c2w @ t_w2c)
        R_list.append(R_c2w)
    return np.array(R_list), np.array(t_list)

R_X = np.eye(3)

# 测试 7
name = '7'
ras_dir = '/home/an/data/ras/7_vggt_omega'
hawor_npz = '/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz'
params = '/home/an/data/hawor/7/output/alignment/transform_params.npz'

R_r, t_r = load_ras_extrinsics(ras_dir)
h = dict(np.load(hawor_npz, allow_pickle=True))
R_h = h['R_c2w']
t_h = h['t_c2w']
p = dict(np.load(params))
R_hg_cur = p['R_hand_to_glb']
t_hg_cur = p['t_hand_to_glb']
s_cur = float(p['scale_ratio'])

n_ras = len(t_r)
n_ha = len(t_h)
ha_indices = np.linspace(0, n_ha - 1, n_ras, dtype=int)

R_r0 = R_r[0]
R_h0 = R_h[0]

# === 方法A: 直接用 test_multi_displacement.py 的逻辑 ===
ras_cam0 = np.array([R_r0.T @ (t_r[i] - t_r[0]) for i in range(n_ras)])
ha_cam0_raw = np.array([R_h0.T @ (t_h[i] - t_h[0]) for i in range(n_ha)])
ha_cam0 = ha_cam0_raw[ha_indices]

K = 3
src_disps = np.array([ha_cam0[i] - ha_cam0[0] for i in range(1, K)])
dst_disps = np.array([ras_cam0[i] - ras_cam0[0] for i in range(1, K)])

print(f"=== 方法A (test_multi_displacement) ===")
print(f"src_disps:\n{src_disps}")
print(f"dst_disps:\n{dst_disps}")
print(f"src_disps norms: {[np.linalg.norm(x) for x in src_disps]}")
print(f"dst_disps norms: {[np.linalg.norm(x) for x in dst_disps]}")

# Kabsch
src_c = src_disps - src_disps.mean(axis=0)
dst_c = dst_disps - dst_disps.mean(axis=0)
H = src_c.T @ dst_c
U, S, Vt = np.linalg.svd(H)
R_A = Vt.T @ U.T
if np.linalg.det(R_A) < 0:
    Vt[-1] *= -1
    R_A = Vt.T @ U.T
ang_A = np.degrees(np.arccos(np.clip((np.trace(R_A)-1)/2, -1, 1)))
print(f"R_cam_align:\n{R_A}")
print(f"旋转角度: {ang_A:.2f}°")

R_new_A = R_r0 @ R_A @ R_h0.T
t_new_A = t_r[0] - s_cur * R_new_A @ t_h[0]
t_h_A = s_cur * (R_new_A @ t_h.T).T + t_new_A
t_h_A_s = t_h_A[ha_indices]

def pca_angle(a, b):
    ac = a - a.mean(0); bc = b - b.mean(0)
    _, _, Va = np.linalg.svd(ac, full_matrices=False)
    _, _, Vb = np.linalg.svd(bc, full_matrices=False)
    d = np.clip(np.dot(Va[0], Vb[0]), -1, 1)
    if d < 0: d = -d
    return np.degrees(np.arccos(d))

pca_A = pca_angle(t_h_A_s, t_r)
dist_A = np.linalg.norm(t_h_A_s - t_r, axis=1).mean()
print(f"PCA: {pca_A:.1f}°, 距离: {dist_A:.4f}m")

# === 方法B: 模拟 001_align_scene.py 的代码 ===
print(f"\n=== 方法B (001_align_scene.py) ===")
from importlib.machinery import SourceFileLoader
align = SourceFileLoader('align', '001_align_scene.py').load_module()

# 用同样的数据调用 _kabsch_displacement_align
src_pts = ha_cam0  # 在 001 中, src_pts = hawor_cam_in_cam0[hi_ras] = ha_cam0
dst_pts = ras_cam0  # 在 001 中, dst_pts = ras_cam_in_cam0 = ras_cam0

R_B = align._kabsch_displacement_align(src_pts, dst_pts, 3)
ang_B = np.degrees(np.arccos(np.clip((np.trace(R_B)-1)/2, -1, 1)))
print(f"R_cam_align:\n{R_B}")
print(f"旋转角度: {ang_B:.2f}°")

# 使用 001 中的 scale 计算
scale_ratio = align.umeyama_scale(src_pts, dst_pts)
print(f"scale_ratio (Umeyama): {scale_ratio:.6f} (原始: {s_cur:.6f})")

R_new_B = R_r0 @ R_B @ R_X @ R_h0.T
t_new_B = t_r[0] - scale_ratio * R_new_B @ t_h[0]
t_h_B = scale_ratio * (R_new_B @ t_h.T).T + t_new_B
t_h_B_s = t_h_B[ha_indices]

pca_B = pca_angle(t_h_B_s, t_r)
dist_B = np.linalg.norm(t_h_B_s - t_r, axis=1).mean()
print(f"PCA: {pca_B:.1f}°, 距离: {dist_B:.4f}m")

# 检查差异
print(f"\nR_A == R_B? {np.allclose(R_A, R_B)}")
print(f"R_A diff from R_B: {np.linalg.norm(R_A - R_B):.6f}")