#!/usr/bin/env python3
"""诊断脚本: 对比 yingshe.py 和 01_align_scene.py 两种对齐方式"""

import numpy as np
import trimesh
from glob import glob
import os

R_X = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_X

RAS_DIR = "/home/an/data/ras/my_7mp4_result"
HAWOR_NPZ = "/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz"


def load_ras_cameras(ras_output):
    ext_dir = os.path.join(ras_output, 'extrinsics')
    ext_files = sorted(glob(os.path.join(ext_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    cam_positions = []
    R_c2w_list = []
    for f in ext_files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_c2w = ext[:3, :3].T
        cam_pos = -R_c2w @ ext[:3, 3]
        cam_positions.append(cam_pos)
        R_c2w_list.append(R_c2w)
    return np.array(cam_positions), np.array(R_c2w_list)


def umeyama_align(src_pts, dst_pts):
    assert src_pts.shape == dst_pts.shape
    n = src_pts.shape[0]
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
    if sigma_src < 1e-8:
        return 1.0, np.eye(3), dst_mean - src_mean
    scale = sigma_dst / sigma_src
    cov = (dst_centered.T @ src_centered) / n
    U, D, VH = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[2, 2] = -1
    R = U @ S @ VH
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


def find_frame_correspondence(n_ras, n_hawor):
    common_frames = []
    for ras_i in range(n_ras):
        hawor_i = round(ras_i * (n_hawor - 1) / (n_ras - 1)) if n_ras > 1 else 0
        common_frames.append((ras_i, hawor_i))
    return common_frames


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("1. 加载数据")
print("=" * 70)

ras_cam_ydown, ras_R_c2w = load_ras_cameras(RAS_DIR)
ras_cam_yup = (R_X @ ras_cam_ydown.T).T

hawor_data = dict(np.load(HAWOR_NPZ, allow_pickle=True))
hawor_cam = hawor_data['t_c2w']
hawor_R_c2w = hawor_data['R_c2w']
pred_trans = hawor_data['pred_trans']
pred_valid = hawor_data['pred_valid']

n_ras = len(ras_cam_ydown)
n_hawor = len(hawor_cam)
common_frames = find_frame_correspondence(n_ras, n_hawor)

print(f"RAS: {n_ras} 帧")
print(f"  cam_ydown[0] = {ras_cam_ydown[0]}")
print(f"  cam_yup[0]   = {ras_cam_yup[0]}")
print(f"  cam_ydown 范围: x[{ras_cam_ydown[:,0].min():.4f},{ras_cam_ydown[:,0].max():.4f}]"
      f" y[{ras_cam_ydown[:,1].min():.4f},{ras_cam_ydown[:,1].max():.4f}]"
      f" z[{ras_cam_ydown[:,2].min():.4f},{ras_cam_ydown[:,2].max():.4f}]")

print(f"\nHaWoR: {n_hawor} 帧")
print(f"  cam[0] = {hawor_cam[0]}")
print(f"  cam 范围: x[{hawor_cam[:,0].min():.4f},{hawor_cam[:,0].max():.4f}]"
      f" y[{hawor_cam[:,1].min():.4f},{hawor_cam[:,1].max():.4f}]"
      f" z[{hawor_cam[:,2].min():.4f},{hawor_cam[:,2].max():.4f}]")

# ============================================================
# 2. GLB 顶点对比: geometry.items() vs dump()
# ============================================================
print("\n" + "=" * 70)
print("2. GLB 顶点对比: geometry.items() vs dump()")
print("=" * 70)

glb_path = os.path.join(RAS_DIR, 'final_scene.glb')
tm_scene = trimesh.load(glb_path, force='scene')

# 方式A: geometry.items() (yingshe.py 的方式)
verts_geom = []
for name, geom in tm_scene.geometry.items():
    verts_geom.append(geom.vertices)
verts_geom = np.vstack(verts_geom)
center_geom = verts_geom.mean(axis=0)

# 方式B: dump() (01_align_scene.py 的方式)
meshes_dump = tm_scene.dump()
verts_dump = np.vstack([m.vertices for m in meshes_dump])
center_dump = verts_dump.mean(axis=0)

print(f"geometry.items() 中心: {center_geom}")
print(f"dump() 中心:         {center_dump}")
print(f"中心差异:            {np.linalg.norm(center_geom - center_dump):.6f}")
print(f"geometry.items() 范围: min={verts_geom.min(axis=0)}, max={verts_geom.max(axis=0)}")
print(f"dump() 范围:         min={verts_dump.min(axis=0)}, max={verts_dump.max(axis=0)}")

# 检查场景图变换
print(f"\n场景图节点数: {len(tm_scene.graph.nodes)}")
for node_name in list(tm_scene.graph.nodes)[:5]:
    transform = tm_scene.graph.get(node_name)[0]
    if transform is not None:
        print(f"  节点 {node_name}: 变换矩阵\n{transform}")

# ============================================================
# 3. 方式A: yingshe.py 的对齐方式 (RAS相机不转换, 用raw geometry)
# ============================================================
print("\n" + "=" * 70)
print("3. 方式A: yingshe.py 对齐 (RAS相机直接用, GLB用geometry.items())")
print("=" * 70)

src_pts_A = np.array([hawor_cam[hi] for _, hi in common_frames])
dst_pts_A = np.array([ras_cam_ydown[ri] for ri, _ in common_frames])

s_A, R_A, t_A = umeyama_align(src_pts_A, dst_pts_A)
angle_A = np.degrees(np.arccos(np.clip((np.trace(R_A) - 1) / 2, -1, 1)))
print(f"尺度: {s_A:.6f}")
print(f"旋转角度: {angle_A:.2f}°")
print(f"旋转矩阵:\n{R_A}")
print(f"平移: {t_A}")

aligned_A = s_A * (R_A @ src_pts_A.T).T + t_A
errors_A = np.linalg.norm(aligned_A - dst_pts_A, axis=1)
print(f"对齐误差: mean={errors_A.mean():.6f}m, max={errors_A.max():.6f}m")

s_inv_A = 1.0 / s_A
R_inv_A = R_A.T
t_inv_A = -s_inv_A * (R_inv_A @ t_A)

# GLB center → HaWoR
glb_center_hawor_A = s_inv_A * (R_inv_A @ center_geom) + t_inv_A
print(f"\nGLB中心 (raw geometry): {center_geom}")
print(f"GLB中心 (HaWoR render y-up): {glb_center_hawor_A}")

# 手部中心 → HaWoR
for hi in [0, 1]:
    v = pred_valid[hi]
    if v.any():
        hand_mean = pred_trans[hi, v].mean(axis=0)
        dist = np.linalg.norm(hand_mean - glb_center_hawor_A)
        label = "左手" if hi == 0 else "右手"
        print(f"{label}均值 (HaWoR y-up): {hand_mean}")
        print(f"{label} → GLB中心距离: {dist:.4f}m")

# GLB → SAPIEN
glb_sapien_A = (RXWORLD_TO_SAPIEN @ glb_center_hawor_A)
hand_sapien = (RXWORLD_TO_SAPIEN @ pred_trans[1, pred_valid[1]].mean(axis=0))
dist_sapien_A = np.linalg.norm(hand_sapien - glb_sapien_A)
print(f"\nGLB中心 (SAPIEN z-up): {glb_sapien_A}")
print(f"右手均值 (SAPIEN z-up): {hand_sapien}")
print(f"手-GLB距离 (SAPIEN): {dist_sapien_A:.4f}m")

# ============================================================
# 4. 方式B: 01_align_scene.py 的对齐方式 (RAS相机转y-up, GLB用dump)
# ============================================================
print("\n" + "=" * 70)
print("4. 方式B: 01_align_scene.py 对齐 (RAS相机转y-up, GLB用dump())")
print("=" * 70)

src_pts_B = np.array([hawor_cam[hi] for _, hi in common_frames])
dst_pts_B = np.array([ras_cam_yup[ri] for ri, _ in common_frames])

s_B, R_B, t_B = umeyama_align(src_pts_B, dst_pts_B)
angle_B = np.degrees(np.arccos(np.clip((np.trace(R_B) - 1) / 2, -1, 1)))
print(f"尺度: {s_B:.6f}")
print(f"旋转角度: {angle_B:.2f}°")
print(f"旋转矩阵:\n{R_B}")
print(f"平移: {t_B}")

aligned_B = s_B * (R_B @ src_pts_B.T).T + t_B
errors_B = np.linalg.norm(aligned_B - dst_pts_B, axis=1)
print(f"对齐误差: mean={errors_B.mean():.6f}m, max={errors_B.max():.6f}m")

s_inv_B = 1.0 / s_B
R_inv_B = R_B.T
t_inv_B = -s_inv_B * (R_inv_B @ t_B)

# GLB center → HaWoR
glb_center_hawor_B = s_inv_B * (R_inv_B @ center_dump) + t_inv_B
print(f"\nGLB中心 (dump y-up): {center_dump}")
print(f"GLB中心 (HaWoR render y-up): {glb_center_hawor_B}")

# 手部中心 → HaWoR
for hi in [0, 1]:
    v = pred_valid[hi]
    if v.any():
        hand_mean = pred_trans[hi, v].mean(axis=0)
        dist = np.linalg.norm(hand_mean - glb_center_hawor_B)
        label = "左手" if hi == 0 else "右手"
        print(f"{label}均值 (HaWoR y-up): {hand_mean}")
        print(f"{label} → GLB中心距离: {dist:.4f}m")

# GLB → SAPIEN
glb_sapien_B = (RXWORLD_TO_SAPIEN @ glb_center_hawor_B)
dist_sapien_B = np.linalg.norm(hand_sapien - glb_sapien_B)
print(f"\nGLB中心 (SAPIEN z-up): {glb_sapien_B}")
print(f"右手均值 (SAPIEN z-up): {hand_sapien}")
print(f"手-GLB距离 (SAPIEN): {dist_sapien_B:.4f}m")

# ============================================================
# 5. 方式C: yingshe.py方式但用dump()顶点
# ============================================================
print("\n" + "=" * 70)
print("5. 方式C: yingshe.py对齐 + dump()顶点")
print("=" * 70)

glb_center_hawor_C = s_inv_A * (R_inv_A @ center_dump) + t_inv_A
print(f"GLB中心 (dump y-up): {center_dump}")
print(f"GLB中心 (HaWoR render y-up): {glb_center_hawor_C}")

for hi in [0, 1]:
    v = pred_valid[hi]
    if v.any():
        hand_mean = pred_trans[hi, v].mean(axis=0)
        dist = np.linalg.norm(hand_mean - glb_center_hawor_C)
        label = "左手" if hi == 0 else "右手"
        print(f"{label} → GLB中心距离: {dist:.4f}m")

glb_sapien_C = (RXWORLD_TO_SAPIEN @ glb_center_hawor_C)
dist_sapien_C = np.linalg.norm(hand_sapien - glb_sapien_C)
print(f"手-GLB距离 (SAPIEN): {dist_sapien_C:.4f}m")

# ============================================================
# 6. 方式D: 01方式但用geometry.items()顶点
# ============================================================
print("\n" + "=" * 70)
print("6. 方式D: 01_align_scene.py对齐 + geometry.items()顶点")
print("=" * 70)

glb_center_hawor_D = s_inv_B * (R_inv_B @ center_geom) + t_inv_B
print(f"GLB中心 (raw geometry): {center_geom}")
print(f"GLB中心 (HaWoR render y-up): {glb_center_hawor_D}")

for hi in [0, 1]:
    v = pred_valid[hi]
    if v.any():
        hand_mean = pred_trans[hi, v].mean(axis=0)
        dist = np.linalg.norm(hand_mean - glb_center_hawor_D)
        label = "左手" if hi == 0 else "右手"
        print(f"{label} → GLB中心距离: {dist:.4f}m")

glb_sapien_D = (RXWORLD_TO_SAPIEN @ glb_center_hawor_D)
dist_sapien_D = np.linalg.norm(hand_sapien - glb_sapien_D)
print(f"手-GLB距离 (SAPIEN): {dist_sapien_D:.4f}m")

# ============================================================
# 7. 汇总对比
# ============================================================
print("\n" + "=" * 70)
print("7. 汇总对比")
print("=" * 70)
print(f"方式A (yingshe: RAS原始 + geometry.items()): 手-GLB距离 = {dist_sapien_A:.4f}m")
print(f"方式B (01_align: RAS转y-up + dump()):        手-GLB距离 = {dist_sapien_B:.4f}m")
print(f"方式C (yingshe对齐 + dump()顶点):             手-GLB距离 = {dist_sapien_C:.4f}m")
print(f"方式D (01对齐 + geometry.items()顶点):         手-GLB距离 = {dist_sapien_D:.4f}m")

best = min([(dist_sapien_A, "A"), (dist_sapien_B, "B"), (dist_sapien_C, "C"), (dist_sapien_D, "D")])
print(f"\n最佳方式: 方式{best[1]} (距离={best[0]:.4f}m)")
