#!/usr/bin/env python3
"""诊断脚本: 对比两种对齐方式，找出GLB放置错误的根本原因"""

import numpy as np
import trimesh
from glob import glob
import os

R_X = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_X

RAS_DIR = "/home/an/data/ras/my_7mp4_result"
HAWOR_FILE = "/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz"
GLB_PATH = os.path.join(RAS_DIR, "final_scene.glb")


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


def load_ras_cameras(ras_dir):
    ext_dir = os.path.join(ras_dir, 'extrinsics')
    ext_files = sorted(glob(os.path.join(ext_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    cam_positions = []
    for f in ext_files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_c2w = ext[:3, :3].T
        cam_pos = -R_c2w @ ext[:3, 3]
        cam_positions.append(cam_pos)
    return np.array(cam_positions)


def load_hawor_cameras(hawor_file):
    data = dict(np.load(hawor_file, allow_pickle=True))
    return data['t_c2w'], data['R_c2w'], data['pred_trans'], data['pred_valid']


print("=" * 70)
print("1. 加载原始数据")
print("=" * 70)

ras_cam_ydown = load_ras_cameras(RAS_DIR)
ras_cam_yup = (R_X @ ras_cam_ydown.T).T
hawor_cam, hawor_R_c2w, pred_trans, pred_valid = load_hawor_cameras(HAWOR_FILE)

print(f"RAS 相机: {len(ras_cam_ydown)} 帧")
print(f"  y-down[0] = {ras_cam_ydown[0]}")
print(f"  y-up[0]   = {ras_cam_yup[0]}")
print(f"  y-down 范围: x[{ras_cam_ydown[:,0].min():.4f},{ras_cam_ydown[:,0].max():.4f}]"
      f" y[{ras_cam_ydown[:,1].min():.4f},{ras_cam_ydown[:,1].max():.4f}]"
      f" z[{ras_cam_ydown[:,2].min():.4f},{ras_cam_ydown[:,2].max():.4f}]")

print(f"\nHaWoR 相机: {len(hawor_cam)} 帧")
print(f"  cam[0] = {hawor_cam[0]}")
print(f"  范围: x[{hawor_cam[:,0].min():.4f},{hawor_cam[:,0].max():.4f}]"
      f" y[{hawor_cam[:,1].min():.4f},{hawor_cam[:,1].max():.4f}]"
      f" z[{hawor_cam[:,2].min():.4f},{hawor_cam[:,2].max():.4f}]")

print(f"\nRAS y-down 跨度: {np.linalg.norm(ras_cam_ydown.max(axis=0) - ras_cam_ydown.min(axis=0)):.4f}")
print(f"RAS y-up   跨度: {np.linalg.norm(ras_cam_yup.max(axis=0) - ras_cam_yup.min(axis=0)):.4f}")
print(f"HaWoR      跨度: {np.linalg.norm(hawor_cam.max(axis=0) - hawor_cam.min(axis=0)):.4f}")

print("\n" + "=" * 70)
print("2. 加载 GLB 顶点 (两种方式)")
print("=" * 70)

tm_scene = trimesh.load(GLB_PATH, force='scene')

# 方式A: dump() - 应用场景图变换
meshes_dump = tm_scene.dump()
verts_dump = np.vstack([m.vertices for m in meshes_dump])
glb_center_dump = verts_dump.mean(axis=0)

# 方式B: geometry.items() - 原始顶点 (不应用场景图变换)
verts_geom = []
for name, geom in tm_scene.geometry.items():
    verts_geom.append(geom.vertices)
verts_geom = np.vstack(verts_geom)
glb_center_geom = verts_geom.mean(axis=0)

print(f"GLB dump() 顶点数: {len(verts_dump)}")
print(f"  中心: {glb_center_dump}")
print(f"  范围: min={verts_dump.min(axis=0)}, max={verts_dump.max(axis=0)}")

print(f"\nGLB geometry 顶点数: {len(verts_geom)}")
print(f"  中心: {glb_center_geom}")
print(f"  范围: min={verts_geom.min(axis=0)}, max={verts_geom.max(axis=0)}")

print(f"\n两种方式中心差异: {np.linalg.norm(glb_center_dump - glb_center_geom):.6f}")
print(f"  dump - geom = {glb_center_dump - glb_center_geom}")

# 检查场景图变换
print(f"\n场景图节点数: {len(tm_scene.graph.nodes)}")
for node_name in list(tm_scene.graph.nodes)[:5]:
    transform = tm_scene.graph.get(node_name)[0]
    if transform is not None:
        print(f"  节点 {node_name}: 变换矩阵 =\n{transform}")

print("\n" + "=" * 70)
print("3. 帧对应")
print("=" * 70)

n_ras = len(ras_cam_ydown)
n_hawor = len(hawor_cam)
common_frames = []
for ras_i in range(n_ras):
    hawor_i = round(ras_i * (n_hawor - 1) / (n_ras - 1)) if n_ras > 1 else 0
    common_frames.append((ras_i, hawor_i))
print(f"RAS {n_ras} 帧 -> HaWoR {n_hawor} 帧, 共 {len(common_frames)} 对")

print("\n" + "=" * 70)
print("4. Umeyama 对齐 - 方式A (01_align_scene.py: RAS y-up)")
print("=" * 70)

src_A = np.array([hawor_cam[hi] for _, hi in common_frames])
dst_A = np.array([ras_cam_yup[ri] for ri, _ in common_frames])
s_A, R_A, t_A = umeyama_align(src_A, dst_A)
angle_A = np.degrees(np.arccos(np.clip((np.trace(R_A) - 1) / 2, -1, 1)))
print(f"尺度: {s_A:.4f}")
print(f"旋转角度: {angle_A:.2f}°")
print(f"旋转矩阵:\n{R_A}")
print(f"平移: {t_A}")

aligned_A = s_A * (R_A @ src_A.T).T + t_A
errors_A = np.linalg.norm(aligned_A - dst_A, axis=1)
print(f"对齐误差: mean={errors_A.mean():.6f}m, max={errors_A.max():.6f}m")

# GLB dump 顶点 → HaWoR → SAPIEN
s_inv_A = 1.0 / s_A
R_inv_A = R_A.T
t_inv_A = -s_inv_A * (R_inv_A @ t_A)

glb_hawor_A = s_inv_A * (R_inv_A @ glb_center_dump) + t_inv_A
glb_sapien_A = RXWORLD_TO_SAPIEN @ glb_hawor_A
print(f"\nGLB dump 中心 → HaWoR: {glb_hawor_A}")
print(f"GLB dump 中心 → SAPIEN: {glb_sapien_A}")

print("\n" + "=" * 70)
print("5. Umeyama 对齐 - 方式B (yingshe.py: RAS 原始坐标)")
print("=" * 70)

src_B = np.array([hawor_cam[hi] for _, hi in common_frames])
dst_B = np.array([ras_cam_ydown[ri] for ri, _ in common_frames])
s_B, R_B, t_B = umeyama_align(src_B, dst_B)
angle_B = np.degrees(np.arccos(np.clip((np.trace(R_B) - 1) / 2, -1, 1)))
print(f"尺度: {s_B:.4f}")
print(f"旋转角度: {angle_B:.2f}°")
print(f"旋转矩阵:\n{R_B}")
print(f"平移: {t_B}")

aligned_B = s_B * (R_B @ src_B.T).T + t_B
errors_B = np.linalg.norm(aligned_B - dst_B, axis=1)
print(f"对齐误差: mean={errors_B.mean():.6f}m, max={errors_B.max():.6f}m")

# GLB geometry 顶点 → HaWoR → SAPIEN
s_inv_B = 1.0 / s_B
R_inv_B = R_B.T
t_inv_B = -s_inv_B * (R_inv_B @ t_B)

glb_hawor_B = s_inv_B * (R_inv_B @ glb_center_geom) + t_inv_B
glb_sapien_B = RXWORLD_TO_SAPIEN @ glb_hawor_B
print(f"\nGLB geom 中心 → HaWoR: {glb_hawor_B}")
print(f"GLB geom 中心 → SAPIEN: {glb_sapien_B}")

print("\n" + "=" * 70)
print("6. 手部位置 (HaWoR render world → SAPIEN)")
print("=" * 70)

for hi in [0, 1]:
    v = pred_valid[hi]
    if v.any():
        hand_mean_render = pred_trans[hi, v].mean(axis=0)
        hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render
        label = "左手" if hi == 0 else "右手"
        print(f"{label} 均值 (render y-up): {hand_mean_render}")
        print(f"{label} 均值 (SAPIEN z-up): {hand_mean_sapien}")

        dist_A = np.linalg.norm(hand_mean_sapien - glb_sapien_A)
        dist_B = np.linalg.norm(hand_mean_sapien - glb_sapien_B)
        print(f"  → GLB距离 (方式A, dump+yup): {dist_A:.4f}m")
        print(f"  → GLB距离 (方式B, geom+ydown): {dist_B:.4f}m")

print("\n" + "=" * 70)
print("7. 交叉验证: 所有可能的组合")
print("=" * 70)

combos = [
    ("A1: RAS yup + GLB dump", ras_cam_yup, glb_center_dump),
    ("A2: RAS yup + GLB geom", ras_cam_yup, glb_center_geom),
    ("A3: RAS ydown + GLB dump", ras_cam_ydown, glb_center_dump),
    ("A4: RAS ydown + GLB geom", ras_cam_ydown, glb_center_geom),
]

for hi in [0, 1]:
    v = pred_valid[hi]
    if not v.any():
        continue
    hand_mean_render = pred_trans[hi, v].mean(axis=0)
    hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render
    label = "左手" if hi == 0 else "右手"

    print(f"\n--- {label} ---")
    for name, ras_cam, glb_center in combos:
        src = np.array([hawor_cam[h_i] for _, h_i in common_frames])
        dst = np.array([ras_cam[r_i] for r_i, _ in common_frames])
        s, R, t = umeyama_align(src, dst)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        s_inv = 1.0 / s
        R_inv = R.T
        t_inv = -s_inv * (R_inv @ t)
        glb_hawor = s_inv * (R_inv @ glb_center) + t_inv
        glb_sapien = RXWORLD_TO_SAPIEN @ glb_hawor
        dist = np.linalg.norm(hand_mean_sapien - glb_sapien)
        print(f"  {name}: 角度={angle:.1f}°, 尺度={s:.4f}, 手-GLB距离={dist:.4f}m, GLB_SAPIEN={glb_sapien}")

print("\n" + "=" * 70)
print("8. 关键检查: RAS 外参到底是什么格式?")
print("=" * 70)

ext_dir = os.path.join(RAS_DIR, 'extrinsics')
ext_files = sorted(glob(os.path.join(ext_dir, '*.txt')),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))
ext0 = np.loadtxt(ext_files[0])
if ext0.shape == (3, 4):
    ext0 = np.vstack([ext0, [0, 0, 0, 1]])
print(f"RAS 外参[0] (4x4):\n{ext0}")
print(f"  左上3x3 (R_w2c?):\n{ext0[:3,:3]}")
print(f"  det(R) = {np.linalg.det(ext0[:3,:3]):.6f}")
print(f"  R_w2c[0] 接近单位矩阵? {np.allclose(ext0[:3,:3], np.eye(3), atol=0.1)}")

R_c2w_0 = ext0[:3, :3].T
cam_pos_v1 = -R_c2w_0 @ ext0[:3, 3]  # 方式1: cam_pos = -R_c2w @ t_w2c
cam_pos_v2 = ext0[:3, 3]              # 方式2: 如果外参是 c2w, 则直接是相机位置

print(f"\n  cam_pos (方式1: -R_c2w@t_w2c) = {cam_pos_v1}")
print(f"  cam_pos (方式2: 直接取t)       = {cam_pos_v2}")
print(f"  两种方式差异: {np.linalg.norm(cam_pos_v1 - cam_pos_v2):.6f}")

print("\n" + "=" * 70)
print("9. 检查 RAS 外参是否是 c2w 格式")
print("=" * 70)

cam_positions_v1 = []
cam_positions_v2 = []
for f in ext_files:
    ext = np.loadtxt(f)
    if ext.shape == (3, 4):
        ext = np.vstack([ext, [0, 0, 0, 1]])
    R_c2w = ext[:3, :3].T
    cam_positions_v1.append(-R_c2w @ ext[:3, 3])
    cam_positions_v2.append(ext[:3, 3])

cam_v1 = np.array(cam_positions_v1)
cam_v2 = np.array(cam_positions_v2)

print(f"方式1 (-R_c2w@t) 范围: {cam_v1.min(axis=0)} ~ {cam_v1.max(axis=0)}")
print(f"方式2 (直接取t) 范围: {cam_v2.min(axis=0)} ~ {cam_v2.max(axis=0)}")
print(f"方式1 跨度: {np.linalg.norm(cam_v1.max(axis=0) - cam_v1.min(axis=0)):.4f}")
print(f"方式2 跨度: {np.linalg.norm(cam_v2.max(axis=0) - cam_v2.min(axis=0)):.4f}")

# 用方式2 (c2w) 做对齐
print("\n--- 用方式2 (直接取t作为相机位置) 做对齐 ---")
for hi in [0, 1]:
    v = pred_valid[hi]
    if not v.any():
        continue
    hand_mean_render = pred_trans[hi, v].mean(axis=0)
    hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render
    label = "左手" if hi == 0 else "右手"

    for glb_name, glb_center in [("dump", glb_center_dump), ("geom", glb_center_geom)]:
        src = np.array([hawor_cam[h_i] for _, h_i in common_frames])
        dst_v2_ydown = np.array([cam_v2[r_i] for r_i, _ in common_frames])
        dst_v2_yup = (R_X @ dst_v2_ydown.T).T

        for dst_name, dst in [("ydown", dst_v2_ydown), ("yup", dst_v2_yup)]:
            s, R, t = umeyama_align(src, dst)
            angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            s_inv = 1.0 / s
            R_inv = R.T
            t_inv = -s_inv * (R_inv @ t)
            glb_hawor = s_inv * (R_inv @ glb_center) + t_inv
            glb_sapien = RXWORLD_TO_SAPIEN @ glb_hawor
            dist = np.linalg.norm(hand_mean_sapien - glb_sapien)
            print(f"  {label} | v2_{dst_name}+GLB_{glb_name}: 角度={angle:.1f}°, 尺度={s:.4f}, 距离={dist:.4f}m")

print("\n完成!")
