#!/usr/bin/env python3
"""诊断脚本3: 验证正确的对齐方式 - 使用GLB原始顶点 + R_X转换"""

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


def load_hawor_cameras(hawor_file):
    data = dict(np.load(hawor_file, allow_pickle=True))
    return data['t_c2w'], data['R_c2w'], data['pred_trans'], data['pred_valid']


ras_cam_ydown, ras_R_c2w = load_ras_cameras(RAS_DIR)
ras_cam_yup = (R_X @ ras_cam_ydown.T).T
hawor_cam, hawor_R_c2w, pred_trans, pred_valid = load_hawor_cameras(HAWOR_FILE)

n_ras = len(ras_cam_ydown)
n_hawor = len(hawor_cam)

tm_scene = trimesh.load(GLB_PATH, force='scene')

# 原始顶点 (y-down, z-forward, 与RAS相机同坐标系)
verts_geom_list = []
for name, geom in tm_scene.geometry.items():
    verts_geom_list.append(geom.vertices)
verts_geom = np.vstack(verts_geom_list)
glb_center_geom = verts_geom.mean(axis=0)

# dump顶点 (错误的坐标系)
meshes_dump = tm_scene.dump()
verts_dump = np.vstack([m.vertices for m in meshes_dump])
glb_center_dump = verts_dump.mean(axis=0)

# 原始顶点转换到 y-up
glb_center_geom_yup = R_X @ glb_center_geom

# 手部位置
hand_idx = 0 if pred_valid[0].any() else 1
v = pred_valid[hand_idx]
hand_mean_render = pred_trans[hand_idx, v].mean(axis=0)
hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render
label = "左手" if hand_idx == 0 else "右手"

print("=" * 70)
print("正确对齐方式验证")
print("=" * 70)
print(f"使用 {label}")
print(f"手部均值 (SAPIEN): {hand_mean_sapien}")
print(f"手部均值 (render y-up): {hand_mean_render}")

print(f"\nGLB原始中心 (y-down): {glb_center_geom}")
print(f"GLB原始中心 (y-up):   {glb_center_geom_yup}")
print(f"GLB dump中心 (y-up?): {glb_center_dump}")

# 帧对应
common_frames_uniform = [(ri, round(ri * (n_hawor - 1) / (n_ras - 1))) for ri in range(n_ras)]
common_frames_direct = [(ri, ri) for ri in range(min(n_ras, n_hawor))]

print("\n" + "=" * 70)
print("方案对比: 所有可能的正确组合")
print("=" * 70)

# 所有可能的正确组合:
# 1. RAS y-up + GLB原始 y-up (R_X转换) + Umeyama
# 2. RAS y-up + GLB原始 y-up + 固定R=I (从相机朝向得出)
# 3. RAS y-down + GLB原始 y-down + Umeyama (两者同坐标系)
# 4. RAS y-up + GLB dump + Umeyama (当前01的方式)

tests = [
    ("A: RAS yup + GLB geom yup + Umeyama (均匀)", ras_cam_yup, glb_center_geom_yup, common_frames_uniform, True),
    ("B: RAS yup + GLB geom yup + Umeyama (直接)", ras_cam_yup, glb_center_geom_yup, common_frames_direct, True),
    ("C: RAS ydown + GLB geom ydown + Umeyama (均匀)", ras_cam_ydown, glb_center_geom, common_frames_uniform, True),
    ("D: RAS ydown + GLB geom ydown + Umeyama (直接)", ras_cam_ydown, glb_center_geom, common_frames_direct, True),
    ("E: RAS yup + GLB dump + Umeyama (当前方式)", ras_cam_yup, glb_center_dump, common_frames_uniform, True),
    ("F: RAS yup + GLB geom yup + R=I (均匀)", ras_cam_yup, glb_center_geom_yup, common_frames_uniform, False),
    ("G: RAS yup + GLB geom yup + R=I (直接)", ras_cam_yup, glb_center_geom_yup, common_frames_direct, False),
]

for name, ras_cam, glb_center, common_frames, use_umeyama in tests:
    src = np.array([hawor_cam[hi] for _, hi in common_frames])
    dst = np.array([ras_cam[ri] for ri, _ in common_frames])

    if use_umeyama:
        s, R, t = umeyama_align(src, dst)
    else:
        # 固定 R = I, 只计算 s 和 t
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean
        sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
        sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
        s = sigma_dst / sigma_src if sigma_src > 1e-8 else 1.0
        R = np.eye(3)
        t = dst_mean - s * src_mean

    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

    aligned = s * (R @ src.T).T + t
    errors = np.linalg.norm(aligned - dst, axis=1)

    s_inv = 1.0 / s
    R_inv = R.T
    t_inv = -s_inv * (R_inv @ t)

    glb_hawor = s_inv * (R_inv @ glb_center) + t_inv
    glb_sapien = RXWORLD_TO_SAPIEN @ glb_hawor
    dist = np.linalg.norm(hand_mean_sapien - glb_sapien)

    print(f"\n{name}:")
    print(f"  旋转={angle:.1f}°, 尺度={s:.4f}, 对齐误差={errors.mean():.6f}m")
    print(f"  手-GLB距离={dist:.4f}m, GLB_SAPIEN={glb_sapien}")

print("\n" + "=" * 70)
print("关键验证: GLB原始顶点是否与RAS相机在同一坐标系?")
print("=" * 70)

# 如果GLB原始顶点和RAS相机在同一坐标系(y-down),
# 那么GLB中心应该在相机前方(正z方向)
print(f"RAS相机[0] (y-down): {ras_cam_ydown[0]}")
print(f"RAS相机[19] (y-down): {ras_cam_ydown[19]}")
print(f"GLB原始中心 (y-down): {glb_center_geom}")
print(f"GLB z范围: [{verts_geom[:,2].min():.3f}, {verts_geom[:,2].max():.3f}]")
print(f"相机z范围: [{ras_cam_ydown[:,2].min():.3f}, {ras_cam_ydown[:,2].max():.3f}]")

# 相机看+z方向(OpenCV), GLB在z=0.4~1.0, 相机在z=-0.01~0.05
# 所以GLB在相机前方 ✓

# 验证: 从相机位置到GLB中心的方向
cam_to_glb = glb_center_geom - ras_cam_ydown[0]
cam_to_glb_norm = cam_to_glb / np.linalg.norm(cam_to_glb)
print(f"\n相机[0]→GLB中心方向 (y-down): {cam_to_glb_norm}")
print(f"  z分量={cam_to_glb_norm[2]:.3f} (正=前方, 与OpenCV +z一致)")

# RAS相机前向 (OpenCV +z)
ras_forward_ydown = ras_R_c2w[0][:, 2]
print(f"RAS相机[0]前向 (y-down): {ras_forward_ydown}")
print(f"方向与前向的点积: {np.dot(cam_to_glb_norm, ras_forward_ydown):.3f}")

print("\n" + "=" * 70)
print("最终方案: 使用相机朝向固定旋转 + GLB原始顶点")
print("=" * 70)

# 从相机朝向计算R_rel
R_rels = []
for ri, hi in common_frames_uniform:
    ras_R_yup = R_X @ ras_R_c2w[ri]
    hawor_R = hawor_R_c2w[hi]
    R_rel = ras_R_yup @ hawor_R.T
    R_rels.append(R_rel)

R_rels = np.array(R_rels)
R_rel_mean = R_rels.mean(axis=0)
U, _, VH = np.linalg.svd(R_rel_mean)
R_orient = U @ VH
if np.linalg.det(R_orient) < 0:
    U[:, -1] *= -1
    R_orient = U @ VH

angle_orient = np.degrees(np.arccos(np.clip((np.trace(R_orient) - 1) / 2, -1, 1)))
print(f"相机朝向旋转 R_orient: 角度={angle_orient:.1f}°")
print(f"R_orient =\n{R_orient}")

# 用R_orient做对齐 (RAS y-up + GLB geom yup)
src = np.array([hawor_cam[hi] for _, hi in common_frames_uniform])
dst = np.array([ras_cam_yup[ri] for ri, _ in common_frames_uniform])

src_rotated = (R_orient @ src.T).T
src_mean = src_rotated.mean(axis=0)
dst_mean = dst.mean(axis=0)
src_centered = src_rotated - src_mean
dst_centered = dst - dst_mean
sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
s_orient = sigma_dst / sigma_src if sigma_src > 1e-8 else 1.0
t_orient = dst_mean - s_orient * src_mean

aligned = s_orient * (R_orient @ src.T).T + t_orient
errors = np.linalg.norm(aligned - dst, axis=1)

s_inv = 1.0 / s_orient
R_inv = R_orient.T
t_inv = -s_inv * (R_inv @ t_orient)

glb_hawor = s_inv * (R_inv @ glb_center_geom_yup) + t_inv
glb_sapien = RXWORLD_TO_SAPIEN @ glb_hawor
dist = np.linalg.norm(hand_mean_sapien - glb_sapien)

print(f"\n使用 R_orient + GLB原始y-up:")
print(f"  尺度={s_orient:.4f}, 对齐误差={errors.mean():.6f}m")
print(f"  手-GLB距离={dist:.4f}m, GLB_SAPIEN={glb_sapien}")

# 也对直接对应做一次
R_rels_direct = []
for ri, hi in common_frames_direct:
    ras_R_yup = R_X @ ras_R_c2w[ri]
    hawor_R = hawor_R_c2w[hi]
    R_rel = ras_R_yup @ hawor_R.T
    R_rels_direct.append(R_rel)

R_rels_direct = np.array(R_rels_direct)
R_rel_mean_d = R_rels_direct.mean(axis=0)
U, _, VH = np.linalg.svd(R_rel_mean_d)
R_orient_d = U @ VH
if np.linalg.det(R_orient_d) < 0:
    U[:, -1] *= -1
    R_orient_d = U @ VH

src_d = np.array([hawor_cam[hi] for _, hi in common_frames_direct])
dst_d = np.array([ras_cam_yup[ri] for ri, _ in common_frames_direct])

src_rotated_d = (R_orient_d @ src_d.T).T
src_mean_d = src_rotated_d.mean(axis=0)
dst_mean_d = dst_d.mean(axis=0)
src_centered_d = src_rotated_d - src_mean_d
dst_centered_d = dst_d - dst_mean_d
sigma_src_d = np.sqrt(np.mean(np.sum(src_centered_d ** 2, axis=1)))
sigma_dst_d = np.sqrt(np.mean(np.sum(dst_centered_d ** 2, axis=1)))
s_orient_d = sigma_dst_d / sigma_src_d if sigma_src_d > 1e-8 else 1.0
t_orient_d = dst_mean_d - s_orient_d * src_mean_d

s_inv_d = 1.0 / s_orient_d
R_inv_d = R_orient_d.T
t_inv_d = -s_inv_d * (R_inv_d @ t_orient_d)

glb_hawor_d = s_inv_d * (R_inv_d @ glb_center_geom_yup) + t_inv_d
glb_sapien_d = RXWORLD_TO_SAPIEN @ glb_hawor_d
dist_d = np.linalg.norm(hand_mean_sapien - glb_sapien_d)

angle_orient_d = np.degrees(np.arccos(np.clip((np.trace(R_orient_d) - 1) / 2, -1, 1)))
print(f"\n使用 R_orient(直接对应) + GLB原始y-up:")
print(f"  旋转={angle_orient_d:.1f}°, 尺度={s_orient_d:.4f}")
print(f"  手-GLB距离={dist_d:.4f}m, GLB_SAPIEN={glb_sapien_d}")

print("\n完成!")
