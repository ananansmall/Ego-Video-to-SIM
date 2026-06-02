#!/usr/bin/env python3
"""
逐步测试脚本: 先测平移, 再测旋转
目标: 把GLB放到机械臂/手旁边, 看看哪种方式最贴合

测试策略:
  Test 1: 纯平移 — 把GLB中心直接移到手腕位置 (无旋转, 无缩放)
  Test 2: yingshe.py方式 — RAS相机(y-down) + geometry.items() + Umeyama
  Test 3: 01_align方式 — RAS相机(y-up) + dump() + Umeyama
  Test 4: 只用平移+缩放 — Umeyama的s和t, 但R=I
  Test 5: yingshe.py方式但R=I — 只用s和t
"""

import numpy as np
import trimesh
from glob import glob
import os
import sys

R_X = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_X

RAS_DIR = "/home/an/data/ras/my_7mp4_result"
HAWOR_NPZ = "/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz"
OUTPUT_DIR = "/home/an/robot_world_ws/src/dex-retargeting/example/combination/output/alignment_test"


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


def save_test_glb(meshes, transform_fn, output_path, test_name):
    """对GLB的每个mesh应用transform_fn后保存"""
    sapien_scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        verts = mesh.vertices.copy()
        verts_new = transform_fn(verts)
        mesh_new = mesh.copy()
        mesh_new.vertices = verts_new
        sapien_scene.add_geometry(mesh_new, geom_name=f"object_{i}")
    sapien_scene.export(output_path)

    tm_check = trimesh.load(output_path, force='scene')
    all_v = []
    for name, g in tm_check.geometry.items():
        all_v.append(g.vertices)
    if all_v:
        all_v = np.vstack(all_v)
        center = all_v.mean(axis=0)
        print(f"  [{test_name}] GLB中心(SAPIEN): {center}")
        print(f"  [{test_name}] GLB范围: min={all_v.min(axis=0)}, max={all_v.max(axis=0)}")
    return center


# ============================================================
# 1. 加载所有数据
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

# 手部位置
hand_idx = 1
v = pred_valid[hand_idx]
hand_mean_render = pred_trans[hand_idx, v].mean(axis=0)
hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render

print(f"RAS相机: {n_ras} 帧")
print(f"  cam_ydown[0] = {ras_cam_ydown[0]}")
print(f"  cam_yup[0]   = {ras_cam_yup[0]}")
print(f"HaWoR相机: {n_hawor} 帧")
print(f"  cam[0] = {hawor_cam[0]}")
print(f"右手均值 (render y-up): {hand_mean_render}")
print(f"右手均值 (SAPIEN z-up): {hand_mean_sapien}")

# GLB
glb_path = os.path.join(RAS_DIR, 'final_scene.glb')
tm_scene = trimesh.load(glb_path, force='scene')

meshes_geom = list(tm_scene.geometry.values())
verts_geom = np.vstack([m.vertices for m in meshes_geom])
center_geom = verts_geom.mean(axis=0)

meshes_dump = tm_scene.dump()
verts_dump = np.vstack([m.vertices for m in meshes_dump])
center_dump = verts_dump.mean(axis=0)

print(f"\nGLB geometry.items() 中心: {center_geom}")
print(f"GLB dump() 中心:         {center_dump}")
print(f"两者差异: {np.linalg.norm(center_geom - center_dump):.4f}")

# 场景图变换矩阵
z_up_to_y_up = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
print(f"\nz-up → y-up 变换验证: {z_up_to_y_up[:3,:3] @ center_geom} ≈ {center_dump}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. Umeyama对齐参数 (两种方式)
# ============================================================
print("\n" + "=" * 70)
print("2. Umeyama对齐参数")
print("=" * 70)

# 方式A: yingshe.py (RAS y-down + geometry.items())
src_A = np.array([hawor_cam[hi] for _, hi in common_frames])
dst_A = np.array([ras_cam_ydown[ri] for ri, _ in common_frames])
s_A, R_A, t_A = umeyama_align(src_A, dst_A)
angle_A = np.degrees(np.arccos(np.clip((np.trace(R_A) - 1) / 2, -1, 1)))
print(f"\n方式A (yingshe: RAS y-down + geom):")
print(f"  s={s_A:.4f}, R角度={angle_A:.2f}°, t={t_A}")

# 方式B: 01_align (RAS y-up + dump())
src_B = np.array([hawor_cam[hi] for _, hi in common_frames])
dst_B = np.array([ras_cam_yup[ri] for ri, _ in common_frames])
s_B, R_B, t_B = umeyama_align(src_B, dst_B)
angle_B = np.degrees(np.arccos(np.clip((np.trace(R_B) - 1) / 2, -1, 1)))
print(f"\n方式B (01_align: RAS y-up + dump):")
print(f"  s={s_B:.4f}, R角度={angle_B:.2f}°, t={t_B}")

# ============================================================
# Test 1: 纯平移 — GLB中心直接移到手腕位置
# ============================================================
print("\n" + "=" * 70)
print("Test 1: 纯平移 — GLB中心直接移到手腕位置 (无旋转, 无缩放)")
print("=" * 70)

# 用dump()顶点(y-up), 直接变换到SAPIEN
# 步骤: dump(y-up) → RXWORLD_TO_SAPIEN → SAPIEN(z-up)
# 然后平移使GLB中心 = 手腕位置

glb_dump_sapien_center = RXWORLD_TO_SAPIEN @ center_dump
offset_1 = hand_mean_sapien - glb_dump_sapien_center
print(f"  GLB dump中心直接转SAPIEN: {glb_dump_sapien_center}")
print(f"  手腕SAPIEN: {hand_mean_sapien}")
print(f"  需要平移: {offset_1}")
print(f"  平移距离: {np.linalg.norm(offset_1):.4f}m")


def transform_test1(verts):
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts.T).T + offset_1
    return verts_sapien

save_test_glb(meshes_dump, transform_test1,
              os.path.join(OUTPUT_DIR, "test1_pure_translation.glb"), "Test1")

# 也用geometry.items()试一下
glb_geom_sapien_center = RXWORLD_TO_SAPIEN @ center_geom
offset_1b = hand_mean_sapien - glb_geom_sapien_center
print(f"\n  GLB geom中心直接转SAPIEN: {glb_geom_sapien_center}")
print(f"  需要平移: {offset_1b}")


def transform_test1b(verts):
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts.T).T + offset_1b
    return verts_sapien

save_test_glb(meshes_geom, transform_test1b,
              os.path.join(OUTPUT_DIR, "test1b_pure_translation_geom.glb"), "Test1b")

# ============================================================
# Test 2: yingshe.py方式完整变换
# ============================================================
print("\n" + "=" * 70)
print("Test 2: yingshe.py方式 (RAS y-down + geometry.items() + Umeyama)")
print("=" * 70)

s_inv_A = 1.0 / s_A
R_inv_A = R_A.T
t_inv_A = -s_inv_A * (R_inv_A @ t_A)

print(f"  正变换: p_ras = {s_A:.4f} * R @ p_hawor + t")
print(f"  逆变换: p_hawor = {s_inv_A:.4f} * R^T @ p_ras + t_inv")
print(f"  R角度: {angle_A:.2f}°")


def transform_test2(verts):
    verts_hawor = s_inv_A * (R_inv_A @ verts.T).T + t_inv_A
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_center_hawor_A = s_inv_A * (R_inv_A @ center_geom) + t_inv_A
glb_center_sapien_A = RXWORLD_TO_SAPIEN @ glb_center_hawor_A
dist_A = np.linalg.norm(hand_mean_sapien - glb_center_sapien_A)
print(f"  GLB中心(HaWoR): {glb_center_hawor_A}")
print(f"  GLB中心(SAPIEN): {glb_center_sapien_A}")
print(f"  手-GLB距离: {dist_A:.4f}m")

save_test_glb(meshes_geom, transform_test2,
              os.path.join(OUTPUT_DIR, "test2_yingshe_full.glb"), "Test2")

# ============================================================
# Test 3: 01_align方式完整变换
# ============================================================
print("\n" + "=" * 70)
print("Test 3: 01_align方式 (RAS y-up + dump() + Umeyama)")
print("=" * 70)

s_inv_B = 1.0 / s_B
R_inv_B = R_B.T
t_inv_B = -s_inv_B * (R_inv_B @ t_B)

print(f"  正变换: p_ras = {s_B:.4f} * R @ p_hawor + t")
print(f"  逆变换: p_hawor = {s_inv_B:.4f} * R^T @ p_ras + t_inv")
print(f"  R角度: {angle_B:.2f}°")


def transform_test3(verts):
    verts_hawor = s_inv_B * (R_inv_B @ verts.T).T + t_inv_B
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_center_hawor_B = s_inv_B * (R_inv_B @ center_dump) + t_inv_B
glb_center_sapien_B = RXWORLD_TO_SAPIEN @ glb_center_hawor_B
dist_B = np.linalg.norm(hand_mean_sapien - glb_center_sapien_B)
print(f"  GLB中心(HaWoR): {glb_center_hawor_B}")
print(f"  GLB中心(SAPIEN): {glb_center_sapien_B}")
print(f"  手-GLB距离: {dist_B:.4f}m")

save_test_glb(meshes_dump, transform_test3,
              os.path.join(OUTPUT_DIR, "test3_01align_full.glb"), "Test3")

# ============================================================
# Test 4: 只用平移+缩放, R=I (yingshe参数)
# ============================================================
print("\n" + "=" * 70)
print("Test 4: 只用平移+缩放, R=I (yingshe的s和t, 不用R)")
print("=" * 70)

# 用yingshe的s和t, 但R=I
# p_hawor = (1/s) * p_ras + t_inv_noR
# t_inv_noR = -(1/s) * t  (因为R=I, R_inv=I)
t_inv_noR = -s_inv_A * t_A

print(f"  s={s_A:.4f}, t={t_A}")
print(f"  t_inv(无旋转) = {t_inv_noR}")


def transform_test4(verts):
    verts_hawor = s_inv_A * verts + t_inv_noR
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_center_hawor_4 = s_inv_A * center_geom + t_inv_noR
glb_center_sapien_4 = RXWORLD_TO_SAPIEN @ glb_center_hawor_4
dist_4 = np.linalg.norm(hand_mean_sapien - glb_center_sapien_4)
print(f"  GLB中心(HaWoR): {glb_center_hawor_4}")
print(f"  GLB中心(SAPIEN): {glb_center_sapien_4}")
print(f"  手-GLB距离: {dist_4:.4f}m")

save_test_glb(meshes_geom, transform_test4,
              os.path.join(OUTPUT_DIR, "test4_scale_translate_noR.glb"), "Test4")

# ============================================================
# Test 5: 只用平移+缩放, R=I (01_align参数, dump顶点)
# ============================================================
print("\n" + "=" * 70)
print("Test 5: 只用平移+缩放, R=I (01_align的s和t, dump顶点)")
print("=" * 70)

t_inv_noR_B = -s_inv_B * t_B

print(f"  s={s_B:.4f}, t={t_B}")
print(f"  t_inv(无旋转) = {t_inv_noR_B}")


def transform_test5(verts):
    verts_hawor = s_inv_B * verts + t_inv_noR_B
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_center_hawor_5 = s_inv_B * center_dump + t_inv_noR_B
glb_center_sapien_5 = RXWORLD_TO_SAPIEN @ glb_center_hawor_5
dist_5 = np.linalg.norm(hand_mean_sapien - glb_center_sapien_5)
print(f"  GLB中心(HaWoR): {glb_center_hawor_5}")
print(f"  GLB中心(SAPIEN): {glb_center_sapien_5}")
print(f"  手-GLB距离: {dist_5:.4f}m")

save_test_glb(meshes_dump, transform_test5,
              os.path.join(OUTPUT_DIR, "test5_scale_translate_noR_dump.glb"), "Test5")

# ============================================================
# Test 6: 关键测试 — 相机原点对齐
#   RAS相机原点 ≈ [0,0,0], HaWoR相机原点 ≈ [0,0,0]
#   GLB在RAS中相对于相机的偏移 = center_geom - cam_ras[0]
#   同样的偏移应该存在于HaWoR中
#   所以: glb_hawor = hawor_cam[0] + (center_geom - cam_ras[0]) / s
# ============================================================
print("\n" + "=" * 70)
print("Test 6: 相机原点对齐 (直接用相对偏移)")
print("=" * 70)

# RAS中GLB相对于相机的偏移 (geometry.items(), 原始坐标系)
offset_ras_geom = center_geom - ras_cam_ydown[0]
# RAS中GLB相对于相机的偏移 (dump(), y-up)
offset_ras_dump = center_dump - ras_cam_yup[0]

print(f"  RAS相机[0] (y-down): {ras_cam_ydown[0]}")
print(f"  RAS相机[0] (y-up):   {ras_cam_yup[0]}")
print(f"  HaWoR相机[0]:         {hawor_cam[0]}")
print(f"  GLB相对相机偏移 (geom, y-down): {offset_ras_geom}")
print(f"  GLB相对相机偏移 (dump, y-up):   {offset_ras_dump}")

# 假设: 两个坐标系方向一致, 只是尺度不同
# GLB在HaWoR中的位置 = HaWoR相机[0] + offset / scale
# 但我们不知道scale, 先用Umeyama的scale

# 用geometry.items() + y-down偏移
glb_hawor_6a = hawor_cam[0] + offset_ras_geom / s_A
glb_sapien_6a = RXWORLD_TO_SAPIEN @ glb_hawor_6a
dist_6a = np.linalg.norm(hand_mean_sapien - glb_sapien_6a)
print(f"\n  Test6a: geom + y-down偏移 / s_A")
print(f"    GLB(HaWoR): {glb_hawor_6a}")
print(f"    GLB(SAPIEN): {glb_sapien_6a}")
print(f"    手-GLB距离: {dist_6a:.4f}m")

# 用dump() + y-up偏移
glb_hawor_6b = hawor_cam[0] + offset_ras_dump / s_B
glb_sapien_6b = RXWORLD_TO_SAPIEN @ glb_hawor_6b
dist_6b = np.linalg.norm(hand_mean_sapien - glb_sapien_6b)
print(f"\n  Test6b: dump + y-up偏移 / s_B")
print(f"    GLB(HaWoR): {glb_hawor_6b}")
print(f"    GLB(SAPIEN): {glb_sapien_6b}")
print(f"    手-GLB距离: {dist_6b:.4f}m")


def transform_test6a(verts):
    verts_hawor = hawor_cam[0] + (verts - ras_cam_ydown[0]) / s_A
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

save_test_glb(meshes_geom, transform_test6a,
              os.path.join(OUTPUT_DIR, "test6a_cam_origin_align.glb"), "Test6a")


def transform_test6b(verts):
    verts_hawor = hawor_cam[0] + (verts - ras_cam_yup[0]) / s_B
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

save_test_glb(meshes_dump, transform_test6b,
              os.path.join(OUTPUT_DIR, "test6b_cam_origin_align.glb"), "Test6b")

# ============================================================
# Test 7: 不用Umeyama的scale, 直接用1.0 (假设两系统尺度一致)
# ============================================================
print("\n" + "=" * 70)
print("Test 7: scale=1.0, 只做坐标轴旋转 (y-up → SAPIEN z-up)")
print("=" * 70)

# 如果RAS和HaWoR的尺度本来就一致, 那scale=1
# GLB(dump, y-up) → 直接用RXWORLD_TO_SAPIEN转到SAPIEN
# 然后平移使相机位置对齐

glb_sapien_7 = RXWORLD_TO_SAPIEN @ center_dump
cam_ras_sapien = RXWORLD_TO_SAPIEN @ ras_cam_yup[0]
cam_hawor_sapien = RXWORLD_TO_SAPIEN @ hawor_cam[0]
offset_7 = cam_hawor_sapien - cam_ras_sapien

print(f"  RAS相机(SAPIEN): {cam_ras_sapien}")
print(f"  HaWoR相机(SAPIEN): {cam_hawor_sapien}")
print(f"  相机偏移(SAPIEN): {offset_7}")
print(f"  GLB中心(SAPIEN, 无偏移): {glb_sapien_7}")
print(f"  GLB中心(SAPIEN, 有偏移): {glb_sapien_7 + offset_7}")
dist_7 = np.linalg.norm(hand_mean_sapien - (glb_sapien_7 + offset_7))
print(f"  手-GLB距离: {dist_7:.4f}m")


def transform_test7(verts):
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts.T).T + offset_7
    return verts_sapien

save_test_glb(meshes_dump, transform_test7,
              os.path.join(OUTPUT_DIR, "test7_scale1_cam_offset.glb"), "Test7")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("汇总: 手-GLB距离对比")
print("=" * 70)
results = [
    ("Test1:  纯平移(dump)", np.linalg.norm(hand_mean_sapien - (RXWORLD_TO_SAPIEN @ center_dump + offset_1)),
     "test1_pure_translation.glb"),
    ("Test1b: 纯平移(geom)", np.linalg.norm(hand_mean_sapien - (RXWORLD_TO_SAPIEN @ center_geom + offset_1b)),
     "test1b_pure_translation_geom.glb"),
    ("Test2:  yingshe完整", dist_A,
     "test2_yingshe_full.glb"),
    ("Test3:  01align完整", dist_B,
     "test3_01align_full.glb"),
    ("Test4:  s+t无R(geom)", dist_4,
     "test4_scale_translate_noR.glb"),
    ("Test5:  s+t无R(dump)", dist_5,
     "test5_scale_translate_noR_dump.glb"),
    ("Test6a: 相机原点(geom)", dist_6a,
     "test6a_cam_origin_align.glb"),
    ("Test6b: 相机原点(dump)", dist_6b,
     "test6b_cam_origin_align.glb"),
    ("Test7:  scale=1相机偏移", dist_7,
     "test7_scale1_cam_offset.glb"),
]

results.sort(key=lambda x: x[1])
for name, dist, fname in results:
    marker = " ← 最小" if dist == results[0][1] else ""
    print(f"  {name}: {dist:.4f}m  → {fname}{marker}")

print(f"\n所有测试GLB保存在: {OUTPUT_DIR}")
print(f"\n查看命令:")
print(f"  cd /home/an/robot_world_ws/src/dex-retargeting/example/combination")
print(f"  conda run -n dex python 02_render_scene.py --mode hand_only --viewer \\")
print(f"      --hawor-dir /home/an/data/hawor/7 \\")
print(f"      --ras-dir /home/an/data/ras/my_7mp4_result \\")
print(f"      --sapien-glb {OUTPUT_DIR}/testX_xxx.glb")
