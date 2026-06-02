#!/usr/bin/env python3
"""
逐步测试 v2: 用已知旋转替代Umeyama旋转

核心思路:
  Umeyama给出的旋转角度178°/138°都是错的 (应该接近0°或已知坐标轴旋转)
  原因: RAS相机和GLB顶点不在同一坐标系, 导致Umeyama拟合出错误旋转

正确做法:
  1. 先确定GLB顶点和相机分别在什么坐标系
  2. 用已知旋转统一坐标系
  3. 只用Umeyama算scale和translation

坐标系分析:
  RAS extrinsics: w2c格式, R_w2c[0]≈I → 世界=OpenCV相机坐标系 (y-down, z-forward)
  RAS GLB geometry.items(): 经过 apply_transform [[1,0,0],[0,0,1],[0,-1,0],[0,0,0,1]]
    这个变换把 y-down,z-forward → z-up,y-forward
    所以 geometry.items() 是 z-up
  RAS GLB dump(): 在geometry基础上再乘场景图变换 [[1,0,0],[0,0,1],[0,-1,0],[0,0,0,1]]
    z-up → y-up
    所以 dump() 是 y-up

  HaWoR t_c2w/R_c2w: OpenGL y-up (已含R_x翻转)
  HaWoR pred_trans: 与t_c2w同坐标系 (render world, y-up)

关键: RAS相机(y-down) 和 GLB geometry.items()(z-up) 不在同一坐标系!
  需要先把RAS相机转到和GLB同一坐标系, 再做Umeyama
"""

import numpy as np
import trimesh
from glob import glob
import os

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


def umeyama_align(src_pts, dst_pts, force_R=None):
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

    if force_R is not None:
        R = force_R
    else:
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
    return center


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print("1. 加载数据")
print("=" * 70)

ras_cam_ydown, ras_R_c2w = load_ras_cameras(RAS_DIR)

hawor_data = dict(np.load(HAWOR_NPZ, allow_pickle=True))
hawor_cam = hawor_data['t_c2w']
pred_trans = hawor_data['pred_trans']
pred_valid = hawor_data['pred_valid']

n_ras = len(ras_cam_ydown)
n_hawor = len(hawor_cam)
common_frames = find_frame_correspondence(n_ras, n_hawor)

hand_idx = 1
v = pred_valid[hand_idx]
hand_mean_render = pred_trans[hand_idx, v].mean(axis=0)
hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render

# RAS相机的多种坐标系表示
ras_cam_yup = (R_X @ ras_cam_ydown.T).T
# y-down → z-up: [x,y,z] → [x,z,-y]
R_ydown_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
ras_cam_zup = (R_ydown_to_zup @ ras_cam_ydown.T).T

print(f"RAS相机[0]:")
print(f"  y-down: {ras_cam_ydown[0]}")
print(f"  y-up:   {ras_cam_yup[0]}")
print(f"  z-up:   {ras_cam_zup[0]}")
print(f"\nHaWoR相机[0] (y-up): {hawor_cam[0]}")
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

print(f"\nGLB geometry.items() 中心 (z-up): {center_geom}")
print(f"GLB dump() 中心 (y-up):           {center_dump}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. 坐标系验证
# ============================================================
print("\n" + "=" * 70)
print("2. 坐标系验证: RAS相机和GLB是否在同一坐标系?")
print("=" * 70)

# 如果RAS世界是y-down(OpenCV), GLB geometry.items()是z-up
# 那么把相机也转到z-up, 看看GLB和相机的相对位置是否合理
print("RAS相机(z-up)范围: "
      f"x[{ras_cam_zup[:,0].min():.4f},{ras_cam_zup[:,0].max():.4f}] "
      f"y[{ras_cam_zup[:,1].min():.4f},{ras_cam_zup[:,1].max():.4f}] "
      f"z[{ras_cam_zup[:,2].min():.4f},{ras_cam_zup[:,2].max():.4f}]")
print(f"GLB中心(z-up): {center_geom}")
print(f"GLB-相机[0]偏移(z-up): {center_geom - ras_cam_zup[0]}")
print("  → 如果z-up, GLB在相机上方0.74m, 前方0.15m")
print("  → 对于第一人称视角, 物体应该在前方而不是上方")
print("  → 说明GLB和相机不在同一坐标系!")

# 如果RAS世界是y-down, GLB也在y-down呢?
# 把GLB geometry.items()从z-up转回y-down
R_zup_to_ydown = R_ydown_to_zup.T  # 逆变换
center_geom_ydown = R_zup_to_ydown @ center_geom
print(f"\nGLB中心转回y-down: {center_geom_ydown}")
print(f"GLB-相机[0]偏移(y-down): {center_geom_ydown - ras_cam_ydown[0]}")
print("  → y-down: GLB在相机下方0.74m, 前方0.15m")
print("  → 对于俯视桌面场景, 这是合理的!")

# ============================================================
# 3. 核心测试: 用已知旋转 + Umeyama只算s和t
# ============================================================
print("\n" + "=" * 70)
print("3. 核心测试: 已知旋转 + Umeyama(s,t)")
print("=" * 70)

src_pts = np.array([hawor_cam[hi] for _, hi in common_frames])

results = []

# ---- Test A: RAS相机y-down + GLB geom(z-up) → 先统一到y-up ----
# RAS相机: y-down → y-up (乘R_X)
# GLB geom: z-up → y-up (乘R_AXIS)
# 两者都是y-up, 然后Umeyama
print("\n--- Test A: RAS相机y-up + GLB geom→y-up + Umeyama ---")
dst_A = ras_cam_yup
s_A, R_A, t_A = umeyama_align(src_pts, dst_A)
angle_A = np.degrees(np.arccos(np.clip((np.trace(R_A) - 1) / 2, -1, 1)))
print(f"  Umeyama自由旋转: s={s_A:.4f}, R角度={angle_A:.2f}°")

# 用R=I强制
s_A2, R_A2, t_A2 = umeyama_align(src_pts, dst_A, force_R=np.eye(3))
print(f"  强制R=I: s={s_A2:.4f}, t={t_A2}")

# GLB geom(z-up) → y-up → Umeyama逆 → HaWoR(y-up) → SAPIEN
glb_yup_A = R_AXIS @ center_geom
s_inv_A2 = 1.0 / s_A2
t_inv_A2 = -s_inv_A2 * t_A2
glb_hawor_A2 = s_inv_A2 * glb_yup_A + t_inv_A2
glb_sapien_A2 = RXWORLD_TO_SAPIEN @ glb_hawor_A2
dist_A2 = np.linalg.norm(hand_mean_sapien - glb_sapien_A2)
print(f"  GLB中心(SAPIEN): {glb_sapien_A2}, 手-GLB距离: {dist_A2:.4f}m")
results.append(("A2: RAS yup + geom→yup, R=I", dist_A2))

# ---- Test B: RAS相机z-up + GLB geom(z-up) → 同一坐标系! ----
# 两者都是z-up, 然后转y-up对齐HaWoR
print("\n--- Test B: RAS相机z-up + GLB geom(z-up) + Umeyama ---")
# 先把两者都转y-up
ras_cam_yup_from_zup = (R_AXIS @ ras_cam_zup.T).T
glb_yup_from_zup = R_AXIS @ center_geom

# 验证: R_AXIS @ z-up 应该等于 R_X @ y-down?
print(f"  R_AXIS @ zup相机[0]: {ras_cam_yup_from_zup[0]}")
print(f"  R_X @ ydown相机[0]:  {ras_cam_yup[0]}")
print(f"  两者是否一致: {np.allclose(ras_cam_yup_from_zup, ras_cam_yup)}")

# ---- Test C: 关键测试 — RAS相机和GLB都在y-down ----
# RAS相机: y-down (原始)
# GLB geom: z-up → y-down (乘R_zup_to_ydown)
# 两者都在y-down, Umeyama对齐到HaWoR y-up
print("\n--- Test C: RAS相机y-down + GLB geom→y-down + Umeyama ---")
dst_C = ras_cam_ydown  # RAS相机y-down
glb_center_ydown = R_zup_to_ydown @ center_geom  # GLB也y-down

s_C, R_C, t_C = umeyama_align(src_pts, dst_C)
angle_C = np.degrees(np.arccos(np.clip((np.trace(R_C) - 1) / 2, -1, 1)))
print(f"  Umeyama自由旋转: s={s_C:.4f}, R角度={angle_C:.2f}°")

# 强制R=I
s_C2, R_C2, t_C2 = umeyama_align(src_pts, dst_C, force_R=np.eye(3))
print(f"  强制R=I: s={s_C2:.4f}, t={t_C2}")

# GLB(y-down) → Umeyama逆 → HaWoR(y-up) → SAPIEN
# 但GLB是y-down, HaWoR是y-up, 需要转换
# 逆变换: p_hawor_yup = s_inv * p_ras_ydown + t_inv  (但这混了y-down和y-up!)
# 正确: 先把GLB从y-down转到y-up, 再用y-up的Umeyama参数
# 或者: 把Umeyama参数也转到y-down → y-up

# 实际上, Umeyama对齐的是 HaWoR(y-up) → RAS(y-down)
# 逆变换: p_hawor_yup = s_inv * R^T @ p_ras_ydown + t_inv
# 但这把RAS y-down坐标映射到HaWoR y-up坐标, 中间有坐标系翻转

# 更好的方式: 把RAS相机也转y-up, 重新做Umeyama
# 这就是01_align_scene.py的方式, 但R角度=138°还是太大

# ---- Test D: 把GLB也转到y-down, 和相机在同一坐标系, 然后整体转y-up ----
print("\n--- Test D: GLB和相机都在y-down, 整体转y-up后Umeyama ---")
# GLB geom z-up → y-down
glb_verts_ydown = (R_zup_to_ydown @ verts_geom.T).T
center_ydown = glb_verts_ydown.mean(axis=0)

# 现在GLB和相机都在y-down, 验证相对位置
print(f"  GLB中心(y-down): {center_ydown}")
print(f"  相机[0](y-down): {ras_cam_ydown[0]}")
print(f"  GLB-相机偏移(y-down): {center_ydown - ras_cam_ydown[0]}")
print("  → y-down: GLB在下方0.74m, 前方0.15m (合理: 俯视桌面)")

# 整体转y-up: 乘R_X
glb_verts_yup_from_ydown = (R_X @ glb_verts_ydown.T).T
center_yup_from_ydown = glb_verts_yup_from_ydown.mean(axis=0)
ras_cam_yup_from_ydown = ras_cam_yup  # 已经算过了

print(f"  GLB中心(y-up, 从y-down转): {center_yup_from_ydown}")
print(f"  相机[0](y-up): {ras_cam_yup_from_ydown[0]}")
print(f"  GLB-相机偏移(y-up): {center_yup_from_ydown - ras_cam_yup_from_ydown[0]}")

# 现在GLB和相机都在y-up, 和HaWoR同坐标系, Umeyama
s_D, R_D, t_D = umeyama_align(src_pts, ras_cam_yup_from_ydown)
angle_D = np.degrees(np.arccos(np.clip((np.trace(R_D) - 1) / 2, -1, 1)))
print(f"\n  Umeyama自由旋转: s={s_D:.4f}, R角度={angle_D:.2f}°")

# 强制R=I
s_D2, R_D2, t_D2 = umeyama_align(src_pts, ras_cam_yup_from_ydown, force_R=np.eye(3))
print(f"  强制R=I: s={s_D2:.4f}, t={t_D2}")

# 用强制R=I的结果变换GLB
s_inv_D2 = 1.0 / s_D2
t_inv_D2 = -s_inv_D2 * t_D2


def transform_D2(verts):
    verts_ydown = (R_zup_to_ydown @ verts.T).T
    verts_yup = (R_X @ verts_ydown.T).T
    verts_hawor = s_inv_D2 * verts_yup + t_inv_D2
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_hawor_D2 = s_inv_D2 * center_yup_from_ydown + t_inv_D2
glb_sapien_D2 = RXWORLD_TO_SAPIEN @ glb_hawor_D2
dist_D2 = np.linalg.norm(hand_mean_sapien - glb_sapien_D2)
print(f"  GLB中心(HaWoR): {glb_hawor_D2}")
print(f"  GLB中心(SAPIEN): {glb_sapien_D2}")
print(f"  手-GLB距离: {dist_D2:.4f}m")
results.append(("D2: GLB ydown→yup, R=I", dist_D2))

save_test_glb(meshes_geom, transform_D2,
              os.path.join(OUTPUT_DIR, "testD2_glb_ydown_yup_R_I.glb"), "TestD2")

# 用Umeyama自由旋转的结果
s_inv_D = 1.0 / s_D
R_inv_D = R_D.T
t_inv_D = -s_inv_D * (R_inv_D @ t_D)


def transform_D(verts):
    verts_ydown = (R_zup_to_ydown @ verts.T).T
    verts_yup = (R_X @ verts_ydown.T).T
    verts_hawor = s_inv_D * (R_inv_D @ verts_yup.T).T + t_inv_D
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_hawor_D = s_inv_D * (R_inv_D @ center_yup_from_ydown) + t_inv_D
glb_sapien_D = RXWORLD_TO_SAPIEN @ glb_hawor_D
dist_D = np.linalg.norm(hand_mean_sapien - glb_sapien_D)
print(f"\n  Umeyama自由R: GLB中心(SAPIEN): {glb_sapien_D}, 手-GLB距离: {dist_D:.4f}m")
results.append(("D: GLB ydown→yup, Umeyama R", dist_D))

save_test_glb(meshes_geom, transform_D,
              os.path.join(OUTPUT_DIR, "testD_glb_ydown_yup_umeyama.glb"), "TestD")

# ---- Test E: dump()顶点(y-up) + RAS相机y-up + 强制R=I ----
print("\n--- Test E: dump(y-up) + RAS相机y-up + 强制R=I ---")
s_E, R_E, t_E = umeyama_align(src_pts, ras_cam_yup, force_R=np.eye(3))
print(f"  强制R=I: s={s_E:.4f}, t={t_E}")

s_inv_E = 1.0 / s_E
t_inv_E = -s_inv_E * t_E


def transform_E(verts):
    verts_hawor = s_inv_E * verts + t_inv_E
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_hawor_E = s_inv_E * center_dump + t_inv_E
glb_sapien_E = RXWORLD_TO_SAPIEN @ glb_hawor_E
dist_E = np.linalg.norm(hand_mean_sapien - glb_sapien_E)
print(f"  GLB中心(SAPIEN): {glb_sapien_E}, 手-GLB距离: {dist_E:.4f}m")
results.append(("E: dump yup + RAS yup, R=I", dist_E))

save_test_glb(meshes_dump, transform_E,
              os.path.join(OUTPUT_DIR, "testE_dump_yup_R_I.glb"), "TestE")

# ---- Test F: 直接用yingshe.py的方式, 但强制R=I ----
print("\n--- Test F: yingshe方式 (RAS y-down + geom) + 强制R=I ---")
s_F, R_F, t_F = umeyama_align(src_pts, ras_cam_ydown, force_R=np.eye(3))
print(f"  强制R=I: s={s_F:.4f}, t={t_F}")

s_inv_F = 1.0 / s_F
t_inv_F = -s_inv_F * t_F


def transform_F(verts):
    verts_hawor = s_inv_F * verts + t_inv_F
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_hawor_F = s_inv_F * center_geom + t_inv_F
glb_sapien_F = RXWORLD_TO_SAPIEN @ glb_hawor_F
dist_F = np.linalg.norm(hand_mean_sapien - glb_sapien_F)
print(f"  GLB中心(SAPIEN): {glb_sapien_F}, 手-GLB距离: {dist_F:.4f}m")
results.append(("F: yingshe R=I (geom ydown)", dist_F))

save_test_glb(meshes_geom, transform_F,
              os.path.join(OUTPUT_DIR, "testF_yingshe_R_I.glb"), "TestF")

# ---- Test G: yingshe方式 + 强制R=R_X (y-down→y-up) ----
print("\n--- Test G: yingshe方式 + 强制R=R_X ---")
s_G, R_G, t_G = umeyama_align(src_pts, ras_cam_ydown, force_R=R_X)
print(f"  强制R=R_X: s={s_G:.4f}, t={t_G}")

s_inv_G = 1.0 / s_G
R_inv_G = R_X.T  # R_X^T = R_X (对称矩阵)
t_inv_G = -s_inv_G * (R_inv_G @ t_G)


def transform_G(verts):
    verts_hawor = s_inv_G * (R_inv_G @ verts.T).T + t_inv_G
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

glb_hawor_G = s_inv_G * (R_inv_G @ center_geom) + t_inv_G
glb_sapien_G = RXWORLD_TO_SAPIEN @ glb_hawor_G
dist_G = np.linalg.norm(hand_mean_sapien - glb_sapien_G)
print(f"  GLB中心(SAPIEN): {glb_sapien_G}, 手-GLB距离: {dist_G:.4f}m")
results.append(("G: yingshe R=R_X", dist_G))

save_test_glb(meshes_geom, transform_G,
              os.path.join(OUTPUT_DIR, "testG_yingshe_R_RX.glb"), "TestG")

# ---- Test H: 关键 — 用相机相对偏移, 不依赖Umeyama ----
print("\n--- Test H: 相机相对偏移法 (最直接) ---")
# 在RAS y-down坐标系中:
#   相机[0] ≈ [0,0,0]
#   GLB中心 ≈ [-0.035, -0.740, 0.148] (y-down, 从z-up转回)
#   GLB相对相机偏移 ≈ [-0.035, -0.740, 0.148]
#
# 在HaWoR y-up坐标系中:
#   相机[0] ≈ [0.004, 0.004, 0.001]
#   手均值 ≈ [-0.014, 0.009, 0.012]
#
# 如果两个坐标系描述同一个物理世界:
#   RAS中GLB相对相机的偏移, 经过坐标系转换和缩放后,
#   应该等于HaWoR中GLB相对相机的偏移

# RAS偏移(y-down) → y-up: R_X @ offset_ydown
offset_ras_ydown = center_geom_ydown - ras_cam_ydown[0]
offset_ras_yup = R_X @ offset_ras_ydown
print(f"  RAS GLB-相机偏移(y-down): {offset_ras_ydown}")
print(f"  RAS GLB-相机偏移(y-up):   {offset_ras_yup}")

# 缩放: 用Umeyama的s
offset_hawor_yup = offset_ras_yup / s_A
glb_hawor_H = hawor_cam[0] + offset_hawor_yup
glb_sapien_H = RXWORLD_TO_SAPIEN @ glb_hawor_H
dist_H = np.linalg.norm(hand_mean_sapien - glb_sapien_H)
print(f"  缩放后偏移(HaWoR y-up): {offset_hawor_yup}")
print(f"  GLB中心(HaWoR): {glb_hawor_H}")
print(f"  GLB中心(SAPIEN): {glb_sapien_H}")
print(f"  手-GLB距离: {dist_H:.4f}m")
results.append(("H: 相机偏移法 (ydown→yup, /s)", dist_H))


def transform_H(verts):
    verts_ydown = (R_zup_to_ydown @ verts.T).T
    offset_from_cam = verts_ydown - ras_cam_ydown[0]
    offset_yup = (R_X @ offset_from_cam.T).T
    verts_hawor = hawor_cam[0] + offset_yup / s_A
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

save_test_glb(meshes_geom, transform_H,
              os.path.join(OUTPUT_DIR, "testH_cam_offset_method.glb"), "TestH")

# ---- Test I: 同上但不用缩放 (s=1) ----
print("\n--- Test I: 相机偏移法, s=1 (无缩放) ---")
offset_hawor_I = offset_ras_yup  # 不除以s
glb_hawor_I = hawor_cam[0] + offset_hawor_I
glb_sapien_I = RXWORLD_TO_SAPIEN @ glb_hawor_I
dist_I = np.linalg.norm(hand_mean_sapien - glb_sapien_I)
print(f"  GLB中心(SAPIEN): {glb_sapien_I}")
print(f"  手-GLB距离: {dist_I:.4f}m")
results.append(("I: 相机偏移法 s=1", dist_I))


def transform_I(verts):
    verts_ydown = (R_zup_to_ydown @ verts.T).T
    offset_from_cam = verts_ydown - ras_cam_ydown[0]
    offset_yup = (R_X @ offset_from_cam.T).T
    verts_hawor = hawor_cam[0] + offset_yup
    verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
    return verts_sapien

save_test_glb(meshes_geom, transform_I,
              os.path.join(OUTPUT_DIR, "testI_cam_offset_s1.glb"), "TestI")

# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 70)
print("汇总: 手-GLB距离对比 (SAPIEN坐标系)")
print("=" * 70)
results.sort(key=lambda x: x[1])
for name, dist in results:
    marker = " ← " if dist == results[0][1] else ""
    print(f"  {name}: {dist:.4f}m{marker}")

print(f"\n所有测试GLB保存在: {OUTPUT_DIR}")
print(f"\n推荐查看命令 (选距离最小的test):")
print(f"  cd /home/an/robot_world_ws/src/dex-retargeting/example/combination")
print(f"  conda run -n dex python 02_render_scene.py --mode hand_only --viewer \\")
print(f"      --hawor-dir /home/an/data/hawor/7 \\")
print(f"      --ras-dir /home/an/data/ras/my_7mp4_result \\")
print(f"      --sapien-glb {OUTPUT_DIR}/testX_xxx.glb")
