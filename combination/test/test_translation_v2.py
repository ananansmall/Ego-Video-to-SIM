#!/usr/bin/env python3
"""
test_translation_v2.py — 更精确的平移测试

关键发现:
  1. geometry.items() 和 dump() 读取的GLB顶点中心差异1.067m
  2. Umeyama尺度 s≈3.1149 (RAS比HaWoR大约3倍)
  3. 纯平移(R=I, s=1)时GLB中心移到手腕=0m, 但GLB可能太大
  4. 用s=3.1149时距离约0.22-0.26m

本测试:
  A. 先看GLB在SAPIEN中的实际大小 (用s=1 vs s=3.1149)
  B. 逐步测试: 先平移, 再加尺度, 最后加旋转
  C. 每种方式生成GLB, 可在Viewer中查看

用法:
    python test_translation_v2.py \
        --ras_output /home/an/data/ras/my_7mp4_result \
        --hawor_reconstruction /home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz \
        --output_dir ./output/alignment_test_v3
"""

import argparse
import os
import numpy as np
from glob import glob

try:
    import trimesh
except ImportError:
    trimesh = None


R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
R_X = np.diag([1.0, -1.0, -1.0])
RXWORLD_TO_SAPIEN = R_AXIS @ R_X


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


def load_hawor_data(hawor_reconstruction):
    hawor_data = dict(np.load(hawor_reconstruction, allow_pickle=True))
    return (hawor_data['t_c2w'], hawor_data['R_c2w'],
            hawor_data['pred_trans'], hawor_data['pred_valid'])


def find_frame_correspondence(n_ras, n_hawor):
    common_frames = []
    for ras_i in range(n_ras):
        hawor_i = round(ras_i * (n_hawor - 1) / (n_ras - 1)) if n_ras > 1 else 0
        common_frames.append((ras_i, hawor_i))
    return common_frames


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


def transform_and_save_glb(meshes, transform_func, output_path, test_name):
    sapien_scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        vertices = mesh.vertices.copy()
        vertices_new = transform_func(vertices)
        mesh_copy = mesh.copy()
        mesh_copy.vertices = vertices_new
        sapien_scene.add_geometry(mesh_copy, geom_name=f"object_{i}")
    sapien_scene.export(output_path)

    tm_check = trimesh.load(output_path, force='scene')
    all_v = []
    for name, g in tm_check.geometry.items():
        all_v.append(g.vertices)
    if all_v:
        all_v = np.vstack(all_v)
        center = all_v.mean(axis=0)
        extent = all_v.max(axis=0) - all_v.min(axis=0)
        print(f"  {test_name}:")
        print(f"    中心(SAPIEN)={center}")
        print(f"    范围: min={all_v.min(axis=0)}, max={all_v.max(axis=0)}")
        print(f"    尺寸(XYZ): {extent}")
    return center


def main():
    parser = argparse.ArgumentParser(description="精确平移测试")
    parser.add_argument("--ras_output", type=str, required=True)
    parser.add_argument("--hawor_reconstruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/alignment_test_v3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  精确平移测试: 先平移, 再尺度, 最后旋转")
    print("=" * 70)

    ras_cam_ydown, ras_R_c2w = load_ras_cameras(args.ras_output)
    ras_cam_yup = (R_X @ ras_cam_ydown.T).T
    hawor_cam, hawor_R_c2w, pred_trans, pred_valid = load_hawor_data(args.hawor_reconstruction)

    n_ras = len(ras_cam_ydown)
    n_hawor = len(hawor_cam)
    common_frames = find_frame_correspondence(n_ras, n_hawor)

    hand_idx = 1 if pred_valid[1].any() else 0
    valid_frames = np.where(pred_valid[hand_idx])[0]
    hand_mean_render = pred_trans[hand_idx, valid_frames].mean(axis=0)
    hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render

    src_pts = np.array([hawor_cam[hi] for _, hi in common_frames])

    print(f"\n  手腕均值(SAPIEN z-up): {hand_mean_sapien}")
    print(f"  RAS相机(y-down)范围: x[{ras_cam_ydown[:,0].min():.4f},{ras_cam_ydown[:,0].max():.4f}]"
          f" y[{ras_cam_ydown[:,1].min():.4f},{ras_cam_ydown[:,1].max():.4f}]"
          f" z[{ras_cam_ydown[:,2].min():.4f},{ras_cam_ydown[:,2].max():.4f}]")
    print(f"  HaWoR相机(y-up)范围:  x[{hawor_cam[:,0].min():.4f},{hawor_cam[:,0].max():.4f}]"
          f" y[{hawor_cam[:,1].min():.4f},{hawor_cam[:,1].max():.4f}]"
          f" z[{hawor_cam[:,2].min():.4f},{hawor_cam[:,2].max():.4f}]")

    ras_glb_path = os.path.join(args.ras_output, 'final_scene.glb')
    if not os.path.exists(ras_glb_path):
        print(f"\n✗ GLB文件不存在: {ras_glb_path}")
        return

    scene_raw = trimesh.load(ras_glb_path, force='scene')
    meshes_dump = scene_raw.dump()
    verts_dump = np.vstack([m.vertices for m in meshes_dump])
    center_dump = verts_dump.mean(axis=0)

    scene_geom = trimesh.load(ras_glb_path)
    verts_geom = []
    for name, geom in scene_geom.geometry.items():
        verts_geom.append(geom.vertices)
    verts_geom = np.vstack(verts_geom)
    center_geom = verts_geom.mean(axis=0)

    print(f"\n  GLB dump() 中心: {center_dump}  (场景图变换后 y-up)")
    print(f"  GLB geom() 中心: {center_geom}  (原始 z-up)")

    dst_pts_ydown = np.array([ras_cam_ydown[ri] for ri, _ in common_frames])
    dst_pts_yup = np.array([ras_cam_yup[ri] for ri, _ in common_frames])

    s_ydown, R_ydown, t_ydown = umeyama_align(src_pts, dst_pts_ydown)
    s_yup, R_yup, t_yup = umeyama_align(src_pts, dst_pts_yup)

    angle_ydown = np.degrees(np.arccos(np.clip((np.trace(R_ydown) - 1) / 2, -1, 1)))
    angle_yup = np.degrees(np.arccos(np.clip((np.trace(R_yup) - 1) / 2, -1, 1)))

    print(f"\n  Umeyama (RAS y-down): s={s_ydown:.4f}, R角度={angle_ydown:.2f}°")
    print(f"  Umeyama (RAS y-up):   s={s_yup:.4f}, R角度={angle_yup:.2f}°")

    results = {}

    # ================================================================
    # Test A: 纯平移, s=1, R=I — dump顶点直接转SAPIEN, 中心移到手腕
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test A: 纯平移 (s=1, R=I) — GLB dump → SAPIEN, 中心移到手腕")
    print("  (不经过Umeyama, 直接看GLB在SAPIEN中的原始大小)")
    print("=" * 70)

    glb_dump_sapien_center_A = RXWORLD_TO_SAPIEN @ center_dump
    offset_A = hand_mean_sapien - glb_dump_sapien_center_A

    print(f"  GLB dump中心(SAPIEN): {glb_dump_sapien_center_A}")
    print(f"  手腕SAPIEN: {hand_mean_sapien}")
    print(f"  平移量: {offset_A}")

    def transform_A(verts):
        return (RXWORLD_TO_SAPIEN @ verts.T).T + offset_A

    center_A = transform_and_save_glb(meshes_dump, transform_A,
                                       os.path.join(args.output_dir, "testA_pure_translation_s1.glb"),
                                       "TestA")
    dist_A = np.linalg.norm(center_A - hand_mean_sapien)
    results["A_纯平移_s1"] = dist_A
    print(f"  手-GLB距离: {dist_A:.4f}m")

    # ================================================================
    # Test B: 平移+尺度, R=I — dump顶点, 用Umeyama的s
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test B: 平移+尺度 (s=Umeyama, R=I) — dump顶点 y-up")
    print("  (先缩放GLB, 再平移, 不旋转)")
    print("=" * 70)

    s_B = s_yup
    R_B = np.eye(3)
    t_B = dst_pts_yup.mean(axis=0) - s_B * (R_B @ src_pts.mean(axis=0))
    s_inv_B = 1.0 / s_B
    R_inv_B = R_B.T
    t_inv_B = -s_inv_B * (R_inv_B @ t_B)

    glb_dump_hawor_B = s_inv_B * (R_inv_B @ center_dump) + t_inv_B
    glb_dump_sapien_B = RXWORLD_TO_SAPIEN @ glb_dump_hawor_B
    dist_B = np.linalg.norm(glb_dump_sapien_B - hand_mean_sapien)

    print(f"  s={s_B:.4f}, R=I, t={t_B}")
    print(f"  GLB dump中心(HaWoR y-up): {glb_dump_hawor_B}")
    print(f"  GLB dump中心(SAPIEN z-up): {glb_dump_sapien_B}")
    print(f"  手-GLB距离: {dist_B:.4f}m")

    def transform_B(verts):
        verts_hawor = s_inv_B * (R_inv_B @ verts.T).T + t_inv_B
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_B = transform_and_save_glb(meshes_dump, transform_B,
                                       os.path.join(args.output_dir, "testB_scale_translation_RI.glb"),
                                       "TestB")
    results["B_平移+尺度_RI"] = dist_B

    # ================================================================
    # Test C: yingshe.py 方式 — geom顶点(z-up), RAS相机y-down, Umeyama完整
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test C: yingshe.py 完整方式 (geom z-up, RAS y-down, Umeyama)")
    print("=" * 70)

    s_C = s_ydown
    R_C = R_ydown
    t_C = t_ydown
    s_inv_C = 1.0 / s_C
    R_inv_C = R_C.T
    t_inv_C = -s_inv_C * (R_inv_C @ t_C)

    glb_geom_hawor_C = s_inv_C * (R_inv_C @ center_geom) + t_inv_C
    glb_geom_sapien_C = RXWORLD_TO_SAPIEN @ glb_geom_hawor_C
    dist_C = np.linalg.norm(glb_geom_sapien_C - hand_mean_sapien)

    print(f"  s={s_C:.4f}, R角度={angle_ydown:.2f}°, t={t_C}")
    print(f"  GLB geom中心(HaWoR y-up): {glb_geom_hawor_C}")
    print(f"  GLB geom中心(SAPIEN z-up): {glb_geom_sapien_C}")
    print(f"  手-GLB距离: {dist_C:.4f}m")

    def transform_C(verts):
        verts_hawor = s_inv_C * (R_inv_C @ verts.T).T + t_inv_C
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_C = transform_and_save_glb(meshes_dump, transform_C,
                                       os.path.join(args.output_dir, "testC_yingshe_full.glb"),
                                       "TestC")
    results["C_yingshe_完整"] = dist_C

    # ================================================================
    # Test D: yingshe.py 方式 + 强制R=I
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test D: yingshe.py 方式 + 强制R=I (geom z-up, RAS y-down)")
    print("=" * 70)

    R_D = np.eye(3)
    t_D = dst_pts_ydown.mean(axis=0) - s_C * (R_D @ src_pts.mean(axis=0))
    s_inv_D = 1.0 / s_C
    R_inv_D = R_D.T
    t_inv_D = -s_inv_D * (R_inv_D @ t_D)

    glb_geom_hawor_D = s_inv_D * (R_inv_D @ center_geom) + t_inv_D
    glb_geom_sapien_D = RXWORLD_TO_SAPIEN @ glb_geom_hawor_D
    dist_D = np.linalg.norm(glb_geom_sapien_D - hand_mean_sapien)

    print(f"  s={s_C:.4f}, R=I, t={t_D}")
    print(f"  GLB geom中心(HaWoR y-up): {glb_geom_hawor_D}")
    print(f"  GLB geom中心(SAPIEN z-up): {glb_geom_sapien_D}")
    print(f"  手-GLB距离: {dist_D:.4f}m")

    def transform_D(verts):
        verts_hawor = s_inv_D * (R_inv_D @ verts.T).T + t_inv_D
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_D = transform_and_save_glb(meshes_dump, transform_D,
                                       os.path.join(args.output_dir, "testD_yingshe_RI.glb"),
                                       "TestD")
    results["D_yingshe_RI"] = dist_D

    # ================================================================
    # Test E: 关键测试 — 用相机原点对齐方式
    #   思路: GLB dump(y-up)中心 → RAS相机y-up坐标系 → 
    #         用第一帧RAS相机位置和HaWoR相机位置计算偏移
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test E: 相机原点对齐 (s=Umeyama, R=I)")
    print("  思路: RAS相机y-up[0] ≈ 原点, HaWoR相机[0] ≈ 原点")
    print("  GLB dump中心在RAS y-up坐标系中 → 直接减去RAS相机原点偏移")
    print("=" * 70)

    print(f"  RAS相机y-up[0]: {ras_cam_yup[0]}")
    print(f"  HaWoR相机[0]: {hawor_cam[0]}")
    print(f"  GLB dump中心(y-up): {center_dump}")

    glb_in_ras_yup = center_dump
    glb_rel_ras_origin = glb_in_ras_yup - ras_cam_yup[0]
    print(f"  GLB相对RAS原点(y-up): {glb_rel_ras_origin}")

    glb_hawor_E = s_inv_B * glb_rel_ras_origin + hawor_cam[0]
    glb_sapien_E = RXWORLD_TO_SAPIEN @ glb_hawor_E
    dist_E = np.linalg.norm(glb_sapien_E - hand_mean_sapien)

    print(f"  GLB中心(HaWoR y-up): {glb_hawor_E}")
    print(f"  GLB中心(SAPIEN z-up): {glb_sapien_E}")
    print(f"  手-GLB距离: {dist_E:.4f}m")

    def transform_E(verts):
        verts_rel_ras = verts - ras_cam_yup[0]
        verts_hawor = s_inv_B * verts_rel_ras + hawor_cam[0]
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_E = transform_and_save_glb(meshes_dump, transform_E,
                                       os.path.join(args.output_dir, "testE_cam_origin_align.glb"),
                                       "TestE")
    results["E_相机原点对齐"] = dist_E

    # ================================================================
    # Test F: 相机原点对齐 + s=1 (不缩放)
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test F: 相机原点对齐 + s=1 (不缩放)")
    print("=" * 70)

    glb_hawor_F = (center_dump - ras_cam_yup[0]) + hawor_cam[0]
    glb_sapien_F = RXWORLD_TO_SAPIEN @ glb_hawor_F
    dist_F = np.linalg.norm(glb_sapien_F - hand_mean_sapien)

    print(f"  GLB中心(HaWoR y-up): {glb_hawor_F}")
    print(f"  GLB中心(SAPIEN z-up): {glb_sapien_F}")
    print(f"  手-GLB距离: {dist_F:.4f}m")

    def transform_F(verts):
        verts_rel_ras = verts - ras_cam_yup[0]
        verts_hawor = verts_rel_ras + hawor_cam[0]
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_F = transform_and_save_glb(meshes_dump, transform_F,
                                       os.path.join(args.output_dir, "testF_cam_origin_s1.glb"),
                                       "TestF")
    results["F_相机原点_s1"] = dist_F

    # ================================================================
    # Test G: 直接用 yingshe.py 的变换参数, 但应用到 dump 顶点
    #   yingshe.py 用 geometry.items() 读顶点(z-up), 我们用 dump() 读(y-up)
    #   关键: dump() 顶点已经是 y-up, 不需要再转
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test G: yingshe参数 + dump顶点 (RAS y-down → RAS y-up 转换)")
    print("  yingshe用RAS相机y-down, dump顶点y-up, 需要先把dst转到y-up")
    print("=" * 70)

    s_G = s_ydown
    R_G = R_ydown
    t_G = t_ydown
    s_inv_G = 1.0 / s_G
    R_inv_G = R_G.T
    t_inv_G = -s_inv_G * (R_inv_G @ t_G)

    print(f"  yingshe Umeyama: s={s_G:.4f}, R角度={angle_ydown:.2f}°")
    print(f"  注意: yingshe的dst是RAS相机y-down, 但dump顶点是y-up")
    print(f"  需要先把dump顶点从y-up转到y-down才能用yingshe的变换")

    YUP_TO_YDOWN = R_X
    center_dump_ydown = R_X @ center_dump
    print(f"  dump中心(y-up): {center_dump}")
    print(f"  dump中心(y-down): {center_dump_ydown}")

    glb_ydown_hawor_G = s_inv_G * (R_inv_G @ center_dump_ydown) + t_inv_G
    glb_ydown_sapien_G = RXWORLD_TO_SAPIEN @ glb_ydown_hawor_G
    dist_G = np.linalg.norm(glb_ydown_sapien_G - hand_mean_sapien)

    print(f"  GLB(y-down→HaWoR y-up)中心: {glb_ydown_hawor_G}")
    print(f"  GLB(SAPIEN z-up)中心: {glb_ydown_sapien_G}")
    print(f"  手-GLB距离: {dist_G:.4f}m")

    def transform_G(verts):
        verts_ydown = (R_X @ verts.T).T
        verts_hawor = s_inv_G * (R_inv_G @ verts_ydown.T).T + t_inv_G
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_G = transform_and_save_glb(meshes_dump, transform_G,
                                       os.path.join(args.output_dir, "testG_yingshe_param_dump.glb"),
                                       "TestG")
    results["G_yingshe参数_dump"] = dist_G

    # ================================================================
    # Test H: yingshe参数 + dump顶点 + 强制R=I
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test H: yingshe参数 + dump顶点(y-down) + 强制R=I")
    print("=" * 70)

    R_H = np.eye(3)
    t_H = dst_pts_ydown.mean(axis=0) - s_G * (R_H @ src_pts.mean(axis=0))
    s_inv_H = 1.0 / s_G
    R_inv_H = R_H.T
    t_inv_H = -s_inv_H * (R_inv_H @ t_H)

    glb_ydown_hawor_H = s_inv_H * (R_inv_H @ center_dump_ydown) + t_inv_H
    glb_ydown_sapien_H = RXWORLD_TO_SAPIEN @ glb_ydown_hawor_H
    dist_H = np.linalg.norm(glb_ydown_sapien_H - hand_mean_sapien)

    print(f"  s={s_G:.4f}, R=I, t={t_H}")
    print(f"  GLB(y-down→HaWoR y-up)中心: {glb_ydown_hawor_H}")
    print(f"  GLB(SAPIEN z-up)中心: {glb_ydown_sapien_H}")
    print(f"  手-GLB距离: {dist_H:.4f}m")

    def transform_H(verts):
        verts_ydown = (R_X @ verts.T).T
        verts_hawor = s_inv_H * (R_inv_H @ verts_ydown.T).T + t_inv_H
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_H = transform_and_save_glb(meshes_dump, transform_H,
                                       os.path.join(args.output_dir, "testH_yingshe_param_dump_RI.glb"),
                                       "TestH")
    results["H_yingshe参数_dump_RI"] = dist_H

    # ================================================================
    # 汇总
    # ================================================================
    print("\n" + "=" * 70)
    print("  汇总: 手-GLB距离 (越小越好)")
    print("=" * 70)
    for name, dist in sorted(results.items(), key=lambda x: x[1]):
        marker = " ★" if dist < 0.10 else " ✓" if dist < 0.20 else " ⚠" if dist < 0.35 else " ✗"
        print(f"  {name}: {dist:.4f}m{marker}")

    best_name = min(results, key=results.get)
    best_dist = results[best_name]
    print(f"\n  最佳: {best_name} ({best_dist:.4f}m)")

    print(f"\n  所有测试GLB保存在: {args.output_dir}")
    print(f"\n  查看效果命令 (替换<test_name>):")
    print(f"  python 02_render_scene.py --mode hand_only --viewer \\")
    print(f"      --hawor-dir /home/an/data/hawor/7 \\")
    print(f"      --ras-dir /home/an/data/ras/my_7mp4_result \\")
    print(f"      --sapien-glb {args.output_dir}/<test_name>.glb")


if __name__ == "__main__":
    main()
