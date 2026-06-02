#!/usr/bin/env python3
"""
test_translation_only.py — 逐步测试对齐: 先平移, 再旋转

参考 yingshe.py 的方式, 但分步测试:
  Test 1: 纯平移 (R=I, s=1) — 只把GLB中心移到手腕位置
  Test 2: yingshe.py 方式 (Umeyama s+R+t, 用 geometry.items())
  Test 3: yingshe.py 方式 + 强制R=I
  Test 4: 01_align 方式 (Umeyama s+R+t, 用 dump())
  Test 5: 01_align 方式 + 强制R=I

每种方式生成一个 SAPIEN GLB, 可在 02_render_scene.py 中查看效果。

用法:
    python test_translation_only.py \
        --ras_output /home/an/data/ras/my_7mp4_result \
        --hawor_reconstruction /home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz \
        --output_dir ./output/alignment_test_v2
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
    hawor_cam_pos = hawor_data['t_c2w']
    hawor_R_c2w = hawor_data['R_c2w']
    pred_trans = hawor_data['pred_trans']
    pred_valid = hawor_data['pred_valid']
    return hawor_cam_pos, hawor_R_c2w, pred_trans, pred_valid


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


def load_glb_vertices_geom(ras_glb_path):
    scene = trimesh.load(ras_glb_path)
    all_verts = []
    for name, geom in scene.geometry.items():
        all_verts.append(geom.vertices)
    return np.vstack(all_verts)


def load_glb_vertices_dump(ras_glb_path):
    scene = trimesh.load(ras_glb_path, force='scene')
    meshes = scene.dump()
    return np.vstack([m.vertices for m in meshes]), meshes


def transform_and_save_glb(meshes, transform_func, output_path, test_name):
    sapien_scene = trimesh.Scene()
    for i, mesh in enumerate(meshes):
        vertices = mesh.vertices.copy()
        vertices_new = transform_func(vertices)
        mesh.vertices = vertices_new
        sapien_scene.add_geometry(mesh, geom_name=f"object_{i}")
    sapien_scene.export(output_path)

    tm_check = trimesh.load(output_path, force='scene')
    all_v = []
    for name, g in tm_check.geometry.items():
        all_v.append(g.vertices)
    if all_v:
        all_v = np.vstack(all_v)
        center = all_v.mean(axis=0)
        print(f"  {test_name}: GLB中心(SAPIEN)={center}, 范围min={all_v.min(axis=0)}, max={all_v.max(axis=0)}")
    return center


def main():
    parser = argparse.ArgumentParser(description="逐步测试对齐: 先平移, 再旋转")
    parser.add_argument("--ras_output", type=str, required=True)
    parser.add_argument("--hawor_reconstruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/alignment_test_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  逐步测试对齐: 先平移, 再旋转")
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

    print(f"\n  RAS相机(y-down)[0]: {ras_cam_ydown[0]}")
    print(f"  RAS相机(y-up)[0]:   {ras_cam_yup[0]}")
    print(f"  HaWoR相机(y-up)[0]: {hawor_cam[0]}")
    print(f"  手腕均值(render y-up): {hand_mean_render}")
    print(f"  手腕均值(SAPIEN z-up): {hand_mean_sapien}")

    ras_glb_path = os.path.join(args.ras_output, 'final_scene.glb')
    if not os.path.exists(ras_glb_path):
        print(f"\n✗ GLB文件不存在: {ras_glb_path}")
        return

    verts_geom = load_glb_vertices_geom(ras_glb_path)
    verts_dump, meshes_dump = load_glb_vertices_dump(ras_glb_path)

    center_geom = verts_geom.mean(axis=0)
    center_dump = verts_dump.mean(axis=0)

    print(f"\n  GLB geometry.items() 中心(原始): {center_geom}")
    print(f"  GLB dump() 中心(场景图变换后): {center_dump}")
    print(f"  两者差异: {np.linalg.norm(center_geom - center_dump):.6f}")

    src_pts = np.array([hawor_cam[hi] for _, hi in common_frames])

    results = {}

    # ================================================================
    # Test 1: 纯平移 (R=I, s=1) — GLB dump → SAPIEN, 中心移到手腕
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 1: 纯平移 (R=I, s=1) — dump顶点 y-up → SAPIEN z-up, 中心移到手腕")
    print("=" * 70)

    glb_dump_sapien_center = RXWORLD_TO_SAPIEN @ center_dump
    offset_1 = hand_mean_sapien - glb_dump_sapien_center

    print(f"  GLB dump中心直接转SAPIEN: {glb_dump_sapien_center}")
    print(f"  手腕SAPIEN: {hand_mean_sapien}")
    print(f"  需要平移: {offset_1}")
    print(f"  平移距离: {np.linalg.norm(offset_1):.4f}m")

    def transform_test1(verts):
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts.T).T + offset_1
        return verts_sapien

    center_1 = transform_and_save_glb(meshes_dump, transform_test1,
                                       os.path.join(args.output_dir, "test1_pure_translation.glb"),
                                       "Test1")
    dist_1 = np.linalg.norm(center_1 - hand_mean_sapien)
    results["Test1_纯平移"] = dist_1
    print(f"  手-GLB距离: {dist_1:.4f}m")

    # ================================================================
    # Test 2: yingshe.py 方式 (Umeyama, geometry.items(), RAS相机y-down)
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 2: yingshe.py 方式 (Umeyama, geometry.items(), RAS相机y-down)")
    print("=" * 70)

    dst_pts_ydown = np.array([ras_cam_ydown[ri] for ri, _ in common_frames])
    s_2, R_2, t_2 = umeyama_align(src_pts, dst_pts_ydown)
    angle_2 = np.degrees(np.arccos(np.clip((np.trace(R_2) - 1) / 2, -1, 1)))
    print(f"  Umeyama: s={s_2:.4f}, R角度={angle_2:.2f}°, t={t_2}")

    s_inv_2 = 1.0 / s_2
    R_inv_2 = R_2.T
    t_inv_2 = -s_inv_2 * (R_inv_2 @ t_2)

    glb_geom_hawor = s_inv_2 * (R_inv_2 @ center_geom) + t_inv_2
    glb_geom_sapien = RXWORLD_TO_SAPIEN @ glb_geom_hawor
    dist_2 = np.linalg.norm(glb_geom_sapien - hand_mean_sapien)
    print(f"  GLB geom中心(HaWoR y-up): {glb_geom_hawor}")
    print(f"  GLB geom中心(SAPIEN z-up): {glb_geom_sapien}")
    print(f"  手-GLB距离: {dist_2:.4f}m")

    def transform_test2(verts):
        verts_hawor = s_inv_2 * (R_inv_2 @ verts.T).T + t_inv_2
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_2 = transform_and_save_glb(meshes_dump, transform_test2,
                                       os.path.join(args.output_dir, "test2_yingshe_full.glb"),
                                       "Test2")
    results["Test2_yingshe_Umeyama"] = dist_2

    # ================================================================
    # Test 3: yingshe.py 方式 + 强制R=I
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 3: yingshe.py 方式 + 强制R=I (geometry.items(), RAS相机y-down)")
    print("=" * 70)

    R_3 = np.eye(3)
    t_3 = dst_pts_ydown.mean(axis=0) - s_2 * (R_3 @ src_pts.mean(axis=0))
    s_inv_3 = 1.0 / s_2
    R_inv_3 = R_3.T
    t_inv_3 = -s_inv_3 * (R_inv_3 @ t_3)

    glb_geom_hawor_3 = s_inv_3 * (R_inv_3 @ center_geom) + t_inv_3
    glb_geom_sapien_3 = RXWORLD_TO_SAPIEN @ glb_geom_hawor_3
    dist_3 = np.linalg.norm(glb_geom_sapien_3 - hand_mean_sapien)
    print(f"  s={s_2:.4f}, R=I, t={t_3}")
    print(f"  GLB geom中心(HaWoR y-up): {glb_geom_hawor_3}")
    print(f"  GLB geom中心(SAPIEN z-up): {glb_geom_sapien_3}")
    print(f"  手-GLB距离: {dist_3:.4f}m")

    def transform_test3(verts):
        verts_hawor = s_inv_3 * (R_inv_3 @ verts.T).T + t_inv_3
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_3 = transform_and_save_glb(meshes_dump, transform_test3,
                                       os.path.join(args.output_dir, "test3_yingshe_R_I.glb"),
                                       "Test3")
    results["Test3_yingshe_R_I"] = dist_3

    # ================================================================
    # Test 4: 01_align 方式 (Umeyama, dump(), RAS相机y-up)
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 4: 01_align 方式 (Umeyama, dump(), RAS相机y-up)")
    print("=" * 70)

    dst_pts_yup = np.array([ras_cam_yup[ri] for ri, _ in common_frames])
    s_4, R_4, t_4 = umeyama_align(src_pts, dst_pts_yup)
    angle_4 = np.degrees(np.arccos(np.clip((np.trace(R_4) - 1) / 2, -1, 1)))
    print(f"  Umeyama: s={s_4:.4f}, R角度={angle_4:.2f}°, t={t_4}")

    s_inv_4 = 1.0 / s_4
    R_inv_4 = R_4.T
    t_inv_4 = -s_inv_4 * (R_inv_4 @ t_4)

    glb_dump_hawor_4 = s_inv_4 * (R_inv_4 @ center_dump) + t_inv_4
    glb_dump_sapien_4 = RXWORLD_TO_SAPIEN @ glb_dump_hawor_4
    dist_4 = np.linalg.norm(glb_dump_sapien_4 - hand_mean_sapien)
    print(f"  GLB dump中心(HaWoR y-up): {glb_dump_hawor_4}")
    print(f"  GLB dump中心(SAPIEN z-up): {glb_dump_sapien_4}")
    print(f"  手-GLB距离: {dist_4:.4f}m")

    def transform_test4(verts):
        verts_hawor = s_inv_4 * (R_inv_4 @ verts.T).T + t_inv_4
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_4 = transform_and_save_glb(meshes_dump, transform_test4,
                                       os.path.join(args.output_dir, "test4_01align_full.glb"),
                                       "Test4")
    results["Test4_01align_Umeyama"] = dist_4

    # ================================================================
    # Test 5: 01_align 方式 + 强制R=I
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 5: 01_align 方式 + 强制R=I (dump(), RAS相机y-up)")
    print("=" * 70)

    R_5 = np.eye(3)
    t_5 = dst_pts_yup.mean(axis=0) - s_4 * (R_5 @ src_pts.mean(axis=0))
    s_inv_5 = 1.0 / s_4
    R_inv_5 = R_5.T
    t_inv_5 = -s_inv_5 * (R_inv_5 @ t_5)

    glb_dump_hawor_5 = s_inv_5 * (R_inv_5 @ center_dump) + t_inv_5
    glb_dump_sapien_5 = RXWORLD_TO_SAPIEN @ glb_dump_hawor_5
    dist_5 = np.linalg.norm(glb_dump_sapien_5 - hand_mean_sapien)
    print(f"  s={s_4:.4f}, R=I, t={t_5}")
    print(f"  GLB dump中心(HaWoR y-up): {glb_dump_hawor_5}")
    print(f"  GLB dump中心(SAPIEN z-up): {glb_dump_sapien_5}")
    print(f"  手-GLB距离: {dist_5:.4f}m")

    def transform_test5(verts):
        verts_hawor = s_inv_5 * (R_inv_5 @ verts.T).T + t_inv_5
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_5 = transform_and_save_glb(meshes_dump, transform_test5,
                                       os.path.join(args.output_dir, "test5_01align_R_I.glb"),
                                       "Test5")
    results["Test5_01align_R_I"] = dist_5

    # ================================================================
    # Test 6: 只用 s=1 + 平移 (不用Umeyama的s), R=I
    #         直接用第一帧相机位置对齐
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 6: s=1, R=I, 第一帧相机对齐 (dump顶点)")
    print("=" * 70)

    t_6 = ras_cam_yup[0] - hawor_cam[0]
    s_inv_6 = 1.0
    R_inv_6 = np.eye(3)
    t_inv_6 = -t_6

    glb_dump_hawor_6 = s_inv_6 * (R_inv_6 @ center_dump) + t_inv_6
    glb_dump_sapien_6 = RXWORLD_TO_SAPIEN @ glb_dump_hawor_6
    dist_6 = np.linalg.norm(glb_dump_sapien_6 - hand_mean_sapien)
    print(f"  s=1, R=I, t={t_6}")
    print(f"  GLB dump中心(HaWoR y-up): {glb_dump_hawor_6}")
    print(f"  GLB dump中心(SAPIEN z-up): {glb_dump_sapien_6}")
    print(f"  手-GLB距离: {dist_6:.4f}m")

    def transform_test6(verts):
        verts_hawor = s_inv_6 * (R_inv_6 @ verts.T).T + t_inv_6
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_6 = transform_and_save_glb(meshes_dump, transform_test6,
                                       os.path.join(args.output_dir, "test6_s1_RI_cam0.glb"),
                                       "Test6")
    results["Test6_s1_RI_cam0"] = dist_6

    # ================================================================
    # Test 7: yingshe.py 的坐标原点对齐方式
    #         GLB geom顶点(z-up) → R_axis转y-down → 与RAS相机(y-down)同坐标系
    #         然后Umeyama对齐
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 7: yingshe.py 完整复刻 (geom z-up → R_axis y-down)")
    print("=" * 70)

    ZUP_TO_YDOWN = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    verts_geom_ydown = (ZUP_TO_YDOWN @ verts_geom.T).T
    center_geom_ydown = verts_geom_ydown.mean(axis=0)
    print(f"  GLB geom中心(z-up原始): {center_geom}")
    print(f"  GLB geom中心(y-down转换后): {center_geom_ydown}")

    s_7, R_7, t_7 = umeyama_align(src_pts, dst_pts_ydown)
    angle_7 = np.degrees(np.arccos(np.clip((np.trace(R_7) - 1) / 2, -1, 1)))
    print(f"  Umeyama: s={s_7:.4f}, R角度={angle_7:.2f}°, t={t_7}")

    s_inv_7 = 1.0 / s_7
    R_inv_7 = R_7.T
    t_inv_7 = -s_inv_7 * (R_inv_7 @ t_7)

    glb_geom_ydown_hawor = s_inv_7 * (R_inv_7 @ center_geom_ydown) + t_inv_7
    glb_geom_ydown_sapien = RXWORLD_TO_SAPIEN @ glb_geom_ydown_hawor
    dist_7 = np.linalg.norm(glb_geom_ydown_sapien - hand_mean_sapien)
    print(f"  GLB geom(y-down)中心(HaWoR y-up): {glb_geom_ydown_hawor}")
    print(f"  GLB geom(y-down)中心(SAPIEN z-up): {glb_geom_ydown_sapien}")
    print(f"  手-GLB距离: {dist_7:.4f}m")

    def transform_test7(verts):
        verts_ydown = (ZUP_TO_YDOWN @ verts.T).T
        verts_hawor = s_inv_7 * (R_inv_7 @ verts_ydown.T).T + t_inv_7
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_7 = transform_and_save_glb(meshes_dump, transform_test7,
                                       os.path.join(args.output_dir, "test7_yingshe_zup2ydown.glb"),
                                       "Test7")
    results["Test7_yingshe_zup2ydown"] = dist_7

    # ================================================================
    # Test 8: yingshe.py 完整复刻 + 强制R=I
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 8: yingshe.py 完整复刻 + 强制R=I (geom z-up → R_axis y-down)")
    print("=" * 70)

    R_8 = np.eye(3)
    t_8 = dst_pts_ydown.mean(axis=0) - s_7 * (R_8 @ src_pts.mean(axis=0))
    s_inv_8 = 1.0 / s_7
    R_inv_8 = R_8.T
    t_inv_8 = -s_inv_8 * (R_inv_8 @ t_8)

    glb_geom_ydown_hawor_8 = s_inv_8 * (R_inv_8 @ center_geom_ydown) + t_inv_8
    glb_geom_ydown_sapien_8 = RXWORLD_TO_SAPIEN @ glb_geom_ydown_hawor_8
    dist_8 = np.linalg.norm(glb_geom_ydown_sapien_8 - hand_mean_sapien)
    print(f"  s={s_7:.4f}, R=I, t={t_8}")
    print(f"  GLB geom(y-down)中心(HaWoR y-up): {glb_geom_ydown_hawor_8}")
    print(f"  GLB geom(y-down)中心(SAPIEN z-up): {glb_geom_ydown_sapien_8}")
    print(f"  手-GLB距离: {dist_8:.4f}m")

    def transform_test8(verts):
        verts_ydown = (ZUP_TO_YDOWN @ verts.T).T
        verts_hawor = s_inv_8 * (R_inv_8 @ verts_ydown.T).T + t_inv_8
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_8 = transform_and_save_glb(meshes_dump, transform_test8,
                                       os.path.join(args.output_dir, "test8_yingshe_zup2ydown_R_I.glb"),
                                       "Test8")
    results["Test8_yingshe_zup2ydown_R_I"] = dist_8

    # ================================================================
    # 汇总
    # ================================================================
    print("\n" + "=" * 70)
    print("  汇总: 手-GLB距离 (越小越好)")
    print("=" * 70)
    for name, dist in sorted(results.items(), key=lambda x: x[1]):
        marker = " ★" if dist < 0.15 else " ✓" if dist < 0.30 else " ⚠" if dist < 0.50 else " ✗"
        print(f"  {name}: {dist:.4f}m{marker}")

    best_name = min(results, key=results.get)
    best_dist = results[best_name]
    print(f"\n  最佳: {best_name} ({best_dist:.4f}m)")

    print(f"\n  所有测试GLB保存在: {args.output_dir}")
    print(f"\n  查看效果命令:")
    print(f"  python 02_render_scene.py --mode hand_only --viewer \\")
    print(f"      --hawor-dir <hawor_dir> --ras-dir <ras_dir> \\")
    print(f"      --sapien-glb {args.output_dir}/<test_name>.glb")


if __name__ == "__main__":
    main()
