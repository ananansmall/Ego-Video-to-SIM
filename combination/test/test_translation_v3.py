#!/usr/bin/env python3
"""
test_translation_v3.py — 正确的平移测试: 通过相机轨迹建立坐标系对应

核心思路:
  1. RAS相机在RAS坐标系中有位置, HaWoR相机在HaWoR坐标系中有位置
  2. 两者描述同一个物理相机的运动, 存在变换关系
  3. GLB顶点在RAS坐标系中, 需要变换到HaWoR坐标系
  4. 然后从HaWoR坐标系转到SAPIEN坐标系

关键问题: RAS相机和GLB顶点是否在同一坐标系?
  - RAS外参是w2c, 世界坐标系是Room World
  - GLB dump顶点经过场景图变换后是y-up
  - RAS相机位置从w2c恢复, 是Room World坐标系(y-down, z-forward)

  所以: RAS相机(y-down) 和 GLB dump(y-up) 不在同一坐标系!
  需要先把GLB dump从y-up转到y-down, 或者把RAS相机从y-down转到y-up

测试方案:
  A: GLB dump(y-up) → y-down → 与RAS相机同坐标系 → Umeyama逆变换 → HaWoR → SAPIEN
  B: RAS相机(y-down) → y-up → 与GLB dump同坐标系 → Umeyama逆变换 → HaWoR → SAPIEN
  C: 直接用s=1, 通过相机第一帧位置计算平移

用法:
    python test_translation_v3.py \
        --ras_output /home/an/data/ras/my_7mp4_result \
        --hawor_reconstruction /home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz \
        --output_dir ./output/alignment_test_v4
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
        print(f"    尺寸(XYZ): {extent}")
    return center


def main():
    parser = argparse.ArgumentParser(description="正确的平移测试")
    parser.add_argument("--ras_output", type=str, required=True)
    parser.add_argument("--hawor_reconstruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/alignment_test_v4")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  正确的平移测试: 通过相机轨迹建立坐标系对应")
    print("=" * 70)

    ras_cam_ydown, ras_R_c2w = load_ras_cameras(args.ras_output)
    ras_cam_yup = (R_X @ ras_cam_ydown.T).T
    hawor_cam, hawor_R_c2w, pred_trans, pred_valid = load_hawor_data(args.hawor_reconstruction)

    n_ras = len(ras_cam_ydown)
    n_hawor = len(hawor_cam)
    common_frames = find_frame_correspondence(n_ras, n_hawor)

    src_pts = np.array([hawor_cam[hi] for _, hi in common_frames])

    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            label = "左手" if hi == 0 else "右手"
            hand_mean = pred_trans[hi, v].mean(axis=0)
            hand_sapien = RXWORLD_TO_SAPIEN @ hand_mean
            print(f"  {label}均值(render y-up): {hand_mean}")
            print(f"  {label}均值(SAPIEN z-up): {hand_sapien}")

    ras_glb_path = os.path.join(args.ras_output, 'final_scene.glb')
    if not os.path.exists(ras_glb_path):
        print(f"\n✗ GLB文件不存在: {ras_glb_path}")
        return

    scene_raw = trimesh.load(ras_glb_path, force='scene')
    meshes_dump = scene_raw.dump()
    verts_dump = np.vstack([m.vertices for m in meshes_dump])
    center_dump = verts_dump.mean(axis=0)

    print(f"\n  GLB dump中心(y-up): {center_dump}")
    print(f"  RAS相机(y-down)[0]: {ras_cam_ydown[0]}")
    print(f"  RAS相机(y-up)[0]: {ras_cam_yup[0]}")
    print(f"  HaWoR相机(y-up)[0]: {hawor_cam[0]}")

    results = {}

    # ================================================================
    # 方法1: yingshe.py 的正确方式
    #   GLB顶点用 geometry.items() 读取 (z-up原始)
    #   RAS相机用 y-down (OpenCV)
    #   Umeyama: src=HaWoR(y-up), dst=RAS(y-down)
    #   逆变换: GLB(z-up) → HaWoR(y-up) → SAPIEN(z-up)
    #
    #   关键: GLB z-up顶点和RAS y-down相机不在同一坐标系
    #   yingshe.py假设它们在同一坐标系, 这是错误的!
    #   但yingshe.py的Umeyama旋转178.83°补偿了这个差异
    # ================================================================

    # ================================================================
    # 方法2: 正确的坐标系对齐
    #   GLB dump顶点(y-up) 和 RAS相机(y-up) 在同一坐标系
    #   Umeyama: src=HaWoR(y-up), dst=RAS(y-up)
    #   逆变换: GLB(y-up) → HaWoR(y-up) → SAPIEN(z-up)
    # ================================================================

    dst_pts_yup = np.array([ras_cam_yup[ri] for ri, _ in common_frames])
    s_yup, R_yup, t_yup = umeyama_align(src_pts, dst_pts_yup)
    angle_yup = np.degrees(np.arccos(np.clip((np.trace(R_yup) - 1) / 2, -1, 1)))

    print(f"\n  Umeyama (HaWoR y-up → RAS y-up): s={s_yup:.4f}, R角度={angle_yup:.2f}°")

    # ================================================================
    # Test 1: 完整Umeyama (s+R+t), dump(y-up), RAS(y-up)
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 1: 完整Umeyama (s+R+t), dump(y-up), RAS(y-up)")
    print("=" * 70)

    s_inv_1 = 1.0 / s_yup
    R_inv_1 = R_yup.T
    t_inv_1 = -s_inv_1 * (R_inv_1 @ t_yup)

    def transform_1(verts):
        verts_hawor = s_inv_1 * (R_inv_1 @ verts.T).T + t_inv_1
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_1 = transform_and_save_glb(meshes_dump, transform_1,
                                       os.path.join(args.output_dir, "test1_umeyama_full.glb"),
                                       "Test1")
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_sapien = RXWORLD_TO_SAPIEN @ pred_trans[hi, v].mean(axis=0)
            dist = np.linalg.norm(center_1 - hand_sapien)
            label = "左手" if hi == 0 else "右手"
            results[f"Test1_Umeyama_{label}"] = dist
            print(f"    {label}距离: {dist:.4f}m")

    # ================================================================
    # Test 2: Umeyama + 强制R=I, dump(y-up), RAS(y-up)
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 2: Umeyama + 强制R=I, dump(y-up), RAS(y-up)")
    print("=" * 70)

    R_2 = np.eye(3)
    t_2 = dst_pts_yup.mean(axis=0) - s_yup * (R_2 @ src_pts.mean(axis=0))
    s_inv_2 = 1.0 / s_yup
    R_inv_2 = R_2.T
    t_inv_2 = -s_inv_2 * (R_inv_2 @ t_2)

    print(f"  s={s_yup:.4f}, R=I, t={t_2}")

    def transform_2(verts):
        verts_hawor = s_inv_2 * (R_inv_2 @ verts.T).T + t_inv_2
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_2 = transform_and_save_glb(meshes_dump, transform_2,
                                       os.path.join(args.output_dir, "test2_umeyama_RI.glb"),
                                       "Test2")
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_sapien = RXWORLD_TO_SAPIEN @ pred_trans[hi, v].mean(axis=0)
            dist = np.linalg.norm(center_2 - hand_sapien)
            label = "左手" if hi == 0 else "右手"
            results[f"Test2_Umeyama_RI_{label}"] = dist
            print(f"    {label}距离: {dist:.4f}m")

    # ================================================================
    # Test 3: s=1, R=I, 通过第一帧相机对齐计算平移
    #   思路: RAS相机[0]≈原点, HaWoR相机[0]≈原点
    #   GLB在RAS坐标系中, 需要变换到HaWoR坐标系
    #   p_hawor = R^T @ (p_ras_yup - t)  (s=1)
    #   其中 t = ras_cam_yup[0] - hawor_cam[0]
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 3: s=1, R=I, 第一帧相机对齐")
    print("  GLB dump(y-up) → 减去RAS原点 → 加上HaWoR原点 → SAPIEN")
    print("=" * 70)

    t_3 = ras_cam_yup[0] - hawor_cam[0]
    print(f"  t = RAS_cam_yup[0] - HaWoR_cam[0] = {t_3}")

    def transform_3(verts):
        verts_hawor = verts - t_3
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_3 = transform_and_save_glb(meshes_dump, transform_3,
                                       os.path.join(args.output_dir, "test3_s1_RI_cam_align.glb"),
                                       "Test3")
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_sapien = RXWORLD_TO_SAPIEN @ pred_trans[hi, v].mean(axis=0)
            dist = np.linalg.norm(center_3 - hand_sapien)
            label = "左手" if hi == 0 else "右手"
            results[f"Test3_s1_cam_{label}"] = dist
            print(f"    {label}距离: {dist:.4f}m")

    # ================================================================
    # Test 4: s=1, R=I, 用均值相机位置对齐
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 4: s=1, R=I, 均值相机位置对齐")
    print("=" * 70)

    t_4 = ras_cam_yup.mean(axis=0) - hawor_cam.mean(axis=0)
    print(f"  t = RAS_cam_yup_mean - HaWoR_cam_mean = {t_4}")

    def transform_4(verts):
        verts_hawor = verts - t_4
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_4 = transform_and_save_glb(meshes_dump, transform_4,
                                       os.path.join(args.output_dir, "test4_s1_RI_mean_cam.glb"),
                                       "Test4")
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_sapien = RXWORLD_TO_SAPIEN @ pred_trans[hi, v].mean(axis=0)
            dist = np.linalg.norm(center_4 - hand_sapien)
            label = "左手" if hi == 0 else "右手"
            results[f"Test4_s1_mean_cam_{label}"] = dist
            print(f"    {label}距离: {dist:.4f}m")

    # ================================================================
    # Test 5: 关键测试 — 验证RAS相机和GLB顶点是否在同一坐标系
    #   如果在同一坐标系, RAS相机应该在GLB场景内部或附近
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 5: 验证RAS相机和GLB的空间关系")
    print("  如果RAS相机(y-down)和GLB dump(y-up)在同一坐标系,")
    print("  相机应该在场景附近")
    print("=" * 70)

    ras_cam_ydown_mean = ras_cam_ydown.mean(axis=0)
    print(f"  RAS相机均值(y-down): {ras_cam_ydown_mean}")
    print(f"  GLB dump中心(y-up): {center_dump}")

    ras_cam_yup_mean = ras_cam_yup.mean(axis=0)
    print(f"  RAS相机均值(y-up): {ras_cam_yup_mean}")

    glb_min = verts_dump.min(axis=0)
    glb_max = verts_dump.max(axis=0)
    print(f"  GLB范围(y-up): min={glb_min}, max={glb_max}")

    print(f"\n  RAS相机(y-up)是否在GLB范围内?")
    for ax, name in enumerate(['X', 'Y', 'Z']):
        in_range = glb_min[ax] <= ras_cam_yup_mean[ax] <= glb_max[ax]
        print(f"    {name}: {ras_cam_yup_mean[ax]:.4f} ∈ [{glb_min[ax]:.4f}, {glb_max[ax]:.4f}] → {'✓' if in_range else '✗'}")

    print(f"\n  RAS相机(y-down)是否在GLB范围内?")
    glb_ydown_min = R_X @ glb_max
    glb_ydown_max = R_X @ glb_min
    for ax, name in enumerate(['X', 'Y', 'Z']):
        in_range = glb_ydown_min[ax] <= ras_cam_ydown_mean[ax] <= glb_ydown_max[ax]
        print(f"    {name}: {ras_cam_ydown_mean[ax]:.4f} ∈ [{glb_ydown_min[ax]:.4f}, {glb_ydown_max[ax]:.4f}] → {'✓' if in_range else '✗'}")

    # ================================================================
    # Test 6: yingshe.py 方式 — 用 geometry.items() 顶点(z-up)
    #   RAS相机(y-down), Umeyama
    #   但这次我们正确理解: GLB z-up 和 RAS y-down 不在同一坐标系
    #   需要先把 GLB z-up → y-down 才能用 RAS y-down 的 Umeyama
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 6: yingshe方式 + 正确坐标转换")
    print("  GLB geom(z-up) → y-down → 与RAS相机同坐标系 → Umeyama → HaWoR → SAPIEN")
    print("=" * 70)

    scene_geom = trimesh.load(ras_glb_path)
    verts_geom = []
    for name, geom in scene_geom.geometry.items():
        verts_geom.append(geom.vertices)
    verts_geom = np.vstack(verts_geom)
    center_geom = verts_geom.mean(axis=0)

    ZUP_TO_YDOWN = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    verts_geom_ydown = (ZUP_TO_YDOWN @ verts_geom.T).T
    center_geom_ydown = verts_geom_ydown.mean(axis=0)

    print(f"  GLB geom中心(z-up): {center_geom}")
    print(f"  GLB geom中心(y-down): {center_geom_ydown}")

    dst_pts_ydown = np.array([ras_cam_ydown[ri] for ri, _ in common_frames])
    s_ydown, R_ydown, t_ydown = umeyama_align(src_pts, dst_pts_ydown)
    angle_ydown = np.degrees(np.arccos(np.clip((np.trace(R_ydown) - 1) / 2, -1, 1)))

    print(f"  Umeyama (HaWoR y-up → RAS y-down): s={s_ydown:.4f}, R角度={angle_ydown:.2f}°")

    s_inv_6 = 1.0 / s_ydown
    R_inv_6 = R_ydown.T
    t_inv_6 = -s_inv_6 * (R_inv_6 @ t_ydown)

    def transform_6(verts):
        verts_ydown = (ZUP_TO_YDOWN @ verts.T).T
        verts_hawor = s_inv_6 * (R_inv_6 @ verts_ydown.T).T + t_inv_6
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_6 = transform_and_save_glb(meshes_dump, transform_6,
                                       os.path.join(args.output_dir, "test6_yingshe_correct.glb"),
                                       "Test6")
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_sapien = RXWORLD_TO_SAPIEN @ pred_trans[hi, v].mean(axis=0)
            dist = np.linalg.norm(center_6 - hand_sapien)
            label = "左手" if hi == 0 else "右手"
            results[f"Test6_yingshe_correct_{label}"] = dist
            print(f"    {label}距离: {dist:.4f}m")

    # ================================================================
    # Test 7: yingshe方式 + 强制R=I
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 7: yingshe方式 + 强制R=I")
    print("=" * 70)

    R_7 = np.eye(3)
    t_7 = dst_pts_ydown.mean(axis=0) - s_ydown * (R_7 @ src_pts.mean(axis=0))
    s_inv_7 = 1.0 / s_ydown
    R_inv_7 = R_7.T
    t_inv_7 = -s_inv_7 * (R_inv_7 @ t_7)

    print(f"  s={s_ydown:.4f}, R=I, t={t_7}")

    def transform_7(verts):
        verts_ydown = (ZUP_TO_YDOWN @ verts.T).T
        verts_hawor = s_inv_7 * (R_inv_7 @ verts_ydown.T).T + t_inv_7
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_7 = transform_and_save_glb(meshes_dump, transform_7,
                                       os.path.join(args.output_dir, "test7_yingshe_RI.glb"),
                                       "Test7")
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_sapien = RXWORLD_TO_SAPIEN @ pred_trans[hi, v].mean(axis=0)
            dist = np.linalg.norm(center_7 - hand_sapien)
            label = "左手" if hi == 0 else "右手"
            results[f"Test7_yingshe_RI_{label}"] = dist
            print(f"    {label}距离: {dist:.4f}m")

    # ================================================================
    # Test 8: s=1, R=I, yingshe坐标转换 + 第一帧相机对齐
    # ================================================================
    print("\n" + "=" * 70)
    print("  Test 8: s=1, R=I, geom(z-up→y-down) + 第一帧相机对齐")
    print("=" * 70)

    t_8 = ras_cam_ydown[0] - hawor_cam[0]
    print(f"  t = RAS_cam_ydown[0] - HaWoR_cam[0] = {t_8}")

    def transform_8(verts):
        verts_ydown = (ZUP_TO_YDOWN @ verts.T).T
        verts_hawor = verts_ydown - t_8
        verts_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        return verts_sapien

    center_8 = transform_and_save_glb(meshes_dump, transform_8,
                                       os.path.join(args.output_dir, "test8_s1_ydown_cam.glb"),
                                       "Test8")
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_sapien = RXWORLD_TO_SAPIEN @ pred_trans[hi, v].mean(axis=0)
            dist = np.linalg.norm(center_8 - hand_sapien)
            label = "左手" if hi == 0 else "右手"
            results[f"Test8_s1_ydown_cam_{label}"] = dist
            print(f"    {label}距离: {dist:.4f}m")

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


if __name__ == "__main__":
    main()
