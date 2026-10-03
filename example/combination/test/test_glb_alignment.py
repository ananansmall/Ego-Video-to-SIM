#!/usr/bin/env python3
"""
test_glb_alignment.py — GLB 对齐测试: 可视化坐标系原点 + 对比 yingshe.py 和 01_align_scene.py

功能:
  1. 运行 yingshe.py 方式的 Umeyama 对齐 (假设两者都是 y-up)
  2. 运行 01_align_scene.py 方式的 Umeyama 对齐 (HaWoR y-up → RAS y-down)
  3. 可视化两个坐标系的原点 (红色=RAS, 蓝色=HaWoR, 绿色=GLB中心)
  4. 对比两种方法的 GLB 变换结果
  5. 生成 SAPIEN 场景可视化 (含坐标原点标注 + 坐标轴)

坐标系说明:
  geometry.items() → 原始 z-up 顶点 (不含场景图变换)
  dump()           → y-up 顶点 (已含场景图变换, GLB/glTF 标准)
  RAS 相机         → y-down (OpenCV)
  HaWoR 相机       → y-up (OpenGL, 已含 R_x)

  ZUP_TO_YDOWN = [[1,0,0],[0,0,-1],[0,1,0]]  → (x,y,z) → (x,-z,y)  真正的 z-up → OpenCV y-down
  YUP_TO_YDOWN = [[1,0,0],[0,-1,0],[0,0,-1]]  → (x,y,z) → (x,-y,-z) y-up → y-down

用法:
    python test/test_glb_alignment.py \
        --ras_output /home/an/data/ras/my_7mp4_result \
        --hawor_reconstruction /home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz \
        --output_dir ./output/alignment_test_v6
"""

import argparse
import os
import numpy as np
from glob import glob

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
R_X = np.diag([1.0, -1.0, -1.0])
RXWORLD_TO_SAPIEN = R_AXIS @ R_X
ZUP_TO_YDOWN = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
YUP_TO_YDOWN = np.diag([1.0, -1.0, -1.0])


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


def load_hawor_cameras(hawor_reconstruction):
    hawor_data = dict(np.load(hawor_reconstruction, allow_pickle=True))
    return hawor_data['t_c2w'], hawor_data['R_c2w'], hawor_data


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


def load_glb_geom_zup(glb_path):
    if not os.path.exists(glb_path) or trimesh is None:
        return None, None

    scene = trimesh.load(glb_path)
    all_verts = []
    for name, geom in scene.geometry.items():
        all_verts.append(geom.vertices)
    all_verts = np.vstack(all_verts)
    return all_verts, all_verts.mean(axis=0)


def load_glb_dump_yup(glb_path):
    if not os.path.exists(glb_path) or trimesh is None:
        return None, None

    tm_scene = trimesh.load(glb_path, force='scene')
    world_meshes = tm_scene.dump()
    all_verts = []
    for mesh in world_meshes:
        all_verts.append(mesh.vertices.copy())
    all_verts = np.vstack(all_verts)
    return all_verts, all_verts.mean(axis=0)


def plot_origins_matplotlib(ras_origin_sapien, hawor_origin_sapien, glb_centers, output_path):
    if not HAS_MPL:
        print("  matplotlib 不可用, 跳过 3D 可视化")
        return

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(*ras_origin_sapien, c='red', s=300, marker='o', label='RAS Origin (RED)', zorder=5, edgecolors='darkred', linewidths=2)
    ax.scatter(*hawor_origin_sapien, c='blue', s=300, marker='^', label='HaWoR Origin (BLUE)', zorder=5, edgecolors='darkblue', linewidths=2)

    axis_len = 0.05
    for origin, color, lw in [(ras_origin_sapien, 'red', 2.5), (hawor_origin_sapien, 'blue', 2.5)]:
        for axis_i, (dx, dy, dz) in enumerate([(axis_len,0,0), (0,axis_len,0), (0,0,axis_len)]):
            ax.quiver(origin[0], origin[1], origin[2], dx, dy, dz,
                      color=color, arrow_length_ratio=0.2, linewidth=lw)

    for label, (center, color) in glb_centers.items():
        ax.scatter(*center, c=[color], s=200, marker='s', label=label, zorder=4, edgecolors='black', linewidths=1)

    ax.set_xlabel('X (SAPIEN)')
    ax.set_ylabel('Y (SAPIEN)')
    ax.set_zlabel('Z (SAPIEN)')
    ax.set_title('Coordinate System Origins in SAPIEN (z-up)\nRED=RAS Origin, BLUE=HaWoR Origin')
    ax.legend(loc='upper left', fontsize=8)

    all_pts = [ras_origin_sapien, hawor_origin_sapien] + [c for _, (c, _) in glb_centers.items()]
    all_pts = np.array(all_pts)
    center = all_pts.mean(axis=0)
    max_range = max(np.ptp(all_pts, axis=0)) / 2 + 0.1
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  3D 原点图已保存: {output_path}")


def generate_sapien_glb_with_origins(glb_verts_sapien, ras_origin_sapien, hawor_origin_sapien,
                                      glb_centers_dict, output_path):
    if trimesh is None:
        return

    scene = trimesh.Scene()

    if glb_verts_sapien is not None:
        mesh = trimesh.Trimesh(vertices=glb_verts_sapien)
        mesh.visual.vertex_colors = [150, 150, 150, 100]
        scene.add_geometry(mesh, geom_name="glb_scene")

    origin_radius = 0.02

    ras_sphere = trimesh.creation.icosphere(subdivisions=3, radius=origin_radius)
    ras_sphere.visual.vertex_colors = [255, 0, 0, 255]
    ras_sphere.apply_translation(ras_origin_sapien)
    scene.add_geometry(ras_sphere, geom_name="RAS_origin_RED")

    hawor_sphere = trimesh.creation.icosphere(subdivisions=3, radius=origin_radius)
    hawor_sphere.visual.vertex_colors = [0, 0, 255, 255]
    hawor_sphere.apply_translation(hawor_origin_sapien)
    scene.add_geometry(hawor_sphere, geom_name="HaWoR_origin_BLUE")

    for label, (center, color_rgb) in glb_centers_dict.items():
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=origin_radius * 0.8)
        r, g, b = int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255)
        sphere.visual.vertex_colors = [r, g, b, 255]
        sphere.apply_translation(center)
        scene.add_geometry(sphere, geom_name=label)

    axis_len = 0.05
    for origin, name_prefix in [(ras_origin_sapien, "RAS"), (hawor_origin_sapien, "HaWoR")]:
        for axis_i, axis_color in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
            end = origin.copy()
            end[axis_i] += axis_len
            path = trimesh.load_path(np.array([origin, end]))
            r, g, b = int(axis_color[0]*255), int(axis_color[1]*255), int(axis_color[2]*255)
            path.colors = np.array([[r, g, b, 255]])
            scene.add_geometry(path, geom_name=f"{name_prefix}_axis{axis_i}")

    for label, (center, color_rgb) in glb_centers_dict.items():
        for axis_i, axis_color in enumerate([(1,0,0), (0,1,0), (0,0,1)]):
            end = center.copy()
            end[axis_i] += axis_len * 0.6
            path = trimesh.load_path(np.array([center, end]))
            r, g, b = int(axis_color[0]*255), int(axis_color[1]*255), int(axis_color[2]*255)
            path.colors = np.array([[r, g, b, 255]])
            scene.add_geometry(path, geom_name=f"{label}_axis{axis_i}")

    scene.export(output_path)
    print(f"  SAPIEN GLB (含原点标注) 已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GLB 对齐测试: 可视化坐标系原点")
    parser.add_argument("--ras_output", type=str, required=True)
    parser.add_argument("--hawor_reconstruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/alignment_test_v6")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # 1. 加载数据
    # ============================================================
    print("=" * 70)
    print("Step 1: 加载数据")
    print("=" * 70)

    ras_cam_pos, ras_R_c2w = load_ras_cameras(args.ras_output)
    hawor_cam_pos, hawor_R_c2w, hawor_data = load_hawor_cameras(args.hawor_reconstruction)

    n_ras = len(ras_cam_pos)
    n_hawor = len(hawor_cam_pos)
    print(f"  RAS: {n_ras} 帧, cam[0] = {ras_cam_pos[0]}")
    print(f"  HaWoR: {n_hawor} 帧, cam[0] = {hawor_cam_pos[0]}")
    print(f"  RAS 坐标系: y-down, z-forward (OpenCV)")
    print(f"  HaWoR 坐标系: y-up (OpenGL, 已含 R_x)")

    common_frames = find_frame_correspondence(n_ras, n_hawor)

    ras_glb_path = os.path.join(args.ras_output, 'final_scene.glb')

    # 加载 GLB 顶点 (两种方式)
    glb_verts_zup, glb_center_zup = load_glb_geom_zup(ras_glb_path)
    glb_verts_yup_dump, glb_center_yup_dump = load_glb_dump_yup(ras_glb_path)

    if glb_center_zup is not None:
        print(f"\n  GLB geom.items() 中心 (z-up): {glb_center_zup}")
    if glb_center_yup_dump is not None:
        print(f"  GLB dump() 中心 (y-up): {glb_center_yup_dump}")

    # ============================================================
    # 2. yingshe.py 方式: 假设两者都是 y-up, 直接 Umeyama
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 2: yingshe.py 方式 (假设 RAS 和 HaWoR 都是 y-up)")
    print("=" * 70)
    print("  yingshe.py 的假设: RAS 相机位置 = y-up, HaWoR 相机位置 = y-up")
    print("  实际: RAS 相机位置 = y-down, HaWoR 相机位置 = y-up")
    print("  因此 Umeyama 的旋转 R 会补偿 y-up/y-down 的差异")

    src_pts_yingshe = np.array([hawor_cam_pos[hi] for _, hi in common_frames])
    dst_pts_yingshe = np.array([ras_cam_pos[ri] for ri, _ in common_frames])

    s_y, R_y, t_y = umeyama_align(src_pts_yingshe, dst_pts_yingshe)

    angle_y = np.degrees(np.arccos(np.clip((np.trace(R_y) - 1) / 2, -1, 1)))
    print(f"  尺度: {s_y:.4f}")
    print(f"  旋转角度: {angle_y:.2f}°")
    print(f"  旋转矩阵:\n{R_y}")
    print(f"  平移: {t_y}")

    aligned_y = s_y * (R_y @ src_pts_yingshe.T).T + t_y
    errors_y = np.linalg.norm(aligned_y - dst_pts_yingshe, axis=1)
    print(f"\n  对齐误差:")
    print(f"    mean  = {errors_y.mean():.6f} 米")
    print(f"    max   = {errors_y.max():.6f} 米")
    print(f"    rmse  = {np.sqrt(np.mean(errors_y**2)):.6f} 米")

    s_inv_y = 1.0 / s_y
    R_inv_y = R_y.T
    t_inv_y = -s_inv_y * (R_inv_y @ t_y)

    # yingshe.py: GLB geometry.items() 顶点 (实际 z-up, yingshe 假设 y-up) → 直接 Umeyama 逆 → HaWoR y-up
    glb_center_yingshe_sapien = None
    glb_verts_yingshe_sapien = None
    if glb_verts_zup is not None:
        verts_hawor_yingshe = s_inv_y * (R_inv_y @ glb_verts_zup.T).T + t_inv_y
        glb_verts_yingshe_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor_yingshe.T).T
        glb_center_yingshe_sapien = glb_verts_yingshe_sapien.mean(axis=0)
        print(f"\n  yingshe GLB中心 (SAPIEN z-up): {glb_center_yingshe_sapien}")
        print(f"  注意: yingshe 把 geometry.items() 顶点当作 y-up, 但实际是 z-up")

    # ============================================================
    # 3. 01_align_scene.py 方式: 正确区分坐标系
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 3: 01_align_scene.py 方式 (正确区分坐标系)")
    print("=" * 70)
    print("  变换链: GLB geom.items() z-up → ZUP_TO_YDOWN → y-down → Umeyama 逆 → HaWoR y-up → SAPIEN z-up")
    print(f"  ZUP_TO_YDOWN = {ZUP_TO_YDOWN.tolist()}")
    print(f"  效果: (x,y,z) → (x,-z,y)")

    # Umeyama: src=HaWoR(y-up), dst=RAS(y-down)
    src_pts_01 = np.array([hawor_cam_pos[hi] for _, hi in common_frames])
    dst_pts_01 = np.array([ras_cam_pos[ri] for ri, _ in common_frames])

    s_01, R_01, t_01 = umeyama_align(src_pts_01, dst_pts_01)

    angle_01 = np.degrees(np.arccos(np.clip((np.trace(R_01) - 1) / 2, -1, 1)))
    print(f"\n  尺度: {s_01:.4f}")
    print(f"  旋转角度: {angle_01:.2f}°")
    print(f"  旋转矩阵:\n{R_01}")
    print(f"  平移: {t_01}")

    aligned_01 = s_01 * (R_01 @ src_pts_01.T).T + t_01
    errors_01 = np.linalg.norm(aligned_01 - dst_pts_01, axis=1)
    print(f"\n  对齐误差:")
    print(f"    mean  = {errors_01.mean():.6f} 米")
    print(f"    max   = {errors_01.max():.6f} 米")
    print(f"    rmse  = {np.sqrt(np.mean(errors_01**2)):.6f} 米")

    s_inv_01 = 1.0 / s_01
    R_inv_01 = R_01.T
    t_inv_01 = -s_inv_01 * (R_inv_01 @ t_01)

    # 方法 A: geometry.items() z-up → ZUP_TO_YDOWN → y-down → Umeyama 逆 → HaWoR → SAPIEN
    glb_center_01_geom_sapien = None
    glb_verts_01_geom_sapien = None
    if glb_verts_zup is not None:
        verts_ydown = (ZUP_TO_YDOWN @ glb_verts_zup.T).T
        verts_hawor = s_inv_01 * (R_inv_01 @ verts_ydown.T).T + t_inv_01
        glb_verts_01_geom_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor.T).T
        glb_center_01_geom_sapien = glb_verts_01_geom_sapien.mean(axis=0)
        print(f"\n  方法A [geom z-up → ZUP_TO_YDOWN → Umeyama逆 → SAPIEN]:")
        print(f"    GLB中心 (SAPIEN z-up): {glb_center_01_geom_sapien}")

    # 方法 B: dump() y-up → YUP_TO_YDOWN → y-down → Umeyama 逆 → HaWoR → SAPIEN
    glb_center_01_dump_sapien = None
    glb_verts_01_dump_sapien = None
    if glb_verts_yup_dump is not None:
        verts_ydown_from_yup = (YUP_TO_YDOWN @ glb_verts_yup_dump.T).T
        verts_hawor_dump = s_inv_01 * (R_inv_01 @ verts_ydown_from_yup.T).T + t_inv_01
        glb_verts_01_dump_sapien = (RXWORLD_TO_SAPIEN @ verts_hawor_dump.T).T
        glb_center_01_dump_sapien = glb_verts_01_dump_sapien.mean(axis=0)
        print(f"\n  方法B [dump y-up → YUP_TO_YDOWN → Umeyama逆 → SAPIEN]:")
        print(f"    GLB中心 (SAPIEN z-up): {glb_center_01_dump_sapien}")

    # ============================================================
    # 4. 坐标系原点可视化
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 4: 坐标系原点可视化")
    print("=" * 70)

    ras_origin_ydown = np.zeros(3)
    ras_origin_sapien = RXWORLD_TO_SAPIEN @ (s_inv_01 * (R_inv_01 @ ras_origin_ydown) + t_inv_01)

    hawor_origin_yup = np.zeros(3)
    hawor_origin_sapien = RXWORLD_TO_SAPIEN @ hawor_origin_yup

    print(f"  RAS 原点 (0,0,0 in y-down) → HaWoR → SAPIEN: {ras_origin_sapien}")
    print(f"  HaWoR 原点 (0,0,0 in y-up) → SAPIEN: {hawor_origin_sapien}")

    ras_origin_yingshe_sapien = RXWORLD_TO_SAPIEN @ (s_inv_y * (R_inv_y @ np.zeros(3)) + t_inv_y)
    print(f"  RAS 原点 (0,0,0, yingshe假设y-up) → HaWoR → SAPIEN: {ras_origin_yingshe_sapien}")

    glb_centers = {}
    if glb_center_yingshe_sapien is not None:
        glb_centers["yingshe (假设y-up)"] = (glb_center_yingshe_sapien, (0.0, 0.8, 0.0))
    if glb_center_01_geom_sapien is not None:
        glb_centers["01align geom+ZUP2YDOWN"] = (glb_center_01_geom_sapien, (1.0, 0.5, 0.0))
    if glb_center_01_dump_sapien is not None:
        glb_centers["01align dump+YUP2YDOWN"] = (glb_center_01_dump_sapien, (0.0, 0.5, 1.0))

    plot_path = os.path.join(args.output_dir, "origin_comparison.png")
    plot_origins_matplotlib(ras_origin_sapien, hawor_origin_sapien, glb_centers, plot_path)

    # ============================================================
    # 5. 手部 vs GLB 距离验证
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 5: 手部 vs GLB 距离验证")
    print("=" * 70)

    pred_trans = hawor_data['pred_trans']
    pred_valid = hawor_data['pred_valid']

    hand_sapien_dict = {}
    for hi in [0, 1]:
        v = pred_valid[hi]
        if v.any():
            hand_mean_render = pred_trans[hi, v].mean(axis=0)
            hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render
            label = "左手" if hi == 0 else "右手"
            hand_sapien_dict[label] = hand_mean_sapien
            print(f"\n  {label}均值 (render y-up): {hand_mean_render}")
            print(f"  {label}均值 (SAPIEN z-up): {hand_mean_sapien}")

            for method_name, (center, color) in glb_centers.items():
                dist = np.linalg.norm(hand_mean_sapien - center)
                print(f"  {label} → {method_name} GLB中心距离: {dist:.4f}m")

    # ============================================================
    # 6. 生成带原点标注的 GLB
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 6: 生成带原点标注的 GLB")
    print("=" * 70)

    best_verts = glb_verts_01_geom_sapien if glb_verts_01_geom_sapien is not None else glb_verts_01_dump_sapien
    if best_verts is not None:
        glb_path = os.path.join(args.output_dir, "alignment_with_origins.glb")
        generate_sapien_glb_with_origins(
            best_verts, ras_origin_sapien, hawor_origin_sapien,
            glb_centers, glb_path)

    if glb_verts_yingshe_sapien is not None:
        glb_path_y = os.path.join(args.output_dir, "yingshe_alignment_with_origins.glb")
        generate_sapien_glb_with_origins(
            glb_verts_yingshe_sapien, ras_origin_yingshe_sapien, hawor_origin_sapien,
            {"yingshe (假设y-up)": (glb_center_yingshe_sapien, (0.0, 0.8, 0.0))},
            glb_path_y)

    # ============================================================
    # 7. 纯原点平移测试
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 7: 纯原点平移测试 (s=1, R=I, 只用平移对齐原点)")
    print("=" * 70)

    if glb_center_zup is not None:
        glb_center_ydown = ZUP_TO_YDOWN @ glb_center_zup
        hand_mean_yup = pred_trans[0, pred_valid[0]].mean(axis=0) if pred_valid[0].any() else np.zeros(3)

        translation_only = hand_mean_yup - glb_center_ydown
        print(f"  GLB中心 (z-up, geometry.items): {glb_center_zup}")
        print(f"  GLB中心 (y-down, ZUP_TO_YDOWN后): {glb_center_ydown}")
        print(f"  手部均值 (y-up): {hand_mean_yup}")
        print(f"  纯平移 (hand_yup - glb_ydown): {translation_only}")
        print(f"  注意: y-up 和 y-down 坐标系不同, 此平移无直接物理意义")

        if glb_center_01_geom_sapien is not None:
            hand_sapien = RXWORLD_TO_SAPIEN @ hand_mean_yup
            print(f"\n  01_align GLB中心 (SAPIEN): {glb_center_01_geom_sapien}")
            print(f"  手部均值 (SAPIEN): {hand_sapien}")
            print(f"  距离: {np.linalg.norm(hand_sapien - glb_center_01_geom_sapien):.4f}m")

    # ============================================================
    # 8. yingshe.py 自身测试
    # ============================================================
    print("\n" + "=" * 70)
    print("Step 8: yingshe.py 自身对齐测试")
    print("=" * 70)
    print("  yingshe.py 假设: RAS 和 HaWoR 坐标系都是 y-up, 米制, 相机看+z")
    print(f"  实际 RAS 相机坐标: y-down, z-forward (OpenCV)")
    print(f"  实际 HaWoR 相机坐标: y-up (OpenGL, 已含 R_x)")
    print(f"  实际 GLB geometry.items() 顶点: z-up (不是 y-up)")
    print()

    if glb_center_zup is not None:
        print(f"  GLB中心 (原始 z-up, geometry.items): {glb_center_zup}")
        print(f"  如果当作 y-up 解释: {glb_center_zup} (yingshe 的做法)")
        print(f"  正确的 y-down 解释: {ZUP_TO_YDOWN @ glb_center_zup}")
        print()

        if glb_center_yingshe_sapien is not None:
            print(f"  yingshe GLB中心 (SAPIEN z-up): {glb_center_yingshe_sapien}")
            for label, hand_sapien in hand_sapien_dict.items():
                dist = np.linalg.norm(hand_sapien - glb_center_yingshe_sapien)
                print(f"  yingshe {label} → GLB距离: {dist:.4f}m")

        if glb_center_01_geom_sapien is not None:
            print(f"\n  01_align GLB中心 (SAPIEN z-up): {glb_center_01_geom_sapien}")
            for label, hand_sapien in hand_sapien_dict.items():
                dist = np.linalg.norm(hand_sapien - glb_center_01_geom_sapien)
                print(f"  01_align {label} → GLB距离: {dist:.4f}m")

    # ============================================================
    # 对比总结
    # ============================================================
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    print(f"  yingshe.py 方式: s={s_y:.4f}, R角度={angle_y:.2f}°, 误差mean={errors_y.mean():.6f}m")
    print(f"  01_align 方式:   s={s_01:.4f}, R角度={angle_01:.2f}°, 误差mean={errors_01.mean():.6f}m")
    print()
    print("  注意: 两种方式的 Umeyama 参数完全相同 (相同的 src/dst 点)")
    print("  因为 RAS 相机位置直接从 w2c 计算, 没有做坐标变换")
    print("  区别在于 GLB 顶点的坐标解释和变换链不同:")
    print()
    print("  yingshe.py (错误但可能碰巧接近):")
    print("    GLB geom.items() 顶点 (实际 z-up) → 假设为 y-up → 直接 Umeyama 逆 → HaWoR y-up → SAPIEN")
    print()
    print("  01_align_scene.py (正确):")
    print("    GLB geom.items() 顶点 (z-up) → ZUP_TO_YDOWN → y-down → Umeyama 逆 → HaWoR y-up → SAPIEN")
    print()
    print("  ZUP_TO_YDOWN = [[1,0,0],[0,0,-1],[0,1,0]]")
    print("    效果: (x,y,z) → (x,-z,y)")
    print("    z-up 中 z 轴朝上 → y-down 中 y 轴朝下 (z 变成 -y, y 变成 z)")

    print(f"\n  输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
