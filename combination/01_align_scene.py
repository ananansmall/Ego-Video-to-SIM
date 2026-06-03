"""
01_align_scene.py — 基于第一帧相机位姿对齐 RAS GLB 到 HaWoR 坐标系

核心原理:
  RAS 和 HaWoR 处理同一个视频, 第一帧相机的位姿在两个系统中描述同一个物理相机。
  以第一帧相机为锚点, 计算 RAS 世界 → HaWoR 世界的变换。

坐标系分析:
  RAS 外参: Z-UP 坐标系 (房间对齐后, 地板在 z=0)
  RAS GLB:  Y-UP 坐标系 (导出时做了 z-up → y-up 转换)
  HaWoR:    render world (含 R_x = diag(1,-1,-1) 变换)

  RAS 相机: OpenCV 约定 (X=right, Y=down, Z=forward)
  HaWoR 相机: OpenGL 约定 (X=right, Y=up, Z=backward)

变换链:
  Step 1: RAS 外参 Z-UP → Y-UP (与 GLB 一致)
    ZUP_TO_YUP = [[1,0,0],[0,0,1],[0,-1,0]]
    R_c2w_yup = ZUP_TO_YUP @ R_c2w_zup @ ZUP_TO_YUP.T
    t_c2w_yup = ZUP_TO_YUP @ t_c2w_zup

  Step 2: 第一帧相机位姿对齐 (RAS Y-UP → HaWoR render world)
    同一个物理相机, OpenCV 和 OpenGL 约定之间差 OPENCV_TO_OPENGL = diag(1,-1,-1)
    R_align = R_c2w_hawor @ OPENCV_TO_OPENGL @ R_c2w_ras_yup.T
    t_align = t_c2w_hawor - R_align @ t_c2w_ras_yup

  Step 3: 尺度校正 (Umeyama)
    RAS 和 HaWoR 的尺度不同, 通过相机轨迹计算尺度比
    s_inv = sigma_hawor / sigma_ras (RAS 缩放到 HaWoR 的尺度)

  最终变换 (GLB Y-UP → HaWoR render world):
    p_hawor = s_inv * R_align @ p_glb + t_align_scaled
    其中 t_align_scaled = t_c2w_hawor - R_align @ (s_inv * t_c2w_ras_yup)

  保存到 transform_params.npz:
    s_inv, R_inv=R_align, t_inv=t_align_scaled

  HaWoR (y-up render world) → SAPIEN (z-up):
    p_sapien = RXWORLD_TO_SAPIEN @ p_hawor

用法:
    python 01_align_scene.py \\
        --ras_output /path/to/ras_output \\
        --hawor_reconstruction /path/to/hawor_results_*.npz \\
        --output_dir ./output/alignment
"""

import argparse
import os
import numpy as np
from glob import glob

ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
R_X = np.diag([1.0, -1.0, -1.0])
RXWORLD_TO_SAPIEN = R_AXIS @ R_X


def umeyama_scale(src_pts, dst_pts):
    src_centered = src_pts - src_pts.mean(axis=0)
    dst_centered = dst_pts - dst_pts.mean(axis=0)
    sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
    if sigma_src < 1e-8:
        return 1.0
    return sigma_dst / sigma_src


def main():
    parser = argparse.ArgumentParser(
        description="基于第一帧相机位姿对齐 RAS GLB 到 HaWoR 坐标系"
    )
    parser.add_argument("--ras_output", type=str, required=True)
    parser.add_argument("--hawor_reconstruction", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/alignment")
    parser.add_argument("--force_scale", type=float, default=None,
                        help="强制对齐尺度因子 (None=Umeyama自动计算)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ============================================================
    # 1. 加载 RAS 第一帧相机 (Z-UP 坐标系)
    # ============================================================
    print("=" * 60)
    print("Step 1: 加载 RAS 相机位姿 (Z-UP 坐标系)")
    print("=" * 60)

    ext_dir = os.path.join(args.ras_output, 'extrinsics')
    ext_files = sorted(glob(os.path.join(ext_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    if not ext_files:
        raise FileNotFoundError(f"未找到RAS外参: {ext_dir}/")

    ras_cam_pos_zup = []
    ras_R_c2w_zup = []
    for f in ext_files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        R_c2w = R_w2c.T
        cam_pos = -R_c2w @ t_w2c
        ras_cam_pos_zup.append(cam_pos)
        ras_R_c2w_zup.append(R_c2w)
    ras_cam_pos_zup = np.array(ras_cam_pos_zup)
    ras_R_c2w_zup = np.array(ras_R_c2w_zup)

    print(f"  RAS: {len(ext_files)} 帧 (Z-UP 坐标系)")
    print(f"  cam[0] (Z-UP): {ras_cam_pos_zup[0]}")

    # ============================================================
    # 2. 转换 RAS 相机到 Y-UP (与 GLB 一致)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: RAS Z-UP → Y-UP (与 GLB 坐标系一致)")
    print("=" * 60)

    ras_cam_pos_yup = (ZUP_TO_YUP @ ras_cam_pos_zup.T).T
    ras_R_c2w_yup = np.array([ZUP_TO_YUP @ R @ ZUP_TO_YUP.T for R in ras_R_c2w_zup])

    print(f"  cam[0] (Y-UP): {ras_cam_pos_yup[0]}")
    print(f"  R_c2w[0] (Y-UP) ≈ I: {np.allclose(ras_R_c2w_yup[0], np.eye(3), atol=1e-2)}")

    # ============================================================
    # 3. 加载 HaWoR 第一帧相机
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: 加载 HaWoR 相机位姿")
    print("=" * 60)

    hawor_data = dict(np.load(args.hawor_reconstruction, allow_pickle=True))
    hawor_cam_pos = hawor_data['t_c2w']
    hawor_R_c2w = hawor_data['R_c2w']

    R_c2w_hawor = hawor_R_c2w[0]
    t_c2w_hawor = hawor_cam_pos[0]
    print(f"  HaWoR: {len(hawor_cam_pos)} 帧")
    print(f"  cam[0]: {t_c2w_hawor}")
    print(f"  R_c2w[0]:\n{R_c2w_hawor}")

    # ============================================================
    # 4. 第一帧相机位姿对齐
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: 第一帧相机位姿对齐 (RAS Y-UP → HaWoR)")
    print("=" * 60)
    print("  同一个物理相机, RAS 用 OpenCV 约定, HaWoR 用 OpenGL 约定")
    print("  R_align = R_c2w_hawor @ OPENCV_TO_OPENGL @ R_c2w_ras_yup.T")
    print("  t_align = t_c2w_hawor - R_align @ t_c2w_ras_yup")

    R_c2w_ras = ras_R_c2w_yup[0]
    t_c2w_ras = ras_cam_pos_yup[0]

    R_align = R_c2w_hawor @ OPENCV_TO_OPENGL @ R_c2w_ras.T
    t_align = t_c2w_hawor - R_align @ t_c2w_ras

    angle = np.degrees(np.arccos(np.clip((np.trace(R_align) - 1) / 2, -1, 1)))
    print(f"\n  R_align:\n{R_align}")
    print(f"  t_align: {t_align}")
    print(f"  旋转角度: {angle:.2f}° (接近0°说明方向对齐正确)")

    det = np.linalg.det(R_align)
    if abs(det - 1.0) > 0.01:
        print(f"  ⚠ R_align 行列式 = {det:.6f} (应为1.0), 可能不是合法旋转矩阵!")
    orth_err = np.linalg.norm(R_align.T @ R_align - np.eye(3))
    if orth_err > 0.01:
        print(f"  ⚠ R_align 正交性误差 = {orth_err:.6f} (应接近0)")

    # ============================================================
    # 5. 尺度校正 (Umeyama)
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 5: 尺度校正 (Umeyama)")
    print("=" * 60)

    n_ras = len(ras_cam_pos_yup)
    n_hawor = len(hawor_cam_pos)
    common_frames = []
    for ri in range(n_ras):
        hi = round(ri * (n_hawor - 1) / (n_ras - 1)) if n_ras > 1 else 0
        common_frames.append((ri, hi))

    ras_cam_in_hawor = (R_align @ ras_cam_pos_yup.T).T + t_align

    src_pts = np.array([hawor_cam_pos[hi] for _, hi in common_frames])
    dst_pts = np.array([ras_cam_in_hawor[ri] for ri, _ in common_frames])

    src_centered = src_pts - src_pts.mean(axis=0)
    dst_centered = dst_pts - dst_pts.mean(axis=0)
    sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))

    STATIC_SIGMA_THRESHOLD = 0.01
    is_static_camera = sigma_src < STATIC_SIGMA_THRESHOLD or sigma_dst < STATIC_SIGMA_THRESHOLD

    scale_ratio = umeyama_scale(src_pts, dst_pts)
    s_inv = 1.0 / scale_ratio

    if is_static_camera:
        print(f"  ⚠ 检测到近似静态相机 (sigma_src={sigma_src:.6f}, sigma_dst={sigma_dst:.6f})")
        print(f"  Umeyama 尺度不可靠, 尝试基于手-GLB距离估算 ...")

        pred_trans_check = hawor_data['pred_trans']
        pred_valid_check = hawor_data['pred_valid']
        hand_idx_check = 0 if pred_valid_check[0].any() else 1
        valid_frames_check = np.where(pred_valid_check[hand_idx_check])[0]

        ras_glb_path_check = os.path.join(args.ras_output, 'final_scene.glb')
        if len(valid_frames_check) > 0 and os.path.exists(ras_glb_path_check):
            try:
                import trimesh
                glb_check = trimesh.load(ras_glb_path_check)
                glb_verts_list = []
                for _, geom in glb_check.geometry.items():
                    if len(geom.vertices) > 0:
                        glb_verts_list.append(geom.vertices)
                if glb_verts_list:
                    glb_verts = np.vstack(glb_verts_list)
                    glb_center_ras = glb_verts.mean(axis=0)
                    hand_mean_hawor = pred_trans_check[hand_idx_check, valid_frames_check].mean(axis=0)
                    hand_mean_ras_render = R_x @ hand_mean_hawor

                    glb_center_in_hawor_unscaled = (R_align @ glb_center_ras.T).T + t_align
                    dist_unscaled = np.linalg.norm(hand_mean_hawor - glb_center_in_hawor_unscaled)
                    dist_expected = 0.15

                    if dist_unscaled > 1e-6:
                        s_inv_heuristic = dist_expected / dist_unscaled
                        print(f"  手-GLB距离(未缩放): {dist_unscaled:.4f}m, 期望约: {dist_expected}m")
                        print(f"  启发式 s_inv = {s_inv_heuristic:.6f} (覆盖 Umeyama 值 {s_inv:.6f})")
                        s_inv = s_inv_heuristic
                        scale_ratio = 1.0 / s_inv
                    else:
                        print(f"  手-GLB距离过小 ({dist_unscaled:.6f}m), 无法启发式估算, 使用 Umeyama 值")
            except Exception as e:
                print(f"  启发式估算失败: {e}, 使用 Umeyama 值")
        else:
            print(f"  无有效手部帧或 GLB 文件, 无法启发式估算, 使用 Umeyama 值")

    if args.force_scale is not None:
        print(f"  ⚠ 使用强制尺度: s_inv = {args.force_scale:.6f} (覆盖计算值 {s_inv:.6f})")
        s_inv = args.force_scale
        scale_ratio = 1.0 / s_inv

    if s_inv <= 0:
        raise ValueError(f"s_inv = {s_inv:.6f} 为负数或零, 尺度校正失败! 请检查数据或使用 --force_scale")
    if s_inv > 10.0 or s_inv < 0.01:
        print(f"  ⚠ s_inv = {s_inv:.6f} 异常 (正常范围 0.01~10.0), 尺度可能不准确!")

    print(f"  RAS/HaWoR 尺度比: {scale_ratio:.6f}")
    print(f"  s_inv (RAS→HaWoR 缩放): {s_inv:.6f}")

    t_align_scaled = t_c2w_hawor - R_align @ (s_inv * t_c2w_ras)
    print(f"  t_align_scaled: {t_align_scaled}")

    # ============================================================
    # 6. 验证
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 6: 验证对齐")
    print("=" * 60)

    pred_trans = hawor_data['pred_trans']
    pred_valid = hawor_data['pred_valid']
    hand_idx = 0 if pred_valid[0].any() else 1
    hand_label = "左手" if hand_idx == 0 else "右手"
    valid_frames = np.where(pred_valid[hand_idx])[0]

    hand_mean = pred_trans[hand_idx, valid_frames].mean(axis=0)
    hand_sapien = RXWORLD_TO_SAPIEN @ hand_mean
    print(f"  {hand_label}均值 (HaWoR): {hand_mean}")
    print(f"  {hand_label}均值 (SAPIEN): {hand_sapien}")

    ras_glb_path = os.path.join(args.ras_output, 'final_scene.glb')
    try:
        import trimesh
        scene = trimesh.load(ras_glb_path)
        all_verts = []
        for name, geom in scene.geometry.items():
            all_verts.append(geom.vertices)
        all_verts = np.vstack(all_verts)

        glb_hawor = s_inv * (R_align @ all_verts.T).T + t_align_scaled
        glb_center_hawor = glb_hawor.mean(axis=0)
        glb_center_sapien = RXWORLD_TO_SAPIEN @ glb_center_hawor

        print(f"\n  GLB中心 (HaWoR): {glb_center_hawor}")
        print(f"  GLB中心 (SAPIEN): {glb_center_sapien}")

        dist = np.linalg.norm(hand_mean - glb_center_hawor)
        print(f"  {hand_label} → GLB中心距离: {dist:.4f}m")

        from scipy.spatial import cKDTree
        tree = cKDTree(glb_hawor)
        dists, _ = tree.query(pred_trans[hand_idx, valid_frames])
        print(f"  {hand_label} → GLB最近顶点: min={dists.min():.4f}m, mean={dists.mean():.4f}m")

        cam_pos_sapien = RXWORLD_TO_SAPIEN @ t_c2w_hawor
        cam_R_sapien = RXWORLD_TO_SAPIEN @ R_c2w_hawor
        forward = -cam_R_sapien[:, 2]

        glb_sapien = (RXWORLD_TO_SAPIEN @ glb_hawor.T).T
        cam_to_glb = glb_sapien.mean(axis=0) - cam_pos_sapien
        cam_to_glb_norm = cam_to_glb / np.linalg.norm(cam_to_glb)
        dot = np.dot(cam_to_glb_norm, forward)
        print(f"\n  相机→GLB 方向点积: {dot:.4f} (正=前方可见)")

        cam_to_hand = hand_sapien - cam_pos_sapien
        cam_to_hand_norm = cam_to_hand / np.linalg.norm(cam_to_hand)
        dot_hand = np.dot(cam_to_hand_norm, forward)
        print(f"  相机→手  方向点积: {dot_hand:.4f} (正=前方可见)")
    except ImportError:
        print("  trimesh 未安装, 跳过 GLB 验证")

    # ============================================================
    # 7. 保存变换参数
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 7: 保存变换参数")
    print("=" * 60)

    R_inv = R_align
    t_inv = t_align_scaled

    params_path = os.path.join(args.output_dir, 'transform_params.npz')
    np.savez(params_path,
             scale=scale_ratio,
             R=R_align,
             t=t_align,
             s_inv=s_inv,
             R_inv=R_inv,
             t_inv=t_inv,
             R_align=R_align,
             t_align=t_align,
             t_align_scaled=t_align_scaled,
             R_c2w_ras_yup=R_c2w_ras,
             t_c2w_ras_yup=t_c2w_ras,
             R_c2w_hawor=R_c2w_hawor,
             t_c2w_hawor=t_c2w_hawor)
    print(f"  保存到: {params_path}")
    print(f"  s_inv = {s_inv:.6f} (RAS→HaWoR 缩放)")
    print(f"  R_inv (GLB Y-UP → HaWoR):")
    print(f"    = R_c2w_hawor @ OPENCV_TO_OPENGL @ R_c2w_ras_yup.T")
    print(f"  R_inv:\n{R_inv}")
    print(f"  t_inv: {t_inv}")

    # ============================================================
    # 8. 报告
    # ============================================================
    report_path = os.path.join(args.output_dir, 'alignment_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RAS GLB -> HaWoR 对齐报告 (第一帧相机位姿 + Umeyama尺度)\n")
        f.write("=" * 60 + "\n\n")
        f.write("坐标系:\n")
        f.write("  RAS 外参: Z-UP (房间对齐后)\n")
        f.write("  RAS GLB:  Y-UP (导出时 z-up → y-up)\n")
        f.write("  HaWoR:    render world (含 R_x 变换)\n\n")
        f.write("方法: 第一帧相机位姿对齐 + Umeyama尺度校正\n")
        f.write(f"R_align 旋转角度: {angle:.2f}°\n")
        f.write(f"尺度比 (RAS/HaWoR): {scale_ratio:.6f}\n")
        f.write(f"s_inv (RAS→HaWoR): {s_inv:.6f}\n\n")
        f.write(f"R_c2w_ras_yup (Y-UP):\n{R_c2w_ras}\n\n")
        f.write(f"t_c2w_ras_yup: {t_c2w_ras}\n\n")
        f.write(f"R_c2w_hawor:\n{R_c2w_hawor}\n\n")
        f.write(f"t_c2w_hawor: {t_c2w_hawor}\n\n")
        f.write(f"变换 (GLB Y-UP → HaWoR):\n")
        f.write(f"  p_hawor = s_inv * R_inv @ p_glb + t_inv\n")
        f.write(f"  R_inv = R_c2w_hawor @ OPENCV_TO_OPENGL @ R_c2w_ras_yup.T\n")
        f.write(f"  R_inv =\n{R_inv}\n")
        f.write(f"  t_inv = {t_inv}\n")
    print(f"  报告: {report_path}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"  在 02_render_scene.py 中使用 load_glb_transformed() 加载原始 GLB")
    print(f"  变换参数: {params_path}")


if __name__ == "__main__":
    main()
