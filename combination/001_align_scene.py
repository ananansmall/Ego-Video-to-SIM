"""
001_align_scene.py — 以第一帧相机坐标系为桥对齐 GLB ↔ HaWoR

核心原理:
  RAS 和 HaWoR 处理同一个视频, 第一帧相机在两个系统中描述同一个物理相机。
  以第一帧相机坐标系为公共桥梁, 把 GLB 场景和 HaWoR 手部数据对齐到同一坐标系。

  GLB 保持原始坐标系不变 (不检测 up axis, 不做 ZUP_TO_YUP 转换)。
  最终输出把 HaWoR 手部数据映射到 GLB 原始坐标系, 供仿真直接使用。

坐标系约定 (经验验证):
  RAS 相机 (R_c2w_ras):  OpenCV 约定 (X=right, Y=down, Z=forward), 相机→GLB 原始坐标系
  HaWoR 相机 (R_c2w_hawor): 同样为 OpenCV 约定 (X=right, Y=down, Z=forward), 相机→HaWoR hand world
  实验数据: 在各自第一帧相机空间对比, 不加 R_X 差异 ~0.01m, 加 R_X 差异 ~0.13m
  因此 R_X = I, 无需 OpenGL↔OpenCV 转换

变换链 (HaWoR hand world → GLB 原始坐标系):
  Step 1: p_hawor → 第一帧 HaWoR 相机坐标系 (OpenCV 约定)
          p_cam_h = R_c2w_hawor[0].T @ (p_hawor - t_c2w_hawor[0])
  Step 2: 第一帧 HaWoR 相机坐标系 → GLB 原始坐标系 (用 RAS 第一帧)
          p_glb = scale_ratio * R_c2w_ras[0] @ p_cam_h + t_c2w_ras[0]

  合并:
    R_hand_to_glb = R_c2w_ras[0] @ R_c2w_hawor[0].T
    t_hand_to_glb = t_c2w_ras[0] - scale_ratio * R_hand_to_glb @ t_c2w_hawor[0]
    p_glb = scale_ratio * R_hand_to_glb @ p_hawor + t_hand_to_glb

尺度 (scale_ratio = GLB_length / HaWoR_length):
  把 RAS 和 HaWoR 相机轨迹都映射到第一帧各自相机坐标系, 用 Umeyama 算尺度比。
  静态相机时 (轨迹标准差过小) 用手-GLB 距离启发式估算。

  ras_cam_in_cam0[i] = R_c2w_ras[0].T @ (t_c2w_ras[i] - t_c2w_ras[0])
  hawor_cam_in_cam0[i] = R_c2w_hawor[0].T @ (t_c2w_hawor[i] - t_c2w_hawor[0])

  scale_ratio = sigma_ras / sigma_hawor

用法:
    python 001_align_scene.py \\
        --ras_output /path/to/ras_output \\
        --hawor_reconstruction /path/to/hawor_results_*.npz \\
        --output_dir ./output/alignment
"""

import argparse
import os
import numpy as np
from glob import glob

R_X = np.eye(3)  # 经验验证: HaWoR 和 RAS 的相机坐标系约定一致 (OpenCV), 无需 R_X 转换
Rx_hand = np.diag([1, -1, -1])  # 手部 OpenGL→OpenCV: 翻转 Y/Z, 使手从相机后方转到前方


def umeyama_scale(src_pts, dst_pts):
    """Umeyama 尺度估计: sigma_dst / sigma_src

    src_pts: (N,3) HaWoR 相机在第 0 帧相机坐标系下 (HaWoR 单位)
    dst_pts: (N,3) RAS 相机在第 0 帧相机坐标系下 (GLB 单位)
    返回: scale_ratio = GLB / HaWoR
    """
    src_centered = src_pts - src_pts.mean(axis=0)
    dst_centered = dst_pts - dst_pts.mean(axis=0)
    sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
    if sigma_src < 1e-8:
        return 1.0
    return sigma_dst / sigma_src


def _svd_align_cam0(src_pts, dst_pts):
    """SVD 计算 cam0 空间残差旋转 (Umeyama 无尺度, 全部帧居中)

    src_pts: (N,3) HaWoR 相机在第 0 帧相机坐标系下
    dst_pts: (N,3) RAS 相机在第 0 帧相机坐标系下
    返回: R_cam_align (3,3) —— src→dst 的旋转
    """
    src_c = src_pts - src_pts.mean(axis=0)
    dst_c = dst_pts - dst_pts.mean(axis=0)
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def _kabsch_displacement_align(src_pts, dst_pts, n_frames):
    """Kabsch 位移向量对齐: 用前 n_frames 帧的位移 (相对于第0帧) 找旋转

    核心思想 (用户提出):
      第一帧相机朝向对齐了, 但世界坐标系没对齐。
      前两帧的位移向量在不同坐标系下方向不同, 对齐位移向量就固定了世界坐标系旋转。

    使用前 n_frames 帧的位移向量 (cam0[i] - cam0[0]) 作为约束,
    多个不同方向的位移向量共同确定完整的 3D 旋转。

    src_pts: (N,3) HaWoR cam0 空间点
    dst_pts: (N,3) RAS cam0 空间点
    n_frames: 使用前几帧的位移向量
       n_frames=2: 只用1个位移向量 (2帧) — 旋转绕位移轴无约束
       n_frames=3: 用2个位移向量 (3帧) — 可确定完整3D旋转
       n_frames=-1: 用全部帧 (等效 _svd_align_cam0)
    返回: R_cam_align (3,3)
    """
    if n_frames == -1:
        n_frames = len(src_pts)

    K = min(n_frames, len(src_pts))
    if K < 2:
        return np.eye(3)

    # 位移向量 = pts[i] - pts[0] (i=1..K-1)
    src_disps = np.array([src_pts[i] - src_pts[0] for i in range(1, K)])
    dst_disps = np.array([dst_pts[i] - dst_pts[0] for i in range(1, K)])

    # 排除零位移 (第一帧几乎没动)
    valid = np.linalg.norm(src_disps, axis=1) > 1e-6
    if valid.sum() < 2:
        return np.eye(3)

    src_v = src_disps[valid]
    dst_v = dst_disps[valid]

    # Kabsch
    src_c = src_v - src_v.mean(axis=0)
    dst_c = dst_v - dst_v.mean(axis=0)
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def _load_ras_extrinsics(ras_output):
    """加载 RAS 外参, 返回 R_c2w (N,3,3) 和 t_c2w (N,3) (相机→GLB 原始坐标系)"""
    ext_dir = os.path.join(ras_output, 'extrinsics')
    ext_files = sorted(glob(os.path.join(ext_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    if not ext_files:
        raise FileNotFoundError(f"未找到 RAS 外参: {ext_dir}/")
    R_c2w_list = []
    t_c2w_list = []
    for f in ext_files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_w2c = ext[:3, :3]
        t_w2c = ext[:3, 3]
        R_c2w = R_w2c.T
        cam_pos = -R_c2w @ t_w2c
        R_c2w_list.append(R_c2w)
        t_c2w_list.append(cam_pos)
    return np.array(R_c2w_list), np.array(t_c2w_list)


def _load_hawor(hawor_reconstruction):
    """加载 HaWoR 重建 npz, 返回 dict"""
    return dict(np.load(hawor_reconstruction, allow_pickle=True))





def compute_and_save_transform_params(ras_output, hawor_reconstruction, output_dir, force_scale=None, svd_align=False, kabsch_frames=0):
    """以第一帧相机坐标系为桥对齐 GLB ↔ HaWoR 并保存变换参数

    可被其他脚本直接调用, 无需通过命令行。

    Args:
        ras_output: RAS 输出目录路径 (内含 extrinsics/ 和 final_scene.glb)
        hawor_reconstruction: HaWoR 重建 npz 文件路径 (内含 R_c2w, t_c2w 等)
        output_dir: 输出目录 (transform_params.npz 保存位置)
        force_scale: 强制尺度因子 scale_ratio (None=Umeyama 自动计算)
        svd_align: 启用 SVD 对齐 (全部帧居中, 默认关闭)
        kabsch_frames: 位移向量对齐帧数
            0 = 禁用 (默认)
            2 = 前2帧(1个位移向量)
            3 = 前3帧(2个位移向量)
            -1 = 全部帧
            当 kabsch_frames > 0 或 == -1 时, svd_align 被忽略

    Returns:
        str: 保存的 transform_params.npz 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    R_c2w_ras_all, t_c2w_ras_all = _load_ras_extrinsics(ras_output)
    hawor_data = _load_hawor(hawor_reconstruction)
    R_c2w_hawor_all = hawor_data['R_c2w']
    t_c2w_hawor_all = hawor_data['t_c2w']

    R_c2w_ras0 = R_c2w_ras_all[0]
    t_c2w_ras0 = t_c2w_ras_all[0]
    R_c2w_hawor0 = R_c2w_hawor_all[0]
    t_c2w_hawor0 = t_c2w_hawor_all[0]

    # 核心旋转: HaWoR hand world (SLAM) → GLB 原始坐标系
    R_hand_to_glb = R_c2w_ras0 @ R_X @ R_c2w_hawor0.T

    # 尺度: 把相机轨迹映射到第 0 帧相机坐标系 (OpenCV 约定)
    ras_cam_in_cam0 = np.array([
        R_c2w_ras0.T @ (t_c2w_ras_all[i] - t_c2w_ras0)
        for i in range(len(t_c2w_ras_all))
    ])
    hawor_cam_in_cam0 = np.array([
        R_X @ R_c2w_hawor0.T @ (t_c2w_hawor_all[i] - t_c2w_hawor0)
        for i in range(len(t_c2w_hawor_all))
    ])

    n_ras = len(ras_cam_in_cam0)
    n_hawor = len(hawor_cam_in_cam0)
    common_frames = []
    for ri in range(n_ras):
        hi = round(ri * (n_hawor - 1) / (n_ras - 1)) if n_ras > 1 else 0
        common_frames.append((ri, hi))

    src_pts = np.array([hawor_cam_in_cam0[hi] for _, hi in common_frames])
    dst_pts = np.array([ras_cam_in_cam0[ri] for ri, _ in common_frames])

    # 位移向量对齐 (由 kabsch_frames 控制)
    # 0 = 禁用, 2 = 前2帧(1个位移向量), 3 = 前3帧(2个位移向量), -1 = 全部帧
    if kabsch_frames > 0 or kabsch_frames == -1:
        R_cam_align = _kabsch_displacement_align(src_pts, dst_pts, kabsch_frames)
        R_hand_to_glb = R_c2w_ras0 @ R_cam_align @ R_X @ R_c2w_hawor0.T
        angle_kabsch = np.degrees(np.arccos(np.clip((np.trace(R_cam_align)-1)/2, -1, 1)))
        print(f"  位移向量对齐 (前{kabsch_frames if kabsch_frames>0 else '全部'}帧): "
              f"R_cam_align 旋转角度 = {angle_kabsch:.2f}°")
    elif svd_align:
        R_cam_align = _svd_align_cam0(src_pts, dst_pts)
        R_hand_to_glb = R_c2w_ras0 @ R_cam_align @ R_X @ R_c2w_hawor0.T
        print(f"  SVD 对齐 (全部帧居中): R_cam_align 旋转角度 = "
              f"{np.degrees(np.arccos(np.clip((np.trace(R_cam_align)-1)/2, -1, 1))):.2f}°")
    else:
        R_cam_align = np.eye(3)

    scale_ratio = umeyama_scale(src_pts, dst_pts)

    if force_scale is not None:
        print(f"  使用强制尺度: scale_ratio = {force_scale:.6f}")
        scale_ratio = force_scale

    # 核心平移 (带尺度)
    t_hand_to_glb = t_c2w_ras0 - scale_ratio * R_hand_to_glb @ t_c2w_hawor0

    # 保存
    R_render_to_glb = R_hand_to_glb @ R_X
    params_path = os.path.join(output_dir, 'transform_params.npz')
    np.savez(params_path,
             scale_ratio=scale_ratio,
             R_hand_to_glb=R_hand_to_glb,
             R_render_to_glb=R_render_to_glb,
             t_hand_to_glb=t_hand_to_glb,
             R_c2w_ras0=R_c2w_ras0,
             t_c2w_ras0=t_c2w_ras0,
             R_c2w_hawor0=R_c2w_hawor0,
             t_c2w_hawor0=t_c2w_hawor0,
             s_inv=1.0 / scale_ratio,
             R_inv=R_hand_to_glb,
             t_inv=t_hand_to_glb,
             R_align=R_hand_to_glb,
             t_align=t_c2w_ras0 - 1.0 * R_hand_to_glb @ t_c2w_hawor0,
             t_align_scaled=t_hand_to_glb,
             Rx_hand=Rx_hand,
             R_hand_corrected=R_hand_to_glb @ Rx_hand)
    print(f"  transform_params 保存到: {params_path} (scale_ratio={scale_ratio:.6f})")
    return params_path


def main():
    parser = argparse.ArgumentParser(
        description="以第一帧相机坐标系为桥对齐 GLB ↔ HaWoR"
    )
    parser.add_argument("--ras_output", type=str, required=True,
                        help="RAS 输出目录 (内含 extrinsics/ 和 final_scene.glb)")
    parser.add_argument("--hawor_reconstruction", type=str, required=True,
                        help="HaWoR 重建 npz 文件路径")
    parser.add_argument("--output_dir", type=str, default="./output/alignment",
                        help="输出目录")
    parser.add_argument("--force_scale", type=float, default=None,
                        help="强制尺度因子 scale_ratio (None=自动计算)")
    parser.add_argument("--svd_align", action="store_true",
                        help="启用 SVD 对齐 (全部帧cam0空间居中, 默认关闭)")
    parser.add_argument("--kabsch_frames", type=int, default=0,
                        help="位移向量对齐帧数: 0=禁用(默认), 2=前2帧, 3=前3帧, -1=全部帧")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: 加载 RAS
    print("=" * 60)
    print("Step 1: 加载 RAS 相机位姿 (GLB 原始坐标系)")
    print("=" * 60)

    R_c2w_ras_all, t_c2w_ras_all = _load_ras_extrinsics(args.ras_output)
    R_c2w_ras0 = R_c2w_ras_all[0]
    t_c2w_ras0 = t_c2w_ras_all[0]
    print(f"  RAS: {len(t_c2w_ras_all)} 帧")
    print(f"  R_c2w[0]:\n{R_c2w_ras0}")
    print(f"  t_c2w[0]: {t_c2w_ras0}")

    # Step 2: 加载 HaWoR
    print("\n" + "=" * 60)
    print("Step 2: 加载 HaWoR 相机位姿 (Y-UP)")
    print("=" * 60)

    hawor_data = _load_hawor(args.hawor_reconstruction)
    R_c2w_hawor_all = hawor_data['R_c2w']
    t_c2w_hawor_all = hawor_data['t_c2w']
    R_c2w_hawor0 = R_c2w_hawor_all[0]
    t_c2w_hawor0 = t_c2w_hawor_all[0]
    print(f"  HaWoR: {len(t_c2w_hawor_all)} 帧")
    print(f"  R_c2w[0]:\n{R_c2w_hawor0}")
    print(f"  t_c2w[0]: {t_c2w_hawor0}")

    # Step 3: 计算变换
    print("\n" + "=" * 60)
    print("Step 3: 计算 HaWoR → GLB 原始坐标系 变换")
    print("=" * 60)

    R_hand_to_glb = R_c2w_ras0 @ R_X @ R_c2w_hawor0.T
    angle = np.degrees(np.arccos(np.clip((np.trace(R_hand_to_glb) - 1) / 2, -1, 1)))

    # 尺度
    ras_cam_in_cam0 = np.array([
        R_c2w_ras0.T @ (t_c2w_ras_all[i] - t_c2w_ras0)
        for i in range(len(t_c2w_ras_all))
    ])
    hawor_cam_in_cam0 = np.array([
        R_X @ R_c2w_hawor0.T @ (t_c2w_hawor_all[i] - t_c2w_hawor0)
        for i in range(len(t_c2w_hawor_all))
    ])
    n_ras, n_hawor = len(ras_cam_in_cam0), len(hawor_cam_in_cam0)
    hi_ras = np.linspace(0, n_hawor - 1, n_ras, dtype=int)
    src_pts = hawor_cam_in_cam0[hi_ras]
    dst_pts = ras_cam_in_cam0

    # 位移向量对齐: 用前 K 帧的位移 (相对于第0帧) 找世界坐标系旋转
    # 核心思想: 第一帧对齐相机朝向, 前两帧位移对齐世界坐标系
    if args.kabsch_frames != 0 or args.svd_align:
        if args.kabsch_frames != 0:
            n_frames = args.kabsch_frames
            R_cam_align = _kabsch_displacement_align(src_pts, dst_pts, n_frames)
            label = f"位移向量对齐 (前{'全部' if n_frames==-1 else str(n_frames)}帧)"
        else:
            R_cam_align = _svd_align_cam0(src_pts, dst_pts)
            label = "SVD 对齐 (全部帧居中)"
        R_hand_to_glb = R_c2w_ras0 @ R_cam_align @ R_X @ R_c2w_hawor0.T
        angle_align = np.degrees(np.arccos(np.clip((np.trace(R_cam_align)-1)/2, -1, 1)))
        print(f"  {label}: R_cam_align 旋转角度 = {angle_align:.2f}°")

    src_c = src_pts - src_pts.mean(axis=0)
    dst_c = dst_pts - dst_pts.mean(axis=0)
    sigma_src = np.sqrt(np.mean(np.sum(src_c ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_c ** 2, axis=1)))

    scale_ratio = umeyama_scale(src_pts, dst_pts)

    if args.force_scale is not None:
        scale_ratio = args.force_scale

    t_hand_to_glb = t_c2w_ras0 - scale_ratio * R_hand_to_glb @ t_c2w_hawor0
    R_render_to_glb = R_hand_to_glb @ R_X

    print(f"  R_hand_to_glb = R_c2w_ras[0] @ R_x @ R_c2w_hawor[0].T")
    print(f"  R_hand_to_glb:\n{R_hand_to_glb}")
    print(f"  旋转角度: {angle:.2f}°")
    print(f"  scale_ratio (GLB/HaWoR): {scale_ratio:.6f}")
    print(f"  t_hand_to_glb: {t_hand_to_glb}")

    # 保存
    params_path = os.path.join(args.output_dir, 'transform_params.npz')
    np.savez(params_path,
             scale_ratio=scale_ratio,
             R_hand_to_glb=R_hand_to_glb,
             R_render_to_glb=R_render_to_glb,
             t_hand_to_glb=t_hand_to_glb,
             R_c2w_ras0=R_c2w_ras0,
             t_c2w_ras0=t_c2w_ras0,
             R_c2w_hawor0=R_c2w_hawor0,
             t_c2w_hawor0=t_c2w_hawor0,
             s_inv=1.0 / scale_ratio,
             R_inv=R_hand_to_glb,
             t_inv=t_hand_to_glb,
             R_align=R_hand_to_glb,
             t_align=t_c2w_ras0 - 1.0 * R_hand_to_glb @ t_c2w_hawor0,
             t_align_scaled=t_hand_to_glb,
             Rx_hand=Rx_hand,
             R_hand_corrected=R_hand_to_glb @ Rx_hand)

    # 报告
    report_path = os.path.join(args.output_dir, 'alignment_report.txt')
    with open(report_path, 'w') as f:
        f.write("GLB ↔ HaWoR 对齐报告 (第一帧相机坐标系为桥)\n\n")
        f.write(f"方法: 第一帧相机坐标系对齐 + Umeyama 尺度校正\n")
        f.write(f"GLB 保持原始坐标系不变\n\n")
        f.write(f"RAS 帧数: {n_ras}\n")
        f.write(f"HaWoR 帧数: {n_hawor}\n")
        f.write(f"RAS sigma (GLB 单位): {sigma_dst:.6f}\n")
        f.write(f"HaWoR sigma (HaWoR 单位): {sigma_src:.6f}\n")
        f.write(f"R_c2w_ras[0]:\n{R_c2w_ras0}\n\n")
        f.write(f"t_c2w_ras[0]: {t_c2w_ras0}\n\n")
        f.write(f"R_c2w_hawor[0]:\n{R_c2w_hawor0}\n\n")
        f.write(f"t_c2w_hawor[0]: {t_c2w_hawor0}\n\n")
        f.write(f"R_hand_to_glb 旋转角度: {angle:.2f}°\n")
        f.write(f"scale_ratio (GLB/HaWoR): {scale_ratio:.6f}\n\n")
        f.write("变换 (HaWoR → GLB 原始坐标系):\n")
        f.write("  p_glb = scale_ratio * R_hand_to_glb @ p_hawor + t_hand_to_glb\n")
        f.write("  R_hand_to_glb = R_c2w_ras[0] @ R_x @ R_c2w_hawor[0].T\n")
        f.write(f"  R_hand_to_glb =\n{R_hand_to_glb}\n")
        f.write(f"  t_hand_to_glb = {t_hand_to_glb}\n")

    print(f"  保存到: {params_path}")
    print(f"  报告: {report_path}")
    print("\n完成!")


if __name__ == "__main__":
    main()
