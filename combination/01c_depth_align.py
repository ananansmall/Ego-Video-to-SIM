#!/usr/bin/env python3
"""
01c_depth_align.py — 在 GLB 空间中用 RAS 深度图校正 HaWoR 手部深度

管线位置: 001_align_scene.py → 01c_depth_align.py → 002_render_scene.py

核心改动 (v2):
  01c 在 001 之后运行，使用 001 的 scale_ratio 统一量纲。
  深度比较在 GLB 空间（经 001 变换后）进行，而非在 HaWoR 相机坐标系。

算法:
  1. 加载 001 的 transform_params (scale_ratio, R_hand_to_glb, t_hand_to_glb)
  2. 加载 RAS 深度图 + 内参 + 外参
  3. 对每帧:
     a. HaWoR 手腕 → 001 变换 → GLB 空间
     b. 通过 RAS 外参投影到 RAS 深度图 → 读取手部 mask 区域深度
     c. RAS 深度图反投影到 GLB 空间 → 得到 RAS 认为的手腕 GLB 位置
     d. 用 GLB 空间中的深度差异修正 HaWoR 的 pred_trans
  4. 输出: 只改 pred_trans，其他 MANO 参数不变

输出文件 (简洁):
  - hawor_results_*_depth_aligned.npz  ← 深度校正后的完整 reconstruction
  不再单独输出 _factors.npz

用法:
    # 完整管线 (推荐)
    python 001_align_scene.py \\
        --ras_output /path/to/ras_output \\
        --hawor_reconstruction /path/to/hawor_results_*.npz \\
        --output_dir ./output/alignment

    python 01c_depth_align.py \\
        --hawor-dir /path/to/hawor \\
        --ras-dir /path/to/ras_output \\
        --transform-params ./output/alignment/transform_params.npz

    # 或显式指定文件
    python 01c_depth_align.py \\
        --hawor-reconstruction /path/to/hawor_results_*.npz \\
        --hawor-masks /path/to/model_masks.npy \\
        --ras-dir /path/to/ras_output \\
        --transform-params ./output/alignment/transform_params.npz \\
        --output /path/to/hawor_results_depth_aligned.npz
"""

import argparse
import os
import numpy as np
import cv2
from glob import glob
from pathlib import Path


# ── 文件查找 ──

def find_reconstruction_file(hawor_dir):
    """在 HaWoR 目录中查找重建结果 npz (排除 depth_aligned)"""
    rec_dir = Path(hawor_dir) / "reconstruction"
    if not rec_dir.exists():
        return None
    for f in sorted(rec_dir.glob("hawor_results_*.npz")):
        if "_depth_aligned" not in str(f):
            return str(f)
    return None


def find_model_masks(hawor_dir):
    """查找 HaWoR model_masks.npy"""
    hawor_path = Path(hawor_dir)
    for d in sorted(hawor_path.glob("tracks_*")):
        mask_file = d / "model_masks.npy"
        if mask_file.exists():
            return str(mask_file)
    return None


# ── RAS 数据加载 ──

def load_ras_depth_dir(ras_dir):
    """加载 RAS 深度图目录，返回排序后的文件列表"""
    depth_dir = os.path.join(ras_dir, 'depth')
    if not os.path.isdir(depth_dir):
        raise FileNotFoundError(f"RAS 深度图目录不存在: {depth_dir}")
    depth_files = sorted(
        glob(os.path.join(depth_dir, '*.png')),
        key=lambda x: int(os.path.basename(x).split('.')[0])
    )
    if not depth_files:
        raise FileNotFoundError(f"深度图目录为空: {depth_dir}")
    return depth_files


def load_ras_intrinsics(ras_dir):
    """加载 RAS 相机内参，返回 K (3,3) 或 None"""
    intrinsic_file = os.path.join(ras_dir, 'intrinsic.txt')
    if not os.path.exists(intrinsic_file):
        print(f"  ⚠ 未找到内参文件 {intrinsic_file}")
        return None
    K = np.loadtxt(intrinsic_file)
    if K.shape == (3, 4):
        K = K[:, :3]
    elif K.size == 9:
        K = K.reshape(3, 3)
    if K.shape != (3, 3):
        print(f"  ⚠ 内参矩阵形状异常 {K.shape}")
        return None
    return K


def load_ras_extrinsics(ras_dir):
    """加载 RAS 外参，返回 R_c2w (N,3,3) 和 t_c2w (N,3)"""
    ext_dir = os.path.join(ras_dir, 'extrinsics')
    ext_files = sorted(glob(os.path.join(ext_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    if not ext_files:
        raise FileNotFoundError(f"未找到 RAS 外参: {ext_dir}/")
    R_c2w_list, t_c2w_list = [], []
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


# ── 深度图操作 ──

def z_to_euclidean_depth(depth_img, K, depth_scale=1000.0):
    """将深度图 z 值转换为相机射线欧氏距离

    对每个像素 (u, v):
      euclidean = (z / scale) * sqrt(((u-cx)/fx)² + ((v-cy)/fy)² + 1)
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth_img.shape
    v_grid, u_grid = np.mgrid[0:H, 0:W].astype(np.float32)
    ray_x = (u_grid - cx) / fx
    ray_y = (v_grid - cy) / fy
    scale_factor = np.sqrt(ray_x ** 2 + ray_y ** 2 + 1.0)
    z_m = depth_img.astype(np.float32) / depth_scale
    return z_m * scale_factor


def extract_hand_depth_ras(depth_img, mask, depth_scale=1000.0, K=None):
    """从 RAS 深度图提取手部区域的中位数深度 (米)

    Args:
        depth_img: (H, W) uint16 深度图
        mask: (H, W) uint8/bool 手部 mask
        depth_scale: 像素→米转换因子
        K: (3,3) 相机内参，用于 z→欧氏距离转换

    Returns:
        float: 手部中位数深度 (米) 或 None
    """
    if mask.shape[:2] != depth_img.shape[:2]:
        mask_uint8 = mask.astype(np.uint8) if mask.dtype == bool else mask
        mask = cv2.resize(mask_uint8, (depth_img.shape[1], depth_img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    if K is not None:
        depth_img = z_to_euclidean_depth(depth_img, K, depth_scale)
        depth_scale = 1.0

    # dilation 扩大采样区域
    mask_dilated = cv2.dilate(mask.astype(np.uint8),
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                              iterations=1)

    # 原始 mask 深度
    hand_pixels_raw = np.where(mask > 0)
    if len(hand_pixels_raw[0]) == 0:
        return None
    raw_depths = depth_img[hand_pixels_raw]
    raw_valid = raw_depths[raw_depths > 0]
    if len(raw_valid) == 0:
        return None
    raw_median = np.median(raw_valid)

    # dilated mask 深度
    hand_pixels = np.where(mask_dilated > 0)
    if len(hand_pixels[0]) == 0:
        return None
    hand_depth_values = depth_img[hand_pixels]
    valid_depth = hand_depth_values[hand_depth_values > 0]
    if len(valid_depth) == 0:
        return None
    dilated_median = np.median(valid_depth)

    # dilated 混入背景噪声时回退
    if raw_median > 1e-6 and abs(dilated_median - raw_median) / raw_median > 0.15:
        return float(raw_median) / depth_scale

    return float(dilated_median) / depth_scale


def depth_map_to_glb_point(u, v, z_depth, K, R_c2w, t_c2w, depth_scale=1000.0):
    """将深度图像素 (u, v, z) 反投影到 GLB 空间

    Args:
        u, v: 像素坐标
        z_depth: 深度图像素值 (uint16)
        K: (3,3) 内参
        R_c2w: (3,3) 相机→GLB 旋转
        t_c2w: (3,) 相机→GLB 平移
        depth_scale: 像素→米

    Returns:
        (3,) GLB 空间中的 3D 点
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z_m = z_depth / depth_scale
    # 像素 → 相机坐标
    x_cam = (u - cx) * z_m / fx
    y_cam = (v - cy) * z_m / fy
    p_cam = np.array([x_cam, y_cam, z_m])
    # 相机 → GLB
    p_glb = R_c2w @ p_cam + t_c2w
    return p_glb


# ── 核心算法 ──

def compute_depth_correction_glb(hawor_data, masks, depth_files, K, R_c2w_ras, t_c2w_ras,
                                  scale_ratio, R_hand_to_glb, t_hand_to_glb, hand_idx,
                                  depth_scale=1000.0, Rx_hand=None, mode="forward"):
    """在 GLB 空间中计算深度校正

    对每个有对应 RAS 深度图的 HaWoR 帧:
      1. HaWoR 手腕 → 001 变换 → GLB 空间 → RAS 外参 → 相机坐标系 → 深度
      2. RAS 深度图手部 mask → 中位数深度 → 反投影到 GLB 空间
      3. 比较: HaWoR 手腕在 RAS 相机下的 z 深度 vs RAS 深度图手部 z 深度
      4. 校正因子 cf = depth_ras_z / depth_hawor_z (统一在 RAS 相机坐标系下比较)

    Args:
        hawor_data: dict, HaWoR reconstruction
        masks: (N_hawor, H, W) 手部 mask
        depth_files: list, RAS 深度图文件
        K: (3,3) RAS 内参
        R_c2w_ras: (N_ras, 3, 3) RAS 相机→GLB 旋转
        t_c2w_ras: (N_ras, 3) RAS 相机→GLB 平移
        scale_ratio: float, 001 的 GLB/HaWoR 尺度比
        R_hand_to_glb: (3,3) HaWoR→GLB 旋转
        t_hand_to_glb: (3,) HaWoR→GLB 平移
        hand_idx: 手部索引
        depth_scale: 深度图像素→米

    Returns:
        (correction_factors, stats)
    """
    pred_trans = hawor_data['pred_trans']
    pred_valid = hawor_data['pred_valid']
    R_c2w_hawor = hawor_data['R_c2w']
    t_c2w_hawor = hawor_data['t_c2w']

    n_hawor = pred_trans.shape[1]
    n_ras = len(depth_files)

    # 帧配对
    if n_ras == 1:
        pairs = [(0, 0)]
    else:
        pairs = [
            (ri, round(ri * (n_hawor - 1) / (n_ras - 1)))
            for ri in range(n_ras)
        ]

    # 手部翻转: 根据 mode 决定是否使用 Rx_hand
    if mode == "noflip":
        R_hand = R_hand_to_glb
        if Rx_hand is not None:
            print(f"  [mode=noflip] 忽略 Rx_hand, 使用 R_hand = R_hand_to_glb")
    else:
        if Rx_hand is None:
            Rx_hand = np.diag([1, -1, -1])
        R_hand = R_hand_to_glb @ Rx_hand

    raw_factors = {}
    for ras_idx, hawor_idx in pairs:
        if not pred_valid[hand_idx, hawor_idx]:
            continue
        if np.any(np.isnan(pred_trans[hand_idx, hawor_idx])):
            continue

        # ── 方法: 在 RAS 相机坐标系下比较 z 深度 ──

        # Step A: HaWoR 手腕 → GLB 空间 (手部用 R_hand = R_hand_to_glb @ Rx_hand)
        p_hawor = pred_trans[hand_idx, hawor_idx]
        p_glb_hawor = scale_ratio * R_hand @ p_hawor + t_hand_to_glb

        # Step B: GLB → RAS 相机坐标系 (取 z 分量 = 深度)
        p_ras_cam = R_c2w_ras[ras_idx].T @ (p_glb_hawor - t_c2w_ras[ras_idx])
        depth_hawor_z = p_ras_cam[2]  # 沿光轴的 z 深度

        if abs(depth_hawor_z) < 1e-6:
            continue

        # Step C: RAS 深度图手部区域中位数 z 深度 (不转欧氏距离，直接比 z)
        depth_img = cv2.imread(depth_files[ras_idx], cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            continue

        mask = masks[hawor_idx]
        depth_ras_z = extract_hand_depth_ras(depth_img, mask, depth_scale=depth_scale, K=None)
        # K=None: 只取 z 深度，不转欧氏距离，和 p_ras_cam[2] 量纲一致

        if depth_ras_z is None or depth_ras_z < 1e-6:
            continue

        # Step D: 校正因子
        # 注意符号: OpenCV 相机坐标系 z 朝前为正
        # depth_ras_z > 0 (深度图存储正值)
        # depth_hawor_z 可正可负，取绝对值
        if mode == "reverse":
            cf = abs(depth_hawor_z) / depth_ras_z  # 反方向: HaWoR/RAS
        else:
            cf = depth_ras_z / abs(depth_hawor_z)  # 默认方向: RAS/HaWoR

        raw_factors[hawor_idx] = cf

        print(f"  RAS帧{ras_idx}→HaWoR帧{hawor_idx}: "
              f"hawor_glb_z={depth_hawor_z:+.4f}m, ras_z={depth_ras_z:.4f}m, cf={cf:.4f}")

    if not raw_factors:
        print("  ⚠ 无有效校正帧, 不进行深度校正")
        return np.ones(n_hawor), {"count": 0, "avg_cf": 1.0}

    # 插值到所有帧
    sorted_indices = sorted(raw_factors.keys())
    correction_factors = np.ones(n_hawor)

    if len(sorted_indices) == 1:
        correction_factors[:] = raw_factors[sorted_indices[0]]
    else:
        for fi in range(n_hawor):
            if fi in raw_factors:
                correction_factors[fi] = raw_factors[fi]
            else:
                if fi < sorted_indices[0]:
                    correction_factors[fi] = raw_factors[sorted_indices[0]]
                elif fi > sorted_indices[-1]:
                    correction_factors[fi] = raw_factors[sorted_indices[-1]]
                else:
                    left_idx = max(i for i in sorted_indices if i <= fi)
                    right_idx = min(i for i in sorted_indices if i >= fi)
                    if left_idx == right_idx:
                        correction_factors[fi] = raw_factors[left_idx]
                    else:
                        t = (fi - left_idx) / (right_idx - left_idx)
                        correction_factors[fi] = (
                            raw_factors[left_idx] * (1 - t) + raw_factors[right_idx] * t
                        )

    # 5 帧移动平均平滑
    if n_hawor > 5:
        kernel = np.ones(5) / 5
        padded = np.pad(correction_factors, 2, mode='edge')
        correction_factors = np.convolve(padded, kernel, mode='valid')[:n_hawor]

    avg_cf = np.mean(list(raw_factors.values()))
    stats = {"count": len(raw_factors), "avg_cf": avg_cf}

    return correction_factors, stats


def apply_depth_correction_glb(hawor_data, correction_factors, hand_idx,
                                scale_ratio, R_hand_to_glb, t_hand_to_glb,
                                R_c2w_ras, t_c2w_ras, n_ras, Rx_hand=None, mode="forward"):
    """在 GLB 空间中应用深度校正

    对每帧:
      1. HaWoR 手腕 → GLB 空间
      2. GLB → RAS 相机坐标系
      3. 保持 x,y 不变，z 方向用 cf 修正
      4. RAS 相机坐标系 → GLB
      5. GLB → HaWoR 空间 (逆变换)
      6. 更新 pred_trans

    只改 pred_trans，其他参数不变。
    """
    if mode == "noflip":
        R_hand = R_hand_to_glb
    else:
        if Rx_hand is None:
            Rx_hand = np.diag([1, -1, -1])
        R_hand = R_hand_to_glb @ Rx_hand

    pred_trans = hawor_data['pred_trans']
    pred_valid = hawor_data['pred_valid']
    n_frames = pred_trans.shape[1]

    corrected_pred_trans = pred_trans.copy()

    # 逆变换: GLB → HaWoR (手部用 R_hand)
    R_glb_to_hand = R_hand.T
    s_inv = 1.0 / scale_ratio

    for fi in range(n_frames):
        if not pred_valid[hand_idx, fi]:
            continue

        cf = correction_factors[fi]
        if abs(cf - 1.0) < 1e-6:
            continue

        # HaWoR → GLB (手部用 R_hand)
        p_hawor = pred_trans[hand_idx, fi]
        p_glb = scale_ratio * R_hand @ p_hawor + t_hand_to_glb

        # 确定对应的 RAS 帧
        if n_ras == 1:
            ras_idx = 0
        else:
            ras_idx = round(fi * (n_ras - 1) / (n_frames - 1))
        ras_idx = min(ras_idx, n_ras - 1)

        # GLB → RAS 相机坐标系
        p_ras_cam = R_c2w_ras[ras_idx].T @ (p_glb - t_c2w_ras[ras_idx])

        # 在 RAS 相机坐标系中校正: 保持 x,y 不变, z 乘 cf
        p_ras_cam_corrected = p_ras_cam.copy()
        p_ras_cam_corrected[2] = abs(p_ras_cam[2]) * cf * (1 if p_ras_cam[2] >= 0 else -1)

        # RAS 相机坐标系 → GLB
        p_glb_corrected = R_c2w_ras[ras_idx] @ p_ras_cam_corrected + t_c2w_ras[ras_idx]

        # GLB → HaWoR (逆变换)
        p_hawor_corrected = s_inv * R_glb_to_hand @ (p_glb_corrected - t_hand_to_glb)

        corrected_pred_trans[hand_idx, fi] = p_hawor_corrected

    return corrected_pred_trans


# ── 主函数 ──

def main():
    parser = argparse.ArgumentParser(
        description="在 GLB 空间中用 RAS 深度图校正 HaWoR 手部深度 (001 之后运行)")
    parser.add_argument("--hawor-reconstruction", type=str, default=None,
                        help="HaWoR reconstruction npz (和 --hawor-dir 二选一)")
    parser.add_argument("--hawor-masks", type=str, default=None,
                        help="HaWoR model_masks.npy (和 --hawor-dir 二选一)")
    parser.add_argument("--hawor-dir", type=str, default=None,
                        help="HaWoR 输出目录 (自动查找 reconstruction npz 和 model_masks)")
    parser.add_argument("--ras-dir", type=str, required=True,
                        help="RAS 输出目录 (含 depth/, extrinsics/, intrinsic.txt)")
    parser.add_argument("--transform-params", type=str, required=True,
                        help="001 输出的 transform_params.npz (必须先跑 001)")
    parser.add_argument("--output", type=str, default=None,
                        help="输出 npz 路径 (默认: 原文件名加 _depth_aligned 后缀)")
    parser.add_argument("--hand-idx", type=int, default=-1,
                        help="手部索引: 0=左手, 1=右手, -1=自动检测")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                        help="深度图像素→米转换因子 (默认 1000, mm→m)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算校正因子, 不保存文件")
    parser.add_argument("--mode", type=str, default="forward",
                        choices=["forward", "reverse", "noflip"],
                        help="校正模式: forward=当前方向, reverse=反方向, noflip=去掉Rx_hand翻转")
    args = parser.parse_args()

    # ── 确定输入文件 ──
    hawor_dir = None
    if args.hawor_dir:
        hawor_dir = args.hawor_dir
        if args.hawor_reconstruction is None:
            rec_file = find_reconstruction_file(hawor_dir)
            if rec_file is None:
                raise FileNotFoundError(
                    f"在 {hawor_dir}/reconstruction/ 中未找到 hawor_results_*.npz "
                    f"(排除 _depth_aligned)")
            args.hawor_reconstruction = rec_file
        if args.hawor_masks is None:
            mask_file = find_model_masks(hawor_dir)
            if mask_file is None:
                raise FileNotFoundError(f"在 {hawor_dir}/tracks_*/ 中未找到 model_masks.npy")
            args.hawor_masks = mask_file

    if args.hawor_reconstruction is None:
        parser.error("必须指定 --hawor-reconstruction 或 --hawor-dir")

    print("=" * 60)
    print("深度校正 v2: RAS 深度图 + GLB 空间 → HaWoR 手部深度")
    print("管线: 001 → 01c → 002")
    print("=" * 60)

    # ── 1. 加载 001 的 transform_params ──
    print("\n[1/5] 加载 001 transform_params ...")
    tp = dict(np.load(args.transform_params, allow_pickle=True))
    # 兼容 001 (scale_ratio) 和 01 (scale) 两种格式
    if 'scale_ratio' in tp:
        scale_ratio = float(tp['scale_ratio'])
    elif 'scale' in tp:
        scale_ratio = float(tp['scale'])
    else:
        raise KeyError("transform_params 中未找到 scale_ratio 或 scale")
    # 兼容两种 key 名
    if 'R_hand_to_glb' in tp:
        R_hand_to_glb = tp['R_hand_to_glb']
    elif 'R_align' in tp:
        R_hand_to_glb = tp['R_align']
    else:
        R_hand_to_glb = tp.get('R', tp.get('R_inv'))
    if 't_hand_to_glb' in tp:
        t_hand_to_glb = tp['t_hand_to_glb']
    elif 't_align_scaled' in tp:
        t_hand_to_glb = tp['t_align_scaled']
    else:
        t_hand_to_glb = tp.get('t_align', tp.get('t', tp.get('t_inv')))
    print(f"  scale_ratio = {scale_ratio:.6f}")
    print(f"  R_hand_to_glb:\n{R_hand_to_glb}")
    print(f"  t_hand_to_glb: {t_hand_to_glb}")

    # ── 2. 加载 HaWoR 数据 ──
    print("\n[2/5] 加载 HaWoR 数据 ...")
    hawor_data = dict(np.load(args.hawor_reconstruction, allow_pickle=True))
    pred_trans = hawor_data['pred_trans']
    pred_valid = hawor_data['pred_valid']
    n_frames = pred_trans.shape[1]
    print(f"  HaWoR: {n_frames} 帧")
    print(f"  输入: {args.hawor_reconstruction}")

    # ── 3. 加载 RAS 数据 (提前加载 mask 用于手部选择) ──
    print("\n[3/5] 加载 RAS 深度图 + 内参 + 外参 ...")
    masks = np.load(args.hawor_masks, allow_pickle=True)
    if masks.ndim == 0:
        masks = masks.item()
    if masks.ndim > 3:
        if masks.shape[0] == 1:
            masks = masks[0]
        else:
            print(f"  ⚠ mask 维度异常: {masks.shape}, 取第0帧")
            masks = masks[0] if isinstance(masks, np.ndarray) else masks
    print(f"  Masks: shape={masks.shape}")

    # 手部选择: 选有效非NaN帧最多的手 (同时考虑 mask 是否有像素)
    if args.hand_idx < 0:
        best_hand, best_count = 0, 0
        for h in range(pred_trans.shape[0]):
            count = 0
            for fi in range(n_frames):
                if pred_valid[h, fi] and not np.any(np.isnan(pred_trans[h, fi])):
                    if fi < masks.shape[0] and np.any(masks[fi]):
                        count += 1
            print(f"  手{['左','右'][h]}: {count} 有效帧")
            if count > best_count:
                best_count, best_hand = count, h
        hand_idx = best_hand
    else:
        hand_idx = args.hand_idx
    hand_label = "左手" if hand_idx == 0 else "右手"
    print(f"  校正手: {hand_label} (idx={hand_idx})")

    depth_files = load_ras_depth_dir(args.ras_dir)
    n_ras = len(depth_files)
    first_depth = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
    print(f"  RAS 深度图: {n_ras} 帧, {first_depth.shape[1]}x{first_depth.shape[0]}")

    K = load_ras_intrinsics(args.ras_dir)
    if K is not None:
        print(f"  RAS 内参: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.0f}, cy={K[1,2]:.0f}")
    else:
        print(f"  ⚠ 未加载内参 (深度校正不需要内参，直接比 z)")

    R_c2w_ras, t_c2w_ras = load_ras_extrinsics(args.ras_dir)
    print(f"  RAS 外参: {len(t_c2w_ras)} 帧")

    # ── 4. 计算校正因子 ──
    print(f"\n[4/5] 在 GLB 空间计算逐帧校正因子 ({n_ras} RAS帧 → {n_frames} HaWoR帧) ...")
    print(f"  模式: {args.mode}")
    correction_factors, stats = compute_depth_correction_glb(
        hawor_data, masks, depth_files, K, R_c2w_ras, t_c2w_ras,
        scale_ratio, R_hand_to_glb, t_hand_to_glb, hand_idx,
        depth_scale=args.depth_scale,
        Rx_hand=np.diag([1, -1, -1]) if args.mode != "noflip" else None,
        mode=args.mode
    )

    if stats['count'] == 0:
        print("  ⚠ 无有效校正数据, 不进行深度校正")
        return

    print(f"\n  有效校正帧: {stats['count']}")
    print(f"  平均校正因子: {stats['avg_cf']:.4f}")
    print(f"  校正因子范围: [{correction_factors.min():.4f}, {correction_factors.max():.4f}]")

    # ── 5. 应用校正 + 保存 ──
    print("\n[5/5] 应用深度校正 ...")
    Rx_hand = np.diag([1, -1, -1])
    R_hand = R_hand_to_glb @ Rx_hand
    if args.dry_run:
        print("  [dry-run] 不保存文件")
        for fi in range(min(10, n_frames)):
            if pred_valid[hand_idx, fi]:
                cf = correction_factors[fi]
                p_hawor = pred_trans[hand_idx, fi]
                p_glb = scale_ratio * R_hand @ p_hawor + t_hand_to_glb
                print(f"    帧{fi}: cf={cf:.4f}  hawor_glb=({p_glb[0]:.4f}, {p_glb[1]:.4f}, {p_glb[2]:.4f})")
        return

    corrected_pred_trans = apply_depth_correction_glb(
        hawor_data, correction_factors, hand_idx,
        scale_ratio, R_hand_to_glb, t_hand_to_glb,
        R_c2w_ras, t_c2w_ras, n_ras,
        Rx_hand=np.diag([1, -1, -1]) if args.mode != "noflip" else None,
        mode=args.mode
    )

    # 保存: 只输出一个 _depth_aligned.npz
    if args.output:
        output_path = args.output
    else:
        base = args.hawor_reconstruction
        output_path = base.replace('.npz', '_depth_aligned.npz')
        if output_path == base:
            output_path = base.replace('.npz', '') + '_depth_aligned.npz'

    hawor_data['pred_trans'] = corrected_pred_trans
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True) \
        if os.path.dirname(output_path) else None
    np.savez(output_path, **hawor_data)

    # 报告
    print(f"\n{'=' * 60}")
    print("深度校正完成!")
    print(f"{'=' * 60}")
    print(f"  输入: {args.hawor_reconstruction}")
    print(f"  输出: {output_path}")
    print(f"  scale_ratio (001): {scale_ratio:.6f}")
    print(f"  有效校正帧: {stats['count']}/{n_ras}")
    print(f"  平均校正因子: {stats['avg_cf']:.4f}")
    print(f"  校正因子范围: [{correction_factors.min():.4f}, {correction_factors.max():.4f}]")
    print()
    print("后续步骤:")
    print(f"  python 002_render_scene.py \\")
    print(f"      --hawor-dir {hawor_dir or '...'} \\")
    print(f"      --ras-dir {args.ras_dir} \\")
    print(f"      --transform-params {args.transform_params}")


if __name__ == "__main__":
    main()
