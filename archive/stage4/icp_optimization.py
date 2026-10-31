"""
ICP Fine-Tuning for Stage 4.

Classical Iterative Closest Point with progressive threshold tightening,
used as Phase B after the 2D-3D correspondence-based coarse alignment.

Algorithm (per iteration):
  1. Sample mesh surface points, transform to world coordinates
  2. Render mesh at current pose -> mask_ren
  3. Within mask_ren, extract VGGT 3D points (depth-close region)
  4. KDTree nearest-neighbor: VGGT points -> mesh points
  5. Umeyama + RANSAC: estimate similarity transform T_delta
  6. IoU-gated acceptance: T_candidate = T_delta @ T_current
  7. Progressive threshold tightening for convergence
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from archive.stage4.renderer import MeshRenderer
from archive.stage4.umeyama import umeyama_alignment_ransac, decompose_similarity_transform
from archive.stage4.projection_alignment import compute_depth_iou, compute_depth_accuracy


def icp_fine_tuning(mesh, current_T, world_points, world_points_conf, depths,
                    extrinsics, intrinsic, renderer, sample_frames=None,
                    num_iterations=8, inlier_ratio=0.05, progress_prefix=""):
    """
    ICP fine-tuning using VGGT 3D points as target.

    Uses Phase A result as initialization. Progressive threshold tightening
    ensures convergence from coarse to fine.

    Args:
        mesh: trimesh.Trimesh object
        current_T: (4, 4) current transformation (from Phase A)
        world_points: (T, H, W, 3) VGGT 3D points
        world_points_conf: (T, H, W) confidence or None
        depths: (T, H, W) VGGT depth maps
        extrinsics: (T, 4, 4) camera extrinsics
        intrinsic: (3, 3) camera intrinsic
        renderer: MeshRenderer instance
        sample_frames: frames to use for ICP
        num_iterations: number of ICP iterations
        inlier_ratio: RANSAC inlier ratio threshold
        progress_prefix: prefix for progress messages

    Returns:
        best_T: (4, 4) optimized transformation
        best_iou: float, best IoU achieved
    """
    num_frames = len(extrinsics)
    if sample_frames is None:
        sample_frames = np.linspace(0, num_frames - 1, min(6, num_frames), dtype=int).tolist()

    eval_frames = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)

    try:
        mesh_pts_local, _ = trimesh.sample.sample_surface(mesh, 2000)
    except Exception:
        mesh_pts_local = mesh.vertices.copy()
    ones_local = np.ones((mesh_pts_local.shape[0], 1))
    mesh_pts_hom = np.hstack([mesh_pts_local, ones_local])

    # ── 尺寸相对阈值 ──
    # 旧实现用绝对米制 (0.25→0.075m): 对 3cm 的耳机盒, 收敛阈值仍是物体
    # 尺寸的 2.5 倍, 邻近表面 (桌面) 的点全部落进对应点集合, 小物体的
    # ICP 退化为"向周围几何吸附" (实测 wxD: button_0 的 IoU 迭代无变化)。
    # 改为物体尺寸相对制: 参考量 = 世界空间最大轴跨 (下限 5cm;
    # SAM3D 局部网格被归一化到 ~1.0, 局部尺寸无意义)。大物体向后兼容:
    # 0.9m 桌最大轴 0.9 → 起始 0.9*0.28≈0.25 (与旧值一致)。
    mesh_pts_w0 = (current_T @ mesh_pts_hom.T).T[:, :3]
    ext_w0 = mesh_pts_w0.max(axis=0) - mesh_pts_w0.min(axis=0)
    size_ref = max(float(ext_w0.max()), 0.05)
    base_dist = 0.28 * size_ref
    base_depth = 0.22 * size_ref

    best_T = current_T.copy()
    best_acc = compute_depth_accuracy(mesh, current_T, depths, extrinsics, renderer, eval_frames, tol=min(max(0.28*size_ref, 0.02), 0.10))
    current_T = current_T.copy()

    s_init, _, _ = decompose_similarity_transform(current_T)
    init_iou = compute_depth_iou(mesh, current_T, depths, extrinsics, renderer, eval_frames, tol=min(max(0.28*size_ref, 0.02), 0.15))
    print(f"{progress_prefix} Initial: Acc={best_acc:.4f}, IoU={init_iou:.4f}, scale={s_init:.3f}, size_ref={size_ref:.3f}")

    for iteration in range(num_iterations):
        progress = iteration / max(num_iterations - 1, 1)
        dist_thresh = base_dist * (1.0 - 0.7 * progress)
        depth_thresh = base_depth * (1.0 - 0.5 * progress)

        print(f"{progress_prefix} Iter {iteration+1}/{num_iterations} [{progress*100:0.0f}%]: dist_thresh={dist_thresh:.3f}, depth_thresh={depth_thresh:.3f}", end="")

        mesh_pts_w = (current_T @ mesh_pts_hom.T).T[:, :3]

        vggt_pts_all = []
        mesh_pts_all = []

        for f in sample_frames:
            _, depth_ren, mask_ren = renderer.render_mesh(mesh, current_T, extrinsics[f])
            depth_real = depths[f]

            if mask_ren.sum() < 30:
                continue

            depth_close = mask_ren & (depth_real > 0) & (depth_ren > 0) & \
                          (np.abs(depth_ren - depth_real) < depth_thresh)
            if depth_close.sum() < 15:
                depth_close = mask_ren & (depth_real > 0) & (depth_ren > 0) & \
                              (np.abs(depth_ren - depth_real) < depth_thresh * 3)

            vggt_pts = world_points[f][depth_close]
            conf = world_points_conf[f][depth_close] if world_points_conf is not None else None
            if conf is not None and len(conf) > 0:
                keep = conf >= np.percentile(conf, 30)
                vggt_pts = vggt_pts[keep]
            valid = ~np.isnan(vggt_pts).any(axis=1)
            vggt_pts = vggt_pts[valid]
            if len(vggt_pts) > 3000:
                idx = np.random.choice(len(vggt_pts), 3000, replace=False)
                vggt_pts = vggt_pts[idx]

            if len(vggt_pts) < 10:
                continue

            tree = cKDTree(mesh_pts_w)
            dists, idxs = tree.query(vggt_pts)
            close = dists < dist_thresh
            if close.sum() < 5:
                continue

            vggt_pts_all.append(vggt_pts[close])
            mesh_pts_all.append(mesh_pts_w[idxs[close]])

        if len(vggt_pts_all) == 0:
            print(f" -> no correspondences, skipping")
            continue

        vggt_combined = np.concatenate(vggt_pts_all, axis=0)
        mesh_combined = np.concatenate(mesh_pts_all, axis=0)

        if len(vggt_combined) > 10000:
            idx = np.random.choice(len(vggt_combined), 10000, replace=False)
            vggt_combined = vggt_combined[idx]
            mesh_combined = mesh_combined[idx]

        print(f" -> correspondences={len(vggt_combined)}", end="")

        try:
            T_delta, inliers, s_d, R_d, t_d = umeyama_alignment_ransac(
                mesh_combined, vggt_combined, with_scale=False,
                inlier_threshold=dist_thresh * 0.5,
                max_iterations=500, min_inliers=10,
            )
        except Exception:
            print(f" -> RANSAC failed, skipping")
            continue

        if s_d < 0.5 or s_d > 2.0:
            print(f" -> scale_delta={s_d:.3f} invalid, skipping")
            continue

        T_candidate = T_delta @ current_T
        cand_acc = compute_depth_accuracy(mesh, T_candidate, depths, extrinsics, renderer, eval_frames, tol=min(max(0.28*size_ref, 0.02), 0.10))
        cand_iou = compute_depth_iou(mesh, T_candidate, depths, extrinsics, renderer, eval_frames, tol=min(max(0.28*size_ref, 0.02), 0.15))

        inlier_ratio_cur = inliers.sum() / len(inliers) if len(inliers) > 0 else 0
        print(f", inliers={inliers.sum()} ({inlier_ratio_cur:.1%}), "
              f"scale_delta={s_d:.3f}, Acc={cand_acc:.4f}, IoU={cand_iou:.4f}")

        if cand_acc >= best_acc * 0.98:
            current_T = T_candidate
            if cand_acc > best_acc:
                best_acc = cand_acc
                best_T = T_candidate.copy()
        else:
            current_T = best_T.copy()

    final_iou = compute_depth_iou(mesh, best_T, depths, extrinsics, renderer, eval_frames, tol=min(max(0.28*size_ref, 0.02), 0.15))
    return best_T, best_acc, final_iou
