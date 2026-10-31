"""
2D-3D Correspondence-Based Pose Refinement for Stage 4.

Core idea (MASt3R-style):
  Instead of comparing depth values (which leads to scale adjustment),
  we establish 2D pixel-level correspondences between mesh and VGGT:

  1. 3D->2D: Project mesh vertices to 2D pixel coordinates
  2. 2D->3D: At the same pixel, look up VGGT world_points to get ground-truth 3D
  3. The mesh_3D and VGGT_3D at the same pixel should correspond to the same surface point
  4. Use these 3D-3D correspondences to estimate a RIGID transformation (R, t only, NO scale)

This approach:
  - Preserves mesh size (no scale adjustment)
  - Uses 2D-consistent correspondences (same pixel = same surface point)
  - Directly adjusts position based on VGGT 3D reconstruction
  - Is robust to initial pose errors
"""

import numpy as np
from scipy.spatial.transform import Rotation
from stage4.renderer import MeshRenderer
from stage4.umeyama import umeyama_alignment, umeyama_alignment_ransac, decompose_similarity_transform


def project_world_to_pixel(pts_world, extrinsic, intrinsic):
    """
    3D->2D: Project world points to pixel coordinates.

    Uses VGGT convention (same as project_3D_points_np):
      X_cam = extrinsic @ X_world
      uv = X_cam[:2] / X_cam[2]
      pixel = K @ [uv, 1]

    No FLIP matrix needed - consistent with VGGT internal projection.py.

    Args:
        pts_world: (N, 3) world coordinates
        extrinsic: (4, 4) VGGT camera extrinsic
        intrinsic: (3, 3) camera intrinsic

    Returns:
        u: (N,) pixel x coordinates (invalid = -1)
        v: (N,) pixel y coordinates (invalid = -1)
        valid: (N,) bool mask for valid projections
    """
    pts_hom = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_cam = (extrinsic @ pts_hom.T).T[:, :3]

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    valid = pts_cam[:, 2] > 1e-6

    u = np.full(len(pts_world), -1.0)
    v = np.full(len(pts_world), -1.0)

    u[valid] = fx * pts_cam[valid, 0] / pts_cam[valid, 2] + cx
    v[valid] = fy * pts_cam[valid, 1] / pts_cam[valid, 2] + cy

    return u, v, valid


def unproject_depth_to_world(depth_map, extrinsic, intrinsic):
    """
    2D->3D: Unproject depth map to world coordinates (vectorized).

    Uses the SAME convention as VGGT's depth_to_world_coords_points:
      x_cam = (u - cx) * depth / fx
      y_cam = (v - cy) * depth / fy
      z_cam = depth
      cam_to_world = inv(extrinsic)
      world = cam_to_world @ cam

    No FLIP matrix needed - consistent with VGGT internal geometry.py.

    Args:
        depth_map: (H, W) depth in meters
        extrinsic: (4, 4) VGGT camera extrinsic
        intrinsic: (3, 3) camera intrinsic

    Returns:
        world_points: (H, W, 3) world coordinates
    """
    H, W = depth_map.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))

    valid = depth_map > 1e-6

    x_cam = np.where(valid, (u_grid - cx) * depth_map / fx, 0)
    y_cam = np.where(valid, (v_grid - cy) * depth_map / fy, 0)
    z_cam = np.where(valid, depth_map, 0)

    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(depth_map)], axis=-1)

    c2w = np.linalg.inv(extrinsic)

    pts_world = (c2w @ pts_cam.reshape(-1, 4).T).T[:, :3]
    world_points = pts_world.reshape(H, W, 3)

    return world_points


def establish_2d3d_correspondences(mesh, T, world_points, world_points_conf,
                                    depths, extrinsics, intrinsic, renderer,
                                    sample_frames=None, depth_tolerance=0.3,
                                    min_correspondences=20):
    """
    Establish 2D-3D correspondences between mesh and VGGT via pixel-level matching.

    For each camera view:
      1. Render mesh -> get depth_ren, mask_ren
      2. For pixels in mask_ren where VGGT depth is also valid:
         - mesh_3d = unproject(depth_ren[pixel])  (mesh surface point in world)
         - vggt_3d = world_points[frame][pixel]    (VGGT 3D point at same pixel)
      3. These are 3D-3D correspondences established via 2D pixel overlap

    Args:
        mesh: trimesh.Trimesh object
        T: (4, 4) current transformation matrix
        world_points: (T, H, W, 3) VGGT world coordinates
        world_points_conf: (T, H, W) confidence or None
        depths: (T, H, W) VGGT depth maps
        extrinsics: (T, 4, 4) camera extrinsics
        intrinsic: (3, 3) camera intrinsic
        renderer: MeshRenderer instance
        sample_frames: list of frame indices to use
        depth_tolerance: max depth difference for valid correspondence
        min_correspondences: minimum correspondences required

    Returns:
        mesh_pts: (M, 3) mesh surface points in world coordinates
        vggt_pts: (M, 3) VGGT 3D points at corresponding pixels
    """
    num_frames = len(extrinsics)
    H, W = depths.shape[1], depths.shape[2]

    if sample_frames is None:
        sample_frames = np.linspace(0, num_frames - 1, min(6, num_frames), dtype=int).tolist()

    mesh_pts_all = []
    vggt_pts_all = []

    for f in sample_frames:
        _, depth_ren, mask_ren = renderer.render_mesh(mesh, T, extrinsics[f])
        depth_real = depths[f]

        valid_pixel = mask_ren & (depth_real > 0) & (depth_ren > 0)

        if valid_pixel.sum() < 10:
            continue

        depth_diff = np.abs(depth_ren - depth_real)
        close_pixel = valid_pixel & (depth_diff < depth_tolerance)

        if close_pixel.sum() < 10:
            close_pixel = valid_pixel & (depth_diff < depth_tolerance * 2)
        if close_pixel.sum() < 5:
            continue

        vggt_pts_frame = world_points[f][close_pixel]
        valid_3d = ~np.isnan(vggt_pts_frame).any(axis=1)
        vggt_pts_frame = vggt_pts_frame[valid_3d]

        if world_points_conf is not None:
            conf = world_points_conf[f][close_pixel][valid_3d]
            if len(conf) > 0 and conf.max() > 0:
                keep = conf >= np.percentile(conf[conf > 0], 20)
                vggt_pts_frame = vggt_pts_frame[keep]

        if len(vggt_pts_frame) < 5:
            continue

        vy, vx = np.where(close_pixel)
        vy = vy[valid_3d]
        vx = vx[valid_3d]
        if world_points_conf is not None and len(conf) > 0 and conf.max() > 0:
            vy = vy[keep]
            vx = vx[keep]

        # Get mesh 3D points by back-projecting rendered depth (论文方法 Step 2)
        # P_ren = π⁻¹(q_j, D_ren,v(q_j); K, T_v)
        # pyrender 返回的 depth 数值上等价于 OpenCV z-forward 深度，可直接反投影
        world_points_ren = unproject_depth_to_world(depth_ren, extrinsics[f], intrinsic)
        mesh_pts_frame = world_points_ren[vy, vx]
        valid_mesh = depth_ren[vy, vx] > 1e-6

        if valid_mesh.sum() < 5:
            continue

        mesh_pts_frame = mesh_pts_frame[valid_mesh]
        vggt_pts_matched = vggt_pts_frame[valid_mesh] if len(vggt_pts_frame) == len(valid_mesh) else vggt_pts_frame[:valid_mesh.sum()]

        if len(mesh_pts_frame) > 3000:
            idx = np.random.choice(len(mesh_pts_frame), 3000, replace=False)
            mesh_pts_frame = mesh_pts_frame[idx]
            vggt_pts_matched = vggt_pts_matched[idx]

        mesh_pts_all.append(mesh_pts_frame)
        vggt_pts_all.append(vggt_pts_matched)

    if len(mesh_pts_all) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    mesh_pts = np.concatenate(mesh_pts_all, axis=0)
    vggt_pts = np.concatenate(vggt_pts_all, axis=0)

    if len(mesh_pts) > 15000:
        idx = np.random.choice(len(mesh_pts), 15000, replace=False)
        mesh_pts = mesh_pts[idx]
        vggt_pts = vggt_pts[idx]

    return mesh_pts, vggt_pts


def establish_vertex_correspondences(mesh, T, world_points, world_points_conf,
                                      depths, extrinsics, intrinsic,
                                      sample_frames=None, depth_tolerance=0.3):
    """
    Establish vertex-level 2D-3D correspondences.

    For each mesh vertex:
      1. Transform to world: pt_world = T @ pt_local
      2. Project to 2D: (u, v) = project(pt_world, ext, int)
      3. Lookup VGGT 3D: vggt_pt = world_points[frame][v, u]
      4. If depth consistent: (pt_world, vggt_pt) is a correspondence pair

    Args:
        mesh: trimesh.Trimesh object
        T: (4, 4) current transformation matrix
        world_points: (T, H, W, 3) VGGT world coordinates
        world_points_conf: (T, H, W) confidence or None
        depths: (T, H, W) VGGT depth maps
        extrinsics: (T, 4, 4) camera extrinsics
        intrinsic: (3, 3) camera intrinsic
        sample_frames: list of frame indices
        depth_tolerance: max depth difference

    Returns:
        mesh_pts: (M, 3) mesh vertex positions in world coordinates
        vggt_pts: (M, 3) VGGT 3D points at corresponding pixels
    """
    num_frames = len(extrinsics)
    H, W = depths.shape[1], depths.shape[2]

    if sample_frames is None:
        sample_frames = np.linspace(0, num_frames - 1, min(5, num_frames), dtype=int).tolist()

    verts_local = mesh.vertices
    ones = np.ones((len(verts_local), 1))
    verts_hom = np.hstack([verts_local, ones])
    verts_world = (T @ verts_hom.T).T[:, :3]

    mesh_pts_all = []
    vggt_pts_all = []

    for f in sample_frames:
        u, v, proj_valid = project_world_to_pixel(verts_world, extrinsics[f], intrinsic)

        in_image = proj_valid & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)

        ui = np.clip(np.round(u).astype(int), 0, W - 1)
        vi = np.clip(np.round(v).astype(int), 0, H - 1)

        depth_real = depths[f]
        has_depth = np.zeros(len(verts_world), dtype=bool)
        depth_at_pixel = np.zeros(len(verts_world))
        for i in range(len(verts_world)):
            if in_image[i]:
                d = depth_real[vi[i], ui[i]]
                if d > 1e-6:
                    has_depth[i] = True
                    depth_at_pixel[i] = d

        c2w = np.linalg.inv(extrinsics[f])
        pts_cam = (extrinsics[f] @ verts_hom.T).T[:, :3]
        mesh_depth = pts_cam[:, 2]

        depth_consistent = has_depth & (np.abs(mesh_depth - depth_at_pixel) < depth_tolerance)

        if depth_consistent.sum() < 5:
            depth_consistent = has_depth & (np.abs(mesh_depth - depth_at_pixel) < depth_tolerance * 2)
        if depth_consistent.sum() < 5:
            continue

        idx = np.where(depth_consistent)[0]

        vggt_pts_frame = world_points[f][vi[idx], ui[idx]]
        valid_3d = ~np.isnan(vggt_pts_frame).any(axis=1)
        idx = idx[valid_3d]
        vggt_pts_frame = vggt_pts_frame[valid_3d]

        if world_points_conf is not None:
            conf = world_points_conf[f][vi[idx], ui[idx]]
            if len(conf) > 0 and conf.max() > 0:
                keep = conf >= np.percentile(conf[conf > 0], 20)
                idx = idx[keep]
                vggt_pts_frame = vggt_pts_frame[keep]

        mesh_pts_all.append(verts_world[idx])
        vggt_pts_all.append(vggt_pts_frame)

    if len(mesh_pts_all) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))

    mesh_pts = np.concatenate(mesh_pts_all, axis=0)
    vggt_pts = np.concatenate(vggt_pts_all, axis=0)

    return mesh_pts, vggt_pts


def estimate_rigid_transform(src_pts, dst_pts, with_scale=False):
    """
    Estimate rigid transformation from src_pts to dst_pts using Umeyama.

    Args:
        src_pts: (N, 3) source points (mesh)
        dst_pts: (N, 3) destination points (VGGT)
        with_scale: if True, estimate scale; if False, rigid only (position adjustment)

    Returns:
        T: (4, 4) transformation matrix
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    s, R, t, T = umeyama_alignment(src_pts, dst_pts, with_scale=with_scale)
    return T, s, R, t


def estimate_rigid_transform_ransac(src_pts, dst_pts, with_scale=False,
                                     inlier_threshold=0.05, max_iterations=1000,
                                     min_inliers=10):
    """
    RANSAC-robust rigid transformation estimation.

    Args:
        src_pts: (N, 3) source points
        dst_pts: (N, 3) destination points
        with_scale: if True, estimate scale; if False, rigid only
        inlier_threshold: distance threshold for inlier classification
        max_iterations: max RANSAC iterations
        min_inliers: minimum inliers to accept

    Returns:
        T: (4, 4) best transformation
        inliers: (N,) bool inlier mask
        s: scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    T, inliers, s, R, t = umeyama_alignment_ransac(
        src_pts, dst_pts, with_scale=with_scale,
        inlier_threshold=inlier_threshold,
        max_iterations=max_iterations,
        min_inliers=min_inliers,
    )
    return T, inliers, s, R, t


def compute_depth_iou(mesh, T, depths, extrinsics, renderer, sample_frames, tol=0.15):
    """Compute depth-consistent Mask IoU across views.

    For each view, the "VGGT mask" is defined as pixels where VGGT depth
    is within `tol` of the rendered depth (not just depth > 0, which would
    cover the entire image and make IoU degenerate to coverage rate).

    IoU = (mask_ren ∩ mask_close) / (mask_ren ∪ mask_close)
    where mask_close = |depth_ren - depth_vggt| / depth_vggt < tol
    """
    ious = []
    for f in sample_frames:
        _, depth_ren, mask_ren = renderer.render_mesh(mesh, T, extrinsics[f])
        depth_real = depths[f]
        overlap = mask_ren & (depth_real > 1e-6)
        if overlap.sum() == 0:
            ious.append(0.0)
            continue
        rel_diff = np.abs(depth_ren[overlap] - depth_real[overlap]) / np.maximum(depth_real[overlap], 1e-6)
        close_pixels = np.zeros_like(mask_ren)
        close_pixels[overlap] = rel_diff < tol
        intersection = (mask_ren & close_pixels).sum()
        union = (mask_ren | close_pixels).sum()
        if union == 0:
            ious.append(0.0)
        else:
            ious.append(float(intersection) / float(union))
    return np.mean(ious) if ious else 0.0


def compute_depth_accuracy(mesh, T, depths, extrinsics, renderer, sample_frames, tol=0.10):
    """Compute depth-based alignment rate (relative tolerance).

    For each rendered pixel where VGGT depth is also valid:
      accuracy = |depth_ren - depth_vggt| / depth_vggt < tol

    This uses relative tolerance so that objects at different distances
    are evaluated fairly (e.g. 10% error at 1m = 0.1m, at 0.5m = 0.05m).
    """
    accs = []
    for f in sample_frames:
        _, depth_ren, mask_ren = renderer.render_mesh(mesh, T, extrinsics[f])
        depth_real = depths[f]
        overlap = mask_ren & (depth_real > 1e-6)
        if overlap.sum() == 0:
            accs.append(0.0)
            continue
        rel_diff = np.abs(depth_ren[overlap] - depth_real[overlap]) / np.maximum(depth_real[overlap], 1e-6)
        close = rel_diff < tol
        accs.append(float(close.sum()) / float(overlap.sum()))
    return np.mean(accs) if accs else 0.0


def projection_based_alignment(mesh, initial_T, world_points, world_points_conf,
                                depths, extrinsics, intrinsic, renderer,
                                sample_frames=None, num_iterations=8,
                                with_scale=False, inlier_threshold=0.08,
                                depth_tolerance=0.3, progress_prefix=""):
    """
    Projection-based pose alignment using 2D-3D correspondences.

    Algorithm:
      For each iteration:
        1. Render mesh at current pose -> depth_ren, mask_ren
        2. At pixels where both mesh and VGGT are visible:
           - mesh_3d = unproject(depth_ren[pixel]) via camera params
           - vggt_3d = world_points[frame][pixel]
           -> pixel-level 3D-3D correspondences
        3. Estimate rigid transform (R, t) from mesh_3d -> vggt_3d
        4. Apply: T_new = T_delta @ T_current
        5. Accept only if IoU improves

    Args:
        mesh: trimesh.Trimesh object
        initial_T: (4, 4) initial transformation
        world_points: (T, H, W, 3) VGGT 3D points
        world_points_conf: (T, H, W) confidence or None
        depths: (T, H, W) VGGT depth maps
        extrinsics: (T, 4, 4) camera extrinsics
        intrinsic: (3, 3) camera intrinsic
        renderer: MeshRenderer instance
        sample_frames: frames to use
        num_iterations: number of ICP-style iterations
        with_scale: if True, estimate scale; if False, rigid only
        inlier_threshold: RANSAC inlier threshold
        depth_tolerance: max depth difference for correspondence
        progress_prefix: prefix for progress messages

    Returns:
        best_T: (4, 4) optimized transformation
        best_iou: float, best IoU achieved
    """
    num_frames = len(extrinsics)
    if sample_frames is None:
        sample_frames = np.linspace(0, num_frames - 1, min(6, num_frames), dtype=int).tolist()

    eval_frames = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)

    current_T = initial_T.copy()
    best_T = initial_T.copy()
    best_acc = compute_depth_accuracy(mesh, initial_T, depths, extrinsics, renderer, eval_frames)

    s_init, R_init, t_init = decompose_similarity_transform(initial_T)
    init_iou = compute_depth_iou(mesh, initial_T, depths, extrinsics, renderer, eval_frames)
    print(f"{progress_prefix} Initial: Acc={best_acc:.4f}, IoU={init_iou:.4f}, scale={s_init:.3f}")

    for iteration in range(num_iterations):
        progress = iteration / max(num_iterations - 1, 1)
        cur_depth_tol = depth_tolerance * (1.0 - 0.5 * progress)
        cur_inlier_thresh = inlier_threshold * (1.0 - 0.3 * progress)

        print(f"{progress_prefix} Iter {iteration+1}/{num_iterations} [{progress*100:0.0f}%]: depth_tol={cur_depth_tol:.3f}, inlier_thresh={cur_inlier_thresh:.3f}", end="")

        mesh_pts, vggt_pts = establish_2d3d_correspondences(
            mesh, current_T, world_points, world_points_conf,
            depths, extrinsics, intrinsic, renderer,
            sample_frames=sample_frames,
            depth_tolerance=cur_depth_tol,
        )

        if len(mesh_pts) < 10:
            print(f" -> too few correspondences ({len(mesh_pts)}), skipping")
            continue

        T_delta, inliers, s_d, R_d, t_d = estimate_rigid_transform_ransac(
            mesh_pts, vggt_pts,
            with_scale=with_scale,
            inlier_threshold=cur_inlier_thresh,
            max_iterations=800,
            min_inliers=max(10, len(mesh_pts) // 10),
        )

        if with_scale and (s_d < 0.7 or s_d > 1.5):
            T_delta_no_scale, _, s_ns, R_ns, t_ns = estimate_rigid_transform_ransac(
                mesh_pts[inliers], vggt_pts[inliers],
                with_scale=False,
                inlier_threshold=cur_inlier_thresh,
                max_iterations=500,
                min_inliers=10,
            )
            T_delta = T_delta_no_scale
            s_d = s_ns

        T_candidate = T_delta @ current_T

        cand_acc = compute_depth_accuracy(mesh, T_candidate, depths, extrinsics, renderer, eval_frames)
        cand_iou = compute_depth_iou(mesh, T_candidate, depths, extrinsics, renderer, eval_frames)

        inlier_ratio = inliers.sum() / len(inliers) if len(inliers) > 0 else 0
        print(f", inliers={inliers.sum()} ({inlier_ratio:.1%}), scale_delta={s_d:.3f}, Acc={cand_acc:.4f}, IoU={cand_iou:.4f}")

        if cand_acc >= best_acc * 0.99:
            current_T = T_candidate
            if cand_acc > best_acc:
                best_acc = cand_acc
                best_T = T_candidate.copy()
        else:
            current_T = best_T.copy()

    final_iou = compute_depth_iou(mesh, best_T, depths, extrinsics, renderer, eval_frames)
    return best_T, final_iou


def vertex_projection_alignment(mesh, initial_T, world_points, world_points_conf,
                                 depths, extrinsics, intrinsic, renderer,
                                 sample_frames=None, num_iterations=5,
                                 with_scale=False, inlier_threshold=0.08,
                                 depth_tolerance=0.3):
    """
    Vertex-level projection alignment.

    Similar to projection_based_alignment but uses mesh vertex projection
    instead of dense pixel-level correspondences. More precise for sparse meshes.

    Args:
        Same as projection_based_alignment

    Returns:
        best_T: (4, 4) optimized transformation
        best_iou: float, best IoU achieved
    """
    num_frames = len(extrinsics)
    if sample_frames is None:
        sample_frames = np.linspace(0, num_frames - 1, min(5, num_frames), dtype=int).tolist()

    eval_frames = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)

    current_T = initial_T.copy()
    best_T = initial_T.copy()
    best_acc = compute_depth_accuracy(mesh, initial_T, depths, extrinsics, renderer, eval_frames)

    for iteration in range(num_iterations):
        progress = iteration / max(num_iterations - 1, 1)
        cur_depth_tol = depth_tolerance * (1.0 - 0.5 * progress)
        cur_inlier_thresh = inlier_threshold * (1.0 - 0.3 * progress)

        mesh_pts, vggt_pts = establish_vertex_correspondences(
            mesh, current_T, world_points, world_points_conf,
            depths, extrinsics, intrinsic,
            sample_frames=sample_frames,
            depth_tolerance=cur_depth_tol,
        )

        if len(mesh_pts) < 10:
            print(f"      V-Iter {iteration}: too few correspondences ({len(mesh_pts)})")
            continue

        T_delta, inliers, s_d, R_d, t_d = estimate_rigid_transform_ransac(
            mesh_pts, vggt_pts,
            with_scale=with_scale,
            inlier_threshold=cur_inlier_thresh,
            max_iterations=800,
            min_inliers=max(10, len(mesh_pts) // 10),
        )

        if with_scale and (s_d < 0.7 or s_d > 1.5):
            T_delta_no_scale, _, _, _, _ = estimate_rigid_transform_ransac(
                mesh_pts[inliers], vggt_pts[inliers],
                with_scale=False,
                inlier_threshold=cur_inlier_thresh,
                max_iterations=500,
                min_inliers=10,
            )
            T_delta = T_delta_no_scale

        T_candidate = T_delta @ current_T
        cand_acc = compute_depth_accuracy(mesh, T_candidate, depths, extrinsics, renderer, eval_frames)
        cand_iou = compute_depth_iou(mesh, T_candidate, depths, extrinsics, renderer, eval_frames)

        inlier_ratio = inliers.sum() / len(inliers) if len(inliers) > 0 else 0
        print(f"      V-Iter {iteration}: corr={len(mesh_pts)}, inliers={inliers.sum()} "
              f"({inlier_ratio:.1%}), scale_delta={s_d:.3f}, Acc={cand_acc:.4f}, IoU={cand_iou:.4f}")

        if cand_acc >= best_acc * 0.99:
            current_T = T_candidate
            if cand_acc > best_acc:
                best_acc = cand_acc
                best_T = T_candidate.copy()
        else:
            current_T = best_T.copy()

    final_iou = compute_depth_iou(mesh, best_T, depths, extrinsics, renderer, eval_frames)
    return best_T, final_iou
