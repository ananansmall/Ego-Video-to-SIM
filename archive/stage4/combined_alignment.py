"""
Stage 4: Iterative Visual-Spatial Alignment

Implements the render-match-optimize iterative alignment algorithm from the paper
(Section 3.4), inspired by the classical Iterative Closest Point (ICP) framework
but driven by visual correspondences rather than purely geometric proximity.

Paper method (Section 3.4):
  Step 1: Rendering and Matching
    - Render object using current T(i-1) -> rendered RGB + depth
    - Use MASt3R to establish dense 2D correspondences between real and rendered
  Step 2: 3D Lifting and Aggregation
    - Back-project 2D matches to 3D world coordinates using camera params
    - Aggregate across temporal neighborhood views
  Step 3: Similarity Alignment
    - Umeyama algorithm to estimate T(i) = {s, R, t} aligning P_ren to P_real
  Selection: Choose T* with max mean IoU across views

Our implementation supports two modes:
  Mode 1 (MASt3R): Uses MASt3R model for 2D matching (paper's original method)
  Mode 2 (Depth): Uses depth-consistency for 2D matching (fallback when MASt3R unavailable)

  Phase A: 2D-3D Correspondence-Based Alignment
    - MASt3R mode: Render -> MASt3R 2D match -> 3D lift -> Umeyama
    - Depth mode:  Render -> depth pixel match -> 3D lift -> Umeyama

  Phase B: ICP Fine-Tuning (classical 3D nearest-neighbor)
    - Uses Phase A result as initialization
    - KDTree nearest-neighbor correspondences within rendered mask
    - Progressive threshold tightening for convergence
    - Umeyama + RANSAC, IoU-gated acceptance
"""

import numpy as np
import trimesh
from archive.stage4.renderer import MeshRenderer
from archive.stage4.umeyama import decompose_similarity_transform, umeyama_alignment_ransac
from archive.stage4.projection_alignment import (
    projection_based_alignment,
    compute_depth_iou,
    compute_depth_accuracy,
    unproject_depth_to_world,
)
from archive.stage4.icp_optimization import icp_fine_tuning


def _try_import_mast3r():
    """Try to import MASt3R, return True if available."""
    import os
    import sys
    mast3r_root = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), 'mast3r')
    mast3r_root = os.path.normpath(mast3r_root)
    if mast3r_root not in sys.path:
        sys.path.insert(0, mast3r_root)
    try:
        import mast3r
        return True
    except ImportError:
        return False


def _mast3r_phase_a(mesh, current_T, world_points, world_points_conf,
                     depths, extrinsics, intrinsic, colors, renderer,
                     sample_frames, num_iterations, with_scale, device='cuda',
                     progress_prefix=""):
    """
    Phase A using MASt3R model (paper's original method).

    For each iteration:
      1. Render mesh at current pose -> rendered RGB + depth
      2. Run MASt3R between real RGB and rendered RGB -> 2D correspondences
      3. Lift 2D correspondences to 3D via depth unprojection
      4. Umeyama + RANSAC to estimate rigid transform
      5. IoU-gated acceptance
    """
    from archive.stage4.mast3r_matcher import MASt3RMatcher

    num_frames = len(extrinsics)
    eval_frames = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)

    best_T = current_T.copy()
    _msz = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    _tol_acc = min(max(0.3 * _msz, 0.02), 0.10)
    _tol_iou = min(max(0.3 * _msz, 0.02), 0.15)
    best_score = compute_depth_accuracy(mesh, current_T, depths, extrinsics, renderer, eval_frames, tol=_tol_acc)

    s_init, _, _ = decompose_similarity_transform(current_T)
    init_iou = compute_depth_iou(mesh, current_T, depths, extrinsics, renderer, eval_frames, tol=_tol_iou)
    print(f"{progress_prefix} MASt3R Phase A Initial: Acc={best_score:.4f}, IoU={init_iou:.4f}, scale={s_init:.3f}")

    mast3r_matcher = MASt3RMatcher(device=device)

    try:
        for iteration in range(num_iterations):
            progress = iteration / max(num_iterations - 1, 1)

            print(f"{progress_prefix} MASt3R Iter {iteration+1}/{num_iterations} [{progress*100:0.0f}%]", end="")

            all_mesh_pts = []
            all_vggt_pts = []
            all_conf = []

            for f in sample_frames:
                # Step 1: Render mesh at current pose
                rgb_ren, depth_ren, mask_ren = renderer.render_mesh(mesh, current_T, extrinsics[f])

                if mask_ren.sum() < 30:
                    continue

                # Get real RGB image
                rgb_real = colors[f]

                # Step 2: MASt3R matching -> 2D correspondences
                # Step 3: 3D lifting
                mesh_pts, vggt_pts, conf = mast3r_matcher.establish_3d_correspondences(
                    mesh, current_T,
                    rgb_real=rgb_real,
                    rgb_rendered=rgb_ren,
                    depth_real=depths[f],
                    depth_rendered=depth_ren,
                    extrinsic=extrinsics[f],
                    intrinsic=intrinsic,
                    world_points_frame=world_points[f],
                    conf_threshold=1.0,
                )

                if len(mesh_pts) > 0:
                    all_mesh_pts.append(mesh_pts)
                    all_vggt_pts.append(vggt_pts)
                    all_conf.append(conf)

            if len(all_mesh_pts) == 0:
                print(f" -> no correspondences, skipping")
                continue

            mesh_combined = np.concatenate(all_mesh_pts, axis=0)
            vggt_combined = np.concatenate(all_vggt_pts, axis=0)
            conf_combined = np.concatenate(all_conf, axis=0)

            if len(mesh_combined) > 15000:
                idx = np.random.choice(len(mesh_combined), 15000, replace=False)
                mesh_combined = mesh_combined[idx]
                vggt_combined = vggt_combined[idx]
                conf_combined = conf_combined[idx]

            print(f" -> correspondences={len(mesh_combined)}", end="")

            # Step 3: Umeyama + RANSAC
            inlier_thresh = 0.08 * (1.0 - 0.3 * progress)
            T_delta, inliers, s_d, R_d, t_d = umeyama_alignment_ransac(
                mesh_combined, vggt_combined,
                with_scale=with_scale,
                inlier_threshold=inlier_thresh,
                max_iterations=800,
                min_inliers=max(10, len(mesh_combined) // 10),
            )

            # Scale protection: reject extreme delta scale, fallback to rigid-only
            if with_scale and (s_d < 0.7 or s_d > 1.5):
                T_delta, inliers, s_d, R_d, t_d = umeyama_alignment_ransac(
                    mesh_combined, vggt_combined,
                    with_scale=False,
                    inlier_threshold=inlier_thresh,
                    max_iterations=800,
                    min_inliers=max(10, len(mesh_combined) // 10),
                )

            T_candidate = T_delta @ current_T
            cand_score = compute_depth_accuracy(mesh, T_candidate, depths, extrinsics, renderer, eval_frames, tol=_tol_acc)
            cand_iou = compute_depth_iou(mesh, T_candidate, depths, extrinsics, renderer, eval_frames, tol=_tol_iou)

            inlier_ratio = inliers.sum() / len(inliers) if len(inliers) > 0 else 0
            print(f", inliers={inliers.sum()} ({inlier_ratio:.1%}), "
                  f"scale_delta={s_d:.3f}, Acc={cand_score:.4f}, IoU={cand_iou:.4f}")

            if cand_score > best_score:
                current_T = T_candidate
                best_score = cand_score
                best_T = T_candidate.copy()
            elif cand_score >= best_score * 0.995 and inlier_ratio > 0.3:
                current_T = T_candidate
            else:
                current_T = best_T.copy()

    finally:
        mast3r_matcher.delete()

    final_iou = compute_depth_iou(mesh, best_T, depths, extrinsics, renderer, eval_frames, tol=_tol_iou)
    return best_T, best_score, final_iou


def _apply_constraint_after_alignment(mesh, T, relationship, walls_info=None, camera_pos=None):
    """Re-apply spatial constraint after alignment to prevent constraint violation.

    After Stage 4 alignment (Umeyama rigid transform), the object may violate
    its spatial relationship (e.g. lifted off floor, pushed away from wall).
    This function re-applies the constraint while preserving the alignment
    improvement as much as possible.

    Strategy:
      - supported by floor: only adjust z-translation to keep bottom at z=0
      - attached to wall / embedded in wall: only adjust horizontal position
        to snap back to wall plane, keep z (height) from alignment
      - other: no constraint (free alignment)
    """
    if relationship is None:
        return T

    rel = relationship.lower().replace("_", " ")

    if rel == "supported by floor" or rel == "supported by floor":
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(T)
        z_min = transformed_mesh.bounds[0, 2]
        if abs(z_min) > 0.005:
            z_fix = -z_min
            T_fix = np.eye(4)
            T_fix[2, 3] = z_fix
            T = T_fix @ T
        return T

    if rel in ("attached to wall", "embedded in wall"):
        if walls_info is None or not walls_info:
            return T
        from src.sp_refinement import _get_wall_alignment_target, _select_closest_wall
        forward_vector = T[:3, 2]
        forward_norm = np.linalg.norm(forward_vector)
        if forward_norm < 1e-8:
            return T
        forward_vector = forward_vector / forward_norm
        align_vector, wall_axis = _get_wall_alignment_target(forward_vector, angle_tolerance=30.0)
        if align_vector is None:
            return T
        center = T[:3, 3]
        nearest_wall, min_dist = _select_closest_wall(walls_info, wall_axis, center, span_margin=0.3)
        if nearest_wall is None or min_dist > 0.5:
            return T
        axis_idx = 0 if wall_axis == 'x' else 1
        if rel == "attached to wall":
            transformed_vertices = trimesh.transformations.transform_points(mesh.vertices.copy(), T)
            if camera_pos is not None:
                if camera_pos[axis_idx] > center[axis_idx]:
                    contact_val = transformed_vertices[:, axis_idx].min()
                else:
                    contact_val = transformed_vertices[:, axis_idx].max()
            else:
                contact_val = transformed_vertices[:, axis_idx].min()
            snap_offset = nearest_wall['position'] - contact_val
        else:
            offset = nearest_wall['position'] - center[axis_idx]
            snap_offset = offset
        T_fix = np.eye(4)
        T_fix[axis_idx, 3] = snap_offset
        T = T_fix @ T
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(T)
        z_min = transformed_mesh.bounds[0, 2]
        if z_min < 0.0:
            T_zfix = np.eye(4)
            T_zfix[2, 3] = -z_min
            T = T_zfix @ T
        return T

    return T


def refine_single_instance_combined(
    instance_info,
    instance_masks,
    optimal_frame_id,
    world_points,
    world_points_conf,
    depths,
    extrinsics,
    intrinsic,
    colors,
    num_icp_iterations=8,
    temporal_radius=5,
    inlier_threshold_ratio=0.05,
    instance_index=0,
    total_instances=1,
    instance_name="instance",
    use_mast3r=True,
    mast3r_device='cuda',
    relationship=None,
    walls_info=None,
    camera_pos=None,
):
    """
    Iterative Visual-Spatial Alignment (Paper Section 3.4).

    Phase A: 2D-3D Correspondence-Based Alignment
      - MASt3R mode: Render -> MASt3R 2D match -> 3D lift -> Umeyama
      - Depth mode:  Render -> depth pixel match -> 3D lift -> Umeyama

    Phase B: ICP Fine-Tuning (classical 3D nearest-neighbor)
      - Uses Phase A result as initialization
      - KDTree nearest-neighbor + progressive threshold
      - Umeyama + RANSAC, IoU-gated acceptance

    Selection: Choose best T by max mean IoU across views (Eq. 9)
    """
    mesh = instance_info['original_mesh']
    current_T = instance_info['T'].copy()

    num_frames = len(extrinsics)
    H, W = depths.shape[1], depths.shape[2]

    view_indices = list(range(max(0, optimal_frame_id - temporal_radius),
                              min(num_frames - 1, optimal_frame_id + temporal_radius) + 1))

    real_masks_per_view = {}
    for v in view_indices:
        frame_masks = [im['mask'] for im in instance_masks if im['frame_id'] == v]
        if frame_masks:
            combined = np.zeros((H, W), dtype=bool)
            for m in frame_masks:
                combined |= m.astype(bool)
            real_masks_per_view[v] = combined

    valid_views = [v for v in view_indices if v in real_masks_per_view]
    if len(valid_views) == 0:
        valid_views = list(range(min(8, num_frames)))

    renderer = MeshRenderer(intrinsic, W, H)
    sample_eval = np.linspace(0, num_frames - 1, min(8, num_frames), dtype=int)
    sample_align = np.linspace(0, num_frames - 1, min(6, num_frames), dtype=int)

    # ── 尺寸相对容差 ──
    # 绝对容差 (depth_tol=0.15 / iou_tol=0.10) 对 3cm 小物体无分辨力
    # (整个物体都在容差内, 指标恒定, accept/reject 退化为噪声)。
    # 按物体对角线缩放, 大物体夹回旧值 (向后兼容)。
    _wm = mesh.copy(); _wm.apply_transform(current_T)
    _msz = float(np.max(_wm.bounds[1] - _wm.bounds[0]))   # 世界空间最大轴跨
    _tol_iou = min(max(0.3 * _msz, 0.02), 0.15)
    _tol_acc = min(max(0.3 * _msz, 0.02), 0.10)
    _tol_phaseA = min(max(0.5 * _msz, 0.05), 0.15)
    print(f"    [{instance_index+1}/{total_instances}] size_ref={_msz:.3f} "
          f"tol(iou/acc/phaseA)={_tol_iou:.3f}/{_tol_acc:.3f}/{_tol_phaseA:.3f}")

    initial_iou = compute_depth_iou(mesh, current_T, depths, extrinsics, renderer, sample_eval, tol=_tol_iou)
    initial_acc = compute_depth_accuracy(mesh, current_T, depths, extrinsics, renderer, sample_eval, tol=_tol_acc)
    s_init, _, t_init = decompose_similarity_transform(current_T)
    print(f"    [{instance_index+1}/{total_instances}] Initial: IoU={initial_iou:.4f}, acc={initial_acc:.4f}, scale={s_init:.3f}")

    # ── Phase A: 2D-3D Correspondence-Based Alignment ──
    # Choose MASt3R or depth-based matching
    mast3r_available = use_mast3r and _try_import_mast3r()

    if mast3r_available:
        print(f"    [{instance_index+1}/{total_instances}] Phase A: MASt3R Correspondence-Based Alignment (paper method)...")
        pa_T, pa_acc, pa_iou = _mast3r_phase_a(
            mesh, current_T, world_points, world_points_conf,
            depths, extrinsics, intrinsic, colors, renderer,
            sample_frames=sample_align.tolist(),
            num_iterations=num_icp_iterations,
            with_scale=True,
            device=mast3r_device,
            progress_prefix=f"    [{instance_index+1}/{total_instances}]",
        )
    else:
        if use_mast3r:
            print(f"    [{instance_index+1}/{total_instances}] MASt3R not available, falling back to depth-based matching")
        print(f"    [{instance_index+1}/{total_instances}] Phase A: Depth-Based Correspondence Alignment...")
        pa_T, pa_iou = projection_based_alignment(
            mesh, current_T, world_points, world_points_conf,
            depths, extrinsics, intrinsic, renderer,
            sample_frames=sample_align.tolist(),
            num_iterations=num_icp_iterations,
            with_scale=True,
            inlier_threshold=0.04,
            depth_tolerance=_tol_phaseA,
            progress_prefix=f"    [{instance_index+1}/{total_instances}] Phase A",
        )
        pa_acc = compute_depth_accuracy(mesh, pa_T, depths, extrinsics, renderer, sample_eval, tol=_tol_acc)

    s_pa, _, _ = decompose_similarity_transform(pa_T)
    print(f"    [{instance_index+1}/{total_instances}] Phase A result: Acc={pa_acc:.4f}, IoU={pa_iou:.4f}, scale={s_pa:.3f}")

    if pa_acc > initial_acc + 0.005:
        current_T = pa_T
        print(f"    [{instance_index+1}/{total_instances}] Phase A accepted (accuracy improved)")
    elif pa_acc > initial_acc and pa_iou > initial_iou + 0.01:
        current_T = pa_T
        print(f"    [{instance_index+1}/{total_instances}] Phase A accepted (acc+IoU both improved)")
    else:
        print(f"    [{instance_index+1}/{total_instances}] Phase A not accepted, keeping current T")

    # ── Phase B: ICP Fine-Tuning (classical 3D nearest-neighbor) ──
    print(f"    [{instance_index+1}/{total_instances}] Phase B: ICP Fine-Tuning (3D nearest-neighbor)...")
    icp_T, icp_acc, icp_iou = icp_fine_tuning(
        mesh, current_T, world_points, world_points_conf, depths,
        extrinsics, intrinsic, renderer,
        sample_frames=sample_align.tolist(),
        num_iterations=num_icp_iterations,
        inlier_ratio=inlier_threshold_ratio,
        progress_prefix=f"    [{instance_index+1}/{total_instances}] Phase B",
    )
    s_icp, _, _ = decompose_similarity_transform(icp_T)
    print(f"    [{instance_index+1}/{total_instances}] Phase B result: Acc={icp_acc:.4f}, IoU={icp_iou:.4f}, scale={s_icp:.3f}")

    current_acc = compute_depth_accuracy(mesh, current_T, depths, extrinsics, renderer, sample_eval)
    current_iou = compute_depth_iou(mesh, current_T, depths, extrinsics, renderer, sample_eval)
    if icp_acc > current_acc + 0.005:
        current_T = icp_T
        print(f"    [{instance_index+1}/{total_instances}] Phase B accepted (accuracy improved)")
    elif icp_acc > current_acc and icp_iou > current_iou + 0.01:
        current_T = icp_T
        print(f"    [{instance_index+1}/{total_instances}] Phase B accepted (acc+IoU both improved)")
    else:
        print(f"    [{instance_index+1}/{total_instances}] Phase B not accepted")

    # ── Final Selection + Constraint Re-application ──
    final_iou = compute_depth_iou(mesh, current_T, depths, extrinsics, renderer, sample_eval)
    final_acc = compute_depth_accuracy(mesh, current_T, depths, extrinsics, renderer, sample_eval)
    s_final, _, t_final = decompose_similarity_transform(current_T)

    visible_count = 0
    for f in sample_eval:
        _, _, mr = renderer.render_mesh(mesh, current_T, extrinsics[f])
        if mr.sum() > 20:
            visible_count += 1

    total_eval = len(sample_eval)
    print(f"    [{instance_index+1}/{total_instances}] Final (before constraint): IoU={final_iou:.4f}, acc={final_acc:.4f}, "
          f"scale={s_final:.3f}, visible={visible_count}/{total_eval}")
    print(f"    [{instance_index+1}/{total_instances}] Translation change: dt={np.linalg.norm(t_final - t_init):.4f}m, "
          f"scale change: ds={abs(s_final - s_init)/max(s_init, 0.001):.2%}")

    acc_improved = final_acc > initial_acc + 0.005
    iou_improved = final_iou > initial_iou + 0.005
    is_visible = visible_count > 0
    scale_change = abs(s_final - s_init) / max(s_init, 0.001)
    scale_stable = scale_change < 0.3
    translation_dist = np.linalg.norm(t_final - t_init)

    accepted = False
    if (acc_improved or iou_improved) and is_visible and scale_stable:
        accepted = True
        print(f"    [{instance_index+1}/{total_instances}] Result: ACCEPTED (Acc: {initial_acc:.4f}->{final_acc:.4f}, IoU: {initial_iou:.4f}->{final_iou:.4f})")
    else:
        reject_reason = []
        if not acc_improved and not iou_improved:
            reject_reason.append(f"Acc/IoU not improved ({initial_acc:.4f}->{final_acc:.4f}, {initial_iou:.4f}->{final_iou:.4f})")
        if not is_visible:
            reject_reason.append(f"not visible ({visible_count}/{total_eval})")
        if not scale_stable:
            reject_reason.append(f"scale unstable ({scale_change:.1%})")
        print(f"    [{instance_index+1}/{total_instances}] Result: REJECTED ({'; '.join(reject_reason)}), keeping original T")

    if accepted:
        constrained_T = _apply_constraint_after_alignment(
            mesh, current_T, relationship, walls_info, camera_pos)
        constrained_acc = compute_depth_accuracy(mesh, constrained_T, depths, extrinsics, renderer, sample_eval)
        constrained_iou = compute_depth_iou(mesh, constrained_T, depths, extrinsics, renderer, sample_eval)
        if constrained_acc >= initial_acc * 0.95:
            current_T = constrained_T
            final_acc = constrained_acc
            final_iou = constrained_iou
            if not np.allclose(constrained_T, current_T, atol=1e-6):
                print(f"    [{instance_index+1}/{total_instances}] Constraint re-applied: Acc={constrained_acc:.4f}, IoU={constrained_iou:.4f}")
        else:
            print(f"    [{instance_index+1}/{total_instances}] Constraint would hurt alignment too much (Acc={constrained_acc:.4f} < {initial_acc*0.95:.4f}), keeping unconstrained")
        instance_info['T'] = current_T

    renderer.delete()
    return instance_info
