import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2
import trimesh
import pyrender
from scipy.spatial import KDTree


def load_extrinsics(extrinsics_dir):
    ext_files = sorted([f for f in os.listdir(extrinsics_dir) if f.endswith('.txt')],
                       key=lambda x: int(os.path.splitext(x)[0]))
    extrinsics = []
    for f in ext_files:
        ext = np.loadtxt(os.path.join(extrinsics_dir, f))
        extrinsics.append(ext)
    return extrinsics


def render_depth_at_view(scene_mesh, intrinsic, extrinsic, width, height):
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

    c2w_opencv = np.linalg.inv(extrinsic)
    opencv_to_opengl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
    c2w_opengl = c2w_opencv @ opencv_to_opengl
    zup_to_yup = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    cam_pose = zup_to_yup @ c2w_opengl

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(scene_mesh)
    scene.add(camera, pose=cam_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()

    return color, depth


def compute_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_depth_rmse(depth1, depth2, valid_mask=None):
    if valid_mask is None:
        valid_mask = (depth1 > 0) & (depth2 > 0) & np.isfinite(depth1) & np.isfinite(depth2)
    if valid_mask.sum() == 0:
        return float('inf')
    diff = depth1[valid_mask] - depth2[valid_mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_depth_absrel(depth1, depth2, valid_mask=None):
    if valid_mask is None:
        valid_mask = (depth1 > 0) & (depth2 > 0) & np.isfinite(depth1) & np.isfinite(depth2)
    if valid_mask.sum() == 0:
        return float('inf')
    denom = depth2[valid_mask]
    denom = np.where(denom < 1e-6, 1e-6, denom)
    return float(np.mean(np.abs(depth1[valid_mask] - depth2[valid_mask]) / denom))


def sample_points_and_normals(mesh, num_points=100000):
    points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_idx]
    return points, normals


def normalize_point_clouds(pred_points, gt_points, pred_normals=None, gt_normals=None):
    gt_center = gt_points.mean(axis=0)
    gt_points_norm = gt_points - gt_center
    pred_points_norm = pred_points - gt_center

    gt_max_extent = np.max(np.linalg.norm(gt_points_norm, axis=1))
    if gt_max_extent > 1e-8:
        gt_points_norm = gt_points_norm / gt_max_extent
        pred_points_norm = pred_points_norm / gt_max_extent

    pred_normals_norm = pred_normals
    gt_normals_norm = gt_normals

    return pred_points_norm, gt_points_norm, pred_normals_norm, gt_normals_norm


def compute_chamfer_distance(pred_points, gt_points):
    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(pred_points)

    dist_pred_to_gt, _ = gt_tree.query(pred_points)
    dist_gt_to_pred, _ = pred_tree.query(gt_points)

    cd_l2 = float((np.mean(dist_pred_to_gt ** 2) + np.mean(dist_gt_to_pred ** 2)) / 2.0)
    cd_l1 = float((np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2.0)

    return cd_l2, cd_l1, dist_pred_to_gt, dist_gt_to_pred


def compute_accuracy_completeness(pred_points, gt_points, max_dist=None):
    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(pred_points)

    dist_pred_to_gt, _ = gt_tree.query(pred_points)
    dist_gt_to_pred, _ = pred_tree.query(gt_points)

    if max_dist is not None:
        acc_dists = dist_pred_to_gt[dist_pred_to_gt < max_dist]
        comp_dists = dist_gt_to_pred[dist_gt_to_pred < max_dist]
        accuracy = float(acc_dists.mean()) if len(acc_dists) > 0 else float('inf')
        completeness = float(comp_dists.mean()) if len(comp_dists) > 0 else float('inf')
    else:
        accuracy = float(dist_pred_to_gt.mean())
        completeness = float(dist_gt_to_pred.mean())

    overall = (accuracy + completeness) / 2.0

    return accuracy, completeness, overall


def compute_f_score(pred_points, gt_points, threshold=0.01):
    gt_tree = KDTree(gt_points)
    pred_tree = KDTree(pred_points)

    dist_pred_to_gt, _ = gt_tree.query(pred_points)
    dist_gt_to_pred, _ = pred_tree.query(gt_points)

    precision = float(np.mean(dist_pred_to_gt < threshold))
    recall = float(np.mean(dist_gt_to_pred < threshold))

    if precision + recall > 0:
        f_score = 2.0 * precision * recall / (precision + recall)
    else:
        f_score = 0.0

    return f_score, precision, recall


def compute_normal_consistency(pred_points, pred_normals, gt_points, gt_normals):
    gt_tree = KDTree(gt_points)
    _, indices = gt_tree.query(pred_points)

    gt_normals_matched = gt_normals[indices]

    dot_products = np.abs(np.sum(pred_normals * gt_normals_matched, axis=1))
    nc = float(dot_products.mean())

    return nc


def evaluate_spatial_accuracy(output_path, sample_count=None, glb_path=None):
    if glb_path is None:
        glb_path = os.path.join(output_path, 'final_scene_stage5.glb')
    intrinsic_path = os.path.join(output_path, 'intrinsic.txt')
    extrinsics_dir = os.path.join(output_path, 'extrinsics')
    color_dir = os.path.join(output_path, 'color')
    depth_dir = os.path.join(output_path, 'depth')

    for p, name in [(glb_path, 'GLB file'), (intrinsic_path, 'intrinsic.txt'),
                     (extrinsics_dir, 'extrinsics/'), (depth_dir, 'depth/')]:
        if not os.path.exists(p):
            print(f"⚠️  未找到 {name}，跳过空间精度评估")
            return {}

    print(f"📦 加载场景: {glb_path}")
    scene = trimesh.load(glb_path)
    if isinstance(scene, trimesh.Scene):
        mesh = trimesh.util.concatenate(scene.dump())
    else:
        mesh = scene

    mesh_pr = pyrender.Mesh.from_trimesh(mesh)

    intrinsic = np.loadtxt(intrinsic_path)
    extrinsics = load_extrinsics(extrinsics_dir)

    gt_files = sorted([f for f in os.listdir(color_dir) if f.endswith(('.jpg', '.png'))],
                      key=lambda x: int(os.path.splitext(x)[0]))
    if gt_files:
        sample_img = cv2.imread(os.path.join(color_dir, gt_files[0]))
        height, width = sample_img.shape[:2]
    else:
        height, width = 480, 640

    frame_indices = list(range(len(extrinsics)))
    if sample_count and sample_count < len(frame_indices):
        indices = np.linspace(0, len(frame_indices) - 1, sample_count, dtype=int)
        frame_indices = [frame_indices[i] for i in indices]

    mask_ious = []
    depth_rmses = []
    depth_absrels = []

    print(f"🖼️  渲染 {len(frame_indices)} 个视角进行空间精度评估...")

    for idx in frame_indices:
        ext = extrinsics[idx]
        try:
            _, rendered_depth = render_depth_at_view(mesh_pr, intrinsic, ext, width, height)
        except Exception as e:
            print(f"   ⚠️  帧{idx}渲染失败: {e}")
            continue

        depth_path = os.path.join(depth_dir, f"{idx}.png")
        if not os.path.exists(depth_path):
            continue

        gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0

        if gt_depth.shape != rendered_depth.shape:
            rendered_depth = cv2.resize(rendered_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        valid_mask = (gt_depth > 0) & (rendered_depth > 0) & np.isfinite(gt_depth) & np.isfinite(rendered_depth)

        if valid_mask.sum() < 100:
            continue

        rmse = compute_depth_rmse(rendered_depth, gt_depth, valid_mask)
        absrel = compute_depth_absrel(rendered_depth, gt_depth, valid_mask)
        depth_rmses.append(rmse)
        depth_absrels.append(absrel)

        gt_mask = (gt_depth > 0).astype(np.uint8)
        rendered_mask = (rendered_depth > 0).astype(np.uint8)
        iou = compute_mask_iou(gt_mask, rendered_mask)
        mask_ious.append(iou)

    if not depth_rmses:
        print("⚠️  无有效帧，无法计算空间精度")
        return {}

    results = {
        'MaskIoU': float(np.mean(mask_ious)),
        'DepthRMSE': float(np.mean(depth_rmses)),
        'DepthAbsRel': float(np.mean(depth_absrels)),
        'num_eval_frames': len(depth_rmses),
        'MaskIoU_per_frame': mask_ious,
        'DepthRMSE_per_frame': depth_rmses,
        'DepthAbsRel_per_frame': depth_absrels,
    }

    return results


def evaluate_mesh_geometry(generated_mesh_path, reference_mesh_path,
                           num_sample_points=100000, f_thresholds=None,
                           normalize=True):
    if f_thresholds is None:
        f_thresholds = [0.01, 0.02, 0.05]

    print(f"   加载生成场景: {generated_mesh_path}")
    gen_scene = trimesh.load(generated_mesh_path)
    if isinstance(gen_scene, trimesh.Scene):
        gen_mesh = trimesh.util.concatenate(gen_scene.dump())
    else:
        gen_mesh = gen_scene

    print(f"   加载参考场景: {reference_mesh_path}")
    ref_scene = trimesh.load(reference_mesh_path)
    if isinstance(ref_scene, trimesh.Scene):
        ref_mesh = trimesh.util.concatenate(ref_scene.dump())
    else:
        ref_mesh = ref_scene

    print(f"   采样点云 ({num_sample_points} 点)...")
    gen_points, gen_normals = sample_points_and_normals(gen_mesh, num_sample_points)
    ref_points, ref_normals = sample_points_and_normals(ref_mesh, num_sample_points)

    if len(ref_points) == 0 or len(gen_points) == 0:
        print("⚠️  点云为空，无法计算几何指标")
        return {}

    print(f"   生成点数: {len(gen_points)}, 参考点数: {len(ref_points)}")

    results = {}

    if normalize:
        print("   归一化点云 (MonoSDF协议: 中心对齐 + 缩放到单位球)...")
        gen_points_n, ref_points_n, gen_normals_n, ref_normals_n = normalize_point_clouds(
            gen_points, ref_points, gen_normals, ref_normals)

        print("   [归一化空间] 计算 Chamfer Distance...")
        cd_l2, cd_l1, _, _ = compute_chamfer_distance(gen_points_n, ref_points_n)
        results['ChamferDistance_L2'] = cd_l2
        results['ChamferDistance_L1'] = cd_l1

        print("   [归一化空间] 计算 Accuracy / Completeness / Overall...")
        accuracy, completeness, overall = compute_accuracy_completeness(gen_points_n, ref_points_n)
        results['Accuracy'] = accuracy
        results['Completeness'] = completeness
        results['Overall'] = overall

        for thresh in f_thresholds:
            print(f"   [归一化空间] 计算 F-Score (threshold={thresh})...")
            f_score, precision, recall = compute_f_score(gen_points_n, ref_points_n, threshold=thresh)
            thresh_str = str(thresh).replace('.', '_')
            results[f'F-Score_{thresh_str}'] = f_score
            results[f'Precision_{thresh_str}'] = precision
            results[f'Recall_{thresh_str}'] = recall

        print("   [归一化空间] 计算 Normal Consistency...")
        nc = compute_normal_consistency(gen_points_n, gen_normals_n, ref_points_n, ref_normals_n)
        results['NormalConsistency'] = nc

        print("   [原始空间] 计算 Chamfer Distance (未归一化)...")
        cd_l2_raw, cd_l1_raw, _, _ = compute_chamfer_distance(gen_points, ref_points)
        results['ChamferDistance_L2_raw'] = cd_l2_raw
        results['ChamferDistance_L1_raw'] = cd_l1_raw

        accuracy_raw, completeness_raw, overall_raw = compute_accuracy_completeness(gen_points, ref_points)
        results['Accuracy_raw'] = accuracy_raw
        results['Completeness_raw'] = completeness_raw
        results['Overall_raw'] = overall_raw

        results['normalized'] = True
    else:
        print("   计算 Chamfer Distance (未归一化)...")
        cd_l2, cd_l1, dist_pred_to_gt, dist_gt_to_pred = compute_chamfer_distance(gen_points, ref_points)
        results['ChamferDistance_L2'] = cd_l2
        results['ChamferDistance_L1'] = cd_l1

        print("   计算 Accuracy / Completeness / Overall...")
        accuracy, completeness, overall = compute_accuracy_completeness(gen_points, ref_points)
        results['Accuracy'] = accuracy
        results['Completeness'] = completeness
        results['Overall'] = overall

        for thresh in f_thresholds:
            print(f"   计算 F-Score (threshold={thresh})...")
            f_score, precision, recall = compute_f_score(gen_points, ref_points, threshold=thresh)
            thresh_str = str(thresh).replace('.', '_')
            results[f'F-Score_{thresh_str}'] = f_score
            results[f'Precision_{thresh_str}'] = precision
            results[f'Recall_{thresh_str}'] = recall

        print("   计算 Normal Consistency...")
        nc = compute_normal_consistency(gen_points, gen_normals, ref_points, ref_normals)
        results['NormalConsistency'] = nc

        results['normalized'] = False

    results['num_gen_points'] = len(gen_points)
    results['num_ref_points'] = len(ref_points)

    print(f"   CD-L2: {results['ChamferDistance_L2']:.6f}")
    print(f"   CD-L1: {results['ChamferDistance_L1']:.6f}")
    print(f"   Accuracy: {results['Accuracy']:.6f}")
    print(f"   Completeness: {results['Completeness']:.6f}")
    print(f"   Overall: {results['Overall']:.6f}")
    for thresh in f_thresholds:
        thresh_str = str(thresh).replace('.', '_')
        fs_key = f'F-Score_{thresh_str}'
        print(f"   F-Score@{thresh}: {results[fs_key]:.4f}")
    print(f"   Normal Consistency: {results['NormalConsistency']:.4f}")

    return results


def evaluate_point_cloud_geometry(generated_ply_path, reference_ply_path,
                                  f_thresholds=None, voxel_size=0.005,
                                  normalize=True):
    if f_thresholds is None:
        f_thresholds = [0.01, 0.02, 0.05]

    import open3d as o3d

    print(f"   加载生成点云: {generated_ply_path}")
    gen_pcd = o3d.io.read_point_cloud(generated_ply_path)
    print(f"   加载参考点云: {reference_ply_path}")
    ref_pcd = o3d.io.read_point_cloud(reference_ply_path)

    gen_points = np.asarray(gen_pcd.points)
    ref_points = np.asarray(ref_pcd.points)

    if len(ref_points) == 0 or len(gen_points) == 0:
        print("⚠️  点云为空，无法计算几何指标")
        return {}

    if voxel_size > 0:
        gen_pcd = gen_pcd.voxel_down_sample(voxel_size)
        ref_pcd = ref_pcd.voxel_down_sample(voxel_size)
        gen_points = np.asarray(gen_pcd.points)
        ref_points = np.asarray(ref_pcd.points)

    gen_normals = np.asarray(gen_pcd.normals) if gen_pcd.has_normals() else None
    ref_normals = np.asarray(ref_pcd.normals) if ref_pcd.has_normals() else None

    print(f"   生成点数: {len(gen_points)}, 参考点数: {len(ref_points)}")

    results = {}

    if normalize:
        print("   归一化点云 (MonoSDF协议: 中心对齐 + 缩放到单位球)...")
        gen_points_n, ref_points_n, gen_normals_n, ref_normals_n = normalize_point_clouds(
            gen_points, ref_points, gen_normals, ref_normals)

        print("   [归一化空间] 计算 Chamfer Distance...")
        cd_l2, cd_l1, _, _ = compute_chamfer_distance(gen_points_n, ref_points_n)
        results['ChamferDistance_L2'] = cd_l2
        results['ChamferDistance_L1'] = cd_l1

        print("   [归一化空间] 计算 Accuracy / Completeness / Overall...")
        accuracy, completeness, overall = compute_accuracy_completeness(gen_points_n, ref_points_n)
        results['Accuracy'] = accuracy
        results['Completeness'] = completeness
        results['Overall'] = overall

        for thresh in f_thresholds:
            print(f"   [归一化空间] 计算 F-Score (threshold={thresh})...")
            f_score, precision, recall = compute_f_score(gen_points_n, ref_points_n, threshold=thresh)
            thresh_str = str(thresh).replace('.', '_')
            results[f'F-Score_{thresh_str}'] = f_score
            results[f'Precision_{thresh_str}'] = precision
            results[f'Recall_{thresh_str}'] = recall

        if gen_normals_n is not None and ref_normals_n is not None:
            print("   [归一化空间] 计算 Normal Consistency...")
            nc = compute_normal_consistency(gen_points_n, gen_normals_n, ref_points_n, ref_normals_n)
            results['NormalConsistency'] = nc
        else:
            print("   ⚠️  点云缺少法线信息，跳过 Normal Consistency")

        print("   [原始空间] 计算 Chamfer Distance (未归一化)...")
        cd_l2_raw, cd_l1_raw, _, _ = compute_chamfer_distance(gen_points, ref_points)
        results['ChamferDistance_L2_raw'] = cd_l2_raw
        results['ChamferDistance_L1_raw'] = cd_l1_raw

        accuracy_raw, completeness_raw, overall_raw = compute_accuracy_completeness(gen_points, ref_points)
        results['Accuracy_raw'] = accuracy_raw
        results['Completeness_raw'] = completeness_raw
        results['Overall_raw'] = overall_raw

        results['normalized'] = True
    else:
        print("   计算 Chamfer Distance (未归一化)...")
        cd_l2, cd_l1, dist_pred_to_gt, dist_gt_to_pred = compute_chamfer_distance(gen_points, ref_points)
        results['ChamferDistance_L2'] = cd_l2
        results['ChamferDistance_L1'] = cd_l1

        print("   计算 Accuracy / Completeness / Overall...")
        accuracy, completeness, overall = compute_accuracy_completeness(gen_points, ref_points)
        results['Accuracy'] = accuracy
        results['Completeness'] = completeness
        results['Overall'] = overall

        for thresh in f_thresholds:
            print(f"   计算 F-Score (threshold={thresh})...")
            f_score, precision, recall = compute_f_score(gen_points, ref_points, threshold=thresh)
            thresh_str = str(thresh).replace('.', '_')
            results[f'F-Score_{thresh_str}'] = f_score
            results[f'Precision_{thresh_str}'] = precision
            results[f'Recall_{thresh_str}'] = recall

        if gen_normals is not None and ref_normals is not None:
            print("   计算 Normal Consistency...")
            nc = compute_normal_consistency(gen_points, gen_normals, ref_points, ref_normals)
            results['NormalConsistency'] = nc
            print(f"   Normal Consistency: {nc:.4f}")
        else:
            print("   ⚠️  点云缺少法线信息，跳过 Normal Consistency")

        results['normalized'] = False

    results['num_gen_points'] = len(gen_points)
    results['num_ref_points'] = len(ref_points)

    print(f"   CD-L2: {results['ChamferDistance_L2']:.6f}")
    print(f"   CD-L1: {results['ChamferDistance_L1']:.6f}")
    print(f"   Accuracy: {results['Accuracy']:.6f}")
    print(f"   Completeness: {results['Completeness']:.6f}")
    print(f"   Overall: {results['Overall']:.6f}")
    for thresh in f_thresholds:
        thresh_str = str(thresh).replace('.', '_')
        fs_key = f'F-Score_{thresh_str}'
        print(f"   F-Score@{thresh}: {results[fs_key]:.4f}")
    if 'NormalConsistency' in results:
        print(f"   Normal Consistency: {results['NormalConsistency']:.4f}")

    return results
