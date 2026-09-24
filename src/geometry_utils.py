import numpy as np
import trimesh
import matplotlib
from scipy.spatial import Delaunay

def compute_surface_area_from_pointmap(pointmap, mask, max_triangle_size = 2e-4):
    """
    Compute the surface area of an object given its pointmap and mask using Delaunay triangulation.
    
    Args:
        pointmap: HxWx3 array where each element is a 3D coordinate (X, Y, Z).
        mask: HxW binary array where True indicates the object and False is background.
        max_triangle_size: Maximum allowed area for a triangle to be considered valid (to filter outliers).
    
    Returns:
        The computed surface area of the object.
    """
    H, W, _ = pointmap.shape
    
    # Extract points from pointmap based on the mask
    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 3:
        return 0.0
    
    points_3d = pointmap[y_coords, x_coords]
    
    # Ensure there are at least 3 valid points
    if points_3d.shape[0] < 3:
        return 0.0
    
    pixel_coords = np.column_stack([x_coords, y_coords])

    try:
        tri = Delaunay(pixel_coords)
    except Exception as e:
        print(f"Delaunay triangulation failed: {e}")
        return 0.0
    
    simplices = tri.simplices
    triangles_3d = points_3d[simplices]
    
    # Calculate vectors AB and AC for each triangle
    AB = triangles_3d[:, 1] - triangles_3d[:, 0]
    AC = triangles_3d[:, 2] - triangles_3d[:, 0]

    # Calculate cross product and triangle areas
    cross_product = np.cross(AB, AC)
    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    
    # Filter out invalid triangles
    valid_triangle_mask = (triangle_areas > 0) & (triangle_areas < max_triangle_size)
    valid_areas = triangle_areas[valid_triangle_mask]
    
    total_area = np.sum(valid_areas)
    
    return total_area

def predictions_to_pcd(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    prediction_mode="Predicted Depthmap",
) -> trimesh.Scene:
    """
    Copied from vggt/visual_util.py
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    return point_cloud_data

def get_plane_info(pointmap, mask):
    '''
    Compute the plane parameters from a pointmap and binary mask using PCA.
    
    Args:
    - pointmap: HxWx3 numpy array where each element is a 3D coordinate (X, Y, Z).
    - mask: HxW binary numpy array where True indicates the object and False is background.
    Returns:
    - A dictionary
        {
            'normal': np.ndarray of shape (3,), the normal vector of the plane,
            'd': float, the distance from the plane to the origin,
            'area': float, the surface area of the plane,
            'centroid': np.ndarray of shape (3,), the centroid of the plane
            'mean_distance': float, the mean distance of points to the fitted plane
        }
    '''
    masked_points = pointmap[mask]  # shape (N, 3)
    centroid = np.mean(masked_points, axis=0)  # shape (3,)
    centered_points = masked_points - centroid
    cov_matrix = np.dot(centered_points.T, centered_points)
    try:
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    except Exception as e:
        return {
            'normal': np.array([0, 0, 1]),  # Default normal vector
            'd': -centroid[2],  # Distance from the plane to the origin
            'area': 0,  # Area based on the number of points
            'centroid': centroid,  # Centroid of the points
            'mean_distance': 1e6  # Large mean distance to indicate poor fit
        }
    min_eigenvalue_idx = np.argmin(eigenvalues)
    normal = eigenvectors[:, min_eigenvalue_idx]
    normal = -normal if normal[0] < 0 else normal
    normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    d = -np.dot(normal, centroid)
    distance_numerator = np.abs(np.dot(masked_points, normal) + d)
    distance_denominator = np.linalg.norm(normal)
    point_distances = distance_numerator / distance_denominator
    mean_distance = np.mean(point_distances)
    area = compute_surface_area_from_pointmap(pointmap, mask)

    return {
        'normal': normal,
        'd': d,
        'area': area,
        'centroid': centroid,
        'mean_distance': mean_distance
    }

def align_to_room_coordinate_system(world_points, wall_masks, floor_masks, wall_mean_distance_thres=0.02, floor_mean_distance_thres=0.02):
    '''
    Align the scene to a room coordinate system based on the detected wall and floor masks.
    
    Args:
    - world_points: numpy array of shape (T, H, W, 3) representing the 3D coordinates of each pixel in each frame.
    - wall_masks: list of dictionaries containing 'frame_id' and 'mask' for detected walls.
    - floor_masks: list of dictionaries containing 'frame_id' and 'mask' for detected floors.
    
    Returns:
    - R: numpy array of shape (3, 3) representing the rotation matrix to align the scene to the room coordinate system.
    - t: numpy array of shape (3,) representing the translation vector to align the scene to the room coordinate system.
    '''
    wall_plane_infos = []
    floor_plane_infos = []
    for wall_mask in wall_masks:
        frame_id = wall_mask['frame_id']
        mask = wall_mask['mask']
        pointmap = world_points[frame_id]  # shape (H, W, 3)
        plane_info = get_plane_info(pointmap, mask)
        if plane_info['mean_distance'] < wall_mean_distance_thres:
            wall_plane_infos.append(plane_info)
    for floor_mask in floor_masks:
        frame_id = floor_mask['frame_id']
        mask = floor_mask['mask']
        pointmap = world_points[frame_id]  # shape (H, W, 3)
        plane_info = get_plane_info(pointmap, mask)
        if plane_info['mean_distance'] < floor_mean_distance_thres:
            floor_plane_infos.append(plane_info)
    if len(floor_plane_infos) == 0:
        return np.eye(3), np.zeros(3)
    # choose the floor plane with the largest area and normal vector close to mean floor normal vector (in case wrong floor segmentation)
    mean_floor_normal = np.mean([info['normal'] for info in floor_plane_infos], axis=0)
    mean_floor_normal = mean_floor_normal / np.linalg.norm(mean_floor_normal)
    vaild_floor_plane_infos = [info for info in floor_plane_infos if abs(np.dot(info['normal'], mean_floor_normal)) > np.cos(np.radians(30))]
    if len(vaild_floor_plane_infos) == 0:
        # 各帧 floor 平面法向分歧过大 (>30°), 无法一致对齐 → 返回单位阵表示失败,
        # 让 run_stage2 回退到 Stage2-relaxed / VLM / large_plane / GeoCalib 兜底链
        return np.eye(3), np.zeros(3)
    floor_plane_info = max(vaild_floor_plane_infos, key=lambda x: x['area'])
    floor_normal = floor_plane_info['normal']
    # choose the wall plane with the largest area and orthogonal (within 5 degrees) to the floor
    orthogonal_wall_plane_infos = [info for info in wall_plane_infos if abs(np.dot(info['normal'], floor_normal)) < np.cos(np.radians(85))]
    if len(orthogonal_wall_plane_infos) == 0:
        return np.eye(3), np.zeros(3)
    wall_plane_info = max(orthogonal_wall_plane_infos, key=lambda x: x['area'])
    wall_normal_1 = wall_plane_info['normal']
    # the floor normal should be upward, use the wall centroid to determine the direction of the wall normal
    floor_to_wall_vector = wall_plane_info['centroid'] - floor_plane_info['centroid']
    if np.dot(floor_to_wall_vector, floor_normal) < 0:
        floor_normal = -floor_normal
    # get the third axis by cross product and refine the wall normal by cross product to ensure orthogonality
    wall_normal_2 = np.cross(floor_normal, wall_normal_1)
    wall_normal_2 = wall_normal_2 / np.linalg.norm(wall_normal_2)
    wall_normal_1 = np.cross(wall_normal_2, floor_normal)
    wall_normal_1 = wall_normal_1 / np.linalg.norm(wall_normal_1)
    R = np.stack([wall_normal_1, wall_normal_2, floor_normal], axis=0)
    # use the floor plane to determine the translation, set the floor plane to be at z=0
    floor_centroid = floor_plane_info['centroid']
    rotated_floor_centroid = floor_centroid @ R.T
    current_floor_z = rotated_floor_centroid[2]
    t = np.zeros(3)
    t[2] = -current_floor_z
    # set the origin to the center of the scene bbox
    all_points = world_points.reshape(-1, 3)
    rotated_points = all_points @ R.T
    min_coords = np.min(rotated_points, axis=0)
    max_coords = np.max(rotated_points, axis=0)
    center = (min_coords + max_coords) / 2
    t[:2] = -center[:2]
    return R, t


def _estimate_floor_centroid(all_points, floor_normal, bottom_percentile=10):
    """用 floor_normal 方向上最低的 N% 点的质心近似 floor_centroid.

    比用整个点云质心更准确, 因为 floor 在场景底部.
    """
    projections = all_points @ floor_normal
    threshold = np.percentile(projections, bottom_percentile)
    bottom_mask = projections <= threshold
    return np.mean(all_points[bottom_mask], axis=0)


def _orient_floor_normal(floor_normal, floor_centroid, all_points, camera_positions=None):
    """确保 floor_normal 朝上 (朝向场景上方).

    优先用点云质心判断 (floor 在底部, 场景质心在上方);
    当 floor_centroid ≈ all_centroid (如 GeoCalib 阶段用 bottom-percentile 仍可能接近) 时,
    用相机位置作为 "上方" 参考 (相机总在 floor 上方).
    """
    all_centroid = np.mean(all_points, axis=0)
    diff = all_centroid - floor_centroid
    if np.linalg.norm(diff) > 1e-6:
        if np.dot(diff, floor_normal) < 0:
            return -floor_normal
    elif camera_positions is not None:
        mean_cam = np.mean(camera_positions, axis=0)
        if np.dot(mean_cam - floor_centroid, floor_normal) < 0:
            return -floor_normal
    return floor_normal


def _build_R_t_from_floor(world_points, floor_normal, floor_centroid, wall_normal_1=None):
    """
    给定 floor 平面法线和质心, 构造 R, t.
    如果提供 wall_normal_1, 用它确定水平方向; 否则用点云 PCA.
    """
    floor_normal = floor_normal / np.linalg.norm(floor_normal)

    if wall_normal_1 is not None:
        wall_normal_1 = wall_normal_1 / np.linalg.norm(wall_normal_1)
        wall_normal_1 = wall_normal_1 - np.dot(wall_normal_1, floor_normal) * floor_normal
        wall_normal_1 = wall_normal_1 / np.linalg.norm(wall_normal_1)
    else:
        all_points_flat = world_points.reshape(-1, 3)
        centered = all_points_flat - np.mean(all_points_flat, axis=0)
        cov = np.dot(centered.T, centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        wall_normal_1 = eigenvectors[:, -1]
        wall_normal_1 = wall_normal_1 - np.dot(wall_normal_1, floor_normal) * floor_normal
        wall_normal_1 = wall_normal_1 / np.linalg.norm(wall_normal_1)

    wall_normal_2 = np.cross(floor_normal, wall_normal_1)
    wall_normal_2 = wall_normal_2 / np.linalg.norm(wall_normal_2)
    wall_normal_1 = np.cross(wall_normal_2, floor_normal)
    wall_normal_1 = wall_normal_1 / np.linalg.norm(wall_normal_1)

    R = np.stack([wall_normal_1, wall_normal_2, floor_normal], axis=0)

    rotated_floor_centroid = floor_centroid @ R.T
    t = np.zeros(3)
    t[2] = -rotated_floor_centroid[2]
    all_points = world_points.reshape(-1, 3)
    rotated_points = all_points @ R.T
    min_coords = np.min(rotated_points, axis=0)
    max_coords = np.max(rotated_points, axis=0)
    center = (min_coords + max_coords) / 2
    t[:2] = -center[:2]

    return R, t


def align_via_objects(world_points, wall_masks, floor_masks,
                      wall_mean_distance_thres=0.05,
                      floor_mean_distance_thres=0.05,
                      orthogonal_threshold_deg=80):
    '''
    阶段2: 放宽阈值 + 只用floor平面建立坐标系.

    与 align_to_room_coordinate_system 的区别:
      1. 放宽 mean_distance 阈值 (0.02 → 0.05)
      2. 放宽正交条件 (85° → 80°)
      3. 如果没有正交wall, 只用floor平面 + 点云PCA确定水平方向

    Returns:
      (R, t, info_dict): 成功
      (np.eye(3), np.zeros(3), info_dict): 失败
    '''
    wall_plane_infos = []
    floor_plane_infos = []
    for wall_mask in wall_masks:
        frame_id = wall_mask['frame_id']
        mask = wall_mask['mask']
        pointmap = world_points[frame_id]
        plane_info = get_plane_info(pointmap, mask)
        if plane_info['mean_distance'] < wall_mean_distance_thres:
            wall_plane_infos.append(plane_info)
    for floor_mask in floor_masks:
        frame_id = floor_mask['frame_id']
        mask = floor_mask['mask']
        pointmap = world_points[frame_id]
        plane_info = get_plane_info(pointmap, mask)
        if plane_info['mean_distance'] < floor_mean_distance_thres:
            floor_plane_infos.append(plane_info)

    if len(floor_plane_infos) == 0:
        return np.eye(3), np.zeros(3), {'reason': 'no_valid_floor', 'n_floor': len(floor_masks), 'n_wall': len(wall_masks)}

    mean_floor_normal = np.mean([info['normal'] for info in floor_plane_infos], axis=0)
    mean_floor_normal = mean_floor_normal / np.linalg.norm(mean_floor_normal)
    valid_floor_plane_infos = [info for info in floor_plane_infos
                               if abs(np.dot(info['normal'], mean_floor_normal)) > np.cos(np.radians(30))]
    if len(valid_floor_plane_infos) == 0:
        return np.eye(3), np.zeros(3), {'reason': 'no_consistent_floor', 'n_floor': len(floor_masks)}
    floor_plane_info = max(valid_floor_plane_infos, key=lambda x: x['area'])
    floor_normal = floor_plane_info['normal']
    floor_centroid = floor_plane_info['centroid']

    all_points = world_points.reshape(-1, 3)
    floor_normal = _orient_floor_normal(floor_normal, floor_centroid, all_points)

    orthogonal_wall_plane_infos = [info for info in wall_plane_infos
                                   if abs(np.dot(info['normal'], floor_normal)) < np.cos(np.radians(orthogonal_threshold_deg))]

    if len(orthogonal_wall_plane_infos) > 0:
        wall_plane_info = max(orthogonal_wall_plane_infos, key=lambda x: x['area'])
        wall_normal_1 = wall_plane_info['normal']
        method = 'floor+wall'
    else:
        wall_normal_1 = None
        method = 'floor+pca'

    R, t = _build_R_t_from_floor(world_points, floor_normal, floor_centroid, wall_normal_1)

    return R, t, {
        'method': method,
        'floor_area': float(floor_plane_info['area']),
        'floor_mean_distance': float(floor_plane_info['mean_distance']),
        'n_floor_valid': len(floor_plane_infos),
        'n_wall_orthogonal': len(orthogonal_wall_plane_infos),
    }


def align_via_large_plane(world_points, large_plane_masks, colors=None, sam3_image_model=None,
                          vlm_model=None, vlm_processor=None, floor_mean_distance_thres=0.05):
    '''
    阶段3: 用SAM3大平面mask拟合floor平面.

    增强版: 如果提供了 VLM 模型, 先用场景类型识别来选择更有针对性的 prompt.

    Args:
      world_points: (T, H, W, 3)
      large_plane_masks: list of dicts with 'frame_id' and 'mask'
      colors: (S, H, W, 3) RGB 图像 (可选, 用于 VLM 场景类型识别)
      sam3_image_model: SAM3 image model (可选, 用于 VLM 辅助分割)
      vlm_model: VLM 模型 (可选, 用于场景类型识别)
      vlm_processor: VLM processor (可选)
      floor_mean_distance_thres: 平面拟合 mean_distance 阈值

    Returns:
      (R, t, info_dict): 成功
      (np.eye(3), np.zeros(3), info_dict): 失败
    '''
    # 如果没有 large_plane_masks 但有 VLM, 尝试用 VLM 辅助分割
    enhanced_masks = list(large_plane_masks) if large_plane_masks else []
    scene_info = {}
    if len(enhanced_masks) == 0 and colors is not None and sam3_image_model is not None and vlm_model is not None:
        from src.object_segmentation import segment_large_flat_surfaces
        print("   大平面 mask 为空, 尝试 VLM 增强的大平面分割...", flush=True)
        enhanced_masks = segment_large_flat_surfaces(
            colors, sam3_image_model,
            vlm_model=vlm_model, vlm_processor=vlm_processor,
            min_mask_area=5000, sample_frame_for_vlm=0
        )
        scene_info['enhanced_from_vlm'] = True

    if len(enhanced_masks) == 0:
        return np.eye(3), np.zeros(3), {
            'reason': 'no_large_plane',
            'n_masks': len(large_plane_masks),
            'n_enhanced': len(enhanced_masks),
        }

    plane_infos = []
    for pm in enhanced_masks:
        frame_id = pm['frame_id']
        mask = pm['mask']
        pointmap = world_points[frame_id]
        info = get_plane_info(pointmap, mask)
        if info['mean_distance'] < floor_mean_distance_thres and info['area'] > 0:
            plane_infos.append(info)

    if len(plane_infos) == 0:
        info_dict = {'reason': 'no_valid_plane', 'n_masks': len(enhanced_masks)}
        info_dict.update(scene_info)
        return np.eye(3), np.zeros(3), info_dict

    floor_plane_info = max(plane_infos, key=lambda x: x['area'])
    floor_normal = floor_plane_info['normal']
    floor_centroid = floor_plane_info['centroid']

    all_points = world_points.reshape(-1, 3)
    floor_normal = _orient_floor_normal(floor_normal, floor_centroid, all_points)

    R, t = _build_R_t_from_floor(world_points, floor_normal, floor_centroid, wall_normal_1=None)

    result_info = {
        'method': 'large_plane+pca',
        'floor_area': float(floor_plane_info['area']),
        'floor_mean_distance': float(floor_plane_info['mean_distance']),
        'n_planes': len(plane_infos),
    }
    result_info.update(scene_info)
    return R, t, result_info


def align_via_vlm_floor_points(world_points, images, sam3_image_model,
                               vlm_model, vlm_processor,
                               num_sample_frames=4,
                               floor_mean_distance_thres=0.05):
    """
    阶段2.5: VLM识别地面参考点 + SAM3 box prompt分割 + 平面拟合.

    SAM3的Sam3Processor没有point prompt, 因此:
      1. 用VLM从采样帧中识别地面代表性点 (归一化坐标)
      2. 围绕每个点构造小box, 调用SAM3 add_geometric_prompt生成mask
      3. 用mask拟合floor平面, 建立坐标系

    Args:
        world_points: (T, H, W, 3)
        images: (T, H, W, 3) uint8 RGB
        sam3_image_model: SAM3 image processor (已加载)
        vlm_model: VLM模型 (已加载)
        vlm_processor: VLM processor (已加载)
        num_sample_frames: 采样多少帧进行VLM检测 (默认4)
        floor_mean_distance_thres: 平面拟合mean_distance阈值

    Returns:
        (R, t, info_dict): 成功或失败
    """
    from src.object_segmentation import (
        detect_floor_reference_points_with_vlm,
        segment_floor_with_box_prompts,
    )

    T = images.shape[0]
    if num_sample_frames is not None and T > num_sample_frames:
        indices = np.round(np.linspace(0, T - 1, num_sample_frames)).astype(int)
    else:
        indices = np.arange(T)

    reference_points_per_frame = {}
    for idx in indices:
        points = detect_floor_reference_points_with_vlm(
            images[idx], vlm_model, vlm_processor, num_points=4
        )
        if points:
            reference_points_per_frame[int(idx)] = points

    if len(reference_points_per_frame) == 0:
        return np.eye(3), np.zeros(3), {
            'reason': 'vlm_no_floor_points',
            'n_sampled_frames': len(indices),
        }

    floor_masks = segment_floor_with_box_prompts(
        images, sam3_image_model, reference_points_per_frame, box_size=0.05
    )

    if len(floor_masks) == 0:
        return np.eye(3), np.zeros(3), {
            'reason': 'sam3_no_floor_masks',
            'n_vlm_points': sum(len(v) for v in reference_points_per_frame.values()),
        }

    # 用现有 align_via_objects 的逻辑拟合平面 (去掉wall, 只用floor)
    # 这里复用 get_plane_info + _build_R_t_from_floor
    plane_infos = []
    for fm in floor_masks:
        frame_id = fm['frame_id']
        mask = fm['mask']
        pointmap = world_points[frame_id]
        info = get_plane_info(pointmap, mask)
        if info['mean_distance'] < floor_mean_distance_thres and info['area'] > 0:
            plane_infos.append(info)

    if len(plane_infos) == 0:
        return np.eye(3), np.zeros(3), {
            'reason': 'no_valid_floor_plane',
            'n_sam3_masks': len(floor_masks),
        }

    # 选最大面积的floor平面
    floor_plane_info = max(plane_infos, key=lambda x: x['area'])
    floor_normal = floor_plane_info['normal']
    floor_centroid = floor_plane_info['centroid']

    all_points = world_points.reshape(-1, 3)
    floor_normal = _orient_floor_normal(floor_normal, floor_centroid, all_points)

    R, t = _build_R_t_from_floor(world_points, floor_normal, floor_centroid, wall_normal_1=None)

    return R, t, {
        'method': 'vlm_floor_points+sam3_box',
        'floor_area': float(floor_plane_info['area']),
        'floor_mean_distance': float(floor_plane_info['mean_distance']),
        'n_vlm_frames': len(reference_points_per_frame),
        'n_sam3_masks': len(floor_masks),
        'n_valid_planes': len(plane_infos),
    }


def align_via_geocalib(images, world_points, extrinsics, max_frames=30, mad_threshold=3.0, quality_threshold=0.3):
    '''
    阶段4: 用 GeoCalib 从图像估计重力方向, 构造坐标系对齐.

    关键修复: GeoCalib 返回的 gravity 是相机坐标系下的向量, 必须用 extrinsics
    变换到世界坐标系后才能作为 floor_normal. 原代码直接用相机坐标系 gravity, 导致
    R[2,2] 偏小 (如 scene15 的 R[2,2]=0.2259, z 轴偏 77°).

    流程:
      1. 对每帧图像运行 GeoCalib → 得到 gravity 向量 (相机坐标系, 指向 DOWN)
      2. 用 extrinsics 将每帧 gravity 从相机坐标变换到世界坐标
      3. 置信度加权的球面平均 + MAD 异常值剔除 → 世界坐标系下的 gravity
      4. floor_normal = -gravity (UP), 用 bottom-percentile 估计 floor_centroid
      5. 构造 R, t; 若 |R[2,2]| < quality_threshold 则判定对齐失败

    Args:
      images: numpy array of shape (S, H, W, 3), uint8 RGB
      world_points: (T, H, W, 3)
      extrinsics: (N, 4, 4) or (N, 3, 4) w2c 外参, 与 images 帧数对应
      max_frames: 最多处理多少帧 (GeoCalib 推理较慢, 默认30帧)
      mad_threshold: MAD 异常值剔除阈值 (默认3.0)
      quality_threshold: 质量阈值, |R[2,2]| < threshold 判定失败 (默认0.3, 原0.5)

    Returns:
      (R, t, info_dict): 成功
      (np.eye(3), np.zeros(3), info_dict): 失败
    '''
    try:
        import torch
        from geocalib import GeoCalib
    except ImportError:
        return np.eye(3), np.zeros(3), {'reason': 'geocalib_not_installed'}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = GeoCalib(weights="pinhole").to(device)
        model.eval()
    except Exception as e:
        return np.eye(3), np.zeros(3), {'reason': f'geocalib_load_failed: {e}'}

    S = images.shape[0]
    if max_frames is not None and S > max_frames:
        indices = np.round(np.linspace(0, S - 1, max_frames)).astype(int)
    else:
        indices = np.arange(S)

    gravity_world_vecs = []  # 世界坐标系下的 gravity 向量
    confidences = []
    per_frame_info = []

    with torch.no_grad():
        for idx in indices:
            img = images[idx]
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            img_tensor = img_tensor.to(device)
            try:
                results = model.calibrate(img_tensor, camera_model="pinhole")
                grav = results["gravity"]
                grav_cam = grav.vec3d.squeeze(0).cpu().numpy()  # [3] 相机坐标系, 指向 DOWN
                up_conf = results["up_confidence"].mean().item()
                lat_conf = results["latitude_confidence"].mean().item()
                conf = (up_conf + lat_conf) / 2.0

                R_w2c = extrinsics[idx, :3, :3]  # (3,3) world→camera rotation
                grav_world = R_w2c.T @ grav_cam    # camera→world: R_c2w = R_w2c.T

                gravity_world_vecs.append(grav_world)
                confidences.append(conf)
                per_frame_info.append({
                    'frame_id': int(idx),
                    'grav_cam': grav_cam.tolist(),
                    'grav_world': grav_world.tolist(),
                    'confidence': conf,
                })
            except Exception as e:
                per_frame_info.append({
                    'frame_id': int(idx),
                    'error': str(e),
                })

    if len(gravity_world_vecs) == 0:
        return np.eye(3), np.zeros(3), {
            'reason': 'geocalib_no_frames',
            'n_attempted': len(indices),
            'per_frame': per_frame_info,
        }

    vecs = torch.from_numpy(np.stack(gravity_world_vecs)).float()  # [N, 3] world coords
    confs = torch.tensor(confidences)  # [N]

    def spherical_mean(v, w=None):
        if w is not None:
            w = w / w.sum()
            m = (v * w.unsqueeze(-1)).sum(dim=0)
        else:
            m = v.mean(dim=0)
        return torch.nn.functional.normalize(m, dim=-1)

    mean_vec = spherical_mean(vecs)
    dots = (vecs * mean_vec.unsqueeze(0).expand_as(vecs)).sum(dim=-1).clamp(-1.0, 1.0)
    angles = torch.acos(dots)
    median_angle = angles.median()
    mad = (angles - median_angle).abs().median().clamp(min=1e-6)
    threshold = median_angle + mad_threshold * mad
    inlier_mask = angles <= threshold
    n_inliers = inlier_mask.sum().item()

    if n_inliers == 0:
        return np.eye(3), np.zeros(3), {
            'reason': 'geocalib_all_outliers',
            'n_frames': len(vecs),
            'per_frame': per_frame_info,
        }

    inlier_vecs = vecs[inlier_mask]
    inlier_confs = confs[inlier_mask]
    final_vec = spherical_mean(inlier_vecs, w=inlier_confs)

    # final_vec 是世界坐标系下的 gravity (DOWN), floor_normal = -gravity (UP)
    gravity_world = final_vec.numpy()  # [3], world coords, points DOWN
    floor_normal = -gravity_world / np.linalg.norm(gravity_world)  # negate → UP

    all_points = world_points.reshape(-1, 3)
    floor_centroid = _estimate_floor_centroid(all_points, floor_normal, bottom_percentile=10)

    # VGGT extrinsics 是 w2c (world→camera): p_cam = R @ p_world + t
    # 相机在世界坐标的位置 = -R.T @ t (不是 t 本身)
    R_w2c_all = extrinsics[:, :3, :3]  # (N, 3, 3)
    t_w2c = extrinsics[:, :3, 3]       # (N, 3)
    camera_positions = -np.einsum('nji,nj->ni', R_w2c_all, t_w2c)  # (N, 3) cam_pos = -R.T @ t
    floor_normal = _orient_floor_normal(
        floor_normal, floor_centroid, all_points, camera_positions=camera_positions
    )

    R, t = _build_R_t_from_floor(world_points, floor_normal, floor_centroid, wall_normal_1=None)

    # 质量检查: R[2,2] = floor_normal[2], 理想值 ≈ ±1.0
    # |R[2,2]| < quality_threshold 说明 floor 法线偏离竖直角度过大, 对齐失败
    if abs(R[2, 2]) < quality_threshold:
        return np.eye(3), np.zeros(3), {
            'reason': f'geocalib_poor_alignment R[2,2]={R[2,2]:.4f}',
            'method': 'geocalib',
            'gravity_world': gravity_world.tolist(),
            'floor_normal': floor_normal.tolist(),
            'floor_centroid': floor_centroid.tolist(),
            'n_frames': len(vecs),
            'n_inliers': n_inliers,
            'per_frame': per_frame_info,
            'extrinsics_convention': 'w2c (world→camera)',
            'gravity_transform': 'grav_world = R_w2c.T @ grav_cam (= R_c2w @ grav_cam, camera→world)',
            'camera_position_transform': 'cam_pos = -R_w2c.T @ t_w2c (w2c 的 t 不是相机位置)',
        }

    return R, t, {
        'method': 'geocalib',
        'gravity_world': gravity_world.tolist(),
        'floor_normal': floor_normal.tolist(),
        'floor_centroid': floor_centroid.tolist(),
        'n_frames': len(vecs),
        'n_inliers': n_inliers,
        'per_frame': per_frame_info,
        'extrinsics_convention': 'w2c (world→camera)',
        'gravity_transform': 'grav_world = R_w2c.T @ grav_cam (= R_c2w @ grav_cam, camera→world)',
        'camera_position_transform': 'cam_pos = -R_w2c.T @ t_w2c (w2c 的 t 不是相机位置)',
    }


def align_vggt_predictions(predictions, R, t):
    '''
    Align the VGGt predictions to the room coordinate system using the given rotation and translation.
    
    Args:
    - predictions: dictionary containing VGGt predictions.
    - R: numpy array of shape (3, 3) representing the rotation matrix to align the scene to the room coordinate system.
    - t: numpy array of shape (3,) representing the translation vector to align the scene to the room coordinate system.
    Returns:
    - predictions: dictionary containing the aligned VGGt predictions.
    '''

    # Update extrinsic matrices in predictions
    c2w_old = predictions["extrinsics"]  # shape: (N, 4, 4)
    R_c2w_old = c2w_old[:, :3, :3]      # shape: (N, 3, 3)
    t_c2w_old = c2w_old[:, :3, 3]       # shape: (N, 3)
    R_c2w_new = R_c2w_old @ R.T           # shape: (N, 3, 3)
    t_c2w_new = t_c2w_old - (R_c2w_new @ t)  # shape: (N, 3)
    predictions["extrinsics"][:, :3, :3] = R_c2w_new
    predictions["extrinsics"][:, :3, 3] = t_c2w_new

    # update world points in predictions
    predictions['world_points'] = predictions['world_points'] @ R.T + t

    # update pcd
    predictions['point_cloud_data'].apply_transform(np.vstack([np.hstack([R, t.reshape(3, 1)]), [0, 0, 0, 1]]))
    return predictions

def get_optimal_view_frame_id(world_points, instance_masks, motion_threshold=0.10, world_points_conf=None):
    '''
    Get the optimal view frame id for each instance.
    Uses robust motion detection: computes per-frame centroid displacement,
    then uses median displacement to distinguish true motion from VGGT drift.

    Args:
        world_points: numpy array of shape (T, H, W, 3)
        instance_masks: list of dicts with 'frame_id' and 'mask'
        motion_threshold: median displacement threshold (meters) for dynamic classification
        world_points_conf: optional numpy array of shape (T, H, W), confidence weights (≥1.05)
    Returns:
        (optimal_frame_id, is_dynamic, motion_info)
        - optimal_frame_id: int
        - is_dynamic: bool
        - motion_info: dict with median_disp, max_disp, first_valid_frame, num_valid_frames
    '''
    centroids = []
    for instance_mask in instance_masks:
        frame_id = instance_mask['frame_id']
        mask = instance_mask['mask']
        pointmap = world_points[frame_id]
        valid = mask > 0
        if not np.any(valid):
            centroids.append((frame_id, None))
            continue
        pts = pointmap[valid]
        finite = np.all(np.isfinite(pts), axis=-1)
        if not np.any(finite):
            centroids.append((frame_id, None))
            continue
        # 置信度加权质心
        if world_points_conf is not None:
            confs = world_points_conf[frame_id][valid]
            # 自适应置信度阈值: 取 conf 的 10th percentile
            conf_thresh = np.percentile(confs[finite], 10)
            conf_mask = finite & (confs > conf_thresh)
            if not np.any(conf_mask):
                conf_mask = finite
            weights = confs[conf_mask]
            w_pts = pts[conf_mask]
            centroid = np.average(w_pts, axis=0, weights=weights)
        else:
            centroid = np.mean(pts[finite], axis=0)
        centroids.append((frame_id, centroid))

    valid_centroids = [(fid, c) for fid, c in centroids if c is not None]
    num_valid_frames = len(valid_centroids)

    if num_valid_frames < 2:
        first_valid_frame = valid_centroids[0][0] if valid_centroids else instance_masks[0]['frame_id']
        motion_info = {'median_disp': 0.0, 'max_disp': 0.0,
                       'first_valid_frame': first_valid_frame, 'num_valid_frames': num_valid_frames}
        return first_valid_frame, False, motion_info

    # 时间平滑: 中值滤波 (窗口=5)
    from scipy.signal import medfilt
    centroid_arr = np.array([c for _, c in valid_centroids])
    if len(centroid_arr) >= 5:
        smoothed = np.zeros_like(centroid_arr)
        for axis in range(3):
            smoothed[:, axis] = medfilt(centroid_arr[:, axis], kernel_size=5)
        valid_centroids = [(valid_centroids[i][0], smoothed[i]) for i in range(len(valid_centroids))]

    consecutive_disps = []
    for i in range(1, len(valid_centroids)):
        disp = np.linalg.norm(valid_centroids[i][1] - valid_centroids[i-1][1])
        consecutive_disps.append(disp)

    disps = np.array(consecutive_disps)

    # MAD 离群点剔除
    med = np.median(disps)
    mad = np.median(np.abs(disps - med))
    if mad > 1e-6:
        z_scores = 0.6745 * (disps - med) / mad
        inlier_mask = np.abs(z_scores) < 3.5
        clean_disps = disps[inlier_mask]
    else:
        clean_disps = disps

    median_disp = float(np.median(clean_disps)) if len(clean_disps) > 0 else 0.0
    max_disp = float(np.max(clean_disps)) if len(clean_disps) > 0 else 0.0
    first_valid_frame = valid_centroids[0][0]

    is_dynamic = median_disp > motion_threshold

    motion_info = {
        'median_disp': round(median_disp, 4),
        'max_disp': round(max_disp, 4),
        'first_valid_frame': first_valid_frame,
        'num_valid_frames': num_valid_frames,
    }

    # 无论动态/静态, 都用最大面积帧生成 mesh (动态物体在 mainv2 中再调整位置/姿态到 first_valid_frame)
    optimal_frame_id = -1
    max_area = 0
    for instance_mask in instance_masks:
        frame_id = instance_mask['frame_id']
        mask = instance_mask['mask']
        pointmap = world_points[frame_id]
        area = compute_surface_area_from_pointmap(pointmap, mask)
        if area > max_area:
            max_area = area
            optimal_frame_id = frame_id

    if optimal_frame_id < 0:
        optimal_frame_id = first_valid_frame

    return optimal_frame_id, is_dynamic, motion_info

def get_walls_info(world_points, wall_masks):
    '''
    Get the wall info from the world points and wall masks in the aligned room coordinate system. 
    Args:
        world_points: numpy array of shape (T, H, W, 3) representing the 3D coordinates of each pixel in each frame.
        wall_masks: list of dictionaries containing 'frame_id' and 'mask' for detected walls.
    Returns:   A list of dictionaries containing wall info, each dictionary contains:
        {
            'axis': 'x' or 'y', the axis of the wall,
            'position': float, the position of the wall along the axis,
            'span': tuple of two floats, the start and end position of the wall along the other axis
        }
    '''
    wall_candidates = []
    for wall_mask in wall_masks:
        frame_id = wall_mask['frame_id']
        mask = wall_mask['mask']
        pointmap = world_points[frame_id]  # shape (H, W, 3)
        plane_info = get_plane_info(pointmap, mask)
        normal = plane_info['normal']
        mean_distance = plane_info['mean_distance']
        # filter out the wall planes with large mean distance (indicating poor plane fitting and likely wrong segmentation)
        if mean_distance > 0.1:
            continue
        # We only consider the walls that are roughly vertical (normal vector close to x or y axis)
        if np.abs(np.dot(normal, np.array([1, 0, 0]))) > np.cos(np.radians(10)):
            axis = 'x'
            position = np.mean(pointmap[mask][:, 0])
            other_axis_coords = pointmap[mask][:, 1]
            span = (np.min(other_axis_coords), np.max(other_axis_coords))
        elif np.abs(np.dot(normal, np.array([0, 1, 0]))) > np.cos(np.radians(10)):
            axis = 'y'
            position = np.mean(pointmap[mask][:, 1])
            other_axis_coords = pointmap[mask][:, 0]
            span = (np.min(other_axis_coords), np.max(other_axis_coords))
        else:
            continue
        wall_candidates.append({
            'axis': axis,
            'position': float(position),
            'span': (float(span[0]), float(span[1])),
        })

    # Cluster the detected walls based on their axis and position to get the final wall info
    if len(wall_candidates) == 0:
        return []

    all_points = world_points.reshape(-1, 3)
    x_threshold = (np.max(all_points[:, 0]) - np.min(all_points[:, 0])) / 10.0
    y_threshold = (np.max(all_points[:, 1]) - np.min(all_points[:, 1])) / 10.0

    clustered_walls = []
    for axis in ('x', 'y'):
        axis_candidates = [w for w in wall_candidates if w['axis'] == axis]
        if len(axis_candidates) == 0:
            continue

        axis_candidates.sort(key=lambda w: w['position'])
        threshold = x_threshold if axis == 'x' else y_threshold

        clusters = []
        current_cluster = [axis_candidates[0]]
        for candidate in axis_candidates[1:]:
            if abs(candidate['position'] - current_cluster[-1]['position']) < threshold:
                current_cluster.append(candidate)
            else:
                clusters.append(current_cluster)
                current_cluster = [candidate]
        clusters.append(current_cluster)

        for cluster in clusters:
            position = float(np.mean([w['position'] for w in cluster]))
            span_start = float(np.min([w['span'][0] for w in cluster]))
            span_end = float(np.max([w['span'][1] for w in cluster]))
            clustered_walls.append({
                'axis': axis,
                'position': position,
                'span': (span_start, span_end),
            })

    return clustered_walls