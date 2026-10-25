import os
import cv2
import numpy as np
import torch
import trimesh
try:
    import open3d as o3d
except Exception:  # hawor 环境未装 open3d; 用 numpy/scipy 兜底
    o3d = None


def _overlap_ratio_numpy(source_pts, target_pts, dist_thresh=None):
    """get_overlap_ratio 的 numpy 兜底实现 (无 open3d 时)。

    与 open3d 版逻辑一致:
      1. bbox 不相交 → 0
      2. overlap = source 中到 target 最近点距离 < 固定阈值 的比例
    用 scipy 的 KDTree 算最近邻, 复杂度 O((N+M) log M), 比 open3d 的
    compute_point_cloud_distance 慢一些但结果等价。
    """
    if dist_thresh is None:
        dist_thresh = OVERLAP_DIST_THRESH
    src = np.asarray(source_pts, dtype=np.float64)
    tgt = np.asarray(target_pts, dtype=np.float64)
    if len(src) == 0 or len(tgt) == 0:
        return 0.0
    s_min, s_max = src.min(axis=0), src.max(axis=0)
    t_min, t_max = tgt.min(axis=0), tgt.max(axis=0)
    if (s_max < t_min).any() or (s_min > t_max).any():
        return 0.0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(tgt).query(src, k=1)
    return float(np.sum(d < dist_thresh) / len(src))

class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootX] = rootY

# 重叠判据的固定距离阈值 (米)。
# 为什么不能用"3 × 平均最近邻距离": 那个阈值随点云密度自缩放 —— 点越密,
# 最近邻距离越小, 阈值越小。两个不同帧的 mask 即使罩着同一个物体, VGGT 的
# 帧间重建漂移 (厘米级) 也会让几乎所有点落在那个被压到毫米级的阈值之外。
# 实测 weixin_v10: button[0] 与 stone[0] 的 3D 中心只差 5cm、尺寸相当,
# 明显是同一物体被 VLM 用两个名字各报一次, 但密度缩放阈值只算出
# ov=0.042/0.011 → 跨类去重没合并, 于是一个物体变成两个实例。
# 30mm 是实测选值: 该阈值下 button/stone 为 44%/95% (正确合并), 而
# "物体已移动"(button[0] vs button[1]) 仍是 0%/0% (正确不合并),
# "大云包住小物体"(stone vs device) 仍是 100%/11% 不对称 (and 规则挡住)。
OVERLAP_DIST_THRESH = 0.03


def get_overlap_ratio(source_pts, target_pts, dist_thresh=None):
    """
    Compute overlap ratio of source in target.
    Args:
        source_pts: numpy array of shape (N1, 3)
        target_pts: numpy array of shape (N2, 3)
        dist_thresh: 判定"靠近"的距离阈值 (米)。None 用 OVERLAP_DIST_THRESH。
    Returns:
        Overlap_ratio of source pcd in target pcd
    """
    if dist_thresh is None:
        dist_thresh = OVERLAP_DIST_THRESH
    if o3d is None:
        return _overlap_ratio_numpy(source_pts, target_pts, dist_thresh)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts)
    
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts)

    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        return 0.0
    
    # bbox judgement
    s_min = source_pcd.get_min_bound()
    s_max = source_pcd.get_max_bound()
    t_min = target_pcd.get_min_bound()
    t_max = target_pcd.get_max_bound()
    
    if (s_max[0] < t_min[0] or s_min[0] > t_max[0] or  # X
        s_max[1] < t_min[1] or s_min[1] > t_max[1] or  # Y
        s_max[2] < t_min[2] or s_min[2] > t_max[2]):   # Z
        return 0.0
    
    # 用固定物理阈值, 不用"3 × 平均最近邻距离" (密度自缩放, 见上方常量注释)
    dists = source_pcd.compute_point_cloud_distance(target_pcd)
    dists = np.asarray(dists)

    overlap_count = np.sum(dists < dist_thresh)
    return overlap_count / len(source_pcd.points)

def self_category_deduplicate(category_masks, world_points, world_points_conf, conf_k=30, overlap_thre=0.3, category_name=""):
    '''
    Deduplication inside a single category (only 3D point cloud overlap_ratio)
    Args:
        category_masks: A list of segmented masks returned by segment_and_track function
        world_points: numpy array of shape (S, H, W, 3)
        world_points_conf: numpy array of shape (S, H, W)
        conf_k: Only keep top conf_k percent confidence 3D points
        overlap_thre: Merge instance if overlap ratio exceeds overlap_thre
        category_name: Category name for logging
    Returns:
        deduplicated_category_masks: A list of deduplicated instance masks
    '''
    if not category_masks:
        return []

    print(f"  [类内去重] {category_name}: {len(category_masks)} 个原始实例, overlap_thre={overlap_thre}")

    # back-project instance masks into pointcloud 
    instance_point_arrays = [] 
    valid_indices = [] 

    for idx, instance_frames in enumerate(category_masks):
        obj_pts_list = []
        obj_conf_list = []

        for frame_data in instance_frames:
            frame_id = frame_data['frame_id']
            mask = frame_data['mask']
            
            # 边界检查
            if frame_id >= world_points.shape[0]:
                continue
                
            pts_frame = world_points[frame_id]
            conf_frame = world_points_conf[frame_id]
            
            valid_pixels = mask > 0
            if not np.any(valid_pixels):
                continue
                
            v_pts = pts_frame[valid_pixels]
            v_conf = conf_frame[valid_pixels]
            
            if len(v_conf) > 0:
                obj_pts_list.append(v_pts)
                obj_conf_list.append(v_conf)
        
        # Global Filtering per Instance
        if obj_pts_list:
            full_pts = np.concatenate(obj_pts_list, axis=0)
            full_conf = np.concatenate(obj_conf_list, axis=0)
            
            # confidence threshold
            thresh = 0.0
            if conf_k > 0:
                thresh = np.percentile(full_conf, conf_k)
                
            conf_mask = (full_conf >= thresh) & (full_conf > 1e-5)
            
            if np.any(conf_mask):
                final_pts = full_pts[conf_mask]
                # point num filter
                if len(final_pts) >= 50: 
                    instance_point_arrays.append(final_pts)
                    valid_indices.append(idx)

    if len(instance_point_arrays) <= 1:
        return category_masks

    # Union-Find Clustering
    uf = UnionFind(range(len(instance_point_arrays)))
    n = len(instance_point_arrays)

    merge_details = []
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            
            ov1 = get_overlap_ratio(instance_point_arrays[i], instance_point_arrays[j])
            ov2 = get_overlap_ratio(instance_point_arrays[j], instance_point_arrays[i])

            if ov1 >= overlap_thre or ov2 >= overlap_thre:
                uf.union(i, j)
                reason = []
                if ov1 >= overlap_thre:
                    reason.append(f"ov1={ov1:.3f}>={overlap_thre}")
                if ov2 >= overlap_thre:
                    reason.append(f"ov2={ov2:.3f}>={overlap_thre}")
                merge_details.append(f"    inst_{valid_indices[i]} + inst_{valid_indices[j]} ← {', '.join(reason)}")
                
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(valid_indices[i])

    # get deduplicated category masks
    deduplicated_category_masks = []
    for group_indices in groups.values():
        merged_frames = {}
        for orig_idx in group_indices:
            original_instance = category_masks[orig_idx]
            for frame_data in original_instance:
                fid = frame_data['frame_id']
                msk = frame_data['mask']
                if fid not in merged_frames:
                    merged_frames[fid] = np.zeros_like(msk)
                merged_frames[fid] = np.logical_or(merged_frames[fid], msk)
        
        merged_instance_list = []
        for fid in sorted(merged_frames.keys()):
            merged_instance_list.append({
                'frame_id': fid,
                'mask': merged_frames[fid]
            })
            
        deduplicated_category_masks.append(merged_instance_list)

    print(f"  -> {category_name}: {len(category_masks)} instances → {len(deduplicated_category_masks)} instances (类内去重)")
    if merge_details:
        for d in merge_details:
            print(d)
    return deduplicated_category_masks


def _build_candidates(all_masks, world_points, world_points_conf, conf_k=30):
    """从 all_masks 构建 candidate 列表，每个 candidate 包含 3D 点云、时间信息等。"""
    all_candidates = []
    for cat_name, instances_list in all_masks.items():
        for inst_idx, instance_frames in enumerate(instances_list):
            frame_ids = [f['frame_id'] for f in instance_frames]
            obj_pts_list = []
            obj_conf_list = []
            for frame_data in instance_frames:
                frame_id = frame_data['frame_id']
                mask = frame_data['mask']
                if frame_id >= world_points.shape[0]:
                    continue
                pts_frame = world_points[frame_id]
                conf_frame = world_points_conf[frame_id]
                valid_pixels = mask > 0
                if not np.any(valid_pixels):
                    continue
                v_pts = pts_frame[valid_pixels]
                v_conf = conf_frame[valid_pixels]
                if len(v_conf) > 0:
                    obj_pts_list.append(v_pts)
                    obj_conf_list.append(v_conf)
            if obj_pts_list:
                full_pts = np.concatenate(obj_pts_list, axis=0)
                full_conf = np.concatenate(obj_conf_list, axis=0)
                thresh = 0.0
                if conf_k > 0:
                    thresh = np.percentile(full_conf, conf_k)
                conf_mask = (full_conf >= thresh) & (full_conf > 1e-5)
                if np.any(conf_mask):
                    final_pts = full_pts[conf_mask]
                    if len(final_pts) >= 50:
                        all_candidates.append({
                            "category": cat_name,
                            "inst_idx": inst_idx,
                            "points": final_pts,
                            "original_masks": instance_frames,
                            "frame_ids": set(frame_ids),
                            "time_start": min(frame_ids),
                            "time_end": max(frame_ids),
                            "time_total": len(frame_ids),
                        })
    return all_candidates


def _merge_candidates_to_masks(all_candidates, uf, N):
    """根据 Union-Find 分组结果，合并 candidate 的 masks 为 all_masks dict。"""
    final_groups = {}
    for i in range(N):
        root = uf.find(i)
        if root not in final_groups:
            final_groups[root] = []
        final_groups[root].append(i)

    deduplicated_all_masks = {}
    for group_indices in final_groups.values():
        # 选择长寿命实例的 category
        target_idx = max(group_indices, key=lambda x: all_candidates[x]['time_total'])
        cat = all_candidates[target_idx]["category"]

        merged_masks_dict = {}
        for idx in group_indices:
            for frame_data in all_candidates[idx]["original_masks"]:
                fid = frame_data['frame_id']
                msk = frame_data['mask']
                if fid not in merged_masks_dict:
                    merged_masks_dict[fid] = np.zeros_like(msk)
                merged_masks_dict[fid] = np.logical_or(merged_masks_dict[fid], msk)

        merged_masks_list = []
        for fid in sorted(merged_masks_dict.keys()):
            merged_masks_list.append({'frame_id': fid, 'mask': merged_masks_dict[fid]})

        if len(merged_masks_list) < 2:
            continue
        if cat not in deduplicated_all_masks:
            deduplicated_all_masks[cat] = []
        deduplicated_all_masks[cat].append(merged_masks_list)

    return deduplicated_all_masks


def cross_category_deduplicate(all_masks, world_points, world_points_conf, conf_k=30, overlap_thre=0.3, **kwargs):
    '''
    Cross-category deduplication (only different categories, only 3D overlap_ratio).
    Same-category pairs are skipped — they are handled by self_category_deduplicate.
    Dynamic splits and static occlusion are handled by merge_dynamic_splits_all (called separately).

    Args:
        all_masks: dict {category: list[list[dict{frame_id, mask}]]}
        world_points: numpy array of shape (S, H, W, 3)
        world_points_conf: numpy array of shape (S, H, W)
        conf_k: confidence percentile threshold
        overlap_thre: merge threshold for overlap_ratio
    Returns:
        deduplicated_all_masks: dict (same format as input)
    '''
    print(f"Starting cross-category deduplication (only different categories)...")

    all_candidates = _build_candidates(all_masks, world_points, world_points_conf, conf_k)
    if len(all_candidates) <= 1:
        return all_masks

    N = len(all_candidates)

    # 计算 **仅不同类别** 实例对的 overlap
    print(f"  [跨类去重] 计算 {N} 个实例中不同类别的 3D 点云 overlap...")
    overlap_matrix = {}
    for i in range(N):
        for j in range(i + 1, N):
            ci, cj = all_candidates[i], all_candidates[j]
            if ci['category'] == cj['category']:
                continue
            ov1 = get_overlap_ratio(ci['points'], cj['points'])
            ov2 = get_overlap_ratio(cj['points'], ci['points'])
            overlap_matrix[(i, j)] = (ov1, ov2)

    uf = UnionFind(range(N))
    cross_merge_details = []

    for i in range(N):
        for j in range(i + 1, N):
            ci, cj = all_candidates[i], all_candidates[j]

            # 同类实例跳过，已由 self_category_deduplicate 处理
            if ci['category'] == cj['category']:
                continue

            if uf.find(i) == uf.find(j):
                continue

            key = (i, j)
            if key not in overlap_matrix:
                continue

            ov1, ov2 = overlap_matrix[key]

            # 必须双向都超阈值才算"同一物体被检出成两个类别"。
            #
            # 为什么从 or 改成 and: 单向成立会误吞真实的相邻物体。
            # 小物体叠在大物体上时, 小物体点云有很大比例落在大物体附近
            # (ov_small→big 很高), 但反过来大物体只有很小比例靠近小物体
            # (ov_big→small 很低) —— 用 or 就会把两个不同的物体合并掉。
            # 实测 basic_pick_place_66: duck_0 + toy_2 的 ov1=0.113 / ov2=0.784,
            # 靠 or 被合并, 而真实情况是两个独立物体。
            # 真重复的特征是双向都接近 1: 同场景 block_0+cube_0 是 1.000/0.999,
            # bowl_0+cup_0 是 0.936/0.967, 改为 and 后这些仍会正确合并。
            if ov1 >= overlap_thre and ov2 >= overlap_thre:
                uf.union(i, j)
                cross_merge_details.append(
                    f"    {ci['category']}_{ci['inst_idx']} + {cj['category']}_{cj['inst_idx']} ← "
                    f"overlap: ov1={ov1:.3f}, ov2={ov2:.3f}"
                )

    deduplicated_all_masks = _merge_candidates_to_masks(all_candidates, uf, N)

    total_before = sum(len(v) for v in all_masks.values())
    total_after = sum(len(v) for v in deduplicated_all_masks.values())
    print(f"  -> 跨类去重: {total_before} instances → {total_after} instances")
    if cross_merge_details:
        for d in cross_merge_details:
            print(d)
    return deduplicated_all_masks


def merge_dynamic_splits_all(all_masks, world_points, world_points_conf, conf_k=30,
                              overlap_thre=0.3, subset_thre=0.5, dynamic_ov_thre=0.2,
                              lifespan_thre=0.5, gap_thre=10):
    '''
    全量动态分裂 + 静态遮挡合并（不分同类/跨类，所有实例统一处理）。

    三种合并条件（按优先级）:
      1. 静态重叠: ov1 >= overlap_thre or ov2 >= overlap_thre → 合并
      2. 静态遮挡: 单向 overlap >= subset_thre（子集关系）AND 子集方帧数更少 → 合并
      3. 动态分裂: lifespan_ratio < lifespan_thre AND time_gap <= gap_thre
                   AND 衔接帧 overlap >= dynamic_ov_thre → 合并

    Args:
        all_masks: dict {category: list[list[dict{frame_id, mask}]]}
        world_points: numpy array of shape (S, H, W, 3)
        world_points_conf: numpy array of shape (S, H, W)
        conf_k: confidence percentile threshold
        overlap_thre: bidirectional overlap merge threshold
        subset_thre: one-directional overlap threshold for static occlusion
        dynamic_ov_thre: junction overlap threshold for dynamic splits
        lifespan_thre: max lifespan_ratio for dynamic split candidates
        gap_thre: max time gap (frames) for dynamic split candidates
    Returns:
        deduplicated_all_masks: dict (same format as input)
    '''
    print(f"Starting merge_dynamic_splits_all (all instances, static occlusion + dynamic splits)...")

    all_candidates = _build_candidates(all_masks, world_points, world_points_conf, conf_k)
    if len(all_candidates) <= 1:
        return all_masks

    N = len(all_candidates)

    # 预计算所有实例的 overlap matrix
    print(f"  [动态/遮挡合并] 计算 {N} 个实例的 3D 点云 overlap...")
    overlap_matrix = {}
    for i in range(N):
        for j in range(i + 1, N):
            ci, cj = all_candidates[i], all_candidates[j]
            ov1 = get_overlap_ratio(ci['points'], cj['points'])
            ov2 = get_overlap_ratio(cj['points'], ci['points'])
            overlap_matrix[(i, j)] = (ov1, ov2)

    uf = UnionFind(range(N))
    merge_details = []

    for i in range(N):
        for j in range(i + 1, N):
            if uf.find(i) == uf.find(j):
                continue

            ci, cj = all_candidates[i], all_candidates[j]
            key = (i, j)
            if key not in overlap_matrix:
                continue

            ov1, ov2 = overlap_matrix[key]
            label_i = f"{ci['category']}_{ci['inst_idx']}"
            label_j = f"{cj['category']}_{cj['inst_idx']}"

            # ===== 1. 静态重叠 =====
            # 同 cross_category_deduplicate: 改 or → and。
            # 单向成立会把"小物体叠在大物体上"误判成同一物体。
            # 非对称的情形交给下面的规则 2 (静态遮挡) 处理 —— 那里用
            # max_ov >= subset_thre AND min_ov < overlap_thre 并附加存活时长证据,
            # 比单纯的 or 判据可靠。
            if ov1 >= overlap_thre and ov2 >= overlap_thre:
                uf.union(i, j)
                merge_details.append(f"    {label_i} + {label_j} ← static_overlap: ov1={ov1:.3f}, ov2={ov2:.3f}")
                continue

            # ===== 2. 静态遮挡（单向子集关系）=====
            # 短命实例的 3D 点云大部分在长命实例内部 → 说明是同一物体被遮挡后的碎片
            max_ov = max(ov1, ov2)
            min_ov = min(ov1, ov2)
            if max_ov >= subset_thre and min_ov < overlap_thre:
                # ov1 高 → i 的点在 j 内部 → i 可能是 j 的碎片
                # ov2 高 → j 的点在 i 内部 → j 可能是 i 的碎片
                if ov1 >= subset_thre and ci['time_total'] < cj['time_total']:
                    uf.union(i, j)
                    merge_details.append(
                        f"    {label_i} + {label_j} ← static_occlusion(subset): "
                        f"ov(i→j)={ov1:.3f}>={subset_thre}, i帧数({ci['time_total']})<j帧数({cj['time_total']})"
                    )
                    continue
                elif ov2 >= subset_thre and cj['time_total'] < ci['time_total']:
                    uf.union(i, j)
                    merge_details.append(
                        f"    {label_i} + {label_j} ← static_occlusion(subset): "
                        f"ov(j→i)={ov2:.3f}>={subset_thre}, j帧数({cj['time_total']})<i帧数({ci['time_total']})"
                    )
                    continue

            # ===== 3. 动态分裂 =====
            lifespan_ratio = min(ci['time_total'], cj['time_total']) / max(ci['time_total'], cj['time_total'])
            if lifespan_ratio >= lifespan_thre:
                continue

            time_gap = min(
                abs(ci['time_start'] - cj['time_end']),
                abs(cj['time_start'] - ci['time_end'])
            )
            if time_gap > gap_thre:
                continue

            # 在衔接帧范围内计算点云 overlap
            merge_window = range(
                max(ci['time_start'], cj['time_start']) - time_gap,
                min(ci['time_end'], cj['time_end']) + time_gap + 1
            )
            merge_window = [f for f in merge_window if 0 <= f < world_points.shape[0]]

            if not merge_window:
                continue

            joint_ov1 = 0.0
            joint_ov2 = 0.0
            joint_count = 0

            for fid in merge_window:
                mask_i = next((f['mask'] for f in ci['original_masks'] if f['frame_id'] == fid), None)
                mask_j = next((f['mask'] for f in cj['original_masks'] if f['frame_id'] == fid), None)

                if mask_i is None or mask_j is None:
                    continue

                pts_i = world_points[fid][mask_i > 0]
                pts_j = world_points[fid][mask_j > 0]

                if len(pts_i) < 10 or len(pts_j) < 10:
                    continue

                joint_ov1 += get_overlap_ratio(pts_i, pts_j)
                joint_ov2 += get_overlap_ratio(pts_j, pts_i)
                joint_count += 1

            if joint_count == 0:
                continue

            avg_ov1 = joint_ov1 / joint_count
            avg_ov2 = joint_ov2 / joint_count

            if avg_ov1 >= dynamic_ov_thre or avg_ov2 >= dynamic_ov_thre:
                uf.union(i, j)
                merge_details.append(
                    f"    {label_i} + {label_j} ← dynamic_split: "
                    f"lifespan_ratio={lifespan_ratio:.2f}, gap={time_gap}, "
                    f"junction_overlap={avg_ov1:.3f}/{avg_ov2:.3f}"
                )

    deduplicated_all_masks = _merge_candidates_to_masks(all_candidates, uf, N)

    total_before = sum(len(v) for v in all_masks.values())
    total_after = sum(len(v) for v in deduplicated_all_masks.values())
    print(f"  -> 动态/遮挡合并: {total_before} instances → {total_after} instances")
    if merge_details:
        for d in merge_details:
            print(d)
    return deduplicated_all_masks


def _bbox_iou_3d(bbox_a, bbox_b):
    """
    Compute 3D bounding box IoU.
    Args:
        bbox_a: (2, 3) array, [min_corner, max_corner]
        bbox_b: (2, 3) array, [min_corner, max_corner]
    Returns:
        IoU value in [0, 1]
    """
    min_a, max_a = bbox_a
    min_b, max_b = bbox_b
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = np.prod(inter_size)
    vol_a = np.prod(max_a - min_a)
    vol_b = np.prod(max_b - min_b)
    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


def deduplicate_3d_assets(all_instances, all_masks=None, iou_thre=0.3):
    """
    Post-generation 3D mesh deduplication.
    Compares meshes in world space (after applying T) to find and remove duplicates.

    Args:
        all_instances: dict {category: list[dict{"original_mesh": trimesh, "T": (4,4)}]}
        all_masks: dict {category: list[list[dict{frame_id, mask}]]}, synced with all_instances
        iou_thre: Bounding box IoU threshold for considering two meshes as duplicates
    Returns:
        deduplicated_instances: dict (same format), with duplicates removed
        deduplicated_masks: dict (same format as all_masks), synced with deduplicated_instances
        removed_info: list of (category, index, reason) for logging
    """
    print(f"\n🔍 3D Mesh 去重 (post-generation deduplication)...")
    flat_list = []
    for cat, inst_list in all_instances.items():
        for idx, inst in enumerate(inst_list):
            mesh = inst["original_mesh"]
            T = inst["T"]
            transformed = mesh.copy()
            transformed.apply_transform(T)
            centroid = transformed.centroid
            bbox = transformed.bounds
            flat_list.append({
                "category": cat,
                "index": idx,
                "centroid": centroid,
                "bbox": bbox,
                "volume": transformed.volume if transformed.is_watertight else abs(transformed.volume),
                "vertex_count": len(transformed.vertices),
            })

    n = len(flat_list)
    if n <= 1:
        print(f"  Only {n} instance, no deduplication needed.")
        return all_instances, all_masks, []

    uf = UnionFind(range(n))
    mesh_merge_details = []
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            iou = _bbox_iou_3d(flat_list[i]["bbox"], flat_list[j]["bbox"])
            if iou >= iou_thre:
                uf.union(i, j)
                mesh_merge_details.append(
                    f"    {flat_list[i]['category']}_{flat_list[i]['index']} + "
                    f"{flat_list[j]['category']}_{flat_list[j]['index']} ← bbox_iou={iou:.3f}>={iou_thre}"
                )

    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    keep_indices = set()
    removed_info = []
    for group_indices in groups.values():
        if len(group_indices) == 1:
            keep_indices.add(group_indices[0])
            continue
        best_idx = max(group_indices, key=lambda i: flat_list[i]["vertex_count"])
        keep_indices.add(best_idx)
        for idx in group_indices:
            if idx != best_idx:
                info = flat_list[idx]
                best_info = flat_list[best_idx]
                removed_info.append((
                    info["category"], info["index"],
                    f"duplicate of {best_info['category']}_{best_info['index']} "
                    f"(centroid_dist={np.linalg.norm(info['centroid'] - best_info['centroid']):.4f}m)"
                ))

    deduplicated_instances = {}
    deduplicated_masks = {}
    for i in sorted(keep_indices):
        cat = flat_list[i]["category"]
        orig_idx = flat_list[i]["index"]
        if cat not in deduplicated_instances:
            deduplicated_instances[cat] = []
            if all_masks is not None:
                deduplicated_masks[cat] = []
        deduplicated_instances[cat].append(all_instances[cat][orig_idx])
        if all_masks is not None and cat in all_masks:
            deduplicated_masks[cat].append(all_masks[cat][orig_idx])

    total_before = sum(len(v) for v in all_instances.values())
    total_after = sum(len(v) for v in deduplicated_instances.values())
    if removed_info:
        print(f"  ⚠️  Removed {len(removed_info)} duplicate 3D assets:")
        for cat, idx, reason in removed_info:
            print(f"      {cat}_{idx}: {reason}")
    if mesh_merge_details:
        for d in mesh_merge_details:
            print(d)
    print(f"  -> {total_before} instances → {total_after} instances (3D Mesh去重)")
    return deduplicated_instances, deduplicated_masks, removed_info