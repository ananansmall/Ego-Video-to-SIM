#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""静态物体的快速 mesh 生成: 直接从 VGGT 点云构网, 跳过 SAM3D。

动机
----
Stage 3 的 `generate_3d_asset_in_subprocess` 对**每一个**实例都跑一遍 SAM3D
(单实例约 30~60s, 且常因显存被占而 OOM)。但绝大多数场景里只有 1~2 个物体
真的被抓取/操作, 其余都是静态背景。

对这些静态物体, 用 VGGT 点云直接构网即可:
  - 尺度**天然正确** (点云本身是米制世界坐标), 不需要 Stage 3 后面那套
    逐物体点云校准 + guard, 顺带绕开尺度漂移问题;
  - 速度差 2~3 个数量级 (实测凸包 2 万个点 8ms);
  - 不占 GPU, 不再有 OOM。

被抓取/操作的动态物体仍然走 SAM3D (需要完整形状来支撑后续位姿与仿真)。

注意: 这条路生成的 mesh 只适合"静态固定资产", 不保证闭合或完整 ——
被遮挡、薄壁、反光物体可能破面。所以只在"该实例不参与任何 pick/place
交互"时才启用。
"""
import numpy as np

# 静态 mesh 的最大输入点数 (降采样上限, 避免构网耗时失控)
MAX_POINTS = 8000
# 点云点数少于此值则直接放弃构网, 退回凸包
MIN_POINTS = 60


def _to_trimesh(vertices, faces):
    import trimesh
    if len(vertices) == 0 or len(faces) == 0:
        return None
    tm = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64),
                         faces=np.asarray(faces, dtype=np.int64),
                         process=True)
    if len(tm.faces) == 0 or len(tm.vertices) < 4:
        return None
    return tm


def build_mesh_from_pointcloud(points, max_points=MAX_POINTS):
    """从世界坐标点云构建 trimesh (静态物体快速路径)。

    依次尝试:
      1. ball-pivoting  (open3d) —— 非凸物体也能保住形状
      2. alpha shape    (open3d) —— 比 ball pivoting 更兜底一些
      3. 凸包           (trimesh) —— 永远能出结果, 形状最粗糙

    Args:
        points: (N, 3) 世界坐标点云
        max_points: 降采样上限
    Returns:
        trimesh.Trimesh 或 None (点数不足/构网失败)
    """
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < MIN_POINTS:
        return None

    # 降采样: 构网复杂度对点数很敏感
    if len(pts) > max_points:
        sel = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[sel]

    # 去离群点: VGGT 在遮挡边界会有飞点, 会让构网长出尖刺
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd.points) < MIN_POINTS:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        dist = float(np.mean(pcd.compute_nearest_neighbor_distance()))
        for factor in (2.0, 4.0, 8.0):
            mesh = o3d.geometry.TriangleMesh.create_from_ball_pivoting(
                pcd, o3d.utility.DoubleVector([dist * factor]))
            tm = _to_trimesh(np.asarray(mesh.vertices),
                             np.asarray(mesh.triangles))
            if tm is not None:
                return tm
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, dist * 4.0)
        tm = _to_trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        if tm is not None:
            return tm
    except Exception:
        pass

    # 凸包兜底 (不依赖 open3d)
    try:
        import trimesh
        tm = trimesh.convex.convex_hull(pts)
        if tm is not None and len(tm.faces) > 0:
            return tm
    except Exception:
        return None
    return None


def is_planar_pointcloud(points, flat_ratio=0.12, min_extent=0.25):
    """判断点云是否为"大平面/大墙面" —— 只有这类才适合点云构网。

    为什么需要这个判据:
    点云构网 (ball pivoting / 凸包) 只能抓住可见表面。对**有结构的物体**
    会丢掉背面与内部构造 —— 实测 weixin 场景: 桌子是有桌腿的, 点云构网
    只得到一个 1.13x0.90x0.05 的平板, 桌腿全丢, 连带把小物体(耳机盒)
    也挤掉。而真正的大平面 (地板、墙、桌面表面) 用点云构网又快又准。

    Args:
        points: (N, 3) 世界坐标点云
        flat_ratio: 最薄维度 / 最大维度 小于该值才算平面
        min_extent: 最大水平跨度至少要这么大 (否则是小物体, 不算平面)
    Returns:
        bool
    """
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 200:
        return False
    ext = pts.max(axis=0) - pts.min(axis=0)
    if max(ext[0], ext[1]) < min_extent:
        return False                       # 小物体 → 不是"大平面"
    mx = max(ext)
    if mx <= 1e-6:
        return False
    return bool((min(ext) / mx) < flat_ratio)


def build_static_instance(mesh, category, inst_idx):
    """包成与 SAM3D 产物同构的 instance_info。

    关键: 点云已是世界坐标, 所以 T 用单位矩阵 (而 SAM3D 产物需要后续
    的平移/尺度校准才能到位)。
    """
    return {
        "original_mesh": mesh,
        "T": np.eye(4),
        "node_name": f"{category}_{inst_idx}",
        "mesh_source": "pointcloud_fast",   # 标记来源, 便于诊断
    }
