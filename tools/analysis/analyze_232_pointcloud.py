"""
分析 232 output 的点云质量对坐标系对齐的实际影响
"""
import numpy as np
import trimesh
import os
import sys
import cv2
import json

OUTPUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/232'

print("=" * 80)
print("232 场景点云质量实际影响分析")
print("=" * 80)

# 1. 加载点云
ply_path = os.path.join(OUTPUT_DIR, 'point_cloud.ply')
pcd = trimesh.load(ply_path)
vertices = pcd.vertices
print(f"\n[1] 点云基本信息")
print(f"    总点数: {len(vertices)}")
print(f"    X 范围: [{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
print(f"    Y 范围: [{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
print(f"    Z 范围: [{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")

# 2. 分析点云的统计分布 - 看看有没有明显的离群点
print(f"\n[2] 点云统计分布")
for axis, name in enumerate(['X', 'Y', 'Z']):
    vals = vertices[:, axis]
    p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
    iqr = p75 - p25
    print(f"    {name}: median={p50:.3f}, IQR={iqr:.3f}, P5={p5:.3f}, P95={p95:.3f}")

# 3. 检查是否有明显的平面结构（墙壁/地板）
print(f"\n[3] 平面结构检测（PCA 分析整个点云）")
centered = vertices - vertices.mean(axis=0)
cov = np.dot(centered.T, centered) / len(centered)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
eigenvalues = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]
total_var = eigenvalues.sum()
print(f"    特征值: {eigenvalues}")
print(f"    方差占比: {[f'{v/total_var:.4f}' for v in eigenvalues]}")
print(f"    主方向1: {eigenvectors[:,0]}")
print(f"    主方向2: {eigenvectors[:,1]}")
print(f"    主方向3: {eigenvectors[:,2]}")

# 如果最小特征值占比很小，说明有一个平面结构
min_ratio = eigenvalues[2] / total_var
if min_ratio < 0.01:
    print(f"    ✅ 最小特征值占比 {min_ratio:.4f} < 0.01 → 存在明显平面结构（可能是地板）")
elif min_ratio < 0.05:
    print(f"    ⚠️ 最小特征值占比 {min_ratio:.4f} → 平面结构不太明显")
else:
    print(f"    ❌ 最小特征值占比 {min_ratio:.4f} > 0.05 → 没有明显平面结构，点云混乱")

# 4. 加载 extrinsics 检查相机轨迹
print(f"\n[4] 相机轨迹分析")
ext_dir = os.path.join(OUTPUT_DIR, 'extrinsics')
ext_files = sorted([f for f in os.listdir(ext_dir) if f.endswith('.txt')])
camera_positions = []
for ef in ext_files:
    ext = np.loadtxt(os.path.join(ext_dir, ef))
    # c2w format: camera_pos = -R^T @ t
    R = ext[:3, :3]
    t = ext[:3, 3]
    cam_pos = -R.T @ t
    camera_positions.append(cam_pos)
camera_positions = np.array(camera_positions)
print(f"    帧数: {len(camera_positions)}")
print(f"    相机 X 范围: [{camera_positions[:,0].min():.3f}, {camera_positions[:,0].max():.3f}]")
print(f"    相机 Y 范围: [{camera_positions[:,1].min():.3f}, {camera_positions[:,1].max():.3f}]")
print(f"    相机 Z 范围: [{camera_positions[:,2].min():.3f}, {camera_positions[:,2].max():.3f}]")

# 计算相邻帧相机位移
displacements = np.linalg.norm(np.diff(camera_positions, axis=0), axis=1)
print(f"    相邻帧平均位移: {displacements.mean():.4f}m")
print(f"    相邻帧最大位移: {displacements.max():.4f}m")
print(f"    相邻帧位移标准差: {displacements.std():.4f}m")

# 检查是否有突变帧（可能对应动态物体）
sudden_frames = np.where(displacements > displacements.mean() + 3 * displacements.std())[0]
if len(sudden_frames) > 0:
    print(f"    ⚠️ 检测到 {len(sudden_frames)} 个突变帧: {sudden_frames.tolist()}")
    print(f"    这些帧的位移: {displacements[sudden_frames]}")
else:
    print(f"    ✅ 没有检测到明显的相机轨迹突变")

# 5. 加载深度图分析
print(f"\n[5] 深度图一致性分析")
depth_dir = os.path.join(OUTPUT_DIR, 'depth')
depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])
# 采样几帧
sample_indices = [0, len(depth_files)//4, len(depth_files)//2, 3*len(depth_files)//4, len(depth_files)-1]
depth_stats = []
for idx in sample_indices:
    depth = cv2.imread(os.path.join(depth_dir, depth_files[idx]), cv2.IMREAD_UNCHANGED)
    depth_m = depth.astype(np.float32) / 1000.0  # 还原为米
    valid = depth_m > 0
    if valid.any():
        depth_stats.append({
            'frame': idx,
            'mean': depth_m[valid].mean(),
            'std': depth_m[valid].std(),
            'min': depth_m[valid].min(),
            'max': depth_m[valid].max(),
            'valid_ratio': valid.mean()
        })

print(f"    采样帧深度统计:")
for s in depth_stats:
    print(f"    Frame {s['frame']}: mean={s['mean']:.3f}m, std={s['std']:.3f}m, "
          f"range=[{s['min']:.3f}, {s['max']:.3f}]m, valid={s['valid_ratio']:.2%}")

# 6. 检查 intrinsic
intrinsic = np.loadtxt(os.path.join(OUTPUT_DIR, 'intrinsic.txt'))
print(f"\n[6] 相机内参")
print(f"    fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")
print(f"    cx={intrinsic[0,2]:.1f}, cy={intrinsic[1,2]:.1f}")

# 7. 检查 final_scene_basic.glb
print(f"\n[7] 最终场景分析")
glb_path = os.path.join(OUTPUT_DIR, 'final_scene_basic.glb')
if os.path.exists(glb_path):
    scene = trimesh.load(glb_path)
    if hasattr(scene, 'geometry'):
        for name, geom in scene.geometry.items():
            if hasattr(geom, 'bounds'):
                bmin, bmax = geom.bounds
                extents = bmax - bmin
                print(f"    {name}: 中心={((bmin+bmax)/2).round(3)}, 尺寸={extents.round(3)}")
    else:
        print(f"    单个 mesh, bounds: {scene.bounds}")

# 8. 检查精修后的场景
refined_path = os.path.join(OUTPUT_DIR, 'final_scene_refined.glb')
if os.path.exists(refined_path):
    print(f"\n[8] 精修后场景分析")
    scene_r = trimesh.load(refined_path)
    if hasattr(scene_r, 'geometry'):
        for name, geom in scene_r.geometry.items():
            if hasattr(geom, 'bounds'):
                bmin, bmax = geom.bounds
                extents = bmax - bmin
                print(f"    {name}: 中心={((bmin+bmax)/2).round(3)}, 尺寸={extents.round(3)}")

# 9. 检查 relations_refined.json
json_path = os.path.join(OUTPUT_DIR, 'relations_refined.json')
if os.path.exists(json_path):
    import json
    with open(json_path, 'r') as f:
        relations = json.load(f)
    print(f"\n[9] 物体关系 JSON")
    print(f"    内容: {json.dumps(relations, indent=2, ensure_ascii=False)}")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
