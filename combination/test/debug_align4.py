#!/usr/bin/env python3
"""诊断脚本4: 检查RAS尺度 + 直接在RAS坐标系中验证手-物关系"""

import numpy as np
import trimesh
from glob import glob
import os

R_X = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
RXWORLD_TO_SAPIEN = R_AXIS @ R_X

RAS_DIR = "/home/an/data/ras/my_7mp4_result"
HAWOR_FILE = "/home/an/data/hawor/7/reconstruction/hawor_results_0_113.npz"
GLB_PATH = os.path.join(RAS_DIR, "final_scene.glb")


def load_ras_cameras(ras_dir):
    ext_dir = os.path.join(ras_dir, 'extrinsics')
    ext_files = sorted(glob(os.path.join(ext_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
    cam_positions = []
    R_c2w_list = []
    for f in ext_files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        R_c2w = ext[:3, :3].T
        cam_pos = -R_c2w @ ext[:3, 3]
        cam_positions.append(cam_pos)
        R_c2w_list.append(R_c2w)
    return np.array(cam_positions), np.array(R_c2w_list), ext_files


ras_cam_ydown, ras_R_c2w, ext_files = load_ras_cameras(RAS_DIR)
hawor_data = dict(np.load(HAWOR_FILE, allow_pickle=True))
hawor_cam = hawor_data['t_c2w']
hawor_R_c2w = hawor_data['R_c2w']
pred_trans = hawor_data['pred_trans']
pred_valid = hawor_data['pred_valid']

print("=" * 70)
print("1. RAS 内参和尺度检查")
print("=" * 70)

intrinsic_path = os.path.join(RAS_DIR, 'intrinsic.txt')
if os.path.exists(intrinsic_path):
    intrinsic = np.loadtxt(intrinsic_path)
    print(f"RAS 内参矩阵:\n{intrinsic}")
    fx = intrinsic[0, 0] if intrinsic.shape == (3, 3) else intrinsic[0]
    fy = intrinsic[1, 1] if intrinsic.shape == (3, 3) else intrinsic[1]
    print(f"fx={fx}, fy={fy}")
    if fx > 100:
        print(f"内参是像素单位, fx_px={fx}")
    else:
        print(f"内参可能已归一化")
else:
    print("未找到 intrinsic.txt")

# 检查图片分辨率
from PIL import Image
img_path = os.path.join(RAS_DIR, 'color', '0.jpg')
if os.path.exists(img_path):
    img = Image.open(img_path)
    print(f"图片尺寸: {img.size}")
    if os.path.exists(intrinsic_path):
        intrinsic = np.loadtxt(intrinsic_path)
        if intrinsic.shape == (3, 3):
            fx = intrinsic[0, 0]
            fov_h = 2 * np.arctan(img.size[0] / (2 * fx))
            print(f"水平FOV: {np.degrees(fov_h):.1f}°")

print("\n" + "=" * 70)
print("2. RAS 相机轨迹详细分析")
print("=" * 70)

print("RAS 相机位置 (y-down, 逐帧):")
for i in range(len(ras_cam_ydown)):
    print(f"  帧{i:2d}: [{ras_cam_ydown[i,0]:8.5f}, {ras_cam_ydown[i,1]:8.5f}, {ras_cam_ydown[i,2]:8.5f}]")

print(f"\nRAS 相机轨迹跨度: {np.linalg.norm(ras_cam_ydown[-1] - ras_cam_ydown[0]):.6f}m")
print(f"RAS 相机轨迹总路径长度: {sum(np.linalg.norm(ras_cam_ydown[i+1] - ras_cam_ydown[i]) for i in range(len(ras_cam_ydown)-1)):.6f}m")

print("\nHaWoR 相机位置 (y-up, 前20帧):")
for i in range(min(20, len(hawor_cam))):
    print(f"  帧{i:2d}: [{hawor_cam[i,0]:8.5f}, {hawor_cam[i,1]:8.5f}, {hawor_cam[i,2]:8.5f}]")

print(f"\nHaWoR 前20帧轨迹跨度: {np.linalg.norm(hawor_cam[19] - hawor_cam[0]):.6f}m")
print(f"HaWoR 全部113帧轨迹跨度: {np.linalg.norm(hawor_cam[-1] - hawor_cam[0]):.6f}m")

print("\n" + "=" * 70)
print("3. 直接在RAS坐标系中验证手-物关系")
print("=" * 70)

# 如果RAS和HaWoR的相机朝向一致(R_rel≈I), 
# 那从HaWoR到手部位置的向量, 经过坐标系变换后, 应该与RAS中相机到GLB的向量一致

# HaWoR中: 相机[0] → 手均值
hand_idx = 0 if pred_valid[0].any() else 1
v = pred_valid[hand_idx]
hand_mean_render = pred_trans[hand_idx, v].mean(axis=0)
label = "左手" if hand_idx == 0 else "右手"

cam_to_hand_hawor = hand_mean_render - hawor_cam[0]
print(f"{label} 均值 (HaWoR render): {hand_mean_render}")
print(f"HaWoR 相机[0]: {hawor_cam[0]}")
print(f"相机→手 向量 (HaWoR y-up): {cam_to_hand_hawor}")
print(f"相机→手 距离: {np.linalg.norm(cam_to_hand_hawor):.4f}m")

# RAS中: 相机[0] → GLB中心
tm_scene = trimesh.load(GLB_PATH, force='scene')
verts_geom_list = []
for name, geom in tm_scene.geometry.items():
    verts_geom_list.append(geom.vertices)
verts_geom = np.vstack(verts_geom_list)
glb_center = verts_geom.mean(axis=0)

cam_to_glb_ras = glb_center - ras_cam_ydown[0]
print(f"\nGLB中心 (RAS y-down): {glb_center}")
print(f"RAS 相机[0]: {ras_cam_ydown[0]}")
print(f"相机→GLB 向量 (RAS y-down): {cam_to_glb_ras}")
print(f"相机→GLB 距离: {np.linalg.norm(cam_to_glb_ras):.4f}m")

# 将HaWoR的相机→手向量转换到RAS坐标系
# HaWoR y-up → RAS y-down: 乘 R_X (因为 R_X = R_X^T = R_X^{-1})
cam_to_hand_ras_ydown = R_X @ cam_to_hand_hawor
print(f"\n相机→手 向量 (转换到RAS y-down): {cam_to_hand_ras_ydown}")

# 比较方向
dir_hand = cam_to_hand_ras_ydown / np.linalg.norm(cam_to_hand_ras_ydown)
dir_glb = cam_to_glb_ras / np.linalg.norm(cam_to_glb_ras)
angle = np.degrees(np.arccos(np.clip(np.dot(dir_hand, dir_glb), -1, 1)))
print(f"方向比较: 手方向={dir_hand}, GLB方向={dir_glb}")
print(f"方向夹角: {angle:.1f}°")

# 比较距离比
dist_ratio = np.linalg.norm(cam_to_glb_ras) / np.linalg.norm(cam_to_hand_hawor)
print(f"距离比 (GLB距离/手距离): {dist_ratio:.4f}")

print("\n" + "=" * 70)
print("4. 直接用相机→手/物的方向和距离来对齐")
print("=" * 70)

# 如果相机朝向一致, 那手和物体应该在相机的同一方向
# 尺度因子 = GLB距离 / 手距离
# 但这假设手和物体在相机的同一方向和距离, 这不一定对

# 更好的方法: 用相机轨迹的尺度比
# RAS相机轨迹跨度 / HaWoR相机轨迹跨度 (前20帧)
ras_span = np.linalg.norm(ras_cam_ydown[-1] - ras_cam_ydown[0])
hawor_span_20 = np.linalg.norm(hawor_cam[19] - hawor_cam[0])
hawor_span_all = np.linalg.norm(hawor_cam[-1] - hawor_cam[0])

scale_from_span_20 = ras_span / hawor_span_20 if hawor_span_20 > 1e-6 else 0
scale_from_span_all = ras_span / hawor_span_all if hawor_span_all > 1e-6 else 0

print(f"RAS 相机轨迹跨度: {ras_span:.6f}m")
print(f"HaWoR 前20帧轨迹跨度: {hawor_span_20:.6f}m")
print(f"HaWoR 全部帧轨迹跨度: {hawor_span_all:.6f}m")
print(f"尺度比 (前20帧): {scale_from_span_20:.4f}")
print(f"尺度比 (全部帧): {scale_from_span_all:.4f}")

# 用方向对齐 + 距离比来验证
# 如果方向夹角很小, 说明手和物体在相机的同一侧
# 如果距离比合理, 说明尺度正确

print("\n" + "=" * 70)
print("5. 检查RAS外参是否可能是c2w格式")
print("=" * 70)

# 如果RAS外参是c2w而不是w2c, 那相机位置应该直接是ext[:3,3]
ext0 = np.loadtxt(ext_files[0])
if ext0.shape == (3, 4):
    ext0 = np.vstack([ext0, [0, 0, 0, 1]])

cam_pos_w2c = -ext0[:3, :3].T @ ext0[:3, 3]  # 假设w2c
cam_pos_c2w = ext0[:3, 3]                      # 假设c2w

print(f"帧0 外参矩阵:\n{ext0}")
print(f"cam_pos (假设w2c): {cam_pos_w2c}")
print(f"cam_pos (假设c2w): {cam_pos_c2w}")

# 检查R_w2c是否接近I
R_w2c = ext0[:3, :3]
print(f"\nR_w2c[0] 接近I? {np.allclose(R_w2c, np.eye(3), atol=0.1)}")
print(f"R_w2c[0] 接近R_X? {np.allclose(R_w2c, R_X, atol=0.1)}")

# 如果R_w2c接近R_X, 那外参可能是c2w格式, 且R_c2w接近I
# 这意味着世界坐标系就是相机坐标系(标准OpenGL y-up)
if np.allclose(R_w2c, R_X, atol=0.1):
    print("\n⚠ R_w2c[0] 接近 R_X! 外参可能是 c2w 格式!")
    print("  如果是c2w: R_c2w = ext[:3,:3], t_c2w = ext[:3,3]")
    print("  世界坐标系 = 相机坐标系 (y-up, OpenGL)")
    
    # 重新计算相机位置 (假设c2w)
    cam_positions_c2w = []
    for f in ext_files:
        ext = np.loadtxt(f)
        if ext.shape == (3, 4):
            ext = np.vstack([ext, [0, 0, 0, 1]])
        cam_positions_c2w.append(ext[:3, 3])
    cam_c2w = np.array(cam_positions_c2w)
    
    print(f"\n  相机位置 (假设c2w):")
    for i in range(len(cam_c2w)):
        print(f"    帧{i:2d}: [{cam_c2w[i,0]:8.5f}, {cam_c2w[i,1]:8.5f}, {cam_c2w[i,2]:8.5f}]")
    
    print(f"  跨度: {np.linalg.norm(cam_c2w[-1] - cam_c2w[0]):.6f}m")
    print(f"  范围: x[{cam_c2w[:,0].min():.5f},{cam_c2w[:,0].max():.5f}]"
          f" y[{cam_c2w[:,1].min():.5f},{cam_c2w[:,1].max():.5f}]"
          f" z[{cam_c2w[:,2].min():.5f},{cam_c2w[:,2].max():.5f}]")
    
    # 如果是c2w, 那GLB顶点应该在什么坐标系?
    # 如果世界=相机=y-up(OpenGL), 那GLB顶点也应该在y-up
    # GLB原始顶点(y-down)需要转换到y-up才能与世界坐标系对齐
    
    cam_to_glb_c2w = glb_center - cam_c2w[0]
    print(f"\n  相机→GLB向量 (假设c2w, y-down): {cam_to_glb_c2w}")
    
    # GLB原始顶点转换到y-up
    glb_center_yup = R_X @ glb_center
    cam_to_glb_c2w_yup = glb_center_yup - cam_c2w[0]
    print(f"  相机→GLB向量 (假设c2w, GLB转y-up): {cam_to_glb_c2w_yup}")

print("\n" + "=" * 70)
print("6. 检查RAS外参的R矩阵模式")
print("=" * 70)

# 逐帧检查R矩阵
for i in [0, 1, 5, 10, 19]:
    if i >= len(ext_files):
        continue
    ext = np.loadtxt(ext_files[i])
    if ext.shape == (3, 4):
        ext = np.vstack([ext, [0, 0, 0, 1]])
    R = ext[:3, :3]
    det = np.linalg.det(R)
    is_rot = np.allclose(R @ R.T, np.eye(3), atol=0.01)
    print(f"  帧{i}: det(R)={det:.6f}, 是旋转矩阵? {is_rot}")
    if not is_rot:
        print(f"    R@R^T 对角线: {(R @ R.T).diagonal()}")
    
    # 检查R是否接近R_X
    if np.allclose(np.abs(R), np.abs(R_X), atol=0.1):
        print(f"    ⚠ R接近±R_X!")

print("\n完成!")
