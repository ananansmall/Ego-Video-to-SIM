#!/usr/bin/env python3
"""诊断脚本2: 验证帧对应关系 + 使用相机朝向确定旋转"""

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


def umeyama_align(src_pts, dst_pts):
    assert src_pts.shape == dst_pts.shape
    n = src_pts.shape[0]
    src_mean = src_pts.mean(axis=0)
    dst_mean = dst_pts.mean(axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean
    sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
    if sigma_src < 1e-8:
        return 1.0, np.eye(3), dst_mean - src_mean
    scale = sigma_dst / sigma_src
    cov = (dst_centered.T @ src_centered) / n
    U, D, VH = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[2, 2] = -1
    R = U @ S @ VH
    t = dst_mean - scale * (R @ src_mean)
    return scale, R, t


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


def load_hawor_cameras(hawor_file):
    data = dict(np.load(hawor_file, allow_pickle=True))
    return data['t_c2w'], data['R_c2w'], data['pred_trans'], data['pred_valid']


ras_cam_ydown, ras_R_c2w, ext_files = load_ras_cameras(RAS_DIR)
ras_cam_yup = (R_X @ ras_cam_ydown.T).T
hawor_cam, hawor_R_c2w, pred_trans, pred_valid = load_hawor_cameras(HAWOR_FILE)

n_ras = len(ras_cam_ydown)
n_hawor = len(hawor_cam)

print("=" * 70)
print("1. RAS 外参文件数量和名称")
print("=" * 70)
print(f"RAS 外参数量: {n_ras}")
print(f"文件名: {[os.path.basename(f) for f in ext_files[:5]]} ... {[os.path.basename(f) for f in ext_files[-3:]]}")
print(f"HaWoR 帧数: {n_hawor}")

# 检查 RAS color 目录有多少图片
color_dir = os.path.join(RAS_DIR, 'color')
color_files = glob(os.path.join(color_dir, '*.jpg'))
print(f"RAS color 图片数: {len(color_files)}")
if color_files:
    color_indices = sorted([int(os.path.basename(f).split('.')[0]) for f in color_files])
    print(f"图片索引范围: {color_indices[0]} ~ {color_indices[-1]}")

print("\n" + "=" * 70)
print("2. 对比不同帧对应策略")
print("=" * 70)

# 加载GLB
tm_scene = trimesh.load(GLB_PATH, force='scene')
meshes_dump = tm_scene.dump()
verts_dump = np.vstack([m.vertices for m in meshes_dump])
glb_center_dump = verts_dump.mean(axis=0)

# 手部位置
hand_idx = 0 if pred_valid[0].any() else 1
v = pred_valid[hand_idx]
hand_mean_render = pred_trans[hand_idx, v].mean(axis=0)
hand_mean_sapien = RXWORLD_TO_SAPIEN @ hand_mean_render
label = "左手" if hand_idx == 0 else "右手"
print(f"使用 {label}, 均值(SAPIEN): {hand_mean_sapien}")

correspondence_strategies = [
    ("均匀映射 (当前)", [(ri, round(ri * (n_hawor - 1) / (n_ras - 1))) for ri in range(n_ras)]),
    ("直接对应 (ras_i=hawor_i)", [(ri, ri) for ri in range(min(n_ras, n_hawor))]),
    ("每隔6帧", [(ri, ri * 6) for ri in range(n_ras) if ri * 6 < n_hawor]),
    ("每隔7帧", [(ri, ri * 7) for ri in range(n_ras) if ri * 7 < n_hawor]),
]

for strategy_name, common_frames in correspondence_strategies:
    if len(common_frames) < 3:
        print(f"\n  {strategy_name}: 帧对太少, 跳过")
        continue

    print(f"\n--- {strategy_name} ({len(common_frames)} 对) ---")

    for cam_name, ras_cam in [("yup", ras_cam_yup), ("ydown", ras_cam_ydown)]:
        src = np.array([hawor_cam[hi] for _, hi in common_frames])
        dst = np.array([ras_cam[ri] for ri, _ in common_frames])
        s, R, t = umeyama_align(src, dst)
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

        s_inv = 1.0 / s
        R_inv = R.T
        t_inv = -s_inv * (R_inv @ t)
        glb_hawor = s_inv * (R_inv @ glb_center_dump) + t_inv
        glb_sapien = RXWORLD_TO_SAPIEN @ glb_hawor
        dist = np.linalg.norm(hand_mean_sapien - glb_sapien)

        print(f"  RAS_{cam_name}: 角度={angle:.1f}°, 尺度={s:.4f}, 手-GLB距离={dist:.4f}m")

print("\n" + "=" * 70)
print("3. 使用相机朝向 (R_c2w) 确定坐标系旋转")
print("=" * 70)

# RAS R_c2w 在 y-down 坐标系
# HaWoR R_c2w 在 y-up 坐标系
# 对应帧: 先用直接对应
for strategy_name, common_frames in correspondence_strategies[:2]:
    print(f"\n--- {strategy_name} ---")
    for ri, hi in common_frames[:5]:
        ras_R = ras_R_c2w[ri]
        hawor_R = hawor_R_c2w[hi]

        # RAS 相机前向 (OpenCV z-forward)
        ras_forward_ydown = ras_R[:, 2]
        # 转换到 y-up
        ras_forward_yup = R_X @ ras_forward_ydown

        # HaWoR 相机前向 (OpenGL -z, 即 R_c2w[:, 2] 是 camera z 在 world 中的方向)
        # 但 OpenGL 相机看 -z, 所以前向是 -R_c2w[:, 2]
        # 不对, t_c2w 是 camera-to-world, R_c2w 把 camera 坐标转到 world 坐标
        # camera 的 z 轴在 world 中是 R_c2w[:, 2]
        # OpenCV: camera 看 +z, 所以前向 = R_c2w[:, 2]
        # OpenGL: camera 看 -z, 所以前向 = -R_c2w[:, 2]
        hawor_forward = -hawor_R_c2w[hi][:, 2]

        # RAS 相机在 y-down 中的前向
        ras_forward_ydown_dir = ras_R[:, 2]

        angle_between = np.degrees(np.arccos(np.clip(
            np.dot(ras_forward_yup, hawor_forward) /
            (np.linalg.norm(ras_forward_yup) * np.linalg.norm(hawor_forward) + 1e-8), -1, 1)))

        print(f"  RAS[{ri}]→HaWoR[{hi}]: RAS_fwd(yup)={ras_forward_yup}, HaWoR_fwd={hawor_forward}, 夹角={angle_between:.1f}°")

print("\n" + "=" * 70)
print("4. 从 R_c2w 计算坐标系旋转矩阵")
print("=" * 70)

# 对于对应帧, 计算 R_rel = R_c2w_ras_yup @ R_c2w_hawor.T
# 这给出从 HaWoR 坐标系到 RAS y-up 坐标系的旋转
# 如果旋转一致, 说明帧对应正确

common_frames_direct = [(ri, ri) for ri in range(min(n_ras, n_hawor))]

R_rels = []
for ri, hi in common_frames_direct:
    ras_R_ydown = ras_R_c2w[ri]
    ras_R_yup = R_X @ ras_R_ydown
    hawor_R = hawor_R_c2w[hi]

    R_rel = ras_R_yup @ hawor_R.T
    R_rels.append(R_rel)

R_rels = np.array(R_rels)

# 检查 R_rel 是否一致
print("R_rel (RAS_yup @ HaWoR.T) 对前5帧:")
for i in range(min(5, len(R_rels))):
    angle = np.degrees(np.arccos(np.clip((np.trace(R_rels[i]) - 1) / 2, -1, 1)))
    print(f"  帧{i}: 角度={angle:.1f}°, R=\n{R_rels[i]}")

# 取平均旋转
R_rel_mean = R_rels.mean(axis=0)
U, _, VH = np.linalg.svd(R_rel_mean)
R_rel_best = U @ VH
if np.linalg.det(R_rel_best) < 0:
    U[:, -1] *= -1
    R_rel_best = U @ VH

angle_mean = np.degrees(np.arccos(np.clip((np.trace(R_rel_best) - 1) / 2, -1, 1)))
print(f"\n平均 R_rel: 角度={angle_mean:.1f}°")
print(f"R_rel_best =\n{R_rel_best}")

# 用均匀映射再算一次
common_frames_uniform = [(ri, round(ri * (n_hawor - 1) / (n_ras - 1))) for ri in range(n_ras)]
R_rels_uniform = []
for ri, hi in common_frames_uniform:
    ras_R_yup = R_X @ ras_R_c2w[ri]
    hawor_R = hawor_R_c2w[hi]
    R_rel = ras_R_yup @ hawor_R.T
    R_rels_uniform.append(R_rel)

R_rels_uniform = np.array(R_rels_uniform)
R_rel_mean_u = R_rels_uniform.mean(axis=0)
U, _, VH = np.linalg.svd(R_rel_mean_u)
R_rel_best_u = U @ VH
if np.linalg.det(R_rel_best_u) < 0:
    U[:, -1] *= -1
    R_rel_best_u = U @ VH

angle_mean_u = np.degrees(np.arccos(np.clip((np.trace(R_rel_best_u) - 1) / 2, -1, 1)))
print(f"\n均匀映射 平均 R_rel: 角度={angle_mean_u:.1f}°")
print(f"R_rel_best_uniform =\n{R_rel_best_u}")

print("\n" + "=" * 70)
print("5. 用 R_c2w 推导的旋转做对齐")
print("=" * 70)

# 已知: R_rel_best 将 HaWoR 坐标旋转到 RAS y-up 坐标
# 所以: p_ras_yup = s * R_rel_best @ p_hawor + t
# 用这个旋转, 只从位置计算 s 和 t

for strategy_name, common_frames in [("直接对应", common_frames_direct), ("均匀映射", common_frames_uniform)]:
    print(f"\n--- {strategy_name} ---")

    src = np.array([hawor_cam[hi] for _, hi in common_frames])
    dst = np.array([ras_cam_yup[ri] for ri, _ in common_frames])

    # 用 R_rel_best 作为旋转
    R_fixed = R_rel_best

    # 计算尺度: s = sigma_dst / sigma_src (在旋转对齐后)
    src_rotated = (R_fixed @ src.T).T
    src_mean = src_rotated.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src_rotated - src_mean
    dst_centered = dst - dst_mean
    sigma_src = np.sqrt(np.mean(np.sum(src_centered ** 2, axis=1)))
    sigma_dst = np.sqrt(np.mean(np.sum(dst_centered ** 2, axis=1)))
    s_fixed = sigma_dst / sigma_src if sigma_src > 1e-8 else 1.0
    t_fixed = dst_mean - s_fixed * src_mean

    aligned = s_fixed * (R_fixed @ src.T).T + t_fixed
    errors = np.linalg.norm(aligned - dst, axis=1)

    s_inv = 1.0 / s_fixed
    R_inv = R_fixed.T
    t_inv = -s_inv * (R_inv @ t_fixed)

    glb_hawor = s_inv * (R_inv @ glb_center_dump) + t_inv
    glb_sapien = RXWORLD_TO_SAPIEN @ glb_hawor
    dist = np.linalg.norm(hand_mean_sapien - glb_sapien)

    print(f"  R_fixed 角度: {angle_mean:.1f}°")
    print(f"  尺度: {s_fixed:.4f}")
    print(f"  对齐误差: mean={errors.mean():.6f}m, max={errors.max():.6f}m")
    print(f"  手-GLB距离: {dist:.4f}m")
    print(f"  GLB SAPIEN: {glb_sapien}")

print("\n" + "=" * 70)
print("6. 检查 RAS 外参文件名 vs 视频帧索引")
print("=" * 70)

# RAS color 目录有 140 张图片 (0.jpg ~ 139.jpg)
# RAS extrinsics 目录有 20 个文件 (0.txt ~ 19.txt)
# 关键问题: extrinsic 文件名是否对应 color 图片索引?

# 如果 RAS 每隔 7 帧采样一次 (140/20=7), 那么:
# extrinsic 0.txt → color 0.jpg
# extrinsic 1.txt → color 7.jpg
# extrinsic 2.txt → color 14.jpg
# ...
# extrinsic 19.txt → color 133.jpg

# 或者 extrinsic 文件名就是 color 图片索引:
# extrinsic 0.txt → color 0.jpg
# extrinsic 1.txt → color 1.jpg
# ...
# extrinsic 19.txt → color 19.jpg

# HaWoR 处理了 113 帧 (0 ~ 112)
# 如果 RAS 采样了 20 帧, 对应关系取决于采样策略

# 让我检查 RAS 相机轨迹的形状来判断
print("RAS 相机位置 (y-down):")
for i in range(min(5, n_ras)):
    print(f"  帧{i}: {ras_cam_ydown[i]}")
print(f"  ...")
for i in range(max(5, n_ras-3), n_ras):
    print(f"  帧{i}: {ras_cam_ydown[i]}")

print("\nHaWoR 相机位置:")
for i in range(5):
    print(f"  帧{i}: {hawor_cam[i]}")
print(f"  ...")
for i in range(n_hawor-3, n_hawor):
    print(f"  帧{i}: {hawor_cam[i]}")

# 计算相邻帧之间的位移
print("\nRAS 相邻帧位移:")
for i in range(1, min(5, n_ras)):
    delta = ras_cam_ydown[i] - ras_cam_ydown[i-1]
    print(f"  帧{i-1}→{i}: {delta}, |delta|={np.linalg.norm(delta):.6f}")

print("\nHaWoR 相邻帧位移:")
for i in range(1, 5):
    delta = hawor_cam[i] - hawor_cam[i-1]
    print(f"  帧{i-1}→{i}: {delta}, |delta|={np.linalg.norm(delta):.6f}")

# 如果 RAS 每隔7帧采样, 那么 RAS 相邻帧位移应该约等于 HaWoR 7帧位移
print("\nHaWoR 7帧位移 (如果RAS每隔7帧采样):")
for i in range(0, min(20, n_hawor-7), 7):
    delta = hawor_cam[min(i+7, n_hawor-1)] - hawor_cam[i]
    print(f"  帧{i}→{i+7}: {delta}, |delta|={np.linalg.norm(delta):.6f}")

print("\n完成!")
