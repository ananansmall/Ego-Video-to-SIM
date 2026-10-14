"""
诊断 FoundationPose 位姿偏差来源：区分「初始注册偏」与「跟踪漂移」。

思路：用同一套 mesh_to_mask 评估函数，分别算
  (a) 全序列自身跟踪 → 观察是否漂移
  (b) 把真值(ICP)位姿喂进同一评估器 → 给出该评估器的能力上界
若 (b) 也只有 ~0.6，说明问题在评估器/mesh 对齐，不在跟踪。
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from fp_run_tracking import mesh_to_mask  # noqa: E402


def iou_against_masks(verts, poses, frames_by_id, data_dir, n_frames):
    ious = {}
    for t in range(n_frames):
        fr = frames_by_id.get(t)
        if fr is None:
            continue
        m = np.asarray(fr['mask']) > 0
        H, W = m.shape
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        P = poses[t]
        vc = verts @ P[:3, :3].T + P[:3, 3]
        pm = mesh_to_mask(vc, K, H, W)
        inter = (pm & m).sum()
        union = (pm | m).sum()
        ious[t] = float(inter / union) if union else 0.0
    return ious


def seg_stats(ious, name):
    ks = sorted(ious)
    out = [f'{name}']
    for label, a, b in (('静止0-59', 0, 59), ('举升60-80', 60, 80), ('放置81-112', 81, 112)):
        v = [ious[t] for t in ks if a <= t <= b]
        if v:
            out.append(f'  {label}: mean={np.mean(v):.3f} min={np.min(v):.3f}')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/output_v2/7_vggt_omega')
    ap.add_argument('--obj', default='toy_inst1')
    args = ap.parse_args()
    sd = args.scene_dir
    data_dir = os.path.join(sd, 'foundationpose_data')
    fp_out = os.path.join(sd, 'foundationpose_out')

    import trimesh
    mesh = trimesh.load(os.path.join(data_dir, 'mesh.obj'))
    verts = np.asarray(mesh.vertices)

    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        all_masks = pickle.load(f)['all_masks']
    cat, inst_s = args.obj.rsplit('_inst', 1)
    frames_by_id = {fr['frame_id']: fr for fr in all_masks[cat][int(inst_s)]}
    n_frames = len(os.listdir(os.path.join(data_dir, 'rgb')))

    # (a) FoundationPose 自身结果
    fp = {t: np.loadtxt(os.path.join(fp_out, 'ob_in_cam', f'{t:06d}.txt'))
          for t in range(n_frames)}

    report = {}

    def add(name, poses):
        ious = iou_against_masks(verts, poses, frames_by_id, data_dir, n_frames)
        report[name] = ious
        print(seg_stats(ious, name))
        return ious

    add('FP_self_tracking', fp)

    # (b) 旧 ICP 结果（真值基线）喂进同一评估器
    # 旧 2D 叠加评估器的口径: 用 centroids(世界系物体位置) + trajectory 的旋转,
    # 即 world 系位姿 = [[R, centroid],[0,1]]; 再用 extrinsics 转相机系。
    g = np.load(os.path.join(sd, 'object_trajectories', f'{args.obj}_traj.npz'),
                allow_pickle=True)
    icp_cam = {}
    for t in range(n_frames):
        ext = np.loadtxt(os.path.join(sd, 'extrinsics', f'{t}.txt'))
        T_w = np.eye(4)
        T_w[:3, :3] = g['trajectory'][t][:3, :3]
        T_w[:3, 3] = g['centroids'][t]
        icp_cam[t] = np.linalg.inv(ext) @ T_w
    add('ICP_baseline_same_evaluator', icp_cam)

    with open(os.path.join(fp_out, 'fp_diagnose.json'), 'w') as f:
        json.dump({k: {str(t): v for t, v in d.items()} for k, d in report.items()}, f, indent=2)
    print('\n=> 若 ICP_baseline 也只有 ~0.6, 问题在 mesh/评估器; '
          '若 ICP 明显更高(0.8+), 问题在 FP 跟踪本身。')


if __name__ == '__main__':
    main()