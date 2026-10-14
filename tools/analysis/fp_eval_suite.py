"""FoundationPose 跟踪质量评测: FP vs mainv3 ICP 6DoF 逐帧对比。

回答三个问题:
  1. FP 位姿准不准      → 面片 IoU + 深度反算 + 静止段漂移
  2. FP 比现有方法好在哪 → 静止段漂移 / 遮挡帧覆盖 / 旋转量级
  3. 每个模块贡献什么   → 分阶段拆解误差来源
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from fp_iou_eval import rasterize_mesh  # noqa: E402


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def load_fp(sd, obj):
    d = np.load(os.path.join(sd, 'fp_glb', f'{obj}_fp_trajectory.npz'), allow_pickle=True)
    return d['poses'], d['centers']


def load_icp(sd, obj):
    p = os.path.join(sd, 'object_trajectories', f'{obj}_traj.npz')
    d = np.load(p, allow_pickle=True)
    return d['trajectory'], d['centroids'], d['valid_frames'], d['occluded_frames']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir',
                    default='/mnt/data_8THDD/lza/workspace/robot_world_ws/src/'
                             'ReplicateAnyScene/output_v2/7_vggt_omega')
    ap.add_argument('--obj', default='toy_inst1')
    args = ap.parse_args()
    sd, obj = args.scene_dir, args.obj
    data_dir = os.path.join(sd, 'foundationpose_data')
    fp_out = os.path.join(sd, 'foundationpose_out')

    fp_poses, fp_cent = load_fp(sd, obj)
    icp_traj, icp_cent, icp_valid, icp_occ = load_icp(sd, obj)
    n = len(fp_poses)

    mesh = trimesh.load(os.path.join(data_dir, 'mesh.obj'))
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        am = pickle.load(f)['all_masks']
    cat, inst_s = obj.rsplit('_inst', 1)
    fb = {fr['frame_id']: fr for fr in am[cat][int(inst_s)]}
    ts = sorted(fb)

    rep = {'obj': obj, 'n_frames': n}

    # ---------- 1. 逐帧 IoU (FP 投影 vs SAM3 mask) ----------
    ious, depth_err = [], []
    for t in ts:
        if t >= n:
            continue
        gt = np.asarray(fb[t]['mask']) > 0
        if gt.sum() < 50:
            continue
        H, W = gt.shape
        K = np.loadtxt(os.path.join(data_dir, 'cam_K', f'{t:06d}.txt'))
        P = np.loadtxt(os.path.join(fp_out, 'ob_in_cam', f'{t:06d}.txt'))
        pm = rasterize_mesh(mesh, P, K, H, W)
        ious.append(((pm & gt).sum() / max((pm | gt).sum(), 1), t))

        # 深度反算: mesh 顶点投影 z vs depth 图实测 z (取 mask 内像素)
        dp = os.path.join(data_dir, 'depth', f'{t:06d}.png')
        if os.path.isfile(dp):
            dm = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32)
            dm = dm / 1000.0 if dm.max() > 20 else dm
            V = mesh.vertices @ P[:3, :3].T + P[:3, 3]
            z = V[:, 2]
            ok = z > 1e-6
            u = (V[ok, 0] / z[ok] * K[0, 0] + K[0, 2]).astype(int)
            v = (V[ok, 1] / z[ok] * K[1, 1] + K[1, 2]).astype(int)
            iu = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u, v, z = u[iu], v[iu], z[iu]
            good = z < dm[v, u] + 0.03
            if good.sum() > 100:
                depth_err.append(float(np.median(dm[v[good], u[good]] - z[good])))

    iou_v = np.array([a for a, _ in ious])
    iou_t = np.array([b for _, b in ious])
    rep['iou'] = {'mean': float(iou_v.mean()), 'median': float(np.median(iou_v)),
                  'min': float(iou_v.min()), 'max': float(iou_v.max()),
                  'p10': float(np.percentile(iou_v, 10)),
                  'n_lt_0.5': int((iou_v < 0.5).sum())}
    rep['depth_residual_m'] = {'median': float(np.median(depth_err)),
                               'abs_median': float(np.median(np.abs(depth_err)))}

    # 分阶段: 真实静止段由逐帧 step 阈值自动定(step>1cm 视为运动起点),
    # 不能拍脑袋写 0-59 —— 那样会把举起动作算进静止段, 得出"漂移 17cm"的假结论。
    step_all = np.linalg.norm(np.diff(fp_cent[:n], axis=0), axis=1)
    t_move = int(np.argmax(step_all > 0.01)) + 1 if (step_all > 0.01).any() else n
    rep['move_start_frame'] = t_move
    for label, a, b in (('static_0_move', 0, t_move - 1),
                        ('moving_tail', t_move, n - 1)):
        sel = iou_v[(iou_t >= a) & (iou_t <= b)]
        if len(sel):
            rep['iou'][label] = float(sel.mean())

    # ---------- 2. FP vs ICP 位置对比 ----------
    # FP 中心 -> 相机系; ICP centroids 在世界系, 需统一
    # FP poses 已是世界系(Z-up), centers 也是世界系质心
    common = min(n, len(icp_cent))
    fp_c = fp_cent[:common]
    ic_c = icp_cent[:common]
    d_pos = np.linalg.norm(fp_c - ic_c, axis=1)

    # 静止段漂移: 相对首帧的位移, 这是最关键的静止伪影指标。
    # 静止段终点必须用自动检出的运动起点, 否则把举起动作算进去 -> 假的 17cm。
    t_static = rep['move_start_frame']

    def drift(c):
        return np.linalg.norm(c[:t_static] - c[0], axis=1)

    rep['fp_vs_icp'] = {
        'pos_delta_mean_m': float(d_pos.mean()),
        'pos_delta_max_m': float(d_pos.max()),
        'static_end_frame': int(t_static),
        'fp_static_drift_mean_m': float(drift(fp_c).mean()),
        'fp_static_drift_max_m': float(drift(fp_c).max()),
        'icp_static_drift_mean_m': float(drift(ic_c).mean()),
        'icp_static_drift_max_m': float(drift(ic_c).max()),
    }

    # ---------- 3. 运动学对比 ----------
    def kin(c):
        step = np.linalg.norm(np.diff(c, axis=0), axis=1)
        return {'path_len_m': float(step.sum()),
                'net_disp_m': float(np.linalg.norm(c[-1] - c[0])),
                'max_step_m': float(step.max()),
                'median_step_m': float(np.median(step))}

    rep['fp_kinematics'] = kin(fp_c)
    rep['icp_kinematics'] = kin(ic_c)

    # ---------- 4. 旋转量级(用相对首帧的累积旋转, 避免 rotvec 退化) ----------
    def rot_stats(poses):
        R0 = poses[0][:3, :3]
        cum, per = [], []
        for i in range(1, len(poses)):
            R = poses[i][:3, :3] @ R0.T
            cum.append(rot_deg(R))
            per.append(rot_deg(poses[i][:3, :3] @ poses[i - 1][:3, :3].T))
        return {'total_deg': float(cum[-1]), 'per_frame_mean_deg': float(np.mean(per)),
                'per_frame_max_deg': float(np.max(per)),
                'static_cum_deg': float(cum[t_static - 1]) if len(cum) >= t_static else None}

    rep['fp_rotation'] = rot_stats(fp_poses[:common])
    rep['icp_rotation'] = rot_stats(icp_traj[:common])

    # ---------- 5. 覆盖度 ----------
    rep['coverage'] = {
        'fp_frames': int(n),
        'icp_frames': int(len(icp_traj)),
        'icp_valid_frames': int(icp_valid.sum()),
        'icp_occluded': int(len(icp_occ)),
        'fp_iou_ge_0.5_ratio': float((iou_v >= 0.5).mean()),
        'fp_iou_ge_0.7_ratio': float((iou_v >= 0.7).mean()),
    }

    # ---------- 打印 ----------
    def p(d, ind=0):
        pad = ' ' * ind
        for k, v in d.items():
            if isinstance(v, dict):
                print(f'{pad}{k}:')
                p(v, ind + 2)
            elif isinstance(v, float):
                print(f'{pad}{k}: {v:.4f}')
            else:
                print(f'{pad}{k}: {v}')

    print(f'=== {obj} FoundationPose 评测 ===')
    p(rep)

    out = os.path.join(sd, 'fp_glb', 'fp_eval_report.json')
    with open(out, 'w') as f:
        json.dump(rep, f, indent=2)
    print(f'\n-> {out}')


if __name__ == '__main__':
    main()