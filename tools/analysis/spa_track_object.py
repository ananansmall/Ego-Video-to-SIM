"""SpatialTrackerV2 物体刚体位姿跟踪适配 (FoundationPose 替代路线)。

思路: 不用模型自己估计的世界系 (其 BA 世界系与 VGGT/GT 不同), 只取它的 2D 点轨迹;
逐帧用我们的 depth + cam_K + extrinsics 把 2D 轨迹反投影回 VGGT 世界系,
保证与 GT 轨迹/评测同系同尺度。位姿恢复:
  1) query 帧: mesh 采样点 vs mask+depth 物体点云 ICP (PCA 多初始值) -> P0
  2) x_local = inv(P0) @ X_world[t0]
  3) 逐帧加权 Kabsch(x_local -> X_world[t]) -> P_t (2 轮剔除离群)

输出 fp_out_<obj>_spa/{ob_in_world, ob_in_cam}/*.txt, 与 FP 管线格式一致,
可直接用 fp_iou_eval.py --out_dir 和 fp_gif_2d3d.py --poses 评测/出图。

用法 (foundationpose env, GPU):
  CUDA_VISIBLE_DEVICES=2 python tools/analysis/spa_track_object.py \
      --scene_dir output_v2/7_vggt_omega --obj toy_inst1
"""
import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import trimesh

SPA_ROOT = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/SpaTrackerV2'


def load_gt_masks(sd, obj):
    sel = json.load(open(os.path.join(sd, 'fp_mask_select', f'{obj}.json')))
    cat = obj.rsplit('_inst', 1)[0]
    with open(os.path.join(sd, '_pre_dedup_masks.pkl'), 'rb') as f:
        am = pickle.load(f)['all_masks']
    fb = {fr['frame_id']: fr for fr in am[cat][sel['best_pre_inst']]}
    return fb, sel


def backproject_uv(u, v, z, K):
    x = (u - K[0, 2]) / K[0, 0] * z
    y = (v - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, np.full_like(z, z)], -1)


def sample_depth_median(depth, v, u, win=2):
    """(2win+1)^2 窗口中有效深度中值, 抑制边缘/噪声。"""
    h, w = depth.shape
    y0, y1 = max(v - win, 0), min(v + win + 1, h)
    x0, x1 = max(u - win, 0), min(u + win + 1, w)
    patch = depth[y0:y1, x0:x1]
    ok = patch > 0.05
    if not ok.any():
        return np.nan
    return float(np.median(patch[ok]))


def pca_init_rots(pts):
    """PCA 主轴 -> 4 个 180 度符号组合的旋转 (绕 y,z 翻转覆盖手性)。"""
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt.T
    if np.linalg.det(ax) < 0:
        ax[:, 2] *= -1
    rots = []
    for sx in (1, -1):
        for sz in (1, -1):
            a = ax.copy()
            a[:, 0] *= sx
            a[:, 2] *= sz
            if np.linalg.det(a) < 0:  # 保持右手系
                a[:, 1] *= -1
            rots.append(a)
    return rots


def axis_align_inits(v_m, a_gt):
    """mesh 长轴 v_m 对到 ±GT 长轴 a_gt 的旋转, 再绕 a_gt 加 0/90/180/270 自转
    (旋转对称物体的自转不可观, 全部枚举为 ICP 初始值)。"""
    from scipy.spatial.transform import Rotation
    v = np.asarray(v_m, float)
    v /= np.linalg.norm(v)
    a = np.asarray(a_gt, float)
    a /= np.linalg.norm(a)
    inits = []
    for s in (1.0, -1.0):
        R0 = Rotation.align_vectors([s * a], [v])[0].as_matrix()
        for k in range(4):
            Rspin = Rotation.from_rotvec(a * np.pi * k / 2).as_matrix()
            inits.append(Rspin @ R0)
    return inits


def icp_mesh_to_cloud(mesh, cloud, init_rots=None, thr=0.05, axis_pair=None):
    """open3d ICP: point-to-point 粗 + point-to-plane 精, 多初始值取最优。
    mesh 原点可能离物体质心很远 (toy 0.62m), 先质心平移对齐再 ICP。
    axis_pair=(v_m, a_gt): 长轴朝向约束 —— bottle 等近旋转对称物体 ICP 会锚歪
    ~70° 的错误局部极小 (3chair-bottle 实测 P0 偏 149°), 先按
    |R@v_m · a_gt| >= 0.90 过滤候选再选 rmse 最小, 无候选过轴才回退全局最优。"""
    import open3d as o3d
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(mesh)
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(cloud)
    dst.estimate_normals()

    c_dst = cloud.mean(0)
    results = []
    # mesh 可能本身就是世界系摆放的 (真值变换≈单位阵), 单位阵必须参与尝试
    inits = [np.eye(3)] + (init_rots or [])
    if axis_pair is not None:
        inits += axis_align_inits(*axis_pair)
    for R0 in inits:
        T = np.eye(4)
        T[:3, :3] = R0
        T[:3, 3] = c_dst - R0 @ mesh.mean(0)
        reg = o3d.pipelines.registration.registration_icp(
            src, dst, thr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        best1 = (reg.inlier_rmse, reg.fitness, reg.transformation)
        # point-to-plane 精配在噪声法向下可能发散, 只有变好才采纳
        try:
            reg2 = o3d.pipelines.registration.registration_icp(
                src, dst, thr * 0.4, reg.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            if reg2.fitness >= reg.fitness and reg2.inlier_rmse < reg.inlier_rmse:
                best1 = (reg2.inlier_rmse, reg2.fitness, reg2.transformation)
        except Exception:
            pass
        results.append(best1)

    def _adot(r):
        av = r[2][:3, :3] @ np.asarray(axis_pair[0], float)
        return abs(float(av @ axis_pair[1]) / np.linalg.norm(av))

    if axis_pair is not None:
        good = [r for r in results if _adot(r) >= 0.90]
        best = min(good or results, key=lambda r: r[0])
        return best[2], best[0], best[1], _adot(best)
    best = min(results, key=lambda r: r[0])
    return best[2], best[0], best[1], None


def kabsch(src, dst, w):
    """加权 Kabsch: dst ≈ R@src + t。"""
    w = w / w.sum()
    c1 = (w[:, None] * src).sum(0)
    c2 = (w[:, None] * dst).sum(0)
    H = (w[:, None] * (src - c1)).T @ (dst - c2)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = c2 - R @ c1
    return R, t


def fit_pose(x_loc, x_w, w, n_rounds=2, drop_k=2.5, min_res=0.015):
    """2 轮剔除离群的加权 Kabsch, 返回 (P, n_used, med_res)。"""
    sel = np.ones(len(x_loc), bool)
    best = None
    for _ in range(n_rounds):
        R, t = kabsch(x_loc[sel], x_w[sel], w[sel])
        res = np.linalg.norm(x_w - (x_loc @ R.T + t), axis=1)
        best = (R, t, res)
        med = np.median(res[sel])
        thr = max(drop_k * med, min_res)
        sel = res <= thr
        if sel.sum() < 8:
            break
    R, t, res = best
    P = np.vstack([np.column_stack([R, t]), [0, 0, 0, 1]])
    return P, int(sel.sum()), float(np.median(res[sel]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene_dir', required=True)
    ap.add_argument('--obj', required=True)
    ap.add_argument('--n_pts', type=int, default=300, help='query 点数')
    ap.add_argument('--depth_win', type=int, default=2, help='深度采样半窗')
    ap.add_argument('--out_tag', default='spa')
    ap.add_argument('--smooth_win', type=int, default=0,
                    help='savgol 平滑窗口 (0=不平滑; 建议 7 轻平滑去深度噪声抖动)')
    ap.add_argument('--lock_static', action='store_true',
                    help='静止段 (窗口5净位移<3cm) 锁定为段中值位姿, 抑制抖动')
    ap.add_argument('--med_win', type=int, default=0,
                    help='位姿中值滤波窗口 (平移中值+旋转测地均值), 建议 5')
    ap.add_argument('--gpu', default=None, help='如 2 (等价 CUDA_VISIBLE_DEVICES)')
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.path.insert(0, SPA_ROOT)

    sd = args.scene_dir
    data = os.path.join(sd, f'fp_data_{args.obj}')
    out_dir = os.path.join(sd, f'fp_out_{args.obj}_{args.out_tag}')
    n = len([f for f in os.listdir(os.path.join(data, 'rgb')) if f.endswith('.png')])
    T = n
    print(f'{args.obj}: {T} frames')

    fb, _ = load_gt_masks(sd, args.obj)

    # ---------- 载入数据 (原生分辨率, 模型内部自行缩放) ----------
    video = np.stack([cv2.imread(os.path.join(data, 'rgb', f'{t:06d}.png'))[..., ::-1]
                      for t in range(T)]).astype(np.float32)          # (T,H,W,3) RGB 0-255
    depth = np.stack([cv2.imread(os.path.join(data, 'depth', f'{t:06d}.png'),
                                 cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
                      for t in range(T)])                              # (T,H,W) m
    Ks = np.stack([np.loadtxt(os.path.join(data, 'cam_K', f'{t:06d}.txt'))
                   for t in range(T)])                                 # (T,3,3)
    ext_dir = os.path.join(sd, 'extrinsics')
    exts = sorted(os.listdir(ext_dir), key=lambda x: int(os.path.splitext(x)[0]))[:T]
    w2c = np.stack([np.loadtxt(os.path.join(ext_dir, fn)) for fn in exts])  # (T,4,4)
    c2w = np.linalg.inv(w2c)

    H, W = depth.shape[1:]

    # ---------- query 点: mask 内均匀采样, query 于首个有效帧 ----------
    q_t = min(k for k in sorted(fb) if np.asarray(fb[k]['mask']).sum() >= 50)
    m0 = np.asarray(fb[q_t]['mask']) > 0
    if m0.shape != (H, W):
        m0 = cv2.resize(m0.astype(np.uint8), (W, H)) > 0
    ys, xs = m0.nonzero()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(xs), min(args.n_pts, len(xs)), replace=False)
    q_uv = np.stack([xs[pick], ys[pick]], 1).astype(np.float32)          # (N,2)
    queries = np.column_stack([np.full(len(pick), q_t), q_uv])           # (N,3) t,x,y
    print(f'query: t={q_t}, {len(queries)} pts, mask0 area={m0.sum()}')

    # ---------- SpatialTrackerV2 前馈 (结果缓存, 调试 ICP/Kabsch 时免重跑) ----------
    cache = os.path.join(out_dir, 'spa_tracks.npz')
    if os.path.isfile(cache):
        z = np.load(cache)
        uv, vis, conf = z['uv'], z['vis'], z['conf']
        print(f'loaded cached tracks {cache} {uv.shape}')
    else:
        import torch
        from models.SpaTrackV2.models.predictor import Predictor

        model = Predictor.from_pretrained('Yuxihenry/SpatialTrackerV2-Offline')
        model.spatrack.track_num = max(756, len(queries))
        model.eval().to('cuda')
        vt = torch.from_numpy(video).permute(0, 3, 1, 2).cuda()          # (T,C,H,W) 0-255
        dt = torch.from_numpy(depth).cuda()
        kt = torch.from_numpy(Ks).cuda()
        et = torch.from_numpy(w2c).cuda()

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                outs = model.forward(vt, depth=dt, intrs=kt, extrs=et,
                                     queries=queries.astype(np.float32),
                                     fps=1, full_point=False, iters_track=4,
                                     query_no_BA=True, fixed_cam=False, stage=1,
                                     unc_metric=None, support_frame=T - 1,
                                     replace_ratio=0.2)
        track2d = np.squeeze(outs[5].detach().float().cpu().numpy())     # (T,N,3) 输入分辨率
        vis = np.squeeze(outs[6].detach().float().cpu().numpy())
        conf = np.squeeze(outs[7].detach().float().cpu().numpy())
        print('track2d', track2d.shape, 'vis', vis.shape, 'conf', conf.shape)
        uv = track2d[:, :, :2]
        os.makedirs(out_dir, exist_ok=True)
        np.savez_compressed(cache, uv=uv, vis=vis, conf=conf)

    # ---------- 逐帧反投影回 VGGT 世界系 ----------
    X_w = np.full((T, len(queries), 3), np.nan, np.float64)
    for t in range(T):
        z = np.array([sample_depth_median(depth[t], int(round(v)), int(round(u)),
                                          args.depth_win) for u, v in uv[t]])
        ok = np.isfinite(z)
        if not ok.any():
            continue
        xcam = backproject_uv(uv[t, ok, 0], uv[t, ok, 1], z[ok], Ks[t])
        X_w[t, ok] = (c2w[t] @ np.column_stack([xcam, np.ones(len(xcam))]).T).T[:, :3]

    # ---------- P0: mesh ICP 到 query 帧物体点云 ----------
    mesh = trimesh.load(os.path.join(data, 'mesh.obj'))
    ms = trimesh.sample.sample_surface(mesh, 10000)[0]
    mm = m0 if q_t == 0 else np.asarray(fb[q_t]['mask']) > 0
    dep0 = depth[q_t]
    yy, xx = mm.nonzero()
    z0 = np.array([sample_depth_median(dep0, v, u, 1) for u, v in zip(xx, yy)])
    ok0 = np.isfinite(z0)
    pc0 = backproject_uv(xx[ok0].astype(float), yy[ok0].astype(float), z0[ok0], Ks[q_t])
    Xc0 = (c2w[q_t] @ np.column_stack([pc0, np.ones(len(pc0))]).T).T[:, :3]
    # 长轴朝向约束: GT 点云与 mesh 各自的 PCA 主轴 (bottle 等旋转对称物体的 ICP 朝向锚)
    _, _, vt_c = np.linalg.svd(Xc0 - Xc0.mean(0), full_matrices=False)
    _, _, vt_m = np.linalg.svd(ms - ms.mean(0), full_matrices=False)
    P0, rmse, fit, adot = icp_mesh_to_cloud(ms, Xc0, pca_init_rots(ms),
                                            axis_pair=(vt_m[0], vt_c[0]))
    print(f'ICP: rmse={rmse * 1000:.1f}mm fitness={fit:.2f} axis_dot={adot:.2f} '
          f'(mesh {len(mesh.vertices)}v, cloud {len(Xc0)}pts)')

    x_loc = (np.linalg.inv(P0) @ np.column_stack(
        [X_w[q_t], np.ones(len(queries))]).T).T[:, :3]
    ok_q = np.isfinite(x_loc).all(1) & np.isfinite(X_w[q_t]).all(1)
    x_loc = x_loc[ok_q]
    idx_keep = np.where(ok_q)[0]

    # ---------- 逐帧 Kabsch ----------
    Ps = []
    per_frame = {}
    prev = P0
    for t in range(T):
        X = X_w[t, idx_keep]
        ok = np.isfinite(X).all(1)
        if ok.sum() >= 8:
            w = (vis[t, idx_keep][ok] if vis.ndim == 2 else 1.0) * \
                (conf[t, idx_keep][ok] if conf.ndim == 2 else 1.0)
            w = np.clip(np.asarray(w, float), 1e-3, None)
            try:
                P, n_used, med_res = fit_pose(x_loc[ok], X[ok], w)
                per_frame[t] = {'n': n_used, 'med_res_mm': med_res * 1000}
                prev = P
            except Exception as e:
                per_frame[t] = {'error': str(e)}
        Ps.append(prev)
    Ps = np.stack(Ps)

    # ---------- 位姿中值滤波 (平移中值 + 旋转测地均值) ----------
    if args.med_win and T > 2:
        from scipy.spatial.transform import Rotation
        h = args.med_win // 2
        out = []
        for t in range(T):
            a, b = max(t - h, 0), min(t + h + 1, T)
            Rm = Rotation.from_matrix(Ps[a:b, :3, :3]).mean()
            tm = np.median(Ps[a:b, :3, 3], axis=0)
            out.append(np.vstack([np.column_stack([Rm.as_matrix(), tm]),
                                  [0, 0, 0, 1]]))
        Ps = np.stack(out)
        print(f'median filtered (window={args.med_win})')

    # ---------- savgol 平滑: 平移逐轴 + 旋转四元数 ----------
    if args.smooth_win and T > args.smooth_win:
        from scipy.spatial.transform import Rotation
        from scipy.signal import savgol_filter
        wlen = args.smooth_win if args.smooth_win % 2 == 1 else args.smooth_win + 1
        quat = Rotation.from_matrix(Ps[:, :3, :3]).as_quat()             # (T,4) xyzw
        # 逐帧符号连续化: 相邻帧点积为负则翻转, 避免 savgol 跨半球插值
        for i in range(1, len(quat)):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        qs = savgol_filter(quat, wlen, 3, axis=0)
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        Rs = Rotation.from_quat(qs).as_matrix()
        ts = savgol_filter(Ps[:, :3, 3], wlen, 3, axis=0)
        Ps = np.stack([np.vstack([np.column_stack([Rs[i], ts[i]]), [0, 0, 0, 1]])
                       for i in range(T)])
        print(f'savgol smoothed (window={wlen})')

    # ---------- 静止段锁定: 窗口5净位移<3cm 的段用中值位姿 ----------
    if args.lock_static and T > 5:
        from scipy.spatial.transform import Rotation
        win = 5
        net = np.linalg.norm(Ps[win:, :3, 3] - Ps[:-win, :3, 3], axis=1)
        static = np.ones(T, bool)          # net[i] 覆盖帧 i..i+win, 动则整窗标记运动
        for i in np.where(net >= 0.03)[0]:
            static[i:i + win + 1] = False
        # 按连续段取中值位姿
        i = 0
        n_locked = 0
        while i < T:
            if not static[i]:
                i += 1
                continue
            j = i
            while j < T and static[j]:
                j += 1
            seg = Ps[i:j]
            Rm = Rotation.from_matrix(seg[:, :3, :3]).mean()
            tm = np.median(seg[:, :3, 3], axis=0)
            Pm = np.vstack([np.column_stack([Rm.as_matrix(), tm]), [0, 0, 0, 1]])
            Ps[i:j] = Pm
            n_locked += j - i
            i = j
        print(f'static locked: {n_locked}/{T} frames')

    os.makedirs(os.path.join(out_dir, 'ob_in_world'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'ob_in_cam'), exist_ok=True)
    report = {'icp_rmse': rmse, 'icp_fitness': fit, 'n_query': len(queries),
              'per_frame': per_frame}
    for t in range(T):
        P = Ps[t]
        np.savetxt(os.path.join(out_dir, 'ob_in_world', f'{t:06d}.txt'), P)
        np.savetxt(os.path.join(out_dir, 'ob_in_cam', f'{t:06d}.txt'), w2c[t] @ P)

    c = np.asarray(mesh.vertices).mean(0)
    Ctr = np.einsum('tij,j->ti', Ps[:, :3, :3], c) + Ps[:, :3, 3]
    report['centroid_path_len'] = float(np.linalg.norm(np.diff(Ctr, axis=0), axis=1).sum())
    report['per_frame_step_mm'] = {
        'median': float(np.median(np.linalg.norm(np.diff(Ps[:, :3, 3], axis=0), axis=1)) * 1000)}
    with open(os.path.join(out_dir, 'spa_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f'done -> {out_dir} (质心轨迹总长 {report["centroid_path_len"]:.2f}m, '
          f'逐帧step中位 {report["per_frame_step_mm"]["median"]:.0f}mm)')


if __name__ == '__main__':
    main()
