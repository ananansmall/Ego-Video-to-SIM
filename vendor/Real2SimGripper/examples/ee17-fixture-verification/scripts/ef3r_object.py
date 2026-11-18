#!/usr/bin/env python3
"""ef3r_object.py — 从 EgoFound3R 深度点云里分割**被抓物体实例**, 简单拟合 (凸包) 成 OBJ。

物体中心 = PICK 段手部质心 (手正握着物体); 物体点 = 该中心半径内的场景点
(只取 PICK 前后锚点帧 —— 物体还在桌上/刚离桌的形态, 作为仿真里的初始摆放)。
输出 npz: verts/faces (EF3R 世界系, cb.main 做 geo 对齐后生成 actor),
          center_pre / center_placed (放置成功判定的参考位置)。
"""
import argparse
import json
import os

import numpy as np
from scipy.spatial import ConvexHull


def unproject_anchors(ef3r_dir, hand, stride=2, zmin=0.05, zmax=2.5):
    meta = json.load(open(os.path.join(ef3r_dir, "meta.json")))
    anchors = meta["camera"]["anchors"]
    W, H = meta["camera"]["image_size_hw"][1], meta["camera"]["image_size_hw"][0]
    K_all = np.load(os.path.join(ef3r_dir, "camera/K_full.npy")).reshape(-1, 3, 3)
    c2w = np.load(os.path.join(ef3r_dir, "camera/traj_c2w.npy"))
    dep = np.load(os.path.join(ef3r_dir, "camera/depth_metric_global.npy"))
    Jw = np.load(os.path.join(ef3r_dir, "hands/joint_xyz_world.npy"))[:, 1 if hand == "right" else 0]
    us, vs = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    out = {}
    for i, f in enumerate(anchors):
        if f >= Jw.shape[0]:
            continue
        K = K_all[f]
        z = dep[i][vs, us]
        m = np.isfinite(z) & (z > zmin) & (z < zmax)
        u, v, zz = us[m], vs[m], z[m]
        p_cam = np.stack([(u - K[0, 2]) / K[0, 0] * zz,
                          (v - K[1, 2]) / K[1, 1] * zz, zz], axis=1)
        T = c2w[f]
        out[f] = p_cam @ T[:3, :3].T + T[:3, 3]
    return out, Jw, anchors


def load_rgb(ef3r_dir, frames_dir, f, W, H):
    """抽帧彩色化 (0500.jpg 命名, resize 到画布)。无帧目录返回 None。"""
    from PIL import Image
    if not frames_dir:
        return None
    p = os.path.join(frames_dir, f"{f:04d}.jpg")
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("RGB").resize((W, H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef3r-dir", required=True)
    ap.add_argument("--hand", default="right", choices=["left", "right"])
    ap.add_argument("--radius", type=float, default=0.08)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--frames-dir", default=None, help="原始抽帧 (亮度分割用)")
    ap.add_argument("--f2g-results", default=None, help="F2G-Pose results.npz (物体逐帧位轨迹)")
    ap.add_argument("--min-thick", type=float, default=0.024, help="薄轴最小厚度 (m), 补偿深度欠测量")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.ef3r_dir, "meta.json")))
    W, H = meta["camera"]["image_size_hw"][1], meta["camera"]["image_size_hw"][0]
    pts_by_f, Jw, anchors = unproject_anchors(args.ef3r_dir, args.hand, args.stride)
    N = Jw.shape[0]
    states = {}
    sl = os.path.join(args.ef3r_dir, "state_labels.json")
    if os.path.exists(sl):
        states = json.load(open(sl))["states"]
    pick = states.get("PICK", (int(N * 0.2), int(N * 0.4)))
    rel = states.get("RELEASED", (int(N * 0.75), N))
    # 物体点选窗用**校准抓取帧** (接触概率峰值) 而非 PICK 段起点: PICK[0] 时刻
    # 手可能还在途中 (实测 pickplace_7: PICK=F22 手在盘子上方, 掌心球抓到旁边
    # 蓝色玩偶; 真实接触在 F27)。无校准时退回 PICK 窗。
    _gf = None
    _cal = os.path.join(args.ef3r_dir, "pick_place_calib.npz")
    if os.path.exists(_cal):
        _gf = int(np.load(_cal)["grasp_frame"])
    _cwin = (_gf - 2, _gf + 5) if _gf is not None else pick

    def palm_centroid(f0, f1):
        fs = [f for f in anchors if f0 <= f < min(f1, N)]
        if not fs:                      # 短段/越界窗 (如 3 帧 RELEASED + 偏移): 取 ≤f0 最近锚点
            _fc = max(0, min(f0, N - 1))
            _below = [f for f in anchors if f <= _fc]
            fs = [_below[-1]] if _below else [anchors[0]]
        return np.concatenate([Jw[f][[0, 5, 9, 13, 17]] for f in fs]).reshape(-1, 3).mean(0)

    c_pre = palm_centroid(_cwin[0], _cwin[1])                   # 物体在桌上/手中 (抓取时刻)
    c_placed = palm_centroid(rel[0] + 10, min(rel[1], N) - 1)   # 放置参考 (RELEASED 段)
    print(f"物体中心(抓取帧窗 {_cwin}): {np.round(c_pre,3)}  放置参考(RELEASED): {np.round(c_placed,3)}")

    obj, lum_all = [], []
    def _extract(lum_max):
        """掌心球+高度带(+可选亮度上限)过滤。亮度分档重试: 0.30 深色物体 →
        0.55 中等 → None 不过滤 (棕色/透明/高光物体会被硬阈值全灭, 实测 ee13)。"""
        obj, lum_all = [], []
        for f, P in pts_by_f.items():
            if not (_cwin[0] - 15 <= f <= _cwin[1] + 20):  # 接近+抓取前后: 物体未被手遮挡
                continue
            rgb = load_rgb(args.ef3r_dir, args.frames_dir, f, W, H)
            d_hand = np.linalg.norm(P[:, None, :] - Jw[f][None, :, :], axis=2).min(1)
            up_band = np.abs(P[:, 1] - c_pre[1]) < 0.07      # EF3R y-up: 掌心±7cm 高度带
            near = (np.linalg.norm(P - c_pre, axis=1) < args.radius) & (d_hand > 0.02) & up_band
            if rgb is not None:
                v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                px = np.stack([u.ravel(), v.ravel()], 1)
                from scipy.spatial import cKDTree
                K_all = np.load(os.path.join(args.ef3r_dir, "camera/K_full.npy")).reshape(-1, 3, 3)
                K = K_all[f]
                zz = np.load(os.path.join(args.ef3r_dir, "camera/depth_metric_global.npy"))[anchors.index(f)]
                uu = (P[:, 0] / P[:, 2] * K[0, 0] + K[0, 2]).round().astype(int)
                vv = (P[:, 1] / P[:, 2] * K[1, 1] + K[1, 2]).round().astype(int)
                ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                lum = np.full(len(P), 1.0)
                lum[ok] = rgb[vv[ok], uu[ok]].mean(1) / 255.0
                sel = near if lum_max is None else near & (lum < lum_max)
                obj.append(P[sel]); lum_all.append(lum[sel])
            else:
                obj.append(P[near])
        return np.concatenate(obj) if obj else np.empty((0, 3))

    O = _extract(0.30)
    if len(O) < 30:
        O = _extract(0.55)
        print(f"  亮度档 0.30 不足, 放宽到 0.55: {len(O)} 点")
    if len(O) < 30:
        O = _extract(None)
        print(f"  亮度档 0.55 仍不足, 取消亮度过滤: {len(O)} 点")
    print(f"亮度+半径过滤后 {len(O)} 点")

    # ---- 深度空洞降级: 透明/反光玻璃器皿不返回深度 (实测 ee13 棕色试剂瓶,
    #      球内 0 点, 最近点=11cm 下桌面) —— 写**掌心处占位盒**契约, 让管线
    #      活到 SAM3D 阶段 (RGB 能看到玻璃; 4b 成功即替换, 失败则凸包盒兜底)。
    if len(O) < 20:
        _sz = np.array([0.03, 0.03, 0.05])
        cal = os.path.join(args.ef3r_dir, "pick_place_calib.npz")
        _off = np.zeros(3)
        if os.path.exists(cal):
            _off = np.asarray(np.load(cal)["obj_in_palm_ef3r"], float)
        _ctr = c_pre + _off + np.array([0, 0, -_sz[2]])   # y-up 世界: 盒在掌心下方
        _v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                       [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * _sz + _ctr
        _f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                       [2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]])
        out = args.out or os.path.join(args.ef3r_dir, "object_instance.npz")
        np.savez(out, verts=_v.astype(np.float32), faces=_f.astype(np.int64),
                 center_pre=_ctr, center_placed=_ctr, obj_points=np.empty((0,3)),
                 n_points=0, pca_center=_ctr, pca_axes=np.eye(3),
                 pca_ext=_sz*2, source="depth_void_fallback")
        print(f"  ⚠ 深度空洞 (玻璃/反光), 写掌心占位盒 {np.round(_sz*2000).tolist()}mm @ {np.round(_ctr,3).tolist()}")
        print(f"  ✓ {out} (等 SAM3D 用 RGB 重建真实形状)")
        return

    # ---- 最大连通域 (体素 BFS, 去掉零星杂物) ----
    vox = 0.010
    key = np.floor(O / vox).astype(np.int64)
    keyset = {tuple(k): i for i, k in enumerate(key)}
    adj = {}
    for k in keyset:
        nbrs = [tuple(np.array(k) + d) for d in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]
        adj[keyset[k]] = [keyset[n] for n in nbrs if n in keyset]
    seen, best = set(), []
    for start in list(adj):
        if start in seen: continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            cur = stack.pop(); comp.append(cur)
            for n in adj[cur]:
                if n not in seen: seen.add(n); stack.append(n)
        if len(comp) > len(best): best = comp
    O = O[np.array(sorted(best), dtype=np.int64)] if best else O
    print(f"最大连通域: {len(O)} 点 (体素 12mm)"
          f"{'' if best else ' —— 连通域为空, 保留全部过滤点 (半径重试见上)'}")

    # 离群修剪 (p80): 残余阴影/桌面点再滤一轮
    ctr = O.mean(0)
    d_ctr = np.linalg.norm(O - ctr, axis=1)
    O = O[d_ctr < np.percentile(d_ctr, 80)]

    hull = ConvexHull(O)
    remap = {old_i: new_i for new_i, old_i in enumerate(hull.vertices)}
    verts = O[hull.vertices]
    faces = np.array([[remap[a], remap[b], remap[c]] for a, b, c in hull.simplices])
    # 面片朝向修正: 凸包质心在内部, 法线应指向外侧 (与质心→面心同向), 否则
    # 渲染时全部背面剔除 → 物体不可见 (碰撞不受影响, 曾致"物理存在视觉消失")
    _c = verts.mean(0)
    for i, (a, b, c3) in enumerate(faces):
        n = np.cross(verts[b] - verts[a], verts[c3] - verts[a])
        fc = (verts[a] + verts[b] + verts[c3]) / 3
        if n @ (fc - _c) < 0:
            faces[i] = [a, c3, b]
    # PCA 主轴: axes[:,0]=最长轴 ... axes[:,2]=最薄轴 (夹爪开合方向应对准它)
    ctr2 = verts.mean(0)
    _, _, Vt = np.linalg.svd(verts - ctr2)
    axes = Vt.T
    ext_pca = np.ptp((verts - ctr2) @ axes, axis=0)
    print(f"凸包: {len(verts)} 顶点 {len(faces)} 面 | PCA 轴长 {np.round(ext_pca*1000,1)} mm (长→薄)")
    # 深色薄物体深度欠测量 (背面遮挡+吸收): 沿薄轴补厚到 --min-thick (默认 24mm),
    # 否则 R1 夹爪最小内距 17mm 夹不到 13mm 物体。补厚以中心对称。
    if ext_pca[2] < args.min_thick:
        coords = (verts - ctr2) @ axes[:, 2]
        scale_t = args.min_thick / max(ext_pca[2], 1e-6)
        verts = verts + np.outer(coords * (scale_t - 1), axes[:, 2])
        ext_pca[2] = args.min_thick
        print(f"  薄轴补厚: {coords.max()-coords.min():.3f}m → {args.min_thick}m (深度欠测量补偿)")

    # ---- F2G-Pose 物体轨迹 (可选): 逐帧 6D 位姿 → EF3R 世界系 ----
    # T_cam_obj_smooth(相机系) 与 EF3R 手部关节同一相机系 (OpenCV, 同视频帧),
    # 组合 EF3R 的 c2w 得到物体在世界系的逐帧位姿 —— 抓取跟随的参考轨迹。
    traj_T = None
    f2g = os.path.join(os.path.dirname(args.ef3r_dir.rstrip("/")), "video_456483158",
                       "f2g_results", "results.npz")
    f2g = args.f2g_results or (f2g if os.path.exists(f2g) else None)
    if f2g:
        from scipy.spatial.transform import Rotation as Rot
        fg = np.load(f2g)
        T_co = fg["T_cam_obj_smooth"]                      # (300,4,4) 相机→物体
        c2w_all = np.load(os.path.join(args.ef3r_dir, "camera/traj_c2w.npy"))
        traj_T = np.einsum("nij,njk->nik", c2w_all, T_co)  # 世界系逐帧位姿
        # 一致性诊断: F2G (MoGe 深度) 与 EF3R 手部 (自标定米制) 的深度尺度不一致时
        # 物体-掌心距离会漂移 —— 此时绝对轨迹不可混用, 只能用手系刚体跟随 (cb 端默认)
        hold_w = states.get("HOLD", (int(N*0.5), int(N*0.6)))
        d_ho = [np.linalg.norm(traj_T[f][:3, 3] - Jw[f][[0, 5, 9, 13, 17]].mean(0))
                for f in range(hold_w[0], min(hold_w[1], N), 10)]
        _rigid = np.std(d_ho) < 0.03
        print(f"F2G 物体轨迹: {len(traj_T)} 帧 | HOLD 段 物体-掌心距离 "
              f"mean {np.mean(d_ho)*1000:.1f}mm std {np.std(d_ho)*1000:.1f}mm → "
              f"{'刚体一致, 可用 F2G 轨迹' if _rigid else '深度尺度不一致 (MoGe vs 手部标定), cb 端将用手系刚体跟随'}")

    # ---- 支撑台面拟合 (点云→面片): PICK / RELEASED 区域各拟合一块台面 ----
    # v2 (2026-10-29): 重力感知 —— gravity_align 给出真实竖直方向后, 台面 = 物体
    # 下方壳层的水平网格 (旧版未知重力下 PCA 选面, PICK 台面法向曾偏 69.8°)。
    import gravity_align as ga
    _ginfo = ga.load_or_build(args.ef3r_dir)
    _g_ef = np.asarray(_ginfo["g_ef"], np.float64) if _ginfo else None
    if _g_ef is not None:
        print(f"  支撑台面: 重力感知拟合 (g_ef={np.round(_g_ef,3)}, "
              f"与文档+y 偏 {float(_ginfo['tilt_vs_doc_y_deg']):.1f}°)")

    def fit_surface(center, f_window, radius=0.28):
        """支撑台面 mesh: 重力感知水平网格; 无重力数据回退旧 PCA-Delaunay。"""
        from scipy.spatial import Delaunay
        pts = []
        for f, P in pts_by_f.items():
            if not (f_window[0] <= f < f_window[1]):
                continue
            if len(P) > 4000:
                P = P[::len(P) // 4000 + 1]
            near = np.linalg.norm(P - center, axis=1) < radius
            d_hand = np.linalg.norm(P[:, None, :] - Jw[f][None, :, :], axis=2).min(1)
            pts.append(P[near & (d_hand > 0.03)])
        P = np.concatenate(pts) if pts else np.zeros((1, 3))
        if _g_ef is not None:
            V, F, h_top, n_shell = ga.fit_support_surface(P, center, _g_ef, half=radius)
            if V is not None:
                print(f"    surface mesh (重力感知): {len(V)} 顶点 水平网格 | "
                      f"壳层 {n_shell} 点 | 台高(沿重力) {h_top*1000:.0f}mm")
                e1 = np.cross(_g_ef, [1, 0, 0])
                e1 /= max(np.linalg.norm(e1), 1e-9)
                e2 = np.cross(_g_ef, e1)
                return dict(center=center, R=np.stack([e1, e2, _g_ef], axis=1),
                            V=V, F=F, half_u=radius, half_v=radius, n_pts=len(P))
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c)
        n, u, v = Vt[2], Vt[0], Vt[1]
        pu, pv, pn = (P - c) @ u, (P - c) @ v, (P - c) @ n
        # 只保留主表面层 (去手/物体下方的悬空点): 法向 |pn| < 3cm 的主层
        main = np.abs(pn - np.median(pn)) < 0.03
        if main.sum() < 12:
            raise RuntimeError("主表面层点不足")
        try:
            tri = Delaunay(np.stack([pu[main], pv[main]], 1))
        except Exception as e:      # 退化/共面点 (Qhull 错误): 放弃该支撑面
            raise RuntimeError(f"Delaunay 退化: {e}") from e
        V = c + np.outer(pu[main], u) + np.outer(pv[main], v) + np.outer(pn[main], n)
        F = tri.simplices
        ext_u = np.ptp(pu[main]); ext_v = np.ptp(pv[main])
        print(f"    surface mesh: {len(V)} 顶点 {len(F)} 面 | 范围 {ext_u*200:.0f}x{ext_v*200:.0f}mm")
        return dict(center=c, R=np.stack([u, v, n], axis=1), V=V, F=F,
                    half_u=ext_u / 2, half_v=ext_v / 2, n_pts=len(P))
    surf_pick = surf_place = None
    for _nm, _c, _win in (("PICK", c_pre, (pick[0] - 20, pick[1] + 20)),
                          ("PLACE", c_placed, (rel[0], min(rel[1], N)))):
        try:
            _sf = fit_surface(_c, _win)
            if _nm == "PICK":
                surf_pick = _sf
            else:
                surf_place = _sf
            print(f"支撑台面 {_nm}: {_sf['half_u']*2000:.0f}mm 见方 ({_sf['n_pts']}点)")
        except (RuntimeError, Exception) as _e:
            print(f"  ⚠ 支撑台面 {_nm} 拟合失败 (跳过, 管线继续): {str(_e)[:80]}")
            if _nm == "PICK":
                surf_pick = None
    if surf_pick is None and surf_place is None:
        print("  ⚠ 两侧支撑面均失败 —— 物体契约仍输出 (无支撑面)")

    # 放置点支撑面: RELEASED 段锚点中, 靠近 c_placed 的场景点的支撑高度
    # v2: 重力感知 —— 沿 g_ef 的壳层中位高度 (旧版用 y 分位, 混入墙面点)
    sup_pts = []
    for f, P in pts_by_f.items():
        if not (rel[0] <= f < min(rel[1], N)):
            continue
        near = np.linalg.norm(P - c_placed, axis=1) < 0.25
        sup_pts.append(P[near])
    sup = np.concatenate(sup_pts) if sup_pts else np.zeros((1, 3))
    if len(sup) == 0:                 # 锚点全空 (list 非空但元素皆空数组)
        sup = np.zeros((1, 3))
    if _g_ef is not None:
        h_sup = (sup - c_placed) @ _g_ef
        shell = h_sup[(h_sup > -0.09) & (h_sup < -0.005)]
        h_top = float(np.median(shell)) if len(shell) else -0.02
        sup_pt = c_placed + _g_ef * h_top                # 支撑面上一点 (EF3R 系)
        print(f"放置支撑面: 沿重力壳层高度 {h_top*1000:.0f}mm ({len(shell)} 点)")
    else:
        sup_z = float(np.percentile(sup[:, 1], 90))      # y-up: 表面 = 高位分位
        sup_pt = np.array([c_placed[0], sup_z, c_placed[2]])
        print(f"放置支撑面: y={sup_z:.3f} (RELEASED 段 {len(sup)} 点拟合)")

    out = args.out or os.path.join(args.ef3r_dir, "object_instance.npz")
    payload = dict(verts=verts.astype(np.float64), faces=faces.astype(np.int64),
                   center_pre=c_pre, center_placed=c_placed, obj_points=O, n_points=len(O),
                   pca_center=ctr2, pca_axes=axes, pca_ext=ext_pca,
                   support_placed=sup_pt)
    if traj_T is not None:
        payload["traj_T"] = traj_T
    if surf_pick is not None:        # 拟合失败的键不写入 (cb 走 support_placed 回退)
        payload["surface_pick_V"] = surf_pick["V"]; payload["surface_pick_F"] = surf_pick["F"]
    if surf_place is not None:
        payload["surface_place_V"] = surf_place["V"]; payload["surface_place_F"] = surf_place["F"]
    np.savez(out, **payload)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
