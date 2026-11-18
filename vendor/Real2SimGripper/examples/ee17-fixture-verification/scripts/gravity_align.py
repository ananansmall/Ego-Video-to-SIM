#!/usr/bin/env python3
"""gravity_align.py — EF3R 世界重力方向提取 + 尺度/坐标系对齐 (M' 轴变换)。

背景 (2026-10-29 实测, 456483158):
  EF3R meta.json 声称世界 y-up, 但三个独立结构探针一致表明该导出的竖直方向
  ≈ +z (与 +y 夹角 ~68°):
    ① 桌面壳层: 沿实测重力的物体支撑面局部法向与重力夹角 1.0~2.3°
       (沿 +y 计算则 65°+);
    ② 场景云高度分布: 沿实测重力在 0.45m 处出现桌面尖峰 (沿 +y 完全抹开);
    ③ 静止物体薄轴 (最小惯性轴) ≈ +z。
  旧链路 (ef3r_to_targets / cb 的 M_EF3R_TO_SIM) 把 EF3R +y 当竖直 → 整个仿真
  世界带着 ~68° 倾斜在跑: 台面倾斜 (PICK 台面法向偏 69.8°)、放置落斜面、
  跨链 Umeyama 出现非物理的 18° 自由旋转。

方法 (借鉴 ananansmall/Ego-Video-to-SIM combination/001_align_scene.py 的
"重力对齐场景 + 首帧锚点 + 轨迹尺度"思想, 适配 EF3R 数据形态):
  探针 A (跨链, 首选): AGILE/MoGe 深度点云在手部正下方 RANSAC 拟合桌面 →
          相机系重力; 经 EF3R c2w 映射到世界系 (需同视频 AGILE prep)。
  探针 B (EF3R 内部回退): 场景云高度直方图峰度最大化 —— 在球面上搜索方向 d,
          使点云沿 d 的高度直方图最"尖" (水平桌面/地面产生尖峰, 倾斜则抹平)。
  验证: 物体支撑壳层 (物体中心下方 0.5~9cm 的点) 局部法向与重力的夹角,
        <5° 视为可信; 同时报告文档 +y 与重力的偏差。

输出 (缓存到 <ef3r-dir>/gravity_align.npz):
  g_ef        EF3R 世界系重力方向 (单位向量, 指向上)
  R_g         旋转: EF3R 世界 → 重力对齐世界 (g_ef → +y), 最小旋转
  M_sim       3x3: sim = M_sim @ p_ef, 其中 M_sim = M_EF3R_TO_SIM @ R_g
  method/validation 诊断字段
"""
import json
import os

import numpy as np

# EF3R 世界 (x右/y上[声称]/z后, 右手) -> sim 世界 (x右/y前/z上): 旧链路的轴变换
M_EF3R_TO_SIM = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]], np.float64)


def _rodrigues_toward(a, b):
    """最小旋转矩阵 R: a → b (a,b 单位向量)。"""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:                       # 反向: 任取正交轴转 180°
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a @ axis) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / ((1 + c)) * (1.0)


# ---------------------------------------------------------------- 探针 A
def gravity_from_moge(agile_seq_dir, aligned_dir, ef3r_c2w, max_frames=8,
                      seed=0, verbose=True):
    """MoGe 深度点云在手部正下方 RANSAC 桌面 → 相机系重力 → EF3R 世界系。

    agile_seq_dir: AGILE prep 目录 (含 moge/depth + cam_K.txt)
    aligned_dir:   AGILE aligned 目录 (optimized_mano_joints + camera_data)
    返回 (g_ef, info) 或 (None, info)。
    """
    import cv2
    import glob
    try:
        K = np.loadtxt(os.path.join(agile_seq_dir, "moge", "cam_K.txt"))
        ag = np.load(os.path.join(aligned_dir, "optimized_mano_joints.npz"))["mano_joints"][:, 0]
        pred = np.load(os.path.join(aligned_dir, "camera_data.npz"))["prediction_frame_ids"]
        dep_all = sorted(glob.glob(os.path.join(agile_seq_dir, "moge", "depth", "*.png")))
    except Exception as e:                                     # noqa: BLE001
        return None, {"error": f"AGILE prep 不可读: {e}"}
    step = max(1, len(ag) // max_frames)
    pts = []
    for i in range(0, len(ag), step):
        dep = cv2.imread(dep_all[pred[i]], cv2.IMREAD_UNCHANGED)
        if dep is None:
            continue
        dep = dep.astype(np.float32) / 1000.0
        h, w = dep.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = np.clip(ag[i, 0, 2], 1e-6, None)
        uw = ag[i, 0, 0] / z * K[0, 0] + K[0, 2]
        vw = ag[i, 0, 1] / z * K[1, 1] + K[1, 2]
        # 手腕投影正下方的图像窗 (手腕以下 26cm、左右 ±220px) = 桌面区域
        m = ((np.abs(u - uw) < 220) & (v > vw) & (v < vw + 260)
             & np.isfinite(dep) & (dep > 0.3) & (dep < 1.2))
        pts.append(np.stack([(u[m] - K[0, 2]) / K[0, 0] * dep[m],
                             (v[m] - K[1, 2]) / K[1, 1] * dep[m], dep[m]], -1))
    if not pts:
        return None, {"error": "MoGe 深度窗为空"}
    Pc = np.vstack(pts)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(1200):
        i = rng.choice(len(Pc), 3, replace=False)
        p0, p1, p2 = Pc[i]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d0 = -n @ p0
        inl = np.abs(Pc @ n + d0) < 0.006
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 500:
        return None, {"error": "RANSAC 平面内点不足"}
    S = Pc[best]
    c = S.mean(0)
    _, _, V = np.linalg.svd(S - c, full_matrices=False)
    n_cam = V[2]
    if n_cam[2] < 0:
        n_cam = -n_cam                                        # 朝相机 (up)
    # → EF3R 世界 (跨锚点平均, 报告散布)
    vf = np.asarray(pred, np.int64) * 3
    ds = np.stack([ef3r_c2w[min(k, len(ef3r_c2w) - 1), :3, :3] @ n_cam for k in vf])
    ds /= np.linalg.norm(ds, axis=1, keepdims=True)
    g = ds.mean(0)
    g /= np.linalg.norm(g)
    spread = float(np.degrees(np.arccos(np.clip(ds @ ds[0], -1, 1)).max()))
    if verbose:
        print(f"  探针A (MoGe 桌面, {len(S)} 内点): 相机系 {np.round(n_cam,3)} → "
              f"EF3R 世界 {np.round(g,3)} (锚点间散布 {spread:.1f}°)")
    return g, {"n_inliers": int(best.sum()), "spread_deg": spread,
               "n_cam": n_cam.tolist()}


# ---------------------------------------------------------------- 探针 B
def _hist_peakiness(P, d, lo_q=1, hi_q=99, bins=90):
    """点云沿方向 d 的高度直方图峰度 (最大 bin 占比, 越尖越好)。"""
    h = P @ d
    lo, hi = np.percentile(h, [lo_q, hi_q])
    cnt, _ = np.histogram(h, bins=bins, range=(lo, hi))
    return cnt.max() / max(cnt.sum(), 1)


def gravity_from_scene(P, palm=None, verbose=True, seed=0):
    """EF3R 内部回退: 球面搜索使高度直方图最尖的方向 (水平面准则)。

    P: (N,3) 场景点云 (工作区邻域)。palm: (M,3) 掌心轨迹 (把搜索权重集中在
    手部工作区, 弱化远处墙面干扰)。
    """
    rng = np.random.default_rng(seed)
    if palm is not None and len(palm):
        ctr = palm.mean(0)
        keep = np.linalg.norm(P[:, :2] - ctr[None, [0, 2]], axis=1) < 0.9 \
            if False else np.ones(len(P), bool)
        # 工作区: 水平距离 <1.0m 且沿候选竖直 ±0.8m 由直方图自身裁剪, 这里先粗筛
        keep = np.linalg.norm(P - ctr, axis=1) < 1.5
        Pw = P[keep]
    else:
        Pw = P
    if len(Pw) > 60000:
        Pw = Pw[rng.choice(len(Pw), 60000, replace=False)]
    # 斐波那契球面粗搜索
    n_dirs = 400
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs)
    theta = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    scores = np.array([_hist_peakiness(Pw, d) for d in dirs])
    # 只接受"上方向与文档 +y 夹角 <80°"的候选 (排除完全倒置的等价解)
    ok = dirs @ np.array([0, 1, 0]) > -0.17
    scores[~ok] = -1
    d_best = dirs[int(np.argmax(scores))]
    # 局部细化 (扰动 30° 内随机重采样)
    best_s, best_d = scores.max(), d_best
    for _ in range(200):
        d = best_d + rng.normal(0, 0.25, 3)
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        d = d / n
        if d @ np.array([0, 1, 0]) < -0.17:
            continue
        s = _hist_peakiness(Pw, d)
        if s > best_s:
            best_s, best_d = s, d
    if verbose:
        print(f"  探针B (场景云直方图峰度): 方向 {np.round(best_d,3)} "
              f"峰度 {best_s:.3f} (均匀基线 {1/90:.3f})")
    return best_d, {"peakiness": float(best_s)}


# ---------------------------------------------------------------- 验证
def validate_gravity(P, anchors, g, band=(-0.09, -0.005), tol_deg=5.0, seed=0):
    """支撑壳层验证: 物体中心下方 band 内的点, 局部法向与 g 的夹角。

    anchors: {name: (3,) 中心}。返回 {name: 角度}。
    """
    rng = np.random.default_rng(seed)
    out = {}
    for nm, cen in anchors.items():
        sel = P[(np.abs(P[:, 0] - cen[0]) < 0.4) & (np.abs(P[:, 2] - cen[2]) < 0.4)]
        if len(sel) > 80000:
            sel = sel[rng.choice(len(sel), 80000, replace=False)]
        h = (sel - cen) @ g
        shell = sel[(h > band[0]) & (h < band[1])]
        if len(shell) < 200:
            out[nm] = None
            continue
        c = shell.mean(0)
        _, _, V = np.linalg.svd(shell - c, full_matrices=False)
        n = V[2]
        if n @ g < 0:
            n = -n
        out[nm] = float(np.degrees(np.arccos(np.clip(n @ g, -1, 1))))
    return out


# ---------------------------------------------------------------- 支撑面拟合
def fit_support_surface(P, center, g, half=0.28, band=(-0.09, -0.005),
                        grid=14, min_pts=150, seed=0):
    """重力感知的支撑台面拟合: 物体中心下方 band 壳层 → 水平网格面片。

    旧 fit_surface 在未知重力下做 PCA, 混入墙面/地板 (456483158 实测 PICK
    "台面"法向偏 69.8°)。重力已知后问题变成: 壳层内点的 (沿重力) 高度中位数
    即桌面高度, 面片 = 该高度上 half×half 的水平网格 —— 真实桌子就是水平的。
    返回 (V (n,3), F (m,3), 高度, 内点数) 或 (None, None, None, 0)。
    """
    rng = np.random.default_rng(seed)
    sel = P[(np.abs(P[:, 0] - center[0]) < half + 0.15)
            & (np.abs(P[:, 2] - center[2]) < half + 0.15)]
    if len(sel) > 80000:
        sel = sel[rng.choice(len(sel), 80000, replace=False)]
    if len(sel) < min_pts:
        return None, None, None, 0
    h = (sel - center) @ g
    shell = sel[(h > band[0]) & (h < band[1])]
    if len(shell) < min_pts:
        return None, None, None, 0
    h_top = float(np.median((shell - center) @ g))
    # 壳层内点在重力⊥平面上的范围 → 面片覆盖实际支撑区
    e1 = np.cross(g, [1, 0, 0])
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(g, e1)
    uv = (shell - center) @ np.stack([e1, e2], 1)
    ext = np.percentile(np.abs(uv), 92, axis=0)
    ext = np.clip(ext, 0.10, half)
    us = np.linspace(-ext[0], ext[0], grid)
    vs = np.linspace(-ext[1], ext[1], grid)
    base = center + g * h_top
    V = np.array([[base + e1 * u + e2 * v for u in us] for v in us]).reshape(-1, 3)
    F = []
    for r in range(grid - 1):
        for cidx in range(grid - 1):
            a = r * grid + cidx
            F += [[a, a + 1, a + grid], [a + 1, a + grid + 1, a + grid]]
    return V, np.array(F, np.int64), h_top, len(shell)


# ---------------------------------------------------------------- 主接口
def build_gravity_align(ef3r_dir, agile_seq_dir=None, aligned_dir=None,
                        force=False, verbose=True):
    """估计重力并缓存 <ef3r-dir>/gravity_align.npz。返回该 npz 的 dict。"""
    cache = os.path.join(ef3r_dir, "gravity_align.npz")
    if os.path.exists(cache) and not force:
        d = dict(np.load(cache, allow_pickle=True))
        if verbose:
            print(f"gravity_align: 复用缓存 {cache} (g={np.round(d['g_ef'],3)}, "
                  f"method={str(d['method'])})")
        return d

    P = np.load(os.path.join(ef3r_dir, "pointcloud", "scene_cloud.npz"))["points"]
    c2w = np.load(os.path.join(ef3r_dir, "camera", "traj_c2w.npy"))
    ef = np.load(os.path.join(ef3r_dir, "hands", "joint_xyz_world.npy"))
    palm = ef[:, 1 if ef.shape[1] == 2 else 0][:, [0, 5, 9, 13, 17]].mean(1)
    anchors = {}
    obj_p = os.path.join(ef3r_dir, "object_instance.npz")
    if os.path.exists(obj_p):
        od = np.load(obj_p)
        for k, nm in (("center_pre", "PICK"), ("center_placed", "RELEASED")):
            if k in od.files:
                anchors[nm] = od[k]

    info = {"ef3r_dir": os.path.basename(ef3r_dir.rstrip("/"))}
    anchors_any = bool(anchors)

    def _score(g):
        if not anchors_any:
            return 1e9
        val = validate_gravity(P, anchors, g)
        vals = [v for v in val.values() if v is not None]
        return float(np.mean(vals)) if vals else 1e9

    # 探针 B (EF3R 内部, 稳定) —— 总是跑
    gB, info_b = gravity_from_scene(P, palm, verbose=verbose)
    info["probeB"] = info_b
    cand = [(gB, "scene_hist_peak", info_b)]
    # 探针 A (跨链 MoGe) —— 有 AGILE prep 时跑, 作为交叉验证
    if agile_seq_dir and aligned_dir and os.path.isdir(agile_seq_dir):
        gA, info_a = gravity_from_moge(agile_seq_dir, aligned_dir, c2w, verbose=verbose)
        info["probeA"] = info_a
        if gA is not None:
            cand.append((gA, "moge_table", info_a))
    # 择优: 支撑壳层验证角最小者优先 (两探针同时可用时互为交叉验证)
    scored = [( _score(g), g, m, ei) for g, m, ei in cand]
    scored.sort(key=lambda t: t[0])
    s_best, g, method, _ei = scored[0]
    info["probe_agreement_deg"] = (round(float(np.degrees(np.arccos(np.clip(
        cand[0][0] @ cand[-1][0], -1, 1)))), 2) if len(cand) > 1 else None)
    info["selected_by_shell_deg"] = round(s_best, 2) if s_best < 1e8 else None

    val = validate_gravity(P, anchors, g) if anchors_any else {}
    info["validation_shell_deg"] = {k: (round(v, 2) if v is not None else None)
                                    for k, v in val.items()}
    tilt_doc = float(np.degrees(np.arccos(np.clip(g @ np.array([0, 1, 0.0]), -1, 1))))
    info["tilt_vs_doc_y_deg"] = round(tilt_doc, 2)
    R_g = _rodrigues_toward(g, np.array([0, 1, 0.0]))
    M_sim = M_EF3R_TO_SIM @ R_g
    if verbose:
        print(f"  重力 (EF3R 世界): {np.round(g,4)} | 与文档+y 偏差 {tilt_doc:.1f}° | "
              f"壳层验证 {info['validation_shell_deg']} | 方法 {method}")
        print(f"  M_sim = M_EF3R_TO_SIM @ R_g 已生成 (sim = M_sim @ p_ef)")
    np.savez(cache, g_ef=g, R_g=R_g, M_sim=M_sim, method=np.array(method),
             tilt_vs_doc_y_deg=tilt_doc,
             validation=json.dumps(info["validation_shell_deg"]),
             info=json.dumps(info, default=str))
    return dict(np.load(cache, allow_pickle=True))


def load_or_build(ef3r_dir, verbose=False):
    """cb/转换器用: 有缓存读缓存, 没有则现场估计 (静默)。失败返回 None。"""
    try:
        return build_gravity_align(ef3r_dir, verbose=verbose)
    except Exception:                                          # noqa: BLE001
        return None


def sim_rotation(ef3r_dir):
    """返回 M_sim (3x3) 或 None (不可得时调用方回退 M_EF3R_TO_SIM)。"""
    d = load_or_build(ef3r_dir)
    if d is None or "M_sim" not in d:
        return None
    return np.asarray(d["M_sim"], np.float64)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef3r-dir", required=True)
    ap.add_argument("--agile-seq-dir", default=None)
    ap.add_argument("--aligned-dir", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_gravity_align(a.ef3r_dir, a.agile_seq_dir, a.aligned_dir, a.force)
