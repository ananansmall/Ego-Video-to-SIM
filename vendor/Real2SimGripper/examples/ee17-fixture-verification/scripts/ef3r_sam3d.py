#!/usr/bin/env python3
"""ef3r_sam3d.py — 用 SAM3D 给 EF3R 快路径重建**高质量物体 mesh** (用户标准链
"先 SAM3D 建模被抓物体" 的落地)。

快路径原物体模型 = 深度点云凸包 (39 顶点的粗糙壳, 鸭头/嘴全糊)。SAM3D 从
单帧 RGB+mask 重建带几何细节的 mesh。本脚本:

  1. 选帧: PICK 前 5~25 帧窗口的**锚点帧**, 取物体点云投影在画面内点数最多者
     (物体在画面里最大且未被手遮挡)
  2. EF3R 物体点云 (object_instance.npz 的 obj_points, 相机世界系) 经 c2w 投影
     到该帧 → 物体 mask (凸包填充+膨胀; 复用 EF3R 自身分割, 不引入新模型)
  3. 调 SAM3D (conda sam3d env, generate_mesh_sam3d.py) 重建带纹理 mesh
  4. mesh (自身系/尺度) → PCA 轴匹配 (免对应) 对齐 EF3R 物体点云 → 世界系
  5. 写 object_instance_sam3d.npz: 与凸包版**同一契约**, cb_large_range_sim 无感替换

⚠ K 分辨率: EF3R 的 K (K_full/K_global) 是其**处理分辨率** (512×288) 的,
投影到视频帧 (1280×720) 必须按 W/2cx, H/2cy 缩放 —— 否则 mask 落在画面
左上角 (实测鸭子上方 2.5 倍偏差)。

用法 (robotwin_curobo env):
    python ef3r_sam3d.py --ef3r-dir <dir> --video <mp4> [--gpu 0] [--frame N]
输出: <ef3r-dir>/object_instance_sam3d.npz (失败时保留凸包版, 管线不中断)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

SAM3D_DIR = "/path/to/project_data/mano_ws/do-as-i-do/reconstruction/modules/sam-3d-objects"
SAM3D_CFG = ("/path/to/project_data/mano_ws/do-as-i-do/reconstruction/"
             "weights/sam3d_shared/hf/checkpoints/pipeline.yaml")
CONDA_SH = "/path/to/user_home/miniconda3/etc/profile.d/conda.sh"


def k_for_video(K, W, H):
    """EF3R 处理分辨率 K → 视频分辨率 K (均匀缩放, 主点×2 = 图宽/高)。"""
    scx = W / (2.0 * K[0, 2])
    scy = H / (2.0 * K[1, 2])
    # xy 允许小幅独立缩放 (源视频非 16:9 时 EF3R 畸变缩放到 512×288, 实测
    # ee13: scx=3.75 vs scy=3.81 差 1.6%); 大偏差仍报错 (约定破裂)
    if abs(scx - scy) / max(scx, scy) > 0.05:
        raise ValueError(f"K 缩放严重非均匀 scx={scx:.3f} scy={scy:.3f}, 检查分辨率约定")
    Ks = K.copy()
    Ks[0] *= scx
    Ks[1] *= scy
    return Ks


def grasp_frame(ef3r_dir, N):
    """抓取时刻: 优先 fewframe 校准帧 (接触概率峰值)。PICK[0] 时手可能还在
    途中 (实测 pickplace_7: PICK=F22 手在盘子上方, 校准帧=F27 指尖才到鸭)。"""
    f = None
    cal = os.path.join(ef3r_dir, "pick_place_calib.npz")
    if os.path.exists(cal):
        f = int(np.load(cal)["grasp_frame"])
    if f is None:
        sl = os.path.join(ef3r_dir, "state_labels.json")
        if os.path.exists(sl):
            st = json.load(open(sl)).get("states", {})
            if "PICK" in st:
                f = st["PICK"][0]
    return f if f is not None else int(N * 0.2)


def pick_frame(ef3r_dir, obj_pts, c2w, K_all, W, H, N):
    """预抓取窗 (grasp-25 ~ grasp-5) 内投影点数最多的锚点帧。"""
    f_g = grasp_frame(ef3r_dir, N)
    anchors = [a for a in json.load(open(os.path.join(ef3r_dir, "meta.json")))
               ["camera"]["anchors"] if a < N]
    best, best_n = None, -1
    for f in anchors:
        if not (f_g - 25 <= f <= f_g - 5):
            continue
        pc = (obj_pts - c2w[f][:3, 3]) @ c2w[f][:3, :3]
        m = pc[:, 2] > 0.05
        if m.sum() < 10:
            continue
        Ks = k_for_video(K_all[f], W, H)
        u = (pc[m, 0] / pc[m, 2]) * Ks[0, 0] + Ks[0, 2]
        v = (pc[m, 1] / pc[m, 2]) * Ks[1, 1] + Ks[1, 2]
        n = int(((u >= 0) & (u < W) & (v >= 0) & (v < H)).sum())
        if n > best_n:
            best, best_n = f, n
    if best is None:
        pre = [a for a in anchors if a < f_g]
        best = pre[-1] if pre else anchors[len(anchors) // 2]
    return best


SAM3_SCRIPT = ("/path/to/project_data/mano_ws/do-as-i-do/reconstruction/"
               "scripts/run_sam3_video.py")
SAM3_CKPT = ("/path/to/project_data/mano_ws/do-as-i-do/reconstruction/"
             "weights/sam3/sam3.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ef3r-dir", required=True)
    ap.add_argument("--video", required=True, help="原始 (或翻转后) 视频")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--frame", type=int, default=None, help="重建帧 (默认自动选)")
    ap.add_argument("--text", default=None,
                    help="SAM3 文本提示 (如 'pink rubber duck') —— 优先于点提示")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import imageio.v2 as imageio
    import torch

    E = a.ef3r_dir
    od = np.load(os.path.join(E, "object_instance.npz"), allow_pickle=True)
    K_all = np.load(os.path.join(E, "camera/K_full.npy")).reshape(-1, 3, 3)
    c2w = np.load(os.path.join(E, "camera/traj_c2w.npy"))
    N = c2w.shape[0]
    workdir = os.path.join(E, "sam3d")
    os.makedirs(workdir, exist_ok=True)

    # ---- 0) 视频分辨率 + 抓取帧 + 相机系手 ----
    r0 = imageio.get_reader(a.video)
    _fr0 = next(iter(r0))
    H, W = _fr0.shape[:2]
    r0.close()
    f_g = grasp_frame(E, N)
    _d = torch.load(os.path.join(E, "inference_output.pt"),
                    map_location="cpu", weights_only=False)
    Jc = _d["joint_xyz"][:, 1].float().numpy()      # 相机系右手 (与 K 自投影一致)
    f_g = min(f_g, N - 1)

    # ---- 1) 点提示: 抓取帧指尖投影 (指尖正压在物体上 = 必在物体上) ----
    # 物体世界位置: 抓取帧指尖沿抓取方向外推 4cm (手指压在物体表面, 外推=物体
    # 体内); 再投影到段首帧做提示 —— 段首手还远, 物体完整可见 (指尖提示只会
    # 落在手皮肤上, 实测 SAM3 整手被分割; 物体静置 → 世界位置跨帧有效)
    tip = Jc[f_g][[4, 5, 8]].mean(0)
    _dv = tip - Jc[f_g][0]
    _dv = _dv / (np.linalg.norm(_dv) + 1e-9)
    p_obj_cam = tip + _dv * 0.04
    p_obj_w = p_obj_cam @ c2w[f_g][:3, :3].T + c2w[f_g][:3, 3]

    # ---- 2) 抽预抓取帧段 + SAM3 视频分割 (conda sam3) ----
    f0, f1 = max(0, f_g - 18), f_g                  # 预抓取段 (物体在桌上未被遮挡)
    Ks_0 = k_for_video(K_all[f0], W, H)
    _pc0 = (p_obj_w - c2w[f0][:3, 3]) @ c2w[f0][:3, :3]
    px = int(round(_pc0[0] / _pc0[2] * Ks_0[0, 0] + Ks_0[0, 2]))
    py = int(round(_pc0[1] / _pc0[2] * Ks_0[1, 1] + Ks_0[1, 2]))
    print(f"  点提示: 段首帧 F{f0} 物体投影 ({px},{py}) (物体世界位置自抓取帧三角化)")
    frames_dir = os.path.join(workdir, "sam3_frames")
    os.makedirs(frames_dir, exist_ok=True)
    need = list(range(f0, f1 + 1))
    have = {f for f in need if os.path.exists(
        os.path.join(frames_dir, f"frame_{f:06d}.png"))}
    if len(have) < len(need):
        r = imageio.get_reader(a.video)
        for i, img in enumerate(r):
            if i > f1:
                break
            if i in need:
                imageio.imwrite(os.path.join(frames_dir, f"frame_{i:06d}.png"), img)
        r.close()
    sam3_out = os.path.join(workdir, "sam3_seg")
    mask_dir = os.path.join(sam3_out, "masks")
    tgt = a.frame if a.frame is not None else f1 - 6     # 默认: 抓取前 ~6 帧 (最大未遮挡)
    tgt = min(max(tgt, f0), f1)
    def _sam3_runs():
        """文本提示链: 全描述 → 末词 (过度具体的描述在遮挡下会全空, 实测 ee13
        'brown glass reagent bottle' 0px, 'bottle' 22k px)。"""
        prompts = ([a.text, a.text.split()[-1]] if (a.text and " " in a.text)
                   else ([a.text] if a.text else []))
        for _t in prompts:
            print(f"  调 SAM3 视频分割 (conda sam3, 帧 {f0}~{f1}, 文本 '{_t}')...")
            cmd = (f"source {CONDA_SH} && conda activate sam3 && "
                   f"unset PYTHONPATH PYTHONHOME && export PYTHONNOUSERSITE=1 && "
                   f"cd {os.path.dirname(SAM3_SCRIPT)} && "
                   f"CUDA_VISIBLE_DEVICES={a.gpu} python run_sam3_video.py "
                   f"--video {frames_dir} --text '{_t}' --obj_id object "
                   f"--checkpoint {SAM3_CKPT} --output_dir {sam3_out}")
            subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            _mx = 0
            for _fi in range(f0, f1 + 1):
                _mp = os.path.join(mask_dir, f"frame_{_fi - f0:06d}_masks", "object.png")
                if os.path.exists(_mp):
                    _mx = max(_mx, int((imageio.imread(_mp) > 0).sum()))
            if _mx >= 300:
                return
            print(f"  提示 '{_t}' 全空 (最大 {_mx}px), 换下一级提示")
        # 文本全败 → 点提示 (指尖三角化投影)
        print(f"  调 SAM3 视频分割 (conda sam3, 点提示 {px},{py} @帧0)...")
        cmd = (f"source {CONDA_SH} && conda activate sam3 && "
               f"unset PYTHONPATH PYTHONHOME && export PYTHONNOUSERSITE=1 && "
               f"cd {os.path.dirname(SAM3_SCRIPT)} && "
               f"CUDA_VISIBLE_DEVICES={a.gpu} python run_sam3_video.py "
               f"--video {frames_dir} --points {px},{py} --point_labels 1 "
               f"--frame_idx 0 --obj_id object "
               f"--checkpoint {SAM3_CKPT} --output_dir {sam3_out}")
        subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)

    if not os.path.exists(os.path.join(mask_dir, f"frame_{tgt - f0:06d}_masks", "object.png")):
        _sam3_runs()
        if not os.path.isdir(mask_dir):
            print("  ⚠ SAM3 失败")
            sys.exit(1)
    # 目标帧 = mask 最大的可用帧 (SAM3 跟踪会中途丢失, 固定帧可能拿到空 mask,
    # 实测 ee13 F37 全空 → SAM3D 除零崩溃)
    _best_px, tgt = -1, None
    for _fi in range(f0, f1 + 1):
        _mp = os.path.join(mask_dir, f"frame_{_fi - f0:06d}_masks", "object.png")
        if os.path.exists(_mp):
            _n = int((imageio.imread(_mp) > 0).sum())
            if _n > _best_px:
                _best_px, tgt = _n, _fi
    if tgt is None or _best_px < 300:
        print(f"  ⚠ SAM3 mask 全空/过小 (最大 {_best_px}px) — 物体不可分割, 保留凸包版")
        sys.exit(1)
    print(f"  重建帧选定 F{tgt} (mask {_best_px}px, 取 mask 最大的可用帧)")
    m_png = os.path.join(mask_dir, f"frame_{tgt - f0:06d}_masks", "object.png")

    # ---- 3) SAM3D 重建 (conda sam3d, 复用同一 env 链) ----
    rgb_png = os.path.join(frames_dir, f"frame_{tgt:06d}.png")
    mdir = os.path.join(workdir, f"frame_{tgt:06d}_masks")
    os.makedirs(mdir, exist_ok=True)
    mask_png = os.path.join(mdir, "object_0.png")
    _m = imageio.imread(m_png) > 0
    _fr = imageio.imread(rgb_png)
    H, W = _fr.shape[:2]
    # 深度带过滤: 文本/点提示过匹配时 mask 罩住背景 (实测 ee11 'flask'
    # 298k px 含台面)。物体在前景 = mask 内**最近深度带** (5% 分位 +15cm)。
    import torch as _th
    _efd = _th.load(os.path.join(E, 'inference_output.pt'),
                    map_location='cpu', weights_only=False)
    _ai = int(np.argmin(np.abs(_efd['depth_frame_indices'].numpy() - tgt)))
    _dep = _efd['depth'].numpy()[_ai]
    _Kd = _efd['K'].numpy()[_ai]
    _ys, _xs = np.where(_m)
    _sc_y, _sc_x = H / _dep.shape[0], W / _dep.shape[1]
    _zd = _dep[np.clip((_ys / _sc_y).astype(int), 0, _dep.shape[0] - 1),
               np.clip((_xs / _sc_x).astype(int), 0, _dep.shape[1] - 1)]
    _zd = _zd[np.isfinite(_zd) & (_zd > 0.1)]
    if len(_zd) > 100:
        _d0 = np.percentile(_zd, 5)
        _zs_full = np.full((H, W), np.nan)
        _z_resize = np.kron(_dep, np.ones((max(1, H // _dep.shape[0]),
                                           max(1, W // _dep.shape[1]))))[:H, :W]
        _zs_full[:_z_resize.shape[0], :_z_resize.shape[1]] = _z_resize
        _keep = _m & np.isfinite(_zs_full) & \
            (_zs_full > _d0 - 0.02) & (_zs_full < _d0 + 0.15)
        if _keep.sum() > 800:
            _m = _keep
    imageio.imwrite(mask_png, (_m > 0).astype(np.uint8) * 255)
    print(f"  重建帧 F{tgt} (mask {int((_m > 0).sum())} px, "
          f"深度带 {(_d0 if len(_zd) > 100 else 0):.2f}m)")

    print("  调 SAM3D (conda sam3d)...")
    cmd = (f"source {CONDA_SH} && conda activate sam3d && "
           f"unset PYTHONPATH PYTHONHOME && export PYTHONNOUSERSITE=1 && "
           f"cd {SAM3D_DIR} && export PYTHONPATH=$PWD && "
           f"CUDA_VISIBLE_DEVICES={a.gpu} python generate_mesh_sam3d.py "
           f"--image_path {rgb_png} --masks_dir {mdir} --mask_files object_0.png "
           f"--config {SAM3D_CFG}")
    log = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    mesh_p = os.path.join(mdir, "object_0", "object_0.obj")
    if log.returncode != 0 or not os.path.exists(mesh_p):
        print("  ⚠ SAM3D 失败 (保留凸包版), 日志尾:")
        print("   " + "\n   ".join((log.stdout + log.stderr).splitlines()[-6:]))
        sys.exit(1)

    # ---- 4) 对齐: mesh (自身系) → 相机系深度点 (物体世界位置球邻域) → 世界系 ----
    # 物体世界位置: 抓取帧指尖 (指尖=物体表面) — 跨帧经 c2w 传播 (相机世界系
    # 自洽; 抓取前物体静置)
    tip_w = tip @ c2w[f_g][:3, :3].T + c2w[f_g][:3, 3]
    ai = int(np.argmin(np.abs(_d["depth_frame_indices"].numpy() - tgt)))
    fdep = int(_d["depth_frame_indices"].numpy()[ai])
    z = _d["depth"].numpy()[ai]
    vs, us = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    m = np.isfinite(z) & (z > 0.05) & (z < 2.5)
    zz = z[m]
    pts_cam = np.stack([(us[m] - _d["K"].numpy()[ai][0, 2]) / _d["K"].numpy()[ai][0, 0] * zz,
                        (vs[m] - _d["K"].numpy()[ai][1, 2]) / _d["K"].numpy()[ai][1, 1] * zz,
                        zz], 1)
    tip_dep = tip @ np.linalg.inv(c2w[fdep])[:3, :3].T + \
        np.linalg.inv(c2w[fdep])[:3, 3]          # 抓取帧指尖 → 深度锚帧相机系
    # 物体静置: 深度锚帧下的物体中心 ≈ 指尖位置沿原抓取方向再前 4cm; 此处直接
    # 用指尖球 (r=0.07) — 抓取帧指尖贴物体, 静置假设下锚帧同一世界位置
    d = np.linalg.norm(pts_cam - tip_dep, axis=1)
    obj_pts = pts_cam[d < 0.07]
    if len(obj_pts) < 30:                        # 遮挡/深度空洞: 放宽
        obj_pts = pts_cam[d < 0.10]
    print(f"  对齐参考点: 深度锚帧 F{fdep} 球内 {len(obj_pts)} 点")

    import trimesh
    from itertools import product
    tm = trimesh.load(mesh_p, force="mesh", process=False)
    V, F = np.asarray(tm.vertices, float), np.asarray(tm.faces, int)

    if len(obj_pts) < 30:
        # ---- 深度空洞几何回退 (玻璃/反光): 无物体深度点可用。尺度由目标帧
        # mask 的像素范围 × 指尖深度 / fx 估计; 位置 = 三角化物体中心
        # (抓取帧指尖外推); 姿态取直立 (仿真端 geo 对齐会再吸收残差)。
        _mp = os.path.join(mask_dir, f"frame_{tgt - f0:06d}_masks", "object.png")
        _m = imageio.imread(_mp) > 0
        ys, xs = np.where(_m)
        Kt = k_for_video(K_all[tgt], W, H)
        tip_t = tip @ np.linalg.inv(c2w[tgt])[:3, :3].T + \
            np.linalg.inv(c2w[tgt])[:3, 3]       # 指尖 → 目标帧相机系
        Z = max(float(tip_t[2]), 0.1)
        w_m = (xs.max() - xs.min()) * Z / Kt[0, 0]
        h_m = (ys.max() - ys.min()) * Z / Kt[1, 1]
        span = np.ptp(V, axis=0)
        s = float(np.median([w_m / max(span.max(), 1e-6),
                             h_m / max(np.median(span), 1e-6)]))
        p_obj_cam = tip_t + np.array([0, 0, -0.02])   # 物体中心≈指尖下方 2cm
        Vw = ((V - V.mean(0)) * s) + p_obj_cam
        print(f"  深度空洞几何回退: mask {int(_m.sum())}px → 尺度 {s:.3f} "
              f"(mask {w_m*1000:.0f}×{h_m*1000:.0f}mm @Z={Z:.2f}m)")
        ext_sam = np.ptp(Vw, axis=0) * 1000
        ext_old = np.ptp(od["verts"], axis=0) * 1000
        print(f"  bbox mm SAM3D {np.round(ext_sam,1).tolist()} vs 凸包 {np.round(ext_old,1).tolist()}")
        out = a.out or os.path.join(E, "object_instance_sam3d.npz")
        _c = Vw.mean(0); _qc = Vw - _c
        _cov = _qc.T @ _qc / max(len(_qc), 1)
        _ev, _evec = np.linalg.eigh(_cov)
        _axes = _evec[:, np.argsort(_ev)[::-1]].T
        upd = dict(od)
        upd.update(verts=Vw.astype(np.float32), faces=F.astype(np.int64),
                   pca_center=_c, pca_axes=_axes,
                   pca_ext=np.ptp(_qc @ _axes.T, axis=0).astype(float),
                   source="sam3d", sam3d_scale=s, sam3d_frame=tgt)
        np.savez(out, **upd)
        print(f"  ✓ {out} (几何回退对齐; 凸包版保留)")
        return

    def _pca(X):
        Xc = X - X.mean(0)
        C = Xc.T @ Xc / len(Xc)
        ev, evec = np.linalg.eigh(C)
        A = evec[:, np.argsort(ev)[::-1]].T
        return X.mean(0), A, np.ptp(Xc @ A.T, axis=0)

    cp, Ap, sp = _pca(obj_pts)
    cq, Aq, sq = _pca(V)
    s = float(np.median(sp / np.maximum(sq, 1e-9)))
    _pct = np.linspace(5, 95, 19)
    _ref = [np.percentile((obj_pts - cp) @ Ap[i], _pct) for i in range(3)]
    best, best_score, Vw = None, -np.inf, None
    for sx, sy, sz in product([1, -1], repeat=3):
        A = np.diag([sx, sy, sz]) @ Ap @ Aq.T
        if np.linalg.det(A) < 0:
            continue
        Vw_t = ((V - cq) * s) @ A.T + cp
        sc = sum(float(np.corrcoef(
            _ref[i], np.percentile((Vw_t - cp) @ Ap[i], _pct))[0, 1])
            for i in range(3))
        if sc > best_score:
            best, best_score, Vw = A, sc, Vw_t
    ext_sam = np.ptp(Vw, axis=0) * 1000
    ext_old = np.ptp(od["verts"], axis=0) * 1000
    print(f"  对齐: 尺度 {s:.3f} (score {best_score:.2f}) | bbox mm SAM3D "
          f"{np.round(ext_sam, 1).tolist()} vs 凸包 {np.round(ext_old, 1).tolist()}")

    # ---- 5) 同契约落盘 (深度锚帧相机系 → 相机世界系, 与支撑面共系) ----
    out = a.out or os.path.join(E, "object_instance_sam3d.npz")
    Vw = Vw @ c2w[fdep][:3, :3].T + c2w[fdep][:3, 3]
    _c = Vw.mean(0)
    _qc = Vw - _c
    _cov = _qc.T @ _qc / max(len(_qc), 1)
    evals, evecs = np.linalg.eigh(_cov)
    _axes = evecs[:, np.argsort(evals)[::-1]].T
    _ext = np.ptp(_qc @ _axes.T, axis=0)
    upd = dict(od)
    upd.update(verts=Vw.astype(np.float32), faces=F.astype(np.int64),
               pca_center=_c, pca_axes=_axes, pca_ext=_ext.astype(float),
               source="sam3d", sam3d_scale=s, sam3d_frame=tgt)
    np.savez(out, **upd)
    print(f"  ✓ {out} (凸包版保留为 object_instance.npz)")


if __name__ == "__main__":
    main()
