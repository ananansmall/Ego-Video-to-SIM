#!/usr/bin/env python3
"""
左手 3D trajectory 对比图 (单图版)

对齐链与 View/render_quick.py load_data 完全一致 (数据驱动, 无固定旋转近似):
  MANO FK (HaWoR 输入相机系)
    -> p_hw = R_c2w[f] @ p_cam + t_c2w[f]        逐帧真实相机位姿
    -> p_glb = s * R_hand_to_glb @ p_hw + t_hand_to_glb   transform_params 相似变换

图中内容:
  - 原始左手 (hand0) MANO FK 轨迹: 手腕 j0 / 拇指尖 j20 / 食指尖 j16 / pinch 中点
  - F50 抓取帧高亮
  - 管线各 stage 轨迹 (细线): Stage0 base+指尖 / Stage1 抓取点 / Stage2 重建 / Stage3
  - 物体初始位置

输出: output/gripper_right/photo/left_hand_3d_trajectory.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R_scipy

_TJ_DIR = Path(__file__).resolve().parent.parent          # trajectory_optimization/
_COMB_DIR = _TJ_DIR.parent                                # example/combination/
sys.path.insert(0, str(_COMB_DIR.parent / "position_retargeting"))

from mano_layer import MANOLayer  # noqa: E402

HAWOR_DIR = Path("/home/an/data/hawor/7")
OUTPUT_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _TJ_DIR / "output" / "gripper_right"
PHOTO_DIR = OUTPUT_DIR / "photo"
_tp_candidates = [OUTPUT_DIR / "stage0" / "alignment" / "transform_params.npz",
                  OUTPUT_DIR / "alignment" / "transform_params.npz"]
TP_PATH = next(p for p in _tp_candidates if p.exists())

# 与管线实际加载一致 (stage*.log: HaWoR 数据已加载 hawor_results_0_113_depth_aligned.npz);
# depth_aligned 才是对齐到 RAS 深度的版本, base 版 pred_trans z 偏高 ~11cm
_files = sorted((HAWOR_DIR / "reconstruction").glob("hawor_results_*.npz"))
NPZ_PATH = next(f for f in _files if "_depth_aligned" in f.name and "_final" not in f.name and "_noflip" not in f.name and "_reverse" not in f.name)
HAND_IDX = 0          # 0=左手 (真实操作手); 1=右手 (HaWoR 幻觉, pred_trans z≈0)
F50_IDX = 50
FINGER_FORWARD = np.array([0.03689, 0.0, 0.0])  # 夹爪 base → 手指中点


def load_transform_params():
    tp = np.load(str(TP_PATH))
    return (float(tp["scale_ratio"]),
            np.asarray(tp["R_hand_to_glb"], dtype=np.float64),
            np.asarray(tp["t_hand_to_glb"], dtype=np.float64).reshape(3))


def mano_fk_glb(d, hidx, s, R_h2g, t_h2g):
    """MANO FK + 数据驱动坐标链 → GLB 世界系关节 {j: (N,3)}"""
    side = {0: "left", 1: "right"}[hidx]
    n = len(d["pred_trans"][hidx])
    betas = None
    for fi in range(n):
        if not np.any(np.isnan(d["pred_betas"][hidx][fi])):
            betas = d["pred_betas"][hidx][fi].astype(np.float32)
            break
    ml = MANOLayer(side, betas)
    R_c2w, t_c2w = d["R_c2w"], d["t_c2w"]

    joints = np.full((n, 21, 3), np.nan)
    import torch
    for fi in range(n):
        if hidx < d["pred_valid"].shape[0] and not d["pred_valid"][hidx][fi]:
            continue
        p = torch.from_numpy(np.concatenate([
            d["pred_rot"][hidx][fi], d["pred_hand_pose"][hidx][fi]
        ]).astype(np.float32)).unsqueeze(0)
        t = torch.from_numpy(d["pred_trans"][hidx][fi].astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            _, j = ml(p, t)
        jc = j[0].numpy().astype(np.float64)          # (21,3) 相机系
        j_hw = np.einsum("ij,kj->ki", R_c2w[fi], jc) + t_c2w[fi]
        joints[fi] = s * (R_h2g @ j_hw.T).T + t_h2g
    return joints


def main():
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    d = dict(np.load(str(NPZ_PATH), allow_pickle=True))
    s, R_h2g, t_h2g = load_transform_params()
    print(f"npz={NPZ_PATH.name}, N={len(d['pred_trans'][0])}, s={s:.4f}")

    joints = mano_fk_glb(d, HAND_IDX, s, R_h2g, t_h2g)
    wrist, thumb, index = joints[:, 0], joints[:, 20], joints[:, 16]
    pinch = (thumb + index) / 2.0
    # 幻觉右手 (hand1): 管线 --side right 实际跟踪的轨迹
    joints_r = mano_fk_glb(d, 1, s, R_h2g, t_h2g)
    wrist_r = joints_r[:, 0]

    # --- 管线 stage 数据 (缺失的 stage 自动跳过) ---
    stage_plots = []
    s0_path = OUTPUT_DIR / "stage0" / f"grasp_gripper_only_{OUTPUT_DIR.name.split('_')[-1]}_qpos.npy"
    if s0_path.exists():
        s0_qpos = np.load(s0_path)
        s0_base = s0_qpos[:, :3]
        s0_R = R_scipy.from_euler("ZYX", s0_qpos[:, 3:6]).as_matrix()   # (rz,ry,rx) → Rz·Ry·Rx
        s0_tip = s0_base + np.einsum("fij,j->fi", s0_R, FINGER_FORWARD)
        stage_plots += [("Stage0 gripper base", s0_base, "tab:blue", "--"),
                        ("Stage0 fingertip", s0_tip, "tab:cyan", ":")]

    s1 = np.load(OUTPUT_DIR / "stage1" / "stage1" / "best_grasp.npz", allow_pickle=True)
    s1_pos = np.asarray(s1["pos"]).reshape(3)

    s2_candidates = sorted(OUTPUT_DIR.glob("stage2/**/stage2_recon.npz"))
    if s2_candidates:
        s2 = np.load(s2_candidates[0], allow_pickle=True)
        stage_plots.append(("Stage2 recon (F38-62)", np.asarray(s2["pos"]), "tab:red", "-"))

    s3 = np.load(OUTPUT_DIR / "stage3" / "stage3_result.npz", allow_pickle=True) \
        if (OUTPUT_DIR / "stage3" / "stage3_result.npz").exists() \
        else np.load(OUTPUT_DIR / "stage3" / "stage3" / "stage3_result.npz", allow_pickle=True)
    s3r = s3["result"].item()
    s3_opt = np.asarray(s3r["opt_pos"])
    obj_start = np.asarray(s3r["obj_pos_traj"])[0]

    # --- sanity: 左右手 F50 pinch 到物体中心的距离 ---
    for h, jh in ((0, joints), (1, joints_r)):
        ph = (jh[:, 20] + jh[:, 16]) / 2.0
        tr = d["pred_trans"][h]
        print(f"hand{h} ({'left' if h == 0 else 'right'}): pred_trans z="
              f"[{tr[:, 2].min():.3f},{tr[:, 2].max():.3f}]m, "
              f"F50 pinch→obj_center = {np.linalg.norm(ph[F50_IDX] - obj_start):.4f}m")

    # --- 绘图: 一张 3D 图 ---
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(wrist[:, 0], wrist[:, 1], wrist[:, 2], color="0.25", lw=2.5, label="Left hand wrist (j0)")
    ax.plot(thumb[:, 0], thumb[:, 1], thumb[:, 2], color="darkorange", lw=1.4, alpha=0.85, label="Left thumb tip (j20)")
    ax.plot(index[:, 0], index[:, 1], index[:, 2], color="purple", lw=1.4, alpha=0.85, label="Left index tip (j16)")
    ax.plot(pinch[:, 0], pinch[:, 1], pinch[:, 2], color="red", lw=2.5, label="Left pinch midpoint")
    ax.plot(wrist_r[:, 0], wrist_r[:, 1], wrist_r[:, 2], color="0.55", lw=1.2, ls=":",
            label="Right hand wrist (hallucinated — what the pipeline tracked)")

    # F50 高亮 + 首末帧标记
    ax.scatter(*pinch[F50_IDX], c="red", s=180, marker="*", zorder=6, label="F50 (grasp frame)")
    ax.text(*pinch[F50_IDX] + [0, 0, 0.03], "F50", color="red", fontsize=11, fontweight="bold")
    for fi, lbl in ((0, "F0"), (len(pinch) - 1, f"F{len(pinch)-1}")):
        ax.scatter(*wrist[fi], c="0.25", s=45, marker="o")
        ax.text(*wrist[fi] + [0, 0, 0.02], lbl, color="0.25", fontsize=9)

    # 管线 stage (细线, 弱化)
    for lbl, pts, color, style in stage_plots:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=1.0, ls=style,
                alpha=0.75, label=lbl)
    ax.scatter(*s1_pos, c="tab:green", s=140, marker="*", zorder=6, label="Stage1 CEM grasp")
    ax.plot(s3_opt[:, 0], s3_opt[:, 1], s3_opt[:, 2], color="m", lw=1.4, ls="--", alpha=0.7,
            label="Stage3 opt trajectory")
    ax.scatter(*obj_start, c="tab:brown", s=110, marker="s", zorder=6, label="Object start")

    ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)"), ax.set_zlabel("Z (m)")
    ax.set_title(f"Left-hand 3D trajectory (GLB) vs pipeline stages — {OUTPUT_DIR.name}", y=0.98)
    all_pts = np.vstack([pinch] + [p for _, p, _, _ in stage_plots] + [s3_opt])
    rng = all_pts.max(axis=0) - all_pts.min(axis=0)
    ctr = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    lim = rng.max() / 2 * 1.05
    ax.set_xlim(ctr[0] - lim, ctr[0] + lim)
    ax.set_ylim(ctr[1] - lim, ctr[1] + lim)
    ax.set_zlim(ctr[2] - lim, ctr[2] + lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass  # 旧版 matplotlib 无 set_box_aspect
    ax.view_init(elev=22, azim=-60)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.9))

    out = PHOTO_DIR / "left_hand_3d_trajectory.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
