#!/usr/bin/env python3
"""
左手 3D 轨迹对比图 — 中点口径 (2026-09-13 重构)

用户约定 (2026-09-13): 所有轨迹只渲染中点, 不画手腕/单指 —
  MANO   → 拇指尖 j20 + 食指尖 j16 的中点
  夹爪   → base + R*[0.03689,0,0] (finger_mid)
有夹爪数据选夹爪, 没夹爪用 MANO 拇食中点.

三条主轨迹 (全部 7a 数据, 同一 transform_params s=2.5458, GLB=ras/7_vggt_omega):
  PRE  未处理 MANO 拇食中点: preprocess_traj.npz joints_raw (平滑前, GLB 系) [紫]
  SG   修复后 (SG 平滑):     同 npz joints (平滑后) + stage0/1/2/3 夹爪轨迹 [蓝系]
  MINT 修复后 (MINT UKF+RTS): mint_hawor7a_left/preprocess_traj.npz joints [绿系]
  场景: GLB 物体 bbox + obj_center (glb_4)

缺失产物自动跳过; MINT stage2 gate FAIL (PRE_F50_READY, tracking 38.7mm 超限)
未进 stage3 — 如实标注在图例.

输出: <OUTPUT_DIR>/photo/left_hand_3d_trajectory.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R_scipy
import trimesh

_TJ_DIR = Path(__file__).resolve().parent.parent          # trajectory_optimization/
GLB_PATH = Path("/home/an/data/ras/7_vggt_omega/final_scene.glb")
OUTPUT_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _TJ_DIR / "output" / "hawor7a_left"
MINT_DIR = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else _TJ_DIR / "output" / "mint_hawor7a_left"
PHOTO_DIR = OUTPUT_DIR / "photo"
F50_IDX = 50
FINGER_FORWARD = np.array([0.03689, 0.0, 0.0])  # 夹爪 base → 手指中点


def finger_mid(pos, R):
    """夹爪 base 轨迹 → finger_mid 轨迹"""
    return np.asarray(pos) + np.einsum("fij,j->fi", np.asarray(R), FINGER_FORWARD)


def load_stage0_finger_mid(run_dir, side="left"):
    """stage0 qpos (base xyz + ZYX euler) → finger_mid; 缺失返回 None"""
    p = run_dir / "stage0" / f"grasp_gripper_only_{side}_qpos.npy"
    if not p.exists():
        return None
    q = np.load(p)
    base, Rm = q[:, :3], R_scipy.from_euler("ZYX", q[:, 3:6]).as_matrix()
    return finger_mid(base, Rm)


def load_stage2_finger_mid(run_dir):
    """stage2_recon.npz pos+R → finger_mid (F38-62); 缺失返回 None"""
    cands = sorted(run_dir.glob("stage2/**/stage2_recon.npz"))
    if not cands:
        return None
    d = np.load(cands[0], allow_pickle=True)
    if "R" not in d:
        return np.asarray(d["pos"])
    return finger_mid(d["pos"], d["R"])


def load_stage3(run_dir):
    """stage3_result.npz → (opt 中点轨迹或 None, obj_start 或 None)"""
    for c in (run_dir / "stage3" / "stage3_result.npz",
              run_dir / "stage3" / "stage3" / "stage3_result.npz"):
        if c.exists():
            r = np.load(c, allow_pickle=True)["result"].item()
            opt = np.asarray(r["opt_pos"])
            obj0 = np.asarray(r["obj_pos_traj"])[0]
            if "opt_R" in r:
                opt = finger_mid(opt, np.asarray(r["opt_R"]))
            return opt, obj0
    return None, None


def load_stage1_point(run_dir):
    """stage1 best_grasp.npz → 抓取点 (grasp_pose_method=svd_palm 时 pos 已是手掌锚点)"""
    for c in (run_dir / "stage1" / "best_grasp.npz",
              run_dir / "stage1" / "stage1" / "best_grasp.npz"):
        if c.exists():
            return np.asarray(np.load(c, allow_pickle=True)["pos"]).reshape(3)
    return None


def mano_mids(npz_path):
    """preprocess_traj.npz → (raw 中点, 平滑后中点) 拇食中点 (GLB 系)"""
    d = np.load(npz_path, allow_pickle=True)
    jr, js = np.asarray(d["joints_raw"]), np.asarray(d["joints"])
    return (0.5 * (jr[:, 16] + jr[:, 20]), 0.5 * (js[:, 16] + js[:, 20]),
            np.asarray(d["obj_center"]))


def load_glb_boxes():
    """GLB 物体 bbox (原始顶点 = Z-up 世界坐标), 跳过 grid/floor/wall"""
    if not GLB_PATH.exists():
        return []
    scene = trimesh.load(str(GLB_PATH), force="scene")
    out = []
    for node_name in scene.graph.nodes_geometry:
        if any(k in node_name.lower() for k in ("grid", "floor", "wall")):
            continue
        _, gname = scene.graph.get(node_name)
        if gname is None or gname not in scene.geometry:
            continue
        v = np.asarray(scene.geometry[gname].vertices, dtype=float)
        out.append((node_name, v.mean(0), np.array([v.min(0), v.max(0)])))
    return out


def main():
    PHOTO_DIR.mkdir(parents=True, exist_ok=True)

    # ── 三条 MANO 中点主轨迹 ──
    pre_mid, sg_mid, obj_center = mano_mids(OUTPUT_DIR / "preprocess_traj.npz")
    mint_mid = None
    if (MINT_DIR / "preprocess_traj.npz").exists():
        mint_mid = mano_mids(MINT_DIR / "preprocess_traj.npz")[1]
    print(f"PRE raw mid: {len(pre_mid)} 帧 | SG smooth mid: {len(sg_mid)} 帧 | "
          f"MINT mid: {None if mint_mid is None else len(mint_mid)} 帧")

    # ── 夹爪轨迹 (有夹爪数据选夹爪) ──
    sg_s0 = load_stage0_finger_mid(OUTPUT_DIR)
    sg_s2 = load_stage2_finger_mid(OUTPUT_DIR)
    sg_s3, obj_start = load_stage3(OUTPUT_DIR)
    sg_s1 = load_stage1_point(OUTPUT_DIR)
    mt_s0 = load_stage0_finger_mid(MINT_DIR)
    mt_s1 = load_stage1_point(MINT_DIR)
    mt_s2 = load_stage2_finger_mid(MINT_DIR)
    print(f"SG: stage0={sg_s0 is not None} stage1={sg_s1 is not None} "
          f"stage2={sg_s2 is not None} stage3={sg_s3 is not None}")
    print(f"MINT: stage0={mt_s0 is not None} stage1={mt_s1 is not None} stage2={mt_s2 is not None}")

    # ── 绘图 ──
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")

    # GLB 物体 bbox
    for node_name, ct, bd in load_glb_boxes():
        p0, p1 = bd
        corners = np.array([[p0[0],p0[1],p0[2]],[p1[0],p0[1],p0[2]],[p1[0],p1[1],p0[2]],[p0[0],p1[1],p0[2]],
                            [p0[0],p0[1],p1[2]],[p1[0],p0[1],p1[2]],[p1[0],p1[1],p1[2]],[p0[0],p1[1],p1[2]]])
        for e in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
            ax.plot3D(*corners[[e[0], e[1]]].T, color="0.75", alpha=0.5, lw=0.7)
        ax.scatter(*ct, c="0.6", s=20)
        ax.text(ct[0], ct[1], ct[2] + 0.04, node_name, color="0.45", fontsize=7)

    # 三条 MANO 拇食中点主轨迹
    ax.plot(pre_mid[:, 0], pre_mid[:, 1], pre_mid[:, 2], color="purple", lw=2.4,
            label="PRE: 未处理 MANO 拇食中点 (raw)")
    ax.plot(sg_mid[:, 0], sg_mid[:, 1], sg_mid[:, 2], color="tab:blue", lw=2.2,
            label="SG 修复后 MANO 拇食中点")
    if mint_mid is not None:
        ax.plot(mint_mid[:, 0], mint_mid[:, 1], mint_mid[:, 2], color="tab:green", lw=2.2,
                label="MINT 修复后 MANO 拇食中点")

    # 夹爪轨迹 (虚线)
    if sg_s0 is not None:
        ax.plot(sg_s0[:, 0], sg_s0[:, 1], sg_s0[:, 2], "--", color="tab:blue", lw=1.4,
                alpha=0.85, label="SG stage0 夹爪 finger_mid")
    if sg_s2 is not None:
        ax.plot(sg_s2[:, 0], sg_s2[:, 1], sg_s2[:, 2], "-", color="tab:red", lw=1.6,
                alpha=0.9, label="SG stage2 重建 finger_mid (F38-62)")
    if sg_s3 is not None:
        ax.plot(sg_s3[:, 0], sg_s3[:, 1], sg_s3[:, 2], "--", color="m", lw=1.6,
                alpha=0.85, label="SG stage3 优化 finger_mid")
    if mt_s0 is not None:
        ax.plot(mt_s0[:, 0], mt_s0[:, 1], mt_s0[:, 2], "--", color="tab:green", lw=1.4,
                alpha=0.85, label="MINT stage0 夹爪 finger_mid")
    if mt_s2 is not None:
        ax.plot(mt_s2[:, 0], mt_s2[:, 1], mt_s2[:, 2], "-", color="lime", lw=1.6,
                alpha=0.9, label="MINT stage2 重建 finger_mid (F38-62)")

    # F50 / stage1 抓取点 / 物体
    ax.scatter(*pre_mid[F50_IDX], c="purple", s=150, marker="*", zorder=6, label="F50 (raw)")
    if sg_s1 is not None:
        ax.scatter(*sg_s1, c="tab:blue", s=140, marker="*", zorder=6, label="SG stage1 抓取点")
    if mt_s1 is not None:
        ax.scatter(*mt_s1, c="tab:green", s=140, marker="*", zorder=6, label="MINT stage1 抓取点")
    ax.scatter(*obj_center, c="m", s=150, marker="p", edgecolors="k", zorder=10,
               label="obj_center (glb_4)")

    ax.set_xlabel("X (m)"), ax.set_ylabel("Y (m)"), ax.set_zlabel("Z (m)")
    mint_note = "" if mt_s2 is not None else " | ⚠ MINT stage2 gate FAIL (tracking 38.7mm) 未进 stage3"
    ax.set_title(f"中点口径轨迹对比: 未处理 MANO vs SG vs MINT — {OUTPUT_DIR.name}{mint_note}",
                 fontsize=11, y=0.98)
    all_pts = [pre_mid, sg_mid] + [p for p in (mint_mid, sg_s0, sg_s2, sg_s3, mt_s0, mt_s2) if p is not None]
    all_pts += [p for p in (sg_s1, mt_s1, obj_center) if p is not None]
    all_pts = np.vstack(all_pts)
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
    ax.view_init(elev=20, azim=-58)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))

    out = PHOTO_DIR / "left_hand_3d_trajectory.png"
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
