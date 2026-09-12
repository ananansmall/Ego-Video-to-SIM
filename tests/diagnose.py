#!/usr/bin/env python3
"""
diagnose.py — Stage 1 MANO/抓取坐标链诊断

合并自 diagnose_full.py (--mode full) 与 diagnose_mano.py (--mode mano)。
两者共享 MANO SLAM→SAPIEN 变换常量与 HaWoR 加载逻辑, 仅诊断项不同:
  --mode full  追踪 MANO SLAM→SAPIEN→gripper base 的每一步, 与 Stage3 比对
  --mode mano  对比 Original MANO 手掌(palm vertex 745) vs Stage 3 mano_pos

路径自适应: 脚本放在 tests/ 下, OUTPUT 指向父目录 output/。
数据根默认 /home/an/data, 可用 --data-root 覆盖。

用法:
  python diagnose.py --mode full   --side left
  python diagnose.py --mode mano   --side left
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent        # tests/
ROOT_DIR = SCRIPT_DIR.parent                         # trajectory_optimization/
OUTPUT_DIR = ROOT_DIR / "output"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting")

# ── 共享变换常量 (与 002_render_scene.py / grasp_hawor.py 一致) ──
RX = np.diag([1.0, -1.0, -1.0])
R_AXIS = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def load_params(side, data_root):
    transform = OUTPUT_DIR / f"gripper_only_{side}" / "alignment" / "transform_params.npz"
    out_dir = OUTPUT_DIR / f"gripper_only_{side}"
    p = np.load(str(transform))
    SCALE = float(p["scale_ratio"])
    R_H2G = p["R_hand_to_glb"]
    T_H2G = p["t_hand_to_glb"].ravel()
    R_hand = R_H2G @ RX
    return SCALE, R_H2G, T_H2G, R_hand, out_dir


def mano_to_sapien(pts_slam, SCALE, R_hand, T_H2G):
    pts_glb = SCALE * (R_hand @ pts_slam.T).T + T_H2G
    return (R_AXIS @ pts_glb.T).T


def load_hawor_joints(side, data_root):
    """加载 HaWoR MANO, 用 mano_layer 做 FK, 返回 (jn, palm_745, betas)"""
    from mano_layer import MANOLayer
    hawor_dir = Path(data_root) / "hawor" / "7"
    d = np.load(str(sorted((hawor_dir / "reconstruction").glob("hawor_results_*.npz"))[0]), allow_pickle=True)
    v = d["pred_valid"][0] & ~np.isnan(d["pred_trans"][0]).any(1) & ~np.isnan(d["pred_rot"][0]).any(1)
    trans_v = d["pred_trans"][0][v]; rot_v = d["pred_rot"][0][v]; hp_v = d["pred_hand_pose"][0][v]
    betas = d["pred_betas"][0][v].mean(0) if v.any() else np.zeros(10)
    ml = MANOLayer("left", betas=betas)
    i = 0
    r = torch.from_numpy(rot_v[i:i + 1].astype(np.float32))
    hp = torch.from_numpy(hp_v[i:i + 1].astype(np.float32))
    t = torch.from_numpy(trans_v[i:i + 1].astype(np.float32))
    vo = ml(torch.cat([r, hp], dim=1), t)
    vertices, joints = (vo[0], vo[1]) if isinstance(vo, (tuple, list)) else (vo, None)
    vn = vertices.detach().cpu().numpy()
    jn = joints.detach().cpu().numpy() if joints is not None else None
    return jn, vn[0, 745, :], betas


def diagnose_full(side, data_root):
    print(f"[diagnose:full] side={side}")
    SCALE, R_H2G, T_H2G, R_hand, out_dir = load_params(side, data_root)
    jn, _, _ = load_hawor_joints(side, data_root)

    print(f"SCALE={SCALE:.4f}")
    j0_s = mano_to_sapien(jn[0, 0:1], SCALE, R_hand, T_H2G)[0]
    j4_s = mano_to_sapien(jn[0, 4:5], SCALE, R_hand, T_H2G)[0]
    j8_s = mano_to_sapien(jn[0, 8:9], SCALE, R_hand, T_H2G)[0]
    print(f"\nSTEP 1: MANO FK in SLAM space\n  J0={jn[0,0].round(6)}  J4={jn[0,4].round(6)}  J8={jn[0,8].round(6)}")
    print(f"STEP 2: _mano_to_sapien\n  J0={j0_s.round(6)}\n  J4={j4_s.round(6)}\n  J8={j8_s.round(6)}")

    from data_loader import compute_analytical_gripper_pose
    root_pos, root_R, j1, j2 = compute_analytical_gripper_pose(j0_s, j4_s, j8_s, prefix="left")
    print(f"STEP 3: compute_analytical_gripper_pose\n  root_pos={root_pos.round(6)}\n  j1={j1:.6f}, j2={j2:.6f}")

    s3 = np.load(str(out_dir / "stage3" / "stage3_result.npz"), allow_pickle=True)
    s3_mano = np.array(s3["result"].item()["mano_pos"])
    print(f"STEP 4: COMPARISON\n  Computed base={root_pos.round(6)}\n  Stage3 mano_pos[0]={s3_mano[0].round(6)}")
    print(f"  Diff={np.linalg.norm(s3_mano[0]-root_pos):.6f} m  (J4 dist={np.linalg.norm(root_pos-j4_s):.6f} m)")

    from trajectory_loader import compute_analytical_gripper_pose as compute_old
    root_old, *_ = compute_old(j0_s, j4_s, j8_s)
    print(f"\nCOMPARISON data_loader vs trajectory_loader:\n  new={root_pos.round(6)}\n  old={root_old.round(6)}\n  dist={np.linalg.norm(root_old-root_pos):.6f} m")
    print("[diagnose:full] done.")


def diagnose_mano(side, data_root):
    print(f"[diagnose:mano] side={side}")
    SCALE, R_H2G, T_H2G, R_hand, out_dir = load_params(side, data_root)
    jn, palm_745, _ = load_hawor_joints(side, data_root)

    print(f"Frame0 palm v745 (SLAM)={palm_745.round(6)}")
    palm_glb = SCALE * (R_hand @ palm_745) + T_H2G
    palm_sapien = (R_AXIS @ palm_glb).ravel()
    print(f"  palm GLB={palm_glb.round(6)}\n  palm SAPIEN={palm_sapien.round(6)}")

    s3 = np.load(str(out_dir / "stage3" / "stage3_result.npz"), allow_pickle=True)
    s3_mano = np.array(s3["result"].item()["mano_pos"])
    s3_glb = (R_AXIS.T @ s3_mano.T).T
    print(f"\nStage3 mano GLB[0]={s3_glb[0].round(6)}")
    print(f"COMPARISON Frame0:\n  palm GLB={palm_glb.round(6)}\n  Stage3 GLB={s3_glb[0].round(6)}")
    print(f"  Diff={np.linalg.norm(s3_glb[0]-palm_glb):.4f} m")
    if jn is not None:
        wrist = jn[0, 0]
        print(f"\nMANO joints: wrist={wrist.round(6)} palm745={palm_745.round(6)} J4={jn[0,4].round(6)} J8={jn[0,8].round(6)}")
        print(f"  Wrist→Palm={np.linalg.norm(palm_745-wrist):.4f}  J4→J8={np.linalg.norm(jn[0,8]-jn[0,4]):.4f} m")
    print("[diagnose:mano] done.")


def main():
    parser = argparse.ArgumentParser(description="Stage1 MANO/抓取坐标链诊断")
    parser.add_argument("--mode", choices=["full", "mano"], default="full")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--data-root", default="/home/an/data")
    args = parser.parse_args()
    if args.mode == "full":
        diagnose_full(args.side, args.data_root)
    else:
        diagnose_mano(args.side, args.data_root)


if __name__ == "__main__":
    main()
