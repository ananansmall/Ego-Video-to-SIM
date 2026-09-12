"""对比实验：现有 SG 平滑 (smooth_joints_seq) vs MINT UKF 平滑 (mint_cleaning.final_clean)。

三路对照（都走相同 FK + SVD 重定向，只有平滑层不同）：
  A. 不平滑（raw）                     —— 基准
  B. 现有方案：平滑 FK 后关节（SG）     —— smooth_joints_seq
  C. MINT 方案：先参数域 final_clean（UKF+RTS），再 FK

输出指标（相对 raw）：
  - 每帧夹爪 pos/R 偏差（mean/max/以及重要帧 F38/F50/F62）
  - 平滑度（位置二阶差分 rms / 角速度二阶差分 rms）
  - 验证 F50 单帧（Stage1 pin 点）是否受影响
"""
import numpy as np
from pathlib import Path

np.set_printoptions(precision=4, suppress=True)

BASE = Path("/home/an/robot_world_ws/src/dex-retargeting/example/combination/trajectory_optimization")
DATA = Path("/home/an/data/hawor/7a/reconstruction/hawor_results_0_113.npz")
SIDE = "left"           # 管线只用 index0 = 真实左手
HAND_IDX = 0
betas_mean = None       # 用数据自带 pred_betas

import sys
sys.path.insert(0, str(BASE))
from trajectory_loader import compute_mano_joints
from data_loader import compute_mano_based_gripper_pose
from mano_smoothing import smooth_joints_seq
from mint_cleaning.final_clean import final_clean
sys.path.insert(0, "/home/an/robot_world_ws/src/dex-retargeting/example/position_retargeting")
from mano_layer import MANOLayer

# ---------- 加载数据 ----------
z = np.load(str(DATA), allow_pickle=True)
pred_trans    = np.asarray(z["pred_trans"], np.float64)[HAND_IDX]        # (N,3)
pred_rot      = np.asarray(z["pred_rot"], np.float64)[HAND_IDX]          # (N,3)
pred_hand_pose= np.asarray(z["pred_hand_pose"], np.float64)[HAND_IDX]    # (N,45)
pred_betas    = np.asarray(z["pred_betas"], np.float32)[HAND_IDX]        # (N,10)
pred_valid    = np.asarray(z["pred_valid"])[HAND_IDX]
R_c2w         = np.asarray(z["R_c2w"], np.float64)                       # (N,3,3)
t_c2w         = np.asarray(z["t_c2w"], np.float64)                       # (N,3)
N = len(pred_trans)
print(f"数据: N={N}, valid={int(pred_valid.sum())}/{N}, hand_idx={HAND_IDX}({SIDE})")

# MANO layer（用首帧 betas）
ml = MANOLayer(SIDE, pred_betas[0].astype(np.float32))

def fk_sequence(trans, rot, hand_pose):
    """逐帧 FK → (N,21,3) joints，正/_valid 帧才计算"""
    J = np.zeros((N, 21, 3), np.float64)
    for i in range(N):
        if pred_valid[i]:
            _, j = compute_mano_joints(ml, rot[i], hand_pose[i], trans[i])
            J[i] = j
    return J

# ---------- 三种平滑参数 ----------
def retarget(J):
    """SVD 重定向 → 夹爪 pos/R（与数据_loader 同函数）"""
    pos = np.zeros((N, 3)); Rm = np.zeros((N, 3, 3))
    for i in range(N):
        if pred_valid[i]:
            p, R, _, _ = compute_mano_based_gripper_pose(J[i], prefix=SIDE)
            pos[i], Rm[i] = p, R
    return pos, Rm

# ---------- 三路 ----------
# A: raw 关节（FK 原始）
J_raw = fk_sequence(pred_trans, pred_rot, pred_hand_pose)
# B: 现有 — 平滑 FK 后关节（SG）
J_sg = smooth_joints_seq(J_raw)
# C: MINT — 参数域 final_clean，再 FK。用相机 c2w 做相机感知
# final_clean 硬编码处理 2 只手 (for h in range(2))，把左手数据复制到两手槽位，取 result[0]
pred_result = [np.stack([pred_trans, pred_trans], axis=0),
               np.stack([pred_rot, pred_rot], axis=0),
               np.stack([pred_hand_pose, pred_hand_pose], axis=0),
               np.stack([pred_betas, pred_betas], axis=0),
               np.stack([pred_valid, pred_valid], axis=0)]
cleaned = final_clean(pred_result,
                      cam_c2w=np.concatenate([R_c2w, t_c2w[..., None]], axis=-1))  # (N,3,4)
# 拆回 (N, ...)
J_mint = fk_sequence(np.asarray(cleaned[0][0], np.float64),
                     np.asarray(cleaned[1][0], np.float64),
                     np.asarray(cleaned[2][0], np.float64))

pos_raw, R_raw = retarget(J_raw)
pos_sg,  R_sg  = retarget(J_sg)
pos_mit, R_mit = retarget(J_mint)

# ---------- 指标 ----------
def pos_dev(a, b, name, mask):
    d = np.linalg.norm(a - b, axis=1)
    print(f"  [{name}] pos|Δ| mean={d[mask].mean()*1000:.1f}mm max={d[mask].max()*1000:.1f}mm "
          f"F38={d[38]*1000:.1f}mm F50={d[50]*1000:.1f}mm F62={d[62]*1000:.1f}mm")

def angular_dev(Ra, Rb, name, mask):
    # R 相对转角的轴角幅值
    rel = Rb @ np.transpose(Ra, (0,2,1))
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2)-1)/2, -1, 1)))
    print(f"  [{name}] ∠|Δ| mean={ang[mask].mean():.2f}° max={ang[mask].max():.2f}° "
          f"F38={ang[38]:.2f}° F50={ang[50]:.2f}° F62={ang[62]:.2f}°")

def smoothness(pos, name):
    acc = np.linalg.norm(np.diff(pos, axis=0, n=2), axis=1)
    print(f"  [{name}] 位置二阶差分 rms={acc.mean()*1000:.3f}mm")

mask = pred_valid
print("\n===== 相对 raw 的夹爪偏差 (valid 帧) =====")
pos_dev(pos_sg,  pos_raw, "现有SG", mask)
pos_dev(pos_mit, pos_raw, "MINT-UKF", mask)
print("----- 角度 -----")
angular_dev(R_sg,  R_raw, "现有SG", mask)
angular_dev(R_mit, R_raw, "MINT-UKF", mask)

print("\n===== 平滑度 (越小越平滑, 各方案自身) =====")
smoothness(pos_raw, "raw  ")
smoothness(pos_sg,  "现有SG")
smoothness(pos_mit, "MINT-UKF")

print("\n===== SG 轨迹 vs MINT 轨迹 互差 (两方案之间的主要差异) =====")
d = np.linalg.norm(pos_sg - pos_mit, axis=1)
print(f"  pos|SG-MINT| mean={d[mask].mean()*1000:.1f}mm max={d[mask].max()*1000:.1f}mm "
      f"F38={d[38]*1000:.1f} F50={d[50]*1000:.1f} F62={d[62]*1000:.1f}")
rel = R_mit @ np.transpose(R_sg, (0,2,1))
ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2)-1)/2, -1, 1)))
print(f"  ∠|SG-MINT| mean={ang[mask].mean():.2f}° max={ang[mask].max():.2f}° "
      f"F38={ang[38]:.2f} F50={ang[50]:.2f} F62={ang[62]:.2f}")