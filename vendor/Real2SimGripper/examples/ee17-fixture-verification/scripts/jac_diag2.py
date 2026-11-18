#!/usr/bin/env python3
"""jac_diag2.py —— 用大扰动(±0.05 rad)实测 pad 位移，判定解析/数值谁对。

小差分可能被 set_qpos+get_pose 的往返精度吃掉；大扰动则不可能。
若实测位移与解析预测一致 => 解析对、小差分噪声大；反之则解析错。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robotwin_r1_duck_grasp as G
import sapien.core as sapien
from robot_config import absolutize_urdf

_fd, ABS = tempfile.mkstemp(suffix=".urdf", prefix="r1_abs_")
os.close(_fd)
absolutize_urdf(G.R1_URDF, ABS)

scene = G.setup_scene(G.sapien.Engine())
scene.set_timestep(1 / 250)
loader = scene.create_urdf_loader()
loader.fix_root_link = True
robot = loader.load(ABS)
robot.set_root_pose(sapien.Pose(p=[0.30, -0.38, 0.0], q=G.R1_BASE_Q))

all_j = robot.get_active_joints()
name2idx = {j.get_name(): k for k, j in enumerate(all_j)}
sel = [name2idx[n] for n in G.ARM_JOINTS]
g_link = robot.find_link_by_name(G.GRIP_LINK)
link_idx = g_link.index
pm = robot.create_pinocchio_model(gravity=[0, 0, -9.81])


def set_arm(x):
    q = robot.get_qpos().copy()
    for k, si in enumerate(sel):
        q[si] = x[k]
    robot.set_qpos(q)


def pad_of():
    M = g_link.get_pose().to_transformation_matrix()
    return M[:3, 3] + M[:3, 0] * G.PAD_FWD, M[:3, :3]


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


x = np.array([0.35, 0.55, -0.25, -1.05, 0.0, 0.0])
set_arm(x)
p0, R = pad_of()
print("pad =", np.round(p0, 4).tolist(), " 相对基座 =", np.round(p0 - np.array([0.30, -0.38, 0]), 4).tolist())

# 解析
q = robot.get_qpos().copy()
pm.compute_full_jacobian(q)
Jf = np.asarray(pm.get_link_jacobian(link_idx, local=False))
Ja = (Jf[:3, :] + skew(R[:, 0] * G.PAD_FWD) @ Jf[3:, :])[:, sel]

D = 0.05
print(f"\n{'关节':<6}{'实测 Δpad/Δq':<34}{'解析预测':<34}{'cos':>7}")
print("-" * 84)
for k in range(6):
    set_arm(x)
    xp = x.copy(); xp[k] += D
    set_arm(xp)
    pp, _ = pad_of()
    meas = (pp - p0) / D
    pred = Ja[:, k]
    na, nb = np.linalg.norm(meas), np.linalg.norm(pred)
    cos = float(meas @ pred / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0
    print(f"j{k:<5}{np.round(meas,4).tolist()!s:<34}{np.round(pred,4).tolist()!s:<34}{cos:7.3f}")

# 数值差分(小 EPS) vs 大扰动实测 —— 验证当前生产梯度是否可靠
print("\n=== 数值差分 vs 大扰动实测（验证当前生产梯度）===")
set_arm(x); p0c, _ = pad_of()
for eps in [1e-2, 1e-3]:
    set_arm(x)
    Jn = np.zeros((3, 6))
    for k in range(6):
        xp = x.copy(); xp[k] += eps
        set_arm(xp)
        pp, _ = pad_of()
        Jn[:, k] = (pp - p0c) / eps
    # 与 D=0.05 的实测比
    Jm = np.zeros((3, 6))
    for k in range(6):
        xp = x.copy(); xp[k] += D
        set_arm(xp)
        pp, _ = pad_of()
        Jm[:, k] = (pp - p0c) / D
    print(f"  EPS={eps:.0e}: ‖Jn−J_large‖/‖J_large‖ = "
          f"{np.linalg.norm(Jn-Jm)/max(np.linalg.norm(Jm),1e-12):.5f}")

print("\n=== 数值差分 vs EPS（看是否被精度吃掉）===")
print(f"{'EPS':<10}{'‖Jn−Ja‖/‖Ja‖':>16}")
for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
    set_arm(x); p0b, _ = pad_of()
    Jn = np.zeros((3, 6))
    for k in range(6):
        xp = x.copy(); xp[k] += eps
        set_arm(xp)
        pp, _ = pad_of()
        Jn[:, k] = (pp - p0b) / eps
    print(f"{eps:<10.0e}{np.linalg.norm(Jn - Ja)/max(np.linalg.norm(Ja),1e-12):16.4f}")
