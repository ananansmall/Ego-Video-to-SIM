#!/usr/bin/env python3
"""jac_diag.py —— 逐列对比解析/数值 pad 雅可比，定位不一致的原因。

一次跑完就打印：数值列 vs 解析列的相关系数、范数比、以及是否只是列置换。
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


def pad_num(x):
    q = robot.get_qpos().copy()
    for k, si in enumerate(sel):
        q[si] = x[k]
    robot.set_qpos(q)
    M = g_link.get_pose().to_transformation_matrix()
    return M[:3, 3] + M[:3, 0] * G.PAD_FWD, M[:3, :3]


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


x = np.array([0.35, 0.55, -0.25, -1.05, 0.0, 0.0])
EPS = 1e-4
p0, R = pad_num(x)

# 数值列
Jn = np.zeros((3, 6))
for k in range(6):
    dx = np.zeros(6); dx[k] = EPS
    p1, _ = pad_num(x + dx)
    Jn[:, k] = (p1 - p0) / EPS

# 解析: 三种取法
q = robot.get_qpos().copy()
for k, si in enumerate(sel):
    q[si] = x[k]

variants = {}
pm.compute_full_jacobian(q)
J_full = np.asarray(pm.get_link_jacobian(link_idx, local=False))
variants["full,local=False"] = (J_full[:3, :] + skew(R[:, 0] * G.PAD_FWD) @ J_full[3:, :])
J_loc = np.asarray(pm.get_link_jacobian(link_idx, local=True))
variants["full,local=True"] = (J_loc[:3, :] + skew(R[:, 0] * G.PAD_FWD) @ J_loc[3:, :])
J_sl = np.asarray(pm.compute_single_link_local_jacobian(q, link_idx))
variants["single_local"] = (J_sl[:3, :] + skew(R[:, 0] * G.PAD_FWD) @ J_sl[3:, :])

print("数值 pad 雅可比 (3x6):")
print(np.round(Jn, 3))
print()
for name, J in variants.items():
    print(f"--- {name}: shape={J.shape} ---")
    print(np.round(J, 3))
    # 尝试各种列选择
    for csel, label in [(np.arange(J.shape[1]), "全部列"),
                        (sel, "sel 列"),
                        ([s - sel[0] for s in sel], "sel-首列")]:
        try:
            sub = J[:, csel]
            if sub.shape != (3, 6):
                continue
            rel = np.linalg.norm(sub - Jn) / max(np.linalg.norm(Jn), 1e-12)
            # 逐列相关
            corr = []
            for k in range(6):
                a, b = sub[:, k], Jn[:, k]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                corr.append(float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0)
            print(f"   {label:10s} rel_err={rel:8.3f}  逐列cos={np.round(corr,2).tolist()}")
        except Exception as e:
            print(f"   {label}: {e}")
    print()
