#!/usr/bin/env python3
"""ik_curve.py —— 测 cost 随迭代的改善曲线，为停滞早退定阈值。

上一轮教训: IK_CONV_TOL=1e-6 这个判据在数学上不可能满足(cost 末值 ~8,
差 7 个数量级), 所以"收敛早退"从未触发。这次先量出真实曲线再定阈值。
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
lo, hi = G.IK_LIMITS[:, 0], G.IK_LIMITS[:, 1]

# 一个可达的目标(桌面上的盘子附近) 和 一个不可达的目标(太高)
TARGETS = {
    "reachable(盘上0.80)": np.array([0.184, 0.228, 0.800]),
    "too_high(0.95)":      np.array([0.184, 0.228, 0.950]),
}
app = np.array([0.0, 0.0, -1.0])
fin = np.array([0.0, 1.0, 0.0])


def cost_of(xv, pos_t):
    q = robot.get_qpos().copy()
    for k, si in enumerate(sel):
        q[si] = xv[k]
    robot.set_qpos(q)
    M = g_link.get_pose().to_transformation_matrix()
    p = M[:3, 3] + M[:3, 0] * G.PAD_FWD
    R = M[:3, :3]
    c = 3000.0 * np.sum((p - pos_t) ** 2)
    c += 2.0 * (1.0 - R[:, 0] @ app)
    c += 1.0 * (1.0 - abs(R[:, 1] @ fin))
    return c, p


for tname, pos_t in TARGETS.items():
    print(f"\n=== 目标 {tname} ===")
    x = np.clip(np.array(G.R1_LEFT_START, dtype=float), lo, hi)
    step = 0.4
    c, p = cost_of(x, pos_t)
    traj = [(0, c, np.linalg.norm(p - pos_t))]
    it = 0
    for it in range(1, 1501):
        g = np.zeros(6)
        for k in range(6):
            dx = np.zeros(6); dx[k] = 1e-4
            g[k] = (cost_of(x + dx, pos_t)[0] - c) / 1e-4
        xn = np.clip(x - step * g, lo, hi)
        cn, pn = cost_of(xn, pos_t)
        if cn < c:
            x, c, p = xn, cn, pn
        else:
            step *= 0.5
            if step < 1e-5:
                break
        traj.append((it, c, np.linalg.norm(p - pos_t)))
    n = len(traj)
    print(f"  实际迭代 {n} 次, 末 cost={c:.4e}, 末位置误差={np.linalg.norm(p-pos_t)*1000:.2f} mm")
    # 每 100 迭代打一个点
    for i in range(0, n, max(1, n // 10)):
        _, ci, ei = traj[i]
        print(f"    it={i:5d}  cost={ci:10.4e}  位置误差={ei*1000:7.2f} mm")
    # 相邻迭代的 cost 改善量
    imp = np.array([traj[i][1] - traj[i + 1][1] for i in range(n - 1)])
    rel = imp / np.maximum(np.array([t[1] for t in traj[:-1]]), 1e-12)
    print(f"  绝对改善量: 前100次中位={np.median(imp[:100]):.3e}  "
          f"末100次中位={np.median(imp[-100:]) if n>100 else float('nan'):.3e}")
    print(f"  相对改善量: 前100次中位={np.median(rel[:100]):.3e}  "
          f"末100次中位={np.median(rel[-100:]) if n>100 else float('nan'):.3e}")
    # 各阈值下会在第几次退出
    for eps in [1e-4, 1e-6, 1e-8]:
        for N in [10, 20]:
            stall = 0; cut = n
            for i in range(n - 1):
                if imp[i] < eps:
                    stall += 1
                    if stall >= N:
                        cut = i + 1
                        break
                else:
                    stall = 0
            print(f"    阈值 eps={eps:.0e} N={N:2d} -> 第 {cut:5d} 次退出 "
                  f"(省 {100*(1-cut/n):.0f}%)")
