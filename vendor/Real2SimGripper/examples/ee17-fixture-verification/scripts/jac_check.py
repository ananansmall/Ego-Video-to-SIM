#!/usr/bin/env python3
"""jac_check.py —— 一分钟小验证：Pinocchio 解析雅可比 vs 数值差分。

目的（对应下一步建议第 1 条）：
    在抓取/搬运/释放 3 个实际姿态上，核对 pad 位置的解析雅可比与数值差分是否一致，
    并测耗时。不改目标、不改限位、不改种子数，只验梯度本身。

pad 定义（与核心脚本一致）：
    pad = link_origin + R[:, 0] * PAD_FWD
    => J_pad = J_v + skew(R[:,0] * PAD_FWD) @ J_w        （空间雅可比前3行线性/后3行角速度）

通过判据：每个姿态的相对误差 < 1e-4（数值差分步长 1e-4，截断误差同量级）。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import robotwin_r1_duck_grasp as G
import sapien.core as sapien
from robot_config import absolutize_urdf

import tempfile
_fd, ABS = tempfile.mkstemp(suffix=".urdf", prefix="r1_abs_")
os.close(_fd)
absolutize_urdf(G.R1_URDF, ABS)          # src -> dst, dst 是文件不是目录
BASE_XY = [0.30, -0.38]

# 3 个姿态：初始位 / 类抓取（前伸）/ 类搬运（抬高）
POSES = {
    "start":  np.array(G.R1_LEFT_START, dtype=float),
    "reach":  np.array([0.35, 0.55, -0.25, -1.05, 0.00, 0.00]),
    "carry":  np.array([0.15, 0.85, -0.15, -0.75, 0.00, 0.00]),
}


def main():
    scene = G.setup_scene(G.sapien.Engine())
    scene.set_timestep(1 / 250)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(ABS)
    robot.set_root_pose(sapien.Pose(p=[*BASE_XY, 0.0], q=G.R1_BASE_Q))

    all_j = robot.get_active_joints()
    name2idx = {j.get_name(): k for k, j in enumerate(all_j)}
    sel = [name2idx[n] for n in G.ARM_JOINTS]
    g_link = robot.find_link_by_name(G.GRIP_LINK)
    link_idx = [l.index for l in robot.get_links() if l.get_name() == G.GRIP_LINK]
    if not link_idx:
        # 退化: 用 index 属性直接找
        link_idx = [g_link.index]
    link_idx = link_idx[0]

    pm = robot.create_pinocchio_model(gravity=[0, 0, -9.81])
    print(f"[ok] Pinocchio 模型已建立; gripper link index = {link_idx}; "
          f"臂关节索引 = {sel}", flush=True)

    # ---------- 数值 FK（与核心脚本同一个 pad 定义） ----------
    def pad_num(x):
        q = robot.get_qpos().copy()
        for k, si in enumerate(sel):
            q[si] = x[k]
        robot.set_qpos(q)
        M = g_link.get_pose().to_transformation_matrix()
        return M[:3, 3] + M[:3, 0] * G.PAD_FWD, M[:3, :3]

    def skew(a):
        return np.array([[0.0, -a[2], a[1]],
                         [a[2], 0.0, -a[0]],
                         [-a[1], a[0], 0.0]])

    # ---------- 解析雅可比 ----------
    def pad_jac_analytic(x, use_pinocchio_fk=False):
        """解析 pad 雅可比。⚠ get_link_jacobian 读的是缓存, 必须先 compute_full_jacobian
        (第一次跑漏了这句, 结果解析雅可比是零矩阵, 相对误差正好 1.0)。"""
        q = robot.get_qpos().copy()
        for k, si in enumerate(sel):
            q[si] = x[k]
        pm.compute_full_jacobian(q)                          # ← 必须先算, 否则读到空缓存
        J = np.asarray(pm.get_link_jacobian(link_idx, local=False))   # 6 x nv, WORLD 系
        Jlin, Jang = J[:3, :], J[3:, :]
        if use_pinocchio_fk:
            # 完全走 Pinocchio, 不碰 PhysX 的 set_qpos/get_pose
            pm.compute_forward_kinematics(q)
            R = pm.get_link_pose(link_idx).to_transformation_matrix()[:3, :3]
        else:
            _, R = pad_num(x)
        return Jlin + skew(R[:, 0] * G.PAD_FWD) @ Jang

    EPS = 1e-4
    rows = []
    for tag, x in POSES.items():
        p0, _ = pad_num(x)
        Ja = pad_jac_analytic(x)                      # 3 x nv
        Ja = Ja[:, sel]                               # 只留 6 个臂关节列
        # 数值差分
        Jn = np.zeros((3, 6))
        for k in range(6):
            dx = np.zeros(6); dx[k] = EPS
            p1, _ = pad_num(x + dx)
            Jn[:, k] = (p1 - p0) / EPS
        denom = max(np.linalg.norm(Jn), 1e-12)
        rel = np.linalg.norm(Ja - Jn) / denom
        rows.append((tag, rel, Ja, Jn))
        print(f"  {tag:6s} ‖J_analytic−J_numeric‖/‖J_numeric‖ = {rel:.3e}   "
              f"pad = {np.round(p0, 4).tolist()}", flush=True)

    # ---------- 耗时对比（在 carry 姿态上）──────────
    x = POSES["carry"]
    n = 300
    t0 = time.time()
    for _ in range(n):
        pad_num(x)
    t_fk = (time.time() - t0) / n
    t0 = time.time()
    for _ in range(n):
        pad_jac_analytic(x, use_pinocchio_fk=True)
    t_jac = (time.time() - t0) / n
    # 数值梯度（核心脚本现在的做法）= 6 次 FK
    t0 = time.time()
    for _ in range(30):
        for k in range(6):
            dx = np.zeros(6); dx[k] = EPS
            pad_num(x + dx)
    t_num6 = (time.time() - t0) / 30

    print()
    print(f"  单次 FK(set_qpos+get_pose)      : {t_fk*1e6:8.1f} us")
    print(f"  解析雅可比(get_link_jacobian)   : {t_jac*1e6:8.1f} us")
    print(f"  6 次数值差分（当前生产做法）     : {t_num6*1e6:8.1f} us")
    if t_num6 > 0:
        print(f"  => 解析梯度相对数值差分快        : {t_num6/t_jac:6.1f} x")

    worst = max(r[1] for r in rows)
    print()
    print("结论：" + ("解析雅可比与数值差分一致，可接入生产 IK"
                      if worst < 1e-4 else "误差偏大，需查轴系/参考系"))
    print(f"JAC_CHECK_DONE worst_rel={worst:.3e}")


if __name__ == "__main__":
    main()
