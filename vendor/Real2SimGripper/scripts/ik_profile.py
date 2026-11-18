#!/usr/bin/env python3
"""ik_profile.py —— 回答两个问题：(1) GPU 用上了吗 (2) 慢在哪。

测三件事：
  A. PhysX 是否跑在 GPU 上
  B. 单次 FK 的耗时拆分：set_qpos vs get_pose（哪个是瓶颈）
  C. 当前早退判据到底触发没有 —— 插桩 solve_ik 的迭代次数
"""
import os
import subprocess
import sys
import tempfile
import time

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

# ---------- A. GPU 状态 ----------
print("=== A. PhysX / GPU 状态 ===")
try:
    import sapien.physx as physx
    for attr in ("get_device", "is_gpu_enabled", "get_cuda_device"):
        if hasattr(physx, attr):
            try:
                print(f"  physx.{attr}() = {getattr(physx, attr)()}")
            except Exception as e:
                print(f"  physx.{attr}() -> {e}")
    print("  physx 模块属性:", [a for a in dir(physx) if not a.startswith('_')][:14])
except Exception as e:
    print("  sapien.physx 导入失败:", e)
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=5).stdout.strip()
    print("  GPU:", out.replace("\n", " | "))
except Exception:
    pass

# ---------- B. FK 耗时拆分 ----------
print("\n=== B. 单次 FK 拆分 (2000 次平均) ===")
x = np.array([0.35, 0.55, -0.25, -1.05, 0.0, 0.0])
q = robot.get_qpos().copy()
for k, si in enumerate(sel):
    q[si] = x[k]
robot.set_qpos(q)
n = 2000
t0 = time.time()
for _ in range(n):
    robot.set_qpos(q)
t_set = (time.time() - t0) / n
t0 = time.time()
for _ in range(n):
    g_link.get_pose().to_transformation_matrix()
t_get = (time.time() - t0) / n
print(f"  set_qpos            : {t_set*1e6:7.2f} us")
print(f"  get_pose+to_matrix  : {t_get*1e6:7.2f} us")
print(f"  合计                : {(t_set+t_get)*1e6:7.2f} us")
print(f"  纯 FK 数学量级估算  : ~2 us (20-DoF 串联链)")
print(f"  => 往返开销占       : {100*(1-2e-6/(t_set+t_get)):.0f}%")

# ---------- C. 早退触发情况 ----------
print("\n=== C. 当前早退判据触发情况 ===")
print(f"  IK_CONV_TOL = {G.IK_CONV_TOL:.0e} (cost 低于它才早退)")
print("  cost = 3000*||p-p*||^2 + 2(1-a.ah) + 1(1-|f.fh|) + 0.05||x-x_ref||^2")
print("  => 若臂不可达, ||p-p*|| 有地板; 3000*地板^2 常远大于 1e-6 => 永不早退")

pos_t = np.array([0.184, 0.228, 0.82])
app = np.array([0.0, 0.0, -1.0])
fin = np.array([0.0, 1.0, 0.0])


def cost_of(xv):
    qq = robot.get_qpos().copy()
    for k, si in enumerate(sel):
        qq[si] = xv[k]
    robot.set_qpos(qq)
    M = g_link.get_pose().to_transformation_matrix()
    p = M[:3, 3] + M[:3, 0] * G.PAD_FWD
    R = M[:3, :3]
    c = 3000.0 * np.sum((p - pos_t) ** 2)
    c += 2.0 * (1.0 - R[:, 0] @ app)
    c += 1.0 * (1.0 - abs(R[:, 1] @ fin))
    return c, p


lo, hi = G.IK_LIMITS[:, 0], G.IK_LIMITS[:, 1]
print(f"\n  单种子实测 (目标={pos_t.tolist()}):")
for trial in range(3):
    base = np.array(G.R1_LEFT_START, dtype=float)
    if trial == 1:
        base = base + np.array([0.2, -0.2, 0.1, -0.1, 0.1, -0.1])
    if trial == 2:
        base = base + np.array([-0.2, 0.2, -0.1, 0.1, -0.1, 0.1])
    xs = np.clip(base, lo, hi)
    step = 0.4
    c, p = cost_of(xs)
    it = 0
    for it in range(1, 1501):
        if c < G.IK_CONV_TOL:
            break
        g = np.zeros(6)
        for k in range(6):
            dx = np.zeros(6); dx[k] = 1e-4
            g[k] = (cost_of(xs + dx)[0] - c) / 1e-4
        xn = np.clip(xs - step * g, lo, hi)
        cn = cost_of(xn)[0]
        if cn < c:
            xs, c = xn, cn
        else:
            step *= 0.5
            if step < 1e-5:
                break
    print(f"    种子{trial}: 迭代 {it:5d} 次  末 cost={c:.3e}  "
          f"位置项={3000.0*np.sum((p-pos_t)**2):.3e}  pad={np.round(p,4).tolist()}")

print("\n结论见上。若迭代数接近 1500 且末 cost >> 1e-6 => 早退判据形同不存在")
