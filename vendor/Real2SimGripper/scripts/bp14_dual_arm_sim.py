#!/usr/bin/env python3
"""bp14_dual_arm_sim.py — bp14 双手重定向轨迹 → 完整 R1 双臂物理仿真渲染。

设计原则 (对齐用户要求: "只学重定向, 渲染/仿真沿用之前那套"):
  - **重定向**: 用 Ego-Video-to-SIM 的 DexRetargeting + RelaxedIK 产出的双臂关节
    轨迹 (保存在 tracking/*.npy, 6 臂关节 + 2 指, 纯关节空间, 与基座位姿无关)。
  - **场景/物理/渲染**: 完全沿用之前管线 robotwin_r1_duck_grasp.py —— setup_scene
    光照 + create_table 桌子 + 完整 robot.urdf + PD 驱动 + 逐物理步重力补偿 +
    多视角渲染。**不加载任何 GLB 道具** (bp14 无场景重建)。

关节约定: floating arm URDF 与完整 robot.urdf 的 *_arm_joint1..6 轴/限位完全一致;
但左臂重定向输出用的是镜像约定 (LEFT_ARM_STARTING), 需翻转 j1/2/3/4/6 对齐 URDF。
脚本会逐臂自动判定 (identity / 镜像) 并在接入前做关节限位裁剪。
"""
import os
import sys
import json
import argparse
import numpy as np

import sapien.core as sapien
import imageio
from scipy.optimize import least_squares, minimize
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robotwin_r1_duck_grasp as rg   # noqa: E402  复用场景/物理/相机/常量

REPO_LIBS = "/path/to/workspace/Ego-Video-to-SIM/libs"
R1_DIR = os.path.join(REPO_LIBS, "GalaxeaManipSim", "galaxea_sim", "assets", "r1")
R1_URDF = os.path.join(R1_DIR, "robot.urdf")

# bp14 重定向关节轨迹 (Ego-Video-to-SIM DexRetargeting+RelaxedIK 产出)
QPOS_DIR = "/path/to/workspace/_scratch/bp14/render/tracking"


def out_gif(mode, yaw):
    """按基座方案命名输出, 便于各方案独立测试/对比。"""
    tag = f"bp14_dual_arm_{mode}" + ("_yaw" if yaw else "")
    return os.path.join(rg.OUT_DIR, f"{tag}.gif")


RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")


def _dump_metrics(payload):
    """按基座方案把指标+配置落盘到 results/bp14_{mode}.json, 供后续对比。

    此前指标只在终端打印、不留档 -> 每次改动无法直接对比, 只能重跑。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"bp14_{payload['config']['base_mode']}.json")
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=float)
    print(f"  指标 -> {path}")
    return path

# HaWoR 原始数据 (用于从 MANO 求夹爪开合量; 重定向存下的指关节是常量, 不可用)
HAWOR_DIR = "/path/to/project_data/mano_ws/output/mano/basic_pick_place_14/hawor"
_EGO = "/path/to/workspace/Ego-Video-to-SIM"
FINGER_BASE_DIST = 0.026906   # 夹爪两指基距 (对齐 Ego-Video-to-SIM physics_utils.py)
GRIP_MAX = 0.05               # 指关节上限 (URDF prismatic limit)

# 完整 R1 关节限位 (URDF)
ARM_LIMITS = {
    "left_arm_joint1": (-2.8798, 2.8798), "left_arm_joint2": (0.0, 3.2289),
    "left_arm_joint3": (-3.3161, 0.0), "left_arm_joint4": (-2.8798, 2.8798),
    "left_arm_joint5": (-1.6581, 1.6581), "left_arm_joint6": (-2.8798, 2.8798),
    "right_arm_joint1": (-2.8798, 2.8798), "right_arm_joint2": (0.0, 3.2289),
    "right_arm_joint3": (-3.3161, 0.0), "right_arm_joint4": (-2.8798, 2.8798),
    "right_arm_joint5": (-1.6581, 1.6581), "right_arm_joint6": (-2.8798, 2.8798),
}
# 重定向约定 -> URDF 约定的逐关节符号 (左臂为镜像: 翻转 j1/2/3/4/6)
LEFT_M = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
RIGHT_M = np.ones(6)


def _limits(side):
    return np.array([ARM_LIMITS[f"{side}_arm_joint{i+1}"] for i in range(6)])


def _map_arm(q6, side, m):
    """重定向关节 -> URDF 约定 + 限位裁剪。"""
    lo, hi = _limits(side).T
    return np.clip(q6 * m, lo, hi)


def _pick_convention(q6, side):
    """在 {identity, 镜像} 中自动选越限更少的约定。"""
    cands = {"identity": np.ones(6), "mirror": LEFT_M if side == "left" else RIGHT_M}
    best, best_bad = "identity", 1 << 30
    for name, m in cands.items():
        lo, hi = _limits(side).T
        q = q6 * m
        bad = int(np.sum((q < lo - 1e-6) | (q > hi + 1e-6)))
        if bad < best_bad:
            best, best_bad = name, bad
        print(f"    [{side}] 约定 {name}: 越限帧 {bad}")
    return cands[best], best


def finger_from_aperture(hand_idx, n_frames):
    """从 MANO 拇指尖-食指尖距离求夹爪指关节开合量。

    对齐 Ego-Video-to-SIM compute_analytical_gripper_pose 的思路 (该库的 SVD 也是
    先用 finger_dist 定指关节): joint = clip((dist - base)/2, 0, 0.05)。
    MANO FK 依赖 manopth, 仅在 vpho 环境可用; 本仿真在 robotwin_curobo 环境,
    故优先读预生成缓存 (grip_*.npy, 见 _scratch/gen_grip_cache.py), 无缓存再回退现算。
    """
    prefix = "left" if hand_idx == 0 else "right"
    cache = os.path.join(QPOS_DIR, f"grip_{prefix}.npy")
    if os.path.exists(cache):
        out = np.load(cache)
        print(f"  [{prefix}] 夹爪开合量(缓存): {out[:,0].min():.4f}~{out[:,0].max():.4f} m")
        return out
    import torch
    sys.path.insert(0, os.path.join(_EGO, "libs", "position_retargeting"))
    sys.path.insert(0, os.path.join(_EGO, "combination", "hand_track"))
    from mano_layer import MANOLayer
    from common import load_hawor_data, compute_mano_joints
    hd = load_hawor_data(HAWOR_DIR, hand_idx=hand_idx)
    ml = MANOLayer(prefix, hd["pred_betas"][0].astype("float32"))
    out = np.zeros((n_frames, 2))
    last = np.zeros(2)
    n = min(n_frames, len(hd["pred_rot"]))
    for i in range(n):
        if not hd["pred_valid"][i]:
            out[i] = last
            continue
        _, J = compute_mano_joints(ml, hd["pred_rot"][i], hd["pred_hand_pose"][i], hd["pred_trans"][i])
        d = float(np.linalg.norm(J[4] - J[8]))          # 拇指尖(4) - 食指尖(8)
        jv = float(np.clip((d - FINGER_BASE_DIST) / 2.0, 0.0, GRIP_MAX))
        last = np.array([jv, jv])
        out[i] = last
    out[n:] = last
    print(f"  [{prefix}] 夹爪开合量: {out[:,0].min():.4f}~{out[:,0].max():.4f} m "
          f"(MANO 指尖距范围)")
    return out


def _gripper_pad_fk(robot, side, sel_idx, q6):
    """设臂关节 -> 返回 (pad 世界位置, gripper 旋转)。"""
    q = robot.get_qpos().copy()
    for k, si in enumerate(sel_idx):
        q[si] = q6[k]
    robot.set_qpos(q)
    M = robot.find_link_by_name(f"{side}_gripper_link").get_pose().to_transformation_matrix()
    return M[:3, 3] + M[:3, 0] * rg.PAD_FWD, M[:3, :3].copy()


def limit_aware_resolve(robot, side, sel_idx, raw_q6, lo, hi, w_pos=10.0, w_ori=10.0,
                        target=None, seed=None, diff_step=1e-4, ori_mode="mat"):
    """把关节重解到关节限位内, 同时匹配末端位姿(位置 + 完整姿态)。

    target=(p0,R0) 显式给定世界系目标 (用于底盘移动: 目标固定于世界, 基座已挪位);
    seed 热启动 (用上一帧解, 保证帧间连续/平滑)。

    姿态残差 ori_mode:
      - "mat" (默认, 推荐): 直接用旋转矩阵差 (R - R0).reshape(-1), 仅当 R=R0 时为零。
      - "cross" (旧版, **有缺陷**): cross(a,app0)+cross(f,fin0)。反对称部分正比于 sinθ,
        在 θ=0 与 θ=180° **都为零** -> 求解器会收敛到"反平行"分支, 实测姿态误差 180°。

    diff_step: least_squares 数值差分步长。**必须显式给定**: SAPIEN 连杆位姿是 float32,
    默认差分步长 (~1.5e-8 相对) 造成的位姿变化低于 float32 精度 -> 雅可比被判为 0 ->
    求解器 nfev=1 直接返回种子 (种子不准时表现为"求解器不动")。1e-4 实测可正常收敛。
    """
    if target is None:
        p0, R0 = _gripper_pad_fk(robot, side, sel_idx, raw_q6)
    else:
        p0, R0 = target
    app0, fin0 = R0[:, 0], R0[:, 1]

    def resid(q6):
        p, R = _gripper_pad_fk(robot, side, sel_idx, q6)
        if ori_mode == "mat":
            oe = (R - R0).reshape(-1)
        else:
            a, f = R[:, 0], R[:, 1]
            oe = np.cross(a, app0) + np.cross(f, fin0)
        return np.concatenate([w_pos * (p - p0), w_ori * oe])

    x0 = np.clip(seed if seed is not None else raw_q6, lo + 1e-4, hi - 1e-4)
    sol = least_squares(resid, x0, bounds=(lo, hi), method="trf",
                        xtol=1e-11, ftol=1e-11, max_nfev=500, diff_step=diff_step)
    return sol.x


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def se2_pose(dx, dy, yaw):
    """基座位姿 = R1_BASE_P/Q 叠加世界系水平平移 (dx,dy) 与绕竖直轴 yaw。"""
    q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    qq = _qmul(q, np.asarray(rg.R1_BASE_Q, np.float64))
    p = np.asarray(rg.R1_BASE_P, np.float64) + np.array([dx, dy, 0.0])
    return sapien.Pose(p=p, q=qq)


def _yaw_R(yaw):
    """绕世界竖直轴的旋转矩阵 (用于世界<->基座变换)。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def traj_metrics(q):
    """关节轨迹平滑度: (max|Δq| deg, rms|jerk| deg)。"""
    dq = np.diff(q, axis=0)
    if len(q) >= 4:
        jerk = np.diff(q, n=3, axis=0)
        rms_jerk = np.degrees(np.sqrt((jerk ** 2).mean()))
    else:
        rms_jerk = 0.0
    return np.degrees(np.abs(dq).max()), rms_jerk


def build_scene_and_robot():
    engine = sapien.Engine()
    scene = rg.setup_scene(engine)
    # 桌子 + 后墙 (与之前管线一致)
    rg.create_table(scene, sapien.Pose(p=[0, 0, rg.TABLE_Z]), length=1.2, width=0.7,
                    height=rg.TABLE_Z, thickness=0.05, is_static=True)
    rg.create_box(scene, sapien.Pose(p=[0, 1, 1.5]), half_size=[3, 0.6, 1.5],
                  color=(1, 0.9, 0.9), name="wall", is_static=True)

    tmp_urdf = "/tmp/r1_dual_abs.urdf"
    rg.make_r1_urdf_abs(R1_URDF, tmp_urdf)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(tmp_urdf)
    # 双臂任务: 基座与桌子沿用之前管线常量, 朝向=R1_BASE_Q (面朝 +y, 桌子在 +y 侧)
    robot.set_root_pose(sapien.Pose(p=rg.R1_BASE_P, q=rg.R1_BASE_Q))
    return scene, robot


def main(base_mode="opt", use_yaw=False):
    qL = np.load(os.path.join(QPOS_DIR, "hawor_r1_dual_tracking_left.npy"))    # (N,8)
    qR = np.load(os.path.join(QPOS_DIR, "hawor_r1_dual_tracking_right.npy"))   # (N,8)
    N = min(len(qL), len(qR))
    print(f"bp14 重定向轨迹: {N} 帧 (左臂 {qL.shape}, 右臂 {qR.shape})")

    mL, cnL = _pick_convention(qL[:, :6], "left")
    mR, cnR = _pick_convention(qR[:, :6], "right")
    print(f"  采用约定: left={cnL}, right={cnR}")

    # 原始(未裁剪)重定向关节 (世界系任务目标来源); 随后用限位感知 IK 重解到限位内
    armL_raw = rg.smooth_traj(np.stack([qL[i, :6] * mL for i in range(N)]), 5, 2)
    armR_raw = rg.smooth_traj(np.stack([qR[i, :6] * mR for i in range(N)]), 5, 2)
    raw_viol = {}
    for side, arm in (("left", armL_raw), ("right", armR_raw)):
        lim = _limits(side)
        viol = int(np.sum(np.any((arm < lim[:, 0] - 1e-6) | (arm > lim[:, 1] + 1e-6), axis=1)))
        raw_viol[side] = viol
        print(f"  [{side}] 原始重定向越限帧 = {viol}/{N}")
    # 夹爪开合: 由 MANO 拇指-食指开合量数据驱动 (重定向存的指关节是常量, 弃用)
    gripL = finger_from_aperture(0, N)
    gripR = finger_from_aperture(1, N)

    scene, robot = build_scene_and_robot()
    joints = robot.get_active_joints()
    jmap = {j.get_name(): j for j in joints}
    nm2i = {j.get_name(): k for k, j in enumerate(joints)}
    Lsel = [nm2i[f"left_arm_joint{i+1}"] for i in range(6)]
    Rsel = [nm2i[f"right_arm_joint{i+1}"] for i in range(6)]
    limL, limR = _limits("left"), _limits("right")

    # 限位感知 IK: 把重定向轨迹重解到关节限位内 (保持世界系末端位姿), 消除越限裁剪误差
    robot.set_root_pose(sapien.Pose(p=rg.R1_BASE_P, q=rg.R1_BASE_Q))
    # 世界系任务目标 (固定于世界; 基座移动时目标不变)
    tgtL = [_gripper_pad_fk(robot, "left", Lsel, armL_raw[k]) for k in range(N)]
    tgtR = [_gripper_pad_fk(robot, "right", Rsel, armR_raw[k]) for k in range(N)]

    def solve_arms(bxv, byv, yawv):
        """给定基座 SE2 轨迹, 热启动逐帧限位感知 IK 求双臂关节 (保证帧间连续/平滑)。"""
        AL, AR = [], []
        sL = sR = None
        for k in range(N):
            robot.set_root_pose(se2_pose(bxv[k], byv[k], yawv[k]))
            qLk = limit_aware_resolve(robot, "left", Lsel, armL_raw[k], limL[:, 0], limL[:, 1],
                                      target=tgtL[k], seed=sL)
            qRk = limit_aware_resolve(robot, "right", Rsel, armR_raw[k], limR[:, 0], limR[:, 1],
                                      target=tgtR[k], seed=sR)
            AL.append(qLk); AR.append(qRk); sL, sR = qLk, qRk
        return np.stack(AL), np.stack(AR)

    # 世界系 pad 目标 (位置/朝向), 供基座方案使用
    pLw = np.array([t[0] for t in tgtL]); RLw = np.array([t[1] for t in tgtL])
    pRw = np.array([t[0] for t in tgtR]); RRw = np.array([t[1] for t in tgtR])
    mid = 0.5 * (pLw + pRw)
    BASE_P = np.asarray(rg.R1_BASE_P, np.float64)
    BASE_CLAMP = 0.25
    zb = np.zeros(N)

    # ===== 基座方案 (每次运行只算一种 => 各方案独立测试、独立出图) =====
    if base_mode == "static":
        bx, by, byaw = zb, zb, zb
        print("  [static] 静态基座 (基线, 无底盘移动)")
    elif base_mode == "heuristic":
        bx = np.clip(rg.smooth_traj((mid[:, 0] - mid[0, 0])[:, None], 15, 3)[:, 0], -BASE_CLAMP, BASE_CLAMP)
        by = np.clip(rg.smooth_traj((mid[:, 1] - mid[0, 1])[:, None], 15, 3)[:, 0], -BASE_CLAMP, BASE_CLAMP)
        byaw = zb
        print("  [heuristic] 固定启发式基座 (跟踪两手中点 + 低通)")
    elif base_mode == "opt":
        # 目标: 让"任务目标在基座系内的运动"尽量小 (= 臂少动/更平滑), 同时基座自身平滑、幅度受限。
        # 与 RoboTwin 的 world->base 变换同源: rel_p = wRb^T @ (p_world - base_p)。
        K = 7                    # 基座轨迹控制点数 (三次样条插值到 N)
        node_x = np.linspace(0, N - 1, K)
        free = ["dx", "dy"] + (["yaw"] if use_yaw else [])

        def base_traj(theta):
            d = {nm: CubicSpline(node_x, theta[i * K:(i + 1) * K])(np.arange(N)) for i, nm in enumerate(free)}
            z = np.zeros(N)
            return d.get("dx", z), d.get("dy", z), d.get("yaw", z)

        def rel_cost(theta):
            dxx, dyy, yaw = base_traj(theta)
            cost = 0.0; prev = None
            for k in range(N):
                Rb = _yaw_R(yaw[k])
                bp = BASE_P + np.array([dxx[k], dyy[k], 0.0])
                cur = np.concatenate([Rb.T @ (pLw[k] - bp), Rb.T @ (pRw[k] - bp),
                                      Rb.T @ RLw[k][:, 0], Rb.T @ RRw[k][:, 0]])  # 位置 + 接近轴
                if prev is not None:
                    cost += float(np.sum((cur - prev) ** 2))
                prev = cur
            bvec = np.stack([dxx, dyy, yaw], axis=1)
            cost += 25.0 * float(np.sum(np.diff(bvec, axis=0) ** 2))        # 基座速度 (强正则 -> 平滑底盘)
            cost += 8.0 * float(np.sum(np.diff(bvec, n=2, axis=0) ** 2))    # 基座加速度
            cost += 1.0 * float(np.sum(bvec ** 2))                          # 基座幅度
            over = (np.maximum(np.abs(dxx) - BASE_CLAMP, 0) + np.maximum(np.abs(dyy) - BASE_CLAMP, 0)
                    + np.maximum(np.abs(yaw) - 0.5, 0))
            cost += 5e2 * float(np.sum(over ** 2))
            return cost

        print(f"  [opt] 优化基座轨迹 (控制点 K={K}, 自由度 {free})...")
        res = minimize(rel_cost, np.zeros(K * len(free)), method="Powell",
                       options={"maxfev": 8000, "xtol": 1e-4, "ftol": 1e-4})
        bx, by, byaw = base_traj(res.x)
        print(f"  [opt] 最优代价 {res.fun:.4f}")
    else:
        raise ValueError(f"未知基座方案: {base_mode} (可选 static/heuristic/opt)")

    print(f"  [{base_mode}] 底盘: x {bx.min()*100:+.1f}~{bx.max()*100:+.1f}cm (幅度{np.ptp(bx)*100:.1f}cm)  "
          f"y {by.min()*100:+.1f}~{by.max()*100:+.1f}cm (幅度{np.ptp(by)*100:.1f}cm)  "
          f"yaw {np.degrees(np.ptp(byaw)):.1f}deg  "
          f"步长max={np.abs(np.diff(np.stack([bx, by], axis=1), axis=0)).max()*1000:.2f}mm/帧")
    # 地面路径标记: 起点(红) + 沿轨迹采样(灰), 直观显示底盘是否在移动
    rg.create_box(scene, sapien.Pose(p=[rg.R1_BASE_P[0], rg.R1_BASE_P[1], 0.006]),
                  half_size=[0.06, 0.06, 0.006], color=(1, 0, 0), name="base_start", is_static=True)
    if base_mode != "static":
        for kk in range(0, N, 8):
            rg.create_box(scene, sapien.Pose(p=[rg.R1_BASE_P[0] + bx[kk], rg.R1_BASE_P[1] + by[kk], 0.004]),
                          half_size=[0.018, 0.018, 0.004], color=(0.1, 0.1, 0.1),
                          name=f"base_path_{kk}", is_static=True)

    print(f"  [{base_mode}] 限位感知 IK (热启动)...")
    armL, armR = solve_arms(bx, by, byaw)

    # 本方案指标
    m0 = traj_metrics(rg.smooth_traj(armL, 5, 2)); m1 = traj_metrics(rg.smooth_traj(armR, 5, 2))
    print(f"  [{base_mode}] 平滑度 (max|Δq| deg, rms|jerk| deg): "
          f"left {m0[0]:.2f}/{m0[1]:.3f}   right {m1[0]:.2f}/{m1[1]:.3f}")

    armL = rg.smooth_traj(armL, 5, 2); armR = rg.smooth_traj(armR, 5, 2)
    resolve_viol = {}
    for side, arm in (("left", armL), ("right", armR)):
        lim = _limits(side)
        viol = int(np.sum(np.any((arm < lim[:, 0] - 1e-6) | (arm > lim[:, 1] + 1e-6), axis=1)))
        resolve_viol[side] = viol
        print(f"  [{side}] 底盘重解后越限帧 = {viol}/{N}")
    # ---- 指标落盘 (各基座方案独立对比) ----
    _dump_metrics({
        "task": "bp14_dual_arm",
        "config": {"base_mode": base_mode, "use_yaw": bool(use_yaw), "n_frames": int(N)},
        "base": {"amp_x_cm": float(np.ptp(bx) * 100),
                 "amp_y_cm": float(np.ptp(by) * 100),
                 "yaw_deg": float(np.degrees(np.ptp(byaw))),
                 "step_max_mm": float(np.abs(np.diff(np.stack([bx, by], axis=1), axis=0)).max() * 1000)
                 if N > 1 else 0.0},
        "arm": {"left": {"max_dq_deg": float(m0[0]), "rms_jerk_deg": float(m0[1])},
                "right": {"max_dq_deg": float(m1[0]), "rms_jerk_deg": float(m1[1])}},
        "limit_viol": {"raw": raw_viol, "after_resolve": resolve_viol},
    })
    for j in joints:
        j.set_drive_property(stiffness=rg.ARM_STIFF, damping=rg.ARM_DAMP)

    # 初始位形: 双臂 frame0 + 夹爪张开, 静置稳定
    def set_targets(k):
        for i in range(6):
            jmap[f"left_arm_joint{i+1}"].set_drive_target(float(armL[k, i]))
            jmap[f"right_arm_joint{i+1}"].set_drive_target(float(armR[k, i]))
        jmap["left_gripper_finger_joint1"].set_drive_target(float(gripL[k, 0]))
        jmap["left_gripper_finger_joint2"].set_drive_target(float(gripL[k, 1]))
        jmap["right_gripper_finger_joint1"].set_drive_target(float(gripR[k, 0]))
        jmap["right_gripper_finger_joint2"].set_drive_target(float(gripR[k, 1]))

    for t in ("torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"):
        if t in jmap:
            jmap[t].set_drive_target(0.0)
    set_targets(0)
    robot.set_root_pose(se2_pose(bx[0], by[0], byaw[0]))
    rg.step_sim(scene, robot, 200)   # 静置到位

    # ---- 多视角相机 (沿用之前管线的观察位: 正面|侧面 / 胸部|俯视) ----
    W, H = 480, 360
    cams = []
    front = scene.add_camera(name="front", width=W, height=H, fovy=np.deg2rad(45), near=0.05, far=100)
    front.entity.set_pose(rg.look_at_pose([1.55, -1.60, 1.55], [0.0, -0.20, 0.95])); cams.append(front)
    side = scene.add_camera(name="side", width=W, height=H, fovy=np.deg2rad(45), near=0.05, far=100)
    side.entity.set_pose(rg.look_at_pose([-1.85, -0.50, 1.55], [0.0, -0.20, 0.95])); cams.append(side)
    # 胸部 ego 视角: 装在躯干前上方(须在躯干网格之外, 否则被自身遮挡), 朝前俯视工作区
    chest = scene.add_camera(name="chest", width=W, height=H, fovy=np.deg2rad(60), near=0.05, far=100)
    top = scene.add_camera(name="top", width=W, height=H, fovy=np.deg2rad(60), near=0.05, far=100)

    def update_ego_cams(dx, dy, yaw=0.0):
        """胸部 ego 相机随底盘动 (随 yaw 一起转, 始终对正工作区); 俯视用世界固定视角。"""
        bxy = np.array([rg.R1_BASE_P[0] + dx, rg.R1_BASE_P[1] + dy])
        Rb = _yaw_R(yaw)[:2, :2]
        head = bxy + Rb @ np.array([0.0, 0.12])
        look = bxy + Rb @ np.array([0.0, 0.54])
        chest.entity.set_pose(rg.look_at_pose([head[0], head[1], 1.46], [look[0], look[1], 0.80]))
        top.entity.set_pose(rg.look_at_pose([0.2, -1.7, 2.1], [0.15, -0.35, 0.35]))  # 固定世界系俯前视
    cams.extend([chest, top])
    update_ego_cams(0.0, 0.0, 0.0)

    def render4(k):
        update_ego_cams(bx[k], by[k], byaw[k])
        scene.update_render()
        for c in cams:
            c.take_picture()
        imgs = [(np.clip(c.get_picture("Color")[:, :, :3], 0, 1) * 255).astype(np.uint8) for c in cams]
        top_row = np.concatenate(imgs[:2], axis=1)   # 正面 | 侧面
        bot_row = np.concatenate(imgs[2:4], axis=1)  # 胸部 | 俯视
        return np.concatenate([top_row, bot_row], axis=0)

    # ---- 逐帧物理回放 + 渲染 (底盘随 base_xy 逐帧移动) ----
    STEPS = 25   # 每控制步仿真步数 (与之前管线 CTRL_STEPS 一致, 10Hz)
    frames = []
    for k in range(N):
        robot.set_root_pose(se2_pose(bx[k], by[k], byaw[k]))   # 运动学移动底盘
        set_targets(k)
        rg.step_sim(scene, robot, STEPS)
        frames.append(render4(k))
        if (k + 1) % 15 == 0:
            print(f"  帧 {k+1}/{N}")

    gif = out_gif(base_mode, use_yaw)
    imageio.mimsave(gif, frames, fps=10, loop=0)
    print(f"\n✓ bp14 [{base_mode}] 双臂完整 R1 仿真 GIF -> {gif} ({len(frames)} 帧)")
    return gif


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="bp14 双臂重定向 → 完整 R1 仿真渲染 (按基座方案独立测试)")
    ap.add_argument("--base", choices=["static", "heuristic", "opt"], default="opt",
                    help="基座方案: static=静态基线 / heuristic=跟踪两手中点 / opt=优化逐帧底盘")
    ap.add_argument("--yaw", action="store_true", help="允许底盘转向 (yaw 自由度, opt 方案有效)")
    a = ap.parse_args()
    main(base_mode=a.base, use_yaw=a.yaw)
