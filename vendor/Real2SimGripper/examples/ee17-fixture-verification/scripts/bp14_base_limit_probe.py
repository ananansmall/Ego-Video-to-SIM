#!/usr/bin/env python3
"""bp14_base_limit_probe.py — 验证「移动基座能否补偿左臂关节限位」。

物理模型: 真实人手世界轨迹固定 (数据给定)。基座是自由量 —— 基座挪位后,
同一条世界系末端目标所需的机械臂位形改变, 从而可能避开关节限位。

做法:
  1. 用重定向左臂关节 + 基座 B0 做 FK, 得到 61 帧"世界系末端目标"(pos + 接近轴
     + 开合轴), 这是一条固定的任务轨迹。
  2. 对候选基座位姿 (x,y,yaw), 逐帧用快速 DLS-IK 重解左臂关节, 记录:
       - 位置残差 (mm): DLS 收敛后 pad 与目标距离
       - 越限帧数: 有任一关节超出 IK_LIMITS 的帧数
  3. 对比静态 B0 与若干基座偏移, 看能否把越限降到 0 / 残差降下来。
只读/只算, 不改主脚本。
"""
import os, sys, numpy as np
import sapien.core as sapien
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bp14_dual_arm_sim as bp   # noqa: E402
import robotwin_r1_duck_grasp as rg  # noqa: E402


def yaw_q(y):
    return np.array([np.cos(y / 2), 0, 0, np.sin(y / 2)])


def main():
    qL = np.load(os.path.join(bp.QPOS_DIR, "hawor_r1_dual_tracking_left.npy"))
    N = qL.shape[0]
    mL, _ = bp._pick_convention(qL[:, :6], "left")
    # 原始(未裁剪)重定向关节: 用于生成"世界系任务目标", 才不会被裁剪掩盖限位问题
    armL_raw = np.stack([qL[i, :6] * mL for i in range(N)])
    armL_raw = rg.smooth_traj(armL_raw, 5, 2)

    scene, robot = bp.build_scene_and_robot()
    joints = robot.get_active_joints()
    nm2i = {j.get_name(): k for k, j in enumerate(joints)}
    Lsel = [nm2i[f"left_arm_joint{i+1}"] for i in range(6)]
    Lg = robot.find_link_by_name("left_gripper_link")
    PAD = rg.PAD_FWD

    def set_root(base_p, yaw):
        robot.set_root_pose(sapien.Pose(p=np.array([base_p[0], base_p[1], 0.0], np.float32),
                                        q=yaw_q(yaw).astype(np.float32)))

    def fk_pad(q6):
        q = robot.get_qpos().copy()
        for k, si in enumerate(Lsel):
            q[si] = q6[k]
        robot.set_qpos(q)
        M = Lg.get_pose().to_transformation_matrix()
        return M[:3, 3] + M[:3, 0] * PAD, M[:3, :3]

    # 1) 世界系任务目标 (原始重定向关节 @ 基座 B0 的 FK)
    set_root(rg.R1_BASE_P, 0.0)
    targets = []
    for k in range(N):
        p, R = fk_pad(armL_raw[k])
        targets.append((p.copy(), R[:, 0].copy(), R[:, 1].copy()))

    lo, hi = rg.IK_LIMITS[:, 0], rg.IK_LIMITS[:, 1]

    # 原始重定向关节本身的越限统计
    seed_viol = int(np.sum(np.any((armL_raw < lo - 1e-6) | (armL_raw > hi + 1e-6), axis=1)))
    print(f"  原始重定向关节越限帧 = {seed_viol}/{N} (这些帧被裁剪 => 位置误差来源)")

    rng = np.random.default_rng(0)

    def pose_err(q6, pos_t, app, fin):
        p, R = fk_pad(q6)
        a = R[:, 0]; f = R[:, 1]
        e_p = pos_t - p
        e_o = np.cross(a, app) + np.cross(f, fin)     # 小角度朝向误差
        return np.concatenate([e_p, 0.3 * e_o])

    def dls_ik(pos_t, app, fin, seed, iters=100):
        """限位内 6D DLS (位置+朝向): 全程 clip 到关节限位; 返回 (q6, 位置残差, 朝向残差deg)。"""
        q6 = np.clip(seed.copy(), lo, hi)
        for _ in range(iters):
            e = pose_err(q6, pos_t, app, fin)
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 5e-3:
                break
            J = np.zeros((6, 6))
            for k in range(6):
                dq = np.zeros(6); dq[k] = 1e-3
                J[:, k] = (pose_err(np.clip(q6 + dq, lo, hi), pos_t, app, fin) - e) / 1e-3
            q6 = np.clip(q6 + J.T @ np.linalg.solve(J @ J.T + 0.02 * np.eye(6), np.clip(e, -0.05, 0.05)), lo, hi)
        e = pose_err(q6, pos_t, app, fin)
        return q6, float(np.linalg.norm(e[:3])), float(np.degrees(np.linalg.norm(e[3:])))

    def evaluate(base_p, yaw, tag):
        set_root(base_p, yaw)
        rp, ro = [], []
        for k in range(N):
            p, app, fin = targets[k]
            seeds = [np.clip(armL_raw[k], lo, hi)] + [rng.uniform(lo, hi) for _ in range(6)]
            best = min((dls_ik(p, app, fin, s) for s in seeds), key=lambda t: t[1] + t[2] * 2e-4)
            rp.append(best[1]); ro.append(best[2])
        rp = np.array(rp) * 1000; ro = np.array(ro)
        viol = int(np.sum(rp > 10.0))
        print(f"  [{tag:16s}] base=({base_p[0]:+.2f},{base_p[1]:+.2f}) yaw={np.degrees(yaw):+5.0f}°  "
              f"位残差 RMS={rp.mean():6.1f}mm max={rp.max():7.1f}mm  "
              f"姿残差 max={ro.max():5.1f}°  不可达帧={viol}/{N}")
        return rp.mean(), viol

    print("=== 基座扫描 (左臂 61 帧, 限位内 6D 可达性) ===")
    evaluate(rg.R1_BASE_P, 0.0, "B0 静态")
    for dx in (-0.15, -0.05, 0.05, 0.15):
        evaluate([rg.R1_BASE_P[0] + dx, rg.R1_BASE_P[1]], 0.0, f"B0+dx{dx:+.2f}")
    for dy in (-0.10, 0.10):
        evaluate([rg.R1_BASE_P[0], rg.R1_BASE_P[1] + dy], 0.0, f"B0+dy{dy:+.2f}")
    for yaw in (np.deg2rad(-20), np.deg2rad(20)):
        evaluate(rg.R1_BASE_P, yaw, f"B0+yaw{np.degrees(yaw):+.0f}")
    print("\n注: 残差大=限位内不可达; 不可达帧数下降说明基座补偿有效。")


if __name__ == "__main__":
    main()
