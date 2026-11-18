#!/usr/bin/env python3
"""bp14_err_probe.py — 量化 bp14 双臂回放的误差。

测三类误差:
  A. PD 跟踪误差: 指令关节目标 vs 实际 qpos (逐帧), 每关节 RMS/最大 (deg)。
  B. 末端位置误差: 指令关节 FK 的 gripper 位置 vs 实际 EE 位置 (mm)。
  C. 重定向残差: 实际 EE 轨迹 vs MANO 手腕轨迹 (需同一坐标系; 无场景 => 仅报
     两个手之间的相对距离一致性, 供参考)。
只读/只算, 不修改主脚本。
"""
import os, sys, numpy as np
import sapien.core as sapien
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import bp14_dual_arm_sim as bp   # noqa: E402
import robotwin_r1_duck_grasp as rg  # noqa: E402

def fk_pos(robot, joint_names, q_map, link_name):
    """给定本机器人当前 qpos 设定 joint, 返回 link 世界位置 (不改物理, 仅 FK 近似用 get_link_pose 当前)。"""
    pass

def main():
    qL = np.load(os.path.join(bp.QPOS_DIR, "hawor_r1_dual_tracking_left.npy"))
    qR = np.load(os.path.join(bp.QPOS_DIR, "hawor_r1_dual_tracking_right.npy"))
    N = min(len(qL), len(qR))
    mL, _ = bp._pick_convention(qL[:, :6], "left"); mR, _ = bp._pick_convention(qR[:, :6], "right")
    armL = rg.smooth_traj(np.stack([bp._map_arm(qL[i,:6],"left",mL) for i in range(N)]), 5, 2)
    armR = rg.smooth_traj(np.stack([bp._map_arm(qR[i,:6],"right",mR) for i in range(N)]), 5, 2)

    scene, robot = bp.build_scene_and_robot()
    joints = robot.get_active_joints(); jmap = {j.get_name(): j for j in joints}
    for j in joints: j.set_drive_property(stiffness=rg.ARM_STIFF, damping=rg.ARM_DAMP)

    def set_targets(k):
        for i in range(6):
            jmap[f"left_arm_joint{i+1}"].set_drive_target(float(armL[k,i]))
            jmap[f"right_arm_joint{i+1}"].set_drive_target(float(armR[k,i]))
    for t in ("torso_joint1","torso_joint2","torso_joint3","torso_joint4"):
        if t in jmap: jmap[t].set_drive_target(0.0)
    set_targets(0); rg.step_sim(scene, robot, 200)

    Lg = robot.find_link_by_name("left_gripper_link"); Rg = robot.find_link_by_name("right_gripper_link")
    def qpos_of(side):
        return np.array([robot.get_qpos()[{j.get_name():i for i,j in enumerate(joints)}[f"{side}_arm_joint{k+1}"]] for k in range(6)])

    STEPS = 25
    errJ = {"left": [], "right": []}; errP = {"left": [], "right": []}
    ee = {"left": [], "right": []}
    for k in range(N):
        set_targets(k); rg.step_sim(scene, robot, STEPS)
        qa_L = qpos_of("left"); qa_R = qpos_of("right")
        errJ["left"].append(qa_L - armL[k]); errJ["right"].append(qa_R - armR[k])
        # 无限小 FK: 用 set_qpos 指令值 -> get_link_pose (不改物理, 仅刷新渲染/模型)
        saveQ = robot.get_qpos().copy()
        qv = saveQ.copy()
        idxmap = {j.get_name(): i for i, j in enumerate(joints)}
        for i in range(6):
            qv[idxmap[f"left_arm_joint{i+1}"]] = armL[k,i]
            qv[idxmap[f"right_arm_joint{i+1}"]] = armR[k,i]
        robot.set_qpos(qv); scene.update_render()
        pl = Lg.get_pose().p.copy(); pr_ = Rg.get_pose().p.copy()
        robot.set_qpos(saveQ)
        ee["left"].append(pl); ee["right"].append(pr_)
        # 实际 EE (恢复后)
        pl_a = Lg.get_pose().p.copy(); pr_a = Rg.get_pose().p.copy()
        errP["left"].append(pl_a - pl); errP["right"].append(pr_a - pr_)

    for side in ("left","right"):
        eJ = np.array(errJ[side]); eP = np.array(errP[side])
        print(f"=== {side} ===")
        print(f"  PD 关节跟踪误差 (deg): RMS={np.degrees(np.sqrt((eJ**2).mean())):.3f}  "
              f"max={np.degrees(np.abs(eJ).max()):.3f}")
        print(f"  末端位置误差 (mm): RMS={np.linalg.norm(eP,axis=1).mean()*1000:.2f}  "
              f"max={np.linalg.norm(eP,axis=1).max()*1000:.2f}")
        E = np.array(ee[side])
        print(f"  EE 轨迹范围 (m): x{np.ptp(E[:,0]):.3f} y{np.ptp(E[:,1]):.3f} z{np.ptp(E[:,2]):.3f}")
    # 双手间距 (相对一致性)
    dl = np.array(ee["left"]); dr = np.array(ee["right"])
    dist = np.linalg.norm(dl-dr, axis=1)
    print(f"\n双手 gripper 间距 (m): mean={dist.mean():.3f} min={dist.min():.3f} max={dist.max():.3f}")

if __name__ == "__main__":
    main()
