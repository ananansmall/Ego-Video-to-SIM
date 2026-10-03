#!/usr/bin/env python
"""诊断: MANO 手朝向 vs 解算出的夹爪姿态 R 朝向 (两者是不同量, 必须分开看)。

三组独立指标, 不要混淆:
  A. MANO 原始手朝向  — 指尖相对手腕的高度 (MANO 关节, 与 R 无关)
  B. 夹爪 R 朝向      — gripper X 轴 (approach/手指伸出方向) 的世界 z 分量
  C. 夹爪指尖落点     — base + R @ [0.03689,0,0] 相对 base 的高度 (直接验证朝下)

Gripper 几何约定 (physics_utils):
  X 轴 = approach / 手指伸出方向 (手指在 base 前方 3.689cm)
  Y 轴 = 开合方向 (_FINGER1_AXIS = [0,-1,0]);  Z = X × Y
SAPIEN/GLB 世界系: Z 向上, 重力 g = [0,0,-1]。

用法:
    python test/diag_mano_R_orientation.py
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

HAWOR_DIR = "/home/an/data/hawor/7"
TP_PATH = ROOT / "output" / "gripper_right" / "alignment" / "transform_params.npz"
OBJ_Z_PATH = ROOT / "output" / "gripper_right" / "stage0" / "run_obj_z_traj.npy"
SIDE = "right"
HAND_IDX = 1  # 0=左手, 1=右手 (对齐 grasp_hawor.hand_indices)

FINGER_FORWARD = np.array([0.03689, 0.0, 0.0])  # gripper base → 手指中点
GRAVITY = np.array([0.0, 0.0, -1.0])
Z_UP = np.array([0.0, 0.0, 1.0])


def flip_hand_pose_down(joints, wrist_idx=0):
    """把整只 MANO 手绕**水平轴**翻转 180° (以腕为支点), 使掌心朝地/指尖朝下。

    翻转轴 h = normalize(cross(z_up, pointing)):
      - h 水平 (⊥ 重力), 且 ⊥ pointing 的水平投影
      - 绕 h 转 180°: v' = 2(v·h)h - v  → v 的**垂直分量取反**, 沿 h 的分量保持
      - 效果: pointing 的 z 取反 (朝上→朝下), 掌面法向 z 取反 (朝天→朝地),
              手的水平朝向 (pointing 的水平投影方向) 保持不变
    这是**旋转** (det=+1), 不是关于水平面的镜像, 不改变手性。
    """
    j = np.array(joints, dtype=np.float64, copy=True)
    w = j[wrist_idx, :3]
    pointing = ((j[4, :3] + j[8, :3]) / 2.0) - w
    h = np.cross(Z_UP, pointing)
    nl = float(np.linalg.norm(h))
    if nl < 1e-8:
        return j  # pointing 已垂直, 无水平分量可定义翻转轴
    h = h / nl
    rel = j[:, :3] - w
    proj = (rel @ h)[:, None] * h[None, :]
    j[:, :3] = w + (2.0 * proj - rel)
    return j


def main():
    from trajectory_loader import load_hawor_data, load_hawor_c2w, compute_mano_joints
    from data_loader import (compute_analytical_gripper_pose,
                             compute_mano_based_gripper_pose)

    hd = load_hawor_data(HAWOR_DIR, hand_idx=HAND_IDX)
    n = len(hd["pred_trans"])
    R_c2w_all, t_c2w_all = load_hawor_c2w(HAWOR_DIR)

    tp = np.load(str(TP_PATH))
    s = float(tp["scale_ratio"])
    R_h2g = np.asarray(tp["R_hand_to_glb"], dtype=np.float64)
    t_h2g = np.asarray(tp["t_hand_to_glb"], dtype=np.float64).reshape(3)

    obj_z0 = float("nan")
    if OBJ_Z_PATH.exists():
        obj_z0 = float(np.asarray(np.load(str(OBJ_Z_PATH))).ravel()[0])

    from mano_layer import MANOLayer
    ml = MANOLayer(SIDE, hd["pred_betas"][0].astype(np.float32))

    def to_world(pts, fi):
        p = (R_c2w_all[fi] @ np.asarray(pts).T).T + t_c2w_all[fi]
        return s * (R_h2g @ p.T).T + t_h2g

    # 【E】尺度检查: MANO 关节间距是否符合人手量级
    _, j50 = compute_mano_joints(
        ml, hd["pred_rot"][50], hd["pred_hand_pose"][50], hd["pred_trans"][50])
    j50 = to_world(j50, 50)
    _d = lambda a, b: float(np.linalg.norm(j50[a, :3] - j50[b, :3])) * 100
    print("=" * 118)
    print("【E】尺度检查 (SAPIEN 世界系, cm) — MANO 手是否与场景/夹爪同尺度:")
    print(f"  scale_ratio        = {s:.4f}")
    print(f"  掌长   j0→j5(食MCP)= {_d(0,5):6.1f}   (人手 ~9cm)")
    print(f"  食指   j5→j8(食指尖)= {_d(5,8):6.1f}   (人手 ~8cm)")
    print(f"  拇指   j0→j4(拇指尖)= {_d(0,4):6.1f}   (人手 ~12cm)")
    print(f"  手总长 j0→j8       = {_d(0,8):6.1f}   (人手 ~18cm)")
    print(f"  腕→指尖中点 |p|    = {np.linalg.norm((j50[4,:3]+j50[8,:3])/2 - j50[0,:3])*100:6.1f}   (人手 ~14cm)")
    print(f"  参考: 夹爪手指前伸 FINGER_FORWARD_NEUTRAL = 3.689cm, 手指半间距 = 2.0cm")

    print("=" * 118)
    print(f"物体初始 z = {obj_z0:.4f} m (stage0/run_obj_z_traj.npy 首帧)")
    print("=" * 118)
    print("【A】MANO 原始手朝向          【B】夹爪 R 朝向 (解算结果)         【C】夹爪指尖落点")
    print(f"{'F':>4} | {'腕z':>7} {'指尖z':>7} {'掌法z':>7} | "
          f"{'X.z':>7} {'倾角°':>7} {'点积':>7} | {'指尖-base(cm)':>13} {'base_z':>7}")
    print("-" * 118)

    rows = []
    for fi in [f for f in (0, 20, 40, 50, 60, 80, 100) if f < n]:
        _, joints = compute_mano_joints(
            ml, hd["pred_rot"][fi], hd["pred_hand_pose"][fi], hd["pred_trans"][fi])
        js = to_world(joints, fi)
        pos_a, Ra, _, _ = compute_analytical_gripper_pose(
            js[0, :3], js[4, :3], js[8, :3], prefix=SIDE)

        # 【A】MANO 原始手
        wrist_z = js[0, 2]
        tip_mid_z = ((js[4, :3] + js[8, :3]) / 2)[2]
        # 掌面法向: cross(食指MCP-腕, 小指MCP-腕) — MANO j5=食指MCP, j17=小指MCP
        pnorm = np.cross(js[5, :3] - js[0, :3], js[17, :3] - js[0, :3])
        pnorm = pnorm / max(np.linalg.norm(pnorm), 1e-8)

        # 【B】夹爪 R: X 轴与重力点积 ( >0 = 朝下)
        x_a = Ra[:, 0]
        dot_g = float(np.dot(x_a, GRAVITY))
        tilt = float(np.degrees(np.arccos(np.clip(dot_g, -1.0, 1.0))))  # 与重力夹角

        # 【C】夹爪指尖实际落点 (base + R @ [0.03689,0,0])
        tip_world = pos_a + Ra @ FINGER_FORWARD
        dh_tip = (tip_world[2] - pos_a[2]) * 100

        # 【D】对齐误差: R 的 X 轴 是否忠实于 MANO 指向方向 pointing
        pointing = ((js[4, :3] + js[8, :3]) / 2) - js[0, :3]
        p_len = float(np.linalg.norm(pointing))
        p_unit = pointing / max(p_len, 1e-8)
        ang_x = float(np.degrees(np.arccos(np.clip(np.dot(Ra[:, 0], p_unit), -1.0, 1.0))))

        print(f"{fi:>4} | {wrist_z:>7.3f} {tip_mid_z:>7.3f} {pnorm[2]:>7.3f} | "
              f"{Ra[2,0]:>7.3f} {tilt:>7.1f} {dot_g:>7.3f} | {dh_tip:>13.2f} {pos_a[2]:>7.3f}")
        print(f"{'':>4} |  pointing: |p|={p_len*100:5.1f}cm  单位z={p_unit[2]:+.3f}  "
              f"→ X轴偏离pointing {ang_x:5.1f}°  (加权SVD W_Y=5 让开合方向优先, approach 被牺牲)")
        rows.append((Ra[2, 0], dot_g, tilt, dh_tip, tip_mid_z - wrist_z))

    # 全序列统计
    xz_a, dot_a, tilt_a, dhtip_a, dhmano_a, xz_s = [], [], [], [], [], []
    for fi in range(n):
        try:
            _, joints = compute_mano_joints(
                ml, hd["pred_rot"][fi], hd["pred_hand_pose"][fi], hd["pred_trans"][fi])
        except Exception:
            continue
        js = to_world(joints, fi)
        pos_a, Ra, _, _ = compute_analytical_gripper_pose(
            js[0, :3], js[4, :3], js[8, :3], prefix=SIDE)
        _, Rs, _, _ = compute_mano_based_gripper_pose(js, prefix=SIDE)
        dot = float(np.dot(Ra[:, 0], GRAVITY))
        xz_a.append(Ra[2, 0])
        dot_a.append(dot)
        tilt_a.append(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
        dhtip_a.append(((pos_a + Ra @ FINGER_FORWARD)[2] - pos_a[2]) * 100)
        dhmano_a.append(((js[4, :3] + js[8, :3]) / 2 - js[0, :3])[2] * 100)
        xz_s.append(Rs[2, 0])

    xz_a, dot_a = np.asarray(xz_a), np.asarray(dot_a)
    tilt_a = np.asarray(tilt_a)
    dhtip_a, dhmano_a, xz_s = np.asarray(dhtip_a), np.asarray(dhmano_a), np.asarray(xz_s)

    print("=" * 118)
    print(f"全序列 N={len(xz_a)}")
    print()
    print("【B】夹爪 R 朝向 (解算结果 — 这才是 R 翻转是否生效的判据):")
    print(f"  X.z        : mean={xz_a.mean():+.3f} min={xz_a.min():+.3f} max={xz_a.max():+.3f} "
          f"| 朝下(<0) 占比 {np.mean(xz_a < 0) * 100:5.1f}%")
    print(f"  dot(X, g)  : mean={dot_a.mean():+.3f} min={dot_a.min():+.3f} max={dot_a.max():+.3f} "
          f"| 为正(朝下) 占比 {np.mean(dot_a > 0) * 100:5.1f}%")
    print(f"  与重力夹角 : mean={tilt_a.mean():.1f}°  (0°=完全垂直朝下, 90°=水平)")
    print(f"  svd_palm X.z: mean={xz_s.mean():+.3f} | 朝下占比 {np.mean(xz_s < 0) * 100:5.1f}%")
    print()
    print("【C】夹爪指尖落点 (直接几何验证):")
    print(f"  指尖 - base 高度: mean={dhtip_a.mean():+.2f}cm "
          f"(负=手指朝下伸出; 完全垂直时 = -3.69cm)")
    print()
    print("【A】MANO 原始手朝向 (与 R 无关, 反映变换链是否整体翻转):")
    print(f"  指尖 - 手腕 高度: mean={dhmano_a.mean():+.2f}cm "
          f"(人手伸向桌面物体时应为负 = 指尖低于手腕)")
    print("=" * 118)

    # 【F】方案对比: 仅翻转夹爪 R (当前 v6.9 下游补丁) vs 翻转 MANO 手姿态 (上游修正)
    print()
    print("=" * 118)
    print("【F】方案对比 — 上游翻转 MANO 手姿态 vs 当前仅翻转夹爪 R")
    print(f"  物体 z = {obj_z0:.3f} m; 指尖越接近该高度越合理")
    print(f"{'F':>4} | {'当前(不翻转MANO): 掌法z':>22} {'指尖-腕':>9} {'point.z':>8} {'指尖z':>7} | "
          f"{'翻转MANO后: 掌法z':>18} {'指尖-腕':>9} {'point.z':>8} {'指尖z':>7} {'解算X.z':>8} {'点积':>7}")
    print("-" * 118)

    def _palm_z(js):
        n_ = np.cross(js[5, :3] - js[0, :3], js[17, :3] - js[0, :3])
        return float((n_ / max(np.linalg.norm(n_), 1e-8))[2])

    for fi in [f for f in (0, 25, 50, 75, 100) if f < n]:
        _, joints = compute_mano_joints(
            ml, hd["pred_rot"][fi], hd["pred_hand_pose"][fi], hd["pred_trans"][fi])
        js = to_world(joints, fi)
        js_f = flip_hand_pose_down(js)  # 在世界系翻转 (翻转轴依赖世界 z_up)

        p0 = ((js[4, :3] + js[8, :3]) / 2) - js[0, :3]
        p0u = p0 / max(np.linalg.norm(p0), 1e-8)
        p1 = ((js_f[4, :3] + js_f[8, :3]) / 2) - js_f[0, :3]
        p1u = p1 / max(np.linalg.norm(p1), 1e-8)

        _, Rf, _, _ = compute_analytical_gripper_pose(
            js_f[0, :3], js_f[4, :3], js_f[8, :3], prefix=SIDE)
        dot_f = float(np.dot(Rf[:, 0], GRAVITY))

        print(f"{fi:>4} | {_palm_z(js):>22.3f} {((js[4,:3]+js[8,:3])/2 - js[0,:3])[2]*100:>9.2f} "
              f"{p0u[2]:>8.3f} {((js[4,:3]+js[8,:3])/2)[2]:>7.3f} | "
              f"{_palm_z(js_f):>18.3f} {((js_f[4,:3]+js_f[8,:3])/2 - js_f[0,:3])[2]*100:>9.2f} "
              f"{p1u[2]:>8.3f} {((js_f[4,:3]+js_f[8,:3])/2)[2]:>7.3f} {Rf[2,0]:>8.3f} {dot_f:>7.3f}")
    print("=" * 118)
    print("判读: 翻转后 掌法z<0 (掌心朝地) 且 指尖-腕<0 (指尖朝下) 且 解算点积>0 = 上游修正可行")
    print("=" * 118)


if __name__ == "__main__":
    main()
