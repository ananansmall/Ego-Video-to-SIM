"""MANO 轨迹离线平滑 (P0-2): 移植自 05_gripper_test.py 的四重平滑前三重.

① Savitzky-Golay 前置滤波 (位置/关节)
② 高斯加权四元数 Slerp (朝向, Markley 方法)
③ min-jerk 5 阶多项式轨迹约束 (位置三轴; 关节用 1/20 约束)

第④重 EMA 为命令级在线平滑, 由调用方主循环执行 (本 pass 不接入).
默认 max_vel=0.020 m/帧, max_acc=0.005 m/帧² (05 test9 调优值).
"""
import logging

import numpy as np
from scipy.signal import savgol_filter
import scipy.spatial.transform as sst  # sst.Rotation (匹配 05_gripper_test.py 约定)

logger = logging.getLogger("grasp_hawor")


def _gaussian_quat_smooth(quats_wxyz, sigma=2.0, window_half=5):
    """高斯加权四元数平均 (Markley 方法)"""
    n = len(quats_wxyz)
    smoothed = np.zeros_like(quats_wxyz)
    for i in range(n):
        start = max(0, i - window_half)
        end = min(n, i + window_half + 1)
        idx = np.arange(start, end)
        w = np.exp(-0.5 * ((idx - i) / sigma) ** 2)
        w = w / w.sum()
        M = np.zeros((4, 4), dtype=np.float64)
        for jj, ww in zip(idx, w):
            q = quats_wxyz[jj].astype(np.float64)
            M += ww * np.outer(q, q)
        eigvals, eigvecs = np.linalg.eigh(M)
        smoothed[i] = eigvecs[:, np.argmax(eigvals)]
    for i in range(1, n):
        if np.dot(smoothed[i], smoothed[i - 1]) < 0:
            smoothed[i] = -smoothed[i]
    return smoothed


def _apply_smooth_sg(raw_pos, raw_quat, raw_joint, raw_valid):
    """Savitzky-Golay + 高斯Slerp 平滑

    调优 (2026-08): sg_window 5→17, sg_order 2 保持;
    朝向 sigma 2→5, window_half 5→9, 抑制 MANO 逐帧噪声导致的晃动.
    """
    sg_window = 17
    sg_order = 2
    smooth_pos = raw_pos.copy()
    smooth_joint = raw_joint.copy()
    smooth_quat = raw_quat.copy()
    valid_idx = np.where(raw_valid)[0]
    if len(valid_idx) > sg_window:
        inner = valid_idx[sg_window // 2: -(sg_window // 2)]
        if len(inner) > sg_window:
            smooth_pos[inner] = savgol_filter(raw_pos[inner], sg_window, sg_order, axis=0, mode='mirror')
            smooth_joint[inner] = savgol_filter(raw_joint[inner], sg_window, sg_order, mode='mirror')
    smooth_joint = np.clip(smooth_joint, 0.0, 0.05)
    if len(valid_idx) > 5:
        quat_inner = _gaussian_quat_smooth(raw_quat[valid_idx], sigma=5.0, window_half=9)
        norms = np.linalg.norm(quat_inner, axis=1)
        bad = norms < 1e-6
        quat_inner = quat_inner / np.maximum(norms, 1e-10)[:, np.newaxis]
        if bad.any():
            quat_inner[bad] = raw_quat[valid_idx][bad]
        smooth_quat[valid_idx] = quat_inner
    return smooth_pos, smooth_quat, smooth_joint


def min_jerk_smooth(data, max_vel, max_acc):
    """对一维轨迹做 minimum jerk 平滑, 约束速度和加速度"""
    n = len(data)
    result = data.copy()

    # 检测跳变点, 标记为关键点
    max_jump = max_vel * 2  # 超过2倍最大速度的跳变
    key_points = [0]
    for i in range(1, n):
        if abs(data[i] - data[i-1]) > max_jump:
            key_points.append(i)
    if key_points[-1] != n - 1:
        key_points.append(n - 1)

    # 对每段关键点之间做 minimum jerk 插值
    for seg in range(len(key_points) - 1):
        i0, i1 = key_points[seg], key_points[seg + 1]
        if i1 - i0 < 2:
            continue

        # 5阶多项式: p(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        # 边界条件: p(0)=p0, p(T)=p1, p'(0)=v0, p'(T)=v1, p''(0)=0, p''(T)=0
        T = i1 - i0
        p0, p1 = data[i0], data[i1]

        # 计算端点速度 (用相邻点差分, 受速度约束)
        v0 = 0.0
        if i0 > 0:
            v0 = np.clip(data[i0] - data[i0-1], -max_vel, max_vel)
        v1 = 0.0
        if i1 < n - 1:
            v1 = np.clip(data[i1+1] - data[i1], -max_vel, max_vel)

        # 5阶多项式系数 (最小 jerk, 零端点加速度)
        # 解线性方程组
        a0 = p0
        a1 = v0
        a2 = 0.0  # 零初始加速度
        # 从 p(T)=p1, p'(T)=v1, p''(T)=0 解 a3, a4, a5
        # p(T) = a0 + a1*T + a3*T³ + a4*T⁴ + a5*T⁵ = p1
        # p'(T) = a1 + 3*a3*T² + 4*a4*T³ + 5*a5*T⁴ = v1
        # p''(T) = 6*a3*T + 12*a4*T² + 20*a5*T³ = 0
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        b = np.array([
            p1 - a0 - a1*T,
            v1 - a1,
            0.0
        ])
        try:
            coeffs = np.linalg.solve(A, b)
            a3, a4, a5 = coeffs
        except np.linalg.LinAlgError:
            continue

        # 生成插值轨迹
        for j in range(i0, i1 + 1):
            t = j - i0
            result[j] = a0 + a1*t + a2*t*t + a3*t**3 + a4*t**4 + a5*t**5

        # 验证速度/加速度约束, 如果违反则用线性插值替代
        valid = True
        for j in range(i0 + 1, i1 + 1):
            vel = result[j] - result[j-1]
            if abs(vel) > max_vel * 1.5:  # 允许1.5倍余量
                valid = False
                break
        if not valid:
            # 退化为线性插值 + 端点速度约束
            for j in range(i0, i1 + 1):
                alpha = (j - i0) / T
                result[j] = p0 * (1 - alpha) + p1 * alpha

    return result


def quat_from_matrix(R):
    """旋转矩阵 → 四元数 (w,x,y,z)"""
    return sst.Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def matrix_from_quat(q):
    """四元数 (w,x,y,z) → 旋转矩阵"""
    return sst.Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def smooth_mano_trajectory(raw_pos, raw_quat, raw_joint, raw_valid,
                           max_vel=0.020, max_acc=0.005):
    """统一平滑入口 — 四重平滑的前三重 (离线批量):
      ① Savitzky-Golay 前置滤波 (位置/关节)
      ② 高斯加权四元数 Slerp (朝向)
      ③ min-jerk 5阶多项式轨迹约束 (位置三轴; 关节用 1/20 的约束)
    第④重 EMA 为命令级在线平滑, 由调用方主循环执行
    (test9: EmaTargetSmoother + rotvec EMA + 关节标量 EMA).

    max_vel/max_acc: min-jerk 速度/加速度约束 (m/帧),
    默认为 test9 调优值 20mm/帧, 5mm/帧².
    """
    smooth_pos, smooth_quat, smooth_joint = _apply_smooth_sg(raw_pos, raw_quat, raw_joint, raw_valid)
    for d in range(3):
        smooth_pos[:, d] = min_jerk_smooth(smooth_pos[:, d], max_vel, max_acc)
    smooth_joint = min_jerk_smooth(smooth_joint, max_vel * 0.05, max_acc * 0.05)
    smooth_joint = np.clip(smooth_joint, 0.0, 0.05)
    return smooth_pos, smooth_quat, smooth_joint


def smooth_gripper_traj(traj_dict, side, max_vel=0.020, max_acc=0.005):
    """对 _mano_gripper_traj[side] 做离线平滑 (前三重), 原地改写 pos/R/j1/j2.

    R→quat(w,x,y,z)→平滑→R; j1,j2→joint=(j1+j2)/2→平滑→拆回 j1=j2.
    raw_valid: precompute 已过滤无效帧, 此处全 True.
    返回 True 表示已平滑, False 表示轨迹太短/缺失跳过.
    """
    t = traj_dict.get(side)
    if t is None or len(t.get('pos', [])) == 0:
        return False
    pos = np.asarray(t['pos'], dtype=np.float64)
    R_traj = np.asarray(t['R'], dtype=np.float64)
    j1 = np.asarray(t.get('j1', []), dtype=np.float64)
    j2 = np.asarray(t.get('j2', []), dtype=np.float64)
    N = len(pos)
    if N < 3:
        return False
    # ★★★ 2026-09-13 (用户定位 "开头反向移动 / F22 停顿 / 末尾停顿" 的真凶后接线):
    #   先做**姿态连续性修正**, 再做 quat 平滑.
    #   背景: physics_utils.unflip_rotation_traj_continuity 的文档记录了 180° 台阶 ——
    #     align_gripper_approach_to_gravity 用 `dot(X,g)<0` 逐帧硬阈值, 当 approach 轴
    #     接近水平时符号来回跳, 姿态序列出现真实翻转 (7a/左手实测:
    #     F7->F8 179.5° | F9->F10 179.1° | F13->F14 178.8° | F108->F109 179.3°,
    #     而全序列平均帧间变化仅 7.63°).
    #   危害: 1 个控制周期(33ms)内命令翻转 180°, PD 跟不住 → 基座姿态滞后 11~48°、
    #     角速度 589°/s → 手指(伸出 7.74cm)扫偏厘米级 → 接近段撞物体 → 手指被顶到
    #     闭合极限 → 全程接触.  这既是"开头反向移动"(F7-F14), 也是"末尾长停顿"(F108).
    #   **该修复函数此前已写好但全仓无任何调用点(死代码)** — 这里接线.
    #   顺序要求: 必须在 quat 平滑**之前**做, 否则 180° 台阶会被平滑器抹成中间乱转.
    _n_unflip = 0
    try:
        from physics_utils import unflip_rotation_traj_continuity

        R_traj, _n_unflip = unflip_rotation_traj_continuity(R_traj)
        if _n_unflip:
            logger.info(
                "[朝向连续性] 翻回 %d/%d 帧 (消除 align_gripper_approach_to_gravity "
                "逐帧硬阈值造成的 180° 姿态台阶)", _n_unflip, N)
    except Exception as _exc:  # noqa: BLE001
        logger.warning("[朝向连续性] unflip 失败, 保持原样: %s", _exc)
    # R → quat (w,x,y,z)
    quat = np.array([quat_from_matrix(R_traj[i]) for i in range(N)], dtype=np.float64)
    joint = (j1 + j2) / 2.0 if (len(j1) == N and len(j2) == N) else np.zeros(N)
    raw_valid = np.ones(N, dtype=bool)
    sp, sq, sj = smooth_mano_trajectory(pos, quat, joint, raw_valid, max_vel, max_acc)
    # quat → R
    new_R = np.array([matrix_from_quat(sq[i]) for i in range(N)], dtype=np.float64)
    t['pos'] = sp
    t['R'] = new_R
    t['j1'] = sj
    t['j2'] = sj
    return True


def smooth_joints_seq(joints, max_vel=0.020, max_acc=0.005):
    """v51 §1 (2026-09-05): 重定向前平滑 — 对 MANO 关节序列 (N,21,3) 做 ①SG + ③min-jerk.

    顺序依据 §1 设计 "坐标变换 → 平滑 → 解算夹爪" (用户 2026-09-05 确认: 先平滑后重定向;
    旧实现在重定向后才平滑夹爪轨迹, 现已移到关节层).
    ② 高斯四元数 Slerp 在此不适用 — 夹爪 R 由平滑关节经 SVD 派生, 天然平滑.
    SG 窗口/阶数与 _apply_smooth_sg 一致 (17/2, mirror).
    2026-09-06 (用户: "不能太过于平滑, 夹爪能跟随渲染即可"): **移除 min-jerk** —
    其实现将无跳变长段整段替换为五次多项式, 形状失真实测远离原始值 (平滑前后偏差
    达 14cm, 见 CHANGE_LOG 2026-09-05); SG 为局部滤波器, 只去抖不改形状, 平滑后
    贴合原始值. 与 v51 §1.4 ③ 的偏离已记录 (待用户确认后修订计划).
    """
    N = len(joints)
    if N < 21:
        return joints.copy()
    flat = joints.reshape(N, -1).copy()
    sg_window, sg_order = 17, 2
    if N > sg_window:
        flat[sg_window // 2: -(sg_window // 2)] = savgol_filter(
            flat[sg_window // 2: -(sg_window // 2)], sg_window, sg_order, axis=0, mode='mirror')
    # 2026-09-06: min-jerk 移除 (形状失真, 见上注) — 仅 SG 去抖
    return flat.reshape(joints.shape)
