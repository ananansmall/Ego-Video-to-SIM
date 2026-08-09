"""stage2_reconstruct_mixin.py — Stage 2: 轨迹重建 (Minimum Jerk 平滑插值)

从 grasp_hawor.py 抽出, 提供 Stage2Mixin 类。
Stage 2: 基于 Stage 1 的最优 grasp_pose, 重建 F38-F62 关键帧过渡轨迹
依赖主类 GraspSimulator 的以下属性 (由主类初始化):
  - self._mano_gripper_traj, self._mano_neutral_offset
  - self._frame_params (可选)
"""
import logging
import numpy as np

logger = logging.getLogger("grasp_hawor")


class Stage2Mixin:
    """Stage 2: 轨迹重建 Mixin"""

    # === reconstruct_trajectory (L5869) ===
    def reconstruct_trajectory(self, grasp_pose, side, F50_IDX=50, N_TRANS=4):
        """Stage 2: Minimum Jerk 轨迹重建 + MANO x,y 位置参考 (F44~F56)

        v4.6 核心设计:
        - F46~F49 (接近/边走边夹): 位置从 MANO+offset 插值到 F50,
          姿态保持 MANO (微调到 F50), 夹爪从 MANO 开合度逐渐闭合到 F50
        - F50 (抓取点): 位置/姿态/夹爪 = Stage 1 优化值 (with MANO x,y 参考)
        - F51~F54 (夹持保持): 位置从 F50 插值到 MANO+offset,
          姿态保持 F50 (保证夹持对齐), 夹爪保持闭合 (维持夹持力!)
        """
        from scipy.spatial.transform import Rotation as R_scipy
        from scipy.spatial.transform import Slerp

        traj = self._mano_gripper_traj.get(side)
        if traj is None or len(traj["pos"]) == 0:
            logger.error("[Stage 2] 无 MANO 轨迹")
            return None

        pos_traj_raw = np.asarray(traj["pos"])
        R_traj = np.asarray(traj["R"])
        j1 = np.asarray(traj["j1"]) if "j1" in traj else None
        j2 = np.asarray(traj["j2"]) if "j2" in traj else None
        if j1 is not None and j2 is not None:
            gripper_traj = (j1 + j2) / 2.0
        else:
            gripper_traj = np.zeros(len(pos_traj_raw))

        # ===== 0. 坐标空间对齐: MANO 原始轨迹 + offset =====
        offset = getattr(self, '_mano_neutral_offset', {}).get(side)
        if offset is not None:
            pos_traj = pos_traj_raw + offset
            logger.info(f"[Stage 2] MANO 轨迹已加 offset={np.asarray(offset).round(4)}")
        else:
            pos_traj = pos_traj_raw.copy()
            logger.warning("[Stage 2] 无 _mano_neutral_offset, 使用原始 MANO 轨迹")

        N = len(pos_traj)
        f_start = max(0, min(N - 1, F50_IDX - N_TRANS))
        f_end = max(0, min(N - 1, F50_IDX + N_TRANS))
        F50_IDX = max(0, min(N - 1, F50_IDX))

        def _finite_diff(arr, idx):
            if idx <= 0 or idx + 1 >= len(arr):
                return 0.0 if np.isscalar(arr[idx]) else np.zeros_like(arr[idx])
            return (arr[idx + 1] - arr[idx - 1]) / 2.0

        def _min_jerk_quintic(p0, v0, p1, v1, T, t):
            """Minimum Jerk 五次插值, 边界位置/速度匹配, 边界加速度为 0."""
            if T <= 1e-12:
                return np.asarray(p1, dtype=np.float64)
            T2, T3, T4, T5 = T * T, T ** 3, T ** 4, T ** 5
            dp = np.asarray(p1 - p0 - v0 * T, dtype=np.float64)
            dv = np.asarray(v1 - v0, dtype=np.float64)
            A = np.array([[T3, T4, T5],
                          [3 * T2, 4 * T3, 5 * T4],
                          [6 * T, 12 * T2, 20 * T3]], dtype=np.float64)
            b = np.stack([dp, dv, np.zeros_like(dp)], axis=-1)
            c = np.linalg.solve(A, b.T)
            c3, c4, c5 = c[0], c[1], c[2]
            tt = np.array([t, t * t, t ** 3, t ** 4, t ** 5], dtype=np.float64)
            return np.asarray(p0 + v0 * tt[1] + c3 * tt[2] + c4 * tt[3] + c5 * tt[4], dtype=np.float64)

        def _lerp(a, b, t):
            """线性插值"""
            return a * (1 - t) + b * t

        # ===== 1. F50 位置: 直接使用 Stage 1 优化结果 (v4.7 简化) =====
        # v4.7: 不再参考 MANO xy, 直接使用 Stage 1 优化的 (x, y, z) 作为 F50 抓取点.
        # 用户: "F50: 抓取点 位置: Stage 1 优化 + MANO x,y 参考 就不要参考mano的xy了, 就是stage1得到的最优位姿就ok"
        # 原因: MANO xy 与物体位置可能差距很大, 用 MANO xy 会导致夹爪不在物体位置.
        mano_f50_xy = pos_traj[F50_IDX][:2]
        stage1_f50_xy = np.asarray(grasp_pose['pos'], dtype=np.float64)[:2]
        xy_gap = float(np.linalg.norm(mano_f50_xy - stage1_f50_xy))

        optimized_f50_pos = np.asarray(grasp_pose['pos'], dtype=np.float64).copy()
        # F50 z 值: Stage1 输出的 pos[2] 已经是 enclose_z (手指接触物体时的 base z).
        # 兼容旧 npz: 若 pos[2] > enclose_z, 说明 pos[2] 是 hover_z, 需要用 enclose_z 替换.
        _enclose_z = float(grasp_pose.get('enclose_z', 0.0))
        if abs(_enclose_z) < 1.0 and optimized_f50_pos[2] > _enclose_z + 0.005:
            # 旧 npz: pos[2] 是 hover_z, 需要替换为 enclose_z
            logger.info(f"[Stage 2] F50 z: pos[2]={grasp_pose['pos'][2]:.4f} (hover_z) → enclose_z={_enclose_z:.4f}")
            optimized_f50_pos[2] = _enclose_z
        else:
            logger.info(f"[Stage 2] F50 z: pos[2]={optimized_f50_pos[2]:.4f} (已经是 enclose_z)")

        # F50 xy: 直接使用 Stage 1 优化的 (x, y), 不参考 MANO xy
        logger.info(f"[Stage 2] F50 xy: 直接使用 Stage1 优化结果 "
                    f"(Stage1={stage1_f50_xy.round(4)}, MANO={mano_f50_xy.round(4)}, gap={xy_gap*1000:.1f}mm)")

        # ===== 2. 位置 smoothstep 插值 =====
        # F46~F49: MANO[offset] → F50 (接近期: 边下降边闭合)
        # F50: 夹取位姿 (enclose_z)
        # F51~F54: 保持 F50 位置 (确保夹持稳固, 不能立刻移动)
        # F55~F58: F50 → MANO[offset] (携带物体跟随 MANO 轨迹移动)
        #   MANO 有抬升就有, 没有就没有; 从 F50 位姿平滑过渡到 MANO+offset 位姿
        # 关键: F46~F49 分两段:
        #   F46-F47: z 保持在 hover_z 附近 (手指全开下降, 避免碰物体)
        #   F48-F49: z 从 hover_z 快速下降到 enclose_z (手指开始闭合, 不会推开物体)
        # hover_z - enclose_z 间距: 和 Stage 1 一致 (0.058)
        _hover_z_f50 = _enclose_z + 0.058 if abs(_enclose_z) < 1.0 else optimized_f50_pos[2] + 0.058
        # F55 开始过渡: F50 之后 5 帧保持, 再 3 帧过渡
        # v4.9.2: _HOLD_FRAMES 4 → 6, 让 F50-F56 保持 F50 位置 (PD 7 帧收敛时间)
        # 临界阻尼系统 τ=2*damping/stiffness=0.4s≈10帧, F50 时 qpos=0.018 远大于 target=0.0034
        # 给 PD 7 帧 (F50-F56) 收敛, F57-F62 才开始 smoothstep 过渡到 MANO
        _HOLD_FRAMES = 6  # v4.9.2: 4 → 6 (F51-F56: 保持 F50 位置)
        _TRANS_START = F50_IDX + _HOLD_FRAMES + 1  # F55 开始过渡
        pos_interp = {}
        T_in = max(F50_IDX - f_start, 1)
        T_out = max(f_end - _TRANS_START + 1, 1)
        for f in range(f_start, f_end + 1):
            if f < F50_IDX:
                t = (f - f_start) / T_in
                s = t * t * (3 - 2 * t)  # smoothstep
                xy_interp = _lerp(pos_traj[f_start][:2], optimized_f50_pos[:2], s)
                if t < 0.5:
                    z_interp = _hover_z_f50
                else:
                    z_t = (t - 0.5) / 0.5
                    z_s = z_t * z_t * (3 - 2 * z_t)
                    z_interp = _hover_z_f50 * (1 - z_s) + optimized_f50_pos[2] * z_s
                pos_interp[f] = np.array([xy_interp[0], xy_interp[1], z_interp])
            elif f <= F50_IDX + _HOLD_FRAMES:
                # F50-F54: 保持 F50 位置 (确保夹持稳固)
                pos_interp[f] = optimized_f50_pos
            else:
                # F55-F58: 从 F50 位姿平滑过渡到 MANO[offset] 位姿
                t = (f - _TRANS_START + 1) / max(T_out, 1)
                s = min(1.0, t) * min(1.0, t) * (3 - 2 * min(1.0, t))  # smoothstep
                pos_interp[f] = optimized_f50_pos * (1 - s) + pos_traj[f] * s

        # ===== 3. 姿态插值 =====
        # F46~F49: 保持 MANO 姿态, 仅在 F49→F50 微调到 Stage 1 R
        # F50: Stage 1 优化姿态 (已约束在 MANO F50 的 5° 以内)
        # F51~F54: 保持 F50 姿态 (保证夹持对齐)
        # F55~F58: 从 F50 姿态平滑过渡到 MANO 姿态 (跟随 MANO 运动趋势)
        grasp_R = np.asarray(grasp_pose['R'], dtype=np.float64)
        rot_interp = {}
        for f in range(f_start, f_end + 1):
            if f < F50_IDX:
                t = (f - f_start) / max(F50_IDX - f_start, 1)
                if t < 0.8:
                    rot_interp[f] = R_traj[f].copy()
                else:
                    slerp_t = (t - 0.8) / 0.2
                    slerp_f = Slerp([0, 1], R_scipy.from_matrix([R_traj[f], grasp_R]))
                    _R = slerp_f(slerp_t).as_matrix()
                    rot_interp[f] = _R[0] if _R.ndim == 3 else _R
            elif f <= F50_IDX + _HOLD_FRAMES:
                # F50-F54: 保持 F50 姿态 (保证夹持对齐)
                rot_interp[f] = grasp_R.copy()
            else:
                # F55-F58: 从 F50 姿态平滑过渡到 MANO 姿态
                t = (f - _TRANS_START + 1) / max(T_out, 1)
                s = min(1.0, t) * min(1.0, t) * (3 - 2 * min(1.0, t))
                slerp_f = Slerp([0, 1], R_scipy.from_matrix([grasp_R, R_traj[f]]))
                _R = slerp_f(s).as_matrix()
                rot_interp[f] = _R[0] if _R.ndim == 3 else _R

        # ===== 4. 夹爪开合度 =====
        # 和 Stage 1 rollout 一致: 边下降边闭合
        # F46~F47: 手指全开 (在 hover_z 上方, 不碰物体)
        # F48~F49: 手指逐渐预闭合 (z 开始下降, 手指也跟着闭合, 不会推开物体)
        # F50: 闭合到夹持开合度 (在 enclose_z 闭合夹住物体)
        # F51~F54: 保持闭合 (维持夹持力!)
        g_grasp = float(grasp_pose.get('gripper_qpos', 0.002))  # F50 夹持闭合度
        _MAX_OPEN = 0.04  # 手指最大开合度 (全开)
        _PRELOAD = 0.0040  # 预紧点 (同 Stage 1 rollout)
        gripper_interp = {}
        for f in range(f_start, f_end + 1):
            if f < F50_IDX:
                t = (f - f_start) / max(F50_IDX - f_start, 1)
                if t < 0.5:
                    # 前半段: 手指全开 (z 在 hover_z, 不会碰物体)
                    gripper_interp[f] = _MAX_OPEN
                else:
                    # 后半段: 手指从全开逐渐预闭合 (z 开始下降, 需要手指收拢避免碰物体)
                    close_t = (t - 0.5) / 0.5
                    close_s = close_t * close_t * (3 - 2 * close_t)
                    gripper_interp[f] = _MAX_OPEN + (_PRELOAD - _MAX_OPEN) * close_s
            elif f == F50_IDX:
                gripper_interp[f] = g_grasp
            else:
                gripper_interp[f] = g_grasp

        logger.info(f"[Stage 2] 重建 F{f_start}-F{f_end} 过渡轨迹, F50={F50_IDX}")
        logger.info(f"[Stage 2] 夹爪策略: F{f_start}-F47={_MAX_OPEN:.4f}(全开) → "
                    f"F48-F49=预闭合({_PRELOAD:.4f}) → F50={g_grasp:.4f}(夹持) → "
                    f"F{f_end}={g_grasp:.4f}(保持夹持)")
        return {
            'pos': pos_interp,
            'R': rot_interp,
            'gripper_qpos': gripper_interp,
            'frames': list(range(f_start, f_end + 1)),
        }
