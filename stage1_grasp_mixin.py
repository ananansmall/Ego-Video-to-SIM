"""stage1_grasp_mixin.py — Stage 1: 抓取姿态优化 (6DOF CMA-ES)

从 grasp_hawor.py 抽出, 提供 Stage1Mixin 类。
Stage 1: 生成候选抓取姿态 → 物理仿真验证 → CMA-ES 优化 → 选出最优 grasp_pose
依赖主类 GraspSimulator 的以下属性/方法 (由主类初始化):
  - self._mano_gripper_traj, self._mano_neutral_offset, self._mano_grasp_frame
  - self.grasp_controllers, self.obj_bbox_centers, self.obj_info
  - self.scene, self.robot_info, self.obj_actors, self._ground_z
"""
import logging
import numpy as np
import sapien

from physics_env import (
    FINGER_EFFECTIVE_HALF_SPACING, _GRIP_CLOSE, _GRIP_OVERCLOSURE,
    GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_FORCE,
    GRASP_STRATEGIES, GRIPPER_INIT_OPEN, GRIPPER_MAX_OPEN, MAX_ROOT_STEP,
    OBJECT_MIN_MASS, OBJECT_DENSITY, GRIPPER_FRICTION,
)
from stage1_recipe import (
    STAGE1_CLOSE_CONTROLS,
    STAGE1_SETTLE_CONTROLS,
    build_physics_config,
    build_scene_digest,
    build_stage1_grasp_recipe,
    stage1_close_q_sequence,
    stage_q_open,
)

logger = logging.getLogger("grasp_hawor")


class Stage1Mixin:
    """Stage 1: 抓取姿态优化 Mixin"""

    # === compute_gripper_qpos (L4961) ===
    def compute_gripper_qpos(self, obj_bbox_size):
        """根据物体 bbox 最大水平边计算夹爪开合度目标值

        返回: gripper_qpos 使手指能真正接触并挤压物体表面。
        使用最大水平边(而非最小边), 保证平行指在任意水平朝向都能碰到物体;
        目标 qpos 比几何接触点再小 _GRIP_OVERCLOSURE (默认 0.5mm),
        让 PD 驱动器在接触后仍有位置误差, 从而产生持续正压力。
        """
        _obj_max_dim = float(max(obj_bbox_size[:2])) if obj_bbox_size is not None else 0.035
        _q_contact = max(0.0, _obj_max_dim / 2.0 - FINGER_EFFECTIVE_HALF_SPACING)
        return max(_GRIP_CLOSE, _q_contact - _GRIP_OVERCLOSURE)

    # === _get_mano_f50_pose (L4973) ===
    def _get_mano_f50_pose(self, side='right'):
        """获取 MANO F50 的位姿 (pos, R)

        返回世界坐标系下的 MANO F50 位姿。若 _mano_neutral_offset 已计算,
        则应用该偏移将 MANO 局部坐标对齐到目标物体; 否则返回原始 MANO 局部坐标。
        """
        traj = self._mano_gripper_traj.get(side, {})
        pos_traj = np.asarray(traj.get("pos", []))
        R_traj = np.asarray(traj.get("R", []))
        n = len(pos_traj)
        if n == 0:
            return None, None
        f50_idx = min(50, n - 1)
        pos = pos_traj[f50_idx].copy()
        R = R_traj[f50_idx] if len(R_traj) > f50_idx else None
        # 外部附加旋转角 (用户建议: 只是一个额外的旋转角, 从命令行 --f50-tilt-deg 传入)
        # 绕世界 Y 轴旋转 F50 姿态. 0=关闭(默认); -90 使手指倾斜向下
        # (F50 X列 x分量为负, 需用 -90°; 实测 X列 [-0.552,0.754,0.357] -> [-0.357,0.754,-0.552])
        _tilt_deg = float(getattr(self, 'f50_tilt_deg', 0.0) or 0.0)
        if R is not None and abs(_tilt_deg) > 1e-6:
            from scipy.spatial.transform import Rotation as _Rsci
            _tilt_R = _Rsci.from_euler('y', _tilt_deg, degrees=True).as_matrix()
            _R_old = np.asarray(R, dtype=np.float64)
            R = _tilt_R @ _R_old
            if not getattr(self, '_f50_tilt_logged', False):
                self._f50_tilt_logged = True
                logger.info(f"[F50附加旋转] 绕世界Y轴 {_tilt_deg}°: "
                            f"X列 {_R_old[:, 0].round(3)} -> {np.asarray(R)[:, 0].round(3)}")
        offset = getattr(self, '_mano_neutral_offset', {}).get(side)
        if offset is not None:
            pos = pos + offset
        return pos, R

    # === score_grasp_quality (L4993) ===
    def score_grasp_quality(self, grasp_pose, target_name):
        """无仿真 Grasp Quality 评分 (第2层预筛选)

        基于几何分析评估候选 grasp_pose 的抓取质量，无需物理引擎。
        评分项 (与 plan v4.5 对齐):
        - antipodal check (0.35分): 手指接触点位于物体两侧, 物体中心在手指连线附近
        - force closure (0.25分): 接触法线反向共线, 可抵抗任意外力/力矩
        - reachability (0.20分): 候选相对 MANO[F50] 的距离
        - z-safety (0.20分): base 高度合理, 不碰桌面也不悬空过高

        返回: (score_0_1, passes_hard_constraint, details)
        """
        from scipy.spatial.transform import Rotation as R_scipy

        score = 0.0
        details = {}

        # ---- 获取目标物体信息 ----
        if target_name is None or target_name not in self.obj_info:
            logger.warning("[score_grasp] 无法获取目标物体信息")
            return 0.0, False, details

        obj_bbox_size = self.obj_info[target_name].get('bbox_size', None)
        obj_center_3d = np.array(self.obj_bbox_centers.get(target_name, [0, 0, 0]))

        # ---- 获取 MANO F50 位姿 ----
        mano_f50_pos, mano_f50_R = self._get_mano_f50_pose('right')
        if mano_f50_pos is None:
            mano_f50_pos = obj_center_3d.copy()
            mano_f50_pos[2] += 0.03689

        # ---- 夹爪几何参数 ----
        _FINGER_ORIGIN_OFFSET = 0.03689
        R = np.asarray(grasp_pose['R'], dtype=np.float64)
        base_pos = np.asarray(grasp_pose['pos'], dtype=np.float64)
        close_dir = R @ np.array([0.0, 1.0, 0.0])   # 手指闭合方向 (world)
        forward_dir = R @ np.array([1.0, 0.0, 0.0]) # 从 base 指向手指原点 (world)
        finger_origin = base_pos + _FINGER_ORIGIN_OFFSET * forward_dir
        gripper_q = self.compute_gripper_qpos(obj_bbox_size)
        # 接触点半距使用有效半间距(扣除碰撞盒厚度)
        half_sep = FINGER_EFFECTIVE_HALF_SPACING + gripper_q

        # 预估接触点 (闭合 target 处)
        p1 = finger_origin + half_sep * close_dir
        p2 = finger_origin - half_sep * close_dir

        # 物体几何近似
        if obj_bbox_size is not None and len(obj_bbox_size) >= 3:
            obj_radius = max(float(obj_bbox_size[0]), float(obj_bbox_size[1])) / 2.0
            obj_half_height = float(obj_bbox_size[2]) / 2.0
        else:
            obj_radius = 0.02
            obj_half_height = 0.02

        # ---- 1. Antipodal Check (0.35分) ----
        # 物体中心应位于两手指接触点之间, 且到手指连线的距离不超过物体半径
        vec = p1 - p2
        line_len = np.linalg.norm(vec)
        if line_len > 1e-6:
            line_dir = vec / line_len
            proj = np.dot(obj_center_3d - p2, line_dir)
            t_proj = proj / line_len          # 0 在 p2, 1 在 p1
            closest = p2 + proj * line_dir
            dist_to_line = float(np.linalg.norm(obj_center_3d - closest))
        else:
            t_proj = 0.0
            dist_to_line = float(np.linalg.norm(obj_center_3d - finger_origin))

        details['antipodal_t'] = round(float(t_proj), 3)
        details['antipodal_dist'] = round(dist_to_line, 4)
        antipodal_ok = (0.1 <= t_proj <= 0.9) and (dist_to_line <= obj_radius + 0.005)
        if antipodal_ok:
            score += 0.35
            details['antipodal'] = 'PASS'
        elif (0.0 <= t_proj <= 1.0) and (dist_to_line <= obj_radius + 0.015):
            score += 0.15
            details['antipodal'] = 'MARGINAL'
        else:
            details['antipodal'] = 'FAIL'

        # ---- 2. Force Closure (0.25分) ----
        # 平行指夹爪力封闭: 接触法线反向共线, 且作用线通过或接近物体质心
        # antipodal_ok 已保证作用线穿过物体, 法线自然反向, 故同条件判定
        if antipodal_ok:
            score += 0.25
            details['force_closure'] = 'PASS'
        else:
            details['force_closure'] = 'FAIL'

        # ---- 3. Reachability (0.20分) ----
        # 候选 base 位置相对 MANO[F50] 的 3D 距离, 越近越好
        dist_3d = float(np.linalg.norm(base_pos - mano_f50_pos))
        details['dist_to_mano_3d'] = round(dist_3d, 4)
        score += 0.20 * np.exp(-dist_3d / 0.02)

        # ---- 4. Z-Safety (0.20分) ----
        # base 应在物体顶面之上, 但不能过高导致手指碰不到物体
        obj_top_z = obj_center_3d[2] + obj_half_height
        z_gap = float(base_pos[2] - obj_top_z)
        details['z_gap'] = round(z_gap, 4)
        if 0.005 <= z_gap <= 0.10:
            score += 0.20
            details['z_safety'] = 'OK'
        elif z_gap > 0.10:
            score += 0.05
            details['z_safety'] = 'HIGH'
        else:
            details['z_safety'] = 'LOW'

        # ---- MANO 贴合硬约束 ----
        # hard_ok 由调用方用 _check_mano_hard_constraint 统一判断,
        # 本函数只做几何质量评分。
        hard_ok = True

        return score, hard_ok, details

    # === generate_grasp_candidates (L5109) ===
    def generate_grasp_candidates(self, side, n_candidates=32, base_R=None):
        """Stage 0: 生成候选抓取姿态 (6DOF: pos[3] + euler[3])

        v4.5: 以物体位置为基准, MANO F50 姿态为参考, 局部扰动,
        保证候选天然贴合 MANO 姿态, 同时覆盖物体上方 xy/z/ry 微调。
        """
        from scipy.spatial.transform import Rotation as R_scipy
        from data_loader import rotmat_to_zyx_euler

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        if target_name is None or target_name not in self.obj_bbox_centers:
            if not self.obj_bbox_centers:
                return []
            target_name = list(self.obj_bbox_centers.keys())[0]

        obj_pos = np.array(self.obj_bbox_centers[target_name], dtype=np.float64)
        _rot_close = R_scipy.from_euler("y", +90, degrees=True).as_matrix()

        # v4.7: 候选以物体位置为中心，MANO 姿态为参考
        # 关键修复: base_pos 必须考虑手指偏移方向 (gripper X → world)
        # 手指 origin 在 gripper X=+0.03689 处, 旋转后方向随 base_R 变化
        # base_R = _rot_close (ry=+90°): 手指朝 -Z, base 在物体正上方 3.689cm
        # base_R = mano_f50_R: 手指朝斜方向, base 需偏移使手指到达物体中心
        base_R = (base_R.copy() if base_R is not None else _rot_close.copy())
        ref_R = base_R.copy()  # 硬约束参考

        # 手指偏移: gripper frame [0.03689, 0, 0] → world frame
        _FINGER_OFFSET_GRIPPER = np.array([0.03689, 0.0, 0.0])
        finger_offset_world = base_R @ _FINGER_OFFSET_GRIPPER
        # base_pos = obj_pos - finger_offset (使手指中心对准物体)
        # 不加 hover (hover 在 rollout_grasp_only 中通过 enclose_z/hover_z 控制)
        base_pos = obj_pos - finger_offset_world

        logger.info(f"[Stage 0] 以物体为基准: obj_pos={obj_pos.round(4)}, base_pos={base_pos.round(4)}")
        if base_R is not None:
            logger.info(f"[Stage 0] 姿态基准: MANO F50 R (det={np.linalg.det(base_R):.3f})")
        else:
            logger.info(f"[Stage 0] 姿态基准: _rot_close (手指朝下)")

        logger.info(f"[Stage 0] 以物体为基准: obj_pos={obj_pos.round(4)}, base_pos={base_pos.round(4)}")

        # v4.35 [位置搜索扩展] 用户: "夹爪姿态定了, 位置还没定, 这个部分".
        # 旧网格在世界系 xy ±7.5mm / z ±2.5mm 平移 + ry ±1.1° 微扰,
        # 若 MANO 闭合方向不横跨物体两侧, 该范围无法让双指同时触物 -> 单指最优.
        # 新网格: 姿态完全锁定 MANO R (去掉 ry 扰动), 沿夹爪局部轴扫位置:
        #   close (gripper Y, 闭合方向): ±17.5mm step 5mm -> 8 档 (决定双指能否跨住物体)
        #   depth (gripper X, base->手指): ±12.5mm step 5mm -> 6 档 (决定包住深浅)
        #   z (world): ±10mm step 5mm -> 5 档 (v4.36 放宽: 原 ±2.5mm 无法修正垂直偏差,
        #     实测最优解 base 比物体高 ~4cm 导致双指悬空、物体不抬起)
        candidates = []
        n_close, n_depth, n_z = 8, 6, 5
        _close_l = np.array([0.0, 1.0, 0.0])   # gripper Y: 手指开合方向
        _fwd_l = np.array([1.0, 0.0, 0.0])     # gripper X: base -> 手指原点
        for i in range(min(n_candidates, n_close * n_depth * n_z)):
            icl = i % n_close
            idp = (i // n_close) % n_depth
            iz = (i // (n_close * n_depth)) % n_z
            d_close = (icl - (n_close - 1) / 2.0) * 0.005
            d_fwd = (idp - (n_depth - 1) / 2.0) * 0.005
            dz = (iz - (n_z - 1) / 2.0) * 0.01
            delta_pos = base_R @ (_fwd_l * d_fwd + _close_l * d_close)
            delta_pos[2] += dz
            pos = base_pos + delta_pos
            # v4.23 (GLB 坐标): 不再假设 base 必须在物体上方。统一坐标系后 MANO F50
            # 手指方向是真实手势 (可能朝上/斜向, X 轴 z>0), base 自然在物体下方;
            # 只要 base 不穿地即可, 抓取可行性交给 Stage1b/1c 物理验证。
            _g0 = float(getattr(self, '_ground_z', 0.0))
            if pos[2] < _g0 + 0.01:
                continue  # base 不能穿地 (GLB 地面 z=0)
            # v4.35: 姿态锁定 MANO R (用户: 姿态定了), 只搜位置
            R = base_R.copy()
            candidates.append({'pos': pos, 'R': R,
                               'euler': np.array(rotmat_to_zyx_euler(R)),
                               'intended_finger_offset': delta_pos,
                               'ref_base_R': ref_R.copy(),
                               'mano_anchor': True})

        for _ic, _cand in enumerate(candidates[:3]):
            logger.info(f"[Stage 0] cand {_ic}: pos={_cand['pos'].round(4)}, "
                        f"offset={_cand.get('intended_finger_offset', [0,0,0])[:2]}")
        logger.info(f"[Stage 0] 生成 {len(candidates)}/{n_candidates} 个候选, "
                    f"obj_pos={obj_pos.round(4)}, base_pos={base_pos.round(4)}")
        return candidates

    # === rollout_grasp_only (L5188) ===
    def rollout_grasp_only(self, grasp_pose, side, strategy='pd_then_lock', n_frames=160, record_video=False):
        """Stage 1 统一动作模板 (v5.1, 用户指定): 给定位姿 → 直接闭合 → 竖直提起 20cm.

        无下降段 — 位姿已由 CEM 优化好, 夹爪直接在位姿处就位 (一次性 set_qpos),
        动作完全固定、无自由度 — Stage1 唯一的变量是 grasp_pose 本身:
          - 就位:    set_qpos 到位姿 (手指张开包住物体) + warmup 5 帧
          - F0-F19:  原地闭合到 _GRIP_TARGET (基于物体尺寸的 PD 目标)
          - F20-F29: 维持夹持, 等待双侧接触稳定建立
          - F30-末:  竖直提起 20cm (速度自适应 = 0.20m / 抬升帧数), 边抬边保持 PD 压力

        默认 n_frames=160 (4s @ 40Hz): 抬升段 130 帧 × 1.54mm/帧 = 20cm.
        reward 只问一件事: 物体最终有没有和夹爪一起提升 (见 stage1_reward v5.0).

        Args:
            grasp_pose: dict with 'pos'(3,) and 'R'(3x3)
            side: 'right' or 'left'
            n_frames: 仿真帧数 (默认 160, 保证完整 20cm 竖直抬升)
            record_video: 是否录制视频 (需要 cameras 已由 run() 初始化)

        Returns:
            dict with 'contact_count', 'both_contact_count', 'obj_lift',
                      'obj_xy_drift', 'grip_stability', 'obj_z_final',
                      'peak_grip_force', 'locked_gripper_qpos'
        """
        from data_loader import rotmat_to_zyx_euler
        from physics_env import physics_step, get_finger_contacts, get_grasp_force

        robot = self.robot_info["robot"]
        virtual_idx = self.robot_info.get("virtual_idx", {})
        gripper_idx1 = self.robot_info["gripper_idx1"]
        gripper_idx2 = self.robot_info["gripper_idx2"]
        arm_joint_indices = self.robot_info.get("arm_joint_indices", [])
        active_joints = robot.get_active_joints()

        # v6.0 统一物理: 手指 PD 参数由 setup_robot 统一配置 (K=1000/D=200/F=100N),
        # 不再在 rollout 内重复设置 (历史: 此处曾用 force_limit=100N 重设, 与
        # 主回放的 4000N 并存, 全管线手指物理不一致)

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None

        # 记录初始物体位姿
        if not hasattr(self, '_obj_initial_poses') or self._obj_initial_poses is None:
            self._obj_initial_poses = {actor.get_name(): actor.get_pose() for actor in self.obj_actors}

        # === 1. 重置场景 ===
        # v51 (2026-09-10): 让 Stage1 与 Stage2 使用**同一碰撞作用域**
        # (stage1_target_isolated; 含 _NO_GRIPPER_GROUND 实验旋钮), 否则 Stage1 的
        # anchor 是在"夹爪能碰地面/邻物"的环境里搜出来的, Stage2 无法复现.
        if target_name is not None:
            try:
                self._s1_collision_scope = self._configure_stage1_target_isolated_collisions(
                    robot, side, self.obj_actors, target_name)
            except Exception as _scope_err:
                logger.warning(f"[Stage1] 碰撞作用域配置失败 (沿用默认): {_scope_err}")
                self._s1_collision_scope = None
        robot.set_qpos(self.robot_info["init_qpos"].copy())
        robot.set_qvel(np.zeros_like(robot.get_qvel()))
        if virtual_idx:
            robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        robot.set_root_linear_velocity([0, 0, 0])
        robot.set_root_angular_velocity([0, 0, 0])
        initial_poses = getattr(self, '_obj_initial_poses', None)
        if initial_poses:
            for actor in self.obj_actors:
                name = actor.get_name()
                if name in initial_poses:
                    actor.set_pose(initial_poses[name])

        # === 2. 初始化夹爪位置 ===
        target_pos = np.asarray(grasp_pose['pos'], dtype=np.float64)
        target_R = grasp_pose['R']
        rz, ry, rx = rotmat_to_zyx_euler(target_R)

        # 记录物体初始位置 (在设置夹爪位置之前, 确保场景已重置)
        obj_init_pos, obj_init_z, obj_init_xy = None, 0.0, np.zeros(2)
        if target_name is not None:
            for actor in self.obj_actors:
                if target_name in actor.get_name():
                    obj_init_pos = np.array(actor.get_pose().p)
                    obj_init_z = float(obj_init_pos[2])
                    obj_init_xy = obj_init_pos[:2].copy()
                    break
        logger.info(f"[Stage1 rollout] input pos={np.asarray(grasp_pose['pos']).round(4)}, "
                    f"obj_init_pos={obj_init_pos.round(4) if obj_init_pos is not None else None}")

        # Stage 1 高度策略 (v5.1 无下降段):
        #   enclose_z = obj_z - finger_offset_z (gripper_base z 使手指到达物体中心高度)
        #   v4.7 关键修复: 手指偏移方向随 base_R 变化, enclose_z 不再固定为 obj_z+0.03689
        #   - base_R = _rot_close (ry=+90°): finger_offset_z = -0.03689, enclose_z = obj_z + 0.03689
        #   - base_R = mano_f50_R: finger_offset_z 随旋转变化, enclose_z 相应调整
        #   就位:    set_qpos 直接到位姿 (z 允许在 enclose_z ±1cm 微调, 防手指插地)
        #   F0-F19   原地闭合
        #   F20-F29  维持夹持等待接触稳定
        #   F30-末   竖直提起 20cm
        _GRASP_R = np.asarray(grasp_pose.get('R', np.eye(3)), dtype=np.float64)
        _FINGER_OFFSET_GRIPPER = np.array([0.03689, 0.0, 0.0])
        _finger_offset_world = _GRASP_R @ _FINGER_OFFSET_GRIPPER
        _finger_offset_z = float(_finger_offset_world[2])  # 手指相对 base 的 Z 偏移 (世界坐标)
        # enclose_z: base 的 z 值, 使手指中心 z = obj_z
        # finger_z = base_z + _finger_offset_z → base_z = obj_z - _finger_offset_z
        _ENCLOSE_Z_OFFSET = -_finger_offset_z  # 对 _rot_close: = 0.03689
        _enclose_z = obj_init_z + _ENCLOSE_Z_OFFSET
        self._last_rollout_enclose_z = _enclose_z  # 供 Stage1 保存到 best_output
        # v5.1: 无下降段 — CEM 的 z 直接决定初始夹持高度, 在 enclose_z ±1cm 内搜索
        # (既保留对夹持高度的优化自由度, 又避免手指插进桌面)
        _Z_SEARCH_RANGE = 0.01
        target_pos[2] = np.clip(target_pos[2], _enclose_z - _Z_SEARCH_RANGE, _enclose_z + _Z_SEARCH_RANGE)
        start_pos = target_pos.copy()  # 直接从位姿开始, 无 hover 下降

        qpos = robot.get_qpos().copy()
        if virtual_idx:
            qpos[virtual_idx['vx']] = start_pos[0]
            qpos[virtual_idx['vy']] = start_pos[1]
            qpos[virtual_idx['vz']] = start_pos[2]
            qpos[virtual_idx['rz']] = rz
            qpos[virtual_idx['ry']] = ry
            qpos[virtual_idx['rx']] = rx
        # v51 §3.1 (2026-09-10): 规范入口 = Stage2 F50 入口:
        # 实际/命令 q 都是 GRIPPER_MAX_OPEN(0.05), 不再用 _DESCEND_OPEN=0.012 的
        # 隐藏 warmup. 旧行为会先把手指收到 0.012 再 warmup, 目标被顶起 ~4.5cm
        # 后被"空中夹住", anchor 依赖隐藏初始化, 无法迁移到 Stage2 的 q=0.05 入口
        # (T1 F50_TRANSFER_EQUIVALENCE 实测 0/3: 张开时锚点与目标互穿, 一上物理
        #  物体即被弹飞 35cm). 因此 Stage1 必须从规范入口验证 anchor.
        _q_open = stage_q_open()
        qpos[gripper_idx1] = _q_open
        qpos[gripper_idx2] = _q_open
        robot.set_qpos(qpos)

        if virtual_idx:
            for vkey, vval in [('vx', start_pos[0]), ('vy', start_pos[1]),
                               ('vz', start_pos[2]), ('rz', rz), ('ry', ry), ('rx', rx)]:
                active_joints[virtual_idx[vkey]].set_drive_target(float(vval))
        active_joints[gripper_idx1].set_drive_target(_q_open)
        active_joints[gripper_idx2].set_drive_target(_q_open)

        # warmup: 命令保持 q_open (不再跟随实际 qpos)
        for _ in range(5):
            physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                         np.array([]), _q_open, _q_open,
                         self.scene)

        # 规范入口自检: warmup 后不得已与目标接触 (否则该 anchor 在 Stage2 的
        # q=0.05 入口会穿模弹飞物体, 不能作为可迁移 anchor).
        _entry_f1, _entry_f2, _entry_objs = get_finger_contacts(
            robot, side, self.scene, self.obj_actors)
        self._last_entry_target_contact = bool(
            target_name is not None and target_name in _entry_objs)
        if self._last_entry_target_contact:
            logger.info(
                "[Stage1][入口自检] anchor 在 q_open 入口已与 %s 接触 "
                "(f1=%s f2=%s) → 该候选不是可迁移 anchor",
                target_name, _entry_f1, _entry_f2)

        # PD 闭合目标: 基于物体尺寸计算, 让手指闭合到物体表面 (略收紧0.5mm产生正压力)
        # v4.8 修复: PD_target=0.0 会让手指闭合到关节下限 (间距4.4mm), 物体宽20.9mm根本夹不住
        # 正确做法: PD_target = compute_gripper_qpos (略小于接触点), 让 PD 在物体表面产生持续压力
        _GRIP_TARGET = 0.0  # 兜底默认值, 后面按物体尺寸覆盖
        if target_name is not None and target_name in self.obj_info:
            _bbox_size = self.obj_info[target_name].get('bbox_size', None)
            if _bbox_size is not None:
                _computed_target = self.compute_gripper_qpos(_bbox_size)
                _bbox_max = float(max(_bbox_size[0], _bbox_size[1]))
                _q_contact = max(0.0, _bbox_max / 2.0 - FINGER_EFFECTIVE_HALF_SPACING)
                _GRIP_TARGET = float(_computed_target)  # 略小于接触点, 产生持续正压力
                # v51 §3.4 (2026-09-05): 闭合深度微调 Δ (环境变量测试旋钮) —
                # 过闭合仅 0.5mm 时挤压力不足 (小物体实测 1.08N 滑脱), 深压机理同 stage3 -0.004
                import os as _os
                _GRIP_TARGET -= float(_os.environ.get('_S1_GRIP_DELTA', '0'))
                logger.info(f"[Stage1] 目标物体 {_bbox_max*100:.2f}cm, "
                            f"接触 qpos={_q_contact:.4f}, compute_gripper_qpos={_computed_target:.4f}, "
                            f"PD_target={_GRIP_TARGET:.4f} (基于物体尺寸, 略小于接触点产生正压力), "
                            f"enclose_z={_enclose_z:.4f} (v5.1 无下降段, z 在 enclose±1cm)")

        # === 3. 仿真 ===
        contact_count = 0
        both_contact_count = 0
        obj_z_history = []
        obj_pos_history_s1 = []      # 初始移动检测用 (2026-09-06)
        _s1_first_both = None        # 首次双侧接触帧
        fingertip_dist_history = []
        peak_grip_force = 0.0
        # 2026-08-31: 实际基座 vs 目标轨迹 (跟随误差用)
        _base_actual_traj_s1 = []
        _base_target_traj_s1 = []
        # v5.1 统一动作模板 (用户指定): 无下降段 — 直接闭合 → 竖直提起 20cm
        _CLOSE_FRAMES = STAGE1_CLOSE_CONTROLS    # F0-F19:  原地闭合 (从张开包住到 _GRIP_TARGET)
        _SETTLE_FRAMES = STAGE1_SETTLE_CONTROLS  # F20-F29: 维持夹持等待接触稳定
        _LIFT_START_FRAME = _CLOSE_FRAMES + _SETTLE_FRAMES
        # 抬升段占满剩余全部帧, 总抬升固定 20cm.
        # 160 帧 → 抬升 130 帧 × 1.54mm/帧 ≈ 62mm/s; reward 以"跟随提满 20cm"为主导.
        _LIFT_TOTAL_M = 0.20
        _LIFT_FRAMES = max(1, n_frames - _LIFT_START_FRAME)
        _LIFT_SPEED = _LIFT_TOTAL_M / _LIFT_FRAMES  # 自适应速度, 保证无论 n_frames 都提满 20cm

        # v5 纯物理 (2026-08-25): 删除 Hack① 手指-地面碰撞禁用 与 Hack② 邻物碰撞禁用,
        # 对齐 test8 场景原则 —— 碰撞一律真实生效, 抓取解必须在无失真环境下可行.

        # 闭合阶段: F0-F19 原地快速闭合到 _GRIP_TARGET,
        # 让抬升全过程保持 PD 位置误差 → 持续正压力, 防止物体滑落.
        # v51 §3.5: 闭合序列与 Stage2 共用 stage1_recipe.stage1_close_q_sequence
        # (cubic smoothstep, linspace(0,1,20)), 数值与历史实现逐值一致.
        _close1_q = stage1_close_q_sequence(stage_q_open(), _GRIP_TARGET, _CLOSE_FRAMES)
        # 供 Stage1 artifact 导出 recipe 时校验"实际命令的 q_grasp"
        self._last_grip_target_s1 = float(_GRIP_TARGET)

        for frame in range(n_frames):
            qpos = robot.get_qpos().copy()

            # 位置: 就位维持 → 竖直抬升 (v5.1 无下降段)
            if frame < _LIFT_START_FRAME:
                current_pos = target_pos.copy()
            else:
                _lift_t = frame - _LIFT_START_FRAME
                current_pos = target_pos.copy()
                current_pos[2] = target_pos[2] + _LIFT_SPEED * _lift_t

            if virtual_idx:
                qpos[virtual_idx['vx']] = current_pos[0]
                qpos[virtual_idx['vy']] = current_pos[1]
                qpos[virtual_idx['vz']] = current_pos[2]
                qpos[virtual_idx['rz']] = rz
                qpos[virtual_idx['ry']] = ry
                qpos[virtual_idx['rx']] = rx

            # 手指: v5.1 — 就位时已张开包住物体 (set_qpos 初始 GRIPPER_MAX_OPEN),
            # F0-F19 原地闭合, 之后保持 PD 夹紧.
            if frame < _CLOSE_FRAMES:
                gripper_target = float(_close1_q[frame])
            else:
                gripper_target = _GRIP_TARGET  # 保持夹紧 (接触并挤压)

            # 闭合/维持/抬升阶段: 只设 PD target, 让手指自然收敛
            active_joints[gripper_idx1].set_drive_target(float(gripper_target))
            active_joints[gripper_idx2].set_drive_target(float(gripper_target))

            # v4.17: 虚拟关节也用纯 PD (对齐 test8), 不再 vlock teleport
            if virtual_idx:
                for vkey, vval in [('vx', current_pos[0]), ('vy', current_pos[1]),
                                   ('vz', current_pos[2]), ('rz', rz), ('ry', ry), ('rx', rx)]:
                    active_joints[virtual_idx[vkey]].set_drive_target(float(vval))

            physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                         np.array([]), float(gripper_target), float(gripper_target),
                         self.scene, lock_root_pose=None)

            # 2026-08-31: 采集实际基座位置 (用 virtual_idx), 供"跟随误差"指标
            if virtual_idx:
                try:
                    _aq = robot.get_qpos()
                    _base_actual_traj_s1.append([
                        float(_aq[virtual_idx['vx']]),
                        float(_aq[virtual_idx['vy']]),
                        float(_aq[virtual_idx['vz']]),
                    ])
                    _base_target_traj_s1.append(current_pos.copy())
                except Exception:
                    _base_actual_traj_s1.append(current_pos.copy())
                    _base_target_traj_s1.append(current_pos.copy())

            # v4.16: 录制视频 — vk DeviceLost 保护 + 帧标注 (2026-09-06)
            try:
                if record_video and getattr(self.scene, "_render_available", False):
                    # 2026-09-06 (用户: 视频看不到碰撞 box): 必须先更新碰撞 actor 位姿,
                    # 再 update_render 同步变换 — 顺序反了 take_picture 渲染的是旧位姿 (原点)
                    self._update_collision_visualization()
                    self.scene.update_render()
                    # 相机视角逐帧跟随 SLAM 相机轨迹 (F50 起缓推, 抓取测试段)
                    if hasattr(self, '_cam_view') and self._cam_view is not None \
                            and getattr(self, '_c2w_all', None) is not None:
                        _Rc, _tc = self._c2w_all
                        _gi = self.start_frame + int(getattr(self, '_frame_params', {}).get('F50_IDX', 50)) \
                            + int(frame * 15.0 / max(n_frames, 1))
                        if _gi < len(_Rc):
                            _cp, _cq = self._cam_pose_glb(_Rc[_gi], _tc[_gi])
                            self._cam_view.set_local_pose(sapien.Pose(_cp.tolist(), _cq.tolist()))
                    _label = (f"Stage1 grasp rollout | sim f{frame + 1}/{n_frames} | "
                              f"F50=stage1 最优位姿 | 就位->闭合->提升")
                    import cv2 as _cv2
                    if getattr(self, '_cam_view', None) is not None and getattr(self, '_writer_cam', None) is not None:
                        self._cam_view.take_picture()
                        _rgb = self._cam_view.get_picture("Color")[..., :3]
                        _bgr = np.ascontiguousarray((np.clip(_rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        _cv2.putText(_bgr, _label, (15, 28), _cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                     (0, 255, 255), 2, _cv2.LINE_AA)
                        self._writer_cam.write(_bgr)
                    if getattr(self, '_god_view', None) is not None and getattr(self, '_writer_god', None) is not None:
                        self._god_view.take_picture()
                        _rgb = self._god_view.get_picture("Color")[..., :3]
                        _bgr = np.ascontiguousarray((np.clip(_rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        _cv2.putText(_bgr, _label, (15, 28), _cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                     (0, 255, 255), 2, _cv2.LINE_AA)
                        self._writer_god.write(_bgr)
                    if getattr(self, '_right_view', None) is not None and getattr(self, '_writer_right', None) is not None:
                        self._right_view.take_picture()
                        _rgb = self._right_view.get_picture("Color")[..., :3]
                        _bgr = np.ascontiguousarray((np.clip(_rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        _cv2.putText(_bgr, _label, (15, 28), _cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                     (0, 255, 255), 2, _cv2.LINE_AA)
                        self._writer_right.write(_bgr)
            except Exception as _vk_err:
                logger.warning(f"[render] 渲染帧失败 (跳过, 不影响仿真与指标): {_vk_err}")

            if target_name is not None:
                f1_c, f2_c, contact_objs = get_finger_contacts(
                    robot, side, self.scene, self.obj_actors)
                if f1_c or f2_c:
                    if target_name in contact_objs:
                        contact_count += 1
                if f1_c and f2_c and target_name in contact_objs:
                    both_contact_count += 1

                _obj_cur = None
                for actor in self.obj_actors:
                    if target_name in actor.get_name():
                        _obj_cur = np.array(actor.get_pose().p)
                        obj_z_history.append(float(_obj_cur[2]))
                        obj_pos_history_s1.append(_obj_cur.copy())
                        if f1_c and f2_c and target_name in contact_objs and _s1_first_both is None:
                            _s1_first_both = len(obj_pos_history_s1) - 1  # 首次双侧接触帧
                        break
                # v4.23: 记录手指中点与物体的距离 (reward: 物体必须被抓在夹爪里)
                _fp1w = _fp2w = None
                for _link in robot.get_links():
                    _ln = _link.get_name()
                    if _ln == f"{side}_gripper_finger_link1":
                        _fp1w = np.array(_link.get_pose().p)
                    elif _ln == f"{side}_gripper_finger_link2":
                        _fp2w = np.array(_link.get_pose().p)
                if _obj_cur is not None and _fp1w is not None and _fp2w is not None:
                    _fmid = 0.5 * (_fp1w + _fp2w)
                    fingertip_dist_history.append(float(np.linalg.norm(_fmid - _obj_cur)))

                # 峰值力 (通过接触 impulse / dt 估算)
                _f = get_grasp_force(side, self.scene, self.obj_actors, robot)
                peak_grip_force = max(peak_grip_force, _f)

                if frame % 5 == 0:
                    _q = robot.get_qpos()
                    # 调试: 手指世界位置
                    _finger_pos = {}
                    for _link in robot.get_links():
                        _ln = _link.get_name()
                        if _ln in (f"{side}_gripper_finger_link1", f"{side}_gripper_finger_link2"):
                            _p = np.array(_link.get_pose().p)
                            _finger_pos[_ln] = _p
                    _fp1 = _finger_pos.get(f"{side}_gripper_finger_link1", np.zeros(3))
                    _fp2 = _finger_pos.get(f"{side}_gripper_finger_link2", np.zeros(3))
                    # 调试: 当前所有 gripper-object 接触对
                    _contacts = []
                    for _c in self.scene.get_contacts():
                        _b0, _b1 = _c.bodies[0], _c.bodies[1]
                        _n0 = _b0.entity.get_name() if hasattr(_b0, 'entity') else 'unknown'
                        _n1 = _b1.entity.get_name() if hasattr(_b1, 'entity') else 'unknown'
                        if 'gripper_finger' in _n0 or 'gripper_finger' in _n1:
                            _contacts.append((_n0, _n1))
                    logger.info(f"[Stage1 debug F{frame}] pos_z={current_pos[2]:.4f}, "
                                f"f1_qpos={_q[gripper_idx1]:.4f}, f2_qpos={_q[gripper_idx2]:.4f}, "
                                f"obj_z={obj_z_history[-1] if obj_z_history else -1:.4f}, "
                                f"f1_c={f1_c}, f2_c={f2_c}, force={_f:.2f}N\n"
                                f"  f1_pos={_fp1.round(4)}, f2_pos={_fp2.round(4)}, contacts={_contacts}")

        # === 4. 收集指标 ===
        obj_final_pos = None
        # 初始移动检测 (2026-09-06, 用户: "刚开始是不是有移动，导致目标位置没夹取成功"):
        # 物体在"放置→首次接触"之间的位移 — 若大, 说明就位/闭合前物体已被碰走
        _pre_close_disp = float('nan')
        if _s1_first_both is not None and 0 < _s1_first_both < len(obj_pos_history_s1) \
                and obj_pos_history_s1[0] is not None:
            _pre_close_disp = float(np.linalg.norm(
                np.asarray(obj_pos_history_s1[_s1_first_both])
                - np.asarray(obj_pos_history_s1[0])))
            logger.info(f"[Stage1][初始移动检测] 放置→首次双侧接触: 物体位移 "
                        f"{_pre_close_disp*1000:.1f}mm (frame={_s1_first_both}) "
                        f"{'⚠ 闭合前物体已被碰走' if _pre_close_disp > 0.01 else '✓ 闭合前物体稳定'}")
        obj_z_final, obj_xy_drift = 0.0, 0.0
        if target_name is not None:
            for actor in self.obj_actors:
                if target_name in actor.get_name():
                    obj_final_pos = np.array(actor.get_pose().p)
                    obj_z_final = float(obj_final_pos[2])
                    obj_xy_drift = float(np.linalg.norm(obj_final_pos[:2] - obj_init_xy))
                    break
        # v4.23: 手指中点-物体距离 (reward: 物体被抓在夹爪里)
        fingertip_obj_dist = float(fingertip_dist_history[-1]) if fingertip_dist_history else 9.9
        fingertip_obj_dist_min = float(min(fingertip_dist_history)) if fingertip_dist_history else 9.9

        obj_lift = max(0.0, obj_z_final - obj_init_z)
        # 峰值抬升: 抓取期间的物体最高z (物体可能在中途掉落, 最终z不代表真实抬升能力)
        _obj_peak_z = obj_init_z
        if len(obj_z_history) > 0:
            _obj_z_arr = np.array(obj_z_history)
            # v5.1: 抬升区间 = 闭合+稳定前缀之后到结尾 (与 _LIFT_START_FRAME 一致)
            _grasp_start = min(_LIFT_START_FRAME, len(_obj_z_arr))
            _grasp_end = min(len(_obj_z_arr), n_frames)
            if _grasp_start < _grasp_end:
                _obj_peak_z = float(np.max(_obj_z_arr[_grasp_start:_grasp_end]))
        obj_lift_peak = max(0.0, _obj_peak_z - obj_init_z)

        final_q = robot.get_qpos()
        locked_val1 = float(final_q[gripper_idx1])
        locked_val2 = float(final_q[gripper_idx2])

        logger.info(f"[Stage1 debug] final f1_qpos={locked_val1:.4f}, "
                    f"f2_qpos={locked_val2:.4f}, "
                    f"obj_z={obj_z_final:.4f}, obj_lift={obj_lift*100:.2f}cm, "
                    f"obj_lift_peak={obj_lift_peak*100:.2f}cm, "
                    f"peak_force={peak_grip_force:.2f}N, "
                    f"both_contact={both_contact_count}, xy_drift={obj_xy_drift*100:.2f}cm")

        # v6.0 统一物理验证: 夹持段恒定距离 (抬升段手指中点-物体距离, 基准=抬升起始帧)
        from physics_utils import validate_carry, CARRY_DEV_TOL_M
        _s1_val = None
        if len(fingertip_dist_history) > _LIFT_START_FRAME + 2:
            _s1_ref = float(fingertip_dist_history[_LIFT_START_FRAME])
            _s1_devs = [abs(d - _s1_ref) for d in fingertip_dist_history[_LIFT_START_FRAME:]]
            _s1_max_dev = float(max(_s1_devs))
            _s1_val = dict(ref_dist_mm=_s1_ref * 1000, max_dev_mm=_s1_max_dev * 1000,
                           passed=bool(_s1_max_dev <= CARRY_DEV_TOL_M))
            logger.info(f"[PhysicsValidation][stage1] 恒定距离(抬升段 F{_LIFT_START_FRAME}→末): "
                        f"ref={_s1_val['ref_dist_mm']:.1f}mm, max_dev={_s1_val['max_dev_mm']:.1f}mm "
                        f"(tol={CARRY_DEV_TOL_M*1000:.0f}mm) → {'PASS ✓' if _s1_val['passed'] else 'FAIL ✗'}")

        # 2026-08-31: 偏差② 跟随误差 — 实际基座 vs 本 stage 目标轨迹 (同基准)
        from physics_utils import compute_tracking_error
        _s1_te = compute_tracking_error(np.array(_base_actual_traj_s1),
                                        np.array(_base_target_traj_s1),
                                        start_frame=0, end_frame=None)
        logger.info(f"[Stage1 偏差②] 跟随误差 (实际基座 vs 目标轨迹, 同基准): "
                    f"mean={_s1_te['mean_mm']:.1f}mm, max={_s1_te['max_mm']:.1f}mm, "
                    f"final={_s1_te['final_mm']:.1f}mm (n={_s1_te['n_frames']})")

        return {
            'carry_validation': _s1_val,
            'tracking_error': _s1_te,
            'entry_target_contact': bool(getattr(self, '_last_entry_target_contact', False)),
            'contact_count': contact_count,
            'both_contact_count': both_contact_count,
            'obj_lift': obj_lift_peak,  # 用峰值抬升 (物体可能中途掉落)
            'obj_lift_final': obj_lift,  # 最终抬升 (保守值)
            'obj_xy_drift': obj_xy_drift,
            'grip_stability': contact_count / max(n_frames, 1),
            'obj_z_final': obj_z_final,
            'obj_init_z': obj_init_z,
            'obj_z_history': obj_z_history,
            'peak_grip_force': peak_grip_force,
            'locked_gripper_qpos': (locked_val1 + locked_val2) / 2.0,
            'fingertip_obj_dist': fingertip_obj_dist,
            'fingertip_obj_dist_min': fingertip_obj_dist_min,
        }

    # === stage1_reward (L5574) ===
    def stage1_reward(self, sim_result):
        """v5.0: 统一动作模板下的 reward — 只问一件事:
        "物体最终有没有和夹爪一起竖直提升 20cm?" (用户指定, 代价很大)

        动作模板固定无自由度 (下降→闭合→竖直提起 20cm), Stage1 唯一变量是 F50 位姿,
        因此 reward 不需要奖励接近过程, 只需区分三种结局:
          A. 物体跟随夹爪一起提升 (真抓取)         -> 高分 (主导项, 梯度贯穿 0~20cm)
          B. 手指碰到物体但物体没跟上 (压住/滑落)    -> 低分
          C. 物体被推飞 (竖直提升时 xy 大幅漂移)     -> 罚分

        与 v4.37 的区别:
          - lift 主导项: min(20, lift*400) (3.2cm 即拿满 20+8 分) → 40*min(1, lift/0.20),
            2cm 只值 4 分 (旧版 11 分), 提满 20cm 才拿满 — "碰一下就得分"的根源被移除
          - 距离项降级为"夹爪中心抓取" gate (保证 Stage1 输出中心抓取位姿)
          - xy_drift 在竖直提升范式下语义 = 推飞检测: 正常抓取物体 xy 应基本不动
          - 新增中途滑落检测: 峰值远高于终值 = 提起后掉落
        """
        lift_final = sim_result['obj_lift_final']
        lift_peak = sim_result.get('obj_lift', 0.0)
        dist = sim_result.get('fingertip_obj_dist', 1.0)
        both_contact = sim_result.get('both_contact_count', 0)
        xy_drift = sim_result['obj_xy_drift']
        peak_force = sim_result.get('peak_grip_force', 0.0)

        r = 0.0

        # ── 可迁移 anchor 门 (v51 §3.1): 规范入口 q=0.05 时不得已与目标接触 ──
        # 否则该 anchor 在 Stage2 的规范入口会穿模弹飞物体 (T1 实测 0/3).
        if sim_result.get('entry_target_contact', False):
            r -= 20.0

        # ── 主导项: 跟随竖直提升 20cm (线性梯度贯穿全程, 每 1cm +2 分) ──
        r += 40.0 * float(min(1.0, max(0.0, lift_final) / 0.20))
        if lift_final >= 0.15:
            r += 10.0  # ≥15cm 强奖励: 稳定夹持不滑落

        # ── gate 1: 双侧同时夹持 (硬约束的软化表达) ──
        # 提起且双侧接触 ≥10 帧 (0.25s): 真抓取; 接触却没提起: 压住不夹
        if both_contact >= 10 and lift_final > 0.05:
            r += 6.0
        elif both_contact > 0 and lift_final <= 0.005:
            r -= 6.0

        # ── gate 2: 夹爪中心抓取 (手指中点-物体中心距离) ──
        r += 4.0 * float(np.exp(-dist / 0.02))

        # ── 推飞检测: 竖直提升范式下物体 xy 应基本不动 (<1.5cm) ──
        if lift_final > 0.02:
            if xy_drift > 0.03:
                r -= 15.0   # 提起来了但物体被横向推飞 -> 铲起不是抓取
            elif xy_drift > 0.015:
                r -= 6.0
            else:
                r += 3.0    # 干净的竖直提升
        else:
            if xy_drift > 0.05:
                r -= 8.0    # 没提起还把物体推走

        # ── 中途滑落检测: 峰值远高于终值 = 提起后又掉落 ──
        if lift_peak > 0.08 and lift_final < 0.5 * lift_peak:
            r -= 8.0

        # ── 力 sanity: 有夹持力但不爆表 ──
        if 5.0 < peak_force < 200.0:
            r += 1.0

        return float(r)

    # === cem_grasp_pose_optimize (L5621) ===
    def cem_grasp_pose_optimize(self, side, n_iterations=15, population_size=32,
                                  top_k=8):
        # 2026-09-06 (用户): 固定随机种子 — CEM 可复现; 种子随重试轮次偏移 (轮间不同, 轮内一致)
        import random as _random
        _cem_seed = 42 + 1000 * int(getattr(self, '_cem_retry_count', 0))
        np.random.seed(_cem_seed)
        _random.seed(_cem_seed)
        logger.info(f"[Stage1] CEM 随机种子 = {_cem_seed} (固定, 可复现)")
        """Stage 1 抓取姿态优化: 分层搜索架构 (v4.5)

        流程:
        1. Stage 0: MANO 局部扰动生成候选 (无仿真)
        2. Stage 1a: Grasp Quality 预筛选 (无仿真, MANO 贴合硬约束)
        3. Stage 1b: Top-K 候选进入短物理验证 (pd_then_lock)
        4. Stage 1c: CMA-ES 微调 6DOF (以 Top-1 为起点)

        输出: dict with pos, R, euler, gripper_qpos, strategy='pd_then_lock'
        """
        from scipy.spatial.transform import Rotation as R_scipy
        from data_loader import rotmat_to_zyx_euler

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        if target_name is None or target_name not in self.obj_bbox_centers:
            if self.obj_bbox_centers:
                target_name = list(self.obj_bbox_centers.keys())[0]
            else:
                logger.error("[Stage1] 无目标物体")
                return None

        obj_pos = np.array(self.obj_bbox_centers[target_name], dtype=np.float64)

        # 获取 MANO F50 位姿 (仅用于信息展示和软约束奖励)
        mano_f50_pos, mano_f50_R = self._get_mano_f50_pose(side)
        if mano_f50_pos is None:
            mano_f50_pos = obj_pos.copy()
            mano_f50_pos[2] += 0.03689
        if mano_f50_R is None:
            mano_f50_R = np.eye(3)
        logger.info(f"[Stage1] MANO F50 参考: pos={np.asarray(mano_f50_pos).round(4)}, "
                    f"仅作姿态参考, xyz 不约束")
        logger.info(f"[Stage1] MANO F50 旋转: X={mano_f50_R[:,0].round(3)}, "
                    f"Y={mano_f50_R[:,1].round(3)}, Z={mano_f50_R[:,2].round(3)}")

        # v6.9b 诊断: 打印 MANO 轨迹全链路绝对值 (定位 CEM 参考位姿偏离物体的根因)
        try:
            _traj = self._mano_gripper_traj.get(side, {})
            _p = np.asarray(_traj.get("pos", []))
            _Rt = np.asarray(_traj.get("R", []))
            _no = getattr(self, '_mano_neutral_offset', None)
            _no_s = _no.get(side) if isinstance(_no, dict) else _no
            _off = np.zeros(3) if _no_s is None else np.asarray(_no_s).ravel()
            if len(_p) > 50 and len(_Rt) > 50:
                _fw = np.array([0.03689, 0.0, 0.0])
                _fm = _p[50] + _Rt[50] @ _fw
                logger.info(f"[Stage1 诊断] 链路绝对值: traj.pos[F50]={np.round(_p[50],4)} | "
                            f"neutral_offset={np.round(_off,4)} | "
                            f"finger_mid=pos+R@[0.03689,0,0]={np.round(_fm,4)} | "
                            f"物体={np.round(obj_pos,4)} | X.z={_Rt[50][2,0]:+.3f} | "
                            f"指尖-物体={np.round(_fm - obj_pos,4)} | n={len(_p)}")
        except Exception as _e:
            logger.info(f"[Stage1 诊断] 链路打印失败: {_e}")

        # ---- 夹爪开合度根据物体尺寸计算 ----
        obj_bbox_size = self.obj_info.get(target_name, {}).get('bbox_size', None)
        computed_gripper_qpos = self.compute_gripper_qpos(obj_bbox_size)
        logger.info(f"[Stage1] 目标 {target_name}, bbox={obj_bbox_size}, "
                    f"computed_gripper_qpos={computed_gripper_qpos:.4f}")

        def _check_mano_hard_constraint(pos, R, ref_R_for_constraint=None):
            """MANO 姿态硬约束: 相对 MANO F50 姿态的旋转差 ≤ 5° (xyz 位置不约束).

            v4.7: 严格按 plan L155/L224 (≤5°), 不放宽到 30°.
            用户反馈: "约束也改成那么多, 你觉得正确吗? F1-F45 z保持hover_z 这个都违背跟随mano参数了"
            遵循 MANO 姿态参考, 不强行使用竖直下抓姿态.
            """
            ref = ref_R_for_constraint if ref_R_for_constraint is not None else R
            try:
                R_diff = R @ np.asarray(ref).T
                rotvec = R_scipy.from_matrix(R_diff).as_rotvec()
                angle_diff = float(np.linalg.norm(rotvec))
                rot_ok = abs(angle_diff) <= np.radians(5.0)
            except Exception:
                rot_ok = True
            return rot_ok

        def _eval_pose(grasp_pose):
            """评估一个抓取姿态 (统一使用 pd_then_lock 策略)."""
            # v5.0 统一动作模板: 160 帧 = 40 帧接近闭合 + 120 帧竖直抬升 (提满 20cm),
            # reward 以"物体是否跟随夹爪一起提升"为主导, 从而判断该 grasp_pose 是否中心抓取.
            sim_result = self.rollout_grasp_only(grasp_pose, side, strategy='pd_then_lock', n_frames=160)
            r = self.stage1_reward(sim_result)
            # 保存 enclose_z 到 sim_result (Stage2/3 需要知道手指接触物体时的 base z)
            sim_result['enclose_z'] = getattr(self, '_last_rollout_enclose_z', None)
            return r, sim_result

        # ===== Stage 0: 候选生成 (以物体位置为基准, MANO F50 姿态为参考) =====
        logger.info("[Stage1] ===== Stage 0: 生成候选 =====")
        candidates = self.generate_grasp_candidates(side, n_candidates=240, base_R=mano_f50_R)
        if not candidates:
            logger.error("[Stage1] Stage 0 未生成候选")
            return None

        # ===== Stage 1a: Grasp Quality 预筛选 (无仿真) =====
        logger.info("[Stage1] ===== Stage 1a: Grasp Quality 预筛选 =====")
        scored_candidates = []
        for cand in candidates:
            q_score, hard_ok, details = self.score_grasp_quality(cand, target_name)
            mano_ok = _check_mano_hard_constraint(cand['pos'], cand['R'], ref_R_for_constraint=mano_f50_R)
            passed = hard_ok and mano_ok
            scored_candidates.append({
                'cand': cand,
                'q_score': q_score,
                'hard_ok': hard_ok,
                'mano_ok': mano_ok,
                'details': details,
                'passed': passed,
            })
            logger.info(f"  cand dx={cand.get('intended_finger_offset', [0,0,0])[0]:.4f}, "
                        f"dy={cand.get('intended_finger_offset', [0,0,0])[1]:.4f}, "
                        f"q_score={q_score:.2f}, hard={hard_ok}, mano={mano_ok}, pass={passed}")

        passed_candidates = [sc for sc in scored_candidates if sc['passed']]
        if not passed_candidates:
            logger.warning("[Stage1] 无候选通过硬约束, 放宽条件取 q_score 最高的 Top-K")
            passed_candidates = sorted(scored_candidates, key=lambda x: x['q_score'], reverse=True)[:top_k]

        # 按 q_score 排序取 Top-K 进入物理验证
        top_candidates = sorted(passed_candidates, key=lambda x: x['q_score'], reverse=True)[:top_k]
        logger.info(f"[Stage1] {len(top_candidates)}/{len(candidates)} 候选进入 Stage 1b")

        # ===== Stage 1b: Top-K 短物理验证 =====
        logger.info("[Stage1] ===== Stage 1b: Top-K 短物理验证 (pd_then_lock) =====")
        best_output = None
        best_reward = -float('inf')
        best_result = None
        for sc in top_candidates:
            cand = sc['cand']
            r, sim_result = _eval_pose(cand)
            logger.info(f"  Top-K cand r={r:.3f}, lift={sim_result['obj_lift']*100:.2f}cm, "
                        f"drift={sim_result['obj_xy_drift']*100:.2f}cm, "
                        f"pos={cand['pos'][:3].round(4)}")
            if r > best_reward:
                best_reward = r
                best_output = {
                    'pos': cand['pos'].copy(),
                    'R': cand['R'].copy(),
                    'euler': rotmat_to_zyx_euler(cand['R']),
                    'gripper_qpos': computed_gripper_qpos,
                    'strategy': 'pd_then_lock',
                    'force_limit': float(GRASP_STRATEGIES['pd_then_lock']['force_limit']),
                    'reward': best_reward,
                    'ref_base_R': cand.get('ref_base_R', cand['R'].copy()),
                    'enclose_z': sim_result.get('enclose_z'),
                }
                best_result = sim_result

        logger.info(f"[Stage1] Stage 1b 最优 reward={best_reward:.3f}, "
                    f"pos={best_output['pos'].round(4) if best_output else None}")

        if best_output is None:
            logger.error("[Stage1] Stage 1b 未产生有效结果")
            return None

        # ===== Stage 1c: CMA-ES 微调 6DOF =====
        logger.info("[Stage1] ===== Stage 1c: CMA-ES 微调 6DOF =====")
        dim = 6
        # 固定参考位姿 (Stage 1b 最优), 搜索空间为相对该参考的 6DOF 偏移
        ref_pos = best_output['pos'].copy()
        ref_R = best_output['R'].copy()
        mean = np.zeros(dim)
        std = np.array([0.008, 0.008, 0.004, 0.015, 0.015, 0.015])
        elite_frac = 0.25

        for iteration in range(n_iterations):
            samples = np.random.randn(population_size, dim) * std + mean
            rewards = []
            for i in range(population_size):
                dx, dy, dz, dry, drx, drz = samples[i]
                pos = ref_pos + np.array([dx, dy, dz])
                delta_rot = R_scipy.from_euler("xyz", [drx, dry, drz]).as_matrix()
                R_actual = delta_rot @ ref_R

                # MANO 硬约束: 相对 MANO F50 姿态的旋转差不超过 5°
                if not _check_mano_hard_constraint(pos, R_actual, ref_R_for_constraint=mano_f50_R):
                    rewards.append(-50.0)
                    continue


                grasp_pose = {'pos': pos, 'R': R_actual}
                r, sim_result = _eval_pose(grasp_pose)
                rewards.append(r)

                if r > best_reward:
                    best_reward = r
                    best_output = {
                        'pos': pos.copy(),
                        'R': R_actual.copy(),
                        'euler': rotmat_to_zyx_euler(R_actual),
                        'gripper_qpos': computed_gripper_qpos,
                        'strategy': 'pd_then_lock',
                        'force_limit': float(GRASP_STRATEGIES['pd_then_lock']['force_limit']),
                        'reward': best_reward,
                        'ref_base_R': mano_f50_R.copy(),
                        'enclose_z': sim_result.get('enclose_z'),
                    }
                    best_result = sim_result

            rewards = np.array(rewards)
            n_elite = max(1, int(population_size * elite_frac))
            elite_indices = np.argsort(rewards)[-n_elite:]
            elite_samples = samples[elite_indices]
            mean = np.mean(elite_samples, axis=0)
            std = np.std(elite_samples, axis=0) + 1e-4

            # 防止均值漂出 MANO 可行域: 若均值位姿违反硬约束, 重置到参考位姿并收缩搜索
            mean_pos = ref_pos + mean[:3]
            mean_R = R_scipy.from_euler("xyz", mean[3:6]).as_matrix() @ ref_R
            if not _check_mano_hard_constraint(mean_pos, mean_R, ref_R_for_constraint=mano_f50_R):
                logger.info(f"[Stage1] CMA-ES iter {iteration}: 均值漂出可行域, 重置到参考位姿")
                mean = np.zeros(dim)
                std = np.minimum(std * 0.8, np.array([0.005, 0.005, 0.003, 0.015, 0.015, 0.015]))

            logger.info(f"[Stage1] CMA-ES iter {iteration}/{n_iterations}: "
                        f"mean={np.mean(rewards):.3f}, best={best_reward:.3f}, "
                        f"elite_mean={np.mean(rewards[elite_indices]):.3f}, "
                        f"std_pos={std[:3].round(5)}, std_rot={std[3:6].round(5)}")

        # ===== Stage 1d: 纯平移细网格精修 (v4.37) =====
        # 用户思路: 姿态锁定 MANO (正确), 抓取成败取决于位置. CMA-ES 是随机搜索,
        # 收敛后在最优位置周围做确定性平移扫描收尾:
        #   gripper 局部 close 轴 ±4mm / fwd 轴 ±4mm × world z ±4mm, 步长 2mm,
        #   共 124 个纯平移候选 (旋转不变, 天然满足 MANO 硬约束).
        logger.info("[Stage1] ===== Stage 1d: 纯平移细网格精修 (±4mm step 2mm) =====")
        _pol_R = best_output['R'].copy()
        _pol_close_ax = _pol_R @ np.array([0.0, 1.0, 0.0])
        _pol_fwd_ax = _pol_R @ np.array([1.0, 0.0, 0.0])
        _pol_ref_pos = best_output['pos'].copy()
        _pol_best_r = best_reward
        _pol_n_better = 0
        for dcl in (-0.004, -0.002, 0.0, 0.002, 0.004):
            for dfw in (-0.004, -0.002, 0.0, 0.002, 0.004):
                for dz in (-0.004, -0.002, 0.0, 0.002, 0.004):
                    if dcl == 0.0 and dfw == 0.0 and dz == 0.0:
                        continue
                    p_pol = _pol_ref_pos + _pol_close_ax * dcl + _pol_fwd_ax * dfw \
                        + np.array([0.0, 0.0, dz])
                    gp_pol = {'pos': p_pol, 'R': _pol_R.copy(),
                              'gripper_qpos': best_output['gripper_qpos']}
                    r_pol, sr_pol = _eval_pose(gp_pol)
                    if r_pol > _pol_best_r:
                        _pol_best_r = r_pol
                        _pol_n_better += 1
                        best_reward = r_pol
                        best_result = sr_pol
                        best_output = {
                            'pos': p_pol.copy(), 'R': _pol_R.copy(),
                            'euler': np.array(best_output['euler']),
                            'gripper_qpos': best_output['gripper_qpos'],
                            'strategy': 'pd_then_lock',
                            'force_limit': float(GRASP_STRATEGIES['pd_then_lock']['force_limit']),
                            'enclose_z': sr_pol.get('enclose_z'),
                        }
                        logger.info(f"[Stage1] Stage1d 改进: d_close={dcl*1000:+.0f}mm "
                                    f"d_fwd={dfw*1000:+.0f}mm dz={dz*1000:+.0f}mm -> "
                                    f"reward={r_pol:.3f}")
        logger.info(f"[Stage1] Stage1d 完成: 扫描 124 个平移候选, "
                    f"改进 {_pol_n_better} 次, 最终 reward={best_reward:.3f}")

        logger.info(f"[Stage1] 最终最优: pos={best_output['pos'].round(4)}, "
                    f"euler={np.degrees(best_output['euler']).round(2)}, "
                    f"gripper_qpos={best_output['gripper_qpos']:.4f}, "
                    f"reward={best_reward:.3f}")

        # 关键修复: pos[2] 从 hover_z (起始悬停高度) 改为 enclose_z (实际夹取时 base z)
        # Stage1 rollout 内部从 hover_z 下降到 enclose_z 夹取, 但输出应代表"夹取位姿",
        # 即手指接触物体时 base 的位置 (enclose_z), 而非起始悬停位置 (hover_z).
        # Stage2/3 使用此 pos 作为 F50 的目标位姿.
        _enclose_z_val = best_output.get('enclose_z')
        # 2026-09-06 (用户): 移除 enclose_z 覆盖 — 中选位姿的高度已经过 CEM 物理验证
        # (实测 160/160 帧双侧接触, 23.7cm lift), 用最后一次 rollout 的推算 enclose_z
        # 覆盖会把高度改低 ~1cm, 终选门在错误高度上跑 (实测 0/3, 手指低于物体空合)。
        # 物理验证过的高度就是正确高度, 不覆盖。

        # ============================================================
        # v6.1 终选验证门: 单次 rollout 有混沌方差 (纯物理下边际抓取
        # "侥幸得高分"的 survivor bias), 中选 grasp 必须 3 次重复验证,
        # ≥2 次 lift_final ≥ 3cm 才交付; 否则重跑一轮 CEM (最多 2 次).
        # 实测案例: reward=62.17 的中选位姿, 验证 rollout 仅抬 0.56cm、
        # 推飞 15cm (2026-08-30 rerun) — 单次评估不可信.
        # ============================================================
        _retry = int(getattr(self, '_cem_retry_count', 0))
        _v_lifts = []
        for _k in range(3):
            _vr = self.rollout_grasp_only(
                {'pos': best_output['pos'].copy(), 'R': best_output['R'].copy()},
                side, strategy='pd_then_lock', n_frames=160)
            _v_lifts.append(float(_vr.get('obj_lift_final', 0.0)))
        _n_pass = sum(1 for _l in _v_lifts if _l >= 0.03)
        logger.info(f"[Stage1][验证门] 终选位姿 3 次重复: lift_final="
                    f"{[round(_l*100, 1) for _l in _v_lifts]}cm → {_n_pass}/3 通过 (门槛 2)")

        best_output['verification'] = {'lifts_final_cm': [round(_l * 100, 1) for _l in _v_lifts],
                                       'n_pass': _n_pass, 'gate': 'lift_final>=3cm, >=2/3'}

        if _n_pass < 2 and _retry < 2:
            logger.warning(f"[Stage1][验证门] 未通过 (第 {_retry+1} 次), 重跑 CEM "
                           f"(方差: 边际抓取单次评估侥幸得分)")
            self._cem_retry_count = _retry + 1
            try:
                _res = self.cem_grasp_pose_optimize(side, n_iterations=n_iterations,
                                                    population_size=population_size,
                                                    top_k=top_k)
            finally:
                self._cem_retry_count = _retry
            # 2026-09-06 (用户): 成绩好的保留, 不好的退出 — 跨轮次比较验证门成绩,
            # 重试轮若不如本轮则保留本轮候选 (旧实现无条件被重试轮覆盖)
            if _res is not None and _res.get('verification'):
                _r_v = _res['verification']
                _r_key = (_r_v.get('n_pass', 0), max(_r_v.get('lifts_final_cm', [0])))
                _c_key = (_n_pass, max(_v_lifts) * 100 if _v_lifts else 0)
                if _r_key > _c_key:
                    logger.info(f"[Stage1][验证门] 重试轮更优 (n_pass={_r_key[0]}/3, "
                                f"best={_r_key[1]:.1f}cm > 本轮 {_c_key[1]:.1f}cm) → 采用重试轮")
                    return _res
            logger.info(f"[Stage1][验证门] 本轮候选更优, 保留 (n_pass={_n_pass}/3)")
        elif _n_pass < 2:
            logger.error(f"[Stage1][验证门] 重试 2 轮仍未通过 — 本数据/姿态组合的物理抓取 "
                         f"不稳定, best_grasp 带警告交付 (lifts={_v_lifts})")

        # 保存 Stage1 最优数据供 Stage2/Stage3 参考
        try:
            import json as _json

            _save_dir = self.output_dir / "stage1"
            _save_dir.mkdir(parents=True, exist_ok=True)
            _save_path = _save_dir / "best_grasp.npz"
            # v51 §3.1/§3.5: 导出 stage1_grasp_recipe — Stage2 只读该配方, 保证
            # 被抓住物体的 scale/位置与闭合命令两阶段逐字段一致 (不得重新硬编码).
            _recipe = self._export_stage1_recipe(target_name, best_output)
            _recipe_json = _json.dumps(_recipe, sort_keys=True) if _recipe else ""
            if _recipe is not None:
                # 同进程内 Stage2/Stage3 直接复用该 recipe, 无需从磁盘回读.
                best_output['stage1_recipe'] = _recipe
            np.savez(
                _save_path,
                pos=best_output['pos'],
                R=best_output['R'],
                euler=best_output['euler'],
                gripper_qpos=best_output['gripper_qpos'],
                strategy=best_output['strategy'],
                force_limit=best_output['force_limit'],
                reward=best_reward,
                ref_base_R=best_output.get('ref_base_R', best_output['R']),
                obj_lift=best_result['obj_lift'] if best_result else 0.0,
                obj_xy_drift=best_result['obj_xy_drift'] if best_result else 0.0,
                peak_grip_force=best_result['peak_grip_force'] if best_result else 0.0,
                locked_gripper_qpos=best_result['locked_gripper_qpos'] if best_result else best_output['gripper_qpos'],
                enclose_z=best_output.get('enclose_z', 0.0),
                verification=best_output['verification'],
                target_object=str(target_name),
                q_open=float(GRIPPER_MAX_OPEN),
                close_controls=int(STAGE1_CLOSE_CONTROLS),
                settle_controls=int(STAGE1_SETTLE_CONTROLS),
                scene_digest_hash=str(_recipe.get('scene_digest_hash', '')) if _recipe else '',
                stage1_recipe_json=_recipe_json,
            )
            if _recipe:
                (_save_dir / "stage1_grasp_recipe.json").write_text(
                    _json.dumps(_recipe, indent=2, sort_keys=True), encoding="utf-8"
                )
            logger.info(f"[Stage1] 最优 grasp 数据已保存到 {_save_path}")
        except Exception as _e:
            logger.warning(f"[Stage1] 保存最优 grasp 数据失败: {_e}")

        return best_output

    # === _export_stage1_recipe ===
    def _export_stage1_recipe(self, target_name, best_output):
        """Build the transferable Stage1 recipe + scene/object digest.

        The digest pins the grasped object's scale (``bbox_size`` after
        ``_OBJ_SCALE``), its reset pose, mass, ground and solver parameters so
        Stage2 can prove it is replaying the same physical anchor.
        """
        import json as _json

        if not target_name:
            return None
        q_grasp = float(best_output['gripper_qpos'])
        q_grasp_actual = getattr(self, '_last_grip_target_s1', q_grasp)
        if abs(float(q_grasp_actual) - q_grasp) > 1e-9:
            logger.warning(
                "[Stage1][recipe] 记录 gripper_qpos=%.4f 与实际命令 PD_target=%.4f 不一致 "
                "(通常来自 _S1_GRIP_DELTA); recipe 采用前者 (Stage2 anchor)",
                q_grasp, float(q_grasp_actual),
            )
        physics_config = build_physics_config()
        gravity = [0.0, 0.0, -9.81]
        try:
            gravity = [float(v) for v in self.scene.get_gravity()]
        except Exception:
            pass
        digest = build_scene_digest(
            target_name=target_name,
            obj_info=getattr(self, 'obj_info', {}) or {},
            obj_initial_poses=getattr(self, '_obj_initial_poses', None) or {},
            obj_actors=getattr(self, 'obj_actors', []) or [],
            ground_z=float(getattr(self, '_ground_z', 0.0)),
            gravity=gravity,
            physics_config=physics_config,
        )
        recipe = build_stage1_grasp_recipe(
            target_name=target_name,
            q_open=stage_q_open(),
            q_grasp=q_grasp,
            physics_config=physics_config,
            scene_digest=digest,
        )
        logger.info(
            "[Stage1][recipe] target=%s bbox=%s init_pos=%s mass=%s q_open=%.4f "
            "q_grasp=%.4f close/settle=%d/%d digest=%s",
            target_name,
            digest.get('bbox_size'),
            digest.get('initial_pose_pos'),
            digest.get('mass'),
            recipe['q_open'],
            recipe['q_grasp'],
            recipe['close_controls'],
            recipe['settle_controls'],
            recipe['scene_digest_hash'][:12],
        )
        return recipe
