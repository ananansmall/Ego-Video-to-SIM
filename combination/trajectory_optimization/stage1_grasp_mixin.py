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
    GRASP_STRATEGIES, GRIPPER_INIT_OPEN, MAX_ROOT_STEP,
    OBJECT_MIN_MASS, OBJECT_DENSITY, GRIPPER_FRICTION,
    _DESCEND_OPEN,
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

        # 候选网格: 物体为中心, xy 范围扩大 ±1.5cm, z 范围 ±0.5cm
        candidates = []
        xy_step = 0.005     # ±0.75cm 步进
        z_step = 0.005      # ±0.5cm 步进
        ry_step = 0.02      # ±~1.1deg 姿态微调
        n_xy = 4
        n_z = 2
        n_ry = 3
        for i in range(min(n_candidates, n_xy * n_xy * n_z * n_ry)):
            ix = i % n_xy
            iy = (i // n_xy) % n_xy
            iz = (i // (n_xy * n_xy)) % n_z
            iry = (i // (n_xy * n_xy * n_z)) % n_ry
            dx = (ix - (n_xy - 1) / 2.0) * xy_step
            dy = (iy - (n_xy - 1) / 2.0) * xy_step
            dz = (iz - (n_z - 1) / 2.0) * z_step
            dry = (iry - (n_ry - 1) / 2.0) * ry_step
            delta_pos = np.array([dx, dy, dz], dtype=np.float64)
            pos = base_pos + delta_pos
            if pos[2] < obj_pos[2]:
                continue  # base 不能低于物体中心
            # 姿态扰动绕 Y 轴 (开合方向) 微调
            R = base_R.copy()
            if abs(dry) > 1e-6:
                R = R_scipy.from_euler("y", dry, degrees=False).as_matrix() @ R
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
    def rollout_grasp_only(self, grasp_pose, side, strategy='pd_then_lock', n_frames=80, record_video=False):
        """Stage 1 短仿真: 验证 F50 grasp_pose 能不能夹住并抬升物体。

        v4.8 策略 (80 帧):
        - F0-F10:   从上方 3cm 下降到 hover_z, 手指保持中张并锁定防擦地
        - F10-F30:  边下降到 enclose_z 边快速闭合到预紧点 (_PRELOAD_QPOS)
        - F30-F40:  在 enclose_z 维持夹持, 等待双侧接触稳定建立
        - F40-F80:  缓慢抬升 base, 边抬升边继续闭合到 _GRIP_TARGET, 验证跟随

        Args:
            grasp_pose: dict with 'pos'(3,) and 'R'(3x3)
            side: 'right' or 'left'
            n_frames: 仿真帧数 (默认 80, 让完整 40 帧抬升生效)
            record_video: 是否录制视频 (需要 cameras 已由 run() 初始化)

        Returns:
            dict with 'contact_count', 'both_contact_count', 'obj_lift',
                      'obj_xy_drift', 'grip_stability', 'obj_z_final',
                      'peak_grip_force', 'locked_gripper_qpos'
        """
        from data_loader import rotmat_to_zyx_euler
        from physics_env import physics_step, get_finger_contacts, get_grasp_force

        # Stage 1 统一使用 pd_then_lock 策略
        force_limit = float(GRASP_STRATEGIES['pd_then_lock']['force_limit'])

        robot = self.robot_info["robot"]
        virtual_idx = self.robot_info.get("virtual_idx", {})
        gripper_idx1 = self.robot_info["gripper_idx1"]
        gripper_idx2 = self.robot_info["gripper_idx2"]
        arm_joint_indices = self.robot_info.get("arm_joint_indices", [])
        active_joints = robot.get_active_joints()

        # 设置手指 PD 参数 (Stage 1 用较大刚度+阻尼, 稳定夹持)
        for gidx in (gripper_idx1, gripper_idx2):
            if gidx is not None:
                active_joints[gidx].set_drive_property(
                    stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING,
                    force_limit=force_limit
                )

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None

        # 记录初始物体位姿
        if not hasattr(self, '_obj_initial_poses') or self._obj_initial_poses is None:
            self._obj_initial_poses = {actor.get_name(): actor.get_pose() for actor in self.obj_actors}

        # === 1. 重置场景 ===
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

        # Stage 1 短仿真高度策略:
        #   hover_z  = enclose_z + 0.021 (手指在物体上方约 21mm, 下降阶段不擦地)
        #   enclose_z= obj_z - finger_offset_z (gripper_base z 使手指到达物体中心高度)
        #   v4.7 关键修复: 手指偏移方向随 base_R 变化, enclose_z 不再固定为 obj_z+0.03689
        #   - base_R = _rot_close (ry=+90°): finger_offset_z = -0.03689, enclose_z = obj_z + 0.03689
        #   - base_R = mano_f50_R: finger_offset_z 随旋转变化, enclose_z 相应调整
        #   F0-F10   下降到 hover_z, 手指用 set_qpos 锁定防触地漂移
        #   F10-F30  边下降到 enclose_z 边闭合手指 (边走边夹)
        #   F30-F35  在 enclose_z 维持夹持, 等待接触稳定
        #   F35-F50  抬升验证
        _GRASP_R = np.asarray(grasp_pose.get('R', np.eye(3)), dtype=np.float64)
        _FINGER_OFFSET_GRIPPER = np.array([0.03689, 0.0, 0.0])
        _finger_offset_world = _GRASP_R @ _FINGER_OFFSET_GRIPPER
        _finger_offset_z = float(_finger_offset_world[2])  # 手指相对 base 的 Z 偏移 (世界坐标)
        # enclose_z: base 的 z 值, 使手指中心 z = obj_z
        # finger_z = base_z + _finger_offset_z → base_z = obj_z - _finger_offset_z
        _ENCLOSE_Z_OFFSET = -_finger_offset_z  # 对 _rot_close: = 0.03689
        _HOVER_Z_OFFSET = _ENCLOSE_Z_OFFSET + 0.021  # hover 在 enclose 上方 2.1cm
        _hover_z = obj_init_z + _HOVER_Z_OFFSET
        _enclose_z = obj_init_z + _ENCLOSE_Z_OFFSET
        self._last_rollout_enclose_z = _enclose_z  # 供 Stage1 保存到 best_output
        # z 在 _hover_z 附近 ±1cm 搜索 (避免手指触地也允许上下微调)
        _Z_SEARCH_RANGE = 0.01
        target_pos[2] = np.clip(target_pos[2], _hover_z - _Z_SEARCH_RANGE, _hover_z + _Z_SEARCH_RANGE)
        start_pos = target_pos.copy()
        start_pos[2] += 0.03  # 从 hover 上方 3cm 开始

        qpos = robot.get_qpos().copy()
        if virtual_idx:
            qpos[virtual_idx['vx']] = start_pos[0]
            qpos[virtual_idx['vy']] = start_pos[1]
            qpos[virtual_idx['vz']] = start_pos[2]
            qpos[virtual_idx['rz']] = rz
            qpos[virtual_idx['ry']] = ry
            qpos[virtual_idx['rx']] = rx
        qpos[gripper_idx1] = _DESCEND_OPEN
        qpos[gripper_idx2] = _DESCEND_OPEN
        robot.set_qpos(qpos)

        if virtual_idx:
            for vkey, vval in [('vx', start_pos[0]), ('vy', start_pos[1]),
                               ('vz', start_pos[2]), ('rz', rz), ('ry', ry), ('rx', rx)]:
                active_joints[virtual_idx[vkey]].set_drive_target(float(vval))
        active_joints[gripper_idx1].set_drive_target(_DESCEND_OPEN)
        active_joints[gripper_idx2].set_drive_target(_DESCEND_OPEN)

        # warmup
        for _ in range(5):
            _q = robot.get_qpos()
            physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                         np.array([]), float(_q[gripper_idx1]), float(_q[gripper_idx2]),
                         self.scene)

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
                logger.info(f"[Stage1] 目标物体 {_bbox_max*100:.2f}cm, "
                            f"接触 qpos={_q_contact:.4f}, compute_gripper_qpos={_computed_target:.4f}, "
                            f"PD_target={_GRIP_TARGET:.4f} (基于物体尺寸, 略小于接触点产生正压力), "
                            f"hover_z={_hover_z:.4f}, enclose_z={_enclose_z:.4f}")

        # === 3. 仿真 ===
        contact_count = 0
        both_contact_count = 0
        obj_z_history = []
        peak_grip_force = 0.0
        _DESCEND_FRAMES = 10    # F0-F10:  下降到 hover_z, 手指锁定
        _CLOSE_FRAMES = 20      # F10-F30: 边下降到 enclose_z 边闭合
        _SETTLE_FRAMES = 10     # F30-F40: 在 enclose_z 等待接触稳定
        _LIFT_FRAMES = 40       # F40-F80: 更缓慢抬升验证
        _LIFT_SPEED = 0.0004    # 每帧抬升 0.4mm, 40帧 → 1.6cm (降低速度防滑落)

        # Stage 1: 暂时禁用手指-地面碰撞, 让长手指能真正闭合小物体
        # (仿真环境对齐, 但 R1 手指 mesh 过长, 对小物体必然触地, 导致无法夹紧)
        _ground_actor = None
        _orig_groups = []  # [(shape, original_groups), ...]
        _FINGER_IGNORE_ID = 1
        _FINGER_IGNORE_BIT = 1 << (_FINGER_IGNORE_ID - 1)
        try:
            for _actor in self.scene.get_all_actors():
                if _actor.get_name() == 'ground':
                    _ground_actor = _actor
                    break
            if _ground_actor is not None:
                for _comp in _ground_actor.get_components():
                    if isinstance(_comp, sapien.pysapien.physx.PhysxRigidStaticComponent):
                        for _cs in _comp.get_collision_shapes():
                            _orig_groups.append((_cs, list(_cs.get_collision_groups())))
                            _g = list(_cs.get_collision_groups())
                            _g[2] |= _FINGER_IGNORE_BIT
                            _g[3] = _FINGER_IGNORE_ID
                            _cs.set_collision_groups(_g)
                logger.info("[Stage1] 已禁用手指-地面碰撞 (仅本次 rollout)")
        except Exception as _e:
            logger.warning(f"[Stage1] 禁用手指-地面碰撞失败: {_e}")

        # v4.11 用户要求: 临时禁用邻物碰撞, 让手指能穿过邻物只夹 target_name
        # (粉色 glb_1 周围有 glb_0/glb_2 紧挨, 4cm 宽手指必然同时碰多物体, 必须禁用邻物碰撞)
        # 保留 target_name 自身碰撞, 其他 glb_ 物体全部禁用
        _NEIGHBOR_ORIG_GROUPS = []  # [(shape, original_groups), ...] 用于恢复
        try:
            if target_name is not None:
                for _actor in self.scene.get_all_actors():
                    _name = _actor.get_name()
                    if _name == target_name or not _name.startswith('glb_'):
                        continue
                    for _comp in _actor.get_components():
                        if hasattr(_comp, 'get_collision_shapes'):
                            for _cs in _comp.get_collision_shapes():
                                _NEIGHBOR_ORIG_GROUPS.append((_cs, list(_cs.get_collision_groups())))
                                _cs.set_collision_groups([0, 0, 0, 0])
                if _NEIGHBOR_ORIG_GROUPS:
                    logger.info(f"[Stage1] 已禁用邻物碰撞 ({len(_NEIGHBOR_ORIG_GROUPS)} 个 shape), "
                                f"target={target_name} (仅本次 rollout)")
        except Exception as _e:
            logger.warning(f"[Stage1] 禁用邻物碰撞失败: {_e}")

        # 闭合阶段: F10-F30 快速闭合到预紧点, F30-F80 边抬升边缓慢闭合,
        # 让抬升全过程都保持 PD 位置误差 → 持续正压力, 防止物体滑落.
        # v4.8: _PRELOAD_QPOS 必须 >= _GRIP_TARGET, 否则 F30-F80 会反向张开
        # _PRELOAD_QPOS 略大于 _GRIP_TARGET 0.002, 让 F30-F80 缓慢收敛到目标产生正压力
        _PRELOAD_QPOS = max(0.0040, _GRIP_TARGET + 0.002)  # 预紧点: 略大于目标, 让PD继续闭合
        _close1_t = np.linspace(0, 1, _CLOSE_FRAMES)
        _close1_smooth = _close1_t * _close1_t * (3 - 2 * _close1_t)
        _close2_frames = n_frames - _DESCEND_FRAMES - _CLOSE_FRAMES
        _close2_t = np.linspace(0, 1, max(1, _close2_frames))
        _close2_smooth = _close2_t * _close2_t * (3 - 2 * _close2_t)

        for frame in range(n_frames):
            qpos = robot.get_qpos().copy()

            # 位置: 下降(hover) → 边走边夹 → 维持 → 抬升
            if frame < _DESCEND_FRAMES:
                t = frame / _DESCEND_FRAMES
                s = t * t * (3 - 2 * t)
                current_pos = start_pos * (1 - s) + target_pos * s
            elif frame < _DESCEND_FRAMES + _CLOSE_FRAMES:
                # F10-F30: 边下降边快速闭合到预紧点
                _i = frame - _DESCEND_FRAMES
                _s = _close1_smooth[_i]
                current_pos = target_pos.copy()
                current_pos[2] = _hover_z * (1 - _s) + _enclose_z * _s
            elif frame < _DESCEND_FRAMES + _CLOSE_FRAMES + _SETTLE_FRAMES:
                current_pos = target_pos.copy()
                current_pos[2] = _enclose_z
            else:
                _lift_t = frame - (_DESCEND_FRAMES + _CLOSE_FRAMES + _SETTLE_FRAMES)
                current_pos = target_pos.copy()
                current_pos[2] = _enclose_z + _LIFT_SPEED * _lift_t

            if virtual_idx:
                qpos[virtual_idx['vx']] = current_pos[0]
                qpos[virtual_idx['vy']] = current_pos[1]
                qpos[virtual_idx['vz']] = current_pos[2]
                qpos[virtual_idx['rz']] = rz
                qpos[virtual_idx['ry']] = ry
                qpos[virtual_idx['rx']] = rx

            # 手指: 中张(下降阶段锁定) → F10-F30 快速预紧 → F30-F80 边抬升边缓慢闭合
            if frame < _DESCEND_FRAMES:
                gripper_target = _DESCEND_OPEN
                # 下降阶段用 set_qpos 锁定手指, 防止擦地导致 qpos 漂移/不对称
                qpos[gripper_idx1] = float(_DESCEND_OPEN)
                qpos[gripper_idx2] = float(_DESCEND_OPEN)
                robot.set_qpos(qpos)
                robot.set_qvel(np.zeros_like(robot.get_qvel()))
            elif frame < _DESCEND_FRAMES + _CLOSE_FRAMES:
                _i = frame - _DESCEND_FRAMES
                gripper_target = _DESCEND_OPEN - (_DESCEND_OPEN - _PRELOAD_QPOS) * _close1_smooth[_i]
            else:
                _i = frame - (_DESCEND_FRAMES + _CLOSE_FRAMES)
                gripper_target = _PRELOAD_QPOS - (_PRELOAD_QPOS - _GRIP_TARGET) * _close2_smooth[min(_i, _close2_frames - 1)]

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

            # v4.16: 录制视频 (如果已启用)
            if record_video:
                self.scene.update_render()
                if hasattr(self, '_cam_view') and self._cam_view is not None:
                    self._cam_view.take_picture()
                    rgb_cam = self._cam_view.get_picture("Color")[..., :3]
                    if hasattr(self, '_writer_cam') and self._writer_cam is not None:
                        bgr_cam = np.ascontiguousarray((np.clip(rgb_cam, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        self._writer_cam.write(bgr_cam)
                if hasattr(self, '_god_view') and self._god_view is not None:
                    self._god_view.take_picture()
                    rgb_god = self._god_view.get_picture("Color")[..., :3]
                    if hasattr(self, '_writer_god') and self._writer_god is not None:
                        bgr_god = np.ascontiguousarray((np.clip(rgb_god, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                        self._writer_god.write(bgr_god)

            # 接触检测
            if target_name is not None:
                f1_c, f2_c, contact_objs = get_finger_contacts(
                    robot, side, self.scene, self.obj_actors)
                if f1_c or f2_c:
                    if target_name in contact_objs:
                        contact_count += 1
                if f1_c and f2_c and target_name in contact_objs:
                    both_contact_count += 1

                for actor in self.obj_actors:
                    if target_name in actor.get_name():
                        obj_z_history.append(float(np.array(actor.get_pose().p)[2]))
                        break

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
        obj_z_final, obj_xy_drift = 0.0, 0.0
        if target_name is not None:
            for actor in self.obj_actors:
                if target_name in actor.get_name():
                    obj_final_pos = np.array(actor.get_pose().p)
                    obj_z_final = float(obj_final_pos[2])
                    obj_xy_drift = float(np.linalg.norm(obj_final_pos[:2] - obj_init_xy))
                    break

        obj_lift = max(0.0, obj_z_final - obj_init_z)
        # 峰值抬升: 抓取期间的物体最高z (物体可能在中途掉落, 最终z不代表真实抬升能力)
        _obj_peak_z = obj_init_z
        if len(obj_z_history) > 0:
            _obj_z_arr = np.array(obj_z_history)
            # v4.8 修复: F40-F80 为实际抬升区间 (与 _LIFT_FRAMES=40 一致)
            # 之前 F30-F60 错过 F60-F75 的主要抬升阶段, 导致 obj_lift_peak 偏低
            _grasp_start = min(_DESCEND_FRAMES + _CLOSE_FRAMES + _SETTLE_FRAMES, len(_obj_z_arr))
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

        # 恢复手指-地面碰撞设置
        try:
            for _cs, _g in _orig_groups:
                _cs.set_collision_groups(_g)
        except Exception as _e:
            logger.warning(f"[Stage1] 恢复碰撞组失败: {_e}")

        # v4.11: 恢复邻物碰撞设置
        try:
            for _cs, _g in _NEIGHBOR_ORIG_GROUPS:
                _cs.set_collision_groups(_g)
        except Exception as _e:
            logger.warning(f"[Stage1] 恢复邻物碰撞组失败: {_e}")

        return {
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
        }

    # === stage1_reward (L5574) ===
    def stage1_reward(self, sim_result):
        """Stage 1 奖励: 只看"能不能夹住并抬起来" (与 plan v4.5 对齐)

        核心指标: 提升高度; 辅助: 双侧接触、夹持力效率、推飞/掉落惩罚.
        对小幅抬升也给予正向反馈, 避免 CMA-ES 初期所有候选奖励为负。
        """
        lift = sim_result['obj_lift']
        contact = sim_result['contact_count']
        both_contact = sim_result.get('both_contact_count', 0)
        xy_drift = sim_result['obj_xy_drift']
        peak_force = sim_result.get('peak_grip_force', 10.0)
        obj_z_final = sim_result.get('obj_z_final', 0.0)

        r = 0.0
        # 1. 提升奖励 (核心): 连续奖励, 小抬升也有分
        r += min(5.0, lift * 100.0)        # 每 1cm 抬升 +1.0, 封顶 5.0
        if lift > 0.03:
            r += 3.0
        elif lift > 0.01:
            r += 1.0
        elif lift > 0.001:
            r += 0.2

        # 2. 接触奖励: 双侧同时接触更重要
        r += min(2.0, both_contact / 5.0) * 2.0
        r += min(1.0, contact / 10.0)

        # 3. 夹持力效率
        if lift > 0.005 and peak_force < 15.0:
            r += 1.0
        elif lift > 0.005 and peak_force >= 15.0:
            r -= 0.5

        # 4. 推飞/漂移惩罚 (分级, 不过度惩罚小漂移)
        if xy_drift > 0.05:
            r -= 5.0
        elif xy_drift > 0.02:
            r -= 1.0
        elif xy_drift > 0.01:
            r -= 0.3

        # 5. 穿透/掉落惩罚
        if obj_z_final < 0.001:
            r -= 3.0

        return float(r)

    # === cem_grasp_pose_optimize (L5621) ===
    def cem_grasp_pose_optimize(self, side, n_iterations=15, population_size=32,
                                  top_k=8):
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
            # Stage1 验证用 80 帧, 让 F40-F80 完整 40 帧抬升生效 (4.8cm),
            # 从而判断该 grasp_pose 是否能稳定夹持并跟随。
            sim_result = self.rollout_grasp_only(grasp_pose, side, strategy='pd_then_lock', n_frames=80)
            r = self.stage1_reward(sim_result)
            # 保存 enclose_z 到 sim_result (Stage2/3 需要知道手指接触物体时的 base z)
            sim_result['enclose_z'] = getattr(self, '_last_rollout_enclose_z', None)
            return r, sim_result

        # ===== Stage 0: 候选生成 (以物体位置为基准, MANO F50 姿态为参考) =====
        logger.info("[Stage1] ===== Stage 0: 生成候选 =====")
        candidates = self.generate_grasp_candidates(side, n_candidates=32, base_R=mano_f50_R)
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
        std = np.array([0.005, 0.005, 0.003, 0.015, 0.015, 0.015])
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

        logger.info(f"[Stage1] 最终最优: pos={best_output['pos'].round(4)}, "
                    f"euler={np.degrees(best_output['euler']).round(2)}, "
                    f"gripper_qpos={best_output['gripper_qpos']:.4f}, "
                    f"reward={best_reward:.3f}")

        # 关键修复: pos[2] 从 hover_z (起始悬停高度) 改为 enclose_z (实际夹取时 base z)
        # Stage1 rollout 内部从 hover_z 下降到 enclose_z 夹取, 但输出应代表"夹取位姿",
        # 即手指接触物体时 base 的位置 (enclose_z), 而非起始悬停位置 (hover_z).
        # Stage2/3 使用此 pos 作为 F50 的目标位姿.
        _enclose_z_val = best_output.get('enclose_z')
        if _enclose_z_val is not None and abs(_enclose_z_val) < 1.0:
            _old_z = best_output['pos'][2]
            best_output['pos'][2] = float(_enclose_z_val)
            logger.info(f"[Stage1] 输出 pos[2]: hover_z={_old_z:.4f} → enclose_z={_enclose_z_val:.4f} "
                        f"(手指接触物体时的 base z)")

        # 保存 Stage1 最优数据供 Stage2/Stage3 参考
        try:
            _save_dir = self.output_dir / "stage1"
            _save_dir.mkdir(parents=True, exist_ok=True)
            _save_path = _save_dir / "best_grasp.npz"
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
            )
            logger.info(f"[Stage1] 最优 grasp 数据已保存到 {_save_path}")
        except Exception as _e:
            logger.warning(f"[Stage1] 保存最优 grasp 数据失败: {_e}")

        return best_output
