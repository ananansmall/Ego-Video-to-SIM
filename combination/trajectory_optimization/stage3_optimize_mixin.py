"""stage3_optimize_mixin.py — Stage 3: 全局轨迹优化 (CEM + 平滑过渡)

从 grasp_hawor.py 抽出, 提供 Stage3Mixin 类。
Stage 3: 基于 Stage 2 重建轨迹, 用 CEM 优化 F1-F45 + F55-F89 段, 或 v4.7 平滑过渡
依赖主类 GraspSimulator 的以下属性/方法:
  - self._mano_gripper_traj, self._mano_neutral_offset
  - self._v4_stage2_recon (Stage 2 输出), self._frame_params, self._fixed_offsets_654
  - self.grasp_controllers, self.obj_actors, self.scene, self.robot_info
  - self._step_full_robot() (基础设施方法, 跨 stage 共享)
"""
import logging
import numpy as np
import sapien

logger = logging.getLogger("grasp_hawor")

from physics_env import (
    GRIPPER_STIFFNESS, GRIPPER_DAMPING, GRIPPER_FORCE,
    MAX_ROOT_STEP, GRIPPER_INIT_OPEN,
    DECIMATION, GRASP_STRATEGIES,
)


class Stage3Mixin:
    """Stage 3: 轨迹优化 Mixin"""

    # === stage3_reward (L6064) ===
    def stage3_reward(self, sim_result):
        """Stage 3 奖励: 分区间差异化权重

        v4.4 (与 plan 对齐):
        - F1~F45: 位置+姿态贴合 MANO
        - F46~F54: 锁定 Stage 2 输出, 不计算
        - F55~F90: 位置+姿态贴合 MANO (更高权重)
        - F91~F112: 完全跟随 MANO, 不计算
        - 全局: 平滑性 + 抓取成功 + 弹飞惩罚
        """
        from data_loader import rotation_distance

        opt_pos = sim_result.get('opt_pos', None)
        opt_R = sim_result.get('opt_R', None)
        mano_pos = sim_result.get('mano_pos', None)
        mano_R = sim_result.get('mano_R', None)

        r = 0.0
        # ===== 1. 位置/姿态跟踪 (分区间权重) =====
        if opt_pos is not None and opt_R is not None and mano_pos is not None and mano_R is not None:
            n_frames = min(len(opt_pos), len(mano_pos), len(mano_R))
            for f in range(1, min(91, n_frames)):
                if f < 46:
                    w_pos, w_rot = 1.0, 30.0
                elif f <= 54:
                    continue  # F46~F54 锁定, 不参与优化
                elif f <= 90:
                    w_pos, w_rot = 2.0, 40.0
                else:
                    continue

                pos_err = float(np.linalg.norm(opt_pos[f] - mano_pos[f]))
                r -= w_pos * pos_err ** 2

                rot_err = rotation_distance(opt_R[f], mano_R[f])
                r -= w_rot * (rot_err ** 2)

                # 帧间平滑惩罚 (二阶差分), 只在优化区间内部
                if 1 < f < 90:
                    accel = np.linalg.norm(opt_pos[f + 1] - 2 * opt_pos[f] + opt_pos[f - 1])
                    r -= 10.0 * accel ** 2

        # ===== 2. 抓取成功奖励 (v4.6: 大幅提高 lift/contact 权重) =====
        lift = sim_result.get('obj_lift', 0.0)
        contact = sim_result.get('contact_frames_in_close', 0)
        if lift > 0.03 and contact >= 3:
            r += 100.0
        elif lift > 0.03:
            r += 50.0
        elif lift > 0.01:
            r += 15.0
        elif lift > 0.005:
            r += 5.0

        # 接触帧奖励: 鼓励 CLOSE 阶段持续双侧/单侧接触
        r += min(20.0, contact * 2.0)

        # ===== 3. 物体跟随惩罚 (F42~F58 缓冲区) =====
        obj_follow_errors = sim_result.get('obj_follow_errors', [])
        for err in obj_follow_errors:
            r -= 500.0 * err ** 2

        # ===== 4. 弹飞/漂移惩罚 (v4.6: 强惩罚横向推开物体) =====
        xy_drift = sim_result.get('xy_drift', 0.0)
        r -= 200.0 * xy_drift ** 2  # 5cm drift -> -0.5
        if xy_drift > 0.2:
            r -= 10.0
        elif xy_drift > 0.05:
            r -= 2.0

        # 弹飞惩罚 (与 plan 对齐: 1000 * max_launch)
        max_launch = sim_result.get('max_launch', 0.0)
        r -= 1000.0 * max_launch

        # ===== 5. 穿透惩罚 =====
        pen = sim_result.get('max_penetration', 0.0)
        pen_pen = max(0.0, pen - 0.01)
        r -= pen_pen * 500.0

        return float(r)

    # === rollout_v4_stage3 (L6143) ===
    def rollout_v4_stage3(self, opt_params=None, side='right', decimation=None):
        """Stage 3 rollout: 平滑过渡 + 物理验证 (v4.7, 不做CEM优化)

        策略 (v4.7):
        - F1~F45:  跟随 MANO 轨迹 (无偏移)
        - F46~F55: 锁定 Stage 2 输出 (不优化)
        - F56~F89: 从 Stage 2 F55 位姿 smoothstep 过渡到 MANO
        - F90~F112: 跟随 MANO

        夹爪: 分阶段定死, F50-F95 保持夹持
        """
        from data_loader import rotmat_to_zyx_euler
        from physics_env import physics_step, get_finger_contacts

        traj = self._mano_gripper_traj.get(side, {})
        if traj is None or len(traj.get("pos", [])) == 0:
            logger.error("[Stage 3 rollout] 无 MANO 轨迹")
            return {'reward': -float('inf')}

        if decimation is None:
            decimation = max(1, DECIMATION)

        # v4.7: 不再使用 opt_params, 直接构建平滑轨迹
        mano_pos = np.array(traj["pos"])
        mano_R = np.array(traj["R"])
        N = len(mano_pos)

        # Stage 2 重建轨迹 (F46~F55)
        stage2_recon = getattr(self, '_v4_stage2_recon', None)

        # 获取物体初始位置 (用于接近期z偏移计算)
        ctrl = self.grasp_controllers.get(side)
        target_obj_name = ctrl.target_obj if ctrl else None
        _obj_init_z_early = 0.0
        if target_obj_name is not None:
            for actor in self.obj_actors:
                if target_obj_name in actor.get_name():
                    _obj_init_z_early = float(np.array(actor.get_pose().p)[2])
                    break

        # ===== 构建 Stage 3 全局轨迹 =====
        # v4.7: 不再使用 opt_params/CEM, 直接构建平滑轨迹
        # 关键: MANO 轨迹需要加 _mano_neutral_offset (让 F50 手指中点对齐物体中心)
        # 不加 offset 时 MANO 轨迹在负坐标, 完全不在物体附近
        _neutral_off = getattr(self, '_mano_neutral_offset', {}).get(side)
        if _neutral_off is None:
            _neutral_off = np.zeros(3)
        mano_pos_offset = mano_pos + _neutral_off  # 带offset的MANO轨迹

        # F1-F45: MANO+offset 轨迹, 但需要从 F1 平滑过渡到 F46 (Stage 2 第1帧)
        # F46-F55: Stage 2 重建 (锁定, 已包含offset)
        # F56-F89: Stage 2 F55 → MANO+offset smoothstep 过渡
        # F90+: MANO+offset 轨迹 (直接跟随)
        opt_pos_full = mano_pos_offset.copy()
        opt_R_full = mano_R.copy()

        _TRANS_END_TO_S2 = 35   # F1-F35: 从 MANO 起点 → F46 的过渡
        _TRANS_START_FROM_S2 = 56  # F56-F89: 从 F55 → MANO 的过渡
        _TRANS_END_FROM_S2 = 89

        if stage2_recon is not None:
            # F46-F55: 使用 Stage 2 重建 (已包含offset)
            _s2_frames = stage2_recon.get('frames', [])
            for f in _s2_frames:
                if f in stage2_recon.get('pos', {}):
                    opt_pos_full[f] = np.asarray(stage2_recon['pos'][f])
                    opt_R_full[f] = np.asarray(stage2_recon['R'][f])

            # F1-F45: 直接使用 MANO+offset 轨迹 (边走边抓)
            # 不再从远处过渡, 因为 MANO+offset 已经在物体附近
            # 接近期只需手指全开避免碰物体

            # F56-F89: smoothstep 过渡到 MANO+offset
            _s2_last_f = max(_s2_frames) if _s2_frames else 55
            _trans_start_pos = opt_pos_full[_s2_last_f].copy()
            _trans_start_R = opt_R_full[_s2_last_f].copy()

            for f in range(_s2_last_f + 1, min(_TRANS_END_FROM_S2 + 1, N)):
                _t = (f - _s2_last_f) / max(_TRANS_END_FROM_S2 - _s2_last_f, 1)
                _s = _t * _t * (3 - 2 * _t)  # smoothstep
                opt_pos_full[f] = _trans_start_pos * (1 - _s) + mano_pos_offset[f] * _s
                try:
                    from scipy.spatial.transform import Rotation, Slerp
                    _r0 = Rotation.from_matrix(_trans_start_R)
                    _r1 = Rotation.from_matrix(mano_R[f])
                    _slerp = Slerp([0, 1], Rotation.concatenate([_r0, _r1]))
                    opt_R_full[f] = _slerp(_s).as_matrix()
                except Exception:
                    opt_R_full[f] = _trans_start_R * (1 - _s) + mano_R[f] * _s

        robot = self.robot_info["robot"]
        robot.set_qpos(self.robot_info["init_qpos"].copy())
        robot.set_qvel(np.zeros_like(robot.get_qvel()))
        if self.robot_info.get("virtual_idx"):
            robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        else:
            robot.set_root_pose(sapien.Pose(self._base_pos.tolist(), self._base_quat.tolist()))
        robot.set_root_linear_velocity([0, 0, 0])
        robot.set_root_angular_velocity([0, 0, 0])

        # 确保物体初始位姿已记录 (test 入口可能跳过完整仿真循环)
        if not hasattr(self, '_obj_initial_poses') or self._obj_initial_poses is None:
            self._obj_initial_poses = {actor.get_name(): actor.get_pose() for actor in self.obj_actors}

        for actor in self.obj_actors:
            name = actor.get_name()
            if name in self._obj_initial_poses:
                actor.set_pose(self._obj_initial_poses[name])
                try:
                    for comp in actor.get_components():
                        if hasattr(comp, 'set_linear_velocity'):
                            comp.set_linear_velocity([0, 0, 0])
                            comp.set_angular_velocity([0, 0, 0])
                            break
                except Exception:
                    pass

        gi1, gi2 = self.robot_info["gripper_idx1"], self.robot_info["gripper_idx2"]
        aji = self.robot_info.get("arm_joint_indices", [])

        # 设置手指 PD 参数: 和 Stage 1 一致
        _s3_force_limit = float(GRASP_STRATEGIES['pd_then_lock']['force_limit'])  # 40N
        active_joints = robot.get_active_joints()
        for gidx in (gi1, gi2):
            if gidx is not None:
                active_joints[gidx].set_drive_property(
                    stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING,
                    force_limit=_s3_force_limit
                )

        for _ in range(5):
            _q = robot.get_qpos()
            physics_step(robot, aji, gi1, gi2, np.array([]),
                         float(_q[gi1]), float(_q[gi2]), self.scene)

        # Stage 3: 禁用手指-地面碰撞 (和 Stage 1 rollout 一致)
        # R1 手指 mesh 过长, 对小物体必然触地, 导致无法夹紧
        _ground_actor_s3 = None
        _orig_groups_s3 = []
        _FINGER_IGNORE_ID_S3 = 1
        _FINGER_IGNORE_BIT_S3 = 1 << (_FINGER_IGNORE_ID_S3 - 1)
        try:
            for _actor in self.scene.get_all_actors():
                if _actor.get_name() == 'ground':
                    _ground_actor_s3 = _actor
                    break
            if _ground_actor_s3 is not None:
                for _comp in _ground_actor_s3.get_components():
                    if isinstance(_comp, sapien.pysapien.physx.PhysxRigidStaticComponent):
                        for _cs in _comp.get_collision_shapes():
                            _orig_groups_s3.append((_cs, list(_cs.get_collision_groups())))
                            _g = list(_cs.get_collision_groups())
                            _g[2] |= _FINGER_IGNORE_BIT_S3
                            _g[3] = _FINGER_IGNORE_ID_S3
                            _cs.set_collision_groups(_g)
                logger.info("[Stage 3 rollout] 已禁用手指-地面碰撞")
        except Exception as _e:
            logger.warning(f"[Stage 3 rollout] 禁用手指-地面碰撞失败: {_e}")

        # v4.11: 禁用邻物碰撞 (和 Stage 1 rollout 一致), 让手指能穿过邻物只夹 target
        _NEIGHBOR_ORIG_GROUPS_S3 = []
        ctrl = self.grasp_controllers.get(side)
        target_obj = ctrl.target_obj if ctrl else None
        try:
            if target_obj is not None:
                for _actor in self.scene.get_all_actors():
                    _name = _actor.get_name()
                    if _name == target_obj or not _name.startswith('glb_'):
                        continue
                    for _comp in _actor.get_components():
                        if hasattr(_comp, 'get_collision_shapes'):
                            for _cs in _comp.get_collision_shapes():
                                _NEIGHBOR_ORIG_GROUPS_S3.append((_cs, list(_cs.get_collision_groups())))
                                _cs.set_collision_groups([0, 0, 0, 0])
                if _NEIGHBOR_ORIG_GROUPS_S3:
                    logger.info(f"[Stage 3 rollout] 已禁用邻物碰撞 "
                                f"({len(_NEIGHBOR_ORIG_GROUPS_S3)} shape), target={target_obj}")
        except Exception as _e:
            logger.warning(f"[Stage 3 rollout] 禁用邻物碰撞失败: {_e}")

        obj_init_z, obj_init_xy = 0.0, np.zeros(2)
        if target_obj is not None:
            for actor in self.obj_actors:
                if target_obj in actor.get_name():
                    pos = np.array(actor.get_pose().p)
                    obj_init_z = float(pos[2])
                    obj_init_xy = pos[:2].copy()
                    break

        contact_frames = 0
        close_offsets = []
        _close_dists = []
        opt_pos_traj = []
        opt_R_traj = []
        obj_pos_traj = []

        # 夹持闭合度 (由 Stage 1 计算)
        best_grasp = getattr(self, '_v4_best_grasp', None)
        g_grasp = best_grasp['gripper_qpos'] if best_grasp else 0.008

        # v4.14 夹爪开合控制 (用户明确: 严格按 plan 4 段时序, 独立模块):
        # 阶段 1 (F0-F44,  45 帧): 跟随 MANO 开合参数 (j1, j2)
        # 阶段 2 (F45-F49,  5 帧): smoothstep 从 MANO 开合 → F50 闭合度 (逐渐闭合调整到 F50)
        # 阶段 3 (F50-F89, 40 帧): close 阶段保持不动 (PD target = _GRIP_PD_HOLD 固定, 持续正压力夹住物体)
        # 阶段 4a (F90-F95,  6 帧): smoothstep 从 F50 闭合度 → MANO 开合 (逐渐释放)
        # 阶段 4b (F96+):           释放后跟随 MANO 开合参数 (j1, j2)
        # _GRIP_PD_HOLD = best_grasp['gripper_qpos'] (理论值, 基于物体尺寸计算, 对称闭合)
        # 阶段 3 close 阶段: 夹爪开合 PD target 固定不变, 不跟随 MANO 开合
        # 物体是否被夹住完全由物理仿真决定, 需调整仿真参数 (摩擦/PD/质量) 使物体在 close 阶段与夹爪保持相对关系
        _GRIP_PD_HOLD = float(best_grasp['gripper_qpos']) if best_grasp else 0.0034
        _fp = getattr(self, '_frame_params', None) or {}
        _F_GRASP = _fp.get('F50_IDX', 50)      # F50: 夹持到位帧 (close 阶段开始)
        _F_HOLD_END = _fp.get('F_HOLD_END', 89)  # close 阶段结束 (F89)
        _F_RELEASE_START = min(_F_HOLD_END + 1, N - 1)  # F90: 释放开始
        _F_RELEASE_END = min(95, N - 1)               # F95: 释放结束
        # v4.14: F45 = F50 - 5, 与 plan "前 45 帧跟随 MANO, 45-50 调整到 F50" 一致
        _F_TRANSITION_START = _fp.get('F_TRANSITION_START', max(0, _F_GRASP - 5))

        # MANO 夹爪开合度
        j1_arr = np.asarray(traj["j1"]) if "j1" in traj else None
        j2_arr = np.asarray(traj["j2"]) if "j2" in traj else None

        for local_idx in range(min(self.num_frames, N)):
            self._current_local_idx = local_idx

            # ===== 1. 位姿: 直接使用预构建的平滑轨迹 =====
            target_pos = opt_pos_full[local_idx]
            target_R = opt_R_full[local_idx]

            # ===== 2. 夹爪开合度: 严格 4 段时序 (用户明确要求 v4.13) =====
            _mano_g = float((j1_arr[local_idx] + j2_arr[local_idx]) / 2.0) if (
                j1_arr is not None and j2_arr is not None and local_idx < len(j1_arr)) else 0.012

            if local_idx < _F_TRANSITION_START:
                # 阶段 1 (F0-F44): 跟随 MANO 开合参数 (plan v4.11 用户明确)
                gripper_pd_target = _mano_g
            elif local_idx < _F_GRASP:
                # 阶段 2 (F45-F49): smoothstep 从 MANO 开合 → F50 闭合度 (5 帧过渡)
                _t = (local_idx - _F_TRANSITION_START) / float(_F_GRASP - _F_TRANSITION_START)
                _s = _t * _t * (3 - 2 * _t)
                gripper_pd_target = _mano_g * (1 - _s) + _GRIP_PD_HOLD * _s
            elif local_idx <= _F_HOLD_END:
                # 阶段 3 (F50-F89): close 阶段保持夹持 (PD target 固定)
                gripper_pd_target = _GRIP_PD_HOLD
                # v4.17: 不再锁定 base_z, 跟随 MANO 轨迹 (纯 PD 驱动 base)
            elif local_idx <= _F_RELEASE_END:
                # 阶段 4a (F90-F95): 逐渐打开释放 (smoothstep)
                _t = (local_idx - _F_RELEASE_START) / max(_F_RELEASE_END - _F_RELEASE_START, 1)
                _s = _t * _t * (3 - 2 * _t)
                gripper_pd_target = _GRIP_PD_HOLD * (1 - _s) + _mano_g * _s
                # v4.14k: 释放阶段 base_z 也从 close_lock_z 平滑过渡到 MANO z (避免突变)
                if hasattr(self, '_close_lock_z') and self._close_lock_z is not None:
                    target_pos = target_pos.copy()
                    target_pos[2] = self._close_lock_z * (1 - _s) + target_pos[2] * _s
            else:
                # 阶段 4b (F96+): 释放后跟随 MANO 开合参数
                gripper_pd_target = _mano_g

            # 设置虚拟关节目标
            _vid = self.robot_info.get("virtual_idx", {})
            if _vid:
                _rz, _ry, _rx = rotmat_to_zyx_euler(target_R)
                for vk, vv in [
                    (_vid['vx'], float(target_pos[0])),
                    (_vid['vy'], float(target_pos[1])),
                    (_vid['vz'], float(target_pos[2])),
                    (_vid['rz'], float(_rz)),
                    (_vid['ry'], float(_ry)),
                    (_vid['rx'], float(_rx)),
                ]:
                    active_joints[vk].set_drive_target(float(vv))

                # v4.17: 去掉 virtual_lock_targets, 纯 PD 驱动 (对齐 test8)
                physics_step(robot, aji, gi1, gi2,
                             np.array([]), float(gripper_pd_target), float(gripper_pd_target),
                             self.scene, lock_root_pose=None, decimation=decimation)
            else:
                # full_robot 模式: 走臂 IK
                arm_target, _ = self._step_full_robot(np.zeros((21, 3)))
                physics_step(robot, aji, gi1, gi2,
                             np.asarray(arm_target) if len(arm_target) else np.array([]),
                             float(gripper_pd_target), float(gripper_pd_target),
                             self.scene, lock_root_pose=None, decimation=decimation)

            opt_pos_traj.append(target_pos.copy())
            opt_R_traj.append(target_R.copy())

            # 关键帧诊断: F44~F95 (close 阶段全覆盖) + 每20帧
            _DIAG_FRAMES = set(range(44, 96)) | set(range(0, N, 20))
            if local_idx in _DIAG_FRAMES and target_obj is not None:
                for actor in self.obj_actors:
                    if target_obj in actor.get_name():
                        _op = np.array(actor.get_pose().p)
                        _qpos = robot.get_qpos()
                        _g1 = float(_qpos[gi1]) if gi1 is not None else -1
                        _g2 = float(_qpos[gi2]) if gi2 is not None else -1
                        _dist = np.linalg.norm(target_pos - _op)
                        # 手指 link 位置: 通过名称查找
                        _f1_pos, _f2_pos = np.zeros(3), np.zeros(3)
                        for _lk in robot.get_links():
                            _ln = _lk.get_name()
                            if 'finger_link1' in _ln:
                                _f1_pos = np.array(_lk.get_entity_pose().p)
                            elif 'finger_link2' in _ln:
                                _f2_pos = np.array(_lk.get_entity_pose().p)
                        _finger_mid = (_f1_pos + _f2_pos) / 2.0
                        _obj_to_finger_mid = np.linalg.norm(_op - _finger_mid)
                        _finger_gap = np.linalg.norm(_f1_pos - _f2_pos)
                        # 接触状态
                        f1_c, f2_c, cobjs = get_finger_contacts(robot, side, self.scene, self.obj_actors)
                        _contact_str = "both" if (f1_c and f2_c and target_obj in cobjs) else \
                                       "f1" if (f1_c and target_obj in cobjs) else \
                                       "f2" if (f2_c and target_obj in cobjs) else "none"
                        logger.info(f"  [DIAG F{local_idx}] base=({target_pos[0]:.4f},{target_pos[1]:.4f},{target_pos[2]:.4f}) "
                                    f"obj=({_op[0]:.4f},{_op[1]:.4f},{_op[2]:.4f}) "
                                    f"fmid=({_finger_mid[0]:.4f},{_finger_mid[1]:.4f},{_finger_mid[2]:.4f}) "
                                    f"obj2fmid={_obj_to_finger_mid*1000:.1f}mm fgap={_finger_gap*1000:.1f}mm "
                                    f"q=({_g1:.3f},{_g2:.3f}) contact={_contact_str}")
                        break

            if target_obj is not None:
                for actor in self.obj_actors:
                    if target_obj in actor.get_name():
                        obj_pos = np.array(actor.get_pose().p)
                        obj_pos_traj.append(obj_pos.copy())
                        dist = float(np.linalg.norm(target_pos - obj_pos))
                        _close_dists.append(dist)
                        close_offsets.append(obj_pos - target_pos)
                        f1_c, f2_c, cobjs = get_finger_contacts(robot, side, self.scene, self.obj_actors)
                        if (f1_c or f2_c) and target_obj in cobjs:
                            contact_frames += 1
                        break

            # v4.15: 录制视频 (如果相机已由 run() 初始化)
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

        obj_final_z, obj_final_xy = 0.0, np.zeros(2)
        if target_obj is not None:
            for actor in self.obj_actors:
                if target_obj in actor.get_name():
                    pos = np.array(actor.get_pose().p)
                    obj_final_z = float(pos[2])
                    obj_final_xy = pos[:2].copy()
                    break

        follow_score = 0.0
        follow_score_last5 = 0.0
        if len(close_offsets) >= 3:
            offs = np.array(close_offsets)
            var = np.var(offs, axis=0).mean()
            consistency = 1.0 / (1.0 + var * 100.0)
            lift = max(0.0, obj_final_z - obj_init_z)
            lift_score = 1.0 - 1.0 / (1.0 + max(lift - 0.01, 0.0) * 100.0)
            displacement = float(np.linalg.norm(obj_final_xy - obj_init_xy))
            moved = 1.0 - 1.0 / (1.0 + np.sqrt(displacement**2 + lift**2) * 50.0)
            proximity = 1.0 / (1.0 + (float(np.mean(_close_dists[-5:])) if len(_close_dists) >= 5 else 1.0) * 20.0)
            follow_score = consistency * max(moved, 0.1) * max(proximity, 0.1) * max(lift_score, 0.1)
            if len(offs) >= 5:
                var5 = np.var(offs[-5:], axis=0).mean()
                consistency5 = 1.0 / (1.0 + var5 * 100.0)
                follow_score_last5 = consistency5 * max(moved, 0.1) * max(proximity, 0.1) * max(lift_score, 0.1)

        xy_drift = float(np.linalg.norm(obj_final_xy - obj_init_xy))

        obj_follow_errors = []
        for f in range(42, 58):
            if f < len(close_offsets):
                obj_follow_errors.append(float(np.linalg.norm(close_offsets[f - 42])))

        max_launch = 0.0
        if len(obj_pos_traj) >= 2:
            max_launch = float(np.max(np.abs(np.diff(np.array(obj_pos_traj)[:, 2]))))

        # 计算抓取期间 (F50-F95) 的峰值抬升, 而非最终位置 (释放后物体会掉落)
        _obj_peak_z = obj_init_z
        if len(obj_pos_traj) > 0:
            _obj_z_arr = np.array(obj_pos_traj)[:, 2]
            _grasp_start_idx = min(_F_GRASP, len(_obj_z_arr))
            _grasp_end_idx = min(_F_RELEASE_START + 1, len(_obj_z_arr))
            if _grasp_start_idx < _grasp_end_idx:
                _obj_peak_z = float(np.max(_obj_z_arr[_grasp_start_idx:_grasp_end_idx]))

        result = dict(
            contact_frames_in_close=contact_frames,
            obj_init_z=obj_init_z, obj_final_z=obj_final_z,
            obj_lift=max(0.0, _obj_peak_z - obj_init_z),  # 抓取期间峰值抬升
            obj_init_xy=obj_init_xy, obj_final_xy=obj_final_xy,
            xy_drift=xy_drift,
            follow_score=follow_score, follow_score_last5=follow_score_last5,
            max_penetration=0.0,
            max_launch=max_launch,
            opt_pos=np.array(opt_pos_traj, dtype=np.float64),
            opt_R=np.array(opt_R_traj, dtype=np.float64),
            mano_pos=np.array(traj["pos"], dtype=np.float64),
            mano_R=np.array(traj["R"], dtype=np.float64),
            obj_follow_errors=obj_follow_errors,
            obj_pos_traj=np.array(obj_pos_traj, dtype=np.float64) if len(obj_pos_traj) > 0 else np.zeros((0, 3)),
        )

        # v4.7: 核心指标 — 物体-MANO轨迹偏差 (close阶段应逐渐减小)
        _obj_traj = result['obj_pos_traj']
        _mano_pos = result['mano_pos']
        _close_start = _F_GRASP   # F50: close 阶段开始
        _close_end = _F_HOLD_END   # F89: close 阶段结束 (plan v4.11)
        if len(_obj_traj) >= _close_end:
            _obj_mano_dists = []
            for f in range(_close_start, min(_close_end, len(_obj_traj))):
                # 用带offset的MANO轨迹计算差值 (物体应该跟随MANO+offset)
                _mano_p = mano_pos[f] + _neutral_off if f < len(mano_pos) else _obj_traj[f]
                _d = float(np.linalg.norm(_obj_traj[f] - _mano_p))
                _obj_mano_dists.append(_d)
            result['obj_mano_dists'] = _obj_mano_dists
            result['obj_mano_dist_mean'] = float(np.mean(_obj_mano_dists))
            result['obj_mano_dist_f50'] = _obj_mano_dists[min(6, len(_obj_mano_dists)-1)] if _obj_mano_dists else float('inf')
            result['obj_mano_dist_f80'] = _obj_mano_dists[min(36, len(_obj_mano_dists)-1)] if _obj_mano_dists else float('inf')
        else:
            result['obj_mano_dists'] = []
            result['obj_mano_dist_mean'] = float('inf')
            result['obj_mano_dist_f50'] = float('inf')
            result['obj_mano_dist_f80'] = float('inf')

        logger.info(f"[Stage 3] obj-mano距离: mean={result['obj_mano_dist_mean']*1000:.1f}mm, "
                    f"F50={result['obj_mano_dist_f50']*1000:.1f}mm, F80={result['obj_mano_dist_f80']*1000:.1f}mm, "
                    f"lift={result['obj_lift']*100:.1f}cm, drift={xy_drift*100:.1f}cm")

        # v4.11: 恢复邻物碰撞设置 (Stage 3 rollout 结束)
        try:
            for _cs, _g in _NEIGHBOR_ORIG_GROUPS_S3:
                _cs.set_collision_groups(_g)
        except Exception as _e:
            logger.warning(f"[Stage 3 rollout] 恢复邻物碰撞组失败: {_e}")

        result['reward'] = self.stage3_reward(result)
        return result

    # === cem_stage3_optimize (L6578) ===
    def cem_stage3_optimize(self, side, n_iterations=10, population_size=12, init_params=None):
        """Stage 3 CEM: 优化 F1~F45 和 F55~F90 的偏移

        v4.4: F46~F54 锁定 (Stage 2 输出), F91~F112 锁定跟随 MANO.
        优化变量: 非锁定帧的 6DOF 偏移.

        Args:
            side: 'right' or 'left'
            n_iterations: CEM 迭代轮数
            population_size: 每轮采样数
            init_params: 初始参数 (可选, 非零时从该点开始搜索)
        """
        from traj_optimize import compute_frame_params, POS_RANGE, ROT_RANGE

        traj = self._mano_gripper_traj.get(side)
        if traj is None or len(traj["pos"]) == 0:
            logger.error("[Stage 3] 无 MANO 轨迹")
            return None
        N = len(traj["pos"])

        _fp = getattr(self, '_frame_params', None)
        _fo = getattr(self, '_fixed_offsets_654', None)
        if _fp is None or _fo is None:
            logger.error("[Stage 3] 缺少 _frame_params 或 _fixed_offsets_654")
            return None

        fixed_set = set(_fp['fixed_frames'])  # 只用 _frame_params 定义固定帧, 不用 _fo.keys()
        opt_frames = [f for f in range(N) if f not in fixed_set]
        dim = len(opt_frames) * 6
        if dim <= 0:
            logger.warning("[Stage 3] 无优化帧")
            return np.zeros(0), -float('inf')

        logger.info(f"[Stage 3] CEM: {len(opt_frames)} 帧 × 6DOF = {dim}D, "
                    f"iter={n_iterations}, pop={population_size}")

        # 初始均值: 若提供 init_params 则使用, 否则从零开始
        mu = np.zeros(dim, dtype=np.float64)
        if init_params is not None and len(init_params) == dim:
            mu = np.asarray(init_params, dtype=np.float64).copy()
            logger.info(f"[Stage 3 CEM] 使用 init_params, ||mu||={np.linalg.norm(mu):.4f}")
        elif init_params is not None and len(init_params) > 0:
            # 截断或零填充到匹配维度
            _mu = np.zeros(dim, dtype=np.float64)
            _copy_len = min(len(init_params), dim)
            _mu[:_copy_len] = init_params[:_copy_len]
            mu = _mu
            logger.info(f"[Stage 3 CEM] init_params 维度不匹配 ({len(init_params)} vs {dim}), 截断填充")
        std = np.tile(np.array([0.005, 0.005, 0.005, 0.02, 0.02, 0.02], dtype=np.float64), len(opt_frames))
        # 如果 init_params 非零 (已有好解), 缩小搜索范围
        if np.linalg.norm(mu) > 0.01:
            std *= 0.5  # 缩小扰动: 位置 2.5mm, 姿态 ~0.6°
            logger.info(f"[Stage 3 CEM] init_params 非零, 缩小 std (微调模式)")
        bounds_low = np.tile(np.array([-POS_RANGE, -POS_RANGE, -POS_RANGE,
                                       -ROT_RANGE, -ROT_RANGE, -ROT_RANGE], dtype=np.float64), len(opt_frames))
        bounds_high = np.tile(np.array([POS_RANGE, POS_RANGE, POS_RANGE,
                                        ROT_RANGE, ROT_RANGE, ROT_RANGE], dtype=np.float64), len(opt_frames))

        best_params = mu.copy()
        best_reward = -float('inf')
        n_elite = max(2, int(population_size * 0.25))

        for it in range(n_iterations):
            samples = np.random.randn(population_size, dim) * std + mu
            samples = np.clip(samples, bounds_low, bounds_high)
            rewards = []
            for s in samples:
                r = self.rollout_v4_stage3(s, side)['reward']
                rewards.append(r)
            rewards = np.array(rewards)
            elite_idx = np.argsort(rewards)[-n_elite:]
            elite = samples[elite_idx]
            mu = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-4

            if rewards.max() > best_reward:
                best_reward = float(rewards.max())
                best_params = samples[np.argmax(rewards)].copy()

            logger.info(f"[Stage 3 CEM] iter {it}/{n_iterations}: best={best_reward:.3f}, "
                        f"mean={rewards.mean():.3f}, dim={dim}")

        logger.info(f"[Stage 3] 最优 reward={best_reward:.3f}, params norm={np.linalg.norm(best_params):.4f}")
        return best_params, best_reward
