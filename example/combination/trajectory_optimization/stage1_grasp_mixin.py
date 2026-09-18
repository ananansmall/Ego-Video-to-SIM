"""stage1_grasp_mixin.py — Stage 1: 抓取姿态优化 (6DOF CMA-ES)

从 grasp_hawor.py 抽出, 提供 Stage1Mixin 类。
Stage 1: 生成候选抓取姿态 → 物理仿真验证 → CMA-ES 优化 → 选出最优 grasp_pose
依赖主类 GraspSimulator 的以下属性/方法 (由主类初始化):
  - self._mano_gripper_traj, self._mano_neutral_offset, self._mano_grasp_frame
  - self.grasp_controllers, self.obj_bbox_centers, self.obj_info
  - self.scene, self.robot_info, self.obj_actors, self._ground_z
"""
import logging
import os
import numpy as np
import sapien

from physics_env import (
    FINGER_EFFECTIVE_HALF_SPACING, FINGER_GAP0, _GRIP_CLOSE, _GRIP_OVERCLOSURE,
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

        # ★★★★ 2026-09-18 Step 1: CoACD 版位姿代价 (用户: "可以的" / 只量测, **不改任何决策** ) ★★★★
        #   用户判断 (本轮已验证 ✓): "主要还是抓取吧 ... reward 需要设置的更精细一点, 主要从这个
        #     coacd, 比如我们判断哪个是最优位姿, 然后给 stage1 的位姿有个代价"
        #   依据: 本函数上面的 antipodal / force-closure 两项**只用 bbox 近似** ✗
        #     (obj_radius = max(bbox)/2 ✗) ⇒ 看不到 CoACD 凸包的凹凸/棱/朝向 ✗
        #     ⇒ 选位姿对**真实形状是盲选** ✗ ⇒ 选出"边缘位姿" ✗
        #   实测症状 (CoACD 64 凸包, output/fine64 等):
        #     · 接近段擦碰 APPROACH_TARGET_CONTACT 0/1 ~ 0/24 ✗
        #     · 两侧接触时间差 f1 比 f2 晚 3~5 个控制点 ✗
        #     · 握持余量只有 0.3mm (指缝 56.2mm vs 物体宽 55.9mm) ✗
        #   本块只算**三项几何代价**并打印 ✓ (全部只用"凸包顶点在闭合轴上的投影" ✓ ⇒ 无需法向 ✓):
        #     c_center : 物心在闭合轴上的投影偏移 ✓           (对中 ✗)
        #     c_clear  : max_i(凸包在该轴的横向最大值) − 指缝/2 ✓ (擦碰余量 ✗)
        #     c_spread : 两侧支撑点距物心的距离之差 ✓          (两侧对称性 ✗ ⇒ 对应"一侧先碰上" ✗)
        #   凸包来源: 直接读**引擎里该 actor 的 convex shapes** ✓ ⇒ 与物理完全一致 ✓
        #            (避免再走一遍网格/缓存管线 ✓)
        try:
            import sapien as _sp

            _actor = None
            for _a in getattr(self, "obj_actors", []) or []:
                _n = _a.get_name()
                if _n == target_name or target_name in _n:
                    _actor = _a
                    break
            _hulls_w = []
            if _actor is not None:
                _H = np.asarray(
                    _actor.get_pose().to_transformation_matrix(), dtype=np.float64)
                for _comp in _actor.components:
                    if not isinstance(
                            _comp, _sp.pysapien.physx.PhysxRigidDynamicComponent):
                        continue
                    for _sh in _comp.get_collision_shapes():
                        try:
                            _v = np.asarray(_sh.get_vertices(), dtype=np.float64)
                        except Exception:  # noqa: BLE001
                            continue
                        if _v.ndim == 2 and _v.shape[1] == 3 and len(_v):
                            _hulls_w.append(_v @ _H[:3, :3].T + _H[:3, 3])
            if _hulls_w:
                _rel = np.concatenate(
                    [h - np.asarray(obj_center_3d, dtype=np.float64) for h in _hulls_w],
                    axis=0)
                _s = _rel @ close_dir            # 沿闭合轴的有符号坐标
                _f = _rel @ forward_dir          # 沿接近方向 (仅记录)
                # c_center: 物心已为原点 ⇒ 由两侧支撑点不对称度体现; 另给"轴向重心偏移"
                _lo, _hi = float(_s.min()), float(_s.max())
                c_center = abs(0.5 * (_lo + _hi)) * 1000.0     # mm
                c_clear = (max(abs(_lo), abs(_hi)) - half_sep) * 1000.0   # mm, 负=安全
                c_spread = abs(abs(_hi) - abs(_lo)) * 1000.0   # mm
                _n_vert = int(len(_rel))
                logger.info(
                    "[CoACD质量] target=%s 凸包=%d 顶点=%d | 闭合轴范围=[%+.1f, %+.1f]mm "
                    "(含-物心) 指缝/2=%.1fmm | c_center=%.2fmm c_clear=%+.2fmm c_spread=%.2fmm "
                    "| 轴向范围=[%+.1f, %+.1f]mm | 总评分=%.3f",
                    target_name, len(_hulls_w), _n_vert,
                    _lo * 1000.0, _hi * 1000.0, half_sep * 1000.0,
                    c_center, c_clear, c_spread,
                    float(_f.min()) * 1000.0, float(_f.max()) * 1000.0,
                    float(score),
                )
            else:
                logger.warning(
                    "[CoACD质量] target=%s 未取到凸包 (可能未启用 CoACD 或 shapes 无顶点接口)",
                    target_name)
        except Exception as _eq:  # noqa: BLE001
            logger.warning("[CoACD质量] 采集失败: %s", _eq)

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

    # === generate_grasp_candidates_multiframe (2026-09-12, 用户定策 C) ===
    def generate_grasp_candidates_multiframe(
        self, side, half_span=15, n_close=3, n_fwd=3, n_z=3
    ):
        """扫描 grasp_frame ± half_span 的**每一帧** MANO 夹爪位姿, 每帧配一个小平移网格.

        用户定策 (2026-09-12): "stage1 目前不一定是选取 F50 旁边的, F25 到 F60 都可以去优化";
        "现在有很多的位姿可以筛选, 可以根据 reward 选择一个最大的位姿".

        与旧 `generate_grasp_candidates` 的区别:
          - 旧: 姿态**锁死在 F50** (base_R=mano_f50_R), 只在位置网格上搜 —— 位姿来源单一.
          - 新: 逐帧取该帧 MANO 姿态 (F35–F65 共 31 帧), 每帧再配 ±10/±8/±5mm 的位置微调,
                候选总数 31×(3×3×3)=837, 交给既有 Stage1a 解析预筛 + CEM 物理打分,
                最终由 reward 选出最优位姿 —— 位姿选择有物理依据.

        下游完全复用 (返回 dict 结构与旧函数一致, 额外带 'source_frame' 便于追溯).
        """
        from data_loader import rotmat_to_zyx_euler

        traj = self._mano_gripper_traj.get(side, {})
        pos_traj = np.asarray(traj.get("pos", []))
        R_traj = np.asarray(traj.get("R", []))
        n = len(pos_traj)
        if n == 0:
            logger.warning("[Stage1][多帧扫描] MANO 夹爪轨迹为空, 回退单帧候选")
            return []

        # 扫描中心: grasp_frame (默认沿用历史 F50)
        g = int(getattr(self, "grasp_frame_idx", 0) or 0)
        if g <= 0:
            g = min(50, n - 1)
        lo = max(0, g - int(half_span))
        hi = min(n - 1, g + int(half_span))
        _tilt_deg = float(getattr(self, "f50_tilt_deg", 0.0) or 0.0)
        if abs(_tilt_deg) > 1e-6:
            from scipy.spatial.transform import Rotation as _Rsci
            _tilt_R = _Rsci.from_euler('y', _tilt_deg, degrees=True).as_matrix()
        else:
            _tilt_R = None

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        if target_name is None or target_name not in self.obj_bbox_centers:
            if not self.obj_bbox_centers:
                return []
            target_name = list(self.obj_bbox_centers.keys())[0]
        obj_pos = np.array(self.obj_bbox_centers[target_name], dtype=np.float64)

        _FINGER_OFFSET_GRIPPER = np.array([0.03689, 0.0, 0.0])
        _fwd_l = np.array([1.0, 0.0, 0.0])     # gripper X: base -> 手指原点
        _close_l = np.array([0.0, 1.0, 0.0])   # gripper Y: 开合方向
        _g0 = float(getattr(self, '_ground_z', 0.0))
        d_close_list = [(i - (n_close - 1) / 2.0) * 0.010 for i in range(n_close)]
        d_fwd_list = [(i - (n_fwd - 1) / 2.0) * 0.008 for i in range(n_fwd)]
        d_z_list = [(i - (n_z - 1) / 2.0) * 0.005 for i in range(n_z)]

        # ★★★★ 2026-09-14 用户纠正: "F25→F50 ±5° 是**每个候选都能调整 ±5 度**,
        #      不是只有 F50 是 ±5 度" ★★★★
        #   查证: 旧实现对每个源帧 f 只做**位置**微调 (d_close/d_fwd/dz) ✗,
        #         姿态 R 完全等于该帧的 MANO 原样 ✗ ⇒ "每个候选都能调 ±5°"没有实现 ✗
        #   修复: 给每个源帧再加一组**姿态扰动** (绕夹爪自身的 Y 与 Z 各 ±5° ✓),
        #         绕夹爪 X (接近轴) **不扰** —— 绕它转只改开合平面朝向, 交给
        #         Stage1 的 roll 约束去定 ✓ (否则候选爆炸且物理含义重复 ✓)
        #   候选数: 31 帧 × 27 位置 × (3×3) 姿态 = 7533 —— 由 Stage1a 解析预筛先砍一刀 ✓
        #   可用 _STAGE1_ROT_SPAN_DEG 调整 (0 = 关闭, 回到旧行为 ✓)
        _rot_span = float(os.environ.get("_STAGE1_ROT_SPAN_DEG", "5.0"))
        if _rot_span > 1e-6:
            from scipy.spatial.transform import Rotation as _Rrot
            _d_rot = [-_rot_span, 0.0, _rot_span]
            _rot_list = [
                (_Rrot.from_euler('y', _ry, degrees=True).as_matrix()
                 @ _Rrot.from_euler('z', _rz, degrees=True).as_matrix())
                for _ry in _d_rot for _rz in _d_rot
            ]
        else:
            _rot_list = [np.eye(3)]

        candidates = []
        for f in range(lo, hi + 1):
            _R0 = np.asarray(R_traj[f], dtype=np.float64)
            if _tilt_R is not None:
                _R0 = _tilt_R @ _R0
            for _Rd in _rot_list:
                R = _R0 @ _Rd                      # ★ 扰动施加在**夹爪自身坐标系** ✓
                base_pos = obj_pos - R @ _FINGER_OFFSET_GRIPPER
                for d_close in d_close_list:
                    for d_fwd in d_fwd_list:
                        for dz in d_z_list:
                            pos = base_pos + R @ (_fwd_l * d_fwd + _close_l * d_close)
                            pos[2] += dz
                            if pos[2] < _g0 + 0.01:
                                continue
                            candidates.append({
                                'pos': pos,
                                'R': R.copy(),
                                'euler': np.array(rotmat_to_zyx_euler(R)),
                                'source_frame': float(f),
                            })
        logger.info(
            f"[Stage1][多帧扫描] grasp_frame=F{g} 扫描 F{lo}-F{hi} ({hi - lo + 1} 帧) "
            f"× {len(d_close_list) * len(d_fwd_list) * len(d_z_list)} 位置微调 "
            f"→ {len(candidates)} 个候选 (含 source_frame 便于追溯)"
        )
        return candidates

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
        # ★★★ 2026-09-14 用户反馈"开合为什么还那么大, 0.05 即可" ★★★
        #   查证: 全仓 5 个调用点都写 `stage_q_open()` **不带参数** ✗ →
        #        stage1_recipe 里的"按物体尺寸自动开度"规则从未生效,
        #        永远回落到 GRIPPER_MAX_OPEN=0.05.
        #   而 0.05 的**物理含义**是: 指缝 = FINGER_GAP0 + 2q = 30.4 + 100 = **130.4mm** ✗
        #        (关节值 0.05 ≠ 开度 5cm; 因为真实指形的指缝基准就有 30.4mm)
        #        ⇒ 对 55.9mm 的物体张到 13cm, 就是"看起来那么大"的原因.
        #   修复: 传入目标物体的**最大水平边** → 自动 q_open = (1.20·W − GAP0)/2.
        #   ⚠ 必须用 max(bbox[:2]) 而不是 min ✗ —— 实测教训 (2026-09-14):
        #     用 min(44.9mm) 算出 q_open=0.0117 → 指缝 **53.9mm** < 物体最宽 55.9mm ✗
        #     ⇒ Stage1 入口自检立即报 "anchor 在 q_open 入口已与 glb_4 接触
        #       (f1=True f2=True) → 该候选不是可迁移 anchor" ✗
        #     因为夹爪进入时不知道物体的水平朝向, 必须包住**最坏情况** (最大水平边),
        #     这与 compute_gripper_qpos 里"q_open 用 max 边"的理由完全一致.
        #     用 max: q_open=0.0183 → 指缝 67.1mm > 55.9mm ⇒ 安全进入 ✓ (余量 11.2mm)
        _ow = None
        try:
            _ctrls = getattr(self, "grasp_controllers", None) or {}
            _ctrl = _ctrls.get(getattr(self, "side", "left")) if hasattr(_ctrls, "get") else None
            _tn = getattr(_ctrl, "target_obj", None) if _ctrl is not None else None
            if _tn:
                _bb = (getattr(self, "obj_info", {}) or {}).get(_tn, {}).get("bbox_size")
                if _bb is not None:
                    _ow = float(max(_bb[0], _bb[1]))
        except Exception:  # noqa: BLE001
            _ow = None
        _q_open = stage_q_open(_ow)
        logger.info("[Stage1 开度] 物体最短水平边=%s → q_open=%.4f → 指缝=%.1fmm "
                    "(旧: 无参数调用 → 0.05 → 指缝 130.4mm)",
                    f"{_ow*1000:.1f}mm" if _ow else "n/a", _q_open,
                    (FINGER_GAP0 + 2.0 * _q_open) * 1000.0)
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
                _q_contact_old = max(0.0, _bbox_max / 2.0 - FINGER_EFFECTIVE_HALF_SPACING)
                # ★★★ 任务4 (2026-09-13) 定量诊断: 物体**在夹爪闭合轴上的实际投影跨度**.
                #   旧代码用 max(bbox[:2]) —— 那是"最坏水平朝向"的保守值, 对 **q_open(张开)**
                #   是对的 (保证任意朝向都能包住), 但对 **q_grasp(闭合目标)** 是错的 ✗:
                #   闭合必须对准物体在**闭合轴**(夹爪局部 Y)上的真实尺寸。
                #   实测 (scale0.5): bbox max=55.9mm 而闭合轴投影仅 ~41.9mm → 差 14mm。
                #   薄板时代指缝能到 10mm, 大行程把这个错误"吸收"了; 真实指形(指缝 30.4mm)
                #   下闭合到 54.9mm 时**手指根本没碰到物体** → Stage1 抬不起来 (0/3)。
                _proj_span = None
                try:
                    _R_g = np.asarray(target_R, dtype=np.float64)
                    _y_ax = _R_g[:, 1]                       # 夹爪闭合轴(局部Y)在世界系
                    _bb = np.asarray(_bbox_size, dtype=np.float64)
                    # 半尺寸盒在轴上的投影跨度 = 2*Σ|axis_i|*half_i
                    _proj_span = float(2.0 * np.sum(np.abs(_y_ax) * (_bb / 2.0)))
                except Exception:  # noqa: BLE001
                    _proj_span = None
                # ★★★ v52 (2026-09-14) 关键修正 —— 闭合目标必须**保守取小** ★★★
                #   力学事实: 手指从 q_open 闭合, 要么停在 PD 目标, 要么被物体挡住.
                #     · 目标 ≤ 真实接触点 → 物体挡停 → 产生正压力 → 夹紧 ✓
                #     · 目标 >  真实接触点 → 手指停在半空 → **没夹住** → 物体掉落 ✗
                #   实测教训 (task4, 2026-09-13):
                #     bbox_max 估计 = 55.9mm → 目标 0.0127
                #     闭合轴投影   = 67.0mm → 目标 0.0183  ← 我改成这个 (太大 ✗)
                #     物体**真实**跨度 ≈ 53.2mm (由 n53 成功 run 的 locked 值 0.0114 反推)
                #     ⇒ 目标 0.0183 > 真实接触 0.0103 → 手指停在 66.0mm 缝 → 没碰到物体
                #     ⇒ 物体掉落 → 手指空夹到底 → locked=0.0047 (铁证: 完全闭合)
                #   因此: 两个估计**取最小** (保守), 保证目标一定 ≤ 真实接触点.
                _q_contact_proj = None
                if _proj_span is not None and _proj_span > 1e-6:
                    _q_contact_proj = max(0.0, _proj_span / 2.0 - FINGER_EFFECTIVE_HALF_SPACING)
                # 保守估计: 用物体**最小水平边** (物体倾向转成最短轴朝夹爪 → 这是最安全的估计)
                _bbox_min_dim = float(min(_bbox_size[0], _bbox_size[1]))
                _q_contact_min = max(0.0, _bbox_min_dim / 2.0 - FINGER_EFFECTIVE_HALF_SPACING)
                _cands = [c for c in (_q_contact_min, _q_contact_old, _q_contact_proj) if c is not None]
                _q_contact = float(min(_cands)) if _cands else float(_computed_target)
                _GRIP_TARGET = max(0.0, _q_contact - _GRIP_OVERCLOSURE)
                logger.info(
                    "[Stage1 闭合目标] 估计三者取最小: min边%.1fmm→%.4f | bbox_max%.1fmm→%.4f | "
                    "投影%.1fmm→%s ⇒ 采用 q_contact=%.4f → PD_target=%.4f (FINGER_GAP0=%.1fmm)",
                    _bbox_min_dim * 1000.0, _q_contact_min, _bbox_max * 1000.0, _q_contact_old,
                    (_proj_span * 1000.0) if _proj_span else -1.0,
                    f"{_q_contact_proj:.4f}" if _q_contact_proj is not None else "n/a",
                    _q_contact, _GRIP_TARGET, FINGER_GAP0 * 1000.0)
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
        # ★★★★ 2026-09-17 接近段接触帧数 (用户: "接近段是不允许接触的, 这个代价还设置少了") ★★★★
        #   为什么必须**分阶段**统计: 抬升段本来**必须**接触 ✓ ⇒ 用全程 contact_count 惩罚
        #   会误伤正常抓持 (曾误写 -3.0×count ⇒ -300 ✗ 毁掉抬升奖励 ✗)
        #   只有"抬升前 (frame < _LIFT_START_FRAME)"的接触才是**不该有**的 ✓
        approach_contact_count = 0
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

        # ★ 2026-09-14 P0: 录像写帧计数器 (用 list 便于闭包内自增; 每帧循环前重置)
        _s1_video_frames = [0]
        # ★★★ 2026-09-14 用户: "每次测试完夹爪的位姿和开合都得保留一个 log 或 csv 文件"
        #   目的: 用户看视频时发现"开始前几帧和最后几帧夹爪又向上了" ✗ —— 需要逐帧数值佐证 ✓
        #   记录内容: 帧号 / 夹爪目标位置 / 夹爪 ZYX 欧拉(含"朝下"判据) / 两指命令与实际 q /
        #             物体 z / 双侧接触 / 夹持力 / 夹爪-物距离
        _s1_trace_rows = []

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

            # ★ 2026-09-14 逐帧 CSV 记录 (用户要求: 位姿 + 开合留档)
            #   ★★ 用户纠正: "如果和刚开始同向的那不都成向上了" ★★
            #     视频里看到的是 **实际姿态** ✓, 而我第一版只记了 **命令姿态** ✗
            #     (命令全程朝下 ✓ 但 PD 跟不住 → 实际翻过来 ✗ → 观感"朝上" ✓)
            #   ⇒ 现在同时记两路: cmd_* (命令) 与 act_* (实际), 并各给一个 down 判据 ✓
            try:
                _Rz_ = np.asarray(target_R, dtype=np.float64)
                _ez_, _ey_, _ex_ = rotmat_to_zyx_euler(_Rz_)
                _Xw_ = _Rz_[:, 0]                       # 命令的夹爪 X
                _down_ = float(np.dot(_Xw_, np.array([0.0, 0.0, -1.0])))   # >0 = 朝下 ✓
                # ---- 实际夹爪位姿 (物理里真正发生的那一个 ✓) ----
                _Ta_ = self._gripper_pose_matrix(robot, side)
                if _Ta_ is not None and np.isfinite(_Ta_).all():
                    _Ra_ = np.asarray(_Ta_, dtype=np.float64)[:3, :3]
                    _az_, _ay_, _ax_ = rotmat_to_zyx_euler(_Ra_)
                    _Xa_ = _Ra_[:, 0]
                    _adown_ = float(np.dot(_Xa_, np.array([0.0, 0.0, -1.0])))
                    _apos_ = np.asarray(_Ta_, dtype=np.float64)[:3, 3]
                else:
                    _ax_ = _ay_ = _az_ = float("nan")
                    _adown_ = float("nan")
                    _apos_ = np.full(3, np.nan)
                _fq1_ = float(qpos[gripper_idx1])
                _fq2_ = float(qpos[gripper_idx2])
                _s1_trace_rows.append((
                    int(frame),
                    float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                    float(np.rad2deg(_ex_)), float(np.rad2deg(_ey_)), float(np.rad2deg(_ez_)),
                    float(_down_),
                    float(np.rad2deg(_ax_)), float(np.rad2deg(_ay_)), float(np.rad2deg(_az_)),
                    float(_adown_),
                    float(_fq1_), float(_fq2_),
                    float(FINGER_GAP0 + 2.0 * _fq1_),      # 指缝 (物理口径 ✓)
                    float(qpos[gripper_idx1]), float(qpos[gripper_idx2]),
                    float(current_obj_z) if 'current_obj_z' in dir() else float("nan"),
                ))
            except Exception:  # noqa: BLE001
                pass

            # v4.16: 录制视频 — vk DeviceLost 保护 + 帧标注 (2026-09-06)
            # ★★ 2026-09-14 P0 修复: 静默失败 → 响亮报错 ★★
            #   实测 --test-stage1 产出 **44 字节空 MP4** (内容仅 "ftypisomisomiso2mp41 freemdat",
            #   零帧) ✗, 但调用方只判断 `_writer_* is not None` 就打印"视频已保存" → 假成功 ✗.
            #   真因: record_video=True 但 scene._render_available=False → 整个录像分支被跳过,
            #   静默 continue 了一整个 rollout. 现在遇到立即 logger.error, 且累计写帧数.
            try:
                _render_ok = bool(getattr(self.scene, "_render_available", False))
                if record_video and not _render_ok:
                    logger.error(
                        "[Stage1 录像] record_video=True 但 scene._render_available=False "
                        "→ **不会写入任何帧** (产出 mp4 只有 44 字节空壳) ✗ | "
                        "检查: views/camera 初始化是否成功 (CPU 模式无 RenderSystem?)"
                    )
                if record_video and _render_ok:
                    _s1_video_frames[0] += 1
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
                        if frame < _LIFT_START_FRAME:
                            approach_contact_count += 1   # 抬升前就接触 ⇒ 接近段撞到了 ✗
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

        # ★★★ 2026-09-14 用户要求: 每次测试输出逐帧 CSV (位姿 + 开合留档) ★★★
        #   用途: 视频里"前几帧/最后几帧夹爪朝上"这类观感 → 直接查 down 列 (ZYX 的 X·ĝ)
        #   文件: <output_dir>/stage1/gripper_trace.csv  (有 output_dir 时)
        try:
            if _s1_trace_rows:
                _od = getattr(self, "output_dir", None) or os.environ.get("_OUTPUT_DIR")
                if _od:
                    _csv_dir = os.path.join(str(_od), "stage1")
                else:
                    _csv_dir = os.path.join(str(getattr(self, "hawor_dir", ".")), "..", "stage1_trace")
                os.makedirs(_csv_dir, exist_ok=True)
                _csv_path = os.path.join(_csv_dir, "gripper_trace.csv")
                with open(_csv_path, "w", encoding="utf-8") as _fh:
                    _fh.write(
                        "frame,pos_x,pos_y,pos_z,euler_rx_deg,euler_ry_deg,euler_rz_deg,"
                        "cmd_down_dot,act_euler_rx_deg,act_euler_ry_deg,act_euler_rz_deg,"
                        "act_down_dot,finger1_q,finger2_q,finger_gap_m,cmd_q1,cmd_q2,obj_z\n")
                    for _r in _s1_trace_rows:
                        _fh.write(",".join(f"{_v:.6f}" if isinstance(_v, float) else str(_v)
                                           for _v in _r) + "\n")
                _nd = sum(1 for _r in _s1_trace_rows if _r[7] < 0.0)          # 命令朝上
                _nda = sum(1 for _r in _s1_trace_rows
                           if isinstance(_r[11], float) and _r[11] < 0.0)      # ★ 实际朝上
                logger.info(
                    "[Stage1 CSV] 逐帧位姿+开合已存: %s (%d 帧) | "
                    "**命令**朝上(dot<0) 帧数=%d %s | **实际**朝上 帧数=%d %s",
                    _csv_path, len(_s1_trace_rows), _nd,
                    "(命令很好 ✓)" if _nd == 0 else "✗",
                    _nda,
                    "← ★ PD 没跟上, 这就是视频里看到的'朝上' ✗"
                    if _nda else "✓ 实际全程朝下")
        except Exception as _e:  # noqa: BLE001
            logger.warning("[Stage1 CSV] 写出失败: %s", _e)

        # ★ 2026-09-14 P0: 录像完整性断言 —— 不许再"静默产出 44 字节空 MP4"
        if record_video:
            _nf = int(_s1_video_frames[0])
            if _nf == 0:
                logger.error(
                    "[Stage1 录像] rollout 结束但**一帧都没写** (期望 %d 帧) ✗ — "
                    "产出的 mp4 是 44 字节空壳; 根因几乎总是 scene._render_available=False",
                    n_frames)
            elif _nf < n_frames:
                logger.error(
                    "[Stage1 录像] 帧数不足: 实际 %d / 期望 %d (缺 %d 帧) ✗",
                    _nf, n_frames, n_frames - _nf)
            else:
                logger.info("[Stage1 录像] 写帧数 %d / %d ✓", _nf, n_frames)

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

        # ★ 2026-09-13: 夹爪**闭合轴** (世界系) —— 供 stage1_reward 的 gate 3 姿态偏好使用.
        #   手指沿夹爪局部 Y 收拢 ⇒ 闭合轴 = R_base @ [0,1,0].
        #   旋转约定与虚拟关节一致: R = Rz(rz) · Ry(ry) · Rx(rx) (见 rotmat_to_zyx_euler).
        #   失败则置 None → 偏好项自动跳过 (不否决任何候选, 硬性约束不变).
        _close_axis = None
        _approach_axis = None
        try:
            from scipy.spatial.transform import Rotation as _Rsci_ax
            # ★ 注意 scipy 约定: 大写 'ZYX' = **extrinsic** 依次绕 Z→Y→X 转,
            #   即 R = Rz(yaw)·Ry(pitch)·Rx(roll) —— 与 data_loader.rotmat_to_zyx_euler
            #   的 URDF 链定义一致。用小写 'zyx'(intrinsic) 会算错 (往返误差实测 0.302)。
            _R_base_ax = _Rsci_ax.from_euler("ZYX", [rz, ry, rx]).as_matrix()
            _close_axis = _R_base_ax @ np.array([0.0, 1.0, 0.0])
            _approach_axis = _R_base_ax @ np.array([1.0, 0.0, 0.0])   # X=手指伸出方向
        except Exception:  # noqa: BLE001
            _close_axis = None
            _approach_axis = None

        return {
            'carry_validation': _s1_val,
            'tracking_error': _s1_te,
            'gripper_close_axis': _close_axis,
            'gripper_approach_axis': _approach_axis,   # ★ 供 gate3 的"朝下"判据
            'entry_target_contact': bool(getattr(self, '_last_entry_target_contact', False)),
            'approach_contact_count': int(approach_contact_count),   # 抬升前的接触帧数 ✓
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
            # ★★★★ 2026-09-17 夹持对称性 (用户: "夹空这个确实需要改正一下") ★★★★
            #   两指被物体挡停的开合差 |act1-act2|:
            #     夹得正 (对着平面) ⇒ box 版实测 0.0127/0.0182 ⇒ 相差 5.5mm ✓
            #     斜着卡棱          ⇒ CoACD 版实测 0.0105/0.0304 ⇒ 相差 19.9mm ✗
            #   ⇒ 对称性差 = 夹得不正 ⇒ 携带必滑脱 ✗ (CoACD 版 173.97mm ✗)
            'finger_qpos_asym': abs(float(locked_val1) - float(locked_val2)),
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
        # ★★★★ 2026-09-17 对称性**奖励**已回滚 (实测有害 ✗) ★★★★
        #   实测 (output/coacd_sym): 把 |act1-act2| 放进 reward 后,
        #     Stage1 验证门 lift_final=[0,0,0]cm ⇒ **0/3 失败** ✗ (重试 2 轮也不通过 ✗),
        #     Stage2 反而更差 (dist2obj 32.2mm → 42.4mm ✗)
        #   ⇒ CEM 为追求"对称"牺牲了"抬升" ⇒ 找不到能抓起来的解 ✗
        #   ⇒ 结论: 对称性只能作**门禁筛选 / 软指标** ✓, 不能进 reward ✗
        #   (finger_qpos_asym 仍保留在 sim_result 里, 供判据使用 ✓)
        # ★★★★ 2026-09-17 接近段"入场即接触目标"惩罚 (用户: "可以的") ★★★★
        #   实测 (output/coacd_rot15): Stage1 把旋转搜索放宽到 ±15° 后选出的位姿
        #     "能抬起来 ✓" (lift 19.6/19.2cm, 2/3 通过 ✓)
        #     但 Stage2 **立刻**在 PRE_F50_READY 拦下 ✗ (q tracking error=3.06mm ✗)
        #     ⇒ 说明接近段手指就顶到了物体 ✗ (凸包下更常见)
        #   ⇒ Stage1 的 reward 只看抬升 ✗ ⇒ 会不断选出"能抬但接近段会撞"的偏科位姿 ✗
        #   修: 对 `entry_target_contact` 加惩罚 ✓
        #     —— 它与"抬升"**同向** ✓ (干净地接近才有机会抓稳), 不会像对称性那样打架 ✗
        # ★ 用户定策 (2026-09-17): "接近段是**不允许接触**的, 这个代价还设置少了"
        #   ⇒ 代价加重 ✓。接近段接触 = 撞着物体走过去 ⇒ 即便侥幸抬起也不可用 ✗
        if bool(sim_result.get('entry_target_contact', False)):
            r -= 80.0                                   # 入场即接触 ⇒ 重罚 (接近段禁止接触)
        # 注意: **不能**无条件按 contact_count 罚 ✗ —— 抬升段本来就必须接触 ✓
        #   (曾误写成 -3.0×contact_count ⇒ 100+ 帧 ⇒ -300 ✗, 会直接毁掉抬升奖励 ✗)
        #   只在"抬不起来却接触很多"时罚 ⇒ 那才是"撞着拖" ✓
        if float(sim_result.get('contact_count', 0)) > 25 and lift_final < 0.05:
            r -= 60.0
        # ★ 分阶段精确惩罚 (2026-09-17, 用户: "接近段是不允许接触的, 这个代价还设置少了")
        #   只罚**抬升前**的接触 ✓ ⇒ 权重可以设很大 ✓, 不会误伤抬升段必需的接触 ✓
        _appr_cc = float(sim_result.get('approach_contact_count', 0))
        if _appr_cc > 0.0:
            r -= 15.0 * _appr_cc
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

        # ── gate 3 ★ 2026-09-13 新增 (用户定策): 姿态 / 夹持方向偏好 ──
        #   用户原话: "对的, 就是夹持方向, 就是姿态" + "把姿态也考虑上去".
        #   背景: 现有 reward 只问"抬得起来吗" → 会选到**物体斜着卡在指缝里**的姿态.
        #     实测: 物体 |y|max=41.9mm 而 bbox 的 y=67.1mm (差 25mm) ⇒ 换真实指形后
        #     可用行程只剩 11.5mm → 夹不住 (Stage1 抬起仅 1.0cm).
        #   判据: 让手指的**闭合方向**尽量沿物体的**最短轴**——
        #     这样物体"正着"卡进指缝, 投影≈最短轴≈bbox 最小边, 行程最大化。
        #   score = |cos∠(闭合轴, 物体最短轴)| ∈ [0,1], 1 = 完美。
        #   注: 硬性约束(必须抬得起)不变 —— 本项只在候选之间排序, 不否决任何候选。
        try:
            _ctrls = getattr(self, "grasp_controllers", None) or {}
            _ctrl = _ctrls.get(self.side) if hasattr(_ctrls, "get") else None
            _tn = getattr(_ctrl, "target_obj", None) if _ctrl is not None else None
            _bbox = (
                (getattr(self, "obj_info", {}) or {}).get(_tn, {}).get("bbox_size")
                if _tn else None
            )
            _ax = sim_result.get("gripper_close_axis")  # 世界系闭合轴 (缺失则跳过本项)
            if _ax is not None and _bbox is not None:
                _ax = np.asarray(_ax, dtype=np.float64)
                _n = float(np.linalg.norm(_ax))
                _bb = np.asarray(_bbox, dtype=np.float64)
                if _n > 1e-9 and _bb.size == 3 and float(np.min(_bb)) > 0:
                    _ax = _ax / _n
                    # ★★★★ 2026-09-14 (方案A) 禁止**翻转解** ★★★★
                    #   实测证据 (final_q 日志):
                    #     [Stage2 track] MANO_FOLLOW n=23 rot=9.33/**169.02deg** ✗✗
                    #   ⇒ 前缀段 (MANO 姿态) 与接近段 (Stage1 锚点姿态) 差 ~169° ⇒ 夹爪**翻了半圈** ✗
                    #   ⇒ PD 物理上跟不住 → 甩动 → Stage3 搬运时物体被甩出 (drift=1333mm) ✗
                    #   根因: 上面的 _align 用 |cos| (**对称**), 而开合轴的两个方向
                    #         (±ê, 即 Y 与 -Y) 对"对齐"是**等价**的 ✗ ⇒ 优化器可能选中
                    #         与 MANO 反向 180° 的那个解 ✗ (align 分一样高 ✗).
                    #   修复: 用 **MANO 在 F50 的闭合轴**作为参考,
                    #         若候选闭合轴与它反向 (点积 < 0) ⇒ 是翻转解 ⇒ 对齐分清零 ✓
                    #   这样"对齐但不翻转"仍得满分, "对齐但翻转"得 0 ✓
                    _flip = False
                    try:
                        _traj_m = (getattr(self, "_mano_gripper_traj", None) or {}).get(
                            getattr(self, "side", "left"), {})
                        _R_m = np.asarray(_traj_m.get("R"), dtype=np.float64)
                        _f50i = int(getattr(self, "_frame_params", {}).get("F50_IDX", 50))
                        if _R_m.ndim == 3 and _R_m.shape[0] > _f50i:
                            _ax_ref = _R_m[_f50i][:, 1]              # MANO 闭合轴
                            if float(np.dot(_ax, _ax_ref)) < 0.0:
                                _flip = True
                    except Exception:  # noqa: BLE001
                        _flip = False
                    # ★★★★ 2026-09-14 用户关键指正 ★★★★
                    #   "夹爪的旋转角度好像并没有要求…位姿都一样, 但是夹爪中间的旋转是不一样的"
                    #   查证结果: 夹爪只有 2 个旋转自由度被约束
                    #     X(接近轴): align_gripper_approach_to_gravity 强制朝下 ✓
                    #     Y(开合轴): 只要求"水平" ✗ —— **绕 X 的 roll(方位角) 完全自由** ✗
                    #   原因就在这里: 旧代码拿"物体**全局**最短轴"当作对齐目标,
                    #     而本物体的全局最短轴(轴2, 39.3mm)是**竖直**的 ✗
                    #     ⇒ Y 必须水平 ⇒ |Y·ê_竖直| ≈ 0 恒成立 ⇒ 该项**永远≈0** ✗
                    #     ⇒ 姿态 reward 实际上**没有约束 Y 的方位角** ⇒ 绕 X 自由旋转 ✗✗
                    #   修复: 只在物体的**水平轴**里挑最短的那根作为对齐目标 ✓
                    #     (物理含义: 手指从最窄的水平方向合拢, 行程最短、最不容易滑)
                    #   重力方向 g=(0,0,-1); 某轴的"水平度" = 1 - |axis·ĝ|
                    _gw = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                    _Ro = sim_result.get("obj_R")          # 物体世界旋转 (缺省按单位阵)
                    _Ro = np.eye(3) if _Ro is None else np.asarray(_Ro, dtype=np.float64)
                    _horiz_axes = []                        # (边长, 水平度, 世界系轴向量)
                    for _i in range(3):
                        _av = _Ro[:, _i]
                        _h = 1.0 - abs(float(np.dot(_av, _gw)))
                        _horiz_axes.append((float(_bb[_i]), _h, _av))
                    # 先筛"确实水平"的轴 (水平度 > 0.7, 即与竖直夹角 < 45°)
                    _pool = [t for t in _horiz_axes if t[1] > 0.7]
                    _src = "水平轴" if _pool else "全局轴(无水平轴, 回退)"
                    if not _pool:
                        _pool = _horiz_axes
                    _pick = min(_pool, key=lambda t: t[0])   # 其中最短的
                    _e = np.asarray(_pick[2], dtype=np.float64)
                    _align = abs(float(np.dot(_ax, _e)))
                    # ★★★ 2026-09-13 关键修正 (用户: "之前 stage0 跑的效果很好, 基本上都是向下的"):
                    #   上面用 |cos| 是**对称**的 → 朝上夹与朝下夹得分相同 ✗
                    #   而 physics_utils.align_gripper_approach_to_gravity 已保证 Stage0 的
                    #   夹爪 X (手指伸出方向) 朝下; Stage1 的 CEM 若把它转歪, 就会出现
                    #   "夹爪上下不对". 因此这里把"朝下"作为**方向约束**补上:
                    #     X_gripper · g >= 0  (X 与重力同向 = 朝下) → 满分方向项
                    #     否则按夹角余弦线性扣分 (最差 -12, 即整项抵消)
                    #   g = 世界重力方向 = (0,0,-1) (SAPIEN)
                    # X 轴从 sim_result 取 (不能跨函数用 _R_base_ax)
                    _X_ax = sim_result.get("gripper_approach_axis")
                    _down = 0.5   # 缺数据时按中性, 不误罚
                    if _X_ax is not None:
                        _xa = np.asarray(_X_ax, dtype=np.float64)
                        _nn = float(np.linalg.norm(_xa))
                        if _nn > 1e-9:
                            _xg = float(np.dot(_xa / _nn, np.array([0.0, 0.0, -1.0])))
                            _down = max(0.0, _xg)   # 1=完全朝下, 0=水平/朝上
                    # ★★★★ 2026-09-14 用户报告"反向"的关键修复 ★★★★
                    #   症状: 实测闭合轴 = [0.003, -0.662, -0.75] → **z=-0.75 朝下** ✗
                    #         ⇒ 两个手指一个在物体上方、一个在下方 ("上下夹" = 视觉上的"反向") ✗
                    #         ⇒ 正常抓取闭合轴必须**水平** (左右夹) ✓
                    #   根因链 (两条独立机制叠加):
                    #     ① 物体最短轴(39.3mm)刚好是**竖直**的 → 本项 align 奖励
                    #        "闭合轴对齐最短轴" → 把夹爪转成上下夹 ✗
                    #     ② physics_utils.align_gripper_approach_to_gravity 保证的是
                    #        夹爪 **X**(手指伸出方向) 朝下, 并不约束闭合轴 Y 的朝向 ✗
                    #        ⇒ X 朝下 + Y 竖直 是自洽的, 于是"上下夹"被放行 ✗
                    #   修复: 给闭合轴加**水平约束** —— |cos∠(Y, g)| 越大罚越狠,
                    #         竖直闭合轴直接把这个 +12 分项清零 (不再把夹爪转歪).
                    #   g = 世界重力方向 (0,0,-1), 竖直度 = |Y·ĝ| ∈ [0,1]
                    _vert = abs(float(np.dot(_ax, np.array([0.0, 0.0, -1.0]))))
                    _horiz = 1.0 - _vert            # 1=完全水平(理想), 0=完全竖直(反向 ✗)
                    # ★ 方案A: 翻转解 (与 MANO 闭合轴反向) 直接清零 ✓
                    if _flip:
                        _align_eff = 0.0
                        logger.warning(
                            "[Stage1 防翻转] 闭合轴 %s 与 MANO 参考反向 → **翻转解** ✗ "
                            "对齐分已清零 (原 align=%.3f)", np.round(_ax, 3).tolist(), _align)
                    else:
                        _align_eff = _align * _horiz
                    _align_eff = 0.0 if _flip else _align_eff
                    r += 12.0 * _align_eff * (0.25 + 0.75 * _down)
                    sim_result["pose_align_score"] = _align
                    sim_result["pose_down_score"] = _down
                    sim_result["pose_horiz_score"] = _horiz
                    sim_result["pose_flipped"] = bool(_flip)
                    if _vert > 0.5:
                        logger.warning(
                            "[Stage1 反向告警] 闭合轴 %s 竖直度=%.2f (>0.5) → **上下夹** ✗ "
                            "本候选姿态分已被清零 (对齐项=%.3f, 水平项=%.3f)",
                            np.round(_ax, 3).tolist(), _vert, _align, _horiz)
                    # 闭合轴方位角 (绕 X 的 roll 的体现) —— 用它监测"位姿一样但旋转不同" ✓
                    _az = float(np.degrees(np.arctan2(_ax[1], _ax[0])))
                    sim_result["pose_close_azimuth_deg"] = _az
                    logger.info(
                        "[Stage1 姿态] 闭合轴=%s 竖直度=%.2f 水平度=%.2f 方位角=%.1f° | "
                        "对齐目标(%s, 边长%.0fmm) align=%.3f → 有效=%.3f (+%.1f分) bbox=%s",
                        np.round(_ax, 3).tolist(), _vert, _horiz, _az, _src,
                        float(_pick[0]) * 1000.0, _align, _align_eff,
                        12.0 * _align_eff * (0.25 + 0.75 * _down),
                        np.round(_bb, 4).tolist())
        except Exception:  # noqa: BLE001
            pass

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

        # ===== Stage 0: 候选生成 =====
        # 2026-09-12 (用户定策 C): 默认改为 grasp_frame ± 15 **逐帧扫描** —— 每帧取该帧
        # MANO 夹爪姿态 + 三维位置微调, 再由 Stage1a 预筛 + CEM 物理打分, reward 选最优.
        # 旧行为 (姿态锁死 F50 + 240 位置网格) 可用 _STAGE1_MULTIFRAME=0 回退.
        logger.info("[Stage1] ===== Stage 0: 生成候选 =====")
        candidates = []
        if os.environ.get("_STAGE1_MULTIFRAME", "1") == "1":
            candidates = self.generate_grasp_candidates_multiframe(
                side, half_span=int(os.environ.get("_STAGE1_FRAME_SPAN", "15")))
        if not candidates:
            candidates = self.generate_grasp_candidates(side, n_candidates=240, base_R=mano_f50_R)
        if not candidates:
            logger.error("[Stage1] Stage 0 未生成候选")
            return None

        # ===== Stage 1a: Grasp Quality 预筛选 (无仿真) =====
        logger.info("[Stage1] ===== Stage 1a: Grasp Quality 预筛选 =====")
        # ★★★★ 2026-09-14 用户: "stage1肯定得考虑碰撞, 才能把桌面的物体夹起,
        #      不然肯定每次测试优化都失败了" + "stage1的位姿优化现在应该考虑障碍物的碰撞问题" ★★★★
        #   现状: Stage1a 预筛只看 (q_score 抓取质量 ✓ / hard 硬约束 ✓ / mano 贴合 ✓)
        #         **完全不看夹爪与桌面/其他物体的碰撞** ✗
        #         ⇒ 优化器会选出"手指插进桌面"的位姿 ✓ → 物理里撞桌 → 物体被带倒 ✓
        #           (实测: 开桌面碰撞后 Stage1 恒定距离 max_dev=**168.5mm** ✗ 物体掉 16.8cm ✓)
        #   做法: 用与 N-5 **同一套 OBB 几何** (与物理实测误差 0.2mm ✓) 评估每个候选的
        #         手指-障碍物穿透, 超过阈值的候选直接标 hard_ok=False ✓
        _n5_avail = True
        try:
            from stage2_approach_opt import (
                _FINGER_HALF_EXTENT, _FINGER_HALF_GAP, _FINGER_LOCAL_CENTER, obb_aabb_signed,
            )
        except Exception:  # noqa: BLE001
            _n5_avail = False
        _ground_z_s1 = float(getattr(self, "_ground_z", 0.0))
        _obst_s1 = []                       # (center(3,), half(3,)) 障碍物 AABB 列表
        if _n5_avail:
            # ① 桌面: ground_z 下方一块大平板 (与 N-5 的障碍物模型一致 ✓)
            _obst_s1.append((np.array([0.0, 0.0, _ground_z_s1 - 0.05], dtype=np.float64),
                             np.array([0.5, 0.5, 0.05], dtype=np.float64)))
            # ② 其他物体 (non-target)
            for _a in (getattr(self, "obj_actors", []) or []):
                _nm = _a.get_name()
                if _nm == target_name:
                    continue
                try:
                    _p = np.asarray(_a.get_pose().p, dtype=np.float64)
                    _h = np.asarray((getattr(self, "obj_info", {}) or {}).get(
                        _nm, {}).get("bbox_size", [0.04, 0.04, 0.04]), dtype=np.float64) / 2.0
                    _obst_s1.append((_p, _h))
                except Exception:  # noqa: BLE001
                    continue
        logger.info("[Stage1 碰撞预筛] 障碍物 %d 个 (桌面 + non-target), N-5 几何可用=%s",
                    len(_obst_s1), _n5_avail)
        scored_candidates = []
        for cand in candidates:
            q_score, hard_ok, details = self.score_grasp_quality(cand, target_name)
            mano_ok = _check_mano_hard_constraint(cand['pos'], cand['R'], ref_R_for_constraint=mano_f50_R)
            # ★ 障碍物穿透检查 (两指 OBB vs 障碍物 AABB ✓)
            _obs_pen = 0.0
            if _n5_avail and _obst_s1:
                try:
                    _Rc = np.asarray(cand['R'], dtype=np.float64)
                    _pc = np.asarray(cand['pos'], dtype=np.float64)
                    _hg = _FINGER_HALF_GAP + float(stage_q_open(
                        float(max((getattr(self, "obj_info", {}) or {}).get(
                            target_name, {}).get("bbox_size", [0.04, 0.04, 0.04])))))
                    for _sg in (+1.0, -1.0):
                        _lk = _pc + _Rc @ np.array([0.03689, -_hg * _sg, 0.0])
                        _lc2 = _FINGER_LOCAL_CENTER.copy()
                        _lc2[1] *= _sg
                        _bc = _lk + _Rc @ _lc2
                        for (_oc, _oh) in _obst_s1:
                            _d = obb_aabb_signed(_bc, _Rc, _FINGER_HALF_EXTENT, _oc, _oh)
                            if _d < 0.0:
                                _obs_pen = max(_obs_pen, -_d)
                except Exception:  # noqa: BLE001
                    _obs_pen = 0.0
            if _obs_pen > 0.0:
                hard_ok = False            # 手指穿进障碍物 ⇒ 该候选直接淘汰 ✓
            passed = hard_ok and mano_ok
            scored_candidates.append({
                'cand': cand,
                'q_score': q_score,
                'hard_ok': hard_ok,
                'mano_ok': mano_ok,
                'details': details,
                'passed': passed,
            })
            if len(scored_candidates) < 20:  # ★ 7533 候选时不再逐条刷屏 (用户: "目前输出太多了")
                logger.info(
                    "  [cand#%d] dx=%.4f, dy=%.4f, q_score=%.2f, hard=%s, mano=%s, pass=%s",
                    len(scored_candidates),
                    cand.get('intended_finger_offset', [0, 0, 0])[0],
                    cand.get('intended_finger_offset', [0, 0, 0])[1],
                    q_score, hard_ok, mano_ok, passed)

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
