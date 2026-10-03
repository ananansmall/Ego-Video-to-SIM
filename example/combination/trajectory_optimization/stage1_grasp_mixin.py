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
    # ★ 2026-09-28 补导入 ✓: `_export_stage1_recipe` 里用相位名切 close 段
    #   (修 `close_ctrl_q` 写全段 ✗ 的 bug, md §04) ⇒ 缺它 ⇒ NameError ⇒ 7 个 t07e 回归全红 ✗
    STAGE1_PHASE_CLOSE,
    STAGE1_PHASE_SETTLE,
    build_physics_config,
    build_scene_digest,
    build_stage1_grasp_recipe,
    stage1_close_q_sequence,
    stage_q_open,
)

logger = logging.getLogger("grasp_hawor")


class Stage1Mixin:
    """Stage 1: 抓取姿态优化 Mixin"""

    # === adaptive_grip_overclosure (2026-09-18 方案 B) ===
    def adaptive_grip_overclosure(self, q_contact, fallback=None):
        """**按物体尺寸自适应**的过盈量 (用户 2026-09-18 选定方案 B ✓).

        为什么需要 ✓: 旧的固定 `_GRIP_OVERCLOSURE = 0.008` (8mm) ✗
          · 对 55.9mm 宽的小物体 ⇒ 手指被命令**嵌入 8mm** ✗
          · 而 CoACD 的 64 个**重叠凸包**在深嵌入时接触法向冲突 ⇒ **求解爆炸** ✗
            (实测: act 一到 0.0107 ⇒ 物体 4 个控制点内被弹射 **130mm** ✗)
          · 实测安全窗口 (output/dumps / floor* 系列, 55.9mm 物体, q_contact≈0.0127):
              act < 0.0107 ⇒ 爆炸 ✗  (嵌入 > 2.0mm ✗)
              act ≈ 0.0127 ⇒ 刚接触 ✓ (嵌入 = 0)
            ⇒ **安全嵌入区间 = (0, 2.0)mm** ✓
        本函数 ✓: 过盈 ∝ 物体尺寸, 并**硬性限制在 (0.8, 2.2)mm 内** ✓
          · 55.9mm 物体 ⇒ 1.68mm ✓ (落在实测安全区 ✓, 且足以压出正压力 ✓)
          · 28mm 物体   ⇒ 1.0mm ✓ (下限保底 ✓)
          · 100mm 物体  ⇒ 2.2mm ✓ (不超上限 ✓)
        开关 ✓: `_GRIP_OVERCLOSURE_MODE=fixed` ⇒ 完全回到旧行为 (0.008) ✓
                `_GRIP_OVERCLOSURE_COEF` 可调比例 (默认 0.03) ✓
        """
        _mode = os.environ.get("_GRIP_OVERCLOSURE_MODE", "adaptive")
        if _mode != "adaptive":
            return float(_GRIP_OVERCLOSURE if fallback is None else fallback)
        _coef = float(os.environ.get("_GRIP_OVERCLOSURE_COEF", "0.03"))
        # q_contact = obj_max_dim/2 − FINGER_EFFECTIVE_HALF_SPACING ⇒ 反推 obj_max_dim ✓
        _obj_max_dim = 2.0 * (float(q_contact) + float(FINGER_EFFECTIVE_HALF_SPACING))
        return float(min(max(_coef * _obj_max_dim, 0.0008), 0.0022))

    # === compute_gripper_qpos (L4961) ===
    def compute_gripper_qpos(self, obj_bbox_size, close_half_width=None):
        """根据物体 bbox 最大水平边计算夹爪开合度目标值

        返回: gripper_qpos 使手指能真正接触并挤压物体表面。
        使用最大水平边(而非最小边), 保证平行指在任意水平朝向都能碰到物体;
        目标 qpos 比几何接触点再小 _GRIP_OVERCLOSURE (默认 0.5mm),
        让 PD 驱动器在接触后仍有位置误差, 从而产生持续正压力。

        ★★★★ 2026-09-20 【用户】"现在的位姿说实话效果很好了, 就是**闭合尺度太小了**" ★★★★
          诊断 (output/cost3 实测) ✓:
            `目标 glb_4, bbox=[0.0449, 0.0559, 0.0393] ⇒ compute_gripper_qpos=0.0111`
            `接触 qpos=0.0072, PD_target=**0.0059**` ⇒ 指缝半间距 = 0.0152+0.0059 = **21.1mm**
            ⇒ 65mm 行程**只用了 ~11%** ✗ ⇒ 手指几乎不动 ⇒ 看起来"没闭合" ✗
          双重缩水 ✗:
            ① `bbox[:2]` 取的是 **bbox 最大水平边** (5.59cm ✓), 而**真实闭合轴**上的
               凸包支撑半宽可能完全不同 ⇒ 与接触面**不匹配** ✗
            ② 目标被 `_OBJ_SCALE=0.5` 缩小后 bbox 本身就只有 5.59cm ⇒ 开度必然小 ✗
          修法 ✓: 允许调用方传入 **CoACD 凸包在闭合轴上的真实支撑半宽** (`close_half_width` ✓,
            由 `score_grasp_quality` 算出的 `coacd_close_half_width_m` 提供 ✓)
            ⇒ `q = close_half_width − FINGER_EFFECTIVE_HALF_SPACING + 过闭合量` ✓
            ⇒ 手指一定**真正压到物体表面** ✓ (而不是悬在两侧 ✗)
          开关 ✓: `_S1_CLOSE_FROM_HULL=0` 回到旧 bbox 行为 (对照 ✓);
                  `_GRIP_OVERCLOSURE_MM` 可调过闭合量 (默认 0.5mm ✓)
        """
        _oc_env = os.environ.get("_GRIP_OVERCLOSURE_MM")
        _oc_forced = float(_oc_env) / 1000.0 if _oc_env not in (None, "") else None
        if close_half_width is not None:
            _q_c = float(close_half_width) - FINGER_EFFECTIVE_HALF_SPACING
            # ★ 2026-09-20 【方向钉死】过闭合必须**减** (夹得更深 ✓) 不是加 ✓
            #   力学: 目标 q 越小 ⇒ PD 越想闭合 ⇒ 被物体挡停 ⇒ 产生正压力 ✓
            #   目标 q 越大 ⇒ 手指停在物体**外面** ⇒ 零压力 ⇒ 抬起即滑落 ✗
            #   默认 2.0mm (= 实测安全窗口上限 ✓, 再深会因 64 重叠凸包法向冲突而弹射 ✗)
            _oc = (_oc_forced if _oc_forced is not None else 0.0020)
            # ★★★★ 2026-09-20 【P0】2.0mm 硬上限改为**可配置** (原来是写死的 ✗) ★★★★
            #   为什么必须放开 (实测铁证 ✓): 这个 `min(_oc, 0.0020)` 把过闭合**钉死在 2.0mm**,
            #     ⇒ `_S1_CLOSE_OVERCLOSURE_MAX_MM` / `_GRIP_OVERCLOSURE_MM` **全都无效** ✗
            #     ⇒ 11 组实验 (E0~E7/C1~C3/F1~F3) 的 `gripper_qpos` **逐位相同 (0.0111)** ✗,
            #       C3 设了 `上限 4.0mm` 但日志仍是 `过闭合 2.0mm` ✗ ⇒ **闭合深度从未被真正试过** ✗
            #   而上限 2.0mm 的"安全窗口"结论是在**闭合轴竖直 (上下夹)** 的旧配置下测的 ✗,
            #     那个配置本身已被证明是错的 (竖直度 0.71~0.76 ⇒ 指尖插桌面 ⇒ 429~742N ✗);
            #     现在闭合轴已水平化 (竖直度 0.00 ✓) ⇒ 该结论**不必然成立**, 必须重测 ✓
            #   实测依据 (output/f1_fgnd_0920): 过闭合 2.0mm ⇒ 模型指缝 44.2mm,
            #     而手指**畅通无阻地到达 PD 目标** (`f1_qpos=0.0069` == `PD_target=0.0067` ✓,
            #     且 `f1_c=False` ✗) ⇒ **根本没夹到物体** ⇒ 物体始终 `obj_z=0.0195` 不动 ✗
            #     ⇒ 2.0mm 的过闭合对"在夹持高度上比最大跨度更窄"的物体**不足以建立接触** ✗
            #   开关 ✓: `_S1_OC_HARD_CAP_MM` (默认 2.0 = 旧行为逐位不变 ✓)
            _oc_cap_mm = float(os.environ.get("_S1_OC_HARD_CAP_MM", "2.0") or 2.0)
            _oc = min(_oc, _oc_cap_mm / 1000.0)
            return max(_GRIP_CLOSE, _q_c - _oc)
        _obj_max_dim = float(max(obj_bbox_size[:2])) if obj_bbox_size is not None else 0.035
        _q_contact = max(0.0, _obj_max_dim / 2.0 - FINGER_EFFECTIVE_HALF_SPACING)
        return max(_GRIP_CLOSE, _q_contact - self.adaptive_grip_overclosure(_q_contact))

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
        # ★★★★ 2026-09-18 【P0 判据修正】"不要碰物体" 必须用 **接近段(全开)半间距** ★★★★
        #   用户定调: "不是不能有代价 ... 就是**不要碰物体**, 用 coacd 优化一下" ✓
        #     ⇒ 这条判据问的是 **接近段(手指全开 q_open)会不会顶到物体** ✓
        #   旧行为 ✗: c_clear 用 `half_sep` (= 有效半间距 + **闭合目标开度** q_grasp≈0.005)
        #     ⇒ 半间距只有 0.0152+0.0047 = 19.9mm, 而物体沿闭合轴半宽 27.5mm ✓
        #     ⇒ c_clear = 27.5 − 19.9 = **+7.6mm > 0 恒定** ✗
        #     ⇒ **7533 个候选全部**被判"手指必然顶到物体" ✗ ⇒ 判据零区分度 ✗
        #       (实测 output/s1horiz: c_clear ∈ [+3.40, +8.03]mm, 无一为负 ✗;
        #        κ 因 k_clear 恒 = 1e-3 被压到 0.045~0.055 ✗ ⇒ "没有好的位姿"的假象 ✗)
        #   修正后 ✓: 半间距 = 0.0152 + q_open(≈0.05) = 65.2mm
        #     ⇒ c_clear = 27.5 − 65.2 = **−37.7mm** (>0 只表示"张不开 ⇒ 真会顶到" ✓)
        #     ⇒ 判据恢复区分度 ⇔ 允许"很多位姿" ✓ (与用户判断一致 ✓)
        #   开关 ✓: `_S1_CLEAR_REF=close` 回到旧行为 (仅作对照 ✓)
        if os.environ.get("_S1_CLEAR_REF", "open") == "close":
            half_sep_approach = half_sep
        else:
            _w_open = None
            try:
                if obj_bbox_size is not None and len(obj_bbox_size) >= 3:
                    _w_open = float(max(obj_bbox_size))
            except Exception:  # noqa: BLE001
                _w_open = None
            half_sep_approach = FINGER_EFFECTIVE_HALF_SPACING + float(stage_q_open(_w_open))

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
        # ★★★★ 2026-09-20 【P0 计量基准修正】凸包必须与**同一个物心**配对 ★★★★
        #   实测 (output/cctrav3, 同一次运行的三次 Stage1 尝试, max_hull=64):
        #     第1次: 闭合轴范围=[−18.8, +18.8]mm  → c_clear=−46.6mm ✓   (物体 3.8cm ✓ 自洽)
        #     第2次: 闭合轴范围=[+130.4,+169.2]mm  → c_clear=+112.7mm ✗  (量出 17cm ✗ 荒谬)
        #     第3次: 闭合轴范围=[+127.8,+166.5]mm  → c_clear=+200.8mm ✗
        #   根因 ✗: 凸包读的是 actor 的**当前世界位姿** ✓, 减掉的却是**加载时缓存**的
        #     `self.obj_bbox_centers` ✗。上一次尝试的 rollout 把物体甩飞 13~19cm 后
        #     **没有复位**就进入下一次尝试 ⇒ 物心过期 ⇒ 三项几何量全部爆炸 ✗
        #     ⇒ κ 从 0.54 崩到 0.026 ⇒ **所有候选都被判"会碰物体"** ✗ ⇒ 搜索被毒化 ✗
        #   修法 ✓: 以 **actor 当前位姿的平移分量** 作为物心 (与凸包同源、同帧 ✓),
        #     并对"与缓存物心的偏差"打警告 ⇒ 一旦偏移过大, 在日志里立刻可见 ✓
        _ctr_now = None
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
                # 顶点在加载时已居中于 bbox 中心 ⇒ actor 平移 = 当前物心 ✓ (同源同帧 ✓)
                _ctr_now = _H[:3, 3].astype(np.float64).copy()
                _drift_mm = float(np.linalg.norm(
                    _ctr_now - np.asarray(obj_center_3d, dtype=np.float64))) * 1000.0
                if _drift_mm > 5.0:
                    logger.warning(
                        "[CoACD质量] ★ 物体已偏离缓存物心 %.1fmm (当前=%s 缓存=%s) "
                        "⇒ 本函数已改用**当前物心**计量 ✓ (否则几何量会虚假爆炸 ✗); "
                        "如需干净基准, 应在每次 Stage1 尝试前复位物体 ✓",
                        _drift_mm, np.round(_ctr_now, 4).tolist(),
                        np.round(np.asarray(obj_center_3d, dtype=np.float64), 4).tolist())
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
                # ★ 计量基准 = **当前物心** (与凸包同源同帧 ✓), 退化时才用缓存物心 ✓
                _ctr_meas = np.asarray(
                    obj_center_3d if _ctr_now is None else _ctr_now, dtype=np.float64)
                _rel = np.concatenate(
                    [h - _ctr_meas for h in _hulls_w],
                    axis=0)
                _s = _rel @ close_dir            # 沿闭合轴的有符号坐标 (物心=0)
                _f = _rel @ forward_dir          # 沿接近方向
                _lo, _hi = float(_s.min()), float(_s.max())
                # ---- 三项"有区分度"的几何代价 ✓ (上一版 c_center/c_spread 因对称而恒 0 ✗, 已重写 ✓) ----
                #   c_center: **手指原点**与**物心**在闭合轴上的偏移 ✓ (真·对中 ✗)
                _fo_s = float(np.dot(np.asarray(finger_origin, dtype=np.float64)
                                     - _ctr_meas, close_dir))
                c_center = abs(_fo_s) * 1000.0                              # mm
                #   c_clear : 接近段手指面(**全开半间距** half_sep_approach)到凸包最近横向支撑的距离 ✓
                #     · <0 = 全开时能包住物体, 接近不会碰 ✓ (大多数位姿应当如此 ⇒ 选择权很大 ✓)
                #     · >0 = 全开也包不住 ⇒ 接近必然顶到物体 ✗ (真·不可行 ✓)
                c_clear = (max(abs(_lo), abs(_hi)) - half_sep_approach) * 1000.0   # mm
                #   c_axial : 指尖沿接近方向相对凸包轴向范围的位置 ✓ (太大=够不到 / 太小=扎进去 ✗)
                c_axial = (float(np.dot(np.asarray(finger_origin, dtype=np.float64)
                                        - _ctr_meas, forward_dir))
                           - 0.5 * (float(_f.min()) + float(_f.max()))) * 1000.0   # mm
                _n_vert = int(len(_rel))
                # ★★★★ 2026-09-20 【P0 回归修复】"基座高度"必须用**凸包**算, 不能整段丢掉 ★★★★
                #   实测对比 (同一物体/scale, 唯一过门那次 vs 本轮):
                #     `output/s1show` (过门 3/3, lift=21.3cm): 候选 base_z ≈ **0.0509~0.0513** ✓
                #     `output/ccacd1`  (CoACD 排序后):          候选 base_z ≈ **0.0375** ✗ (−13mm)
                #   原因 ✓: 我把 bbox 版四项分整体换成 CoACD 分时, **连带丢掉了 `z-safety`** ✗
                #     ——它是当时唯一约束"基座要高过物体顶面"的项 ⇒ 没有了它, 搜索一路往下跑 ✗
                #   修法 ✓: 用**凸包的世界 z 范围**重算同一个量 (满足"不用 bbox" ✓):
                #     c_zgap = base_z − max(凸包 z)   (mm, >0 = 基座在物体之上 ✓)
                _zmax_h = max(float(np.asarray(h_w, dtype=np.float64)[:, 2].max())
                              for h_w in _hulls_w)
                c_zgap = (float(np.asarray(base_pos, dtype=np.float64)[2]) - _zmax_h) * 1000.0
                # ★ 偏心门**按被控物体自适应** ✓ (用户: "需要根据被控物体判断, 鲁棒性要强" ✓)
                #   默认 = 物体沿闭合轴半宽的 **10%** (下限 1.5mm) ✓
                #   本物体: 半宽 ≈18.8mm ⇒ 门 ≈1.9mm ✓ (与之前手调的 2mm 一致 ✓)
                #   换更小/更大的物体 ⇒ 门自动跟着变 ✓ (不再是一个写死的 2mm ✗)
                _ctr_adapt = max(1.5, 0.10 * max(abs(_lo), abs(_hi)) * 1000.0)
                # ★★★★ 2026-09-20 【P0】"物体是否真的落在**两指夹持带**内" ★★★★
                #   为什么要它 ✓ (实测 output/final_d1 逐帧证据):
                #     选出的位姿 CoACD 报"对中 ≤2.5mm ✓", 但物理里
                #       F40/F50: **f1_c=True, f2_c=False** ⇒ **只有一根手指擦到, 另一根全程没碰** ✗
                #       obj_lift 全程 0 ⇒ 不是夹住, 是"单侧擦过" ✗
                #   为什么旧判据抓不到 ✗: `c_center` 只量**沿闭合轴 (Y)** 的偏移 ✓,
                #     完全没量"物体是否落在两指之间的**夹持窗口**里"(沿接近轴 X 的深度 + 沿 Z 的高度)✗
                #     ⇒ "手指平面与物体错位"这类错误**测不出来** ✗ (正是 `_GRIP_CLOSE_AXIS_HORIZONTAL`
                #       只修姿态、没修位置所造成的那种错位 ✓)
                #   本判据 ✓: 把凸包顶点投到**夹爪坐标系**, 数落在夹持窗口内的点数 ✓
                #     x_rel ∈ [−jaw_depth, +0.02]  (指尖伸进物体的深度带 ✓)
                #     |y_rel| ≤ half_sep_approach (闭合轴方向落在两指之间 ✓)
                #     |z_rel| ≤ finger_half_z      (落在手指高度范围内 ✓)
                #   门 ✓: `_S1_JAW_MIN_PTS` (默认 8 个点; 0 = 关闭 ✓)
                _jaw_pts = None
                try:
                    _Xa = np.asarray(forward_dir, dtype=np.float64)    # 夹爪 X (接近轴)
                    _Ya = np.asarray(close_dir, dtype=np.float64)      # 夹爪 Y (闭合轴)
                    _Za = np.asarray(R, dtype=np.float64)[:, 2]        # 夹爪 Z (第三轴)
                    _jaw_depth = float(os.environ.get("_S1_JAW_DEPTH_M", "0.050") or 0.050)
                    _z_half = float(os.environ.get("_S1_JAW_Z_HALF_M", "0.019") or 0.019)
                    _rel_fo = _rel + (_ctr_meas - np.asarray(finger_origin, dtype=np.float64))
                    _xr = _rel_fo @ _Xa
                    _yr = _rel_fo @ _Ya
                    _zr = _rel_fo @ _Za
                    _in = ((_xr > -_jaw_depth) & (_xr < 0.020)
                           & (np.abs(_yr) <= half_sep_approach)
                           & (np.abs(_zr) <= _z_half))
                    _jaw_pts = int(np.count_nonzero(_in))
                    details['coacd_jaw_pts'] = _jaw_pts
                except Exception as _ej:  # noqa: BLE001
                    _jaw_pts = None
                    details['coacd_jaw_pts'] = None
                # ---- CoACD 版质量因子 κ ✓: 只在**启用了 CoACD 的目标**上生效 ✓ ----
                #   用户要求 (2026-09-18): "如果使用了 coacd, 那都得重新去判断评分了,
                #     不可能还用 bbox 近似算代价" ✓ ⇒ 这里用**凸包真实几何**修正总分 ✓
                #   未启用 CoACD 的目标 ⇒ 走原 bbox 路径, **行为完全不变** ✓ (保护成功版 ✓)
                _ctgt = str(os.environ.get("_COACD_TARGET", "") or "")
                _coacd_on_here = bool(_ctgt) and _ctgt.lower() != "none" and (
                    _ctgt in target_name)
                _kappa = None
                if _coacd_on_here:
                    _m_clear = max(0.0, -c_clear)                    # mm, >0 = 有安全余量 ✓
                    k_clear = min(1.0, _m_clear / 3.0)               # 余量 >=3mm 记满分 ✓
                    k_center = 1.0 / (1.0 + c_center / 2.0)          # 偏移 2mm 折半 ✓
                    k_axial = 1.0 / (1.0 + abs(c_axial) / 10.0)      # 轴向 10mm 折半 ✓
                    _kappa = float((max(k_clear, 1e-3) * k_center * k_axial) ** (1.0 / 3.0))
                    # ★ 注意: 本块跑在 antipodal/force-closure **计算之前** ✗
                    #   ⇒ 此刻 score 还是 0 ✗ ⇒ **不能在这里乘** ✗ (实测 "评分 0.000→0.000" ✓)
                    #   ⇒ 只把 κ 存进 details ✓, 由**函数末尾**在四项基础分算完后统一施加 ✓
                    details['coacd_kappa'] = round(_kappa, 4)
                    details['coacd_c_clear'] = round(c_clear, 3)
                    details['coacd_c_center'] = round(c_center, 3)
                    details['coacd_c_axial'] = round(c_axial, 3)
                    details['coacd_c_zgap'] = round(c_zgap, 3)
                    details['coacd_ctr_max_mm'] = round(float(_ctr_adapt), 3)
                    # ★★★★ 2026-09-20 【闭合尺度】凸包在**闭合轴**上的真实支撑半宽 (m) ✓ ★★★★
                    #   用户: "位姿效果很好了, 就是**闭合尺度太小**了" ✓
                    #   `c_clear = (max|support| − half_sep_approach)·1000` ⇒ 反解出真实半宽 ✓
                    #   ⇒ Stage1 用它作为 PD 闭合目标 (`compute_gripper_qpos(close_half_width=...)`) ✓
                    #   ⇒ 不再用**bbox 最大水平边**猜 ✗ (与真实接触面不匹配 ✗)
                    details['coacd_close_half_width_m'] = round(
                        float(max(abs(_lo), abs(_hi))), 6)
                    # ★ 2026-09-18 供 Stage1 reward 取用**同一来源**的关键判据 ✓
                    #   位置理由: 本函数手上正好有候选位姿 ✓ (rollout 侧未必有 ✓)
                    #     ⇒ 暂存到 self, 由 rollout 的 sim_result 拼接 → stage1_reward 读 ✓
                    #   正式版应改为"按候选逐一配对" ✓ (这里是先打通链路的低风险版本 ✓)
                    self._last_coacd_criteria = {
                        'coacd_c_center_mm': float(c_center),
                        'coacd_c_clear_mm': float(c_clear),
                        'coacd_c_axial_mm': float(c_axial),
                        'coacd_kappa': float(_kappa),
                    }
                logger.info(
                    "[CoACD质量] target=%s 凸包=%d 顶点=%d | 闭合轴范围=[%+.1f, %+.1f]mm "
                    "指缝/2(闭合)=%.1fmm 指缝/2(全开)=%.1fmm | c_clear=%+.2fmm "
                    "c_center=%.2fmm c_axial=%+.2fmm "
                    "| κ=%s 基础分=%.3f (κ 在函数末尾统一施加 ✓) | 轴向范围=[%+.1f, %+.1f]mm",
                    target_name, len(_hulls_w), _n_vert,
                    _lo * 1000.0, _hi * 1000.0,
                    half_sep * 1000.0, half_sep_approach * 1000.0,
                    c_clear, c_center, c_axial,
                    ("%.3f" % _kappa) if _kappa is not None else "n/a(未启用CoACD)",
                    float(score),
                    float(_f.min()) * 1000.0, float(_f.max()) * 1000.0,
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

        # ★★★★ 2026-09-20 【用户定策】Stage1 的候选排序 **必须由 CoACD 几何决定, 不再用 bbox** ★★★★
        #   用户原话 (2026-09-20): "**stage1 一定是用 coacd 排的**, 目前的情况, **不要再用 bbox 了**" ✓
        #   为什么必须换 ✓:
        #     上面四项基础分 (antipodal / force-closure / reachability / z-safety) 全部基于
        #     **bbox 近似** ✗ (`obj_radius = max(bbox)/2` ✗) ⇒
        #       · 对真实形状是"盲选" ✗ (看不到凸包的凹凸/棱/朝向 ✗)
        #       · 实测 (output/ccrep2): 全候选 `c_center 中位 = 6.25mm` (偏心),
        #         对中 <2mm 仅 **11.1%** ✗ —— 而 Top-K 完全按这四项分排序 ✗
        #       · 后果 (闭环证据 ✓): Top-32 全部 `drift≈4.2cm`、`lift 0.4~4cm` ✗
        #         ⇒ **闭合时沿切向把物体"搓"走** ⇒ 一个都没能干净抬起来 (0/3) ✗
        #   本改法 ✓ (只在**启用 CoACD 的目标**上生效 ✓; 未启用 ⇒ 原 bbox 路径一字不动 ✓):
        #     ① **评分 = CoACD 几何为主**:  `score = w·κ + (1−w)·k_reach`
        #          κ = (k_clear · k_center · k_axial)^(1/3) —— 全部来自**真实凸包顶点投影** ✓
        #          k_reach = exp(−d(MANO F50)/20mm) —— 它约束"跟随 MANO" ✓, 与 bbox 无关 ✓
        #     ② **硬门 (默认开启 ✓)**: c_clear ≤ 0 (接近段全开能包住) / c_center ≤ 2mm / |c_axial| ≤ 10mm
        #        这些正是"夹得正"的三个必要条件 ✓ ⇒ 不合格的**根本不进物理验证** ✓
        #     ③ 开关 ✓: `_S1_SCORE_SOURCE=bbox` 可整段回退旧行为 (对照用 ✓)
        #        `_S1_CENTER_MAX_MM` / `_S1_AXIAL_MAX_MM` / `_S1_COACD_SCORE_W` 可调 ✓
        _k_coacd = details.get('coacd_kappa')
        _coacd_score_on = (_k_coacd is not None
                           and os.environ.get("_S1_SCORE_SOURCE", "coacd") != "bbox")
        if _coacd_score_on:
            _k_reach = float(np.exp(-float(dist_3d) / 0.02))         # ∈ (0,1] ✓
            # ---- ② 硬门: 只留"物理上不可能"的三条 ⇒ 其余全部降为**代价** ✓ ----
            #   ★★★★ 2026-09-20 【用户定策】"桌面需要更改 ... 夹爪最好是**远离桌面**
            #     (至少在 stage1) ... 但是**物体在夹爪中心的代价更重要**" ★★★★
            #   ⇒ 三条**硬门** (真的做不到, 不该浪费物理预算 ✓):
            #       ① 全开都包不住物体 (c_clear > 0)  ⇒ 接近必然顶到物体 ✗
            #       ② 轴向够不到/扎进去 (|c_axial| 过大) ✗
            #       ③ 物体不在两指夹持带内 (jaw_pts 太少) ✗
            #   ⇒ 两条**降级为代价** (可调权重, 不再是 pass/fail ✓):
            #       ① 偏心 c_center  ⇒ 权重**最高** (用户: "更重要" ✓)
            #       ② 离桌面余量   ⇒ 权重**最低** (用户: "最好是远离" ✓, 软偏好 ✗ 硬门)
            #   ⇒ 旧版把 `z_gap < 5mm` 当**硬门** ✗ ⇒ 实测把 776~1893 个候选直接判死 ✗,
            #     而 `c_center` 只当 2.5mm 硬门 ⇒ 与用户"对中更重要"的优先级**正好相反** ✗
            _ctr_env = os.environ.get("_S1_CENTER_MAX_MM")
            _ctr_max = (float(_ctr_env) if _ctr_env not in (None, "")
                        else 6.0)                     # ★ 6mm 宽松门 (0 = 关闭 ⇒ 纯软代价 ✓)
            _ax_max = float(os.environ.get("_S1_AXIAL_MAX_MM", "10.0") or 0.0)
            _cl_s = details.get('coacd_c_clear')
            _cc_s = details.get('coacd_c_center')
            _ca_s = details.get('coacd_c_axial')
            # ---- ★ 2026-09-20 离桌面余量必须**在本函数内**算 ✓ ----
            #   坑 ✗: 它原先只在**调用方** (Stage1a 候选循环) 里算 ⇒ 那时本函数**已经返回** ✗
            #     ⇒ `details['ground_clear_mm']` 在这里恒为 None ⇒ `k_gnd` 恒 = 0.5 (中性) ✗
            #     ⇒ "远离桌面"这一项**形同不存在** ✗ (这次差点又犯同样的错 ✓)
            #   本判据 ✓: 无限水平面的 OBB 支撑函数 (精确解析 ✓, 与 scene.add_ground 一致 ✓)
            #     clearance = center_z − Σ_j|R[2,j]|·half_j − ground_z
            #   只在桌面碰撞**真开**时才算; 关闭时留 None ⇒ k_gnd 取中性 ✓
            if os.environ.get("_NO_GRIPPER_GROUND", "0") != "1":
                try:
                    from stage2_approach_opt import (
                        _FINGER_HALF_EXTENT, _FINGER_HALF_GAP, _FINGER_LOCAL_CENTER,
                    )
                    _Rg = np.asarray(R, dtype=np.float64)
                    _bp = np.asarray(base_pos, dtype=np.float64)
                    _gz = float(getattr(self, "_ground_z", 0.0))
                    _w_o = None
                    if obj_bbox_size is not None and len(obj_bbox_size) >= 3:
                        _w_o = float(max(obj_bbox_size))
                    _hg_c = float(_FINGER_HALF_GAP) + float(stage_q_open(_w_o))
                    _hfe = np.asarray(_FINGER_HALF_EXTENT, dtype=np.float64)
                    _sup_c = float(np.sum(np.abs(_Rg[2, :]) * _hfe))
                    _minc = None
                    for _sg in (+1.0, -1.0):
                        _lk = _bp + _Rg @ np.array([0.03689, -_hg_c * _sg, 0.0])
                        _lc2 = np.asarray(_FINGER_LOCAL_CENTER, dtype=np.float64).copy()
                        _lc2[1] *= _sg
                        _bc = _lk + _Rg @ _lc2
                        _c_f = float(_bc[2]) - _sup_c - _gz
                        _minc = _c_f if _minc is None else min(_minc, _c_f)
                    # 掌部/基座代理 (保守球近似 ✓)
                    _palm_r = float(os.environ.get("_S1_PALM_PROXY_M", "0.030") or 0.030)
                    _minc = min(_minc, float(_bp[2]) - _palm_r - _gz)
                    details['ground_clear_mm'] = round(_minc * 1000.0, 3)
                except Exception as _egc:  # noqa: BLE001
                    pass
            _gc_s = details.get('ground_clear_mm')     # 夹爪最低点 − 桌面 (mm ✓)
            if _cl_s is not None and float(_cl_s) > 0.0:
                hard_ok = False
                details['coacd_screen'] = 'REJECT(全开都包不住)'
            elif _ca_s is not None and _ax_max > 0.0 and abs(float(_ca_s)) > _ax_max:
                hard_ok = False
                details['coacd_screen'] = 'REJECT(轴向>%.1fmm)' % _ax_max
            elif _cc_s is not None and _ctr_max > 0.0 and float(_cc_s) > _ctr_max:
                hard_ok = False
                details['coacd_screen'] = 'REJECT(偏心>%.1fmm)' % _ctr_max
            else:
                _jaw_min = int(os.environ.get("_S1_JAW_MIN_PTS", "8"))
                _jp = details.get('coacd_jaw_pts')
                if _jaw_min > 0 and _jp is not None and int(_jp) < _jaw_min:
                    hard_ok = False
                    details['coacd_screen'] = 'REJECT(物体不在夹持带内: %d<%d点)' % (
                        int(_jp), _jaw_min)
            # ---- ① 评分: **对中主导** + 桌面软偏好 (权重全部可调 ✓) ----
            #   k_center: 1/(1 + c_center/2)  ⇒ 0mm→1.0, 2mm→0.5, 6mm→0.25 ✓
            #   k_clear : 全开余量 ≥3mm 记满 ✓
            #   k_axial : 轴向 10mm 折半 ✓
            #   k_reach : 跟随 MANO (20mm 折半 ✓, 与 bbox 无关 ✓)
            #   k_gnd   : 离桌面余量 ⇒ ≥GOOD 记满, ≤BAD 记 0, 中间线性 ✓ (软偏好 ✓)
            _k_center = (1.0 / (1.0 + float(_cc_s) / 2.0)) if _cc_s is not None else 1.0
            _m_clear = max(0.0, -float(_cl_s)) if _cl_s is not None else 3.0
            _k_clear = min(1.0, _m_clear / 3.0)
            _k_axial = (1.0 / (1.0 + abs(float(_ca_s)) / 10.0)) if _ca_s is not None else 1.0
            # ★★★★ 2026-09-21 【P0 标定错误修复】"贴桌面"绝不该是高分 ✗ ★★★★
            #   真值铁证 ✓ (output/s2_diag, Stage2 的 settle 段):
            #     `[Stage2 真夹持判据] 挡停量 指1=**+16.74**mm 指2=**+5.98**mm ⇒ 真夹持率 100% ✓
            #      两指**距离差=0.7mm** (对中完美 ✓) | **最近侧=12.5mm** ✗ | **指尖离桌面=0.0mm** ⚠`
            #     ⇒ 解释 ✓: **指尖正贴在桌面上** ⇒ 闭合时手指**先被桌面挡住**
            #       ⇒ 手指**离物体表面还有 12.5mm** 就停了 ✗ ⇒ **没有真正压紧物体** ✗
            #       ⇒ 于是 Stage2 的 CARRY (水平移动) **一启动就脱开** ✗
            #         (`CARRY: T_G_O drift=**177~192mm**` ✗)
            #     而 Stage1 那次能抬起来 ✓ 是因为它是**竖直**提升 (重力把物体压向指腹 ✓),
            #       不需要水平摩擦 ⇒ "贴桌面"在 Stage1 看不出问题 ✗ ⇒ 到 Stage2 才暴露 ✓
            #   根因 ✓: `_k_gnd` (离桌面余量) 的**标定反了** ✗
            #     旧值 `good=10mm / bad=-30mm` ⇒ 在 **0mm (贴桌面)** 处
            #       `k_gnd = (0-(-30))/(10-(-30)) = **0.75**` ⇒ **几乎满分** ✗✗
            #       ⇒ 于是 Stage1 的候选排序**专挑"指尖贴桌面"的位姿** ✗
            #   新标定 ✓ (按物理意义): **≥3mm 才满分**; **0mm ⇒ 低分** (0.25 ✓, 必然被桌面挡 ✗)
            #     ⇒ `_S1_GND_GOOD_MM` 默认 **10 → 3.0** ✓ / `_S1_GND_BAD_MM` 默认 **−30 → −1.0** ✓
            #   ⚠⚠ 2026-09-21 【实测回滚 —— 同一课第四遍】 ⚠⚠
            #     把上面两条标定真的改成 3.0 / −1.0 (output/s2_clr) ⇒ Stage1 验证门
            #     `0/3, 0/3` ✗✗ (改前是 2/3 ✓) ⇒ **又是"改求和项/权重 ⇒ 换盆地"** ✗
            #     ⇒ 已把默认值**回退**为 10.0 / −30.0 ✓ (与本轮唯一成功的 `iso_nofake` 一致 ✓)
            #   正确做法 ✓ (按 md §42 的三层设计): 离桌面余量走 **门 / 平局裁决** ✓,
            #     **不再动 reward 与候选评分的权重** ✗
            #   注 ✓: 与代码注释 "revise Stage1 instead" 一致 ✓
            #     (Stage2 的 `dz_grid` 改 anchor 已被移除 ⇒ 高度必须在 Stage1 决定 ✓)
            _gnd_good = float(os.environ.get("_S1_GND_GOOD_MM", "10.0") or 10.0)
            _gnd_bad = float(os.environ.get("_S1_GND_BAD_MM", "-30.0") or -30.0)
            if _gc_s is None:
                _k_gnd = 0.5                      # 桌面关闭时给中性分 (不惩罚 ✓)
            else:
                _gc = float(_gc_s)
                if _gc >= _gnd_good:
                    _k_gnd = 1.0
                elif _gc <= _gnd_bad:
                    _k_gnd = 0.0
                else:
                    _k_gnd = (_gc - _gnd_bad) / max(1e-6, (_gnd_good - _gnd_bad))
            # ★★★★ 2026-09-21 【G5 解析硬门 —— 手指必须能越过桌面 ✓】(md §46.6) ★★★★
            #   实测铁证 ✓ (output/s2_gate): Top-K 候选的**真值**是
            #     `指尖离桌面 = 0.23 ~ 2.01mm` (**全部 <3mm** ✗) ⇒ 而 +5mm 的 z 微调候选
            #     **确实在池子里** ✓ (`d_z_list = ±5mm` ✓) 却**进不了 Top-K** ✗
            #     ⇒ 因为"离桌面余量"在评分里只占 `_S1_W_GND=0.10` ✗ (且 0mm 时还有 0.75 分 ✗)
            #   ⇒ 本门 ✓: `ground_clear_mm ≥ _S1_GATE_GND_MIN_MM` (默认 **0 = 关** ✓)
            #     ⇒ 直接把"指尖贴桌面"的候选**在解析预筛就淘汰** ✓
            #     依据 ✓: 指尖贴桌面 ⇒ 闭合先被桌面挡 ⇒ **没压住物体** ✗
            #       ⇒ Stage1 的**竖直**提升能"托"起来 (看不出来 ✗),
            #         但 Stage2 的 **CARRY 是水平**移动 ⇒ **一启动就脱** ✗ (drift 177mm ✗)
            #   为什么用"门"而非"权重/reward" ✓: 本轮已**四次**验证
            #     "改权重/reward ⇒ 换盆地 ⇒ 丢好解" ✗ (md §46.4) ⇒ 门只改**候选集** ✓
            _g5_gate_mm = float(os.environ.get("_S1_GATE_GND_MIN_MM", "0.0") or 0.0)
            if (_g5_gate_mm > 0.0 and hard_ok and _gc_s is not None
                    and float(_gc_s) < _g5_gate_mm):
                hard_ok = False
                details['coacd_screen'] = 'REJECT(离桌面%.1f<%.1fmm ⇒ 会被桌面挡停)' % (
                    float(_gc_s), _g5_gate_mm)
            _w_center = float(os.environ.get("_S1_W_CENTER", "0.40") or 0.40)
            _w_clear = float(os.environ.get("_S1_W_CLEAR", "0.20") or 0.20)
            _w_axial = float(os.environ.get("_S1_W_AXIAL", "0.15") or 0.15)
            _w_reach = float(os.environ.get("_S1_W_REACH", "0.15") or 0.15)
            _w_gnd = float(os.environ.get("_S1_W_GND", "0.10") or 0.10)
            _w_sum = _w_center + _w_clear + _w_axial + _w_reach + _w_gnd
            if _w_sum <= 1e-9:
                _w_center, _w_clear, _w_axial, _w_reach, _w_gnd = 0.40, 0.20, 0.15, 0.15, 0.10
                _w_sum = 1.0
            score = (_w_center * _k_center + _w_clear * _k_clear + _w_axial * _k_axial
                     + _w_reach * _k_reach + _w_gnd * _k_gnd) / _w_sum
            details['s1_k_center'] = round(_k_center, 4)
            details['s1_k_clear'] = round(_k_clear, 4)
            details['s1_k_axial'] = round(_k_axial, 4)
            details['s1_k_gnd'] = round(_k_gnd, 4)
            details['score_source'] = 'coacd'
            details['s1_k_reach'] = round(_k_reach, 4)
            details['s1_w'] = dict(center=_w_center, clear=_w_clear, axial=_w_axial,
                                   reach=_w_reach, gnd=_w_gnd)
            details['coacd_score'] = round(float(score), 4)

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
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-24 【新分工第一步】候选窗口**截止到指定帧** ✓ (用户: "到 F60 就停下来") ★★★★
        #   用户原话 ✓: "先(31 帧)到 **F60** 就停下来吧, **26 帧**(F35–F60 ✓), 然后这 26 帧
        #     直接跑评估打分优化, 选出最好的三帧, 再进行优化" ✓
        #   ⇒ `_S1_FRAME_HI=60` (默认 0=不裁剪 ✓ 旧行为不变 ✗) ⇒ 窗口变为 F35–F60 = 26 帧 ✓
        #   ⚠ 只收窄**候选来源** ✓, 不改任何门/奖励 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _hi_cap = int(os.environ.get("_S1_FRAME_HI", "0") or 0)
        if _hi_cap > 0:
            hi = min(hi, _hi_cap)
            logger.info("[候选窗口] ✓ 已按 `_S1_FRAME_HI=%d` 收窄: 帧 **%d–%d (%d 帧)** ✓",
                        _hi_cap, lo, hi, hi - lo + 1)
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
        # ★ 2026-09-21 【z 微调可调】默认与历史逐值一致 (±5mm, 3 档 ✓)；
        #   需要"夹物体上半部以避开桌面"时把它加大 ✓ (如 `_S1_NZ=5 _S1_DZ_MM=5` ⇒ ±10mm)
        #   依据 ✓: 手指(≈37mm) ≫ 物体(≈20mm) ⇒ 只靠默认 ±5mm 可能不够抬高 ✓
        #     (`output/s2_gate`: Top-K 候选真值 `指尖离桌面` 全 <2mm ✗)
        _n_z_eff = int(os.environ.get("_S1_NZ", str(n_z)) or n_z)
        _dz_step_m = float(os.environ.get("_S1_DZ_MM", "5.0") or 5.0) / 1000.0
        d_z_list = [(i - (_n_z_eff - 1) / 2.0) * _dz_step_m for i in range(_n_z_eff)]

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

        # ★★★★ 2026-09-21 【对齐口径: 手指原点 vs 虎口中心】(见上文说明 ✓) ★★★★
        #   `finger_origin` (默认 ✓ = 历史行为): base→**手指原点** 对齐物体中心 ✗
        #   `finger_mid`   (推荐 ✓): base→**虎口中心** 对齐物体中心 ✓ (= MANO `finger_mid` ✓)
        #     · 虎口中心 = 手指盒中心在**两指之间**的中点 ⇒ y 分量取 0 (两指对称 ✓)
        #     · 偏移 = `_FINGER_OFFSET_GRIPPER + [LC_x, 0, LC_z]` ✓
        #       = [0.03689-0.0045, 0, 0+0.0043] = [0.03239, 0, 0.0043] ✓
        _align_mode = str(os.environ.get("_S1_ALIGN_POINT", "finger_origin") or "finger_origin")
        if _align_mode == "finger_mid":
            from stage2_approach_opt import _FINGER_LOCAL_CENTER as _FLC_AL
            _align_off = _FINGER_OFFSET_GRIPPER + np.array(
                [float(_FLC_AL[0]), 0.0, float(_FLC_AL[2])], dtype=np.float64)
            logger.info(
                "[Stage1][对齐口径] **虎口中心对齐物体中心** ✓ (与 MANO `finger_mid` 同口径 ✓): "
                "offsets 手指原点=%s → 虎口中心=%s (差 %s mm) ⇒ 消除『手指前端下探碰桌面』✗",
                np.round(_FINGER_OFFSET_GRIPPER, 5), np.round(_align_off, 5),
                np.round((_align_off - _FINGER_OFFSET_GRIPPER) * 1000.0, 2))
        else:
            _align_off = _FINGER_OFFSET_GRIPPER

        candidates = []
        for f in range(lo, hi + 1):
            _R0 = np.asarray(R_traj[f], dtype=np.float64)
            if _tilt_R is not None:
                _R0 = _tilt_R @ _R0
            for _Rd in _rot_list:
                R = _R0 @ _Rd                      # ★ 扰动施加在**夹爪自身坐标系** ✓

                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-25 【摆正】绕**接近轴**滚转 ⇒ 开合轴水平 ⇒ **两指尖等高** ✓ ★★★★
                #   实测铁证 ✓ (cc22 `[Stage1 真值摆正]`): 所有候选两指**高差 48~54mm**,
                #     **几乎不随候选变化** ⇒ 它**不是搜索变量**, 而是**初始姿态自带** ✗
                #     ⇒ 后果: 上指在物体顶面之上(够不到侧壁 ✗) + 下指插桌被卡(张不开 ✗)
                #       (cc19/21 `[张开受阻检测]`: finger1 残差=目标值, 接触=外部 ⇒ OPEN_BLOCKED ✗)
                #       ⇒ 这就是"双侧接触 180 次, 但**抬起永远 0**"的几何根因 ✓
                #   几何 ✓: 接近轴 = 局部 x (`_FINGER_OFFSET_GRIPPER=[0.03689,0,0]` ✓),
                #     开合轴 = 局部 y (关节轴 (0,-1,0) ✓)
                #     ⇒ 令世界系**开合轴 z 分量 = 0** ⇒ 两指并排、等高 ⇒ 都能压在侧壁上 ✓
                #   求解 ✓ (Rodrigues 绕 a=R[:,0]): u'=u·cosφ+(a×u)·sinφ, 要求 u'_z=0
                #     ⇒ tanφ = −u_z / n_z  (n = R[:,2] = a×u ✓)
                #     ⇒ 两个解相差 π, 取 **|φ| 更小** 的 (最贴近原意图 ✓)
                #   开关 ✓: `_S1_LEVEL_FINGERS=1` 默认启用 (0 = 回到旧行为 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                if str(os.environ.get("_S1_LEVEL_FINGERS", "1")) == "1":
                    try:
                        _a_lv = np.asarray(R[:, 0], dtype=np.float64)   # 接近轴 (世界)
                        _u_lv = np.asarray(R[:, 1], dtype=np.float64)   # 开合轴 (世界)
                        _n_lv = np.asarray(R[:, 2], dtype=np.float64)   # = a × u ✓
                        _phi = float(np.arctan2(-float(_u_lv[2]), float(_n_lv[2])))
                        if abs(_phi) > np.pi / 2.0:    # 取 |φ| 更小的等价解 (差 π ✓)
                            _phi -= np.pi if _phi > 0 else -np.pi
                        _c_lv, _s_lv = float(np.cos(_phi)), float(np.sin(_phi))
                        # 绕世界轴 a 的旋转矩阵 (Rodrigues ✓)
                        _ax, _ay, _az = _a_lv
                        _K_lv = np.array([[0.0, -_az, _ay], [_az, 0.0, -_ax], [-_ay, _ax, 0.0]])
                        _Rlv = (np.eye(3) + _s_lv * _K_lv
                                + (1.0 - _c_lv) * (_K_lv @ _K_lv))
                        R = _Rlv @ R
                        # 校验 (读回 ✓): 摆正后开合轴 z 分量应 ≈0
                        if abs(float(R[2, 1])) > 1e-6:
                            logger.warning("[Stage1][摆正] ✗ 读回不一致: 开合轴 z=%.6f",
                                           float(R[2, 1]))
                    except Exception as _elv:  # noqa: BLE001
                        logger.warning("[Stage1][摆正] 失败(保持原姿态 ✓): %s", _elv)

                base_pos = obj_pos - R @ _align_off      # ★ 对齐口径可切 (见上文 ✓)
                for d_close in d_close_list:
                    for d_fwd in d_fwd_list:
                        for dz in d_z_list:
                            pos = base_pos + R @ (_fwd_l * d_fwd + _close_l * d_close)
                            # ★★★★ 2026-09-21 【对齐口径修正】(用户: "夹持要和这个 mano 对应") ★★★★
                            #   用户 (2026-09-21): "**不要夹持高度**, 夹持要和这个 **mano 对应**"
                            #     + "**桌面怎么会挡停呢? 说明夹爪夹取的时候碰到了桌面**, 这个需要注意"
                            #   几何铁证 ✓ (纯解析算出, `output/s2_gate` 实测 指尖离桌面 0.0mm ✓):
                            #     · MANO 预处理的对齐点是 **`finger_mid`** (两指指尖中点 ✓,
                            #       日志: `[pre_translate] F50 finger_mid=... → obj_center=...` ✓)
                            #     · 而候选生成用的 `_FINGER_OFFSET_GRIPPER=[0.03689,0,0]` 是
                            #       **手指原点** ⇒ 指1 原点 z=21.40mm / 指2 原点 z=38.04mm
                            #       (两者差 16.6mm ⇒ **它根本不是"夹持点"** ✗)
                            #       ⇒ 而手指沿接近轴**前伸 92mm** (半长 0.046 ✓),
                            #         前端下角算得 z=**1.37mm** ⇒ **闭合/移动中必然下探碰桌面** ✗✗
                            #   修法 ✓: 用**虎口中心** (两指内侧面中点 ✓ = MANO `finger_mid` 的
                            #     机器人对应点 ✓) 对齐物体中心 ✓ —— 这不是新自由度 ✗,
                            #     而是**把对齐口径从"手指原点"改成"夹持点"** ✓ (与 MANO 一致 ✓)
                            #   开关 ✓: `_S1_ALIGN_POINT=finger_mid` 启用 (默认 = 历史行为 ✓)
                            pos[2] += dz
                            if pos[2] < _g0 + 0.01:
                                continue
                            # ★ T07e-2 ✓ 记下**位姿→源帧号**映射 (下游调用处会重建 pose dict,
                            #   `source_frame` 会丢 ✗ ⇒ 用映射按 pos/R 取回 ✓ 不编造 ✓)
                            try:
                                _kb7 = (tuple(np.round(np.asarray(pos, dtype=np.float64), 6)),
                                        tuple(np.round(np.asarray(R, dtype=np.float64),
                                                       6).reshape(-1)))
                                _mb7 = getattr(self, "_s1_pose_src_frame", None)
                                if _mb7 is None:
                                    _mb7 = {}
                                    self._s1_pose_src_frame = _mb7
                                _mb7[_kb7] = int(f)
                                # ★ 同族帧号 (CEM 扰动位姿会 miss 精确键 ⇒ 用本扫描族的帧号 ✓
                                #   仍是**真实 MANO 帧号**, 不是编造 ✗)
                                self._s1_last_cand_frame = int(f)
                            except Exception:  # noqa: BLE001
                                pass
                            candidates.append({
                                'pos': pos,
                                'R': R.copy(),
                                'euler': np.array(rotmat_to_zyx_euler(R)),
                                'source_frame': float(f),
                                # ★★★★ 2026-09-18 每候选带上**源帧参考姿态** ✓
                                #   用户原话 (2026-09-14): "F25→F50 ±5° 是**每个候选都能调整
                                #     ±5 度**, 不是只有 F50 是 ±5 度" ✓
                                #   ⇒ 硬约束的参考应当是**该候选自己的源帧** ✓ (而不是恒 F50 ✗)
                                #     否则 F35~F65 里离 F50 超过 5° 的帧**全部被判违规** ✗
                                #     ⇒ 31 帧的"位姿选择权"塌缩成 F50 附近 1 个位姿 ✗
                                #       (实测 output/s1horiz: 7533 候选 → 仅 6 个进入 Stage1b ✗)
                                #   开关: `_STAGE1_MANO_REF=frame` 启用 (默认 f50 = 历史行为 ✓)
                                'ref_R': _R0.copy(),
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
    def _measure_grasp_symmetry(self, target_name, robot=None):
        """★★★ 2026-09-20 【真值对中】两指各自到目标凸包的**真实最近距离** (mm) ★★★

        用户要求 (2026-09-20): "**抓住要抓紧一点**" + "可以验证一下" ✓
        为什么必须有这个量 ✓ (实测真伪对照 ✓):
          · `c_center`(CoACD 对中) 报 ≤2.5mm 时, **真实**两指到物体距离差仍有 **10mm** ✗
            (实测 output/cent1: 到指1=3.49mm, 到指2=13.57mm ✓)
            ⇒ `c_center` 描述的是"物体相对闭合轴心" ✗, 不是"两指各自能不能压到物体" ✗
          · 后果 ✗: 一指贴上、另一指悬空 10mm ⇒ **没有双侧力闭合** ⇒ 物体被压住/滑落 ✗
        返回: (asym_mm, dmin_mm)
          · `asym_mm` = |d1 − d2| ⇒ **越小越居中** ✓ (真值对中 ✓)
          · `dmin_mm` = min(d1, d2) ⇒ 两侧都贴住时应 ≈0 ✓ (接触质量 ✓)
        取不到 (未启用 CoACD / 无手指) ⇒ 返回 (None, None) ✓ (老路径零影响 ✓)
        """
        # ── 距离实现已抽到 `gap_utils.convex_gap_mm` ✓ (方案 §10.2: 单一实现可单测 ✓) ──
        from gap_utils import convex_gap_min_mm as _gap_min_mm   # noqa: F401

        def _qp_gap_mm(_A, _B):
            """两个凸多面体(顶点给出)的**精确最小距离**(mm) + 是否相交 ✓

            为什么不再用我写的 GJK ✓: 复审反例(**真实间隙 40mm**)实测报 **45.2769mm** ✗
              = √(40²+15²+15²) ⇒ 我那份实现把**角点距**当成了最小距 ✗ (单纯形最近点更新有 bug ✗)
            本实现 ✓: 用 `ConvexHull` 取半空间 `n·x + d ≤ 0`, 解小规模 QP:
                 min |a − b|²   s.t. a∈A, b∈B
              ⇒ 对任意朝向/任意凸分片都成立 ✓; 距离 ≈0 即接触/相交 ✓
            """
            from scipy.spatial import ConvexHull
            from scipy.optimize import minimize
            _A2 = np.asarray(_A, dtype=np.float64)
            _B2 = np.asarray(_B, dtype=np.float64)
            if _A2.ndim != 2 or _B2.ndim != 2 or len(_A2) < 4 or len(_B2) < 4:
                raise ValueError("点太少, 无法构凸包")
            _eqA = ConvexHull(_A2).equations           # (n,4): n·x + d ≤ 0 ✓
            _eqB = ConvexHull(_B2).equations

            def _cA(x):
                return -( _eqA[:, :3] @ x[0:3] + _eqA[:, 3] )

            def _cB(x):
                return -( _eqB[:, :3] @ x[3:6] + _eqB[:, 3] )

            def _obj(x):
                _dd = x[0:3] - x[3:6]
                return float(_dd @ _dd)

            _x0 = np.concatenate([_A2.mean(axis=0), _B2.mean(axis=0)])
            _res = minimize(_obj, _x0, method="SLSQP",
                            constraints=[{"type": "ineq", "fun": _cA},
                                         {"type": "ineq", "fun": _cB}],
                            options={"maxiter": 300, "ftol": 1e-12})
            _gap_m = float(np.sqrt(max(_obj(_res.x), 0.0)))
            return (_gap_m * 1000.0, bool(_gap_m < 1e-9))

        # ── GJK: 两个**凸体**的真实表面间隙 (复审 §11.2 反例 A ✓) ──
        def _gjk_gap_mm(_A, _B, _iters=64, _eps=1e-12):
            """返回 (gap_mm, hit)。hit=True ⇒ **相交/穿透** (gap 记 0, 另设诊断标志 ✓)。

            凸体用顶点云的**支撑函数**表示: `sup(P,d) = argmax(P·d)` ✓
            ⇒ 最近点在**面内**也能算出来 (旧顶点↔顶点做不到 ✗ ⇒ 反例返回 21.21mm ✗)
            """
            _A = np.asarray(_A, dtype=np.float64)
            _B = np.asarray(_B, dtype=np.float64)
            if _A.size == 0 or _B.size == 0 or _A.ndim != 2 or _B.ndim != 2:
                return None, False
            # ★ W5 ✓: 优先用**精确 QP 解** (我写的 GJK 在复审反例上错了 ✗ ⇒ 已降级为兜底 ✓)
            try:
                return _qp_gap_mm(_A, _B)
            except Exception:  # noqa: BLE001
                pass

            def _sup(_P, _d):
                return _P[int(np.argmax(_P @ _d))]

            def _closest(_W):
                """单纯形 (1/2/3 点) 上离原点最近点 ⇒ 返回 (p, 保留子集) ✓"""
                if len(_W) == 1:
                    return _W[0], list(_W)
                if len(_W) == 2:
                    _a, _b = _W[0], _W[1]
                    _ab = _b - _a
                    _t = float(np.dot(-_a, _ab) / max(float(np.dot(_ab, _ab)), _eps))
                    if _t <= 0.0:
                        return _a, [_a]
                    if _t >= 1.0:
                        return _b, [_b]
                    return _a + _t * _ab, list(_W)
                # 3 点: 先试三角形内部, 否则退化为三条边的最优
                _a, _b, _c = _W[0], _W[1], _W[2]
                _ab, _ac = _b - _a, _c - _a
                _n = np.cross(_ab, _ac)
                _nn = float(np.dot(_n, _n))
                if _nn > _eps:
                    _nrm = float(np.dot(_a, _n)) / _nn
                    _p = _n * _nrm                       # 原点到平面垂足
                    # 重心坐标符号判定是否在三角形内 ✓
                    _v0 = np.cross(_b - _p, _c - _p)
                    _v1 = np.cross(_c - _p, _a - _p)
                    _v2 = np.cross(_a - _p, _b - _p)
                    if (np.dot(_v0, _n) >= -_eps and np.dot(_v1, _n) >= -_eps
                            and np.dot(_v2, _n) >= -_eps):
                        return _p, list(_W)
                _best, _bestsub, _bn = None, list(_W), None
                for _i2, _j2 in ((0, 1), (0, 2), (1, 2)):
                    _pp, _ss = _closest([_W[_i2], _W[_j2]])
                    _nn2 = float(np.dot(_pp, _pp))
                    if _bn is None or _nn2 < _bn:
                        _best, _bestsub, _bn = _pp, _ss, _nn2
                return _best, _bestsub

            _d = _B.mean(0) - _A.mean(0)
            if (not np.isfinite(_d).all()) or np.linalg.norm(_d) < _eps:
                _d = np.array([1.0, 0.0, 0.0])
            _d = _d / max(float(np.linalg.norm(_d)), _eps)
            _sim = [_sup(_A, _d) - _sup(_B, -_d)]
            _d = -_sim[0]
            if np.linalg.norm(_d) < _eps:
                return 0.0, True                      # 原点在差集内 ⇒ 相交 ✓
            _d = _d / max(float(np.linalg.norm(_d)), _eps)
            for _ in range(_iters):
                _v = _sup(_A, _d) - _sup(_B, -_d)
                if float(np.dot(_v, _d)) < 0.0:       # 无法越过原点 ⇒ 分离 ✓
                    _p, _ = _closest(_sim)
                    return float(np.linalg.norm(_p)) * 1000.0, False
                _sim = _sim + [_v]
                _p, _sub = _closest(_sim)
                if float(np.linalg.norm(_p)) < _eps:
                    return 0.0, True                  # 相交 ✓
                _sim = list(_sub)
                _d = -_p
                if np.linalg.norm(_d) < _eps:
                    return 0.0, True
                _d = _d / max(float(np.linalg.norm(_d)), _eps)
            _p, _ = _closest(_sim)
            return float(np.linalg.norm(_p)) * 1000.0, False

        try:
            _hulls_m = self._coacd_world_hulls(target_name)
            if not _hulls_m:
                return None, None
            _HV = np.concatenate([np.asarray(h, dtype=np.float64) for h in _hulls_m], axis=0)
            _vv_m = self._finger_world_vertices(robot)
            if len(_vv_m) != 2:
                return None, None
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-24 【用户要求】几何**只导出一次** ✓ 之后离线扫描 ✓ ★★★★
            #   用户原话 ✓: "**你不要实时计算凸包, 就一次仿真保存一下然后再运行**" ✓
            #   ⇒ `_S1_DUMP_GEOM=1` (默认 0=不改行为 ✗) 时, 在**本次运行只写一次** ✓:
            #     目标 CoACD 凸包(世界系 ✓) + 两指碰撞顶点(世界系 ✓) + 当前夹爪位姿 ✓
            #     ⇒ `output/<run>/geom_dump.npz` ✓ ⇒ `tools/access_scan.py` 直接读它 ✓
            #     ⇒ 扫描在**秒级**完成 (不再依赖仿真 ✓ 也不再实时算凸包 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ 2026-09-24 修正 ✓: geo1 实测导出发生在**基座未到位**时 ⇒ 间距 2400mm ✗
            #   ⇒ 必须**靠近后才导** ✓ (判据: 最近指-物间隙 < 0.2m ✓) + 同时导出**姿态 R** ✓
            _near_for_dump = False
            try:
                _g0, _ = _gjk_gap_mm(_HV, _vv_m[0])
                _g1, _ = _gjk_gap_mm(_HV, _vv_m[1])
                _near_for_dump = bool((_g0 is not None and _g0 < 200.0)
                                      or (_g1 is not None and _g1 < 200.0))
            except Exception:  # noqa: BLE001
                _near_for_dump = False
            if (os.environ.get("_S1_DUMP_GEOM", "0") == "1" and _near_for_dump
                    and not getattr(self, "_geom_dumped", False)):
                try:
                    _od = getattr(self, "output_dir", None) or os.environ.get("_OUTPUT_DIR") or "."
                    _dst = os.path.join(str(_od), "geom_dump.npz")
                    os.makedirs(os.path.dirname(_dst), exist_ok=True)
                    _gp = None
                    _gq_q = None
                    try:
                        _pose_root = self.robot.get_root_pose()
                        _gp = np.asarray(_pose_root.p, dtype=np.float64)
                        _gq_q = np.asarray(_pose_root.q, dtype=np.float64)  # ★ 姿态 ✓ (扫描要用 ✓)
                    except Exception:  # noqa: BLE001
                        _gp = None
                    np.savez(
                        _dst,
                        target_hulls_world=_HV,
                        n_hulls=len(_hulls_m),
                        finger1_world=np.asarray(_vv_m[0], dtype=np.float64),
                        finger2_world=np.asarray(_vv_m[1], dtype=np.float64),
                        gripper_pos=(_gp if _gp is not None else np.zeros(3)),
                        gripper_quat=(_gq_q if _gq_q is not None else np.array([1.0, 0.0, 0.0, 0.0])),
                        gripper_qpos=float(getattr(self, "_last_grip_target_s1", 0.0) or 0.0),
                    )
                    self._geom_dumped = True
                    logger.info("[几何导出] ✓ 已写一次: %s (凸包 %d 片 / 指1 %d 点 / 指2 %d 点) "
                                "⇒ 之后用 tools/access_scan.py **离线秒级**扫描 ✓",
                                _dst, len(_hulls_m), len(_vv_m[0]), len(_vv_m[1]))
                except Exception as _eg:  # noqa: BLE001
                    logger.warning("[几何导出] 失败 (不阻断): %s", _eg)
            _d = []
            _hit_any = False
            for _V in _vv_m:
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-23 【方案 §11.2 反例 A 修正】必须用**真实表面间隙** ✗ ★★★★
                #   复审原话 ✓ + 他们给出的合成反例 ✓:
                #     目标 x∈[-10,10]; 左指 x∈[-20,-10]; 右指 x∈[10,20] (y,z 覆盖)
                #     ⇒ 两指内侧面与目标两面**已接触**, 真实间隙 **0** ✓
                #     旧实现 (顶点↔**抽样**顶点) 返回 **21.2132mm** ✗✗
                #     ⇒ 我日志里的\"8~20mm\"**不能证明够不到物体** ✗ (我此前据此下的结论作废 ✗)
                #   本改 ✓: 用 **GJK** 求两个凸体 (CoACD 凸包 / 指碰撞几何) 的**最小距离** ✓
                #     · `hit=True` ⇒ **相交/穿透** ⇒ 进入穿透诊断, 不能裁成 0 后当满分 ✓
                #     · 距离来自**支撑函数** (沿方向最远顶点) ⇒ 面内最近点也算得出来 ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ W5 (2026-09-26) 【逐分片求最小】**不得**合并 CoACD 分片 ✗ ★★★★
                #   复审反例 ✓: 手指离目标实际 **15mm**, 却返回 **0 并判相交** ✗✗
                #   病根 ✓: `_HV = np.concatenate(所有分片)` ⇒ 把分片**点云**当单个凸体求 GJK
                #     ⇒ 等价于取"所有分片的**凸包**" ⇒ **填实凹槽** ✗
                #     ⇒ 距离被系统性低估, 甚至把"跨过凹槽"误判成**相交** ✗
                #   本改 ✓: 对**每个分片**分别求 GJK, 取**最小值**; 任一分片相交则记 hit ✓
                #     ⇒ 凹形物体的真实表面间隙得以保留 ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _best_gap = None
                for _h_m in _hulls_m:
                    _g_m, _hit_m = _gjk_gap_mm(_h_m, _V)
                    if _g_m is None:
                        continue
                    if _best_gap is None or float(_g_m) < _best_gap:
                        _best_gap = float(_g_m)
                    if bool(_hit_m):
                        _hit_any = True
                if _best_gap is None:
                    return None, None
                _d.append(float(_best_gap))
            # 相交时 gap 记 0 ✓ 但**另有诊断标志** ✓ (不能把负间隙裁成零后拿满分 ✗)
            if _hit_any:
                self._last_rollout_surface_intersect = True
            return abs(_d[0] - _d[1]), min(_d[0], _d[1])
        except Exception:  # noqa: BLE001
            return None, None

    def _finger_world_vertices(self, robot=None):
        """两指**碰撞几何**顶点在世界系的数组列表 (2 个元素) ✓

        抽出原因 (2026-09-20): `_measure_grasp_symmetry` (真值对中) 与
        `_measure_finger_ground_clearance` (指尖离桌面) 必须用**同一份**顶点 ⇒ 单一来源 ✓
        取不到 (无机器人 / 不是 2 个手指) ⇒ 返回 [] ✓ (老路径零影响 ✓)
        """
        try:
            import sapien as _sp_v
            from sapien.pysapien.physx import (
                PhysxArticulationLinkComponent as _ALC_v)
            _robot = robot if robot is not None else getattr(self, 'robot', None)
            if _robot is None:
                return []
            _side = getattr(self, '_active_side_s1', None) or self.side
            _out = []
            for _l in _robot.get_links():
                if _l.get_name() not in (f"{_side}_gripper_finger_link1",
                                         f"{_side}_gripper_finger_link2"):
                    continue
                _Tl = _l.get_pose().to_transformation_matrix()
                _acc = []
                for _comp in _l.entity.components:
                    if not isinstance(_comp, _ALC_v):
                        continue
                    for _sh in _comp.get_collision_shapes():
                        try:
                            _v = np.asarray(_sh.get_vertices(), dtype=np.float64)
                        except Exception:  # noqa: BLE001
                            continue
                        _Ts = _Tl @ _sh.get_local_pose().to_transformation_matrix()
                        _acc.append(_v @ _Ts[:3, :3].T + _Ts[:3, 3])
                if _acc:
                    _out.append(np.concatenate(_acc, axis=0))
            return _out if len(_out) == 2 else []
        except Exception:  # noqa: BLE001
            return []

    def _measure_finger_ground_clearance(self, robot=None):
        """两指碰撞几何**最低点** − 桌面高度 (mm ✓; <0 = 指尖插进桌面 ✗)

        ★★★★ 2026-09-20 【P0 根因】为什么必须有这个量 ★★★★
          手指全长 77.4mm (`stage2_approach_opt._FINGER_BOX_HALF[0]=0.0387`),
          而 glb_4 只有 39.3mm 高 ⇒ 在物体中部闭合时**指尖必然低于桌面** ✓
          在 `_NO_GRIPPER_GROUND=0` (桌面碰撞**开启**, 用户硬性要求 ✓) 下:
            指尖被桌面挡停 ⇒ 关节停在 q≈0.008~0.012 (目标 0.0021) ✗
            ⇒ 指缝 46~54mm ≫ 物体跨度 38.8mm ⇒ **永远夹不住** ✗
          实测铁证 (output/h2_hzaxis_slow_0920, 1839 个 rollout ✓):
            · debug 里 `('left_gripper_finger_link1','ground')` 与 `force=0.00N` **同时**出现 ✓
            · `obj_lift` **1839/1839 全为 0.00cm** ✗ (只有 8 次峰值 >5cm, 且都中途掉落 ✗)
            · `f1_qpos` 停在 0.0081~0.0105 而目标 0.0021 ⇒ 被挡停, 不是被物体挡 ✓
          修法 ✓: 把本量喂进 reward ⇒ CEM 才有"把基座抬高"的梯度 ✓
            (旧代码只在 `score_grasp_quality` 的**候选排序**里用它, 且权重最低 ✗
             ⇒ CEM 的 reward **完全看不到**它 ⇒ 优化器无动机抬升 ✗)
        返回 None 表示取不到 ✓ (老路径零影响 ✓)
        """
        _vv = self._finger_world_vertices(robot)
        if not _vv:
            return None
        return (float(min(np.min(_V[:, 2]) for _V in _vv))
                - float(getattr(self, '_ground_z', 0.0))) * 1000.0

    def _measure_finger_tilt_mm(self, robot=None):
        """两指**高差**(mm ✓) —— "摆正"代价 (2026-09-25 ✓ 用户定调: 位姿要摆正)

        ★★★★ 实测铁证 (cc19/cc21 `[张开受阻检测]`) ★★★★
          `f1_pos.z=0.0237` / `f2_pos.z=0.0559` ⇒ **两指高差 32mm** ✗
          而 glb_4 仅 ~39mm 高且**贴桌** ⇒
            · 上指已在物体顶面**之上** ⇒ 永远够不到侧壁 ✗
            · 下指被压在物体底部与桌面之间 ⇒ `OPEN_BLOCKED`(张不开 ✗)
          ⇒ 只调张开量/力限都无效 ✓ (实测: 力限 100N 仍不动 ✗) ⇒ **必须摆正 + 落在侧壁内**
        返回 None = 取不到 ✓ (老路径零影响 ✓)
        """
        _vv = self._finger_world_vertices(robot)
        if not _vv or len(_vv) < 2:
            return None
        try:
            _z0 = float(np.mean(np.asarray(_vv[0])[:, 2]))
            _z1 = float(np.mean(np.asarray(_vv[1])[:, 2]))
            return abs(_z0 - _z1) * 1000.0
        except Exception:  # noqa: BLE001
            return None

    def _coacd_world_hulls(self, target_name):
        """目标物体在**世界系**的 CoACD 凸包顶点列表 (直接读引擎实际使用的 shapes ✓).

        返回 [] 表示未启用 CoACD / 取不到 ⇒ 调用方自动跳过 ✓ (老路径零影响 ✓)
        """
        import sapien as _sp

        if not target_name:
            return []
        for _a in getattr(self, "obj_actors", []) or []:
            _n = _a.get_name()
            if _n == target_name or target_name in _n:
                _H = np.asarray(_a.get_pose().to_transformation_matrix(), dtype=np.float64)
                _out = []
                for _c in _a.components:
                    if not isinstance(
                            _c, _sp.pysapien.physx.PhysxRigidDynamicComponent):
                        continue
                    for _sh in _c.get_collision_shapes():
                        try:
                            _v = np.asarray(_sh.get_vertices(), dtype=np.float64)
                        except Exception:  # noqa: BLE001
                            continue
                        if _v.ndim == 2 and _v.shape[1] == 3 and len(_v):
                            _out.append(_v @ _H[:3, :3].T + _H[:3, 3])
                return _out
        return []

    def _coacd_key_fields(self, grasp_pose):
        """按**这一个候选**算 CoACD 关键判据, 供 Stage1 reward / Stage2 判据共用 ✓

        与上一版失败做法 (自 self 缓存 ✗, output/qual6: reward 爆 −24722 ✗ / 验证门 0/3 ✗)
        的区别: 这里的输入是**本候选的** grasp_pose ✓ ⇒ 判据必然配对正确 ✓
        来源: 共享单一来源 `stage1_recipe.coacd_key_criteria` ✓
        未启用 CoACD / 取不到凸包 ⇒ 返回 {} ⇒ reward 侧自动跳过 ✓
        """
        try:
            _ctgt = str(os.environ.get("_COACD_TARGET", "") or "")
            if (not _ctgt) or _ctgt.lower() == "none":
                return {}
            _tn = (grasp_pose.get('target') if isinstance(grasp_pose, dict) else None) \
                or getattr(self, 'target_name', None)
            if not _tn:
                _cands = [k for k in getattr(self, 'obj_info', {}) or {} if _ctgt in k]
                _tn = _cands[0] if _cands else None
            if not _tn:
                return {}
            _hulls = self._coacd_world_hulls(_tn)
            if not _hulls:
                return {}
            _bsize = (getattr(self, 'obj_info', {}) or {}).get(_tn, {}).get('bbox_size')
            if _bsize is None:
                return {}
            # ★★★★ 2026-09-18 【P0 判据修正】"不要碰物体" 的参考间距必须用**接近段(全开)** ★★★★
            #   语义 ✓: 这条 CoACD 判据问的是 **接近段(手指全开)会不会顶到物体** ✓
            #     (Stage2 对应门禁 = `APPROACH_TARGET_CONTACT` ✓, 用户定调 "不要碰物体" ✓)
            #     ⇒ 参考间距必须是 **接近段开度 `stage_q_open`** ✓
            #   旧值 ✗: `compute_gripper_qpos(_bsize)` = **闭合目标开度** (q_grasp≈0.0047 ✓)
            #     ⇒ 半间距只有 0.0152+0.0047 = 19.9mm, 而物体沿闭合轴半宽 27.5mm ✓
            #     ⇒ c_clear = 27.5 − 19.9 = **+7.6mm > 0 恒定** ✗
            #     ⇒ 7533 个候选**全部**被判"手指必然顶到物体" ✗ ⇒ 判据零区分度 ✗
            #       ⇒ reward 侧的 −80 罚项对**每个**候选都生效 ✗ (退化成常量 ✗)
            #   修正后 ✓: 半间距 = 0.0152 + q_open(≈0.05) = 65.2mm
            #     ⇒ c_clear = 27.5 − 65.2 = **−37.7mm** ⇒ 只在"真张不开"时才 >0 ✓
            #   ★ 与 `score_grasp_quality.half_sep_approach` **同源同款** ✓ (单一来源 ✓)
            #   开关 ✓: `_S1_CLEAR_REF=close` 回到旧行为 (仅作对照 ✓)
            if os.environ.get("_S1_CLEAR_REF", "open") == "close":
                _sep = float(FINGER_EFFECTIVE_HALF_SPACING) + float(
                    self.compute_gripper_qpos(_bsize))
            else:
                try:
                    _w_open = float(max(_bsize))
                except Exception:  # noqa: BLE001
                    _w_open = None
                _sep = float(FINGER_EFFECTIVE_HALF_SPACING) + float(stage_q_open(_w_open))
            from stage1_recipe import coacd_key_criteria as _ckc
            _ctr = np.asarray(
                getattr(self, 'obj_bbox_centers', {}).get(_tn, [0.0, 0.0, 0.0]),
                dtype=np.float64)
            _cr = _ckc(np.asarray(grasp_pose['R'], dtype=np.float64),
                       np.asarray(grasp_pose['pos'], dtype=np.float64),
                       _ctr, _hulls, _sep)
            return {
                'coacd_c_center_mm': float(_cr['c_center_mm']),
                'coacd_c_clear_mm': float(_cr['c_clear_mm']),
                'coacd_c_axial_mm': float(_cr['c_axial_mm']),
                'coacd_kappa': float(_cr['kappa']),
            }
        except Exception as _e:  # noqa: BLE001
            logger.warning("[CoACD判据] 计算失败(跳过): %s", _e)
            return {}

    def _assess_grasp(self, result, criteria=None):
        """★ 2026-09-22 【复审 I.2 建议】**统一的成功评估函数** ✓

        复审原话 ✓: "建议用一个共享的 `assess_grasp(result, criteria)` 返回
          **数据有效性、物理可行性、任务成功、失败原因**。候选筛选、重复确认、Stage1d、
          最终验证**都调用它**; 无成功候选时允许按连续失败代价继续搜索, 但**禁止交付为成功** ✓
          不要各处再复制一套 `_SEL_*`、`_ACC_*` 判断" ✓

        为什么必须统一 ✓ (复审 I.2 末段 ✓): "Top-K 新加了一套支撑/接触门, CEM 的重复成功数
          仍主要按峰值 lift、漂移和旧可选门计算 … **搜索在追逐旧目标, 最后才被新门淘汰**" ✗
          ⇒ 会造成**多轮 CEM 都无效** ✗

        返回 dict ✓:
          `data_valid`     : 必备字段是否齐全 (缺测 ⇒ **不可信** ✓ 复审 I.2-4 ✓)
          `phys_feasible`  : 物理可行性 (未被桌面挡停 / 未明显穿透 / 目标不再被桌面支撑 ✓)
          `task_success`   : 任务成功 (lift / 净空 / 真夹持率 / 漂移 **全部**达标 ✓)
          `reasons`        : 失败原因列表 ✓
          `missing`        : 缺失字段列表 ✓
        语义 ✓ (复审 I.2 末段 ✓): "挡停率仍只叫挡停率; 目标两指接触、法向与接触位置另算" ✓
          ⇒ 本函数**只**消费已命名的物理量, 不再自造"真夹持率"这类含混指标 ✓
        """
        _cr = dict(criteria or {})
        _lift_min = float(_cr.get("lift_cm", 3.0))
        _gnd_min = float(_cr.get("gnd_mm", 0.0))
        _tg_min = float(_cr.get("tg_min", 0.5))
        _drift_max = float(_cr.get("drift_mm", 20.0))
        _allow_missing = bool(_cr.get("allow_missing", False))
        _extra_ok = _cr.get("extra_gate_ok", True)

        _res = dict(result or {})
        _lift = _res.get("obj_lift_final")
        _gnd = _res.get("finger_ground_clear_mm")
        _tg = _res.get("s1_true_grasp_ratio")
        _dft = _res.get("obj_xy_drift")
        _sup = _res.get("s1_obj_supported")
        _cls = _res.get("s1_gripper_ground_class")
        _dmin = _res.get("s1_dmin_truth_mm")

        # ① **数据有效性** ✓ (复审 I.2-4: 缺测**不得**当成通过 ✗)
        #    ★★★★ 2026-09-22 【计划 §4.3-2 修复】只查 `None` **不够** ✗ ★★★★
        #      计划实测 ✓: "令 lift、桌面净空、挡停率、漂移为 **NaN**, 其余字段合法,
        #        实际返回 `data_valid=True, task_success=True`" ✗✗
        #      ⇒ NaN/inf 也是"缺测" ⇒ 必须一并判为**不可信** ✓
        # ⚠ 2026-09-22 修正 ✗: 第一版把**字符串字段** (`s1_gripper_ground_class`)
        #   也交给 `float()` ⇒ 抛异常 ⇒ 被判"缺测" ✗ ⇒ **所有结果都变缺测** ✗✗
        #   (实测: `test_all_good_passes` / `test_legal_zero_q_is_not_penalised` 立即失败 ✓)
        #   ⇒ 数值字段查 `None` + **有限性** ✓; 字符串字段只查 `None`/空 ✓
        def _is_missing(_v, _numeric=True):
            if _v is None:
                return True
            if not _numeric:
                return not bool(str(_v))
            try:
                return not bool(np.isfinite(float(_v)))
            except Exception:  # noqa: BLE001
                return True

        _miss = [k for k, v in (("obj_lift_final", _lift),
                                ("finger_ground_clear_mm", _gnd),
                                ("s1_true_grasp_ratio", _tg),
                                ("obj_xy_drift", _dft),
                                ("s1_obj_supported", _sup)) if _is_missing(v)]
        if _is_missing(_cls, _numeric=False):
            _miss.append("s1_gripper_ground_class")
        _nan = [k for k, v in (("obj_lift_final", _lift),
                               ("finger_ground_clear_mm", _gnd),
                               ("s1_true_grasp_ratio", _tg),
                               ("obj_xy_drift", _dft)) if v is not None and _is_missing(v)]
        _valid = bool(not _miss) or _allow_missing

        # ② **物理可行性** ✓ (复审 I.2-3: 既要排 `load_bearing_block`,
        #    也要排更严重的 `obvious_penetration` ✓; `None/未知` 不给"通过" ✗)
        _inf = []
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-24 【方案 §11.3 实施】**不删除**硬门, 但把新标签**纳入**不可交付集 ✗ ★★★★
        #   方案原话 ✓: \"`CONTACT_CONSTRAINED` … 证据不足则 `UNKNOWN`, **不得放行为成功**\" ✓
        #     \"**不直接跳过 `load_bearing_block` 硬门**\" ✓
        #   ⇒ 允许保留的是 `TABLE_TOUCH` / `PRE_CONTACT` (贴桌轻触 ✓ 方案 §11.3 表 ✓);
        #     `UNKNOWN` 与 `CONTACT_CONSTRAINED` **仍不可交付** ✓ (不许"没证据=通过" ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if _cls in ("load_bearing_block", "obvious_penetration",
                    "CONTACT_CONSTRAINED", "UNKNOWN"):
            _inf.append(f"ground_class={_cls}")
        if _cls is None:
            _inf.append("ground_class=unknown")
        if _sup is True:
            _inf.append("still_supported_by_table")
        if _sup is None:
            _inf.append("support=unknown")
        _feasible = bool(not _inf)

        # ③ **任务成功** ✓ (缺测 ⇒ 记原因, **不通过** ✗)
        _reasons = []
        if not _valid:
            _reasons.append("missing:" + ",".join(_miss))
        if _lift is None or float(_lift) * 100.0 < _lift_min:
            _reasons.append("lift")
        if _gnd is None or float(_gnd) < _gnd_min:
            _reasons.append("ground_clear")
        if _tg is None or float(_tg) < _tg_min:
            _reasons.append("true_grasp")
        if _dft is None or float(_dft) * 1000.0 > _drift_max:
            _reasons.append("drift")
        if not _extra_ok:
            _reasons.append("extra_gate")
        _success = bool(_feasible and _valid and not _reasons)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 任务03 (2026-09-28) 【md 新口径 v2 门: 并列报告, 不改 v1 判据】 ✓ ★★★★
        #   md 原话 ✓: "旧'真夹持率'改名 `joint_block_ratio` **诊断**; 真实门检查
        #     **目标双指载荷、目标离桌、固定 C0 的相对平移/旋转和任务进度**。
        #     **v1/v2 先并列报告**" ✓
        #   本改 ✓: **不动** v1 (不放松也不收紧既有门 ✓), 只在结果里**并列**给出 v2 ✓
        #   ⚠ 诚实标注 ✗ (不假装通过):
        #     · `目标双指载荷` 用 `s1_dual_load_ratio` (**双侧接触帧代理** ⇒
        #       真值需"每帧双指法向冲量"探针, 见 md 记录 ✗)
        #     · `固定 C0 相对平移/旋转` 需"夹爪 6DOF 位姿逐帧记录" ⇒ 当前**未评估**(None) ✗
        #     · `None` 一律**不算通过** (md: 缺测不得当通过 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _v2_load = _res.get("s1_dual_load_ratio")
        # ★ md §03 遗留 ✓: **优先用真探针** (双侧真实法向冲量最小值 > 0 ⇒ 全程双侧载荷 ✓);
        #   探针不可得 ⇒ 回退"接触帧占比"代理 ✓ 并**显式标注用了哪个口径** ✓ (不混用 ✗)
        _v2_load_imp = _res.get("s1_dual_load_impulse_min")
        _v2_load_src = ("impulse_probe(sum_imp_raw)" if _v2_load_imp is not None
                        else "proxy(both_contact_ratio)")
        _v2_c0_t = _res.get("s1_c0_trans_max_mm")
        _v2_c0_r = _res.get("s1_c0_rot_max_deg")
        _v2_gates = {
            # 代理量 ⇒ 门槛也按代理口径写清 (≥0.9 = 抬升段 ≥90% 帧双侧都在接触) ✓
            ("dual_finger_load[%s]" % _v2_load_src):
                (None if (_v2_load_imp is None and _v2_load is None)
                 else bool(float(_v2_load_imp) > 0.0) if _v2_load_imp is not None
                 else bool(float(_v2_load) >= 0.9)),
            "target_off_table":
                (None if _sup is None else bool(_sup is False)),
            "fixed_c0(trans<=5mm & rot<=5deg)":
                (None if (_v2_c0_t is None or _v2_c0_r is None)
                 else bool(float(_v2_c0_t) <= 5.0 and float(_v2_c0_r) <= 5.0)),
            "progress(>=3cm)":
                (None if _lift is None else bool(float(_lift) >= 0.03)),
        }
        _v2_all = bool(_v2_gates) and all(_v is True for _v in _v2_gates.values())
        return {"data_valid": _valid, "phys_feasible": _feasible,
                "task_success": _success, "reasons": _reasons, "missing": _miss,
                "nan_fields": _nan,   # ★ 计划 §4.3-2 ✓: NaN 字段单独列出 ✓ (便于定位 ✗)
                "infeasible": _inf, "dmin_mm": _dmin,
                # ★ 任务03 ✓: 旧"真夹持率"**改名**为 `joint_block_ratio`, 并明确它是**诊断** ✗
                "joint_block_ratio": _tg,
                "true_grasp_ratio": _tg,          # ← 兼容旧调用 (同一量的别名 ✓)
                "joint_block_ratio_is_diagnostic_only": True,
                # ★ 任务03 ✓: v2 门**并列报告** (不影响 task_success ✗)
                "v2_gates": _v2_gates,
                "v2_all_pass": _v2_all,
                "v2_note": ("v1=旧门(见 task_success); v2=md §03 新口径. "
                            "载荷用双侧接触帧代理; C0 项缺夹爪位姿探针 ⇒ 未评估(None) ✗"),
                "criteria": {"lift_cm": _lift_min, "gnd_mm": _gnd_min,
                             "tg_min": _tg_min, "drift_mm": _drift_max}}

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 任务01 (2026-09-28) 【物理预算接账本: 先预占, 再 reset/step】 ✓ ★★★★
    #   md §01-4 原话 ✓: "**唯一物理 episode 入口先预占预算**, 再 reset/step; 异常也消耗
    #     已占次数。生成 ledger 记录 task/candidate/episode/purpose、累计次数、物理子步数" ✓
    #   本改 ✓: `rollout_grasp_only` / `rollout_follow_trajectory` 入口统一经此取预算 ✓
    #   ⚠ 失败(账本不可用) ⇒ 返回 None 并告警 ⇒ **不阻断**既有流程 (逐步推广, 不搞大爆炸 ✗)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    @staticmethod
    def _v2_gripper_base_pose(robot, side):
        """★ md §03 遗留 (2026-09-30) ✓: 取夹爪**基座位姿** `T_W_G` (取不到 ⇒ None ✓)"""
        try:
            for _l in robot.get_links():
                if str(_l.get_name()) == f"{side}_gripper_base_link":
                    return np.asarray(_l.get_pose().to_transformation_matrix(),
                                      dtype=np.float64)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _budget01(self, task="stage1", budget=None):
        """取(惰性建)本 run 的物理预算器 ⇒ `reserve()` 必须发生在 reset/step **之前** ✓"""
        try:
            if getattr(self, "_budget01_obj", None) is None:
                from pathlib import Path as _P
                from pipeline_budget import EpisodeBudget as _EB
                _act = "reuse"
                try:
                    import sys as _sys
                    _gv = vars(_sys.modules.get("__main__", None)) or {}
                    _pl = _gv.get("args", None)
                    # ★ 多源检测 ✓ (s102_deep 实测: 仅看 `_stage1_plan.action` 会漏 ✗ ——
                    #   `--stage1-action search` 时 `_stage1_plan` 可能为 None 或字段名不同 ⇒
                    #   误判 reuse ⇒ budget=0 ⇒ **显式 search 也被拒** ✗)
                    for _src in (getattr(_pl, "_stage1_plan", None),
                                 getattr(_pl, "stage1_action", None),
                                 os.environ.get("_STAGE1_ACTION", "")):
                        _a = str(getattr(_src, "action", _src) or "").lower()
                        if _a in ("search", "verify", "reuse", "legacy"):
                            _act = _a
                            break
                except Exception:  # noqa: BLE001
                    pass
                logger.warning("[任务01][预算] action 检测=%s (源: args/env ✓)", _act)
                _od = _P(str(getattr(self, "output_dir", "output") or "output"))
                self._budget01_obj = _EB(_od / "budget_ledger.json", task=task,
                                         action=_act, physical_budget=budget)
                logger.warning(
                    "[任务01] ✓ 物理预算接账本: %s | action=%s budget=%d "
                    "(md §01-4: 先预占再 step ✓)",
                    _od / "budget_ledger.json", _act, int(self._budget01_obj.budget))
            return self._budget01_obj
        except Exception as _eb01:  # noqa: BLE001
            logger.warning("[任务01] 预算账本不可用 (不阻断): %s", _eb01)
            return None

    def rollout_grasp_only(self, grasp_pose, side, strategy='pd_then_lock', n_frames=160,
                           record_video=False, q_override=None):
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-23 【方案 §4.3 实施】各阶段时长**独立相加** ⇒ 提升不再被压缩 ✗ ★★★★
        #   方案原话 ✓: \"`:1924` 用 `n_frames-前缀长度` 分摊固定 20 cm ✗ ⇒ 各阶段独立按秒定义,
        #     总时长相加, **不保持总 160 帧而让后段越来越快**\" ✓
        #   我这轮的**实测铁证** ✓: `_S1_TRAVEL_FRAMES=90` + 闭合 ~30 + 稳定 40 = 160 ✗
        #     ⇒ **抬升段 = 0 帧** ✗ ⇒ `抬起=False 脱桌=False` 是**构造必然** ✗
        #     (我因此白跑了两轮 ✗ ⇒ 正是方案警告的那种\"后段越挤越没\"✗)
        #   本改 ✓: `_S1_N_FRAMES` 可**独立**指定总帧数 ✓ (默认 0 = 用调用方给的 n_frames ✓
        #     行为不变 ✓) ⇒ 就位/闭合/稳定/抬升/保持 各段按秒数**相加** ✓
        #     例 ✓: 30(就位1s) + 48(闭合1.6s) + 15(稳定0.5s) + 120(抬升**4s** ✓) + 15(保持0.5s) = 228 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-24 【用户定策】粗筛: "只需要得到最优抓取位姿, 不要跑抬升" ✓ ★★★★
        #   用户原话 ✓: "你只需要得到**最优的抓取位姿**, 而不是把其他的部分放上" ✓;
        #     "先选三个得分高的, 再对它们优化得到最终抓取位姿" ✓
        #   ⇒ `_S1_SCREEN_FRAMES` (默认 0=全程 ✓ 不改旧行为 ✗): 设 92 ⇒ 只跑
        #     **就位30 + 闭合48 + 稳定15 = 93 帧** (≈无抬升 ✓) ⇒ 判断"能否双侧接触"足够 ✓
        #   ⚠ **只作用于搜索调用** ✓: `q_override is None` 才裁剪 ✗ ⇒
        #     **终验/冻结复验/录像** 一律用完整模板 (含 20cm 抬升 ✓) 不受影响 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _screen_n = int(os.environ.get("_S1_SCREEN_FRAMES", "0") or 0)
        if _screen_n > 0 and q_override is None and int(_screen_n) < int(n_frames):
            logger.info("[粗筛] ✓ 搜索调用裁剪到 **%d 帧** (就位+闭合+稳定, 不含抬升 ✓); "
                        "终验/复验仍用完整 %d 帧 ✓ (含 20cm 抬升 ✓)", _screen_n, int(n_frames))
            n_frames = int(_screen_n)
        _n_frames_env = int(os.environ.get("_S1_N_FRAMES", "0") or 0)
        if _n_frames_env > 0:
            logger.info("[方案 §4.3] `_S1_N_FRAMES=%d` ⇒ 覆盖调用方 n_frames=%d ✓ "
                        "(阶段时长独立相加, 提升不再被挤压 ✓)", _n_frames_env, int(n_frames))
            n_frames = int(_n_frames_env)
        """Stage1 抓取 rollout ✓

        ★ 2026-09-22 【复审第 4 条】新增 `q_override` ✓:
          复审 ✓: "**冻结动作没有真正接通**: 最终验证仍**重算闭合动作**" ✗
          ⇒ 传入该值时, F50 的闭合/保持命令**直接采用它** ✓
            (终验与交付消费**同一份冻结命令** ⇒ "验证动作 = 交付动作" 才成立 ✓)
          默认 `None` ⇒ 保持原行为 ✓ (按物体尺寸重算 ✓)
        """
        # ★★★★ 2026-09-18 桌面几何预筛 (用户: "先测试会不会与桌面交互, 交互直接 pass,
        #   然后再测试抬起, 这样会快一点" ) ★★★★
        #   背景 ✓: 一旦**开启桌面碰撞** (`_NO_GRIPPER_GROUND` 不设 ✓), Stage1 立刻 0/3 ✗
        #     (实测 output/s1table: lift=[0,0,0]cm ✗; 且候选的 c_clear=+7.9~+9.1mm ✗ 几何全坏 ✗)
        #     ⇒ 原因是 CEM 在"会撞桌面"的候选上白白花掉物理 rollout ✗, 且搜不到干净位姿 ✗
        #   本预筛 ✓: **纯几何**判断该候选会不会与桌面交互 ⇒ 会 ⇒ **直接 pass(剔除)** ✓
        #     · 只做一件极便宜的事: 比较候选基座高度与阈值 `_S1_TABLE_MIN_Z` (默认 0.02m ✓)
        #     · 门槛可扫 ✓: 0.02 = "基座低于 20mm"(对 39mm 高、物心 19.6mm 的物体 ⇒ 明显扎进桌面 ✓)
        #   ★ 与之前三次失败筛选的本质区别 ✓:
        #     之前筛的是**物体几何好坏** (c_center / c_clear ✗) ⇒ 属偏好 ✗ ⇒ 挤掉了可行解 ✗
        #     本筛的是**桌面** ✓ ⇒ 硬物理约束 ✓ (碰桌面 = 必然失败 ✗) ⇒ 剔除不会误伤 ✓✓✓
        #   开关 ✓: `_S1_TABLE_PRESCREEN=1` 显式开启; **默认关** ✓
        #     ★ 2026-09-20 【P0】默认由 "1" 改为 **"0"** —— 本预筛的几何是**错的** ✗
        #       实测 (output/s1_coacd_table_on_0920, 桌面碰撞开启): 预筛触发 **1829 次** ✗,
        #         CEM 从 iter0 到 iter14 只用 **7 秒** ⇒ **一次物理 rollout 都没跑** ✗,
        #         best 恒为 −28.000 不动 ⇒ 搜索直接死掉 ✗ (验证门 0/3 ✗)
        #       根因 ✓: `_lowest = min(p1z,p2z) − 0.020` 里
        #         ① p1/p2 是**手指原点沿闭合轴 ± 半间距(65mm)** 的**模型点** ✗, 不是真实指尖 ✗;
        #         ② 减掉的 "手指半厚 20mm" 是拍的 ✗, 远大于真实指厚 ⇒ 保守到失真 ✗
        #         本任务 (39mm 高物体、侧向捏合) 的真实指尖本来就会低于物心、贴近桌面 ✓
        #         ⇒ 该预筛把**所有可行候选**判为"碰桌" ✗
        #       结论 ✓: 预筛只是省时间的启发式, 而物理 rollout 本身就是**真值** ✓
        #         ⇒ 默认关掉, 让物理说话; 需要省时间时再显式 `_S1_TABLE_PRESCREEN=1` ✓
        if os.environ.get("_S1_TABLE_PRESCREEN", "0") == "1":
            try:
                _min_z = float(os.environ.get("_S1_TABLE_MIN_Z", "0.005"))
                _R = np.asarray(grasp_pose['R'], dtype=np.float64)
                _p = np.asarray(grasp_pose['pos'], dtype=np.float64)
                _cd = _R @ np.array([0.0, 1.0, 0.0])      # 闭合轴 ✓
                _fw = _R @ np.array([1.0, 0.0, 0.0])      # 指尖方向 ✓
                _fo = _p + 0.03689 * _fw                  # 手指原点 ✓
                _sep = float(FINGER_EFFECTIVE_HALF_SPACING) + float(
                    os.environ.get("_Q_OPEN", "0.05") or 0.05)
                _p1 = _fo + _sep * _cd
                _p2 = _fo - _sep * _cd
                # 指尖最低点再减"手指半厚"(保守取 20mm ✓ ⇒ 偏保守 ⇒ 不会漏掉真碰桌的候选 ✓)
                _lowest = min(float(_p1[2]), float(_p2[2])) - 0.020
                if _lowest < _min_z:
                    logger.info(
                        "[Stage1 桌面预筛] 剔除: 指尖最低点=%.4f (base_z=%.4f) < %.4f ⇒ 会与桌面交互 ✗",
                        _lowest, float(_p[2]), _min_z,
                    )
                    return {
                        'obj_lift_final': 0.0, 'obj_lift': 0.0,
                        'fingertip_obj_dist': 1.0, 'fingertip_obj_dist_min': 1.0,
                        'both_contact_count': 0, 'contact_count': 0,
                        'obj_xy_drift': 1.0, 'peak_grip_force': 0.0,
                        'grip_stability': 0.0, 'obj_z_final': 0.0, 'obj_init_z': 0.0,
                        'obj_z_history': [], 'locked_gripper_qpos': 0.0,
                        'finger_qpos_asym': 0.0,
                        'entry_target_contact': False, 'approach_contact_count': 0,
                        'gripper_approach_axis': None,
                        'table_prescreened': True,
                    }
            except Exception as _etp:  # noqa: BLE001
                logger.warning("[Stage1 桌面预筛] 失败(跳过): %s", _etp)

                # ===== md 06.4-f0 (2026-09-30) 负 drive 未声明 ⇒ 首步前拒绝 =====
                # f0: 实际 q 越界与 drive 参考越界是两回事; 允许负 drive 作限位外预紧必须
                #     spec 明确(范围/力限/行程保护/最终 drive 读回); 未定义则拒绝该控制模式,
                #     不自动改 URDF 或钳成 0.
                # 实测: URDF 手指 limit lower=0 ⇒ 本 artifact 实际下发 min=-0.000978 越界
                #       (见 artifact_migration_audit.json / joint_command_policy.json)
                # 本改: 未声明时请求负 q ⇒ 在 reserve/step 之前拒绝(零物理), 并给出声明方式
                _allow_neg01 = str(os.environ.get("_S1_ALLOW_NEGATIVE_DRIVE", "0")).strip().lower() not in ("0", "false", "no", "")
                if not _allow_neg01:
                    for _qk01 in ("gripper_qpos", "q_grasp", "q_target", "q_cmd"):
                        try:
                            _qv01 = grasp_pose.get(_qk01) if isinstance(grasp_pose, dict) else None
                        except Exception:  # noqa: BLE001
                            _qv01 = None
                        if isinstance(_qv01, (int, float)) and float(_qv01) < -1e-9:
                            logger.error(
                                "[f0] 拒绝该控制模式: 目标 %s=%.6f < URDF lower(0) 且负 drive 未声明"
                                " ⇒ 首步前拒绝 (不钳 0, 不改 URDF). 如需启用: _S1_ALLOW_NEGATIVE_DRIVE=1"
                                " + 声明 q 下限/力限/行程保护 + 最终 drive 读回",
                                _qk01, float(_qv01))
                            return {
                                'obj_lift_final': 0.0, 'obj_lift': 0.0,
                                'fingertip_obj_dist': 1.0, 'fingertip_obj_dist_min': 1.0,
                                'both_contact_count': 0, 'contact_count': 0,
                                'obj_xy_drift': 1.0, 'peak_grip_force': 0.0,
                                'grip_stability': 0.0, 'obj_z_final': 0.0, 'obj_init_z': 0.0,
                                'obj_z_history': [], 'locked_gripper_qpos': 0.0,
                                'finger_qpos_asym': 0.0, 'entry_target_contact': False,
                                'approach_contact_count': 0, 'gripper_approach_axis': None,
                                'table_prescreened': False,
                                'rejected_reason': 'NEGATIVE_DRIVE_UNDECLARED',
                            }

        # ★★ 任务01 (2026-09-28) 【物理 episode 入口: **先预占预算, 再 reset/step**】 ✓ ★★
        #   md §01-4 ✓: "唯一物理 episode 入口先预占预算, 再 reset/step; 异常也消耗已占次数" ✓
        #   ★ 位置修正 (B3) ✓: 放在**桌面几何预筛之后** —— 预筛是**纯几何**、零物理 ✗,
        #     被它剔除的候选**不应**消耗 48 次物理预算 ✓ (旧放法会让搜索更快耗尽 ✗)
        #   ⇒ 若本 run 未显式 `--stage1-action search` 而 action=reuse (budget=0),
        #     这里仍会**在任何 reset/step 之前**抛 `BudgetExhausted` ✓ (预算不可绕过 ✓)
        # ★ B2/复用链修正 (2026-09-28) ✓: **verify 不是搜索** ⇒ 必须走独立预算池
        #   md §3.1 ✓: "Stage1 verify ≤3" ⇒ 用 `q_override` 是否传入区分用途 ✓:
        #     · q_override=None          ⇒ **搜索用途** ⇒ task=stage1 (action=reuse ⇒ budget=0 ⇒
        #                                  正确拒绝"默认 reuse 却跑搜索" ✗)
        #     · q_override is not None   ⇒ **冻结终验** ⇒ task=stage1_verify (budget=3 ✓,
        #                                  可被 `_S1_VERIFY_BUDGET` 覆盖) ⇒ 不再误拒 ✓
        _is_verify01 = (q_override is not None)
        _bud01 = self._budget01(
            task=("stage1_verify" if _is_verify01 else "stage1"),
            budget=(int(os.environ.get("_S1_VERIFY_BUDGET", "3") or 3) if _is_verify01 else None))
        if _bud01 is not None:
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ md §09-2/3/4 (2026-09-30) 【搜索阶段**必须给终验留预算**】 ✓ ★★★★
            #   实测铁证 ✗: `s102_deep` 与 `s102_deep2` **两次都是 48/48 全被搜索吃掉、
            #     零 recipe** ⇒ md 的有界调度 (筛选≤21 + 局部≤24 + **终验 3** = 48)
            #     **从未实现** ✗ ⇒ 即使搜到好候选也**交付不了** (终验没预算) ✗
            #   本改 ✓: **搜索用途**下, 剩余 ≤ `_S1_VERIFY_RESERVE`(默认 3) ⇒ 立即停止搜索
            #     (抛与"预算用尽"**同类**的 `BudgetExhausted` ⇒ 走既有上层处理路径 ✓);
            #     **终验**走独立池 `stage1_verify` (budget=3, 见 C3 ✓) ⇒ 两者相加 ≤48 ✓
            #   ⚠ 不静默继续 ✗: 明确报"主动停止的原因" ✓ (md §6: 不一致/异常要可见 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if not _is_verify01:
                _keep01 = int(os.environ.get("_S1_VERIFY_RESERVE", "3") or 3)
                if _bud01.remaining() <= _keep01:
                    # ⚠ 2026-09-30 修复 ✗: 首版此处 `raise BudgetExhausted(...)` 只传 1 个参数,
                    #   而签名为 `(rule, detail)` ⇒ 抛 `TypeError` (**不是** BudgetExhausted)
                    #   ⇒ 上层捕获不到 ⇒ `s102_deep3` 在 **45/48 处崩溃** ✗✗
                    #   (实测: `BudgetExhausted.__init__() missing 1 required positional
                    #    argument: 'detail'`)
                    #   本改 ✓: **不再走异常路径** ⇒ 直接返回"跳过"结果 (与"桌面预筛剔除"
                    #   的返回**同构** ✓) ⇒ 搜索侧按差候选处理、**不消耗物理** ✓
                    #   ⇒ 剩余预算留给终验 (独立池 ✓), 且**不可能**再因异常中断 ✓
                    logger.warning(
                        "[md §09] **搜索阶段主动停止**: 剩余 %d ≤ 保留 %d (为终验留位 ✓) "
                        "⇒ 本候选**跳过**(不耗物理 ✓); 终验走独立池 stage1_verify ✓",
                        int(_bud01.remaining()), int(_keep01))
                    return {
                        'obj_lift_final': 0.0, 'obj_lift': 0.0,
                        'fingertip_obj_dist': 1.0, 'fingertip_obj_dist_min': 1.0,
                        'both_contact_count': 0, 'contact_count': 0,
                        'obj_xy_drift': 1.0, 'peak_grip_force': 0.0,
                        'grip_stability': 0.0, 'obj_z_final': 0.0, 'obj_init_z': 0.0,
                        'obj_z_history': [], 'locked_gripper_qpos': 0.0,
                        'finger_qpos_asym': 0.0,
                        'entry_target_contact': False, 'approach_contact_count': 0,
                        'gripper_approach_axis': None,
                        'table_prescreened': False,
                        'budget_reserved_skip': True,     # ← 明确标记 (非"物理失败" ✗)
                    }
            _bud01.reserve(
                purpose=("stage1_frozen_verify" if _is_verify01 else "stage1_rollout"),
                candidate=(str(grasp_pose.get("candidate_id", "") or "") or None),
                phase=str(strategy))
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
                    robot, side, self.obj_actors, target_name,
                    scene=getattr(self, "scene", None))   # ★ 任务02 ✓ 同上: 地面也要设
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
                    # ★★★★ 2026-09-22 【P0-3 修复】必须**同时清零速度** ✗ ★★★★
                    #   审计文档 §P0-3-1 原话 ✓: "Stage1 每次 rollout 对物体只 `set_pose`,
                    #     没有同时清零其线速度、角速度。优化器入口处清零一次不等于
                    #     每个候选/重复验证都清零" ✗
                    #   ⇒ 上一候选残留的**物体速度**会带着物体继续跑 ✗
                    #     ⇒ 各候选的"初态"其实**不相同** ✗ (这正是"一开始物体就不对" ✓)
                    #   本改 ✓: 每次 reset 都清零**线速度 + 角速度** ✓
                    # ⚠⚠ 2026-09-22 【复审修正 — 清速度**从未生效** ✗】⚠⚠
                    #   实测 ✓: `'sapien.pysapien.Entity' object has no attribute
                    #     'set_linear_velocity'` ✗ ⇒ 每次 reset 都刷一串 WARNING ✗,
                    #     而物体速度**从来没被清零** ✗ (复审第 1 条 ✓)
                    #   正确写法 ✓: 找到该 actor 的 **PhysxRigidDynamicComponent** 再设 ✓
                    try:
                        _cleared = False
                        for _comp in actor.components:
                            if isinstance(
                                    _comp,
                                    sapien.pysapien.physx.PhysxRigidDynamicComponent):
                                _comp.set_linear_velocity(np.zeros(3))
                                _comp.set_angular_velocity(np.zeros(3))
                                _cleared = True
                                break
                        if not _cleared:
                            logger.warning(
                                "[Stage1 reset] %s 无动态组件 ⇒ 速度未清零 ⚠", name)
                    except Exception as _ev:  # noqa: BLE001
                        logger.warning("[Stage1 reset] 物体速度清零失败 (%s): %s", name, _ev)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【计划 §6 · A 组根因】Stage1 必须与 Stage2 **共享同一初态** ✓ ★★★★
            #   实测铁证 ✓ (output/Ab1, **同一冻结动作** ×3 ✓):
            #     lift = **0.0 / 0.0 / 20.2cm** ✗ | "物体↔地面接触占比" = **1.00 / 0.98 / 0.09** ✗
            #     ⇒ 两次**根本没离桌** ✗ ⇒ **旧候选本身不可重放 (1/3)** ✗
            #   而 Stage2 侧初态是**一致的** ✓ (实测: `物体静置自稳定 90 步: 位移 0.12mm` ✓ /
            #     `驱动目标已复位 ✓` / `静置后与规划输入一致 ✓`)
            #   ⇒ 差异就在 Stage1 侧 ✗: 只做了 `set_pose` + 清速度 (**P0-3-1** ✓),
            #     **没有** ① **驱动目标复位** ② **静置自稳定** ✗
            #   本改 ✓ (对齐 Stage2 ✓):
            #     ① 各主动关节 **drive target = 当前 qpos** ✓ (消除上一轮驱动目标残留 ✗)
            #     ② 物体 **静置自稳定** `_S1_SETTLE_STEPS` (默认 **90** ✓ 与 Stage2 一致 ✓) 步,
            #        期间**钉住机器人** (`set_qpos/set_qvel` 每步重设 ✓) ⇒ 只让物体落稳 ✓
            #        静置后**再清一次物体速度** ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                # ⚠ 2026-09-23 修正 ✗: 上一版用 `joint.get_index()` ⇒ 该 SAPIEN 版本**没有**这个
                #   方法 ⇒ 实测 `驱动目标已复位 ✓ (**0** 个主动关节)` ✗ ⇒ **复位从未生效** ✗
                #   本改 ✓: 与 **Stage2 完全同写法** ⇒ `enumerate(active_joints)` 对齐 qpos ✓,
                #    并同时复位 `set_drive_velocity_target(0)` ✓ (Stage2 也这么做 ✓)
                _n_dr = 0
                _active_j = list(robot.get_active_joints())
                _q_now = np.asarray(robot.get_qpos(), dtype=np.float64).copy()
                for _ji, _j in enumerate(_active_j):
                    _val = float(_q_now[_ji]) if _ji < len(_q_now) else 0.0
                    for _setter, _v2 in (("set_drive_target", _val),
                                         ("set_drive_velocity_target", 0.0)):
                        try:
                            getattr(_j, _setter)(_v2)
                            if _setter == "set_drive_target":
                                _n_dr += 1
                        except Exception:  # noqa: BLE001
                            continue
                logger.warning("[Stage1 reset] ★ **驱动目标已复位** ✓ (active_joints=%d/%d, "
                               "与 Stage2 同写法 ✓)", _n_dr, len(_active_j))
            except Exception as _edt:  # noqa: BLE001
                logger.warning("[Stage1 reset] 驱动目标复位失败: %s", _edt)
            # ★ 计划 §6 ✓: A vs B 的**第一处分叉**证据 ⇒ 记录"**闭合开始前**"的状态 ✓
            #   (计划原话 ✓: "比较闭合开始时物体 T、v、ω, 实际夹爪 T、q、qdot")
            try:
                _ta = None
                for _a2 in self.obj_actors:
                    if target_name and (str(target_name) in _a2.get_name()
                                        or _a2.get_name() in str(target_name)):
                        _ta = _a2
                        break
                _oT = _ta.get_pose().to_transformation_matrix() if _ta is not None else None
                _ov = _ow = None
                if _ta is not None:
                    for _c3 in _ta.components:
                        if isinstance(_c3, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                            try:
                                _ov = np.asarray(_c3.get_linear_velocity(), dtype=np.float64)
                                _ow = np.asarray(_c3.get_angular_velocity(), dtype=np.float64)
                            except Exception:  # noqa: BLE001
                                pass
                            break
                logger.warning(
                    "[A/B 对比·闭合前状态] 物体 T_pos=%s | v=%s | ω=%s | 夹爪 q=%s | qdot=%s",
                    np.round(np.asarray(_oT)[:3, 3], 4).tolist() if _oT is not None else "n/a",
                    np.round(_ov, 5).tolist() if _ov is not None else "n/a",
                    np.round(_ow, 5).tolist() if _ow is not None else "n/a",
                    np.round(np.asarray(robot.get_qpos(), dtype=np.float64), 4).tolist()[:6],
                    np.round(np.asarray(robot.get_qvel(), dtype=np.float64), 4).tolist()[:6])
            except Exception as _edg:  # noqa: BLE001
                logger.warning("[A/B 对比·闭合前状态] 记录失败: %s", _edg)
            try:
                _n_settle = int(os.environ.get("_S1_SETTLE_STEPS", "90") or 0)
                if _n_settle > 0:
                    _q_keep = np.asarray(robot.get_qpos(), dtype=np.float64).copy()
                    _qv_keep = np.zeros_like(np.asarray(robot.get_qvel(), dtype=np.float64))
                    for _s in range(_n_settle):
                        self.scene.step()
                        robot.set_qpos(_q_keep)
                        robot.set_qvel(_qv_keep)
                    for _actor2 in self.obj_actors:
                        try:
                            for _comp2 in _actor2.components:
                                if isinstance(_comp2,
                                              sapien.pysapien.physx.PhysxRigidDynamicComponent):
                                    _comp2.set_linear_velocity(np.zeros(3))
                                    _comp2.set_angular_velocity(np.zeros(3))
                                    break
                        except Exception:  # noqa: BLE001
                            pass
                    logger.info("[Stage1 reset] ★ 物体**静置自稳定 %d 步** ✓ (与 Stage2 同流程 ✓)",
                                _n_settle)
            except Exception as _est:  # noqa: BLE001
                logger.warning("[Stage1 reset] 静置自稳定失败: %s", _est)

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
        # ★★★★ 2026-09-20 【真值闭环·对中偏置】夹爪中线与物体存在**系统性偏移** ⇒ 直接补 ★★★★
        #   实测 (output/grip2, 引擎真值 ✓): 5 个不同候选的偏心**几乎恒定** ✓
        #     `两指到物体距离差=7.6/7.7/7.6/7.6/7.7mm`, 最近侧恒为 **1.3~1.6mm** ✓
        #     ⇒ 不是"候选不行" ✗, 而是**夹爪中心线相对物体偏了 ≈3.8mm** ✓ (系统性 ✓)
        #       ⇒ 一指几乎贴上 (1.6mm), 另一指悬空 ~9mm ⇒ **单指接触** ⇒ 抓不紧 ✗
        #   修法 ✓ (真值闭环 ✓): 沿**闭合轴**把基座平移 `asym/2` 到**远离的那一侧** ⇒ 重新居中 ✓
        #     方向定义 ✓: finger1 在夹爪局部 −Y, finger2 在 +Y (见闭合目标推导 ✓)
        #       ⇒ 物体偏 finger1 一侧 (d1 小) ⇒ 基座应向 **−R[:,1]** 侧移动 ✓
        #   开关 ✓: `_S1_CENTER_BIAS_MM` (默认 0 = 关闭 ✓ 行为不变 ✓; 正=朝 −R[:,1] ✓)
        _cbias_mm = float(os.environ.get("_S1_CENTER_BIAS_MM", "0") or 0.0)
        if _cbias_mm != 0.0:
            target_pos = target_pos + np.asarray(target_R, dtype=np.float64)[:, 1] * (
                -_cbias_mm / 1000.0)
            logger.info("[Stage1 对中偏置] 沿闭合轴平移 %+.2fmm (朝 %s) 补偿系统性偏心 ✓",
                        -_cbias_mm, "finger1 侧")
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
        # ★ 2026-09-22 【复审第 4 条】有**冻结命令** ⇒ 以它为准 ✓ (不再按尺寸重算 ✗)
        _q_frozen = (float(q_override) if q_override is not None else None)
        if _q_frozen is not None:
            _GRIP_TARGET = _q_frozen
            logger.info("[复审第4条] ✓ 采用**冻结的 F50 命令** q=%.4f ⇒ "
                        "验证动作 = 交付动作 ✓ (不再现场重算 ✗)", _q_frozen)
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
                # ★★★★ 2026-09-20 【P0 真根因】闭合目标必须用 **CoACD 真实投影跨度** ★★★★
                #   用户定策 (2026-09-20): "stage1 一定是用 coacd 排的 ... **不要再用 bbox 了**" ✓
                #   实测 (output/ccacd3, 终选位姿的 rollout 内部):
                #     `_GRIP_TARGET = 0.0059` ⇒ 指缝 = GAP0 + 2q = 30.4 + 11.8 = **42.2mm** ✗
                #     而物体沿闭合轴的**真实跨度 = 37.6mm** (CoACD 凸包 ±18.8mm ✓)
                #     ⇒ **4.6mm 空隙** ⇒ 手指只是"贴住"没"夹紧" ✗
                #     ⇒ 实测 `both_contact=18~21 帧` ✓ 但 `obj_lift_peak=0.15~0.76cm`
                #       然后 `final obj_lift=0.00cm` ✗ = **抬起即滑落** ✓ (peak_force 仅 0.05~0.27N ✗)
                #   来源 ✗: 0.0059 是从 `min(bbox[0],bbox[1]) = 44.9mm` 推的 ⇒ 高估 **7.3mm** ✗
                #     (bbox 是物体**局部系**的盒, 与该位姿下闭合轴的**真实投影**无关 ✗)
                #   修法 ✓: 直接用 CoACD 凸包在**本候选闭合轴**上的跨度算 q_contact ✓
                #     (跨度 = max−min, 与物心无关 ⇒ 平移不变 ✓ 不会被"物体漂移"污染 ✓)
                #   仍走 `min(...)` ⇒ 只会**更保守**(更夹紧) ✓, 不会变松 ✓
                _q_contact_coacd = None
                _band_info = None
                if os.environ.get("_S1_LEGACY_CLOSE_TARGET", "0") != "1":
                    try:
                        _hulls_c = self._coacd_world_hulls(target_name)
                        if _hulls_c:
                            _y_ax_c = np.asarray(target_R, dtype=np.float64)[:, 1]
                            _V = np.concatenate(
                                [np.asarray(h, dtype=np.float64) for h in _hulls_c],
                                axis=0)
                            _s_all = _V @ _y_ax_c
                            _span_all = float(_s_all.max() - _s_all.min())
                            # ★★★★ 2026-09-20 【P0 真因】跨度必须只统计**两指所在 z 带内**的凸包顶点 ★★★★
                            #   实测铁证 (output/vis1, 终选位姿 3 次重复 ✓):
                            #     `final f1_qpos=0.0206 == f2_qpos=0.0206 == PD_target=0.0204`
                            #       ⇒ 手指**自由到达目标位**, 没有任何阻挡 ✓
                            #     `peak_force=0.00N`, `obj_lift=0.00cm` ⇒ **零接触** ✓
                            #     而 `指缝=71.3mm < 物体跨度=75.3mm` ⇒ 若真挡住, 手指**不可能**到位 ✗
                            #     ⇒ 结论 ✓: **物体在手指所在高度处的局部宽度 < 71.3mm** ✓
                            #       (75.3mm 是**整个物体**沿闭合轴的投影, 属于"把物体当一根柱子"的
                            #        盒式近似残留 ✗ —— 物体上部窄、下部宽, 手指夹在窄处 ⇒ 夹空 ✓)
                            #   修法 ✓: 用**手指 OBB 的 z 跨度**作带通 (与 q 无关 ✓
                            #     因为手指沿局部 Y 平移, z 不变 ✓), 只统计带内顶点 ✓
                            #   开关 ✓: `_S1_CLOSE_BAND_MM<0` 关闭带通 (回退整物体投影 ✓)
                            _band_mm = float(os.environ.get("_S1_CLOSE_BAND_MM", "2.0") or 2.0)
                            _z_lo = _z_hi = None
                            if _band_mm >= 0.0:
                                try:
                                    from stage2_approach_opt import (
                                        _FINGER_HALF_EXTENT as _FHE,
                                        _FINGER_LOCAL_CENTER as _FLC,
                                        _FINGER_HALF_GAP as _FHG,
                                    )
                                    _Rg_c = np.asarray(target_R, dtype=np.float64)
                                    _bp_c = np.asarray(target_pos, dtype=np.float64)
                                    _hg_c = float(_FHG) + float(_GRIP_TARGET)
                                    _hfe_c = np.asarray(_FHE, dtype=np.float64)
                                    for _sg in (+1.0, -1.0):
                                        _lk = _bp_c + _Rg_c @ np.array([0.03689, -_hg_c * _sg, 0.0])
                                        _lc3 = np.asarray(_FLC, dtype=np.float64).copy()
                                        _lc3[1] *= _sg
                                        _bc = _lk + _Rg_c @ _lc3
                                        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                                        # ★★★★ T5c (2026-09-26) 【窄带=抓取高度附近】✗ 不用整根手指 OBB ✗ ★★★★
                                        #   实测铁证 ✓ (output/s2/s3/s5 日志): 带打印为
                                        #     `带=[-1.4, 61.6]mm` ⇒ **63mm 高** ✗ ⇒ 几乎等于全物体 ✗
                                        #   病根 ✓ (本行旧写法): `_sup = Σ|R_z,·|·(手指半尺寸)`
                                        #     = **整根手指 OBB 的 z 跨度** ✗ —— 手指长约 77mm,
                                        #     倾斜时 z 跨度很大 ⇒ 带内包含整个物体高度 ✗
                                        #     ⇒ 局部截面退化成"全物体投影" ⇒ 闭合目标偏宽
                                        #       ⇒ 两侧各差 4~6mm 没碰到 ⇒ 力 0 ✗ (实测 4.38/5.46mm ✓)
                                        #   ⇒ 本改 ✓: 带 = **抓取高度附近 ±3mm** ✓
                                        #     (接触发生在指腹/指尖附近 ✓ ⇒ 只统计该高度的局部截面 ✓)
                                        #     轴/手指自转已由上面 `_bc` 给出**实际接触高度** `_bc[2]` ✓
                                        #   可用 `_S1_BAND_HALF_MM` 调 (默认 3.0mm ✓)
                                        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                                        _sup = float(os.environ.get("_S1_BAND_HALF_MM", "3.0")
                                                     or 3.0) / 1000.0
                                        _lo_f, _hi_f = float(_bc[2]) - _sup, float(_bc[2]) + _sup
                                        _z_lo = _lo_f if _z_lo is None else min(_z_lo, _lo_f)
                                        _z_hi = _hi_f if _z_hi is None else max(_z_hi, _hi_f)
                                except Exception:  # noqa: BLE001
                                    _z_lo = _z_hi = None
                            if _z_lo is not None:
                                _m = _band_mm / 1000.0
                                _mask = (_V[:, 2] >= _z_lo - _m) & (_V[:, 2] <= _z_hi + _m)
                                _n_in = int(_mask.sum())
                                if _n_in < 8:
                                    # ★ 带太薄 (指厚仅 ~10mm, 凸包顶点稀疏) ⇒ 回退:
                                    #   取**离指带中心最近的 K 个顶点** (K = max(24, 2% 顶点) ✓)
                                    #   —— 仍是"手指所在高度"的局部几何 ✓, 不会退回整物体投影 ✗
                                    _zc = 0.5 * (_z_lo + _z_hi)
                                    _K = max(24, int(0.02 * _V.shape[0]))
                                    _idx = np.argsort(np.abs(_V[:, 2] - _zc))[:_K]
                                    _span_c = float(_s_all[_idx].max() - _s_all[_idx].min())
                                    _band_info = (-_K, _V.shape[0], _z_lo, _z_hi, _span_all)
                                elif _n_in >= 8:
                                    _span_c = float(_s_all[_mask].max() - _s_all[_mask].min())
                                    _band_info = (_n_in, _V.shape[0], _z_lo, _z_hi, _span_all)
                            else:
                                _span_c = _span_all
                            _q_contact_coacd = max(
                                0.0, _span_c / 2.0 - FINGER_EFFECTIVE_HALF_SPACING)
                            logger.info(
                                "[Stage1 闭合目标] ★CoACD 跨度=%.1fmm (%s) → q_contact=%.4f "
                                "(bbox 估计 min边=%.1fmm→%.4f ✗ 高估 %.1fmm)",
                                _span_c * 1000.0,
                                ("整物体投影" if _band_info is None else
                                 ("z带内 %d/%d 顶点, 带=[%.1f,%.1f]mm, 整物体投影=%.1fmm"
                                  % (_band_info[0], _band_info[1], _band_info[2] * 1000.0,
                                     _band_info[3] * 1000.0, _band_info[4] * 1000.0))
                                 if _band_info[0] > 0 else
                                 ("近邻 %d/%d 顶点(带太薄), 指带心=%.1fmm, 整物体投影=%.1fmm"
                                  % (-_band_info[0], _band_info[1],
                                     0.5 * (_band_info[2] + _band_info[3]) * 1000.0,
                                     _band_info[4] * 1000.0))),
                                _q_contact_coacd,
                                _bbox_min_dim * 1000.0, _q_contact_min,
                                (_bbox_min_dim - _span_c) * 1000.0)
                    except Exception as _ec_c:  # noqa: BLE001
                        _q_contact_coacd = None
                        logger.warning("[Stage1 闭合目标] CoACD 跨度获取失败 (回退 bbox): %s", _ec_c)
                _cands = [c for c in (_q_contact_min, _q_contact_old, _q_contact_proj,
                                      _q_contact_coacd) if c is not None]
                # ★★★★ 2026-09-20 【用户】"位姿效果很好了, 就是**闭合尺度太小**了" ★★★★
                #   旧行为 ✗: `min(bbox_min边, bbox_max, bbox投影, CoACD投影)` **取最小** ✓
                #     实测 (output/ccacd3): 采用值 0.0072 (**bbox min 边** 44.9mm 主导 ✓)
                #       ⇒ 被 CoACD 真实跨度 (37.6mm ⇒ 应取 0.0036) **挤掉** ✗
                #       (因为 `_q_contact_min` 用的是**物体局部 bbox**, 与本候选闭合轴无关 ✗)
                #     ⇒ 指缝 = 30.4 + 2×7.2 = **44.8mm** ⇒ 比物体跨度 37.6mm **大 7.2mm** ✗
                #     ⇒ 手指停在物体两侧**悬空** ⇒ 零压紧力 ⇒ 抬起即滑落 ✗ (与实测一致 ✗)
                #   新行为 ✓ (开关 `_S1_CLOSE_FROM_HULL=1`, 默认开 ✓):
                #     ① **以 CoACD 真实投影跨度为基准** (本候选闭合轴 ✓, 与位姿一一对应 ✓)
                #     ② 再**减去过闭合量** `_S1_CLOSE_OVERCLOSURE_MM` (默认 2.0mm = 安全上限 ✓)
                #        ⇒ 手指一定被物体挡停并产生正压力 ✓
                #     ③ 天花板不变 (嵌入 ≤2.0mm ✓, 再深会因 64 重叠凸包法向冲突弹射 ✗)
                _from_hull = (os.environ.get("_S1_CLOSE_FROM_HULL", "1") == "1"
                              and _q_contact_coacd is not None)
                if _from_hull:
                    _q_contact = float(_q_contact_coacd)
                    _oc_mm = float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MM", "2.0") or 2.0)
                    # ★★★★ 2026-09-20 【纠错】过闭合量**必须小**: 它是"弹射"的源 ✗ ★★★★
                    #   用户定调 ✓: "**硬压是为了保证能够抓住物体, 有摩擦**" —— 压紧本身是对的 ✓,
                    #     但"压紧"= 手指停在**物体表面**、靠 PD 保持正压力 ✓ (静力平衡 ✓),
                    #     不是"压进去 X 毫米" ✗ —— 后者只会让求解器产生巨量分离冲量 ✗
                    #   实测 (output/final1, 过闭合 6.5mm ⇒ 指缝 63.0mm vs 物体跨度 76.0mm) ✓:
                    #     真值探针: F40 顶点最近 6.7mm + f1_c=f2_c=True (**已经压上了** ✓)
                    #       ⇒ 然后再被推 13mm ⇒ 冲量爆炸 ⇒ 物体**飞出画面** ✗
                    #       (探针最近距离 6.7→23→96mm 一路增大 = 物体被轰走 ✓)
                    #     ⇒ 说明"抓不住"的原因**不是压得不够** ✗, 而是**压过头把自己顶飞** ✗
                    #   正确取值 ✓: 过闭合 ≈ 0.5mm (只需保证"接触后有正压力" ✓),
                    #     靠 `_GRIP_FRICTION`(4.0) 提供摩擦 ⇒ 既不飞也能夹住 ✓
                    #   若确需更大正压力 ⇒ 提高手指 PD 刚度 `_FINGER_PD_K`, 而不是加大过闭合 ✓
                    # ★★★★ 2026-09-20 【P0 真因】过闭合上限必须可调 (默认放宽到 8.0mm) ★★★★
                    #   实测铁证 (output/band2, oc65 ✓):
                    #     `z带内 728/775 顶点, 带=[9.6, 74.3]mm` ⇒ **手指是 37mm 高长指**,
                    #       覆盖物体整个高度 ⇒ "局部宽度=整体投影" ✓ ⇒ 跨度 76.0mm 可信 ✓
                    #     `PD_target=0.0208` ⇒ 模型指缝 = 30.4 + 2×20.8 = **72.0mm** < 76.0mm ✓
                    #       ⇒ 按理必须撞上 ✗ 但 `peak_force=0.00N` 且 `f1_qpos` **自由到目标位** ✗
                    #     ⇒ 唯一自洽解释 ✓: **真实指缝比模型大** ✓ ——
                    #       F0 实测两指 link 原点间距 138.7mm, 而模型 (q=0.05) = 130.4mm
                    #       ⇒ **真实 = 模型 + 8.3mm** ✓ 线性外推到终选 q:
                    #          真实指缝 ≈ 72.0 + 8.3 = **80.3mm > 76.0mm** ⇒ 悬空 4.3mm ⇒ 零力 ✓✓
                    #   修法 ✓: 按**真实间隙差 (≈8.3mm)** 反推所需额外过闭合 ≈ 4.15mm ⇒
                    #     总过闭合 ≈ 2.0 + 4.15 ≈ **6.2mm** (本旋钮 ✓)
                    #   为什么旧的 2.0mm 上限不再锁死: 那是按**模型嵌入量**定的 ✗,
                    #     而真实嵌入 = 模型过闭合 − 8.3/2×2 ≈ 模型 − 8.3 ⇒ 6.5mm 模型 ≈ 实入 0.7mm ✓ 安全
                    _oc_cap = float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MAX_MM", "2.0") or 2.0)
                    # ★ 复审第 4 条 ✓: 有冻结命令时**不得覆盖** ✗
                    _GRIP_TARGET = (_q_frozen if _q_frozen is not None else
                                    max(0.0, _q_contact - min(_oc_mm, _oc_cap) / 1000.0))
                    # ★★★★ 2026-09-20 【用户】"**夹爪应该开合有一个阈值范围**这种的, 需要正确考虑" ★★★★
                    #   物理事实 ✓: 手指关节 q ∈ [q_min, q_max] ⇒ **指缝 ∈ [GAP_MIN, GAP_MAX]** ✓
                    #     GAP = FINGER_GAP0 + 2q,  q ∈ [_GRIP_CLOSE, GRIPPER_MAX_OPEN] ✓
                    #   由此得到**物体尺寸可行域** (与位姿无关的硬边界 ✓):
                    #     · 跨度 > GAP_MAX ⇒ 手指**全开也包不住** ⇒ 夹持不可能 ✗ (只会推开)
                    #     · 跨度 < GAP_MIN ⇒ 手指**全闭也够不到** ⇒ 夹持不可能 ✗ (夹穿/挤走)
                    #     · 跨度 ∈ [GAP_MIN, GAP_MAX] ⇒ 可行 ✓, 所需闭合量
                    #         q_need = 跨度/2 − FINGER_EFFECTIVE_HALF_SPACING 必须落在 [q_min, q_max] ✓
                    #   旧代码 ✗: 只有 `max(_GRIP_CLOSE, ...)` 的**下截断** ✗,
                    #     **没有上界检查** ⇒ 物体会被"超出量程"地继续压 ✗ (或静默失效 ✗)
                    try:
                        from physics_utils import GRIPPER_MAX_OPEN as _GMO_e
                        _q_min_j, _q_max_j = float(_GRIP_CLOSE), float(_GMO_e)
                        _gap_lo_mm = (FINGER_GAP0 + 2.0 * _q_min_j) * 1000.0
                        _gap_hi_mm = (FINGER_GAP0 + 2.0 * _q_max_j) * 1000.0
                        _span_mm_e = ((_q_contact + FINGER_EFFECTIVE_HALF_SPACING)
                                      * 2.0) * 1000.0
                        _q_need = _q_contact
                        if _span_mm_e > _gap_hi_mm:
                            _verdict = "✗ 物体比**最大开度**还宽 ⇒ 夹持不可能"
                        elif _span_mm_e < _gap_lo_mm:
                            _verdict = "✗ 物体比**最小指缝**还窄 ⇒ 夹持不可能"
                        elif not (_q_min_j <= _q_need <= _q_max_j):
                            _verdict = "✗ 所需闭合量**超出量程** ⇒ 夹持不可能"
                        else:
                            _verdict = "✓ 在量程内"
                        logger.info(
                            "[Stage1 开合量程] 关节 q ∈ [%.4f, %.4f] ⇒ 指缝 ∈ [%.1f, %.1f]mm | "
                            "本物体跨度=%.1fmm, 所需 q_need=%.4f, 实际 PD_target=%.4f %s",
                            _q_min_j, _q_max_j, _gap_lo_mm, _gap_hi_mm,
                            _span_mm_e, _q_need, _GRIP_TARGET, _verdict)
                    except Exception as _ee_e:  # noqa: BLE001
                        logger.warning("[Stage1 开合量程] 检查失败: %s", _ee_e)
                    logger.info(
                        "[Stage1 闭合目标] ★以 **CoACD 真实跨度** 为基准: q_contact=%.4f → "
                        "PD_target=%.4f (过闭合 **%.1f**mm, 上限 %.1fmm; 旧逻辑会取 "
                        "bbox_min边=%.4f ✗ 差 %.2fmm) ⇒ 模型指缝=%.1fmm vs 物体跨度=%.1fmm",
                        _q_contact, _GRIP_TARGET, min(_oc_mm, _oc_cap), _oc_cap,
                        _q_contact_min,
                        (_q_contact - _q_contact_min) * 1000.0,
                        (FINGER_GAP0 + 2.0 * _GRIP_TARGET) * 1000.0,
                        (2.0 * _q_contact + 2.0 * FINGER_EFFECTIVE_HALF_SPACING) * 1000.0)
                else:
                    _q_contact = float(min(_cands)) if _cands else float(_computed_target)
                    # ★ 复审第 4 条 ✓: 有冻结命令时**不得覆盖** ✗
                    _GRIP_TARGET = (_q_frozen if _q_frozen is not None else
                                    max(0.0,
                                        _q_contact - self.adaptive_grip_overclosure(_q_contact)))
                    # ★ 过盈上限保护 (2026-09-20): 实测安全窗口 = 嵌入 (0, 2.0)mm ✓
                    #   (act < 0.0107 即嵌入 >2.0mm 时, CoACD 64 重叠凸包接触法向冲突
                    #     ⇒ 求解爆炸、物体被弹射 ✗ —— 见 `adaptive_grip_overclosure` 的实测记录 ✓)
                    # ★ 复审第 4 条 ✓: 有冻结命令时**不做这步抬升** ✗ (保持冻结值 ✓)
                    if _q_frozen is None:
                        _GRIP_TARGET = max(_GRIP_TARGET, 0.0, _q_contact - 0.0020)
                logger.info(
                    "[Stage1 闭合目标][对照估算] min边%.1fmm→%.4f | bbox_max%.1fmm→%.4f | "
                    "bbox投影%.1fmm→%s | **CoACD投影=%s** ⇒ 采用 q_contact=%.4f → "
                    "PD_target=%.4f (FINGER_GAP0=%.1fmm)",
                    _bbox_min_dim * 1000.0, _q_contact_min, _bbox_max * 1000.0, _q_contact_old,
                    (_proj_span * 1000.0) if _proj_span else -1.0,
                    f"{_q_contact_proj:.4f}" if _q_contact_proj is not None else "n/a",
                    f"{_q_contact_coacd:.4f}" if _q_contact_coacd is not None else "n/a",
                    _q_contact, _GRIP_TARGET, FINGER_GAP0 * 1000.0)
                # v51 §3.4 (2026-09-05): 闭合深度微调 Δ (环境变量测试旋钮) —
                # 过闭合仅 0.5mm 时挤压力不足 (小物体实测 1.08N 滑脱), 深压机理同 stage3 -0.004
                import os as _os
                # ★ 2026-09-22 【复审 I.3 修复】有**冻结命令**时**不得**再叠加偏移 ✗
                #   复审原话 ✓: "即使传入 `q_override`, 后续仍**无条件减** `_S1_GRIP_DELTA`;
                #     该旋钮非零时, **冻结的实际 q 可能再次被修改**" ✗
                #   ⇒ 本改: 只有**没有**冻结值时才应用该实验旋钮 ✓
                if _q_frozen is None:
                    _GRIP_TARGET -= float(_os.environ.get('_S1_GRIP_DELTA', '0'))
                    logger.info("[Stage1 闭合目标] 应用 `_S1_GRIP_DELTA=%s` ⇒ PD_target=%.4f",
                                _os.environ.get('_S1_GRIP_DELTA', '0'), _GRIP_TARGET)
                else:
                    logger.info("[Stage1 闭合目标] **使用冻结命令** q=%.4f ⇒ 不叠加 "
                                "`_S1_GRIP_DELTA` ✓ (验证动作 = 交付动作 ✓ 复审 I.3 ✓)",
                                float(_q_frozen))
                logger.info(f"[Stage1] 目标物体 {_bbox_max*100:.2f}cm, "
                            f"接触 qpos={_q_contact:.4f}, compute_gripper_qpos={_computed_target:.4f}, "
                            f"PD_target={_GRIP_TARGET:.4f} (基于物体尺寸, 略小于接触点产生正压力), "
                            f"enclose_z={_enclose_z:.4f} (v5.1 无下降段, z 在 enclose±1cm)")

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【复审 I.3 修复】**最后一道锁**: 冻结命令必须在仿真前生效 ✓ ★★★★
        #   复审原话 ✓: "只接受**一次**'把实验旋钮转换成命令'的过程, **执行器不再重算/叠加偏移**" ✓
        #   ⇒ 本处: 若本次是"验证/回放"调用 (`q_override` 非 None ✓) ⇒
        #     **无条件**把闭合目标设为冻结值 ✓ (在仿真开始前的**最后一步** ⇒ 不可能再被改 ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if _q_frozen is not None:
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【根因修复】冻结命令**不得与真实跨度矛盾** ✗ ★★★★
            #   实测铁证 ✓ (meas2 日志, 同一函数内相邻两行):
            #     `[Stage1 闭合目标] ★CoACD 跨度=**47.8mm** (516/516 顶点) → q_contact=**0.0087**` ✓
            #     `[Stage1 闭合目标] **使用冻结命令** q=**0.0027**` ✗ ← 本行把它覆盖了
            #   ⇒ 指缝只到 30.4+2×2.7 = **35.8mm**, 而物体跨度 **47.8mm**
            #     ⇒ **每侧压入 (47.8−35.8)/2 = 6.0mm** ✗✗ ⇒ 这不是"夹住", 是**挤压**
            #     ⇒ 物体会被推走/翻转 ⇒ 正是用户说的"碰到了, 但物体变化比较厉害" ✓✓
            #   本改 ✓: 冻结值与"**按当前真实跨度算出的目标**"相差 > `_S1_FROZEN_Q_TOL_MM`
            #     (默认 **1.0mm** ✓) ⇒ **以真实跨度为准** ✓ 并 **ERROR 报出** ✗
            #     理由 ✓: 跨度是**物理真值** ✓; 旧 recipe 的 q 是在**旧物理/旧跨度**下冻结的 ✗
            #     ⇒ 冻结语义变为"**除闭合量外**全部复用" ✓ (闭合量必须随真实几何走 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _qf = float(_q_frozen)
            _span_target = None
            try:
                if '_q_contact_coacd' in locals() and _q_contact_coacd is not None:
                    _span_target = max(
                        0.0, float(_q_contact_coacd)
                        - float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MM", "0.5") or 0.5) / 1000.0)
            except Exception:  # noqa: BLE001
                _span_target = None
            _tol_mm = float(os.environ.get("_S1_FROZEN_Q_TOL_MM", "1.0") or 1.0)
            if (_span_target is not None
                    and abs(_qf - _span_target) * 1000.0 > _tol_mm):
                logger.error(
                    "[闭合量根因修复] ✗ 冻结 q=%.4f 与**真实跨度基准** q=%.4f 相差 **%.1fmm** "
                    "(> %.1fmm) ⇒ **以真实跨度为准** ✓ | 指缝 %.1fmm→**%.1fmm** "
                    "(消除每侧 **%.1fmm** 的过压 ✗ ⇒ 由'挤压'变'刚好贴住' ✓)",
                    _qf, _span_target, abs(_qf - _span_target) * 1000.0, _tol_mm,
                    (FINGER_GAP0 + 2.0 * _qf) * 1000.0,
                    (FINGER_GAP0 + 2.0 * _span_target) * 1000.0,
                    max(0.0, (_qf - _span_target)) * 1000.0)
                _GRIP_TARGET = float(_span_target)
            else:
                _GRIP_TARGET = _qf

        # === 3. 仿真 ===
        contact_count = 0
        both_contact_count = 0
        # ★ md §03 遗留 (2026-09-30) ✓: **真载荷探针**累积器 (不再只有"接触帧占比"代理 ✗)
        _v2_load_n = 0                     # 统计帧数
        _v2_load_frames = 0                # 双指法向冲量**均 > 0** 的帧数 ✓
        _v2_load_min_imp = float("inf")    # 双侧冲量最小值 (最弱时刻 ✓)
        # ★ 2026-09-30 ✓: **实际生效的闭合值**(T1b 钳位**之后**)
        _s1_frozen_eff = None
        # ★ md §03 遗留 ✓: **固定 C0** 的 6DOF 相对偏差累积器 (抬升段 ✓)
        _v2_C0 = None                      # C0 = 抬升段首帧夹爪基座位姿 (显式记录 ✓)
        _v2_c0_t = 0.0                     # 相对平移最大值 (mm)
        _v2_c0_r = 0.0                     # 相对旋转最大值 (deg)
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
        _CLOSE_FRAMES = int(os.environ.get("_S1_CLOSE_FRAMES", STAGE1_CLOSE_CONTROLS))
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-23 【计划 §6 / 用户原则】先**就位**再闭合 ✓ ★★★★
        #   实测铁证 ✓ (§131): 同一 anchor
        #     · A (Stage1 rollout): **F0 就闭合**, 而基座**还在赶往 anchor** ✗
        #       ⇒ 手指在途中就闭完了 ⇒ 到达时已经合拢 ⇒ 形不成对称捏持 ⇒ 真夹持率 **0.11** ✗
        #     · B (Stage2 接近): 基座**就位后**才闭合 ✓ ⇒ 真夹持率 **100%** ✓
        #       (且两指都接触 glb_4 ✓ n=1166/975 ✓)
        #   ⇒ 本改 ✓: 闭合前插入 `_TRAVEL_FRAMES` 帧"**保持张开、等基座到位**" ✓
        #     默认 **0 = 关闭** ✓ (不改既有默认行为 ✗); 实验时设 15~25 ✓
        #   ⚠ 副作用 ✓: 抬升段会**后移** `_TRAVEL_FRAMES` 帧 ⇒ 已在下方同步
        #     `_LIFT_START_FRAME` ✓ (否则"抬升段统计窗"会错位 ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _TRAVEL_FRAMES = int(os.environ.get("_S1_TRAVEL_FRAMES", "0") or 0)
        _SETTLE_FRAMES = int(os.environ.get("_S1_SETTLE_FRAMES", STAGE1_SETTLE_CONTROLS))
        # ★★★★ 2026-09-20 【P0 时序修正】闭合/稳定必须**在接触高度上完成**, 再开始抬 ✗ ★★★★
        #   真值铁证 (output/fix76, 过闭合 10mm ⇒ 模型指缝 56.0mm ≪ 物体跨度 76.0mm) ✓:
        #     `F0 目标顶点到指1=29.18mm` → `F80 = 51.44mm` ✗ (**越闭越远** ✗)
        #     而同期 `pos_z: 0.0380 (F0) → 0.1982 (F80)` ⇒ **基座在"闭合段"就抬了 16cm** ✗✗
        #     ⇒ 手指是在**物体上方合拢**的 ✗ ⇒ 物体矮 (76mm) 时根本碰不到 ✓
        #       (并解释为什么放大的 116.8mm 物体反而 2/3 通过 ✓ —— 它够高, F0 就 1.56mm ✓)
        #     ⇒ 根因不是"闭合量"、不是"指缝模型"、不是"对中" ✗, 而是**时序**: 抬升开始得太早 ✓
        #   修法 ✓: 把闭合+稳定帧数**可调**并默认加长 (16+10=26 → 30+20=50 ✓),
        #     保证"闭合到位并保持"之后才开始竖直抬升 ✓ (无下降段的模板保持不变 ✓)
        # ★ 2026-09-23 ✓: 抬升起点必须**加上就位等待帧** `_TRAVEL_FRAMES` ✓
        #   (否则"抬升段统计窗"整体**前移** ⇒ 真夹持率/离桌判定都会看错帧 ✗)
        _LIFT_START_FRAME = _TRAVEL_FRAMES + _CLOSE_FRAMES + _SETTLE_FRAMES
        logger.info("[Stage1 时序] 闭合帧=%d, 稳定帧=%d ⇒ **抬升从 F%d 开始** "
                    "(旧默认 16+10=26; 真值显示 F80 已抬高 16cm ⇒ 必须延后 ✓)",
                    _CLOSE_FRAMES, _SETTLE_FRAMES, _LIFT_START_FRAME)
        _obj_p0 = None      # ★ 真值探针用: 物心起始位置 (引擎真值, 与仿真同源 ✓)
        # ★ 真值对中状态: 每次 rollout 必须清零 (否则上一轮的偏心会污染本轮罚分 ✗)
        self._last_rollout_asym_mm = None
        self._last_rollout_dmin_mm = None
        # ★ 指尖离桌面真值也必须清零 (与对中同理 ✓; None = 未启用桌面碰撞/取不到 ✓)
        self._last_rollout_fgnd_mm = None
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
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-25 【数据流修正】计划侧 CoACD 真值**必须**传给执行器 ✗ ★★★★
        #   用户定调 ✓: "你得把这个数据传输给搞清楚 … 读取数据、仿真和计划的数据之间的差异"
        #   实测铁证 ✓ (cc26/cc27, 同一候选内三个值互相对不上):
        #     计划侧 `[CoACD质量]`: 闭合轴跨度 51.6mm ⇒ **指缝(闭合)=52.6mm** ⇒ q≈**11.1mm** ✓
        #     执行侧 `[准确性校验]`: F220/F225 命令 q=**0.0195** ⇒ 指缝=30.4+39=**69.4mm** ✗
        #     真值侧 `[真值对中]`: 最近侧=**19~23mm**, asym=**20~53mm** ✗ (手指停在半空 ✓)
        #   ⇒ 差 **16.8mm** ⇒ 手指**从未真正压上物体** ⇒ 无夹持力(force=0.00N ✗) ⇒ 抬不起 ✗
        #   ⇒ 病根: 执行器用的是 **bbox/投影估计**(≈0.018~0.020 ✗), 而 CoACD 真值(≈11.1mm ✓)
        #     没有被用于闭合序列 ⇒ 正是方案 §11.2 / §8 说的"**计划与执行两个值**" ✗
        #   本改 ✓: 以 `_q_contact_coacd`(**CoACD 真值** ✓ 平移/漂移不变 ✓) 减过闭合量
        #     为**唯一基准** ✓; 相差 >0.2mm ⇒ **ERROR 明确报出**并改用真值 ✓
        #     (只让闭合量随真实几何走 ✓; 其余全不动 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            if '_q_contact_coacd' in locals() and _q_contact_coacd is not None:
                _ov_m = float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MM", "0.5") or 0.5) / 1000.0
                _tgt_true = max(0.0, float(_q_contact_coacd) - _ov_m)
                _diff_mm = abs(float(_GRIP_TARGET) - _tgt_true) * 1000.0
                if _diff_mm > 0.2:
                    logger.error(
                        "[数据流修正] ✗ 执行器闭合目标 %.4f(**指缝 %.1fmm**) 与 "
                        "**CoACD 真值基准** %.4f(**指缝 %.1fmm**) 相差 **%.1fmm** ⇒ "
                        "**以 CoACD 真值为准** ✓ (计划与执行必须同一个值 ✓)",
                        float(_GRIP_TARGET), (FINGER_GAP0 + 2.0 * float(_GRIP_TARGET)) * 1000.0,
                        _tgt_true, (FINGER_GAP0 + 2.0 * _tgt_true) * 1000.0, _diff_mm)
                    _GRIP_TARGET = float(_tgt_true)
                else:
                    logger.info("[数据流校验] ✓ 计划=执行 (闭合目标 %.4f, 指缝 %.1fmm, 差 %.2fmm)",
                                float(_GRIP_TARGET),
                                (FINGER_GAP0 + 2.0 * float(_GRIP_TARGET)) * 1000.0, _diff_mm)
            else:
                logger.warning("[数据流校验] ⚠ CoACD 真值不可用 ⇒ 闭合目标沿用 %.4f (指缝 %.1fmm) "
                               "⇒ 若手指停半空, 优先查此处 ✗",
                               float(_GRIP_TARGET),
                               (FINGER_GAP0 + 2.0 * float(_GRIP_TARGET)) * 1000.0)
        except Exception as _edf:  # noqa: BLE001
            logger.warning("[数据流修正] 失败(沿用原值 ✓): %s", _edf)
        _close1_q = stage1_close_q_sequence(stage_q_open(), _GRIP_TARGET, _CLOSE_FRAMES)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-23 【复核指令 · 受控闭合】给闭合目标加**速率限制** ✗ ★★★★
        #   复审原话 ✓: "保持同一套 PD 参数 … **对闭合目标增加速度限制, 避免目标位置快速越过
        #     接触位置**" ✓
        #   为什么必要 ✓: 现在是**开环位置序列** ✗ —— 目标位置可以一步跨过接触点 ⇒
        #     0.33s 内压入设计深度 ⇒ 冲击/挤飞 ✓ (与"有界建立夹持"相冲突 ✗)
        #   本改 ✓: 逐帧限幅 `|Δq| ≤ _S1_CLOSE_RATE_MAX` (默认 **0.0015 m/帧** ✓ 可配 ✓)
        #     ⇒ 目标**渐进**逼近接触位 ✓; 配合"过闭合 ≤1mm"(=`_S1_CLOSE_OVERCLOSURE_MM` ✓)
        #       构成"**有界预紧**" ✓ (碰到后不再无界加深 ✓)
        #   ⚠ 与 Stage2 的一致性 ✓: 同一序列由 recipe 导出 ✓ (`close_ctrl_q` ✓) ⇒
        #     验证与重放**用同一条闭合曲线** ✓ (不会"Stage1 用反馈、Stage2 只读一个旧 q" ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _dq_max = float(os.environ.get("_S1_CLOSE_RATE_MAX", "0.0015") or 0.0)
            if _dq_max > 0.0 and len(_close1_q) > 1:
                _seq = np.asarray(_close1_q, dtype=np.float64).copy()
                _hit = 0
                for _i2 in range(1, len(_seq)):
                    _lo2, _hi2 = _seq[_i2 - 1] - _dq_max, _seq[_i2 - 1] + _dq_max
                    _new = float(np.clip(_seq[_i2], _lo2, _hi2))
                    if abs(_new - _seq[_i2]) > 1e-12:
                        _hit += 1
                    _seq[_i2] = _new
                _close1_q = _seq
                logger.info("[受控闭合] ✓ 闭合目标**速率限制** ±%.4f m/帧 (限幅 %d 帧; "
                            "防止目标位置快速越过接触点 ✓); 有界预紧 = 过闭合 %.2fmm ✓",
                            _dq_max, _hit,
                            float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MM", "0") or 0.0))
        except Exception as _erl:  # noqa: BLE001
            logger.warning("[受控闭合] 速率限制失败: %s", _erl)
        # 供 Stage1 artifact 导出 recipe 时校验"实际命令的 q_grasp"
        self._last_grip_target_s1 = float(_GRIP_TARGET)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 任务03 (2026-09-28) 【唯一命令状态: 上一周期**实际下发** q】 ✓ ★★★★
        #   md §03 原话 ✓: "闭合/预紧/稳定/提升**共用上一周期实际下发 q**;
        #     清理 `_q_ramp` 与 `_q_cmd_cont` **各走各的状态**; freeze 不能让后续相位隐式留在 CLOSE" ✓
        #   实测旧病 ✗: 三处命令量各自初始化 ——
        #     · `_q_cmd_cont` 起点 = 闭合序列末点 (2452)
        #     · `_q_ramp`   起点 = `stage_q_open()` (3057) ⇒ **不是**上一周期实际值 ✗
        #     · 导出用 `_last_grip_target_s1` 在 2387 被**硬写成目标** `_GRIP_TARGET` ✗
        #       ⇒ 导出侧把"目标"当"实际下发值"用 ✗ (语义漂移)
        #   ⇒ 本改 ✓: 只保留**一个**真源 `_s1_q_cmd_state` (= 上一周期实际下发的夹爪 PD 目标),
        #     上面三个量都从它取/写回 ✓ ⇒ 相位边界不再跳变, 跟踪误差可归因 ✓
        #     (初值 = 张开 `stage_q_open()` ✓ 与物理初态一致)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        self._s1_q_cmd_state = float(stage_q_open())

        # ★ 2026-09-14 P0: 录像写帧计数器 (用 list 便于闭包内自增; 每帧循环前重置)
        _s1_video_frames = [0]
        # ★★★ 2026-09-14 用户: "每次测试完夹爪的位姿和开合都得保留一个 log 或 csv 文件"
        #   目的: 用户看视频时发现"开始前几帧和最后几帧夹爪又向上了" ✗ —— 需要逐帧数值佐证 ✓
        #   记录内容: 帧号 / 夹爪目标位置 / 夹爪 ZYX 欧拉(含"朝下"判据) / 两指命令与实际 q /
        #             物体 z / 双侧接触 / 夹持力 / 夹爪-物距离
        _s1_trace_rows = []
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ S2-1 (2026-09-27) 【S1-D 冻结 Stage2 输入】每个 rollout 起点**重置** ✓ ★★★★
        #   方案 §S1-D ✓: recipe 必须保存"**该获胜 run 的实际执行数组**",
        #     "**不能取最后一次仿真缓存**, 更不能只存 q 终点后让 Stage2 重生成另一条曲线" ✓
        #   实测缺陷 ✓ (s12 产物): `q_sequence_source=**regenerated_nominal**` ✗
        #     ⇒ 因为 `_last_close_q_executed` **只 append 从不重置** ⇒ 跨 rollout 串味,
        #        且调用点**根本没传** `close_q_actual` ⇒ 只能重生成 ✗
        #   ⇒ 本改 ✓: rollout 起点清空 (q 序列 + 时标/基座命令) ⇒ 获胜 run 的就是它自己的 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ S2-1b ✓ **先存档上一条** ⇒ 导出时才能按 winner 的 q 终点取回"它自己"的曲线 ✗
        #   (实测 s15: 导出发生在**另一段作用域**, 那时 `_last_close_q_executed` 已被
        #    后续 rollout 重置 ⇒ 拿到 **0 帧** ⇒ 又退回 regenerated_nominal ✗)
        try:
            _prev_q = list(getattr(self, "_last_close_q_executed", None) or [])
            if _prev_q:
                _hist = getattr(self, "_s1_exec_q_by_rollout", None) or []
                _tr_p = getattr(self, "_s1_exec_trace", None) or {}
                _hist.append({"q": _prev_q,
                              "t": list(_tr_p.get("t") or []),
                              "base_pos": list(_tr_p.get("base_pos") or [])})
                self._s1_exec_q_by_rollout = _hist[-64:]      # 有界 ✓ (只留最近 64 条)
        except Exception:  # noqa: BLE001
            pass
        # ★ T072 ✓ 开启**驱动层下发 spy** (独立于 trace ⇒ 可证明"记的是限速后真值" ✓)
        try:
            from physics_env import set_dispatch_spy as _sds072
            from physics_env import set_step_tag as _sst8b
            _sds072(True)
            _sst8b("stage1")        # ★ T08-B ✓ 标注调用方 ⇒ 供"同入口"证明 ✓
        except Exception as _e072a:  # noqa: BLE001
            logger.warning("[T072] spy 开启失败: %s", _e072a)
        self._last_close_q_executed = []
        # ★ T07c (2026-09-27) ✓ 完整执行记录容器 (唯一记录点 = 命令下发处 ✓)
        self._s1_exec_trace = {
            "control_index": [], "t_s": [], "T_W_G": [], "q_cmd": [],
            "phase": [], "source_phase": [], "q_act": [], "obj_z": [], "valid": [],
            "base_pos": [], "q": [], "t": [],      # 兼容旧字段名 (S2-1 口径 ✓)
        }

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-23 【方案 §4.2-7 / §1.2 实施】相位边界 q 必须**连续** ✗ ★★★★
        #   方案原话 ✓: \"将实际生成的完整 `q[t]`、时标与状态边界冻结,
        #     **CLOSE/SETTLE/提升之间最后一个 q 连续**\" ✓
        #   方案实测的缺陷 ✓ (§10): `close=16`、每帧 Δq≤1.5mm、最终 q=3.3mm 时
        #     `q_open=30mm → 限速末点 9.2232mm → SETTLE 3.3mm`: **跳变 −5.9232mm** ✗;
        #     `q_open=50mm` ⇒ **−25.1050mm** ✗✗
        #   病根 ✓: 闭合段是**限速**序列 (`stage1_close_q_sequence` ✓), 它的末点
        #     **到不了** `_GRIP_TARGET` ✗; 而 SETTLE 段直接令 `gripper_target = _GRIP_TARGET`
        #     ⇒ **一步大跳** ✗ (正是方案 §1.2 表里\"新限速跨阶段失效\"✗)
        #   本改 ✓: 用一个**跨相位连续**的命令量 `_q_cmd_cont` ✓:
        #     · 初值 = 闭合序列的**末点** ✓ (不跳 ✓)
        #     · 每帧只允许推进 ≤ `_S1_CLOSE_RATE_MAX` ✓ ⇒ **任意相位边界处连续** ✓
        #     · 到达 `_GRIP_TARGET` 后自然保持 ✓ (与旧行为终点一致 ✓)
        #   ⚠ 这**改变实际动作** ⇒ 已计入执行器语义 ✓ (方案 §3.1 要求 ✓): recipe 侧
        #     以 `_last_grip_target_s1` = 实际下发值为准 ✓ (方案 §8 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 任务03 ✓: 起点取**唯一命令状态** (上一周期实际下发 q); 无 ⇒ 回落闭合末点
        _q_cmd_cont = float(getattr(self, "_s1_q_cmd_state", None)
                            or (float(np.asarray(_close1_q).reshape(-1)[-1])
                                if np.asarray(_close1_q).size else float(_GRIP_TARGET)))
        _q_rate_max_c = float(os.environ.get("_S1_CLOSE_RATE_MAX", "0.0015") or 0.0015)
        # ★ 方案 §11.3 证据 ✓: 逐帧记录"命令变化 vs 实际变化" ⇒ 区分轻触/真被挡停 ✓
        try:
            from physics_env import set_substep_ground_evidence as _set_ground_ev
        except Exception:  # noqa: BLE001
            _set_ground_ev = None
        self._ev_prev_q = None
        self._ev_prev_cmd = None
        logger.info("[方案 §4.2-7] 相位连续命令: 闭合末点 q=%.4f ⇒ 以 ≤%.4f/帧 "
                    "继续推进到 %.4f ✓ (消除相位边界跳变 ✗)",
                    _q_cmd_cont, _q_rate_max_c, float(_GRIP_TARGET))

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ W4 (2026-09-26) 【通用几何对中】沿**闭合轴**把夹爪中心对齐到物体中心 ✓ ★★★★
        #   为什么必须加 ✓ (自查后确认, 不是读取错误):
        #     · GJK 校准测试 4 passed ✓ ⇒ `asym` 量法可信 ✓
        #     · 探针每子步调用 ✓, both_force_steps 写/读/清零一致 ✓ ⇒ "双侧受力帧=0"是**真的** ✓
        #     · Stage0 的 13.1cm 前偏移已由 neutral offsets 补平(实测 dz≈5mm) ⇒ 不是偏移没接上 ✓
        #     ⇒ 于是剩下的是**真的对中偏差**: 实测 `asym=6.5~7.0mm` ⇒ 单侧先碰到 ⇒
        #       一挤压就把物体**挤射出去**(实测 |Δxy| 达 1.8~2.4m ✗) ⇒ 永远抓不到 ✓
        #   本改 ✓ (**纯几何、与物体无关 ⇒ 可泛化** ✓):
        #     · 闭合轴 u = 归一化的两指连线 (f2−f1) ✓ (就是开合方向 ✓ 不依赖 URDF 约定)
        #     · 夹爪中心 = (f1+f2)/2 ; 物体中心 = CoACD 凸包顶点均值 ✓
        #     · 只沿 u **平移** `target_pos` 使两者重合 ⇒ **不改变高度/姿态** ✓
        #       (⇒ 不会引入新的插桌风险 ✓, 也不靠调力/调摩擦 ✓)
        #   ⚠ 只读几何 + 一次性平移 ⇒ 不改判据、不改奖励、不动物理 ✓
        #   验收 ✓: `asym` 应降到 ≤2mm, 且随后**双侧受力帧 > 0** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ 方案 §10.4 合规修正 ✗: 本对中**必须撤出默认路径** ✓
        #   方案原话 ✓: "当前 `_S1_AUTO_CENTER` 默认开启、在 rollout 内部静默改 `target_pos`,
        #     与'不得无条件重新物心对齐'的约束不一致 ⇒ **下一实现应将它从默认路径撤出,
        #     作为固定候选的可开关诊断提案**; 不得在冻结复验/Stage2 重放时再自动对中" ✓
        #   ⇒ 本改: 默认 **0 = 关闭** ✓ (仅在固定候选诊断时显式开启 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ G4b / T5 (2026-09-26) 【B 组: 一次有界横向修正】按方案 §10.4 ✓ ★★★★
        #   方案要求 ✓ (逐条对应):
        #     ① 闭合轴 **优先从关节轴经 FK 变换**获得 ✗ 不用"两指 link 连线" ✗
        #        (连线不必等于真实闭合方向 ✗)
        #     ② 用**离线小 q 扰动验证两指沿该轴反向运动** ✓ ⇒ 本实现直接校验两指关节轴**反向** ✓
        #     ③ 有向间隙 `gL = o_min − l`, `gR = r − o_max` ⇒ `δ = (gL − gR)/2` ✓
        #     ④ **有界** ✓: `|δ| > 3mm` ⇒ 报 **BOUND_EXCEEDED** 并**不修正** ✗ (不默默扩大 ✓)
        #     ⑤ 保存**修正量 + 实际 anchor** ✓ (写入快照 ✓)
        #     ⑥ 默认**关闭** ✓ (仅 B 组实验显式开启 ⇒ A/B 可对照 ✓)
        #     ⑦ 冻结复验/Stage2 重放**不得**自动对中 ✓ (该分支只在本 rollout 内生效 ✓)
        #   实测依据 ✓ (output/s2 快照): CLOSE_END 左 gap=**2.49mm** / 右 gap=**7.31mm**
        #     ⇒ δ = (2.49 − 7.31)/2 = **−2.41mm** ✓ 在预算内 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        self._s1_center_shift_mm = 0.0
        self._s1_center_axis = None
        self._s1_center_status = "OFF"
        if str(os.environ.get("_S1_AUTO_CENTER", "0")) == "1":
            try:
                def _quat_to_R(_q):
                    """(w,x,y,z) → 3x3 ✓ (就地实现, 不依赖外部工具 ✓)"""
                    _w, _x, _y, _z = (float(_q[0]), float(_q[1]),
                                      float(_q[2]), float(_q[3]))
                    return np.array([
                        [1 - 2 * (_y * _y + _z * _z), 2 * (_x * _y - _w * _z), 2 * (_x * _z + _w * _y)],
                        [2 * (_x * _y + _w * _z), 1 - 2 * (_x * _x + _z * _z), 2 * (_y * _z - _w * _x)],
                        [2 * (_x * _z - _w * _y), 2 * (_y * _z + _w * _x), 1 - 2 * (_x * _x + _y * _y)],
                    ], dtype=np.float64)

                from gap_utils import convex_gap_min_mm as _gmin_c
                _lc = {}
                for _l_c in robot.get_links():
                    _n_c = _l_c.get_name()
                    if _n_c in (f"{side}_gripper_finger_link1",
                                f"{side}_gripper_finger_link2"):
                        _lc[_n_c] = _l_c
                if len(_lc) == 2:
                    _l1c = _lc[f"{side}_gripper_finger_link1"]
                    _l2c = _lc[f"{side}_gripper_finger_link2"]
                    # ★ 修正 ✓: 两指的**局部轴符号可以不同** ✗ (我给两指用了同一符号 ⇒ dot=+1 ✗)
                    #   ⇒ 在 ±1 四种组合里取**最反向**的一对 ✓ (= "两指沿该轴反向运动"的校验 ✓)
                    _R1c = _quat_to_R(_l1c.get_pose().q)
                    _R2c = _quat_to_R(_l2c.get_pose().q)
                    _ax0 = np.array([0.0, 1.0, 0.0])             # 关节轴方向(局部) ✓
                    _best_ax = None
                    for _s1c in (1.0, -1.0):
                        for _s2c in (1.0, -1.0):
                            _a1 = _R1c @ (_s1c * _ax0)
                            _a2 = _R2c @ (_s2c * _ax0)
                            _dd = float(np.dot(_a1, _a2))
                            if _best_ax is None or _dd < _best_ax[0]:
                                _best_ax = (_dd, _a1, _a2)
                    _opp, _u1c, _u2c = _best_ax
                    _n1c = float(np.linalg.norm(_u1c))
                    _n2c = float(np.linalg.norm(_u2c))
                    # ① "小 q 扰动验证"等价校验 ✓: 两指关节轴必须**反向** ✗ 否则不修 ✓
                    if _opp > -0.9:
                        self._s1_center_status = f"AXIS_NOT_OPPOSED(dot={_opp:.3f})"
                        logger.error("[T5 有界修正] ✗ 两指关节轴未反向(dot=%.3f) ⇒ **不修正** ✓",
                                     _opp)
                    else:
                        _hv_c = self._coacd_world_hulls(target_name) or []
                        _vv_c = self._finger_world_vertices(robot) or []
                        if _hv_c and len(_vv_c) == 2:
                            _gL, _, _ = _gmin_c(_hv_c, _vv_c[0])
                            _gR, _, _ = _gmin_c(_hv_c, _vv_c[1])
                            _delta_mm = 0.5 * (float(_gL) - float(_gR))    # δ=(gL−gR)/2 ✓
                            _lim_mm = float(os.environ.get("_S1_CENTER_LIMIT_MM", "3.0") or 3.0)
                            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                            # ★★★★ T7 (2026-09-26) 【测量时刻改到闭合到位后】 ★★★★
                            #   T6 实测铁证 ✓ (s9 `[几何快照]`): `LIFT+15 左gap=0.00mm 右gap=5.41mm`,
                            #     `冲量 指1=2.63 指2=0.000` ⇒ 闭合后**只有单侧贴上** ⇒ 非夹持 ⇒ 抬升即滑落 ✗
                            #   而张开态测到的 δ=**1.93mm**(已修正) ⇒ 闭合后仍差 5.41mm ✗
                            #     ⇒ 张开→闭合过程中几何会变, **张开态测的 δ 不足以代表闭合后的偏心** ✗
                            #   ⇒ 本改 ✓: 张开态只**记录**(不做动作) ⇒ 修正推迟到 CLOSE_END 用同款 GJK 重量 ✓
                            #     (仍 ±3mm 有界 ✓ 仍**只修正一次** ✓, 见循环内 T7 块 ✓)
                            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                            _c_meas_mode = str(os.environ.get("_S1_CENTER_MEASURE", "close"))
                            if _c_meas_mode == "close":
                                self._t7_delta_open_mm = float(_delta_mm)
                                self._s1_center_status = f"DEFERRED_TO_CLOSE(delta_open={_delta_mm:+.2f}mm)"
                                logger.warning(
                                    "[T7 居中推迟] 张开态 δ=%+.2fmm (gL=%.2f gR=%.2f) **仅记录不修正** ✓ "
                                    "⇒ 改为**闭合到位后**测量并修正一次 ✓ (实测张开态修不够: 闭合后仍差 5.41mm ✗)",
                                    _delta_mm, float(_gL), float(_gR))
                            elif abs(_delta_mm) < 0.05:
                                self._s1_center_status = "ALREADY_CENTERED"
                            elif abs(_delta_mm) > _lim_mm:
                                # ② **越界 ⇒ 不修正** ✓ (方案: 报 BOUND_EXCEEDED, 不默默扩大 ✗)
                                self._s1_center_status = (
                                    f"BOUND_EXCEEDED(delta={_delta_mm:+.2f}mm > {_lim_mm:.2f}mm)")
                                logger.error(
                                    "[T5 有界修正] ✗ **BOUND_EXCEEDED** δ=%+.2fmm 超 %.2fmm 预算 "
                                    "⇒ **不修正** ✓ (不得自动继续叠偏移 ✗)",
                                    _delta_mm, _lim_mm)
                            else:
                                _u_hat = _u1c / _n1c
                                _mid_c = (np.asarray(_l2c.get_pose().p, dtype=np.float64)
                                          - np.asarray(_l1c.get_pose().p, dtype=np.float64))
                                if float(np.dot(_u_hat, _mid_c)) < 0.0:
                                    _u_hat = -_u_hat          # 方向: 指向"从指1到指2" ✓
                                # ★ 符号按**实测**定 ✓: 沿 `u`(由 FK 轴给出)平移 `+1.93mm` 时
                                #   实测 左 gap 2.49→**0.60** ✓ 但右 gap 7.31→**9.15** ✗
                                #   ⇒ 说明 `u` 的实际指向与"指1→指2"**相反** ✓
                                #   ⇒ 取 `s = +δ`(δ=(gL−gR)/2) ⇒ 使**大间隙侧收窄** ✓ 趋于相等 ✓
                                _s_mm = float(_delta_mm)
                                target_pos = (np.asarray(target_pos, dtype=np.float64)
                                              + (_s_mm / 1000.0) * _u_hat)
                                self._s1_center_shift_mm = float(_s_mm)
                                self._s1_center_axis = [float(x) for x in _u_hat]
                                self._s1_center_status = "APPLIED"
                                logger.warning(
                                    "[T5 有界修正] ✓ 沿**关节轴(FK)**平移 **%+.2fmm** | "
                                    "gL=%.2fmm gR=%.2fmm (δ=%+.2fmm ≤ %.2fmm ✓) | "
                                    "轴=(%.3f,%.3f,%.3f) 两指轴反向校验 dot=%.3f ✓ | "
                                    "新 anchor=%s",
                                    _s_mm, float(_gL), float(_gR), _delta_mm, _lim_mm,
                                    float(_u_hat[0]), float(_u_hat[1]), float(_u_hat[2]), _opp,
                                    np.asarray(target_pos).round(4).tolist())
            except Exception as _e_w4:  # noqa: BLE001
                self._s1_center_status = f"ERROR:{_e_w4}"
                logger.warning("[T5 有界修正] 失败(按原位姿执行 ✓): %s", _e_w4)

        _t7_center_done = False      # ★ T7 ✓: 保证\"闭合到位后\"只修正一次 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T8 (2026-09-26) 【按"法向力"闭合】状态量 (替代"按几何闭合" ✓) ★★★★
        #   依据 ✓ (T6/T7 实测): CLOSE_END 几何 gap=0.00/0.00(已居中 ✓) 但抬升 15 帧后
        #     **右 gap=5.37mm** 且 **指2 冲量恒 0.000** ✗ ⇒ 右侧无真实法向力
        #     ⇒ 物体绕左侧支点转动 ⇒ 滑落 ✗ (SLIP / UNILATERAL)
        #   ⇒ 本改 ✓: 闭合段**继续压入**直到 **两侧都有真实法向冲量**(`sum_imp_raw>0` ✓
        #      §5.1 标定口径, 不做阈值 ✓) 或 达上限 `计划 − _S1_T8_EXTRA_MM` ✓
        #   ⚠ 有界: 额外 ≤2mm(q) ⇒ 指缝再收 4mm ✓; 步长 0.05mm/帧 ✓ 温和不冲击 ✓
        #   ⚠ 只改"闭合何时结束" ✓: 不新增门 / 不改奖励 ✓; 抬升**相应推迟** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _t8_ok_bil = False          # 上一帧是否"两侧都有真实法向冲量" ✓
        _t8_ext_used = 0            # 已用的额外帧数 ⇒ 抬升推迟同样帧数 ✓
        _t8_press_m = 0.0           # 已额外压入 (m, q 口径 ✓)
        _t8_extra_m = float(os.environ.get("_S1_T8_EXTRA_MM", "2.0") or 2.0) / 1000.0
        _t8_step_m = float(os.environ.get("_S1_T8_STEP_MM", "0.05") or 0.05) / 1000.0
        _t8_max_frames = int(np.ceil(_t8_extra_m / max(1e-9, _t8_step_m)))

        for frame in range(n_frames):
            qpos = robot.get_qpos().copy()

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T2c (2026-09-25) 物体 z 必须**逐帧采集** ✗ ★★★★
            #   实测铁证 ✓ (output/v7/stage1/gripper_trace.csv): `obj_z` 列 **300 帧全是 `nan`** ✗
            #     且 `obj_lift=0.00 / obj_lift_peak=0.00` ✗ ⇒ `抬起=False` ✗ —— 但同轮 `真夹持率
            #     =**100%**` ✓ `挡停量 2.98/3.98mm` ✓ ⇒ **物体其实被夹住了** ✓
            #   病根 ✓: 物体 z 原来只在**单个测量帧**(`_LIFT_START_FRAME-1`) 被 append ✗
            #     ⇒ `obj_z_history` 只有 1 个样本 ⇒ `final − initial ≡ 0` ⇒ **抬升量恒 0** ✗
            #     ⇒ 判据把"已夹住并抬起"误判成"没抬起" ✗ (与"读取数据不正确"是同一类错误 ✓)
            #   本改 ✓: **每帧**采集真值 (`actor.get_pose().p[2]` ✓ 引擎真值 ✓)
            #     ⇒ `obj_z_history` 变完整时间序列 ⇒ 抬升量/脱桌/稳定都可测 ✓
            #   ⚠ 只**增加采样** ✓: 不改判据阈值、不改奖励、不动物理 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ G3② ✓ (方案 §10.3-4): 每帧**只消费一次**"本控制周期新增交集子步" ✗
            #   (消费式接口 ✓ ⇒ 历史一次接触不会让后续帧看起来"当前仍双侧" ✓)
            try:
                from physics_env import get_target_contact_report_current as _gtcr_cur
                _cur_rep = _gtcr_cur() or {}
                self._cur_both_new = int(_cur_rep.get("both_force_steps_new", 0) or 0)
                self._cur_invalid = int(_cur_rep.get("sep_invalid", 0) or 0) + int(
                    _cur_rep.get("normal_invalid", 0) or 0) + int(
                    _cur_rep.get("probe_error", 0) or 0)
            except Exception:  # noqa: BLE001
                self._cur_both_new = 0
                self._cur_invalid = 0
            try:
                for _a_t2c in self.obj_actors:
                    if target_name in _a_t2c.get_name():
                        _p_t2c = np.asarray(_a_t2c.get_pose().p, dtype=np.float64)
                        obj_z_history.append(float(_p_t2c[2]))
                        # ★ T2j (2026-09-25) 记录**物体 xy** (不只看 z ✓) ⇒ 判"被挤走/没夹上" ✗
                        if not hasattr(self, '_s1_obj_xy_hist'):
                            self._s1_obj_xy_hist = []
                        self._s1_obj_xy_hist.append(_p_t2c.copy())
                        break
            except Exception:  # noqa: BLE001
                pass
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T2j (2026-09-25) 【物体 xy 追踪】闭合/抬升过程中物体**被挤走**了吗? ✗ ★★★★
            #   为什么必须加 ✓: V1 实测 `抬升段 法向力=0.0000N ✗ + 双侧=False ✗ + 物体z恒 0.0195 ✗`
            #     ⇒ 三种可能自洽解释都指向同一件事: **物体已不在两指之间** ✓
            #     · 若 `|Δxy|` 明显(几 mm) ⇒ **闭合把物体侧向推走** ✗ ⇒ 修闭合速度/过闭合 ✓
            #     · 若 `|Δxy| ≈ 0` 而指缝仍大于跨度 ⇒ **根本没错开对**(跨度/指缝口径 ✗)
            #   ⇒ 本诊断每 20 帧打印 `Δx/Δy/Δz` + **两指间距** + 指缝 ⇒ 一眼分清 ✓
            #   ⚠ 只读不写 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if (frame % 20 == 0 and frame >= int(_TRAVEL_FRAMES)
                    and hasattr(self, '_s1_obj_xy_hist') and len(self._s1_obj_xy_hist) > 0):
                try:
                    _h_t2j = self._s1_obj_xy_hist
                    _d_t2j = _h_t2j[-1] - _h_t2j[0]
                    _fp_t2j = {}
                    for _l_t2j in robot.get_links():
                        _n_t2j = _l_t2j.get_name()
                        if _n_t2j in (f"{side}_gripper_finger_link1",
                                      f"{side}_gripper_finger_link2"):
                            _fp_t2j[_n_t2j] = np.asarray(_l_t2j.get_pose().p,
                                                         dtype=np.float64)
                    _gap_t2j = float('nan')
                    if len(_fp_t2j) == 2:
                        _vv_t2j = list(_fp_t2j.values())
                        _gap_t2j = float(np.linalg.norm(_vv_t2j[0] - _vv_t2j[1])) * 1000.0
                    logger.info(
                        "[T2j 物体xy追踪] F%d | 物体 Δx=%+.1fmm Δy=%+.1fmm Δz=%+.1fmm "
                        "(|Δxy|=%.1fmm) | 两指中心距=%.1fmm | 指缝=%.1fmm",
                        int(frame), _d_t2j[0] * 1000.0, _d_t2j[1] * 1000.0,
                        _d_t2j[2] * 1000.0,
                        float(np.linalg.norm(_d_t2j[:2])) * 1000.0, _gap_t2j,
                        (FINGER_GAP0 + 2.0 * float(_q_ramp if '_q_ramp' in dir()
                                                   else _GRIP_TARGET)) * 1000.0)
                except Exception as _e_t2j:  # noqa: BLE001
                    logger.warning("[T2j 物体xy追踪] 失败: %s", _e_t2j)

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T7 (2026-09-26) 【闭合到位后 · 有界再居中】一次修正, 仍 ±3mm ✓ ★★★★
            #   T6 实测铁证 ✓ (s9 `[几何快照]`, 同一 rollout):
            #     `LIFT+15: 左gap=0.00mm(分片19) 右gap=**5.41mm**(分片47) | 冲量 指1=2.63 指2=**0.000**`
            #     `LIFT+30: 左14.86 右18.61 | LIFT+60: 51.4/52.3 | LIFT+90: 88.1/87.9` ⇒ gap 随抬升线性增大
            #     ⇒ 结论 ✓: **闭合后只有单侧贴上(另一侧差 5.4mm) ⇒ 不构成夹持 ⇒ 抬升=滑落** ✗
            #   ⇒ 张开态测的 δ=1.93mm **修不够** ✗ (张开→闭合几何会变)
            #   ⇒ 本改 ✓: 在 **CLOSE_END** 用同款 GJK 重量 δ=(gL−gR)/2;
            #       0.05 < |δ| ≤ 3mm ⇒ 沿**两指关节轴**(最反向的一对, 与 T5 同口径 ✓)平移**剩余轨迹** anchor ✓
            #       |δ|≤0.05 ⇒ 已居中 ✓; 越界/异常 ⇒ **不修正** ✓ (报 BOUND_EXCEEDED, 不默默叠偏移 ✗)
            #   ⚠ 只修正一次 ✓ (`_t7_center_done`) + 有界 ±3mm ✓ + 不新增门/不改奖励 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if (not _t7_center_done
                    and str(os.environ.get("_S1_CENTER_MEASURE", "close")) == "close"
                    and int(frame) == int(_TRAVEL_FRAMES + _CLOSE_FRAMES)):
                _t7_center_done = True
                try:
                    from gap_utils import convex_gap_min_mm as _gmin_t7
                    _hv7 = self._coacd_world_hulls(target_name) or []
                    _vv7 = self._finger_world_vertices(robot) or []
                    if _hv7 and len(_vv7) == 2:
                        _gL7, _, _ = _gmin_t7(_hv7, _vv7[0])
                        _gR7, _, _ = _gmin_t7(_hv7, _vv7[1])
                        _d7_mm = 0.5 * (float(_gL7) - float(_gR7))
                        _lim7 = float(os.environ.get("_S1_CENTER_LIMIT_MM", "3.0") or 3.0)
                        if abs(_d7_mm) < 0.05:
                            self._s1_center_status = "CENTERED_OK_AT_CLOSE"
                            logger.warning("[T7 闭合后居中] ✓ 已居中 (gL=%.2f gR=%.2fmm δ=%+.2f) ✓ "
                                           "⇒ 两侧应同时贴住 ✓", float(_gL7), float(_gR7), _d7_mm)
                        elif abs(_d7_mm) > _lim7:
                            self._s1_center_status = (
                                f"BOUND_EXCEEDED_AT_CLOSE(delta={_d7_mm:+.2f}mm)")
                            logger.error("[T7 闭合后居中] ✗ **BOUND_EXCEEDED** δ=%+.2fmm > %.2fmm "
                                         "(gL=%.2f gR=%.2f) ⇒ **不修正** ✓ (不默默叠偏移 ✗)",
                                         _d7_mm, _lim7, float(_gL7), float(_gR7))
                        else:
                            def _q2R7(_q):
                                _w, _x, _y, _z = (float(_q[0]), float(_q[1]),
                                                  float(_q[2]), float(_q[3]))
                                return np.array([
                                    [1 - 2 * (_y * _y + _z * _z), 2 * (_x * _y - _w * _z),
                                     2 * (_x * _z + _w * _y)],
                                    [2 * (_x * _y + _w * _z), 1 - 2 * (_x * _x + _z * _z),
                                     2 * (_y * _z - _w * _x)],
                                    [2 * (_x * _z - _w * _y), 2 * (_y * _z + _w * _x),
                                     1 - 2 * (_x * _x + _y * _y)],
                                ], dtype=np.float64)

                            _lc7 = {}
                            for _l7 in robot.get_links():
                                _n7 = _l7.get_name()
                                if _n7 in (f"{side}_gripper_finger_link1",
                                           f"{side}_gripper_finger_link2"):
                                    _lc7[_n7] = _l7
                            if len(_lc7) == 2:
                                _L1c7 = _lc7[f"{side}_gripper_finger_link1"]
                                _L2c7 = _lc7[f"{side}_gripper_finger_link2"]
                                _R1c7 = _q2R7(_L1c7.get_pose().q)
                                _R2c7 = _q2R7(_L2c7.get_pose().q)
                                _ax7 = np.array([0.0, 1.0, 0.0])
                                _best7 = None
                                for _s1_7 in (1.0, -1.0):
                                    for _s2_7 in (1.0, -1.0):
                                        _a1c7 = _R1c7 @ (_s1_7 * _ax7)
                                        _a2c7 = _R2c7 @ (_s2_7 * _ax7)
                                        _dd7 = float(np.dot(_a1c7, _a2c7))
                                        if _best7 is None or _dd7 < _best7[0]:
                                            _best7 = (_dd7, _a1c7)
                                if _best7 is not None and _best7[0] <= -0.9:
                                    _u7 = _best7[1] / float(np.linalg.norm(_best7[1]))
                                    _mid7 = (np.asarray(_L2c7.get_pose().p, dtype=np.float64)
                                             - np.asarray(_L1c7.get_pose().p, dtype=np.float64))
                                    if float(np.dot(_u7, _mid7)) < 0.0:
                                        _u7 = -_u7      # 指向"指1→指2" ✓ (与大间隙侧相反时取反 ✓)
                                    target_pos = (np.asarray(target_pos, dtype=np.float64)
                                                  + (_d7_mm / 1000.0) * _u7)
                                    self._s1_center_shift_mm = float(_d7_mm)
                                    self._s1_center_status = "APPLIED_AT_CLOSE"
                                    logger.error(
                                        "[T7 闭合后居中] ✓ 沿关节轴平移 **%+.2fmm** "
                                        "(gL=%.2f gR=%.2fmm δ=%+.2f ≤ %.2f ✓ 轴 dot=%.3f ✓) ⇒ "
                                        "**后续抬升按新 anchor 执行** ✓ 期望两侧间隙趋同 ⇒ 双侧同时贴住 ✓",
                                        _d7_mm, float(_gL7), float(_gR7), _d7_mm, _lim7, float(_best7[0]))
                except Exception as _e7:  # noqa: BLE001
                    logger.warning("[T7 闭合后居中] 失败(按原 anchor 继续 ✓): %s", _e7)

            # 位置: 就位维持 → 竖直抬升 (v5.1 无下降段)
            # ★ T8 ✓: 抬升**推迟** `_t8_ext_used` 帧 (先建立双侧法向力, 再抬 ✓ 用户要求 ✓
            #   "使双侧法向力真实建立后才抬升" ✓)
            if frame < _LIFT_START_FRAME + _t8_ext_used:
                current_pos = target_pos.copy()
            else:
                _lift_t = frame - _LIFT_START_FRAME - _t8_ext_used
                current_pos = target_pos.copy()
                current_pos[2] = target_pos[2] + _LIFT_SPEED * _lift_t
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T2c (2026-09-25) 抬升诊断: **命令 vs 实际 vs 物体** 三线对齐 ✗ ★★★★
                #   背景 ✓: v7/v8 实测 `pos_z` 恒为 0.0339**计划值** ✗(CSV 只存计划 ✗)、
                #     物体 z 恒 0.0195 ✗、`obj_lift=0.00cm` ✗ ⇒ "抬起=False" 究竟是真没抬
                #     还是**没下发**/没跟上, 无从分辨 ✗ (正是"读数不完整"✗)
                #   本诊断 ✓: 在抬升起点/中点/终点各打一次:
                #     · **命令 z** (`current_pos[2]`, 由 `_LIFT_SPEED × 帧数` 推出 ✓)
                #     · **实际基座 z** (虚拟关节 `vz` 的 qpos 真值 ✓)
                #     · **物体 z** (引擎真值 ✓)
                #   ⇒ 三种可能一眼分清:
                #     A 命令在涨 而 实际不涨 ⇒ **PD/驱动**跟不上 ✗ (物理侧)
                #     B 命令本身不涨 ⇒ **相位/变量**错 ✗ (如 vlock 覆盖 current_pos)
                #     C 命令/实际都涨 而 物体不涨 ⇒ **夹持/摩擦**不够 ✗ (抓不住)
                #   ⚠ 只读不写 ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                pass
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ G4 (§10.5-②) 【固定候选 · 同帧几何快照】✓ 三个时刻 ✓ ★★★★
            #   方案原话 ✓: "保存每个 collision shape 的局部几何与变换、目标/两指实际 T、实际 q、
            #     候选来源和完整预处理变换链; **保留 CoACD 分片**, 不只存拼接点云;
            #     在**开放就位、闭合末端、抬升前**三个时刻记录左右 gap/最近点/法向、
            #     指腹覆盖区域、原始冲量及桌面支撑" ✓
            #   本实现 ✓: 逐分片求最小(保留分片 ID ✓) + 最近点对(来自 QP 解 ✓)
            #     + 每指**原始法向冲量** `sum_imp`(不做阈值 ✓) + 桌面支撑三态 ✓
            #     + 无效计数(sep/normal/probe ✓) ⇒ 一眼看清"观测是否有效" ✓
            #   ⚠ 只读 + 追加写 JSONL ✓; 不改控制/判据/奖励 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T6c (2026-09-26) 【抬升中段采样】必须采到"接触丢失"那一刻 ✗ ★★★★
            #   实测铁证 ✓ (s7/s8): F170 物体已升 **+40.8~79.5mm** ✓ 但 F200 掉回 **+1.1mm** ✗
            #     双侧受力帧 **59~60/160 ≈ 37%** ✗ ⇒ 丢失发生在 **F170~F200** 之间 ✓
            #   本改 ✓: 抬升中段加采 4 帧(+15/+30/+60/+90), 并记录**物体位姿(位置+四元数)**
            #     与**两指位置** ⇒ 一眼分辨三类机理:
            #       · 两侧 gap 同时变大 ⇒ 沿接近轴**滑落** ✗ (接触面积小/表面太滑)
            #       · 一侧 gap 变大     ⇒ **被挤开/物体转动** ✗
            #       · 物体四元数转角大 ⇒ **在指间转动** ✗
            #   ⚠ 只读 + 追加写 JSONL ✓; 不改控制/判据/奖励 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if frame in (int(_TRAVEL_FRAMES), int(_TRAVEL_FRAMES + _CLOSE_FRAMES),
                         int(_LIFT_START_FRAME),
                         int(_LIFT_START_FRAME) + 15, int(_LIFT_START_FRAME) + 30,
                         int(_LIFT_START_FRAME) + 60, int(_LIFT_START_FRAME) + 90):
                try:
                    import json as _json
                    from gap_utils import convex_gap_points_mm as _cgp
                    from physics_env import (get_target_contact_report as _rp_s,
                                             object_ground_report as _ogr_s)
                    _vv_s = self._finger_world_vertices(robot) or []
                    _hv_s = self._coacd_world_hulls(target_name) or []
                    _rep_s = _rp_s() or {}
                    _ogr_sv = _ogr_s() or {}
                    _phase_s = ("OPEN_READY" if frame == int(_TRAVEL_FRAMES)
                                else ("CLOSE_END"
                                      if frame == int(_TRAVEL_FRAMES + _CLOSE_FRAMES)
                                      else ("PRE_LIFT" if frame == int(_LIFT_START_FRAME)
                                            else f"LIFT+{int(frame) - int(_LIFT_START_FRAME)}")))
                    # ★ T6c ✓: 物体位姿 + 两指位置 (用于判"滑落 vs 转动/挤开" ✓)
                    self._t6c_obj_pos = self._t6c_obj_quat = None
                    self._t6c_f1_pos = self._t6c_f2_pos = None
                    for _a_t6 in getattr(self, "obj_actors", []) or []:
                        if target_name in str(_a_t6.get_name()):
                            try:
                                self._t6c_obj_pos = np.asarray(_a_t6.get_pose().p,
                                                              dtype=np.float64)
                                self._t6c_obj_quat = np.asarray(_a_t6.get_pose().q,
                                                               dtype=np.float64)
                            except Exception:  # noqa: BLE001
                                pass
                            break
                    for _l_t6 in robot.get_links():
                        _n_t6 = _l_t6.get_name()
                        if _n_t6 == f"{side}_gripper_finger_link1":
                            self._t6c_f1_pos = np.asarray(_l_t6.get_pose().p, dtype=np.float64)
                        elif _n_t6 == f"{side}_gripper_finger_link2":
                            self._t6c_f2_pos = np.asarray(_l_t6.get_pose().p, dtype=np.float64)
                    _rec_s = {
                        "phase": _phase_s, "frame": int(frame),
                        # ★ T6c ✓: 物体位姿 + 两指位置 (判"滑落/被挤开/转动" ✓)
                        "obj_pos_m": (None if self._t6c_obj_pos is None
                                      else [float(x) for x in self._t6c_obj_pos]),
                        "obj_quat": (None if self._t6c_obj_quat is None
                                     else [float(x) for x in self._t6c_obj_quat]),
                        "finger1_pos_m": (None if self._t6c_f1_pos is None
                                          else [float(x) for x in self._t6c_f1_pos]),
                        "finger2_pos_m": (None if self._t6c_f2_pos is None
                                          else [float(x) for x in self._t6c_f2_pos]),
                        # ★ 修 bug ✓: `gripper_target` 在**本帧更后面**才赋值 ⇒ 早期帧会 NameError ✗
                        "q_cmd": float(locals().get("gripper_target", float("nan"))),
                        # ★ 修 bug ✓: `_q` 在本帧**更后面**才赋值 ⇒ 早期帧会 NameError ✗
                        "q_act1": float(locals().get("_q", np.array([np.nan]))[gripper_idx1])
                        if "_q" in locals() else float("nan"),
                        "q_act2": float(locals().get("_q", np.array([np.nan]))[gripper_idx2])
                        if "_q" in locals() else float("nan"),
                        "n_hulls": int(len(_hv_s)),
                        "obj_supported": _ogr_sv.get("supported"),
                        "obj_supported_state": _ogr_sv.get("state"),
                        "finger1_sum_imp_raw": float((_rep_s.get("finger1") or {}).get("sum_imp", 0.0) or 0.0),
                        "finger2_sum_imp_raw": float((_rep_s.get("finger2") or {}).get("sum_imp", 0.0) or 0.0),
                        "both_force_steps": int(_rep_s.get("both_force_steps", 0) or 0),
                        "sep_invalid": int(_rep_s.get("sep_invalid", 0) or 0),
                        "normal_invalid": int(_rep_s.get("normal_invalid", 0) or 0),
                        "probe_error": int(_rep_s.get("probe_error", 0) or 0),
                        # ★ T5 ✓ (方案 §10.4): 保存**修正量 + 实际 anchor** ✓ (B 组可复查 ✓)
                        "center_status": str(getattr(self, "_s1_center_status", "") or ""),
                        "center_shift_mm": float(getattr(self, "_s1_center_shift_mm", 0.0) or 0.0),
                        "center_axis": getattr(self, "_s1_center_axis", None),
                        "anchor_pos_m": [float(x) for x in
                                         np.asarray(locals().get("target_pos",
                                                                 np.zeros(3)), dtype=np.float64)],
                    }
                    for _sk_s, _vi_s in (("left", 0), ("right", 1)):
                        if _vi_s >= len(_vv_s):
                            continue
                        _best_s = None
                        for _pi_s, _h_s in enumerate(_hv_s):
                            try:
                                _g_s, _hit_s, _pa_s, _pb_s = _cgp(_h_s, _vv_s[_vi_s])
                            except Exception:  # noqa: BLE001
                                continue
                            if _best_s is None or float(_g_s) < _best_s["gap_mm"]:
                                _best_s = {"gap_mm": float(_g_s), "hit": bool(_hit_s),
                                           "piece": int(_pi_s),
                                           "nearest_target_m": [float(x) for x in _pa_s],
                                           "nearest_finger_m": [float(x) for x in _pb_s]}
                        if _best_s is not None:
                            _rec_s[f"{_sk_s}_nearest"] = _best_s
                    _od_s = (getattr(self, "output_dir", None)
                             or os.environ.get("_OUTPUT_DIR") or ".")
                    with open(os.path.join(str(_od_s), "geom_snapshot.jsonl"), "a") as _fs:
                        _fs.write(_json.dumps(_rec_s, ensure_ascii=False) + "\n")
                    logger.warning(
                        "[几何快照] %s: 左gap=%.2fmm(分片%d) 右gap=%.2fmm(分片%d) | 分片数=%d | "
                        "冲量 指1=%.3e 指2=%.3e | 双侧子步=%d | 无效(sep/normal/probe)=%d/%d/%d",
                        _phase_s,
                        float((_rec_s.get("left_nearest") or {}).get("gap_mm", -1.0)),
                        int((_rec_s.get("left_nearest") or {}).get("piece", -1)),
                        float((_rec_s.get("right_nearest") or {}).get("gap_mm", -1.0)),
                        int((_rec_s.get("right_nearest") or {}).get("piece", -1)),
                        int(len(_hv_s)),
                        float(_rec_s["finger1_sum_imp_raw"]),
                        float(_rec_s["finger2_sum_imp_raw"]),
                        int(_rec_s["both_force_steps"]),
                        int(_rec_s["sep_invalid"]), int(_rec_s["normal_invalid"]),
                        int(_rec_s["probe_error"]))
                except Exception as _es_s:  # noqa: BLE001
                    logger.warning("[几何快照] 失败(不阻断 ✓): %s", _es_s)

            if frame in (int(_LIFT_START_FRAME), int(_LIFT_START_FRAME) + 30,
                             int(_LIFT_START_FRAME) + 60, int(n_frames) - 1):
                    try:
                        _qa_t2c = robot.get_qpos()
                        _z_act_t2c = (float(_qa_t2c[virtual_idx['vz']])
                                      if virtual_idx else float('nan'))
                        logger.info(
                            "[T2c 抬升诊断] F%d | **命令z=%.4f** (target=%.4f + %.4f×%d帧) | "
                            "**实际基座z=%.4f** (差 %+.1fmm) | **物体z=%.4f** (升 %+.1fmm)",
                            int(frame), float(current_pos[2]), float(target_pos[2]),
                            float(_LIFT_SPEED), int(_lift_t), _z_act_t2c,
                            (_z_act_t2c - float(current_pos[2])) * 1000.0,
                            (float(obj_z_history[-1]) if obj_z_history else float('nan')),
                            ((float(obj_z_history[-1]) - float(obj_z_history[0])) * 1000.0
                             if len(obj_z_history) > 1 else float('nan')))
                    except Exception as _e_t2c:  # noqa: BLE001
                        logger.warning("[T2c 抬升诊断] 失败: %s", _e_t2c)
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ T2d (2026-09-25) 【夹持面诊断】夹**侧壁** 还是 **托顶/底**? ✗ ★★★★
                    #   背景 ✓: T2c 查明抬升链路正常(命令z 0.0358→0.2342, 实际紧跟 1.5mm ✓),
                    #     但物体 **F205 升 +9.7mm → F235 又回到 0.0191** ✗ = "提起 1cm 后滑落" ✗
                    #     ⇒ 必须分清: 手指压的是**侧壁**(⇒真夹住, 滑落=力/摩擦问题 ✓)
                    #       还是**顶面/底面**(⇒只是"托/勾" ✗ ⇒ 位姿/高度要改 ✓)
                    #   判据 ✓ (全部引擎真值 ✓, 不用接触点 API ✗ 那个上次取不到 ✓):
                    #     `dz = 指link中心z − 物体中心z`; 侧壁带 = `±(半高 − 5mm)` ✓
                    #     ⇒ `|dz| ≤ 带` ⇒ **侧壁 ✓**; `dz > 0` 且超出 ⇒ **顶面/托 ✗**;
                    #       `dz < 0` 且超出 ⇒ **底面/勾 ✗**
                    #   通过标准 ✓: 判定 **侧壁 ✓** 且 **物体 z 随基座同步上升** ✓
                    #   ⚠ 只读不写 ✓
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    try:
                        _fz_t2d = {}
                        for _l_t2d in robot.get_links():
                            _n_t2d = _l_t2d.get_name()
                            if _n_t2d in (f"{side}_gripper_finger_link1",
                                          f"{side}_gripper_finger_link2"):
                                _fz_t2d[_n_t2d] = float(
                                    np.asarray(_l_t2d.get_pose().p, dtype=np.float64)[2])
                        _cz_t2d = _hh_t2d = float('nan')
                        _hv_t2d = self._coacd_world_hulls(target_name) or []
                        if _hv_t2d:
                            _P_t2d = np.vstack([np.asarray(_h, dtype=np.float64)
                                                for _h in _hv_t2d])
                            _cz_t2d = 0.5 * (float(_P_t2d[:, 2].min())
                                             + float(_P_t2d[:, 2].max()))
                            _hh_t2d = 0.5 * (float(_P_t2d[:, 2].max())
                                             - float(_P_t2d[:, 2].min()))
                        _band_t2d = _hh_t2d - 0.005      # 侧壁带 (离顶底各留 5mm ✓)
                        _dz1_t2d = _fz_t2d.get(f"{side}_gripper_finger_link1",
                                               float('nan')) - _cz_t2d
                        _dz2_t2d = _fz_t2d.get(f"{side}_gripper_finger_link2",
                                               float('nan')) - _cz_t2d

                        def _face_t2d(_dz):
                            if not np.isfinite(_dz) or not np.isfinite(_band_t2d):
                                return "未知"
                            if abs(_dz) <= _band_t2d:
                                return "**侧壁 ✓**"
                            return "**顶面/托 ✗**" if _dz > 0 else "**底面/勾 ✗**"

                        logger.info(
                            "[T2d 夹持面诊断] F%d | 物体中心z=%.4f 半高=%.1fmm ⇒ 侧壁带±%.1fmm | "
                            "指1 dz=%+.1fmm → %s | 指2 dz=%+.1fmm → %s | 物体升 %+.1fmm",
                            int(frame), _cz_t2d, _hh_t2d * 1000.0, _band_t2d * 1000.0,
                            _dz1_t2d * 1000.0, _face_t2d(_dz1_t2d),
                            _dz2_t2d * 1000.0, _face_t2d(_dz2_t2d),
                            ((float(obj_z_history[-1]) - float(obj_z_history[0])) * 1000.0
                             if len(obj_z_history) > 1 else float('nan')))
                    except Exception as _e_t2d:  # noqa: BLE001
                        logger.warning("[T2d 夹持面诊断] 失败: %s", _e_t2d)

            if virtual_idx:
                qpos[virtual_idx['vx']] = current_pos[0]
                qpos[virtual_idx['vy']] = current_pos[1]
                qpos[virtual_idx['vz']] = current_pos[2]
                qpos[virtual_idx['rz']] = rz
                qpos[virtual_idx['ry']] = ry
                qpos[virtual_idx['rx']] = rx

            # 手指: v5.1 — 就位时已张开包住物体 (set_qpos 初始 GRIPPER_MAX_OPEN),
            # F0-F19 原地闭合, 之后保持 PD 夹紧.
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【计划 §6】相位顺序改为 **就位 → 闭合 → 保持 → 抬升** ✓ ★★★★
            #   旧 ✗: F0 即闭合, 而基座此时**还在赶往 anchor** ⇒ 手指途中合拢 ⇒ 夹不住 ✗
            #     (实测 A 真夹持率 0.11 ✗ / 同 anchor 的 B 就位后闭合 = 100% ✓)
            #   新 ✓: `_TRAVEL_FRAMES` 帧**保持张开**等基座到位 ⇒ 再按 `_close1_q` 闭合 ✓
            #     (`_TRAVEL_FRAMES=0` ⇒ 与旧行为**逐帧相同** ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ T8 ✓: 本帧是否处于"为建立**双侧法向力**而额外压入"的状态 ✓
            #   (只在"闭合窗口已结束 + 双侧法向力仍未建立 + 预算未用完"时为真 ✓)
            _t8_ext_on = bool((not _t8_ok_bil) and (_t8_ext_used < _t8_max_frames)
                              and (_t8_press_m < _t8_extra_m)
                              and int(frame) >= int(_TRAVEL_FRAMES + _CLOSE_FRAMES))

            if frame < _TRAVEL_FRAMES:
                gripper_target = float(stage_q_open())        # ★ 保持张开, 等基座到位 ✓
            elif (frame < _TRAVEL_FRAMES + _CLOSE_FRAMES) or _t8_ext_on or (
                    '_bil_freeze_q' in dir() and _bil_freeze_q is not None):
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-24 【md §4.2 实施】**接触感知闭合** ✓ (治"碰都没碰上" ✗) ★★★★
                #   实测铁证 ✓ (t3a F165): `pos_z=0.1537 f1_qpos=0.0031 f2_qpos=0.0033
                #     **contacts=[]** obj_z=0.01` ⇒ 手指合拢、夹爪抬起, 但**与物体零接触** ✗
                #   病根 ✓: 闭合终点 q 由**跨度估算**一次算定 ✗ (开环 ✗) ⇒ 停在半空 ✗
                #   本改 ✓ (md §4.2 逐条):
                #     ② 近接触(任一指有接触 或 表面间隙<3mm) ⇒ 速度 20mm/s→**5mm/s** ✓
                #     ④ 双侧受力**连续 3 控制点** ⇒ 记录 q_touch ⇒ **+0.5mm 预紧** 并冻结 ✓
                #        (`_S1_PRELOAD_M` 默认 0.5mm ✓ md §4.2-5 ✓)
                #     ⑦ 冻结后不再推进 ⇒ 与 SETTLE/抬升**连续** ✓
                #   ⚠ v1 未做 (诚实标注 ✗): 单侧"再推 ≤2mm 后判 UNILATERAL_GRASP" ✓
                #     与"最长闭合 4s"上限 ✓ —— 下一版补 ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                if '_q_ramp' not in dir():
                    # ★ 任务03 ✓: `_q_ramp` **必须**从上一周期实际下发 q 出发 ✗ (旧: station_q_open)
                    _q_ramp = float(getattr(self, "_s1_q_cmd_state", None)
                                    or locals().get("_q_cmd_cont")
                                    or stage_q_open())
                    _bil_ct = 0
                    _bil_freeze_q = None
                _any_c, _bil_c, _sep_min = False, False, 999.0
                try:
                    from physics_env import get_target_contact_report as _gtcr_c
                    _rc = _gtcr_c()
                    _r1 = _rc.get("finger1") or {}
                    _r2 = _rc.get("finger2") or {}
                    _any_c = bool((_r1.get("n") or 0) > 0 or (_r2.get("n") or 0) > 0)
                    _bil_c = bool((_r1.get("n_force") or 0) > 0
                                  and (_r2.get("n_force") or 0) > 0)
                    _seps = [x for x in (_r1.get("min_sep"), _r2.get("min_sep"))
                             if x is not None]
                    _sep_min = float(min(_seps)) if _seps else 999.0
                except Exception:  # noqa: BLE001
                    pass
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-24 【用户 + md §4.2-5】**力 = K × 挡停量** ✓ ★★★★
                #   用户原话 ✓: "**挡停量和力就是 k 的关系吗, 好好搞清楚了**; 关键给力, 提起来" ✓
                #   md §4.2-5 原文 ✓: "K=1000 N/m 下 0.5–1.5mm 的稳态误差对应约 **0.5–1.5N**
                #     的驱动弹性项" ✓ ⇒ **F ≈ K·Δq** ✓ (Δq = 命令 − 实际 ✓)
                #   为什么必须用它 ✓: SAPIEN 对本场景的"静态贴住"接触对上报 **impulse=0** ✗
                #     (cc4 实测 Σimp=0.000e+00 ✗) ⇒ md §5.1 的冲量通道**读不出值** ✗
                #     ⇒ K·Δq 就是等价且可用的力估计 ✓ (md 自己给的量级 ✓)
                #   阈值 ✓: 每指 > `_S1_BIL_FMIN_N` (默认 **0.5N** ✓ md 量级下沿 ✓)
                #     ⇒ 对 0.099kg 物体只需 ≈0.24N/指 (2μN≥mg ✓) ⇒ 0.5N 有 2× 余量 ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _kf_c = float(os.environ.get("_FINGER_PD_K", "1000") or 1000.0)
                _fmin_c = float(os.environ.get("_S1_BIL_FMIN_N", "0.5") or 0.5)
                _f1n = _f2n = 0.0
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-24 【修正】关节下标必须用 `robot_info["gripper_idx1/2"]` ✓ ★★★★
                #   cc5 实测铁证 ✗: 我用 `qpos[-2]` ⇒ **读到非手指关节** ⇒ Δq1=29.66mm(=全开量 ✗)
                #     ⇒ **假力 29.7N** ✗ ⇒ 冻结判据被乱触发 (65 次 ✗) ⇒ 数据不可用 ✗
                #   ⇒ 本改 ✓: 用 pipeline 自己的下标 (`robot_info` ✓ 与 stage3 同源 ✓);
                #     **取不到就保持 0** ✓ ⇒ 宁可不估计, 也不产生假力 ✓ (未知不得冒充 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-24 【照抄已验证实现】手指下标必须**按关节名取** ✓ ★★★★
                #   参考 ✓: `example/combination/05_gripper_test.py` (test8 **成功** ✓
                #     "双侧夹紧 43N / 提起 20cm / 搬运 40cm" ✓) 的写法:
                #       `idx1 = joint_names.index(f"{prefix}_gripper_finger_joint1")`
                #       `idx2 = joint_names.index(...joint2)`
                #       `a_j1, a_j2 = actual_qpos[idx1], actual_qpos[idx2]` ✓
                #       `active_joints[idx1/idx2].set_drive_target(同一个 target)` ✓
                #   ⇒ 与我一致 ✓: 两指共用同一目标 ✓; 差别只在**读回必须按名字取下标** ✓
                #   cc5/cc6 的假力 (29.7N ✗) 全因我猜下标 (qpos[-2] / 活动列表下标 ✗)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _dq_src = "unavailable"
                try:
                    _ri_c = getattr(self, "robot_info", {}) or {}
                    _rb_c = _ri_c.get("robot")
                    _i1_dq = _ri_c.get("gripper_idx1")
                    _i2_dq = _ri_c.get("gripper_idx2")
                    if (_rb_c is not None and _i1_dq is not None and _i2_dq is not None):
                        _qa_v = np.asarray(_rb_c.get_qpos(), dtype=np.float64).reshape(-1)
                        _f1n = _kf_c * abs(float(gripper_target) - float(_qa_v[int(_i1_dq)]))
                        _f2n = _kf_c * abs(float(gripper_target) - float(_qa_v[int(_i2_dq)]))
                        _dq_src = "robot_info(与physics_step同源)"
                    else:
                        _f1n = _f2n = 0.0
                except Exception as _edq:  # noqa: BLE001
                    _f1n = _f2n = 0.0
                    _dq_src = f"error:{type(_edq).__name__}"
                # ★★★ 2026-09-24 【cc10 铁证 ⇒ 修正】判据是"**双侧接触**", 不是"先有力" ✗ ★★★
                #   cc10 实测: 两指都到位 (Δq1=0.16mm Δq2=0.18mm ✓) 但力只有 0.16/0.18N ✗
                #   ⇒ 我要求"受力>0.5N" 才确认双侧 ⇒ 才加预紧 ⇒ **永远到不了** (鸡生蛋死锁 ✗✗)
                #   md §4.2-5 原文 ✓: "双侧接触 ⇒ 连续 3 个控制点确认 ⇒ **目标有限推进 0.5mm
                #     作为初始预紧**" ✓ ⇒ 顺序是: **先接触 → 再预紧 → 力才上升** ✓
                #   ⇒ 本改 ✓: 判据 = 两指**都有接触记录** ✓ (力仍记录, 预紧后再校验阈值 ✓)
                # ★★★ 2026-09-24 【cc11 铁证 ⇒ 再修正】"接触"必须用**几何真值** ✗ ★★★
                #   cc11: 用探针 `n>0` ⇒ 在手指**还全开**(q=0.0295)时就误判"已接触" ✗
                #     (contact_offset 会提前生成接触对 ✓; 实测间隙 19.57mm 也报接触 ✗)
                #   ⇒ 改用**两侧引擎 separation 都 <1mm** ✓ (几何真值 ✓ 不受驱动异常干扰 ✓)
                #     separation 为 None(缺测) ⇒ 不算接触 ✓ (未知不得冒充 ✓)
                _bil_contact = False
                try:
                    if '_r1' in dir() and '_r2' in dir():
                        _s1_c = _r1.get("min_sep")
                        _s2_c = _r2.get("min_sep")
                        _bil_contact = bool(_s1_c is not None and _s2_c is not None
                                            and float(_s1_c) < 1.0
                                            and float(_s2_c) < 1.0)
                except Exception:  # noqa: BLE001
                    _bil_contact = False
                # ★★★ 2026-09-25 【cc14 铁证】**力不能用来确认接触** ✗ ★★★
                #   触发源=KΔq>阈值, 但 sep=(19.57mm,None) ⇒ 手指还差 19.6mm ✗
                #   ⇒ 因为 finger1 **张不开** ⇒ Δq1=29.33mm ⇒ 假力 29.3N ✗ (驱动故障不是接触 ✗)
                #   ⇒ 判据只保留**几何**: 两侧分离间隙都 <1mm ✓ (不受驱动异常干扰 ✓)
                #     力 F=K·Δq 仍计算并打印 ✓ 但**不参与接触判定** ✓ (等 finger1 修好再用 ✓)
                # ★ 2026-09-25 【cc15 之后】finger1 的假残差**已消失** ✓
                #   (29.3N ⇒ 10.4N ⇒ **3.07N** ✓; F2=3.16N ✓) ⇒ 力判据**重新可信** ✓
                #   ⇒ 两指 K·Δq 都 >0.5N ⇒ 判"**已建立双侧承载**" ⇒ 冻结 + 0.5mm 预紧 ✓ (md §4.2-5)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ W1 (2026-09-25) 【方案 §8.2-1】KΔq 必须**退出控制判据** ✗✗ ★★★★
                #   方案原话 ✓: "仍将 `_bil_force` 与 `_bil_c` 做 OR, 且 `_near_c` 也使用它 …
                #     两指真实 `n_force=(0,0)`、毫米级大间隙时, **跟踪误差约 0.94mm 仍被打印成
                #     双侧力**; **按 7.2-A 移除; KΔq 仅是执行器诊断, 不作接触兜底**" ✓
                #   为什么危险 ✓: 关节跟踪误差包含"驱动跟不上/被限幅/摩擦滞留"等一切来源 ✗
                #     ⇒ 用 `F=K·Δq` 当接触力 ⇒ 会把**假的**接触/双侧/降速全部喂给状态机 ✗
                #     (本会话早期 cc15 的 `Δq1=29.33mm ⇒ 假力 29.3N` 就是实例 ✗)
                #   ⇒ 本改 ✓: `_bil_force` **只计算、只打印** ✓
                #     · 合法双侧接触只认**真实接触量**: `n_force>0`(有冲量) 或 `sep<1mm` ✓
                #     · `_near_c`(限速) 也只看**几何间隙** ✓ (见下)
                #   ⚠ 不撤任何失败门 ✓, 只是把"假力"这个污染源移出判定 ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _bil_force = bool(_f1n > _fmin_c and _f2n > _fmin_c)   # **诊断专用** ✓ 不判接触 ✗
                _bil_c = bool(_bil_c or _bil_contact)
                # ★★★ 2026-09-24 修正 ✓: 旧判据 `_any_c` 会被 contact_offset 的
                #   **提前接触对**误触发 ✗ (实测: 间隙 19.57mm 却报"接触=True" ✗) ⇒
                #   全程只剩 5mm/s ✗ ⇒ 120 帧只走 6mm ✗ ⇒ **永远压不上去** ✗
                #   ⇒ 新判据: 只有**真的受力** 或 **间隙<1mm** 才降速 ✓ (不碰就全速走 ✓)
                # ★ W1 ✓ (方案 §8.2-1): 降速判据只用**几何间隙**(探针 `min_sep`, 单位 mm ✓)
                #   不再用 KΔq 假力 ✗ ⇒ "没真接触就全速闭合", "真贴近才降到 5mm/s" ✓
                _near_c = bool(_sep_min is not None and float(_sep_min) < 1.0)
                _bil_ct = (_bil_ct + 1) if _bil_c else 0
                if int(frame) == int(_TRAVEL_FRAMES):
                    # ★ 一次性诊断 ✓: 打出活动关节名 ⇒ 查"按名字取下标"为何失败 ✓ (cc7: F=0 ✗)
                    # ★ 正确句柄 ✓ (cc8 铁证: `self.robot` 不存在 ✗) ⇒ 用 robot_info ✓
                    try:
                        _ri_d = getattr(self, "robot_info", {}) or {}
                        _rb_d = _ri_d.get("robot")
                        _g1 = _ri_d.get("gripper_idx1")
                        _g2 = _ri_d.get("gripper_idx2")
                        _aj = (getattr(_rb_d, "get_active_joints", None) or (lambda: []))()
                        _qa_d = np.asarray(_rb_d.get_qpos(), dtype=np.float64).reshape(-1)
                        for _tag, _gi in (("finger1", _g1), ("finger2", _g2)):
                            _jm = ""
                            _jt = _fl = _k = _d = None
                            try:
                                _jj = _aj[int(_gi)]
                                try:
                                    _jm = str(_jj.get_name())
                                except Exception:  # noqa: BLE001
                                    _jm = "?"
                                try:
                                    _jt = float(_jj.get_drive_target())
                                except Exception:  # noqa: BLE001
                                    _jt = None
                                try:
                                    _fl = float(_jj.get_drive_force_limit())
                                except Exception:  # noqa: BLE001
                                    _fl = None
                            except Exception:  # noqa: BLE001
                                pass
                            logger.error(
                                "[§4.2·诊断] %s: 下标=%s 名字=%s | target=%s | actual=%.5f | "
                                "力上限=%s (q 实际全长=%d)",
                                _tag, _gi, _jm,
                                ("None" if _jt is None else f"{_jt:.5f}"),
                                float(_qa_d[int(_gi)]) if int(_gi) < len(_qa_d)
                                else float("nan"),
                                ("None" if _fl is None else f"{_fl:.2f}"), int(len(_qa_d)))
                    except Exception as _en_c:  # noqa: BLE001
                        logger.error("[§4.2·诊断] 取关节信息失败: %s", _en_c)
                if frame % 12 == 0:
                    # ★ 单一来源 ✓: Δq **由 F/K 反算** (不再另用下标 ⇒ 不再自相矛盾 ✗)
                    logger.info("[§4.2·力估计] F%d F1=%.3fN (Δq1=%.2fmm) | F2=%.3fN (Δq2=%.2fmm) | "
                                "阈值=%.2fN ⇒ 双侧=%s | **下标来源=%s**",
                                int(frame), float(_f1n), 1000.0 * float(_f1n) / max(_kf_c, 1e-9),
                                float(_f2n), 1000.0 * float(_f2n) / max(_kf_c, 1e-9),
                                _fmin_c, _bil_force, _dq_src)
                if _bil_ct >= 3 and _bil_freeze_q is None:
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ T1 (2026-09-25) 【拒绝误冻结】两道硬校验 ✗ ★★★★
                    #   实测铁证 ✓ (cc28 `[§4.2]`): `F36 接触=False 双侧=False 间隙=999.00mm 力=0`
                    #     却 `冻结=**0.0295**(≈全开)` ✗ ⇒ `_qcap_c=0.0295` ⇒ `_d_c=0` ⇒ 步长 0
                    #     ⇒ **闭合相位形同不存在**(q 全程 0.030→0.0295 ✗) ⇒ 手指张开 ⇒ 离物体 36mm ✗
                    #   两道校验 ✓ (都不满足就**忽略冻结** ⇒ 按计划 `_GRIP_TARGET` 继续闭合 ✓):
                    #     ① **必须真有接触证据**: `_bil_c` 或 两侧 n_force>0 或 sep<1mm 或 sep_min<1mm
                    #     ② **冻结值必须真的"闭过"**: `_q_ramp ≤ stage_q_open() − 1mm`
                    #        (冻在全开位置 = 把闭合相位废掉 ✗)
                    #   ⚠ 只加**拒绝**条件 ✓: 不新增门、不改奖励、不动物理 ✓
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    try:
                        _t1_nf1 = (_r1.get("n_force") or 0) if '_r1' in dir() else 0
                        _t1_nf2 = (_r2.get("n_force") or 0) if '_r2' in dir() else 0
                        _t1_s1 = (_r1.get("min_sep") if '_r1' in dir() else None)
                        _t1_s2 = (_r2.get("min_sep") if '_r2' in dir() else None)
                        _t1_sep_ok = bool(
                            isinstance(_t1_s1, (int, float)) and isinstance(_t1_s2, (int, float))
                            and min(float(_t1_s1), float(_t1_s2)) < 0.001)
                        _t1_force_ok = bool(_t1_nf1 > 0 and _t1_nf2 > 0)
                        _t1_sepmin_ok = bool(
                            _sep_min is not None and float(_sep_min) < 1.0)
                        # ★ T1a-fix (2026-09-25) `_bil_c` **不可信** ✗ ⇒ 只信**真实物理证据** ✓
                        #   实测 ✓ (v1 日志): `接触证据=True (bil=**True** n_force=(0,0)
                        #     sep=(**13.04**,None) sep_min=13.04)` ⇒ 零力 + 13mm 间隙却判"双侧接触"✗
                        #   ⇒ 把 `_bil_c` **从证据链移除** ✓ (它是污染源 ✓), 只保留:
                        #     探针两侧法向力>0 ✓ / 探针最小间隙<1mm ✓ / 全局间隙<1mm ✓
                        # ★★★★ W3a (2026-09-26) 冻结必须基于**真实双侧受力** ✗ 幻象分离不行 ✗ ★★★★
                        #   实测铁证 ✓ (output/d3): `接触帧=139`(碰到 ✓) 但 `**双侧受力帧=0**` 且
                        #     `峰值力=0.00N` ✗ ⇒ 手指只"贴到"、**没压出压力** ⇒ 无摩擦 ⇒ 提不起 ✗
                        #   病根 ✓: 冻结证据里保留了 `sep<1mm` 这类**几何/接触记录**判据 ✗
                        #     (contact_offset=1mm ⇒ 没真压上也会有"近接触/接触记录" ✗)
                        #     ⇒ 一旦据此冻结 ⇒ 停在"刚好贴上"的位置 ⇒ **压力恒为 0** ✗
                        #   ⇒ 本改 ✓: 冻结证据**只用** `both_force_steps > 0`
                        #     (W1 新增的**每子步左右交集**计数 ✓ = 真·同时受力 ✓)
                        #     ⇒ 力没出现就**继续闭合**到计划深度(过闭合) ⇒ 压出压力 ✓ 再冻结 ✓
                        _t1_both_steps = 0
                        try:
                            from physics_env import get_target_contact_report as _gtcr_t1
                            _t1_both_steps = int((_gtcr_t1() or {}).get("both_force_steps", 0) or 0)
                        except Exception:  # noqa: BLE001
                            _t1_both_steps = 0
                        # ★ G3② ✓: 用**本控制周期新增**(消费式 ✓) ⇒ 不再被历史接触污染 ✗
                        _t1_ok_contact = bool(int(getattr(self, '_cur_both_new', 0)) > 0)
                        if not _t1_ok_contact and int(_t1_both_steps) > 0:
                            pass      # 累计窗口仍为报告用途 ✓ (不参与判定 ✗)
                    except Exception:  # noqa: BLE001
                        _t1_nf1 = _t1_nf2 = 0
                        _t1_s1 = _t1_s2 = None
                        _t1_ok_contact = False
                    try:
                        _t1_q_open = float(stage_q_open())
                    except Exception:  # noqa: BLE001
                        _t1_q_open = 0.05
                    # ★ T1a-fix (2026-09-25) "已闭合"必须**接近计划** ⇒ 冻结只允许"**早停一点点**" ✓
                    #   旧判据 `q_ramp ≤ 全开−1mm` 太松 ✗ ⇒ 刚一动就够条件 ⇒ 冻在 0.029 ✗
                    #   新判据 ✓: `q_ramp ≤ 计划目标 + 窗口(默认 5mm)` ⇒ 必须先**真闭到位**才可能冻结 ✓
                    #     (窗口内早停 = 正常"接触早停" ✓; 窗口外 = 未闭合就冻结 ⇒ 拒绝 ✓)
                    _t1_win_m = float(os.environ.get("_S1_FREEZE_WINDOW_MM", "5.0") or 5.0) / 1000.0
                    _t1_ok_closed = bool(float(_q_ramp) <= float(_GRIP_TARGET) + _t1_win_m)
                    if not (_t1_ok_contact and _t1_ok_closed):
                        logger.error(
                            "[T1 冻结校验] ✗ **拒绝误冻结** | 接触证据=%s (bil=%s n_force=(%s,%s) "
                            "sep=(%s,%s) sep_min=%s) | 已闭合=%s (q_ramp=%.4f vs 全开 %.4f) "
                            "⇒ **忽略本次冻结** ✓ 按计划 q=%.4f (指缝 %.1fmm) 继续闭合 ✓",
                            _t1_ok_contact, _bil_c, _t1_nf1, _t1_nf2, _t1_s1, _t1_s2, _sep_min,
                            _t1_ok_closed, float(_q_ramp), _t1_q_open, float(_GRIP_TARGET),
                            (FINGER_GAP0 + 2.0 * float(_GRIP_TARGET)) * 1000.0)
                        _bil_ct = 0
                    else:
                        _pre_m = float(os.environ.get("_S1_PRELOAD_M", "0.0005") or 0.0005)
                        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                        # ★★★★ T2d (2026-09-25) 【预紧方向修正】符号反了 ✗✗ ★★★★
                        #   实测铁证 ✓ (`[T2d 夹持面诊断]`, output/t2d):
                        #     `F175 指1 dz=+5.2mm → 侧壁 ✓`(夹持点**正确** ✓) 但
                        #     `F205 物体升 +9.7mm` → `F235/F299 物体 0/-1.2mm` 且指 dz→**+203mm**
                        #     ⇒ **手指随基座抬起 20cm, 物体被丢下** ✗ = "提起 1cm 后滑脱" ✓
                        #   病根 ✓ (本行): 冻结值 = `q_ramp **+** _pre_m` ⇒ 把目标往**开**的方向
                        #     推 `_pre_m`(默认 0.5~2mm) ✗ ⇒ 所谓"预紧"实际是**退开** ✗
                        #     ⇒ 手指只"碰到"物体、**没有压入** ⇒ 接触压力 ≈ 0 ⇒ 无摩擦 ⇒ 必滑 ✗
                        #     (与 `挡停量 = 实际 − 计划 = +3~4mm` 完全一致 ✓: 差的是"没压下去的距离" ✓)
                        #   本改 ✓: 改为 `q_ramp **−** _pre_m` ⇒ 越过接触点**压入** `_pre_m` ✓
                        #     ⇒ 产生真实法向力 `F ≈ K·_pre_m` (K=1000 ⇒ 1mm ⇒ **≈1N/指** ✓)
                        #     ⇒ 夹持摩擦 `2·μ·F ≈ 4N` ≫ 物体重力 **0.93N** ✓ ⇒ 提得起来 ✓
                        #   ⚠ 下界由 T1b 钳位保证 ✓ (`≥ _GRIP_TARGET` ⇒ 不会压过计划深度 ✓)
                        #   ⚠ 方向定义 ✓: q **减小 = 闭合** (指缝 = GAP0 + 2q ✓) ⇒ 压入用小 q ✓
                        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                        _bil_freeze_q = float(_q_ramp) - _pre_m
                    # ★ 触发源必须可见 ✓ (cc13: 冻结发生在全开位置 ⇒ 不知是谁触发 ✗)
                    try:
                        _nf1 = (_r1.get("n_force") or 0) if '_r1' in dir() else 0
                        _nf2 = (_r2.get("n_force") or 0) if '_r2' in dir() else 0
                        _s1v = (_r1.get("min_sep") if '_r1' in dir() else None)
                        _s2v = (_r2.get("min_sep") if '_r2' in dir() else None)
                        _src_bil = ("separation<1mm" if _bil_contact else
                                    ("探针n_force" if (_nf1 > 0 and _nf2 > 0) else
                                     ("KΔq>阈值" if (_f1n > _fmin_c and _f2n > _fmin_c)
                                      else "未知(可疑 ✗)")))
                    except Exception:  # noqa: BLE001
                        _s1v = _s2v = None
                        _src_bil = "异常"
                    # ★ T1c (2026-09-25) 防崩溃 ✓: T1a 可能**拒绝冻结** ⇒ `_bil_freeze_q` 仍为 None ✗
                    #   实测 ✓ (v3 crash): `float(_bil_freeze_q) ⇒ TypeError: float() argument
                    #     must be ... not 'NoneType'` ✗ ⇒ rollout 直接崩 ⇒ 整轮无输出 ✗
                    #   ⇒ 用**安全值**打印 ✓ (-1.0 = 未冻结 ✓); `_pre_m` 也可能未定义 ⇒ 取值保护 ✓
                    _t1_freeze_now = (float(_bil_freeze_q)
                                      if _bil_freeze_q is not None else -1.0)
                    _t1_pre_now = float(_pre_m) if '_pre_m' in dir() else 0.0
                    logger.error("[§4.2 接触感知闭合] ✓ **双侧接触确认**(连续3控制点) ⇒ 冻结 "
                                 "q=%.4f + 预紧 %.2fmm ✓ | **触发源=%s** | sep=(%s,%s) "
                                 "n_force=(%s,%s) F=(%.3f,%.3f)N ⇒ 若 sep/n_force 为空却仍触发, "
                                 "说明判据仍不可信 ✗  (-1.0000 = T1a **已拒绝**该冻结 ✓)",
                                 _t1_freeze_now, _t1_pre_now * 1000.0, str(_src_bil),
                                 str(_s1v), str(_s2v), str(_nf1), str(_nf2),
                                 float(_f1n), float(_f2n))
                _rate_c = 0.000167 if _near_c else 0.00067      # 5mm/s vs 20mm/s @30Hz ✓
                _qcap_c = float(_bil_freeze_q if _bil_freeze_q is not None
                                else _GRIP_TARGET)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T1b (2026-09-25) 【冻结值钳位】闭合上限**永远**不许 ≈ 全开 ✗ ★★★★
                #   为什么还要第二道 ✓: T1a 在"冻结那一刻"拒绝 ✗, 但实测仍有 rollout 打印
                #     `[§4.2] F36 q=0.0295 **冻结=0.0295**` ✗ (冻结值仍等于全开 ⇒ `_d_c=0` ⇒
                #     步长 0 ⇒ 手指全程张开 ⇒ 离物体 36mm ✗)
                #   ⇒ 在使用处**无条件钳位** ✓ (一次钳位管住所有来源, 不依赖冻结时刻的检查 ✓):
                #     · 上界 = `stage_q_open() − 1mm` ✓ (冻在全开 = 把闭合相位废掉 ✗)
                #     · 下界 = `_GRIP_TARGET` ✓ (**保留"接触早停"语义** ✓: 碰到就停得早一点 ✓,
                #       但绝不允许"停在比计划更靠外的位置"造成空夹 ✓)
                #   ⚠ 只钳**上限** ✓: 不新增门、不改奖励、不动物理 ✓
                #   ★ 期望读数 ✓: `[§4.2] q` 从 0.030 **单调下降**到 ≈0.0084 (指缝 90.4→47.2mm ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T8 (2026-09-26) 【按法向力闭合 · 继续压入】 ★★★★
                #   判据 ✓: **两侧真实法向冲量** `sum_imp_raw > 0` (§5.1 标定, 不做阈值 ✓)
                #     ⇒ 实测可用性 ✓: s10 快照 `指1=5.11 指2=0.000` ⇒ 该量能**明确**区分单/双侧 ✓
                #   ⚠ 只做"是否继续压"的判定 ✓; 压入量由下面的 `_q_lo_c` 放开 ✓ (有界 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                try:
                    _t8_imp1 = float((_r1.get("sum_imp_raw") or 0.0) if '_r1' in dir() else 0.0)
                    _t8_imp2 = float((_r2.get("sum_imp_raw") or 0.0) if '_r2' in dir() else 0.0)
                    _t8_ok_bil = bool(abs(_t8_imp1) > 0.0 and abs(_t8_imp2) > 0.0)
                    # ★ md §03 遗留 ✓: 用**真实法向冲量** (`sum_imp_raw`) 做载荷探针 ✓
                    try:
                        _v2_load_n += 1
                        _mn_imp = min(abs(float(_t8_imp1)), abs(float(_t8_imp2)))
                        _v2_load_min_imp = min(_v2_load_min_imp, _mn_imp)
                        if _mn_imp > 0.0:
                            _v2_load_frames += 1
                    except Exception:  # noqa: BLE001
                        pass
                    # ★ md §03 遗留 ✓: **固定 C0 相对平移/旋转** (v2 门的 C0 项真探针 ✓)
                    #   C0 = **抬升段首帧**夹爪基座位姿 (显式记录 ⇒ 语义写清 ✓, 不冒充
                    #   "close 入口" ✗); 之后逐帧取相对偏差的**最大值** ✓
                    try:
                        _mc_v2 = self._v2_gripper_base_pose(robot, side)
                        if _mc_v2 is not None:
                            if _v2_C0 is None:
                                _v2_C0 = np.asarray(_mc_v2, dtype=np.float64)
                            _dt_v2 = float(np.linalg.norm(
                                np.asarray(_mc_v2)[:3, 3] - _v2_C0[:3, 3]) * 1000.0)
                            _dR_v2 = _v2_C0[:3, :3].T @ np.asarray(_mc_v2)[:3, :3]
                            _cc_v2 = float(np.clip((np.trace(_dR_v2) - 1.0) / 2.0, -1.0, 1.0))
                            _v2_c0_t = max(_v2_c0_t, _dt_v2)
                            _v2_c0_r = max(_v2_c0_r, float(np.degrees(np.arccos(_cc_v2))))
                    except Exception:  # noqa: BLE001
                        pass
                    if (not _t8_ok_bil
                            and int(frame) >= int(_TRAVEL_FRAMES + _CLOSE_FRAMES)
                            and _t8_ext_used < _t8_max_frames
                            and _t8_press_m < _t8_extra_m):
                        _t8_press_m = min(_t8_extra_m, _t8_press_m + _t8_step_m)
                        _t8_ext_used += 1
                    if int(frame) % 12 == 0:
                        logger.error(
                            "[T8 按法向力闭合] F%d 双侧法向冲量=(%.3e,%.3e) **建立=%s** | "
                            "已额外压入=%.2f/%.2fmm (用 %d/%d 帧 ⇒ 抬升推迟 %d 帧) | "
                            "闭合下限 q=%.4f (指缝 %.1fmm)",
                            int(frame), _t8_imp1, _t8_imp2, _t8_ok_bil,
                            _t8_press_m * 1000.0, _t8_extra_m * 1000.0,
                            _t8_ext_used, _t8_max_frames, _t8_ext_used,
                            max(0.0, float(_GRIP_TARGET) - _t8_press_m),
                            (FINGER_GAP0 + 2.0 * max(0.0, float(_GRIP_TARGET) - _t8_press_m)) * 1000.0)
                except Exception as _e8:  # noqa: BLE001
                    logger.warning("[T8 按法向力闭合] 失败(按几何闭合 ✓): %s", _e8)

                try:
                    _q_hi_c = float(stage_q_open()) - 0.001
                    # ★ W2 ✓ (方案 §8.3-A): 下界**不得**用名义终点 ⇒ 否则合法预紧被裁回 ✗
                    #   方案原话 ✓: "随后 T1b 却把下界设为 `_GRIP_TARGET`; 当预紧终点小于
                    #     名义目标时, **压入被裁回名义值**" ✗
                    #   ⇒ 下界 = `名义目标 − 预紧预算`(预留合法的压入行程 ✓),
                    #     且不低于 0 (URDF 行程下界 ✓) ⇒ 预紧不再被偷偷取消 ✓
                    _pre_budget = float(os.environ.get("_S1_PRELOAD_M", "0.0005") or 0.0005)
                    # ★ T8 ✓: 上限再放开 `_t8_press_m` (**有界** ≤ `_S1_T8_EXTRA_MM` ✓)
                    #   ⇒ 双侧法向力未建立时可继续压入 ✓; 建立后 `_t8_press_m` 不再增长 ✓
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ 2026-09-30 (A 路) 【深夹被**静默钳成全开** ✗ 真病根修复】 ✓ ★★★★
                    #   实测铁证 ✗ (`output/s102_deep2`):
                    #     `[T1b 冻结钳位] ✗ 闭合上限 -0.0015(指缝 27.4mm) 越界 ⇒
                    #       **钳到 0.0000**(指缝 30.4mm)` ⇒ **等于废掉整段闭合** ✗
                    #   真病根 ✓: 旧 `max(0.0, ...)` 把 q 的**下界硬设为 0 = 全开** ✗ ——
                    #     而闭合方向恰恰是 q **减小/为负** (`指缝 = 30.4mm + 2q`, q<0 ⇒ 过闭合 ✓)
                    #     ⇒ 任何负 q 都被 `max(-x, 0.0)` 钳回 0 ✗
                    #     ⇒ `_S1_CLOSE_OVERCLOSURE_MM` **形同虚设** ✗ (stage1_grasp_mixin.py:104
                    #       早已自注"全都无效 ✗" —— 本轮才定位到具体那一行 ✓)
                    #   本改 ✓: 下界 = **显式过闭合预算** (按 `_S1_CLOSE_OVERCLOSURE_MM`,
                    #     并 ≤ `_S1_CLOSE_OVERCLOSURE_MAX_MM` ✓), 换算 Δq = -(mm/2)/1000 ✓
                    #     (`指缝 = 30.4mm + 2q` ⇒ 指缝减 1mm ⟺ q 减 0.5mm ✓)
                    #   ⚠ 红线 ✓: 既**不静默取消**闭合 (md §3.2 ✗), 也**不无限加深** (有界 ✓)
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    _oc_mm = min(
                        float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MM", "3.0") or 3.0),
                        float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MAX_MM", "3.0") or 3.0))
                    _oc_dq = -max(0.0, _oc_mm) / 2000.0        # mm -> Δq (指缝 = 30.4 + 2q)
                    _q_lo_c = max(_oc_dq, float(_GRIP_TARGET) - _pre_budget - _t8_press_m)
                    if frame % 60 == 0:
                        logger.warning(
                            "[A路·过闭合预算] ✓ 允许下界 q=%.4f (过闭合 %.1fmm) ⇒ "
                            "不再被静默钳成全开 ✓ (旧 max(0.0,...) 是病根 ✗)", _q_lo_c, _oc_mm)
                    _qcap_raw = float(_qcap_c)
                    _qcap_c = float(min(max(_qcap_raw, _q_lo_c), max(_q_hi_c, _q_lo_c)))
                    try:
                        _s1_frozen_eff = float(_qcap_c)   # 钳位后实际生效值 ✓
                    except Exception:  # noqa: BLE001
                        pass
                    if abs(_qcap_c - _qcap_raw) > 1e-9 and frame % 12 == 0:
                        logger.error("[T1b 冻结钳位] ✗ 闭合上限 %.4f(指缝 %.1fmm) 越界 ⇒ "
                                     "**钳到 %.4f**(指缝 %.1fmm) ✓ "
                                     "(下界=计划目标, 上界=全开−1mm; 冻在全开=废掉闭合 ✗)",
                                     _qcap_raw, (FINGER_GAP0 + 2.0 * _qcap_raw) * 1000.0,
                                     _qcap_c, (FINGER_GAP0 + 2.0 * _qcap_c) * 1000.0)
                except Exception as _e1b:  # noqa: BLE001
                    logger.warning("[T1b 冻结钳位] 失败(沿用原值 ✓): %s", _e1b)
                # ★ 修正 ✓ (cc1 实测: q 恒为 0.0300 ✗): 闭合是 q **减小** ⇒ 必须按**方向**步进 ✗
                #   (旧写法 `min(rate, max(0, cap-q))` 在 cap<q 时恒为 0 ⇒ 手指**从未闭合** ✗✗)
                _d_c = float(_qcap_c) - float(_q_ramp)
                _step_c = min(abs(_d_c), float(_rate_c)) * (1.0 if _d_c > 0.0 else -1.0)
                _q_ramp = float(_q_ramp + _step_c)
                gripper_target = float(_q_ramp)
                # ★ 任务03 ✓: 唯一命令状态 = **本周期实际下发** q (供下一周期与导出消费 ✓)
                self._s1_q_cmd_state = float(_q_ramp)
                self._last_grip_target_s1 = float(_q_ramp)
                if frame % 12 == 0:
                    # ★ md §5.1 标定 ✓: 同时打印**原始法向冲量** (不做阈值 ✓) ⇒ 看真实量级 ✓
                    _raw1 = (_r1.get("sum_imp_raw", 0.0) if '_r1' in dir() else 0.0)
                    _raw2 = (_r2.get("sum_imp_raw", 0.0) if '_r2' in dir() else 0.0)
                    _mx1 = (_r1.get("max_imp_raw", 0.0) if '_r1' in dir() else 0.0)
                    _mx2 = (_r2.get("max_imp_raw", 0.0) if '_r2' in dir() else 0.0)
                    _fdt = 1.0 / 30.0
                    logger.info("[§4.2] F%d 接触=%s 双侧=%s 间隙=%.2fmm 速度=%.1fmm/s q=%.4f "
                                "(冻结=%s) | **原始法向力** 指1: Σimp=%.3eN·s ⇒ 均力≈%.4fN "
                                "(峰值 %.3e) 指2: Σimp=%.3eN·s ⇒ 均力≈%.4fN (峰值 %.3e)",
                                int(frame), _any_c, _bil_c, _sep_min,
                                _rate_c * 1000.0 * 30.0, float(_q_ramp),
                                ("否" if _bil_freeze_q is None else f"{_bil_freeze_q:.4f}"),
                                float(_raw1), float(_raw1) / _fdt, float(_mx1),
                                float(_raw2), float(_raw2) / _fdt, float(_mx2))
                    # ★★★ T2i (2026-09-25) 斜坡内部量诊断 ✓: 为何 q 停在 0.0070 不再压入 ✗
                    try:
                        logger.info(
                            "[T2i 斜坡诊断] F%d | 上限qcap=%.4f | 当前q_ramp=%.4f | "
                            "d_c=%+.5f | 速率=%.5f/帧 | 步长=%+.5f ⇒ %s | 计划=%.4f",
                            int(frame), float(_qcap_c), float(_q_ramp),
                            float(_d_c), float(_rate_c) if '_rate_c' in dir() else -1.0,
                            float(_step_c) if '_step_c' in dir() else float('nan'),
                            ("**未推进** ✗" if abs(float(_step_c)) < 1e-9 else "推进中 ✓"),
                            float(_GRIP_TARGET))
                    except Exception as _e_t2i:  # noqa: BLE001
                        logger.warning("[T2i 斜坡诊断] 失败: %s", _e_t2i)
            else:
                # ★ 方案 §4.2-7 ✓: **连续**推进 (每帧 ≤ `_S1_CLOSE_RATE_MAX` ✓ 不再一步跳 ✗)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T2h (2026-09-25) 【保持相位必须压住接触点】✗✗ 最后一环 ★★★★
                #   实测铁证 ✓ (output/t2e `[§4.2]` 抬升段 F180~F264 全程):
                #     `接触=True **双侧=False** 间隙=-0.01mm` +
                #     `命令 q=**0.0070**` 而 `冻结 q=**0.0060**` ✗
                #     `**原始法向力 = 0.0000N(全程)**` ✗✗
                #   ⇒ 命令比接触点**更开 1mm** ⇒ 手指从接触点**倒退** ⇒ 压力归零 ⇒ 零摩擦 ⇒ 滑落 ✓
                #   病根 ✓ (本行): 保持相位只朝 `_GRIP_TARGET`(计划 0.0084) 走 ✗,
                #     把更深的**接触冻结点**丢掉了 ✗
                #   本改 ✓: 保持目标 = `min(计划, 接触冻结点)` ✓ (q 越小=越闭合 ⇒ 取 min=取更深 ✓)
                #     ⇒ 手指会继续压到"接触点 − 预紧" ⇒ 产生 `F ≈ K·预紧 ≈ 1~2N/指` ✓
                #     ⇒ 摩擦 `2μF ≈ 4~8N` ≫ 物体重力 **0.93N** ✓ ⇒ **提得起来** ✓
                #   ⚠ 下界仍由 T1b 钳位与冻结值共同保证 ✓ (不会无界压入 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _hold_tgt = float(_GRIP_TARGET)
                try:
                    if '_bil_freeze_q' in dir() and _bil_freeze_q is not None:
                        _hold_tgt = min(_hold_tgt, float(_bil_freeze_q))
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ T6b (2026-09-26) 【持续承载】保持相位必须**继续压入** ✗ ★★★★
                    #   实测铁证 ✓ (output/s7, SLIP):
                    #     · 物体在 F170 被抬起 **+79.5mm** ✓ 但 F200 又掉回 (**+1.8mm**) ✗
                    #     · **双侧受力只覆盖 59~60/160 帧 ≈ 37%** ✗ ⇒ **间歇失联** ✓
                    #     · 不是力不够: μ=2、法向 3.68N×2 ⇒ 摩擦 ≫ 自重 ✓
                    #     · 也不是桌面阻挡: 指尖净空 **+7.4mm** (不插桌 ✓)
                    #   病根 ✓: 旧写法在保持相位把 q **恒定** ⇒ 物体一微动就脱离接触 ✗
                    #   ⇒ 本改 ✓: 保持目标 = `min(计划, 冻结) − 预紧` ✓
                    #     · 预紧**有界**: 用 `_S1_PRELOAD_M` (0.5~2mm ✓), 且 ≥0 ✓
                    #     · 速率仍受限 (2mm/s 档 ✓ W2 的唯一出口钳位 ✓) ⇒ 不冲击 ✓
                    #     ⇒ 抬升全程**保持压入** ⇒ 接触不丢 ⇒ 物体随基座走 ✓
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    _pre_hold = float(os.environ.get("_S1_PRELOAD_M", "0.0005") or 0.0005)
                    _hold_tgt = max(0.0, _hold_tgt - _pre_hold)
                    if frame % 30 == 0:
                        logger.info("[T6b 持续压入] 保持目标=min(计划 %.4f, 冻结 %s) − 预紧 %.2fmm "
                                    "= **%.4f** ⇒ 指缝 %.1fmm (抬升全程保持接触 ✓)",
                                    float(_GRIP_TARGET),
                                    ("n/a" if '_bil_freeze_q' not in dir()
                                     or _bil_freeze_q is None else f"{float(_bil_freeze_q):.4f}"),
                                    _pre_hold * 1000.0, _hold_tgt,
                                    (FINGER_GAP0 + 2.0 * _hold_tgt) * 1000.0)
                        if frame % 30 == 0:
                            logger.info("[T2h 保持压入] 保持目标=min(计划 %.4f, 冻结 %.4f) = "
                                        "**%.4f** ⇒ 指缝 %.1fmm (继续压住接触点 ✓ 不再倒退 ✗)",
                                        float(_GRIP_TARGET), float(_bil_freeze_q),
                                        _hold_tgt, (FINGER_GAP0 + 2.0 * _hold_tgt) * 1000.0)
                except Exception:  # noqa: BLE001
                    pass
                _q_cmd_cont = float(_q_cmd_cont + float(np.clip(
                    _hold_tgt - float(_q_cmd_cont),
                    -_q_rate_max_c, _q_rate_max_c)))
                gripper_target = float(_q_cmd_cont)  # 保持夹紧 (接触并挤压 ✓ 且与闭合段连续 ✓)
                # ★ 方案 §8 ✓: 逐帧记录**实际下发**的 q ⇒ 交付产物用它 (不再名义重生成 ✗)
                try:
                    _log = getattr(self, "_last_close_q_executed", None) or []
                    _log.append(float(gripper_target))
                    self._last_close_q_executed = _log
                    # ★ S2-1 ✓ 同帧记录**时标 + 基座命令** ⇒ recipe 能冻结完整输入 ✓
                    #   (§S1-D 要求 actual_control_times / actual_base_commands ✓)
                    # ★ T07c ✓ 逐帧记录**已移到唯一命令下发处** (md: 不得只记某个 else 分支 ✗)
                    #   本处保留 `_last_close_q_executed` 仅为兼容旧口径 ✓
                except Exception:  # noqa: BLE001
                    pass
                # ★★ 方案 §11.3 证据 ✓: 逐帧写"**命令变化 vs 实际变化**" ⇒
                #   分类器才有权说"被挡停"(真约束 ✗) 还是"轻触"(仍能按命令运动 ✓)
                if _set_ground_ev is not None:
                    try:
                        _qa = np.asarray(qpos, dtype=np.float64).reshape(-1)
                        _qd = float(abs(float(gripper_target)
                                        - float(self._ev_prev_cmd if self._ev_prev_cmd is not None
                                                else float(gripper_target))))
                        if self._ev_prev_q is not None:
                            _ad = float(np.linalg.norm(
                                _qa - np.asarray(self._ev_prev_q, dtype=np.float64).reshape(-1)))
                            _set_ground_ev({
                                "q_residual": float(abs(_qd - _ad)),  # 命令在动而实际不动 ⇒ 大 ✗
                                "actual_vel": float(_ad),            # 实际位移 (≈速度 ✓)
                            })
                        self._ev_prev_q = _qa.copy()
                        self._ev_prev_cmd = float(gripper_target)
                    except Exception:  # noqa: BLE001
                        pass
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-23 【用户要求 · 决定性测量】"够不着" 还是 "压过头"? ✓ ★★★★
                #   为什么只做这一件 ✓: 最近几轮唯一没变的硬事实是
                #     `最近侧 = 15~20mm` ✗ 且 `asym ≈ 0~2mm` ✓ ⇒ **对称地够不着** ✓
                #     ⇒ 奖励/物理/门的改动全在**下游** ⇒ 不可能改变它 ✗ (用户的批评成立 ✓)
                #   本测量 ✓ (闭合+稳定结束那一刻 = `_LIFT_START_FRAME − 1` ✓):
                #     ① **实际指缝** = `FINGER_GAP0 + (q1_act + q2_act)` ✓ (引擎 q ✓)
                #     ② **物体在闭合轴上的真实跨度** = 读 **actor 的 collision shapes 顶点** ✓
                #        (与世界位姿相乘后投影到**闭合轴** = 夹爪局部 y ✓) ⇒ **引擎真值** ✓
                #     ③ **两指是否各自接触目标** = `_contact_detail_pairs` 的**名字**归因 ✓
                #   判读 ✓:
                #     · 跨度 > 指缝 ⇒ **闭合深度不够**(每侧差 (跨度−指缝)/2 ✗)
                #     · 跨度 < 指缝 且 无接触 ⇒ **夹持高度/位置不对**(该高度跨度更小 ✗)
                #     · 有接触 ⇒ 已经贴上 ⇒ 再看挡停/滑移 ✓
                #   只在**验证/重放**(`q_override` 非 None ✓)时打印 ⇒ 不刷屏 ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                if (_q_frozen is not None) and int(frame) == int(_LIFT_START_FRAME) - 1:
                    try:
                        _q1a = float(qpos[gripper_idx1])
                        _q2a = float(qpos[gripper_idx2])
                        _gap_mm = (FINGER_GAP0 + (_q1a + _q2a)) * 1000.0
                        _Tg2 = self._gripper_pose_matrix(robot, side)
                        _Ta2 = None
                        for _a2b in self.obj_actors:
                            if target_name and (str(target_name) in _a2b.get_name()
                                                or _a2b.get_name() in str(target_name)):
                                _Ta2 = _a2b
                                break
                        _span_mm = None
                        if _Tg2 is not None and _Ta2 is not None:
                            _Tg2 = np.asarray(_Tg2, dtype=np.float64)
                            _Tw2 = np.asarray(_Ta2.get_pose().to_transformation_matrix(),
                                              dtype=np.float64)
                            _axis2 = _Tg2[:3, :3][:, 1]          # 闭合轴 = 夹爪局部 y ✓
                            _orig2 = _Tg2[:3, 3]
                            _vlist = []
                            try:
                                for _c9 in _Ta2.components:
                                    for _sh9 in _c9.get_collision_shapes():
                                        try:
                                            _vs9 = np.asarray(_sh9.get_vertices(), dtype=np.float64)
                                            if _vs9.size:
                                                _vlist.append(_vs9.reshape(-1, 3))
                                        except Exception:  # noqa: BLE001
                                            pass
                            except Exception:  # noqa: BLE001
                                pass
                            if _vlist:
                                _Vw2 = np.concatenate(_vlist, axis=0) @ _Tw2[:3, :3].T + _Tw2[:3, 3]
                                _pr2 = (_Vw2 - _orig2) @ _axis2
                                _span_mm = float(_pr2.max() - _pr2.min()) * 1000.0
                        _f1c = _f2c = False
                        try:
                            for _ln9, _on9, _dp9 in self._contact_detail_pairs(robot, side, self.scene):
                                _s9 = str(_ln9)
                                _hit = bool(target_name) and (
                                    str(target_name) in str(_on9) or str(_on9) in str(target_name))
                                if _hit and "finger_link1" in _s9:
                                    _f1c = True
                                elif _hit and "finger_link2" in _s9:
                                    _f2c = True
                        except Exception:  # noqa: BLE001
                            pass
                        logger.warning(
                            "[够不着?] 闭合+稳定结束: **指缝=%.1fmm** | **物体沿闭合轴跨度=%s mm** | "
                            "**每侧余量=%s mm** | 指1触目标=%s 指2触目标=%s | q1=%.4f q2=%.4f ⇒ %s",
                            _gap_mm,
                            ("n/a" if _span_mm is None else f"{_span_mm:.1f}"),
                            ("n/a" if _span_mm is None else f"{( _span_mm - _gap_mm) / 2.0:+.2f}"),
                            _f1c, _f2c, _q1a, _q2a,
                            ("**没贴上**(跨度>指缝 ⇒ 深度不够 ✗)"
                             if (_span_mm is not None and _span_mm > _gap_mm and not (_f1c and _f2c))
                             else ("**已贴上** ⇒ 问题在挡停/持稳 ✓"
                                   if (_f1c and _f2c)
                                   else "**高度/位置不对**(该高度跨度小 ✗)")))
                    except Exception as _edm:  # noqa: BLE001
                        logger.warning("[够不着?] 测量失败: %s", _edm)

            # 闭合/维持/抬升阶段: 只设 PD target, 让手指自然收敛
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ W2 (2026-09-26) 【方案 §8.3-B】**唯一出口**做跨相位限速 ⇒ 不可能跳变 ✓ ★★★★
            #   方案实测 ✓ (t2j, 控制帧): `frame159: cmd_q1=0.007787 → frame160: 0.006769`
            #     ⇒ **单步 Δq = -1.018mm**(30Hz ⇒ 30.5mm/s) ✗ 既超 `_S1_CLOSE_RATE_MAX`
            #       (0.67mm/帧) 也超近接触段 5mm/s ✗✗
            #   病根 ✓: 闭合用 `_q_ramp`, 保持用**另一个变量** `_q_cmd_cont`(初值=名义
            #     `_close1_q[-1]`) ⇒ 两个源, 相位切换时**不继承** ⇒ 一步跳到终点 ✗
            #   本改 ✓ (不重构三个分支, 而是**在唯一出口统一限速** ✓):
            #     · 维护 `_q_last_cmd` = 上一帧**真实下发**的值 ✓
            #     · 下发前把本帧命令钳到 `|q[k] − q[k−1]| ≤ v_phase·dt` ✓
            #     · 越界 ⇒ **ERROR 记录**并钳住 ✓ ⇒ 逐帧命令序列必然单调且无跳变 ✓
            #   ⇒ 无论上游是哪个分支/是否冻结, **出口只有这一处** ✓ (命令只许一处来源 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                if frame == 0:
                    _q_last_cmd = float(stage_q_open())          # 起始 = 全开 ✓
                _vmax_cmd = float(os.environ.get("_S1_CLOSE_RATE_MAX", "0.0015") or 0.0015)
                _q_prev = float(_q_last_cmd)
                _q_now = float(gripper_target)
                _dq_cmd = _q_now - _q_prev
                if abs(_dq_cmd) > _vmax_cmd + 1e-12:
                    logger.error(
                        "[W2 命令跳变] F%d ✗ 单步 Δq=**%+.3fmm** 超过限速 %.3fmm/帧 "
                        "(%s→%s) ⇒ **钳到限速** ✓ (消除跨相位跳变 ✗)",
                        int(frame), _dq_cmd * 1000.0, _vmax_cmd * 1000.0,
                        f"{_q_prev:.6f}", f"{_q_now:.6f}")
                    _q_now = _q_prev + float(np.clip(_dq_cmd, -_vmax_cmd, _vmax_cmd))
                gripper_target = float(_q_now)
                _q_last_cmd = float(_q_now)                       # 下一帧的基准 ✓
            except Exception as _e_w2:  # noqa: BLE001
                logger.warning("[W2 命令跳变] 保护失败(按原值下发 ✓): %s", _e_w2)

            active_joints[gripper_idx1].set_drive_target(float(gripper_target))
            active_joints[gripper_idx2].set_drive_target(float(gripper_target))

            # v4.17: 虚拟关节也用纯 PD (对齐 test8), 不再 vlock teleport
            if virtual_idx:
                for vkey, vval in [('vx', current_pos[0]), ('vy', current_pos[1]),
                                   ('vz', current_pos[2]), ('rz', rz), ('ry', ry), ('rx', rx)]:
                    active_joints[virtual_idx[vkey]].set_drive_target(float(vval))

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T07c (2026-09-27) 【唯一命令下发处记录】**所有 clip/限速/保护之后** ✓ ★★★★
            #   md T07.1 原话 ✓: "记录点放在**所有 clip/限速/保护完成之后、唯一命令下发处**;
            #     每个控制周期都记录, 包括张开、闭合、冻结、稳定、提升及异常停止,
            #     **不能只记某个 else 分支**; 一次 step 完成后挂接相同 index 的 post-step
            #     observation, 明确 command→step→observation 顺序, **禁止错一帧**" ✓
            #   ⇒ 本处紧跟 W2 命令跳变保护(3596) 与虚拟关节下发(3605-3608) 之后 ✓
            #     · T_W_G: 4×4 (**含旋转** ✓ 此前只有平移 ✗) = [target_R | current_pos]
            #     · phase: travel/close/settle/lift/stopped ✓ (每周期都有值 ⇒ 不留空 ✗)
            #     · source_phase: MANO 源相位 (取得到就记, 否则 None ✓ 不编造 ✗)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                _tr_c = getattr(self, "_s1_exec_trace", None)
                if _tr_c is None:
                    _tr_c = {}
                    self._s1_exec_trace = _tr_c
                _ph_c = ("travel" if int(frame) < int(_TRAVEL_FRAMES)
                         else ("close" if int(frame) < int(_TRAVEL_FRAMES + _CLOSE_FRAMES)
                               else ("settle" if int(frame) < int(_LIFT_START_FRAME + _t8_ext_used)
                                     else "lift")))
                _src_ph_c = None
                for _k_c in ("_src_phase", "source_phase", "src_phase", "_s_src", "_mano_s"):
                    if _k_c in locals() and locals()[_k_c] is not None:
                        _src_ph_c = locals()[_k_c]
                        break
                _T_gg_c = np.eye(4, dtype=np.float64)
                _T_gg_c[:3, :3] = np.asarray(target_R, dtype=np.float64)
                _T_gg_c[:3, 3] = np.asarray(current_pos, dtype=np.float64).reshape(-1)[:3]
                _tr_c.setdefault("control_index", []).append(int(frame))
                _tr_c.setdefault("t_s", []).append(float(frame) / 30.0)
                _tr_c.setdefault("T_W_G", []).append(
                    [[float(_v) for _v in _row] for _row in _T_gg_c])
                _tr_c.setdefault("q_cmd", []).append(float(gripper_target))
                _tr_c.setdefault("phase", []).append(str(_ph_c))
                _tr_c.setdefault("source_phase", []).append(_src_ph_c)
                # 兼容旧字段名 (S2-1 的 actual_* 口径 ✓)
                _tr_c.setdefault("t", []).append(float(frame) / 30.0)
                _tr_c.setdefault("base_pos", []).append(
                    [float(_v) for _v in np.asarray(current_pos, dtype=np.float64).reshape(-1)[:3]])
                _tr_c.setdefault("q", []).append(float(gripper_target))
            except Exception as _e7c:  # noqa: BLE001
                logger.warning("[T07c] 命令记录失败(不阻断 ✓): %s", _e7c)

            physics_step(robot, arm_joint_indices, gripper_idx1, gripper_idx2,
                         np.array([]), float(gripper_target), float(gripper_target),
                         self.scene, lock_root_pose=None)

            # ★ T07c ✓ **同 index 的 post-step observation** (command→step→observation ✓)
            try:
                _qo_c = np.asarray(robot.get_qpos(), dtype=np.float64).reshape(-1)
                _oz_c = None
                for _a_c in (getattr(self, "obj_actors", []) or []):
                    if target_name in str(_a_c.get_name()):
                        try:
                            _oz_c = float(np.asarray(_a_c.get_pose().p,
                                                     dtype=np.float64).reshape(-1)[2])
                        except Exception:  # noqa: BLE001
                            _oz_c = None
                        break
                _tr_o = getattr(self, "_s1_exec_trace", None) or {}
                # ★ 错帧自检 ✓ (md: command→step→observation **禁止错一帧**):
                #   本次 observation 的 index 必须 == 最近一条命令的 index ✓
                _ci_c = int(_tr_o.get("control_index", [int(frame)])[-1])
                if _ci_c != int(frame):
                    logger.error("[T07c] ✗ **observation 错帧**: cmd_index=%d obs_index=%d "
                                 "(md 禁止 ✗)", _ci_c, int(frame))
                _tr_o.setdefault("q_act", []).append(
                    [float(_qo_c[gripper_idx1]), float(_qo_c[gripper_idx2])])
                _tr_o.setdefault("obj_z", []).append(_oz_c)
                _tr_o.setdefault("valid", []).append(
                    bool(np.all(np.isfinite(_qo_c[[gripper_idx1, gripper_idx2]]))
                         and (_oz_c is None or np.isfinite(_oz_c))))
            except Exception as _e7co:  # noqa: BLE001
                logger.warning("[T07c] post-step observation 失败(不阻断 ✓): %s", _e7co)

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-25 【张开受阻检测】**用碰撞**判定"张不开"是不是**位姿**问题 ✓ ★★★★
            #   用户定调 ✓: "张不开你要检测碰撞了, 这样你就知道到底是不是位姿有问题;
            #               张不开这种位姿就很难, **但是你不要张的太开**, 不然所有位姿都不行" ✓
            #   背景实测 ✓ (cc17/cc18): finger1 `target=0.0300` 而 `actual=0.00000` (纹丝不动 ✗),
            #     且调试行 contacts 里有 `('left_gripper_finger_link1','ground')` ✗
            #     ⇒ 要么被**桌面/物体挡住**(位姿问题 ✗), 要么**驱动/下限**问题(与位姿无关 ✓)
            #     ⇒ 不区分就会把"驱动问题"错怪成"位姿差" ✗ (我前几轮正是这样乱改 ✗)
            #   本改 ✓ (两件必须同时测):
            #     ① **跟踪残差** = |目标q − 实际q| (真残差 ✓)
            #     ② **该手指此刻正与谁接触** (用 `scene.get_contacts()` ✓ 与 Stage2 同款 ✓)
            #     判定 ✓: 残差大 + **有接触** ⇒ `OPEN_BLOCKED`(**位姿问题** ✗ 该候选应换位置)
            #             残差大 + **无接触** ⇒ `DRIVE_LIMIT`(**不是位姿问题** ✓ 不要淘汰位姿 ✓)
            #   ⚠ **只诊断/记录, 不新增硬门** ✓ ⇒ 不会把所有位姿都打死 ✗
            #   ⚠ 张开幅度**不由这里放大** ✓ (用户: 不要张太开) ⇒ 由 `_Q_OPEN` 外部控制 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if frame in (0, 1, 5, 20):
                try:
                    _q_act_ob = robot.get_qpos()
                    _ob_e1 = abs(float(_q_act_ob[gripper_idx1]) - float(gripper_target))
                    _ob_e2 = abs(float(_q_act_ob[gripper_idx2]) - float(gripper_target))
                    _ob_tol = float(os.environ.get("_S1_OPEN_BLOCK_MM", "1.5") or 1.5) / 1000.0
                    _ob_h1, _ob_h2, _ob_who = False, False, set()

                    def _ob_name(_b):
                        try:
                            _n1 = getattr(_b, "name", None)
                            if _n1:
                                return str(_n1)
                            _gn = getattr(_b, "get_name", None)
                            if callable(_gn):
                                return str(_gn())
                        except Exception:  # noqa: BLE001
                            pass
                        return ""

                    for _ct in self.scene.get_contacts():
                        try:
                            if len(_ct.bodies) < 2:
                                continue
                            _b0, _b1 = _ct.bodies[:2]
                            _n0, _n1 = _ob_name(_b0), _ob_name(_b1)
                            if "finger_link1" in _n0 or "finger_link1" in _n1:
                                _ob_h1 = True
                            if "finger_link2" in _n0 or "finger_link2" in _n1:
                                _ob_h2 = True
                            for _nn in (_n0, _n1):
                                if _nn and "finger_link" not in _nn:
                                    _ob_who.add(_nn)
                            # ★ 对方**没有名字**(实测一直 `对方=[]` ✗) ⇒ 改判**是不是机器人自己** ✓
                            #   两个 body 都带机器人部件特征 ⇒ **自接触**(手指撞到自己/另一指 ✗)
                            #   只有一个带 ⇒ **外部接触**(桌面 / 物体 ✓ 无名)
                            def _is_robot_part(_nm):
                                return any(k in _nm for k in
                                           ("finger", "gripper", "hand", "link", "left_", "right_"))
                            if _is_robot_part(_n0) and _is_robot_part(_n1):
                                _ob_who.add("**自接触**(机器人自身 ✗)")
                            elif _is_robot_part(_n0) or _is_robot_part(_n1):
                                _ob_who.add("**外部**(桌面/物体)")
                            else:
                                _ob_who.add("**外部**(无名)")
                            # ★ 对方身份: 名字取不到(实测 `对方=[]` ✗) ⇒ 用**接触点高度**判 ✓
                            #   桌面 z≈0 ⇒ z<12mm 记"桌面"; 否则"物体/其他" ✓ (不依赖名字 ✓)
                            try:
                                _pts = getattr(_ct, "points", None)
                                if _pts is not None and len(_pts) > 0:
                                    _pz = float(np.asarray(_pts[0]).reshape(-1)[2]) * 1000.0
                                    _ob_who.add(("桌面" if _pz < 12.0 else "物体/其他")
                                                + f"(z={_pz:.1f}mm)")
                            except Exception:  # noqa: BLE001
                                pass
                        except Exception:  # noqa: BLE001
                            continue
                    _ob_v1 = "OPEN_BLOCKED(位姿✗)" if (_ob_e1 > _ob_tol and _ob_h1) else (
                        "DRIVE_LIMIT(非位姿✓)" if _ob_e1 > _ob_tol else "ok")
                    _ob_v2 = "OPEN_BLOCKED(位姿✗)" if (_ob_e2 > _ob_tol and _ob_h2) else (
                        "DRIVE_LIMIT(非位姿✓)" if _ob_e2 > _ob_tol else "ok")
                    logger.warning(
                        "[张开受阻检测] F%d side=%s 目标q=%.4f | 残差 q1=%.4f(%s) q2=%.4f(%s)"
                        " | 接触 finger1=%s finger2=%s 对方=%s",
                        frame, side, float(gripper_target), _ob_e1, _ob_v1, _ob_e2, _ob_v2,
                        _ob_h1, _ob_h2, sorted(_ob_who)[:3])
                    if "OPEN_BLOCKED" in (_ob_v1, _ob_v2):
                        self._s1_open_blocked_frames = int(
                            getattr(self, "_s1_open_blocked_frames", 0)) + 1
                except Exception as _eob:  # noqa: BLE001
                    logger.warning("[张开受阻检测] 失败(不影响主流程 ✓): %s", _eob)

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
                    # ★★★★ 2026-09-21 【列错修复】这两列本是 `cmd_q1/cmd_q2` (命令 ✓)
                    #   但旧代码写的是 `qpos[gripper_idx*]` = **实际值** ✗ ⇒ 两列与
                    #   前两列(finger1_q/finger2_q)完全重复 ✗
                    #   实测铁证 ✓: `output/s2_pass2/.../gripper_trace.csv` 末帧
                    #     `finger1_q=0.0127` 而 `cmd_q1=**0.0127**` (相等 ✗)
                    #   危害 ✗: 任何"按 CSV 判手指是否被挡停"的分析都会**恒得 0** ✗
                    #     (本次实现真夹持判据时首跑就中招: 挡停量中位 0.00mm ✗)
                    #   现改为写**真正的逐帧指令** `gripper_target` ✓
                    #     (闭合段 = `_close1_q[frame]` ✓ / 之后 = `_GRIP_TARGET` ✓,
                    #      两指共用同一目标 ✓ 见 `set_drive_target(float(gripper_target))` ×2 ✓)
                    float(gripper_target), float(gripper_target),
                    # ★ W1 ✓ (方案 §8.2-5): 旧代码用**未定义的** `current_obj_z` ⇒ CSV 300 行全 NaN ✗
                    #   ⇒ 改用**本帧真实采样**的物体 z (由本循环逐帧采集的 `obj_z_history` ✓)
                    (float(obj_z_history[-1]) if obj_z_history else float("nan")),
                ))
            except Exception:  # noqa: BLE001
                pass

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【复审 ②③】抬高段**起点**开始统计桌面相关接触 ✓ ★★★★
            #   复审 ② ✓: "目标**不再由桌面支撑**" ⇒ 用「**物体↔地面**」逐子步接触占比 ✓
            #   复审 ③ ✓: "夹爪**未被桌面卡住**" ⇒ 用 P1-3 的四级分级 ✓
            #   为什么在抬高段起点重启 ✓: 闭合阶段物体**本来就该被桌面支撑** ✓ ⇒
            #     只统计 **lift→结束** 这一窗口 ✓ ⇒ 占比低才等于**真的脱离桌面** ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-24 【粗筛配套修复】接触窗口必须在**闭合起点**就开 ✓ ★★★★
            #   p9a 实测 ✗: 粗筛只有 92 帧(无抬升) ⇒ 窗口(设在抬升起点 ✓)**永远不开** ✗
            #     ⇒ 78 个候选 `双侧受力=0.00` **全部为 0** ✗ (不是物理 ✗ 是窗口没开 ✗)
            #   本改 ✓: 在**闭合起点** (`_TRAVEL_FRAMES`) 也开窗并绑定目标名 ✓ ⇒
            #     闭合/稳定阶段的**逐子步双指接触**才被记录 ✓ (粗筛正是要看这一段 ✓)
            #   注 ✓: 地面/支撑窗仍在抬升起点**重新开** (下方原有逻辑 ✓) ⇒ 语义不变 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if int(frame) == int(_TRAVEL_FRAMES):
                try:
                    from physics_env import (
                        reset_substep_ground_log as _rsg_c,
                        set_object_ground_target as _sogt_c,
                        reset_target_contact_log as _rtcl_c)
                    _sogt_c(target_name)                       # 绑定目标名 ✓
                    _rsg_c(enable=True, phase="close")         # 开窗 (闭合/稳定 ✓)
                    # ★ 2026-09-25 cc13 铁证 ✗: 冻结在 q=0.0292(全开) ⇒ 探针里**残留了别处数据**
                    #   ⇒ 必须在闭合起点**显式清空**目标接触记录 ✓ (杜绝上一轮/掌部/另一手残留 ✗)
                    _rtcl_c(enable=True, target=str(target_name))
                    logger.info("[接触窗] ✓ 已从**闭合起点** F%d 开启逐子步接触记录 ✓ "
                                "(粗筛需要这一段 ✓; 抬升起点会再开一次 ✓)", int(_TRAVEL_FRAMES))
                except Exception as _ec:  # noqa: BLE001
                    logger.warning("[接触窗] 开启失败: %s", _ec)

            if int(frame) == int(_LIFT_START_FRAME) and not getattr(
                    self, "_s1_ground_window_open", False):
                try:
                    from physics_env import (
                        reset_substep_ground_log as _rsg,
                        set_object_ground_target as _sogt)
                    _rsg(enable=True)
                    _sogt(target_name)
                    self._s1_ground_window_open = True
                    logger.info("[复审②] ✓ 已从**抬高段起点**重启「物体↔地面」统计 "
                                "(目标=%s ✓; 用于判断「目标不再由桌面支撑」✓)", target_name)
                except Exception as _er2:  # noqa: BLE001
                    logger.warning("[复审②] 桌面/物体统计启动失败: %s", _er2)

            # v4.16: 录制视频 — vk DeviceLost 保护 + 帧标注 (2026-09-06)
            # ★★ 2026-09-14 P0 修复: 静默失败 → 响亮报错 ★★
            #   实测 --test-stage1 产出 **44 字节空 MP4** (内容仅 "ftypisomisomiso2mp41 freemdat",
            #   零帧) ✗, 但调用方只判断 `_writer_* is not None` 就打印"视频已保存" → 假成功 ✗.
            #   真因: record_video=True 但 scene._render_available=False → 整个录像分支被跳过,
            #   静默 continue 了一整个 rollout. 现在遇到立即 logger.error, 且累计写帧数.
            try:
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-22 【录像修复】`record_video` 默认 **False** ✗ ⇒ 从不写帧 ✗ ★★★★
                #   实测 ✓: `output/{allfix,p03_snapshot,geom5,geom6}/stage3/*.mp4` **全为 257B** ✗
                #      (合法 mp4 头 + **空 mdat** ✓) 且**无任何 .gif** ✗
                #   本改 ✓: `_RECORD_VIDEO_ALL=1` (默认 **1** ✓) ⇒ **强制录制前 K 次 rollout**
                #     (K = `_RECORD_ROLLOUTS`, 默认 **2** ✓ —— 有上限, 不会把上百次 CEM 全录 ✗)
                #   关闭 ✓: `_RECORD_VIDEO_ALL=0` ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                if not record_video and os.environ.get("_RECORD_VIDEO_ALL", "1") == "1":
                    _cap = int(os.environ.get("_RECORD_ROLLOUTS", "2") or 0)
                    _cnt = int(getattr(self, "_s1_recorded_rollouts", 0))
                    if _cnt < _cap and getattr(self, "_writer_cam", None) is not None:
                        record_video = True
                        self._s1_recorded_rollouts = _cnt + 1
                        logger.info(
                            "[Stage1 录像] ★ **强制录制** 第 %d/%d 次 rollout ✓ "
                            "(`_RECORD_VIDEO_ALL=1` ✓; 关闭: `_RECORD_VIDEO_ALL=0` ✓)", _cnt + 1, _cap)
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
                # ★ W1b (2026-09-26) 【双侧帧必须**受力**】✗ 接触记录 ≠ 受力 ✓
                #   实测铁证 ✓ (output/d2): `双侧帧=118~131` 却 `峰值力=**0.00N**` ✗
                #     ⇒ contact_offset=1mm 会**提前**产生接触记录 ✓ ⇒ 有记录但没有压力 ✗
                #     ⇒ `both_contact_count` 一直被这个假象**灌水** ✗ 而它又喂给门与奖励 ✗
                #   ⇒ 本改 ✓: 双侧帧 = 接触**且**该控制周期内**两指同时受力**
                #     (用 W1 新增的 `both_force_steps` = 每子步左右交集计数 ✓)
                # ★ G3② ✓: 同一帧**复用**已消费的"本控制周期新增"(不再二次读取 ✗)
                _both_force_now = int(getattr(self, '_cur_both_new', 0) or 0)
                if f1_c and f2_c and target_name in contact_objs:
                    if _both_force_now > 0:
                        both_contact_count += 1
                    else:
                        # 只接触不受力 ⇒ **不计**双侧 ✓ (另记诊断量 ✓ 便于看"假双侧"有多少)
                        _both_touch_only = locals().get('_both_touch_only', 0) + 1

                # ★★★★ 2026-09-20 【真值对中】闭合+稳定结束那一刻量真实对称性 ★★★★
                #   放在 `_LIFT_START_FRAME - 1` ✓: 此刻手指已闭合到位、且还没抬升 ✓
                #   ⇒ 量到的就是"抓取瞬间两指各自离物体多远" ✓ (引擎真值 ✓)
                if frame == _LIFT_START_FRAME - 1:
                    _asym_m, _dmn_m = self._measure_grasp_symmetry(target_name, robot)
                    self._last_rollout_asym_mm = _asym_m
                    self._last_rollout_dmin_mm = _dmn_m
                    # ★★★★ 2026-09-20 【P0】指尖离桌面真值 (与对中同一帧 ✓) ★★★★
                    self._last_rollout_fgnd_mm = self._measure_finger_ground_clearance(robot)
                    # ★ 2026-09-25 【摆正】两指高差 (与对中/净空**同一帧** ✓)
                    self._last_rollout_tilt_mm = self._measure_finger_tilt_mm(robot)
                    if self._last_rollout_tilt_mm is not None:
                        logger.info("[Stage1 真值摆正] 两指**高差**=%.1fmm (越大越斜 ⇒ "
                                    "一侧够不到侧壁/另一侧插桌 ✗)",
                                    float(self._last_rollout_tilt_mm))
                    _asym_max_mm = float(os.environ.get("_S1_ASYM_MAX_MM", "6.0") or 6.0)
                    logger.info(
                        "[Stage1 真值对中] 闭合到位: 两指到物体距离差=%.1fmm, 最近侧=%.1fmm "
                        "⇒ %s (门槛 asym≤%.1fmm; 差距大 ⇒ 单指接触 ⇒ 抓不紧 ✗) | "
                        "**指尖最低点离桌面=%.1fmm** (<0 ⇒ 被桌面挡停 ✗)",
                        -1.0 if _asym_m is None else _asym_m,
                        -1.0 if _dmn_m is None else _dmn_m,
                        ("✓ 双侧贴合" if (_asym_m is not None and _asym_m <= _asym_max_mm)
                         else "✗ 偏心(单指接触)"),
                        _asym_max_mm,
                        -999.0 if self._last_rollout_fgnd_mm is None
                        else float(self._last_rollout_fgnd_mm))

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
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ 2026-09-25 【准确性校验】"先验证是否准确, 再看情况" (用户定调 ✓) ★★★★
                    #   为什么要它 ✓: 我已证明**盲调参数只会换数字、不换结论** ✗ (cc19~cc25) ⇒ 必须先
                    #     把"命令 / 执行 / 几何 / 物理"四条链**同时**打印出来对齐 ✓
                    #   四问 ✓ (一眼分清三种可能):
                    #     ① `指缝公式(10+2q)` vs **实测两指中心距** ⇒ 执行器语义对不对 ✓
                    #        (若公式说 50mm 而实测 80mm ⇒ **关节轴/映射**错 ✓ = 物理侧)
                    #     ② **物体 AABB/跨度** ⇒ "最近侧 23mm" 是否是"指缝本来就没合上"✓
                    #     ③ **物体位移** ⇒ 是不是闭合过程**把物体推走了** ✓
                    #     ④ 命令 q vs 实际 q ⇒ 是否被卡住 (与 [张开受阻检测] 互证 ✓)
                    #   ⚠ 只读不写 ✓ (纯诊断, 零行为改变 ✓)
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    try:
                        _q1o = float(_q[gripper_idx1])
                        _gap_formula = (0.010 + 2.0 * _q1o) * 1000.0      # 公式口径 ✓
                        _sp_meas = float(np.linalg.norm(
                            np.asarray(_fp1, dtype=np.float64)
                            - np.asarray(_fp2, dtype=np.float64))) * 1000.0
                        _hulls_v = self._coacd_world_hulls(target_name) or []
                        if _hulls_v:
                            _PV = np.vstack([np.asarray(_h, dtype=np.float64)
                                             for _h in _hulls_v])
                            _ext_v = (_PV.max(0) - _PV.min(0)) * 1000.0
                            _ext_s = f"{_ext_v[0]:.1f}x{_ext_v[1]:.1f}x{_ext_v[2]:.1f}mm"
                            _span_y = float(_PV[:, 1].max() - _PV[:, 1].min()) * 1000.0
                        else:
                            _ext_s, _span_y = "n/a", float("nan")
                        _od_mm = float("nan")
                        try:
                            _oa = None
                            for _a2 in getattr(self, "obj_actors", []) or []:
                                if target_name in str(_a2.get_name()):
                                    _oa = _a2
                            if _oa is not None:
                                _op_now = np.asarray(_oa.get_pose().p, dtype=np.float64)
                                _ini_map = getattr(self, "_obj_initial_poses", None) or {}
                                _op0 = None
                                if isinstance(_ini_map, dict):
                                    for _k2, _v2 in _ini_map.items():
                                        if target_name in str(_k2):
                                            _vv2 = getattr(_v2, "p", _v2)
                                            _op0 = np.asarray(_vv2, dtype=np.float64)[:3]
                                            break
                                if _op0 is not None:
                                    _od_mm = float(np.linalg.norm(_op_now - _op0)) * 1000.0
                        except Exception:  # noqa: BLE001
                            pass
                        logger.info(
                            "[准确性校验 F%d] 命令q=%.4f 实际q1=%.4f q2=%.4f (差 %.2f/%.2f mm) | "
                            "指缝公式(10+2q)=%.1fmm vs **实测两指中心距=%.1fmm** (差 %.1fmm) | "
                            "物体AABB=%s 跨度y=%.1fmm | 物体位移=%.1fmm",
                            frame, float(gripper_target), _q1o, float(_q[gripper_idx2]),
                            abs(float(gripper_target) - _q1o) * 1000.0,
                            abs(float(gripper_target) - float(_q[gripper_idx2])) * 1000.0,
                            _gap_formula, _sp_meas, _gap_formula - _sp_meas,
                            _ext_s, _span_y, _od_mm)
                    except Exception as _ev:  # noqa: BLE001
                        logger.warning("[准确性校验] 失败(不影响主流程 ✓): %s", _ev)
                    # ★★★★ 2026-09-20 【真值闭环探针】只用**引擎真实数据**, 不用任何模型 ★★★★
                    #   用户要求 (2026-09-20): "好好判定一下, **肯定需要真值闭环来进行计算**" ✓
                    #   真值来源 (全部来自引擎 ✓, 零假设 ✓):
                    #     ① 手指: `robot.get_link(...).get_pose()` + 该 link 的
                    #        `collision_shape.get_vertices()` 经 shape→link→world 变换 ✓
                    #     ② 目标: `_coacd_world_hulls()` (直接读引擎 collision shapes ✓)
                    #   判定量 ✓:
                    #     · `含点` = 目标凸包顶点落进**真实手指碰撞体世界 AABB** 的个数
                    #     · `最近` = 两顶点集之间的**真实最近距离** (mm ✓)
                    #       ⇒ `含点>0 或 最近≈0` 却仍 `force=0` ⇒ **碰撞结算/作用域**问题 ✗
                    #       ⇒ `最近` 明显 >0 ⇒ **手指压根没到物体上** ⇒ 位置/开合映射错 ✗
                    if os.environ.get("_S1_PROBE", "0") == "1" and \
                            (frame % 10 == 0 or frame == _LIFT_START_FRAME):
                        try:
                            import sapien as _sp_p
                            from sapien.pysapien.physx import (
                                PhysxArticulationLinkComponent as _ALC_p)
                            _hulls_p = self._coacd_world_hulls(target_name)
                            if _hulls_p:
                                _HV = np.concatenate(
                                    [np.asarray(h, dtype=np.float64) for h in _hulls_p], axis=0)
                                _hlo, _hhi = _HV.min(axis=0), _HV.max(axis=0)
                                # ★ 2026-09-20 物心位移也用**引擎真值**打出来 ✓
                                #   (`xy_drift` 指标实测失灵: 物体飞出画面却报 0.00cm ✗
                                #    ⇒ 这里直接读 actor pose, 与仿真完全同源 ✓)
                                _objp_p = _hlo * 0.0
                                for _a_p in self.obj_actors:
                                    if target_name and target_name in _a_p.get_name():
                                        _objp_p = np.asarray(_a_p.get_pose().p, dtype=np.float64)
                                        break
                                if _obj_p0 is None:
                                    _obj_p0 = _objp_p.copy()
                                # ★★★★ 2026-09-20 【标定】两指**真实中心距** vs 模型指缝 ★★★★
                                #   为什么必须标定 ✓ (实测非单调性铁证 ✓):
                                #     过闭合 6.5mm ⇒ F40 顶点最近 6.71mm
                                #     过闭合 0.5mm ⇒ F40 顶点最近 10.15mm
                                #     ⇒ 模型指缝变化 **+12.0mm**, 真实距离只变 **+3.44mm** ✗
                                #     ⇒ 真实/模型 ≈ **29%** ✗ ⇒ `指缝 = GAP0 + 2q` **严重高估行程** ✗
                                #       (q→指缝的传递比不是 1:1 ✗) ⇒ 所有"要压多少"的推算全偏 ✗
                                #   本项 ✓: 直接量**两指碰撞体 AABB 中心的真实距离** ⇒
                                #     与模型 `GAP0 + 2q` 并排打印 ⇒ 一眼看出传递比 ✓
                                _fdata = {}
                                for _fn in (f"{side}_gripper_finger_link1",
                                            f"{side}_gripper_finger_link2"):
                                    _lk = None
                                    for _l in robot.get_links():
                                        if _l.get_name() == _fn:
                                            _lk = _l
                                            break
                                    if _lk is None:
                                        continue
                                    _Tl = _lk.get_pose().to_transformation_matrix()
                                    _vlist = []
                                    for _comp in _lk.entity.components:
                                        if not isinstance(_comp, _ALC_p):
                                            continue
                                        for _sh in _comp.get_collision_shapes():
                                            try:
                                                _vv = np.asarray(
                                                    _sh.get_vertices(), dtype=np.float64)
                                            except Exception:  # noqa: BLE001
                                                continue
                                            _Ts = (_Tl @ _sh.get_local_pose()
                                                   .to_transformation_matrix())
                                            _vlist.append(_vv @ _Ts[:3, :3].T + _Ts[:3, 3])
                                    if _vlist:
                                        _fdata[_fn] = np.concatenate(_vlist, axis=0)
                                _keys = sorted(_fdata.keys())
                                if len(_keys) == 2:
                                    _V1, _V2 = _fdata[_keys[0]], _fdata[_keys[1]]
                                    _c1 = 0.5 * (_V1.min(0) + _V1.max(0))
                                    _c2 = 0.5 * (_V2.min(0) + _V2.max(0))
                                    _real_span_axis = float(np.linalg.norm(_c1 - _c2)) * 1000.0
                                    # ★★★★ 2026-09-20 【真正的指缝】投影法 (引擎真值 ✓) ★★★★
                                    #   上一版用"两指 **AABB 中心距**"✗ —— 它**含指体自身厚度** ✗,
                                    #     所以恒定 +58.5mm 只是"指体厚度" ✓, **不是标定偏差** ✗
                                    #     (我上次据此下的结论不严谨, 在此更正 ✓)
                                    #   本项 ✓: 把两指顶点分别投影到**闭合轴**上 ⇒
                                    #     gap = min(指2 投影) − max(指1 投影)  (取正的那一侧 ✓)
                                    #     ⇒ 这才是**手指面之间**的真实间隙 ✓ (负值 = 已互相穿透 ✓)
                                    _ax_c = np.asarray(target_R, dtype=np.float64)[:, 1]
                                    _p1 = _V1 @ _ax_c
                                    _p2 = _V2 @ _ax_c
                                    _gap_true = float(min(_p2.min() - _p1.max(),
                                                          _p1.min() - _p2.max())) * 1000.0
                                    # ★ 两指**真实最近距离** (体-体 ✓ 无任何假设 ✓) ——
                                    #   这一项才是"指缝对不对"的最终裁决 ✓
                                    _d12 = float(np.linalg.norm(
                                        _V1[::max(1, len(_V1) // 80)][:, None, :]
                                        - _V2[None, ::max(1, len(_V2) // 80), :],
                                        axis=-1).min()) * 1000.0
                                    _q_now = float(_q[gripper_idx1])
                                    _model_gap = (FINGER_GAP0 + 2.0 * _q_now) * 1000.0
                                    _dm1 = float(np.linalg.norm(
                                        _HV[:, None, :] - _V1[None, ::max(1, len(_V1) // 60), :],
                                        axis=-1).min()) * 1000.0
                                    _dm2 = float(np.linalg.norm(
                                        _HV[:, None, :] - _V2[None, ::max(1, len(_V2) // 60), :],
                                        axis=-1).min()) * 1000.0
                                    logger.info(
                                        "[Stage1 真值探针 F%d] q=%.4f | 模型指缝=%.1fmm | "
                                        "两指中心距=%.1fmm | **两指真实最近距离=%.1fmm** | "
                                        "目标顶点到指1=%.2fmm 到指2=%.2fmm | "
                                        "**物心位移=%.1fmm** (引擎真值 ✓ 与仿真同源 ✓)",
                                        frame, _q_now, _model_gap, _real_span_axis, _d12,
                                        _dm1, _dm2,
                                        float(np.linalg.norm(_objp_p - _obj_p0)) * 1000.0)
                        except Exception as _ep:  # noqa: BLE001
                            import traceback as _tb_p
                            logger.warning("[Stage1 真值探针] 失败: %s\n%s",
                                           _ep, _tb_p.format_exc())

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

        # ★★★★ 2026-09-21 【用户】"**stage1 的抓取是假的吗? 不是两侧抓取?**" ★★★★
        #   ⇒ 用**两个可判别的物理事实**回答, 而不是靠观感 ✓:
        #     ① **两指是否被物体挡停** ✓: 真夹持时 `act > cmd` (手指想闭合却被物体顶住 ✓)
        #        实测 (output/s2_pass2 抬升成功的两次) ✓✓:
        #          `cmd=0.0026` 而 `f1_qpos=0.0125→0.0127, f2_qpos=0.0082→0.0083`
        #          ⇒ 被挡停 **+10.1mm / +5.7mm** ✓ ⇒ **真反力** ✓
        #        对比"空夹"那两次 ✗: `f1_qpos=f2_qpos=**0.0026** == cmd` ⇒ **自由到达**
        #          ⇒ 零阻挡 ⇒ 假抓取 (推/托/勾) ✗
        #     ② **物体与基座同步上升** ✓: 成功时 `pos_z 0.1888→0.2336` (+45mm) 而
        #        `obj_z 0.1665→0.2113` (**+45mm**, 误差 <1mm ✓); 且 `both_contact=**144/145 帧**` ✓
        #   ⇒ 本指标 ✓: 抬升段内"**两指同时被挡停**"的帧占比 = 真两侧夹持率 ✓
        #     (tol 默认 **1.0mm** ✓; 抬升段 = `frame >= _LIFT_START_FRAME` ✓)
        #   tol 标定 ✓ (用两份已有 CSV 复算, 真指令 `_GRIP_TARGET` 对比 ✓):
        #     · 抬升成功那次 (s2_pass2, cmd=0.0026): 挡停量 指1=**+10.06mm** 指2=**+5.72mm**
        #       ⇒ 真夹持率 **100.0%** (110/110 帧) ✓✓
        #     · 空夹那次 (true_grasp, cmd=0.0025): 挡停量 +0.24/+0.23mm
        #       ⇒ 若 tol=0.3mm 会误判成 45.5% ✗ ⇒ **故 tol 定 1.0mm** ⇒ 0% ✓ 干净区分 ✓
        _blk_tol_m = float(os.environ.get("_S1_BLOCK_TOL_MM", "1.0") or 1.0) / 1000.0
        _true_ratio, _blk1_mm, _blk2_mm, _n_lift_t, _n_both_blk = 0.0, 0.0, 0.0, 0, 0
        try:
            # ⚠ 坑 ✗ (已实测): CSV 的 `cmd_q1/cmd_q2` 两列**写的其实是实际值** ✗
            #   (`output/s2_pass2/.../gripper_trace.csv` 末帧: `f1=0.0127` 而 `cmd1=**0.0127**` 相等 ✗)
            #   ⇒ 用 CSV 算"被挡停"会恒得 0 ✗ (本次首跑就是这样: 挡停量中位 0.00mm ✗)
            #   正确做法 ✓: 与**真指令** `_GRIP_TARGET` 比 (抬升段它恒定 ✓, 见
            #   `gripper_target = _GRIP_TARGET  # 保持夹紧` ✓; 且已存于
            #   `self._last_grip_target_s1` ✓)
            _grip_cmd_m = float(getattr(self, "_last_grip_target_s1", None) or 0.0)
            _lift_rows = [_r for _r in _s1_trace_rows
                          if int(_r[0]) >= int(_LIFT_START_FRAME)]
            _n_lift_t = len(_lift_rows)
            if _n_lift_t:
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-22 【复审 I.2-1 修复】合法 `q=0` 不得被**系统性否决** ✗ ★★★★
                #   复审原话 ✓: "`s1_true_grasp_ratio` 仍依赖 `_grip_cmd_m > 0.0`, q=0 时
                #     结果**固定为 0**; 但新终验默认要求该值 ≥0.5 ⇒ **合法零目标因此被系统性否决**" ✗
                #   本改 ✓ (两处都不再用 `> 0` 当真值判据 ✓):
                #     ① 执行条件 `_n_lift_t and _grip_cmd_m > 0.0` → **只看 `_n_lift_t`** ✓
                #     ② 逐帧真指令 (第 15/16 列) 的判空 `> 0.0` → **有限性检查** ✓
                #        ⇒ `q=0` 是**合法值** ✓ 不再被当成"取不到" ✗
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def _cmd_ok(_v):
                    try:
                        return bool(np.isfinite(float(_v)))
                    except Exception:  # noqa: BLE001
                        return False

                # ★ 2026-09-21 CSV 列错已修 ⇒ 优先用**逐帧真指令** (第 15/16 列 ✓);
                #   取不到 (非有限/异常) 时回退到常量 `_GRIP_TARGET` ✓ (抬升段二者相等 ✓)
                _c1 = np.array([float(_r[15]) if _cmd_ok(_r[15]) else _grip_cmd_m
                                for _r in _lift_rows])
                _c2 = np.array([float(_r[16]) if _cmd_ok(_r[16]) else _grip_cmd_m
                                for _r in _lift_rows])
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T2 (2026-09-25) 挡停量基准 ⇒ **计划的闭合意图** (对冻结免疫 ✓) ★★★★
                #   实测铁证 ✓ (output/v5): 接触冻结把**命令对齐到实际**
                #     (`冻结=0.0128` 且逐帧命令也 = 0.0128 ✗) ⇒ `挡停量 = |实际 − 命令| ≡ 0` ✗
                #     ⇒ 明明 `间隙=0.30mm` + `peak_force=**7.22N**` ✓, 却被判
                #       "**假抓取: 两指自由到达(空夹/托/勾)**" ✗✗ —— 判据对冻结**结构性免疫** ✗
                #   本改 ✓: 基准取 `min(逐帧命令, 计划目标 `_GRIP_TARGET`)` ✓
                #     · 冻结只会把命令**抬高到计划目标之外** ⇒ 取 min 即"计划想闭到哪" ✓
                #     · 实际停在冻结位(被物体挡住) ⇒ 残差 = **真实挡停量** ✓
                #   ⚠ 判别力不受影响 ✓: 真"空夹"(手指自由到达计划目标) ⇒ 残差仍 ≈0 ⇒ 仍判 0% ✓
                #     (与标定记录一致: s2_pass2 +10.06/+5.72mm 判定 100% ✓; true_grasp +0.24mm 判 0% ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _c1 = np.minimum(_c1, float(_grip_cmd_m))
                _c2 = np.minimum(_c2, float(_grip_cmd_m))
                _b1 = np.array([float(_r[12]) for _r in _lift_rows]) - _c1
                _b2 = np.array([float(_r[13]) for _r in _lift_rows]) - _c2
                _both_blk = (_b1 > _blk_tol_m) & (_b2 > _blk_tol_m)
                _true_ratio = float(_both_blk.mean())
                _n_both_blk = int(_both_blk.sum())
                _blk1_mm = float(np.median(_b1) * 1000.0)
                _blk2_mm = float(np.median(_b2) * 1000.0)
        except Exception as _etg:  # noqa: BLE001
            logger.warning("[Stage1 真夹持判据] 计算失败: %s", _etg)
        logger.info(
            "[Stage1 真夹持判据] 抬升段 %d 帧 | **两指同时被挡停** %d 帧 ⇒ **真两侧夹持率=%.1f%%** "
            "| 挡停量(中位) 指1=**%.2f**mm 指2=**%.2f**mm (tol=%.2fmm) | both_contact=%d, "
            "peak_force=%.2fN ⇒ %s",
            _n_lift_t, _n_both_blk, 100.0 * _true_ratio, _blk1_mm, _blk2_mm,
            _blk_tol_m * 1000.0, both_contact_count, peak_grip_force,
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-26 【方案 §8.4-D】**失败模式分类** (不再"笼统说没抓到" ✗) ★★★★
            #   方案原话 ✓: "用**一个固定任务候选诊断** … 分清 NO_CONTACT、UNILATERAL、SLIP、
            #     TABLE_BLOCKED、OBSERVATION_INVALID; **不因一次没抓到就改奖励/力/位置全部参数**" ✓
            #   判据全部来自**已采集的真值** ✓ (W1 之后这些量才可信 ✓):
            #     · OBSERVATION_INVALID: 物体 z 序列缺失/全空 ⇒ 根本没测到 ⇒ **不能下结论** ✓
            #     · TABLE_BLOCKED: 抬升段指尖仍低于桌面 (fgnd<0) ⇒ 被桌挡住, 不是抓不住 ✓
            #     · NO_CONTACT: 无接触帧 且 峰值力≈0 ⇒ 压根没碰到 ⇒ 看位姿/闭合行程 ✓
            #     · UNILATERAL: 有接触但**几乎无双侧帧** ⇒ 只一侧碰到 ⇒ 看对中/跨度 ✓
            #     · SLIP: 双侧接触**曾经成立**但物体没跟着抬 ⇒ 夹持力/摩擦不足 ✓
            #   ⚠ 只分类不改判据 ✓; 每轮必打 ⇒ 以后每次失败都能直接归因而非盲调 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            ("**真两侧夹持** ✓" if _true_ratio >= 0.5 else
             ("部分夹持" if _true_ratio > 0.0
              else "**假抓取: 两指自由到达(空夹/托/勾)** ✗")))
        try:
            _obs_ok = bool(len(obj_z_history) >= 2)
            _lift_mm = ((float(obj_z_history[-1]) - float(obj_z_history[0])) * 1000.0
                        if _obs_ok else float('nan'))
            _fgnd = getattr(self, '_last_rollout_fgnd_mm', None)
            if not _obs_ok:
                _fail_cls = "**OBSERVATION_INVALID**(观测缺失 ✗ 不能下结论)"
            elif (_fgnd is not None and float(_fgnd) < 0.0):
                _fail_cls = "**TABLE_BLOCKED**(指尖插桌被挡 ✗)"
            elif int(contact_count) == 0 and float(peak_grip_force) < 0.05:
                _fail_cls = "**NO_CONTACT**(全程零接触 ✗)"
            elif (int(both_contact_count) <= max(1, int(0.02 * max(1, int(_n_lift_t))))
                  or float(peak_grip_force) < 0.05):
                # ★ 用力说话 ✓: 实测 SLIP 样本 `双侧帧=118~124` 却 `峰值力=**0.00N**` ✗
                #   ⇒ contact_offset=1mm 会提前产生**接触记录** ✗ (有记录 ≠ 受力 ✓)
                #   ⇒ 只看双侧帧会把"没真压上"误判成"双侧成立" ✗ (方案 §8.2 ✓)
                #   ⇒ 本改: 双侧帧**且**峰值力达标才算双侧成立; 否则判 UNILATERAL ✓
                _fail_cls = ("**UNILATERAL**(有接触但未建立双侧受力 ✗)"
                             if int(contact_count) > 0
                             else "**NO_CONTACT**(无受力接触 ✗)")
            elif not (_lift_mm >= 30.0):
                _fail_cls = "**SLIP**(双侧受力曾成立但物体没跟着抬 ✗)"
            else:
                _fail_cls = "**OK**(已抬起 ✓)"
            logger.warning(
                "[Stage1 失败分类] %s | 抬升段=%d帧 接触帧=%d 双侧帧=%d 峰值力=%.2fN | "
                "物体升=%s mm | 指尖净空=%s mm | 真两侧夹持率=%.1f%%",
                _fail_cls, int(_n_lift_t), int(contact_count), int(both_contact_count),
                float(peak_grip_force),
                ("n/a" if not _obs_ok else f"{_lift_mm:+.1f}"),
                ("n/a" if _fgnd is None else f"{float(_fgnd):+.1f}"),
                100.0 * float(_true_ratio))
        except Exception as _efc:  # noqa: BLE001
            logger.warning("[Stage1 失败分类] 失败: %s", _efc)
        # ★ 2026-09-22 【复审 ②③】把"目标是否仍由桌面支撑 / 夹爪是否被桌面卡住"写入结果 ✓
        try:
            from physics_env import (object_ground_report as _ogr,
                                     get_substep_ground_report as _gsr3)
            _og = _ogr()
            _gg = _gsr3()
            self._last_obj_ground_report = _og
            # ⚠ 2026-09-22 修 bug ✗: 此处 `sim_result` **尚未定义** ✗
            #   (实测: `桌面真值写入失败: name 'sim_result' is not defined` ✗)
            #   ⇒ 改为存**属性** ✓, 再由结果字典 (下方字面量 ✓) 取回 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【计划 §4.3-1 修复】**未知不得降级为 False** ✗ ★★★★
            #   计划原话 ✓: "`:2371` 的 `bool(_og.get('supported'))` 把统计层的 `None`
            #     转成 False, **丢失'未知'和'确认无支撑'的区别**; 下游再检查 None 已来不及" ✗
            #   ⇒ 后果: 支撑**未知**会被当成"**不受支撑**" ⇒ 反而**更容易通过**验收 ✗✗
            #   本改 ✓: `None` 保持 `None` ✓ (与 `object_ground_report` 的三态一致 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _sup_raw = _og.get('supported')
            self._last_obj_supported = (None if _sup_raw is None else bool(_sup_raw))
            self._last_obj_supported_state = str(_og.get('state') or '')
            self._last_gripper_ground_class = str(_gg.get('classification'))
            self._last_gripper_ground_min_sep_mm = (
                None if _gg.get('min_sep_mm') is None else round(float(_gg['min_sep_mm']), 3))
            if 'sim_result' in locals() and isinstance(locals().get('sim_result'), dict):
                sim_result['s1_obj_supported'] = self._last_obj_supported
                sim_result['s1_gripper_ground_class'] = self._last_gripper_ground_class
            logger.info(
                "[复审②③] 抬高段窗口 ✓: 物体↔地面 **接触占比=%.2f** (阈值 %.2f) ⇒ "
                "**目标%s由桌面支撑** | 夹爪↔桌面 分级=**%s** (最小 separation=%s mm)",
                float(_og.get('contact_ratio', 0.0)), float(_og.get('threshold', 0.5)),
                "**仍**" if _og.get('supported') else "**不再**", _gg.get('classification'),
                ("n/a" if _gg.get('min_sep_mm') is None
                 else f"{float(_gg['min_sep_mm']):+.3f}"))
        except Exception as _er3:  # noqa: BLE001
            logger.warning("[复审②③] 桌面真值写入失败: %s", _er3)
        self._s1_ground_window_open = False      # 下一轮 rollout 重新开窗 ✓
        self._last_rollout_true_ratio = _true_ratio

        # ★ 2026-09-14 P0: 录像完整性断言 —— 不许再"静默产出 44 字节空 MP4"
        if record_video:
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【GIF 修 ②】**fail-safe finalize** ✓ ★★★★
            #   实测 ✗: `output/vidfix` 明明写了 3×160 帧 ✓, 但**没有 GIF、mp4 也没 finalize** ✗
            #     根因 ✓: Stage1 0/3 ⇒ 管线**提前 return** ✗ ⇒ 后面
            #       `writer.release()` + `transcode_to_h264()` (grasp_hawor:5456/5672/5855 ✗)
            #       **全都没执行** ✗ ⇒ mp4 无 moov atom ✗ (`moov atom not found` ✓)
            #   本改 ✓: **录满上限后立即落盘** ✓ (release → transcode → GIF ✓),
            #     不依赖后续成功路径 ✓ ⇒ 无论管线成功或失败, **都有可看的 GIF** ✓
            #   开关 ✓: `_GIF_FAILSAFE=1` (默认 1 ✓; 关闭: =0 ✓)
            if (record_video
                    and os.environ.get("_GIF_FAILSAFE", "1") == "1"
                    and int(getattr(self, "_s1_recorded_rollouts", 0))
                    >= int(os.environ.get("_RECORD_ROLLOUTS", "2") or 0)
                    and not getattr(self, "_gif_failsafe_done", False)):
                try:
                    self._gif_failsafe_done = True
                    from grasp_hawor import transcode_to_h264 as _t2h
                    for _wp in (getattr(self, "_video_path_cam", None),
                                getattr(self, "_video_path_god", None),
                                getattr(self, "_video_path_right", None)):
                        if _wp:
                            try:
                                if getattr(self, "_writer_cam", None) is not None \
                                        and _wp == getattr(self, "_video_path_cam", None):
                                    self._writer_cam.release()
                                elif getattr(self, "_writer_god", None) is not None \
                                        and _wp == getattr(self, "_video_path_god", None):
                                    self._writer_god.release()
                                elif getattr(self, "_writer_right", None) is not None \
                                        and _wp == getattr(self, "_video_path_right", None):
                                    self._writer_right.release()
                            except Exception:  # noqa: BLE001
                                pass
                            _t2h(_wp)      # ⇒ cam 视角会自动导出 GIF ✓ (§80 ✓)
                    logger.info("[录像 fail-safe] ✓ 已在录满上限后落盘+转码+GIF ✓ "
                                "(不依赖后续成功路径 ✓; 关闭: `_GIF_FAILSAFE=0` ✓)")
                except Exception as _efs:  # noqa: BLE001
                    logger.warning("[录像 fail-safe] 落盘失败: %s", _efs)
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

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T07b (2026-09-27) 【交付强绑定】每条 rollout 自带**身份 + 不可变执行记录** ✓ ★★★★
        #   md T07.1 原话 ✓: "rollout 返回 episode_id + candidate_id + immutable_record +
        #     assessment; 搜索、重复验证、winner 选择、导出**始终传递这个对象/唯一引用**;
        #     **不再凭 q 终点、最后一次缓存或'最长轨迹'找 record**" ✗
        #     (我上一版正是"按 q 终点匹配 + 最长回退" ✗ ⇒ 可能把 B 的曲线安到 A 头上 ✗)
        #   ⇒ episode_id: 全局单调唯一 (同一候选的**重复 run 也必须不同** ✓)
        #     candidate_id: 由**本 run 首个基座命令**哈希 ⇒ 同候选重复 run **相同** ✓
        #     immutable_record: **本 run** 的执行序列快照 (返回即冻结 ✓ 不被后续 rollout 覆盖 ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            self._s1_episode_seq = int(getattr(self, "_s1_episode_seq", 0)) + 1
            _s1_episode_id = "s1ep-%06d" % int(self._s1_episode_seq)
        except Exception:  # noqa: BLE001
            _s1_episode_id = "s1ep-unknown"
        # ★ T07e ✓ 取本 run 的**整数源帧号** (合同要求整数, 不是矩阵 ✗)
        _sf_idx_rollout = None
        try:
            if isinstance(grasp_pose, dict) and grasp_pose.get('source_frame') is not None:
                _sf_idx_rollout = int(round(float(grasp_pose['source_frame'])))
            else:
                # ★ T07e-2 ✓ 从**位姿→帧号映射**取回 (调用处重建 pose dict 会丢 source_frame ✗)
                _mp7 = getattr(self, "_s1_pose_src_frame", None) or {}
                try:
                    _kb7r = (tuple(np.round(np.asarray(grasp_pose.get('pos'),
                                                       dtype=np.float64), 6)),
                             tuple(np.round(np.asarray(grasp_pose.get('R'),
                                                       dtype=np.float64), 6).reshape(-1)))
                    if _kb7r in _mp7:
                        _sf_idx_rollout = int(_mp7[_kb7r])
                except Exception:  # noqa: BLE001
                    pass
                if _sf_idx_rollout is None:
                    # ★ 末级回退 ✓: 本扫描族的帧号 (CEM 扰动位姿不在精确映射里 ✗)
                    #   ⇒ 值仍是**真实 MANO 帧号**; 若与实际锚点不符, 合同不因此放行 ✓
                    _lf7 = getattr(self, "_s1_last_cand_frame", None)
                    if _lf7 is not None:
                        _sf_idx_rollout = int(_lf7)
        except Exception:  # noqa: BLE001
            _sf_idx_rollout = None
        _s1_candidate_id = "s1cand-unknown"
        try:
            import hashlib as _hl_b
            _bp0 = (getattr(self, "_s1_exec_trace", None) or {}).get("base_pos") or []
            _seed = "%s|%s|%s|%s" % (
                np.round(np.asarray(_bp0[0], dtype=np.float64), 6).tolist() if _bp0 else "na",
                str(side), str(strategy), int(n_frames))
            _s1_candidate_id = "s1cand-" + _hl_b.sha1(_seed.encode("utf-8")).hexdigest()[:12]
        except Exception:  # noqa: BLE001
            pass
        _s1_immutable_record = None
        try:
            _tr_b = getattr(self, "_s1_exec_trace", None) or {}
            _s1_immutable_record = {
                "schema_version": "s1_exec_trace/v2",          # ★ T07c ✓ 增字段 ⇒ 显式升版
                "episode_id": str(_s1_episode_id),
                "candidate_id": str(_s1_candidate_id),
                "control_index": [int(_x) for _x in (_tr_b.get("control_index") or [])],
                "t_s": [float(_x) for _x in (_tr_b.get("t_s") or _tr_b.get("t") or [])],
                "T_W_G": [[[float(_v) for _v in _row] for _row in _M]
                          for _M in (_tr_b.get("T_W_G") or [])],      # ★ 含旋转 4×4 ✓
                "q_cmd": [float(_x) for _x in (_tr_b.get("q_cmd") or _tr_b.get("q") or [])],
                "phase": [str(_x) for _x in (_tr_b.get("phase") or [])],
                "source_phase": list(_tr_b.get("source_phase") or []),
                "q_act": [list(_x) for _x in (_tr_b.get("q_act") or [])],
                "obj_z": list(_tr_b.get("obj_z") or []),
                "valid": [bool(_x) for _x in (_tr_b.get("valid") or [])],
                "base_pos": [[float(_v) for _v in _p]
                             for _p in (_tr_b.get("base_pos") or [])],
                "close_controls": int(_CLOSE_FRAMES),
                "settle_controls": int(_SETTLE_FRAMES),
                "n_frames": int(n_frames),
            }
            # ★ T07c ✓ 长度自检: command 与 post-step observation **必须同长** ✗ (错帧即告警 ✓)
            _n_cmd = len(_s1_immutable_record["q_cmd"])
            for _k_ck in ("t_s", "T_W_G", "phase", "control_index"):
                if len(_s1_immutable_record[_k_ck]) != _n_cmd:
                    logger.error("[T07c] ✗ 记录长度不一致: %s=%d vs q_cmd=%d (md 要求同长 ✗)",
                                 _k_ck, len(_s1_immutable_record[_k_ck]), _n_cmd)
            for _k_ck in ("q_act", "obj_z", "valid"):
                if len(_s1_immutable_record.get(_k_ck) or []) not in (0, _n_cmd):
                    logger.error("[T07c] ✗ observation 长度不一致: %s=%d vs cmd=%d (错帧 ✗)",
                                 _k_ck, len(_s1_immutable_record[_k_ck]), _n_cmd)
            # ★ T07e ✓ 登记表已**移到 `_s1_assessment` 之后** (此前引用未定义变量 ⇒ 登记失败 ✗)
        except Exception as _eb:  # noqa: BLE001
            logger.warning("[T07b] 执行记录快照失败(记录为 None ⇒ 导出将报 MISSING ✓): %s", _eb)
        # ★ T072 ✓ 【trace == 下发 spy】逐点核对 (证伪"记的是 clip 前值" ✗)
        try:
            from physics_env import get_dispatch_spy as _gds072
            from stage1_recipe import compare_trace_with_spy as _ctws072
            _spy072 = _gds072()
            _ok072, _det072 = _ctws072(
                [float(_x) for _x in ((_s1_immutable_record or {}).get("q_cmd") or [])],
                _spy072)
            if isinstance(_s1_immutable_record, dict):
                _s1_immutable_record["spy_match"] = bool(_ok072)
                _s1_immutable_record["spy_len"] = int(len(_spy072))
            if not _ok072:
                logger.error("[T072] ✗ **trace ≠ 下发 spy** ⇒ 记录点可能记了 clip 前值/错帧: %s",
                             _det072)
            elif int(len(_spy072)) > 0:
                logger.info("[T072] ✓ trace == 下发 spy (%d 点) ⇒ 记的是**限速后**真值 ✓",
                            len(_spy072))
        except Exception as _e072b:  # noqa: BLE001
            logger.warning("[T072] spy 核对失败: %s", _e072b)
        _s1_assessment = {
            "obj_lift": float(obj_lift_peak),
            "obj_lift_final": float(obj_lift),
            "peak_grip_force": float(peak_grip_force),
            "both_contact_count": int(both_contact_count),
            "close_controls": int(_CLOSE_FRAMES),
            "settle_controls": int(_SETTLE_FRAMES),
        }
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T07e (2026-09-27) 【episode 登记表】**按 id 取回**, 不靠猜 ✗ ★★★★
        #   实测 ✓ (s19/s20): winner 的 sim_result 到导出时**没有 immutable_record** ✗
        #     ⇒ 导出退化到"最后缓存"回退 (md 禁止 ✗) ⇒ 合同拒绝 ✗
        #   ⇒ 本改 ✓: 每条 rollout 把**完整记录**按 episode_id 登记 (有界 64 条 ✓)
        #     ⇒ 导出若在 best_output 上找不到, 可按 **winner 自己的 episode_id** 取回 ✓
        #   ⚠ 放在此处 ✓: `_s1_immutable_record` / `_s1_assessment` 都已构建完成 ✓
        #     (此前放在长度自检处 ⇒ `local variable '_s1_assessment' referenced before
        #      assignment` ✗ 实测 s22 每帧报此错 ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _reg = getattr(self, "_s1_episode_records", None)
            if _reg is None:
                _reg = {}
                self._s1_episode_records = _reg
            _reg[str(_s1_episode_id)] = {
                "record": _s1_immutable_record,
                "candidate_id": str(_s1_candidate_id),
                "assessment": _s1_assessment,
                "source_frame": _sf_idx_rollout,
            }
            if len(_reg) > 64:
                for _k_old in list(_reg.keys())[:-64]:
                    _reg.pop(_k_old, None)
            logger.info("[T07e] 登记 episode=%s | trace=%d 帧 | candidate=%s | src_frame=%s ✓",
                        _s1_episode_id,
                        len((_s1_immutable_record or {}).get("q_cmd") or []),
                        _s1_candidate_id, _sf_idx_rollout)
        except Exception as _e7e:  # noqa: BLE001
            logger.warning("[T07e] episode 登记失败: %s", _e7e)
        return {
            'episode_id': str(_s1_episode_id),              # ★ T07b ✓ 唯一 run
            'candidate_id': str(_s1_candidate_id),          # ★ T07b ✓ 候选(重复 run 共享)
            'immutable_record': _s1_immutable_record,       # ★ T07b ✓ 本 run 执行序列
            'assessment': _s1_assessment,                   # ★ T07b ✓ 本 run 评估
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
            # ★★ 任务03 (2026-09-28) 【md §03 v2 门的可算量】 ✓ ★★
            #   md ✓: "真实门检查 **目标双指载荷**、目标离桌、固定 C0 相对平移/旋转和任务进度" ✓
            #   ⚠ 诚实标注 ✗: 本管线**没有**"每帧双指法向冲量"回传探针 ⇒
            #     这里用**双侧接触帧占比**作为"目标双指载荷"的**代理量** ✓ (可复算、可解释),
            #     并在报告里明确它是代理 (v2 门判 `None`/待接真探针 ✗), 不冒充精确 ✓
            's1_dual_load_ratio': (float(both_contact_count) / float(max(_LIFT_FRAMES, 1))),
            's1_lift_frames': int(_LIFT_FRAMES),
            # ★ md §03 遗留 ✓ 真探针 (与上面"接触帧代理"**并列**报 ✓, 交叉可核对 ✓)
            's1_dual_load_impulse_ratio': (float(_v2_load_frames) / float(max(_v2_load_n, 1))),
            's1_dual_load_impulse_min': (
                None if (_v2_load_n == 0 or _v2_load_min_imp == float("inf"))
                else float(_v2_load_min_imp)),
            's1_dual_load_probe': 'sum_imp_raw (双侧真实法向冲量)',
            # md 06.4-f0: settle 段末实际下发命令 = anchor 的正确来源
            #   (f0 明令: 不能用整条 min(q) 或舍入日志替代指定 run 的终态)
            # ★ md 06.4-g2 ✓: 延迟预紧(T8)**实际额外压入的帧数** ⇒ 供 Stage2 换算成
            #   控制点数并**无损并入** settle (不因默认 10 点被截 ✗)
            's1_t8_extra_frames': int(locals().get('_t8_ext_used', 0) or 0),
            's1_settle_end_q_cmd': (float(np.asarray(_close1_q).reshape(-1)[-1])
                                    if np.asarray(_close1_q).size else None),
            's1_settle_end_index': int(max(0, int(_CLOSE_FRAMES) + int(_SETTLE_FRAMES) - 1)),
            's1_close_end_q_cmd': (float(np.asarray(_close1_q).reshape(-1)[-1])
                                   if np.asarray(_close1_q).size else None),
            's1_frozen_q_effective': (None if _s1_frozen_eff is None
                                      else float(_s1_frozen_eff)),
            # ★ md §03 遗留 ✓: C0 项真探针 (夹爪 6DOF 位姿逐帧 ✓; 无记录 ⇒ None ✗)
            's1_c0_trans_max_mm': (float(_v2_c0_t) if _v2_C0 is not None else None),
            's1_c0_rot_max_deg': (float(_v2_c0_r) if _v2_C0 is not None else None),
            's1_c0_note': 'C0 = 抬升段首帧夹爪基座位姿 (非 close 入口, 已显式标注 ✗)', 
            'obj_xy_drift': obj_xy_drift,
            'grip_stability': contact_count / max(n_frames, 1),
            'obj_z_final': obj_z_final,
            'obj_init_z': obj_init_z,
            'obj_z_history': obj_z_history,
            'peak_grip_force': peak_grip_force,
            # ★ 2026-09-22 【P0-4】把**实际执行的时长**随结果一起交付 ✓
            #   审计文档 §P0-4 ✓: "`_S1_CLOSE_FRAMES` / `_S1_SETTLE_FRAMES` 可以改变实际
            #     执行时长, 但 recipe 导出**没有传入这些实际值**, 仍采用默认 16/10 ✗
            #     ⇒ 默认配置相同时不触发, **覆盖环境变量时会破坏回放一致性** ✗"
            #   ⇒ 记录实际值 ✓, 供 recipe 导出与 Stage2 回放使用 ✓
            'close_controls': int(_CLOSE_FRAMES),
            'settle_controls': int(_SETTLE_FRAMES),
            # ★ 2026-09-22 【复审 ②③】桌面真值随结果返回 ✓ (供终验判据 ✓)
            's1_obj_supported': getattr(self, '_last_obj_supported', None),
            's1_gripper_ground_class': getattr(self, '_last_gripper_ground_class', None),
            's1_obj_ground': (getattr(self, '_last_obj_ground_report', None) or {}),
            's1_gripper_ground_min_sep_mm': getattr(
                self, '_last_gripper_ground_min_sep_mm', None),
            'locked_gripper_qpos': (locked_val1 + locked_val2) / 2.0,
            # ★★★★ 2026-09-17 夹持对称性 (用户: "夹空这个确实需要改正一下") ★★★★
            #   两指被物体挡停的开合差 |act1-act2|:
            #     夹得正 (对着平面) ⇒ box 版实测 0.0127/0.0182 ⇒ 相差 5.5mm ✓
            #     斜着卡棱          ⇒ CoACD 版实测 0.0105/0.0304 ⇒ 相差 19.9mm ✗
            #   ⇒ 对称性差 = 夹得不正 ⇒ 携带必滑脱 ✗ (CoACD 版 173.97mm ✗)
            'finger_qpos_asym': abs(float(locked_val1) - float(locked_val2)),
            # ★★★★ 2026-09-20 【P0】两指碰撞几何最低点 − 桌面 (mm; <0 = 指尖插进桌面 ✗) ★★★★
            #   桌面碰撞开启时才有意义 ⇒ 未开启时恒 None ⇒ reward 罚项自动跳过 ✓
            'finger_ground_clear_mm': getattr(self, '_last_rollout_fgnd_mm', None),
            # ★★★★ 2026-09-21 【G6 真值】两指到物体**最近侧**距离 (mm ✓) ★★★★
            #   设计 md §46.6 ✓: 用于"物体是否真在两指之间"的**硬门** ✓
            #   (此前只在 rollout 内存里, 没进 sim_result ⇒ Stage1b 的判据拿不到 ✗)
            's1_dmin_truth_mm': getattr(self, '_last_rollout_dmin_mm', None),
            'fingertip_obj_dist': fingertip_obj_dist,
            'fingertip_obj_dist_min': fingertip_obj_dist_min,
            # ★★★★ 2026-09-21 【真两侧夹持判据】见 rollout 内"真夹持判据"块 ✓ ★★★★
            #   `true_grasp_ratio` = 抬升段内"**两指同时被物体挡停**"的帧占比 ✓
            #     1.0 ⇒ 全程真夹持 ✓ ; 0.0 ⇒ 两指自由到达 (空夹/托/勾) ✗
            's1_true_grasp_ratio': _true_ratio,
            's1_block_mm': (_blk1_mm, _blk2_mm),
            's1_n_lift_frames': _n_lift_t,
            # ★★★★ 2026-09-18 CoACD 关键判据 —— 按**本候选**逐一配对 ✓ (上一版自 self 缓存
            #   做法已废弃 ✗: output/qual6 因配对错位导致 reward 爆 −24722 ✗ / 验证门 0/3 ✗) ★★★★
            #   机理已被证明有效 ✓: κ 从 0.42 → **0.756** ✓ ⇒ 搜索**会**转向几何更好的位姿 ✓
            #   本版 ✓: 判据由**本函数收到的 grasp_pose** 现算 ✓ ⇒ 必然属于这个候选 ✓
            #   来源 ✓: 共享单一来源 stage1_recipe.coacd_key_criteria (Stage2 也调它 ✓)
            #   未启用 CoACD ⇒ 返回 {} ⇒ stage1_reward 侧自动跳过 ⇒ **老路径行为完全不变** ✓
            #   ★ 2026-09-18 已是**单一代价**版本 (只留"会不会碰物体" ✓) ⇒ **默认开启** ✓
            #   历史: 曾带 c_center/c_axial 两项 ✗ ⇒ 实测 (output/qual7) 罚得太频繁
            #     ⇒ 即使权重降到 2/5/1 也把"能抬起来的解"挤掉 ✗ (验证门 0/3 ✗)
            #   现版本 ✓: 只有"几何判定会碰物体 ✗ 且 SAPIEN 没报接触"才罚 −80 ✓
            #     而正常候选 (物体 55.9mm 宽 vs 手指 130mm 开) 该项恒不触发 ✓
            #     ⇒ 对搜索几乎无干扰 ✓ (老路径行为不变 ✓)
            #   ★★ 2026-09-18 定论: **默认关闭** ✓ —— 本项被证明"层放错了" ✗ ★★
            #   实测 (output/qual8): `c_clear > 0`(会碰) 占 **15066/22599 = 67%** ✗
            #     ⇒ 频繁触发 −80 ✗ ⇒ 验证门仍 **0/3 失败** ✗ (reward −325 ✗)
            #   根因 ✓: 本量的 half_sep 用的是**夹持开度** ✗, 而 Stage1 的手指是
            #     **全开 (q_open=0.05 ⇒ 指缝 130mm)** 接近的 ✓ ⇒ 全开接近时碰不到物体 ✗
            #     ⇒ "不要碰物体"这件事**在 Stage1 这一层表达不了** ✗
            #   ⇒ 正确落点 (用户 2026-09-18 原话): "**stage2 优化轨迹肯定要有代价** ... 就是不要碰物体"
            #     ⇒ 应放进 **Stage2 的轨迹代价** ✓ (那里手指从 130mm 收到 ~56mm,
            #       接近末段才会擦碰 ✓ ⇒ 用 CoACD 几何替换 `avoid_target_contact` 的触发量 ✓)
            #   开关: `_S1_COACD_PENALTY=1` 可临时开启 ✓ (默认 0 = 关闭 ✓)
            **(self._coacd_key_fields(grasp_pose)
               if os.environ.get("_S1_COACD_PENALTY", "0") == "1" else {}),
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

        # ★★★★ 2026-09-20 【P0 reward 修正】"没夹住"就不能拿"漂亮姿态/对中"的分 ★★★★
        #   用户定策 (2026-09-20): "奖励代价肯定需要调整, 重新跑一遍" ✓
        #   实测依据 (output/final_d1, 同一份源码逐项代回):
        #     选出位姿 reward = **+6.540** ⇒ 其中
        #       lift 主导项 (L1574)            = **0**   ✗ (lift=0)
        #       gate1 双侧夹持 (L1637)         = **0**   ✗ (both_contact=0)
        #       gate2 中心抓取 (L1643)         = +0.34
        #       gate3 **姿态偏好** (L1761)     = **+6.20** ← 6.2/6.54 全在这里 ✗
        #     逐帧轨迹证明该位姿 **只有 f1 一根手指擦到物体, f2 全程没碰** ✗
        #       (F40/F50: f1_c=True, f2_c=False; obj_lift 全程 0)
        #   ⇒ 结论: reward 把"姿态漂亮但没夹住"排成了最优 ✗
        #     ⇒ CEM 只能忠实爬到它 (mean 6.30 / best 6.49, 收敛极稳 ✓)
        #   修法 ✓: 把"**双侧接触已建立**"作为拿"姿态/对中"分的**前置条件** ✓,
        #     并给"完全没碰到 (空夹到底)"一个明确罚分 ✓
        #   开关 ✓: `_S1_MIN_BOTH_CONTACT` (默认 8 帧 ✓; 0 = 关掉本门, 回到旧行为 ✓)
        _min_both = int(os.environ.get("_S1_MIN_BOTH_CONTACT", "8"))
        _grip_ok = (both_contact >= _min_both) if _min_both > 0 else True
        if (not _grip_ok) and float(both_contact) <= 0.0:
            r -= 20.0            # 空夹到底: 手指完全没碰到物体 ⇒ 重罚 (旧版这里给 +6.5 ✗)

        # ★★★★ 2026-09-20 【真值对中罚】用**引擎真值**判"是否真的双侧贴合" ★★★★
        #   用户要求 (2026-09-20): "**抓住要抓紧一点**" + "可以验证一下" ✓
        #   为什么必须用真值 ✗: `c_center`(CoACD 对中) 报 ≤2.5mm 时,
        #     真实两指到物体距离差仍有 **10mm** ✗ (实测 output/cent1: 3.49 vs 13.57mm ✓)
        #     ⇒ 一指贴、一指悬空 ⇒ **没有双侧力闭合** ⇒ 压住/滑落 ⇒ 抬不起来 ✗
        #   本罚项 ✓: `asym = |d1 − d2|` (闭合到位那一帧的引擎真值 ✓)
        #     · asym ≤ 门槛 (默认 6mm) ⇒ ✓ 双侧贴合, 不罚 ✓
        #     · asym > 门槛 ⇒ 单指接触 ⇒ **罚** (默认 −15, 按超出量线性) ⇒
        #       CEM/候选排序会自动偏向"真正居中"的位姿 ✓ (= "抓紧一点"的物理前提 ✓)
        #   开关 ✓: `_S1_ASYM_MAX_MM`(门槛) / `_S1_ASYM_PENALTY`(罚分上限, 0=关闭 ✓)
        _asym_v = getattr(self, '_last_rollout_asym_mm', None)
        _asym_lim = float(os.environ.get("_S1_ASYM_MAX_MM", "6.0") or 6.0)
        _asym_pen = float(os.environ.get("_S1_ASYM_PENALTY", "15.0") or 0.0)
        if _asym_v is not None and _asym_pen > 0.0 and _asym_v > _asym_lim:
            _excess = min(1.0, (_asym_v - _asym_lim) / 20.0)
            r -= _asym_pen * (0.3 + 0.7 * _excess)
            logger.info("[Stage1 reward] 真值偏心罚: asym=%.1fmm > %.1fmm ⇒ %.2f",
                        _asym_v, _asym_lim, -_asym_pen * (0.3 + 0.7 * _excess))
        sim_result['s1_asym_mm'] = _asym_v

        # ★★★★ 2026-09-20 【P0 根因修复】指尖插进桌面 ⇒ 闭合被桌面挡停 ⇒ 永远夹不住 ★★★★
        #   为什么必须进 reward (实测铁证 ✓): `output/h2_hzaxis_slow_0920` 里
        #     闭合轴已水平化 (竖直度 0.00 ✓)、两指确实碰到物体 (最近侧 1.8~3.1mm ✓)、
        #     峰值力也从 429~742N 降到 26~35N ✓ —— 但 `obj_lift` **1839/1839 = 0.00cm** ✗
        #   机理 ✓: 手指全长 77.4mm ≫ 物体高 39.3mm ⇒ 在物体中部闭合时指尖必然插进桌面 ✓;
        #     桌面碰撞开启 (用户硬性要求 ✓) ⇒ 指尖被桌面挡停 ⇒ 关节停在 q≈0.0081~0.0105
        #     (目标 0.0021 ✗) ⇒ 指缝 46~54mm ≫ 物体跨度 38.8mm ⇒ **夹不到物体** ✗
        #   旧代码 ✗: 该量只在 `score_grasp_quality` 的**候选排序**里用 (权重最低 ✗)
        #     ⇒ CEM 的 reward **完全看不到**它 ⇒ 优化器没有任何"抬高基座"的动机 ✗
        #   本罚项 ✓: 穿透越深罚越狠 (20mm 及以上封顶 −30 分, 与 +40 抬升主导项同量级 ✓)
        #   开关 ✓: `_S1_FGND_PENALTY` (默认 30.0, 设 0 = 关闭 ✓ 回到旧行为 ✓)
        _fg_v = sim_result.get('finger_ground_clear_mm')
        _fg_pen = float(os.environ.get("_S1_FGND_PENALTY", "30.0") or 0.0)
        # ★★★★ 2026-09-21 【P0 补强】罚项必须从"**贴桌面**"起算, 而不是"插进桌面" ✗ ★★★★
        #   实测铁证 ✓ (output/s2_diag, Stage2 settle 段真值):
        #     `指尖离桌面=**0.0mm**` (正好贴住, **不满足旧条件 `<0`** ⇒ **一分不罚** ✗)
        #     `挡停 +16.74/+5.98mm 但**最近侧=12.5mm**` ⇒ 手指**先被桌面挡住**,
        #       离物体还有 12.5mm 就停了 ⇒ **没压紧物体** ⇒ CARRY 一启动就脱 ✗ (drift 177~192mm)
        #   ⇒ 阈值改为**带通** ✓: 余量 < `_S1_FGND_MIN_MM` (默认 **3.0mm**) 即开始罚 ✓
        #     0mm ⇒ 罚满 (0.2 + 0.8×(3-0)/3 ≈ 1.0 倍 ✓); ≤ −1mm ⇒ 仍罚满 ✓
        #     (与 `score_grasp_quality` 的 `k_gnd` 新标定 (≥3mm 满分 ✓) **同一口径** ✓)
        _fg_min_mm = float(os.environ.get("_S1_FGND_MIN_MM", "0.0") or 0.0)
        if _fg_v is not None and _fg_pen > 0.0 and float(_fg_v) < _fg_min_mm:
            _fg_short = max(0.0, _fg_min_mm - float(_fg_v))          # 距门槛还差多少 ✗
            # 归一化尺度固定 20mm ✓ ⇒ `_S1_FGND_MIN_MM=0` 时与**旧行为逐位一致** ✓
            #   (旧: `min(1, -v/20)`; 新: `min(1, (-v)/20)` ✓ 相同 ✓)
            _fg_d = min(1.0, _fg_short / 20.0)
            r -= _fg_pen * (0.2 + 0.8 * _fg_d)
            logger.info(
                "[Stage1 reward] **指尖贴桌面罚** ✓: 余量=%.1fmm < %.1fmm (还差 %.1fmm) ⇒ %.2f "
                "(0.0mm 时手指会被桌面挡停 ⇒ 夹不到物体 ⇒ CARRY 必脱 ✗; 门槛 3mm 会破坏盆地 ✗ 见 md §46)",
                float(_fg_v), _fg_min_mm, _fg_short, -_fg_pen * (0.2 + 0.8 * _fg_d))
        sim_result['s1_finger_ground_mm'] = _fg_v

        # ── 可迁移 anchor 门 (v51 §3.1): 规范入口 q=0.05 时不得已与目标接触 ──
        # 否则该 anchor 在 Stage2 的规范入口会穿模弹飞物体 (T1 实测 0/3).
        if sim_result.get('entry_target_contact', False):
            r -= 20.0

        # ── 主导项: 跟随竖直提升 20cm (线性梯度贯穿全程, 每 1cm +2 分) ──
        r += 40.0 * float(min(1.0, max(0.0, lift_final) / 0.20))

        # ★★★★ 2026-09-21 【用户】"先去修改一下代价和奖励, 看看能不能优化出来 ... **至少得先碰到**" ★★★★
        #   为什么必须加 (实测铁证 ✓): 现 reward 只有两端有梯度 ✗
        #     · `lift=0` ⇒ 主导项 **0 分** (没有任何"往哪走"的信息 ✗)
        #     · 空夹到底 ⇒ 只有 −20 的**阶跃罚** ✗ (梯度为零 ⇒ CEM 拿不到方向 ✗)
        #     ⇒ 结果: 候选之间**分不出"差 1mm 接触"和"差 30mm 接触"** ✗
        #       (实测 `k_center` 中位仅 0.167、`both_contact=0` ⇒ 优化器在"盲走" ✗)
        #   本 shaping ✓ (与 lift **同向** ✓ 不打架, 都是"越接近抓稳越高分" ✓):
        #     ① `接触接近度`: 指尖中点↔物心距离越近分越高 (指数衰减 ✓)
        #        用 `fingertip_obj_dist`(引擎真值 ✓) —— Stage1 里该量随位姿单调变化 ✓
        #        ⇒ 给出**指向接触的连续梯度** ✓
        #     ② `双侧接触 bonus`: 只要碰到就立刻给正分 ✓
        #        ⇒ "先碰到"成为可优化目标 ✓ (正是用户要的第一步 ✓)
        #   开关 ✓: `_S1_REACH_BONUS`(默认 25, 0=关 ✓) / `_S1_REACH_SCALE_MM`(默认 20mm ✓)
        #           `_S1_CONTACT_BONUS`(默认 15, 0=关 ✓)
        _reach_b = float(os.environ.get("_S1_REACH_BONUS", "0.0") or 0.0)
        _reach_s = max(1e-6, float(os.environ.get("_S1_REACH_SCALE_MM", "20.0") or 20.0) / 1000.0)
        _contact_b = float(os.environ.get("_S1_CONTACT_BONUS", "0.0") or 0.0)
        # ⚠⚠ 2026-09-21 【实测警告 —— 同类坑第二次出现】 ⚠⚠
        #   首跑 (output/reach5, ±5° + shaping 开) 结果 ✗:
        #     `final f1_qpos=0.0026 == PD_target` (手指**自由到达**目标位 ⇒ 无阻挡 ✓)
        #     `both_contact=**0**` (**完全没碰到物体** ✗) 而 reward 却给到 **+33** ✗
        #     ⇒ 同配方在**关掉 shaping** 时 (output/g1_oc6_0920 / g1_full_0921) 是
        #       `both_contact≥17`、`lift_final=[0.0, 20.0, 19.8]cm → **2/3 通过**` ✓✓
        #   ⇒ 结论 ✓ (与 `coacd_sym` 那次完全同型 ✗):
        #     **"接近度/对称性"这类 shape 项会被 CEM 反向利用** ——
        #       它能在"不接触"的前提下把 `fingertip_obj_dist_min` 压小 ✓ (只要把手指
        #       挪到物体**旁边/上方**近处即可 ✗) ⇒ 于是优化器**放弃真正夹住** ✗
        #   故 ✓: **默认关闭 shaping** (`_S1_REACH_BONUS=0 _S1_CONTACT_BONUS=0`),
        #     仅在明确需要"诊断是否靠近"时手动打开 ✓ (用户: "至少得先碰到" ⇒
        #     正确做法是让它**真接触** ✓, 而不是给它一个可以作弊的近似量 ✗)
        if _reach_b > 0.0:
            # ★ 用 `fingertip_obj_dist_min` (全 rollout 的**最小**距离 ✓) 而不是末帧值 ✗:
            #   末帧在抬升 20cm 之后 ⇒ 距离会被"抬起来"这件事本身拉大 ✗ (与接触无关 ✗)
            #   最小值才真正回答"手指有没有靠近过物体" ✓ (正是"至少得先碰到"的判据 ✓)
            _dmin_r = float(sim_result.get('fingertip_obj_dist_min', 1.0) or 1.0)
            _k_reach_r = float(np.exp(-_dmin_r / _reach_s))
            r += _reach_b * _k_reach_r
            sim_result['s1_dmin_r'] = round(_dmin_r, 5)
            sim_result['s1_k_reach_r'] = round(_k_reach_r, 4)
        if _contact_b > 0.0:
            r += _contact_b * min(1.0, float(both_contact) / 10.0)

        # ★★★★ 2026-09-21 【假抓取罚】"托起来 / 勾起来"不算夹住 ⇒ 必须罚 ★★★★
        #   用户 (2026-09-21): "**stage1 的抓取是假的吗? 不是两侧抓取?**" ✓
        #   实测回答 ✓ (output/s2_pass2): 抬升成功的两次是**真两侧夹持** ✓
        #     `both_contact=144/145 帧`, 两指被挡停 `+10.1/+5.7mm` (cmd=0.0026 vs act=0.0127/0.0083),
        #     `peak_force=1.47/2.96N`, 物体与基座**同步升 45mm** ✓
        #   但**Stage1 的门只看 lift ≥ 2cm** ✗ ⇒ "托住/勾住/顶住上升"也能拿满分 ✗
        #     ⇒ 这种解到 Stage2 的 **CARRY (水平移动)** 必然掉 ✗
        #       (实测 `CARRY: T_G_O drift=**183.54mm**` ✗, 指距凸包 105→151mm 越走越远 ✗)
        #   本罚项 ✓: `lift ≥ 2cm` 但真夹持率 <  门槛 ⇒ 判为"假抓取" ⇒ **重罚** ✓
        #     ⇒ CEM 会自动偏向"**真两侧夹持**"的位姿 ✓ (= CARRY 能过的前提 ✓)
        #   开关 ✓: `_S1_REQUIRE_TRUE_GRASP`(默认 1 ✓) / `_S1_TRUE_GRASP_MIN`(默认 0.5 ✓)
        #           `_S1_FAKE_LIFT_PENALTY`(默认 30 ✓)
        # ⚠ 2026-09-21 默认改为 **0 (关闭)** ✓ —— 隔离实验 (output/iso_nofake) 证明:
        #   "假抓取罚"也会改变 CEM 地形 ⇒ 打开时 `0/3`, 关闭时 `2/3` (同一入口 ✓)
        #   ⇒ 判据保留为**可观测指标 + 可开门禁** ✓, 不作为 reward 求和项 ✓
        if os.environ.get("_S1_REQUIRE_TRUE_GRASP", "0") == "1":
            _tg_r = float(sim_result.get('s1_true_grasp_ratio', 1.0) or 0.0)
            # ★★★★ 2026-09-21 【真夹持率 → reward **主导项**】 ★★★★
            #   用户 (2026-09-21): "**继续** ... stage1 的抓取是假的吗? 不是两侧抓取?" ✓
            #   动机 ✓: 旧 reward 的主导项只有 `lift` ✗ ⇒ **"托住/勾住上升"也能拿满分** ✗
            #     (实测: 抬升成功那两次 `both_contact=144/145`, 而空夹那次 `lift=0` 但
            #      `both_contact=18` ⇒ 二值 lift 分不出"真夹"与"托起来" ✗)
            #   为什么这一项**不会被反向利用** ✓ (与 §26 的"接近度"**本质不同** ✓):
            #     · `fingertip_obj_dist_min` 可以在**不接触**时被压小 ✗ ⇒ 可作弊 ✗
            #     · `真夹持率` 要求"**两指同时被物体挡停** (`act > cmd + 1mm`)" ✓
            #       ⇒ 必须**真把物体夹在指间并挤压**才能拿到 ✓ ⇒ 物理上无法伪造 ✓
            #     · 且与 `lift` **同向** ✓ (真夹住 ⇒ 才抬得起来 ✓) ⇒ 不会像"对称性"那样打架 ✓
            #   ⚠⚠⚠ 2026-09-21 【实测回滚 —— 同一课第三遍!】默认改为 **0 (关闭)** ⚠⚠⚠
            #     隔离实验铁证 ✓ (output/iso_nofake, 与成功那两次**同一入口** `run_sweep.sh`):
            #       · 真夹持项**全关** ⇒ `lift_final=[0.0, **20.0, 19.8**]cm → **2/3 通过**` ✓✓
            #         且最终最优 = `pos=[-0.3277 -0.5202 0.0410], euler=[49.92 28.59 0.17]`
            #         (与 `s2_pass2` / `clean_g1_0921` 的成功解**完全一致** ✓)
            #       · 真夹持项**打开** (W=25) ⇒ 选出的解变成
            #         `euler=[52.65 25.65 1.25], reward=**10.3**` ⇒ **0/3** ✗ (且 reward 远低于 66 ✓)
            #     关键 ✓: `reward=10.3 ≈ 25×0.41` ⇒ 得分**全来自真夹持项、一分 lift 都没有** ✗
            #       ⇒ 不是"排序错" ✗, 而是 **CEM 根本没评估到那个好解** ✗
            #       ⇒ 加了求和项 ⇒ 改变早期 CEM 的均值/方差更新 ⇒ **把样本推离好解区域** ✗
            #     结论 ✓: **真夹持率是好指标, 但不能进 reward 的求和项** ✗
            #       正确用法 = **① 验证门** (lift 达标者还要求真夹持率 ≥ 门槛 ✓)
            #                 **② 平局裁决** (先按 lift 排序, 同档内按真夹持率排 ✓)
            #       —— 与 §26 的"对称性"结论**完全同型** (那次的教训: 对称性只能作门禁/软指标 ✗)
            #   诊断用法 ✓: 需要观察分布时手动设 `_S1_TRUE_GRASP_W=25` ✓ (仅诊断, 不用于选优 ✓)
            #     实测分布 `38.8%~60.4%` (output/tg2~tg4) ⇒ 该量**信息量大、分辨率足够** ✓
            _tg_w = float(os.environ.get("_S1_TRUE_GRASP_W", "0.0") or 0.0)
            if _tg_w > 0.0:
                r += _tg_w * _tg_r
                sim_result['s1_true_grasp_bonus'] = round(_tg_w * _tg_r, 3)
            _tg_min = float(os.environ.get("_S1_TRUE_GRASP_MIN", "0.5") or 0.5)
            _fake_pen = float(os.environ.get("_S1_FAKE_LIFT_PENALTY", "30.0") or 30.0)
            if float(lift_final) >= 0.02 and _tg_r < _tg_min:
                r -= _fake_pen * (1.0 - _tg_r / max(1e-6, _tg_min))
                logger.info(
                    "[Stage1 reward] **假抓取罚** ✓: lift=%.1fcm 但真夹持率仅 %.1f%% (< %.0f%%) "
                    "⇒ −%.2f 分 (两指未同时被挡停 ⇒ 托/勾/空夹 ✗)",
                    float(lift_final) * 100.0, 100.0 * _tg_r, 100.0 * _tg_min,
                    _fake_pen * (1.0 - _tg_r / max(1e-6, _tg_min)))
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
        # ★★★★ 2026-09-18 CoACD 关键判据惩罚 (用户: "stage1 和 stage2 虽然代价不同,
        #   但是在关键点上肯定得一样") ★★★★
        #   与 Stage2 的**同一来源**: stage1_recipe.coacd_key_criteria ✓ (共享单一来源 ✓)
        #   实测依据 (output/qual2, CoACD 64 凸包):
        #     Stage1 选出的"最优位姿": c_center = **6~14mm** ✗ / c_clear ≈ **-1mm** ✗
        #     而 Stage2 要求"双侧对齐夹住" ✗ ⇒ **一个显式优化、一个隐式要求 ⇒ 必然打架** ✗
        #   故把**同一对量**显式罚进 reward ✓ (与上面的 entry_target_contact 同款、同向 ✓,
        #     不会像 2026-09-17 回滚掉的"对称性奖励"那样与抬升项打架 ✗)
        #   数据来源: sim_result 由 rollout 侧填入 ✓; **缺失时跳过** ⇒ 老路径行为完全不变 ✓
        # ★★★★ 2026-09-18 CoACD 判"会不会碰物体" —— **只留这一条代价** ✓ ★★★★
        #   用户定调 (2026-09-18): "不是不能有代价 ... 我认为的奖励代价这种优化应该是很简单的,
        #     就是**不要碰物体**, 用 coacd 优化一下"
        #   为什么需要它 ✓: 上面的 `entry_target_contact` 罚 −80 ✓ 依赖 **SAPIEN 接触检测** ✗,
        #     而本项目已实测: CoACD 的 64 个重叠凸包下**接触对信号不可靠** ✗
        #     (f1/f2 随闭合深度**非单调**: 16→5 个 / 20→6 个 / 24→0 个 ✗)
        #     ⇒ "碰不碰物体"这件事**测不准** ✗ ⇒ 罚不到点上 ✗
        #   本项 ✓: 纯几何判定, 与引擎无关 ✓
        #     物体在**闭合轴**上的半宽 > **接近段(全开)手指半间距** ⇒ 手指必然顶到物体 ✓
        #     (`coacd_c_clear_mm` = max|支撑| − half_sep_approach, >0 即必然接触 ✓
        #      见 stage1_recipe.coacd_key_criteria ✓; ★ 2026-09-18 修正: half_sep 必须是**接近段开度**
        #      (q_open) 而非闭合开度 (q_grasp) ✓ ⇒ 否则对任何物体恒 >0 ⇒ 判据白废 ✓)
        #   与既有项的关系 ✓: **只在 SAPIEN 没报接触时才补罚** ✓ ⇒ 不重复计罚 ✓
        #   为什么删掉 c_center / c_axial ✗: 实测 (output/qual7) 它们罚得频繁 (c_center 6~14mm ✗)
        #     ⇒ 即使权重降得很小也把"能抬起来的解"挤掉了 ✗ ⇒ 与用户"主要得成功"相悖 ✓
        _cl = sim_result.get('coacd_c_clear_mm')
        if _cl is not None and float(_cl) > 0.0 \
                and not bool(sim_result.get('entry_target_contact', False)):
            r -= 80.0                                   # 与接触判据**同款同权** ✓ 不重复 ✓
        # 注意: **不能**无条件按 contact_count 罚 ✗ —— 抬升段本来就必须接触 ✓
        #   (曾误写成 -3.0×contact_count ⇒ 100+ 帧 ⇒ -300 ✗, 会直接毁掉抬升奖励 ✗)
        #   只在"抬不起来却接触很多"时罚 ⇒ 那才是"撞着拖" ✓
        if float(sim_result.get('contact_count', 0)) > 25 and lift_final < 0.05:
            r -= 60.0
        # ★★★★ 2026-09-20 【P0 修正】"抬升前接触"罚 **撤销** —— 与 v5.1 模板结构性冲突 ✗ ★★★★
        #   实测铁证 (output/s1_coacd_only_nopre_0920, **同一次 CEM 内** 的 rollout 对比):
        #     · 一个**完美抓取**: F15 起双侧接触、asym=1.4mm、力 1.51N、
        #       物体全程跟随抬升 **21.12cm**、xy_drift 2.98cm ⇒ 真值上完全成功 ✓
        #       但它的 reward ≈ +56 − 15×11 = **−109** ✗ (F15..F25 = 11 帧"抬升前接触" ✗)
        #     · 而**碰不到物体**的失败候选只拿 **−74** ✗
        #     ⇒ 奖励把"成功"排在"失败"后面 ⇒ CEM 只能忠实爬到失败解 ✗✗
        #   根因 ✓: 本罚项是给**旧模板 (含下降接近段)** 写的 —— 那时"抬升前接触" = 撞着物体走过去 ✗;
        #     而 v5.1 模板**没有下降段**: F0..F25 就是**闭合+稳定**段 ✓,
        #     这一段的接触**正是抓取本身** ✓ (不接触才叫空夹 ✗) ⇒ 罚它 = 罚成功 ✗
        #   撤销后 ✓: 成功候选 ≈ +54~+66, 失败候选 ≈ −74 ⇒ 梯度方向正确 ✓
        #   "入口即碰物体"这条真约束仍由 `entry_target_contact` (−80) 独立承担 ✓ (未受影响 ✓)
        #   计数器保留在 sim_result 里, 仅供诊断 ✓
        _appr_cc = float(sim_result.get('approach_contact_count', 0))
        if lift_final >= 0.15:
            r += 10.0  # ≥15cm 强奖励: 稳定夹持不滑落

        # ── gate 1: 双侧同时夹持 (硬约束的软化表达) ──
        # 提起且双侧接触 ≥10 帧 (0.25s): 真抓取; 接触却没提起: 压住不夹
        if both_contact >= 10 and lift_final > 0.05:
            r += 6.0
        elif both_contact > 0 and lift_final <= 0.005:
            r -= 6.0

        # ── gate 2: 夹爪中心抓取 (手指中点-物体中心距离) ──
        #   ★ 2026-09-20: **只有建立了双侧接触**才配拿这项分 ✓ (否则"空夹"也会得分 ✗)
        if _grip_ok:
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
                    # ★★★★ 2026-09-20 用户定策: 开合方向**跟随 MANO** ⇒ 本"必须水平"约束同步降级 ★★★★
                    #   理由: 水平化关掉后 Y 就是 MANO 的真实捏合轴 (|Y·g|≈0.836),
                    #     若这里仍按"_horiz=0.164"折减, Stage1 会继续把候选从 MANO 姿态
                    #     **扭向水平** ⇒ 与用户的"跟随 MANO"目标对着干 ✗
                    #   故: 本折减只在 `_GRIP_CLOSE_AXIS_HORIZONTAL=1` (左右夹策略) 时生效;
                    #     跟随 MANO 时退化为**只诊断**(不影响排序), 该 +12 分项按原始对齐度给分。
                    _horiz_policy = os.environ.get("_GRIP_CLOSE_AXIS_HORIZONTAL", "0") == "1"
                    _horiz_pen = _horiz if _horiz_policy else 1.0
                    # ★ 方案A: 翻转解 (与 MANO 闭合轴反向) 直接清零 ✓
                    if _flip:
                        _align_eff = 0.0
                        logger.warning(
                            "[Stage1 防翻转] 闭合轴 %s 与 MANO 参考反向 → **翻转解** ✗ "
                            "对齐分已清零 (原 align=%.3f)", np.round(_ax, 3).tolist(), _align)
                    else:
                        _align_eff = _align * _horiz_pen
                    _align_eff = 0.0 if _flip else _align_eff
                    # ★ 2026-09-20: 同样**只在双侧接触成立时**才给姿态分 ✓
                    #   (实测证据: 该 +6.2 正是让"漂亮但空夹"的位姿夺冠的原因 ✗)
                    _pose_bonus = 12.0 * _align_eff * (0.25 + 0.75 * _down)
                    if _grip_ok:
                        r += _pose_bonus
                    else:
                        _pose_bonus = 0.0
                    sim_result["pose_align_score"] = _align
                    sim_result["pose_down_score"] = _down
                    sim_result["pose_horiz_score"] = _horiz
                    sim_result["pose_flipped"] = bool(_flip)
                    if _vert > 0.5:
                        # 2026-09-20 修正文案: 原写"本候选姿态分已被清零"是**早期版本的残留**,
                        #   代码里从来没有清零动作 —— 实际只是上面那个 ×_horiz 的软折减。
                        #   且"跟随 MANO"策略下连折减都不做了 ⇒ 此时只报数字, 不报警。
                        if _horiz_policy:
                            logger.warning(
                                "[Stage1 反向告警] 闭合轴 %s 竖直度=%.2f (>0.5) → 偏上下夹 ✗ "
                                "姿态项被折减 ×%.2f (对齐项=%.3f) —— 仅软扣分, 不淘汰候选",
                                np.round(_ax, 3).tolist(), _vert, _horiz, _align)
                        else:
                            logger.info(
                                "[Stage1 姿态] 闭合轴 %s 竖直度=%.2f → **跟随 MANO 策略: 不做水平惩罚** "
                                "(姿态项按原始对齐度 %.3f 计分; 若需左右夹请设 "
                                "_GRIP_CLOSE_AXIS_HORIZONTAL=1)",
                                np.round(_ax, 3).tolist(), _vert, _align)
                    # 闭合轴方位角 (绕 X 的 roll 的体现) —— 用它监测"位姿一样但旋转不同" ✓
                    _az = float(np.degrees(np.arctan2(_ax[1], _ax[0])))
                    sim_result["pose_close_azimuth_deg"] = _az
                    logger.info(
                        "[Stage1 姿态] 闭合轴=%s 竖直度=%.2f 水平度=%.2f 方位角=%.1f° | "
                        "对齐目标(%s, 边长%.0fmm) align=%.3f → 有效=%.3f (+%.1f分%s) bbox=%s",
                        np.round(_ax, 3).tolist(), _vert, _horiz, _az, _src,
                        float(_pick[0]) * 1000.0, _align, _align_eff,
                        _pose_bonus,
                        "" if _grip_ok else ", **双侧接触未建立 ⇒ 本项不给分**",
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
            # ★ 2026-09-20: 由**阶跃罚**改为**分级罚** —— 恢复梯度 ✓
            #   旧写法 `if drift>0.05: r -= 8` 是阶跃 ⇒ 只要弹出去就**同一个分** ✗
            #   实测 (output/s1_coacd_only_nopre_0920): 弹射候选 drift 从 **0.13m 到 6.34m** ✗,
            #     阶跃罚把它们全压成 −8 ⇒ CEM 分不出"轻推一下"和"弹飞 6 米" ✗
            #   现写法: 0.05m 起罚 8 分, 线性加到 0.50m 的 20 分 ⇒ 越弹越差 ✓
            if xy_drift > 0.05:
                _fl = float(min(1.0, (xy_drift - 0.05) / 0.45))
                r -= 8.0 + 12.0 * _fl

        # ── 中途滑落检测: 峰值远高于终值 = 提起后又掉落 ──
        if lift_peak > 0.08 and lift_final < 0.5 * lift_peak:
            r -= 8.0

        # ── 力 sanity: 有夹持力但不爆表 ──
        # ★ 2026-09-22 【P0-2 后续】力单位修正后**必须重新标定阈值** ✗ (审计文档 §P0-2 ✓)
        #   旧值 `5 / 200` 是在**低估 8 倍**的读数下标定的 ✗ (`physics_env.get_grasp_force`
        #     原除以 `PHYSICS_TIMESTEP*DECIMATION` ✗, 已修 ✓)
        #   ⇒ 同一物理量现在读 **8 倍** ✓ ⇒ 旧阈值不再等价 ✗
        #   本改 ✓: 阈值可调 ✓ `_S1_FORCE_LO` (默认 **2.0N**) / `_S1_FORCE_HI` (默认 200N)
        #     依据 ✓: Stage1 历史好解 `peak_force` 在**旧单位**下 1.47 / 2.96N
        #       ⇒ 修正后 ≈ **12 / 24N** ✓ ⇒ 下限 2N 属宽松 ✓ (保持"真夹住 ⇒ 有力"的语义 ✓)
        #   ⚠ 真正的标定等**分项力**实测 ✓ (`physics_env.get_grasp_force_detail` 的
        #     `f_target_f1/f_target_f2` ✓, 文档 §5: 力不是越大越好 ✓, 不要沿用总峰值 ✓)
        _f_lo = float(os.environ.get("_S1_FORCE_LO", "2.0") or 2.0)
        _f_hi = float(os.environ.get("_S1_FORCE_HI", "200.0") or 200.0)
        if _f_lo < peak_force < _f_hi:
            r += 1.0

        # ★ 2026-09-20 诊断: 逐 rollout 的 reward 分解 (默认关, `_S1_REWARD_TRACE=1` 开 ✓)
        #   用途: 直接看到"成功抓取"和"失败候选"各拿了多少分, 避免再靠猜测定位罚项 ✓
        if os.environ.get("_S1_REWARD_TRACE", "0") == "1":
            logger.info(
                "[S1 reward trace] r=%.2f | lift_final=%.3f peak=%.3f drift=%.3f | "
                "both=%s contact=%s appr=%s entry=%s asym=%s c_clear=%s | "
                "dist=%.4f force=%.2f grip_ok=%s pose=%.2f",
                r, lift_final, lift_peak, xy_drift,
                both_contact, sim_result.get('contact_count'), _appr_cc,
                sim_result.get('entry_target_contact'), sim_result.get('s1_asym_mm'),
                sim_result.get('coacd_c_clear_mm'),
                dist, peak_force, _grip_ok,
                float(sim_result.get('pose_align_score', -1.0)))

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-23 【用户 + 计划 §6】把"**接触存在 / 最近侧**"纳入**搜索** ✗ ★★★★
        #   用户原话 ✓: "为什么现在 **stage1 的抬升放到 stage2** 里面呢? **stage2 是负责复现原来轨迹**的"
        #     ⇒ 分工确认 ✓: **抬升属 Stage1** ✓ (本 rollout 本来就有 20cm 抬升 ✓);
        #       Stage2 只做**复现** ✓ ⇒ 搜索必须在**这里**把"是否真的夹住"决定好 ✓
        #   病根 ✓ (实测 `output/fixf5`): 终验 0/3, 而 `[Stage1 真值对中]` 显示
        #     "两指到物体距离差=2.8mm ✓ 但 **最近侧=16.3mm** ✗" ⇒ 两指**对称但离物体 16mm** ✗
        #     ⇒ 闭合时**夹空** ✗ —— 而**搜索的 reward 完全不看这个** ✗
        #     (计划 §6 ✓: "搜索在追逐旧目标, 最后才被新门淘汰 ⇒ 多轮 CEM 都无效" ✗)
        #   本改 ✓: 在 `stage1_reward` 末尾加**接触侧约束** ✓ ⇒ 搜索**直接**被推离"空夹/托举"解 ✓
        #     · `s1_dmin_truth_mm`(最近侧) 缺测 ⇒ **重罚** −25 ✓
        #     · 最近侧 > `_S1_REWARD_DMIN_MAX_MM`(默认 **6mm**) ⇒ 按超出量线性罚 (上限 −60) ✓
        #     · `s1_obj_supported is True`(仍被桌面支撑 ⟹ 不是真抓 ✓) ⇒ −20 ✓
        #   ⚠ 只改**搜索的排序依据** ✓, 不改任何终验门 ✓ (门仍然是 `_assess_grasp` ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _sr_ = sim_result if isinstance(sim_result, dict) else {}
            _dmin_ = _sr_.get('s1_dmin_truth_mm')
            _dmax_ = float(os.environ.get('_S1_REWARD_DMIN_MAX_MM', '6.0') or 6.0)
            if _dmin_ is None:
                r -= 25.0
                logger.debug("[Stage1 reward·接触] 最近侧**缺测** ⇒ −25 ✓")
            elif float(_dmin_) > _dmax_:
                _p = min(60.0, 3.0 * (float(_dmin_) - _dmax_))
                r -= _p
                logger.debug("[Stage1 reward·接触] 最近侧=%.1fmm > %.1fmm ⇒ −%.1f ✓",
                             float(_dmin_), _dmax_, _p)
            if _sr_.get('s1_obj_supported') is True:
                r -= 20.0
                logger.debug("[Stage1 reward·接触] 目标**仍被桌面支撑** ⇒ −20 ✓ (非真抓 ✓)")
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【用户/计划 §6】**对中(asym) 也必须纳入搜索** ✓ ★★★★
            #   实测 ✓ (`output/heavy1`): 接触约束已把"最近侧"从 **16.3mm → 6.3mm** ✓✓,
            #     但 `[Stage1 真值对中] 两指到物体距离差 asym = **6.1~6.4mm**` ✗
            #     ⇒ 判"**偏心(单指接触)**" ✗ (门槛 6.0mm) ⇒ 真夹持率上不去 ✗
            #   病根同前 ✓: `asym` 此前**只打日志**(见 `_S1_ASYM_MAX_MM` ✓), **不进 reward** ✗
            #     ⇒ 搜索不知道"单指接触"差在哪 ✓ ⇒ 一直交偏心解 ✗
            #   本改 ✓: `asym > _S1_REWARD_ASYM_MAX_MM`(默认 **3.0mm** ✓) ⇒ 线性罚 (上限 −50) ✓;
            #     缺测 ⇒ −20 ✓ ⇒ 搜索被同时推往"**两指都贴上**" ✓
            #   ⚠ 只改**搜索排序依据** ✓, 不动终验门 ✓ (门仍是 `_assess_grasp` + `_S1_ASYM_MAX_MM` ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _asym_ = getattr(self, "_last_rollout_asym_mm", None)
            _amax_ = float(os.environ.get('_S1_REWARD_ASYM_MAX_MM', '3.0') or 3.0)
            if _asym_ is None:
                r -= 20.0
                logger.debug("[Stage1 reward·对中] asym **缺测** ⇒ −20 ✓")
            elif float(_asym_) > _amax_:
                _pa_ = min(50.0, 4.0 * (float(_asym_) - _amax_))
                r -= _pa_
                logger.debug("[Stage1 reward·对中] asym=%.1fmm > %.1fmm ⇒ −%.1f ✓",
                             float(_asym_), _amax_, _pa_)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【用户】"最关键还是**优化出正确的位姿**" ⇒ **指尖离桌**入搜索 ✗ ★★★★
            #   用户原话 ✓: "目前还是没有**夹到中心点**, 所以导致失败, 之前的夹取姿态更好" ✓
            #   实测 ✓ (`output/heavy2`): CEM 交付的候选是 `指尖离桌 = **−1.5 / −1.5 / −1.7mm**` ✗
            #     ⇒ **指尖插进桌面** ✗ ⇒ 与"夹住中心"直接冲突 ✓
            #   病根 ✓: 该量此前**只在终验门**里检查 (`_S1_ACC_GND_MM` ✓), **搜索看不到** ✗
            #     ⇒ 与前两项(最近侧/asym)完全同型 ✓ ⇒ 用同一处理方式补上 ✓
            #   本改 ✓: `指尖离桌 < _S1_REWARD_GND_MIN_MM`(默认 0mm ✓) ⇒ 罚 (越插越重, 上限 −50) ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _gc_ = _sr_.get('finger_ground_clear_mm')
            _gcmin_ = float(os.environ.get('_S1_REWARD_GND_MIN_MM', '0.0') or 0.0)
            if _gc_ is None:
                r -= 20.0
                logger.debug("[Stage1 reward·离桌] 指尖离桌**缺测** ⇒ −20 ✓")
            elif float(_gc_) < _gcmin_:
                _pg_ = min(50.0, 10.0 + 20.0 * (float(_gcmin_) - float(_gc_)))
                r -= _pg_
                logger.debug("[Stage1 reward·离桌] 指尖离桌=%.2fmm < %.2fmm (插桌 ✗) ⇒ −%.1f ✓",
                             float(_gc_), _gcmin_, _pg_)
        except Exception as _erc_:  # noqa: BLE001
            logger.warning("[Stage1 reward·接触] 约束计算失败: %s", _erc_)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-23 【用户明确要求】三级**硬淘汰** ✓ (不是扣分 ✗) ★★★★
        #   用户原话 ✓: "你**首先得碰到**, 然后再谈**夹住**, **都没碰到的可以直接淘汰了**,
        #     然后**碰飞的物理参数也要淘汰**, 然后才能得到最终的效果" ✓
        #   ⇒ 与之前(全部做成"扣分")的差别 ✓: 扣分可能被其它正奖励**补回来** ✗ ⇒ 必须**硬淘汰** ✓
        #   实现 ✓ (直接压到极低, 使 CEM 排序必然排在合格解之后 ✓):
        #     ① **没碰到** (最近侧 > `_S1_REJECT_NOCONTACT_MM`, 默认 5mm ✓) 或 真夹持率 = 0
        #        ⇒ `min(r, −500)` ✓ **淘汰** ✓
        #     ② **碰飞** (物体 xy 漂移 > `_S1_REJECT_FLY_MM`(默认 150mm ✓) 或
        #        `obj_lift_final` > `_S1_REJECT_FLY_LIFT_M`(默认 0.35m ✓, 乱飞 ✓))
        #        ⇒ `min(r, −300)` ✓ **淘汰** ✓
        #     ③ 其余 ⇒ 按"接触/对中/离桌/持稳"排序 ✓ ⇒ 才进入"夹住"的竞争 ✓
        #   ⚠ 这些阈值都可配 ✓, 且**只影响搜索排序** ✓, 不动终验门 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _sr2_ = sim_result if isinstance(sim_result, dict) else {}
            _dmin_h = _sr2_.get('s1_dmin_truth_mm')
            _tg_h = _sr2_.get('s1_true_grasp_ratio')
            _dft_h = _sr2_.get('obj_xy_drift')
            _lift_h = _sr2_.get('obj_lift_final')
            _nc_max = float(os.environ.get('_S1_REJECT_NOCONTACT_MM', '5.0') or 5.0)
            _no_contact = (_dmin_h is None) or (float(_dmin_h) > _nc_max)
            _ejected = False
            _fly_mm = float(os.environ.get('_S1_REJECT_FLY_MM', '150') or 150) / 1000.0
            _fly_lift = float(os.environ.get('_S1_REJECT_FLY_LIFT_M', '0.35') or 0.35)
            if _dft_h is not None and abs(float(_dft_h)) > _fly_mm:
                _ejected = True
            if _lift_h is not None and abs(float(_lift_h)) > _fly_lift:
                _ejected = True
            if _no_contact or (_tg_h is not None and float(_tg_h) <= 0.0):
                logger.debug("[Stage1 淘汰①] 没碰到 (最近侧=%s > %.1fmm 或 真夹持率=%s) ⇒ 淘汰 ✓",
                             _dmin_h, _nc_max, _tg_h)
                r = min(float(r), -500.0)
            elif _ejected:
                logger.debug("[Stage1 淘汰②] **碰飞** (drift=%s / lift=%s) ⇒ 淘汰 ✓",
                             _dft_h, _lift_h)
                r = min(float(r), -300.0)
        except Exception as _erej:  # noqa: BLE001
            logger.warning("[Stage1 淘汰] 判定失败: %s", _erej)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-23 【复核指令 · 奖励**分级**】不再用"总分加减项"✗ ★★★★
        #   复审原话 ✓: "**不建议继续往现有总分里加罚项** … 改成'**先分级, 再在同级内比较代价**'" ✓
        #   分级表 ✓ (与本函数上部"硬淘汰"一致 ✓):
        #     T1 无效/初始穿透/异常失稳            ⇒ **不允许交付** ✓ (base −1000 ✓)
        #     T2 **未建立双侧目标接触**             ⇒ base **+100** − 两指到目标面的距离(引导靠近 ✓)
        #     T3 建立接触, 但抬起/保持失败          ⇒ base **+500** − (滑移/旋转/接触丢失 ✓)
        #     T4 **脱桌后稳定保持 + 完成规定运动**   ⇒ base **+1000** − (偏移/平滑 ✓) ⇒ 排在失败解之前 ✓
        #   关键约束 ✓ (复审三条 ✓):
        #     ① **抬升高度不再主导** ✓ (T4 只要求"完成规定提升"作为**门槛** ✓, 不再越高越好 ✓)
        #     ② "**飞出去**"与"**没夹住**"**分开计分** ✓ (T1/T2/T3 分离 ✓)
        #     ③ 不再把 峰值力 / 行程对称 / 接触次数 / 指尖中点近物心 / 单次最大提升 当成功奖励 ✓
        #        (它们仍在 sim_result 里**仅供诊断** ✓)
        #   ⇒ 本分级**统一用于** Top-K / CEM elite / 重复确认 / 终验 ✓ (本函数是所有排序的共同入口 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _s3_ = sim_result if isinstance(sim_result, dict) else {}
            _tg3 = _s3_.get('s1_true_grasp_ratio')
            _sup3 = _s3_.get('s1_obj_supported')
            _lift3 = _s3_.get('obj_lift_final')
            _dft3 = _s3_.get('obj_xy_drift')
            _dmin3 = _s3_.get('s1_dmin_truth_mm')
            _gnd3 = _s3_.get('finger_ground_clear_mm')
            _cls3 = str(_s3_.get('s1_gripper_ground_class') or '')
            _lift_min_cm = float(os.environ.get('_S1_TIER_LIFT_CM', '3.0') or 3.0)
            _drift_max_mm = float(os.environ.get('_S1_TIER_DRIFT_MM', '20.0') or 20.0)
            _tg_min = float(os.environ.get('_S1_TIER_TG_MIN', '0.5') or 0.5)

            # ★ 2026-09-24 修正 ✓: `_invalid` 只表示**缺测** ✗ 不再混入 `obvious_penetration`
            #   (p7b 实测 ✗: 穿透候选走了 `_invalid` 分支 ⇒ base=-1000 ⇒ 我的穿透梯度分支
            #     **永远不可达** ✗ ⇒ 分层必须各归各 ✓: 缺测=-1000 / 穿透=连续深度罚 ✓
            #     交付侧两者都仍被 `_assess_grasp` 否决 ✓ 不放行 ✓)
            _invalid = ((_dmin3 is None) or (_tg3 is None) or (_lift3 is None))
            _touch = (not _invalid) and (float(_dmin3) <= float(
                os.environ.get('_S1_REJECT_NOCONTACT_MM', '5.0') or 5.0))
            _bilateral = (not _invalid) and (float(_tg3) >= _tg_min)
            _lifted = (not _invalid) and (float(_lift3) * 100.0 >= _lift_min_cm)
            _off_table = (_sup3 is False)
            _stable = (not _invalid) and (_dft3 is not None
                                          and abs(float(_dft3)) * 1000.0 <= _drift_max_mm)

            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【方案 §6.2 实施】失败**必须有梯度**, 不能全压成一个常数 ✗ ★★★★
            #   方案原话 ✓: \"无接触候选**不交付**, 但**可以保留连续失败代价用于找到接触**;
            #     把它们**全部设成同一个 -500/-1000 会丢失改进方向**\" ✗✗
            #   我的实测铁证 ✓ (本轮): `_NO_GRIPPER_GROUND=0`(方案要求桌面碰撞可见 ✓) 之后
            #     `tier=1 × 101` ✗ (手指蹭桌 ⇒ 一律 -1000 ✗ ⇒ CEM 无从改进 ✗);
            #     而早前用\"禁止夹爪接地\"绕过时 ⇒ 立刻 `mean 233→350, best 375` ✓
            #   ⇒ 说明缺的就是**梯度** ✓, 不是\"必须把碰撞关掉\" ✗
            #   本改 ✓ (两分):
            #     ① `_invalid` (缺测/NaN/明显穿透) ⇒ **仍 -1000** ✓ (真无效不许学 ✓)
            #     ② `load_bearing_block` (手指/夹爪被桌面承载 ✓) ⇒ **降为连续代价** ✓:
            #        base **-200** − 净空不足罚 (每 1mm 罚 10 分, 上限 150 ✓)
            #        ⇒ 越贴近桌面越高分 ✓ (CEM 会主动把位姿抬高 ✓)
            #        ⚠ **交付仍被拦** ✓: `_assess_grasp` 里 `load_bearing_block` 仍判不可行 ✗
            #          ⇒ 本改动**只影响搜索排序**, 不影响验收 ✓ (方案 §6.2 / §6.4 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if _invalid:
                _tier, _base = 1, -1000.0        # 真无效 ⇒ 不允许学, 不允许交付 ✓
                # ★ 2026-09-24 【用户要求"抓紧完成"】tier=1 必须说清**缺哪一项** ✓
                #   (实测 p6a: 1154 个 rollout 全是 tier=1 ✗ 而日志不写缺项 ⇒ 无法定位 ✗;
                #    最近侧有值 ⇒ 缺的是 tg/lift 之一 ⇒ 打出来才能修上游 ✓)
                logger.error("[Stage1 tier-invalid] ✗ **缺测项**: dmin=%s | tg=%s | lift=%s | "
                             "ground_class=%s ⇒ 该候选**不能学** ✗ (先修上游数据来源 ✓)",
                             _dmin3, _tg3, _lift3, (_cls3 or None))
            elif _cls3 in ('obvious_penetration',):
                # ★★★ 2026-09-24 【p7a 决定性诊断 ⇒ 修正】 ★★★
                #   p7a 实测 ✓: tier=1 的真正原因**不是缺测** ✗, 而是
                #     `ground_class=obvious_penetration` (手指插桌 >0.5mm ✓)
                #     且统计: 23/25 例 tg=0.0, 25/25 lift=0.0 ⇒ 候选**还没碰到物体就插桌** ✗
                #   ⇒ 该档原来是 -1000 **无梯度** ✗ ⇒ CEM 学不会"把手指抬离桌面" ✗
                #     (与 §6.2 "全设成同一个 -1000 会丢失改进方向" 同型 ✗)
                #   本改 ✓: **连续深度罚** (每 0.5mm 罚 10 分, 上限 150 ✓) ⇒
                #     插得越浅分越高 ⇒ CEM 有方向可学 ✓
                #   ⚠ **交付仍被拦** ✓: `_assess_grasp` 里 `obvious_penetration`
                #     依旧判不可行 ✓ (方案 §11.3: 不允许穿透冒充夹持 ✓)
                _tier, _base = 1, -200.0
                logger.error("[Stage1 tier-invalid] ✗ **明显穿透**(插桌) ⇒ 连续深度罚 "
                             "(min_sep=%.3fmm ✓ GJK 真实间隙=%.3fmm) ⇒ 不可交付 ✓ 但可学 ✓",
                             (float(_s3_.get('finger_ground_min_sep_mm') or 0.0) * 1000.0
                              if _s3_.get('finger_ground_min_sep_mm') is not None else -1.0),
                             (_dmin3 if _dmin3 is not None else float('nan')))
            elif _cls3 in ('load_bearing_block',):
                _tier, _base = 1, -200.0         # ★ 可学但不可交付 ✓ (连续代价见下 ✓)
            elif _bilateral and _lifted and _off_table and _stable:
                _tier, _base = 4, 1000.0         # 成功级 ✓
            elif _touch or (_tg3 is not None and float(_tg3) > 0.0):
                _tier, _base = 3, 500.0          # 接触但未完成 ✓
            else:
                _tier, _base = 2, 100.0          # 尚未建立接触 ⇒ 引导靠近 ✓
            _sub = 0.0
            if _tier == 4:
                _sub = (abs(float(_dft3)) * 1000.0) * 2.0 \
                    + (1.0 - min(1.0, float(_tg3))) * 50.0
            elif _tier == 3:
                _sub = (0.0 if _dmin3 is None else float(_dmin3) * 10.0) \
                    + (0.0 if _dft3 is None else abs(float(_dft3)) * 1000.0) \
                    + (1.0 - min(1.0, max(0.0, float(_tg3 or 0.0)))) * 100.0
            elif _tier == 2:
                _sub = 0.0 if _dmin3 is None else float(_dmin3) * 10.0   # 越近越好 ✓
            elif _tier == 1 and _cls3 in ('load_bearing_block', 'obvious_penetration'):
                # ★ 方案 §6.3 `J_clear` ✓: **净空不足量/穿透深度**的连续代价 ✓
                #   ⇒ CEM 能看到\"抬高一点分就高一点\" ✓ ⇒ 有梯度可学 ✓
                #   (`_gnd3` = `finger_ground_clear_mm` ✓, 负=穿透 ✗)
                _need_mm = float(os.environ.get('_S1_TIER_CLEAR_MM', '1.0') or 1.0)
                if _gnd3 is None:
                    _sub = 25.0                       # 缺测 ⇒ 轻罚 ✓ (与 §6.1 缺测不可交付一致 ✓)
                elif _cls3 == 'obvious_penetration':
                    _sub = min(150.0, 20.0 * max(0.0, -float(_gnd3)))  # 插得越深罚越重 ✓
                else:
                    _sub = min(150.0, 10.0 * max(0.0, _need_mm - float(_gnd3)))
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-24 【新方案·奖励公式】(复审给定 ✓ 不再整轮扫权重 ✓) ★★★★
            #   R = 100×成功 + 20×提升完成度 + 10×真实双侧接触比例
            #       − 10×归一化相对滑移代价 − 5×末端桌面支撑比例
            #   ✓ "真实双侧接触比例" = min(f1,f2) 的**受力**子步占比
        #     (来自新分部件探针 ✓ 掌部不参与 ✓ 复审点名 ✗)
            #   ✓ 不再奖励最近距离 / 峰值力 / 抬得越高越好 ✗
            #   ✓ tier/base/sub 保留为**交付判定**与诊断 ✓ (交付门不变 ✓)
            #   ⚠ "相对滑移"暂用 `obj_xy_drift/0.05mm` 归一 ✗ (严格相对滑移待 §5.2 观测量 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                from physics_env import get_target_contact_report as _gtcr
                _tcr = _gtcr()
                _bil_raw = _tcr.get("bilateral_force_ratio")
                _bil = float(_bil_raw) if _bil_raw is not None else 0.0
                _palm_touch = bool(_tcr.get("palm_touching"))
            except Exception:  # noqa: BLE001
                _bil, _palm_touch = 0.0, False
            _success = bool(_bilateral and _lifted and _off_table and _stable)
            _lift_done = (0.0 if _lift3 is None
                          else min(1.0, max(0.0, float(_lift3) / 0.20)))
            _slip = (0.0 if _dft3 is None
                     else min(1.0, abs(float(_dft3)) / 0.05))
            _end_sup = 1.0 if _sup3 is True else 0.0
            r = (100.0 * float(_success) + 20.0 * _lift_done + 10.0 * _bil
                 - 10.0 * _slip - 5.0 * _end_sup)
            # ★★★ 2026-09-24 【p8a 实测修正】公式**只该用于已接触候选** ✗ ★★★
            #   p8a 实测 ✓: CEM 收敛到 **R=−5 平台** (全部候选未接触 ⇒ 各项全 0 ⇒ 无梯度 ✗)
            #   复审方案的分工 ✓: 公式用于"已确认两指接触的候选"的**预紧对比** ✓;
            #     **从零搜索**仍需要旧分的"引导靠近"梯度 ✓ (tier2/最近侧项 ✓)
            #   本改 ✓: **未建立任何接触** (碰=False 且 tg=0) ⇒ 用旧分层分 (可学 ✓);
            #     **已接触** ⇒ 用新公式 R ✓ (两项取其一定义域 ✓ 交付门不变 ✓)
            if os.environ.get("_S1_SCREEN_SCORE", "0") == "1":
                # ★ 2026-09-24 【用户定策】粗筛打分 ✓: 只看**夹持质量** (双侧受力+对称) ✓
                #   (无抬升窗口 ⇒ 不能用 R 的"成功/提升完成"项 ✗)
                _asym = getattr(self, "_last_rollout_asym_mm", None)
                _sym = (1.0 if _asym is None
                        else max(0.0, 1.0 - min(1.0, abs(float(_asym)) / 5.0)))
                r = 100.0 * _bil + 20.0 * _sym - 10.0 * _slip
                logger.info("[Stage1 粗筛分] 双侧受力=%.2f(+%.1f) 对称=%.2f(+%.1f) 滑移=%.2f(−%.1f) "
                            "⇒ **分=%.1f** (粗筛模式 ✓ 只看夹持质量 ✓ 掌部接触=%s)",
                            _bil, 100.0 * _bil, _sym, 20.0 * _sym, _slip, 10.0 * _slip,
                            float(r), _palm_touch)
            elif (not bool(_touch)) and (_tg3 is None or float(_tg3) <= 0.0):
                # ★★★★ 2026-09-25 【摆正代价】两指**高差** ⇒ 连续罚分 (CEM 才有"摆正"梯度 ✓)
                #   实测铁证 (cc19/cc21): f1.z=0.0237 / f2.z=0.0559 ⇒ 高差 32mm ✗
                #     ⇒ 上指越过顶面够不到侧壁 / 下指插桌被卡张不开 ✗ ⇒ 力限再大也无效 ✓
                #   ⚠ **只加连续代价, 不新增硬门** ✓ (否则所有位姿都被打死 ✗)
                _tilt3 = getattr(self, "_last_rollout_tilt_mm", None)
                if _tilt3 is not None:
                    try:
                        _tilt_w = float(os.environ.get("_S1_TILT_W", "6.0") or 6.0)
                        _sub = float(_sub) + min(200.0, _tilt_w * float(_tilt3))
                    except Exception:  # noqa: BLE001
                        pass
                r = float(min(r, float(_base - _sub)))
                logger.info("[Stage1 奖励公式] 未接触 ⇒ 回退旧分层分 (%.1f) ✓ 保留靠近梯度 ✓",
                            float(r))
            logger.info("[Stage1 奖励公式] 成功=%s(+%.0f) 提升完成=%.2f(+%.1f) "
                        "双侧受力=%.2f(+%.1f) 滑移=%.2f(−%.1f) 末端支撑=%.1f(−%.1f) "
                        "⇒ **R=%.1f** (tier=%d 旧分=%.1f 仅诊断 ✓ 掌部接触=%s)",
                        _success, 100.0 * float(_success), _lift_done, 20.0 * _lift_done,
                        _bil, 10.0 * _bil, _slip, 10.0 * _slip, _end_sup, 5.0 * _end_sup,
                        float(r), _tier, float(_base - _sub), _palm_touch)
            # ⚠ 我上一轮用 `logger.debug` ✗ ⇒ 日志级别下**完全看不到** ✗
            #   (自审: 实测最近一轮 `grep "Stage1 分级"` = **0 次** ⇒ 我无法验证它 ✗)
            #   ⇒ 改为 **INFO/汇总** ✓ (每轮能看到 tier 分布 ⇒ 可验证 ✓)
            logger.info("[Stage1 分级] tier=%d base=%.0f sub=%.1f ⇒ r=%.1f "
                        "(碰=%s 双侧=%s 抬起=%s 脱桌=%s 稳=%s | 最近侧=%s asym=%s 离桌=%s)",
                        _tier, _base, _sub, r, _touch, _bilateral, _lifted,
                        _off_table, _stable, _dmin3,
                        getattr(self, "_last_rollout_asym_mm", None),
                        _gnd3)
        except Exception as _etr:  # noqa: BLE001
            logger.warning("[Stage1 分级] 计算失败 (保留原分): %s", _etr)
        return float(r)

    # === _restore_stage1_object_poses (2026-09-20 新增) ===
    def _restore_stage1_object_poses(self, reason=""):
        """把物体复位到**基准位姿** —— 每次 Stage1 尝试开始前调用 ✓.

        为什么必须 ✓ (2026-09-20 实测):
          Stage1 验证门不通过会**重跑 CEM** (最多 2 轮), 而每轮都会做大量物理 rollout;
          失败的 rollout 会把物体**甩飞 13~19cm** ✗, 而下一轮的
          Stage0 候选生成 (基于 `obj_bbox_centers`) 与 Stage1a 的 CoACD 几何量
          (基于 actor 当前位姿) **都不复位** ✗ ⇒ 两者基准不一致 ⇒ 判据全废 ✗.
          实测 (output/cctrav3, 同一次运行): 第1轮 闭合轴范围=[−18.8,+18.8]mm ✓,
          第2轮 [+130.4,+169.2]mm ✗, 第3轮 [+127.8,+166.5]mm ✗
          ⇒ κ 0.54 → 0.026 ⇒ 所有候选被判"会碰物体" ✗.
        复位基准 = `self._obj_initial_poses` (在**首次** rollout 前捕获 ⇒ 天然是干净位姿 ✓);
        尚未捕获 (第 1 轮) ⇒ 物体本来就干净, 直接跳过 ✓.
        """
        _ip = getattr(self, '_obj_initial_poses', None)
        if not _ip:
            return 0
        _n = 0
        for _a in getattr(self, 'obj_actors', []) or []:
            _p = _ip.get(_a.get_name())
            if _p is None:
                continue
            try:
                _a.set_pose(_p)
                for _c in _a.get_components():
                    if hasattr(_c, 'set_linear_velocity'):
                        _c.set_linear_velocity([0, 0, 0])
                        _c.set_angular_velocity([0, 0, 0])
                        break
                _n += 1
            except Exception as _e:  # noqa: BLE001
                logger.warning("[Stage1 复位] %s 复位失败: %s", _a.get_name(), _e)
        if _n:
            logger.info("[Stage1 复位] %s 已把 %d 个物体复位到基准位姿 ✓ "
                        "(保证候选基准与判据基准一致 ✓)", reason, _n)
        return _n

    # === cem_grasp_pose_optimize (L5621) ===
    def cem_grasp_pose_optimize(self, side, n_iterations=15, population_size=32,
                                  top_k=8):
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T9 (2026-09-25) 【规模可调】单轮验证必须能压到"**分钟级**" ✗ ★★★★
        #   用户疑问 ✓: "为什么有那么多仿真内存, 你到底在运行什么, 一次运行有多久?"
        #   实测铁证 ✓ (本机, output/v9):
        #     · 一次 run = `15 iter × 32 种群 = **480 次 rollout**` ✗
        #     · 单次 rollout 实测 **30~45 秒** (300 帧 × 480Hz 物理 × decim16 子步
        #       + 逐帧探针/GJK/接触查询 ✓)
        #     · ⇒ **一轮 ≈ 2~5 小时** ✗✗ 而我用 `nohup ... & sleep 400` 只等 **6.7 分钟**
        #       ⇒ 每次实验都留下**仍在跑**的进程 ⇒ 累积 **24 个** ⇒ 内存 13/15GB ✗ (爆内存 ✓)
        #   本旋钮 ✓: `_S1_CEM_ITERS` / `_S1_CEM_POP` 可缩小规模做**功能验证** ✓
        #     例: `_S1_CEM_ITERS=1 _S1_CEM_POP=2` ⇒ 单轮 **1~2 分钟** ✓ (内存/时间均可控 ✓)
        #   ⚠ **默认完全不变** ✓ (不设环境变量 ⇒ 仍是 15/32 ⇒ 正式搜索语义一致 ✓)
        #   ⚠ 覆盖时会打 WARNING ✓ ⇒ 避免"用缩小的规模当成正式结论" ✗
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _ci = int(float(os.environ.get("_S1_CEM_ITERS", "0") or 0))
            _cp = int(float(os.environ.get("_S1_CEM_POP", "0") or 0))
            if _ci > 0:
                n_iterations = _ci
            if _cp > 0:
                population_size = max(2, _cp)
            if _ci > 0 or _cp > 0:
                logger.warning("[规模覆盖] CEM iters=%d pop=%d (默认 15/32) ⇒ "
                               "**仅用于功能验证** ✓; 正式搜索请勿缩小 ✗",
                               int(n_iterations), int(population_size))
        except Exception as _ecm:  # noqa: BLE001
            logger.warning("[规模覆盖] 失败(用默认 15/32 ✓): %s", _ecm)
        # ★★★★ 2026-09-18 【位姿遍历】Top-K 可调 (用户: "姿态的选择权应该是很多的,
        #   你还没有遍历完找到最优, 可能是测试太慢了") ★★★★
        #   背景 ✓: 几何判据修好之后 (c_clear 用全开半间距), 7533 个候选里
        #     有 ~6% (≈1400 个) 几何上是"能包住 + 对中 + 轴向到位"的**好位姿** ✓
        #     但 `top_k=8` 一次只物理验证 8 个 ⇒ **遍历不到最优** ✗
        #   本旋钮 ✓: `_STAGE1_TOP_K` 直接放大进入 Stage1b 的候选数 ✓
        #     物理成本线性增长 (单次 rollout ≈0.2s ⇒ 48 个 ≈10s, 完全可接受 ✓)
        #   默认 8 ⇒ 行为与历史完全一致 ✓
        # ★★★★ 2026-09-20 【一键回退】`_S1_LEGACY=1` = 完全回到"我改动之前"的行为 ★★★★
        #   用户关心 (2026-09-20): "你执行的时候是不是就和原来的有差别呢? 差别会不会很大"
        #   ⇒ 是, 而且**不小** ✓ (见下表). 本开关把本轮所有"默认值变更"一次性还原 ✓,
        #     用来做 A/B 对照, 量化"我到底改动了多少" ✓.
        #   还原清单 (只在未显式设置时生效 ✓ ⇒ 你显式给的 env 永远优先 ✓):
        #     _S1_SCORE_SOURCE=bbox     候选排序回 bbox 四项分 (我改成 CoACD κ 主导)
        #     _S1_CLEAR_REF=close       c_clear 回"闭合开度"语义 (我改成接近段全开)
        #     _STAGE1_CEM_REF=source    CEM 参考回"锚点源帧" (我改成 anchor)
        #     _STAGE1_MANO_REF=f50      硬约束参考回 F50 (我默认也是 f50, 此项仅显式化)
        #     _STAGE1_REPEAT=1          单次评估 (我默认 1, 此项仅显式化)
        #     _S1_CENTER_MAX_MM=0       关偏心硬门; _S1_AXIAL_MAX_MM=0 关轴向门;
        #     _S1_Z_GAP_MIN_MM=0        关"基座高于物顶"门
        #     _S1_LEGACY_GROUND=1       桌面检查回"**始终**用有限 1m×1m AABB"
        #     _S1_NO_OBJ_RESET=1        不做每次尝试前的物体复位
        _legacy_s1 = os.environ.get("_S1_LEGACY", "0") == "1"
        if _legacy_s1:
            for _k, _v in (
                ("_S1_SCORE_SOURCE", "bbox"),
                ("_S1_CLEAR_REF", "close"),
                ("_STAGE1_CEM_REF", "source"),
                ("_STAGE1_MANO_REF", "f50"),
                ("_STAGE1_REPEAT", "1"),
                ("_S1_CENTER_MAX_MM", "0"),
                ("_S1_AXIAL_MAX_MM", "0"),
                ("_S1_Z_GAP_MIN_MM", "0"),
                ("_S1_LEGACY_GROUND", "1"),
                ("_S1_LEGACY_CLOSE_TARGET", "1"),
                ("_S1_NO_OBJ_RESET", "1"),
            ):
                os.environ.setdefault(_k, _v)
            logger.warning("[Stage1] **LEGACY 模式** (_S1_LEGACY=1): 已把本轮所有默认值变更"
                           "一次性还原 ⇒ 行为 == 我改动之前的版本 ✓ (A/B 对照用)")
        top_k = int(os.environ.get("_STAGE1_TOP_K", str(top_k)))
        # ★★★★ 2026-09-20 【耗时旋钮】CEM 的迭代数/种群可调 ★★★★
        #   用户问 (2026-09-20): "跑一次 stage1 需要那么久吗" ✓
        #   实测 (output/final_d1, 单次 rollout 中位 0.172s):
        #     一轮 Stage1 ≈ 177s (3 分钟), 其中 **CEM 480 个样本 ≈ 90s = 一半** ✗
        #     ⇒ 想快速看趋势就 `_STAGE1_CEM_ITER=5 _STAGE1_CEM_POP=16` (约 1/6 时间 ✓)
        #     ⇒ 想跑完整精度就保持默认 15×32 ✓
        n_iterations = int(os.environ.get("_STAGE1_CEM_ITER", str(n_iterations)))
        population_size = int(os.environ.get("_STAGE1_CEM_POP", str(population_size)))
        # 2026-09-06 (用户): 固定随机种子 — CEM 可复现; 种子随重试轮次偏移 (轮间不同, 轮内一致)
        import random as _random
        _cem_seed = 42 + 1000 * int(getattr(self, '_cem_retry_count', 0))
        np.random.seed(_cem_seed)
        _random.seed(_cem_seed)
        logger.info(f"[Stage1] CEM 随机种子 = {_cem_seed} (固定, 可复现)")
        # ★★★★ 2026-09-20 【P0 鲁棒性】每次尝试开始前**复位物体** ★★★★
        #   上一轮失败的 rollout 会把物体甩飞十几厘米 ✗, 而本轮 Stage0 候选
        #   (基于 `obj_bbox_centers`) 与 Stage1a 几何判据 (基于 actor 当前位姿)
        #   **都不复位** ⇒ 基准不一致 ⇒ 判据全废 (实测 κ 0.54 → 0.026 ✗).
        #   放在 Stage0 之前 ⇒ 候选生成与判据都拿到同一份干净场景 ✓.
        if os.environ.get("_S1_NO_OBJ_RESET", "0") != "1":
            self._restore_stage1_object_poses(
                reason=f"(第 {int(getattr(self, '_cem_retry_count', 0)) + 1} 轮, 种子 {_cem_seed})")
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
        # ★★★★ 2026-09-21 【P0 单一来源】导出的 `gripper_qpos` 必须是**真实命令过的**值 ★★★★
        #   为什么 ✓: `gripper_qpos` 会被 4 处消费 ——
        #     ① `best_grasp.npz`  ② `recipe.q_grasp`  ③ `recon["grasp_anchor"].gripper_qpos`
        #     ④ Stage3 的 `grasp_pose["gripper_qpos"]` (stage2_reconstruct_mixin.py:417/912/1046)
        #   若导出 nominal (不含过闭合) ✗:
        #     · Stage2/3 会按 nominal 闭合 ⇒ 比 Stage1 验证时少压 6mm ⇒ **夹空** ✗
        #     · 且 recipe 与 anchor 不一致 ⇒ `stage2_reconstruct_mixin.py:571`
        #       直接抛 `ValueError: Stage2 close command does not match the Stage1 recipe` ✗
        #       (实测 g1_fix_0921: 配方改了而 anchor 没改 ⇒ Stage2 **崩溃** ✗)
        #   修法 ✓: 统一取 `_last_grip_target_s1` (= rollout 里真正下发的 PD_target,
        #     含过闭合 ✓; 由闭合段 `self._last_grip_target_s1 = float(_GRIP_TARGET)` 写入 ✓)
        #     ⇒ 4 处**同源** ⇒ 下游复现 Stage1 的同一深度 ✓ 校验自然通过 ✓
        _gq_final = float(getattr(self, "_last_grip_target_s1", computed_gripper_qpos)
                          or computed_gripper_qpos)
        logger.info(f"[Stage1] 目标 {target_name}, bbox={obj_bbox_size}, "
                    f"computed_gripper_qpos={computed_gripper_qpos:.4f} "
                    f"⇒ **导出用 gripper_qpos={_gq_final:.4f}** (含过闭合 ✓ 4 处同源 ✓)")

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
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-23 【用户要求】姿态限制 **5° → 20°** ✓ ★★★★
                #   用户原话 ✓: "**我觉得对于位姿的限制, 可以达到 20 度, 不是原来的 5 度**,
                #     目前看上去效果可能不好实现" ✓
                #   ⇒ 目的 ✓: 给搜索**更大的姿态自由度** ⇒ 才可能找到"**夹到中心**"的位姿 ✓
                #     (现有 5° 太紧 ⇒ `最近侧` 一直 15~18mm ✗ ⇒ 说明搜索**够不到**正确构型 ✓)
                #   本改 ✓: 阈值改为可配 `_S1_ROT_TOL_DEG` (默认 **5.0** ⇒ 不改变既有行为 ✓;
                #     实验用 **20** ✓)
                # ⚠ 这里只放宽**候选筛选** ✓; 搜索跨度另有 `_STAGE1_ROT_SPAN_DEG` ✓ (那处也要一起放宽 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                rot_ok = abs(angle_diff) <= np.radians(
                    float(os.environ.get("_S1_ROT_TOL_DEG", "5.0") or 5.0))
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
            # ★ 2026-09-21: 记录**本次 rollout 真正下发的闭合目标** (含过闭合 ✓)
            #   ⇒ 导出 `gripper_qpos` 时按"获胜候选自己的值"取 ✓ (不是函数开头读的陈旧值 ✗)
            sim_result['gripper_qpos_target'] = float(
                getattr(self, '_last_grip_target_s1', 0.0) or 0.0)
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
        # ★★★★ 2026-09-20 【P0 作用域一致性】"夹爪↔邻物"在物理里是**关闭**的 ⇒ 这里也不许筛 ★★★★
        #   依据 (代码即真源 ✓): `stage2_reconstruct_mixin.py:104/127`
        #     · `stage1_target_isolated` 矩阵: **gripper↔non_target OFF** / target↔non_target OFF
        #       / gripper↔ground(默认环境) **ON** ✓
        #     · `DISABLED_CONTACT_PAIRS = {gripper↔non_target, target↔non_target}` ✓
        #   实测症状 ✗ (output/cost1, cost2): `被**障碍物穿透**剔除=**3213**` ⇔ 恰好等于
        #     "通过 MANO 的候选总数" ⇒ **每一个候选都被邻物判死** ✗ ⇒ `通过=0` ✗
        #     ⇒ 只能走 fallback ("无候选通过硬约束, 放宽条件取 q_score 最高的 Top-K") ✗
        #     ⇒ Stage1 的位姿筛选项**形同虚设** ✗
        #   本质 ✓: Stage1a 在**筛一个物理上不存在的碰撞对** ✗ (越界 ✓)
        #     ⇒ 若按它筛, 任何"手指靠近邻物"的合理解都会被冤杀 ✗
        #   修法 ✓: 默认**不筛邻物** (与作用域一致 ✓); 需要对照时 `_S1_CHECK_NON_TARGET=1` ✓
        _s1_check_neighbors = os.environ.get("_S1_CHECK_NON_TARGET", "0") == "1"
        _obst_s1 = []                       # (center(3,), half(3,)) 障碍物 AABB 列表
        if _n5_avail:
            # ① ~~桌面~~ ★★★★ 2026-09-20 【P0 几何一致性】删掉"有限 1m×1m 桌面 AABB" ★★★★
            #   依据 (docs/2026-09-20-桌面与CoACD物体全管线跑通建议.md §A.2) ✓:
            #     · 物理里的桌面是 `scene.add_ground()` = **无限水平面** ✓
            #     · 而这里过去是一块中心 [0,0]、半宽 **0.5m** 的有限 AABB ✗
            #     · 目标物在 **y ≈ −0.494m**, 几乎就压在该 AABB 的 −0.5m 边缘上 ✗
            #       ⇒ 手指伸出 AABB 之外时几何判"无碰撞" ✓, 而 PhysX 的无限地面**照样接触** ✗
            #     · 且本 AABB **无条件**加入 ✗ —— 即使 `_NO_GRIPPER_GROUND=1`
            #       (夹爪↔地面碰撞已关) 依然在筛 ⇒ 两套世界模型不一致 ✗
            #   修法 ✓: 桌面改用**无限平面的 signed clearance** (见下面 `_plane_clear`),
            #     并且**只在桌面碰撞真正开启时**才检查 ✓ (服从碰撞作用域真源 ✓)
            #   ② 其他物体 (non-target) —— 保持 AABB (它们是有限刚体 ✓)
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
        # ★ 桌面碰撞是否真的开启 (唯一真源 = 碰撞作用域的 env ✓)
        _table_on_s1 = os.environ.get("_NO_GRIPPER_GROUND", "0") != "1"
        # ★ LEGACY (`_S1_LEGACY=1` / `_S1_LEGACY_GROUND=1`): 还原"**始终**加有限 1m×1m 桌面 AABB" ✗
        #   (与物理的无限地面不一致, 且 table off 时也照样筛 —— 这就是我要修的那个 bug ✓)
        if os.environ.get("_S1_LEGACY_GROUND", "0") == "1":
            _obst_s1.append((np.array([0.0, 0.0, _ground_z_s1 - 0.05], dtype=np.float64),
                             np.array([0.5, 0.5, 0.05], dtype=np.float64)))
            _table_on_s1 = False
        _ground_margin_m = float(os.environ.get("_S1_GROUND_MARGIN_MM", "1.0") or 0.0) / 1000.0
        # 掌部/基座代理半尺寸 (保守球近似: 基座下方 30mm 内不允许有桌面 ✓)
        _PALM_PROXY_R = float(os.environ.get("_S1_PALM_PROXY_M", "0.030") or 0.030)

        def _obb_plane_lowest(center, R, half, ground_z):
            """OBB 到**无限水平面**的最低点 (精确解析式 ✓), 返回 clearance (m, <0 = 穿透 ✓).

            clearance = center_z − Σ_j |R[2,j]|·half_j − ground_z
            (对无限平面无需 SAT: 沿 z 的支持函数就是 Σ|R[2,j]|·half_j ✓)
            """
            _R = np.asarray(R, dtype=np.float64)
            _h = np.asarray(half, dtype=np.float64)
            _support_z = float(np.sum(np.abs(_R[2, :]) * _h))
            return float(center[2]) - _support_z - float(ground_z)

        logger.info("[Stage1 碰撞预筛] non-target AABB=%d 个, N-5 几何可用=%s | "
                    "桌面(无限平面)=%s (margins %.1fmm)",
                    len(_obst_s1), _n5_avail,
                    "**开启** ✓" if _table_on_s1 else "关闭 (不筛) ✓",
                    _ground_margin_m * 1000.0)
        # ★★★★ 2026-09-18 MANO 硬约束的参考姿态 (`_STAGE1_MANO_REF`) ★★★★
        #   `f50`   = 历史行为: 相对 **MANO F50** 姿态 ≤5° ✓ (默认, 不变 ✓)
        #   `frame` = 相对**该候选自己的源帧**姿态 ≤5° ✓
        #     依据: 用户 (2026-09-14) "F25→F50 ±5° 是**每个候选都能调整 ±5 度**,
        #       不是只有 F50 是 ±5 度" ✓ ⇒ 多帧扫描 (F35~F65) 才有意义 ✓
        #     否则离 F50 >5° 的源帧**全部被判违规** ✗ ⇒ 位姿选择权塌缩 ✗
        _mano_ref_mode = os.environ.get("_STAGE1_MANO_REF", "f50")
        scored_candidates = []
        for cand in candidates:
            q_score, hard_ok, details = self.score_grasp_quality(cand, target_name)
            _ref_cand = cand.get('ref_R') if _mano_ref_mode == "frame" else None
            mano_ok = _check_mano_hard_constraint(
                cand['pos'], cand['R'],
                ref_R_for_constraint=(_ref_cand if _ref_cand is not None else mano_f50_R))
            # ★ 障碍物穿透检查 (两指 OBB vs non-target AABB + **无限桌面平面** ✓)
            _obs_pen = 0.0
            _gnd_clear = None
            if _n5_avail:
                try:
                    _Rc = np.asarray(cand['R'], dtype=np.float64)
                    _pc = np.asarray(cand['pos'], dtype=np.float64)
                    _hg = _FINGER_HALF_GAP + float(stage_q_open(
                        float(max((getattr(self, "obj_info", {}) or {}).get(
                            target_name, {}).get("bbox_size", [0.04, 0.04, 0.04])))))
                    _min_clear = None
                    for _sg in (+1.0, -1.0):
                        _lk = _pc + _Rc @ np.array([0.03689, -_hg * _sg, 0.0])
                        _lc2 = _FINGER_LOCAL_CENTER.copy()
                        _lc2[1] *= _sg
                        _bc = _lk + _Rc @ _lc2
                        # ① non-target 物体 —— **默认不筛** ✓ (物理里该碰撞对已关闭 ✓)
                        #   只有显式 `_S1_CHECK_NON_TARGET=1` 才启用 (对照实验用 ✓)
                        if _s1_check_neighbors:
                            for (_oc, _oh) in _obst_s1:
                                _d = obb_aabb_signed(_bc, _Rc, _FINGER_HALF_EXTENT, _oc, _oh)
                                if _d < 0.0:
                                    _obs_pen = max(_obs_pen, -_d)
                        # ② 桌面: **只量余量, 不再硬剔除** ✓
                        #   ★★★★ 2026-09-20 【用户定策】"夹爪最好是远离桌面 (至少在 stage1),
                        #     但**物体在夹爪中心的代价更重要**" ★★★★
                        #   旧行为 ✗: `_min_clear < margin` ⇒ hard_ok=False
                        #     ⇒ 实测把 **3213 个通过 MANO 的候选全部判死** ✗ (通过=0 ⇒ 走 fallback ✗)
                        #     ⇒ "桌面"这一条**压过**了"对中/夹得正" ✗ (与用户优先级相反 ✗)
                        #   新行为 ✓: 桌面余量只作为 `score_grasp_quality` 里的
                        #     **软代价 k_gnd (权重 0.10, 最低 ✓)**, 由对中 k_center(0.40) 主导 ✓
                        if _table_on_s1:
                            _c_f = _obb_plane_lowest(_bc, _Rc, _FINGER_HALF_EXTENT, _ground_z_s1)
                            _min_clear = _c_f if _min_clear is None else min(_min_clear, _c_f)
                    # ③ 掌部/基座代理 (只量余量 ✓)
                    if _table_on_s1:
                        _c_p = float(_pc[2]) - _PALM_PROXY_R - _ground_z_s1
                        _min_clear = _c_p if _min_clear is None else min(_min_clear, _c_p)
                        _gnd_clear = _min_clear
                except Exception:  # noqa: BLE001
                    _obs_pen = 0.0
            if _obs_pen > 0.0:
                hard_ok = False            # 手指穿进 **non-target 物体** ⇒ 该候选直接淘汰 ✓
                details['coacd_screen'] = details.get('coacd_screen') or 'REJECT(穿邻物)'
            passed = hard_ok and mano_ok
            details['ground_clear_mm'] = (None if _gnd_clear is None
                                          else round(float(_gnd_clear) * 1000.0, 3))
            scored_candidates.append({
                'cand': cand,
                'q_score': q_score,
                'hard_ok': hard_ok,
                'mano_ok': mano_ok,
                'obs_pen_mm': float(_obs_pen) * 1000.0,
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

        # ★★★★ 2026-09-18 【诊断】位姿遍历全景 —— 回答"到底有多少可选位姿" ★★★★
        #   用户判断 (2026-09-18): "我觉得**姿态的选择权应该是很多的**, 你还没有遍历完找到最优,
        #     可能是测试太慢了" ✓
        #   ⇒ 本统计把**全部候选**按剔除原因分层计数 + 给出 CoACD 几何分布 ✓
        #     (纯统计, 不改变任何决策 ✓ 成本 = 一次线性扫描 ✓)
        #   ⇒ 用来判断: 是"位姿真的很少" ✗, 还是"筛选把位姿杀光了" ✓
        try:
            _n_all = len(scored_candidates)
            _n_mano_bad = sum(1 for sc in scored_candidates if not sc['mano_ok'])
            _n_obs_bad = sum(1 for sc in scored_candidates
                             if sc['obs_pen_mm'] > 0.0 and sc['mano_ok'])
            _n_hard_other = sum(1 for sc in scored_candidates
                                if (not sc['hard_ok']) and sc['mano_ok']
                                and sc['obs_pen_mm'] <= 0.0)
            _n_pass = sum(1 for sc in scored_candidates if sc['passed'])
            # ★ 2026-09-20 CoACD 硬门的剔除明细 (回答"到底有多少位姿真的夹得正") ✓
            _rej_center = sum(1 for sc in scored_candidates
                              if str(sc['details'].get('coacd_screen', '')).startswith('REJECT(偏心'))
            _rej_axial = sum(1 for sc in scored_candidates
                             if str(sc['details'].get('coacd_screen', '')).startswith('REJECT(轴向'))
            _rej_clear = sum(1 for sc in scored_candidates
                             if str(sc['details'].get('coacd_screen', '')).startswith('REJECT(全开'))
            _rej_zgap = sum(1 for sc in scored_candidates
                            if str(sc['details'].get('coacd_screen', '')).startswith('REJECT(基座'))
            _rej_jaw = sum(1 for sc in scored_candidates
                           if str(sc['details'].get('coacd_screen', '')).startswith('REJECT(物体不在夹持带'))
            _n_coacd_scored = sum(1 for sc in scored_candidates
                                  if sc['details'].get('score_source') == 'coacd')
            _mano_ok_scs = [sc for sc in scored_candidates if sc['mano_ok']]

            def _metric(key):
                _v = [sc['details'].get(key) for sc in _mano_ok_scs]
                _v = np.array([float(x) for x in _v if x is not None], dtype=np.float64)
                return _v[np.isfinite(_v)] if _v.size else _v

            _cl = _metric('coacd_c_clear')
            _cc = _metric('coacd_c_center')
            _ca = _metric('coacd_c_axial')
            _ka = _metric('coacd_kappa')
            logger.info(
                "[Stage1 位姿遍历] 候选总数=%d | 被**MANO 5°硬约束**剔除=%d ✗ | "
                "被**障碍物穿透**剔除=%d ✗ | 被其他硬筛剔除=%d | **通过=%d** ✓",
                _n_all, _n_mano_bad, _n_obs_bad, _n_hard_other, _n_pass)
            _ctr_gate_med = float(np.median(_metric('coacd_ctr_max_mm'))) \
                if _metric('coacd_ctr_max_mm').size else float('nan')
            # ★ 2026-09-20 桌面 clearance 分布 (回答"到底差多少毫米才不撞桌" ✓)
            _gc = _metric('ground_clear_mm')
            if _gc.size:
                logger.info(
                    "[Stage1 桌面余量] 通过 MANO 的 %d 个候选中: 夹爪最低点余量 "
                    "min/中位/max=%+.2f/%+.2f/%+.2fmm | **>= %.1fmm 的占比=%.1f%%** ✓",
                    _gc.size, float(_gc.min()), float(np.median(_gc)), float(_gc.max()),
                    _ground_margin_m * 1000.0,
                    100.0 * float((_gc >= _ground_margin_m * 1000.0).mean()))
            logger.info(
                "[Stage1 CoACD硬门] 参与 CoACD 评分=%d 个候选 | 只留 3 条**物理不可行**硬门 ✗: "
                "全开包不住=%d / 轴向(>%.1fmm)=%d / 物体不在夹持带内=%d | "
                "偏心(>%.1fmm)=%d (**宽松门**, 主要靠代价排序 ✓) "
                "⇒ **几何可行位姿 = %d 个** ✓",
                _n_coacd_scored, _rej_clear,
                float(os.environ.get("_S1_AXIAL_MAX_MM", "10.0") or 0.0), _rej_axial,
                _rej_jaw,
                _ctr_gate_med, _rej_center,
                max(0, _n_coacd_scored - _rej_center - _rej_axial - _rej_clear - _rej_jaw))
            # ★ 2026-09-20 软代价分布 (回答"对中 vs 离桌面 谁主导") ✓
            _kc = _metric('s1_k_center')
            _kg = _metric('s1_k_gnd')
            _sc = _metric('coacd_score')
            if _sc.size:
                logger.info(
                    "[Stage1 软代价] score(=对中0.40+包住0.20+轴向0.15+跟随0.15+桌面0.10) "
                    "中位=%.3f, >0.6 占比=%.1f%% | k_center 中位=%.3f (权重最高 ✓) | "
                    "k_gnd 中位=%.3f (软偏好 ✓)",
                    float(np.median(_sc)), 100.0 * float((_sc > 0.6).mean()),
                    float(np.median(_kc)) if _kc.size else float('nan'),
                    float(np.median(_kg)) if _kg.size else float('nan'))
            if _cl.size:
                logger.info(
                    "[Stage1 位姿遍历] CoACD 几何分布 (仅统计通过 MANO 约束的 %d 个候选): "
                    "c_clear(接近余量) min/中位/max=%+.2f/%+.2f/%+.2fmm, <0(能包住)占比=%.1f%% | "
                    "c_center(对中偏差) 中位=%.2fmm, <2mm 占比=%.1f%% | "
                    "|c_axial|<10mm 占比=%.1f%% | κ 中位=%.3f, >0.5 占比=%.1f%%",
                    _cl.size, float(_cl.min()), float(np.median(_cl)), float(_cl.max()),
                    100.0 * float((_cl < 0).mean()),
                    float(np.median(_cc)) if _cc.size else float('nan'),
                    100.0 * float((_cc < 2.0).mean()) if _cc.size else float('nan'),
                    100.0 * float((np.abs(_ca) < 10.0).mean()) if _ca.size else float('nan'),
                    float(np.median(_ka)) if _ka.size else float('nan'),
                    100.0 * float((_ka > 0.5).mean()) if _ka.size else float('nan'),
                )
        except Exception as _estat:  # noqa: BLE001
            logger.warning("[Stage1 位姿遍历] 统计失败(跳过): %s", _estat)

        passed_candidates = [sc for sc in scored_candidates if sc['passed']]
        if not passed_candidates:
            logger.warning("[Stage1] 无候选通过硬约束, 放宽条件取 q_score 最高的 Top-K")
            passed_candidates = sorted(scored_candidates, key=lambda x: x['q_score'], reverse=True)[:top_k]

        # 按 q_score 排序取 Top-K 进入物理验证
        top_candidates = sorted(passed_candidates, key=lambda x: x['q_score'], reverse=True)[:top_k]
        logger.info(f"[Stage1] {len(top_candidates)}/{len(candidates)} 候选进入 Stage 1b")

        # ===== Stage 1b: Top-K 短物理验证 =====
        # ★★★★ 2026-09-20 【P0 低方差评估】同一位姿跑 **R 次重复**, 按**成功率**排序 ★★★★
        #   用户 2026-09-20: "两个都可以试试, 然后看看 stage1 最终的效果"
        #   实测依据 (output/cc64v1 / cctrav3):
        #     · 单次评估的方差极大 ⇒ 同一锚点 3 次重复能出 lift_final=[0, 0, 21.8]cm ✗
        #       验证门日志自己就在说 "方差: 边际抓取单次评估侥幸得分" ✓
        #     · 653 次真实 rollout 中位耗时 **0.235s** ⇒ 重复成本很低 ✓
        #       (48 候选 × 3 次 ≈ 34s; 完全可接受 ✓)
        #   ⇒ 排序键 = (**成功次数**, 平均 reward) ✓, 成功 = `lift_final ≥ _STAGE1_SUCCESS_LIFT_M`
        #   开关 ✓: `_STAGE1_REPEAT` (默认 1 = 完全回到旧行为 ✓)
        _n_rep = max(1, int(os.environ.get("_STAGE1_REPEAT", "1")))
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-24 【方案 §11.5/A.4 实施】**统一 criteria** (搜索 ≠ 终验 ✗) ★★★★
        #   复审原话 ✓: \"搜索默认门是 **10 cm / 30 mm**, 终验默认是 **3 cm / 20 mm**;
        #     **同一个函数不等于同 criteria**; 某轮环境变量可能对齐, 但**代码默认仍会分叉**\" ✗
        #   ⇒ 后果: 搜索挑出的"成功"候选, 到终验被更严的门否掉 ⇒ **搜得多、交付不出来** ✗
        #   本改 ✓: 搜索**不再**读自己的 `_STAGE1_SUCCESS_*`, 而是与终验**共用**
        #     同一份 criteria (`_s1_criteria()` ✓ 含 hash ✓) ⇒ 两处**永不**分叉 ✓
        #     · 仍允许显式覆盖: 设 `_STAGE1_SUCCESS_LIFT_M` / `_STAGE1_SUCCESS_DRIFT_M`
        #       才会覆盖, 且**记入 hash** ⇒ 任何一轮用了非默认值都可追溯 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _crit_s = self._s1_criteria()
        _succ_lift = float(os.environ.get("_STAGE1_SUCCESS_LIFT_M", "")
                           or (_crit_s["lift_cm"] / 100.0))
        # ★★★★ 2026-09-20 成功判据必须**同时**要求"抬起"与"不拖飞" ★★★★
        #   实测 (output/ccrep1, 只判 lift≥10cm): 排出的最优是
        #     `成功=3/3, lift=30.26cm, **drift=49.26cm**` ✗
        #   ⇒ 那是"手指从底下把物体**铲/拖**起来" ✗, 不是夹取 ✓
        #   ⇒ 判据补上 `drift ≤ _STAGE1_SUCCESS_DRIFT_M` (默认 30mm ✓)
        #   (注意: reward 其实"知道"这点 —— 该候选 reward=−307 反而**低于** 0/3 的 −291 ✗;
        #    问题在**排序键**, 不在 reward ✓)
        # ★ 方案 §11.5/A.4 ✓: 同上 ⇒ 与终验**共用** criteria (不再 30mm vs 20mm 分叉 ✗)
        _succ_drift = float(os.environ.get("_STAGE1_SUCCESS_DRIFT_M", "")
                            or (_crit_s["drift_mm"] / 1000.0))
        logger.info("[Stage1] ===== Stage 1b: Top-K 物理验证 (pd_then_lock, 重复 %d 次/候选, "
                    "成功判据 lift≥%.0fcm 且 drift≤%.0fmm) =====",
                    _n_rep, _succ_lift * 100, _succ_drift * 1000)

        # ★★★★ 2026-09-21 【G5t / G6 硬门 —— **不动 reward** ✓】(设计 md §46.6) ★★★★
        #   实测铁证 ✓ (output/s2_diag / s2_diag2 / s2_restore, Stage2 settle 段**真值**):
        #     `挡停 +16.74/+5.98mm, 真夹持率 **100%** ✓, 两指距离差 **0.7mm** ✓
        #      但 **最近侧=12.5mm** ✗ 且 **指尖离桌面=0.0mm** ⚠`
        #     ⇒ 手指对称地被**桌面**挡住 (不是被物体 ✗) ⇒ **没压住物体** ⇒
        #       Stage1 的**竖直**提升看不出来 (重力帮忙 ✓), 但 Stage2 的 **CARRY 是水平**移动
        #       ⇒ 无摩擦可用 ⇒ **一启动就脱** ✗ (`CARRY: T_G_O drift=177.49mm` ✗)
        #   ⇒ 两条门 (默认 **0 = 关闭** ✓ ⇒ 既有 2/3 行为完全不受影响 ✓):
        #     · **G5t**: `finger_ground_clear_mm ≥ _S1_GATE_GND_TRUTH_MM` (真值, 建议 3mm ✓)
        #     · **G6** : `s1_dmin_truth_mm      ≤ _S1_GATE_DMIN_MAX_MM`  (真值, 建议 2mm ✓)
        #   为什么是"门"而不是 reward ✓: 本轮已**四次**验证
        #     "改 reward / 改评分权重 ⇒ 换盆地 ⇒ 丢掉已找到的好解" ✗
        #     ⇒ 门只改**排序键里的"成功次数"** ✓ (reward 地形一字不动 ✓)
        #   取不到真值时(如 `_NO_GRIPPER_GROUND=1`) ⇒ 判为**不通过** ✓ (宁可保守 ✗ 不放行假抓取 ✓)
        _g5t = float(os.environ.get("_S1_GATE_GND_TRUTH_MM", "0.0") or 0.0)
        _g6 = float(os.environ.get("_S1_GATE_DMIN_MAX_MM", "0.0") or 0.0)
        # ★ 2026-09-21 【G4 真夹持门】`_S1_GATE_TG_MIN` (默认 0 = 关 ✓)
        #   实测反证 ✓ (output/s2_lift2, 已把 G5 打开 ⇒ 候选离桌面 6.1~6.6mm ✓):
        #     抬高后验证门变成 `lift_final=[0,0,0]cm → 0/3` ✗
        #     ⇒ **把手指从桌面抬离, 物体就抬不起来了** ✓✓
        #     ⇒ **反证 Stage1 那 2/3 的抬升是"托"(指尖靠桌面形成托盘 ✗) 而不是"夹"** ✗
        #       (与用户之问"stage1 的抓取是假的吗"完全对上 ✓, 答案是:
        #        抬升段有相当成分是**托** ✗ ⇒ 竖直能抬 ✓ 水平一移动就掉 ✗)
        #   ⇒ 本门 ✓: 要求"**两指同时被物体挡停**"的帧占比 ≥ 门槛 ✓ (= 真摩擦夹持 ✓)
        _gtg = float(os.environ.get("_S1_GATE_TG_MIN", "0.0") or 0.0)

        def _s1_gate_rollout(_sr):
            """G4/G5t/G6 真值硬门 ⇒ 只影响"成功次数" ✓, 不动 reward ✓"""
            if _g5t > 0.0:
                _v = _sr.get('finger_ground_clear_mm')
                if _v is None or float(_v) < _g5t:
                    return False
            if _g6 > 0.0:
                _v2 = _sr.get('s1_dmin_truth_mm')
                if _v2 is None or float(_v2) > _g6:
                    return False
            if _gtg > 0.0:
                _v3 = _sr.get('s1_true_grasp_ratio')
                if _v3 is None or float(_v3) < _gtg:
                    return False
            return True

        def _eval_pose_repeat(grasp_pose):
            """跑 R 次重复 (每次都由 rollout 内部复位物体 ✓) → (成功数, 平均reward, 最好reward, 最好sim)"""
            _rs, _srs = [], []
            for _ri in range(_n_rep):
                _r, _sr = _eval_pose(grasp_pose)
                _rs.append(float(_r))
                _srs.append(_sr)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【复审 I.2 修复】CEM 的**重复成功数**改用**同一评估函数** ✓ ★★★★
            #   复审原话 ✓: "Top-K 新加了一套支撑/接触门, **CEM 的重复成功数仍主要按峰值 lift、
            #     漂移和旧可选门计算**, 精英更新仍按 reward ✗ ⇒ **搜索在追逐旧目标,
            #     最后才被新门淘汰** ⇒ 会造成**多轮 CEM 都无效**" ✗✗
            #   本改 ✓: `_ns` 与**最终验收**走**同一个** `self._assess_grasp(...)` ✓
            #     ⇒ 搜索的成功定义 = 验收的成功定义 ✓ (不再"追旧目标"✓)
            #     ⚠ 保留原单位兼容 ✓: `_succ_lift` 若已是"米" (≤0.5) ⇒ ×100 转 cm ✓;
            #       若本就按 cm 给出 ⇒ 直接用 ✓
            #     ⚠ 缺测 ⇒ **不算成功** ✓ (复审 I.2-4 ✓; 但仍可继续搜索 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _succ_lift_v = float(_succ_lift)
            _succ_drift_v = float(_succ_drift)
            _succ_crit = {
                'lift_cm': (_succ_lift_v * 100.0) if _succ_lift_v <= 0.5 else _succ_lift_v,
                'drift_mm': ((_succ_drift_v * 1000.0) if _succ_drift_v <= 0.5
                             else _succ_drift_v),
                'gnd_mm': float(os.environ.get("_S1_ACC_GND_MM", "0.0") or 0.0),
                'tg_min': float(os.environ.get("_S1_ACC_TG_MIN", "0.5") or 0.5),
                'allow_missing': False,
            }
            _ns = sum(
                1 for _sr in _srs
                if self._assess_grasp(
                    _sr, dict(_succ_crit,
                              extra_gate_ok=bool(_s1_gate_rollout(_sr)))
                )["task_success"])
            _mean = float(np.mean(_rs))
            _bi = int(np.argmax(_rs))
            return _ns, _mean, float(_rs[_bi]), _srs[_bi]

        best_output = None
        best_reward = -float('inf')
        best_result = None
        best_key = (-1, -float('inf'))
        # ★ 2026-09-22 【门优先选优 ✓】独立保留"门通过位" ✓
        #   (不动 `best_key` 的 2 元组语义 ✗ ⇒ 下游日志/逻辑零改动 ✓)
        best_gate = -1
        for sc in top_candidates:
            cand = sc['cand']
            _ns, _mean, r, sim_result = _eval_pose_repeat(cand)
            logger.info(f"  Top-K cand 成功={_ns}/{_n_rep}, mean_r={_mean:.3f}, best_r={r:.3f}, "
                        f"lift={sim_result['obj_lift']*100:.2f}cm, "
                        f"drift={sim_result['obj_xy_drift']*100:.2f}cm, "
                        # ★ 2026-09-21 打出 G5t/G6 用的真值 ✓ (门判据的依据 ✓)
                        f"**指尖离桌面={sim_result.get('finger_ground_clear_mm')}mm, "
                        f"最近侧={sim_result.get('s1_dmin_truth_mm')}mm, "
                        f"真夹持率={sim_result.get('s1_true_grasp_ratio')}**, "
                        f"src_frame={cand.get('source_frame')}, "
                        f"pos={cand['pos'][:3].round(4)}")
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【门优先选优 ✓】排序纳入**真值门** ✗ (不再只看 reward ✗) ★★★★
            #   为什么 ✓: §96 实测交付的候选 **物体仍由桌面支撑 ✗ + 夹爪插桌/被卡 ✗**;
            #     而 §82 的 D 数据表明"**lift≥3cm 且不插桌**"的候选历史上有 **83 个** ✓
            #     ⇒ **池里有合格候选, 是排序把它们丢了** ✗ ⇒ 把门位排到最前 ✓
            #   门 (全部可调 ✓): `_S1_SEL_LIFT_CM`(3cm) / `_S1_SEL_GND_MM`(0=不插桌 ✓) /
            #     `_S1_SEL_TG_MIN`(真夹持率 0.5 ✓) / ②`目标不再由桌面支撑` ✓ /
            #     ③`夹爪↔桌面分级 ≠ load_bearing_block` ✓
            #   兼容 ✓: 门位相同才比旧 `(_ns, _mean)` ✓ ⇒ 无候选过门时行为与旧版**完全一致** ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _sel_lift = float(os.environ.get("_S1_SEL_LIFT_CM", "3.0") or 3.0)
            _sel_gnd = float(os.environ.get("_S1_SEL_GND_MM", "0.0") or 0.0)
            _sel_tg = float(os.environ.get("_S1_SEL_TG_MIN", "0.5") or 0.5)
            _g_lift = float(sim_result.get('obj_lift', 0.0) or 0.0) * 100.0
            _g_gnd = sim_result.get('finger_ground_clear_mm')
            _g_tg = sim_result.get('s1_true_grasp_ratio')
            _g_sup = sim_result.get('s1_obj_supported')
            _g_cls = str(sim_result.get('s1_gripper_ground_class') or "")
            _gate_pass = bool(
                _g_lift >= _sel_lift
                and (_g_gnd is None or float(_g_gnd) >= _sel_gnd)
                and (_g_tg is None or float(_g_tg) >= _sel_tg)
                and (_g_sup is False)
                and (_g_cls != "load_bearing_block"))
            logger.info(
                "    ⇒ **门**: lift=%.2fcm(≥%.1f) 指尖离桌=%s(≥%.1f) 真夹持率=%s(≥%.2f) "
                "②目标仍由桌面支撑=%s ③夹爪↔桌面=%s ⇒ **%s**",
                _g_lift, _sel_lift, _g_gnd, _sel_gnd, _g_tg, _sel_tg, _g_sup, _g_cls,
                "✓ 过门" if _gate_pass else "✗ 不过门")
            _key = (_ns, _mean)
            _gate_bit = 1 if _gate_pass else 0
            _better = (_gate_bit > best_gate) or (
                _gate_bit == best_gate and _key > best_key)
            if _better:
                best_gate = _gate_bit
                best_key = _key
                best_reward = r
                best_output = {
                    'pos': cand['pos'].copy(),
                    'R': cand['R'].copy(),
                    'euler': rotmat_to_zyx_euler(cand['R']),
                    # ★ 2026-09-22 【P0-4 修复】`or _gq_final` ✗ 会把**合法 0.0 目标**当成缺失 ✗
                    #   (审计文档 §P0-4 ✓: "多处 `value or fallback` 会把合法
                    #    `gripper_qpos_target=0.0` 当成缺失, 换成 `_gq_final`" ✗)
                    #   ⇒ 改为**显式 None 判定** ✓ (缺失才兜底 ✓, 零值是有效值 ✓)
                    'gripper_qpos': float(
                        sim_result.get('gripper_qpos_target')
                        if sim_result.get('gripper_qpos_target') is not None else _gq_final),
                    'strategy': 'pd_then_lock',
                    'force_limit': float(GRASP_STRATEGIES['pd_then_lock']['force_limit']),
                    'reward': best_reward,
                    'ref_base_R': cand.get('ref_base_R', cand['R'].copy()),
                    # ★ 2026-09-18 本候选的**源帧参考姿态** —— 供 Stage1c 的 MANO 硬约束参考 ✓
                    #   (`_STAGE1_MANO_REF=frame` 时 CEM 相对它判 ±5°;
                    #    否则若锚点姿态离 F50 >5° ✗ ⇒ CEM 的**每个样本**都被判 −50 ✗
                    #    ⇒ 搜索彻底死掉 (实测 s1horiz: iter0~7 全部 "均值漂出可行域" ✗) ✓)
                    'ref_R': (cand.get('ref_R').copy()
                              if cand.get('ref_R') is not None else cand['R'].copy()),
                    'enclose_z': sim_result.get('enclose_z'),
                }
                best_result = sim_result

        logger.info(f"[Stage1] Stage 1b 最优: 成功 {best_key[0]}/{_n_rep}, mean_r={best_key[1]:.3f}, "
                    f"best_r={best_reward:.3f}, pos={best_output['pos'].round(4) if best_output else None}")

        if best_output is None:
            logger.error("[Stage1] Stage 1b 未产生有效结果")
            return None

        # ===== Stage 1c: CMA-ES 微调 6DOF =====
        logger.info("[Stage1] ===== Stage 1c: CMA-ES 微调 6DOF =====")
        dim = 6
        # 固定参考位姿 (Stage 1b 最优), 搜索空间为相对该参考的 6DOF 偏移
        ref_pos = best_output['pos'].copy()
        ref_R = best_output['R'].copy()
        # ★★★★ 2026-09-20 【P0】CEM 的硬约束参考 —— 修好它 CEM 才真的在工作 ★★★★
        #   实测 (output/cc64v1, 653 次 rollout): CEM 15 iter × 32 样本 = 480 个样本/轮,
        #     三轮合计**只有 137 次真实 rollout** ⇒ **有效率 ≈9%** ✗
        #     证据: iter0 还有 mean=−113/−143, **iter1 之后几乎全是 `mean=−50.000`** ✗
        #       (mean=−50 ⇒ 该 iter 全部样本违反约束 ⇒ 一次 rollout 都没跑 ✗)
        #       std 一路缩到 0.001rad ⇒ 搜索彻底退化 ✗
        #   根因 ✗: 参考取的是锚点的**源帧**姿态, 而锚点自己已经带了 Stage0 的
        #     ±5°(Y)·±5°(Z) 扰动 (合成可达 ~7°) ⇒ `R_actual = ΔR·R_锚点` 恒 >5° ✗
        #     ⇒ 无论 ΔR 怎么采都违规 ⇒ 全 −50 ✗ (CEM 想"取消"这个偏差只能靠往锚点反方向走,
        #       而它是在锚点附近**小球**采样 ⇒ 结构性不可达 ✗)
        #   修法 ✓: 参考 = **锚点自身姿态** ⇒ 语义变成"在锚点 ±5° 内微调" ✓
        #     (这与用户的"位姿先定下来、再做微调"一致 ✓; 且 `_STAGE1_MANO_REF=frame`
        #      已保证锚点本身在源帧 ±5° 内 ✓ ⇒ 总偏差仍受控 ✓)
        #   开关 ✓: `_STAGE1_CEM_REF` = `anchor`(默认 ✓) / `source`(旧行为) / `f50`
        _cem_ref_mode = os.environ.get("_STAGE1_CEM_REF", "anchor")
        _cem_ref_R = mano_f50_R
        if isinstance(best_output, dict):
            _bo_ref = best_output.get('ref_R')          # 锚点源帧 (旧行为)
            _bo_R = best_output.get('R')                # 锚点自身 (新默认)
            if _cem_ref_mode == "source" and _bo_ref is not None:
                _cem_ref_R = np.asarray(_bo_ref, dtype=np.float64)
            elif _cem_ref_mode == "anchor" and _bo_R is not None:
                _cem_ref_R = np.asarray(_bo_R, dtype=np.float64)
        logger.info("[Stage1] CEM 硬约束参考 = %s (可用 _STAGE1_CEM_REF 改: anchor/source/f50)",
                    _cem_ref_mode)
        mean = np.zeros(dim)
        std = np.array([0.008, 0.008, 0.004, 0.015, 0.015, 0.015])
        elite_frac = 0.25

        for iteration in range(n_iterations):
            samples = np.random.randn(population_size, dim) * std + mean
            rewards = []
            _iter_poses = []            # 与 rewards **同序** (仅有效样本) → 供 top-1 确认 ✓
            for i in range(population_size):
                dx, dy, dz, dry, drx, drz = samples[i]
                pos = ref_pos + np.array([dx, dy, dz])
                delta_rot = R_scipy.from_euler("xyz", [drx, dry, drz]).as_matrix()
                R_actual = delta_rot @ ref_R

                # MANO 硬约束: 相对参考姿态的旋转差不超过 5°
                #   参考 = `_STAGE1_MANO_REF=f50` ⇒ MANO F50 ✓ (历史行为 ✓)
                #          `_STAGE1_MANO_REF=frame` ⇒ 锚点自己的源帧姿态 ✓ (多帧扫描才生效 ✓)
                if not _check_mano_hard_constraint(pos, R_actual, ref_R_for_constraint=_cem_ref_R):
                    rewards.append(-50.0)
                    _iter_poses.append(None)
                    continue


                grasp_pose = {'pos': pos, 'R': R_actual}
                r, sim_result = _eval_pose(grasp_pose)
                rewards.append(r)
                _iter_poses.append(grasp_pose)

                # ★★★★ 2026-09-20 【P0】CEM 的"更好"必须**经重复确认**才算数 ★★★★
                #   实测 (output/ccacd1): CEM 单次评估爬到 **best=+2.016** (第 3 轮) ✓,
                #     但终选位姿的验证门仍是 **0/3** ✗ ⇒ 说明被选中的是**单次走运的样本** ✗
                #     (单次评估的方差远大于候选间差异 ✓)
                #   ⇒ 修法: 单次 reward 超过当前最好时, 先用 `_n_rep` 次重复**确认** ✓,
                #     以 (成功数, 平均reward) 为准 ✓ —— 与 Stage1b 同一把尺子 ✓
                #   (`_n_rep=1` ⇒ 完全回到旧行为 ✓, 且只在"可能刷新纪录"时才多花 rollout ✓)
                if r > best_reward and _n_rep > 1:
                    _c_ns, _c_mean, _c_best, _c_sr = _eval_pose_repeat(grasp_pose)
                    if (_c_ns, _c_mean) > best_key:
                        best_key = (_c_ns, _c_mean)
                        logger.info("[Stage1] CEM 确认新最好: 成功 %d/%d, mean_r=%.3f "
                                    "(单次 r=%.3f), pos=%s",
                                    _c_ns, _n_rep, _c_mean, r, np.round(pos, 4).tolist())
                    else:
                        continue

                if r > best_reward:
                    best_reward = r
                    best_output = {
                        'pos': pos.copy(),
                        'R': R_actual.copy(),
                        'euler': rotmat_to_zyx_euler(R_actual),
                        # ★ 2026-09-22 【P0-4 修复】`or _gq_final` ✗ 会把**合法 0.0 目标**当成缺失 ✗
                    #   (审计文档 §P0-4 ✓: "多处 `value or fallback` 会把合法
                    #    `gripper_qpos_target=0.0` 当成缺失, 换成 `_gq_final`" ✗)
                    #   ⇒ 改为**显式 None 判定** ✓ (缺失才兜底 ✓, 零值是有效值 ✓)
                    'gripper_qpos': float(
                        sim_result.get('gripper_qpos_target')
                        if sim_result.get('gripper_qpos_target') is not None else _gq_final),
                        'strategy': 'pd_then_lock',
                        'force_limit': float(GRASP_STRATEGIES['pd_then_lock']['force_limit']),
                        'reward': best_reward,
                        'ref_base_R': mano_f50_R.copy(),
                        'ref_R': np.asarray(_cem_ref_R, dtype=np.float64).copy(),
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
            if not _check_mano_hard_constraint(mean_pos, mean_R, ref_R_for_constraint=_cem_ref_R):
                logger.info(f"[Stage1] CMA-ES iter {iteration}: 均值漂出可行域, 重置到参考位姿")
                mean = np.zeros(dim)
                std = np.minimum(std * 0.8, np.array([0.005, 0.005, 0.003, 0.015, 0.015, 0.015]))

            # ★★★★ 2026-09-20 【P0】每个 iter 的 **top-1 也确认一次** ★★★★
            #   为什么必须 ✓: 上面的"单次刷新纪录才确认"有个盲区 ——
            #     一个**可复现**的好解, 单次分可能不高 (< best_reward) ⇒ **永远不触发确认** ✗
            #     ⇒ 会被"单次走运的高分样本"压住 ✗ (实测 ccacd1: best=+2.016 但门 0/3 ✗)
            #   本步 ✓: 每个 iter 花 `_n_rep` 次 rollout, 确认该 iter 的 top-1 ✓
            #     (它正是驱动下一轮均值的那一个 ⇒ 信息量最大 ✓; 15 iter ⇒ 仅多 ~45 次 rollout ✓)
            if _n_rep > 1 and _iter_poses:
                _ti = int(np.argmax(rewards))
                _tp = _iter_poses[_ti]
                if _tp is not None and float(rewards[_ti]) > -50.0:
                    _t_ns, _t_mean, _t_best, _t_sr = _eval_pose_repeat(_tp)
                    if (_t_ns, _t_mean) > best_key:
                        best_key = (_t_ns, _t_mean)
                        best_reward = max(float(_t_best), float(rewards[_ti]))
                        best_result = _t_sr
                        best_output = {
                            'pos': np.asarray(_tp['pos'], dtype=np.float64).copy(),
                            'R': np.asarray(_tp['R'], dtype=np.float64).copy(),
                            'euler': rotmat_to_zyx_euler(_tp['R']),
                            # ★ 2026-09-22 【P0-4】显式 None 判定 ✓ (合法 0.0 不再被换成兜底 ✗)
                            'gripper_qpos': float(
                                (_t_sr or {}).get('gripper_qpos_target')
                                if (_t_sr or {}).get('gripper_qpos_target') is not None
                                else _gq_final),
                            'strategy': 'pd_then_lock',
                            'force_limit': float(GRASP_STRATEGIES['pd_then_lock']['force_limit']),
                            'reward': best_reward,
                            'ref_base_R': mano_f50_R.copy(),
                            'ref_R': np.asarray(_cem_ref_R, dtype=np.float64).copy(),
                            'enclose_z': _t_sr.get('enclose_z'),
                        }
                        logger.info("[Stage1] CEM iter %d top-1 确认: 成功 %d/%d, mean_r=%.3f "
                                    "(单次 %.3f) ⇒ **接受** ✓",
                                    iteration, _t_ns, _n_rep, _t_mean, float(rewards[_ti]))

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
                    # ★ 2026-09-20: Stage1d 的"改进"同样**必须先经重复确认** ✓
                    #   (否则一个单次走运的平移点会覆盖掉 Stage1b/CEM 找到的稳解 ✗)
                    if r_pol > _pol_best_r and _n_rep > 1:
                        _p_ns, _p_mean, _p_best, _p_sr = _eval_pose_repeat(gp_pol)
                        if (_p_ns, _p_mean) <= best_key:
                            continue
                        best_key = (_p_ns, _p_mean)
                        r_pol, sr_pol = _p_best, _p_sr
                        logger.info("[Stage1] Stage1d 确认: 成功 %d/%d mean_r=%.3f ⇒ 接受 ✓",
                                    _p_ns, _n_rep, _p_mean)
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
                            # ★ 保留 ref_R (否则这一支会把 Stage1c 写下的 ref_R 丢掉 ✓)
                            'ref_R': best_output.get('ref_R'),
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
        _v_gnd, _v_dmin = [], []          # ★ G5t/G6 真值 (与 Top-K 用**同一个门** ✓)
        # ★ 2026-09-22 【复审 · 先确认抓住 ✓】新增四项真值收集 ✓
        #   (真夹持率 / 水平漂移 / **目标是否仍由桌面支撑** / **夹爪↔桌面分级** ✓)
        _v_tg, _v_drift = [], []
        _v_sup, _v_ggc = [], []
        _v_eps = []                       # ★ T07e4 ✓ 每次重复 run 的 episode_id (md 要求 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【复审第 4 条】"先冻结动作 ⇒ 再用它做终验" ✓ ★★★★
        #   复审 ✓: "`冻结动作`**没有真正接通**: 最终验证仍重算闭合动作,
        #     记录在**验证之后**才生成, 也未完整传给下游" ✗
        #   本处 ✓: 把动作记录**提前到终验之前**冻结 ✓ ⇒ 终验与后续交付消费**同一份** ✓
        #   注 ✓: 终验调用点已带 `q_override` 参数位 (见 `rollout_grasp_only` ✓) ⇒
        #     `_k == 0` 的那次会用它 ✓ (其余两次保持现状, 便于观察一致性 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            from physics_env import CONTROL_FREQ as _CF4
        except Exception:  # noqa: BLE001
            _CF4 = 30.0
        try:
            _cc4 = int(best_output.get('close_controls')
                       if best_output.get('close_controls') is not None
                       else STAGE1_CLOSE_CONTROLS)
            _sc4 = int(best_output.get('settle_controls')
                       if best_output.get('settle_controls') is not None
                       else STAGE1_SETTLE_CONTROLS)
            # ★ g2 ✓: T8 额外压入帧数 → **控制点数** (每控制点的帧数由 close 段标定 ✓)
            _t8f = int(best_output.get('s1_t8_extra_frames') or 0)
            try:
                _fpc = max(1, int(round(float(locals().get('_CLOSE_FRAMES', 0) or 0)
                                        / max(1, int(_cc4)))))
            except Exception:  # noqa: BLE001
                _fpc = 1
            _pc4 = int(-(-_t8f // _fpc)) if _t8f > 0 else 0   # ceil ✓
            best_output['action_record'] = {
                "frozen": True,
                "phase": "before_verification",     # ★ 顺序标记 ✓ (复审第 4 条 ✓)
                "target_name": target_name,
                "base_pos": np.asarray(best_output['pos'], dtype=np.float64).copy(),
                "base_R": np.asarray(best_output['R'], dtype=np.float64).copy(),
                # ★ 2026-09-30 修复 ✗: 名义值 ≠ 实际下发 ⇒ Stage2 报 "anchor 与 recipe 不符"
                #   本改: 优先用 rollout 回传的"T1b 钳位后实际生效值" ⇒ anchor 落在实际值域内 ✓
                "q_cmd": float(best_output.get('s1_settle_end_q_cmd')
   if best_output.get('s1_settle_end_q_cmd') is not None
   else (best_output.get('s1_frozen_q_effective')
         if best_output.get('s1_frozen_q_effective') is not None
         else best_output['gripper_qpos'])),
                "q_cmd_nominal": float(best_output['gripper_qpos']),
                "q_source": ("rollout_effective(T1b-clamped)"
                             if best_output.get('s1_frozen_q_effective') is not None
                             else "nominal_candidate(未过钳位)"),
                "dt": float(1.0 / float(_CF4)),
                "close_controls": _cc4,
                # ★ g2 ✓: settle 长度 = 基础 settle + **T8 实际延迟预紧** (真实值 ✓)
                #   ⇒ Stage2 读它即无损迁移; 不再用写死的 10 ✗
                "preload_extra_controls": int(max(0, _pc4)),
                "settle_controls": int(_sc4) + int(max(0, _pc4)),
                "origin": "winner_candidate_pre_verify",
            }
            self._stage1_action_record = best_output['action_record']
            logger.info(
                "[P0-4 动作记录] ✓ **终验前**已冻结: q_cmd=%.4f, close/settle=%d/%d, dt=%.5fs "
                "(⇒ 终验与交付消费**同一份** ✓, 复审第 4 条 ✓)",
                float(best_output['gripper_qpos']), _cc4, _sc4, float(1.0 / float(_CF4)))
        except Exception as _ear4:  # noqa: BLE001
            logger.warning("[P0-4 动作记录] 冻结失败: %s", _ear4)

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【复审 I.3 修复】三次终验**一律**执行**同一份冻结命令** ✓ ★★★★
        #   复审原话 ✓: "当前**只有 `_k == 0`** 的第一次终验传 `q_override`; 第二、第三次仍
        #     **现场重算** ⇒ 2/3 的结果可能来自**不同闭合命令**, 不能证明导出配方可重放" ✗
        #   本改 ✓ (复审 I.3 要求"一次小修完成闭环" ✓):
        #     · 先取出冻结 `q_cmd` ✓ (来自 `action_record` ✓ = 即将**落盘**的那份 ✓)
        #     · 三次**全部**传同一值 ✓ ⇒ "验证动作 = 交付动作" **逐位成立** ✓
        #     · 执行端另有**最后一道锁** ✓ (见 `rollout_grasp_only` 内"仿真前"处 ✓)
        #       ⇒ 不再被 `_S1_GRIP_DELTA` 等旋钮二次修改 ✗
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _q_frozen_v = None
        try:
            if isinstance(best_output.get('action_record'), dict):
                _q_frozen_v = best_output['action_record'].get('q_cmd')
                if _q_frozen_v is not None:
                    _q_frozen_v = float(_q_frozen_v)
        except Exception as _eqf:  # noqa: BLE001
            logger.warning("[P0-4 冻结命令] 读取失败: %s", _eqf)
            _q_frozen_v = None
        logger.info("[Stage1][终验] ★ 三次重复**统一使用冻结命令** q=%s ✓ "
                    "(复审 I.3 ✓: 否则 2/3 可能来自不同闭合命令 ✗)",
                    "n/a" if _q_frozen_v is None else f"{_q_frozen_v:.4f}")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-28 补"成功录像"缺口 ✓ (R 系列"交付物/录像"缺口) ★★★★
        #   实测 ✗: 成功 rollout **从来不录** ——
        #     · 搜索期 `_eval_pose` 调 `rollout_grasp_only()` **不传** `record_video` ✗
        #     · 终验(此处)同样**不传** ✗
        #     · `_RECORD_VIDEO_ALL=1` 只强制录**前 K 次** ⇒ 录到的全是**扫描早期的失败候选** ✗
        #       (实测 output/gif2_s31: 前 3 段 tier=3/3/3, 抬起全 False ✗)
        #     ⇒ 于是"抓取成功/抬起"的视频**永远拿不到** ✗ (用户 2026-09-28: "把抬起的给我看看")
        #   本改 ✓: 终验 = 对**获胜候选**用**冻结命令**重放 ⇒ 正是"成功那一刻" ✓
        #     只给第 1 次重复录像 (第 2/3 次纯复现, 不必再录 ⇒ 省 2/3 开销 ✓)
        #   开关 ✓: `_S1_RECORD_FINAL_VERIFY=0` 可关 (默认 1 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _rec_verify28 = (str(os.environ.get("_S1_RECORD_FINAL_VERIFY", "1")) == "1")
        for _k in range(3):
            _vr = self.rollout_grasp_only(
                {'pos': best_output['pos'].copy(), 'R': best_output['R'].copy()},
                side, strategy='pd_then_lock', n_frames=160,
                record_video=bool(_rec_verify28 and _k == 0),
                **({"q_override": _q_frozen_v} if _q_frozen_v is not None else {}))
            _v_lifts.append(float(_vr.get('obj_lift_final', 0.0)))
            _v_gnd.append(_vr.get('finger_ground_clear_mm'))
            try:                              # ★ T07e4 ✓ 记录本次重复 run 的身份 ✓
                _v_eps.append(_vr.get('episode_id'))
            except Exception:  # noqa: BLE001
                _v_eps.append(None)
            _v_dmin.append(_vr.get('s1_dmin_truth_mm'))
            # ★ 复审 ✓: "先确认**抓住**, 不要继续只看抬升高度" ✓ ⇒ 一并收集真值 ✓
            _v_tg.append(_vr.get('s1_true_grasp_ratio'))
            _v_drift.append(_vr.get('obj_xy_drift'))
            # ★ 复审 ②③ ✓: "目标是否仍由桌面支撑" + "夹爪是否被桌面卡住" ✓
            _v_sup.append(_vr.get('s1_obj_supported'))
            _v_ggc.append(_vr.get('s1_gripper_ground_class'))
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【复审 · Stage1 第一步】"**先确认抓住**" 的**真值验收** ✓ ★★★★
        #   复审原话 ✓: "以前的'托起来'和'两指真夹持'结论都**需要重新核验**:
        #     两指挡停不等于被目标挡住, 物体升高也不等于能够稳定搬运" ✗
        #     "必须确认 **① 两指分别接触目标 ② 目标不再由桌面支撑 ③ 夹爪没有被桌面卡住
        #      ④ 物体相对夹爪的位姿稳定**" ✓
        #   本处 ✓: `_n_pass` 不再只看 lift ✗ ⇒ 改用**真值合取** ✓ (门槛可调 ✓):
        #     · `lift ≥ _S1_ACC_LIFT_CM` (默认 3cm ✓)
        #     · `指尖离桌面 ≥ _S1_ACC_GND_MM` (默认 **0** ⇒ **不插桌** ✓; 这是 §82 D 数据
        #       里 83 个候选满足的那一条 ✓)
        #     · `真夹持率 ≥ _S1_ACC_TG_MIN` (默认 **0.5** ✓) —— 它是"**两指同时被挡停**"的
        #       帧占比 ✓ ⇒ 用来**核验①"两指分别接触目标"** ✓ (复审要求重新核验 ✓)
        #     · `obj_xy_drift ≤ _S1_ACC_DRIFT_MM` (默认 **20mm** ✓) ⇒ 粗略核验④"稳定" ✓
        #   ⚠ 仍**测不到**的两项 (如实标注 ✓, 待补探针 ✗):
        #     ② 目标**不再由桌面支撑** ⇒ 需要"**物体↔地面**"接触探针 ✗
        #        (当前 P1-3 只测**夹爪↔地面** ✗ ⇒ 见 md §91 ✓)
        #     ③ 夹爪**未被桌面卡住** ⇒ P1-3 的 `load_bearing_block` 分级已具备 ✓,
        #        但本 rollout **未把该分级回传** ✗ (待接 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _acc_lift = float(os.environ.get("_S1_ACC_LIFT_CM", "3.0") or 3.0)
        _acc_gnd = float(os.environ.get("_S1_ACC_GND_MM", "0.0") or 0.0)
        _acc_tg = float(os.environ.get("_S1_ACC_TG_MIN", "0.5") or 0.5)
        _acc_drift = float(os.environ.get("_S1_ACC_DRIFT_MM", "20.0") or 20.0)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【复审 I.2 修复】终验门改为调用**统一评估函数** ✓ ★★★★
        #   复审原话 ✓:
        #     · "新门**只排除** `_ggc == 'load_bearing_block'`, 却**没有排除**
        #        `obvious_penetration`; `None/未知` 也没有统一失败规则" ✗
        #     · "地面净空、夹持率、漂移等字段采用 `value is None or ...`, **缺测可能被当成通过**" ✗
        #     · "终验仍向 `_s1_gate_rollout()` **只传**地面净空和 dmin, **没有传 true-grasp 字段**;
        #        打开旧 `_S1_GATE_TG_MIN` 时仍会**因缺字段失败**" ✗
        #     · "建议用一个共享的 `assess_grasp(result, criteria)` … **不要各处再复制一套**" ✓
        #   本改 ✓:
        #     · **唯一**判定入口 = `self._assess_grasp(...)` ✓ (数据有效性 + 物理可行性 + 任务成功 ✓)
        #     · `obvious_penetration` 与 `load_bearing_block` **都**判不可行 ✓
        #     · `None/未知` ⇒ **不可信 ⇒ 不通过** ✓ (默认; `_S1_ACC_ALLOW_MISSING=1` 才放宽 ✗)
        #     · 真夹持字段**一并传入**门 ✓ (旧 `_S1_GATE_TG_MIN` 不再因缺字段失败 ✓)
        #     · 失败原因**逐条打印** ✓ ⇒ 每轮都能看出"被哪一条否掉" ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _acc_allow_missing = os.environ.get("_S1_ACC_ALLOW_MISSING", "0") == "1"
        _acc_ok = []
        _acc_reasons = []
        for _i, _l in enumerate(_v_lifts):
            _g = _v_gnd[_i]
            _t = _v_tg[_i]
            _d = _v_drift[_i]
            _sup = _v_sup[_i]
            _ggc = _v_ggc[_i]
            # ★ I.2-2 ✓: 门也要拿到 true-grasp 字段 (不再只给净空/dmin ✓)
            _extra_ok = bool(_s1_gate_rollout({
                'finger_ground_clear_mm': _g,
                's1_dmin_truth_mm': _v_dmin[_i],
                's1_true_grasp_ratio': _t,
                'obj_lift_final': _l,
            }))
            _as = self._assess_grasp(
                {'obj_lift_final': _l, 'finger_ground_clear_mm': _g,
                 's1_true_grasp_ratio': _t, 'obj_xy_drift': _d,
                 's1_obj_supported': _sup, 's1_gripper_ground_class': _ggc,
                 's1_dmin_truth_mm': _v_dmin[_i]},
                {'lift_cm': _acc_lift, 'gnd_mm': _acc_gnd, 'tg_min': _acc_tg,
                 'drift_mm': _acc_drift, 'allow_missing': _acc_allow_missing,
                 'extra_gate_ok': _extra_ok})
            _acc_ok.append(bool(_as["task_success"]))
            _acc_reasons.append(_as)
        logger.info("[Stage1][真值验收] ★ 统一评估 (`_assess_grasp` ✓ 复审 I.2): 逐次原因 = %s",
                    ["通过 ✓" if _a["task_success"] else
                     ("+".join(_a["reasons"] + _a["infeasible"]) or "unknown")
                     for _a in _acc_reasons])
        _n_pass = int(sum(1 for _o in _acc_ok if _o))
        logger.info(
            "[Stage1][真值验收] 逐次: %s | 门槛: lift≥%.1fcm, 指尖离桌≥%.1fmm, "
            "真夹持率≥%.2f, drift≤%.0fmm, **② 目标不再由桌面支撑 ✓, ③ 夹爪未被桌面卡住 ✓** "
            "⇒ **%d/3 通过** ✓",
            ["✓" if _o else "✗" for _o in _acc_ok], _acc_lift, _acc_gnd, _acc_tg, _acc_drift,
            _n_pass)
        logger.info(
            "[Stage1][真值验收] 四项确认 (复审 ✓): ① 两指分别接触目标(真夹持率)=%s | "
            "② 目标仍由桌面支撑=%s | ③ 夹爪↔桌面分级=%s | ④ 相对稳定(drift)=%s",
            [None if _t is None else round(float(_t), 2) for _t in _v_tg],
            _v_sup, _v_ggc,
            ["n/a" if _d is None else f"{float(_d)*1000:.1f}mm" for _d in _v_drift])
        logger.info(f"[Stage1][验证门] 终选位姿 3 次重复: lift_final="
                    f"{[round(_l*100, 1) for _l in _v_lifts]}cm → {_n_pass}/3 通过 (门槛 2) | "
                    f"真值: 指尖离桌面={[None if _g is None else round(_g, 1) for _g in _v_gnd]}mm, "
                    f"最近侧={[None if _d is None else round(_d, 1) for _d in _v_dmin]}mm "
                    f"(G5t≥{_g5t}mm / G6≤{_g6}mm, 0=关闭 ✓)")

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T07e4 (2026-09-27) 【verification_results 收全重复 run 明细】 ✓ ★★★★
        #   md T07.1 ✓: "`verification_results`: **全部重复 run 的 ID、数据有效性、
        #     各硬门数值/阈值/结果、失败原因、n_pass 和被选成功 episode**" ✓
        #   实测 ✓ (s29): `runs=0`, 只有 n_pass=3 ✗ ⇒ 入口合同虽过, 但**追溯不完整** ✗
        #   ⇒ 本改 ✓: 逐次记录 episode_id + 各项真值 + 是否过门 + 失败原因 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _runs_v = []
            for _i_v in range(len(_v_lifts)):
                _rv = (_acc_reasons[_i_v] if _i_v < len(_acc_reasons) else {}) or {}
                _runs_v.append({
                    "run_index": int(_i_v),
                    "episode_id": (_v_eps[_i_v] if _i_v < len(_v_eps) else None),
                    "lift_final_cm": round(float(_v_lifts[_i_v]) * 100.0, 2),
                    "finger_ground_mm": (None if _v_gnd[_i_v] is None
                                         else round(float(_v_gnd[_i_v]), 2)),
                    "dmin_mm": (None if _v_dmin[_i_v] is None
                                else round(float(_v_dmin[_i_v]), 2)),
                    "true_grasp_ratio": (None if _v_tg[_i_v] is None
                                         else round(float(_v_tg[_i_v]), 3)),
                    "drift_mm": (None if _v_drift[_i_v] is None
                                 else round(float(_v_drift[_i_v]) * 1000.0, 2)),
                    "obj_supported": _v_sup[_i_v],
                    "gripper_ground_class": _v_ggc[_i_v],
                    "passed": bool(_acc_ok[_i_v] if _i_v < len(_acc_ok) else False),
                    "reasons": (list(_rv.get("reasons") or [])
                                + list(_rv.get("infeasible") or [])),
                })
            best_output['verification'] = {
                'lifts_final_cm': [round(_l * 100, 1) for _l in _v_lifts],
                'n_pass': _n_pass,
                'n_repeat': int(len(_runs_v)),
                'gate': 'lift_final>=3cm, >=2/3',
                'criteria': {'lift_cm': _acc_lift, 'gnd_mm': _acc_gnd,
                             'tg_min': _acc_tg, 'drift_mm': _acc_drift},
                'runs': _runs_v,
                # ★★ 任务04 (2026-09-28) ✓ md: "`selected_episode` **必须**在同 candidate/profile/
                #   controller/criteria 的 **passed 列表**中; 不能从搜索 `best_result` 补另一个
                #   episode ✗" —— 实测旧病: s31/s44 都写 `s1ep-000124`(搜索最优) 而通过的是
                #   128/129/130 ✗ ⇒ 记录与验证**不绑定** 🔴
                #   本改 ✓: 从**通过终验的 run** 里选; 无通过 ⇒ None (不编造 ✗)
                'selected_episode': next(
                    (_r.get("episode_id") for _r in _runs_v
                     if _r.get("passed") and _r.get("episode_id")), None),
                # ★ 任务04 ✓: `n_pass` **由列表计算** (不沿用别处计数 ✗)
                'n_pass_by_runs': int(sum(1 for _r in _runs_v if _r.get("passed"))),
                'data_validity': [bool(_r.get("passed")) for _r in _runs_v],
            }
            logger.warning(
                "[T07e4] ✓ verification_results 已记录 **%d 次重复 run 明细** "
                "(n_pass=%d; episodes=%s) ✓",
                len(_runs_v), int(_n_pass), [ _r.get("episode_id") for _r in _runs_v ])
        except Exception as _e74:  # noqa: BLE001
            logger.warning("[T07e4] runs 明细写入失败: %s", _e74)
            best_output['verification'] = {
                'lifts_final_cm': [round(_l * 100, 1) for _l in _v_lifts],
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
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【P0-4】**不可变动作记录** ✓ (审计 §P0-4 ✓) ★★★★
            #   审计原话 ✓: "建议: 获胜候选保存**不可变完整动作记录**, 而不是依赖 `_last_*`。
            #     最终验证和 Stage2 都**执行这份记录**, 保存实际 `base_pos/base_R/q_cmd/dt`
            #     及分段边界" ✓
            #   本实现 ✓: 在导出**之前**把动作冻结成一份记录 ✓ (只读副本 ✓),
            #     并记录**实际时长** (§64 ✓) 与**分段边界** ✓ ⇒ 后续验证/回放/Stage2
            #     一律消费这份记录 ✓, 不再"现场重算" ✗
            #   注 ✓: 该记录同时作为**唯一身份**用于 `_export_stage1_recipe` ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                # ⚠ 2026-09-22 修 BUG ✓: 本文件**未导入** `CONTROL_FREQ` ✗
                #   (实测 p03_snapshot: `冻结失败: name 'CONTROL_FREQ' is not defined` ✗)
                #   ⇒ 就地按共享来源取 ✓ (取不到则用 30Hz 的同源默认 ✓)
                try:
                    from physics_env import CONTROL_FREQ as _CF
                except Exception:  # noqa: BLE001
                    _CF = 30.0
                _sr_win = None
                if isinstance(best_result, dict):
                    _sr_win = best_result
                _cc = int(best_output.get('close_controls')
                          if best_output.get('close_controls') is not None
                          else STAGE1_CLOSE_CONTROLS)
                _sc = int(best_output.get('settle_controls')
                          if best_output.get('settle_controls') is not None
                          else STAGE1_SETTLE_CONTROLS)
                best_output['action_record'] = {
                    "frozen": True,
                    "target_name": target_name,
                    "base_pos": np.asarray(best_output['pos'], dtype=np.float64).copy(),
                    "base_R": np.asarray(best_output['R'], dtype=np.float64).copy(),
                    # ★ 2026-09-30 修复 ✗: 名义值 ≠ 实际下发 ⇒ Stage2 报 "anchor 与 recipe 不符"
                    #   本改: 优先用 rollout 回传的"T1b 钳位后实际生效值" ⇒ anchor 落在实际值域内 ✓
                    "q_cmd": float(best_output.get('s1_settle_end_q_cmd')
   if best_output.get('s1_settle_end_q_cmd') is not None
   else (best_output.get('s1_frozen_q_effective')
         if best_output.get('s1_frozen_q_effective') is not None
         else best_output['gripper_qpos'])),
                    "q_cmd_nominal": float(best_output['gripper_qpos']),
                    "q_source": ("rollout_effective(T1b-clamped)"
                                 if best_output.get('s1_frozen_q_effective') is not None
                                 else "nominal_candidate(未过钳位)"),
                    "dt": float(1.0 / float(_CF)),
                    "close_controls": _cc,
                    "settle_controls": _sc,
                    "segments": {
                        "close_start": int((_sr_win or {}).get('close_start', -1)),
                        "close_end": int((_sr_win or {}).get('close_end', -1)),
                        "settle_start": int((_sr_win or {}).get('settle_start', -1)),
                        "settle_end": int((_sr_win or {}).get('settle_end', -1)),
                    },
                    "origin": "winner_candidate",
                }
                self._stage1_action_record = best_output['action_record']
                logger.info(
                    "[P0-4 动作记录] ✓ 已冻结: q_cmd=%.4f, close/settle=%d/%d, dt=%.5fs, "
                    "base_pos=%s (后续验证/Stage2 必须执行**同一份** ✓)",
                    float(best_output['gripper_qpos']), _cc, _sc,
                    float(1.0 / float(_CF)),
                    np.round(best_output['pos'], 4).tolist())
            except Exception as _ear:  # noqa: BLE001
                logger.warning("[P0-4 动作记录] 冻结失败 (不阻断, 但交付一致性下降 ⚠): %s", _ear)
            # ★ T07b ✓ **强绑定**: winner 的身份/记录直接取自**它自己的 sim_result** ✓
            #   (`best_result` 就是被选中的那条 run ⇒ 不需要按 q 终点猜、也不会串到别的候选 ✗)
            try:
                if isinstance(best_output, dict):
                    _br = best_result if isinstance(best_result, dict) else {}
                    for _k7b in ("episode_id", "candidate_id", "immutable_record",
                                 "assessment", "source_frame"):   # ★ T07e3 ✓ 帧号也要一起带 ✓
                        if best_output.get(_k7b) is None and _br.get(_k7b) is not None:
                            best_output[_k7b] = _br[_k7b]
                    logger.warning(
                        "[T07b 绑定] ✓ winner episode=%s candidate=%s | 记录 %s (来自 best_result ✓)",
                        best_output.get("episode_id"), best_output.get("candidate_id"),
                        ("✓" if best_output.get("immutable_record") else "✗ 缺失"))
            except Exception as _e7b:  # noqa: BLE001
                logger.warning("[T07b 绑定] 失败(导出将报 WINNER_RECORD_MISSING ✓): %s", _e7b)
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
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T07d (2026-09-27) 【导出门禁 + staging】**不覆盖既有成功记录** ✓ ★★★★
                #   md T07.1 原话 ✓: "导出前做**全合同验证**并写入 staging 路径,
                #     成功后再更新正式 artifact 引用; **失败不覆盖既有成功记录**" ✓
                #   ⇒ 流程: 写 staging ⇒ 全合同校验 ⇒
                #       通过 ⇒ 更新正式 recipe (export_valid=True) ✓
                #       失败 ⇒ 只写 **.INVALID.json 诊断文件**, 正式文件**保持不动** ✗→✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                try:
                    from stage1_recipe import validate_recipe_contract as _vrc7d
                    _ok_7d, _errs_7d = _vrc7d(_recipe)
                except Exception as _ec7d:  # noqa: BLE001
                    _ok_7d, _errs_7d = False, ["contract check 异常: %s" % _ec7d]
                _ok_7d = bool(_ok_7d and (_recipe.get("winner_binding") or {}).get(
                    "record_present", False))
                _recipe["contract_errors"] = list(_errs_7d)
                _recipe["export_valid"] = bool(_ok_7d)
                try:
                    _stage_dir = _save_dir / "staging"
                    _stage_dir.mkdir(parents=True, exist_ok=True)
                    (_stage_dir / "stage1_grasp_recipe.json").write_text(
                        _json.dumps(_recipe, indent=2, sort_keys=True), encoding="utf-8")
                except Exception as _es7d:  # noqa: BLE001
                    logger.warning("[T07d] staging 写入失败: %s", _es7d)
                if _recipe["export_valid"]:
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ 任务04 (2026-09-28) 【成功产物**只读**, 新 revision **另存**】 ✓ ★★★★
                    #   md §04 原话 ✓: "**成功产物只读, 新 revision 另存**; 诊断记录**不能更新
                    #     正式成功指针**" ✓
                    #   实测旧病 ✗: 只要本次合同过就直接 `write_text` 覆盖 ⇒ 既有成功记录
                    #     (如 s31: export_valid=True, n_pass=3) 可被**后来的诊断 run**顶掉 ✗
                    #   本改 ✓: 目标已存在**且是成功产物**(export_valid ∧ n_pass≥2) ⇒
                    #     **绝不覆盖**, 本次写成 `stage1_grasp_recipe.rev<时间戳>.json` ✓;
                    #     否则(不存在/旧的是无效产物) ⇒ 正常写正式名 ✓
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    _p_final = _save_dir / "stage1_grasp_recipe.json"
                    _prev_ok = False
                    try:
                        if _p_final.exists():
                            _prev = _json.loads(_p_final.read_text())
                            _prev_ok = bool(_prev.get("export_valid")) and int(
                                ((_prev.get("verification_results") or {}).get("n_pass") or 0)) >= 2
                    except Exception:  # noqa: BLE001
                        _prev_ok = False
                    if _prev_ok:
                        import time as _t_rev
                        _p_rev = _save_dir / ("stage1_grasp_recipe.rev%s.json"
                                              % _t_rev.strftime("%Y%m%d-%H%M%S"))
                        try:
                            _p_rev.write_text(_json.dumps(_recipe, indent=2, sort_keys=True),
                                              encoding="utf-8")
                            logger.warning(
                                "[任务04] ✓ 既有**成功** recipe 为只读 ⇒ 本次**另存**为 %s ✓ "
                                "(不覆盖成功记录 ✗ md §04 ✓)", _p_rev.name)
                        except Exception as _er_rev:  # noqa: BLE001
                            logger.error("[任务04] ✗ revision 另存失败: %s", _er_rev)
                    else:
                        _p_final.write_text(
                            _json.dumps(_recipe, indent=2, sort_keys=True), encoding="utf-8")
                        logger.warning(
                            "[T07d] ✓ **合同校验通过** ⇒ 正式 recipe 已更新 | export_valid=True | "
                            "winner=%s candidate=%s | N=%d 相位=%s",
                            (_recipe.get("winner_binding") or {}).get("episode_id"),
                            (_recipe.get("winner_binding") or {}).get("candidate_id"),
                            len(_recipe.get("actual_q_commands") or []),
                            sorted(k for k in (_recipe.get("phase_boundaries") or {})
                                   if k != "n"))
                else:
                    try:
                        (_save_dir / "stage1_grasp_recipe.INVALID.json").write_text(
                            _json.dumps(_recipe, indent=2, sort_keys=True), encoding="utf-8")
                    except Exception:  # noqa: BLE001
                        pass
                    logger.error(
                        "[T07d] ✗ **合同校验失败** ⇒ export_valid=false, "
                        "**正式 recipe 保持不动(不覆盖既有成功记录 ✓)** | errors=%s",
                        list(_errs_7d)[:6])
            logger.info(f"[Stage1] 最优 grasp 数据已保存到 {_save_path}")
        except Exception as _e:
            logger.warning(f"[Stage1] 保存最优 grasp 数据失败: {_e}")

        return best_output

    # === _export_stage1_recipe ===
    def _s1_criteria(self):
        """★ 2026-09-24 【方案 §11.5 / §11.6-A.4】**唯一**的 Stage1 验收 criteria ✓

        为什么必须唯一 ✓ (复审原话 ✓): \"搜索默认 **10cm/30mm**, 终验默认 **3cm/20mm**;
          同一个函数**不等于**同 criteria; 代码默认**仍会分叉**\" ✗ ⇒ 搜索挑出的"成功"候选
          到终验被更严的门否掉 ⇒ **搜得多、交付不出来** ✗
        本改 ✓: 搜索 / Top-K / 冻结复验 / 终验 **共用这一份** ✓, 并附
          `criteria_hash` ⇒ 任何一轮用了非默认值都可追溯 ✓ (§11.5: "统一 criteria 对象并
          保存 hash, 从搜索传至终验, 不是在每个入口分别读两套环境变量" ✓)
        默认沿用现有终验门 ✓: lift≥3cm / 指尖离桌≥0mm / 真夹持率≥0.5 / drift≤20mm ✓
        """
        import hashlib as _hl
        _c = {
            "lift_cm": float(os.environ.get("_S1_ACC_LIFT_CM", "3.0") or 3.0),
            "gnd_mm": float(os.environ.get("_S1_ACC_GND_MM", "0.0") or 0.0),
            "tg_min": float(os.environ.get("_S1_ACC_TG_MIN", "0.5") or 0.5),
            "drift_mm": float(os.environ.get("_S1_ACC_DRIFT_MM", "20.0") or 20.0),
        }
        _src = "|".join(f"{k}={_c[k]!r}" for k in sorted(_c))
        _c["criteria_source"] = _src
        _c["criteria_hash"] = _hl.sha256(_src.encode("utf-8")).hexdigest()[:12]
        return _c

    def _export_stage1_recipe(self, target_name, best_output):
        """Build the transferable Stage1 recipe + scene/object digest.

        The digest pins the grasped object's scale (``bbox_size`` after
        ``_OBJ_SCALE``), its reset pose, mass, ground and solver parameters so
        Stage2 can prove it is replaying the same physical anchor.
        """
        import json as _json

        if not target_name:
            return None
        # ★★★★ 2026-09-21 【P0 交付一致性 · 单一来源】 ★★★★
        #   铁证链 ✓:
        #     ① G1 成功配方: Stage1 验证轮实际命令 `PD_target = 0.0026` (**含 6.0mm 过闭合** ✓)
        #        而交付产物写的是 `gripper_qpos = close_ctrl_q[-1] = 0.0111` (**不含过闭合** ✗)
        #        ⇒ Stage2/3 按 0.0111 闭合 ⇒ 指缝 = 30.4+22.1 = **52.5mm**
        #          vs 物体跨度 47.7mm ⇒ **留 4.8mm 空隙** ⇒ 下游**夹空**
        #          (实测 `g1_oc6_0920` / `g1_verify_0921`: `Stage2 FAIL(PRE_F50_READY)`,
        #           `[Stage2 contact] 全程无夹爪相关接触` ✓ 完全自洽 ✓)
        #     ② 另一条更硬的约束: `stage2_reconstruct_mixin.py:567-573` 会**逐位严格校验**
        #          `stage1_close_q_sequence(q_open, **anchor.gripper_qpos**, 16)`
        #          是否等于 `recipe["close_ctrl_q"]` (`np.allclose(..., atol=0.0)` ✗ 不容差)
        #        ⇒ 只要 recipe 与 anchor 用的不是**同一个值**, 立刻
        #          `ValueError: Stage2 close command does not match the Stage1 recipe` ✗
        #        (实测两次: ① 只改 recipe ⇒ 崩; ② recipe 取"最后一次 rollout"而 anchor 取
        #         "获胜候选" ⇒ 差 2.87e-5 ⇒ 仍崩 ✗)
        #   本改 ✓: 两者**同一次来源** —— 都用**获胜候选自己**的真实命令值 ✓
        #     (`_eval_pose` 已把 `sim_result['gripper_qpos_target']` 与该候选一一对应记录 ✓,
        #      三处 winners 导出也已统一取该字段 ✓ ⇒ recipe 直接跟随, 不再另取 ✓)
        q_grasp = float(best_output['gripper_qpos'])
        q_grasp_actual = float(getattr(self, '_last_grip_target_s1', q_grasp) or q_grasp)
        if abs(q_grasp_actual - q_grasp) > 1e-9:
            logger.info(
                "[Stage1][recipe] 获胜候选闭合目标=%.6f, 最后一次 rollout=%.6f (仅诊断; "
                "recipe/anchor 一律用**获胜候选值**以保证 Stage2 严格校验通过 ✓)",
                q_grasp, q_grasp_actual,
            )
        # nominal (不含过闭合) 供追溯: 由 bbox 直接重算 ⇒ 无新状态 ✓
        try:
            _bs = (getattr(self, 'obj_info', {}) or {}).get(target_name, {}).get('bbox_size')
            q_grasp_nominal = (float(self.compute_gripper_qpos(_bs))
                               if _bs is not None else q_grasp)
        except Exception:  # noqa: BLE001
            q_grasp_nominal = q_grasp
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
        # ★ S2-1 (2026-09-27) ✓ 取**本轮获胜 run** 实际下发的 q 序列
        #   ⇒ 有值 ⇒ `q_sequence_source=**actual_executed**` ✓ (S1-D 要求, 也是本任务验收项 ✓)
        #   ★ S2-1b ✓: 导出在另一段作用域, 不能只看"当前"属性 ⇒ **按 winner 的 q 终点匹配**
        #     从 `_s1_exec_q_by_rollout` 里取**最后一点 == best_output['gripper_qpos']** 的那条 ✓
        #     (匹配不到才退回最长的一条, 并**告警** ✓; 都不行才用当前属性)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T07b (2026-09-27) 【强绑定】**只认 winner 自带的 immutable_record** ✓ ★★★★
        #   md T07.1 明令禁止 ✗: "**不再凭 q 终点、最后一次缓存或'最长轨迹'找 record**"
        #     (我上一版 S2-1b 正是"按 q 终点匹配 + 最长回退" ✗ ⇒ 可能把 B 的曲线安到 A 头上 ✗)
        #   ⇒ 本版: 记录由 rollout **返回时快照**并随 winner 传递; 缺失即报
        #     **WINNER_RECORD_MISSING** + `export_valid=false` ⇒ **不静默造假** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        from stage1_recipe import (bind_winner_record, derive_phase_boundaries,
                                   validate_recipe_contract, hash_resolved_profile,
                                   STAGE1_RECIPE_SCHEMA)   # ★ T07b/T07d ✓ 纯函数 ⇒ 可单测 ✓
        _exec_q, _exec_trace = [], {}
        _rec7b, _err7b = bind_winner_record(best_output)
        # ★ T07e ✓ 合规兜底: 按 **winner 自己的 episode_id** 从登记表取回 ✓
        #   (这**不是**"最后缓存/最长轨迹" ✗: id 由该 run 自己产生并唯一 ✓)
        if _err7b is not None:
            try:
                _ep_id7e = str(best_output.get("episode_id") or "")
                _reg7e = getattr(self, "_s1_episode_records", None) or {}
                _ent7e = _reg7e.get(_ep_id7e) if _ep_id7e else None
                if isinstance(_ent7e, dict) and isinstance(_ent7e.get("record"), dict):
                    best_output["immutable_record"] = _ent7e["record"]
                    best_output.setdefault("candidate_id", _ent7e.get("candidate_id"))
                    best_output.setdefault("source_frame", _ent7e.get("source_frame"))
                    _rec7b, _err7b = bind_winner_record(best_output)
                    logger.warning(
                        "[T07e][recipe] ✓ 按 episode_id=%s 从登记表取回记录 (**%d 帧**) ✓",
                        _ep_id7e, len((_rec7b or {}).get("q_cmd") or []))
                else:
                    logger.error("[T07e][recipe] ✗ 登记表也无 episode=%r 的条目 ⇒ "
                                 "无法取回 winner 记录 ✗", _ep_id7e)
            except Exception as _e7e2:  # noqa: BLE001
                logger.warning("[T07e][recipe] 登记表取回失败: %s", _e7e2)
        if _err7b is None:
            _exec_q = [float(_x) for _x in (_rec7b.get("q_cmd") or [])]
            _exec_trace = {"t": list(_rec7b.get("t_s") or []),
                           "base_pos": list(_rec7b.get("base_pos") or []),
                           "q": list(_rec7b.get("q_cmd") or []),
                           # ★ T07c/T07d ✓ 完整字段 (相位/旋转/索引 ⇒ 合同校验用 ✓)
                           "phase": list(_rec7b.get("phase") or []),
                           "control_index": list(_rec7b.get("control_index") or []),
                           "T_W_G": list(_rec7b.get("T_W_G") or []),
                           "source_phase": list(_rec7b.get("source_phase") or []),
                           "q_act": list(_rec7b.get("q_act") or []),
                           "obj_z": list(_rec7b.get("obj_z") or []),
                           "valid": list(_rec7b.get("valid") or [])}
            logger.warning("[T07b][recipe] ✓ 强绑定: episode=%s candidate=%s | **%d 帧**实际序列",
                           _rec7b.get("episode_id"), _rec7b.get("candidate_id"), len(_exec_q))
        else:
            logger.error("[T07b][recipe] ✗ **WINNER_RECORD_MISSING**: winner=%s candidate=%s "
                         "**没有 immutable_record** ⇒ export_valid=false ✗ "
                         "(**不用** q 终点/最长轨迹/最后缓存替代 ✓)",
                         best_output.get("episode_id"), best_output.get("candidate_id"))
        if not _exec_q:
            logger.error("[S2-1][recipe] ✗ **拿不到实际执行序列** ⇒ 只能重生成名义曲线 "
                         "(`q_sequence_source=regenerated_nominal` ✗ 违反 §S1-D ✗)")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-28 修 `close_ctrl_q` 语义 ✗ (R03 范围: "重建 close 切片"/"统一旧字段") ✓
        #   实测铁证 ✗ (output/gif_s31 + s31/s44 三处一致):
        #     `recipe["close_ctrl_q"]` 长度 **350** (= 全段实际 q ✗),
        #     而消费侧两处都期望它是 **close 相位段**:
        #       ① `stage2_reconstruct_mixin.verify_stage1_transfer_equivalence:5261`
        #          要求 `stored_q.shape[0] == len(实际 close 段)` 且逐位相等 (atol=1e-12)
        #       ② `_stage1_reuse_report` 同口径
        #     ⇒ 长度 350 ≠ 16 ⇒ `--test-stage1-transfer` **恒 FAIL(STAGE1_REUSE_MISMATCH)**
        #       ⇒ 该入口从未通过 ⇒ "成功录像"永远拿不到 ✗
        #   本改 ✓: 只用 winner 记录自带的 `phase` 切出 **close 段** 传给 `close_q_actual` ✓;
        #     全段仍在 `q_cmd`/`actual_q_commands` 里逐点保留 ✓ (**不是**尾裁剪 ✗, 一个点都不扔 ✓)
        #   ⚠ 无 phase 或长度不齐 ⇒ 回落全段 + 明确警告 (不静默 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _phase_exec28 = [str(x) for x in (_exec_trace.get("phase") or [])]
        _close_seg28 = []
        if _exec_q and len(_phase_exec28) == len(_exec_q):
            _close_seg28 = [float(_q) for _q, _ph in zip(_exec_q, _phase_exec28)
                            if _ph in ("close", STAGE1_PHASE_CLOSE)]
            logger.warning("[R03][recipe] ✓ `close_ctrl_q` 取 **close 相位实际段** %d 点 "
                           "(全段 %d 点保留在 q_cmd ✓; 相位=%s ✓)",
                           len(_close_seg28), len(_exec_q),
                           sorted({_p for _p in _phase_exec28 if _p in ("close", "settle")})
                           or sorted(set(_phase_exec28))[:4])
        else:
            logger.error("[R03][recipe] ✗ winner 记录 phase 缺失/长度不齐 (%d vs %d) ⇒ "
                         "`close_ctrl_q` 回落到**全段** ⇒ 下游 reuse 核对会判 MISMATCH ✗",
                         len(_phase_exec28), len(_exec_q))
        recipe = build_stage1_grasp_recipe(
            close_q_actual=(_close_seg28 or _exec_q or None),
            target_name=target_name,
            q_open=stage_q_open(),
            q_grasp=q_grasp,
            physics_config=physics_config,
            scene_digest=digest,
            q_grasp_nominal=q_grasp_nominal,
            # ★ 2026-09-22 【P0-4 修复】导出**实际执行的时长** ✓ (不再固定用默认 16/10 ✗)
            #   审计文档 §P0-4 ✓: "`_S1_CLOSE_FRAMES` / `_S1_SETTLE_FRAMES` 可以改变
            #     实际执行时长, 但 recipe 导出**没有传入这些实际值**, 仍采用默认 16/10 ✗
            #     ⇒ 默认配置相同时不触发, **覆盖环境变量时会破坏回放一致性** ✗"
            #   ⇒ 取获胜候选记录里的**实际值** ✓ (缺失才回退到共享常量 ✓, 显式 None ✓)
            close_controls=int(best_output.get('close_controls')
                               if best_output.get('close_controls') is not None
                               else STAGE1_CLOSE_CONTROLS),
            settle_controls=int(best_output.get('settle_controls')
                                if best_output.get('settle_controls') is not None
                                else STAGE1_SETTLE_CONTROLS),
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
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ S2-1 (2026-09-27) 【§S1-D 冻结 Stage2 输入】补齐契约字段 ✓ ★★★★
        #   方案 §S1-D 原话 ✓: 保存 source_frame, task_grasp_frame, optimized_T,
        #     optimized_preload, **actual_control_times / actual_base_commands /
        #     actual_q_commands / phase_boundaries**, physics_hash, canonical_state_hash,
        #     criteria_revision, verification_results
        #     · `source_frame` 是**位姿来源**; `task_grasp_frame` 是**原任务抓取时刻**,
        #       两者**不混用** (选中 F42 姿态, 不自动把原任务 F50 闭合时刻改成 F42 ✓)
        #   ⚠ 只读补充 ✓: 不改执行器/物理/成功门 ✓; 取不到就置 None 并告警 (不静默造假 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            import hashlib as _hl
            import json as _js
            _tr3 = _exec_trace or (getattr(self, "_s1_exec_trace", None) or {})
            recipe["actual_control_times"] = [float(x) for x in (_tr3.get("t") or [])]
            recipe["actual_base_commands"] = [[float(v) for v in _p]
                                              for _p in (_tr3.get("base_pos") or [])]
            recipe["actual_q_commands"] = [float(x) for x in _exec_q]
            _tf = int(locals().get("_TRAVEL_FRAMES", int(os.environ.get(
                "_S1_TRAVEL_FRAMES", "0") or 0)))
            _cf = int(locals().get("_CLOSE_FRAMES", int(os.environ.get(
                "_S1_CLOSE_FRAMES", "130") or 130)))
            _sf = int(locals().get("_SETTLE_FRAMES", int(os.environ.get(
                "_S1_SETTLE_FRAMES", "15") or 15)))
            # ★ T07d ✓: 相位边界**从本 run 的 phase 数组推导** (半开区间, 不重叠不遗漏 ✓)
            #   (旧写法按帧数硬凑 ✗ ⇒ 与实际记录不符 ⇒ 合同校验会拒 ✓)
            recipe["phase_boundaries"] = derive_phase_boundaries(
                list((_exec_trace or {}).get("phase") or []))
            recipe["phase_counts"] = {"travel": _tf, "close": _cf, "settle": _sf}
            # ★ T07d ✓ 三类 hash 指向**实际 resolved profile** (不是 env 字符串/预期默认 ✗)
            recipe["schema_version"] = STAGE1_RECIPE_SCHEMA
            recipe["physics_hash"] = hash_resolved_profile(physics_config)
            _ctl_prof = {}
            try:
                import physics_utils as _pu7d
                for _k7d in ("FINGER_PD_KP", "FINGER_PD_KD", "FINGER_PD_FORCE_LIMIT",
                             "GRIPPER_FORCE", "GRIPPER_STIFFNESS", "GRIPPER_DAMPING",
                             "GRIPPER_MAX_OPEN", "FINGER_GAP0", "PHYSICS_TIMESTEP"):
                    if hasattr(_pu7d, _k7d):
                        _ctl_prof[_k7d] = getattr(_pu7d, _k7d)
            except Exception as _e7d1:  # noqa: BLE001
                logger.warning("[T07d] controller profile 读取失败: %s", _e7d1)
            # ★ T07e ✓ 修 `side` 未定义 ✗ (导出函数没有 side 参数 ⇒ 实测 NameError
            #   'side' is not defined ⇒ 整个 S1-D 块被跳过 ⇒ 合同全空 ✗)
            _side_h = str(getattr(self, "side", None)
                          or os.environ.get("_TC_SIDE", "left") or "left")
            recipe["controller_hash"] = hash_resolved_profile({
                "profile": _ctl_prof, "side": _side_h,
                "close_profile": str(recipe.get("close_profile", "")),
                "close_controls": int(recipe.get("close_controls", -1)),
                "settle_controls": int(recipe.get("settle_controls", -1))})
            recipe["canonical_state_hash"] = str(recipe.get("scene_digest_hash", ""))
            recipe["initial_state_hash"] = hash_resolved_profile({
                "digest": recipe.get("canonical_state_hash", ""),
                "target": str(recipe.get("target_object", ""))})
            recipe["criteria_revision"] = str(
                os.environ.get("_CRITERIA_REVISION", "criteria_v2"))
            # ★ T07d ✓ criteria_hash 指向**实际判据配置** (不是 env 字符串 ✗)
            recipe["criteria_hash"] = hash_resolved_profile({
                "criteria_revision": recipe["criteria_revision"],
                "gate": {_k: os.environ.get(_k) for _k in (
                    "_S1_GATE_LIFT_CM", "_S1_GATE_FGND_MM", "_S1_GATE_TG_RATIO",
                    "_S1_FROZEN_Q_TOL_MM", "_S1_CLOSE_OVERCLOSURE_MM",
                    "_S1_PRELOAD_M", "_S1_T8_EXTRA_MM")},
                "task_grasp_frame": int(recipe.get("task_grasp_frame", 50))})
            recipe["winner_binding"] = {
                "episode_id": best_output.get("episode_id"),
                "candidate_id": best_output.get("candidate_id"),
                "record_present": bool(_exec_q),
                "record_frames": int(len(_exec_q)),
            }
            # ★ T07b ✓: 缺实际记录 ⇒ **export_valid=false** ⇒ 不得称"仍可供 Stage2 使用" ✓
            recipe["export_valid"] = bool(_exec_q)
            recipe["verification_results"] = {
                "q_sequence_source": str(recipe.get("q_sequence_source", "")),
                "close_controls": int(recipe.get("close_controls", -1)),
                "settle_controls": int(recipe.get("settle_controls", -1)),
                "n_close_q": int(len(recipe.get("close_ctrl_q", []) or [])),
            }
            # source_frame = 位姿来源(4x4);  task_grasp_frame = 原任务抓取时刻(帧号) ⇒ 不混用 ✓
            # ★ T07d ✓ md 原话: "`source_frame` = **原始 MANO 候选整数帧号**(如 42);
            #   **不是矩阵、控制索引或优化后的时间**"; "`task_grasp_frame` = 原任务抓取整数帧号,
            #   **当前默认 50**; 不因选 F42 或闭合多了几帧而改变" ✓
            _sf_idx = best_output.get("source_frame")
            try:
                recipe["source_frame"] = (int(round(float(_sf_idx)))
                                          if _sf_idx is not None else None)
            except Exception:  # noqa: BLE001
                recipe["source_frame"] = None
            recipe["task_grasp_frame_default"] = 50
            recipe["task_grasp_frame"] = int(os.environ.get("_TASK_GRASP_FRAME", "50"))
            # 位姿放**独立字段** ✓ (4×4, 与帧号语义分开 ✓)
            _sf_p = best_output.get("pos")
            _sf_R = best_output.get("R")
            if _sf_p is not None and _sf_R is not None:
                _T_sf = np.eye(4, dtype=np.float64)
                _T_sf[:3, :3] = np.asarray(_sf_R, dtype=np.float64)
                _T_sf[:3, 3] = np.asarray(_sf_p, dtype=np.float64).reshape(-1)[:3]
                recipe["optimized_T_W_G"] = [[float(_v) for _v in _row] for _row in _T_sf]
            else:
                recipe["optimized_T_W_G"] = None
            logger.warning(
                "[S2-1][recipe] ✓ §S1-D 冻结: q_sequence_source=**%s** | 实际下发 %d 帧 "
                "(时标/基座/相位边界已写入) | physics_hash=%s | source_frame=%s "
                "task_grasp_frame=%d (**两者不混用** ✓)",
                recipe.get("q_sequence_source"), int(len(_exec_q)),
                str(recipe.get("physics_hash"))[:12],
                ("✓" if recipe.get("source_frame") is not None else "n/a"),
                int(recipe.get("task_grasp_frame", -1)))
        except Exception as _es1:  # noqa: BLE001
            logger.warning("[S2-1][recipe] §S1-D 字段补充失败(recipe 仍可用 ✓): %s", _es1)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T07e (2026-09-27) 【合同关键字段放在 S1-D try **之外**】 ✓ ★★★★
        #   实测 ✓ (s26): S1-D 块里 `side` 未定义 ⇒ 异常 ⇒ 若别名也写在同一个 try 内,
        #     `time_s/T_W_G_cmd/phase/control_index` 会**全空** ✗ ⇒ 合同必拒 ✗
        #   ⇒ 本改 ✓: 这些字段与合同校验独立执行 ⇒ 局部失败**不再连带清零** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            recipe["schema_version"] = STAGE1_RECIPE_SCHEMA
            recipe["time_s"] = list(recipe.get("actual_control_times") or [])
            recipe["actual_q_commands"] = list(_exec_q)
            recipe["q_cmd"] = list(_exec_q)        # ★ T07e3 ✓ md 字段名别名 (Stage2 合同按此 ✓)
            # ★ T07e3 ✓ source_frame 兜底链: best_output → 登记表 → 同族帧号 (全是**真实帧号** ✓)
            if recipe.get("source_frame") is None:
                _sf_b = (best_output or {}).get("source_frame")
                if _sf_b is None:
                    _ent_b = (getattr(self, "_s1_episode_records", None)
                              or {}).get(str((best_output or {}).get("episode_id") or ""))
                    if isinstance(_ent_b, dict):
                        _sf_b = _ent_b.get("source_frame")
                if _sf_b is None:
                    _sf_b = getattr(self, "_s1_last_cand_frame", None)
                if _sf_b is not None:
                    try:
                        recipe["source_frame"] = int(round(float(_sf_b)))
                        logger.warning("[T07e3][recipe] ✓ source_frame 兜底取回 = %d ✓",
                                       recipe["source_frame"])
                    except Exception:  # noqa: BLE001
                        pass
            recipe["T_W_G_cmd"] = list((_exec_trace or {}).get("T_W_G") or [])
            recipe["phase"] = list((_exec_trace or {}).get("phase") or [])
            recipe["control_index"] = list((_exec_trace or {}).get("control_index") or [])
            recipe["source_phase"] = list((_exec_trace or {}).get("source_phase") or [])
            recipe["observations"] = {
                "q_act": list((_exec_trace or {}).get("q_act") or []),
                "obj_z": list((_exec_trace or {}).get("obj_z") or []),
                "valid": list((_exec_trace or {}).get("valid") or [])}
            recipe["phase_boundaries"] = derive_phase_boundaries(
                list((_exec_trace or {}).get("phase") or []))
            recipe["winner_binding"] = {
                "episode_id": (best_output or {}).get("episode_id"),
                "candidate_id": (best_output or {}).get("candidate_id"),
                "record_present": bool(_exec_q),
                "record_frames": int(len(_exec_q)),
            }
            recipe["export_valid"] = bool(_exec_q)
            _ok_c2, _errs_c2 = validate_recipe_contract(recipe)
            recipe["contract_errors"] = list(_errs_c2)
            # ★ 任务04 ✓: `n_pass` 必须与 runs 列表**自洽** (md: 由列表计算 ✓)
            try:
                _vr04 = recipe.get("verification_results") or {}
                _runs04 = list(_vr04.get("runs") or [])
                if _runs04:
                    _np04 = int(sum(1 for _r in _runs04 if _r.get("passed")))
                    _vr04["n_pass_by_runs"] = _np04
                    if _vr04.get("n_pass") is not None and int(_vr04["n_pass"]) != _np04:
                        logger.error("[任务04] ✗ n_pass=%s 与 runs 统计=%d **不一致** ⇒ "
                                     "记录不可信 ✗", _vr04.get("n_pass"), _np04)
                        _errs_c2 = list(_errs_c2) + [
                            "n_pass(%s) != runs 统计(%d) ⇒ 记录与验证不绑定 ✗"
                            % (_vr04.get("n_pass"), _np04)]
                    else:
                        _vr04["n_pass"] = _np04
                    # selected 必须在 passed 列表内 ✓ (md §04)
                    _sel04 = _vr04.get("selected_episode")
                    _passed04 = [_r.get("episode_id") for _r in _runs04 if _r.get("passed")]
                    if _sel04 is not None and _sel04 not in _passed04:
                        logger.error("[任务04] ✗ selected_episode=%s 不在 passed=%s ⇒ 拒绝 ✗",
                                     _sel04, _passed04)
                        _errs_c2 = list(_errs_c2) + [
                            "selected_episode %s 不在 passed 列表 %s 中 ✗" % (_sel04, _passed04)]
                    recipe["verification_results"] = _vr04
                    # ★ 任务04 ✓: 用**独立可验证的纯函数**再核一遍 (md: 抽成可独立验证入口 ✓)
                    try:
                        from stage1_recipe import assert_verification_binding as _avb04
                        _ok04b, _errs04b = _avb04(_vr04)
                        if not _ok04b:
                            logger.error("[任务04] ✗ 绑定校验未过: %s", _errs04b[:3])
                            _errs_c2 = list(_errs_c2) + list(_errs04b)
                    except Exception as _e04b:  # noqa: BLE001
                        logger.warning("[任务04] 绑定纯函数校验异常: %s", _e04b)
                    logger.warning("[任务04] ✓ 绑定自洽: n_pass=%s | selected=%s | passed=%s",
                                   _vr04.get("n_pass"), _sel04, _passed04)
            except Exception as _e04:  # noqa: BLE001
                logger.warning("[任务04] 绑定校验失败(不阻断): %s", _e04)
            recipe["export_valid"] = bool(_ok_c2 and _exec_q)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T07e3 (2026-09-27) 【verification_results 接上真实重复 run】 ✓ ★★★★
            #   md T07.1 ✓: "`verification_results`: **全部重复 run 的 ID、数据有效性、
            #     各硬门数值/阈值/结果、失败原因、n_pass 和被选成功 episode**"
            #   ⇒ 归档对象里已有 `best_output['verification']` (写 npz 的那个 ✓)
            #     ⇒ 直接接入 ⇒ Stage2 入口合同(evaluates n_pass/runs)才有依据 ✓
            #   ⚠ 不编造 ✓: `verification` 缺失/无 n_pass ⇒ 保持空, 入口门将拒绝 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                _ver_b = (best_output or {}).get("verification")
                if isinstance(_ver_b, dict):
                    recipe["verification_results"] = {
                        "runs": list(_ver_b.get("runs") or _ver_b.get("episodes") or []),
                        "n_pass": _ver_b.get("n_pass"),
                        "n_repeat": _ver_b.get("n_repeat"),
                        "criteria": _ver_b.get("criteria"),
                        "failure_reason": _ver_b.get("failure"),
                        # ★ 任务04 ✓: 同上 —— 只认 **passed run** 的 episode; 无 ⇒ None ✗
                        "selected_episode": next(
                            (_r.get("episode_id") for _r in (_ver_b.get("runs") or [])
                             if _r.get("passed") and _r.get("episode_id")), None),
                    }
            except Exception as _ev3:  # noqa: BLE001
                logger.warning("[T07e3][recipe] verification_results 接入失败: %s", _ev3)
            if not _ok_c2:
                logger.error("[T07e][recipe] 合同未过: %s", list(_errs_c2)[:6])
        except Exception as _ee2:  # noqa: BLE001
            logger.error("[T07e][recipe] 合同字段写入失败: %s", _ee2)
        return recipe
