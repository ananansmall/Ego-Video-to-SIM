"""Shared Stage1 grasp recipe and scene/object digest.

This module is the single source of truth for the physically verified Stage1
grasp primitive:

* ``STAGE1_CLOSE_CONTROLS`` = 20 close controls following a cubic smoothstep
  ``s(t) = t²(3-2t)`` from ``q_open`` to ``q_grasp``;
* ``STAGE1_SETTLE_CONTROLS`` = 10 settle controls holding ``q_grasp``.

Stage1 exports this recipe (together with a scene/object digest) into its
artifact.  Stage2 reads the recipe and must not re-hardcode the constants, so
the grasped object's scale/position and the finger command are provably the
same in both stages (plan §3.1 ``stage1_grasp_recipe``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os

import numpy as np

logger = logging.getLogger(__name__)

# 2026-09-13 (用户定策): 24 → 5. 用户要求"除了抓取帧我能给它五帧, 其他都得去和处理后的
# 夹爪轨迹对齐" —— 抓取帧处只保留 5 个控制点 (3 闭合 + 2 保持).
# 物理可行性 (已核算): 手指从 q_open=0.05 起, 只需**约 10mm 自由行程**即可触及 40mm 物体
# (实测 act 在 q≈0.035~0.045 处触物), 按 PD 时间常数 τ=D/K=0.2s 约 2 帧到位; 而夹持力
# ∝ 位置误差 (K×Δq) 是**瞬时建立**的, 不需要时间积累 (预紧下 Δq≈0.024 → ~24N). 故 5 帧足够.
# 注: 这是 Stage1/Stage2 共享的单一来源 (stage1_reuse_report 会校验两边一致), 必须在此处改.
STAGE1_CLOSE_CONTROLS = 16
STAGE1_SETTLE_CONTROLS = 10
STAGE1_STOP_CONTROLS = STAGE1_CLOSE_CONTROLS + STAGE1_SETTLE_CONTROLS
# ★★★★ 2026-09-17 5 → 16 (用户: "continue" / 目标: 让 CoACD 线的 Stage2 通过) ★★★★
#   上面 2026-09-13 那条"5 帧足够"的核算, 前提在 `_OBJ_SCALE=0.5` 下**已不成立** ✗:
#     · 当时按"~10mm 自由行程即触物 / act q≈0.035~0.045 触物 / 2 帧到位"估算 ✓
#       —— 那是**40mm 级物体** (scale 更大) 的情形 ✓
#     · 今天 `_OBJ_SCALE=0.5` 实测 (output/dumps 闭合段逐点转储 ✓, 已复现三次):
#         行程 q: 0.0500(close_start) → **0.0130**(双侧触物) = **37mm** ✗ (3.7 倍 ✗)
#         到位需 **~14 个控制点** ✗ (7 倍 ✗)
#     · 后果: 5 个闭合点 (idx 51-55) 走完时手指才到 act=0.0349 ✗,
#       一直拖到 idx 63-64 才双侧接触 ✗ ⇒ 静置窗 (56-65) 前 8 个样本必然单侧 ✗
#       ⇒ `PROVISIONAL_GRASP` 的 `sum(bilateral) >= settle_n-2` **结构性达不到** ✗
#       ⇒ 这也解释了"预紧每 +0.002 只多 1 个双侧样本" ✗ (在跟 ~10 点的结构缺口较劲 ✗)
#   取 16 = 实测 14 + 2 余量 ✓ (原值 24 也可, 但会拖长 timeline ✓)
#   注: 本常量是 Stage1/Stage2 的**单一来源** ✓ ⇒ 两边 timeline 与
#       `stage1_close_q_sequence(q_open, q_grasp, N)` 一起自动同步 ✓,
#       `stage1_reuse_report` 会校验两边一致 ✓
#   附: 这是**控制点数** (timeline 形状) ✓, 不是物理常量 ✗;

STAGE1_CLOSE_PROFILE = "cubic_smoothstep"
STAGE1_PHASE_CLOSE = "STAGE1_CLOSE"
STAGE1_PHASE_SETTLE = "STAGE1_SETTLE"
STAGE1_RECIPE_REVISION = "stage1_grasp_recipe_v1"


def coacd_key_criteria(pose_R, pose_pos, obj_center, hulls_w, half_sep,
                       finger_origin_offset=0.03689,
                       gate_center_mm=2.0, gate_clear_mm=1.0,
                       full_clear_mm=3.0, full_axial_mm=10.0):
    """CoACD 几何关键判据 —— **Stage1 reward 与 Stage2 判据的共享单一来源**.

    动机 (用户 2026-09-18): "stage1 和 stage2 虽然代价不同, 但是在关键点上肯定得一样".
      一个显式优化、一个隐式要求 ⇒ 必然打架. 实测 (output/qual2, CoACD 64 凸包):
        Stage1 选出的"最优位姿": c_center = **6~14mm** / c_clear ≈ **-1mm**
        而 Stage2 要求 "双侧对齐夹住" ⇒ 必然过不了.
      故把"对准物心 / 接近余量 / 轴向位置"这三个关键量**只在这里定义一次** ✓

    Args:
        pose_R: (3,3) 抓取基座旋转 (world).
        pose_pos: (3,) 抓取基座位置 (world).
        obj_center: (3,) 物心 (world).
        hulls_w: list[(N,3)] 物体在**世界系**的凸包顶点 (CoACD 的 K 个凸包).
        half_sep: 手指接触面半间距 (m) = FINGER_EFFECTIVE_HALF_SPACING + gripper_q.
            ★ 2026-09-18 语义钉死: 调用方**必须**传 **接近段(全开)半间距** ✓
              (即 FINGER_EFFECTIVE_HALF_SPACING + stage_q_open ✓, 不是闭合开度 q_grasp ✗).
              理由: 本函数的 c_clear / kappa 回答的是"**接近段会不会顶到物体**" ✓ ——
              用闭合开度 (≈0.005, 半间距仅 19.9mm) 会让 c_clear 对任何比指缝大的物体
              **恒 > 0** ⇒ 所有候选都被判"必然接触" ⇒ 判据**零区分度** ✗
              (实测 output/s1horiz: 7533 个候选 c_clear ∈ [+3.40, +8.03]mm, 无一为负 ✗).
              用接近段全开 (≈0.05, 半间距 65.2mm) ⇒ c_clear 只在"真张不开"时 >0 ✓
              ⇒ 大多数位姿 -37.7mm ⇒ 判据恢复区分度 ✓
        finger_origin_offset: base → 手指原点沿 forward 的距离 (m), 默认 36.89mm.

    Returns:
        dict: c_center_mm / c_clear_mm / c_axial_mm / kappa / n_hulls / n_verts
        语义: c_clear < 0 = 有安全余量 ✓ ; c_center/c_axial 越小越好 ✓
    """
    R = np.asarray(pose_R, dtype=np.float64)
    base = np.asarray(pose_pos, dtype=np.float64)
    close_dir = R @ np.array([0.0, 1.0, 0.0])      # 手指闭合方向 (world) ✓
    forward_dir = R @ np.array([1.0, 0.0, 0.0])    # base → 指尖方向 (world) ✓
    finger_origin = base + float(finger_origin_offset) * forward_dir
    ctr = np.asarray(obj_center, dtype=np.float64)

    rel = np.concatenate([np.asarray(h, dtype=np.float64) - ctr for h in hulls_w], axis=0)
    s = rel @ close_dir                            # 沿闭合轴 (物心=0)
    f = rel @ forward_dir                          # 沿接近方向
    lo, hi = float(s.min()), float(s.max())
    fo = finger_origin - ctr
    c_center = abs(float(fo @ close_dir)) * 1000.0                                  # mm
    c_clear = (max(abs(lo), abs(hi)) - float(half_sep)) * 1000.0                    # mm
    c_axial = (float(fo @ forward_dir) - 0.5 * (float(f.min()) + float(f.max()))) * 1000.0

    m_clear = max(0.0, -c_clear)
    k_clear = min(1.0, m_clear / float(full_clear_mm))
    k_center = 1.0 / (1.0 + c_center / max(1e-6, float(gate_center_mm)))
    k_axial = 1.0 / (1.0 + abs(c_axial) / max(1e-6, float(full_axial_mm)))
    kappa = float((max(k_clear, 1e-3) * k_center * k_axial) ** (1.0 / 3.0))
    return {
        "c_center_mm": round(c_center, 4),
        "c_clear_mm": round(c_clear, 4),
        "c_axial_mm": round(c_axial, 4),
        "kappa": round(kappa, 6),
        "n_hulls": int(len(hulls_w)),
        "n_verts": int(len(rel)),
    }

# Digest fields that must match byte-for-byte between Stage1 and Stage2.  The
# collision-scope block is intentionally absent: non-target isolation is the one
# whitelisted difference between the two stages (plan §1.2 / §3.4).
SCENE_DIGEST_FIELDS = (
    "target_object",
    "obj_scale_env",
    "obj_scale_target_env",
    "obj_shape_env",
    "obj_shape_target_env",
    "no_gripper_ground_env",
    "bbox_size",
    "bbox_min",
    "bbox_max",
    "body_type",
    "initial_pose_pos",
    "initial_pose_quat",
    "mass",
    "ground_z",
    "gravity",
    "physics",
)


# 2026-09-13 (用户定策): 夹爪开合上限与"按物体尺寸定开度".
# 几何模型: 统一用 physics_utils 的**唯一真源** finger_inner_y/finger_gap
#   (真实指形: 虎口内侧 = 0.0152 + q; 指缝 = 0.0304 + 2q)
#   历史: 薄板时代写死 0.005 → 换真实指形后此处未同步 → 模型不一致
#   ⇒ 指缝间隙 = 0.010 + 2q  (m)
# 用户反馈"夹爪开合还是很大": 旧默认 q_open=0.05 ⇒ 指缝 110mm, 而物体仅 50mm
#   (2.2 倍开度). 改为**按物体宽度定开度**, 默认 1.35 倍 (留 35% 余量).
# 硬上限 0.070: URDF finger joint upper=0.075 (stage1_recipe 注释), 留 5mm 余量.
Q_OPEN_OBJECT_MULTIPLE = float(os.environ.get("_Q_OPEN_MULTIPLE", "1.20"))
Q_OPEN_LIMIT = float(os.environ.get("_Q_OPEN_LIMIT", "0.070"))

# ★ 2026-09-14: stage_q_open() 的记忆单元 —— 第一次带物体尺寸的调用在此锁定全局开度,
#   之后所有无参调用复用同一个值 (保证 Stage1 recipe 与 Stage2 入口 q 绝对一致,
#   否则 stage1_reuse_report 会 STAGE1_REUSE_MISMATCH fail-fast)。
_Q_OPEN_MEMO: list = []


def q_open_for_object_width(object_width_m: float | None) -> float | None:
    """按目标物体尺寸算"规范入口开度": 指缝 ≈ MULTIPLE × 尺寸, 并钳到上限.

    ``指缝 = 0.010 + 2q`` ⇒ ``q = (MULTIPLE·W − 0.010) / 2``

    ★★★ 2026-09-13 踩坑记录 (务必遵守):
      **``object_width_m`` 必须传"物体在夹爪闭合轴(y)上的实际投影跨度", 或其保守上界。**
      绝不能传 GLB 局部系的 bbox 宽度 —— 物体带姿态, 局部 bbox 与投影可差很多。
      实测 (scale=0.8): 局部 bbox y=89.4mm, 但日志实测 ``物体|y|max=60.1mm``
      (即跨度 120.2mm), 结果虎口内侧 60.4mm 只比物体大 **0.3mm** → 手指撞穿
      11.4mm、跟踪被顶开 24.4mm、``PRE_F50_READY`` FAIL。
      **保守上界取 bbox 对角线** ``√(x²+y²+z²)``: 它对任意朝向都不小于真实投影跨度。
    返回 None 表示无法计算 (尺寸无效), 由调用方回退到默认值.
    """
    try:
        w = float(object_width_m)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(w) or w <= 0.0:
        return None
    # ★ 2026-09-14 修正: 基准从 0.010 (**薄板时代**的指缝) 改为 FINGER_GAP0 (真实指形)
    #   指缝(q) = FINGER_GAP0 + 2q  ⇒  要张开到 1.20·w 需要 q = (1.20·w − FINGER_GAP0)/2
    #   旧值 0.010 使算出的开度偏大 10.2mm/个手指 (关节值 +0.0102) ✗
    #   实测对比 (物体 55.9mm): 旧 0.0285 (指缝 87.5mm) → 新 0.0183 (指缝 67.0mm) ✓
    from physics_utils import FINGER_GAP0 as _FGAP0
    q = (Q_OPEN_OBJECT_MULTIPLE * w - _FGAP0) / 2.0
    return float(np.clip(q, 0.0, Q_OPEN_LIMIT))


def stage_q_open(object_width_m: float | None = None) -> float:
    """Shared Stage1/Stage2 full-open finger command ("F50 入口开度").

    Default is the historical ``GRIPPER_MAX_OPEN`` (0.05).  ``_Q_OPEN`` overrides
    it for experiments, but the value must stay shared: ``stage1_reuse_report``
    compares the Stage1 recipe's ``q_open`` against the Stage2 entry q and fails
    fast with ``STAGE1_REUSE_MISMATCH`` otherwise.

    Note: the URDF finger joint limit is ``upper=0.075``; commands above that are
    physically clamped by the joint limit.

    优先级: ``_Q_OPEN`` 环境变量 > 按 ``object_width_m`` 自动计算 > 历史默认.
    任何来源的值都会被钳到 ``Q_OPEN_LIMIT`` (URDF 上限 0.075 留余量).

    ★★ 2026-09-14 记忆化 (用户: "开合为什么还那么大") ★★
      历史缺陷: 全仓 5 个调用点都写 ``stage_q_open()`` **不带参数** ✗ →
               "按物体尺寸自动计算"这条优先级从未被触发,
               永远回落到 GRIPPER_MAX_OPEN=0.05 → 指缝 130.4mm ✗ (张到 13cm).
      现在: **第一次带 ``object_width_m`` 的调用锁定全局值**, 之后所有无参调用
            返回同一个值 —— 这同时保证 Stage1 与 Stage2 的 q_open 绝对一致,
            否则 ``stage1_reuse_report`` 会以 STAGE1_REUSE_MISMATCH fail-fast.
      (``_Q_OPEN`` 环境变量优先级最高且不走记忆, 保持实验旋钮语义不变.)
    """
    env = os.environ.get("_Q_OPEN")
    if env:
        q = float(env)
        if q > Q_OPEN_LIMIT:
            logger.warning(
                "[q_open] _Q_OPEN=%.4f 超过上限 %.4f, 已钳位 (URDF finger upper=0.075)",
                q, Q_OPEN_LIMIT)
        return float(min(q, Q_OPEN_LIMIT))
    if object_width_m is not None:
        q_auto = q_open_for_object_width(object_width_m)
        if q_auto is not None:
            if not _Q_OPEN_MEMO:            # 首次带尺寸 → 锁定
                from physics_utils import FINGER_GAP0
                _Q_OPEN_MEMO.append(float(q_auto))
                logger.info(
                    "[q_open] 按物体尺寸锁定开度: 宽度 %.1fmm → q_open=%.4f "
                    "(指缝 %.1fmm) —— 后续所有 stage_q_open() 无参调用复用此值",
                    float(object_width_m) * 1000.0, float(q_auto),
                    (FINGER_GAP0 + 2.0 * float(q_auto)) * 1000.0)
            return _Q_OPEN_MEMO[0]
    if _Q_OPEN_MEMO:                        # 已被尺寸锁定 → 全局一致
        return _Q_OPEN_MEMO[0]
    from physics_utils import GRIPPER_MAX_OPEN

    return float(min(float(GRIPPER_MAX_OPEN), Q_OPEN_LIMIT))


def cubic_smoothstep(value):
    """``s(t) = t²(3-2t)`` clamped to ``[0, 1]`` (Stage1 close profile)."""
    t = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def stage1_close_q_sequence(q_open, q_grasp, close_controls=STAGE1_CLOSE_CONTROLS):
    """The exact 20-point Stage1 close command sequence.

    ``linspace(0, 1, N)`` fed through the cubic smoothstep, starting at
    ``q_open`` and ending exactly at ``q_grasp`` (mirrors Stage1's
    ``_close1_smooth`` loop one-to-one).
    """
    close_controls = int(close_controls)
    if close_controls < 1:
        raise ValueError("close_controls must be >= 1")
    t = np.linspace(0.0, 1.0, close_controls)
    return float(q_open) - (float(q_open) - float(q_grasp)) * cubic_smoothstep(t)


def _round_list(values, digits=9):
    return [round(float(v), digits) for v in np.asarray(values, dtype=np.float64).ravel()]


def _actor_mass(actor):
    for component in actor.get_components():
        mass = getattr(component, "mass", None)
        if mass is not None:
            try:
                return round(float(mass), 9)
            except (TypeError, ValueError):
                continue
    return None


def find_actor(actors, target_name):
    """Locate the target actor by exact name, then by substring."""
    for actor in actors:
        if actor.get_name() == target_name:
            return actor
    for actor in actors:
        if target_name and target_name in actor.get_name():
            return actor
    return None


def build_physics_config():
    """Shared Stage1/Stage2 physical parameters (single source of truth)."""
    from physics_env import (
        CONTROL_FREQ,
        DECIMATION,
        GRIPPER_DAMPING,
        GRIPPER_FORCE,
        GRIPPER_FRICTION,
        GRIPPER_STIFFNESS,
        PHYSICS_TIMESTEP,
    )

    return {
        "control_frequency_hz": float(CONTROL_FREQ),
        "physics_timestep_s": float(PHYSICS_TIMESTEP),
        "decimation": int(DECIMATION),
        "gripper_stiffness": float(GRIPPER_STIFFNESS),
        "gripper_damping": float(GRIPPER_DAMPING),
        "gripper_force_limit": float(GRIPPER_FORCE),
        "gripper_friction": float(GRIPPER_FRICTION),
    }


def build_scene_digest(
    *,
    target_name,
    obj_info,
    obj_initial_poses,
    obj_actors,
    ground_z,
    gravity,
    physics_config,
):
    """Digest of everything that must be identical between Stage1 and Stage2.

    ``bbox_size`` already encodes ``_OBJ_SCALE``; the raw env knobs are recorded
    too so a mismatch report can name the cause.  ``initial_pose_pos`` is the
    object's reset position — the same value both stages restore before a
    rollout, which is what makes the two rollouts comparable.
    """
    entry = dict(obj_info.get(target_name, {}))
    initial_pose = obj_initial_poses.get(target_name) if obj_initial_poses else None
    pose_pos = [float(v) for v in np.asarray(initial_pose.p, dtype=np.float64)] if initial_pose is not None else None
    pose_quat = [float(v) for v in np.asarray(initial_pose.q, dtype=np.float64)] if initial_pose is not None else None
    actor = find_actor(obj_actors, target_name)

    return {
        "target_object": str(target_name),
        "obj_scale_env": str(os.environ.get("_OBJ_SCALE", "1.0")),
        "obj_scale_target_env": str(os.environ.get("_OBJ_SCALE_TARGET", "")),
        "obj_shape_env": str(os.environ.get("_OBJ_SHAPE", "box")),
        "obj_shape_target_env": str(os.environ.get("_OBJ_SHAPE_TARGET", "")),
        "no_gripper_ground_env": str(os.environ.get("_NO_GRIPPER_GROUND", "0")),
        "bbox_size": _round_list(entry.get("bbox_size", [])) if entry.get("bbox_size") else [],
        "bbox_min": _round_list(entry.get("bbox_min", [])) if entry.get("bbox_min") else [],
        "bbox_max": _round_list(entry.get("bbox_max", [])) if entry.get("bbox_max") else [],
        "body_type": str(entry.get("body_type", "")) if entry.get("body_type") else "",
        "initial_pose_pos": pose_pos,
        "initial_pose_quat": pose_quat,
        "mass": _actor_mass(actor) if actor is not None else None,
        "ground_z": round(float(ground_z), 9),
        "gravity": _round_list(gravity),
        "physics": dict(physics_config),
    }


def scene_digest_hash(digest) -> str:
    return hashlib.sha256(
        json.dumps(digest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_stage1_grasp_recipe(
    *,
    target_name,
    q_open,
    q_grasp,
    physics_config,
    scene_digest,
    close_controls=STAGE1_CLOSE_CONTROLS,
    settle_controls=STAGE1_SETTLE_CONTROLS,
    revision=STAGE1_RECIPE_REVISION,
    q_grasp_nominal=None,
    close_q_actual=None,
):
    """The immutable Stage1 grasp recipe consumed verbatim by Stage2.

    ★ 2026-09-23 【方案 §8 实施】必须保存**实际执行过**的闭合序列 ✓ (不再重生成 ✗)
      方案原话 ✓: \"Stage1 生成器**不能保存'名义 q 曲线'却实际执行裁剪曲线**。
        重放不可隐式重新选 q_open、重新计算过闭合 ✗\" ⇒ 现加入 `close_q_actual` ✓:
        有实际数组 ⇒ **逐值照抄** ✓ (`q_sequence_source=\"actual_executed\"` ✓);
        没有 ⇒ 仍按旧路重生成 ✓ 但**显式标注** `regenerated_nominal` ⚠ (可追溯 ✓)。
    """
    _src = "regenerated_nominal"
    close_q = None
    if close_q_actual is not None:
        try:
            _vals = [float(x) for x in list(close_q_actual)]
            if _vals:
                close_q = _vals
                _src = "actual_executed"
        except Exception:  # noqa: BLE001
            close_q = None
    if close_q is None:
        close_q = stage1_close_q_sequence(q_open, q_grasp, close_controls)
    return {
        "revision": str(revision),
        "q_sequence_source": _src,      # ★ 方案 §8 ✓: 可追溯\"名义 vs 实际\" ✓
        "target_object": str(target_name),
        "q_open": float(q_open),
        # ★ 2026-09-21: `q_grasp` = **Stage1 真值验证过的闭合终点** (含过闭合 ✓)
        #   `q_grasp_nominal` = 不含过闭合的名义值 (仅追溯用 ✓, 下游**不要**用它 ✗)
        "q_grasp": float(q_grasp),
        "q_grasp_nominal": (float(q_grasp_nominal)
                            if q_grasp_nominal is not None else float(q_grasp)),
        "close_controls": int(close_controls),
        "settle_controls": int(settle_controls),
        "close_profile": STAGE1_CLOSE_PROFILE,
        "close_ctrl_t": _round_list(np.linspace(0.0, 1.0, int(close_controls))),
        "close_ctrl_q": _round_list(close_q),
        "physics": dict(physics_config),
        "scene_digest": dict(scene_digest),
        "scene_digest_hash": scene_digest_hash(scene_digest),
    }


def recipe_mismatches(recipe, current_digest) -> list[str]:
    """Field-by-field diff between a Stage1 recipe and the live Stage2 scene."""
    stored = dict((recipe or {}).get("scene_digest", {}) or {})
    mismatches = []
    for field in SCENE_DIGEST_FIELDS:
        if stored.get(field) != current_digest.get(field):
            mismatches.append(field)
    return mismatches


def recipe_summary(recipe) -> dict:
    """Compact recipe view for logs / manifests."""
    if not recipe:
        return {}
    return {
        "revision": recipe.get("revision"),
        "target_object": recipe.get("target_object"),
        "q_open": recipe.get("q_open"),
        "q_grasp": recipe.get("q_grasp"),
        "close_controls": recipe.get("close_controls"),
        "settle_controls": recipe.get("settle_controls"),
        "close_profile": recipe.get("close_profile"),
        "scene_digest_hash": recipe.get("scene_digest_hash"),
    }


def load_stage1_grasp(path):
    """Load a ``best_grasp.npz`` artifact and attach its parsed recipe.

    Returns ``(grasp, recipe)`` where ``recipe`` is ``None`` for stale artifacts
    produced before the recipe contract existed.  Callers must fail fast with
    ``STALE_STAGE1_ARTIFACT`` instead of inventing defaults.
    """
    data = np.load(str(path), allow_pickle=True)
    grasp = {key: data[key] for key in data.keys()}
    for key in ("pos", "R", "euler", "ref_base_R"):
        if key in grasp:
            grasp[key] = np.asarray(grasp[key], dtype=np.float64)
    if "gripper_qpos" in grasp:
        grasp["gripper_qpos"] = float(grasp["gripper_qpos"])
    if "force_limit" in grasp:
        grasp["force_limit"] = float(grasp["force_limit"])
    recipe = None
    if "stage1_recipe_json" in grasp:
        raw = grasp["stage1_recipe_json"]
        try:
            recipe = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError):
            recipe = None
    if recipe is not None:
        grasp["stage1_recipe"] = recipe
    # ★★★★ 2026-09-21 【P0-5 修复】必须校验"**验证结果**" ✗ ★★★★
    #   静态审查结论 ✓ (用户 2026-09-21): "**Stage1 失败产物仍可进入 Stage2** ✗ —— 无合格候选
    #     仍回捡 Top-K, 终选失败仍『带警告交付』✗; **加载器不检查验证成功次数** ✗"
    #   修法 ✓: 解析产物里的 `verification` (Stage1 终选门结果 ✓) 并暴露 `verified` /
    #     `verified_n_pass` ✓ ⇒ 下游用 `assert_stage1_verified()` **拒绝执行** ✗
    _ver = grasp.get("verification")
    try:
        if isinstance(_ver, np.ndarray) and _ver.shape == ():
            _ver = _ver.item()
    except Exception:  # noqa: BLE001
        _ver = None
    if not isinstance(_ver, dict):
        _ver = None
    grasp["verification"] = _ver
    _np_ = int((_ver or {}).get("n_pass", -1)) if _ver is not None else -1
    grasp["verified"] = bool(_np_ >= 2)          # 门槛: 3 次重复中 ≥2 次 (与验证门一致 ✓)
    grasp["verified_n_pass"] = _np_
    logger.info(
        "[Stage1 产物校验] %s: verified=%s (n_pass=%s, 门槛 2/3 ✓) %s",
        os.path.basename(str(path)), grasp["verified"], _np_,
        "" if grasp["verified"] else "✗ ⇒ **不得进入 Stage2** ✗")
    # ★ 2026-09-21 【P0-5】默认**拒绝**未通过验证的产物 ✓ (一处覆盖全部入口 ✓)
    #   审查建议 ✓: "Stage1 失败只保存诊断; **Stage2 校验成功状态**" ✓
    #   逃生开关 ✓: `_ALLOW_UNVERIFIED_STAGE1=1` (仅用于事后诊断/复现 ✓)
    if not grasp["verified"] and os.environ.get("_ALLOW_UNVERIFIED_STAGE1", "0") != "1":
        raise ValueError(
            f"STAGE1_NOT_VERIFIED: {os.path.basename(str(path))} 的验证未通过 "
            f"(n_pass={_np_} < 2) ✗ ⇒ **失败产物不得进入 Stage2** ✗ "
            f"(如确需诊断, 设 `_ALLOW_UNVERIFIED_STAGE1=1` ✓)")
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ T072 / T08-A (2026-09-27) 【Stage2 公共入口合同】**一处覆盖全部入口** ✓ ★★★★
    #   md T07.2 原话 ✓: "判据标签 v2 但**无对应评估记录**、profile 不符、n_pass 不足,
    #     **所有正常 Stage2 入口均拒绝**; 诊断 override **不能产生正式 PASS**" ✓
    #   md T07.1 ✓: 旧 artifact (无完整实际记录) **不能自动迁移**成 strict_replay 证据 ✗
    #   ⚠ 只读校验 ✓: 不改写 artifact; override 时**只放行、不标 PASS** ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    try:
        _ok_e, _errs_e = validate_stage2_entry(
            recipe, mode=os.environ.get("_S2_MODE", "feedback"),
            allow_unverified=(os.environ.get("_ALLOW_UNVERIFIED_STAGE1", "0") == "1"))
    except Exception as _ee:  # noqa: BLE001
        _ok_e, _errs_e = False, ["stage2_entry 校验异常: %s" % _ee]
    grasp["stage2_entry_ok"] = bool(_ok_e)
    grasp["stage2_entry_errors"] = list(_errs_e)
    if _errs_e:
        logger.warning("[Stage2 入口合同] errors=%s", list(_errs_e)[:4])
    if not _ok_e:
        raise ValueError(
            "ARTIFACT_INVALID: %s 不满足 Stage2 入口合同 ⇒ 拒绝 ✗ | %s"
            % (os.path.basename(str(path)), list(_errs_e)[:3]))
    return grasp, recipe


def normalize_stage1_verification(grasp) -> dict:
    """★ 2026-09-22 【P0-5 内存入口】把**任意来源**的 Stage1 结果规范化 ✓

    审计文档 §P0-5 ✓: "以**规范化 verification 记录**为依据, 在 Stage2 公共入口统一校验,
      **无论结果来自内存还是磁盘** ✗" 且 "helper 声明了 `min_pass`, 实际判断只看
      `verified` 布尔值, **未按传入门槛核对次数**" ✗
    本函数 ✓: 从 `verification` (可为 dict / 0-d ndarray ✓) 解析 `n_pass` ✓, 写回
      `verified` / `verified_n_pass` ✓ —— **内存对象与磁盘对象走同一条规范化路径** ✓
    """
    _ver = grasp.get("verification")
    try:
        if isinstance(_ver, np.ndarray) and _ver.shape == ():
            _ver = _ver.item()
    except Exception:  # noqa: BLE001
        _ver = None
    if not isinstance(_ver, dict):
        _ver = None
    grasp["verification"] = _ver
    _np_ = int((_ver or {}).get("n_pass", -1)) if _ver is not None else -1
    grasp["verified"] = bool(_np_ >= 2)          # 门槛: 3 次重复中 ≥2 次 ✓
    grasp["verified_n_pass"] = _np_
    return grasp


def assert_stage1_verified(grasp, *, min_pass: int = 2, require_recipe: bool = True,
                           requirement: str = "STAGE1_NOT_VERIFIED") -> None:
    """★ 2026-09-21 【P0-5】失败/未验证的 Stage1 产物 ⇒ **拒绝执行** ✗

    审查建议 ✓: "**Stage1 失败只保存诊断**; **Stage2 校验成功状态、验证结果及几何指纹**" ✓
    几何指纹部分 (不同 CoACD 几何不得复用 ✓) 由调用方另行比对 `scene_digest` ⚠ (待接 ✓)。

    ★ 2026-09-22 修正 ✓ (审计文档 §P0-5 ✓):
      · 先调 `normalize_stage1_verification()` ✓ ⇒ **内存对象也能校验** ✓ (原来只覆盖磁盘 ✓)
      · **按传入门槛核对次数** ✓ (`n_pass >= min_pass` ✓), 不再只看 `verified` 布尔 ✗
      · 逃生开关 ✓ `_ALLOW_UNVERIFIED_STAGE1=1` (仅诊断 ✓, 与磁盘 loader 同口径 ✓)
    """
    grasp = normalize_stage1_verification(grasp)
    if require_recipe and not grasp.get("stage1_recipe"):
        raise ValueError(f"{requirement}: stale artifact without stage1_recipe ✗")
    _np_ = int(grasp.get("verified_n_pass", -1))
    if _np_ < int(min_pass) and os.environ.get("_ALLOW_UNVERIFIED_STAGE1", "0") != "1":
        raise ValueError(
            f"{requirement}: Stage1 验证未通过 (n_pass={_np_} < {min_pass}) ✗ ⇒ "
            f"失败产物不得进入 Stage2 ✗ (诊断用: `_ALLOW_UNVERIFIED_STAGE1=1` ✓)")


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ T07b (2026-09-27) 【获胜记录强绑定】**纯函数** ⇒ 可单测、不跑物理 ✓ ★★★★
#   md T07.1 原话 ✓: "不再凭 q 终点、最后一次缓存或'最长轨迹'找 record;
#     rollout 返回 episode_id + candidate_id + immutable_record + assessment;
#     搜索/重复验证/winner 选择/导出**始终传递这个对象/唯一引用**" ✓
#   ⇒ 本函数**只认 winner 自带的 immutable_record** ✓:
#       有 ⇒ 返回它; 没有 ⇒ **WINNER_RECORD_MISSING**(不猜、不回退、不替代 ✓)
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def bind_winner_record(best_output):
    """Return ``(record, error)``；``error`` ∈ {None, "WINNER_RECORD_MISSING"}.

    只允许**该 run 自带**的 ``immutable_record``；
    禁止按 q 终点匹配、禁止取"最长轨迹"、禁止取最后一次缓存 ✗ (md T07.1 ✓)
    """
    if not isinstance(best_output, dict):
        return None, "WINNER_RECORD_MISSING"
    _rec = best_output.get("immutable_record")
    if not isinstance(_rec, dict):
        return None, "WINNER_RECORD_MISSING"
    _q = _rec.get("q_cmd")
    if not isinstance(_q, (list, tuple)) or len(_q) == 0:
        return None, "WINNER_RECORD_MISSING"
    return _rec, None


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ T07d (2026-09-27) 【数据合同工具】**纯函数** ⇒ 可单测、不跑物理 ✓ ★★★★
#   md T07.1 ✓: 同长 N>0 / 半开区间 [start,end) 不重叠不遗漏 / 有限值 / 合法旋转 /
#     时标递增 / source_frame=**整数帧号** / task_grasp_frame=默认 **50** (不随闭合帧数变 ✓)
#   ⇒ 导出前必须跑本合同校验; 失败 ⇒ export_valid=false 且**不覆盖**既有成功记录 ✓
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
STAGE1_RECIPE_SCHEMA = "stage1_recipe/v2"

_PHASE_ORDER = ("travel", "close", "settle", "lift")


def derive_phase_boundaries(phase_list):
    """从**本 run 的 phase 数组**推导半开区间 ``[start,end)`` ✓ (不重叠、不遗漏 ✓).

    返回 ``{(phase): [start, end)}`` + ``"n"`` 总长；空数组 ⇒ ``{"n": 0}`` ✓
    """
    _ph = [str(x) for x in (phase_list or [])]
    out = {"n": int(len(_ph))}
    if not _ph:
        return out
    _bounds = {}
    _i = 0
    while _i < len(_ph):
        _cur = _ph[_i]
        _j = _i
        while _j + 1 < len(_ph) and _ph[_j + 1] == _cur:
            _j += 1
        _bounds.setdefault(_cur, []).append([int(_i), int(_j + 1)])
        _i = _j + 1
    for _k, _v in _bounds.items():
        out[_k] = _v
    return out


def validate_recipe_contract(recipe):
    """返回 ``(ok, errors)``；errors 是字符串列表 (可为空) ✓.

    只做**合同级**检查(不判成功率 ✓): 同长 N>0、有限值、合法旋转、时标递增、
    相位边界覆盖、source_frame/task_grasp_frame 为整数 ✓
    """
    import math as _m

    errs = []
    if not isinstance(recipe, dict):
        return False, ["recipe 不是 dict"]
    if str(recipe.get("schema_version", "")) != STAGE1_RECIPE_SCHEMA:
        errs.append("schema_version 非 %s: %r" % (STAGE1_RECIPE_SCHEMA,
                                                 recipe.get("schema_version")))
    # ① 同长 N>0
    _q = list(recipe.get("actual_q_commands") or [])
    _t = list(recipe.get("actual_control_times") or [])
    _T = list(recipe.get("T_W_G_cmd") or [])
    _ph = list(recipe.get("phase") or [])
    _ci = list(recipe.get("control_index") or [])
    _N = len(_q)
    if _N <= 0:
        errs.append("actual_q_commands 为空 (N=%d)" % _N)
    for _name, _arr in (("actual_control_times", _t), ("T_W_G_cmd", _T),
                        ("phase", _ph), ("control_index", _ci)):
        if len(_arr) != _N:
            errs.append("长度不一致: %s=%d vs actual_q_commands=%d" % (_name, len(_arr), _N))
    # ② 有限值 + 时标严格递增 + control_index 连续递增
    if _q and not all(_m.isfinite(float(x)) for x in _q):
        errs.append("actual_q_commands 含 NaN/Inf")
    if _t:
        if not all(_m.isfinite(float(x)) for x in _t):
            errs.append("actual_control_times 含 NaN/Inf")
        elif any(float(_t[_i + 1]) <= float(_t[_i]) for _i in range(len(_t) - 1)):
            errs.append("时标非严格递增 (重复或倒退 ✗)")
    if _ci and any(int(_ci[_i + 1]) != int(_ci[_i]) + 1 for _i in range(len(_ci) - 1)):
        errs.append("control_index 非连续递增 (错帧/跳号 ✗)")
    # ③ 合法旋转 (T_W_G_cmd: N×4×4, 旋转块正交且 det≈1)
    for _i, _M in enumerate(_T):
        _A = _M
        try:
            if len(_A) != 4 or any(len(_r) != 4 for _r in _A):
                errs.append("T_W_G_cmd[%d] 不是 4×4" % _i)
                break
            _R3 = [[float(_A[_r][_c]) for _c in range(3)] for _r in range(3)]
            _det = (_R3[0][0] * (_R3[1][1] * _R3[2][2] - _R3[1][2] * _R3[2][1])
                    - _R3[0][1] * (_R3[1][0] * _R3[2][2] - _R3[1][2] * _R3[2][0])
                    + _R3[0][2] * (_R3[1][0] * _R3[2][1] - _R3[1][1] * _R3[2][0]))
            if abs(_det - 1.0) > 1e-3:
                errs.append("T_W_G_cmd[%d] 旋转块 det=%.6f (应≈1 ✗)" % (_i, _det))
                break
        except Exception as _e:  # noqa: BLE001
            errs.append("T_W_G_cmd[%d] 解析失败: %s" % (_i, _e))
            break
    # ④ 相位边界: 必须**覆盖 [0,N) 不重叠不遗漏** ✓
    _pb = recipe.get("phase_boundaries") or {}
    if _ph:
        _segs = []
        for _k, _v in _pb.items():
            if _k == "n":
                continue
            for _seg in (_v or []):
                _segs.append((int(_seg[0]), int(_seg[1])))
        _segs.sort()
        if not _segs:
            errs.append("phase_boundaries 为空但 phase 数组非空 ✗")
        else:
            if _segs[0][0] != 0 or _segs[-1][1] != _N:
                errs.append("phase_boundaries 未覆盖 [0,%d): 首个=%s 末个=%s ✗"
                            % (_N, _segs[0], _segs[-1]))
            for _i in range(len(_segs) - 1):
                if _segs[_i][1] != _segs[_i + 1][0]:
                    errs.append("phase_boundaries 在 %d 处重叠/断裂 ✗" % _segs[_i][1])
                    break
    # ⑤ 帧号必须是**整数**且语义分开 ✓
    _sf = recipe.get("source_frame")
    if not isinstance(_sf, int):
        errs.append("source_frame 必须是**整数帧号**(实测 %r ✗ 不是矩阵/控制索引 ✓)" % (_sf,))
    _tf = recipe.get("task_grasp_frame")
    if not isinstance(_tf, int):
        errs.append("task_grasp_frame 必须是整数帧号 (实测 %r ✗)" % (_tf,))
    elif _tf != int(recipe.get("task_grasp_frame_default", 50)):
        errs.append("task_grasp_frame=%d 与默认 50 不符 (md: 不随闭合帧数变化 ✗)" % _tf)
    # ⑥ 三类 hash 指向实际 profile ✓
    for _hk in ("physics_hash", "controller_hash", "criteria_hash"):
        if not str(recipe.get(_hk) or ""):
            errs.append("%s 缺失 (必须指向实际 resolved profile ✗)" % _hk)
    return (len(errs) == 0), errs


def compare_trace_with_spy(trace_q, spy_q, atol=1e-9):
    """★ T072 ✓ 返回 ``(ok, detail)``: trace 必须**逐点等于下发 spy**(clip 之后) ✓.

    md T07.2 ✓: "强制触发出口限速, trace 必须等于**下发 spy 记录**, 而不是 clip 前值" ✗
    """
    try:
        _t = [float(x) for x in (trace_q or [])]
        _s = [float(x) for x in (spy_q or [])]
    except Exception as _e:  # noqa: BLE001
        return False, "解析失败: %s" % _e
    if not _s:
        return False, "spy 为空 (未启用 spy 或未下发 ✗)"
    if len(_t) != len(_s):
        return False, "长度不一致: trace=%d spy=%d (错帧/漏记 ✗)" % (len(_t), len(_s))
    for _i, (_a, _b) in enumerate(zip(_t, _s)):
        if abs(_a - _b) > max(float(atol), 0.0):
            return False, "第 %d 点不一致: trace=%.9f spy=%.9f (疑记了 clip 前值 ✗)" % (
                _i, _a, _b)
    return True, "trace == spy (%d 点) ✓" % len(_s)


def validate_stage2_entry(recipe, mode="feedback", min_n_pass=2,
                          allow_unverified=False):
    """★ T072/T08-A ✓ Stage2 **公共入口合同** ⇒ 返回 ``(ok, errors)`` ✓.

    md T07.2 ✓: "判据标签 v2 但**无对应评估记录**、profile 不符、n_pass 不足,
      **所有正常 Stage2 入口均拒绝**; 诊断 override **不能产生正式 PASS**" ✓
    md T07.1 ✓: "无完整实际记录的旧 artifact **不能自动迁移**成 strict_replay 成功证据" ✓
    """
    errs = []
    if not isinstance(recipe, dict) or not recipe:
        return False, ["recipe 缺失/非法"]
    _mode = str(mode or "feedback")
    _schema = str(recipe.get("schema_version", ""))
    _is_v2 = (_schema == STAGE1_RECIPE_SCHEMA)
    # ① v2 合同: 必须 export_valid + 有绑定记录
    if _is_v2:
        if recipe.get("export_valid") is not True:
            errs.append("v2 但 export_valid!=True (合同未过 ✗): %s"
                        % (recipe.get("contract_errors") or recipe.get("export_valid")))
        if not (recipe.get("winner_binding") or {}).get("record_present", False):
            errs.append("v2 但缺少 winner_binding.record_present (无实际执行记录 ✗)")
        for _k in ("time_s", "T_W_G_cmd", "q_cmd", "phase", "control_index"):
            if not (recipe.get(_k) or []):
                errs.append("v2 但 %s 为空 (无完整记录 ✗)" % _k)
    else:
        errs.append("schema_version=%r 非 %s ⇒ **旧 artifact 不得当作严格重放证据** ✗"
                    % (_schema, STAGE1_RECIPE_SCHEMA))
        if _mode == "strict_replay":
            errs.append("strict_replay 要求 v2 完整合同 ✗ (旧 artifact 不可自动迁移 ✓)")
    # ② 评估记录 / n_pass (与磁盘 loader 同口径 ✓)
    _vr = recipe.get("verification_results")
    if not isinstance(_vr, dict):
        errs.append("verification_results 缺失 (md: v2 标签但**无对应评估记录** ⇒ 拒绝 ✗)")
    else:
        _runs = _vr.get("runs")
        _npass = _vr.get("n_pass", None)
        if _runs is None and _npass is None:
            errs.append("verification_results 无 runs/n_pass (无评估记录 ⇒ 拒绝 ✗)")
        elif _npass is not None and int(_npass) < int(min_n_pass):
            errs.append("n_pass=%s < %d (不足 ✗)" % (_npass, int(min_n_pass)))
    # ③ profile 身份
    if _is_v2 and not str(recipe.get("physics_hash") or ""):
        errs.append("physics_hash 缺失 (profile 身份不明 ✗)")
    if allow_unverified:
        # 诊断 override: **只放行, 不产生正式 PASS** ✓
        return True, errs
    return (len(errs) == 0), errs


def hash_resolved_profile(obj):
    """对**实际 resolved profile** 取 hash ✓ (不是 env 字符串、不是预期默认值 ✗)"""
    import hashlib as _hl
    import json as _js

    try:
        _s = _js.dumps(obj, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        _s = str(obj)
    return _hl.sha256(_s.encode("utf-8")).hexdigest()[:32]


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ T08-A (2026-09-27) 【Stage2 只读加载 + 模式分流】**纯函数** ⇒ 不跑物理 ✓ ★★★★
#   md 原话 ✓: "先过 T07 schema/identity/profile 门, 再选模式。
#     **feedback 消费 ExecutionSpec; strict_replay 消费完整实际 trace。
#     两者均不得再由 q 终点调用 `stage1_close_q_sequence` 生成另一条'等价'曲线**" ✗
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def split_trace_by_phase(recipe):
    """按 ``phase_boundaries``(半开区间 ✓) 把 ``q_cmd`` 切成各相位命令段 ✓.

    返回 ``{phase: {"q": [...], "idx": [a,b), "t0": .., "t1": ..}}``；
    边界缺失/不覆盖 ⇒ ``{}`` + ``_errors`` ✓ (由调用方拒绝 ✗)
    """
    out = {"_errors": []}
    _pb = recipe.get("phase_boundaries") or {}
    _q = [float(x) for x in (recipe.get("q_cmd") or [])]
    _t = [float(x) for x in (recipe.get("time_s") or [])]
    if not _q:
        out["_errors"].append("q_cmd 为空 (无实际记录 ✗)")
        return out
    if not _pb:
        out["_errors"].append("phase_boundaries 缺失 ✗")
        return out
    for _ph, _segs in _pb.items():
        if _ph == "n":
            continue
        for _s in (_segs or []):
            _a, _b = int(_s[0]), int(_s[1])
            if _a < 0 or _b > len(_q) or _b <= _a:
                out["_errors"].append("相位 %s 区间非法 [%d,%d) N=%d ✗" % (_ph, _a, _b, len(_q)))
                continue
            out[_ph] = {"q": _q[_a:_b], "idx": [_a, _b],
                        "t0": (_t[_a] if _a < len(_t) else None),
                        "t1": (_t[_b - 1] if _b - 1 < len(_t) else None)}
    return out


def build_execution_spec(recipe):
    """从 recipe 构造 **ExecutionSpec** ✓ (feedback 模式的输入 ✓).

    含: 控制器版本/参数、anchor、profile、载荷/预紧目标、状态转换、安全边界 ✓
    """
    try:
        import physics_utils as _pu
        _prof = {k: getattr(_pu, k) for k in (
            "FINGER_PD_KP", "FINGER_PD_KD", "FINGER_PD_FORCE_LIMIT", "GRIPPER_FORCE",
            "GRIPPER_STIFFNESS", "GRIPPER_DAMPING", "GRIPPER_MAX_OPEN", "FINGER_GAP0",
            "PHYSICS_TIMESTEP") if hasattr(_pu, k)}
    except Exception:  # noqa: BLE001
        _prof = {}
    _tr = split_trace_by_phase(recipe)
    _close = (_tr.get("close") or {}).get("q") or []
    _settle = (_tr.get("settle") or {}).get("q") or []
    return {
        "schema_version": STAGE1_RECIPE_SCHEMA,
        "controller_version": str(recipe.get("controller_hash") or ""),
        "controller_profile": _prof,
        "anchor": {"optimized_T_W_G": recipe.get("optimized_T_W_G"),
                   "q_grasp": recipe.get("q_grasp"), "q_open": recipe.get("q_open"),
                   "source_frame": recipe.get("source_frame"),
                   "task_grasp_frame": recipe.get("task_grasp_frame")},
        "profile": {"physics_hash": recipe.get("physics_hash"),
                    "close_profile": recipe.get("close_profile"),
                    "close_controls": recipe.get("close_controls"),
                    "settle_controls": recipe.get("settle_controls")},
        # 载荷/预紧目标: 取**实际执行段**的首末 (不是名义重生成 ✗)
        "load_target": (float(_close[-1]) if _close else None),
        "close_entry_q": (float(_close[0]) if _close else float(recipe.get("q_open", 0.0))),
        "settle_end_q": (float(_settle[-1]) if _settle else None),
        "phase_boundaries": recipe.get("phase_boundaries"),
        "safety_bounds": {"q_min": 0.0,
                          "q_max": float(recipe.get("q_open") or 0.0),
                          "preload_m": recipe.get("optimized_preload")},
    }


def select_close_mode(recipe, mode="feedback"):
    """★ T08-A ✓ 返回 ``(mode, payload, errors)``；**绝不**调用
    ``stage1_close_q_sequence`` 重生成"等价"曲线 ✓ (md 明令禁止 ✗).

    ``strict_replay`` ⇒ payload = **完整实际 trace 的 close 段** ✓
    ``feedback``      ⇒ payload = ``ExecutionSpec`` (+ 入口 q / 目标), 闭合段由 runner 追加 ✓
    """
    errs = []
    _mode = str(mode or "feedback")
    if _mode not in ("feedback", "strict_replay"):
        return _mode, None, ["未知模式 %r (只允许 feedback / strict_replay ✗)" % _mode]
    _ok_e, _errs_e = validate_stage2_entry(recipe, mode=_mode)
    if not _ok_e:
        return _mode, None, ["入口合同未过: %s" % _errs_e[:3]]
    _tr = split_trace_by_phase(recipe)
    if _tr.get("_errors"):
        errs.extend(_tr["_errors"])
    if _mode == "strict_replay":
        _close = (_tr.get("close") or {})
        _settle = (_tr.get("settle") or {})
        if not _close.get("q"):
            errs.append("strict_replay 要求**完整实际 close 段** (实测缺失 ✗)")
        if not _settle.get("q"):
            errs.append("strict_replay 要求**完整实际 settle 段** (实测缺失 ✗)")
        if errs:
            return _mode, None, errs
        return _mode, {"provenance": "actual_executed",
                       "close_q": [float(x) for x in _close["q"]],
                       "settle_q": [float(x) for x in _settle["q"]],
                       "close_idx": list(_close["idx"]),
                       "settle_idx": list(_settle["idx"])}, []
    _spec = build_execution_spec(recipe)
    if _spec.get("close_entry_q") is None:
        errs.append("feedback 需要闭合入口 q (实际记录缺失 ✗)")
    if not _spec.get("load_target"):
        errs.append("feedback 需要载荷目标 (实际 close 末点缺失 ✗)")
    if errs:
        return _mode, None, errs
    return _mode, {"provenance": "execution_spec", "spec": _spec,
                   "adaptive_close": True}, []


def assert_verification_binding(verification):
    """★ 任务04 (2026-09-28) ✓ md §04: "`selected_episode` **必须**在同 candidate/profile/
    controller/criteria 的 **passed 列表**中; **`n_pass` 由列表计算**; 不能从搜索
    `best_result` 补另一个 episode ✗" ✓

    纯函数 ⇒ **零物理可测** ✓ (md §7.2: 先让失败反例成立 ✓)
    实测旧病 ✓ (s31 / s44 / success_rec): `selected_episode='s1ep-000124'`(搜索最优)
      而终验通过的是 **128/129/130** ⇒ 记录与验证**不绑定** 🔴

    Returns ``(ok, errors)``:
      · `n_pass` 与 `runs` 统计**必须一致** (不一致 ⇒ 记录不可信 ✗)
      · `selected_episode` 非空时**必须**在 passed 列表内 ✗
      · runs 每项须含 `episode_id` 与 `passed` (缺 ⇒ 追溯不完整 ✗)
    """
    errs = []
    vr = verification if isinstance(verification, dict) else {}
    runs = list(vr.get("runs") or [])
    if not runs:
        errs.append("verification_results.runs 为空 ⇒ 无**重复 run 明细** ⇒ 不可追溯 ✗")
        return False, errs
    bad_items = []
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            bad_items.append(i)
            continue
        if r.get("episode_id") in (None, ""):
            bad_items.append(i)
        if "passed" not in r:
            bad_items.append(i)
    if bad_items:
        errs.append("runs 第 %s 项缺 episode_id/passed ⇒ 追溯不完整 ✗" % bad_items[:6])
    n_by_runs = int(sum(1 for r in runs if isinstance(r, dict) and r.get("passed")))
    n_pass = vr.get("n_pass")
    if n_pass is None:
        errs.append("n_pass 缺失 (md: 由列表计算 ✓)")
    elif int(n_pass) != n_by_runs:
        errs.append("n_pass=%s ≠ runs 统计=%d ⇒ 记录与验证不绑定 ✗" % (n_pass, n_by_runs))
    sel = vr.get("selected_episode")
    passed_ids = [r.get("episode_id") for r in runs
                  if isinstance(r, dict) and r.get("passed")]
    if sel in (None, ""):
        errs.append("selected_episode 缺失 ✗ (md: 必须来自 passed 列表 ✓)")
    elif sel not in passed_ids:
        errs.append("selected_episode=%s **不在** passed=%s ⇒ 不得作为成功记录 ✗"
                    % (sel, passed_ids))
    return (len(errs) == 0), errs


def reuse_report_items(recipe, observed=None, mode="feedback"):
    """★ T08-A ✓ ``stage1_reuse_report`` 的**逐项 expected/observed/match** ✓.

    md ✓: "复用同时核对**执行器/物理/anchor 身份**, 而不是只比较名义 q 数组" ✓
    """
    _obs = observed or {}
    _items = []

    def _add(name, expected, _newname=None):
        _o = _obs.get(name if _newname is None else _newname)
        _m = (str(_o) == str(expected)) if _o is not None else None
        _items.append({"field": name, "expected": expected, "observed": _o, "match": _m})

    _add("schema_version", str(recipe.get("schema_version", "")))
    _add("physics_hash", str(recipe.get("physics_hash", "")))
    _add("controller_hash", str(recipe.get("controller_hash", "")))
    _add("criteria_hash", str(recipe.get("criteria_hash", "")))
    _add("optimized_T_W_G", recipe.get("optimized_T_W_G"))
    _add("source_frame", recipe.get("source_frame"))
    _add("task_grasp_frame", recipe.get("task_grasp_frame"))
    _add("q_grasp", recipe.get("q_grasp"))
    _tr = split_trace_by_phase(recipe)
    _add("close_controls_actual", len((_tr.get("close") or {}).get("q") or []))
    _add("settle_controls_actual", len((_tr.get("settle") or {}).get("q") or []))
    _add("mode", str(mode))
    _missing = [i for i in _items if i["observed"] is None]
    _bad = [i for i in _items if i["match"] is False]
    return {"items": _items, "n_missing": len(_missing), "n_mismatch": len(_bad),
            "passed": (len(_missing) == 0 and len(_bad) == 0)}


# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★★★★ T08-C (2026-09-27) 【连续接近审计】纯函数 ⇒ 不跑物理即可验收 ✓ ★★★★
#   md 原话 ✓: "明确三个量: 仿真时间 `t` **持续增长**, MANO 源相位 `s` **单调不减**,
#     控制索引 `k` 每周期递增"; "按真实可跟踪速度/加速度**平滑减速**到 anchor;
#     固定 anchor 闭合需要末速度归零, 但**不在更早的拼接点先停再起**;
#     world 线/角速度匹配, SO(3) 插值明确参考系";
#     "**去掉**额外末尾重复点、张开空等及先抬高再落回的 clearance 补丁";
#     "ready 复用最后两个有效控制周期: 基座位置≤1mm、角度≤1°、公共 q 跟踪≤0.5mm、
#      基座线速≤5mm/s、角速≤5°/s; **连续 2 周期**";
#     "接近后目标相对闭合入口若已偏移 ⇒ **ENTRY_STATE_MISMATCH**(2mm/2°), 不能 set_pose 纠正" ✓
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
def audit_phase_clocks(source_frame=None, control_index=None, sim_time=None):
    """★ 三个量: ``s`` 单调不减 / ``k`` 严格递增 / ``t`` 严格递增 ✓ 返回 ``(ok, problems)``"""
    _p = []
    _s = [float(x) for x in (source_frame or [])]
    _k = [int(x) for x in (control_index or [])]
    _t = [float(x) for x in (sim_time or [])]
    for _i in range(1, len(_s)):
        if _s[_i] < _s[_i - 1] - 1e-9:
            _p.append("源相位 s 倒退: %g → %g (第 %d 点)" % (_s[_i - 1], _s[_i], _i))
    for _i in range(1, len(_k)):
        if _k[_i] != _k[_i - 1] + 1:
            _p.append("控制索引 k 非严格递增: %d → %d (第 %d 点)" % (_k[_i - 1], _k[_i], _i))
    for _i in range(1, len(_t)):
        if _t[_i] <= _t[_i - 1]:
            _p.append("仿真时间 t 未增长: %g → %g (第 %d 点)" % (_t[_i - 1], _t[_i], _i))
    return (len(_p) == 0), _p


def detect_duplicate_tail_points(commands, tol=1e-9):
    """★ 去掉**额外末尾重复点** (md ✓): 返回 ``(n_dup, keep_len)`` ✓"""
    _c = [float(x) for x in (commands or [])]
    _n = 0
    while len(_c) - _n >= 2 and abs(_c[-1 - _n] - _c[-2 - _n]) <= tol:
        _n += 1
    return int(_n), int(max(1, len(_c) - _n))


def classify_phase_idle(phases, q=None, q_open=None, close_open_frac=0.98,
                        base_pos=None, base_move_eps=1e-4):
    """★ 把【有用的闭合/受力建立】与【张开原地空等】**分开** (md ✓).

    判据 ✓: 该周期相位属"接近/就位前"**且**公共 q ≥ q_open×frac
      **且基座几乎不动** ⇒ 记为**张开空等** ✓
      (md: "额外张开空等建议 ≤2 控制周期, **必要闭合/稳定不计空等**" ✓)
    ★ T08-C2b (2026-09-27) ✓: 补**基座静止**条件 ⇒
      "带着张开的手**在接近移动**" 不属于空等 ✓ (实测: 旧定义把接近段 52 周期
      全判为空等 ✗ ⇒ 该指标永远过不了闸 ✗, 与 md"等待不是获得 ready 的唯一手段"
      也不一致: 移动中的接近不需要等待 ✓)
      · 不传 `base_pos` ⇒ 退回旧行为 (逐周期计数 ✓, 保持向后兼容 ✓)
    返回 ``{"useful": {phase:n}, "open_idle_total": n, "open_idle_by_phase": {...}}`` ✓
    """
    _ph = [str(x) for x in (phases or [])]
    _q = [float(x) for x in (q or [])]
    _bp = [np.asarray(_p, dtype=np.float64).reshape(-1)[:3] for _p in (base_pos or [])]
    _useful = {}
    for _p in _ph:
        _useful[_p] = int(_useful.get(_p, 0)) + 1
    _idle_ph = {}
    _idle_total = 0
    if _q and q_open is not None:
        _thr = float(q_open) * float(close_open_frac)
        for _i, _qq in enumerate(_q):
            _p = _ph[_i] if _i < len(_ph) else "?"
            if _qq < _thr or _p in ("close", "settle"):
                continue
            if _bp and _i < len(_bp):
                if _i == 0:
                    continue                      # 首点无前值 ⇒ 不计空等 ✓
                _d = float(np.linalg.norm(_bp[_i] - _bp[_i - 1]))
                if _d > float(base_move_eps):
                    continue                      # **在移动** ⇒ 不是空等 ✓
            _idle_total += 1
            _idle_ph[_p] = int(_idle_ph.get(_p, 0)) + 1
    return {"useful": _useful, "open_idle_total": int(_idle_total),
            "open_idle_by_phase": _idle_ph}


def audit_velocity_continuity(t_s=None, vel_world=None, omega_world=None,
                             stop_eps=1e-3, jump_ratio=10.0):
    """★ world **线/角速度**边界审计 (md ✓: 轴向 entry **不先停再启** ✗).

    判据 ✓ (两类才算问题):
      ① **中途停顿再启动**: 某点 |v|≈0 而**两侧**都显著 ⇒ 拼接点先停再起 ✗
      ② **突跳**: 相邻差 > `jump_ratio`×中位差 ⇒ 非平滑拼接 ✗
    ⚠ 平滑减速 (逐步变小)**不算**问题 ✓
    """
    _p = []

    def _n(_x):
        return float(sum(float(_v) * float(_v) for _v in _x)) ** 0.5

    for _name, _arr in (("world 线速度", vel_world), ("world 角速度", omega_world)):
        _a = [list(map(float, _v)) for _v in (_arr or [])]
        if len(_a) < 2:
            continue
        _mag = [_n(_v) for _v in _a]
        _deltas = [_n([_a[_k][_m] - _a[_k - 1][_m] for _m in range(len(_a[_k]))])
                   for _k in range(1, len(_a))]
        _sd = sorted(_deltas)
        _med = _sd[len(_sd) // 2] if _sd else 0.0
        for _k in range(1, len(_a) - 1):
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T08-C2c 修 (2026-09-27) 【区分"减速后停住"与"停一下再起步"】 ✓ ★★★★
            #   旧判据 ✓: `|v|≈0 且 (**前一步显著 or 后一步显著**)` ⇒ 用 `or` ✗
            #     ⇒ 把"**平滑减速到 anchor 后停住**"(md 明确要求 ✓)也误判成问题 ✗
            #   实测 (s37): `第 50 点中途停顿再启动 (0.02712 → 7.105e-15 → 0)` ✗
            #     ⇒ 其实是接近段末尾**减速到零**的后半段 ✓ 而非"再起步" ✗
            #   ⇒ 本改 ✓: 必须是"**之后又起步**"(下一步显著 ⇒ 有重新启动 ✗) 才算问题 ✓
            #     · 减速到 0 并保持 ⇒ 不报 ✓ (md: 平滑减速到 anchor ✓ / 闭合末速度归零 ✓)
            #     · 真"停一下再起步" ⇒ 会在**停止点的下一点**被捕获 ✓ (遗漏 ⇒ 无 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if _mag[_k] <= float(stop_eps) and _mag[_k + 1] > 10.0 * float(stop_eps):
                _p.append("%s 在第 %d 点**中途停顿再启动** (%.4g → %.4g → %.4g) ✗ "
                          "⇒ 不得在拼接点先停再起 ✓" % (_name, _k, _mag[_k - 1],
                                                    _mag[_k], _mag[_k + 1]))
        if _med > 0.0:
            for _k, _d in enumerate(_deltas):
                if _d > float(jump_ratio) * _med and _d > 1e-9:
                    _p.append("%s 第 %d→%d 点**突跳** (Δ=%.4g > %.1f×中位 %.4g) ✗"
                              % (_name, _k, _k + 1, _d, float(jump_ratio), _med))
    return (len(_p) == 0), _p


def check_ready_samples(base_pos_err_m=None, base_ang_err_deg=None, q_err_m=None,
                        lin_vel_mps=None, ang_vel_dps=None, need_cycles=2):
    """★ ready ✓: 复用最后两个**有效**控制周期, 五项同时满足 ⇒ 连续 2 周期 ✓.

    容差 (md, 待标定) ✓: 位置≤1mm / 角≤1° / 公共 q≤0.5mm / 线速≤5mm/s / 角速≤5°/s ✓
    """
    _ok = []
    for _i in range(len(base_pos_err_m or [])):
        _c = []
        _c.append(float(base_pos_err_m[_i]) <= 0.001)
        if base_ang_err_deg is not None and _i < len(base_ang_err_deg):
            _c.append(float(base_ang_err_deg[_i]) <= 1.0)
        if q_err_m is not None and _i < len(q_err_m):
            _c.append(float(q_err_m[_i]) <= 0.0005)
        if lin_vel_mps is not None and _i < len(lin_vel_mps):
            _c.append(float(lin_vel_mps[_i]) <= 0.005)
        if ang_vel_dps is not None and _i < len(ang_vel_dps):
            _c.append(float(ang_vel_dps[_i]) <= 5.0)
        _ok.append(all(_c))
    _run = 0
    for _b in _ok:
        _run = _run + 1 if _b else 0
        if _run >= int(need_cycles):
            return True, "ready ✓ (连续 %d 周期达标)" % int(need_cycles)
    return False, "NOT_READY (连续 %d 周期未达标, 实测 %s ✗)" % (int(need_cycles),
                                                            "".join("1" if b else "0" for b in _ok))


def check_entry_state(delta_pos_m=None, delta_ang_deg=None,
                      pos_tol_m=0.002, ang_tol_deg=2.0):
    """★ 入口状态 ✓: 超容差 ⇒ **ENTRY_STATE_MISMATCH** (md: **不能 set_pose 纠正** ✗)"""
    _dp = None if delta_pos_m is None else float(delta_pos_m)
    _da = None if delta_ang_deg is None else float(delta_ang_deg)
    if _dp is None and _da is None:
        return False, "ENTRY_STATE_UNKNOWN (无实测 ⇒ 不得当作匹配 ✗)"
    _bad = []
    if _dp is not None and abs(_dp) > float(pos_tol_m):
        _bad.append("位置偏差 %.4fmm > %.1fmm" % (_dp * 1000.0, pos_tol_m * 1000.0))
    if _da is not None and abs(_da) > float(ang_tol_deg):
        _bad.append("角度偏差 %.3f° > %.1f°" % (_da, ang_tol_deg))
    if _bad:
        return False, "ENTRY_STATE_MISMATCH: %s ⇒ **报告并停止, 不得 set_pose 纠正** ✗" % (
            " / ".join(_bad))
    return True, "入口匹配 ✓ (Δp=%.4fmm Δθ=%.3f°)" % (
        (_dp or 0.0) * 1000.0, _da or 0.0)

def resolve_anchor_q(*, q_nominal, q_settle_end=None, q_effective_min=None,
                     allow_negative=False, urdf_lower=0.0, oc_max_mm=3.0,
                     q_per_finger=None, asym_max_mm=None):
    """(见下方正文; 新增 `q_per_finger` 与 `asym_max_mm` ⇒ 覆盖 f0 的**非对称 q**反例 ✓)"""
    """★ md 06.4-f0 (2026-09-30) 【anchor 的 q 取值策略 + 负 drive 政策】纯函数 ✓

    为什么是纯函数 ✓: f0 要求验收测试"必须调用生产入口" ⇒ 把策略抽到这里,
      生产路径(`build_stage2_timeline` / Stage1 冻结)与测试**共用同一实现** ✓

    规则 ✓ (逐条对应 f0):
      · 优先 **q_settle_end**(指定成功 run 的**终态** ✓)
        —— f0 明令: "不能用整条 `min(q)` 或舍入日志替代指定成功 run 的终态" ✗
      · 无 settle 末 ⇒ 次选 `q_effective_min`(**降级**并显式标注 ✗), 再退 `q_nominal`
      · q < URDF lower ⇒ 必须 `allow_negative`; 否则 **拒绝该控制模式** ✗
        (f0: "不自动改 URDF 或钳成 0" ✓)
      · 已声明负 drive 时, 负值仍**有界**: 下限 = -(oc_max/2)/1000 ✓
    返回 `(q_anchor, info)`; 违规 ⇒ `raise ValueError` (调用方应在首步前调用 ✓)
    """
    info = {"nominal": None if q_nominal is None else float(q_nominal),
            "settle_end": None if q_settle_end is None else float(q_settle_end),
            "effective_min": None if q_effective_min is None else float(q_effective_min),
            "allow_negative": bool(allow_negative),
            "urdf_lower": float(urdf_lower), "oc_max_mm": float(oc_max_mm),
            "source": None, "degraded": False}
    q = None
    for _name, _v in (("settle_end_cmd", q_settle_end),
                      ("effective_min(fallback·degraded)", q_effective_min),
                      ("nominal(fallback·degraded)", q_nominal)):
        if _v is not None:
            q = float(_v)
            info["source"] = _name
            info["degraded"] = ("fallback" in _name)
            break
    if q is None:
        raise ValueError("[f0] anchor q 无法解析: settle末/effective_min/nominal 全为 None ⇒ 拒绝 ✗")
    floor_neg = -max(0.0, float(oc_max_mm)) / 2000.0
    info["floor_negative"] = floor_neg
    if q < float(urdf_lower) - 1e-9:
        if not allow_negative:
            raise ValueError(
                "[f0] 拒绝该控制模式: anchor q=%.6f < URDF lower=%.3f 且**负 drive 未声明** "
                "⇒ 不钳成 0 ✗, 不改 URDF ✗; 如需启用: `_S1_ALLOW_NEGATIVE_DRIVE=1` + "
                "声明 q 下限/力限/行程保护 + 最终 drive 读回" % (q, float(urdf_lower)))
        if q < floor_neg - 1e-9:
            raise ValueError(
                "[f0] anchor q=%.6f 超出**已声明**的负值下限 %.6f (=-oc_max/2/1000) ⇒ 拒绝 ✗"
                % (q, floor_neg))
        info["negative_drive_declared"] = True
    # ★ f0 反例: **非对称 q** (两指命令不一致 ⇒ 必须显式校验, 不能静默按平均值处理 ✗)
    if q_per_finger is not None:
        try:
            _f1, _f2 = float(q_per_finger[0]), float(q_per_finger[1])
        except Exception as _e:  # noqa: BLE001
            raise ValueError("[f0] q_per_finger 无法解析 (需 2 个数值): %s" % _e)
        _asym_mm = abs(_f1 - _f2) * 1000.0
        info["per_finger"] = [_f1, _f2]
        info["asym_mm"] = _asym_mm
        _lim = (float(os.environ.get("_S1_ASYM_MAX_MM", "1.0") or 1.0)
                if asym_max_mm is None else float(asym_max_mm))
        info["asym_max_mm"] = _lim
        if _asym_mm > _lim + 1e-9:
            raise ValueError(
                "[f0] 非对称 q 超限: |f1-f2|=%.3fmm > %.3fmm ⇒ **拒绝** "
                "(不得按平均值静默处理 ✗)" % (_asym_mm, _lim))
    info["q_anchor"] = q
    return q, info
