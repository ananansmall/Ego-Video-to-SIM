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
):
    """The immutable Stage1 grasp recipe consumed verbatim by Stage2."""
    close_q = stage1_close_q_sequence(q_open, q_grasp, close_controls)
    return {
        "revision": str(revision),
        "target_object": str(target_name),
        "q_open": float(q_open),
        "q_grasp": float(q_grasp),
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
    return grasp, recipe
