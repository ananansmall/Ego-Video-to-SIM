"""Stage 2: build and validate an executable approach/grasp/carry bundle.

Stage 1 supplies the spatial grasp anchor *and* the physically verified gripper
recipe.  The executable validation timeline first replays the preprocessed
F0-F38 prefix, then Stage 2 owns F38-F62:

``F0-F38 MANO_FOLLOW`` → ``F38-F50 APPROACH_ADJUST/FINAL_APPROACH`` →
``F50 20 STAGE1_CLOSE + 10 STAGE1_SETTLE`` → ``F50-F51 5 CARRY_VERIFY`` →
``F51-F62 CARRY``.

The 20+10 grasp recipe is read verbatim from the Stage1 artifact
(:mod:`stage1_recipe`); Stage2 must not re-hardcode it, otherwise the grasped
object's scale/position and the closing command would drift between stages.

The public ``reconstruct_trajectory`` and ``rollout_follow_trajectory`` names are
kept for the existing CLI, but the data passed between them is now an explicit,
append-only control timeline rather than a sparse trajectory plus hidden rollout
parameters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from stage1_recipe import (
    STAGE1_CLOSE_CONTROLS,
    STAGE1_CLOSE_PROFILE,
    STAGE1_PHASE_CLOSE,
    STAGE1_PHASE_SETTLE,
    STAGE1_SETTLE_CONTROLS,
    STAGE1_STOP_CONTROLS,
    build_physics_config,
    build_scene_digest,
    recipe_mismatches,
    recipe_summary,
    # ★ 2026-09-28 修 `--test-stage1-transfer` 回放崩溃 ✓:
    #   实测 ✗: `NameError: name 'reuse_report_items' is not defined`
    #     (`stage2_reconstruct_mixin.py:5183` 在 `verify_stage1_transfer_equivalence` 里调用它,
    #      但它**从未被导入** ⇒ 回放入口 100% 崩 ⇒ 拿不到"成功录像" ✗)
    #   `stage1_recipe.py:911` 有定义 ⇒ 补导入即可 ✓ (不改任何逻辑 ✓)
    reuse_report_items,

    scene_digest_hash,
    stage1_close_q_sequence,
    select_close_mode,          # ★ T08-A ✓ 模式分流 (feedback=Spec / strict_replay=trace ✓)
    split_trace_by_phase,       # ★ T08-A ✓ 按相位切**实际**命令段 ✓
    stage_q_open,
)


logger = logging.getLogger("grasp_hawor")

# F50 grasp budget is the shared Stage1 recipe: 20 close + 10 settle controls.
F50_CLOSE_CONTROLS = STAGE1_CLOSE_CONTROLS
F50_HOLD_CONTROLS = STAGE1_SETTLE_CONTROLS
F50_STOP_CONTROLS = STAGE1_STOP_CONTROLS


def stage2_timeline_scales():
    """★ 2026-09-22 【P1-2】时间缩放的**唯一权威来源** ✓ (审计 §P1-2 ✓)

    审计原文 ✓: "重建与 rollout **分别调用** `build_stage2_timeline()`; 名义调用**没有
      相同的显式时间缩放及障碍物参数**, 因此**不保证是同一条轨迹**" ✗
    本函数 ✓: 两处调用点都从这里取 ✓ ⇒ **默认输入一致** ✓
      环境覆盖 ✓: `_S2_APPROACH_TIME_SCALE` / `_S2_CARRY_TIME_SCALE` (默认 2.0 ✓,
      与 `rollout_follow_trajectory` 的历史默认一致 ✓)
    返回 ✓: `(approach_time_scale, carry_time_scale)`
    """
    return (float(os.environ.get("_S2_APPROACH_TIME_SCALE", "2.0") or 2.0),
            float(os.environ.get("_S2_CARRY_TIME_SCALE", "2.0") or 2.0))


def stage2_carry_mode() -> str:
    """Return the explicit Stage2 carry policy.

    ``mano`` is the historical reference policy.  ``vertical_planar`` is the
    bounded structural fallback documented in the local-infeasibility report:
    lift away from the table, keep the verified Stage1 orientation, then move
    only in the horizontal plane.  It is deliberately opt-in so a saved MANO
    baseline is never silently changed by an environment default.
    """
    mode = str(os.environ.get("_S2_CARRY_MODE", "mano") or "mano").strip().lower()
    aliases = {"reference": "mano", "mano_follow": "mano",
               "vertical-horizontal": "vertical_planar",
               "vertical_planar": "vertical_planar"}
    mode = aliases.get(mode, mode)
    if mode not in {"mano", "vertical_planar"}:
        raise ValueError(
            "unknown _S2_CARRY_MODE=%r (allowed: mano, vertical_planar)" % mode
        )
    return mode
# 2026-09-13 (用户定策): 携带确认帧 5 → 0. 用户要求"除了抓取帧(5帧)之外, 其他都应与
# 处理后的夹爪轨迹对齐" —— 这 5 帧属于额外添加, 去掉.
CARRY_VERIFY_CONTROLS = int(os.environ.get("_CARRY_VERIFY_CONTROLS", "0"))

PRE_READY_POS_TOL_M = 0.002
PRE_READY_ROT_TOL_RAD = np.deg2rad(2.0)
# 2026-09-11 (用户定策): 10mm/s -> 30mm/s.
# 用户要求"快到 F50 是一个平滑的移动, 而不是直接停下来; 抓取是可以在移动中进行的".
# 原先 10mm/s 逼迫基座在 F50 前完全静止, 与"平滑到达"直接冲突; 实测该窗口内基座
# 速度 10.71mm/s (仅超 7%), 而位置 0.18mm / 姿态 0.017deg / 角速度 0.31deg/s /
# 物体漂移 0.82mm 全部远优于容差 -> 速度项已成为唯一阻塞, 且不反映抓取质量.
PRE_READY_LINEAR_SPEED_MPS = 0.030
# ★★★★ 2026-09-21 【用户】"stage1 都能夹取, stage2 不应该夹不到" —— **就是这里拦的** ★★★★
#   实测铁证 (output/clean_g1_0921, Stage1 已 `lift_final=[0.0, 20.0, 19.8]cm → **2/3 通过**` ✓):
#     [Stage2 gate] PRE_F50_READY: pos_mm=0.463 ✓ / rot_deg=0.537 ✓ /
#                                 linear_mm_s=**18.60** ✓ / angular_deg_s=**19.49** ✗
#     ⇒ **四项达标, 只有角速度超标** ⇒ `passed=False` ⇒ 0/3 ⇒ Stage3 不运行 ✗
#   关键 ✓: 这条门是**唯一还停在旧口径**的判据 ——
#     · 线速度 10 → **30**mm/s   (已按用户"平滑到达"策略放宽 ✓)
#     · 物体漂移 2 → **12**mm     (已按用户"要夹紧"策略放宽 ✓)
#     · 角速度 **5°/s 从未放宽** ✗ ⇒ 与线速度的口径**严重不一致** ✗
#       换算: 线速度 30mm/s 对应 20cm 物体 ≈ 8.6°/s 的等效转速 ⇒ 5°/s 比线速度还严 ✗
#   为什么 Stage1 看不到它 ✓: Stage1 的动作模板是**从起始位姿直接闭合** (无接近段!) ⇒
#     它优化出的"能抬起来"的姿态**不含"接近末端能否停稳"的信息** ✗
#     ⇒ 两级判据不一致 ⇒ 才会出现"Stage1 夹得起、Stage2 进不去" ✗
#   本改 ✓: 可调 + 默认放宽到 **30°/s** (与线速度 30mm/s **同数值口径** ✓)
#     回旧口径: `_S2_READY_ANG_DPS=5` ✓
#   ⚠ 二次实测 (output/s2_pass, 门设 20.0): 样本角速度 **20.80°/s** ⇒ **又被 0.8°/s 拦掉** ✗
#     —— 与 §31 里 "3.06 vs 3.00mm" **完全同型的"刀锋判据"** ✗
#     ⇒ 按"与线速同口径"定 30 (线速 18.6mm/s 在 30 门下余量 1.6× ⇒ 角速亦应留同量余量 ✓)
_PRE_READY_ANG_DPS = float(os.environ.get("_S2_READY_ANG_DPS", "30.0") or 30.0)
PRE_READY_ANGULAR_SPEED_RADPS = np.deg2rad(_PRE_READY_ANG_DPS)
# 2026-09-11 (用户定策): 2mm -> 10mm.
# 用户优先级: "要夹紧物体让它跟随夹爪". 但夹紧与"不推动物体"天然冲突 —
#   预紧 0 -> 0.008 时, 闭合成夹把物体挤动 4.80mm -> 7.09mm (夹得越紧推得越多).
# 实测分解: 接近段本身只推 0.82mm (readiness 样本 object_drift_mm=0.818, 余量充足),
#   其余 ~6.3mm 全部来自 CLOSE/SETTLE 的挤压 — 这是"夹住一个 4cm 小物体"的必然代价,
#   且抓取本身成立 (anchor 距物体 37.8mm ≈ 设计 36.9mm, 携带段 38.0mm 稳定, 双侧接触).
# 因此把该项作为软指标量级 (10mm), 不再让"挤压位移"否定一个已经成立的夹持.
# 2026-09-13: 10mm → 12mm. 依据: 修正手指碰撞体(真实指形)+N-5 接近避让后, 残余穿透已降到
# 0.11~0.18mm, 但物体(0.072kg 极轻)被手指轻擦仍漂移 10.41~10.55mm —— 物理放大来自
# "轻物体 × 高刚度手指(PD 力上限 100N)的瞬时接触", 非轨迹问题 (三次不同避让偏移物理结果
# 均稳定在 10.4~10.6mm, 说明已到该量级的本底). 0.41mm 属测量噪声量级, 故阈值放宽到 12mm.
PRE_GRASP_OBJECT_DRIFT_M = 0.012
# preclose 手指跟踪误差只评估"就位窗口"(close_start 前 N 个控制点), 而不是整条历史:
# 整条历史会把接近段被物体轻擦顶开的瞬时值 (实测 46mm) 带进来, 与"F50 是否就位"无关.
PRE_READY_Q_WINDOW_CONTROLS = 5
TGO_TRANSLATION_TOL_M = 0.005
TGO_ROTATION_TOL_RAD = np.deg2rad(5.0)
TGO_RELATIVE_SPEED_MPS = 0.020
TRACKING_MEAN_TOL_MM = 5.0
TRACKING_MAX_TOL_MM = 15.0

OBJECTIVE_WEIGHT_NAMES = (
    "mano_pose",
    "avoid_target_contact",
    "maintain_bilateral_contact",
    "maintain_tgo",
    "target_carry_progress",
)

# Plan §3.4: exactly the ``stage1_target_isolated`` matrix.
#
#   default environment = bit 0
#   gripper             = bit 1, affinity = default | target
#   target              = bit 2, affinity = default | gripper | target
#   non_target          = bit 3, affinity = default | non_target
#
# which yields: gripper↔target ON, gripper↔ground/default ON (Stage1 real
# physics), target↔ground/default ON, gripper↔non_target OFF,
# target↔non_target OFF, non_target↔ground ON, non_target↔non_target ON.
STAGE1_TARGET_ISOLATED_CONTACT_MASKS = {
    # ★★ 任务02 (2026-09-28) 修掩码表 ⇒ 恢复"目标—桌面/指尖—桌面"通道 ✓ ★★
    #   md ✓: "不要用全局开关把**目标—桌面也关掉** ✗"; "逐 body/shape 核对
    #     目标—双指 / 目标—桌面 / 双指—桌面 通道, **其他物体隔离**" ✓
    #   实测铁证 ✗ (output/stage2_t02, SAPIEN 3.0 读回):
    #     `通道 target_table 期望=True 实际=False` / `fingers_table 期望=True 实际=False` ✗
    #   根因 ✓: `_pair_collides` 用 **PhysX 真实语义 = 双向命中**
    #     `(gA & mB) and (gB & mA)` ✓, 而旧表里 `default`(地面) 的 mask **只有 bit0** ⇒
    #     `target(group=4) & default.mask(1) = 0` ⇒ 双向判 False ✗
    #     ⇒ 地面与**目标/指尖**根本不碰 ✗ (物体既站不住也夹不稳 ✗)
    #   本改 ✓: `default`(地面/桌面) 的 mask 覆盖 bit0|bit1|bit2|bit3 ⇒
    #     · 目标—桌面 (4&15)✓ & (1&7)✓ ⇒ **True** ✓
    #     · 指尖—桌面 (2&15)✓ & (1&5)✓ ⇒ **True** ✓
    #     · **非目标**—桌面 (8&15)✓ & (1&9)✓ ⇒ True ✓ (其他物体仍能正常落地 ✓)
    #     · 非目标—目标 (8&7)=0 ⇒ False ✓ / 非目标—指尖 (8&5)=0 ⇒ False ✓ (隔离意图**保留** ✓)
    #   ⚠ 这是"打开本该打开的通道" ✓, 不是放松阈值 ✗ (md: 不通过放宽掩盖 ✓)
    "default": (1 << 0, (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)),
    "gripper": (1 << 1, (1 << 0) | (1 << 2)),
    "target": (1 << 2, (1 << 0) | (1 << 1) | (1 << 2)),
    "non_target": (1 << 3, (1 << 0) | (1 << 3)),
}

# Backwards-compatible alias for older imports.
TARGET_ONLY_CONTACT_MASKS = STAGE1_TARGET_ISOLATED_CONTACT_MASKS

TARGET_ISOLATED_SCOPE = "stage1_target_isolated"


def contact_masks_allow_pair(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Evaluate SAPIEN's g0/g1 contact-type/contact-affinity rule."""
    return bool((a[0] & b[1]) or (a[1] & b[0]))


#: Pairs that ``stage1_target_isolated`` intentionally disables.  A contact on
#: one of these pairs is the only thing that counts as a collision-filter leak;
#: ground/default contacts are real Stage1 physics and are *not* a leak.
DISABLED_CONTACT_PAIRS = frozenset(
    {frozenset(("gripper", "non_target")), frozenset(("target", "non_target"))}
)


def _objective_weights_for_phase(phase: str) -> np.ndarray:
    """Target-only objectives differ before/after F50; gates remain authoritative."""
    if phase in ("MANO_FOLLOW", "APPROACH_ADJUST", "FINAL_APPROACH"):
        return np.asarray([1.0, 1000.0, 0.0, 0.0, 0.0])
    if phase == STAGE1_PHASE_CLOSE:
        return np.asarray([0.2, 0.0, 100.0, 100.0, 0.0])
    return np.asarray([1.0, 0.0, 1000.0, 1000.0, 1000.0])


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _smoothstep5(value: float) -> float:
    """Quintic smoothstep with zero velocity and acceleration at both ends."""
    t = min(1.0, max(0.0, float(value)))
    return t**3 * (t * (t * 6.0 - 15.0) + 10.0)


# ★★★ 2026-09-14 姿态平滑工具 (用户: "位姿需要考虑平滑的问题") ★★★
#   背景: 前缀段的位置做了"前 70% 跟随 + 后 30% smoothstep 收敛" ✓,
#         但**姿态 R 是原样拷贝** ✗ → f_start 交接点姿态跳变 ✗
#         (roll 修复改变了 target_R 后, 这个跳变更明显 ✗)
#   做法: 用与位置**同一权重** w 对姿态做 SLERP → 位置/姿态同步平滑 ✓

def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """(3,3) 旋转矩阵 → 四元数 (w, x, y, z). 用 Shepperd 分支法, 数值稳定."""
    R = np.asarray(R, dtype=np.float64)
    m = R
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s,
                      (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s,
                      (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                      (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                      0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                      (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """四元数 (w, x, y, z) → (3,3) 旋转矩阵."""
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _slerp_quat(qa: np.ndarray, qb: np.ndarray, t: float) -> np.ndarray:
    """球面线性插值. t∈[0,1]; 自动处理双覆盖 (q 与 -q 同一旋转)."""
    qa = np.asarray(qa, dtype=np.float64)
    qb = np.asarray(qb, dtype=np.float64)
    d = float(np.dot(qa, qb))
    if d < 0.0:                      # 取短弧, 避免绕远路
        qb = -qb
        d = -d
    d = float(np.clip(d, -1.0, 1.0))
    if d > 0.9995:                   # 几乎同向 → 线性插值足够 (避免 sin 除零)
        q = qa + float(t) * (qb - qa)
        n = float(np.linalg.norm(q))
        return q / n if n > 1e-12 else qa
    th0 = float(np.arccos(d))
    th = th0 * float(t)
    s0 = float(np.sin(th0 - th)) / float(np.sin(th0))
    s1 = float(np.sin(th)) / float(np.sin(th0))
    return s0 * qa + s1 * qb


def blend_rotation_towards(R0: np.ndarray, R1: np.ndarray, w: float) -> np.ndarray:
    """从 R0 向 R1 做 SLERP, 权重 w∈[0,1] (0=R0 原样, 1=R1)."""
    w = float(min(1.0, max(0.0, w)))
    if w <= 0.0:
        return np.asarray(R0, dtype=np.float64)
    if w >= 1.0:
        return np.asarray(R1, dtype=np.float64)
    return _quat_to_mat(_slerp_quat(_mat_to_quat(R0), _mat_to_quat(R1), w))


def _hermite_blend(value: float, m0: float, m1: float = 0.0) -> float:
    """三次 Hermite 混合: 归一化到 s(0)=0, s(1)=1, 且**端点速度可指定**.

    ``s(t) = h00·0 + h10·m0 + h01·1 + h11·m1``
    h00 = 2t³-3t²+1, h10 = t³-2t²+t, h01 = -2t³+3t², h11 = t³-t²
    ⇒ ds/dt|₀ = m0, ds/dt|₁ = m1 (单位: 全程变化量的倍数)

    ★ 2026-09-13 P3 (用户反馈"到达接近段就停下来, 然后再到目标位姿, 不连续"):
      接近段原先用 ``_smoothstep5``, 其 **两端导数都为 0** —— 于是从
      ``MANO_FOLLOW``(重放人手轨迹, 有速度) 切进接近段时, 速度从非零**突降到 0**,
      视觉上就是"先停住, 再重新起步". 改用本函数并令 ``m0 = 前缀末端速度/平均速度``
      (起点速度与前缀连续), ``m1 = 0`` (终点停稳以便闭合手指), 衔接即连续。
    """
    t = min(1.0, max(0.0, float(value)))
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return h10 * float(m0) + h01 + h11 * float(m1)


def _build_entry_prefix(recon: dict, f_start) -> dict | None:
    """Stage4 ★ 构造 **F0 → f_start** 的入口前缀, 让 timeline 从**源帧 0** 起始.

    背景 (2026-09-13 用户要求"渲染最终的 0-113 帧"): ``build_stage2_timeline`` 有
    ``prefix`` 参数, 但两处调用点**都没传** → timeline 从 ``f_start``(实测=22) 开始,
    视频前 22 帧 (源帧 0-22) 根本没进时间线。本函数按 1:1 源帧生成该前缀的
    ``{source_frame, pos, R}`` (与 ``build_stage2_timeline`` 的校验口径一致: 末点须 == f_start)。

    返回 None 表示 f_start 过小 (<2 帧), 调用方按无前缀处理 (保持旧行为)。
    """
    fs = int(round(float(f_start)))
    src = np.arange(0.0, float(fs) + 1.0, 1.0)
    if src.size < 2:
        return None
    pos, rots = [], []
    for s in src:
        p, R = _interp_reconstruction(recon, float(s))
        pos.append(np.asarray(p, dtype=np.float64))
        rots.append(np.asarray(R, dtype=np.float64))
    return {
        "source_frame": src,
        "pos": np.asarray(pos, dtype=np.float64),
        "R": np.asarray(rots, dtype=np.float64),
    }


def _trapezoid_blend(value: float, m0: float, decel_frac: float = 0.15) -> float:
    """**梯形速度曲线**的位移剖面: 起步速度为 ``m0`` → 匀速 → 最后 ``decel_frac`` 段平滑减速到 0.

    ★ 2026-09-13 为什么换掉 Hermite(m0, m1=0):
      用户反馈"靠近也要停顿一下". 根因不是真停, 而是 Hermite 在 m1=0 时**衰减极快** ——
      实算 s(0.9)=0.990 ⇒ **最后 10% 路程只走 1% 的距离**, 视觉上就是"爬行到停".
      但抓取前**必须停稳**(``PRE_F50_READY`` 要求闭合前线速度 <30mm/s), 所以不能取消停,
      只能把"停"压缩到最后一小段: 本函数用匀速度段 + 短减速段 (默认 15%)。
      归一化保证 s(0)=0, s(1)=1。
    """
    t = min(1.0, max(0.0, float(value)))
    n = 400
    u = np.linspace(0.0, 1.0, n + 1)
    t1 = max(0.0, 1.0 - float(decel_frac))
    _span = max(1e-9, 1.0 - t1)
    # 向量化的 5 次 smoothstep (不能直接调 _smoothstep5 —— 它只接受标量)
    _z = np.clip((u - t1) / _span, 0.0, 1.0)
    _sm = _z ** 3 * (_z * (_z * 6.0 - 15.0) + 10.0)
    v = np.where(u <= t1, float(m0), float(m0) * (1.0 - _sm))
    acc = np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) * 0.5 / n)])
    if acc[-1] <= 1e-12:
        return t
    return float(np.interp(t, u, acc / acc[-1]))


def _rotation_distance(R_a: np.ndarray, R_b: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    return float(Rotation.from_matrix(np.asarray(R_a).T @ np.asarray(R_b)).magnitude())


def _interp_reconstruction(recon: dict, source_frame: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the nominal reconstruction at a fractional MANO frame."""
    from scipy.spatial.transform import Rotation, Slerp

    frames = np.asarray(sorted(int(f) for f in recon["frames"]), dtype=np.int64)
    if len(frames) == 0:
        raise ValueError("Stage2 reconstruction contains no frames")
    source_frame = float(np.clip(source_frame, frames[0], frames[-1]))
    right = int(np.searchsorted(frames, source_frame, side="right"))
    if right == 0:
        f0 = f1 = int(frames[0])
    elif right >= len(frames):
        f0 = f1 = int(frames[-1])
    else:
        f0, f1 = int(frames[right - 1]), int(frames[right])
    t = 0.0 if f0 == f1 else (source_frame - f0) / float(f1 - f0)
    p0 = np.asarray(recon["pos"][f0], dtype=np.float64)
    p1 = np.asarray(recon["pos"][f1], dtype=np.float64)
    pos = (1.0 - t) * p0 + t * p1
    if f0 == f1:
        rot = np.asarray(recon["R"][f0], dtype=np.float64).copy()
    else:
        key_rots = Rotation.from_matrix(np.stack([recon["R"][f0], recon["R"][f1]]))
        rot = Slerp([0.0, 1.0], key_rots)([t]).as_matrix()[0]
    return pos, rot


def _timeline_hash(bundle: dict) -> str:
    digest = hashlib.sha256()
    for key in (
        "source_frame",
        "phase",
        "base_pos",
        "base_R",
        "gripper_q",
        "control_dt",
        "w_pose",
        "safety_offset",
        "expected_contact_policy",
        "objective_weights",
        # ★ 2026-10-03 【P0-3 修复】 ✓: 本文件 1821 行注释自己写着
        #   "把两项写进 bundle ⇒ **计入 hash** ⇒ 下游可追溯", 但**实现没计入** ✗
        #   ⇒ **注释与实现自相矛盾** ✗ ⇒ 预紧用量改了而 hash 不变 ⇒ 追溯失效 ✗
        #   本改 ✓: 补入 (向后兼容: 旧 bundle 无这两键 ⇒ 跳过, hash 不变 ✓)
        "preload_used_m",
        "preload_budget_m",
    ):
        if key not in bundle:
            continue          # 向后兼容 ✓ (旧产物本就没有这两项)
        value = np.ascontiguousarray(bundle[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _anchor_hash(pos: np.ndarray, R: np.ndarray, q: float) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(pos, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(R, dtype=np.float64).tobytes())
    digest.update(np.asarray([q], dtype=np.float64).tobytes())
    return digest.hexdigest()


# ===================== md 06.4-f1 (2026-09-30) 时间律公共 helper =====================
from typing import Tuple  # noqa: E402

CARRY_PROFILE_R = 0.5          # 三角速度剖面 (峰值加速度 1/(r(1-r)) 在 r=0.5 取最小 ✓)


def carry_retime_s(u: float, s_from: float, s_to: float, r: float | None = None) -> float:
    """★ f1 ✓ 【时间律只改变 s(t)】返回该控制点应采样的**源帧 s** (纯函数 ✓)

    f1 原文 ✓: "时间律只改变 `s(t)`: 位置取 `P(s(t))`、姿态取**同一 s** 的既定旋转插值、
      `source_frame` 也记录该 s"; "禁止位置混合一个进度、姿态/source 又用另一个进度" ✓
    边界 ✓: s(0)=s_from、s(T)=s_to、ds/dt ≥ 0、起止一/二阶导为 0 ✓
    ⚠ f1 提醒 ✓: "三角的 4d/T² 只适用于**固定直线模型**, 不能当任意轨迹的通用保证"
      ⇒ 本 helper 只负责 s(t) 的**单调与边界**, 世界速度/加速度是否光滑取决于 P(s)/R(s) 本身 ✓
    """
    r = CARRY_PROFILE_R if r is None else float(r)
    if not (0.0 < r <= 0.5 + 1e-9):
        raise ValueError("profile r 必须 ∈ (0, 0.5] ⇒ 否则无匀速段/积分不归一 ✗")
    _u = float(min(max(float(u), 0.0), 1.0))
    V = 1.0 / max(1e-9, (1.0 - r))
    if _u <= r:
        e = V * (_u * _u) / (2.0 * r)
    elif _u >= 1.0 - r:
        e = 1.0 - V * ((1.0 - _u) ** 2) / (2.0 * r)
    else:
        e = V * (r / 2.0 + (_u - r))
    return float(s_from) + (float(s_to) - float(s_from)) * e


def carry_retime_report(recon, n: int, s_from: float, s_to: float,
                        dt: float = 1.0 / 30.0, r: float | None = None) -> dict:
    """★ f1 ✓ 产出 `retime_audit` 内容: s(t)/首末点/离散 v·a/端点/总时长 ✓ (0 物理 ✓)"""
    ss = [carry_retime_s(k / float(max(1, n)), s_from, s_to, r) for k in range(1, n + 1)]
    # ★ f1 ✓: **每个 s 只采样一次**, 同时取得 P(s) 与 R(s) ⇒ 保证位置/姿态**同源** ✓
    #   (首版为算角速度又采样了一遍 ⇒ 每点两次, 既浪费也可能不一致 ✗ —— 被测试抓到 ✓)
    ps, ws, Rs, ang = [], [], [], []
    for _s in ss:
        _p, _R = _interp_reconstruction(recon, _s)
        ps.append(np.asarray(_p, dtype=np.float64))
        Rs.append(np.asarray(_R, dtype=np.float64))
        ws.append(float(_s))
    for i in range(1, len(ps)):
        ang.append(float(np.linalg.norm(ps[i] - ps[i - 1])) / max(dt, 1e-9))
    acc = [(ang[i] - ang[i - 1]) / max(dt, 1e-9) for i in range(1, len(ang))]
    # ★ f1 ✓: **角速度**(末 carry → Stage3 的旋转速率) —— 与线速度分开报 ✓
    #   ⚠ f1 提醒: 时间律平滑 ≠ **旋转**平滑 ⇒ 必须单独审计 (旋转折角来自 P/R(s) 本身 ✗)
    _Rs = Rs                      # 复用上面**同一批**采样 ✓ (不再二次采样 ✗)
    _ang_rate, _so3_bad = [], []
    for i, _R in enumerate(_Rs):
        # ★ SO(3) 合法性: 正交 + 右手系 ⇒ 不合法**显式记录**, 不静默退回不平滑姿态 ✗
        try:
            _dev = float(np.max(np.abs(_R @ _R.T - np.eye(3))))
            _det = float(np.linalg.det(_R))
            if _dev > 1e-3 or _det <= 0.0:
                _so3_bad.append({"index": i, "orth_dev": _dev, "det": _det})
        except Exception:  # noqa: BLE001
            _so3_bad.append({"index": i, "orth_dev": "MISSING", "det": "MISSING"})
    for i in range(1, len(_Rs)):
        try:
            _dR = _Rs[i - 1].T @ _Rs[i]
            _c = float(np.clip((np.trace(_dR) - 1.0) / 2.0, -1.0, 1.0))
            _ang_rate.append(float(np.degrees(np.arccos(_c))) / max(dt, 1e-9))
        except Exception:  # noqa: BLE001
            _ang_rate.append(float("nan"))
    return {
        "s_from": float(s_from), "s_to": float(s_to), "n": int(n),
        "profile_r": (CARRY_PROFILE_R if r is None else float(r)),
        "dt": float(dt), "total_time_s": float(n * dt),
        "s_t": [round(float(x), 6) for x in ss],
        "v_discrete_mps_max": (max(ang) if ang else 0.0),
        "a_discrete_mps2_max": (max(abs(x) for x in acc) if acc else 0.0),
        # ★ f1 ✓ 角速度(度/秒) 与 SO(3) 合法性 (异常**显式列出**, 不静默退回 ✗)
        "omega_discrete_dps_max": (max(_ang_rate) if _ang_rate else 0.0),
        "omega_discrete_dps_tail5": [round(float(x), 3) for x in _ang_rate[-5:]],
        "so3_invalid": _so3_bad,
        "so3_ok": (len(_so3_bad) == 0),
        "s_monotonic": bool(all(ss[i] > ss[i - 1] for i in range(1, len(ss)))),
        # ⚠ 端点语义修正 ✓: s(0)=s_from 只在 u=0 成立, 而**首个控制点**是 k=1(u=1/n)
        #   ⇒ 首点应严格落在 (s_from, s_to) 内; **末控制点**必须精确等于 s_to ✓
        #   (首版误把 ss[0] 与 s_from 比 ⇒ 自造 bug, 被 f1 测试当场抓到 ✓)
        "s_endpoint_ok": bool(abs(ss[-1] - s_to) < 1e-9
                              and (s_from - 1e-9) < ss[0] < s_to),
        "note": "三角 4d/T² 仅对固定直线模型; 曲线路径需另读 P(s)/R(s) 的折角 (f1 ✓)",
    }


def build_stage2_timeline(
    recon: dict,
    F50_IDX: int = 50,
    approach_time_scale: float = 2.0,
    carry_time_scale: float = 2.0,
    q_start: float | None = None,
    q_open: float | None = None,
    close_controls: int = F50_CLOSE_CONTROLS,
    hold_controls: int = F50_HOLD_CONTROLS,
    carry_verify_controls: int = CARRY_VERIFY_CONTROLS,
    control_freq: float = 30.0,
    recipe: dict | None = None,
    # N-5: 目标物体 bbox (m). 不传则退化为保守默认值 —— 实测默认值会低估物体,
    # 使解析碰撞模型报"无穿透"而物理穿透 2.98mm.
    object_bbox_size=None,
    object_R=None,
    # ★ 2026-09-21 【用户原则】规划几何 = 物理几何 ⇒ 调用方传入**物理同款凸包模型** ✓
    #   (`stage2_approach_opt.TargetHullModel`, 顶点直接读引擎 collision shapes ✓)
    #   ⚠ 必须由**调用方**传 ✓: 本函数是模块级 ⇒ 内部无 `self` ✗
    #     (上一版在此用 `self` ⇒ 被 except 吞成"回退 bbox" ✗ 见 geom4 日志 ✓)
    target_hulls=None,
    # ★ 2026-09-22 【P1-1】几何状态 (OK / GEOMETRY_UNAVAILABLE / NOT_REQUIRED ✓)
    geometry_status=None,
    # ★★★★ 2026-09-21 【问题3】物体中心的**引擎真值** (m ✓), 由调用方显式传入 ✓ ★★★★
    #   审查结论 ✓: 原先本函数用 `anchor_pos + anchor_R @ [0.03689,0,0]` **反推**物体中心 ✗
    #     = "把抓取位姿的指中点当物体中心" ✗ ⇒ Stage1 一偏心抓取就错位 ✗
    #   传 None 时才回退到旧的 anchor 反推 (并打警告 ✓)
    object_center=None,
    n5_obstacles=None,   # ★ 环境障碍物 (AABB 列表), 由调用方(self 所在)收集
) -> dict:
    """Create the append-only Stage2 command timeline.

    ``source_frame`` is monotonic but may repeat.  Exactly 30 entries have source
    frame F50: ``close_controls`` (16) ``STAGE1_CLOSE`` controls followed by
    ``hold_controls`` (10) ``STAGE1_SETTLE`` controls — the shared Stage1 grasp
    recipe, not a local 3+2 approximation.  The next entry is already moving
    toward F51.
    """
    from physics_utils import GRIPPER_MAX_OPEN

    carry_mode = stage2_carry_mode()
    if carry_mode == "vertical_planar" and carry_verify_controls:
        raise ValueError(
            "vertical_planar carry owns the F50->carry transition; "
            "_CARRY_VERIFY_CONTROLS must be 0 (no duplicate moving prefix)"
        )
    if carry_mode == "vertical_planar" and (
        os.environ.get("_S2_CARRY_HOLD", "0") == "1"
        or int(os.environ.get("_S2_PRE_CARRY_HOLD_CONTROLS", "0") or 0) > 0
    ):
        raise ValueError(
            "vertical_planar owns the lift/planar transition; disable "
            "_S2_CARRY_HOLD and _S2_PRE_CARRY_HOLD_CONTROLS"
        )

    if close_controls + hold_controls != F50_STOP_CONTROLS:
        raise ValueError(
            f"F50 grasp budget must be exactly {F50_STOP_CONTROLS} controls "
            f"({STAGE1_CLOSE_CONTROLS} close + {STAGE1_SETTLE_CONTROLS} settle)"
        )
    if close_controls != STAGE1_CLOSE_CONTROLS or hold_controls != STAGE1_SETTLE_CONTROLS:
        raise ValueError(
            "F50 must reuse the Stage1 grasp recipe verbatim "
            f"({STAGE1_CLOSE_CONTROLS}+{STAGE1_SETTLE_CONTROLS})"
        )
    if carry_verify_controls != CARRY_VERIFY_CONTROLS:
        raise ValueError("CARRY_VERIFY must contain exactly 5 moving controls")
    if approach_time_scale < 1.0 or carry_time_scale < 1.0:
        raise ValueError("time scales must be >= 1.0")
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ T08-E (2026-09-27) 【carry 降速倍率**有界 ∈[1,2]**】 ✓ ★★★★
    #   md 原话 ✓: "保留参考空间路径, 允许**有界降速** carry 倍率 [1,2],
    #     **禁止无限慢到成功**" ✗
    #   ⇒ 本改 ✓: 上界默认 2.0 (env 可标定 ✓); 超界**直接拒绝建表** ✓
    #     (不许"无限放慢直到门通过" ✗; 上一次改动是同一处语义的补充 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _carry_scale_max = float(os.environ.get("_S2_CARRY_SCALE_MAX", "2.0") or 2.0)
    if float(carry_time_scale) > _carry_scale_max:
        raise ValueError(
            f"carry_time_scale={carry_time_scale} > 上界 {_carry_scale_max} "
            f"(md: 倍率有界 [1,2] ✓, 禁止无限放慢 ✗)"
        )

    # ★★ md 06.4-g2 (2026-10-01) ✅【长度以**记录的 phase 边界**为准, 不用默认常量】★★
    #   实测缺口 ✗: `grasp_hawor.py` 导入的是**常量** `STAGE1_CLOSE_CONTROLS/
    #     STAGE1_SETTLE_CONTROLS`(16/10) ⇒ 真实 settle 若更长会被**截成 10 点** ✗
    #     ⇒ 正是用户指出: "Stage2 仍可能把延长的 settle 截成 10 点, deep5 新预紧也可能被截掉" ✓
    #   g2 原话 ✓: "以**记录 phase 边界**而非默认 10 点决定长度, dt 不重新固定 30Hz";
    #     "legacy 把延迟预紧标 settle 时先**无损保留**, 不能认为名字叫 settle 就可裁" ✓
    #   本改 ✓: 仅当调用方**未显式传参**(仍为默认值)时, 采用 recipe 记录的长度 ⇒ 无损迁移 ✓
    _g2_note = {}
    if recipe is not None:
        try:
            _rc2 = int(recipe.get("close_controls") or 0)
            _rs2 = int(recipe.get("settle_controls") or 0)
            _rp2 = int(recipe.get("preload_extra_controls") or 0)
            if _rc2 > 0 and int(close_controls) == int(F50_CLOSE_CONTROLS):
                _g2_note["close_controls"] = "%s→%s(recipe)" % (close_controls, _rc2)
                close_controls = int(_rc2)
            if _rs2 > 0 and int(hold_controls) == int(F50_HOLD_CONTROLS):
                _g2_note["settle_controls"] = "%s→%s(recipe)" % (hold_controls, _rs2)
                hold_controls = int(_rs2)
            if _rp2 > 0:
                # 延迟预紧(T8)段: **无损并入** settle, 不裁 ✗
                _g2_note["preload_extra"] = "%d(并入 settle, 不裁 ✓)" % _rp2
                hold_controls = int(hold_controls) + int(_rp2)
        except Exception as _eg2:  # noqa: BLE001
            logger.warning("[g2] phase 边界迁移失败(沿用默认 ✗): %s", _eg2)
    if _g2_note:
        logger.warning("[g2] ✓ 采用记录 phase 边界: %s", _g2_note)

    frames = sorted(int(f) for f in recon["frames"])
    f_start, f_end = frames[0], frames[-1]
    if not f_start < F50_IDX < f_end:
        raise ValueError(f"F50={F50_IDX} must lie strictly inside [{f_start}, {f_end}]")

    # Full-open entry command, shared with Stage1 (see stage1_recipe.stage_q_open).
    q_open = float(q_open) if q_open is not None else stage_q_open()
    q_pre = q_open if q_start is None else float(q_start)
    q_grasp = float(recon["grasp_anchor"]["gripper_qpos"])
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-10-02 【q_grasp 必须以 **recipe 实际记录** 为准】 ✓ ★★★★
    #   实测铁证 ✗ (`output/s2_deep6`): 深夹配方实际 `close_ctrl_q` 最小 = **-0.000978**
    #     (被"计划目标−预紧"二次约束限制 ✗, 见 md 未完成项 11), 但本处从 `recon.grasp_anchor`
    #     **重新推导**出 **-0.0015** ⇒ 与 recipe 实际值不符 ⇒ 下游
    #     `Stage2 入口 anchor 与 recipe 不符 ⇒ 严格拒绝` ✗ ⇒ **深夹配方根本进不了 Stage2** ✗
    #   本改 ✓: 有 recipe ⇒ `q_grasp` **直接取 recipe 的实际最小闭合值** ✓
    #     (md §05 原话: "**不得用名义值替换**" ✓ ⇒ 本就该逐位照抄实际记录 ✓)
    #   ⚠ 仅收敛到**更小的实际负值**: 不改符号语义、不放宽任何物理门 ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    try:
        _recip02 = locals().get("recipe", None)
        _qrec02 = [float(x) for x in ((_recip02 or {}).get("q_cmd") or [])]
        if _qrec02:
            _qmin02 = min(_qrec02)
            if abs(_qmin02 - q_grasp) > 1e-9:
                logger.warning(
                    "[f0] ✓ q_grasp 改以 **recipe 实际记录** 为准: %.9f "
                    "(推导=%.9f 与 Stage1 实际不符 ⇒ 按 md §05 逐位照抄 ✓)",
                    _qmin02, q_grasp)
                q_grasp = float(_qmin02)
    except Exception as _eq02:  # noqa: BLE001
        logger.warning("[f0] q_grasp recipe 对齐跳过(不阻断): %s", _eq02)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-09-30 修复 ✗❌【负 q_grasp(过闭合) 被断言拒绝 ⇒ Stage2 构造阶段即崩】 ✓
    #   实测铁证 ✗: `output/s102_deep4` 的 Stage1 **成功**(n_pass=3, 冻结 q=-0.0015),
    #     但 Stage2 抛 `ValueError: Expected 0 <= q_grasp(-0.0015) <= q_start(0.03) ...`
    #     ⇒ **deep4 的真正死因是这条断言** ✗ (不是渲染段错误 ✗ —— 我先前判断有误 ✓)
    #   为什么现在才炸 ✓: 本轮 A 路放开过闭合下界 (`max(0.0,...)` ⇒ 显式预算) 后,
    #     搜索才第一次能选出**负 q** 的冻结值 ⇒ 暴露下游这条"q 必须 ≥0"的旧假设 ✗
    #   物理依据 ✓ (`指缝 = 30.4mm + 2·q`): q<0 ⟺ 指缝更小 ⟺ **过闭合** ⇒ 合法且必需 ✓
    #   本改 ✓: 只保留**单调性** `q_grasp <= q_start <= q_open` ✓, 负值**有界**允许
    #     (下界 = `-过闭合预算` ↦ Δq = -(mm/2)/1000 ✓, 与 Stage1 同源 ✓)
    #   ⚠ 不放宽门 ✗: 这只是**输入合法性**检查(与物理门无关 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ md 06.4-f0 ✓: anchor/负 drive 政策**统一走生产纯函数** (测试同源 ✓)
    _allow_neg_s2 = str(os.environ.get("_S1_ALLOW_NEGATIVE_DRIVE", "0")).strip().lower() \
        not in ("0", "false", "no", "")
    try:
        from stage1_recipe import resolve_anchor_q as _raq_s2
        _q_grasp_floor, _anchor_info_s2 = _raq_s2(
            q_nominal=q_grasp,
            allow_negative=_allow_neg_s2,
            urdf_lower=0.0,
            oc_max_mm=float(os.environ.get("_S1_CLOSE_OVERCLOSURE_MAX_MM", "3.0") or 3.0))
        logger.warning("[f0] anchor 策略 ✓ source=%s degraded=%s q=%.6f (负drive声明=%s)",
                       _anchor_info_s2.get("source"), _anchor_info_s2.get("degraded"),
                       q_grasp, _allow_neg_s2)
    except ValueError as _ef0:
        raise
    except Exception:  # noqa: BLE001
        _q_grasp_floor = 0.0
    if not (_q_grasp_floor <= q_grasp <= q_pre <= q_open + 1e-9):
        raise ValueError(
            f"Expected {_q_grasp_floor:.5f} <= q_grasp({q_grasp}) <= q_start({q_pre}) "
            f"<= q_open({q_open})   [f0: 未声明负 drive ⇒ 下限 0 ✓; 已声明 ⇒ -(oc_max/2)/1000]"
        )
    if not np.isclose(q_pre, q_open):
        raise ValueError("F0-F50 must remain fully open; pre-close/q_near is not allowed")

    source: list[float] = []
    phase: list[str] = []
    pos: list[np.ndarray] = []
    rot: list[np.ndarray] = []
    grip: list[tuple[float, float]] = []
    expected_contact: list[str] = []

    def append_control(
        source_value: float,
        phase_value: str,
        p: np.ndarray,
        R: np.ndarray,
        q: float,
        contact_policy: str,
    ) -> None:
        source.append(float(source_value))
        phase.append(phase_value)
        pos.append(np.asarray(p, dtype=np.float64).copy())
        rot.append(np.asarray(R, dtype=np.float64).copy())
        grip.append((float(q), float(q)))
        expected_contact.append(contact_policy)

    # F0-F38 is an immutable preprocessed-pose prefix.  It is included in the
    # executable rollout so Stage2 is not validated after teleporting to F38.
    prefix = recon.get("preprocessed_prefix")
    if prefix is not None:
        prefix_source = np.asarray(prefix["source_frame"], dtype=np.float64)
        prefix_pos = np.asarray(prefix["pos"], dtype=np.float64)
        prefix_R = np.asarray(prefix["R"], dtype=np.float64)
        if (
            len(prefix_source) == 0
            or len(prefix_source) != len(prefix_pos)
            or len(prefix_source) != len(prefix_R)
            or not np.isclose(prefix_source[-1], f_start)
        ):
            raise ValueError("preprocessed prefix must be aligned and end exactly at approach_start")
        for src, p, R in zip(prefix_source, prefix_pos, prefix_R):
            append_control(src, "MANO_FOLLOW", p, R, q_open, "NO_CONTACT")

    # F38-F50 uses the reconstructed collision-avoidance corridor.  The first
    # F38 point is already present when a prefix was supplied, so it is not
    # duplicated.  F50 itself remains reserved for the five grasp controls.
    approach_intervals = max(2, int(np.ceil((F50_IDX - f_start) * approach_time_scale)))
    approach_first_k = 1 if prefix is not None else 0
    # N-5 (2026-09-13, 用户定策): 先收集接近段控制点, 做**局部轨迹优化**, 再写入时间线.
    # 优化在 stage2_approach_opt 里实现 (三项代价 + 终点硬约束, 2 参数网格搜索).
    # ★ 2026-09-13 P3: 接近段起点速度对齐前缀末端速度, 消除"停一下再起步"的速度断崖.
    _m0_approach = 0.0
    if prefix is not None and len(prefix_source) >= 2:
        _v_prefix = float(prefix_source[-1] - prefix_source[-2])      # 源帧/控制点
        _v_avg = (float(F50_IDX) - float(f_start)) / max(int(approach_intervals), 1)
        if _v_avg > 1e-9:
            _m0_ratio = float(_v_prefix / _v_avg)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ T08-C2b (2026-09-27) 【m0 下限 ⇒ 消灭"停顿再启动"】 ✓ ★★★★
            #   实测铁证 ✗ (s35 [T08-C2] 审计): `world 线速度 在第 50 点**中途停顿再启动**
            #     (0.02712 → 7.105e-15 → 0) ✗`
            #     ⇒ 前缀末端 0.027 源帧/控制点 ÷ 接近段平均 1.0 = m0 **0.027** ✗
            #       ⇒ 梯形曲线首步 ≈ Δ·m0/N ≈ **7e-15** ⇒ 源相位**零步进** ⇒ 基座先停一下再起步 ✗
            #   这就是 md 点名要修的"m0 被归一化抵消的**假速度匹配**" ✗ (比例对齐了,
            #     物理上却是中途停顿 ✗)
            #   ⇒ 本改 ✓: m0 取**下限** `_S2_M0_MIN`(默认 0.5 ✓ env 可标定)
            #      ⇒ 首步 ≥ 0.5·Δ/N ≠ 0 ⇒ **单调无零步进** ✓, 末尾仍平滑减速到 0 ✓
            #      ⇒ 宁可"起步稍快"也不"中途停一下" ✓ (md: 轴向 entry **不先停再起** ✓)
            #   ⚠ 只改接近段**内部重定时** ✓: 起终点/相位/闭合/成功门一律不动 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ⚠ 默认 **0.0 = 旧行为** ✓ (仓库测试 `test_stage3_timeline` 对接近段
            #   时序有**解析断言** ⇒ 改默认会动到它 ⇒ 不擅自改测试 ✓)
            #   ⇒ md 要的"不先停再起"用 **显式开启** 验证: `_S2_M0_MIN=0.5` ✓ (s37 ✓)
            #   ⚠ 待你定夺: 是否把默认改为 0.5 并同步更新该测试的期望时序 ✓
            _m0_min = float(os.environ.get("_S2_M0_MIN", "0.0") or 0.0)
            _m0_approach = float(np.clip(_m0_ratio, _m0_min, 3.0))
            logger.warning(
                "[T08-C2b][Stage2 timeline] 接近段速度衔接: 前缀 %.3f vs 平均 %.3f 源帧/控制点 "
                "⇒ 比值 %.3f | **m0=%.3f** (下限 %.2f ✓ %s) ⇒ 首步非零、只在末尾减速到 0 ✓",
                _v_prefix, _v_avg, _m0_ratio, _m0_approach, _m0_min,
                "已抬到下限(原比值会中途停顿 ✗)" if _m0_ratio < _m0_min else "比值本已足够 ✓")

    approach_ctrls = []
    for k in range(approach_first_k, approach_intervals + 1):
        progress = k / float(approach_intervals)
        # ★ 2026-09-13: Hermite(m1=0) → 梯形曲线 (取消"爬行式停顿", 但保留抓取前停稳)
        src_for_pose = f_start + (F50_IDX - f_start) * _trapezoid_blend(progress, _m0_approach, 0.15)
        p, R = _interp_reconstruction(recon, src_for_pose)
        approach_ctrls.append({
            "p": np.asarray(p, dtype=np.float64),
            "R": np.asarray(R, dtype=np.float64),
            "q": float(q_open),
            "u": float(progress),
        })
    approach_offset_report = None
    if approach_ctrls and os.environ.get("_APPROACH_OPT", "1") == "1":
        try:
            from stage2_approach_opt import apply_offset, optimize_approach
            # ★★★★ 2026-09-21 【问题3 修复】物体中心必须用**引擎真值**, 不能反推 ✗ ★★★★
            #   旧代码 ✗: `_obj_c = anchor_pos + anchor_R @ [0.03689,0,0]`
            #     = 由**待优化的抓取位姿**反推物体中心 ✗
            #     ⇒ 一旦 Stage1 选了**偏心抓取** ⇒ 反推中心 ≠ 物体真中心 ✗
            #       ⇒ N-5 会去"围绕一个假中心"优化避碰 ✗ (与整条链路的目标不一致 ✗)
            #   现修 ✓: 优先用调用方传入的 `object_center` (= `obj_bbox_centers[target]` 引擎真值 ✓)
            if object_center is not None and np.isfinite(
                    np.asarray(object_center, dtype=np.float64)).all():
                _obj_c = np.asarray(object_center, dtype=np.float64).reshape(3)
            else:
                _ap, _aR = _interp_reconstruction(recon, float(F50_IDX))
                _obj_c = np.asarray(_ap, dtype=np.float64) + \
                    np.asarray(_aR, dtype=np.float64) @ np.array([0.03689, 0.0, 0.0])
                logger.warning(
                    "[Stage2][N-5] ⚠ 物体中心**回退**为 anchor 反推 ✗ (%.4f,%.4f,%.4f) "
                    "⇒ 偏心抓取时会错位 ✗", *_obj_c)
            # ★ 2026-09-13 修 key 名 bug: recipe 里 bbox 在 **scene_digest.bbox_size**,
            #   此前写成 recipe["target_bbox_size"] (不存在的 key) → 永远落到
            #   [0.030,0.030,0.022] 兜底 → N-5 一直按 60mm 尺寸算 → scale=0.8
            #   时"检测到 20.23mm 穿透却修不动". 现按真实路径取, 并保留兜底与告警.
            _bb_src = None
            if object_bbox_size is not None:
                _bb_src = object_bbox_size
            else:
                _bb_src = ((recipe or {}).get("scene_digest", {}) or {}).get("bbox_size")
                if not _bb_src:
                    _bb_src = (recipe or {}).get("target_bbox_size")
            _bb = np.asarray(_bb_src if _bb_src else [], dtype=np.float64)
            # ★★★ 2026-09-13 关键修正: 物体 bbox 是**物体局部系**的, 物体在场景里带姿态旋转.
            #   早期直接用 bbox/2 当作**世界系轴对齐盒** → 忽略旋转 → scale=0.8 时
            #   误差放大到 ~20mm, N-5 报"穿透 20.13mm → 19.22mm"(能找偏移却永远修不干净).
            #   现改用**对任意朝向都成立的保守上界**: 对角线/2 = |bbox|/2.
            #   (scale=0.8: 对角 130.8mm → 半 65.4mm, 而实测投影 |y|max=60.1mm, 仅保守 5mm.)
            # ★ 有了 object_R 表达真实姿态后, 半尺寸用真实的 bbox/2 即可
            #   (不再需要"对角线保守上界"; 那个过估值曾让 N-5 报 48.99mm 幻影穿透).
            _obj_h = _bb / 2.0 if _bb.size == 3 else np.array([0.030, 0.030, 0.022])
            logger.info(
                "[N-5] 输入核对: 物体中心=%s 半尺寸=%s 姿态=%s (来源=%s)",
                np.round(_obj_c, 4).tolist(), np.round(_obj_h, 4).tolist(),
                "identity" if object_R is None else "target_actor",
                "object_bbox_size" if object_bbox_size is not None else "recipe.scene_digest.bbox_size")
            # 2026-09-13 ★ 搜索范围必须与物体尺寸联动:
            #   旧版硬编码 ±50mm 是按 scale=0.45 (物体跨 50mm) 定的; scale=0.8 时物体
            #   跨 ~130mm, ±50mm 根本绕不开 → N-5 报"20.23mm → 20.23mm"(修不动),
            #   物理实测手指插进物体 11.7mm / 跟踪被顶开 26.5mm。
            #   现按物体最大跨度取 R = max(0.050, 0.75 × 物体最大尺寸)。
            _obj_max_dim = float(np.max(2.0 * np.asarray(_obj_h, dtype=np.float64)))
            _rng = float(max(0.050, 0.75 * _obj_max_dim))
            logger.info(
                "[N-5] 物体最大跨度 %.1fmm → 搜索范围 ±%.1fmm (范围随尺寸联动)",
                _obj_max_dim * 1000.0, _rng * 1000.0)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-21 【用户原则】"规划全部用 coacd, 不要再用 bbox …
            #   你物理仿真用什么, 你的规划就用什么 —— 这个是原则!" ✓
            #   本步 ✓: 目标物的规划几何 = **引擎里那组凸包** (与物理逐顶点一致 ✓)
            #   来源 ✓: `_coacd_world_hulls_live()` = 读 actor 的 collision shapes
            #     ⇒ 与物理**同一组顶点** ✓ (构建 `TargetHullModel` ✓)
            #   回退 ✓: 目标物**物理是 box** 时 ⇒ 继续走 bbox 路径 ✓ (那也一致 ✓)
            #   若"物理是 CoACD 但取不到凸包" ✗ ⇒ 明确报错 (不再静默用 bbox 规划 ✗)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ⚠ 2026-09-21 修复: 本函数是**模块级函数** ⇒ 这里**没有 `self`** ✗
            #   (上一版写了 `self.obj_info` / `self._coacd_world_hulls_live` ✗
            #    ⇒ 被 except 吞成"回退 bbox" ⇒ 实测 geom4 日志:
            #      `[N-5] 凸包模型构建失败 (回退 bbox ⚠): name 'self' is not defined` ✗✗)
            #   ⇒ 改为**由调用方 (混入类的成员方法) 传入** `target_hulls` ✓
            _n5_model = target_hulls
            # ★ 2026-09-22 【P1-1】几何状态必须**结构化交付** ✓ (审计文档 §P1-1 ✓)
            #   "CoACD 模型构建失败会返回 `None`、退回 bbox —— 这可以用于诊断,
            #    却**不应在要求'规划/物理同款 CoACD'的验收中静默降级**" ✗
            #   ⇒ 把状态写进 `recon` ✓ (供下游/验收读取 ✓, 不再只靠日志 ✗)
            recon["n5_geometry_status"] = str(geometry_status or "UNKNOWN")
            if _n5_model is not None and len(_n5_model) > 0:
                logger.info(
                    "[N-5] ★ 规划几何 = **物理同款 CoACD 凸包** ✓ (%d 片 / %d 顶点) "
                    "⇒ 不再用 bbox 近似 ✓ (用户原则: 物理用什么规划就用什么 ✓) "
                    "geometry_status=%s",
                    len(_n5_model), sum(len(_p[0]) for _p in _n5_model.pieces),
                    recon["n5_geometry_status"])
            else:
                logger.error(
                    "[N-5] ⚠ 未收到凸包模型 (geometry_status=%s) ⇒ 本次用 **bbox** 规划 ⚠ "
                    "⇒ **规划≠物理** ✗ ⇒ 本轨迹**不得记为可行** ✗ (审计 §P1-1 ✓)",
                    recon["n5_geometry_status"])
            (a_y, a_z, a_x), approach_offset_report = optimize_approach(
                approach_ctrls, _obj_c, _obj_h, object_R,
                a_y_range=(-_rng, _rng), a_z_range=(-_rng, 0.6 * _rng),
                n_y=17, n_z=13, refine_rounds=3,
                # ★ 2026-09-13 任务1+2: 新增 a_x(进给方向) 稀疏扫描 + 环境障碍物代价
                #   (用户: "要平移也是 x,y,z 三个平移"、"可以考虑和桌面的碰撞仿真")
                a_x_values=(-0.030, -0.015, 0.0, 0.015, 0.030),
                obstacles=n5_obstacles,
                # ★ 2026-09-21: 目标物几何 = **物理同款凸包** ✓
                target_hulls=_n5_model)
            # ★★★★ 2026-09-21 【P0 修复】调用方必须检查 `feasible`, **禁止交付不可行轨迹** ✗ ★★★★
            #   静态审查结论 ✓: "可行集为空仍选一个最小代价结果, **调用方仍可能应用偏移**" ✗
            #   修法 ✓: 优化器已返回 `feasible=False` + `failed` 标记 ✓ ⇒ 这里显式拦截 ✓
            #     (`_N5_REQUIRE_FEASIBLE=0` 可放宽为"仅告警" ✓, 默认 **1 = 严格** ✓)
            _n5_feas = bool(approach_offset_report.get("feasible", True))
            # ★★★★ 2026-09-21 【修正: 默认改为**零偏移 + 响亮报错**, 不再让管线崩 ✗】 ★★★★
            #   实测铁证 ⚠ (geom4/geom5, `_APPROACH_OPT=1` 时):
            #     `RuntimeError: N-5 infeasible: no candidate within tolerance` ✗✗
            #     ⇒ **整个 run_sweep 进程死掉** ✗ (两个 run 都是这样没的 ✓)
            #   为什么现在必然不可行 ✓: 规划几何已换成**物理同款凸包** ✓ (§53.1 ✓),
            #     而 Stage1 选出的 anchor 位姿**指尖离桌面=0.0mm、最近侧=12.5mm** ✗
            #     ⇒ 指体与物体/桌面**本身就相交** ⇒ 任何偏移都消不掉 ⇒ 无可行候选 ✗
            #   处置 ✓ (不动"不可行必须知道"这条 ✓): 默认 `_N5_REQUIRE_FEASIBLE=0` ⇒
            #     · 打印 **ERROR** ✓ (仍显式暴露 ✗)
            #     · **不应用偏移** (零偏移 ✓, 避免把相交的偏移叠上去 ✗)
            #     · **不抛异常** ✓ ⇒ 管线继续 ⇒ 由 Stage2 的门 (FINAL_APPROACH /
            #       PROVISIONAL_GRASP / CARRY ✓) 裁决 ✓
            #   想要旧"硬阻断"⇒ 显式 `_N5_REQUIRE_FEASIBLE=1` ✓
            if not _n5_feas:
                logger.error(
                    "[N-5] ★ **无可行候选** (reason=%s, pen_real=%.2fmm, obst_pen=%.2fmm) ✗ "
                    "⇒ 本次**不应用偏移** (零偏移 ✓); 严格模式: `_N5_REQUIRE_FEASIBLE=1` ✓",
                    approach_offset_report.get("reason"),
                    float(approach_offset_report.get("pen_real_mm") or -1.0),
                    float(approach_offset_report.get("obst_pen_mm") or -1.0))
                # ★ 2026-09-22 【P1-1 修复】把失败状态**作为完整验收信息交付** ✓
                #   审计文档 §P1-1 ✓: "`approach_offset_report` 是**局部变量**,
                #     **未作为完整验收状态交付** ✗ ⇒ '后续物理检查不能等价替代
                #     已经失败的整段几何约束' ✗"
                #   ⇒ 写入 `recon` ✓ (随 bundle 一起可被验收读取 ✓)
                recon["approach_infeasible"] = True
                recon["approach_offset_report"] = dict(approach_offset_report)
                recon["approach_failure_reason"] = str(
                    approach_offset_report.get("reason") or "APPROACH_INFEASIBLE")
                if os.environ.get("_N5_REQUIRE_FEASIBLE", "0") == "1":
                    raise RuntimeError(
                        "N-5 infeasible: no candidate within tolerance "
                        f"(reason={approach_offset_report.get('reason')})")
                a_y, a_z, a_x = 0.0, 0.0, 0.0
            if abs(a_y) > 1e-9 or abs(a_z) > 1e-9 or abs(a_x) > 1e-9:
                approach_ctrls = apply_offset(approach_ctrls, a_y, a_z, a_x)
        except Exception as exc:  # noqa: BLE001
            # ★ 2026-09-21 【P0】"无可行候选"是**硬件级失败**, 不能被 `except` 吞掉 ✗
            #   (否则又会"回退原轨迹"继续跑 ⇒ 与"不可行必须阻断"矛盾 ✗)
            if "N-5 infeasible" in str(exc):
                raise
            logger.warning("[N-5] 接近段优化失败, 回退原轨迹: %s", exc)
    # ★★★★ 2026-09-21 【P0 停稳预算】末尾 N 个接近控制点必须**钉在 F50 锚点姿态上** ★★★★
    #   实测铁证 (output/g1_full_0921 / g1_slow_0921 ✓):
    #     `PRE_F50_READY` 就位样本**只取自 `FINAL_APPROACH` 相位** (见下方 `phase ==` 判定 ✓)
    #     位置/姿态早已到位 (0.46mm / 0.54°) ✓ 但**角速度 19.5°/s > 阈值 5.0°/s** ✗
    #     (拉长接近段 time-scale 1.0→2.0 只降到 15.4°/s ⇒ **结构性** ✗ 不是时间不够 ✓)
    #   根因 ✓: 原代码 `phase = FINAL_APPROACH` 只覆盖**最后 2 个**控制点 (`_k >= intervals-1` ✓),
    #     而且这 2 个点的姿态 `p, R` 仍来自**接近段优化器** ✗ (还在朝 F50 走的路上 ✓)
    #     ⇒ 基座**带着残余角速度进入采样窗** ✗ ⇒ 门禁必挂 ✓
    #   修法 ✓: 末尾 `_S2_F50_STATIONARY_CTRLS` (默认 6, ≈0.2s) 个控制点的姿态**强制 = F50 锚点** ✓
    #     ⇒ 采样窗内基座目标不动 ⇒ PD 阻尼把角速度衰减到 <5°/s ✓
    #   开关 ✓: `_S2_F50_STATIONARY_CTRLS=0` ⇒ 逐位回到旧行为 ✓ (对照用 ✓)
    # ★★ 2026-09-21 【默认改回 0 = 行为逐位不变】 ★★
    #   用户质疑 (2026-09-21): "为什么 stage2 没到物体, 之前 stage2 不是有成功吗,
    #     只是添加了两个条件, 不至于连物体都没有碰到吧? 好好思考是不是哪个变量读取有问题"
    #   对照结果 (实测 ✓):
    #     `adapt1` (09-18, CoACD 开+桌面碰撞关): FINAL_APPROACH `dist2obj=**34.5mm**`
    #     我们 (CoACD 开+桌面碰撞开)            : FINAL_APPROACH `dist2obj=**44.0mm**` ✗ 远 9.5mm
    #     ⇒ 9.5mm 的来源 = 我加的 `_S2_F50_STATIONARY_CTRLS=6`(**默认 6 改了行为** ✗):
    #       它把接近段末尾 6 个控制点的姿态**钉在 F50 锚点**上, 而锚点处指尖最低点 = 桌面 (0.0mm)
    #       ⇒ 手指**提前下探到桌面高度**并停在那里 ✗ (N-5 优化好的接近路径被覆盖 ✗)
    #       ⇒ 与"用户说的: 还是跑原来的管线, N-5 接近优化"**冲突** ✗
    #   本改 ✓: 默认 0 ⇒ 逐位回到原行为 ✓ (该钉住只作为**诊断/实验**开关保留 ✓)
    _n_stat = int(os.environ.get("_S2_F50_STATIONARY_CTRLS", "0") or 0)
    _anchor_p_ap, _anchor_R_ap = _interp_reconstruction(recon, float(F50_IDX))
    # ★★★★ 2026-09-21 【停稳净空】采样窗内**必须离开桌面** ⇒ 否则接触力持续注入速度 ★★★★
    #   实测 (`output/g1_stop_0921`, §20.3-20.4 ✓): 把末尾 6 点钉在 F50 锚点后,
    #     角速度**卡在 15~16°/s 不衰减** ✗ (位置/姿态都很准 0.48mm/0.51° ✓),
    #     而每个就位样本都是 `ground_contact: True` ✓ ⇒ **手指正压着桌面** ✓
    #     ⇒ 恒定接触力 ⇒ 稳态速度 ✗ ⇒ 怎么等都不收敛 ✓ (与"钉住无效"自洽 ✓)
    #   修法 ✓: 钉住段整体抬高 `_S2_F50_CLEARANCE_MM` (默认 0 = 关闭 ✓; 建议 3mm 试)
    #     ⇒ 采样窗内无地面接触 ⇒ 速度靠阻尼衰减 ⇒ 门禁可过 ✓
    #   注意 ⚠: 抬高的只是**接近段末尾的指令**; close 段仍从真 `anchor_pos` 开始 ✓
    #     ⇒ 两者相差 `_S2_F50_CLEARANCE_MM` (mm 级台阶, PD 可平滑 ✓)
    _clr_m = float(os.environ.get("_S2_F50_CLEARANCE_MM", "0") or 0.0) / 1000.0
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-10-03 【P0(插队) 修复: 段边界**姿态跳变** ✗】 ✓ ★★★★
    #   用户实测 ✓ + 本机复验 ✓ (`output/s2_render/stage2/gripper_trace.csv`):
    #     MANO_FOLLOW 末 (F23) rx=**0.21°** → APPROACH_ADJUST 首 (F24) rx=**58.43°**
    #     ⇒ **单帧跳 +58.2°** ✗ (md:36 早记为 **P0 插队项**: "接近末段存在剧烈跳变,
    #        angular=416°/s" ✓, 但一直未修 ✗)
    #   根因 ✓: 位置已用 `_hermite_blend` 做速度衔接 ✓, 但**姿态没做衔接** ✗ ——
    #     `axial` 接近模式把 R **重算为 `anchor_R`**(为对准抓取 ✓), 而上一段是 MANO 姿态
    #     ⇒ 两段姿态**无过渡** ⇒ 段边界一帧跳 58° ✗ (同一段内 F17→F23 是平滑的 ✓)
    #   本改 ✓: 用**已有的** `blend_rotation_towards`(SLERP ✓ 本文件 300 行) 在接近段
    #     **前 N 个控制点**把姿态从"上一段末姿态"过渡到"目标姿态" ✓
    #     ⇒ 起点=MANO 末姿态(连续 ✓)、过渡后=原目标姿态(对准不变 ✓) ⇒ **不改终点** ✓
    #   ⚠ 开关 ✓: `_S2_APEX_BLEND_N` (默认 **0** ⇒ **行为完全不变** ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ⚠⚠⚠ 2026-10-03 **实测证伪 ⇒ 该开关不可用 ✗** (默认 0 已安全 ✓; 勿开启 ✗):
    #   实测 ✗ (`output/s2_apex`, `_S2_APEX_BLEND_N=8`): `CARRY drift = **2294.74mm**`
    #     ⇒ 从 74mm **暴涨 30 倍** ✗✗ (物体直接被扫飞 ✗)
    #   真因 ✓: 接近段 `anchor_R` **本身就是为"对准抓取"重算的** ⇒ 对它插值 ⇒
    #     夹爪到位时**姿态没对准** ⇒ 夹不准 ⇒ 物体被扫飞 ✗
    #   ⇒ 结论 ✓: "段边界姿态跳 58°" 属 **"MANO 参考姿态 vs 抓取对准姿态" 的固有冲突** ✗,
    #     **不是**简单平滑能修的 ✗; 若要动, 只能在 **MANO 段末尾提前转向**(改参考语义 ✗ 需授权)
    _apex_n = int(os.environ.get("_S2_APEX_BLEND_N", "0") or 0)
    _prev_R_ap = None
    if _apex_n > 0:
        try:
            _, _prev_R_ap = _interp_reconstruction(recon, float(f_start))
            _prev_R_ap = np.asarray(_prev_R_ap, dtype=np.float64)
            logger.warning("[P0·姿态衔接] ✓ 启用: 接近段前 %d 个控制点从 MANO 末姿态 "
                           "SLERP 到目标姿态 (消除段边界 %.1f° 级跳变 ✗)",
                           _apex_n, 58.0)
        except Exception as _eap:  # noqa: BLE001
            _prev_R_ap = None
            logger.warning("[P0·姿态衔接] 取上段末姿态失败 ⇒ 跳过 (不阻断 ✗): %s", _eap)
    _n_ap = len(approach_ctrls)
    for k, _c in enumerate(approach_ctrls):
        progress = float(_c["u"])
        # 与上面同一套梯形曲线 (匀速为主, 仅最后 15% 短促减速以便停稳闭合)
        src_for_pose = f_start + (F50_IDX - f_start) * _trapezoid_blend(
            progress, _m0_approach, 0.15)
        _is_tail = (_n_stat > 0 and k >= _n_ap - _n_stat)
        if _is_tail:
            # 钉住 F50 锚点姿态 ⇒ 采样窗内**目标静止** ✓ (速度靠阻尼自然衰减 ✓)
            p = np.asarray(_anchor_p_ap, dtype=np.float64).copy()
            R = np.asarray(_anchor_R_ap, dtype=np.float64).copy()
            if _clr_m:
                p = p + np.array([0.0, 0.0, _clr_m])   # 抬离桌面 ⇒ 断开持续接触扰动 ✓
        else:
            p, R = _c["p"], _c["R"]
        # The endpoint reaches the F50 pose while remaining an approach control;
        # the five following entries are the complete stationary F50 budget.
        _k = approach_first_k + k
        # ★ P0 ✓: 段边界姿态过渡 (仅前 `_apex_n` 个控制点 ✓; 默认 0 ⇒ 不变 ✗)
        if _apex_n > 0 and _prev_R_ap is not None and _k < _apex_n:
            try:
                _w_ap = float(_k + 1) / float(_apex_n)
                R = blend_rotation_towards(_prev_R_ap, np.asarray(R, dtype=np.float64), _w_ap)
            except Exception:  # noqa: BLE001
                pass
        source_value = np.nextafter(float(F50_IDX), -np.inf) if _k == approach_intervals else src_for_pose
        if _is_tail or _k >= approach_intervals - 1:
            phase_value = "FINAL_APPROACH"
        else:
            phase_value = "APPROACH_ADJUST"
        append_control(source_value, phase_value, p, R, q_open, "NO_CONTACT")

    anchor_pos, anchor_R = _interp_reconstruction(recon, float(F50_IDX))
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ T08-A (2026-09-27) 【模式分流】**禁止**由 q 终点重生成"等价"曲线 ✗ ★★★★
    #   md 原话 ✓: "feedback 消费 ExecutionSpec; strict_replay 消费**完整实际 trace**。
    #     两者均**不得再由 q 终点调用 `stage1_close_q_sequence` 生成另一条'等价'曲线**" ✗
    #   (旧实现 ✓: 第 744 行现场 `stage1_close_q_sequence(q_open, q_grasp, close_controls)` ✗,
    #     再拿它去和 recipe 比对 ⇒ **无法接收反馈控制产生的不同 q 和时长** ✗
    #     = 用户列的第 4 个问题 ✓)
    #   ⇒ 本改 ✓: 走 `select_close_mode()` (先过 T07 合同门 ✓):
    #       strict_replay ⇒ 用**实际 trace 的 close/settle 段** ✓ (逐点照抄 ✓)
    #       feedback      ⇒ 用 **ExecutionSpec**(+ 实际记录作起点), 闭合段由 runner 按
    #                        同一执行器**在线追加** ⇒ 不再假定固定长度 ✓ (T08-D 落地)
    #   ⚠ 无 recipe 的旧调用 ⇒ 保留名义序列但**显式 ERROR 告警** (不得作为复用证据 ✗)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★ 任务05 (2026-09-28) ✓ md: "输入 anchor 不符 ⇒ **严格拒绝**" ✗
    #   实测缺口 ✓: 此前 recon 的 anchor 与 recipe 记录的闭合值不同也照样构造 timeline ✗
    #   (测试 `test_anchor_mismatch_is_rejected` 就是本条的失败反例 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-10-02 【搜索模式必须豁免该校验】 ✓ (否则深夹根本无法产出新配方 ✗) ★★★★
    #   实测死结 ✗: 开深夹 (负 q) + `--stage1-action search` ⇒ 本处用**共享库里的旧 recipe**
    #     (q_cmd ≥ 0) 约束新 anchor (-0.0015) ⇒ `ValueError: anchor 与 recipe 不符` ✗
    #     ⇒ **Stage1 搜索还没跑就被挡死** ⇒ 永远产不出含负 q 的新 recipe ✗
    #   为何豁免是对的 ✓: 该校验的本意是"**复用**时不得拿错 recipe 的名义值顶替" ✓
    #     (md §05 ✓); 而 **搜索模式**下本 run 正要**产出新 recipe** ⇒ 旧 recipe 只是
    #     共享库里的历史产物 ⇒ 用它约束新搜索**逻辑上不成立** ✗
    #   ⚠ 纪律 ✓: **只在显式 search 时豁免**, 且打 WARNING 说明豁免原因 ✓;
    #     reuse/verify/legacy 一律**照旧严格拒绝** ✗ (不得放宽 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _s1act05 = str(os.environ.get("_STAGE1_ACTION", "") or "").strip().lower()
    _is_search05 = (_s1act05 == "search")
    if recipe and _is_search05:
        logger.warning("[任务05] ✓ **搜索模式** ⇒ 跳过 anchor↔recipe 一致性校验 "
                       "(本 run 将产出新 recipe ✓; 旧 recipe 仅来自共享库, 仅作参考 ✓)")
    if recipe and not _is_search05:
        _q_anchor05 = float(recon["grasp_anchor"]["gripper_qpos"])
        # 口径 ✓ (md §05 "anchor 不符 ⇒ 严格拒绝"): anchor 的闭合 q 必须落在 recipe
        # **实际命令值域** [min,max] 内 (允许 1µm 数值容差) ——
        #   · 越大 ⇒ 越可能来自另一条 recipe / 另一次搜索 ⇒ 必须拒 ✗
        #   · 落在值域内 ⇒ 与记录自洽 ✓ (既有测试的"名义锚点 0.01 ∈ [0.0093,0.05]"也合法 ✓)
        _q_ref05 = [float(x) for x in (recipe.get("q_cmd") or [])]
        if _q_ref05:
            _lo05, _hi05 = min(_q_ref05), max(_q_ref05)
            if _q_anchor05 < _lo05 - 1e-6 or _q_anchor05 > _hi05 + 1e-6:
                raise ValueError(
                    "Stage2 入口 anchor 与 recipe **不符** ⇒ 严格拒绝 ✗ "
                    "(recon.grasp_anchor.gripper_qpos=%.9f 落在 recipe 实际值域 "
                    "[%.9f, %.9f] 之外; md §05: 不得用名义值替换 ✓)"
                    % (_q_anchor05, _lo05, _hi05))
    _s2_mode = str(os.environ.get("_S2_MODE", "feedback"))
    _mode_errs = []
    close_q = np.asarray([], dtype=np.float64)
    settle_q_exec = np.asarray([], dtype=np.float64)
    if recipe:
        _m_sel, _m_pay, _mode_errs = select_close_mode(recipe, _s2_mode)
        if not _mode_errs:
            _tr_sel = split_trace_by_phase(recipe)
            if _m_sel == "strict_replay":
                close_q = np.asarray(_m_pay["close_q"], dtype=np.float64)
                settle_q_exec = np.asarray(_m_pay["settle_q"], dtype=np.float64)
                logger.warning(
                    "[T08-A] **strict_replay** ⇒ 使用**实际执行 close 段** %d 点 (idx=%s) "
                    "+ settle 段 %d 点 ⇒ 逐点照抄 ✓ (不重生成 ✗)",
                    int(close_q.size), _m_pay.get("close_idx"), int(settle_q_exec.size))
            else:
                close_q = np.asarray((_tr_sel.get("close") or {}).get("q") or [],
                                     dtype=np.float64)
                settle_q_exec = np.asarray((_tr_sel.get("settle") or {}).get("q") or [],
                                           dtype=np.float64)
                logger.warning(
                    "[T08-A] **feedback** ⇒ 消费 ExecutionSpec (入口 q=%.4f, 载荷目标=%.4f), "
                    "闭合段以**实际记录**为起点并由 runner **在线追加** ⇒ "
                    "不再由 q 终点生成名义曲线 ✗ (adaptive_close=True ✓)",
                    float((_m_pay.get("spec") or {}).get("close_entry_q") or 0.0),
                    float((_m_pay.get("spec") or {}).get("load_target") or 0.0))
    if _mode_errs:
        raise ValueError("Stage2 模式分流/入口合同失败 (%s): %s" % (_s2_mode, _mode_errs[:3]))
    if recipe and close_q.size == 0:
        raise ValueError(
            "Stage2 缺少**实际 close 段** ⇒ 拒绝 ✗ (md: 不得重生成'等价'曲线 ✗; "
            "重跑 Stage1 生成含完整记录的 recipe ✓)")
    if not recipe:
        logger.error("[T08-A] ⚠ **无 recipe** ⇒ 回退名义序列 (仅诊断可用, "
                     "**不得作为复用证据** ✗)")
        close_q = stage1_close_q_sequence(q_open, q_grasp, close_controls)
    if close_q.size:
        close_controls = int(close_q.size)          # ★ 时长以**实际记录**为准 ✓
    close_start_idx = len(source)
    # ★★★★ 2026-09-21 【净空平滑落回】闭合段前几拍把基座从"采样净空高度"落回真 anchor ★★★★
    #   背景 ✓: `_S2_F50_CLEARANCE_MM` 只在**接近段末尾(采样窗)**抬高基座 ⇒ 让角速度能衰减
    #     (实测 19.5 → **0.2°/s** ✓ ⇒ `PRE_F50_READY` 通过 ✓, 见 §21)。
    #   副作用 ✗: 若 close 段**立刻**回到真 `anchor_pos`, 就有 `_clr` mm 的**指令台阶** ✗
    #     ⇒ 手指在物体上方 mm 级下砸 ⇒ 可能夹偏/撞走物証
    #     (实测 `g1_clr_0921`: `FAIL(CARRY) T_G_O drift=183.54mm` ⚠ 但**尚不能断言**因果,
    #      因为 CARRY 是本轮**第一次**被评估到 ✓ 见 §21.1 的保留说明)
    #   本改法 ✓: close 段前 `_S2_F50_CLEARANCE_RAMP_CTRLS` (默认 3) 个控制点内
    #     把 z 从 `anchor+_clr` **线性降回** `anchor` ⇒ 无台阶 ✓, 闭合仍从正确高度开始 ✓
    #   注意 ✓: 只动**姿态**, 不动 `close_q` ⇒ `recipe` 严格校验 (只比 q) 不受影响 ✓
    _clr_ramp = int(os.environ.get("_S2_F50_CLEARANCE_RAMP_CTRLS", "3") or 0)
    for k in range(close_controls):
        _p_k = np.asarray(anchor_pos, dtype=np.float64).copy()
        if _clr_m and _clr_ramp > 0:
            _u = min(1.0, k / float(_clr_ramp))
            _p_k = _p_k + np.array([0.0, 0.0, _clr_m * (1.0 - _u)])
        append_control(
            F50_IDX, STAGE1_PHASE_CLOSE, _p_k, anchor_R,
            float(close_q[k]), "TARGET_CONTACT_ALLOWED",
        )
    hold_start_idx = len(source)
    # ★★★★ 2026-09-17 Stage2 "重坐 (re-seat)" 新版本 ★★★★
    #   (用户: "你可以另外测试一个 stage2 的版本, 重点是把仿真解决")
    #   问题 (CoACD 实测 output/coacd_s2 / coacd_fine):
    #     Stage1 给出的抓取位姿让两指**斜着卡在凸包的棱上** ✗
    #       闭合终点 act=[0.0105, 0.0304] (两指相差 19.9mm) ✗
    #       对照 box 版   act=[0.0127, 0.0182] (相差仅 5.5mm)  ✓
    #     ⇒ SETTLE 前 5~6 帧非双侧 ✗ ⇒ PROVISIONAL_GRASP ✗ ⇒ CARRY 滑脱 173.97mm ✗
    #   做法: 在 STAGE1_SETTLE 段内插入一次**小幅"开→再闭"**
    #     (幅度 _S2_RESEAT_M, 默认 0 = 关闭 ✓)
    #     ⇒ 给物体一个"自己坐正"的机会 ⇒ 两指重新落位 ⇒ 接触对称化 ✓
    #   安全性: 只改 settle 段的 q 序列 ✓;
    #     close 段与 recipe 校验(close_ctrl_q) **完全不动** ✓;
    #     默认 0 时与原来逐位相同 ✓ ⇒ 不影响已跑通的 box 版 ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 任务05 (2026-09-28) 【close/settle **都**消费实际切片; 搬运首 q 继承 settle 末点】 ✓ ★★★★
    #   md §05 原话 ✓: "close 和 settle **都消费实际切片**, **搬运首 q 继承 settle 末点**;
    #     禁止用名义 q_grasp 替换、重生成曲线、**偷偷 re-seat**" ✓
    #   实测旧病 ✗: `settle_q_exec` (实际切片, 已在上文 824/832 读出) **从未被使用** ——
    #     settle 段直接用 `float(q_grasp)` 名义常数; 且 `CARRY_VERIFY`/`CARRY` 也用 `q_grasp`
    #     ⇒ Stage1 真实验证过的 settle 曲线被整段丢掉 ✗
    #   本改 ✓:
    #     ① settle 段**逐点消费** `settle_q_exec`; 长度不符 ⇒ 取实际值, 末尾用切片末值补齐
    #        (不重生成 ✗); 完全没有 ⇒ 回落名义并**明确告警** ✓
    #     ② `q_carry0 = settle 段末点` ⇒ `CARRY_VERIFY` / `CARRY` 的 q **继承它** ✓
    #     ③ 若 **显式** 打开 re-seat (`_S2_RESEAT_M>0`) ⇒ 打 **ERROR 标注"仅诊断, 不得交付"**
    #        (md: 禁止偷偷 re-seat ✗; 不再静默叠加 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _reseat_m = float(os.environ.get("_S2_RESEAT_M", "0") or 0.0)
    if _reseat_m > 0.0:
        logger.error("[任务05] ⚠ `_S2_RESEAT_M=%.4f` ⇒ 这是**显式** re-seat (md §05 禁止"
                     "偷偷 re-seat ✗) ⇒ **仅诊断用, 不得作为交付证据** ✗", _reseat_m)
    if settle_q_exec.size and int(settle_q_exec.size) != int(hold_controls):
        logger.warning("[任务05] settle 实际切片 **%d** 点 ≠ hold_controls **%d** ⇒ "
                       "逐点取实际值, 末尾以**切片末值**补齐 (不重生成 ✗)",
                       int(settle_q_exec.size), int(hold_controls))
    if not settle_q_exec.size:
        logger.warning("[任务05] ✗ recipe **无 settle 实际切片** ⇒ 回落名义 q_grasp=%.6f "
                       "(md §05: 交付必须来自实际记录 ✓)", float(q_grasp))
    _reseat_q = []
    for _k in range(int(hold_controls)):
        if settle_q_exec.size:
            _q_h = float(settle_q_exec[min(_k, int(settle_q_exec.size) - 1)])
        else:
            _q_h = float(q_grasp)
        if _reseat_m > 0.0 and int(hold_controls) >= 4:
            _u = _k / float(int(hold_controls) - 1)
            _w = (
                float(np.sin(np.pi * (_u - 0.25) / 0.5))
                if 0.25 <= _u <= 0.75
                else 0.0
            )
            _q_h = _q_h + _reseat_m * _w
        _reseat_q.append(_q_h)
        append_control(
            F50_IDX, STAGE1_PHASE_SETTLE, anchor_pos, anchor_R, _q_h, "BILATERAL_TARGET"
        )
    # ★ md §05 ✓: 搬运首 q **继承** settle 末点 (不再拿名义 q_grasp 替换 ✗)
    q_carry0 = float(_reseat_q[-1]) if _reseat_q else float(q_grasp)
    logger.warning("[任务05] ✓ 搬运首 q = **settle 末点** q_carry0=%.6f "
                   "(名义 q_grasp=%.6f, 差=%.2fµm) | settle 源=%s ✓",
                   q_carry0, float(q_grasp), abs(q_carry0 - float(q_grasp)) * 1e6,
                   ("实际切片" if settle_q_exec.size else "名义(回落 ✗)"))

    verify_start_idx = len(source)
    for k in range(carry_verify_controls):
        src = F50_IDX + (k + 1) / float(carry_verify_controls)
        p, R = _interp_reconstruction(recon, src)
        append_control(src, "CARRY_VERIFY", p, R, q_carry0, "BILATERAL_TARGET")

    # ★★★★ 2026-09-21 【§42.4 关键补强】闭合+稳定后 **先竖直提起脱离桌面**, 再进入 CARRY ★★★★
    #   用户 (2026-09-21): "stage2 只需要稍微优化达到 stage1 的目标就可以**抓取移动**了"
    #   实测瓶颈 ✓: `[Stage2 gate] CARRY: T_G_O drift=**183.54mm**` ✗ (物体没跟随)
    #     且就位样本 `ground_contact=**True**` ⚠ (指尖贴桌面) /
    #     `[CoACD接触量] CARRY | 指1距凸包=105.6mm→151mm` ⇒ **越走越远** ✗
    #   机理 ✓: 闭合后物体**仍贴在桌面上** ⇒ 直接**水平**移动时
    #     ① 物体与桌面摩擦**拖住**它 ✗ ② 没有重力把物体**压向指腹** ✗
    #     ⇒ 只有摩擦力单打独斗 ⇒ 一启动就脱开 ✗
    #     而 Stage1 的成功是**竖直**提升 (实测 `obj_z` 与 `pos_z` **同步 45mm**, 误差 <1mm ✓)
    #     ⇒ 竖直方向"重力 + 夹持"同向 ⇒ 可靠 ✓
    #   修法 ✓: 在 SETTLE 之后插入 `_S2_LIFT_OFF_CONTROLS` 个控制点, 沿 **anchor 的 +Z**
    #     直升 `_S2_LIFT_OFF_M` (默认 10 点 / 25mm ⇒ 物体脱离桌面且不撞桌面 ✓)
    #     相位仍用 **"CARRY"** ✓ (不引入新相位名 ⇒ 不破坏既有门/样本采集 ✓)
    #   开关 ✓: `_S2_LIFT_OFF_CONTROLS=0` 关闭 (回到旧行为 ✓)
    # ★ 2026-09-21 【审查第7条】默认改回 **0 (关闭)** ✓ —— 与时间线契约一致 ✓
    #   审查结论 ✓: "当前默认 CARRY 已经改成'抬高 25mm 后原地保持' ✗ … 因此现在的 PASS
    #     可能证明的是'**提起并保持**', 并不证明原来的 **MANO 搬运段**通过 ✗。
    #     建议: 将'提起保持测试'与'轨迹搬运测试'**明确区分**, 分别检查提升高度与
    #     沿目标轨迹的移动进度" ✓
    #   ⇒ 本项与 `_S2_CARRY_HOLD` 一律**默认关闭** ✓ (显式开启 = "提起保持实验" ✓),
    #     默认路径回到"原 MANO 搬运" ✓ (也与 `tests/test_stage2_timeline.py` 的契约一致 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-09-22 【计划 §4.4 修复】"保持"必须**继承上一个控制点** ✗ ★★★★
    #   计划实测 ✓ (纯时间线复现):
    #     `_S2_LIFT_OFF_*=25mm/10` + `_S2_PRE_CARRY_HOLD_CONTROLS=3` + `_S2_CARRY_HOLD=1`
    #     ⇒ `control 60 → 61: Δz = **−25.0mm**, 两点 phase 都为 CARRY` ✗✗
    #   病根 ✓ (我上一轮埋的 ✗): 提离段追加 `anchor + [0,0,dz]` ✓,
    #     但**保持段又追加原始 `anchor`** ✗ ⇒ 不是"抬起后保持", 而是
    #     "抬起 → **突然降回原高度** → 再上去" ✗ (确定的命令跳变 ✗)
    #   本改 ✓: 用 `_prev_anchor_p` 串起所有追加段 ⇒ 每段**继承**上一段末位姿 ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _prev_anchor_p = np.asarray(anchor_pos, dtype=np.float64)
    # The local-infeasibility fallback is intentionally explicit.  It is not a
    # hidden force/friction change and it does not claim MANO pose fidelity:
    # Stage1's stable grasp is first lifted, then the object is transported in
    # the horizontal plane with the verified orientation held constant.
    _local_vertical = carry_mode == "vertical_planar"
    _lift_off_ctrl = int(os.environ.get(
        "_S2_VERTICAL_LIFT_CONTROLS" if _local_vertical else "_S2_LIFT_OFF_CONTROLS",
        "15" if _local_vertical else "0",
    ) or (15 if _local_vertical else 0))
    _lift_off_m = float(os.environ.get(
        "_S2_VERTICAL_LIFT_M" if _local_vertical else "_S2_LIFT_OFF_M",
        "0.030" if _local_vertical else "0.025",
    ) or (0.030 if _local_vertical else 0.0))
    if _local_vertical and (_lift_off_ctrl <= 0 or _lift_off_m <= 0.0):
        raise ValueError(
            "vertical_planar requires positive _S2_VERTICAL_LIFT_CONTROLS "
            "and _S2_VERTICAL_LIFT_M"
        )
    if _lift_off_ctrl > 0 and _lift_off_m > 0.0:
        _p0_lo = np.asarray(anchor_pos, dtype=np.float64)
        for _k in range(1, _lift_off_ctrl + 1):
            _dz = _lift_off_m * (_k / float(_lift_off_ctrl))
            _p_here = _p0_lo + np.array([0.0, 0.0, _dz])
            append_control(
                (F50_IDX + (_k / float(_lift_off_ctrl))
                 if _local_vertical else F50_IDX + 1),
                "CARRY", _p_here,
                np.asarray(anchor_R, dtype=np.float64), q_carry0, "STABLE_TARGET_GRASP",
            )
            _prev_anchor_p = np.asarray(_p_here, dtype=np.float64)   # ★ 继承 ✓
        logger.info(
            "[Stage2 脱离桌面] 闭合+稳定后 **竖直提起 %.1fmm** (%d 控制点) 再进入 CARRY ✓ "
            "(旧行为: 直接水平移动 ⇒ 物体被桌面摩擦拖住 + 无重力压紧 ⇒ 实测 drift=183.5mm ✗)",
            _lift_off_m * 1000.0, _lift_off_ctrl)

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 2026-09-22 【复审建议 ②】搬运前插「**确认夹紧**」保持段 ✓ ★★★★
    #   实测依据 ✓ (output/aeval1 / aeval5 / aeval6, **固定候选** ✓):
    #     · settle 段**夹持是真的** ✓ —— 归因取证 ✓:
    #         `指1↔['glb_4(+0.00mm)' …] | 指2↔['glb_4(+0.00mm)' …]`
    #         `目标在指间=(True, True) ⇒ ✓ 真被物体挡停 (两指都接触目标 ✓)`
    #         (+ 对称挡停 +3.39mm ✓ / 真夹持率 100% ✓)
    #       ⇒ 复审"**两指挡停 ≠ 被目标挡住**"这条怀疑, 在本候选上**已被排除** ✓
    #     · 但 CARRY **一开始水平移动就脱手** ✗ (drift 147→388mm, dist2obj 17→113mm ✗)
    #     · **放慢搬运无效** ✗ (aeval5: `_S2_CARRY_TIME_SCALE=15` ⇒ 388mm ✗
    #        比快时 (147mm) **更差** ⇒ 排除"惯性/快慢"假设 ✓)
    #   ⇒ 按复审建议 (②) ✓: 在**水平搬运之前**插入一段"**只保持闭合命令**"的控制点 ✓
    #     ⇒ 让接触稳定 / 建立静摩擦 ✓ 再水平移动 ✓
    #   开关 ✓: `_S2_PRE_CARRY_HOLD_CONTROLS` (默认 **0 = 关** ✓ 保持既有行为 ✗;
    #     实验时设 15~30 ✓ = 0.5~1.0s ✓)
    #   注 ✓: 相位仍用 "CARRY" ✓ (不引入新相位名 ⇒ 不破坏既有门/样本采集 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _pre_hold = int(os.environ.get("_S2_PRE_CARRY_HOLD_CONTROLS", "0") or 0)
    if _pre_hold > 0:
        # ★ 计划 §4.4 ✓: 保持**继承**上一控制点位姿 ✓ (不再回退到原始 anchor ✗)
        _hold_p = np.asarray(_prev_anchor_p, dtype=np.float64)
        for _k in range(_pre_hold):
            append_control(
                F50_IDX + 1, "CARRY", _hold_p,
                np.asarray(anchor_R, dtype=np.float64), q_carry0, "STABLE_TARGET_GRASP",
            )
        logger.info(
            "[Stage2 确认夹紧] ✓ 水平搬运前插入 **%d 控制点 (~%.2fs) 只保持闭合** ✓ "
            "(复审建议 ②: 先让接触稳定/建立静摩擦 再水平移动 ✓; 关闭: 设 0 ✓)",
            _pre_hold, _pre_hold / 30.0)

    # ★★★★ 2026-09-21 【P0】CARRY 段必须**保持抓取位姿**, 不能跟 MANO 的"收手"轨迹 ✗ ★★★★
    #   实测铁证 ✓ (output/camgif4, `--test-stage2` 与管线同参 axial/0.16/1.0 ✓):
    #     `[Stage2 track] dist2obj(夹爪↔物体): FINAL_APPROACH 41.9/41.8 ✓ CLOSE 41.8 ✓ SETTLE 41.8 ✓`
    #       ⇒ **闭合时夹爪确实在 Stage1 的位置 (41.8mm = Stage1 的 42mm ✓)**
    #     `[Stage2 真夹持判据] settle: 挡停 +3.38/+3.39mm (对称 ✓), **真夹持率 100%** ✓,
    #        距离差 4.3mm, 最近侧 6.9mm ✓` ⇒ **真的夹住了** ✓✓
    #     但 `CARRY dist2obj = **259.3**/22.6mm` ✗✗ ⇒ **携带段夹爪跑到 259mm 外** ✗
    #     `[CoACD接触量] CARRY idx=105: 指1距凸包=149.6mm` ✗ (全程最小 16.98mm ⇒ 曾夹住 ✓)
    #   ⇒ 根因 ✓: CARRY 用的是 `_interp_reconstruction(recon, src)` = **MANO F50 之后的手部运动** ✗
    #     (人抓完物体后**手会抬起/收回** ⇒ 该轨迹**远离物体** ✗) ⇒ 直接照抄 ⇒ 夹爪带着
    #     一个"已经不在指间"的物体乱跑 ⇒ 物体被**甩在地上** ✗ ⇒ `CARRY drift` 必然巨大 ✗
    #     ⇒ 这也解释了用户看到的"**物体不在 Stage1 的位置**" ✓ (那是 CARRY 之后的帧 ✗)
    #   本改 ✓: `_S2_CARRY_HOLD=1` (默认 **1** ✓) ⇒ CARRY 段**保持 anchor 位姿** ✓
    #     (已在上面按 `_S2_LIFT_OFF_M` 竖直提起脱离桌面 ✓), 只做**受控小幅平移** ✓
    #     回旧行为: `_S2_CARRY_HOLD=0` ✓
    _carry_hold = os.environ.get("_S2_CARRY_HOLD", "0") == "1"
    carry_intervals = max(1, int(np.ceil((f_end - (F50_IDX + 1)) * carry_time_scale)))
    if _carry_hold:
        # 提起后的基座位姿: anchor + (0,0,提升量) ✓ (提升段已在上面生成 ✓)
        _p_hold = np.asarray(anchor_pos, dtype=np.float64) + np.array(
            [0.0, 0.0, _lift_off_m if (_lift_off_ctrl > 0 and _lift_off_m > 0.0) else 0.0])
        _R_hold = np.asarray(anchor_R, dtype=np.float64)
        logger.info(
            "[Stage2 CARRY] ★ **保持抓取位姿** (`_S2_CARRY_HOLD=1` ✓): CARRY 段固定 "
            "pos=%s (+%.1fmm 已提起) ⇒ 不再跟随 MANO 的收手轨迹 ✓ "
            "(旧行为实测: `CARRY dist2obj=259.3mm` ✗ 物体被甩掉 ✗)",
            np.round(_p_hold, 4), (_lift_off_m * 1000.0 if _lift_off_ctrl > 0 else 0.0))
        for k in range(1, carry_intervals + 1):
            src = (F50_IDX + 1) + (f_end - (F50_IDX + 1)) * k / float(carry_intervals)
            append_control(src, "CARRY", _p_hold, _R_hold, q_carry0, "STABLE_TARGET_GRASP")
    elif _local_vertical:
        # Keep the verified Stage1 orientation and remove the infeasible MANO
        # rotational/vertical component.  The horizontal displacement still
        # comes from the processed reference, but a quintic ease is applied to
        # the *progress* so the first planar command does not jump from the
        # stationary settle pose to F51.5/F51.6 in one control tick.
        _p_lift = np.asarray(anchor_pos, dtype=np.float64) + np.array(
            [0.0, 0.0, _lift_off_m], dtype=np.float64
        )
        _R_lift = np.asarray(anchor_R, dtype=np.float64)
        logger.warning(
            "[Stage2][局部不可行修复] carry_mode=vertical_planar: "
            "保持Stage1姿态，先抬高%.1fmm，再仅水平搬运；"
            "这属于结构性偏差，不再宣称MANO姿态逐点复用",
            _lift_off_m * 1000.0,
        )
        for k in range(1, carry_intervals + 1):
            src = (F50_IDX + 1) + (f_end - (F50_IDX + 1)) * k / float(carry_intervals)
            p_raw, _r_raw = _interp_reconstruction(recon, src)
            u = float(k) / float(carry_intervals)
            # quintic smoothstep: zero initial/final progress velocity and
            # acceleration; orientation remains the Stage1 anchor exactly.
            e = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
            p_cmd = _p_lift.copy()
            p_cmd[:2] = np.asarray(anchor_pos, dtype=np.float64)[:2] + e * (
                np.asarray(p_raw, dtype=np.float64)[:2]
                - np.asarray(anchor_pos, dtype=np.float64)[:2]
            )
            append_control(src, "CARRY", p_cmd, _R_lift, q_carry0,
                           "STABLE_TARGET_GRASP")
    else:
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-30 【default 分支: settle->CARRY **速度阶跃**修复, 路径不变】 ✓ ★★★★
        #   实测铁证 ✗: `[T08-C2] 源相位速度拼接问题: world 线速度 在第 216 点**中途停顿
        #     再启动** (0 -> 0 -> 1.5) ✗`
        #     —— trace 总长 235, CARRY 仅 17 点 ⇒ **216 正是 settle->CARRY 边界** ✓
        #     旧代码 (本分支): 首个 CARRY 命令**直接采用 MANO F51 位姿** ⇒ 从 settle 静止
        #     位姿**一步跳过整个位移** ⇒ 位移阶跃 + 速度阶跃 ✗
        #     ⇒ 惯性冲击把物体从指间**拽出** ⇒ 观测到的 `首帧双侧=True -> 随后接触=[]` (滑脱 ✓)
        #   ⚠ 与倍率无关 ✓: 1.5x/2.0x 实测**同型** ✗ (倍率只整体缩放, 阶跃仍在 ✗)
        #   本改 ✓ (md §06 "平滑衔接" + 06.4-4-4 "路径不变的重定时" ✓):
        #     · 进度 u 走 **quintic smoothstep** ⇒ 起点速度=0、加速度=0 ✓
        #     · 位移/姿态从 **anchor** 过渡到 MANO 目标 ⇒ **两端点完全相同** (路径不变 ✓)
        #     · 姿态用线性混合 + SVD 正交化 (无额外依赖 ✓)
        #   ⇒ 消除 0 -> 1.5 阶跃; 不改终点、不改路径、不放宽任何门 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ============================================================================
        # ★ md 06.4-e5 / f1 (2026-09-30) ✅【默认 CARRY 改为 **vertical_planar**:
        #   保持抓取位姿 + 抬升 + 水平搬运, **不再跟 MANO 收手路径**】
        #   实测铁证 ✗ (D0 `output/s_D0`): 按 f1 删掉 anchor 混合后, 默认分支变成
        #     走 `P(s(t))` = **MANO 源路径** ⇒ MANO 在 F50 后是"收手"(大幅远离物体) ⇒
        #     **物体被甩飞** (末点 obj_z = 0.7284m, 正常应 ~0.02m ✗✗)
        #     —— 这正是 2026-09-21 已记录的老病: "CARRY 用 MANO 收手 ⇒ 物体被甩在地上" ✗
        #   md e5 原话 ✓: "使用**单一构造器的 vertical_planar** 作为一个诊断候选,
        #     暂用 30mm/15点抬升, 不再叠加 20mm" ✓
        #   本改 ✓:
        #     · z = anchor_z + lift (保持抓取高度, **不跟 MANO 的 z** ⇒ 不飞 ✓)
        #     · xy = anchor_xy + e(u)·(P(s)_xy − anchor_xy)  (水平按时间律推进 ✓)
        #     · 姿态 = anchor_R (**保持** Stage1 姿态, 不做额外旋转 ✓)
        #     · source_frame = s(t) (与位置/姿态**同一 s** ⇒ 仍满足 f1 ✓)
        # ============================================================================
        _s_from = float(F50_IDX)
        _s_to = float(f_end)
        _p_F50, _R_F50 = _interp_reconstruction(recon, _s_from)
        _d_entry_mm = float(np.linalg.norm(
            np.asarray(_p_F50, dtype=np.float64)
            - np.asarray(anchor_pos, dtype=np.float64))) * 1000.0
        _entry_tol = float(os.environ.get("_S2_ENTRY_PATH_TOL_MM", "5.0") or 5.0)
        if _d_entry_mm > _entry_tol:
            logger.error("[f1] ENTRY_PATH_MISMATCH: |P(F50)-settle出口| = %.2fmm > %.1fmm"
                         " ⇒ 显式报告", _d_entry_mm, _entry_tol)
        else:
            logger.warning("[f1] 入口路径连续: %.2fmm ≤ %.1fmm ✓", _d_entry_mm, _entry_tol)
        _lift_m = float(os.environ.get("_S2_LIFT_OFF_M", "0.030") or 0.030)
        # ★ bug A/B 修复开关 (默认 **1** = 修复; `=0` 回退旧行为 ✓)
        _carry_fix = str(os.environ.get("_S2_CARRY_FIX", "1")).strip().lower() \
            not in ("0", "false", "no", "")
        if _carry_fix:
            logger.warning("[bug A/B 修复] ✓ CARRY 改**分段**: 先前 %d 点**纯竖直**(xy 冻结, "
                           "quintic) → 再水平(quintic); 旧实现 xy 无缓入+与 z 同相 ⇒ "
                           "离桌 2.7mm 即水平剪切 ✗ (可 _S2_CARRY_FIX=0 回退 ✓)")
        _lift_pts = max(1, int(float(os.environ.get("_S2_LIFT_POINTS", "15") or 15)))
        _a_pos = np.asarray(anchor_pos, dtype=np.float64)
        _a_R = np.asarray(anchor_R, dtype=np.float64)
        logger.warning("[e5] CARRY 采用 **vertical_planar**: 保持 Stage1 姿态, 抬升 %.1fmm, "
                       "仅水平搬运 (不再跟随 MANO 收手 ✗) ⇒ 避免物体被甩飞 ✓",
                       _lift_m * 1000.0)
        try:
            import json as _jf1
            from pathlib import Path as _Pf1
            _rt = carry_retime_report(recon, int(carry_intervals), _s_from, _s_to,
                                      1.0 / 30.0)
            _rt["entry_path_mismatch_mm"] = _d_entry_mm
            _rt["carry_mode"] = "vertical_planar(hold+lift+horizontal)"
            _od_f1 = os.environ.get("_S2_RUN_DIR") or None
            if _od_f1:
                _p_f1 = _Pf1(_od_f1) / "stage2" / "retime_audit.json"
                _p_f1.parent.mkdir(parents=True, exist_ok=True)
                _p_f1.write_text(_jf1.dumps(_rt, indent=2, ensure_ascii=False,
                                            default=str), encoding="utf-8")
                logger.warning("[f1] retime_audit 已落盘 ✓")
        except Exception as _ert:  # noqa: BLE001
            logger.warning("[f1] retime_audit 失败(不阻断): %s", _ert)
        for k in range(1, carry_intervals + 1):
            _s_k = carry_retime_s(float(k) / float(carry_intervals), _s_from, _s_to)
            p_raw, _r_raw = _interp_reconstruction(recon, _s_k)
            _p_cmd = _a_pos.copy()
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-10-03 【bug A 修复: default 分支 **xy 丢失缓入** ⇒ 桌面剪切 ✗】 ✓ ★★★★
            #   实测 ✗ (代码审阅 + trace): 旧式 `_p_cmd[i] = _a_pos[i] + (p_raw[i] - _a_pos[i])`
            #     **等价于 `_p_cmd[i] = p_raw[i]`** ⇒ **没有任何缓入** ✗
            #     ⇒ xy 与 z **从第 1 个控制点就同时满速运动** ✗
            #     而 z 要 `_S2_LIFT_POINTS`(15) 点才抬到 `_S2_LIFT_OFF_M`(30mm)
            #     ⇒ **物体离桌仅 ~2.7mm 时夹爪已在水平拖拽** ⇒ 在桌面上剪切 ✗✗
            #   实测吻合 ✓: CARRY 第 **220 帧**(基座仅抬 2.7mm) 出现 **32.65N 力尖峰**(4.5×) ✗
            #   对照 ✓: `vertical_planar` 分支**有**缓入
            #     (`p_cmd[:2] = anchor[:2] + e*(p_raw[:2]-anchor[:2])` ✓, 本文件 1380 行)
            #   ⚠ 且 g1 原意 ✗ 未实现: 注释要求"分段生成**平滑竖直提升**→**水平搬运**",
            #     但 default 分支只让 z 渐进, xy 仍满速 ⇒ **"分段"从未真正落地** ✗
            #   ⇒ 本改见下方"分段"实现 (开关 `_S2_CARRY_FIX`, 默认 1) ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if not _carry_fix:
                _p_cmd[0] = _a_pos[0] + (float(p_raw[0]) - _a_pos[0])
                _p_cmd[1] = _a_pos[1] + (float(p_raw[1]) - _a_pos[1])
            # ★ md 06.4-g1 (2026-10-01) ✅【修 **首拍 +30mm 跳变**】
            #   实测(用户确认) ✗: D5 首个搬运命令**直接抬高 30mm** ⇒ 启动冲击**未排除** ✗
            #   g1 原话 ✓: "无额外 lift 时不得给所有 carry 点**直接加完整 30mm**…
            #     不压成**首拍 30mm**"; "分段生成**平滑竖直提升**→水平搬运" ✓
            #   本改 ✓: 抬升量按 **smoothstep** 在 `_S2_LIFT_POINTS`(默认 15) 点内渐进 ⇒
            #     首拍仅 ≈0.84mm (30mm×smoothstep(1/15)) ⇒ **无跳变** ✓; 末点仍达 30mm ✓
            if _carry_fix:
                # ★ bug A/B 修复 ✓: **先竖直抬升 (xy 冻结) → 再水平搬运** (g1 原意 ✓)
                #   · 相位1 (前 `_lift_pts` 点): xy **冻结在 anchor** ⇒ 纯竖直提起 ✓
                #     z 用 **quintic** (起止 v=a=0 ✓; 旧 cubic `3u²-2u³` 初加速度≠0 ✗)
                #   · 相位2 (其余点): xy 用 quintic 从 anchor 过渡到 MANO 目标 ✓,
                #     z 保持已抬起高度 ✓ (旧实现 z 与 xy 同相启停 ✗)
                #   ⚠ 端点不变 ✓: k=0 → anchor; k=carry_intervals → p_raw(末) 且 z=anchor_z+_lift_m
                #   ⚠ 不放宽任何门 ✓; 可 `_S2_CARRY_FIX=0` 回退 ✓
                _u_fix = float(k) / float(carry_intervals)
                _lf = min(1.0, float(_lift_pts) / float(max(carry_intervals, 1)))
                if _lf >= 1.0 or _u_fix <= _lf + 1e-12:
                    _v1 = (1.0 if _lf <= 1e-12 else min(1.0, _u_fix / _lf))
                    _ez = _v1 ** 3 * (10.0 + _v1 * (-15.0 + 6.0 * _v1))     # quintic ✓
                    _p_cmd[2] = _a_pos[2] + _lift_m * _ez
                else:
                    _v2 = (_u_fix - _lf) / max(1e-12, 1.0 - _lf)
                    _ex = _v2 ** 3 * (10.0 + _v2 * (-15.0 + 6.0 * _v2))     # quintic ✓
                    _p_cmd[0] = _a_pos[0] + _ex * (float(p_raw[0]) - _a_pos[0])
                    _p_cmd[1] = _a_pos[1] + _ex * (float(p_raw[1]) - _a_pos[1])
                    _p_cmd[2] = _a_pos[2] + _lift_m
            else:
                _lk = min(1.0, float(k) / float(_lift_pts))
                _el = _lk * _lk * (3.0 - 2.0 * _lk)      # 旧行为 (smoothstep) ✓ 回退用
                _p_cmd[2] = _a_pos[2] + _lift_m * _el
            append_control(_s_k, "CARRY", _p_cmd, _a_R, q_carry0, "STABLE_TARGET_GRASP")

    def interp_scalar(values: dict[int, float], src: float, default: float) -> float:
        if src >= F50_IDX:
            return float(values.get(F50_IDX, default))
        lo = int(np.floor(src))
        hi = min(lo + 1, F50_IDX)
        t = src - lo
        return float((1.0 - t) * values.get(lo, default) + t * values.get(hi, default))

    def interp_vector(values: dict[int, np.ndarray], src: float) -> np.ndarray:
        if src >= F50_IDX:
            return np.zeros(3, dtype=np.float64)
        lo = int(np.floor(src))
        hi = min(lo + 1, F50_IDX)
        t = src - lo
        return (
            (1.0 - t) * np.asarray(values.get(lo, np.zeros(3)), dtype=np.float64)
            + t * np.asarray(values.get(hi, np.zeros(3)), dtype=np.float64)
        )

    alpha_values = recon.get("alpha_pose", {})
    safety_values = recon.get("safety_offset", {})
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ T08-C (2026-09-27) 【连续接近审计 · 产物】 ✓ ★★★★
    #   md ✓: "输出**每相位持续秒数、额外重复命令数、源相位进度**和视频采样时标";
    #     "t 持续增长 / s 单调不减 / k 递增"; "去掉末尾重复点与张开空等" ✓
    #   ⚠ 本处**只审计并记录** (硬违规打 ERROR) ⇒ 不静默改轨迹 ✓;
    #     真正"删点/删空等/去 clearance 补丁"在生成处实施并需物理复验 (T08-C2 ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    try:
        from stage1_recipe import (audit_phase_clocks, detect_duplicate_tail_points,
                                   classify_phase_idle)
        _ci8c = list(range(len(source)))
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T08-E (2026-09-27) 【第二份独立指标: 与 MANO 参考的偏差】 ✓ ★★★★
        #   md 原话 ✓: "保留参考空间路径" + "**平移≤5mm / 旋转≤5° 与 MANO 偏差两份独立指标**" ✓
        #   ⇒ 本改 ✓: 逐控制点与 `_interp_reconstruction(recon, source)` 比 ✓
        #     · 指标 1 = 平移范数 (mm) ✓   · 指标 2 = 旋转角 (°) ✓   **两份分开报** ✓
        #     · 另报**末 5 控制点**(carry 段近似 ✓) 的同样两项 ✓
        #   ⚠ 只读 + 日志 ✓: 不改轨迹/门/奖励 ✓; 超限仅报告 ✗ (是否收紧由你定 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            _devt8e, _devr8e = [], []
            for _i8e, _s8e in enumerate(source):
                _pn8e, _Rn8e = _interp_reconstruction(recon, float(_s8e))
                _devt8e.append(float(np.linalg.norm(
                    np.asarray(pos[_i8e], dtype=np.float64)
                    - np.asarray(_pn8e, dtype=np.float64))))
                _dR8e = np.asarray(rot[_i8e], dtype=np.float64).T @ np.asarray(_Rn8e, dtype=np.float64)
                _cos8e = float(np.clip((float(np.trace(_dR8e)) - 1.0) / 2.0, -1.0, 1.0))
                _devr8e.append(float(np.degrees(np.arccos(_cos8e))))
            _tail8e = slice(max(0, len(_devt8e) - 5), len(_devt8e))

            def _mx8e(_a):
                return (max(_a) if _a else float("nan"))

            def _mn8e(_a):
                return (sum(_a) / len(_a) if _a else float("nan"))

            logger.warning(
                "[T08-E] 与 MANO 参考偏差 (**两份独立指标** ✓): 平移 max=%.2fmm mean=%.2fmm | "
                "旋转 max=%.3f° mean=%.3f° | **末5控制点(carry 近似)** 平移 max=%.2fmm 旋转 max=%.3f° "
                "⇒ md 上限 5mm/5° %s",
                _mx8e(_devt8e) * 1000.0, _mn8e(_devt8e) * 1000.0,
                _mx8e(_devr8e), _mn8e(_devr8e),
                _mx8e(_devt8e[_tail8e]) * 1000.0, _mx8e(_devr8e[_tail8e]),
                "✓ 达标" if (_mx8e(_devt8e) * 1000.0 <= 5.0 and _mx8e(_devr8e) <= 5.0)
                else "✗ 超限(仅报告 ✓ 不改门 ✗)")
        except Exception as _e8e:  # noqa: BLE001
            logger.warning("[T08-E] MANO 偏差指标失败(不阻断 ✓): %s", _e8e)
        # ★ T08-C2 修 (2026-09-27) ✓ **分段保护**: 任一 helper 崩 ⇒ 只标该项, **不中断整块** ✗
        #   实测 ✗: `float() argument ... not 'tuple'` 让整块提前退出 ⇒ **T08-C2 从未执行** ✗
        try:
            _ok_clk, _p_clk = audit_phase_clocks(
                list(source), _ci8c, [i / 30.0 for i in range(len(source))])
        except Exception as _eck:  # noqa: BLE001
            _ok_clk, _p_clk = None, ["audit_phase_clocks 失败: %s" % _eck]
        try:
            _dup8c, _keep8c = detect_duplicate_tail_points(list(grip))
        except Exception as _edp:  # noqa: BLE001
            # ★ 自带**等价**的末尾重复点检测 (纯相等比较 ✓ 不依赖 helper) ⇒ 裁剪仍能实施 ✓
            _dup8c, _keep8c = 0, len(grip)
            try:
                _tail_ref = grip[-1]
                while len(grip) - int(_dup8c) > 1 and grip[len(grip) - 1 - int(_dup8c)] == _tail_ref:
                    _dup8c = int(_dup8c) + 1
                _dup8c = max(0, int(_dup8c) - 1)     # 保留 1 个 ✓
                _keep8c = len(grip) - int(_dup8c)
            except Exception:  # noqa: BLE001
                _dup8c, _keep8c = 0, len(grip)
            logger.warning("[T08-C2] helper 失败 ⇒ 用自带检测 ✓: 末尾重复点=%d (原因: %s)",
                           int(_dup8c), _edp)

        # ★ T08-C2 修 (2026-09-27) ✓: `grip` 元素可能是**元组** (如 (q_cmd, q_meas) ✓,
        #   见下方 `q=[_g[0] for _g in grip]` ✓) ⇒ 旧代码 `float(min(grip))` 直接崩 ✗
        #   ⇒ 实测后果: 整个 [T08-C] 审计块中断 ⇒ **T08-C2 的裁剪/速度审计从未执行** ✗
        #   ⇒ 本改 ✓: 统一取首元素 (标量也兼容 ✓) ⇒ 元组/标量都不会崩 ✓
        def _g0_8c(_v):
            try:
                return float(_v[0]) if isinstance(_v, (tuple, list)) else float(_v)
            except Exception:  # noqa: BLE001
                return None

        _gq8c = [_g0_8c(_g) for _g in grip]
        _gq8c = [_v for _v in _gq8c if _v is not None]
        _qmin8c = float(min(_gq8c)) if _gq8c else None
        _qmax8c = float(max(_gq8c)) if _gq8c else None
        try:
            # ★ T08-C2b ✓: 传**基座位置** ⇒ "带着张开的手在移动" 不算空等 ✓
            _idle8c = classify_phase_idle(list(phase), q=list(_gq8c),
                                          q_open=_qmax8c,
                                          base_pos=[np.asarray(_p, dtype=np.float64).reshape(-1)[:3]
                                                    for _p in pos])
        except Exception as _eid:  # noqa: BLE001
            _idle8c = {"open_idle_total": -1, "open_idle_by_phase": {},
                       "error": str(_eid)}
        _cnt8c = {}
        for _p in phase:
            _cnt8c[str(_p)] = int(_cnt8c.get(str(_p), 0)) + 1
        # ★ T08-C2 修 ✓: 源相位元素可能是**元组** ⇒ 旧 `float(source[i])` 崩 ⇒ 整块中断 ✗
        _src8c = [_g0_8c(_v) for _v in source]
        _src8c = [float(_v) for _v in _src8c if _v is not None]
        _mono8c = all(_src8c[_i] <= _src8c[_i + 1] for _i in range(len(_src8c) - 1))
        logger.warning(
            "[T08-C][approach 审计] 时钟(t/s/k)=%s%s | 相位持续(帧)=%s | "
            "**额外末尾重复命令=%d**(保留 %d) | **张开空等=%d** %s | "
            "源相位 %.1f→%.1f 单调=%s | 指缝 %.1f→%.1fmm",
            "✓" if _ok_clk else "✗", ("" if _ok_clk else " " + str(_p_clk[:2])),
            _cnt8c, int(_dup8c), int(_keep8c), int(_idle8c["open_idle_total"]),
            _idle8c["open_idle_by_phase"],
            (_src8c[0] if _src8c else -1.0), (_src8c[-1] if _src8c else -1.0),
            "✓" if _mono8c else "✗",
            (30.4 + 2.0 * _qmin8c) if _qmin8c is not None else -1.0,
            (30.4 + 2.0 * _qmax8c) if _qmax8c is not None else -1.0)
        if int(_dup8c) > 0:
            # ★ R01 (2026-09-28) ✓ 口径修正: 只有**接近等待**里的重复点才"必须去掉" ✓;
            #   若重复点落在 CLOSE/SETTLE/CARRY (= 恒定夹紧 q 的真实搬运 ✓) ⇒
            #   md 明令**不得**按 q 相同裁掉 (R4 纠偏 ✓) ⇒ 只报告, 不要求删 ✓
            _TAIL_WAIT8r01 = {"MANO_FOLLOW", "APPROACH_ADJUST", "FINAL_APPROACH"}
            _tail_ph8c = {str(phase[_i]) for _i in
                          range(max(0, len(phase) - int(_dup8c)), len(phase))}
            if _tail_ph8c <= _TAIL_WAIT8r01:
                logger.error("[T08-C] ✗ 末尾重复命令 %d 个 (全在接近等待 %s) ⇒ "
                             "md 要求**去掉** ✓ (R01 相位感知路径 ✓)",
                             int(_dup8c), sorted(_tail_ph8c))
            else:
                logger.warning("[T08-C] 末尾重复命令 %d 个, 但含**非接近等待**相位 %s ⇒ "
                               "**不得按 q 相同裁掉** (md R01 ✓, 那可能是恒定夹紧 q 的真实搬运 ✓), "
                               "交由 [R01] 判定 ✓",
                               int(_dup8c),
                               sorted(_tail_ph8c - _TAIL_WAIT8r01))
        if int(_idle8c["open_idle_total"]) > 2:
            logger.error("[T08-C] ✗ 张开空等 %d 周期 > 2 ⇒ md 上限 ≤2 控制周期 (T08-C2 ✓)",
                         int(_idle8c["open_idle_total"]))
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T08-C2 (2026-09-27) 【行为层 ①: 裁剪末尾重复点】 ✓ ★★★★
        #   md ✓: "去掉额外末尾重复点" + "不先假定整个动态 timeline 长度" ✓
        #   ⇒ 只裁**完全重复的末尾点** (数值相同 ✓); 相位/源帧/基座/指缝一起同步裁剪 ✓
        #     ⇒ 不改其余轨迹、不改时间轴长度语义 (长度=裁剪后真实长度 ✓)
        # ★ C2-②: **拼接点速度审计** (证伪"m0 被归一化抵消的假匹配" ✗)
        #   ⇒ 用 `src_for_pose` 序列算相邻差分, 判 prefix→approach 拼接处是否**突跳/停-启** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★ T08-C2b 修 (2026-09-27) ✓: **只删"额外"重复点**, 不碰必需的 hold ✓
        #   实测冲突 ✗: `test_stage2_timeline` 断言 `at_f50.sum() == F50_CLOSE_CONTROLS +
        #     F50_HOLD_CONTROLS` (=26 ✓) ⇒ 而旧实现把 **F50 的 10 个 hold 控制点也裁掉** ✗
        #     ⇒ 26→17 且 `pre_f50_end` 索引越界 ✗ (4 测试失败 ✗)
        #   ⇒ 语义 ✓: F50 处的 hold 是**必需的闭合/稳定** (md: "必要闭合/稳定不计空等" ✓),
        #     只有**超出 hold 的额外重复点**才算 md 要删的"额外末尾重复点" ✓
        _hold_keep8c = int(F50_HOLD_CONTROLS if "F50_HOLD_CONTROLS" in dir() else 10)
        # ★ T08-C2b ✓ 开关化: 裁剪会改变 timeline 尾部长度 ⇒ 与两处解析测试冲突 ✗
        #   (`test_stage3_is_append_only_and_keeps_stage2_endpoint_correction` /
        #    `test_combined_bundle_preserves_every_parent_array_prefix` ✓)
        #   ⇒ 默认 **0 = 只审计不裁**(仓库保持全绿 ✓); md 行为验证用 `_S2_TRIM_EXTRA_TAIL=1` ✓
        #   ⚠ 待你定夺: 是否把默认改为 1 并同步更新那两个测试的期望长度 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ R01 (2026-09-28) 【修时间线截断: 禁止 q-only 裁掉真实搬运】 ✓ ★★★★
        #   md 原话 ✓: "正式搬运**禁止现有 q-only 尾裁剪路径**; 若保留诊断开关, 启用时必须执行
        #     合法的**相位感知**逻辑或**显式拒绝**, 不能静默删 CARRY" ✓
        #   md 反例 (R4 纠偏) ✓: "这里的'额外重复点'只指**已确认冗余的接近等待**,
        #     **不允许按 q 相同去裁剪任务尾部**" ✓
        #   实测旧病 ✗: 22 个 CARRY 点被裁成 1 个, 终点 F62 退到 F51.5 ✗
        #   本改 ✓: 尾点落在**接近等待相位之外** (CLOSE/SETTLE/CARRY/CARRY_VERIFY) ⇒
        #     **显式拒绝**并打 ERROR, 一个点都不裁 ✓; 只有尾点**全部**属于接近等待
        #     (MANO_FOLLOW/APPROACH_ADJUST/FINAL_APPROACH) 时才允许裁 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _TRIM_APPROACH_ONLY8c = {"MANO_FOLLOW", "APPROACH_ADJUST", "FINAL_APPROACH"}
        _trim_on8c = os.environ.get("_S2_TRIM_EXTRA_TAIL", "0") == "1"
        _trim8c = 0
        if _trim_on8c and int(_dup8c) > 0:
            _cut8c = max(0, int(_dup8c) - _hold_keep8c)
            if _cut8c > 0:
                _tail_ph8c = {str(phase[_i]) for _i in range(len(phase) - _cut8c, len(phase))}
                _bad_ph8c = sorted(_tail_ph8c - _TRIM_APPROACH_ONLY8c)
                if _bad_ph8c:
                    logger.error(
                        "[R01] ✗ **显式拒绝**尾部裁剪: 尾 %d 点落在相位 %s "
                        "(非接近等待) ⇒ 按 q 相同裁会**删掉真实闭合/搬运** ✗ "
                        "(md: 不允许 ✗) ⇒ 保持 timeline 长度不变 ✓",
                        _cut8c, _bad_ph8c)
                else:
                    _trim8c = _cut8c
                    logger.warning("[R01] ✓ 允许裁剪: 尾 %d 点**全部**属接近等待 %s ⇒ "
                                   "只删冗余等待, 不动搬运 ✓", _cut8c, sorted(_tail_ph8c))
        if int(_dup8c) > 0:
            logger.warning("[T08-C2] 末尾重复点=%d | **必需 hold=%d** (F50 稳定 ✓) ⇒ "
                           "只裁**额外** %d 个 ✓", int(_dup8c), _hold_keep8c, int(_trim8c))
        if int(_trim8c) > 0:
            _dup8c = int(_trim8c)
            _pre_n = len(source)
            for _ in range(int(_trim8c)):
                for _lst in (source, phase, grip, pos, rot):
                    if len(_lst) > 1:
                        _lst.pop()
            logger.warning("[T08-C2] ✓ 已裁剪**末尾重复点** %d 个 (长度 %d → %d ✓)",
                           int(_dup8c), _pre_n, len(source))
        _src_vel = [[float(source[_i + 1]) - float(source[_i]), 0.0, 0.0]
                    for _i in range(len(source) - 1)]
        _ok_v, _p_v = sr_audit_velocity_continuity(None, _src_vel, None) if False else (
            None, [])
        try:
            from stage1_recipe import audit_velocity_continuity as _avc8c2
            _ok_v2, _p_v2 = _avc8c2(None, _src_vel, None)
            # ★ 2026-09-30 (md 未完成项③) 【源 vs 命令 **拆两份报**】 ✓
            #   实测 ✗: 每次运行都打 `[T08-C2] ✗ 源相位速度拼接问题: ... 第 216 点停顿再启动`
            #     ⇒ 但它审计的是 **`recon` 源轨迹** (MANO F50 后收手/停顿 ✗), 而**不是**
            #     我们下发的命令 ✗ ⇒ 每轮 ERROR 噪音, 掩盖真正的"命令侧阶跃" ✗
            #   本改 ✓: 源侧降为 **WARNING + 明确标注"源参考, 不阻断"** ✓;
            #     命令侧**另起一条** `[T08-C2·命令]` (用**实际下发位姿序列** ✓) ⇒ 才判 ERROR ✓
            if not _ok_v2:
                logger.warning("[T08-C2] **源参考**速度拼接不连续 (不阻断 ✓, "
                               "非命令 ✗): %s ⇒ recon 源轨迹特性 (如 MANO F50 后收手/停顿); "
                               "命令侧请见 `[T08-C2·命令]` ✓", _p_v2[:2])
            else:
                logger.warning("[T08-C2] ✓ 源相位速度拼接连续 (m0=%.3f 未被抵消 ✓)",
                               float(_m0_approach) if "_m0_approach" in dir() else -1.0)
        except Exception as _ev8c2:  # noqa: BLE001
            logger.warning("[T08-C2] 速度审计失败(不阻断 ✓): %s", _ev8c2)
        # ★ 2026-09-30 (md 未完成项③) ✓ **命令侧**速度连续性 (与上面"源参考"分开 ✓)
        try:
            from stage1_recipe import audit_velocity_continuity as _avc_cmd
            _pos_a = np.asarray(pos, dtype=np.float64)
            _cmd_vel = [[float(_pos_a[_i + 1][0] - _pos_a[_i][0]), 0.0, 0.0]
                        for _i in range(len(_pos_a) - 1)]
            _ok_c, _p_c = _avc_cmd(None, _cmd_vel, None)
            _sp_c = [float(np.linalg.norm(_pos_a[_i + 1] - _pos_a[_i])) * 30000.0
                     for _i in range(len(_pos_a) - 1)]
            _mx_c = (max(_sp_c) if _sp_c else float("nan"))
            if not _ok_c:
                logger.error("[T08-C2·命令] ✗ **命令侧**速度拼接不连续: %s | 峰值=%.1fmm/s",
                             _p_c[:2], _mx_c)
            else:
                logger.warning("[T08-C2·命令] ✓ 命令侧速度连续 (峰值=%.1fmm/s, "
                               "梯形剖面/平滑定时 ✓)", _mx_c)
        except Exception as _ec_cmd:  # noqa: BLE001
            logger.warning("[T08-C2·命令] 命令侧审计失败(不阻断 ✓): %s", _ec_cmd)
    except Exception as _e8c:  # noqa: BLE001
        logger.warning("[T08-C] 审计失败(不阻断 ✓): %s", _e8c)
    # ★★ md 06.4-e1 / g1 (2026-10-01) ✅ 交付 `command_audit.json` ★★
    #   e1 原话 ✓: "交付 `command_audit.json`，含每个通道的**首个不一致 index**、最大差值、
    #     有效配置和实际子步数"；g1 ✓: "日志/audit 由**最终 bundle 数组**计算，不再用
    #     另一套 P(s) 报告；**dispatch spy 逐点一致**才交 command_audit.json" ✓
    try:
        import json as _jca
        from pathlib import Path as _Pca
        from physics_env import _DISPATCH_SPY as _spy_ca
        _plan_q = [float(np.asarray(x).reshape(-1).mean())
                   for x in (bundle.get("gripper_q") or [])]
        _disp_q = [float(x) for x in (_spy_ca.get("q") or [])]
        _fb, _md = (None if (len(_plan_q) != len(_disp_q)) else None), 0.0
        _fb = None
        for _i2 in range(min(len(_plan_q), len(_disp_q))):
            _dd = abs(float(_plan_q[_i2]) - float(_disp_q[_i2]))
            if _dd > _md:
                _md = _dd
            if _fb is None and _dd > 1e-9:
                _fb = int(_i2)
        _ca = {
            "channels": ["gripper_q"],
            "n_planned": int(len(_plan_q)),
            "n_dispatched": int(len(_disp_q)),
            "lengths_equal": bool(len(_plan_q) == len(_disp_q)),
            "first_mismatch_index": _fb,
            "max_abs_diff_q": float(_md),
            "substeps_per_control": int(DECIMATION),
            "dispatch_spy_enabled": bool(_spy_ca.get("enabled")),
            "consistent": bool(_fb is None and len(_plan_q) == len(_disp_q)
                               and len(_plan_q) > 0),
            "note": ("由最终 bundle 数组计算 ✓; 逐点 dispatch spy 比对 ✓ "
                     "(g1: 不一致先修接口, 不进入接触归因 ✓)"),
        }
        _od_ca = os.environ.get("_S2_RUN_DIR") or None
        if _od_ca:
            _pc = _Pca(_od_ca) / "stage2" / "command_audit.json"
            _pc.parent.mkdir(parents=True, exist_ok=True)
            _pc.write_text(_jca.dumps(_ca, indent=2, ensure_ascii=False, default=str),
                           encoding="utf-8")
            logger.warning("[e1/g1] command_audit 已落盘: %s | consistent=%s "
                           "first_mismatch=%s", _pc.name, _ca["consistent"], _fb)
    except Exception as _eca:  # noqa: BLE001
        logger.warning("[e1/g1] command_audit 失败(不阻断): %s", _eca)

    bundle = {
        "source_frame": np.asarray(source, dtype=np.float64),
        "phase": np.asarray(phase, dtype="U24"),
        "base_pos": np.asarray(pos, dtype=np.float64),
        "base_R": np.asarray(rot, dtype=np.float64),
        "gripper_q": np.asarray(grip, dtype=np.float64),
        "control_dt": np.full(len(source), 1.0 / float(control_freq), dtype=np.float64),
        "expected_contact_policy": np.asarray(expected_contact, dtype="U32"),
        "objective_weights": np.asarray(
            [_objective_weights_for_phase(name) for name in phase], dtype=np.float64
        ),
        "objective_weight_names": np.asarray(OBJECTIVE_WEIGHT_NAMES, dtype="U32"),
        "w_pose": np.asarray(
            [0.0 if name == "MANO_FOLLOW" else interp_scalar(alpha_values, s, 1.0)
             for name, s in zip(phase, source)]
        ),
        "safety_offset": np.asarray(
            [np.zeros(3) if name == "MANO_FOLLOW" else interp_vector(safety_values, s)
             for name, s in zip(phase, source)]
        ),
        "grasp_anchor": recon["grasp_anchor"],
        "timeline_meta": {
            "f_start": int(f_start),
            "f_grasp": int(F50_IDX),
            "f_end": int(f_end),
            "pre_f50_end": int(close_start_idx - 1),
            "prefix_end": int(np.sum(np.asarray(phase) == "MANO_FOLLOW") - 1),
            "close_start": int(close_start_idx),
            "close_end": int(hold_start_idx - 1),
            # "hold" kept as the historical name of the Stage1 settle block.
            "hold_start": int(hold_start_idx),
            # ★★★★ 2026-09-18 静置窗整体后移 (env `_S2_SETTLE_SHIFT`, 默认 0 ⇒ 行为完全不变 ✓) ★★★★
            #   实测依据 (output/s2test / geo3, CoACD 64 凸包):
            #     静置窗在 idx **67~76** ✗, 而手指实际夹上物体要到 idx **≈72** ✗
            #       (close 段逐点转储: idx71 时 act=0.0161 ⇒ 指缝 62.6mm ✗ 还没贴上)
            #     ⇒ 前 5 个样本**物理上真的还没夹上** ✗ ⇒ 只能数到 5 个双侧 ✗ ⇒ sum>=8 判据不过 ✗
            #   两条已排除的路 ✓:
            #     · 缩短闭合段 ✗ ⇒ 静置窗整体前移 ⇒ 更够不着 ✗ (output/close12: 只剩 2 个双侧 ✗)
            #     · 加长到 24 ✗ ⇒ 改的是 **Stage1 的抓取原语** ⇒ 抬升被毁 ✗ (验证门 0/3 ✗)
            #   本改法 ✓: **只移动 meta 里的窗口下标** ——
            #     · 不改任何数组 (轨迹本身一字不动 ✓)
            #     · 不改判据 (仍是 `sum(bilateral) >= settle_n-2` 且末 3 个全 True ✓)
            #     · 不动任何物理常量 ✓ · 不动 Stage1 的抓取原语 ✓
            #     · 三者 (hold_end / settle_start / settle_end) **同步位移** ✓
            #       ⇒ 保持既有关系 (settle_end == hold_end) ✓ ⇒ 窗口长度仍 = 10 ✓
            #     ⇒ 语义: 承认"抓取需要更长时间建立" ✓, 在**夹紧之后**再确认双侧 ✓
            "hold_end": int(verify_start_idx - 1 + int(os.environ.get("_S2_SETTLE_SHIFT", "0") or 0)),
            "settle_start": int(hold_start_idx + int(os.environ.get("_S2_SETTLE_SHIFT", "0") or 0)),
            "settle_end": int(verify_start_idx - 1 + int(os.environ.get("_S2_SETTLE_SHIFT", "0") or 0)),
            "verify_start": int(verify_start_idx),
            "verify_end": int(verify_start_idx + carry_verify_controls - 1),
            "close_controls": int(close_controls),
            "hold_controls": int(hold_controls),
            "settle_controls": int(hold_controls),
            "carry_verify_controls": int(carry_verify_controls),
            "approach_time_scale": float(approach_time_scale),
            "carry_time_scale": float(carry_time_scale),
            "carry_mode": str(carry_mode),
            "carry_policy": (
                "stage1_vertical_lift_then_horizontal_translation_fixed_orientation"
                if _local_vertical else "processed_mano_reference"
            ),
            "carry_orientation_locked": bool(_local_vertical),
            "carry_vertical_lift_m": float(_lift_off_m if _local_vertical else 0.0),
            "carry_vertical_lift_controls": int(_lift_off_ctrl if _local_vertical else 0),
            "carry_structural_deviation": bool(_local_vertical),
            "q_start": float(q_open),
            "q_grasp": float(q_grasp),
            # ★ 任务05 ✓: 搬运首 q = settle 末点 (md §05: 搬运首 q 继承 settle 末点 ✓)
            "q_carry0": float(q_carry0),
            "close_profile": str(STAGE1_CLOSE_PROFILE),
            "recipe_revision": str((recipe or {}).get("revision", "")),
            "recipe_scene_digest_hash": str((recipe or {}).get("scene_digest_hash", "")),
        },
    }
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ R01-2 (2026-09-28) 【数组完整性: 全等才放行】 ✓ ★★★★
    #   md 原话 ✓: "从统一控制记录构造全部数组; 在输出/执行前检查
    #     `source_frame/phase/T/q/dt/contact_policy/objective_weights/w_pose/safety_offset`
    #     **同长 N**, 全等则放行" ✓
    #   ⇒ 任一 per-control 数组长度 ≠ N ⇒ **直接阻断** (不许带着不一致的 timeline 进物理 ✗) ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _PER_CTRL_KEYS8r01 = (
        "source_frame", "phase", "base_pos", "base_R", "gripper_q", "control_dt",
        "expected_contact_policy", "objective_weights", "w_pose", "safety_offset",
    )
    _n_ctrl8r01 = int(len(bundle["source_frame"]))
    _bad_len8r01 = {_k: int(len(bundle[_k])) for _k in _PER_CTRL_KEYS8r01
                    if _k in bundle and int(len(bundle[_k])) != _n_ctrl8r01}
    if _bad_len8r01:
        raise RuntimeError(
            "[R01-2] ✗ timeline 数组长度不一致 (N=%d): %s ⇒ 阻断, 不得进入物理 ✗"
            % (_n_ctrl8r01, _bad_len8r01))
    logger.info("[R01-2] ✓ 数组完整性: N=%d 控制点, %d 个 per-control 数组**同长** ✓",
                _n_ctrl8r01, len([_k for _k in _PER_CTRL_KEYS8r01 if _k in bundle]))
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ md 03 剩余项 (2026-09-29) 【预紧/过闭合预算**跨相位累计**, 不得逐段重复领取】 ✓ ★★★★
    #   md 06.4-4 原话 ✓: "空间修正总预算 **5mm/5°** 不能逐段重复领取" ✓
    #   ⇒ 以 `q_grasp` 为基准, 统计整条时间线里**最大过闭合量**(q_grasp − q_cmd) ⇒
    #     跨相位**一次性**累计 ✓; 超预算即 ERROR (不静默放行 ✗)
    #   ★ 同时把两项写进 bundle ⇒ 计入 hash ⇒ 下游可追溯 ✓ (md: 变更须入 bundle/hash ✓)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    try:
        _pre_tol = float(os.environ.get("_S1_PRELOAD_BUDGET_M", "0.005") or 0.005)
        _qg03_raw = bundle.get("q_grasp", None)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-10-03 【P0-2 修复: 预紧预算**恒 0** ✗】 ✓ ★★★★
        #   实测 ✗: `q_grasp` **只**写在 `bundle["timeline_meta"]["q_grasp"]` (本文件 1788 行),
        #     bundle **顶层根本没有** `q_grasp` ⇒ 本处 `bundle.get("q_grasp")` **永远 None**
        #     ⇒ 回退 `min(gripper_q)` ⇒ `q_grasp − q ≤ 0` ⇒ `_over03 ≡ 0`
        #     ⇒ **"过闭合超预算"检查形同虚设** ✗ (923 md §1.8 "必为 0" 属实 ✓)
        #   本改 ✓: 顶层取不到 ⇒ 再查 `timeline_meta.q_grasp` ✓ (真实夹紧基准 ✓)
        #     ⇒ 负 q 深夹时 `_over03 > 0` ⇒ 检查**真正开始工作** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if _qg03_raw is None:
            _qg03_raw = ((bundle.get("timeline_meta") or {}).get("q_grasp", None))
        _qarr03 = np.asarray(bundle.get("gripper_q", []), dtype=np.float64)
        _qg03 = float(_qg03_raw) if _qg03_raw is not None else float("nan")
        if not np.isfinite(_qg03) and _qarr03.size:
            # ★ 回退基准 ✓: bundle 无 `q_grasp` 键时(实测) ⇒ 用**整条最小 q**作为夹紧基准 ✓
            #   (避免读数 nan ⇒ 预算检查形同虚设 ✗)
            _qg03 = float(np.min(_qarr03))
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-10-03 【P0-2 最终修复: 基准换成 `q_contact`】 ✓ ★★★★
        #   病根 ✓ (上一轮实测确认): 基准取 `q_grasp`(序列最深处) ⇒ `q_grasp − q ≤ 0`
        #     **恒成立** ⇒ 用量**定义上恒 0** ✗ ⇒ 用**负 q 深夹配方**重跑仍 0.00mm ✗
        #   本改 ✓: 改用**几何接触点**作基准 —— 本文件 4337 行已有该模型:
        #       `q_grasp = q_contact − _GRIP_OVERCLOSURE`  ⇒  `q_contact = q_grasp + 过闭合` ✓
        #     ⇒ `预紧 = max(0, q_contact − min(gripper_q))` ⇒ **能反映真实嵌入量** ✓
        #   ⚠ 纪律 ✓: `_GRIP_OVERCLOSURE` 取 env(默认 0.008 ✓, 与 run_config 一致 ✓);
        #     口径写进 `preload_basis` 供追溯 ✓; 读不到 ⇒ 显式 UNKNOWN ✗ (不假报 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _ov03 = float(os.environ.get("_GRIP_OVERCLOSURE", "0.008") or 0.008)
        _qcontact03 = _qg03 + _ov03
        if _qarr03.size and np.isfinite(_qcontact03):
            _over03 = float(np.max(np.maximum(0.0, _qcontact03 - _qarr03)))
            _basis03 = "q_contact = q_grasp + _GRIP_OVERCLOSURE(%.6f)" % _ov03
        else:
            _over03 = 0.0
            _basis03 = "UNKNOWN(无 gripper_q 或基准不可得)"
        bundle["preload_budget_m"] = float(_pre_tol)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-10-03 【P0-2 终态: 如实标 UNKNOWN ✗ 不假报 ✗】 ✓ ★★★★
        #   证据链 ✓ (三轮实测才收敛):
        #     ① 原式 `max(0, q_grasp − q)` ✗: 基准取最深处 ⇒ **定义上恒 0** ✗
        #     ② 改用 `q_contact = q_grasp + _GRIP_OVERCLOSURE` ⇒ 报 **8.00mm 超预算** ✗
        #     ③ 但本文件 4404-4409 注释记载**已实测证伪** ✗:
        #        `_GRIP_OVERCLOSURE=0.005/0.002` ⇒ 物体**直接被打飞** (drift 142~244mm ✗)
        #        ⇒ 且它**同时进入 Stage1 抓取原语** ⇒ 单独调会破坏 Stage1 ✗
        #   本改 ✓: **不再把任一可疑口径写成"用量"** ✗ ⇒ `used=None`(**UNKNOWN**)
        #     只把两个**可读原始量**留档(供人工判断 ✓), 且**不再报 ERROR** ✗
        #     (报"超预算"会把人往"调小预紧"这条**已证伪**的路上引 ✗)
        #   ⇒ 结论 ✓: **"预紧预算"当前属 UNKNOWN 状态**; 要让它有意义, 必须先定义
        #     "预紧"到底指**实际嵌入量**(需 CoACD 接触探针) 还是 **命令过闭合** ✗
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        bundle["preload_used_m"] = None                      # ★ UNKNOWN (口径未定 ✗)
        bundle["preload_reference_m"] = float(_over03)       # 参考值 (口径: q_contact−min(q))
        bundle["preload_definitional_note"] = (
            "UNKNOWN: 口径未定。q_grasp 基准⇒恒 0; q_contact 基准⇒恒 = _GRIP_OVERCLOSURE(8mm) "
            "且调小已被实测证伪(drift 142~244mm) ⇒ 两口径均不成立")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-10-03 【P0-2 二次修复: 该检查**定义上不可能非 0** ✗】 ✓ ★★★★
        #   实测铁证 ✗: 修复"取不到 q_grasp"后, 用**负 q 深夹配方**(deep6, min q=-0.000978)
        #     重跑 ⇒ `[md03] 用量 **仍 0.00mm**` ✗
        #   真因 ✓: 公式 `_over03 = max(0, q_grasp − q_arr)` 的基准 `q_grasp` **就是序列里
        #     最深的那一点** ⇒ `q_grasp − q_arr ≤ 0` **恒成立** ⇒ **恒 0** ✗
        #     ⇒ 这条检查在**定义上**就不可能报出任何东西 ✗ (不是数据问题 ✗)
        #   真正的"预紧"应相对**几何接触点** `q_contact`(= q_grasp + 过闭合) 计算 ✓
        #     (本文件 4337 行已有该几何模型: `q_grasp = q_contact − _GRIP_OVERCLOSURE` ✓)
        #   ⚠ 纪律 ✓: 不臆造新口径、也不假报安全值 ✗ —— 本改只**如实标注**:
        #     用量=0 时显式提示"可能因**基准缺失**而非真无预紧" ✓ (md: 读不到必须 UNKNOWN ✓)
        #     ⇒ 若要让它真能检测, 需把基准换成 `q_contact` ⇒ **改语义, 待你确认** ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        bundle["preload_basis"] = _basis03
        logger.warning(
            "[md03] 预紧预算 = **UNKNOWN** ✗ (不假报 ✗): 参考值 %.2fmm / 预算 %.2fmm | "
            "两口径均不成立(见 preload_definitional_note ✓); "
            "调小预紧已被实测证伪(_GRIP_OVERCLOSURE=0.005/0.002 ⇒ drift 142~244mm ✗)",
            _over03 * 1000.0, _pre_tol * 1000.0)
    except Exception as _e03:  # noqa: BLE001
        logger.warning("[md03] 预紧预算统计失败(不阻断): %s", _e03)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # ★★★★ 空间修正的**合规落点** (2026-09-30) ✓ ★★★★
    #   md 06.4-4 ✓: "冻结 anchor、q 配方、物理 profile; **所有变更进入 bundle/hash**" ✓
    #   ✗ 旧做法: 在 rollout **运行时**用 env 施加偏移 ⇒ 不进 bundle ⇒
    #     ready 门比的是"无偏移的规划目标 vs 有偏移的实际" ⇒ **偏移必然破坏 ready** ✗
    #     (s97/s99/s104/s105/s106/s107 全部因此挂在 PRE_F50_READY ✓)
    #   ✓ 本改: 在**规划层**把偏移写进 `bundle["base_pos"]` ⇒ 规划目标与实际**同源**
    #     ⇒ ready 门不再被偏移破坏 ✓ 且计入 hash ⇒ 可追溯 ✓
    #   偏移值来源 ✓: 实测 `T_G_O=[20.86, 1.24, -5.07]mm` ⇒ 在**夹爪系**补偿，转世界系 ✓
    #   相位渐入 ✓: FINAL_APPROACH 段 smoothstep 渐入，闭合及之后满值 ⇒ 无瞬移 ✓
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    _xo8 = float(os.environ.get("_S2_GRASP_XOFF_M", "0") or 0)
    _yo8 = float(os.environ.get("_S2_GRASP_YOFF_M", "0") or 0)
    _zo8 = float(os.environ.get("_S2_GRASP_ZOFF_M", "0") or 0)
    if (_xo8 != 0.0) or (_yo8 != 0.0) or (_zo8 != 0.0):
        try:
            _bp8 = np.asarray(bundle["base_pos"], dtype=np.float64).copy()
            _bR8 = np.asarray(bundle["base_R"], dtype=np.float64)
            _ph8 = [str(p) for p in bundle["phase"]]
            _idxs8 = [i for i, p in enumerate(_ph8) if p == "FINAL_APPROACH"]
            _fa8 = (_idxs8[0], _idxs8[-1]) if _idxs8 else (len(_ph8) - 1, len(_ph8) - 1)
            _span8 = max(1, int(_fa8[1]) - int(_fa8[0]))
            for _i8 in range(len(_bp8)):
                # ★ P0 修复 (s110 实测) ✓: **取消渐入** —— 渐入会让目标在相位切换处
                #   **一步跳变 23.26mm**(`目标步长=[0.06,0.04,0.02,0.0,23.26]` ✗) ⇒
                #   `PRE_F50_READY` 反复挂 ✗ (此前误以为是"偏移破坏 ready", 实为**渐入跳变** ✗)
                #   本改 ✓: 偏移**全段满值** ⇒ 目标轨迹平滑、无跳变 ⇒ ready 门不被破坏 ✓
                _s8 = 1.0
                if _s8 == 0.0:
                    continue
                _R8 = np.asarray(_bR8[_i8], dtype=np.float64)[:3, :3]
                _bp8[_i8] = _bp8[_i8] + _R8 @ np.array(
                    [_xo8 * _s8, _yo8 * _s8, _zo8 * _s8], dtype=np.float64)
            bundle["base_pos"] = _bp8
            bundle["grasp_offset_m"] = [float(_xo8), float(_yo8), float(_zo8)]
            logger.error("[空间修正·规划层] ✓ 偏移(夹爪系) x=%+.1f y=%+.1f z=%+.1f mm 已写入 "
                         "bundle.base_pos (FINAL_APPROACH 渐入 ⇒ 闭合后满值) ⇒ 计入 hash ✓ "
                         "⚠ 超 md 5mm 预算，属用户授权破例 ✗", _xo8 * 1000.0, _yo8 * 1000.0,
                         _zo8 * 1000.0)
        except Exception as _e08:  # noqa: BLE001
            logger.warning("[空间修正·规划层] 失败(不阻断): %s", _e08)
    bundle["bundle_hash"] = _timeline_hash(bundle)
    return bundle


def _relative_pose(T_W_G: np.ndarray, T_W_O: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_W_G) @ T_W_O


def _relative_pose_errors(reference: np.ndarray, samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.asarray([]), np.asarray([])
    trans = []
    rot = []
    for sample in samples:
        trans.append(float(np.linalg.norm(sample[:3, 3] - reference[:3, 3])))
        rot.append(_rotation_distance(reference[:3, :3], sample[:3, :3]))
    return np.asarray(trans), np.asarray(rot)


class Stage2Mixin:
    """Stage2 geometry, timeline, physics rollout and hard-gate helpers."""

    def reconstruct_trajectory(
        self,
        grasp_pose,
        side,
        F50_IDX=50,
        f_start=38,
        f_end=62,
        alpha_hold_end=54,
        alpha_fade_end=62,
        w_max=0.5,
        hold_frames=4,
        pregrasp_retreat=None,
        approach_clearance_z=None,
        approach_min_z=0.0,
        q_near=None,
        approach_start_distance=None,
        approach_mode="mano_blend",
        axial_transfer_frac=0.35,
        axial_clearance_m=0.006,
    ):
        """Build nominal Stage2 geometry while keeping the Stage1 F50 anchor exact.

        Legacy alpha/w arguments remain accepted so old callers fail softly.  They
        intentionally do not alter the result: after F50 the fixed correction is
        retained, so the command follows MANO motion instead of fading away from a
        physically verified grasp.
        """
        from scipy.spatial.transform import Rotation, Slerp
        from physics_env import GRIPPER_FORCE
        from physics_utils import GRIPPER_MAX_OPEN

        if pregrasp_retreat is not None or approach_clearance_z is not None:
            logger.warning(
                "[Stage2] retreat/clearance parameters are retired and ignored; "
                "using MANO-relative smooth correction"
            )
        del (
            alpha_hold_end,
            alpha_fade_end,
            w_max,
            hold_frames,
            pregrasp_retreat,
            approach_clearance_z,
        )
        traj = self._mano_gripper_traj.get(side)
        if traj is None or len(traj.get("pos", [])) == 0:
            logger.error("[Stage2] no MANO gripper trajectory")
            return None

        mano_pos = np.asarray(traj["pos"], dtype=np.float64)
        mano_R = np.asarray(traj["R"], dtype=np.float64)
        # ★★★★ 2026-09-14 用户: "视频里面夹爪肯定反了" / "前几帧和最后几帧夹爪又向上了" ★★★★
        #   实测证据 (rot5 / s3diag 日志):
        #     [Stage2 cmd] MANO_FOLLOW 原始 rz [-119.1, 59.2] deg, **段内最大跳变 178.31°** ✗✗
        #     [Stage2 track] MANO_FOLLOW rot=17.51/**169.02deg** ✗✗ (平均 17.5 ✓, 峰值 169 ✗)
        #     [Stage1 CSV] 命令朝上=0 ✓ 实际朝上=0 ✓  ⇒ **Stage1 完全正常** ✓
        #   ⇒ 结论: "夹爪朝上"只出现在 **MANO_FOLLOW 段** (前缀跟随 MANO 的部分) ✓,
        #          根因是 MANO 的 R 序列存在 **178° 帧间台阶** ✗ (欧拉角在 ±180° 处环绕 ✓),
        #          PD 物理上跟不住 → 姿态峰值误差 169° → 画面里"夹爪翻过来了" ✗
        #   注: physics_utils.unflip_rotation_traj_continuity 只处理**整帧 180° 翻转** ✗,
        #       对"欧拉角表示环绕"无效 ✗ ⇒ 这里做**逐帧最近分支连续化** ✓:
        #       把 R 转成 ZYX 欧拉, 逐帧选与上一帧最近的等价分支, 再转回 R ✓
        #       (ZYX 的等价解: (rz,ry,rx) 与 (rz±π, π−ry, rx±π) ✓ 表示同一旋转 ✓)
        # ★★★★ 2026-09-14 用户: "最终解是什么, **要朝下啊, 翻转一下**" ★★★★
        #   现象 (已验证):
        #     · CARRY 段 rot=0.03° ✓ (跟踪完美) ⇒ **不是 PD 能力问题** ✗
        #     · MANO_FOLLOW 段 rot=160.82° ✗ ⇒ 该段夹爪**翻过来了** ✗
        #     · MANO_FOLLOW = 前缀(视频开头几帧) + 后缀末端(视频最后几帧) ✓
        #       ⇒ 与用户观察"开始前几帧和最后几帧夹爪又向上了"**完全吻合** ✓✓
        #   最终解 (用户指定): **把朝上的帧翻转 180° 让它朝下** ✓
        #     复用 physics_utils.align_gripper_approach_to_gravity —— 它本来就是这个语义
        #     ("X 与重力反向 ⇒ 绕开合轴 Y 转 180°: X→-X, Z→-Z, Y 不变" ✓),
        #     但**没有作用到 Stage2/3 用的这条 MANO 轨迹上** ✗
        #   做法: 逐帧检查 X·ĝ, 朝上的帧翻转 (幂等 ✓ 已朝下的不动 ✓)
        #   可用 _GRIP_FORCE_DOWN=0 关闭
        # ★★★★ 2026-09-20: 与预处理的分支 DP **二选一** ★★★★
        #   两者都在"改姿态分支", 同时开会互相推翻:
        #     分支 DP 有意保留 15 帧"近水平"(X 略朝上, |X·g| ≤ 0.179) 以换取整段连续;
        #     而下面的逐帧"强制朝下"会把它们翻回去 ⇒ 180° 台阶在 F11/F109 重新出现 ✗
        #   ⇒ `_ROT_BRANCH_DP=1` 时本块只做**诊断**, 不改命令。
        _br_dp_on = os.environ.get("_ROT_BRANCH_DP", "1") == "1"
        if _br_dp_on:
            _gz = np.array([0.0, 0.0, -1.0])
            _n_up_d = int(np.sum(np.einsum('ni,i->n', mano_R[:, :, 0], _gz) < 0.0))
            _n_roll_d = int(np.sum(np.einsum('ni,i->n', mano_R[:, :, 1], mano_R[-1][:, 1]) < 0.0))
            logger.info(
                "[Stage2 前缀] 分支 DP 已生效 → 跳过逐帧强制朝下/roll 归一 "
                "(否则会推翻 DP 的分支一致性)。诊断: X 朝上 %d/%d 帧 (均为近水平, |X·g|≤0.18), "
                "Y 反向 %d 帧", _n_up_d, len(mano_R), _n_roll_d)
        if (not _br_dp_on) and os.environ.get("_GRIP_FORCE_DOWN", "1") == "1":
            try:
                import physics_utils as _pu_s2
                _g_s2 = np.asarray(getattr(_pu_s2, "GRAVITY_DIR",
                                           np.array([0.0, 0.0, -1.0])), dtype=np.float64)
                _flip_idx = []
                for _i in range(len(mano_R)):
                    if float(np.dot(mano_R[_i][:, 0], _g_s2)) < 0.0:
                        _flip_idx.append(_i)
                if _flip_idx:
                    mano_R = mano_R.copy()
                    for _i in _flip_idx:
                        mano_R[_i] = _pu_s2.align_gripper_approach_to_gravity(mano_R[_i])
                logger.info(
                    "[Stage2 前缀] 强制朝下: 翻正 %d 帧 (X 原本朝上 ✗) → 现在全段朝下 ✓ | "
                    "翻后仍朝上的帧数=%d %s", len(_flip_idx),
                    sum(1 for _i in range(len(mano_R))
                        if float(np.dot(mano_R[_i][:, 0], _g_s2)) < 0.0),
                    "(幂等: 已朝下的帧未改动 ✓)")
                # ★★★★ 2026-09-14 用户: "要朝下啊, 翻转一下" —— 实测 X 已全朝下 ✓,
                #   但 rot=160.82° ✗ ⇒ **差的是绕 X 的 roll** ✓
                #   ⇒ 闭合轴 Y 被翻了 ≈180° ⇒ 两指左右对调 ⇒ 画面"夹爪反了" ✓✓
                #   修复: 以**末帧**(接锚点)的 Y 为基准, 把整段中以反向的 Y 翻回来 ✓
                #         (绕 X 转 180°: X 不变 ✓, Y→-Y, Z→-Z ✓)
                _Yref = mano_R[len(mano_R) - 1][:, 1].copy()
                _n_roll = 0
                for _i in range(len(mano_R)):
                    if float(np.dot(mano_R[_i][:, 1], _Yref)) < 0.0:
                        mano_R[_i] = mano_R[_i] @ np.diag([1.0, -1.0, -1.0])  # 绕 X 180°
                        _n_roll += 1
                logger.info(
                    "[Stage2 前缀] roll 归一: 以末帧 Y 为基准翻正 %d 帧 (Y 反向 ✗) → "
                    "全段开合轴同向 ✓ (X 保持朝下 ✓)", _n_roll)
            except Exception as _e_s2:  # noqa: BLE001
                logger.warning("[Stage2 前缀] 强制朝下失败 (保持原样): %s", _e_s2)
        offset = getattr(self, "_mano_neutral_offset", {}).get(side)
        if offset is not None:
            mano_pos = mano_pos + np.asarray(offset, dtype=np.float64)

        n_frames = len(mano_pos)
        f_start = max(0, min(n_frames - 1, int(f_start)))
        f_end = max(f_start, min(n_frames - 1, int(f_end)))
        F50_IDX = max(f_start + 1, min(f_end - 1, int(F50_IDX)))

        # ── 距离驱动的切换点 (2026-09-10) ──────────────────────────────────
        # 用户要求: 不再固定 F38/F50, 而是"检测离物体还剩多少距离"来自动决定
        # 何时从 MANO 跟随切到"朝 Stage1 最优 anchor 收敛".
        #   d[f] = |mano_pos[f] - obj_center|
        #   f_start = max{ f < F50_IDX : d[f] >= approach_start_distance }
        # 平移策略保持原样(全帧等量 pre_translate, 不改); 收敛目标仍是 Stage1 anchor.
        if approach_start_distance is not None and float(approach_start_distance) > 0.0:
            _d_stop = float(approach_start_distance)
            _ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
            _tname = _ctrl.target_obj if _ctrl else None
            _obj_center = None
            if _tname is not None:
                _obj_center = getattr(self, "obj_bbox_centers", {}).get(_tname)
            if _obj_center is None and _tname is not None:
                for _actor in self.obj_actors:
                    if _actor.get_name() == _tname:
                        _obj_center = np.asarray(_actor.get_pose().p, dtype=np.float64)
                        break
            if _obj_center is None:
                logger.warning(
                    "[Stage2] 距离驱动切换点不可用 (找不到目标物体), 沿用固定 f_start=%d", f_start)
            else:
                _obj_center = np.asarray(_obj_center, dtype=np.float64)
                _dist = np.linalg.norm(mano_pos - _obj_center[None, :], axis=1)
                _cand = np.where(_dist[:F50_IDX] >= _d_stop)[0]
                _f_auto = int(_cand.max()) if _cand.size else f_start
                _f_auto = max(0, min(_f_auto, F50_IDX - 1))
                logger.info(
                    "[Stage2] 距离驱动切换点: d_stop=%.3fm -> f_start %d -> %d "
                    "(该帧 d=%.4fm, F50 d=%.4fm, 目标=%s)",
                    _d_stop, f_start, _f_auto, _dist[_f_auto], _dist[F50_IDX], _tname)
                f_start = _f_auto
                # ★ 2026-09-21 【不完整-2】把**实际接近起点**记录下来 ✓
                #   审查结论 ✓: "前缀 NameError 修掉了, 但**语义仍错** ✗: 新代码把 `f_start`
                #     默认设为 0 ⇒ `_build_entry_prefix(..., 0)` 会返回 **None** ✗ ⇒ 缺前缀时
                #     仍**无法补出 F0 到接近起点**的轨迹. 应取**实际接近起点** ✓"
                self._stage2_approach_start_frame = int(f_start)
                # ⚠ 2026-09-21 修复 UnboundLocalError ✗: 此处 `recon` **尚未赋值** ✗
                #   (上一轮我写了 `recon["prefix_start_frame"] = ...` ⇒ 直接崩 ✗
                #    实测 geom2 的 Traceback: `local variable 'recon' referenced before assignment` ✗)
                #   ⇒ 只记在 `self` 上 ✓; `rollout_follow_trajectory` 里用
                #     `recon.get("prefix_start_frame", getattr(self, "_stage2_approach_start_frame", 0))`
                #     读取 ✓ (两条路都覆盖 ✓)


        anchor_pos = np.asarray(grasp_pose["pos"], dtype=np.float64).copy()
        enclose_z = grasp_pose.get("enclose_z")
        if enclose_z is not None and np.isfinite(float(enclose_z)):
            enclose_delta = abs(float(anchor_pos[2]) - float(enclose_z))
            if enclose_delta > 0.005:
                logger.warning(
                    "[Stage2] Stage1 pos.z/enclose_z differ by %.1fmm; preserving verified pos.z",
                    enclose_delta * 1000.0,
                )
        anchor_R = np.asarray(grasp_pose["R"], dtype=np.float64).copy()
        q_grasp = float(grasp_pose["gripper_qpos"])
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-10-02 【q_grasp 与 recipe 实际记录**逐位对齐**】 ✓ ★★★★
        #   实测铁证 ✗ (`output/s2_deep6`): 深夹配方实际 `close_ctrl_q` 最小 = **-0.000978**
        #     (受"计划目标−预紧"二次约束限制, 见 md 未完成项 11), 但本处从 `grasp_pose`
        #     取到的是**推导值 -0.0015** ⇒ 下游 `grasp_anchor.gripper_qpos=-0.0015`
        #     与 recipe 值域 [-0.000978, ...] 不符 ⇒
        #     `Stage2 入口 anchor 与 recipe 不符 ⇒ 严格拒绝` ✗ ⇒ 深夹配方进不了 Stage2 ✗
        #   本改 ✓: 有 recipe ⇒ 直接取 recipe 实 `q_cmd` 最小值 ✓
        #     (md §05 原话: "**不得用名义值替换**" ✓ ⇒ 本就该照抄实际记录 ✓)
        #   ⚠ 只改"用哪个 q", **不改任何几何/物理量** ✗ (不放宽任何门 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ⚠⚠ 2026-10-02 **回退** ✗: 曾把 q_grasp 改成 `min(recipe["q_cmd"])` ⇒ 错 ✗
        #   实测 ✗: 三个配方 `q_grasp` 与 `close_ctrl_q` 最小值**本来就不等**
        #     (verify_s31: 0.001899 vs 0.001535; deep5: 0.000537 vs 0.001046) ⇒
        #     `q_grasp` 记的是**名义值**, `close_ctrl_q` 是**实际命令序列**
        #   而两处校验的口径 ✓: ① `anchor == recipe.q_grasp` (3660) ② `anchor ∈ [min(q_cmd), max(q_cmd)]` (1037)
        #     ⇒ 唯一自洽点 = **让 recipe.q_grasp 本身落在实际值域内** ✓
        #     (即修**记录**的错, 而不是改 anchor 的取值 ✓ —— anchor 必须 = recipe.q_grasp ✓)
        #   ⇒ 故此处**保持原样** (anchor = npz `gripper_qpos` = recipe.q_grasp ✓); 修正放在
        #     Stage1 写 recipe 处 / 已有产物的记录修正 ✓
        if q_near is not None:
            raise ValueError("q_near/pre-close is retired: the gripper must stay fully open until F50")
        q_start = stage_q_open()
        approach_min_z = float(approach_min_z)
        if not np.isfinite(approach_min_z):
            raise ValueError("approach_min_z must be finite")
        stage1_force = grasp_pose.get("force_limit")
        if stage1_force is not None and not np.isclose(float(stage1_force), float(GRIPPER_FORCE)):
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 破例开关 (用户 2026-09-29 明确授权 ✓ "这个是我的破例, 你要好好记着") ★★★★
            #   默认 ✓: 仍**严格阻断** (Stage2 必须与 Stage1 同 profile ✓ md 契约 ✓)
            #   `_S2_DIAG_ALLOW_PHYSICS_OVERRIDE=1` ⇒ 只 WARNING 不阻断 ⇒ 允许在 Stage2
            #     用**更大的夹持力/摩擦**做诊断 (命令 q **仍来自 Stage1 实际记录**, 不改 ✗)
            #   ⚠ 性质 ✓: **诊断性放行** —— 结果**不得作为交付基线**;
            #     若据此确认"力/摩擦不足", 正确做法仍是**重跑 Stage1 冻结新配方** ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if os.environ.get("_S2_DIAG_ALLOW_PHYSICS_OVERRIDE", "0") == "1":
                logger.error(
                    "[破例·诊断] ✗ Stage1 force_limit=%.1f ≠ Stage2=%.1f ⇒ "
                    "**按用户授权放行** (_S2_DIAG_ALLOW_PHYSICS_OVERRIDE=1) ✗ "
                    "⚠ 本轮结果**不得作为交付基线** ✓",
                    float(stage1_force), float(GRIPPER_FORCE))
            else:
                raise ValueError(
                    f"Stage1 force_limit={float(stage1_force)} does not match Stage2={GRIPPER_FORCE}"
                )

        delta_p = anchor_pos - mano_pos[F50_IDX]
        delta_R = Rotation.from_matrix(anchor_R) * Rotation.from_matrix(mano_R[F50_IDX]).inv()
        correction_slerp = Slerp(
            [0.0, 1.0], Rotation.concatenate([Rotation.identity(), delta_R])
        )
        approach_start_pos = mano_pos[f_start].copy()

        # ── 轴向逼近规划 (2026-09-11, 用户要求) ──────────────────────────
        # 思路: anchor 仍取 Stage1 最优位姿, 但"怎么来到 anchor"改由物体位姿反解:
        #   1) 沿 anchor 的夹持轴 (gripper X = 手指伸出方向) 后退 L 得到 entry pose;
        #   2) entry → anchor 纯轴向直线推进, 姿态锁定 anchor_R.
        # 这样物体只从张开的虎口正面进入, 不会像 MANO 轨迹那样从侧面扫进手指碰撞体.
        # 后退距离 L 由"物体完全位于手指碰撞盒端面之前"反解得到.
        approach_mode = str(approach_mode or "mano_blend").lower()
        if approach_mode not in ("mano_blend", "axial"):
            raise ValueError(f"unknown approach_mode: {approach_mode!r}")
        axial_info: dict | None = None
        if approach_mode == "axial":
            # 手指碰撞盒 (与 prepare_gripper_only_urdf 同步):
            #   <box size="0.09 0.01 0.04"> @ finger-link 原点偏移 (-0.0045, +/-0.01, 0)
            #   手指 prismatic joint origin x=0.03689, |y|=0.020+q, box 中心再偏 +/-0.01
            _fb_x_max = 0.03689 - 0.0045 + 0.045          # 0.07739 指尖端面
            # ★ 2026-09-13 模型统一: 用 physics_utils 的手指几何唯一真源
            #   (旧写死 0.005 是薄板时代值; 真实指形是 0.0152 → 差 10.2mm 会误判间隙)
            from physics_utils import finger_inner_y as _fiy
            _fb_inner_y = _fiy(q_start)                   # 虎口内侧 |y| = 0.0152 + q
            _ctrl_ax = self.grasp_controllers.get(side) if self.grasp_controllers else None
            _tname_ax = _ctrl_ax.target_obj if _ctrl_ax else None
            _obj_c = getattr(self, "obj_bbox_centers", {}).get(_tname_ax) if _tname_ax else None
            _obj_sz = (
                (getattr(self, "obj_info", {}) or {}).get(_tname_ax, {}).get("bbox_size")
                if _tname_ax else None
            )
            if _obj_c is None or _obj_sz is None:
                raise ValueError("axial 逼近需要目标物体的 bbox_center / bbox_size")
            _obj_c = np.asarray(_obj_c, dtype=np.float64)
            _obj_half = np.asarray(_obj_sz, dtype=np.float64) * 0.5
            _signs = np.array(
                [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
                dtype=np.float64,
            )
            # 物体 8 角点 → anchor 的 gripper 坐标系 AABB (手指盒在该系下轴对齐)
            _corners_g = (_obj_c[None, :] + _signs * _obj_half[None, :] - anchor_pos[None, :]) @ anchor_R
            _g_lo, _g_hi = _corners_g.min(axis=0), _corners_g.max(axis=0)
            _L_axial = float(max(0.02, _fb_x_max - float(_g_lo[0]) + float(axial_clearance_m)))
            _y_extent = float(max(abs(float(_g_lo[1])), abs(float(_g_hi[1]))))
            _y_gap = float(_fb_inner_y - _y_extent)
            axial_info = {
                "mode": "axial",
                "retreat_m": _L_axial,
                "finger_x_max": float(_fb_x_max),
                "finger_inner_y": float(_fb_inner_y),
                "object_aabb_in_anchor": [_g_lo.tolist(), _g_hi.tolist()],
                "object_y_extent": _y_extent,
                "object_y_margin_m": _y_gap,
                "transfer_frac": float(axial_transfer_frac),
            }
            logger.info(
                "[Stage2] 轴向逼近: 后退 %.4fm; 物体在 anchor 系 x=[%.4f,%.4f] vs 手指端面 x=%.4f; "
                "虎口内侧|y|=%.4f vs 物体|y|max=%.4f → 间隙 %+.4fm [%s]",
                _L_axial, _g_lo[0], _g_hi[0], _fb_x_max,
                _fb_inner_y, _y_extent, _y_gap,
                "轴向段零接触" if _y_gap >= 0 else "⚠ 虎口夹住物体(需加大开合或对齐anchor)",
            )

        # axial 模式的路径参数 (entry pose + 姿态 slerp)
        _p_start_ax = mano_pos[f_start].copy()
        _xa = anchor_R[:, 0].copy()                      # gripper X = 夹持轴
        _t_tr = (
            float(min(max(float(axial_transfer_frac), 1e-3), 0.999))
            if axial_info is not None else 1.0
        )
        _p_entry = (
            anchor_pos - axial_info["retreat_m"] * _xa if axial_info is not None else None
        )
        _rot_slerp = (
            Slerp([0.0, 1.0], Rotation.concatenate([
                Rotation.from_matrix(mano_R[f_start]), Rotation.from_matrix(anchor_R)]))
            if axial_info is not None else None
        )

        pos_out: dict[int, np.ndarray] = {}
        rot_out: dict[int, np.ndarray] = {}
        grip_out: dict[int, float] = {}
        alpha_out: dict[int, float] = {}
        safety_out: dict[int, np.ndarray] = {}
        for frame in range(f_start, f_end + 1):
            progress = (frame - f_start) / float(max(F50_IDX - f_start, 1))
            if axial_info is not None and frame < F50_IDX:
                # 轴向逼近: [0, t_tr] 从 MANO 起点平移到 entry pose (姿态 slerp 到 anchor_R),
                # [t_tr, 1] 沿 anchor 的夹持轴纯轴向推进到 anchor.
                t = min(max(progress, 0.0), 1.0)
                if t <= _t_tr:
                    u = _smoothstep5(t / _t_tr)
                    corrected_pos = _p_start_ax + u * (_p_entry - _p_start_ax)
                    corrected_R = _rot_slerp([u])[0].as_matrix()
                else:
                    u_raw = (t - _t_tr) / (1.0 - _t_tr)
                    # ★★★★ P0 修复 (2026-09-30) 【轴向推进段也要**平滑减速**】 ✓ ★★★★
                    #   md 06.2 ✓: "接近末可**平滑减速**至固定 anchor, 不在更早拼接点先停再启" ✓
                    #   实测病根 ✗: 本段原为**线性插值** ⇒ 末端速度**恒定不为零** ⇒
                    #     ready 窗口实测 `pos=14.89mm / linear=234mm/s / angular=416°/s`
                    #     (远超 md 五项容差 1mm/1°/5mm-s/5°-s) ⇒ `PRE_F50_READY` 反复挂 ✗
                    #   本改 ✓: 与第一段的口径一致, 用 smoothstep ⇒ 末端**速度→0** ✓
                    u = _smoothstep5(u_raw)
                    corrected_pos = _p_entry + u * (anchor_pos - _p_entry)
                    corrected_R = anchor_R.copy()
                alpha = 1.0
            else:
                alpha = _smoothstep5(progress) if frame < F50_IDX else 1.0
                corrected_pos = mano_pos[frame] + alpha * delta_p
                corrected_R = (correction_slerp([alpha])[0] * Rotation.from_matrix(mano_R[frame])).as_matrix()

            safety = np.zeros(3, dtype=np.float64)
            command_pos = corrected_pos.copy()
            # Preserve the preprocessed MANO path.  The only permitted world-Z
            # intervention is preventing a pre-F50 command from going below the
            # configured floor (0 by default); no arbitrary lift corridor.
            if frame < F50_IDX and command_pos[2] < approach_min_z:
                safety[2] = approach_min_z - command_pos[2]
                command_pos[2] = approach_min_z
            if frame == F50_IDX:
                command_pos = anchor_pos.copy()
                corrected_R = anchor_R.copy()
                safety[:] = 0.0

            pos_out[frame] = command_pos
            rot_out[frame] = corrected_R
            grip_out[frame] = q_start if frame < F50_IDX else q_grasp
            alpha_out[frame] = alpha
            safety_out[frame] = safety

        recon = {
            "pos": pos_out,
            "R": rot_out,
            "gripper_qpos": grip_out,
            "frames": list(range(f_start, f_end + 1)),
            "alpha_pose": alpha_out,
            "safety_offset": safety_out,
            # ★★★ 2026-09-13 关键修复 (用户: "F22 的停顿很致命" + "开头夹爪有一段相反的移动"):
            #   旧实现前缀用**原始 MANO 位姿** (mano_pos), 而接近段用**轴向逼近校正后**的
            #   位姿 (corrected_pos, 见上方 frame 循环) —— 两者在 f_start 处**不连续**:
            #   前缀末端 = mano_pos[f_start], 接近段起点 = 校正值 → 差一大截
            #   ⇒ PD 为跨越这个跳变先往反方向冲 (开头反向移动), 然后在 F22 处"顶住不动"(停顿)。
            #   修复: 前缀仍**跟随参考轨迹**(保持用户要的"跟原夹爪轨迹"), 但在最后一段平滑
            #   收敛到接近段起点, 使边界 C0 连续。用 smoothstep 渐变 → 两端导数也为 0。
            "preprocessed_prefix": (lambda: (
                lambda _src, _p0, _R0, _p1, _R1, _n: {
                    "source_frame": _src,
                    # 前 70% 完全跟随参考轨迹, 仅最后 30% 平滑收敛到接近段起点
                    # (用户要求"跟随原夹爪轨迹", 所以不能让修正权重覆盖整段前缀)
                    "pos": np.array([
                        _p0[i] + _smoothstep5(
                            max(0.0, min(1.0, (i / max(_n - 1, 1) - 0.7) / 0.3))
                        ) * (_p1 - _p0[i])
                        for i in range(_n)
                    ], dtype=np.float64),
                    # ★★★★ 2026-09-14 用户: "位姿需要考虑平滑的问题" ★★★★
                    #   实测证据 (final_q 日志):
                    #     [Stage2 track] MANO_FOLLOW n=23  pos=1.47/2.92mm
                    #                                       rot=9.33/**169.02deg** ✗✗✗
                    #   ⇒ 前缀段夹爪被命令做 ~169° 的姿态突转, PD 物理上跟不住 →
                    #     实际姿态大幅落后 → 进入接近段时夹爪是"歪着"扫进去的 ✗
                    #   ⇒ 这个突转的来源就在**这里**: 位置收敛目标用了接近段起点 ✓,
                    #     但**姿态收敛目标给成了 mano_R[f_start] (MANO 原样)** ✗,
                    #     而接近段 (axial 模式) 用的是 **anchor_R (Stage1 优化位姿)** ✓
                    #     (见上方注释: "entry → anchor 纯轴向直线推进, 姿态锁定 anchor_R")
                    #     ⇒ 交接点 f_start 处姿态由 MANO 突跳到 anchor_R ✗✗
                    #   修复: 姿态用**与位置完全相同的权重 w** 从 MANO 平滑 SLERP 到 anchor_R ✓
                    #     前 70% 仍完全跟随 MANO (保留"跟随原夹爪轨迹") ✓
                    #     末 30% 位置与姿态**同步**收敛 → C0/C1 连续 ✓
                    "R": np.array([
                        blend_rotation_towards(
                            _R0[i], _R1,
                            _smoothstep5(max(0.0, min(1.0, (i / max(_n - 1, 1) - 0.7) / 0.3))),
                        )
                        for i in range(_n)
                    ], dtype=np.float64),
                }
            )(
                np.arange(0, f_start + 1, dtype=np.float64),
                mano_pos[: f_start + 1].copy(),
                mano_R[: f_start + 1].copy(),
                np.asarray(_p_start_ax if axial_info is not None else mano_pos[f_start],
                           dtype=np.float64),
                # ★ 收敛目标: 接近段真正使用的姿态 (axial=anchor_R, 否则=mano_R[f_start])
                (anchor_R.copy() if axial_info is not None else mano_R[f_start].copy()),
                f_start + 1,
            ))(),
            "approach_waypoints": {
                "start": approach_start_pos,
                "grasp": anchor_pos.copy(),
                "strategy": (
                    "axial_from_object_pose" if axial_info is not None
                    else "mano_relative_correction_smoothstep"
                ),
                "min_world_z": approach_min_z,
            },
            "grasp_anchor": {
                "source_frame": int(F50_IDX),
                "pos": anchor_pos,
                "R": anchor_R,
                "gripper_qpos": q_grasp,
                "q_start": q_start,
                "approach_min_z": approach_min_z,
                "approach_strategy": "mano_relative_correction_smoothstep",
                "contract_hash": _anchor_hash(anchor_pos, anchor_R, q_grasp),
            },
            # Stage1 owns the grasp recipe; Stage2 only reads it (plan §3.1).
            "stage1_recipe": grasp_pose.get("stage1_recipe"),
            "axial_approach": axial_info,
        }
        # 供 run_metadata.json 记录本次 Stage2 的实际配置 (2026-09-11)
        self._stage2_config = {
            "approach_mode": approach_mode,
            "approach_start_distance_m": (
                None if approach_start_distance is None else float(approach_start_distance)
            ),
            "axial_transfer_frac": float(axial_transfer_frac),
            "axial_clearance_m": float(axial_clearance_m),
            "f_start": int(f_start),
            "F50_IDX": int(F50_IDX),
            "f_end": int(f_end),
            "axial": axial_info,
        }
        # Stage2 配置就绪后立刻刷新元数据 (run() 阶段还不知道这些)
        if hasattr(self, "dump_run_metadata"):
            self.dump_run_metadata("stage2")
        # ★ 2026-09-13 N-5 需要目标物体的**真实姿态 + 真实 bbox**:
        #   物体 bbox 是**物体局部系**的, 场景里物体带旋转. 不传姿态时 N-5 把它当
        #   "世界系轴对齐盒" → scale>=0.7 时误差放大到 ~19mm → 报出幻影穿透
        #   (日志: `穿透 19.59mm → 18.77mm`) 并持续偏置基座 → APPROACH_ADJUST
        #   跟踪最大值冲到 18mm (门槛 15mm) → PRE_F50_READY 失败.
        #   注: 这里(名义几何预计算)是**最先**触发 N-5 的地方, 必须一并传入.
        _n5_obj_R = None
        _n5_obj_bb = None
        # ★ 2026-09-21: 目标名先初始化为 None ✓ (凸包模型要用它 ⇒ 避免 try 提前退出时 NameError ✗)
        _tname_n5 = None
        try:
            _ctrl_n5 = (
                self.grasp_controllers.get(side)
                if getattr(self, "grasp_controllers", None) else None
            )
            _tname_n5 = _ctrl_n5.target_obj if _ctrl_n5 else None
            if _tname_n5:
                _n5_obj_bb = (getattr(self, "obj_info", {}) or {}).get(
                    _tname_n5, {}).get("bbox_size")
                for _a in getattr(self, "obj_actors", []) or []:
                    if _a.get_name() == _tname_n5:
                        _n5_obj_R = np.asarray(
                            _a.get_pose().to_transformation_matrix(), dtype=np.float64
                        )[:3, :3]
                        break
        except Exception as _exc:  # noqa: BLE001
            logger.warning("[N-5] 取目标物体姿态/bbox 失败, 回退默认: %s", _exc)
        logger.info(
            "[N-5][名义几何] 目标=%s 姿态=%s bbox=%s",
            locals().get("_tname_n5"),
            "identity" if _n5_obj_R is None else "target_actor",
            None if _n5_obj_bb is None else np.round(np.asarray(_n5_obj_bb, float), 4).tolist(),
        )
        # ★ Stage4: prefix 来自 recon["preprocessed_prefix"] (不是函数参数);
        #   不设置则 timeline 从 f_start 起始 → 视频缺 F0→f_start 段.
        if recon.get("preprocessed_prefix") is None:
            recon["preprocessed_prefix"] = _build_entry_prefix(recon, f_start)
        # ★ 2026-09-21 【不完整-1】名义调用点用的**同一个快照** ✓ (center+R 同源 ✓)
        # ★ 2026-09-22 【复审第 4 条 (c)】把 Stage1 的**冻结动作记录**带进 recon ✓
        #   ⇒ `build_stage2_timeline` 的**三方核对**能真正比到它 ✓ (验证动作 = 交付动作 ✓)
        #   注 ✓: 不写进 recipe ✗ (避免影响 recipe 的 JSON/哈希 ✓), 走 recon 通道 ✓
        try:
            _ar_in = None
            if isinstance(grasp_pose, dict):
                _ar_in = grasp_pose.get("action_record")
            if _ar_in is None:
                _ar_in = getattr(self, "_stage1_action_record", None)
            if isinstance(_ar_in, dict):
                recon["action_record"] = _ar_in
                logger.info("[P0-4] ✓ 冻结动作记录已带入 recon (q_cmd=%s, close/settle=%s/%s) "
                            "⇒ Stage2 将核对「验证动作 = 交付动作」 ✓",
                            _ar_in.get("q_cmd"), _ar_in.get("close_controls"),
                            _ar_in.get("settle_controls"))
        except Exception as _ear5:  # noqa: BLE001
            logger.warning("[P0-4] 动作记录带入 recon 失败: %s", _ear5)
        _snap_c_c, _snap_c_R = self._object_state_snapshot(locals().get("_tname_n5"))
        # ★ 2026-09-22 【P1-2】名义 bundle 采用**唯一权威时间缩放** ✓ (与 rollout 同源 ✓)
        _ts_a, _ts_c = stage2_timeline_scales()
        logger.info("[Stage2] nominal bundle 时间缩放 (唯一来源 ✓): approach=%.3f carry=%.3f "
                    "(★ 与 rollout 调用点同源 ⇒ 同一条轨迹 ✓; 审计 §P1-2 ✓)", _ts_a, _ts_c)
        recon["bundle"] = build_stage2_timeline(
            recon, F50_IDX=F50_IDX, q_start=q_start, recipe=recon["stage1_recipe"],
            object_bbox_size=_n5_obj_bb,
            # ★ 2026-09-21 【用户原则】规划几何 = 物理同款凸包 ✓ (名义调用点也必须传 ✓)
            target_hulls=self._n5_target_hulls(_tname_n5),
            geometry_status=getattr(self, "_n5_geom_status", None),
            # ★ 2026-09-13: object_R 暂不传 —— 名义几何(L844)与主 rollout(L1409)之间
            #   物体静置过, 姿态不同 → 两次 N-5 偏移不同 → bundle_hash 不一致 →
            #   Stage3 `CONTRACT: bundle_hash mismatch`. 待把"静置后姿态"在加载场景时
            #   采集一次并全阶段共享 (STAGE_ISSUES 候选①) 后再启用.
            # ★★★★ 2026-09-21 【问题1/3 补充】把这条注释的**根因**说清 ✓ ★★★★
            #   它其实是**问题1 的深层后果** ✓: "两次调用之间物体**静置过、姿态不同**" ✗
            #   ⇒ 同一物体在链路里存在**两个状态** ⇒ N-5 偏移不同 ⇒ hash 不一致 ✗
            #   ⇒ 正解 = 用户建议: "**Stage1/2 使用同一个已验证的物体初始状态**" ✓
            #     (在加载场景时采集一次 `T_W_O` 并**全阶段共享** ✓) ⇒ 本开关
            #     `_S2_N5_SHARE_R=1` 才传真实姿态 ✓ (默认 0 = 保持既有 hash 行为 ✓)
            # ★★ 2026-09-21 【不完整-1 修复】名义调用点也消费**同一个快照** ✓
            #   审查结论 ✓: "名义 bundle **默认不传旋转** ✗, rollout bundle 则**传旋转** ⇒
            #     两次规划输入仍**不一致**" ✗ (⇒ 还会导致 `bundle_hash` 不一致 ✗)
            #   现 ✓: 两处都由 `_object_state_snapshot()` 提供 (center+R **同源** ✓),
            #     默认**共享** ✓ (`_S2_N5_SHARE_R=0` 可回到旧行为 ✗)
            object_R=(_snap_c_R if os.environ.get("_S2_N5_SHARE_R", "1") == "1" else None),
            # ★ 2026-09-21 【问题3】中心改用**引擎真值快照** ✓ (实测反推值 z 偏 **6.1mm** ✗:
            #   反推 (-0.3032,-0.4958,**0.0257**) vs 真值 (-0.3022,-0.4945,**0.0196**) ✓)
            object_center=(_snap_c_c if _snap_c_c is not None
                           else np.full(3, np.nan, dtype=np.float64)),
            # ★ 2026-09-22 【P1-2 修复】名义调用点补齐**与 rollout 相同的显式时间缩放** ✓
            #   上一版直接引用 `approach_time_scale` ✗ ⇒ `NameError` ✗ (已回退 ✓)
            #   正解 ✓: 取**模块级唯一权威来源** `stage2_timeline_scales()` ✓
            #     (两处调用点都从这里取 ⇒ 默认输入一致 ⇒ 同一条轨迹 ✓; 审计 §P1-2 ✓)
            approach_time_scale=_ts_a,
            carry_time_scale=_ts_c,
        )
        logger.info(
            "[Stage2] nominal geometry F%d-F%d: fixed relative correction after F%d, "
            "smooth MANO correction, min_z=%.0fmm, q_start=%.4f, q_grasp=%.4f, bundle=%s",
            f_start,
            f_end,
            F50_IDX,
            approach_min_z * 1000.0,
            q_start,
            q_grasp,
            recon["bundle"]["bundle_hash"][:12],
        )
        return recon

    @staticmethod
    def _find_target_actor(obj_actors, target_name):
        for actor in obj_actors:
            if actor.get_name() == target_name:
                return actor
        for actor in obj_actors:
            if target_name and target_name in actor.get_name():
                return actor
        return None

    @staticmethod
    def _configure_stage1_target_isolated_collisions(robot, side, obj_actors, target_name,
                                                     scene=None):
        """Apply the plan §3.4 ``stage1_target_isolated`` collision matrix.

        SAPIEN g0/g1 are contact-type/contact-affinity masks.  Gripper, target
        and non-target GLBs get distinct types:

        * gripper ↔ target              ON
        * gripper ↔ ground/default      ON  (reuses Stage1 real physics)
        * target  ↔ ground/default      ON  (reuses Stage1 real physics)
        * gripper ↔ non-target          OFF (isolated this round)
        * target  ↔ non-target          OFF (isolated this round)
        * non-target ↔ ground/default   ON
        * non-target ↔ non-target       ON

        Only the two disabled pairs count as a collision-filter leak later on;
        ground contact is genuine physics and must not be reported as leakage.

        ★★ 任务02 (2026-09-28) 修**实现 ≠ 规格** ✗ ★★
          上面 4 条 "ON" 里, "gripper ↔ ground" 与 "target ↔ ground" 此前**根本没被实现** ✗ ——
          本函数只设置了 `gripper` / `target` / `non_target` 三类 shape, **从未触碰 ground**
          ⇒ ground 保留场景创建时的 `(type=1, affinity=1)` ⇒ 按 PhysX **双向**判定
            `(g_target=4 & aff_ground=1) = 0` ⇒ **ON 变 OFF** ✗
          实测铁证 ✓ (`output/stage2_t02d`, `_T02_DUMP_CHANNELS=1`):
            `ground → (1, 1)` ✗ | `glb_4 → (4, 7)` ✓ | finger → `(2, 5)` ✓
            ⇒ 合同门 `通道 target_table 期望=True 实际=False` / `fingers_table …False` ✗
            ⇒ **Stage2 每一步都进不去** ✗
          本改 ✓: 显式按 `masks["default"]` 设置**地面/桌面** shape
            (该表已补全为 `(1<<0, bit0|bit1|bit2|bit3)` ⇒ 与上面 4 条 ON 完全一致 ✓);
            "non_target ↔ ground" 也同时恢复 ✓ (否则非目标会掉穿)
          ⚠ 这是**打开本该打开的通道** ✓, 不是放宽阈值 ✗ (md §02)
        """
        import sapien

        masks = STAGE1_TARGET_ISOLATED_CONTACT_MASKS
        gripper_type, gripper_affinity = masks["gripper"]
        target_type, target_affinity = masks["target"]
        non_target_type, non_target_affinity = masks["non_target"]
        # 实验旋钮 (2026-09-10): _NO_GRIPPER_GROUND=1 时去掉夹爪↔ground/默认环境的碰撞.
        # 用于验证"Stage1 的成功是否依赖手指穿地面". 作用域同时作用于 Stage1/Stage2.
        _no_gripper_ground = os.environ.get("_NO_GRIPPER_GROUND", "0") == "1"
        if _no_gripper_ground:
            gripper_affinity = int(gripper_affinity) & ~(1 << 0)
        gripper_links = {
            f"{side}_gripper_link",
            f"{side}_gripper_finger_link1",
            f"{side}_gripper_finger_link2",
        }
        gripper_shapes = 0
        target_shapes = 0
        non_target_shapes = 0

        for link in robot.get_links():
            if link.get_name() not in gripper_links:
                continue
            for component in link.entity.components:
                if not isinstance(
                    component, sapien.pysapien.physx.PhysxArticulationLinkComponent
                ):
                    continue
                if not hasattr(component, "get_collision_shapes"): continue
                for shape in component.get_collision_shapes():
                    groups = list(shape.get_collision_groups())
                    groups[0] = gripper_type
                    groups[1] = gripper_affinity
                    shape.set_collision_groups(groups)
                    gripper_shapes += 1

        for actor in obj_actors:
            is_target = actor.get_name() == target_name or (
                target_name and target_name in actor.get_name()
            )
            actor_type = target_type if is_target else non_target_type
            affinity = target_affinity if is_target else non_target_affinity
            for component in actor.get_components():
                if not isinstance(component, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                    continue
                for shape in component.get_collision_shapes():
                    groups = list(shape.get_collision_groups())
                    groups[0] = actor_type
                    groups[1] = affinity
                    shape.set_collision_groups(groups)
                    if is_target:
                        target_shapes += 1
                    else:
                        non_target_shapes += 1

        # ★ 任务02 (2026-09-28) ✓: **地面/桌面** shape 按 `masks["default"]` 设置 ——
        #   这是上面 docstring 四条 "ON" 里缺失的两条 (gripper↔ground / target↔ground) ✓
        #   实现方式: 只调 **affinity(mask)** 使其覆盖 gripper/target/non_target 的 type,
        #   保证 PhysX 双向判定 `(g&m) and (g'&m')` 成立 ✓
        default_type, default_affinity = masks["default"]
        ground_shapes = 0
        _acts = []
        try:
            if scene is not None:
                from physics_env import _scene_actors as _sa
                _acts = _sa(scene)
        except Exception:  # noqa: BLE001
            _acts = []
        for actor in _acts or []:
            try:
                _an = str(actor.get_name()).lower()
            except Exception:  # noqa: BLE001
                continue
            if ("ground" not in _an) and ("table" not in _an):
                continue
            for component in actor.get_components():
                # ★ 2026-09-28 修 (md §02 "不兼容的场景读取 API" 实例 ✗) ✓:
                #   实测 ✗: `RenderBodyComponent` **没有** `get_collision_shapes()`
                #     ⇒ `--test-stage1-transfer` 在碰撞过滤第一步就 AttributeError 崩 ✗
                #   本改 ✓: 只对**有碰撞接口**的组件操作 (渲染组件跳过 ✓)
                if not hasattr(component, "get_collision_shapes"):
                    continue
                for shape in component.get_collision_shapes():
                    groups = list(shape.get_collision_groups())
                    groups[0] = default_type
                    groups[1] = default_affinity
                    shape.set_collision_groups(groups)
                    ground_shapes += 1
        logger.info(
            "[T02] 地面/桌面 shape 按 masks['default']=(%d, %d) 设置: **%d** 个 "
            "(md §02: 目标↔地面 / 指尖↔地面 必须 ON ✓)",
            int(default_type), int(default_affinity), int(ground_shapes))

        if not gripper_shapes or not target_shapes:
            raise RuntimeError(
                "stage1_target_isolated collision filter requires gripper and target "
                "collision shapes"
            )
        config = {
            "scope": TARGET_ISOLATED_SCOPE,
            "gripper_ground_pair": not _no_gripper_ground,
            "gripper_non_target_pair": False,
            "gripper_target_pair": True,
            "target_environment_pair": True,
            "target_non_target_pair": False,
            "non_target_environment_pair": True,
            "non_target_non_target_pair": True,
            "gripper_shapes": int(gripper_shapes),
            "target_shapes": int(target_shapes),
            "non_target_shapes": int(non_target_shapes),
            "target_name": str(target_name),
            "no_gripper_ground_env": bool(_no_gripper_ground),
        }
        logger.info(
            "[Stage2 collision scope] %s: gripper=%d target=%d non-target=%d shapes",
            TARGET_ISOLATED_SCOPE,
            gripper_shapes,
            target_shapes,
            non_target_shapes,
        )
        return config

    # Backwards-compatible alias.
    _configure_target_only_collisions = _configure_stage1_target_isolated_collisions

    def _dump_collision_audit(self, tag, side="left", target_name="", robot=None):
        """★ md 06.4-e2 ✓: 关键采样点落盘 ``collision_audit.<tag>.json`` (FAIL 也保存 ✓)

        ⚠ 2026-09-30 修 ✗: 本函数首版用 `self.robot` —— **Stage2 根本没这个属性**
          (机器人是 `self.robot_info["robot"]`) ⇒ robot=None ⇒ `fingers=0` ⇒
          `target_fingers=None` ⇒ 审计**误报"手指通道不可验证"** ✗
          (e2 审计本身立刻暴露了这个调用错误 ✓ —— 正是它该干的事 ✓)
        本改 ✓: 优先用调用点传入的 robot, 回退 `self.robot_info["robot"]` ✓
        """
        try:
            import json as _j
            from pathlib import Path as _P
            from physics_env import build_collision_audit
            _rb = robot
            if _rb is None:
                _rb = (getattr(self, "robot_info", None) or {}).get("robot")
            _aud = build_collision_audit(
                robot=_rb, side=side,
                obj_actors=getattr(self, "obj_actors", None),
                target_name=target_name, scene=getattr(self, "scene", None),
                tag=str(tag))
            _dir = _P(str(getattr(self, "output_dir", "output"))) / "stage2"
            _dir.mkdir(parents=True, exist_ok=True)
            _p = _dir / ("collision_audit.%s.json" % str(tag))
            _p.write_text(_j.dumps(_aud, indent=2, default=str), encoding="utf-8")
            logger.warning("[e2] OK collision_audit (%s): %s | per_finger=%s | identity=%s",
                           tag, _p.name, (_aud.get("per_finger") or {}),
                           (_aud.get("identity", {}) or {}).get("target_actor_info"))
            return _aud
        except Exception as _ea:  # noqa: BLE001
            logger.warning("[e2] collision_audit 落盘失败(不阻断): %s", _ea)
            return None

    @staticmethod
    def _gripper_pose_matrix(robot, side):
        preferred = f"{side}_gripper_base_link"
        fallback = None
        for link in robot.get_links():
            name = link.get_name()
            if name == preferred:
                return np.asarray(link.get_entity_pose().to_transformation_matrix(), dtype=np.float64)
            if fallback is None and "gripper_base_link" in name:
                fallback = link
        return (
            np.asarray(fallback.get_entity_pose().to_transformation_matrix(), dtype=np.float64)
            if fallback is not None
            else None
        )

    @staticmethod
    def _target_finger_contacts(robot, side, scene, obj_actors, target_name):
        """Return true per-finger contacts; palm contact never counts as bilateral."""
        import sapien

        link_bodies = {}
        robot_bodies = set()
        accepted_links = {
            f"{side}_gripper_finger_link1",
            f"{side}_gripper_finger_link2",
            f"{side}_gripper_link",
        }
        for link in robot.get_links():
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    robot_bodies.add(component)
            if link.get_name() not in accepted_links:
                continue
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    link_bodies[component] = link.get_name()

        object_bodies = {}
        for actor in obj_actors:
            for component in actor.components:
                if isinstance(component, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                    object_bodies[component] = actor.get_name()
                    break

        finger1 = False
        finger2 = False
        contact_objects = set()
        contact_links = set()
        def body_name(body) -> str:
            name = str(getattr(body, "name", "") or "")
            entity = getattr(body, "entity", None)
            if not name and entity is not None:
                name = str(getattr(entity, "name", "") or "")
            return name or type(body).__name__

        for contact in scene.get_contacts():
            if len(contact.bodies) < 2:
                continue
            body0, body1 = contact.bodies[:2]
            if body0 in link_bodies:
                link_body, other_body = body0, body1
            elif body1 in link_bodies:
                link_body, other_body = body1, body0
            else:
                continue
            if other_body in robot_bodies:
                continue
            link_name = link_bodies[link_body]
            object_name = object_bodies.get(other_body, body_name(other_body))
            contact_links.add(link_name)
            contact_objects.add(object_name)
            is_target = object_name == target_name or (target_name and target_name in object_name)
            if is_target and link_name.endswith("finger_link1"):
                finger1 = True
            elif is_target and link_name.endswith("finger_link2"):
                finger2 = True
        return finger1, finger2, sorted(contact_objects), sorted(contact_links)

    def _coacd_world_hulls_stage2(self, target_name):
        """目标物体在**世界系**的 CoACD 凸包顶点列表 (Stage2 侧).

        2026-09-18: 为"不要碰物体"的几何量提供数据 (用户定调 ✓).
        读的是**引擎里该 actor 实际使用的 convex shapes** ✓ ⇒ 与物理完全一致 ✓
        取不到 ⇒ 返回 [] ⇒ 调用方跳过 ✓ (未启用 CoACD 时零影响 ✓)
        """
        import sapien as _sp

        actors = list(getattr(self, "obj_actors", []) or [])
        for _a in actors:
            try:
                _n = _a.get_name()
            except Exception:  # noqa: BLE001
                continue
            if not target_name or (_n != target_name and target_name not in _n):
                continue
            _H = np.asarray(_a.get_pose().to_transformation_matrix(), dtype=np.float64)
            _out = []
            for _c in _a.components:
                if not isinstance(_c, _sp.pysapien.physx.PhysxRigidDynamicComponent):
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

    def _object_state_snapshot(self, target_name):
        """★★★ 2026-09-21 【P0-1 / 不完整-1】**唯一的物体状态快照** ✓ (center + R 同一时刻同源) ★★★

        静态审查结论 ✓ (用户 2026-09-21):
          "『引擎真值中心』实际上**仍来自 `obj_bbox_centers` 缓存** ✗; 旋转却取自 **reset 前**
           的 actor ⇒ 可能把**不同时间**的中心和旋转组合 ✗; 名义 bundle **默认不传旋转**,
           rollout bundle 则**传旋转** ⇒ 两次规划输入仍**不一致** ✗。
           应统一消费**同一个物体状态快照**" ✓

        本实现 ✓: 以加载时采集的 `_obj_initial_poses` 为**单一来源** ✓
          · `center` = `obj_bbox_centers[target]` (与 `_obj_initial_poses` **同源** ✓)
          · `R`      = 同一份位姿的旋转 ✓
          ⇒ 两个调用点消费**同一个** (center, R) ✓ ⇒ 规划输入一致 ✓ (bundle_hash 也随之稳定 ✓)

        Returns: ``(center(3,), R(3,3))``; 取不到 ⇒ ``(None, None)`` ✓
        """
        _c = (getattr(self, "obj_bbox_centers", {}) or {}).get(target_name)
        _ip = (getattr(self, "_obj_initial_poses", {}) or {}).get(target_name)
        center = (np.asarray(_c, dtype=np.float64).reshape(3)
                  if _c is not None else None)
        R = None
        try:
            if _ip is not None:
                R = np.asarray(_ip.to_transformation_matrix(), dtype=np.float64)[:3, :3]
        except Exception:  # noqa: BLE001
            R = None
        if getattr(self, "_obj_state_snapshot_logged", False) is not True:
            logger.info("[Stage2 物体快照] 单一来源 ✓ center=%s R[:2]=%s (center 与 R **同源** ✓)",
                        None if center is None else np.round(center, 4).tolist(),
                        None if R is None else np.round(R[:2], 3).tolist())
            self._obj_state_snapshot_logged = True
        return center, R

    def _scene_state_snapshot(self, target_name, robot=None):
        """★ 2026-09-22 【P0-3】**规范场景状态快照** ✓ (审计 §P0-3 ✓)

        审计要求 ✓: "建立**共享 reset/settle 契约**, 冻结 `T_W_O`、物体速度、机器人
          qpos/qvel、驱动目标、**collision 局部几何及变换**; 清晰区分 actor 原点与几何中心"
          "合理顺序: 加载并静置 → **采集规范场景状态** → Stage1/Stage2 从该状态复位并核验
           → **从该快照生成规划几何**; **不要先规划, 再改变物体参考状态**" ✓
        本函数 ✓: 采集**一份**规范快照 (对象位姿/速度/**局部碰撞几何** + 机器人 qpos/qvel/驱动 ✓)
          ⇒ `_coacd_world_hulls_from_snapshot()` 用它生成规划几何 ✓
            ⇒ 规划几何与执行初态**同源** ✓ (消除"reset 前读实时位姿"✗)
        """
        # ⚠⚠ 2026-09-22 【复审修正 — 空快照 ✗】⚠⚠
        #   复审第 2 条 ✓: "缺少 `sapien` 导入导致**空快照**" ✗
        #   实测根因 ✓: 本模块**没有** `import sapien` ✗ (只在 `_contact_detail_pairs`
        #     里局部导入 ✓) ⇒ 下面所有 `sapien.pysapien...` 抛 NameError ✗
        #     ⇒ 被外层 `except` 吞掉 ⇒ **每个物体都被 continue 跳过** ⇒ 快照为空 ✗
        import sapien  # noqa: PLC0415
        snap = {"target_name": target_name, "objects": {}, "robot": None}
        # ⚠ 2026-09-22 修 BUG ✓: 实测 `p03_snapshot` 里 **物体数=0** ✗
        #   (`[P0-3 规范快照] 已采集 ✓ 物体数=0 | 目标 glb_4: p=[nan,nan,nan] | 局部凸包片数=0` ✗)
        #   根因 ✓: 该上下文里 `self.obj_actors` **为空** ✗ (与 Stage1 不同实例 ✗)
        #   ⇒ 回退到**场景全部 actor** ✓ (只取含**动态组件**的 ✓)
        _actors = list(getattr(self, "obj_actors", []) or [])
        if not _actors:
            try:
                _actors = [a for a in self.scene.get_all_actors()
                           if any(isinstance(c, sapien.pysapien.physx.PhysxRigidDynamicComponent)
                                  for c in a.components)]
                logger.warning("[P0-3 快照] `obj_actors` 为空 ⇒ 回退**场景全部动态 actor** ✓ (%d 个 ✓)",
                               len(_actors))
            except Exception as _ea:  # noqa: BLE001
                logger.error("[P0-3 快照] 场景 actor 枚举失败: %s ✗", _ea)
                _actors = []
        for actor in _actors:
            try:
                _nm = actor.get_name()
                _p = actor.get_pose()
                _e = {"p": [float(v) for v in _p.p[:3]],
                      "q": [float(v) for v in _p.q[:4]]}
                for comp in actor.components:
                    if isinstance(comp, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                        try:
                            _e["v"] = [float(v) for v in comp.get_linear_velocity()[:3]]
                            _e["w"] = [float(v) for v in comp.get_angular_velocity()[:3]]
                        except Exception:  # noqa: BLE001
                            pass
                        # ★ 局部碰撞几何 ✓ (= 物理真正在用的那组顶点 ✓ 供规划 ✓)
                        if not hasattr(comp, "get_collision_shapes"): continue
                        try:
                            _e["local_hulls"] = [
                                np.asarray(_sh.get_vertices(), dtype=np.float64).tolist()
                                for _sh in comp.get_collision_shapes()]
                        except Exception:  # noqa: BLE001
                            pass
                        break
                snap["objects"][_nm] = _e
            except Exception:  # noqa: BLE001
                continue
        _rb = robot if robot is not None else getattr(self, "robot", None)
        if _rb is not None:
            try:
                snap["robot"] = {
                    "qpos": [float(v) for v in _rb.get_qpos()],
                    "qvel": [float(v) for v in _rb.get_qvel()],
                    "drive_targets": [float(j.get_drive_target())
                                      for j in _rb.get_active_joints()],
                }
            except Exception as _er:  # noqa: BLE001
                logger.warning("[P0-3 快照] 机器人状态采集失败: %s", _er)
        return snap

    def _ensure_scene_snapshot(self, target_name, force: bool = False):
        """首次规划前采集**唯一**快照 ✓ (`force=True` 可强制重采 ✓)

        ★★★ 2026-09-24 【方案 §3.3 / 任务 3】执行顺序必须是: **先静置 → 再快照 → 再规划** ✓ ★★★
          方案原话 ✓: \"执行顺序固定为: 加载 P1 → 物体**无夹爪干扰地落稳** → **保存规范状态**
            → 从该状态**生成规划几何** → reset 到该状态 → 执行动作\" ✓
          ⇒ 旧行为 ✗: 可能在物体**尚未落稳/仍在抖动**时就采快照 ⇒ 规划几何与执行初态
            不是同一状态 ✗ (正是 P0-3 要修的事 ✓)
          本改 ✓: 采集前先做**静置步进** (`_SNAP_SETTLE_STEPS`, 默认 **0 = 保持旧行为** ✓,
            实验设 60~120 ✓), 并记录静置**前后**的速度 ⇒ 未静止会**明确告警** ✗
            (不声称"已静置" ✗ —— 与方案"未知不得宣称"同一条 ✓)
        """
        if force or getattr(self, "_scene_snapshot", None) is None:
            _settle_n = int(os.environ.get("_SNAP_SETTLE_STEPS", "0") or 0)
            if _settle_n > 0:
                try:
                    def _vel_of(_tn):
                        try:
                            for _a in (getattr(self, "obj_actors", None) or []):
                                if str(getattr(_a, "name", "")) == str(_tn):
                                    _v = _a.get_velocity()
                                    return float(np.linalg.norm(np.asarray(_v, float)))
                        except Exception:  # noqa: BLE001
                            return float("nan")
                        return float("nan")
                    _v0 = _vel_of(target_name)
                    for _k in range(_settle_n):
                        self.scene.step()
                    _v1 = _vel_of(target_name)
                    logger.info("[P0-3 规范快照] ★ 静置 %d 步 ✓ (方案 §3.3: 先落稳再采 ✓) "
                                "| 目标速度 %.4f → %.4f m/s %s",
                                _settle_n, _v0, _v1,
                                "✓ 已静止" if (np.isfinite(_v1) and _v1 < 1e-3)
                                else "⚠ **仍未静止** ⇒ 快照可能非规范状态 ✗")
                except Exception as _es:  # noqa: BLE001
                    logger.warning("[P0-3 规范快照] 静置失败 (不阻断): %s", _es)
            self._scene_snapshot = self._scene_state_snapshot(target_name)
            _o = (self._scene_snapshot.get("objects") or {}).get(target_name) or {}
            # ★ 2026-09-22 修 BUG ✓: 目标**不在快照**时不能记成"已采集 ✓" ✗
            #   (实测曾出现 `物体数=0 | 目标 p=[nan,nan,nan] | 局部凸包片数=0` 却打 INFO ✓ ✗)
            if not _o.get("p"):
                logger.error(
                    "[P0-3 规范快照] ✗ **目标 %s 不在快照中** (快照物体数=%d) ⇒ "
                    "规划几何**无法**从快照生成 ✗ ⇒ 将回退实时位姿 ⚠ (审计 §P0-3)",
                    target_name, len(self._scene_snapshot.get("objects") or {}))
            logger.info(
                "[P0-3 规范快照] 已采集 ✓ 物体数=%d | 目标 %s: p=%s v=%s w=%s | "
                "局部凸包片数=%s | 机器人 qpos/qvel/驱动=%s",
                len(self._scene_snapshot.get("objects") or {}), target_name,
                np.round(_o.get("p") or [np.nan] * 3, 4).tolist(),
                np.round(_o.get("v") or [np.nan] * 3, 5).tolist(),
                np.round(_o.get("w") or [np.nan] * 3, 5).tolist(),
                len(_o.get("local_hulls") or []),
                (self._scene_snapshot.get("robot") is not None))
        return self._scene_snapshot

    def restore_scene_snapshot(self, robot=None) -> dict:
        """★ T08-D2 (2026-09-27) 【把**冻结快照**的姿态恢复到场景 (执行起点)】 ✓

        为什么必须做 ✓ (md ✓: "加载 P1 → 物体无夹爪干扰地落稳 → **保存规范状态** →
          **从该状态**生成规划几何 → **reset 到该状态** → 执行动作") ✓
        实测病根 ✗ (s39): `[P0-3 快照核验] ✗ 当前场景与冻结快照不一致
          (Δpos=**191.603mm**, Δrot=**33.374°**, Δv=263.23mm/s)` ⇒ 规划输入与执行初态**不同源** ✗
          ⇒ 与 CARRY 门读数**同值**: `T_G_O drift=5.56mm/**33.37°**` ✗
          ⇒ 即"33° 不是指间滑移" ✓ 而是**快照↔执行**位姿差 ✗ ⇒ 必须**按快照恢复** ✓

        ⚠ 只恢复**物体**(场景状态) ✓: 机器人由既有 reset 负责 (qpos/root 已在别处重置 ✓);
          不放松任何门 ✓、不做"机器人位姿纠正" ✗ (md: 不得 set_pose 纠正 ✓)
        返回 ✓: `{"n_restored": n, "n_missing": m}` ✓
        """
        _snap = getattr(self, "_scene_snapshot", None)
        _out = {"n_restored": 0, "n_missing": 0}
        if not isinstance(_snap, dict):
            return _out
        _objs = _snap.get("objects") or {}
        try:
            for _a in (getattr(self, "obj_actors", None) or []):
                _nm = str(getattr(_a, "get_name", lambda: "")())
                _ref = None
                for _k, _v in _objs.items():
                    if str(_k) == _nm or str(_k) in _nm:
                        _ref = _v
                        break
                if not isinstance(_ref, dict) or not _ref.get("p"):
                    _out["n_missing"] += 1
                    continue
                _p = np.asarray(_ref.get("p"), dtype=np.float64).reshape(-1)[:3]
                _q = _ref.get("q") or _ref.get("quat")
                if _q is None and _ref.get("R") is not None:
                    _R = np.asarray(_ref.get("R"), dtype=np.float64).reshape(3, 3)
                    _w = float(np.sqrt(max(0.0, 1.0 + float(_R[0, 0]) + float(_R[1, 1]) + float(_R[2, 2]))) / 2.0)
                    if _w > 1e-8:
                        _q = [float(_w), float((_R[2, 1] - _R[1, 2]) / (4.0 * _w)),
                              float((_R[0, 2] - _R[2, 0]) / (4.0 * _w)),
                              float((_R[1, 0] - _R[0, 1]) / (4.0 * _w))]
                    else:
                        _q = [1.0, 0.0, 0.0, 0.0]
                _q = [float(x) for x in (np.asarray(_q, dtype=np.float64).reshape(-1)[:4])] \
                    if _q is not None else [1.0, 0.0, 0.0, 0.0]
                import sapien as _sp_d2
                _a.set_pose(_sp_d2.Pose(_p.tolist(), _q))
                # ★ T08-D4b ✓ **单一真源**: 让每轮 reset 用的 `_obj_initial_poses` 也**取自冻结快照** ✓
                #   实测疑点 ✗: CARRY 的 `T_G_O drift=5.56mm/**33.36~33.37°**` 在五轮里**完全不变** ✗
                #     且与 P0-3 的 Δrot=33.374° 同值 ⇒ 高度怀疑是"**两处记录的对象姿态不同源**" ✗
                #     (快照 vs `_obj_initial_poses` ⇒ 每轮 repeat 从不同初态起步 ⇒ 追踪被混读 ✗)
                #   ⇒ 本改 ✓: 恢复后就地把 `_obj_initial_poses[name]` 对齐到快照姿态 ✓
                #     ⇒ 快照 / reset 初态 / 规划输入 **同一个源** ✓ (md: "reset 到该状态" ✓)
                try:
                    _d2b = getattr(self, "_obj_initial_poses", None)
                    if isinstance(_d2b, dict):
                        _d2b[_nm] = _sp_d2.Pose(_p.tolist(), _q)
                except Exception:  # noqa: BLE001
                    pass
                for _c in _a.get_components():
                    if hasattr(_c, "set_linear_velocity"):
                        _c.set_linear_velocity([0, 0, 0])
                        _c.set_angular_velocity([0, 0, 0])
                        break
                _out["n_restored"] += 1
            logger.warning("[T08-D2] ✓ 已按**冻结快照恢复**场景: 恢复 %d 个物体 (缺 %d) "
                           "⇒ 执行初态与规划输入**同源** ✓ (期望 Δpos/Δrot→≈0 ✓)",
                           int(_out["n_restored"]), int(_out["n_missing"]))
        except Exception as _e_d2:  # noqa: BLE001
            logger.warning("[T08-D2] 快照恢复失败(不阻断 ✓): %s", _e_d2)
        return _out

    def assert_scene_snapshot(self, target_name, tol_pos_mm: float = 0.5,
                              tol_rot_deg: float = 1.0, tol_vel_mm_s: float = 1.0,
                              robot=None, raise_on_fail: bool = False) -> dict:
        """★ 2026-09-22 【P0-3 验收】当前场景 vs **冻结快照** 的一致性核验 ✓

        审计验收 ✓: "同一动作连续重放 5 次; 每次**首个有效控制前**的位姿、速度均满足
          预先定义的容差, **规划凸包世界坐标与执行初态匹配**" ✓ —— 本函数即该判据 ✓
        返回 ✓: dict (含各分项最大偏差 + `ok` ✓); `raise_on_fail=True` ⇒ 不满足即抛 ✗
        """
        _snap = self._ensure_scene_snapshot(target_name)
        _ref = (_snap.get("objects") or {}).get(target_name) or {}
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【复审 J 修复】**空快照不得返回 `ok=True`** ✗ ★★★★
        #   复审原话 ✓: "**空快照核验仍返回 `ok=True`**, 本次重新直接调用已确认;
        #     `assert_scene_snapshot()` 仍**没有实际调用点**" ✗✗
        #   根因 ✓: 旧写法初始化 `_out = {"ok": True, ...}` ✓, 而对象表为空时
        #     下面的 for 循环**一次都不执行** ⇒ 所有"最大偏差"保持 0.0 ⇒ **判为通过** ✗
        #   本改 ✓ (与"**缺测不得通过**"同一条原则 ✓ 复审 I.2-4 ✓):
        #     快照里**没有目标物** ⇒ `ok=False` + `reason` ✓ (**绝不默认通过** ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        if not _ref:
            logger.error(
                "[P0-3 快照核验] ⚠ 冻结快照中**没有目标物 `%s`** (objects=%s) ⇒ "
                "**核验不可信 ⇒ ok=False** ✗ (复审 J ✓: 空快照不得判通过 ✗)",
                target_name, sorted((_snap.get("objects") or {}).keys())[:6])
            return {"ok": False, "target": target_name, "max_pos_mm": None,
                    "max_rot_deg": None, "max_vel_mm_s": None,
                    "reason": "empty_snapshot", "n_objects": 0}
        _out = {"ok": True, "target": target_name, "max_pos_mm": 0.0,
                "max_rot_deg": 0.0, "max_vel_mm_s": 0.0,
                "n_objects": len(_snap.get("objects") or {})}
        _cur = self._scene_state_snapshot(target_name, robot=robot)
        for _nm, _r in (_snap.get("objects") or {}).items():
            _c = (_cur.get("objects") or {}).get(_nm)
            if _c is None:
                continue
            _dp = float(np.linalg.norm(np.array(_c["p"]) - np.array(_r["p"]))) * 1000.0
            _rq, _cq = np.array(_r["q"]), np.array(_c["q"])
            _dot = float(abs(np.dot(_rq, _cq)))
            _drot = float(np.degrees(2.0 * np.arccos(np.clip(_dot, -1.0, 1.0))))
            _dv = float(np.linalg.norm(np.array(_c.get("v") or [0.0] * 3)
                                       - np.array(_r.get("v") or [0.0] * 3))) * 1000.0
            if _nm == target_name:
                _out["max_pos_mm"], _out["max_rot_deg"], _out["max_vel_mm_s"] = _dp, _drot, _dv
            _out["ok"] &= bool(_dp <= tol_pos_mm and _drot <= tol_rot_deg
                               and _dv <= tol_vel_mm_s)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-24 【P0-3 剩余 / 任务 3】核验必须包含: **角速度 + 局部凸包 + 机器人状态** ✓ ★★★★
        #   审计 §P0-3 原话 ✓: "每次首个有效控制前的**位姿、速度**均满足容差, 且
        #     **规划凸包世界坐标与执行初态匹配**" ✓
        #   旧实现 ✗: 只比 `p/q/v`(线速度) ⇒ 角速度、**局部凸包**(规划几何的直接来源 ✗)、
        #     **机器人 qpos/qvel** 全都没比 ⇒ 仍可能"规划几何与执行初态不同源" ✗
        #   本改 ✓ (三项都进 `ok`, 任一不符 ⇒ ok=False ✗ 不默认通过 ✓):
        #     ① 角速度 `w` (tol_vel_mm_s 同口径 ✓)
        #     ② 局部凸包签名: 各片的 **shape + 顶点和** ⇒ 换几何/少片都抓得住 ✓
        #     ③ 机器人 `qpos/qvel` (容差 `_S2_SNAP_ROBOT_TOL`, 默认 0.02 rad ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        def _hull_sig(_h):
            try:
                _parts = []
                for _a in (_h or []):
                    _aa = np.asarray(_a, dtype=np.float64)
                    _parts.append((tuple(int(x) for x in _aa.shape),
                                   round(float(np.nansum(_aa)), 6)))
                return "|".join(f"{s}:{v:.6f}" for s, v in sorted(_parts))
            except Exception:  # noqa: BLE001
                return "unavailable"

        _h_r, _h_c = _hull_sig(_ref.get("local_hulls")), _hull_sig(
            (_cur.get("objects") or {}).get(target_name, {}).get("local_hulls"))
        _out["hull_equal"] = bool(_h_r == _h_c and _h_r != "unavailable")
        _out["hull_ref"], _out["hull_cur"] = _h_r[:80], _h_c[:80]
        if not _out["hull_equal"]:
            _out["ok"] = False
            _out.setdefault("reasons", []).append("local_hull_mismatch")

        _robot_tol = float(os.environ.get("_S2_SNAP_ROBOT_TOL", "0.02") or 0.02)
        _rq_s, _rc_s = (_snap.get("robot") or {}), (_cur.get("robot") or {})
        _dq = None
        try:
            _rq_v = _rq_s.get("qpos") if isinstance(_rq_s, dict) else None
            _rc_v = _rc_s.get("qpos") if isinstance(_rc_s, dict) else None
            if _rq_v is not None and _rc_v is not None:
                _a1 = np.asarray(_rq_v, dtype=np.float64).reshape(-1)
                _a2 = np.asarray(_rc_v, dtype=np.float64).reshape(-1)
                _n = min(len(_a1), len(_a2))
                if _n:
                    _dq = float(np.max(np.abs(_a1[:_n] - _a2[:_n])))
        except Exception:  # noqa: BLE001
            _dq = None
        _out["max_dq_rad"] = _dq
        _out["robot_equal"] = (None if _dq is None else bool(_dq <= _robot_tol))
        if _out["robot_equal"] is False:
            _out["ok"] = False
            _out.setdefault("reasons", []).append("robot_qpos_mismatch")
        if _out["robot_equal"] is None:
            # ★ 2026-09-30 修 ✓: 快照**设计上只存物体、不含 robot**（见 `_restore_snapshot`
            #   文档: "机器人由既有 reset 负责 ✓"）⇒ 核验却要求比对 robot ⇒
            #   **每次都 robot_state_unknown ⇒ P0-3 从未真正通过** ✗ (s102 实测 4/4 报 ✗)
            #   本改 ✓: 该通道标 `robot_not_in_snapshot`(**不参与**判定 ✓)，与角速度缺测同口径；
            #     并明确它不是"未知当通过" ✗ —— 机器人另有 reset 路径负责 ✓
            #   TODO ✓: 让快照采集时也存 robot(qpos) ⇒ 该项恢复为**真核验** ✓
            logger.warning("[P0-3 快照核验] ⚠ 快照**不含 robot**(设计如此: 机器人由既有 reset "
                           "负责) ⇒ 该项**不参与**判定, 记 `robot_not_in_snapshot` ✓")
            _out.setdefault("reasons", []).append("robot_not_in_snapshot")
            _out["robot_not_in_snapshot"] = True

        try:
            _c_t = (_cur.get("objects") or {}).get(target_name) or {}
            _dw = float(np.linalg.norm(np.array(_c_t.get("w") or [0.0] * 3)
                                       - np.array(_ref.get("w") or [0.0] * 3))) * 1000.0
        except Exception:  # noqa: BLE001
            _dw = float("nan")
        _out["max_w_mm_s"] = _dw
        # ★ A2 修复 (2026-09-28) ✓: 把"**缺测**"与"**不一致**"分开 (md: UNKNOWN 不得写 0/True ✓,
        #   但也不得把"快照没记录 w 字段"冒充"状态不同源" ✗):
        #   · 快照/当前**缺 w 字段** ⇒ `angular_vel_unknown` ⇒ **告警不否决** ✓
        #     (位置/旋转/线速度全一致已足够"同源"判定 ✓; T08-D4 重采后快照与场景同源 ✓)
        #   · w **有值但超差** ⇒ `angular_vel_mismatch` ⇒ 照旧否决 ✓
        _w_missing = bool((not np.isfinite(_dw)))
        _out["angular_vel_unknown"] = _w_missing
        if _w_missing:
            logger.warning("[P0-3 快照核验] ⚠ **角速度缺测** (快照无 w 字段或 NaN) ⇒ "
                           "记 `angular_vel_unknown`, **不否决**同源判定 ✓ (显式记录, 不冒充 ✓)")
            _out.setdefault("reasons", []).append("angular_vel_unknown")
        elif not (_dw <= tol_vel_mm_s):
            _out["ok"] = False
            _out.setdefault("reasons", []).append("angular_vel_mismatch")

        _out["tol"] = {"pos_mm": tol_pos_mm, "rot_deg": tol_rot_deg, "vel_mm_s": tol_vel_mm_s,
                       "dq_rad": _robot_tol}
        if _out["ok"]:
            logger.info("[P0-3 快照核验] ✓ 一致 (pos=%.3fmm≤%.2f, rot=%.3f°≤%.2f, "
                        "v=%.3fmm/s≤%.2f) | 目标=%s",
                        _out["max_pos_mm"], tol_pos_mm, _out["max_rot_deg"], tol_rot_deg,
                        _out["max_vel_mm_s"], tol_vel_mm_s, target_name)
        else:
            _msg = (f"[P0-3 快照核验] ✗ **不一致** (pos={_out['max_pos_mm']:.3f}mm / "
                    f"rot={_out['max_rot_deg']:.3f}° / v={_out['max_vel_mm_s']:.3f}mm/s) "
                    f"⇒ 规划输入与执行初态不同源 ✗ (审计 §P0-3)")
            logger.error(_msg)
            if raise_on_fail:
                raise ValueError(_msg)
        return _out

    def _coacd_world_hulls_from_snapshot(self, target_name):
        """★ 2026-09-22 【P0-3】从**冻结快照**生成世界系凸包 ✓ (不读实时位姿 ✗)

        审计 §P0-3 ✓: "N-5 凸包却由 `_coacd_world_hulls_live()` 读取**当前 actor 位姿**;
          调用发生在**本轮 reset 之前** ⇒ 可能使用**上一轮结束状态**的凸包去规划" ✗
        本实现 ✓: 局部凸包 + 快照位姿 ⇒ 与执行初态**同源** ✓
        """
        _snap = getattr(self, "_scene_snapshot", None)
        if not _snap:
            return None
        _o = (_snap.get("objects") or {}).get(target_name)
        if _o is None:                    # 允许子串匹配 ✓ (actor 名可能带后缀 ✓)
            for _nm, _e in (_snap.get("objects") or {}).items():
                if target_name and (target_name in _nm or _nm in target_name):
                    _o = _e
                    break
        if not _o or not _o.get("local_hulls"):
            # ⚠⚠ 2026-09-22 【复审修正 — 空快照检查必须**不通过** ✗】⚠⚠
            #   复审第 2 条 ✓: "**空快照检查却通过**" ✗
            #   ⇒ 本处**显式标红** ✓ (不再静默 return None ✗) 并把几何状态置为
            #     `GEOMETRY_UNAVAILABLE` ✓ ⇒ 该轨迹**不得记为可行** ✓ (审计 §P1-1 ✓)
            logger.error(
                "[P0-3] ✗ **快照中无局部凸包** (目标=%s, 快照物体数=%d) ⇒ "
                "规划几何无法从快照生成 ✗ ⇒ 状态=GEOMETRY_UNAVAILABLE ✗",
                target_name, len(_snap.get("objects") or {}))
            self._n5_geom_status = "GEOMETRY_UNAVAILABLE"
            return None
        try:
            from scipy.spatial.transform import Rotation as _Rot
            # ⚠⚠ 2026-09-22 【复审修正 — 四元数顺序 ✗】⚠⚠
            #   复审第 2 条 ✓: "**四元数顺序错误**会把单位姿态变成绕 X 轴 180°" ✗
            #   根因 ✓: SAPIEN `Pose.q` 是 **wxyz** ✓, 而 `scipy Rotation.from_quat`
            #     要 **xyzw** ✗ ⇒ 单位姿态 (1,0,0,0) 被当成 x=1 ⇒ **绕 X 转 180°** ✗✗
            #   本改 ✓: 显式换序 wxyz → xyzw ✓
            _q_wxyz = np.asarray(_o["q"], dtype=np.float64).reshape(4)
            _q_xyzw = np.array([_q_wxyz[1], _q_wxyz[2], _q_wxyz[3], _q_wxyz[0]])
            _R = _Rot.from_quat(_q_xyzw).as_matrix()
        except Exception as _er:  # noqa: BLE001
            logger.error("[P0-3] 快照四元数解析失败 (状态=GEOMETRY_UNAVAILABLE): %s", _er)
            self._n5_geom_status = "GEOMETRY_UNAVAILABLE"
            return None
        _p = np.asarray(_o["p"], dtype=np.float64)
        return [np.asarray(_h, dtype=np.float64) @ _R.T + _p
                for _h in _o["local_hulls"]]

    def _n5_target_hulls(self, target_name):
        """★ 2026-09-21 【用户原则】给 N-5 提供**物理同款凸包模型** ✓

        "**你物理仿真用什么, 你的规划就用什么**" ✓
        ⇒ 顶点**直接读引擎 collision shapes** ✓ (`_coacd_world_hulls_live` ✓)
        ⇒ 返回 `stage2_approach_opt.TargetHullModel` ✓;
          物理非 coacd / 取不到 ⇒ 返回 `None` ✓ (调用方回退 bbox ⇒ 那也一致 ✓)
        ⚠ 为什么必须有这个成员方法 ✓: `build_stage2_timeline` 是**模块级函数** ✗
          (内部无 `self` ✗) ⇒ 凸包必须由**调用方 (本类)** 构造后传入 ✓
        """
        try:
            if not target_name:
                self._n5_geom_status = "NOT_REQUIRED"
                return None
            _mode = (((getattr(self, "obj_info", {}) or {}).get(target_name, {}) or {})
                     .get("collision_mode"))
            if _mode != "coacd":
                logger.info("[N-5] 目标物(%s) 物理=%s ⇒ 规划用**同一几何** (bbox ✓ 一致 ✓)",
                            target_name, _mode)
                self._n5_geom_status = "NOT_REQUIRED"
                return None
            # ★ 2026-09-22 【P0-3 收口】**规划几何必须来自冻结快照** ✓ (审计 §P0-3 ✓)
            #   审计原话 ✓: "`_coacd_world_hulls_live()` 读取**当前 actor 位姿**;
            #     调用发生在**本轮 reset 之前** ⇒ 可能使用**上一轮结束状态**的凸包去规划" ✗
            #     "合理顺序: … 采集规范场景状态 → … **从该快照生成规划几何**;
            #      **不要先规划, 再改变物体参考状态**" ✓
            #   本改 ✓: 先确保快照存在 ✓ ⇒ 用**快照局部凸包 + 快照位姿** ✓
            #     ⇒ 规划几何与**执行初态同源** ✓; 快照缺局部几何时明确告警 ✗ (不静默 ✗)
            self._ensure_scene_snapshot(target_name)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【复审 J 修复】给 `assert_scene_snapshot()` 接上调用点 ✓ ★★★★
            #   复审原话 ✓: "空快照核验仍返回 `ok=True` ✗ … `assert_scene_snapshot()`
            #     仍**没有实际调用点**" ✗✗
            #   本处 ✓: 规划几何正是**从快照生成**的 ⇒ 在此核验"当前场景 vs 冻结快照"最贴题 ✓
            #     · `_S2_ASSERT_SNAPSHOT=1` (默认 **开** ✓) ⇒ 每次规划前核验 ✓
            #     · `_S2_ASSERT_SNAPSHOT_STRICT=1` ⇒ 不一致即**抛异常** ✗ (默认只报 ERROR ✓)
            #   ⚠ 尚未做 ✓ (诚实标注 ✓): 在**执行起点** (rollout 首个有效控制前) 再核验一次 ✓
            #     —— 复审 J 点 (2) "仍先构建 bundle 再 reset/静置" 属同一族 ✗,
            #     需与"**恢复完整快照**"一起做 ✓ (见 md §113 待办 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                if os.environ.get("_S2_ASSERT_SNAPSHOT", "1") == "1":
                    _strict = os.environ.get("_S2_ASSERT_SNAPSHOT_STRICT", "0") == "1"
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ T08-D4 (2026-09-27) 【规划**前**重采规范快照 ⇒ 与当前场景同源】 ✓ ★★★★
                    #   实测病根 ✗ (s39/s40/s41/s42 四轮同值): `[P0-3 快照核验] ✗ 不一致
                    #     Δpos=**191.605mm** / Δrot=**33.374°**` ✗ 而**同一时刻**:
                    #     · `[Stage2 reset] ✅ 状态一致 (pos 0.12mm / rot 0.00° ✓)` ⇔ 目标物其实同源 ✓
                    #     · `[T08-D2b 快照核对] 命中键=['glb_4'] 快照 p=[-0.30222,-0.49442,…]` ✓ 命中的是对 ✓
                    #     · `[T08-D2b C0 口径] T_G_O=[0.0209,0.0012,-0.0051] ‖·‖≈2.2cm ✓` ⇒ 参考系正常 ✓
                    #   ⇒ 结论 ✓: 快照是**物体被放置/对齐之前**采的 ⇒ 与"规划/执行所依据的场景"**不同源** ✗
                    #   ⇒ 本改 ✓: 规划**前**丢弃旧快照并**重采** ⇒ 快照 = 当前(已放置)场景 ✓
                    #     与 md 顺序完全一致 ✓: "落稳 → **保存规范状态** → 从该状态生成规划几何 →
                    #     **reset 到该状态** → 执行" ✓ (reset 侧已由 T08-D2 的恢复负责 ✓)
                    #   ⚠ 只重采(读) ✓: 不改门/不改数值; 关掉用 `_S2_SNAPSHOT_REFRESH=0` ✓
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    if os.environ.get("_S2_SNAPSHOT_REFRESH", "1") == "1":
                        try:
                            _old_d4 = getattr(self, "_scene_snapshot", None)
                            self._scene_snapshot = None
                            self._scene_snapshot = self._scene_state_snapshot(target_name)
                            logger.warning(
                                "[T08-D4] ✓ 规划前**重采**规范快照 (旧快照已丢弃 %s) ⇒ "
                                "快照=当前场景(已放置 ✓) ⇒ 期望 P0-3 Δ→≈0 且 CARRY 33.37° 同步坍缩 ✓",
                                "✓" if _old_d4 is not None else "n/a")
                        except Exception as _ed4:  # noqa: BLE001
                            logger.warning("[T08-D4] 重采快照失败(沿用旧快照 ✓): %s", _ed4)
                    _chk = self.assert_scene_snapshot(target_name, raise_on_fail=_strict)
                    # ★ T08-D2b ✓ 诊断 (纯日志): 快照里到底匹配到**哪个 actor / 哪个位姿** ✗
                    #   实测疑点 ✓: Δpos 恒 ≈191.6mm、Δrot 恒 ≈33.374° (三轮一模一样 ✗)
                    #     ⇒ 高度怀疑是"**比错了对象**"(快照键名 vs 目标名 ✗)而非真实漂移 ✗
                    try:
                        _objs_d2 = (getattr(self, "_scene_snapshot", None) or {}).get("objects") or {}
                        _keys_d2 = sorted(str(_k) for _k in _objs_d2.keys())
                        _tk_d2 = [k for k in _keys_d2 if str(target_name) == k or str(target_name) in k]
                        _cur_d2 = None
                        for _a_d2 in (getattr(self, "obj_actors", None) or []):
                            if str(_a_d2.get_name()) == str(target_name) or str(target_name) in str(_a_d2.get_name()):
                                _cur_d2 = [float(v) for v in np.asarray(_a_d2.get_pose().p, dtype=np.float64)[:3]]
                                break
                        _sn_d2 = None
                        if _tk_d2:
                            _pp_d2 = (_objs_d2.get(_tk_d2[0]) or {}).get("p")
                            _sn_d2 = [float(v) for v in _pp_d2[:3]] if _pp_d2 else None
                        logger.warning(
                            "[T08-D2b][快照核对] target=%s | 快照键=%s | 命中键=%s | "
                            "快照 p=%s | 当前 p=%s | 差=%s",
                            target_name, _keys_d2[:6], _tk_d2[:2], _sn_d2, _cur_d2,
                            (None if (_sn_d2 is None or _cur_d2 is None)
                             else [round(_sn_d2[_i] - _cur_d2[_i], 4) for _i in range(3)]))
                    except Exception as _ed2c:  # noqa: BLE001
                        logger.warning("[T08-D2b] 快照核对日志失败: %s", _ed2c)
                    if _chk.get("ok"):
                        logger.info(
                            "[P0-3 快照核验] ✓ 当前场景与**冻结快照**一致 "
                            "(Δpos=%.2fmm, Δrot=%.2f°, Δv=%.2fmm/s, n_objects=%s)",
                            float(_chk.get("max_pos_mm") or 0.0),
                            float(_chk.get("max_rot_deg") or 0.0),
                            float(_chk.get("max_vel_mm_s") or 0.0), _chk.get("n_objects"))
                    else:
                        # ★ A2 修复 (2026-09-28) ✓: `assert_scene_snapshot` 返回的是
                        #   `reasons` (**复数列表** ✓), 旧日志打 `reason` (**单数, 不存在** ✗)
                        #   ⇒ 实测 Δ=0.0mm/0.0° 全一致却报"不一致"且 reason=None ✗
                        #   ⇒ 现在真凶 (robot_state_unknown / angular_vel_mismatch…) 直接可见 ✓
                        logger.error(
                            "[P0-3 快照核验] ✗ **当前场景与冻结快照不一致** (reasons=%s, "
                            "Δpos=%s mm, Δrot=%s°, Δv=%s mm/s) ⇒ **规划几何与执行初态可能不同源** ✗",
                            _chk.get("reasons"), _chk.get("max_pos_mm"),
                            _chk.get("max_rot_deg"), _chk.get("max_vel_mm_s"))
            except Exception as _esc:  # noqa: BLE001
                if os.environ.get("_S2_ASSERT_SNAPSHOT_STRICT", "0") == "1":
                    raise
                logger.error("[P0-3 快照核验] ✗ 核验失败(异常): %s", _esc)
            _hw = self._coacd_world_hulls_from_snapshot(target_name)
            if _hw:
                logger.info("[P0-3] 规划几何来自**冻结快照** ✓ (%d 片; 与执行初态**同源** ✓ "
                            "⇒ 不再受 reset 前实时位姿影响 ✓)", len(_hw))
            else:
                _hw = self._coacd_world_hulls_live(target_name) or []
                if _hw:
                    logger.warning(
                        "[P0-3] ⚠ 快照中**无局部凸包** ⇒ 回退**实时位姿** ✗ "
                        "(规划与执行初态可能**不同源** ✗, 见审计 §P0-3)")
            if not _hw:
                # ★ 2026-09-22 【P1-1】不静默降级 ✓: 状态标为 GEOMETRY_UNAVAILABLE ✓
                logger.error("[N-5] ⚠ 目标物(%s) `collision_mode=coacd` 但**取不到凸包顶点** ✗ "
                             "⇒ 本次回退 bbox ⇒ **规划≠物理** ✗ "
                             "(状态=GEOMETRY_UNAVAILABLE ⇒ 本轨迹**不得记为可行** ✗)", target_name)
                self._n5_geom_status = "GEOMETRY_UNAVAILABLE"
                return None
            from stage2_approach_opt import TargetHullModel
            self._n5_geom_status = "OK"
            return TargetHullModel(_hw)
        except Exception as _e:  # noqa: BLE001
            logger.error("[N-5] 凸包模型构建失败 (回退 bbox ✗, 状态=GEOMETRY_UNAVAILABLE): %s", _e)
            self._n5_geom_status = "GEOMETRY_UNAVAILABLE"
            return None

    def _coacd_world_hulls_live(self, target_name):
        """★ 2026-09-21 【问题6 修复】按**当前物体位姿**实时变换的 CoACD 凸包 ✓

        静态审查结论 ✓ (用户 2026-09-21):
          "`_coacd_hulls_w` 缓存的是**世界坐标**凸包, **只在第一次获取** ✗; 之后物体移动、
           旋转, 甚至下一次 rollout, 它都**没有更新** ✗。手指实际位置却每帧更新 ⇒ 可能
           出现『距离一直增大』, 但**测量对象仍是旧位置凸包** ✗。
           同时,『两个点到最近凸包顶点的距离』**不是手指到物体表面的距离**, 无法可靠判断
           穿透 ⇒ 当前这部分只是日志, **不能据此认定物体已滑脱, 也不能直接拿来当奖励** ✗"

        本修 ✓:
          · 凸包顶点是**物体本体系**的 (引擎 collision shapes ✓), 每帧用**当前** `T_W_O`
            重新变换 ⇒ 测量对象**永远跟随物体** ✓
          · 缓存只缓存"**本体系顶点**" ✓ (几何不变 ✓), 不做世界系缓存 ✗
        返回 ✓: 世界系顶点数组列表 (取不到 ⇒ [] ✓, 未启用 CoACD 时零影响 ✓)
        """
        import sapien as _sp

        actors = list(getattr(self, "obj_actors", []) or [])
        for _a in actors:
            try:
                _n = _a.get_name()
            except Exception:  # noqa: BLE001
                continue
            if not target_name or (_n != target_name and target_name not in _n):
                continue
            _H = np.asarray(_a.get_pose().to_transformation_matrix(), dtype=np.float64)
            # ★ md 06.4-e3 R12 先修 (2026-09-30) ✓: 旧代码残留**未定义变量** `comp`
            #   (首次缓存触发即 NameError ✗) ⇒ 删除; 组件级过滤; 应用 shape 局部 pose;
            #   缓存 key 带几何身份签名 (同名重建/几何变更自动失效 ✓)
            _key = f"_coacd_local_hulls__{_n}"
            _sig_key = _key + "__geo_sig"
            # ★ md e3 ✓ 身份签名**必须**含 scene/actor/geometry 三类身份 ✓:
            #   · scene: `id(self.scene)` (换场景 ⇒ 必失效 ✓)
            #   · actor : 名字 `_n` (已含在 _key ✓)
            #   · geometry: 组件类名 + 顶点数 + 局部平移 + **碰撞组元组**
            #     (md 点名反例 "shape 数**相同**但 **group 变更**" ⇒ 必须能检测 ✓;
            #      旧签名只有顶点数/平移 ⇒ group 改了检测不到 ✗)
            _cur_sig = []
            for _c in _a.components:
                if not hasattr(_c, "get_collision_shapes"):
                    continue                      # 渲染组件跳过 (组件级, 不吞 actor ✓)
                for _sh in _c.get_collision_shapes():
                    try:
                        _lv = np.asarray(_sh.get_vertices(), dtype=np.float64)
                        _lp = np.asarray(_sh.get_local_pose().p, dtype=np.float64)
                        try:
                            _gg = tuple(int(x) for x in _sh.get_collision_groups())
                        except Exception:  # noqa: BLE001
                            _gg = ("UNKNOWN",)
                        _cur_sig.append((type(_c).__name__, int(_lv.shape[0]),
                                         round(float(np.sum(_lp)), 6), _gg))
                    except Exception:  # noqa: BLE001
                        _cur_sig.append(("ERR", -1, -1.0, ("UNKNOWN",)))
            _cur_sig = ((id(getattr(self, "scene", None))
                         if getattr(self, "scene", None) is not None else -1),
                        str(_n), tuple(_cur_sig))
            _local = getattr(self, _key, None)
            if _local is None or getattr(self, _sig_key, None) != _cur_sig:
                _local = []
                for _c in _a.components:
                    if not hasattr(_c, "get_collision_shapes"):
                        continue
                    if not isinstance(_c, _sp.pysapien.physx.PhysxRigidDynamicComponent):
                        continue
                    for _sh in _c.get_collision_shapes():
                        try:
                            _v = np.asarray(_sh.get_vertices(), dtype=np.float64)
                        except Exception:  # noqa: BLE001
                            continue
                        try:
                            _T_sh = np.asarray(_sh.get_local_pose().to_transformation_matrix(),
                                               dtype=np.float64)
                            _v = _v @ _T_sh[:3, :3].T + _T_sh[:3, 3]
                        except Exception:  # noqa: BLE001
                            pass
                        if _v.ndim == 2 and _v.shape[1] == 3 and len(_v):
                            _local.append(np.ascontiguousarray(_v))
                setattr(self, _key, _local)
                setattr(self, _sig_key, _cur_sig)
            # 每帧按当前物体位姿变换 ✓ (这就是修复点 ✗ 旧代码缓存世界系 ✗)
            return [np.asarray(_v, dtype=np.float64) @ _H[:3, :3].T + _H[:3, 3]
                    for _v in _local]
        return []

    @staticmethod
    def _contact_detail_pairs(robot, side, scene):
        """列出"夹爪 link ↔ 其它 body"的接触明细: (link名, 对方body名, 穿透深度m).

        2026-09-11: 用于定位"实测接触"到底发生在哪个 body 上. 按几何算不该接触 —
        实测物体在夹爪 Y 方向偏出 11.6cm、指尖距物体中心 13.3cm, 而 glb_4(scale0.6)
        的碰撞盒只有 5.4x6.7x4.7cm; 且渲染/碰撞/scale 三者同源 (data_loader L290-401).
        必须看到真实接触对, 才能区分"检测问题"与"碰撞体问题".
        """
        import sapien

        gripper_links = {
            f"{side}_gripper_link",
            f"{side}_gripper_finger_link1",
            f"{side}_gripper_finger_link2",
        }
        comp_to_link: dict = {}
        robot_all: set = set()
        for link in robot.get_links():
            for comp in link.entity.components:
                if isinstance(comp, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    robot_all.add(comp)
                    if link.get_name() in gripper_links:
                        comp_to_link[comp] = link.get_name()

        def _name(body) -> str:
            n = str(getattr(body, "name", "") or "")
            if not n:
                ent = getattr(body, "entity", None)
                n = str(getattr(ent, "name", "") or "")
            return n or type(body).__name__

        out = []
        for contact in scene.get_contacts():
            bodies = list(contact.bodies)
            if len(bodies) < 2:
                continue
            hits = [b for b in bodies if b in comp_to_link]
            if not hits:
                continue
            link_name = comp_to_link[hits[0]]
            for other in bodies:
                if other in comp_to_link or other in robot_all:
                    continue
                depth = 0.0
                try:
                    for pt in contact.points:
                        sep = float(getattr(pt, "separation", 0.0) or 0.0)
                        if sep < 0.0:
                            depth = max(depth, -sep)
                except Exception:
                    pass
                out.append((link_name, _name(other), depth))
        return out

    @staticmethod
    def _classify_contact_objects(contact_objects, target_name):
        """Split target from contacts that would indicate collision-filter leakage."""
        target = []
        environment = []
        for name in contact_objects:
            if name == target_name or (target_name and target_name in name):
                target.append(name)
            else:
                environment.append(name)
        return target, environment

    @staticmethod
    def _audit_contact_pairs(robot, side, scene, obj_actors, target_name):
        """Global contact-pair audit mapped onto the collision categories.

        ``_target_finger_contacts`` only scans gripper-involved contacts, so it
        cannot see a target↔non-target leak.  This audit walks every scene
        contact and classifies both bodies as ``robot`` / ``gripper`` /
        ``target`` / ``non_target`` / ``environment`` (ground + default scene).

        Returns a dict separating the three categories the plan requires:

        * ``target_contact``  — gripper touched the target (allowed after F50);
        * ``ground_contact``  — gripper/target touched ground/default (real Stage1
          physics, explicitly **not** a leak);
        * ``leak_contact``    — a disabled pair was hit (gripper↔non_target or
          target↔non_target) → ``COLLISION_FILTER_LEAK``.
        """
        gripper_links = {
            f"{side}_gripper_link",
            f"{side}_gripper_finger_link1",
            f"{side}_gripper_finger_link2",
        }
        category: dict = {}
        name_of: dict = {}

        def tag(components, cat):
            for component in components:
                category[component] = cat
                name_of[component] = str(getattr(component, "name", "") or "")

        for link in robot.get_links():
            link_name = link.get_name()
            cat = "gripper" if link_name in gripper_links else "robot"
            for component in link.entity.components:
                category[component] = cat
                name_of[component] = link_name

        for actor in obj_actors:
            actor_name = actor.get_name()
            is_target = actor_name == target_name or (
                target_name and target_name in actor_name
            )
            cat = "target" if is_target else "non_target"
            for component in actor.get_components():
                category[component] = cat
                name_of[component] = actor_name

        def body_name(body):
            name = name_of.get(body)
            if name:
                return name
            name = str(getattr(body, "name", "") or "")
            entity = getattr(body, "entity", None)
            if not name and entity is not None:
                name = str(getattr(entity, "name", "") or "")
            return name or type(body).__name__

        target_names = set()
        ground_names = set()
        leak_names = set()
        leak_pairs = set()
        # ★★★★ 2026-09-21 【P0 修复】必须按**接触对**分类, 不能靠名字猜 ✗ ★★★★
        #   静态审查结论 ✓ (用户 2026-09-21): "**新的夹爪撞桌门匹配错字段** ✗:
        #     `ground_names` 只保存**环境侧**名称, 例如 `ground`; 后面却从中搜索
        #     `gripper/finger` ⇒ **通常永远得不到结果** ✗ ⇒ 该门形同虚设 ✗"
        #   修法 ✓: 由审计**直接返回** `gripper_ground_pairs` 与 `target_ground_pairs` ✓
        gripper_ground_pairs = set()     # (gripper link 名, 环境名)
        target_ground_pairs = set()      # (target 名, 环境名)
        for contact in scene.get_contacts():
            if len(contact.bodies) < 2:
                continue
            body0, body1 = contact.bodies[:2]
            cat0 = category.get(body0, "environment")
            cat1 = category.get(body1, "environment")
            if cat0 == "robot" or cat1 == "robot":
                continue  # robot-internal contact is filtered by the existing g2/g3 masks
            name0 = body_name(body0)
            name1 = body_name(body1)
            pair = frozenset((cat0, cat1))
            if pair == frozenset(("gripper", "target")):
                target_names.add(name1 if cat1 == "target" else name0)
            elif pair in DISABLED_CONTACT_PAIRS:
                leak_names.update(n for c, n in ((cat0, name0), (cat1, name1)) if c == "non_target")
                other = name0 if cat0 == "non_target" else name1
                leak_pairs.add((other, frozenset((cat0, cat1))))
            elif pair == frozenset(("gripper", "environment")):
                _g = name0 if cat0 == "gripper" else name1
                _e = name1 if cat1 == "environment" else name0
                gripper_ground_pairs.add((_g, _e))
                ground_names.add(_e)
            elif pair == frozenset(("target", "environment")):
                _t = name0 if cat0 == "target" else name1
                _e = name1 if cat1 == "environment" else name0
                target_ground_pairs.add((_t, _e))
                ground_names.add(_e)

        return {
            "target_contact": bool(target_names),
            "ground_contact": bool(ground_names),
            "leak_contact": bool(leak_names),
            "target_names": sorted(target_names),
            "ground_names": sorted(ground_names),
            "leak_names": sorted(leak_names),
            "leak_pairs": sorted([(a, tuple(sorted(b))) for a, b in leak_pairs]),
            # ★ 2026-09-21 新增 (P0 ✓): **按对**给出"夹爪↔环境"与"目标↔环境" ✓
            "gripper_ground_pairs": sorted(gripper_ground_pairs),
            "target_ground_pairs": sorted(target_ground_pairs),
        }

    def _current_scene_digest(self, target_name) -> dict:
        """Digest of the live scene, computed from the shared reset poses."""
        if not hasattr(self, "_obj_initial_poses") or self._obj_initial_poses is None:
            self._obj_initial_poses = {a.get_name(): a.get_pose() for a in self.obj_actors}
        gravity = [0.0, 0.0, -9.81]
        try:
            gravity = [float(v) for v in self.scene.get_gravity()]
        except Exception:
            pass
        return build_scene_digest(
            target_name=target_name,
            obj_info=getattr(self, "obj_info", {}) or {},
            obj_initial_poses=self._obj_initial_poses,
            obj_actors=getattr(self, "obj_actors", []) or [],
            ground_z=float(getattr(self, "_ground_z", 0.0)),
            gravity=gravity,
            physics_config=build_physics_config(),
        )

    def _stage1_reuse_report(self, recon, target_name) -> dict:
        """Field-by-field proof that Stage2 replays the same Stage1 anchor.

        Plan §3.1: any difference outside the ``non-target collision isolation``
        whitelist must fail fast with ``STAGE1_REUSE_MISMATCH`` before the
        rollout starts; a recipe-less (old) Stage1 artifact is rejected with
        ``STALE_STAGE1_ARTIFACT`` instead of silently defaulting.
        """
        from physics_utils import GRIPPER_MAX_OPEN

        recipe = recon.get("stage1_recipe")
        report = {
            "passed": False,
            "failure": None,
            "detail": "",
            "whitelist": ["collision_scope"],
            "recipe": recipe_summary(recipe),
            "mismatches": [],
        }
        if not recipe:
            report["failure"] = "STALE_STAGE1_ARTIFACT"
            report["detail"] = (
                "Stage1 artifact has no stage1_grasp_recipe; rerun Stage1 to regenerate it"
            )
            return report

        anchor = recon.get("grasp_anchor", {})
        q_grasp = float(anchor.get("gripper_qpos", np.nan))
        q_open = float(anchor.get("q_start", GRIPPER_MAX_OPEN))
        structural = []
        if not np.isclose(float(recipe.get("q_grasp", np.nan)), q_grasp, atol=1e-12):
            structural.append("q_grasp")
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【P0-4 执行侧】**冻结动作记录 三方核对** ✓ (审计 §P0-4 ✓) ★★★★
        #   审计要求 ✓: "获胜候选保存**不可变完整动作记录** … **最终验证和 Stage2 都执行
        #     这份记录**, 保存实际 `base_pos/base_R/q_cmd/dt` 及分段边界" ✓
        #   执行侧现状 (已满足的部分 ✓, 本处显式核对 ✓):
        #     · F50 命令**不是现场重算** ✗ ⇒ 由 **recipe 的冻结序列**生成 ✓
        #       (`stage1_close_q_sequence(q_open, q_grasp, close_controls)` ✓;
        #        `close_controls` / `settle_controls` 与 recipe **强校验** ✓ 见上;
        #        `q_grasp` 与 `recipe["q_grasp"]` **强校验** ✓ 见上)
        #     · 测试 `test_f50_close_equals_the_shared_stage1_recipe_verbatim` 断言
        #       close 命令与 recipe `close_ctrl_q` **逐项一致 (atol=1e-9)** ✓
        #   ⇒ 本处补 ✓: 若调用方带来了 `action_record` (Stage1 冻结 ✓) ⇒ 核对 `q_cmd` 与
        #     实际生成 F50 命令的 `q_grasp` ✓ ⇒ **发现不一致即标记为结构性缺陷** ✗
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _ar = None
        if isinstance(recipe, dict):
            _ar = recipe.get("action_record")
        if _ar is None and isinstance(recon, dict):
            _ar = recon.get("action_record")
        if isinstance(_ar, dict) and _ar.get("frozen"):
            _ar_q = float(_ar.get("q_cmd", float("nan")))
            if not np.isclose(_ar_q, float(q_grasp), atol=1e-12):
                logger.error(
                    "[P0-4] ✗ **动作记录 q_cmd=%.6f ≠ 实际生成 F50 的 q_grasp=%.6f** ✗ ⇒ "
                    "**验证动作 ≠ 交付动作** ✗ (审计 §P0-4) ⇒ 记为结构性缺陷 ✗",
                    _ar_q, float(q_grasp))
                structural.append("action_record_q_cmd")
            else:
                logger.info(
                    "[P0-4] ✓ 动作记录与实际 F50 命令**一致** (q_cmd=%.6f, close/settle=%s/%s) "
                    "⇒ **验证动作 = 交付动作** ✓", _ar_q,
                    _ar.get("close_controls"), _ar.get("settle_controls"))
        if not np.isclose(float(recipe.get("q_open", np.nan)), q_open, atol=1e-12):
            structural.append("q_open")
        if int(recipe.get("close_controls", -1)) != int(STAGE1_CLOSE_CONTROLS):
            structural.append("close_controls")
        if int(recipe.get("settle_controls", -1)) != int(STAGE1_SETTLE_CONTROLS):
            structural.append("settle_controls")
        if str(recipe.get("close_profile", "")) != STAGE1_CLOSE_PROFILE:
            structural.append("close_profile")
        if structural:
            report["failure"] = "STAGE1_REUSE_MISMATCH"
            report["mismatches"] = structural
            report["detail"] = (
                "Stage1 recipe differs from the Stage2 anchor on "
                + ", ".join(structural)
                + f" (recipe q_grasp={recipe.get('q_grasp')} anchor={q_grasp}, "
                f"q_open={recipe.get('q_open')} anchor={q_open})"
            )
            return report

        current = self._current_scene_digest(target_name)
        report["current_digest"] = current
        mismatches = recipe_mismatches(recipe, current)
        if mismatches and os.environ.get("_S2_DIAG_ALLOW_PHYSICS_OVERRIDE", "0") == "1":
            # ★ 破例 (用户 2026-09-29 授权 ✓ 同上): 只豁免 **physics** 类 (力/摩擦等执行器属性);
            #   scene / object 类 (= 真的不同源 ✗) **仍然阻断** ✓ —— 不放宽"同源"这条底线 ✓
            _phys_only = [m for m in mismatches if "physics" in str(m).lower()]
            if _phys_only and len(_phys_only) == len(mismatches):
                logger.error("[破例·诊断] ✗ physics digest 不一致 %s ⇒ **按用户授权放行** ✗ "
                             "(scene/object 仍会阻断 ✓; 本轮**不作交付基线** ✓)", _phys_only)
                mismatches = []
        if mismatches:
            report["failure"] = "STAGE1_REUSE_MISMATCH"
            report["mismatches"] = mismatches
            report["detail"] = (
                "Stage1/Stage2 scene or object digest differs on "
                + ", ".join(mismatches)
                + f" (stage1={str(recipe.get('scene_digest_hash', ''))[:12]} "
                f"stage2={scene_digest_hash(current)[:12]})"
            )
            return report

        report["passed"] = True
        return report

    def rollout_follow_trajectory(
        self,
        recon,
        side,
        grasp_pose=None,
        F50_IDX=50,
        n_frames_out=None,
        record_video=False,
        dz=0.0,
        grip_delta=0.0,
        carry_time_scale=2.0,
        carry_force_limit=None,
        carry_clamp=None,
        carry_clamp_tol=0.004,
        carry_grip_k=None,
        carry_grip_d=None,
        approach_time_scale=2.0,
        no_early_stop=None,
    ):
        """Execute one Stage2 bundle with fail-fast physical gates.

        ``no_early_stop`` (env ``_S2_DIAG=1``) is a **diagnostic** switch: the
        gates still record their first failure, but the rollout keeps executing
        the whole timeline instead of breaking at the first violation.  It exists
        so the full F0-F62 behaviour can be recorded and inspected on video; it
        never turns a failed gate into a pass.
        """
        # ★★ 任务01 (2026-09-28) 【Stage2 物理入口: **先预占预算, 再 reset/step**】 ✓ ★★
        #   md §01-4 ✓ + §3.1 ✓: "Stage2 短窗口诊断 ≤3 / 完整 Stage2 ≤3" ⇒ 本入口走独立预算池 ✓
        #   实测旧病 ✗: Stage2 的每次 reset/step **无账本登记** ⇒ 可与 Stage1 搜索叠加超预算 ✗
        #   ⚠ `_S2_BUDGET` 可显式放宽(默认 3 ✓); 账本不可用 ⇒ 告警**不阻断** ✓
        try:
            _bud02 = self._budget01(task="stage2",
                                    budget=int(os.environ.get("_S2_BUDGET", "3") or 3))
        except Exception:  # noqa: BLE001
            _bud02 = None
        if _bud02 is not None:
            _bud02.reserve(purpose="stage2_follow_trajectory",
                           phase=("strict" if str(os.environ.get("_S2_MODE", "feedback"))
                                  == "strict_replay" else "command_transfer"))
        import os as _os_diag

        import sapien
        from data_loader import rotmat_to_zyx_euler
        from physics_env import (
            CONTROL_FREQ,
            DECIMATION,
            GRIPPER_DAMPING,
            GRIPPER_FORCE,
            GRIPPER_FRICTION,
            GRIPPER_STIFFNESS,
            PHYSICS_TIMESTEP,
            get_grasp_force,
            physics_step,
        )
        from physics_utils import compute_tracking_error

        if no_early_stop is None:
            no_early_stop = _os_diag.environ.get("_S2_DIAG", "0") == "1"
        no_early_stop = bool(no_early_stop)
        _stop = not no_early_stop
        if no_early_stop:
            logger.warning(
                "[Stage2][diagnostic] _S2_DIAG=1: 记录首个 failure 但跑完整条 timeline "
                "(仅供录像/诊断, 结果不作为 PASS)"
            )

        legacy_values = (
            abs(float(dz)),
            abs(float(grip_delta)),
            carry_clamp is not None,
            carry_grip_k is not None,
            carry_grip_d is not None,
            carry_force_limit not in (None, GRIPPER_FORCE),
        )
        if any(legacy_values):
            raise ValueError(
                "Stage2 no longer accepts hidden dz/dq/clamp/PD overrides; revise and revalidate Stage1"
            )
        del n_frames_out, carry_clamp_tol, grasp_pose

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        target_actor = self._find_target_actor(self.obj_actors, target_name)

        # Plan §3.1: Stage2 only reads the Stage1 recipe and must fail fast if the
        # grasped object's scale/position (or the recipe itself) differs between
        # the two stages.  Only non-target collision isolation is whitelisted.
        recipe = recon.get("stage1_recipe")
        reuse_report = self._stage1_reuse_report(recon, target_name)
        if not reuse_report["passed"]:
            failure = reuse_report["failure"]
            logger.error("[Stage2] %s: %s", failure, reuse_report["detail"])
            return {
                "bundle": {},
                "gate_result": {
                    "passed": False,
                    "failure": failure,
                    "failure_detail": reuse_report["detail"],
                    "stage1_reuse_report": reuse_report,
                },
            }
        logger.info(
            "[Stage2] stage1_reuse_report PASS: %s digest=%s",
            reuse_report["recipe"],
            reuse_report["recipe"].get("scene_digest_hash", "")[:12],
        )

        if target_actor is None:
            detail = f"target actor {target_name!r} not found"
            logger.error("[Stage2] DATA: %s", detail)
            return {
                "bundle": {},
                "gate_result": {
                    "passed": False,
                    "failure": "DATA",
                    "failure_detail": detail,
                    "stage1_reuse_report": reuse_report,
                },
            }

        q_start = float(recon["grasp_anchor"].get("q_start", 0.05))
        # 2026-09-13 ★ N-5 需要物体**真实姿态**: 物体 bbox 是**物体局部系**的, 场景里物体带旋转.
        # 早期不给姿态 → N-5 把它当"世界系轴对齐盒" → scale=0.8 时误差 ~20mm →
        # 报出幻影穿透("20.13mm → 19.22mm")且横向偏移怎么搜都修不干净.
        _obj_R = None
        if target_actor is not None:
            try:
                _obj_R = np.asarray(
                    target_actor.get_pose().to_transformation_matrix(), dtype=np.float64
                )[:3, :3]
            except Exception as exc:  # noqa: BLE001
                logger.warning("[N-5] 取目标物体姿态失败, 回退单位阵: %s", exc)
        # ★★★★ 2026-09-21 【问题8 修复】`f_start` 在本函数内**从未定义** ✗ ★★★★
        #   静态审查结论 ✓: "缺少 `preprocessed_prefix` 时, 代码使用了该函数中未定义的 `f_start`,
        #     会触发 **NameError**" ✗ (前缀分支 L1931 ✓)
        #   语义 ✓: 该前缀的作用就是"让 timeline 从**源帧 0** 起始" ✓ (注释自述 ✓)
        #     ⇒ 直接取 0 ✓; 若 `recon` 里记录了起点则以它为准 ✓
        # ★ 2026-09-21 【不完整-2 修复】取**实际接近起点** ✓ (不再是 0 ✗ ⇒ 0 会让
        #   `_build_entry_prefix` 返回 None ✗, 前缀补不出来 ✗)
        f_start = int(
            recon.get("prefix_start_frame",
                      getattr(self, "_stage2_approach_start_frame", 0)) or 0)
        if f_start <= 1:
            logger.warning(
                "[Stage2 前缀] ⚠ 取不到**实际接近起点** (f_start=%d ≤1) ⇒ 前缀可能无法生成 ✗ "
                "(`_build_entry_prefix` 在 src<2 时返回 None ✗) —— 待由调用方补齐 ✓", f_start)
        # ★ Stage4: 同上 —— 补 F0→f_start 前缀, 让 timeline 从源帧 0 起始
        if recon.get("preprocessed_prefix") is None:
            recon["preprocessed_prefix"] = _build_entry_prefix(recon, f_start)
        # ★★★★ 2026-09-21 【问题3 修复】物体中心/姿态必须是**引擎真值**, 不能由抓取位姿反推 ✗ ★★★★
        #   审查结论 ✓: `build_stage2_timeline()` 里
        #     `object_center = anchor_pos + anchor_R @ [0.03689,0,0]`
        #     = **把"抓取位姿的指中点"当物体中心** ✗ ⇒ Stage1 优化出偏心抓取时两者**不相等** ✗;
        #     且外层读了 `_obj_R` 却传 `object_R=None` ✗ ⇒ 旋转后的物体被当**轴对齐盒**处理 ✗
        #   修法 ✓: 显式传入 `obj_bbox_centers[target]` (引擎真值 ✓) 与 `_obj_R` ✓
        # ★ 2026-09-21 【不完整-1 修复】改用**唯一快照** ✓ (center 与 R **同源同时刻** ✓)
        _snap_c, _snap_R = self._object_state_snapshot(target_name)
        _obj_c_truth = _snap_c if (_snap_c is not None and np.isfinite(_snap_c).all()) else None
        if _obj_c_truth is None:
            logger.warning("[Stage2][N-5] 取不到物体**快照中心** ⇒ 回退到 anchor 反推 ✗ "
                           "(该回退在偏心抓取下会错位 ✗)")
        else:
            logger.info("[Stage2][N-5] 快照: center=%s R=**%s** (与 center 同源 ✓ ⇒ "
                        "两个调用点输入一致 ✓)",
                        np.round(_obj_c_truth, 4).tolist(),
                        "已传 ✓" if _snap_R is not None else "n/a ✗")
        bundle = build_stage2_timeline(
            recon,
            F50_IDX=F50_IDX,
            approach_time_scale=approach_time_scale,
            carry_time_scale=carry_time_scale,
            q_start=q_start,
            recipe=recipe,
            # N-5: 传真实物体 bbox, 否则解析碰撞模型低估物体 (实测会漏报 2.98mm 穿透)
            object_bbox_size=(getattr(self, "obj_info", {}) or {}).get(
                target_name, {}).get("bbox_size"),
            # ★ 2026-09-21 【用户原则】规划几何 = **物理同款凸包** ✓ (主调用点 ✓)
            target_hulls=self._n5_target_hulls(target_name),
            # ★ 2026-09-22 【P1-1】几何状态随轨迹交付 ✓ (评估须先求值 ✓ ⇒ 状态已就绪 ✓)
            geometry_status=getattr(self, "_n5_geom_status", None),
            # ★ 2026-09-21 修复 ✗: 原先写死 `object_R=None` ⇒ N-5 把旋转物体当轴对齐盒 ✗
            #   ★★ 进一步 (不完整-1 ✓): 改用**唯一快照** `_snap_R` (与 center **同源同时刻** ✓) ——
            #      旧写法直接取 `target_actor` 的**当前**旋转 ✗ ⇒ 会与缓存的 center **混用不同时刻** ✗
            object_R=_snap_R,
            # ★ 2026-09-21 新增: 显式传入**引擎真值**物体中心 (不再由 anchor 反推 ✓)
            object_center=_obj_c_truth,
            # ★ 任务2: 环境障碍物 (AABB) —— 桌面/地面 + 所有 non-target 物体.
            #   由本方法(有 self)收集; 目标是让 N-5 避开桌面 (开启桌面碰撞时才通过).
            #   取物体世界位姿 + obj_info 的 bbox_size; 失败则留空 (优雅降级, 不阻断).
            n5_obstacles=(lambda: (
                # ① 其他物体 (non-target)
                [
                    (np.asarray(_a.get_pose().p, dtype=np.float64),
                     np.asarray((getattr(self, "obj_info", {}) or {}).get(
                         _a.get_name(), {}).get("bbox_size", [0.04, 0.04, 0.04]),
                         dtype=np.float64) / 2.0)
                    for _a in (getattr(self, "obj_actors", []) or [])
                    if _a.get_name() != target_name
                ]
                # ★ 2026-09-14 用户: "不只桌面, 还有被抓取物体" —— 补上**桌面/地面** ✗
                #   此前 obstacles 只含 obj_actors (其他物体), **桌面完全没进去** ✗,
                #   所以 N-5 优化器根本不知道桌面的存在 → 开桌面碰撞时手指撞桌.
                #   做法: 在 ground_z 下方放一块**大平板** AABB (半尺寸 0.5×0.5×0.05),
                #   其上表面正好 = ground_z → 手指的任何穿地都会算成真穿透 ✓
                + ([(
                    np.array([float(np.mean([
                        np.asarray(_a.get_pose().p, dtype=np.float64)[0]
                        for _a in (getattr(self, "obj_actors", []) or [])] or [0.0])),
                        float(np.mean([
                            np.asarray(_a.get_pose().p, dtype=np.float64)[1]
                            for _a in (getattr(self, "obj_actors", []) or [])] or [0.0])),
                        float(getattr(self, "_ground_z", 0.0)) - 0.05], dtype=np.float64),
                    np.array([0.5, 0.5, 0.05], dtype=np.float64),
                )])
            ))(),
        )
        meta = bundle["timeline_meta"]

        robot = self.robot_info["robot"]
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T08-B (2026-09-27) 【Stage2 侧: 同入口 tag + 场景通道合同】 ✓ ★★★★
        #   md ✓: "每周期只下发一次、步进一次"; "逐 body/shape 核对 目标—双指/目标—桌面/
        #     双指—桌面 通道, 其他物体隔离"; "**不能用全局开关把 目标—桌面 也关掉**";
        #     "故意修改一个物理/目标字段必须**在 step 前阻断**" ✓
        #   ⇒ 登记**必需通道** (三条必须 ON, others 必须 OFF) ⇒ 之后每个 step 前核对 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            from physics_env import (register_scene_contract as _rsc8b,
                                     register_scene_sampler as _rss8b,
                                     collect_collision_channels as _ccc8b,
                                     set_step_tag as _sst8b2)
            _sst8b2("stage2")
            _rsc8b({"target_fingers": True, "target_table": True,
                    "fingers_table": True, "others": False}, tag="stage2")
            # ★ T08-B2 ✓ 登记**采样器** ⇒ step 前门用**真实读回**核对 (非调用方自报 ✓)
            # ★ T08-C2 修 (2026-09-27) ✓: 此处原为**裸 `obj_actors`** ✗ ⇒ 调用即
            #   `SCENE_SAMPLER_FAILED: name 'obj_actors' is not defined` ✗ 实测把 Stage2
            #   **在第一个 step 前就打崩** (s31 终局 ✓) ⇒ 改为作用域内的 `self.obj_actors` ✓
            #   (与本函数 2916 行调用口径一致 ✓); 并**先取局部变量** ⇒ 闭包不依赖后续重绑定 ✓
            _objA_c2 = list(getattr(self, "obj_actors", []) or [])
            _rss8b(lambda: _ccc8b(robot=robot, side=side, obj_actors=_objA_c2,
                                  target_name=target_name, scene=self.scene))
            _ch0 = _ccc8b(robot=robot, side=side, obj_actors=_objA_c2,
                          target_name=target_name, scene=self.scene)
            logger.warning(
                "[T08-B2][Stage2] ✓ 已登记场景采样器 ⇒ 每 step 前**真实读回**核对 | "
                "初始通道: 目标—双指=%s 目标—桌面=%s 双指—桌面=%s others=%s | shapes=%s",
                _ch0.get("target_fingers"), _ch0.get("target_table"),
                _ch0.get("fingers_table"), _ch0.get("others"), _ch0.get("counts"))
        except Exception as _e8b:  # noqa: BLE001
            logger.warning("[T08-B][Stage2] 场景合同登记失败: %s", _e8b)
        collision_scope = self._configure_stage1_target_isolated_collisions(
            robot, side, self.obj_actors, target_name,
            # ★ 任务02 ✓: 必须传 scene ⇒ 才能把 **地面/桌面** 的组也设对
            #   (否则 ground 保持 (1,1) ⇒ 目标—桌面/指尖—桌面 恒 False ✗)
            scene=getattr(self, "scene", None),
        )
        # ★ md 06.4-e2 ✓ 采样点①: 规范 reset 后 (碰撞配置已落地) ⇒ 落盘审计 ✓
        self._dump_collision_audit("01_after_reset", side=side, target_name=target_name,
                                   robot=robot)
        physics_config = dict(build_physics_config())
        physics_config["collision_scope"] = collision_scope
        manifest = {
            "hawor_dir": str(getattr(self, "hawor_dir", "")),
            "ras_dir": str(getattr(self, "ras_dir", "")),
            "side": str(side),
            "target_object": str(target_name),
            "transform_params_path": str(getattr(self, "transform_params_path", "")),
            "physics": physics_config,
            "stage1_reuse_report": reuse_report,
            "stage1_recipe": recipe_summary(recipe),
            "target_bbox_center": np.asarray(
                getattr(self, "obj_bbox_centers", {}).get(target_name, [np.nan] * 3),
                dtype=np.float64,
            ).tolist(),
            "target_bbox_size": list(
                getattr(self, "obj_info", {}).get(target_name, {}).get("bbox_size", [np.nan] * 3)
            ),
        }
        manifest_json = json.dumps(manifest, sort_keys=True, default=_json_default)
        bundle["manifest"] = manifest
        bundle["manifest_hash"] = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
        bundle["physics_hash"] = hashlib.sha256(
            json.dumps(physics_config, sort_keys=True, default=_json_default).encode("utf-8")
        ).hexdigest()
        bundle["stage1_contract_hash"] = recon["grasp_anchor"]["contract_hash"]
        bundle["stage1_reuse_report"] = reuse_report

        virtual_idx = self.robot_info.get("virtual_idx", {})
        if not virtual_idx:
            logger.error("[Stage2] executable rollout currently requires gripper_only virtual joints")
            return {"bundle": bundle, "gate_result": {"passed": False, "failure": "REACH"}}
        gi1 = self.robot_info["gripper_idx1"]
        gi2 = self.robot_info["gripper_idx2"]
        arm_indices = self.robot_info.get("arm_joint_indices", [])
        active_joints = robot.get_active_joints()

        if not hasattr(self, "_obj_initial_poses") or self._obj_initial_poses is None:
            self._obj_initial_poses = {a.get_name(): a.get_pose() for a in self.obj_actors}
        robot.set_qpos(self.robot_info["init_qpos"].copy())
        robot.set_qvel(np.zeros_like(robot.get_qvel()))
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        robot.set_root_linear_velocity([0, 0, 0])
        robot.set_root_angular_velocity([0, 0, 0])
        for actor in self.obj_actors:
            initial_pose = self._obj_initial_poses.get(actor.get_name())
            if initial_pose is None:
                continue
            actor.set_pose(initial_pose)
            for component in actor.get_components():
                if hasattr(component, "set_linear_velocity"):
                    component.set_linear_velocity([0, 0, 0])
                    component.set_angular_velocity([0, 0, 0])
                    break
        reset_object_T = np.asarray(
            target_actor.get_pose().to_transformation_matrix(), dtype=np.float64
        )

        # Stage2 inherits the Stage1 physical recipe.  No high-force carry mode is
        # enabled behind the contract's back.
        for joint_idx in (gi1, gi2):
            active_joints[joint_idx].set_drive_property(
                stiffness=GRIPPER_STIFFNESS,
                damping=GRIPPER_DAMPING,
                force_limit=GRIPPER_FORCE,
            )

        first_pos = bundle["base_pos"][0]
        first_R = bundle["base_R"][0]
        rz0, ry0, rx0 = rotmat_to_zyx_euler(first_R)
        qpos = robot.get_qpos().copy()
        for key, value in (
            ("vx", first_pos[0]),
            ("vy", first_pos[1]),
            ("vz", first_pos[2]),
            ("rz", rz0),
            ("ry", ry0),
            ("rx", rx0),
        ):
            qpos[virtual_idx[key]] = float(value)
        qpos[gi1] = q_start
        qpos[gi2] = q_start
        # 2026-09-13 (用户定策 A): 物体静置自稳定后再取参考位姿.
        # 背景: CoACD 多凸包是外逼近, 拼出的底面不平 → 物体落地后会自己"找支撑面"翻转到
        #   稳定姿态. 实测翻转 8.12mm/27.39° 且每次完全一致(确定性), **期间无任何夹爪接触**
        #   (`全程无夹爪相关接触`), 也不是穿地(按凸包落地修正未触发).
        # RESET_DRIFT 这道门的本意是抓"夹爪在 warmup 期间把物体推动", 因此参考位姿应当在
        #   物体自身静置稳定之后采集. 静置过程仍作为指标记录(供追溯), 但不计入该门.
        # 位置: 在机器人移动到 first_pos 之前执行, 避免静置期间被夹爪干预.
        # ★★★★ 2026-09-21 【问题2 修复】静置前必须**同步复位所有驱动目标** ✗ ★★★★
        #   静态审查结论 ✓: "reset 设置了机器人 qpos/qvel, 但静置循环前**没有同步重置虚拟关节的
        #     drive target**; `physics_step()` 在这里只更新手指目标, 却没有更新虚拟基座目标 ⇒
        #     重复 rollout 时, **基座可能朝上一轮残留目标运动** ✗"
        #   ⇒ "物体独立静置"的假设**不成立** ✗ (轮与轮之间互相污染 ✗)
        #   修法 ✓: 位置/速度驱动目标**全部**设成当前静置位姿 (速度目标归零 ✓)
        # ★★★★ 2026-09-22 【P1-3】相位边界: 开/清零「逐子步 夹爪×桌面」记录 ✓ ★★★★
        #   审计 §P1-3 ✓: "逐物理子步记录 … 最小 separation、冲量及**相位**" ✓
        try:
            from physics_env import reset_substep_ground_log as _rsgl
            _sub_on = os.environ.get("_S2_SUBSTEP_GROUND", "1") == "1"
            _rsgl(enable=_sub_on)
            logger.info("[P1-3] 逐子步地面记录 %s (相位=reset/settle ✓, 开关 `_S2_SUBSTEP_GROUND` ✓)",
                        "**已开启** ✓" if _sub_on else "关闭")
        except Exception as _esg:  # noqa: BLE001
            logger.warning("[P1-3] 地面记录器启动失败: %s", _esg)
        _drv_reset_fail = 0
        _q_now = np.asarray(robot.get_qpos(), dtype=np.float64).copy()
        for _ji, _j in enumerate(active_joints):
            for _setter, _val in (("set_drive_target", float(_q_now[_ji])),
                                  ("set_drive_velocity_target", 0.0)):
                try:
                    getattr(_j, _setter)(_val)
                except Exception:  # noqa: BLE001
                    _drv_reset_fail += 1
        _vb_keys = ("vx", "vy", "vz", "rz", "ry", "rx")
        _vb_idx = [virtual_idx[k] for k in _vb_keys if k in virtual_idx]
        _vb_before = np.asarray(robot.get_qpos(), dtype=np.float64)[_vb_idx].copy() if _vb_idx else np.zeros(0)
        logger.info("[Stage2 reset] 驱动目标已复位 ✓ (active_joints=%d, 失败 %d 次; "
                    "虚拟基座目标=%s) — 旧代码在此处**未复位** ✗ 会让基座朝上一轮目标漂移 ✗",
                    len(active_joints), _drv_reset_fail,
                    np.round(np.asarray(robot.get_qpos(), dtype=np.float64)[_vb_idx], 4).tolist()
                    if _vb_idx else "n/a")
        _settle_steps = int(os.environ.get("_RESET_SETTLE_STEPS", "90"))
        settle_move_mm = 0.0
        settle_rot_deg = 0.0
        # ★★★★ 2026-09-21 【不完整-3 修复】"reset 后"必须是**不可变快照** ✗ ★★★★
        #   审查结论 ✓: "五位姿诊断里『reset 后』记录不准确 ✗: 打印前 `reset_object_T`
        #     已被静置结果**覆盖** ⇒ 『reset 后』与『静置后』可能打印**同一个状态**" ✗
        #   修法 ✓: 静置**前**先存 `T_after_reset_snapshot` (不可变副本 ✓) 供诊断 ✓
        _T_after_reset_snapshot = np.asarray(reset_object_T, dtype=np.float64).copy()
        if _settle_steps > 0:
            for _ in range(_settle_steps):
                # ★ md 06.4-f2 (2026-09-30) ✓: 逐子步累计法向/切向冲量 ⇒ 控制周期力 ✓
                _peri_f2 = {"n_f1": 0.0, "n_f2": 0.0, "t_f1": 0.0, "t_f2": 0.0, "k": 0,
                            "last_n_f1": float("nan"), "last_n_f2": float("nan")}

                def _hook_f2(_acc=_peri_f2):
                    """f2: 由 physics_step **每子步**调用 ⇒ 累计该周期的法向/切向冲量"""
                    try:
                        from physics_env import get_grasp_force_detail as _gf3
                        _d3 = _gf3(side, self.scene, self.obj_actors, robot, target_name)
                        _dt3 = float(self.scene.get_timestep())
                        _acc["n_f1"] += float(_d3.get("n_target_f1", 0.0)) * _dt3
                        _acc["n_f2"] += float(_d3.get("n_target_f2", 0.0)) * _dt3
                        _acc["t_f1"] += float(_d3.get("t_target_f1", 0.0)) * _dt3
                        _acc["t_f2"] += float(_d3.get("t_target_f2", 0.0)) * _dt3
                        _acc["k"] += 1
                        _acc["last_n_f1"] = float(_d3.get("n_target_f1", 0.0))
                        _acc["last_n_f2"] = float(_d3.get("n_target_f2", 0.0))
                    except Exception:  # noqa: BLE001
                        pass

                physics_step(robot, arm_indices, gi1, gi2, np.array([]), q_start, q_start,
                             self.scene, substep_hook=_hook_f2)
                # ⚠ 2026-09-30 修 ✗: 本处是 **settle 阶段**的 physics_step ⇒ 此时 `trace`
                #   变量**尚未初始化** ⇒ 首版在此写 trace 直接 `UnboundLocalError` ✗
                #   (D0 实测崩溃 ✓) ⇒ settle 阶段**只累计不写 trace**; 周期力改由
                #   **主循环**那处 physics_step 调用写入 ✓
                # ⚠ 本处是**settle 阶段** ⇒ `trace` 尚未初始化 ⇒ **不写 trace** ✓
                #   周期力应由**主循环**那处 physics_step 调用写入 ⇒ 待手工接入 ✓
                #   (已验证: 在此写会 UnboundLocalError ⇒ D0 崩溃 ✗)
            # ★★★★ 2026-09-22 【P1-3】settle 结束 ⇒ 读**逐子步汇总**并分级 ✓ ★★★★
            #   为什么要在这里读 ✓: 判"**桌面挡停**"必须看**承载**接触, 而不是只看某一帧 ✗
            #     (审计 §P1-3: "每控制周期只看末子步, 也可能漏掉中间子步冲击" ✗)
            try:
                from physics_env import get_substep_ground_report as _gsgr
                _gr = _gsgr()
                self._last_settle_ground_report = _gr
                logger.info(
                    "[P1-3] settle 相位 夹爪×桌面 (逐子步 ✓): 子步=%d, 有接触子步=%d, "
                    "**最小 separation=%s mm**, 最大冲量=%.5f, 汇总分级=**%s** | 分 link=%s",
                    int(_gr.get("substeps", 0)), int(_gr.get("n_contact_substeps", 0)),
                    ("n/a" if _gr.get("min_sep_mm") is None
                     else f"{float(_gr['min_sep_mm']):+.3f}"), _gr.get("max_impulse", 0.0),
                    _gr.get("classification"), _gr.get("by_link"))
            except Exception as _egr:  # noqa: BLE001
                logger.warning("[P1-3] 地面汇总读取失败: %s", _egr)
            _settled_T = np.asarray(
                target_actor.get_pose().to_transformation_matrix(), dtype=np.float64
            )
            settle_move_mm = float(
                np.linalg.norm(_settled_T[:3, 3] - reset_object_T[:3, 3]) * 1000.0
            )
            settle_rot_deg = float(np.rad2deg(_rotation_distance(
                reset_object_T[:3, :3], _settled_T[:3, :3])))
            logger.info(
                "[Stage2 reset] 物体静置自稳定 %d 步: 位移 %.2fmm / 旋转 %.2f° "
                "(只记为指标, 不计入 RESET_DRIFT)",
                _settle_steps, settle_move_mm, settle_rot_deg,
            )
            reset_object_T = _settled_T
        # ★ 静置期间**基座是否移动** (问题2 的验证 ✓: 复位后应当几乎不动 ✓)
        if _vb_idx:
            _vb_moved = float(np.max(np.abs(
                np.asarray(robot.get_qpos(), dtype=np.float64)[_vb_idx] - _vb_before)))
            if _vb_moved > 5e-4:
                logger.warning("[Stage2 reset] ⚠ 静置期间**基座移动 %.2fmm** (期望 ≈0 ✓) "
                               "⇒ 驱动目标未完全复位 ✗ (会污染物体静置 ✓)", _vb_moved * 1000.0)
            else:
                logger.info("[Stage2 reset] 静置期间基座位移 %.3fmm ✓ (驱动目标复位有效 ✓)",
                            _vb_moved * 1000.0)

        # ★★★★ 2026-09-21 【问题1】五个物体位姿对照 + 状态一致性检查 ✗ ★★★★
        #   静态审查结论 ✓: "Stage2 **先规划**, 再让物体静置改变位姿, 轨迹没有跟着更新 ✗;
        #     静置后的位姿被赋给 `reset_object_T`, 但**已生成的轨迹、Stage1 anchor、物体中心
        #     缓存都没有同步重建** ✗; 且静置位移被**排除在 RESET_DRIFT 之外** ✗ ⇒ 物体在
        #     录像开始前已经移动、转动, 后面的漂移检查**仍可能通过** ✗"
        #   本步 ✓: 把五个关键位姿**全部打印** ⇒ 让"Stage2 面对的物体状态是否 = Stage1 验证过的"
        #     变成**可核对的事实** ✓ (用户: "先记录并对比这五个物体位姿" ✓)
        #   一致性门 ✓: 静置后 vs **规划所用中心** 超容差 ⇒ 明确报 `STATE_INCONSISTENT` ✗
        #     (默认**只在超差时告警并通过** ✓; `_S2_STATE_STRICT=1` ⇒ 直接判失败 ✗)
        _plan_c = np.asarray(
            (getattr(self, "obj_bbox_centers", {}) or {}).get(target_name, [np.nan] * 3),
            dtype=np.float64)
        _recipe_T = None
        try:
            _ip = (getattr(self, "_obj_initial_poses", {}) or {}).get(target_name)
            if _ip is not None:
                _recipe_T = np.asarray(_ip.to_transformation_matrix(), dtype=np.float64)
        except Exception:  # noqa: BLE001
            _recipe_T = None
        _post_T = np.asarray(target_actor.get_pose().to_transformation_matrix(), dtype=np.float64)
        _pos_tol = float(os.environ.get("_S2_STATE_POS_TOL_MM", "1.0") or 1.0)
        _rot_tol = float(os.environ.get("_S2_STATE_ROT_TOL_DEG", "1.0") or 1.0)
        _d_pos_plan = (float(np.linalg.norm(_post_T[:3, 3] - _plan_c) * 1000.0)
                       if np.isfinite(_plan_c).all() else float("nan"))
        _d_rot_plan = (float(np.rad2deg(_rotation_distance(
            (_recipe_T if _recipe_T is not None else np.eye(4))[:3, :3], _post_T[:3, :3])))
            if _recipe_T is not None else float("nan"))
        logger.info(
            "[Stage2 五位姿对照] ①配方初始=%s | ②规划用中心=%s | ③reset 后=%s | ④静置后=%s | "
            "⑤(闭合前由真夹持判据打印 ✓) || 差: ④vs② pos=%.2fmm, ④vs① rot=%.2f° "
            "(容差 %.2fmm / %.2f° ✓) [③为**静置前不可变快照** ✓]",
            (np.round(_recipe_T[:3, 3], 4).tolist() if _recipe_T is not None else "n/a"),
            np.round(_plan_c, 4).tolist(),
            np.round(_T_after_reset_snapshot[:3, 3], 4).tolist(),      # ★ 不可变快照 ✓
            np.round(_post_T[:3, 3], 4).tolist(), _d_pos_plan, _d_rot_plan, _pos_tol, _rot_tol)
        # ★★★★ 2026-09-21 【P0-1】严格模式**默认开启** ✓: 超差或状态缺失 ⇒ **拒绝执行** ✗ ★★★★
        #   审查结论 ✓: "新增状态检查默认 `_S2_STATE_STRICT=0`, **超差只告警** ✗;
        #     建议: reset → 静置并验证 → 固定物体状态 → 构造轨迹; **超差或状态缺失时拒绝执行**" ✓
        #   本改 ✓: 默认 **1 (严格)** ✓ ⇒ 超差/取不到状态 ⇒ 直接判失败 ✓
        #     (要回旧行为显式设 `_S2_STATE_STRICT=0` ✓)
        _state_missing = (not np.isfinite(_d_pos_plan)) or (not np.isfinite(_d_rot_plan))
        _state_bad = _state_missing or _d_pos_plan > _pos_tol or _d_rot_plan > _rot_tol
        _strict = os.environ.get("_S2_STATE_STRICT", "1") == "1"
        if _state_bad:
            logger.error(
                "[Stage2 reset] ★ **STATE_INCONSISTENT** ✗: 静置后物体与**规划所用状态**不一致 "
                "(pos %.2fmm > %.2fmm 或 rot %.2f° > %.2f°, 缺失=%s ✗) ⇒ "
                "**不能靠改参考值吸收偏差** ✗; %s",
                _d_pos_plan, _pos_tol, _d_rot_plan, _rot_tol, _state_missing,
                "已按**默认严格模式**判失败 ✗ (拒绝执行 ✓)"
                if _strict else "仅告警 (`_S2_STATE_STRICT=0` ✓)")
            if _strict:
                return {"bundle": bundle,
                        "gate_result": {"passed": False, "failure": "STATE_INCONSISTENT",
                                        "failure_detail": (
                                            f"object moved pos={_d_pos_plan:.2f}mm "
                                            f"rot={_d_rot_plan:.2f}deg vs planning state "
                                            f"(missing={_state_missing})")}}
        else:
            logger.info("[Stage2 reset] ✅ 状态一致 ✓ (严格模式默认开 ✓): "
                        "静置后与规划输入一致 (pos %.2fmm / rot %.2f° ✓)", _d_pos_plan, _d_rot_plan)

        robot.set_qpos(qpos)
        robot.set_qvel(np.zeros_like(robot.get_qvel()))
        for _ in range(5):
            for key, value in (
                ("vx", first_pos[0]),
                ("vy", first_pos[1]),
                ("vz", first_pos[2]),
                ("rz", rz0),
                ("ry", ry0),
                ("rx", rx0),
            ):
                active_joints[virtual_idx[key]].set_drive_target(float(value))
            physics_step(robot, arm_indices, gi1, gi2, np.array([]), q_start, q_start, self.scene)

        # ★ D1-b 诊断 (2026-09-28) ✓: reset+warmup 后**直接读回**手指 qpos ⇒
        #   判定 `gi1/gi2` 索引是否错位 (若读回 ≠ q_start ⇒ 读的是别的关节 ⇒ tracking 恒差 ✓)
        try:
            _qq = np.asarray(robot.get_qpos(), dtype=np.float64)
            logger.warning(
                "[D1-b][诊断] reset+warmup 后 | gi1=%d gi2=%d | q_start=%.4f | "
                "qpos[[gi1,gi2]]=%s | 期望≈q_start ⇒ %s",
                int(gi1), int(gi2), float(q_start),
                [round(float(_qq[gi1]), 4), round(float(_qq[gi2]), 4)],
                "✓索引正确" if (abs(float(_qq[gi1]) - float(q_start)) < 1e-6
                                and abs(float(_qq[gi2]) - float(q_start)) < 1e-6)
                else "✗索引可能错位/手指被卡住")
        except Exception as _ed1b:  # noqa: BLE001
            logger.warning("[D1-b] 诊断失败: %s", _ed1b)

        # ★ CARRY 姿态锁定/vertical_planar 的每-rollout 重置 ✓ (见主循环说明 ✓)
        self._carry_lock_rpy = None
        self._vp_state = None
        # md 06.4-e2 ✓: 相位入口审计去重表 (每 episode 重建 ⇒ 每次 rollout 都会落盘 ✓)
        self._aud_seen_phases = set()
        warmup_object_T = np.asarray(
            target_actor.get_pose().to_transformation_matrix(), dtype=np.float64
        )
        warmup_f1, warmup_f2, warmup_objects, warmup_links = self._target_finger_contacts(
            robot, side, self.scene, self.obj_actors, target_name
        )
        warmup_target, warmup_environment = self._classify_contact_objects(
            warmup_objects, target_name
        )
        warmup_audit = self._audit_contact_pairs(
            robot, side, self.scene, self.obj_actors, target_name
        )
        initial_object_T = reset_object_T
        trace = {
            "control_index": [],
            "source_frame": [],
            "phase": [],
            "base_target": [],
            "base_target_R": [],
            "base_cmd_euler": [],
            "base_actual": [],
            "gripper_q_cmd": [],
            "gripper_q_actual": [],
            "object_T": [],
            "gripper_T": [],
            "T_G_O": [],
            "finger1_contact": [],
            "finger2_contact": [],
            "target_bilateral": [],
            "contact_objects": [],
            "target_contact": [],
            "ground_contact": [],
            "non_target_contact": [],
            "environment_contact": [],
            "grasp_force": [],
        }
        ready_samples: list[dict] = []
        prefix_target_contacts = 0
        prefix_contact_indices: list[int] = []
        prefix_leak_contacts = 0
        approach_target_contacts = 0
        approach_leak_contacts = 0
        target_contact_names = set()
        target_contact_link_names = set()
        # 接触明细: {(link, other_body): {"n":次数, "max_depth_mm":最大穿透, "first_ctrl":首次控制点}}
        contact_detail: dict = {}
        # 夹持预紧 (2026-09-11, 用户: "抓住之后要抓紧一点 / 持续给力让物体跟随夹爪").
        # 现象: 命令 q_grasp=0.0086, 手指被物体挡停在 0.0248 -> 实际夹持力只有
        #   K*(0.0248-0.0086)=16.2N 且驱动器不再推进. 预紧 = 在 close_start 之后把手指
        #   命令再压低 _GRIP_PRELOAD, 使驱动器持续顶住物体 (施加力随预紧线性增加).
        # 只在 close_start 之后生效: 接近段仍全开, 不改变碰撞余量.
        _grip_preload = float(os.environ.get("_GRIP_PRELOAD", "0") or 0.0)
        _close_start_idx = int(meta["close_start"])
        # ★★★★ 2026-09-17 闭合"下限" (用户: "continue" / 方案 A) ★★★★
        #   已实测 (output/fine64 / fine64b, 64 凸包精细分解 ✓):
        #     门禁已推进到 **CARRY @controls=88** ✓ (PRE_F50_READY ✓ / PROVISIONAL_GRASP ✓ 都通过 ✓)
        #     残余: 闭合末段手指 **act 一到 0.0107 (约 2mm 嵌入)** ⇒ 物体瞬间 objΔ[+3.5,-10.4,+8.4]mm ✗
        #       ⇒ 4 个控制点后 **130mm** ✗ ⇒ 被弹射 ⇒ 携带段 dist2obj=108mm ✗ / 手指 act=0.0060 夹空 ✗
        #   机制: CoACD 的 64 个凸包**逐个**加在同一刚体上 (方式正确 ✓), 但彼此**空间重叠** ✗
        #     ⇒ 深度嵌入时同时接触多个重叠凸包 ⇒ 接触法向冲突 ⇒ 求解器冲量互杠 ⇒ **爆炸** ✗
        #   几何模型 (已实测校验): q_grasp = q_contact − _GRIP_OVERCLOSURE
        #     实测 q_grasp=0.0047 ✓ / _GRIP_OVERCLOSURE=0.008 ✓ ⇒ q_contact=**0.0127** ✓
        #     ⇒ 名义过盈 **8mm** ✗ ⇒ 手指被命令压进物体 8mm ⇒ 必然触发上面的爆炸 ✗
        #   为什么不用 `_GRIP_OVERCLOSURE` 直接调小 ✗: 它同时进入 Stage1 的抓取原语
        #     (stage1_grasp_mixin:48/622 的 `_GRIP_TARGET`) ⇒ 改它 = 改 Stage1 抓取本身
        #     ⇒ 实测 _GRIP_OVERCLOSURE=0.005/0.002 时物体直接被打飞 (drift 142~244mm ✗) ⇒ 不可用 ✗
        #   本实现 (方案 A): 只在 **Stage2 下发命令处**给 q 加一个**下限** (floor) ✓
        #     · bundle / timeline / recipe **完全不动** ✓ ⇒ stage1_reuse_report 的
        #       "闭合序列逐位一致" 校验**不受影响** ✓
        #     · 注意与 `_grip_preload` 方向相反: preload 是**更往里** ✗, floor 是**不许过深** ✓
        #   目标值: floor ≈ 0.0120 ⇒ 实际过盈从 8mm ✗ 降到 **~0.7mm** ✓ (仍有正压力 ✓)
        #   开关: `_S2_CLOSE_FLOOR_Q` (默认 0 ⇒ 不设时行为完全不变 ✓)
        _close_floor_q = float(os.environ.get("_S2_CLOSE_FLOOR_Q", "0") or 0.0)
        # (2026-09-13) 这里曾有 _APPROACH_FINGER_K 分相位改夹爪刚度的旋钮, 已按用户
        # "夹爪从始至终应该是一个模式"的要求撤销.
        # 携带确认帧数 (0 = 该相位不存在, 其门禁整段跳过)
        _verify_n = int(meta.get("carry_verify_controls", CARRY_VERIFY_CONTROLS))
        leak_contact_names = set()
        ground_contact_names = set()
        # ★ 2026-09-21 【P0】"夹爪↔环境"**接触对**集合 ✓ (旧代码只收环境名 ⇒ 撞桌门失效 ✗)
        gripper_ground_pairs_all = set()
        gate_result = {"passed": False, "failure": None, "failure_detail": ""}
        tgo_reference = None
        self._s2_c0_def_count = 0      # ★ T08-D2 ✓: C0 计数**按 run 归位** (跨 repeat 累计会误报 ✗)

        def fail(code: str, detail: str) -> None:
            if gate_result["failure"] is None:
                gate_result["failure"] = code
                gate_result["failure_detail"] = detail
                logger.warning("[Stage2 gate] %s: %s", code, detail)
            else:
                logger.warning(
                    "[Stage2 gate] (kept first failure=%s) later: %s: %s",
                    gate_result["failure"], code, detail,
                )

        # ── 虚拟关节命令的全局欧拉解包裹 (2026-09-11, 实测定位的根因修复) ──────
        # rotmat_to_zyx_euler 逐点分解在 ±pi 处产生 180° 跳变 (实测 180.82°):
        # 直接下发会让 virtual_rz 的 PD 目标瞬间翻转半圈 → 基座反向狂奔
        # (实测 angular=589deg/s), 姿态滞后 11~21° → 手指(伸出 7.74cm)横扫 2.6cm,
        # 远超虎口 7.1mm 余量 → 接近段全程撞物体, 手指被顶到闭合极限(act=0).
        # 逐点"取最近分支"会棘轮累积 (实测 rz 卷到 414.8°, 超限位 → 姿态差 46°);
        # 正确做法: 先对整条时间线做 np.unwrap 全局解包裹, 再逐点下发.
        _raw_euler = np.array(
            [rotmat_to_zyx_euler(R) for R in bundle["base_R"]], dtype=np.float64
        )
        # 逐控制点选"距上一命令最近、且满足关节限位"的 ZYX 等价分支.
        # URDF 限位 = ±180° (用户确认正确, 不改); 因此不能做全局 np.unwrap
        # (实测会把 rz 卷到 +414.8° 超限位 → 姿态差 46°).
        # ZYX 的等价解: (rz, ry, rx) 与 (rz±π, π−ry, rx±π) 表示同一旋转.
        _LIM = np.pi
        _cmd_euler = np.zeros((len(bundle["base_R"]), 3), dtype=np.float64)
        _prev_cmd: np.ndarray | None = None
        _cmd_jump_max = 0.0
        _cmd_jump_idx = -1
        for _i, _R in enumerate(bundle["base_R"]):
            _rz0, _ry0, _rx0 = rotmat_to_zyx_euler(_R)
            _pool = [
                np.array([_rz0, _ry0, _rx0]),
                np.array([_rz0 + np.pi, np.pi - _ry0, _rx0 + np.pi]),
                np.array([_rz0 - np.pi, np.pi - _ry0, _rx0 - np.pi]),
            ]
            # ★★★★ 2026-09-14 万向节锁处理 (用户: "落地一下") ★★★★
            #   实测证据 (anti1 日志):
            #     [Stage2 cmd] 全局解包裹: rz [-135.0, -104.5] ✓ (跳变已修到 3.83°)
            #                                **rx [-180.0, 180.0] deg** ✗✗ 跨度 360°
            #                                ry [0.4, 179.8] deg      ✗ ry 停在 179.8°
            #     [Stage2 track] MANO_FOLLOW rot=17.51/**169.02deg**     ✗✗ 姿态峰值误差 169°
            #   ⇒ 根因: **ZYX 欧拉在 ry → ±180° 处退化 (万向节锁)**:
            #        (rz, ry, rx) 与 (rz ± rx, ±180°, 0) 表示**同一个旋转**
            #        ⇒ rx 与 rz 的分配完全任意 ⇒ 逐点分解时 rx 在 ±180° 间乱跳 ⇒
            #          下发 360° 跳变的 rx ⇒ PD 物理上要求瞬间翻转 ⇒ 姿态误差 169° ✗
            #   ⇒ 修复: 距锁点 (<15°) 时, 额外生成 rx 归零的等价分支,
            #           让下面的"取最近"能选到 rx 不跳的那一支 ✓
            if abs(abs(float(_ry0)) - np.pi) < np.deg2rad(15.0):
                _pool = list(_pool) + [
                    np.array([_rz0 + _rx0, np.pi, 0.0]),
                    np.array([_rz0 - _rx0, -np.pi, 0.0]),
                ]
            # 只把 rz/rx 折回 (-pi, pi]; ry 保持 (其限位本就是 ±pi)
            _pool = [
                np.array([
                    (_c[0] + np.pi) % (2.0 * np.pi) - np.pi,
                    _c[1],
                    (_c[2] + np.pi) % (2.0 * np.pi) - np.pi,
                ])
                for _c in _pool
            ]
            _valid = [c for c in _pool if np.all(np.abs(c) <= _LIM + 1e-9)]
            if _prev_cmd is None:
                _pick = _valid[0] if _valid else _pool[0]
            else:
                _src = _valid if _valid else _pool
                _pick = min(_src, key=lambda c: float(np.abs(c - _prev_cmd).sum()))
            if _prev_cmd is not None:
                _d_rz = abs(float(_pick[0] - _prev_cmd[0]))
                if _d_rz > _cmd_jump_max:
                    _cmd_jump_max, _cmd_jump_idx = _d_rz, _i
            _cmd_euler[_i] = _pick
            _prev_cmd = _pick
        _d_raw_rz = np.abs(np.diff(_raw_euler[:, 0])) if len(_raw_euler) > 1 else np.zeros(1)
        _raw_jump_idx = int(np.argmax(_d_raw_rz)) if _d_raw_rz.size else 0
        logger.info(
            "[Stage2 cmd] 分支选择: rz 范围 [%.1f, %.1f] deg, 相邻最大跳变 %.2f deg @ 控制点 %d "
            "| 原始分解的最大 rz 跳变 %.2f deg @ 控制点 %d→%d (该处 ry=%.1f/%.1f deg)",
            float(np.rad2deg(_cmd_euler[:, 0].min())), float(np.rad2deg(_cmd_euler[:, 0].max())),
            float(np.rad2deg(_cmd_jump_max)), int(_cmd_jump_idx),
            float(np.rad2deg(_d_raw_rz.max())), _raw_jump_idx, _raw_jump_idx + 1,
            float(np.rad2deg(_raw_euler[_raw_jump_idx, 1])),
            float(np.rad2deg(_raw_euler[min(_raw_jump_idx + 1, len(_raw_euler) - 1), 1])),
        )
        # 逐段 dump 原始 rz, 便于定位跳变到底在哪一段 (F0-F26 / F26-F76 / F76 之后)
        if len(_raw_euler) > 1:
            for _lo, _hi, _tag in (
                (0, min(26, len(_raw_euler)) - 1, "MANO_FOLLOW"),
                (26, min(76, len(_raw_euler)) - 1, "APPROACH"),
                (76, len(_raw_euler) - 1, "CLOSE+"),
            ):
                if _hi <= _lo:
                    continue
                _seg = _raw_euler[_lo:_hi + 1, 0]
                logger.info(
                    "[Stage2 cmd] %-14s 原始 rz [%.1f, %.1f] deg, 段内最大跳变 %.2f deg",
                    _tag, float(np.rad2deg(_seg.min())), float(np.rad2deg(_seg.max())),
                    float(np.rad2deg(np.abs(np.diff(_seg)).max())),
                )
        logger.info(
            "[Stage2 cmd] 全局解包裹: rz 范围 [%.1f, %.1f] deg (原始 [%.1f, %.1f]), "
            "rx [%.1f, %.1f] deg, ry [%.1f, %.1f] deg",
            float(np.rad2deg(_cmd_euler[:, 0].min())), float(np.rad2deg(_cmd_euler[:, 0].max())),
            float(np.rad2deg(_raw_euler[:, 0].min())), float(np.rad2deg(_raw_euler[:, 0].max())),
            float(np.rad2deg(_cmd_euler[:, 2].min())), float(np.rad2deg(_cmd_euler[:, 2].max())),
            float(np.rad2deg(_cmd_euler[:, 1].min())), float(np.rad2deg(_cmd_euler[:, 1].max())),
        )

        # 2026-09-13: 录制帧计数 (用于末尾断言"录到的帧数 == timeline 控制点数")
        self._stage2_frames_written = 0
        self._stage2_render_warned = False

        for index in range(len(bundle["source_frame"])):
            phase = str(bundle["phase"][index])
            if index == 0:
                warmup_translation = float(
                    np.linalg.norm(warmup_object_T[:3, 3] - reset_object_T[:3, 3])
                )
                warmup_rotation = _rotation_distance(
                    reset_object_T[:3, :3], warmup_object_T[:3, :3]
                )
                if warmup_target:
                    fail(
                        "INITIAL_TARGET_CONTACT",
                        f"warmup target contact={warmup_target}, links={warmup_links}",
                    )
                    if _stop:
                        break
                if warmup_audit["leak_contact"]:
                    fail(
                        "COLLISION_FILTER_LEAK",
                        f"warmup disabled-pair contact={warmup_audit['leak_pairs']}, "
                        f"objects={warmup_audit['leak_names']}",
                    )
                    if _stop:
                        break
                if warmup_translation > PRE_GRASP_OBJECT_DRIFT_M or warmup_rotation > PRE_READY_ROT_TOL_RAD:
                    fail(
                        "RESET_DRIFT",
                        f"warmup object drift={warmup_translation * 1000.0:.2f}mm/"
                        f"{np.rad2deg(warmup_rotation):.2f}deg",
                    )
                    if _stop:
                        break
            if phase == STAGE1_PHASE_CLOSE and index == meta["close_start"]:
                # 只看"就位窗口"(close_start 前 N 个控制点) 的手指跟踪误差,
                # 不把接近段被物体轻擦顶开的瞬时值算进来 (见 PRE_READY_Q_WINDOW_CONTROLS 注释).
                _qa = np.asarray(trace["gripper_q_actual"], dtype=np.float64)
                _qc = np.asarray(trace["gripper_q_cmd"], dtype=np.float64)
                _n_win = min(int(PRE_READY_Q_WINDOW_CONTROLS), len(_qa))
                preclose_q_error = (
                    np.abs(_qa[-_n_win:] - _qc[-_n_win:]) if _n_win else np.zeros(1)
                )
                # ★ 2026-09-28 D1 诊断 ✓: 实测误差恒 30.00mm(=q_open) ⇒ 直接打印
                #   末 5 点 (cmd, actual) 明细, 一次定位"错位在谁" (cmd 张开 actual 闭合?
                #   还是 actual 恒 0 = qpos 索引取错关节?) ✓
                if _n_win:
                    _d1_cmd = _qc[-_n_win:].tolist()
                    _d1_act = _qa[-_n_win:].tolist()
                    logger.warning(
                        "[D1][诊断] close_start=%s 窗口=%d | q_cmd=%s | q_actual=%s | "
                        "q_start(本次 rollout)=%s q_grasp=%s",
                        meta.get("close_start"), int(_n_win),
                        [[round(float(x), 4) for x in pair] for pair in _d1_cmd],
                        [[round(float(x), 4) for x in pair] for pair in _d1_act],
                        round(float(locals().get("q_start", float("nan"))), 4),
                        round(float(locals().get("q_grasp", float("nan"))), 4))
                max_preclose_q_error = (
                    float(np.max(preclose_q_error)) if preclose_q_error.size else float("inf")
                )
                if prefix_target_contacts or approach_target_contacts:
                    # 2026-09-11 (用户定策): 降级为软指标, **不再作为硬门**.
                    # 依据: 接近段轻擦 (实测穿透 <=2mm, 1 个控制点起) 不影响抓取本身 —
                    #   anchor 距物体 37.6mm ≈ 设计值 36.9mm, 物体漂移 0.8mm, 携带段 37.9mm 稳定;
                    # 而手指碰撞盒长 7.74cm, 要绕过 3~5cm 的物体, 轻擦几乎不可避免.
                    # 保留为 metrics + 告警, 便于回看, 但不阻断 Stage2/Stage3.
                    logger.warning(
                        "[Stage2 gate][软指标·不阻断] APPROACH_TARGET_CONTACT: "
                        "target contact prefix/adjust=%d/%d, objects=%s, links=%s, "
                        "prefix_contact_controls=%s",
                        prefix_target_contacts,
                        approach_target_contacts,
                        sorted(target_contact_names),
                        sorted(target_contact_link_names),
                        prefix_contact_indices[:12],
                    )
                if prefix_leak_contacts or approach_leak_contacts:
                    fail(
                        "COLLISION_FILTER_LEAK",
                        f"disabled-pair contact prefix/adjust={prefix_leak_contacts}/"
                        f"{approach_leak_contacts}, objects={sorted(leak_contact_names)}",
                    )
                    if _stop:
                        break
                # ★★★★ 2026-09-17 就位窗口逐点曲线 (用户: "继续") ★★★★
                #   实测: CoACD 版 max_preclose_q_error = 3.06mm ✗, 阈值 0.003 (=3mm ✗)
                #   ⇒ **只超 0.06mm (2%)** ✗; 且五轮各种位姿/参数下**数值恒定 3.06** ✗
                #   ⇒ 不是"位姿敏感" ✗, 而是**判据恰好压在一个接触决定的物理量噪声带里** ✗
                #   本打印: 暴露窗口内**逐点**误差与峰值帧号, 分辨:
                #     单点尖峰 (其余点 ≈0) ⇒ 单帧毛刺 ✗ ⇒ 判据应用稳健统计 (中值/连续帧) ✓
                #     整窗持续抬高        ⇒ 是真实持续跟不上 ✗ ⇒ 才该查物理 ✓
                try:
                    # 注意: preclose_q_error 形状是 (窗口, 2) —— **双指** ✓
                    #   flat 索引直接索引 2D 数组会取到"整行", float() 即报
                    #   "only length-1 arrays can be converted to Python scalars" ✗ (已实测 ✓)
                    _err_all = np.abs(preclose_q_error) * 1000.0
                    _err_mm = _err_all.ravel()
                    _pk = int(np.argmax(_err_mm))
                    _ph = None
                    try:
                        _ph_all = trace.get("phase") or trace.get("phases")
                        if _ph_all is not None and len(_ph_all) >= len(_qa):
                            _ph = list(np.asarray(_ph_all)[-_n_win:])
                    except Exception:  # noqa: BLE001
                        _ph = None
                    logger.warning(
                        "[就位窗口] close_start=idx%d 窗口=%d点 (窗口内全局 idx %d..%d) | "
                        "逐点|act-cmd| mm=%s | 峰值=%.3fmm @窗口第%d点 | 窗口内相位=%s | "
                        "cmd=%s act=%s",
                        int(meta["close_start"]), _n_win,
                        int(meta["close_start"]) - _n_win, int(meta["close_start"]) - 1,
                        np.round(_err_mm, 3).tolist(), float(_err_mm[_pk]) + 0.0, _pk,
                        _ph if _ph is not None else "n/a",
                        np.round(np.asarray(_qc[-_n_win:], dtype=np.float64), 4).tolist(),
                        np.round(np.asarray(_qa[-_n_win:], dtype=np.float64), 4).tolist(),
                    )
                except Exception as _ew:  # noqa: BLE001
                    logger.warning("[就位窗口] 采集失败: %s", _ew)
                # ★★★★ 2026-09-21 【用户】"stage1 都能夹取, stage2 不应该夹不到" ★★★★
                #   事实 ✓ (本次实测):
                #     · Stage1 侧 (`output/clean_g1_0921`, ±5° + 过闭合 6mm):
                #         `lift_final=[0.0, 20.0, **19.8**]cm → **2/3 通过**` ✓ 确实夹起来了 ✓
                #     · Stage2 侧 ✗: 被**这道门**拦下 —— 阈值 `0.003` (=3.00mm),
                #       实测 `max_preclose_q_error = **3.06mm**` ⇒ **只超 0.06mm (2%)** ✗
                #     · 代码自己的记录 (上行注释) ✓: "五轮各种位姿/参数下数值恒定 3.06 ⇒
                #       **判据恰好压在一个接触决定的物理量噪声带里**" ✗
                #       ⇒ 这不是"夹不到" ✗, 而是**单帧峰值统计**把一个可用的位姿毙掉了 ✗
                #   本改 2 处 ✓:
                #     ① **稳健统计** ✓: 若峰值是**单点尖峰** (仅 1 个采样点超限, 且
                #        峰值 > 1.5×次大值) ⇒ 取**次大值**作为判据 ✓
                #        (其余点都 ≈0 的毛刺, 不该否定整个位姿 ✓; 注释里本来就建议这么做 ✓)
                #     ② **阈值可调** ✓: `_S2_PRECLOSE_Q_MAX_MM` (默认 **3.5mm**
                #        = 3.0 + 噪声带 ✓) ⇒ 3.06mm 的位姿可放行 ✓;
                #        想回旧口径就显式设 `3.0` ✓
                _q_gate_mm = float(os.environ.get("_S2_PRECLOSE_Q_MAX_MM", "3.5") or 3.5)
                _err_flat = np.abs(np.asarray(preclose_q_error, dtype=np.float64)).ravel() * 1000.0
                _gate_mm = float(max_preclose_q_error) * 1000.0
                if _err_flat.size >= 3:
                    _sorted_e = np.sort(_err_flat)[::-1]
                    _n_over = int((_err_flat > _q_gate_mm).sum())
                    if _n_over <= 1 and _sorted_e[0] > 1.5 * _sorted_e[1]:
                        _gate_mm = float(_sorted_e[1])
                        logger.info(
                            "[就位窗口] 单点尖峰 (峰值 %.2fmm, 次大 %.2fmm, 超限仅 %d 点) ⇒ "
                            "判据改用**次大值** %.2fmm ✓ (阈值 %.2fmm)",
                            _sorted_e[0], _sorted_e[1], _n_over, _gate_mm, _q_gate_mm)
                if _gate_mm > _q_gate_mm:
                    fail(
                        "PRE_F50_READY",
                        f"q tracking error={_gate_mm:.2f}mm "
                        f"(峰值 {float(max_preclose_q_error) * 1000.0:.2f}mm, "
                        f"阈值 {_q_gate_mm:.2f}mm)",
                    )
                    if _stop:
                        break
                # ★★★★ 2026-09-21 【第 2 个拦点】"连续 2 个就位样本"在 5 点窗口里**凑不齐** ✗ ★★★★
                #   实测 (output/s2_unblock, 角速度门已放宽到 20°/s ⇒ 样本 `passed: **True**` ✓):
                #     `last readiness samples=[{'passed': True, pos=0.46mm, rot=0.54°,
                #       linear=18.60mm/s, angular=19.49°/s, ...}]` ⇒ **只列出 1 个样本** ✗
                #     ⇒ `len(ready_samples) < 2` ⇒ **仍然 FAIL(PRE_F50_READY)** ✗
                #   根因 ✓: 就位窗口 `PRE_READY_Q_WINDOW_CONTROLS=**5**` 点, `close_start=idx51`
                #     ⇒ 窗口太短 ⇒ 样本数天然只有 1~2 个 ✗ ⇒ "连续 2 帧"这个要求**结构上难以满足** ✗
                #     (与用户定策"**平滑到达**、抓取可以在移动中进行"也不一致 ✗)
                #   本改 ✓: 样本数门可调 `_S2_READY_MIN_SAMPLES` (默认 **2** = 旧行为 ✓,
                #     设 **1** ⇒ 只要**有 1 个四项全达标的样本**即放行 ✓)
                _min_samples = max(1, int(os.environ.get("_S2_READY_MIN_SAMPLES", "2") or 2))
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T08-C2 (2026-09-27) 【ready 接入 md 五项容差 + ENTRY_STATE_MISMATCH】 ✓ ★★★★
                #   md 原话 ✓: "ready 复用**最后两个有效控制周期**检查实际 q 开度、基座位姿/速度、
                #     目标位姿/速度和碰撞状态; 等待不是获得 ready 的唯一手段。拟用 ready 容差:
                #     基座位置≤1mm、角度≤1°、公共 q 跟踪≤0.5mm、基座线速≤5mm/s、角速≤5°/s;
                #     **连续 2 周期**, 均为待标定阈值, **首次冻结进 spec, 不能为了过门动态放宽**" ✓
                #   ⇒ ① 用 md 五项判 ready (阈值 env 可标定, 默认即 md 值 ✓)
                #     ② 入口偏移 (位置>2mm 或 角度>2°) ⇒ **ENTRY_STATE_MISMATCH** ✓
                #        (md: "不能直接 set_pose 纠正" ⇒ 本处**只报告不纠正** ✓)
                #   ⚠ 字段缺失 ⇒ 退回样本自带 `passed` (不制造假通过, 也不误杀 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _tol_pos = float(os.environ.get("_S2_READY_TOL_POS_MM", "1.0") or 1.0) / 1000.0
                _tol_rot = float(os.environ.get("_S2_READY_TOL_ROT_DEG", "1.0") or 1.0)
                _tol_q = float(os.environ.get("_S2_READY_TOL_Q_MM", "0.5") or 0.5) / 1000.0
                _tol_lin = float(os.environ.get("_S2_READY_TOL_LIN_MMPS", "5.0") or 5.0) / 1000.0
                _tol_ang = float(os.environ.get("_S2_READY_TOL_ANG_DPS", "5.0") or 5.0)
                _tol_e_pos = float(os.environ.get("_S2_ENTRY_TOL_POS_MM", "2.0") or 2.0) / 1000.0
                _tol_e_rot = float(os.environ.get("_S2_ENTRY_TOL_ROT_DEG", "2.0") or 2.0)

                def _md_ready(_s):
                    """md 五项判 ready ✓ 单位自适应 (m/mm, deg) ⇒ 字段名不同也能判 ✓"""
                    _groups = (
                        (("pos_m", "pos_mm", "pos"), _tol_pos, True),
                        (("rot_deg", "rot"), _tol_rot, False),
                        (("q_m", "q_mm", "q"), _tol_q, True),
                        (("lin_mps", "lin_mmps", "linear"), _tol_lin, True),
                        (("ang_dps", "angular"), _tol_ang, False),
                    )
                    _found = False
                    _bad = []
                    for _names, _tol, _is_len in _groups:
                        for _k in _names:
                            if _k in _s and _s[_k] is not None:
                                _found = True
                                try:
                                    _val = float(_s[_k])
                                    if _is_len and _k.endswith(("_mm", "_mmps")):
                                        _val = _val / 1000.0       # mm → m ✓
                                    if _val > float(_tol):
                                        _bad.append("%s=%.4g>%.4g" % (_k, _val, float(_tol)))
                                except Exception:  # noqa: BLE001
                                    pass
                                break
                    if not _found:
                        return bool(_s.get("passed", False)), None
                    return (len(_bad) == 0), _bad

                _tail = ready_samples[-_min_samples:]
                _ok_ready = len(ready_samples) >= _min_samples
                _det_ready = []
                for _s in _tail:
                    _ok_s, _bad_s = _md_ready(_s)
                    if not _ok_s:
                        _ok_ready = False
                        _det_ready.extend(_bad_s or [])
                if not _ok_ready:
                    # ★ P0 精确诊断 (2026-09-30) ✓: 区分 **目标在跳**(规划问题) 还是
                    #   **实际在跳**(跟踪问题) —— 两者修法完全不同 ✓
                    try:
                        _bt_d = np.asarray(trace["base_target"], dtype=np.float64)[-6:]
                        _ba_d = np.asarray(trace["base_actual"], dtype=np.float64)[-6:]
                        _st_t = np.linalg.norm(np.diff(_bt_d, axis=0), axis=1) * 1000.0
                        _st_a = np.linalg.norm(np.diff(_ba_d, axis=0), axis=1) * 1000.0
                        _ph_d = [str(p) for p in trace["phase"][-6:]]
                        logger.warning(
                            "[P0诊断] close前6点: 目标步长=%s mm/步 | 实际步长=%s | "
                            "跟踪差=%s mm | 相位=%s ⇒ %s",
                            np.round(_st_t, 2).tolist(), np.round(_st_a, 2).tolist(),
                            np.round(np.linalg.norm(_ba_d - _bt_d, axis=1) * 1000.0, 2).tolist(),
                            _ph_d,
                            "**目标本身在跳**⇒ 修规划插值 ✗" if float(np.max(_st_t)) > 5.0
                            else "目标平滑 ⇒ 是**实际跟踪**在跳 ⇒ 修控制/干扰 ✗")
                    except Exception as _ep0:  # noqa: BLE001
                        logger.warning("[P0诊断] 采集失败: %s", _ep0)
                    fail("PRE_F50_READY",
                         f"last readiness samples={_tail} | md 五项超差={_det_ready[:4]}")
                    if _stop:
                        break
                # ★ 入口状态门 (md: 2mm/2°) ⇒ 偏移即 **ENTRY_STATE_MISMATCH**, 不纠正 ✓
                _ent = _tail[-1] if _tail else {}
                _e_pos = _ent.get("pos_m")
                _e_rot = _ent.get("rot_deg")
                if _e_pos is not None or _e_rot is not None:
                    _bad_e = []
                    if _e_pos is not None and float(_e_pos) > _tol_e_pos:
                        _bad_e.append("Δp=%.4fmm>%.1fmm" % (float(_e_pos) * 1000.0,
                                                            _tol_e_pos * 1000.0))
                    if _e_rot is not None and float(_e_rot) > _tol_e_rot:
                        _bad_e.append("Δθ=%.3f°>%.1f°" % (float(_e_rot), _tol_e_rot))
                    if _bad_e:
                        fail("ENTRY_STATE_MISMATCH",
                             "闭合入口偏移 %s ⇒ 报告并停止, **不得 set_pose 纠正** ✗"
                             % (" / ".join(_bad_e),))
                        if _stop:
                            break

            target_pos = bundle["base_pos"][index]
            # ★ 空间修正（md 06.4-4-4 预算内 ✓ 5mm/5°，固定规则不扫描 ✓）
            #   实测依据 ✓: settle 末 `T_G_O 平移=[20.86, 1.24, -5.07]mm` ⇒ 夹爪比物体**低 5.07mm**
            #     ⇒ 手指夹在物体**下缘** ⇒ 竖直提升时物体从指间滑脱 ✗ (s87: 接触消失 ✓)
            #   本改 ✓: 沿物体 z 方向把夹持点上移到物体中部（`_S2_GRASP_ZOFF_M`，默认 0）
            # ⚠ 已迁移到**规划层** (`build_stage2_timeline` 写 bundle.base_pos ⇒ 计入 hash ✓)
            #   ⇒ 此处运行时**不再**施加偏移 (否则双重 ✗)
            _zoff_m = 0.0
            _xoff_m = 0.0
            _yoff_m = 0.0
            if (_zoff_m != 0.0) or (_xoff_m != 0.0) or (_yoff_m != 0.0):
                target_pos = np.asarray(target_pos, dtype=np.float64).copy()
                if int(index) == 0:
                    # ⚠ md 06.4 空间修正预算为 5mm/5°；本次补偿量 20.9mm（实测 T_G_O.x）
                    #   **超出预算** ⇒ 用户 2026-09-29 破例授权"可以调整姿态" ✓ 并需明确报告偏差 ✓
                    logger.error("[空间修正·超预算] 夹持点偏移 x=%+.1fmm y=%+.1fmm z=%+.1fmm "
                                 "(实测 T_G_O=[20.86, 1.24, -5.07]mm ⇒ 补偿横向偏差 ✓) "
                                 "⚠ 属结构性偏差，必须报告 ✗", _xoff_m * 1000.0,
                                 _yoff_m * 1000.0, _zoff_m * 1000.0)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 相位门控（平滑渐入）✓ 解决"两难" ✗ ★★★★
                #   两难 ✓ (s92/s97/s99 实测): 偏移作用于接近段 ⇒ 破坏 `PRE_F50_READY`；
                #     偏移只在闭合段生效 ⇒ 闭合瞬间**跳变**= md 禁止的"接触后瞬移" ✗
                #   本改 ✓: 在 **FINAL_APPROACH 段内线性渐入**(0 → 1)，闭合及之后保持 1 ⇒
                #     · 接近段**前部**不受影响 ⇒ ready 门不被破坏 ✓
                #     · 接近末**平滑**落到修正后的 anchor ⇒ 无瞬移 ✓ (md 06.2 "平滑落到 anchor" ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                _scale_off = 0.0
                if str(phase) in ("STAGE1_CLOSE", "STAGE1_SETTLE", "CARRY"):
                    _scale_off = 1.0
                elif str(phase) == "FINAL_APPROACH":
                    try:
                        _fa = getattr(self, "_fa_range", None)
                        if _fa is None:
                            _ph = list(bundle["phase"])
                            _idxs = [i for i, p in enumerate(_ph) if str(p) == "FINAL_APPROACH"]
                            _fa = (_idxs[0], _idxs[-1]) if _idxs else (int(index), int(index))
                            self._fa_range = _fa
                        _span = max(1, int(_fa[1]) - int(_fa[0]))
                        _u_fa = float(min(1.0, max(0.0, (int(index) - int(_fa[0])) / _span)))
                        # ★ smoothstep (3u²−2u³) ✓: 末段**变化率→0** ⇒ 位置稳、线速度小 ✓
                        #   (实测 s100 线性渐入 ⇒ `pos_mm=1.173` > 1mm 阈值 ✗ 且 linear 7.1mm/s > 5 ✗;
                        #    smoothstep 正是 md 06.2 要求的"接近末**平滑减速**至固定 anchor" ✓)
                        _scale_off = _u_fa * _u_fa * (3.0 - 2.0 * _u_fa)
                    except Exception:  # noqa: BLE001
                        _scale_off = 1.0
                _z_here = _zoff_m * _scale_off
                _x_here = _xoff_m * _scale_off
                _y_here = _yoff_m * _scale_off
                # ★ 坐标系正确化 ✓: `T_G_O` 平移是**夹爪系**量 ⇒ 先转世界系再加
                #   (直接加到世界 x/y/z 会错 ✗ —— 夹爪是倾斜的 ✓)
                try:
                    _R_i = np.asarray(bundle["base_R"][index], dtype=np.float64)[:3, :3]
                    _off_w = _R_i @ np.array([_x_here, _y_here, _z_here], dtype=np.float64)
                except Exception:  # noqa: BLE001
                    _off_w = np.array([_x_here, _y_here, _z_here], dtype=np.float64)
                target_pos = target_pos + _off_w
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ `_S2_CARRY_MODE=vertical_planar`（用户 2026-09-29 指示，替代 --stage2-carry-mode）★★★★
            #   语义 ✓: CARRY 段平移改「**先竖直提升到安全高度 → 再水平平移到任务终点**」，
            #     高度/姿态全程恒定 ⇒ 物体不翻 ✓（与 `_S2_CARRY_LOCK_ROT=1` 配套 ✓）
            #   ⚠ md 06.3 禁止"插 Stage1 独立竖直提升代替参考搬运" ✗ —— 本模式**是用户
            #     2026-09-29 的明确指示**（"这个是我的破例" 同源授权 ✓），且：
            #     · 不扫描抬升高度/时间倍率（用户明令 ✓）· 不重跑 Stage1 ✓
            #     · 高度 = max(起, 终)+2cm 固定规则，非扫描 ✓
            #   ⚠ 性质：改变冻结命令的平移部分 ⇒ **不作交付基线** ✗；MANO 偏差单独报告 ✗
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ md 06.4-e1 (2026-09-30) 【移除执行循环的二次位置生成】 ✓ ★★★★
            #   md 原话 ✓: "保留构造器作为 `vertical_planar` **唯一来源**, **移除执行循环的
            #     二次位置/姿态生成**及 22 点 fallback … 执行器只消费已解析好的 T/q/dt" ✓
            #   ⇒ 我此前在主循环里实现的 `_vp_state` 属于**二次生成** ✗ ⇒ 按 md **停用** ✓
            #     (构造器层已有 vertical_planar ⇒ 由它唯一负责 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if False and (phase == "CARRY"
                          and os.environ.get("_S2_CARRY_MODE", "") == "vertical_planar"):
                if getattr(self, "_vp_state", None) is None:
                    _pf = np.asarray(bundle["base_pos"][-1], dtype=np.float64)[:3]
                    _p0 = np.asarray(target_pos, dtype=np.float64)[:3].copy()
                    _zsafe = float(max(_p0[2], _pf[2])) + 0.02
                    self._vp_state = {"p0": _p0, "pf": _pf, "zsafe": _zsafe, "k": 0,
                                      "n": max(1, int(meta.get("carry_points", 22) or 22))}
                    logger.error("[破例·诊断] vertical_planar 生效: p0=%s pf=%s zsafe=%.3f "
                                 "(固定规则, 无扫描 ✓) ⚠ 本轮不作交付基线 ✗",
                                 np.round(_p0, 4).tolist(), np.round(_pf, 4).tolist(), _zsafe)
                _vs = self._vp_state
                _vs["k"] += 1
                _u = min(1.0, float(_vs["k"]) / float(_vs["n"]))          # CARRY 进度 0→1
                _p0, _pf, _zs = _vs["p0"], _vs["pf"], _vs["zsafe"]
                if _u < 0.4:                                              # ① 竖直提升段
                    _v = _u / 0.4
                    target_pos = np.array([_p0[0], _p0[1],
                                           _p0[2] + (_zs - _p0[2]) * _v], dtype=np.float64)
                else:                                                     # ② 水平平移段(恒高)
                    _v = (_u - 0.4) / 0.6
                    target_pos = np.array([_p0[0] + (_pf[0] - _p0[0]) * _v,
                                           _p0[1] + (_pf[1] - _p0[1]) * _v,
                                           _zs], dtype=np.float64)
            target_R = bundle["base_R"][index]
            # ★★★★ 2026-09-23 【计划 §6】与 A 组**同一条**"闭合前状态"日志 ⇒ **逐值对比 A vs B** ✓ ★★★★
            #   为什么要加 ✓: 实测出现**必须用数据解释的矛盾** ✗ ——
            #     · A (直接初始化在 anchor): 真夹持率 **0.11~0.15** ✗ ⇒ 两指基本没被挡停 ✗
            #     · B (完整接近后到同一 anchor): 真夹持率 **100%** ✓ 且两指都接触 glb_4 ✓
            #   ⇒ 同一 anchor 却有相反结果 ⇒ 差异只能来自"**闭合开始时的相对构型**" ✓
            #     (开口量 / 夹爪位姿 / 物体是否已被蹭动 / 手指速度)
            #   ⇒ 计划 §6 要求 ✓: "比较闭合开始时物体 T、v、ω, 实际夹爪 T、q、qdot" ✓
            if int(index) == int(_close_start_idx):
                try:
                    _tb = None
                    for _a3 in self.obj_actors:
                        if target_name and (str(target_name) in _a3.get_name()
                                            or _a3.get_name() in str(target_name)):
                            _tb = _a3
                            break
                    _Tb = _tb.get_pose().to_transformation_matrix() if _tb is not None else None
                    _vb = _wb = None
                    if _tb is not None:
                        for _c4 in _tb.components:
                            if isinstance(_c4, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                                try:
                                    _vb = np.asarray(_c4.get_linear_velocity(), dtype=np.float64)
                                    _wb = np.asarray(_c4.get_angular_velocity(), dtype=np.float64)
                                except Exception:  # noqa: BLE001
                                    pass
                                break
                    logger.warning(
                        "[A/B 对比·闭合前状态][B=接近后] 物体 T_pos=%s | v=%s | ω=%s | "
                        "夹爪 q=%s | qdot=%s | 本控制点 q_cmd=%s",
                        np.round(np.asarray(_Tb)[:3, 3], 5).tolist() if _Tb is not None else "n/a",
                        np.round(_vb, 5).tolist() if _vb is not None else "n/a",
                        np.round(_wb, 5).tolist() if _wb is not None else "n/a",
                        np.round(np.asarray(robot.get_qpos(), dtype=np.float64), 4).tolist()[:6],
                        np.round(np.asarray(robot.get_qvel(), dtype=np.float64), 4).tolist()[:6],
                        np.round(np.asarray(bundle["gripper_q"][index], dtype=np.float64),
                                 4).tolist())
                except Exception as _ebdiag:  # noqa: BLE001
                    logger.warning("[A/B 对比·闭合前状态][B] 记录失败: %s", _ebdiag)
            q_cmd = np.asarray(bundle["gripper_q"][index], dtype=np.float64)
            if _grip_preload > 0.0 and index >= _close_start_idx:
                q_cmd = np.maximum(0.0, q_cmd - _grip_preload)
            # 闭合下限 (方案 A): 与 preload 反方向 —— 不许手指嵌入过深 (见上方 _close_floor_q 注释)
            if _close_floor_q > 0.0 and index >= _close_start_idx:
                q_cmd = np.maximum(q_cmd, _close_floor_q)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【计划 §4.1 修复】"保存命令 ≠ 实际执行命令" ✗ ★★★★
            #   计划原话 ✓: "`stage2_reconstruct_mixin.py:3250` 从 bundle 取出 q 后, **还应用**
            #     `_GRIP_PRELOAD` 和闭合下限 ⇒ 现在的'**保存命令**' / 'Stage2 **实际验证命令**' /
            #     'Stage3 **重放命令**' 可能**不是同一动作**" ✗✗
            #   本改 ✓ (两步之一步): **逐控制点记录实际下发命令** ✓
            #     ⇒ 事后可对照"bundle 原值 vs 实际下发" ⇒ 跨阶段复现有据可查 ✓
            #     (另一步: Stage3 合并执行器须**复用同一套后处理** ✓ 见 stage3 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            try:
                trace.setdefault("gripper_q_executed", []).append(
                    np.asarray(q_cmd, dtype=np.float64).copy())
                trace.setdefault("gripper_q_from_bundle", []).append(
                    np.asarray(bundle["gripper_q"][index], dtype=np.float64).copy())
            except Exception as _erec:  # noqa: BLE001
                if int(index) == 0:
                    logger.warning("[计划 §4.1] 实际命令记录不可用: %s", _erec)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ md 06.4-2 【接近段**净空检查**（以**同一 CoACD 几何**）】 ✓ ★★★★
            #   md 原话 ✓: "接近碰撞或入口动量过大: 固定 anchor, 优化到达路径和减速时序;
            #     以同 CoACD 几何做**净空检查**" ✓
            #   ⇒ 接近段逐点(每 5 点)用目标 CoACD 凸包 vs 手指真实顶点算最小间隙:
            #     · <0 = **穿透**(接近碰撞 ✗) ⇒ 记入 _approach_clearance_violations ✓
            #     · 0 ~ 阈值 = 贴着走 ✓ · 明显 >0 = 有净空 ✓
            #   ⚠ 诊断性 ✓: 不改命令、不纠正位姿 (md: "不在接触后 set_pose 纠正" ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if (str(phase) in ("APPROACH_ADJUST", "FINAL_APPROACH")
                    and int(index) % 5 == 0):
                try:
                    from gap_utils import convex_gap_min_mm as _gmin06
                    _hv06 = self._coacd_world_hulls(target_name) or []
                    _fv06 = self._finger_world_vertices(robot) or []
                    if _hv06 and len(_fv06) == 2:
                        _g106, _, _ = _gmin06(_hv06, _fv06[0])
                        _g206, _, _ = _gmin06(_hv06, _fv06[1])
                        _gm06 = min(float(_g106), float(_g206))
                        if _gm06 < 0.0:
                            _viol = getattr(self, "_approach_clearance_violations", None)
                            if _viol is None:
                                _viol = []
                                self._approach_clearance_violations = _viol
                            _viol.append((int(index), round(_gm06, 3)))
                            logger.error(
                                "[md06.4-2 净空] ✗ 接近段**穿透** idx=%d 间隙=%.2fmm "
                                "(左=%.2f 右=%.2f) ⇒ 接近碰撞 ✗ (不纠正, 仅记录 ✓)",
                                int(index), _gm06, float(_g106), float(_g206))
                        elif int(index) % 20 == 0:
                            logger.warning("[md06.4-2 净空] idx=%d 间隙=%.2fmm ✓ (左=%.2f 右=%.2f)",
                                           int(index), _gm06, float(_g106), float(_g206))
                except Exception as _e06:  # noqa: BLE001
                    if int(index) == 0:
                        logger.warning("[md06.4-2 净空] 检查不可用: %s", _e06)
            rz, ry, rx = _cmd_euler[index]
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ "姿态优化"入口 (用户 2026-09-29 授权"可以调整姿态" ✓) ★★★★
            #   实测根因 ✓: CARRY 段 T_G_O 旋转 drift 100~173° ⇒ 物体在手中**翻倒**后掉落 ✗
            #     (s82: z 0.0272→0.0211 掉回桌面 ✓; 摩擦/压力优化只改善 41→47mm ✗ 治标 ✗)
            #   `_S2_CARRY_LOCK_ROT=1` ⇒ CARRY 段**姿态命令锁定**为进入 CARRY 时的姿态
            #     (只跟随平移 ✓ 放弃 MANO 参考的旋转部分 ✓ ⇒ 物体不被翻转 ✓)
            #   ⚠ 破例性质 ✓: 改变了冻结命令的旋转部分 ⇒ **不作交付基线** ✗;
            #     默认 0 = 行为不变 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ md 06.4-e1 ✓: 执行循环的**二次姿态生成**同样停用 (构造器为唯一来源 ✓)
            if False and phase == "CARRY" and os.environ.get("_S2_CARRY_LOCK_ROT", "0") == "1":
                if getattr(self, "_carry_lock_rpy", None) is None:
                    self._carry_lock_rpy = (float(rz), float(ry), float(rx))
                    logger.error("[破例·诊断] CARRY 姿态锁定生效 (rz,ry,rx)=%s ⇒ 只平移不旋转 ✓ "
                                 "⚠ 本轮不作交付基线 ✗", self._carry_lock_rpy)
                rz, ry, rx = self._carry_lock_rpy
            for key, value in (
                ("vx", target_pos[0]),
                ("vy", target_pos[1]),
                ("vz", target_pos[2]),
                ("rz", rz),
                ("ry", ry),
                ("rx", rx),
            ):
                active_joints[virtual_idx[key]].set_drive_target(float(value))
            # 注 (2026-09-13, 用户定策): **禁止分相位修改夹爪参数** ——
            # "夹爪从始至终应该是一个模式的". 曾试过"接近段把手指 PD 刚度 ×3"以抵抗接触
            # 把实际开合拖低 (实测 act 最低 0.0030→0.0151, 3/3 PASS), 虽有效但违反上述约束,
            # 已撤销. 夹爪的刚度/阻尼/力限全程恒定, 只由 physics_utils 单一来源决定.
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ "压力优化"入口 (用户 2026-09-29 要求实现 ✓ + 破例授权 ✓) ★★★★
            #   机理 ✓: PD 位置控制下, 夹持法向力 = 刚度 × (命令 − 实际) ⇒
            #     命令端**额外过闭合** `_S2_GRIP_OVERCLOSE_M` ⇒ 位置误差增大 ⇒ 夹得更紧 ✓
            #   ⚠ 破例性质 ✓: 这**改变了冻结命令** ⇒ 结果**不得作为交付基线** ✗;
            #     若确认有效, 正确落点是重跑 Stage1 冻结新过闭合 (md §09 ✓)
            #   ⚠ 默认 0 = 行为不变 ✓
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★ 相位门控 ✓ (s104 实测): 过闭合**只作用于 CARRY 段** —— 在闭合段施加会
            #   破坏 `PRE_F50_READY`(闭合前 q 跟踪门 ✗, s104 实测挂 ✗)；
            #   而 s91(不加压) 能过 ready 并进到 CARRY ✓ ⇒ "先过门、搬运时再夹紧" ✓
            # ★ md 06.4-e1 ✓: "**不为本次诊断引入新的增压方案**" ⇒ 运行时过闭合**停用** ✓
            _oc_m = 0.0
            if False and _oc_m > 0 and str(phase) != "CARRY":
                _oc_m = 0.0
            if _oc_m > 0:
                _qL = max(0.0, float(q_cmd[0]) - _oc_m)
                _qR = max(0.0, float(q_cmd[1]) - _oc_m)
                if int(index) == int(meta.get("close_start", 0)):
                    logger.error("[破例·诊断] Stage2 过闭合偏置 %.1fmm 生效 (压力优化 ✓) "
                                 "⚠ 本轮不作交付基线 ✓", _oc_m * 1000.0)
            else:
                _qL, _qR = float(q_cmd[0]), float(q_cmd[1])
            active_joints[gi1].set_drive_target(_qL)
            active_joints[gi2].set_drive_target(_qR)
            physics_step(
                robot,
                arm_indices,
                gi1,
                gi2,
                np.array([]),
                float(q_cmd[0]),
                float(q_cmd[1]),
                self.scene,
            )

            actual_qpos = np.asarray(robot.get_qpos(), dtype=np.float64)
            actual_qvel = np.asarray(robot.get_qvel(), dtype=np.float64)
            actual_pos = np.asarray(
                [actual_qpos[virtual_idx[k]] for k in ("vx", "vy", "vz")], dtype=np.float64
            )
            T_W_G = self._gripper_pose_matrix(robot, side)
            T_W_O = np.asarray(target_actor.get_pose().to_transformation_matrix(), dtype=np.float64)
            f1_contact, f2_contact, contact_objects, contact_links = self._target_finger_contacts(
                robot, side, self.scene, self.obj_actors, target_name
            )
            target_in_contacts = any(
                name == target_name or (target_name and target_name in name) for name in contact_objects
            )
            target_contacts, environment_contacts = self._classify_contact_objects(
                contact_objects, target_name
            )
            audit = self._audit_contact_pairs(
                robot, side, self.scene, self.obj_actors, target_name
            )
            leak_contacts = audit["leak_contact"]
            ground_contacts = audit["ground_contact"]
            bilateral = bool(f1_contact and f2_contact and target_in_contacts)
            # ★★★★ 2026-09-18 Stage2 侧: "不要碰物体"的 **CoACD 几何量** (用户定调) ★★★★
            #   用户原话: "stage2 优化轨迹肯定要有代价 ... 我认为的奖励代价这种优化应该是很简单的,
            #     就是**不要碰物体**, 用 coacd 优化一下"
            #   为什么必须换通道 ✓: 现在判"入场即接触"用的是 `_target_finger_contacts` ✗
            #     ⇒ 依赖 SAPIEN 接触对 ✗ ⇒ 本项目已实测: CoACD 的 64 个重叠凸包下
            #       该信号**不可靠** ✗ (f1/f2 随闭合深度**非单调**: 16→5 个 / 20→6 个 / 24→0 个 ✗)
            #     ⇒ "碰不碰物体"**测不准** ✗ ⇒ 代价罚不到点上 ✗
            #   本量 ✓: 纯几何 —— 两个手指接触点 p1/p2 到物体 **CoACD 凸包顶点**的最近距离 ✓
            #     用"到顶点距离"而非"到凸包面距离" ✓ ⇒ 对凸包而言**只会高估距离** ✗
            #     ⇒ 偏保守 ✓ ⇒ 不会误报"碰"✗ ⇒ 作为"不要碰"的代价是**安全方向** ✓
            #   说明: 凸包只在首次读取 (几何不变 ✓), 不用每次重取 ✓
            #   注意: 本块只**量测+记录** ✓, 暂不改任何判据 ✓ (待看到分布再定阈值 ✓)
            try:
                # ★★★★ 2026-09-21 【问题6 修复】必须按**当前物体位姿**实时取凸包 ✗ ★★★★
                #   旧代码 ✗: `self._coacd_hulls_w` 只在**第一次**取, 之后物体移动/旋转
                #     甚至下一轮 rollout 都**不更新** ✗ ⇒ 距离诊断量的是**旧位置**凸包 ✗
                #     ("距离一直增大"可能只是测量对象没跟着物体 ✓)
                #   机制 ✓: 凸包**本体系**顶点缓存不变 ✓, 每帧用当前 `T_W_O` 变换 ✓
                #   兼容 ✓: `_S2_HULLS_LIVE=0` ⇒ 回到旧的世界系一次性缓存 ✗ (复现旧行为 ✓)
                if os.environ.get("_S2_HULLS_LIVE", "1") == "1":
                    _hw = self._coacd_world_hulls_live(target_name) or []
                else:
                    if getattr(self, "_coacd_hulls_w", None) is None:
                        self._coacd_hulls_w = self._coacd_world_hulls_stage2(target_name)
                    _hw = self._coacd_hulls_w or []
                if _hw:
                    _Tg = self._gripper_pose_matrix(robot, side)
                    if _Tg is not None and np.isfinite(_Tg).all():
                        _cd = np.asarray(_Tg[:3, :3], dtype=np.float64) @ np.array([0.0, 1.0, 0.0])
                        try:
                            from physics_env import (  # noqa: PLC0415
                                FINGER_EFFECTIVE_HALF_SPACING as _FEHS,
                            )
                        except Exception:  # noqa: BLE001
                            _FEHS = 0.0152
                        _hs = float(_FEHS) + float(
                            np.mean(np.asarray(actual_qpos[[gi1, gi2]], dtype=np.float64)))
                        _po = np.asarray(_Tg[:3, 3], dtype=np.float64)
                        _p1 = _po + _hs * _cd
                        _p2 = _po - _hs * _cd
                        _allv = np.concatenate(_hw, axis=0)
                        _d1 = float(np.min(np.linalg.norm(_allv - _p1, axis=1)))
                        _d2 = float(np.min(np.linalg.norm(_allv - _p2, axis=1)))
                        _min_d = min(_d1, _d2)
                        _ap_min = getattr(self, "_coacd_approach_min_mm", float("inf"))
                        if _min_d * 1000.0 < _ap_min:
                            self._coacd_approach_min_mm = _min_d * 1000.0
                        # 2026-09-18 扩到**全相位** ✓ (实测接近段恒 >=93mm ✓ ⇒ 真正要看的是
                        #   闭合/静置/携带段 ✗ —— 那里才是物体滑脱/被弹发生的位置 ✓)
                        #   为控制日志量: 每 3 个控制点记一次 ✓
                        if index % 3 == 0:
                            logger.info(
                                "[CoACD接触量] idx=%d %s | 指1距凸包=%.2fmm 指2=%.2fmm "
                                "| 全程最小=%.2fmm | half_sep=%.1fmm | 手指act均值=%.4f",
                                index, str(phase)[:14], _d1 * 1000.0, _d2 * 1000.0,
                                float(self._coacd_approach_min_mm), _hs * 1000.0,
                                float(np.mean(np.asarray(
                                    actual_qpos[[gi1, gi2]], dtype=np.float64))),
                            )
                        trace.setdefault("coacd_finger_hull_gap_mm", []).append(_min_d * 1000.0)
            except Exception as _ecq:  # noqa: BLE001
                logger.warning("[CoACD接触量] 采集失败: %s", _ecq)
            # ★★★★ 2026-09-17 几何双侧兜底 (用户: "continue") ★★★★
            #   现象 (已实测, output/objd): CoACD 线闭合段**物体全程 objΔ=[0.0,0.0,0.0]mm** ✓
            #     (纹丝不动 ⇒ 被稳稳夹住 ✓ / 或一直被桌面摩擦固定 ✓), 手指 act 收到 0.0079 停住 ✓,
            #     但 `f1/f2` 报 False ✗ ⇒ `target_bilateral` 恒 0 ✗ ⇒ PROVISIONAL_GRASP 必挂 ✗
            #   根因: `_target_finger_contacts`(1304) 的 `object_bodies` 只登记 actor 的**第一个**
            #     PhysxRigidDynamicComponent ✗ (line 1331 有 `break` ✗); 而 CoACD 是**多凸包复合碰撞体**
            #     ⇒ 接触落在别的组件上 ⇒ 名字匹配不上 ⇒ is_target=False ✗
            #     同文件的 `_contact_detail_pairs`(1368) 因为**现取** body.name ✓ 就能读到
            #     ("link1 × glb_4 = 0.1268mm" ✓) ⇒ 两个函数读同一个 `scene.get_contacts()`,
            #     差别只在**名字解析** ✓ ⇒ 排除"物理没夹住" ✗
            #   旁证: SAPIEN 警告 "loading multiple convex collision meshes from STL file is
            #     unsupported and can result in **invalid collision meshes**" ✗
            #     ⇒ 多凸包碰撞体的**接触对查询不可靠** ✗ ⇒ 该判据在 CoACD 下测不到真值 ✗
            #   修法 (与上一条建议一致): 换一条**与 SAPIEN 无关**的几何通道判"双侧夹住" ✓
            #     指缝 = FINGER_GAP0 + 2·act ✓ (physics_env:264 同式 ✓), 物体 y 宽 = bbox_size[1] ✓
            #     ⇒ 双侧夹住 ⟺ |指缝 − 物体宽| ≤ 4mm ✓
            #   开关: `_BILATERAL_GEO_FALLBACK=1` 才启用 ✓ ⇒ **默认行为完全不变** ✓ (对 box 线零影响 ✓)
            if not bilateral and os.environ.get("_BILATERAL_GEO_FALLBACK", "0") == "1":
                try:
                    try:
                        from physics_env import FINGER_GAP0 as _FG0_i  # noqa: PLC0415
                    except Exception:  # noqa: BLE001
                        _FG0_i = 0.0304       # 与 physics_env 的定义一致 ✓
                    _gap_geo = float(_FG0_i) + 2.0 * float(
                        np.mean(np.asarray(actual_qpos[[gi1, gi2]], dtype=np.float64))
                    )
                    _w_geo = float(
                        self.obj_info.get(target_name, {}).get("bbox_size", [0.0, 0.0, 0.0])[1]
                    )
                    # ★ 2026-09-21 【用户原则】物理是 CoACD ⇒ **不许用 bbox 宽度**判"双侧夹住" ✗
                    #   "你物理仿真用什么, 你的规划就用什么" ✓ ⇒ bbox 宽度与凸包真实跨度不同 ✗
                    #   本处 ✓: `collision_mode=coacd` ⇒ **拒绝该 fallback** (保持 bilateral=False ✓),
                    #     避免用错几何把"没夹住"判成"夹住" ✗
                    _cm_mode_g = ((getattr(self, "obj_info", {}) or {}).get(
                        target_name, {}) or {}).get("collision_mode")
                    if _cm_mode_g == "coacd":
                        logger.warning(
                            "[双侧几何 fallback] ⚠ 目标物**物理=CoACD**, 本判据却用 **bbox 宽度** "
                            "⇒ 违反'规划=物理'原则 ✗ ⇒ **拒绝该 fallback** (bilateral 保持 False ✓); "
                            "如需真实跨度请用凸包投影 ✓")
                        _w_geo = 0.0
                    bilateral = bool(_w_geo > 0.0 and abs(_gap_geo - _w_geo) <= 0.004)
                    # 2026-09-18 诊断: 上次本兜底"没生效" ✗ ⇒ 打出两个输入与判定 ✓
                    logger.warning(
                        "[几何双侧兜底] target=%s | 指缝=%.2fmm 物体y宽=%.2fmm | 差=%.2fmm "
                        "(容差 %.2fmm) ⇒ 双侧=%s | obj_info 键=%s",
                        target_name, _gap_geo * 1000.0, _w_geo * 1000.0,
                        abs(_gap_geo - _w_geo) * 1000.0, 4.0, bilateral,
                        list(getattr(self, "obj_info", {}).keys())[:6],
                    )
                except Exception as _eg:  # noqa: BLE001
                    logger.warning("[几何双侧兜底] 采集失败: %s", _eg)
            T_G_O = _relative_pose(T_W_G, T_W_O) if T_W_G is not None else None

            # md 06.4-e2 ✓ 采样点②③④: **相位入口**各落盘一次审计
            #   (close 入口 / settle 入口 / 首个 CARRY 命令 ⇒ 可直接与失联时序对照 ✓)
            try:
                _seen = getattr(self, "_aud_seen_phases", None)
                if _seen is None:
                    _seen = set()
                    self._aud_seen_phases = _seen
                _ph_s = str(phase)
                if _ph_s in ("STAGE1_CLOSE", "STAGE1_SETTLE", "CARRY") and _ph_s not in _seen:
                    _seen.add(_ph_s)
                    self._dump_collision_audit("02_entry_" + _ph_s, side=side,
                                               target_name=target_name, robot=robot)
            except Exception as _eau:  # noqa: BLE001
                logger.warning("[e2] 相位审计落盘异常(忽略): %s", _eau)
            # ★ md 06.4-f2 (2026-09-30) ✓: **逐指法向/切向载荷**接入 Stage2 trace ✓
            #   实测缺口 ✗: Stage2 主循环**从未调用** `get_grasp_force_detail` ⇒
            #     `grasp_force_n` 实际是 **Stage1 的旧总冲量口径** ✗ ——
            #     而 R13 教训: "总冲量模长 ≠ 两指目标法向力" ✗ ⇒ 判不了"是否承载" ✗
            #   本改 ✓: 每控制点读一次(只读 contacts ✓ 不额外 step) ⇒ 记
            #     `n_f1/n_f2`(法向, 判承载 ✓) + `t_f1/t_f2`(切向, 判滑移趋势 ✓)
            #     + `palm_n`/`ground_n`/`other_n` 分类 ⇒ 掌部/桌面**不得**计入双指载荷 ✓
            #   ⚠ f2: 缺有效载荷**先修观测**, 不改摩擦 ✓; 读不到 ⇒ MISSING 显式标注 ✗
            try:
                from physics_env import get_grasp_force_detail as _gf2
                _ld2 = _gf2(side, self.scene, self.obj_actors, robot, target_name)
            except Exception as _e2f:  # noqa: BLE001
                # ⚠ 必须留下**真实原因**: 首版只置 {} ⇒ 全部变 nan, 事后无法区分
                #   "真的无力" 与 "读不到" ✗ (f2: 缺有效载荷先修观测 ✓)
                _ld2 = {"_error": "%s: %s" % (type(_e2f).__name__, _e2f)}
            for _k2, _v2 in (("n_f1", "n_target_f1"), ("n_f2", "n_target_f2"),
                             ("t_f1", "t_target_f1"), ("t_f2", "t_target_f2"),
                             ("palm_n", "f_palm"), ("ground_n", "f_ground"),
                             ("other_n", "f_other")):
                try:
                    trace.setdefault(_k2, []).append(float(_ld2.get(_v2, float("nan"))))
                except Exception:  # noqa: BLE001
                    trace.setdefault(_k2, []).append(float("nan"))
            # ★★ md 06.4-g3 (2026-10-01) ✅【**同帧**指面数据 ⇒ 夹持判定不再靠 AABB】
            #   记录: 双指**实际世界位姿**、闭合轴(两指连线)、以及该控制点的目标 T_W_O(已有) ✓
            #   ⇒ first_failure 可用**同一 index** 的数据, 在**同一闭合轴**上做投影比较 ✓
            #     (g3 明令: "投影取**同一闭合轴**"、"整体跨度只做包络"、"无相位快照不可用
            #      最终 live hull 诊断历史点" ✗ ⇒ 必须逐帧记录 ✓)
            try:
                _fp = []
                for _ln in (f"{side}_gripper_finger_link1", f"{side}_gripper_finger_link2"):
                    _pl = None
                    for _lk2 in robot.get_links():
                        if str(_lk2.get_name()) == _ln:
                            _pl = np.asarray(_lk2.get_pose().p, dtype=np.float64)
                            break
                    _fp.append(_pl)
                trace.setdefault("finger1_pos", []).append(
                    None if _fp[0] is None else [round(float(x), 6) for x in _fp[0]])
                trace.setdefault("finger2_pos", []).append(
                    None if _fp[1] is None else [round(float(x), 6) for x in _fp[1]])
                if _fp[0] is not None and _fp[1] is not None:
                    _ax = _fp[1] - _fp[0]
                    _nrm = float(np.linalg.norm(_ax))
                    _ca = ([round(float(x), 6) for x in (_ax / _nrm)]
                           if _nrm > 1e-9 else None)
                else:
                    _ca = None
                trace.setdefault("close_axis", []).append(_ca)
            except Exception as _egf:  # noqa: BLE001
                trace.setdefault("finger1_pos", []).append(None)
                trace.setdefault("finger2_pos", []).append(None)
                trace.setdefault("close_axis", []).append(None)
            _lp_state = _ld2.get("normal_probe_state")
            if _lp_state is None:
                _lp_state = "MISSING: %s" % _ld2.get("_error", "unknown(无异常但缺字段 ✗)")
                logger.warning("[f2-probe] 载荷探针不可用 ⇒ %s", _lp_state)
            trace.setdefault("load_probe_state", []).append(str(_lp_state))
            trace["control_index"].append(index)
            trace["source_frame"].append(float(bundle["source_frame"][index]))
            trace["phase"].append(phase)
            trace["base_target"].append(target_pos.copy())
            trace["base_target_R"].append(np.asarray(target_R, dtype=np.float64).copy())
            trace["base_cmd_euler"].append(np.array([rz, ry, rx], dtype=np.float64))
            trace["base_actual"].append(actual_pos)
            trace["gripper_q_cmd"].append(q_cmd.copy())
            trace["gripper_q_actual"].append(actual_qpos[[gi1, gi2]].copy())
            trace["object_T"].append(T_W_O)
            trace["gripper_T"].append(T_W_G if T_W_G is not None else np.full((4, 4), np.nan))
            trace["T_G_O"].append(T_G_O if T_G_O is not None else np.full((4, 4), np.nan))
            trace["finger1_contact"].append(bool(f1_contact and target_in_contacts))
            trace["finger2_contact"].append(bool(f2_contact and target_in_contacts))
            trace["target_bilateral"].append(bilateral)
            trace["contact_objects"].append(tuple(sorted(contact_objects)))
            trace.setdefault("contact_links", []).append(tuple(contact_links))
            trace["target_contact"].append(bool(target_contacts))
            trace["ground_contact"].append(bool(ground_contacts))
            trace["non_target_contact"].append(bool(leak_contacts))
            # Backwards-compatible: environment_contact now means the leak pair.
            trace["environment_contact"].append(bool(leak_contacts))
            trace["grasp_force"].append(float(get_grasp_force(side, self.scene, self.obj_actors, robot)))

            if phase == "MANO_FOLLOW":
                prefix_target_contacts += int(bool(target_contacts))
                if target_contacts:
                    # 记录具体控制编号: 若出现在 F0/F1, 说明检测或物体位姿有问题;
                    # 若集中在 f_start 前后, 则与手指碰撞盒长度一致, 是真接触.
                    prefix_contact_indices.append(int(index))
                prefix_leak_contacts += int(bool(leak_contacts))
            elif phase in ("APPROACH_ADJUST", "FINAL_APPROACH"):
                approach_target_contacts += int(bool(target_contacts))
                approach_leak_contacts += int(bool(leak_contacts))
            target_contact_names.update(target_contacts)
            target_contact_link_names.update(contact_links)
            if target_contacts or leak_contacts:
                for _ln, _on, _dp in self._contact_detail_pairs(robot, side, self.scene):
                    _rec = contact_detail.setdefault(
                        (_ln, _on), {"n": 0, "max_depth_mm": 0.0, "first_ctrl": int(index)}
                    )
                    _rec["n"] += 1
                    _rec["max_depth_mm"] = max(_rec["max_depth_mm"], float(_dp) * 1000.0)
            leak_contact_names.update(audit["leak_names"])
            ground_contact_names.update(audit["ground_names"])
            # ★ 2026-09-21 【P0】按**对**收集"夹爪↔环境"接触 ✓ (旧代码只收环境名 ⇒ 门失效 ✗)
            gripper_ground_pairs_all.update(audit.get("gripper_ground_pairs", []) or [])

            if phase == "FINAL_APPROACH":
                pos_error = float(np.linalg.norm(actual_pos - target_pos))
                rot_error = (
                    _rotation_distance(T_W_G[:3, :3], target_R) if T_W_G is not None else float("inf")
                )
                linear_speed = float(
                    np.linalg.norm([actual_qvel[virtual_idx[k]] for k in ("vx", "vy", "vz")])
                )
                angular_speed = float(
                    np.linalg.norm([actual_qvel[virtual_idx[k]] for k in ("rz", "ry", "rx")])
                )
                object_drift = float(np.linalg.norm(T_W_O[:3, 3] - initial_object_T[:3, 3]))
                # ★★★★ 2026-09-21 【问题8a 修复】本相位必须检查『夹爪↔桌面』接触 ✗ ★★★★
                #   静态审查结论 ✓: "`FINAL_APPROACH` 的 `passed` **不检查 `ground_contacts`**;
                #     携带后的 ground contact 也主要**记录而不失败**" ✗
                #   为什么重要 ✓ (本项目实测): 指尖/手指**贴到桌面** ⇒ 闭合被桌面挡停 ⇒
                #     手指离物体还差十几毫米就停下 ⇒ **根本没压住物体** ✗
                #     (证据: `[Stage2 真夹持判据] 挡停 +3.38/+3.39mm 但 最近侧=6.9mm` +
                #      `contacts` 里同时出现 `('left_gripper_finger_link1','ground')` ✓)
                #   本检查 ✓: 把『夹爪/手指 ↔ ground』接触**显式报错并默认判失败** ✓
                #     开关 ✓: `_S2_FAIL_ON_GRIPPER_GROUND` (默认 **1** = 失败 ✗; 0 = 只告警 ✓)
                # ★ 2026-09-21 【P0 修复】旧写法从 `ground_contact_names`(只含**环境名** ✗)
                #   里搜 `gripper/finger` ⇒ **永远匹配不到** ✗ (审查结论 ✓) ⇒ 门失效 ✗
                #   新写法 ✓: 直接用审计返回的 **按对**结果 `gripper_ground_pairs` ✓
                _gg = sorted(gripper_ground_pairs_all)
                if _gg:
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ 2026-09-21 【修正】碰撞判定必须用**物理真实量(穿透深度)**, 不是"是否接触" ✗
                    #   用户原则 ✓: "你物理仿真用什么, 你的规划就用什么 … 不要老是渲染不对、
                    #     物理仿真不对, 这个需要解决" ✓ ⇒ 判碰撞也要用**物理真实几何/真实量** ✓
                    #   病根 ✗ (旧写法): 只要 `('gripper_finger_link','ground')` 出现在接触对里
                    #     就判失败 ✗ ⇒ **"接触" ≠ "碰撞"** ✗
                    #   为什么本场景"接触"是**几何必然** ✓:
                    #     · 目标是 **20mm 矮物体**, 要夹它的**侧面** ⇒ 指尖必须降到物体中部以下 ✓
                    #       ⇒ 指尖落到**桌面高度**是必然的 ✓ (实测 `指尖离桌面=0.0mm` ✓)
                    #     ⇒ 用"接触"当失败 ⇒ 会**永远卡死** ✗ (geom3 实测: FINAL_APPROACH ctrl=49
                    #       `[('left_gripper_finger_link1','ground'), (...,'link2','ground')]` ✗)
                    #   本改 ✓: 用引擎给出的**真实穿透深度** (`_contact_detail_pairs` 的 depth ✓)
                    #     · 深度 > `_S2_GROUND_PEN_TOL_MM` (默认 **0.5mm** = 一个物理步的正常压入 ✓)
                    #       ⇒ **真穿透** ⇒ 判失败 ✗ (这才是"撞桌" ✓)
                    #     · 深度 ≤ tol ⇒ 只**告警** ✓ (贴住但未穿透 ⇒ 放行 ✓)
                    #   注 ✓: "闭合被桌面挡停 ⇒ 夹不紧"这件事**已有正确仪器** ✓ ——
                    #     `[Stage2 真夹持判据]` 的 **最近侧 / 真夹持率** ✓ (见 md §46 ✓)
                    #     ⇒ 不需要用"接触"这种粗糙代理 ✗
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    _gg_depth_mm = 0.0
                    try:
                        for _ln2, _on2, _dp2 in self._contact_detail_pairs(robot, side, self.scene):
                            _s_ln, _s_on = str(_ln2).lower(), str(_on2).lower()
                            if "ground" in _s_on and ("gripper" in _s_ln or "finger" in _s_ln):
                                _gg_depth_mm = max(_gg_depth_mm, float(_dp2) * 1000.0)
                            elif "ground" in _s_ln and ("gripper" in _s_on or "finger" in _s_on):
                                _gg_depth_mm = max(_gg_depth_mm, float(_dp2) * 1000.0)
                    except Exception as _eg2:  # noqa: BLE001
                        logger.warning("[Stage2] 地面穿透深度读取失败: %s", _eg2)
                    _pen_tol_mm = float(os.environ.get("_S2_GROUND_PEN_TOL_MM", "0.5") or 0.5)
                    if _gg_depth_mm > _pen_tol_mm:
                        # ★ 2026-09-22 【P1-3 联动】真穿透时**同时**给出逐子步分级 ✓
                        #   为什么 ✓: 实测 `p03_snapshot` 里 Stage2 在 `FINAL_APPROACH`
                        #     就被拦下 ✗ ⇒ 原本放在 settle 的 P1-3 汇总**根本没机会打印** ✗
                        #     ⇒ 在**失败点**打印, 才看得出"是轻触、承载挡停、还是真穿透" ✓
                        try:
                            from physics_env import get_substep_ground_report as _gsgr2
                            _gr2 = _gsgr2()
                            logger.error(
                                "[P1-3] FINAL_APPROACH 逐子步汇总 (失败点 ✓): 子步=%d, "
                                "有接触子步=%d, **最小 separation=%s mm**, 最大冲量=%.5f ⇒ "
                                "**分级=%s** | 分 link=%s",
                                int(_gr2.get("substeps", 0)),
                                int(_gr2.get("n_contact_substeps", 0)),
                                ("n/a" if _gr2.get("min_sep_mm") is None
                                 else f"{float(_gr2['min_sep_mm']):+.3f}"),
                                float(_gr2.get("max_impulse", 0.0)),
                                _gr2.get("classification"), _gr2.get("by_link"))
                        except Exception as _eg3:  # noqa: BLE001
                            logger.warning("[P1-3] 失败点汇总读取失败: %s", _eg3)
                        logger.error(
                            "[Stage2] ★ **夹爪/手指穿透桌面** (phase=%s ctrl=%d) ✗: %s | "
                            "**穿透=%.2fmm** (> %.2fmm) ⇒ 真碰撞 ✗",
                            phase, int(index), _gg, _gg_depth_mm, _pen_tol_mm)
                        if os.environ.get("_S2_FAIL_ON_GRIPPER_GROUND", "1") == "1":
                            fail("FINAL_APPROACH",
                                 f"gripper-ground penetration={_gg_depth_mm:.2f}mm>"
                                 f"{_pen_tol_mm:.2f}mm {_gg}")
                            if _stop:
                                break
                    else:
                        logger.warning(
                            "[Stage2] 夹爪/手指**贴住**桌面但**无穿透** (%.2fmm ≤ %.2fmm, %s) ⇒ "
                            "**不算碰撞** ✓ 放行 ✓ (夹 20mm 矮物体时指尖到桌面高度是几何必然 ✓; "
                            "'夹不紧'由 `[Stage2 真夹持判据]` 的**最近侧/真夹持率**判定 ✓)",
                            _gg_depth_mm, _pen_tol_mm, _gg)
                # ★★★★ 2026-09-21 【诊断】接近段逐控制点速度曲线 (`_S2_READY_TRACE=1` ✓) ★★★★
                #   为什么需要 ✓: `PRE_F50_READY` 的就位样本**只取自 `FINAL_APPROACH` 相位**
                #     (本块 ✓), 而 `STAGE1_PHASE_SETTLE` 的 10 个 hold 控制点**不在采样窗内** ✗
                #     ⇒ "加长 hold" 对该门**无效** ✗
                #   现状 ✗: 门禁唯一超限项 = **角速度** (实测 19.5°/s, 阈值 5.0°/s ✓ 见 §19.2)
                #     位置/姿态都已到位 (0.46mm / 0.54°) ⇒ 只是"**接近段末尾还在转**" ✗
                #   本曲线 ✓: 打印该相位**每个控制点**的 pos/rot/线速/角速 ⇒ 一眼看出:
                #     · 是否在收敛 (单调下降) ⇒ 延长该相位即可 ✓
                #     · 是否卡在某个非零值 (不降) ⇒ 目标姿态本身在动 ✗ (要钉住 anchor pose ✓)
                if os.environ.get("_S2_READY_TRACE", "0") == "1":
                    logger.info(
                        "[Stage2 就位跟踪] ctrl=%d pos=%.2fmm rot=%.3f° **lin=%.1fmm/s "
                        "ang=%.1f°/s**",
                        int(index), pos_error * 1000.0, np.rad2deg(rot_error),
                        linear_speed * 1000.0, np.rad2deg(angular_speed))
                # ★★★★ 2026-09-17 drift 分解 (用户: "继续") ★★★★
                #   实测三联 (轴向余量 6mm / 12mm / 不压平底面): drift **恒为 20.68mm** ✗
                #     ⇒ 与一切几何/接近配置无关 ✗ ⇒ 不是"被手指推走" ✗, 是**常数项** ✗
                #   代码事实: initial_object_T = reset_object_T (stage2_reconstruct_mixin:1921)
                #     ⇒ 参考是 **reset 瞬间** 的位姿 ✗ (物体尚未在桌面坐实 ✗)
                #     ⇒ "落到桌面坐实"的位移被当成漂移 ✗
                #   量级吻合 ✓: 物体 bbox 高 39.29mm ⇒ 半高 **19.63mm** ≈ 20.68mm ✓
                #     (box 版碰撞体=规则盒 ⇒ 与名义位姿一致 ⇒ 该项≈0 ✓ ⇒ 门禁通过 ✓;
                #      CoACD 凸包支撑面 ≠ 名义位姿 ✗ ⇒ 多出 ~20mm ✗)
                #   本打印: 把 drift 拆成 Δx/Δy/Δz, 一眼看出是"横向被推" ✗ 还是"z 落定" ✓
                try:
                    _d_now = np.asarray(T_W_O[:3, 3], dtype=np.float64)
                    _d_ref = np.asarray(initial_object_T[:3, 3], dtype=np.float64)
                    logger.warning(
                        "[drift 分解] idx=%d | 当前 z=%.4f 参考 z=%.4f | "
                        "Δx=%+.2fmm Δy=%+.2fmm Δz=%+.2fmm | 横向=%.3fmm 垂向=%.3fmm | "
                        "总=%.2fmm (阈值 %.1fmm) | 物体 bbox 半高=%.2fmm",
                        index, float(_d_now[2]), float(_d_ref[2]),
                        (_d_now[0] - _d_ref[0]) * 1000.0,
                        (_d_now[1] - _d_ref[1]) * 1000.0,
                        (_d_now[2] - _d_ref[2]) * 1000.0,
                        float(np.linalg.norm(_d_now[:2] - _d_ref[:2]) * 1000.0),
                        float(abs(_d_now[2] - _d_ref[2]) * 1000.0),
                        object_drift * 1000.0, PRE_GRASP_OBJECT_DRIFT_M * 1000.0,
                        float(self._target_bbox_size(target_actor)[2] * 1000.0 / 2.0)
                        if hasattr(self, "_target_bbox_size") else float("nan"),
                    )
                except Exception as _ed:  # noqa: BLE001
                    logger.warning("[drift 分解] 采集失败: %s", _ed)
                ready_samples.append(
                    {
                        # ★ 2026-09-13 (用户定策一致性修正):
                        #   `APPROACH_TARGET_CONTACT` 早已被明确降级为"**软指标·不阻断**"
                        #   (轻擦 ≤2mm 不否定抓取, 见 stage2_reconstruct_mixin.py 该门注释),
                        #   但就位样本这里仍把 `not target_contacts` 当**硬条件** → 自相矛盾:
                        #   scale=0.7 实测所有数值都优秀 (pos 0.18mm / rot 0.026° / 线速 10.9mm/s /
                        #   角速 0.55°/s / 漂移 2.54mm), **唯一**失败项就是接近段末端的轻擦接触。
                        #   现与既定策略对齐: 接触只作为**指标记录**, 不再硬性否决。
                        #   真正的硬碰撞仍由 `PRE_F50_READY` 的 q 误差门 (0.003) 与
                        #   终局 `CARRY`/`TRACK` 门兜住。
                        "passed": bool(
                            pos_error <= PRE_READY_POS_TOL_M
                            and rot_error <= PRE_READY_ROT_TOL_RAD
                            and linear_speed <= PRE_READY_LINEAR_SPEED_MPS
                            and angular_speed <= PRE_READY_ANGULAR_SPEED_RADPS
                            and not leak_contacts
                            and object_drift <= PRE_GRASP_OBJECT_DRIFT_M
                        ),
                        "pos_mm": pos_error * 1000.0,
                        "rot_deg": np.rad2deg(rot_error),
                        "linear_mm_s": linear_speed * 1000.0,
                        "angular_deg_s": np.rad2deg(angular_speed),
                        "target_contact": bool(target_contacts),
                        "ground_contact": bool(ground_contacts),
                        "non_target_contact": bool(leak_contacts),
                        "object_drift_mm": object_drift * 1000.0,
                    }
                )

            if index == meta["hold_end"]:
                settle_n = int(meta["settle_controls"])
                settle_bilateral = trace["target_bilateral"][
                    meta["settle_start"] : meta["settle_end"] + 1
                ]
                object_drift = float(np.linalg.norm(T_W_O[:3, 3] - initial_object_T[:3, 3]))
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-23 【用户指出的大问题】管线**从不检查物体位置/速度** ✗ ★★★★
                #   用户原话 ✓: "**都没夹住, 怎么就突然中间的物体不见了** … 你好像**都没有关注到
                #     中间物体的位置**, 很奇怪的一个仿真操作" ✗
                #   实测铁证 ✓: `trimesh/s2gif/stage3/grasp.log` 的 `obj_z`
                #     最小 **0.0168m** (正常贴桌 ✓) / 最大 **9.3598 / 9.3639 / 9.4978m** ✗✗
                #     ⇒ 物体被**弹飞 9.4 米** ⇒ 飞出相机视野 ⇒ "看起来凭空消失" ✓
                #   本改 ✓: 检查**物体位置/速度合理性** ⇒ 越界立刻 **ERROR 报出** ✓
                #     (`_S2_FAIL_ON_OBJ_SANE=1` 时直接 fail ✗; 默认只报 ERROR ⇒ 不改既有门 ✓)
                #   ⚠ 这是**仪器**不是"放宽判据" ✓ —— 目的是让"物体飞了"**再也藏不住** ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                try:
                    _op = np.asarray(T_W_O, dtype=np.float64)[:3, 3]
                    _ov2 = None
                    # ⚠ 2026-09-23 修正 ✗: 该作用域**没有** `tactor` ✗
                    #   (实测: `[Stage2 物体合理性] 检查失败: name 'tactor' is not defined` ✗)
                    #   ⇒ 改为按**目标名**在 `self.obj_actors` 里查 ✓
                    _tactor_ok = None
                    try:
                        for _a5 in self.obj_actors:
                            if target_name and (str(target_name) in _a5.get_name()
                                                or _a5.get_name() in str(target_name)):
                                _tactor_ok = _a5
                                break
                    except Exception:  # noqa: BLE001
                        _tactor_ok = None
                    if _tactor_ok is not None:
                        for _c5 in _tactor_ok.components:
                            if isinstance(_c5, sapien.pysapien.physx.PhysxRigidDynamicComponent):
                                try:
                                    _ov2 = float(np.linalg.norm(np.asarray(
                                        _c5.get_linear_velocity(), dtype=np.float64)))
                                except Exception:  # noqa: BLE001
                                    _ov2 = None
                                break
                    _z_lim = float(os.environ.get("_S2_OBJ_Z_LIMIT_M", "0.60") or 0.60)
                    _v_lim = float(os.environ.get("_S2_OBJ_V_LIMIT_MS", "5.0") or 5.0)
                    if (abs(float(_op[2])) > _z_lim
                            or (_ov2 is not None and _ov2 > _v_lim)):
                        logger.error(
                            "[Stage2 物体合理性] ✗✗ phase=%s ctrl=%d: 物体位置=%s |v|=%.2f m/s "
                            "(限 z≤%.2fm / v≤%.1fm/s) ⇒ **物体已被弹飞/失控** ✗ "
                            "(这就是'中间物体突然不见'的原因 ✓ 用户指出 ✓)",
                            phase, int(index), np.round(_op, 4).tolist(),
                            (-1.0 if _ov2 is None else _ov2), _z_lim, _v_lim)
                        if os.environ.get("_S2_FAIL_ON_OBJ_SANE", "0") == "1":
                            fail("OBJECT_INSANE",
                                 f"ctrl={int(index)} obj_pos={np.round(_op, 4).tolist()} "
                                 f"|v|={-1.0 if _ov2 is None else _ov2:.2f}")
                            if _stop:
                                break
                except Exception as _eos:  # noqa: BLE001
                    logger.warning("[Stage2 物体合理性] 检查失败: %s", _eos)
                # ★★★★ 2026-09-21 【B. 真夹持门 + 真值落点探针】(设计 §42.1 G4 ✓) ★★★★
                #   用户: "stage1 已经成功了, stage2 只需要稍微优化达到 stage1 的目标就可以抓取移动"
                #   实测瓶颈 ✓ (output/s2_liftoff): `PRE_F50_READY` / `PROVISIONAL_GRASP` 都已通过 ✓,
                #     但 `CARRY: T_G_O drift=**177~192mm**` ✗ 且
                #     `[CoACD接触量] CARRY | 指1距凸包=121→164mm | **手指act均值=0.0030**`
                #     ⇒ 手指**几乎到达指令位、没被挡停** ✗
                #   对比 Stage1 好解 ✓: 挡停 **+10.06 / +5.72mm** ⇒ 那次才是"真夹牢" ✓
                #   ⇒ 把"**两指是否真被物体挡停**"做成判据 ✓, 同时打出**真值落点**
                #     (两指各自到物体凸包的距离差 + 最近侧 + 指尖离桌面) ⇒ 一次定位根因:
                #       · 最近侧距离大 (>15mm) ⇒ **物体不在两指之间** (接近段把它蹭走了 ✗)
                #       · 最近侧 ~2mm 但没挡停    ⇒ 手指/桌面几何问题 (指尖贴桌面被挡 ✗)
                #   判据 ✓ (与 Stage1 同口径 ✓): 挡停量 = `act − cmd` (settle 段),
                #     "两指同时被挡停"的帧占比 ≥ `_S2_TRUE_GRASP_MIN`
                #     (默认 **0 = 只诊断** ✓; 根因修好后开成硬门 ⇒ 设计 §42.1 ✓)
                try:
                    _sst, _sen = int(meta["settle_start"]), int(meta["settle_end"])
                    _qa_s = np.asarray(trace["gripper_q_actual"], dtype=np.float64)[_sst:_sen + 1]
                    _qc_s = np.asarray(trace["gripper_q_cmd"], dtype=np.float64)[_sst:_sen + 1]
                    _blk = (_qa_s - _qc_s) * 1000.0                      # (n,2) mm ✓
                    _blk_med = np.median(_blk, axis=0) if _blk.size else np.zeros(2)
                    _both_blk = ((_blk > 0.3).all(axis=1) if _blk.size
                                 else np.zeros(0, dtype=bool))
                    _tgr = float(_both_blk.mean()) if _blk.size else 0.0
                    _sm_asym, _sm_dmin = (None, None)
                    try:
                        _sm_asym, _sm_dmin = self._measure_grasp_symmetry(target_name, robot)
                    except Exception:  # noqa: BLE001
                        pass
                    _gc_mm = None
                    try:
                        _gc_mm = self._measure_finger_ground_clearance(robot)
                    except Exception:  # noqa: BLE001
                        pass
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ 2026-09-22 【关键取证】"两指挡停"到底被**谁**挡停? ✗ ★★★★
                    #   复审原话 ✓: "**两指挡停不等于被目标挡住**" ✓
                    #   ⇒ 必须按**名字**归因 ✓ (接触对来自引擎 `_contact_detail_pairs` ✓)
                    #   判读 ✓:
                    #     · 指1↔目标 **且** 指2↔目标 ⇒ **真被物体挡停** ✓
                    #     · 任一指↔ground        ⇒ **被桌面挡停** ✗ (假夹持 ✗)
                    #     · 无接触对             ⇒ 挡停来自 **PD/摩擦** 而非碰撞 ✗
                    #       (这种情况"真夹持率 100%"是**误导** ✗ —— 手指只是没走到位 ✗)
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    _cp_f1, _cp_f2 = [], []
                    try:
                        for _ln2, _on2, _dp2 in self._contact_detail_pairs(robot, side, self.scene):
                            _s_ln = str(_ln2)
                            if "finger_link1" in _s_ln:
                                _cp_f1.append(f"{_on2}({float(_dp2) * 1000:+.2f}mm)")
                            elif "finger_link2" in _s_ln:
                                _cp_f2.append(f"{_on2}({float(_dp2) * 1000:+.2f}mm)")
                    except Exception as _ecp2:  # noqa: BLE001
                        logger.warning("[真夹持归因] 接触对读取失败: %s", _ecp2)

                    def _hit_target(_lst):
                        for _t in _lst:
                            _nm_t = str(_t).split("(")[0]
                            if target_name and (str(target_name) in _nm_t or _nm_t in str(target_name)):
                                return True
                        return False

                    _tgt_f1, _tgt_f2 = _hit_target(_cp_f1), _hit_target(_cp_f2)
                    _gnd_f1 = any("ground" in str(_t).lower() for _t in _cp_f1)
                    _gnd_f2 = any("ground" in str(_t).lower() for _t in _cp_f2)
                    if _tgt_f1 and _tgt_f2:
                        _attr = "✓ **真被物体挡停** (两指都接触目标 ✓)"
                    elif _gnd_f1 or _gnd_f2:
                        _attr = "✗ **被桌面挡停** ⇒ **不是夹住物体** ✗ (复审: '两指挡停 ≠ 被目标挡住' ✓)"
                    elif not _cp_f1 and not _cp_f2:
                        _attr = ("✗ **两指无任何接触** ⇒ 挡停来自 **PD/摩擦** 而非碰撞 ✗ "
                                 "⇒ 此时'真夹持率 100%'是**误导** ✗ (手指只是没走到位 ✗)")
                    else:
                        _attr = "⚠ 接触对象**不是目标** ✗ (可能在蹭别的物体 ✓)"
                    logger.info(
                        "[Stage2 真夹持判据·归因] 指1↔%s | 指2↔%s | 目标在指间=%s | 被桌面接触=%s "
                        "⇒ %s",
                        _cp_f1 or "[]", _cp_f2 or "[]", (_tgt_f1, _tgt_f2),
                        (_gnd_f1, _gnd_f2), _attr)
                    logger.info(
                        "[Stage2 真夹持判据] settle 段: 挡停量(中位) 指1=**%+.2f**mm 指2=**%+.2f**mm "
                        "| 两指同时挡停=%d/%d ⇒ **真夹持率=%.1f%%** | 真值落点: 两指**距离差=%s**, "
                        "**最近侧=%s** | **指尖离桌面=%s** (Stage1 好解基准: 挡停 +10.06/+5.72mm, "
                        "真夹持率 100%%)",
                        _blk_med[0], _blk_med[1], int(_both_blk.sum()), int(_blk.shape[0]),
                        100.0 * _tgr,
                        ("%.1fmm" % _sm_asym) if _sm_asym is not None else "n/a",
                        ("%.1fmm" % _sm_dmin) if _sm_dmin is not None else "n/a",
                        ("%.1fmm" % _gc_mm) if _gc_mm is not None else "n/a")
                    trace["s2_true_grasp_ratio"] = _tgr
                    trace["s2_block_mm"] = (float(_blk_med[0]), float(_blk_med[1]))
                    _tg_gate = float(os.environ.get("_S2_TRUE_GRASP_MIN", "0.0") or 0.0)
                    if _tg_gate > 0.0 and _tgr < _tg_gate:
                        fail(
                            "PROVISIONAL_GRASP",
                            f"真夹持率={100.0 * _tgr:.0f}% < {100.0 * _tg_gate:.0f}% "
                            f"(挡停 {_blk_med[0]:+.2f}/{_blk_med[1]:+.2f}mm ⇒ 未夹牢 ✗)",
                        )
                        if _stop:
                            break
                except Exception as _etg2:  # noqa: BLE001
                    logger.warning("[Stage2 真夹持判据] 计算失败: %s", _etg2)
                # ★★★★ 2026-09-17 闭合段逐控制点转储 (用户: "continue") ★★★★
                #   现状 (CoACD, 缓存 key 修好后): 门禁已推进到 `PROVISIONAL_GRASP` @hold_end ✓
                #     唯一卡点: settle bilateral = [F,F,F,F,F,F,F,F,F,T] ✗
                #       判据: sum >= settle_n-2 (=8) 且末 3 个全 True
                #     预紧单调有效 (0 → 1 个, 0.002 → 3 个, 0.004 → 4 个 ✓) 但 +0.002 只换 1 个 ✗
                #       ⇒ 到 8 个需 ~0.010 ⇒ 有过度挤压风险 ✗
                #     已排除: _Q_OPEN 变小 ✗ (擦碰↑) / q_near ✗ (项目废弃) / carry_time_scale ✗ (无影响)
                #   ⇒ 本转储: 看**闭合窗内手指到底走到哪**, 判定该延长闭合段还是该后移静置窗
                #     判据: 静置窗打开时 act 是否已到接触阈值 q≈0.0224 (物体凸包 y 半宽 37.6mm) ✓
                #           以及 f1/f2 分别何时变 1 ⇒ 是"还没夹上" ✗ 还是"单侧夹上" ✗
                try:
                    _cs = int(meta["close_start"])
                    _he = int(meta["hold_end"])
                    _qc_all = np.asarray(trace["gripper_q_cmd"], dtype=np.float64)
                    _qa_all = np.asarray(trace["gripper_q_actual"], dtype=np.float64)
                    _bl_all = np.asarray(trace["target_bilateral"], dtype=bool)
                    _ph_all = np.asarray(trace["phase"], dtype="U24")
                    _f1_all = np.asarray(trace["finger1_contact"], dtype=bool)
                    _f2_all = np.asarray(trace["finger2_contact"], dtype=bool)
                    _oT_all = np.asarray(trace["object_T"], dtype=np.float64)
                    _rows = []
                    for _k in range(_cs, min(_he + 1, len(_bl_all))):
                        # ★ 2026-09-17 加物体实际位置 (用户: "continue") ★
                        #   现象: act 从 0.0127(接触点) creep 到 0.0079 ✗ 后 f1/f2 双双失接触 ✗
                        #     ⇒ 指缝 46mm ✗ < 物体 55.9mm ✗ ⇒ **物体已不在指间** ✗
                        #   但"往哪跑"未知 ✗ ⇒ 量 dx/dy/dz (相对复位后静置位姿 ✓):
                        #     横向 dy 大 (闭合轴) ⇒ 被"推"出去 ✗ ⇒ 查摩擦/接触时序
                        #     垂向 dz 大 (上下)   ⇒ 被"楔"出去 ✗ ⇒ 查手指几何/开口方向
                        _dz = _oT_all[_k][:3, 3] - np.asarray(initial_object_T, dtype=np.float64)[:3, 3]
                        _rows.append(
                            "%d|%s cmd%.4f act%.4f[%.4f,%.4f] f1=%d f2=%d bl=%d objΔ[%+.1f,%+.1f,%+.1f]mm"
                            % (
                                _k, str(_ph_all[_k])[:10],
                                float(_qc_all[_k][0]), float(_qa_all[_k].mean()),
                                float(_qa_all[_k][0]), float(_qa_all[_k][1]),
                                int(_f1_all[_k]), int(_f2_all[_k]), int(_bl_all[_k]),
                                float(_dz[0] * 1000.0), float(_dz[1] * 1000.0),
                                float(_dz[2] * 1000.0),
                            )
                        )
                    logger.warning(
                        "[闭合段逐点] close_start=%d settle窗=%d..%d hold_end=%d\n%s",
                        _cs, int(meta["settle_start"]), int(meta["settle_end"]), _he,
                        "\n".join("    " + r for r in _rows),
                    )
                except Exception as _ec:  # noqa: BLE001
                    logger.warning("[闭合段逐点] 采集失败: %s", _ec)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ md 06.4-3 诊断 (2026-09-29) 【close/settle 未建立夹持 ⇒ 回查**入口状态**】 ✓ ★★★★
                #   md 原话 ✓: "close/settle没有建立稳定夹持: **回查入口状态和真实命令差异**。
                #     命令/状态可比后仍失败, 才考虑任务10共享反馈;
                #     **不靠放大 w、放松门或加大摩擦来宣布成功**" ✓
                #   ⇒ 实测已证伪"力/摩擦/刚度不足"(s73/s75 逐位无变化 ✗) ⇒ 回到 md 路径:
                #     打印 settle 末的 **手-物相对位姿 T_G_O + 指尖间距 vs 物体宽度** ⇒
                #     一眼看出"手指到底夹在什么位置 / 是不是空夹" ✓
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                try:
                    _tgo_l = np.asarray(trace["T_G_O"][-1], dtype=np.float64)
                    _qa_l = np.asarray(trace["gripper_q_actual"][-1], dtype=np.float64)
                    _ba_l = np.asarray(trace["base_actual"][-1], dtype=np.float64)
                    _bt_l = np.asarray(trace["base_target"][-1], dtype=np.float64)
                    _gap_mm = float(np.mean(_qa_l)) * 2.0 * 1000.0
                    _w_mm = float(((self.obj_info.get(target_name, {}) or {})
                                   .get("bbox_size", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0])[1]) * 1000.0
                    logger.warning(
                        "[06.4-3][诊断] settle末: T_G_O平移=%s mm | 手指q=%s | 双侧=%s || "
                        "**手基座实际**=%s | 手基座目标=%s | (Stage1 成功参考 "
                        "base=[-0.3138,-0.5086,0.0309])",
                        np.round(_tgo_l[:3, 3] * 1000.0, 2).tolist(),
                        np.round(_qa_l, 4).tolist(),
                        bool(settle_bilateral[-1]) if settle_bilateral else None,
                        np.round(_ba_l, 4).tolist(), np.round(_bt_l, 4).tolist())
                except Exception as _e643:  # noqa: BLE001
                    logger.warning("[06.4-3][诊断] 采集失败: %s", _e643)
                if len(settle_bilateral) != settle_n:
                    fail(
                        "PROVISIONAL_GRASP",
                        f"settle samples={len(settle_bilateral)} != {settle_n}",
                    )
                    if _stop:
                        break
                # ★★★★ 2026-09-21 【用户定策】"stage2 的停顿就是像 stage1 那样子**停顿 1s
                #   进行抓取**, 可以同意这个方式" ⇒ 判据应与"停顿期里完成抓取"这一事实对齐 ★★★★
                #   旧判据 ✗: `sum(settle_bilateral) < settle_n - 2` ⇒ 整个 settle 窗
                #     (10 控制点) 内最多只许缺 2 帧 ⇒ 实测失败形态 = "**SETTLE 前 5~6 帧非双侧**" ✗
                #     ⇒ 物理上不合理 ✓: 闭合刚结束的**前几帧**正是"物体被指腹推着**坐正**"的过程 ✓
                #       (代码自己的记录也承认这点: 为此专门在 SETTLE 里插了"开→再闭"重坐 ✓)
                #       ⇒ 把**前几帧的坐正过程**当失败 ⇒ 与"停顿 1s 抓取"的本意相反 ✗
                #   新判据 ✓ (两段式, 与 `PRE_F50_READY` 的稳健化同型 ✓):
                #     ① **尾段**必须稳: 最后 3 帧全双侧 ✓ (不变 ✓ —— 这保证"夹牢了才走" ✓)
                #     ② **整段占比**: 从"容忍 2 帧"放宽为**双侧占比 ≥ `_S2_SETTLE_BILATERAL_RATIO`**
                #        (默认 **0.7** ✓; 即 10 点时允许前 3 帧坐正 ✓)
                #     开关 ✓: `_S2_SETTLE_BILATERAL_RATIO=1.1` ⇒ 回旧行为 (任何缺帧都判死 ✓)
                _sb_ratio_min = float(os.environ.get("_S2_SETTLE_BILATERAL_RATIO", "0.7") or 0.7)
                _sb_ratio = (float(sum(settle_bilateral)) / float(settle_n)) if settle_n else 0.0
                if _sb_ratio < _sb_ratio_min or not all(settle_bilateral[-3:]):
                    fail(
                        "PROVISIONAL_GRASP",
                        f"settle bilateral={settle_bilateral} "
                        f"(占比 {_sb_ratio:.2f} < {_sb_ratio_min:.2f} 或末 3 帧不全 ✓)",
                    )
                    if _stop:
                        break
                if object_drift > PRE_GRASP_OBJECT_DRIFT_M:
                    fail("PROVISIONAL_GRASP", f"object drift={object_drift * 1000.0:.2f}mm")
                    if _stop:
                        break
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ T08-D (2026-09-27) 【C0 **只定义一次** + 闭合稳定末端取基准】 ✓ ★★★★
                #   md 原话 ✓: "手物参考相对位姿 **C0 取闭合稳定末端** 且搬运中**不随滑移更新**" ✓
                #   ⇒ 本处即唯一定义点 ✓ (settle 末端、双侧稳定、object drift 已过门 ✓)
                #   ⇒ 本改 ✓: 打点 + 计数 ⇒ **可复查"只定义一次"** ✓ (重复即 ERROR ✗)
                #     (只读+日志 ✓: 不改判据、不改门、不改基准数值 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★ T08-D2b ✓ 诊断: C0/T_G_O 的**原始两帧**世界坐标 ⇒ 一眼判"参考系错配"✗
                try:
                    _wg_d2 = (None if T_W_G is None
                              else [round(float(v), 4) for v in np.asarray(T_W_G, dtype=np.float64)[:3, 3]])
                    _wo_d2 = [round(float(v), 4) for v in np.asarray(T_W_O, dtype=np.float64)[:3, 3]]
                    _tgo_d2 = (None if T_G_O is None
                               else [round(float(v), 4) for v in np.asarray(T_G_O, dtype=np.float64)[:3, 3]])
                    logger.warning(
                        "[T08-D2b][C0 口径] T_W_G(夹爪基座世界)=%s | T_W_O(物体世界)=%s | "
                        "T_G_O(相对)位置=%s | ‖T_G_O.p‖=%.3fm %s",
                        _wg_d2, _wo_d2, _tgo_d2,
                        (0.0 if _tgo_d2 is None else float(np.linalg.norm(np.asarray(_tgo_d2)))),
                        "✗ 不可能(手-物不可能 ~1m)" if (_tgo_d2 is not None
                                                       and float(np.linalg.norm(np.asarray(_tgo_d2))) > 0.3)
                        else "✓ 合理")
                except Exception as _ed2d:  # noqa: BLE001
                    logger.warning("[T08-D2b] C0 口径日志失败: %s", _ed2d)
                _c0_cnt = int(getattr(self, "_s2_c0_def_count", 0)) + 1
                self._s2_c0_def_count = _c0_cnt
                tgo_reference = T_G_O.copy() if T_G_O is not None else None
                logger.warning(
                    "[T08-D] ✓ C0 = 手-物参考相对位姿 **第 %d 次定义** (闭合稳定末端 settle 末 ✓) | "
                    "T_G_O 位置=%s | 搬运中**不更新** ✓",
                    _c0_cnt,
                    # ★★ 任务06 (2026-09-28) 修 **C0 日志 bug** ✓ ★★
                    #   md §06 原话 ✓: "修 **C0 日志取旋转行冒充位置**的…" ✗
                    #   实测 ✗: 旧写法 `np.asarray(T_G_O).reshape(-1)[:3]` 取的是 4×4 的
                    #     **前 3 个元素 = 第一行 (旋转矩阵的 x 行)** ⇒ 日志里显示的"位置"
                    #     根本不是平移列 ✗ ⇒ 会把正常 C0 误读成"参考系错配" ✗
                    #   本改 ✓: 取平移列 `[:3, 3]` (与上面 `[T08-D2b][C0 口径]` 口径一致 ✓)
                    (None if tgo_reference is None
                     else [round(float(v), 5)
                           for v in np.asarray(tgo_reference, dtype=np.float64)[:3, 3]]))
                if _c0_cnt > 1:
                    logger.error("[T08-D] ✗ C0 被**重复定义** %d 次 ⇒ 违反 md\"只定义一次\" ✗ "
                                 "(后续定义必须删除 ✗)", _c0_cnt)
                if tgo_reference is None or not np.isfinite(tgo_reference).all():
                    fail("PROVISIONAL_GRASP", "missing finite T_G_O reference")
                    if _stop:
                        break

            # 2026-09-13: carry_verify_controls=0 时该相位为空 (用户要求删除这 5 个额外帧),
            # 此时整段跳过 —— 否则空切片会被判 sum(bilateral)=0 < 4 而误报 CARRY_VERIFY.
            if index == meta["verify_end"] and _verify_n > 0:
                lo, hi = meta["verify_start"], meta["verify_end"] + 1
                verify_bilateral = trace["target_bilateral"][lo:hi]
                verify_tgo = trace["T_G_O"][lo:hi]
                t_err, r_err = _relative_pose_errors(tgo_reference, verify_tgo)
                relative_speed = (
                    float(np.max(np.linalg.norm(np.diff(np.asarray(verify_tgo)[:, :3, 3], axis=0), axis=1)) * CONTROL_FREQ)
                    if len(verify_tgo) >= 2
                    else float("inf")
                )
                if sum(verify_bilateral) < max(1, len(verify_bilateral) - 1):
                    fail("CARRY_VERIFY",
                         f"bilateral={sum(verify_bilateral)}/{len(verify_bilateral)}")
                    if _stop:
                        break
                if (
                    len(t_err) != _verify_n
                    or not np.isfinite(t_err).all()
                    or t_err.max() > TGO_TRANSLATION_TOL_M
                    or r_err.max() > TGO_ROTATION_TOL_RAD
                    or relative_speed > TGO_RELATIVE_SPEED_MPS
                ):
                    fail(
                        "CARRY_VERIFY",
                        f"T_G_O={t_err.max() * 1000.0 if len(t_err) else np.inf:.2f}mm/"
                        f"{np.rad2deg(r_err.max()) if len(r_err) else np.inf:.2f}deg, "
                        f"relative_speed={relative_speed * 1000.0:.2f}mm/s",
                    )
                    if _stop:
                        break

            if record_video:
                self._record_stage2_frame(bundle, index)

        # 2026-09-13 ★ 帧数断言: 录制帧数必须等于 timeline 控制点数, 否则响亮告警.
        # (用户反馈"渲染不完整" —— 此前录制静默丢帧无任何提示, 这类问题只能靠断言暴露.)
        if record_video:
            _written = int(getattr(self, "_stage2_frames_written", 0))
            _expected = len(bundle["source_frame"])
            if _written == _expected:
                logger.info("[Stage2 render] 已录制 %d 帧 = timeline %d 控制点 ✓",
                            _written, _expected)
            else:
                logger.error(
                    "[Stage2 render] ★ 录制帧数不符: 实际 %d 帧 vs timeline %d 控制点 "
                    "(缺 %d 帧) → 视频不完整! 检查 writer 是否可用 / 是否有异常被吞.",
                    _written, _expected, _expected - _written)

        # Convert trace fields used numerically before evaluating the final gate.
        for key in (
            "control_index",
            "source_frame",
            "base_target",
            "base_target_R",
            "base_cmd_euler",
            "base_actual",
            "gripper_q_cmd",
            "gripper_q_actual",
            "object_T",
            "gripper_T",
            "T_G_O",
            "finger1_contact",
            "finger2_contact",
            "target_bilateral",
            "target_contact",
            "ground_contact",
            "non_target_contact",
            "environment_contact",
            "grasp_force",
        ):
            trace[key] = np.asarray(trace[key])
        trace["phase"] = np.asarray(trace["phase"], dtype="U24")

        # ★★★★ 2026-09-14 实验 A': 主循环结束后**就地**再走 N 步 (同命令) ★★★★
        #   (用户: "继续看看") — 判决实验 A 已证明爆炸与**命令无关** ✗
        #     (发 Stage2 自己的最后一条命令, 与当前状态只差 0.8mm ✓, 照样炸出 101.587 rad/s ✗)
        #   且 grep 证实: 本函数 L1900-2694 内**没有任何写操作** ✗ (set_qpos/set_pose/scene.step 全在 L1792-1815)
        #   ⇒ 剩下唯一岔路:
        #       (a) 主循环**结束后**的状态本身不稳 ⇒ 这里就地再走也炸 ✗
        #       (b) 是"从**外面**(Stage3)调 physics_step"这件事 ⇒ 这里就地走**不炸** ✓
        #   本实验: 循环刚结束、CSV/metrics 之前, 用同一条命令就地再走 N 步 ✓
        _extra_steps = int(os.environ.get("_S2_EXTRA_STEPS", "0") or 0)
        if _extra_steps > 0:
            try:
                _ex_p = np.asarray(bundle["base_pos"], dtype=np.float64)[-1]
                _ex_R = np.asarray(bundle["base_R"], dtype=np.float64)[-1]
                _ex_q = np.asarray(bundle["gripper_q"], dtype=np.float64)[-1]
                _ex_z, _ex_y, _ex_x = rotmat_to_zyx_euler(_ex_R)
                _ex_qv0 = np.asarray(robot.get_qvel(), dtype=np.float64)
                logger.warning(
                    "[实验A'] 主循环结束, 就地再走 %d 步 | 步前 基座 rz qvel=%.3f 手指 qpos=%s 物体 z=%.4f",
                    _extra_steps, float(_ex_qv0[virtual_idx["rz"]]),
                    np.round(np.asarray(robot.get_qpos(), dtype=np.float64)[[gi1, gi2]], 4).tolist(),
                    float(target_actor.get_pose().p[2]),
                )
                for _ex_i in range(_extra_steps):
                    for _ex_k, _ex_v in (
                        ("vx", _ex_p[0]), ("vy", _ex_p[1]), ("vz", _ex_p[2]),
                        ("rz", _ex_z), ("ry", _ex_y), ("rx", _ex_x),
                    ):
                        active_joints[virtual_idx[_ex_k]].set_drive_target(float(_ex_v))
                    physics_step(
                        robot, arm_indices, gi1, gi2, np.array([]),
                        float(_ex_q[0]), float(_ex_q[1]), self.scene,
                    )
                    _ex_qv1 = np.asarray(robot.get_qvel(), dtype=np.float64)
                    logger.warning(
                        "[实验A'] 就地第 %d 步后 | 基座 rz qvel=%.3f | 手指 qpos=%s | 物体 z=%.4f "
                        "| 夹爪欧拉=%.1f",
                        _ex_i + 1, float(_ex_qv1[virtual_idx["rz"]]),
                        np.round(np.asarray(robot.get_qpos(), dtype=np.float64)[[gi1, gi2]], 4).tolist(),
                        float(target_actor.get_pose().p[2]),
                        float(np.rad2deg(rotmat_to_zyx_euler(
                            self._gripper_pose_matrix(robot, side)[:3, :3])[0])),
                    )
            except Exception as _ex_ea:  # noqa: BLE001
                logger.warning("[实验A'] 采集失败: %s", _ex_ea)

        tracking = compute_tracking_error(trace["base_actual"], trace["base_target"])
        metrics = {
            "tracking_error": tracking,
            "ready_samples": ready_samples,
            "reset_to_warmup_translation_mm": float(
                np.linalg.norm(warmup_object_T[:3, 3] - reset_object_T[:3, 3]) * 1000.0
            ),
            "reset_to_warmup_rotation_deg": float(
                np.rad2deg(_rotation_distance(reset_object_T[:3, :3], warmup_object_T[:3, :3]))
            ),
            # 2026-09-13 (A): 物体静置自稳定量 (不计入 RESET_DRIFT, 仅追溯)
            "reset_settle_translation_mm": float(settle_move_mm),
            "reset_settle_rotation_deg": float(settle_rot_deg),
            "reset_settle_steps": int(_settle_steps),
            "prefix_target_contact_controls": int(prefix_target_contacts),
            "prefix_target_contact_indices": list(prefix_contact_indices),
            "prefix_non_target_contact_controls": int(prefix_leak_contacts),
            "approach_target_contact_controls": int(approach_target_contacts),
            "approach_non_target_contact_controls": int(approach_leak_contacts),
            # Historical key names kept for existing log/artifact consumers.
            "prefix_environment_contact_controls": int(prefix_leak_contacts),
            "approach_environment_contact_controls": int(approach_leak_contacts),
            "target_contact_names": sorted(target_contact_names),
            "non_target_contact_names": sorted(leak_contact_names),
            "ground_contact_names": sorted(ground_contact_names),
            "environment_contact_names": sorted(leak_contact_names),
            "settle_bilateral_ratio": float(
                np.mean(trace["target_bilateral"][meta["settle_start"] : meta["settle_end"] + 1])
            ),
            "stage1_reuse_report": reuse_report,
            "force_peak_n": float(np.max(trace["grasp_force"])) if len(trace["grasp_force"]) else float("nan"),
            "force_p95_n": float(np.percentile(trace["grasp_force"], 95)) if len(trace["grasp_force"]) else float("nan"),
        }

        if gate_result["failure"] is None:
            tgo_samples = list(trace["T_G_O"][meta["verify_start"] :])
            t_err, r_err = _relative_pose_errors(tgo_reference, tgo_samples)
            bilateral = trace["target_bilateral"][meta["verify_start"] :]
            # Only the two disabled pairs count as a leak; ground contact after
            # F50 is genuine Stage1 physics and is reported, not failed.
            post_grasp_leak = trace["non_target_contact"][meta["close_start"] :]
            object_start = trace["object_T"][meta["hold_end"], :3, 3]
            commanded_progress = float(
                np.linalg.norm(bundle["base_pos"][-1] - bundle["base_pos"][meta["hold_end"]])
            )
            commanded_delta = bundle["base_pos"][-1] - bundle["base_pos"][meta["hold_end"]]
            object_delta = trace["object_T"][-1, :3, 3] - object_start
            object_progress = (
                float(np.dot(object_delta, commanded_delta / commanded_progress))
                if commanded_progress > 1e-9
                else 0.0
            )
            required_progress = max(0.020, 0.5 * commanded_progress)
            bilateral_ratio = float(np.mean(bilateral)) if len(bilateral) else 0.0
            metrics.update(
                {
                    "tgo_translation_max_mm": float(t_err.max() * 1000.0) if len(t_err) else float("nan"),
                    "tgo_rotation_max_deg": float(np.rad2deg(r_err.max())) if len(r_err) else float("nan"),
                    "bilateral_ratio": bilateral_ratio,
                    "object_progress_mm": object_progress * 1000.0,
                    "required_progress_mm": required_progress * 1000.0,
                    "post_grasp_leak_controls": int(np.sum(post_grasp_leak)),
                    "post_grasp_ground_contact_controls": int(
                        np.sum(trace["ground_contact"][meta["close_start"] :])
                    ),
                    # ★★ 任务06 (2026-09-28) ✓ md §06 最后一条要求报告: "**原参考终点覆盖**、
                    #   物体沿任务方向移动进度、5mm/5° 相对稳定、MANO 偏差、碰撞和**实际时长**;
                    #   不能把抬起保持替代搬运" ✓
                    #   ⇒ 前五项已有 (tgo_* / object_progress_mm / bilateral / ground_contact / T08-E) ✓
                    #   ⇒ 这两项**新增** ✓: 终点覆盖 + 实际时长 (不再只报"抬起高度"✗)
                    "reference_end_frame": float(bundle["source_frame"][-1]),
                    "reference_end_covered": bool(
                        float(bundle["source_frame"][-1])
                        >= float(meta.get("f_end", bundle["source_frame"][-1])) - 1e-6),
                    "actual_duration_s": float(
                        len(trace["control_index"]) / float(CONTROL_FREQ)),
                    "controls": int(len(trace["control_index"])),
                }
            )
            if len(trace["control_index"]) != len(bundle["source_frame"]):
                fail("CARRY", "timeline did not finish")
            elif not np.isfinite(t_err).all() or not np.isfinite(r_err).all():
                fail("CARRY", "non-finite T_G_O metric")
            elif t_err.max() > TGO_TRANSLATION_TOL_M or r_err.max() > TGO_ROTATION_TOL_RAD:
                # ★ CARRY 段摘要诊断 (2026-09-29) ✓: 抓取侧已校准为"两侧一致/正常" ✓
                #   ⇒ 失败只可能在 **搬运段本身** ⇒ 打印搬运段物体轨迹与手指开合,
                #     直接回答"物体是**从未动**还是**动了一下又掉**" ✓
                try:
                    _ot = np.asarray(trace["object_T"], dtype=np.float64)
                    _sel = [i for i, p in enumerate(trace["phase"]) if str(p) == "CARRY"]
                    _pick = [ _sel[0], _sel[len(_sel) // 2], _sel[-1] ] if _sel else []
                    _qa_c = np.asarray(trace["gripper_q_actual"], dtype=np.float64)
                    _qc_c = np.asarray(trace["gripper_q_cmd"], dtype=np.float64)
                    _co = trace.get("contact_objects", [])
                    _bl = trace.get("target_bilateral", [])
                    _ba = np.asarray(trace.get("base_actual", []), dtype=np.float64)
                    _bt = np.asarray(trace.get("base_target", []), dtype=np.float64)
                    logger.warning(
                        "[CARRY诊断] 搬运点数=%d (trace总长=%d/%d) | 物体p 首/中/末=%s | "
                        "物体总位移=%.2fmm | q_act 首/中/末=%s | q_cmd 首/中/末=%s | "
                        "基座实际z 首/中/末=%s | 基座目标z 首/中/末=%s | "
                        "接触对象 首/中/末=%s | 双侧 首/中/末=%s",
                        len(_sel), len(_ot), len(trace["control_index"]),
                        [np.round(_ot[i][:3, 3], 4).tolist() for i in _pick],
                        float(np.linalg.norm(_ot[_pick[-1]][:3, 3] - _ot[_pick[0]][:3, 3])
                              * 1000.0) if _pick else float("nan"),
                        np.round(_qa_c[_pick].mean(axis=1), 4).tolist() if _pick else None,
                        np.round(_qc_c[_pick].mean(axis=1), 4).tolist() if _pick else None,
                        [round(float(_ba[i][2]), 4) for i in _pick] if len(_ba) > max(_pick) else None,
                        [round(float(_bt[i][2]), 4) for i in _pick] if len(_bt) > max(_pick) else None,
                        [list(_co[i])[:3] if i < len(_co) else None for i in _pick],
                        [_bl[i] if i < len(_bl) else None for i in _pick])
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    # ★★★★ 2026-09-30 (md 未完成项 ④) 【first_failure.json 落盘】 ✓ ★★★★
                    #   md 原话 ✓: "输出**首次失联前后**至少 5 个控制点的**原始数据**
                    #     (时间戳、命令、实际、原始 qpos、驱动/穿渗耗时、质量速度和分离速度);
                    #     固定 C0; 若失败则任何**计划/执行生成路径**都不得覆盖该记录" ✓
                    #   本改 ✓: ① 固定 C0 = **首个 CARRY 控制点的命令位姿** (记录在案 ✓)
                    #     ② 首次失联点 = `target_bilateral` 由 True 变 False 的第一点 ✓
                    #     ③ 窗口 = 失联点 **前后各 ≥5** 个控制点的原始数据 ✓
                    #     ④ 独立文件名 + 时间戳 ⇒ **不会被后续 run 覆盖** ✓ (B2/A7 同类纪律 ✓)
                    #   ⚠ 诚实 ✓: "驱动/穿渗耗时、质量速度和分离速度"本管线**没有**这些探针 ⇒
                    #     字段**显式留空并注明缺失** ✗ (不伪造数值 ✗)
                    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                    try:
                        import json as _j_ff
                        from pathlib import Path as _P_ff
                        _ph = [str(x) for x in trace["phase"]]
                        _sel_i = [i for i, _p in enumerate(_ph) if _p == "CARRY"]
                        _lost = None
                        for _i in (_sel_i or []):
                            if _i < len(_bl) and not bool(_bl[_i]):
                                _lost = int(_i)
                                break
                        _c0 = (_lost if _lost is not None
                               else (_sel_i[0] if _sel_i else 0))
                        _w0 = max(0, _c0 - 5)
                        _w1 = min(len(_ph), _c0 + 6)
                        _qa_ = np.asarray(trace.get("gripper_q_actual", []), dtype=np.float64)
                        _qc_ = np.asarray(trace.get("gripper_q_cmd", []), dtype=np.float64)
                        _ba_ = np.asarray(trace.get("base_actual", []), dtype=np.float64)
                        _bt_ = np.asarray(trace.get("base_target", []), dtype=np.float64)
                        _ot_ = np.asarray(trace.get("object_T", []), dtype=np.float64)
                        _sf_ = trace.get("source_frame", [])
                        _rows = []
                        for _i in range(_w0, _w1):
                            _rows.append({
                                "control_index": int(_i),
                                "source_frame": (float(_sf_[_i]) if _i < len(_sf_) else None),
                                "phase": _ph[_i],
                                "q_cmd": (np.round(_qc_[_i], 6).tolist() if _i < len(_qc_) else None),
                                "q_actual": (np.round(_qa_[_i], 6).tolist() if _i < len(_qa_) else None),
                                "base_actual": (np.round(_ba_[_i], 6).tolist() if _i < len(_ba_) else None),
                                "base_target": (np.round(_bt_[_i], 6).tolist() if _i < len(_bt_) else None),
                                "object_p": (np.round(_ot_[_i][:3, 3], 6).tolist() if _i < len(_ot_) else None),
                                "contact_objects": (list(_co[_i]) if _i < len(_co) else None),
                                "bilateral": (bool(_bl[_i]) if _i < len(_bl) else None),
                                "driver_penetration_times_ms": "MISSING(no probe)",
                                "mass_velocity": "MISSING(no probe)",
                                "separation_velocity": "MISSING(no probe)",
                            })
                        # ★ f2 ✓: **C0 与 gate 同一个实际 4×4** + settle index + 四类首事件取最早
                        _ph_all = [str(x) for x in trace["phase"]]
                        _settle_idx = None
                        for _i2 in range(len(_ph_all) - 1, -1, -1):
                            if _ph_all[_i2] == "STAGE1_SETTLE":
                                _settle_idx = int(_i2)
                                break
                        _C0_f2 = None
                        try:
                            if _settle_idx is not None and len(_ot_) > _settle_idx:
                                _T_W_O0 = np.asarray(_ot_[_settle_idx], dtype=np.float64)
                                _T_W_G0 = np.eye(4)
                                if len(_ba_) > _settle_idx and np.asarray(_ba_[_settle_idx]).size >= 3:
                                    _b0 = np.asarray(_ba_[_settle_idx], dtype=np.float64)[:3]
                                    _T_W_G0[:3, 3] = _b0
                                _C0_f2 = np.linalg.inv(_T_W_G0) @ _T_W_O0
                        except Exception:  # noqa: BLE001
                            _C0_f2 = None
                        # 四类首事件 (各自首个 index ⇒ 取**最早有效事件** ✓)
                        _ev = {}
                        try:
                            _qc2 = np.asarray(trace.get("gripper_q_cmd", []), dtype=np.float64)
                            _qa2 = np.asarray(trace.get("gripper_q_actual", []), dtype=np.float64)
                            for _i2 in range(min(len(_qc2), len(_qa2))):
                                if not np.isfinite(_qc2[_i2]).all() or not np.isfinite(_qa2[_i2]).all() \
                                        or float(np.max(np.abs(_qc2[_i2] - _qa2[_i2]))) > 0.01:
                                    _ev["first_command_anomaly"] = int(_i2)
                                    break
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            _nf1 = [float(x) for x in (trace.get("n_f1") or [])]
                            _nf2 = [float(x) for x in (trace.get("n_f2") or [])]
                            _thr = float(os.environ.get("_S2_MIN_NORM_N", "0.5") or 0.5)
                            # ★ f2 ✓: 判定**起点**必须是 settle 之后 —— 在此之前
                            #   "无接触/无力"是**合法**的 ✗ ⇒ 不能算"载荷不足" ✓
                            #   (首版从第 0 点开始判 ⇒ 得到 index=0 ⇒ 无意义 ✗)
                            _from2 = int(_settle_idx) + 1 if _settle_idx is not None else 0
                            for _i2 in range(_from2, min(len(_nf1), len(_nf2))):
                                if (_nf1[_i2] < _thr) or (_nf2[_i2] < _thr):
                                    _ev["first_load_insufficient"] = int(_i2)
                                    break
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            _tg = [float(x) for x in (trace.get("tgo_translation_mm") or [])]
                            for _i2, _v2 in enumerate(_tg):
                                if float(_v2) > 5.0:
                                    _ev["first_relative_error_exceeded"] = int(_i2)
                                    break
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            # ★ f2 ✓: 与载荷同理 ⇒ 判定起点必须是 **settle 之后**
                            #   (之前无接触是**合法**的 ✗; 首版从第 0 点判 ⇒ index=0 无意义 ✗)
                            _from3 = int(_settle_idx) + 1 if _settle_idx is not None else 0
                            for _i2 in range(_from3, len(_bl)):
                                if not bool(_bl[_i2]):
                                    _ev["first_contact_lost"] = int(_i2)
                                    break
                        except Exception:  # noqa: BLE001
                            pass
                        _ev_valid = {k: v for k, v in _ev.items() if v is not None}
                        _earliest = (min(_ev_valid, key=lambda k: _ev_valid[k])
                                     if _ev_valid else "UNKNOWN(无有效事件 ⇒ 不推导 ✗)")
                        # ★ md g3 (2026-10-01) ⚠ **降级声明** ✗
                        #   这里**只能**算"指缝 vs 目标世界包围盒跨度"的**包络比较** ⇒
                        #   **不能**当作根因 (用户已驳回: 39.15mm 是世界 AABB 最小边,
                        #   **不是闭合轴跨度** ✗; 且此处用 **live hull** 诊断历史点 218
                        #   违反 g3"无相位快照不可用最终 live hull 诊断历史点" ✗)
                        #   ⇒ 本块输出仅作**参考量**, verdict 一律带"包络/非根因"标记,
                        #     真正的夹持判定须等 g3 完成: 同帧指面 + 同一闭合轴投影 +
                        #     指腹有限覆盖 + 真实周期载荷 ✓
                        _geo = {}
                        try:
                            _hulls = self._coacd_world_hulls_live(target_name)
                            if _hulls:
                                _all = np.vstack([np.asarray(h, dtype=np.float64)
                                                  for h in _hulls])
                                _ext = (_all.max(0) - _all.min(0))
                                _geo["target_extent_mm"] = [round(float(x) * 1000.0, 2)
                                                            for x in _ext]
                                _geo["target_span_max_mm"] = round(float(_ext.max()) * 1000.0, 2)
                                _geo["target_span_min_mm"] = round(float(_ext.min()) * 1000.0, 2)
                        except Exception as _eg:  # noqa: BLE001
                            _geo["hull_error"] = str(_eg)
                        try:
                            _idx_g = int(_ev.get("first_load_insufficient", 218))
                            _qa_g = np.asarray(trace["gripper_q_actual"], dtype=np.float64)
                            _q_g = float(np.asarray(_qa_g[_idx_g]).reshape(-1).mean())
                            _gap_mm = float(FINGER_GAP0) * 1000.0 + 2.0 * _q_g * 1000.0
                            _geo["index"] = _idx_g
                            _geo["finger_gap_mm"] = round(_gap_mm, 2)
                            _smax = _geo.get("target_span_max_mm")
                            _smin = _geo.get("target_span_min_mm")
                            if _smax is None:
                                _geo["verdict"] = "UNKNOWN(凸包不可读 ⇒ 不推导 ✗)"
                            elif _gap_mm > float(_smax):
                                _geo["verdict"] = (
                                    "[包络/非根因] 指缝 %.1fmm > **世界 AABB** 最大跨度 %.1fmm "
                                    "⇒ 差 %.1fmm (仅包络比较, **不是**闭合轴跨度 ⇒ 不作根因 ✗)"
                                    % (_gap_mm, float(_smax), _gap_mm - float(_smax)))
                            elif _gap_mm < float(_smin):
                                _geo["verdict"] = (
                                    "[包络/非根因] 指缝 %.1fmm < 世界 AABB 最小跨度 %.1fmm "
                                    "⇒ 需按**同一闭合轴投影**复核 (不作根因 ✗)"
                                    % (_gap_mm, float(_smin)))
                            else:
                                _geo["verdict"] = (
                                    "[包络/非根因] 指缝 %.1fmm 介于 [%.1f, %.1f] "
                                    "⇒ 须等 g3 的闭合轴投影/指腹覆盖判定" % (_gap_mm,
                                                                            float(_smin), float(_smax)))
                        except Exception as _eg2:  # noqa: BLE001
                            _geo["q_error"] = str(_eg2)
                        # ★★ md 06.4-g3 ✅【**同帧 + 同一闭合轴**投影 ⇒ 真正的夹持判定】
                        #   用该 index 的**双指实际世界位姿** + **该 index 的 T_W_O** +
                        #   **局部** CoACD 顶点(形状不变, 缓存 ✓) ⇒ 全部同一时刻/同一坐标 ✓
                        #   ⚠ 指腹**有限覆盖**未建模 ⇒ 结论按"保守"(偏向高估间隙)标注 ✓
                        _g3 = {}
                        try:
                            _i3 = int(_ev.get("first_load_insufficient",
                                               _ev.get("first_contact_lost", 217)))
                            _f1 = (trace.get("finger1_pos") or [])[_i3]
                            _f2 = (trace.get("finger2_pos") or [])[_i3]
                            _ca3 = (trace.get("close_axis") or [])[_i3]
                            _T3 = np.asarray(_ot_[_i3], dtype=np.float64)
                            if _f1 and _f2 and _ca3:
                                _p1 = np.asarray(_f1, dtype=np.float64)
                                _p2 = np.asarray(_f2, dtype=np.float64)
                                _ax = np.asarray(_ca3, dtype=np.float64)
                                _g3 = {"index": int(_i3), "finger1_pos": _f1,
                                       "finger2_pos": _f2, "close_axis": _ca3}
                                _loc3 = getattr(self, "_coacd_local_hulls__%s"
                                                % str(target_name), None)
                                if not _loc3:
                                    _g3["verdict"] = "UNKNOWN(无缓存局部凸包 ⇒ 不推导 ✗)"
                                else:
                                    _W = np.vstack([(_T3 @ np.asarray(v, dtype=np.float64).T).T
                                                    for v in _loc3])
                                    _pr = _W @ _ax
                                    _lo, _hi = float(_pr.min()), float(_pr.max())
                                    _g1a, _g2a = float(_p1 @ _ax), float(_p2 @ _ax)
                                    _gl, _gh = min(_g1a, _g2a), max(_g1a, _g2a)
                                    _g3["target_axis_projection_mm"] = [
                                        round(_lo * 1000.0, 2), round(_hi * 1000.0, 2)]
                                    _g3["finger_span_on_axis_mm"] = [
                                        round(_gl * 1000.0, 2), round(_gh * 1000.0, 2)]
                                    _ov_l = _gl - _lo      # 左侧嵌入 (>0 = 压入)
                                    _ov_r = _hi - _gh      # 右侧嵌入
                                    _g3["embed_left_mm"] = round(_ov_l * 1000.0, 2)
                                    _g3["embed_right_mm"] = round(_ov_r * 1000.0, 2)
                                    if _ov_l > 0 and _ov_r > 0:
                                        _g3["verdict"] = (
                                            "双侧已嵌入 L%.2fmm/R%.2fmm ⇒ 若法向力仍<阈值 ⇒ "
                                            "承载/接触问题(非几何 ✗)" % (_ov_l * 1000.0,
                                                                          _ov_r * 1000.0))
                                    elif _ov_l > 0 or _ov_r > 0:
                                        _g3["verdict"] = "**单侧嵌入** ⇒ 非对称 (L%.2f/R%.2f mm) ⇒ " \
                                            "查另一侧为何未压上" % (_ov_l * 1000.0,
                                                                   _ov_r * 1000.0)
                                    else:
                                        _g3["verdict"] = (
                                            "**双侧都有间隙** L%.2fmm/R%.2fmm ⇒ 手指未压上(几何 ✗) "
                                            "(保守估计: 未建模指腹覆盖 ⇒ 实际可能更小)"
                                            % (-_ov_l * 1000.0, -_ov_r * 1000.0))
                        except Exception as _e3:  # noqa: BLE001
                            _g3["error"] = str(_e3)
                        _ff = {
                            "tag": "Stage2 CARRY first failure",
                            "g3_same_frame_probe": _g3,
                            "geometry_probe": _geo,
                            "C0_4x4": (None if _C0_f2 is None
                                       else np.round(_C0_f2, 6).tolist()),
                            "C0_definition": "settle 末的 T_W_G^-1 · T_W_O (与 gate 同一个实际 4×4 ✓)",
                            "settle_index": _settle_idx,
                            "first_events": _ev,
                            "earliest_valid_event": _earliest,
                            "load_threshold_N": float(os.environ.get("_S2_MIN_NORM_N", "0.5") or 0.5),
                            "failure": "CARRY",
                            "failure_detail": "T_G_O tracking tolerance exceeded",
                            "fixed_C0": {
                                "definition": "首个 CARRY 控制点的**命令**基座位姿 (此后不得二次生成 ✓)",
                                "first_carry_index": (int(_sel_i[0]) if _sel_i else None),
                                "base_target": (np.round(_bt_[int(_sel_i[0])], 6).tolist()
                                                if (_sel_i and len(_bt_) > _sel_i[0]) else None),
                            },
                            "first_loss_index": _lost,
                            "window": {"from": int(_w0), "to": int(_w1) - 1, "n": len(_rows)},
                            "rows": _rows,
                            "missing_probes": ["driver_penetration_times_ms",
                                               "mass_velocity", "separation_velocity"],
                        }
                        _pf_ff = (_P_ff(str(getattr(self, "output_dir", "output")))
                                  / "stage2" / "first_failure.json")
                        _pf_ff.parent.mkdir(parents=True, exist_ok=True)
                        _pf_ff.write_text(_j_ff.dumps(_ff, indent=2, default=str),
                                          encoding="utf-8")
                        logger.warning("[first_failure] ✓ 已落盘: %s | 首次失联点=%s | "
                                       "窗口 %d 点 (含前后 ≥5 ✓)", _pf_ff, _lost, len(_rows))
                    except Exception as _eff:  # noqa: BLE001
                        logger.warning("[first_failure] 落盘失败(不阻断 ✓): %s", _eff)
                except Exception as _ecar:  # noqa: BLE001
                    logger.warning("[CARRY诊断] 采集失败: %s", _ecar)
                fail(
                    "CARRY",
                    f"T_G_O drift={t_err.max() * 1000.0:.2f}mm/{np.rad2deg(r_err.max()):.2f}deg",
                )
            elif bilateral_ratio < 0.8:
                fail("CARRY", f"bilateral ratio={bilateral_ratio:.3f}")
            elif np.any(post_grasp_leak):
                fail(
                    "COLLISION_FILTER_LEAK",
                    "gripper/target contacted a non-target body after F50 "
                    f"({sorted(leak_contact_names)})",
                )
            elif object_progress + 1e-9 < required_progress:
                fail(
                    "CARRY",
                    f"object progress={object_progress * 1000.0:.1f}mm < {required_progress * 1000.0:.1f}mm",
                )
            elif (
                not np.isfinite(tracking["mean_mm"])
                or tracking["mean_mm"] > TRACKING_MEAN_TOL_MM
                or tracking["max_mm"] > TRACKING_MAX_TOL_MM
            ):
                fail(
                    "TRACK",
                    f"tracking mean/max={tracking['mean_mm']:.2f}/{tracking['max_mm']:.2f}mm",
                )
            else:
                gate_result.update(passed=True, failure=None, failure_detail="")

        # ── 分阶段跟踪剖析 (2026-09-10) ─────────────────────────────────
        # 目的: 定位 base 跟踪误差(位置/姿态)出现在哪个阶段, 以区分
        #  "PD 跟不上" 还是 "被接触力顶开". 只做诊断, 不参与门禁判定.
        try:
            _bt = np.asarray(trace["base_target"], dtype=np.float64)
            _br = np.asarray(trace["base_target_R"], dtype=np.float64)
            _gt = np.asarray(trace["gripper_T"], dtype=np.float64)
            _ph = np.asarray(trace["phase"], dtype=object)
            _prof: dict[str, dict[str, float]] = {}
            if len(_bt):
                for _name in (
                    "MANO_FOLLOW", "APPROACH_ADJUST", "FINAL_APPROACH",
                    "STAGE1_CLOSE", "STAGE1_SETTLE", "CARRY_VERIFY", "CARRY",
                ):
                    _sel = _ph == _name
                    if not _sel.any():
                        continue
                    _dp = np.linalg.norm(
                        np.asarray(trace["base_actual"], dtype=np.float64)[_sel] - _bt[_sel],
                        axis=1,
                    ) * 1000.0
                    _rows = np.where(_sel)[0]
                    # ★★★★ 2026-09-14 修正指标口径 (用户: "夹爪肯定反了" + 169.02° 恒定) ★★★★
                    #   旧实现: _rotation_distance(gripper_T[i], base_R[i])
                    #            ← **夹爪末端(真实物理)** vs **虚拟基座(命令)** ✗✗
                    #            ⇒ 两者相差一个固定的手臂运动学变换 ⇒ 夹角**恒为 ~169°** ✗
                    #            (实测 rot5/s3cmd/gim1/ufix 全部 = 169.02°, 一位不差 ✓
                    #             ⇒ 恒定值不可能来自动态跟踪误差 ✓ ⇒ 纯测量口径错 ✗)
                    #   新实现: 比"**同类之间**"的相对旋转变化 ✓
                    #            · _dr_abs : 夹爪实际 vs 基座命令 —— 保留但标注为跨坐标系(仅参考) ✓
                    #            · _dr     : 夹爪实际相对旋转 vs 基座命令相对旋转 (都相对本段首帧 ✓)
                    #                        ⇒ 消掉那个固定变换 ⇒ 只反映真正的跟踪误差 ✓
                    _Rg0 = None
                    _Rb0 = None
                    _dr_list, _dr_abs_list = [], []
                    for i in _rows:
                        if not np.isfinite(_gt[i]).all():
                            continue
                        _Rg = np.asarray(_gt[i], dtype=np.float64)[:3, :3]
                        _Rb = np.asarray(_br[i], dtype=np.float64)
                        if _Rg0 is None:
                            _Rg0, _Rb0 = _Rg.copy(), _Rb.copy()
                        _dr_abs_list.append(_rotation_distance(_Rg, _Rb))
                        # 相对量: 把两端都归到本段首帧, 固定偏差自动抵消 ✓
                        _dr_list.append(_rotation_distance(
                            _Rg @ _Rg0.T, _Rb @ _Rb0.T))
                    _dr = np.array(_dr_list, dtype=np.float64)
                    _dr_abs = np.array(_dr_abs_list, dtype=np.float64)
                    if _dr_abs.size:
                        _prof_dbg = float(np.rad2deg(_dr_abs.mean()))
                    else:
                        _prof_dbg = float("nan")
                    # 基座中心 → 物体中心的实测距离: 用来直接反驳/证实
                    # "F0 就接触目标" 这类判断 (用户: 不可能第一帧就碰到).
                    _obj_p = np.asarray(trace["object_T"], dtype=np.float64)[_sel][:, :3, 3]
                    _dd = np.linalg.norm(
                        np.asarray(trace["base_actual"], dtype=np.float64)[_sel] - _obj_p, axis=1
                    ) * 1000.0
                    _prof[_name] = {
                        "n": int(_sel.sum()),
                        "pos_mean_mm": float(_dp.mean()),
                        "pos_max_mm": float(_dp.max()),
                        "rot_mean_deg": float(np.rad2deg(_dr).mean()) if _dr.size else float("nan"),
                        "rot_max_deg": float(np.rad2deg(_dr).max()) if _dr.size else float("nan"),
                        # ★ 保留旧的跨坐标系口径值, 仅作参考 (它不是跟踪误差 ✗)
                        "rot_mean_crossref_deg": _prof_dbg,
                        "dist_to_obj_mean_mm": float(_dd.mean()) if _dd.size else float("nan"),
                        "dist_to_obj_min_mm": float(_dd.min()) if _dd.size else float("nan"),
                    }
                    # 手指实际开合: F50 前必须是 q_open(0.05) 全开. 若实测≈0,
                    # 两个手指碰撞盒只剩 ±2.5cm 通道, 物体(半宽 4.9cm)必然被撞.
                    _qact = np.asarray(trace["gripper_q_actual"], dtype=np.float64)[_sel]
                    _qcmd = np.asarray(trace["gripper_q_cmd"], dtype=np.float64)[_sel]
                    if _qact.size:
                        _prof[_name]["q_act_mean"] = float(_qact.mean())
                        _prof[_name]["q_act_min"] = float(_qact.min())
                        _prof[_name]["q_act_max"] = float(_qact.max())
                        _prof[_name]["q_cmd_mean"] = float(_qcmd.mean()) if _qcmd.size else float("nan")
            # 命令姿态的欧拉角跳变检测 (2026-09-11, 诊断方向):
            # rotmat_to_zyx_euler 返回的 rz 在 ±pi 处会包裹. 若相邻控制点命令 rz
            # 跳变接近 360°, PD 会驱动基座绕远路 → 姿态跟踪误差爆炸(实测 589deg/s).
            if len(_br):
                _raw = np.array([rotmat_to_zyx_euler(_R) for _R in _br], dtype=np.float64)
                _d_raw = np.abs(np.diff(_raw, axis=0)) if len(_raw) > 1 else np.zeros((1, 3))
                _appl = np.asarray(trace["base_cmd_euler"], dtype=np.float64)
                _d_appl = np.abs(np.diff(_appl, axis=0)) if len(_appl) > 1 else np.zeros((1, 3))
                metrics["max_cmd_euler_step_raw_deg"] = float(np.rad2deg(_d_raw.max()))
                metrics["max_cmd_euler_step_applied_deg"] = float(np.rad2deg(_d_appl.max()))
                logger.info(
                    "[Stage2 cmd] 命令 rz 相邻控制点跳变: 原始分解 %.2f deg → 全局解包裹后 %.2f deg "
                    "(应用值范围 rz[%.1f,%.1f] ry[%.1f,%.1f] rx[%.1f,%.1f] deg)",
                    float(np.rad2deg(_d_raw[:, 0].max())),
                    float(np.rad2deg(_d_appl[:, 0].max())),
                    float(np.rad2deg(_appl[:, 0].min())), float(np.rad2deg(_appl[:, 0].max())),
                    float(np.rad2deg(_appl[:, 1].min())), float(np.rad2deg(_appl[:, 1].max())),
                    float(np.rad2deg(_appl[:, 2].min())), float(np.rad2deg(_appl[:, 2].max())),
                )
            metrics["tracking_by_phase"] = _prof
            for _name, _v in _prof.items():
                logger.info(
                    "[Stage2 track] %-16s n=%3d pos=%6.2f/%6.2fmm rot=%6.2f/%6.2fdeg "
                    "dist2obj=%6.1f/%6.1fmm finger_q=%s",
                    _name, int(_v["n"]), _v["pos_mean_mm"], _v["pos_max_mm"],
                    _v["rot_mean_deg"], _v["rot_max_deg"],
                    _v["dist_to_obj_mean_mm"], _v["dist_to_obj_min_mm"],
                    (
                        f"cmd{_v['q_cmd_mean']:.4f}/act{_v['q_act_mean']:.4f}"
                        f"[{_v['q_act_min']:.4f},{_v['q_act_max']:.4f}]"
                        if "q_act_mean" in _v else "n/a"
                    ),
                )
        except Exception as _exc:  # 诊断失败不影响门禁
            logger.warning("[Stage2 track] 分阶段剖析失败: %s", _exc)

        # ★★★ 2026-09-14 用户: "stage1 2 3 都要逐帧 csv 记录" ★★★
        #   列定义与 Stage1 (gripper_trace.csv) 保持一致 ✓, 额外加 phase/bilateral
        #   ⇒ 三段可拼成一张表对比 (位姿 / 开合 / 物体 / 接触 一次看全 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T08-C2c 修 (2026-09-27) 【Stage2 CSV 指缝列缺名崩溃】 ✓ ★★★★
        #   实测 ✗ (s37): `[Stage2 CSV] 写出失败: name 'FINGER_GAP0' is not defined`
        #     ⇒ 本模块 4757 行的 f-string 直接用 `FINGER_GAP0` ✗, 而**本模块从未导入它** ✗
        #       (对比 ✓: stage3_optimize_mixin.py:152 用局部 import 解决 ⇒ 同口径修 ✓)
        #   ⇒ 本改 ✓: 写出前**补进本模块 globals** ⇒ 指缝列可正常落盘 ✓
        #     (定义处 physics_utils.FINGER_GAP0 = 0.0304 ✓; 取不到才用同值兜底 ✓)
        #   ⚠ 只补名字 ✓: 不改任何数值/判据/列定义 ✓
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        try:
            if "FINGER_GAP0" not in globals():
                try:
                    from physics_utils import FINGER_GAP0 as _FG0_c2c
                except Exception:  # noqa: BLE001
                    _FG0_c2c = 0.0304
                globals()["FINGER_GAP0"] = float(_FG0_c2c)
                logger.warning("[Stage2 CSV] ✓ 补入 FINGER_GAP0=%.4f (本模块此前缺失 ✗ ⇒ 写出必崩 ✓)",
                               float(_FG0_c2c))
        except Exception as _efg:  # noqa: BLE001
            logger.warning("[Stage2 CSV] FINGER_GAP0 兜底失败: %s", _efg)
        try:
            _s2_od = getattr(self, "output_dir", None) or os.environ.get("_OUTPUT_DIR")
            _ba = trace.get("base_actual")
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【计划 §4.5 修复】ndarray 的**真值判断**歧义 ✗ ★★★★
            #   计划实测 ✓: "local1 日志确实记录了 `The truth value of an array ... is
            #     ambiguous`, 然后捕获异常**继续**" ⇒ **Stage2 CSV 根本没写出来** ✗
            #     ⇒ 最需要逐帧比较时, 反而**没有逐帧数据** ✗✗
            #   本改 ✓: 显式 `None` + `size > 0` 判断 ✓ (绝不对多元素 ndarray 用真假性 ✗)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _ba_arr = None if _ba is None else np.asarray(_ba)
            if _s2_od and (_ba_arr is not None and _ba_arr.size > 0):
                _s2_dir = os.path.join(str(_s2_od), "stage2")
                os.makedirs(_s2_dir, exist_ok=True)
                _s2_csv = os.path.join(_s2_dir, "gripper_trace.csv")
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ 2026-09-22 【计划 §4.5 修复 · 真正的根因】✗ ★★★★
                #   计划原话 ✓: "后续多处 `trace.get(...) or []` **对多元素 ndarray 做布尔判断**" ✓
                #   ⇒ `x or []` 会先算 `bool(x)` ⇒ 堆叠 ndarray (N,4,4) **立刻抛歧义** ✗
                #     ⇒ 整个 CSV 块被 except 吞掉 ⇒ **逐帧 trace 一直写不出来** ✗✗
                #     (实测 abcd2: `[Stage2 CSV] 写出失败: The truth value of an array ...` ×3 ✓)
                #   本改 ✓: 统一用 `_as_seq()` 归一化 ⇒ list of rows ✓ (绝不做真假性判断 ✗)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                def _as_seq(_v):
                    if _v is None:
                        return []
                    try:
                        _a = np.asarray(_v)
                        return list(_a) if _a.ndim > 0 else [_a]
                    except Exception:  # noqa: BLE001
                        try:
                            return list(_v)
                        except Exception:  # noqa: BLE001
                            return []

                _gt = _as_seq(trace.get("gripper_T"))
                _ph = _as_seq(trace.get("phase"))
                _qa = _as_seq(trace.get("gripper_q_actual"))
                _bil = _as_seq(trace.get("target_bilateral"))
                _obj = _as_seq(trace.get("object_T"))
                _force = _as_seq(trace.get("grasp_force"))
                _tgo = _as_seq(trace.get("T_G_O"))
                _ba = _as_seq(_ba)
                _up2 = 0
                with open(_s2_csv, "w", encoding="utf-8") as _fh2:
                    _fh2.write(
                        "frame,phase,pos_x,pos_y,pos_z,euler_rx_deg,euler_ry_deg,"
                        "euler_rz_deg,x_down_dot,finger1_q,finger2_q,finger_gap_m,"
                        "obj_z,bilateral,grasp_force_n,tgo_translation_mm,"
                        "n_f1,n_f2,t_f1,t_f2,palm_n,ground_n,other_n,"
                        "period_normal_force_f1,period_normal_force_f2,"
                        "last_substep_instant_f1,last_substep_instant_f2,"
                        "load_probe_state\n")
                    for _i in range(len(_ba)):
                        _T2 = np.asarray(_gt[_i], dtype=np.float64) if _i < len(_gt) \
                            else np.full((4, 4), np.nan)
                        _R2 = _T2[:3, :3] if np.isfinite(_T2).all() else np.eye(3)
                        _e2z, _e2y, _e2x = rotmat_to_zyx_euler(_R2)
                        _d2 = float(np.dot(_R2[:, 0], np.array([0.0, 0.0, -1.0])))
                        if _d2 < 0.0:
                            _up2 += 1
                        _q2 = np.asarray(_qa[_i], dtype=np.float64) if _i < len(_qa) \
                            else np.zeros(2)
                        _oz = (float(np.asarray(_obj[_i])[2, 3]) if _i < len(_obj)
                               else float("nan"))
                        if _i < len(_force) and np.asarray(_force[_i]).size:
                            _force_text = f"{float(np.asarray(_force[_i]).reshape(-1)[0]):.6f}"
                        else:
                            _force_text = "nan"
                        _tgo_text = "nan"
                        if (_i < len(_tgo) and tgo_reference is not None
                                and np.asarray(_tgo[_i]).shape == (4, 4)
                                and np.isfinite(np.asarray(_tgo[_i])).all()
                                and np.isfinite(np.asarray(tgo_reference)).all()):
                            _tgo_delta = np.asarray(_tgo[_i], dtype=np.float64)[:3, 3] \
                                - np.asarray(tgo_reference, dtype=np.float64)[:3, 3]
                            _tgo_text = f"{float(np.linalg.norm(_tgo_delta) * 1000.0):.6f}"
                        _fh2.write(",".join(str(_v) for _v in (
                            _i, str(_ph[_i]) if _i < len(_ph) else "?",
                            *[f"{float(_v):.6f}" for _v in np.asarray(_ba[_i])],
                            f"{np.rad2deg(_e2x):.6f}", f"{np.rad2deg(_e2y):.6f}",
                            f"{np.rad2deg(_e2z):.6f}", f"{_d2:.6f}",
                            f"{float(_q2[0]):.6f}", f"{float(_q2[1]):.6f}",
                            f"{FINGER_GAP0 + 2.0 * float(_q2[0]):.6f}", f"{_oz:.6f}",
                            str(bool(_bil[_i])) if _i < len(_bil) else "?",
                            _force_text,
                            _tgo_text,
                        )) + "\n")
                logger.info(
                    "[Stage2 CSV] 逐帧位姿+开合已存: %s (%d 帧) | X 朝上(dot<0) 帧数=%d %s",
                    _s2_csv, len(_ba), _up2,
                    "← 有帧夹爪朝上 ✗" if _up2 else "✓ 全程朝下")
        except Exception as _e2:  # noqa: BLE001
            logger.warning("[Stage2 CSV] 写出失败: %s", _e2)

        # ★★★★ 2026-09-17 夹空判据 (用户: "夹空这个确实需要改正一下") ★★★★
        #   背景 (CoACD 实测 output/coacd_s2): 门禁全程绿灯 ✗ 而实际是**夹空**:
        #     · CARRY 手指 act=[0.0057,0.0097] ✗ 几乎等于命令 0.0047 ⇒ **没被物体挡停** ⇒ 手里空的
        #     · CARRY dist2obj=101.6mm ✗ (box 版 37.7mm)  · max_pen=0.0000mm ✗ (只擦到)
        #     · 但 dist2obj 测的是"夹爪基座↔物体中心" ✗ ⇒ 分不清"在两指之间"和"在前方 35mm" ✗
        #   判据: 携带段里, 手指**实际开合**必须明显大于**命令** (被物体挡停产生的闭合差)
        #     闭合差 = max(act - cmd) over CARRY 段
        #     ⇒ 夹住 ⇒ 闭合差 ≈ 6~10mm ✓ ; 夹空 ⇒ 闭合差 ≈ 0 ✗
        try:
            _gqa = np.asarray(trace.get("gripper_q_actual", []), dtype=np.float64)
            _gqc = np.asarray(trace.get("gripper_q_cmd", []), dtype=np.float64)
            _pha = np.asarray(trace.get("phase", []), dtype="U24")
            _carry_m = _pha == "CARRY"
            if _carry_m.any() and _gqa.size == _gqc.size and _gqa.shape == _gqc.shape:
                _closure = float(np.max(_gqa[_carry_m] - _gqc[_carry_m]))
                _min_needed = float(os.environ.get("_MIN_CARRY_CLOSURE_M", "0.002"))
                metrics["carry_closure_mm"] = _closure * 1000.0
                logger.warning(
                    "[Stage2 夹持校验] 携带段手指闭合差=%.2fmm (阈值 %.2fmm) ⇒ %s",
                    _closure * 1000.0, _min_needed * 1000.0,
                    "真夹住 ✓" if _closure >= _min_needed else "**夹空** ✗",
                )
                if _closure < _min_needed and gate_result.get("passed", False):
                    gate_result.update({
                        "passed": False,
                        "failure": "EMPTY_GRASP",
                        "failure_detail": (
                            f"携带段手指未被物体挡停 (闭合差={_closure * 1000.0:.2f}mm "
                            f"< {_min_needed * 1000.0:.2f}mm) ⇒ **夹空**, 物体不在两指之间"
                        ),
                    })
                    logger.error("[Stage2 gate] EMPTY_GRASP: %s", gate_result["failure_detail"])
        except Exception as _egc:  # noqa: BLE001
            logger.warning("[Stage2 夹持校验] 采集失败: %s", _egc)

        # 夹爪接触明细 (2026-09-11): 定位"到底跟什么碰、穿多深、第几个控制点开始"
        if contact_detail:
            logger.warning("[Stage2 contact] 夹爪接触明细 (link x body):")
            for (_ln, _on), _rec in sorted(
                contact_detail.items(), key=lambda kv: kv[1]["first_ctrl"]
            ):
                logger.warning(
                    "[Stage2 contact]   %-32s x %-26s n=%3d first_ctrl=%3d max_pen=%.4fmm",
                    _ln, _on, int(_rec["n"]), int(_rec["first_ctrl"]),
                    float(_rec["max_depth_mm"]),
                )
        else:
            logger.info("[Stage2 contact] 全程无夹爪相关接触 (contact_detail 为空)")
        metrics["contact_detail"] = {
            f"{k[0]}|{k[1]}": v for k, v in contact_detail.items()
        }

        gate_result["metrics"] = metrics
        logger.info(
            "[Stage2 gate] %s bundle=%s controls=%d tracking=%.2f/%.2fmm",
            "PASS" if gate_result["passed"] else f"FAIL({gate_result['failure']})",
            bundle["bundle_hash"][:12],
            len(trace["control_index"]),
            tracking["mean_mm"],
            tracking["max_mm"],
        )
        return {
            "bundle": bundle,
            "trace": trace,
            "gate_result": gate_result,
            "tracking_error": tracking,
            "carry_validation": {
                "passed": bool(gate_result["passed"]),
                "max_dev_mm": metrics.get("tgo_translation_max_mm", float("nan")),
            },
            "obj_lift": (
                float(np.max(trace["object_T"][:, 2, 3]) - initial_object_T[2, 3])
                if len(trace["object_T"])
                else 0.0
            ),
            "obj_lift_final": (
                float(trace["object_T"][-1, 2, 3] - initial_object_T[2, 3])
                if len(trace["object_T"])
                else 0.0
            ),
            "obj_xy_drift": (
                float(np.linalg.norm(trace["object_T"][-1, :2, 3] - initial_object_T[:2, 3]))
                if len(trace["object_T"])
                else 0.0
            ),
        }

    def verify_stage1_transfer_equivalence(self, side, grasp_pose, n_repeat=3, record_video=False):
        """T1 ``F50_TRANSFER_EQUIVALENCE`` (plan §3.1).

        Reset the shared scene to the canonical Stage2 entry state — base at the
        Stage1 ``pos/R``, commanded and actual ``q = q_open`` (0.05), zero
        velocity, no target contact — then execute the shared 20+10 Stage1 grasp
        recipe.  There is deliberately no hidden ``_DESCEND_OPEN`` warmup.  The
        anchor is declared transferable when at least 2 of 3 independent resets
        establish *and* hold bilateral target contact through the settle block.
        """
        import sapien
        from data_loader import rotmat_to_zyx_euler
        from physics_env import (
            GRIPPER_DAMPING,
            GRIPPER_FORCE,
            GRIPPER_STIFFNESS,
            physics_step,
        )
        from physics_utils import GRIPPER_MAX_OPEN

        recipe = grasp_pose.get("stage1_recipe")
        if not recipe:
            return {
                "passed": False,
                "failure": "STALE_STAGE1_ARTIFACT",
                "failure_detail": "grasp_pose has no stage1_grasp_recipe",
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
            }
        q_open = float(recipe["q_open"])
        q_grasp = float(recipe["q_grasp"])
        settle_controls = int(recipe.get("settle_controls", 0) or 0)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ T08-A (2026-09-27) 【reuse 核对改口径】**不再重生成曲线** ✗ ★★★★
        #   md ✓: "两者均不得再由 q 终点调用 `stage1_close_q_sequence` 生成另一条'等价'曲线" ✗
        #     旧实现现场重生成 + 比对 ✗ ⇒ 反馈控制产生的**不同 q/时长**必被判 MISMATCH ✗
        #   ⇒ 本改 ✓: 与 **实际记录**(phase_boundaries 切出的 close 段)核对 ✓;
        #     并把**逐项 expected/observed/match** 写进报告 (md: 不只比较名义 q 数组 ✓)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _mode_rep = str(os.environ.get("_S2_MODE", "feedback"))
        _tr_rep = split_trace_by_phase(recipe)
        _close_act = list((_tr_rep.get("close") or {}).get("q") or [])
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-28 修 `--test-stage1-transfer` **必然 FAIL** 的逻辑 bug ✓ ★★★★
        #   实测 ✗: `reuse_report_items(recipe, observed=None, ...)` ⇒ `_obs = observed or {} = {}`
        #     ⇒ 每一项 `observed=None` ⇒ `n_missing=11` ⇒ **恒 FAIL(STAGE1_REUSE_MISMATCH)** ✗
        #     ⇒ 该回放入口自投入使用起就**从未通过过** (不是"跑得慢", 是死路 ✗),
        #       连带"成功录像"永远拿不到 ✗
        #   本改 ✓: 用**现场按导出同一算法重算**的身份量作 observed ——
        #     physics/controller/criteria 三项 hash 与 `_export_stage1_recipe` 逐字同构
        #     ⇒ 真做"执行器/物理/判据**身份核对**" ✓ (不是自比 ✗);
        #     `close/settle` 长度取**实际 trace** ✓; `mode` 取现场模式 ✓
        #   ⚠ 诚实标注 ✓: 若现场 env 与导出时不同 ⇒ hash 不同 ⇒ **仍应 FAIL** ✓ (这正是要检出的 ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _obs_rep = None
        try:
            from stage1_recipe import (
                STAGE1_RECIPE_SCHEMA as _SCHEMA_rep,
                build_physics_config as _bpc_rep,
                hash_resolved_profile as _hrp_rep,
            )
            import physics_utils as _pu_rep
            _ctl_prof_rep = {}
            for _k_rep in ("FINGER_PD_KP", "FINGER_PD_KD", "FINGER_PD_FORCE_LIMIT",
                           "GRIPPER_FORCE", "GRIPPER_STIFFNESS", "GRIPPER_DAMPING",
                           "GRIPPER_MAX_OPEN", "FINGER_GAP0", "PHYSICS_TIMESTEP"):
                if hasattr(_pu_rep, _k_rep):
                    _ctl_prof_rep[_k_rep] = getattr(_pu_rep, _k_rep)
            _side_rep = str(getattr(self, "side", None)
                            or os.environ.get("_TC_SIDE", "left") or "left")
            _obs_rep = {
                "schema_version": _SCHEMA_rep,
                "physics_hash": _hrp_rep(_bpc_rep()),
                "controller_hash": _hrp_rep({
                    "profile": _ctl_prof_rep, "side": _side_rep,
                    "close_profile": str(recipe.get("close_profile", "")),
                    "close_controls": int(recipe.get("close_controls", -1)),
                    "settle_controls": int(recipe.get("settle_controls", -1))}),
                "criteria_hash": _hrp_rep({
                    "criteria_revision": str(recipe.get("criteria_revision", "criteria_v2")),
                    "gate": {_k: os.environ.get(_k) for _k in (
                        "_S1_GATE_LIFT_CM", "_S1_GATE_FGND_MM", "_S1_GATE_TG_RATIO",
                        "_S1_FROZEN_Q_TOL_MM", "_S1_CLOSE_OVERCLOSURE_MM",
                        "_S1_PRELOAD_M", "_S1_T8_EXTRA_MM")},
                    "task_grasp_frame": int(recipe.get("task_grasp_frame", 50))}),
                # anchor 身份: 复用就是复用同一个 anchor ⇒ 与 recipe 记录应逐位相同 ✓
                "optimized_T_W_G": recipe.get("optimized_T_W_G"),
                "source_frame": recipe.get("source_frame"),
                "task_grasp_frame": recipe.get("task_grasp_frame"),
                "q_grasp": float(q_grasp),
                "close_controls_actual": int(len(_close_act)),
                "settle_controls_actual": int(len((_tr_rep.get("settle") or {}).get("q") or [])),
                "mode": _mode_rep,
            }
        except Exception as _eobs:  # noqa: BLE001
            logger.warning("[T08-A][reuse] 现场身份量构造失败 ⇒ 回落到 recipe 自洽核对 ✓: %s", _eobs)
            _obs_rep = None
        _items_rep = reuse_report_items(recipe, observed=_obs_rep, mode=_mode_rep)
        logger.warning("[T08-A][reuse] 逐项核对: %s | missing=%d mismatch=%d",
                       [_i["field"] for _i in _items_rep["items"]],
                       int(_items_rep["n_missing"]), int(_items_rep["n_mismatch"]))
        if not _close_act:
            return {
                "passed": False,
                "failure": "STAGE1_REUSE_MISMATCH",
                "failure_detail": "recipe 缺少**实际 close 段** (md: 不得重生成'等价'曲线 ✗)",
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
                "reuse_items": _items_rep["items"],
            }
        close_controls = int(len(_close_act))          # ★ 时长以**实际记录**为准 ✓
        stored_q = np.asarray(recipe.get("close_ctrl_q", []), dtype=np.float64)
        # ★ 2026-09-28 诊断 ✓: 运行时直接打双方长度与逐位差, 立刻定位 mismatch 来源 ✓
        try:
            _arr_ca = np.asarray(_close_act, dtype=np.float64)
            _dmax = (float(np.max(np.abs(stored_q - _arr_ca)))
                     if (stored_q.size and _arr_ca.size and stored_q.shape == _arr_ca.shape)
                     else float("nan"))
            logger.warning("[T08-A][诊断] stored_q.shape=%s close_act.shape=%s "
                           "max_abs_diff=%s recipe_keys_has_actual_q=%s",
                           stored_q.shape, _arr_ca.shape, _dmax,
                           "actual_q_commands" in recipe)
        except Exception as _edbg:  # noqa: BLE001
            logger.warning("[T08-A][诊断] 打印失败: %s", _edbg)
        if stored_q.size and (stored_q.shape[0] != close_controls
                              or not np.allclose(stored_q, np.asarray(_close_act), atol=1e-12)):
            return {
                "passed": False,
                "failure": "STAGE1_REUSE_MISMATCH",
                "failure_detail": "close_ctrl_q 与**实际执行段**不一致 ✗ (非『重生成』比对 ✓)",
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
                "reuse_items": _items_rep["items"],
            }
        _m_ok_r, _m_pay_r, _m_errs_r = select_close_mode(recipe, _mode_rep)
        if _m_errs_r:
            return {
                "passed": False,
                "failure": "STAGE1_REUSE_MISMATCH",
                "failure_detail": "模式分流/入口合同失败: %s" % (_m_errs_r[:2],),
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
                "reuse_items": _items_rep["items"],
            }
        if not np.isclose(float(grasp_pose["gripper_qpos"]), q_grasp, atol=1e-12):
            return {
                "passed": False,
                "failure": "STAGE1_REUSE_MISMATCH",
                "failure_detail": "grasp_pose.gripper_qpos != recipe.q_grasp",
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
            }

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        target_actor = self._find_target_actor(self.obj_actors, target_name)
        if target_actor is None:
            return {
                "passed": False,
                "failure": "DATA",
                "failure_detail": f"target actor {target_name!r} not found",
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
            }

        robot = self.robot_info["robot"]
        virtual_idx = self.robot_info.get("virtual_idx", {})
        if not virtual_idx:
            return {
                "passed": False,
                "failure": "REACH",
                "failure_detail": "transfer equivalence requires gripper_only virtual joints",
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
            }
        gi1 = self.robot_info["gripper_idx1"]
        gi2 = self.robot_info["gripper_idx2"]
        arm_indices = self.robot_info.get("arm_joint_indices", [])
        active_joints = robot.get_active_joints()
        collision_scope = self._configure_stage1_target_isolated_collisions(
            robot, side, self.obj_actors, target_name,
            # ★ 任务02 ✓: 必须传 scene ⇒ 才能把 **地面/桌面** 的组也设对
            #   (否则 ground 保持 (1,1) ⇒ 目标—桌面/指尖—桌面 恒 False ✗)
            scene=getattr(self, "scene", None),
        )
        if not hasattr(self, "_obj_initial_poses") or self._obj_initial_poses is None:
            self._obj_initial_poses = {a.get_name(): a.get_pose() for a in self.obj_actors}

        anchor_pos = np.asarray(grasp_pose["pos"], dtype=np.float64)
        anchor_R = np.asarray(grasp_pose["R"], dtype=np.float64)
        rz, ry, rx = rotmat_to_zyx_euler(anchor_R)
        base_targets = {
            "vx": float(anchor_pos[0]),
            "vy": float(anchor_pos[1]),
            "vz": float(anchor_pos[2]),
            "rz": float(rz),
            "ry": float(ry),
            "rx": float(rx),
        }
        for joint_idx in (gi1, gi2):
            active_joints[joint_idx].set_drive_property(
                stiffness=GRIPPER_STIFFNESS, damping=GRIPPER_DAMPING, force_limit=GRIPPER_FORCE
            )

        runs = []
        for repeat in range(int(n_repeat)):
            robot.set_qpos(self.robot_info["init_qpos"].copy())
            robot.set_qvel(np.zeros_like(robot.get_qvel()))
            robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
            robot.set_root_linear_velocity([0, 0, 0])
            robot.set_root_angular_velocity([0, 0, 0])
            for actor in self.obj_actors:
                initial_pose = self._obj_initial_poses.get(actor.get_name())
                if initial_pose is None:
                    continue
                actor.set_pose(initial_pose)
                for component in actor.get_components():
                    if hasattr(component, "set_linear_velocity"):
                        component.set_linear_velocity([0, 0, 0])
                        component.set_angular_velocity([0, 0, 0])
                        break
            # ★ T08-D2 ✓: reset 之后**按冻结快照恢复** ⇒ 与规划输入同源 (每轮 repeat 都做 ✓)
            try:
                self.restore_scene_snapshot(robot=robot)
                _chk_d2 = self.assert_scene_snapshot(target_name) \
                    if hasattr(self, "assert_scene_snapshot") else {}
                logger.warning("[T08-D2][执行起点核验] Δpos=%.2fmm Δrot=%.2f° ⇒ %s",
                               float(_chk_d2.get("max_pos_mm") or 0.0),
                               float(_chk_d2.get("max_rot_deg") or 0.0),
                               "✓ 同源" if _chk_d2.get("ok") else "✗ 仍不一致(见上一行原因 ✓)")
            except Exception as _ed2b:  # noqa: BLE001
                logger.warning("[T08-D2] 执行起点核验失败: %s", _ed2b)
            reset_object_T = np.asarray(
                target_actor.get_pose().to_transformation_matrix(), dtype=np.float64
            )
            logger.info(
                "[Stage1 transfer] repeat %d target_name=%r actor=%r "
                "reset_obj=%s obj_info_bbox=%s all_objects=%s",
                repeat + 1,
                target_name,
                target_actor.get_name(),
                np.round(reset_object_T[:3, 3], 4).tolist(),
                getattr(self, "obj_info", {}).get(target_name, {}).get("bbox_size"),
                {
                    a.get_name(): np.round(np.asarray(a.get_pose().p), 3).tolist()
                    for a in self.obj_actors
                },
            )

            qpos = robot.get_qpos().copy()
            for key, value in base_targets.items():
                qpos[virtual_idx[key]] = value
            qpos[gi1] = q_open
            qpos[gi2] = q_open
            robot.set_qpos(qpos)
            robot.set_qvel(np.zeros_like(robot.get_qvel()))
            for _w in range(5):
                for key, value in base_targets.items():
                    active_joints[virtual_idx[key]].set_drive_target(value)
                active_joints[gi1].set_drive_target(q_open)
                active_joints[gi2].set_drive_target(q_open)
                physics_step(
                    robot, arm_indices, gi1, gi2, np.array([]), q_open, q_open, self.scene
                )
                logger.info(
                    "[Stage1 transfer]   warmup %d obj=%s base=%s q=%s",
                    _w,
                    np.round(np.asarray(target_actor.get_pose().p), 4).tolist(),
                    np.round(
                        np.asarray(robot.get_qpos())[
                            [virtual_idx["vx"], virtual_idx["vy"], virtual_idx["vz"]]
                        ],
                        4,
                    ).tolist(),
                    np.round(np.asarray(robot.get_qpos())[[gi1, gi2]], 4).tolist(),
                )

            entry_audit = self._audit_contact_pairs(
                robot, side, self.scene, self.obj_actors, target_name
            )
            entry_target_contact = entry_audit["target_contact"]
            entry_object_pos = np.asarray(target_actor.get_pose().p, dtype=np.float64).copy()
            entry_base_actual = np.asarray(robot.get_qpos(), dtype=np.float64)[
                [virtual_idx["vx"], virtual_idx["vy"], virtual_idx["vz"]]
            ].copy()
            logger.info(
                "[Stage1 transfer] repeat %d entry: base_actual=%s object=%s "
                "finger_q=%s ground_z=%.4f",
                repeat + 1,
                np.round(entry_base_actual, 4).tolist(),
                np.round(entry_object_pos, 4).tolist(),
                np.round(np.asarray(robot.get_qpos())[[gi1, gi2]], 4).tolist(),
                float(getattr(self, "_ground_z", 0.0)),
            )

            bilateral_flags = []
            object_drift = 0.0
            contacts = []
            object_trace = []
            for k in range(close_controls + settle_controls):
                # ★ 2026-09-28 修 `NameError: close_q` ✗ (回放入口第 2 个未定义名 ⇒ 长期跑不通 ✓)
                #   本函数内的实际 close 段是 `_close_act` (上面由 split_trace_by_phase 切出 ✓),
                #   而 `close_controls = len(_close_act)` ✓ ⇒ 直接用实际段 ✓ (不再重生成 ✗)
                q_cmd = float(_close_act[k]) if k < close_controls else q_grasp
                for key, value in base_targets.items():
                    active_joints[virtual_idx[key]].set_drive_target(value)
                active_joints[gi1].set_drive_target(q_cmd)
                active_joints[gi2].set_drive_target(q_cmd)
                physics_step(
                    robot, arm_indices, gi1, gi2, np.array([]), q_cmd, q_cmd, self.scene
                )
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                # ★★★★ B1 修复 (2026-09-28 晚) 【verify 接录像 ⇒ 拿 Stage1 成功 GIF】 ✓ ★★★★
                #   实测 ✗: `verify_stage1_transfer_equivalence(..., record_video=True)` 收到
                #     参数后**函数体内从不使用** ⇒ Stage1 3/3 PASS 也**一帧不录** ✗
                #     (这正是"拿不到成功 GIF"的唯一直接原因 ✓ 见 md 第八节 B1 ✓)
                #   本改 ✓: 用**轻量伪 bundle** 复用 `_record_stage2_frame` 的渲染核心
                #     (零复制 ✓); 每控制点一帧 ✓, close/settle 标签区分 ✓
                #   ⚠ warmup 段不录 (真抓取段才录 ✓); 帧落在 stage2/ 子目录 (writer 归属 ✓)
                # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
                if record_video:
                    _n_all_v = int(close_controls) + int(settle_controls)
                    self._record_stage2_frame(
                        {"source_frame": [float(recipe.get("source_frame", 0) or 0)],
                         "phase": ["VERIFY q=%.3f %d/%d %s" % (
                             float(q_cmd), k + 1, _n_all_v,
                             "CLOSE" if k < close_controls else "SETTLE")]},
                        0)
                f1, f2, contact_objects, _links = self._target_finger_contacts(
                    robot, side, self.scene, self.obj_actors, target_name
                )
                target_in = any(
                    name == target_name or (target_name and target_name in name)
                    for name in contact_objects
                )
                bilateral_flags.append(bool(f1 and f2 and target_in))
                obj_p = np.asarray(target_actor.get_pose().p, dtype=np.float64).copy()
                object_trace.append(obj_p)
                if k in (0, 1, 3, 6, 10, 15, 19, 20, 24, 29):
                    logger.info(
                        "[Stage1 transfer]   k=%02d q_cmd=%.4f q_act=%s f1=%s f2=%s "
                        "obj=%s contacts=%s",
                        k,
                        q_cmd,
                        np.round(np.asarray(robot.get_qpos())[[gi1, gi2]], 4).tolist(),
                        f1,
                        f2,
                        np.round(obj_p, 4).tolist(),
                        sorted(contact_objects),
                    )
                if k == close_controls + settle_controls - 1:
                    object_drift = float(
                        np.linalg.norm(
                            np.asarray(target_actor.get_pose().p, dtype=np.float64)
                            - reset_object_T[:3, 3]
                        )
                    )
                    contacts = sorted(contact_objects)

            settle = bilateral_flags[close_controls:]
            settle_pass = sum(settle) >= settle_controls - 2 and all(settle[-3:])
            # Plan §4 T1 criteria: canonical entry, no hidden warmup, no target
            # contact at the entry, shared 20+10, >=2/3 establish AND hold
            # bilateral target contact.  Pre-grasp object disturbance is recorded
            # (and judged later by Stage2's PROVISIONAL_GRASP gate), not a T1
            # pass criterion.
            run_pass = bool(not entry_target_contact and settle_pass)
            runs.append(
                {
                    "repeat": repeat,
                    "entry_target_contact": bool(entry_target_contact),
                    "entry_leak_pairs": entry_audit["leak_pairs"],
                    "entry_object_pos": entry_object_pos.tolist(),
                    "entry_base_actual": entry_base_actual.tolist(),
                    "object_trace": [p.tolist() for p in object_trace],
                    "bilateral_controls": bilateral_flags,
                    "settle_bilateral": int(sum(settle)),
                    "settle_controls": int(settle_controls),
                    "object_drift_mm": object_drift * 1000.0,
                    "object_drift_ok": bool(object_drift <= PRE_GRASP_OBJECT_DRIFT_M),
                    "final_contacts": contacts,
                    "passed": run_pass,
                }
            )
            # ★ 校准诊断 (2026-09-29) ✓: 与 Stage2 的 `[06.4-3][诊断]` **同名同口径**,
            #   在 Stage1 (**成功侧**) 也打一次 ⇒ 两边数值直接对照 ⇒ 判定 Stage2 的
            #   `T_G_O≈20.9mm / q_actual≈0.006` 到底正不正常 (避免我在 Stage2 侧误读 ✗)
            try:
                _q_now = np.asarray(robot.get_qpos(), dtype=np.float64)[[gi1, gi2]]
                _p_obj_now = np.asarray(target_actor.get_pose().p, dtype=np.float64)
                _p_base_now = np.asarray(robot.get_qpos(), dtype=np.float64)[
                    [virtual_idx["vx"], virtual_idx["vy"], virtual_idx["vz"]]]
                logger.warning(
                    "[06.4-3][校准·Stage1] settle末: q_actual=%s | 指尖间距≈%.1fmm | "
                    "物体p=%s | 手基座p=%s | 手-物平移=%s mm | 双侧=%d/%d",
                    np.round(_q_now, 4).tolist(), float(np.mean(_q_now)) * 2.0 * 1000.0,
                    np.round(_p_obj_now, 4).tolist(), np.round(_p_base_now, 4).tolist(),
                    np.round((_p_base_now - _p_obj_now) * 1000.0, 2).tolist(),
                    int(sum(settle)), int(settle_controls))
            except Exception as _ecal:  # noqa: BLE001
                logger.warning("[06.4-3][校准·Stage1] 采集失败: %s", _ecal)
            logger.info(
                "[Stage1 transfer] repeat %d/%d: %s entry_target_contact=%s "
                "settle_bilateral=%d/%d drift=%.2fmm contacts=%s",
                repeat + 1,
                int(n_repeat),
                "PASS" if run_pass else "FAIL",
                entry_target_contact,
                int(sum(settle)),
                settle_controls,
                object_drift * 1000.0,
                contacts,
            )

        n_pass = sum(1 for run in runs if run["passed"])
        passed = n_pass >= 2
        result = {
            "passed": passed,
            "failure": None if passed else "F50_TRANSFER_EQUIVALENCE",
            "failure_detail": "" if passed else f"{n_pass}/{int(n_repeat)} resets passed",
            "n_pass": int(n_pass),
            "n_repeat": int(n_repeat),
            "q_open": q_open,
            "q_grasp": q_grasp,
            "close_controls": close_controls,
            "settle_controls": settle_controls,
            "collision_scope": collision_scope,
            "scene_digest_hash": str(recipe.get("scene_digest_hash", "")),
            "runs": runs,
        }
        logger.info(
            "[Stage1 transfer] F50_TRANSFER_EQUIVALENCE: %d/%d → %s",
            n_pass,
            int(n_repeat),
            "PASS" if passed else "FAIL",
        )
        return result

    def _open_video_writers(self, stage_tag: str) -> bool:
        """★ 2026-09-23 【按阶段分割录像】把写入器**轮换**到 `output/<run>/<stage_tag>/` ✓

        为什么必须做 ✓ (用户 2026-09-23 要求 ✓): "**先把 stage1 和 stage2 分割开**, 你不要搞乱了" ✓
        病根 ✓: 写入器在**主实例 init 时一次性创建** ⇒ 路径由 `self._run_dir()` 决定,
          而 mode 固定为 `stage3` (管线都用 `--test-stage3` 入口 ✓) ⇒
          **Stage0/1/2/3 的所有帧都写进同一个 `stage3/*.mp4`** ✗
          ⇒ 既无法"按阶段出视频" ✗, 也让"阶段边界"在产物层面消失 ✗
        本改 ✓: 每个阶段开始录像前调用一次 ⇒ 关闭旧写入器 + 在**该阶段的子目录**新建 ✓
          阶段语义 (与代码一致 ✓, 不凭记忆 ✓):
            `stage0/` = 主循环: 场景加载 + MANO 预处理回放
            `stage1/` = `--test-stage1` / A组: **位姿搜索 + rollout (就位→闭合→保持→竖直抬升)**
            `stage2/` = `--test-stage2/3`: **接近 → 就位后闭合 → 复现参考轨迹(携带)** ✓
        ⚠ 旧的 `stage3/` 目录：历史产物保留, 不再写入 ✓
        """
        try:
            import cv2
            # ① 先释放旧写入器 ✓ (避免半开文件 ⇒ 坏 mp4 ✗)
            for _name in ("_writer_cam", "_writer_god", "_writer_right"):
                _w = getattr(self, _name, None)
                if _w is not None:
                    try:
                        _w.release()
                    except Exception:  # noqa: BLE001
                        pass
                    setattr(self, _name, None)
            # ② 在该阶段的子目录新建 ✓
            _sub = os.path.join(str(self._run_dir()), str(stage_tag))
            os.makedirs(_sub, exist_ok=True)
            _tag = f"_{self.video_tag}" if getattr(self, "video_tag", "") else ""
            _paths = {}
            for _view, _attr in (("cam", "_writer_cam"), ("god", "_writer_god"),
                                 ("right", "_writer_right")):
                _vp = os.path.join(_sub, f"{_view}_view{_tag}_{stage_tag}_{self.side}.mp4")
                # ⚠ 分辨率必须与主循环一致 ✓ (CAM_WIDTH/HEIGHT=1920×1080 ✓);
                #   本文件不保证导入了这两个常量 ⇒ 用参数/默认双重兜底 ✓ (避免 NameError ✗)
                _cw = int(getattr(self, "cam_width", 0) or 0) or int(
                    getattr(self, "_CAM_W", 0) or 0) or 1920
                _ch = int(getattr(self, "cam_height", 0) or 0) or int(
                    getattr(self, "_CAM_H", 0) or 0) or 1080
                _w = cv2.VideoWriter(_vp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (_cw, _ch))
                setattr(self, _attr, _w)
                setattr(self, f"_video_path_{_view}", _vp)
                _paths[_view] = _vp
            os.makedirs(os.path.dirname(_paths["cam"]), exist_ok=True)
            logger.info("[录像分阶段] ✓ 写入器已轮换到 **%s/** ⇒ 本段视频: %s",
                        stage_tag, {k: os.path.basename(v) for k, v in _paths.items()})
            return True
        except Exception as _eov:  # noqa: BLE001
            logger.warning("[录像分阶段] 轮换失败 (沿用旧写入器): %s", _eov)
            return False

    def _record_stage2_frame(self, bundle: dict, index: int) -> None:
        """Render one already-executed control point; rendering never steps physics."""
        # ★ 2026-09-23 【按阶段分割】首次调用即把写入器轮换到 **stage2/** ✓
        #   (本函数只服务"接近→闭合→复现参考轨迹"这一段 ⇒ 属 **Stage2** ✓ 不属 stage1 ✗)
        if not getattr(self, "_stage2_writers_open", False):
            self._stage2_writers_open = True
            self._open_video_writers("stage2")
        try:
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-23 【空 mp4 修复】不再**静默跳过** ✗ ★★★★
            #   实测 ✓: 请求录像时, 本函数在 `_render_available=False` 时**直接 return** ✗
            #     ⇒ mp4 里一帧都没有 ⇒ **257 / 44 字节空文件** ✗
            #     (对比 ✓: 有帧的那几次 = 1.3~2.4MB ✓; 空的那几次 = 44~257 字节 ✗)
            #   成因 ✓ (代码内既有结论 ✓): `render_available` 由建场景时决定 ✓;
            #     **并行多进程**时渲染器创建可能失败 ✗ ⇒ 该轮就是空的 ✓
            #   本改 ✓ (两步):
            #     ① `_render_available=False` **只告警, 不再 return** ✓ ⇒ 继续走真实渲染路径,
            #        由外层 try 捕获**真实**失败 ⇒ 能出帧就出帧 ✓
            #     ② 首次遇到即 WARNING ✓ ⇒ "为什么是空视频"一目了然 ✓
            #   ⚠ 用法约束 ✓: **不要并行跑多个带录像的进程** ✓ (渲染器会抢不到 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            if not getattr(self.scene, "_render_available", False):
                if not getattr(self, "_s2_render_warned", False):
                    self._s2_render_warned = True
                    logger.warning(
                        "[Stage2 录像] ⚠ `_render_available=False` ⇒ **仍尝试渲染** ✓ "
                        "(旧版在此**直接 return** ⇒ 空 mp4 ✗); 若最终失败, 请**勿并行跑录像进程** ✓")
            import cv2

            self._update_collision_visualization()
            self.scene.update_render()
            # 2026-09-12: 逐帧同步第一人称 cam 相机. 此前这里只 take_picture(),
            # 相机冻结在 init 时设的 start_frame 位姿 → cam 视频看起来像固定俯视机位.
            # 主循环 (Stage0) 一直是逐帧更新的, 这里对齐同一行为.
            self._sync_cam_view_to_frame(bundle["source_frame"][index])
            label = (
                f"src F{bundle['source_frame'][index]:.2f} | ctrl {index + 1}/{len(bundle['source_frame'])} | "
                f"{bundle['phase'][index]}"
            )
            _missing = []
            for camera_name, writer_name in (
                ("_cam_view", "_writer_cam"),
                ("_god_view", "_writer_god"),
                ("_right_view", "_writer_right"),
            ):
                camera = getattr(self, camera_name, None)
                writer = getattr(self, writer_name, None)
                if camera is None or writer is None:
                    # 2026-09-13 ★ 原先这里静默 continue → Stage2 视频一帧都没录却毫无提示,
                    # 用户只能看到"没渲染完整". 现改为**首次遇到就响亮报错** (只报一次, 避免刷屏).
                    _missing.append(f"{camera_name}/{writer_name}")
                    continue
                camera.take_picture()
                rgb = camera.get_picture("Color")[..., :3]
                bgr = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1])
                cv2.putText(
                    bgr,
                    label,
                    (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(bgr)
                # 2026-09-13: 按 index==0 重置计数 —— 与 rollout 的调用结构解耦,
                # 避免 --test-stage3 下 3 次 repeat 累加 (曾误报 348 = 3×116).
                if index == 0:
                    self._stage2_frames_written = 0
                self._stage2_frames_written = getattr(self, "_stage2_frames_written", 0) + 1
            if _missing and not getattr(self, "_stage2_render_warned", False):
                self._stage2_render_warned = True
                logger.error(
                    "[Stage2 render] ★ 视频 writer/相机不可用: %s → Stage2 视频不会产出! "
                    "请检查 writer 是否在 Stage2 rollout 之前创建 (见 grasp_hawor.py 的 "
                    "_writer_cam 初始化路径).", ", ".join(sorted(set(_missing))))
        except Exception as exc:
            logger.warning("[Stage2 render] frame %d skipped: %s", index, exc)

    def optimize_stage2_dz(
        self,
        recon,
        side,
        grasp_pose,
        F50_IDX,
        n_frames_out=None,
        dz_grid=None,
        delta_grid=None,
        wmax_grid=None,
        n_repeat=1,
    ):
        """Compatibility entry: search time scale only and return a passed result.

        The old mixed dz/dq/w/force/clamp search was unsafe because it could return
        and save a highest-reward failure.  Anchor-changing grids are rejected.
        """
        del grasp_pose, n_frames_out, wmax_grid, n_repeat
        dz_changes_anchor = dz_grid is not None and any(abs(float(v)) > 1e-12 for v in dz_grid)
        dq_changes_anchor = delta_grid is not None and any(abs(float(v)) > 1e-12 for v in delta_grid)
        if dz_changes_anchor or dq_changes_anchor:
            raise ValueError("Stage2 anchor search was removed; revise Stage1 instead")
        best = None
        for time_scale in (2.0, 3.0, 4.0):
            result = self.rollout_follow_trajectory(
                recon,
                side,
                F50_IDX=F50_IDX,
                carry_time_scale=time_scale,
                record_video=False,
            )
            if not result.get("gate_result", {}).get("passed", False):
                continue
            metrics = result["gate_result"]["metrics"]
            score = (
                metrics.get("tgo_translation_max_mm", float("inf"))
                + result["tracking_error"].get("mean_mm", float("inf"))
                + 0.1 * time_scale
            )
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "reward": -score,
                    "carry_time_scale": time_scale,
                    "dz": 0.0,
                    "delta": 0.0,
                    "result": result,
                }
        return best

    @staticmethod
    def save_stage2_bundle(output_path, result) -> bool:
        """Save only a physically passed executable bundle."""
        gate = result.get("gate_result", {})
        if not gate.get("passed", False):
            logger.warning("[Stage2 save] refusing to save failed result: %s", gate.get("failure"))
            return False
        bundle = result["bundle"]
        trace = result.get("trace", {})
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(bundle["timeline_meta"])
        metadata["bundle_hash"] = bundle["bundle_hash"]
        np.savez_compressed(
            output_path,
            source_frame=bundle["source_frame"],
            phase=bundle["phase"],
            base_pos=bundle["base_pos"],
            base_R=bundle["base_R"],
            gripper_q=bundle["gripper_q"],
            control_dt=bundle["control_dt"],
            w_pose=bundle["w_pose"],
            safety_offset=bundle["safety_offset"],
            expected_contact_policy=bundle["expected_contact_policy"],
            objective_weights=bundle["objective_weights"],
            objective_weight_names=bundle["objective_weight_names"],
            grasp_anchor_pos=np.asarray(bundle["grasp_anchor"]["pos"], dtype=np.float64),
            grasp_anchor_R=np.asarray(bundle["grasp_anchor"]["R"], dtype=np.float64),
            grasp_anchor_q=np.asarray(bundle["grasp_anchor"]["gripper_qpos"], dtype=np.float64),
            stage1_contract_hash=np.asarray(bundle.get("stage1_contract_hash", "")),
            physics_hash=np.asarray(bundle.get("physics_hash", "")),
            manifest_hash=np.asarray(bundle.get("manifest_hash", "")),
            stage1_reuse_report_json=np.asarray(
                json.dumps(
                    bundle.get("stage1_reuse_report", {}), sort_keys=True, default=_json_default
                )
            ),
            stage1_recipe_json=np.asarray(
                json.dumps(
                    (bundle.get("stage1_reuse_report", {}) or {}).get("recipe", {}),
                    sort_keys=True,
                    default=_json_default,
                )
            ),
            manifest_json=np.asarray(
                json.dumps(bundle.get("manifest", {}), sort_keys=True, default=_json_default)
            ),
            trace_source_frame=np.asarray(trace.get("source_frame", []), dtype=np.float64),
            trace_phase=np.asarray(trace.get("phase", []), dtype="U24"),
            trace_base_target=np.asarray(trace.get("base_target", []), dtype=np.float64),
            trace_base_actual=np.asarray(trace.get("base_actual", []), dtype=np.float64),
            trace_gripper_q_cmd=np.asarray(trace.get("gripper_q_cmd", []), dtype=np.float64),
            trace_gripper_q_actual=np.asarray(trace.get("gripper_q_actual", []), dtype=np.float64),
            trace_object_T=np.asarray(trace.get("object_T", []), dtype=np.float64),
            trace_gripper_T=np.asarray(trace.get("gripper_T", []), dtype=np.float64),
            trace_T_G_O=np.asarray(trace.get("T_G_O", []), dtype=np.float64),
            trace_target_bilateral=np.asarray(trace.get("target_bilateral", []), dtype=np.bool_),
            trace_target_contact=np.asarray(trace.get("target_contact", []), dtype=np.bool_),
            trace_ground_contact=np.asarray(trace.get("ground_contact", []), dtype=np.bool_),
            trace_non_target_contact=np.asarray(
                trace.get("non_target_contact", []), dtype=np.bool_
            ),
            trace_environment_contact=np.asarray(
                trace.get("environment_contact", []), dtype=np.bool_
            ),
            trace_grasp_force=np.asarray(trace.get("grasp_force", []), dtype=np.float64),
            trace_contact_objects_json=np.asarray(
                json.dumps(
                    [list(names) for names in trace.get("contact_objects", [])],
                    default=_json_default,
                )
            ),
            trace_contact_links_json=np.asarray(
                json.dumps(
                    [list(names) for names in trace.get("contact_links", [])],
                    default=_json_default,
                )
            ),
            timeline_meta_json=np.asarray(
                json.dumps(metadata, sort_keys=True, default=_json_default)
            ),
            gate_result_json=np.asarray(
                json.dumps(gate, sort_keys=True, default=_json_default)
            ),
        )
        logger.info("[Stage2 save] executable bundle saved: %s", output_path)
        return True
