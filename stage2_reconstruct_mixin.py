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
    scene_digest_hash,
    stage1_close_q_sequence,
    stage_q_open,
)


logger = logging.getLogger("grasp_hawor")

# F50 grasp budget is the shared Stage1 recipe: 20 close + 10 settle controls.
F50_CLOSE_CONTROLS = STAGE1_CLOSE_CONTROLS
F50_HOLD_CONTROLS = STAGE1_SETTLE_CONTROLS
F50_STOP_CONTROLS = STAGE1_STOP_CONTROLS
CARRY_VERIFY_CONTROLS = 5

PRE_READY_POS_TOL_M = 0.002
PRE_READY_ROT_TOL_RAD = np.deg2rad(2.0)
# 2026-09-11 (用户定策): 10mm/s -> 30mm/s.
# 用户要求"快到 F50 是一个平滑的移动, 而不是直接停下来; 抓取是可以在移动中进行的".
# 原先 10mm/s 逼迫基座在 F50 前完全静止, 与"平滑到达"直接冲突; 实测该窗口内基座
# 速度 10.71mm/s (仅超 7%), 而位置 0.18mm / 姿态 0.017deg / 角速度 0.31deg/s /
# 物体漂移 0.82mm 全部远优于容差 -> 速度项已成为唯一阻塞, 且不反映抓取质量.
PRE_READY_LINEAR_SPEED_MPS = 0.030
PRE_READY_ANGULAR_SPEED_RADPS = np.deg2rad(5.0)
# 2026-09-11 (用户定策): 2mm -> 10mm.
# 用户优先级: "要夹紧物体让它跟随夹爪". 但夹紧与"不推动物体"天然冲突 —
#   预紧 0 -> 0.008 时, 闭合成夹把物体挤动 4.80mm -> 7.09mm (夹得越紧推得越多).
# 实测分解: 接近段本身只推 0.82mm (readiness 样本 object_drift_mm=0.818, 余量充足),
#   其余 ~6.3mm 全部来自 CLOSE/SETTLE 的挤压 — 这是"夹住一个 4cm 小物体"的必然代价,
#   且抓取本身成立 (anchor 距物体 37.8mm ≈ 设计 36.9mm, 携带段 38.0mm 稳定, 双侧接触).
# 因此把该项作为软指标量级 (10mm), 不再让"挤压位移"否定一个已经成立的夹持.
PRE_GRASP_OBJECT_DRIFT_M = 0.010
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
    "default": (1 << 0, 1 << 0),
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
    ):
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
) -> dict:
    """Create the append-only Stage2 command timeline.

    ``source_frame`` is monotonic but may repeat.  Exactly 30 entries have source
    frame F50: ``close_controls`` (20) ``STAGE1_CLOSE`` controls followed by
    ``hold_controls`` (10) ``STAGE1_SETTLE`` controls — the shared Stage1 grasp
    recipe, not a local 3+2 approximation.  The next entry is already moving
    toward F51.
    """
    from physics_utils import GRIPPER_MAX_OPEN

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

    frames = sorted(int(f) for f in recon["frames"])
    f_start, f_end = frames[0], frames[-1]
    if not f_start < F50_IDX < f_end:
        raise ValueError(f"F50={F50_IDX} must lie strictly inside [{f_start}, {f_end}]")

    # Full-open entry command, shared with Stage1 (see stage1_recipe.stage_q_open).
    q_open = float(q_open) if q_open is not None else stage_q_open()
    q_pre = q_open if q_start is None else float(q_start)
    q_grasp = float(recon["grasp_anchor"]["gripper_qpos"])
    if not 0.0 <= q_grasp <= q_pre <= q_open + 1e-9:
        raise ValueError(
            f"Expected 0 <= q_grasp({q_grasp}) <= q_start({q_pre}) <= q_open({q_open})"
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
    for k in range(approach_first_k, approach_intervals + 1):
        progress = k / float(approach_intervals)
        src_for_pose = f_start + (F50_IDX - f_start) * _smoothstep5(progress)
        p, R = _interp_reconstruction(recon, src_for_pose)
        # The endpoint reaches the F50 pose while remaining an approach control;
        # the five following entries are the complete stationary F50 budget.
        source_value = np.nextafter(float(F50_IDX), -np.inf) if k == approach_intervals else src_for_pose
        if k >= approach_intervals - 1:
            phase_value = "FINAL_APPROACH"
        else:
            phase_value = "APPROACH_ADJUST"
        append_control(source_value, phase_value, p, R, q_open, "NO_CONTACT")

    anchor_pos, anchor_R = _interp_reconstruction(recon, float(F50_IDX))
    # The 20 close controls are the Stage1 cubic-smoothstep sequence verbatim.
    close_q = stage1_close_q_sequence(q_open, q_grasp, close_controls)
    if recipe is not None:
        recipe_q = np.asarray(recipe.get("close_ctrl_q", []), dtype=np.float64)
        if recipe_q.shape != close_q.shape or not np.allclose(recipe_q, close_q, atol=0.0):
            raise ValueError("Stage2 close command does not match the Stage1 recipe")
        if int(recipe.get("close_controls", -1)) != int(close_controls):
            raise ValueError("Stage2 close_controls does not match the Stage1 recipe")
    close_start_idx = len(source)
    for k in range(close_controls):
        append_control(
            F50_IDX, STAGE1_PHASE_CLOSE, anchor_pos, anchor_R,
            float(close_q[k]), "TARGET_CONTACT_ALLOWED",
        )
    hold_start_idx = len(source)
    for _ in range(hold_controls):
        append_control(F50_IDX, STAGE1_PHASE_SETTLE, anchor_pos, anchor_R, q_grasp, "BILATERAL_TARGET")

    verify_start_idx = len(source)
    for k in range(carry_verify_controls):
        src = F50_IDX + (k + 1) / float(carry_verify_controls)
        p, R = _interp_reconstruction(recon, src)
        append_control(src, "CARRY_VERIFY", p, R, q_grasp, "BILATERAL_TARGET")

    carry_intervals = max(1, int(np.ceil((f_end - (F50_IDX + 1)) * carry_time_scale)))
    for k in range(1, carry_intervals + 1):
        src = (F50_IDX + 1) + (f_end - (F50_IDX + 1)) * k / float(carry_intervals)
        p, R = _interp_reconstruction(recon, src)
        append_control(src, "CARRY", p, R, q_grasp, "STABLE_TARGET_GRASP")

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
            "hold_end": int(verify_start_idx - 1),
            "settle_start": int(hold_start_idx),
            "settle_end": int(verify_start_idx - 1),
            "verify_start": int(verify_start_idx),
            "verify_end": int(verify_start_idx + carry_verify_controls - 1),
            "close_controls": int(close_controls),
            "hold_controls": int(hold_controls),
            "settle_controls": int(hold_controls),
            "carry_verify_controls": int(carry_verify_controls),
            "approach_time_scale": float(approach_time_scale),
            "carry_time_scale": float(carry_time_scale),
            "q_start": float(q_open),
            "q_grasp": float(q_grasp),
            "close_profile": str(STAGE1_CLOSE_PROFILE),
            "recipe_revision": str((recipe or {}).get("revision", "")),
            "recipe_scene_digest_hash": str((recipe or {}).get("scene_digest_hash", "")),
        },
    }
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
        if q_near is not None:
            raise ValueError("q_near/pre-close is retired: the gripper must stay fully open until F50")
        q_start = stage_q_open()
        approach_min_z = float(approach_min_z)
        if not np.isfinite(approach_min_z):
            raise ValueError("approach_min_z must be finite")
        stage1_force = grasp_pose.get("force_limit")
        if stage1_force is not None and not np.isclose(float(stage1_force), float(GRIPPER_FORCE)):
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
            _fb_inner_y = 0.005 + q_start                 # 虎口内侧 |y| (q_open=0.05 → 0.055)
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
                    u = (t - _t_tr) / (1.0 - _t_tr)
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
            "preprocessed_prefix": {
                "source_frame": np.arange(0, f_start + 1, dtype=np.float64),
                "pos": mano_pos[: f_start + 1].copy(),
                "R": mano_R[: f_start + 1].copy(),
            },
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
        recon["bundle"] = build_stage2_timeline(
            recon, F50_IDX=F50_IDX, q_start=q_start, recipe=recon["stage1_recipe"]
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
    def _configure_stage1_target_isolated_collisions(robot, side, obj_actors, target_name):
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
            elif pair == frozenset(("gripper", "environment")) or pair == frozenset(
                ("target", "environment")
            ):
                ground_names.update(n for c, n in ((cat0, name0), (cat1, name1)) if c == "environment")

        return {
            "target_contact": bool(target_names),
            "ground_contact": bool(ground_names),
            "leak_contact": bool(leak_names),
            "target_names": sorted(target_names),
            "ground_names": sorted(ground_names),
            "leak_names": sorted(leak_names),
            "leak_pairs": sorted([(a, tuple(sorted(b))) for a, b in leak_pairs]),
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
        bundle = build_stage2_timeline(
            recon,
            F50_IDX=F50_IDX,
            approach_time_scale=approach_time_scale,
            carry_time_scale=carry_time_scale,
            q_start=q_start,
            recipe=recipe,
        )
        meta = bundle["timeline_meta"]

        robot = self.robot_info["robot"]
        collision_scope = self._configure_stage1_target_isolated_collisions(
            robot, side, self.obj_actors, target_name
        )
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
        leak_contact_names = set()
        ground_contact_names = set()
        gate_result = {"passed": False, "failure": None, "failure_detail": ""}
        tgo_reference = None

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
                if max_preclose_q_error > 0.003:
                    fail(
                        "PRE_F50_READY",
                        f"q tracking error={max_preclose_q_error * 1000.0:.2f}mm",
                    )
                    if _stop:
                        break
                if len(ready_samples) < 2 or not all(sample["passed"] for sample in ready_samples[-2:]):
                    fail("PRE_F50_READY", f"last readiness samples={ready_samples[-2:]}")
                    if _stop:
                        break

            target_pos = bundle["base_pos"][index]
            target_R = bundle["base_R"][index]
            q_cmd = np.asarray(bundle["gripper_q"][index], dtype=np.float64)
            if _grip_preload > 0.0 and index >= _close_start_idx:
                q_cmd = np.maximum(0.0, q_cmd - _grip_preload)
            rz, ry, rx = _cmd_euler[index]
            for key, value in (
                ("vx", target_pos[0]),
                ("vy", target_pos[1]),
                ("vz", target_pos[2]),
                ("rz", rz),
                ("ry", ry),
                ("rx", rx),
            ):
                active_joints[virtual_idx[key]].set_drive_target(float(value))
            active_joints[gi1].set_drive_target(float(q_cmd[0]))
            active_joints[gi2].set_drive_target(float(q_cmd[1]))
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
            T_G_O = _relative_pose(T_W_G, T_W_O) if T_W_G is not None else None

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
                ready_samples.append(
                    {
                        "passed": bool(
                            pos_error <= PRE_READY_POS_TOL_M
                            and rot_error <= PRE_READY_ROT_TOL_RAD
                            and linear_speed <= PRE_READY_LINEAR_SPEED_MPS
                            and angular_speed <= PRE_READY_ANGULAR_SPEED_RADPS
                            and not target_contacts
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
                if len(settle_bilateral) != settle_n:
                    fail(
                        "PROVISIONAL_GRASP",
                        f"settle samples={len(settle_bilateral)} != {settle_n}",
                    )
                    if _stop:
                        break
                if sum(settle_bilateral) < settle_n - 2 or not all(settle_bilateral[-3:]):
                    fail("PROVISIONAL_GRASP", f"settle bilateral={settle_bilateral}")
                    if _stop:
                        break
                if object_drift > PRE_GRASP_OBJECT_DRIFT_M:
                    fail("PROVISIONAL_GRASP", f"object drift={object_drift * 1000.0:.2f}mm")
                    if _stop:
                        break
                tgo_reference = T_G_O.copy() if T_G_O is not None else None
                if tgo_reference is None or not np.isfinite(tgo_reference).all():
                    fail("PROVISIONAL_GRASP", "missing finite T_G_O reference")
                    if _stop:
                        break

            if index == meta["verify_end"]:
                lo, hi = meta["verify_start"], meta["verify_end"] + 1
                verify_bilateral = trace["target_bilateral"][lo:hi]
                verify_tgo = trace["T_G_O"][lo:hi]
                t_err, r_err = _relative_pose_errors(tgo_reference, verify_tgo)
                relative_speed = (
                    float(np.max(np.linalg.norm(np.diff(np.asarray(verify_tgo)[:, :3, 3], axis=0), axis=1)) * CONTROL_FREQ)
                    if len(verify_tgo) >= 2
                    else float("inf")
                )
                if sum(verify_bilateral) < 4:
                    fail("CARRY_VERIFY", f"bilateral={sum(verify_bilateral)}/5")
                    if _stop:
                        break
                if (
                    len(t_err) != CARRY_VERIFY_CONTROLS
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
                }
            )
            if len(trace["control_index"]) != len(bundle["source_frame"]):
                fail("CARRY", "timeline did not finish")
            elif not np.isfinite(t_err).all() or not np.isfinite(r_err).all():
                fail("CARRY", "non-finite T_G_O metric")
            elif t_err.max() > TGO_TRANSLATION_TOL_M or r_err.max() > TGO_ROTATION_TOL_RAD:
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
                    _dr = np.array([
                        _rotation_distance(_gt[i][:3, :3], _br[i])
                        for i in _rows if np.isfinite(_gt[i]).all()
                    ])
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
        close_controls = int(recipe["close_controls"])
        settle_controls = int(recipe["settle_controls"])
        close_q = stage1_close_q_sequence(q_open, q_grasp, close_controls)
        stored_q = np.asarray(recipe.get("close_ctrl_q", []), dtype=np.float64)
        if stored_q.shape == close_q.shape and not np.allclose(stored_q, close_q, atol=0.0):
            return {
                "passed": False,
                "failure": "STAGE1_REUSE_MISMATCH",
                "failure_detail": "stored recipe close_ctrl_q does not reproduce",
                "n_pass": 0,
                "n_repeat": int(n_repeat),
                "runs": [],
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
            robot, side, self.obj_actors, target_name
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
                q_cmd = float(close_q[k]) if k < close_controls else q_grasp
                for key, value in base_targets.items():
                    active_joints[virtual_idx[key]].set_drive_target(value)
                active_joints[gi1].set_drive_target(q_cmd)
                active_joints[gi2].set_drive_target(q_cmd)
                physics_step(
                    robot, arm_indices, gi1, gi2, np.array([]), q_cmd, q_cmd, self.scene
                )
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

    def _record_stage2_frame(self, bundle: dict, index: int) -> None:
        """Render one already-executed control point; rendering never steps physics."""
        try:
            if not getattr(self.scene, "_render_available", False):
                return
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
            for camera_name, writer_name in (
                ("_cam_view", "_writer_cam"),
                ("_god_view", "_writer_god"),
                ("_right_view", "_writer_right"),
            ):
                camera = getattr(self, camera_name, None)
                writer = getattr(self, writer_name, None)
                if camera is None or writer is None:
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
