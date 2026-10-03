"""Stage 3: extend a physically validated Stage 2 executable bundle.

Stage 3 never reconstructs or resets the grasp.  A formal rollout first replays
the exact Stage 2 bundle from the shared scene reset and requires its hard gate
to pass.  It then continues from that physical state with an append-only suffix:

* Stage3A keeps the Stage 2 grasp command and carries the object to the release
  candidate while preserving the Stage2-end gripper/MANO correction.
* Stage3B opens only when the object is supported and slow.  Carry success and
  release success are deliberately reported as separate gates.

The old CEM entry point remains as an explicitly disabled compatibility method;
the previous implementation sampled parameters that did not affect commands.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from stage2_reconstruct_mixin import (
    OBJECTIVE_WEIGHT_NAMES,
    TGO_ROTATION_TOL_RAD,
    TGO_TRANSLATION_TOL_M,
    TRACKING_MAX_TOL_MM,
    TRACKING_MEAN_TOL_MM,
    _json_default,
    _relative_pose,
    _relative_pose_errors,
    _rotation_distance,
    _smoothstep5,
)


logger = logging.getLogger("grasp_hawor")

STAGE3_BILATERAL_RATIO_MIN = 0.80
STAGE3_TASK_PROGRESS_MIN = 0.80
STAGE3_MIN_TASK_MOTION_M = 0.005
STAGE3_RELEASE_LINEAR_SPEED_MPS = 0.020
STAGE3_RELEASE_ANGULAR_SPEED_RADPS = np.deg2rad(5.0)
STAGE3_RELEASE_DRIFT_M = 0.010
STAGE3_OPEN_TRACK_TOL_M = 0.003
# 2026-09-12 (用户定策 A): 释放判据改为"物理间隙"后, 指缝相对物体宽度的安全余量.
STAGE3_OPEN_CLEARANCE_M = 0.010
STAGE3_SUPPORT_GAP_M = 0.005


def _interp_pose(pos: np.ndarray, rot: np.ndarray, source_frame: float):
    """Interpolate a pose in MANO source-frame coordinates."""
    from scipy.spatial.transform import Rotation, Slerp

    n = min(len(pos), len(rot))
    if n == 0:
        raise ValueError("empty MANO pose trajectory")
    source_frame = float(np.clip(source_frame, 0.0, n - 1.0))
    lo = int(np.floor(source_frame))
    hi = min(lo + 1, n - 1)
    t = source_frame - lo
    p = (1.0 - t) * pos[lo] + t * pos[hi]
    if lo == hi:
        R = rot[lo].copy()
    else:
        R = Slerp(
            [0.0, 1.0], Rotation.from_matrix(np.stack([rot[lo], rot[hi]]))
        )([t]).as_matrix()[0]
    return p, R


def _scaled_rotation(R_correction: np.ndarray, weight: float) -> np.ndarray:
    """Return a fraction of a rotation correction (identity at 0)."""
    from scipy.spatial.transform import Rotation, Slerp

    weight = float(np.clip(weight, 0.0, 1.0))
    return Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack([np.eye(3), np.asarray(R_correction)])),
    )([weight]).as_matrix()[0]


def _stage3_hash(bundle: dict) -> str:
    digest = hashlib.sha256()
    digest.update(str(bundle.get("parent_bundle_hash", "")).encode("ascii"))
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


def build_stage3_timeline(
    mano_pos_aligned: np.ndarray,
    mano_R: np.ndarray,
    stage2_bundle: dict,
    release_candidate: int = 90,
    carry_time_scale: float = 2.0,
    # 2026-09-12 (用户定策): F90 放置不需要"添加帧", 直接打开夹爪.
    #   现状曾为 release=5 + pre_settle=3 + settle=10 (共 18 帧) —— 用户: "添加的帧顶多是在
    #   F50 这个地方进行夹取时添加". F50 的 20 close + 10 settle 是预设, 保留;
    #   F90 这 18 帧全部裁掉 → 后缀 96 → ~78 控制点.
    #   release_controls=1 时 smoothstep5(1/1)=1 → q 直接置为 GRIPPER_MAX_OPEN, 无渐变.
    #   PRE_RELEASE_SETTLE / RELEASE_SETTLE 计为 0 时对应循环不执行, slice 为空 (安全).
    # ★ 2026-09-22 【契约同步】放开段 = **5 个控制点** ✓ (审计 §7-3 第 3 项 ✓)
    #   依据 ✓: 测试契约名明确为 "explicit **five** control release" ✓
    #     (`tests/test_stage3_timeline.py::test_stage3_holds_grasp_until_explicit_five_control_release` ✓
    #      断言 `int(np.sum(phase == "RELEASE")) == 5` ✓)
    #   实际旧值 ✗: 3 ⇒ 与契约不符 ✗ ⇒ 按契约改为 **5** ✓ (并与测试同一常量 ✓)
    #   注 ✓: 5 个控制点 @30Hz ≈ 0.167s 的连续放开 ✓ (无明显速度冲击 ✓, 见下行 smoothstep ✓)
    release_controls: int = 5,
    pre_release_settle_controls: int = 0,
    release_settle_controls: int = 0,
    # 2026-09-12 (用户定策): 退避段继续缩小 → 2 帧.
    # 原实现从 release_candidate+1 一直跑到轨迹末帧 (F90→F112 = 49 帧), 属"添加帧过多".
    # ★ 2026-09-13 (用户定策): 释放后要**继续跟随原夹爪轨迹到轨迹末帧** (90→112),
    #   而不是只跟 2 帧合成退避. 历史: 原实现跑到轨迹末帧(49 帧), 2026-09-12 被
    #   缩减为 2 帧 —— 用户明确要求恢复"跟随原夹爪轨迹"的行为.
    retreat_controls: int = 23,
    control_freq: float = 30.0,
) -> dict:
    """Build a Stage3 suffix and an immutable-prefix combined command bundle.

    The first suffix command has ``source_frame > Stage2.f_end``.  During carry,
    the Stage2-end correction relative to MANO is fixed and the fingers remain at
    ``q_grasp``.  Only the explicit RELEASE phase changes the finger command.
    """
    from physics_utils import GRIPPER_MAX_OPEN
    # ★★★ 2026-09-14 用户: "之前夹爪开合大…我说的这些问题还是存在"
    #   查证: Stage1 已改用"按物体尺寸自动开度" ✓, 但 **Stage3 的 RELEASE 仍用裸的
    #   GRIPPER_MAX_OPEN (0.05)** ✗ —— 换算成指缝 = FINGER_GAP0 + 2×0.05 = **130.4mm** ✗✗
    #   (真实指形的指缝基准就有 30.4mm, 所以关节值 0.05 ≠ 张开 5cm ✓)
    #   而按尺寸自动算 = 0.0225 → 指缝 75.5mm ✓ (对 55.9mm 物体刚好 ✓)
    #   ⇒ 改为复用 stage_q_open() (与 Stage1/Stage2 **同一个值** ✓, 保证三段一致 ✓)
    from stage1_recipe import stage_q_open as _stage_q_open_s3
    from physics_utils import FINGER_GAP0 as _FINGER_GAP0_S3   # CSV 指缝换算用 ✓
    FINGER_GAP0 = _FINGER_GAP0_S3

    mano_pos_aligned = np.asarray(mano_pos_aligned, dtype=np.float64)
    mano_R = np.asarray(mano_R, dtype=np.float64)
    if mano_pos_aligned.ndim != 2 or mano_pos_aligned.shape[1] != 3:
        raise ValueError("mano_pos_aligned must have shape (N, 3)")
    if mano_R.shape != (len(mano_pos_aligned), 3, 3):
        raise ValueError("mano_R must have shape (N, 3, 3)")
    if carry_time_scale < 1.0:
        raise ValueError("carry_time_scale must be >= 1.0")
    # 2026-09-12 (用户定策): 只要求 RELEASE 至少 1 帧; 两个 settle 段允许为 0
    # (F90 直接开夹爪, 不做 pre-settle / post-settle).
    if release_controls < 1:
        raise ValueError("release phase must be non-empty")
    if pre_release_settle_controls < 0 or release_settle_controls < 0:
        raise ValueError("settle phases must be >= 0")

    parent_source = np.asarray(stage2_bundle["source_frame"], dtype=np.float64)
    if len(parent_source) == 0 or np.any(np.diff(parent_source) < 0.0):
        raise ValueError("Stage2 bundle source_frame must be non-empty and monotonic")
    handoff = float(parent_source[-1])
    release_candidate = int(release_candidate)
    if not handoff < release_candidate < len(mano_pos_aligned):
        raise ValueError(
            f"release candidate F{release_candidate} must lie after handoff "
            f"F{handoff:g} and before F{len(mano_pos_aligned)}"
        )

    parent_pos = np.asarray(stage2_bundle["base_pos"], dtype=np.float64)
    parent_R = np.asarray(stage2_bundle["base_R"], dtype=np.float64)
    parent_q = np.asarray(stage2_bundle["gripper_q"], dtype=np.float64)
    if parent_q.ndim != 2 or parent_q.shape[1] != 2 or len(parent_q) == 0:
        raise ValueError("Stage2 gripper_q must have shape (N, 2) and be non-empty")
    _q_endpoint = np.asarray(parent_q[-1], dtype=np.float64).reshape(2)
    if not np.isfinite(_q_endpoint).all():
        raise ValueError("Stage2 endpoint gripper_q must be finite")
    # Stage2's actual settle/vertical-planar endpoint is authoritative.  The
    # historical metadata q_grasp is a nominal recipe value and can differ
    # after measured close/settle or a structural carry policy.  Rejecting that
    # difference made a valid Stage2→Stage3 handoff impossible.
    _q_nominal = (stage2_bundle.get("timeline_meta", {}) or {}).get("q_grasp")
    _q_nominal_arr = np.asarray(_q_nominal, dtype=np.float64).reshape(-1) \
        if _q_nominal is not None else np.asarray([], dtype=np.float64)
    if _q_nominal_arr.size and not np.allclose(
        _q_endpoint, _q_nominal_arr if _q_nominal_arr.size == 2
        else _q_nominal_arr[0], atol=1e-12, rtol=0.0
    ):
        logger.warning(
            "[Stage3] 使用 Stage2 实际末端 q=%s；忽略名义 timeline_meta.q_grasp=%s",
            np.round(_q_endpoint, 9).tolist(), _q_nominal,
        )
    _q_grasp_vec = _q_endpoint.copy()
    q_grasp = (float(_q_endpoint[0]) if np.allclose(
        _q_endpoint[0], _q_endpoint[1], atol=1e-12, rtol=0.0
    ) else _q_grasp_vec.copy())
    q_grasp_meta = (float(_q_grasp_vec[0]) if np.allclose(
        _q_grasp_vec[0], _q_grasp_vec[1], atol=1e-12, rtol=0.0
    ) else _q_grasp_vec.tolist())

    mano_handoff_pos, mano_handoff_R = _interp_pose(mano_pos_aligned, mano_R, handoff)
    delta_p = parent_pos[-1] - mano_handoff_pos
    delta_R = parent_R[-1] @ mano_handoff_R.T

    source: list[float] = []
    phase: list[str] = []
    pos: list[np.ndarray] = []
    rot: list[np.ndarray] = []
    grip: list[tuple[float, float]] = []
    policy: list[str] = []

    def append(src, phase_name, p, R, q, contact_policy):
        source.append(float(src))
        phase.append(str(phase_name))
        pos.append(np.asarray(p, dtype=np.float64).copy())
        rot.append(np.asarray(R, dtype=np.float64).copy())
        _q = np.asarray(q, dtype=np.float64).reshape(-1)
        if _q.size == 1:
            grip.append((float(_q[0]), float(_q[0])))
        elif _q.size == 2:
            grip.append((float(_q[0]), float(_q[1])))
        else:
            raise ValueError("Stage3 gripper command must be scalar or a 2-vector")
        policy.append(str(contact_policy))

    carry_intervals = max(1, int(np.ceil((release_candidate - handoff) * carry_time_scale)))
    for k in range(1, carry_intervals + 1):
        progress = k / float(carry_intervals)
        # ★★★★ 2026-09-15 后缀源帧推进方式 (用户: "stage2 后面继续跟随夹爪, 我认为没有那么大的偏差")
        #   ★★★★
        #   原写法: src = handoff + Δ * smoothstep5(progress) ✗
        #     smoothstep5(0)=0 且 smoothstep5'(0)=0 ⇒ **后缀起点的源帧推进速度 = 0** ✗
        #     而 Stage2 在 F62 (handoff) 结束时**正在运动** ✓
        #       实测: 末 11 个控制点物体位移 161.9mm ⇒ ≈14.7mm/点 ⇒ ≈0.44 m/s ✓
        #     ⇒ 后缀一上来要求速度归零 ✗ ⇒ **速度突变** ⇒ 惯性冲量 ⇒ 物体被甩脱 ✗
        #     实测滑脱点恰好落在 smoothstep5 的**加速度峰值区**(后缀第 15 点 ✗) ✓ 完全吻合 ✓
        #   新写法: **线性推进** (carry_time_scale=1 时 = 1 源帧/控制点 ✓)
        #     ⇒ 起点源帧速度 = Stage2 的速率 ✓ ⇒ 位置与速度双连续 (C1) ✓
        #     这比"加大夹持力"重要得多: 力只能缓解 ✗, 速度不连续才是根源 ✓
        #   回退: _S3_CARRY_EASE=smooth ✓
        if os.environ.get("_S3_CARRY_EASE", "linear") == "smooth":
            src = handoff + (release_candidate - handoff) * _smoothstep5(progress)
        else:
            src = handoff + (release_candidate - handoff) * progress
        mano_p, mano_rot = _interp_pose(mano_pos_aligned, mano_R, src)
        append(
            src,
            "STAGE3_CARRY",
            mano_p + delta_p,
            delta_R @ mano_rot,
            q_grasp,
            "STABLE_TARGET_GRASP",
        )
    carry_end = len(source) - 1

    # Quantify "close to MANO" in the anchor-relative convention used by the
    # plan.  A fixed left SE(3) correction should preserve carry shape exactly;
    # keeping the measured values in the artifact catches future regressions.
    carry_mano_pos = []
    carry_mano_R = []
    for src in source[: carry_end + 1]:
        mano_p, mano_rot = _interp_pose(mano_pos_aligned, mano_R, src)
        carry_mano_pos.append(mano_p)
        carry_mano_R.append(mano_rot)
    carry_mano_pos = np.asarray(carry_mano_pos)
    carry_mano_R = np.asarray(carry_mano_R)
    command_carry_pos = np.asarray(pos[: carry_end + 1])
    command_carry_R = np.asarray(rot[: carry_end + 1])
    position_shape_error = np.linalg.norm(
        (command_carry_pos - parent_pos[-1]) - (carry_mano_pos - mano_handoff_pos), axis=1
    )
    rotation_shape_error = np.asarray(
        [
            _rotation_distance(
                parent_R[-1].T @ command_carry_R[index],
                mano_handoff_R.T @ carry_mano_R[index],
            )
            for index in range(len(command_carry_R))
        ]
    )
    command_steps = np.diff(np.vstack([parent_pos[-1], command_carry_pos]), axis=0)
    mano_steps = np.diff(np.vstack([mano_handoff_pos, carry_mano_pos]), axis=0)
    command_path_length = float(np.sum(np.linalg.norm(command_steps, axis=1)))
    mano_path_length = float(np.sum(np.linalg.norm(mano_steps, axis=1)))
    valid_steps = (np.linalg.norm(command_steps, axis=1) > 1e-9) & (
        np.linalg.norm(mano_steps, axis=1) > 1e-9
    )
    direction_cosine = (
        float(
            np.mean(
                np.sum(command_steps[valid_steps] * mano_steps[valid_steps], axis=1)
                / (
                    np.linalg.norm(command_steps[valid_steps], axis=1)
                    * np.linalg.norm(mano_steps[valid_steps], axis=1)
                )
            )
        )
        if np.any(valid_steps)
        else float("nan")
    )
    mano_fidelity = {
        "position_shape_mean_mm": float(np.mean(position_shape_error) * 1000.0),
        "position_shape_max_mm": float(np.max(position_shape_error) * 1000.0),
        "rotation_shape_mean_deg": float(np.rad2deg(np.mean(rotation_shape_error))),
        "rotation_shape_max_deg": float(np.rad2deg(np.max(rotation_shape_error))),
        "velocity_direction_cosine": direction_cosine,
        "path_length_ratio": (
            command_path_length / mano_path_length if mano_path_length > 1e-9 else float("nan")
        ),
    }

    release_pos = pos[-1].copy()
    release_R = rot[-1].copy()
    pre_release_start = len(source)
    for _ in range(pre_release_settle_controls):
        append(
            release_candidate,
            "PRE_RELEASE_SETTLE",
            release_pos,
            release_R,
            q_grasp,
            "SUPPORTED_OR_GRASPED",
        )
    pre_release_end = len(source) - 1

    release_start = len(source)
    # ★ 2026-09-14 用户: "夹爪开合大" —— RELEASE 的开合目标改为**按物体尺寸自动** ✓
    #   旧: GRIPPER_MAX_OPEN=0.05 → 指缝 130.4mm ✗ (物体 55.9mm 的 2.3 倍 ✗)
    #   新: stage_q_open() → 0.0225 → 指缝 75.5mm ✓ (与 Stage1/Stage2 完全一致 ✓)
    # 2026-09-14 用户: "物理参数不要老是乱改, 一次测试就一套" ⇒ **回退为原始值** ✓
    _q_open_s3 = float(GRIPPER_MAX_OPEN)
    for k in range(1, release_controls + 1):
        q = _q_grasp_vec + (_q_open_s3 - _q_grasp_vec) * _smoothstep5(
            k / float(release_controls)
        )
        append(
            release_candidate,
            "RELEASE",
            release_pos,
            release_R,
            q,
            "SUPPORTED_RELEASE",
        )
    release_end = len(source) - 1

    release_settle_start = len(source)
    for _ in range(release_settle_controls):
        append(
            release_candidate,
            "RELEASE_SETTLE",
            release_pos,
            release_R,
            GRIPPER_MAX_OPEN,
            "SUPPORTED_NO_GRASP",
        )
    release_settle_end = len(source) - 1

    retreat_start = len(source)
    # 2026-09-13: 恢复"跟随原夹爪轨迹到末帧" (retreat_controls=23 → 覆盖 F90→F113)
    last_frame = min(len(mano_pos_aligned) - 1, release_candidate + max(1, int(retreat_controls)))
    correction_fade_end = min(release_candidate + 3, last_frame)
    for frame in range(release_candidate + 1, last_frame + 1):
        correction_progress = (frame - release_candidate) / float(
            max(correction_fade_end - release_candidate, 1)
        )
        correction_weight = (
            1.0 - _smoothstep5(correction_progress)
            if frame <= correction_fade_end
            else 0.0
        )
        mano_p, mano_rot = _interp_pose(mano_pos_aligned, mano_R, frame)
        append(
            frame,
            "POST_RELEASE_BLEND" if frame <= correction_fade_end else "MANO_FOLLOW",
            mano_p + correction_weight * delta_p,
            _scaled_rotation(delta_R, correction_weight) @ mano_rot,
            GRIPPER_MAX_OPEN,
            "NO_TARGET_CONTACT",
        )
    retreat_end = len(source) - 1

    def stage3_objective(phase_name):
        if phase_name in ("STAGE3_CARRY", "PRE_RELEASE_SETTLE"):
            return np.asarray([1.0, 0.0, 1000.0, 1000.0, 1000.0])
        if phase_name == "RELEASE":
            return np.asarray([0.5, 0.0, 0.0, 0.0, 0.0])
        if phase_name == "RELEASE_SETTLE":
            return np.asarray([0.5, 1000.0, 0.0, 0.0, 0.0])
        return np.asarray([1.0, 1000.0, 0.0, 0.0, 0.0])

    suffix = {
        "source_frame": np.asarray(source, dtype=np.float64),
        "phase": np.asarray(phase, dtype="U24"),
        "base_pos": np.asarray(pos, dtype=np.float64),
        "base_R": np.asarray(rot, dtype=np.float64),
        "gripper_q": np.asarray(grip, dtype=np.float64),
        "control_dt": np.full(len(source), 1.0 / float(control_freq), dtype=np.float64),
        "w_pose": np.ones(len(source), dtype=np.float64),
        "safety_offset": np.zeros((len(source), 3), dtype=np.float64),
        "expected_contact_policy": np.asarray(policy, dtype="U32"),
        "objective_weights": np.asarray(
            [stage3_objective(name) for name in phase], dtype=np.float64
        ),
        "objective_weight_names": np.asarray(OBJECTIVE_WEIGHT_NAMES, dtype="U32"),
        "parent_bundle_hash": str(stage2_bundle["bundle_hash"]),
        "mano_fidelity": mano_fidelity,
        "timeline_meta": {
            "handoff_source_frame": handoff,
            "release_candidate": release_candidate,
            "carry_start": 0,
            "carry_end": carry_end,
            "pre_release_start": pre_release_start,
            "pre_release_end": pre_release_end,
            "release_start": release_start,
            "release_end": release_end,
            "release_settle_start": release_settle_start,
            "release_settle_end": release_settle_end,
            "retreat_start": retreat_start,
            "retreat_end": retreat_end,
            "correction_fade_end": correction_fade_end,
            "carry_time_scale": float(carry_time_scale),
            "release_controls": int(release_controls),
            "pre_release_settle_controls": int(pre_release_settle_controls),
            "release_settle_controls": int(release_settle_controls),
            "q_grasp": q_grasp_meta,
            "q_hold_actual": _q_grasp_vec.tolist(),
            "q_open": float(GRIPPER_MAX_OPEN),
            "delta_p": delta_p.tolist(),
            "delta_R": delta_R.tolist(),
        },
    }
    suffix["bundle_hash"] = _stage3_hash(suffix)
    suffix["combined_bundle"] = combine_stage2_stage3(stage2_bundle, suffix)
    return suffix


def combine_stage2_stage3(stage2_bundle: dict, suffix: dict, suffix_count: int | None = None) -> dict:
    """Concatenate Stage2 and an executed Stage3 suffix without mutating either."""
    n_suffix = len(suffix["source_frame"])
    if suffix_count is None:
        suffix_count = n_suffix
    suffix_count = int(np.clip(suffix_count, 0, n_suffix))
    combined = {}
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
        parent = np.asarray(stage2_bundle[key])
        child = np.asarray(suffix[key])[:suffix_count].astype(parent.dtype, copy=False)
        combined[key] = np.concatenate([parent.copy(), child], axis=0)
    combined.update(
        parent_bundle_hash=str(stage2_bundle["bundle_hash"]),
        stage3_suffix_hash=str(suffix["bundle_hash"]),
        stage1_contract_hash=str(stage2_bundle.get("stage1_contract_hash", "")),
        physics_hash=str(stage2_bundle.get("physics_hash", "")),
        manifest_hash=str(stage2_bundle.get("manifest_hash", "")),
        manifest=dict(stage2_bundle.get("manifest", {})),
        mano_fidelity=dict(suffix.get("mano_fidelity", {})),
        grasp_anchor=stage2_bundle.get("grasp_anchor", {}),
        objective_weight_names=np.asarray(
            stage2_bundle.get("objective_weight_names", OBJECTIVE_WEIGHT_NAMES), dtype="U32"
        ),
        timeline_meta={
            "stage2_control_count": int(len(stage2_bundle["source_frame"])),
            "stage3_executed_control_count": suffix_count,
            "stage3_planned_control_count": n_suffix,
            "stage2_meta": dict(stage2_bundle.get("timeline_meta", {})),
            "stage3_meta": dict(suffix.get("timeline_meta", {})),
        },
    )
    combined["bundle_hash"] = _stage3_hash(combined)
    return combined


def _empty_gate(failure: str, detail: str = "") -> dict:
    return {"passed": False, "failure": failure, "failure_detail": detail, "metrics": {}}


class Stage3Mixin:
    """Stage3 executable continuation and hard-gate helpers."""

    def stage3_reward(self, sim_result):
        """Compatibility score derived from gates, never used as an acceptance rule."""
        gate_a = sim_result.get("stage3a_gate", {})
        gate_b = sim_result.get("stage3b_gate", {})
        if not gate_a.get("passed", False):
            return -100.0
        metrics = gate_a.get("metrics", {})
        score = 100.0
        score -= float(metrics.get("tgo_translation_max_mm", 0.0))
        score -= float(metrics.get("tracking_mean_mm", 0.0))
        if gate_b.get("passed", False):
            score += 30.0
        return float(score)

    @staticmethod
    def _target_support_contact(robot, target_actor, scene):
        """Return whether target contacts a non-robot body and the body names."""
        robot_bodies = set()
        for link in robot.get_links():
            robot_bodies.update(link.entity.components)
        target_bodies = set(target_actor.components)
        support_names = set()

        def body_name(body):
            name = str(getattr(body, "name", "") or "")
            entity = getattr(body, "entity", None)
            if not name and entity is not None:
                name = str(getattr(entity, "name", "") or "")
            return name or type(body).__name__

        for contact in scene.get_contacts():
            if len(contact.bodies) < 2:
                continue
            body0, body1 = contact.bodies[:2]
            if body0 in target_bodies:
                other = body1
            elif body1 in target_bodies:
                other = body0
            else:
                continue
            if other in robot_bodies or other in target_bodies:
                continue
            support_names.add(body_name(other))
        return bool(support_names), sorted(support_names)

    def _support_ready(self, robot, target_actor, object_T_samples, target_name):
        from physics_env import CONTROL_FREQ

        supported, support_names = self._target_support_contact(robot, target_actor, self.scene)
        bbox_size = np.asarray(
            getattr(self, "obj_info", {}).get(target_name, {}).get("bbox_size", [np.nan] * 3),
            dtype=np.float64,
        )
        ground_z = float(getattr(self, "_ground_z", 0.0))
        pose_z = float(np.asarray(target_actor.get_pose().p)[2])
        bottom_gap = float("nan")
        if len(bbox_size) == 3 and np.isfinite(bbox_size[2]):
            bottom_gap = pose_z - 0.5 * float(bbox_size[2]) - ground_z
            if -STAGE3_SUPPORT_GAP_M <= bottom_gap <= STAGE3_SUPPORT_GAP_M:
                supported = True
                support_names = sorted(set(support_names) | {"ground_geometry"})

        samples = np.asarray(object_T_samples, dtype=np.float64)
        linear_speed = float("inf")
        angular_speed = float("inf")
        if len(samples) >= 3:
            recent = samples[-min(4, len(samples)) :]
            linear_speed = float(
                np.max(np.linalg.norm(np.diff(recent[:, :3, 3], axis=0), axis=1)) * CONTROL_FREQ
            )
            angular_speed = float(
                max(
                    _rotation_distance(recent[i, :3, :3], recent[i + 1, :3, :3])
                    for i in range(len(recent) - 1)
                )
                * CONTROL_FREQ
            )
        ready = bool(
            supported
            and linear_speed <= STAGE3_RELEASE_LINEAR_SPEED_MPS
            and angular_speed <= STAGE3_RELEASE_ANGULAR_SPEED_RADPS
        )
        return {
            "ready": ready,
            "supported": bool(supported),
            "support_names": support_names,
            "bottom_gap_mm": bottom_gap * 1000.0,
            "linear_speed_mm_s": linear_speed * 1000.0,
            "angular_speed_deg_s": np.rad2deg(angular_speed),
        }

    @staticmethod
    def _as_trace_arrays(trace):
        numeric = (
            "control_index",
            "source_frame",
            "base_target",
            "base_actual",
            "gripper_q_cmd",
            "gripper_q_actual",
            "object_T",
            "gripper_T",
            "T_G_O",
            "finger1_contact",
            "finger2_contact",
            "target_bilateral",
            "grasp_force",
            "illegal_contact",
        )
        for key in numeric:
            trace[key] = np.asarray(trace[key])
        trace["phase"] = np.asarray(trace["phase"], dtype="U24")
        return trace

    def _rollout_s2plus3_combined(self, side, release_candidate=90, carry_time_scale=1.0,
                                 enable_release=True):
        # ★ 任务07 (2026-09-28) ✓ md §07: "参数/父证书/profile 检查在 combined 分流**之前**
        #   统一执行; 禁止参数两路都拒绝; **支持参数完整传递**" ✓
        #   实测旧病 ✗: 本函数体内 `if stage3a_gate["passed"] and **enable_release**:` (1793)
        #     引用了**未定义名** ⇒ 一旦 Stage3A 通过就 `NameError` ✗ (参数没传进来) ✗
        """NEW 2026-09-15 (用户: "新建一个 stage2 测试一下, 看看效果如何").

        把 **Stage2 bundle + Stage3 后缀** 拼成**一条完整时间线**, 在**单次连续运行**里跑完
        ⇒ 彻底取消"接续"这个动作 ✗.

        为什么值得做:
          实测已确认 (用户推理也成立): F62 **跟随了参考轨迹** ✓ (CARRY tracking 0.67/0.89mm),
          后缀与 F62 末端**连续** ✓ (0.5~0.8mm / 0.1°), 抓持真实 ✓ (跟随比 1.036),
          物体未接触桌面 ✓ (ground_contact=False) ...
          可是"从函数外接续再走一步"必炸 ✗ (101.587 rad/s, 一次性, 与命令无关 ✗).
          ⇒ 若"单次连续"不炸 ✓ ⇒ 元凶是"接续/跨调用" ✓
            若"单次连续"也炸 ✗ ⇒ 与接续无关, 问题在接触求解本身 ✓
        本方法**不改动** rollout_follow_trajectory / Stage2 任何一行 ✓.
        """
        import sapien
        from data_loader import zyx_euler_continuous
        from physics_env import CONTROL_FREQ, physics_step

        parent = getattr(self, "_v4_stage2_bundle", None)
        recon = getattr(self, "_v4_stage2_recon", None)
        if parent is None or recon is None:
            return {"stage3a_gate": _empty_gate("DATA", "missing stage2 bundle/recon"),
                    "stage3b_gate": _empty_gate("NOT_RUN", "no parent"), "reward": -100.0}

        traj = self._mano_gripper_traj.get(side, {})
        mano_pos = np.asarray(traj.get("pos", []), dtype=np.float64)
        mano_R = np.asarray(traj.get("R", []), dtype=np.float64)
        _off = getattr(self, "_mano_neutral_offset", {}).get(side)
        if _off is not None:
            mano_pos = mano_pos + np.asarray(_off, dtype=np.float64)
        suffix = build_stage3_timeline(
            mano_pos, mano_R, parent,
            release_candidate=release_candidate,
            carry_time_scale=carry_time_scale,
            control_freq=CONTROL_FREQ,
        )

        bp = np.concatenate([np.asarray(parent["base_pos"], dtype=np.float64),
                             np.asarray(suffix["base_pos"], dtype=np.float64)], axis=0)
        bR = np.concatenate([np.asarray(parent["base_R"], dtype=np.float64),
                             np.asarray(suffix["base_R"], dtype=np.float64)], axis=0)
        bq = np.concatenate([np.asarray(parent["gripper_q"], dtype=np.float64),
                             np.asarray(suffix["gripper_q"], dtype=np.float64)], axis=0)
        n_par = len(parent["base_pos"])
        cmd_eul, _st = zyx_euler_continuous(bR, prev_cmd=None)
        logger.warning(
            "[S2+3连续] 单条时间线: Stage2 %d 点 + 后缀 %d 点 = %d 点 | 段间命令跳变 %.2fmm | "
            "euler 选支后相邻最大 rz 跳变 %.2f°",
            n_par, len(suffix["base_pos"]), len(bp),
            float(np.linalg.norm(bp[n_par] - bp[n_par - 1]) * 1000.0),
            _st["pick_jump_max_deg"],
        )

        robot = self.robot_info["robot"]
        vidx = self.robot_info.get("virtual_idx", {})
        gi1 = self.robot_info["gripper_idx1"]
        gi2 = self.robot_info["gripper_idx2"]
        arm_idx = self.robot_info.get("arm_joint_indices", [])
        active = robot.get_active_joints()
        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        tname = ctrl.target_obj if ctrl else None
        tactor = self._find_target_actor(self.obj_actors, tname)

        if not hasattr(self, "_obj_initial_poses") or self._obj_initial_poses is None:
            self._obj_initial_poses = {a.get_name(): a.get_pose() for a in self.obj_actors}
        robot.set_qpos(self.robot_info["init_qpos"].copy())
        robot.set_qvel(np.zeros_like(robot.get_qvel()))
        robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        for _a in self.obj_actors:
            _ip = self._obj_initial_poses.get(_a.get_name())
            if _ip is None:
                continue
            _a.set_pose(_ip)
            for _c in _a.get_components():
                if hasattr(_c, "set_linear_velocity"):
                    _c.set_linear_velocity([0, 0, 0])
                    _c.set_angular_velocity([0, 0, 0])
                    break

        q_start = float(recon["grasp_anchor"].get("q_start", 0.05))
        for _ in range(5):
            for _k, _v in (("vx", bp[0][0]), ("vy", bp[0][1]), ("vz", bp[0][2]),
                           ("rz", cmd_eul[0][0]), ("ry", cmd_eul[0][1]), ("rx", cmd_eul[0][2])):
                active[vidx[_k]].set_drive_target(float(_v))
            physics_step(robot, arm_idx, gi1, gi2, np.array([]), q_start, q_start, self.scene)

        # ★★★★ 2026-09-15 "固定的力" vs "压低命令" 对比实验 ★★★★
        #   (用户: "预紧和 set_drive_property 有什么区别, 你先测试一下把")
        #   区别 (机理):
        #     · _GRIP_PRELOAD: q_cmd = q_grasp - p ✗ ⇒ **位置命令被压低** ✗ ⇒ 手指要多闭进 3~8mm
        #       ⇒ 接触点沿物体表面**滑移** ✗ ⇒ 两侧不对称 ⇒ 净扭矩 ⇒ 物体**被拧**
        #       (实测 T_G_O **旋转**漂移 6.75°/12.16°, 而**位置**漂移仅 3.40mm ✓)
        #     · set_drive_property: 只改 K / D / force_limit ✓ ⇒ 命令**不动** ✓
        #       ⇒ 手指被刚体物体挡住, **位置由几何决定、保持不变** ✓
        #       ⇒ 同一位置下力线性增大 ✓ ⇒ 摩擦更大而**接触构型不变** ✓
        #   本实验: 携带段 (i >= n_par) 把夹爪刚度/力上限按环境变量放大 ✓ (不动任何全局常量 ✓)
        from physics_env import GRIPPER_DAMPING, GRIPPER_FORCE, GRIPPER_STIFFNESS
        _gk = float(os.environ.get("_S3_GRIP_K_SCALE", "1.0") or 1.0)
        _gf = float(os.environ.get("_S3_GRIP_F_SCALE", "1.0") or 1.0)
        _boost = (_gk != 1.0) or (_gf != 1.0)
        if _boost:
            logger.warning(
                "[S2+3连续] 携带段夹爪增强 (set_drive_property): K×%.2f=%s Fmax×%.2f=%s "
                "⇒ 位置命令**不变** ✓",
                _gk, GRIPPER_STIFFNESS * _gk, _gf, GRIPPER_FORCE * _gf)

        obj_t, tgo_t, bil_t = [], [], []
        # ★ 计划 §4.1 ✓: Stage3 复用 Stage2 的**命令后处理规则** ⇒ 先读同一组旋钮 ✓
        try:
            _gp3 = float(os.environ.get("_GRIP_PRELOAD", "0") or 0.0)
            _cf3 = float(os.environ.get("_S2_CLOSE_FLOOR_Q", "0") or 0.0)
        except Exception as _egp3:  # noqa: BLE001
            _gp3, _cf3 = 0.0, 0.0
        if _gp3 > 0.0 or _cf3 > 0.0:
            logger.info("[S2+3连续] ★ 计划 §4.1: Stage3 **复用** Stage2 命令后处理 "
                        "(`_GRIP_PRELOAD=%.4f` / `_S2_CLOSE_FLOOR_Q=%.4f`) ⇒ 与 Stage2 同规则 ✓",
                        _gp3, _cf3)
        for i in range(len(bp)):
            for _k, _v in (("vx", bp[i][0]), ("vy", bp[i][1]), ("vz", bp[i][2]),
                           ("rz", cmd_eul[i][0]), ("ry", cmd_eul[i][1]), ("rx", cmd_eul[i][2])):
                active[vidx[_k]].set_drive_target(float(_v))
            if _boost and i >= n_par:
                for _gj in (gi1, gi2):
                    active[_gj].set_drive_property(
                        stiffness=GRIPPER_STIFFNESS * _gk,
                        damping=GRIPPER_DAMPING,
                        force_limit=GRIPPER_FORCE * _gf,
                    )
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-09-22 【计划 §4.1 修复】Stage3 合并执行器**直接执行 `bq[i]`** ✗ ★★★★
            #   计划原话 ✓: "`stage3_optimize_mixin.py:701` 附近直接执行 `bq[i]`,
            #     **没有复用 Stage2 的这套命令后处理** … ⇒ '保存命令'/'Stage2 实际验证命令'/
            #     'Stage3 重放命令' **可能不是同一动作**" ✗✗
            #   本改 ✓: 接上**同一套后处理规则** (`_GRIP_PRELOAD` / `_S2_CLOSE_FLOOR_Q`) ✓
            #     ⇒ 同一份配置 ⇒ 同一份**实际下发命令** ✓ (默认值 0 ⇒ 与旧行为**完全一致** ✓)
            #   ⚠ 真正的正解是"**共享一个执行器**" ✓ (计划 §4.2 建议 ✓) ⇒ **尚未完成** ✓,
            #     本处只能保证"规则一致" ✓, 不能保证"执行循环/reset 一致" ✗ (诚实标注 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            _q_i0, _q_i1 = float(bq[i][0]), float(bq[i][1])
            if _gp3 > 0.0:
                _q_i0 = max(0.0, _q_i0 - _gp3)
                _q_i1 = max(0.0, _q_i1 - _gp3)
            if _cf3 > 0.0:
                _q_i0 = max(_q_i0, _cf3)
                _q_i1 = max(_q_i1, _cf3)
            physics_step(robot, arm_idx, gi1, gi2, np.array([]),
                         _q_i0, _q_i1, self.scene)
            _To = np.asarray(tactor.get_pose().to_transformation_matrix(), dtype=np.float64)
            _Tg = self._gripper_pose_matrix(robot, side)
            _f1, _f2, _objs, _ = self._target_finger_contacts(
                robot, side, self.scene, self.obj_actors, tname)
            _tgt = any(n == tname or (tname and tname in n) for n in _objs)
            obj_t.append(_To)
            tgo_t.append(_relative_pose(_Tg, _To) if _Tg is not None else np.full((4, 4), np.nan))
            bil_t.append(bool(_f1 and _f2 and _tgt))
            # 后缀段录像 (用户: "整个管线的最终视频我需要看得到") ✓
            if i >= n_par and os.environ.get("_S3_RECORD", "1") != "0":
                try:
                    self._record_stage3_frame(suffix, i - n_par)
                except Exception as _erv:  # noqa: BLE001
                    logger.warning("[S2+3连续] 录像失败 (第%d点): %s", i - n_par, _erv)
            if i >= n_par - 1 and (i <= n_par + 2 or (i - n_par) % 5 == 0):
                logger.warning(
                    "[S2+3连续] i=%d/%d (Stage2末=%d, 后缀第%d点) | 物体z=%.4f 双侧=%s "
                    "| T_G_O位移=%.1fmm | 手指q=%s",
                    i, len(bp), n_par - 1, i - n_par, float(_To[2, 3]), str(bil_t[-1]),
                    float(np.linalg.norm(tgo_t[-1][:3, 3])) * 1000.0,
                    np.round(np.asarray(robot.get_qpos(), dtype=np.float64)[[gi1, gi2]], 4).tolist())

        # ★★★★ 2026-09-15 判据口径修正 (用户: "现在的问题到底出在哪里") ★★★★
        #   原判据用**整个后缀**算 T_G_O drift ✗ ⇒ 把"RELEASE 松手后物体自由落体"当成失败 ✗.
        #   Stage3A 的设计: **carry 段内物体必须跟随** ✓; **释放后脱离是预期** ✓
        #   (实测: 释放前 carry 段 T_G_O 稳定 37.0~38.2mm ✓, 释放后 z 从 26cm 掉到桌面 ✓)
        #   ⇒ 只对 carry 段判定 ✓, 同时把整段 drift 作为参考打出来 ✓
        _ph_s3 = np.asarray(suffix["phase"], dtype="U24")
        _carry_n = int(np.sum(_ph_s3 == "STAGE3_CARRY"))
        _end_carry = int(min(len(tgo_t), n_par + _carry_n))
        _ref = tgo_t[n_par - 1]
        _dr = [float(np.linalg.norm((t - _ref)[:3, 3])) * 1000.0 for t in tgo_t[n_par:_end_carry]]
        _dr_all = [float(np.linalg.norm((t - _ref)[:3, 3])) * 1000.0 for t in tgo_t[n_par:]]
        _max_drift = max(_dr) if _dr else float("nan")
        _bil_ratio = float(np.mean(bil_t[n_par:_end_carry])) if _end_carry > n_par else float("nan")
        _passed = bool(np.isfinite(_max_drift) and _max_drift <= 10.0 and _bil_ratio >= 0.80)
        logger.warning(
            "[S2+3连续] 结论: **carry 段(%d点)** max T_G_O 位移=%.1fmm 双侧比例=%.2f "
            "| 整段(含释放)=%.1fmm | Stage2末物体z=%.4f → carry末z=%.4f → 段末z=%.4f ⇒ %s",
            _carry_n, _max_drift, _bil_ratio, (max(_dr_all) if _dr_all else float("nan")),
            float(obj_t[n_par - 1][2, 3]),
            float(obj_t[_end_carry - 1][2, 3]) if _end_carry > n_par else float("nan"),
            float(obj_t[-1][2, 3]),
            "PASS ✓ (carry 段跟随稳定)" if _passed else "FAIL ✗ (carry 段漂移)",
        )
        # ★★★★ 2026-09-15 补 bundle / trace (Stage3A PASS 后 save_stage3_bundle 会读它们 ✓)
        #   原先缺 "bundle" ⇒ save_stage3_bundle 抛 KeyError ✗ (rc70 实测 ✓)
        #   合并 bundle 用现成的 combine_stage2_stage3 ✓ (与旧路径口径一致 ✓)
        _combined_bundle = {}
        try:
            _combined_bundle = combine_stage2_stage3(parent, suffix)
        except Exception as _ecb:  # noqa: BLE001
            logger.warning("[S2+3连续] combined bundle 构造失败: %s", _ecb)

        # ★★★★ 2026-09-15 Stage3B 简化 (用户: "stage3 只看物体能不能跟随夹爪,
        #   放置这个部分不用管 z 轴的") ★★★★
        #   原 Stage3B 有 4 条判据: OPEN_TRACK / NOT_RELEASED / 物体速度与漂移 /
        #     **支撑检查 (bottom_gap 与 ground_z 比较 ⇒ 看 z)** ✗
        #   按用户定策: **放置段不看 z** ✗ ⇒ 去掉"支撑/速度/漂移"三条 ✓,
        #     只保留"夹爪是否真的张开到接近全开" ✓ ⇒ **空中释放因此天然合法** ✓
        from physics_env import GRIPPER_MAX_OPEN as _S3_MAX_OPEN
        _fq_final = np.asarray(robot.get_qpos(), dtype=np.float64)[[gi1, gi2]]
        _open_thr = float(_S3_MAX_OPEN) - float(STAGE3_OPEN_TRACK_TOL_M)
        _open_ok = bool(np.min(_fq_final) >= _open_thr)
        logger.warning(
            "[S2+3连续] Stage3B(简): 最终手指q=%s 阈值=%.4f ⇒ %s",
            np.round(_fq_final, 4).tolist(), _open_thr,
            "PASS ✓ (张开到位)" if _open_ok else "FAIL ✗ (没张开到位)",
        )
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 2026-09-22 【计划 §4.2 修复】`passed` 必须来自**本次前缀实测** ✗ ★★★★
        #   计划原话 ✓: "`:776` 返回的 `stage2_replay.gate_result.passed` 是**直接写死的 True**,
        #     不是本次前缀物理测量得出的结论" ✗✗
        #     "Stage2 先前通过, 只能作为**进入 Stage3 的前提**, 不能替代合并重放
        #     这一次前缀的**实际核验**" ✓
        #   本改 ✓: 用**本段 trace 的 `T_G_O`** 实测"物体相对夹爪"的平移 + 旋转 ⇒
        #     与正式路径同判据 (平移 ≤10mm ∧ 旋转 ≤5°) ⇒ 不再写死 ✓
        #   ⚠ 未知/异常 ⇒ **判 False** ✓ (不允许"没测 = 通过" ✗)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _prefix_passed, _prefix_dpos_mm, _prefix_drot_deg = False, float("nan"), float("nan")
        try:
            _T = np.asarray(tgo_t, dtype=np.float64)
            if _T.ndim == 3 and len(_T):
                _R0, _p0 = _T[0, :3, :3], _T[0, :3, 3]
                _prefix_dpos_mm = float(np.max(np.linalg.norm(
                    _T[:, :3, 3] - _p0, axis=1))) * 1000.0
                _rots = []
                for _k in range(len(_T)):
                    _tr = float(np.trace(_T[_k, :3, :3].T @ _R0))
                    _rots.append(float(np.degrees(np.arccos(
                        np.clip((_tr - 1.0) / 2.0, -1.0, 1.0)))))
                _prefix_drot_deg = float(np.max(_rots)) if _rots else float("nan")
                _prefix_passed = bool(np.isfinite(_prefix_dpos_mm)
                                      and np.isfinite(_prefix_drot_deg)
                                      and _prefix_dpos_mm <= 10.0
                                      and _prefix_drot_deg <= 5.0)
        except Exception as _ep:  # noqa: BLE001
            logger.warning("[S2+3连续] 前缀实测判据失败 ⇒ 判 FAIL ✗: %s", _ep)
        logger.info(
            "[S2+3连续] ★ **前缀实测判据** (计划 §4.2 ✓ 不再写死 True): "
            "Δpos=%.2fmm / Δrot=%.2f° ⇒ %s",
            _prefix_dpos_mm, _prefix_drot_deg,
            "PASS ✓" if _prefix_passed else "FAIL ✗ (本次前缀未能维持相对位姿 ✗)")
        return {
            "bundle": _combined_bundle,
            "stage2_replay": {"gate_result": {"passed": bool(_prefix_passed),
                                              "dpos_mm": _prefix_dpos_mm,
                                              "drot_deg": _prefix_drot_deg},
                              "bundle": parent},
            "trace": {
                "control_index": list(range(len(bp))),
                "object_T": np.asarray(obj_t, dtype=np.float64),
                "T_G_O": np.asarray(tgo_t, dtype=np.float64),
                "target_bilateral": np.asarray(bil_t, dtype=bool),
                "basis": "stage2_plus_stage3_single_run",
                "n_parent_controls": int(n_par),
                "carry_controls": int(_carry_n),
            },
            "stage3a_gate": {
                "passed": _passed,
                "failure": None if _passed else "T_G_O",
                "failure_detail": f"combined-run max_drift={_max_drift:.1f}mm bil={_bil_ratio:.2f}",
                "metrics": {"tgo_translation_max_mm": _max_drift, "bilateral_ratio": _bil_ratio},
            },
            "stage3b_gate": {
                "passed": _open_ok,
                "failure": None if _open_ok else "OPEN_TRACK",
                "failure_detail": (
                    f"final_q={np.round(_fq_final, 4).tolist()} 阈值={_open_thr:.4f} "
                    f"(简化版: 不看 z ✓)"
                ),
                "metrics": {"final_q": _fq_final.tolist(), "open_threshold": _open_thr},
            },
            "reward": 0.0 if _passed else -100.0,
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            # ★★★★ 2026-10-03 【P0-1: 总门在**默认路径**上形同虚设 ✗】 ✓ ★★★★
            #   实测 ✗: `_S3_COMBINED` 默认 "1" ⇒ 走本(合并)路径; 而 `task_pass`
            #     **只在旧路径(1968 行)计算** ⇒ 本路径 result **没有 `task_pass`** ⇒
            #     save(2040)必然走"回退口径"(Stage3A ∧ 释放) ⇒ md §07 总门失效 ✗
            #   本改 ✓: 按官方定义(1960-1969)在本路径**同样计算并回传** ✓
            #     = 数据有效 ∧ artifact 有效 ∧ 父段重放 ∧ carry ∧(未启用释放 或 释放通过)∧ 非诊断
            #   ⚠ 影响 ✓: 加了之后 save 不再回退 ⇒ 总门真正生效 ⇒ 不满足则**拒绝保存** ✗
            #     临时回退: `_S3_TASK_PASS_ENFORCE=0` (只记录不拦截 ✓)
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
            "enable_release": bool(enable_release),
            "task_pass_terms": {
                "data_valid": True,
                "artifact_valid": bool(getattr(self, "_s3_artifact_valid", True)),
                "parent_replay_passed": bool(_prefix_passed),
                "carry_passed": bool(_passed),
                "release_ok": (not bool(enable_release)) or bool(_open_ok),
                "diagnostic": bool(getattr(self, "_s3_diagnostic", False)
                                   or str(os.environ.get("_S3_DIAGNOSTIC", "0")) == "1"),
            },
            "task_pass": bool(
                True
                and bool(getattr(self, "_s3_artifact_valid", True))
                and bool(_prefix_passed)
                and bool(_passed)
                and ((not bool(enable_release)) or bool(_open_ok))
                and not (bool(getattr(self, "_s3_diagnostic", False))
                         or str(os.environ.get("_S3_DIAGNOSTIC", "0")) == "1")
            ) if str(os.environ.get("_S3_TASK_PASS_ENFORCE", "1")) != "0" else True,
        }

    def rollout_v4_stage3(
        self,
        opt_params=None,
        side="right",
        decimation=None,
        carry_time_scale=2.0,
        release_candidate=90,
        enable_release=True,
        record_video=False,
    ):
        """Replay exact Stage2, then execute Stage3A and conditional Stage3B."""
        # ★★★★ 2026-09-15 合并时间线 (默认路径, 用户: "可以的") ★★★★
        #   实测结论 (comb1): 旧路径"从函数外接续"在接续点第 1 子步就炸 (101.587 rad/s ✗);
        #   而**单条时间线 + 单次连续运行**下, 接续点完全平稳 ✓:
        #     i=76 物体z=0.1467 → i=77 z=0.1472 → i=78 z=0.1480 → i=81 z=0.1585 (正常抬起 ✓)
        #     双侧=True 全程 ✓, T_G_O 稳定 38.2~38.3mm ✓ **没有爆炸** ✓
        #   ⇒ 元凶 = "跨函数接续" ✗ ⇒ 默认改走本路径 ✓ (复用 Stage2 的 reset+warmup+循环 ✓,
        #     不动物理参数 ✓, 也不改动 rollout_follow_trajectory 任何一行 ✓)
        #   旧路径保留: _S3_COMBINED=0 可回退 ✓
        if os.environ.get("_S3_COMBINED", "1") != "0":
            logger.warning("[Stage3][合并模式] 单条时间线连续跑完 (无接续) ✓ 默认路径")
            return self._rollout_s2plus3_combined(
                side,
                release_candidate=release_candidate,
                carry_time_scale=carry_time_scale,
                enable_release=enable_release,     # ★ 任务07 ✓ 参数完整传递 (不得两路都拒 ✗)
            )
        from data_loader import rotmat_to_zyx_euler, zyx_euler_continuous
        from physics_env import CONTROL_FREQ, DECIMATION, get_grasp_force, physics_step

        if opt_params is not None and np.asarray(opt_params).size:
            raise ValueError("Stage3 CEM/pose offsets are disabled; only time scaling is allowed")
        if decimation is not None and int(decimation) != int(DECIMATION):
            raise ValueError("Stage3 must use the shared Stage1/Stage2 decimation")

        recon = getattr(self, "_v4_stage2_recon", None)
        parent = getattr(self, "_v4_stage2_bundle", None)
        if recon is None or parent is None:
            detail = "missing passed _v4_stage2_recon/_v4_stage2_bundle"
            logger.error("[Stage3] %s", detail)
            return {"stage3a_gate": _empty_gate("CONTRACT", detail), "reward": -100.0}

        parent_meta = parent.get("timeline_meta", {})
        # ★★★★ 2026-09-14 重放前后自检 (用户: "还是看看 stage3") ★★★★
        #   已确认事实:
        #     · 后缀第 0 帧: 双侧=False ✗ 夹爪-物距离=**274.6mm** ✗ 物体 z 走抛物线 0.056→0.239→0.142
        #       ⇒ 物体在**后缀开始前就被抛飞**了 ✓
        #     · 但 replay_gate **通过**了 ✓ (失败码是 T_G_O 而非 STAGE2_REPLAY ✓)
        #     ⇒ 矛盾: 重放过了 ✓ 后缀起手却 274mm ✗ ⇒ 两者**状态不同步** ✓
        #   本自检: 在重放**前后**各打一次关键状态, 一次分清
        #     (a) 重放前就已经飞了 → 场景初始化问题 ✓
        #     (b) 重放把它撞飞了    → 重放与 Stage2 的执行差异 ✓
        #     (c) 重放后正常(≤40mm) → 问题在"重放→后缀"的衔接 ✓
        def _s3_state(tag):
            # 自包含实现: 本处 (L576) 早于 target_name(L678)/is_target(L733) 的定义,
            # 不能引用它们 ✗ ⇒ 从控制器直接取目标名, 接触判定用"任一物体"近似 ✓
            try:
                _ctrl_x = (getattr(self, "grasp_controllers", None) or {}).get(side)
                _tn_x = getattr(_ctrl_x, "target_obj", None) if _ctrl_x is not None else None
                _ta = self._find_target_actor(self.obj_actors, _tn_x) if _tn_x else None
                if _ta is None:
                    logger.warning("[Stage3 重放自检] %s: 找不到目标物体 ✗", tag)
                    return None
                _To = np.asarray(_ta.get_pose().to_transformation_matrix(), dtype=np.float64)
                # ★ 注意: 本闭包所在处早于 "robot = self.robot_info['robot']" (L693) ✗,
                #   直接引用 robot 会被 Python 判定为"自由变量未赋值" ✗ ⇒ 每次现取 ✓
                _rb = self.robot_info["robot"]
                _Tg = self._gripper_pose_matrix(_rb, side)
                _d = (float(np.linalg.norm(_Tg[:3, 3] - _To[:3, 3]) * 1000.0)
                      if _Tg is not None and np.isfinite(_Tg).all() else float("nan"))
                _f1, _f2, _objs, _ = self._target_finger_contacts(
                    _rb, side, self.scene, self.obj_actors, _tn_x)
                _bil = bool(_f1 and _f2 and len(_objs) > 0)
                logger.warning(
                    "[Stage3 重放自检] %-8s 物体z=%.4f 夹爪-物距离=%.1fmm 双侧=%s",
                    tag, float(_To[2, 3]), _d, str(_bil))
                return _d
            except Exception as _e:  # noqa: BLE001
                logger.warning("[Stage3 重放自检] %s 采集失败: %s", tag, _e)
                return None
        _s3_state("接续前")
        # ★★★★ 2026-09-14 接续真实 Stage2 末态 (用户: "你不能分开渲染, 就是 stage2 正确的出来,
        #   你 stage3 之前接上, 肯定得按照 stage2 原来的参数和夹持来, 然后调整到 stage3,
        #   我觉得重要的还是仿真的真实轨迹, 前面复现的一些参考点后面可能还需要重新渲染") ★★★★
        #   原实现: 这里**再次 reset 并完整重放一遍 Stage2 bundle** ✗
        #     ⇒ 后缀起手的物理态来自"第二次仿真" ✗, 与 Stage2 真正跑出、已渲染的那一次
        #       不是同一条真实轨迹 ✗
        #     ⇒ 实测: 后缀起手物体根本不在手里 (fix3 idx=0 双侧=False / 物体z=0.0269 ✗),
        #       而重放门禁仍判 PASS ✓ —— 它只测标称量 (夹爪-物距离≈37mm / 双侧接触),
        #       实测"接续前"与"接续后"四位小数完全相同 ⇒ 该自检分辨不出是否抓住 ✗
        #     ⇒ 且视频里 Stage2 段与 Stage3 段来自两次独立仿真, 拼接必然不连续 ✗
        #   新实现: 直接复用**进程内上一次真实 Stage2 rollout 的 trace 与末态** ✓
        #     场景**不复位**, 后缀在同一个场景、同一条物理轨迹上继续 ✓
        #     (调用方 grasp_hawor.py 在 Stage2 repeat 之后立刻在此处接续, 中间只写文件不动物理 ✓)
        _live_parent = getattr(self, "_v4_stage2_result", None) or {}
        _live_trace = _live_parent.get("trace") or {}
        if _live_trace:
            replay = _live_parent
            logger.info(
                "[Stage3] 接续真实 Stage2 末态 (不 reset / 不重放): 后缀与 Stage2 同一场景 ✓ "
                "trace=%d 控制点 gate_passed=%s",
                int(np.asarray(_live_trace.get("control_index", [])).size),
                (_live_parent.get("gate_result") or {}).get("passed"),
            )
        else:
            logger.warning("[Stage3] 进程内没有真实 Stage2 末态, 回退 reset+重放 (旧路径)")
            replay = self.rollout_follow_trajectory(
                recon,
                side,
                F50_IDX=int(parent_meta.get("f_grasp", 50)),
                approach_time_scale=float(parent_meta.get("approach_time_scale", 2.0)),
                carry_time_scale=float(parent_meta.get("carry_time_scale", 2.0)),
                record_video=bool(record_video),
            )
        _s3_state("接续后")
        replay_gate = replay.get("gate_result", {})
        replay_bundle = replay.get("bundle", {})
        if not replay_gate.get("passed", False):
            detail = f"formal Stage2 replay failed: {replay_gate.get('failure')}"
            # 诊断旁路 (2026-09-11, 用户要求"赶快渲染一下 stage3"):
            # 契约上 Stage3A 的父输入必须是过了门的 Stage2. 当 Stage2 只差边缘门时,
            # _S3_DIAG=1 允许继续跑 Stage3 后缀 (F50→F90), 以便先观察携带/释放表现.
            # ⚠ 该路径产出的 Stage3 结果**不作为 PASS**.
            if os.environ.get("_S3_DIAG", "0") == "1":
                logger.warning(
                    "[Stage3][诊断] formal Stage2 replay 未过 (%s); _S3_DIAG=1 → "
                    "继续执行 Stage3 后缀, 结果不作为 PASS",
                    replay_gate.get("failure"),
                )
            else:
                logger.warning("[Stage3] %s", detail)
                return {
                    "stage2_replay": replay,
                    "stage3a_gate": _empty_gate("STAGE2_REPLAY", detail),
                    "stage3b_gate": _empty_gate("NOT_RUN", "Stage3A prerequisite failed"),
                    "reward": -100.0,
                }
        contract_keys = ("bundle_hash", "stage1_contract_hash", "physics_hash", "manifest_hash")
        mismatch = [
            key for key in contract_keys
            if str(parent.get(key, "")) != str(replay_bundle.get(key, ""))
        ]
        if mismatch:
            detail = f"Stage2 replay contract mismatch: {mismatch}"
            logger.error("[Stage3] %s", detail)
            return {
                "stage2_replay": replay,
                "stage3a_gate": _empty_gate("CONTRACT", detail),
                "stage3b_gate": _empty_gate("NOT_RUN", "Stage3A prerequisite failed"),
                "reward": -100.0,
            }

        # ★★★★ 2026-09-14 接续前的"真抓持"硬校验 (用户: "stage2 整体没问题, 说明 stage3 调用
        #   stage2 有问题, 你需要完成 stage3 的修复") ★★★★
        #   实测已定论:
        #     · 爆炸与**命令无关** ✗ —— 发 Stage2 自己的最后一条命令也炸, 数值逐位相同 (101.587) ✓
        #     · 爆炸与**Stage3 代码无关** ✗ —— 在 Stage2 主循环内就地再走一步也炸 ✓
        #     ⇒ **继承来的"Stage2 末态"本身就不是可继续的状态** ✗
        #   而 Stage2 自己的门禁仍 PASS ✓ —— 因为它的判据全是**标称量**:
        #     tracking(夹爪vs命令) / dist2obj(夹爪vs物体距离) / T_G_O / 双侧接触
        #     ⇒ "物体纹丝不动"时这些量**全都是稳定的** ✗ ⇒ 假通过 ✗
        #   本校验只问一个问题: **夹爪动的时候, 物体跟着动了吗?** ✓
        #     不跟动 ⇒ 抓持是假的 ✓ ⇒ 立刻以 HANDOFF_NO_GRASP 失败,
        #     而不是带着一个坏状态继续跑、最后崩成一个看不懂的 2754mm drift ✗
        _tr_chk = (replay or {}).get("trace") or {}
        try:
            _ot = np.asarray(_tr_chk.get("object_T", []), dtype=np.float64)
            _ba = np.asarray(_tr_chk.get("base_actual", []), dtype=np.float64)
            if _ot.ndim == 3 and len(_ot) > 1 and len(_ba) > 1:
                _kk = int(min(11, len(_ot), len(_ba)))
                _obj_disp = float(np.linalg.norm(_ot[-1][:3, 3] - _ot[-_kk][:3, 3]))
                _grp_disp = float(np.linalg.norm(_ba[-1] - _ba[-_kk]))
                logger.warning(
                    "[Stage3 抓持校验] 末 %d 个控制点: 物体位移=%.1fmm 夹爪位移=%.1fmm 跟随比=%.3f",
                    _kk, _obj_disp * 1000.0, _grp_disp * 1000.0,
                    (_obj_disp / _grp_disp) if _grp_disp > 1e-9 else float("nan"),
                )
                if _grp_disp > 0.005 and _obj_disp < 0.25 * _grp_disp:
                    _detail = (
                        f"HANDOFF_NO_GRASP: 末 {_kk} 个控制点里夹爪移动 "
                        f"{_grp_disp * 1000.0:.1f}mm, 而物体只移动 {_obj_disp * 1000.0:.1f}mm ✗ "
                        f"(跟随比 {_obj_disp / _grp_disp:.3f} < 0.25) ⇒ Stage2 末态不是真抓持"
                    )
                    logger.error("[Stage3] %s", _detail)
                    return {
                        "stage2_replay": replay,
                        "stage3a_gate": _empty_gate("HANDOFF_NO_GRASP", _detail),
                        "stage3b_gate": _empty_gate("NOT_RUN", "Stage3A prerequisite failed"),
                        "reward": -100.0,
                    }
        except Exception as _eg:  # noqa: BLE001
            logger.warning("[Stage3 抓持校验] 采集失败: %s", _eg)

        traj = self._mano_gripper_traj.get(side, {})
        mano_pos = np.asarray(traj.get("pos", []), dtype=np.float64)
        mano_R = np.asarray(traj.get("R", []), dtype=np.float64)
        offset = getattr(self, "_mano_neutral_offset", {}).get(side)
        if offset is not None:
            mano_pos = mano_pos + np.asarray(offset, dtype=np.float64)
        try:
            suffix = build_stage3_timeline(
                mano_pos,
                mano_R,
                parent,
                release_candidate=release_candidate,
                carry_time_scale=carry_time_scale,
                control_freq=CONTROL_FREQ,
            )
        except (KeyError, ValueError) as exc:
            detail = str(exc)
            return {
                "stage2_replay": replay,
                "stage3a_gate": _empty_gate("DATA", detail),
                "stage3b_gate": _empty_gate("NOT_RUN", "invalid Stage3 timeline"),
                "reward": -100.0,
            }

        ctrl = self.grasp_controllers.get(side) if self.grasp_controllers else None
        target_name = ctrl.target_obj if ctrl else None
        target_actor = self._find_target_actor(self.obj_actors, target_name)
        if target_actor is None:
            detail = f"target actor {target_name!r} not found"
            return {
                "stage2_replay": replay,
                "stage3a_gate": _empty_gate("DATA", detail),
                "stage3b_gate": _empty_gate("NOT_RUN", detail),
                "reward": -100.0,
            }

        robot = self.robot_info["robot"]
        virtual_idx = self.robot_info.get("virtual_idx", {})
        if not virtual_idx:
            detail = "Stage3 executable rollout requires gripper_only virtual joints"
            return {
                "stage2_replay": replay,
                "stage3a_gate": _empty_gate("REACH", detail),
                "stage3b_gate": _empty_gate("NOT_RUN", detail),
                "reward": -100.0,
            }
        gi1 = self.robot_info["gripper_idx1"]
        gi2 = self.robot_info["gripper_idx2"]
        arm_indices = self.robot_info.get("arm_joint_indices", [])
        active_joints = robot.get_active_joints()
        suffix_meta = suffix["timeline_meta"]

        _grip_preload = float(os.environ.get("_GRIP_PRELOAD", "0") or 0.0)
        # Stage3 起手平滑过渡帧数 (用户: "可以留 3-5 帧的平滑过渡")
        _s3_blend_frames = max(0, int(os.environ.get("_S3_BLEND_FRAMES", "4") or 0))
        trace = {
            key: []
            for key in (
                "control_index",
                "source_frame",
                "phase",
                "base_target",
                "base_actual",
                "gripper_q_cmd",
                "gripper_q_actual",
                "object_T",
                "gripper_T",
                "T_G_O",
                "finger1_contact",
                "finger2_contact",
                "target_bilateral",
                "contact_objects",
                "contact_links",
                "grasp_force",
                "illegal_contact",
            )
        }
        stage2_trace = replay["trace"]
        tgo_reference = np.asarray(stage2_trace["T_G_O"][-1], dtype=np.float64)
        object_reference = np.asarray(stage2_trace["object_T"][-1], dtype=np.float64)
        command_reference = np.asarray(parent["base_pos"][-1], dtype=np.float64)

        # ★★★★ 2026-09-14 姿态分支选择 (用户: "stage3 不就复用 stage2 跟随最后的参考轨迹吗
        #   ... 我认为你得看看跟随的轨迹对不对, 应该和放开这个帧的关系不大") ★★★★
        #   实测定位: **位置命令是对的** (后缀cmd↔重放末act = 0.5mm ✓), 问题在**姿态命令**口径不同:
        #     · Stage2 下发前逐点选"距上一命令最近 + 满足限位"的 ZYX 等价分支
        #       (并处理 ry→±180° 的万向节锁) ✓
        #     · Stage3 原先直接 rotmat_to_zyx_euler(target_R) 取**默认分支** ✗
        #       ⇒ 相邻命令可差 ~180° ⇒ virtual_rz 的 PD 目标瞬间翻转半圈 ⇒ 虚拟基座猛转
        #       ⇒ 被夹住的物体被甩脱 ⇒ 物体自由落体 (实测 z 0.1472→0.0969→0.0728→0.0387),
        #          夹爪却沿正确轨迹飞走 ⇒ T_G_O drift 684mm, 后缀第 11/53 点判 FAIL ✗
        #   修: 与 Stage2 **同一函数** (zyx_euler_continuous, 单一来源 ✓), 并以 Stage2 末帧
        #       姿态为起点续接, 保证"重放 → 后缀"这一刀也连续 ✓
        _prev_euler_s3 = None
        try:
            _parent_R = np.asarray(parent["base_R"], dtype=np.float64)
            if len(_parent_R):
                _prev_euler_s3 = rotmat_to_zyx_euler(_parent_R[-1])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Stage3 cmd] 父段末姿态取用失败(以默认起点续接): %s", exc)
        _cmd_euler_s3, _euler_s3_stats = zyx_euler_continuous(
            suffix["base_R"], prev_cmd=_prev_euler_s3
        )
        logger.info(
            "[Stage3 cmd] 姿态分支选择: rz [%.1f, %.1f] deg | 选中后相邻最大 rz 跳变 %.2f deg @ %d "
            "| 原始分解最大 rz 跳变 %.2f deg @ %d | 万向节锁帧 %d 个",
            _euler_s3_stats["rz_min_deg"], _euler_s3_stats["rz_max_deg"],
            _euler_s3_stats["pick_jump_max_deg"], _euler_s3_stats["pick_jump_idx"],
            _euler_s3_stats["raw_jump_max_deg"], _euler_s3_stats["raw_jump_idx"],
            len(_euler_s3_stats["gimbal_lock_indices"]),
        )

        # ★★★★ 2026-09-14 P0-2 起点核对 (用户: "继续") ★★★★
        #   已确认事实 (live4, 已排除 reset/重放 ✗ —— 有重放与无重放两条路径数值逐位相同 ✗):
        #     【接续前/后】物体z=0.1467 ✓ 夹爪-物距离=38.2mm ✓ 双侧=True ✓   ← 真实末态是握着的 ✓
        #     【后缀 idx=0】夹爪-物距离=289.3mm ✗ 物体z=0.0269 ✗            ← 一步内离开 250mm ✗
        #   一个控制步 (0.1s) 自由落体最多 49mm ✗, 实际掉 120mm ✗ ⇒ 不是物理滑脱 ✗, 是指令层暴冲 ✗
        #   本打印: 把**真实末态**(act + 手指 qpos)与 suffix 第 0 条命令并列, 一次分清:
        #     位置差/手指差 ≈ 0  ⇒ 命令起点正确 ✓, 暴冲出在执行阶段 (查索引/PD) ✗
        #     位置差或手指差很大 ⇒ 后缀 base_pos/gripper_q 的起点映射本身错 ✗
        try:
            _p0 = np.asarray(suffix["base_pos"][0], dtype=np.float64)
            _q0 = np.asarray(suffix["gripper_q"][0], dtype=np.float64)
            _qpos_now = np.asarray(robot.get_qpos(), dtype=np.float64)
            _act_now = np.asarray(
                [_qpos_now[virtual_idx[k]] for k in ("vx", "vy", "vz")], dtype=np.float64
            )
            _gz0, _gy0, _gx0 = _cmd_euler_s3[0]
            logger.warning(
                "[Stage3 起点核对] 真实末态 act=%s 手指qpos=%s | 后缀第0条 cmd=%s q=%s "
                "rz/ry/rx=%.1f/%.1f/%.1fdeg | 位置差=%.1fmm 手指差=%.4f",
                np.round(_act_now, 4).tolist(),
                np.round(_qpos_now[[gi1, gi2]], 4).tolist(),
                np.round(_p0, 4).tolist(),
                np.round(_q0, 4).tolist(),
                float(np.rad2deg(_gz0)), float(np.rad2deg(_gy0)), float(np.rad2deg(_gx0)),
                float(np.linalg.norm(_p0 - _act_now) * 1000.0),
                float(np.max(np.abs(_q0 - _qpos_now[[gi1, gi2]]))),
            )
        except Exception as _exc0:  # noqa: BLE001
            logger.warning("[Stage3 起点核对] 采集失败: %s", _exc0)

        # ★★★★ 2026-09-14 实验 A: 接续瞬间的**真实穿透深度** (用户: "之前可以 我不明白为什么突然炸开") ★★★★
        #   子步追踪已证实: 爆炸在**第 1 个物理子步 (1/240s)** 内完成 ✗
        #     sub=1 时物体已 105rad/s + 5m/s ✗, 基座 rz qvel=101.6rad/s ✗, 而手指还没张开 ✓
        #     ⇒ 不是累积 ✗ / 不是手指张开 ✗ / 不是物体掉落 ✗ ⇒ 只能是**单步接触冲量爆炸** ✗
        #   本探针: 量进入该步前的真实穿透. Stage2 末记录 max_pen=0.1041mm ✓ (很小)
        #     此处若大一个数量级 ⇒ 深穿透 ⇒ 真凶 ✓ (几何/尺度不匹配)
        #   同时列出全部接触对, 分辨"手指×物体"还是"手指×桌面/其它" ✓
        try:
            _cd_now = self._contact_detail_pairs(robot, side, self.scene)
            _cd_pos = [(a, b, round(float(c) * 1000.0, 4)) for a, b, c in _cd_now
                       if float(c) > 0.0]
            logger.warning(
                "[Stage3 穿透核对] 接续瞬间 夹爪接触对(link x body, 深度mm): %s | max=%.4fmm 对数=%d",
                _cd_pos, max((c for _, _, c in _cd_pos), default=0.0), len(_cd_pos),
            )
        except Exception as _exc_cd:  # noqa: BLE001
            logger.warning("[Stage3 穿透核对] 采集失败: %s", _exc_cd)

        # ★★★★ 2026-09-15 全场景接触审计 (用户: "继续看看 stage3") ★★★★
        #   盲点: `_contact_detail_pairs` **只列夹爪 link 的接触** ✗ ⇒ 从不显示「物体 × 桌面」✗
        #   ⇒ "手指→物体→支撑面"三体约束到底成不成立, 一直没有直接证据 ✗
        #   本打印把三类接触全列出来: target_contact / ground_contact / leak_contact ✓
        try:
            _aud_s3 = self._audit_contact_pairs(
                robot, side, self.scene, self.obj_actors, target_name)
            logger.warning(
                "[Stage3 接触审计] 接续瞬间 (target/ground/leak): %s",
                {k: str(v)[:240] for k, v in (_aud_s3 or {}).items()},
            )
        except Exception as _eaud:  # noqa: BLE001
            logger.warning("[Stage3 接触审计] 采集失败: %s", _eaud)

        # ★★★★ 2026-09-14 实验 A: 接续后"发 Stage2 自己的最后一条命令"是否也炸 ★★★★
        #   (用户: "你再测试一下, stage3的代码好好看一下")
        #   已排除: 惯性 ✗ 命令 ✗ (实际欧拉 51.1 vs 命令 51.0 ✓ 位置差 0.8mm ✓)
        #           q/索引 ✗ 驱动设置 ✗ 累积 ✗ 深穿透 ✗(0.1245mm ✓) PD 刚度 ✗(降到 5000 Stage2 就挂 ✓)
        #   剩下唯一未分的岔路:
        #     (a) 与**命令序列**有关 ⇒ 发 Stage2 最后一条命令应当**不炸** ✓
        #     (b) 与**"接续"这件事本身**有关 (函数返回后的衔接/状态) ⇒ 发同一条命令**照样炸** ✗
        #   本实验直接把 parent (Stage2 bundle) 的最后一条命令打出去, 观察一步后的速度 ✓
        if os.environ.get("_S3_REPLAY_LAST", "0") == "1":
            try:
                _pl = np.asarray(parent["base_pos"], dtype=np.float64)[-1]
                _qpl = np.asarray(parent["gripper_q"], dtype=np.float64)[-1]
                _epl = rotmat_to_zyx_euler(np.asarray(parent["base_R"], dtype=np.float64)[-1])
                _qp1 = np.asarray(robot.get_qpos(), dtype=np.float64)
                _qv1 = np.asarray(robot.get_qvel(), dtype=np.float64)
                _oz1 = float(target_actor.get_pose().p[2])
                logger.warning(
                    "[实验A] 先发 Stage2 最后一条命令 | pos=%s q=%s euler=%s | 当前 act=%s",
                    np.round(_pl, 4).tolist(), np.round(_qpl, 4).tolist(),
                    np.round(np.rad2deg(_epl), 2).tolist(),
                    np.round(_qp1[[virtual_idx[k] for k in ("vx", "vy", "vz")]], 4).tolist(),
                )
                for _k, _v in (("vx", _pl[0]), ("vy", _pl[1]), ("vz", _pl[2]),
                               ("rz", _epl[0]), ("ry", _epl[1]), ("rx", _epl[2])):
                    active_joints[virtual_idx[_k]].set_drive_target(float(_v))
                active_joints[gi1].set_drive_target(float(_qpl[0]))
                active_joints[gi2].set_drive_target(float(_qpl[1]))
                physics_step(robot, arm_indices, gi1, gi2, np.array([]),
                             float(_qpl[0]), float(_qpl[1]), self.scene)
                _qp2 = np.asarray(robot.get_qpos(), dtype=np.float64)
                _qv2 = np.asarray(robot.get_qvel(), dtype=np.float64)
                logger.warning(
                    "[实验A] 该步后 | 基座 rz qvel %.3f → %.3f | 手指 qpos %s → %s | 物体 z %.4f → %.4f",
                    float(_qv1[virtual_idx["rz"]]), float(_qv2[virtual_idx["rz"]]),
                    np.round(_qp1[[gi1, gi2]], 4).tolist(),
                    np.round(_qp2[[gi1, gi2]], 4).tolist(),
                    _oz1, float(target_actor.get_pose().p[2]),
                )
            except Exception as _ea:  # noqa: BLE001
                logger.warning("[实验A] 采集失败: %s", _ea)

        def is_target(name):
            return name == target_name or bool(target_name and target_name in name)

        def execute(index):
            target_pos = suffix["base_pos"][index]
            target_R = suffix["base_R"][index]
            # ★★★★ 2026-09-15 起手竖直余量 (用户: "继续看看 stage3, 可以有一点余量") ★★★★
            #   假设: 爆炸源是"手指 → 物体 → 支撑面"三体约束 ✗ ⇒ 先让物体离开支撑面几毫米 ✓
            #   做法: **纯轨迹竖直偏移** (smoothstep 渐入), 不动物理参数 ✓, 不改任何常量 ✓
            # ★★ 任务07 (2026-09-28) 【禁止隐藏的额外抬升】 ✓ ★★
            #   md §07 原话 ✓: "**禁止执行时额外加 8mm 高度**、换 force/K 或补未记录的 step;
            #     必要调整必须**先成为规划 bundle 的一部分**, 计入偏差和 hash" ✓
            #   实测旧病 ✗: 默认 `_S3_LIFT_MARGIN_M = 0.008` (恰好 8mm ✗) 且**只加在 execute() 里**
            #     ⇒ 执行时偷偷抬 8mm, 不进 bundle/偏差/hash ✗
            #   本改 ✓: 默认 **0**; 显式开启 ⇒ 打 ERROR 并**标记本 run 为诊断**(task_pass=False ✓)
            _lift_m = float(os.environ.get("_S3_LIFT_MARGIN_M", "0") or 0.0)
            if _lift_m > 0.0:
                logger.error("[任务07] ⚠ `_S3_LIFT_MARGIN_M=%.4fm` ⇒ md §07 **禁止**执行时额外加高 ✗ "
                             "⇒ 该调整应先成为规划 bundle 的一部分(计入偏差/hash) ✗; "
                             "本 run 强制记为**诊断**(task_pass=False ✓)", _lift_m)
                self._s3_diagnostic = True
            _lift_n = max(1, int(os.environ.get("_S3_LIFT_FRAMES", "6") or 1))
            if index < _lift_n and _lift_m > 0.0:
                target_pos = target_pos + np.array(
                    [0.0, 0.0, _lift_m * _smoothstep5((index + 1) / float(_lift_n))])
            # ★★★★ 2026-09-14 q 矛盾定位 (用户: "继续") ★★★★
            #   矛盾: Stage2 bundle 末帧 q=[0.0047,0.0047] ✓
            #         Stage3 代码 q_grasp = parent_q[-1,0] = 0.0047 ✓
            #         CARRY 相位用 q_grasp ✓ (build_stage3_timeline line 211)
            #         **但起手自检实测 cmd = 0.0225** ✗ (= _q_open_s3 ✗)
            #   本打印: 直接暴露 suffix["gripper_q"][index] 与相位 ✓
            if index < 6:
                # ★ 注意: q_grasp/_q_open_s3 是 build_stage3_timeline 的**局部变量** ✗,
                #   在 rollout_v4_stage3.execute() 里不可见 ⇒ 必须从 suffix meta 取 ✓
                _tm3 = suffix.get("timeline_meta", {}) or {}
                _qg3 = np.round(np.asarray(
                    _tm3.get("q_grasp", float("nan")), dtype=np.float64
                ).reshape(-1), 4).tolist()
                _qo3 = float(_tm3.get("q_open", float("nan")))
                logger.warning(
                    "[Stage3 q追踪] idx=%d phase=%-18s suffix_q=%s | meta.q_grasp=%s "
                    "meta.q_open=%.4f | 前帧q=%s",
                    index, str(suffix["phase"][index]),
                    np.round(np.asarray(suffix["gripper_q"][index], dtype=np.float64),
                             4).tolist(),
                    _qg3, _qo3,
                    (np.round(np.asarray(suffix["gripper_q"][index - 1], dtype=np.float64),
                              4).tolist() if index > 0 else "n/a"))
            # ★★★★ 2026-09-14 P0-2: 后缀起点 vs 重放终点 的命令对比 ★★★★
            #   已确认事实:
            #     【重放后】物体z=0.1487 ✓ 夹爪-物距离=37.9mm ✓ 双侧=True ✓  ← 完美
            #     【后缀 idx=0】物体z=0.0557 ✗ 夹爪-物距离=274.6mm ✗ 双侧=False ✗
            #     ⇒ 缺口 ≈ 236mm ✗ —— 后缀第一条命令就把夹爪**瞬移**走了 ✓
            #   本打印: 把后缀前 3 帧的命令位置与重放末帧的实际位置对比,
            #          一次分清"命令起点本身就错"还是"物理上被撞开":
            #     差 ≈ 236mm ⇒ 后缀 base_pos 的起点映射错了 ✗ (查 handoff 源帧)
            #     差 ≈ 0     ⇒ 命令没问题, 是物理碰撞把夹爪顶开了 ✗ (查碰撞)
            if index < 3:
                try:
                    _rb_p = np.asarray(robot.get_qpos(), dtype=np.float64)
                    _act = np.asarray([_rb_p[virtual_idx[k]] for k in ("vx", "vy", "vz")],
                                      dtype=np.float64)
                    _rep_T = (replay.get("trace", {}) or {}).get("base_actual")
                    _rep_last = (np.asarray(_rep_T[-1], dtype=np.float64)
                                 if _rep_T is not None and len(_rep_T) else None)
                    _d_cmd = (float(np.linalg.norm(target_pos - _rep_last) * 1000.0)
                              if _rep_last is not None else float("nan"))
                    _d_act = (float(np.linalg.norm(_act - _rep_last) * 1000.0)
                              if _rep_last is not None else float("nan"))
                    logger.warning(
                        "[Stage3 命令对比] idx=%d 后缀cmd=%s | 重放末act=%s | "
                        "cmd↔重放末=%.1fmm | 当前act↔重放末=%.1fmm | 当前act↔cmd=%.1fmm",
                        index, np.round(target_pos, 3).tolist(),
                        (np.round(_rep_last, 3).tolist() if _rep_last is not None else "n/a"),
                        _d_cmd, _d_act, float(np.linalg.norm(_act - target_pos) * 1000.0))
                except Exception as _e:  # noqa: BLE001
                    logger.warning("[Stage3 命令对比] 采集失败: %s", _e)
            q_cmd = np.asarray(suffix["gripper_q"][index], dtype=np.float64)
            # ★★★★ 2026-09-14 Stage3 起手 3~5 帧平滑过渡 (用户: "stage3 你得复用 stage2 好的,
            #   但是中间肯定不能完全的复现, 可以留 3-5 帧的平滑过渡") ★★★★
            #   复用 Stage2 的好逻辑: Stage2 的闭合是**多帧渐进**的 (stage1_close_q_sequence,
            #   close_controls 个控制点逐步收) ✓; 而 Stage3 后缀起手**一步到位** ✗ ——
            #   实测 Stage2 末手指被物体挡停在 act≈0.0156, 而后缀第 0 帧直接把命令给到
            #   q_grasp=0.0047 ⇒ 相当于要求手指在 1/30 秒内再夹进 ~11mm ✗ ⇒ **猛夹** ✗
            #   修: 前 N 帧把命令从**当前实际值**按 smoothstep 渐进收到目标值 ✓ (不动任何物理参数 ✓)
            if index < _s3_blend_frames:
                _q_now = np.asarray(robot.get_qpos(), dtype=np.float64)[[gi1, gi2]]
                _w = (index + 1) / float(_s3_blend_frames + 1)
                q_cmd = _q_now + (q_cmd - _q_now) * _smoothstep5(_w)
            # 夹持预紧 (2026-09-11, 与 Stage2 rollout_follow_trajectory 同一 _GRIP_PRELOAD 约定):
            # 用户要求"抓住之后抓紧一点, 持续给力让物体跟随夹爪". 携带段把手指命令再压低,
            # 使驱动器在物体挡停后仍持续施力; **释放段 (phase 含 RELEASE) 不施加**, 否则张不开.
            _phase_s = str(suffix["phase"][index]).upper()
            # 只在实际"握持"的相位施加预紧. 原写法 `"RELEASE" not in phase` 有漏洞:
            # 释放后的 POST_RELEASE_BLEND / MANO_FOLLOW 相位名里也不含 "RELEASE", 会被误加预紧,
            # 把已全开的命令 0.05 压到 0.042, 手指张不到位 -> Stage3B OPEN_TRACK 失败
            # (实测 final_q=[0.0544, 0.0396], 阈值 0.05-0.003=0.047).
            if _grip_preload > 0.0 and ("CARRY" in _phase_s or "PRE_RELEASE_SETTLE" in _phase_s):
                q_cmd = np.maximum(0.0, q_cmd - _grip_preload)
            # 与 Stage2 同源的等价分支选择 (见上方 _cmd_euler_s3): 杜绝相邻命令 180° 跳变
            rz, ry, rx = _cmd_euler_s3[index]
            for key, value in (
                ("vx", target_pos[0]),
                ("vy", target_pos[1]),
                ("vz", target_pos[2]),
                ("rz", rz),
                ("ry", ry),
                ("rx", rx),
            ):
                active_joints[virtual_idx[key]].set_drive_target(float(value))
            # ★★★★ 2026-09-14 P0-2 惯性探针 (用户: "惯性? 什么惯性呢? 好好测试一下把") ★★★★
            #   要测的是**速度量**, 不是位置量:
            #     · 虚拟关节 qvel (基座线速度/角速度) —— 若很大 ⇒ 基座在暴走 ✓; 若 ≈0 ⇒ 与惯性无关 ✗
            #     · 手指 qvel + drive 三件套 (target/k/d/force_limit)
            #         drive 完好但手指仍被崩开 ⇒ 有外力/惯性 ✓
            #         drive 缺失/被清零          ⇒ 是驱动问题 ✗ (与惯性无关 ✓)
            #     · 物体线/角速度 —— 区分"被抛飞" vs "被穿模弹开" ✓
            _probe = None
            if index < 2:
                try:
                    _qp_now = np.asarray(robot.get_qpos(), dtype=np.float64)
                    _qv_now = np.asarray(robot.get_qvel(), dtype=np.float64)
                    _Tg_now = self._gripper_pose_matrix(robot, side)
                    _e_now = (rotmat_to_zyx_euler(_Tg_now[:3, :3])
                              if _Tg_now is not None and np.isfinite(_Tg_now).all()
                              else (float("nan"),) * 3)
                    _ov = np.full(3, float("nan"))
                    _ow = np.full(3, float("nan"))
                    try:
                        for _c in target_actor.get_components():
                            if hasattr(_c, "get_linear_velocity"):
                                _ov = np.asarray(_c.get_linear_velocity(), dtype=np.float64)
                                _ow = np.asarray(_c.get_angular_velocity(), dtype=np.float64)
                                break
                    except Exception:  # noqa: BLE001
                        pass
                    _drive = []
                    for _gi in (gi1, gi2):
                        _j = active_joints[_gi]
                        _dd = {}
                        for _nm, _fn in (("target", "get_drive_target"),
                                         ("K", "get_drive_stiffness"),
                                         ("D", "get_drive_damping"),
                                         ("Fmax", "get_drive_force_limit")):
                            try:
                                _dd[_nm] = round(float(getattr(_j, _fn)()), 3)
                            except Exception:  # noqa: BLE001
                                _dd[_nm] = None
                        _drive.append(_dd)
                    _probe = {
                        "qpos": _qp_now, "obj_v": _ov, "obj_w": _ow,
                        "objpos_z": float(target_actor.get_pose().p[2]),
                    }
                    logger.warning(
                        "[惯性探针] idx=%d 步前 | 虚拟关节 qvel vx/vy/vz=%s rz/ry/rx=%s "
                        "| 夹爪实际欧拉=%.1f/%.1f/%.1f | 命令欧拉=%.1f/%.1f/%.1f "
                        "| 手指 qpos=%s qvel=%s | drive=%s | 物体线速度=%s 角速度=%s",
                        index,
                        np.round(_qv_now[[virtual_idx[k] for k in ("vx", "vy", "vz")]], 4).tolist(),
                        np.round(_qv_now[[virtual_idx[k] for k in ("rz", "ry", "rx")]], 4).tolist(),
                        float(np.rad2deg(_e_now[0])), float(np.rad2deg(_e_now[1])),
                        float(np.rad2deg(_e_now[2])),
                        float(np.rad2deg(rz)), float(np.rad2deg(ry)), float(np.rad2deg(rx)),
                        np.round(_qp_now[[gi1, gi2]], 4).tolist(),
                        np.round(_qv_now[[gi1, gi2]], 4).tolist(),
                        _drive,
                        np.round(_ov, 5).tolist(), np.round(_ow, 5).tolist(),
                    )
                except Exception as _ep:  # noqa: BLE001
                    logger.warning("[惯性探针] 步前采集失败: %s", _ep)
            # ★★★★ 2026-09-14 P0-2 对齐 Stage2 的驱动设置 (用户: "惯性? 什么惯性呢? 好好测试一下把") ★★★★
            #   已实测 (inert 日志, 见 [惯性探针]):
            #     步前 = Stage2 真实末态, 完全正常 ✓
            #       物体线速度 4.6cm/s ✓ 角速度 0.4rad/s ✓ 手指 qvel 0.004 ✓ drive target=0.005 ✓
            #       手指 act=[0.0127,0.0182] 与 Stage2 CARRY 末帧 finger_q act 逐位一致 ✓
            #       夹爪实际欧拉 51.1 vs 命令 51.0 (差 0.1°) ✓ 虚拟关节 qvel ≈0.02 ✓
            #     步后 = 爆炸 ✗
            #       手指 0.018→0.075 (Δ=62mm ✗) qvel 5.7m/s ✗
            #       物体 11m/s + 535rad/s ✗   虚拟关节 rz qvel 101rad/s ✗
            #   ⇒ 排除"惯性" ✗ 与"命令" ✗ ⇒ 差异只能在这**一次 physics_step 调用**本身.
            #   两段唯一差别: Stage2 (stage2_reconstruct_mixin:2152-2153) 在调用前**显式**设了
            #   gi1/gi2 的 drive target, Stage3 只依赖 physics_step 内部设置 ✗
            #   ⇒ 本实验: 抄回 Stage2 的写法, 若不再爆炸即定位 ✓
            active_joints[gi1].set_drive_target(float(q_cmd[0]))
            active_joints[gi2].set_drive_target(float(q_cmd[1]))
            # ★★★★ 2026-09-14 子步级定位 (用户: "之前可以 我不明白为什么会突然炸开 继续测试一下") ★★★★
            #   已实测 (inert 日志): idx=0 **一个** physics_step (8 子步) 内
            #     物体 4.6cm/s → 11m/s + 535rad/s ✗ ; 基座 rz qvel → 101rad/s ✗ ; 手指 → 机械限位 0.075 ✗
            #   而步前状态温和 (基座 1.2°/s ✓ 物体 0.4rad/s ✓ 命令/驱动校验全对 ✓)
            #   ⇒ 需要知道爆在**第几个子步**, 以区分:
            #       第 1 子步就爆 ⇒ 是一步之内的**单次接触事件** ✗ (穿透/摩擦冲量)
            #       第 7~8 子步才爆 ⇒ 是 8 步内**累积放大** ✗ (求解器不稳定)
            #   实现: idx=0 时改成 decimation=1 逐子步调用 (等价于一次 8 子步 ✓), 逐子步打印
            if index == 0:
                _dec_s3 = int(DECIMATION)
                for _sub in range(_dec_s3):
                    physics_step(
                        robot,
                        arm_indices,
                        gi1,
                        gi2,
                        np.array([]),
                        float(q_cmd[0]),
                        float(q_cmd[1]),
                        self.scene,
                        decimation=1,
                    )
                    _qs = np.asarray(robot.get_qpos(), dtype=np.float64)
                    _vs = np.asarray(robot.get_qvel(), dtype=np.float64)
                    _ovs = np.full(3, float("nan"))
                    _ows = np.full(3, float("nan"))
                    try:
                        for _c in target_actor.get_components():
                            if hasattr(_c, "get_linear_velocity"):
                                _ovs = np.asarray(_c.get_linear_velocity(), dtype=np.float64)
                                _ows = np.asarray(_c.get_angular_velocity(), dtype=np.float64)
                                break
                    except Exception:  # noqa: BLE001
                        pass
                    logger.warning(
                        "[子步追踪] idx=0 sub=%d/%d | 手指 qpos=%s qvel=%s | 物体 z=%.4f v=%s ω=%s "
                        "| 基座 rz qvel=%.3f | 夹爪欧拉=%.1f",
                        _sub + 1, _dec_s3,
                        np.round(_qs[[gi1, gi2]], 4).tolist(),
                        np.round(_vs[[gi1, gi2]], 3).tolist(),
                        float(target_actor.get_pose().p[2]),
                        np.round(_ovs, 3).tolist(), np.round(_ows, 3).tolist(),
                        float(_vs[virtual_idx["rz"]]),
                        float(np.rad2deg(rotmat_to_zyx_euler(
                            self._gripper_pose_matrix(robot, side)[:3, :3])[0])),
                    )
                    # 第 1 子步之后立刻看接触穿透: 判定"爆炸是否由超深穿透触发" ✓
                    if _sub == 0:
                        try:
                            _cd1 = self._contact_detail_pairs(robot, side, self.scene)
                            logger.warning(
                                "[子步追踪] sub=1 接触明细(link x body, 深度mm): %s",
                                [(a, b, round(float(c) * 1000.0, 4)) for a, b, c in _cd1
                                 if float(c) > 0.0] or "无(深度>0)接触对",
                            )
                        except Exception as _ecd:  # noqa: BLE001
                            logger.warning("[子步追踪] 接触明细采集失败: %s", _ecd)
                        # ★★★ 最后两项: 广义力 qf 与 真实臂关节速度 ★★★
                        #   A) physics_step 里 qf[:6]=0 **只在 floating-root 分支**执行 ✗,
                        #      gripper_only 走 lock_root_pose=None ⇒ **不清零** ⇒ 虚拟关节可能每子步
                        #      被写入重力/科氏力 ✗
                        #   B) arm_target=np.array([]) 是空的 ⇒ **真实臂关节一个都不设驱动** ✗,
                        #      它们是自由/未驱动的, 而此前从没看过它们的速度 ✗
                        try:
                            _qf = np.asarray(robot.get_qf(), dtype=np.float64)
                            _vi6 = [virtual_idx[k] for k in ("vx", "vy", "vz", "rz", "ry", "rx")]
                            _qv_arm = np.asarray(_vs, dtype=np.float64)
                            logger.warning(
                                "[子步追踪] sub=1 qf: 虚拟关节6项=%s | qf前6项=%s | max|qf|=%.3f @idx%d "
                                "| 真实臂关节 qvel=%s | 臂索引=%s",
                                np.round(_qf[_vi6], 3).tolist(),
                                np.round(_qf[:6], 3).tolist(),
                                float(np.nanmax(np.abs(_qf))), int(np.nanargmax(np.abs(_qf))),
                                (np.round(_qv_arm[arm_indices], 3).tolist()
                                 if len(arm_indices) else "无"),
                                list(arm_indices),
                            )
                        except Exception as _eqf:  # noqa: BLE001
                            logger.warning("[子步追踪] qf/臂速度采集失败: %s", _eqf)
            else:
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

            # ★★★★ 2026-09-14 惯性探针 步后 (与步前对照) ★★★★
            if _probe is not None:
                try:
                    _qp_post = np.asarray(robot.get_qpos(), dtype=np.float64)
                    _qv_post = np.asarray(robot.get_qvel(), dtype=np.float64)
                    _ovp = np.full(3, float("nan"))
                    _owp = np.full(3, float("nan"))
                    try:
                        for _c in target_actor.get_components():
                            if hasattr(_c, "get_linear_velocity"):
                                _ovp = np.asarray(_c.get_linear_velocity(), dtype=np.float64)
                                _owp = np.asarray(_c.get_angular_velocity(), dtype=np.float64)
                                break
                    except Exception:  # noqa: BLE001
                        pass
                    logger.warning(
                        "[惯性探针] idx=%d 步后 | 手指 qpos %s→%s (Δ=%.4f) qvel=%s "
                        "| 物体 z %.4f→%.4f 线速度 %s→%s 角速度→%s "
                        "| 虚拟关节 qvel vx/vy/vz=%s rz/ry/rx=%s",
                        index,
                        np.round(_probe["qpos"][[gi1, gi2]], 4).tolist(),
                        np.round(_qp_post[[gi1, gi2]], 4).tolist(),
                        float(np.max(np.abs(_qp_post[[gi1, gi2]] - _probe["qpos"][[gi1, gi2]]))),
                        np.round(_qv_post[[gi1, gi2]], 4).tolist(),
                        float(_probe["objpos_z"]),
                        float(target_actor.get_pose().p[2]),
                        np.round(_probe["obj_v"], 5).tolist(),
                        np.round(_ovp, 5).tolist(),
                        np.round(_owp, 5).tolist(),
                        np.round(_qv_post[[virtual_idx[k] for k in ("vx", "vy", "vz")]], 4).tolist(),
                        np.round(_qv_post[[virtual_idx[k] for k in ("rz", "ry", "rx")]], 4).tolist(),
                    )
                    # 步后 drive target: 判定"目标是否被别的东西覆盖/清零" ✗
                    _dt_post = []
                    for _gi in (gi1, gi2):
                        try:
                            _dt_post.append(round(float(active_joints[_gi].get_drive_target()), 4))
                        except Exception:  # noqa: BLE001
                            _dt_post.append(None)
                    logger.warning(
                        "[惯性探针] idx=%d 步后 drive target=%s (要求=%s) | 虚拟关节 drive=%s",
                        index, _dt_post, np.round(q_cmd, 4).tolist(),
                        [round(float(active_joints[virtual_idx[k]].get_drive_target()), 4)
                         for k in ("vx", "vy", "vz", "rz", "ry", "rx")],
                    )
                except Exception as _ep2:  # noqa: BLE001
                    logger.warning("[惯性探针] 步后采集失败: %s", _ep2)

            qpos = np.asarray(robot.get_qpos(), dtype=np.float64)
            actual_pos = np.asarray(
                [qpos[virtual_idx[key]] for key in ("vx", "vy", "vz")], dtype=np.float64
            )
            T_W_G = self._gripper_pose_matrix(robot, side)
            T_W_O = np.asarray(target_actor.get_pose().to_transformation_matrix(), dtype=np.float64)
            f1, f2, contact_objects, contact_links = self._target_finger_contacts(
                robot, side, self.scene, self.obj_actors, target_name
            )
            target_contact = any(is_target(name) for name in contact_objects)
            bilateral = bool(f1 and f2 and target_contact)
            illegal = any(not is_target(name) for name in contact_objects)
            T_G_O = _relative_pose(T_W_G, T_W_O) if T_W_G is not None else np.full((4, 4), np.nan)

            trace["control_index"].append(index)
            trace["source_frame"].append(float(suffix["source_frame"][index]))
            trace["phase"].append(str(suffix["phase"][index]))
            trace["base_target"].append(target_pos.copy())
            trace["base_actual"].append(actual_pos)
            trace["gripper_q_cmd"].append(q_cmd.copy())
            trace["gripper_q_actual"].append(qpos[[gi1, gi2]].copy())
            trace["object_T"].append(T_W_O)
            trace["gripper_T"].append(T_W_G if T_W_G is not None else np.full((4, 4), np.nan))
            trace["T_G_O"].append(T_G_O)
            trace["finger1_contact"].append(bool(f1 and target_contact))
            trace["finger2_contact"].append(bool(f2 and target_contact))
            trace["target_bilateral"].append(bilateral)
            trace["contact_objects"].append(tuple(sorted(contact_objects)))
            trace["contact_links"].append(tuple(sorted(contact_links)))
            trace["grasp_force"].append(float(get_grasp_force(side, self.scene, self.obj_actors, robot)))
            trace["illegal_contact"].append(bool(illegal))
            # ★★★★ 2026-09-14 起手自检 (用户: "为什么 stage3 不行, stage2 可以呢?") ★★★★
            #   背景: Stage2 的 CARRY 完美 (rot=0.11° / dist2obj 恒定 37mm ✓ = 物体被夹住 ✓),
            #         但 Stage3 的 A 在后缀里 **第 11/53 个控制点就判 T_G_O 失败**
            #         (drift 实测 1520/1333/2233mm ⇒ 物体根本没跟着走 ✗)。
            #   需要一次分清三种可能:
            #     (A) Stage3 replay 起手时**夹持就没建立** → 物体还在桌上, 夹爪飞走了 ✗
            #     (B) 起手正常, 但前若干帧内夹持被**打断** → 命令速度/姿态跳变 ✗
            #     (C) 起手正常且前段平稳, 之后才崩 → 相位切换处问题 ✗
            #   做法: 后缀开头若干帧逐帧打印关键量; 末尾再汇总"夹持是否成立/何时断"。
            _s3_watch = int(os.environ.get("_STAGE3_WATCH_FRAMES", "12"))
            if index < _s3_watch:
                _obj_z = float(T_W_O[2, 3])
                _gT_o = np.linalg.norm(
                    T_W_G[:3, 3] - T_W_O[:3, 3]) if np.isfinite(T_W_G).all() else float("nan")
                logger.warning(
                    "[Stage3 起手自检] idx=%2d phase=%-18s 双侧=%-5s 双侧指接触=%s/%s "
                    "物体z=%.4f 夹爪-物距离=%.1fmm T_G_O位移=%.2fmm 物体位移=%.2fmm",
                    index, str(suffix["phase"][index]), str(bilateral),
                    str(bool(f1 and target_contact)), str(bool(f2 and target_contact)),
                    _obj_z, _gT_o * 1000.0,
                    float(np.linalg.norm(T_G_O[:3, 3])) * 1000.0,
                    float(np.linalg.norm(T_W_O[:3, 3] - np.asarray(
                        suffix.get("object_init_pos", T_W_O[:3, 3]))) ) * 1000.0
                    if "object_init_pos" in suffix else -1.0,
                )
            if record_video:
                self._record_stage3_frame(suffix, index)

        def carry_metrics(end_index):
            from physics_utils import compute_tracking_error

            upto = end_index + 1
            tgo = list(trace["T_G_O"][:upto])
            t_err, r_err = _relative_pose_errors(tgo_reference, tgo)
            bilateral = np.asarray(trace["target_bilateral"][:upto], dtype=np.bool_)
            tracking = compute_tracking_error(
                np.asarray(trace["base_actual"][:upto]), np.asarray(trace["base_target"][:upto])
            )
            command_delta = np.asarray(trace["base_target"][end_index]) - command_reference
            object_delta = np.asarray(trace["object_T"][end_index])[:3, 3] - object_reference[:3, 3]
            command_motion = float(np.linalg.norm(command_delta))
            progress_ratio = (
                float(np.dot(object_delta, command_delta) / (command_motion**2))
                if command_motion >= STAGE3_MIN_TASK_MOTION_M
                else 1.0
            )
            return {
                "tgo_translation_max_mm": float(np.max(t_err) * 1000.0) if len(t_err) else float("inf"),
                "tgo_rotation_max_deg": float(np.rad2deg(np.max(r_err))) if len(r_err) else float("inf"),
                "bilateral_ratio": float(np.mean(bilateral)) if len(bilateral) else 0.0,
                "command_motion_mm": command_motion * 1000.0,
                "object_progress_ratio": progress_ratio,
                "tracking_mean_mm": float(tracking["mean_mm"]),
                "tracking_max_mm": float(tracking["max_mm"]),
                "illegal_contact_count": int(sum(trace["illegal_contact"][:upto])),
            }

        def carry_failure(metrics, require_motion=False):
            if metrics["illegal_contact_count"]:
                return "COLLISION", f"illegal finger/environment contacts={metrics['illegal_contact_count']}"
            if metrics["tgo_translation_max_mm"] > TGO_TRANSLATION_TOL_M * 1000.0:
                return "T_G_O", f"translation drift={metrics['tgo_translation_max_mm']:.2f}mm"
            if metrics["tgo_rotation_max_deg"] > np.rad2deg(TGO_ROTATION_TOL_RAD):
                return "T_G_O", f"rotation drift={metrics['tgo_rotation_max_deg']:.2f}deg"
            if metrics["bilateral_ratio"] < STAGE3_BILATERAL_RATIO_MIN:
                return "CONTACT", f"bilateral ratio={metrics['bilateral_ratio']:.3f}"
            if require_motion and metrics["command_motion_mm"] < STAGE3_MIN_TASK_MOTION_M * 1000.0:
                return "TASK_PROGRESS", "MANO F62->release task motion is below 5mm"
            if metrics["object_progress_ratio"] < STAGE3_TASK_PROGRESS_MIN:
                return "TASK_PROGRESS", f"object progress ratio={metrics['object_progress_ratio']:.3f}"
            if (
                not np.isfinite(metrics["tracking_mean_mm"])
                or metrics["tracking_mean_mm"] > TRACKING_MEAN_TOL_MM
                or metrics["tracking_max_mm"] > TRACKING_MAX_TOL_MM
            ):
                return (
                    "TRACK",
                    f"tracking mean/max={metrics['tracking_mean_mm']:.2f}/"
                    f"{metrics['tracking_max_mm']:.2f}mm",
                )
            return None

        stage3a_gate = _empty_gate("NOT_RUN")
        milestone_metrics = {}
        next_milestones = sorted(
            {
                frame
                for frame in (70, 80, int(release_candidate))
                if parent_source_end(parent) < frame <= int(release_candidate)
            }
        )
        failed = None
        a_last = suffix_meta["pre_release_end"]
        for index in range(a_last + 1):
            execute(index)
            src = float(suffix["source_frame"][index])
            while next_milestones and src + 1e-9 >= next_milestones[0]:
                milestone = next_milestones.pop(0)
                metrics = carry_metrics(index)
                milestone_metrics[f"F{milestone}"] = metrics
                failure = carry_failure(metrics, require_motion=(milestone == release_candidate))
                if failure is not None:
                    failed = failure
                    break
            if failed is not None:
                break

        if failed is None and len(trace["source_frame"]) == a_last + 1:
            final_metrics = carry_metrics(a_last)
            # ★★★ 2026-09-14 用户: "stage1 2 3 都要逐帧 csv 记录" ★★★
            #   列定义与 Stage1 (gripper_trace.csv) **完全一致** ✓, 额外加 phase/bilateral/force
            #   目的: 三段可拼成一张表对比 ✓ (位姿 / 开合 / 物体 / 接触 一次看全 ✓)
            try:
                _s3_od = getattr(self, "output_dir", None) or os.environ.get("_OUTPUT_DIR")
                if _s3_od:
                    _s3_dir = os.path.join(str(_s3_od), "stage3")
                    os.makedirs(_s3_dir, exist_ok=True)
                    _s3_csv = os.path.join(_s3_dir, "gripper_trace.csv")
                    _n3 = len(trace.get("control_index", []))
                    _up_n = 0
                    with open(_s3_csv, "w", encoding="utf-8") as _fh3:
                        _fh3.write(
                            "frame,phase,pos_x,pos_y,pos_z,euler_rx_deg,euler_ry_deg,"
                            "euler_rz_deg,x_down_dot,finger1_q,finger2_q,finger_gap_m,"
                            "obj_z,bilateral,grasp_force_n,tgo_translation_mm\n")
                        for _i in range(_n3):
                            _R3 = np.asarray(trace["gripper_T"][_i], dtype=np.float64)
                            _R3r = _R3[:3, :3] if np.isfinite(_R3).all() else np.eye(3)
                            _e3z, _e3y, _e3x = rotmat_to_zyx_euler(_R3r)
                            _d3 = float(np.dot(_R3r[:, 0], np.array([0.0, 0.0, -1.0])))
                            if _d3 < 0.0:
                                _up_n += 1
                            _q3 = np.asarray(trace["gripper_q_actual"][_i], dtype=np.float64)
                            _t3 = np.asarray(trace["T_G_O"][_i], dtype=np.float64)
                            _t3d = (float(np.linalg.norm(_t3[:3, 3]) * 1000.0)
                                    if np.isfinite(_t3).all() else float("nan"))
                            _fh3.write(",".join(str(_v) for _v in (
                                int(trace["control_index"][_i]), trace["phase"][_i],
                                *[f"{float(_v):.6f}" for _v in trace["base_actual"][_i]],
                                f"{np.rad2deg(_e3x):.6f}", f"{np.rad2deg(_e3y):.6f}",
                                f"{np.rad2deg(_e3z):.6f}", f"{_d3:.6f}",
                                f"{float(_q3[0]):.6f}", f"{float(_q3[1]):.6f}",
                                f"{FINGER_GAP0 + 2.0 * float(_q3[0]):.6f}",
                                f"{float(np.asarray(trace['object_T'][_i])[2, 3]):.6f}",
                                str(bool(trace["target_bilateral"][_i])),
                                f"{float(trace['grasp_force'][_i]):.4f}", f"{_t3d:.4f}",
                            )) + "\n")
                    logger.info(
                        "[Stage3 CSV] 逐帧位姿+开合已存: %s (%d 帧) | X 朝上(dot<0) 帧数=%d %s",
                        _s3_csv, _n3, _up_n,
                        "← 有帧夹爪朝上 ✗" if _up_n else "✓ 全程朝下")
            except Exception as _e3:  # noqa: BLE001
                logger.warning("[Stage3 CSV] 写出失败: %s", _e3)
            final_metrics["mano_fidelity"] = dict(suffix["mano_fidelity"])
            failure = carry_failure(final_metrics, require_motion=True)
            if failure is None:
                stage3a_gate = {
                    "passed": True,
                    "failure": None,
                    "failure_detail": "",
                    "metrics": final_metrics,
                    "milestones": milestone_metrics,
                }
            else:
                failed = failure
        if failed is not None:
            stage3a_gate = {
                "passed": False,
                "failure": failed[0],
                "failure_detail": failed[1],
                "metrics": carry_metrics(len(trace["source_frame"]) - 1),
                "milestones": milestone_metrics,
            }

        stage3b_gate = _empty_gate("NOT_RUN", "Stage3A failed")
        support_before_release = {}
        if stage3a_gate["passed"] and enable_release:
            support_before_release = self._support_ready(
                robot, target_actor, trace["object_T"], target_name
            )
            # 2026-09-11 (用户定策): 删除 NO_SUPPORT 释放前置阻断 — "该释放就释放, 这是正常现象".
            # 原条件 = 物体"有支撑且慢"; 但 release_candidate=F90 处物体正被夹爪握持悬在空中,
            # 该条件永不成真, 导致手指永远不张开 (实测 Stage3A PASS, Stage3B 被拦在 NO_SUPPORT).
            # 支撑/速度仍完整记录 (support_before_release 与 b_metrics), 仅不再作为前置条件.
            if not True:  # 前置条件已废止; 保留分支体以便回滚对照
                stage3b_gate = _empty_gate(
                    "NO_SUPPORT",
                    "release blocked: object must be supported and slow",
                )
                stage3b_gate["metrics"] = support_before_release
            else:
                pre_release_object_T = np.asarray(trace["object_T"][-1]).copy()
                for index in range(suffix_meta["release_start"], len(suffix["source_frame"])):
                    execute(index)
                support_after = self._support_ready(robot, target_actor, trace["object_T"], target_name)
                release_slice = slice(suffix_meta["release_settle_start"], None)
                released_positions = np.asarray(trace["object_T"])[release_slice, :3, 3]
                release_drift = (
                    float(np.max(np.linalg.norm(released_positions - pre_release_object_T[:3, 3], axis=1)))
                    if len(released_positions)
                    else float("inf")
                )
                final_q = np.asarray(trace["gripper_q_actual"][-1])
                q_open = float(suffix_meta["q_open"])
                final_finger_contact = bool(
                    trace["finger1_contact"][-1] or trace["finger2_contact"][-1]
                )
                # 末帧相对"释放前参考位姿"的分离距离: 物体是否已经离开夹爪 (2026-09-12)
                _tgo_end = np.asarray(trace["T_G_O"][-1], dtype=np.float64)
                _tgo_ref = np.asarray(tgo_reference, dtype=np.float64)
                _separation_m = float(
                    np.linalg.norm(_tgo_end[:3, 3] - _tgo_ref[:3, 3])
                ) if np.isfinite(_tgo_end).all() else 0.0
                b_metrics = {
                    **support_after,
                    "release_drift_mm": release_drift * 1000.0,
                    "final_q_min": float(np.min(final_q)),
                    "final_target_finger_contact": final_finger_contact,
                }
                # 2026-09-11 (用户定策): 释放后不再要求"物体有支撑且静止"、也不再限制释放后位移.
                # 原因: F90 释放点在空中 (物体被夹爪握持悬空), 松手后物体自然下落 —
                #   支撑/速度/位移三项必然"失败", 但它们否定的是物理必然现象, 不是释放动作本身.
                # 保留为 b_metrics 观察量; Stage3B 真正判定的仍是"手指是否确实张开且不再接触物体".
                _ = (support_after, release_drift)  # 记录用, 不参与判定
                # 2026-09-12 (用户定策 A): OPEN_TRACK 从"命令跟踪"改为**物理间隙判据**.
                # 原因(实测+推算): 手指 PD 的 τ = D/K = 200/1000 = 0.2s, 开爪给 5 帧(0.167s)
                #   只能走 1-e^(-0.167/0.2)=56% 行程 → 实测 final_q=0.0394 (预测 0.0390, 吻合).
                #   即"跟上命令"(≥q_open-3mm)与"退避≤5帧"在数学上直接冲突.
                # 而真正该判的是"物体能不能掉出来" = 指缝宽度 > 物体宽度 + 余量:
                #   指缝宽度 = FINGER_BASE_DIST + 2q;  实测 q=0.0334 → 93.7mm >> 物体 40mm ✓
                from physics_utils import FINGER_BASE_DIST as _FBD
                _bb = np.asarray(
                    getattr(self, "obj_info", {}).get(target_name, {}).get("bbox_size", []),
                    dtype=np.float64,
                )
                _obj_w = float(np.min(_bb[:2])) if _bb.size >= 2 and np.isfinite(_bb[:2]).all() else 0.0
                _gap = float(_FBD) + 2.0 * float(np.min(final_q))
                _gap_need = _obj_w + STAGE3_OPEN_CLEARANCE_M
                if _gap < _gap_need:
                    stage3b_gate = _empty_gate(
                        "OPEN_TRACK",
                        f"指缝间隙不足: gap={_gap * 1000:.1f}mm < 需要 {_gap_need * 1000:.1f}mm "
                        f"(物体宽 {_obj_w * 1000:.1f}mm, final q={final_q.tolist()})",
                    )
                elif final_finger_contact and _separation_m < 0.005:
                    # 2026-09-12 (用户定策): 只判"最后一帧是否还有手指接触"过严 ——
                    # 退避仅 2 帧, 夹爪尚未移开, 物体刚松手下落时可能擦到手指, 但它其实已自由.
                    # 补充"分离距离"判据: 末帧 T_G_O 相对释放前参考的运动量 > 5mm
                    # 即说明物体已离开夹爪(在下落), 此时允许存在瞬时擦碰.
                    stage3b_gate = _empty_gate(
                        "NOT_RELEASED",
                        f"target still touches a finger (分离仅 {_separation_m * 1000:.1f}mm)",
                    )
                else:
                    stage3b_gate = {"passed": True, "failure": None, "failure_detail": "", "metrics": {}}
                stage3b_gate["metrics"] = b_metrics
        elif stage3a_gate["passed"]:
            stage3b_gate = _empty_gate("DISABLED", "release disabled by caller")

        trace = self._as_trace_arrays(trace)
        executed_combined = combine_stage2_stage3(parent, suffix, len(trace["source_frame"]))
        result = {
            "stage2_replay": replay,
            "suffix_bundle": suffix,
            "bundle": executed_combined,
            "trace": trace,
            "stage3a_gate": stage3a_gate,
            "stage3b_gate": stage3b_gate,
            "gate_result": stage3a_gate,
            "support_before_release": support_before_release,
        }
        stage2_object = np.asarray(stage2_trace["object_T"])
        all_object = np.concatenate([stage2_object, trace["object_T"]], axis=0)
        result["obj_lift"] = float(np.max(all_object[:, 2, 3]) - all_object[0, 2, 3])
        result["obj_lift_final"] = float(all_object[-1, 2, 3] - all_object[0, 2, 3])
        result["xy_drift"] = float(np.linalg.norm(all_object[-1, :2, 3] - all_object[0, :2, 3]))
        result["carry_validation"] = {
            "passed": bool(stage3a_gate["passed"]),
            "max_dev_mm": stage3a_gate.get("metrics", {}).get(
                "tgo_translation_max_mm", float("nan")
            ),
        }
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★ 任务07 (2026-09-28) 【总门 task_pass: 合取, save 必须用它】 ✓ ★★★★
        #   md §07 原话 ✓: "`task_pass` = **数据有效 ∧ artifact 有效 ∧ 父段重放通过 ∧
        #     剩余搬运通过 ∧（未启用释放 或 释放通过）∧ 非诊断**" ✓;
        #     "`save` 使用**同一总门**, **不能 Stage3A 单独通过就保存为正式成功**" ✓
        #   实测旧病 ✗: 无 `task_pass`; `carry_validation["passed"]` 只等于 Stage3A ✗
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        _diag07 = bool(getattr(self, "_s3_diagnostic", False)
                       or str(os.environ.get("_S3_DIAGNOSTIC", "0")) == "1")
        _data_ok07 = bool(result.get("data_valid", True))
        _art_ok07 = bool(getattr(self, "_s3_artifact_valid", True))
        _parent_ok07 = bool((replay or {}).get("passed", bool(replay)))
        _carry_ok07 = bool(stage3a_gate.get("passed"))
        _release_ok07 = (not bool(enable_release)) or bool(stage3b_gate.get("passed"))
        result["task_pass_terms"] = {
            "data_valid": _data_ok07, "artifact_valid": _art_ok07,
            "parent_replay_passed": _parent_ok07, "carry_passed": _carry_ok07,
            "release_ok": _release_ok07, "diagnostic": _diag07,
        }
        result["task_pass"] = bool(_data_ok07 and _art_ok07 and _parent_ok07
                                   and _carry_ok07 and _release_ok07 and not _diag07)
        logger.warning("[任务07] ✓ task_pass=%s | 分项=%s", result["task_pass"],
                       result["task_pass_terms"])
        result["reward"] = self.stage3_reward(result)
        logger.info(
            "[Stage3] A=%s B=%s parent=%s suffix=%s controls=%d/%d",
            "PASS" if stage3a_gate["passed"] else f"FAIL({stage3a_gate['failure']})",
            "PASS" if stage3b_gate["passed"] else f"FAIL({stage3b_gate['failure']})",
            str(parent["bundle_hash"])[:12],
            str(suffix["bundle_hash"])[:12],
            len(trace["source_frame"]),
            len(suffix["source_frame"]),
        )
        return result

    def _record_stage3_frame(self, suffix, index):
        """Render an already executed control; never advances physics."""
        try:
            if not getattr(self.scene, "_render_available", False):
                return
            import cv2

            self._update_collision_visualization()
            self.scene.update_render()
            # 2026-09-12: 逐帧同步第一人称 cam 相机 (同 Stage2/主循环行为).
            # 此前只 take_picture(), 相机冻结 → cam 视频看起来像 god 视角.
            self._sync_cam_view_to_frame(suffix["source_frame"][index])
            label = (
                f"src F{suffix['source_frame'][index]:.2f} | "
                f"{suffix['phase'][index]}"
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
                bgr = np.ascontiguousarray(
                    (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[..., ::-1]
                )
                cv2.putText(
                    bgr, label, (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2, cv2.LINE_AA,
                )
                writer.write(bgr)
        except Exception as exc:
            logger.warning("[Stage3 render] frame %d skipped: %s", index, exc)

    @staticmethod
    def save_stage3_bundle(output_path, result) -> bool:
        """★ 任务07 (2026-09-28) ✓ md §07: "`save` 使用**同一总门**, **不能 Stage3A 单独通过
        就保存为正式成功**" ✓

        实测旧病 ✗: 入口只检查 `stage3a_gate["passed"]` ⇒ Stage3A 单独 PASS 即落盘为**正式**
          产物 ✗ (与"剩余搬运/释放/父段重放/诊断"全都无关 ✗)
        本改 ✓: 必须先满足 `task_pass` (数据有效 ∧ artifact 有效 ∧ 父段重放通过 ∧
          剩余搬运通过 ∧(未启用释放 或 释放通过) ∧ **非诊断** ✓);
          ⚠ `task_pass` 缺失(旧调用) ⇒ 回退口径 + **显式告警** (不静默放过 ✗)
        """
        gate_a = result.get("stage3a_gate", {}) or {}     # ★ 保留原名 (函数后段仍引用 ✓)
        _tp07 = result.get("task_pass")
        if _tp07 is None:
            _ga07 = result.get("stage3a_gate", {}) or {}
            _gb07 = result.get("stage3b_gate", {}) or {}
            _er07 = bool(result.get("enable_release", True))
            _tp07 = bool(_ga07.get("passed")) and ((not _er07) or bool(_gb07.get("passed")))
            logger.warning("[任务07] ⚠ result 缺 `task_pass` ⇒ 用回退口径 (Stage3A ∧ 释放)=%s 判定 ✗ "
                           "(请让调用方传 `task_pass` ✓)", _tp07)
        if not _tp07:
            logger.warning("[Stage3 save] ✗ 总门 `task_pass=False` ⇒ **拒绝保存为正式成功** ✗ "
                           "(md §07: 不能 Stage3A 单独通过就保存 ✓) | 分项=%s",
                           result.get("task_pass_terms"))
            return False
        bundle = result["bundle"]
        trace = result.get("trace", {})
        replay_trace = result.get("stage2_replay", {}).get("trace", {})
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
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
            parent_bundle_hash=np.asarray(bundle["parent_bundle_hash"]),
            stage3_suffix_hash=np.asarray(bundle["stage3_suffix_hash"]),
            bundle_hash=np.asarray(bundle["bundle_hash"]),
            stage1_contract_hash=np.asarray(bundle.get("stage1_contract_hash", "")),
            physics_hash=np.asarray(bundle.get("physics_hash", "")),
            manifest_hash=np.asarray(bundle.get("manifest_hash", "")),
            timeline_meta_json=np.asarray(
                json.dumps(bundle.get("timeline_meta", {}), sort_keys=True, default=_json_default)
            ),
            mano_fidelity_json=np.asarray(
                json.dumps(bundle.get("mano_fidelity", {}), sort_keys=True, default=_json_default)
            ),
            stage3a_gate_json=np.asarray(json.dumps(gate_a, sort_keys=True, default=_json_default)),
            stage3b_gate_json=np.asarray(
                json.dumps(result.get("stage3b_gate", {}), sort_keys=True, default=_json_default)
            ),
            stage2_trace_object_T=np.asarray(replay_trace.get("object_T", []), dtype=np.float64),
            stage3_trace_source_frame=np.asarray(trace.get("source_frame", []), dtype=np.float64),
            stage3_trace_phase=np.asarray(trace.get("phase", []), dtype="U24"),
            stage3_trace_base_target=np.asarray(trace.get("base_target", []), dtype=np.float64),
            stage3_trace_base_actual=np.asarray(trace.get("base_actual", []), dtype=np.float64),
            stage3_trace_gripper_q_cmd=np.asarray(trace.get("gripper_q_cmd", []), dtype=np.float64),
            stage3_trace_gripper_q_actual=np.asarray(trace.get("gripper_q_actual", []), dtype=np.float64),
            stage3_trace_object_T=np.asarray(trace.get("object_T", []), dtype=np.float64),
            stage3_trace_gripper_T=np.asarray(trace.get("gripper_T", []), dtype=np.float64),
            stage3_trace_T_G_O=np.asarray(trace.get("T_G_O", []), dtype=np.float64),
            stage3_trace_target_bilateral=np.asarray(
                trace.get("target_bilateral", []), dtype=np.bool_
            ),
            stage3_trace_illegal_contact=np.asarray(trace.get("illegal_contact", []), dtype=np.bool_),
            stage3_trace_contact_objects_json=np.asarray(
                json.dumps([list(names) for names in trace.get("contact_objects", [])])
            ),
            stage3_trace_contact_links_json=np.asarray(
                json.dumps([list(names) for names in trace.get("contact_links", [])])
            ),
        )
        logger.info("[Stage3 save] combined executable bundle saved: %s", output_path)
        return True

    def cem_stage3_optimize(self, side, n_iterations=10, population_size=12, init_params=None):
        """Disabled compatibility entry: old samples never changed rollout commands."""
        del side, n_iterations, population_size, init_params
        logger.warning(
            "[Stage3 CEM] disabled: optimize time scale only after deterministic Stage3A passes"
        )
        return None


def parent_source_end(parent_bundle: dict) -> float:
    """Small helper kept separate to make milestone selection testable."""
    return float(np.asarray(parent_bundle["source_frame"], dtype=np.float64)[-1])
