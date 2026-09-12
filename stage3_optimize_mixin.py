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
    release_controls: int = 5,
    pre_release_settle_controls: int = 3,
    release_settle_controls: int = 10,
    control_freq: float = 30.0,
) -> dict:
    """Build a Stage3 suffix and an immutable-prefix combined command bundle.

    The first suffix command has ``source_frame > Stage2.f_end``.  During carry,
    the Stage2-end correction relative to MANO is fixed and the fingers remain at
    ``q_grasp``.  Only the explicit RELEASE phase changes the finger command.
    """
    from physics_utils import GRIPPER_MAX_OPEN

    mano_pos_aligned = np.asarray(mano_pos_aligned, dtype=np.float64)
    mano_R = np.asarray(mano_R, dtype=np.float64)
    if mano_pos_aligned.ndim != 2 or mano_pos_aligned.shape[1] != 3:
        raise ValueError("mano_pos_aligned must have shape (N, 3)")
    if mano_R.shape != (len(mano_pos_aligned), 3, 3):
        raise ValueError("mano_R must have shape (N, 3, 3)")
    if carry_time_scale < 1.0:
        raise ValueError("carry_time_scale must be >= 1.0")
    if release_controls < 1 or pre_release_settle_controls < 1:
        raise ValueError("release and pre-release settle phases must be non-empty")
    if release_settle_controls < 1:
        raise ValueError("release settle phase must be non-empty")

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
    q_grasp = float(stage2_bundle.get("timeline_meta", {}).get("q_grasp", parent_q[-1, 0]))
    if not np.allclose(parent_q[-1], q_grasp):
        raise ValueError("Stage2 endpoint does not hold q_grasp")

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
        grip.append((float(q), float(q)))
        policy.append(str(contact_policy))

    carry_intervals = max(1, int(np.ceil((release_candidate - handoff) * carry_time_scale)))
    for k in range(1, carry_intervals + 1):
        progress = k / float(carry_intervals)
        src = handoff + (release_candidate - handoff) * _smoothstep5(progress)
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
    for k in range(1, release_controls + 1):
        q = q_grasp + (float(GRIPPER_MAX_OPEN) - q_grasp) * _smoothstep5(
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
    last_frame = len(mano_pos_aligned) - 1
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
            "q_grasp": q_grasp,
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
        from data_loader import rotmat_to_zyx_euler
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
        replay = self.rollout_follow_trajectory(
            recon,
            side,
            F50_IDX=int(parent_meta.get("f_grasp", 50)),
            approach_time_scale=float(parent_meta.get("approach_time_scale", 2.0)),
            carry_time_scale=float(parent_meta.get("carry_time_scale", 2.0)),
            record_video=bool(record_video),
        )
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

        def is_target(name):
            return name == target_name or bool(target_name and target_name in name)

        def execute(index):
            target_pos = suffix["base_pos"][index]
            target_R = suffix["base_R"][index]
            q_cmd = np.asarray(suffix["gripper_q"][index], dtype=np.float64)
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
            rz, ry, rx = rotmat_to_zyx_euler(target_R)
            for key, value in (
                ("vx", target_pos[0]),
                ("vy", target_pos[1]),
                ("vz", target_pos[2]),
                ("rz", rz),
                ("ry", ry),
                ("rx", rx),
            ):
                active_joints[virtual_idx[key]].set_drive_target(float(value))
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
                if float(np.min(final_q)) < q_open - STAGE3_OPEN_TRACK_TOL_M:
                    stage3b_gate = _empty_gate("OPEN_TRACK", f"final q={final_q.tolist()}")
                elif final_finger_contact:
                    stage3b_gate = _empty_gate("NOT_RELEASED", "target still touches a finger")
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
        """Save a combined executable artifact only when Stage3A passed."""
        gate_a = result.get("stage3a_gate", {})
        if not gate_a.get("passed", False):
            logger.warning("[Stage3 save] refusing failed Stage3A: %s", gate_a.get("failure"))
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
