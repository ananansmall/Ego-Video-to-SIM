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
import os

import numpy as np

STAGE1_CLOSE_CONTROLS = 20
STAGE1_SETTLE_CONTROLS = 10
STAGE1_STOP_CONTROLS = STAGE1_CLOSE_CONTROLS + STAGE1_SETTLE_CONTROLS

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


def stage_q_open() -> float:
    """Shared Stage1/Stage2 full-open finger command ("F50 入口开度").

    Default is the historical ``GRIPPER_MAX_OPEN`` (0.05).  ``_Q_OPEN`` overrides
    it for experiments, but the value must stay shared: ``stage1_reuse_report``
    compares the Stage1 recipe's ``q_open`` against the Stage2 entry q and fails
    fast with ``STAGE1_REUSE_MISMATCH`` otherwise.

    Note: the URDF finger joint limit is ``upper=0.075``; commands above that are
    physically clamped by the joint limit.
    """
    env = os.environ.get("_Q_OPEN")
    if env:
        return float(env)
    from physics_utils import GRIPPER_MAX_OPEN

    return float(GRIPPER_MAX_OPEN)


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
