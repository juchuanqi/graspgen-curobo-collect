"""Current-pose-aware representations of a parallel-jaw grasp.

The local TCP Z axis is the approach axis. Right-multiplying its rotation by
diag(-1, -1, 1) exchanges the fingers without changing the TCP or approach.
Each representation still needs its own geometry, IK and full-path checks.
"""
from __future__ import annotations

import copy
import numpy as np

MODES = ("off", "near-first")


def mode(value="off"):
    if value not in MODES:
        raise ValueError(f"Unknown grasp symmetry mode: {value!r}")
    return value


def _transform(value):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0)):
        raise ValueError("Grasp symmetry requires a finite rigid 4x4 transform")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-4, rtol=0)):
        raise ValueError("Grasp symmetry requires a proper rotation")
    return transform


def rotation_distance_deg(current, target):
    """SO(3) distance in [0, 180], independent of quaternion sign/Euler wrap."""
    rotations = []
    for transform in (current, target):
        # Model matrices have float32 roundoff; project onto SO(3) before
        # measuring an angle, especially around zero and pi.
        u, _, vh = np.linalg.svd(_transform(transform)[:3, :3])
        rotations.append(u @ vh)
    relative = rotations[0].T @ rotations[1]
    skew = np.array([relative[2, 1] - relative[1, 2],
                     relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    return float(np.degrees(np.arctan2(np.linalg.norm(skew) / 2,
                                      (np.trace(relative) - 1) / 2)))


def _flip(candidate):
    local = _transform(candidate["T_object_tcp"]).copy()
    local[:3, :2] *= -1
    candidate["T_object_tcp"] = local.tolist()
    # A transformed pose has no newly evaluated network confidence. Preserve
    # parent-only evidence under explicit names instead of attributing it to
    # a pose that was never passed through the model or object-only checker.
    candidate["score"] = None
    if "candidate_source_kind" in candidate:
        candidate["parent_candidate_source_kind"] = candidate["candidate_source_kind"]
    candidate["candidate_source_kind"] = "parallel_jaw_z180_symmetry"
    for key in ("T_object_grasp", "execution_quality", "width_attempts"):
        if key in candidate:
            candidate["parent_" + key] = candidate.pop(key)
    span = candidate.get("pad_span")
    if span is not None:
        positive, negative = span["positive_y_m"], span["negative_y_m"]
        span["positive_y_m"], span["negative_y_m"] = -negative, -positive
        if "center_y_m" in span:
            span["center_y_m"] *= -1
        if "positive_samples" in span and "negative_samples" in span:
            span["positive_samples"], span["negative_samples"] = (
                span["negative_samples"], span["positive_samples"])


def split_candidates(candidates, object_world, current_tcp_world):
    """Return N near and N far poses, preserving source order within groups.

    Angles use world TCP frames after the existing model/TCP adapter. An exact
    tie keeps the original in the near group. Inputs are never modified.
    """
    obj = _transform(object_world)
    current = _transform(current_tcp_world)
    near, far = [], []
    for index, parent in enumerate(candidates):
        original, flipped = copy.deepcopy(parent), copy.deepcopy(parent)
        _flip(flipped)
        pair = [original, flipped]
        angles = [rotation_distance_deg(current, obj @ _transform(c["T_object_tcp"]))
                  for c in pair]
        near_index = 0 if angles[0] <= angles[1] + 1e-9 else 1
        for variant_index, candidate in enumerate(pair):
            variant = "original" if variant_index == 0 else "z180"
            group = "near" if variant_index == near_index else "far"
            candidate.update(
                candidate_id=f"sym{index:04d}_{variant}",
                parent_original_rank=parent["original_rank"],
                parent_model_score=parent.get("score"),
                parent_T_object_tcp=copy.deepcopy(parent["T_object_tcp"]),
                symmetry_pair_index=index, symmetry_variant=variant,
                symmetry_group=group, rotation_from_current_deg=angles[variant_index],
                paired_rotation_from_current_deg=angles[1 - variant_index])
            (near if group == "near" else far).append(candidate)
    return near, far


def planning_key(candidate, joint_bound, joint_total, index):
    """Near representations first; retain joint-motion ranking within a group."""
    priority = 1 if candidate.get("symmetry_group") == "far" else 0
    return priority, joint_bound, joint_total, index


def identity(candidate):
    """Keep both representations distinct in saved planning/check artifacts."""
    return {key: candidate[key] for key in (
        "candidate_id", "parent_original_rank", "symmetry_pair_index",
        "symmetry_variant", "symmetry_group", "rotation_from_current_deg")
        if key in candidate}
