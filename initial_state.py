"""Verify requested object starts independently of case IDs and physics scores."""
from __future__ import annotations

import math
from pathlib import Path
import time

from common import write


def requested_pose(entry, asset, reference_xy, settled=True):
    angle = math.radians(entry["yaw_deg"]) / 2
    return [reference_xy[0] + entry["xy_offset_m"][0],
            reference_xy[1] + entry["xy_offset_m"][1],
            -float(asset["bounds_min"][2]) + (0. if settled else .002),
            math.cos(angle), 0., 0., math.sin(angle)]


def pose_comparison(expected, observed, position_tolerance_m=1e-4, rotation_tolerance_rad=1e-3):
    if observed is None or len(observed) != 7 or not all(math.isfinite(v) for v in observed):
        return dict(status="missing_or_nonfinite")
    position = math.dist(expected[:3], observed[:3])
    qa, qb = expected[3:], observed[3:]
    norms = math.sqrt(sum(v*v for v in qa) * sum(v*v for v in qb))
    if norms == 0:
        return dict(status="invalid_quaternion")
    rotation = 2 * math.acos(min(1., abs(sum(a*b for a, b in zip(qa, qb))) / norms))
    return dict(status="matched" if position <= position_tolerance_m and rotation <= rotation_tolerance_rad else "mismatch",
                position_error_m=position, rotation_error_rad=rotation,
                position_tolerance_m=position_tolerance_m, rotation_tolerance_rad=rotation_tolerance_rad)


def install_reset_guard(episode, env, entry, asset, reference_xy, output, raw_pose, prime_first=False):
    """Prime a new worker, trace reset, and reject wrong starts before planning.

    The existing two-control park protocol is also used before a worker's first
    reset. No failed reset/rollout is retried. The original 120-control settling,
    recorded rollout, contact readings and success policy remain in the runtime.
    """
    original_reset = episode.reset
    output = Path(output)

    def checked_reset():
        if prime_first:
            started = time.monotonic()
            # The original collector creates targets in _reset_idx. A new
            # worker has not reset yet, so hold its current joints for priming.
            if not hasattr(env, "targets"):
                env.targets = env.robot.data.joint_pos[:, env.joint_ids].clone()
            env.park_object()
            episode.timing["first_reset_prime"] = time.monotonic() - started
        native_reset = env.reset
        trace = []

        def traced_reset(*args, **kwargs):
            result = native_reset(*args, **kwargs)
            trace.append(raw_pose())
            write(output / "RESET_READBACK.json", dict(
                expected_before_settle=requested_pose(entry, asset, reference_xy, settled=False),
                raw_readbacks=trace,
                note="Pre-step readback is diagnostic; deferred simulator writes are checked after settling."))
            return result

        env.reset = traced_reset
        try:
            original_reset()
        finally:
            env.reset = native_reset
        expected = requested_pose(entry, asset, reference_xy)
        cached = episode.row["object_pose"].tolist()
        raw = raw_pose()
        checks = dict(requested_vs_cached=pose_comparison(expected, cached),
                      requested_vs_raw=pose_comparison(expected, raw),
                      cached_vs_raw=pose_comparison(cached, raw))
        passed = all(c["status"] == "matched" for c in checks.values())
        write(output / "INITIAL_STATE_AUDIT.json", dict(
            status="passed" if passed else "failed", case_id=entry["case_id"],
            first_reset_primed=prime_first, expected_settled_pose=expected,
            cached_settled_pose=cached, raw_settled_pose=raw, checks=checks))
        if not passed:
            episode.reason, episode.success = "requested_initial_state_mismatch", False

    episode.reset = checked_reset
