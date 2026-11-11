"""CPU helpers for validated platform episodes and the official LeRobot API."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CAMERAS = ("base_camera", "wrist_camera")
TOOL_FILES = ("bridge.py", "select_sources.py", "export_lerobot.py", "audit_lerobot.py", "run.py")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical8(values):
    return np.concatenate((values[..., :7], values[..., 7:9].sum(-1, keepdims=True)), -1).astype(np.float32)


def compare_pose(expected, observed):
    a, b = np.asarray(expected, dtype=float), np.asarray(observed, dtype=float)
    if a.shape != (7,) or b.shape != (7,) or not np.isfinite([a, b]).all():
        return {"status": "invalid_pose"}
    norm = np.linalg.norm(a[3:]) * np.linalg.norm(b[3:])
    if norm == 0:
        return {"status": "invalid_quaternion"}
    position = float(np.linalg.norm(a[:3] - b[:3]))
    rotation = 2 * math.acos(min(1., abs(float(a[3:] @ b[3:])) / norm))
    return dict(status="passed" if position <= 1e-4 and rotation <= 1e-3 else "failed",
                position_error_m=position, rotation_error_rad=rotation)


def verify_inputs(episode):
    source = Path(episode["source"])
    for name, expected in episode["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Source changed: {source / name}")


def feature_spec():
    arm = [f"panda_joint{i}" for i in range(1, 8)]
    raw9 = arm + ["left_finger_m", "right_finger_m"]

    def vector(names, dtype="float32"):
        return dict(dtype=dtype, shape=(len(names),), names=names)

    features = {
        "observation.state": vector(arm + ["total_finger_opening_m"]),
        "action": vector(arm + ["total_finger_target_m"]),
        "aux.qpos_raw": vector(raw9),
        "aux.qvel_raw": vector([f"{x}_rad_s" for x in arm] + ["left_finger_m_s", "right_finger_m_s"]),
        "aux.position_target_raw": vector(raw9),
        "aux.phase": vector(["planner_phase"], "int64"),
        "aux.sim_time_s": vector(["source_simulation_time_s"], "float64"),
        "next.done": vector(["derived_episode_boundary"], "bool"),
        "episode_success": vector(["validated_stable_grasp_hold"], "bool"),
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = dict(
            dtype="image", shape=(256, 256, 3), names=["height", "width", "channels"])
    return features
