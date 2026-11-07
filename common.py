"""Collection jobs, device selection and local planner protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import socket

HERE = Path(__file__).resolve().parent
OBJECTS = ("block55", "slender95", "cylinder36h52", "disk56h24")
REFERENCE = {
    "block55": (-.010, .006, -23), "slender95": (-.012, .009, -29),
    "cylinder36h52": (-.013, .009, -47), "disk56h24": (.010, .012, 67),
}


def local_setting(name, default=None):
    if os.environ.get(name):
        return os.environ[name]
    path = HERE / "local_env.json"
    return read(path).get(name, default) if path.is_file() else default


def collection_plan(objects, episodes, workers=1, seed=2026207000):
    if not 1 <= episodes <= 10000:
        raise ValueError("--episodes-per-object must be between 1 and 10000")
    if not 1 <= workers <= 3:
        raise ValueError("--workers must be between 1 and 3")
    if not objects or len(set(objects)) != len(objects) or any(n not in OBJECTS for n in objects):
        raise ValueError("Select distinct bundled object IDs, or all")
    if not 0 <= seed <= 2**32 - 400000:
        raise ValueError("--seed must be between 0 and 2**32 - 400000")
    workload = []
    for name in objects:
        x, y, yaw = REFERENCE[name]
        for index in range(episodes):
            scene_seed = seed + OBJECTS.index(name) * 100000 + index
            rng = random.Random(scene_seed)
            dx, dy, angle = (0., 0., 0.) if index == 0 else (
                rng.uniform(-.002, .002), rng.uniform(-.002, .002), rng.uniform(-8, 8))
            workload.append(dict(
                case_id=f"{name}_{index:06d}", object_id=name, scene_seed=scene_seed,
                xy_offset_m=[max(-.015, min(.015, x+dx)), max(-.015, min(.015, y+dy))],
                yaw_deg=yaw+angle, role="collection", methods=["graspgen"],
                task="pick_up_any_stable_region", included_in_formal_score=False,
                max_physical_attempts_per_method=1, fixed_index=index,
                diagnostic_known_geometry=True))
    return dict(
        schema=1, objects=list(objects), total_attempt_slots=len(workload),
        unique_scene_count=len(workload), seed=seed,
        stages=[dict(name="collection", lanes=workers, persistent=True, cases=workload)],
        physics_hz=480, control_hz=60, rgb_hz=60, camera_count=2,
        camera_resolution=[256, 256], max_control_steps=2400, hold_controls=120,
        pipeline="mesh -> GraspGen -> cuRobo -> Isaac Lab / PhysX -> NPZ + dual RGB",
        failures="One attempt per scheduled scene; retain failures; no automatic replacement",
        scope="Bundled known meshes, fixed local XY/yaw sampling; scene IDs count scheduled starts")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases(per_object=6):
    if not 1 <= per_object <= 96:
        raise ValueError("per_object must be between 1 and 96 to preserve distinct scene seeds")
    result = []
    for index, name in enumerate(OBJECTS):
        x, y, yaw = REFERENCE[name]
        for number in range(per_object):
            seed = 2026207000 + index * 100 + number
            rng = random.Random(seed)
            dx, dy, angle = (0., 0., 0.) if number == 0 else (
                rng.uniform(-.002, .002), rng.uniform(-.002, .002), rng.uniform(-8, 8))
            result.append(dict(
                case_id=f"{name}_{number:02d}", object_id=name, scene_seed=seed,
                xy_offset_m=[max(-.015, min(.015, x + dx)), max(-.015, min(.015, y + dy))],
                yaw_deg=yaw + angle, role="development", methods=["graspgen"],
                task="pick_up_any_stable_region", included_in_formal_score=False,
                max_physical_attempts_per_method=1, fixed_index=number,
                diagnostic_known_geometry=True,
            ))
    return result


def make_plan(full=False, only=None, smoke=False, episodes_per_worker=None):
    if episodes_per_worker is not None:
        if smoke or full or only != "parallel3":
            raise ValueError("episodes_per_worker requires only='parallel3', without smoke/full")
        if not isinstance(episodes_per_worker, int) or not 1 <= episodes_per_worker <= 32:
            raise ValueError("episodes_per_worker must be an integer between 1 and 32")
    workload = cases(3 * episodes_per_worker if episodes_per_worker is not None else 6)
    if smoke:
        stages = [dict(name="smoke", lanes=1, persistent=True, cases=workload[:2])]
    else:
        paired = workload if full else [c for c in workload if c["fixed_index"] < 2]
        stages = [
            dict(name="cold1", lanes=1, persistent=False, cases=paired),
            dict(name="warm1", lanes=1, persistent=True, cases=paired),
            dict(name="parallel3", lanes=3, persistent=True, cases=workload),
        ]
        if only:
            stages = [s for s in stages if s["name"] == only]
    return dict(
        schema=1, stages=stages, objects=list(OBJECTS),
        parallel_episodes_per_worker=episodes_per_worker if episodes_per_worker is not None else 2,
        total_attempt_slots=sum(len(s["cases"]) for s in stages),
        unique_scene_count=len({c["case_id"] for s in stages for c in s["cases"]}),
        physics_hz=480, control_hz=60, rgb_hz=60, camera_resolution=[256, 256],
        camera_count=2, max_control_steps=2400, hold_controls=120,
        candidate_source="Locally generated GraspGen mesh pools; model inference is a separate preparation stage",
        failures="Keep every fixed slot; no replacement, automatic retry or success-only timing",
        environment_kind="One real environment per persistent simulation process; at most three simulation processes",
        pipeline="simulation state + complete mesh geometry -> GraspGen cached candidates -> cuRobo -> Isaac rollout -> NPZ and dual RGB NPY",
        benchmark_scope="Collection and raw data save; validation is timed separately; LeRobot/video export excluded",
    )


def gpu_tokens(selection, inherited=None):
    indices = [int(v) for v in selection.split(",")]
    if len(indices) not in (1, 2, 4) or len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError("--gpus accepts one GPU, two GPUs (simulation,planner), or four GPUs (three simulations,planner)")
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES") if inherited is None else inherited
    if inherited is not None:
        visible = [v.strip() for v in inherited.split(",") if v.strip()]
        if not visible or max(indices) >= len(visible):
            raise ValueError("--gpus indexes outside the inherited CUDA_VISIBLE_DEVICES allocation")
        result = [visible[i] for i in indices]
    else:
        result = [str(i) for i in indices]
    if len(result) == 1:
        return result * 4
    if len(result) == 2:
        return [result[0]] * 3 + [result[1]]
    return result


class PlannerClient:
    def __init__(self, address, timeout=600):
        self.address = ("127.0.0.1", int(address["port"]))
        self.timeout = timeout
        self.token = address["token"]

    def request(self, request, fresh=False):
        # One request per connection: concurrent clients cannot mix replies.
        payload = dict(request=request, fresh=bool(fresh), token=self.token)
        with socket.create_connection(self.address, timeout=self.timeout) as connection:
            connection.settimeout(self.timeout)
            with connection.makefile("rwb") as stream:
                stream.write(json.dumps(payload, allow_nan=False).encode() + b"\n")
                stream.flush()
                line = stream.readline(16 * 1024 * 1024)
                if not line:
                    raise RuntimeError("Planner service closed without a reply")
                result = json.loads(line)
                if result.get("transport_error"):
                    raise RuntimeError(result["transport_error"])
                return result
