"""Select fixed, validated, correctly reset trajectories without changing raw runs."""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import numpy as np

from bridge import CAMERAS, TOOL_FILES, compare_pose, read, sha, write


def main(args):
    if args.per_object is not None and args.per_object < 1:
        raise ValueError("--per-object must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    run = args.run.resolve()
    runtime = args.runtime.resolve()
    receipt = read(run / "RUN.json")
    if receipt.get("status") != "completed" or receipt.get("validation_requested") is not True:
        raise ValueError("Export requires a completed run with data validation enabled")
    plan = read(run / "PLAN.json")
    if (plan.get("control_hz"), plan.get("rgb_hz"), plan.get("camera_count"), plan.get("camera_resolution")) != (
        60, 60, 2, [256, 256]
    ):
        raise ValueError("Export requires the 60 Hz dual 256x256 RGB collection format")
    stages = {s["name"]: s for s in plan["stages"]}
    if args.stage is None and len(stages) == 1:
        stage = next(iter(stages.values()))
    elif args.stage in stages:
        stage = stages[args.stage]
    else:
        raise ValueError(f"Choose --stage from: {', '.join(stages)}")
    stage_name = stage["name"]
    scene_config = runtime / "config/scene.json"
    xy = np.asarray(read(scene_config)["object_reference_xy_m"], dtype=float)
    assets = read(runtime / "assets/MANIFEST.json")["objects"]
    selected, inspected, selected_counts, starts = [], [], Counter(), []
    for entry in sorted(stage["cases"], key=lambda c: (c["object_id"], c["fixed_index"])):
        matches = list((run / stage_name / "workers").glob(f"*/{entry['case_id']}/METADATA.json"))
        if len(matches) > 1:
            raise ValueError(f"Ambiguous source {entry['case_id']}")
        record = dict(case_id=entry["case_id"], object_id=entry["object_id"], eligible=False)
        inspected.append(record)
        if not matches:
            record["exclusion"] = "missing_fixed_slot"
            continue
        source = matches[0].parent
        metadata = read(matches[0])
        if read(source / "ENTRY.json") != entry:
            raise ValueError(f"ENTRY differs from run plan: {source}")
        validations = [source / n for n in ("VALIDATION.json", "RAW_FORCE_VALIDATION.json")]
        if not (metadata["success"] is True and metadata["accepted"] is True and
                metadata["reason"] == "stable_grasp_hold" and
                all(p.exists() and read(p).get("status") == "passed" for p in validations)):
            record["exclusion"] = "physical_success_or_validation_missing"
            continue
        angle = math.radians(entry["yaw_deg"]) / 2
        expected = [*(xy + entry["xy_offset_m"]), -assets[entry["object_id"]]["bounds_min"][2],
                    math.cos(angle), 0., 0., math.sin(angle)]
        with np.load(source / "trajectory.npz", allow_pickle=False) as raw:
            native = raw["contact_substep_object_pose_w"][0, -1].astype(float)
            cached = raw["object_pose"][0].astype(float)
            initial_qpos = raw["qpos"][0].astype(float)
        checks = dict(requested_vs_native=compare_pose(expected, native),
                      requested_vs_cached=compare_pose(expected, cached),
                      cached_vs_native=compare_pose(cached, native))
        record["initial_state_checks"] = checks
        if any(c["status"] != "passed" for c in checks.values()):
            record["exclusion"] = "requested_initial_state_mismatch"
            continue
        online = source / "INITIAL_STATE_AUDIT.json"
        if not online.is_file() or read(online).get("status") != "passed":
            record["exclusion"] = "online_initial_audit_missing_or_failed"
            continue
        if not np.isfinite(initial_qpos).all():
            record["exclusion"] = "invalid_initial_joints"
            continue
        record["eligible"] = True
        duplicate = any(
            obj == entry["object_id"] and compare_pose(native, pose)["position_error_m"] <= 1e-5
            and compare_pose(native, pose)["rotation_error_rad"] <= 1e-4
            and np.max(np.abs(initial_qpos - joints)) <= 1e-5
            for obj, pose, joints in starts)
        if duplicate:
            record["exclusion"] = "duplicate_measured_start"
            continue
        starts.append((entry["object_id"], native, initial_qpos))
        if args.per_object is not None and selected_counts[entry["object_id"]] >= args.per_object:
            record["exclusion"] = "pilot_per_object_limit"
            continue
        files = ["trajectory.npz", "METADATA.json", "ENTRY.json", "SCENE.json", "ENVIRONMENT.json",
                 "NATIVE_PHYSICS.json", "SOURCE_AT_LAUNCH.json", "VALIDATION.json", "RAW_FORCE_VALIDATION.json"]
        files += [f"cameras/{c}.npy" for c in CAMERAS]
        files += ["INITIAL_STATE_AUDIT.json"]
        if (source / "RESET_READBACK.json").is_file():
            files.append("RESET_READBACK.json")
        selected.append(dict(export_episode_index=len(selected), source=str(source),
                             stage=stage_name, case_id=entry["case_id"], object_id=entry["object_id"],
                             scene_seed=entry["scene_seed"], length=metadata["actions"],
                             task="Pick up the object and hold it above the table.",
                             expected_initial_pose=expected, initial_state_checks=checks,
                             source_sha256={n: sha(source / n) for n in files}))
        selected_counts[entry["object_id"]] += 1
        record["selected"] = True
    inputs = [run / "PLAN.json", run / "RUN.json", runtime / "assets/MANIFEST.json",
              scene_config]
    result = dict(status="selected_before_export" if selected else "no_eligible_episodes",
                  source_run=str(run), runtime=str(runtime),
                  stage=stage_name, fixed_slots=len(stage["cases"]),
                  eligible_slots=sum(r["eligible"] for r in inspected),
                  selected_episodes=len(selected), per_object=dict(selected_counts),
                  role="engineering_format_pilot" if args.per_object else "validated_expert_export",
                  selection_policy="sorted object ID and fixed index; physical success, both validators, online audit, raw/cached initial pose; deduplicate measured starts; optional fixed per-object limit",
                  original_inputs_sha256={str(p): sha(p) for p in inputs},
                  episodes=selected, inspected_slots=inspected,
                  tool_sha256={name: sha(Path(__file__).parent / name) for name in TOOL_FILES},
                  new_physics=False, training=False)
    write(args.output, result)
    print({k: result[k] for k in ("fixed_slots", "eligible_slots", "selected_episodes", "per_object")}, flush=True)
    if not selected:
        raise ValueError("No eligible episodes; see inspected_slots in the selection report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--per-object", type=int)
    main(parser.parse_args())
