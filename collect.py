"""Collect Panda grasps in Isaac Lab using GraspGen and cuRobo."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from common import HERE, OBJECTS, collection_plan, gpu_tokens, local_setting, make_plan, read, sha, write


def interpreter(explicit, variable, fallback=None):
    value = explicit or local_setting(variable) or fallback or sys.executable
    result = shutil.which(value) or value
    if not Path(result).is_file():
        raise FileNotFoundError(f"Set {variable} or its command-line option: {value}")
    return str(Path(result).absolute())


def check_environment(sim_python, planner_python, runtime):
    from tools.check_assets import check_assets
    check_assets(runtime)
    checks = {}
    for name, python, modules in (
        ("simulation", sim_python, ["isaaclab", "isaacsim", "torch", "numpy", "scipy", "trimesh"]),
        ("planning", planner_python, ["curobo", "torch", "numpy", "scipy", "trimesh"]),
    ):
        code = '''
import importlib, importlib.util, json, sys
mods = json.loads(sys.argv[1])
missing = [m for m in mods if importlib.util.find_spec(m) is None]
result = dict(python=sys.version, missing=missing, versions={}, cpu_import_error=None)
if not missing:
    try:
        for name in ('torch', 'numpy', 'scipy', 'trimesh'):
            module = importlib.import_module(name)
            result['versions'][name] = str(module.__version__)
        import torch
        assert torch.ones(2, device='cpu').sum().item() == 2
    except Exception as error:
        result['cpu_import_error'] = repr(error)
result['python_3_11'] = sys.version_info[:2] == (3, 11)
print(json.dumps(result))
sys.exit(bool(missing) or bool(result['cpu_import_error']) or not result['python_3_11'])
'''
        result = subprocess.run([python, "-B", "-c", code, json.dumps(modules)],
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                                capture_output=True, text=True, timeout=90)
        checks[name] = dict(python=python, returncode=result.returncode,
                            stdout=result.stdout, stderr=result.stderr)
    failed = [name for name, check in checks.items() if check["returncode"]]
    if failed:
        raise RuntimeError(
            "CPU dependency check failed for " + ", ".join(failed) + ". "
            "Set SIM_PYTHON and PLANNER_PYTHON in the environment or local_env.json "
            "to working Python 3.11 environments. Details: " + json.dumps(checks))
    return checks


class Processes:
    def __init__(self):
        self.children = []

    def start(self, command, environment, log):
        stream = Path(log).open("x")
        try:
            child = subprocess.Popen(command, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=str(Path(log).resolve().parent))
        finally:
            stream.close()
        self.children.append(child)
        return child

    @staticmethod
    def stop(child):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=10)

    def close(self):
        for child in reversed(self.children):
            self.stop(child)


def environment(gpu, output):
    cache_root = Path(os.environ.get("COLLECT_CACHE_DIR", str(HERE / ".cache"))).expanduser().resolve()
    result = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "OMNI_KIT_ACCEPT_EULA": "YES",
            "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUROBO_TORCH_COMPILE_DISABLE": "1", "MAX_JOBS": "4"}
    for variable, folder in (("TORCH_EXTENSIONS_DIR", "torch_extensions"),
                             ("XDG_CACHE_HOME", "xdg"), ("CUDA_CACHE_PATH", "cuda"),
                             ("TRITON_CACHE_DIR", "triton"), ("TMPDIR", "tmp")):
        location = Path(result.get(variable) or cache_root / folder).expanduser().resolve()
        location.mkdir(parents=True, exist_ok=True)
        result[variable] = str(location)
    return result


def gpu_monitor(output, stop):
    with (output / "gpu_samples.jsonl").open("x") as stream:
        while not stop.is_set():
            started = time.time()
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,power.draw",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
                value = dict(unix_s=started, returncode=result.returncode,
                             csv=result.stdout, stderr=result.stderr)
            except Exception as error:
                value = dict(unix_s=started, error=repr(error))
            stream.write(json.dumps(value) + "\n")
            stream.flush()
            stop.wait(3)


def job_waves(stage):
    waves = []
    for object_id in OBJECTS:
        group = [c for c in stage["cases"] if c["object_id"] == object_id]
        if not stage["persistent"]:
            waves.extend([(0, [case])] for case in group)
        else:
            wave = [(lane, group[lane::stage["lanes"]]) for lane in range(stage["lanes"])]
            if group:
                waves.append([(lane, cases) for lane, cases in wave if cases])
    return waves


def run_stage(stage, args, tokens, sim_python, planner_python, processes):
    out = args.output / stage["name"]
    out.mkdir()
    started = time.monotonic()
    receipt = dict(status="running", name=stage["name"], planned_slots=len(stage["cases"]),
                   lanes=stage["lanes"], started_unix_s=time.time(), jobs=[],
                   gpu_tokens=tokens, stage_includes="planner service startup, all worker startups, rollout/save and teardown")
    write(out / "STAGE.json", receipt)
    service = processes.start([
        planner_python, "-B", "-u", str(HERE / "planner_service.py"),
        "--runtime", str(args.runtime), "--ready", str(out / "planner_address.json")],
        environment(tokens[3], args.output), out / "planner.log")
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not (out / "planner_address.json").exists():
            if service.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Planner service failed to start; see " + str(out / "planner.log"))
            time.sleep(.2)
        receipt["planner_service_startup_s"] = time.monotonic() - started
        waves = job_waves(stage)
        next_update = 0.
        for wave_number, wave in enumerate(waves):
            active = []
            for lane, workload in wave:
                worker_id = f"{workload[0]['object_id']}_wave{wave_number:02d}_lane{lane}"
                job = dict(stage=stage["name"], worker_id=worker_id,
                           persistent=stage["persistent"], cases=workload,
                           candidate_root=str(args.candidate_root))
                path = out / "jobs" / (worker_id + ".json")
                write(path, job)
                worker_output = out / "workers" / worker_id
                command = [sim_python, "-B", "-u", str(HERE / "sim_worker.py"),
                           "--runtime", str(args.runtime), "--job", str(path),
                           "--address", str(out / "planner_address.json"), "--output", str(worker_output),
                           "--enable_cameras", "--device", "cuda:0"]
                if not args.gui:
                    command.append("--headless")
                child = processes.start(command, environment(tokens[lane], args.output), out / (worker_id + ".log"))
                row = dict(worker_id=worker_id, simulation_gpu=tokens[lane],
                           pid=child.pid, cases=[c["case_id"] for c in workload],
                           output=str(worker_output), command=command, started_unix_s=time.time())
                receipt["jobs"].append(row)
                active.append((child, row, time.monotonic() + args.startup_timeout + len(workload) * args.episode_timeout))
            write(out / "STAGE.json", receipt)
            while any(child.poll() is None for child, _, _ in active):
                if service.poll() is not None:
                    raise RuntimeError("Planner service exited during collection")
                for child, row, limit in active:
                    if child.poll() is None and time.monotonic() > limit:
                        row["timeout"] = True
                        raise TimeoutError("Worker timed out without retry: " + row["worker_id"])
                if time.monotonic() >= next_update:
                    done = sum(1 for _ in out.glob("workers/*/*/METADATA.json"))
                    print(f"[{stage['name']}] {done}/{len(stage['cases'])} slots saved, "
                          f"elapsed {(time.monotonic()-started)/60:.1f} min", flush=True)
                    next_update = time.monotonic() + 30
                time.sleep(.2)
            for child, row, _ in active:
                row.update(returncode=child.returncode, ended_unix_s=time.time())
                if child.returncode:
                    raise RuntimeError("Worker failed without retry: " + row["worker_id"])
        receipt["status"] = "completed"
    except BaseException:
        receipt["status"] = "interrupted_or_technical_failure"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        # End collection before recording final time; service lifetime is included.
        for child, _, _ in locals().get("active", []):
            processes.stop(child)
        processes.stop(service)
        receipt.update(wall_s=time.monotonic()-started, ended_unix_s=time.time())
        write(out / "STAGE.json", receipt)


def validate(output, runtime, python):
    started = time.monotonic()
    for path in sorted(output.glob("*/workers/*/*/METADATA.json")):
        directory = path.parent
        for name, result_name in (("validate_episode.py", "VALIDATION.json"),
                                   ("validate_raw_forces.py", "RAW_FORCE_VALIDATION.json")):
            with (directory / (name + ".log")).open("x") as log:
                result = subprocess.run([python, "-B", str(runtime / "validators" / name),
                                         str(directory), "--output", str(directory / result_name)],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode:
                raise RuntimeError("Saved data validation failed: " + str(directory / (name + ".log")))
    write(output / "VALIDATION_SUMMARY.json", dict(status="passed", wall_s=time.monotonic()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime", type=Path, default=HERE / "runtime")
    parser.add_argument("--gpus", default="0", help="Logical GPU indices: 0; 0,1; or 0,1,2,3")
    parser.add_argument("--objects", default="all", help="all or comma-separated object IDs")
    parser.add_argument("--episodes-per-object", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="Persistent simulation processes (1 to 3)")
    parser.add_argument("--seed", type=int, default=2026207000)
    parser.add_argument("--gui", action="store_true", help="Show Isaac Sim window (one worker)")
    parser.add_argument("--candidate-root", type=Path, default=HERE / "generated/candidates")
    parser.add_argument("--benchmark", action="store_true", help="Run original cold/warm/parallel timing comparison")
    parser.add_argument("--startup-timeout", type=float, default=600.)
    parser.add_argument("--sim-python")
    parser.add_argument("--planner-python")
    parser.add_argument("--dry-run", "--plan", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true", help="24 identical scenes in all three modes, 72 slots")
    parser.add_argument("--only", choices=("cold1", "warm1", "parallel3"))
    parser.add_argument("--episodes-per-worker", type=int,
                        help="With --only parallel3: 1–32 consecutive episodes per worker per object (10 means 120 fixed slots)")
    parser.add_argument("--episode-timeout", type=float, default=1200.)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    if args.episode_timeout <= 0 or args.startup_timeout <= 0:
        parser.error("Timeouts must be positive")
    if args.smoke and (args.full or args.only):
        parser.error("--smoke cannot be combined with --full or --only")
    if args.smoke and args.skip_validation:
        parser.error("--smoke requires data validation")
    try:
        if args.smoke:
            plan = collection_plan(["block55"], 2, 1, args.seed)
        elif args.benchmark:
            if args.objects != "all" or args.episodes_per_object != 2:
                parser.error("--objects and --episodes-per-object apply to collection, not benchmark")
            plan = make_plan(args.full, args.only, False, args.episodes_per_worker)
        else:
            if args.full or args.only or args.episodes_per_worker is not None:
                parser.error("Timing comparison options require --benchmark")
            objects = list(OBJECTS) if args.objects == "all" else args.objects.split(",")
            plan = collection_plan(objects, args.episodes_per_object, args.workers, args.seed)
        tokens = gpu_tokens(args.gpus)
        lanes = max(s["lanes"] for s in plan["stages"])
        if lanes > 1 and len(args.gpus.split(",")) != 4:
            parser.error("Multiple workers require four GPUs; use --workers 1 for one/two GPUs")
        if args.gui and lanes > 1:
            parser.error("--gui requires one worker")
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    args.runtime = args.runtime.resolve()
    args.candidate_root = args.candidate_root.resolve()
    sim_python = interpreter(args.sim_python, "SIM_PYTHON")
    planner_python = interpreter(args.planner_python, "PLANNER_PYTHON")
    checks = check_environment(sim_python, planner_python, args.runtime)
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi unavailable; use a GPU-enabled platform job")
    if args.check:
        print(json.dumps(dict(status="passed", environments=checks, gpu_tokens=tokens), indent=2))
        return
    subprocess.run([planner_python, "-B", str(HERE / "tools/check_assets.py"),
                    "--runtime", str(args.runtime), "--candidate-root", str(args.candidate_root),
                    "--objects", ",".join(plan["objects"])], check=True)
    args.output = (args.output or HERE / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "PLAN.json", plan)
    write(args.output / "ENVIRONMENT.json", dict(checks=checks, gpu_tokens=tokens,
          inherited_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
          bundle_sha256={p.name: sha(p) for p in HERE.glob("*.py")}))
    print(f"Output: {args.output}\nFixed attempt slots: {plan['total_attempt_slots']}; "
          f"planned case IDs: {plan['unique_scene_count']}; GPUs: {tokens}", flush=True)
    processes, stop = Processes(), threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(args.output, stop), daemon=True)
    monitor.start()
    started = time.monotonic()
    status, error = "completed", None
    try:
        for stage in plan["stages"]:
            run_stage(stage, args, tokens, sim_python, planner_python, processes)
    except BaseException:
        status, error = "interrupted_or_technical_failure", traceback.format_exc()
    finally:
        processes.close()
        stop.set()
        monitor.join(timeout=10)
    collection_wall_s = time.monotonic() - started
    if status == "completed" and not args.skip_validation:
        try:
            validate(args.output, args.runtime, planner_python)
        except BaseException:
            status, error = "validation_failed", traceback.format_exc()
    write(args.output / "RUN.json", dict(status=status, error=error, collection_wall_s=collection_wall_s,
          total_wall_s=time.monotonic()-started, validation_requested=not args.skip_validation))
    from report import summarize
    summary = summarize(args.output)
    smoke_verified = sum(s["requested_state_validated_successes"] for s in summary["stages"])
    if args.smoke and status == "completed" and smoke_verified != 2:
        status, error = "smoke_grasp_failure", "Smoke requires two successful grasps with passed data and requested-initial-state checks; inspect saved failures before expanding the batch."
        receipt = read(args.output / "RUN.json")
        receipt.update(status=status, error=error)
        write(args.output / "RUN.json", receipt)
        summary = summarize(args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("Report: " + str(args.output / "REPORT.md"), flush=True)
    if status != "completed":
        print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
