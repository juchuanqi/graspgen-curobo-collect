"""One persistent Isaac environment, one asset, multiple fixed episode slots."""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import weakref
from types import SimpleNamespace

from common import PlannerClient, read, sha, write
from initial_state import install_reset_guard
from recovery import durable_write, initialization_resource_error


def episode_options(runtime, output, job, index):
    return SimpleNamespace(
        output=output, candidate_root=Path(job["candidate_root"]), role="collection",
        method="graspgen", qualify_only=False, io_mode="optimized",
        audit_io=False, full_settle=False, cold_planner=not job["persistent"],
        reused_env=index > 0, stage=job["stage"], worker_id=job["worker_id"],
        overlap_lift=job.get('execution_profile', 'baseline') == 'accelerated',
        execution_profile=job.get('execution_profile', 'baseline'),
        grasp_symmetry=job.get("grasp_symmetry","off"))


def main():
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--address", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job = read(args.job)
    if len({c["object_id"] for c in job["cases"]}) != 1:
        raise ValueError("A persistent worker requires one asset")
    app, env = None, None
    completed = []
    phase = 'initializing'
    def worker_state(state, **extra):
        durable_write(args.output / 'WORKER_STATE.json', dict(
            protocol=1, pid=os.getpid(), state=state, completed=completed,
            updated_unix_s=time.time(), **extra))
    worker_state(phase)
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        launcher = AppLauncher(args)
        app = launcher.app
        sys.path.insert(0, str(args.runtime))
        sys.path.insert(0, str(args.runtime / "collector"))
        import numpy as np
        import torch
        from collection_env import PickupCfg, URDF, NAMES, LINKS, ROOT, TRANSFER
        accelerated = job.get('execution_profile', 'baseline') == 'accelerated'
        if accelerated:
            from fast_env_v6 import PickupEnv
        else:
            from fast_env_v2 import PickupEnv
        from episode_factory import create_episode
        from geometry.api import manifest
        torch.set_num_threads(2)
        first = job["cases"][0]
        asset = manifest()["objects"][first["object_id"]]
        cfg = PickupCfg()
        cfg.sim.device = args.device
        cfg.seed = int(first["scene_seed"])
        env = PickupEnv(cfg, object_id=first["object_id"], asset=asset, occlusion="none")
        if accelerated:
            env.configure_sensors('ring')
            contact_setup = env.configure_contacts('graph')
            env.configure_cameras('manual')
            actuator_setup = env.configure_actuators('graph')
            write(args.output / 'ACCELERATION_SETUP.json', dict(
                execution_profile='accelerated', sensor_schedule='ring', camera_schedule='manual',
                contacts=contact_setup, actuators=actuator_setup, closure_lift_overlap=True))
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics
        solvers = {str(p.GetPath()): str(PhysxSchema.PhysxSceneAPI(p).GetSolverTypeAttr().Get())
                   for p in omni.usd.get_context().get_stage().Traverse() if p.IsA(UsdPhysics.Scene)}
        if not solvers or set(solvers.values()) != {"PGS"}:
            raise ValueError("Requested PGS does not match native USD solver")
        native = dict(solver_scenes=solvers, contact_force_bound=env.contact_force_bound)
        for field, getter in (("mass_kg", "get_masses"), ("inertias", "get_inertias"),
                              ("materials", "get_material_properties"), ("coms_actor_local_pose_xyzw", "get_coms")):
            values = getattr(env.object.root_physx_view, getter)()
            native[field] = values.cpu().numpy().tolist() if hasattr(values, "cpu") else np.asarray(values).tolist()
        initialization_s = time.monotonic() - entered
        write(args.output / "READY.json", dict(pid=os.getpid(), initialization_s=initialization_s,
                                               object_id=first["object_id"], num_envs=1))
        phase = 'ready'
        worker_state(phase)
        source_paths = [p for folder in (args.runtime / "collector", args.runtime / "geometry")
                        for p in folder.glob("*.py") if p.name != "__init__.py"]
        source_paths.extend(Path(__file__).with_name(name) for name in
                            ("sim_worker.py", "common.py", "initial_state.py", "recovery.py"))
        source_hashes = {str(p): sha(p) for p in source_paths}
        planner = PlannerClient(read(args.address))
        reference_xy = TRANSFER["object_reference_xy_m"]

        def raw_object_pose():
            xyzw = env.object.root_physx_view.get_transforms()[0].cpu().tolist()
            return [*xyzw[:3], xyzw[6], *xyzw[3:6]]

        casegrid = dict(candidate_budget=dict(
            raw_model_candidates_per_object=200,
            scene_qualified_candidates=100 if job.get("grasp_symmetry")=="near-first" else 50,
            grasp_symmetry=job.get("grasp_symmetry","off"),
            scene_qualified_candidates_per_orientation_group=50,
            max_full_planned_candidates_per_scene=5,
            max_planning_wall_s_per_candidate_including_retries=30,
            max_control_steps_per_sequence=2400, max_physical_attempts_per_scene_method=1))
        for index, entry in enumerate(job["cases"]):
            slot_started_ns = time.monotonic_ns()
            out = args.output / entry["case_id"]
            out.mkdir(exist_ok=False)
            phase = 'running'
            worker_state(phase, case_id=entry['case_id'])
            def slot_state(state, **extra):
                durable_write(out / 'SLOT_STATE.json', dict(
                    protocol=1, case_id=entry['case_id'], state=state,
                    slot_started_ns=slot_started_ns,
                    updated_unix_s=time.time(), **extra))
            slot_state('initializing')
            write(out / "ENTRY.json", entry)
            write(out / "NATIVE_PHYSICS.json", native)
            write(out / "SOURCE_AT_LAUNCH.json", source_hashes)
            snapshot = out / "source_snapshot"
            snapshot.mkdir()
            for source in source_paths:
                shutil.copy2(source, snapshot / source.name)
            write(out / "ENVIRONMENT.json", dict(
                physical_environment="Isaac Lab DirectRLEnv / PhysX", robot="Panda",
                urdf=str(URDF), asset=asset, joint_names=NAMES, contact_links=LINKS,
                physics_hz=480, control_hz=60, rgb_hz=60, camera_resolution=[256, 256],
                arm_pd=[10000, 200], finger_pd=[400, 20], physics_self_collision=False,
                self_collision_check="cuRobo commanded-path checks retained",
                candidate_source="fixed mesh-derived GraspGen cache"))
            options = episode_options(args.runtime, out, job, index)
            episode = create_episode(env, entry, asset, options, casegrid, planner,
                                     entered, source_hashes, initialization_s)
            original_status = episode.status
            def status(state):
                if state.startswith('planning_'):
                    slot_state('planning', detail=state)
                elif state.startswith('executing_'):
                    slot_state('executing', detail=state)
                original_status(state)
            episode.status = status
            original_plan = episode.plan
            def plan(command, **kwargs):
                result = original_plan(command, **kwargs)
                slot_state('executing', detail='after_' + command)
                return result
            episode.plan = plan
            install_reset_guard(episode, env, entry, asset, reference_xy, out,
                                raw_object_pose, prime_first=index == 0)
            error = None
            try:
                episode.rollout()
            except BaseException:
                error = traceback.format_exc()
                write(out / "ERROR.json", dict(error=error))
                episode.reason, episode.success = "technical_exception", False
            result = episode.save(error)
            control = np.asarray(episode.control_timings, dtype=float).reshape(-1, 4)
            write(out / "PERFORMANCE.json", dict(
                execution_step_s=control[:, :2].sum(axis=1).tolist(),
                phase_counts={str(p): episode.phases.count(p) for p in range(5)},
                initial_qpos=np.asarray(episode.rows[0]['qpos']).tolist() if episode.rows else None,
                initial_object_pose=np.asarray(episode.rows[0]['object_pose']).tolist() if episode.rows else None))
            completed.append(dict(case_id=entry["case_id"], success=result["success"],
                                  reason=result["reason"], output=str(out),
                                  episode_wall_s=result["episode_wall_s"]))
            write(args.output / "PROGRESS.json", dict(completed=completed, planned=len(job["cases"])))
            slot_state('saved' if error is None else 'technical_failure',
                       saved_monotonic_ns=time.monotonic_ns(),
                       success=result['success'], reason=result['reason'])
            phase = 'ready'
            worker_state(phase)
            print(f"EPISODE {entry['case_id']} success={result['success']} "
                  f"seconds={result['episode_wall_s']:.2f} reason={result['reason']}", flush=True)
            if error:
                raise RuntimeError("Technical episode failure retained in " + str(out))
            if episode.reason == "requested_initial_state_mismatch":
                raise RuntimeError("Requested initial state failed; diagnostics retained in " + str(out))
            if accelerated:
                reference = weakref.ref(episode)
                del episode, original_status, original_plan, status, plan
                cleanup_started = time.monotonic()
                collected = gc.collect()
                memory = dict(case_id=entry['case_id'], monotonic_ns=time.monotonic_ns(),
                    episode_object_released=reference() is None, gc_collected=collected,
                    gc_wall_s=time.monotonic() - cleanup_started,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(env.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(env.device))
                for line in Path('/proc/self/status').read_text().splitlines():
                    if line.startswith(('VmRSS:', 'VmHWM:')):
                        memory[line.split(':')[0]] = int(line.split()[1]) * 1024
                with (args.output / 'MEMORY.jsonl').open('a') as stream:
                    import json
                    stream.write(json.dumps(memory) + '\n')
            if accelerated and job.get('validation_handoffs'):
                # Outside the episode directory: saved raw evidence is immutable.
                accepted = Path(job['validation_handoffs']) / f"{entry['case_id']}.json"
                deadline = time.monotonic() + job.get('handoff_timeout_s', 600)
                while not accepted.exists():
                    if Path(job['validation_abort']).exists():
                        raise RuntimeError('Validation pipeline aborted; no next physical slot')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Saved-record validation acceptance timed out')
                    time.sleep(.05)
                if read(accepted).get('status') != 'accepted':
                    raise RuntimeError('Saved record rejected by validation pipeline')
                if index == 0 and len(job['cases']) > 1:
                    while not Path(job['measurement_barrier']).exists():
                        if Path(job['validation_abort']).exists():
                            raise RuntimeError('Warmup validation failed; no measurement execution')
                        if time.monotonic() > deadline:
                            raise TimeoutError('Warmup barrier timed out')
                        time.sleep(.05)
        # app.close may exit Python; completion must precede application cleanup.
        write(args.output / 'WORKER_COMPLETE.json', dict(status='completed', completed=completed,
              process_wall_before_cleanup_s=time.monotonic() - entered))
        worker_state('completed')
    except BaseException:
        error = traceback.format_exc()
        write(args.output / "WORKER_ERROR.json", dict(error=error, completed=completed))
        worker_state('failed', failed_phase=phase, error=error,
                     recoverable_initialization=phase == 'initializing' and initialization_resource_error(error))
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
    write(args.output / "EXIT.json", dict(status="completed", completed=completed,
                                          process_wall_s=time.monotonic() - entered))
    worker_state('completed')


if __name__ == "__main__":
    main()
