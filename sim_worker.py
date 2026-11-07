"""One persistent Isaac environment, one asset, multiple fixed episode slots."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

from common import PlannerClient, read, sha, write
from initial_state import install_reset_guard


def episode_options(runtime, output, job, index):
    return SimpleNamespace(
        output=output, candidate_root=Path(job["candidate_root"]), role="collection",
        method="graspgen", qualify_only=False, io_mode="optimized",
        audit_io=False, full_settle=False, cold_planner=not job["persistent"],
        reused_env=index > 0, stage=job["stage"], worker_id=job["worker_id"])


def main():
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--address", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job = read(args.job)
    if len({c["object_id"] for c in job["cases"]}) != 1:
        raise ValueError("A persistent worker requires one asset")
    launcher = AppLauncher(args)
    app, env = launcher.app, None
    completed = []
    try:
        sys.path.insert(0, str(args.runtime))
        sys.path.insert(0, str(args.runtime / "collector"))
        import numpy as np
        import torch
        from collection_env import PickupCfg, URDF, NAMES, LINKS, ROOT, TRANSFER
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
        import omni.usd
        from pxr import PhysxSchema, UsdPhysics
        solvers = {str(p.GetPath()): str(PhysxSchema.PhysxSceneAPI(p).GetSolverTypeAttr().Get())
                   for p in omni.usd.get_context().get_stage().Traverse() if p.IsA(UsdPhysics.Scene)}
        if not solvers or set(solvers.values()) != {"PGS"}:
            raise ValueError("Requested PGS does not match native USD solver")
        native = dict(solver_scenes=solvers, contact_force_bound=env.contact_force_bound)
        for field, getter in (("mass_kg", "get_masses"), ("inertias", "get_inertias"),
                              ("materials", "get_material_properties")):
            values = getattr(env.object.root_physx_view, getter)()
            native[field] = values.cpu().numpy().tolist() if hasattr(values, "cpu") else np.asarray(values).tolist()
        initialization_s = time.monotonic() - entered
        write(args.output / "READY.json", dict(pid=os.getpid(), initialization_s=initialization_s,
                                               object_id=first["object_id"], num_envs=1))
        source_paths = [p for folder in (args.runtime / "collector", args.runtime / "geometry")
                        for p in folder.glob("*.py") if p.name != "__init__.py"]
        source_paths.extend(Path(__file__).with_name(name) for name in
                            ("sim_worker.py", "common.py", "initial_state.py"))
        source_hashes = {str(p): sha(p) for p in source_paths}
        planner = PlannerClient(read(args.address))
        reference_xy = TRANSFER["object_reference_xy_m"]

        def raw_object_pose():
            xyzw = env.object.root_physx_view.get_transforms()[0].cpu().tolist()
            return [*xyzw[:3], xyzw[6], *xyzw[3:6]]

        casegrid = dict(candidate_budget=dict(
            raw_model_candidates_per_object=200, scene_qualified_candidates=50,
            max_full_planned_candidates_per_scene=5,
            max_planning_wall_s_per_candidate_including_retries=30,
            max_control_steps_per_sequence=2400, max_physical_attempts_per_scene_method=1))
        for index, entry in enumerate(job["cases"]):
            out = args.output / entry["case_id"]
            out.mkdir(exist_ok=False)
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
            print(f"EPISODE {entry['case_id']} success={result['success']} "
                  f"seconds={result['episode_wall_s']:.2f} reason={result['reason']}", flush=True)
            if error:
                raise RuntimeError("Technical episode failure retained in " + str(out))
            if episode.reason == "requested_initial_state_mismatch":
                raise RuntimeError("Requested initial state failed; diagnostics retained in " + str(out))
    except BaseException:
        write(args.output / "WORKER_ERROR.json", dict(error=traceback.format_exc(), completed=completed))
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            app.close()
    write(args.output / "EXIT.json", dict(status="completed", completed=completed,
                                          process_wall_s=time.monotonic() - entered))


if __name__ == "__main__":
    main()
