"""Generate one 200-pose GraspGen pool per mesh, then apply Panda geometry checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

from common import HERE, OBJECTS, local_setting, read, sha, write

sys.path.insert(0, str(HERE / "third_party/GraspGen"))
sys.path.insert(0, str(HERE / "runtime"))


def surface_sample(mesh, count, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    if not mesh.area > 0:
        raise ValueError("Input mesh has zero area")
    face = rng.choice(len(mesh.faces), count, p=mesh.area_faces / mesh.area)
    uv = rng.random((count, 2))
    flip = uv.sum(axis=1) > 1
    uv[flip] = 1 - uv[flip]
    tri = mesh.triangles[face]
    return (tri[:, 0] + uv[:, :1]*(tri[:, 1]-tri[:, 0]) +
            uv[:, 1:]*(tri[:, 2]-tri[:, 0])).astype(np.float32)


def restore_grasps(raw, center):
    import numpy as np
    poses = np.array(raw, dtype=np.float32, copy=True)
    if poses.shape != (200, 4, 4):
        raise ValueError("Expected exactly 200 grasp transforms")
    poses[:, :3, 3] += center
    poses[:, 3, :] = [0, 0, 0, 1]
    if not np.isfinite(poses).all():
        raise ValueError("Nonfinite model output")
    return poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", default="all")
    parser.add_argument("--model-config", type=Path,
                        default=Path(local_setting("GRASPGEN_CONFIG", str(HERE / "models/checkpoints/graspgen_franka_panda.yml"))))
    parser.add_argument("--output-root", type=Path, default=HERE / "generated/candidates")
    parser.add_argument("--gpu", default="0", help="Logical GPU within CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    names = list(OBJECTS) if args.objects == "all" else args.objects.split(",")
    if not names or len(set(names)) != len(names) or any(n not in OBJECTS for n in names):
        parser.error("--objects must select bundled object IDs, or all")
    from common import gpu_tokens
    token = gpu_tokens(args.gpu)[0]
    if "," in args.gpu:
        parser.error("--gpu requires a single logical index")
    os.environ["CUDA_VISIBLE_DEVICES"] = token
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_TF", "0")
    import numpy as np
    import torch
    import trimesh
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.dataset.dataset import collate
    from geometry.api import manifest, prepare_candidates

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_grasp_cfg(str(args.model_config.resolve()))
    if cfg.data.gripper_name != "franka_panda" or int(cfg.data.num_points) != 2048:
        raise ValueError("Use the official Panda checkpoint with 2048 input points")
    identity = {"config": sha(args.model_config), "generator": sha(cfg.eval.checkpoint),
                "discriminator": sha(cfg.discriminator.checkpoint)}
    identity["source"] = __import__("hashlib").sha256(json.dumps({
        str(p.relative_to(HERE / "third_party/GraspGen")): sha(p)
        for p in sorted((HERE / "third_party/GraspGen/grasp_gen").rglob("*.py"))
    }, sort_keys=True).encode()).hexdigest()
    identity["adapter"] = sha(__file__)
    sampler = None
    for name in names:
        asset = manifest()["objects"][name]
        folder = args.output_root.resolve() / "graspgen" / name
        seed = int(asset["candidate_cache_seed"])
        key = dict(mesh_sha256=sha(asset["mesh_path"]), surface_sha256=sha(asset["surface_path"]),
                   parts=asset["parts"], seed=seed, models=identity, points=2048, grasps=200,
                   geometry_source_sha256={p.name: sha(p) for p in (HERE / "runtime/geometry").glob("*.py")},
                   robot_sha256=sha(HERE / "runtime/robot/panda_v2.urdf"))
        report_path = folder / "REPORT.json"
        if folder.exists():
            if report_path.is_file():
                previous = read(report_path)
                if (previous.get("cache_key") == key and
                        previous.get("source_cache_sha256") == sha(folder / "candidates.npz")):
                    if not previous.get("candidates"):
                        raise RuntimeError(f"Existing pool has no qualified candidates: {folder}; no automatic redraw")
                    print(f"CACHE_HIT {name}", flush=True)
                    continue
            raise FileExistsError(f"Existing incomplete/different pool retained: {folder}; choose a new --output-root")
        if sampler is None:
            sampler = GraspGenSampler(cfg)
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        mesh = trimesh.load(asset["mesh_path"], force="mesh", process=False)
        points = surface_sample(mesh, 2048, seed)
        if not np.isfinite(points).all():
            raise ValueError("Nonfinite mesh surface points")
        pc = torch.as_tensor(points, device="cuda", dtype=torch.float32)
        center = pc.mean(dim=0)
        centered = pc - center[None]
        data = dict(task="pick", inputs=torch.cat([centered, torch.zeros_like(centered)], dim=-1), points=centered)
        sampler.model.grasp_generator.num_grasps_per_object = 200
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            prediction, _, _ = sampler.model.infer(collate([data]))
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - started
        raw = prediction["grasps_pred"][0].detach().cpu().numpy()
        scores = prediction["grasp_confidence"][0, :, 0].detach().cpu().numpy().astype(np.float32)
        center_np = center.cpu().numpy()
        poses = restore_grasps(raw, center_np)
        if scores.shape != (200,) or not np.isfinite(scores).all():
            raise ValueError("Expected 200 finite discriminator scores")
        ranks = np.empty(200, dtype=np.int32)
        ranks[np.argsort(-scores, kind="stable")] = np.arange(1, 201, dtype=np.int32)
        report = prepare_candidates(name, poses, scores, original_rank=ranks)
        folder.mkdir(parents=True, exist_ok=False)
        cache = folder / "candidates.npz"
        np.savez_compressed(cache, T_object_grasp=poses, score=scores, original_rank=ranks,
                            generation_index=np.arange(200, dtype=np.int32), points_object=points,
                            center_object=center_np, T_centered_grasp_raw=raw)
        report.update(source_kind="GraspGen official Panda diffusion + discriminator, mesh point cloud",
                      source_cache="candidates.npz", source_cache_sha256=sha(cache),
                      input_mesh_sha256=key["mesh_sha256"], cache_key=key,
                      generation_calls=1, generation_s=generation_s,
                      preprocess="2048 surface points, meters, mean subtracted and restored exactly once; fixed TCP adapter in geometry")
        write(report_path, report)
        print(f"GENERATED {name}: 200 poses, {len(report['candidates'])} geometry-qualified; {folder}", flush=True)
        if not report["candidates"]:
            raise RuntimeError(f"No geometry-qualified grasp for {name}; evidence retained, no automatic redraw")


if __name__ == "__main__":
    main()
