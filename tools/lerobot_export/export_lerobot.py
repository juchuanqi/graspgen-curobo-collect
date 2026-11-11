"""Export selected platform NPY cameras and NPZ controls to native LeRobot v3."""
from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import time
from pathlib import Path

import datasets
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

from bridge import CAMERAS, feature_spec, physical8, read, sha, verify_inputs, write


def main(args):
    if not __debug__:
        raise RuntimeError("Export validation requires Python assertions to be enabled")
    started = time.monotonic()
    if CODEBASE_VERSION != "v3.0" or importlib.metadata.version("lerobot") != "0.4.4":
        raise ValueError("Use the verified lerobot==0.4.4 / v3.0 environment")
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = read(args.selection)
    if selection["status"] != "selected_before_export" or not selection["episodes"]:
        raise ValueError("The selection must contain eligible episodes")
    for name, expected in selection["tool_sha256"].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError(f"Implementation changed after selection: {name}")
    for name, expected in selection["original_inputs_sha256"].items():
        if sha(name) != expected:
            raise ValueError(f"Run input changed after selection: {name}")
    for episode in selection["episodes"]:
        verify_inputs(episode)
    args.output.mkdir(parents=True)
    shutil.copy2(args.selection, args.output / "SELECTION.json")
    for name in selection["tool_sha256"]:
        shutil.copy2(Path(__file__).parent / name, args.output / name)
    torch.set_num_threads(2)
    datasets.disable_progress_bars()
    writer = LeRobotDataset.create(repo_id=args.repo_id, root=args.output / "dataset", fps=60,
                                  robot_type="isaaclab_panda_native_parallel_fingers",
                                  features=feature_spec(), use_videos=False,
                                  image_writer_processes=0, image_writer_threads=0)
    records = []
    try:
        for episode in selection["episodes"]:
            source = Path(episode["source"])
            n = episode["length"]
            with np.load(source / "trajectory.npz", allow_pickle=False) as archive:
                raw = {k: archive[k] for k in ("qpos", "qvel", "action_targets", "position_targets", "sim_time", "phase")}
            assert raw["qpos"].shape == raw["qvel"].shape == (n + 1, 9)
            assert raw["action_targets"].shape == (n, 9)
            np.testing.assert_array_equal(raw["action_targets"], raw["position_targets"][1:])
            np.testing.assert_allclose(raw["sim_time"] - raw["sim_time"][0], np.arange(n + 1) / 60, atol=2e-6, rtol=0)
            assert np.isfinite(raw["qpos"]).all() and np.isfinite(raw["action_targets"]).all()
            images = {c: np.load(source / "cameras" / f"{c}.npy", mmap_mode="r", allow_pickle=False) for c in CAMERAS}
            assert all(v.shape == (n + 1, 256, 256, 3) and v.dtype == np.uint8 for v in images.values())
            sidecar = args.output / "sidecars" / f"episode_{episode['export_episode_index']:06d}"
            sidecar.mkdir(parents=True)
            for name, expected in episode["source_sha256"].items():
                if name.startswith("cameras/"):
                    continue
                target = sidecar / name
                shutil.copy2(source / name, target)
                assert sha(target) == expected
            for camera, frames in images.items():
                np.save(sidecar / f"terminal_{camera}.npy", frames[n], allow_pickle=False)
            states, actions = physical8(raw["qpos"]), physical8(raw["action_targets"])
            for t in range(n):
                frame = {
                    "task": episode["task"], "observation.state": states[t], "action": actions[t],
                    "aux.qpos_raw": raw["qpos"][t].astype(np.float32),
                    "aux.qvel_raw": raw["qvel"][t].astype(np.float32),
                    "aux.position_target_raw": raw["action_targets"][t].astype(np.float32),
                    "aux.phase": np.array([raw["phase"][t]], dtype=np.int64),
                    "aux.sim_time_s": np.array([raw["sim_time"][t]], dtype=np.float64),
                    "next.done": np.array([t == n - 1], dtype=bool),
                    "episode_success": np.array([True], dtype=bool),
                }
                for camera, frames in images.items():
                    frame[f"observation.images.{camera}"] = np.array(frames[t], copy=True)
                writer.add_frame(frame)
            writer.save_episode()
            records.append(dict(episode, sidecar=str(sidecar.relative_to(args.output))))
            print("EXPORTED", episode["case_id"], n, flush=True)
    finally:
        writer.finalize()
        writer.stop_image_writer()
    info = read(args.output / "dataset/meta/info.json")
    assert info["total_episodes"] == len(records)
    assert info["total_frames"] == sum(e["length"] for e in records)
    write(args.output / "EXPORT_MANIFEST.json", dict(
        status="exported_awaiting_readback", repo_id=args.repo_id, dataset_root="dataset",
        dataset_format=CODEBASE_VERSION, lerobot_package=importlib.metadata.version("lerobot"),
        fps=60, episodes=records, rows=info["total_frames"], role=selection["role"],
        image_storage="lossless PNG bytes embedded in Parquet",
        state_mapping="qpos[t, :7] + sum(qpos[t, 7:9]); 7 measured radians + total opening meters",
        action_mapping="action_targets[t, :7] + sum(action_targets[t, 7:9]); absolute 7 joint position targets + total target opening",
        terminal_policy="N actions with obs[0:N]; original N+1 raw arrays and terminal camera NPYs retained in sidecars",
        training_inputs=["observation.state", "observation.images.base_camera", "observation.images.wrist_camera"],
        training_target="action", labels_excluded_from_training=["aux.*", "next.done", "episode_success"],
        selection_sha256=sha(args.output / "SELECTION.json"), wall_s=time.monotonic() - started,
        new_physics=False, training=False))
    print("EXPORT_COMPLETED", info["total_frames"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/graspgen_curobo_panda")
    main(parser.parse_args())
