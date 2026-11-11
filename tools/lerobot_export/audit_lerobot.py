"""Independently compare every official-loader row with original NPZ/NPY data."""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.datasets import lerobot_dataset as official
from PIL import Image

from bridge import CAMERAS, read, sha, verify_inputs, write


def array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def rgb(value):
    value = array(value)
    assert value.shape == (3, 256, 256) and value.dtype == np.float32
    assert np.isfinite(value).all() and value.min() >= 0 and value.max() <= 1
    return np.rint(value.transpose(1, 2, 0) * 255).astype(np.uint8)


def audit(package):
    if not __debug__:
        raise RuntimeError("Readback validation requires Python assertions to be enabled")
    manifest = read(package / "EXPORT_MANIFEST.json")
    if importlib.metadata.version("lerobot") != manifest["lerobot_package"] or manifest["lerobot_package"] != "0.4.4":
        raise ValueError("Use lerobot==0.4.4 for readback")
    assert sha(package / "SELECTION.json") == manifest["selection_sha256"]
    assert official.CODEBASE_VERSION == manifest["dataset_format"] == "v3.0"
    dataset_root = package / manifest["dataset_root"]

    def load(**kwargs):
        return official.LeRobotDataset(manifest["repo_id"], root=dataset_root, download_videos=False, **kwargs)

    ds = load()
    assert ds.fps == 60 and ds.num_episodes == len(manifest["episodes"])
    assert len(ds) == manifest["rows"] == sum(e["length"] for e in manifest["episodes"])
    assert set(ds.meta.image_keys) == {f"observation.images.{c}" for c in CAMERAS}
    assert not ds.meta.video_keys
    offsets = {"observation.state": [-2, -1, 0], "action": [0, 1, 15]}
    padded = load(delta_timestamps={k: [v / 60 for v in vs] for k, vs in offsets.items()})
    report = dict(episodes=[], full_rows=0, rgb_comparisons=0, terminal_rgb_comparisons=0, padding_checks=0)
    start = 0
    for index, episode in enumerate(manifest["episodes"]):
        verify_inputs(episode)
        n = episode["length"]
        source = Path(episode["source"])
        sidecar = package / episode["sidecar"]
        for name, expected in episode["source_sha256"].items():
            if not name.startswith("cameras/"):
                assert sha(sidecar / name) == expected
        with np.load(source / "trajectory.npz", allow_pickle=False) as raw:
            joints = raw["qpos"]
            target = raw["action_targets"]
            velocity = raw["qvel"]
            phase, source_time = raw["phase"], raw["sim_time"]
            # Build expectations here without using the exporter's physical8 mapping.
            states = np.column_stack((joints[:, :7], joints[:, 7] + joints[:, 8])).astype(np.float32)
            actions = np.column_stack((target[:, :7], target[:, 7] + target[:, 8])).astype(np.float32)
            assert joints.shape == (n + 1, 9) and target.shape == (n, 9)
            np.testing.assert_array_equal(target, raw["position_targets"][1:])
        cameras = {c: np.load(source / "cameras" / f"{c}.npy", mmap_mode="r", allow_pickle=False) for c in CAMERAS}
        epmeta = ds.meta.episodes[index]
        assert int(epmeta["dataset_from_index"]) == start
        assert int(epmeta["dataset_to_index"]) == start + n
        assert int(epmeta["length"]) == n
        for t in range(n):
            row = ds[start + t]
            expected = {"observation.state": states[t], "action": actions[t],
                        "aux.qpos_raw": joints[t], "aux.qvel_raw": velocity[t],
                        "aux.position_target_raw": target[t]}
            for key, value in expected.items():
                actual = array(row[key])
                assert actual.dtype == value.dtype and actual.shape == value.shape
                np.testing.assert_array_equal(actual, value)
            assert int(row["episode_index"].item()) == index
            assert int(row["frame_index"].item()) == t
            assert int(row["index"].item()) == start + t
            assert int(row["aux.phase"].item()) == int(phase[t])
            assert bool(row["next.done"].item()) == (t == n - 1)
            assert bool(row["episode_success"].item())
            assert row["task"] == episode["task"]
            assert abs(row["timestamp"].item() - t / 60) < 1e-5
            # The official loader infers float32 for Python float values.
            np.testing.assert_array_equal(array(row["aux.sim_time_s"]), np.asarray(source_time[t], dtype=np.float32))
            for camera in CAMERAS:
                np.testing.assert_array_equal(rgb(row[f"observation.images.{camera}"]), cameras[camera][t])
                report["rgb_comparisons"] += 1
            report["full_rows"] += 1
        for camera in CAMERAS:
            np.testing.assert_array_equal(np.load(sidecar / f"terminal_{camera}.npy", allow_pickle=False), cameras[camera][n])
            report["terminal_rgb_comparisons"] += 1
        for t in (0, 1, n - 2, n - 1):
            row = padded[start + t]
            for key, values in offsets.items():
                unbounded = t + np.asarray(values)
                np.testing.assert_array_equal(array(row[key + "_is_pad"]), (unbounded < 0) | (unbounded >= n))
                expected = (states if key == "observation.state" else actions)[np.clip(unbounded, 0, n - 1)]
                np.testing.assert_array_equal(array(row[key]), expected)
                report["padding_checks"] += 1
        report["episodes"].append(dict(episode_index=index, case_id=episode["case_id"], rows=n,
                                       state_action_exact=True, every_rgb_lossless=True))
        start += n
        print("AUDITED", episode["case_id"], n, flush=True)
    # Embedded image bytes and stored double timestamps are checked independently
    # of the loader; no external PNG filename is needed to read any training row.
    parquet_rows = 0
    time_by_episode = {}
    for index, episode in enumerate(manifest["episodes"]):
        with np.load(Path(episode["source"]) / "trajectory.npz", allow_pickle=False) as raw:
            time_by_episode[index] = raw["sim_time"]
    for path in sorted((dataset_root / "data").glob("*/*.parquet")):
        parquet = pq.ParquetFile(path)
        assert str(parquet.schema_arrow.field("aux.sim_time_s").type) == "double"
        columns = ["episode_index", "frame_index", "aux.sim_time_s", *ds.meta.image_keys]
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                assert row["aux.sim_time_s"] == time_by_episode[row["episode_index"]][row["frame_index"]]
                for key in ds.meta.image_keys:
                    encoded = row[key]["bytes"]
                    assert encoded and encoded.startswith(b"\x89PNG\r\n\x1a\n")
                    with Image.open(io.BytesIO(encoded)) as image:
                        assert image.size == (256, 256) and image.mode == "RGB"
                parquet_rows += 1
    assert parquet_rows == len(ds)
    last = len(manifest["episodes"]) - 1
    subset = load(episodes=[last], delta_timestamps={k: [v / 60 for v in vs] for k, vs in offsets.items()})
    assert len(subset) == manifest["episodes"][last]["length"]
    assert int(subset[0]["index"].item()) == len(ds) - len(subset)
    assert int(subset[-1]["index"].item()) == len(ds) - 1
    np.testing.assert_array_equal(array(subset[0]["observation.state_is_pad"]), [True, True, False])
    np.testing.assert_array_equal(array(subset[-1]["action_is_pad"]), [False, True, True])
    report.update(parquet_rows=parquet_rows, embedded_png_every_row=True,
                  exact_float64_times_in_storage=True, filtered_episode_indices_and_padding_passed=True,
                  terminal_observations_preserved_without_extra_actions=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        report = audit(args.package)
        report.update(status="passed", new_physics=False, training=False,
                      official_loader_sha256=sha(official.__file__), audit_source_sha256=sha(__file__),
                      wall_s=time.monotonic() - started)
        write(args.output, report)
    except Exception:
        write(args.output, dict(status="failed", traceback=traceback.format_exc(), wall_s=time.monotonic() - started))
        raise
    print({k: report[k] for k in ("status", "full_rows", "rgb_comparisons", "padding_checks")}, flush=True)
