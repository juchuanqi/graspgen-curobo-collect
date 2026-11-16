"""Select, export and independently verify a completed dual RGB collection."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[1]
sys.path.insert(0, str(ROOT))
from collect import interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Raw collection directory")
    parser.add_argument("--output", type=Path, required=True, help="New export directory")
    parser.add_argument("--python", help="Python with lerobot==0.4.4 (or set LEROBOT_PYTHON)")
    parser.add_argument("--runtime", type=Path, default=ROOT / "runtime",
                        help="Runtime config used by the source collection")
    parser.add_argument("--stage", help="Stage to export if the plan contains more than one")
    parser.add_argument("--per-object", type=int, help="Limit selection for a pilot export")
    parser.add_argument("--workers", type=int, default=1,
                        help="1: one package; >1: one fully audited package per object, exported concurrently")
    parser.add_argument('--allow-legacy-unverified', action='store_true', help='Historical format debugging only')
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if not (run / "PLAN.json").is_file() or not (run / "RUN.json").is_file():
        parser.error("--run must contain PLAN.json and RUN.json")
    if output == run or output.is_relative_to(run):
        parser.error("--output must be outside the source run")
    if output.exists():
        parser.error(f"Output already exists: {output}")
    if args.per_object is not None and args.per_object < 1:
        parser.error("--per-object must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")

    python = interpreter(args.python, "LEROBOT_PYTHON")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HOME": str(output / ".cache" / "huggingface"),
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "USE_TF": "0", "USE_FLAX": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    installed = subprocess.check_output(
        [python, "-B", "-c", "from importlib.metadata import version; print(version('lerobot'))"],
        text=True, env=env,
    ).strip()
    if installed != "0.4.4":
        parser.error(f"Use lerobot==0.4.4 in {python}; found {installed}")
    output.mkdir(parents=True, exist_ok=False)
    selection = output / "SELECTION.json"
    portable = output / "portable"
    audit = output / "AUDIT.json"
    select = ["--run", run, "--runtime", args.runtime.resolve(), "--output", selection]
    if args.stage:
        select.extend(["--stage", args.stage])
    if args.per_object is not None:
        select.extend(["--per-object", str(args.per_object)])
    if args.allow_legacy_unverified:
        select.append('--allow-legacy-unverified')

    stages = [("selection", "select_sources.py", select)]
    if args.workers > 1:
        stages.append(("shards", "sharded_export.py", ["--selection", selection, "--output", output,
                                                      "--workers", str(args.workers)]))
    else:
        stages.extend([
        ("export", "export_lerobot.py", ["--selection", selection, "--output", portable]),
        ("audit", "audit_lerobot.py", [portable, "--output", audit]),
        ])
    for stage, script, arguments in stages:
        log = output / f"{stage}.log"
        print(f"{stage}: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run([python, "-B", str(TOOLS / script), *arguments],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{stage} failed (exit {result.returncode}); see {log}")
    if args.workers > 1:
        (portable / 'DATA_USAGE.md').write_text('''# Local sharded grasp dataset

Use `lerobot==0.4.4`. Every `shards/<object>/dataset` is a standalone LeRobot v3
dataset with lossless embedded PNG images. Each shard passed full official-loader
readback and independent statistics. `SHARDS.json` binds their audit receipts and
records the exact union of the global `SELECTION.json`; excluded slots remain there.

Load all objects together with the official multi-dataset loader:

```python
import json
from pathlib import Path
from lerobot.datasets.lerobot_dataset import MultiLeRobotDataset

portable = Path('/absolute/path/to/portable')
manifest = json.loads((portable / 'SHARDS.json').read_text())
data = MultiLeRobotDataset(
    repo_ids=[f"{s['object_id']}/dataset" for s in manifest['shards']],
    root=portable / 'shards',
    delta_timestamps={'action': [i / 60 for i in range(16)]},
)
# Pass data.stats to the policy's official preprocessor for joint normalization.
```

The package uses relative shard paths and can be moved as a whole. Each episode
has its own indices and padding masks; never join action chunks across episodes.
Control and RGB are 60 Hz. State/action are float32 8D: seven arm angles/absolute
position targets in radians plus measured/target total finger opening in metres.
Two 256x256 RGB observations and the measured state are model inputs. Actions are
position targets, not velocity or delta. N actions pair with observations 0:N;
terminal observation N and physical evidence remain in each shard's sidecars.
Training does not need the original raw run. Repeating raw-data comparison does.

The project train_check.py accepts this portable directory and its parent AUDIT.json
directly. A short ACT check validates data consumption, not closed-loop performance.
''')
        print(f"Shards: {portable / 'SHARDS.json'}\nReadback: {audit}")
        return
    (portable / 'DATA_USAGE.md').write_text('''# Local grasp dataset

Use `lerobot==0.4.4`. The `dataset/` directory is native LeRobot v3.0 and contains
its Parquet data, lossless embedded PNG images, episode metadata and corrected
normalization statistics. Training can read this directory without the simulator.
For example:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
data = LeRobotDataset("local/graspgen_curobo_panda", root="/absolute/path/to/dataset")
```

- Sampling is 60 Hz control/RGB and 480 Hz physics, with eight physical substeps.
- `observation.state` is float32, shape (8,): Panda joints 1 through 7 in radians,
  then the measured total opening of the two fingers in metres.
- `action` has the same order and units: seven absolute joint position targets,
  followed by the target total finger opening. It is not a velocity or delta.
- Row t contains the observation before action t. The action matches the saved
  position target at observation t+1. Each N-action episode has N training rows;
  observation N is retained in `sidecars/` and is not another action.
- Base and wrist RGB images are synchronized uint8 256x256x3 in storage; the
  official loader returns float32 CHW images in [0,1]. Use the stored statistics
  with the target policy's official preprocessor.
- Official delta-timestamp loading pads within each episode and returns padding
  masks. Never construct action chunks across episode boundaries.
- `aux.*` retains raw nine-joint states/targets and simulation timestamps.
  Two raw finger coordinates become one total opening in the training fields.

Only strictly qualified successes are exported. Fixed-slot failures and missing
infrastructure outcomes remain in the source run and its slot ledger. Counts of
exported successes alone do not define a batch success rate.

`SELECTION.json`, `EXPORT_MANIFEST.json` and sidecars retain source identity and
provenance. Repeating the independent raw-data audit additionally requires the
original source records at the paths in the selection manifest; training itself
does not. The parent export directory contains `AUDIT.json`; a pipeline delivery
also seals its final files with `COMMITTED.json`.

An ACT short check, when requested, verifies data consumption and finite updates.
It is recorded separately and does not measure closed-loop policy performance.
''')
    print(f"Dataset: {portable / 'dataset'}\nReadback: {audit}")


if __name__ == "__main__":
    main()
