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

    python = interpreter(args.python, "LEROBOT_PYTHON")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
           "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "HF_HOME": str(output / ".cache" / "huggingface")}
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

    for stage, script, arguments in (
        ("selection", "select_sources.py", select),
        ("export", "export_lerobot.py", ["--selection", selection, "--output", portable]),
        ("audit", "audit_lerobot.py", [portable, "--output", audit]),
    ):
        log = output / f"{stage}.log"
        print(f"{stage}: {log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run([python, "-B", str(TOOLS / script), *arguments],
                                    env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"{stage} failed (exit {result.returncode}); see {log}")
    print(f"Dataset: {portable / 'dataset'}\nReadback: {audit}")


if __name__ == "__main__":
    main()
