"""Write an ignored local environment config when the existing shared disk is mounted."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True, help="User directory containing project/ and venvs/")
    parser.add_argument("--graspgen-python", type=Path, help="Override the shared GraspGen interpreter")
    parser.add_argument("--output", type=Path, default=HERE / "local_env.json")
    args = parser.parse_args()
    root = args.shared_root.expanduser().resolve()
    environment = root / "venvs/graspbench_20261107"
    result = dict(
        SIM_PYTHON=str(environment / "simulation/bin/python"),
        PLANNER_PYTHON=str(environment / "planning/bin/python"),
        GRASPGEN_PYTHON=str(args.graspgen_python or environment / "graspgen/bin/python"),
        GRASPGEN_CONFIG=str(root / "models/graspgen/checkpoints/graspgen_franka_panda.yml"))
    missing = [f"{key}={path}" for key, path in result.items() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Shared paths unavailable; configure the local environments described in docs/INSTALL.md: " + "; ".join(missing))
    write(args.output, result)
    print("Configured", args.output)


if __name__ == "__main__":
    main()
