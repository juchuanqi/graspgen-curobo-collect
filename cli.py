"""Small standard-library launcher for separate simulation, planning and model environments."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from common import HERE, local_setting
from collect import environment, interpreter

COMMANDS = {
    "collect": "Collect raw episodes and validate them",
    "prepare": "Generate official GraspGen candidate pools",
    "check": "Check local environments, model files and assets",
    "export": "Export validated episodes and audit LeRobot data",
    "pipeline": "Run recoverable collection and export (use --data-only for data delivery)",
    "download-models": "Download or verify official model weights",
    "benchmark": "Run the historical cold/warm/parallel benchmark",
}


def command_arguments(values):
    """Keep legacy collection flags while exposing the available commands."""
    parser = argparse.ArgumentParser(
        prog="bash run.sh", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(f"  {name:17} {description}" for name, description in COMMANDS.items())
        + "\n\nUse bash run.sh COMMAND --help for command options."
        + "\nWithout a command, collection flags retain their original meaning.")
    if values and values[0] in {"-h", "--help"}:
        parser.parse_args(values)
    if values and not values[0].startswith("-"):
        if values[0] not in COMMANDS:
            parser.error(f"unknown command: {values[0]!r}")
        return values[0], values[1:]
    return "collect", list(values)


def graspgen_check(python):
    config = Path(local_setting("GRASPGEN_CONFIG", str(HERE / "models/checkpoints/graspgen_franka_panda.yml")))
    code = '''
import importlib.util, json, sys
from pathlib import Path
modules = ["torch", "numpy", "scipy", "trimesh", "grasp_gen", "pointnet2_ops", "spconv", "torch_scatter", "diffusers", "omegaconf"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
missing_files = []
config = Path(sys.argv[1])
if not config.is_file():
    missing_files.append(str(config))
else:
    import yaml
    values = yaml.safe_load(config.read_text())
    for name in ('eval', 'discriminator'):
        path = config.parent / values[name]['checkpoint']
        if not path.is_file(): missing_files.append(str(path))
import_error = None
if not missing and not missing_files:
    try:
        import torch
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info
        assert torch.ones(2, device='cpu').sum().item() == 2
        cfg = load_grasp_cfg(str(config))
        assert cfg.data.gripper_name == 'franka_panda' and int(cfg.data.num_points) == 2048
        assert len(get_gripper_info('franka_panda').collision_mesh.vertices) > 0
    except Exception as error:
        import_error = repr(error)
print(json.dumps(dict(python=sys.executable, missing=missing, missing_model_files=missing_files,
                      cpu_import_error=import_error)))
sys.exit(bool(missing) or bool(missing_files) or bool(import_error))
'''
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "USE_FLAX": "0", "USE_TF": "0",
           "PYTHONPATH": str(HERE / "third_party/GraspGen") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([python, "-B", "-c", code, str(config)], env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    command, values = command_arguments(sys.argv[1:])
    help_requested = any(value in {"-h", "--help"} for value in values)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if command in {"export", "pipeline"}:
        script = HERE / "tools" / ("lerobot_export/run.py" if command == "export" else "pipeline.py")
        os.execve(sys.executable, [sys.executable, "-B", "-u", str(script), *values], env)
    if command in {"prepare", "download-models"}:
        env = environment(os.environ.get("CUDA_VISIBLE_DEVICES", ""), HERE)
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        python = sys.executable if help_requested else interpreter(None, "GRASPGEN_PYTHON")
        script = HERE / ("generate_candidates.py" if command == "prepare" else "tools/download_models.py")
        os.execve(python, [python, "-B", "-u", str(script), *values], env)
    if command == "check":
        if not help_requested:
            graspgen_check(interpreter(None, "GRASPGEN_PYTHON"))
        values.insert(0, "--check")
    if command == "benchmark":
        values.insert(0, "--benchmark")
    # collect.py is a standard-library supervisor; it starts both environments explicitly.
    os.execve(sys.executable, [sys.executable, "-B", "-u", str(HERE / "collect.py"), *values], env)


if __name__ == "__main__":
    main()
