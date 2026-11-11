"""Small standard-library launcher for separate simulation, planning and model environments."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

from common import HERE, local_setting
from collect import environment, interpreter


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
    values = sys.argv[1:]
    commands = {"collect", "prepare", "check", "benchmark", "download-models", "export"}
    command = values.pop(0) if values and values[0] in commands else "collect"
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if command == "export":
        script = HERE / "tools/lerobot_export/run.py"
        os.execve(sys.executable, [sys.executable, "-B", str(script), *values], env)
    if command in {"prepare", "download-models"}:
        env = environment(os.environ.get("CUDA_VISIBLE_DEVICES", ""), HERE)
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        python = interpreter(None, "GRASPGEN_PYTHON")
        script = HERE / ("generate_candidates.py" if command == "prepare" else "tools/download_models.py")
        os.execve(python, [python, "-B", "-u", str(script), *values], env)
    if command == "check":
        graspgen_check(interpreter(None, "GRASPGEN_PYTHON"))
        values.insert(0, "--check")
    if command == "benchmark":
        values.insert(0, "--benchmark")
    # collect.py is a standard-library supervisor; it starts both environments explicitly.
    os.execve(sys.executable, [sys.executable, "-B", "-u", str(HERE / "collect.py"), *values], env)


if __name__ == "__main__":
    main()
