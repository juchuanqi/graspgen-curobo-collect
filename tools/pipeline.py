"""Collect fixed slots, strictly qualify, export, read back and train locally."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import write, sha, gpu_tokens
from collect import interpreter


def freeze_source(root, runtime, destination):
    """Run every stage against one source capture, even if the checkout changes."""
    destination.mkdir(parents=True, exist_ok=False)
    for path in [*root.glob('*.py'), root / 'local_env.json']:
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.cache')
    shutil.copytree(runtime, destination / 'runtime', ignore=ignore)
    shutil.copytree(root / 'tools', destination / 'tools', ignore=ignore)
    files = {str(p.relative_to(destination)): sha(p)
             for p in sorted(destination.rglob('*')) if p.is_file()}
    write(destination / 'SOURCE.json', dict(source=str(root), runtime=str(runtime), sha256=files))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--objects', default='all')
    parser.add_argument('--episodes-per-object', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026111300)
    parser.add_argument('--candidate-root', type=Path, default=ROOT / 'generated/candidates')
    parser.add_argument('--runtime', type=Path, default=ROOT / 'runtime')
    parser.add_argument('--lerobot-python')
    parser.add_argument('--train-steps', type=int, default=100)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or any(output.is_relative_to(p) for p in (args.runtime.resolve(), ROOT / 'tools')):
        parser.error('--output must be outside the runtime and tools directories')
    python = interpreter(args.lerobot_python, 'LEROBOT_PYTHON')
    if args.train_steps < 1: parser.error('--train-steps must be positive')
    output.mkdir(parents=True, exist_ok=False)
    source = freeze_source(ROOT, args.runtime.resolve(), output / 'source')
    started = time.monotonic()
    receipt = dict(status='running', arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   stages=[], retries=False, fixed_denominator=True,
                   source_snapshot=str(source), source_manifest_sha256=sha(source / 'SOURCE.json'))
    write(output / 'PIPELINE.json', receipt)
    offline = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'USE_TF': '0', 'USE_FLAX': '0', 'OMP_NUM_THREADS': '2'}
    commands = [
        ('collection', [sys.executable, '-B', str(source / 'collect.py'), '--output', str(output / 'raw'),
         '--runtime', str(source / 'runtime'), '--gpus', args.gpus, '--workers', str(args.workers),
         '--objects', args.objects, '--episodes-per-object', str(args.episodes_per_object),
         '--seed', str(args.seed), '--arm-speed-profile', 'baseline', '--grasp-symmetry', 'off',
         '--candidate-root', str(args.candidate_root.resolve())], offline),
        ('export_and_readback', [sys.executable, '-B', str(source / 'tools/lerobot_export/run.py'),
         '--run', str(output / 'raw'), '--runtime', str(source / 'runtime'),
         '--output', str(output / 'export'), '--python', python], offline),
        ('training', [python, '-B', str(source / 'tools/lerobot_export/train_check.py'),
         '--portable', str(output / 'export/portable'), '--audit', str(output / 'export/AUDIT.json'),
         '--output', str(output / 'training'), '--steps', str(args.train_steps)],
         {**offline, 'CUDA_VISIBLE_DEVICES': gpu_tokens(args.gpus)[0]})]
    try:
        for name, command, env in commands:
            stage = dict(name=name, command=command, status='running', started_unix_s=time.time())
            receipt['stages'].append(stage); write(output / 'PIPELINE.json', receipt)
            print(f'{name}: {output / (name + ".log")}', flush=True)
            with (output / (name + '.log')).open('x') as stream:
                result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
            stage.update(returncode=result.returncode, ended_unix_s=time.time(), status='passed' if result.returncode == 0 else 'failed')
            write(output / 'PIPELINE.json', receipt)
            if result.returncode: raise RuntimeError(f'{name} failed; inspect retained log and fixed slots')
        receipt['status'] = 'passed'
    except BaseException:
        receipt.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        receipt['wall_s'] = time.monotonic() - started
        write(output / 'PIPELINE.json', receipt)


if __name__ == '__main__': main()
