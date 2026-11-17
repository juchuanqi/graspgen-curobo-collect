"""Recoverable fixed-slot collection, qualified data delivery and optional ACT check."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import read, gpu_tokens, device_layout, collection_plan, OBJECTS
from collect import interpreter, Processes
from recovery import (RunLock, candidate_manifest, digest, durable_write, tree_manifest,
                      verify_candidates, verify_manifest)


def freeze_source(root, runtime, destination):
    """Run every stage against one source capture, even if the checkout changes."""
    destination.mkdir(parents=True, exist_ok=False)
    for path in [*root.glob('*.py'), root / 'local_env.json']:
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.cache')
    shutil.copytree(runtime, destination / 'runtime', ignore=ignore)
    shutil.copytree(root / 'tools', destination / 'tools', ignore=ignore)
    durable_write(destination / 'SOURCE.json', dict(source=str(root), runtime=str(runtime),
                                                   sha256=tree_manifest(destination)))
    return destination


def verify_commit(directory, run_id):
    commit = read(directory / 'COMMITTED.json')
    if commit['run_id'] != run_id:
        raise ValueError(f'Stage belongs to another registered run: {directory}')
    verify_manifest(directory, commit['sha256'], exact=True, exclude=('COMMITTED.json',))
    return commit


class Pipeline:
    def __init__(self, output, receipt, lock):
        self.output, self.receipt = output, receipt
        self.source = Path(receipt['source_snapshot'])
        self.args = receipt['arguments']
        self.processes = Processes(lock.fd)
        self.env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
            'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'USE_TF': '0',
            'USE_FLAX': '0', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}

    def save(self):
        durable_write(self.output / 'PIPELINE.json', self.receipt)

    def check_inputs(self):
        if 'source_manifest_sha256' in self.receipt:
            if digest(self.source / 'SOURCE.json') != self.receipt['source_manifest_sha256']:
                raise ValueError('Frozen source manifest changed')
            verify_manifest(self.source, read(self.source / 'SOURCE.json')['sha256'],
                            exact=True, exclude=('SOURCE.json',))
            verify_candidates(self.args['candidate_root'], self.receipt['objects'],
                              self.receipt['candidate_sha256'])
        if self.receipt.get('plan_sha256') and digest(self.args['plan_json']) != self.receipt['plan_sha256']:
            raise ValueError('Registered fixed plan changed')

    def execute(self, name, command, env=None):
        self.check_inputs()
        index = len(self.receipt['stages'])
        log = self.output / 'logs' / f'{index:03d}_{name}.log'
        log.parent.mkdir(exist_ok=True)
        stage = dict(name=name, command=list(map(str, command)), log=str(log), status='running',
                     started_unix_s=time.time())
        self.receipt['stages'].append(stage); self.save()
        print(f'{name}: {log}', flush=True)
        started = time.monotonic()
        try:
            child = self.processes.start(stage['command'], env or self.env, log)
            stage['pid'] = child.pid; self.save()
            result = child.wait()
            stage.update(returncode=result, status='passed' if result == 0 else 'failed')
            if result:
                raise RuntimeError(f'{name} failed; inspect {log}. Resume preserves existing raw slots.')
        except BaseException:
            stage.update(status='failed', error=traceback.format_exc())
            raise
        finally:
            stage.update(ended_unix_s=time.time(), wall_s=time.monotonic()-started)
            self.save()

    def collect(self):
        raw = self.output / 'raw'
        seal = raw / 'RAW_COMMITTED.json'
        if seal.exists():
            commit = read(seal)
            if commit['run_id'] != self.receipt['run_id']:
                raise ValueError('Raw records belong to another pipeline')
            verify_manifest(raw, commit['sha256'], exact=True, exclude=('.collection.lock', 'RAW_COMMITTED.json'))
            saved = self.receipt.get('raw_commit_sha256')
            if saved is not None and saved != digest(seal):
                raise ValueError('Raw commit receipt changed')
            self.receipt['collection_status'] = 'passed'; self.save()
            return
        command = [sys.executable, '-B', str(self.source / 'collect.py'), '--output', str(raw),
            '--runtime', str(self.source / 'runtime'), '--gpus', self.args['gpus'],
            '--workers', str(self.args['workers']), '--objects', self.args['objects'],
            '--episodes-per-object', str(self.args['episodes_per_object']), '--seed', str(self.args['seed']),
            '--arm-speed-profile', self.args.get('arm_speed_profile', 'baseline'),
            '--execution-profile', self.args.get('execution_profile', 'baseline'),
            '--grasp-symmetry', 'off',
            '--candidate-root', self.args['candidate_root'],
            '--initialization-retries', str(self.args['initialization_retries'])]
        if self.args.get('plan_json'):
            command.extend(['--plan-json', self.args['plan_json']])
        if self.args.get('stop_after_per_object') is not None:
            command.extend(['--stop-after-per-object', str(self.args['stop_after_per_object'])])
        if raw.exists():
            if not (raw / 'SLOTS.json').exists():
                raise ValueError('Raw directory lacks a recovery ledger; preserve and inspect it before continuing')
            command.append('--resume')
        self.execute('collection', command)
        run = read(raw / 'RUN.json')
        slots = read(raw / 'SLOTS.json')['slots']
        if run['status'] == 'checkpoint':
            limit = self.args.get('stop_after_per_object')
            allowed = {'success', 'physical_failure', 'planning_failure', 'initialization_ineligible', 'infrastructure_failed'}
            if limit is None or any(s['state'] not in allowed for s in slots.values()
                                    if s['entry']['fixed_index'] < limit):
                raise ValueError('Collection checkpoint lacks accounted prefix outcomes')
            self.receipt['collection_status'] = 'checkpoint'; self.save()
            return False
        if run['status'] != 'completed' or run['validation_requested'] is not True or any(
                s['state'] not in {'success', 'physical_failure', 'planning_failure', 'initialization_ineligible', 'infrastructure_failed'}
                for s in slots.values()):
            raise ValueError('Collection did not account for every fixed slot')
        manifest = tree_manifest(raw, exclude=('.collection.lock',))
        durable_write(seal, dict(run_id=self.receipt['run_id'], sha256=manifest,
                                fixed_slots=len(slots), slot_outcomes=run['slot_outcomes']))
        self.receipt.update(collection_status='passed', raw_commit_sha256=digest(seal)); self.save()

    def stage(self, name, destination, make_command, validate, env=None):
        final = self.output / destination
        if final.exists():
            commit = verify_commit(final, self.receipt['run_id'])
            saved = self.receipt.setdefault('commits', {}).get(name)
            if saved is not None and saved != digest(final / 'COMMITTED.json'):
                raise ValueError('Committed stage receipt changed')
        else:
            work_root = self.output / 'work'
            work_root.mkdir(exist_ok=True)
            work = work_root / f'{destination}_{len(self.receipt["stages"]):03d}'
            if work.exists():
                raise ValueError(f'Unexpected work-directory collision: {work}')
            self.execute(name, make_command(work), env)
            validate(work)
            commit = dict(run_id=self.receipt['run_id'], name=name,
                          committed_unix_s=time.time(), sha256=tree_manifest(work))
            durable_write(work / 'COMMITTED.json', commit)
            work.rename(final)
            directory = os.open(self.output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        self.receipt.setdefault('commits', {})[name] = digest(final / 'COMMITTED.json')
        self.save()
        return commit

    def check_export(self, directory):
        selection = read(directory / 'SELECTION.json')
        audit = read(directory / 'AUDIT.json')
        if (selection.get('static_and_motion_verified') is not True
                or audit.get('status') != 'passed' or audit.get('statistics', {}).get('status') != 'passed'):
            raise ValueError('Qualified delivery requires strict selection, full readback and independent statistics')
        if (directory / 'portable/SHARDS.json').exists():
            sys.path.insert(0, str(self.source / 'tools/lerobot_export'))
            from delivery import validate_delivery
            validate_delivery(directory / 'portable', directory / 'AUDIT.json')

    def check_training(self, directory):
        result = read(directory / 'RESULT.json')
        if (result.get('status') != 'passed' or result.get('steps') != self.args['train_steps']
                or result.get('audit_sha256') != digest(self.output / 'export/AUDIT.json')
                or result.get('selection_sha256') != digest(self.output / 'export/portable/SELECTION.json')):
            raise ValueError('ACT check result does not match this data delivery')

    def run(self):
        started = time.monotonic()
        for previous in self.receipt.get('invocations', []):
            if previous['status'] == 'running':
                previous.update(status='interrupted', timing_complete=False)
        for previous in self.receipt['stages']:
            if previous['status'] == 'running':
                previous.update(status='interrupted', timing_complete=False)
        if self.receipt.get('training_status') == 'running':
            self.receipt['training_status'] = 'interrupted'
        invocation = dict(started_unix_s=time.time(), data_only=self.args['data_only'], status='running',
                          stop_after_per_object=self.args.get('stop_after_per_object'))
        self.receipt.setdefault('invocations', []).append(invocation)
        self.receipt.update(status='running', error=None); self.save()
        old_handler = signal.getsignal(signal.SIGTERM)
        def interrupted(*unused):
            raise KeyboardInterrupt('Pipeline supervisor received SIGTERM')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            self.check_inputs()
            if self.collect() is False:
                self.receipt['status'] = invocation['status'] = 'checkpoint'
                return
            python = self.args['lerobot_python']
            self.stage('export_and_readback', 'export', lambda work: [sys.executable, '-B',
                str(self.source / 'tools/lerobot_export/run.py'), '--run', str(self.output / 'raw'),
                '--runtime', str(self.source / 'runtime'), '--output', str(work), '--python', python,
                '--workers', str(self.args.get('export_workers', 1))],
                self.check_export)
            self.receipt['data_status'] = 'passed'; self.save()
            if not self.args['data_only']:
                self.receipt['training_status'] = 'running'; self.save()
                self.stage('training', 'training', lambda work: [python, '-B',
                    str(self.source / 'tools/lerobot_export/train_check.py'), '--portable',
                    str(self.output / 'export/portable'), '--audit', str(self.output / 'export/AUDIT.json'),
                    '--output', str(work), '--steps', str(self.args['train_steps'])], self.check_training,
                    {**self.env, 'CUDA_VISIBLE_DEVICES': gpu_tokens(self.args['gpus'])[0]})
                self.receipt['training_status'] = 'passed'
            self.check_inputs()
            verify_commit(self.output / 'export', self.receipt['run_id'])
            raw_commit = read(self.output / 'raw/RAW_COMMITTED.json')
            verify_manifest(self.output / 'raw', raw_commit['sha256'], exact=True,
                            exclude=('.collection.lock', 'RAW_COMMITTED.json'))
            self.receipt['status'] = 'passed'
            invocation['status'] = 'passed'
        except BaseException:
            self.receipt.update(status='failed', error=traceback.format_exc())
            if self.receipt.get('training_status') == 'running':
                self.receipt['training_status'] = 'failed'
            invocation['status'] = 'failed'
            raise
        finally:
            self.processes.close()
            signal.signal(signal.SIGTERM, old_handler)
            invocation.update(ended_unix_s=time.time(), active_wall_s=time.monotonic()-started)
            self.receipt['active_wall_s'] = sum(i['active_wall_s'] for i in self.receipt['invocations']
                                               if 'active_wall_s' in i)
            self.receipt['elapsed_since_registration_s'] = time.time()-self.receipt['created_unix_s']
            self.receipt['timing_complete'] = all(i.get('timing_complete', True) for i in self.receipt['invocations'])
            self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpus')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--export-workers', type=int,
                        help='1: single package (default); >1: parallel audited packages per object')
    parser.add_argument('--execution-profile', choices=('baseline', 'accelerated'),
                        help='New runs default to the existing accelerated sensor/contact/camera/actuator runtime')
    parser.add_argument('--arm-speed-profile', choices=('baseline', 'recommended'),
                        help='New runs default to recommended (approach 1.0 / carry 0.3 rad/s); resume keeps the registered profile')
    parser.add_argument('--objects')
    parser.add_argument('--episodes-per-object', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--plan-json', type=Path)
    parser.add_argument('--stop-after-per-object', type=int)
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--lerobot-python')
    parser.add_argument('--train-steps', type=int)
    parser.add_argument('--initialization-retries', type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--data-only', dest='data_only', action='store_true', default=None)
    mode.add_argument('--with-training', dest='data_only', action='store_false')
    args = parser.parse_args()
    if args.stop_after_per_object is not None and args.stop_after_per_object < 1:
        parser.error('--stop-after-per-object must be positive')
    if args.export_workers is not None and args.export_workers < 1:
        parser.error('--export-workers must be positive')
    output = args.output.resolve()
    if args.resume:
        receipt = read(output / 'PIPELINE.json')
        if receipt.get('protocol') != 1:
            parser.error('This run predates general recovery; its original evidence must remain sealed')
        source = Path(receipt['source_snapshot'])
        if digest(source / 'SOURCE.json') != receipt['source_manifest_sha256']:
            raise ValueError('Frozen source manifest changed')
        verify_manifest(source, read(source / 'SOURCE.json')['sha256'], exact=True, exclude=('SOURCE.json',))
        if source.resolve() != ROOT.resolve():
            os.execve(sys.executable, [sys.executable, '-B', str(source / 'tools/pipeline.py'), *sys.argv[1:]], os.environ)
        values = dict(receipt['arguments'])
        for key, value in vars(args).items():
            if key in {'output', 'resume', 'data_only', 'stop_after_per_object'} or value is None:
                continue
            value = str(value.resolve()) if isinstance(value, Path) else value
            registered = values.get(key, 1 if key == 'export_workers' else
                                    'baseline' if key in {'arm_speed_profile', 'execution_profile'} else None)
            if registered != value:
                parser.error(f'--{key.replace("_", "-")} differs from the registered run')
        if args.data_only is not None:
            values['data_only'] = args.data_only
        values['stop_after_per_object'] = args.stop_after_per_object
        receipt['arguments'] = values
    else:
        values = dict(gpus='0', workers=1, export_workers=1, arm_speed_profile='recommended', execution_profile='accelerated',
            objects='all', episodes_per_object=5, seed=2026111300,
            candidate_root=str(ROOT / 'generated/candidates'), runtime=str(ROOT / 'runtime'),
            lerobot_python=None, train_steps=100, initialization_retries=2, data_only=False,
            plan_json=None, stop_after_per_object=None)
        for key, value in vars(args).items():
            if key not in {'output', 'resume'} and value is not None:
                values[key] = str(value.resolve()) if isinstance(value, Path) else value
        objects = list(OBJECTS) if values['objects'] == 'all' else values['objects'].split(',')
        collection_plan(objects, values['episodes_per_object'], values['workers'], values['seed'])
        if values['plan_json']:
            from coverage_plan import load_fixed_plan
            load_fixed_plan(values['plan_json'], objects, values['episodes_per_object'], values['workers'], values['seed'])
        tokens = gpu_tokens(values['gpus'])
        if values['workers'] > 1 and len(values['gpus'].split(',')) not in (4, 8):
            parser.error('Multiple workers require four or eight GPUs')
        try:
            device_layout(tokens, values['workers'])
        except ValueError as error:
            parser.error(str(error))
        if values['train_steps'] < 1 or not 0 <= values['initialization_retries'] <= 2:
            parser.error('train steps must be positive; initialization retries must be 0–2')
        runtime = Path(values['runtime']).resolve()
        if output == ROOT or output.is_relative_to(runtime) or output.is_relative_to(ROOT / 'tools'):
            parser.error('--output must be outside runtime and tools')
        values['lerobot_python'] = interpreter(values['lerobot_python'], 'LEROBOT_PYTHON')
        candidates = candidate_manifest(values['candidate_root'], objects)
        output.mkdir(parents=True, exist_ok=False)
        with RunLock(output / '.pipeline.lock') as lock:
            if values['plan_json']:
                shutil.copy2(values['plan_json'], output / 'FIXED_PLAN.json')
                values['plan_json'] = str(output / 'FIXED_PLAN.json')
            source = freeze_source(ROOT, runtime, output / 'source')
            receipt = dict(protocol=1, run_id=uuid.uuid4().hex, created_unix_s=time.time(),
                arguments=values, source_snapshot=str(source),
                source_manifest_sha256=digest(source / 'SOURCE.json'), candidate_sha256=candidates,
                objects=objects, stages=[], status='registered', data_status='not_started',
                training_status='not_requested' if values['data_only'] else 'not_started',
                physical_grasp_retries=0, fixed_denominator=True)
            if values['plan_json']:
                receipt['plan_sha256'] = digest(values['plan_json'])
            durable_write(output / 'PIPELINE.json', receipt)
            Pipeline(output, receipt, lock).run()
        return
    with RunLock(output / '.pipeline.lock') as lock:
        current = read(output / 'PIPELINE.json')
        if current['run_id'] != receipt['run_id']:
            raise ValueError('Pipeline identity changed while acquiring ownership')
        current['arguments']['data_only'] = receipt['arguments']['data_only']
        current['arguments']['stop_after_per_object'] = args.stop_after_per_object
        verify_candidates(current['arguments']['candidate_root'], current['objects'], current['candidate_sha256'])
        Pipeline(output, current, lock).run()


if __name__ == '__main__':
    main()
