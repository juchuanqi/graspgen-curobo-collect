"""Fixed-slot collection with durable, conservative recovery (no grasp retries)."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from common import read, device_layout
from recovery import (RunLock, candidate_manifest, digest, durable_write, tree_manifest,
                      verify_candidates, verify_manifest, initialization_resource_error)

TERMINAL = {'success', 'physical_failure', 'planning_failure', 'initialization_ineligible',
            'infrastructure_failed', 'invalid_data', 'interrupted_unknown'}
VALIDATORS = [('validate_episode.py', 'VALIDATION.json'),
              ('validate_raw_forces.py', 'RAW_FORCE_VALIDATION.json')]


def source_manifest(root, runtime):
    files = {str(p): digest(p) for p in Path(root).glob('*.py')}
    if (Path(root) / 'local_env.json').exists():
        files[str(Path(root) / 'local_env.json')] = digest(Path(root) / 'local_env.json')
    for folder in [Path(root) / 'tools', Path(runtime)]:
        files.update({str(folder / name): value for name, value in tree_manifest(folder).items()})
    return files


class Collector:
    def __init__(self, plan, args, tokens, checks, sim_python, planner_python, lock):
        from collect import HERE, Processes
        self.root, self.plan, self.args, self.tokens = HERE, plan, args, tokens
        self.sim_python, self.planner_python = sim_python, planner_python
        self.processes = Processes(lock.fd)
        self.path = args.output / 'SLOTS.json'
        self.stage = plan['stages'][0]
        self.out = args.output / self.stage['name']
        self.layout = device_layout(tokens, self.stage['lanes'])
        self.accelerated = plan.get('execution_profile', 'baseline') == 'accelerated'
        self.validation_pool = None
        self.validation_futures = {}
        configuration = dict(runtime=str(args.runtime), candidate_root=str(args.candidate_root),
            sim_python=sim_python, planner_python=planner_python,
            initialization_retries=args.initialization_retries,
            skip_validation=args.skip_validation, gui=args.gui,
            startup_timeout=args.startup_timeout, episode_timeout=args.episode_timeout,
            device_layout=self.layout)
        if args.resume:
            self.ledger = read(self.path)
            if self.ledger.get('protocol') != 1 or read(args.output / 'PLAN.json') != plan:
                raise ValueError('Resume requires the original fixed plan and recovery protocol')
            if self.ledger['configuration'] != configuration:
                raise ValueError('Resume configuration differs from the registered run')
            if source_manifest(HERE, args.runtime) != self.ledger['source_sha256']:
                raise ValueError('Collection source/configuration/assets changed')
            verify_candidates(args.candidate_root, plan['objects'], self.ledger['candidate_sha256'])
            for slot in self.ledger['slots'].values():
                if slot.get('completed_sha256'):
                    verify_manifest(slot['output'], slot['completed_sha256'])
        else:
            self.ledger = dict(protocol=1, configuration=configuration,
                source_sha256=source_manifest(HERE, args.runtime),
                candidate_sha256=candidate_manifest(args.candidate_root, plan['objects']),
                created_unix_s=time.time(), invocations=[], attempts=[], planner_attempts=[],
                slots={c['case_id']: dict(entry=c, state='pending', initialization_failures=0,
                                         launches=[]) for c in self.stage['cases']})
            self.out.mkdir(parents=True)
            (self.out / 'workers').mkdir()
            durable_write(args.output / 'PLAN.json', plan)
            durable_write(args.output / 'ENVIRONMENT.json', dict(checks=checks, gpu_tokens=tokens,
                arm_speed_profile=plan['arm_speed_profile'], grasp_symmetry=plan['grasp_symmetry'],
                inherited_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
            self.save()

    def save(self):
        self.ledger['counts'] = dict(Counter(s['state'] for s in self.ledger['slots'].values()))
        durable_write(self.path, self.ledger)

    def set_slot(self, slot, state, **values):
        slot.update(state=state, updated_unix_s=time.time(), **values)
        self.save()

    def prepare_saved(self, slot, directory):
        if slot['state'] in TERMINAL:
            return False
        metadata = read(directory / 'METADATA.json')
        if read(directory / 'ENTRY.json') != slot['entry'] or any(
                metadata.get(k) != slot['entry'][k] for k in ('case_id', 'object_id', 'scene_seed')):
            self.set_slot(slot, 'invalid_data', reason='saved_identity_mismatch')
            return False
        if metadata.get('error') or metadata.get('reason') == 'technical_exception':
            self.set_slot(slot, 'interrupted_unknown', reason='technical_episode_failure')
            return False
        if 'raw_sha256' not in slot:
            self.set_slot(slot, 'validating', output=str(directory), raw_sha256=tree_manifest(directory))
        verify_manifest(directory, slot['raw_sha256'])
        return True

    def validate_record(self, slot, directory):
        """Only per-record files change here; the supervisor owns all ledger writes."""
        metadata = read(directory / 'METADATA.json')
        completed = dict(slot.get('validators', {}))
        try:
            if not self.args.skip_validation:
                for script, name in VALIDATORS:
                    if name in completed:
                        verify_manifest(directory, {name: completed[name]})
                        continue
                    folder = directory / 'validation_attempts'
                    folder.mkdir(exist_ok=True)
                    index = len(list(folder.glob(script + '.*.log')))
                    result_path = folder / f'{script}.{index}.json'
                    command = [self.planner_python, '-B', str(self.args.runtime / 'validators' / script),
                               str(directory), '--output', str(result_path)]
                    child = self.processes.start(command, {**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                                 folder / f'{script}.{index}.log')
                    try:
                        returncode = child.wait(timeout=180)
                    finally:
                        self.processes.stop(child)
                    if returncode or read(result_path).get('status') != 'passed':
                        raise ValueError(f'Saved record validation failed: {result_path}')
                    durable_write(directory / name, read(result_path))
                    completed[name] = digest(directory / name)
            state = ('initialization_ineligible' if metadata['reason'] in {
                'initial_object_not_static', 'invalid_initial_geometry', 'requested_initial_state_mismatch'}
                else 'success' if metadata['success'] else 'planning_failure'
                if metadata.get('physical_attempts') == 0 else 'physical_failure')
            return dict(state=state, reason=metadata['reason'], validators=completed,
                        physical_attempts=metadata.get('physical_attempts'),
                        completed_sha256=tree_manifest(directory),
                        validation_completed_ns=time.monotonic_ns())
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return dict(state='invalid_data', reason='saved_validation_failed', error=traceback.format_exc())

    def apply_validation(self, slot, result):
        result = dict(result)
        self.set_slot(slot, result.pop('state'), **result)

    def validate_saved(self, slot, directory):
        if self.prepare_saved(slot, directory):
            self.apply_validation(slot, self.validate_record(copy.deepcopy(slot), directory))

    def harvest_validation(self, wait=False):
        for case, (future, slot) in list(self.validation_futures.items()):
            if wait or future.done():
                self.apply_validation(slot, future.result())
                del self.validation_futures[case]

    def accept_saved(self, slot, directory, handoffs):
        if slot['state'] in TERMINAL or slot['entry']['case_id'] in self.validation_futures:
            return
        # Two active validators plus at most four waiting records.
        if len(self.validation_futures) >= 6 or not self.prepare_saved(slot, directory):
            return
        case = slot['entry']['case_id']
        self.validation_futures[case] = (
            self.validation_pool.submit(self.validate_record, copy.deepcopy(slot), directory), slot)
        durable_write(handoffs / f'{case}.json', dict(status='accepted', accepted_ns=time.monotonic_ns(),
                      validation_complete=False))

    def reconcile(self, attempt):
        if attempt.get('reconciled'):
            return
        worker = Path(attempt['output'])
        state_path = worker / 'WORKER_STATE.json'
        state = read(state_path) if state_path.exists() else {}
        recoverable = (state.get('protocol') == 1 and state.get('state') == 'failed'
            and state.get('failed_phase') == 'initializing'
            and state.get('recoverable_initialization') is True
            and initialization_resource_error(state.get('error', ''))
            and not (worker / 'READY.json').exists()
            and not any((worker / case).exists() for case in attempt['cases']))
        attempt['worker_state'] = state
        for case in attempt['cases']:
            slot = self.ledger['slots'][case]
            if slot['state'] in TERMINAL:
                continue
            directory = worker / case
            marker = directory / 'SLOT_STATE.json'
            if marker.exists() and read(marker).get('state') == 'saved':
                self.validate_saved(slot, directory)
            elif directory.exists():
                self.set_slot(slot, 'interrupted_unknown', output=str(directory),
                              reason='slot_entered_without_complete_saved_record')
            elif recoverable:
                slot['initialization_failures'] += 1
                retry = slot['initialization_failures'] <= self.args.initialization_retries
                self.set_slot(slot, 'pending' if retry else 'infrastructure_failed',
                              reason='initialization_resource_failure', last_worker=str(worker))
            elif state.get('protocol') == 1 and (state.get('state') in {'ready', 'running', 'completed'}
                    or state.get('state') == 'failed' and state.get('failed_phase') in {'ready', 'running'}):
                # This protocol persists the slot directory before any slot work.
                self.set_slot(slot, 'pending', reason='not_entered_before_worker_exit')
            elif state.get('protocol') == 1 and (
                    state.get('state') == 'initializing' or state.get('failed_phase') == 'initializing'):
                self.set_slot(slot, 'infrastructure_failed', reason='nonretryable_initialization_failure',
                              last_worker=str(worker))
            else:
                self.set_slot(slot, 'interrupted_unknown', reason='worker_dispatch_or_progress_unknown',
                              last_worker=str(worker))
        attempt['reconciled'] = True
        self.save()

    def start_planners(self):
        from collect import environment
        # Launch all dedicated services together, before any physical slot.
        archive = self.ledger.setdefault('planner_attempts_archive', [])
        for planner_id in range(len(self.layout['planners'])):
            previous = [a for a in self.ledger['planner_attempts'] if a.get('planner_id', 0) == planner_id]
            if any(a['state'] == 'ready' for a in previous):
                archive.extend(previous)
                self.ledger['planner_attempts'] = [a for a in self.ledger['planner_attempts']
                                                   if a.get('planner_id', 0) != planner_id]

        def launch(planner_id):
            attempts = [a for a in self.ledger['planner_attempts'] if a.get('planner_id', 0) == planner_id]
            if len(attempts) > self.args.initialization_retries:
                raise RuntimeError(f'Planner {planner_id} initialization retries exhausted')
            if attempts and attempts[-1].get('retryable') is False:
                raise RuntimeError(f'Planner {planner_id} has a retained nonretryable initialization failure')
            index = len(archive) + len(self.ledger['planner_attempts'])
            address = self.out / f'planner_address_{index:03d}.json'
            log = self.out / f'planner_{index:03d}.log'
            gpu = self.layout['planners'][planner_id]
            row = dict(index=index, planner_id=planner_id, gpu=gpu, state='starting',
                       started_unix_s=time.time(), log=str(log), address=str(address))
            self.ledger['planner_attempts'].append(row); self.save()
            child = self.processes.start([self.planner_python, '-B', '-u',
                str(self.root / 'planner_service.py'), '--runtime', str(self.args.runtime),
                '--ready', str(address)], environment(gpu, self.args.output), log)
            row['pid'] = child.pid; self.save()
            return child, address, log, row, time.monotonic() + self.args.startup_timeout

        pending = {i: launch(i) for i in range(len(self.layout['planners']))}
        ready = {}
        while pending:
            for planner_id, (child, address, log, row, deadline) in list(pending.items()):
                if child.poll() is not None or time.monotonic() >= deadline:
                    self.processes.stop(child)
                    row.update(state='failed', returncode=child.returncode, ended_unix_s=time.time(),
                               retryable=initialization_resource_error(log.read_text(errors='replace')))
                    self.save()
                    if not row['retryable']:
                        raise RuntimeError(f'Planner {planner_id} initialization failed: {log}')
                    pending[planner_id] = launch(planner_id)
                elif address.exists():
                    row.update(state='ready', ready_unix_s=time.time()); self.save()
                    ready[planner_id] = (child, address)
                    del pending[planner_id]
            if pending:
                time.sleep(.05)
        return ready

    def dispatch(self, cases, lane, address, wave=None):
        from collect import environment
        index = len(self.ledger['attempts'])
        worker_id = f"{cases[0]['object_id']}_attempt{index:04d}_lane{lane}"
        job = dict(stage=self.stage['name'], worker_id=worker_id, persistent=True, cases=cases,
                   candidate_root=str(self.args.candidate_root), grasp_symmetry=self.plan['grasp_symmetry'],
                   execution_profile=self.plan.get('execution_profile', 'baseline'))
        if self.accelerated:
            job.update(validation_handoffs=str(self.out / 'validation_handoffs'),
                       validation_abort=str(self.abort_path),
                       measurement_barrier=wave['measurement_barrier'], handoff_timeout_s=600)
        job_path = self.out / 'jobs' / f'{worker_id}.json'
        durable_write(job_path, job)
        directory = self.out / 'workers' / worker_id
        row = dict(worker_id=worker_id, cases=[c['case_id'] for c in cases], output=str(directory),
                   simulation_gpu=self.layout['simulation'][lane],
                   planner_id=self.layout['lane_to_planner'][lane], planner_address=str(address),
                   started_unix_s=time.time(), state='dispatching')
        self.ledger['attempts'].append(row)
        for case in row['cases']:
            slot = self.ledger['slots'][case]
            slot['launches'].append(index)
            slot['state'] = 'initializing'
        self.save()  # write-ahead: even a death during Popen cannot cause a duplicate dispatch
        command = [self.sim_python, '-B', '-u', str(self.root / 'sim_worker.py'),
            '--runtime', str(self.args.runtime), '--job', str(job_path), '--address', str(address),
            '--output', str(directory), '--enable_cameras', '--device', 'cuda:0']
        if not self.args.gui:
            command.append('--headless')
        child = self.processes.start(command, environment(self.layout['simulation'][lane], self.args.output),
                                     self.out / f'{worker_id}.log')
        row.update(pid=child.pid, command=command, state='running'); self.save()
        return child, row, time.monotonic() + self.args.startup_timeout + len(cases) * self.args.episode_timeout

    def run(self):
        for previous in self.ledger['invocations']:
            if previous['status'] == 'running':
                previous.update(status='interrupted', timing_complete=False)
        prefix_limit = getattr(self.args, 'stop_after_per_object', None)
        invocation = dict(started_unix_s=time.time(), resumed=self.args.resume, status='running',
                          stop_after_per_object=prefix_limit)
        self.ledger['invocations'].append(invocation); self.save()
        self.abort_path = self.out / f"invocation_{len(self.ledger['invocations']):03d}_ABORT.json"
        if self.accelerated:
            self.validation_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='record-validation')
        started = time.monotonic()
        status, error = 'completed', None
        old_handler = signal.getsignal(signal.SIGTERM)
        def interrupted(*unused):
            raise KeyboardInterrupt('Collection supervisor received SIGTERM')
        signal.signal(signal.SIGTERM, interrupted)
        try:
            for attempt in self.ledger['attempts']:
                self.reconcile(attempt)
            if any(s['state'] in {'interrupted_unknown', 'invalid_data'} for s in self.ledger['slots'].values()):
                raise RuntimeError('Execution progress is unknown; no automatic re-execution is allowed. See SLOTS.json')
            eligible = lambda s: prefix_limit is None or s['entry']['fixed_index'] < prefix_limit
            pending = lambda: [s['entry'] for s in self.ledger['slots'].values()
                               if s['state'] == 'pending' and eligible(s)]
            if pending():
                services = self.start_planners()
                next_update = 0.
                for name in self.plan['objects']:
                    while (group := [c for c in pending() if c['object_id'] == name]):
                        verify_candidates(self.args.candidate_root, self.plan['objects'], self.ledger['candidate_sha256'])
                        jobs = [(lane, group[lane::self.stage['lanes']])
                                for lane in range(min(self.stage['lanes'], len(group)))]
                        waves = self.ledger.setdefault('waves', [])
                        wave = dict(object_id=name, started_unix_s=time.time(),
                            warmup_cases=[cases[0]['case_id'] for _, cases in jobs],
                            measurement_cases=[c['case_id'] for _, cases in jobs for c in cases[1:]],
                            measurement_barrier=str(self.out / 'barriers' / f'wave_{len(waves):03d}.json'))
                        waves.append(wave); self.save()
                        active = [self.dispatch(cases, lane,
                                  services[self.layout['lane_to_planner'][lane]][1], wave)
                                  for lane, cases in jobs]
                        while any(child.poll() is None for child, _, _ in active):
                            self.harvest_validation()
                            if any(service.poll() is not None for service, _ in services.values()):
                                raise RuntimeError('Planner exited while slots were active')
                            for child, row, limit in active:
                                if child.poll() is None and time.monotonic() > limit:
                                    row['timeout'] = True
                                    self.processes.stop(child)
                                for case in row['cases']:
                                    marker = Path(row['output']) / case / 'SLOT_STATE.json'
                                    if marker.exists():
                                        state = read(marker)['state']
                                        slot = self.ledger['slots'][case]
                                        if state in {'initializing', 'planning', 'executing'} and slot['state'] != state:
                                            self.set_slot(slot, state)
                                        elif self.accelerated and state == 'saved':
                                            self.accept_saved(slot, marker.parent,
                                                              self.out / 'validation_handoffs')
                                if (self.accelerated and child.poll() is not None and
                                        self.ledger['slots'][row['cases'][0]]['state'] not in TERMINAL and
                                        not (Path(row['output']) / row['cases'][0] / 'SLOT_STATE.json').exists()):
                                    raise RuntimeError('Worker exited before warmup; retained startup evidence')
                            if any(s['state'] in {'invalid_data', 'interrupted_unknown'}
                                   for s in self.ledger['slots'].values()):
                                raise RuntimeError('Saved record validation failed; stopping new physical slots')
                            if (self.accelerated and wave['measurement_cases'] and
                                    'measurement_started_ns' not in wave and all(
                                    self.ledger['slots'][case]['state'] in TERMINAL
                                    for case in wave['warmup_cases'])):
                                wave['measurement_started_ns'] = time.monotonic_ns()
                                durable_write(wave['measurement_barrier'], dict(
                                    status='open', measurement_started_ns=wave['measurement_started_ns'],
                                    warmup_validated=True))
                                self.save()
                            if time.monotonic() >= next_update:
                                print(f"[collection] {self.ledger['counts']}; elapsed {time.monotonic()-started:.1f}s", flush=True)
                                next_update = time.monotonic() + 30
                            time.sleep(.2)
                        self.harvest_validation(wait=True)
                        for child, row, _ in active:
                            row.update(returncode=child.returncode, ended_unix_s=time.time(), state='exited')
                            self.save(); self.reconcile(row)
                        wave['ended_unix_s'] = time.time()
                        if self.accelerated and wave['measurement_cases']:
                            markers = [read(Path(row['output']) / case / 'SLOT_STATE.json')
                                       for _, row, _ in active for case in row['cases'][1:]
                                       if (Path(row['output']) / case / 'SLOT_STATE.json').exists()]
                            wave['last_measurement_saved_ns'] = max(
                                (m.get('saved_monotonic_ns', 0) for m in markers), default=0)
                            wave['last_measurement_validated_ns'] = max(
                                (self.ledger['slots'][c].get('validation_completed_ns', 0)
                                 for c in wave['measurement_cases']), default=0)
                        self.save()
                        if any(s['state'] in {'interrupted_unknown', 'invalid_data'} for s in self.ledger['slots'].values()):
                            raise RuntimeError('Incomplete execution or invalid saved data; retained without grasp retry')
            verify_manifest(Path('.'), self.ledger['source_sha256'])
            verify_candidates(self.args.candidate_root, self.plan['objects'], self.ledger['candidate_sha256'])
            if any(s['state'] not in TERMINAL or s['state'] in {'interrupted_unknown', 'invalid_data'}
                   for s in self.ledger['slots'].values() if eligible(s)):
                raise RuntimeError('Not all fixed slots have valid terminal outcomes')
            if any(s['state'] == 'pending' for s in self.ledger['slots'].values()):
                status = 'checkpoint'
        except BaseException:
            status, error = 'interrupted_or_technical_failure', traceback.format_exc()
            if self.accelerated:
                durable_write(self.abort_path, dict(status='failed', error=error))
        finally:
            if self.validation_pool is not None:
                # Wake waiting producers on failure, then let saved-record checks finish.
                try:
                    self.harvest_validation(wait=True)
                except BaseException:
                    status, error = 'interrupted_or_technical_failure', traceback.format_exc()
                self.validation_pool.shutdown(wait=True)
            self.processes.close()
            # Reconcile after children exit; interrupts never turn into automatic grasp retries.
            try:
                for attempt in self.ledger['attempts']:
                    self.reconcile(attempt)
            except BaseException:
                status, error = 'interrupted_or_technical_failure', traceback.format_exc()
            signal.signal(signal.SIGTERM, old_handler)
            invocation.update(status=status, error=error, ended_unix_s=time.time(),
                              active_wall_s=time.monotonic()-started)
            self.save()
            total_active = sum(i.get('active_wall_s', 0.) for i in self.ledger['invocations'])
            receipt = dict(status=status, error=error, validation_requested=not self.args.skip_validation,
                collection_wall_s=total_active, total_wall_s=total_active,
                timing_complete=all(i.get('timing_complete', True) for i in self.ledger['invocations']),
                time_semantics='sum of completed supervisor invocations, including validation/teardown; lower bound if an invocation was killed; excludes downtime',
                elapsed_since_registration_s=time.time()-self.ledger['created_unix_s'],
                fixed_slots=len(self.ledger['slots']), slot_outcomes=self.ledger['counts'],
                physical_grasp_retries=0, slot_ledger=str(self.path))
            durable_write(self.args.output / 'RUN.json', receipt)
            durable_write(self.out / 'STAGE.json', dict(status=status, jobs=self.ledger['attempts'],
                planned_slots=len(self.ledger['slots']), lanes=self.stage['lanes'],
                wall_s=total_active, time_semantics=receipt['time_semantics']))
            if status == 'completed' and not self.args.skip_validation:
                durable_write(self.args.output / 'VALIDATION_SUMMARY.json', dict(status='passed',
                    saved_slots=sum(bool(s.get('completed_sha256')) for s in self.ledger['slots'].values()),
                    slot_outcomes=self.ledger['counts']))
        if status not in {'completed', 'checkpoint'}:
            raise RuntimeError(error)
        return receipt


def run_collection(plan, args, tokens, checks, sim_python, planner_python):
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with RunLock(args.output / '.collection.lock') as lock:
        collector = Collector(plan, args, tokens, checks, sim_python, planner_python, lock)
        return collector.run()
