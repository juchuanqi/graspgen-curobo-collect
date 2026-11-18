"""Fault injection through actual subprocesses; these fixtures do not simulate grasps."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from common import collection_plan, read
from recovery import RunLock, digest, durable_write, tree_manifest, verify_manifest
from resumable_collect import run_collection
from recoverable_pipeline import Pipeline, verify_commit, main as pipeline_main


WORKER = '''
import argparse, os, sys, time, subprocess
from pathlib import Path
from recovery import durable_write as write
from common import read
from runtime.collector.obstacle_scene import planner_obstacles
p=argparse.ArgumentParser();p.add_argument('--job');p.add_argument('--output');p.add_argument('--address');a,_=p.parse_known_args()
job=read(a.job);out=Path(a.output);out.mkdir(parents=True)
write(out/'ROUTING.json',dict(simulation_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
    planner_gpu=read(a.address).get('gpu'),planner_address=a.address))
control=Path(os.environ['RECOVERY_TEST_CONTROL']);mode=control.read_text()
events=Path(os.environ['RECOVERY_TEST_EVENTS'])
def event(value):
    with events.open('a') as s: s.write(value+'\\n');s.flush();os.fsync(s.fileno())
event('startup')
write(out/'WORKER_STATE.json',dict(protocol=1,state='initializing'))
if mode.startswith('oom') or mode=='import_error':
    if mode=='oom_once': control.write_text('success')
    write(out/'WORKER_STATE.json',dict(protocol=1,state='failed',failed_phase='initializing',
        recoverable_initialization=mode!='import_error',error='CUDA out of memory' if mode!='import_error' else 'ImportError'))
    raise SystemExit(17)
write(out/'READY.json',{})
write(out/'WORKER_STATE.json',dict(protocol=1,state='ready'))
for i,c in enumerate(job['cases']):
    d=out/c['case_id'];d.mkdir()
    write(d/'SLOT_STATE.json',dict(state='executing'))
    write(out/'WORKER_STATE.json',dict(protocol=1,state='running',case_id=c['case_id']))
    event('physical:'+c['case_id'])
    if mode=='execution_exit': os._exit(23)
    write(d/'ENTRY.json',c)
    obstacle='none' if mode=='obstacle_downgrade' else job.get('obstacle','none')
    write(d/'SCENE.json',dict(obstacle_mode=obstacle,obstacles=planner_obstacles(obstacle)))
    failed=mode=='physical_failure'
    write(d/'METADATA.json',dict(case_id=c['case_id'],object_id=c['object_id'],scene_seed=c['scene_seed'],
        reason='stable_hold_not_reached' if failed else 'stable_grasp_hold',success=not failed,
        physical_attempts=1,error=None))
    (d/'payload').write_bytes(b'unchanged recorded evidence')
    write(d/'SLOT_STATE.json',dict(state='saved',saved_monotonic_ns=time.monotonic_ns()))
    write(out/'WORKER_STATE.json',dict(protocol=1,state='ready'))
    if job.get('validation_handoffs'):
        accepted=Path(job['validation_handoffs'])/(c['case_id']+'.json')
        waiting=[accepted]
        if i==0 and len(job['cases'])>1: waiting.append(Path(job['measurement_barrier']))
        for marker in waiting:
            deadline=time.monotonic()+10
            while not marker.exists():
                if Path(job['validation_abort']).exists():
                    write(out/'WORKER_STATE.json',dict(protocol=1,state='failed',failed_phase='ready'))
                    raise RuntimeError('Validation aborted before next slot')
                if time.monotonic()>deadline: raise TimeoutError('Fixture handoff timed out')
                time.sleep(.01)
    if mode=='pause_after_first' and i==0:
        # A descendant inherits the run lock. Parent-death cleanup must include it.
        subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],close_fds=False)
        event('saved_barrier')
        time.sleep(60)
write(out/'WORKER_STATE.json',dict(protocol=1,state='completed'))
'''


class Fixture:
    def __init__(self, base, episodes=1, workers=1, accelerated=False):
        self.base=base; self.root=base/'scripts'; self.root.mkdir()
        self.runtime=base/'runtime'; (self.runtime/'validators').mkdir(parents=True)
        self.candidates=base/'candidates'
        folder=self.candidates/'graspgen/block55';folder.mkdir(parents=True)
        (folder/'cache.npz').write_bytes(b'fixed candidate input')
        durable_write(folder/'REPORT.json',dict(source_cache='cache.npz'))
        preamble=f'import sys\nsys.path.insert(0,{str(ROOT)!r})\n'
        (self.root/'sim_worker.py').write_text(preamble+WORKER)
        (self.root/'planner_service.py').write_text(
            'import argparse,json,time,os\nfrom pathlib import Path\n'
            'p=argparse.ArgumentParser();p.add_argument("--ready");a,_=p.parse_known_args()\n'
            'Path(a.ready).write_text(json.dumps({"port":1,"gpu":os.environ.get("CUDA_VISIBLE_DEVICES")}))\ntime.sleep(60)\n')
        for script in ('validate_episode.py','validate_raw_forces.py'):
            (self.runtime/'validators'/script).write_text(
                'import argparse,json\nfrom pathlib import Path\n'
                'p=argparse.ArgumentParser();p.add_argument("episode");p.add_argument("--output");a=p.parse_args()\n'
                'Path(a.output).write_text(json.dumps({"status":"passed"}))\n')
        self.control=base/'control';self.control.write_text('success')
        self.events=base/'events'
        self.env={**os.environ,'RECOVERY_TEST_CONTROL':str(self.control),
                  'RECOVERY_TEST_EVENTS':str(self.events),'PYTHONPATH':str(ROOT)}
        self.args=SimpleNamespace(output=base/'raw',runtime=self.runtime,candidate_root=self.candidates,
            resume=False,initialization_retries=2,skip_validation=False,gui=False,
            startup_timeout=10.,episode_timeout=10.)
        self.tokens=list(map(str,range(8))) if workers>3 else ['0']*4
        self.plan=collection_plan(['block55'],episodes,workers,2026111500)
        self.plan.update(arm_speed_profile={'name':'baseline'},grasp_symmetry='off')
        if accelerated:
            self.plan['execution_profile']='accelerated'

    def run(self, resume=False):
        self.args.resume=resume
        with patch('collect.HERE',self.root),patch.dict(os.environ,self.env):
            return run_collection(self.plan,self.args,self.tokens,{},sys.executable,sys.executable)

    def lines(self):
        return self.events.read_text().splitlines() if self.events.exists() else []


class CollectionRecoveryTests(unittest.TestCase):
    def test_wall_reaches_worker_subprocess_and_resume_keeps_mode(self):
        for accelerated in (False, True):
            with self.subTest(accelerated=accelerated), tempfile.TemporaryDirectory() as tmp:
                f = Fixture(Path(tmp), episodes=2, accelerated=accelerated)
                f.plan['obstacle'] = 'wall'
                self.assertEqual(f.run()['status'], 'completed')
                ledger = read(f.args.output / 'SLOTS.json')
                self.assertEqual(ledger['counts'], {'success': 2})
                for slot in ledger['slots'].values():
                    self.assertEqual(read(Path(slot['output']) / 'SCENE.json')['obstacle_mode'], 'wall')
                before = f.lines()
                f.run(resume=True)
                self.assertEqual(before, f.lines())
                f.plan['obstacle'] = 'none'
                with self.assertRaisesRegex(ValueError, 'original fixed plan'):
                    f.run(resume=True)
                self.assertEqual(before, f.lines())

    def test_silent_wall_downgrade_is_invalid_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            f.plan['obstacle'] = 'wall'
            f.control.write_text('obstacle_downgrade')
            with self.assertRaises(RuntimeError):
                f.run()
            slot = read(f.args.output / 'SLOTS.json')['slots']['block55_000000']
            self.assertEqual(slot['state'], 'invalid_data')
            self.assertEqual(slot['reason'], 'saved_obstacle_scene_mismatch')
            before = f.lines()
            with self.assertRaises(RuntimeError):
                f.run(resume=True)
            self.assertEqual(before, f.lines())

    def test_eight_dedicated_planners_async_validation_and_warmup_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp),episodes=16,workers=8,accelerated=True)
            self.assertEqual(f.run()['status'],'completed')
            ledger=read(f.args.output/'SLOTS.json')
            self.assertEqual(ledger['counts'],{'success':16})
            self.assertEqual(len(ledger['planner_attempts']),8)
            self.assertEqual({a['gpu'] for a in ledger['planner_attempts']},set(f.tokens))
            self.assertEqual(len(ledger['attempts']),8)
            for attempt in ledger['attempts']:
                route=read(Path(attempt['output'])/'ROUTING.json')
                self.assertEqual(route['simulation_gpu'],route['planner_gpu'])
                self.assertEqual(route['simulation_gpu'],attempt['simulation_gpu'])
            wave=ledger['waves'][0]
            self.assertEqual(len(wave['warmup_cases']),8)
            self.assertEqual(len(wave['measurement_cases']),8)
            self.assertTrue(all(ledger['slots'][c]['validation_completed_ns'] <= wave['measurement_started_ns']
                                for c in wave['warmup_cases']))
            self.assertGreater(wave['last_measurement_saved_ns'],wave['measurement_started_ns'])
            self.assertGreater(wave['last_measurement_validated_ns'],wave['measurement_started_ns'])
            for case in ledger['slots']:
                self.assertEqual(f.lines().count('physical:'+case),1)
            before=tree_manifest(f.args.output/'collection/workers')
            f.run(resume=True)
            self.assertEqual(before,tree_manifest(f.args.output/'collection/workers'))
            f.tokens.reverse()
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                f.run(resume=True)

    def test_failed_warmup_validation_blocks_all_measurement_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp),episodes=16,workers=8,accelerated=True)
            script=f.runtime/'validators/validate_raw_forces.py'
            script.write_text(script.read_text().replace('"passed"', '"failed"'))
            with self.assertRaisesRegex(RuntimeError,'validation failed'):
                f.run()
            calls=[v for v in f.lines() if v.startswith('physical:')]
            self.assertEqual(len(calls),8)
            self.assertTrue(all(int(v.rsplit('_',1)[1])<8 for v in calls))
            self.assertFalse(read(f.args.output/'SLOTS.json')['waves'][0].get('measurement_started_ns'))
            before=f.lines()
            with self.assertRaisesRegex(RuntimeError,'unknown'):
                f.run(resume=True)
            self.assertEqual(before,f.lines())

    def test_dedicated_planner_failure_prevents_every_worker_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp),episodes=8,workers=8,accelerated=True)
            script=f.root/'planner_service.py'
            script.write_text(script.read_text().replace('Path(a.ready).write_text',
                "if os.environ.get('CUDA_VISIBLE_DEVICES')=='3': raise RuntimeError('injected init error')\nPath(a.ready).write_text"))
            with self.assertRaisesRegex(RuntimeError,'initialization failed'):
                f.run()
            self.assertEqual(f.lines(),[])
            ledger=read(f.args.output/'SLOTS.json')
            self.assertEqual(ledger['counts'],{'pending':8})
            self.assertEqual(ledger['attempts'],[])

    def test_checkpoint_keeps_full_plan_and_resume_executes_only_remaining_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp),episodes=3)
            f.args.stop_after_per_object=1
            f.control.write_text('physical_failure')
            receipt=f.run()
            self.assertEqual(receipt['status'],'checkpoint')
            ledger=read(f.args.output/'SLOTS.json')
            self.assertEqual(ledger['counts'],{'physical_failure':1,'pending':2})
            first=ledger['slots']['block55_000000']
            before=tree_manifest(first['output'])
            self.assertEqual(read(f.args.output/'PLAN.json'),f.plan)
            self.assertFalse((f.args.output/'VALIDATION_SUMMARY.json').exists())
            # Repeating the same checkpoint launches nothing.
            events=f.lines();f.run(resume=True)
            self.assertEqual(events,f.lines())
            f.args.stop_after_per_object=None
            f.control.write_text('success')
            self.assertEqual(f.run(resume=True)['status'],'completed')
            self.assertEqual(read(f.args.output/'SLOTS.json')['counts'],{'physical_failure':1,'success':2})
            self.assertEqual(before,tree_manifest(first['output']))
            for index in range(3):
                self.assertEqual(f.lines().count(f'physical:block55_{index:06d}'),1)

    def test_initialization_oom_recovers_without_duplicate_execution_and_resume_reuses_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp));f.control.write_text('oom_once');f.run()
            slots=read(f.args.output/'SLOTS.json')
            self.assertEqual(slots['counts'],{'success':1})
            self.assertEqual(len(slots['attempts']),2)
            self.assertEqual(f.lines().count('physical:block55_000000'),1)
            slot=slots['slots']['block55_000000']
            before=tree_manifest(slot['output'])
            f.run(resume=True)
            self.assertEqual(f.lines().count('physical:block55_000000'),1)
            self.assertEqual(before,tree_manifest(slot['output']))

    def test_resource_retry_is_bounded_and_other_initialization_errors_are_not_retried(self):
        for mode,launches in [('oom_always',3),('import_error',1)]:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(Path(tmp));f.control.write_text(mode);f.run()
                self.assertEqual(f.lines().count('startup'),launches)
                self.assertFalse(any(v.startswith('physical:') for v in f.lines()))
                self.assertEqual(read(f.args.output/'SLOTS.json')['counts'],{'infrastructure_failed':1})
                f.run(resume=True)
                self.assertEqual(f.lines().count('startup'),launches)

    def test_physical_failure_is_terminal_across_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp));f.control.write_text('physical_failure');f.run()
            f.control.write_text('success');f.run(resume=True)
            self.assertEqual(f.lines().count('physical:block55_000000'),1)
            self.assertEqual(read(f.args.output/'SLOTS.json')['counts'],{'physical_failure':1})

    def test_interrupted_execution_is_retained_and_resume_refuses_to_repeat_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp));f.control.write_text('execution_exit')
            with self.assertRaisesRegex(RuntimeError,'Incomplete execution'):
                f.run()
            f.control.write_text('success')
            with self.assertRaisesRegex(RuntimeError,'unknown'):
                f.run(resume=True)
            self.assertEqual(f.lines().count('physical:block55_000000'),1)
            self.assertEqual(read(f.args.output/'SLOTS.json')['counts'],{'interrupted_unknown':1})

    def test_changed_raw_source_or_candidates_refuse_resume_before_dispatch(self):
        for target in ['raw','source','candidates']:
            with self.subTest(target=target),tempfile.TemporaryDirectory() as tmp:
                f=Fixture(Path(tmp));f.run();before=f.lines()
                if target=='raw':
                    slot=read(f.args.output/'SLOTS.json')['slots']['block55_000000']
                    path=Path(slot['output'])/'payload'
                elif target=='source': path=f.runtime/'validators/validate_episode.py'
                else: path=f.candidates/'graspgen/block55/cache.npz'
                path.write_bytes(b'changed input')
                with self.assertRaisesRegex(ValueError,'changed'):
                    f.run(resume=True)
                self.assertEqual(f.lines(),before)

    def test_sigkill_stops_owned_children_and_resumes_only_unstarted_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Fixture(Path(tmp),episodes=2);f.control.write_text('pause_after_first')
            driver=f.base/'driver.py'
            values={k:str(v) if isinstance(v,Path) else v for k,v in vars(f.args).items()}
            driver.write_text(f'''
import sys,json
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,{str(ROOT)!r})
import collect
from resumable_collect import run_collection
collect.HERE=Path({str(f.root)!r})
a=SimpleNamespace(**json.loads({json.dumps(values)!r}))
for k in ('output','runtime','candidate_root'): setattr(a,k,Path(getattr(a,k)))
run_collection(json.loads({json.dumps(f.plan)!r}),a,['0']*4,{{}},sys.executable,sys.executable)
''')
            with (f.base/'driver.log').open('w') as log:
                parent=subprocess.Popen([sys.executable,'-B',str(driver)],env=f.env,stdout=log,stderr=log)
                try:
                    deadline=time.monotonic()+10
                    while 'saved_barrier' not in f.lines() and parent.poll() is None and time.monotonic()<deadline:
                        time.sleep(.05)
                    self.assertIn('saved_barrier',f.lines())
                    os.kill(parent.pid,signal.SIGKILL);parent.wait(timeout=5)
                    deadline=time.monotonic()+10
                    while True:
                        try:
                            with RunLock(f.args.output/'.collection.lock'): pass
                            break
                        except RuntimeError:
                            if time.monotonic()>deadline: raise
                            time.sleep(.05)
                    f.control.write_text('success');f.run(resume=True)
                    self.assertEqual(read(f.args.output/'SLOTS.json')['counts'],{'success':2})
                    self.assertFalse(read(f.args.output/'RUN.json')['timing_complete'])
                    self.assertEqual(f.lines().count('physical:block55_000000'),1)
                    self.assertEqual(f.lines().count('physical:block55_000001'),1)
                finally:
                    if parent.poll() is None: parent.kill();parent.wait(timeout=5)

    def test_production_worker_records_pre_app_initialization_resource_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);job=base/'job.json';output=base/'worker'
            durable_write(job,dict(cases=[dict(case_id='block55_000000',object_id='block55')]))
            code='''
import runpy,sys,types
class Launcher:
    @staticmethod
    def add_app_launcher_args(parser): pass
    def __init__(self,args): raise RuntimeError('CUDA out of memory: injected before application construction')
package=types.ModuleType('isaaclab');package.__path__=[]
module=types.ModuleType('isaaclab.app');module.AppLauncher=Launcher
sys.modules['isaaclab']=package;sys.modules['isaaclab.app']=module
sys.argv=['sim_worker.py','--runtime',sys.argv[1],'--job',sys.argv[2],'--address',sys.argv[3],'--output',sys.argv[4]]
runpy.run_path('sim_worker.py',run_name='__main__')
'''
            result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'runtime'),str(job),
                                   str(base/'address'),str(output)],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            state=read(output/'WORKER_STATE.json')
            self.assertEqual(state['failed_phase'],'initializing')
            self.assertTrue(state['recoverable_initialization'])
            self.assertFalse((output/'READY.json').exists())
            self.assertFalse((output/'block55_000000').exists())


class PipelineRecoveryTests(unittest.TestCase):
    def test_speed_and_obstacle_reach_frozen_collector_and_survive_resume(self):
        for requested, expected, obstacle in [(None, 'recommended', None), ('baseline', 'baseline', None),
                                               (None, 'recommended', 'wall')]:
            with self.subTest(profile=requested, obstacle=obstacle), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                checkout = base / 'checkout'
                (checkout / 'runtime').mkdir(parents=True)
                (checkout / 'tools').mkdir()
                candidates = base / 'candidates'
                folder = candidates / 'graspgen/block55'
                folder.mkdir(parents=True)
                (folder / 'cache.npz').write_bytes(b'fixed candidate input')
                durable_write(folder / 'REPORT.json', dict(source_cache='cache.npz'))
                # A CPU-only collector fixture writes a checkpoint. It exercises the
                # real pipeline registration, source freeze, subprocess and resume.
                (checkout / 'collect.py').write_text(f'''
import argparse, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(ROOT / 'runtime/collector')!r})
from common import write
from arm_speed import parse
p = argparse.ArgumentParser()
p.add_argument('--output', type=Path)
p.add_argument('--arm-speed-profile', required=True)
p.add_argument('--obstacle', required=True)
p.add_argument('--resume', action='store_true')
a, _ = p.parse_known_args()
a.output.mkdir(parents=True, exist_ok=a.resume)
write(a.output / 'PROFILE.json', parse(a.arm_speed_profile).describe())
write(a.output / 'OBSTACLE.json', dict(mode=a.obstacle))
write(a.output / 'RUN.json', dict(status='checkpoint'))
write(a.output / 'SLOTS.json', dict(slots={{'block55_000000': dict(
    state='success', entry=dict(fixed_index=0))}}))
with (a.output / 'launches').open('a') as stream:
    stream.write(a.arm_speed_profile + '\\n')
''')
                output = base / 'run'
                argv = ['pipeline.py', '--output', str(output), '--objects', 'block55',
                        '--episodes-per-object', '1', '--stop-after-per-object', '1',
                        '--candidate-root', str(candidates), '--lerobot-python', sys.executable]
                if requested:
                    argv += ['--arm-speed-profile', requested]
                if obstacle:
                    argv += ['--obstacle', obstacle]
                # An ambient setting must not override the registered CLI/default.
                # Geometry has separate real-asset entrypoint regressions. This
                # fixture intentionally contains no robot or object assets.
                with patch('initial_geometry.preflight', return_value=dict(status='passed', cases=[])), \
                        patch('recoverable_pipeline.ROOT', checkout), patch('sys.argv', argv), \
                        patch.dict(os.environ, {'ARM_SPEED_PROFILE': 'baseline' if expected == 'recommended' else 'recommended'}):
                    pipeline_main()
                self.assertEqual(read(output / 'PIPELINE.json')['arguments']['arm_speed_profile'], expected)
                self.assertEqual(read(output / 'raw/PROFILE.json')['name'], expected)
                self.assertEqual(read(output / 'PIPELINE.json')['arguments']['obstacle'], obstacle or 'none')
                self.assertEqual(read(output / 'raw/OBSTACLE.json')['mode'], obstacle or 'none')

                resume = ['pipeline.py', '--output', str(output), '--resume',
                          '--stop-after-per-object', '1']
                # main normally re-execs this immutable source snapshot on resume.
                with patch('recoverable_pipeline.ROOT', output / 'source'), patch('sys.argv', resume):
                    pipeline_main()
                self.assertEqual((output / 'raw/launches').read_text().splitlines(), [expected, expected])
                self.assertEqual(read(output / 'raw/PROFILE.json')['name'], expected)
                self.assertEqual(read(output / 'raw/OBSTACLE.json')['mode'], obstacle or 'none')

                opposite = 'baseline' if expected == 'recommended' else 'recommended'
                with patch('recoverable_pipeline.ROOT', output / 'source'), \
                        patch('sys.argv', resume + ['--arm-speed-profile', opposite]), \
                        self.assertRaises(SystemExit) as rejected:
                    pipeline_main()
                self.assertEqual(rejected.exception.code, 2)
                self.assertEqual((output / 'raw/launches').read_text().splitlines(), [expected, expected])
                with patch('recoverable_pipeline.ROOT', output / 'source'), \
                        patch('sys.argv', resume + ['--obstacle', 'none' if obstacle == 'wall' else 'wall']), \
                        self.assertRaises(SystemExit) as rejected:
                    pipeline_main()
                self.assertEqual(rejected.exception.code, 2)
                self.assertEqual((output / 'raw/launches').read_text().splitlines(), [expected, expected])

    def test_export_interruption_and_training_failure_preserve_data_and_resume_at_the_needed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source';tools=source/'tools/lerobot_export';tools.mkdir(parents=True)
            raw=base/'raw';raw.mkdir();(raw/'payload').write_text('original raw')
            durable_write(raw/'RAW_COMMITTED.json',dict(run_id='test',sha256=tree_manifest(raw)))
            export_control=base/'export_control';export_control.write_text('fail')
            train_control=base/'train_control';train_control.write_text('fail')
            events=base/'events'
            shared=f'''
import argparse,sys
from pathlib import Path
sys.path.insert(0,{str(ROOT)!r})
from common import read
from recovery import durable_write as write,digest
events=Path({str(events)!r})
'''
            (tools/'run.py').write_text(shared+f'''
p=argparse.ArgumentParser();p.add_argument('--output');a,_=p.parse_known_args()
out=Path(a.output);out.mkdir();(out/'portable').mkdir()
with events.open('a') as s: s.write('export\\n')
control=Path({str(export_control)!r})
if control.read_text()=='fail': raise SystemExit(13)
write(out/'SELECTION.json',dict(static_and_motion_verified=True))
write(out/'portable/SELECTION.json',dict(static_and_motion_verified=True))
write(out/'AUDIT.json',dict(status='passed',statistics=dict(status='passed')))
''')
            (tools/'train_check.py').write_text(shared+f'''
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--portable');p.add_argument('--audit');p.add_argument('--steps',type=int);a=p.parse_args()
out=Path(a.output);out.mkdir()
with events.open('a') as s: s.write('training\\n')
control=Path({str(train_control)!r})
if control.read_text()=='fail': raise SystemExit(14)
write(out/'RESULT.json',dict(status='passed',steps=a.steps,audit_sha256=digest(a.audit),selection_sha256=digest(Path(a.portable)/'SELECTION.json')))
''')
            receipt=dict(run_id='test',source_snapshot=str(source),created_unix_s=time.time(),stages=[],
                arguments=dict(lerobot_python=sys.executable,data_only=False,gpus='0',train_steps=100),
                data_status='not_started',training_status='not_started')
            with RunLock(base/'.pipeline.lock') as lock:
                pipeline=Pipeline(base,receipt,lock)
                with self.assertRaisesRegex(RuntimeError,'export_and_readback failed'):pipeline.run()
                self.assertFalse((base/'export').exists())
                self.assertTrue((base/'work/export_000').exists())
                raw_before=tree_manifest(raw)
                export_control.write_text('success')
                with self.assertRaisesRegex(RuntimeError,'training failed'):pipeline.run()
                self.assertEqual(receipt['data_status'],'passed')
                self.assertEqual(receipt['training_status'],'failed')
                export_before=tree_manifest(base/'export')
                receipt['arguments']['data_only']=True;pipeline.run()
                self.assertEqual(events.read_text().splitlines(),['export','export','training'])
                receipt['arguments']['data_only']=False;train_control.write_text('success');pipeline.run()
                self.assertEqual(receipt['training_status'],'passed')
                self.assertEqual(events.read_text().splitlines(),['export','export','training','training'])
                self.assertEqual(raw_before,tree_manifest(raw))
                self.assertEqual(export_before,tree_manifest(base/'export'))
                verify_commit(base/'export','test')
                (base/'export/portable/SELECTION.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'changed'):pipeline.run()
                self.assertEqual(events.read_text().splitlines(),['export','export','training','training'])

    def test_run_lock_excludes_a_second_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock'
            with RunLock(path):
                result=subprocess.run([sys.executable,'-B','-c',
                    'from recovery import RunLock\nimport sys\nwith RunLock(sys.argv[1]): pass',str(path)],
                    cwd=ROOT,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('still owned',result.stderr)
            with RunLock(path):pass


if __name__=='__main__':unittest.main()
