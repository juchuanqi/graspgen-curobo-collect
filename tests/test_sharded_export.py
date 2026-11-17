"""Selection union, receipt integrity and failed parallel-stage publication."""
from copy import deepcopy
from pathlib import Path
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/lerobot_export'))
from bridge import read, write, sha
from delivery import validate_delivery, verify_shards
from sharded_export import partition_selection, run_sharded


def selection():
    episodes = [dict(case_id=f'{obj}_{i}', object_id=obj, export_episode_index=n,
                     source=f'/raw/{obj}_{i}', source_sha256={'trajectory.npz': 'fixed'}, length=10+i,
                     task='pickup', scene_seed=n)
                for n, (obj, i) in enumerate([('block55',0), ('block55',1), ('disk56h24',0)])]
    inspected = [dict(case_id=ep['case_id'], object_id=ep['object_id'], eligible=True, selected=True)
                 for ep in episodes]
    inspected.append(dict(case_id='disk56h24_1', object_id='disk56h24', eligible=False,
                          exclusion='physical_success_or_validation_missing'))
    return dict(status='selected_before_export', static_and_motion_verified=True,
                fixed_slots=4, eligible_slots=3, selected_episodes=3,
                per_object={'block55':2, 'disk56h24':1}, episodes=episodes, inspected_slots=inspected)


def delivered(root):
    original = selection()
    write(root / 'SELECTION.json', original)
    records = []
    for obj, chosen in partition_selection(original):
        package = root / 'shards' / obj
        package.mkdir(parents=True)
        write(package / 'SELECTION.json', chosen)
        export = dict(selection_sha256=sha(package / 'SELECTION.json'), dataset_root='dataset',
                      episodes=[dict(ep,sidecar=f"sidecars/episode_{i:06d}") for i,ep in enumerate(chosen['episodes'])],
                      rows=sum(ep['length'] for ep in chosen['episodes']))
        write(package / 'EXPORT_MANIFEST.json', export)
        write(package / 'AUDIT.json', dict(status='passed', full_rows=export['rows'],
              rgb_comparisons=2*export['rows'], terminal_rgb_comparisons=2*len(chosen['episodes']),
              padding_checks=8*len(chosen['episodes']),
              statistics=dict(status='passed',checks=7,max_absolute_error=0.)))
        records.append(dict(object_id=obj, package=str(package.relative_to(root)),
            episodes=len(chosen['episodes']), rows=export['rows'],
            selection_sha256=sha(package / 'SELECTION.json'),
            export_manifest_sha256=sha(package / 'EXPORT_MANIFEST.json'),
            audit_sha256=sha(package / 'AUDIT.json')))
    write(root / 'SHARDS.json', dict(schema=1,format='lerobot_v3_shards',shards=records,
          episodes=3, rows=31, selection_sha256=sha(root / 'SELECTION.json')))
    totals = verify_shards(root)
    write(root / 'AUDIT.json',dict(status='passed',**totals,shards_sha256=sha(root / 'SHARDS.json')))


class ShardedExportTests(unittest.TestCase):
    def test_partition_reindexes_without_losing_failure_or_mutating_global_selection(self):
        original = selection()
        before = deepcopy(original)
        parts = dict(partition_selection(original))
        self.assertEqual(original, before)
        self.assertEqual(parts['disk56h24']['episodes'][0]['export_episode_index'],0)
        self.assertEqual(parts['disk56h24']['fixed_slots'],2)
        self.assertEqual(parts['disk56h24']['inspected_slots'][-1]['exclusion'],
                         'physical_success_or_validation_missing')
        self.assertEqual(sum(p['selected_episodes'] for p in parts.values()),3)
        original['episodes'].append(deepcopy(original['episodes'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            partition_selection(original)

    def test_delivery_moves_as_one_directory_and_keeps_exact_union(self):
        with tempfile.TemporaryDirectory() as temp:
            old = Path(temp)/'work';old.mkdir()
            delivered(old)
            moved = Path(temp)/'final'
            old.rename(moved)
            validate_delivery(moved,moved/'AUDIT.json')
            manifest = read(moved/'SHARDS.json')
            manifest['shards'].pop()
            write(moved/'SHARDS.json',manifest)
            with self.assertRaisesRegex(ValueError,'union'):
                verify_shards(moved)

    def test_receipt_tampering_and_failed_audit_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);delivered(root)
            audit=root/'shards/block55/AUDIT.json'
            value=read(audit);value['status']='failed';write(audit,value)
            with self.assertRaisesRegex(ValueError,'receipt changed'):
                verify_shards(root)
            manifest=read(root/'SHARDS.json');manifest['shards'][0]['audit_sha256']=sha(audit)
            write(root/'SHARDS.json',manifest)
            with self.assertRaisesRegex(ValueError,'Every shard'):
                verify_shards(root)

    def test_external_package_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);delivered(root)
            manifest=read(root/'SHARDS.json');manifest['shards'][0]['package']='../../external'
            write(root/'SHARDS.json',manifest)
            with self.assertRaisesRegex(ValueError,'stay inside'):
                verify_shards(root)

    def test_failed_parallel_subprocess_never_publishes_combined_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);chosen=root/'SELECTION.json';write(chosen,selection())
            # Exercise real process creation, cleanup and failure propagation;
            # no LeRobot, simulator or model invocation is required.
            script=root/'fail.py';script.write_text('raise SystemExit(19)\n')
            import collect
            start=collect.Processes.start
            def failing_start(processes, command, env, log):
                return start(processes,[sys.executable,'-B',str(script)],env,log)
            with patch.object(collect.Processes,'start',failing_start):
                with self.assertRaisesRegex(RuntimeError,'export failed'):
                    run_sharded(chosen,root,sys.executable,os.environ.copy(),2)
            self.assertFalse((root/'AUDIT.json').exists())
            self.assertFalse((root/'portable/SHARDS.json').exists())

    def test_pipeline_publishes_moves_and_reuses_shards_before_later_training(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from recoverable_pipeline import Pipeline, verify_commit
        from recovery import RunLock, durable_write, tree_manifest
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);tools=base/'source/tools/lerobot_export';tools.mkdir(parents=True)
            raw=base/'raw';raw.mkdir();(raw/'payload').write_text('original evidence')
            durable_write(raw/'RAW_COMMITTED.json',dict(run_id='shards',sha256=tree_manifest(raw)))
            preamble=f"import sys\nsys.path[:0]=[{str(ROOT)!r},{str(ROOT/'tests')!r},{str(ROOT/'tools/lerobot_export')!r}]\n"
            (tools/'delivery.py').write_text((ROOT/'tools/lerobot_export/delivery.py').read_text())
            (tools/'run.py').write_text(preamble+'''
import argparse,shutil
from pathlib import Path
from test_sharded_export import delivered
p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--workers',type=int);a,_=p.parse_known_args()
assert a.workers==4
out=Path(a.output);out.mkdir();(out/'portable').mkdir();delivered(out/'portable')
shutil.copy2(out/'portable/SELECTION.json',out/'SELECTION.json')
shutil.copy2(out/'portable/AUDIT.json',out/'AUDIT.json')
''')
            (tools/'train_check.py').write_text(preamble+'''
import argparse
from pathlib import Path
from delivery import validate_delivery,sha
from bridge import write
p=argparse.ArgumentParser();p.add_argument('--portable');p.add_argument('--audit');p.add_argument('--steps',type=int);p.add_argument('--output');a=p.parse_args()
validate_delivery(Path(a.portable),a.audit)
out=Path(a.output);out.mkdir()
write(out/'RESULT.json',dict(status='passed',steps=a.steps,audit_sha256=sha(a.audit),selection_sha256=sha(Path(a.portable)/'SELECTION.json')))
''')
            receipt=dict(run_id='shards',source_snapshot=str(base/'source'),created_unix_s=time.time(),stages=[],
                         arguments=dict(lerobot_python=sys.executable,data_only=True,gpus='0',train_steps=2,export_workers=4))
            with RunLock(base/'.pipeline.lock') as lock:
                pipeline=Pipeline(base,receipt,lock);pipeline.run()
                before=tree_manifest(base/'export')
                validate_delivery(base/'export/portable',base/'export/AUDIT.json')
                receipt['arguments']['data_only']=False
                pipeline.run();pipeline.run()
                self.assertEqual([s['name'] for s in receipt['stages']],['export_and_readback','training'])
                self.assertEqual(before,tree_manifest(base/'export'))
                self.assertEqual(receipt['training_status'],'passed')
                verify_commit(base/'export','shards')


if __name__ == '__main__':
    unittest.main()
