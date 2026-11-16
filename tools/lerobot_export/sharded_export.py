"""Partition one global selection and run unchanged exporters/auditors in parallel."""
from __future__ import annotations

import copy
from pathlib import Path
import re
import time

from bridge import read, sha, write
from delivery import verify_shards


def partition_selection(selection):
    """Keep same-object duplicates and per-object limits resolved globally."""
    if selection.get('status') != 'selected_before_export' or not selection.get('episodes'):
        raise ValueError('No eligible episodes to partition')
    cases = [ep['case_id'] for ep in selection['episodes']]
    if len(set(cases)) != len(cases):
        raise ValueError('Duplicate selected episode')
    objects = sorted({ep['object_id'] for ep in selection['episodes']})
    result = []
    for obj in objects:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', obj):
            raise ValueError('Unsafe object ID in shard selection')
        child = copy.deepcopy(selection)
        child['episodes'] = [ep for ep in child['episodes'] if ep['object_id'] == obj]
        for index, ep in enumerate(child['episodes']):
            ep['export_episode_index'] = index
        inspected = [r for r in child['inspected_slots'] if r['object_id'] == obj]
        child.update(inspected_slots=inspected, fixed_slots=len(inspected),
                     eligible_slots=sum(r['eligible'] for r in inspected),
                     selected_episodes=len(child['episodes']), per_object={obj: len(child['episodes'])})
        result.append((obj, child))
    return result


def run_sharded(selection_path, output, python, env, workers):
    # Imports remain standard-library-only in run.py; this module runs in the
    # verified LeRobot environment after strict selection has completed.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from collect import Processes
    from recovery import RunLock
    if workers < 2:
        raise ValueError('Parallel export requires at least two workers')
    started = time.monotonic()
    tools = Path(__file__).resolve().parent
    selection = read(selection_path)
    parts = partition_selection(selection)
    portable = output / 'portable'
    portable.mkdir()
    (portable / 'shards').mkdir()
    (output / 'shard_selections').mkdir()
    import shutil
    shutil.copy2(selection_path, portable / 'SELECTION.json')
    jobs = []
    for obj, child in parts:
        chosen = output / 'shard_selections' / f'{obj}.json'
        write(chosen, child)
        jobs.append(dict(object_id=obj, selection=chosen, package=portable / 'shards' / obj))
    running, finished = [], []
    with RunLock(output / '.shards.lock') as lock:
        processes = Processes(lock.fd)
        try:
            while jobs or running:
                while jobs and len(running) < workers:
                    job = jobs.pop(0)
                    job.update(phase='export', started=time.monotonic())
                    job['child'] = processes.start([python, '-B', str(tools / 'export_lerobot.py'),
                        '--selection', str(job['selection']), '--output', str(job['package'])], env,
                        output / f"{job['object_id']}_export.log")
                    running.append(job)
                for job in list(running):
                    code = job['child'].poll()
                    if code is None:
                        continue
                    if code:
                        raise RuntimeError(f"{job['object_id']} {job['phase']} failed; inspect its shard log")
                    if job['phase'] == 'export':
                        job['phase'] = 'audit'
                        job['child'] = processes.start([python, '-B', str(tools / 'audit_lerobot.py'),
                            str(job['package']), '--output', str(job['package'] / 'AUDIT.json')], env,
                            output / f"{job['object_id']}_audit.log")
                    else:
                        job['wall_s'] = time.monotonic() - job['started']
                        running.remove(job)
                        finished.append(job)
                        print('SHARD_AUDITED', job['object_id'], flush=True)
                if jobs or running:
                    time.sleep(.1)
        finally:
            processes.close()
    records = []
    for job in sorted(finished, key=lambda j: j['object_id']):
        package = job['package']
        manifest = read(package / 'EXPORT_MANIFEST.json')
        records.append(dict(object_id=job['object_id'], package=str(package.relative_to(portable)),
            episodes=len(manifest['episodes']), rows=manifest['rows'], wall_s=job['wall_s'],
            selection_sha256=sha(package / 'SELECTION.json'),
            export_manifest_sha256=sha(package / 'EXPORT_MANIFEST.json'), audit_sha256=sha(package / 'AUDIT.json')))
    write(portable / 'SHARDS.json', dict(schema=1, format='lerobot_v3_shards', fps=60,
        selection_sha256=sha(portable / 'SELECTION.json'), shards=records,
        episodes=len(selection['episodes']), rows=sum(ep['length'] for ep in selection['episodes']),
        workers=workers, partition='one package per object after global selection',
        excluded_slots=[r for r in selection['inspected_slots'] if not r.get('selected')]))
    totals = verify_shards(portable)
    write(output / 'AUDIT.json', dict(status='passed', **totals, shards_sha256=sha(portable / 'SHARDS.json'),
        wall_s=time.monotonic()-started, new_physics=False, training=False,
        scope='All shards passed official full readback; exact global selection union verified'))


if __name__ == '__main__':
    import argparse
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    import sys
    run_sharded(args.selection, args.output, sys.executable, os.environ.copy(), args.workers)
