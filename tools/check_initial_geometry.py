"""Check every requested reset/settled geometry without Isaac, torch or CUDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_plan(plan, runtime, obstacle):
    # This helper runs in its own interpreter so geometry always comes from
    # the requested runtime, including a pipeline's frozen snapshot.
    runtime = Path(runtime).resolve()
    sys.path[:0] = [str(runtime), str(runtime / 'collector')]
    from geometry.api import check_robot_path, manifest
    from obstacle_scene import planner_obstacles
    from initial_state import requested_pose
    from common import read

    scene = read(runtime / 'config/scene.json')
    assets = manifest()['objects']
    obstacles = planner_obstacles(obstacle)
    q = scene['initial_qpos']
    rows = []
    for stage in plan['stages']:
        for entry in stage['cases']:
            states = {}
            for name, settled in (('spawn', False), ('settled', True)):
                pose = requested_pose(entry, assets[entry['object_id']],
                                      scene['object_reference_xy_m'], settled=settled)
                geometry = check_robot_path(q, sum(q[7:]), pose, entry['object_id'],
                                            obstacles=obstacles, compress=False)
                states[name] = dict(pose_wxyz=pose, geometry=geometry)
            rows.append(dict(stage=stage['name'], case_id=entry['case_id'],
                             object_id=entry['object_id'], states=states,
                             valid=all(s['geometry']['valid'] for s in states.values())))
    if not rows:
        raise ValueError('Initial geometry preflight requires a nonempty schedule')
    invalid = sum(not row['valid'] for row in rows)
    return dict(protocol=1, status='failed' if invalid else 'passed',
                scheduled_slots=len(rows), invalid_slots=invalid,
                obstacle=obstacle, cases=rows,
                scope='Complete robot/target/table/obstacle geometry at requested spawn and settled poses; '
                      'physical support tolerance unchanged; no sampling, filtering or retries. '
                      'Online settled-pose and geometry audits remain required.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--obstacle', choices=('none', 'wall'), required=True)
    args = parser.parse_args()
    print(json.dumps(check_plan(json.load(sys.stdin), args.runtime, args.obstacle),
                     ensure_ascii=False, allow_nan=False))
