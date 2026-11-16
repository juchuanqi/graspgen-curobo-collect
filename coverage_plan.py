"""Frozen, stratified starts for the four bundled upright objects."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

from common import OBJECTS, collection_plan, read, write
from initial_state import requested_pose

LOCATIONS = {'center': (0., 0.), 'southwest': (-.014, -.014),
             'southeast': (.014, -.014), 'northwest': (-.014, .014),
             'northeast': (.014, .014)}
ANGLE_ORDER = (0, 5, 2, 7, 4, 9, 1, 6, 3, 8)


def make_coverage_plan(runtime, workers=3, seed=2026111510):
    plan = collection_plan(list(OBJECTS), 50, workers, seed)
    assets = read(Path(runtime) / 'assets/MANIFEST.json')['objects']
    reference = read(Path(runtime) / 'config/scene.json')['object_reference_xy_m']
    for case in plan['stages'][0]['cases']:
        index = case['fixed_index']
        location = list(LOCATIONS)[index % 5]
        rng = random.Random(case['scene_seed'])
        case['xy_offset_m'] = [v + rng.uniform(-.001, .001) for v in LOCATIONS[location]]
        axial = case['object_id'] in ('cylinder36h52', 'disk56h24')
        angle_bin = ANGLE_ORDER[index // 5]
        case['yaw_deg'] = 0. if axial else 9. + 18. * angle_bin
        case['coverage'] = dict(location=location,
            orientation='axial_equivalent' if axial else f'yaw_bin_{angle_bin:02d}',
            geometric_yaw_period_deg=None if axial else 180,
            checkpoint='initial40' if index < 10 else 'remaining160')
        case['requested_settled_pose_wxyz'] = requested_pose(case, assets[case['object_id']], reference)
    plan['scope'] = 'Known meshes; upright stable support; five XY strata in the supported +/-15 mm square'
    plan['coverage_protocol'] = dict(version=1, slots_per_object=50, initial_slots_per_object=10,
        locations=LOCATIONS, jitter_m=.001, orientation_bins=10,
        orientation_order=list(ANGLE_ORDER),
        symmetry='Both rectangular prisms have 180-degree yaw equivalence; upright cylinder/disk yaw is fixed at zero and counts as one geometric orientation. No roll/pitch coverage.',
        continuation='Inspect first40 completeness and all failures, then continue the same list. A required code/config change requires a new registered batch; no replacement successes.',
        denominators=['success/planned', 'success/physical_attempts', 'export/planned',
                      'success/static_qualified_with_known_grasp_outcome'])
    return plan


def load_fixed_plan(path, objects, episodes, workers, seed):
    """Accept pose changes only within the existing collection contract."""
    plan = read(path)
    base = collection_plan(objects, episodes, workers, seed)
    for key, expected in base.items():
        if key not in {'stages', 'scope'} and plan.get(key) != expected:
            raise ValueError(f'Fixed plan contract mismatch: {key}')
    stages = plan.get('stages', [])
    if len(stages) != 1 or {k: v for k, v in stages[0].items() if k != 'cases'} != {
            k: v for k, v in base['stages'][0].items() if k != 'cases'}:
        raise ValueError('Fixed plan requires the registered single collection stage and worker count')
    cases = stages[0].get('cases', [])
    if len(cases) != len(base['stages'][0]['cases']):
        raise ValueError('Fixed plan slot count mismatch')
    for case, expected in zip(cases, base['stages'][0]['cases']):
        for key, value in expected.items():
            if key not in {'xy_offset_m', 'yaw_deg'} and case.get(key) != value:
                raise ValueError(f'Fixed plan identity/attempt contract mismatch: {key}')
        xy, yaw = case.get('xy_offset_m'), case.get('yaw_deg')
        if (not isinstance(xy, list) or len(xy) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                    not math.isfinite(v) or abs(v) > .015 for v in xy) or
                isinstance(yaw, bool) or not isinstance(yaw, (float, int)) or
                not math.isfinite(yaw) or abs(yaw) > 180):
            raise ValueError('Fixed plan pose exceeds the supported XY/yaw domain')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, default=Path(__file__).resolve().parent / 'runtime')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026111510)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write(args.output, make_coverage_plan(args.runtime, args.workers, args.seed))
