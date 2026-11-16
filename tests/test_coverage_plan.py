"""Coverage and fixed input validation; no simulator or GPU required."""
import copy
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import OBJECTS, write
from coverage_plan import make_coverage_plan, load_fixed_plan
from initial_state import requested_pose
from common import read


class CoveragePlanTests(unittest.TestCase):
    def test_preregistered_prefix_balances_locations_and_avoids_axial_rotation_inflation(self):
        plan = make_coverage_plan(ROOT / 'runtime')
        assets=read(ROOT/'runtime/assets/MANIFEST.json')['objects']
        xy=read(ROOT/'runtime/config/scene.json')['object_reference_xy_m']
        self.assertEqual(plan,make_coverage_plan(ROOT/'runtime'))
        self.assertEqual(plan['total_attempt_slots'],200)
        for name in OBJECTS:
            cases=[c for c in plan['stages'][0]['cases'] if c['object_id']==name]
            self.assertEqual(len(cases),50)
            self.assertEqual(set(Counter(c['coverage']['location'] for c in cases[:10]).values()),{2})
            self.assertEqual(len({tuple(c['xy_offset_m']) for c in cases}),50)
            self.assertTrue(all(abs(v)<=.015 for c in cases for v in c['xy_offset_m']))
            angles={c['yaw_deg'] for c in cases}
            self.assertEqual(len(angles),1 if name in ('cylinder36h52','disk56h24') else 10)
            for c in cases:
                self.assertEqual(c['requested_settled_pose_wxyz'],requested_pose(c,assets[name],xy))

    def test_fixed_plan_rejects_identity_attempt_physics_and_out_of_region_changes(self):
        plan=make_coverage_plan(ROOT/'runtime')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json'
            write(path,plan)
            self.assertEqual(load_fixed_plan(path,list(OBJECTS),50,3,2026111510),read(path))
            for mutation in ('physics','identity','attempts','xy','nan','length','lanes'):
                with self.subTest(mutation=mutation):
                    changed=copy.deepcopy(plan);case=changed['stages'][0]['cases'][0]
                    if mutation=='physics':changed['physics_hz']=240
                    if mutation=='identity':case['case_id']='../other'
                    if mutation=='attempts':case['max_physical_attempts_per_method']=2
                    if mutation=='xy':case['xy_offset_m'][0]=.016
                    if mutation=='nan':case['yaw_deg']=float('nan')
                    if mutation=='length':changed['stages'][0]['cases'].pop()
                    if mutation=='lanes':changed['stages'][0]['lanes']=1
                    if mutation=='nan':
                        import json
                        path.write_text(json.dumps(changed))
                    else:write(path,changed)
                    with self.assertRaises(ValueError):
                        load_fixed_plan(path,list(OBJECTS),50,3,2026111510)


if __name__=='__main__':unittest.main()
