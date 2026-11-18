"""Larger declared pools must survive preparation and require explicit opt-in."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'runtime'),str(ROOT/'runtime/collector')]
from generate_candidates import restore_grasps
from geometry.api import prepare_candidates
from target_provider import load_candidate_source, screen_scene_candidates, source_limit


class SourceBudgetTests(unittest.TestCase):
    def test_variable_output_shape_restores_center_once(self):
        for count in (200,500,1000):
            raw=np.repeat(np.eye(4)[None],count,axis=0)
            center=np.array([.01,-.02,.03])
            actual=restore_grasps(raw,center,count)
            np.testing.assert_allclose(actual[:,:3,3],np.broadcast_to(center,(count,3)))
            np.testing.assert_array_equal(raw[:,:3,3],0)
            with self.assertRaises(ValueError):restore_grasps(raw[:-1],center,count)

    def test_no_silent_200_cap_in_object_qualification(self):
        count=1000;poses=np.repeat(np.eye(4)[None],count,axis=0)
        poses[:,0,3]=np.arange(count)*.006
        span=dict(span_m=.02,center_y_m=0.,positive_y_m=.01,negative_y_m=-.01)
        with patch('geometry.api._contact_span',return_value=span),patch(
                'geometry.api.check_gripper_path',return_value=dict(valid=True,reason='clear')):
            report=prepare_candidates('synthetic',poses,np.ones(count),max_candidates=count)
        self.assertEqual(report['generated'],count)
        self.assertEqual(report['unique_screened'],count)
        self.assertEqual(report['accepted_geometry'],count)

    def test_small_qualified_pool_cannot_hide_large_generation_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'graspgen/test';folder.mkdir(parents=True)
            cache=folder/'candidates.npz';cache.write_bytes(b'fixed test cache')
            report=dict(object_id='test',source_cache=cache.name,source_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                        source_kind='official test source',generated=1000,candidates=[dict(T_object_tcp=np.eye(4).tolist())])
            (folder/'REPORT.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):load_candidate_source(root,'graspgen','test')
            with self.assertRaises(ValueError):load_candidate_source(root,'graspgen','test',500)
            pool,_=load_candidate_source(root,'graspgen','test',1000)
            self.assertEqual(len(pool),1)

    def test_defaults_and_invalid_budgets_remain_guarded(self):
        self.assertEqual(source_limit(),200)
        for bad in (0,201,1001,True,500.,'1000'):
            with self.assertRaises(ValueError):source_limit(bad)
        with self.assertRaises(ValueError):screen_scene_candidates([{}]*201,{})
        with self.assertRaises(ValueError):screen_scene_candidates([],dict(source_candidate_limit=1000),maximum=51)


if __name__=='__main__':unittest.main()
