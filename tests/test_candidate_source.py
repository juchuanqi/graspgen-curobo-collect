"""New collection must not silently use handwritten pools in the graspgen namespace."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'runtime'), str(ROOT/'runtime/collector')]
from target_provider import load_candidate_source
from geometry.api import pad_normal_quality


class CandidateSourceTests(unittest.TestCase):
    def test_official_model_pool_loads_but_handwritten_and_mislabeled_pools_do_not(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/'graspgen/test'
            folder.mkdir(parents=True)
            cache = folder/'candidates.npz'
            np.savez_compressed(cache, T_object_grasp=np.eye(4)[None])
            report = dict(object_id='test', source_cache=cache.name,
                source_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                source_kind='GraspGen official Panda diffusion + discriminator, mesh point cloud',
                generation_calls=1, generated=200, model_candidates=200,
                candidates=[dict(T_object_tcp=np.eye(4).tolist())])
            path = folder/'REPORT.json'
            path.write_text(json.dumps(report))
            pool, _ = load_candidate_source(root, 'graspgen', 'test')
            self.assertEqual(len(pool), 1)
            for override in (
                    dict(source_kind='deterministic_axial_primitive_grid_v1 (no GraspGen inference)',
                         generation_calls=0, model_candidates=0),
                    dict(generation_calls=0), dict(model_candidates=0),
                    dict(source_kind='unknown source')):
                with self.subTest(override=override):
                    path.write_text(json.dumps({**report, **override}))
                    with self.assertRaisesRegex(ValueError, 'GraspGen model candidates'):
                        load_candidate_source(root, 'graspgen', 'test')

    def test_curved_rim_contact_has_lower_alignment_than_center_contact(self):
        center = np.diag([1., -1., -1., 1.])
        center[2,3] = .004
        rim = center.copy()
        rim[0,3] = .017
        good = pad_normal_quality('cylinder36h52', center)
        poor = pad_normal_quality('cylinder36h52', rim)
        self.assertTrue(good['preferred'])
        self.assertFalse(poor['preferred'])
        self.assertGreater(good['minimum_side_q25_normal_alignment'],
                           poor['minimum_side_q25_normal_alignment'])
        np.testing.assert_array_equal(center[:2,3], [0.,0.])


if __name__ == '__main__':
    unittest.main()
