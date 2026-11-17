"""A geometric pool has a fixed budget, correct frame, and explicit domain."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'runtime')]
from tools.generate_axial_candidates import axial_grid
from geometry.robot import GRASPGEN_TO_TCP
from geometry.api import pad_normal_quality


class AxialCandidateTests(unittest.TestCase):
    def test_budget_and_contact_frame_scale_with_asset_height(self):
        for height in (.024, .052):
            poses, params = axial_grid({'parts':[dict(shape='cylinder', length=height)]})
            self.assertEqual(poses.shape, (200,4,4))
            self.assertEqual(len(params), 200)
            tcps = poses @ GRASPGEN_TO_TCP
            np.testing.assert_allclose(tcps[:, :2, 3], 0., atol=1e-15)
            np.testing.assert_allclose(tcps[:, 2, 3], [p['height_m'] for p in params], atol=1e-15)
            np.testing.assert_allclose(np.linalg.det(tcps[:, :3, :3]), 1., atol=1e-14)
            self.assertTrue((tcps[:, 2, 2] < -.86).all())

    def test_nonaxial_or_shifted_geometry_is_rejected(self):
        for parts in ([], [dict(shape='box')], [dict(shape='cylinder',position=[.01,0,0])],
                      [dict(shape='cylinder',rpy=[.1,0,0])]):
            with self.assertRaises(ValueError): axial_grid({'parts':parts})

    def test_curved_rim_contact_has_lower_alignment_than_center_contact(self):
        center = np.diag([1., -1., -1., 1.])
        center[2,3] = .004
        rim = center.copy(); rim[0,3] = .017
        good = pad_normal_quality('cylinder36h52', center)
        poor = pad_normal_quality('cylinder36h52', rim)
        self.assertTrue(good['preferred'])
        self.assertFalse(poor['preferred'])
        self.assertGreater(good['minimum_side_q25_normal_alignment'],
                           poor['minimum_side_q25_normal_alignment'])
        np.testing.assert_array_equal(center[:2,3], [0.,0.])



if __name__ == '__main__': unittest.main()
