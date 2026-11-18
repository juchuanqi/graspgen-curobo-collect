"""Planning margins must survive exact sweeps and adaptive endpoint checks."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime'), str(ROOT / 'runtime/collector')]
from common import read
from geometry.api import check_gripper_path, check_robot_path
from geometry.robot import collision_bodies, gripper_transforms, fk
from geometry.primitives import primitive_convex
from planning_clearance import geometry_kwargs


class ExecutionClearanceTests(unittest.TestCase):
    def test_submillimeter_table_gap_is_valid_measured_geometry_but_not_a_plan(self):
        pose = np.eye(4)
        bottom = min(b.vertices[:, 2].min() for b, _ in collision_bodies(gripper_transforms(pose, .08)).values())
        pose[2, 3] = -bottom + .0002
        args = (pose, .08, [3, 3, 3, 1, 0, 0, 0], 'block55')
        self.assertTrue(check_gripper_path(*args)['valid'])
        report = check_gripper_path(*args, **geometry_kwargs())
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'robot_table_collision')
        self.assertEqual(report['table_clearance_m'], .001)

    def test_four_mm_wall_gap_is_rejected_after_adaptive_full_arm_refinement(self):
        q = np.asarray(read(ROOT / 'runtime/config/scene.json')['initial_qpos'])
        right = max(b.vertices[:, 0].max() for b, _ in collision_bodies(fk(q)).values())
        wall = dict(name='wall', dims=[.002, 10., 10.], pose=[right + .005, 0, 0, 1, 0, 0, 0])
        args = (np.stack((q, q)), float(q[7:].sum()), [3, 3, 3, 1, 0, 0, 0], 'block55')
        self.assertTrue(check_robot_path(*args, table=False, obstacles=[wall])['valid'])
        report = check_robot_path(*args, table=False, obstacles=[wall], **geometry_kwargs())
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'robot_obstacle_collision')
        self.assertGreater(report['refinement']['endpoint_violations'], 0)
        self.assertEqual(report['obstacle_clearance_m'], .006)

    def test_margin_cannot_relax_the_existing_geometry_floor(self):
        for value in (0., -.001, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                check_gripper_path(np.eye(4), .08, [3, 3, 3, 1, 0, 0, 0], 'block55',
                                   table=False, obstacle_clearance_m=value)

    def test_carried_target_keeps_wall_margin_when_robot_is_far_from_wall(self):
        q = np.asarray(read(ROOT / 'runtime/config/scene.json')['initial_qpos'])
        attachment = np.eye(4)
        attachment[0, 3] = 2.
        held = fk(q)['panda_hand_tcp'] @ attachment
        part = read(ROOT / 'runtime/assets/MANIFEST.json')['objects']['block55']['parts'][0]
        body = primitive_convex(part).moved(held)
        center = body.vertices.mean(axis=0)
        center[0] = body.vertices[:, 0].max() + .005
        wall = dict(name='wall', dims=[.002, .2, .2], pose=[*center, 1, 0, 0, 0])
        args = (np.stack((q, q)), float(q[7:].sum()), held, 'block55')
        kwargs = dict(table=False, obstacles=[wall], allow_pad_contact=True,
                      attached_to_tcp=attachment, check_target_table=False)
        self.assertTrue(check_robot_path(*args, **kwargs)['valid'])
        report = check_robot_path(*args, **kwargs, **geometry_kwargs())
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'object_obstacle_collision')


if __name__ == '__main__':
    unittest.main()
