"""Collision regressions: wall checks must cover fingers, paths and held objects."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'runtime'), str(ROOT / 'runtime/collector')]
from common import read
from geometry.api import check_gripper_path, check_robot_path
from geometry.robot import collision_bodies, fk, gripper_transforms
from obstacle_scene import planner_obstacles
from target_provider import screen_scene_candidates


def box(center, dims=(.004, .004, .004)):
    return dict(name='wall', dims=list(dims), pose=[*center, 1., 0., 0., 0.])


def distant_object():
    pose = np.eye(4)
    pose[:3, 3] = [3., 3., 3.]
    return pose


class ObstacleGeometryTests(unittest.TestCase):
    def test_allowed_target_pad_contact_does_not_allow_wall_contact(self):
        bodies = collision_bodies(gripper_transforms(np.eye(4), .08))
        pad = next(body for body, role in bodies.values() if role == 'pad')
        wall = box(pad.vertices.mean(axis=0))
        args = (np.eye(4), .08, distant_object(), 'block55')
        self.assertTrue(check_gripper_path(*args, table=False, allow_pad_contact=True)['valid'])
        report = check_gripper_path(*args, table=False, allow_pad_contact=True, obstacles=[wall])
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'robot_obstacle_collision')
        self.assertEqual(report['pair'][1], 'wall')

    def test_clear_gripper_endpoints_do_not_allow_a_sweep_through_wall(self):
        body = next(iter(collision_bodies(gripper_transforms(np.eye(4), .08)).values()))[0]
        wall = box(body.vertices.mean(axis=0))
        poses = np.repeat(np.eye(4)[None], 2, axis=0)
        poses[:, 0, 3] = [-.4, .4]
        for pose in poses:
            self.assertTrue(check_gripper_path(pose, .08, distant_object(), 'block55',
                                               table=False, obstacles=[wall])['valid'])
        report = check_gripper_path(poses, .08, distant_object(), 'block55', table=False, obstacles=[wall])
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'robot_obstacle_collision')

    def test_full_robot_and_attached_object_both_check_wall(self):
        q = np.asarray(read(ROOT / 'runtime/config/scene.json')['initial_qpos'])
        transforms = fk(q, .08)
        body = next(iter(collision_bodies(transforms).values()))[0]
        report = check_robot_path(q, .08, distant_object(), 'block55', table=False,
                                  obstacles=[box(body.vertices.mean(axis=0))])
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'robot_obstacle_collision')
        attachment = np.eye(4)
        attachment[:3, 3] = [2., 0., 0.]
        held = transforms['panda_hand_tcp'] @ attachment
        args = dict(table=False, check_target_table=False, allow_pad_contact=True,
                    attached_to_tcp=attachment)
        self.assertTrue(check_robot_path(q, .08, held, 'block55', **args)['valid'])
        report = check_robot_path(q, .08, held, 'block55', obstacles=[box(held[:3, 3])], **args)
        self.assertFalse(report['valid'])
        self.assertEqual(report['reason'], 'object_obstacle_collision')

    def test_candidate_filter_passes_scene_wall_to_real_geometry(self):
        from geometry.api import check_gripper_path as actual
        scene = dict(object_id='block55', object_pose=[3, 3, 3, 1, 0, 0, 0],
                     obstacles=[box([0., 0., -.05], (1., 1., 1.))])
        local = np.eye(4)
        local[:3, 3] = [-3., -3., -3.]
        candidate = dict(T_object_tcp=local.tolist(), original_rank=0)
        # Remove only the unrelated table for this isolated candidate fixture.
        def check(*args, **kwargs):
            return actual(*args, **kwargs, table=False)
        with patch('geometry.api.check_gripper_path', side_effect=check):
            valid, report = screen_scene_candidates([candidate], scene)
        self.assertEqual(valid, [])
        checks = report['rejections'][0]['scene_width_checks']
        self.assertTrue(all(c['reason'] == 'robot_obstacle_collision' for c in checks))

    @unittest.skipUnless(importlib.util.find_spec("curobo") is not None, "requires the cuRobo planning environment")
    def test_planner_world_and_final_path_check_keep_wall_during_carry(self):
        from generic_planner import Planner
        planner = Planner.__new__(Planner)
        planner.base = np.eye(4)
        planner.base_inverse = np.eye(4)
        planner.scene = dict(obstacles=planner_obstacles('wall'), qpos=[0.] * 7 + [.04, .04],
                             object_pose=[3, 3, 3, 1, 0, 0, 0], object_id='block55',
                             table_pose=[0, 0, 0, 1, 0, 0, 0],
                             table_shape_local_pose=[0, 0, 0, 1, 0, 0, 0],
                             table_shape_half_size=[.6, .6, .025], object_parts=[])
        np.testing.assert_allclose(planner.build_world(planner.scene, remove_object=True)['cuboid']['wall']['dims'],
                                   [.032, .312, .232])
        planner.self_valid = lambda *args: {'valid': True}
        with tempfile.TemporaryDirectory() as tmp:
            planner.output = Path(tmp)
            with patch('generic_planner.check_robot_path', return_value={'valid': False}) as check:
                result = planner.validate(np.zeros((1, 7)), .08, 'carry',
                                          attached_to_tcp=np.eye(4), allow_pad=True)
            self.assertFalse(result['valid'])
            self.assertEqual(check.call_args.kwargs['obstacles'], planner.scene['obstacles'])
            np.testing.assert_array_equal(check.call_args.kwargs['attached_to_tcp'], np.eye(4))

    def test_invalid_obstacle_dimensions_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            check_gripper_path(np.eye(4), .08, distant_object(), 'block55',
                               table=False, obstacles=[box([0., 0., 0.], dims=[0., 1., 1.])])
