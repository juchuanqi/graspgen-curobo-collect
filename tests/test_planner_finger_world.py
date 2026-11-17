"""GPU regressions for finger collision constraints used by the actual planner.

Opt in with RUN_CUROBO_GPU_TESTS=1 in the planning environment.
"""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime/collector'), str(ROOT / 'runtime'), str(ROOT)]


@unittest.skipUnless(os.environ.get('RUN_CUROBO_GPU_TESTS') == '1', 'explicit GPU test opt-in required')
class PlannerFingerWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from common import read
        from generic_planner import Planner, SPEED_PROFILE, pose_list
        from geometry.robot import fk

        torch.set_num_threads(2)
        torch.manual_seed(41)
        cls.directory = tempfile.TemporaryDirectory()
        cls.q = np.asarray(read(ROOT / 'runtime/config/scene.json')['initial_qpos'])
        cls.scene = dict(object_id='cylinder36h52',
            urdf_path=str(ROOT / 'runtime/robot/panda_v2.urdf'),
            joint_names=[f'panda_joint{i}' for i in range(1, 8)] + ['panda_finger_joint1', 'panda_finger_joint2'],
            qpos=cls.q.tolist(), tcp_pose=pose_list(fk(cls.q)['panda_hand_tcp']),
            base_pose=[0, 0, 0, 1, 0, 0, 0],
            table_pose=[0, 0, -3, 1, 0, 0, 0], table_shape_local_pose=[0, 0, 0, 1, 0, 0, 0],
            table_shape_half_size=[.6, .6, .025], object_pose=[3, 3, 3, 1, 0, 0, 0],
            object_parts=[dict(shape='box', size=[.02, .02, .02], position=[0, 0, 0], rpy=[0, 0, 0])],
            obstacles=[], arm_speed_profile=SPEED_PROFILE.describe())
        cls.planner = Planner(cls.scene, Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.planner.update(copy.deepcopy(self.scene), Path(self.directory.name))

    def finger_wall(self, finger):
        import torch
        state = self.planner.motion_gen.kinematics.get_state(
            torch.tensor(self.q[None, :7], dtype=torch.float32, device='cuda'))
        indices = self.planner.motion_gen.kinematics.kinematics_config.get_sphere_index_from_link_name(finger)
        center = state.link_spheres_tensor[0, indices[0], :3].cpu().numpy()
        return dict(name='wall', dims=[.004, .004, .004], pose=[*center.tolist(), 1, 0, 0, 0])

    def start_valid(self):
        import torch
        from curobo.types.robot import JointState
        # cuRobo's diagnostic may disable self collision to identify a world failure.
        self.planner.motion_gen.rollout_fn.robot_self_collision_constraint.enable_cost()
        return self.planner.motion_gen.check_start_state(JointState.from_position(
            torch.tensor(self.q[None, :7], dtype=torch.float32, device='cuda'),
            joint_names=self.planner.joint_names))

    def test_each_finger_wall_collision_is_rejected_during_planning(self):
        from curobo.wrap.reacher.motion_gen import MotionGenStatus
        from generic_planner import FINGERS

        self.assertTrue(self.start_valid()[0])
        for finger in FINGERS:
            with self.subTest(finger=finger):
                scene = copy.deepcopy(self.scene)
                scene['obstacles'] = [self.finger_wall(finger)]
                self.planner.update(scene, Path(self.directory.name))
                valid, status = self.start_valid()
                self.assertFalse(valid)
                self.assertEqual(status, MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION)
                self.planner.update(copy.deepcopy(self.scene), Path(self.directory.name))
                self.assertTrue(self.start_valid()[0])

    def test_width_update_and_self_check_keep_finger_world_constraints(self):
        from generic_planner import FINGERS

        for width in (.06, .06, .08):
            self.planner.set_width(width)
            self.planner.self_valid(self.q[None, :7], width)
            for finger in FINGERS:
                with self.subTest(width=width, finger=finger):
                    scene = copy.deepcopy(self.scene)
                    scene['qpos'][7:] = [width / 2, width / 2]
                    scene['obstacles'] = [self.finger_wall(finger)]
                    self.planner.update(scene, Path(self.directory.name))
                    self.planner.self_valid(self.q[None, :7], width)
                    self.assertFalse(self.start_valid()[0])

    def test_grasped_target_removal_retains_finger_wall_constraints(self):
        from curobo.wrap.reacher.motion_gen import MotionGenStatus

        scene = copy.deepcopy(self.scene)
        scene['obstacles'] = [self.finger_wall('panda_leftfinger')]
        self.planner.update(scene, Path(self.directory.name), remove_object=True)
        self.assertIn('wall', self.planner.world['cuboid'])
        self.assertFalse(any(name.startswith('target_') for name in self.planner.world['cuboid']))
        self.assertEqual(self.start_valid(), (False, MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION))

    def test_target_is_not_exempted_before_grasp(self):
        scene = copy.deepcopy(self.scene)
        scene['object_pose'] = self.finger_wall('panda_leftfinger')['pose']
        scene['object_parts'][0]['size'] = [.004, .004, .004]
        self.planner.update(scene, Path(self.directory.name))
        self.assertFalse(self.start_valid()[0])
        self.planner.update(scene, Path(self.directory.name), remove_object=True)
        self.assertTrue(self.start_valid()[0])


if __name__ == '__main__':
    unittest.main()
