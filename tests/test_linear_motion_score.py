"""Analytic motion and acceptance boundary tests for the isolated candidate."""
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime/collector"))
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from linear_motion_score import LinearMotionScore, MotionInputError, interval_motion


def row(pose=None, step=0, previous_pose=None):
    pose = np.array([.6, 0., .03, 1., 0., 0., 0.] if pose is None else pose, float)
    previous = pose if previous_pose is None else previous_pose
    samples = np.tile(pose, (8, 1))
    samples[:, :3] = np.linspace(previous[:3], pose[:3], 9)[1:]
    return dict(object_pose=pose, tcp_pose=np.array([.6, 0., .15, 1., 0., 0., 0.]),
        qpos=np.r_[np.zeros(7), .02, .02], qvel=np.zeros(9), position_targets=np.zeros(9),
        object_linear_velocity=np.zeros(3), object_angular_velocity=np.zeros(3),
        link_contact_force=np.zeros((10, 3, 3)), link_contact_peak_n=np.zeros((10, 3)),
        object_table_peak_n=np.array(0.), object_wall_peak_n=np.array(0.),
        pad_object_force_n=np.array([1., 1.]), nonpad_object_peak_n=np.zeros(2),
        nonpad_object_force_bound_n=np.zeros(2), max_contact_penetration_m=np.array(0.),
        finger_contact_pos_w=np.zeros((2, 3)), contact_substep_object_pose_w=samples,
        contact_substep_physics_tick=np.arange(step * 8, (step + 1) * 8, dtype=np.int64),
        contact_substep_sample_count=np.array(8), sim_time=np.array(step / 60))


def lifted_rows(steps=150, speed=0., native=.04):
    initial = row()
    previous = initial['object_pose']
    yield initial
    for i in range(1, steps + 1):
        pose = np.array([.6 + max(i - 1, 0) * speed / 60, 0., .15, 1., 0., 0., 0.])
        value = row(pose, i, previous)
        value['object_linear_velocity'][2] = native
        previous = pose
        yield value


class MotionTests(unittest.TestCase):
    def test_stationary_native_residual_accepts_after_full_window(self):
        stream = iter(lifted_rows())
        score = LinearMotionScore(next(stream), [0., 0., 0.])
        for value in stream:
            d = score.update(value)
            if score.reason:
                break
        self.assertTrue(score.success)
        self.assertEqual(score.first_success_step, 121)
        self.assertEqual(d['pose_interval_linear_speed_m_s'], 0.)
        self.assertEqual(d['native_linear_speed_m_s'], .04)

    def test_known_translation(self):
        a = row(); b = row(a['object_pose'] + [.003, -.004, 0, 0, 0, 0, 0], 1, a['object_pose'])
        d = interval_motion(a['object_pose'], 7, 0., b, [.02, -.01, .03])
        self.assertAlmostEqual(d['pose_interval_linear_speed_m_s'], .3, places=12)

    def test_offset_rotation_with_stationary_com(self):
        a = row(); com = np.array([.03, -.01, .02])
        rotations = Rotation.from_euler('z', np.linspace(0., .1, 9)[:, None])
        poses = np.zeros((9, 7)); poses[:, 3:] = rotations.as_quat()[:, [3, 0, 1, 2]]
        poses[:, :3] = a['object_pose'][:3] + com - rotations.apply(np.tile(com, (9, 1)))
        b = row(poses[-1], 1); b['contact_substep_object_pose_w'] = poses[1:]
        d = interval_motion(poses[0], 7, 0., b, com)
        self.assertLess(d['pose_interval_linear_speed_m_s'], 1e-12)
        self.assertAlmostEqual(d['pose_interval_angular_speed_rad_s'], 6., places=12)

    def test_rotating_offset_com_arc(self):
        a = row(); rotations = Rotation.from_euler('z', np.linspace(0., .1, 9)[:, None])
        b = row(a['object_pose'], 1); b['contact_substep_object_pose_w'][:, 3:] = rotations.as_quat()[1:, [3, 0, 1, 2]]
        b['object_pose'] = b['contact_substep_object_pose_w'][-1].copy()
        d = interval_motion(a['object_pose'], 7, 0., b, [.03, 0, 0])
        self.assertAlmostEqual(d['pose_interval_linear_speed_m_s'], 8 * 2 * .03 * np.sin(.1 / 16) * 60, places=12)

    def test_back_and_forth_is_not_cancelled(self):
        a = row(); b = row(a['object_pose'], 1)
        b['contact_substep_object_pose_w'][:, 0] += [.001, 0., .001, 0., .001, 0., .001, 0.]
        d = interval_motion(a['object_pose'], 7, 0., b, [0, 0, 0])
        self.assertAlmostEqual(d['pose_interval_linear_speed_m_s'], .48)

    def test_fast_real_slip_rejected_despite_zero_native(self):
        values = iter(lifted_rows(speed=.03, native=0.))
        score = LinearMotionScore(next(values), [0, 0, 0])
        for value in values:
            score.update(value)
        self.assertFalse(score.success)
        self.assertEqual(score.best_window, 0)
        self.assertIn('linear_motion_too_fast', score.last_blockers)

    def test_slow_real_slip_rejected_by_relative_drift(self):
        values = iter(lifted_rows(speed=.005, native=0.))
        score = LinearMotionScore(next(values), [0, 0, 0])
        for value in values:
            score.update(value)
        self.assertFalse(score.success)
        self.assertIn('relative_translation_drift', score.last_blockers)

    def test_nonfinite_native_is_terminal(self):
        score = LinearMotionScore(row(), [0, 0, 0]); b = row(step=1)
        b['object_linear_velocity'][0] = np.nan
        score.update(b)
        self.assertEqual(score.reason, 'nonfinite_or_missing_state')

    def test_invalid_intervals_fail_closed(self):
        mutations = {
            'missing': lambda b: b.pop('contact_substep_object_pose_w'),
            'short': lambda b: b.update(contact_substep_object_pose_w=np.zeros((7, 7))),
            'count': lambda b: b.update(contact_substep_sample_count=7),
            'fractional_count': lambda b: b.update(contact_substep_sample_count=8.5),
            'ticks_gap': lambda b: b.update(contact_substep_physics_tick=np.arange(9, 17)),
            'ticks_duplicate': lambda b: b['contact_substep_physics_tick'].__setitem__(3, 10),
            'ticks_float': lambda b: b.update(contact_substep_physics_tick=np.arange(8., 16.)),
            'wrong_time': lambda b: b.update(sim_time=.02),
            'nonfinite_pose': lambda b: b['contact_substep_object_pose_w'].__setitem__((1, 0), np.nan),
            'bad_quaternion': lambda b: b['contact_substep_object_pose_w'].__setitem__((1, 3), 2.),
            'endpoint_position': lambda b: b['contact_substep_object_pose_w'].__setitem__((-1, 0), .601),
            'endpoint_rotation': lambda b: b['contact_substep_object_pose_w'].__setitem__((-1, slice(3, 7)), [np.cos(.1), 0, 0, np.sin(.1)]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                score = LinearMotionScore(row(), [0, 0, 0]); b = row(step=1); mutate(b)
                score.update(b)
                self.assertEqual(score.reason, 'invalid_motion_observation')
                self.assertFalse(score.success)

    def test_initial_com_and_interval_required(self):
        for com in (None, [0, 0], [0, np.nan, 0]):
            with self.assertRaises(MotionInputError):
                LinearMotionScore(row(), com)
        a = row(); a['contact_substep_sample_count'] = 0
        with self.assertRaises(MotionInputError):
            LinearMotionScore(a, [0, 0, 0])

    def test_safety_and_grasp_gates_retained(self):
        mutations = {
            'lift_below_60mm': lambda b: (b['object_pose'].__setitem__(2, .04), b['contact_substep_object_pose_w'][:, 2].fill(.04)),
            'bilateral_pad_contact_missing': lambda b: b['pad_object_force_n'].__setitem__(1, .49),
            'opening_out_of_range': lambda b: b['qpos'].__setitem__(slice(7, 9), [.04, .04]),
            'object_on_table': lambda b: b.update(object_table_peak_n=.51),
            'hard_failure': lambda b: b['link_contact_peak_n'].__setitem__((8, 1), 1.01),
        }
        for expected, mutate in mutations.items():
            with self.subTest(gate=expected):
                values = iter(lifted_rows()); score = LinearMotionScore(next(values), [0, 0, 0])
                for value in values:
                    mutate(value); score.update(value)
                    if score.reason:
                        break
                self.assertFalse(score.success)
                self.assertIn(expected, score.last_blockers)

    def test_angular_pose_gate_remains_active(self):
        values = iter(lifted_rows(native=0.)); initial = next(values)
        score = LinearMotionScore(initial, [0, 0, 0])
        for i, value in enumerate(values, 1):
            angles = np.linspace((i - 1) * .01, i * .01, 9)[1:]
            value['contact_substep_object_pose_w'][:, 3:] = Rotation.from_euler('z', angles[:, None]).as_quat()[:, [3, 0, 1, 2]]
            value['object_pose'][3:] = value['contact_substep_object_pose_w'][-1, 3:]
            score.update(value)
        self.assertFalse(score.success)
        self.assertIn('angular_pose_motion_too_fast', score.last_blockers)

    def test_penetration_and_fall_are_terminal(self):
        for condition in ('excess_contact_penetration', 'object_fell'):
            with self.subTest(condition=condition):
                score = LinearMotionScore(row(), [0, 0, 0]); b = row(step=1)
                if condition == 'excess_contact_penetration':
                    b['max_contact_penetration_m'] = .00101
                else:
                    b['object_pose'][2] = -.006
                    b['contact_substep_object_pose_w'][:, 2] = -.006
                score.update(b)
                self.assertEqual(score.reason, condition)

    def test_required_handle_and_hold_enable_are_not_bypassed(self):
        for task, enabled, expected in [('pick_up_outer_handle_bar', True, 'required_handle_contact_missing'),
                                        ('pick_up_any_stable_region', False, 'hold_disabled')]:
            values = iter(lifted_rows()); score = LinearMotionScore(next(values), [0, 0, 0], task=task)
            for value in values:
                score.update(value, hold_enabled=enabled)
            self.assertFalse(score.success)
            self.assertIn(expected, score.last_blockers)


if __name__ == '__main__':
    unittest.main(verbosity=2)
