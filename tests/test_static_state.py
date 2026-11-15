"""Analytic pre-planning evidence and collector stopping boundaries."""
from pathlib import Path
import copy
import sys
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/collector'))
from static_state import evaluate
from collection_score import CollectionScore
from test_linear_motion_score import lifted_rows, row


def trace():
    samples = np.tile([.6, 0, .03, 1, 0, 0, 0], (31, 8, 1)).astype(float)
    return dict(object_pose=samples[:, -1].copy(), contact_substep_object_pose_w=samples,
                contact_substep_physics_tick=np.arange(248).reshape(31, 8),
                contact_substep_sample_count=np.full(31, 8), link_contact_peak_n=np.zeros((31, 10, 3)))


def audit(data):
    return evaluate(data, [-.03, -.02, -.01], [.03, .02, .01])


class StaticTests(unittest.TestCase):
    def test_stationary_passes(self):
        result = audit(trace())
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['physical_intervals'], 240)
        self.assertEqual(result['surface_path_bound_m'], 0.)

    def test_cumulative_translation_rotation_and_cancellation_rejected(self):
        for mode in ('translation', 'rotation', 'back_forth'):
            with self.subTest(mode=mode):
                data = trace(); samples = data['contact_substep_object_pose_w']
                if mode == 'translation':
                    samples[..., 0] += np.linspace(0, .001, 248).reshape(31, 8)
                elif mode == 'back_forth':
                    samples[..., 0] += np.tile([0, .00001], 124).reshape(31, 8)
                else:
                    q = Rotation.from_rotvec(np.column_stack([np.zeros(248), np.zeros(248), np.linspace(0, .05, 248)])).as_quat()
                    samples[..., 3:] = q[:, [3, 0, 1, 2]].reshape(31, 8, 4)
                data['object_pose'] = samples[:, -1].copy()
                self.assertEqual(audit(data)['status'], 'ineligible')

    def test_prior_robot_contact_rejected(self):
        data = trace(); data['link_contact_peak_n'][0, 9, 0] = .01
        self.assertEqual(audit(data)['status'], 'ineligible')

    def test_malformed_trace_rejected(self):
        for key in trace():
            data = trace(); data[key] = data[key][:-1]
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit(data)
        data = trace(); data['contact_substep_physics_tick'][3, 2] += 1
        with self.assertRaises(ValueError): audit(data)
        data = trace(); data['contact_substep_object_pose_w'][1, 2, 3] = 2
        with self.assertRaises(ValueError): audit(data)
        data = trace(); data['object_pose'][2, 0] += .01
        with self.assertRaises(ValueError): audit(data)

    def test_collection_full_window_and_terminal_freeze(self):
        values = iter(lifted_rows()); score = CollectionScore(next(values), com_local_m=[0, 0, 0])
        initial_keys = set(score.derive(row()))
        for step in range(1, 122):
            derived = score.update(next(values))
            self.assertEqual(set(derived), initial_keys)
            self.assertEqual(score.success, step == 121)
        self.assertEqual(score.first_success_step, 121)
        with self.assertRaises(RuntimeError): score.update(next(values))

    def test_collection_invalid_motion_is_terminal_result(self):
        score = CollectionScore(row(), com_local_m=[0, 0, 0])
        value = row(step=1); value.pop('contact_substep_object_pose_w')
        derived = score.update(value)
        self.assertFalse(derived['motion_measurement_valid'])
        self.assertEqual(score.reason, 'invalid_motion_observation')
