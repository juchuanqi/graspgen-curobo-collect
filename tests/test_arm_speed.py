"""CPU checks for path timing and independently replayable speed limits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime.collector import arm_speed


class ArmSpeedTests(unittest.TestCase):
    def test_retime_endpoints_velocity_and_degenerate_paths(self):
        # Unequal segments, corners and repeated knots; limits must hold on
        # output targets, not just on the internal scalar progress variable.
        path = np.array([[0, 0], [.2, 0], [.2, 0], [.2, -.3], [-.1, -.3]])
        for velocity, acceleration in ((1., 5.), (.3, 2.), (.01, 100.)):
            timed = arm_speed.retime(path, velocity, acceleration)
            np.testing.assert_allclose(timed[[0, -1]], path[[0, -1]], atol=1e-8)
            self.assertLessEqual(np.abs(np.diff(timed, axis=0)).max(), velocity / 60 + 1e-7)
        np.testing.assert_array_equal(arm_speed.retime(np.zeros((3, 7)), 1., 5.), np.zeros((1, 7)))
        self.assertEqual(arm_speed.retime(np.ones((1, 7)), 1., 5.).shape, (1, 7))

    def test_invalid_rates_are_rejected(self):
        for value in ('0,5,.3,2', '-1,5,.3,2', 'nan,5,.3,2', '1,inf,.3,2', '1,5,.3', '3,5,.3,2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                arm_speed.parse(value)

    def test_saved_profile_is_independent_of_environment(self):
        fast = arm_speed.parse('recommended')
        actions = np.full((1, 9), .01)
        previous = np.zeros((1, 9))
        with patch.dict(os.environ, {'ARM_SPEED_PROFILE': 'baseline'}):
            arm_speed.validate_action_deltas(actions, previous, [0], {'arm_speed_profile': fast.describe()})
        with patch.dict(os.environ, {'ARM_SPEED_PROFILE': 'recommended'}):
            # Legacy recordings without metadata retain the old strict bound.
            with self.assertRaises(AssertionError):
                arm_speed.validate_action_deltas(actions, previous, [0], {})
        self.assertEqual(arm_speed.from_metadata({'arm_speed_profile': fast.describe()}), fast)

    def test_contact_and_carry_phases_keep_separate_limits(self):
        meta = {'arm_speed_profile': arm_speed.parse('recommended').describe()}
        actions = np.full((1, 9), .01)
        for phase in (1, 2, 3, 4):
            with self.subTest(phase=phase), self.assertRaises(AssertionError):
                arm_speed.validate_action_deltas(actions, np.zeros((1, 9)), [phase], meta)
        with self.assertRaises(ValueError):
            arm_speed.from_metadata({'arm_speed_profile': {**meta['arm_speed_profile'], 'control_dt_s': .1}})

    def test_cli_profile_is_visible_and_overrides_environment(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'collect.py'), '--dry-run',
                                 '--arm-speed-profile', 'baseline'], env={**os.environ, 'ARM_SPEED_PROFILE': 'recommended'},
                                capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)['arm_speed_profile']['enabled'])


if __name__ == '__main__':
    unittest.main()
