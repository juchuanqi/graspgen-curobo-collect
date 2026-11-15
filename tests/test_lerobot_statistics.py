"""Regression checks for the uint8 overflow and float32 statistics defects."""
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/lerobot_export'))
try:
    if version('lerobot') != '0.4.4':
        raise unittest.SkipTest('Requires the separate LeRobot 0.4.4 environment')
    from stable_statistics import safe_stats, stable_episode_stats, stable_statistics
    from lerobot.datasets import lerobot_dataset
except (ModuleNotFoundError, PackageNotFoundError) as error:
    raise unittest.SkipTest('Requires the separate LeRobot 0.4.4 environment') from error


class LeRobotStatisticsTests(unittest.TestCase):
    def test_uint8_image_variance_does_not_overflow(self):
        images = np.zeros((2, 3, 2, 2), dtype=np.uint8)
        images[1] = 255
        result = safe_stats(images, image=True)
        np.testing.assert_allclose(result['mean'], .5, rtol=0, atol=0)
        np.testing.assert_allclose(result['std'], .5, rtol=0, atol=0)
        np.testing.assert_array_equal(result['count'], [2])
        self.assertEqual(result['std'].shape, (3, 1, 1))

    def test_small_variation_around_large_joint_offset(self):
        values = np.array([[10000.], [10000.01], [9999.99]], dtype=np.float32)
        result = safe_stats(values)
        np.testing.assert_allclose(result['std'], np.std(values.astype(np.float64), axis=0),
                                   rtol=0, atol=1e-14)
        self.assertGreater(float(result['std'][0]), .005)

    def test_timestamp_stats_describe_stored_float32_values(self):
        values = np.arange(100) / 60.
        result = stable_episode_stats({'timestamp': values}, {'timestamp': {'dtype': 'float32'}})
        np.testing.assert_array_equal(result['timestamp']['mean'],
                                      [values.astype(np.float32).astype(np.float64).mean()])

    def test_boolean_bounds_keep_arrow_type(self):
        result = safe_stats(np.array([[False], [True]], dtype=bool))
        self.assertEqual(result['min'].dtype, np.dtype(bool))
        self.assertEqual(result['max'].dtype, np.dtype(bool))

    def test_process_hook_is_restored_after_failure(self):
        original = lerobot_dataset.compute_episode_stats
        with self.assertRaisesRegex(RuntimeError, 'write failed'):
            with stable_statistics():
                raise RuntimeError('write failed')
        self.assertIs(lerobot_dataset.compute_episode_stats, original)
