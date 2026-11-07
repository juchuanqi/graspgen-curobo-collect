"""Compare all snapshot fields at the same physical tick without advancing it."""
from collections import Counter
import numpy as np


class LiveIOAudit:
    def __init__(self):
        self.counts = Counter()
        self.maximum_float_error = 0.

    def compare(self, env, row, images):
        from collection_env import PickupEnv as BaselinePickupEnv
        reference, _ = BaselinePickupEnv.snapshot(env, capture_rgb=False)
        assert row.keys() == reference.keys(), (row.keys()-reference.keys(), reference.keys()-row.keys())
        for key, value in row.items():
            other = reference[key]
            assert value.shape == other.shape and value.dtype == other.dtype, key
            if value.dtype.kind in 'biu':
                np.testing.assert_array_equal(value, other, err_msg=key)
            else:
                np.testing.assert_array_equal(np.isfinite(value), np.isfinite(other), err_msg=key)
                np.testing.assert_allclose(value, other, rtol=1e-6, atol=2e-7, equal_nan=True, err_msg=key)
                finite = np.isfinite(value)
                if finite.any():
                    self.maximum_float_error = max(self.maximum_float_error,
                        float(np.max(np.abs(value[finite]-other[finite]))))
        for name, image in images.items():
            expected = env.cameras[name].data.output['rgb'][0, ..., :3].cpu().numpy()
            np.testing.assert_array_equal(image, expected, err_msg=name)
        self.counts['same_tick_snapshots_checked'] += 1
        self.counts['fields_checked'] += len(row)
        self.counts['rgb_images_checked'] += len(images)
        self.counts['nonzero_raw_contact_points'] += int(row['raw_finger_contact_valid'].sum())
        self.counts['bilateral_pad_contact_observations'] += int((row['pad_object_force_n'] >= .5).all())
        assert int(row['contact_substep_sample_count']) == 8
        assert np.array_equal(np.diff(row['contact_substep_physics_tick']), np.ones(7, dtype=np.int64))

    def report(self):
        return dict(status='passed', counts=dict(self.counts),
                    maximum_float_absolute_error=self.maximum_float_error,
                    scope='Optimized snapshot versus old getters at the same physical tick; no extra physics. CPU analytic old/new comparison separately covers all8 raw-contact samples, saturation and contact tails.')
