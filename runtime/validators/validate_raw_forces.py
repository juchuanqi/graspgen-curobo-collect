"""Independent CPU sums of every saved contact sample and threshold decision."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('episode', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.episode/'trajectory.npz', allow_pickle=False) as data:
        force = data['contact_substep_normal_force_n']
        valid = data['contact_substep_valid']
        pad = data['contact_substep_pad_mask']
        separation = data['contact_substep_separation_m']
        assert force.shape == valid.shape == pad.shape == separation.shape
        assert force.shape[1:] == (8, 2, 128)
        assert np.all(~pad | valid)
        # Variable-length reference sums match the old classification consumer.
        pad_sums = np.zeros(force.shape[:-1], dtype=np.float32)
        nonpad_sums = np.zeros_like(pad_sums)
        depths = np.zeros(force.shape[:2], dtype=np.float32)
        for row in range(len(force)):
            for tick in range(8):
                for finger in range(2):
                    f = force[row, tick, finger]
                    p = pad[row, tick, finger]
                    v = valid[row, tick, finger]
                    pad_sums[row, tick, finger] = f[p].sum(dtype=np.float32)
                    nonpad_sums[row, tick, finger] = f[v & ~p].sum(dtype=np.float32)
                    if v.any():
                        depths[row, tick] = max(depths[row, tick],
                            float(np.maximum(-separation[row, tick, finger, v], 0).max()))
        saved_pad = data['contact_substep_pad_force_n']
        np.testing.assert_allclose(saved_pad, pad_sums, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(data['pad_object_force_n'], pad_sums[:, -1], rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(data['nonpad_object_peak_n'], nonpad_sums.max(axis=1), rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(data['max_contact_penetration_m'], depths.max(axis=1), rtol=0, atol=1e-8)
        np.testing.assert_array_equal(saved_pad >= .5, pad_sums >= .5)
        # The same material-derived multiplier is present in each observation.
        saved_nonpad = data['nonpad_object_peak_n']
        saved_bound = data['nonpad_object_force_bound_n']
        multiplier = np.divide(saved_bound, saved_nonpad, out=np.ones_like(saved_bound), where=saved_nonpad != 0)
        reconstructed_bound = nonpad_sums.max(axis=1)*multiplier
        np.testing.assert_array_equal(saved_bound > 1., reconstructed_bound > 1.)
        np.testing.assert_array_equal(data['max_contact_penetration_m'] > .001, depths.max(axis=1) > .001)
        result = dict(status='passed', observations=len(force), physics_samples=len(force)*8,
            valid_finger_contact_points=int(valid.sum()), pad_contact_points=int(pad.sum()),
            nonpad_contact_points=int((valid & ~pad).sum()),
            maximum_pad_force_sum_difference_n=float(np.abs(saved_pad-pad_sums).max()),
            pad_half_newton_nonpad_one_newton_depth_one_mm_decisions_equal=True,
            scope='CPU variable-length reference sums for every recorded8-substep raw force; flags and actual execution scoring separately replayed by validate_episode.py')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
