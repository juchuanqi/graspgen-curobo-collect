"""CPU safety and feedback checks before expensive physical qualification."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.collector.grip_control import GripController, GripPolicy


class GripControlTests(unittest.TestCase):
    def test_fixed_native_spring_reaches_force_without_zero_target(self):
        controller = GripController(.08)
        target = .04
        for _ in range(300):
            # Two different contact widths model a slightly off-center grasp.
            force = np.maximum(0., (np.array([.028, .0275]) - target)*400)
            command = controller.command(force)
            self.assertEqual(command[0], command[1])
            self.assertLessEqual(abs(float(command[0])-target), .000451)
            target = float(command[0])
        self.assertTrue(controller.contact_seen)
        self.assertGreater(target, .015)
        self.assertLess(abs(float(force.mean())-3.), .31)

    def test_force_spike_and_contact_loss_stay_bounded(self):
        controller = GripController(.06)
        previous = .03
        for force in ([0.,0.], [2.,4.], [1000.,1000.], *([[0.,0.]]*300)):
            command = controller.command(force)
            self.assertTrue(np.isfinite(command).all())
            self.assertGreaterEqual(float(command.min()), 0.)
            self.assertLessEqual(float(command.max()), .0300001)
            self.assertLessEqual(abs(float(command[0])-previous), .000451)
            previous = float(command[0])
        self.assertEqual(previous, 0.)

    def test_bad_contact_data_cannot_produce_command(self):
        for force in ([np.nan,1], [np.inf,1], [-1,2], [1], [[1,2]]):
            with self.subTest(force=force), self.assertRaises(ValueError):
                GripController(.08).command(force)
        for opening in (0., -.1, .081, np.inf, np.nan):
            with self.assertRaises(ValueError): GripController(opening)
        for kwargs in ({'force_n':0}, {'filter_alpha':0}, {'gain_m_per_n':np.nan}):
            with self.assertRaises(ValueError): GripPolicy(**kwargs)


if __name__ == '__main__': unittest.main()
