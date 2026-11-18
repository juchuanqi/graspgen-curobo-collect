"""Reject the historically overlapping slender/wall schedule before startup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import collection_plan, read, write
from initial_geometry import preflight


class InitialGeometryTests(unittest.TestCase):
    def test_complete_batch_rejects_overlap_and_accepts_paired_corrected_starts(self):
        plan = collection_plan(['slender95'], 16, 1, 2026111700)
        with self.assertRaisesRegex(ValueError, 'object_obstacle_collision'):
            preflight(plan, ROOT / 'runtime', 'wall', sys.executable)
        self.assertEqual(preflight(plan, ROOT / 'runtime', 'none', sys.executable)['invalid_slots'], 0)
        with tempfile.TemporaryDirectory() as temp:
            runtime = Path(temp) / 'runtime'
            runtime.mkdir()
            for name in ('assets', 'geometry', 'collector', 'robot'):
                (runtime / name).symlink_to(ROOT / 'runtime' / name, target_is_directory=True)
            scene = read(ROOT / 'runtime/config/scene.json')
            scene['object_reference_xy_m'][0] += .020
            write(runtime / 'config/scene.json', scene)
            report = preflight(plan, runtime, 'wall', sys.executable)
            self.assertEqual(report['scheduled_slots'], 16)
            self.assertEqual(report['invalid_slots'], 0)
            self.assertEqual(len(report['cases']), 16)
            self.assertTrue(all(set(r['states']) == {'spawn', 'settled'} for r in report['cases']))

    def test_entrypoints_do_not_register_or_start_invalid_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            for entrypoint in ('collect.py', 'tools/pipeline.py'):
                with self.subTest(entrypoint=entrypoint):
                    output = Path(temp) / Path(entrypoint).stem
                    command = [sys.executable, '-B', str(ROOT / entrypoint), '--output', str(output),
                               '--objects', 'slender95', '--episodes-per-object', '16',
                               '--obstacle', 'wall', '--runtime', str(ROOT / 'runtime')]
                    result = subprocess.run(command, text=True, capture_output=True,
                                            env={**os.environ, 'PLANNER_PYTHON': sys.executable,
                                                 'SIM_PYTHON': sys.executable})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('Initial geometry preflight rejected the whole batch', result.stderr)
                    self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
