"""Public entry points must expose help without installed GPU environments."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_cli(self, *arguments):
        env = dict(os.environ)
        for key in ("SIM_PYTHON", "PLANNER_PYTHON", "GRASPGEN_PYTHON", "GRASPGEN_CONFIG"):
            env[key] = str(ROOT / "not-installed" / key)
        return subprocess.run([sys.executable, "-B", str(ROOT / "cli.py"), *arguments],
                              env=env, text=True, capture_output=True, timeout=20)

    def test_help_lists_commands_and_all_commands_work_without_gpu_environments(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        overview = result.stdout
        for command in ("collect", "prepare", "check", "export", "pipeline", "download-models", "benchmark"):
            self.assertIn(command, overview)
            with self.subTest(command=command):
                result = self.run_cli(command, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_typo_fails_and_legacy_collection_flags_still_work(self):
        result = self.run_cli("colect")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown command", result.stderr)
        result = self.run_cli("--dry-run", "--objects", "block55")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["objects"], ["block55"])


if __name__ == "__main__":
    unittest.main()
