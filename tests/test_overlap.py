"""Overlapped planning must drain RPCs and preserve the cumulative budget."""
from pathlib import Path
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime/collector'))
from overlap import run_overlap


class OverlapTests(unittest.TestCase):
    def test_over_budget_reply_is_rejected_after_both_tasks_finish(self):
        started = threading.Event()
        release = threading.Event()
        def rpc():
            started.set()
            self.assertTrue(release.wait(2))
            return dict(success=True, wall_s=4.)
        def geometry():
            self.assertTrue(started.wait(2))
            release.set()
            return dict(valid=True)
        checked, reply, events = run_overlap(rpc, geometry, 3.)
        self.assertTrue(checked['valid'])
        self.assertFalse(reply['success'])
        self.assertTrue(reply['speculative_reply_rejected'])
        self.assertTrue(events['both_completed_before_execution'])

    def test_geometry_exception_still_drains_rpc(self):
        finished = threading.Event()
        def rpc():
            time.sleep(.03)
            finished.set()
            return dict(success=True, wall_s=.03)
        def geometry():
            raise ValueError('invalid closure')
        with self.assertRaisesRegex(ValueError, 'invalid closure'):
            run_overlap(rpc, geometry, 20.)
        self.assertTrue(finished.is_set())


if __name__ == '__main__':
    unittest.main()
