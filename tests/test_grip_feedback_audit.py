"""Detect feedback chronology corruption and hidden actuator target changes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from runtime.collector.grip_control import GripController
from tools.audit_grip_feedback import audit


class FeedbackAuditTests(unittest.TestCase):
    def make_episode(self,path):
        source=ROOT/'runtime/collector/grip_control.py'
        snapshot=path/'source_snapshot';snapshot.mkdir();shutil.copy2(source,snapshot/source.name)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        controller=GripController(.08)
        meta=dict(case_id='test',grip_policy=controller.describe(),source_sha256={str(source):digest})
        (path/'METADATA.json').write_text(json.dumps(meta))
        forces=np.array([[0,0],[0,0],[1,1],[3,3],[5,5],[2,2]],dtype=np.float32)
        phases=np.array([0,2,2,3,4]);actions=np.zeros((5,9),dtype=np.float32)
        for t,phase in enumerate(phases):
            actions[t,7:]=controller.command(forces[t]) if phase>=2 else .04
        return dict(phase=phases,pad_object_force_n=forces,action_targets=actions,
                    position_targets=np.vstack([np.zeros((1,9),np.float32),actions]))

    def test_saved_controller_and_previous_force_reconstruct_exact_commands(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);data=self.make_episode(p);np.savez_compressed(p/'trajectory.npz',**data)
            result=audit(p)
            self.assertEqual(result['controls'],4)
            self.assertTrue(result['preceding_observations_used'])
            data['pad_object_force_n']=np.roll(data['pad_object_force_n'],-1,axis=0)
            np.savez_compressed(p/'trajectory.npz',**data)
            with self.assertRaises(AssertionError):audit(p)

    def test_actual_target_changes_and_source_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);data=self.make_episode(p)
            data['position_targets'][3,7]+=.001
            np.savez_compressed(p/'trajectory.npz',**data)
            with self.assertRaises(AssertionError):audit(p)
            data['position_targets'][1:]=data['action_targets']
            np.savez_compressed(p/'trajectory.npz',**data)
            with (p/'source_snapshot/grip_control.py').open('a') as file:file.write('\n# changed\n')
            with self.assertRaisesRegex(AssertionError,'hash differs'):audit(p)


if __name__=='__main__':unittest.main()
