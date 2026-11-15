"""Ensure a live checkout edit cannot change a running pipeline's source."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from pipeline import freeze_source
from common import sha


class PipelineSnapshotTests(unittest.TestCase):
    def test_subprocess_uses_captured_source_after_checkout_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'checkout';root.mkdir()
            (root/'runtime').mkdir();(root/'tools').mkdir()
            (root/'common.py').write_text('VALUE = "captured"\n')
            (root/'runner.py').write_text('import common\nprint(common.VALUE)\n')
            (root/'local_env.json').write_text('{}')
            (root/'runtime/physics.json').write_text('{"hz":480}')
            (root/'tools/stage.py').write_text('print("stage")\n')
            snapshot=freeze_source(root,root/'runtime',base/'snapshot')
            (root/'common.py').write_text('VALUE = "changed during run"\n')
            (root/'runtime/physics.json').write_text('{"hz":1}')
            result=subprocess.check_output([sys.executable,'-B',str(snapshot/'runner.py')],text=True)
            self.assertEqual(result.strip(),'captured')
            self.assertEqual(json.loads((snapshot/'runtime/physics.json').read_text())['hz'],480)
            manifest=json.loads((snapshot/'SOURCE.json').read_text())['sha256']
            self.assertTrue(all(sha(snapshot/name)==expected for name,expected in manifest.items()))
