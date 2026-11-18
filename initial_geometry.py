"""CPU preflight boundary for stdlib collection and pipeline supervisors."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess


def preflight(plan, runtime, obstacle, planner_python):
    """Reject the entire schedule before registration; never filter its slots."""
    helper = Path(__file__).resolve().parent / 'tools/check_initial_geometry.py'
    result = subprocess.run(
        [str(planner_python), '-B', str(helper), '--runtime', str(Path(runtime).resolve()),
         '--obstacle', obstacle], input=json.dumps(plan, allow_nan=False),
        text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError('Initial geometry helper failed before registration: ' + result.stderr.strip())
    report = json.loads(result.stdout)
    if report['status'] != 'passed':
        rejected = [row for row in report['cases'] if not row['valid']]
        raise ValueError('Initial geometry preflight rejected the whole batch; no slots registered: '
                         + json.dumps(rejected, ensure_ascii=False, allow_nan=False))
    return report
