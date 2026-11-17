"""Reconstruct recorded finger commands from preceding contact observations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def audit(directory):
    directory=Path(directory)
    meta=json.loads((directory/'METADATA.json').read_text())
    with np.load(directory/'trajectory.npz',allow_pickle=False) as data:
        phases=data['phase'];forces=data['pad_object_force_n']
        actions=data['action_targets'];targets=data['position_targets']
    n=len(actions)
    assert actions.shape==(n,9) and phases.shape==(n,) and forces.shape==(n+1,2)
    assert targets.shape==(n+1,9)
    np.testing.assert_array_equal(actions,targets[1:])
    description=meta.get('grip_policy',{})
    if 'maximum_finger_position_m' not in description:
        if description.get('version','zero')!='zero':assert not np.any(phases>=2)
        return dict(status='passed',case_id=meta['case_id'],feedback_activated=False,controls=0,
                    actual_commands_recorded=True)
    source=directory/'source_snapshot/grip_control.py'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    expected=[v for k,v in meta['source_sha256'].items() if Path(k).name=='grip_control.py']
    assert expected==[digest], 'Saved feedback controller hash differs from launch metadata'
    name='_saved_grip_'+digest
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    module=sys.modules[name]
    policy=module.GripPolicy(**{k:description[k] for k in module.GripPolicy.__dataclass_fields__})
    controller=module.GripController(2*description['maximum_finger_position_m'],policy)
    assert description==controller.describe(), 'Unknown or inconsistent feedback policy metadata'
    count=0
    for t,phase in enumerate(phases):
        if phase>=2:
            expected=controller.command(forces[t])
            np.testing.assert_array_equal(expected,actions[t,7:])
            count+=1
    return dict(status='passed',case_id=meta['case_id'],feedback_activated=True,controls=count,
                source_sha256=digest,preceding_observations_used=True,actual_commands_recorded=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.episode),indent=2))


if __name__=='__main__':main()
