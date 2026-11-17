"""Experimental deterministic 200-pose pool for a single upright cylinder.

This is a geometric prior, not GraspGen inference. It replaces the pool for a
selected object; never append it to an existing 200-pose source. The legacy
pipeline directory is named graspgen, while REPORT/source_kind records origin.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'runtime')]
from common import read, write, sha
from geometry.api import manifest, prepare_candidates
from geometry.robot import GRASPGEN_TO_TCP


def axial_grid(asset):
    parts = asset['parts']
    if (len(parts) != 1 or parts[0]['shape'] != 'cylinder'
            or not np.allclose(parts[0].get('position', [0,0,0]), 0)
            or not np.allclose(parts[0].get('rpy', [0,0,0]), 0)):
        raise ValueError('Axial prior requires one centered Z-axis cylinder')
    poses, parameters = [], []
    for height in (.25*parts[0]['length'], 0.):
        for tilt in (0., 15., -15., 30., -30.):
            for yaw in np.arange(20)*18.:
                tcp = np.eye(4)
                tcp[:3,:3] = (Rotation.from_euler('z', yaw, degrees=True)
                    * Rotation.from_euler('y', tilt, degrees=True)
                    * Rotation.from_euler('x', 180, degrees=True)).as_matrix()
                tcp[2,3] = height
                poses.append(tcp @ np.linalg.inv(GRASPGEN_TO_TCP))
                parameters.append(dict(yaw_deg=float(yaw), tilt_deg=tilt, height_m=height))
    return np.asarray(poses), parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    asset = manifest()['objects'][args.object]
    folder = args.output_root.resolve()/'graspgen'/args.object
    if folder.exists(): raise FileExistsError('Candidate pools are immutable; choose another output root')
    poses, parameters = axial_grid(asset)
    # Equal numeric priorities are only an adapter to the geometry qualifier;
    # they are removed from accepted poses and never treated as confidence.
    report = prepare_candidates(args.object, poses, np.ones(200))
    for candidate in report['candidates']:
        candidate['score'] = None
        candidate['candidate_source_kind'] = 'deterministic_axial_primitive_grid_v1'
        candidate['grid_parameters'] = parameters[candidate['generation_index']]
    folder.mkdir(parents=True, exist_ok=False)
    cache = folder/'candidates.npz'
    np.savez_compressed(cache, T_object_grasp=poses, generation_index=np.arange(200))
    report.update(source_kind='deterministic_axial_primitive_grid_v1 (no GraspGen inference)',
        source_cache=cache.name, source_cache_sha256=sha(cache), generation_calls=0,
        model_candidates=0, source_candidates=200, input_mesh_sha256=sha(asset['mesh_path']),
        cache_key=dict(mesh_sha256=sha(asset['mesh_path']),surface_sha256=sha(asset['surface_path']),
            parts=asset['parts'],generator_sha256=sha(__file__),
            geometry_source_sha256={p.name:sha(p) for p in (ROOT/'runtime/geometry').glob('*.py')}),
        parameters=parameters, budget_policy='Replacement pool of exactly200 poses; no appended model pool',
        score_policy='No neural score; original_rank is fixed grid enumeration after geometry deduplication',
        method_namespace='graspgen is the legacy pipeline key; source_kind is authoritative')
    write(folder/'REPORT.json',report)
    print(f'{len(report["candidates"])} geometry-qualified of200 geometric prior poses: {folder}')


if __name__ == '__main__': main()
