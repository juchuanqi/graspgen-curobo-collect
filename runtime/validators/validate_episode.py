"""CPU-only chronology, sensor, camera and exact collection-score replay check.

Default validation is for newly collected v2 episodes. --replay-only may inspect
old recordings, but explicitly cannot qualify formal results. No labels change.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RAW_KEYS = (
    'qpos', 'qvel', 'position_targets', 'object_pose', 'tcp_pose',
    'object_linear_velocity', 'object_angular_velocity', 'link_contact_force',
    'link_contact_peak_n', 'object_table_peak_n', 'object_wall_peak_n',
    'pad_object_force_n', 'nonpad_object_peak_n', 'nonpad_object_force_bound_n',
    'max_contact_penetration_m', 'finger_contact_pos_w',
    'raw_finger_contact_points_w', 'raw_finger_contact_normal_force_n',
    'raw_finger_contact_pad_mask', 'raw_finger_contact_valid',
    'contact_substep_object_pose_w', 'contact_substep_physics_tick',
    'contact_substep_sample_count',
)
SUBSTEP_KEYS = (
    'contact_substep_points_w', 'contact_substep_normals_w', 'contact_substep_normal_force_n',
    'contact_substep_separation_m', 'contact_substep_valid', 'contact_substep_pad_mask',
    'contact_substep_pad_force_n', 'contact_substep_finger_pose_w',
    'contact_substep_object_velocity_w', 'contact_substep_cached_object_pose_w',
    'contact_substep_hand_pose_w', 'contact_substep_tcp_pose_w',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoring_from(source):
    # Import only CPU scoring modules, never the Isaac collector bootstrap.
    for name in ('pose_motion_score', 'collection_score'):
        path = source / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules['collection_score'], sys.modules['pose_motion_score']


def equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
        return bool(np.array_equal(a, b))
    return bool(np.allclose(a, b, rtol=2e-6, atol=1e-8, equal_nan=True))


def validate(directory, collector=None, replay_only=False):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'METADATA.json').read_text())
    source = Path(collector).resolve() if collector else directory / 'source_snapshot'
    if not (source / 'collection_score.py').is_file():
        if not replay_only:
            raise AssertionError('New v2 episode requires its own complete source_snapshot')
        source = HERE / 'collector'
    score_module, motion_module = scoring_from(source)
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as z:
        data = {key: z[key] for key in z.files}
    assert all(key in data for key in RAW_KEYS + SUBSTEP_KEYS), 'Missing required raw/contact field'
    actions, phases = data['action_targets'], data['phase']
    n, states = len(actions), len(data['qpos'])
    assert actions.shape == (n, 9) and phases.shape == (n,)
    assert states == n + 1 == metadata['observations'] and n == metadata['actions']
    assert np.isfinite(actions).all() and n <= 2400
    assert np.all((phases >= 0) & (phases <= 4)) and np.all(np.diff(phases.astype(int)) >= 0)
    for key, array in data.items():
        assert len(array) == (n if key in ('action_targets', 'phase', 'control_timing_s') else states), key + ' chronology'
    phase_timing = {}
    if not replay_only or 'control_timing_s' in data:
        timing = data['control_timing_s']
        columns = ['env_step', 'snapshot', 'record', 'total']
        assert metadata['control_timing_columns'] == columns
        assert timing.shape == (n, 4) and np.isfinite(timing).all() and np.all(timing >= 0)
        assert np.all(timing[:, 3] + 1e-8 >= timing[:, :3].sum(axis=1))
        totals = timing.sum(axis=0)
        for index, key in enumerate(columns[:3]):
            assert np.isclose(totals[index], metadata['timing_s'][key], rtol=1e-10, atol=1e-6), key + ' timer sum'
        assert metadata['timing_s']['physics_camera'] + 1e-6 >= totals[:2].sum()
        assert metadata['timing_s']['physics_camera'] <= totals[3] + 1e-6
        assert totals[3] <= metadata['episode_wall_s'] + 1e-6
        for phase in np.unique(phases):
            selected = timing[phases == phase]
            phase_timing[str(int(phase))] = dict(
                controls=len(selected), **dict(zip(columns, selected.sum(axis=0).tolist())))
    for key in RAW_KEYS + SUBSTEP_KEYS:
        assert np.isfinite(data[key]).all(), key + ' nonfinite'
    assert np.all(data['contact_substep_sample_count'] == 8)
    ticks = data['contact_substep_physics_tick']
    assert ticks.shape == (states, 8) and ticks.dtype.kind in 'iu'
    assert np.all(np.diff(ticks, axis=1) == 1)
    assert np.all(ticks[1:, 0] == ticks[:-1, -1] + 1)
    assert data['contact_substep_object_pose_w'].shape == (states, 8, 7)
    for key in SUBSTEP_KEYS:
        assert data[key].shape[1] == 8, key + ' substep count'
    # Check the initial physical interval too; the scorer consumes it as a
    # reference and validates all later intervals using its unmodified helper.
    pose = data['object_pose']
    physical = data['contact_substep_object_pose_w']
    rotation = motion_module.rotation
    rotation(pose)
    rotation(physical.reshape(-1, 7))
    assert np.max(np.linalg.norm(physical[:, -1, :3] - pose[:, :3], axis=-1)) <= 1e-6
    endpoint_error = (rotation(physical[:, -1]) * rotation(pose).inv()).magnitude()
    assert endpoint_error.max() <= 1e-5
    if n:
        assert np.allclose(actions, data['position_targets'][1:], rtol=0, atol=1e-6), 'Action/next target alignment'
        assert np.max(np.abs(actions[:, :7] - data['position_targets'][:-1, :7])) <= .0025011
    assert np.allclose(data['sim_time'], np.arange(states) / 60, rtol=0, atol=1e-10)
    cameras = metadata['camera_names']
    assert set(cameras) == {'base_camera', 'wrist_camera'}
    assert data['camera_frame'].shape == (states, len(cameras))
    assert np.all(np.diff(data['camera_frame'], axis=0) == 1), 'Missing/reused camera sensor frame'
    camera_shapes = {}
    for name in cameras:
        image = np.load(directory / 'cameras' / (name + '.npy'), mmap_mode='r', allow_pickle=False)
        assert image.shape == (states, 256, 256, 3) and image.dtype == np.uint8, name + ' RGB chronology'
        camera_shapes[name] = list(image.shape)
        del image

    def row_at(index):
        row = {key: np.array(data[key][index], copy=True) for key in RAW_KEYS}
        # sanitize() stores absent centroids as zero plus a validity mask. Undo
        # that serialization convention before replaying the original hook.
        valid = data['finger_contact_pos_w_valid'][index]
        row['finger_contact_pos_w'][~valid] = np.nan
        return row

    score = score_module.CollectionScore(row_at(0), metadata['task'])
    differences = []
    derived_keys = None
    first_terminal = None
    for index in range(states):
        row = row_at(index)
        hold = bool(phases[index - 1] >= 3) if index else False
        derived = score.derive(row) if index == 0 else score.update(row, hold_enabled=hold)
        stored = score_module.sanitize(derived)
        if derived_keys is None:
            derived_keys = sorted(stored)
        if not replay_only:
            for key, value in stored.items():
                assert key in data, 'Missing saved score field ' + key
                if not equal(data[key][index], value):
                    differences.append(dict(observation=index, field=key))
                    if len(differences) >= 10:
                        raise AssertionError('Saved score fields differ from replay: ' + str(differences))
        if score.reason is not None and first_terminal is None:
            first_terminal = index
    report = score.report(metadata['reason'])
    assert not score.primary.motion_errors, score.primary.motion_errors
    if not replay_only:
        assert not differences, differences
        assert metadata['success_contract'] == score_module.PROTOCOL
        assert metadata['accepted'] == metadata['success'] == score.success
        assert metadata['best_stable_window'] == score.best_window
        assert metadata['score_diagnostics'] == report, 'Saved primary/legacy diagnostics differ from exact replay'
        assert score.reason is None or score.reason == metadata['reason']
        assert first_terminal is None or first_terminal == n, 'Collection continued after primary terminal'
        assert metadata['control_hz'] == 60 and metadata['physics_hz'] == 480
        assert metadata.get('error') is None and metadata['reason'] not in {
            'technical_exception', 'planner_worker_timeout_or_exit'}
        assert metadata['physical_attempts'] in (0, 1)
        if metadata['role'] == 'formal':
            entry = json.loads((directory / 'ENTRY.json').read_text())
            assert entry['included_in_formal_score'] is True and entry['max_physical_attempts_per_method'] == 1
            assert entry['scene_seed'] == metadata['scene_seed']
        for path, expected in metadata['source_sha256'].items():
            snapshot = directory / 'source_snapshot' / Path(path).name
            assert sha(snapshot) == expected, 'Launch source snapshot hash changed: ' + str(snapshot)
    return dict(
        status='passed', scope='retrospective_replay_only' if replay_only else 'new_v2_episode_integrity',
        formal_eligible=not replay_only and metadata['role'] == 'formal',
        episode=str(directory), case_id=metadata['case_id'],
        included_in_formal_score=metadata['role'] == 'formal' and not replay_only,
        actions=n, observations=states, physical_samples=(n + 1) * 8,
        complete_substeps=True, consecutive_physics_ticks=True, physical_control_pose_endpoints_aligned=True,
        max_endpoint_rotation_error_rad=float(endpoint_error.max()), camera_shapes=camera_shapes,
        phase_timing_s=phase_timing,
        primary_success=bool(score.success), primary_reason=report['primary_reason'],
        legacy_success_on_collected_prefix=report['legacy_success_on_collected_prefix'],
        legacy_reason_on_collected_prefix=report['legacy_reason_on_collected_prefix'],
        exact_saved_derived_fields_replayed=not replay_only, derived_fields_checked=derived_keys,
        score_diagnostics=report, trajectory_sha256=sha(directory / 'trajectory.npz'),
        scoring_source=str(source), score_source_sha256={name: sha(source / name) for name in (
            'collection_score.py', 'pose_motion_score.py', 'legacy_scoring.py')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', nargs='+', type=Path)
    parser.add_argument('--collector', type=Path, help='Explicit CPU scoring source; defaults to episode source_snapshot')
    parser.add_argument('--replay-only', action='store_true', help='Inspect old data; excludes formal qualification')
    parser.add_argument('--output', type=Path, help='Optional new JSON report; refuses overwrite')
    args = parser.parse_args()
    reports = [validate(p, args.collector, args.replay_only) for p in args.episode]
    result = reports[0] if len(reports) == 1 else dict(status='passed', episodes=reports)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')
