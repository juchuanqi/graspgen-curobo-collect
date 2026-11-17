"""Validate delivery receipts and load single or sharded data for training.

Receipt checks use only the standard library so the pipeline supervisor does not
need the separate LeRobot environment. Dataset imports happen only on loading.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    path = root / relative
    require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root.resolve()),
            'Shard path must stay inside the portable delivery')
    return path


def verify_shards(portable):
    """Check every shard audit and the exact union of the global selection."""
    portable = Path(portable)
    manifest = read(portable / 'SHARDS.json')
    selection = read(portable / 'SELECTION.json')
    require(manifest.get('schema') == 1 and manifest.get('format') == 'lerobot_v3_shards',
            'Unsupported shard manifest')
    require(manifest['selection_sha256'] == sha(portable / 'SELECTION.json'), 'Global selection changed')
    require(selection.get('status') == 'selected_before_export' and selection.get('episodes'),
            'Missing qualified selection')
    expected = {ep['case_id']: ep for ep in selection['episodes']}
    require(len(expected) == len(selection['episodes']), 'Duplicate global episode')
    seen, packages = {}, set()
    totals = dict(full_rows=0, rgb_comparisons=0, terminal_rgb_comparisons=0, padding_checks=0)
    statistics_checks, max_error = 0, 0.
    for shard in manifest['shards']:
        package = inside(portable, shard['package'])
        require(package.resolve() not in packages, 'Duplicate shard package')
        packages.add(package.resolve())
        for name, key in [('SELECTION.json', 'selection_sha256'),
                          ('EXPORT_MANIFEST.json', 'export_manifest_sha256'),
                          ('AUDIT.json', 'audit_sha256')]:
            require(sha(package / name) == shard[key], f'Shard receipt changed: {package / name}')
        child_selection = read(package / 'SELECTION.json')
        export = read(package / 'EXPORT_MANIFEST.json')
        audit = read(package / 'AUDIT.json')
        require(audit.get('status') == 'passed' and audit.get('statistics', {}).get('status') == 'passed',
                'Every shard requires full readback and independent statistics')
        require(child_selection.get('static_and_motion_verified') == selection.get('static_and_motion_verified'),
                'Shard qualification changed')
        require(export['selection_sha256'] == shard['selection_sha256'] and export['dataset_root'] == 'dataset',
                'Shard export does not match its selection')
        require(export['episodes'] and len(export['episodes']) == shard['episodes'] == len(child_selection['episodes']),
                'Shard episode count mismatch')
        lengths = 0
        for index, (ep, chosen) in enumerate(zip(export['episodes'], child_selection['episodes'])):
            case = ep['case_id']
            require(case not in seen and case in expected, 'Duplicate or unexpected shard episode')
            require(ep['export_episode_index'] == chosen['export_episode_index'] == index,
                    'Shard episode indices must start at zero and be consecutive')
            for key, value in expected[case].items():
                if key != 'export_episode_index':
                    require(ep.get(key) == chosen.get(key) == value, f'Shard content changed: {case}/{key}')
            seen[case] = ep
            lengths += ep['length']
        require(lengths == export['rows'] == shard['rows'] == audit['full_rows'], 'Shard row count mismatch')
        for key in totals:
            totals[key] += audit[key]
        statistics_checks += audit['statistics']['checks']
        max_error = max(max_error, audit['statistics']['max_absolute_error'])
    require(set(seen) == set(expected), 'Shard union differs from the global selection')
    require(len(seen) == manifest['episodes'] and totals['full_rows'] == manifest['rows'],
            'Combined shard counts mismatch')
    return dict(**totals, statistics=dict(status='passed', checks=statistics_checks,
                max_absolute_error=max_error), selected_episodes=len(seen), shards=len(packages))


def validate_delivery(portable, audit_path):
    portable = Path(portable)
    selection, audit = read(portable / 'SELECTION.json'), read(audit_path)
    require(selection.get('static_and_motion_verified') is True, 'Delivery requires static and motion qualification')
    require(audit.get('status') == 'passed' and audit.get('statistics', {}).get('status') == 'passed',
            'Delivery requires full readback and independent statistics')
    if (portable / 'SHARDS.json').exists():
        require(audit.get('shards_sha256') == sha(portable / 'SHARDS.json'), 'Shard manifest changed')
        totals = verify_shards(portable)
        for key in totals:
            require(audit.get(key) == totals[key], f'Combined audit mismatch: {key}')
    else:
        require('shards_sha256' not in audit, 'Missing shard manifest')
    return selection


def load_training_data(portable, *, delta_timestamps=None):
    """Return (official dataset, normalization statistics, feature schema)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
    portable = Path(portable)
    if (portable / 'SHARDS.json').exists():
        verify_shards(portable)
        manifest = read(portable / 'SHARDS.json')
        # Each package is shards/<object>/dataset; the repo IDs are valid two-part
        # local identifiers. All paths remain relative when the delivery moves.
        repo_ids = [f"{Path(s['package']).name}/dataset" for s in manifest['shards']]
        require(all(s['package'] == f"shards/{Path(s['package']).name}" for s in manifest['shards']),
                'Unexpected shard layout')
        dataset = MultiLeRobotDataset(repo_ids=repo_ids, root=portable / 'shards',
                                     delta_timestamps=delta_timestamps, download_videos=False)
        require(not dataset.disabled_features, 'Shards must have identical training features')
        return dataset, dataset.stats, dataset._datasets[0].meta.features
    dataset = LeRobotDataset('local/graspgen_curobo_panda', root=portable / 'dataset',
                             delta_timestamps=delta_timestamps, download_videos=False)
    return dataset, dataset.meta.stats, dataset.meta.features
