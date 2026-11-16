"""Offline ACT training plumbing check on an audited native LeRobot dataset."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from bridge import read, sha, write


def main(args):
    if args.steps < 1:
        raise ValueError('steps must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = read(args.audit)
    if audit['status'] != 'passed':
        raise ValueError('Official exact readback must pass before training')
    selection = read(args.portable / 'SELECTION.json')
    if selection.get('static_and_motion_verified') is not True:
        raise ValueError('Training check requires static and motion qualified source data')
    started = time.monotonic()
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.manual_seed(20261113)
    np.random.seed(20261113)
    chunk = 16
    dataset = LeRobotDataset('local/graspgen_curobo_panda', root=args.portable / 'dataset',
                             delta_timestamps={'action': [i / 60 for i in range(chunk)]})
    features = dataset_to_policy_features(dataset.meta.features)
    keys = ['observation.state', 'observation.images.base_camera', 'observation.images.wrist_camera']
    config = ACTConfig(input_features={k: features[k] for k in keys},
        output_features={'action': features['action']}, device='cuda', push_to_hub=False,
        pretrained_backbone_weights=None, chunk_size=chunk, n_action_steps=chunk,
        dim_model=128, n_heads=4, dim_feedforward=512, n_encoder_layers=2,
        n_decoder_layers=1, n_vae_encoder_layers=2)
    policy = ACTPolicy(config).to('cuda').train()
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=dataset.meta.stats)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-4)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(20261113))
    initial = next(policy.parameters()).detach().clone()
    iterator = iter(loader)
    metrics = []
    # Inspect an actual episode boundary, including LeRobot's official padding.
    last_index = int(dataset.meta.episodes[0]['dataset_to_index']) - 1
    boundary = dataset[last_index]
    assert boundary['action_is_pad'].shape == (chunk,)
    assert not boundary['action_is_pad'][0] and boundary['action_is_pad'][1:].all()
    with (args.output / 'metrics.jsonl').open('x') as stream:
        for step in range(args.steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader); batch = next(iterator)
            batch = preprocessor({k: v for k, v in batch.items() if k in keys or k in ('action', 'action_is_pad')})
            assert batch['action'].shape[1:] == (chunk, 8)
            loss, values = policy.forward(batch)
            assert torch.isfinite(loss), 'Nonfinite training loss'
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.)
            assert torch.isfinite(grad) and grad > 0, 'Missing or invalid gradients'
            optimizer.step()
            record = dict(step=step + 1, loss=float(loss.detach()), grad_norm=float(grad),
                          **{k: float(v) for k, v in values.items()})
            metrics.append(record)
            stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush()
            if (step + 1) % 10 == 0: print(record, flush=True)
    change = float((next(policy.parameters()).detach() - initial).abs().max())
    assert change > 0
    policy.save_pretrained(args.output / 'checkpoint')
    preprocessor.save_pretrained(args.output / 'checkpoint')
    postprocessor.save_pretrained(args.output / 'checkpoint')
    torch.save(optimizer.state_dict(), args.output / 'optimizer.pt')
    policy.eval()
    with torch.no_grad():
        prediction = postprocessor(policy.select_action({k: v for k, v in batch.items() if k != 'action'}))
    assert prediction.shape == (len(batch['observation.state']), 8) and torch.isfinite(prediction).all()
    write(args.output / 'RESULT.json', dict(status='passed', steps=args.steps, batch_size=4,
        dataset_episodes=dataset.num_episodes, dataset_rows=len(dataset), input_keys=keys,
        output_shape=list(prediction.shape), action_chunk_size=chunk,
        official_boundary_padding_checked=True, first_loss=metrics[0]['loss'], last_loss=metrics[-1]['loss'],
        maximum_first_parameter_change=change, all_losses_and_gradients_finite=True,
        audit_sha256=sha(args.audit), selection_sha256=sha(args.portable / 'SELECTION.json'),
        wall_s=time.monotonic() - started, limitation='Training plumbing only; no closed-loop policy success claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portable', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    main(parser.parse_args())
