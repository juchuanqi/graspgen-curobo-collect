"""Process-local LeRobot 0.4.4 statistics with centered float64 moments.

Uses the correction already validated by the 2026-11-04 v3 pipeline, while
retaining official image sampling, histogram quantiles and count conventions.
"""
from contextlib import contextmanager
from threading import RLock

import numpy as np
from lerobot.datasets import lerobot_dataset
from lerobot.datasets.compute_stats import get_feature_stats, sample_images

_LOCK = RLock()


def safe_stats(values, image=False, quantile_list=None):
    original_dtype = np.asarray(values).dtype
    values = np.asarray(values, dtype=np.float64)
    axes, keepdims = ((0, 2, 3), True) if image else (0, values.ndim == 1)
    result = get_feature_stats(values, axis=axes, keepdims=keepdims, quantile_list=quantile_list)
    result['mean'] = np.mean(values, axis=axes, keepdims=keepdims, dtype=np.float64)
    result['std'] = np.std(values, axis=axes, keepdims=keepdims, dtype=np.float64, ddof=0)
    if image:
        result = {key: value if key == 'count' else np.squeeze(value / 255., axis=0)
                  for key, value in result.items()}
    else:
        for key in ('min', 'max'):
            result[key] = result[key].astype(original_dtype)
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Nonfinite episode statistics')
    return result


def stable_episode_stats(episode_data, features, quantile_list=None):
    result = {}
    for key, values in episode_data.items():
        dtype = features[key]['dtype']
        if dtype == 'string':
            continue
        image = dtype in ('image', 'video')
        # Quantize numeric buffers to their actual Arrow storage dtype first.
        values = sample_images(values) if image else np.asarray(values, dtype=np.dtype(dtype))
        result[key] = safe_stats(values, image, quantile_list)
    return result


@contextmanager
def stable_statistics():
    with _LOCK:
        original = lerobot_dataset.compute_episode_stats
        lerobot_dataset.compute_episode_stats = stable_episode_stats
        try:
            yield
        finally:
            lerobot_dataset.compute_episode_stats = original
