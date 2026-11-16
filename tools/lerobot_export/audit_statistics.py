"""Independent stored-data statistics audit, adapted from the verified 2026-11-04 validator.

RGB moments use exact uint8 frequencies; numeric moments use math.fsum.
Histogram quantiles and aggregation retain LeRobot's published conventions.
"""
import io
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from lerobot.datasets.compute_stats import sample_indices
from bridge import read

QUANTILES = [.01, .10, .50, .90, .99]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quantile_from_hist(hist, edges, quantile):
    cumulative = np.cumsum(hist)
    target = quantile * cumulative[-1]
    index = int(np.searchsorted(cumulative, target))
    if index == 0:
        return edges[0]
    if index >= len(hist):
        return edges[-1]
    previous = cumulative[index - 1]
    count = cumulative[index] - previous
    if count == 0:
        return edges[index]
    return edges[index] + (target - previous) / count * (
        edges[index + 1] - edges[index]
    )


def independent_numeric_stats(array):
    """Centered moments and independently constructed 5000-bin quantiles."""
    array = np.asarray(array)
    data = array.astype(np.float64)
    if data.ndim == 1:
        data = data[:, None]
    require(
        data.ndim == 2 and len(data) > 0 and np.isfinite(data).all(),
        "Invalid numeric statistics input",
    )
    count, dimensions = data.shape
    means = np.array(
        [math.fsum(data[:, column]) / count for column in range(dimensions)]
    )
    variances = np.array(
        [
            math.fsum((float(value) - means[column]) ** 2 for value in data[:, column])
            / count
            for column in range(dimensions)
        ]
    )
    result = {
        "min": data.min(axis=0),
        "max": data.max(axis=0),
        "mean": means,
        "std": np.sqrt(variances),
        "count": np.array([count], dtype=np.int64),
    }
    for quantile in QUANTILES:
        values = []
        for column in range(dimensions):
            if count < 2:
                values.append(means[column])
                continue
            edges = np.linspace(
                result["min"][column] - 1e-10, result["max"][column] + 1e-10, 5001
            )
            histogram, _ = np.histogram(data[:, column], bins=edges)
            values.append(quantile_from_hist(histogram, edges, quantile))
        result[f"q{int(quantile * 100):02d}"] = np.array(values)
    return result


def independent_image_stats(frequencies, frames):
    """No uint8 square or cancellation: use a 256-value frequency distribution."""
    require(frequencies.shape == (3, 256), "Image histogram shape mismatch")
    pixels = frequencies.sum(axis=1)
    require(np.all(pixels == frames * 256 * 256), "Image histogram count mismatch")
    intensities = np.arange(256, dtype=np.float64)
    result = {
        name: np.empty(3, dtype=np.float64) for name in ("min", "max", "mean", "std")
    }
    for channel, counts in enumerate(frequencies):
        occupied = np.flatnonzero(counts)
        minimum, maximum = int(occupied[0]), int(occupied[-1])
        mean = math.fsum(
            float(value) * int(number) for value, number in zip(intensities, counts)
        ) / int(pixels[channel])
        variance = math.fsum(
            (float(value) - mean) ** 2 * int(number)
            for value, number in zip(intensities, counts)
        ) / int(pixels[channel])
        for name, value in (
            ("min", minimum),
            ("max", maximum),
            ("mean", mean),
            ("std", math.sqrt(variance)),
        ):
            result[name][channel] = value
        edges = np.linspace(minimum - 1e-10, maximum + 1e-10, 5001)
        histogram, _ = np.histogram(intensities, bins=edges, weights=counts)
        for quantile in QUANTILES:
            key = f"q{int(quantile * 100):02d}"
            result.setdefault(key, np.empty(3, dtype=np.float64))[channel] = (
                quantile_from_hist(histogram, edges, quantile)
            )
    result = {key: value.reshape(3, 1, 1) / 255.0 for key, value in result.items()}
    result["count"] = np.array([frames], dtype=np.int64)
    return result


def independently_combine(episodes):
    result = {}
    require(
        all(set(item) == set(episodes[0]) for item in episodes),
        "Episode statistic feature mismatch",
    )
    for feature in episodes[0]:
        selected = [episode[feature] for episode in episodes]
        counts = [int(item["count"][0]) for item in selected]
        total = sum(counts)
        mean = (
            sum(item["mean"] * count for item, count in zip(selected, counts)) / total
        )
        variance = (
            sum(
                count * (item["std"] ** 2 + (item["mean"] - mean) ** 2)
                for item, count in zip(selected, counts)
            )
            / total
        )
        combined = {
            "min": np.minimum.reduce([item["min"] for item in selected]),
            "max": np.maximum.reduce([item["max"] for item in selected]),
            "mean": mean,
            "std": np.sqrt(variance),
            "count": np.array([total]),
        }
        for quantile in QUANTILES:
            key = f"q{int(quantile * 100):02d}"
            combined[key] = (
                sum(item[key] * count for item, count in zip(selected, counts)) / total
            )
        result[feature] = combined
    return result


def compare_statistics(actual, expected, location):
    require(set(actual) == set(expected), f"Statistics feature mismatch: {location}")
    checks = 0
    max_error = 0.0
    for feature, statistics in expected.items():
        require(
            set(actual[feature]) == set(statistics),
            f"Statistic names mismatch: {location}/{feature}",
        )
        for statistic, expected_value in statistics.items():
            value = np.asarray(actual[feature][statistic])
            require(
                value.shape == expected_value.shape,
                f"Statistic shape mismatch: {location}/{feature}/{statistic}",
            )
            require(
                np.isfinite(value).all(),
                f"Nonfinite statistic: {location}/{feature}/{statistic}",
            )
            if statistic == "count":
                require(
                    np.array_equal(value, expected_value),
                    f"Statistic count mismatch: {location}/{feature}",
                )
            else:
                error = float(np.max(np.abs(value.astype(np.float64) - expected_value)))
                max_error = max(max_error, error)
                require(
                    np.allclose(value, expected_value, rtol=2e-10, atol=2e-11),
                    f"Statistic mismatch: {location}/{feature}/{statistic}, max error {error}",
                )
            checks += 1
    return {"checks": checks, "max_absolute_error": max_error}


def audit_statistics(package):
    root = Path(package) / 'dataset'
    info = read(root / 'meta/info.json')
    stored_global = read(root / 'meta/stats.json')
    features = list(stored_global)
    images = [key for key in features if info['features'][key]['dtype'] == 'image']
    numeric = [key for key in features if key not in images]
    metadata = {}
    for path in sorted((root / 'meta/episodes').glob('*/*.parquet')):
        for row in pq.read_table(path).to_pylist():
            index = int(row['episode_index'])
            require(index not in metadata, 'Duplicate episode metadata')
            metadata[index] = row
    require(len(metadata) == info['total_episodes'], 'Episode metadata count mismatch')
    sampled = {i: set(sample_indices(int(row['length']))) for i, row in metadata.items()}
    values = {i: {key: [] for key in numeric} for i in metadata}
    frequencies = {i: {key: np.zeros((3, 256), dtype=np.int64) for key in images} for i in metadata}
    seen = {i: [] for i in metadata}
    for path in sorted((root / 'data').glob('*/*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=features):
            for row in batch.to_pylist():
                i, frame = int(row['episode_index']), int(row['frame_index'])
                require(i in metadata, 'Unknown episode in data')
                seen[i].append(frame)
                for key in numeric:
                    values[i][key].append(row[key])
                if frame in sampled[i]:
                    for key in images:
                        with Image.open(io.BytesIO(row[key]['bytes'])) as image:
                            require(image.mode == 'RGB' and image.size == (256, 256), 'RGB statistics shape mismatch')
                            pixels = np.asarray(image)
                        for channel in range(3):
                            frequencies[i][key][channel] += np.bincount(pixels[..., channel].ravel(), minlength=256)
    expected_episodes = []
    checks, maximum = 0, 0.
    for i, row in sorted(metadata.items()):
        require(seen[i] == list(range(int(row['length']))), 'Nonconsecutive statistics frames')
        expected = {key: independent_numeric_stats(np.asarray(values[i][key], dtype=info['features'][key]['dtype']))
                    for key in numeric}
        expected.update({key: independent_image_stats(frequencies[i][key], len(sampled[i])) for key in images})
        actual = {}
        for name, value in row.items():
            if name.startswith('stats/'):
                key, statistic = name[6:].rsplit('/', 1)
                actual.setdefault(key, {})[statistic] = value
        result = compare_statistics(actual, expected, 'episode/' + str(i))
        checks += result['checks']; maximum = max(maximum, result['max_absolute_error'])
        expected_episodes.append(expected)
    result = compare_statistics(stored_global, independently_combine(expected_episodes), 'global')
    checks += result['checks']; maximum = max(maximum, result['max_absolute_error'])
    return dict(status='passed', episode_count=len(metadata), checks=checks, max_absolute_error=maximum,
                method='Independent numeric fsum, exact RGB frequency moments, 5000-bin episode quantiles and count-weighted global aggregation')
