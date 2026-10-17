"""Window splits and finite channel standardization for real sensor inputs."""

from dataclasses import dataclass

import numpy as np


def split_window_indices(count, *, ratios=(.7, .1, .2), seed=42, supplied=None):
    names = ("train", "validation", "test")
    if supplied is not None:
        result = {name: np.asarray(supplied[name], dtype=np.int64) for name in names}
        for indices in result.values():
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= count):
                raise ValueError("Supplied window indices must be 1-D and in range")
        return result
    if len(ratios) != 3 or any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError("Three nonnegative split ratios must sum to one")
    order = np.random.default_rng(seed).permutation(count)
    first, second = int(count * ratios[0]), int(count * (ratios[0] + ratios[1]))
    return dict(zip(names, np.split(order, (first, second)), strict=True))


@dataclass
class ChannelStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, windows, indices, *, epsilon=1e-8, chunk_size=256):
        if windows.ndim != 3 or len(indices) == 0:
            raise ValueError("Fit requires [windows,channels,time] and nonempty training indices")
        if not np.isfinite(epsilon) or epsilon <= 0 or chunk_size < 1:
            raise ValueError("epsilon and chunk_size must be positive")
        channels = windows.shape[1]
        total, count = np.zeros(channels), np.zeros(channels)
        for start in range(0, len(indices), chunk_size):
            values = windows[indices[start:start+chunk_size]].astype(np.float64)
            valid = np.isfinite(values)
            total += np.where(valid, values, 0).sum(axis=(0, 2))
            count += valid.sum(axis=(0, 2))
        mean = total / np.maximum(count, 1)
        squared = np.zeros(channels)
        for start in range(0, len(indices), chunk_size):
            values = windows[indices[start:start+chunk_size]].astype(np.float64)
            centered = np.where(np.isfinite(values), values - mean[None, :, None], 0)
            squared += (centered * centered).sum(axis=(0, 2))
        scale = np.sqrt(squared / np.maximum(count, 1))
        scale = np.where(scale > epsilon, scale, 1)
        return cls(mean, scale)

    def transform(self, values):
        if values.shape[-2] != len(self.mean):
            raise ValueError("Input channel dimension does not match normalization")
        mean, scale = self.mean[:, None], self.scale[:, None]
        filled = np.where(np.isfinite(values), values, mean)
        return np.asarray((filled - mean) / scale, dtype=np.float32)
