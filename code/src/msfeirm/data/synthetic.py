"""Reproducible learnable sensor groups for the I-4 integration experiment."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch.utils.data import Dataset

from msfeirm.config import ModelConfig, SegmentationConfig
from msfeirm.data.schema import GroupSample


@dataclass(frozen=True)
class SyntheticDataConfig:
    sample_count: int = 12
    min_members: int = 13
    max_members: int = 21
    seed: int = 20261015
    noise_std: float = 0.02

    def __post_init__(self):
        for name in ("sample_count", "min_members", "max_members"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_members > self.max_members:
            raise ValueError("min_members must not exceed max_members")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)


class SyntheticGroupDataset(Dataset):
    """Members of a group share an activity waveform, with independent jitter.

    Activities are distinct within each window. N and K vary independently over
    the configured ranges. Inputs contain sensor patterns only, not target K,
    IDs or assignments. A private generator leaves global training RNG alone.
    """

    version = 1

    def __init__(self, model_config=None, segmentation_config=None, config=None):
        self.model_config = model_config or ModelConfig(window_size=128)
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.config = config or SyntheticDataConfig()
        if self.config.min_members < self.segmentation_config.k_max:
            raise ValueError("Synthetic member count must allow every configured K")
        if self.model_config.num_classes < self.segmentation_config.k_max:
            raise ValueError("Synthetic activity count must allow distinct group activities")
        generator = torch.Generator().manual_seed(self.config.seed)
        time = torch.linspace(0, 2 * torch.pi, self.model_config.window_size)
        channel = torch.arange(self.model_config.input_channels)[:, None]
        prototypes = []
        for activity in range(self.model_config.num_classes):
            waveform = torch.sin((1 + activity % 3) * time + .7 * channel)
            waveform = waveform + .6 * torch.cos((1 + activity // 3) * time + .3 * channel)
            offset = .8 * torch.randn(self.model_config.input_channels, 1, generator=generator)
            prototypes.append(waveform + offset)
        self.samples = []
        for index in range(self.config.sample_count):
            k = self.segmentation_config.k_min + index % (self.segmentation_config.k_max - self.segmentation_config.k_min + 1)
            n = int(torch.randint(self.config.min_members, self.config.max_members + 1, (), generator=generator))
            activities = torch.randperm(self.model_config.num_classes, generator=generator)[:k]
            ids = torch.arange(n) % k
            noise = self.config.noise_std * torch.randn(n, self.model_config.input_channels, self.model_config.window_size, generator=generator)
            imu = torch.stack([prototypes[int(activities[group])] for group in ids]) + noise
            permutation = torch.randperm(n, generator=generator)
            imu, ids = imu[permutation], ids[permutation]
            self.samples.append(GroupSample(
                imu, sample_id=f"synthetic_{index:04d}", group_ids=ids, group_labels=activities,
                pair_targets=(ids[:, None] == ids[None, :]).float(), k_clusters=k,
                noise_mask=torch.zeros(n, dtype=torch.bool),
            ))
        digest = hashlib.sha256(json.dumps(self.specification(), sort_keys=True).encode())
        for sample in self.samples:
            for tensor in (sample.imu, sample.group_ids, sample.group_labels, sample.pair_targets):
                digest.update(tensor.contiguous().numpy().tobytes())
        self.fingerprint = digest.hexdigest()

    def specification(self):
        return {"version": self.version, "model": self.model_config.to_dict(),
                "segmentation": self.segmentation_config.to_dict(), "synthetic": self.config.to_dict()}

    def manifest(self):
        return {**self.specification(), "sha256": self.fingerprint,
                "samples": [{"sample_id": sample.sample_id, "shape": list(sample.imu.shape),
                             "k_clusters": sample.k_clusters} for sample in self.samples]}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
