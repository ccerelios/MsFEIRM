"""Configuration shared by the feature extractor and later model stages."""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 6
    window_size: int = 200
    num_scales: int = 4
    partition_channels: int = 8
    scale_embedding_dim: int = 32
    eca_kernel_size: int = 3
    conv_bias: bool = True
    linear_bias: bool = True
    local_depth: int = 2
    global_depth: int = 1
    attention_heads: int = 2
    attention_scale: bool = False
    ffn_expansion: int = 2
    dropout: float = 0.0
    epsilon: float = 1e-8
    num_classes: int = 9

    def __post_init__(self) -> None:
        positive = (
            "input_channels", "window_size", "num_scales", "partition_channels",
            "scale_embedding_dim", "eca_kernel_size", "local_depth", "global_depth",
            "attention_heads", "ffn_expansion", "num_classes",
        )
        for name in positive:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_scales != 4:
            raise ValueError("The complete model uses four hierarchical scales")
        if self.eca_kernel_size % 2 == 0:
            raise ValueError("eca_kernel_size must be odd for same-length padding")
        if self.scale_embedding_dim % self.attention_heads:
            raise ValueError("scale_embedding_dim must be divisible by attention_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be in (0, 1)")
        for name in ("conv_bias", "linear_bias", "attention_scale"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")

    @property
    def expanded_channels(self) -> int:
        return self.num_scales * self.partition_channels

    @property
    def node_dim(self) -> int:
        return self.num_scales * self.scale_embedding_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelConfig":
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown)}")
        return cls(**dict(values))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return values


def load_model_config(path: str | Path) -> ModelConfig:
    values = load_yaml(path)
    return ModelConfig.from_mapping(values.get("model", values))


@dataclass(frozen=True)
class SegmentationConfig:
    """Count range belongs to a data protocol, not an individual input sample."""

    k_min: int = 3
    k_max: int = 6
    pair_hidden_dim: int = 64
    cardinality_hidden_dim: int = 64
    clustering_seed: int = 42
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    gap_tolerance: float = 1e-6
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "k_min", "k_max", "pair_hidden_dim", "cardinality_hidden_dim",
            "kmeans_n_init", "kmeans_max_iter",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.k_min > self.k_max:
            raise ValueError("k_min must be no greater than k_max")
        if type(self.clustering_seed) is not int or not 0 <= self.clustering_seed < 2**32:
            raise ValueError("clustering_seed must be an integer in [0, 2**32)")
        if not math.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("epsilon must be finite and in (0, 1)")
        if not math.isfinite(self.gap_tolerance) or not 0 <= self.gap_tolerance < 1:
            raise ValueError("gap_tolerance must be finite and in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SegmentationConfig":
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown segmentation configuration fields: {sorted(unknown)}")
        return cls(**dict(values))


def load_segmentation_config(path: str | Path) -> SegmentationConfig:
    values = load_yaml(path)
    return SegmentationConfig.from_mapping(values.get("segmentation", values))


@dataclass(frozen=True)
class LossConfig:
    lambda_cardinality: float = 2.0
    lambda_member: float = 1.5
    ignore_index: int = -100

    def __post_init__(self) -> None:
        for name in ("lambda_cardinality", "lambda_member"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.ignore_index) is not int:
            raise ValueError("ignore_index must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LossConfig":
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown loss configuration fields: {sorted(unknown)}")
        return cls(**dict(values))


def load_loss_config(path: str | Path) -> LossConfig:
    values = load_yaml(path)
    return LossConfig.from_mapping(values.get("loss", values))
