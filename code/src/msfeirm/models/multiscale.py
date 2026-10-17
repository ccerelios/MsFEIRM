"""Hierarchical channel cascade and scale adapters, equations (1)-(7)."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msfeirm.config import ModelConfig
from msfeirm.models.eca import EfficientChannelAttention


def channel_slices(width: int, count: int = 4) -> tuple[slice, ...]:
    return tuple(slice(index * width, (index + 1) * width) for index in range(count))


class ChannelLinear(nn.Module):
    """Apply a learned linear layer to C at each (H, W), preserving time order."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, feature_map: Tensor) -> Tensor:
        return self.linear(feature_map.movedim(1, -1)).movedim(-1, 1)


@dataclass
class CascadeOutput:
    features: Tensor
    scale_slices: tuple[slice, ...]
    partitions: tuple[Tensor, ...] = ()
    hidden_states: tuple[Tensor, ...] = ()
    scale_outputs: tuple[Tensor, ...] = ()


class HierarchicalFeatureCascade(nn.Module):
    """Keep scale 0; transfer half of scales 1/2; keep the final hidden state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.channels = config.partition_channels
        self.num_scales = config.num_scales
        self.scale_slices = channel_slices(self.channels, self.num_scales)
        self.partition_conv = nn.Conv2d(
            self.channels, self.channels, kernel_size=1, bias=config.conv_bias
        )
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(2 * self.channels, self.channels, 1, bias=config.conv_bias)
            for _ in range(self.num_scales - 2)
        ])
        self.stage_linears = nn.ModuleList([
            ChannelLinear(
                self.channels,
                self.channels if index == self.num_scales - 1 else 2 * self.channels,
                bias=config.linear_bias,
            )
            for index in range(1, self.num_scales)
        ])

    def forward(self, expanded: Tensor, *, capture_trace: bool = False) -> CascadeOutput:
        if expanded.ndim != 4 or expanded.shape[1] != self.channels * self.num_scales:
            raise ValueError(
                f"Cascade input must be [M, {self.channels * self.num_scales}, H, W]"
            )
        partitions = tuple(expanded[:, interval] for interval in self.scale_slices)
        outputs = [partitions[0]]
        hidden_states = []
        carry = None
        for index in range(1, self.num_scales):
            current = self.partition_conv(partitions[index])
            if carry is not None:
                current = self.fusion_convs[index - 2](torch.cat((carry, current), dim=1))
            hidden = self.stage_linears[index - 1](current)
            if capture_trace:
                hidden_states.append(hidden)
            if index < self.num_scales - 1:
                carry, output = hidden.split(self.channels, dim=1)
            else:
                output = hidden
            outputs.append(output)
        return CascadeOutput(
            features=torch.cat(outputs, dim=1),
            scale_slices=self.scale_slices,
            partitions=partitions if capture_trace else (),
            hidden_states=tuple(hidden_states),
            scale_outputs=tuple(outputs) if capture_trace else (),
        )


@dataclass
class FeatureOutput:
    refined_maps: Tensor                 # [B, N, 32, 1, T]
    scale_vectors: tuple[Tensor, ...]    # four [B, N, d_scale] tensors
    node_features: Tensor               # [B, N, 4 * d_scale]
    channel_weights: Tensor             # [B, N, 32]
    member_mask: Tensor                 # [B, N]
    scale_slices: tuple[slice, ...]      # map-channel boundaries
    vector_slices: tuple[slice, ...]     # node-vector boundaries
    cascade_trace: CascadeOutput | None = None


class MultiScaleFeatureExtractor(nn.Module):
    """Full individual feature stage: expansion -> cascade -> ECA -> four vectors."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.expansion = nn.Conv2d(
            self.config.input_channels, self.config.expanded_channels, 1,
            bias=self.config.conv_bias,
        )
        self.cascade = HierarchicalFeatureCascade(self.config)
        self.eca = EfficientChannelAttention(self.config.eca_kernel_size)
        self.adapters = nn.ModuleList([
            nn.Linear(
                self.config.partition_channels * self.config.window_size,
                self.config.scale_embedding_dim,
                bias=self.config.linear_bias,
            )
            for _ in range(self.config.num_scales)
        ])
        self.vector_slices = channel_slices(
            self.config.scale_embedding_dim, self.config.num_scales
        )

    def forward(
        self, imu: Tensor, member_mask: Tensor | None = None, *, capture_trace: bool = False,
    ) -> FeatureOutput:
        if imu.ndim != 4:
            raise ValueError("imu must have shape [B, N, C, T]")
        batch, members, channels, time = imu.shape
        if batch < 1 or members < 1:
            raise ValueError("B and N must be positive")
        if channels != self.config.input_channels or time != self.config.window_size:
            raise ValueError(
                f"Expected C={self.config.input_channels}, T={self.config.window_size}; "
                f"received C={channels}, T={time}"
            )
        if not imu.is_floating_point():
            raise ValueError("imu must be a floating-point tensor")
        if member_mask is None:
            member_mask = torch.ones((batch, members), dtype=torch.bool, device=imu.device)
        if member_mask.shape != (batch, members) or member_mask.dtype != torch.bool:
            raise ValueError("member_mask must be a bool tensor with shape [B, N]")
        if member_mask.device != imu.device:
            raise ValueError("imu and member_mask must be on the same device")
        safe_imu = torch.where(member_mask[:, :, None, None], imu, 0)
        expanded = self.expansion(safe_imu.reshape(batch * members, channels, 1, time))
        cascade = self.cascade(expanded, capture_trace=capture_trace)
        refined, weights = self.eca.forward_with_weights(cascade.features)
        flat_mask = member_mask.reshape(batch * members)
        refined = refined * flat_mask[:, None, None, None]
        scale_vectors = tuple(
            adapter(refined[:, interval].flatten(1)).reshape(batch, members, -1)
            * member_mask[:, :, None]
            for interval, adapter in zip(cascade.scale_slices, self.adapters, strict=True)
        )
        return FeatureOutput(
            refined_maps=refined.reshape(batch, members, self.config.expanded_channels, 1, time),
            scale_vectors=scale_vectors,
            node_features=torch.cat(scale_vectors, dim=-1),
            channel_weights=weights.reshape(batch, members, -1) * member_mask[:, :, None],
            member_mask=member_mask,
            scale_slices=cascade.scale_slices,
            vector_slices=self.vector_slices,
            cascade_trace=cascade if capture_trace else None,
        )
