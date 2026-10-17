"""Efficient channel attention, equations (5)-(7)."""

import torch
from torch import Tensor, nn


class EfficientChannelAttention(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be a positive odd integer")
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def attention_weights(self, feature_map: Tensor) -> Tensor:
        if feature_map.ndim != 4:
            raise ValueError("ECA input must have shape [M, C, H, W]")
        descriptor = feature_map.mean(dim=(-2, -1))  # [M, C]
        return torch.sigmoid(self.conv(descriptor.unsqueeze(1)).squeeze(1))

    def forward_with_weights(self, feature_map: Tensor) -> tuple[Tensor, Tensor]:
        weights = self.attention_weights(feature_map)
        return feature_map * weights[:, :, None, None], weights

    def forward(self, feature_map: Tensor) -> Tensor:
        return self.forward_with_weights(feature_map)[0]
