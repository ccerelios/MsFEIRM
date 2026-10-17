"""Differentiable pair membership and normalized subgroup cardinality heads."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msfeirm.config import SegmentationConfig
from msfeirm.models.graph import pair_mask, prepare_nodes


@dataclass
class MembershipOutput:
    logits: Tensor
    probabilities: Tensor
    valid_pairs: Tensor


class PairwiseMembershipHead(nn.Module):
    """Symmetric pair scorer, including learned diagonal entries for supervision."""

    def __init__(self, node_dim: int, hidden_dim: int = 64, *, bias: bool = True) -> None:
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (node_dim, hidden_dim)):
            raise ValueError("node_dim and hidden_dim must be positive integers")
        self.node_dim = node_dim
        self.scorer = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=bias),
        )

    def forward(self, nodes: Tensor, member_mask: Tensor | None = None) -> MembershipOutput:
        safe_nodes, member_mask = prepare_nodes(nodes, member_mask)
        if safe_nodes.shape[-1] != self.node_dim:
            raise ValueError(f"Expected node feature dimension {self.node_dim}")
        left, right = safe_nodes[:, :, None, :], safe_nodes[:, None, :, :]
        pairs = torch.cat(((left - right).abs(), left * right), dim=-1)
        logits = self.scorer(pairs).squeeze(-1)
        logits = (logits + logits.transpose(1, 2)) * 0.5
        valid_pairs = pair_mask(member_mask)
        logits = torch.where(valid_pairs, logits, 0)
        probabilities = torch.where(valid_pairs, logits.sigmoid(), 0)
        return MembershipOutput(logits, probabilities, valid_pairs)


@dataclass
class CardinalityOutput:
    score: Tensor
    continuous_count: Tensor
    decoded_count: Tensor
    sample_mask: Tensor
    valid_counts: Tensor


class SubgroupCardinalityHead(nn.Module):
    """Masked mean -> FFN -> sigmoid; rounding is outside the gradient path."""

    def __init__(
        self, node_dim: int, config: SegmentationConfig | None = None, *, bias: bool = True,
    ) -> None:
        super().__init__()
        if type(node_dim) is not int or node_dim <= 0:
            raise ValueError("node_dim must be a positive integer")
        self.node_dim = node_dim
        self.config = config or SegmentationConfig()
        self.predictor = nn.Sequential(
            nn.Linear(node_dim, self.config.cardinality_hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(self.config.cardinality_hidden_dim, 1, bias=bias),
        )

    def normalize_targets(self, counts: Tensor) -> Tensor:
        """Transform count tensors only; no target auditing or clipping."""
        counts = counts.to(dtype=self.predictor[0].weight.dtype)
        span = self.config.k_max - self.config.k_min
        return (counts - self.config.k_min) / span if span else counts * 0

    @torch.no_grad()
    def decode(self, score: Tensor, valid_counts: Tensor) -> Tensor:
        continuous = self.config.k_min + (self.config.k_max - self.config.k_min) * score
        upper = valid_counts.clamp(max=self.config.k_max)
        lower = valid_counts.clamp(max=self.config.k_min)
        return continuous.round().to(torch.long).maximum(lower).minimum(upper)

    def forward(self, nodes: Tensor, member_mask: Tensor | None = None) -> CardinalityOutput:
        safe_nodes, member_mask = prepare_nodes(nodes, member_mask)
        if safe_nodes.shape[-1] != self.node_dim:
            raise ValueError(f"Expected node feature dimension {self.node_dim}")
        valid_counts = member_mask.sum(dim=1)
        sample_mask = valid_counts > 0
        pooled = safe_nodes.sum(dim=1) / valid_counts.clamp_min(1)[:, None]
        score = self.predictor(pooled).squeeze(-1).sigmoid()
        # A fixed range has a zero, still connected, normalized prediction/loss.
        if self.config.k_min == self.config.k_max:
            score = score * 0
        score = torch.where(sample_mask, score, 0)
        continuous = self.config.k_min + (self.config.k_max - self.config.k_min) * score
        continuous = torch.where(sample_mask, continuous, 0)
        return CardinalityOutput(
            score, continuous, self.decode(score, valid_counts), sample_mask, valid_counts,
        )
