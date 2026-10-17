"""Differentiable means of predicted member sets and a linear activity head."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from msfeirm.models.graph import prepare_nodes


@dataclass
class PooledSubgroups:
    features: Tensor                # [B, K_max, D], still attached to shared nodes
    group_ids: Tensor               # [B, K_max], original predicted IDs; padding=-1
    group_mask: Tensor              # [B, K_max]
    member_counts: Tensor           # [B, K_max]
    member_sets: Tensor             # [B, K_max, N], bool
    member_mask: Tensor             # [B, N]


def pool_subgroups(
    nodes: Tensor, assignments: Tensor, member_mask: Tensor | None = None,
) -> PooledSubgroups:
    """Pool the actual predicted sets, independently of cluster ID numbering.

    Negative assignments mean unassigned members. Empty samples have no valid
    group slots; an entirely empty batch uses K_max=0. Masked NaNs cannot enter
    the means. Only integer grouping decisions are detached from autograd.
    """
    safe, member_mask = prepare_nodes(nodes, member_mask)
    if assignments.shape != nodes.shape[:2] or assignments.dtype not in (torch.int32, torch.int64):
        raise ValueError("assignments must be an integer tensor with shape [B, N]")
    if assignments.device != nodes.device:
        raise ValueError("assignments and nodes must be on the same device")
    with torch.no_grad():
        ids = tuple(torch.unique(row[valid & (row >= 0)], sorted=True)
                    for row, valid in zip(assignments, member_mask, strict=True))
        max_groups = max(value.numel() for value in ids)
        group_ids = torch.full((nodes.shape[0], max_groups), -1, dtype=torch.long, device=nodes.device)
        group_mask = torch.zeros_like(group_ids, dtype=torch.bool)
        for batch, values in enumerate(ids):
            group_ids[batch, :values.numel()] = values
            group_mask[batch, :values.numel()] = True
        member_sets = (
            (group_ids[:, :, None] == assignments[:, None, :])
            & group_mask[:, :, None] & member_mask[:, None, :]
        )
        member_counts = member_sets.sum(dim=-1)
    sums = member_sets.to(dtype=safe.dtype) @ safe
    features = sums / member_counts.clamp_min(1)[:, :, None]
    return PooledSubgroups(features, group_ids, group_mask, member_counts, member_sets, member_mask)


@dataclass
class ActivityOutput:
    pooled: PooledSubgroups
    logits: Tensor
    probabilities: Tensor
    predictions: Tensor


class SubgroupActivityHead(nn.Module):
    def __init__(self, node_dim: int, num_classes: int = 9, *, bias: bool = True) -> None:
        super().__init__()
        if any(type(value) is not int or value <= 0 for value in (node_dim, num_classes)):
            raise ValueError("node_dim and num_classes must be positive integers")
        self.classifier = nn.Linear(node_dim, num_classes, bias=bias)

    def forward(
        self, nodes: Tensor, assignments: Tensor, member_mask: Tensor | None = None,
    ) -> ActivityOutput:
        pooled = pool_subgroups(nodes, assignments, member_mask)
        logits = self.classifier(pooled.features)
        logits = torch.where(pooled.group_mask[:, :, None], logits, 0)
        probabilities = torch.where(pooled.group_mask[:, :, None], logits.softmax(dim=-1), 0)
        predictions = logits.argmax(dim=-1).masked_fill(~pooled.group_mask, -1)
        return ActivityOutput(pooled, logits, probabilities, predictions)
