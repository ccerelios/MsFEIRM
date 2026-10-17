"""Differentiable interaction graphs from inverse Euclidean distance."""

import math

import torch
from torch import Tensor


def prepare_nodes(nodes: Tensor, member_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Check tensor interfaces and replace padding (including NaN padding) by zero."""
    if nodes.ndim != 3 or min(nodes.shape) < 1 or not nodes.is_floating_point():
        raise ValueError("nodes must be a floating-point tensor with shape [B, N, D]")
    if member_mask is None:
        member_mask = torch.ones(nodes.shape[:2], dtype=torch.bool, device=nodes.device)
    if member_mask.shape != nodes.shape[:2] or member_mask.dtype != torch.bool:
        raise ValueError("member_mask must be bool with shape [B, N]")
    if member_mask.device != nodes.device:
        raise ValueError("nodes and member_mask must be on the same device")
    return torch.where(member_mask[:, :, None], nodes, 0), member_mask


def pair_mask(member_mask: Tensor) -> Tensor:
    return member_mask[:, :, None] & member_mask[:, None, :]


def inverse_distance_graph(
    nodes: Tensor,
    member_mask: Tensor | None = None,
    *,
    epsilon: float = 1e-8,
    self_loop_weight: float = 1.0,
) -> Tensor:
    """Return symmetric [B,N,N] weights; invalid members have zero rows/columns.

    Norms use float32 for half/bfloat16 input so epsilon and duplicate-node
    weights remain representable. Graph weights stay differentiable.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(self_loop_weight) or self_loop_weight <= 0:
        raise ValueError("self_loop_weight must be finite and positive")
    safe_nodes, member_mask = prepare_nodes(nodes, member_mask)
    if safe_nodes.dtype in (torch.float16, torch.bfloat16):
        safe_nodes = safe_nodes.float()
    differences = safe_nodes[:, :, None, :] - safe_nodes[:, None, :, :]
    distances = torch.linalg.vector_norm(differences, dim=-1)
    weights = (distances + epsilon).reciprocal()
    diagonal = torch.eye(nodes.shape[1], dtype=torch.bool, device=nodes.device)[None]
    weights = torch.where(diagonal, self_loop_weight, weights)
    return torch.where(pair_mask(member_mask), weights, 0)
