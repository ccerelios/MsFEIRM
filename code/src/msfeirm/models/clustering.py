"""Cosine affinity, adaptive pruning and discrete, CPU spectral clustering.

The inverse-distance interaction graph and learned membership probabilities do
not enter this module. Only shared node features and a decoded count are used.
"""

from dataclasses import dataclass
import math
import warnings

import torch
from torch import Tensor, nn
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from msfeirm.config import SegmentationConfig
from msfeirm.models.graph import pair_mask, prepare_nodes


def _check_epsilon(epsilon: float) -> None:
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")


def _prepare_relations(relations: Tensor, member_mask: Tensor | None) -> tuple[Tensor, Tensor]:
    if relations.ndim != 3 or relations.shape[1] != relations.shape[2]:
        raise ValueError("relations must have shape [B, N, N]")
    safe, member_mask = prepare_nodes(relations, member_mask)
    return torch.where(pair_mask(member_mask), safe, 0), member_mask


def cosine_relations(
    nodes: Tensor, member_mask: Tensor | None = None, *, epsilon: float = 1e-8,
) -> Tensor:
    """Raw symmetric cosines in [-1,1], with zero-norm/padding protection.

    This function is differentiable. Detaching occurs only in the clusterer.
    Half-precision norms are evaluated in float32, as in the interaction graph.
    """
    _check_epsilon(epsilon)
    safe, member_mask = prepare_nodes(nodes, member_mask)
    if safe.dtype in (torch.float16, torch.bfloat16):
        safe = safe.float()
    normalized = safe / torch.linalg.vector_norm(safe, dim=-1, keepdim=True).clamp_min(epsilon)
    raw = normalized @ normalized.transpose(1, 2)
    raw = ((raw + raw.transpose(1, 2)) * 0.5).clamp(-1, 1)
    return torch.where(pair_mask(member_mask), raw, 0)


@dataclass
class PrunedRelations:
    affinity: Tensor
    cutoffs: Tensor
    max_relative_drops: Tensor
    retained_counts: Tensor


def prune_cosine_relations(
    raw_cosines: Tensor, member_mask: Tensor | None = None, *,
    epsilon: float = 1e-8, gap_tolerance: float = 1e-6,
) -> PrunedRelations:
    """Keep positive edges above each row's largest relative-drop cutoff.

    Self edges are excluded from sorting. Equal values at a selected cutoff are
    all retained. With no informative drop all positive candidates are kept.
    Directed selections are united before restoring symmetric cosine weights.
    retained_counts counts off-diagonal edges after that union.
    """
    _check_epsilon(epsilon)
    if not math.isfinite(gap_tolerance) or not 0 <= gap_tolerance < 1:
        raise ValueError("gap_tolerance must be finite and in [0,1)")
    safe, member_mask = _prepare_relations(raw_cosines, member_mask)
    if safe.dtype in (torch.float16, torch.bfloat16):
        safe = safe.float()
    weights = ((safe + safe.transpose(1, 2)) * 0.5).clamp(0, 1)
    selected = torch.zeros_like(weights, dtype=torch.bool)
    cutoffs = weights.new_zeros(weights.shape[:2])
    max_drops = weights.new_zeros(weights.shape[:2])
    for batch in range(weights.shape[0]):
        indices = member_mask[batch].nonzero(as_tuple=True)[0]
        for row in indices.tolist():
            others = indices[indices != row]
            candidates = weights[batch, row, others]
            cutoff = weights.new_zeros(())
            if candidates.numel() > 1:
                ordered = candidates.sort(descending=True, stable=True).values
                drops = (ordered[:-1] - ordered[1:]) / (ordered[:-1].abs() + epsilon)
                max_drop, location = drops.max(dim=0)
                max_drops[batch, row] = max_drop
                if max_drop > gap_tolerance:
                    cutoff = ordered[location]
            cutoffs[batch, row] = cutoff
            selected[batch, row, others] = (candidates > 0) & (candidates >= cutoff)
    selected = selected | selected.transpose(1, 2)
    affinity = torch.where(selected, weights, 0)
    diagonal = torch.eye(weights.shape[1], dtype=torch.bool, device=weights.device)[None]
    affinity = torch.where(diagonal & pair_mask(member_mask), 1.0, affinity)
    return PrunedRelations(affinity, cutoffs, max_drops, selected.sum(dim=-1))


def normalized_laplacian(
    affinity: Tensor, member_mask: Tensor | None = None, *, epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Return (D^(-1/2) A D^(-1/2), I_valid - A_hat); padding stays zero."""
    _check_epsilon(epsilon)
    safe, member_mask = _prepare_relations(affinity, member_mask)
    if safe.dtype in (torch.float16, torch.bfloat16):
        safe = safe.float()
    safe = (safe + safe.transpose(1, 2)) * 0.5
    inverse_sqrt_degree = safe.sum(dim=-1).clamp_min(epsilon).rsqrt()
    normalized = safe * inverse_sqrt_degree[:, :, None] * inverse_sqrt_degree[:, None, :]
    identity = torch.diag_embed(member_mask.to(dtype=safe.dtype))
    return normalized, identity - normalized


def _canonicalize(labels: Tensor) -> Tensor:
    """Number clusters by first member, without changing the partition."""
    mapping: dict[int, int] = {}
    values = [mapping.setdefault(value, len(mapping)) for value in labels.tolist()]
    return torch.tensor(values, dtype=torch.long)


def _repair_empty_clusters(labels: Tensor, embedding: Tensor, count: int) -> tuple[Tensor, bool]:
    """Split the largest cluster at its farthest member until exactly K remain."""
    labels = _canonicalize(labels)
    repaired = False
    while int(labels.max()) + 1 < count:
        sizes = torch.bincount(labels)
        donor = int(sizes.argmax())
        candidates = (labels == donor).nonzero(as_tuple=True)[0]
        features = embedding[candidates]
        distances = (features - features.mean(dim=0)).square().sum(dim=-1)
        chosen = candidates[distances.argmax()]
        labels[chosen] = len(sizes)
        repaired = True
    return _canonicalize(labels), repaired


@dataclass
class ClusteringTrace:
    """CPU float64 diagnostics; per-sample eigenvectors exclude padding."""

    raw_cosines: Tensor
    affinity: Tensor
    normalized_affinity: Tensor
    laplacian: Tensor
    cutoffs: Tensor
    max_relative_drops: Tensor
    retained_counts: Tensor
    eigenvalues: tuple[Tensor, ...]
    embeddings: tuple[Tensor, ...]


@dataclass
class ClusteringOutput:
    assignments: Tensor
    counts: Tensor
    requested_counts: Tensor
    member_mask: Tensor
    reasons: tuple[tuple[str, ...], ...]
    trace: ClusteringTrace | None = None


class SpectralClusterer(nn.Module):
    """Discrete inference with a fixed seed; shared input features stay attached."""

    def __init__(self, config: SegmentationConfig | None = None) -> None:
        super().__init__()
        self.config = config or SegmentationConfig()

    @torch.no_grad()
    def forward(
        self, nodes: Tensor, counts: Tensor, member_mask: Tensor | None = None, *,
        capture_trace: bool = False,
    ) -> ClusteringOutput:
        safe, member_mask = prepare_nodes(nodes, member_mask)
        if counts.shape != (safe.shape[0],) or counts.dtype not in (torch.int32, torch.int64):
            raise ValueError("counts must be an integer tensor with shape [B]")
        # The eigensolver and K-Means run on compact CPU matrices in float64.
        # No .detach() is applied to the caller's representation or encoder.
        cpu_nodes = safe.detach().to(device="cpu", dtype=torch.float64)
        cpu_mask = member_mask.cpu()
        raw = cosine_relations(cpu_nodes, cpu_mask, epsilon=self.config.epsilon)
        pruned = prune_cosine_relations(
            raw, cpu_mask, epsilon=self.config.epsilon, gap_tolerance=self.config.gap_tolerance,
        )
        normalized, laplacian = normalized_laplacian(
            pruned.affinity, cpu_mask, epsilon=self.config.epsilon,
        )
        requested = counts.detach().to(device="cpu", dtype=torch.long)
        assignments = torch.full(cpu_mask.shape, -1, dtype=torch.long)
        actual_counts = torch.zeros_like(requested)
        reasons: list[tuple[str, ...]] = []
        eigenvalues: list[Tensor] = []
        embeddings: list[Tensor] = []
        for batch in range(safe.shape[0]):
            indices = cpu_mask[batch].nonzero(as_tuple=True)[0]
            size = indices.numel()
            if size == 0:
                reasons.append(("empty_sample",))
                eigenvalues.append(torch.empty(0, dtype=torch.float64))
                embeddings.append(torch.empty((0, 0), dtype=torch.float64))
                continue
            count = min(max(int(requested[batch]), 1), size)
            sample_reasons: list[str] = []
            if count != int(requested[batch]):
                sample_reasons.append("count_clipped_to_members")
            compact_laplacian = laplacian[batch][indices][:, indices]
            failed = False
            try:
                values, vectors = torch.linalg.eigh(compact_laplacian)
                embedding = vectors[:, :count]
                embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(
                    self.config.epsilon,
                )
            except torch.linalg.LinAlgError:
                failed = True
                values = torch.empty(0, dtype=torch.float64)
                embedding = torch.zeros((size, count), dtype=torch.float64)
                sample_reasons.append("eigendecomposition_fallback")
            eigenvalues.append(values)
            embeddings.append(embedding)
            if count == 1:
                labels = torch.zeros(size, dtype=torch.long)
                sample_reasons.append("single_cluster")
            elif count == size:
                labels = torch.arange(size)
                sample_reasons.append("one_cluster_per_member")
            else:
                compact_affinity = pruned.affinity[batch][indices][:, indices]
                off_diagonal = compact_affinity[~torch.eye(size, dtype=torch.bool)]
                uninformative = (
                    off_diagonal.max() - off_diagonal.min() <= self.config.gap_tolerance
                )
                if failed or uninformative:
                    # Ties contain no evidence for a particular partition. This
                    # deterministic fallback still provides exactly K groups.
                    labels = torch.arange(size) % count
                    if uninformative:
                        sample_reasons.append("uniform_affinity_balanced_fallback")
                else:
                    kmeans = KMeans(
                        n_clusters=count, init="k-means++", n_init=self.config.kmeans_n_init,
                        max_iter=self.config.kmeans_max_iter,
                        random_state=self.config.clustering_seed,
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        labels = torch.from_numpy(kmeans.fit_predict(embedding.numpy())).long()
                    labels, repaired = _repair_empty_clusters(labels, embedding, count)
                    if repaired:
                        sample_reasons.append("empty_cluster_repair")
            assignments[batch, indices] = _canonicalize(labels)
            actual_counts[batch] = count
            reasons.append(tuple(sample_reasons))
        trace = None
        if capture_trace:
            trace = ClusteringTrace(
                raw, pruned.affinity, normalized, laplacian, pruned.cutoffs,
                pruned.max_relative_drops, pruned.retained_counts,
                tuple(eigenvalues), tuple(embeddings),
            )
        return ClusteringOutput(
            assignments.to(nodes.device), actual_counts.to(nodes.device),
            requested.to(nodes.device), member_mask, tuple(reasons), trace,
        )
