"""Membership BCE, continuous cardinality MSE and matched-group cross entropy."""

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from msfeirm.config import LossConfig, SegmentationConfig
from msfeirm.data.schema import GroupBatch
from msfeirm.models.msfeirm import MsFEIRMOutput
from msfeirm.training.matching import BatchMatching, match_subgroups


@dataclass
class JointLossOutput:
    total: Tensor
    group: Tensor
    cardinality: Tensor
    member: Tensor
    matching: BatchMatching
    stats: dict[str, Any]

    def detached_metrics(self) -> dict[str, Any]:
        return {
            "loss_total": float(self.total.detach()), "loss_group": float(self.group.detach()),
            "loss_cardinality": float(self.cardinality.detach()), "loss_member": float(self.member.detach()),
            **self.stats,
        }


class MsFEIRMJointLoss(nn.Module):
    def __init__(
        self, segmentation_config: SegmentationConfig | None = None,
        loss_config: LossConfig | None = None,
    ) -> None:
        super().__init__()
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.config = loss_config or LossConfig()

    def forward(self, output: MsFEIRMOutput, batch: GroupBatch) -> JointLossOutput:
        for name in ("group_ids", "group_labels", "pair_targets", "k_clusters"):
            if getattr(batch, name) is None:
                raise ValueError(f"Joint loss requires batch.{name}")
        if batch.member_mask.shape != output.member_mask.shape or not torch.equal(batch.member_mask, output.member_mask):
            raise ValueError("Batch and model output must use the same member_mask")
        membership = output.segmentation.membership
        cardinality = output.segmentation.cardinality
        if batch.pair_targets.shape != membership.logits.shape:
            raise ValueError("pair_targets must match the shape [B, N, N] of pair logits")
        if batch.k_clusters.shape != cardinality.score.shape:
            raise ValueError("k_clusters must have shape [B]")
        pair_targets = batch.pair_targets.to(device=membership.logits.device, dtype=membership.logits.dtype)
        # Remove masked target NaNs before BCE, so their backward terms are also
        # finite. This is mask handling, not a target-content inspection.
        pair_targets = torch.where(membership.valid_pairs, pair_targets, 0)
        pair_terms = F.binary_cross_entropy_with_logits(membership.logits, pair_targets, reduction="none")
        pair_counts = membership.valid_pairs.sum(dim=(1, 2))
        member_sums = torch.where(membership.valid_pairs, pair_terms, 0).sum(dim=(1, 2))
        member_samples = pair_counts > 0
        member = (member_sums / pair_counts.clamp_min(1)).sum() / member_samples.sum().clamp_min(1)

        counts = batch.k_clusters.to(device=cardinality.score.device, dtype=cardinality.score.dtype)
        counts = torch.where(cardinality.sample_mask, counts, 0)
        span = self.segmentation_config.k_max - self.segmentation_config.k_min
        normalized_counts = (counts - self.segmentation_config.k_min) / span if span else counts * 0
        normalized_counts = torch.where(cardinality.sample_mask, normalized_counts, 0)
        count_terms = (cardinality.score - normalized_counts).square()
        cardinality_loss = torch.where(cardinality.sample_mask, count_terms, 0).sum()
        cardinality_loss = cardinality_loss / cardinality.sample_mask.sum().clamp_min(1)

        matching = match_subgroups(
            output.activity.pooled, batch.group_ids, batch.group_labels,
            ignore_index=self.config.ignore_index,
        )
        group_sum = output.activity.logits.sum() * 0
        for sample, matches in enumerate(matching.samples):
            if matches.classification_count:
                group_sum = group_sum + F.cross_entropy(
                    output.activity.logits[sample, matches.class_pred_indices], matches.class_targets,
                    reduction="sum",
                )
        matching_counts = matching.counts()
        classification_count = matching_counts["classification_group_count"]
        group_loss = group_sum / max(classification_count, 1)
        total = group_loss + self.config.lambda_cardinality * cardinality_loss + self.config.lambda_member * member
        mae_terms = (cardinality.continuous_count.detach() - counts).abs()
        mae = torch.where(cardinality.sample_mask, mae_terms, 0).sum() / cardinality.sample_mask.sum().clamp_min(1)
        stats = {
            **matching_counts,
            "matched_gt_coverage": matching_counts["matched_gt_count"] / max(matching_counts["target_group_count"], 1),
            "member_pairs_per_sample": pair_counts.tolist(),
            "member_sample_count": int(member_samples.sum()),
            "cardinality_sample_count": int(cardinality.sample_mask.sum()),
            "classification_sample_count": sum(sample.classification_count > 0 for sample in matching.samples),
            "no_classification_sample_count": sum(sample.classification_count == 0 for sample in matching.samples),
            "cardinality_mae": float(mae),
            "matching_per_sample": [sample.counts() for sample in matching.samples],
        }
        return JointLossOutput(total, group_loss, cardinality_loss, member, matching, stats)
