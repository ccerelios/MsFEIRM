"""Maximum-IoU set matching for activity supervision on predicted groups."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment

from msfeirm.models.pooling import PooledSubgroups


@dataclass
class SampleMatching:
    pred_indices: Tensor           # Pooling slot indices of positive-overlap matches
    target_indices: Tensor         # Target group IDs, indexing group_labels
    ious: Tensor
    class_pred_indices: Tensor     # Matches whose activity target is not ignore_index
    class_targets: Tensor
    iou_matrix: Tensor              # CPU float64, canonical member-set order
    iou_pred_indices: Tensor        # Pooling slots corresponding to iou_matrix rows
    iou_target_indices: Tensor      # Target IDs corresponding to iou_matrix columns
    predicted_count: int
    target_count: int

    @property
    def matched_gt_count(self) -> int:
        return self.pred_indices.numel()

    @property
    def unmatched_pred_count(self) -> int:
        return self.predicted_count - self.matched_gt_count

    @property
    def unmatched_gt_count(self) -> int:
        return self.target_count - self.matched_gt_count

    @property
    def classification_count(self) -> int:
        return self.class_pred_indices.numel()

    def counts(self) -> dict[str, int]:
        return {
            "predicted_group_count": self.predicted_count,
            "target_group_count": self.target_count,
            "matched_gt_count": self.matched_gt_count,
            "unmatched_pred_count": self.unmatched_pred_count,
            "unmatched_gt_count": self.unmatched_gt_count,
            "classification_group_count": self.classification_count,
            "ignored_matched_group_count": self.matched_gt_count - self.classification_count,
        }


@dataclass
class BatchMatching:
    samples: tuple[SampleMatching, ...]

    def counts(self) -> dict[str, int]:
        values = [sample.counts() for sample in self.samples]
        return {key: sum(value[key] for value in values) for key in values[0]}


def _member_order(member_sets: Tensor) -> Tensor:
    # Nonempty disjoint sets have distinct first members. Canonicalizing both
    # sides makes tie resolution independent of arbitrary group ID numbering.
    return member_sets.to(torch.int64).argmax(dim=-1).argsort(stable=True)


@torch.no_grad()
def match_subgroups(
    pooled: PooledSubgroups, target_group_ids: Tensor,
    group_labels: Sequence[Tensor], *, ignore_index: int = -100,
) -> BatchMatching:
    """Assign maximum total IoU, discarding zero-overlap assignments.

    group_labels[b][g] is the activity of target group ID g. Targets are consumed
    as supplied; empty target sets are simply absent from the set assignment.
    Groups with ignore_index still participate in set matching and its counts,
    then do not contribute activity cross entropy. There is no IoU threshold
    for training beyond a positive intersection; evaluation is a separate API.
    """
    if target_group_ids.shape != pooled.member_mask.shape or target_group_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("target_group_ids must be an integer tensor with shape [B, N]")
    if len(group_labels) != pooled.features.shape[0]:
        raise ValueError("group_labels must contain one 1-D tensor per sample")
    device = pooled.features.device
    target_ids = target_group_ids.detach().cpu()
    member_mask = pooled.member_mask.cpu()
    samples = []
    for batch, labels in enumerate(group_labels):
        if labels.ndim != 1 or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError("Each group_labels entry must be a 1-D integer tensor")
        labels = labels.detach().to(device="cpu", dtype=torch.long)
        pred_indices = pooled.group_mask[batch].nonzero(as_tuple=True)[0].cpu()
        predicted_sets = pooled.member_sets[batch].cpu()[pred_indices]
        pred_order = _member_order(predicted_sets)
        predicted_sets, pred_indices = predicted_sets[pred_order], pred_indices[pred_order]
        target_indices = torch.arange(labels.numel())
        target_sets = (target_indices[:, None] == target_ids[batch][None, :]) & member_mask[batch][None, :]
        nonempty = target_sets.any(dim=-1)
        target_sets, target_indices = target_sets[nonempty], target_indices[nonempty]
        target_order = _member_order(target_sets)
        target_sets, target_indices = target_sets[target_order], target_indices[target_order]
        intersections = (predicted_sets[:, None, :] & target_sets[None, :, :]).sum(dim=-1)
        unions = predicted_sets.sum(-1)[:, None] + target_sets.sum(-1)[None, :] - intersections
        iou = intersections.double() / unions.clamp_min(1)
        if iou.numel():
            pred_rows, target_columns = linear_sum_assignment(iou.numpy(), maximize=True)
            pred_rows = torch.from_numpy(pred_rows).long()
            target_columns = torch.from_numpy(target_columns).long()
            positive = iou[pred_rows, target_columns] > 0
            pred_rows, target_columns = pred_rows[positive], target_columns[positive]
        else:
            pred_rows = target_columns = torch.empty(0, dtype=torch.long)
        matched_pred, matched_target = pred_indices[pred_rows], target_indices[target_columns]
        matched_labels = labels[matched_target]
        usable = matched_labels != ignore_index
        samples.append(SampleMatching(
            pred_indices=matched_pred.to(device), target_indices=matched_target.to(device),
            ious=iou[pred_rows, target_columns].to(device=device, dtype=pooled.features.dtype),
            class_pred_indices=matched_pred[usable].to(device),
            class_targets=matched_labels[usable].to(device), iou_matrix=iou,
            iou_pred_indices=pred_indices, iou_target_indices=target_indices,
            predicted_count=pred_indices.numel(), target_count=target_indices.numel(),
        ))
    return BatchMatching(tuple(samples))
