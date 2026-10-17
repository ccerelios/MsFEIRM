"""Typed fields for variable-size sensor groups."""

from dataclasses import dataclass, fields, replace
from typing import Any

from torch import Tensor


@dataclass
class GroupSample:
    imu: Tensor                          # [N, C, T]
    sample_id: str = ""
    group_ids: Tensor | None = None       # [N]
    group_labels: Tensor | None = None    # [K], activity for group_id g is group_labels[g]
    pair_targets: Tensor | None = None    # [N, N]
    k_clusters: int | None = None
    noise_mask: Tensor | None = None      # [N]
    positions: Tensor | None = None       # [N, 2] or [N, T, 2]
    source_refs: Any = None


@dataclass
class GroupBatch:
    imu: Tensor                          # [B, N_max, C, T]
    member_mask: Tensor                  # [B, N_max], true for real members
    sample_ids: tuple[str, ...]
    group_ids: Tensor | None = None
    group_labels: tuple[Tensor, ...] | None = None
    pair_targets: Tensor | None = None
    k_clusters: Tensor | None = None
    noise_mask: Tensor | None = None
    positions: Tensor | None = None
    source_refs: tuple[Any, ...] = ()

    def to(self, device: str, *, non_blocking: bool = False) -> "GroupBatch":
        updates = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                updates[field.name] = value.to(device, non_blocking=non_blocking)
            elif field.name == "group_labels" and value is not None:
                updates[field.name] = tuple(
                    item.to(device, non_blocking=non_blocking) for item in value
                )
        return replace(self, **updates)
