"""Pad variable member counts without inspecting target content."""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from msfeirm.data.schema import GroupBatch, GroupSample


def _optional(samples: Sequence[GroupSample], field: str) -> list | None:
    values = [getattr(sample, field) for sample in samples]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Field {field} must be present in every sample or omitted in all")
    return values


def _pad_members(values: list, counts: list[int], dtype: torch.dtype, fill=0) -> Tensor:
    tensors = [torch.as_tensor(value, dtype=dtype) for value in values]
    for tensor, count in zip(tensors, counts, strict=True):
        if tensor.ndim < 1 or tensor.shape[0] != count:
            raise ValueError("Member field must have the same N dimension as imu")
    return pad_sequence(tensors, batch_first=True, padding_value=fill)


def collate_groups(samples: Sequence[GroupSample]) -> GroupBatch:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    imu = [torch.as_tensor(sample.imu, dtype=torch.float32) for sample in samples]
    if any(tensor.ndim != 3 or tensor.shape[0] < 1 for tensor in imu):
        raise ValueError("Each imu tensor must have shape [N, C, T] with N >= 1")
    if any(tensor.shape[1:] != imu[0].shape[1:] for tensor in imu):
        raise ValueError("A batch must use the same C and T for all samples")
    if any(tensor.device != imu[0].device for tensor in imu):
        raise ValueError("All sample tensors must be on the same device")
    counts = [tensor.shape[0] for tensor in imu]
    max_members = max(counts)
    device = imu[0].device
    member_mask = torch.arange(max_members, device=device)[None, :] < torch.tensor(
        counts, device=device
    )[:, None]
    padded_imu = pad_sequence(imu, batch_first=True)

    group_ids = _optional(samples, "group_ids")
    noise_mask = _optional(samples, "noise_mask")
    positions = _optional(samples, "positions")
    group_labels = _optional(samples, "group_labels")
    k_clusters = _optional(samples, "k_clusters")
    pair_targets = _optional(samples, "pair_targets")
    padded_pairs = None
    if pair_targets is not None:
        padded_pairs = padded_imu.new_zeros((len(samples), max_members, max_members))
        for index, (value, count) in enumerate(zip(pair_targets, counts, strict=True)):
            tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
            if tensor.shape != (count, count):
                raise ValueError("pair_targets must have shape [N, N]")
            padded_pairs[index, :count, :count] = tensor

    return GroupBatch(
        imu=padded_imu,
        member_mask=member_mask,
        sample_ids=tuple(sample.sample_id for sample in samples),
        group_ids=None if group_ids is None else _pad_members(
            group_ids, counts, torch.int64, fill=-100
        ).to(device),
        group_labels=None if group_labels is None else tuple(
            torch.as_tensor(value, dtype=torch.int64, device=device) for value in group_labels
        ),
        pair_targets=padded_pairs,
        k_clusters=None if k_clusters is None else torch.tensor(
            k_clusters, dtype=torch.int64, device=device
        ),
        noise_mask=None if noise_mask is None else _pad_members(
            noise_mask, counts, torch.bool
        ).to(device),
        positions=None if positions is None else _pad_members(
            positions, counts, torch.float32
        ).to(device),
        source_refs=tuple(sample.source_refs for sample in samples),
    )
