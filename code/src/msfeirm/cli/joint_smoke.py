"""Validate complete M3 and three loss paths, replaying an I-3a restore point."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

import torch

from msfeirm.config import (
    PROJECT_ROOT, LossConfig, ModelConfig, SegmentationConfig,
    load_loss_config, load_model_config, load_segmentation_config,
)
from msfeirm.data.schema import GroupBatch
from msfeirm.models import MsFEIRM
from msfeirm.models.losses import MsFEIRMJointLoss
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv


def _synthetic_targets(imu, mask, segmentation_config, num_classes, ignore_index):
    """Smoke supervision only; no learnable synthetic dataset or training loop."""
    ids = torch.full(mask.shape, -100, dtype=torch.long, device=imu.device)
    labels, counts = [], []
    for sample, valid in enumerate(mask):
        members = valid.nonzero(as_tuple=True)[0]
        count = min(members.numel(), (segmentation_config.k_min + segmentation_config.k_max) // 2)
        counts.append(count)
        if count:
            ids[sample, members] = torch.arange(members.numel(), device=imu.device) % count
        activities = (2 * torch.arange(count, device=imu.device) + sample) % num_classes
        if count > 1:
            activities[-1] = ignore_index
        labels.append(activities)
    pairs = mask[:, :, None] & mask[:, None, :]
    pair_targets = (ids[:, :, None] == ids[:, None, :]).to(imu.dtype).masked_fill(~pairs, float("nan"))
    return GroupBatch(
        imu, mask, tuple(f"smoke_{sample}" for sample in range(mask.shape[0])),
        group_ids=ids, group_labels=tuple(labels), pair_targets=pair_targets,
        k_clusters=torch.tensor(counts, dtype=torch.long, device=imu.device),
    )


def _save(values, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/models/m3.yaml")
    parser.add_argument("--segmentation-config", type=Path, default=PROJECT_ROOT / "configs/segmentation/default.yaml")
    parser.add_argument("--loss-config", type=Path, default=PROJECT_ROOT / "configs/training/loss.yaml")
    parser.add_argument("--segmentation-checkpoint", type=Path)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--members", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--padded-members", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="runs/joint_smoke")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if args.threads < 1:
        parser.error("threads must be positive")
    previous = None
    if args.segmentation_checkpoint is not None:
        if any(value is not None for value in (args.window_size, args.members, args.batch_size, args.padded_members)):
            parser.error("checkpoint mode replays the saved input; omit input shape options")
        previous = torch.load(args.segmentation_checkpoint, map_location="cpu", weights_only=True)
        model_config = ModelConfig.from_mapping(previous["model_config"])
        segmentation_config = SegmentationConfig.from_mapping(previous["segmentation_config"])
    else:
        model_config = load_model_config(args.config)
        if args.window_size is not None:
            model_config = replace(model_config, window_size=args.window_size)
        segmentation_config = load_segmentation_config(args.segmentation_config)
    loss_config = load_loss_config(args.loss_config)
    output_dir = project_output_path(args.output)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = MsFEIRM(model_config, segmentation_config).to(args.device).eval()
    if previous is not None:
        model.segmenter.load_state_dict(previous["model_state_dict"])
        imu = previous["smoke_imu"].to(args.device).clone().requires_grad_()
        mask = previous["member_mask"].to(args.device)
        with torch.no_grad():
            replay = model.segmenter(imu, mask)
        for observed, expected in (
            (replay.node_features, previous["expected_Y_norm"]),
            (replay.membership.probabilities, previous["expected_pair_probabilities"]),
            (replay.cardinality.score, previous["expected_cardinality_score"]),
        ):
            torch.testing.assert_close(observed.cpu(), expected, rtol=1e-5, atol=1e-6)
        assert torch.equal(replay.clustering.assignments.cpu(), previous["expected_assignments"])
    else:
        members = args.members if args.members is not None else 15
        batch_size = args.batch_size if args.batch_size is not None else 2
        padded = args.padded_members if args.padded_members is not None else 0
        if min(members, batch_size) < 1 or not 0 <= padded <= members:
            parser.error("members/batch-size must be positive and padded-members in [0,members]")
        mask = torch.ones(batch_size, members, dtype=torch.bool, device=args.device)
        if padded:
            mask[1:, -padded:] = False
        imu = torch.randn(batch_size, members, model_config.input_channels, model_config.window_size, device=args.device)
        imu = imu.masked_fill(~mask[:, :, None, None], float("nan")).requires_grad_()
    batch = _synthetic_targets(imu, mask, segmentation_config, model_config.num_classes, loss_config.ignore_index)
    criterion = MsFEIRMJointLoss(segmentation_config, loss_config)
    components = {
        "encoder": model.segmenter.encoder, "member": model.segmenter.membership_head,
        "cardinality": model.segmenter.cardinality_head, "group": model.activity_head,
    }
    gradient_paths = {}
    for branch in ("group", "member", "cardinality", "total"):
        model.zero_grad(set_to_none=True)
        imu.grad = None
        output = model(imu, mask)
        losses = criterion(output, batch)
        getattr(losses, branch).backward()
        branch_stats = {}
        for name, component in components.items():
            active = branch == "total" or name in ("encoder", branch)
            parameters = tuple(component.parameters())
            if active:
                finite = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
                absolute_sum = sum(float(p.grad.detach().abs().sum()) for p in parameters if p.grad is not None)
                if not finite:
                    raise RuntimeError(f"{branch} -> {name} has missing/non-finite gradients")
                meaningful = {
                    "group": losses.stats["classification_group_count"] > 0,
                    "member": bool(mask.any()),
                    "cardinality": bool(mask.any()) and segmentation_config.k_min != segmentation_config.k_max,
                    "total": bool(mask.any()),
                }[branch]
                if branch == "total" and name == "group":
                    meaningful = losses.stats["classification_group_count"] > 0
                # A fixed K range legitimately has a connected zero count loss.
                if name == "cardinality" and segmentation_config.k_min == segmentation_config.k_max:
                    meaningful = False
                if meaningful and absolute_sum <= 0:
                    raise RuntimeError(f"{branch} -> {name} has an unexpected zero gradient")
                branch_stats[name] = {"expected_path": True, "finite": True, "absolute_sum": absolute_sum}
            else:
                if any(p.grad is not None for p in parameters):
                    raise RuntimeError(f"Unexpected gradient path: {branch} -> {name}")
                branch_stats[name] = {"expected_path": False, "no_gradient": True}
        if imu.grad is None or not bool(torch.isfinite(imu.grad).all()):
            raise RuntimeError(f"{branch} has a missing/non-finite input gradient")
        branch_stats["input_gradient_finite"] = True
        gradient_paths[branch] = branch_stats

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "msfeirm_weights.pt"
    checkpoint = {
        "stage": "I-3b", "complete_m3": True, "trained": False,
        "checkpoint_type": "complete_model_untrained_component",
        "model_config": model_config.to_dict(), "segmentation_config": segmentation_config.to_dict(),
        "loss_config": loss_config.to_dict(),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "torch_rng_state": torch.get_rng_state(), "smoke_imu": imu.detach().cpu(), "member_mask": mask.cpu(),
        "synthetic_targets": {"group_ids": batch.group_ids.cpu(), "group_labels": tuple(value.cpu() for value in batch.group_labels),
                              "pair_targets": batch.pair_targets.cpu(), "k_clusters": batch.k_clusters.cpu()},
        "expected_activity_logits": output.activity.logits.detach().cpu(),
        "expected_pair_probabilities": output.segmentation.membership.probabilities.detach().cpu(),
        "expected_cardinality_score": output.segmentation.cardinality.score.detach().cpu(),
        "expected_assignments": output.assignments.cpu(),
        "expected_losses": torch.stack((losses.total, losses.group, losses.cardinality, losses.member)).detach().cpu(),
    }
    _save(checkpoint, checkpoint_path)
    state = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    restored = MsFEIRM(ModelConfig.from_mapping(state["model_config"]),
                       SegmentationConfig.from_mapping(state["segmentation_config"])).to(args.device).eval()
    restored.load_state_dict(state["model_state_dict"])
    restored_batch = GroupBatch(state["smoke_imu"], state["member_mask"], batch.sample_ids, **state["synthetic_targets"])
    with torch.no_grad():
        actual = restored(restored_batch.imu, restored_batch.member_mask)
        restored_losses = MsFEIRMJointLoss(restored.segmentation_config, LossConfig.from_mapping(state["loss_config"]))(actual, restored_batch)
    for observed, expected in (
        (actual.activity.logits, state["expected_activity_logits"]),
        (actual.segmentation.membership.probabilities, state["expected_pair_probabilities"]),
        (actual.segmentation.cardinality.score, state["expected_cardinality_score"]),
        (torch.stack((restored_losses.total, restored_losses.group, restored_losses.cardinality, restored_losses.member)), state["expected_losses"]),
    ):
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual.assignments, state["expected_assignments"])
    summary = {
        "stage": "I-3b", "component": "complete_m3_and_joint_losses", "trained": False,
        "mode": "eval_with_gradients", "device": args.device, "seed": args.seed,
        "segmentation_checkpoint": str(args.segmentation_checkpoint) if previous is not None else None,
        "segmentation_restore_verified": previous is not None, "checkpoint_roundtrip": True,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "valid_members": mask.sum(-1).tolist(),
        "shapes": {"imu": list(imu.shape), "Y_norm": list(output.node_features.shape),
                   "pooled": list(output.activity.pooled.features.shape), "activity_logits": list(output.activity.logits.shape)},
        "predicted_counts": output.segmentation.cardinality.decoded_count.tolist(),
        "synthetic_losses_and_counts": losses.detached_metrics(), "gradient_paths": gradient_paths,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }
    for filename, values in (
        ("shape_trace.json", summary),
        ("config.json", {"model": model_config.to_dict(), "segmentation": segmentation_config.to_dict(), "loss": loss_config.to_dict()}),
        ("environment.json", environment_info()),
    ):
        (output_dir / filename).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", output_dir / "requirements.txt")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
