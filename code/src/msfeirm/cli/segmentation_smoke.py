"""Validate the I-3a segmenter, optionally loading an I-2 encoder restore point."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import torch
from torch.nn import functional as F

from msfeirm.config import (
    PROJECT_ROOT, ModelConfig, SegmentationConfig, load_model_config, load_segmentation_config,
)
from msfeirm.models.segmentation import MsFEIRMSegmenter
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv


def _atomic_save(values: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(values, temporary)
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/models/m3.yaml")
    parser.add_argument("--segmentation-config", type=Path, default=PROJECT_ROOT / "configs/segmentation/default.yaml")
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--members", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--padded-members", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="runs/segmentation_smoke")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if min(args.members, args.batch_size, args.threads) < 1:
        parser.error("members, batch-size and threads must be positive")
    if not 0 <= args.padded_members <= args.members:
        parser.error("padded-members must be between zero and members")
    output_dir = project_output_path(args.output)
    previous = None
    if args.encoder_checkpoint is not None:
        previous = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=True)
        model_config = ModelConfig.from_mapping(previous["model_config"])
        if args.window_size is not None and args.window_size != model_config.window_size:
            parser.error("window-size must match the encoder checkpoint's adapter dimensions")
    else:
        model_config = load_model_config(args.config)
        if args.window_size is not None:
            model_config = replace(model_config, window_size=args.window_size)
    segmentation_config = load_segmentation_config(args.segmentation_config)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = MsFEIRMSegmenter(model_config, segmentation_config).to(args.device).eval()
    if previous is not None:
        model.encoder.load_state_dict(previous["model_state_dict"])
        with torch.no_grad():
            actual = model.encoder(
                previous["smoke_imu"].to(args.device), previous["member_mask"].to(args.device),
            ).node_features
        torch.testing.assert_close(actual.cpu(), previous["expected_Y_norm"], rtol=1e-5, atol=1e-6)

    mask = torch.ones(args.batch_size, args.members, dtype=torch.bool, device=args.device)
    if args.padded_members:
        mask[1:, -args.padded_members:] = False
    imu = torch.randn(
        args.batch_size, args.members, model_config.input_channels, model_config.window_size,
        device=args.device,
    ).masked_fill(~mask[:, :, None, None], float("nan")).requires_grad_()
    valid_counts = mask.sum(dim=1)
    synthetic_k = valid_counts.clamp(max=(segmentation_config.k_min + segmentation_config.k_max) // 2)
    synthetic_groups = torch.arange(args.members, device=args.device)[None] % synthetic_k.clamp_min(1)[:, None]
    pair_targets = (synthetic_groups[:, :, None] == synthetic_groups[:, None, :]).to(imu.dtype)
    started = time.monotonic()
    output = model(imu, mask, capture_trace=True)
    # These two objectives validate gradients only. Production joint loss, group
    # matching and activity recognition belong to I-3b.
    pair_terms = F.binary_cross_entropy_with_logits(output.membership.logits, pair_targets, reduction="none")
    per_sample_pair = torch.where(output.membership.valid_pairs, pair_terms, 0).sum(dim=(1, 2))
    per_sample_pair = per_sample_pair / valid_counts.square().clamp_min(1)
    sample_mask = output.cardinality.sample_mask
    member_loss = per_sample_pair.sum() / sample_mask.sum().clamp_min(1)
    count_terms = (output.cardinality.score - model.cardinality_head.normalize_targets(synthetic_k)).square()
    count_loss = torch.where(sample_mask, count_terms, 0).sum() / sample_mask.sum().clamp_min(1)
    (member_loss + count_loss).backward()
    finite_parameters = all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    finite_input = imu.grad is not None and bool(torch.isfinite(imu.grad).all())
    if not finite_parameters or not finite_input:
        raise RuntimeError("Segmenter backward pass produced missing/non-finite gradients")
    clustering = output.clustering
    assert clustering is not None and clustering.trace is not None
    trace = clustering.trace
    summary = {
        "stage": "I-3a", "component": "encoder_and_segmentation", "complete_m3": False,
        "mode": "eval_with_gradients", "device": args.device, "seed": args.seed,
        "encoder_checkpoint": str(args.encoder_checkpoint) if previous is not None else None,
        "encoder_restore_verified": previous is not None,
        "valid_members": valid_counts.tolist(),
        "shapes": {
            "imu": list(imu.shape), "Y_norm": list(output.node_features.shape),
            "pair_logits": list(output.membership.logits.shape),
            "cardinality_score": list(output.cardinality.score.shape),
            "raw_cosines": list(trace.raw_cosines.shape), "affinity": list(trace.affinity.shape),
            "laplacian": list(trace.laplacian.shape),
            "embeddings": [list(value.shape) for value in trace.embeddings],
            "assignments": list(clustering.assignments.shape),
        },
        "cardinality_scores": output.cardinality.score.detach().tolist(),
        "continuous_counts": output.cardinality.continuous_count.detach().tolist(),
        "predicted_counts": output.cardinality.decoded_count.tolist(),
        "actual_cluster_counts": clustering.counts.tolist(),
        "clustering_reasons": clustering.reasons,
        "synthetic_member_loss": float(member_loss.detach()),
        "synthetic_cardinality_loss": float(count_loss.detach()),
        "finite_parameter_gradients": finite_parameters, "finite_input_gradient": finite_input,
        "gradient_absolute_sums": {
            name: float(sum(p.grad.detach().abs().sum() for p in component.parameters()))
            for name, component in (
                ("encoder", model.encoder), ("membership_head", model.membership_head),
                ("cardinality_head", model.cardinality_head),
            )
        },
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "forward_backward_seconds": time.monotonic() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "segmentation_weights.pt"
    checkpoint = {
        "stage": "I-3a", "complete_m3": False,
        "model_config": model_config.to_dict(), "segmentation_config": segmentation_config.to_dict(),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "torch_rng_state": torch.get_rng_state(), "smoke_imu": imu.detach().cpu(),
        "member_mask": mask.cpu(), "expected_Y_norm": output.node_features.detach().cpu(),
        "expected_pair_probabilities": output.membership.probabilities.detach().cpu(),
        "expected_cardinality_score": output.cardinality.score.detach().cpu(),
        "expected_assignments": clustering.assignments.cpu(), "expected_counts": clustering.counts.cpu(),
    }
    _atomic_save(checkpoint, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    restored = MsFEIRMSegmenter(
        ModelConfig.from_mapping(loaded["model_config"]),
        SegmentationConfig.from_mapping(loaded["segmentation_config"]),
    ).to(args.device).eval()
    restored.load_state_dict(loaded["model_state_dict"])
    with torch.no_grad():
        actual = restored(loaded["smoke_imu"], loaded["member_mask"])
    for observed, expected in (
        (actual.node_features, loaded["expected_Y_norm"]),
        (actual.membership.probabilities, loaded["expected_pair_probabilities"]),
        (actual.cardinality.score, loaded["expected_cardinality_score"]),
    ):
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
    assert torch.equal(actual.clustering.assignments, loaded["expected_assignments"])
    assert torch.equal(actual.clustering.counts, loaded["expected_counts"])
    summary["checkpoint_roundtrip"] = True
    summary["checkpoint"] = str(checkpoint_path.relative_to(PROJECT_ROOT))
    _atomic_save({
        "raw_cosines": trace.raw_cosines, "affinity": trace.affinity,
        "normalized_affinity": trace.normalized_affinity, "laplacian": trace.laplacian,
        "cutoffs": trace.cutoffs, "max_relative_drops": trace.max_relative_drops,
        "retained_counts": trace.retained_counts, "eigenvalues": trace.eigenvalues,
        "embeddings": trace.embeddings, "reasons": clustering.reasons,
    }, output_dir / "clustering_trace.pt")
    for filename, values in (
        ("shape_trace.json", summary),
        ("config.json", {"model": model_config.to_dict(), "segmentation": segmentation_config.to_dict()}),
        ("environment.json", environment_info()),
    ):
        (output_dir / filename).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", output_dir / "requirements.txt")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
