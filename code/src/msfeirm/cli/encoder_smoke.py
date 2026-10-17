"""Exercise the feature + interaction encoder and save an encoder-only restore point."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import torch

from msfeirm.config import PROJECT_ROOT, ModelConfig, load_model_config
from msfeirm.models.encoder import MsFEIRMEncoder
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/models/m3.yaml")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--members", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--padded-members", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="runs/encoder_smoke")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if min(args.members, args.batch_size, args.threads) < 1:
        parser.error("members, batch-size and threads must be positive")
    if not 0 <= args.padded_members <= args.members:
        parser.error("padded-members must be between zero and members")
    output_dir = project_output_path(args.output)
    config = load_model_config(args.config)
    if args.window_size is not None:
        config = replace(config, window_size=args.window_size)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = MsFEIRMEncoder(config).to(args.device).eval()
    member_mask = torch.ones((args.batch_size, args.members), dtype=torch.bool, device=args.device)
    if args.padded_members:
        member_mask[1:, -args.padded_members:] = False
    imu = torch.randn(
        args.batch_size, args.members, config.input_channels, config.window_size, device=args.device,
    ).masked_fill(~member_mask[:, :, None, None], float("nan")).requires_grad_()
    target = torch.randn(args.batch_size, args.members, config.node_dim, device=args.device)
    started = time.monotonic()
    output = model(imu, member_mask, capture_trace=True)
    squared_error = (output.node_features - target).square()
    loss = torch.where(member_mask[:, :, None], squared_error, 0).sum()
    loss = loss / (member_mask.sum() * config.node_dim)
    loss.backward()
    finite_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    finite_input_gradient = imu.grad is not None and bool(torch.isfinite(imu.grad).all())
    if not finite_gradients or not finite_input_gradient:
        raise RuntimeError("Encoder backward pass produced missing/non-finite gradients")
    trace = output.interaction.trace
    assert trace is not None
    summary = {
        "stage": "I-2",
        "component": "feature_and_interaction_encoder",
        "mode": "eval_with_gradients",
        "device": args.device,
        "seed": args.seed,
        "valid_members": member_mask.sum(dim=1).tolist(),
        "shapes": {
            "imu": list(imu.shape),
            "scale_vectors": [list(value.shape) for value in output.individual.scale_vectors],
            "local_graphs": [list(value.shape) for value in trace.local_graphs],
            "local_features": [list(value.shape) for value in trace.local_features],
            "local_attention": [[list(value.shape) for value in stack] for stack in trace.local_attention],
            "concatenated_features": list(trace.concatenated_features.shape),
            "global_graph": list(trace.global_graph.shape),
            "global_attention": [list(value.shape) for value in trace.global_attention],
            "Y_norm": list(output.node_features.shape),
        },
        "synthetic_loss": float(loss.detach()),
        "finite_parameter_gradients": finite_gradients,
        "finite_input_gradient": finite_input_gradient,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_backward_seconds": time.monotonic() - started,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "encoder_weights.pt"
    checkpoint = {
        "stage": "I-2",
        "complete_m3": False,
        "model_config": config.to_dict(),
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "torch_rng_state": torch.get_rng_state(),
        "smoke_imu": imu.detach().cpu(),
        "member_mask": member_mask.detach().cpu(),
        "expected_Y_norm": output.node_features.detach().cpu(),
    }
    temporary_path = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    restored = MsFEIRMEncoder(ModelConfig.from_mapping(loaded["model_config"])).to(args.device).eval()
    restored.load_state_dict(loaded["model_state_dict"])
    with torch.no_grad():
        actual = restored(loaded["smoke_imu"], loaded["member_mask"]).node_features
    torch.testing.assert_close(actual, loaded["expected_Y_norm"], rtol=1e-5, atol=1e-6)
    summary["checkpoint_roundtrip"] = True
    summary["checkpoint"] = str(checkpoint_path.relative_to(PROJECT_ROOT))
    for filename, values in (
        ("shape_trace.json", summary),
        ("config.json", config.to_dict()),
        ("environment.json", environment_info()),
    ):
        (output_dir / filename).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", output_dir / "requirements.txt")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
