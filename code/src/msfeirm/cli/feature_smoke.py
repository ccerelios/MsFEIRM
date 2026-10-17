"""Run only the implemented individual feature stage on synthetic input."""

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import torch

from msfeirm.config import PROJECT_ROOT, load_model_config
from msfeirm.models.multiscale import MultiScaleFeatureExtractor
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/models/m3.yaml")
    parser.add_argument("--window-size", type=int)
    parser.add_argument("--members", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", default="runs/feature_smoke")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if min(args.members, args.batch_size, args.threads) < 1:
        parser.error("members, batch-size and threads must be positive")
    output_dir = project_output_path(args.output)
    config = load_model_config(args.config)
    if args.window_size is not None:
        config = replace(config, window_size=args.window_size)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = MultiScaleFeatureExtractor(config).to(args.device)
    imu = torch.randn(
        args.batch_size, args.members, config.input_channels, config.window_size,
        device=args.device, requires_grad=True,
    )
    started = time.monotonic()
    output = model(imu, capture_trace=True)
    loss = output.node_features.square().mean() + output.refined_maps.square().mean()
    loss.backward()
    # Read back tensors to synchronize device work before measuring elapsed time.
    finite_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    finite_input_gradient = imu.grad is not None and bool(torch.isfinite(imu.grad).all())
    if not finite_gradients or not finite_input_gradient:
        raise RuntimeError("Feature-stage backward pass produced missing/non-finite gradients")
    assert output.cascade_trace is not None
    summary = {
        "stage": "I-1",
        "component": "individual_feature_extractor",
        "device": args.device,
        "seed": args.seed,
        "shapes": {
            "imu": list(imu.shape),
            "expanded": [args.batch_size * args.members, config.expanded_channels, 1, config.window_size],
            "cascade_hidden": [list(tensor.shape) for tensor in output.cascade_trace.hidden_states],
            "refined_maps": list(output.refined_maps.shape),
            "scale_vectors": [list(tensor.shape) for tensor in output.scale_vectors],
            "node_features": list(output.node_features.shape),
            "channel_weights": list(output.channel_weights.shape),
        },
        "scale_slices": [[interval.start, interval.stop] for interval in output.scale_slices],
        "vector_slices": [[interval.start, interval.stop] for interval in output.vector_slices],
        "synthetic_loss": float(loss.detach()),
        "finite_parameter_gradients": finite_gradients,
        "finite_input_gradient": finite_input_gradient,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "forward_backward_seconds": time.monotonic() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (
        ("shape_trace.json", summary),
        ("config.json", config.to_dict()),
        ("environment.json", environment_info()),
    ):
        (output_dir / name).write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    requirements = PROJECT_ROOT / "requirements.txt"
    if requirements.is_file():
        shutil.copyfile(requirements, output_dir / "requirements.txt")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
