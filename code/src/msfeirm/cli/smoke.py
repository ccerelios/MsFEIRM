"""Run the I-4 learnable synthetic M3 experiment with resumable training."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil

import torch
import yaml

from msfeirm.config import PROJECT_ROOT, LossConfig, ModelConfig, SegmentationConfig, load_yaml
from msfeirm.data import collate_groups
from msfeirm.data.synthetic import SyntheticDataConfig
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv
from msfeirm.training.checkpoint import file_fingerprint
from msfeirm.training.synthetic import StopRequest, SyntheticTrainer, SyntheticTrainingConfig
from msfeirm.training.verification import verify_training_resume


def _json(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("m3",), default="m3")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=800, help="Total optimizer steps, including saved steps")
    parser.add_argument("--max-wall-time-min", type=float, default=210)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if not args.synthetic:
        parser.error("This work package supports --synthetic only")
    if args.log_every < 1:
        parser.error("log-every must be positive")
    if args.resume is not None:
        if any(value is not None for value in (args.config, args.initialize_from, args.device, args.threads)):
            parser.error("Resume uses the saved configuration; omit configuration overrides")
        source = project_output_path(args.resume)
        if args.output is None and source.parent.name != "checkpoints":
            parser.error("Nonstandard checkpoint paths require --output")
        directory = project_output_path(args.output) if args.output else source.parent.parent
        if directory.exists() and any(directory.iterdir()) and directory != source.parent.parent:
            parser.error("A different resume output directory must be empty")
        trainer = SyntheticTrainer.from_checkpoint(source)
    else:
        directory = project_output_path(args.output or "runs/synthetic_smoke")
        if directory.exists() and any(directory.iterdir()):
            parser.error("Output is not empty; use --resume or a new output directory")
        values = load_yaml(args.config or PROJECT_ROOT / "configs/synthetic/i4_wb.yaml")
        model_config = ModelConfig.from_mapping(values.get("model", {}))
        segmentation_config = SegmentationConfig.from_mapping(values.get("segmentation", {}))
        training_config = SyntheticTrainingConfig(**values.get("training", {}))
        if args.device is not None:
            training_config = replace(training_config, device=args.device)
        if args.threads is not None:
            training_config = replace(training_config, threads=args.threads)
        trainer = SyntheticTrainer(model_config, segmentation_config, LossConfig.from_mapping(values.get("loss", {})),
                                   SyntheticDataConfig(**values.get("synthetic", {})), training_config,
                                   initialize_from=args.initialize_from)
    if args.steps < trainer.global_step:
        parser.error("steps must be at least the saved global_step")
    if trainer.initial_evaluation is None:
        trainer.initial_evaluation = {"global_step": trainer.global_step, "metrics": trainer.evaluate()}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.resolved.yaml").write_text(yaml.safe_dump(trainer.specification(), sort_keys=False), encoding="utf-8")
    _json(directory / "environment.json", environment_info())
    _json(directory / "dataset.manifest.json", trainer.dataset.manifest())
    _json(directory / "provenance.json", {"fingerprints": trainer.fingerprints, "initialization": trainer.initialization})
    _json(directory / "initial_evaluation.json", trainer.initial_evaluation)
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", directory / "requirements.txt")
    metrics_path = directory / "metrics.jsonl"
    if metrics_path.exists() and args.resume is not None:
        shutil.copyfile(metrics_path, directory / "metrics.previous.jsonl")
    with metrics_path.open("w", encoding="utf-8") as stream:
        for record in trainer.history:
            stream.write(json.dumps(record) + "\n")
    checkpoint = directory / "checkpoints/last.pt"
    stop = StopRequest()
    def log(record):
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        if record["global_step"] % args.log_every == 0:
            print(json.dumps({key: record[key] for key in ("global_step", "epoch", "loss_total", "loss_group", "loss_cardinality", "loss_member")}), flush=True)
    with stop.signal_handlers():
        result = trainer.train_until(args.steps, checkpoint=checkpoint,
                                     max_wall_seconds=args.max_wall_time_min * 60, stop_request=stop, on_step=log)
    final = trainer.evaluate()
    restored = SyntheticTrainer.from_checkpoint(checkpoint)
    if restored.evaluate() != final:
        raise AssertionError("Saved model evaluation did not recover exactly")
    batch = collate_groups(trainer.dataset.samples[:2]).to(trainer.config.device)
    trainer.model.eval()
    restored.model.eval()
    with torch.no_grad():
        expected = trainer.model(batch.imu, batch.member_mask)
        actual = restored.model(batch.imu, batch.member_mask)
    torch.testing.assert_close(actual.activity.logits, expected.activity.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.segmentation.membership.probabilities, expected.segmentation.membership.probabilities, rtol=0, atol=0)
    torch.testing.assert_close(actual.segmentation.cardinality.score, expected.segmentation.cardinality.score, rtol=0, atol=0)
    if not torch.equal(actual.assignments, expected.assignments):
        raise AssertionError("Saved partition did not recover exactly")
    columns = ("global_step", "epoch", "loss_total", "loss_group", "loss_cardinality", "loss_member", "cardinality_mae")
    with (directory / "training_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trainer.history)
    initial = trainer.initial_evaluation["metrics"]
    summary = {
        "stage": "I-4", "scope": "synthetic_training_only", "complete_m3": True,
        **result, "initial_evaluation_step": trainer.initial_evaluation["global_step"],
        "initial": initial, "final": final,
        "losses_decreased": {key: final[key] < initial[key] for key in ("loss_total", "loss_group", "loss_cardinality", "loss_member")},
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)), "checkpoint_sha256": file_fingerprint(checkpoint),
        "checkpoint_roundtrip": True, "fingerprints": trainer.fingerprints,
        "shape_trace": {"imu": list(batch.imu.shape), "Y_norm": list(actual.node_features.shape),
                        "activity_logits": list(actual.activity.logits.shape)},
    }
    if args.verify_resume:
        summary["resume_verification"] = verify_training_resume(checkpoint, directory / "resume_verification")
    _json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
