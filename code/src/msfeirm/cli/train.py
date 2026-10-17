"""Train complete MsFEIRM on a frozen UT/WB cache with resumable state."""

import argparse
import csv
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import shutil

import yaml

from msfeirm.config import PROJECT_ROOT, load_yaml
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv
from msfeirm.training.checkpoint import file_fingerprint
from msfeirm.training.real import RealDataConfig, RealTrainer, RealTrainingConfig
from msfeirm.training.real_verification import verify_real_training_resume
from msfeirm.training.synthetic import StopRequest
from msfeirm.training.verification import _assert_exact
from msfeirm.training.timing import TimingRecorder, profile_training


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def export_curves(journal, directory):
    columns = ("event", "global_step", "epoch", "loss_total", "loss_group", "loss_cardinality", "loss_member", "cardinality_mae")
    with journal.open() as source, (directory / "training_curve.csv").open("w", newline="") as training, (directory / "validation_curve.csv").open("w", newline="") as validation:
        writers = {"train": csv.DictWriter(training, columns, extrasaction="ignore"),
                   "validation": csv.DictWriter(validation, columns, extrasaction="ignore")}
        for writer in writers.values():
            writer.writeheader()
        for line in source:
            record = json.loads(line)
            writers["train" if record["event"] == "train" else "validation"].writerow(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("m3",), default="m3")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output")
    parser.add_argument("--steps", type=int, help="Total optimizer steps; default is the configured epoch target")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-limit", type=int, help="Development subset; persisted in checkpoint")
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-wall-time-min", type=float, default=210)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--verify-resume", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Measure training/validation/save/clustering outside checkpoint state")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if args.log_every < 1:
        parser.error("log-every must be positive")
    if args.resume:
        if any(value is not None for value in (args.config,args.epochs,args.train_limit,args.validation_limit,args.seed,args.device,args.threads)):
            parser.error("Resume uses its saved configuration; omit configuration overrides")
        source = project_output_path(args.resume)
        if args.output is None and source.parent.name != "checkpoints":
            parser.error("Nonstandard checkpoint paths require --output")
        directory = project_output_path(args.output) if args.output else source.parent.parent
        if directory.exists() and any(directory.iterdir()) and directory != source.parent.parent:
            parser.error("A different resume directory must be empty")
        trainer = RealTrainer.from_checkpoint(source, journal_path=directory / "metrics.jsonl")
    else:
        values = load_yaml(args.config or PROJECT_ROOT / "configs/training/wb_m3.yaml")
        data = RealDataConfig(**values.get("data", {}))
        training = RealTrainingConfig(**values.get("training", {}))
        data = replace(data, **{name:getattr(args,name) for name in ("train_limit","validation_limit") if getattr(args,name) is not None})
        training = replace(training, **{name:getattr(args,name) for name in ("epochs","seed","device","threads") if getattr(args,name) is not None})
        directory = project_output_path(args.output or "runs/real_training")
        if directory.exists() and any(directory.iterdir()):
            parser.error("Output is not empty; use --resume or a new directory")
        trainer = RealTrainer(data, training, model_config=values.get("model"),
                              segmentation_config=values.get("segmentation"), loss_config=values.get("loss"),
                              journal_path=directory / "metrics.jsonl")
    target = trainer.target_steps if args.steps is None else args.steps
    if not trainer.global_step <= target <= trainer.target_steps:
        parser.error("steps must be between saved progress and configured epoch target")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.resolved.yaml").write_text(yaml.safe_dump(trainer.specification(), sort_keys=False))
    _json(directory / "environment.json", environment_info())
    _json(directory / "dataset.manifest.json", trainer.data_manifest)
    _json(directory / "provenance.json", {"fingerprints":trainer.fingerprints})
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", directory / "requirements.txt")
    checkpoint = directory / "checkpoints/last.pt"
    def log(record):
        if record["global_step"] % args.log_every == 0:
            print(json.dumps({key:record[key] for key in ("global_step","epoch","loss_total")}), flush=True)
    stop = StopRequest()
    timing = TimingRecorder(trainer.config.device) if args.profile else None
    with stop.signal_handlers(), (profile_training(trainer,timing) if timing is not None else nullcontext()):
        result = trainer.train_until(target, checkpoint=checkpoint, max_wall_seconds=args.max_wall_time_min*60,
                                     stop_request=stop, on_step=log)
    final = trainer.evaluate()
    restored = RealTrainer.from_checkpoint(checkpoint)
    if restored.evaluate() != final:
        raise AssertionError("Saved model validation did not recover exactly")
    expected, actual = trainer.state_dict(), restored.state_dict()
    for name in ("model_state_dict","optimizer_state_dict","sampler_state","random_state"):
        _assert_exact(actual[name],expected[name])
    export_curves(directory / "metrics.jsonl", directory)
    summary = {"stage":"II-2", "scope":"real_training", "complete_m3":True,
               "development_subset":trainer.data_config.train_limit is not None or trainer.data_config.validation_limit is not None,
               **result, "initial_validation":trainer.initial_validation, "final_validation":final,
               "best_validation":trainer.best_metric, "last_validation":trainer.last_validation,
               "checkpoint":str(checkpoint.relative_to(PROJECT_ROOT)), "checkpoint_sha256":file_fingerprint(checkpoint),
               "checkpoint_roundtrip":True, "fingerprints":trainer.fingerprints,
               "last_step":trainer.last_record}
    if timing is not None:
        summary["timing"] = timing.summary()
    if args.verify_resume:
        remaining = trainer.target_steps - trainer.global_step
        summary["resume_verification"] = (verify_real_training_resume(checkpoint,directory/"resume_verification")
                                          if remaining >= 5 else {"status":"not_run", "reason":"fewer than five configured steps remain"})
    _json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
