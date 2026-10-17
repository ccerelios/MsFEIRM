"""Exact continuation checks in separate real-data training branches."""

import json
from pathlib import Path

from msfeirm.training.checkpoint import file_fingerprint
from msfeirm.training.real import RealTrainer
from msfeirm.training.verification import _assert_exact


def verify_real_training_resume(checkpoint, output_dir, *, steps=5, split_after=2):
    if not 0 < split_after < steps:
        raise ValueError("split_after must be strictly between zero and steps")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    reference = RealTrainer.from_checkpoint(checkpoint, journal_path=directory / "reference.metrics.jsonl")
    start = reference.global_step
    if start + steps > reference.target_steps:
        raise ValueError("Not enough configured training steps remain for resume verification")
    reference_records = []
    reference.train_until(start + steps, on_step=reference_records.append)
    expected = reference.state_dict()
    interrupted = RealTrainer.from_checkpoint(checkpoint, journal_path=directory / "interrupted.metrics.jsonl")
    actual_records = []
    interrupted.train_until(start + split_after, on_step=actual_records.append)
    midpoint = directory / "midpoint.pt"
    interrupted.save(midpoint)
    resumed = RealTrainer.from_checkpoint(midpoint)
    resumed.train_until(start + steps, on_step=actual_records.append)
    actual = resumed.state_dict()
    checked = ("model_state_dict", "optimizer_state_dict", "sampler_state", "random_state", "best_metric",
               "global_step", "last_validated_epoch", "last_record", "last_validation", "initial_validation")
    for name in checked:
        _assert_exact(actual[name], expected[name])
    _assert_exact(actual_records, reference_records)
    if expected["journal_state"] is not None:
        for name in ("bytes", "sha256"):
            _assert_exact(actual["journal_state"][name], expected["journal_state"][name])
    report = {"status": "passed", "source_checkpoint": str(checkpoint),
              "source_sha256": file_fingerprint(checkpoint), "start_step": start,
              "split_step": start + split_after, "end_step": start + steps,
              "midpoint_checkpoint": str(midpoint), "checked_state": list(checked),
              "tolerance": {"rtol": 0, "atol": 0}, "sample_order_and_step_losses_exact": True,
              "journal_content_exact": expected["journal_state"] is not None}
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
