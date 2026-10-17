"""Compare uninterrupted and interrupted continuations of a trained checkpoint."""

import json
from pathlib import Path

import torch

from msfeirm.training.checkpoint import file_fingerprint
from msfeirm.training.synthetic import SyntheticTrainer


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError("Checkpoint state keys differ")
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError("Checkpoint state sequences differ")
        for a, b in zip(left, right, strict=True):
            _assert_exact(a, b)
    elif left != right:
        raise AssertionError(f"Checkpoint values differ: {left!r}, {right!r}")


def verify_training_resume(checkpoint, output_dir, *, steps=3, split_after=1):
    if not 0 < split_after < steps:
        raise ValueError("split_after must be strictly between zero and steps")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    reference = SyntheticTrainer.from_checkpoint(checkpoint)
    start = reference.global_step
    reference.train_until(start + steps)
    expected = reference.state_dict()
    interrupted = SyntheticTrainer.from_checkpoint(checkpoint)
    interrupted.train_until(start + split_after)
    midpoint = directory / "midpoint.pt"
    interrupted.save(midpoint)
    resumed = SyntheticTrainer.from_checkpoint(midpoint)
    resumed.train_until(start + steps)
    actual = resumed.state_dict()
    checked = ("model_state_dict", "optimizer_state_dict", "sampler_state", "random_state", "history", "best_metric")
    for name in checked:
        _assert_exact(actual[name], expected[name])
    report = {"status": "passed", "source_checkpoint": str(checkpoint),
              "source_sha256": file_fingerprint(checkpoint), "start_step": start,
              "split_step": start + split_after, "end_step": start + steps,
              "midpoint_checkpoint": str(midpoint), "checked_state": list(checked),
              "tolerance": {"rtol": 0, "atol": 0}, "sample_order_and_step_losses_exact": True}
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
