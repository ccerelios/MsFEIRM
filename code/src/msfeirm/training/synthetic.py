"""Single-process complete-M3 training used to validate stage I-4."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import signal
import time
from contextlib import contextmanager

import torch

from msfeirm.config import LossConfig, ModelConfig, SegmentationConfig
from msfeirm.data import collate_groups
from msfeirm.data.synthetic import SyntheticDataConfig, SyntheticGroupDataset
from msfeirm.models import MsFEIRM
from msfeirm.models.losses import MsFEIRMJointLoss
from msfeirm.training.checkpoint import (
    atomic_save_checkpoint, cpu_snapshot, execution_fingerprint, file_fingerprint, json_fingerprint,
)
from msfeirm.training.random_state import capture_random_state, restore_random_state, seed_everything
from msfeirm.training.sampler import StatefulBatchSampler


@dataclass(frozen=True)
class SyntheticTrainingConfig:
    batch_size: int = 1
    learning_rate: float = .001
    weight_decay: float = .0001
    beta1: float = .9
    beta2: float = .999
    adam_epsilon: float = 1e-8
    seed: int = 42
    shuffle: bool = True
    device: str = "cpu"
    threads: int = 1
    checkpoint_every_steps: int = 100
    checkpoint_every_seconds: float = 300.0

    def __post_init__(self):
        for name in ("batch_size", "threads", "checkpoint_every_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if type(self.shuffle) is not bool or self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("Invalid shuffle/device configuration")
        for name in ("learning_rate", "adam_epsilon", "checkpoint_every_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if any(not math.isfinite(value) or not 0 <= value < 1 for value in (self.beta1, self.beta2)):
            raise ValueError("Adam betas must be finite and in [0,1)")

    def to_dict(self):
        return asdict(self)


class StopRequest:
    def __init__(self):
        self.reason = None

    def request(self, reason="stop_requested"):
        if self.reason is None:
            self.reason = reason

    @contextmanager
    def signal_handlers(self):
        def handle(number, _frame):
            self.request(signal.Signals(number).name)
        previous = {number: signal.signal(number, handle) for number in (signal.SIGINT, signal.SIGTERM)}
        try:
            yield self
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


class SyntheticTrainer:
    format_version = 1

    def __init__(self, model_config=None, segmentation_config=None, loss_config=None,
                 data_config=None, training_config=None, *, initialize_from=None):
        self.model_config = model_config or ModelConfig(window_size=128)
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.loss_config = loss_config or LossConfig()
        self.data_config = data_config or SyntheticDataConfig()
        self.config = training_config or SyntheticTrainingConfig()
        torch.set_num_threads(self.config.threads)
        torch.use_deterministic_algorithms(True)
        seed_everything(self.config.seed)
        self.dataset = SyntheticGroupDataset(self.model_config, self.segmentation_config, self.data_config)
        self.model = MsFEIRM(self.model_config, self.segmentation_config).to(self.config.device)
        self.initialization = None
        if initialize_from is not None:
            state = torch.load(initialize_from, map_location="cpu", weights_only=True)
            if state["model_config"] != self.model_config.to_dict() or state["segmentation_config"] != self.segmentation_config.to_dict():
                raise ValueError("Initial weight configurations must match the training model")
            self.model.load_state_dict(state["model_state_dict"])
            self.initialization = {"path": str(initialize_from), "sha256": file_fingerprint(initialize_from)}
        self.criterion = MsFEIRMJointLoss(self.segmentation_config, self.loss_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay,
                                          betas=(self.config.beta1, self.config.beta2), eps=self.config.adam_epsilon)
        self.sampler = StatefulBatchSampler(len(self.dataset), self.config.batch_size,
                                            seed=(self.config.seed + 1) % 2**32, shuffle=self.config.shuffle)
        self.global_step = 0
        self.history = []
        self.best_metric = {"name": "training_loss_total", "mode": "min", "value": None, "step": 0}
        self.last_stop_reason = None
        self.initial_evaluation = None
        self.fingerprints = {**execution_fingerprint(), "config_sha256": json_fingerprint(self.specification()),
                             "data_sha256": self.dataset.fingerprint}

    def specification(self):
        return {"model": self.model_config.to_dict(), "segmentation": self.segmentation_config.to_dict(),
                "loss": self.loss_config.to_dict(), "synthetic": self.data_config.to_dict(),
                "training": self.config.to_dict()}

    def step(self):
        indices = self.sampler.peek()
        batch = collate_groups([self.dataset[index] for index in indices]).to(self.config.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch.imu, batch.member_mask)
        losses = self.criterion(output, batch)
        if not bool(torch.isfinite(losses.total)):
            raise RuntimeError("Non-finite training loss; previous checkpoint remains valid")
        losses.total.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in self.model.parameters()):
            raise RuntimeError("Missing/non-finite gradient; previous checkpoint remains valid")
        self.optimizer.step()
        self.sampler.commit()
        self.global_step += 1
        record = {"global_step": self.global_step, "epoch": self.sampler.epoch,
                  "consumed_in_epoch": self.sampler.position, "sample_indices": list(indices),
                  "shapes": {"imu": list(batch.imu.shape), "Y_norm": list(output.node_features.shape),
                             "activity_logits": list(output.activity.logits.shape)},
                  **losses.detached_metrics(), "finite_gradients": True}
        self.history.append(record)
        if self.best_metric["value"] is None or record["loss_total"] < self.best_metric["value"]:
            self.best_metric.update(value=record["loss_total"], step=self.global_step)
        return record

    def state_dict(self):
        return cpu_snapshot({
            "format_version": self.format_version, "stage": "I-4", "checkpoint_type": "synthetic_training_state",
            "complete_m3": True, "trained": self.global_step > 0,
            "configs": self.specification(), "fingerprints": self.fingerprints,
            "initialization": self.initialization, "model_state_dict": self.model.state_dict(),
            "model_training": self.model.training, "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None, "amp_scaler_state_dict": None,
            "global_step": self.global_step, "epoch": self.sampler.epoch,
            "consumed_in_epoch": self.sampler.position, "sampler_state": self.sampler.state_dict(),
            "random_state": capture_random_state(), "best_metric": dict(self.best_metric),
            "history": self.history, "last_stop_reason": self.last_stop_reason,
            "initial_evaluation": self.initial_evaluation,
        })

    def save(self, path):
        atomic_save_checkpoint(self.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != self.format_version or state.get("checkpoint_type") != "synthetic_training_state":
            raise ValueError("Unsupported training checkpoint format")
        for name, expected in self.fingerprints.items():
            if state["fingerprints"].get(name) != expected:
                raise ValueError(f"Incompatible training checkpoint: {name}")
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.sampler.load_state_dict(state["sampler_state"])
        self.global_step = state["global_step"]
        self.history = state["history"]
        self.best_metric = state["best_metric"]
        self.initialization = state["initialization"]
        self.last_stop_reason = state["last_stop_reason"]
        self.initial_evaluation = state["initial_evaluation"]
        self.model.train(state["model_training"])
        restore_random_state(state["random_state"])

    @classmethod
    def from_checkpoint(cls, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("checkpoint_type") != "synthetic_training_state":
            raise ValueError("Expected an I-4 training checkpoint")
        configs = state["configs"]
        trainer = cls(ModelConfig.from_mapping(configs["model"]), SegmentationConfig.from_mapping(configs["segmentation"]),
                      LossConfig.from_mapping(configs["loss"]), SyntheticDataConfig(**configs["synthetic"]),
                      SyntheticTrainingConfig(**configs["training"]))
        trainer.load(path)
        return trainer

    def train_until(self, total_steps, *, checkpoint=None, max_wall_seconds=None, stop_request=None, on_step=None):
        if type(total_steps) is not int or total_steps < self.global_step:
            raise ValueError("total_steps must be an integer no smaller than global_step")
        if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds < 0):
            raise ValueError("max_wall_seconds must be finite and nonnegative")
        stop_request = stop_request or StopRequest()
        started = last_save = time.monotonic()
        if checkpoint is not None and not Path(checkpoint).exists():
            self.save(checkpoint)
        while self.global_step < total_steps:
            if stop_request.reason is not None:
                self.last_stop_reason = stop_request.reason
                break
            if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
                self.last_stop_reason = "time_budget"
                break
            record = self.step()
            if on_step is not None:
                on_step(record)
            due = (self.global_step % self.config.checkpoint_every_steps == 0 or self.sampler.at_epoch_end
                   or time.monotonic() - last_save >= self.config.checkpoint_every_seconds)
            if checkpoint is not None and due:
                self.save(checkpoint)
                last_save = time.monotonic()
        else:
            self.last_stop_reason = "target_steps_reached"
        if checkpoint is not None:
            self.save(checkpoint)
        return {"global_step": self.global_step, "stop_reason": self.last_stop_reason,
                "elapsed_seconds": time.monotonic() - started}

    @torch.no_grad()
    def evaluate(self):
        was_training, random_state = self.model.training, capture_random_state()
        self.model.eval()
        sums = {"member": 0., "cardinality": 0., "group": 0.}
        samples = groups = correct_activities = correct_counts = exact_partitions = 0
        try:
            for start in range(0, len(self.dataset), self.config.batch_size):
                batch = collate_groups(self.dataset.samples[start:start+self.config.batch_size]).to(self.config.device)
                output = self.model(batch.imu, batch.member_mask)
                losses = self.criterion(output, batch)
                size = batch.imu.shape[0]
                matched = losses.stats["classification_group_count"]
                sums["member"] += float(losses.member) * size
                sums["cardinality"] += float(losses.cardinality) * size
                sums["group"] += float(losses.group) * matched
                samples += size
                groups += matched
                correct_counts += int((output.segmentation.cardinality.decoded_count == batch.k_clusters).sum())
                pairs = batch.member_mask[:, :, None] & batch.member_mask[:, None, :]
                predicted_pairs = output.assignments[:, :, None] == output.assignments[:, None, :]
                exact_partitions += int(((predicted_pairs == batch.pair_targets.bool()) | ~pairs).all(dim=-1).all(dim=-1).sum())
                for index, matches in enumerate(losses.matching.samples):
                    correct_activities += int((output.activity.logits[index, matches.class_pred_indices].argmax(-1) == matches.class_targets).sum())
        finally:
            self.model.train(was_training)
            restore_random_state(random_state)
        values = {"loss_member": sums["member"] / samples, "loss_cardinality": sums["cardinality"] / samples,
                  "loss_group": sums["group"] / max(groups, 1)}
        values["loss_total"] = values["loss_group"] + self.loss_config.lambda_cardinality * values["loss_cardinality"] + self.loss_config.lambda_member * values["loss_member"]
        return {**values, "sample_count": samples, "matched_activity_count": groups,
                "matched_activity_accuracy": correct_activities / max(groups, 1),
                "decoded_count_accuracy": correct_counts / samples, "exact_partition_rate": exact_partitions / samples}
