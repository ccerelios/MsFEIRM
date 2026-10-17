"""Complete-M3 training on frozen real-window group datasets."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time

import torch

from msfeirm.config import PROJECT_ROOT, LossConfig, ModelConfig, SegmentationConfig
from msfeirm.data import collate_groups
from msfeirm.data.prepared import open_prepared_dataset, verify_prepared_files
from msfeirm.models import MsFEIRM
from msfeirm.models.losses import MsFEIRMJointLoss
from msfeirm.training.checkpoint import atomic_save_checkpoint, cpu_snapshot, execution_fingerprint, json_fingerprint
from msfeirm.training.journal import TrainingJournal
from msfeirm.training.random_state import capture_random_state, restore_random_state, seed_everything
from msfeirm.training.sampler import StatefulBatchSampler
from msfeirm.training.synthetic import StopRequest, SyntheticTrainingConfig


@dataclass(frozen=True)
class RealDataConfig:
    prepared_root: str = "data/processed/wbsensor"
    members: int | None = None
    train_limit: int | None = None
    validation_limit: int | None = None

    def __post_init__(self):
        for name in ("members", "train_limit", "validation_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class RealTrainingConfig(SyntheticTrainingConfig):
    epochs: int = 20

    def __post_init__(self):
        super().__post_init__()
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")


class RealTrainer:
    format_version = 1

    def __init__(self, data_config=None, training_config=None, *, model_config=None,
                 segmentation_config=None, loss_config=None, journal_path=None):
        self.data_config = data_config or RealDataConfig()
        self.config = training_config or RealTrainingConfig()
        self.prepared_root = (PROJECT_ROOT / self.data_config.prepared_root).resolve()
        manifest = verify_prepared_files(self.prepared_root)
        self.train_data = open_prepared_dataset(self.prepared_root, "train", self.data_config.members)
        self.validation_data = open_prepared_dataset(self.prepared_root, "validation", self.data_config.members)
        self.train_indices = self._indices(self.train_data, self.data_config.train_limit)
        self.validation_indices = self._indices(self.validation_data, self.data_config.validation_limit)
        expected_model = {"input_channels": manifest["window_shape"][1], "window_size": manifest["window_shape"][2],
                          "num_classes": len(manifest["configuration"]["activity_names"])}
        self.model_config = (model_config if isinstance(model_config, ModelConfig) else
                             ModelConfig.from_mapping({**expected_model, **(model_config or {})}))
        if any(getattr(self.model_config, key) != value for key, value in expected_model.items()):
            raise ValueError("Model input/classes must match the prepared data protocol")
        lo, hi = self.train_data.k_bounds
        self.segmentation_config = (segmentation_config if isinstance(segmentation_config, SegmentationConfig) else
                                    SegmentationConfig.from_mapping({"k_min": lo, "k_max": hi, **(segmentation_config or {})}))
        if self.validation_data.k_bounds != (lo, hi) or (self.segmentation_config.k_min, self.segmentation_config.k_max) != (lo, hi):
            raise ValueError("Training and validation must use the fixed model K protocol")
        ignore = manifest["configuration"]["grouping"]["ignore_index"]
        self.loss_config = (loss_config if isinstance(loss_config, LossConfig) else
                            LossConfig.from_mapping({"ignore_index": ignore, **(loss_config or {})}))
        if self.loss_config.ignore_index != ignore:
            raise ValueError("Loss ignore_index must match the prepared group interface")
        torch.set_num_threads(self.config.threads)
        torch.use_deterministic_algorithms(True)
        seed_everything(self.config.seed)
        self.model = MsFEIRM(self.model_config, self.segmentation_config).to(self.config.device)
        self.criterion = MsFEIRMJointLoss(self.segmentation_config, self.loss_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay,
                                          betas=(self.config.beta1, self.config.beta2), eps=self.config.adam_epsilon)
        self.sampler = StatefulBatchSampler(len(self.train_indices), self.config.batch_size,
                                            seed=(self.config.seed + 1) % 2**32, shuffle=self.config.shuffle)
        self.steps_per_epoch = math.ceil(len(self.train_indices) / self.config.batch_size)
        self.target_steps = self.config.epochs * self.steps_per_epoch
        self.global_step = 0
        self.last_record = self.last_validation = self.initial_validation = None
        self.last_validated_epoch = -1
        self.best_metric = {"name": "validation_loss_total", "mode": "min", "value": None, "step": 0}
        self.last_stop_reason = None
        self.journal = TrainingJournal(journal_path) if journal_path is not None else None
        self.data_manifest = {"dataset": manifest["dataset"], "prepared_root": str(self.prepared_root),
                              "prepared_fingerprint": manifest["fingerprint"],
                              "train_fingerprint": self.train_data.fingerprint, "train_indices": list(self.train_indices),
                              "validation_fingerprint": self.validation_data.fingerprint,
                              "validation_indices": list(self.validation_indices), "model_k_bounds": [lo, hi]}
        self.fingerprints = {**execution_fingerprint(), "config_sha256": json_fingerprint(self.specification()),
                             "data_sha256": json_fingerprint(self.data_manifest)}

    @staticmethod
    def _indices(dataset, limit):
        count = len(dataset) if limit is None else min(len(dataset), limit)
        if not count:
            raise ValueError("Training and validation require nonempty group datasets")
        return tuple(range(count))

    @property
    def completed_epochs(self):
        return self.sampler.epoch + int(self.sampler.at_epoch_end)

    def specification(self):
        # Resolve the cache path so restores do not depend on the shell cwd.
        data = {**self.data_config.to_dict(), "prepared_root": str(self.prepared_root)}
        return {"data": data, "training": self.config.to_dict(), "model": self.model_config.to_dict(),
                "segmentation": self.segmentation_config.to_dict(), "loss": self.loss_config.to_dict()}

    def _emit(self, record):
        if self.journal is not None:
            self.journal.append(record)

    def step(self):
        local_indices = self.sampler.peek()
        indices = [self.train_indices[index] for index in local_indices]
        batch = collate_groups([self.train_data[index] for index in indices]).to(self.config.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.model(batch.imu, batch.member_mask)
        losses = self.criterion(output, batch)
        if not bool(torch.isfinite(losses.total)):
            raise RuntimeError("Non-finite loss; restore the previous valid checkpoint")
        losses.total.backward()
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in self.model.parameters()):
            raise RuntimeError("Missing/non-finite gradient; restore the previous valid checkpoint")
        self.optimizer.step()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
            raise RuntimeError("Non-finite model update; restore the previous valid checkpoint")
        self.sampler.commit()
        self.global_step += 1
        self.last_record = {"event": "train", "global_step": self.global_step, "epoch": self.sampler.epoch,
                            "consumed_in_epoch": self.sampler.position, "sample_indices": indices,
                            "sample_ids": list(batch.sample_ids), "finite_gradients": True,
                            "shapes": {"imu": list(batch.imu.shape), "Y_norm": list(output.node_features.shape),
                                       "activity_logits": list(output.activity.logits.shape)},
                            **losses.detached_metrics()}
        self._emit(self.last_record)
        return self.last_record

    @torch.no_grad()
    def evaluate(self, split="validation"):
        if split not in ("train", "validation"):
            raise ValueError("Training diagnostics support train or validation only")
        dataset, indices = (self.train_data, self.train_indices) if split == "train" else (self.validation_data, self.validation_indices)
        was_training, rng = self.model.training, capture_random_state()
        self.model.eval()
        sums = {name: 0. for name in ("member", "cardinality", "group", "mae")}
        counts = {name: 0 for name in ("member", "cardinality", "group")}
        try:
            for start in range(0, len(indices), self.config.batch_size):
                batch = collate_groups([dataset[index] for index in indices[start:start+self.config.batch_size]]).to(self.config.device)
                losses = self.criterion(self.model(batch.imu, batch.member_mask), batch)
                if not bool(torch.isfinite(losses.total)):
                    raise RuntimeError("Non-finite validation loss")
                for name, key in (("member", "member_sample_count"), ("cardinality", "cardinality_sample_count"),
                                  ("group", "classification_group_count")):
                    count = losses.stats[key]
                    sums[name] += float(getattr(losses, name)) * count
                    counts[name] += count
                sums["mae"] += losses.stats["cardinality_mae"] * losses.stats["cardinality_sample_count"]
        finally:
            self.model.train(was_training)
            restore_random_state(rng)
        metrics = {f"loss_{name}": sums[name] / max(counts[name], 1) for name in counts}
        metrics["loss_total"] = metrics["loss_group"] + self.loss_config.lambda_cardinality * metrics["loss_cardinality"] + self.loss_config.lambda_member * metrics["loss_member"]
        return {**metrics, "sample_count": len(indices), "classification_group_count": counts["group"],
                "cardinality_mae": sums["mae"] / max(counts["cardinality"], 1)}

    def _validate_epoch(self, checkpoint):
        if not self.sampler.at_epoch_end or self.last_validated_epoch == self.sampler.epoch:
            return
        self.last_validation = {"event": "validation", "global_step": self.global_step,
                                "epoch": self.sampler.epoch, **self.evaluate()}
        self.last_validated_epoch = self.sampler.epoch
        self._emit(self.last_validation)
        improved = self.best_metric["value"] is None or self.last_validation["loss_total"] < self.best_metric["value"]
        if improved:
            self.best_metric.update(value=self.last_validation["loss_total"], step=self.global_step)
            if checkpoint is not None:
                self.save(Path(checkpoint).with_name("best.pt"))

    def state_dict(self):
        return cpu_snapshot({
            "format_version": self.format_version, "stage": "II-2", "checkpoint_type": "real_training_state",
            "complete_m3": True, "trained": self.global_step > 0, "configs": self.specification(),
            "fingerprints": self.fingerprints, "data_manifest": self.data_manifest,
            "model_state_dict": self.model.state_dict(), "model_training": self.model.training,
            "optimizer_state_dict": self.optimizer.state_dict(), "scheduler_state_dict": None, "amp_scaler_state_dict": None,
            "global_step": self.global_step, "epoch": self.sampler.epoch, "consumed_in_epoch": self.sampler.position,
            "sampler_state": self.sampler.state_dict(), "random_state": capture_random_state(),
            "best_metric": dict(self.best_metric), "last_validated_epoch": self.last_validated_epoch,
            "last_record": self.last_record, "last_validation": self.last_validation, "initial_validation": self.initial_validation,
            "journal_state": self.journal.state_dict() if self.journal is not None else None,
            "last_stop_reason": self.last_stop_reason,
        })

    def save(self, path):
        atomic_save_checkpoint(self.state_dict(), path)

    def load(self, path, *, journal_path=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("format_version") != self.format_version or state.get("checkpoint_type") != "real_training_state":
            raise ValueError("Unsupported real training checkpoint format")
        for name, expected in self.fingerprints.items():
            if state["fingerprints"].get(name) != expected:
                raise ValueError(f"Incompatible training checkpoint: {name}")
        journal = None
        if state["journal_state"] is not None:
            journal = TrainingJournal.restore(state["journal_state"], journal_path,
                                               fallback=Path(path).parent.parent / "metrics.jsonl")
        elif journal_path is not None:
            journal = TrainingJournal(journal_path)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.sampler.load_state_dict(state["sampler_state"])
        for name in ("global_step", "best_metric", "last_validated_epoch", "last_record", "last_validation", "initial_validation", "last_stop_reason"):
            setattr(self, name, state[name])
        self.journal = journal
        self.model.train(state["model_training"])
        restore_random_state(state["random_state"])

    @classmethod
    def from_checkpoint(cls, path, *, journal_path=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get("checkpoint_type") != "real_training_state":
            raise ValueError("Expected a real training checkpoint")
        config = state["configs"]
        trainer = cls(RealDataConfig(**config["data"]), RealTrainingConfig(**config["training"]),
                      model_config=config["model"], segmentation_config=config["segmentation"], loss_config=config["loss"])
        trainer.load(path, journal_path=journal_path)
        return trainer

    def train_until(self, total_steps=None, *, checkpoint=None, max_wall_seconds=None, stop_request=None, on_step=None):
        total_steps = self.target_steps if total_steps is None else total_steps
        if type(total_steps) is not int or not self.global_step <= total_steps <= self.target_steps:
            raise ValueError("total_steps must be between saved progress and configured epoch target")
        if max_wall_seconds is not None and (not math.isfinite(max_wall_seconds) or max_wall_seconds < 0):
            raise ValueError("max_wall_seconds must be finite and nonnegative")
        stop = stop_request or StopRequest()
        started = last_save = time.monotonic()
        self.last_stop_reason = None
        if checkpoint is not None and not Path(checkpoint).exists():
            self.save(checkpoint)
        while self.global_step < total_steps:
            if stop.reason is not None:
                self.last_stop_reason = stop.reason
                break
            if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
                self.last_stop_reason = "time_budget"
                break
            if self.initial_validation is None:
                self.initial_validation = {"event": "initial_validation", "global_step": self.global_step, **self.evaluate()}
                self._emit(self.initial_validation)
                continue  # Recheck the budget after validation.
            self._validate_epoch(checkpoint)
            record = self.step()
            if on_step is not None:
                on_step(record)
            self._validate_epoch(checkpoint)
            due = (self.global_step % self.config.checkpoint_every_steps == 0 or self.sampler.at_epoch_end
                   or time.monotonic() - last_save >= self.config.checkpoint_every_seconds)
            if checkpoint is not None and due:
                self.save(checkpoint)
                last_save = time.monotonic()
        else:
            self.last_stop_reason = "epochs_completed" if self.global_step == self.target_steps else "target_steps_reached"
        if checkpoint is not None:
            self.save(checkpoint)
            if self.global_step == self.target_steps:
                self.save(Path(checkpoint).with_name(f"epoch{self.config.epochs}.pt"))
        return {"global_step": self.global_step, "completed_epochs": self.completed_epochs,
                "configured_epochs": self.config.epochs, "target_steps": self.target_steps,
                "stop_reason": self.last_stop_reason, "elapsed_seconds": time.monotonic() - started}
