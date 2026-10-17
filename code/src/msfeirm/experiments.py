"""One resumable formal training slice and its fixed-size Chapter IV experiment."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from msfeirm.config import PROJECT_ROOT, load_yaml
from msfeirm.data.prepared import open_prepared_dataset
from msfeirm.runtime import environment_info
from msfeirm.training.checkpoint import execution_fingerprint, file_fingerprint


def local(path):
    path = (PROJECT_ROOT / path).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or path == PROJECT_ROOT:
        raise ValueError("Experiment paths must be inside the project")
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path):
    manifest = load_yaml(path)
    if manifest.get("version") != 1 or manifest.get("model") != "m3":
        raise ValueError("Only version 1 complete M3 experiments are supported")
    runs = manifest["training_runs"]
    experiments = manifest["experiments"]
    ids = [r["run_id"] for r in runs]
    eids = [e["experiment_id"] for e in experiments]
    if len(set(ids)) != len(ids) or len(set(eids)) != len(eids) or not runs:
        raise ValueError("Run and experiment identifiers must be unique")
    outputs = [local(r["output"]) for r in runs] + [local(e["output"]) for e in experiments]
    for i, path_i in enumerate(outputs):
        if any(path_i.is_relative_to(p) or p.is_relative_to(path_i) for p in outputs[:i]):
            raise ValueError("Run and experiment outputs must not overlap")
    local(manifest["registry"])
    local(manifest["protocol"]).read_bytes()
    for run in runs:
        config = load_yaml(local(run["config"]))
        data, training = config["data"], config["training"]
        if data.get("train_limit") is not None or data.get("validation_limit") is not None:
            raise ValueError("Formal experiments require complete train/validation lists")
        if training.get("epochs") != 20 or training.get("batch_size") != 1 or training.get("seed") != run["seed"]:
            raise ValueError("Formal run must use its fixed seed, 20 epochs and batch size 1")
        if not math.isfinite(run["max_wall_time_min"]) or not 0 < run["max_wall_time_min"] <= 210:
            raise ValueError("Training slice budget must be positive and at most 210 minutes")
        primary = [e for e in experiments if e["experiment_id"] == run["primary_experiment"] and e["run_id"] == run["run_id"]]
        if len(primary) != 1 or primary[0]["members"] != [data["members"]]:
            raise ValueError("Primary experiment must test the training member count")
    for experiment in experiments:
        members = experiment["members"]
        if experiment["run_id"] not in ids or not members or len(set(members)) != len(members) or any(type(n) is not int or n < 1 for n in members):
            raise ValueError("Invalid experiment run or member counts")
    return manifest


@contextmanager
def registry_lock(path):
    lock = path.with_suffix(path.suffix + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    # A crash leaves this file visible for explicit recovery; never steal a live run.
    with lock.open("x") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_at": now()}))
    try:
        yield
    finally:
        lock.unlink()


def initialize_registry(manifest_path, manifest):
    path = local(manifest["registry"])
    fingerprint = file_fingerprint(manifest_path)
    if path.exists():
        registry = read(path)
        if registry["manifest_sha256"] != fingerprint:
            raise ValueError("Existing registry belongs to a different manifest")
        return registry
    registry = {"version": 1, "scope": "formal_chapter4", "model": "m3", "created_at": now(),
                "manifest": str(manifest_path), "manifest_sha256": fingerprint,
                "runs": {r["run_id"]: {"status": "PENDING", "configuration": r, "commands": []} for r in manifest["training_runs"]},
                "experiments": {e["experiment_id"]: {"status": "PENDING", "configuration": e} for e in manifest["experiments"]}}
    write(path, registry)
    return registry


def execute_command(argv, stdout, stderr):
    with stdout.open("w") as out, stderr.open("w") as err:
        return subprocess.run(argv, cwd=PROJECT_ROOT, stdout=out, stderr=err).returncode


def verify_final_checkpoint(path, run, config, train_count, validation_count, *, expected_execution=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("checkpoint_type") != "real_training_state" or not state.get("complete_m3") or not state.get("trained"):
        raise ValueError("Formal result requires a trained complete M3 state")
    saved = state["configs"]
    if saved["data"]["train_limit"] is not None or saved["data"]["validation_limit"] is not None:
        raise ValueError("A development subset checkpoint cannot complete a formal run")
    if saved["training"]["epochs"] != 20 or saved["training"]["batch_size"] != 1 or saved["training"]["seed"] != run["seed"]:
        raise ValueError("Checkpoint training configuration does not match the formal run")
    for section in ("training", "model", "segmentation", "loss"):
        if any(saved[section].get(key) != value for key, value in config.get(section, {}).items()):
            raise ValueError("Checkpoint configuration does not match the formal run")
    if state["data_manifest"]["dataset"] != run["dataset"] or saved["data"]["members"] != config["data"]["members"]:
        raise ValueError("Checkpoint data configuration does not match the formal run")
    if Path(saved["data"]["prepared_root"]).resolve() != local(config["data"]["prepared_root"]):
        raise ValueError("Checkpoint prepared data path changed")
    if state["global_step"] != 20 * train_count or state["epoch"] != 19 or state["consumed_in_epoch"] != train_count:
        raise ValueError("Formal checkpoint must complete all 20 epochs")
    if len(state["data_manifest"]["train_indices"]) != train_count or len(state["data_manifest"]["validation_indices"]) != validation_count:
        raise ValueError("Formal checkpoint must use complete data lists")
    for name, expected in (execution_fingerprint() if expected_execution is None else expected_execution).items():
        if state["fingerprints"].get(name) != expected:
            raise ValueError("Checkpoint execution fingerprint changed")
    return state


def verify_files(files):
    for path, expected in files.items():
        if file_fingerprint(local(path)) != expected:
            raise ValueError(f"Saved experiment artifact changed: {path}")


def run_one(manifest_path, *, run_id=None, max_wall_time_min=None, init_only=False, execute=None):
    manifest_path = local(manifest_path)
    manifest = load_manifest(manifest_path)
    registry_path = local(manifest["registry"])
    execute = execute or execute_command
    with registry_lock(registry_path):
        registry = initialize_registry(manifest_path, manifest)
        if init_only:
            return {"status": "REGISTERED", "registry": str(registry_path)}
        choices = [r for r in manifest["training_runs"] if run_id == r["run_id"]] if run_id else [r for r in manifest["training_runs"] if registry["runs"][r["run_id"]]["status"] != "DONE"][:1]
        if not choices:
            if run_id:
                raise ValueError(f"Unknown run: {run_id}")
            return {"status": "ALL_PRIMARY_RUNS_DONE", "registry": str(registry_path)}
        run = choices[0]
        row = registry["runs"][run["run_id"]]
        experiment = registry["experiments"][run["primary_experiment"]]
        if row["status"] == "DONE":
            verify_files(row["artifacts"])
            verify_files(experiment["artifacts"])
            return {"status": "DONE", "run_id": run["run_id"], "already_complete": True}
        budget = run["max_wall_time_min"] if max_wall_time_min is None else max_wall_time_min
        if not math.isfinite(budget) or not 0 < budget <= run["max_wall_time_min"]:
            raise ValueError("Slice override must be positive and no larger than its frozen budget")
        config_path = local(run["config"])
        protocol = local(manifest["protocol"])
        config = load_yaml(config_path)
        prepared = local(config["data"]["prepared_root"])
        members = config["data"]["members"]
        train_data = open_prepared_dataset(prepared, "train", members)
        validation_data = open_prepared_dataset(prepared, "validation", members)
        expected_fingerprints = {**execution_fingerprint(), "training_config_sha256": file_fingerprint(config_path),
                                 "protocol_sha256": file_fingerprint(protocol), "prepared_fingerprint": train_data.manifest["fingerprint"]}
        if "fingerprints" in row and row["fingerprints"] != expected_fingerprints:
            raise ValueError("Formal run configuration/source/data/environment changed; restore its saved version")
        output = local(run["output"])
        if "fingerprints" not in row and output.exists() and any(output.iterdir()):
            raise ValueError("Formal training requires a fresh output directory")
        row.update(fingerprints=expected_fingerprints, environment=environment_info())
        row.setdefault("started_at", now())
        commands = registry_path.parent / "commands" / run["run_id"]
        commands.mkdir(parents=True, exist_ok=True)

        def save():
            registry["updated_at"] = now()
            write(registry_path, registry)

        def command(name, module, arguments):
            index = len(row["commands"]) + 1
            argv = [sys.executable, "-m", module, *map(str, arguments)]
            stdout, stderr = commands / f"{index:03d}_{name}.stdout.log", commands / f"{index:03d}_{name}.stderr.log"
            record = {"name": name, "argv": argv, "started_at": now(), "stdout": str(stdout.relative_to(PROJECT_ROOT)),
                      "stderr": str(stderr.relative_to(PROJECT_ROOT)), "status": "RUNNING"}
            row["commands"].append(record)
            save()
            print(json.dumps({"run_id": run["run_id"], "phase": name, "status": "started"}), flush=True)
            started = time.perf_counter()
            code = execute(argv, stdout, stderr)
            record.update(returncode=code, elapsed_seconds=time.perf_counter() - started, status="DONE" if code == 0 else "FAILED")
            save()
            if code:
                raise RuntimeError(f"{name} exited with {code}; see {stderr}")
            print(json.dumps({"phase": name, "status": "passed", "seconds": record["elapsed_seconds"]}), flush=True)

        try:
            last = output / "checkpoints/last.pt"
            final = output / "checkpoints/epoch20.pt"
            if not final.exists():
                row["status"] = "TRAINING"
                save()
                arguments = ["--resume", last] if last.exists() else ["--config", config_path, "--seed", run["seed"], "--output", output]
                command("train_resume" if last.exists() else "train", "msfeirm.cli.train",
                        [*arguments, "--max-wall-time-min", budget, "--profile", "--log-every", 500])
                summary = read(output / "summary.json")
                row.update(global_step=summary["global_step"], completed_epochs=summary["completed_epochs"],
                           last_checkpoint=str(last.relative_to(PROJECT_ROOT)), stop_reason=summary["stop_reason"])
                if summary["global_step"] != 20 * len(train_data):
                    row["status"] = "PAUSED"
                    save()
                    return {"status": "PAUSED", "run_id": run["run_id"], "global_step": summary["global_step"]}
            state = verify_final_checkpoint(final, run, config, len(train_data), len(validation_data))
            if state["data_manifest"]["prepared_fingerprint"] != expected_fingerprints["prepared_fingerprint"]:
                raise ValueError("Final checkpoint prepared data fingerprint changed")
            row.update(status="TRAINED", global_step=state["global_step"], completed_epochs=20,
                       checkpoint=str(final.relative_to(PROJECT_ROOT)), checkpoint_sha256=file_fingerprint(final))
            save()
            exp_output = local(experiment["configuration"]["output"])
            exp_output.mkdir(parents=True, exist_ok=True)
            experiment["status"] = "RUNNING"
            save()

            # An interrupted immutable output gets a new attempt directory on re-entry.
            attempt = experiment.get("attempt", 0)
            bundle = exp_output / f"attempt_{attempt:03d}"
            while bundle.exists():
                attempt += 1
                bundle = exp_output / f"attempt_{attempt:03d}"
            experiment.update(attempt=attempt, bundle=str(bundle.relative_to(PROJECT_ROOT)))
            bundle.mkdir(parents=True, exist_ok=True)
            command("predict", "msfeirm.cli.predict", ["--checkpoint", final, "--prepared", prepared,
                    "--members", *experiment["configuration"]["members"], "--batch-size", 1, "--output", bundle / "predictions"])
            command("evaluate", "msfeirm.cli.evaluate", ["--predictions", bundle / "predictions", "--protocol", protocol, "--output", bundle / "evaluation"])
            saved = bundle / "saved_predictions"
            saved.mkdir()
            shutil.copyfile(bundle / "predictions/predictions.jsonl", saved / "predictions.jsonl")
            shutil.copyfile(protocol, saved / "protocol.yaml")
            command("recompute", "msfeirm.cli.evaluate", ["--predictions", saved / "predictions.jsonl", "--protocol", saved / "protocol.yaml", "--output", bundle / "recomputed"])
            predictions = read(bundle / "predictions/manifest.json")
            expected_counts = {str(n): len(open_prepared_dataset(prepared, "test", n)) for n in experiment["configuration"]["members"]}
            if predictions["status"] != "DONE" or predictions["record_counts"] != expected_counts:
                raise ValueError("Formal prediction must cover the complete selected test lists")
            if predictions["configuration"]["checkpoint_metadata"]["checkpoint_sha256"] != row["checkpoint_sha256"]:
                raise ValueError("Formal predictions must use the final epoch20 checkpoint")
            if file_fingerprint(bundle / "predictions/predictions.jsonl") != predictions["prediction_sha256"]:
                raise ValueError("Formal prediction file hash changed")
            original, recomputed = read(bundle / "evaluation/metrics.json"), read(bundle / "recomputed/metrics.json")
            if original["status"] != "DONE" or recomputed["status"] != "DONE" or original["overall"]["sample_count"] != sum(expected_counts.values()):
                raise ValueError("Formal evaluation is incomplete")
            if any(original[k] != recomputed[k] for k in ("overall", "by_members", "intervals")):
                raise ValueError("Saved-only evaluation did not reproduce exactly")
            result = {"status": "DONE", "scope": "formal_chapter4", "experiment_id": run["primary_experiment"],
                      "run_id": run["run_id"], "dataset": run["dataset"], "model": "m3", "seed": run["seed"],
                      "epochs": 20, "global_step": state["global_step"], "checkpoint": row["checkpoint"],
                      "checkpoint_sha256": row["checkpoint_sha256"], "fingerprints": expected_fingerprints,
                      "prediction_counts": expected_counts, "saved_only_recomputation": "exact",
                      "metrics": {k: original[k] for k in ("overall", "by_members", "intervals")}}
            write(bundle / "result.json", result)
            experiment.update(status="DONE", completed_at=now(), checkpoint=row["checkpoint"], checkpoint_sha256=row["checkpoint_sha256"],
                              prediction_counts=expected_counts, result=str((bundle / "result.json").relative_to(PROJECT_ROOT)),
                              artifacts={str(p.relative_to(PROJECT_ROOT)): file_fingerprint(p) for p in sorted(bundle.rglob("*")) if p.is_file()})
            row.update(status="DONE", completed_at=now(), artifacts={str(p.relative_to(PROJECT_ROOT)): file_fingerprint(p)
                       for p in sorted(output.rglob("*")) if p.is_file()})
            row.pop("error", None)
            save()
            return {"status": "DONE", "run_id": run["run_id"], "experiment_id": run["primary_experiment"],
                    "global_step": state["global_step"], "prediction_counts": expected_counts, "registry": str(registry_path)}
        except BaseException as error:
            row.update(status="FAILED", error={"type": type(error).__name__, "message": str(error), "at": now()})
            save()
            raise
