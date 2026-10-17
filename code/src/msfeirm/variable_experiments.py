"""Evaluate frozen final weights at variable member counts, without a trainer."""

from collections import Counter
import json
import shutil
import sys
import time

import yaml

from msfeirm import experiments as common
from msfeirm.data.prepared import open_prepared_dataset
from msfeirm.cli.evaluate import read_predictions
from msfeirm.runtime import environment_info
from msfeirm.training.checkpoint import execution_fingerprint, file_fingerprint


def _source(path, counts, checkpoint_sha256, prepared_fingerprint):
    manifest = common.read(path / "manifest.json")
    config = manifest["configuration"]
    if manifest["status"] != "DONE" or manifest["record_counts"] != counts or config["limit_per_member_count"] is not None:
        raise ValueError("Variable experiment requires complete prediction lists")
    if config["checkpoint_metadata"]["checkpoint_sha256"] != checkpoint_sha256 or manifest["prepared_fingerprint"] != prepared_fingerprint:
        raise ValueError("Prediction source must use the frozen final checkpoint and data")
    records, info = read_predictions(path)
    if dict(Counter(str(r["members"]) for r in records)) != counts:
        raise ValueError("Prediction source counts do not match its manifest")
    return records, {**info, "record_counts": counts, "manifest_sha256": file_fingerprint(path / "manifest.json")}, manifest


def evaluate_one(manifest_path, *, experiment_id=None, run_id=None, execute=None):
    """Run a single non-primary experiment; completed training rows stay unchanged."""
    manifest_path = common.local(manifest_path)
    manifest = common.load_manifest(manifest_path)
    registry_path = common.local(manifest["registry"])
    execute = execute or common.execute_command
    with common.registry_lock(registry_path):
        registry = common.initialize_registry(manifest_path, manifest)
        primary_ids = {run["primary_experiment"] for run in manifest["training_runs"]}
        choices = [entry for entry in manifest["experiments"]
                   if entry["experiment_id"] not in primary_ids
                   and (run_id is None or entry["run_id"] == run_id)
                   and (entry["experiment_id"] == experiment_id if experiment_id else registry["experiments"][entry["experiment_id"]]["status"] != "DONE")]
        if not choices:
            if experiment_id or run_id is not None and run_id not in registry["runs"]:
                raise ValueError("Unknown non-primary experiment or run")
            return {"status": "ALL_VARIABLE_EXPERIMENTS_DONE"}
        entry = choices[0]
        exp_id = entry["experiment_id"]
        experiment = registry["experiments"][exp_id]
        row = registry["runs"][entry["run_id"]]
        run = next(value for value in manifest["training_runs"] if value["run_id"] == entry["run_id"])
        if row["status"] != "DONE":
            raise ValueError("Evaluation-only mode requires a completed formal training run")
        common.verify_files(row["artifacts"])
        if experiment["status"] == "DONE":
            common.verify_files(experiment["artifacts"])
            common.verify_files(experiment["dependency_artifacts"])
            return {"status": "DONE", "experiment_id": exp_id, "already_complete": True}
        config_path, protocol = common.local(run["config"]), common.local(manifest["protocol"])
        config = common.load_yaml(config_path)
        prepared = common.local(config["data"]["prepared_root"])
        train_data = open_prepared_dataset(prepared, "train", config["data"]["members"])
        validation_data = open_prepared_dataset(prepared, "validation", config["data"]["members"])
        checkpoint = common.local(row["checkpoint"])
        if checkpoint != common.local(run["output"]) / "checkpoints/epoch20.pt" or file_fingerprint(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Variable experiments must use the registered final epoch20 checkpoint")
        frozen = row["fingerprints"]
        if (file_fingerprint(config_path) != frozen["training_config_sha256"]
                or file_fingerprint(protocol) != frozen["protocol_sha256"]
                or train_data.manifest["fingerprint"] != frozen["prepared_fingerprint"]):
            raise ValueError("Frozen training configuration/protocol/data changed")
        current_execution = execution_fingerprint()
        training_execution = {name: frozen[name] for name in current_execution}
        state = common.verify_final_checkpoint(checkpoint, run, config, len(train_data), len(validation_data),
                                               expected_execution=training_execution)
        if state["data_manifest"]["prepared_fingerprint"] != frozen["prepared_fingerprint"]:
            raise ValueError("Registered final checkpoint prepared data changed")
        expected_counts = {str(n): len(open_prepared_dataset(prepared, "test", n)) for n in entry["members"]}
        primary = registry["experiments"][run["primary_experiment"]]
        reuse_size = config["data"]["members"]
        reuse = reuse_size in entry["members"]
        primary_path = None
        if reuse:
            if primary["status"] != "DONE" or primary["checkpoint_sha256"] != row["checkpoint_sha256"]:
                raise ValueError("Reusable primary predictions must use the same final checkpoint")
            common.verify_files(primary["artifacts"])
            primary_path = common.local(primary["bundle"]) / "predictions"
        experiment.setdefault("commands", [])
        experiment.setdefault("started_at", common.now())
        attempt = experiment.get("attempt", 0)
        bundle = common.local(entry["output"]) / f"attempt_{attempt:03d}"
        while bundle.exists():
            attempt += 1
            bundle = common.local(entry["output"]) / f"attempt_{attempt:03d}"
        bundle.mkdir(parents=True)
        commands = registry_path.parent / "commands" / exp_id
        commands.mkdir(parents=True, exist_ok=True)
        experiment.update(status="RUNNING", attempt=attempt, bundle=str(bundle.relative_to(common.PROJECT_ROOT)))

        def save():
            registry["updated_at"] = common.now()
            common.write(registry_path, registry)

        def command(name, module, arguments):
            argv = [sys.executable, "-m", module, *map(str, arguments)]
            number = len(experiment["commands"]) + 1
            out, err = commands / f"{number:03d}_{name}.stdout.log", commands / f"{number:03d}_{name}.stderr.log"
            record = {"name": name, "argv": argv, "started_at": common.now(), "status": "RUNNING",
                      "stdout": str(out.relative_to(common.PROJECT_ROOT)), "stderr": str(err.relative_to(common.PROJECT_ROOT))}
            experiment["commands"].append(record)
            save()
            print(json.dumps({"experiment_id": exp_id, "phase": name, "status": "started"}), flush=True)
            started = time.perf_counter()
            code = execute(argv, out, err)
            record.update(returncode=code, elapsed_seconds=time.perf_counter() - started,
                          status="DONE" if code == 0 else "FAILED")
            save()
            if code:
                raise RuntimeError(f"{name} exited with {code}; see {err}")
            print(json.dumps({"experiment_id": exp_id, "phase": name, "status": "passed"}), flush=True)

        try:
            save()
            fresh = [n for n in entry["members"] if not reuse or n != reuse_size]
            all_records, sources, source_manifests = [], [], []
            if fresh:
                command("predict", "msfeirm.cli.predict", ["--checkpoint", checkpoint, "--prepared", prepared,
                        "--members", *fresh, "--seed", run["seed"], "--batch-size", 1, "--output", bundle / "fresh_predictions"])
                records, info, source_manifest = _source(bundle / "fresh_predictions", {str(n): expected_counts[str(n)] for n in fresh},
                                                         row["checkpoint_sha256"], frozen["prepared_fingerprint"])
                all_records.extend(records); sources.append({**info, "reused": False}); source_manifests.append(source_manifest)
            if reuse:
                records, info, source_manifest = _source(primary_path, {str(reuse_size): expected_counts[str(reuse_size)]},
                                                         row["checkpoint_sha256"], frozen["prepared_fingerprint"])
                all_records.extend(records); sources.append({**info, "reused": True}); source_manifests.append(source_manifest)
            if (dict(Counter(str(r["members"]) for r in all_records)) != expected_counts
                    or len({(r["members"], r["sample_id"]) for r in all_records}) != len(all_records)):
                raise ValueError("Combined prediction lists must cover each selected sample once")
            all_records.sort(key=lambda record: (entry["members"].index(record["members"]), record["sample_id"]))
            predictions = bundle / "predictions"
            predictions.mkdir()
            temporary = predictions / "predictions.jsonl.tmp"
            with temporary.open("w") as stream:
                for record in all_records:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            temporary.replace(predictions / "predictions.jsonl")
            configuration = {"checkpoint": str(checkpoint), "prepared": str(prepared), "split": "test",
                             "members": entry["members"], "limit_per_member_count": None,
                             "seed": run["seed"], "batch_size": 1,
                             "checkpoint_metadata": source_manifests[0]["configuration"]["checkpoint_metadata"]}
            common.write(predictions / "manifest.json", {"stage": "III-2", "format_version": 1, "status": "DONE",
                         "dataset": run["dataset"], "configuration": configuration, "record_counts": expected_counts,
                         "prepared_fingerprint": frozen["prepared_fingerprint"], "sources": sources,
                         "fingerprints": current_execution, "training_fingerprints": state["fingerprints"],
                         "prediction_file": "predictions.jsonl", "prediction_sha256": file_fingerprint(predictions / "predictions.jsonl"),
                         "sample_count": len(all_records)})
            (predictions / "config.resolved.yaml").write_text(yaml.safe_dump(configuration, sort_keys=False))
            common.write(predictions / "environment.json", environment_info())
            shutil.copyfile(common.PROJECT_ROOT / "requirements.txt", predictions / "requirements.txt")
            command("evaluate", "msfeirm.cli.evaluate", ["--predictions", predictions, "--protocol", protocol, "--output", bundle / "evaluation"])
            saved = bundle / "saved_predictions"
            saved.mkdir()
            shutil.copyfile(predictions / "predictions.jsonl", saved / "predictions.jsonl")
            shutil.copyfile(protocol, saved / "protocol.yaml")
            command("recompute", "msfeirm.cli.evaluate", ["--predictions", saved / "predictions.jsonl", "--protocol", saved / "protocol.yaml", "--output", bundle / "recomputed"])
            original, recomputed = common.read(bundle / "evaluation/metrics.json"), common.read(bundle / "recomputed/metrics.json")
            if (original["status"] != "DONE" or recomputed["status"] != "DONE"
                    or original["overall"]["sample_count"] != len(all_records)
                    or {n: value["sample_count"] for n, value in original["by_members"].items()} != expected_counts):
                raise ValueError("Variable experiment evaluation is incomplete")
            if any(original[k] != recomputed[k] for k in ("overall", "by_members", "intervals")):
                raise ValueError("Saved-only evaluation did not reproduce exactly")
            result = {"status": "DONE", "scope": "formal_chapter4", "experiment_id": exp_id, "run_id": run["run_id"],
                      "dataset": run["dataset"], "model": "m3", "seed": run["seed"], "epochs": 20,
                      "global_step": state["global_step"], "checkpoint": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
                      "fingerprints": {**current_execution, "protocol_sha256": file_fingerprint(protocol), "prepared_fingerprint": frozen["prepared_fingerprint"]},
                      "training_fingerprints": state["fingerprints"], "prediction_counts": expected_counts,
                      "prediction_sources": sources, "saved_only_recomputation": "exact",
                      "metrics": {k: original[k] for k in ("overall", "by_members", "intervals")}}
            common.write(bundle / "result.json", result)
            experiment.update(status="DONE", completed_at=common.now(), checkpoint=row["checkpoint"],
                              checkpoint_sha256=row["checkpoint_sha256"], prediction_counts=expected_counts,
                              result=str((bundle / "result.json").relative_to(common.PROJECT_ROOT)),
                              artifacts={str(p.relative_to(common.PROJECT_ROOT)): file_fingerprint(p) for p in sorted(bundle.rglob("*")) if p.is_file()},
                              dependency_artifacts={str(path.relative_to(common.PROJECT_ROOT)): file_fingerprint(path)
                                                    for path in [checkpoint, config_path, protocol, prepared / "manifest.json"]})
            experiment.pop("error", None)
            save()
            return {"status": "DONE", "experiment_id": exp_id, "prediction_counts": expected_counts,
                    "reused_primary_predictions": reuse, "checkpoint_sha256": row["checkpoint_sha256"]}
        except BaseException as error:
            experiment.update(status="FAILED", error={"type": type(error).__name__, "message": str(error), "at": common.now()})
            save()
            raise
