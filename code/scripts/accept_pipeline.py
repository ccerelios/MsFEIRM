"""Run one dataset's II-4 full-data acceptance bundle in the project venv."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from msfeirm.config import PROJECT_ROOT, load_yaml
from msfeirm.data.prepared import _json, open_prepared_dataset
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv
from msfeirm.training.checkpoint import file_fingerprint
from msfeirm.training.timing import project_training_time


def read(path):
    return json.loads(path.read_text())


def combine_timing(summaries):
    combined = {}
    for name in sorted({key for summary in summaries for key in summary}):
        blocks = [summary[name] for summary in summaries if name in summary]
        count = sum(value["calls"] for value in blocks)
        total = sum(value["total_seconds"] for value in blocks)
        first = blocks[0]["first_seconds"]
        combined[name] = {"calls":count,"failed_calls":sum(value["failed_calls"] for value in blocks),
                          "total_seconds":total,"first_seconds":first,"mean_seconds":total/count,
                          "min_seconds":min(value["min_seconds"] for value in blocks),
                          "max_seconds":max(value["max_seconds"] for value in blocks),
                          "mean_after_first_seconds":(total-first)/(count-1) if count>1 else None}
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=("ut","wb"),required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--budget-seconds",type=float,default=1.2)
    parser.add_argument("--max-wall-time-min",type=float,default=5)
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if not math.isfinite(args.budget_seconds) or args.budget_seconds<=0:
        parser.error("budget-seconds must be finite and positive")
    directory = project_output_path(args.output)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Acceptance bundle must use an empty output directory")
    directory.mkdir(parents=True,exist_ok=True)
    (directory/"commands").mkdir()
    config_path = PROJECT_ROOT/f"configs/training/{args.dataset}_m3.yaml"
    values = load_yaml(config_path)
    prepared = (PROJECT_ROOT/values["data"]["prepared_root"]).resolve()
    train_data = open_prepared_dataset(prepared,"train",values["data"].get("members"))
    validation_data = open_prepared_dataset(prepared,"validation",values["data"].get("members"))
    steps_per_epoch = math.ceil(len(train_data)/values["training"]["batch_size"])
    report = {"stage":"II-4","status":"IN_PROGRESS","scope":"phase_II_acceptance_only", "dataset":args.dataset,
              "configuration":str(config_path),"environment":environment_info(),"commands":[],
              "train_groups":len(train_data),"validation_groups":len(validation_data),"steps_per_epoch":steps_per_epoch}
    started = time.perf_counter()
    _json(directory/"pipeline_summary.json",report)

    def run(name,module,arguments):
        command = [sys.executable,"-m",module,*map(str,arguments)]
        print(json.dumps({"phase":name,"status":"started"}),flush=True)
        before = time.perf_counter()
        completed = subprocess.run(command,cwd=PROJECT_ROOT,capture_output=True,text=True)
        elapsed = time.perf_counter()-before
        (directory/f"commands/{name}.stdout.log").write_text(completed.stdout)
        (directory/f"commands/{name}.stderr.log").write_text(completed.stderr)
        report["commands"].append({"name":name,"argv":command,"elapsed_seconds":elapsed,"returncode":completed.returncode})
        _json(directory/"pipeline_summary.json",report)
        if completed.returncode:
            report["status"] = "FAILED"
            _json(directory/"pipeline_summary.json",report)
            print(completed.stderr[-4000:],flush=True)
            raise SystemExit(completed.returncode)
        print(json.dumps({"phase":name,"status":"passed","seconds":elapsed}),flush=True)
        return elapsed

    training = directory/"training"
    run("budget_stop","msfeirm.cli.train",["--config",config_path,"--max-wall-time-min",args.budget_seconds/60,
                                            "--profile","--output",training])
    initial = read(training/"summary.json")
    if initial["stop_reason"]!="time_budget":
        raise AssertionError("Budget acceptance must stop before the configured 20-epoch target")
    shutil.copyfile(training/"summary.json",training/"budget_stop.summary.json")
    backup = directory/"budget_stop"
    (backup/"checkpoints").mkdir(parents=True)
    shutil.copyfile(training/"checkpoints/last.pt",backup/"checkpoints/last.pt")
    shutil.copyfile(training/"metrics.jsonl",backup/"metrics.jsonl")
    target = max(steps_per_epoch,math.ceil(initial["global_step"]/steps_per_epoch)*steps_per_epoch)
    run("train_resume","msfeirm.cli.train",["--resume",training/"checkpoints/last.pt","--steps",target,
                                             "--max-wall-time-min",args.max_wall_time_min,"--profile","--verify-resume"])
    trained = read(training/"summary.json")
    if trained["global_step"]!=target or trained["resume_verification"]["status"]!="passed":
        raise AssertionError("Full-data epoch/resume acceptance did not complete")
    timing = combine_timing([initial["timing"],trained["timing"]])
    if timing["train_step"]["calls"]!=target:
        raise AssertionError("Timing must include every committed optimizer step")
    run("predict","msfeirm.cli.predict",["--checkpoint",training/"checkpoints/last.pt","--prepared",prepared,
                                          "--batch-size",1,"--output",directory/"predictions"])
    run("evaluate","msfeirm.cli.evaluate",["--predictions",directory/"predictions","--output",directory/"evaluation"])
    saved = directory/"saved_predictions"
    saved.mkdir()
    shutil.copyfile(directory/"predictions/predictions.jsonl",saved/"predictions.jsonl")
    shutil.copyfile(PROJECT_ROOT/"configs/protocols/default.yaml",saved/"protocol.yaml")
    run("recompute","msfeirm.cli.evaluate",["--predictions",saved/"predictions.jsonl","--protocol",saved/"protocol.yaml",
                                            "--output",directory/"recomputed"])
    original = read(directory/"evaluation/metrics.json")
    again = read(directory/"recomputed/metrics.json")
    for key in ("overall","by_members","intervals"):
        if original[key]!=again[key]:
            raise AssertionError("Saved-only metric recomputation changed")
    prediction = read(directory/"predictions/manifest.json")
    expected_counts = {str(n):len(open_prepared_dataset(prepared,"test",n)) for n in train_data.manifest["configuration"]["test_members"]}
    if prediction["record_counts"]!=expected_counts:
        raise AssertionError("Acceptance must cover every test group")
    state = torch.load(training/"checkpoints/last.pt",map_location="cpu",weights_only=True)
    if state["configs"]["data"]["train_limit"] is not None or state["configs"]["data"]["validation_limit"] is not None:
        raise AssertionError("Acceptance requires full train and validation lists")
    projection = project_training_time(timing,steps_per_epoch=steps_per_epoch,epochs=values["training"]["epochs"],
                                       checkpoint_every_steps=values["training"]["checkpoint_every_steps"],
                                       checkpoint_every_seconds=values["training"]["checkpoint_every_seconds"])
    report.update(status="DONE",formal_chapter4_run=False,global_step=target,completed_epochs=trained["completed_epochs"],
                  budget_stop_step=initial["global_step"],checkpoint=str(training/"checkpoints/last.pt"),
                  checkpoint_sha256=file_fingerprint(training/"checkpoints/last.pt"),checkpoint_roundtrip=trained["checkpoint_roundtrip"],
                  resume_verification=trained["resume_verification"],timing=timing,runtime_projection=projection,
                  prediction_counts=prediction["record_counts"],prediction_samples=prediction["sample_count"],
                  interval_counts={key:value["sample_count"] for key,value in original["intervals"].items()},
                  saved_only_recomputation="exact",elapsed_seconds=time.perf_counter()-started)
    _json(directory/"pipeline_summary.json",report)
    print(json.dumps({key:report[key] for key in ("status","dataset","global_step","prediction_samples","budget_stop_step","elapsed_seconds")}),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
