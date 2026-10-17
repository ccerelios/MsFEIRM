"""Recompute frozen metrics from JSONL predictions, without model/data access."""

import argparse
import csv
import json
from pathlib import Path
import shutil

import yaml

from msfeirm.config import PROJECT_ROOT
from msfeirm.data.prepared import _json
from msfeirm.evaluation.metrics import evaluate_slices
from msfeirm.evaluation.protocol import load_protocol
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv
from msfeirm.training.checkpoint import file_fingerprint


def read_predictions(path):
    path = Path(path)
    if path.is_dir():
        manifest = json.loads((path/"manifest.json").read_text())
        if manifest["status"]!="DONE" or manifest["format_version"]!=1:
            raise ValueError("Prediction manifest is incomplete or unsupported")
        path = path/manifest["prediction_file"]
        if file_fingerprint(path)!=manifest["prediction_sha256"]:
            raise ValueError("Prediction file hash does not match manifest")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(record.get("format_version")!=1 for record in records):
        raise ValueError("Unsupported prediction record format")
    return records,{"path":str(path.resolve()),"sha256":file_fingerprint(path),"sample_count":len(records)}


def export_tables(report,directory):
    sections = [("overall",report["overall"]),*[(f"N={key}",value) for key,value in report["by_members"].items()],
                *[(f"N={key}",value) for key,value in report["intervals"].items()]]
    columns = ("slice","sample_count","segmentation_precision","segmentation_recall","segmentation_f1",
               "segmentation_map","iou_recall_auc","activity_precision_macro","activity_recall_macro",
               "activity_f1_macro","activity_accuracy","activity_coverage","activity_end_to_end_f1","cardinality_mae")
    with (directory/"summary.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,columns)
        writer.writeheader()
        for name,metrics in sections:
            segmentation,activity = metrics["segmentation"],metrics["activity_matched"]
            writer.writerow({"slice":name,"sample_count":metrics["sample_count"],
                             **{f"segmentation_{key}":segmentation[key] for key in ("precision","recall","f1","map")},
                             "iou_recall_auc":segmentation["iou_recall_auc"],
                             **{f"activity_{key}":activity[key] for key in ("precision_macro","recall_macro","f1_macro","accuracy","coverage")},
                             "activity_end_to_end_f1":metrics["activity_end_to_end"]["f1"],"cardinality_mae":metrics["cardinality"]["mae"]})
    matrix = report["overall"]["activity_matched"]["confusion_matrix"]
    with (directory/"confusion_matrix.csv").open("w",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target\\prediction",*range(len(matrix))])
        writer.writerows([index,*row] for index,row in enumerate(matrix))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions",type=Path,nargs="+",required=True)
    parser.add_argument("--protocol",type=Path,default=PROJECT_ROOT/"configs/protocols/default.yaml")
    parser.add_argument("--member-sizes",type=int,nargs="+")
    parser.add_argument("--output",required=True)
    args = parser.parse_args(argv)
    require_project_virtualenv()
    directory = project_output_path(args.output)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Evaluation output must be empty; choose a new directory")
    records,inputs = [],[]
    for path in args.predictions:
        values,source = read_predictions(path)
        records.extend(values)
        inputs.append(source)
    if len({record["dataset"] for record in records})>1:
        parser.error("Evaluate each dataset separately because activity mappings differ")
    if args.member_sizes:
        records = [record for record in records if record["members"] in args.member_sizes]
    protocol = load_protocol(args.protocol)
    report = {"stage":"II-3","status":"DONE","inputs":inputs,"protocol":protocol.to_dict(),
              "selected_member_sizes":args.member_sizes,**evaluate_slices(records,protocol)}
    details = report["overall"].pop("samples")
    directory.mkdir(parents=True,exist_ok=True)
    temporary = directory/"matches.jsonl.tmp"
    with temporary.open("w") as stream:
        for value in details:
            stream.write(json.dumps(value,allow_nan=False)+"\n")
    temporary.replace(directory/"matches.jsonl")
    _json(directory/"metrics.json",report)
    (directory/"protocol.resolved.yaml").write_text(yaml.safe_dump({"evaluation":protocol.to_dict()},sort_keys=False))
    _json(directory/"environment.json",environment_info())
    shutil.copyfile(PROJECT_ROOT/"requirements.txt",directory/"requirements.txt")
    export_tables(report,directory)
    print(json.dumps({"status":"DONE","sample_count":report["overall"]["sample_count"],
                      "member_sizes":list(report["by_members"]),"intervals":list(report["intervals"]),
                      "metrics":str(directory/"metrics.json")},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
