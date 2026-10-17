"""Save independent complete-M3 predictions and evaluation targets as JSONL."""

import argparse
import json
from pathlib import Path
import shutil
import time

import torch
import yaml

from msfeirm.config import PROJECT_ROOT
from msfeirm.data import collate_groups
from msfeirm.data.prepared import _json, open_prepared_dataset, verify_prepared_files
from msfeirm.evaluation.prediction import load_inference_model, package_predictions, predict_batch
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv
from msfeirm.training.checkpoint import execution_fingerprint, file_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--prepared",type=Path,required=True)
    parser.add_argument("--split",choices=("train","validation","test"),default="test")
    parser.add_argument("--members",type=int,nargs="+")
    parser.add_argument("--limit",type=int,help="Development sample limit per member count")
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--device",choices=("cpu","cuda","mps"),default="cpu")
    parser.add_argument("--output",required=True)
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if min(args.batch_size,args.threads)<1 or args.limit is not None and args.limit<1:
        parser.error("batch-size, threads and limit must be positive")
    directory = project_output_path(args.output)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Prediction output must be empty; choose a new directory")
    prepared = verify_prepared_files(args.prepared)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    model,metadata = load_inference_model(args.checkpoint,args.device)
    if metadata["dataset"] is not None and metadata["dataset"]!=prepared["dataset"]:
        parser.error("Checkpoint and prepared dataset must use the same activity mapping")
    expected = (prepared["window_shape"][1],prepared["window_shape"][2],len(prepared["configuration"]["activity_names"]))
    if (model.model_config.input_channels,model.model_config.window_size,model.model_config.num_classes)!=expected:
        parser.error("Checkpoint input dimensions/classes do not match prepared windows")
    sizes = list(dict.fromkeys(args.members or (prepared["configuration"]["test_members"] if args.split=="test" else [prepared["configuration"]["train_members"]])))
    datasets = [open_prepared_dataset(args.prepared,args.split,members) for members in sizes]
    if any(dataset.k_bounds!=(model.segmentation_config.k_min,model.segmentation_config.k_max) for dataset in datasets):
        parser.error("Checkpoint and data must share the fixed model K protocol")
    directory.mkdir(parents=True,exist_ok=True)
    config = {"checkpoint":str(args.checkpoint.resolve()),"prepared":str(args.prepared.resolve()),"split":args.split,
              "members":sizes,"limit_per_member_count":args.limit,"batch_size":args.batch_size,
              "seed":args.seed,"threads":args.threads,"device":args.device,"checkpoint_metadata":metadata}
    (directory/"config.resolved.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
    _json(directory/"environment.json",environment_info())
    shutil.copyfile(PROJECT_ROOT/"requirements.txt",directory/"requirements.txt")
    manifest = {"stage":"II-3","format_version":1,"status":"IN_PROGRESS","dataset":prepared["dataset"],
                "configuration":config,"prepared_fingerprint":prepared["fingerprint"],
                "subset_fingerprints":{dataset.key:dataset.fingerprint for dataset in datasets},
                "fingerprints":execution_fingerprint(),"record_counts":{str(n):0 for n in sizes}}
    _json(directory/"manifest.json",manifest)
    started = time.monotonic()
    temporary = directory/"predictions.jsonl.tmp"
    with temporary.open("w") as stream:
        for dataset in datasets:
            count = len(dataset) if args.limit is None else min(args.limit,len(dataset))
            for start in range(0,count,args.batch_size):
                batch = collate_groups([dataset[index] for index in range(start,min(start+args.batch_size,count))]).to(args.device)
                predictions = predict_batch(model,batch.imu,batch.member_mask)
                records = package_predictions(predictions,batch,dataset=prepared["dataset"],split=args.split,
                                               activity_names=prepared["configuration"]["activity_names"])
                for record in records:
                    stream.write(json.dumps(record,allow_nan=False)+"\n")
                stream.flush()
                manifest["record_counts"][str(dataset.members)] += len(records)
                _json(directory/"manifest.json",manifest)
    path = directory/"predictions.jsonl"
    temporary.replace(path)
    manifest.update(status="DONE",prediction_file=path.name,prediction_sha256=file_fingerprint(path),
                    sample_count=sum(manifest["record_counts"].values()),elapsed_seconds=time.monotonic()-started)
    _json(directory/"manifest.json",manifest)
    print(json.dumps({key:manifest[key] for key in ("status","dataset","sample_count","record_counts","prediction_sha256")},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
