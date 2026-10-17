"""Validate fixed/variable-size prepared sensor groups through complete M3."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from msfeirm.config import PROJECT_ROOT, LossConfig, ModelConfig, SegmentationConfig
from msfeirm.data import GroupBatch, collate_groups
from msfeirm.data.prepared import _fingerprint, open_prepared_dataset, verify_prepared_files
from msfeirm.models import MsFEIRM
from msfeirm.models.losses import MsFEIRMJointLoss
from msfeirm.runtime import environment_info, project_output_path, require_project_virtualenv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=PROJECT_ROOT / "data/processed/ut_group")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    sizes = parser.add_mutually_exclusive_group()
    sizes.add_argument("--members", type=int, help="Defaults to the dataset training member count")
    sizes.add_argument("--member-sizes", type=int, nargs="+", help="Explicit member count per sample, for mixed-N batches")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if args.samples < 1:
        parser.error("samples must be positive")
    manifest = verify_prepared_files(args.prepared)
    default_members = manifest["configuration"]["train_members"] if args.members is None else args.members
    member_sizes = args.member_sizes or [default_members] * args.samples
    datasets, consumed, samples = {}, {}, []
    for members in member_sizes:
        if members not in datasets:
            datasets[members] = open_prepared_dataset(args.prepared, args.split, members)
        dataset = datasets[members]
        index = consumed.get(members, 0)
        if len(dataset) <= index:
            parser.error("Not enough prepared groups for the requested sample batch")
        samples.append(dataset[index])
        consumed[members] = index + 1
    dataset = next(iter(datasets.values()))
    if any(value.k_bounds != dataset.k_bounds for value in datasets.values()):
        parser.error("Mixed member counts must share a fixed model K protocol")
    subset_fingerprints = {value.key: value.fingerprint for value in datasets.values()}
    batch_fingerprint = (dataset.fingerprint if len(datasets) == 1 else
                         _fingerprint({"prepared": manifest["fingerprint"], "subsets": subset_fingerprints}))
    name = manifest["dataset"]
    stage = "II-1a" if name == "ut_group" else "II-1b"
    directory = project_output_path(args.output or ("runs/ut_data_smoke" if name == "ut_group" else "runs/wb_data_smoke"))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    batch = collate_groups(samples).to(args.device)
    batch.imu.requires_grad_()
    model_config = ModelConfig(input_channels=batch.imu.shape[2], window_size=batch.imu.shape[3],
                               num_classes=len(dataset.manifest["configuration"]["activity_names"]))
    segmentation_config = SegmentationConfig(k_min=dataset.k_bounds[0], k_max=dataset.k_bounds[1])
    loss_config = LossConfig(ignore_index=dataset.manifest["configuration"]["grouping"]["ignore_index"])
    model = MsFEIRM(model_config, segmentation_config).to(args.device).eval()
    criterion = MsFEIRMJointLoss(segmentation_config, loss_config)
    output = model(batch.imu, batch.member_mask)
    losses = criterion(output, batch)
    losses.total.backward()
    if batch.imu.grad is None or not bool(torch.isfinite(batch.imu.grad).all()):
        raise RuntimeError("Input gradient is missing/non-finite")
    if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
        raise RuntimeError("Model gradient is missing/non-finite")
    components = {"encoder":model.segmenter.encoder, "member":model.segmenter.membership_head,
                  "cardinality":model.segmenter.cardinality_head, "activity":model.activity_head}
    gradient_sums = {name: sum(float(parameter.grad.abs().sum()) for parameter in component.parameters())
                     for name, component in components.items()}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model_weights.pt"
    state = {"stage":stage, "dataset":name, "checkpoint_type":"real_input_interface_smoke", "trained":False,
             "model_config":model_config.to_dict(), "segmentation_config":segmentation_config.to_dict(),
             "loss_config":loss_config.to_dict(), "prepared_fingerprint":batch_fingerprint,
             "prepared_subset_fingerprints":subset_fingerprints,
             "model_state_dict":{key:value.detach().cpu() for key,value in model.state_dict().items()},
             "batch":{"imu":batch.imu.detach().cpu(), "member_mask":batch.member_mask.cpu(),
                      "sample_ids":batch.sample_ids, "group_ids":batch.group_ids.cpu(),
                      "group_labels":tuple(value.cpu() for value in batch.group_labels),
                      "pair_targets":batch.pair_targets.cpu(), "k_clusters":batch.k_clusters.cpu(),
                      "noise_mask":batch.noise_mask.cpu()},
             "expected_activity_logits":output.activity.logits.detach().cpu(),
             "expected_assignments":output.assignments.cpu(),
             "expected_losses":torch.stack((losses.total,losses.group,losses.cardinality,losses.member)).detach().cpu()}
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    state = torch.load(path, map_location=args.device, weights_only=True)
    restored = MsFEIRM(ModelConfig.from_mapping(state["model_config"]), SegmentationConfig.from_mapping(state["segmentation_config"])).to(args.device).eval()
    restored.load_state_dict(state["model_state_dict"])
    restored_batch = GroupBatch(**state["batch"])
    with torch.no_grad():
        actual = restored(restored_batch.imu, restored_batch.member_mask)
        actual_losses = criterion(actual, restored_batch)
    torch.testing.assert_close(actual.activity.logits, state["expected_activity_logits"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.stack((actual_losses.total,actual_losses.group,actual_losses.cardinality,actual_losses.member)), state["expected_losses"], rtol=1e-5, atol=1e-6)
    if not torch.equal(actual.assignments, state["expected_assignments"]):
        raise AssertionError("Restored grouping changed")
    summary = {"stage":stage, "dataset":name, "mode":"real_input_eval_with_gradients", "trained":False,
               "prepared_root":str(dataset.root), "prepared_fingerprint":batch_fingerprint,
               "prepared_subset_fingerprints":subset_fingerprints, "members":member_sizes,
               "split":args.split, "shapes":{"imu":list(batch.imu.shape), "Y_norm":list(output.node_features.shape),
                                                "activity_logits":list(output.activity.logits.shape)},
               "k_bounds":list(dataset.k_bounds), "synthetic_group_composition_on_real_windows":True,
               "finite_input_gradient":True, "finite_parameter_gradients":True,
               "component_gradient_absolute_sums":gradient_sums, "losses":losses.detached_metrics(),
               "checkpoint":str(path.relative_to(PROJECT_ROOT)), "checkpoint_roundtrip":True}
    (directory / "shape_trace.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "environment.json").write_text(json.dumps(environment_info(), indent=2) + "\n")
    (directory / "config.json").write_text(json.dumps({"model":model_config.to_dict(), "segmentation":segmentation_config.to_dict(), "loss":loss_config.to_dict()}, indent=2) + "\n")
    shutil.copyfile(PROJECT_ROOT / "requirements.txt", directory / "requirements.txt")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
