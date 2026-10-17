"""Model-only loading and JSON-compatible inference records."""

import numpy as np
import torch

from msfeirm.config import ModelConfig, SegmentationConfig
from msfeirm.models import MsFEIRM
from msfeirm.training.checkpoint import file_fingerprint


def load_inference_model(checkpoint, device="cpu"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not (state.get("complete_m3") or state.get("checkpoint_type")=="real_input_interface_smoke"):
        raise ValueError("Independent inference requires a complete MsFEIRM checkpoint")
    configs = state.get("configs", {})
    model_config = configs.get("model", state.get("model_config"))
    segmentation = configs.get("segmentation", state.get("segmentation_config"))
    model = MsFEIRM(ModelConfig.from_mapping(model_config), SegmentationConfig.from_mapping(segmentation)).to(device).eval()
    model.load_state_dict(state["model_state_dict"])
    data = state.get("data_manifest", {})
    metadata = {"checkpoint_sha256":file_fingerprint(checkpoint), "checkpoint_type":state.get("checkpoint_type"),
                "trained":state.get("trained",False), "global_step":state.get("global_step"),
                "dataset":data.get("dataset"), "training_prepared_fingerprint":data.get("prepared_fingerprint"),
                "development_checkpoint":configs.get("data",{}).get("train_limit") is not None,
                "model":model.model_config.to_dict(), "segmentation":model.segmentation_config.to_dict()}
    return model, metadata


def group_confidence(probabilities, members):
    matrix = np.asarray(probabilities)[np.ix_(members,members)]
    if len(members)==1:
        return float(matrix[0,0])
    return float(matrix[~np.eye(len(members),dtype=bool)].mean())


@torch.inference_mode()
def predict_batch(model, imu, member_mask):
    """No targets are accepted by this API or passed to the model."""
    model.eval()
    output = model(imu,member_mask)
    records = []
    for index,mask in enumerate(member_mask):
        valid = mask.nonzero(as_tuple=True)[0]
        assignments = output.assignments[index,valid].cpu().tolist()
        matrix = output.segmentation.membership.probabilities[index][valid[:,None],valid[None,:]].cpu().numpy()
        groups = []
        for slot in output.activity.pooled.group_mask[index].nonzero(as_tuple=True)[0].tolist():
            group_id = int(output.activity.pooled.group_ids[index,slot])
            members = [member for member,value in enumerate(assignments) if value==group_id]
            groups.append({"id":group_id,"members":members,"confidence":group_confidence(matrix,members),
                           "activity":int(output.activity.predictions[index,slot]),
                           "activity_probabilities":output.activity.probabilities[index,slot].cpu().tolist()})
        records.append({"member_indices":valid.cpu().tolist(), "assignments":assignments,
                        "membership_probabilities":matrix.tolist(), "groups":groups,
                        "count_score":float(output.segmentation.cardinality.score[index]),
                        "continuous_count":float(output.segmentation.cardinality.continuous_count[index]),
                        "k_clusters":int(output.segmentation.cardinality.decoded_count[index])})
    return records


def package_predictions(predictions, batch, *, dataset, split, activity_names):
    records = []
    for index,prediction in enumerate(predictions):
        valid = batch.member_mask[index].nonzero(as_tuple=True)[0]
        target = None
        if batch.group_ids is not None and batch.group_labels is not None:
            ids = batch.group_ids[index,valid].cpu().tolist()
            groups = [{"id":group,"members":[member for member,value in enumerate(ids) if value==group],
                       "activity":int(batch.group_labels[index][group])} for group in sorted(set(ids)) if group>=0]
            target = {"group_ids":ids,"groups":groups,
                      "k_clusters":int(batch.k_clusters[index]) if batch.k_clusters is not None else len(groups),
                      "noise_mask":batch.noise_mask[index,valid].cpu().tolist() if batch.noise_mask is not None else None}
        records.append({"format_version":1,"sample_id":batch.sample_ids[index],"dataset":dataset,"split":split,
                        "members":len(prediction["assignments"]),"activity_names":activity_names,
                        "source_refs":batch.source_refs[index] if batch.source_refs else None,
                        "prediction":prediction,"target":target})
    return records
