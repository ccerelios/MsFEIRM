"""Set-based segmentation and activity metrics from saved predictions only."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from msfeirm.evaluation.protocol import EvaluationProtocol


def threshold_matching(iou, threshold):
    """Maximum eligible match count, then maximum eligible total IoU."""
    iou = np.asarray(iou,dtype=np.float64)
    if iou.ndim!=2:
        raise ValueError("IoU must be a matrix")
    if not iou.size:
        return []
    eligible = (iou>=threshold) & (iou>0)
    reward = eligible * (min(iou.shape)+1) + np.where(eligible,iou,0)
    rows,columns = linear_sum_assignment(reward,maximize=True)
    return [(int(row),int(column),float(iou[row,column])) for row,column in zip(rows,columns,strict=True) if eligible[row,column]]


def prf(tp,fp,fn):
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),
            "precision":tp/(tp+fp) if tp+fp else 0., "recall":tp/(tp+fn) if tp+fn else 0.,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def _groups(values):
    # Canonical set order removes dependence on arbitrary cluster numbering.
    return sorted([value for value in values if value["members"]],key=lambda value:tuple(sorted(value["members"])))


def _geometry(record):
    if record.get("target") is None:
        raise ValueError("Evaluation requires targets stored with each prediction")
    predicted,target = _groups(record["prediction"]["groups"]),_groups(record["target"]["groups"])
    iou = np.zeros((len(predicted),len(target)),dtype=np.float64)
    for row,left in enumerate(predicted):
        left = set(left["members"])
        for column,right in enumerate(target):
            right = set(right["members"])
            iou[row,column] = len(left & right) / len(left | right)
    return predicted,target,iou


def all_point_ap(true_positive, false_positive, target_count):
    if not target_count or not len(true_positive):
        return 0.
    tp,fp = np.cumsum(true_positive),np.cumsum(false_positive)
    recall = np.concatenate(([0.],tp/target_count,[1.]))
    precision = np.concatenate(([0.],tp/np.maximum(tp+fp,1),[0.]))
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall)*envelope[1:]))


def _average_precision(records, geometries, threshold):
    ranked = []
    for sample,(record,(predicted,_,_)) in enumerate(zip(records,geometries,strict=True)):
        for group,value in enumerate(predicted):
            ranked.append((-value["confidence"],str(record["sample_id"]),group,sample))
    ranked.sort()
    used = [set() for _ in records]
    true_positive,false_positive = [],[]
    for _,_,group,sample in ranked:
        iou = geometries[sample][2]
        candidates = [target for target in range(iou.shape[1]) if target not in used[sample] and iou[group,target]>0 and iou[group,target]>=threshold]
        matched = bool(candidates)
        if matched:
            target = max(candidates,key=lambda target:(iou[group,target],-target))
            used[sample].add(target)
        true_positive.append(int(matched))
        false_positive.append(int(not matched))
    target_count = sum(len(target) for _,target,_ in geometries)
    return {"ap":all_point_ap(true_positive,false_positive,target_count),
            "tp":sum(true_positive),"fp":sum(false_positive),"target_groups":target_count}


def _classification(confusion):
    true_positive = confusion.diagonal()
    support,predicted = confusion.sum(axis=1),confusion.sum(axis=0)
    precision = np.divide(true_positive,predicted,out=np.zeros_like(true_positive,dtype=float),where=predicted>0)
    recall = np.divide(true_positive,support,out=np.zeros_like(true_positive,dtype=float),where=support>0)
    f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros_like(precision),where=precision+recall>0)
    count = int(confusion.sum())
    return {"precision_macro":float(precision.mean()) if count else None,
            "recall_macro":float(recall.mean()) if count else None,
            "f1_macro":float(f1.mean()) if count else None,
            "accuracy":float(true_positive.sum()/count) if count else None,
            "confusion_matrix":confusion.tolist(),"support":support.tolist(),"matched_groups":count,
            "per_class":[{"class_id":index,"precision":float(precision[index]),"recall":float(recall[index]),"f1":float(f1[index])}
                         for index in range(len(confusion))]}


def evaluate_records(records, protocol=None, *, include_details=False):
    protocol = protocol or EvaluationProtocol()
    records = list(records)
    geometries = [_geometry(record) for record in records]
    confusion = np.zeros((protocol.num_classes,protocol.num_classes),dtype=np.int64)
    tp=predicted_count=target_count=active_tp=active_predicted=active_target=0
    pair_tp=pair_fp=pair_fn=0
    count_error = 0.
    details = []
    for record,(predicted,target,iou) in zip(records,geometries,strict=True):
        matches = threshold_matching(iou,protocol.iou_threshold)
        tp += len(matches)
        predicted_count += len(predicted)
        target_count += len(target)
        valid_target = [index for index,value in enumerate(target) if value["activity"]!=protocol.ignore_index]
        valid_predicted = [index for index,value in enumerate(predicted) if len(value["members"])>=protocol.min_activity_group_size]
        active_target += len(valid_target)
        active_predicted += len(valid_predicted)
        for left,right,_ in matches:
            if target[right]["activity"]!=protocol.ignore_index:
                confusion[target[right]["activity"],predicted[left]["activity"]] += 1
        activity_iou = iou[np.ix_(valid_predicted,valid_target)].copy()
        for row,left in enumerate(valid_predicted):
            for column,right in enumerate(valid_target):
                if predicted[left]["activity"]!=target[right]["activity"]:
                    activity_iou[row,column] = 0
        activity_matches = threshold_matching(activity_iou,protocol.iou_threshold)
        active_tp += len(activity_matches)
        n = record["members"]
        probabilities = np.asarray(record["prediction"]["membership_probabilities"]).reshape(n,n)
        ids = np.asarray(record["target"]["group_ids"])
        truth = (ids[:,None]==ids[None,:]) & (ids[:,None]>=0) & (ids[None,:]>=0)
        upper = np.triu(np.ones((n,n),dtype=bool),k=1)
        predicted_pairs = probabilities[upper]>=protocol.member_probability_threshold
        target_pairs = truth[upper]
        pair_tp += int((predicted_pairs & target_pairs).sum())
        pair_fp += int((predicted_pairs & ~target_pairs).sum())
        pair_fn += int((~predicted_pairs & target_pairs).sum())
        count_error += abs(record["prediction"]["k_clusters"]-record["target"]["k_clusters"])
        if include_details:
            details.append({"sample_id":record["sample_id"],"members":n,"iou_matrix":iou.tolist(),
                            "predicted_group_order":[value["id"] for value in predicted],
                            "target_group_order":[value["id"] for value in target],
                            "segmentation":prf(len(matches),len(predicted)-len(matches),len(target)-len(matches)),
                            "matches":[{"predicted_group_id":predicted[left]["id"],"target_group_id":target[right]["id"],
                                        "iou":score,"predicted_activity":predicted[left]["activity"],"target_activity":target[right]["activity"]}
                                       for left,right,score in matches],
                            "activity_end_to_end":prf(len(activity_matches),len(valid_predicted)-len(activity_matches),len(valid_target)-len(activity_matches))})
    aps = {f"{threshold:.2f}":_average_precision(records,geometries,threshold) for threshold in protocol.ap_thresholds}
    recall_curve = [sum(len(threshold_matching(iou,threshold)) for _,_,iou in geometries)/target_count if target_count else 0.
                    for threshold in protocol.auc_thresholds]
    matched = _classification(confusion)
    matched.update(target_activity_groups=active_target,coverage=int(confusion.sum())/active_target if active_target else None)
    result = {"sample_count":len(records),
              "segmentation":{**prf(tp,predicted_count-tp,target_count-tp),"iou_threshold":protocol.iou_threshold,
                              "map":float(np.mean([value["ap"] for value in aps.values()])),"ap_by_iou":aps,
                              "iou_recall_auc":float(np.trapezoid(recall_curve,protocol.auc_thresholds)),
                              "iou_recall_curve":{"thresholds":list(protocol.auc_thresholds),"recall":recall_curve}},
              "membership_pairs":{**prf(pair_tp,pair_fp,pair_fn),"probability_threshold":protocol.member_probability_threshold},
              "cardinality":{"mae":count_error/len(records) if records else None},
              "activity_matched":matched,
              "activity_end_to_end":prf(active_tp,active_predicted-active_tp,active_target-active_tp)}
    if include_details:
        result["samples"] = details
    return result


def evaluate_slices(records, protocol=None):
    protocol = protocol or EvaluationProtocol()
    records = list(records)
    return {"overall":evaluate_records(records,protocol,include_details=True),
            "by_members":{str(n):evaluate_records([record for record in records if record["members"]==n],protocol)
                          for n in sorted({record["members"] for record in records})},
            "intervals":{f"{low}-{high}":evaluate_records([record for record in records if low<=record["members"]<=high],protocol)
                         for low,high in protocol.member_intervals if any(low<=record["members"]<=high for record in records)}}
