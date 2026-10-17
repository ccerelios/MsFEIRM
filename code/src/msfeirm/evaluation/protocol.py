"""Serializable evaluation rules, independent of training supervision matching."""

from dataclasses import asdict, dataclass
import math

from msfeirm.config import load_yaml


@dataclass(frozen=True)
class EvaluationProtocol:
    version: int = 1
    iou_threshold: float = .5
    ap_thresholds: tuple[float, ...] = tuple(round(value/100, 2) for value in range(50,100,5))
    auc_thresholds: tuple[float, ...] = tuple(round(value/100, 2) for value in range(101))
    member_probability_threshold: float = .5
    min_activity_group_size: int = 2
    num_classes: int = 9
    ignore_index: int = -100
    member_intervals: tuple[tuple[int, int], ...] = ((13,15),(16,18),(19,21))

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("Unsupported evaluation protocol version")
        for values in ((self.iou_threshold,), self.ap_thresholds, self.auc_thresholds, (self.member_probability_threshold,)):
            if not values or any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise ValueError("Evaluation thresholds must be nonempty finite values in [0,1]")
        if any(right <= left for left,right in zip(self.auc_thresholds,self.auc_thresholds[1:])):
            raise ValueError("AUC thresholds must be strictly increasing")
        if len(self.auc_thresholds)<2 or self.auc_thresholds[0]!=0 or self.auc_thresholds[-1]!=1:
            raise ValueError("AUC thresholds must span [0,1]")
        if min(self.num_classes,self.min_activity_group_size)<1:
            raise ValueError("num_classes and min_activity_group_size must be positive")
        if any(low<1 or high<low for low,high in self.member_intervals):
            raise ValueError("Member intervals must contain ordered positive bounds")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, values):
        values = dict(values)
        for name in ("ap_thresholds","auc_thresholds"):
            if name in values:
                values[name] = tuple(values[name])
        if "member_intervals" in values:
            values["member_intervals"] = tuple(tuple(pair) for pair in values["member_intervals"])
        return cls(**values)


def load_protocol(path):
    values = load_yaml(path)
    return EvaluationProtocol.from_mapping(values.get("evaluation", values))
