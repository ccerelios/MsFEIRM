"""Complete MsFEIRM forward path: IMU -> predicted sets -> activity logits."""

from dataclasses import dataclass

from torch import Tensor, nn

from msfeirm.config import ModelConfig, SegmentationConfig
from msfeirm.models.pooling import ActivityOutput, SubgroupActivityHead
from msfeirm.models.segmentation import MsFEIRMSegmenter, SegmentationOutput


@dataclass
class MsFEIRMOutput:
    segmentation: SegmentationOutput
    activity: ActivityOutput

    @property
    def node_features(self) -> Tensor:
        return self.segmentation.node_features

    @property
    def member_mask(self) -> Tensor:
        return self.segmentation.member_mask

    @property
    def assignments(self) -> Tensor:
        assert self.segmentation.clustering is not None
        return self.segmentation.clustering.assignments


class MsFEIRM(nn.Module):
    def __init__(
        self, model_config: ModelConfig | None = None,
        segmentation_config: SegmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.segmenter = MsFEIRMSegmenter(self.model_config, self.segmentation_config)
        self.activity_head = SubgroupActivityHead(
            self.model_config.node_dim, self.model_config.num_classes,
            bias=self.model_config.linear_bias,
        )

    def forward(
        self, imu: Tensor, member_mask: Tensor | None = None, *, capture_trace: bool = False,
    ) -> MsFEIRMOutput:
        segmentation = self.segmenter(imu, member_mask, capture_trace=capture_trace)
        assert segmentation.clustering is not None
        activity = self.activity_head(
            segmentation.node_features, segmentation.clustering.assignments, segmentation.member_mask,
        )
        return MsFEIRMOutput(segmentation, activity)
