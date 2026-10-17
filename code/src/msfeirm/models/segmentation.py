"""Shared encoder plus membership, cardinality and discrete grouping (I-3a)."""

from dataclasses import dataclass

from torch import Tensor, nn

from msfeirm.config import ModelConfig, SegmentationConfig
from msfeirm.models.clustering import ClusteringOutput, SpectralClusterer
from msfeirm.models.encoder import EncoderOutput, MsFEIRMEncoder
from msfeirm.models.heads import (
    CardinalityOutput, MembershipOutput, PairwiseMembershipHead, SubgroupCardinalityHead,
)


@dataclass
class SegmentationOutput:
    encoder: EncoderOutput
    membership: MembershipOutput
    cardinality: CardinalityOutput
    clustering: ClusteringOutput | None

    @property
    def node_features(self) -> Tensor:
        return self.encoder.node_features

    @property
    def member_mask(self) -> Tensor:
        return self.encoder.member_mask


class MsFEIRMSegmenter(nn.Module):
    def __init__(
        self, model_config: ModelConfig | None = None,
        segmentation_config: SegmentationConfig | None = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.segmentation_config = segmentation_config or SegmentationConfig()
        self.encoder = MsFEIRMEncoder(self.model_config)
        self.membership_head = PairwiseMembershipHead(
            self.model_config.node_dim, self.segmentation_config.pair_hidden_dim,
            bias=self.model_config.linear_bias,
        )
        self.cardinality_head = SubgroupCardinalityHead(
            self.model_config.node_dim, self.segmentation_config,
            bias=self.model_config.linear_bias,
        )
        self.clusterer = SpectralClusterer(self.segmentation_config)

    def forward(
        self, imu: Tensor, member_mask: Tensor | None = None, *,
        perform_clustering: bool = True, capture_trace: bool = False,
    ) -> SegmentationOutput:
        encoder = self.encoder(imu, member_mask, capture_trace=capture_trace)
        nodes, mask = encoder.node_features, encoder.member_mask
        membership = self.membership_head(nodes, mask)
        cardinality = self.cardinality_head(nodes, mask)
        clustering = None
        if perform_clustering:
            clustering = self.clusterer(
                nodes, cardinality.decoded_count, mask, capture_trace=capture_trace,
            )
        return SegmentationOutput(encoder, membership, cardinality, clustering)
