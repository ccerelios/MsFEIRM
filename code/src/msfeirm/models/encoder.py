"""Connect individual features to shared Y_norm; task heads are the next stage."""

from dataclasses import dataclass

from torch import Tensor, nn

from msfeirm.config import ModelConfig
from msfeirm.models.graph_transformer import InteractionOutput, ScaleAwareGraphTransformer
from msfeirm.models.multiscale import FeatureOutput, MultiScaleFeatureExtractor


@dataclass
class EncoderOutput:
    individual: FeatureOutput
    interaction: InteractionOutput

    @property
    def node_features(self) -> Tensor:
        return self.interaction.node_features

    @property
    def member_mask(self) -> Tensor:
        return self.interaction.member_mask


class MsFEIRMEncoder(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.feature_extractor = MultiScaleFeatureExtractor(self.config)
        self.interaction = ScaleAwareGraphTransformer(self.config)

    def forward(
        self, imu: Tensor, member_mask: Tensor | None = None, *, capture_trace: bool = False,
    ) -> EncoderOutput:
        individual = self.feature_extractor(imu, member_mask, capture_trace=capture_trace)
        interaction = self.interaction(
            individual.scale_vectors, individual.member_mask, capture_trace=capture_trace,
        )
        return EncoderOutput(individual, interaction)
