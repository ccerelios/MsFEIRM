"""Scale-specific graph attention followed by global interaction modeling."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from msfeirm.config import ModelConfig
from msfeirm.models.graph import inverse_distance_graph, pair_mask, prepare_nodes


@dataclass
class AttentionOutput:
    node_features: Tensor
    attention_weights: Tensor | None = None  # [B, heads, N, N], before dropout


class WeightedGraphAttention(nn.Module):
    """Dot-product attention with log edge-weight bias and exact zero-edge masking."""

    def __init__(
        self, dim: int, heads: int = 2, *, scale_logits: bool = False,
        dropout: float = 0.0, bias: bool = True, epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if dim < 1 or heads < 1 or dim % heads:
            raise ValueError("dim must be positive and divisible by positive heads")
        if not 0 <= dropout < 1 or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Invalid dropout or epsilon")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale_logits = scale_logits
        self.epsilon = epsilon
        self.qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.output_projection = nn.Linear(dim, dim, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self, nodes: Tensor, graph_weights: Tensor, member_mask: Tensor | None = None,
        *, capture_attention: bool = False,
    ) -> AttentionOutput:
        safe_nodes, member_mask = prepare_nodes(nodes, member_mask)
        batch, members, dim = nodes.shape
        if dim != self.dim:
            raise ValueError(f"Expected node dimension {self.dim}, received {dim}")
        if graph_weights.shape != (batch, members, members):
            raise ValueError("graph_weights must have shape [B, N, N]")
        if graph_weights.device != nodes.device or not graph_weights.is_floating_point():
            raise ValueError("graph_weights must be floating-point on the nodes' device")
        valid_pairs = pair_mask(member_mask)
        safe_graph = torch.where(valid_pairs, graph_weights, 0)
        if not bool(torch.isfinite(safe_graph).all()) or bool((safe_graph < 0).any()):
            raise ValueError("Valid graph weights must be finite and nonnegative")

        # Every softmax row needs support. Valid isolated nodes fall back to self;
        # invalid query rows temporarily do so too, then are set exactly to zero.
        supported = valid_pairs & (safe_graph > 0)
        isolated = ~supported.any(dim=-1)
        diagonal = torch.eye(members, dtype=torch.bool, device=nodes.device)[None]
        fallback = isolated[:, :, None] & diagonal
        allowed = supported | fallback
        effective_graph = torch.where(fallback, 1, safe_graph)

        qkv = self.qkv(safe_nodes).reshape(batch, members, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        work_dtype = torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
        logits = query.to(work_dtype) @ key.to(work_dtype).transpose(-2, -1)
        if self.scale_logits:
            logits = logits / math.sqrt(self.head_dim)
        logits = logits + torch.log(effective_graph.to(work_dtype) + self.epsilon)[:, None]
        logits = logits.masked_fill(~allowed[:, None], -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = torch.where(valid_pairs[:, None], probabilities, 0)
        attended = self.attention_dropout(probabilities).to(value.dtype) @ value
        merged = attended.transpose(1, 2).reshape(batch, members, dim)
        projected = self.output_projection(merged)
        projected = torch.where(member_mask[:, :, None], projected, 0)
        return AttentionOutput(
            node_features=projected,
            attention_weights=probabilities if capture_attention else None,
        )


class GraphTransformerLayer(nn.Module):
    """Post-normalized attention/FFN residual block with padding after each sublayer."""

    def __init__(self, dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.attention = WeightedGraphAttention(
            dim, config.attention_heads, scale_logits=config.attention_scale,
            dropout=config.dropout, bias=config.linear_bias, epsilon=config.epsilon,
        )
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, config.ffn_expansion * dim, bias=config.linear_bias),
            nn.ReLU(),
            nn.Linear(config.ffn_expansion * dim, dim, bias=config.linear_bias),
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self, nodes: Tensor, graph_weights: Tensor, member_mask: Tensor,
        *, capture_attention: bool = False,
    ) -> AttentionOutput:
        nodes, member_mask = prepare_nodes(nodes, member_mask)
        attended = self.attention(
            nodes, graph_weights, member_mask, capture_attention=capture_attention,
        )
        hidden = self.attention_norm(nodes + self.residual_dropout(attended.node_features))
        hidden = torch.where(member_mask[:, :, None], hidden, 0)
        output = self.ffn_norm(hidden + self.residual_dropout(self.ffn(hidden)))
        return AttentionOutput(
            node_features=torch.where(member_mask[:, :, None], output, 0),
            attention_weights=attended.attention_weights,
        )


@dataclass
class GraphTransformerOutput:
    node_features: Tensor
    attention_weights: tuple[Tensor, ...] = ()


class GraphTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, config: ModelConfig) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("GraphTransformer depth must be positive")
        self.layers = nn.ModuleList([GraphTransformerLayer(dim, config) for _ in range(depth)])

    def forward(
        self, nodes: Tensor, graph_weights: Tensor, member_mask: Tensor,
        *, capture_attention: bool = False,
    ) -> GraphTransformerOutput:
        attention_weights = []
        for layer in self.layers:
            output = layer(nodes, graph_weights, member_mask, capture_attention=capture_attention)
            nodes = output.node_features
            if output.attention_weights is not None:
                attention_weights.append(output.attention_weights)
        return GraphTransformerOutput(nodes, tuple(attention_weights))


@dataclass
class InteractionTrace:
    local_graphs: tuple[Tensor, ...]
    local_features: tuple[Tensor, ...]
    local_attention: tuple[tuple[Tensor, ...], ...]
    concatenated_features: Tensor
    global_graph: Tensor
    global_attention: tuple[Tensor, ...]


@dataclass
class InteractionOutput:
    node_features: Tensor                 # final Y_norm, [B, N, D]
    member_mask: Tensor
    trace: InteractionTrace | None = None


class ScaleAwareGraphTransformer(nn.Module):
    """Four independent local stacks, then a graph on concatenated node features."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.local_transformers = nn.ModuleList([
            GraphTransformer(self.config.scale_embedding_dim, self.config.local_depth, self.config)
            for _ in range(self.config.num_scales)
        ])
        self.global_transformer = GraphTransformer(
            self.config.node_dim, self.config.global_depth, self.config,
        )
        self.final_norm = nn.LayerNorm(self.config.node_dim)

    def forward(
        self, scale_vectors: Sequence[Tensor], member_mask: Tensor | None = None,
        *, capture_trace: bool = False,
    ) -> InteractionOutput:
        if isinstance(scale_vectors, Tensor) or len(scale_vectors) != self.config.num_scales:
            raise ValueError(f"Expected {self.config.num_scales} separate scale vectors")
        first, member_mask = prepare_nodes(scale_vectors[0], member_mask)
        expected_shape = (*first.shape[:2], self.config.scale_embedding_dim)
        local_graphs, local_features, local_attention = [], [], []
        for features, transformer in zip(scale_vectors, self.local_transformers, strict=True):
            if features.shape != expected_shape:
                raise ValueError(f"Each scale must have shape {expected_shape}")
            features, _ = prepare_nodes(features, member_mask)
            graph = inverse_distance_graph(features, member_mask, epsilon=self.config.epsilon)
            output = transformer(features, graph, member_mask, capture_attention=capture_trace)
            local_features.append(output.node_features)
            if capture_trace:
                local_graphs.append(graph)
                local_attention.append(output.attention_weights)

        concatenated = torch.cat(local_features, dim=-1)
        global_graph = inverse_distance_graph(concatenated, member_mask, epsilon=self.config.epsilon)
        global_output = self.global_transformer(
            concatenated, global_graph, member_mask, capture_attention=capture_trace,
        )
        normalized = self.final_norm(global_output.node_features)
        normalized = torch.where(member_mask[:, :, None], normalized, 0)
        trace = None
        if capture_trace:
            trace = InteractionTrace(
                tuple(local_graphs), tuple(local_features), tuple(local_attention),
                concatenated, global_graph, global_output.attention_weights,
            )
        return InteractionOutput(normalized, member_mask, trace)
