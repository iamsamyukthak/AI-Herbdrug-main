"""Proposed heterogeneous GAT → GraphSAGE → SAGPool interaction model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    GATConv,
    GraphConv,
    HeteroConv,
    SAGPooling,
    SAGEConv,
    global_mean_pool,
)

from model.heterogeneous_graph import (
    HeterogeneousGraph,
    normalize_entity_name,
)


NODE_TYPES = ("Herb", "Compound", "Drug", "Protein")
RELATION_TYPES = (
    ("Herb", "contains", "Compound"),
    ("Compound", "associated_with", "Protein"),
    ("Drug", "targets", "Protein"),
    ("Herb", "interacts_with", "Drug"),
)


@dataclass
class SAGPoolProjection:
    """Deterministic homogeneous view used only by SAGPooling.

    Node and edge IDs map back to the original typed graph. The original
    ``HeteroData`` is never modified.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    node_type_ids: torch.Tensor
    node_local_ids: torch.Tensor
    node_types: tuple[str, ...]
    node_names: tuple[str, ...]
    node_offsets: dict[str, int]
    edge_relation_ids: torch.Tensor
    edge_direction_ids: torch.Tensor
    relation_types: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class PredictionThresholds:
    """Configurable display cutoffs; they are not scientifically validated."""

    low_upper_bound: float = 0.33
    medium_upper_bound: float = 0.66

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_upper_bound < self.medium_upper_bound <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= low_upper_bound "
                "< medium_upper_bound <= 1"
            )

    def label(self, score: float) -> str:
        """Map a continuous score to a display label using these cutoffs."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("prediction score must be in [0, 1]")
        if score < self.low_upper_bound:
            return "LOW"
        if score < self.medium_upper_bound:
            return "MEDIUM"
        return "HIGH"


def _message_passing_relations():
    relations = list(RELATION_TYPES)
    relations.extend(
        (destination, f"rev_{relation}", source)
        for source, relation, destination in RELATION_TYPES
    )
    return tuple(relations)


def project_heterogeneous_graph(
    graph: HeteroData,
    node_embeddings: Mapping[str, torch.Tensor],
) -> SAGPoolProjection:
    """Create a reproducible typed-node/relation-aware view for SAGPooling.

    Nodes are concatenated in ``NODE_TYPES`` order; within each type, the
    existing local node-ID order is retained. Edges follow ``RELATION_TYPES``
    order, with each relation's forward edges followed by its reverse edges.
    Edge relation and direction IDs, along with node type/local-ID mappings,
    remain available in the returned projection.
    """
    offsets: dict[str, int] = {}
    ordered_embeddings = []
    node_type_ids = []
    node_local_ids = []
    node_types = []
    node_names = []
    next_offset = 0

    for type_id, node_type in enumerate(NODE_TYPES):
        node_x = node_embeddings[node_type]
        count = node_x.shape[0]
        offsets[node_type] = next_offset
        ordered_embeddings.append(node_x)
        node_type_ids.append(
            torch.full(
                (count,),
                type_id,
                dtype=torch.long,
                device=node_x.device,
            )
        )
        node_local_ids.append(
            torch.arange(count, dtype=torch.long, device=node_x.device)
        )
        node_types.extend([node_type] * count)
        names = graph[node_type].get("name")
        if names is None:
            names = [str(index) for index in range(count)]
        if len(names) != count:
            raise ValueError(f"{node_type} name mapping does not match its node count")
        node_names.extend(str(name) for name in names)
        next_offset += count

    device = ordered_embeddings[0].device
    global_edges = []
    edge_relation_ids = []
    edge_direction_ids = []
    for relation_id, (source, relation, destination) in enumerate(RELATION_TYPES):
        edge_type = (source, relation, destination)
        if edge_type not in graph.edge_types:
            raise ValueError(f"Graph is missing relation store {edge_type}")

        local_edges = graph[edge_type].edge_index.to(device)
        offsets_tensor = local_edges.new_tensor(
            [[offsets[source]], [offsets[destination]]]
        )
        forward_edges = local_edges + offsets_tensor
        reverse_edges = forward_edges.flip(0).contiguous()
        global_edges.extend((forward_edges, reverse_edges))
        edge_count = forward_edges.shape[1]
        edge_relation_ids.extend(
            (
                torch.full(
                    (edge_count,), relation_id, dtype=torch.long, device=device
                ),
                torch.full(
                    (edge_count,), relation_id, dtype=torch.long, device=device
                ),
            )
        )
        edge_direction_ids.extend(
            (
                torch.zeros(edge_count, dtype=torch.long, device=device),
                torch.ones(edge_count, dtype=torch.long, device=device),
            )
        )

    features = torch.cat(ordered_embeddings, dim=0)
    if global_edges:
        edge_index = torch.cat(global_edges, dim=1)
        relation_ids = torch.cat(edge_relation_ids)
        direction_ids = torch.cat(edge_direction_ids)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        relation_ids = torch.empty((0,), dtype=torch.long, device=device)
        direction_ids = torch.empty((0,), dtype=torch.long, device=device)

    return SAGPoolProjection(
        x=features,
        edge_index=edge_index,
        batch=torch.zeros(features.shape[0], dtype=torch.long, device=device),
        node_type_ids=torch.cat(node_type_ids),
        node_local_ids=torch.cat(node_local_ids),
        node_types=tuple(node_types),
        node_names=tuple(node_names),
        node_offsets=offsets,
        edge_relation_ids=relation_ids,
        edge_direction_ids=direction_ids,
        relation_types=RELATION_TYPES,
    )


class HerbDrugHGNN(nn.Module):
    """Score herb-drug pairs from a typed graph and its node features.

    ``forward`` returns one sigmoid score in [0, 1] per input pair. PyG's
    ``SAGPooling`` is homogeneous, so relation-aware heterogeneous GAT and
    GraphSAGE layers run first. A reproducible joint topology is then built
    for pooling with explicit node-type and edge-relation mappings. Pooled
    summaries remain separated by node type; the source ``HeteroData`` remains
    typed and unmodified.
    """

    def __init__(
        self,
        in_channels: Mapping[str, int],
        hidden_channels: int = 32,
        attention_heads: int = 2,
        pool_ratio: float = 0.5,
        dropout: float = 0.2,
        use_graphsage: bool = True,
        use_sagpool: bool = True,
        use_mlp: bool = True,
        num_outputs: int = 1,
    ) -> None:
        super().__init__()

        missing_node_types = set(NODE_TYPES) - set(in_channels)
        if missing_node_types:
            raise ValueError(
                "in_channels is missing node types: "
                f"{sorted(missing_node_types)}"
            )
        if hidden_channels % attention_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by attention_heads"
            )
        if not 0.0 < pool_ratio <= 1.0:
            raise ValueError("pool_ratio must be in the interval (0, 1]")
        if num_outputs < 1:
            raise ValueError("num_outputs must be at least 1")

        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.num_outputs = num_outputs
        self.use_graphsage = use_graphsage
        self.use_sagpool = use_sagpool
        self.use_mlp = use_mlp
        self.input_projection = nn.ModuleDict(
            {
                node_type: nn.Linear(in_channels[node_type], hidden_channels)
                for node_type in NODE_TYPES
            }
        )

        edge_types = _message_passing_relations()
        self.gat = HeteroConv(
            {
                edge_type: GATConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels // attention_heads,
                    heads=attention_heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
                for edge_type in edge_types
            },
            aggr="sum",
        )
        self.graphsage = (
            HeteroConv(
                {
                    edge_type: SAGEConv(
                        (hidden_channels, hidden_channels),
                        hidden_channels,
                        aggr="mean",
                    )
                    for edge_type in edge_types
                },
                aggr="sum",
            )
            if use_graphsage
            else None
        )
        self.sag_pool = (
            SAGPooling(
                hidden_channels,
                ratio=pool_ratio,
                GNN=GraphConv,
            )
            if use_sagpool
            else None
        )

        pooled_context_width = hidden_channels * len(NODE_TYPES)
        self.pooled_context_projection = (
            nn.ModuleDict(
                {
                    node_type: nn.Linear(pooled_context_width, hidden_channels)
                    for node_type in ("Herb", "Drug")
                }
            )
            if use_sagpool
            else None
        )
        pair_width = hidden_channels * 2
        if use_mlp:
            self.interaction_mlp = nn.Sequential(
                nn.Linear(pair_width, hidden_channels * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, num_outputs),
            )
        else:
            self.interaction_mlp = nn.Linear(pair_width, num_outputs)

    @classmethod
    def from_graph(
        cls,
        graph: HeteroData,
        **model_kwargs,
    ) -> "HerbDrugHGNN":
        """Create a model using each node type's feature width in ``graph``."""
        in_channels = {}
        for node_type in NODE_TYPES:
            store = graph[node_type]
            if "x" not in store or store.x.ndim != 2:
                raise ValueError(
                    f"{node_type} nodes require a two-dimensional x feature tensor"
                )
            in_channels[node_type] = int(store.x.shape[1])
        return cls(in_channels=in_channels, **model_kwargs)

    @staticmethod
    def _with_reverse_relations(
        graph: HeteroData,
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        edge_indices = {}
        available_relations = set(graph.edge_types)
        for source, relation, destination in RELATION_TYPES:
            edge_type = (source, relation, destination)
            if edge_type not in available_relations:
                raise ValueError(f"Graph is missing relation store {edge_type}")

            edge_index = graph[edge_type].edge_index
            edge_indices[edge_type] = edge_index
            reverse_type = (destination, f"rev_{relation}", source)
            edge_indices[reverse_type] = edge_index.flip(0).contiguous()
        return edge_indices

    def forward(
        self,
        graph: HeteroData,
        herb_ids: torch.Tensor,
        drug_ids: torch.Tensor,
        return_pooling_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        """Return an inference-derived score for each herb-drug ID pair."""
        message_passing_edges = self._with_reverse_relations(graph)
        node_features = {
            node_type: self.input_projection[node_type](graph[node_type].x)
            for node_type in NODE_TYPES
        }

        gat_features = self.gat(node_features, message_passing_edges)
        gat_features = {
            node_type: F.dropout(
                F.elu(gat_features[node_type]),
                p=self.dropout,
                training=self.training,
            )
            for node_type in NODE_TYPES
        }

        if self.graphsage is not None:
            node_embeddings = self.graphsage(gat_features, message_passing_edges)
            node_embeddings = {
                node_type: F.relu(node_embeddings[node_type])
                for node_type in NODE_TYPES
            }
        else:
            node_embeddings = gat_features

        projection = None
        pooled_edge_index = None
        perm = None
        pooled_context = None
        if self.sag_pool is not None:
            projection = project_heterogeneous_graph(graph, node_embeddings)
            pooled_x, pooled_edge_index, _, pooled_batch, perm, _ = self.sag_pool(
                projection.x,
                projection.edge_index,
                batch=projection.batch,
            )
            pooled_graph_context = global_mean_pool(pooled_x, pooled_batch)
            pooled_node_type_ids = projection.node_type_ids[perm]
            typed_pool_contexts = []
            for type_id in range(len(NODE_TYPES)):
                type_mask = pooled_node_type_ids == type_id
                if torch.any(type_mask):
                    type_context = pooled_x[type_mask].mean(dim=0)
                else:
                    type_context = pooled_graph_context.new_zeros(
                        self.hidden_channels
                    )
                typed_pool_contexts.append(type_context)
            pooled_context = torch.cat(typed_pool_contexts, dim=0).unsqueeze(0)

        device = node_embeddings["Herb"].device
        herb_ids = torch.as_tensor(herb_ids, dtype=torch.long, device=device).view(-1)
        drug_ids = torch.as_tensor(drug_ids, dtype=torch.long, device=device).view(-1)
        if herb_ids.numel() != drug_ids.numel():
            raise ValueError("herb_ids and drug_ids must contain the same number of IDs")
        if herb_ids.numel() == 0:
            return torch.empty(
                (0,),
                dtype=node_embeddings["Herb"].dtype,
                device=device,
            )
        if torch.any(herb_ids < 0) or torch.any(
            herb_ids >= node_embeddings["Herb"].shape[0]
        ):
            raise IndexError("herb_ids contains an ID outside the Herb node store")
        if torch.any(drug_ids < 0) or torch.any(
            drug_ids >= node_embeddings["Drug"].shape[0]
        ):
            raise IndexError("drug_ids contains an ID outside the Drug node store")

        z_h = node_embeddings["Herb"][herb_ids]
        z_d = node_embeddings["Drug"][drug_ids]
        if pooled_context is not None and self.pooled_context_projection is not None:
            pair_context = pooled_context.expand(herb_ids.numel(), -1)
            z_h = z_h + self.pooled_context_projection["Herb"](pair_context)
            z_d = z_d + self.pooled_context_projection["Drug"](pair_context)
        pair_features = torch.cat([z_h, z_d], dim=1)
        model_output = self.interaction_mlp(pair_features)
        if self.num_outputs == 1:
            model_output = torch.sigmoid(model_output).squeeze(-1)
        if not return_pooling_details:
            return model_output

        if projection is None or perm is None or pooled_edge_index is None:
            return model_output, {
                "projection": None,
                "selected_projected_node_ids": torch.empty(
                    (0,), dtype=torch.long, device=device
                ),
                "selected_node_types": (),
                "selected_node_names": (),
            }

        selected_node_type_ids = projection.node_type_ids[perm]
        selected_local_ids = projection.node_local_ids[perm]
        selected_node_types = tuple(
            projection.node_types[int(node_id)] for node_id in perm.tolist()
        )
        selected_node_names = tuple(
            projection.node_names[int(node_id)] for node_id in perm.tolist()
        )
        pool_to_projection = torch.full(
            (projection.x.shape[0],),
            -1,
            dtype=torch.long,
            device=perm.device,
        )
        pool_to_projection[perm] = torch.arange(
            perm.numel(),
            dtype=torch.long,
            device=perm.device,
        )
        source_nodes = projection.edge_index[0]
        target_nodes = projection.edge_index[1]
        retained_edge_mask = (
            (pool_to_projection[source_nodes] >= 0)
            & (pool_to_projection[target_nodes] >= 0)
        )
        pooled_edge_relation_ids = projection.edge_relation_ids[retained_edge_mask]
        pooled_edge_direction_ids = projection.edge_direction_ids[retained_edge_mask]
        mapped_pooled_edge_index = pool_to_projection[
            projection.edge_index[:, retained_edge_mask]
        ]
        pooling_details = {
            "projection": projection,
            "selected_projected_node_ids": perm,
            "selected_node_type_ids": selected_node_type_ids,
            "selected_local_node_ids": selected_local_ids,
            "selected_node_types": selected_node_types,
            "selected_node_names": selected_node_names,
            "pooled_edge_index": mapped_pooled_edge_index,
            "sagpool_edge_index": pooled_edge_index,
            "pooled_edge_relation_ids": pooled_edge_relation_ids,
            "pooled_edge_direction_ids": pooled_edge_direction_ids,
        }
        return model_output, pooling_details

    def predict_score(
        self,
        graph: HeteroData,
        herb_ids: torch.Tensor,
        drug_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit inference alias for callers that prefer a named API."""
        return self.forward(graph, herb_ids, drug_ids)

    def predict_interaction(
        self,
        graph: HeterogeneousGraph,
        herb: str,
        drug: str,
        thresholds: PredictionThresholds | None = None,
    ) -> dict[str, str | float]:
        """Predict a named pair and return score plus configurable label.

        The score is the primary output. LOW/MEDIUM/HIGH labels are optional
        presentation bins and the default thresholds are not validated.
        """
        if self.num_outputs != 1:
            raise RuntimeError(
                "predict_interaction requires a scalar-output HGNN; "
                "ablation classifiers return class logits"
            )
        herb_key = normalize_entity_name(herb)
        drug_key = normalize_entity_name(drug)
        if herb_key is None or herb_key not in graph.node_to_id["Herb"]:
            raise ValueError(f"Unknown herb: {herb!r}")
        if drug_key is None or drug_key not in graph.node_to_id["Drug"]:
            raise ValueError(f"Unknown drug: {drug!r}")

        herb_id = graph.node_to_id["Herb"][herb_key]
        drug_id = graph.node_to_id["Drug"][drug_key]
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                score_tensor = self.predict_score(
                    graph.data,
                    torch.tensor([herb_id], dtype=torch.long),
                    torch.tensor([drug_id], dtype=torch.long),
                )
        finally:
            self.train(was_training)

        prediction_score = float(score_tensor.item())
        thresholds = thresholds or PredictionThresholds()
        return {
            "herb": graph.node_names["Herb"][herb_id],
            "drug": graph.node_names["Drug"][drug_id],
            "prediction_score": prediction_score,
            "prediction_label": thresholds.label(prediction_score),
        }