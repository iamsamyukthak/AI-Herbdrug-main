"""Public heterogeneous graph construction API for the project."""

from pathlib import Path

from model.heterogeneous_graph import (
    EDGE_SOURCES,
    NODE_TYPES,
    EdgeType,
    HeterogeneousGraph,
    NodeType,
    load_heterogeneous_graph,
    normalize_entity_name,
)
from knowledge_graph.node_features import NodeFeatureMetadata, add_node_features


def build_heterogeneous_graph(
    data_dir: str | Path | None = None,
) -> HeterogeneousGraph:
    """Build a typed PyG graph with normalized, source-derived node features."""
    graph = load_heterogeneous_graph(data_dir)
    add_node_features(graph, data_dir=data_dir)
    return graph


def main() -> None:
    graph = build_heterogeneous_graph()
    print("Heterogeneous knowledge graph G = (V, E, R)")
    print("Node counts:")
    for node_type in NODE_TYPES:
        store = graph.data[node_type]
        print(
            f"  {node_type}: {store.num_nodes} nodes, "
            f"{store.x.shape[1]} features"
        )

    print("Relations (source records / unique directed edges):")
    for edge_type in graph.data.edge_types:
        records = graph.edge_records[edge_type]
        edge_count = graph.data[edge_type].edge_index.shape[1]
        print(f"  {edge_type}: {len(records)} / {edge_count}")
    print(f"Skipped rows with missing endpoints: {len(graph.skipped_rows)}")


__all__ = [
    "EDGE_SOURCES",
    "NODE_TYPES",
    "EdgeType",
    "HeterogeneousGraph",
    "NodeType",
    "NodeFeatureMetadata",
    "add_node_features",
    "build_heterogeneous_graph",
    "load_heterogeneous_graph",
    "normalize_entity_name",
]


if __name__ == "__main__":
    main()