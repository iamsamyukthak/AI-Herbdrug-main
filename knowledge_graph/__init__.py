"""Heterogeneous herb-drug knowledge graph package."""

__all__ = [
    "HeterogeneousGraph",
    "NodeFeatureMetadata",
    "add_node_features",
    "build_heterogeneous_graph",
    "load_heterogeneous_graph",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from knowledge_graph import graph_builder

    return getattr(graph_builder, name)