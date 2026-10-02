"""Build a provenance-preserving heterogeneous herb-drug knowledge graph."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import HeteroData


NodeType = str
EdgeType = tuple[str, str, str]

NODE_TYPES = ("Herb", "Compound", "Drug", "Protein")

EDGE_SOURCES: tuple[tuple[str, EdgeType, str, str], ...] = (
    ("herbs.csv", ("Herb", "contains", "Compound"), "Herb", "Active_Compound"),
    (
        "herbs_ddid_edges.csv",
        ("Herb", "contains", "Compound"),
        "Herb",
        "Active_Compound",
    ),
    (
        "external_bio_evidence.csv",
        ("Herb", "contains", "Compound"),
        "Herb",
        "Compound",
    ),
    (
        "compound_protein.csv",
        ("Compound", "associated_with", "Protein"),
        "Compound",
        "Protein",
    ),
    (
        "external_bio_evidence.csv",
        ("Compound", "associated_with", "Protein"),
        "Compound",
        "Target",
    ),
    (
        "drugs.csv",
        ("Drug", "targets", "Protein"),
        "Drug",
        "Target_Protein",
    ),
    (
        "drugs_ddid_edges.csv",
        ("Drug", "targets", "Protein"),
        "Drug",
        "Target_Protein",
    ),
    (
        "ddid_processed.csv",
        ("Herb", "interacts_with", "Drug"),
        "Herb",
        "Drug",
    ),
    (
        "ddid_herb_drug_training.csv",
        ("Herb", "interacts_with", "Drug"),
        "Herb",
        "Drug",
    ),
    (
        "interactions.csv",
        ("Herb", "interacts_with", "Drug"),
        "Herb",
        "Drug",
    ),
    (
        "external_bio_evidence.csv",
        ("Herb", "interacts_with", "Drug"),
        "Herb",
        "Drug",
    ),
)


@dataclass
class HeterogeneousGraph:
    """PyG graph plus normalized names and source-row provenance."""

    data: HeteroData
    node_names: dict[NodeType, tuple[str, ...]]
    node_to_id: dict[NodeType, dict[str, int]]
    node_records: dict[NodeType, pd.DataFrame]
    edge_records: dict[EdgeType, pd.DataFrame]
    skipped_rows: tuple[dict[str, Any], ...]


def normalize_entity_name(value: Any) -> str | None:
    """Normalize a name for identity matching without fuzzy aliasing."""
    if pd.isna(value):
        return None

    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    if not normalized:
        return None
    return normalized.casefold()


def _display_name(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def load_heterogeneous_graph(
    data_dir: str | Path | None = None,
) -> HeterogeneousGraph:
    """Load observed CSV relationships into a normalized PyG HeteroData graph.

    Graph edges are unique by normalized source/target IDs. Every source row,
    including duplicate evidence rows, remains available in ``edge_records``.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "datasets"
    else:
        data_dir = Path(data_dir)

    tables: dict[str, pd.DataFrame] = {}
    for filename, _, source_column, target_column in EDGE_SOURCES:
        if filename not in tables:
            tables[filename] = pd.read_csv(data_dir / filename)
        table = tables[filename]
        missing_columns = {
            source_column,
            target_column,
        } - set(table.columns)
        if missing_columns:
            raise ValueError(
                f"{filename} is missing relationship columns: "
                f"{sorted(missing_columns)}"
            )

    protein_table = pd.read_csv(data_dir / "proteins.csv")
    if "Protein" not in protein_table.columns:
        raise ValueError("proteins.csv is missing the Protein column")
    tables["proteins.csv"] = protein_table

    name_candidates: dict[NodeType, dict[str, set[str]]] = {
        node_type: {} for node_type in NODE_TYPES
    }
    node_observations: dict[NodeType, list[dict[str, Any]]] = {
        node_type: [] for node_type in NODE_TYPES
    }
    raw_edge_records: dict[EdgeType, list[dict[str, Any]]] = {}
    skipped_rows: list[dict[str, Any]] = []

    def observe_node(
        node_type: NodeType,
        value: Any,
        filename: str,
        source_row: int,
        source_column: str,
    ) -> str | None:
        name_key = normalize_entity_name(value)
        if name_key is None:
            return None

        display_name = _display_name(value)
        name_candidates[node_type].setdefault(name_key, set()).add(display_name)
        node_observations[node_type].append(
            {
                "name_key": name_key,
                "display_name": display_name,
                "source_file": f"datasets/{filename}",
                "source_row": source_row,
                "source_column": source_column,
                "source_value": value,
            }
        )
        return name_key

    for row_number, row in enumerate(
        protein_table.to_dict(orient="records"),
        start=2,
    ):
        observe_node(
            "Protein",
            row["Protein"],
            "proteins.csv",
            row_number,
            "Protein",
        )

    for filename, edge_type, source_column, target_column in EDGE_SOURCES:
        source_type, _, target_type = edge_type
        table = tables[filename]

        for row_number, row in enumerate(
            table.to_dict(orient="records"),
            start=2,
        ):
            source_key = observe_node(
                source_type,
                row[source_column],
                filename,
                row_number,
                source_column,
            )
            target_key = observe_node(
                target_type,
                row[target_column],
                filename,
                row_number,
                target_column,
            )

            if source_key is None or target_key is None:
                skipped_rows.append(
                    {
                        "edge_type": edge_type,
                        "source_file": f"datasets/{filename}",
                        "source_row": row_number,
                        "reason": "missing or blank relationship endpoint",
                        "source_values": row,
                    }
                )
                continue

            raw_edge_records.setdefault(edge_type, []).append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "source_file": f"datasets/{filename}",
                    "source_row": row_number,
                    "source_values": row,
                }
            )

    node_names: dict[NodeType, tuple[str, ...]] = {}
    node_to_id: dict[NodeType, dict[str, int]] = {}
    node_records: dict[NodeType, pd.DataFrame] = {}
    graph = HeteroData()

    for node_type in NODE_TYPES:
        candidates = name_candidates[node_type]
        normalized_names = sorted(candidates)
        names = tuple(
            min(candidates[name_key], key=lambda name: (name.casefold(), name))
            for name_key in normalized_names
        )
        ids = {
            name_key: index
            for index, name_key in enumerate(normalized_names)
        }

        node_names[node_type] = names
        node_to_id[node_type] = ids
        graph[node_type].num_nodes = len(names)
        graph[node_type].name = list(names)

        node_records[node_type] = pd.DataFrame(
            [
                {
                    "node_id": ids[observation["name_key"]],
                    **observation,
                }
                for observation in node_observations[node_type]
            ],
            columns=[
                "node_id",
                "name_key",
                "display_name",
                "source_file",
                "source_row",
                "source_column",
                "source_value",
            ],
        )

    edge_records: dict[EdgeType, pd.DataFrame] = {}
    for edge_type in sorted(raw_edge_records):
        source_type, _, target_type = edge_type
        records = raw_edge_records[edge_type]
        edge_frame = pd.DataFrame(records)
        edge_frame["source_id"] = edge_frame["source_key"].map(
            node_to_id[source_type]
        )
        edge_frame["target_id"] = edge_frame["target_key"].map(
            node_to_id[target_type]
        )
        edge_records[edge_type] = edge_frame[
            [
                "source_id",
                "target_id",
                "source_key",
                "target_key",
                "source_file",
                "source_row",
                "source_values",
            ]
        ].reset_index(drop=True)

        unique_edges = (
            edge_frame[["source_id", "target_id"]]
            .drop_duplicates()
            .sort_values(["source_id", "target_id"])
        )
        if unique_edges.empty:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(
                unique_edges.to_numpy().T,
                dtype=torch.long,
            )
        graph[edge_type].edge_index = edge_index

    return HeterogeneousGraph(
        data=graph,
        node_names=node_names,
        node_to_id=node_to_id,
        node_records=node_records,
        edge_records=edge_records,
        skipped_rows=tuple(skipped_rows),
    )


if __name__ == "__main__":
    heterogeneous_graph = load_heterogeneous_graph()
    print("Node counts:")
    for node_type in NODE_TYPES:
        print(f"  {node_type}: {len(heterogeneous_graph.node_names[node_type])}")

    print("Observed edge records and unique graph edges:")
    for edge_type in heterogeneous_graph.data.edge_types:
        record_count = len(heterogeneous_graph.edge_records[edge_type])
        unique_count = heterogeneous_graph.data[edge_type].edge_index.shape[1]
        print(f"  {edge_type}: {record_count} records, {unique_count} edges")
    print(f"Skipped relationship rows: {len(heterogeneous_graph.skipped_rows)}")