"""Build sourced, normalized node features for the heterogeneous graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from model.heterogeneous_graph import HeterogeneousGraph, normalize_entity_name


@dataclass
class NodeFeatureMetadata:
    """Names and normalization statistics for attached per-type features."""

    feature_names: dict[str, tuple[str, ...]]
    numeric_normalization: dict[str, dict[str, dict[str, float]]]


def _category_features(
    graph: HeterogeneousGraph,
    data_dir: Path,
    node_type: str,
    filename: str,
    entity_column: str,
    category_columns: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode observed categorical metadata as deterministic multi-hot columns."""
    table = pd.read_csv(data_dir / filename)
    required = {entity_column, *category_columns}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{filename} is missing feature columns: {sorted(missing)}")

    categories_by_column: dict[str, tuple[str, ...]] = {}
    for column in category_columns:
        categories = {
            normalize_entity_name(value)
            for value in table[column]
            if normalize_entity_name(value) is not None
        }
        categories_by_column[column] = tuple(sorted(categories))

    names = tuple(
        f"{column}={category}"
        for column in category_columns
        for category in categories_by_column[column]
    )
    feature_indices = {
        (column, category): index
        for index, (column, category) in enumerate(
            (column, category)
            for column in category_columns
            for category in categories_by_column[column]
        )
    }

    matrix = np.zeros(
        (len(graph.node_names[node_type]), len(names)),
        dtype=np.float32,
    )
    node_ids = graph.node_to_id[node_type]

    for row in table.to_dict(orient="records"):
        node_key = normalize_entity_name(row[entity_column])
        node_id = node_ids.get(node_key) if node_key is not None else None
        if node_id is None:
            continue

        for column in category_columns:
            category = normalize_entity_name(row[column])
            if category is not None:
                matrix[node_id, feature_indices[(column, category)]] = 1.0

    return matrix, names


def _relation_degree(
    graph: HeterogeneousGraph,
    edge_type: tuple[str, str, str],
    node_type: str,
) -> np.ndarray:
    edge_index = graph.data[edge_type].edge_index
    node_position = 0 if edge_type[0] == node_type else 1
    node_ids = edge_index[node_position].cpu().numpy()
    return np.bincount(
        node_ids,
        minlength=len(graph.node_names[node_type]),
    ).astype(np.float32)


def _evidence_record_count(
    graph: HeterogeneousGraph,
    edge_type: tuple[str, str, str],
    node_type: str,
    source_file: str = "datasets/external_bio_evidence.csv",
) -> np.ndarray:
    records = graph.edge_records.get(edge_type)
    counts = np.zeros(len(graph.node_names[node_type]), dtype=np.float32)
    if records is None or records.empty:
        return counts

    records = records[records["source_file"] == source_file]
    edge_node_column = "source_id" if edge_type[0] == node_type else "target_id"
    if records.empty:
        return counts

    values = records[edge_node_column].value_counts()
    counts[values.index.to_numpy(dtype=np.int64)] = values.to_numpy(dtype=np.float32)
    return counts


def _normalize_counts(
    values: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    logged = np.log1p(values.astype(np.float64))
    means = logged.mean(axis=0)
    scales = logged.std(axis=0)
    scales[scales < 1e-8] = 1.0
    normalized = ((logged - means) / scales).astype(np.float32)
    statistics = {
        name: {
            "log1p_mean": float(means[index]),
            "log1p_scale": float(scales[index]),
        }
        for index, name in enumerate(feature_names)
    }
    return normalized, statistics


def add_node_features(
    graph: HeterogeneousGraph,
    data_dir: str | Path | None = None,
    ligand_dir: str | Path | None = None,
) -> NodeFeatureMetadata:
    """Attach numerical ``x`` tensors using only observed data and metadata.

    Count features use ``log1p`` followed by per-node-type z-score scaling.
    Categorical fields use deterministic multi-hot encoding. Molecular
    descriptors are not synthesized; only exact ligand-file availability is
    exposed for compounds with a matching existing SDF/PDBQT filename.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "datasets"
    else:
        data_dir = Path(data_dir)

    if ligand_dir is None:
        ligand_dir = Path(__file__).resolve().parents[1] / "docking" / "ligands"
    else:
        ligand_dir = Path(ligand_dir)

    herb_compound = ("Herb", "contains", "Compound")
    compound_protein = ("Compound", "associated_with", "Protein")
    drug_protein = ("Drug", "targets", "Protein")
    herb_drug = ("Herb", "interacts_with", "Drug")

    numeric_values: dict[str, np.ndarray] = {
        "Herb": np.column_stack(
            [
                _relation_degree(graph, herb_compound, "Herb"),
                _evidence_record_count(graph, herb_drug, "Herb"),
            ]
        ),
        "Compound": np.column_stack(
            [
                _relation_degree(graph, herb_compound, "Compound"),
                _relation_degree(graph, compound_protein, "Compound"),
                _evidence_record_count(graph, compound_protein, "Compound"),
            ]
        ),
        "Drug": np.column_stack(
            [
                _relation_degree(graph, drug_protein, "Drug"),
                _evidence_record_count(graph, herb_drug, "Drug"),
            ]
        ),
        "Protein": np.column_stack(
            [
                _relation_degree(graph, compound_protein, "Protein"),
                _relation_degree(graph, drug_protein, "Protein"),
            ]
        ),
    }
    numeric_names = {
        "Herb": (
            "log1p_unique_compound_degree",
            "log1p_external_evidence_rows",
        ),
        "Compound": (
            "log1p_herb_degree",
            "log1p_protein_degree",
            "log1p_external_evidence_rows",
        ),
        "Drug": (
            "log1p_explicit_target_degree",
            "log1p_external_evidence_rows",
        ),
        "Protein": (
            "log1p_compound_degree",
            "log1p_drug_target_degree",
        ),
    }

    categorical_config: dict[str, tuple[str, str, tuple[str, ...]]] = {
        "Herb": ("herbs.csv", "Herb", ("Scientific_Name",)),
        "Drug": ("drugs.csv", "Drug", ("Drug_Class",)),
        "Protein": ("proteins.csv", "Protein", ("Gene", "Pathway")),
    }

    categorical_values: dict[str, np.ndarray] = {}
    categorical_names: dict[str, tuple[str, ...]] = {}
    for node_type, (filename, entity_column, category_columns) in categorical_config.items():
        categorical_values[node_type], categorical_names[node_type] = _category_features(
            graph,
            data_dir,
            node_type,
            filename,
            entity_column,
            category_columns,
        )

    structure_names = {
        normalize_entity_name(path.stem)
        for path in ligand_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".sdf", ".pdbqt"}
    } if ligand_dir.exists() else set()
    compound_structure_flag = np.asarray(
        [
            [float(name_key in structure_names)]
            for name_key in graph.node_to_id["Compound"]
        ],
        dtype=np.float32,
    )
    categorical_values["Compound"] = compound_structure_flag
    categorical_names["Compound"] = ("has_exact_ligand_structure_file",)

    all_feature_names: dict[str, tuple[str, ...]] = {}
    normalization: dict[str, dict[str, dict[str, float]]] = {}
    for node_type in ("Herb", "Compound", "Drug", "Protein"):
        normalized_counts, numeric_statistics = _normalize_counts(
            numeric_values[node_type],
            numeric_names[node_type],
        )
        categories = categorical_values.get(
            node_type,
            np.zeros((len(graph.node_names[node_type]), 0), dtype=np.float32),
        )
        graph.data[node_type].x = torch.from_numpy(
            np.concatenate([normalized_counts, categories], axis=1).astype(
                np.float32,
                copy=False,
            )
        )
        names = numeric_names[node_type] + categorical_names.get(node_type, ())
        graph.data[node_type].feature_names = list(names)
        all_feature_names[node_type] = names
        normalization[node_type] = numeric_statistics

    graph.data.feature_schema = all_feature_names
    graph.data.feature_normalization = normalization
    return NodeFeatureMetadata(
        feature_names=all_feature_names,
        numeric_normalization=normalization,
    )